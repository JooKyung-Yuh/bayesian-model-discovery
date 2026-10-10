"""Synthetic initial design set (D0) execution and byte-preservation checks; no provider calls."""
import json

import numpy as np
import pytest

from bayes_model_discovery.experimental import chembench_core as cc
from bayes_model_discovery.experimental import test_chembench_core as old
from bayes_model_discovery.experimental.count_inference import Fit

CONDITIONS = {"C_I": 0, "C_B": 1, "C_P": 0, "Enz": 1, "T": 300, "pH": 7}
# Synthetic test values only; none are claimed to be native default conditions.


def stub_fits(monkeypatch):
    """Patch the adapter with a fake session and a stub fit that returns synthetic particles."""
    monkeypatch.setattr(old.real_adapter, "ChemBenchSession", old._FakeSession)
    monkeypatch.setattr(old.real_adapter, "sample_designs", old._fake_sample_designs)
    def fit(hypothesis, x, y, seed, transform, response_unit, settings, **kwargs):
        """Stub fit: return a synthetic Fit without running inference."""
        particles = {name: np.full(16, 0.1 if name != "b0" else 0.0)
                     for name, role in hypothesis.parameters}
        particles.update({name: np.full(16, 0.4 if name == "sigma" else 2.0)
                          for name in hypothesis.noise_parameters})
        return Fit(hypothesis, particles, -float(len(hypothesis.parameters)), [-1.0, -1.0],
                   {"input_transform": {"center": transform.center, "scale": transform.scale},
                    "interface": kwargs.get("interface"), "response_unit": float(response_unit)}, 0.0)
    monkeypatch.setattr(cc, "fit", fit)


@pytest.mark.usefixtures("require_autoscilab_source")
def test_fixed3_preserves_every_supplied_condition_and_concentration():
    """Check that fixed3 preserves every supplied condition and concentration."""
    rows = cc.fixed3_designs(CONDITIONS)
    assert [row["C_A"] for row in rows] == [0.03, 0.3, 3.0]
    assert all({key: row[key] for key in CONDITIONS} == CONDITIONS for row in rows)


@pytest.mark.parametrize("conditions", [[], {**CONDITIONS, "C_A": 1},
                                        {**CONDITIONS, "T": 0}, {**CONDITIONS, "Enz": float("nan")}])
@pytest.mark.usefixtures("require_autoscilab_source")
def test_fixed3_never_invents_native_defaults(conditions):
    """Check that fixed3 never invents native defaults."""
    with pytest.raises(ValueError):
        cc.fixed3_designs(conditions)


@pytest.mark.usefixtures("require_autoscilab_source")
def test_declared_rule_uses_bounds_reference_temperature_and_neutrality():
    """Check that declared rule uses bounds reference temperature and neutrality."""
    from autoscilab.oracle.chembench import _T_REF, CHEM_INPUT_BOUNDS
    values = cc.fixed3_conditions()
    assert values["C_I"] == values["C_P"] == 0
    assert values["T"] == _T_REF == 310.0
    assert values["pH"] == 7.0
    assert values["C_B"] == np.sqrt(0.01 * 100)
    assert values["Enz"] == np.sqrt(0.01 * 10)
    for row in cc.fixed3_designs():
        assert set(row) == set(CHEM_INPUT_BOUNDS)
        assert all(CHEM_INPUT_BOUNDS[name][0] <= value <= CHEM_INPUT_BOUNDS[name][1]
                   for name, value in row.items())
    changed = cc.fixed3_conditions({"pH": 6.5})
    assert changed == {**values, "pH": 6.5}


def test_fixed3_changes_d0_count_but_keeps_future_random_stream(tmp_path, monkeypatch):
    """Check that fixed3 changes d0 count but keeps future random stream."""
    stub_fits(monkeypatch)
    # The execution fixture's own named coordinates; native seven-value handling
    # is checked independently above, never substituted in production.
    initial = old._fake_sample_designs(3, 159)
    monkeypatch.setattr(cc, "fixed3_designs", lambda conditions: initial)
    output = tmp_path / "run"
    cc.run(old._args(output, old._manifest(tmp_path), d0_policy="fixed3", d0_fixed_inputs=CONDITIONS))
    rows = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert [row["x"] for row in rows[:3]] == initial
    assert rows[-1]["x"] == old._fake_sample_designs(5, cc.seed_for("chem_observation_designs", 11))[4]
    config = json.loads((output / "config.json").read_text())
    assert config["d0_count"] == 3 and config["d0_conditions_source"] == "our_declared_values_mda_undisclosed"
    import csv
    points = list(csv.DictReader((output / "points.csv").open()))
    assert len(points[0]) == 25
    assert [int(row["x"]) for row in points] == [3, 4]


