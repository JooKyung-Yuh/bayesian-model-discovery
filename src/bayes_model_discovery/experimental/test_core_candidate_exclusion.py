"""Synthetic candidate-policy checks; no stored responses or paid calls."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from . import core_count as core
from .test_core_d0_policy import _points_with_authentic_identity, _synthetic_real_fits
from .test_count_proposer import run_args, stub_fits

IDENTITY_FIELDS = ("source_files_sha256", "created_utc", "output", "manifest_ref", "python")
CEP_ADDED_KEYS = ("candidate_exclusion_policy",)


def _run(output, monkeypatch, *, structure="gauss_pow", policy=None, rounds=2):
    real = structure.startswith("gauss_")
    _synthetic_real_fits(monkeypatch) if real else stub_fits(monkeypatch)
    options = {} if policy is None else {"candidate_exclusion_policy": policy}
    core.run(run_args(output, profile="dev", rounds=rounds, seed=31, cell="000",
                      structure=structure, truth_family="normal" if real else "beta_binomial",
                      initial_family="normal" if real else "binomial", d0_policy="stratified4",
                      **options))
    return json.loads((output / "config.json").read_text())


@pytest.mark.parametrize("policy", [None, "none", "observed"])
def test_parser_uses_one_suppressed_config_key(tmp_path, monkeypatch, policy):
    """Check that parser uses one suppressed config key."""
    seen = []
    argv = ["core_count", "--output", str(tmp_path / "not_run")]
    if policy is not None:
        argv.extend(["--candidate-exclusion", policy])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(core, "run", lambda args: seen.append(vars(args)))
    core.main()
    assert len(seen) == 1 and "candidate_exclusion" not in seen[0]
    assert ("candidate_exclusion_policy" in seen[0]) == (policy is not None)
    if policy is not None:
        assert seen[0]["candidate_exclusion_policy"] == policy
    assert core.CEP_ADDED_KEYS == CEP_ADDED_KEYS


def test_unknown_policy_is_refused_before_output(tmp_path, monkeypatch):
    """Check that unknown policy is refused before output."""
    output = tmp_path / "absent"
    monkeypatch.setattr(sys, "argv", ["core_count", "--output", str(output),
                                     "--candidate-exclusion", "unknown"])
    with pytest.raises(SystemExit) as stopped:
        core.main()
    assert stopped.value.code == 2 and not output.exists()
    with pytest.raises(ValueError, match="candidate_exclusion_policy"):
        core.run(run_args(output, candidate_exclusion_policy="unknown"))
    assert not output.exists()


@pytest.mark.parametrize("structure,policy", [
    ("gauss_pow", "observed"), ("out", "none"), ("out", "observed"),
])
def test_explicit_legacy_policy_preserves_science_and_adds_only_declared_metadata(
        tmp_path, monkeypatch, structure, policy):
    """Check that explicit legacy policy preserves science and adds only declared metadata."""
    outputs = [tmp_path / "omitted" / "cell", tmp_path / "explicit" / "cell"]
    if structure == "out" and policy == "observed":
        # Refusal rule: a declared-but-inert count exclusion is refused, not recorded.
        monkeypatch.setattr(core, "bmd_label_receipt", lambda *a, **k: pytest.fail(
            "refusal must precede label or cost reservation"))
        with pytest.raises(ValueError, match="count world forbids.*observed"):
            _run(outputs[1], monkeypatch, structure=structure, policy=policy)
        assert not outputs[1].parent.exists()
        return
    before = _run(outputs[0], monkeypatch, structure=structure)
    after = _run(outputs[1], monkeypatch, structure=structure, policy=policy)
    assert not set(CEP_ADDED_KEYS) & set(before)
    assert set(after) - set(before) == set(CEP_ADDED_KEYS)
    assert after["candidate_exclusion_policy"] == policy
    assert {key: value for key, value in before.items() if key not in IDENTITY_FIELDS} == {
        key: value for key, value in after.items() if key not in IDENTITY_FIELDS + CEP_ADDED_KEYS}
    for name in ("archive.json", "observations.jsonl", "scores.jsonl"):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes(), name
    for left in sorted((outputs[0] / "acquisition").glob("*.json")):
        original = json.loads(left.read_text())
        declared = json.loads((outputs[1] / "acquisition" / left.name).read_text())
        assert declared.pop("candidate_exclusion_policy") == policy
        assert declared == original
    source = Path(__file__).resolve().parents[3]
    assert _points_with_authentic_identity(outputs[0], before, source) == (
        _points_with_authentic_identity(outputs[1], after, source))


def test_none_completes_seventeen_rounds_with_a_real_reobservation(tmp_path, monkeypatch):
    """Check that none completes seventeen rounds with a real reobservation."""
    output = tmp_path / "synthetic_repeats"
    config = _run(output, monkeypatch, policy="none", rounds=17)
    rows = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    acquired = [row for row in rows if row["round"] > 0]
    assert len(acquired) == 17 and config["candidate_exclusion_policy"] == "none"
    by_design = defaultdict(list)
    for row in acquired:
        by_design[row["u"]].append(row)
    repeats = [group for group in by_design.values() if len(group) > 1]
    assert repeats
    world = core.CoreWorld(31, "gauss_pow", "normal")
    for group in repeats:
        first, second = group[:2]
        assert first["y"] != second["y"]
        assert first["observation_seed"] == core.seed_for(
            "observation", 31, "gauss_pow", "normal", float(first["u"]).hex(), 0)
        assert second["observation_seed"] == core.seed_for(
            "observation", 31, "gauss_pow", "normal", float(second["u"]).hex(), 1)
        assert world.observe(first["u"])["y"] == first["y"]
        assert world.observe(first["u"])["y"] == second["y"]
    expected = core.candidate_set_record(np.linspace(0, 1, 16), "core")
    records = sorted((output / "acquisition").glob("*.json"))
    assert len(records) == 17
    for path in records:
        record = json.loads(path.read_text())
        assert record["candidate_set_sha256"] == expected["candidate_set_sha256"]
        assert record["candidate_exclusion_policy"] == "none"


def test_explicit_observed_retains_the_seventeenth_round_exhaustion(tmp_path, monkeypatch):
    """Check that explicit observed retains the seventeenth round exhaustion."""
    output = tmp_path / "explicit_exhaustion"
    with pytest.raises(ValueError, match="Gaussian CORE design pool exhausted: profile=dev, grid_size=16"):
        _run(output, monkeypatch, policy="observed", rounds=17)
    rows = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    acquired = [row["u"] for row in rows if row["round"] > 0]
    assert len(acquired) == len(set(acquired)) == 16
    assert sorted(acquired) == np.linspace(0, 1, 16).tolist()


