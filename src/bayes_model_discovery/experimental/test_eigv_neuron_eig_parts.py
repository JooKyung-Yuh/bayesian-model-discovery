"""Model expected information gain (EIG) and Parameter EIG as opt-in NeuronBench arms on an A=1 cell.

`vector_inference.joint_eig(..., objective=...)` splits the joint estimate into its two chain-rule
parts on the same draws; the default (`objective="eig"`) returns the rows it always returned.
"""
import argparse

import numpy as np
import pytest

from . import neuron_core as nc
from .vector_inference import EIG_PART_OBJECTIVES, joint_eig

BASE_ROW_KEYS = {"design_index", "eig", "mc_standard_error", "samples"}


def _bank(per_model=(6, 5), designs=4, features=3, feature_sigmas=False, model_logweights=(0.0, -0.7), seed=0):
    rng = np.random.default_rng(seed)
    means, sigmas, logweights, indices = [], [], [], []
    weights = np.asarray(model_logweights) - np.logaddexp.reduce(model_logweights)
    for m, (n, w) in enumerate(zip(per_model, weights, strict=True)):
        centre = rng.normal(0, 2, size=(designs, features))
        means.append(centre[None] + rng.normal(0, 1, size=(n, designs, features)))
        sigmas.append(rng.uniform(0.5, 1.5, size=(n, features) if feature_sigmas else n))
        logweights += [w - np.log(n)] * n
        indices += [m] * n
    return {"means": np.concatenate(means), "sigmas": np.concatenate(sigmas),
            "logweights": np.asarray(logweights), "model_indices": np.asarray(indices)}


@pytest.mark.parametrize("feature_sigmas", (False, True))
def test_default_rows_are_unchanged_and_the_parts_sum_to_the_joint(feature_sigmas):
    """Default rows are unchanged and the parts sum to the joint estimate."""
    bank = _bank(feature_sigmas=feature_sigmas)
    chosen, rows = joint_eig(bank, 7, 256)
    assert all(set(row) == BASE_ROW_KEYS for row in rows)
    for objective in EIG_PART_OBJECTIVES:
        part_chosen, part_rows = joint_eig(bank, 7, 256, objective=objective)
        for row, part in zip(rows, part_rows, strict=True):
            assert {k: part[k] for k in BASE_ROW_KEYS} == row
            assert abs(part["structure_eig"] + part["parameter_eig"] - part["eig"]) <= 1e-12
            assert part["parameter_eig"] > 0 and part["structure_eig"] > -3 * part["structure_mc_standard_error"]
        best = min(part_rows, key=lambda r: (-r[objective], r["design_index"]))["design_index"]
        assert part_chosen == best
    assert chosen == min(rows, key=lambda r: (-r["eig"], r["design_index"]))["design_index"]


def test_one_model_has_no_structure_term_and_point_banks_no_parameter_term():
    """One model has no structure term and point banks have no parameter term."""
    one = _bank(per_model=(7,), model_logweights=(0.0,))
    _, rows = joint_eig(one, 3, 128, objective="structure_eig")
    for row in rows:
        assert abs(row["structure_eig"]) <= 1e-12 and abs(row["parameter_eig"] - row["eig"]) <= 1e-12
    points = _bank(per_model=(1, 1, 1), model_logweights=(0.0, -0.3, -1.0))
    _, rows = joint_eig(points, 3, 128, objective="parameter_eig")
    for row in rows:
        assert abs(row["parameter_eig"]) <= 1e-12 and abs(row["structure_eig"] - row["eig"]) <= 1e-12


def test_unknown_objective_is_refused():
    """An unknown EIG objective is refused."""
    with pytest.raises(ValueError, match="unknown EIG objective"):
        joint_eig(_bank(), 3, 16, objective="epig")


def test_the_option_is_suppressed_unless_stated():
    """The design-policy option is suppressed unless stated."""
    unstated = nc.build_parser().parse_args(["--output", "x"])
    assert "design_policy" not in vars(unstated)
    stated = nc.build_parser().parse_args(["--output", "x", "--design-policy", "structure_eig"])
    assert stated.design_policy == "structure_eig"
    assert {k: v for k, v in vars(stated).items() if k != "design_policy"} == vars(unstated)
    with pytest.raises(SystemExit):
        nc.build_parser().parse_args(["--output", "x", "--design-policy", "joint_eig"])


@pytest.mark.parametrize("policy", EIG_PART_OBJECTIVES)
@pytest.mark.parametrize("cell", ("110", "112", "010", "000"))
def test_the_arms_are_refused_off_the_a1_cells_before_any_output(tmp_path, cell, policy):
    """The arms are refused off the A=1 cells before any output exists."""
    out = tmp_path / "never"
    args = nc.build_parser().parse_args(["--output", str(out), "--cell", cell, "--design-policy", policy])
    with pytest.raises(ValueError, match="needs an A=1 cell"):
        nc.run(args)
    assert not out.exists()


class _Reached(Exception):
    pass


@pytest.mark.parametrize("policy", EIG_PART_OBJECTIVES)
def test_the_arms_pass_the_check_on_an_a1_cell(tmp_path, monkeypatch, policy):
    """The arms pass the check on an A=1 cell."""
    out = tmp_path / "never"
    args = nc.build_parser().parse_args(["--output", str(out), "--cell", "111", "--design-policy", policy])

    def reached(*_):
        """Replacement that raises to mark that the check was passed."""
        raise _Reached

    monkeypatch.setattr(nc, "resolve_particles", reached)  # the next call after the check
    with pytest.raises(_Reached):
        nc.run(args)
    assert not out.exists()


def test_points_variant_marks_only_the_two_arms():
    """The points variant marks only the two part arms."""
    assert nc.EIG_PART_VARIANT == {"structure_eig": "design_structure_eig", "parameter_eig": "design_parameter_eig"}
    assert set(nc.EIG_PART_DEFINITION) == set(EIG_PART_OBJECTIVES) == set(nc.EIG_PART_VARIANT)
    assert isinstance(argparse.SUPPRESS, str)
