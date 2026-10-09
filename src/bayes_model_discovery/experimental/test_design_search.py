"""Synthetic design-search contracts, never stored research responses."""

import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from . import design_search as search
from .core_count import seed_for


def box():
    """Return a two-dimensional unit-box search transform with no log inputs."""
    return search.SearchTransform(("x", "y"), ((0.0, 1.0), (0.0, 1.0)), ())


def quadratic(points):
    """Smooth synthetic objective over the unit square, peaking at (0.62, 0.31) with height 1."""
    return 1.0 - 2.0 * (points[:, 0] - 0.62) ** 2 - 0.8 * (points[:, 1] - 0.31) ** 2


def noisy_objective(noise=0.15, flat=False):
    """Build a seed-keyed noisy objective: noise level (and optionally a flat surface) in, evaluator out."""
    def evaluate(points, seed):
        """Evaluate the synthetic objective at the given points with point-keyed noise for this seed."""
        values = np.ones(len(points)) if flat else quadratic(points)
        # Synthetic fixed-seed, point-keyed error; 15% of the unit-height objective.
        perturbations = [np.random.default_rng(seed_for("synthetic_search_error", seed, point.tolist()))
                         .normal(0.0, noise) for point in points]
        return [{"design": point.tolist(), "eig": float(value + error),
                 "mc_standard_error": noise}
                for point, value, error in zip(points, values, perturbations, strict=True)]
    return evaluate


def run_search(seed=11, objective=None, **options):
    """Run one projected SPSA search on the synthetic box with the given seed and objective; return its record."""
    return search.projected_spsa(
        objective or noisy_objective(), box(), global_seed=seed, cell="001", round_index=1,
        acquisition_seed=seed_for("synthetic_acquisition", seed, 1), **options,
        acquisition_stream="synthetic_acquisition",
    )


@pytest.mark.parametrize("values", [
    {"alpha": 0.1, "gamma": 0.1}, {"alpha": 0.1, "gamma": 0.2},
    {"a": 0}, {"A": -1}, {"c": -1}, {"W": 0}, {"W": 1.5}, {"W": True},
    {"kappa": float("nan")}, {"k_sigma": float("inf")}, {"a": True}, {"unknown": 1},
])
def test_invalid_search_gains_are_refused(values):
    """Invalid gain values are refused with a design-search error."""
    with pytest.raises(ValueError, match="design-search"):
        search.validate_parameters(values)


def test_default_gains_are_the_declared_mock_tuning_start():
    """The default gains equal the declared mock tuning start."""
    assert search.validate_parameters() == {
        "a": 0.2, "A": 10.0, "alpha": 0.602, "gamma": 0.101,
        "c": 0.05, "W": 10, "kappa": 1.0, "k_sigma": 3.0,
    }


def test_noise_floor_slots_match_reference_all_pairs_and_ddof_zero():
    """Noise-floor estimates match the reference all-pairs and ddof=0 formulas."""
    a_values = np.linspace(-0.5, 1.3, 10)
    b_values = np.array([0.3, 0.4, -0.1, 0.8, -0.5, 0.6, 0.9, 0.2, -0.4, 0.7])
    differences = a_values - b_values
    pairs = list(itertools.combinations(range(10), 2))
    assert len(pairs) == 45
    pair_delta = np.array([(a_values[i] - a_values[j]) - (b_values[i] - b_values[j])
                           for i, j in pairs])
    actual = search.noise_floor_estimates(a_values, b_values)
    assert actual == {
        "rms": float(np.sqrt(np.mean(differences ** 2)) / np.sqrt(2)),
        "centered": float(np.std(differences - differences.mean(), ddof=0) / np.sqrt(2)),
        "pairwise": float(np.sqrt(np.mean(pair_delta ** 2)) / np.sqrt(2)),
        "centered_times_sqrt2": float(
            np.std(differences - differences.mean(), ddof=0) / np.sqrt(2) * np.sqrt(2)),
    }
    assert search.DEFAULT_NOISE_FLOOR_METHOD == "pairwise"
    assert search.NOISE_FLOOR_METHODS == ("rms", "centered", "pairwise")


@pytest.mark.parametrize("values", [[], [0.0] * 9, [float("nan")] * 10])
def test_noise_floor_refuses_missing_or_nonfinite_check_values(values):
    """The noise floor refuses empty, short or non-finite check values."""
    with pytest.raises(ValueError, match="design-search"):
        search.noise_floor_estimates(values, [0.0] * 10)


def test_zero_noise_estimates_are_finite_without_offset_halt_rule():
    """Identical check values give finite zero noise estimates without an offset halt rule."""
    values = np.arange(10, dtype=float)
    assert search.noise_floor_estimates(values, values) == {
        "rms": 0.0, "centered": 0.0, "pairwise": 0.0, "centered_times_sqrt2": 0.0,
    }


def test_search_only_log_transform_preserves_every_raw_corner():
    """The search-only log transform maps every raw corner to a unit corner and back exactly."""
    names = ("linear", "log_small", "log_large")
    bounds = {"linear": (-2.0, 4.0), "log_small": (0.001, 3.0), "log_large": (1.0, 100.0)}
    transform = search.SearchTransform.from_bounds(names, bounds, ("log_small", "log_large"))
    corners = np.array(list(itertools.product(*[bounds[name] for name in names])))
    unit = transform.forward(corners)
    assert set(unit.ravel()) == {0.0, 1.0}
    np.testing.assert_array_equal(transform.inverse(unit), corners)
    middle = transform.inverse(np.full(3, 0.5))
    np.testing.assert_allclose(middle, [1.0, np.sqrt(0.003), 10.0])
    assert transform.record()["distinct_from_fitting_InputTransform"] is True


@pytest.mark.usefixtures("require_autoscilab_source")
def test_native_chembench_bounds_and_logged_inputs_round_trip_all_corners():
    """Native ChemBench bounds and logged inputs round-trip through all corners."""
    from autoscilab.oracle.chembench import CHEM_INPUT_BOUNDS, CHEM_LOG_VARS

    names = tuple(CHEM_INPUT_BOUNDS)
    transform = search.SearchTransform.from_bounds(names, CHEM_INPUT_BOUNDS, CHEM_LOG_VARS)
    corners = np.array(list(itertools.product(*[CHEM_INPUT_BOUNDS[name] for name in names])))
    unit = transform.forward(corners)
    assert set(unit.ravel()) == {0.0, 1.0}
    np.testing.assert_array_equal(transform.inverse(unit), corners)
    assert set(transform.log_inputs) == set(CHEM_LOG_VARS)


@pytest.mark.parametrize("points", [[[1.1, 0.0]], [[-0.1, 0.5]], [[float("nan"), 0.5]], [[0.5]]])
def test_search_transform_refuses_undeclared_coordinates(points):
    """The search transform refuses coordinates outside the declared box or of the wrong width."""
    with pytest.raises(ValueError, match="design-search"):
        box().inverse(points)


def test_identical_seed_replays_all_rows_and_selection():
    """An identical seed replays all rows and the selection; another seed changes the starts."""
    first, second = run_search(), run_search()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    other = run_search(17)
    assert [row["start_unit"] for row in first["trajectories"]] != [
        row["start_unit"] for row in other["trajectories"]]
    for index, trajectory in enumerate(first["trajectories"], 1):
        expected = seed_for("design_restart", 11, "001", 1, index)
        assert trajectory["seed_i"] == expected
        assert trajectory["start_unit"] == np.random.default_rng(expected).uniform(0, 1, 2).tolist()


def test_batches_keep_all_evaluations_but_choose_only_five_arrivals(monkeypatch):
    """All evaluations are kept in the record but only the five arrival rows are chosen from."""
    evaluated, selected = [], []
    objective = noisy_objective()
    authority = search.choose_design

    def observed_objective(points, seed):
        """Record each objective call, then delegate to the wrapped objective."""
        evaluated.append((points.copy(), seed))
        return objective(points, seed)

    def observed_chooser(rows, key):
        """Record the rows offered to the chooser, then delegate to the real chooser."""
        selected.append(rows)
        return authority(rows, key)

    monkeypatch.setattr(search, "choose_design", observed_chooser)
    record = run_search(objective=observed_objective)
    assert len(selected) == 1 and len(selected[0]) == search.RESTARTS
    assert all(row["phase"] == "arrival" for row in selected[0])
    assert all(len(points) == 1 for points, _ in evaluated[:5] + evaluated[-21:])
    assert all(seed == record["seed"] for _, seed in evaluated[:-16])
    assert all(seed == record["noise_floor_seed"] == seed_for("design_search_noise_floor", 11, 1)
               for _, seed in evaluated[-16:-6])
    assert all(len(points) == 1 for points, _ in evaluated[-6:])
    assert all(seed == record["holdout_seed"] == seed_for("design_search_holdout_final", 11, 1)
               for _, seed in evaluated[-6:])
    assert evaluated[-6][0].tolist() == [record["chosen"]]
    trace = sorted(record["rows"] + record["perturbation_trace"], key=lambda row: row["evaluation_index"])
    assert len(trace) == record["objective_evaluation_count"] == sum(len(points) for points, _ in evaluated)
    assert [row["evaluation_index"] for row in trace] == list(range(len(trace)))
    assert all(row["batch_size"] == 1 for row in record["rows"])
    assert all(not {"eig", "epig", "mc_standard_error"} & set(row) for row in record["perturbation_trace"])
    for call_index, (points, seed) in enumerate(evaluated):
        call_rows = [row for row in trace if row["objective_call_index"] == call_index]
        assert len(call_rows) == len(points)
        assert [row["row_index"] for row in call_rows] == list(range(len(points)))
        assert all(row["batch_size"] == len(points) and row["objective_seed"] == seed for row in call_rows)
    assert record["candidate_exclusion_policy"] == "none"
    assert all("pool_index" not in row for row in record["rows"])
    assert record["policy"]["selection_rows"] == "arrival_rows_seed_A_only"
    assert record["holdout_value"] == objective(
        np.asarray([record["chosen"]]), record["holdout_seed"])[0]["eig"]


def test_clipping_and_trajectory_are_explicit(monkeypatch):
    # The gains are module constants; a test varies one by monkeypatch.
    """Clipping and per-step trajectory fields are explicit and consistent with the traces."""
    monkeypatch.setattr(search, "PARAMETER_DEFAULTS", {**search.PARAMETER_DEFAULTS, "c": 0.8})
    record = run_search()
    assert record["parameters"]["c"] == 0.8
    trace = {row["evaluation_index"]: row for row in record["perturbation_trace"]}
    steps = [step for trajectory in record["trajectories"] for step in trajectory["steps"]]
    assert any(step["plus_clipped"] or step["minus_clipped"] for step in steps)
    for row in record["rows"] + record["perturbation_trace"]:
        assert np.all((np.asarray(row["unit_design"]) >= 0) & (np.asarray(row["unit_design"]) <= 1))
        assert np.all((np.asarray(row["design"]) >= 0) & (np.asarray(row["design"]) <= 1))
    for trajectory in record["trajectories"]:
        assert 20 <= trajectory["stop_step"] <= 500
        assert trajectory["stop_reason"] in ("stopped_after_gain", "stopped_without_gain", "max_steps")
        assert len(trajectory["steps"]) == trajectory["stop_step"]
        assert all(np.isfinite(step["gradient_norm"]) for step in trajectory["steps"])
        assert trajectory["path_length"] == trajectory["steps"][-1]["path_length"]
        assert all(0 <= step["clip_count"] <= 2 for step in trajectory["steps"])
        assert all(step["gradient_zero_displacement_count"] == 0 for step in trajectory["steps"])
        for step in trajectory["steps"]:
            before, delta = np.asarray(step["before_unit"]), np.asarray(step["delta"])
            raw_plus, raw_minus = before + step["c_k"] * delta, before - step["c_k"] * delta
            plus = np.asarray(trace[step["plus_evaluation"]]["unit_design"])
            minus = np.asarray(trace[step["minus_evaluation"]]["unit_design"])
            assert step["clip_count"] == np.count_nonzero((raw_plus != plus) | (raw_minus != minus))
            assert step["min_displacement_fraction"] == float(np.min(np.abs(plus - minus)) / (2 * step["c_k"]))


def test_one_sided_boundary_gradient_is_one_not_one_half():
    """A one-sided boundary gradient is 1, not 1/2."""
    gradient, guarded, displacement = search.displacement_gradient(0.05, [0.05], [0.0])
    np.testing.assert_array_equal(gradient, [1.0])
    np.testing.assert_array_equal(displacement, [0.05])
    assert guarded == 0


def test_unclipped_displacement_gradient_is_exact_standard_spsa():
    """An unclipped displacement gradient equals the standard SPSA estimate exactly."""
    delta = np.array([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    c_k = 0.125
    gradient, guarded, _ = search.displacement_gradient(0.75, 0.5 + c_k * delta, 0.5 - c_k * delta)
    np.testing.assert_array_equal(gradient, 0.75 / (2 * c_k) * delta)
    assert guarded == 0


def test_tiny_coordinate_displacement_is_zeroed_and_counted():
    """A tiny coordinate displacement is zeroed and counted as guarded."""
    gradient, guarded, _ = search.displacement_gradient(1.0, [0.0, 5e-13, 0.25], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(gradient, [0.0, 0.0, 4.0])
    assert guarded == 2


def test_boundary_optimum_is_reached_without_changing_gain_defaults():
    """A boundary optimum is reached without changing the default gains."""
    def objective(points, seed):
        """Objective that increases linearly in the first coordinate."""
        return [{"design": point.tolist(), "eig": float(point[0])} for point in points]

    record = run_search(objective=objective)
    assert abs(record["chosen"][0] - 1.0) == 0.0
    assert record["parameters"] == search.PARAMETER_DEFAULTS
    assert any(step["clip_count"] > 0 for row in record["trajectories"] for step in row["steps"])


def test_noiseless_flat_restart_is_not_called_converged():
    """A noiseless flat restart is labelled stopped-without-gain, not converged."""
    record = run_search(objective=noisy_objective(0.0, flat=True))
    for trajectory in record["trajectories"]:
        assert trajectory["stop_reason"] == "stopped_without_gain"
        assert trajectory["path_length"] == trajectory["value_climbed"] == 0.0
        assert trajectory["stop_step"] == 20
        assert trajectory["stop_z"] is None and trajectory["stop_marginal"] is False
    assert record["holdout_value"] == record["holdout_start_value"] == 1.0
    assert record["holdout_climb"] == 0.0 and record["holdout_survival"] is None


@pytest.mark.parametrize("seed", range(8))
def test_fifteen_percent_noise_floor_and_selected_restart(seed):
    """With 15% noise, stop labels and the selected restart follow the declared rules."""
    record = run_search(seed)
    # These are operational labels, not optimum certificates.
    objective = noisy_objective(0.15)
    for trajectory in record["trajectories"]:
        start = box().inverse(np.array([trajectory["start_unit"]]))
        end = box().inverse(np.array([trajectory["end_unit"]]))
        climb = objective(end, record["seed"])[0]["eig"] - objective(start, record["seed"])[0]["eig"]
        assert trajectory["value_climbed"] == climb
        if not trajectory["movement_stop_met"] or trajectory["stop_step"] == 500:
            expected = "max_steps"
        elif climb > 3.0 * record["noise_floor_sigma_hat"] and trajectory["path_length"] > 0:
            expected = "stopped_after_gain"
        else:
            expected = "stopped_without_gain"
        assert trajectory["stop_reason"] == expected
    chosen = record["trajectories"][record["chosen_restart_index"] - 1]
    assert record["chosen_stop_reason"] == chosen["stop_reason"]
    assert record["chosen_endpoint_unlabeled_gain"] is (chosen["stop_reason"] != "stopped_after_gain")
    assert record["chosen"] == search.choose_design(record["arrival_rows"], "eig")["chosen_design"]
    if chosen["value_climbed"] <= 0:
        assert record["holdout_survival"] is None
    else:
        assert record["holdout_survival"] == record["holdout_climb"] / chosen["value_climbed"]
    for trajectory in record["trajectories"]:
        z = trajectory["value_climbed"] / record["noise_floor_sigma_hat"]
        assert trajectory["stop_z"] == z
        assert trajectory["stop_marginal"] is (abs(z - 3.0) < 1)


def test_noise_floor_uses_saved_a_rows_and_independent_b_only():
    """The noise floor uses the saved A rows and an independent B evaluation only."""
    record = run_search()
    a_rows = [row for row in record["rows"] if row["phase"] in ("start", "arrival")]
    b_rows = [row for row in record["rows"] if row["phase"] == "noise_floor"]
    assert all(row["batch_size"] == 1 and row["row_index"] == 0 for row in a_rows + b_rows)
    assert [row["design"] for row in a_rows] == [row["design"] for row in b_rows]
    assert record["noise_check_points_sha256"]["A"] == record["noise_check_points_sha256"]["B"]
    differences = np.array([a["eig"] - b["eig"] for a, b in zip(a_rows, b_rows, strict=True)])
    expected = search.noise_floor_estimates([row["eig"] for row in a_rows],
                                           [row["eig"] for row in b_rows])
    assert record["noise_floor_method"] == "pairwise"
    assert record["noise_floor_sigma_hat"] == expected["pairwise"]
    assert record["noise_floor_estimates"] == expected
    assert record["noise_floor_offset"] == float(differences.mean())
    assert "noise_mean_difference_magnitude_ge_sigma" not in record
    assert record["noise_check_point_count"] == 10
    assert record["chosen"] == search.choose_design(record["arrival_rows"], "eig")["chosen_design"]


def test_reported_climb_uses_size_one_even_for_batch_dependent_objective():
    """The reported climb uses batch-size-one rows even for a batch-dependent objective."""
    def objective(points, seed):
        """Objective whose value depends on batch size and row position."""
        return [{"design": point.tolist(), "eig": float(point[0] + len(points) + i)}
                for i, point in enumerate(points)]

    record = run_search(objective=objective)
    by_index = {row["evaluation_index"]: row for row in record["rows"]}
    for trajectory in record["trajectories"]:
        start = by_index[trajectory["start_evaluation"]]
        end = by_index[trajectory["end_evaluation"]]
        assert start["batch_size"] == end["batch_size"] == 1
        assert start["row_index"] == end["row_index"] == 0
        assert trajectory["value_climbed"] == end["eig"] - start["eig"]
        assert start["eig"] == trajectory["start_unit"][0] + 1.0
        assert end["eig"] == trajectory["end_unit"][0] + 1.0


def test_no_duplicate_start_or_endpoint_evaluations_remain():
    """No duplicate start or endpoint evaluations remain in the record."""
    record = run_search()
    for phase in ("start", "arrival"):
        rows = [row for row in record["rows"] if row["phase"] == phase]
        assert len(rows) == 5
        assert all(row["batch_size"] == 1 and row["row_index"] == 0 for row in rows)
        assert len({row["objective_call_index"] for row in rows}) == 5
    assert not any(row["phase"].startswith("check_") for row in record["rows"])
    assert all(row["batch_size"] == 1 for row in record["arrival_rows"])
    assert record["policy"]["chosen_value"] == max(row["eig"] for row in record["arrival_rows"])


def test_runner_timing_counts_five_candidates_not_diagnostic_rows(tmp_path):
    """Runner timing counts the five candidates, not the diagnostic rows."""
    import ast
    import inspect

    from . import chembench_core
    from .count_inference import record_acquisition_timing

    record = run_search()
    assignments = [node for node in ast.walk(ast.parse(inspect.getsource(chembench_core.run)))
                   if isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Name) and target.id == "acq_rows" for target in node.targets)
                   and isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name)
                   and node.value.value.id == "record"]
    assert len(assignments) == 1
    key = ast.literal_eval(assignments[0].value.slice)
    assert key == "arrival_rows"
    assert len(record["rows"]) == 26
    assert len(record[key]) == record["policy"]["n_candidates"] == 5
    record_acquisition_timing(tmp_path, 1, "joint_eig", 0.25, len(record[key]))
    timing = json.loads((tmp_path / "acquisition_timings.jsonl").read_text())
    assert timing["n_candidates"] == 5
    assert record["objective_evaluation_count"] > len(record["rows"])


def test_final_holdout_does_not_change_search_selection_or_stop_labels():
    """The final holdout does not change the search selection or stop labels."""
    baseline = run_search()
    objective = noisy_objective()
    final_seed = seed_for(search.HOLDOUT_STREAM, 11, 1)

    def changed_holdout(points, seed):
        """Objective that shifts only the final-holdout evaluations by a large constant."""
        rows = objective(points, seed)
        if seed == final_seed:
            for row in rows:
                row["eig"] += 1000
        return rows

    changed = run_search(objective=changed_holdout)
    assert changed["chosen"] == baseline["chosen"]
    assert changed["trajectories"] == baseline["trajectories"]
    assert changed["holdout_value"] == baseline["holdout_value"] + 1000
    assert changed["holdout_start_value"] == baseline["holdout_start_value"] + 1000
    assert [row for row in changed["rows"] if not row["phase"].startswith("independent_holdout")] == [
        row for row in baseline["rows"] if not row["phase"].startswith("independent_holdout")]


@pytest.mark.parametrize("rho", [0.0, 0.25, 0.5, 0.9, 1.0])
def test_gap_uncertainty_is_the_declared_independent_sum_upper_bound(rho):
    """The gap uncertainty is the declared independent-sum upper bound for any correlation."""
    rows = [{"design": [0.2, 0.4], "eig": 1.0, "mc_standard_error": 0.2},
            {"design": [0.7, 0.8], "eig": 0.9, "mc_standard_error": 0.3}]
    result = search.selection_diagnostics(rows, search.choose_design(rows))
    paired = np.sqrt(0.2 ** 2 + 0.3 ** 2 - 2 * rho * 0.2 * 0.3)
    assert result["estimated_uncertainty"] == float(np.sqrt(0.2 ** 2 + 0.3 ** 2))
    assert result["estimated_uncertainty"] >= paired
    assert result["selection_uncertain"] is True
    assert result["uncertainty_method"] == "independent_sum_upper_bound"


@pytest.mark.parametrize("count,second,with_se", [(1, 0.0, True), (2, 1.0, True),
                                                 (2, 0.9, True), (2, 0.9, False)])
def test_gap_degenerate_cases_have_no_nonfinite_ratios(count, second, with_se):
    """Degenerate gap cases yield no non-finite ratios and strict-JSON-safe diagnostics."""
    rows = [{"design": [0.2, 0.4], "eig": 1.0}, {"design": [0.7, 0.8], "eig": second}][:count]
    if with_se:
        for row in rows:
            row["mc_standard_error"] = 0.0
    result = search.selection_diagnostics(rows, search.choose_design(rows))
    assert result["top_gap_over_uncertainty"] is None
    json.dumps(result, allow_nan=False)
    if count == 1:
        assert result["runner_up_value"] is result["estimated_uncertainty"] is result["top_gap"] is None
        assert result["selection_uncertain"] is False
    elif with_se:
        assert result["selection_uncertain"] is (second == 1.0)
    else:
        assert result["selected_mc_se"] is result["runner_up_mc_se"] is None
        assert result["uncertainty_method"] == "unavailable"
        assert result["selection_uncertain"] is None


def test_pool_holdout_is_one_point_and_does_not_change_rows():
    """The pool holdout is a single point and leaves the rows unchanged."""
    objective = noisy_objective()
    rows = objective(np.array([[0.2, 0.4], [0.7, 0.8]]), 42)
    original = json.dumps(rows)
    decision = search.choose_design(rows)
    calls = []

    def measured(points, seed):
        """Record the call, then delegate to the wrapped objective."""
        calls.append((points, seed))
        return objective(points, seed)

    record = search.pool_diagnostics(measured, rows, decision, global_seed=11, round_index=1)
    assert len(calls) == 1 and calls[0][0].shape == (1, 2)
    assert calls[0][1] == seed_for(search.HOLDOUT_STREAM, 11, 1)
    assert record["holdout_batch_size"] == 1 and record["holdout_batch_order"] == ["chosen"]
    assert record["holdout_value"] == objective(*calls[0])[0]["eig"]
    assert json.dumps(rows) == original


def test_search_parser_omits_all_unstated_keys_and_has_no_gain_options(capsys):
    """The search parser omits all unstated keys and exposes no gain options."""
    import argparse

    parser = argparse.ArgumentParser()
    search.add_cli_arguments(parser)
    assert vars(parser.parse_args([])) == {}
    # The stated mode is recorded alone; each former gain option is refused.
    assert vars(parser.parse_args(["--design-search", "ascent"])) == {"design_search": "ascent"}
    for name, value in search.PARAMETER_DEFAULTS.items():
        flag = "--design-search-" + name.replace("_", "-")
        with pytest.raises(SystemExit) as refused:
            parser.parse_args(["--design-search", "ascent", flag, str(value)])
        assert refused.value.code == 2
        assert "unrecognized arguments: " + flag in capsys.readouterr().err
    assert parser.parse_args(["--design-search-noise-floor-method", "pairwise"]).design_search_noise_floor_method == "pairwise"


@pytest.mark.usefixtures("require_autoscilab_source")
def test_pre_run_receipt_binds_only_declared_settings_and_omitted_path_stays_absent():
    """The pre-run receipt binds only declared settings; an omitted path stays absent."""
    from . import chembench_core, run_label_policy

    base = {"profile": "dev", "rounds": 1, "cell": "001", "design_policy": "joint_eig",
            "prompt_version": "v2.0", "prompt_sha256": "a" * 64}
    omitted = run_label_policy.bmd_runtime_settings("chembench", SimpleNamespace(**base))
    assert "candidate_source" not in omitted and "design_search" not in omitted
    pool = run_label_policy.bmd_runtime_settings("chembench", SimpleNamespace(**base, design_search="pool"))
    assert pool == {**omitted, "candidate_source": "pool"}
    args = SimpleNamespace(**base, design_search="ascent")
    declared = search.declared_settings(args, chembench_core.ACQUISITION_STREAM)
    actual = run_label_policy.bmd_runtime_settings("chembench", args)
    assert actual == {**omitted, **declared}
    assert declared["design_search_constants"] == search.design_search_constants()
    assert declared["design_search_constants_sha256"] == search.constants_sha256(
        declared["design_search_constants"])
    assert "design_search_parameters" not in declared
    assert declared["noise_floor_method"] == "pairwise"
    assert len(set(declared["design_search_seed_streams"].values())) == 3
    assert json.loads(json.dumps(declared)) == declared
    assert not {"rows", "trajectories", "stop_reason", "noise_floor_sigma_hat"} & set(declared)


@pytest.mark.parametrize("options,constants", [
    ({"design_search": "ascent"}, {"alpha": 0.1, "gamma": 0.1}),
    ({"design_search": "ascent", "cell": "000"}, {}),
    ({"design_search": "ascent", "candidate_exclusion_policy": "observed"}, {}),
    ({"design_search": "pool", "design_search_noise_floor_method": "pairwise"}, {}),
    ({"design_search": "invalid"}, {}),
])
def test_invalid_search_options_refuse_before_output(tmp_path, monkeypatch, options, constants):
    """Invalid search options are refused before any output is written."""
    from . import chembench_core

    # A gain is a module constant, so a bad one is monkeypatched in.
    monkeypatch.setattr(search, "PARAMETER_DEFAULTS", {**search.PARAMETER_DEFAULTS, **constants})
    output = tmp_path / "absent"
    args = SimpleNamespace(**{"cell": "001", "design_policy": "joint_eig",
                              "output": str(output), **options})
    with pytest.raises(ValueError, match="design-search"):
        chembench_core.run(args)
    assert not output.exists()


# ---------------------------------------------------------------------------
# Design-search constants (fixed by a project decision,
# 2026-09-23): the constants cannot change at run time and are recorded, with a hash, whenever --design-search
# is stated.
# ---------------------------------------------------------------------------

#: The constants object exactly as decided.
RULING_CONSTANTS = ('{"a":0.2,"A":10,"alpha":0.602,"gamma":0.101,"c":0.05,"W":10,"kappa":1.0,'
                    '"k_sigma":3,"R":5,"K_min":20,"K_max":500}')
#: Its canonical JSON (sort_keys, separators (",", ":")), the text the recorded sha256 hashes.
CANONICAL_CONSTANTS = ('{"A":10,"K_max":500,"K_min":20,"R":5,"W":10,"a":0.2,"alpha":0.602,'
                       '"c":0.05,"gamma":0.101,"k_sigma":3,"kappa":1.0}')
#: The four acquisition-guard declarations (a project decision
#: of 2026-09-24, confirmed on 2026-09-25) registered beside
#: the decision's 11 values, in the constants object's order; a later decision (2026-09-26) moved the
#: guard to v2 and added its three lognormal representability constants.
AGX_CONSTANTS = ('{"acquisition_guard":"particle_conditioning_v2","lognormal_tail_sigma":8,'
                 '"representability_lower":-708.3964185322641,'
                 '"representability_upper":709.782712893384,'
                 '"fallback_policy":"fixed_random_designs_from_native_input_distribution",'
                 '"invalid_reasons":["nonfinite_or_nonpositive_mean"],'
                 '"acquisition_statuses":["ascent_with_exclusions","all_designs_invalid",'
                 '"acquisition_failed"]}')
#: The whole registered object's canonical JSON, the text the recorded sha256 hashes, and that sha.
CANONICAL_AGX_CONSTANTS = (
    '{"A":10,"K_max":500,"K_min":20,"R":5,"W":10,"a":0.2,'
    '"acquisition_guard":"particle_conditioning_v2",'
    '"acquisition_statuses":["ascent_with_exclusions","all_designs_invalid","acquisition_failed"],'
    '"alpha":0.602,"c":0.05,'
    '"fallback_policy":"fixed_random_designs_from_native_input_distribution",'
    '"gamma":0.101,"invalid_reasons":["nonfinite_or_nonpositive_mean"],"k_sigma":3,"kappa":1.0,'
    '"lognormal_tail_sigma":8,"representability_lower":-708.3964185322641,'
    '"representability_upper":709.782712893384}')
CANONICAL_AGX_CONSTANTS_SHA256 = "827a55876ae3fb2753bedb335dd15e31ab29fe7d963928057d83f3b959d4a2cf"
REMOVED_GAIN_FLAGS = ("--design-search-a", "--design-search-A", "--design-search-alpha",
                      "--design-search-gamma", "--design-search-c", "--design-search-W",
                      "--design-search-kappa", "--design-search-k-sigma")


def test_constants_object_is_the_module_constants_and_hashes_its_canonical_json(monkeypatch):
    """The constants object equals the module constants and hashes its canonical JSON."""
    constants = search.design_search_constants()
    module = {**search.PARAMETER_DEFAULTS, "R": search.RESTARTS,
              "K_min": search.MIN_STEPS, "K_max": search.MAX_STEPS}
    # The 11 decided values stay exactly as decided (a subset, with the same types) and lead
    # the object; the guard declarations (four, plus the three added by the later decision) are
    # the only other keys.
    assert len(module) == 11 and list(constants)[:11] == list(module)
    for key, value in module.items():
        assert type(constants[key]) is type(value) and constants[key] == value, key
    ruling = {key: constants[key] for key in module}
    assert ruling == json.loads(RULING_CONSTANTS)
    assert json.dumps(ruling, sort_keys=True, separators=(",", ":")) == CANONICAL_CONSTANTS
    added = {key: value for key, value in constants.items() if key not in module}
    assert json.dumps(added, separators=(",", ":")) == AGX_CONSTANTS
    assert len(constants) == 18
    text = json.dumps(constants, sort_keys=True, separators=(",", ":"))
    assert text == CANONICAL_AGX_CONSTANTS
    assert search.constants_sha256(constants) == hashlib.sha256(text.encode()).hexdigest()
    assert search.constants_sha256(constants) == CANONICAL_AGX_CONSTANTS_SHA256
    # Read at call time: a monkeypatched module constant is the one recorded.
    monkeypatch.setattr(search, "MAX_STEPS", 400)
    monkeypatch.setattr(search, "PARAMETER_DEFAULTS", {**search.PARAMETER_DEFAULTS, "c": 0.8})
    changed = search.design_search_constants()
    assert changed["K_max"] == 400 and changed["c"] == 0.8
    assert search.constants_sha256(changed) != search.constants_sha256(constants)


class StopAtFirstFit(Exception):
    """Raised by the first SMC fit; config.json is written, once, before any fit."""


DEV_MANIFEST = Path(__file__).resolve().parents[3] / "manifests" / "chembench_dev_manifest.json"


def stopped_mock_run(tmp_path, monkeypatch, name, cell="001", **options):
    """A real mock ChemBench run (real adapter, dev manifest), stopped at its first fit."""
    from . import chembench_core
    from .test_chembench_core import _args

    def first_fit(*args, **kwargs):
        """Replacement for the SMC fit that stops the run at its first call."""
        raise StopAtFirstFit

    monkeypatch.setattr(chembench_core, "_timed_fit", first_fit)
    output = tmp_path / name
    with pytest.raises(StopAtFirstFit):
        chembench_core.run(_args(output, DEV_MANIFEST, cell=cell, **options))
    return json.loads((output / "config.json").read_text())


@pytest.mark.usefixtures("require_autoscilab_source")
def test_mock_run_records_constants_and_hash_in_config_and_receipt_only_when_stated(
        tmp_path, monkeypatch):
    """A mock run records the constants and hash in config and receipt only when the search is stated."""
    from . import chembench_core
    from .test_chembench_core import _args

    source = hashlib.sha256(Path(search.__file__).read_bytes()).hexdigest()
    ascent = stopped_mock_run(tmp_path, monkeypatch, "ascent", design_search="ascent")
    constants = ascent["design_search_constants"]
    assert constants == search.design_search_constants()
    recomputed = hashlib.sha256(
        json.dumps(constants, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert ascent["design_search_constants_sha256"] == recomputed
    receipt = ascent["label_receipt"]["binding"]["settings"]
    assert receipt["design_search_constants"] == constants
    assert receipt["design_search_constants_sha256"] == recomputed
    assert ascent["source_sha256"]["design_search.py"] == source
    assert not {"design_search_parameters", "design_search_source_sha256"} & set(ascent)
    # By the pool-refusal rule: on the active seven-input cell the omitted and the explicit pool
    # declarations are both refused before any output exists ...
    for name, options in (("omitted_001", {}), ("pool_001", {"design_search": "pool"})):
        output = tmp_path / name
        with pytest.raises(ValueError, match="design-search pool is refused for joint_eig "
                                             "at input dimension 7 > 2"):
            chembench_core.run(_args(output, DEV_MANIFEST, cell="001", **options))
        assert not output.exists()
    # ... so their recording is checked on cell 100, which reads no pool.
    omitted = stopped_mock_run(tmp_path, monkeypatch, "omitted", cell="100",
                               structure_proposals=None)
    assert "design_search" not in json.dumps(omitted)
    pool = stopped_mock_run(tmp_path, monkeypatch, "pool", cell="100", structure_proposals=None,
                            design_search="pool")
    assert "design_search_constants" not in json.dumps(pool)
    assert pool["source_sha256"]["design_search.py"] == source


@pytest.mark.usefixtures("require_autoscilab_source")
@pytest.mark.parametrize("registered", ["run_hash", "different_hash"])
def test_registered_constants_hash_must_equal_the_runs(tmp_path, registered):
    """A registered constants hash must equal the run's own hash."""
    from . import run_label_policy
    from .test_run_label_policy import registration, runner_args, write_manifest

    settings = run_label_policy.bmd_runtime_settings(
        "chembench", runner_args(cell="001", design_policy="joint_eig", design_search="ascent"))
    assert settings["design_search_constants_sha256"] == search.constants_sha256(
        search.design_search_constants())
    ref, context, manifest = registration(tmp_path, bundle="chembench", task="synthetic",
                                          cell="001", settings=settings)
    if registered == "different_hash":
        different = search.constants_sha256({**settings["design_search_constants"], "c": 0.8})
        assert different != settings["design_search_constants_sha256"]
        manifest["runtime_settings"]["chembench"]["design_search_constants_sha256"] = different
        ref = write_manifest(Path(ref["path"]), manifest)
        with pytest.raises(ValueError, match="Manifest runtime settings mismatch"):
            run_label_policy.verify_manifest(ref, **context)
    else:
        provenance = run_label_policy.verify_manifest(ref, **context)
        assert provenance["matched_row"]["cell"] == "001"


@pytest.mark.parametrize("flag", REMOVED_GAIN_FLAGS)
def test_runner_parser_refuses_each_removed_gain_option_before_output(
        tmp_path, monkeypatch, capsys, flag):
    """The runner parser refuses each removed gain option before any output."""
    from . import chembench_core

    output = tmp_path / "absent"
    monkeypatch.setattr(chembench_core, "run", lambda args: pytest.fail("run() must not be reached"))
    monkeypatch.setattr(sys, "argv", ["chembench_core", "--manifest", "m", "--output", str(output),
                                      "--noise", "0.01", "--design-search", "ascent", flag, "1"])
    with pytest.raises(SystemExit) as refused:
        chembench_core.main()
    assert refused.value.code == 2
    assert "unrecognized arguments: " + flag + " 1" in capsys.readouterr().err
    assert not output.exists()


def test_design_search_source_reads_no_process_environment():
    """The design-search source never reads the process environment."""
    assert b"environ" not in Path(search.__file__).read_bytes()
