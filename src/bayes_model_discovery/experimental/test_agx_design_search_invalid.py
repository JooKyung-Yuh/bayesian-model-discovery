"""Guard rules B1-B9: projected_spsa with explicit invalid objective rows.

Every objective is synthetic: test_design_search's point-keyed noisy quadratic, with the rows a
rule selects replaced by `{"design", "invalid": True, "invalid_reason"}`. Because each row's
value depends only on its own point and seed, the same run without invalid rows is the exact
reference for everything an invalid row must not change.
"""

import json

import numpy as np
import pytest

from . import design_search as search
from .test_design_search import noisy_objective, run_search

REASON = "nonfinite_or_nonpositive_mean"
FIXED_TRACE_KEYS = ["design", "unit_design", "phase", "step", "restart_index", "objective_seed",
                    "evaluation_index", "objective_call_index", "batch_size", "row_index"]
SKIP_KEYS = ["step", "a_k", "c_k", "before_unit", "after_unit", "delta", "plus_clipped",
             "minus_clipped", "clip_count", "plus_evaluation", "minus_evaluation", "path_length",
             "skipped", "skip_reason", "plus_invalid", "minus_invalid"]
POINT_KEYS = ["objective_call_index", "evaluation_index", "phase", "step", "restart_index",
              "design", "unit_design", "invalid_reason"]
#: What an invalid perturbation row may not change in the OTHER restarts: their geometry.
GEOMETRY = ("step", "a_k", "c_k", "before_unit", "after_unit", "delta", "gradient",
            "gradient_norm", "update_clipped", "window_displacement", "path_length")
AGX_RECORD_KEYS = ("acquisition_status", "invalid_design_points", "noise_floor_status",
                   "noise_floor_invalid_check_points")


def invalid_row(point):
    """Return the explicit invalid objective row for a design point."""
    return {"design": point.tolist(), "invalid": True, "invalid_reason": REASON}


def marking(rule):
    """The noisy objective, with the rows `rule(call, row, point, seed)` selects invalid."""
    base = noisy_objective()
    calls = []

    def objective(points, seed):
        call = len(calls)
        calls.append((points.copy(), seed))
        return [invalid_row(point) if rule(call, index, point, seed) else row
                for index, (point, row) in enumerate(zip(points, base(points, seed), strict=True))]

    objective.calls = calls
    return objective


def replayed_deltas(trajectory):
    """Each restart's own generator: one start draw, then one delta draw per step taken."""
    rng = np.random.default_rng(trajectory["seed_i"])
    rng.uniform(0.0, 1.0, len(trajectory["start_unit"]))
    return [rng.choice((-1.0, 1.0), len(trajectory["start_unit"])).tolist()
            for _ in trajectory["steps"]]


def spy_chooser(patch):
    """Patch choose_design to record the rows it receives; return that list."""
    seen = []
    real = search.choose_design

    def chooser(rows, key):
        seen.append(rows)
        return real(rows, key)

    patch.setattr(search, "choose_design", chooser)
    return seen


def call_of(record, phase, restart):
    """Return the objective call index of the first row with the given phase and restart."""
    return next(row["objective_call_index"] for row in record["rows"]
                if row["phase"] == phase and row["restart_index"] == restart)


def test_all_valid_run_hands_the_chooser_the_arrival_list_itself_and_adds_no_key(monkeypatch):
    """With all rows valid the chooser gets the arrival list itself and no record key is added."""
    seen = spy_chooser(monkeypatch)
    record = run_search()
    assert len(seen) == 1 and seen[0] is record["arrival_rows"]
    assert not set(AGX_RECORD_KEYS) & set(record)
    assert not any("skipped" in step for trajectory in record["trajectories"]
                   for step in trajectory["steps"])
    assert all(list(row) == FIXED_TRACE_KEYS for row in record["perturbation_trace"])


def test_one_invalid_side_skips_only_that_restarts_step_and_the_others_update():
    """One invalid perturbation side skips only that restart's step; the other restarts update normally."""
    baseline = run_search()
    # Calls 0-4 are the five starts, so call 5 is step 1's batch; rows 2 and 3 are restart 2.
    record = run_search(objective=marking(lambda call, row, point, seed: call == 5 and row == 2))
    trajectory = record["trajectories"][1]
    skipped = trajectory["steps"][0]
    assert list(skipped) == SKIP_KEYS
    assert (skipped["skipped"], skipped["skip_reason"]) == (True, "invalid_perturbation_design")
    assert (skipped["plus_invalid"], skipped["minus_invalid"]) == (True, False)
    assert skipped["before_unit"] == skipped["after_unit"] == trajectory["start_unit"]
    assert skipped["path_length"] == 0.0
    reference = baseline["trajectories"][1]["steps"][0]
    for key in ("step", "a_k", "c_k", "before_unit", "delta", "plus_clipped", "minus_clipped",
                "clip_count", "plus_evaluation", "minus_evaluation"):
        assert skipped[key] == reference[key], key
    # Step 2 starts where step 1 left the restart: its start.
    assert trajectory["steps"][1]["before_unit"] == trajectory["start_unit"]
    assert "skipped" not in trajectory["steps"][1]
    # The other restarts' search is unchanged, step for step.
    for index in (0, 2, 3, 4):
        mine, theirs = record["trajectories"][index], baseline["trajectories"][index]
        assert [{key: step[key] for key in GEOMETRY} for step in mine["steps"]] == [
            {key: step[key] for key in GEOMETRY} for step in theirs["steps"]]
        for key in ("stop_step", "end_unit", "movement_stop_met", "value_climbed"):
            assert mine[key] == theirs[key], key
    assert record["acquisition_status"] == "ascent_with_exclusions"
    [point] = record["invalid_design_points"]
    assert list(point) == POINT_KEYS
    assert (point["phase"], point["step"], point["restart_index"],
            point["objective_call_index"]) == ("perturbation", 1, 2, 5)
    [trace] = [row for row in record["perturbation_trace"] if "invalid" in row]
    assert list(trace) == FIXED_TRACE_KEYS + ["invalid", "invalid_reason"]
    assert trace["evaluation_index"] == point["evaluation_index"] == skipped["plus_evaluation"]
    assert all(list(row) == FIXED_TRACE_KEYS for row in record["perturbation_trace"]
               if row is not trace)
    # Only perturbation rows were invalid: the noise floor and the chooser are ordinary.
    assert "noise_floor_status" not in record and record["noise_floor_sigma_hat"] is not None
    assert record["chosen"] is not None and record["policy"]["n_candidates"] == 5
    json.dumps(record, allow_nan=False)


def test_both_rows_of_a_restart_invalid_skip_its_step():
    """Both perturbation rows of a restart being invalid skips its step."""
    record = run_search(objective=marking(lambda call, row, point, seed: call == 5
                                          and row in (4, 5)))
    skipped = record["trajectories"][2]["steps"][0]
    assert list(skipped) == SKIP_KEYS
    assert (skipped["plus_invalid"], skipped["minus_invalid"]) == (True, True)
    assert skipped["after_unit"] == record["trajectories"][2]["start_unit"]
    assert [p["restart_index"] for p in record["invalid_design_points"]] == [3, 3]


@pytest.mark.parametrize("rule", [
    lambda call, row, point, seed: call == 5 and row == 2,
    lambda call, row, point, seed: 5 <= call < 40 and row in (0, 1),
    lambda call, row, point, seed: point[0] > 0.7,
])
def test_every_restarts_delta_sequence_is_its_own_generator_with_or_without_invalid_rows(rule):
    """Each restart's delta sequence comes from its own generator, with or without invalid rows."""
    baseline = run_search()
    record = run_search(objective=marking(rule))
    assert any("skipped" in step for trajectory in record["trajectories"]
               for step in trajectory["steps"])
    for mine, theirs in zip(record["trajectories"], baseline["trajectories"], strict=True):
        deltas = [step["delta"] for step in mine["steps"]]
        assert deltas == replayed_deltas(mine)
        common = min(len(deltas), len(theirs["steps"]))
        assert deltas[:common] == [step["delta"] for step in theirs["steps"]][:common]
        assert len(mine["steps"]) == mine["stop_step"]


def test_an_invalid_arrival_leaves_the_chooser_and_the_noise_floor_undetermined(monkeypatch):
    """An invalid arrival row leaves the chooser with valid arrivals only and the noise floor undetermined."""
    baseline = run_search()
    restart = baseline["chosen_restart_index"]
    arrival = call_of(baseline, "arrival", restart)
    seen = spy_chooser(monkeypatch)
    record = run_search(objective=marking(lambda call, row, point, seed: call == arrival))
    # The chooser saw the four valid arrivals only, so it could not pick the invalid one.
    assert len(seen) == 1 and len(seen[0]) == 4
    assert all("invalid" not in row for row in seen[0])
    [bad] = [row for row in record["arrival_rows"] if row.get("invalid")]
    assert bad["design"] == baseline["chosen"] != record["chosen"]
    assert record["chosen"] == search.choose_design(seen[0], "eig")["chosen_design"]
    assert record["policy"]["n_candidates"] == 4
    assert record["acquisition_status"] == "ascent_with_exclusions"
    # Rule B6: its A row makes that check point invalid, so no noise floor is estimated.
    assert record["noise_floor_status"] == "undetermined_invalid_check_point"
    assert record["noise_floor_invalid_check_points"] == [search.RESTARTS + restart - 1]
    assert (record["noise_floor_sigma_hat"], record["noise_floor_estimates"],
            record["noise_floor_offset"]) == (None, None, None)
    assert record["noise_check_point_count"] == 10
    # Rule B7: no climb from an invalid arrival; without a noise floor every label that is not
    # max_steps is null, with no z and no marginal flag.
    assert record["trajectories"][restart - 1]["value_climbed"] is None
    for trajectory in record["trajectories"]:
        if trajectory["movement_stop_met"] and trajectory["stop_step"] != search.MAX_STEPS:
            assert trajectory["stop_reason"] is None
        else:
            assert trajectory["stop_reason"] == "max_steps"
        assert trajectory["stop_z"] is None and trajectory["stop_marginal"] is False
    assert record["chosen_endpoint_unlabeled_gain"] is (
        record["chosen_stop_reason"] != "stopped_after_gain")
    # The search itself is untouched: an arrival row is never a search input.
    assert [t["steps"] for t in record["trajectories"]] == [
        t["steps"] for t in baseline["trajectories"]]
    # Rule B8: the holdout still evaluates all five endpoints under seed C, all valid here.
    assert len(record["holdout_rows"]) == 5
    assert all(value is not None for value in record["endpoint_holdout_values"])
    assert list(record) == list(baseline) + list(AGX_RECORD_KEYS)
    json.dumps(record, allow_nan=False)


def test_all_arrivals_invalid_is_all_designs_invalid_without_chooser_or_holdout(monkeypatch):
    """All arrivals invalid gives all_designs_invalid with no chooser call and no holdout."""
    baseline = run_search()
    arrivals = {call_of(baseline, "arrival", restart) for restart in range(1, 6)}
    seen = spy_chooser(monkeypatch)
    record = run_search(objective=marking(lambda call, row, point, seed: call in arrivals))
    assert seen == []
    assert record["acquisition_status"] == "all_designs_invalid"
    for key in ("chosen", "chosen_restart_index", "chosen_stop_reason",
                "chosen_endpoint_unlabeled_gain", "holdout_value", "holdout_start_value",
                "holdout_climb", "holdout_survival"):
        assert record[key] is None, key
    assert record["holdout_rows"] == [] and record["endpoint_holdout_values"] == []
    assert not any(row["phase"].startswith("independent_holdout") for row in record["rows"])
    assert record["objective_call_count"] == baseline["objective_call_count"] - 6
    assert list(record["policy"]) == [
        "objective_column", "holdout_seed_role", "candidate_source", "holdout_batch_size",
        "holdout_batch_order", "decision_authority", "selection_rows", "trajectory_pointer"]
    assert record["noise_floor_invalid_check_points"] == [5, 6, 7, 8, 9]
    assert [point["phase"] for point in record["invalid_design_points"]] == ["arrival"] * 5
    assert list(record) == list(baseline) + list(AGX_RECORD_KEYS)
    json.dumps(record, allow_nan=False)


def test_an_invalid_start_row_gives_no_climb_and_the_restart_still_searches():
    """An invalid start row gives no climb value but the restart still searches."""
    baseline = run_search()
    record = run_search(objective=marking(lambda call, row, point, seed: call == 0))
    assert record["rows"][0]["invalid"] is True and record["rows"][0]["phase"] == "start"
    trajectory = record["trajectories"][0]
    assert trajectory["value_climbed"] is None
    assert trajectory["steps"] == baseline["trajectories"][0]["steps"]
    assert record["noise_floor_invalid_check_points"] == [0]
    assert record["acquisition_status"] == "ascent_with_exclusions"


def test_invalid_holdout_rows_are_recorded_as_none(monkeypatch):
    """Invalid holdout rows are recorded as None."""
    baseline = run_search()
    chosen = baseline["chosen_restart_index"]
    other = 1 if chosen != 1 else 2
    holdout_other = call_of(baseline, "independent_holdout", other)
    holdout_start = call_of(baseline, "independent_holdout_start", chosen)
    record = run_search(objective=marking(lambda call, row, point, seed: call in (
        holdout_other, holdout_start)))
    assert record["chosen"] == baseline["chosen"]
    assert record["holdout_value"] == baseline["holdout_value"]
    assert record["holdout_start_value"] is None
    assert record["holdout_climb"] is None and record["holdout_survival"] is None
    values = record["endpoint_holdout_values"]
    assert values[other - 1] is None
    assert [v for i, v in enumerate(values) if i != other - 1] == [
        v for i, v in enumerate(baseline["endpoint_holdout_values"]) if i != other - 1]
    assert record["policy"]["holdout_value"] == baseline["holdout_value"]
    assert {p["phase"] for p in record["invalid_design_points"]} == {
        "independent_holdout", "independent_holdout_start"}
    assert "noise_floor_status" not in record


@pytest.mark.parametrize("extra,message", [
    ({"invalid_reason": "overflow"}, "invalid row must carry a registered"),
    ({"invalid_reason": None}, "invalid row must carry a registered"),
    ({"eig": 0.5}, "invalid row must carry a registered"),
    ({"mc_standard_error": 0.1}, "invalid row must carry a registered"),
    ({"invalid": 1}, "must return finite values"),
])
def test_an_invalid_row_needs_the_registered_reason_and_no_value(extra, message):
    """An invalid row must carry the registered reason and no value."""
    def objective(points, seed):
        return [{**invalid_row(point), **extra} for point in points]

    with pytest.raises(ValueError, match=message):
        run_search(objective=objective)


def test_the_four_guard_declarations_are_registered_and_the_gains_validate_alone():
    """The guard declarations are registered in the constants, and the gains validate on their own."""
    constants = search.design_search_constants()
    # Guard v2, with its three lognormal representability constants registered beside it.
    assert constants["acquisition_guard"] == "particle_conditioning_v2"
    assert (constants["lognormal_tail_sigma"], constants["representability_lower"],
            constants["representability_upper"]) == (8, -708.3964185322641, 709.782712893384)
    assert constants["fallback_policy"] == search.FALLBACK_POLICY == (
        "fixed_random_designs_from_native_input_distribution")
    assert constants["invalid_reasons"] == ["nonfinite_or_nonpositive_mean"]
    assert constants["acquisition_statuses"] == [
        "ascent_with_exclusions", "all_designs_invalid", "acquisition_failed"]
    assert set(search.validate_parameters()) == set(search.PARAMETER_DEFAULTS)
