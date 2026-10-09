"""Tests for `--design-search ascent` on BoxingGym's active design arms.

Synthetic posteriors and stub fits only; 0 USD. The ascent generalises the earlier joint expected-information-gain (EIG) compass search to the four active
EIG-family arms (joint EIG, Model EIG, Parameter EIG, expected predictive information gain (EPIG)) on a declared box, and to the exhaustive side of the
rule where the pool is a task's whole legal design set (survival). The record's objective value key is `value`
(the arm's criterion), `pool_argmax` is the pool's best row and `gain_over_pool_argmax` is the improvement.
"""

import json

import numpy as np
import pytest
from scipy.special import expit, xlogy

from . import boxinggym_core as bg
from . import boxinggym_design_search as ds
from .test_named_inputs import bg_run_args, bg_scalar_stub_fits, bg_stub_fits

BOX_TASKS = ("death", "peregrines", "dugongs", "hyperbolic", "location")
EXHAUSTIVE_POOL_TASKS = ("survival", "irt")           # the pool IS the whole legal design set
SAMPLED_POOL_TASKS = ("moral_machines", "emotion")    # neither rule covers a pool sampled from a larger space


class HyperbolicRuleEnvironment:
    """The hyperbolic validator's rule (envs/hyperbolic_temporal_discount.py validate_input), for the env-free tests:
    three items cast by int(float(.)), iR <= dR, D > 0."""

    def __init__(self):
        self.calls = 0

    def accepts(self, handle):
        """Test helper: accepts."""
        self.calls += 1
        try:
            items = [int(float(item.strip())) for item in handle.strip("[]").split(",")]
        except ValueError:
            return False
        return len(items) == 3 and items[0] <= items[1] and items[2] > 0


def synthetic_hyperbolic_objective(seed=5):
    """Exact joint EIG of a synthetic binary posterior: P(delayed) = sigmoid((dR / (1 + k D) - iR) / tau) with 64
    particles of (k, tau). Its information sits at the indifference boundary dR / (1 + k D) = iR, inside the box."""
    rng = np.random.default_rng(seed)
    k = np.exp(rng.normal(np.log(0.02), 0.6, 64))
    tau = np.exp(rng.normal(np.log(8.0), 0.3, 64))

    def entropy(p):
        """Test helper: entropy."""
        return -(xlogy(p, p) + xlogy(1 - p, 1 - p))

    def objective(rows):
        """Test helper: objective."""
        rows = np.asarray(rows, dtype=float).reshape(-1, 3)
        i_r, d_r, days = rows[:, 0], rows[:, 1], rows[:, 2]
        p = expit((d_r[None, :] / (1 + k[:, None] * days[None, :]) - i_r[None, :]) / tau[:, None])
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return entropy(p.mean(0)) - entropy(p).mean(0)

    return objective


def engine_like_pool(space, environment):
    """The 5-per-axis grid build_design_pool makes for hyperbolic, filtered by the validator (75 rows)."""
    axes = [np.unique(np.rint(np.linspace(lo, hi, 5))) for lo, hi in zip(space.low, space.high, strict=True)]
    rows = np.stack([g.reshape(-1) for g in np.meshgrid(*axes, indexing="ij")], axis=1)
    return np.asarray([r for r in rows if environment.accepts(bg.encode_design(bg.TASKS["hyperbolic"], r))])


def hyperbolic_search(seed=31, objective=None, objective_name="eig"):
    """Test helper: hyperbolic search."""
    environment = HyperbolicRuleEnvironment()
    space = ds.SearchSpace(bg.TASKS["hyperbolic"], environment, bg.encode_design)
    objective = objective or synthetic_hyperbolic_objective()
    pool = engine_like_pool(space, environment)
    return space, pool, ds.search(space, objective, pool, objective(pool), seed, objective_name=objective_name)


def test_engine_like_pool_has_the_75_rows_main_counted():
    """Test: engine like pool has the 75 rows the design builder counted."""
    _, pool, _ = hyperbolic_search()
    assert len(pool) == 75 and set(np.unique(pool[:, 0])) == {1, 76, 150, 225, 300}


def test_ascent_value_is_at_least_the_pool_argmax_and_beats_it_on_a_boundary_posterior():
    """Test: ascent value is at least the pool argmax and beats it on a boundary posterior."""
    _space, pool, (row, value, record) = hyperbolic_search()
    objective = synthetic_hyperbolic_objective()
    pool_best = float(np.max(objective(pool)))
    assert value >= pool_best
    assert record["pool_argmax"]["value_rescored"] == pytest.approx(pool_best, abs=0, rel=0)
    assert record["gain_over_pool_argmax"] == pytest.approx(value - pool_best, abs=1e-15)
    assert record["gain_over_pool_argmax"] > 0.05 and record["improved"] is True
    # the recorded value is the projected design's own value (a batch and a single row differ in the last ulp)
    assert float(objective(row[None, :])[0]) == pytest.approx(value, rel=1e-12)


def test_the_four_arm_objectives_each_run_and_stay_at_or_above_their_own_pool_argmax():
    # one synthetic objective per criterion column; each must keep its own pool argmax as a floor
    """Test: the four arm objectives each run and stay at or above their own pool argmax."""
    for name, scale in (("eig", 1.0), ("structure_eig", 0.6), ("parameter_eig", 0.4), ("epig", 0.25)):
        base = synthetic_hyperbolic_objective()
        _space, _pool, (_row, value, record) = hyperbolic_search(objective=lambda r, s=scale, b=base: s * b(r),
                                                              objective_name=name)
        assert record["objective"] == name
        assert value >= record["pool_argmax"]["value_rescored"] - ds.TIE_ATOL
        assert record["gain_over_pool_argmax"] >= -ds.TIE_ATOL


def test_chosen_and_projected_designs_are_valid_hyperbolic_integer_designs():
    """Test: chosen and projected designs are valid hyperbolic integer designs."""
    _space, _, (row, _, record) = hyperbolic_search()
    designs = [row] + [np.asarray(t["projected"]) for t in record["trajectories"]]
    for design in designs:
        assert np.array_equal(design, np.rint(design))
        i_r, d_r, days = design
        assert 1 <= i_r <= d_r <= 300 and 1 <= days <= 365
        assert HyperbolicRuleEnvironment().accepts(bg.encode_design(bg.TASKS["hyperbolic"], design))


def test_chosen_is_never_below_the_pool_argmax_even_when_the_search_cannot_improve():
    # a flat objective: every projected endpoint ties the pool argmax, so the pool argmax (smallest design) wins
    """Test: chosen is never below the pool argmax even when the search cannot improve."""
    _space, _pool, (_row, _value, record) = hyperbolic_search(objective=lambda r: np.zeros(len(np.atleast_2d(r))))
    assert record["gain_over_pool_argmax"] == pytest.approx(0.0, abs=0)
    assert record["improved"] is False
    assert record["chosen"]["source"] == "pool_argmax"
    assert record["chosen"]["design"] == record["pool_argmax"]["design"]


def test_an_objective_pulling_out_of_the_valid_region_still_returns_a_valid_design():
    # value made largest at iR >> dR and D -> 0, both refused by the validator
    """Test: an objective pulling out of the valid region still returns a valid design."""
    def pull(rows):
        """Test helper: pull."""
        rows = np.asarray(rows, dtype=float).reshape(-1, 3)
        return rows[:, 0] - rows[:, 1] - rows[:, 2] / 365.0

    _space, _, (row, _, record) = hyperbolic_search(objective=pull)
    assert row[0] <= row[1] and row[2] >= 1 and np.array_equal(row, np.rint(row))
    assert record["infeasible_neighbours_skipped"] > 0
    assert record["chosen"]["on_edge"]  # this objective's optimum IS on the boundary; the record says so


def test_bounds_are_respected_on_every_declared_box_task_env_free():
    # a linear objective maximised outside the upper bound: the chosen design must stay inside the declared box
    """Test: bounds are respected on every declared box task env free."""
    class AcceptAll:
        """Test helper class: AcceptAll."""
        def accepts(self, handle):
            """Test helper: accepts."""
            return True

    for name in ("death", "peregrines", "dugongs", "location"):
        space = ds.SearchSpace(bg.TASKS[name], AcceptAll(), bg.encode_design)
        pool = np.asarray([space.to_design(np.full(space.dim, f)) for f in (0.1, 0.5, 0.9)])

        def outward(rows, sp=space):
            """Test helper: outward."""
            rows = np.asarray(rows, dtype=float).reshape(-1, sp.dim)
            return np.sum((rows - sp.low) / (sp.high - sp.low), axis=1)

        row, _, record = ds.search(space, outward, pool, outward(pool), seed=3)
        assert np.all(row >= space.low - 1e-9) and np.all(row <= space.high + 1e-9)
        assert record["bounds"]["low"] == space.low.tolist() and record["bounds"]["high"] == space.high.tolist()


def test_search_is_deterministic_for_a_fixed_seed():
    """Test: search is deterministic for a fixed seed."""
    _, _, (row_a, value_a, record_a) = hyperbolic_search(seed=31)
    _, _, (row_b, value_b, record_b) = hyperbolic_search(seed=31)
    assert np.array_equal(row_a, row_b) and value_a == value_b
    assert json.dumps(record_a, sort_keys=True) == json.dumps(record_b, sort_keys=True)


def test_record_carries_the_task_fields_the_u3_plan_requires():
    """Test: record carries the task fields the protocol plan requires."""
    _, _, (_, _, record) = hyperbolic_search()
    # the protocol plan: design_search {mode, objective, starts, steps/step_size, bounds}, each start's end value, the pool
    # argmax value, the chosen design and its criterion, and whether ascent improved (gain)
    assert record["mode"] == "ascent" and record["objective"] == "eig"
    assert [s["source"] for s in record["starts"]] == [f"pool_rank_{k}" for k in range(1, ds.STARTS_POOL + 1)]
    assert record["step_size"]["initial_unit"] == ds.INITIAL_STEP and record["max_iterations"] == ds.MAX_ITERATIONS
    assert len(record["trajectories"]) == ds.STARTS_POOL and record["iterations"] <= ds.MAX_ITERATIONS
    for trajectory in record["trajectories"]:
        assert {"path", "moves", "halvings", "end_relaxed", "end_relaxed_value", "projected", "projected_value",
                "rounded", "projected_edge_axes", "stop_reason"} <= set(trajectory)
    assert {"design", "value", "source", "in_pool", "edge_axes", "bound_axes", "on_edge"} <= set(record["chosen"])
    assert {"pool_index", "design", "value_pool_table", "value_rescored"} <= set(record["pool_argmax"])
    assert {"improved", "gain_over_pool_argmax", "evaluation_failures"} <= set(record)
    assert record["constants_sha256"] == ds.constants_sha256()


def test_evaluator_isolates_a_failing_batch_design_by_design_and_counts_it():
    # a batch that contains a refused design raises; the evaluator retries the batch one design at a time, so a
    # failing design becomes -inf (never moved to, never chosen) while its neighbours keep their value
    """Test: evaluator isolates a failing batch design by design and counts it."""
    def obj(rows):
        """Test helper: obj."""
        rows = np.asarray(rows, dtype=float).reshape(-1, 2)
        if np.any(rows[:, 0] < 0):
            raise ValueError("synthetic refit failure on a negative first coordinate")
        return rows.sum(axis=1)

    evaluate = ds._Evaluator(obj, 2)
    values = evaluate(np.asarray([[1.0, 1.0], [-1.0, 2.0], [3.0, 0.0]]))
    assert np.isfinite(values[0]) and np.isneginf(values[1]) and np.isfinite(values[2])
    record = evaluate.record()
    # the batch call and the one bad single-design retry both raised; exactly one design is infeasible
    assert record["failed_calls"] == 2 and record["failed_designs"] == 1
    assert record["errors"] and "synthetic refit failure" in record["errors"][0]


def test_evaluator_counts_a_non_finite_value_a_call_returns_as_infeasible():
    """Test: evaluator counts a non finite value a call returns as infeasible."""
    evaluate = ds._Evaluator(lambda rows: np.asarray([np.inf, 2.0]), 1)
    values = evaluate(np.asarray([[0.0], [1.0]]))
    assert np.isneginf(values[0]) and values[1] == 2.0
    assert evaluate.record()["non_finite_values"] == 1 and evaluate.record()["failed_designs"] == 0


def test_design_space_mode_splits_box_exhaustive_and_sampled():
    """Test: design space mode splits box exhaustive and sampled."""
    for name in BOX_TASKS:
        assert ds.not_applicable_reason(bg.TASKS[name]) is None
        assert ds.design_space_mode(bg.TASKS[name]) == ("ascent", None)
    for name in EXHAUSTIVE_POOL_TASKS:
        mode, reason = ds.design_space_mode(bg.TASKS[name])
        assert mode == "exhaustive" and reason
    for name in SAMPLED_POOL_TASKS:
        with pytest.raises(ValueError, match="does not apply"):
            ds.design_space_mode(bg.TASKS[name])


def test_exhaustive_record_is_the_pool_argmax_for_each_objective():
    """Test: exhaustive record is the pool argmax for each objective."""
    rows = [{"design": [1.0, 0.0], "eig": 0.2, "structure_eig": 0.1, "parameter_eig": 0.1, "epig": 0.05},
            {"design": [2.0, 1.0], "eig": 0.9, "structure_eig": 0.3, "parameter_eig": 0.6, "epig": 0.40}]
    for name in ("eig", "structure_eig", "parameter_eig", "epig"):
        chosen = max(rows, key=lambda r, n=name: r[n])["design"]
        record = ds.exhaustive_record(name, rows, chosen, "the environment fixes the legal designs as a stored pool")
        assert record["mode"] == "exhaustive" and record["objective"] == name
        assert record["chosen"]["design"] == chosen and record["gain_over_pool_argmax"] == 0.0
        assert record["improved"] is False and record["chosen"]["in_pool"] is True


def test_command_line_option_is_absent_unless_typed():
    """Test: command line option is absent unless typed."""
    parser_args = ["--output", "x", "--task", "hyperbolic"]
    import sys
    from unittest import mock
    with mock.patch.object(sys, "argv", ["boxinggym_core", *parser_args]):
        captured = {}
        with mock.patch.object(bg, "run", side_effect=lambda args: captured.setdefault("args", args)):
            bg.main()
    assert not hasattr(captured["args"], "design_search")
    with (mock.patch.object(sys, "argv", ["boxinggym_core", *parser_args, "--design-search", "ascent"]),
          mock.patch.object(bg, "run", side_effect=lambda args: captured.__setitem__("args", args))):
        bg.main()
    assert captured["args"].design_search == "ascent"


# ---------------------------------------------------------------- with the real environments (boxing_gym source)

@pytest.mark.usefixtures("require_boxinggym_source")
@pytest.mark.parametrize("task", BOX_TASKS)
def test_every_box_task_returns_a_design_its_own_validator_accepts(task):
    """Test: every box task returns a design its own validator accepts."""
    spec = bg.TASKS[task]
    environment = bg.TaskEnvironment(spec, 20261004)
    space = ds.SearchSpace(spec, environment, bg.encode_design)
    # push every coordinate past its upper bound: the projection must still be inside the box and env-valid
    def outward(rows):
        """Test helper: outward."""
        rows = np.asarray(rows, dtype=float).reshape(-1, space.dim)
        return np.sum((rows - space.low) / (space.high - space.low), axis=1)
    pool = space.to_design(np.full((1, space.dim), 0.5))
    pool = np.asarray([space.project(p) for p in pool])
    row, _, _record = ds.search(space, outward, pool, outward(pool), seed=7)
    assert environment.accepts(bg.encode_design(spec, row))
    assert np.all(row >= space.low) and np.all(row <= space.high)
    assert np.array_equal(row[space.integer], np.rint(row[space.integer]))
    if task == "hyperbolic":
        assert row[0] <= row[1] and row[2] >= 1


@pytest.mark.usefixtures("require_boxinggym_source")
@pytest.mark.parametrize("cell,policy,objective", [
    ("111", None, "eig"), ("111", "structure_eig", "structure_eig"),
    ("111", "parameter_eig", "parameter_eig"), ("102", "epig", "epig"),
])
def test_runner_records_each_arm_on_a_box_task_and_observes_the_search_design(tmp_path, monkeypatch, cell, policy,
                                                                              objective):
    """Test: runner records each arm on a box task and observes the search design."""
    adapter = bg_stub_fits(monkeypatch)
    out = tmp_path / f"ascent_{objective}"
    extra = {"design_policy": policy} if policy else {}
    adapter.run(bg_run_args(out, task="hyperbolic", cell=cell, rounds=1, design_search="ascent",
                            run_label="development", **extra))
    config = json.loads((out / "config.json").read_text())
    assert config["design_search_settings"]["design_search_method"] == ds.METHOD
    assert config["design_search_settings"]["mode"] == "ascent"
    assert config["design_search_settings"]["objective"] == objective
    assert config["design_search_settings"]["constants"] == ds.design_search_constants()
    settings = config["label_receipt"]["binding"]["settings"]
    assert settings["design_search"] == "ascent" and settings["design_search_method"] == ds.METHOD
    acquisition = json.loads((out / "acquisition" / "t01.json").read_text())
    record = acquisition["design_search"]
    assert record["objective"] == objective and record["mode"] == "ascent"
    assert acquisition["chosen"] == record["chosen"]["design"]
    assert "pool_chosen" in acquisition and acquisition["pool_chosen"] != acquisition["chosen"] or \
        record["gain_over_pool_argmax"] == 0.0
    assert record["chosen"]["value"] is None or record["chosen"]["value"] >= record["pool_argmax"]["value_rescored"] \
        - ds.TIE_ATOL
    assert len(acquisition["rows"]) == config["design_pool_size"]
    observed = [json.loads(line) for line in (out / "observations.jsonl").read_text().splitlines()]
    round_one = next(row for row in observed if row["round"] == 1)
    assert round_one["x"] == acquisition["chosen"]
    assert round_one["design"] == bg.encode_design(bg.TASKS["hyperbolic"], np.asarray(acquisition["chosen"]))
    i_r, d_r, days = acquisition["chosen"]
    assert float(i_r).is_integer() and i_r <= d_r and days >= 1


@pytest.mark.usefixtures("require_boxinggym_source")
def test_runner_default_path_has_no_design_search_trace(tmp_path, monkeypatch):
    """Test: runner default path has no design search trace."""
    adapter = bg_stub_fits(monkeypatch)
    out = tmp_path / "default"
    adapter.run(bg_run_args(out, task="hyperbolic", cell="111", rounds=1, run_label="development"))
    text = (out / "config.json").read_text() + (out / "acquisition" / "t01.json").read_text()
    assert "design_search" not in text and "pool_chosen" not in text


@pytest.mark.usefixtures("require_boxinggym_source")
def test_runner_scalar_task_scores_each_design_in_its_own_call(tmp_path, monkeypatch):
    """Test: runner scalar task scores each design in its own call."""
    adapter = bg_scalar_stub_fits(monkeypatch)
    out = tmp_path / "location"
    adapter.run(bg_run_args(out, task="location", cell="111", rounds=1, design_search="ascent"))
    record = json.loads((out / "acquisition" / "t01.json").read_text())["design_search"]
    assert record["chosen"]["value"] is None or record["chosen"]["value"] >= record["pool_argmax"]["value_rescored"] \
        - ds.TIE_ATOL
    assert record["bounds"]["integer_inputs"] == []


@pytest.mark.usefixtures("require_boxinggym_source")
def test_runner_survival_is_exhaustive_not_ascent(tmp_path, monkeypatch):
    """Test: runner survival is exhaustive not ascent."""
    adapter = bg_stub_fits(monkeypatch)
    out = tmp_path / "survival"
    adapter.run(bg_run_args(out, task="survival", cell="111", rounds=1, design_search="ascent",
                            run_label="development", stable_log1mexp=True))
    config = json.loads((out / "config.json").read_text())
    assert config["design_search_settings"]["mode"] == "exhaustive"
    acquisition = json.loads((out / "acquisition" / "t01.json").read_text())
    record = acquisition["design_search"]
    assert record["mode"] == "exhaustive" and record["chosen"]["in_pool"] is True
    assert acquisition["chosen"] == acquisition["pool_chosen"] == record["chosen"]["design"]


@pytest.mark.usefixtures("require_boxinggym_source")
@pytest.mark.parametrize("overrides,message", [
    ({"task": "hyperbolic", "cell": "110"}, "requires an active"),
    ({"task": "death", "cell": "110"}, "requires an active"),
    ({"task": "moral_machines", "cell": "111"}, "does not apply to task moral_machines"),
    ({"task": "emotion", "cell": "111"}, "does not apply to task emotion"),
])
def test_runner_refuses_ascent_where_it_cannot_apply_before_any_output(tmp_path, monkeypatch, overrides, message):
    """Test: runner refuses ascent where it cannot apply before any output."""
    adapter = bg_stub_fits(monkeypatch)
    out = tmp_path / "refused"
    with pytest.raises(ValueError, match=message):
        adapter.run(bg_run_args(out, rounds=1, design_search="ascent", run_label="development", **overrides))
    assert not out.exists()
