"""Independent target-measure isolation and coordinate-provenance witnesses."""
import contextlib
import copy
import json
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest

from . import boxinggym_core as bg
from . import count_inference as inference
from .core_count import seed_for


def _assert_numpy_state_equal(before, after):
    """Assert that two NumPy random generator states are identical."""
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.fixture(autouse=True)
def _restore_random_state():
    """Restore the NumPy and Python random states after each test."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    yield
    np.random.set_state(numpy_state)
    random.setstate(python_state)


class _MutatingGoal:
    """Synthetic native goal that mutates the environment cache, call counter and global random state."""
    def __init__(self, environment):
        """Record construction side effects on the environment and optionally raise a constructor failure."""
        self.env = environment
        self.eval_points = []
        self.pointer = 0
        environment.cache.append("constructed")
        environment.calls += 1
        np.random.random()
        random.random()
        bg.pymc.draw("goal-constructor")
        if environment.failure == "constructor":
            raise RuntimeError("synthetic constructor failure")

    def get_goal_eval_question(self, include_prior):
        """Draw one synthetic held-out point, mutating environment state and global random state."""
        assert include_prior is True
        self.env.cache.append(self.pointer)
        self.env.calls += 1
        self.pointer += 1
        bg.pymc.sample_prior_predictive("goal-question")
        value = self.env.truth + np.random.random() + random.random()
        if self.env.constant is not None:
            value = self.env.constant
        if self.env.failure == "draw" and self.pointer == 5:
            raise RuntimeError("synthetic draw failure")
        self.eval_points.append((value, -99.0))
        return "synthetic question"


def _synthetic_environment(monkeypatch, failure=None, constant=None):
    """Build a synthetic environment wrapper and the list of seeds seen by the patched backend."""
    calls = []

    def backend(*args, random_seed=None, **kwargs):
        """Patched sampling backend that records and returns the injected seed."""
        calls.append(random_seed)
        return random_seed

    monkeypatch.setattr(bg.pymc, "draw", backend)
    monkeypatch.setattr(bg.pymc, "sample_prior_predictive", backend)
    native = SimpleNamespace(truth=37.0, cache=[], calls=0, failure=None, constant=constant)
    goal = _MutatingGoal(native)
    goal.eval_points[:] = [(123.0, -9.0), (456.0, -8.0)]
    goal.pointer = 7
    native.failure = failure
    wrapper = SimpleNamespace(env=native, goal=goal, eval_inputs=lambda point: point[:1])
    calls.clear()
    return wrapper, calls


@pytest.mark.parametrize("failure", [None, "constructor", "draw"])
def test_native_sampler_isolates_world_goal_rng_and_outer_seed_counter(monkeypatch, failure):
    """Check that native sampler isolates world goal rng and outer seed counter."""
    environment, backend_seeds = _synthetic_environment(monkeypatch, failure)
    original_world = copy.deepcopy(vars(environment.env))
    points_object = environment.goal.eval_points
    original_points = pickle.dumps(points_object)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    original_draw = bg.pymc.draw
    original_prior = bg.pymc.sample_prior_predictive
    outer_seed, target_seed = 381, 991
    with bg.env_seed_injection(outer_seed) as outer_log:
        outer_draw, outer_prior = bg.pymc.draw, bg.pymc.sample_prior_predictive
        bg.pymc.draw("before")
        if failure:
            with pytest.raises(RuntimeError, match="synthetic .* failure"):
                bg.sample_epig_targets(bg.TASKS["death"], environment, None, target_seed)
        else:
            targets = bg.sample_epig_targets(bg.TASKS["death"], environment, None, target_seed)
            again = bg.sample_epig_targets(bg.TASKS["death"], environment, None, target_seed)
            np.testing.assert_array_equal(targets, again)
            assert targets.shape == (32,)
            # The realized truth survives; no clipping to death's acquisition range occurs.
            assert np.all((targets > 37.0) & (targets < 39.0))
            assert len(np.unique(targets)) == 32
        assert bg.pymc.draw is outer_draw
        assert bg.pymc.sample_prior_predictive is outer_prior
        bg.pymc.sample_prior_predictive("after")
    expected_seeds = [int(np.random.SeedSequence([outer_seed, k]).generate_state(1)[0])
                      for k in range(2)]
    assert [row["call"] for row in outer_log] == [0, 1]
    assert [row["injected_seed"] for row in outer_log] == expected_seeds
    assert backend_seeds[0] == expected_seeds[0] and backend_seeds[-1] == expected_seeds[1]
    assert all(type(value) is int for value in backend_seeds)
    assert bg.pymc.draw is original_draw and bg.pymc.sample_prior_predictive is original_prior
    assert vars(environment.env) == original_world
    assert environment.goal.env is environment.env
    assert environment.goal.eval_points is points_object
    assert pickle.dumps(environment.goal.eval_points) == original_points
    assert environment.goal.pointer == 7
    _assert_numpy_state_equal(numpy_state, np.random.get_state())
    assert random.getstate() == python_state


def test_native_collisions_are_kept_and_counted_without_resampling(monkeypatch):
    """Check that native collisions are kept and counted without resampling."""
    environment, backend_seeds = _synthetic_environment(monkeypatch, constant=123.0)
    targets = bg.sample_epig_targets(bg.TASKS["death"], environment, None, 771)
    assert targets.tolist() == [123.0] * 32
    assert len(backend_seeds) == 33  # One constructor plus exactly 32 native questions.
    record = inference.epig_config(targets, 771, bg.BOXING_GOAL_TARGET_MEASURE,
                                   evaluation_points=np.array([123.0, 123.0, 456.0]))
    assert record["epig_target_evaluation_overlap_count"] == 32
    assert record["epig_target_disjoint_from_evaluation"] is False
    assert record["epig_targets"] == [123.0] * 32
    assert "no_resampling" in record["epig_target_overlap_rule"]


@pytest.mark.parametrize("task", ["survival", "irt"])
def test_finite_pool_exception_uses_only_pool_without_touching_environment(task):
    """Check that finite pool exception uses only pool without touching environment."""
    rows = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
    pool = bg.DesignPool(rows, ("a", "b", "c"), False)
    before = rows.copy()
    numpy_state, python_state = np.random.get_state(), random.getstate()
    targets = bg.sample_epig_targets(bg.TASKS[task], object(), pool, 782)
    again = bg.sample_epig_targets(bg.TASKS[task], object(), pool, 782)
    np.testing.assert_array_equal(targets, again)
    assert targets.shape == (32, 2)
    assert all(tuple(row) in {tuple(point) for point in rows} for row in targets)
    assert not np.shares_memory(targets, rows)
    np.testing.assert_array_equal(rows, before)
    _assert_numpy_state_equal(numpy_state, np.random.get_state())
    assert random.getstate() == python_state
    record = inference.epig_config(targets, 782, bg.BOXING_FINITE_TARGET_MEASURE,
                                   evaluation_points=np.array([[20.0, 1.0]]),
                                   target_measure_note=bg.BOXING_FINITE_TARGET_NOTE)
    assert record["epig_target_measure"] == "uniform_resampling_of_finite_design_pool"
    assert record["epig_target_measure_note"] == "finite_design_set; held_out_measure_differs"
    assert record["epig_target_disjoint_from_evaluation"] is True


def test_coordinate_digest_normalizes_layout_endian_scalar_shape_and_signed_zero():
    """Check that coordinate digest normalizes layout endian scalar shape and signed zero."""
    values = np.array([[0.0, 1.25], [3.5, -8.0], [7.0, 9.0]])
    reference = inference.coordinate_set_sha256(values)
    variants = [np.asfortranarray(values), values.astype(">f8"),
                np.repeat(values, 2, axis=0)[::2]]
    signed_zero = values.copy()
    signed_zero[0, 0] = -0.0
    variants.append(signed_zero)
    for variant in variants:
        original_bytes = variant.tobytes()
        assert inference.coordinate_set_sha256(variant) == reference
        assert variant.tobytes() == original_bytes
    scalar = np.array([-0.0, 1.25, 3.5])
    assert inference.coordinate_set_sha256(scalar) == inference.coordinate_set_sha256(scalar[:, None])


def test_coordinate_digest_preserves_row_order_columns_multiplicity_and_shape():
    """Check that coordinate digest preserves row order columns multiplicity and shape."""
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    variants = [values, values[::-1], values[:, ::-1], values.reshape(2, 3),
                np.concatenate([values, values[:1]])]
    assert len({inference.coordinate_set_sha256(value) for value in variants}) == len(variants)


@pytest.mark.parametrize("values", [0.0, np.zeros((1, 1, 1)), [np.nan], [np.inf], [-np.inf]])
def test_coordinate_digest_rejects_invalid_coordinate_arrays(values):
    """Check that coordinate digest rejects invalid coordinate arrays."""
    with pytest.raises(ValueError, match="finite point-by-input array"):
        inference.coordinate_set_sha256(values)


def test_epig_metadata_counts_exact_target_draws_and_keeps_original_coordinates():
    """Check that epig metadata counts exact target draws and keeps original coordinates."""
    targets = np.tile([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], (8, 1))
    evaluation = np.array([[0.0, 1.0], [-0.0, 1.0], [2.0, np.nextafter(3.0, 4.0)]])
    before = targets.copy()
    record = inference.epig_config(targets, 983, "native_rule", evaluation_points=evaluation,
                                   target_measure="native_measure")
    assert record["epig_target_measure"] == "native_measure"
    assert record["epig_target_seed_stream"] == record["epig_targets_seed_stream"] == "epig_targets"
    assert record["epig_targets_seed"] == 983
    assert record["epig_targets_sampling_rule"] == "native_rule"
    assert record["epig_target_evaluation_overlap_count"] == 8
    assert record["epig_target_disjoint_from_evaluation"] is False
    assert record["epig_targets_sha256"] == inference.coordinate_set_sha256(targets)
    assert record["epig_targets_sha256_rule"] == inference.COORDINATE_SHA256_RULE
    np.testing.assert_array_equal(targets, before)
    np.testing.assert_array_equal(record["epig_targets"], before)
    assert "epig_target_measure_note" not in record


@pytest.mark.parametrize("evaluation", [np.zeros((3, 2)), np.array([np.nan])])
def test_epig_metadata_refuses_incompatible_evaluation_coordinates(evaluation):
    """Check that epig metadata refuses incompatible evaluation coordinates."""
    with pytest.raises(ValueError, match="finite coordinate columns"):
        inference.epig_config(np.arange(32.0), 14, "native_rule", evaluation_points=evaluation)


NATIVE_TASKS = ("death", "peregrines", "dugongs", "hyperbolic", "location",
                "moral_machines", "emotion", "irt", "survival")


@pytest.mark.parametrize("task", NATIVE_TASKS)
@pytest.mark.usefixtures("require_boxinggym_source")
def test_native_task_targets_preserve_heldout_and_emit_measure_receipt(task, tmp_path):
    """Check that native task targets preserve heldout and emit measure receipt."""
    spec = bg.TASKS[task]
    world_seed = seed_for("boxinggym_world", task, 11)
    evaluation_seed = seed_for("boxinggym_evaluation", task, 11, "dev")
    target_seed = seed_for("epig_targets", 11)
    with contextlib.ExitStack() as stack:
        if spec.openai_narration:
            stack.enter_context(bg.openai_narration_stub(spec.openai_narration_module))
        outer_log = stack.enter_context(bg.env_seed_injection(world_seed))
        environment = bg.TaskEnvironment(spec, world_seed)
        evaluation_x, heldout = environment.evaluation_points(3, evaluation_seed)
        if spec.environment_pool_source == "sampled":
            np.random.seed(seed_for("boxinggym_design_pool", task, 11))
        pool = bg.build_design_pool(spec, environment, bg.design_pool_target(spec, 16, 3))
        before_points = pickle.dumps(environment.goal.eval_points)
        before_goal_values = {name: copy.deepcopy(value) for name, value in vars(environment.goal).items()
                              if value is not environment.env and name != "eval_points"}
        before_goal_blob = pickle.dumps(before_goal_values)
        before_x, before_y = evaluation_x.tobytes(), heldout.tobytes()
        before_log = copy.deepcopy(outer_log)
        numpy_state, python_state = np.random.get_state(), random.getstate()
        targets = bg.sample_epig_targets(spec, environment, pool, target_seed)
        repeated = bg.sample_epig_targets(spec, environment, pool, target_seed)
        np.testing.assert_array_equal(targets, repeated)
        assert len(targets) == 32 and np.isfinite(targets).all()
        assert pickle.dumps(environment.goal.eval_points) == before_points
        after_goal_values = {name: value for name, value in vars(environment.goal).items()
                             if value is not environment.env and name != "eval_points"}
        assert pickle.dumps(after_goal_values) == before_goal_blob
        assert evaluation_x.tobytes() == before_x and heldout.tobytes() == before_y
        assert outer_log == before_log
        assert environment.observation_calls == 0
        _assert_numpy_state_equal(numpy_state, np.random.get_state())
        assert random.getstate() == python_state
    finite = spec.environment_fixed_pool or spec.environment_pool_source == "categorical"
    measure = bg.BOXING_FINITE_TARGET_MEASURE if finite else bg.BOXING_GOAL_TARGET_MEASURE
    note = bg.BOXING_FINITE_TARGET_NOTE if finite else None
    record = inference.epig_config(targets, target_seed, measure, evaluation_points=evaluation_x,
                                   target_measure_note=note)
    target_rows = targets.reshape(32, -1)
    evaluation_rows = evaluation_x.reshape(len(evaluation_x), -1)
    overlap = sum(any(np.array_equal(row, point) for point in evaluation_rows) for row in target_rows)
    assert record["epig_target_evaluation_overlap_count"] == overlap
    assert record["epig_target_disjoint_from_evaluation"] is (overlap == 0)
    if task in ("death", "dugongs", "location", "emotion"):
        assert overlap == 0  # Independent continuous-coordinate fixtures are strictly disjoint.
    assert record["epig_target_measure"] == measure
    if finite:
        assert record["epig_target_measure_note"] == bg.BOXING_FINITE_TARGET_NOTE
    else:
        assert "epig_target_measure_note" not in record
    (tmp_path / "Q1_measure.json").write_text(json.dumps({
        "task": task, "target_count": len(targets), "epig_target_measure": measure,
        "epig_target_measure_note": note, "target_seed": target_seed,
        "epig_targets_sha256": record["epig_targets_sha256"],
        "epig_target_disjoint_from_evaluation": record["epig_target_disjoint_from_evaluation"],
        "epig_target_evaluation_overlap_count": overlap,
        "heldout_bytes_unchanged": True, "outer_seed_log_unchanged": True,
        "numpy_python_state_unchanged": True, "paid_api_calls": 0,
    }, indent=2) + "\n")
