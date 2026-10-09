"""The opt-in BoxingGym expected predictive information gain (EPIG) target count, `--epig-target-points N`.

Unstated, every check, message and record keeps the default of 32 targets. Stated, only the
count changes: the same native goal draws (or finite-pool resampling) on the same seed stream,
so the first 32 of N targets are the 32 targets of an unstated run.
"""
import argparse
import contextlib
import inspect
import json
import pickle
import random
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from . import boxinggym_core as bg
from . import count_epig
from . import count_inference as inference
from .core_count import mock_parameters, seed_for
from .count_inference import Fit
from .count_models import Hypothesis

# The two refusal messages exactly as the unstated (default 32-target) path words them.
BASE_CONFIG_MESSAGE = "EPIG requires 32 finite target inputs sampled once per run"
BASE_SAMPLER_MESSAGE = "native goal sampler must supply 32 finite EPIG coordinate rows"


@pytest.fixture(autouse=True)
def _restore_random_state():
    """Restore the NumPy and Python random states after each test."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    yield
    np.random.set_state(numpy_state)
    random.setstate(python_state)


def _assert_numpy_state_equal(before, after):
    """Assert that two NumPy random generator states are identical."""
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_defaults_stay_at_the_module_constant():
    """Check that defaults stay at the module constant."""
    assert inference.EPIG_TARGET_POINTS == 32
    assert inspect.signature(bg.sample_epig_targets).parameters["count"].default == 32
    assert inspect.signature(inference.epig_config).parameters["target_count"].default == 32


class _CappedGoal:
    """A synthetic native goal that stops adding held-out points after `cap` questions."""

    cap = 10

    def __init__(self, env):
        """Store the environment and start with no held-out points."""
        self.env = env
        self.eval_points = []

    def get_goal_eval_question(self, include_prior):
        """Add one held-out point unless the cap is reached and return a synthetic question."""
        if len(self.eval_points) < self.cap:
            self.eval_points.append((float(np.random.random()), float(random.random())))
        return "synthetic question", 0


def _synthetic_environment(cap):
    """Build a synthetic environment wrapper around a goal that stops after `cap` held-out points."""
    native = SimpleNamespace()
    goal = type("Goal", (_CappedGoal,), {"cap": cap})(native)
    return SimpleNamespace(env=native, goal=goal, eval_inputs=lambda point: list(point))


def test_a_capped_native_goal_is_refused_not_padded():
    """Check that a capped native goal is refused not padded."""
    spec = bg.TASKS["location"]
    with pytest.raises(ValueError, match=re.escape("must supply 128 finite EPIG coordinate rows")):
        bg.sample_epig_targets(spec, _synthetic_environment(10), None, 5, count=128)
    with pytest.raises(ValueError) as unstated:
        bg.sample_epig_targets(spec, _synthetic_environment(10), None, 5)
    assert str(unstated.value) == BASE_SAMPLER_MESSAGE
    # The same goal with room for every draw: only the count differs.
    short = bg.sample_epig_targets(spec, _synthetic_environment(10**6), None, 5)
    long = bg.sample_epig_targets(spec, _synthetic_environment(10**6), None, 5, count=128)
    assert short.shape == (32, 2) and long.shape == (128, 2)
    np.testing.assert_array_equal(long[:32], short)


@pytest.mark.parametrize("text,value", [("1", 1), ("32", 32), ("128", 128), (" 128 ", 128), ("4096", 4096)])
def test_target_count_parser_accepts_integers_in_range(text, value):
    """Check that target count parser accepts integers in range."""
    assert bg.parse_epig_target_points(text) == value


@pytest.mark.parametrize("text", ["0", "-1", "4097", "1.5", "128.0", "abc", "", "1e3", "+5", "0x80",
                                  "١٢٨", "²"])
def test_target_count_parser_refuses_everything_else(text):
    """Check that target count parser refuses everything else."""
    with pytest.raises(argparse.ArgumentTypeError, match=r"integer in 1\.\.4096"):
        bg.parse_epig_target_points(text)


def _parse(monkeypatch, tmp_path, *extra):
    """The runner's own parser, with `run` replaced by a recorder."""
    seen = []
    monkeypatch.setattr(bg, "run", seen.append)
    monkeypatch.setattr(sys, "argv", ["boxinggym_core", "--output", str(tmp_path / "out"),
                                      "--task", "hyperbolic", "--proposer", "mock", *extra])
    bg.main()
    return seen[0]


def test_cli_option_leaves_no_key_unless_stated(monkeypatch, tmp_path):
    """Check that cli option leaves no key unless stated."""
    unstated = _parse(monkeypatch, tmp_path, "--cell", "102", "--design-policy", "epig")
    assert "epig_target_points" not in vars(unstated)
    stated = _parse(monkeypatch, tmp_path, "--cell", "102", "--design-policy", "epig",
                    "--epig-target-points", "128")
    assert type(stated.epig_target_points) is int and stated.epig_target_points == 128
    assert {k: v for k, v in vars(stated).items() if k != "epig_target_points"} == vars(unstated)
    with pytest.raises(SystemExit) as refused:
        _parse(monkeypatch, tmp_path, "--cell", "102", "--epig-target-points", "0")
    assert refused.value.code == 2


@pytest.mark.parametrize("cell", ["111", "110"])
def test_option_is_refused_on_non_epig_arms_before_any_output(monkeypatch, tmp_path, cell):
    """Check that option is refused on non epig arms before any output."""
    run = bg.run
    args = _parse(monkeypatch, tmp_path, "--cell", cell, "--epig-target-points", "128")
    with pytest.raises(ValueError, match="applies only to the epig design policy"):
        run(args)
    assert not (tmp_path / "out").exists()


class _Reached(Exception):
    """Sentinel exception raised once execution reaches the point under test."""


def test_option_passes_the_policy_check_on_the_epig_arm(monkeypatch, tmp_path):
    """Check that option passes the policy check on the epig arm."""
    run = bg.run
    args = _parse(monkeypatch, tmp_path, "--cell", "102", "--epig-target-points", "128")

    def reached(*_):
        """Raise the sentinel exception."""
        raise _Reached

    # compat_config is the first call after the check, still before any output exists.
    monkeypatch.setattr(bg, "compat_config", reached)
    with pytest.raises(_Reached):
        run(args)
    assert not (tmp_path / "out").exists()


def test_config_records_the_stated_count_and_keeps_the_unstated_record():
    """Check that config records the stated count and keeps the unstated record."""
    targets = np.random.default_rng(7).uniform(size=(128, 3))
    record = inference.epig_config(targets, 7, "synthetic_rule", target_count=128)
    assert record["epig_target_points"] == 128
    assert record["epig_targets"] == targets.tolist() and len(record["epig_targets"]) == 128
    assert record["epig_targets_sha256"] == inference.coordinate_set_sha256(targets)
    with pytest.raises(ValueError, match=re.escape("EPIG requires 128 finite target inputs")):
        inference.epig_config(targets[:32], 7, "synthetic_rule", target_count=128)
    with pytest.raises(ValueError) as unstated:
        inference.epig_config(targets, 7, "synthetic_rule")
    assert str(unstated.value) == BASE_CONFIG_MESSAGE
    default = inference.epig_config(targets[:32], 7, "synthetic_rule")
    explicit = inference.epig_config(targets[:32], 7, "synthetic_rule", target_count=32)
    assert json.dumps(default) == json.dumps(explicit) and default["epig_target_points"] == 32


def _hyperbolic_fit():
    """A synthetic posterior over the hyperbolic task's own 3-input interface."""
    spec = bg.TASKS["hyperbolic"]
    interface = spec.interface()
    namespace = tuple(item["name"] for item in interface["inputs"]) + tuple(interface["derived_inputs"])
    expression = spec.additive_structure
    hypothesis = Hypothesis(expression, "bernoulli", mock_parameters(expression), namespace)
    rng = np.random.default_rng(3)
    particles = {name: rng.normal(0., 1., 60) for name, _ in hypothesis.parameters}
    return Fit(hypothesis, particles, 0.), interface


def _synthetic_hyperbolic_rows(rng, count):
    """Draw synthetic hyperbolic-task rows (immediate amount, delayed amount, days)."""
    immediate = rng.integers(1, 300, count)
    delayed = rng.integers(immediate + 1, 301)
    days = rng.integers(1, 365, count)
    return np.column_stack([immediate, delayed, days]).astype(float)


def test_epig_score_is_the_mean_information_over_all_stated_targets(monkeypatch):
    """Check that epig score is the mean information over all stated targets."""
    fit, interface = _hyperbolic_fit()
    rng = np.random.default_rng(11)
    designs, targets = _synthetic_hyperbolic_rows(rng, 12), _synthetic_hyperbolic_rows(rng, 128)
    seen = []
    original = count_epig.predictive_information

    def spy(q_pmfs, x_pmfs, particle_weights, q_weights):
        """Spy wrapper that records arguments and delegates to the original function."""
        seen.append((np.shape(q_pmfs), np.shape(x_pmfs), np.asarray(q_weights).copy()))
        return original(q_pmfs, x_pmfs, particle_weights, q_weights)

    monkeypatch.setattr(count_epig, "predictive_information", spy)

    def score(points):
        """Return the EPIG score rows for the given target points."""
        return inference.choose_epig([fit], designs, 41, 128, interface, interface["trials"],
                                     target_points=points)[1]

    rows = score(targets)
    (q_shape, x_shape, q_weights), = seen
    assert q_shape[1] == 128 and x_shape[1] == 12
    np.testing.assert_array_equal(q_weights, np.full(128, 1 / 128))
    # EPIG is the target-weighted mean of I(Y_x; Y_x*): four disjoint 32-target scores on the
    # same particle bank average to the 128-target score.
    chunks = [score(targets[start:start + 32]) for start in range(0, 128, 32)]
    epig = np.array([row["epig"] for row in rows])
    joint_eig = np.array([row["joint_eig"] for row in rows])
    chunk_epig = np.array([[row["epig"] for row in chunk] for chunk in chunks])
    np.testing.assert_allclose(epig, chunk_epig.mean(0), rtol=1e-12, atol=1e-15)
    for chunk in chunks:
        np.testing.assert_allclose([row["joint_eig"] for row in chunk], joint_eig, rtol=1e-12, atol=1e-15)
    assert not np.allclose(epig, chunk_epig[0], rtol=1e-6, atol=0)
    assert np.all(epig >= -1e-14) and np.all(epig <= joint_eig + 1e-12)


@pytest.mark.parametrize("task,width", [("hyperbolic", 3), ("location", 2), ("survival", 2)])
@pytest.mark.usefixtures("require_boxinggym_source")
def test_real_task_supplies_128_finite_targets_on_the_same_stream(task, width):
    """Check that real task supplies 128 finite targets on the same stream."""
    spec = bg.TASKS[task]
    world_seed = seed_for("boxinggym_world", task, 11)
    evaluation_seed = seed_for("boxinggym_evaluation", task, 11, "dev")
    target_seed = seed_for("epig_targets", 11)
    with contextlib.ExitStack() as stack:
        if spec.openai_narration:
            stack.enter_context(bg.openai_narration_stub(spec.openai_narration_module))
        stack.enter_context(bg.env_seed_injection(world_seed))
        environment = bg.TaskEnvironment(spec, world_seed)
        evaluation_x, _ = environment.evaluation_points(3, evaluation_seed)
        pool = bg.build_design_pool(spec, environment, bg.design_pool_target(spec, 16, 3))
        before_points = pickle.dumps(environment.goal.eval_points)
        numpy_state, python_state = np.random.get_state(), random.getstate()
        unstated = bg.sample_epig_targets(spec, environment, pool, target_seed)
        stated = bg.sample_epig_targets(spec, environment, pool, target_seed, count=128)
        again = bg.sample_epig_targets(spec, environment, pool, target_seed, count=128)
        assert pickle.dumps(environment.goal.eval_points) == before_points
        assert environment.observation_calls == 0
        _assert_numpy_state_equal(numpy_state, np.random.get_state())
        assert random.getstate() == python_state
    assert unstated.shape == (32, width) and stated.shape == (128, width)
    assert np.isfinite(stated).all()
    np.testing.assert_array_equal(stated, again)
    np.testing.assert_array_equal(stated[:32], unstated)
    if task == "hyperbolic":
        # The goal's own draws: integer iR < dR <= 300 and 1 <= D <= 364, never clipped to the pool.
        assert np.array_equal(stated, np.round(stated))
        immediate, delayed, days = stated.T
        assert np.all((1 <= immediate) & (immediate < delayed) & (delayed <= 300))
        assert np.all((1 <= days) & (days <= 364))
    finite = spec.environment_fixed_pool or spec.environment_pool_source == "categorical"
    measure = bg.BOXING_FINITE_TARGET_MEASURE if finite else bg.BOXING_GOAL_TARGET_MEASURE
    record = inference.epig_config(stated, target_seed, measure, evaluation_points=evaluation_x,
                                   target_measure_note=bg.BOXING_FINITE_TARGET_NOTE if finite else None,
                                   target_count=128)
    assert record["epig_target_points"] == 128 and len(record["epig_targets"]) == 128
    assert record["epig_target_measure"] == measure
