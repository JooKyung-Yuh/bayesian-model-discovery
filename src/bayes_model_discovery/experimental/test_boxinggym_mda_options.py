"""MDA policies and scoring-only generating means, using synthetic fixed worlds."""

import csv
import importlib
import json
import os
from types import SimpleNamespace


def _external_src(name):
    """The external checkouts the baseline worktree run needs (boxing-gym src,
    LLM-AutoSciLab) come from the environment - no machine-specific path lives
    in the test (the same rule the smoke tooling follows)."""
    value = os.environ.get(name, "")
    if not value:
        pytest.skip(f"set {name} to run the baseline-worktree comparison here")
    return value


import numpy as np
import pytest
from scipy.special import expit

from . import boxinggym_core as bg
from .test_named_inputs import bg_run_args, bg_stub_fits


@pytest.mark.parametrize("task,count", [("death", 2), ("location", 3), ("irt", 2)])
def test_random_d0_is_reproducible_distinct_and_uses_its_own_rng(task, count):
    """Random D0 selection is reproducible, distinct and leaves the global NumPy RNG state untouched."""
    spec = bg.TASKS[task]
    rows = np.arange(20*len(spec.input_names)).reshape(20, -1).astype(float)
    pool = SimpleNamespace(rows=rows, handles=[str(i) for i in range(20)])
    np.random.seed(17)
    state = np.random.get_state()
    selected = bg.select_d0_designs(spec, pool, "random2", 73)
    assert len(selected) == count
    assert len({handle for _, handle in selected}) == count
    expected = np.random.default_rng(bg.seed_for("boxinggym_d0", task, 73)).choice(
        20, size=count, replace=False
    )
    assert [handle for _, handle in selected] == [str(i) for i in expected]
    assert [h for _, h in bg.select_d0_designs(spec, pool, "random2", 73)] == [
        h for _, h in selected
    ]
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_q_exclusion_is_exact_in_all_declared_dimensions():
    """The observed-point mask matches exactly in every declared input dimension."""
    q = np.array([[0., 1.], [0., 2.], [1., 1.], [0., 1.]])
    assert bg.unobserved_q_mask(q, [[0., 1.]]).tolist() == [False, True, True, False]
    assert bg.unobserved_q_mask([1., 1.+1e-12, 2.], [1.]).tolist() == [False, True, True]
    assert bg.unobserved_q_mask(q, []).all()


def test_mda_score_uses_truth_variance_and_running_minimum():
    """The MDA score row normalizes by the truth variance and tracks the running minimum."""
    first = bg.mda_score_row(2, [1., 3.], [0., 2.], "")
    assert first == {"x": 2, "nmse_truth": 1., "brier": "", "best_so_far_nmse_truth": 1.}
    second = bg.mda_score_row(3, [2., 4.], [0., 2.], "", first["best_so_far_nmse_truth"])
    assert second["nmse_truth"] == 4 and second["best_so_far_nmse_truth"] == 1
    third = bg.mda_score_row(4, [0., 2.], [0., 2.], "", second["best_so_far_nmse_truth"])
    assert third["best_so_far_nmse_truth"] == 0
    assert bg.mda_score_row(1, [1., 1.], [1., 1.], "")["nmse_truth"] == "not_available"
    assert bg.mda_score_row(1, [1., 1.], None, "")["nmse_truth"] == "not_available"


def test_emotion_without_generating_parameters_is_explicitly_unavailable():
    """Emotion without generating parameters reports no truth mean."""
    environment = SimpleNamespace(env=SimpleNamespace())
    assert bg.TASKS["emotion"].truth_mean(environment, np.zeros((2, 7))) is None


def test_moral_mean_respects_the_native_choice_reversal():
    """The moral_machines truth mean follows the native choice reversal."""
    env = SimpleNamespace(alpha=1., beta_group=.5, **{
        "beta_"+name: 0. for name in (
            "human_count", "gender", "age", "social_status", "fitness", "species", "intervention"
        )
    })
    truth = bg.TASKS["moral_machines"].truth_mean(SimpleNamespace(env=env), np.zeros((2, 7)))
    np.testing.assert_allclose(truth, expit(-1.5))


def test_dugong_observable_mean_includes_folding_near_zero():
    """The dugongs observable mean accounts for folding near zero."""
    env = SimpleNamespace(a=0., b=0., l=.8)
    mean = bg.TASKS["dugongs"].truth_mean(SimpleNamespace(env=env), np.array([0., 1.]))
    np.testing.assert_allclose(mean, .25*np.sqrt(2/np.pi))


def test_emotion_observable_mean_accounts_for_integer_clipping():
    """The emotion observable mean accounts for integer clipping of the response."""
    env = SimpleNamespace(alpha=np.full(8, 4.5), beta_win=np.zeros(8), beta_PE=np.zeros(8),
                          beta_absPE=np.zeros(8), sigma=np.full(8, .01))
    mean = bg.TASKS["emotion"].truth_mean(SimpleNamespace(env=env), np.zeros((2, 7)))
    np.testing.assert_allclose(mean, 4.)


@pytest.mark.usefixtures("require_boxinggym_source")
def test_sidecar_and_config_follow_random_d0_without_changing_points_contract(tmp_path, monkeypatch):
    """The MDA sidecar and config follow a random D0 while the points.csv contract stays unchanged."""
    adapter = bg_stub_fits(monkeypatch)
    out = tmp_path / "random_death"
    adapter.run(bg_run_args(out, cell="000", rounds=2, d0_policy="random2", q_policy="native"))
    config = json.loads((out/"config.json").read_text())
    assert config["d0_policy"] == "random2" and config["d0_count"] == 2
    assert config["d0_seed_label"] == "boxinggym_d0"
    assert config["q_policy"] == "native" and config["q_excluded_count"] == 0
    assert config["truth_mean_available"] is True
    with (out/"points.csv").open() as stream:
        points_reader = csv.DictReader(stream)
        assert tuple(points_reader.fieldnames) == bg.BOXINGGYM_POINTS_COLUMNS
        assert len(points_reader.fieldnames) == 25
        points = list(points_reader)
    with (out/"points_mda.csv").open() as stream:
        sidecar_reader = csv.DictReader(stream)
        assert sidecar_reader.fieldnames == ["x", "nmse_truth", "brier", "best_so_far_nmse_truth"]
        sidecar = list(sidecar_reader)
    assert [row["x"] for row in sidecar] == [row["x"] for row in points]
    np.testing.assert_allclose([float(row["nmse_truth"]) for row in sidecar],
                               [float(row["y_primary"]) for row in points])
    values = [float(row["nmse_truth"]) for row in sidecar]
    np.testing.assert_allclose([float(row["best_so_far_nmse_truth"]) for row in sidecar],
                               np.minimum.accumulate(values))
    assert len((out/"observations.jsonl").read_text().splitlines()) == 4


@pytest.mark.usefixtures("require_boxinggym_source")
def test_q_policy_removes_observed_points_from_native_and_sidecar_scores(tmp_path, monkeypatch):
    """The exclude_observed Q policy removes observed points from native and sidecar scores."""
    adapter = bg_stub_fits(monkeypatch)
    original_evaluation = adapter.TaskEnvironment.evaluation_points

    def evaluation(self, count, seed):
        """Evaluation points with one synthetic overlap with the observed design."""
        x, y = original_evaluation(self, count, seed)
        # A synthetic Q overlap; the real native evaluator sees this same point.
        x[0] = self.spec.d0()[0]
        point = self.goal.eval_points[0]
        self.goal.eval_points[0] = (float(x[0]), point[-1])
        return x, y

    monkeypatch.setattr(adapter.TaskEnvironment, "evaluation_points", evaluation)
    out = tmp_path/"exclude"
    adapter.run(bg_run_args(out, cell="000", q_policy="exclude_observed"))
    config = json.loads((out/"config.json").read_text())
    assert config["q_excluded_count"] >= 1
    with np.load(out/"evaluation_scoring_only.npz") as stored:
        x = stored["x"]
    observed = [row["x"] for row in map(json.loads, (out/"observations.jsonl").read_text().splitlines())]
    assert bg.unobserved_q_mask(x, observed).all()
    assert len(x) == config["q"]-config["q_excluded_count"]


BASELINE_HARNESS = r'''
import sys
from datetime import datetime
from pathlib import Path
import pytest
from bayes_model_discovery.experimental.test_named_inputs import bg_run_args, bg_stub_fits
class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2001, 1, 1, tzinfo=tz)
with pytest.MonkeyPatch.context() as monkeypatch:
    module = bg_stub_fits(monkeypatch)
    monkeypatch.setattr(module, 'datetime', FrozenDatetime)
    module.run(bg_run_args(Path(sys.argv[1]), cell='000', seed=73))
'''


class _PriorArray:
    def __init__(self, value):
        self.values = np.asarray(value)[None, None, ...]

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)

    def __getitem__(self, key):
        return self.values[key]


class _PriorTrace(dict):
    def __init__(self, values):
        self.prior = {name: _PriorArray(value) for name, value in values.items()}
        super().__init__(prior=self.prior)


class _NumpyPrior:
    """Sample distributions declared by the UNCHANGED native step methods.

    This replaces PyMC compilation, not the environment's formula, validation,
    outcome transform or run_experiment path. Fixed world parameters below
    make large repeated native calls inexpensive and reproducible.
    """
    def __init__(self, monkeypatch, seed=619):
        self.rng = np.random.default_rng(seed)
        self.active = None
        self.fixed = {}
        owner = self

        class Model:
            """Minimal stand-in for a PyMC model context that collects sampled values."""
            def __init__(self, *args, **kwargs):
                self.values = {}

            def __enter__(self):
                owner.active = self
                return self

            def __exit__(self, *args):
                return False

        def normal(name, mu=0., sigma=1., shape=None):
            """Stand-in for pymc.Normal: return a fixed or NumPy-sampled value."""
            value = self.fixed[name] if name in self.fixed else self.rng.normal(mu, sigma, shape)
            self.active.values[name] = value
            return value

        def bernoulli(name, p=None, logit_p=None, shape=None):
            """Stand-in for pymc.Bernoulli: sample with NumPy."""
            probability = expit(logit_p) if p is None else p
            value = self.rng.binomial(1, probability, size=shape)
            self.active.values[name] = value
            return value

        def poisson(name, mu):
            """Stand-in for pymc.Poisson: sample with NumPy."""
            value = self.rng.poisson(mu)
            self.active.values[name] = value
            return value

        def sample(*args, return_inferencedata=True, **kwargs):
            """Stand-in for pymc.sample_prior_predictive over the recorded values."""
            if return_inferencedata:
                return _PriorTrace(self.active.values)
            return {name: np.asarray([value]) for name, value in self.active.values.items()}

        monkeypatch.setattr(bg.pymc, "Model", Model)
        monkeypatch.setattr(bg.pymc, "Normal", normal)
        monkeypatch.setattr(bg.pymc, "Bernoulli", bernoulli)
        monkeypatch.setattr(bg.pymc, "Poisson", poisson)
        monkeypatch.setattr(bg.pymc, "sample_prior_predictive", sample)
        monkeypatch.setattr(bg.pymc.math, "invlogit", expit)


def _synthetic_native_environment(task):
    spec = bg.TASKS[task]
    cls = getattr(importlib.import_module(spec.module), spec.env_class)
    env = cls.__new__(cls)
    env.observed_data = []
    env.observation_data = []
    env.lower_bound, env.upper_bound = 0., 2.
    env.lower_limit, env.upper_limit = 0., 5.
    if task == "death":
        env.N, env.theta = 50, .6
        row, handle = [.8], ".8"
    elif task == "peregrines":
        env.alpha, env.beta1, env.beta2, env.beta3 = 1., .3, -.1, .01
        row, handle = [1.2], "1.2"
    elif task == "dugongs":
        env.a, env.b, env.l = .4, .6, .8
        row, handle = [1.2], "1.2"
    elif task == "hyperbolic":
        env.truth, env.epsilon = (.02, 7.), .07
        row, handle = [30., 100., 90.], "[30,100,90]"
    elif task == "location":
        env.b, env.m, env.alpha, env.sigma, env.dim = .1, .5, .8, .5, 2
        env.true_theta = np.array([[-.6, .3], [.7, -.2]])
        row, handle = [.2, .4], "[.2,.4]"
    elif task == "survival":
        env.beta, env.lambda0, env.num_patients = .7, .2, 1
        row, handle = [2., 1.], "0"
    elif task == "irt":
        env.num_students, env.num_questions, env.mode = 2, 2, "2pl"
        env.restricted_pair = []
        row, handle = [0., 1.], "[0,1]"
    elif task == "moral_machines":
        env.alpha, env.beta_group = .3, .2
        for index, name in enumerate(("human_count", "gender", "age", "social_status", "fitness", "species", "intervention")):
            setattr(env, "beta_"+name, .1*(index+1))
        env.characters = ["human_a", "human_b", "dog", "cat"]
        env._get_char_attr = lambda char: {
            "gender": -1 if char == "human_b" else 0,
            "age": 1 if char == "human_a" else 0,
            "social_status": 0, "fitness": 0, "species": 1 if char.startswith("human") else -1,
        }
        first, second, intervention = ["human_a"], ["human_b"], "swerve"
        row = bg.moral_contrast_row(env, first, second, intervention)
        handle = bg.moral_handle(first, second, intervention)
    elif task == "emotion":
        env.alpha = np.arange(8)*.3+2
        env.beta_win, env.beta_PE, env.beta_absPE = np.full(8, .02), np.full(8, .01), np.full(8, .03)
        env.sigma = np.linspace(.4, 1.5, 8)
        prizes, probabilities, index = [10., 20., 30.], [.2, .3, .5], 1
        row = bg.emotion_input_row(prizes, probabilities, prizes[index])
        handle = bg.emotion_handle(prizes, probabilities, index)
    if task in ("moral_machines", "emotion"):
        env.system = env.user_template = "synthetic numeric test narration"
        env.model_name = "gpt-4o"
        env.llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="synthetic"))])
        )))
    adapter = bg.TaskEnvironment.__new__(bg.TaskEnvironment)
    adapter.env, adapter.spec, adapter.observation_calls = env, spec, 0
    adapter.truth_parameters = {"alpha": np.array([-.4, .9]), "beta": np.array([.1, -.7]),
                                "gamma": np.array([.8, 1.3])}
    return adapter, np.asarray([row]) if len(row) > 1 else np.asarray(row), handle


@pytest.mark.parametrize("task", ["death", "peregrines", "dugongs", "survival", "hyperbolic", "location", "irt", "moral_machines", "emotion"])
@pytest.mark.usefixtures("require_boxinggym_source")
def test_truth_mean_matches_large_native_run_experiment_sample(task, monkeypatch, tmp_path):
    """The scoring-only truth mean matches a large sample from the native run_experiment path."""
    sampling = _NumpyPrior(monkeypatch)
    adapter, x, handle = _synthetic_native_environment(task)
    sampling.fixed = adapter.truth_parameters
    env = adapter.env
    expected = np.asarray(adapter.spec.truth_mean(adapter, x))[0]
    np.random.seed(903)
    count = 6000
    observations = []
    for _ in range(count):
        if task == "survival":
            # This native task stores one draw per patient. Regenerate the
            # synthetic patient from its actual generator before querying.
            env.outcomes = [[env._simulate_patient(1, 2.), 2., 1]]
        elif task == "irt":
            # Re-querying one stored cell is deterministic, not Monte Carlo.
            # Native reset redraws responses with the SAME fixed latent values.
            env.reset()
        observations.append(adapter.observe(handle))
    draws = np.asarray(observations)
    error = np.abs(draws.mean(axis=0)-expected)
    tolerance = 6*draws.std(axis=0, ddof=1)/np.sqrt(count)+.005
    assert np.all(error < tolerance), (task, error, tolerance)
    (tmp_path/f"mc_{task}.json").write_text(json.dumps({
        "task": task, "samples": count, "absolute_error": np.asarray(error).tolist(),
        "tolerance": np.asarray(tolerance).tolist(), "expected": np.asarray(expected).tolist(),
        "method": "native run_experiment with NumPy samplers for declared PyMC distributions; 6SE+.005"
    }, indent=2)+"\n")


@pytest.mark.usefixtures("require_boxinggym_source")
def test_irt_keeps_the_existing_latent_draw_without_a_second_call(monkeypatch):
    """IRT keeps its single latent draw and does not call the prior predictive a second time."""
    sampling = _NumpyPrior(monkeypatch)
    sampling.fixed = {"alpha": np.array([-.4, .9]), "beta": np.array([.1, -.7]),
                      "gamma": np.array([.8, 1.3])}
    from dataclasses import replace
    spec = replace(bg.TASKS["irt"], env_kwargs={"num_students": 2, "num_questions": 2})
    module = importlib.import_module(spec.module)
    # Avoid the native goal's ten random held-out selections; this test isolates
    # the constructor's single prior-predictive draw and hook restoration.
    monkeypatch.setattr(module, spec.goal_class, lambda env: SimpleNamespace(eval_points=[]))
    original = bg.pymc.sample_prior_predictive
    calls = []
    def counted(*args, **kwargs):
        """Count calls to the prior-predictive sampler and delegate to the original."""
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(bg.pymc, "sample_prior_predictive", counted)
    adapter = bg.TaskEnvironment(spec, 17)
    assert calls == [1]
    assert bg.pymc.sample_prior_predictive is counted
    np.testing.assert_array_equal(adapter.truth_parameters["gamma"], [.8, 1.3])
    np.testing.assert_allclose(spec.truth_mean(adapter, [[0, 1]]), expit(1.3*.3))
