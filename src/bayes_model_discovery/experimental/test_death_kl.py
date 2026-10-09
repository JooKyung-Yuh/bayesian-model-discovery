"""death's KL score: `our_scores` records `kl` (the Kullback-Leibler divergence from the truth to the Bayesian
model averaging (BMA) predictive, `KL(truth || BMA)`) and `expected_log_score` for the death task only, both computed by
`core_count.score_prediction` (imported unchanged - see its own docstring at
core_count.py:575-589) against death's own true predictive PMF
(`death_truth_pmf`, `Binomial(N, 1 - exp(-theta*xi))` with the run's one true
`theta` - death_process.py:280 `self.N = N`, :286 the once-only `theta`
draw, :319-:323 `step`) on the SAME fixed evaluation set Q every other death
score already uses (`evaluation_x`). Every other task's points.csv `kl`
column stays blank, never 0 - unaffected by this change (a requirement of it).

Four things are pinned here:
1. Synthetic check: when the BMA log-PMF equals the true PMF exactly, KL is
   <= 1e-12 and expected_log_score equals the truth's own negative entropy -
   independently recomputed with scipy.special.xlogy, not by re-calling
   anything boxinggym_core defines.
2. death_truth_pmf sums to 1 over y=0..50 for every query point, and a real
   run's own true PMF (independently recomputed from config.json's
   world_parameters.theta and TASKS["death"].trials==50) agrees with it.
3. The all-tasks KL path fills the other eight tasks' `kl` through a SEPARATE
   path, so a non-death task (irt - the same dev fixture
   test_paper_profile_seal.py's brier tests already use for the analogous
   brier-absence check) now records its own law's `kl` and carries the all-tasks
   `kl_method` label death's branch never writes, which is what keeps the two
   paths distinguishable in the artifact. The blank-never-0 rule itself is
   pinned by test_boxinggym_kl_all_tasks.py, on a task with no law at all.
4. A real death run records `kl`/`expected_log_score` in every scores.jsonl
   row, in summary.json's "our_scores" (the last round's, same rule every
   other bma key already follows), and in points.csv's `kl` column - and the
   points.csv value matches scores.jsonl's `bma.kl` exactly.
"""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import xlogy
from scipy.stats import binom


def adapter():
    """Import the BoxingGym core adapter, skipping the test when it is unavailable."""
    return pytest.importorskip(
        "bayes_model_discovery.experimental.boxinggym_core",
        reason="adapter imports the read-only boxing_gym package lazily",
    )


def bg_run_args(output, **overrides):
    """The full argparse namespace boxinggym_core.run expects, with default option values."""
    bg = adapter()
    return SimpleNamespace(
        **{
            "output": str(output),
            "task": "death",
            "profile": "dev",
            "proposer": "mock",
            "llm_model": None,
            "allow_paid_api": False,
            "rounds": 1,
            "seed": 11,
            "cell": "111",
            "prior_template": bg.DEFAULT_PRIOR_TEMPLATE,
            "priors": "off",
            # The runner's three death-regime switches plus the
            # development-only prompt variant.
            "initial_family": None,
            "truth_in_pool": False,
            "inject_oracle_at": None,
            "prompt_variant": None,
            **overrides,
        }
    )


# --------------------------------------------------------------------------
# 1. synthetic: BMA == truth -> KL ~ 0, expected_log_score == -entropy(truth)
# --------------------------------------------------------------------------


def test_kl_near_zero_and_expected_log_score_matches_negative_entropy_when_bma_equals_truth():
    """Check that kl near zero and expected log score matches negative entropy when bma equals truth."""
    bg = adapter()
    environment = SimpleNamespace(env=SimpleNamespace(N=50, theta=0.9))
    times = np.array([0.2, 0.7, 1.3, 1.8])
    truth = bg.death_truth_pmf(environment, times)
    assert truth.shape == (4, 51)
    logpmf = np.log(truth)  # the BMA predictive == the truth, exactly
    heldout = np.zeros(len(times), dtype=int)  # unused by kl/expected_log_score
    rng = np.random.default_rng(12345)
    scores, _ = bg.score_prediction(logpmf, heldout, truth, rng)
    assert abs(scores["kl"]) <= 1e-12
    # independently recomputed negative entropy (no reuse of score_prediction's
    # own arithmetic): mean_Q sum_y truth * log(truth), via xlogy so a genuine
    # zero probability contributes 0, not nan.
    negative_entropy = float(np.sum(xlogy(truth, truth), axis=-1).mean())
    assert scores["expected_log_score"] == pytest.approx(negative_entropy, abs=1e-12)


# --------------------------------------------------------------------------
# 2. death_truth_pmf: sums to 1, N=50, matches a real run's own config theta
# --------------------------------------------------------------------------


def test_death_truth_pmf_sums_to_one_and_uses_n50():
    """Check that death truth pmf sums to one and uses n50."""
    bg = adapter()
    environment = SimpleNamespace(env=SimpleNamespace(N=50, theta=0.55))
    times = np.array([0.001, 0.5, 1.0, 1.5, 1.999])
    truth = bg.death_truth_pmf(environment, times)
    assert truth.shape == (5, 51)
    np.testing.assert_allclose(truth.sum(axis=-1), 1.0, atol=1e-12)
    assert np.all(truth >= 0.0)
    assert np.all(np.isfinite(truth))


def test_death_truth_pmf_matches_config_theta_and_trials_on_a_real_run(tmp_path):
    """theta read from config.json (never re-derived from death_truth_pmf's own
    call inside the run), N from TASKS["death"].trials (this task's declared
    trial count, death_process.py:274's default) - both re-typed by hand into
    an independent binom.pmf call and checked against the run's own recorded
    evaluation set AND against death_truth_pmf called on a freshly
    reconstructed environment (same world_seed the run itself used)."""
    bg = adapter()
    pytest.importorskip("pymc")
    spec = bg.TASKS["death"]
    pytest.importorskip(spec.module)
    assert spec.trials == 50
    out = tmp_path / "run"
    bg.run(bg_run_args(out, task="death", rounds=1))
    config = json.loads((out / "config.json").read_text())
    theta = config["world_parameters"]["theta"]
    evaluation = np.load(out / "evaluation_scoring_only.npz")
    independent = binom.pmf(np.arange(51)[None, :], 50, (1.0 - np.exp(-theta * evaluation["x"]))[:, None])
    np.testing.assert_allclose(independent.sum(axis=-1), 1.0, atol=1e-12)
    world_seed = bg.seed_for("boxinggym_world", spec.name, 11)
    with bg.env_seed_injection(world_seed):
        environment = bg.TaskEnvironment(spec, world_seed)
    reconstructed = bg.death_truth_pmf(environment, evaluation["x"])
    np.testing.assert_allclose(reconstructed, independent, atol=1e-12)


# --------------------------------------------------------------------------
# 3. non-death task: the all-tasks path fills `kl`, and says so (irt dev
#    fixture, the same run_args/run pattern test_paper_profile_seal.py's
#    brier tests use for the analogous absence check)
# --------------------------------------------------------------------------


def test_non_death_task_records_kl_through_the_k1_path_not_deaths(tmp_path):
    """Check that non death task records kl through the k1 path not deaths."""
    bg = adapter()
    pytest.importorskip("pymc")
    spec = bg.TASKS["irt"]
    pytest.importorskip(spec.module)
    out = tmp_path / "run"
    bg.run(bg_run_args(out, task="irt", rounds=1))
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    assert rows
    for row in rows:
        # death's branch records `kl`/`expected_log_score` and nothing else;
        # only the all-tasks path labels the method it used.
        assert row["bma"]["kl_method"] == "discrete_exact"
        assert np.isfinite(row["bma"]["kl"]) and row["bma"]["kl"] >= -1e-9
    with (out / "points.csv").open(newline="") as stream:
        points = list(csv.DictReader(stream))
    assert len(points) == len(rows)
    for row, point in zip(rows, points, strict=True):
        assert float(point["kl"]) == pytest.approx(row["bma"]["kl"])
        assert "kl_method" not in point


# --------------------------------------------------------------------------
# 4. death: kl/expected_log_score in scores.jsonl, summary.json, points.csv
# --------------------------------------------------------------------------


def test_death_records_kl_and_expected_log_score_in_scores_summary_and_points(tmp_path):
    """Check that death records kl and expected log score in scores summary and points."""
    bg = adapter()
    pytest.importorskip("pymc")
    pytest.importorskip(bg.TASKS["death"].module)
    out = tmp_path / "run"
    bg.run(bg_run_args(out, task="death", rounds=2))
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    assert rows
    for row in rows:
        kl, expected_log_score = row["bma"]["kl"], row["bma"]["expected_log_score"]
        assert np.isfinite(kl) and np.isfinite(expected_log_score)
        assert kl >= -1e-9  # a KL divergence is never negative
    summary = json.loads((out / "summary.json").read_text())
    assert summary["our_scores"]["kl"] == pytest.approx(rows[-1]["bma"]["kl"])
    assert summary["our_scores"]["expected_log_score"] == pytest.approx(
        rows[-1]["bma"]["expected_log_score"]
    )
    with (out / "points.csv").open(newline="") as stream:
        points = list(csv.DictReader(stream))
    assert len(points) == len(rows)
    for row, point in zip(rows, points, strict=True):
        assert point["kl"] != ""
        assert float(point["kl"]) == pytest.approx(row["bma"]["kl"])
