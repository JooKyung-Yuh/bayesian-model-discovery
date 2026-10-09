"""Paper-profile pre-seal checks and small fixes.

Six things are pinned here:

1. `--profile` defaults to "dev"; `--profile paper` is still
   recognized when given explicitly. Every existing dev-anchor invocation
   already passes --profile dev, so this flip changes no anchor.
2. Evaluation-point generation time is now inside the "setup_s"
   timing bucket; every other bucket is still accumulated exactly where
   earlier versions left it, and every bucket (including the new one) is a real,
   measured sub-interval of the wall time - they can never sum to more than
   it.
3. A design pool that runs out of fresh designs before the
   requested round count ends the run NORMALLY (status
   "design_pool_exhausted", with a reason recorded), rather than crashing -
   and the irt pool itself is still 26 rows, unaffected by this change.
4. The Brier score is written to a binary-outcome task's row
   (survival/hyperbolic/irt/moral_machines), left ABSENT - never 0 - for
   every other outcome type.
5. points.csv carries the contract's task-key rename
   (death->death_process, location->location_finding,
   moral_machines->morals; every other task's key is unchanged), the
   y_name/y_label rule (evaluator_env plus a human y_label, retiring
   "env_<metric>"), and the code_sha/brier/nmsle/rmsle columns - see
   `BOXINGGYM_POINTS_COLUMNS`'s own docstring for the exact 25-column order.
6. death's `y_primary` is `nmse_mda` - the Bayesian model averaging (BMA)
   predictive mean scored against the noise-free truth on the fixed
   evaluation set Q, normalized by that truth's own population variance
   (ddof=0) - not a relabeled copy of the environment's own (noisy-heldout)
   evaluator metric. `mse` in `scores.jsonl`/`summary.json` is unchanged (a
   different, still-computed number, scored against the noisy heldout).

Tests 1 and 5's task-key/column checks call `write_points` directly on a
hand-built row, with no sequential Monte Carlo (SMC) and no environment - it is a pure string/dict emitter.
Tests 2-4 (and 5's brier value) need a real `run()`: dev-profile, 1-5 rounds,
"death" (bounded-count, not binary) and "irt" (binary, and the design-
exclusion task that item 3 is about) are cheap enough for a real fit to be simpler and
more trustworthy here than a fake stand-in for `count_fit` - irt's per-level
("indexed") coefficients need per-level-shaped particle arrays a generic fit
stub does not know how to fake (count_models.particle_logpmf raises on a flat
one), and death's dev-profile SMC is already a handful of small fits (a few
seconds each - the same cost every other end-to-end boxinggym_core test in
this tree already pays).
"""

import csv
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def adapter():
    """Import the BoxingGym adapter module, skipping the test when it is unavailable."""
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
            # The runner's three death-regime switches plus the development-only
            # prompt variant.
            "initial_family": None,
            "truth_in_pool": False,
            "inject_oracle_at": None,
            "prompt_variant": None,
            **overrides,
        }
    )


# --------------------------------------------------------------------------
# --profile default
# --------------------------------------------------------------------------


def test_profile_default_is_dev(monkeypatch):
    """Check that profile default is dev."""
    bg = adapter()
    captured = []
    monkeypatch.setattr(bg, "run", captured.append)
    monkeypatch.setattr(
        sys, "argv", ["boxinggym_core", "--output", "/dev/null/never", "--task", "death"]
    )
    bg.main()
    (args,) = captured
    assert args.profile == "dev"


def test_profile_explicit_paper_is_still_recognized(monkeypatch):
    """Check that profile explicit paper is still recognized."""
    bg = adapter()
    captured = []
    monkeypatch.setattr(bg, "run", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "boxinggym_core", "--output", "/dev/null/never", "--task", "death",
            "--profile", "paper",
        ],
    )
    bg.main()
    (args,) = captured
    assert args.profile == "paper"


def test_profile_rejects_anything_else(monkeypatch):
    """Check that profile rejects anything else."""
    bg = adapter()
    monkeypatch.setattr(
        sys,
        "argv",
        ["boxinggym_core", "--output", "/x", "--task", "death", "--profile", "production"],
    )
    with pytest.raises(SystemExit):
        bg.main()


# --------------------------------------------------------------------------
# The "setup_s" timing bucket
# --------------------------------------------------------------------------


def test_setup_s_bucket_exists_and_every_bucket_sums_within_wall_time(tmp_path):
    """Check that setup s bucket exists and every bucket sums within wall time."""
    bg = adapter()
    pytest.importorskip("pymc")
    pytest.importorskip(bg.TASKS["death"].module)
    out = tmp_path / "run"
    bg.run(bg_run_args(out, task="death", rounds=1))
    summary = json.loads((out / "summary.json").read_text())
    timings = summary["timing_seconds"]
    # exactly the five earlier buckets plus the one new one: nothing lost,
    # nothing extra.
    assert set(timings) == {"setup_s", "fit", "proposal", "eig", "forward", "scoring"}
    # cell "111" with rounds=1 exercises every bucket at least once (D0's four
    # initial fits, one EIG-chosen round, one mock proposal, two checkpoints).
    for key, value in timings.items():
        assert value > 0.0, (key, value)
    # sum identity: every bucket is a real, measured sub-interval of the total
    # wall time - they can never add up to more than it, with or without the
    # new "setup_s" bucket.
    assert sum(timings.values()) <= summary["wall_seconds"]
    assert summary["wall_seconds"] > 0.0


# --------------------------------------------------------------------------
# Design-pool exhaustion terminates normally
# --------------------------------------------------------------------------


def test_design_pool_exhaustion_terminates_normally_instead_of_crashing(tmp_path, monkeypatch):
    """Check that design pool exhaustion terminates normally instead of crashing."""
    bg = adapter()
    pytest.importorskip("pymc")
    spec = bg.TASKS["irt"]
    pytest.importorskip(spec.module)
    real_build_categorical_pool = bg.build_categorical_pool

    def tiny_pool(task_spec, environment):
        """The real irt pool (26 rows), cut down to 5 - enough for D0's four
        distinct fractions and exactly one round, so round 2 exhausts it."""
        pool = real_build_categorical_pool(task_spec, environment)
        return bg.DesignPool(pool.rows[:5], pool.handles[:5], pool.single_input)

    monkeypatch.setattr(bg, "build_categorical_pool", tiny_pool)
    out = tmp_path / "run"
    # must NOT raise - that is the point of this test.
    bg.run(bg_run_args(out, task="irt", rounds=5))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "design_pool_exhausted"
    assert "round 2" in summary["design_pool_exhausted_reason"]
    assert summary["design_pool_size"] == 5
    assert summary["distinct_designs_observed"] == 5
    # the irt pool size ITSELF is unchanged by this change (the decision "keep the irt
    # pool at 26") - only this test's own monkeypatch shrinks it. Checked
    # against the REAL builder (captured above, before the monkeypatch) since
    # `bg.build_design_pool` would still route through the patched stand-in
    # for the rest of this test's duration. Same world-seed derivation AND
    # env_seed_injection wrapping `run()` itself uses for --seed 11 (this
    # task's canonical anchor seed) - irt.py's `reset` fixes its response
    # matrix with a pymc `sample_prior_predictive` draw, which is exactly the
    # OS-entropy-reseeding hazard `env_seed_injection` exists for (module
    # docstring), and `restricted_pair` (hence k, hence 36-k) turns out to
    # depend on that draw: unwrapped, this same construction is NOT
    # reproducible (26/27/28 rows across repeated calls, verified by hand);
    # wrapped, it is 26 every time, matching every --seed 11 anchor/smoke run
    # of this change.
    world_seed = bg.seed_for("boxinggym_world", spec.name, 11)
    with bg.env_seed_injection(world_seed):
        environment = bg.TaskEnvironment(spec, world_seed)
        assert len(real_build_categorical_pool(spec, environment).rows) == 26


def test_design_pool_exhaustion_policy_is_declared_in_config_regardless_of_outcome(tmp_path):
    """The policy string is unconditional (like the other *_RULE fields), so a
    run that never exhausts its pool still documents the behaviour, and its
    own status/summary stay exactly "completed" - no false positive."""
    bg = adapter()
    pytest.importorskip("pymc")
    pytest.importorskip(bg.TASKS["death"].module)
    out = tmp_path / "run"
    bg.run(bg_run_args(out, task="death", rounds=1))
    config = json.loads((out / "config.json").read_text())
    assert config["task_spec"]["design_pool_exhaustion_policy"] == bg.DESIGN_POOL_EXHAUSTION_RULE
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert "design_pool_exhausted_reason" not in summary


# --------------------------------------------------------------------------
# Brier on binary-outcome tasks only, never 0 where undefined
# --------------------------------------------------------------------------


def test_brier_is_recorded_for_a_binary_outcome_task_and_matches_its_definition(tmp_path):
    """Check that brier is recorded for a binary outcome task and matches its definition."""
    bg = adapter()
    pytest.importorskip("pymc")
    spec = bg.TASKS["irt"]
    pytest.importorskip(spec.module)
    out = tmp_path / "run"
    bg.run(bg_run_args(out, task="irt", rounds=1))
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    assert rows, "irt should score at least the round-0 checkpoint"
    for row in rows:
        brier = row["bma"]["brier"]
        assert 0.0 <= brier <= 1.0
        # the Brier score is the mean squared error between the predicted
        # P(Y=1) and the realised 0/1 outcome - for this two-class exhaustive
        # path that IS numerically what "mse" already computes (our_scores's
        # own comment explains why), so this pins the two columns to agree
        # rather than merely asserting brier is present.
        assert brier == pytest.approx(row["bma"]["mse"])
    with (out / "points.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            assert row["brier"] != ""


def test_brier_is_absent_never_zero_for_a_non_binary_outcome_task(tmp_path):
    """Check that brier is absent never zero for a non binary outcome task."""
    bg = adapter()
    pytest.importorskip("pymc")
    pytest.importorskip(bg.TASKS["death"].module)
    out = tmp_path / "run"
    bg.run(bg_run_args(out, task="death", rounds=1))
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    assert rows
    for row in rows:
        assert "brier" not in row["bma"]
    with (out / "points.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            # a blank field, never a literal "0"/"0.0" - "leave blank (never
            # 0) where undefined".
            assert row["brier"] == ""


# --------------------------------------------------------------------------
# points.csv shape, task-key rename, y_name/y_label, code_sha
# --------------------------------------------------------------------------


def _fake_row(**overrides):
    return {
        "observation_count": 4,
        "round": 0,
        "phase": "initial",
        "env_evaluator": {"metric": 0.5, "spread": 0.0, "metric_name": "mse"},
        # "nmse_mda" is read unconditionally for spec.name == "death" -
        # harmless filler for every other task, which never reads it.
        "bma": {"nll": 1.0, "coverage90": 0.9, "width90": 2.0, "nmse_mda": 0.25},
        **overrides,
    }


def _fake_config(**overrides):
    return {
        "output": "/tmp/does/not/need/to/exist",
        "cell": "111",
        "proposer": "mock_fixed_template_responses_NOT_LLM",
        "llm_model": "claude-opus-4-7",
        "seed": 11,
        "profile": "dev",
        "prior_template": "default",
        "priors": "off",
        "source_files_sha256": {"boxinggym_core.py": "a" * 64},
        **overrides,
    }


def test_points_columns_are_the_25_column_contract():
    """code_sha (22), brier (23), nmsle (24), rmsle (25)."""
    bg = adapter()
    assert bg.BOXINGGYM_POINTS_COLUMNS == (
        "suite", "task", "method", "proposer", "seed", "x", "x_unit",
        "y_primary", "y_name", "y_label", "nll", "coverage90", "width90",
        "crps", "is_0.1", "kl", "status", "label", "run_id", "source_path",
        "variant", "code_sha", "brier", "nmsle", "rmsle",
    )
    assert len(bg.BOXINGGYM_POINTS_COLUMNS) == 25


def test_points_nmsle_and_rmsle_are_always_blank_never_zero(tmp_path):
    """ChemBench's own auxiliary columns - BoxingGym rows
    never compute them."""
    bg = adapter()
    spec = bg.TASKS["death"]
    path = tmp_path / "points.csv"
    bg.write_points(path, _fake_config(), spec, [_fake_row()])
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["nmsle"] == ""
    assert rows[0]["rmsle"] == ""


@pytest.mark.parametrize(
    "task,expected_key",
    [
        ("death", "death_process"),
        ("location", "location_finding"),
        ("moral_machines", "morals"),
        ("peregrines", "peregrines"),
        ("dugongs", "dugongs"),
        ("survival", "survival"),
        ("hyperbolic", "hyperbolic"),
        ("irt", "irt"),
        ("emotion", "emotion"),
    ],
)
def test_points_task_key_mapping_table(tmp_path, task, expected_key):
    """The fixed task-key table, applied ONLY at emission - `TASKS`'s own
    key (the internal id) is untouched either way."""
    bg = adapter()
    spec = bg.TASKS[task]
    assert spec.name == task  # the internal id never changes
    path = tmp_path / f"{task}_points.csv"
    bg.write_points(path, _fake_config(), spec, [_fake_row()])
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["task"] == expected_key


@pytest.mark.parametrize(
    "task,expected_y_name",
    [("death", "nmse_mda")]
    + [
        (task, "evaluator_env")
        for task in (
            "peregrines", "dugongs", "location", "survival", "hyperbolic", "irt",
            "moral_machines", "emotion",
        )
    ],
)
def test_points_y_name_retires_env_metric_naming(tmp_path, task, expected_y_name):
    """Check that points y name retires env metric naming."""
    bg = adapter()
    spec = bg.TASKS[task]
    path = tmp_path / f"{task}_points.csv"
    bg.write_points(path, _fake_config(), spec, [_fake_row()])
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["y_name"] == expected_y_name
    assert not rows[0]["y_name"].startswith("env_")
    assert rows[0]["y_label"]  # every runnable task has a non-empty human label


def test_points_code_sha_is_the_runs_own_engine_hash(tmp_path):
    """Check that points code sha is the runs own engine hash."""
    bg = adapter()
    spec = bg.TASKS["death"]
    config = _fake_config()
    path = tmp_path / "points.csv"
    bg.write_points(path, config, spec, [_fake_row()])
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["code_sha"] == config["source_files_sha256"]["boxinggym_core.py"]


# --------------------------------------------------------------------------
# death's y_primary is nmse_mda, not a relabeled copy of the environment's
# own evaluator metric.
# --------------------------------------------------------------------------


def test_death_y_primary_is_nmse_mda_against_independently_recomputed_truth(tmp_path):
    """Recomputes nmse_mda from scratch - theta read from config.json (never
    from `death_noise_free_truth`), the hazard formula re-typed by hand from
    death_process.py:322 (`eta = 1 - exp(-theta * xi)`) with N=50
    (death_process.py:274's default, this task's TaskSpec.trials), and the
    BMA predictive mean read from the run's own saved `predictions/*.npz` -
    then checks it against points.csv's own y_primary and scores.jsonl's own
    `bma.nmse_mda`, independently of any boxinggym_core internals."""
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
    true_mean = 50.0 * (1.0 - np.exp(-theta * evaluation["x"]))
    var_q = np.var(true_mean, ddof=0)  # population variance, ddof=0
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    with (out / "points.csv").open(newline="") as stream:
        points = list(csv.DictReader(stream))
    assert len(rows) == len(points) > 0
    for row, point in zip(rows, points, strict=True):
        prediction = np.load(out / "predictions" / f"t{row['round']:02d}_{row['phase']}.npz")
        independent = float(np.mean((prediction["mean"] - true_mean) ** 2) / var_q)
        assert row["bma"]["nmse_mda"] == pytest.approx(independent)
        assert float(point["y_primary"]) == pytest.approx(independent)
        # and NOT the environment's own (noisy-heldout) evaluator metric -
        # the two are different quantities on this task.
        assert float(point["y_primary"]) != pytest.approx(row["env_evaluator"]["metric"])


def test_death_mse_in_summary_is_unchanged_scored_against_noisy_heldout(tmp_path):
    """`mse` is a DIFFERENT, still-computed number (vs the noisy heldout `y`,
    not the noise-free truth) - `nmse_mda` must not repurpose it."""
    bg = adapter()
    pytest.importorskip("pymc")
    pytest.importorskip(bg.TASKS["death"].module)
    out = tmp_path / "run"
    bg.run(bg_run_args(out, task="death", rounds=1))
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    for row in rows:
        assert "mse" in row["bma"]
        assert "nmse_mda" in row["bma"]
        assert row["bma"]["mse"] != pytest.approx(row["bma"]["nmse_mda"])


# ---------------------------------------------------------------------------
# The shared `core_count.points_label`
# helper - a finding from an earlier smoke run. `write_points`'s own `label`
# column is not a second, independently maintained copy of the rule - it must
# agree with calling `points_label` directly, for every cell of the truth table, including the one an earlier
# ChemBench smoke run hit (a paper-profile MOCK run, previously mislabelled
# "production" here too - `boxinggym_core.py`'s pre-patch `write_points` had
# the identical inline idiom as `core_count.py` and `chembench_core.py`).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile,proposer,expected",
    [
        ("dev", "mock_fixed_template_responses_NOT_LLM", "development"),
        ("dev", "anthropic", "development"),
        ("dev", "openai", "development"),
        # the real `config["proposer"]` for a mock run (`boxinggym_core.run`'s
        # own assignment) - never the bare CLI literal "mock".
        ("paper", "mock_fixed_template_responses_NOT_LLM", "development"),
        ("paper", "anthropic", "development"),
        ("paper", "openai", "development"),
    ],
)
def test_write_points_label_matches_points_label_for_fake_config(
    tmp_path, profile, proposer, expected
):
    """Check that write points label matches points label for fake config."""
    bg = adapter()
    spec = bg.TASKS["death"]
    config = _fake_config(profile=profile, proposer=proposer)
    path = tmp_path / "points.csv"
    bg.write_points(path, config, spec, [_fake_row()])
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    normalized = "mock" if "mock" in proposer else proposer
    assert rows[0]["label"] == bg.points_label(profile, normalized) == expected


# ---------------------------------------------------------------------------
# `--run-label` on boxinggym_core (all-benchmark 2 h smoke).
# ---------------------------------------------------------------------------


def test_18_bg_refuses_run_label_production_before_any_output(tmp_path):
    """REFUSE: production is accepted only for --profile paper with --proposer
    anthropic. A paper+mock run asking for it is refused at run() entry, with
    no output directory left behind."""
    bg = adapter()
    out = tmp_path / "should_not_exist"
    args = bg_run_args(out, task="death", rounds=1, profile="paper", run_label="production")
    with pytest.raises(ValueError, match="--run-label production"):
        bg.run(args)
    assert not out.exists()


@pytest.mark.usefixtures("require_boxinggym_source")
def test_18_bg_accepts_run_label_development_and_records_it(tmp_path):
    """ACCEPT: development always runs, and the chosen label reaches
    config.json (`run_label` and `label`) and every points.csv row."""
    bg = adapter()
    out = tmp_path / "dev_label"
    bg.run(bg_run_args(out, task="death", rounds=1, run_label="development"))
    config = json.loads((out / "config.json").read_text())
    assert config["run_label"] == "development"
    assert config["label"] == "development"
    with (out / "points.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and {row["label"] for row in rows} == {"development"}
