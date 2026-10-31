"""Tests (added 2026-10-03) for FB1_EPIG_TARGETS = uniform (default) | score_q.  No paid call, no network, no key.

Run: cd tools/fb1_snapshot && env -u FORCE_COLOR JAX_PLATFORMS=cpu DP_SCIENCEAGENT_DIR=<ScienceAgent dir> \
      FB1_PYTHON=<jax venv python> FB1_OUT_DIR=/tmp/x/ \
      <jax venv python> -m pytest -q --confcutdir=. test_fb1_epig_targets.py
The runner reads FB1_* at import, so the option checks import it in a child interpreter (test_fb1_label_rule.native_import,
skipped without FB1_PYTHON). The record checks run the runner with FB1_T_EXTRA=0 (one fit round, no acquisition, about a
minute each); the acquisition checks stop the runner at its first
expected predictive information gain (EPIG) call.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SNAPSHOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SNAPSHOT))
os.environ.setdefault("FB1_OUT_DIR", "/tmp/fb1_epig_targets_test_out/")
os.environ.setdefault("FB1_WORLD", "gravity")
# DiscoverPhysics @ 33b7fa9: DP_SCIENCEAGENT_DIR, or a checkout beside the repository
os.environ.setdefault("DP_SCIENCEAGENT_DIR",
                      str(Path(__file__).resolve().parents[3] / "DiscoverPhysics" / "ScienceAgent"))

import run_gravity as rg  # noqa: E402
from test_fb1_label_rule import native_import  # noqa: E402

#: Q_HEAD then Q_SHIFT, written out (run_gravity.py: Q_HEAD and Q_SHIFT next to P1_RANGE).
SCORE_Q_POINTS = [[1.0, 1.0], [2.0, 1.0], [1.0, 2.0],
                  [3.0, 3.0], [3.0, 5.0], [4.0, 3.0], [4.0, 5.0], [5.0, 3.0], [5.0, 5.0]]


# ---------------------------------------------------------------- the target function
@pytest.mark.parametrize("seed", [11, 31, 37, 41])
def test_default_targets_are_the_uniform_stream_unchanged(seed):
    """Test: default targets are the uniform stream unchanged."""
    assert rg.EPIG_TARGETS == "uniform" and rg.EPIG_SCORE_Q is False   # this interpreter has no FB1_EPIG_TARGETS
    for got in (rg.epig_target_points(seed), rg.epig_target_points(seed, mode="uniform")):
        points, meta = got
        ref_points, ref_meta = rg.sample_epig_targets(seed)
        assert np.array_equal(points, ref_points) and meta == ref_meta
        assert points.shape == (32, 2) and meta["distribution"] == "uniform_rectangle"


def test_score_q_targets_are_q_head_then_q_shift_for_every_seed():
    """Test: score q targets are q head then q shift for every seed."""
    assert (rg.Q_HEAD + rg.Q_SHIFT) == [tuple(point) for point in SCORE_Q_POINTS]
    for seed in (11, 31, 37):
        points, meta = rg.epig_target_points(seed, mode="score_q")
        assert points.dtype == np.float64 and points.tolist() == SCORE_Q_POINTS
        assert meta == {"distribution": "score_q", "count": 9, "points": SCORE_Q_POINTS,
                        "source": "Q_HEAD + Q_SHIFT in that order (the scoring designs of score_Q)",
                        "fixed_across_rounds": True}
    # nothing is drawn: the global numpy stream is untouched
    state = np.random.get_state()[1].copy()
    rg.score_q_epig_targets()
    assert np.array_equal(np.random.get_state()[1], state)


def test_an_unknown_mode_is_refused_by_the_target_function():
    """Test: an unknown mode is refused by the target function."""
    with pytest.raises(ValueError, match="EPIG target mode must be uniform or score_q"):
        rg.epig_target_points(31, mode="random")


# ---------------------------------------------------------------- the option at import (child interpreter)
def receipt(proc):
    """Return the JSON receipt printed on the last stdout line of a successful child process."""
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("value", ["", "Score_Q", "score_q ", "random", "uniform_rectangle", "1"])
def test_an_unknown_value_is_refused_before_any_output(tmp_path, value):
    """Test: an unknown value is refused before any output."""
    proc = native_import(tmp_path, {"FB1_ARM": "112", "FB1_EPIG_TARGETS": value})
    assert proc.returncode != 0 and "FB1_EPIG_TARGETS must be uniform or score_q" in proc.stderr


@pytest.mark.parametrize("arm", ["000", "001", "100", "110", "111"])
def test_score_q_on_an_arm_without_epig_is_refused_before_any_output(tmp_path, arm):
    """Test: score q on an arm without epig is refused before any output."""
    proc = native_import(tmp_path, {"FB1_ARM": arm, "FB1_EPIG_TARGETS": "score_q"})
    assert proc.returncode != 0 and "FB1_EPIG_TARGETS=score_q requires the EPIG arm" in proc.stderr


def test_score_q_is_declared_to_the_label_rule_and_uniform_declares_nothing(tmp_path):
    """Test: score q is declared to the label rule and uniform declares nothing."""
    unset = receipt(native_import(tmp_path, {"FB1_ARM": "112"}))["binding"]["settings"]
    explicit = receipt(native_import(tmp_path, {"FB1_ARM": "112", "FB1_EPIG_TARGETS": "uniform"}))["binding"]["settings"]
    score_q = receipt(native_import(tmp_path, {"FB1_ARM": "112", "FB1_EPIG_TARGETS": "score_q"}))["binding"]["settings"]
    assert "epig_targets" not in unset and explicit == unset
    assert score_q == {**unset, "epig_targets": "score_q"}
    # an explicit default on an arm without EPIG is the default run: accepted, nothing declared
    plain = receipt(native_import(tmp_path, {"FB1_ARM": "111"}))
    uniform = receipt(native_import(tmp_path, {"FB1_ARM": "111", "FB1_EPIG_TARGETS": "uniform"}))
    assert uniform == plain and "epig_targets" not in plain["binding"]["settings"]


# ---------------------------------------------------------------- the runner in a child process


def child_env(out_dir, extra):
    """Build the child-process environment: only the given FB1_* options, no API keys, CPU only."""
    dp = os.environ.get("DP_SCIENCEAGENT_DIR", "")
    if not (Path(dp) / "scienceagent").is_dir():
        pytest.skip("set DP_SCIENCEAGENT_DIR for runner checks")
    env = {k: v for k, v in os.environ.items() if not k.startswith("FB1_")}
    env.update(JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENAI_API_KEY="", ANTHROPIC_API_KEY="", OPENAI_COMPAT_API_KEY="",
               DP_SCIENCEAGENT_DIR=dp, FB1_OUT_DIR=str(out_dir) + os.sep, **extra)
    env.pop("FORCE_COLOR", None)
    return env


def run_child(snapshot, out_dir, extra, code=None, timeout=1500):
    """`python -B run_gravity.py` (or `-c code`) in `snapshot` with only the given FB1_* options."""
    args = [sys.executable, "-B"] + (["-c", code] if code is not None else ["run_gravity.py"])
    return subprocess.run(args, cwd=snapshot, env=child_env(out_dir, extra), capture_output=True, text=True,
                          timeout=timeout, check=False)


# ---------------------------------------------------------------- whole run records (FB1_T_EXTRA=0)
ZERO_ROUND = {"FB1_WORLD": "oscillator", "FB1_ARM": "112", "FB1_SEED": "31", "FB1_T_EXTRA": "0"}
RECORD = "FB1_RUN_oscillator_112_seed31_v3.json"


def strip_volatile(record):
    """Remove wall-clock timings and the runner's own sha256 (the only code-identity field this change moves)."""
    record = json.loads(json.dumps(record))
    record.pop("timing_s")
    record["code_sha256"].pop("run_gravity.py")
    for checkpoint in record["checkpoints"].values():
        checkpoint.pop("elapsed_s")
    return record


@pytest.fixture(scope="module")
def zero_round_runs(tmp_path_factory):
    """Run the zero-round experiment for the default and score_q; return the output folders."""
    folders = {name: tmp_path_factory.mktemp(f"zero_round_{name}") for name in ("default", "score_q")}
    for name, snapshot, extra in (("default", SNAPSHOT, {}),
                                  ("score_q", SNAPSHOT, {"FB1_EPIG_TARGETS": "score_q"})):
        done = run_child(snapshot, folders[name], {**ZERO_ROUND, **extra})
        assert done.returncode == 0, (name, done.stderr[-3000:])
    return folders


def without_target_fields(record):
    """Strip volatile fields and the EPIG-target fields from a record; return (record, removed fields)."""
    record = strip_volatile(record)
    removed = {"epig_target_points": record.pop("epig_target_points"),
               "seeds.epig_targets": record["seeds"].pop("epig_targets"),
               "epig.target_count": record["epig"].pop("target_count"),
               "epig.target_measure": record["epig"].pop("target_measure"),
               "arm_config.epig_targets": record["arm_config"].pop("epig_targets", None),
               "score_Q.note": record["score_Q"].pop("note")}
    return record, removed


def test_score_q_zero_round_record_changes_exactly_the_target_fields(zero_round_runs):
    """Test: score q zero round record changes exactly the target fields."""
    default, plain = without_target_fields(json.loads((zero_round_runs["default"] / RECORD).read_text()))
    score_q, targeted = without_target_fields(json.loads((zero_round_runs["score_q"] / RECORD).read_text()))
    # everything else - fits, weights, scores, observations, numerics, label - is the same run
    assert score_q == default
    assert all(targeted[key] != plain[key] for key in plain)
    assert plain["arm_config.epig_targets"] is None and plain["epig.target_measure"] == "uniform_rectangle"
    assert targeted["epig_target_points"] == SCORE_Q_POINTS
    assert targeted["seeds.epig_targets"] == rg.epig_target_points(31, mode="score_q")[1]
    assert targeted["epig.target_count"] == 9 and targeted["epig.target_measure"] == "score_q"
    block = targeted["arm_config.epig_targets"]
    assert block["mode"] == "score_q" and block["points"] == SCORE_Q_POINTS and block["count"] == 9
    assert block["held_out_outcomes_used"] is False and block["option"] == "FB1_EPIG_TARGETS"
    assert "EPIG targets" in targeted["score_Q.note"]


# ---------------------------------------------------------------- what the acquisition receives
SPY = """
import json, numpy as np, run_gravity as r
class Stop(Exception):
    pass
seen = {}
def joint_epig(key, fits, weights, designs, target_points, noise_delta=None):
    seen["joint_epig"] = np.asarray(target_points).tolist(); raise Stop()
def run_design_search(*args, **kwargs):
    seen["run_design_search"] = np.asarray(kwargs["target_points"]).tolist(); raise Stop()
r.joint_epig = joint_epig
if r.ASCENT:
    r.fds.run_design_search = run_design_search
try:
    r.main()
except Stop:
    pass
print(json.dumps(seen))
"""
#: One fit round at n = 2 and no checkpoint there, so the run reaches its first acquisition quickly.
FIRST_ACQUISITION = {"FB1_WORLD": "oscillator", "FB1_ARM": "112", "FB1_SEED": "31", "FB1_T_EXTRA": "1",
                     "FB1_CHECKPOINTS": "3"}


@pytest.mark.parametrize(("extra", "seen_by", "expected"), [
    ({"FB1_EPIG_TARGETS": "score_q"}, "joint_epig", SCORE_Q_POINTS),
    ({"FB1_EPIG_TARGETS": "score_q", "FB1_DESIGN_SEARCH": "ascent"}, "run_design_search", SCORE_Q_POINTS),
    ({}, "joint_epig", None),
], ids=["score_q_random64", "score_q_ascent", "default_random64"])
def test_the_acquisition_receives_the_selected_targets(tmp_path, extra, seen_by, expected):
    """Test: the acquisition receives the selected targets."""
    done = run_child(SNAPSHOT, tmp_path / "out", {**FIRST_ACQUISITION, **extra}, SPY)
    assert done.returncode == 0, done.stderr[-3000:]
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    expected = rg.sample_epig_targets(31)[0].tolist() if expected is None else expected
    assert seen == {seen_by: expected}
