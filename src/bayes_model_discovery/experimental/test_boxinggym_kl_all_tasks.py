"""Runner-level tests: every task records `kl`, and nothing else moves.

The three properties pinned for the eight non-death tasks:

1. points.csv keeps its 25-column contract - the header tuple is
   `BOXINGGYM_POINTS_COLUMNS` and the only column that changes value is `kl`,
   which the rest of the row is checked against, column by column, from the
   scores.jsonl row it was written from.
2. the diagnostics (`kl_method`, `kl_truncated_mass`, `kl_quadrature_error`,
   `expected_log_score`) are in scores.jsonl and summary.json's `our_scores`,
   and in NO points.csv column.
3. the recording is unconditional, with the single measured exception
   (allowed because of its measured cost; `KL_OPTIONAL_TASKS`, the task whose added time exceeded
   10% of a cell's wall clock, behind `--kl-all-tasks`); `config.json` records
   the flag, the list and whether this run's task had a law at all, and a task
   with no law keeps the column blank rather than 0.

death's own path is pinned by `test_death_kl.py` and is not re-tested here.
"""

import csv
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

DIAGNOSTIC_KEYS = ("kl_method", "kl_truncated_mass", "kl_quadrature_error")


def adapter():
    """Test helper: adapter."""
    return pytest.importorskip(
        "bayes_model_discovery.experimental.boxinggym_core",
        reason="adapter imports the read-only boxing_gym package lazily",
    )


def run_args(output, task, **overrides):
    """Test helper: run args."""
    bg = adapter()
    return SimpleNamespace(
        **{
            "output": str(output), "task": task, "profile": "dev", "proposer": "mock",
            "llm_model": None, "allow_paid_api": False, "rounds": 1, "seed": 11,
            "cell": "111", "prior_template": bg.DEFAULT_PRIOR_TEMPLATE, "priors": "off",
            "initial_family": None, "truth_in_pool": False, "inject_oracle_at": None,
            "prompt_variant": None, **overrides,
        }
    )


def execute(tmp_path, task, **overrides):
    """Test helper: execute."""
    bg = adapter()
    pytest.importorskip("pymc")
    pytest.importorskip(bg.TASKS[task].module)
    out = tmp_path / "run"
    bg.run(run_args(out, task, **overrides))
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    with (out / "points.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        points = list(reader)
        header = tuple(reader.fieldnames)
    return out, rows, points, header


# --------------------------------------------------------------------------
# 1 + 2: the column contract, and where the diagnostics may appear
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["survival", "dugongs"])
def test_points_keeps_25_columns_and_only_kl_is_newly_filled(tmp_path, task):
    """Test: points keeps 25 columns and only kl is newly filled."""
    bg = adapter()
    _, rows, points, header = execute(tmp_path, task)
    assert header[:len(bg.BOXINGGYM_POINTS_COLUMNS)] == bg.BOXINGGYM_POINTS_COLUMNS
    assert len(bg.BOXINGGYM_POINTS_COLUMNS) == 25
    assert points and len(points) == len(rows)
    for row, point in zip(rows, points, strict=True):
        # `kl` is the one column that used to be blank and is not any more.
        assert point["kl"] != ""
        assert float(point["kl"]) == pytest.approx(row["bma"]["kl"])
        # every other value-bearing column still comes from exactly where it
        # came from before (`write_points`' own sources), unchanged.
        assert float(point["nll"]) == pytest.approx(row["bma"]["nll"])
        assert float(point["coverage90"]) == pytest.approx(row["bma"]["coverage90"])
        assert float(point["width90"]) == pytest.approx(row["bma"]["width90"])
        assert float(point["y_primary"]) == pytest.approx(row["env_evaluator"]["metric"])
        assert point["crps"] == "" and point["is_0.1"] == ""
        assert point["nmsle"] == "" and point["rmsle"] == ""
        assert point["status"] == "completed"


@pytest.mark.parametrize("task", ["survival", "dugongs"])
def test_diagnostics_are_in_scores_and_summary_and_never_in_points(tmp_path, task):
    """Test: diagnostics are in scores and summary and never in points."""
    out, rows, points, header = execute(tmp_path, task)
    for key in DIAGNOSTIC_KEYS + ("expected_log_score",):
        assert key not in header
        assert all(key not in point for point in points)
    summary = json.loads((out / "summary.json").read_text())
    for row in rows:
        assert np.isfinite(row["bma"]["kl"]) and row["bma"]["kl"] >= -1e-9
        assert isinstance(row["bma"]["kl_method"], str) and row["bma"]["kl_method"]
    assert summary["our_scores"]["kl"] == pytest.approx(rows[-1]["bma"]["kl"])
    assert summary["our_scores"]["kl_method"] == rows[-1]["bma"]["kl_method"]


def test_binary_task_records_the_discrete_pair_and_no_quadrature_diagnostics(tmp_path):
    """Test: binary task records the discrete pair and no quadrature diagnostics."""
    _, rows, _, _ = execute(tmp_path, "survival")
    for row in rows:
        assert row["bma"]["kl_method"] == "discrete_exact"
        assert np.isfinite(row["bma"]["expected_log_score"])
        assert "kl_truncated_mass" not in row["bma"]
        assert "kl_quadrature_error" not in row["bma"]


def test_density_task_records_the_quadrature_diagnostics(tmp_path):
    """Test: density task records the quadrature diagnostics."""
    _, rows, _, _ = execute(tmp_path, "dugongs")
    for row in rows:
        assert row["bma"]["kl_method"].startswith("gauss_legendre_")
        assert 0.0 <= row["bma"]["kl_truncated_mass"] < 1e-13
        # The 32-vs-64-node difference against a real lognormal/gamma mixture
        # (which a first-round pool can leave sharply peaked near y = 0) is
        # measured in the 1e-5..1e-9 band, five orders below the value itself
        # - recorded, never assumed away.
        assert 0.0 <= row["bma"]["kl_quadrature_error"] < 1e-3
        assert "expected_log_score" not in row["bma"]


def test_count_task_records_the_enumeration_depth(tmp_path):
    """Truth-table case 3: the optional task WITH the flag, (law available, recorded) = (True, True)."""
    out, rows, points, _ = execute(tmp_path, "peregrines", kl_all_tasks=True)
    config = json.loads((out / "config.json").read_text())
    assert config["kl_all_tasks"] is True
    assert config["kl_truth_law_available"] is True
    assert config["kl_recorded"] is True
    assert all(point["kl"] != "" for point in points)
    for row in rows:
        assert row["bma"]["kl_method"].startswith("poisson_enumeration_tail_")
        assert row["bma"]["kl_truncated_mass"] < 1e-10
        assert np.isfinite(row["bma"]["expected_log_score"])


# --------------------------------------------------------------------------
# 3: unconditional recording, and the one case that stays blank
# --------------------------------------------------------------------------


def test_config_records_the_law_and_the_measured_cost_fallback(tmp_path):
    """Truth-table case 1: an unconditional task, (law available, recorded) = (True, True)."""
    bg = adapter()
    out, _, _, _ = execute(tmp_path, "dugongs")
    config = json.loads((out / "config.json").read_text())
    assert config["kl_truth_law_available"] is True
    assert config["kl_recorded"] is True
    # the fallback exists but was not used: dugongs records unconditionally.
    assert config["kl_all_tasks"] is False
    assert config["kl_optional_tasks"] == list(bg.KL_OPTIONAL_TASKS) == ["peregrines"]


def test_emotion_is_unconditional_and_peregrines_is_the_only_optional_task():
    """Flipped by a re-measurement on the reference cell.

    An earlier measurement put emotion's cell integration at 32.1 % of the cell's wall clock,
    so it sat behind the flag together with peregrines. The exact mixture-CDF cells
    ask the pool for eight values per query instead of about seven hundred:
    0.97 % on the worst of two repeats of the same reference cell, so emotion
    records `kl` like every other task that is not peregrines.
    """
    bg = adapter()
    assert bg.KL_OPTIONAL_TASKS == ("peregrines",)
    assert bg.TASKS["emotion"].truth_law is not None


def test_the_optional_task_records_nothing_until_the_flag_is_given(tmp_path):
    """Truth-table case 2: an optional task without the flag, (law available, recorded) = (True, False).

    Changed in a later revision: `kl_truth_law_available` used to be False here, which
    said "peregrines has no true observation law" when what was true is only
    that this run did not score it. The law is now built either way and the
    run's own answer moved to `kl_recorded`.
    """
    out, rows, points, _ = execute(tmp_path, "peregrines")
    config = json.loads((out / "config.json").read_text())
    assert config["kl_all_tasks"] is False
    assert config["kl_truth_law_available"] is True
    assert config["kl_recorded"] is False
    assert all("kl" not in row["bma"] for row in rows)
    assert points and all(point["kl"] == "" for point in points)


def test_a_task_without_a_law_records_the_flag_and_keeps_the_column_blank(tmp_path, monkeypatch):
    """Truth-table case 4: irt's latent draw was not captured, (False, False) - blank, never 0."""
    bg = adapter()
    monkeypatch.setitem(
        bg.TASKS, "irt", dataclasses.replace(bg.TASKS["irt"], truth_law=lambda *args: None)
    )
    out, rows, points, _ = execute(tmp_path, "irt")
    config = json.loads((out / "config.json").read_text())
    assert config["kl_truth_law_available"] is False
    assert config["kl_recorded"] is False
    for row in rows:
        assert "kl" not in row["bma"]
        for key in DIAGNOSTIC_KEYS + ("expected_log_score",):
            assert key not in row["bma"]
    assert points and all(point["kl"] == "" for point in points)


def test_composite_value_averages_numbers_and_carries_one_shared_label():
    """Test: composite value averages numbers and carries one shared label."""
    bg = adapter()
    assert bg.composite_value([1.0, 2.0, 3.0]) == 2.0
    assert bg.composite_value(["discrete_exact"] * 3) == "discrete_exact"
    with pytest.raises(ValueError):
        bg.composite_value(["discrete_exact", "gauss_legendre_32_64"])


def test_every_runnable_task_registers_a_truth_law():
    """Test: every runnable task registers a truth law."""
    bg = adapter()
    missing = [name for name, spec in bg.TASKS.items()
               if spec.runnable and spec.truth_law is None]
    assert missing == []
    assert len([name for name, spec in bg.TASKS.items() if spec.truth_law is not None]) == 9


def test_k1c_a_world_without_an_underlying_environment_records_no_law_and_completes(tmp_path, monkeypatch):
    """Hotfix regression: a synthetic world with no `env` has no true law and does not crash.

    The emotion runner on the same synthetic world the acquisition tests use (no underlying
    environment object) completes, records kl_truth_law_available False and kl_recorded
    False, and writes no kl key anywhere in its scores.
    """
    from bayes_model_discovery.experimental.test_epig_positivity_guard import (
        synthetic_emotion_record,
    )

    output = tmp_path / "run"
    synthetic_emotion_record(output, monkeypatch)
    config = json.loads((output / "config.json").read_text())
    assert config["kl_truth_law_available"] is False
    assert config["kl_recorded"] is False
    rows = [json.loads(line) for line in (output / "scores.jsonl").read_text().splitlines()]
    assert rows

    def kl_keys(value):
        """Test helper: kl keys."""
        if isinstance(value, dict):
            return [key for key in value if key == "kl" or key.startswith("kl_")] + [
                found for item in value.values() for found in kl_keys(item)]
        if isinstance(value, list):
            return [found for item in value for found in kl_keys(item)]
        return []

    assert all(kl_keys(row) == [] for row in rows)

