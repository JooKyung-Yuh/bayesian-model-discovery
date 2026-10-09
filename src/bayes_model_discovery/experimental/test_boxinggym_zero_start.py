"""The optional half-count response unit of the initial design set (D0) for a start whose observations are all zero.

Mock proposer, the task's own zero-cost environment, dev profile; no provider call. No recorded
run value is a fixture: the two zero starting observations of peregrines seed 37 are what the
environment itself returns for the registered random2 start.
"""

import argparse
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from . import boxinggym_core as bg
from . import count_proposer as cp
from . import run_label_policy as policy
from .test_named_inputs import bg_run_args


@pytest.mark.parametrize("option", ["refuse", "half_count"])
def test_a_positive_mean_is_the_unit_whatever_the_option(option):
    """A positive D0 mean is the response unit under either option."""
    unit, applied = bg.d0_response_unit([0.0, 3.0, 0.0], option)
    assert unit == 1.0 and applied is False
    unit, applied = bg.d0_response_unit([2.5, 7.5], option)
    assert unit == 5.0 and applied is False


def test_an_all_zero_start_is_refused_by_default_and_half_counted_on_request():
    """An all-zero start is refused by default and half-counted on request."""
    with pytest.raises(ValueError, match="D0 response unit is not positive"):
        bg.d0_response_unit([0.0, 0.0], "refuse")
    with pytest.raises(ValueError, match="D0 response unit is not positive"):
        bg.d0_response_unit([0.0, 0.0])
    assert bg.d0_response_unit([0.0, 0.0], "half_count") == (0.25, True)
    assert bg.d0_response_unit([0.0, 0.0, 0.0, 0.0], "half_count") == (0.125, True)
    assert bg.d0_response_unit(np.zeros(2), "half_count") == (0.25, True)


@pytest.mark.parametrize("values", [[-1.0, 1.0], [-2.0, -2.0], [0.0, float("nan")], [float("inf"), 0.0]])
@pytest.mark.parametrize("option", ["refuse", "half_count"])
def test_a_negative_non_finite_or_mixed_sign_zero_mean_is_refused_under_both_options(values, option):
    # [-1, 1] has a zero mean but is not an all-zero start: refused, not half-counted.
    """Negative, non-finite or mixed-sign zero-mean starts are refused under both options."""
    with pytest.raises(ValueError, match="D0 response unit is not positive"):
        bg.d0_response_unit(values, option)


def test_the_option_is_half_a_count_only_when_the_mean_is_exactly_zero():
    # A mean that merely rounds to zero is a positive mean.
    """The half-count unit applies only when the mean is exactly zero."""
    unit, applied = bg.d0_response_unit([0.0, 1e-300], "half_count")
    assert unit == 5e-301 and applied is False


def test_cli_leaves_the_option_absent_by_default_and_the_choices_are_closed(monkeypatch, tmp_path):
    """The CLI leaves the option absent by default and accepts only the closed choices."""
    seen = []
    monkeypatch.setattr(bg, "run", lambda args: seen.append(vars(args)))
    monkeypatch.setattr(sys, "argv", ["boxinggym_core", "--task", "peregrines", "--output", str(tmp_path / "a")])
    bg.main()
    assert "d0_zero_count_unit" not in seen[0]  # absent unless typed; run() reads it as refuse
    monkeypatch.setattr(sys, "argv", ["boxinggym_core", "--task", "peregrines", "--output", str(tmp_path / "b"),
                                     "--d0-zero-count-unit", "half_count"])
    bg.main()
    assert seen[1]["d0_zero_count_unit"] == "half_count"
    monkeypatch.setattr(sys, "argv", ["boxinggym_core", "--task", "peregrines", "--output", str(tmp_path / "c"),
                                     "--d0-zero-count-unit", "one"])
    with pytest.raises(SystemExit) as error:
        bg.main()
    assert error.value.code == 2 and len(seen) == 2


def test_an_unknown_option_value_is_refused_by_run_before_output(tmp_path):
    """run() refuses an unknown option value before writing any output."""
    with pytest.raises(ValueError, match="d0_zero_count_unit must be refuse or half_count"):
        bg.run(bg_run_args(tmp_path / "d", d0_zero_count_unit="one"))
    assert not (tmp_path / "d").exists()


def test_half_count_is_refused_for_a_production_label_before_output(tmp_path, monkeypatch):
    """half_count is refused for a production label before any output."""
    def forbidden(*args, **kwargs):
        """Fail the test if provider setup is reached."""
        pytest.fail("refused run reached provider setup")
    monkeypatch.setattr(cp, "AnthropicCountProposer", forbidden)
    output = tmp_path / "cell"
    with pytest.raises(ValueError, match="--d0-zero-count-unit half_count"):
        bg.run(bg_run_args(output, task="peregrines", d0_zero_count_unit="half_count", run_label="production"))
    assert not output.exists()
    monkeypatch.setattr(bg, "bmd_label_receipt", lambda *args, **kw: {"run_label": "production"})
    with pytest.raises(ValueError, match="--d0-zero-count-unit half_count"):
        bg.run(bg_run_args(output, task="peregrines", d0_zero_count_unit="half_count"))
    assert not output.exists()


def label_args(**overrides):
    """Build a minimal run-label argument namespace, with optional overrides."""
    return SimpleNamespace(rounds=1, profile="dev", cell="111", seed=37, proposer="mock",
                           d0_policy="random2", q_policy="native", prompt_version="v2.0",
                           prompt_sha256="0" * 64, **overrides)


def test_the_label_settings_register_the_option_exactly_when_stated():
    """The label settings register the option exactly when it is stated."""
    unstated = policy.bmd_runtime_settings("boxinggym", label_args())
    refused = policy.bmd_runtime_settings("boxinggym", label_args(d0_zero_count_unit="refuse"))
    assert unstated == refused and "d0_zero_count_unit" not in unstated
    stated = policy.bmd_runtime_settings("boxinggym", label_args(d0_zero_count_unit="half_count"))
    assert stated == {**unstated, "d0_zero_count_unit": "half_count"}
    with pytest.raises(ValueError, match="d0_zero_count_unit must be refuse or half_count"):
        policy.bmd_runtime_settings("boxinggym", label_args(d0_zero_count_unit="one"))
    # The other bundles do not read the option.
    core = policy.bmd_runtime_settings("core", argparse.Namespace(rounds=1, profile="dev", cell="111", seed=11,
                                                                  proposer="mock", d0_policy="fixed4",
                                                                  prompt_version="v2.0", prompt_sha256="0" * 64,
                                                                  d0_zero_count_unit="half_count"))
    assert "d0_zero_count_unit" not in core


def test_the_snapshot_policy_mirror_is_the_canonical_module():
    """The snapshot copy of the label policy is byte-identical to the canonical module."""
    from pathlib import Path
    root = Path(bg.__file__).resolve().parents[3]
    assert (root / "tools/fb1_snapshot/run_label_policy.py").read_bytes() == Path(policy.__file__).read_bytes()


def one_round(output, **overrides):
    """Run one BoxingGym round on peregrines seed 37 with random2 start, with optional overrides."""
    return bg.run(bg_run_args(output, **{"task": "peregrines", "seed": 37, "d0_policy": "random2", "rounds": 1,
                                         **overrides}))


@pytest.mark.usefixtures("require_boxinggym_source")
def test_peregrines_seed_37_random2_starts_at_two_zero_counts_and_is_refused_by_default(tmp_path):
    """Peregrines seed 37 with random2 starts at two zero counts and is refused by default."""
    with pytest.raises(ValueError, match="D0 response unit is not positive"):
        one_round(tmp_path / "refused")
    rows = [json.loads(line) for line in (tmp_path / "refused" / "observations.jsonl").read_text().splitlines()]
    assert [row["y"] for row in rows] == [0, 0] and all(row["round"] == 0 for row in rows)
    assert not (tmp_path / "refused" / "frozen_d0_policy.json").exists()


@pytest.mark.usefixtures("require_boxinggym_source")
def test_half_count_runs_peregrines_seed_37_and_records_the_unit(tmp_path):
    """half_count runs peregrines seed 37 and records the unit."""
    one_round(tmp_path / "half", d0_zero_count_unit="half_count")
    frozen = json.loads((tmp_path / "half" / "frozen_d0_policy.json").read_text())
    assert frozen["response_unit"] == 0.25 and frozen["d0_zero_count_unit"] == "half_count"
    assert frozen["zero_count_unit_applied"] is True
    assert frozen["policy"] == "frozen_D0_mean_of_the_four_initial_observations"
    config = json.loads((tmp_path / "half" / "config.json").read_text())
    assert config["d0_zero_count_unit"] == "half_count" and config["zero_count_unit_applied"] is True
    assert config["d0_policy"] == "random2" and config["d0_count"] == 2
    rows = [json.loads(line) for line in (tmp_path / "half" / "observations.jsonl").read_text().splitlines()]
    assert [row["y"] for row in rows[:2]] == [0, 0] and len(rows) == 3
    assert (tmp_path / "half" / "summary.json").is_file()


@pytest.mark.usefixtures("require_boxinggym_source")
def test_half_count_with_a_positive_start_records_the_option_as_not_applied(tmp_path):
    """With a positive start, half_count is recorded as not applied."""
    one_round(tmp_path / "positive", seed=31, d0_zero_count_unit="half_count")
    frozen = json.loads((tmp_path / "positive" / "frozen_d0_policy.json").read_text())
    assert frozen["response_unit"] > 0 and frozen["zero_count_unit_applied"] is False
    assert frozen["d0_zero_count_unit"] == "half_count"
    config = json.loads((tmp_path / "positive" / "config.json").read_text())
    assert config["zero_count_unit_applied"] is False
    rows = [json.loads(line) for line in (tmp_path / "positive" / "observations.jsonl").read_text().splitlines()]
    assert sum(row["y"] for row in rows[:2]) / 2 == frozen["response_unit"]
