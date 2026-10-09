"""Graceful stop: stop only at a round boundary, then use the unchanged official scorer."""

import csv
import json
import math
import sys

import pytest

from . import chembench_core as cc
from . import test_chembench_core as fake
from .test_chembench_mda_options import stub_fits


@pytest.fixture(autouse=True)
def synthetic_world(monkeypatch):
    """Fixture patching in a small synthetic world for the run."""
    stub_fits(monkeypatch)


def read_json(path):
    """Load and return a JSON file."""
    return json.loads(path.read_text())


def assert_scored(out, last_round, every):
    """Assert that the run was scored over the completed rounds."""
    with (out / "points.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == last_round + 1
    assert [int(row["x"]) for row in rows] == list(range(4, last_round + 5))
    for index, row in enumerate(rows):
        for column in ("y_primary", "nll", "coverage90", "width90"):
            assert math.isfinite(float(row[column])), (index, column)
        if every or index == last_round:
            assert math.isfinite(float(row["kl"]))
        else:
            assert row["kl"] == ""
    scored = read_json(out / "official_scores_after_finish.json")
    assert scored["evaluation_set"]["count"] == 1000
    assert scored["evaluation_set"]["requested_after_finish_only"] is True
    assert [row["round"] for row in scored["rounds"]] == list(range(last_round + 1))
    timing = [json.loads(line) for line in (out / "round_timings.jsonl").read_text().splitlines()]
    assert [row["round"] for row in timing] == list(range(last_round + 1))
    assert not (out / "failure.json").exists()


@pytest.mark.parametrize("every", [False, True])
@pytest.mark.parametrize("last_round", [0, 2])
def test_stop_scores_completed_rounds_and_never_changes_the_marker(
    tmp_path, monkeypatch, capsys, last_round, every,
):
    """Check that stop scores completed rounds and never changes the marker."""
    marker = tmp_path / "stop.request"
    marker_state = []

    def request_stop():
        """Create the stop-request marker mid-run."""
        marker.write_bytes(b"synthetic stop request\n")
        marker_state.append((marker.read_bytes(), marker.stat().st_mtime_ns))

    if last_round == 0:
        request_stop()
    else:
        original = cc.record_round_timing

        def timing(out, t, *args):
            """Stub timing hook that may request a stop after a given round."""
            original(out, t, *args)
            if t == last_round:
                request_stop()

        monkeypatch.setattr(cc, "record_round_timing", timing)

    # The real fake-session guard raises if any reference is read before finish().
    out = tmp_path / "cell"
    args = fake._args(out, fake._manifest(tmp_path), rounds=3,
                      stop_file=str(marker), kl_every_checkpoint=every)
    assert cc.run(args) is None
    assert_scored(out, last_round, every)
    assert (marker.read_bytes(), marker.stat().st_mtime_ns) == marker_state[0]
    summary = read_json(out / "summary.json")
    assert summary["status"] == "stopped_on_request"
    assert summary["rounds"] == 3
    assert summary["fit_count"] == 4 * (last_round + 1)
    assert summary["stop"] == {
        "reason": "stop_requested", "stop_file": str(marker.resolve()),
        "stop_file_mtime_unix": marker.stat().st_mtime,
        "detected_before_round": last_round + 1,
        "last_completed_round": last_round, "rounds_planned": 3,
    }
    assert read_json(out / "config.json")["stop_file"] == str(marker)
    assert capsys.readouterr().err == (
        f"STOP requested: stopped before round {last_round + 1}; official scores written\n"
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_no_request_completes_without_stop_metadata_or_marker_writes(tmp_path, capsys, explicit):
    """Check that no request completes without stop metadata or marker writes."""
    marker = tmp_path / "absent.request"
    out = tmp_path / "cell"
    extra = {"stop_file": str(marker)} if explicit else {}
    cc.run(fake._args(out, fake._manifest(tmp_path), rounds=1, **extra))
    assert_scored(out, 1, False)
    summary = read_json(out / "summary.json")
    assert summary["status"] == "completed_fixed_pool_first_path"
    assert "stop" not in summary
    assert ("stop_file" in read_json(out / "config.json")) == explicit
    assert not marker.exists()
    assert capsys.readouterr().err == ""


def test_relative_marker_is_recorded_as_absolute_and_round_zero_always_runs(tmp_path, monkeypatch):
    """Check that relative marker is recorded as absolute and round zero always runs."""
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "stop.request"
    marker.write_text("synthetic request")
    out = tmp_path / "cell"
    cc.run(fake._args(out, fake._manifest(tmp_path), rounds=1, stop_file=marker.name))
    assert read_json(out / "summary.json")["stop"]["stop_file"] == str(marker)
    assert_scored(out, 0, False)


def test_success_message_is_not_written_if_official_scoring_fails(tmp_path, monkeypatch, capsys):
    """Check that success message is not written if official scoring fails."""
    marker = tmp_path / "stop.request"
    marker.write_text("synthetic request")

    def failing(*args, **kwargs):
        """Stand-in that raises, simulating a failed official scoring."""
        raise RuntimeError("synthetic scorer failure")

    monkeypatch.setattr(cc, "official_checkpoint_scores", failing)
    out = tmp_path / "cell"
    with pytest.raises(RuntimeError, match="synthetic scorer failure"):
        cc.run(fake._args(out, fake._manifest(tmp_path), stop_file=str(marker)))
    assert "official scores written" not in capsys.readouterr().err
    assert not (out / "summary.json").exists()
    assert (out / "failure.json").exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_suppresses_an_omitted_option_and_finishes_with_exit_zero(tmp_path, monkeypatch, explicit):
    """Check that cli suppresses an omitted option and finishes with exit zero."""
    captured = []
    run = cc.run

    def recording(args):
        """Stub entry point that records the parsed arguments."""
        captured.append(vars(args).copy())
        return run(args)

    marker = tmp_path / "stop.request"
    marker.write_text("synthetic request")
    out = tmp_path / "cell"
    argv = ["chembench_core", "--manifest", str(fake._manifest(tmp_path)),
            "--output", str(out), "--noise", "0.01", "--rounds", "1",
            "--particles", "16", "--q", "8"]
    if explicit:
        argv += ["--stop-file", str(marker)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(cc, "run", recording)
    assert cc.main() is None
    assert ("stop_file" in captured[0]) == explicit
    assert_scored(out, 0 if explicit else 1, False)
