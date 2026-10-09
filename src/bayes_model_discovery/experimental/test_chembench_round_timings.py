"""Per-round timing record (2026-09-26): ChemBench's observational per-round timing record.

`round_timings.jsonl` in the cell directory gets one row per completed round: `round`,
`started_at_unix` (time.time at the top of the round's loop iteration), `finished_at_unix`
(time.time after the round's last work) and `elapsed_seconds` (perf_counter difference). An
all_designs_invalid or acquisition_failed round completes and writes its row; a round that
raises writes none; no config, summary or receipt names the file. Fake eight-input world, mock
proposer, no provider.
"""

import itertools
import json
import math

import pytest

from . import chembench_core
from . import test_agx_chembench_fallback as fallback
from . import test_chembench_core as fake

KEYS = ["elapsed_seconds", "finished_at_unix", "round", "started_at_unix"]
ROUNDS = 2


def rows(run_dir):
    """Read the per-round timing rows from a run directory."""
    return [json.loads(line) for line in (run_dir / "round_timings.jsonl").read_text().splitlines()]


def assert_timing_rows(run_dir, rounds):
    """Assert that the timing rows have the expected shape and order."""
    lines = (run_dir / "round_timings.jsonl").read_text().splitlines()
    table = [json.loads(line) for line in lines]
    assert [row["round"] for row in table] == list(rounds)
    for line, row in zip(lines, table, strict=True):
        assert list(row) == KEYS and line == json.dumps(row, sort_keys=True, allow_nan=False)
        assert type(row["round"]) is int
        values = [row[key] for key in KEYS if key != "round"]
        assert all(type(value) is float and math.isfinite(value) for value in values)
        assert row["started_at_unix"] <= row["finished_at_unix"] and row["elapsed_seconds"] >= 0
    for earlier, later in itertools.pairwise(table):
        assert earlier["finished_at_unix"] <= later["started_at_unix"]
    return table


def assert_named_nowhere(run_dir):
    """Assert that the given name appears in no output file."""
    for path in run_dir.rglob("*.json"):
        assert "round_timings" not in path.read_text(), path.name


@pytest.mark.usefixtures("require_autoscilab_source")
@pytest.mark.parametrize("cell", ("001", "101"))
def test_a_short_mock_run_writes_one_row_per_round_in_order(tmp_path, cell):
    """Check that a short mock run writes one row per round in order."""
    out = tmp_path / "cell"
    fallback.run_cell(out, fake._manifest(tmp_path), cell, rounds=ROUNDS)
    assert not (out / "failure.json").exists()
    assert_timing_rows(out, range(ROUNDS + 1))
    assert_named_nowhere(out)


@pytest.mark.usefixtures("require_autoscilab_source")
@pytest.mark.parametrize("fallback_valid", (True, False))
def test_a_fallback_round_completes_and_writes_its_row(tmp_path, fallback_valid):
    """Round 1 is all_designs_invalid (a valid fallback observes cell 100's design) or
    acquisition_failed (no observation, no refit); either way it completes."""
    out = tmp_path / "cell"
    fallback.run_cell(out, fake._manifest(tmp_path), "101", patch_invalid=fallback_valid)
    status = json.loads((out / "acquisition" / "t01.json").read_text())["acquisition_status"]
    assert status == ("all_designs_invalid" if fallback_valid else "acquisition_failed")
    assert_timing_rows(out, range(ROUNDS + 1))


@pytest.mark.usefixtures("require_autoscilab_source")
def test_a_round_that_raises_leaves_no_row_for_itself(tmp_path, monkeypatch):
    """Check that a round that raises leaves no row for itself."""
    real = chembench_core.guarded_predictive_summary

    def raising(fits, inputs, seed, round_index, *args, **kwargs):
        """Stub step that raises on the chosen round."""
        if round_index == ROUNDS:
            raise RuntimeError("synthetic failure in the last round's evaluation step")
        return real(fits, inputs, seed, round_index, *args, **kwargs)

    monkeypatch.setattr(chembench_core, "guarded_predictive_summary", raising)
    out = tmp_path / "cell"
    with pytest.raises(RuntimeError, match="synthetic failure"):
        fallback.run_cell(out, fake._manifest(tmp_path), "101", rounds=ROUNDS)
    assert (out / "failure.json").exists()
    assert_timing_rows(out, range(ROUNDS))


@pytest.mark.usefixtures("require_autoscilab_source")
def test_a_pool_generation_run_writes_its_one_round(tmp_path, monkeypatch):
    """Check that a pool generation run writes its one round."""
    from . import test_chembench_frozen_pool as frozen

    frozen.stub_fit(monkeypatch)
    out = frozen.generate(tmp_path / "generation")
    assert json.loads((out / "summary.json").read_text())["status"] == "initial_pool_generated"
    assert_timing_rows(out, [0])
    assert_named_nowhere(out)
