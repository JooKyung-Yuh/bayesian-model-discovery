"""Guard rules C1-C5 at ChemBench level: the fallback of an all_designs_invalid round.

The world is test_chembench_core's eight-input fake adapter (f0..f7 on [0.1, 1.0]) with
autoscilab's three ChemBench declaration constants pointed at the same box, as in
test_agx_ascent_nonfinite_mean. Round 1's acquisition is made all-invalid through the guard's
only input, `chembench_core.particle_means_masked`: while round 1's projected_spsa runs every
mask is False; for the one fallback check after it the mask is either the real one (a valid
fallback) or False (an invalid fallback). The guard itself, joint_eig, design_search and the
round loop are the real code. Synthetic data only; seed 31 is a cell seed.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from . import chembench_core, design_search
from . import test_chembench_core as fake

SEED = 31
FAILING_ROUND = 1
#: Operational by name (the classification of operational files and fields): wall-time files and fields.
OPERATIONAL_FILES = ("acquisition_timings.jsonl", "fit_timings.jsonl",
                     "round_timings.jsonl")  # ChemBench's per-round wall time
OPERATIONAL_FIELDS = {"summary.json": ("wall_seconds", "user_cpu_seconds", "system_cpu_seconds",
                                       "timing_seconds"),
                      "fits/*.json": ("fit_seconds", "diagnostics.sampler_attrs._t_sampling"),
                      # Digests of proposals/t00_burst.json, whose bytes hold the output path.
                      "config.json": ("initial_pool_sha256", "initial_pool_receipt.sha256"),
                      "initial_pool_receipt.json": ("initial_pool_sha256",)}


def install_world(patch, observed):
    """Install the eight-input fake world, recording every observed design in `observed`."""
    import autoscilab.oracle.chembench as oracle

    class RecordingSession(fake._FakeSession):
        """Fake session that records every design it observes."""
        def observe(self, design):
            """Record the design, then delegate to the fake session."""
            observed.append(design)
            return super().observe(design)

    patch.setattr(fake.real_adapter, "ChemBenchSession", RecordingSession)
    patch.setattr(fake.real_adapter, "sample_designs", fake._fake_sample_designs)
    patch.setattr(oracle, "CHEM_INPUT_VARS", list(fake.FAKE_NAMES))
    patch.setattr(oracle, "CHEM_INPUT_BOUNDS",
                  {name: tuple(bounds) for name, bounds in fake.FAKE_BOUNDS.items()})
    patch.setattr(oracle, "CHEM_LOG_VARS", set())


def cell_args(output, manifest, cell, rounds=2):
    """Build the runner arguments for one cell of the fallback tests."""
    return fake._args(output, manifest, cell=cell, seed=SEED, rounds=rounds, particles=16, q=8,
                      structure_proposals=None, proposer="mock", design_policy="joint_eig",
                      **({"design_search": "ascent"} if cell[2] == "1" else {}))


def force_round_one_invalid(patch, fallback_valid):
    """Every guard mask False during round 1's search; the fallback check real or False."""
    state = {"round": None, "inside": False, "fallback_checks": 0, "search_calls": 0}
    real_search = design_search.projected_spsa
    real_masked = chembench_core.particle_means_masked
    real_guard = chembench_core._acquisition_bank_for_points

    def search(*args, round_index, **kwargs):
        state.update(round=round_index, inside=True)
        try:
            return real_search(*args, round_index=round_index, **kwargs)
        finally:
            state["inside"] = False

    def masked(result, inputs):
        means, valid = real_masked(result, inputs)
        if state["round"] != FAILING_ROUND or (not state["inside"] and fallback_valid):
            return means, valid
        return means, np.zeros_like(valid)

    def guard(fits, points):
        if state["round"] == FAILING_ROUND:
            state["search_calls" if state["inside"] else "fallback_checks"] += 1
        return real_guard(fits, points)

    patch.setattr(design_search, "projected_spsa", search)
    patch.setattr(chembench_core, "particle_means_masked", masked)
    patch.setattr(chembench_core, "_acquisition_bank_for_points", guard)
    return state


def run_cell(output, manifest, cell, patch_invalid=None, rounds=2):
    """Run one cell, optionally forcing round 1 invalid; return the observed designs and the spy state."""
    observed = []
    with pytest.MonkeyPatch.context() as patch:
        install_world(patch, observed)
        state = force_round_one_invalid(patch, patch_invalid) if patch_invalid is not None else None
        chembench_core.run(cell_args(output, manifest, cell, rounds))
    return observed, state


def read(path):
    """Load a JSON file."""
    return json.loads(Path(path).read_text())


def science_view(run_dir):
    """Every file of a run, keyed by its path: bytes with the run's own output path replaced,
    or, for a file with operational fields, its canonical JSON without them."""
    run_dir = Path(run_dir)
    spellings = sorted({str(run_dir), str(run_dir.resolve())}, key=len, reverse=True)

    def placeholder(text):
        for spelling in spellings:
            text = text.replace(spelling, "<run>")
        return text

    view = {}
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        name = str(path.relative_to(run_dir))
        if name in OPERATIONAL_FILES:
            continue
        fields = next((fields for pattern, fields in OPERATIONAL_FIELDS.items()
                       if Path(name).match(pattern)), None)
        if fields is not None:
            record = json.loads(path.read_bytes())
            for field in fields:
                *parents, leaf = field.split(".")
                holder = record
                for key in parents:
                    holder = holder.get(key, {}) if isinstance(holder, dict) else {}
                if isinstance(holder, dict):
                    holder.pop(leaf, None)
            view[name] = placeholder(json.dumps(record, sort_keys=True)).encode()
        else:
            data = path.read_bytes()
            for spelling in spellings:
                data = data.replace(spelling.encode(), b"<run>")
            view[name] = data
    return view


@pytest.mark.usefixtures("require_autoscilab_source")
def test_all_designs_invalid_observes_the_design_cell_100_observes_in_that_round(tmp_path):
    """An all-invalid round observes the same design that cell 100 observes in that round."""
    manifest = fake._manifest(tmp_path)
    ascent, state = run_cell(tmp_path / "ascent" / "cell", manifest, "101", patch_invalid=True)
    random, _ = run_cell(tmp_path / "random" / "cell", manifest, "100", rounds=1)
    out = tmp_path / "ascent" / "cell"
    record = read(out / "acquisition" / "t01.json")
    assert record["acquisition_status"] == "all_designs_invalid"
    assert record["chosen"] is None and record["chosen_restart_index"] is None
    assert (record["fallback_policy"], record["fallback_design_index"], record["fallback_valid"]) == (
        "fixed_random_designs_from_native_input_distribution", 4, True)
    assert "fallback_excluded_particles" not in record and "fallback_excluded_models" not in record
    # Every call of the round's search excluded every model, and the log says so.
    summary = record["eig_exclusion_summary"]
    assert (summary["objective_call_count"] == summary["invalid_objective_call_count"]
            == summary["calls_with_excluded_particles"] == record["objective_call_count"]
            == state["search_calls"])
    assert record["eig_excluded_models"] == dict.fromkeys(record["policy"]["active_ids"],
                                                          record["objective_call_count"])
    assert len(record["invalid_design_points"]) == record["objective_evaluation_count"]
    # Rule C5: the policy keeps the runner's own five keys.
    for key in ("design_policy", "eig_samples", "eig_estimator", "model_weights", "active_ids"):
        assert key in record["policy"], key
    assert state["fallback_checks"] == 1
    # The observed design is cell 100's round-1 design: the same dict, the same observation row.
    ascent_rows = (out / "observations.jsonl").read_bytes().splitlines()
    random_rows = (tmp_path / "random" / "cell" / "observations.jsonl").read_bytes().splitlines()
    assert ascent_rows[:5] == random_rows[:5]
    assert ascent[4] == random[4] and json.dumps(ascent[4]) == json.dumps(random[4])
    assert record["fallback_design"] == [random[4][name] for name in fake.FAKE_NAMES]
    # Rule C3: the round's score row says it was not an ascent.
    scores = read(out / "official_scores_after_finish.json")
    assert [row.get("acquisition_status") for row in scores["rounds"]] == [
        None, "all_designs_invalid", None]
    # Round 2 is an ordinary ascent round.
    assert not {"acquisition_status", "fallback_policy"} & set(read(out / "acquisition" / "t02.json"))


@pytest.mark.usefixtures("require_autoscilab_source")
@pytest.mark.parametrize("cell", ("101", "001"))
def test_an_invalid_fallback_is_acquisition_failed_with_no_observation_and_no_refit(
        tmp_path, cell):
    """An invalid fallback makes the round acquisition_failed with no observation and no refit."""
    manifest = fake._manifest(tmp_path)
    out = tmp_path / "cell"
    observed, state = run_cell(out, manifest, cell, patch_invalid=False)
    record = read(out / "acquisition" / "t01.json")
    carried = record["policy"]["active_ids"]
    assert record["acquisition_status"] == "acquisition_failed"
    assert record["fallback_valid"] is False and record["fallback_design_index"] == 4
    assert record["fallback_excluded_models"] == carried
    sizes = {}
    for model_id in carried:
        with np.load(out / "fits" / f"t00_{model_id}.npz") as saved:
            sizes[model_id] = len(saved["b0"])
    assert record["fallback_excluded_particles"] == {
        model_id: {"excluded": size, "of": size} for model_id, size in sizes.items()}
    assert state["fallback_checks"] == 1
    # No observation in round 1: four D0 rows and round 2's, and the session saw five calls.
    rows = (out / "observations.jsonl").read_text().splitlines()
    assert len(rows) == len(observed) == 5
    assert read(out / "summary.json")["observations"] == 5
    round_two = read(out / "acquisition" / "t02.json")
    assert observed[4] == dict(zip(fake.FAKE_NAMES, round_two["chosen"], strict=True))
    # No refit of the carried models in round 1; round 2 refits them on the unchanged data.
    for model_id in carried:
        assert (out / "fits" / f"t00_{model_id}.npz").exists()
        assert not (out / "fits" / f"t01_{model_id}.npz").exists()
        assert not (out / "fits" / f"t01_{model_id}.json").exists()
    # The round still predicts and scores on the kept fits, flagged, and the next round runs.
    assert (out / "predictions" / "t01.npz").exists()
    scores = read(out / "official_scores_after_finish.json")["rounds"]
    assert [row.get("acquisition_status") for row in scores] == [None, "acquisition_failed", None]
    assert [row["observation_count"] for row in scores] == [4, 4, 5]
    assert round_two["chosen"] is not None and "acquisition_status" not in round_two
    header = (out / "points.csv").read_text().splitlines()[0]
    assert "acquisition_status" not in header
    assert len((out / "points.csv").read_text().splitlines()) == 1 + 3
    summary = read(out / "summary.json")
    if cell == "001":
        # Rule C4: catalog mode reports the fits actually run, not 4 * (rounds + 1).
        assert summary["fit_count"] == len(list((out / "fits").glob("*.npz"))) == 4 * 3 - 4
        assert summary["active_pool"] == 4
    else:
        assert summary["fit_count"] == len(list((out / "fits").glob("*.npz")))


@pytest.mark.usefixtures("require_autoscilab_source")
def test_an_all_valid_ascent_cell_is_deterministic_and_writes_no_agx_key(tmp_path):
    """An all-valid ascent cell is deterministic and writes none of the guard's record keys."""
    manifest = fake._manifest(tmp_path)
    first, second = tmp_path / "a" / "cell", tmp_path / "b" / "cell"
    run_cell(first, manifest, "101", rounds=1)
    run_cell(second, manifest, "101", rounds=1)
    left, right = science_view(first), science_view(second)
    assert list(left) == list(right)
    assert [name for name in left if left[name] != right[name]] == []
    record = read(first / "acquisition" / "t01.json")
    assert not {"acquisition_status", "invalid_design_points", "noise_floor_status",
                "eig_excluded_particles", "eig_excluded_models", "eig_exclusion_calls",
                "eig_exclusion_summary", "fallback_policy"} & set(record)
