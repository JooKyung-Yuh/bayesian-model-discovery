"""The acquisition guard's reproduction setup: what the guard records, and that the run is deterministic.

test_agx_ascent_nonfinite_mean (absorbed unchanged from the preparation) asserts only that the cell
finishes. Here the same world and the same reproduction condition (its own `install_world`,
`install_reproduction` and `cell_args`, imported, not copied) run twice: the first run's round-1
record must carry the exclusion keys and no failure.json may exist, and the two runs must
agree file for file once the named operational fields are removed (invalid path).
"""

import importlib.util
import json
import os

import numpy as np
import pytest

from . import chembench_core, design_search
from . import test_agx_ascent_nonfinite_mean as prep
from . import test_chembench_core as fake
from .count_models import DEFAULT_GRAMMAR_VERSION, restore_grammar_state
from .test_agx_chembench_fallback import science_view

AGX_KEYS = ("acquisition_status", "eig_excluded_particles", "eig_excluded_models",
            "eig_exclusion_calls", "eig_exclusion_summary", "eig_exclusion_guard")


@pytest.fixture(scope="module")
def reproduction_runs(tmp_path_factory):
    """Run the reproduction cell twice with the same manifest and return (run directory, inflated record) pairs."""
    # The per-test `require_autoscilab_source` fixture checks the pinned source; a module
    # fixture cannot use it, so it only skips when the source is absent altogether.
    if not os.environ.get("BMD_AUTOSCILAB") or importlib.util.find_spec("autoscilab") is None:
        pytest.skip("autoscilab requires BMD_AUTOSCILAB and importable source")
    runs = []
    # One manifest file for both runs, so config.json's manifest path is the same.
    manifest = fake._manifest(tmp_path_factory.mktemp("manifest"))
    for name in ("first", "second"):
        restore_grammar_state((DEFAULT_GRAMMAR_VERSION, {}))
        base = tmp_path_factory.mktemp(name)
        inflated = {}
        with pytest.MonkeyPatch.context() as patch:
            prep.install_world(patch)
            prep.install_reproduction(patch, inflated)
            chembench_core.run(prep.cell_args(base / "cell", manifest))
        runs.append((base / "cell", inflated))
    restore_grammar_state((DEFAULT_GRAMMAR_VERSION, {}))
    return runs


def read(path):
    """Return the parsed JSON content of `path`."""
    return json.loads(path.read_text())


@pytest.mark.usefixtures("require_autoscilab_source")
def test_the_reproduction_round_records_its_particle_exclusion_and_the_cell_finishes(
        reproduction_runs):
    """The failing round records its particle exclusion keys, no failure file exists, and the cell finishes."""
    run_dir, inflated = reproduction_runs[0]
    assert not (run_dir / "failure.json").exists()
    assert read(run_dir / "summary.json")["status"] == "completed_fixed_pool_first_path"
    model_id = inflated["model_id"]
    with np.load(run_dir / "fits" / f"t00_{model_id}.npz") as saved:
        size = len(saved["b0"])
    record = read(run_dir / "acquisition" / f"t{prep.FAILING_ROUND:02d}.json")
    # One particle of one model: excluded where it underflows, never the whole model, so every
    # call keeps a valid bank and no design point is invalid.
    assert record["acquisition_status"] == "ascent_with_exclusions"
    assert "invalid_design_points" not in record and "noise_floor_status" not in record
    assert record["chosen"] is not None and len(record["arrival_rows"]) == design_search.RESTARTS
    calls = record["eig_exclusion_calls"]
    assert calls and all(call == {"objective_call_index": call["objective_call_index"],
                                  "point_count": call["point_count"],
                                  "excluded_particles": {model_id: {"excluded": 1, "of": size}},
                                  "excluded_models": [], "invalid": False} for call in calls)
    # The step-1 perturbation batch the code that predates the guard died in is an excluded call. The
    # second version of the guard excludes the particle earlier as well: at the lowest of the ascent's starts its
    # mean is positive but subnormal (install_reproduction keeps it only above the smallest
    # subnormal), which the second guard version's representability bounds refuse. Later exclusions are whatever
    # calls reach that region again, in any phase. Each logged ordinal is a real call of this
    # record, with that call's own batch size.
    phases = {}
    for row in record["rows"] + record["perturbation_trace"]:
        phases.setdefault(row["objective_call_index"], (row["phase"], row["batch_size"]))
    indices = [call["objective_call_index"] for call in calls]
    assert indices[0] <= design_search.RESTARTS and design_search.RESTARTS in indices
    assert phases[design_search.RESTARTS] == ("perturbation", 2 * design_search.RESTARTS)
    assert all(phases[call["objective_call_index"]][1] == call["point_count"] for call in calls)
    assert indices == sorted(set(indices)) and indices[-1] < record["objective_call_count"]
    assert record["eig_excluded_particles"] == {model_id: {
        "of": size, "calls": len(calls), "excluded_total": len(calls), "excluded_max": 1}}
    assert record["eig_excluded_models"] == {}
    assert record["eig_exclusion_summary"] == {
        "objective_call_count": record["objective_call_count"],
        "calls_with_excluded_particles": len(calls), "invalid_objective_call_count": 0,
        "excluded_particle_count": len(calls), "excluded_model_count": 0,
        "retained_model_weighting": "model_logweights_over_retained_models"}
    assert model_id in record["policy"]["active_ids"]
    # Every round's record carries the keys exactly when its call log is non-empty.
    for t in range(1, prep.ROUNDS + 1):
        other = read(run_dir / "acquisition" / f"t{t:02d}.json")
        present = [key in other for key in AGX_KEYS]
        assert all(present) or not any(present)
        assert all(present) is bool(other.get("eig_exclusion_calls"))
    scores = read(run_dir / "official_scores_after_finish.json")["rounds"]
    assert not any("acquisition_status" in row for row in scores)


@pytest.mark.usefixtures("require_autoscilab_source")
def test_the_reproduction_is_deterministic_file_for_file(reproduction_runs):
    """Both reproduction runs agree file for file once operational fields are removed."""
    (first, first_inflated), (second, second_inflated) = reproduction_runs
    assert first_inflated == second_inflated
    left, right = science_view(first), science_view(second)
    assert list(left) == list(right)
    assert [name for name in left if left[name] != right[name]] == []
