"""The frozen KL rescore reproduces what a run recorded, and nothing else.

Every cell scored below is produced HERE, by the runners themselves on the dev
profile with the mock proposer over one or two rounds - never a results/ cell,
which this module's own contract keeps read-only and which no test may read.
The synthetic cases (a checkpoint with its `kl` removed, one with a shaken
`kl`, a missing fit, a shifted world, a shifted `q`, a truncated
`scores.jsonl`) are COPIES of those dev cells, edited under tmp_path, so a real
run directory is never modified even by a test.

What is pinned
--------------
1  death            the death-task path reproduces bit for bit, its independent
                    `kl_against_pool` cross-check agrees bit for bit, and the
                    pool weights rebuilt from `fits/` are the recorded ones.
2  CORE             the same, on `core_count`'s own `score_prediction` path.
3  the law paths    density (location), folded density (dugongs), count grid
                    (peregrines, `--kl-all-tasks`) and discrete binary
                    (survival), with each law's own diagnostics reproduced.
4  recomputed_only  a checkpoint with no recorded `kl` gets the same number and
                    an empty comparison rather than a silent 0.
5  mismatch         a shaken `kl` is reported, the table is still written, and
                    the exit code becomes 1.
6  rejections       the five refusals, each by its own reason.
7  determinism      two runs of one input produce identical bytes.
8  read-only        the cell tree's file list and hashes are unchanged.
9  emotion          the eight-component composite path, on a REAL dev run.
10 the header       the evaluator's declared identity is its actual identity.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ..provenance.canonical import is_hex64
from . import boxinggym_truth_law, frozen_kl_rescore

MODULE = frozen_kl_rescore


def adapter():
    """Import the BoxingGym adapter module, skipping the test when it is unavailable."""
    return pytest.importorskip(
        "bayes_model_discovery.experimental.boxinggym_core",
        reason="the adapter imports the read-only boxing_gym package",
    )


def boxinggym_args(output, task, **overrides):
    """The runner's argparse namespace, in the same form as test_death_kl.py uses."""
    bg = adapter()
    return SimpleNamespace(
        **{
            "output": str(output),
            "task": task,
            "profile": "dev",
            "proposer": "mock",
            "llm_model": None,
            "allow_paid_api": False,
            "rounds": 2,
            "seed": 11,
            "cell": "111",
            "prior_template": bg.DEFAULT_PRIOR_TEMPLATE,
            "priors": "off",
            "initial_family": None,
            "truth_in_pool": False,
            "inject_oracle_at": None,
            "prompt_variant": None,
            **overrides,
        }
    )


def run_boxinggym(directory, task, **overrides):
    """Run the BoxingGym runner for `task` into `directory` and return the directory."""
    bg = adapter()
    pytest.importorskip("pymc")
    pytest.importorskip(bg.TASKS[task].module)
    bg.run(boxinggym_args(directory, task, **overrides))
    return directory


def run_core(directory, **overrides):
    """`core_count.run` through its OWN parser, so no default is re-typed here."""
    core_count = pytest.importorskip("bayes_model_discovery.experimental.core_count")
    pytest.importorskip("pymc")
    argv = [
        "core_count",
        "--output",
        str(directory),
        "--profile",
        "dev",
        "--proposer",
        "mock",
        "--rounds",
        "2",
    ]
    for name, value in overrides.items():
        argv += [f"--{name.replace('_', '-')}", str(value)]
    saved, sys.argv = sys.argv, argv
    try:
        core_count.main()
    finally:
        sys.argv = saved
    return directory


# --------------------------------------------------------------------------
# the runs, made once for the whole session (each is a real runner execution)
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def death_cell(tmp_path_factory):
    """Session fixture: one real dev-profile run of the death task."""
    return run_boxinggym(tmp_path_factory.mktemp("death") / "run", "death")


@pytest.fixture(scope="session")
def core_cell(tmp_path_factory):
    """Session fixture: one real dev-profile CORE run."""
    return run_core(tmp_path_factory.mktemp("core") / "run")


@pytest.fixture(scope="session", params=["location", "dugongs", "peregrines", "survival"])
def k1_cell(request, tmp_path_factory):
    """Session fixture: one real dev-profile run per per-law task, as `(task, directory)`."""
    task = request.param
    directory = tmp_path_factory.mktemp(task) / "run"
    # peregrines is `KL_OPTIONAL_TASKS`' single member: its `kl` is recorded
    # only under the flag, which is exactly the path this test needs scored.
    extra = {"kl_all_tasks": True} if task == "peregrines" else {}
    return task, run_boxinggym(directory, task, **extra)


@pytest.fixture(scope="session")
def emotion_cell(tmp_path_factory):
    """Session fixture: one real one-round dev-profile run of the emotion task."""
    # One round: eight components are eight independent runners over one
    # experiment, so a second round buys another copy of the same composite
    # path at several minutes' cost.
    return run_boxinggym(tmp_path_factory.mktemp("emotion") / "run", "emotion", rounds=1)


@pytest.fixture(scope="session")
def moral_machines_cell(tmp_path_factory):
    """Session fixture: one real one-round dev-profile run of the moral_machines task."""
    return run_boxinggym(tmp_path_factory.mktemp("moral") / "run", "moral_machines", rounds=1)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def rescore(tmp_path, bundle, cells, name="table.json", extra=()):
    """`(exit code, table, output path)` from the CLI's own entry point."""
    output = tmp_path / name
    argv = ["--bundle", bundle]
    for cell in cells:
        argv += ["--cell", str(cell)]
    argv += ["--output", str(output), *extra]
    return MODULE.main(argv), json.loads(output.read_text()), output


def only(tmp_path, bundle, cell, **kwargs):
    """Rescore exactly one cell and return `(exit code, its record, output path)`."""
    code, table, output = rescore(tmp_path, bundle, [cell], **kwargs)
    assert len(table["cells"]) == 1
    return code, table["cells"][0], output


def tree_digest(directory):
    """Every file under `directory`, by relative path and content hash."""
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file()
    }


def copy_cell(cell, destination):
    """Copy a cell directory to `destination` so a test can edit it without touching the original."""
    shutil.copytree(cell, destination)
    return destination


def edit_last_score(cell, change):
    """Rewrite the copy's last `scores.jsonl` record through `change`."""
    path = Path(cell) / "scores.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows[-1] = change(rows[-1])
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def edit_config(cell, change):
    """Rewrite the copy's `config.json` through `change`."""
    path = Path(cell) / "config.json"
    config = json.loads(path.read_text())
    path.write_text(json.dumps(change(config), indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------
# 1. death: the death-task path, reproduced bit for bit
# --------------------------------------------------------------------------


def test_death_cell_is_reproduced_bit_for_bit_with_its_own_cross_check(tmp_path, death_cell):
    """The death cell's KL is reproduced bit for bit, agrees with its cross-check and records an audit trail."""
    code, record, output = only(tmp_path, "boxinggym", death_cell)
    assert code == 0
    assert record["status"] == "reproduced"
    assert record["task"] == "death" and record["cell_code"] == "111" and record["seed"] == 11
    assert record["round"] == 2 and record["phase"] == "after_proposal"
    # bit-identical, not merely within the tolerance.
    assert record["kl"] == record["stored_kl"]
    assert record["delta"] == 0.0
    # `core_count.score_prediction`'s sum and `kl_against_pool`'s sum are two
    # implementations of one quantity; they agree to the last bit.
    assert record["kl_cross_check"] == record["kl"]
    assert record["kl_method"] == "discrete_exact"
    assert record["max_abs_weight_delta"] <= 1e-15
    assert record["weights_recomputed"] == pytest.approx(record["weights_stored"], abs=1e-15)
    assert record["world_parameters_match"] is True
    assert record["tolerance"] == MODULE.REPRODUCTION_TOLERANCE

    # the audit trail: every file the rescore read, by path relative to the
    # cell and by its actual hash, with the fits it named.
    read = {entry["path"]: entry["sha256"] for entry in record["inputs"]}
    assert set(read) >= {"config.json", "scores.jsonl", "evaluation_scoring_only.npz"}
    for model_id in record["active_ids"]:
        assert f"fits/t02_{model_id}.json" in read
        assert f"fits/t02_{model_id}.npz" in read
    for relative, digest in read.items():
        assert not relative.startswith("/") and ".." not in relative
        assert is_hex64(digest)
        assert digest == hashlib.sha256((Path(death_cell) / relative).read_bytes()).hexdigest()
    assert record["cell"] == Path(death_cell).name and record["cell_path"] == record["cell"]
    assert output.with_name(output.name + ".sha256").is_file()


# --------------------------------------------------------------------------
# 2. CORE
# --------------------------------------------------------------------------


def test_core_cell_is_reproduced_bit_for_bit(tmp_path, core_cell):
    """The CORE cell's stored KL is reproduced bit for bit."""
    code, record, _ = only(tmp_path, "core", core_cell)
    assert code == 0
    assert record["status"] == "reproduced" and record["delta"] == 0.0
    assert record["kl"] == record["stored_kl"]
    assert record["task"] == "core" and record["kl_method"] == "discrete_exact"
    assert record["evaluation_truth_match"] is True
    assert record["round"] == 2 and record["phase"] == "after_proposal"
    assert record["max_abs_weight_delta"] <= 1e-15
    assert is_hex64(record["evaluation_x_sha256"])


# --------------------------------------------------------------------------
# 3. the four per-law paths, each with its law's own diagnostics
# --------------------------------------------------------------------------


EXPECTED_DIAGNOSTICS = {
    "location": ("kl_truncated_mass", "kl_quadrature_error"),
    "dugongs": ("kl_truncated_mass", "kl_quadrature_error"),
    "peregrines": ("kl_truncated_mass", "expected_log_score"),
    "survival": ("expected_log_score",),
}
ABSENT_DIAGNOSTICS = {
    "location": ("expected_log_score",),
    "dugongs": ("expected_log_score",),
    "peregrines": ("kl_quadrature_error",),
    "survival": ("kl_truncated_mass", "kl_quadrature_error"),
}


def test_every_k1_law_path_is_reproduced_with_its_diagnostics(tmp_path, k1_cell):
    """Each per-law path reproduces its stored KL and its own diagnostics, leaving the others null."""
    task, cell = k1_cell
    code, record, _ = only(tmp_path, "boxinggym", cell)
    assert code == 0
    assert record["task"] == task
    assert record["status"] == "reproduced" and record["delta"] == 0.0
    assert record["kl"] == record["stored_kl"]
    stored = json.loads(
        (Path(cell) / "scores.jsonl").read_text().splitlines()[-1]
    )["bma"]
    assert record["kl_method"] == stored["kl_method"]
    for key in EXPECTED_DIAGNOSTICS[task]:
        assert record[key] == stored[key]
    for key in ABSENT_DIAGNOSTICS[task]:
        # a diagnostic the task's own law did not make is null in the record
        # (every row carries RECORD_KEYS) and absent from the checkpoint.
        assert record[key] is None and key not in stored
    assert record["max_abs_weight_delta"] <= 1e-15


# --------------------------------------------------------------------------
# 4. recomputed_only: the same number, and an empty comparison
# --------------------------------------------------------------------------


def test_a_checkpoint_without_a_recorded_kl_is_recomputed_only(tmp_path, death_cell):
    """A checkpoint with no stored KL is `recomputed_only` and yields the same KL as the untouched cell."""
    _, reference, _ = only(tmp_path, "boxinggym", death_cell, name="reference.json")
    copy = copy_cell(death_cell, tmp_path / "no_kl")
    edit_last_score(copy, lambda row: {**row, "bma": {
        key: value for key, value in row["bma"].items() if key != "kl"}})

    code, record, _ = only(tmp_path, "boxinggym", copy, name="table.json")
    assert code == 0
    assert record["status"] == "recomputed_only"
    assert record["stored_kl"] is None and record["delta"] is None
    # the value itself is the one the untouched cell gives - removing the
    # recorded number changes the comparison, never the measurement.
    assert record["kl"] == reference["kl"]


# --------------------------------------------------------------------------
# 5. mismatch: reported, kept in the table, and exit code 1
# --------------------------------------------------------------------------


def test_a_shaken_stored_kl_is_a_mismatch_that_still_writes_the_table(tmp_path, death_cell):
    """A perturbed stored KL is a mismatch, the table is still written and the exit code is 1."""
    copy = copy_cell(death_cell, tmp_path / "shaken")
    shake = 1e-6
    edit_last_score(copy, lambda row: {**row, "bma": {**row["bma"], "kl": row["bma"]["kl"] + shake}})

    code, record, output = only(tmp_path, "boxinggym", copy)
    assert code == 1
    assert record["status"] == "mismatch"
    assert record["delta"] == pytest.approx(-shake, abs=1e-15)
    assert record["stored_kl"] == pytest.approx(record["kl"] + shake, abs=1e-15)
    assert output.is_file() and output.with_name(output.name + ".sha256").is_file()
    assert json.loads(output.read_text())["cells"][0]["status"] == "mismatch"


# --------------------------------------------------------------------------
# 5b. the death cross-check is enforced, not merely recorded
# --------------------------------------------------------------------------


def test_a_death_cross_check_that_disagrees_is_a_mismatch_of_its_own(
    tmp_path, death_cell, monkeypatch
):
    """`score_prediction` and `kl_against_pool` are two implementations of one
    sum; when they disagree beyond the tolerance neither number is trusted,
    whatever the record stored, and the exit code says so."""
    original = MODULE.boxinggym_truth_law.kl_against_pool
    shake = 1e-6

    def shaken(*args, **kwargs):
        out = original(*args, **kwargs)
        return {**out, "kl": out["kl"] + shake}

    monkeypatch.setattr(MODULE.boxinggym_truth_law, "kl_against_pool", shaken)
    code, record, output = only(tmp_path, "boxinggym", death_cell)
    assert code == 1
    assert record["status"] == "cross_check_mismatch"
    assert record["cross_check_delta"] == pytest.approx(-shake, abs=1e-15)
    assert record["kl_cross_check"] == pytest.approx(record["kl"] + shake, abs=1e-15)
    # the stored comparison is still recorded: the stored number agrees with
    # the un-shaken path, and only the cross-check took the status.
    assert record["stored_kl"] == record["kl"] and record["delta"] == 0.0
    assert json.loads(output.read_text())["cells"][0]["status"] == "cross_check_mismatch"


# --------------------------------------------------------------------------
# 5c. a task without a true law is not an input defect
# --------------------------------------------------------------------------


def test_a_cell_whose_task_has_no_truth_law_is_law_unavailable_and_neutral(
    tmp_path, death_cell, monkeypatch
):
    """Where `kl_truth_law_available` is false nothing can be scored; the row
    stays in the table with a null `kl`, `rejected` is kept for input defects,
    and the exit code is not raised by it."""
    monkeypatch.setattr(MODULE.boxinggym_truth_law, "truth_law", lambda *args: None)
    code, record, _ = only(tmp_path, "boxinggym", death_cell)
    assert code == 0
    assert record["status"] == "law_unavailable"
    assert record["kl"] is None and record["stored_kl"] is None and record["delta"] is None
    assert record["error_type"] is None and record["error_message"] is None
    config = json.loads((Path(death_cell) / "config.json").read_text())
    assert record["kl_truth_law_available"] == config["kl_truth_law_available"]
    assert record["world_parameters_match"] is True
    assert "law_unavailable" in MODULE.OK_STATUSES


# --------------------------------------------------------------------------
# 5d. one key set in every row (the table's fixed schema)
# --------------------------------------------------------------------------


def test_every_row_carries_the_same_key_set_whatever_its_path_or_status(
    tmp_path, death_cell, core_cell, emotion_cell, monkeypatch
):
    """Every row has the same key set, and a scorer that writes an extra key yields a failed row."""
    rejected = copy_cell(death_cell, tmp_path / "rejected")
    edit_config(rejected, lambda config: {**config, "q": config["q"] + 1})
    _, table, _ = rescore(tmp_path, "boxinggym", [death_cell, emotion_cell, rejected])
    _, core_table, _ = rescore(tmp_path, "core", [core_cell], name="core.json")
    rows = table["cells"] + core_table["cells"]
    assert [row["status"] for row in rows] == ["reproduced", "reproduced", "rejected", "reproduced"]
    expected = set(MODULE.RECORD_KEYS)
    assert len(expected) == len(MODULE.RECORD_KEYS)
    for row in rows:
        assert set(row) == expected, row["cell"]
    # a scorer that wrote a key outside the set is a failed row, not a new column.
    monkeypatch.setattr(MODULE, "rescore_core", lambda reader, record, option: record.update(extra=1))
    code, record, _ = only(tmp_path, "core", core_cell, name="drift.json")
    assert code == 1 and record["status"] == "failed"
    assert "extra" in record["error_message"]


# --------------------------------------------------------------------------
# 5e. an older generation's config.json (the earlier research wave) is accepted
# --------------------------------------------------------------------------

# The key sets of the research wave's published config.json files, as they were read
# on 2026-09-19 (key names only; every value below is synthetic). The
# boxinggym set predates two later config keys, so it carries neither `kl_recorded` nor
# `kl_truth_law_available`.
HISTORICAL_BOXINGGYM_CONFIG_KEYS = frozenset({
    "active_maximum", "allow_paid_api", "belief_packet", "cell", "created_utc", "d0_count",
    "d0_designs", "d0_policy", "d0_seed_label", "design_count", "design_pool_size",
    "eig_posterior_samples", "env_seed_injection", "env_seed_injection_rule",
    "eval_question_note", "eval_question_policy", "evaluation_seed", "evaluation_use",
    "evidence", "fit_timeout_seconds", "initial_family", "initial_pool", "initial_structures",
    "inject_oracle_at", "input_namespace", "interface", "label", "label_receipt", "llm_model",
    "manifest_ref", "model_prior", "output", "paid_api_calls", "prior_probe", "prior_template",
    "priors", "profile", "prompt_sha256", "prompt_variant", "prompt_version", "proposal_limit",
    "proposal_policy", "proposal_rounds", "proposer", "python", "q", "q_excluded_count",
    "q_policy", "repair_attempts", "rounds", "run_label", "seed", "smc", "source_files_sha256",
    "suite", "task", "task_spec", "truth_in_pool", "truth_mean_available", "versions",
    "world_parameters", "world_seed",
})
HISTORICAL_CORE_CONFIG_KEYS = frozenset({
    "active_maximum", "allow_paid_api", "archive_reevaluation", "archive_reevaluation_rounds",
    "belief_packet", "cell", "created_utc", "design_count", "design_policy",
    "design_policy_applies", "design_policy_target", "eig_posterior_samples",
    "evaluation_input_seed", "evaluation_outcome_seed", "evaluation_use", "evidence",
    "fit_timeout_seconds", "initial_family", "initial_pool", "interface", "label",
    "label_receipt", "llm_model", "model_prior", "output", "paid_api_calls", "points_variant",
    "prior_probe", "prior_template", "priors", "profile", "prompt_sha256", "prompt_version",
    "proposal_cadence", "proposal_limit", "proposal_policy", "proposal_rep_seed_consumed",
    "proposal_rounds", "proposer", "python", "q", "random_order_contract", "repair_attempts",
    "replicate", "rounds", "run_label", "seed", "seed_fields", "smc", "source_files_sha256",
    "structure", "truth_family", "versions",
})


def historical_config(cell, keys):
    """The copy's config.json narrowed to an older generation's key set: keys the
    generation did not write are dropped, keys it wrote that the copy lacks get a
    synthetic value, so the reader is exercised on exactly that key set."""
    def narrow(config):
        kept = {key: value for key, value in config.items() if key in keys}
        for key in keys - set(kept):
            kept[key] = "synthetic"
        return kept
    edit_config(cell, narrow)
    assert set(json.loads((Path(cell) / "config.json").read_text())) == keys


def test_the_required_config_keys_are_within_the_research_waves_key_sets():
    """The config keys the rescore requires are all present in the earlier research wave's key sets."""
    assert set(MODULE.BOXINGGYM_CONFIG_KEYS_REQUIRED) <= HISTORICAL_BOXINGGYM_CONFIG_KEYS
    assert set(MODULE.CORE_CONFIG_KEYS_REQUIRED) <= HISTORICAL_CORE_CONFIG_KEYS
    assert "kl_recorded" not in HISTORICAL_BOXINGGYM_CONFIG_KEYS
    assert "kl_truth_law_available" not in HISTORICAL_BOXINGGYM_CONFIG_KEYS


def test_a_cell_with_the_research_waves_boxinggym_config_keys_is_rescored(tmp_path, death_cell):
    """A BoxingGym cell with the earlier wave's config keys is still rescored, with the newer columns null."""
    _, reference, _ = only(tmp_path, "boxinggym", death_cell, name="reference.json")
    copy = copy_cell(death_cell, tmp_path / "a9cffea_keys")
    historical_config(copy, HISTORICAL_BOXINGGYM_CONFIG_KEYS)
    code, record, _ = only(tmp_path, "boxinggym", copy)
    assert code == 0 and record["status"] == "reproduced"
    assert record["kl"] == reference["kl"] and record["delta"] == 0.0
    # the two newer columns are null, not a KeyError and not a rejected row.
    assert record["kl_recorded"] is None and record["kl_truth_law_available"] is None
    assert set(record) == set(MODULE.RECORD_KEYS)


def test_a_cell_with_the_research_waves_core_config_keys_is_rescored(tmp_path, core_cell):
    """A CORE cell with the earlier wave's config keys is still rescored and reproduced."""
    _, reference, _ = only(tmp_path, "core", core_cell, name="reference.json")
    copy = copy_cell(core_cell, tmp_path / "a9cffea_core_keys")
    historical_config(copy, HISTORICAL_CORE_CONFIG_KEYS)
    code, record, _ = only(tmp_path, "core", copy)
    assert code == 0 and record["status"] == "reproduced"
    assert record["kl"] == reference["kl"] and record["delta"] == 0.0


# --------------------------------------------------------------------------
# 5f. narration tasks are rescored under the runner's own stub, with no key
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["emotion", "moral_machines"])
def test_a_narration_task_is_rescored_without_a_key_under_the_runners_stub(
    tmp_path, emotion_cell, moral_machines_cell, monkeypatch, task
):
    """moral_machines and emotion build `openai.OpenAI()` inside their environment;
    the runner installs `openai_narration_stub` before constructing it and so does
    the rescore, so a cell is rescored with the key variable absent (an empty
    variable would let the client construct and hide the defect)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cell = {"emotion": emotion_cell, "moral_machines": moral_machines_cell}[task]
    code, record, _ = only(tmp_path, "boxinggym", cell)
    assert code == 0
    assert record["status"] in ("reproduced", "recomputed_only")
    assert record["task"] == task and record["kl"] is not None
    assert isinstance(record["narration_calls_absorbed"], int)
    assert record["narration_calls_absorbed"] >= 0
    assert set(record) == set(MODULE.RECORD_KEYS)


def test_a_task_without_narration_records_no_absorbed_calls(tmp_path, death_cell):
    """A task without narration records no absorbed narration calls."""
    _, record, _ = only(tmp_path, "boxinggym", death_cell)
    assert record["narration_calls_absorbed"] is None


# --------------------------------------------------------------------------
# 5g. an honest infinity is recorded, not refused (an earlier run: emotion cells whose
#     recomputed mass is zero where the truth has mass)
# --------------------------------------------------------------------------


def _infinite_scores(*args, **kwargs):
    return {"kl": float("inf"), "expected_log_score": float("-inf"), "kl_method": "discrete_exact"}


def test_an_infinite_recomputed_kl_is_written_with_its_sign_and_the_table_stays_json(
    tmp_path, death_cell, monkeypatch
):
    """The value column is null, the `_nonfinite` sibling names the infinity, and the
    stored finite value is a mismatch against it; the table is still valid JSON."""
    monkeypatch.setattr(MODULE, "score_prediction", lambda *a, **k: ({"kl": float("inf"), "expected_log_score": float("-inf")}, None))
    monkeypatch.setattr(MODULE.boxinggym_truth_law, "kl_against_pool", _infinite_scores)
    code, record, output = only(tmp_path, "boxinggym", death_cell)
    assert code == 1 and record["status"] == "mismatch"
    assert record["kl"] is None and record["kl_nonfinite"] == "+inf"
    assert record["expected_log_score"] is None and record["expected_log_score_nonfinite"] == "-inf"
    assert record["kl_cross_check"] is None and record["kl_cross_check_nonfinite"] == "+inf"
    assert record["delta"] is None and record["delta_nonfinite"] == "+inf"
    # the two infinities agree with each other, so the cross-check did not fire.
    assert record["cross_check_delta"] is None and record["cross_check_delta_nonfinite"] == "nan"
    assert set(record) == set(MODULE.RECORD_KEYS)
    json.loads(output.read_text())


def test_an_infinite_recomputed_kl_without_a_stored_value_is_recomputed_only(tmp_path, death_cell, monkeypatch):
    """An infinite recomputed KL with no stored value is `recomputed_only`, recorded with its sign."""
    monkeypatch.setattr(MODULE, "score_prediction", lambda *a, **k: ({"kl": float("inf"), "expected_log_score": float("-inf")}, None))
    monkeypatch.setattr(MODULE.boxinggym_truth_law, "kl_against_pool", _infinite_scores)
    copy = copy_cell(death_cell, tmp_path / "no_kl_inf")
    edit_last_score(copy, lambda row: {**row, "bma": {k: v for k, v in row["bma"].items() if k != "kl"}})
    code, record, _ = only(tmp_path, "boxinggym", copy)
    assert code == 0 and record["status"] == "recomputed_only"
    assert record["kl"] is None and record["kl_nonfinite"] == "+inf"


def test_a_stored_infinity_reproduced_by_the_same_infinity_is_reproduced(tmp_path, death_cell, monkeypatch):
    """A stored infinite KL matched by the same recomputed infinity is `reproduced`."""
    monkeypatch.setattr(MODULE, "score_prediction", lambda *a, **k: ({"kl": float("inf"), "expected_log_score": float("-inf")}, None))
    monkeypatch.setattr(MODULE.boxinggym_truth_law, "kl_against_pool", _infinite_scores)
    copy = copy_cell(death_cell, tmp_path / "stored_inf")
    edit_last_score(copy, lambda row: {**row, "bma": {**row["bma"], "kl": float("inf")}})
    code, record, _ = only(tmp_path, "boxinggym", copy)
    assert code == 0 and record["status"] == "reproduced"
    assert record["stored_kl"] is None and record["stored_kl_nonfinite"] == "+inf"
    assert record["kl"] is None and record["kl_nonfinite"] == "+inf"
    assert record["delta"] is None and record["delta_nonfinite"] == "nan"


def test_an_infinite_component_of_a_composite_cell_is_recorded_per_component(tmp_path, emotion_cell, monkeypatch):
    """An infinite KL in one component of a composite cell is recorded for that component only."""
    original = MODULE.boxinggym_truth_law.kl_against_pool
    calls = []

    def one_component_infinite(law, fits, x=None, component=None):
        out = original(law, fits, x, component)
        calls.append(component)
        return {**out, "kl": float("inf")} if component == 0 else out

    monkeypatch.setattr(MODULE.boxinggym_truth_law, "kl_against_pool", one_component_infinite)
    code, record, output = only(tmp_path, "boxinggym", emotion_cell)
    assert code == 1 and record["status"] == "mismatch"
    assert record["kl"] is None and record["kl_nonfinite"] == "+inf"
    first = next(entry for entry in record["components"] if entry["component_index"] == 0)
    assert first["kl"] is None and first["kl_nonfinite"] == "+inf"
    others = [entry for entry in record["components"] if entry["component_index"] != 0]
    assert all(entry["kl"] is not None and "kl_nonfinite" not in entry for entry in others)
    json.loads(output.read_text())


# --------------------------------------------------------------------------
# 6. the refusals
# --------------------------------------------------------------------------


def test_a_missing_fit_file_is_rejected_by_name(tmp_path, death_cell):
    """A cell with a missing fit file is rejected, and the message names the file."""
    copy = copy_cell(death_cell, tmp_path / "no_fit")
    last = json.loads((Path(copy) / "scores.jsonl").read_text().splitlines()[-1])
    victim = Path(copy) / "fits" / f"t02_{last['active_ids'][0]}.json"
    victim.unlink()

    code, record, _ = only(tmp_path, "boxinggym", copy)
    assert code == 1
    assert record["status"] == "rejected" and record["error_type"] == "rejected"
    assert record["error_message"].startswith("fit file missing: ")
    assert victim.name in record["error_message"]


def test_a_shifted_world_parameter_is_rejected_with_both_values(tmp_path, death_cell):
    """A shifted world parameter is rejected, recording both the stored and the recomputed values."""
    copy = copy_cell(death_cell, tmp_path / "other_world")
    original = json.loads((Path(copy) / "config.json").read_text())["world_parameters"]
    shifted = {**original, "theta": original["theta"] + 0.5}
    edit_config(copy, lambda config: {**config, "world_parameters": shifted})

    code, record, _ = only(tmp_path, "boxinggym", copy)
    assert code == 1
    assert record["status"] == "rejected"
    assert record["error_message"] == "world_parameters mismatch"
    assert record["world_parameters_match"] is False
    assert record["world_parameters_stored"] == shifted
    assert record["world_parameters_recomputed"] == original


def test_a_shifted_q_is_rejected(tmp_path, death_cell):
    """A changed `q` (the evaluation point count) is rejected."""
    copy = copy_cell(death_cell, tmp_path / "other_q")
    edit_config(copy, lambda config: {**config, "q": config["q"] + 1})

    code, record, _ = only(tmp_path, "boxinggym", copy)
    assert code == 1
    assert record["status"] == "rejected"
    assert record["error_message"].startswith("q mismatch: ")


def test_a_last_record_that_is_not_the_final_after_proposal_is_rejected(tmp_path, death_cell):
    """A last record that is not the final `after_proposal` checkpoint is rejected."""
    copy = copy_cell(death_cell, tmp_path / "not_final")
    path = Path(copy) / "scores.jsonl"
    rows = [line for line in path.read_text().splitlines() if line.strip()]
    keep = [row for row in rows if json.loads(row)["phase"] != "after_proposal"]
    assert json.loads(keep[-1])["phase"] == "before_proposal"
    path.write_text("".join(row + "\n" for row in keep))

    code, record, _ = only(tmp_path, "boxinggym", copy)
    assert code == 1
    assert record["status"] == "rejected"
    assert record["error_message"] == "last record is not the final after_proposal checkpoint"


def test_an_output_inside_a_cell_directory_is_refused_before_anything_is_written(
    tmp_path, death_cell
):
    """An output path inside a cell directory is refused with exit code 2 before anything is written."""
    inside = Path(death_cell) / "table.json"
    with pytest.raises(SystemExit) as raised:
        MODULE.main(
            ["--bundle", "boxinggym", "--cell", str(death_cell), "--output", str(inside)]
        )
    assert raised.value.code == 2
    assert not inside.exists()
    assert not inside.with_name(inside.name + ".sha256").exists()


# --------------------------------------------------------------------------
# 7. determinism, and the declared cell order
# --------------------------------------------------------------------------


def test_two_runs_of_one_input_write_identical_bytes(tmp_path, death_cell):
    """Two runs on one input write identical table bytes and matching SHA-256 sidecar files."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    code_a, _, output_a = only(first, "boxinggym", death_cell)
    code_b, _, output_b = only(second, "boxinggym", death_cell)
    assert code_a == code_b == 0
    assert output_a.read_bytes() == output_b.read_bytes()
    digest_a = output_a.with_name(output_a.name + ".sha256").read_text()
    digest_b = output_b.with_name(output_b.name + ".sha256").read_text()
    assert digest_a == digest_b
    # SHA256SUMS form, over the table's own bytes, naming the file only.
    recorded, name = digest_a.split()
    assert name == output_a.name
    assert recorded == hashlib.sha256(output_a.read_bytes()).hexdigest()


def test_cell_order_follows_the_options_and_cells_root_sorts_by_name(tmp_path, death_cell):
    """Cells keep command-line order, while `--cells-root` sorts them by name and is recorded as a name."""
    copy_a = copy_cell(death_cell, tmp_path / "root" / "b_cell")
    copy_b = copy_cell(death_cell, tmp_path / "root" / "a_cell")
    _, stated, _ = rescore(tmp_path, "boxinggym", [copy_a, copy_b], name="stated.json")
    assert [record["cell"] for record in stated["cells"]] == ["b_cell", "a_cell"]
    assert stated["cells_root"] is None

    output = tmp_path / "rooted.json"
    code = MODULE.main(
        ["--bundle", "boxinggym", "--cells-root", str(tmp_path / "root"),
         "--output", str(output)]
    )
    rooted = json.loads(output.read_text())
    assert code == 0
    assert [record["cell"] for record in rooted["cells"]] == ["a_cell", "b_cell"]
    assert [record["cell_path"] for record in rooted["cells"]] == ["a_cell", "b_cell"]
    # the root reaches the table as a NAME, never as this machine's path.
    assert rooted["cells_root"] == "root"
    assert "/" not in json.dumps(rooted["cells_root"])


# --------------------------------------------------------------------------
# 8. the cell tree is read-only
# --------------------------------------------------------------------------


def test_the_cell_tree_is_untouched_by_a_rescore(tmp_path, death_cell, core_cell):
    """A rescore leaves the files and hashes of the cell tree unchanged."""
    before = {"boxinggym": tree_digest(death_cell), "core": tree_digest(core_cell)}
    for bundle, cell in (("boxinggym", death_cell), ("core", core_cell)):
        code, _, _ = only(tmp_path, bundle, cell, name=f"{bundle}.json")
        assert code == 0
    after = {"boxinggym": tree_digest(death_cell), "core": tree_digest(core_cell)}
    assert after == before
    assert set(after["boxinggym"]) and set(after["core"])


# --------------------------------------------------------------------------
# 9. emotion: the eight-component composite path, on a real dev run
# --------------------------------------------------------------------------


def test_emotion_reproduces_every_component_and_their_composite(tmp_path, emotion_cell):
    """The emotion cell reproduces each of its eight components and their composite."""
    bg = adapter()
    code, record, _ = only(tmp_path, "boxinggym", emotion_cell)
    assert code == 0
    assert record["status"] == "reproduced" and record["delta"] == 0.0
    assert record["kl_method"] == boxinggym_truth_law.EMOTION_CELL_METHOD
    assert record["composite_rule"] == bg.COMPOSITE_SCORE_RULE

    declared = tuple(bg.TASKS["emotion"].components)
    assert len(declared) == 8
    assert [entry["component"] for entry in record["components"]] == list(declared)
    assert [entry["component_index"] for entry in record["components"]] == list(range(8))

    stored_row = json.loads((Path(emotion_cell) / "scores.jsonl").read_text().splitlines()[-1])
    for entry, stored in zip(record["components"], stored_row["components"], strict=True):
        assert entry["kl"] == stored["bma"]["kl"] == entry["stored_kl"]
        assert entry["active_ids"] == stored["active_ids"]
        assert entry["max_abs_weight_delta"] <= 1e-15
    # the cell-level number is the components' own composite, by the runner's rule.
    assert record["kl"] == bg.composite_value([e["kl"] for e in record["components"]])
    assert record["max_abs_weight_delta"] == max(
        entry["max_abs_weight_delta"] for entry in record["components"]
    )
    # the component fits are read under the multi-component stem rule.
    read = {entry["path"] for entry in record["inputs"]}
    for entry in record["components"]:
        for model_id in entry["active_ids"]:
            assert f"fits/t01_{entry['component']}_{model_id}.json" in read


# --------------------------------------------------------------------------
# 10. the header states the evaluator's actual identity
# --------------------------------------------------------------------------


def test_the_header_declares_the_files_and_constants_that_produced_the_numbers(
    tmp_path, death_cell
):
    """The table header declares the module hashes, constants, functions and commit of the evaluator."""
    _, whole, _ = rescore(tmp_path, "boxinggym", [death_cell], name="whole.json")
    evaluator = whole["evaluator"]
    assert whole["schema"] == MODULE.SCHEMA == "frozen_kl_rescore_v1"
    assert whole["bundle"] == "boxinggym" and whole["round"] is None
    assert whole["cells"][0]["status"] == "reproduced"

    assert set(evaluator["module_sha256"]) == set(MODULE.MODULE_FILES)
    for name, digest in evaluator["module_sha256"].items():
        source = Path(MODULE.__file__).with_name(name)
        assert is_hex64(digest)
        assert digest == hashlib.sha256(source.read_bytes()).hexdigest()

    for name in MODULE.FROZEN_CONSTANT_NAMES:
        declared = evaluator["constants"][name]
        actual = getattr(boxinggym_truth_law, name)
        assert declared == (list(actual) if isinstance(actual, tuple) else actual)
    assert evaluator["constants"]["REPRODUCTION_TOLERANCE"] == MODULE.REPRODUCTION_TOLERANCE
    assert len(evaluator["constants"]) == len(MODULE.FROZEN_CONSTANT_NAMES) + 1

    assert evaluator["functions"] == list(MODULE.EVALUATOR_FUNCTIONS)
    assert evaluator["commit_source"] in ("git_rev_parse_head", "absent")
    if evaluator["commit_source"] == "absent":
        assert evaluator["commit"] is None
    else:
        assert len(evaluator["commit"]) == 40
        assert set(evaluator["commit"]) <= set("0123456789abcdef")


def test_the_table_carries_no_nan_and_no_absolute_path(tmp_path, death_cell):
    """The table contains no NaN or Infinity literal and no absolute path outside the cell."""
    _, _, output = only(tmp_path, "boxinggym", death_cell)
    body = output.read_text()
    # `allow_nan=False` is the writer's own guard; this is what it buys.
    assert "NaN" not in body and "Infinity" not in body
    # nothing in the table names a directory outside the cell it came from.
    assert str(Path(death_cell).parent) not in body
    assert json.loads(body)["cells"][0]["cell_path"] == Path(death_cell).name


def test_a_requested_round_that_is_not_the_final_checkpoint_is_rejected(tmp_path, death_cell):
    """A requested round that is not the cell's final checkpoint is rejected; the final round is accepted."""
    code, record, _ = only(tmp_path, "boxinggym", death_cell, extra=["--round", "1"])
    assert code == 1
    assert record["status"] == "rejected"
    assert record["error_message"].startswith("--round 1 is not this cell's final checkpoint")

    code, record, _ = only(tmp_path, "boxinggym", death_cell, name="two.json", extra=["--round", "2"])
    assert code == 0 and record["status"] == "reproduced"


def test_cell_and_cells_root_are_mutually_exclusive(tmp_path, death_cell):
    """Exactly one of `--cell` and `--cells-root` is required, otherwise the exit code is 2 and no table is written."""
    for argv in (
        ["--bundle", "boxinggym", "--output", str(tmp_path / "a.json")],
        ["--bundle", "boxinggym", "--cell", str(death_cell), "--cells-root",
         str(Path(death_cell).parent), "--output", str(tmp_path / "b.json")],
    ):
        with pytest.raises(SystemExit) as raised:
            MODULE.main(argv)
        assert raised.value.code == 2
    assert not list(tmp_path.glob("*.json"))


def test_a_cell_that_raises_is_recorded_as_failed_and_the_table_survives(
    tmp_path, death_cell, monkeypatch
):
    """An unexpected exception is a row, not a lost table (the module's contract)."""
    def explode(*args, **kwargs):
        raise RuntimeError("synthetic evaluator failure")

    monkeypatch.setattr(MODULE, "rescore_boxinggym", explode)
    code, record, output = only(tmp_path, "boxinggym", death_cell)
    assert code == 1
    assert record["status"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert record["error_message"] == "synthetic evaluator failure"
    assert output.is_file()
    # the reader still recorded what it managed to open before the failure.
    assert isinstance(record["inputs"], list)


def test_each_record_carries_the_cells_own_evaluator_generation(tmp_path, death_cell, core_cell):
    """The rescore's identity is in the header; the RUN's is in the row, so a
    table says which generation produced each stored number it judged."""
    core_count = pytest.importorskip("bayes_model_discovery.experimental.core_count")
    for bundle, cell, seed_key in (
        ("boxinggym", death_cell, "evaluation_seed"),
        ("core", core_cell, "evaluation_input_seed"),
    ):
        _, record, _ = only(tmp_path, bundle, cell, name=f"{bundle}.json")
        config = json.loads((Path(cell) / "config.json").read_text())
        assert record["evaluation_seed"] == config[seed_key]
        assert record["run_source_files_sha256"] == config["source_files_sha256"]
        assert all(is_hex64(digest) for digest in record["run_source_files_sha256"].values())
    assert only(tmp_path, "boxinggym", death_cell, name="bg2.json")[1]["evaluation_seed"] == (
        core_count.seed_for("boxinggym_evaluation", "death", 11, "dev")
    )


def test_evaluation_x_sha256_is_the_stored_arrays_own_bytes(tmp_path, death_cell):
    """`evaluation_x_sha256` hashes the stored evaluation array's bytes, and `q` matches its length."""
    _, record, _ = only(tmp_path, "boxinggym", death_cell)
    with np.load(Path(death_cell) / "evaluation_scoring_only.npz") as saved:
        x = saved["x"]
    assert record["evaluation_x_sha256"] == MODULE.array_sha256(x)
    assert record["q"] - record["q_excluded_count"] == len(x)
