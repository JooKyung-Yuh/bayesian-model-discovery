"""Task mode: CLOSED over the paper manifest's tasks, without recorded histories.

The oracle is the real zero-cost simulator and the client is the tool's fake one; no provider
call is made. No published response or history is a fixture.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

import tools.icl_baseline.icl_baseline as module
from tools.icl_baseline.icl_baseline import (
    CLOSED,
    DEFAULT_TASK,
    FIXED,
    TaskSource,
    Unit,
    paper_tasks,
    seal_draft,
)
from tools.icl_baseline.protocol import (
    BOUNDS,
    DOMAIN_WORDS,
    MASKED_NAMES,
    ROOT,
    VISIBLE_NAMES,
    forbidden_words,
)
from tools.icl_baseline.tests.test_icl import synthetic_history
from tools.icl_baseline.transport import settings

pytestmark = pytest.mark.usefixtures("require_autoscilab_source")

TASKS = ("c0_michaelis_menten__easy__v0", "c2_product_inhibition__easy__v0", "c5_pingpong_bisubstrate__easy__v0",
         "c0_michaelis_menten__hard__v0", "c0_michaelis_menten__medium__v0", "c2_product_inhibition__hard__v0",
         "c2_product_inhibition__medium__v0", "c5_pingpong_bisubstrate__hard__v0",
         "c5_pingpong_bisubstrate__medium__v0")
# The registered CLOSED seal's two D0 prediction request hashes (c5 easy, offsets 0 and 128;
# the same pair for each of the three replicates) and its settings.
SEALED_D0_BODY_SHA256 = {0: "6d86fb000ad1edc7b9a3543c2b6c38722fad000fc9f9bbcb954935a5bcb72895",
                         128: "d2bce8d6c82c4d463341d4e21bec33767812d0e850cd2c4f4aabcb31649c9e8d"}
SEALED_SETTINGS = {
    "cap_usd": 90.0, "max_attempts_per_request": 2, "max_retries": 0, "max_tokens": 32000,
    "mock_selection_stream": "icl_synthetic_selection", "model": "claude-opus-5-5",
    "output_config": {"effort": "medium"},
    "reservation_rule": "utf8_request_bytes_plus_1024_input_tokens_and_max_output_tokens",
    "selection_fallback_stream": "icl_selection_fallback", "stream": True,
    "thinking": {"display": "summarized", "type": "adaptive"}, "timeout_seconds": 300,
    "unit_id": "ICL-MASKED-CLOSED-LOOP",
}
# Unit ICL-K adds the projection rule to the settings, so a seal made before it (the c5 easy CLOSED
# seal of 2026-10-01) no longer matches and every task gets a new seal.
ICLK_SETTINGS = {**SEALED_SETTINGS, "projection_rule": "per_kind_max_cost_x1.25_after_10_calls"}
FIXED3_C_A = (0.03, 0.3, 3.0)


def sources(task):
    return [TaskSource(task, replicate) for replicate in (1, 2, 3)]


def test_the_task_option_takes_exactly_the_nine_paper_tasks():
    assert tuple(paper_tasks()) == TASKS and DEFAULT_TASK in TASKS
    assert module.parser().parse_args(["--unit", CLOSED, "--output", "o", "--dry-run", "--registration", "r"]).task == DEFAULT_TASK
    with pytest.raises(SystemExit) as error:
        module.parser().parse_args(["--unit", CLOSED, "--output", "o", "--dry-run", "--registration", "r",
                                    "--task", "c9_unregistered__easy__v0"])
    assert error.value.code == 2


def test_c5_easy_without_histories_reproduces_the_sealed_d0_requests_and_settings():
    draft = seal_draft(CLOSED, 32000, sources(DEFAULT_TASK), ROOT / "prices.json")
    assert draft["d0_prediction_requests"] == [
        {"replicate": replicate, "offset": offset, "body_sha256": SEALED_D0_BODY_SHA256[offset]}
        for replicate in (1, 2, 3) for offset in (0, 128)]
    assert draft["settings"] == ICLK_SETTINGS == settings(CLOSED, 32000)
    assert draft["task"] == DEFAULT_TASK
    assert draft["history_assignment"] == [
        {"path": f"task:{DEFAULT_TASK}", "group": "task", "replicate": replicate} for replicate in (1, 2, 3)]
    assert str(module.PAPER_MANIFEST.resolve()) in draft["files"]
    # With the sealed manifest at hand (its path comes from the environment), the same comparison
    # against the file itself.
    sealed = os.environ.get("ICL_CLOSED_SEAL", "")
    if sealed:
        seal = json.loads(Path(sealed).read_text())
        assert draft["d0_prediction_requests"] == seal["d0_prediction_requests"]
        # The pre-ICL-K seal: the same settings but for the projection rule it did not record.
        assert seal["settings"] == SEALED_SETTINGS
        assert {key: value for key, value in draft["settings"].items() if key != "projection_rule"} == seal["settings"]


@pytest.mark.parametrize("task", TASKS)
def test_the_start_is_fixed3_observed_by_the_task_oracle(task):
    from bayes_model_discovery.experimental.chembench_core import fixed3_designs

    source = TaskSource(task, 1)
    assert source.names == MASKED_NAMES and source.config["noise"] == .01
    assert source.config["task"]["id"] == task
    designs = fixed3_designs(None)
    assert [design["C_A"] for design in designs] == list(FIXED3_C_A)
    assert [o.obs_id for o in source.observations] == [0, 1, 2]
    assert [o.x for o in source.observations] == [tuple(design[name] for name in VISIBLE_NAMES) for design in designs]
    # The same three designs asked of a fresh session: the same y, float for float (tolerance 0;
    # the oracle's noise is a function of the inputs).
    session = source.session(budget=3)
    again = [session.observe(design)["y"] for design in designs]
    assert [o.y for o in source.observations] == again and all(y > 0 for y in again)
    session.finish()
    reference = session.reference_values(designs)
    # y is the oracle's own held-out observation at the design (tolerance 0), and stays within
    # five noise units (noise 0.01) of the noiseless observation.
    assert again == [float(value) for value in reference["heldout_observation"]]
    np.testing.assert_allclose(again, reference["noiseless_observation"], rtol=0.05, atol=0.0)
    assert [o.x for o in TaskSource(task, 3).observations] == [o.x for o in source.observations]
    assert [o.y for o in TaskSource(task, 3).observations] == again


def test_evaluation_points_and_references_come_from_the_runners_own_calls():
    import adapters.chembench_observation as adapter
    from bayes_model_discovery.experimental import chembench_core as engine

    source = TaskSource(DEFAULT_TASK, 1)
    layer = adapter.sample_designs(256, engine.seed_for("chem_evaluation_inputs", 20260907))
    official = adapter.sample_designs(engine.OFFICIAL_EVALUATION_COUNT, engine.OFFICIAL_EVALUATION_SEED)
    for size, designs in ((256, layer), (1000, official)):
        queries = source.queries(size)
        assert queries.shape == (size, 7)
        np.testing.assert_array_equal(queries, [[row[name] for name in VISIBLE_NAMES] for row in designs])
        assert all(lo <= value <= hi for row in queries for value, (lo, hi) in zip(row, BOUNDS))
        session = source.session(budget=1)
        session.finish()
        truth = session.reference_values(designs)
        references = source.references(size)
        assert set(references) == {"primary_rate", "noiseless_observation"}
        for key in references:
            np.testing.assert_array_equal(references[key], truth[key])
    np.testing.assert_array_equal(TaskSource(TASKS[0], 2).queries(256), source.queries(256))
    assert source.ours(256, 3)["primary_rate"] is None and source.files() == [module.PAPER_MANIFEST]


def request_texts(output):
    """Every request body this run wrote, parsed: the corpus records and the transport requests."""
    texts = []
    for path in sorted([*(output / "corpus").rglob("*.json"), *(output / "transport").glob("*.request.json")]):
        body = json.loads(path.read_text())["body"]
        assert set(body) == {"model", "max_tokens", "system", "messages", "output_config", "thinking"}
        texts.append(body["system"] + "\n" + "\n".join(message["content"] for message in body["messages"]))
    return texts


@pytest.mark.parametrize("task", TASKS)
def test_a_dry_run_of_each_task_sends_318_requests_that_name_no_task_and_no_domain(tmp_path, task, capsys):
    output = tmp_path / "closed"
    assert module.main(["--unit", CLOSED, "--dry-run", "--output", str(output), "--task", task,
                        "--registration", str(ROOT / "prices.json")]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "dry_run_complete", "requests": 318, "paid_api_calls": 0}
    receipt = json.loads((output / "receipt.json").read_text())
    starting = json.loads((output / "starting_receipt.json").read_text())
    draft = json.loads((output / "seal_draft.json").read_text())
    assert receipt["task_id"] == starting["task_id"] == draft["task"] == task
    assert receipt["planned_initial_requests"] == receipt["completed_initial_requests"] == 318
    assert receipt["paid_api_calls"] == 0 and receipt["starting_balance_usd"] == 90.0
    corpus = sorted((output / "corpus").rglob("*.json"))
    assert len(corpus) == 318
    texts = request_texts(output)
    assert len(texts) >= 2 * 318
    domain, difficulty, _ = task.split("__")
    named = re.compile("|".join(map(re.escape, (task, domain, *domain.split("_")[1:], difficulty))), re.IGNORECASE)
    for text in texts:
        assert DOMAIN_WORDS.search(text) is None and named.search(text) is None
        assert forbidden_words(text) == []
    # The start each replicate predicted from is the oracle's fixed3 start of this task.
    start = TaskSource(task, 1).observations
    for replicate in (1, 2, 3):
        record = json.loads((output / f"closed_rep{replicate}.json").read_text())
        assert len(record["observations"]) == 63
        assert [(row["obs_id"], tuple(row["x"]), row["y"]) for row in record["observations"][:3]] == [
            (o.obs_id, o.x, o.y) for o in start]
    scores = json.loads((output / "scores.json").read_text())
    assert {cell["group"] for cell in scores["cells"]} == {"task"}
    assert all(cell["ours"]["primary_rate"] is None and cell["direction"] == "no directional statement"
               for cell in scores["cells"])
    assert all(row["direction"] == "no directional statement" for row in scores["timepoint_directions"])


def refused(args, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "live_client", lambda *a, **k: pytest.fail("client must not be constructed"))
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as error:
        module.main(["--output", str(output), "--registration", str(ROOT / "prices.json"), *args])
    assert error.value.code == 2 and not output.exists()


def test_task_mode_is_closed_without_histories_only(tmp_path, monkeypatch):
    other = TASKS[0]
    refused(["--unit", FIXED, "--dry-run"], tmp_path, monkeypatch)
    refused(["--unit", FIXED, "--dry-run", "--task", other], tmp_path, monkeypatch)
    refused(["--unit", CLOSED, "--dry-run", "--fac-history", "a", "--fac-history", "b", "--fac-history", "c"],
            tmp_path, monkeypatch)
    refused(["--unit", CLOSED, "--dry-run", "--task", other, "--history", "a", "--history", "b", "--history", "c"],
            tmp_path, monkeypatch)
    refused(["--unit", CLOSED, "--dry-run", "--history", "a"], tmp_path, monkeypatch)


def test_a_live_task_run_needs_a_seal_that_names_the_task(tmp_path, monkeypatch):
    draft = seal_draft(CLOSED, 32000, sources(TASKS[0]), ROOT / "prices.json")
    for name, seal in (("no_task.json", {key: value for key, value in draft.items() if key != "task"}),
                       ("other_task.json", {**draft, "task": DEFAULT_TASK}), ("none.json", {})):
        path = tmp_path / name
        path.write_text(json.dumps(seal))
        refused(["--unit", CLOSED, "--allow-paid-api", "--seal-manifest", str(path), "--task", TASKS[0]],
                tmp_path, monkeypatch)
    refused(["--unit", CLOSED, "--allow-paid-api", "--task", TASKS[0]], tmp_path, monkeypatch)


def test_a_history_run_carries_no_task_key(tmp_path, monkeypatch):
    histories = [synthetic_history(tmp_path, "v2", replicate) for replicate in (1, 2, 3)]
    monkeypatch.setattr(module.History, "files", lambda self: [])  # the synthetic history has no input files
    draft = seal_draft(FIXED, 32000, histories, ROOT / "prices.json")
    assert "task" not in draft
    unit = Unit(FIXED, tmp_path / "output", histories, dry_run=True, mock=True, client=module.SyntheticClient())
    receipt = unit.run()
    assert "task_id" not in receipt
    assert "task_id" not in json.loads((tmp_path / "output" / "starting_receipt.json").read_text())
