"""Synthetic protocol fixtures only: no published response/history is a fixture."""

import copy
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from bayes_model_discovery.experimental.count_models import LINK_BY_OUTCOME
from tools.icl_baseline.icl_baseline import CLOSED, FIXED, VISIBLE, History, SyntheticClient, Unit
from tools.icl_baseline.protocol import (
    BOUNDS,
    MASKED_NAMES,
    OUTCOME_POLICIES,
    TIMES,
    VISIBLE_NAMES,
    Observation,
    cell_direction,
    closed_direction,
    fixed_direction,
    forbidden_words,
    json_bytes,
    metrics,
    observations_from_rows,
    prediction_result,
    prompt_parts,
    render,
    score_timepoint,
    selection_fallback,
    validate,
)
from tools.icl_baseline.transport import (
    Budget,
    BudgetStopped,
    Transport,
    live_client,
    load_prices,
    request_body,
    settings,
    sha,
    write_new,
)


@pytest.fixture
def observations():
    return [Observation(0, (2., 4., 6., 8., 2., 300., 6.), 4.),
            Observation(1, (1., 5., 7., 9., 3., 320., 8.), 10.)]


def test_prompt_literal_placeholders_order_mapping_and_determinism(observations):
    prompt, mapping = render("prediction", observations, queries=[observations[0].x] * 12)
    system, template, _ = prompt_parts("prediction")
    assert prompt["system"] == system
    assert prompt["messages"][0]["content"].startswith(template.split("Observed experiments")[0])
    assert "2,1.0,5.0,7.0,9.0,3.0,320.0,8.0,10.0\n1,2.0" in prompt["messages"][0]["content"]
    assert [row["id"] for row in mapping] == [f"q{i}" for i in range(1, 13)]
    assert mapping[10]["coordinates"] == list(observations[0].x)
    assert json_bytes(prompt) == json_bytes(render("prediction", observations, queries=[observations[0].x] * 12)[0])
    assert forbidden_words(json.dumps(prompt)) == []


def test_visible_only_registered_substitutions(observations):
    a, _ = render("prediction", observations, queries=[observations[0].x])
    b, _ = render("prediction", observations, queries=[observations[0].x], visible=True, names=VISIBLE_NAMES)
    text = a["messages"][0]["content"].replace("X1..X7", ", ".join(VISIBLE_NAMES))
    for masked, visible in zip(MASKED_NAMES, VISIBLE_NAMES):
        text = text.replace(masked, visible)
    assert text == b["messages"][0]["content"]
    assert forbidden_words(json.dumps(b), visible=True) == []


def test_interface_order_not_dictionary_and_auxiliary_information_is_discarded():
    row = {"observation_index": 1, "x": dict(reversed(list(zip(MASKED_NAMES, range(7))))),
           "y": 2., "outputs": {"secret": 999}, "thinking": "do not leak"}
    obs = observations_from_rows([row], MASKED_NAMES)
    assert obs[0].x == tuple(range(7))
    prompt, mapping = render("selection", obs)
    assert mapping == []
    assert "secret" not in json.dumps(prompt) and "do not leak" not in json.dumps(prompt)
    assert "New inputs" not in json.dumps(prompt)
    with pytest.raises(ValueError, match="allowlist"):
        render("selection", obs, queries=[[999] * 7])
    with pytest.raises(TypeError):
        render("selection", obs, thinking="forbidden")


def test_outcome_policy_exhaustive_and_unimplemented_fail_closed(observations):
    assert set(OUTCOME_POLICIES) == set(LINK_BY_OUTCOME)
    for outcome in set(OUTCOME_POLICIES) - {"positive"}:
        with pytest.raises(NotImplementedError):
            render("selection", observations, outcome_type=outcome)


@pytest.mark.parametrize("value", ['"3"', "true", "null", "NaN", "Infinity", "-Infinity", "0", "-2"])
def test_invalid_prediction_values(value):
    result = validate('{"predictions":{"q1":' + value + '}}', "prediction", query_ids=("q1",))
    assert not result.valid and not result.values


@pytest.mark.parametrize("text", [
    '{"predictions":{}}', '{"predictions":{"q1":1,"q1":2}}',
    '{"predictions":{"q1":1},"other":2}', '{"predictions":{"q1":1},"predictions":{"q1":2}}',
    '{"predictions":{"q1":1,"extra":2}}', '[1]', '{"predictions":',
])
def test_invalid_prediction_structure(text):
    assert not validate(text, "prediction", query_ids=("q1",)).valid


def test_fences_max_tokens_partial_preservation_and_no_clipping(observations):
    assert validate('```json\n{"predictions":{"q1":1}}\n```', "prediction", query_ids=("q1",)).valid
    assert not validate('{"predictions":{"q1":1}}', "prediction", query_ids=("q1",), stop_reason="max_tokens").values
    first = validate('{"predictions":{"q1":1e100,"q2":null}}', "prediction", query_ids=("q1", "q2", "q3"))
    second = validate('{"predictions":{"q1":2,"q2":3}}', "prediction", query_ids=("q1", "q2", "q3"))
    result = prediction_result([first, second], ("q1", "q2", "q3"), observations)
    assert result["values"] == [1e100, 3., 7.]
    assert result["fallback_ids"] == ["q3"]


@pytest.mark.parametrize("bad", [None, True, "1", float("nan"), -1, 101])
def test_selection_rejects_without_clipping(bad):
    point = {key: float(lo) for key, (lo, _) in zip(MASKED_NAMES, BOUNDS)}
    point["X1"] = bad
    result = validate(json.dumps({"x": point}), "selection")
    assert not result.valid and result.values == {}


def test_selection_bounds_precision_and_fallback():
    point = {key: float(hi) for key, (_, hi) in zip(MASKED_NAMES, BOUNDS)}
    point["X1"] = 1.234567891
    result = validate(json.dumps({"x": point}), "selection")
    assert result.valid and result.values["X1"] == 1.234567891
    a = selection_fallback("synthetic-unit", 2, 7)
    assert a == selection_fallback("synthetic-unit", 2, 7)
    assert a != selection_fallback("synthetic-unit", 2, 8)
    assert all(lo <= value <= hi for value, (lo, hi) in zip(a, BOUNDS))


def test_scores_invalid_denominator_and_separate_targets(observations):
    assert metrics([1., 2.], [1., 1.])["status"] == "undefined"
    assert metrics([], [])["nmsle"] is None
    batches = [{"values": [2.] * 5, "valid": [False, True, True, True, True], "fallback_ids": ["q1"]}]
    refs = {"primary_rate": np.arange(1., 6.), "noiseless_observation": np.arange(2., 7.)}
    score = score_timepoint(batches, refs, observations)
    assert score["status"] == "valid" and score["fallback_query_ratio"] == .2
    batches[0]["fallback_ids"].append("q2")
    assert score_timepoint(batches, refs, observations)["status"] == "invalid_timepoint"
    assert score["targets"]["primary_rate"]["conditional_valid"]["count"] == 4
    assert score["targets"]["primary_rate"]["constant"]["count"] == 5


def test_directional_claims_require_both_targets_and_three_complete_replicates():
    low = {"primary_rate": 1., "noiseless_observation": 2.}
    high = {"primary_rate": 2., "noiseless_observation": 3.}
    assert cell_direction(low, high) == "ICL side"
    assert cell_direction(low, {**high, "noiseless_observation": None}) == "undefined"
    pair = {"status": "valid", "icl": low, "ours": high}
    assert fixed_direction([pair] * 3) == "ICL side"
    assert fixed_direction([pair] * 2) == "no directional statement"
    assert fixed_direction([pair, pair, {**pair, "status": "invalid_timepoint"}]) == "no directional statement"
    assert closed_direction([{"status": "valid", **low}] * 3, [{"status": "valid", **high}] * 3) == "ICL side"


def test_budget_reservation_unknown_usage_and_registered_projection(observations):
    body = request_body(render("selection", observations)[0])
    budget = Budget(45, load_prices())
    reserve = budget.reserve(body, 318, "selection")
    assert reserve["reservation_usd"] >= .64
    row = budget.record({}, reserve, status="transport_failure", kind="selection")
    assert row["cost_usd"] == reserve["reservation_usd"]
    budget.cap = budget.spent + reserve["reservation_usd"] / 2
    with pytest.raises(BudgetStopped, match="reservation"):
        budget.reserve(body, 1, "selection")
    budget = Budget(45, load_prices())
    for _ in range(10):
        reserve = budget.reserve(body, 318, "selection")
        budget.record({"input_tokens": 100, "output_tokens": 10000}, reserve, kind="selection")
    with pytest.raises(BudgetStopped, match="projection"):
        budget.reserve(body, 318, "selection")
    budget = Budget(45, load_prices())
    for _ in range(10):
        reserve = budget.reserve(body, 3, "selection")
        budget.record({"input_tokens": 1, "output_tokens": 1}, reserve, kind="selection")
    assert budget.reserve(body, 3, "selection")["c_hi_kind_usd"] < reserve["reservation_usd"]


@pytest.mark.parametrize("tokens", [16000, 32000])
def test_real_sdk_mock_wire_and_thinking_record(tmp_path, observations, tokens):
    from anthropic import Anthropic
    seen = []
    response = {"id": "synthetic-message", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                "content": [{"type": "thinking", "thinking": "synthetic summary", "signature": "sig"},
                            {"type": "text", "text": '{"predictions":{"q1":3}}'}],
                "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 20, "output_tokens": 30}}

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if not body.get("stream"):
            return httpx.Response(200, json=response)
        events = [{"type": "message_start", "message": {**response, "content": [], "stop_reason": None}}]
        for i, block in enumerate(response["content"]):
            events.extend([{"type": "content_block_start", "index": i, "content_block": block},
                           {"type": "content_block_stop", "index": i}])
        events += [{"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 30}}, {"type": "message_stop"}]
        data = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})

    with Anthropic(api_key="synthetic-key", max_retries=0, timeout=300,
                   http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        transport = Transport(client, tmp_path, Budget(45, load_prices()), max_tokens=tokens, synthetic=True)
        prompt, mapping = render("prediction", observations, queries=[observations[0].x])
        result = transport.request(prompt, "synthetic", "prediction", observations, mapping=mapping)
    assert result["values"] == [3.]
    body = seen[0]
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == tokens
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "medium"}
    assert "temperature" not in body
    record = json.loads((tmp_path / "transport/synthetic.attempt0.response.json").read_text())
    assert record["thinking_text"] == ["synthetic summary"]
    assert record["accounting"]["cost_basis"] == "usage"


def test_timeout_consumes_attempt_and_two_attempts_fallback(tmp_path, observations):
    class TimeoutClient:
        def __init__(self):
            self.messages = self
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise TimeoutError("synthetic transport timeout")

    client = TimeoutClient()
    transport = Transport(client, tmp_path, Budget(45, load_prices()), max_tokens=16000, synthetic=True)
    prompt, mapping = render("prediction", observations, queries=[observations[0].x])
    result = transport.request(prompt, "timeout", "prediction", observations, mapping=mapping)
    assert client.calls == 2 and result["values"] == [7.]
    assert result["attempt_statuses"] == ["transport_failure"] * 2
    assert all(row["cost_basis"] == "maximum_reservation_unknown_usage" for row in transport.budget.calls)


def test_repair_keeps_first_valid_value_and_never_sends_thinking(tmp_path, observations):
    class Client(SyntheticClient):
        def __init__(self):
            super().__init__()
            self.bodies = []

        def create(self, **body):
            self.bodies.append(copy.deepcopy(body))
            result = super().create(**body)
            values = {"q1": 5} if len(self.bodies) == 1 else {"q1": 9, "q2": 6}
            result.content[-1].text = json.dumps({"predictions": values})
            result.content[0].thinking = "PRIVATE_SYNTHETIC_THINKING"
            return result

    client = Client()
    transport = Transport(client, tmp_path, Budget(45, load_prices()), max_tokens=16000, synthetic=True)
    prompt, mapping = render("prediction", observations, queries=[observations[0].x] * 2)
    result = transport.request(prompt, "repair", "prediction", observations, mapping=mapping)
    assert result["values"] == [5., 6.]
    assert len(client.bodies[1]["messages"]) == 3
    assert "PRIVATE_SYNTHETIC_THINKING" not in json.dumps(client.bodies[1])


def test_paid_guard_before_client_for_missing_draft_and_changed_file(tmp_path):
    called = []
    factory = lambda **kwargs: called.append(kwargs)
    with pytest.raises(ValueError, match="locked"):
        live_client(False, None, settings(FIXED), [], factory)
    draft = tmp_path / "draft.json"
    write_new(draft, {"status": "draft_not_execution_authority"})
    with pytest.raises(ValueError, match="sealed"):
        live_client(True, draft, settings(FIXED), [], factory)
    from tools.icl_baseline.protocol import ROOT
    files = {str(p.resolve()): sha(p) for p in ROOT.rglob("*") if p.is_file()
             and p.suffix in (".py", ".txt", ".json") and "tests" not in p.parts}
    seal = tmp_path / "seal.json"
    write_new(seal, {"status": "sealed", "settings": settings(FIXED), "files": files})
    live_client(True, seal, settings(FIXED), [], factory)
    assert called == [{"max_retries": 0, "timeout": 300.0}]
    called.clear()
    data = json.loads(seal.read_text())
    data["files"][next(iter(files))] = "0" * 64
    bad = tmp_path / "bad.json"
    write_new(bad, data)
    with pytest.raises(ValueError, match="mismatch"):
        live_client(True, bad, settings(FIXED), [], factory)
    assert called == []


def synthetic_history(tmp_path, group, replicate, names=MASKED_NAMES):
    root = tmp_path / f"{group}_{replicate}"
    root.mkdir()
    x = tuple((lo + hi) / 2 for lo, hi in BOUNDS)
    observations = [Observation(i, x, float(i + 1)) for i in range(63)]
    history = History(root, group, replicate, {}, names, observations)
    for size in (256, 1000):
        prefix = "" if size == 256 else "official_"
        np.savez(root / f"{prefix}evaluation_inputs.npz", x=np.tile(x, (size, 1)), names=names)
        np.savez(root / f"{prefix}reference_after_finish.npz", primary_rate=np.linspace(.1, 4, size),
                 noiseless_observation=np.linspace(.2, 5, size))
    (root / "predictions").mkdir()
    for n in TIMES:
        np.savez(root / "predictions" / f"t{n - 3:02d}.npz", mean=np.ones(256))
    write_new(root / "official_scores_after_finish.json", {"rounds": [
        {"observation_count": n, "nmsle": .4} for n in (10, 25, 63)]})
    return history


@pytest.mark.parametrize("unit,groups,expected", [(FIXED, ("v2",), 138), (FIXED, ("v2", "fac"), 270),
                                                (VISIBLE, ("v2",), 96)])
def test_request_counts_and_n3_sharing(tmp_path, unit, groups, expected):
    histories = [synthetic_history(tmp_path, group, replicate) for group in groups for replicate in (1, 2, 3)]
    unit = Unit(unit, tmp_path / "output", histories, dry_run=True, mock=True, client=SyntheticClient())
    receipt = unit.run()
    assert len(unit.requests) == receipt["completed_initial_requests"] == expected
    assert receipt["paid_api_calls"] == 0 and receipt["call_count"] == 0


def test_closed_continues_after_seven_fallbacks_and_finishes_before_scoring(tmp_path, monkeypatch):
    import adapters.chembench_observation as adapter
    import bayes_model_discovery.experimental.chembench_core as engine
    histories = [synthetic_history(tmp_path, "v2", replicate) for replicate in (1, 2, 3)]
    for history in histories:
        history.config = {"task": {"domain": "synthetic", "difficulty": "synthetic", "law_version": "synthetic"},
                          "noise": .01, "d0_conditions": {}}
    sessions = []

    class Session:
        def __init__(self, *args, **kwargs):
            self.calls, self.finished = 0, False
            sessions.append(self)

        def schema(self):
            return {"input_names": VISIBLE_NAMES}

        def observe(self, x):
            self.calls += 1
            return {"observation_index": self.calls, "x": x, "y": float(self.calls)}

        def finish(self):
            self.finished = True

    class BadClient(SyntheticClient):
        def create(self, **body):
            result = super().create(**body)
            if "New inputs (" not in body["messages"][0]["content"]:
                result.content[-1].text = "{}"
            return result

    monkeypatch.setattr(adapter, "ChemBenchSession", Session)
    monkeypatch.setattr(engine, "fixed3_designs", lambda _: [dict(zip(VISIBLE_NAMES, o.x)) for o in histories[0].observations[:3]])
    original = History.references

    def references(history, size):
        assert all(session.finished for session in sessions)
        return original(history, size)

    monkeypatch.setattr(History, "references", references)
    unit = Unit(CLOSED, tmp_path / "closed", histories, mock=True, client=BadClient())
    receipt = unit.run()
    assert receipt["completed_initial_requests"] == 318
    assert all(session.calls == 63 for session in sessions)
    assert all(row["direction"] == "no directional statement" for row in json.loads(
        (unit.output / "scores.json").read_text())["timepoint_directions"])
    for replicate in (1, 2, 3):
        data = json.loads((unit.output / f"closed_rep{replicate}.json").read_text())
        assert data["status"] == "run_design_failure" and len(data["selections"]) == 60


def test_budget_stopped_has_no_future_scores(tmp_path):
    histories = [synthetic_history(tmp_path, group, r) for group in ("v2", "fac") for r in (1, 2, 3)]
    unit = Unit(FIXED, tmp_path / "stopped", histories, mock=True, client=SyntheticClient())
    unit.budget.cap = .01
    receipt = unit.run()
    assert receipt["status"] == "budget_stopped" and receipt["call_count"] == 0
    assert json.loads((unit.output / "scores.json").read_text())["cells"] == []


@pytest.mark.parametrize("output_tokens,expected_status,expected_requests", [
    (15000, "complete", 48), (24000, "budget_stopped", 10),
])
def test_visible_projection_reduces_before_stopping_without_discarding_calls(
        tmp_path, output_tokens, expected_status, expected_requests):
    class CostlyClient(SyntheticClient):
        def create(self, **body):
            response = super().create(**body)
            response.usage = SimpleNamespace(model_dump=lambda **_: {"input_tokens": 100, "output_tokens": output_tokens})
            return response

    histories = [synthetic_history(tmp_path, "v2", replicate) for replicate in (1, 2, 3)]
    unit = Unit(VISIBLE, tmp_path / "visible", histories, mock=True, client=CostlyClient())
    receipt = unit.run()
    assert receipt["reduced_visible_plan"]
    assert receipt["status"] == expected_status
    assert receipt["completed_initial_requests"] == expected_requests
    assert all(request["n"] in (3, 10, 25, 63) for request in unit.requests)
    assert receipt["spent_usd"] <= 25


def test_missing_usage_success_uses_reservation(tmp_path, observations):
    class MissingUsage(SyntheticClient):
        def create(self, **body):
            response = super().create(**body)
            response.usage = None
            return response

    transport = Transport(MissingUsage(), tmp_path, Budget(45, load_prices()), synthetic=True)
    prompt, mapping = render("prediction", observations, queries=[observations[0].x])
    result = transport.request(prompt, "unknown-usage", "prediction", observations, mapping=mapping)
    assert result["status"] == "valid"
    row = transport.budget.calls[0]
    assert row["cost_usd"] == row["reservation_usd"] and row["input_tokens"] is None


def test_huge_integer_is_invalid_not_parser_crash():
    result = validate('{"predictions":{"q1":' + '9' * 500 + '}}', "prediction", query_ids=("q1",))
    assert not result.valid


def test_frozen_prompt_asset_hashes():
    from tools.icl_baseline.protocol import ROOT
    expected = {
        "prediction.txt": "96d1095efe69b2be35f18603c4c369783041e753b879915febdc3d429f3b1a4b",
        "selection.txt": "9f5e15708ef49024b736f8e5ff9fb95cdaf2dae0a7a0f44aca885cb6d6b4de3f",
        "prediction_visible.txt": "99dd853ac52813813660363ef28526d53f90a8b015c7fb0ebcb5ef193be9771b",
    }
    assert {name: sha(ROOT / "prompts" / name) for name in expected} == expected


def test_cli_missing_seal_does_not_create_output_or_live_client(tmp_path, monkeypatch):
    import tools.icl_baseline.icl_baseline as module
    from tools.icl_baseline.protocol import ROOT

    histories = [synthetic_history(tmp_path, group, replicate) for group in ("v2", "fac") for replicate in (1, 2, 3)]
    monkeypatch.setattr(History, "load", lambda path, group, replicate: next(h for h in histories if h.root == path))
    monkeypatch.setattr(module, "dependencies", list)
    monkeypatch.setattr(History, "files", lambda self: [])
    monkeypatch.setattr(module, "live_client", lambda *args: pytest.fail("client must not be constructed"))
    output = tmp_path / "not-created"
    args = ["--unit", FIXED, "--output", str(output), "--registration", str(ROOT / "prices.json"), "--allow-paid-api"]
    for history in histories:
        args += ["--history" if history.group == "v2" else "--fac-history", str(history.root)]
    with pytest.raises(SystemExit) as error:
        module.main(args)
    assert error.value.code == 2 and not output.exists()
