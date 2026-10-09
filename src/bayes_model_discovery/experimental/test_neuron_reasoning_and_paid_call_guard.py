"""NeuronBench half of the reasoning-transport work: Anthropic reasoning settings and the per-cell paid-call guard.

Synthetic fixtures only (the proposer tests' packet and fake clients, the runner tests' synthetic
world); no provider call, no native fit. The wire test uses the installed `anthropic` SDK over
`httpx.MockTransport` with a synthetic key.
"""

import hashlib
import json
from types import SimpleNamespace as NS

import anthropic
import httpx
import pytest

from . import count_proposer as cp
from . import neuron_core as nc
from . import neuron_proposer as p
from .test_neuron_core_cli import (
    _transport_reply,
    _TransportClient,
    transport_world,  # noqa: F401 - the runner tests' synthetic-world fixture
)
from .test_neuron_proposer import (
    context,
    probe,  # noqa: F401 - the proposer tests' finite synthetic forward
    proposer,
    reply,
    response,
    text_packet,
)

SPEC = "effort=medium,thinking=adaptive,display=summarized"
#: A well-formed reply whose mechanism is not in the declared vocabulary.
OUTSIDE = {"mechanism": "synthetic_outside_vocabulary", "kind": "extra_channel", "parameters": {}}
#: `round_01.request.json` and the response record (without its elapsed seconds, keys sorted) of
#: one default-path proposal with these fixtures, hashed on the tree before this change.
PRE_UNIT_DEFAULT = {
    "anthropic": ("008c2f40d9dc08ffe278f3f868b89db9676190f861d57c1d4a2dcc7a18462bc0",
                  "c5112d32a3df09fe1368b3b5917e466388c7c224267d357e33e5259cb6eaa49c"),
    "openai": ("35dc3fd6c56f7f3e76d1ee3397a8401ffc110e28246b67da4e0018ce63c61c8b",
               "68f2dbcb3faa68ff19f16fdd074c1e13580d9d49b68f48d972c2eb1e81f1631d"),
}


def thinking_response(body=None, *, summaries=("Synthetic summary of reasoning.",), redacted=0, stop=None):
    """A fake Anthropic reply whose content starts with thinking blocks."""
    plain = response("anthropic", body, stop=stop)
    blocks = [NS(type="thinking", thinking=text) for text in summaries]
    blocks += [NS(type="redacted_thinking", data="synthetic") for _ in range(redacted)]
    plain.content = [*blocks, *plain.content]
    return plain


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_without_the_options_the_request_file_and_the_record_keep_their_bytes(tmp_path, provider, probe):  # noqa: F811
    """Without reasoning or cap options, the request file and the record keep their pre-change bytes."""
    instance, client = proposer(tmp_path, provider, [response(provider)])
    candidate, _ = instance.propose_one(text_packet(), context=context())
    assert candidate is not None and instance.max_paid_calls is None and instance.reasoning is None
    assert list(client.calls[0]) == (["model", "max_tokens", "system", "messages"] if provider == "anthropic"
                                     else ["model", "max_tokens", "messages"])
    request = (tmp_path / "records" / "round_01.request.json").read_bytes()
    record = json.loads((tmp_path / "records" / "round_01.response.json").read_text())
    record.pop("elapsed_seconds")
    assert (hashlib.sha256(request).hexdigest(),
            hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()) == PRE_UNIT_DEFAULT[provider]
    assert "thinking_blocks" not in record["responses"][0] and "max_paid_calls" not in record


def test_a_stated_spec_reaches_the_anthropic_body_and_the_record(tmp_path, probe):  # noqa: F811
    """A stated reasoning spec reaches the Anthropic request body and the response record."""
    instance, client = proposer(tmp_path, "anthropic", [thinking_response(redacted=1)],
                                reasoning=SPEC, max_tokens=16000)
    candidate, record = instance.propose_one(text_packet(), context=context())
    body = client.calls[0]
    assert candidate is not None
    assert list(body) == ["model", "max_tokens", "system", "messages", "output_config", "thinking"]
    assert body["max_tokens"] == 16000 and body["output_config"] == {"effort": "medium"}
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "extra_body" not in body and body["system"] == p.PROMPT_FILE.read_text()
    assert record["decoding"] == {
        "temperature": "not_sent/provider_default", "max_tokens": 16000,
        "reasoning": {"effort": "medium", "thinking": "adaptive", "display": "summarized"},
        "token_parameter": "max_tokens", "output_config": {"effort": "medium"},
        "thinking": {"type": "adaptive", "display": "summarized"}}
    assert record["decoding"] == cp.decoding_identity(16000, cp.parse_reasoning_spec(SPEC), provider="anthropic")
    row = record["responses"][0]
    assert row["thinking_blocks"] == 2 and row["redacted_thinking_blocks"] == 1
    assert row["thinking_text"] == ["Synthetic summary of reasoning."]
    assert row["raw_text"] == json.dumps(reply())  # the text block alone is the reply
    written = json.loads((tmp_path / "records" / "round_01.request.json").read_text())
    assert written["bodies"] == [body]
    # Effort alone: no thinking key.
    (tmp_path / "effort_only").mkdir()
    alone, client = proposer(tmp_path / "effort_only", "anthropic", [response("anthropic")], reasoning="effort=high")
    alone.propose_one(text_packet(), context=context())
    assert client.calls[0]["output_config"] == {"effort": "high"} and "thinking" not in client.calls[0]


def test_the_repair_turn_echoes_the_text_and_never_the_thinking(tmp_path, probe):  # noqa: F811
    """The repair turn echoes the reply text and never the thinking blocks."""
    outside = OUTSIDE
    instance, client = proposer(tmp_path, "anthropic", [thinking_response(outside), thinking_response()],
                                reasoning=SPEC)
    candidate, record = instance.propose_one(text_packet(), context=context())
    # The mechanism vocabulary still refuses under a reasoning spec, and still repairs once.
    assert candidate is not None and record["calls"] == 2 and record["repairs"] == 1
    assert record["rejected_candidates"][0]["reason"] == p.MECHANISM_VOCABULARY_REFUSAL_REASON
    assert record["rejected_candidates"][0]["stage"] == "vocabulary"
    second = client.calls[1]
    assert [turn["role"] for turn in second["messages"]] == ["user", "assistant", "user"]
    assert second["messages"][1]["content"] == json.dumps(outside)
    assert "Synthetic summary" not in json.dumps(second["messages"])
    assert second["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert [row["thinking_blocks"] for row in record["responses"]] == [1, 1]


@pytest.mark.parametrize("spec,provider,message", [
    ("max_tokens=2000", "anthropic", "provider anthropic takes"),
    ("effort=medium,exclude", "anthropic", "provider anthropic takes"),
    ("effort=none", "anthropic", "not Anthropic effort levels"),
    ("effort=minimal,thinking=adaptive", "anthropic", "not Anthropic effort levels"),
    (SPEC, "openai", "compatibility options require provider openai_compat"),
    ("effort=medium", "openai", "compatibility options require provider openai_compat"),
])
def test_a_spec_the_provider_cannot_send_is_refused_before_the_records_folder(tmp_path, spec, provider, message):
    """A spec the provider cannot send is refused before the records folder is created."""
    with pytest.raises(ValueError, match=message):
        proposer(tmp_path, provider, [], reasoning=spec)
    assert not (tmp_path / "records").exists()


class Wire:
    """What the SDK put on the wire, captured by the mock transport."""

    def __init__(self):
        self.requests = []

    def handler(self, request):
        """Mock-transport handler: record the request JSON and return a synthetic Anthropic reply."""
        self.requests.append(json.loads(request.content))
        return httpx.Response(200, headers={"request-id": "synthetic-request-id"}, json={
            "id": "msg_synthetic", "type": "message", "role": "assistant", "model": "synthetic-returned-model",
            "content": [{"type": "thinking", "thinking": "Synthetic summary of reasoning.", "signature": "synthetic"},
                        {"type": "text", "text": json.dumps(reply())}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 10}})


def test_the_reasoning_body_reaches_the_wire_through_the_installed_sdk(tmp_path, probe):  # noqa: F811
    """The reasoning body reaches the wire unchanged through the installed SDK."""
    wire = Wire()
    client = anthropic.Anthropic(api_key="sk-ant-synthetic-not-a-key", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(wire.handler)))
    instance = p.NeuronProposer(tmp_path / "records", world="synthetic_world", cell="100", provider="anthropic",
                                client=client, model="synthetic-model", reasoning=SPEC, max_tokens=16000)
    candidate, record = instance.propose_one(text_packet(), context=context())
    assert candidate is not None and record["status"] == "parsed" and len(wire.requests) == 1
    sent = wire.requests[0]
    assert sent["output_config"] == {"effort": "medium"}
    assert sent["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert sent["max_tokens"] == 16000 and sent["model"] == "synthetic-model" and "temperature" not in sent
    recorded = json.loads((tmp_path / "records" / "round_01.request.json").read_text())["bodies"][0]
    assert sent == recorded  # the wire JSON is the recorded body, key for key
    assert record["responses"][0]["thinking_text"] == ["Synthetic summary of reasoning."]
    assert record["responses"][0]["thinking_blocks"] == 1
    assert "sk-ant-synthetic" not in json.dumps(record)


def test_the_guard_refuses_the_call_after_the_cap_before_the_client(tmp_path, probe):  # noqa: F811
    """The guard refuses the call after the cap is reached, before the client is used."""
    instance, client = proposer(tmp_path, "anthropic", [response("anthropic")] * 3, max_paid_calls=2, max_rounds=5)
    packet = json.loads(text_packet())
    for round_number in (1, 2):
        # A fresh mechanism each round, so every reply is a valid new candidate.
        instance.client.responses = iter([response("anthropic", reply(
            mechanism=sorted(packet["constraints"]["allowed_mechanisms"])[round_number]))])
        instance.propose_one(text_packet(), context=context(round_number))
    assert instance.attempts == 2 and len(client.calls) == 2
    with pytest.raises(cp.PaidCallBudgetExhausted) as stop:
        instance.propose_one(text_packet(), context=context(3))
    assert (stop.value.attempts, stop.value.max_paid_calls) == (2, 2)
    assert instance.attempts == 2 and len(client.calls) == 2  # the third request never left
    assert not list((tmp_path / "records").glob("round_03.*"))


def test_a_refused_repair_turn_keeps_the_paid_calls_record(tmp_path, probe):  # noqa: F811
    """A refused repair turn still keeps the record of the paid calls already made."""
    instance, client = proposer(tmp_path, "anthropic", [response("anthropic", OUTSIDE)], max_paid_calls=1)
    with pytest.raises(cp.PaidCallBudgetExhausted):
        instance.propose_one(text_packet(), context=context())
    assert len(client.calls) == 1 and instance.attempts == 1
    record = json.loads((tmp_path / "records" / "round_01.response.json").read_text())
    assert record["status"] == cp.BUDGET_EXHAUSTED and record["max_paid_calls"] == 1
    assert record["calls"] == 1 and record["repairs"] == 0 and len(record["responses"]) == 1
    assert len(json.loads((tmp_path / "records" / "round_01.request.json").read_text())["bodies"]) == 1


@pytest.mark.parametrize("bad", [0, -1, True, "two", 1.5])
def test_an_invalid_cap_is_refused(tmp_path, bad):
    """An invalid paid-call cap is refused before any records folder exists."""
    with pytest.raises(ValueError, match="max-paid-calls"):
        proposer(tmp_path, "anthropic", [], max_paid_calls=bad)
    assert not (tmp_path / "records").exists()


def runner_args(out, *extra):
    """Parse runner command-line arguments for a paid Anthropic cell, plus any extra options."""
    return nc.build_parser().parse_args([
        "--output", str(out), "--cell", "101", "--proposer", "anthropic", "--allow-paid-api",
        "--llm-model", "synthetic_requested_model", "--rounds", "2", *extra])


def test_the_runner_option_is_absent_unless_typed_and_refuses_a_bad_value(tmp_path):
    """The runner option is absent unless typed, and a bad value exits with code 2."""
    assert not hasattr(runner_args(tmp_path / "o"), "max_paid_calls")
    assert runner_args(tmp_path / "o", "--max-paid-calls", "7").max_paid_calls == 7
    with pytest.raises(SystemExit) as error:
        runner_args(tmp_path / "o", "--max-paid-calls", "0")
    assert error.value.code == 2


def test_a_cell_capped_at_two_calls_stops_before_the_third_and_exits_four(
        tmp_path, transport_world, monkeypatch, capsys):  # noqa: F811
    # Round 1 is one valid call; round 2's first reply repeats it (a duplicate) and its repair
    # turn would be the third paid request.
    """A cell capped at two calls stops before the third call and exits with code 4."""
    out = tmp_path / "capped"
    client = _TransportClient("anthropic", [_transport_reply()] * 3)
    args = runner_args(out, "--max-paid-calls", "2", "--llm-reasoning", SPEC, "--llm-max-tokens", "16000")
    real_run = nc.run
    monkeypatch.setattr(nc, "run", lambda parsed: real_run(parsed, client=client))
    monkeypatch.setattr("sys.argv", ["neuron_core", "--output", str(out), "--cell", "101", "--proposer", "anthropic",
                                     "--allow-paid-api", "--llm-model", "synthetic_requested_model", "--rounds", "2",
                                     "--max-paid-calls", "2", "--llm-reasoning", SPEC, "--llm-max-tokens", "16000"])
    assert vars(nc.build_parser().parse_args()) == vars(args)
    with pytest.raises(SystemExit) as stopped:
        nc.main()
    assert stopped.value.code == cp.PAID_CALL_BUDGET_EXIT_CODE == 4
    assert "summary.json records stop_reason budget_exhausted" in capsys.readouterr().err
    assert len(client.calls) == 2
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == summary["stop_reason"] == "budget_exhausted"
    assert (summary["paid_api_calls"], summary["max_paid_calls"]) == (2, 2)
    assert (summary["stopped_round"], summary["last_completed_round"]) == (2, 1)
    assert summary["world"] == "h_sag" and summary["rounds"] == 2 and summary["llm_client_injected"] is True
    assert not (out / "failure.json").exists() and (out / "archive.json").exists()
    records = [json.loads(path.read_text()) for path in sorted((out / "llm_transport").glob("*.response.json"))]
    assert [record["status"] for record in records] == ["parsed", "budget_exhausted"]
    assert [record["calls"] for record in records] == [1, 1]
    config = json.loads((out / "config.json").read_text())
    assert config["max_paid_calls"] == 2
    assert config["llm_reasoning"] == {"effort": "medium", "thinking": "adaptive", "display": "summarized"}
    for call in client.calls:
        assert call["output_config"] == {"effort": "medium"} and call["max_tokens"] == 16000
        assert call["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_the_same_cell_without_a_cap_completes_with_three_calls(tmp_path, transport_world):  # noqa: F811
    """The same cell without a cap completes with three calls and records no cap keys."""
    out = tmp_path / "uncapped"
    client = _TransportClient("anthropic", [_transport_reply()] * 3)
    nc.run(runner_args(out), client=client)
    summary = json.loads((out / "summary.json").read_text())
    config = json.loads((out / "config.json").read_text())
    assert len(client.calls) == 3 and summary["paid_api_calls"] == 3
    assert "stop_reason" not in summary and "max_paid_calls" not in summary
    assert "max_paid_calls" not in config and "llm_reasoning" not in config
    for call in client.calls:
        assert list(call) == ["model", "max_tokens", "system", "messages"]
