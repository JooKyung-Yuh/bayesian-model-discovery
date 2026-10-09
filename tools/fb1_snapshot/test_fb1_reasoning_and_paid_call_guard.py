"""Tests for the ForceBench half of the reasoning-transport change: Anthropic reasoning settings and the per-cell paid-call guard.

Injected scripted clients and synthetic packets only: no key, no network, no paid call. The wire
test uses the installed `anthropic` SDK over `httpx.MockTransport` with a synthetic key.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("jax")  # fb1_proposer imports fb1_core, which needs the ForceBench interpreter

SNAPSHOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SNAPSHOT))

import fb1_proposer as fp
from test_fb1_proposer import _state

SPEC = "effort=medium,thinking=adaptive,display=summarized"
CONTEXT = {"run_id": "t", "world": "gravity", "arm": "111", "policy": "P1", "event": 0, "round": 1,
           "n_designs": 3, "candidate_index": 0, "packet_file": "t03_c0.json"}
#: sha256 of the request file and of the response record (without its elapsed seconds, keys
#: sorted) of one default-path proposal with these fixtures, on the tree before this change (the base commit).
PRE_UNIT_DEFAULT = {
    "mock": ("db1b98598e40bdfce14c460de7673ae246585dd1aefc57912db7fcce94968a20",
             "5d79d50e3e2f76ac834ca244e174bcdaee9ad72edd8471ba72d993192b49d8e0"),
    "scripted": ("8c9cddd118f3ad0a7fea24976fd63c83f8745acf45dd95bad8fb5df94909a713",
                 "730689adc46c41b70411730a9d99eb48e2c5b288d7869f9c2803b16230d1b9e9"),
    "anthropic_named": ("bfed47eaf41554a48c9fc63261df65ea0915fc5d959f236de7235bf202fc9057",
                        "325afc2ebf263b61ef465d130d45ef4c7bab202ce8d375b8c238be0fc0529834"),
}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Remove the reasoning, paid-call, token and provider environment options before each test."""
    for name in ("FB1_LLM_REASONING", "FB1_MAX_PAID_CALLS", "FB1_LLM_MAX_TOKENS", "FB1_LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)


class NamedAnthropic(fp.ScriptedClient):
    """A scripted client that names itself the Anthropic transport; replies may carry thinking."""

    provider = "anthropic"

    def __init__(self, texts, *, thinking=(), redacted=0, **kwargs):
        super().__init__(texts, **kwargs)
        self.thinking, self.redacted = tuple(thinking), redacted

    def _create(self, **body):
        response = super()._create(**body)
        blocks = [SimpleNamespace(type="thinking", thinking=text) for text in self.thinking]
        blocks += [SimpleNamespace(type="redacted_thinking", data="synthetic")] * self.redacted
        response.content = [*blocks, *response.content]
        return response


def packet_and_constraints():
    """Build a serialized synthetic belief packet and the catalogue constraints."""
    constraints = fp.catalog_constraints()
    rows, members, _c, interface = _state()
    return fp.serialize_packet(fp.build_belief_packet(rows, members, [], constraints, interface, 1)), constraints


def propose(proposer, **context):
    """Run one proposal through the proposer with the default context updated by keyword arguments."""
    packet, constraints = packet_and_constraints()
    return proposer.propose_one(packet, context={**CONTEXT, **context}, constraints=constraints)


def record_hashes(records):
    """Return sha256 of the request file and of the response record (without elapsed seconds) in a records folder."""
    request = (records / "event_00_round_01_c0.request.json").read_bytes()
    response = json.loads((records / "event_00_round_01_c0.response.json").read_text())
    response.pop("elapsed_seconds")
    return (hashlib.sha256(request).hexdigest(),
            hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest())


def package():
    """Import the packaged count proposer module, skipping the test when it is unavailable."""
    sys.path.insert(0, str(SNAPSHOT.parents[1] / "src"))
    return pytest.importorskip("bayes_model_discovery.experimental.count_proposer")


@pytest.mark.parametrize("kind", ["mock", "scripted", "anthropic_named"])
def test_without_the_options_the_request_file_and_the_record_keep_their_bytes(tmp_path, kind):
    """Without the new options the request file and the record keep their default-path bytes."""
    client = {"mock": None, "scripted": fp.ScriptedClient([fp.mock_response_text(0)]),
              "anthropic_named": NamedAnthropic([fp.mock_response_text(0)])}[kind]
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client)
    candidates, record = propose(proposer)
    assert len(candidates) == 1 and proposer.reasoning is None and proposer.max_paid_calls is None
    assert record_hashes(tmp_path / "records") == PRE_UNIT_DEFAULT[kind]
    assert "thinking_blocks" not in record["responses"][0] and "max_paid_calls" not in record
    if client is not None:
        assert list(client.bodies[0]) == ["model", "max_tokens", "system", "messages"]


def test_a_mock_transport_never_reads_the_two_environment_options(tmp_path, monkeypatch):
    """A mock transport ignores the reasoning and paid-call environment options."""
    monkeypatch.setenv("FB1_LLM_REASONING", SPEC)
    monkeypatch.setenv("FB1_MAX_PAID_CALLS", "1")
    proposer = fp.Fb1CatalogProposer(tmp_path / "records")
    assert proposer.transport == "mock" and proposer.reasoning is None and proposer.max_paid_calls is None
    propose(proposer)
    assert record_hashes(tmp_path / "records") == PRE_UNIT_DEFAULT["mock"]


@pytest.mark.parametrize("source", ["environment", "argument"])
def test_a_stated_spec_reaches_the_anthropic_body_and_the_record(tmp_path, monkeypatch, source):
    """A stated reasoning spec reaches the Anthropic request body and the response record."""
    client = NamedAnthropic([fp.mock_response_text(0)], thinking=("Synthetic summary of reasoning.",), redacted=1)
    if source == "environment":
        monkeypatch.setenv("FB1_LLM_REASONING", SPEC)
        monkeypatch.setenv("FB1_LLM_MAX_TOKENS", "16000")
        proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client)
    else:
        proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client, reasoning=SPEC, max_tokens=16000)
    candidates, record = propose(proposer)
    body = client.bodies[0]
    assert len(candidates) == 1 and proposer.transport == "anthropic"
    assert list(body) == ["model", "max_tokens", "system", "messages", "output_config", "thinking"]
    assert body["max_tokens"] == 16000 and body["output_config"] == {"effort": "medium"}
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"} and "extra_body" not in body
    assert body["system"] == fp.system_prompt_bytes().decode()
    assert record["decoding"] == package().decoding_identity(
        16000, package().parse_reasoning_spec(SPEC), provider="anthropic")
    row = record["responses"][0]
    assert row["thinking_blocks"] == 2 and row["redacted_thinking_blocks"] == 1
    assert row["thinking_text"] == ["Synthetic summary of reasoning."]
    assert row["raw_text"] == fp.mock_response_text(0)
    saved = json.loads((tmp_path / "records" / "event_00_round_01_c0.request.json").read_text())
    assert saved["bodies"] == [body]


def test_the_repair_turn_echoes_the_text_and_never_the_thinking(tmp_path):
    """The repair turn echoes the reply text and never the thinking blocks."""
    client = NamedAnthropic(["this is not JSON", fp.mock_response_text(0)],
                            thinking=("Synthetic summary of reasoning.",))
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client, reasoning=SPEC)
    candidates, record = propose(proposer)
    assert len(candidates) == 1 and record["calls"] == 2 and record["repairs"] == 1
    second = client.bodies[1]
    assert [turn["role"] for turn in second["messages"]] == ["user", "assistant", "user"]
    assert second["messages"][1]["content"] == "this is not JSON"
    assert "Synthetic summary" not in json.dumps(second["messages"])
    assert second["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert [row["thinking_blocks"] for row in record["responses"]] == [1, 1]


@pytest.mark.parametrize("spec,message", [
    ("max_tokens=2000", "provider anthropic takes"),
    ("effort=medium,exclude", "provider anthropic takes"),
    ("effort=none", "not Anthropic effort levels"),
])
def test_a_spec_anthropic_cannot_send_is_refused_before_the_records_folder(tmp_path, spec, message):
    """A spec the Anthropic transport cannot send is refused before the records folder is created."""
    with pytest.raises(ValueError, match=message):
        fp.Fb1CatalogProposer(tmp_path / "records", client=NamedAnthropic([]), reasoning=spec)
    assert not (tmp_path / "records").exists()


def test_other_transports_keep_their_rules(tmp_path):
    """The other transports keep their own reasoning-option rules."""
    with pytest.raises(ValueError, match="Reasoning control requires the openai_compat transport"):
        fp.Fb1CatalogProposer(tmp_path / "scripted", client=fp.ScriptedClient([]), reasoning=SPEC)
    with pytest.raises(ValueError, match="Reasoning control requires the openai_compat transport"):
        fp.Fb1CatalogProposer(tmp_path / "mock", reasoning="effort=medium")
    with pytest.raises(ValueError, match="Anthropic field"):
        fp.Fb1CatalogProposer(tmp_path / "compat", provider="openai_compat", model="synthetic/requested-model",
                              base_url="https://example.invalid/private/path", key_name="CUSTOM_ROUTER_TOKEN",
                              client=SimpleNamespace(), reasoning="effort=medium,thinking=adaptive")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("spec", ["effort=medium", SPEC, "effort=high,thinking=adaptive", "max_tokens=512",
                                  "effort=none", "effort=low,exclude", "effort=minimal,thinking=adaptive",
                                  "display=summarized,effort=xhigh,thinking=adaptive"])
def test_the_copied_grammar_and_validation_match_the_package(spec):
    """The copied spec grammar and validation agree with the package for accepted specs."""
    cp = package()
    assert fp.parse_reasoning_spec(spec) == cp.parse_reasoning_spec(spec)
    assert list(fp.parse_reasoning_spec(spec)) == list(cp.parse_reasoning_spec(spec))
    parsed = cp.parse_reasoning_spec(spec)
    for transport in ("anthropic", "openai_compat"):
        try:
            expected = cp.validate_provider_reasoning(transport, dict(parsed))
        except ValueError as exc:
            with pytest.raises(ValueError) as copied:
                fp.validate_transport_reasoning(transport, dict(parsed))
            assert str(copied.value) == str(exc)
        else:
            assert fp.validate_transport_reasoning(transport, dict(parsed)) == expected
    assert fp.validate_transport_reasoning("anthropic", None) is None


@pytest.mark.parametrize("spec", ["thinking=adaptive", "effort=medium,display=summarized",
                                  "effort=medium,thinking=extended", "effort=medium,thinking=adaptive,display=full"])
def test_the_copied_grammar_refuses_what_the_package_refuses(spec):
    """The copied grammar raises the same errors as the package for refused specs."""
    cp = package()
    with pytest.raises(ValueError) as expected:
        cp.parse_reasoning_spec(spec)
    with pytest.raises(ValueError) as copied:
        fp.parse_reasoning_spec(spec)
    assert str(copied.value) == str(expected.value)


def test_the_copied_helpers_match_the_package():
    """The copied helper functions and constants agree with the package."""
    cp = package()

    def named(*, model, max_tokens, system, messages, extra_body=None):
        raise AssertionError

    def with_output_config(*, model, max_tokens, system, messages, output_config=None, thinking=None, extra_body=None):
        raise AssertionError

    def open_ended(**body):
        raise AssertionError

    def no_extra(*, model, max_tokens, system, messages):
        raise AssertionError

    plain = {"model": "m", "max_tokens": 5, "system": "s", "messages": []}
    reasoning = {**plain, "output_config": {"effort": "medium"}, "thinking": {"type": "adaptive"}}
    for create in (named, with_output_config, open_ended):
        for body in (plain, reasoning):
            assert fp.sdk_call_kwargs(create, body) == cp.sdk_call_kwargs(create, body)
    assert fp.sdk_call_kwargs(named, reasoning)["extra_body"] == {
        "output_config": {"effort": "medium"}, "thinking": {"type": "adaptive"}}
    assert fp.sdk_call_kwargs(no_extra, plain) == plain
    with pytest.raises(ValueError, match="neither"):
        fp.sdk_call_kwargs(no_extra, reasoning)
    replies = [SimpleNamespace(content=[SimpleNamespace(type="text", text="x")]),
               SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="a"),
                                        SimpleNamespace(type="redacted_thinking", data="b"),
                                        SimpleNamespace(type="text", text="x")]),
               SimpleNamespace(content=None)]
    for reply in replies:
        assert fp.anthropic_thinking_record(reply) == cp.anthropic_thinking_record(reply)
        assert fp._thinking_block_count(reply) == cp._thinking_block_count(reply)
    assert fp.PAID_CALL_BUDGET_EXIT_CODE == cp.PAID_CALL_BUDGET_EXIT_CODE == 4
    assert fp.BUDGET_EXHAUSTED == cp.BUDGET_EXHAUSTED and fp.MAX_PAID_CALLS_ERROR == cp.MAX_PAID_CALLS_ERROR
    for value in ("3", 3, " 12 "):
        assert fp.parse_max_paid_calls(value) == cp.parse_max_paid_calls(value)
    for value in (0, -1, True, "two", 1.5, ""):
        with pytest.raises(ValueError, match="max-paid-calls"):
            fp.parse_max_paid_calls(value)
    assert str(fp.PaidCallBudgetExhausted(2, 2)) == str(cp.PaidCallBudgetExhausted(2, 2))
    # Like the package's, not an Exception; here it is a SystemExit carrying the guard's status,
    # because the runner file is not edited to catch it.
    assert not issubclass(fp.PaidCallBudgetExhausted, Exception)
    assert issubclass(fp.PaidCallBudgetExhausted, SystemExit) and fp.PaidCallBudgetExhausted(2, 2).code == 4


def test_the_reasoning_body_reaches_the_wire_through_the_installed_sdk(tmp_path):
    """The reasoning body reaches the wire unchanged through the installed SDK over a mock transport."""
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, headers={"request-id": "synthetic-request-id"}, json={
            "id": "msg_synthetic", "type": "message", "role": "assistant", "model": "synthetic-returned-model",
            "content": [{"type": "thinking", "thinking": "Synthetic summary of reasoning.", "signature": "synthetic"},
                        {"type": "text", "text": fp.mock_response_text(0)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 10}})

    client = anthropic.Anthropic(api_key="sk-ant-synthetic-not-a-key", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client, provider="anthropic",
                                     model="synthetic-model", reasoning=SPEC, max_tokens=16000)
    candidates, record = propose(proposer)
    assert len(candidates) == 1 and record["status"] == "parsed" and len(sent) == 1
    assert sent[0]["output_config"] == {"effort": "medium"}
    assert sent[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert sent[0]["max_tokens"] == 16000 and sent[0]["model"] == "synthetic-model" and "temperature" not in sent[0]
    recorded = json.loads((tmp_path / "records" / "event_00_round_01_c0.request.json").read_text())["bodies"][0]
    assert sent[0] == recorded  # the wire JSON is the recorded body, key for key
    assert record["responses"][0]["thinking_text"] == ["Synthetic summary of reasoning."]
    assert "sk-ant-synthetic" not in json.dumps(record)


@pytest.mark.parametrize("source", ["environment", "argument"])
def test_the_guard_refuses_the_call_after_the_cap_before_the_client(tmp_path, monkeypatch, capsys, source):
    """The paid-call guard refuses the call after the cap, before the client is used."""
    client = NamedAnthropic([fp.mock_response_text(i) for i in range(3)])
    if source == "environment":
        monkeypatch.setenv("FB1_MAX_PAID_CALLS", "2")
        proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client)
    else:
        proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client, max_paid_calls=2)
    assert proposer.max_paid_calls == 2
    for round_number in (1, 2):
        candidates, _ = propose(proposer, round=round_number, event=round_number - 1)
        assert len(candidates) == 1
    with pytest.raises(fp.PaidCallBudgetExhausted) as stop:
        propose(proposer, round=3, event=2)
    assert (stop.value.attempts, stop.value.max_paid_calls, stop.value.code) == (2, 2, 4)
    assert (stop.value.context["round"], stop.value.context["event"]) == (3, 2)
    assert client.calls == 2 and proposer.attempts == 2  # the third request never left
    assert not list((tmp_path / "records").glob("event_02_round_03*"))
    summary = json.loads((tmp_path / "records" / "stop_summary.json").read_text())
    assert summary["status"] == summary["stop_reason"] == "budget_exhausted"
    assert (summary["paid_api_calls"], summary["max_paid_calls"]) == (2, 2)
    assert (summary["stopped_round"], summary["stopped_event"], summary["stopped_n_designs"]) == (3, 2, 3)
    assert (summary["world"], summary["arm"], summary["policy"]) == ("gravity", "111", "P1")
    assert summary["transport"] == "anthropic" and summary["registered_run"] is False
    assert "stop_summary.json records stop_reason budget_exhausted" in capsys.readouterr().err


def test_a_refused_repair_turn_keeps_the_paid_calls_record(tmp_path):
    """A repair turn refused by the guard keeps the record of the paid calls already made."""
    client = NamedAnthropic(["this is not JSON", fp.mock_response_text(0)])
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client, max_paid_calls=1)
    with pytest.raises(fp.PaidCallBudgetExhausted):
        propose(proposer)
    assert client.calls == 1 and proposer.attempts == 1
    record = json.loads((tmp_path / "records" / "event_00_round_01_c0.response.json").read_text())
    assert record["status"] == "budget_exhausted" and record["max_paid_calls"] == 1
    assert record["calls"] == 1 and record["repairs"] == 0 and len(record["responses"]) == 1
    saved = json.loads((tmp_path / "records" / "event_00_round_01_c0.request.json").read_text())
    assert len(saved["bodies"]) == 1
    summary = json.loads((tmp_path / "records" / "stop_summary.json").read_text())
    assert (summary["paid_api_calls"], summary["max_paid_calls"], summary["stopped_round"]) == (1, 1, 1)


def test_a_run_without_a_cap_writes_no_stop_summary(tmp_path):
    """A run without a paid-call cap writes no stop summary."""
    client = NamedAnthropic([fp.mock_response_text(i) for i in range(3)])
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client)
    for round_number in (1, 2, 3):
        propose(proposer, round=round_number, event=round_number - 1)
    assert client.calls == 3 and not (tmp_path / "records" / "stop_summary.json").exists()


@pytest.mark.parametrize("bad", ["0", "-1", "two", "1.5"])
def test_an_invalid_cap_is_refused_before_the_records_folder(tmp_path, monkeypatch, bad):
    """An invalid paid-call cap is refused before the records folder is created."""
    monkeypatch.setenv("FB1_MAX_PAID_CALLS", bad)
    with pytest.raises(ValueError, match="max-paid-calls"):
        fp.Fb1CatalogProposer(tmp_path / "records", client=NamedAnthropic([]))
    assert not (tmp_path / "records").exists()


#: run_gravity.py's sha256: the `code_sha` column of the published ForceBench contract rows.
#: The ForceBench name-masking change edited the runner once (prompt variant, input-name masking, the Anthropic reasoning
#: label declaration); before it the value was 28e954222cbc3404...
#: The ForceBench gradient-ascent change (2026-10-02) edited the runner again (FB1_DESIGN_SEARCH=random64|ascent, default path
#: byte-identical); before it the value was d5bb0cbc0575adb1...
#: The softening change (2026-10-02) edited the runner once more (FB1_SOFTENING, default byte-identical);
#: before it the value was 597172ca16d53e49...
#: The EPIG-targets change (2026-10-03) edited it again (FB1_EPIG_TARGETS=uniform|score_q, default byte-identical);
#: before it the value was 7936c8861400a2c6...
#: The trigonometric-grammar change (2026-10-03) edited it again (FB1_GRAMMAR_TRIG=0|1, default byte-identical);
#: before it the value was d92882ff750b38a3...
#: The ForceBench change for the unified protocol (2026-10-06) edited it again (unified-protocol options, default byte-identical);
#: before it the value was 7e0bfbabdabc6e72...
#: The second commit of that change (2026-10-06) changed unified-protocol paths only (FB1_CATALOGUE=off packets without catalogue
#: traces, the revival wall_seconds span, arm_config proposal_rounds in rounds; default byte-identical); before it
#: the value was 272c905442433de8...
RUNNER_SHA256 = "7025bb0ac102f9c56c5ec4db99dcf32ec722595c072e89842eb780ed2647ef4c"
STOPPED_CELL = """
import sys
sys.path.insert(0, {snapshot!r})
import fb1_proposer as fp
from test_fb1_reasoning_and_paid_call_guard import NamedAnthropic, propose
client = NamedAnthropic([fp.mock_response_text(i) for i in range(3)])
proposer = fp.Fb1CatalogProposer({records!r}, client=client)
for round_number in (1, 2, 3):
    propose(proposer, round=round_number, event=round_number - 1)
    print("completed round", round_number, flush=True)
"""


def test_the_runner_file_is_pinned_and_an_uncaught_stop_ends_the_process_with_status_four(tmp_path):
    """The runner file hash is pinned and an uncaught stop ends the process with exit status 4."""
    runner = hashlib.sha256((SNAPSHOT / "run_gravity.py").read_bytes()).hexdigest()
    rows = dict(reversed(row.split("  ", 1)) for row in (SNAPSHOT / "SHA256SUMS").read_text().splitlines())
    assert runner == RUNNER_SHA256 == rows["./run_gravity.py"]
    assert rows["./fb1_proposer.py"] == hashlib.sha256((SNAPSHOT / "fb1_proposer.py").read_bytes()).hexdigest()
    # A fresh interpreter, as the runner is: nothing catches the stop, and the status is the guard's.
    records = tmp_path / "records"
    environment = {key: value for key, value in os.environ.items() if not key.startswith("FB1_")}
    done = subprocess.run(
        [sys.executable, "-B", "-c", STOPPED_CELL.format(snapshot=str(SNAPSHOT), records=str(records))],
        env={**environment, "FB1_MAX_PAID_CALLS": "2", "JAX_PLATFORMS": "cpu"},
        capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 4, done.stderr[-2000:]
    assert done.stdout.splitlines() == ["completed round 1", "completed round 2"]
    assert "STOP paid-call budget exhausted: 2 of 2 paid calls made" in done.stderr
    assert "Traceback" not in done.stderr
    summary = json.loads((records / "stop_summary.json").read_text())
    assert summary["stop_reason"] == "budget_exhausted" and summary["stopped_round"] == 3
    assert len(list(records.glob("*.response.json"))) == 2
