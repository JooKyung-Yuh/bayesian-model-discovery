#!/usr/bin/env python3
"""Wire tests of the Claude path: the RELEASED LLMClient -> AnthropicChatShim -> the official anthropic
SDK -> httpx.MockTransport. The assertions read the JSON body and headers the SDK actually put on the
wire (a fake client that accepts any keyword proves nothing). 0 USD: no network.

Run under each SDK pair (both must pass):
  the development host engine venv  anthropic 0.120.2 / httpx 0.27.0
  a compute host SDK pair    anthropic 0.71.0  / httpx 0.28.1   (<development path>)
  env -u FORCE_COLOR PYTHONDONTWRITEBYTECODE=1 <python> tests/test_claude_wire.py
"""
import json
import os
import sys
import tempfile

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(KIT, "tests", "stub"), KIT]
from asl_kit import paths  # noqa: E402

paths.import_checkout()

import anthropic  # noqa: E402
import httpx  # noqa: E402

from asl_kit import claude_client as CC  # noqa: E402
from autoscilab.llm import client as LC  # noqa: E402
from autoscilab.llm.prompts import LLMProposal, SYSTEM_PROMPT  # noqa: E402
from mock_messages import MockMessages, mock_client  # noqa: E402

LC.time.sleep = lambda s: None   # the released retry loop's back-off, skipped in tests
RESULTS = []


def fresh(cap_cell=5.0, cap_folder=50.0, **handler_kw):
    folder = tempfile.mkdtemp(prefix="asl_wire_")
    cell = os.path.join(folder, "cell")
    ledger = CC.UsageLedger(cell, cap_cell, cap_folder, mock=True)
    handler = MockMessages(**handler_kw)
    CC.install_claude_path(ledger, client_factory=lambda: mock_client(handler))
    return ledger, handler


def rows(ledger):
    return [json.loads(l) for l in open(ledger.path)] if os.path.exists(ledger.path) else []


def test_complete_json_wire_body():
    ledger, handler = fresh()
    client = LC.LLMClient(model="claude-opus-5-5", api_key="local", base_url="anthropic-messages://claude-path",
                          max_completion_tokens=4096)
    assert isinstance(client._oa_client, CC.AnthropicChatShim) and client._use_openai and not client._is_thinking
    prompt = "GOAL: synthetic\n\nDOMAIN: unknown_system\nPARAMETERS: X1, X2\n- X1: real number in [0.5, 8]\n- X2: real number in [0.1, 10]\n"
    out = client.complete_json(messages=[{"role": "user", "content": prompt}], system=SYSTEM_PROMPT,
                               schema=LLMProposal, tool_name="propose_experiments")
    assert isinstance(out, LLMProposal) and out.search_regions[0].bounds["X1"] == [0.5, 8.0]
    assert len(handler.requests) == 1
    req = handler.requests[0]
    assert req["method"] == "POST" and req["url"].endswith("/v1/messages"), req["url"]
    assert req["headers"]["x-api-key"] == "stub-not-a-key" and req["headers"].get("anthropic-version")
    body = req["body"]
    assert set(body) == {"model", "max_tokens", "system", "messages", "thinking", "output_config"}, sorted(body)
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 16000
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "medium"}
    assert body["messages"] == [{"role": "user", "content": prompt}]
    # the system text on the wire is the released complete_json system message, byte for byte
    assert body["system"].startswith(SYSTEM_PROMPT + "\n\nYou MUST respond with a single valid JSON object with these fields:")
    assert "SCHEMA:" in body["system"]
    r = rows(ledger)[0]
    assert r["max_tokens_requested"] == 4096 and r["max_tokens_sent"] == 16000
    assert r["dropped"] == ["response_format"] and r["json_mode_requested"] and not r["json_extracted"]
    assert r["usd"] == round((r["input_tokens"] * 4 + r["output_tokens"] * 20) / 1e6, 8)
    assert r["thinking_blocks"] == 1 and r["stop_reason"] == "end_turn"
    req_log = [json.loads(l) for l in open(ledger.requests_path)][0]
    assert req_log["system"] == body["system"] and req_log["messages"] == body["messages"]


def test_complete_small_max_tokens_and_temperature_dropped():
    ledger, handler = fresh()
    client = LC.LLMClient(model="claude-opus-5-5", temperature=0.3)
    raw = client.complete(messages=[{"role": "user", "content": "pick"}],
                          system="Respond with ONLY a single integer - no explanation.", max_tokens=16)
    assert raw == "0"
    body = handler.requests[0]["body"]
    assert body["max_tokens"] == 16000 and "temperature" not in body
    r = rows(ledger)[0]
    assert r["max_tokens_requested"] == 16 and r["dropped"] == ["temperature"] and not r["json_mode_requested"]


def test_json_object_extraction():
    ledger, handler = fresh(prefix_text="Here is the object you asked for:\n")
    client = LC.LLMClient(model="claude-opus-5-5")
    out = client.complete_json(messages=[{"role": "user", "content": "PARAMETERS: X1\n- X1: real number in [0, 1]\n"}],
                               system=SYSTEM_PROMPT, schema=LLMProposal)
    assert isinstance(out, LLMProposal)
    assert rows(ledger)[0]["json_extracted"] is True and len(handler.requests) == 1


def test_refusal_raises_after_recording():
    ledger, handler = fresh(stop_reason="refusal")
    client = LC.LLMClient(model="claude-opus-5-5")
    try:
        client.complete(messages=[{"role": "user", "content": "x"}], system="s")
        raise AssertionError("no refusal raised")
    except CC.ClaudeRefusal:
        pass
    assert len(handler.requests) == 1 and rows(ledger)[0]["stop_reason"] == "refusal"


def test_overloaded_goes_through_released_retry_loop():
    ledger, handler = fresh(status=529)
    client = LC.LLMClient(model="claude-opus-5-5")
    try:
        client.complete(messages=[{"role": "user", "content": "x"}], system="s")
        raise AssertionError("no error raised")
    except anthropic.APIStatusError as exc:
        assert exc.status_code == 529
    assert len(handler.requests) == 3          # the released client's max_retries = 3
    assert all(r.get("error") and r["status_code"] == 529 and r["usd"] == 0.0 for r in rows(ledger))


def test_cost_cap_refuses_before_sending():
    ledger, handler = fresh(cap_cell=0.01)
    client = LC.LLMClient(model="claude-opus-5-5")
    try:
        client.complete(messages=[{"role": "user", "content": "x"}], system="s")
        raise AssertionError("cap not enforced")
    except CC.CostCapExceeded:
        pass
    r = rows(ledger)
    assert handler.requests == [] and len(r) == 1 and r[0]["refused_by_cap"] and r[0]["usd"] == 0.0
    t = ledger.totals()
    assert t["calls"] == 0 and t["refused_by_cap"] == 1 and t["usd"] == 0.0


def test_folder_cap_counts_other_cells():
    ledger, handler = fresh(cap_cell=5.0, cap_folder=0.40)
    other = os.path.join(ledger.folder_dir, "other_cell")
    os.makedirs(other)
    with open(os.path.join(other, "llm_usage.jsonl"), "w") as fh:
        fh.write(json.dumps({"usd": 0.30}) + "\n")
    client = LC.LLMClient(model="claude-opus-5-5")
    try:
        client.complete(messages=[{"role": "user", "content": "x"}], system="s")
        raise AssertionError("folder cap not enforced")
    except CC.CostCapExceeded as exc:
        assert "folder cap" in str(exc)
    assert handler.requests == [] and rows(ledger)[0]["refused_by_cap"]


def test_non_claude_models_keep_released_init():
    fresh()
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-key")
    client = LC.LLMClient(model="gpt-4o-mini")
    assert not isinstance(client._oa_client, CC.AnthropicChatShim)
    assert type(client._oa_client).__module__.startswith("openai")


def test_sdk_kwargs_split():
    c = mock_client(MockMessages())
    kw = CC.sdk_call_kwargs(c.messages.create, {"model": "m", "max_tokens": 1, "system": "s", "messages": [],
                                                "thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}})
    import inspect
    params = inspect.signature(c.messages.create).parameters
    for k in ("thinking", "output_config"):
        assert (k in kw) == (k in params) and ((k in kw.get("extra_body", {})) == (k not in params))


if __name__ == "__main__":
    print(f"anthropic {anthropic.__version__} httpx {httpx.__version__} python {sys.version.split()[0]}")
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", name, type(exc).__name__, exc)
    print("RESULT", "PASS" if not failed else f"FAIL ({failed})")
    sys.exit(1 if failed else 0)
