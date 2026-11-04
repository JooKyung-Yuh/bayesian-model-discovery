#!/usr/bin/env python3
"""Wire tests of the Claude path: the RELEASED BoxingGym callers -> the kit's adapters -> the official anthropic SDK ->
httpx.MockTransport. The assertions read the JSON body and headers the SDK put on the wire. 0 USD: no network.

  env -u FORCE_COLOR PYTHONDONTWRITEBYTECODE=1 <venv python> -B tests/test_claude_wire.py
"""
import asyncio
import json
import os
import sys
import tempfile

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
sys.dont_write_bytecode = True
os.environ.setdefault("PYTENSOR_FLAGS", "base_compiledir=" + tempfile.mkdtemp(prefix="appr_wire_ptc_"))
os.environ.pop("ANTHROPIC_API_KEY", None)

from appr_kit import paths  # noqa: E402

paths.import_checkout()

import anthropic  # noqa: E402
import httpx  # noqa: E402

from appr_kit import claude_path as CP  # noqa: E402
from appr_kit import scripted  # noqa: E402
from appr_kit.mock_transport import MessagesAPIMock, mock_client  # noqa: E402


def fresh(cap_cell=5.0, cap_folder=50.0, task="dugongs", **handler_kw):
    folder = tempfile.mkdtemp(prefix="appr_wire_")
    ledger = CP.UsageLedger(os.path.join(folder, "cell"), cap_cell, cap_folder, mock=True)
    handler = MessagesAPIMock(task, **handler_kw)
    path = CP.ClaudePath(ledger, mock_client(handler))
    CP.bind_box_loop(path)
    return ledger, handler, path


def rows(ledger):
    return ledger.rows()


def test_scientist_wire_body():
    ledger, handler, path = fresh()
    client = CP.ScientistClient(path)
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Think about where to observe next."}]}]
    resp = client.messages.create(model="claude-opus-5-5", system="SYSTEM TEXT", messages=msgs, max_tokens=512,
                                  temperature=0.0)
    assert resp.content[0].text == scripted.reply_text("dugongs", "SYSTEM TEXT", msgs)
    req = handler.requests[0]
    assert req["method"] == "POST" and req["url"].endswith("/v1/messages"), req["url"]
    assert req["headers"]["x-api-key"] == "mock-not-a-key" and req["headers"].get("anthropic-version")
    body = req["body"]
    assert set(body) == scripted.SENT_FIELDS, sorted(body)
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == 16000
    assert body["thinking"] == {"type": "adaptive"} and body["output_config"] == {"effort": "medium"}
    assert body["system"] == "SYSTEM TEXT" and body["messages"] == msgs and "temperature" not in body
    r = rows(ledger)[0]
    assert r["role"] == "scientist" and r["max_tokens_requested"] == 512 and r["max_tokens_sent"] == 16000
    assert r["dropped"] == ["temperature"] and r["thinking_blocks"] == 1 and r["text_blocks"] == 1
    assert r["block_types"] == ["thinking", "text"] and r["stop_reason"] == "end_turn"
    assert r["usd"] == round((r["input_tokens"] * 4 + r["output_tokens"] * 20) / 1e6, 8)
    req_log = [json.loads(l) for l in open(ledger.requests_path)][0]
    assert req_log["system"] == body["system"] and req_log["messages"] == body["messages"]


def test_released_lmexperimenter_through_scientist_client():
    from boxing_gym.agents import agent as A
    ledger, handler, path = fresh(task="death")
    saved = A.anthropic
    A.anthropic = type("NoRealClient", (), {"Anthropic": staticmethod(lambda: None)})
    try:
        sci = A.LMExperimenter("claude-opus-5-5", temperature=0.0, max_tokens=512)
    finally:
        A.anthropic = saved
    sci.llm = CP.ScientistClient(path)
    sci.set_system_message("base system")
    obs = sci.generate_actions(None)
    assert obs == scripted.OBSERVE["death"], obs
    body = handler.requests[0]["body"]
    assert body["system"] == "base system" and body["messages"][0]["role"] == "user"
    assert sci.messages[-1]["role"] == "assistant" and sci.messages[-1]["content"][0]["text"].startswith("<thought>")
    sci.generate_actions("12")
    assert [m["role"] for m in handler.requests[1]["body"]["messages"]] == ["user", "assistant", "user"]


def test_box_loop_adapter_n_choices_and_merge():
    ledger, handler, path = fresh(task="death")
    llm = CP.AnthropicAsOpenAI(max_tokens=4096)
    system = "\nYou are a brilliant statistician specializing in critiquing and proposing revisions of models!\n..."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": "I will give you examples."},
                {"role": "user", "content": "1. Program: ..."}]
    resp = asyncio.run(llm(messages=messages, temperature=0.0, n=2))
    assert len(handler.requests) == 2 and len(resp.choices) == 2
    body = handler.requests[0]["body"]
    assert body["system"] == system and body["max_tokens"] == 16000 and "temperature" not in body
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "I will give you examples."},
                                                             {"type": "text", "text": "1. Program: ..."}]}]
    assert resp.choices[0].message.content == scripted.CRITIC_REPLY
    rr = rows(ledger)
    assert [r["role"] for r in rr] == ["box_loop_critic", "box_loop_critic"]
    assert all(r["merged_consecutive_turns"] == 1 and r["max_tokens_requested"] == 4096 for r in rr)
    assert resp.model == "claude-opus-5-5"
    assert resp.usage.prompt_tokens == sum(r["input_tokens"] for r in rr)
    assert resp.usage.completion_tokens == sum(r["output_tokens"] for r in rr)


def test_released_agents_and_cost_table():
    """model_search.get_llms + the released StanProposalLLMAgent.batch_prompt + BaseAgent.calc_cost, end to end."""
    from boxing_gym.agents import base_agent, model_search
    from boxing_gym.agents.llm import StanProposalLLMAgent
    from appr_kit import shim
    import logging
    ledger, handler, path = fresh(task="peregrines")
    model_search.AsyncOpenAIGPT4V = CP.AnthropicAsOpenAI
    base_agent.MODEL_COST_PER_INPUT[shim.MODEL] = shim.COST_PER_INPUT_TOKEN
    base_agent.MODEL_COST_PER_OUTPUT[shim.MODEL] = shim.COST_PER_OUTPUT_TOKEN
    cfg = type("Cfg", (), {"llm": "GPT4-V", "experiment": "box_loop"})()
    llm = model_search.get_llms(cfg)
    assert isinstance(llm, CP.AnthropicAsOpenAI) and llm.max_tokens == 4096
    agent = StanProposalLLMAgent(llm, model_id="GPT4-V", use_vision=False, batch_mode=False,
                                 logger=logging.getLogger("appr_wire_test"))
    system = "\nYou are a brilliant statistician modeling a dataset.\n..."
    asyncio.set_event_loop(asyncio.new_event_loop())   # the released batch_prompt uses get_event_loop()
    out = agent.batch_prompt(expertise=system, messages=["user prompt"], incontext_info=[], data_image=None,
                             temperature=0.0, n=1)
    assert out[0][0].startswith("Sketch:") and "```python\n" in out[0][0]
    r = rows(ledger)[0]
    assert abs(agent.total_inference_cost - (r["input_tokens"] * 4e-6 + r["output_tokens"] * 2e-5)) < 1e-12


def test_image_block_refused_before_sending():
    ledger, handler, path = fresh()
    llm = CP.AnthropicAsOpenAI(max_tokens=4096)
    messages = [{"role": "system", "content": "s"},
                {"role": "user", "content": [{"type": "text", "text": "x"},
                                             {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]}]
    try:
        asyncio.run(llm(messages=messages, temperature=0.0, n=1))
        raise AssertionError("image block not refused")
    except ValueError:
        pass
    assert handler.requests == []


def test_unknown_kwargs_refused():
    fresh()
    llm = CP.AnthropicAsOpenAI(max_tokens=4096)
    try:
        asyncio.run(llm(messages=[{"role": "user", "content": "x"}], response_format={"type": "json_object"}))
        raise AssertionError("unknown kwarg accepted")
    except TypeError:
        pass


def test_first_text_block_is_the_answer():
    ledger, handler, path = fresh(extra_text_block="SECOND BLOCK")
    ans, _usage, row = path.call("scientist", "s", [{"role": "user", "content": "x"}], 512)
    assert ans == scripted.reply_text("dugongs", "s", [{"role": "user", "content": "x"}]) and row["text_blocks"] == 2
    resp = [json.loads(l) for l in open(ledger.responses_path)][0]
    assert resp["text"] == ans and resp["other_text_blocks"] == ["SECOND BLOCK"]


def test_empty_user_turn_placeholder():
    ledger, handler, path = fresh()
    path.call("box_loop_proposal", "\nYou are a brilliant statistician modeling a dataset.\n", [{"role": "user", "content": ""}], 4096)
    assert handler.requests[0]["body"]["messages"] == [{"role": "user", "content": CP.EMPTY_USER_PLACEHOLDER}]
    assert rows(ledger)[0]["empty_user_turns_replaced"] == 1 and handler.violations == []


def test_refusal_raises_after_recording():
    ledger, handler, path = fresh(stop_reason="refusal")
    try:
        path.call("scientist", "s", [{"role": "user", "content": "x"}], 512)
        raise AssertionError("no refusal raised")
    except CP.ClaudeRefusal:
        pass
    assert len(handler.requests) == 1 and rows(ledger)[0]["stop_reason"] == "refusal"


def test_overloaded_error_recorded_and_raised():
    ledger, handler, path = fresh(status=529)
    try:
        path.call("scientist", "s", [{"role": "user", "content": "x"}], 512)
        raise AssertionError("no error raised")
    except anthropic.APIStatusError as exc:
        assert exc.status_code == 529
    r = rows(ledger)[0]
    assert r["error"] and r["status_code"] == 529 and r["usd"] == 0.0


def test_cell_cap_refuses_before_sending():
    ledger, handler, path = fresh(cap_cell=0.01)
    try:
        path.call("scientist", "s", [{"role": "user", "content": "x"}], 512)
        raise AssertionError("cap not enforced")
    except CP.CostCapExceeded:
        pass
    r = rows(ledger)
    assert handler.requests == [] and len(r) == 1 and r[0]["refused_by_cap"] and r[0]["usd"] == 0.0
    t = ledger.totals()
    assert t["calls"] == 0 and t["refused_by_cap"] == 1 and t["usd"] == 0.0


def test_folder_cap_counts_other_cells():
    ledger, handler, path = fresh(cap_cell=5.0, cap_folder=0.40)
    other = os.path.join(ledger.folder_dir, "other_cell")
    os.makedirs(other)
    with open(os.path.join(other, "llm_usage.jsonl"), "w") as fh:
        fh.write(json.dumps({"usd": 0.30}) + "\n")
    try:
        path.call("scientist", "s", [{"role": "user", "content": "x"}], 512)
        raise AssertionError("folder cap not enforced")
    except CP.CostCapExceeded as exc:
        assert "folder cap" in str(exc)
    assert handler.requests == [] and rows(ledger)[0]["refused_by_cap"]


def test_worst_case_rule():
    ledger, handler, path = fresh()
    system, msg = "a" * 1900, "b" * 1900
    path.call("scientist", system, [{"role": "user", "content": msg}], 512)
    r = rows(ledger)[0]
    assert r["request_chars"] == 3800 and abs(r["worst_usd_checked"] - (2000 * 4 + 16000 * 20) / 1e6) < 1e-9


def test_sdk_kwargs_split():
    c = mock_client(MessagesAPIMock("death"))
    kw = CP.sdk_call_kwargs(c.messages.create, {"model": "m", "max_tokens": 1, "system": "s", "messages": [],
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
