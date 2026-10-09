"""The reasoning-model request reaches the wire through whichever SDK is installed.

Real `openai` and `anthropic` clients over `httpx.MockTransport` (a synthetic key, no network,
synthetic JSON replies), through `propose_one` with no repair turn. This file is run under two
SDK pairs: the pin (openai 1.40.3 + anthropic 0.120.2) and an older pair some hosts run
(openai 2.8.1 + anthropic 0.71.0). Before the fix covered here,
openai 1.40.3's `Completions.create` - no `max_completion_tokens` or `reasoning_effort`
parameter - raised TypeError on the reasoning-transport body before any request, and anthropic
0.71.0's `Messages.create` - no `output_config` parameter - did the same on the effort spec.
"""

import inspect
import json

import anthropic
import httpx
import openai
import pytest

from .count_proposer import (
    AnthropicCountProposer,
    OpenAICompatCountProposer,
    OpenAICountProposer,
    create_anthropic_message,
    create_openai_completion,
    sdk_call_kwargs,
    set_prompt_variant,
)

CONTEXT = {"run_id": "synthetic-rw-test", "world_seed": 99, "cell": "111", "proposal_rep": 0, "round": 1}
PACKET = json.dumps({"synthetic": True, "note": "not a recorded packet"})
REPLY = json.dumps({"expression": "b0+b1*u", "family": "binomial",
                    "parameters": {"b0": "intercept", "b1": "signed"}})
OPENAI_USAGE = {"prompt_tokens": 100, "completion_tokens": 900, "total_tokens": 1000,
                "prompt_tokens_details": {"cached_tokens": 64},
                "completion_tokens_details": {"reasoning_tokens": 850}}


def openai_completion_json():
    """Return a synthetic OpenAI chat-completion reply as a JSON-able dict."""
    return {"id": "chatcmpl-synthetic", "object": "chat.completion", "created": 0,
            "model": "synthetic-returned-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": REPLY},
                         "finish_reason": "stop"}],
            "usage": OPENAI_USAGE}


def anthropic_message_json():
    """Return a synthetic Anthropic message reply as a JSON-able dict."""
    return {"id": "msg_synthetic", "type": "message", "role": "assistant",
            "model": "synthetic-returned-model",
            "content": [{"type": "text", "text": REPLY}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 100, "output_tokens": 10}}


class Wire:
    """What the SDK put on the wire, captured by the mock transport."""

    def __init__(self, reply_json):
        """Store the reply factory and start with no captured requests."""
        self.requests = []
        self.reply_json = reply_json

    def handler(self, request):
        """Capture the request that reached the mock transport and answer with the synthetic reply."""
        self.requests.append({"url": str(request.url), "json": json.loads(request.content),
                              "headers": dict(request.headers)})
        return httpx.Response(200, json=self.reply_json(),
                              headers={"x-request-id": "synthetic-request-id"})

    def client(self):
        """Return an httpx client backed by the mock transport."""
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def flattened(body):
    """The recorded body as the wire carries it: `extra_body` merged into the top level."""
    return {**{k: v for k, v in body.items() if k != "extra_body"}, **(body.get("extra_body") or {})}


@pytest.fixture(autouse=True)
def packaged_prompt():
    """Reset the prompt variant to the packaged default before and after each test."""
    set_prompt_variant(None)
    yield
    set_prompt_variant(None)


OPENAI_CASES = [
    pytest.param({}, id="default"),
    pytest.param({"max_tokens": 2000}, id="max-tokens"),
    pytest.param({"reasoning": "effort=high"}, id="effort"),
    pytest.param({"response_format": "json_object"}, id="json-object"),
    pytest.param({"reasoning": "effort=low", "response_format": "json_object", "max_tokens": 4000},
                 id="effort-json-max"),
]


@pytest.mark.parametrize("options", OPENAI_CASES)
def test_openai_reasoning_body_reaches_the_wire_through_the_pinned_sdk(tmp_path, options):
    """Test that openai reasoning body reaches the wire through the pinned sdk."""
    wire = Wire(openai_completion_json)
    client = openai.OpenAI(api_key="sk-synthetic-not-a-key", http_client=wire.client(), max_retries=0)
    proposer = OpenAICountProposer(tmp_path / "records", client=client, model="synthetic-model", **options)
    models, record = proposer.propose_one(PACKET, context=dict(CONTEXT), initial_family="binomial",
                                          repair_attempts=0)
    assert record["status"] == "parsed" and len(models) == 1
    recorded = json.loads((tmp_path / "records" / "round_01.request.json").read_text())["bodies"][0]
    assert len(wire.requests) == 1
    assert wire.requests[0]["json"] == flattened(recorded)
    if "reasoning" in options:
        assert {"max_completion_tokens", "reasoning_effort"} <= set(wire.requests[0]["json"])
        assert "max_tokens" not in wire.requests[0]["json"]
    assert record["responses"][0]["usage"]["input_tokens"] == 100
    assert record["responses"][0]["finish_reason"] == "stop"
    assert "sk-synthetic" not in json.dumps(record)


COMPAT_CASES = [pytest.param({}, id="default"), pytest.param({"reasoning": "effort=high"}, id="d1-reasoning")]


@pytest.mark.parametrize("options", COMPAT_CASES)
def test_openai_compat_wire_is_unchanged(tmp_path, options):
    """Test that openai compat wire is unchanged."""
    wire = Wire(openai_completion_json)
    client = openai.OpenAI(api_key="sk-synthetic-not-a-key", base_url="https://synthetic.example/v1",
                           http_client=wire.client(), max_retries=0)
    proposer = OpenAICompatCountProposer(tmp_path / "records", base_url="https://synthetic.example/v1",
                                         model="synthetic-model", client=client, **options)
    models, record = proposer.propose_one(PACKET, context=dict(CONTEXT), initial_family="binomial",
                                          repair_attempts=0)
    assert record["status"] == "parsed" and len(models) == 1
    recorded = json.loads((tmp_path / "records" / "round_01.request.json").read_text())["bodies"][0]
    assert wire.requests[0]["json"] == flattened(recorded)
    assert wire.requests[0]["url"].startswith("https://synthetic.example/v1/")
    if options:
        assert recorded["extra_body"] == {"reasoning": {"effort": "high"}}
        assert wire.requests[0]["json"]["reasoning"] == {"effort": "high"}
    assert record["http_request_id"] == "synthetic-request-id"


ANTHROPIC_CASES = [
    pytest.param({}, id="default"),
    pytest.param({"reasoning": "effort=high"}, id="effort"),
    pytest.param({"reasoning": "effort=max,thinking=adaptive", "max_tokens": 8000}, id="effort-thinking"),
]


@pytest.mark.parametrize("options", ANTHROPIC_CASES)
def test_anthropic_body_reaches_the_wire_as_recorded(tmp_path, options):
    """Test that anthropic body reaches the wire as recorded."""
    wire = Wire(anthropic_message_json)
    client = anthropic.Anthropic(api_key="sk-ant-synthetic-not-a-key", http_client=wire.client(), max_retries=0)
    proposer = AnthropicCountProposer(tmp_path / "records", client=client, model="synthetic-model", **options)
    models, record = proposer.propose_one(PACKET, context=dict(CONTEXT), initial_family="binomial",
                                          repair_attempts=0)
    assert record["status"] == "parsed" and len(models) == 1
    recorded = json.loads((tmp_path / "records" / "round_01.request.json").read_text())["bodies"][0]
    assert wire.requests[0]["json"] == recorded
    if "thinking" in options.get("reasoning", ""):
        assert wire.requests[0]["json"]["thinking"] == {"type": "adaptive"}
        assert record["responses"][0]["thinking_blocks"] == 0


def test_ts_summarized_display_reaches_the_wire_and_returns_summary(tmp_path):
    """Test that the summarized thinking display setting reaches the wire and the returned summary is recorded."""
    def reply():
        message = anthropic_message_json()
        message["content"].insert(0, {"type": "thinking", "thinking": "synthetic summary",
                                       "signature": "synthetic-signature"})
        return message

    wire = Wire(reply)
    client = anthropic.Anthropic(api_key="sk-ant-synthetic-not-a-key", http_client=wire.client(), max_retries=0)
    proposer = AnthropicCountProposer(
        tmp_path / "records", client=client, model="synthetic-model",
        reasoning="effort=medium,thinking=adaptive,display=summarized",
    )
    models, record = proposer.propose_one(PACKET, context=dict(CONTEXT), initial_family="binomial",
                                          repair_attempts=0)
    assert record["status"] == "parsed" and len(models) == 1
    recorded = json.loads((tmp_path / "records/round_01.request.json").read_text())["bodies"][0]
    assert len(wire.requests) == 1 and wire.requests[0]["json"] == recorded
    assert recorded["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert recorded["output_config"] == {"effort": "medium"}
    assert record["decoding"]["thinking"] == recorded["thinking"]
    assert record["responses"][0]["thinking_text"] == ["synthetic summary"]


def test_the_split_is_decided_by_the_installed_signature():
    """A `**kwargs` create (every injected fake) and an all-parameter body are passed exactly as
    before; only a key the SDK does not name moves to extra_body; a clash or an SDK without
    extra_body is refused."""
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)

    def pinned_like(*, messages, model, max_tokens=None, response_format=None, extra_body=None):
        calls.append({"messages": messages, "model": model, "max_tokens": max_tokens,
                      "response_format": response_format, "extra_body": extra_body})

    def without_extra_body(*, messages, model, max_tokens=None):
        pass

    default = {"model": "m", "max_tokens": 1600, "messages": [{"role": "user", "content": "x"}]}
    reasoning = {"model": "m", "max_completion_tokens": 1600, "reasoning_effort": "high",
                 "messages": [{"role": "user", "content": "x"}]}
    assert sdk_call_kwargs(fake, default) == default
    assert sdk_call_kwargs(fake, reasoning) == reasoning
    assert sdk_call_kwargs(pinned_like, default) == default
    assert "extra_body" not in sdk_call_kwargs(pinned_like, default)
    split = sdk_call_kwargs(pinned_like, reasoning)
    assert split == {"model": "m", "messages": reasoning["messages"],
                     "extra_body": {"max_completion_tokens": 1600, "reasoning_effort": "high"}}
    merged = sdk_call_kwargs(pinned_like, {**reasoning, "extra_body": {"reasoning": {"effort": "low"}}})
    assert merged["extra_body"] == {"reasoning": {"effort": "low"}, "max_completion_tokens": 1600,
                                    "reasoning_effort": "high"}
    with pytest.raises(ValueError, match="both in the body and in extra_body"):
        sdk_call_kwargs(pinned_like, {**reasoning, "extra_body": {"reasoning_effort": "low"}})
    with pytest.raises(ValueError, match="neither"):
        sdk_call_kwargs(without_extra_body, reasoning)
    assert reasoning == {"model": "m", "max_completion_tokens": 1600, "reasoning_effort": "high",
                         "messages": [{"role": "user", "content": "x"}]}
    # An Anthropic-like signature without output_config (anthropic 0.71.0): the effort block
    # moves, the thinking block stays by name, the default body is untouched.
    def anthropic_like(*, model, max_tokens, messages, system=None, thinking=None, extra_body=None):
        pass

    anthropic_default = {"model": "m", "max_tokens": 1600, "system": "s", "messages": []}
    assert sdk_call_kwargs(anthropic_like, anthropic_default) == anthropic_default
    effort = {**anthropic_default, "output_config": {"effort": "high"}, "thinking": {"type": "adaptive"}}
    assert sdk_call_kwargs(anthropic_like, effort) == {
        **anthropic_default, "thinking": {"type": "adaptive"}, "extra_body": {"output_config": {"effort": "high"}}}
    # The installed signatures: no **kwargs on either; extra_body on both.
    parameters = inspect.signature(openai.resources.chat.completions.Completions.create).parameters
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    assert "extra_body" in parameters and "response_format" in parameters
    parameters = inspect.signature(anthropic.resources.messages.Messages.create).parameters
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    assert "extra_body" in parameters and "thinking" in parameters


def test_the_default_path_hands_the_fake_client_the_body_exactly_as_before():
    """Test that the default path hands the fake client the body exactly as before."""
    calls = []

    class Fake:
        def __init__(self):
            self.chat = type("Chat", (), {})()
            self.chat.completions = self
            self.messages = self

        def create(self, **body):
            calls.append(body)
            raise RuntimeError("stop here")

    body = {"model": "m", "max_tokens": 1600, "messages": [{"role": "user", "content": "x"}]}
    with pytest.raises(RuntimeError):
        create_openai_completion(Fake(), body)
    anthropic_body = {"model": "m", "max_tokens": 1600, "system": "s", "messages": [],
                      "output_config": {"effort": "high"}}
    with pytest.raises(RuntimeError):
        create_anthropic_message(Fake(), anthropic_body)
    assert calls == [body, anthropic_body] and "extra_body" not in calls[0] and "extra_body" not in calls[1]
