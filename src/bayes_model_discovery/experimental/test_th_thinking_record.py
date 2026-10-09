"""An Anthropic reply's thinking summaries are recorded, masked, in order - and only
when the reply carried a thinking block, so records of runs without thinking keep every byte.

Synthetic replies only; the round path (`propose_one`), the burst path (`propose`) and, at the
end, the real anthropic SDK over `httpx.MockTransport` with a message JSON that carries a
thinking block, a redacted one and the text - run under both SDK pairs (the pin and the older pair).
"""

import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from .count_proposer import AnthropicCountProposer, anthropic_thinking_record
from .test_count_proposer import P1_CONTEXT, fixture, one_object

SECRET = "sk-ant-synthetic-not-a-real-key-0123456789abcdef"
MASKED = "<ANTHROPIC_API_KEY:masked>"
SUMMARY_1 = "The pool misses a saturating term; a tanh of the scaled input would bend the mean."
SUMMARY_2 = f"Checking the key {SECRET} is not something a summary should carry."


def thinking(text):
    """Return a synthetic thinking content block carrying the given text."""
    return SimpleNamespace(type="thinking", thinking=text, signature="test-only-signature")


def redacted():
    """Return a synthetic redacted-thinking content block."""
    return SimpleNamespace(type="redacted_thinking", data="test-only-opaque-data")


def text(reply):
    """Return a synthetic text content block carrying the reply."""
    return SimpleNamespace(type="text", text=reply)


def client_with(contents):
    """An injected Messages-API client answering each call with the next content list."""
    calls = []

    def create(**body):
        calls.append(body)
        content = contents[min(len(calls) - 1, len(contents) - 1)]
        return SimpleNamespace(content=list(content), id=f"test-only-{len(calls)}",
                               model="returned-test-model", stop_reason="end_turn",
                               usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 9, "output_tokens": 30}))

    return SimpleNamespace(messages=SimpleNamespace(create=create)), calls


def test_the_record_helper_keeps_order_masks_and_counts_and_is_empty_without_thinking():
    """Test that the record helper keeps order masks and counts and is empty without thinking."""
    response = SimpleNamespace(content=[thinking(SUMMARY_1), redacted(), thinking(SUMMARY_2), text("{}")])
    assert anthropic_thinking_record(response) == {
        "thinking_text": [SUMMARY_1, f"Checking the key {MASKED} is not something a summary should carry."],
        "redacted_thinking_blocks": 1}
    assert anthropic_thinking_record(SimpleNamespace(content=[text("{}")])) == {}
    assert anthropic_thinking_record(SimpleNamespace(content=[])) == {}
    assert anthropic_thinking_record(SimpleNamespace(content=[redacted()])) == {
        "thinking_text": [], "redacted_thinking_blocks": 1}
    assert anthropic_thinking_record(SimpleNamespace(content=[thinking("only a summary"), text("{}")])) == {
        "thinking_text": ["only a summary"]}


def test_round_path_records_the_summaries_and_the_repair_turn_echoes_the_text_only(tmp_path):
    """Round 1: thinking + redacted + a reply that is not a candidate, so the one repair turn
    is taken; round 2's reply is a candidate. Both records carry the keys; the repair turn's
    assistant echo is the text block alone; the request bodies are the request bodies from before thinking summaries were recorded."""
    client, calls = client_with([
        [thinking(SUMMARY_1), redacted(), text("not a candidate")],
        [thinking(SUMMARY_2), text(one_object())],
    ])
    proposer = AnthropicCountProposer(tmp_path / "records", client=client, reasoning="effort=high,thinking=adaptive")
    models, record = proposer.propose_one('{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert record["status"] == "parsed" and len(models) == 1 and len(calls) == 2
    first, second = record["responses"]
    assert first["thinking_text"] == [SUMMARY_1] and first["redacted_thinking_blocks"] == 1
    assert first["thinking_blocks"] == 2 and first["raw_text"] == "not a candidate"
    assert second["thinking_text"] == [f"Checking the key {MASKED} is not something a summary should carry."]
    assert "redacted_thinking_blocks" not in second and second["thinking_blocks"] == 1
    # The repair turn re-sends the text block only: no summary, no secret, anywhere in the body.
    repair = json.dumps(calls[1])
    assert SUMMARY_1 not in repair and SECRET not in repair and MASKED not in repair
    assert calls[1]["messages"][1] == {"role": "assistant", "content": "not a candidate"}
    assert list(calls[0]) == ["model", "max_tokens", "system", "messages", "output_config", "thinking"]
    # The written record is the returned one, masked, and the secret is nowhere on disk.
    stored = (tmp_path / "records" / "round_07.response.json").read_text()
    assert SECRET not in stored and MASKED in stored
    assert json.loads(stored)["responses"][0]["thinking_text"] == [SUMMARY_1]


def test_a_reply_without_thinking_adds_no_key_on_either_path(tmp_path):
    """Test that a reply without thinking adds no key on either path."""
    client, _ = client_with([[text(one_object())]])
    _, record = AnthropicCountProposer(tmp_path / "round", client=client).propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert "thinking_text" not in record["responses"][0]
    assert "redacted_thinking_blocks" not in record["responses"][0]
    assert "thinking" not in (tmp_path / "round" / "round_07.response.json").read_text()
    proposer, _calls, context, public, pool = fixture(tmp_path)
    _, metadata = proposer.propose(public, pool, context=context, initial_family="binomial")
    assert "thinking_text" not in metadata and "redacted_thinking_blocks" not in metadata
    assert "thinking" not in (tmp_path / "transport" / "event_05.response.json").read_text()


def test_burst_path_records_the_summaries_masked(tmp_path):
    """Test that burst path records the summaries masked."""
    client, _calls = client_with([[thinking(SUMMARY_2), redacted(), redacted(), text(json.dumps([json.loads(one_object())]))]])
    _proposer, _, context, public, pool = fixture(tmp_path)
    burst = AnthropicCountProposer(tmp_path / "burst", client=client, reasoning="effort=high,thinking=adaptive")
    _, metadata = burst.propose(public, pool, context=context, initial_family="binomial")
    assert metadata["thinking_text"] == [f"Checking the key {MASKED} is not something a summary should carry."]
    assert metadata["redacted_thinking_blocks"] == 2 and metadata["thinking_blocks"] == 3
    assert metadata["status"] == "parsed"
    stored = (tmp_path / "burst" / "event_05.response.json").read_text()
    assert SECRET not in stored and MASKED in stored


@pytest.mark.parametrize("path", ["round", "burst"])
@pytest.mark.parametrize("with_thinking", [False, True])
def test_ts_display_keeps_masked_summaries_on_both_paths_only_when_returned(tmp_path, path, with_thinking):
    """Test that the summarized thinking display keeps masked summaries on both paths, only when the reply returned thinking."""
    reply = one_object() if path == "round" else json.dumps([json.loads(one_object())])
    content = [thinking(SUMMARY_2), text(reply)] if with_thinking else [text(reply)]
    client, calls = client_with([content])
    proposer = AnthropicCountProposer(tmp_path / "summary", client=client,
                                      reasoning="effort=medium,thinking=adaptive,display=summarized")
    if path == "round":
        _, metadata = proposer.propose_one('{"round":7}', context=P1_CONTEXT, initial_family="binomial")
        record = metadata["responses"][0]
        filename = "round_07.response.json"
    else:
        _, _, context, public, pool = fixture(tmp_path)
        _, metadata = proposer.propose(public, pool, context=context, initial_family="binomial")
        record = metadata
        filename = "event_05.response.json"
    assert metadata["status"] == "parsed"
    assert calls[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
    stored = (tmp_path / "summary" / filename).read_text()
    assert SECRET not in stored
    if with_thinking:
        assert record["thinking_text"] == [f"Checking the key {MASKED} is not something a summary should carry."]
        assert MASKED in stored
    else:
        assert "thinking_text" not in record and "thinking_text" not in stored


def anthropic_message_json():
    """Return a synthetic Anthropic message reply as a JSON-able dict."""
    return {"id": "msg_synthetic", "type": "message", "role": "assistant", "model": "synthetic-returned-model",
            "content": [{"type": "thinking", "thinking": SUMMARY_1, "signature": "synthetic-signature"},
                        {"type": "redacted_thinking", "data": "synthetic-opaque-data"},
                        {"type": "text", "text": one_object()}],
            "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 100, "output_tokens": 10}}


def test_the_real_sdk_parses_the_thinking_blocks_the_record_keeps(tmp_path):
    """Test that the real sdk parses the thinking blocks the record keeps."""
    wire = []

    def handler(request):
        wire.append(json.loads(request.content))
        return httpx.Response(200, json=anthropic_message_json())

    client = anthropic.Anthropic(api_key="sk-ant-synthetic-not-a-key", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    proposer = AnthropicCountProposer(tmp_path / "records", client=client, model="synthetic-model",
                                      reasoning="effort=high,thinking=adaptive")
    models, record = proposer.propose_one('{"round":7}', context=P1_CONTEXT, initial_family="binomial",
                                          repair_attempts=0)
    assert record["status"] == "parsed" and len(models) == 1 and len(wire) == 1
    response = record["responses"][0]
    assert response["thinking_text"] == [SUMMARY_1] and response["redacted_thinking_blocks"] == 1
    assert response["thinking_blocks"] == 2 and response["raw_text"] == one_object()
    assert wire[0]["thinking"] == {"type": "adaptive"} and "output_config" in json.dumps(wire[0])
    assert (tmp_path / "records" / "round_07.request.json").read_text().count("thinking_text") == 0


@pytest.mark.parametrize("version", [anthropic.__version__])
def test_the_installed_sdk_names_the_block_fields_the_record_reads(version):
    """Test that the installed sdk names the block fields the record reads."""
    from anthropic.types import RedactedThinkingBlock, ThinkingBlock

    assert set(ThinkingBlock.model_fields) >= {"thinking", "signature", "type"}
    assert set(RedactedThinkingBlock.model_fields) >= {"data", "type"}
    assert version
