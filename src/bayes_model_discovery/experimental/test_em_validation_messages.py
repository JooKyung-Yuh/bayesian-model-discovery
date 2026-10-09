"""Validation-message wording unit: the validation-message wording (v1 default, v2 opt-in) and its path into a
repair turn; OpenAI JSON mode is tested beside the other transport options in
test_count_proposer.py. Synthetic only: no credential, no network."""

import json
import sys
from types import SimpleNamespace

import pytest

from . import core_count as core
from . import count_models as models
from . import count_proposer as cp
from .count_models import (
    DEFAULT_VALIDATION_MESSAGES,
    MEAN_OPERATOR_LIMIT,
    V1_VALIDATION_REFUSAL,
    VALIDATION_MESSAGE_VERSIONS,
    active_validation_messages,
    set_validation_messages,
    validate_expression,
    validation_refusal,
)

INPUTS = ("z", "u")
ROLES_AB = {"b0": "intercept", "b1": "signed", "b2": "signed"}
#: Exactly 16 mean operators: b0+b1*z has two, each `+b2*square(z)` term adds a binary +, a *
#: and one call; 2 + 3*4 = 14, plus `-b0` (unary minus + binary +) makes 16 (over the limit).
SIXTEEN = "b0+b1*z" + "+b2*square(z)" * 4 + "+(-b0)"
FIFTEEN = "b0+b1*z" + "+b2*square(z)" * 4 + "+b0"
OPERATOR_SENTENCE = ("expression has 16 mean operators; the limit is 15 (each + - * / ** and each unary "
                     "minus count 1, and each call of exp, log, sqrt, tanh, square or power counts 1)")


@pytest.fixture
def messages_version():
    """Restore the process-wide selection after a test that changes it."""
    previous = active_validation_messages()
    yield
    set_validation_messages(previous)


def test_the_default_is_v1_and_the_v1_sentence_is_the_one_every_run_produced(messages_version):
    """The default is v1 and its sentence is the one every earlier run produced."""
    assert DEFAULT_VALIDATION_MESSAGES == "v1" and active_validation_messages() == "v1"
    assert VALIDATION_MESSAGE_VERSIONS == ("v1", "v2") and MEAN_OPERATOR_LIMIT == 15
    assert V1_VALIDATION_REFUSAL == "unused coefficient or more than 15 mean operators"
    for expression, roles in (("b0+b1*z", ROLES_AB), (SIXTEEN, ROLES_AB)):
        with pytest.raises(ValueError, match="^unused coefficient or more than 15 mean operators$"):
            validate_expression(expression, roles, INPUTS)
    validate_expression(FIFTEEN, ROLES_AB, INPUTS)  # 15 operators, every coefficient used
    with pytest.raises(ValueError, match="unregistered validation messages"):
        set_validation_messages("v3")
    assert active_validation_messages() == "v1"


@pytest.mark.parametrize("expression, roles, sentence", [
    ("b0+b1*z", ROLES_AB, "coefficients declared in parameters but not used in expression: b2"),
    ("b0", ROLES_AB, "coefficients declared in parameters but not used in expression: b1, b2"),
    (SIXTEEN, ROLES_AB, OPERATOR_SENTENCE),
    ("b1*z" + "+b2*square(z)" * 4 + "+(-b2)+b2", ROLES_AB,
     "coefficients declared in parameters but not used in expression: b0; " + OPERATOR_SENTENCE),
])
def test_v2_names_only_the_causes_it_found(messages_version, expression, roles, sentence):
    """The v2 messages name only the causes they found."""
    assert set_validation_messages("v2") == "v1"
    with pytest.raises(ValueError) as refused:
        validate_expression(expression, roles, INPUTS)
    assert str(refused.value) == sentence
    # The class of the refusal does not move: v1 and v2 are both parse_format.
    assert models.failure_class(error_type="ValueError", reason=sentence, repair_eligible=True) == "parse_format"
    assert models.failure_class(error_type="ValueError", reason=V1_VALIDATION_REFUSAL,
                                repair_eligible=True) == "parse_format"


def test_v2_sentences_for_a_caller_that_passes_its_own_sets():
    """v2 sentences for a caller that passes its own name and operator sets."""
    assert validation_refusal({"b0"}, {"b0": "intercept", "b1": "signed"}, 3, version="v2") == (
        "coefficients declared in parameters but not used in expression: b1")
    assert validation_refusal({"b0", "b9"}, {"b0": "intercept"}, 3, version="v2") == (
        "names used in expression but not declared in parameters: b9")
    assert validation_refusal({"b0"}, {"b0": "intercept"}, 16, version="v2") == OPERATOR_SENTENCE
    assert validation_refusal({"b0"}, {"b0": "intercept"}, 16, version="v1") == V1_VALIDATION_REFUSAL
    assert validation_refusal({"b0"}, {"b0": "intercept"}, 16, grammar_version="v3", version="v2").endswith(
        "each call of " + ", ".join(list(models.GRAMMAR_CALL_ARITY["v3"])[:-1]) + " or "
        + list(models.GRAMMAR_CALL_ARITY["v3"])[-1] + " counts 1)")
    with pytest.raises(ValueError, match="without a cause"):
        validation_refusal({"b0"}, {"b0": "intercept"}, 3, version="v2")


def anthropic_client(replies):
    """Build a fake Anthropic client returning the given replies in order; return it with its call log."""
    calls = []

    def create(**body):
        """Record the request body and return the next canned reply."""
        calls.append(body)
        text = replies[min(len(calls) - 1, len(replies) - 1)]
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], id=f"test-only-{len(calls)}",
                               model="returned-test-model", stop_reason="end_turn",
                               usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 9, "output_tokens": 3}))

    return SimpleNamespace(messages=SimpleNamespace(create=create)), calls


CONTEXT = {"run_id": "test-only-run", "world_seed": 11, "cell": "111", "proposal_rep": 0, "round": 7}


def candidate(expression, parameters):
    """Serialize a candidate proposal as JSON text."""
    return json.dumps({"expression": expression, "family": "beta_binomial", "parameters": parameters})


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_the_repair_turn_carries_the_selected_sentence(tmp_path, messages_version, version):
    """A reply with a declared-but-unused coefficient earns one repair turn whose `{error}` is the
    sentence the selected version produces; the second reply is accepted."""
    set_validation_messages(version)
    client, calls = anthropic_client([candidate("b0+b1*z", {"b0": "intercept", "b1": "signed", "b2": "signed"}),
                                      candidate("b0+b1*z", {"b0": "intercept", "b1": "signed"})])
    proposer = cp.AnthropicCountProposer(tmp_path / version, client=client)
    proposals, record = proposer.propose_one('{"round":7}', context=CONTEXT, initial_family="binomial")
    assert len(calls) == 2 and len(proposals) == 1 and record["repairs"] == 1
    expected = (V1_VALIDATION_REFUSAL if version == "v1"
                else "coefficients declared in parameters but not used in expression: b2")
    assert record["rejected_candidates"][0]["reason"] == expected
    assert record["rejected_candidates"][0]["failure_class"] == "parse_format"
    assert calls[1]["messages"][-1] == {"role": "user", "content": cp.REPAIR_INSTRUCTION.format(error=expected)}


# ----- the runner path: `--validation-messages v2` reaches the engine switch and the record -----


class _ReplyingOpenAI:
    """A chat-completions client whose first reply declares an unused coefficient."""

    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **body):
        """Record the request body and return the next canned completion."""
        self.calls.append(body)
        index = len(self.calls)
        expression, parameters = ("b0+b1*tanh(3*z)", {"b0": "intercept", "b1": "signed", "b2": "signed"}) \
            if index == 1 else ("b0+b1*tanh(3*z)", {"b0": "intercept", "b1": "signed"})
        text = json.dumps({"expression": expression, "family": "normal", "parameters": parameters})
        return SimpleNamespace(model="synthetic-returned-model", id=f"synthetic-completion-{index}",
                               choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10,
                                                     prompt_tokens_details=SimpleNamespace(cached_tokens=0)))


def _run_cell(monkeypatch, output, *extra):
    client = _ReplyingOpenAI()
    original = cp.OpenAICountProposer
    monkeypatch.setattr(cp, "OpenAICountProposer",
                        lambda records, **kwargs: original(records, client=client, **{
                            key: value for key, value in kwargs.items() if key != "allow_live"}))
    monkeypatch.setattr(sys, "argv", ["core_count", "--output", str(output), "--profile", "dev", "--rounds", "1",
                                      "--seed", "99", "--cell", "101", "--structure", "gauss_call",
                                      "--truth-family", "normal", "--initial-family", "normal",
                                      "--proposer", "openai", "--llm-model", "synthetic-audit-model",
                                      "--allow-paid-api", *extra])
    try:
        core.main()
    except SystemExit as stop:
        assert stop.code in (0, None), stop.code
    return client


@pytest.mark.parametrize("flag", [(), ("--validation-messages", "v2")])
def test_a_cell_with_v2_writes_the_new_sentence_into_its_records_and_its_config(tmp_path, monkeypatch,
                                                                                messages_version, flag):
    """A cell with v2 writes the new sentence into its records and its config."""
    out = tmp_path / ("v2" if flag else "v1")
    client = _run_cell(monkeypatch, out, *flag)
    assert len(client.calls) == 2  # one refusal, one repair turn, one accepted reply
    record = json.loads((out / "proposals" / "t01.json").read_text())["transport"]
    reason = record["rejected_candidates"][0]["reason"]
    expected = ("coefficients declared in parameters but not used in expression: b2" if flag
                else V1_VALIDATION_REFUSAL)
    assert reason == expected and record["repairs"] == 1
    assert client.calls[1]["messages"][-1]["content"] == cp.REPAIR_INSTRUCTION.format(error=expected)
    config = json.loads((out / "config.json").read_text())
    assert config.get("validation_messages") == ("v2" if flag else None)
    assert ("validation_messages" in config) is bool(flag)
