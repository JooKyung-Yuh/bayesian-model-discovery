"""A k > 1 reply's admission and repair on the shared proposer path.

The BoxingGym paid canary showed the defect this fixes: an array reply with one candidate the prior
probe refused failed as a whole, the repair turn asked for "exactly one JSON object", the model
obeyed, the array parser refused the object and the round admitted nothing. Now:

(a) a probe-refused candidate is refused alone (rejected_candidates under its payload index, error_type
    PriorProbeRejection, failure_class prior_probe, repair_eligible False), the others are admitted,
    no repair;
(b) only when no candidate is left does the reply fail, and the one repair turn asks for a JSON array
    of exactly k candidate objects;
(c) a first reply that is one object (FA-C line 20's shape) gets the same array repair text;
(d) an archive review round gets the {"candidates", "revive"} object wording and keeps its revive list;
(e) k = 1 and the round-0 burst keep REPAIR_INSTRUCTION (or the caller's slot text) byte for byte and
    write no new record key.
Both shipped transports (Anthropic, OpenAI) take the same path. Every reply here is synthetic, written
for these tests; no recorded run response is used. The client is scripted: no network, no key.
"""

import hashlib
import json
from types import SimpleNamespace

import pytest

from . import count_proposer
from .belief_packet import task_interface
from .count_models import interface_input_names
from .count_proposer import (
    REPAIR_INSTRUCTION,
    REPAIR_INSTRUCTION_ARRAY,
    REPAIR_INSTRUCTION_ARRAY_EMPTY,
    REPAIR_INSTRUCTION_REVIEW,
    AnthropicCountProposer,
    OpenAICountProposer,
    StageSlots,
)
from .positive_inference import ScalarHypothesis
from .prior_probe_detail import PriorProbeRejection

FAMILIES = ("lognormal", "gamma")
CONTEXT = {"run_id": "r", "world_seed": 31, "cell": "101", "proposal_rep": 0, "round": 5}


def sha(text):
    """Return the SHA-256 hex digest of a text."""
    return hashlib.sha256(text.encode()).hexdigest()


def seed_structure():
    """Build the seed structure hypothesis used as the initial structure in these tests."""
    interface = task_interface(
        "positive", [{"name": name, "range": [0.1, 1.0]} for name in ("x0", "x1")],
        {f"{name}z": f"({name}-0.55)/0.25" for name in ("x0", "x1")}, (0.0, None))
    return ScalarHypothesis("b0", "lognormal", (("b0", "intercept"),), interface_input_names(interface))


def item(expression):
    """Build a candidate dict for an expression, with roles assigned to the coefficients it uses."""
    return {"expression": expression, "family": "lognormal",
            "parameters": {name: ("intercept" if name == "b0" else "signed")
                           for name in ("b0", "b1", "b2") if name in expression}}


A, B, C, D, E, F = (item(form) for form in ("b0+b1*x0z", "b0+b1*x1z", "b0+b1*x0z*x1z",
                                            "b0+b1*x0z+b2*x1z", "b0+b1*x0z*x0z", "b0+b1*x1z*x1z"))
BAD_GRAMMAR = {"expression": "b0+b1*nope", "family": "lognormal",
               "parameters": {"b0": "intercept", "b1": "signed"}}


def refusing(*expressions):
    """An admission hook that refuses the named expressions as the prior probe does."""
    seen = []

    def admit(candidate):
        seen.append(candidate.expression)
        if candidate.expression in expressions:
            raise PriorProbeRejection(f"non-finite under its own prior: {candidate.expression}")
    admit.seen = seen
    return admit


def anthropic(tmp_path, replies, *, stop_reason="end_turn"):
    """Build an Anthropic-style proposer with a scripted client; return the proposer and the list of request bodies."""
    bodies = []

    def create(**body):
        bodies.append(body)
        text = replies[min(len(bodies) - 1, len(replies) - 1)]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text if isinstance(text, str) else json.dumps(text))],
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 1, "output_tokens": 1}),
            id=f"k-array-{len(bodies)}", model="k-array-fake",
            stop_reason=stop_reason if len(bodies) == 1 else "end_turn")

    proposer = AnthropicCountProposer(tmp_path / "records", client=SimpleNamespace(
        messages=SimpleNamespace(create=create)))
    return proposer, bodies


def openai(tmp_path, replies):
    """Build an OpenAI-style proposer with a scripted client; return the proposer and the list of request bodies."""
    bodies = []

    def create(**body):
        bodies.append(body)
        text = replies[min(len(bodies) - 1, len(replies) - 1)]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(text)), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
            id=f"k-array-openai-{len(bodies)}", model="k-array-fake")

    proposer = OpenAICountProposer(tmp_path / "records_openai", client=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    return proposer, bodies


def propose(proposer, *, admit=None, proposal_limit=3, **options):
    """Run one proposal through the proposer with the shared test settings."""
    return proposer.propose_one(
        '{"packet": 1}', context=dict(CONTEXT, **options.pop("context", {})),
        initial_family="lognormal", initial_structures=(seed_structure(),), allowed_families=FAMILIES,
        max_rounds=30, proposal_limit=proposal_limit, admit=admit, **options)


def array_text(k=3):
    """Return the array repair text with its candidate limit set to k."""
    return REPAIR_INSTRUCTION_ARRAY.replace("{limit}", str(k))


@pytest.fixture
def composition_v2():
    """Fixture: switch the composition paragraph to v2 for the test and restore v1 afterwards."""
    count_proposer.set_composition_paragraph("v2")
    yield
    count_proposer.set_composition_paragraph("v1")


# (a) ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("transport", (anthropic, openai))
def test_a_probe_refused_candidate_is_refused_alone(tmp_path, transport):
    """A candidate refused by the prior probe is refused alone, the others are admitted, and no repair turn is made."""
    proposer, bodies = transport(tmp_path, [[A, B, C]])
    admit = refusing(B["expression"])
    models, record = propose(proposer, admit=admit)
    assert [model.expression for model in models] == [A["expression"], C["expression"]]
    assert admit.seen == [A["expression"], B["expression"], C["expression"]]
    assert record["status"] == "parsed" and record["calls"] == 1 and record["repairs"] == 0
    assert record["valid_candidate_count"] == 2 and len(bodies) == 1
    [entry] = record["rejected_candidates"]
    assert entry == {"index": 1, "expression": B["expression"], "family": "lognormal",
                     "reason": f"non-finite under its own prior: {B['expression']}",
                     "parameters": {"b0": "intercept", "b1": "signed"}, "failure_class": "prior_probe",
                     "error_type": "PriorProbeRejection", "repair_eligible": False}


def test_a_payload_index_counts_the_parsers_own_refusals(tmp_path):
    """The payload index of a refused candidate counts the parser's own refusals too."""
    proposer, _ = anthropic(tmp_path, [[BAD_GRAMMAR, A, B]])
    models, record = propose(proposer, admit=refusing(B["expression"]))
    assert [model.expression for model in models] == [A["expression"]]
    assert [(entry["index"], entry.get("error_type")) for entry in record["rejected_candidates"]] == [
        (0, None), (2, "PriorProbeRejection")]
    assert record["repairs"] == 0


# (b) ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("transport", (anthropic, openai))
def test_b_no_candidate_left_earns_the_array_repair_turn(tmp_path, transport):
    """When no candidate is left, the reply fails and earns one array repair turn."""
    proposer, bodies = transport(tmp_path, [[A, B, C], [D, E, F]])
    admit = refusing(A["expression"], B["expression"], C["expression"])
    models, record = propose(proposer, admit=admit)
    assert [model.expression for model in models] == [D["expression"], E["expression"], F["expression"]]
    assert record["status"] == "parsed" and record["calls"] == 2 and record["repairs"] == 1
    first = record["rejected_candidates"][0]
    assert first["attempt"] == 0 and first["repair_kind"] == "prior_probe"
    assert first["failure_class"] == "prior_probe" and first["error_type"] == "PriorProbeRejection"
    assert first["reason"].startswith(count_proposer.ARRAY_REFUSAL_PREFIX)
    assert all(form["expression"] in first["reason"] for form in (A, B, C))
    assert first["repair_instruction_version"] == "repair_array_v1"
    assert first["repair_instruction_sha256"] == sha(array_text())
    repair_turn = bodies[1]["messages"][-1]["content"]
    assert repair_turn == array_text().format(error=first["reason"])
    assert "Reply with a JSON array of exactly 3 candidate objects" in repair_turn
    assert bodies[1]["messages"][-2] == {"role": "assistant", "content": json.dumps([A, B, C])}


# (c) ------------------------------------------------------------------------------------------

def test_c_a_one_object_first_reply_gets_the_array_repair_text(tmp_path):
    """A first reply that is a single object gets the array repair text."""
    proposer, bodies = anthropic(tmp_path, [A, [D, E, F]])
    models, record = propose(proposer, admit=refusing())
    assert len(models) == 3 and record["repairs"] == 1 and record["status"] == "parsed"
    first = record["rejected_candidates"][0]
    assert first["repair_kind"] == "malformed"
    assert first["reason"] == "reply must be a JSON array of 3 candidates"
    assert bodies[1]["messages"][-1]["content"] == (
        "Your previous reply failed validation with this error:\n"
        "reply must be a JSON array of 3 candidates\n"
        "Reply with a JSON array of exactly 3 candidate objects, each with the keys specified above, "
        "and nothing else.")


def test_c_an_empty_k3_reply_gets_the_array_empty_text(tmp_path):
    """An empty reply at k = 3 gets the array-empty repair text."""
    proposer, bodies = anthropic(tmp_path, ["", [D, E, F]], stop_reason="max_tokens")
    models, record = propose(proposer)
    assert len(models) == 3 and record["repairs"] == 1
    first = record["rejected_candidates"][0]
    assert first["repair_kind"] == "empty_body"
    assert first["repair_instruction_version"] == "repair_array_v1_empty"
    template = REPAIR_INSTRUCTION_ARRAY_EMPTY.replace("{limit}", "3")
    assert first["repair_instruction_sha256"] == sha(template)
    assert bodies[1]["messages"][-1] == {
        "role": "user", "content": template.format(finish_reason="max_tokens")}


# (d) ------------------------------------------------------------------------------------------

def test_d_a_review_round_refuses_one_and_keeps_its_revive_list(tmp_path, composition_v2):
    """A review round refuses one candidate and keeps its revive list."""
    proposer, _ = anthropic(tmp_path, [{"candidates": [A, B, C], "revive": ["m1"]}])
    models, record = propose(proposer, admit=refusing(C["expression"]), archive_review=True)
    assert [model.expression for model in models] == [A["expression"], B["expression"]]
    assert record["repairs"] == 0 and record["revive_requested"] == ["m1"]
    assert [entry["index"] for entry in record["rejected_candidates"]] == [2]


@pytest.mark.parametrize("first", (
    {"candidates": [A, B, C], "revive": ["m1"]},   # every candidate refused by the probe
    A,                                              # one candidate object, no "candidates" key
))
def test_d_a_review_round_repairs_with_the_object_wording(tmp_path, composition_v2, first):
    """A review round repairs with the object wording."""
    second = {"candidates": [D, E, F], "revive": ["m2"]}
    proposer, bodies = anthropic(tmp_path, [first, second])
    admit = refusing(A["expression"], B["expression"], C["expression"])
    models, record = propose(proposer, admit=admit, archive_review=True)
    assert len(models) == 3 and record["repairs"] == 1 and record["revive_requested"] == ["m2"]
    entry = record["rejected_candidates"][0]
    template = REPAIR_INSTRUCTION_REVIEW.replace("{limit}", "3")
    assert entry["repair_instruction_version"] == "repair_review_v1"
    assert entry["repair_instruction_sha256"] == sha(template)
    turn = bodies[1]["messages"][-1]["content"]
    assert turn == template.format(error=entry["reason"])
    assert '{"candidates": [...], "revive": [...]}' in turn and "exactly 3 candidate objects" in turn


def test_d_a_review_round_takes_a_bare_array_repair_reply(tmp_path, composition_v2):
    """A review round accepts a bare array as its repair reply."""
    proposer, _ = anthropic(tmp_path, [A, [D, E, F]])
    models, record = propose(proposer, archive_review=True)
    assert len(models) == 3 and record["repairs"] == 1
    assert record["revive_missing"] is True and record["review_reply_shape"] == "array"


# (e) ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("burst", (False, True))
def test_e_k1_and_the_burst_keep_the_object_text_and_the_record(tmp_path, burst):
    """With k = 1 and in the burst, the object repair text and the record layout are unchanged."""
    proposer, bodies = anthropic(tmp_path, [A, B])
    context = {"round": 0, "burst_call_index": 0, "burst_wave": 0} if burst else {}
    models, record = propose(proposer, admit=refusing(A["expression"]), proposal_limit=1,
                             burst=burst, context=context)
    assert [model.expression for model in models] == [B["expression"]] and record["repairs"] == 1
    entry = record["rejected_candidates"][0]
    assert entry["repair_kind"] == "prior_probe"
    assert "repair_instruction_version" not in entry and "repair_instruction_sha256" not in entry
    assert bodies[1]["messages"][-1]["content"] == REPAIR_INSTRUCTION.format(error=entry["reason"])
    assert "Reply with exactly one JSON object" in bodies[1]["messages"][-1]["content"]


def test_e_slots_keep_their_texts_for_k1_and_take_the_k_texts_above():
    """Slot texts are kept for k = 1 and replaced by the k-candidate texts above 1."""
    slots = StageSlots(burst_paragraph="b", burst_refill_paragraph="r", repair_instruction="R {error}",
                       repair_instruction_empty="E {finish_reason}", metadata={})
    assert slots.repair_texts() == slots.repair_texts(1) == {
        "instruction": "R {error}", "empty_instruction": "E {finish_reason}"}
    assert slots.repair_texts(3) == count_proposer.k_repair_texts(3)
    assert slots.repair_texts(3, review=True) == count_proposer.k_repair_texts(3, review=True)
    texts, fields = count_proposer.repair_texts_for(ValueError("x"), 1, slots=slots)
    assert texts == slots.repair_texts() and fields == {}
    texts, fields = count_proposer.repair_texts_for(ValueError("x"), 3, burst=True, slots=slots)
    assert texts == slots.repair_texts() and fields == {}
    assert count_proposer.repair_texts_for(ValueError("x"), 1) == ({}, {})
    with pytest.raises(ValueError, match="k = 1 keeps REPAIR_INSTRUCTION"):
        count_proposer.k_repair_texts(1)


def test_e_the_k1_texts_are_the_slot_file_bytes():
    """The k = 1 texts are byte-identical to the slot file contents."""
    folder = count_proposer.PROMPT_DIRECTORY
    assert (folder / "slot_repair_v1.txt").read_text() == REPAIR_INSTRUCTION
    assert (folder / "slot_repair_v1_empty.txt").read_text() == count_proposer.REPAIR_INSTRUCTION_EMPTY
