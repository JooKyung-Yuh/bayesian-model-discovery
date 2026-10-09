"""Per-candidate admission of a k = 3 reply, written after the paid BoxingGym canary.

The canary (death, seed 31, joint_eig) lost a whole round because one of three valid
candidates was non-finite under its prior: the proposer admitted the array as a whole, failed the reply on
the first PriorProbeRejection, and the repair turn asked for ONE object, which the array parser then
refused. These tests pin the repaired contract with a scripted client (the proposer tests' fixture) and a
scripted admission hook; no fitting, no live client:

- one probe-failing candidate among three → the two others are returned, repairs 0, the refusal recorded
  under the candidate's payload index (error_type PriorProbeRejection, failure_class prior_probe);
- all three refused → one repair turn whose instruction asks for a JSON array of exactly 3 candidate objects;
- a single object as the first k = 3 reply → the same array instruction, then the array is taken;
- review rounds: the object {candidates, revive} with one refused candidate keeps the revive list and the
  two survivors; all refused → the repair instruction names the {"candidates", "revive"} object;
- k = 1 keeps its single-object repair sentence, byte for byte.
"""
import json

from . import count_proposer as cp
from .prior_probe_detail import PriorProbeRejection
from .test_count_proposer import p1_fixture

CELL, FAMILY = "111", "binomial"
INPUT = "u"


def _structures():
    from .count_models import initial_hypotheses
    return initial_hypotheses(FAMILY)


def _item(expression, parameters):
    return {"expression": expression, "family": FAMILY, "parameters": dict(parameters)}


GOOD_A = _item("b0+b1*u", {"b0": "intercept", "b1": "signed"})
GOOD_B = _item("b0+b1*u+b2*square(u)", {"b0": "intercept", "b1": "signed", "b2": "signed"})
BAD = _item("b0+b1*exp(u)", {"b0": "intercept", "b1": "signed"})  # the scripted probe refuses exp(


def _probe(refuse=("exp(",)):
    """A scripted admission hook: refuses an expression that contains any of `refuse`."""
    seen = []

    def admit(candidate):
        """Admission hook: record the expression and raise if it is refused."""
        seen.append(candidate.expression)
        if any(token in candidate.expression for token in refuse):
            raise PriorProbeRejection(
                f"Tried the prior probe; term exp(u) is non-finite at u=1.733 ({candidate.expression})")

    admit.seen = seen
    return admit


def _propose(tmp_path, replies, admit, *, limit=3, archive_review=False, round_=7):
    proposer, calls, context = p1_fixture(tmp_path, replies)
    context = {**context, "round": round_}
    # the BoxingGym runner selects composition paragraph v2 for its k = 3 request
    previous = cp.set_composition_paragraph("v2" if limit == 3 else "v1")
    try:
        models, record = proposer.propose_one("{\"packet\": 1}", context=context, initial_family=FAMILY,
                                              initial_structures=_structures(), admit=admit,
                                              proposal_limit=limit, archive_review=archive_review)
    finally:
        cp.set_composition_paragraph(previous)
    return models, record, calls


def _repair_turn(calls):
    return calls[1]["messages"][-1]["content"]


def test_one_refused_candidate_leaves_the_other_two_and_no_repair(tmp_path):
    """One probe-refused candidate leaves the other two, with no repair turn."""
    admit = _probe()
    models, record, calls = _propose(tmp_path, [json.dumps([BAD, GOOD_A, GOOD_B])], admit)
    assert [m.expression for m in models] == [GOOD_A["expression"], GOOD_B["expression"]]
    assert len(calls) == 1 and record["repairs"] == 0 and record["status"] == "parsed"
    assert record["valid_candidate_count"] == 2
    assert admit.seen == [BAD["expression"], GOOD_A["expression"], GOOD_B["expression"]]
    (refused,) = [r for r in record["rejected_candidates"] if r.get("error_type") == "PriorProbeRejection"]
    assert refused["index"] == 0 and refused["expression"] == BAD["expression"]
    assert refused["failure_class"] == "prior_probe" and refused["repair_eligible"] is False
    assert "non-finite" in refused["reason"]


def test_a_grammar_rejection_and_a_probe_refusal_are_both_recorded_and_the_survivor_returned(tmp_path):
    """A grammar rejection and a probe refusal are both recorded and the survivor is returned."""
    grammar_bad = _item("b0+b1*undeclared_input", {"b0": "intercept", "b1": "signed"})
    models, record, calls = _propose(tmp_path, [json.dumps([grammar_bad, BAD, GOOD_A])], _probe())
    assert [m.expression for m in models] == [GOOD_A["expression"]]
    assert len(calls) == 1 and record["repairs"] == 0
    reasons = {r["index"]: r for r in record["rejected_candidates"]}
    assert reasons[0]["reason"] == "unknown input name undeclared_input"
    assert reasons[1]["error_type"] == "PriorProbeRejection"


def test_all_three_refused_is_one_repair_turn_asking_for_the_array(tmp_path):
    """When all three candidates are refused, one repair turn asks for the JSON array."""
    bad_b = _item("b0+b1*exp(u)+b2*u", {"b0": "intercept", "b1": "signed", "b2": "signed"})
    bad_c = _item("b0+b1*exp(square(u))", {"b0": "intercept", "b1": "signed"})
    first = json.dumps([BAD, bad_b, bad_c])
    second = json.dumps([GOOD_A, GOOD_B])
    models, record, calls = _propose(tmp_path, [first, second], _probe())
    assert len(calls) == 2 and record["repairs"] == 1 and record["status"] == "parsed"
    assert [m.expression for m in models] == [GOOD_A["expression"], GOOD_B["expression"]]
    turn = _repair_turn(calls)
    assert "JSON array of exactly 3 candidate objects" in turn
    assert "exactly one JSON object" not in turn and "{limit}" not in turn
    assert "non-finite" in turn  # the probe sentences travel in the repair turn
    # the record names every refusal: the one whole-reply entry of attempt 0 carries all three
    # probe sentences (the shared fix folds them into its reason; the hook archived each candidate)
    (batch,) = [r for r in record["rejected_candidates"] if r.get("repair_kind") == "prior_probe"]
    assert batch["attempt"] == 0 and batch["repair_eligible"] is True
    assert batch["reason"].count("non-finite") == 3
    for expression in (BAD["expression"], bad_b["expression"], bad_c["expression"]):
        assert expression in batch["reason"]


def test_a_single_object_first_reply_is_repaired_with_the_array_text_and_then_taken(tmp_path):
    """A single-object first reply is repaired with the array wording, then the array is accepted."""
    first = json.dumps(GOOD_A)
    second = json.dumps([GOOD_A, GOOD_B, _item("b0+b1*tanh(u)", {"b0": "intercept", "b1": "signed"})])
    models, record, calls = _propose(tmp_path, [first, second], _probe())
    assert len(calls) == 2 and record["repairs"] == 1 and len(models) == 3
    turn = _repair_turn(calls)
    assert "reply must be a JSON array of 3 candidates" in turn
    assert "JSON array of exactly 3 candidate objects" in turn and "exactly one JSON object" not in turn
    assert record["rejected_candidates"][0]["repair_kind"] == "malformed"


def test_a_review_round_object_with_one_refused_candidate_keeps_revive_and_the_survivors(tmp_path):
    """In a review round, one refused candidate keeps the revive list and the two survivors."""
    body = json.dumps({"candidates": [GOOD_A, BAD, GOOD_B], "revive": ["abc123"]})
    models, record, calls = _propose(tmp_path, [body], _probe(), archive_review=True)
    assert [m.expression for m in models] == [GOOD_A["expression"], GOOD_B["expression"]]
    assert len(calls) == 1 and record["repairs"] == 0
    assert record["revive_requested"] == ["abc123"] and record["reply_shape"] == "object"
    (refused,) = [r for r in record["rejected_candidates"] if r.get("error_type") == "PriorProbeRejection"]
    assert refused["index"] == 1


def test_a_review_round_with_everything_refused_repairs_with_the_object_wording(tmp_path):
    """In a review round, refusing every candidate triggers a repair using the object wording."""
    first = json.dumps({"candidates": [BAD], "revive": []})
    second = json.dumps({"candidates": [GOOD_A], "revive": ["abc123"]})
    models, record, calls = _propose(tmp_path, [first, second], _probe(), archive_review=True)
    assert len(calls) == 2 and record["repairs"] == 1 and [m.expression for m in models] == [GOOD_A["expression"]]
    turn = _repair_turn(calls)
    assert '"candidates"' in turn and '"revive"' in turn and "exactly 3 candidate objects" in turn
    assert "exactly one JSON object {" not in turn
    assert record["revive_requested"] == ["abc123"]


def test_k_one_keeps_its_single_object_repair_sentence(tmp_path):
    """With k = 1 the repair sentence stays the single-object one, byte for byte."""
    first = json.dumps(BAD)
    second = json.dumps(GOOD_A)
    models, record, calls = _propose(tmp_path, [first, second], _probe(), limit=1)
    assert len(calls) == 2 and record["repairs"] == 1 and [m.expression for m in models] == [GOOD_A["expression"]]
    turn = _repair_turn(calls)
    assert turn == cp.REPAIR_INSTRUCTION.format(error=record["rejected_candidates"][0]["reason"])
    assert record["rejected_candidates"][0]["repair_kind"] == "prior_probe"


def test_the_hook_s_other_exceptions_still_end_the_round_without_a_repair(tmp_path):
    """Non-probe exceptions from the admission hook end the round without a repair."""
    def admit(candidate):
        """Admission hook that always raises, simulating a broken hook."""
        raise RuntimeError("the hook itself broke")

    models, record, calls = _propose(tmp_path, [json.dumps([GOOD_A, GOOD_B])], admit)
    assert models == [] and record["status"] == "proposal_failed" and len(calls) == 1
    assert record["rejected_candidates"][0]["repair_eligible"] is False
