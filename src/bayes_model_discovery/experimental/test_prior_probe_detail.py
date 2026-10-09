"""The prior probe names the non-finite sub-expression, and the rejection
enters the proposer's one repair turn. Synthetic expressions and an injected client
only; no credentials, no network, no run text."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from . import count_proposer as cp
from .belief_packet import (
    ARCHIVE_REASON_KEYS,
    _archive_entry,
    core_interface,
    interface_grid,
)
from .core_count import prior_probe
from .count_models import Hypothesis, named_input_values, standardized_expression
from .prior_probe_detail import (
    PRIOR_PROBE_MESSAGE,
    PROBE_SENTENCE_GUIDANCE,
    PROBE_SENTENCE_PREFIX,
    PriorProbeRejection,
    first_nonfinite_subexpression,
    nonfinite_probe_sentence,
    prior_probe_reason,
)

SYNTHETIC_INTERFACE = {
    "inputs": [{"name": "t", "range": [0.0, 10.0]}],
    "derived_inputs": {"tz": standardized_expression("t", 0.0, 10.0)},
}


def synthetic_inputs(points=9):
    """Return synthetic inputs for the probe tests."""
    return named_input_values(SYNTHETIC_INTERFACE, interface_grid(SYNTHETIC_INTERFACE, points))


def synthetic_particles(count=4):
    """Return synthetic posterior particles for the probe tests."""
    rng = np.random.default_rng(7)
    return {"a": rng.lognormal(0.0, 1.0, count), "b": rng.normal(0.0, 1.0, count),
            "c": np.full(count, 0.5)}


# --------------------------------------------------------------------------
# 1. the finder: which node, which input, which value, which range
# --------------------------------------------------------------------------


@pytest.mark.parametrize("expression, term", [
    ("a*sqrt(tz)", "sqrt(tz)"),
    ("b + a*log(tz + c)", "log(tz + c)"),
    ("a*power(tz, 2.5)", "power(tz, 2.5)"),
    ("b + a*tz**0.5", "tz**0.5"),
])
def test_the_innermost_non_finite_node_is_named_with_the_standardized_input_at_minus_one(expression, term):
    """Check that the innermost non finite node is named with the standardized input at minus one."""
    found = first_nonfinite_subexpression(
        expression, synthetic_particles(), synthetic_inputs(), SYNTHETIC_INTERFACE)
    assert found["term"] == term
    assert found["input_name"] == "tz" and found["input_value"] == -1.0
    assert found["range"] == [-1.0, 1.0] and found["range_kind"] == "derived_on_grid"
    assert found["grid_index"] == 0


@pytest.mark.parametrize("expression", ["b + a*tz", "a*exp(b*t)", "log(a) + b*t", "a*sqrt(t)", "a*square(tz)"])
def test_a_finite_expression_names_nothing(expression):
    """Check that a finite expression names nothing."""
    assert first_nonfinite_subexpression(
        expression, synthetic_particles(), synthetic_inputs(), SYNTHETIC_INTERFACE) is None


def test_a_base_input_carries_its_declared_range():
    """Check that a base input carries its declared range."""
    found = first_nonfinite_subexpression(
        "a*log(t - c)", synthetic_particles(), synthetic_inputs(), SYNTHETIC_INTERFACE)
    assert found["term"] == "log(t - c)"
    assert found["input_name"] == "t" and found["input_value"] == 0.0
    assert found["range"] == [0.0, 10.0] and found["range_kind"] == "declared"


def test_a_coefficient_only_term_has_no_input_and_the_sentence_says_so():
    """Check that a coefficient only term has no input and the sentence says so."""
    found = first_nonfinite_subexpression(
        "b + a/(c - c)", synthetic_particles(), synthetic_inputs(), SYNTHETIC_INTERFACE)
    assert found["term"] == "a/(c - c)" and found["input_name"] is None and found["range"] is None
    sentence = nonfinite_probe_sentence(found)
    assert sentence == f"{PROBE_SENTENCE_PREFIX}a/(c - c) is non-finite; {PROBE_SENTENCE_GUIDANCE}"


def test_the_sentence_has_the_general_shape_and_quotes_only_the_term_and_numbers():
    """Check that the sentence has the general shape and quotes only the term and numbers."""
    found = first_nonfinite_subexpression(
        "b + a*log(tz + c)", synthetic_particles(), synthetic_inputs(), SYNTHETIC_INTERFACE)
    sentence = nonfinite_probe_sentence(found)
    assert sentence == (
        "Tried the prior probe; term log(tz + c) is non-finite at tz=-1 "
        "(range on the declared inputs [-1, 1]); " + PROBE_SENTENCE_GUIDANCE
    )
    assert "\n" not in sentence and "—" not in sentence


def test_an_indexed_coefficient_is_read_over_every_level():
    """Check that an indexed coefficient is read over every level."""
    interface = {**SYNTHETIC_INTERFACE, "inputs": [*SYNTHETIC_INTERFACE["inputs"],
                                                   {"name": "g", "range": [0, 2], "kind": "categorical", "levels": ["x", "y", "z"]}]}
    grid = interface_grid(interface, 9)
    inputs = named_input_values(interface, grid)
    particles = {**synthetic_particles(), "d": np.array([[1.0, -1.0, 2.0]] * 4)}
    found = first_nonfinite_subexpression("a + sqrt(d[g])", particles, inputs, interface)
    assert found["term"] == "sqrt(d[g])" and found["input_name"] is None


# --------------------------------------------------------------------------
# 2. the probe and the archive reason
# --------------------------------------------------------------------------


def test_the_core_probe_carries_the_sentence_and_the_reason_carries_it_as_detail():
    """Check that the core probe carries the sentence and the reason carries it as detail."""
    hypothesis = Hypothesis("b0 + b1*sqrt(z)", "binomial", (("b0", "intercept"), ("b1", "signed")))
    probe, predictive = prior_probe(hypothesis, core_interface())
    assert predictive is None and probe["passed"] is False and probe["normalization_warning"] is False
    assert probe["nonfinite"]["term"] == "sqrt(z)" and probe["nonfinite"]["input_name"] == "z"
    assert probe["detail"].startswith(f"{PROBE_SENTENCE_PREFIX}sqrt(z) is non-finite at z=-1 ")
    assert probe["exception"].startswith("ValueError: ")
    reason = prior_probe_reason(probe)
    assert reason == {"validation_message": PRIOR_PROBE_MESSAGE, "detail": probe["detail"]}
    entry = _archive_entry({"model_id": hypothesis.model_id, "expression": hypothesis.expression,
                            "family": hypothesis.family, "proposed_round": 1, "outcome": "rejected",
                            "reason": reason})
    assert entry["reason"] == reason


def test_a_finite_candidate_keeps_the_probe_result_it_had():
    """Check that a finite candidate keeps the probe result it had."""
    hypothesis = Hypothesis("b0 + b1*z", "binomial", (("b0", "intercept"), ("b1", "signed")))
    probe, predictive = prior_probe(hypothesis, core_interface())
    assert probe["passed"] is True and predictive is not None
    assert "nonfinite" not in probe and "exception" not in probe and "detail" not in probe
    assert prior_probe_reason(probe) == {"validation_message": PRIOR_PROBE_MESSAGE}


def test_the_archive_accepts_the_optional_detail_only_on_a_rejection():
    """Check that the archive accepts the optional detail only on a rejection."""
    base = {"model_id": "m", "expression": "b0", "family": "binomial", "proposed_round": 1}
    assert ARCHIVE_REASON_KEYS["rejected"] == {"validation_message"}
    _archive_entry({**base, "outcome": "rejected", "reason": {"validation_message": "x"}})
    with pytest.raises(ValueError, match="archive reason keys"):
        _archive_entry({**base, "outcome": "rejected", "reason": {"validation_message": "x", "other": 1}})
    with pytest.raises(ValueError, match="archive reason keys"):
        _archive_entry({**base, "outcome": "rejected", "reason": {"detail": "x"}})
    with pytest.raises(ValueError, match="archive reason keys"):
        _archive_entry({**base, "outcome": "duplicate",
                        "reason": {"matched_model_id": "m", "matched_expression": "b0", "detail": "x"}})


# --------------------------------------------------------------------------
# 3. the proposer's admission hook: one repair turn carries the sentence
# --------------------------------------------------------------------------

FIRST = {"expression": "b0 + b1*sqrt(z)", "family": "binomial",
         "parameters": {"b0": "intercept", "b1": "signed"}}
SECOND = {"expression": "b0 + b1*z", "family": "binomial",
          "parameters": {"b0": "intercept", "b1": "signed"}}
SENTENCE = f"{PROBE_SENTENCE_PREFIX}sqrt(z) is non-finite at z=-1 (declared range [-1, 1]); {PROBE_SENTENCE_GUIDANCE}"


def injected_proposer(tmp_path, replies, name="transport"):
    """Return a stand-in proposer injected into the round under test."""
    calls = []
    queue = list(replies)

    def create(**body):
        """Stand-in factory used by the test."""
        calls.append(body)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(queue.pop(0)))],
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 12, "output_tokens": 3}),
            id="test-only-response", model="returned-test-model", stop_reason="end_turn",
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return cp.AnthropicCountProposer(tmp_path / name, client=client), calls


CONTEXT = {"run_id": "test-only-run", "world_seed": 11, "cell": "110", "proposal_rep": 0, "round": 1}


def admit_by_expression(rejected_expression, seen):
    """Stand-in admission hook that decides by the candidate expression."""
    def admit(hypothesis):
        """Stand-in admission hook used by the test."""
        seen.append(hypothesis.expression)
        if hypothesis.expression == rejected_expression:
            raise PriorProbeRejection(SENTENCE)
    return admit


def test_a_probe_rejection_earns_one_repair_turn_that_carries_the_sentence(tmp_path):
    """Check that a probe rejection earns one repair turn that carries the sentence."""
    proposer, calls = injected_proposer(tmp_path, [FIRST, SECOND])
    seen = []
    models, record = proposer.propose_one(
        "packet", context=CONTEXT, initial_family="binomial",
        admit=admit_by_expression(FIRST["expression"], seen))
    assert [m.expression for m in models] == [SECOND["expression"]]
    assert seen == [FIRST["expression"], SECOND["expression"]]
    assert record["status"] == "parsed" and record["calls"] == 2 and record["repairs"] == 1
    [rejected] = record["rejected_candidates"]
    assert rejected["repair_kind"] == "prior_probe" and rejected["reason"] == SENTENCE
    assert rejected["error_type"] == "PriorProbeRejection" and rejected["repair_eligible"]
    repair_turn = calls[1]["messages"]
    assert repair_turn[1]["role"] == "assistant" and json.loads(repair_turn[1]["content"]) == FIRST
    assert repair_turn[2]["role"] == "user" and SENTENCE in repair_turn[2]["content"]


def test_two_probe_rejections_end_the_round_as_proposal_failed(tmp_path):
    """Check that two probe rejections end the round as proposal failed."""
    proposer, calls = injected_proposer(tmp_path, [FIRST, FIRST])
    models, record = proposer.propose_one(
        "packet", context=CONTEXT, initial_family="binomial",
        admit=admit_by_expression(FIRST["expression"], []))
    assert models == [] and record["status"] == "proposal_failed" and len(calls) == 2
    assert [r["repair_kind"] for r in record["rejected_candidates"]] == ["prior_probe", "prior_probe"]


def test_an_admitted_candidate_leaves_the_record_exactly_as_without_the_hook(tmp_path):
    """Check that an admitted candidate leaves the record exactly as without the hook."""
    plain, _ = injected_proposer(tmp_path, [SECOND], "plain")
    hooked, _ = injected_proposer(tmp_path, [SECOND], "hooked")
    models_plain, record_plain = plain.propose_one("packet", context=CONTEXT, initial_family="binomial")
    seen = []
    models_hooked, record_hooked = hooked.propose_one(
        "packet", context=CONTEXT, initial_family="binomial", admit=admit_by_expression("never", seen))
    assert seen == [SECOND["expression"]]
    assert [m.model_id for m in models_plain] == [m.model_id for m in models_hooked]
    volatile = {"elapsed_s", "elapsed_seconds", "started_utc", "completed_utc"}
    strip = lambda r: {k: v for k, v in r.items() if k not in volatile and "elapsed" not in k}
    assert strip(record_plain) == strip(record_hooked)
    assert record_hooked["repairs"] == 0 and record_hooked["rejected_candidates"] == []


def test_the_repair_kind_of_the_rejection_is_named():
    """Check that the repair kind of the rejection is named."""
    assert cp.repair_kind(PriorProbeRejection("x")) == "prior_probe"
    assert cp.repair_kind(ValueError("x")) == "malformed"
