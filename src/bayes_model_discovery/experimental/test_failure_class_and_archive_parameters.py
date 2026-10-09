"""Failure-class and archive-parameter tests: every candidate record names its failure class, every
archive entry its declared parameters (a project decision).

Synthetic replies only, through injected clients and stubbed fits: no network, no saved run, no
stored response, and no expression or seed taken from a real run.
"""

import copy
import json
import re
import threading
from types import SimpleNamespace

import pytest

from . import core_count as core
from . import count_proposer as cp
from .belief_packet import build_belief_packet, cell_constraints, core_interface, serialize_packet
from .count_inference import CountInferenceFailure, FitTimeout
from .count_models import (
    ARRAY_REFUSAL_PREFIX,
    CANDIDATE_STATUS_CLASSES,
    COEFFICIENT_COUNT_REFUSAL,
    FAILURE_CLASSES,
    INITIAL_BURST_REFILL_POLICY,
    RESERVED_NAME_REFUSAL,
    failure_class,
    initial_hypotheses,
    judge_candidate,
    read_initial_pool_file,
    validate_expression,
)
from .prior_probe_detail import PriorProbeRejection
from .test_count_proposer import run_args, stub_fits
from .test_neuron_proposer import Client, response

#: The status table, written out here so that changing it is a change to this file too.
STATUS_CLASSES = {
    "fitted": None,
    "duplicate_not_refitted": "duplicate",
    "rejected_prior_probe": "prior_probe",
    "rejected_loglik_prior_probe": "prior_probe",
    "fit_failed": "fit_failed",
    "fit_timeout": "fit_failed",
    "fit_or_predict_failed": "fit_failed",
    "predict_failed": "fit_failed",
    "rejected_family": "parse_format",
}
ZERO = dict.fromkeys(FAILURE_CLASSES, 0)
PAIR = {"b0": "intercept", "b1": "signed"}
SEVEN = {f"b{index}": "intercept" if index == 0 else "signed" for index in range(7)}
P1_CONTEXT = {"run_id": "synthetic-pcx", "world_seed": 11, "cell": "111", "proposal_rep": 0,
              "round": 7}
PROVIDERS = pytest.mark.parametrize("provider", ["anthropic", "openai"])


# --------------------------------------------------------------------------
# 1. the mapping
# --------------------------------------------------------------------------


def test_the_six_classes_and_the_status_table_are_fixed():
    """Check that the six classes and the status table are fixed."""
    assert FAILURE_CLASSES == ("parse_format", "reserved_name", "coefficient_count", "prior_probe",
                               "fit_failed", "duplicate")
    assert CANDIDATE_STATUS_CLASSES == STATUS_CLASSES
    for status, expected in STATUS_CLASSES.items():
        assert failure_class(status=status) == expected
    # An admitted candidate's record (status "fitted") names no class.
    assert failure_class(status="fitted") is None


@pytest.mark.parametrize("parameters", [(), tuple(SEVEN.items())], ids=["none", "seven"])
def test_the_coefficient_count_constant_is_the_message_the_engine_raises(parameters):
    """Check that the coefficient count constant is the message the engine raises."""
    with pytest.raises(ValueError) as refused:
        validate_expression("b0", parameters, ("u", "z"))
    assert str(refused.value) == COEFFICIENT_COUNT_REFUSAL
    item = {"expression": "b0", "family": "binomial", "parameters": dict(parameters)}
    assert judge_candidate(item, "111", initial_hypotheses("binomial"), "binomial") == (
        False, COEFFICIENT_COUNT_REFUSAL)
    assert failure_class(reason=str(refused.value)) == "coefficient_count"


@pytest.mark.parametrize("name", ["exp", "rho", "z"], ids=["call_name", "variance_name", "input_name"])
def test_the_reserved_name_constant_is_the_message_the_engine_raises(name):
    """Check that the reserved name constant is the message the engine raises."""
    parameters = (("b0", "intercept"), (name, "signed"))
    with pytest.raises(ValueError) as refused:
        validate_expression(f"b0+{name}*u", parameters, ("u", "z"))
    assert str(refused.value) == RESERVED_NAME_REFUSAL
    item = {"expression": f"b0+{name}*u", "family": "binomial", "parameters": dict(parameters)}
    assert judge_candidate(item, "111", initial_hypotheses("binomial"), "binomial") == (
        False, RESERVED_NAME_REFUSAL)
    assert failure_class(reason=str(refused.value)) == "reserved_name"


def _one_object_refusal(payload, cell):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    with pytest.raises(ValueError) as refused:
        cp.parse_candidate_object(text, cell, "binomial")
    return str(refused.value)


@pytest.mark.parametrize("payload,cell,expected", [
    ("not json", "111", "parse_format"),
    ("[]", "111", "parse_format"),
    ({"expression": "b0"}, "111", "parse_format"),
    ({"expression": "b0", "family": "binomial", "parameters": ["b0"]}, "111", "parse_format"),
    ({"expression": "b0", "family": "binomial", "parameters": {"b0": 1}}, "111", "parse_format"),
    ({"expression": "b0+b1*q", "family": "binomial", "parameters": PAIR}, "111", "parse_format"),
    ({"expression": "b0", "family": "binomial", "parameters": {"b0": "invented"}}, "111", "parse_format"),
    ({"expression": "b0+b1*z", "family": "normal", "parameters": PAIR}, "111", "parse_format"),
    ({"expression": "b0+b1*z", "family": "beta_binomial", "parameters": PAIR}, "100", "parse_format"),
    ({"expression": "b0+b1*u", "family": "binomial", "parameters": PAIR}, "011", "parse_format"),
    ({"expression": "b0", "family": "binomial", "parameters": {}}, "111", "coefficient_count"),
    ({"expression": "b0+b1*z", "family": "binomial", "parameters": SEVEN}, "111", "coefficient_count"),
    ({"expression": "b0+exp*z", "family": "binomial",
      "parameters": {"b0": "intercept", "exp": "signed"}}, "111", "reserved_name"),
    ({"expression": "b0+z*u", "family": "binomial",
      "parameters": {"b0": "intercept", "z": "signed"}}, "111", "reserved_name"),
], ids=["invalid_json", "array_for_object", "candidate_shape", "parameters_shape",
        "parameter_value_shape", "grammar", "role", "family_menu", "family_f0", "structure_s0",
        "no_coefficient", "seven_coefficients", "reserved_call_name", "input_as_coefficient"])
def test_each_reply_refusal_maps_to_its_class(payload, cell, expected):
    """Check that each reply refusal maps to its class."""
    reason = _one_object_refusal(payload, cell)
    assert failure_class(error_type="ValueError", reason=reason, repair_eligible=True) == expected
    assert failure_class(reason=reason) == expected


@pytest.mark.parametrize("reason", [
    str(cp.EmptyResponseBody("end_turn")),
    "response did not finish with end_turn",
    "response did not finish with stop",
    ARRAY_REFUSAL_PREFIX + COEFFICIENT_COUNT_REFUSAL,
    ARRAY_REFUSAL_PREFIX + RESERVED_NAME_REFUSAL + "; " + COEFFICIENT_COUNT_REFUSAL,
    "candidate 2: " + COEFFICIENT_COUNT_REFUSAL,
    COEFFICIENT_COUNT_REFUSAL + ".",
    " " + RESERVED_NAME_REFUSAL,
], ids=["empty_body", "finish_end_turn", "finish_stop", "array_wrapper_one", "array_wrapper_two",
        "foreign_prefix", "suffix", "leading_space"])
def test_every_other_refusal_is_parse_format_and_no_fragment_is_matched(reason):
    """Check that every other refusal is parse format and no fragment is matched."""
    assert failure_class(error_type="ValueError", reason=reason, repair_eligible=True) == "parse_format"
    assert failure_class(reason=reason) == "parse_format"


def test_the_array_wrapper_is_a_whole_reply_refusal_whatever_it_quotes():
    """The one wrapper in the tree: a k > 1 reply in which nothing was accepted. It is classed
    parse_format explicitly; the engine messages it quotes are not unwrapped."""
    only = {"expression": "b0", "family": "binomial", "parameters": {}}
    with pytest.raises(ValueError) as refused:
        cp.parse_candidate_array(json.dumps([only]), "111", "binomial", proposal_limit=2)
    assert str(refused.value) == ARRAY_REFUSAL_PREFIX + COEFFICIENT_COUNT_REFUSAL
    assert failure_class(error_type="ValueError", reason=str(refused.value),
                         repair_eligible=True) == "parse_format"


def test_a_prior_probe_refusal_is_classed_by_its_type_before_its_sentence():
    """Check that a prior probe refusal is classed by its type before its sentence."""
    refusal = PriorProbeRejection(COEFFICIENT_COUNT_REFUSAL)
    assert failure_class(error_type=type(refusal).__name__, reason=str(refusal),
                         repair_eligible=True) == "prior_probe"
    assert failure_class(error_type="PriorProbeRejection", reason="synthetic sentence") == "prior_probe"


@pytest.mark.parametrize("status", ["proposal_failed", "parsed", "accepted", "rejected", ""])
def test_an_unknown_status_raises_instead_of_defaulting(status):
    """Check that an unknown status raises instead of defaulting."""
    with pytest.raises(ValueError, match="unknown candidate status"):
        failure_class(status=status)


def test_a_transport_error_has_no_class_and_mixed_arguments_are_refused():
    """Check that a transport error has no class and mixed arguments are refused."""
    with pytest.raises(ValueError, match="not repair-eligible"):
        failure_class(error_type="RuntimeError", reason="synthetic transport failure",
                      repair_eligible=False)
    with pytest.raises(ValueError, match="on its own"):
        failure_class(status="fitted", reason="synthetic")
    with pytest.raises(ValueError, match="recorded reason"):
        failure_class(error_type="ValueError")


def test_the_count_is_one_per_classed_record_and_a_repaired_refusal_still_counts():
    """Check that the count is one per classed record and a repaired refusal still counts."""
    transports = [
        {"transport": {"status": "parsed", "rejected_candidates": [
            {"attempt": 0, "repair_eligible": True, "failure_class": "reserved_name"}]}},
        {"transport": {"status": "proposal_failed", "rejected_candidates": [
            {"attempt": 0, "repair_eligible": False}]}},
        {"transport": {"status": "parsed", "rejected_candidates": []}},
        {"transport": {"status": "parsed", "mock": True}},
    ]
    events = [{"status": "fitted", "failure_class": None},
              {"status": "duplicate_not_refitted", "failure_class": "duplicate"}]
    assert core.failure_class_counts(events, transports) == {**ZERO, "reserved_name": 1, "duplicate": 1}
    assert core.failure_class_counts([], []) == ZERO
    with pytest.raises(KeyError):
        core.failure_class_counts([{"failure_class": "invented"}], [])


# --------------------------------------------------------------------------
# 2. both proposers, both entry shapes
# --------------------------------------------------------------------------


def _proposer(directory, provider, bodies):
    client = Client(provider, [body if isinstance(body, Exception) else response(provider, body)
                               for body in bodies])
    cls = cp.AnthropicCountProposer if provider == "anthropic" else cp.OpenAICountProposer
    return cls(directory / provider, client=client), client


def _written(directory, provider, stem="round_07"):
    return json.loads((directory / provider / f"{stem}.response.json").read_text())


@PROVIDERS
def test_both_proposers_class_each_repair_turn_refusal(tmp_path, provider):
    """Check that both proposers class each repair turn refusal."""
    count = {"expression": "b0+b1*z", "family": "binomial", "parameters": SEVEN}
    reserved = {"expression": "b0+exp*z", "family": "binomial",
                "parameters": {"b0": "intercept", "exp": "signed"}}
    proposer, client = _proposer(tmp_path, provider, [count, reserved])
    models, record = proposer.propose_one("synthetic packet", context=P1_CONTEXT,
                                          initial_family="binomial")
    assert models == [] and record["status"] == "proposal_failed" and len(client.calls) == 2
    assert [(entry["attempt"], entry["reason"], entry["repair_kind"], entry["failure_class"])
            for entry in record["rejected_candidates"]] == [
        (0, COEFFICIENT_COUNT_REFUSAL, "malformed", "coefficient_count"),
        (1, RESERVED_NAME_REFUSAL, "malformed", "reserved_name")]
    assert _written(tmp_path, provider)["rejected_candidates"] == record["rejected_candidates"]


@PROVIDERS
def test_both_proposers_class_parse_format_empty_body_and_prior_probe_refusals(tmp_path, provider):
    """Check that both proposers class parse format empty body and prior probe refusals."""
    good = {"expression": "b0+b1*z", "family": "binomial", "parameters": PAIR}

    def refuse(hypothesis):
        """Test helper `refuse`: stand-in or builder used by the surrounding tests."""
        raise PriorProbeRejection("synthetic prior-probe sentence")

    proposer, _ = _proposer(tmp_path, provider, ["not json", good])
    _, record = proposer.propose_one("synthetic packet", context=P1_CONTEXT,
                                     initial_family="binomial", admit=refuse)
    assert [(entry["repair_kind"], entry["failure_class"]) for entry in record["rejected_candidates"]] == [
        ("malformed", "parse_format"), ("prior_probe", "prior_probe")]
    (tmp_path / "empty").mkdir()
    proposer, _ = _proposer(tmp_path / "empty", provider, ["", good])
    models, record = proposer.propose_one("synthetic packet", context=P1_CONTEXT,
                                          initial_family="binomial")
    assert [model.expression for model in models] == ["b0+b1*z"] and record["status"] == "parsed"
    entry, = record["rejected_candidates"]
    assert (entry["repair_kind"], entry["failure_class"]) == ("empty_body", "parse_format")


@PROVIDERS
def test_a_transport_error_is_no_candidate_record_and_names_no_class(tmp_path, provider):
    """Check that a transport error is no candidate record and names no class."""
    proposer, _ = _proposer(tmp_path, provider, [RuntimeError("synthetic transport failure")])
    models, record = proposer.propose_one("synthetic packet", context=P1_CONTEXT,
                                          initial_family="binomial")
    entry, = record["rejected_candidates"]
    assert models == [] and record["status"] == "proposal_failed"
    assert entry["repair_eligible"] is False and "failure_class" not in entry and "repair_kind" not in entry
    assert "failure_class" not in _written(tmp_path, provider)["rejected_candidates"][0]
    # It stays counted where it always was, and never as a failure class.
    transports = [{"transport": record}]
    assert core.wave_event_counts([], transports, 1)["failed_calls"] == 1
    assert core.failure_class_counts([], transports) == ZERO


@PROVIDERS
def test_both_proposers_class_and_declare_each_array_rejection(tmp_path, provider):
    """Check that both proposers class and declare each array rejection."""
    good = {"expression": "b0+b1*z", "family": "binomial", "parameters": PAIR}
    reserved = {"expression": "b0+exp*z", "family": "binomial",
                "parameters": {"b0": {"role": "intercept"}, "exp": "signed", "rho": "shape"}}
    malformed = {"expression": "b0", "family": "binomial", "parameters": ["b0"]}
    proposer, _ = _proposer(tmp_path, provider, [[good, reserved, malformed]])
    models, record = proposer.propose_one("synthetic packet", context=P1_CONTEXT,
                                          initial_family="binomial", proposal_limit=3)
    assert [model.expression for model in models] == ["b0+b1*z"]
    assert record["rejected_candidates"] == [
        {"index": 1, "expression": "b0+exp*z", "family": "binomial", "reason": RESERVED_NAME_REFUSAL,
         "parameters": reserved["parameters"], "failure_class": "reserved_name"},
        {"index": 2, "expression": "b0", "family": "binomial",
         "reason": "parameters must map coefficient names to fixed roles",
         "parameters": None, "failure_class": "parse_format"}]
    assert _written(tmp_path, provider)["rejected_candidates"] == record["rejected_candidates"]


@PROVIDERS
def test_both_p4_transports_class_and_declare_each_rejection(tmp_path, provider):
    """Check that both p4 transports class and declare each rejection."""
    value = {"expression": "b0", "family": "binomial", "parameters": {"b0": 1}}
    count = {"expression": "b0", "family": "binomial", "parameters": {}}
    proposer, _ = _proposer(tmp_path, provider, [[value, count, "not an object"]])
    context = {"run_id": "synthetic-pcx", "world_seed": 11, "cell": "111", "proposal_rep": 0,
               "event": 5}
    public = [{"u": 0.1, "y": 3} for _ in range(9)]
    models, record = proposer.propose(public, [h.as_dict() for h in initial_hypotheses()],
                                      context=context, initial_family="binomial")
    assert models == [] and record["status"] == "parsed"
    assert [(entry["index"], entry["parameters"], entry["failure_class"])
            for entry in record["rejected_candidates"]] == [
        (0, {"b0": 1}, "parse_format"), (1, {}, "coefficient_count"), (2, None, "parse_format")]
    written = _written(tmp_path, provider, "event_05")
    assert written["rejected_candidates"] == record["rejected_candidates"]


# --------------------------------------------------------------------------
# 3. the three runners: every record carries its class, the count is the recount
# --------------------------------------------------------------------------


def declare(expression, family, **extra):
    """A reply declaring every b<k> its expression uses (b0 the intercept), plus `extra`."""
    names = sorted(set(re.findall(r"\bb\d\b", expression)))
    return {"expression": expression, "family": family,
            "parameters": {**{name: "intercept" if name == "b0" else "signed" for name in names},
                           **extra}}


def judged(reply):
    """The role mapping the engine judges: role objects normalized, rho and pi dropped."""
    return {name: role["role"] if isinstance(role, dict) else role
            for name, role in reply["parameters"].items() if name not in ("rho", "pi")}


#: (round, burst_wave, burst_call_index, attempt) -> the class its refusal records.
TRANSPORT_CLASSES = {(0, 0, 2, 0): "coefficient_count", (0, 0, 2, 1): "reserved_name",
                     (0, 0, 3, 0): "parse_format", (0, 0, 3, 1): "prior_probe",
                     (1, None, None, 0): "parse_format"}
TRANSPORT_ERRORS = {(0, 0, 5, 0)}
FORMS = {
    "core": {"family": "binomial", "A": "b0+b1*z", "F": "b0+b1*u", "P": "b0+b1*sqrt(u)",
             "reserved": "b0+exp*z",
             "fillers": ["b0+b1*z+b2*square(z)", "b0+b1*tanh(3*z)", "b0+b1*exp(z)",
                         "b0+b1*z+b2*u", "b0+b1*log(1+u)"]},
    "boxinggym": {"family": "binomial", "A": "b0+b1*tz", "F": "b0+b1*t01", "P": "b0+b1*sqrt(t01)",
                  "reserved": "b0+exp*tz",
                  "fillers": ["b0+b1*tz+b2*square(tz)", "b0+b1*tanh(3*tz)", "b0+b1*tz+b2*tanh(3*tz)",
                              "b0+b1*log(1+t01)", "b0+b1*tz+b2*exp(t01)"]},
    "chembench": {"family": "lognormal", "A": "b0+b1*tanh(f0z)", "F": "b0+b1*tanh(f1z)",
                  "P": "b0+b1*tanh(f2z)", "reserved": "b0+exp*f0z",
                  "fillers": [f"b0+b1*tanh(f0z+{index}/20)" for index in range(1, 6)]},
}


def plan_for(forms):
    """The scripted replies of one runner's burst (K = 6) and its round-1 call.

    Wave 0: A, A again (a duplicate), a seven-coefficient reply repaired into a reserved name,
    invalid JSON repaired into a candidate the prior probe refuses, a candidate whose fit fails,
    and a transport error. Wave 1 asks for the five missing and gets five fillers (the first
    declared with a role object). Round 1: an empty body repaired into A again.
    """
    family, fillers = forms["family"], forms["fillers"]
    first_filler = declare(fillers[0], family)
    first_filler["parameters"]["b0"] = {"role": "intercept"}
    return {
        (0, 0, 0, 0): declare(forms["A"], family, rho="shape"),
        (0, 0, 1, 0): declare(forms["A"], family),
        (0, 0, 2, 0): {"expression": forms["A"], "family": family, "parameters": SEVEN},
        (0, 0, 2, 1): {"expression": forms["reserved"], "family": family,
                       "parameters": {"b0": "intercept", "exp": "signed"}},
        (0, 0, 3, 0): "not json",
        (0, 0, 3, 1): declare(forms["P"], family),
        (0, 0, 4, 0): declare(forms["F"], family),
        (0, 0, 5, 0): RuntimeError("synthetic transport failure"),
        (0, 1, 0, 0): first_filler,
        **{(0, 1, index, 0): declare(form, family) for index, form in enumerate(fillers) if index},
        (1, None, None, 0): "",
        (1, None, None, 1): declare(forms["A"], family),
    }


def keyed_client(monkeypatch, plan):
    """An injected Anthropic transport whose reply is chosen by the call's own identity and
    attempt, never by arrival order, so parallel calls get their scripted replies whatever the
    thread schedule. A repair turn is any request past the first user message."""
    local, lock, seen = threading.local(), threading.Lock(), []
    original = cp.AnthropicCountProposer

    def create(**body):
        """Test helper `create`: stand-in or builder used by the surrounding tests."""
        context = local.context
        key = (context["round"], context.get("burst_wave"), context.get("burst_call_index"),
               0 if len(body["messages"]) == 1 else 1)
        with lock:
            seen.append(key)
        reply = plan[key]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            content=[SimpleNamespace(type="text",
                                     text=reply if isinstance(reply, str) else json.dumps(reply))],
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 5, "output_tokens": 2}),
            id="synthetic-pcx", model="synthetic-model", stop_reason="end_turn")

    def factory(records, **kwargs):
        """Test helper `factory`: stand-in or builder used by the surrounding tests."""
        instance = original(records, client=SimpleNamespace(messages=SimpleNamespace(create=create)))
        propose = instance.propose_one

        def call(packet_text, **options):
            """Test helper `call`: stand-in or builder used by the surrounding tests."""
            local.context = options["context"]
            return propose(packet_text, **options)

        instance.propose_one = call
        return instance

    monkeypatch.setattr(cp, "AnthropicCountProposer", factory)
    return seen


def refuse_prior(monkeypatch, module, name, expression):
    """Test helper `refuse_prior`: stand-in or builder used by the surrounding tests."""
    original = getattr(module, name)

    def probe(hypothesis, *args, **kwargs):
        """Test helper `probe`: stand-in or builder used by the surrounding tests."""
        record, predictive = original(hypothesis, *args, **kwargs)
        if hypothesis.expression == expression:
            record = {**record, "passed": False, "detail": "synthetic prior-probe refusal"}
        return record, predictive

    monkeypatch.setattr(module, name, probe)


def fail_fit(monkeypatch, module, name, expression, error):
    """Test helper `fail_fit`: stand-in or builder used by the surrounding tests."""
    original = getattr(module, name)

    def fit(hypothesis, *args, **kwargs):
        """Test helper `fit`: stand-in or builder used by the surrounding tests."""
        if getattr(hypothesis, "expression", None) == expression:
            raise error
        return original(hypothesis, *args, **kwargs)

    monkeypatch.setattr(module, name, fit)


def run_core(tmp_path, monkeypatch, forms, **options):
    """Test helper `run_core`: stand-in or builder used by the surrounding tests."""
    from .test_core_parallel_wave import cell_args

    stub_fits(monkeypatch)
    fail_fit(monkeypatch, core, "fit", forms["F"],
             CountInferenceFailure("synthetic_numeric", "sampling", "synthetic candidate fit failure"))
    refuse_prior(monkeypatch, core, "prior_probe", forms["P"])
    out = tmp_path / "core"
    core.run(cell_args(out, burst_concurrency=2, **options))
    summary = json.loads((out / "summary.json").read_text())
    return out, json.loads((out / "archive.json").read_text()), summary["candidate_failures"], summary


def run_boxinggym(tmp_path, monkeypatch, forms, **options):
    """Test helper `run_boxinggym`: stand-in or builder used by the surrounding tests."""
    from . import boxinggym_core as bg
    from .test_death_kl import bg_run_args

    stub_fits(monkeypatch)
    monkeypatch.setattr(bg, "count_fit", core.fit)
    fail_fit(monkeypatch, bg, "count_fit", forms["F"], FitTimeout(2.0, 1.0))
    refuse_prior(monkeypatch, bg, "prior_probe", forms["P"])
    out = tmp_path / "boxinggym"
    bg.run(bg_run_args(out, cell="100", rounds=1, proposer="anthropic", allow_paid_api=True,
                       llm_model="synthetic-model", burst_concurrency=2, **options))
    summary = json.loads((out / "summary.json").read_text())
    return out, json.loads((out / "archive.json").read_text()), summary["candidate_failures"], summary


def run_chembench(tmp_path, monkeypatch, forms, **options):
    """Test helper `run_chembench`: stand-in or builder used by the surrounding tests."""
    from . import chembench_core as cc
    from .test_chembench_core import _args, _manifest
    from .test_chembench_mda_options import stub_fits as stub_chembench_fits

    stub_chembench_fits(monkeypatch)
    fail_fit(monkeypatch, cc, "fit", forms["F"], FitTimeout(2.0, 1.0))
    refuse_prior(monkeypatch, cc, "_prior_probe", forms["P"])
    # ChemBench keeps its archive in memory (a project decision): capture what it writes.
    rows, archive_entry = [], cc._archive_entry

    def capture(*args, **kwargs):
        """Test helper `capture`: stand-in or builder used by the surrounding tests."""
        row = archive_entry(*args, **kwargs)
        rows.append(row)
        return row

    monkeypatch.setattr(cc, "_archive_entry", capture)
    out = tmp_path / "chembench"
    cc.run(_args(out, _manifest(tmp_path), cell="100", structure_proposals="free", rounds=1,
                 proposer="anthropic", allow_paid_api=True, **options))
    summary = json.loads((out / "summary.json").read_text())
    lines = [json.loads(line) for line in (out / "candidate_failures.jsonl").read_text().splitlines()]
    assert summary["candidate_failures"] == lines
    return out, rows, lines, summary


RUNNERS = {"core": run_core, "boxinggym": run_boxinggym, "chembench": run_chembench}


def proposal_records(out):
    """Test helper `proposal_records`: stand-in or builder used by the surrounding tests."""
    for path in sorted((out / "proposals").glob("t*.json")):
        record = json.loads(path.read_text())
        transports = [record["transport"]] if isinstance(record.get("transport"), dict) else []
        transports += [item["transport"] for item in record.get("transports", [])]
        yield path.stem, record, transports


def recount(out):
    """`failure_class_counts` recomputed from the run's own proposal files: one per classed
    transport entry and one per classed event, nothing else."""
    counts = dict(ZERO)
    for _, record, transports in proposal_records(out):
        for transport in transports:
            for entry in transport.get("rejected_candidates", []):
                if entry.get("failure_class") is not None:
                    counts[entry["failure_class"]] += 1
        for event in record.get("events", []):
            if event["failure_class"] is not None:
                counts[event["failure_class"]] += 1
    return counts


def check_classes(out):
    """Every transport entry and every event carries the class the mapping gives it."""
    classes, statuses = {}, []
    for _, record, transports in proposal_records(out):
        for transport in transports:
            for entry in transport.get("rejected_candidates", []):
                key = (transport["round"], transport.get("burst_wave"),
                       transport.get("burst_call_index"), entry["attempt"])
                if entry["repair_eligible"]:
                    assert entry["failure_class"] == failure_class(
                        error_type=entry["error_type"], reason=entry["reason"], repair_eligible=True)
                    classes[key] = entry["failure_class"]
                else:
                    assert "failure_class" not in entry and key in TRANSPORT_ERRORS
        for event in record.get("events", []):
            assert event["failure_class"] == STATUS_CLASSES[event["status"]]
            statuses.append(event["status"])
    assert classes == TRANSPORT_CLASSES
    # The provider's own write-once records carry the same classes as the runner's copies.
    written = {}
    for path in sorted((out / "llm_transport").glob("*.response.json")):
        transport = json.loads(path.read_text())
        for entry in transport["rejected_candidates"]:
            key = (transport["round"], transport.get("burst_wave"),
                   transport.get("burst_call_index"), entry["attempt"])
            if entry["repair_eligible"]:
                written[key] = entry["failure_class"]
            else:
                assert "failure_class" not in entry and key in TRANSPORT_ERRORS
    assert written == TRANSPORT_CLASSES
    return statuses


@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_every_runner_records_each_class_and_counts_it_once(tmp_path, monkeypatch, request, runner):
    """Check that every runner records each class and counts it once."""
    if runner == "boxinggym":
        request.getfixturevalue("require_boxinggym_source")
    forms = FORMS[runner]
    plan = plan_for(forms)
    seen = keyed_client(monkeypatch, plan)
    out, archive, failures, summary = RUNNERS[runner](
        tmp_path, monkeypatch, forms, initial_proposals=6, initial_catalogue="off")
    assert sorted(seen, key=repr) == sorted(plan, key=repr)
    statuses = check_classes(out)
    fit_status = {"core": "fit_or_predict_failed", "boxinggym": "fit_timeout", "chembench": "fit_failed"}[runner]
    burst = json.loads((out / "proposals" / "t00_burst.json").read_text())
    assert sorted(event["status"] for event in burst["events"]) == sorted(
        ["fitted"] * 6 + ["duplicate_not_refitted", fit_status])
    assert burst["failed_calls"] == 3  # the two refused calls and the transport error
    later = json.loads((out / "proposals" / "t01.json").read_text())
    # Re-pinning change: ChemBench now writes each later round's per-candidate events like
    # the other two runners, so its round-1 duplicate is on disk beside the burst's and counted.
    assert [event["status"] for event in later["events"]] == ["duplicate_not_refitted"]
    assert statuses.count("duplicate_not_refitted") == 2
    expected = {**ZERO, "parse_format": 2, "reserved_name": 1, "coefficient_count": 1,
                "prior_probe": 1, "fit_failed": 1, "duplicate": 2}  # Re-pinning change: every runner
    counts = summary["proposal_accounting"]["failure_class_counts"]
    assert counts == recount(out) == expected
    assert all(counts[kind] >= 1 for kind in FAILURE_CLASSES)
    # candidate_failures: a candidate status names its class; a proposal_failed call does not -
    # its classes are on its transport entries.
    assert failures
    for entry in failures:
        if entry.get("status") == "proposal_failed":
            assert "failure_class" not in entry
        else:
            assert entry["failure_class"] == STATUS_CLASSES[entry["status"]] == "fit_failed"
    # Archive parameters: every archive entry that names a candidate records the mapping the engine judged.
    replies = [reply for reply in plan.values() if isinstance(reply, dict)
               and reply["parameters"] is not SEVEN and "exp" not in reply["parameters"]]
    declared = {reply["expression"]: judged(reply) for reply in replies}
    assert declared[forms["A"]] == {"b0": "intercept", "b1": "signed"}  # rho is not a coefficient
    named = [entry for entry in archive if entry.get("expression")]
    assert {"active", "duplicate", "rejected", "fit_failed"} <= {entry["outcome"] for entry in named}
    for entry in named:
        assert entry["parameters"] == declared[entry["expression"]], entry
    assert all(entry["parameters"] is None for entry in archive if not entry.get("expression"))
    if runner == "core":
        # The round-1 empty body is a whole-reply failure projected as a transport_failure row.
        assert [entry["outcome"] for entry in archive if not entry.get("expression")] == [
            "transport_failure"]
    # The packets never see either field.
    for path in sorted((out / "packets").glob("*.json")):
        for entry in json.loads(path.read_text())["archive"]:
            assert "parameters" not in entry and "failure_class" not in entry


@pytest.mark.parametrize("runner", ["core", "boxinggym"])
def test_later_parallel_rounds_class_their_events_and_count_them(tmp_path, monkeypatch, request, runner):
    """Check that later parallel rounds class their events and count them."""
    if runner == "boxinggym":
        request.getfixturevalue("require_boxinggym_source")
    forms = FORMS[runner]
    fresh, other = forms["fillers"][1], forms["fillers"][2]  # neither is a catalogue member
    family = forms["family"]
    plan = {(1, 0, 0, 0): declare(fresh, family), (1, 0, 1, 0): declare(fresh, family),
            (1, 1, 0, 0): declare(other, family)}
    keyed_client(monkeypatch, plan)
    out, archive, _, summary = RUNNERS[runner](tmp_path, monkeypatch, forms, parallel_proposals=2)
    record = json.loads((out / "proposals" / "t01.json").read_text())
    assert [(event["status"], event["failure_class"]) for event in record["events"]] == [
        ("fitted", None), ("duplicate_not_refitted", "duplicate"), ("fitted", None)]
    assert summary["proposal_accounting"]["failure_class_counts"] == recount(out) == {
        **ZERO, "duplicate": 1}
    # Catalogue members and proposals alike (b0 the intercept, every other b<k> signed).
    assert {entry["expression"] for entry in archive} >= {fresh, other}
    assert len(archive) > 3
    for entry in archive:
        assert entry["parameters"] == declare(entry["expression"], family)["parameters"], entry


def _time_out_when(monkeypatch, module, name, predicate):
    original = getattr(module, name)

    def fit(hypothesis, *args, **kwargs):
        """Test helper `fit`: stand-in or builder used by the surrounding tests."""
        if predicate(hypothesis, args):
            raise FitTimeout(2.0, 1.0)
        return original(hypothesis, *args, **kwargs)

    monkeypatch.setattr(module, name, fit)


@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_a_non_candidate_fit_timeout_names_its_class_but_is_not_counted(tmp_path, monkeypatch,
                                                                       request, runner):
    """A model that was never a proposal - an active CORE or BoxingGym model refit in round 1,
    a ChemBench catalogue member at round 0 - that times out is a candidate_failures record
    naming fit_failed and an archive entry with its parameters, but no transport entry or event,
    so the count leaves it out."""
    rows = []
    if runner == "core":
        stub_fits(monkeypatch)
        _time_out_when(monkeypatch, core, "fit",
                       lambda h, args: h.expression == "b0+b1*z" and len(args[1]) > 4)
        out = tmp_path / "core"
        core.run(run_args(out, rounds=2, seed=31, cell="100"))
        archive = json.loads((out / "archive.json").read_text())
    elif runner == "boxinggym":
        request.getfixturevalue("require_boxinggym_source")
        from . import boxinggym_core as bg
        from .test_death_kl import bg_run_args

        stub_fits(monkeypatch)
        monkeypatch.setattr(bg, "count_fit", core.fit)
        monkeypatch.setattr(bg, "loglik_prior_probe", lambda *args, **kwargs: {"passed": True})
        _time_out_when(monkeypatch, bg, "count_fit",
                       lambda h, args: h.expression == "b0+b1*tz" and len(args[1]) > 4)
        out = tmp_path / "boxinggym"
        # The guarded path is the one that retires an active model instead of failing the run.
        bg.run(bg_run_args(out, cell="100", rounds=2, fit_guards="v1"))
        archive = json.loads((out / "archive.json").read_text())
    else:
        from . import chembench_core as cc
        from .test_chembench_core import _args, _manifest
        from .test_chembench_mda_options import stub_fits as stub_chembench_fits

        stub_chembench_fits(monkeypatch)
        _time_out_when(monkeypatch, cc, "fit",
                       lambda h, args: not hasattr(h, "expression") and len(h.features) == 1)
        archive_entry = cc._archive_entry

        def capture(*args, **kwargs):
            """Test helper `capture`: stand-in or builder used by the surrounding tests."""
            row = archive_entry(*args, **kwargs)
            rows.append(row)
            return row

        monkeypatch.setattr(cc, "_archive_entry", capture)
        out = tmp_path / "chembench"
        cc.run(_args(out, _manifest(tmp_path), cell="100", structure_proposals="free", rounds=1))
        archive = rows
    summary = json.loads((out / "summary.json").read_text())
    timed_out = [entry for entry in summary["candidate_failures"] if entry.get("status") == "fit_timeout"]
    assert len(timed_out) == 1 and timed_out[0]["failure_class"] == "fit_failed"
    if runner == "chembench":
        lines = [json.loads(line) for line in (out / "candidate_failures.jsonl").read_text().splitlines()]
        assert lines == summary["candidate_failures"]
    if runner == "core":
        # The refit ledger keeps its own shape; only the candidate_failures copy names a class.
        assert timed_out[0]["scope"] == "active"
        assert all("failure_class" not in record for record in summary["refit_failed_models"])
    failed = [entry for entry in archive if entry["outcome"] == "fit_failed"]
    assert len(failed) == 1 and failed[0]["parameters"] == {"b0": "intercept", "b1": "signed"}
    counts = summary["proposal_accounting"]["failure_class_counts"]
    assert counts == recount(out) and counts["fit_failed"] == 0


# --------------------------------------------------------------------------
# 4. archive parameters: projection, packets, and the pool reader
# --------------------------------------------------------------------------


def test_a_projected_array_rejection_carries_its_declaration_and_a_reply_failure_none():
    """Check that a projected array rejection carries its declaration and a reply failure none."""
    declared = {"b0": {"role": "intercept"}, "exp": "signed", "rho": "shape"}
    record = {"repairs": 1, "rejected_candidates": [
        {"attempt": 0, "reason": "reply is not valid JSON: synthetic", "error_type": "ValueError",
         "http_status": None, "repair_eligible": True, "repair_kind": "malformed",
         "failure_class": "parse_format"},
        cp.candidate_rejection(1, {"expression": "b0+exp*z", "family": "binomial",
                                   "parameters": declared}, RESERVED_NAME_REFUSAL),
        cp.candidate_rejection(2, "not an object", "reply must be one JSON object"),
    ]}
    archive = {}
    core.project_transport_rejections(archive, record, [], 3)
    row, = archive.values()
    assert row["outcome"] == "rejected" and row["expression"] == "b0+exp*z"
    assert row["parameters"] == declared
    rows = record["transport_failure_rows"]
    assert len(rows) == 2 and all(item["parameters"] is None for item in rows)
    # The transport entries themselves keep their class and declaration beside the projection keys.
    assert [entry["failure_class"] for entry in record["rejected_candidates"]] == [
        "parse_format", "reserved_name", "parse_format"]
    assert record["rejected_candidates"][1]["parameters"] == declared


@PROVIDERS
def test_a_p4_run_projects_the_rejected_declaration_into_its_archive_row(tmp_path, monkeypatch, provider):
    """Check that a p4 run projects the rejected declaration into its archive row."""
    stub_fits(monkeypatch)
    bad = {"expression": "b0 + b1*z", "family": "binomial",
           "parameters": {"b0": "not_a_role", "b1": "signed"}}
    good = {"expression": "b0", "family": "binomial", "parameters": {"b0": "intercept"}}
    client = Client(provider, [response(provider, [bad, good])])
    name = "AnthropicCountProposer" if provider == "anthropic" else "OpenAICountProposer"
    original = getattr(cp, name)
    monkeypatch.setattr(cp, name, lambda records, **kwargs: original(records, client=client))
    projected, project = [], core.project_transport_rejections

    def capture(archive, *args):
        """Test helper `capture`: stand-in or builder used by the surrounding tests."""
        project(archive, *args)
        projected.extend(copy.deepcopy(row) for row in archive.values() if row["outcome"] == "rejected")

    monkeypatch.setattr(core, "project_transport_rejections", capture)
    out = tmp_path / "cell"
    core.run(run_args(out, seed=31, rounds=5, cell="100", proposer=provider, allow_paid_api=True,
                      proposal_policy="P4"))
    row, = projected
    assert row["parameters"] == bad["parameters"] and row["rejection_kind"] == "engine_rejection"
    record = json.loads((out / "proposals" / "t05.json").read_text())
    entry, = record["transport"]["rejected_candidates"]
    assert (entry["failure_class"], entry["parameters"]) == ("parse_format", bad["parameters"])
    assert all(event["failure_class"] == STATUS_CLASSES[event["status"]] for event in record["events"])
    # P4 keeps its summary shape: no accounting container, so no failure_class_counts.
    assert "proposal_accounting" not in json.loads((out / "summary.json").read_text())


def test_a_packet_built_from_an_archive_carrying_parameters_is_byte_identical():
    """Check that a packet built from an archive carrying parameters is byte identical."""
    observations = [{"obs_id": index, "round": 0, "u": 0.2 * (index + 1), "y": 3} for index in range(4)]
    constraints = cell_constraints("100", "binomial")
    plain = [
        {"model_id": "m1", "expression": "b0+b1*z", "family": "binomial", "proposed_round": 0,
         "outcome": "rejected", "reason": {"validation_message": "synthetic"}, "note": "",
         "source": "initial_burst"},
        {"model_id": "m2", "expression": "b0", "family": "binomial", "proposed_round": 0,
         "outcome": "duplicate", "reason": {"matched_model_id": "m3", "matched_expression": "b0"}},
        {"model_id": "m4", "expression": "b0+b1*u", "family": "binomial", "proposed_round": 0,
         "outcome": "fit_failed", "reason": {"error": "synthetic"}},
        {"model_id": "m5", "expression": "b0+b1*tanh(3*z)", "family": "binomial",
         "proposed_round": 0, "outcome": "active", "reason": {"admitted_round": 0, "active_rounds": [0]}},
    ]
    carrying = [{**entry, "parameters": {"b0": "intercept", "b1": "signed"}} for entry in plain]
    left = serialize_packet(build_belief_packet(observations, [], plain, constraints, core_interface(), 0))
    right = serialize_packet(build_belief_packet(observations, [], carrying, constraints, core_interface(), 0))
    assert left.encode() == right.encode()
    assert all("parameters" not in entry for entry in json.loads(right)["archive"])


def test_the_pool_reader_accepts_burst_events_that_carry_a_failure_class(tmp_path):
    """Check that the pool reader accepts burst events that carry a failure class."""
    from .test_initial_pool_file import burst

    payload = {**burst(), "initial_burst_refill_policy": INITIAL_BURST_REFILL_POLICY}
    for event in payload["events"]:
        event["failure_class"] = None if event["status"] == "fitted" else "parse_format"
    duplicate = {**payload["events"][0], "status": "duplicate_not_refitted", "failure_class": "duplicate"}
    payload["events"].append(duplicate)
    path = tmp_path / "synthetic_pool.json"
    path.write_text(json.dumps(payload) + "\n")
    pool = read_initial_pool_file(path, cell="100", initial_family="binomial")
    assert pool.source == "file_burst"
    assert [h.model_id for h in pool.hypotheses] == payload["after_ids"]


def test_the_pool_reader_accepts_the_burst_file_the_writer_now_produces(tmp_path, monkeypatch):
    """Check that the pool reader accepts the burst file the writer now produces."""
    forms = FORMS["core"]
    keyed_client(monkeypatch, plan_for(forms))
    out, *_ = run_core(tmp_path, monkeypatch, forms, initial_proposals=6, initial_catalogue="off")
    record = json.loads((out / "proposals" / "t00_burst.json").read_text())
    assert all("failure_class" in event for event in record["events"])
    pool = read_initial_pool_file(out / "proposals" / "t00_burst.json", cell="100",
                                  initial_family="binomial")
    assert [h.model_id for h in pool.hypotheses] == record["after_ids"]
    assert len(pool.hypotheses) == 6
