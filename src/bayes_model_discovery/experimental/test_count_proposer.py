"""Transport acceptance with an injected client; no credentials or network."""

import csv
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from . import count_proposer
from .count_models import initial_hypotheses
from .count_proposer import (
    AnthropicCountProposer,
    OpenAICountProposer,
    mask_secrets,
    parse_candidate_object,
    parse_candidates,
)


def item(family="beta_binomial"):
    """Return a valid candidate payload dict used as the base for test replies."""
    return {"expression": "b0", "family": family, "parameters": {"b0": "intercept"}}


def fixture(tmp_path, text=None, error=None, stop_reason="end_turn"):
    """Return the shared fixture used by the tests in this module."""
    calls = []

    def create(**body):
        """Fake client entry point: record the request and return the canned reply."""
        calls.append(body)
        if error is not None:
            raise error
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text=json.dumps([item()]) if text is None else text)
            ],
            usage=SimpleNamespace(
                model_dump=lambda **kwargs: {"input_tokens": 123, "output_tokens": 45}
            ),
            id="test-only-response",
            model="returned-test-model",
            stop_reason=stop_reason,
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    proposer = AnthropicCountProposer(tmp_path / "transport", client=client)
    context = {
        "run_id": "test-only-run",
        "world_seed": 11,
        "cell": "010",
        "proposal_rep": 0,
        "event": 5,
    }
    public = [{"u": 0.1, "y": 3} for _ in range(9)]
    pool = [h.as_dict() for h in initial_hypotheses()]
    return proposer, calls, context, public, pool


def openai_fixture(tmp_path, text=None, error=None, finish_reason="stop"):
    """Injected OpenAI client; no credentials, no network. Mirrors `fixture`
    above for the Anthropic transport - see OpenAICountProposer.propose. The
    fake response mimics the real openai>=1.40 SDK shape:
    choices[0].message.content, usage.prompt_tokens/completion_tokens/
    prompt_tokens_details.cached_tokens, id."""
    calls = []

    def create(**body):
        """Fake client entry point: record the request and return the canned reply."""
        calls.append(body)
        if error is not None:
            raise error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps([item()]) if text is None else text
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=123,
                completion_tokens=45,
                prompt_tokens_details=SimpleNamespace(cached_tokens=7),
            ),
            id="test-only-response",
            model="returned-test-model",
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    proposer = OpenAICountProposer(tmp_path / "transport_openai", client=client)
    context = {
        "run_id": "test-only-run",
        "world_seed": 11,
        "cell": "010",
        "proposal_rep": 0,
        "event": 5,
    }
    public = [{"u": 0.1, "y": 3} for _ in range(9)]
    pool = [h.as_dict() for h in initial_hypotheses()]
    return proposer, calls, context, public, pool


def test_explicit_enable_and_s0f0_refuse_before_client_call(tmp_path):
    """Check that explicit enable and s0f0 refuse before client call."""
    with pytest.raises(ValueError, match="live API disabled"):
        AnthropicCountProposer(tmp_path / "disabled")
    assert not (tmp_path / "disabled").exists()
    proposer, calls, context, public, pool = fixture(tmp_path)
    with pytest.raises(ValueError, match="forbids proposal"):
        proposer.propose(
            public, pool, context={**context, "cell": "000"}, initial_family="binomial"
        )
    assert calls == []
    with pytest.raises(ValueError, match="CORE support"):
        proposer.propose(
            [*public[:-1], {"u": 0.1, "y": 0.5}], pool, context=context, initial_family="binomial"
        )
    assert calls == []


def test_request_body_usage_identity_and_no_temperature_or_provider_seed(tmp_path):
    """Check that request body usage identity and no temperature or provider seed."""
    proposer, calls, context, public, pool = fixture(tmp_path)
    models, record = proposer.propose(public, pool, context=context, initial_family="binomial")
    assert len(calls) == 1 and len(models) == 1 and record["status"] == "parsed"
    assert set(calls[0]) == {"model", "max_tokens", "system", "messages"}
    assert calls[0]["model"] == "claude-opus-4-7" and calls[0]["max_tokens"] == 1600
    prompt = json.loads(calls[0]["messages"][0]["content"])
    assert prompt["observations"] == public and "evaluation" not in prompt
    assert record["returned_model"] == "returned-test-model"
    assert record["usage"]["input_tokens"] == 123 and record["client_injected"]
    assert record["decoding"]["temperature"] == "not_sent/provider_default"
    for key, value in context.items():
        assert record[key] == value
    with pytest.raises(FileExistsError):
        proposer.propose(public, pool, context=context, initial_family="binomial")
    assert len(calls) == 1  # No hidden retry or overwrite of the event identity.


@pytest.mark.parametrize(
    "text",
    [
        "not json at all {",
        json.dumps({"not": "a list"}),
        json.dumps([item()] * 4),
    ],
    ids=["unparseable_json", "top_level_not_a_list", "more_than_three_items"],
)
def test_invalid_json_contract_is_recorded_without_mock_fallback(tmp_path, text):
    """These three are the ONLY batch-level failures left;
    a malformed or grammar-broken individual candidate no longer nukes the whole
    batch - see test_shape_and_grammar_violations_reject_only_that_candidate."""
    proposer, calls, context, public, pool = fixture(tmp_path, text)
    models, record = proposer.propose(public, pool, context=context, initial_family="binomial")
    assert models == [] and record["status"] == "proposal_failed"
    assert len(calls) == 1 and proposer.attempts == 1 and not record["silent_fallback"]
    assert (tmp_path / "transport/event_05.response.json").is_file()


def test_shape_and_grammar_violations_reject_only_that_candidate():
    """A candidate missing a required key, or one whose
    expression fails the AST grammar, used to raise and fail the whole batch
    (see git history of this test); now it is recorded in rejected_candidates
    and parse_candidates returns normally with that candidate simply absent
    from `models`."""
    missing_key = [{k: v for k, v in item().items() if k != "parameters"}]
    models, ignored = parse_candidates(json.dumps(missing_key), "111", "binomial")
    assert models == []
    assert ignored["rejected_candidates"] == [
        {
            "index": 0,
            "expression": "b0",
            "family": "beta_binomial",
            "reason": "proposal candidate must include expression/family/parameters",
            # The entry records the item's `parameters` (None when it
            # declares none) and its failure class.
            "parameters": None,
            "failure_class": "parse_format",
        }
    ]

    bad_grammar = [{**item(), "expression": "__import__('os')"}]
    models, ignored = parse_candidates(json.dumps(bad_grammar), "111", "binomial")
    assert models == []
    assert ignored["rejected_candidates"] == [
        {
            "index": 0,
            "expression": "__import__('os')",
            "family": "beta_binomial",
            "reason": "call outside expression grammar",
            # The declared parameters verbatim and the failure class.
            "parameters": {"b0": "intercept"},
            "failure_class": "parse_format",
        }
    ]


def test_cell_restrictions_stop_reason_and_transport_error_are_explicit(tmp_path):
    # F0/S0 cell-constraint violations reject only that candidate (see
    # test_f0_family_violation_rejects_only_that_candidate and
    # test_s0_structure_violation_rejects_only_that_candidate for mixed
    # good+bad batches); with a single-candidate batch that leaves models==[]
    # and one rejected_candidates entry, no raise.
    """Check that cell restrictions stop reason and transport error are explicit."""
    models, ignored = parse_candidates(json.dumps([item()]), "100", "binomial")
    assert models == [] and len(ignored["rejected_candidates"]) == 1
    assert "initial likelihood" in ignored["rejected_candidates"][0]["reason"]

    models, ignored = parse_candidates(json.dumps([item()]), "010", "binomial")
    assert models[0].family == "beta_binomial"
    assert ignored == {
        "ignored_parameters": [{}],
        "ignored_keys": [{}],
        # One flag per accepted candidate, False when the
        # values already arrived as role strings (see
        # test_p4_object_parameter_values_are_normalized_and_recorded).
        "normalized_parameter_shape": [False],
        "rejected_candidates": [],
    }

    models, ignored = parse_candidates(
        json.dumps([{**item(), "expression": "b0+tanh(u)"}]), "010", "binomial"
    )
    assert models == [] and len(ignored["rejected_candidates"]) == 1
    assert "S0 requires" in ignored["rejected_candidates"][0]["reason"]

    for tag, opts in (
        ("truncated", {"stop_reason": "max_tokens"}),
        ("failed", {"error": RuntimeError("not exposed")}),
    ):
        root = tmp_path / tag
        root.mkdir()
        proposer, calls, context, public, pool = fixture(root, **opts)
        models, record = proposer.propose(public, pool, context=context, initial_family="binomial")
        assert models == [] and record["status"] == "proposal_failed"
        assert len(calls) == 1 and record["sdk_retries"] == 0
        assert "not exposed" not in json.dumps(record)


def test_core_event_reaches_transport_with_injected_client_and_stub_fits(tmp_path, monkeypatch):
    """Check that core event reaches transport with injected client and stub fits."""
    from . import core_count, count_proposer
    from .count_inference import Fit

    args = run_args(
        tmp_path / "core", proposer="anthropic", proposal_policy="P4", proposal_limit=3
    )
    with pytest.raises(ValueError, match="allow-paid-api"):
        core_count.run(args)
    assert not (tmp_path / "core").exists()
    template, calls, _, _, _ = fixture(tmp_path)
    monkeypatch.setattr(
        count_proposer,
        "AnthropicCountProposer",
        lambda records, **kwargs: AnthropicCountProposer(records, client=template.client),
    )

    # The runner passes the CORE interface and trial count through.
    def fake_fit(hypothesis, u, y, seed, settings, interface=None, n_trials=50):
        """Stub fit function that replaces the real SMC fit in tests."""
        particles = {name: np.zeros(8) for name, _ in hypothesis.parameters}
        particles.update({name: np.full(8, 0.1) for name in hypothesis.noise_parameters})
        return Fit(
            hypothesis,
            particles,
            -float(len(y)),
            [-float(len(y))] * 2,
            {"observation_count": len(y), "seed": seed},
        )

    monkeypatch.setattr(core_count, "fit", fake_fit)
    args.allow_paid_api = True  # Only the injected client is reachable in this test.
    core_count.run(args)
    assert len(calls) == 1
    result = json.loads((tmp_path / "core/proposals/t05.json").read_text())
    summary = json.loads((tmp_path / "core/summary.json").read_text())
    assert not result["mock"] and result["transport"]["client_injected"]
    assert result["transport"]["status"] == "parsed" and result["fit_success_count"] == 1
    assert result["transport"]["ignored_parameters"] == [{}]
    assert result["transport"]["ignored_keys"] == [{}]
    assert summary["api_request_attempts"] == 1 and summary["admissions"] == 1
    assert summary["paid_api_calls"] == 1  # actual transport count, never None


# --- Fail-soft contract: reserved variance names / extra keys are
# ignored and recorded rather than rejected; paid_api_calls is 0 for mock and
# the real transport count for anthropic (never None). ---------------------


def test_rho_in_parameters_is_ignored_and_recorded_not_rejected():
    """Check that rho in parameters is ignored and recorded not rejected."""
    payload = [
        {
            "expression": "b0+b1*z",
            "family": "beta_binomial",
            "parameters": {"b0": "intercept", "b1": "signed", "rho": "rho"},
        }
    ]
    models, ignored = parse_candidates(json.dumps(payload), "111", "binomial")
    assert len(models) == 1
    assert models[0].family == "beta_binomial"
    assert dict(models[0].parameters) == {"b0": "intercept", "b1": "signed"}
    assert ignored["ignored_parameters"] == [{"rho": "rho"}]
    assert ignored["ignored_keys"] == [{}]


def test_pi_in_parameters_is_ignored_and_recorded_not_rejected():
    """Check that pi in parameters is ignored and recorded not rejected."""
    payload = [
        {
            "expression": "b0+b1*z",
            "family": "zi_binomial",
            "parameters": {"b0": "intercept", "b1": "signed", "pi": "pi"},
        }
    ]
    models, ignored = parse_candidates(json.dumps(payload), "111", "binomial")
    assert len(models) == 1
    assert models[0].family == "zi_binomial"
    assert dict(models[0].parameters) == {"b0": "intercept", "b1": "signed"}
    assert ignored["ignored_parameters"] == [{"pi": "pi"}]
    assert ignored["ignored_keys"] == [{}]


def test_extra_candidate_key_is_ignored_and_recorded_not_rejected():
    """Check that extra candidate key is ignored and recorded not rejected."""
    payload = [
        {
            "expression": "b0+b1*z",
            "family": "beta_binomial",
            "parameters": {"b0": "intercept", "b1": "signed", "rho": "rho"},
            "noise_parameters": ["rho"],
        }
    ]
    models, ignored = parse_candidates(json.dumps(payload), "111", "binomial")
    assert len(models) == 1
    assert dict(models[0].parameters) == {"b0": "intercept", "b1": "signed"}
    assert ignored["ignored_parameters"] == [{"rho": "rho"}]
    assert ignored["ignored_keys"] == [{"noise_parameters": ["rho"]}]


def test_ignored_fields_are_per_candidate_and_empty_when_nothing_dropped():
    """Check that ignored fields are per candidate and empty when nothing dropped."""
    payload = [
        item(),  # clean candidate: nothing to ignore
        {**item(), "family": "zi_binomial", "parameters": {"b0": "intercept", "pi": "pi"}},
    ]
    models, ignored = parse_candidates(json.dumps(payload), "111", "binomial")
    assert len(models) == 2
    assert ignored["ignored_parameters"] == [{}, {"pi": "pi"}]
    assert ignored["ignored_keys"] == [{}, {}]


def test_role_violation_rejects_only_that_candidate():
    """The rho/pi carve-out is scoped to reserved variance names only; a role
    outside the fixed 4-role policy is still not admitted - but it
    rejects only that candidate rather than raising for the whole batch."""
    bogus_role = [{**item(), "parameters": {"b0": "intercept", "b1": "nonsense_role"}}]
    models, ignored = parse_candidates(json.dumps(bogus_role), "111", "binomial")
    assert models == []
    assert ignored["rejected_candidates"] == [
        {
            "index": 0,
            "expression": "b0",
            "family": "beta_binomial",
            "reason": "coefficient role outside fixed prior policy",
            # The declared parameters verbatim and the failure class.
            "parameters": {"b0": "intercept", "b1": "nonsense_role"},
            "failure_class": "parse_format",
        }
    ]


def test_batch_failures_still_hard():
    """Only these three remain batch-level: JSON not
    parseable, top-level not a list, more than three items."""
    with pytest.raises(json.JSONDecodeError):
        parse_candidates("not json at all {", "111", "binomial")
    with pytest.raises(ValueError, match="at most three models"):
        parse_candidates(json.dumps({"not": "a list"}), "111", "binomial")
    too_many_candidates = [item()] * 4
    with pytest.raises(ValueError, match="at most three models"):
        parse_candidates(json.dumps(too_many_candidates), "111", "binomial")


def test_one_bad_candidate_keeps_the_rest():
    """A 3-candidate batch with exactly one violation in the middle position:
    both good candidates are admitted (in order), only the bad one appears in
    rejected_candidates (at its original index), and models counts only the
    accepted ones (see also test_valid_candidate_count_counts_only_accepted_
    candidates for the transport-level valid_candidate_count field)."""
    good_a = item()  # "b0", beta_binomial, {"b0": "intercept"}
    bad = {**item(), "parameters": {"b0": "nonsense_role"}}  # role outside fixed policy
    good_b = {**item(), "expression": "b0+tanh(u)"}  # tanh(u) uses no extra coefficient
    payload = [good_a, bad, good_b]
    models, ignored = parse_candidates(json.dumps(payload), "111", "binomial")
    assert len(models) == 2
    assert [m.expression for m in models] == ["b0", "b0+tanh(u)"]
    assert ignored["ignored_parameters"] == [{}, {}]
    assert ignored["ignored_keys"] == [{}, {}]
    assert len(ignored["rejected_candidates"]) == 1
    rejected = ignored["rejected_candidates"][0]
    assert rejected["index"] == 1
    assert rejected["reason"] == "coefficient role outside fixed prior policy"


def test_s0_structure_violation_rejects_only_that_candidate():
    """Mixed batch on an S0 cell (010, seen live in cells 010/011):
    the in-pool candidate
    is admitted, the out-of-pool one is rejected alone - the whole point of
    the per-candidate check (the earlier batch-level check rejected both)."""
    payload = [item(), {**item(), "expression": "b0+tanh(u)"}]
    models, ignored = parse_candidates(json.dumps(payload), "010", "binomial")
    assert len(models) == 1 and models[0].expression == "b0"
    assert len(ignored["rejected_candidates"]) == 1
    rejected = ignored["rejected_candidates"][0]
    assert rejected["index"] == 1 and rejected["expression"] == "b0+tanh(u)"
    assert "S0 requires" in rejected["reason"]


def test_f0_family_violation_rejects_only_that_candidate():
    """Mixed batch on an F0 cell (100/101): the
    initial-family candidate is admitted, the expanded-family one is rejected
    alone."""
    payload = [{**item(), "family": "binomial"}, {**item(), "family": "zi_binomial"}]
    models, ignored = parse_candidates(json.dumps(payload), "100", "binomial")
    assert len(models) == 1 and models[0].family == "binomial"
    assert len(ignored["rejected_candidates"]) == 1
    rejected = ignored["rejected_candidates"][0]
    assert rejected["index"] == 1 and rejected["family"] == "zi_binomial"
    assert "initial likelihood" in rejected["reason"]


def test_rejected_candidates_shape():
    """Exact shape: a list of {index, expression, family, reason, parameters,
    failure_class} dicts, one per rejected candidate, no other keys."""
    payload = [{**item(), "parameters": {"b0": "nonsense_role"}}]
    _, ignored = parse_candidates(json.dumps(payload), "111", "binomial")
    assert len(ignored["rejected_candidates"]) == 1
    entry = ignored["rejected_candidates"][0]
    # The entry gains `parameters` (the item's own, verbatim) and
    # `failure_class`; the key set stays exact.
    assert set(entry) == {"index", "expression", "family", "reason", "parameters", "failure_class"}
    assert entry["index"] == 0
    assert entry["expression"] == "b0" and entry["family"] == "beta_binomial"
    assert isinstance(entry["reason"], str) and entry["reason"]
    assert entry["parameters"] == {"b0": "nonsense_role"}
    assert entry["failure_class"] == "parse_format"


def test_valid_candidate_count_counts_only_accepted_candidates(tmp_path):
    """valid_candidate_count (recorded on the transport record by
    AnthropicCountProposer.propose) must count accepted candidates only, not
    the raw payload length."""
    payload = [item(), {**item(), "parameters": {"b0": "nonsense_role"}}]
    proposer, calls, context, public, pool = fixture(tmp_path, json.dumps(payload))
    models, record = proposer.propose(public, pool, context=context, initial_family="binomial")
    assert len(calls) == 1
    assert len(payload) == 2 and len(models) == 1
    assert record["status"] == "parsed"
    assert record["valid_candidate_count"] == 1
    assert len(record["rejected_candidates"]) == 1


def test_paid_api_calls_zero_for_mock_full_run(tmp_path, monkeypatch):
    """Check that paid api calls zero for mock full run."""
    from . import core_count
    from .count_inference import Fit

    # The runner passes the CORE interface and trial count through.
    def fake_fit(hypothesis, u, y, seed, settings, interface=None, n_trials=50):
        """Stub fit function that replaces the real SMC fit in tests."""
        particles = {name: np.zeros(8) for name, _ in hypothesis.parameters}
        particles.update({name: np.full(8, 0.1) for name in hypothesis.noise_parameters})
        return Fit(
            hypothesis,
            particles,
            -float(len(y)),
            [-float(len(y))] * 2,
            {"observation_count": len(y), "seed": seed},
        )

    monkeypatch.setattr(core_count, "fit", fake_fit)
    core_count.run(run_args(tmp_path / "mockrun", rounds=5))
    summary = json.loads((tmp_path / "mockrun/summary.json").read_text())
    config = json.loads((tmp_path / "mockrun/config.json").read_text())
    assert summary["paid_api_calls"] == 0
    assert summary["api_request_attempts"] == 0
    assert config["paid_api_calls"] == 0


# --- policy P1: one candidate per round, one repair ----------


def one_object(expression="b0+b1*z", family="beta_binomial", **extra):
    """Return a single-candidate reply object for the one-candidate round tests."""
    return json.dumps(
        {
            "expression": expression,
            "family": family,
            "parameters": {"b0": "intercept", "b1": "signed"},
            **extra,
        }
    )


def p1_fixture(tmp_path, texts, cell="111"):
    """Injected client that replies with `texts` in order; no credentials, no network."""
    calls, replies = [], list(texts)

    def create(**body):
        """Fake client entry point: record the request and return the canned reply."""
        calls.append(body)
        text = replies[min(len(calls) - 1, len(replies) - 1)]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 9, "output_tokens": 3}),
            id=f"test-only-{len(calls)}",
            model="returned-test-model",
            stop_reason="end_turn",
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    proposer = AnthropicCountProposer(tmp_path / f"transport_{cell}", client=client)
    context = {
        "run_id": "test-only-run",
        "world_seed": 11,
        "cell": cell,
        "proposal_rep": 0,
        "round": 7,
    }
    return proposer, calls, context


def p1_openai_fixture(tmp_path, texts, cell="111"):
    """Injected OpenAI client that replies with `texts` in order; mirrors
    `p1_fixture` above for the Anthropic transport - no credentials, no
    network."""
    calls, replies = [], list(texts)

    def create(**body):
        """Fake client entry point: record the request and return the canned reply."""
        calls.append(body)
        text = replies[min(len(calls) - 1, len(replies) - 1)]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=9,
                completion_tokens=3,
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
            id=f"test-only-openai-{len(calls)}",
            model="returned-test-model",
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    proposer = OpenAICountProposer(tmp_path / f"transport_openai_{cell}", client=client)
    context = {
        "run_id": "test-only-run",
        "world_seed": 11,
        "cell": cell,
        "proposal_rep": 0,
        "round": 7,
    }
    return proposer, calls, context


def test_parse_candidate_object_accepts_one_object_and_records_the_note():
    """Check that parse candidate object accepts one object and records the note."""
    hypothesis, ignored = parse_candidate_object(
        one_object(note="residuals curve upward at large u"), "111", "binomial"
    )
    assert hypothesis.expression == "b0+b1*z" and hypothesis.family == "beta_binomial"
    assert ignored["note"] == "residuals curve upward at large u"
    assert ignored["ignored_keys"] == {} and ignored["ignored_parameters"] == {}
    assert parse_candidate_object(one_object(), "111", "binomial")[1]["note"] == ""


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json at all", "not valid JSON"),
        ('[{"expression": "b0"}]', "not an array"),
        ('"a string"', "one JSON object"),
        ('{"expression": "b0"}', "expression/family/parameters"),
        ('{"expression": "b0", "family": "binomial", "parameters": []}', "coefficient names"),
        (one_object("b0+b1*sin(u)"), "call outside expression grammar"),
        (one_object("b0", "poisson"), "unsupported count family"),
        (one_object("b0+b1*z", "binomial", parameters={"b0": "wrong_role"}), "role outside"),
    ],
)
def test_parse_candidate_object_reports_the_validation_message(text, message):
    """Check that parse candidate object reports the validation message."""
    with pytest.raises(ValueError, match=message):
        parse_candidate_object(text, "111", "binomial")


def test_parse_candidate_object_enforces_the_cell_like_the_batch_path():
    """Check that parse candidate object enforces the cell like the batch path."""
    with pytest.raises(ValueError, match="F0 requires the initial likelihood family"):
        parse_candidate_object(one_object("b0+b1*z", "zi_binomial"), "100", "binomial")
    with pytest.raises(ValueError, match="S0 requires a structure from the initial pool"):
        parse_candidate_object(one_object("b0+b1*tanh(u)", "beta_binomial"), "010", "binomial")
    hypothesis, _ = parse_candidate_object(one_object("b0+b1*z", "beta_binomial"), "010", "binomial")
    assert hypothesis.family == "beta_binomial"


def test_parse_candidate_object_accepts_a_task_specific_allowed_family():
    """No test above passes `allowed_families`, so CORE's own
    three-family menu applies and "poisson" would be rejected (see the
    "unsupported count family" case in test_parse_candidate_object_reports_
    the_validation_message). A caller for a different task (boxinggym's
    peregrines) supplies its own menu instead, and a family outside CORE's
    three but inside THAT menu is admitted - `Hypothesis.__post_init__` itself
    now checks the wider `IMPLEMENTED_FAMILIES` (count_models.py), and this is
    the task-level gate layered on top of it."""
    hypothesis, _ = parse_candidate_object(
        one_object("b0+b1*z", "poisson"),
        "111",
        "poisson",
        allowed_families=("poisson", "negative_binomial"),
    )
    assert hypothesis.family == "poisson"


def test_parse_candidate_object_narrows_to_a_task_specific_allowed_family():
    """The same argument also NARROWS: "binomial" is CORE's own initial family
    and would be accepted with no `allowed_families` argument at all (see
    test_parse_candidate_object_accepts_one_object_and_records_the_note), but a
    caller whose task menu does not include it still rejects it - with the
    identical literal reason a wholly-unrecognized family gets."""
    with pytest.raises(ValueError, match="unsupported count family: binomial"):
        parse_candidate_object(
            one_object("b0+b1*z", "binomial"), "111", "poisson", allowed_families=("poisson",)
        )


def test_parse_candidates_batch_respects_allowed_families():
    """Same widen/narrow contract, for the P4 batch path (count_proposer.
    AnthropicCountProposer.propose never passes this argument - it always uses
    CORE's own three, see count_proposer.py - so this is a direct check of the
    plumbing `parse_candidates` itself offers, not a claim any current caller
    exercises it)."""
    payload = [
        {"expression": "b0", "family": "poisson", "parameters": {"b0": "intercept"}},
        {"expression": "b0+b1*z", "family": "binomial", "parameters": {"b0": "intercept", "b1": "signed"}},
    ]
    models, ignored = parse_candidates(
        json.dumps(payload), "111", "poisson", allowed_families=("poisson",)
    )
    assert [m.family for m in models] == ["poisson"]
    assert len(ignored["rejected_candidates"]) == 1
    assert ignored["rejected_candidates"][0]["reason"] == "unsupported count family: binomial"


def test_note_and_unknown_keys_are_recorded_not_rejected():
    """Check that note and unknown keys are recorded not rejected."""
    long_note = "x" * 201
    _, ignored = parse_candidate_object(one_object(note=long_note), "111", "binomial")
    assert ignored["note"] == "" and ignored["ignored_keys"] == {"note": long_note}
    _, ignored = parse_candidate_object(one_object(note=17), "111", "binomial")
    assert ignored["note"] == "" and ignored["ignored_keys"] == {"note": 17}
    # Forward compatibility: a future `priors` block must never be a hard failure.
    _, ignored = parse_candidate_object(
        one_object(priors={"b1": "Normal(0,3)"}), "111", "binomial"
    )
    assert ignored["ignored_keys"] == {"priors": {"b1": "Normal(0,3)"}}
    _, ignored = parse_candidate_object(
        one_object(note="ok", parameters={"b0": "intercept", "b1": "signed", "rho": "shape"}),
        "111",
        "binomial",
    )
    assert ignored["ignored_parameters"] == {"rho": "shape"} and ignored["note"] == "ok"


def test_propose_one_makes_exactly_one_call_when_the_reply_is_valid(tmp_path):
    """Check that propose one makes exactly one call when the reply is valid."""
    proposer, calls, context = p1_fixture(tmp_path, [one_object(note="try overdispersion")])
    models, record = proposer.propose_one(
        '{"round":7}', context=context, initial_family="binomial"
    )
    assert len(calls) == 1 and len(models) == 1
    assert record["status"] == "parsed" and record["calls"] == 1 and record["repairs"] == 0
    assert record["rejected_candidates"] == [] and record["note"] == "try overdispersion"
    assert record["valid_candidate_count"] == 1 and proposer.attempts == 1
    assert set(calls[0]) == {"model", "max_tokens", "system", "messages"}
    assert calls[0]["system"] == count_proposer.PROMPT_FILE.read_text()
    assert calls[0]["messages"] == [{"role": "user", "content": '{"round":7}'}]


def test_propose_one_repairs_once_after_a_malformed_reply(tmp_path):
    """Malformed then valid gives exactly 2 calls and 1 repair."""
    proposer, calls, context = p1_fixture(tmp_path, ["{oops", one_object()])
    models, record = proposer.propose_one(
        '{"round":7}', context=context, initial_family="binomial"
    )
    assert len(calls) == 2 and len(models) == 1 and proposer.attempts == 2
    assert record["calls"] == 2 and record["repairs"] == 1 and record["status"] == "parsed"
    assert len(record["rejected_candidates"]) == 1
    assert record["rejected_candidates"][0]["attempt"] == 0
    assert "not valid JSON" in record["rejected_candidates"][0]["reason"]
    roles = [message["role"] for message in calls[1]["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert "{oops" == calls[1]["messages"][1]["content"]
    assert "not valid JSON" in calls[1]["messages"][2]["content"]
    assert calls[1]["system"] == calls[0]["system"]


def test_propose_one_stops_after_the_second_failure_with_both_messages(tmp_path):
    """A second failure is proposal_failed, not a third call."""
    proposer, calls, context = p1_fixture(tmp_path, ["{oops", one_object("b0+b1*sin(u)")])
    models, record = proposer.propose_one(
        '{"round":7}', context=context, initial_family="binomial"
    )
    assert len(calls) == 2 and models == [] and proposer.attempts == 2
    assert record["status"] == "proposal_failed"
    assert record["calls"] == 2 and record["repairs"] == 1
    reasons = [item["reason"] for item in record["rejected_candidates"]]
    assert len(reasons) == 2
    assert "not valid JSON" in reasons[0] and "call outside expression grammar" in reasons[1]
    assert [item["attempt"] for item in record["rejected_candidates"]] == [0, 1]
    stored = json.loads((tmp_path / "transport_111" / "round_07.response.json").read_text())
    assert stored["rejected_candidates"] == record["rejected_candidates"]
    assert len(json.loads((tmp_path / "transport_111" / "round_07.request.json").read_text())["bodies"]) == 2


def test_propose_one_records_prompt_identity_and_packet_digest(tmp_path):
    """Prompt version and sha, plus the packet hash, are on every record."""
    proposer, _, context = p1_fixture(tmp_path, [one_object()])
    packet_text = '{"interface":{"link":"logit"},"round":7}'
    _, record = proposer.propose_one(packet_text, context=context, initial_family="binomial")
    identity = count_proposer.prompt_identity()
    assert record["prompt_version"] == "v2.0" == identity["prompt_version"]
    assert record["prompt_sha256"] == identity["prompt_sha256"]
    assert record["prompt_sha256"] == hashlib.sha256(
        count_proposer.PROMPT_FILE.read_bytes()
    ).hexdigest()
    assert record["packet_bytes"] == len(packet_text.encode())
    assert record["packet_sha256"] == hashlib.sha256(packet_text.encode()).hexdigest()
    assert record["policy"] == "P1" and record["round"] == 7


def test_propose_one_refuses_a_bad_identity_before_calling(tmp_path):
    """Check that propose one refuses a bad identity before calling."""
    proposer, calls, context = p1_fixture(tmp_path, [one_object()], cell="000")
    with pytest.raises(ValueError, match="forbids proposal"):
        proposer.propose_one('{"round":7}', context=context, initial_family="binomial")
    with pytest.raises(ValueError, match="identity required"):
        proposer.propose_one(
            '{"round":7}', context={**context, "round": 21}, initial_family="binomial"
        )
    with pytest.raises(ValueError, match="belief packet text required"):
        proposer.propose_one(
            "", context={**context, "cell": "111"}, initial_family="binomial"
        )
    assert calls == []


# --- policy P1 runner contracts (stub fits; no SMC, no API) ---------------------------


def run_args(output, **overrides):
    """The full argparse namespace core_count.run expects, with that experiment phase's defaults."""
    return SimpleNamespace(
        **{
            "output": str(output),
            "profile": "dev",
            "rounds": 5,
            "seed": 11,
            "replicate": 0,
            "cell": "111",
            "structure": "out",
            "truth_family": "beta_binomial",
            "initial_family": "binomial",
            "proposer": "mock",
            "llm_model": "claude-opus-4-7",
            "allow_paid_api": False,
            "proposal_policy": "P1",
            "proposal_rounds": "",
            "proposal_limit": 1,
            "repair_attempts": 1,
            "archive_reevaluation_rounds": "",
            "design_policy": "joint_eig",  # the runner's design-policy arm
            # The runner's two prior-ablation switches
            "prior_template": "default",
            "priors": "off",
            **overrides,
        }
    )


def stub_fits(monkeypatch, sleeper=None, wake_at=8):
    """Deterministic stand-in for SMC: evidence is a fixed function of the model.

    `sleeper` names an expression whose evidence is terrible until the public
    dataset reaches `wake_at` rows and excellent afterwards, which is what makes
    a retired model worth resurrecting.
    """
    from . import core_count
    from .count_inference import Fit

    def fake_fit(hypothesis, u, y, seed, settings, interface=None, n_trials=50):
        # The runner now passes the CORE interface and its trial count
        # through to count_inference.fit; the stub accepts and ignores them.
        """Stub fit function that replaces the real SMC fit in tests."""
        rng = np.random.default_rng(seed % 2**32)
        particles = {name: rng.normal(0.3, 0.4, 8) for name, _ in hypothesis.parameters}
        particles.update({name: rng.beta(2, 8, 8) for name in hypothesis.noise_parameters})
        offset = (int(hypothesis.model_id[:6], 16) % 500) / 100.0
        if sleeper is not None and hypothesis.expression == sleeper:
            offset = 0.0 if len(y) >= wake_at else 50.0
        return Fit(
            hypothesis,
            particles,
            -float(len(y)) - offset,
            [-float(len(y)) - offset] * 2,
            {"observation_count": len(y), "seed": seed},
        )

    monkeypatch.setattr(core_count, "fit", fake_fit)
    return core_count


def test_p1_config_scores_and_points_carry_the_new_contract(tmp_path, monkeypatch):
    """Policy, prompt identity, scoring phases, packet hashes."""
    core_count = stub_fits(monkeypatch)
    core_count.run(run_args(tmp_path / "p1", rounds=3))
    out = tmp_path / "p1"
    config = json.loads((out / "config.json").read_text())
    assert config["proposal_policy"] == "P1"
    assert config["proposal_rounds"] == list(range(1, 21))
    assert config["proposal_limit"] == 1 and config["repair_attempts"] == 1
    assert config["prompt_version"] == "v2.0"
    assert config["prompt_sha256"] == count_proposer.prompt_identity()["prompt_sha256"]
    assert config["archive_reevaluation"] == "off"
    assert config["archive_reevaluation_rounds"] == [] and config["points_variant"] == ""
    assert config["belief_packet"] == "v2" and config["interface"]["link"] == "logit"
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    assert [row["scoring_phase"] for row in rows] == [
        "initial",
        *["pre_proposal", "post_proposal"] * 3,
    ]
    assert all("post_equals_pre" in row for row in rows[1:])
    assert all("one_step_log_score" in row for row in rows if row["scoring_phase"] == "pre_proposal")
    assert not any("one_step_log_score" in row for row in rows if row["phase"] == "after_proposal")
    for index, row in enumerate(rows):
        if row.get("post_equals_pre"):
            previous = rows[index - 1]
            assert row["bma"] == previous["bma"] and row["single"] == previous["single"]
            assert row["per_model_nll"] == previous["per_model_nll"]
            assert row["predictions_from"] == previous["phase"]
            # A pure re-sort of the same fits still copies, but the
            # order-dependent bookkeeping is refreshed from the current pool.
            assert sorted(row["active_ids"]) == sorted(previous["active_ids"])
            assert sum(row["model_weights"]) == pytest.approx(1.0)
    packets = sorted((out / "packets").glob("*.json"))
    assert [path.name for path in packets] == ["t01.json", "t02.json", "t03.json"]
    transport = json.loads((out / "proposals/t03.json").read_text())["transport"]
    assert transport["packet_sha256"] == hashlib.sha256(packets[-1].read_bytes()).hexdigest()
    assert transport["packet_bytes"] == len(packets[-1].read_bytes())
    assert transport["prompt_version"] == "v2.0" and transport["calls"] == 0
    body = json.loads(packets[-1].read_text())
    assert body["round"] == 3 and body["n_public"] == 7
    assert {item["outcome"] for item in body["archive"]} <= {
        "active",
        "retired_by_evidence",
        "duplicate",
        "rejected",
        "fit_failed",
    }
    points = (out / "points.csv").read_text().splitlines()
    # The plotting contract: the emitter writes the 25 declared columns in that exact order. Parsed
    # with `csv`, not `str.split(",")` - `source_path` embeds a comma
    # (`(round=N,phase=...)`) and is therefore quoted, so a naive split
    # mis-aligns every column after it.
    assert points[0].split(",") == list(core_count.CONTRACT_POINTS_COLUMNS)
    assert points[0].split(",")[20:] == ["variant", "code_sha", "brier", "nmsle", "rmsle"]
    assert len(points) == len(rows) + 1
    written = list(csv.DictReader(points))
    assert len(written) == len(rows)
    # `variant` is empty on a default run, and `brier`/`nmsle`/`rmsle` are
    # undefined on the CORE path - blank, never 0.
    for line in written:
        assert [line[c] for c in ("variant", "brier", "nmsle", "rmsle")] == ["", "", "", ""]
        assert len(line["code_sha"]) == 64
    assert json.loads((out / "summary.json").read_text())["proposal_accounting"]["llm_calls"] == 0


def test_p1_records_one_step_scores_as_log_mean_exp_over_particles(tmp_path, monkeypatch):
    """Log of the mean predictive probability, not the mean log."""
    from .count_models import particle_logpmf

    core_count = stub_fits(monkeypatch)
    core_count.run(run_args(tmp_path / "onestep", rounds=2))
    out = tmp_path / "onestep"
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    observations = [
        json.loads(line) for line in (out / "observations.jsonl").read_text().splitlines()
    ]
    second = observations[5]
    pre = next(row for row in rows if row["round"] == 2 and row["scoring_phase"] == "pre_proposal")
    for model_id, value in pre["one_step_log_score"].items():
        meta = json.loads((out / "fits" / f"t01_{model_id}.json").read_text())
        with np.load(out / "fits" / f"t01_{model_id}.npz") as data:
            particles = {name: data[name] for name in data.files}
        hypothesis = core_count.Hypothesis(
            meta["expression"], meta["family"], tuple(meta["parameters"].items())
        )
        table = particle_logpmf(hypothesis, particles, np.asarray([second["u"]]))
        column = table[:, 0, int(second["y"])]
        expected = float(np.log(np.exp(column).mean()))
        assert value == pytest.approx(expected, rel=1e-9)
        assert value != pytest.approx(float(column.mean()), rel=1e-6)
    packet = json.loads((out / "packets/t02.json").read_text())
    born = {item["model_id"]: item["born_round"] for item in packet["pool"]}
    for item in packet["pool"]:
        assert item["one_step_rounds"] == [1, 2]
        if born[item["model_id"]] == 1:
            assert item["one_step_log_scores"][0] is None


def test_p1_fits_carry_the_prior_predictive_record(tmp_path, monkeypatch):
    """Check that p1 fits carry the prior predictive record."""
    core_count = stub_fits(monkeypatch)
    core_count.run(run_args(tmp_path / "prior", rounds=1))
    for path in sorted((tmp_path / "prior/fits").glob("*.json")):
        record = json.loads(path.read_text())["prior_predictive"]
        assert record["n_samples"] == 64 and record["passed"]
        assert record["n_prior_samples"] == 64 and len(record["grid"]) == 16
        assert all(len(record[key]) == 16 for key in ("mean", "q05", "q50", "q95"))
    events = json.loads((tmp_path / "prior/proposals/t01.json").read_text())["events"]
    assert events and all(event["prior_probe"]["passed"] for event in events)
    assert all(event["note"] == "" for event in events)


def test_archive_reevaluation_off_writes_nothing_and_on_resurrects(tmp_path, monkeypatch):
    """The switch is inert when off and records a resurrection when on."""
    core_count = stub_fits(monkeypatch, sleeper="b0", wake_at=8)
    monkeypatch.setattr(core_count, "ACTIVE_MAXIMUM", 4)
    core_count.run(run_args(tmp_path / "off", rounds=4))
    off_summary = json.loads((tmp_path / "off/summary.json").read_text())
    assert "resurrection_events" not in off_summary
    assert not (tmp_path / "off/resurrection").exists()
    assert off_summary["proposal_accounting"]["reevaluation_refits"] == 0
    assert json.loads((tmp_path / "off/config.json").read_text())["archive_reevaluation"] == "off"

    core_count.run(run_args(tmp_path / "on", rounds=4, archive_reevaluation_rounds="4"))
    config = json.loads((tmp_path / "on/config.json").read_text())
    assert config["archive_reevaluation"] == "t4"
    assert config["archive_reevaluation_rounds"] == [4] and config["points_variant"] == "rev4"
    summary = json.loads((tmp_path / "on/summary.json").read_text())
    event = summary["resurrection_events"][0]
    assert event["round"] == 4 and event["refit_count"] >= 1
    assert event["refit_basis"] == "original_prior_and_all_public_observations"
    assert event["resurrected_ids"] and not event["post_equals_pre"]
    assert set(event["resurrected_ids"]) <= set(event["refit_ids"])
    assert event["pre_scores"]["phase"] == "before_proposal"
    assert event["post_scores"]["phase"] == "after_reevaluation"
    assert event["pre_scores"]["bma"]["nll"] != event["post_scores"]["bma"]["nll"]
    assert json.loads((tmp_path / "on/resurrection/t04.json").read_text()) == event
    rows = [json.loads(line) for line in (tmp_path / "on/scores.jsonl").read_text().splitlines()]
    assert "post_reevaluation" in [row.get("scoring_phase") for row in rows]
    points = (tmp_path / "on/points.csv").read_text().splitlines()
    # The plotting contract: `variant` is column 21 of 25, not the last column, and
    # the quoted `source_path` before it rules out a naive comma split.
    assert all(row["variant"] == "rev4" for row in csv.DictReader(points))
    archive = json.loads((tmp_path / "on/archive.json").read_text())
    sleeper = next(item for item in archive if item["expression"] == "b0")
    assert sleeper["outcome"] == "active"
    assert sleeper["reason"]["admitted_round"] == 4


def test_catalog_exhaustion_skips_the_call_and_is_reported(tmp_path, monkeypatch):
    """S0 has at most eight new pairs; after that no call is made."""
    core_count = stub_fits(monkeypatch)
    core_count.run(run_args(tmp_path / "s0", rounds=10, cell="011"))
    summary = json.loads((tmp_path / "s0/summary.json").read_text())
    assert summary["catalog_exhausted_rounds"] == [9, 10]
    accounting = summary["proposal_accounting"]
    assert accounting["catalog_exhausted"] == 2
    assert accounting["valid_candidates"] == 8 and accounting["fit_attempts"] == 8
    assert accounting["proposal_rounds_executed"] == 10
    transport = json.loads((tmp_path / "s0/proposals/t09.json").read_text())["transport"]
    assert transport["status"] == "catalog_exhausted" and transport["calls"] == 0
    assert transport["catalog_size"] == 12
    assert core_count.candidate_catalog("111", "binomial") is None


def test_p4_policy_keeps_the_legacy_shape(tmp_path, monkeypatch):
    """The frozen anchor: no scoring_phase, no packet, four proposal events."""
    core_count = stub_fits(monkeypatch)
    core_count.run(run_args(tmp_path / "p4", rounds=6, proposal_policy="P4", proposal_limit=3))
    out = tmp_path / "p4"
    config = json.loads((out / "config.json").read_text())
    assert config["proposal_rounds"] == [5, 10, 15, 20] and config["proposal_limit"] == 3
    assert config["belief_packet"] == "none" and config["repair_attempts"] == 0
    assert not (out / "packets").exists()
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    assert not any("scoring_phase" in row or "post_equals_pre" in row for row in rows)
    assert not any("one_step_log_score" in row for row in rows)
    assert [path.name for path in sorted((out / "proposals").glob("*.json"))] == ["t05.json"]
    assert len(json.loads((out / "proposals/t05.json").read_text())["events"]) == 3
    summary = json.loads((out / "summary.json").read_text())
    assert "proposal_accounting" not in summary and "resurrection_events" not in summary


def test_same_round_duplicate_gets_a_collision_safe_archive_key(tmp_path, monkeypatch):
    """Two (or more) same-round duplicates of one model_id
    must not collide in the archive - entry count equals event count, the
    first entry survives untouched, and only the second (third, ...) gets a
    `_k{n}` suffix. Injected proposer: round 1 admits one candidate, round 2
    repeats its model_id three times in a single proposal batch (the P4/
    proposal_limit>1 scenario, driven here through
    policy P1 with proposal_limit=3 so archive.json - P1-only - is written)."""
    from .count_models import Hypothesis

    core_count = stub_fits(monkeypatch)
    calls = []
    real_key_fn = core_count.duplicate_archive_key

    def spy(archive, t, key):
        """Wrap the callable and record how it is called."""
        result = real_key_fn(archive, t, key)
        calls.append((archive, t, key, result))
        return result

    monkeypatch.setattr(core_count, "duplicate_archive_key", spy)

    first = Hypothesis(
        "b0+b1*tanh(3*z)", "beta_binomial", (("b0", "intercept"), ("b1", "signed"))
    )

    class FixedProposer:
        """Proposer stub that returns fixed, pre-set replies."""
        def __init__(self):
            self.round = 0
            self.attempts = 0

        def propose_one(self, packet_text, **kwargs):
            """Return the next fixed reply for a single-candidate request."""
            self.round += 1
            self.attempts += 1
            proposals = [first] if self.round == 1 else [first, first, first]
            transport_record = {
                "provider": "none",
                "mock": False,
                "api_request_attempted": True,
                "policy": "P1",
                "status": "parsed",
                "calls": 1,
                "repairs": 0,
                "rejected_candidates": [],
                "note": "",
            }
            return proposals, transport_record

    monkeypatch.setattr(count_proposer, "AnthropicCountProposer", lambda *a, **k: FixedProposer())
    out = tmp_path / "dup"
    core_count.run(
        run_args(out, rounds=2, proposer="anthropic", allow_paid_api=True, proposal_limit=3)
    )
    events = json.loads((out / "proposals/t02.json").read_text())["events"]
    duplicate_events = [e for e in events if e["status"] == "duplicate_not_refitted"]
    # `first` was already admitted (active) after round 1, so all three of its
    # round-2 repeats are duplicates of an already-seen model - none of them
    # is "the first time this round", unlike a model brand new to the run.
    assert len(duplicate_events) == 3
    archive = json.loads((out / "archive.json").read_text())
    duplicate_entries = [
        item
        for item in archive
        if item["outcome"] == "duplicate" and item["model_id"] == first.model_id
    ]
    # Pre-fix this list has length 1: the second and third duplicates' inserts
    # each collide with and overwrite the previous one under the same key.
    assert len(duplicate_entries) == 3
    round2 = [c for c in calls if c[1] == 2]
    assert len(round2) == 3
    base_key = f"duplicate_t02_{first.model_id}"
    assert [c[3] for c in round2] == [base_key, f"{base_key}_k2", f"{base_key}_k3"]
    assert round2[0][0] is round2[1][0] is round2[2][0]  # same dict, mutated in place
    live_archive = round2[0][0]
    assert live_archive[base_key]["outcome"] == "duplicate"
    assert f"{base_key}_k2" in live_archive and f"{base_key}_k3" in live_archive
    summary = json.loads((out / "summary.json").read_text())
    assert summary["proposal_accounting"]["duplicates"] == 3


def test_p1_llm_path_sends_the_packet_and_admits_the_candidate(tmp_path, monkeypatch):
    """End to end with an injected client: packet in, one object out, one repair."""
    from . import count_proposer

    core_count = stub_fits(monkeypatch)
    replies = ["not an object", one_object("b0+b1*tanh(3*z)", "beta_binomial", note="dispersion")]
    proposer, calls, _ = p1_fixture(tmp_path, replies)
    monkeypatch.setattr(
        count_proposer,
        "AnthropicCountProposer",
        lambda records, **kwargs: AnthropicCountProposer(records, client=proposer.client),
    )
    core_count.run(
        run_args(tmp_path / "p1llm", rounds=1, proposer="anthropic", allow_paid_api=True)
    )
    out = tmp_path / "p1llm"
    assert len(calls) == 2
    packet_text = (out / "packets/t01.json").read_text()
    assert calls[0]["messages"][0]["content"] == packet_text
    assert calls[0]["system"] == count_proposer.PROMPT_FILE.read_text()
    body = json.loads(packet_text)
    assert body["interface"] == {
        "outcome_type": "bounded_count",
        "support": [0, 50],
        "trials": 50,
        "inputs": [{"name": "u", "range": [0, 1]}],
        "derived_inputs": {"z": "2*u-1"},
        "link": "logit",
    }
    assert len(body["observations"]) == 5 and len(body["pool"]) == 4
    event = json.loads((out / "proposals/t01.json").read_text())
    transport = event["transport"]
    assert transport["status"] == "parsed" and transport["calls"] == 2
    assert transport["repairs"] == 1 and transport["note"] == "dispersion"
    assert transport["packet_sha256"] == hashlib.sha256(packet_text.encode()).hexdigest()
    assert event["events"][0]["note"] == "dispersion"
    assert event["events"][0]["status"] == "fitted"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["proposal_accounting"]["llm_calls"] == 2
    assert summary["proposal_accounting"]["llm_repairs"] == 1
    assert summary["paid_api_calls"] == 2 and summary["admissions"] == 1
    archived = json.loads((out / "archive.json").read_text())
    admitted = next(item for item in archived if item["expression"] == "b0+b1*tanh(3*z)")
    assert admitted["note"] == "dispersion" and admitted["prior_probe"]["passed"]
    rejected = [item for item in archived if item["outcome"] == "rejected"]
    assert rejected == []
    failures = [item for item in archived if item["outcome"] == "transport_failure"]
    assert len(failures) == 1
    assert failures[0]["rejection_kind"] == "transport_failure"
    assert "not valid JSON" in failures[0]["reason"]["validation_message"]


# --- OpenAI transport, mirroring the Anthropic one ---


def test_openai_refuses_before_client_call_without_allow_live(tmp_path):
    """Check that openai refuses before client call without allow live."""
    with pytest.raises(ValueError, match="live API disabled"):
        OpenAICountProposer(tmp_path / "disabled")
    assert not (tmp_path / "disabled").exists()


def test_openai_request_body_shape_usage_identity_and_no_temperature(tmp_path):
    """Same contract as test_request_body_usage_identity_and_no_temperature_
    or_provider_seed for Anthropic, adjusted for the OpenAI wire shape: SYSTEM
    is the first `messages` entry (role "system"), not a separate field, and
    usage/ids are the normalized {input_tokens, output_tokens,
    cached_tokens} usage plus `completion_id`/`http_request_id` (they replace
    the old `request_id`, which held the completion id)."""
    proposer, calls, context, public, pool = openai_fixture(tmp_path)
    models, record = proposer.propose(public, pool, context=context, initial_family="binomial")
    assert len(calls) == 1 and len(models) == 1 and record["status"] == "parsed"
    assert set(calls[0]) == {"model", "max_tokens", "messages"}
    assert calls[0]["model"] == "gpt-4o" and calls[0]["max_tokens"] == 1600
    assert calls[0]["messages"][0] == {"role": "system", "content": count_proposer.SYSTEM}
    prompt = json.loads(calls[0]["messages"][1]["content"])
    assert prompt["observations"] == public and "evaluation" not in prompt
    assert record["provider"] == "openai" and record["returned_model"] == "returned-test-model"
    assert record["usage"] == {"input_tokens": 123, "output_tokens": 45, "cached_tokens": 7}
    assert record["completion_id"] == "test-only-response" and record["client_injected"]
    assert record["http_request_id"] is None and record["usage_known"]
    assert record["decoding"]["temperature"] == "not_sent/provider_default"
    for key, value in context.items():
        assert record[key] == value
    with pytest.raises(FileExistsError):
        proposer.propose(public, pool, context=context, initial_family="binomial")
    assert len(calls) == 1


def test_openai_propose_one_makes_exactly_one_call_when_the_reply_is_valid(tmp_path):
    """Check that openai propose one makes exactly one call when the reply is valid."""
    proposer, calls, context = p1_openai_fixture(tmp_path, [one_object(note="try overdispersion")])
    models, record = proposer.propose_one(
        '{"round":7}', context=context, initial_family="binomial"
    )
    assert len(calls) == 1 and len(models) == 1
    assert record["status"] == "parsed" and record["calls"] == 1 and record["repairs"] == 0
    assert record["rejected_candidates"] == [] and record["note"] == "try overdispersion"
    assert record["valid_candidate_count"] == 1 and proposer.attempts == 1
    assert set(calls[0]) == {"model", "max_tokens", "messages"}
    assert calls[0]["messages"] == [
        {"role": "system", "content": count_proposer.PROMPT_FILE.read_text()},
        {"role": "user", "content": '{"round":7}'},
    ]
    assert record["provider"] == "openai"
    last_call = {"input_tokens": 9, "output_tokens": 3, "cached_tokens": 0}
    assert record["last_call_usage"] == last_call and record["last_call_usage_known"]
    assert record["usage_total"] == last_call and record["usage_total_complete"]
    assert record["usage_missing_indices"] == []
    assert record["completion_id"] == "test-only-openai-1"
    assert record["http_request_id"] is None


def test_openai_propose_one_repairs_once_after_a_malformed_reply(tmp_path):
    """Malformed then valid gives exactly 2 calls and 1 repair -
    same contract as the Anthropic transport's identically-named test."""
    proposer, calls, context = p1_openai_fixture(tmp_path, ["{oops", one_object()])
    models, record = proposer.propose_one(
        '{"round":7}', context=context, initial_family="binomial"
    )
    assert len(calls) == 2 and len(models) == 1 and proposer.attempts == 2
    assert record["calls"] == 2 and record["repairs"] == 1 and record["status"] == "parsed"
    assert len(record["rejected_candidates"]) == 1
    assert record["rejected_candidates"][0]["attempt"] == 0
    assert "not valid JSON" in record["rejected_candidates"][0]["reason"]
    roles = [message["role"] for message in calls[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "{oops" == calls[1]["messages"][2]["content"]
    assert "not valid JSON" in calls[1]["messages"][3]["content"]
    assert calls[1]["messages"][0] == calls[0]["messages"][0]  # same system entry


def test_openai_propose_one_stops_after_the_second_failure_with_both_messages(tmp_path):
    """Check that openai propose one stops after the second failure with both messages."""
    proposer, calls, context = p1_openai_fixture(tmp_path, ["{oops", one_object("b0+b1*sin(u)")])
    models, record = proposer.propose_one(
        '{"round":7}', context=context, initial_family="binomial"
    )
    assert len(calls) == 2 and models == [] and proposer.attempts == 2
    assert record["status"] == "proposal_failed"
    assert record["calls"] == 2 and record["repairs"] == 1
    reasons = [item["reason"] for item in record["rejected_candidates"]]
    assert len(reasons) == 2
    assert "not valid JSON" in reasons[0] and "call outside expression grammar" in reasons[1]
    assert [item["attempt"] for item in record["rejected_candidates"]] == [0, 1]


def test_openai_core_run_admits_candidate_and_forces_development_label(tmp_path, monkeypatch):
    """Injected-OpenAI admission (P4/event path, mirrors
    test_core_event_reaches_transport_with_injected_client_and_stub_fits) plus
    the runner's label-forcing rule - config.json/summary.json/points.csv all
    read "development" for an openai run even though --profile defaults to
    "dev" already (see the paper-profile edge case in
    test_points_label_forced_to_development_when_provider_is_openai_even_at_
    paper_profile for the case that isolates the OR from --profile alone) -
    and a plain mock run never gets a "label" key in config.json/summary.json
    at all (byte-invariance for the existing paths: the key is added only
    when it is actually being forced)."""
    from . import core_count, count_proposer
    from .count_inference import Fit

    args = run_args(
        tmp_path / "core_openai", proposer="openai", llm_model="gpt-4o",
        proposal_policy="P4", proposal_limit=3,
    )
    with pytest.raises(ValueError, match="allow-paid-api"):
        core_count.run(args)
    assert not (tmp_path / "core_openai").exists()
    template, calls, _, _, _ = openai_fixture(tmp_path)
    monkeypatch.setattr(
        count_proposer,
        "OpenAICountProposer",
        lambda records, **kwargs: OpenAICountProposer(records, client=template.client),
    )

    def fake_fit(hypothesis, u, y, seed, settings, interface=None, n_trials=50):
        """Stub fit function that replaces the real SMC fit in tests."""
        particles = {name: np.zeros(8) for name, _ in hypothesis.parameters}
        particles.update({name: np.full(8, 0.1) for name in hypothesis.noise_parameters})
        return Fit(
            hypothesis,
            particles,
            -float(len(y)),
            [-float(len(y))] * 2,
            {"observation_count": len(y), "seed": seed},
        )

    monkeypatch.setattr(core_count, "fit", fake_fit)
    args.allow_paid_api = True
    core_count.run(args)
    assert len(calls) == 1
    result = json.loads((tmp_path / "core_openai/proposals/t05.json").read_text())
    summary = json.loads((tmp_path / "core_openai/summary.json").read_text())
    config = json.loads((tmp_path / "core_openai/config.json").read_text())
    assert not result["mock"] and result["transport"]["client_injected"]
    assert result["transport"]["status"] == "parsed" and result["fit_success_count"] == 1
    assert result["transport"]["provider"] == "openai"
    assert result["transport"]["usage"] == {
        "input_tokens": 123,
        "output_tokens": 45,
        "cached_tokens": 7,
    }
    assert result["transport"]["completion_id"] == "test-only-response"
    assert summary["api_request_attempts"] == 1 and summary["admissions"] == 1
    assert summary["paid_api_calls"] == 1
    assert config["proposer"] == "openai" and config["label"] == "development"
    assert summary["label"] == "development"
    points = (tmp_path / "core_openai/points.csv").read_text().splitlines()
    header = points[0].split(",")
    label_col = header.index("label")
    assert points[1:] and all(row.split(",")[label_col] == "development" for row in points[1:])

    core_count.run(run_args(tmp_path / "core_mock", rounds=1))
    mock_config = json.loads((tmp_path / "core_mock/config.json").read_text())
    mock_summary = json.loads((tmp_path / "core_mock/summary.json").read_text())
    assert "label" not in mock_config and "label" not in mock_summary


def test_points_label_forced_to_development_when_provider_is_openai_even_at_paper_profile(
    tmp_path,
):
    """Isolates the OR from --profile alone: config["profile"] == "paper"
    would normally read "production" in points.csv (see write_points); the
    proposer being openai forces "development" regardless. A direct
    write_points call, not a full run - profile "paper" numerics are not
    exercised here."""
    from . import core_count

    config = {
        "profile": "paper",
        "proposer": "openai",
        "llm_model": "gpt-4o",
        "cell": "111",
        "seed": 11,
        "structure": "out",
        "truth_family": "beta_binomial",
        "output": str(tmp_path / "paper_openai_run"),
        "source_files_sha256": {"core_count.py": "deadbeef"},
    }
    row = {
        "observation_count": 9,
        "bma": {"nll": 1.0, "coverage90": 0.9, "width90": 1.0, "kl": 0.1},
        "round": 1,
        "phase": "after_proposal",
    }
    path = tmp_path / "points.csv"
    core_count.write_points(path, config, [row], "completed", "")
    rows = list(csv.DictReader(path.read_text().splitlines()))
    assert len(rows) == 1 and rows[0]["label"] == "development"


def test_openai_and_anthropic_propose_one_mask_a_leaked_key_from_the_exception(
    tmp_path, monkeypatch
):
    """An SDK error that happens to echo the live
    key (e.g. a header dump in an auth-failure message) must never reach a
    written record. (a) With the key live in the environment throughout, a
    normal successful call never surfaces it anywhere (baseline). (b) A
    client whose `create()` call itself raises with the key embedded
    (simulating that echo) fails after exactly one call - an API-call-level
    failure never triggers a repair (`raw_text` is still None when the
    except block runs; see AnthropicCountProposer.propose_one/
    OpenAICountProposer.propose_one and the identical existing contract in
    test_cell_restrictions_stop_reason_and_transport_error_are_explicit's
    "failed" case) - and the recorded reason carries the masked placeholder,
    never the raw key. Mirrors the same contract for Anthropic
    (ANTHROPIC_API_KEY) - the shared `mask_secrets` helper is
    provider-agnostic."""
    fake_openai_key = "sk-test-FAKEKEY-21g-openai"
    fake_anthropic_key = "sk-test-FAKEKEY-21g-anthropic"
    monkeypatch.setenv("OPENAI_API_KEY", fake_openai_key)
    monkeypatch.setenv("ANTHROPIC_API_KEY", fake_anthropic_key)
    context = {
        "run_id": "test-only-run",
        "world_seed": 11,
        "cell": "111",
        "proposal_rep": 0,
        "round": 7,
    }

    # (a) succeeds once: baseline with both keys live in the environment.
    ok_proposer, ok_calls, ok_context = p1_openai_fixture(tmp_path, [one_object()])
    ok_models, ok_record = ok_proposer.propose_one(
        '{"round":7}', context=ok_context, initial_family="binomial"
    )
    assert len(ok_calls) == 1 and len(ok_models) == 1 and ok_record["status"] == "parsed"
    assert fake_openai_key not in json.dumps(ok_record)

    # (b) raises with the key embedded in the message.
    def create_openai(**body):
        """Fake OpenAI-style client call returning the canned reply."""
        raise RuntimeError(
            f"upstream auth error; echoed header Authorization: Bearer {fake_openai_key}"
        )

    openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_openai))
    )
    openai_proposer = OpenAICountProposer(tmp_path / "openai_leak", client=openai_client)
    openai_models, openai_record = openai_proposer.propose_one(
        '{"round":7}', context=context, initial_family="binomial"
    )
    assert openai_models == [] and openai_record["status"] == "proposal_failed"
    assert openai_record["calls"] == 1 and openai_record["repairs"] == 0
    dumped_openai = json.dumps(openai_record)
    assert fake_openai_key not in dumped_openai
    assert "<OPENAI_API_KEY:masked>" in dumped_openai
    for path in (tmp_path / "openai_leak").iterdir():
        assert fake_openai_key not in path.read_text()

    def create_anthropic(**body):
        """Fake Anthropic-style client call returning the canned reply."""
        raise RuntimeError(f"upstream auth error; echoed header x-api-key: {fake_anthropic_key}")

    anthropic_client = SimpleNamespace(messages=SimpleNamespace(create=create_anthropic))
    anthropic_proposer = AnthropicCountProposer(tmp_path / "anthropic_leak", client=anthropic_client)
    anthropic_models, anthropic_record = anthropic_proposer.propose_one(
        '{"round":7}', context=context, initial_family="binomial"
    )
    assert anthropic_models == [] and anthropic_record["status"] == "proposal_failed"
    assert anthropic_record["calls"] == 1 and anthropic_record["repairs"] == 0
    dumped_anthropic = json.dumps(anthropic_record)
    assert fake_anthropic_key not in dumped_anthropic
    assert "<ANTHROPIC_API_KEY:masked>" in dumped_anthropic
    for path in (tmp_path / "anthropic_leak").iterdir():
        assert fake_anthropic_key not in path.read_text()


def test_mask_secrets_ignores_unset_or_short_env_values(monkeypatch):
    """Guard for the masking helper itself: an unset or trivially short env
    var value must never turn masking into an unbounded/blank replacement."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert mask_secrets("plain text, nothing to mask") == "plain text, nothing to mask"
    monkeypatch.setenv("OPENAI_API_KEY", "short")  # < 8 chars: not treated as a secret
    assert mask_secrets("contains short verbatim") == "contains short verbatim"
    assert mask_secrets(None) is None


def test_run_never_leaks_the_openai_api_key_across_any_output_file(tmp_path, monkeypatch):
    """Full pipeline: walks every file under a
    completed core_count.run output directory (config.json, summary.json,
    llm_transport/*.json, proposals/*.json, points.csv, packets/*.json, ...)
    and asserts the raw OPENAI_API_KEY value is absent from every one of
    them, while the masked placeholder shows up at least once (proving
    masking actually engaged, not merely that the key was never used). The
    injected client's `create()` raises with the key embedded (simulating an
    SDK error echoing a header) - one call, no repair (see
    test_openai_and_anthropic_propose_one_mask_a_leaked_key_from_the_
    exception); the run itself still completes (a failed proposal round is
    recorded in candidate_failures, not fatal), so every file the runner
    normally writes gets written for this walk to check."""
    from . import count_proposer

    core_count = stub_fits(monkeypatch)
    fake_key = "sk-test-FAKEKEY-21g-fullrun"
    monkeypatch.setenv("OPENAI_API_KEY", fake_key)
    calls = []

    def create(**body):
        """Fake client entry point: record the request and return the canned reply."""
        calls.append(body)
        raise RuntimeError(f"upstream echoed Authorization: Bearer {fake_key}")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(
        count_proposer,
        "OpenAICountProposer",
        lambda records, **kwargs: OpenAICountProposer(records, client=client),
    )
    out = tmp_path / "leak_walk"
    core_count.run(
        run_args(out, rounds=1, proposer="openai", llm_model="gpt-4o", allow_paid_api=True)
    )
    assert len(calls) == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["candidate_failures"] and summary["label"] == "development"
    found_masked = False
    for path in out.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        assert fake_key not in text, f"leaked key in {path}"
        if "<OPENAI_API_KEY:masked>" in text:
            found_masked = True
    assert found_masked


# --- transport hardening ---------
# Every test here uses an injected fake client or an in-process
# httpx.MockTransport; no credential is ever real (the fixtures use obviously
# fake `sk-test-...` values) and no request leaves the process.

import io
import logging
import re
import sys
import traceback

P1_CONTEXT = {
    "run_id": "test-only-run",
    "world_seed": 11,
    "cell": "111",
    "proposal_rep": 0,
    "round": 7,
}


def openai_reply(text, *, index=1, usage=True, finish_reason="stop", http_request_id=None):
    """One fake `chat.completions.create` return value.

    `usage=None` reproduces the case of a perfectly valid JSON
    reply whose `usage` object is absent. `http_request_id` sets the SDK's
    `_request_id` attribute, which newer openai/anthropic SDKs attach to the
    parsed model and older ones do not.
    """
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)],
        usage=(
            SimpleNamespace(
                prompt_tokens=100 + 10 * index,
                completion_tokens=10 + index,
                prompt_tokens_details=SimpleNamespace(cached_tokens=20),
            )
            if usage
            else None
        ),
        id=f"chatcmpl_test_only_{index}",
        model="returned-test-model",
    )
    if http_request_id is not None:
        response._request_id = http_request_id
    return response


def scripted_openai(items):
    """An injected OpenAI-shaped client returning `items` in order; an item
    that is an exception is raised instead of returned."""
    calls = []

    def create(**body):
        """Fake client entry point: record the request and return the canned reply."""
        calls.append(body)
        item = items[min(len(calls) - 1, len(items) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def test_21g2_mask_placeholders_are_exactly_the_two_allowed_forms(monkeypatch):
    """Whatever matched, the replacement is one of
    exactly two placeholders - never a partial display, a length or a prefix.
    Both the env-var values and the bare `sk-`/`sk-ant-` patterns are covered,
    and `sk-ant-` never gets attributed to the OpenAI key."""
    openai_key = "sk-test-openai-FAKEKEY-21g2-abcdefgh"
    anthropic_key = "sk-ant-test-FAKEKEY-21g2-abcdefgh"
    monkeypatch.setenv("OPENAI_API_KEY", openai_key)
    monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic_key)
    text = (
        f"env_openai={openai_key} env_anthropic={anthropic_key} "
        "loose_anthropic=sk-ant-api03-UNRELATEDabcdefgh12 loose_openai=sk-UNRELATEDabcdefgh12"
    )
    masked = mask_secrets(text)
    assert masked == (
        "env_openai=<OPENAI_API_KEY:masked> env_anthropic=<ANTHROPIC_API_KEY:masked> "
        "loose_anthropic=<ANTHROPIC_API_KEY:masked> loose_openai=<OPENAI_API_KEY:masked>"
    )
    assert set(re.findall(r"<[^>]+>", masked)) <= set(count_proposer.MASK_PLACEHOLDERS)
    assert "sk-" not in masked and openai_key[:8] not in masked


def test_21g2_masking_is_recursive_over_containers_and_keys(monkeypatch):
    """`mask_secrets_deep` reaches dict keys, dict values, lists, tuples
    and sets, and leaves every non-string value exactly as it was."""
    fake = "sk-test-FAKEKEY-21g2-recursive"
    monkeypatch.setenv("OPENAI_API_KEY", fake)
    payload = {
        f"header {fake}": [f"a {fake}", {"deep": (f"b {fake}", 3)}],
        "numbers": [1, 2.5, None, True],
        "set": {f"c {fake}"},
    }
    masked = count_proposer.mask_secrets_deep(payload)
    assert fake not in json.dumps(masked, default=sorted)
    masked_key = next(iter(masked))
    assert masked_key == "header <OPENAI_API_KEY:masked>"
    assert masked[masked_key][1]["deep"] == ("b <OPENAI_API_KEY:masked>", 3)
    assert masked["numbers"] == [1, 2.5, None, True]
    assert masked["set"] == {"c <OPENAI_API_KEY:masked>"}


def test_21g2_success_record_masks_raw_text_note_and_ignored_fields(tmp_path, monkeypatch):
    """The "success note echo" counter-example. A VALID reply
    whose note and extra keys carry the credential used to be stored verbatim
    in raw_text/note/ignored_keys - masking only covered `reason`."""
    fake = "sk-test-FAKEKEY-21g2-success-note"
    monkeypatch.setenv("OPENAI_API_KEY", fake)
    reply = json.dumps(
        {
            "expression": "b0+b1*z",
            "family": "beta_binomial",
            "parameters": {"b0": "intercept", "b1": "signed"},
            "note": f"echoed credential {fake}",
            "priors": {f"header {fake}": f"value {fake}"},
        }
    )
    proposer, _calls, context = p1_openai_fixture(tmp_path, [reply])
    models, record = proposer.propose_one(
        '{"round":7}', context=context, initial_family="binomial"
    )
    assert len(models) == 1 and record["status"] == "parsed"
    assert fake not in json.dumps(record)
    assert record["note"] == "echoed credential <OPENAI_API_KEY:masked>"
    assert record["responses"][0]["raw_text"].count("<OPENAI_API_KEY:masked>") == 3
    assert json.dumps(record["ignored_keys"]).count("<OPENAI_API_KEY:masked>") == 2
    for path in proposer.records.iterdir():
        assert fake not in path.read_text()
    saved = (proposer.records / "round_07.response.json").read_text()
    assert saved.count("<OPENAI_API_KEY:masked>") == 6  # raw_text x3, note, ignored_keys x2


def test_21g2_repair_turn_masks_the_assistant_text_and_the_error_before_sending(
    tmp_path, monkeypatch
):
    """The "repair error echo" counter-example. The repair call
    re-sends the model's own reply plus the validation message; both used to go
    on the wire (and into request.json) with the credential intact."""
    fake = "sk-test-FAKEKEY-21g2-repair-turn"
    monkeypatch.setenv("OPENAI_API_KEY", fake)
    leaky = json.dumps(
        {
            "expression": "b0+b1*z",
            "family": fake,
            "parameters": {"b0": "intercept", "b1": "signed"},
        }
    )
    proposer, calls, context = p1_openai_fixture(tmp_path, [leaky, one_object()])
    models, record = proposer.propose_one(
        '{"round":7}', context=context, initial_family="binomial"
    )
    assert len(calls) == 2 and len(models) == 1 and record["repairs"] == 1
    second = json.dumps(calls[1])
    assert fake not in second
    assert second.count("<OPENAI_API_KEY:masked>") >= 2  # echoed reply + validation message
    assert calls[1]["messages"][2]["content"] == leaky.replace(fake, "<OPENAI_API_KEY:masked>")
    assert "<OPENAI_API_KEY:masked>" in calls[1]["messages"][3]["content"]
    assert fake not in json.dumps(record)
    for path in proposer.records.iterdir():
        assert fake not in path.read_text()


def test_21g2_sdk_constructor_exception_is_masked_in_str_repr_and_traceback(
    tmp_path, monkeypatch
):
    """The SDK client constructor sat outside any
    recording try, so an auth/proxy error echoing the credential propagated
    with its str, repr AND formatted traceback intact. It is now built inside a
    try and re-raised masked, and no records directory is created."""
    fake = "sk-test-FAKEKEY-21g2-constructor"
    monkeypatch.setenv("OPENAI_API_KEY", fake)
    openai_module = pytest.importorskip("openai")

    def explode(*args, **kwargs):
        """Fail the test if called: this code path must not run."""
        raise RuntimeError(f"fake SDK constructor error; Authorization: Bearer {fake}")

    monkeypatch.setattr(openai_module, "OpenAI", explode)
    with pytest.raises(count_proposer.MaskedCredentialError) as excinfo:
        OpenAICountProposer(tmp_path / "constructor", allow_live=True)
    caught = excinfo.value
    formatted = "".join(traceback.format_exception(type(caught), caught, caught.__traceback__))
    assert fake not in str(caught) and fake not in repr(caught) and fake not in formatted
    assert "<OPENAI_API_KEY:masked>" in str(caught) and "RuntimeError" in str(caught)
    assert not (tmp_path / "constructor").exists()


def test_21g2_runner_masks_a_proposer_constructor_exception_and_records_it(
    tmp_path, monkeypatch
):
    """The same defect seen from `core_count.run`, which is where it was
    first reproduced. The construction now sits inside a recording try, so
    failure.json is written (masked) and the exception that leaves run() has a
    masked str, repr and traceback. Zero API calls, zero fits."""
    fake = "sk-test-FAKEKEY-21g2-runner-constructor"
    monkeypatch.setenv("OPENAI_API_KEY", fake)
    from . import core_count

    def explode(records, **kwargs):
        """Fail the test if called: this code path must not run."""
        raise RuntimeError(f"fake SDK constructor error; x-api-key: {fake}")

    monkeypatch.setattr(count_proposer, "OpenAICountProposer", explode)
    out = tmp_path / "runner_constructor"
    with pytest.raises(BaseException) as excinfo:
        core_count.run(run_args(out, proposer="openai", llm_model="gpt-4o", allow_paid_api=True))
    caught = excinfo.value
    formatted = "".join(traceback.format_exception(type(caught), caught, caught.__traceback__))
    assert isinstance(caught, count_proposer.MaskedCredentialError)
    assert fake not in str(caught) and fake not in repr(caught) and fake not in formatted
    failure = json.loads((out / "failure.json").read_text())
    assert failure["status"] == "failed" and failure["stage"] == "proposer_construction"
    assert fake not in json.dumps(failure)
    assert "<OPENAI_API_KEY:masked>" in failure["error"]
    assert not (out / "summary.json").exists() and not (out / "points.csv").exists()
    assert list((out / "fits").iterdir()) == []  # the run stopped before any fit
    for path in out.rglob("*"):
        if path.is_file():
            assert fake not in path.read_bytes().decode("utf-8", errors="replace")


def test_21g2_numerical_failure_still_propagates_verbatim(tmp_path, monkeypatch):
    """Masking is for the API transport
    boundary only. An exception with nothing to hide comes back as the SAME
    object, keeping its type and traceback - the numerical propagation rule is
    untouched."""
    plain = ValueError("no finite evidence for any active model")
    assert count_proposer.masked_exception(plain) is plain
    assert not count_proposer.carries_secret(str(plain))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-FAKEKEY-21g2-not-in-this-text")
    assert count_proposer.masked_exception(plain) is plain


def test_21g2_missing_usage_is_unknown_and_never_triggers_a_repair(tmp_path):
    """A valid JSON reply with NO usage object used to
    raise AttributeError inside the broad except and - because raw_text
    already existed - spend a second paid call on a perfectly good reply. It is
    now recorded as unknown, the parse proceeds, and there is exactly one
    call. The first response's identifiers survive in the record."""
    client, calls = scripted_openai([openai_reply(one_object(), usage=False), openai_reply(one_object(), index=2)])
    proposer = OpenAICountProposer(tmp_path / "missing_usage", client=client)
    models, record = proposer.propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial"
    )
    assert len(calls) == 1 and len(models) == 1
    assert record["calls"] == 1 and record["repairs"] == 0 and record["status"] == "parsed"
    assert record["rejected_candidates"] == []
    assert len(record["responses"]) == 1
    assert record["responses"][0]["usage"] is None
    assert record["responses"][0]["usage_known"] is False
    assert record["responses"][0]["completion_id"] == "chatcmpl_test_only_1"
    assert record["last_call_usage"] is None and record["last_call_usage_known"] is False
    assert record["usage_total"] == {} and record["usage_total_complete"] is False
    assert record["usage_missing_indices"] == [0]


def test_21g2_sdk_object_contract_error_is_recorded_and_never_repaired(tmp_path):
    """An SDK object-contract error (here `choices` is empty) is not a
    format failure either - one call, recorded, no repair, and the failure is
    marked as not repair-eligible."""
    broken = SimpleNamespace(choices=[], usage=None, id="chatcmpl_broken", model="m")
    client, calls = scripted_openai([broken, openai_reply(one_object(), index=2)])
    proposer = OpenAICountProposer(tmp_path / "broken_object", client=client)
    models, record = proposer.propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial"
    )
    assert len(calls) == 1 and models == [] and record["status"] == "proposal_failed"
    assert record["calls"] == 1 and record["repairs"] == 0
    assert record["rejected_candidates"][0]["error_type"] == "IndexError"
    assert record["rejected_candidates"][0]["repair_eligible"] is False


def test_21g2_format_failure_is_the_only_thing_marked_repair_eligible(tmp_path):
    """The positive half - invalid JSON and the candidate-constraint
    messages ARE repair-eligible, and still cost exactly one repair."""
    client, calls = scripted_openai(
        [openai_reply("not JSON at all"), openai_reply(one_object(), index=2)]
    )
    proposer = OpenAICountProposer(tmp_path / "format_failure", client=client)
    models, record = proposer.propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial"
    )
    assert len(calls) == 2 and len(models) == 1 and record["repairs"] == 1
    assert record["rejected_candidates"][0]["repair_eligible"] is True
    assert "not valid JSON" in record["rejected_candidates"][0]["reason"]


@pytest.mark.parametrize("value", [2, 3, -1, 1.5, True])
def test_21g2_repair_attempts_accepts_only_zero_or_one(tmp_path, value):
    """`range(1 + max(0, int(repair_attempts)))` let 3
    through and made four calls. 0 or 1 only - refused before any call."""
    client, calls = scripted_openai([openai_reply(one_object())])
    proposer = OpenAICountProposer(tmp_path / f"bound_{value!r}", client=client)
    with pytest.raises(ValueError, match="repair_attempts must be 0 or 1; 2 requires a verified CORE A8"):
        proposer.propose_one(
            '{"round":7}',
            context=P1_CONTEXT,
            initial_family="binomial",
            repair_attempts=value,
        )
    assert calls == [] and proposer.attempts == 0


def test_21g2_repair_attempts_zero_makes_exactly_one_call(tmp_path):
    """0 is still legal and means "no repair at all" (policy P4's value)."""
    client, calls = scripted_openai([openai_reply("not JSON at all")])
    proposer = OpenAICountProposer(tmp_path / "no_repair", client=client)
    models, record = proposer.propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial", repair_attempts=0
    )
    assert len(calls) == 1 and models == [] and record["calls"] == 1 and record["repairs"] == 0


def test_21g2_cli_refuses_a_repair_attempts_above_one(tmp_path, monkeypatch, capsys):
    """The CORE CLI accepted `--repair-attempts 3`; it now refuses, and no
    output directory is created."""
    from . import core_count

    out = tmp_path / "cli_bound"
    monkeypatch.setattr(
        sys, "argv", ["core_count", "--output", str(out), "--repair-attempts", "3"]
    )

    def never(args):
        """Fail the test if called: this code path must not run."""
        raise AssertionError("the CLI accepted an out-of-range --repair-attempts")

    # Defensive: if the bound ever regresses, this test fails immediately
    # instead of launching a real paper-profile run (main() defaults to
    # --profile paper, --rounds 20).
    monkeypatch.setattr(core_count, "run", never)
    with pytest.raises(SystemExit) as excinfo:
        core_count.main()
    assert excinfo.value.code == 2
    assert "--repair-attempts 0 or 1; 2 requires a verified CORE A8 launch contract" in capsys.readouterr().err
    assert not out.exists()


def test_21g2_usage_total_sums_every_response_exactly_once(tmp_path):
    """The top-level usage is the LAST call's usage
    (now named `last_call_usage`), and `usage_total` is the sum over
    responses[*] - each counted once, never both."""
    client, calls = scripted_openai(
        [openai_reply("not JSON at all", index=1), openai_reply(one_object(), index=2)]
    )
    proposer = OpenAICountProposer(tmp_path / "usage_total", client=client)
    models, record = proposer.propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial"
    )
    assert len(calls) == 2 and len(models) == 1
    assert record["responses"][0]["usage"] == {
        "input_tokens": 110, "output_tokens": 11, "cached_tokens": 20
    }
    assert record["responses"][1]["usage"] == {
        "input_tokens": 120, "output_tokens": 12, "cached_tokens": 20
    }
    assert record["last_call_usage"] == record["responses"][1]["usage"]
    assert record["usage_total"] == {
        "input_tokens": 230, "output_tokens": 23, "cached_tokens": 40
    }
    assert record["usage_total_complete"] is True and record["usage_missing_indices"] == []
    assert "usage" not in record and "request_id" not in record


def test_21g2_usage_total_marks_the_missing_call_instead_of_counting_zero(tmp_path):
    """An unknown usage is never counted as 0 - it names its index and
    makes the total incomplete, so a cost ledger reserves a budget for it."""
    client, calls = scripted_openai(
        [openai_reply("not JSON at all", usage=False), openai_reply(one_object(), index=2)]
    )
    proposer = OpenAICountProposer(tmp_path / "usage_partial", client=client)
    models, record = proposer.propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial"
    )
    assert len(calls) == 2 and len(models) == 1
    assert record["usage_total"] == {
        "input_tokens": 120, "output_tokens": 12, "cached_tokens": 20
    }
    assert record["usage_total_complete"] is False
    assert record["usage_missing_indices"] == [0]


def test_21g2_completion_id_and_http_request_id_are_named_apart(tmp_path):
    """`response.id` is the COMPLETION id
    (`chatcmpl_...`), not the HTTP `x-request-id`. The record now names both,
    and `http_request_id` is null when the SDK does not expose it (openai
    1.40.3 does not attach `_request_id` to the parsed model at all)."""
    client, _ = scripted_openai([openai_reply(one_object(), http_request_id="req_test_only_0")])
    proposer = OpenAICountProposer(tmp_path / "ids_present", client=client)
    _models, record = proposer.propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial"
    )
    assert record["completion_id"] == "chatcmpl_test_only_1"
    assert record["http_request_id"] == "req_test_only_0"
    assert record["responses"][0]["completion_id"] == "chatcmpl_test_only_1"
    assert record["responses"][0]["http_request_id"] == "req_test_only_0"

    bare, _ = scripted_openai([openai_reply(one_object())])
    proposer = OpenAICountProposer(tmp_path / "ids_absent", client=bare)
    _models, record = proposer.propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial"
    )
    assert record["completion_id"] == "chatcmpl_test_only_1"
    assert record["http_request_id"] is None


def test_21g2_real_sdk_mocktransport_debug_logs_and_traceback_are_masked(tmp_path, monkeypatch):
    """The installed SDK object, an
    httpx.MockTransport that raises a ConnectError CARRYING the fake key, and
    DEBUG logging on openai/anthropic/httpx/httpcore. Zero occurrences of the
    key in the captured log text, in the formatted traceback, in the record or
    in any written file. No HTTP request leaves the process - MockTransport
    answers every one of them in-process, and there is no paid call."""
    httpx = pytest.importorskip("httpx")
    openai_module = pytest.importorskip("openai")
    fake = "sk-test-FAKEKEY-21g2-sdk-debug-logging"
    monkeypatch.setenv("OPENAI_API_KEY", fake)
    sink = io.StringIO()
    handler = logging.StreamHandler(sink)
    loggers = [logging.getLogger(name) for name in count_proposer.MASKED_LOGGER_ROOTS]
    saved = [(logger, logger.level, logger.propagate) for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.propagate = False

    def refuse(request):
        """Raise to simulate a refused call."""
        raise httpx.ConnectError(f"fake low-level transport failure {fake}", request=request)

    client = openai_module.OpenAI(
        http_client=httpx.Client(transport=httpx.MockTransport(refuse)),
        max_retries=0,
        timeout=1.0,
    )
    try:
        raw = None
        try:
            client.chat.completions.create(
                model="gpt-4o", max_tokens=16, messages=[{"role": "user", "content": "{}"}]
            )
        except BaseException as exc:  # noqa: BLE001 - the exception IS the fixture
            raw = exc
        # The SDK wraps the transport failure (APIConnectionError("Connection
        # error.")), so the key is not in `str(exc)` - it is in the CHAINED
        # cause, which is exactly what the formatted traceback and the SDK's
        # own `exc_info=True` DEBUG log print.
        assert raw is not None
        raw_formatted = "".join(traceback.format_exception(type(raw), raw, raw.__traceback__))
        assert fake in raw_formatted and count_proposer.carries_secret(raw_formatted)
        try:
            raise count_proposer.masked_exception(raw) from None
        except BaseException as masked:  # noqa: BLE001 - re-raised for its traceback
            formatted = "".join(
                traceback.format_exception(type(masked), masked, masked.__traceback__)
            )
            # The substitution DID happen (the traceback carried the key even
            # though `str(exc)` did not - the SDK wrapper's message is clean),
            # and `from None` keeps the chained cause, the half that held the
            # key, out of the formatted traceback entirely.
            assert isinstance(masked, count_proposer.MaskedCredentialError)
            assert str(masked).startswith("APIConnectionError: ")
            assert fake not in str(masked) and fake not in repr(masked)
            assert fake not in formatted
        proposer = OpenAICountProposer(tmp_path / "sdk_debug", client=client)
        models, record = proposer.propose_one(
            '{"round":7}', context=P1_CONTEXT, initial_family="binomial"
        )
    finally:
        client.close()
        for logger, level, propagate in saved:
            logger.removeHandler(handler)
            logger.setLevel(level)
            logger.propagate = propagate
    logs = sink.getvalue()
    assert models == [] and record["calls"] == 1 and record["status"] == "proposal_failed"
    assert record["rejected_candidates"][0]["repair_eligible"] is False
    assert fake not in logs
    assert "<OPENAI_API_KEY:masked>" in logs, "the masking filter never engaged"
    assert fake not in json.dumps(record)
    for path in (tmp_path / "sdk_debug").iterdir():
        assert fake not in path.read_text()


@pytest.mark.parametrize(
    "logger_name",
    ["openai._base_client", "anthropic._base_client", "httpx._client", "httpcore._sync.http11"],
)
def test_21g2_logging_filter_masks_message_args_and_exc_info(logger_name):
    """The filter itself, on a record made through a CHILD logger of each
    of the four library roots the filter names - `openai._base_client` is
    where the OpenAI SDK actually logs, and a plain `logging.Filter` on
    `openai` would never see that record (Logger.handle runs only that
    logger's filters; callHandlers walks ancestors for HANDLERS)."""
    fake = "sk-test-FAKEKEY-21g2-logrecord-filter"
    sink = io.StringIO()
    handler = logging.StreamHandler(sink)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger(logger_name)
    saved = (logger.level, logger.propagate)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    try:
        logger.debug("header %s", f"Bearer {fake}")
        try:
            raise RuntimeError(f"auth failed for {fake}")
        except RuntimeError:
            logger.debug("Encountered Exception", exc_info=True)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved[0])
        logger.propagate = saved[1]
    text = sink.getvalue()
    assert fake not in text
    assert text.count("<OPENAI_API_KEY:masked>") == 2
    assert "RuntimeError" in text and "Traceback" in text


# --- the parse/judge parity rule shared with the ForceBench kit ---------------
# The failing behaviour was a single misleading message,
# "coefficient role outside fixed prior policy", for every one of these shapes
# - including the two whose ROLE was in fact correct.
P4_ROLE_OBJECT = {
    "expression": "b0+b1*z",
    "family": "beta_binomial",
    "parameters": {"b0": {"role": "intercept"}, "b1": {"role": "signed"}},
}
P4_ROLE_AND_DECLARED_PRIOR = {
    "expression": "b0+b1*z",
    "family": "beta_binomial",
    "parameters": {
        "b0": {"role": "intercept", "prior": ["Normal", 0.0, 2.0]},
        "b1": {"role": "signed", "prior": ["Normal", 0.0, 1.0]},
    },
}


def test_p4_object_parameter_values_are_normalized_and_recorded():
    """For CORE: `{"role": X}`, and `{"role": X, "prior": P}` with
    P exactly the declaration the request published, normalize to X and are
    recorded with `normalized_parameter_shape: true`. Same expression, same
    family, same model_id as the role-string form - nothing scientific moves."""
    plain, plain_ignored = parse_candidate_object(
        json.dumps({**P4_ROLE_OBJECT, "parameters": {"b0": "intercept", "b1": "signed"}}),
        "111",
        "binomial",
    )
    assert plain_ignored["normalized_parameter_shape"] is False
    for payload in (P4_ROLE_OBJECT, P4_ROLE_AND_DECLARED_PRIOR):
        hypothesis, ignored = parse_candidate_object(json.dumps(payload), "111", "binomial")
        assert ignored["normalized_parameter_shape"] is True
        assert dict(hypothesis.parameters) == {"b0": "intercept", "b1": "signed"}
        assert hypothesis.model_id == plain.model_id
        models, extra = parse_candidates(json.dumps([payload]), "111", "binomial")
        assert len(models) == 1 and extra["rejected_candidates"] == []
        assert extra["normalized_parameter_shape"] == [True]
        assert models[0].model_id == plain.model_id


def test_p4_a_restated_prior_that_differs_from_the_declaration_is_rejected():
    """For CORE: restating a DIFFERENT prior is a prior
    modification, which the contract forbids outright - so it is rejected with
    the pre-declared wording, not quietly dropped."""
    payload = {
        "expression": "b0+b1*z",
        "family": "beta_binomial",
        "parameters": {
            "b0": {"role": "intercept", "prior": ["Normal", 0.0, 5.0]},
            "b1": {"role": "signed"},
        },
    }
    with pytest.raises(ValueError, match="prior is fixed by the catalogue; do not restate it"):
        parse_candidate_object(json.dumps(payload), "111", "binomial")
    models, extra = parse_candidates(json.dumps([payload]), "111", "binomial")
    assert models == [] and extra["normalized_parameter_shape"] == []
    assert extra["rejected_candidates"][0]["reason"] == count_proposer.PRIOR_RESTATED_MESSAGE


@pytest.mark.parametrize(
    "value,word",
    [({"kind": "intercept"}, "object"), (17, "number"), (["intercept"], "array"), (None, "null")],
)
def test_p4_other_object_or_non_string_values_are_rejected_clearly(value, word):
    """For CORE: anything else is rejected with the shape message and
    the JSON type it actually got - no longer disguised as a role violation."""
    payload = {
        "expression": "b0+b1*z",
        "family": "beta_binomial",
        "parameters": {"b0": value, "b1": "signed"},
    }
    expected = count_proposer.PARAMETER_SHAPE_MESSAGE % word
    with pytest.raises(ValueError) as excinfo:
        parse_candidate_object(json.dumps(payload), "111", "binomial")
    assert str(excinfo.value) == expected
    assert 'parameters values must be role strings, e.g. {"c": "positive"}' in expected
    models, extra = parse_candidates(json.dumps([payload]), "111", "binomial")
    assert models == [] and extra["rejected_candidates"][0]["reason"] == expected


def test_p4_normalization_reaches_the_transport_record(tmp_path):
    """End to end: a reply in the nested shape is admitted over the
    transport and the record says so, instead of the round being lost."""
    client, calls = scripted_openai([openai_reply(json.dumps(P4_ROLE_OBJECT))])
    proposer = OpenAICountProposer(tmp_path / "p4_normalized", client=client)
    models, record = proposer.propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial"
    )
    assert len(calls) == 1 and len(models) == 1 and record["status"] == "parsed"
    assert record["normalized_parameter_shape"] is True
    saved = json.loads((proposer.records / "round_07.response.json").read_text())
    assert saved["normalized_parameter_shape"] is True


def test_22a_core_points_csv_is_the_25_column_v1_3_contract_with_blanks_not_zeros(tmp_path):
    """The plotting contract.

    The CORE emitter writes the 25 declared columns in the declared order -
    the contract's 1..21 ending `variant`, then `code_sha` (22), `brier` (23),
    `nmsle` (24), `rmsle` (25). The three metrics that have no definition on
    the CORE path are EMPTY STRINGS, never 0 (a design decision of the plot contract), and
    `y_label` carries the human name of CORE's own primary metric while
    `y_name` keeps the machine key.

    `write_points` is called directly, so this pins the emitter itself rather
    than a whole run - the same shape the pre-existing
    `test_p4_write_points_label_...` check uses.
    """
    from . import core_count

    assert len(core_count.CONTRACT_POINTS_COLUMNS) == 25
    assert core_count.CONTRACT_POINTS_COLUMNS[20:] == (
        "variant",
        "code_sha",
        "brier",
        "nmsle",
        "rmsle",
    )
    # The legacy 21-column tuple is untouched: boxinggym_core.py still uses it
    # and is owned by another task.
    assert len(core_count.POINTS_COLUMNS) == 21
    assert "brier" not in core_count.POINTS_COLUMNS

    config = {
        "proposer": "mock_fixed_template_responses_NOT_LLM",
        "llm_model": "claude-opus-4-7",
        "seed": 11,
        "cell": "111",
        "profile": "dev",
        "structure": "out",
        "truth_family": "beta_binomial",
        "output": str(tmp_path / "core_points_run"),
        "source_files_sha256": {"core_count.py": "0" * 64},
    }
    row = {
        "observation_count": 9,
        "bma": {"nll": 1.25, "coverage90": 0.9, "width90": 1.5, "kl": 0.125},
        "round": 1,
        "phase": "after_proposal",
    }
    path = tmp_path / "points.csv"
    core_count.write_points(path, config, [row], "completed", "")

    lines = path.read_text().splitlines()
    assert lines[0].split(",") == list(core_count.CONTRACT_POINTS_COLUMNS)
    written = list(csv.DictReader(lines))
    assert len(written) == 1
    written = written[0]
    assert written["suite"] == "core"
    assert written["y_name"] == "nll" and written["y_primary"] == "1.25"
    assert written["y_label"] == core_count.CORE_Y_LABEL == "held-out NLL (nats/obs)"
    assert written["nll"] == "1.25" and written["kl"] == "0.125"
    assert written["coverage90"] == "0.9" and written["width90"] == "1.5"
    # The full 64-hex sha256, not a 16-character prefix.
    assert written["code_sha"] == "0" * 64 and len(written["code_sha"]) == 64
    # Undefined -> empty string, never 0.
    for column in ("crps", "is_0.1", "brier", "nmsle", "rmsle", "variant"):
        assert written[column] == "", column
        assert written[column] != "0"


# --- --run-label on core_count (all-benchmark 2 h smoke) -------


def test_18_core_count_refuses_run_label_production_before_any_output(tmp_path, monkeypatch):
    """REFUSE: `--run-label production` on a run whose derived label is not
    production is refused at run() entry, before the output directory is
    created - so a refused run leaves nothing behind to be mistaken for one
    that happened. `fit` is stubbed defensively; the gate fires long before."""
    core_count = stub_fits(monkeypatch)
    out = tmp_path / "should_not_exist"
    args = run_args(out, profile="paper", proposer="mock", run_label="production")
    with pytest.raises(ValueError, match="--run-label production"):
        core_count.run(args)
    assert not out.exists()


def test_18_core_count_accepts_run_label_development_and_records_it(tmp_path, monkeypatch):
    """ACCEPT: `--run-label development` runs to completion and the chosen
    label reaches config.json (both the `run_label` CLI echo and the `label`
    key) and every points.csv row."""
    core_count = stub_fits(monkeypatch)
    out = tmp_path / "dev_label"
    core_count.run(run_args(out, rounds=1, run_label="development"))
    config = json.loads((out / "config.json").read_text())
    assert config["run_label"] == "development"
    assert config["label"] == "development"
    with (out / "points.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and {row["label"] for row in rows} == {"development"}


def test_18_core_count_run_label_overrides_a_production_default(tmp_path, monkeypatch):
    """Both omission and explicit development stay development."""
    core_count = stub_fits(monkeypatch)

    def labels_for(name, **overrides):
        """Return the labels used for the given test case."""
        out = tmp_path / name
        core_count.run(run_args(out, rounds=1, **overrides))
        config = json.loads((out / "config.json").read_text())
        with (out / "points.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows
        return config.get("label"), {row["label"] for row in rows}

    # Missing label keeps the legacy config shape but no longer promotes it.
    assert labels_for("derived") == (None, {"development"})
    assert labels_for("stated", run_label="development") == ("development", {"development"})


# ---------------------------------------------------------------------------
# The OPTIONAL per-component identity.
# A task that declares several output components calls propose_one once per
# component in the SAME round; the records are write-once, so without the
# component in the identity the second component of a round died on
# FileExistsError. Absent, everything below is byte for byte what it was.
# ---------------------------------------------------------------------------

def test_two_components_of_one_round_write_separate_records(tmp_path):
    """The reproduction case: same round, two components, no exception."""
    proposer, calls, context = p1_fixture(tmp_path, [one_object("b0+b1*tanh(3*z)", "beta_binomial")])
    records = {}
    for component in ("happiness", "sadness"):
        models, record = proposer.propose_one(
            '{"round":7}',
            context={**context, "component": component},
            initial_family="binomial",
        )
        assert models, component
        records[component] = record
    assert len(calls) == 2 and proposer.attempts == 2
    written = sorted(path.name for path in proposer.records.iterdir())
    assert written == [
        "round_07__happiness.request.json", "round_07__happiness.response.json",
        "round_07__sadness.request.json", "round_07__sadness.response.json",
    ]
    for component, record in records.items():
        assert record["component"] == component
        stored = json.loads((proposer.records / f"round_07__{component}.response.json").read_text())
        assert stored["component"] == component
        request = json.loads((proposer.records / f"round_07__{component}.request.json").read_text())
        assert request["identity"]["component"] == component
        assert request["identity"]["round"] == 7


def test_two_components_of_one_round_write_separate_records_on_openai(tmp_path):
    """The same contract on the OpenAI transport - both classes were changed."""
    proposer, calls, context = p1_openai_fixture(
        tmp_path, [one_object("b0+b1*tanh(3*z)", "beta_binomial")])
    for component in ("happiness", "sadness"):
        models, record = proposer.propose_one(
            '{"round":7}',
            context={**context, "component": component},
            initial_family="binomial",
        )
        assert models and record["component"] == component
    assert len(calls) == 2
    assert sorted(path.name for path in proposer.records.iterdir()) == [
        "round_07__happiness.request.json", "round_07__happiness.response.json",
        "round_07__sadness.request.json", "round_07__sadness.response.json",
    ]


def test_a_context_without_a_component_keeps_the_old_names_and_keys(tmp_path):
    """Every single-outcome task: same file names, same record key set as before."""
    proposer, _, context = p1_fixture(tmp_path, [one_object("b0+b1*tanh(3*z)", "beta_binomial")])
    _, record = proposer.propose_one('{"round":7}', context=context, initial_family="binomial")
    assert sorted(path.name for path in proposer.records.iterdir()) == [
        "round_07.request.json", "round_07.response.json",
    ]
    assert "component" not in record
    request = json.loads((proposer.records / "round_07.request.json").read_text())
    assert "component" not in request["identity"]
    assert set(request["identity"]) == {
        "run_id", "world_seed", "cell", "proposal_rep", "round", "provider",
        "requested_model", "policy", "decoding", "prompt_version", "prompt_sha256",
        "packet_bytes", "packet_sha256",
    }


def test_the_component_must_be_a_bare_name(tmp_path):
    """A separator or a dot would write - or overwrite - the wrong file.

    A dot matters because the record paths are built with `with_suffix`, which
    replaces everything after the LAST dot: `a.b` and `a.c` would both land on
    `round_07__a.request.json` and re-create the very collision this fixes.
    """
    proposer, calls, context = p1_fixture(tmp_path, [one_object()])
    for bad in ("", "../escape", "a/b", "a.b", 7, None):
        with pytest.raises(ValueError, match="component"):
            proposer.propose_one(
                '{"round":7}', context={**context, "component": bad}, initial_family="binomial"
            )
    # refused BEFORE any call and before any record
    assert calls == [] and list(proposer.records.iterdir()) == []


def test_the_identity_still_refuses_any_other_extra_key(tmp_path):
    """`component` widened the accepted set by exactly one key, not by anything."""
    proposer, _, context = p1_fixture(tmp_path, [one_object()])
    with pytest.raises(ValueError, match="identity required"):
        proposer.propose_one(
            '{"round":7}', context={**context, "task": "emotion"}, initial_family="binomial"
        )
    with pytest.raises(ValueError, match="identity required"):
        proposer.propose_one(
            '{"round":7}',
            context={**{k: v for k, v in context.items() if k != "cell"}, "component": "happiness"},
            initial_family="binomial",
        )


# --------------------------------------------------------------------- reasoning spec parsing
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("effort=none", {"effort": "none"}),
        ("effort=minimal", {"effort": "minimal"}),
        ("effort=low", {"effort": "low"}),
        ("effort=medium", {"effort": "medium"}),
        ("effort=high", {"effort": "high"}),
        ("effort=xhigh", {"effort": "xhigh"}),
        ("effort=max", {"effort": "max"}),
        ("max_tokens=512", {"max_tokens": 512}),
        ("effort=high,exclude", {"effort": "high", "exclude": True}),
        ("exclude,effort=high", {"effort": "high", "exclude": True}),
        (" max_tokens = 512 , exclude ", {"max_tokens": 512, "exclude": True}),
    ],
)
def test_d1_reasoning_spec_parses_to_one_canonical_dict(text, expected):
    """The dict is canonically ordered, so the same request is the same bytes
    however the option was typed."""
    spec = count_proposer.parse_reasoning_spec(text)
    assert spec == expected and list(spec) == list(expected)
    assert json.dumps(spec) == json.dumps(count_proposer.parse_reasoning_spec(text))


@pytest.mark.parametrize("text", [
    "", "   ", "effort", "exclude", "effort=", "effort=none,max_tokens=512", "max_tokens=0",
    "max_tokens=-1", "max_tokens=1.5", "effort=highest", "effort=none,effort=low",
    "exclude,exclude", "exclude=true", "effort=none,unknown", "effort=none,max_tokens",
    None, 5, {"effort": "none"},
])
def test_d1_invalid_reasoning_spec_is_refused(text):
    """Check that d1 invalid reasoning spec is refused."""
    with pytest.raises(ValueError):
        count_proposer.parse_reasoning_spec(text)


@pytest.mark.parametrize("value", [
    {"effort": "none"}, {"max_tokens": 512}, {"effort": "high", "exclude": True}, "effort=high",
])
def test_d1_normalize_reasoning_accepts_text_or_a_parsed_spec(value):
    """Check that d1 normalize reasoning accepts text or a parsed spec."""
    assert count_proposer.normalize_reasoning(value) == (
        count_proposer.parse_reasoning_spec(value) if isinstance(value, str) else value)
    assert count_proposer.normalize_reasoning(None) is None


@pytest.mark.parametrize("value", [
    {}, {"effort": "none", "max_tokens": 512}, {"effort": "nope"}, {"max_tokens": 0},
    {"exclude": True}, {"effort": "none", "exclude": "yes"}, {"effort": "none", "other": 1}, 5, [],
])
def test_d1_normalize_reasoning_refuses_anything_off_the_grammar(value):
    """Check that d1 normalize reasoning refuses anything off the grammar."""
    with pytest.raises(ValueError):
        count_proposer.normalize_reasoning(value)


@pytest.mark.parametrize("value", ["1", 1, "32768", 32768, "4000"])
def test_d1_max_tokens_accepts_the_declared_range(value):
    """Check that d1 max tokens accepts the declared range."""
    assert count_proposer.parse_max_tokens(value) == int(value)


@pytest.mark.parametrize("value", ["0", 0, -1, "32769", 32769, True, 1.5, "", "x", None, "1e3"])
def test_d1_max_tokens_refuses_everything_else(value):
    """Check that d1 max tokens refuses everything else."""
    with pytest.raises(ValueError):
        count_proposer.parse_max_tokens(value)


def test_d1_stated_budget_and_reasoning_reach_the_body_and_the_identity(tmp_path):
    """(a)+(c): `extra_body` carries the routing order and the reasoning spec
    together, and the identity records the decoding that was actually sent."""
    client, calls = scripted_openai([openai_reply(one_object())])
    proposer = count_proposer.OpenAICompatCountProposer(
        tmp_path / "stated", base_url="https://compat.example.invalid/api/v1",
        key_name="SYNTHETIC_COMPAT_CREDENTIAL", model="synthetic/vendor-edition", client=client,
        provider_order=["alpha"], reasoning="effort=none", max_tokens=4000)
    models, record = proposer.propose_one('{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert len(models) == 1 and calls[0]["max_tokens"] == 4000
    assert calls[0]["extra_body"] == {"provider": {"order": ["alpha"], "allow_fallbacks": False},
                                      "reasoning": {"effort": "none"}}
    assert record["decoding"] == {"temperature": "not_sent/provider_default",
                                  "max_tokens": 4000, "reasoning": {"effort": "none"}}
    stored = json.loads((tmp_path / "stated" / "round_07.request.json").read_text())
    assert stored["bodies"][0]["extra_body"] == calls[0]["extra_body"]


def test_d1_native_transports_take_the_budget_but_never_the_reasoning(tmp_path):
    """Check that d1 native transports take the budget but never the reasoning."""
    client, calls = scripted_openai([openai_reply(one_object())])
    proposer = OpenAICountProposer(tmp_path / "budget", client=client, max_tokens=256)
    _, record = proposer.propose_one('{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert calls[0]["max_tokens"] == 256 and "extra_body" not in calls[0]
    assert record["decoding"] == {"temperature": "not_sent/provider_default", "max_tokens": 256}
    anthropic_calls = []

    def create(**body):
        """Fake client entry point: record the request and return the canned reply."""
        anthropic_calls.append(body)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=one_object())],
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 9}),
            id="test-only-1", model="returned-test-model", stop_reason="end_turn")

    native = AnthropicCountProposer(tmp_path / "anthropic_budget", max_tokens=256,
                                    client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    native.propose_one('{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert anthropic_calls[0]["max_tokens"] == 256
    # The stock path now takes `effort=<level>` (see the reasoning-transport tests); the compatibility
    # endpoint's fields stay refused there.
    with pytest.raises(ValueError, match="effort=<level> only"):
        OpenAICountProposer(tmp_path / "refused", client=object(), reasoning="max_tokens=512")
    assert not (tmp_path / "refused").exists()


# --------------------------------------------------------------------- empty-reply repair
@pytest.mark.parametrize("empty", ["", "   ", None])
def test_d1_empty_reply_repairs_with_one_user_turn_and_no_echo(tmp_path, empty):
    """The probe's failure: attempt 0 returned finish_reason length with an
    empty body, and the repair call echoed an empty assistant turn, which the
    endpoint refused outright. The repair turn now says so in one user turn."""
    client, calls = scripted_openai([openai_reply(empty, finish_reason="length"),
                                     openai_reply(one_object(), index=2)])
    proposer = OpenAICountProposer(tmp_path / "empty_body", client=client)
    models, record = proposer.propose_one('{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert len(calls) == 2 and len(models) == 1 and record["status"] == "parsed"
    assert record["calls"] == 2 and record["repairs"] == 1
    assert [message["role"] for message in calls[1]["messages"]] == ["system", "user", "user"]
    assert calls[1]["messages"][2]["content"] == count_proposer.REPAIR_INSTRUCTION_EMPTY.format(
        finish_reason="length")
    rejected = record["rejected_candidates"][0]
    assert rejected["reason"] == "empty response body (finish_reason=length)"
    assert rejected["repair_kind"] == "empty_body" and rejected["repair_eligible"] is True
    assert record["responses"][0]["raw_text"] == empty


def test_d1_two_empty_replies_stop_after_two_calls(tmp_path):
    """Check that d1 two empty replies stop after two calls."""
    client, calls = scripted_openai([openai_reply("", finish_reason="length")])
    proposer = OpenAICountProposer(tmp_path / "empty_twice", client=client)
    models, record = proposer.propose_one('{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert models == [] and len(calls) == 2 and record["status"] == "proposal_failed"
    assert [item["repair_kind"] for item in record["rejected_candidates"]] == ["empty_body"] * 2
    assert all("assistant" not in [m["role"] for m in call["messages"]] for call in calls)


def test_d1_a_truncated_reply_keeps_the_echo_and_is_marked_malformed(tmp_path):
    """A body that EXISTS but was cut off is the pre-existing path: echo plus
    the validation message, exactly as before."""
    client, calls = scripted_openai([openai_reply('{"expression": "b0+b1*z"',
                                                  finish_reason="length"),
                                     openai_reply(one_object(), index=2)])
    proposer = OpenAICountProposer(tmp_path / "truncated", client=client)
    models, record = proposer.propose_one('{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert len(models) == 1 and [m["role"] for m in calls[1]["messages"]] == [
        "system", "user", "assistant", "user"]
    assert calls[1]["messages"][2]["content"] == '{"expression": "b0+b1*z"'
    rejected = record["rejected_candidates"][0]
    assert rejected["repair_kind"] == "malformed"
    assert rejected["reason"] == "response did not finish with stop"


def test_d1_anthropic_empty_text_repairs_without_echo(tmp_path):
    """Check that d1 anthropic empty text repairs without echo."""
    proposer, calls, context = p1_fixture(tmp_path, ["", one_object()])
    models, record = proposer.propose_one('{"round":7}', context=context, initial_family="binomial")
    assert len(calls) == 2 and len(models) == 1
    assert [message["role"] for message in calls[1]["messages"]] == ["user", "user"]
    assert calls[1]["messages"][1]["content"] == count_proposer.REPAIR_INSTRUCTION_EMPTY.format(
        finish_reason="end_turn")
    rejected = record["rejected_candidates"][0]
    assert rejected["reason"] == "empty response body (finish_reason=end_turn)"
    assert rejected["repair_kind"] == "empty_body"


def test_d1_p4_records_the_empty_body_verdict_without_a_second_call(tmp_path):
    """Check that d1 p4 records the empty body verdict without a second call."""
    proposer, calls, context, public, pool = openai_fixture(tmp_path, text="   ")
    models, metadata = proposer.propose(public, pool, context=context, initial_family="binomial")
    assert models == [] and len(calls) == 1
    assert metadata["status"] == "proposal_failed"
    assert metadata["error_type"] == "EmptyResponseBody"


def test_d1_reasoning_tokens_are_recorded_only_when_the_provider_reports_them(tmp_path):
    """Check that d1 reasoning tokens are recorded only when the provider reports them."""
    reported = openai_reply(one_object())
    reported.usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=64)
    client, _ = scripted_openai([reported])
    _, record = OpenAICountProposer(tmp_path / "reported", client=client).propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert record["last_call_usage"]["reasoning_tokens"] == 64
    assert record["usage_total"]["reasoning_tokens"] == 64
    silent, _ = scripted_openai([openai_reply(one_object())])
    _, plain = OpenAICountProposer(tmp_path / "silent", client=silent).propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert set(plain["last_call_usage"]) == {"input_tokens", "output_tokens", "cached_tokens"}
    assert "reasoning_tokens" not in plain["usage_total"]


# --------------------------------------------------------------------- registered decoding options
def synthetic_manifest(tmp_path, settings):
    """Build a small synthetic run manifest for the registration checks."""
    payload = json.dumps({"runtime_settings": {"core": settings}}).encode()
    path = tmp_path / "synthetic_manifest.json"
    path.write_bytes(payload)
    return {"run_label": "production", "manifest_path": str(path),
            "manifest_sha256": hashlib.sha256(payload).hexdigest()}


@pytest.mark.parametrize("stated", [
    {"llm_reasoning": {"effort": "low"}, "llm_max_tokens": 4000},
    {"llm_reasoning": {"effort": "none"}, "llm_max_tokens": 1600},
    {"llm_reasoning": {"effort": "none", "exclude": True}, "llm_max_tokens": 4000},
    {"llm_max_tokens": True},
])
def test_d1_production_refuses_an_unregistered_decoding_setting(tmp_path, stated):
    """Check that d1 production refuses an unregistered decoding setting."""
    registered = {"llm_reasoning": {"effort": "none"}, "llm_max_tokens": 4000}
    receipt = synthetic_manifest(tmp_path, {"particles": 500, **registered})
    with pytest.raises(SystemExit) as refusal:
        count_proposer.require_registered_decoding(stated, receipt, "core")
    assert refusal.value.code == 2


def test_d1_production_passes_only_on_an_equal_registration(tmp_path):
    """Check that d1 production passes only on an equal registration."""
    stated = {"llm_reasoning": {"effort": "none"}, "llm_max_tokens": 4000}
    receipt = synthetic_manifest(tmp_path, {"particles": 500, **stated})
    assert count_proposer.require_registered_decoding(stated, receipt, "core") is None
    with pytest.raises(SystemExit) as other_bundle:
        count_proposer.require_registered_decoding(stated, receipt, "chembench")
    assert other_bundle.value.code == 2
    (tmp_path / "synthetic_manifest.json").write_bytes(b'{"runtime_settings": {}}')
    with pytest.raises(SystemExit) as replaced:
        count_proposer.require_registered_decoding(stated, receipt, "core")
    assert replaced.value.code == 2


def test_d1_development_and_unstated_options_bind_nothing(tmp_path):
    """Check that d1 development and unstated options bind nothing."""
    stated = {"llm_reasoning": {"effort": "none"}}
    receipt = synthetic_manifest(tmp_path, {"particles": 500})
    assert count_proposer.require_registered_decoding(
        stated, {**receipt, "run_label": "development"}, "core") is None
    assert count_proposer.require_registered_decoding({"llm_model": "x"}, receipt, "core") is None


# --------------------------------------------------------------------- reasoning transport (`test_rm_*`)
#: The canonical-JSON sha256 of the default (no reasoning) P1 request bodies at the base commit - the
#: ChemBench edition 2 registration commit the gpt-4o baseline ran on - computed there with these
#: same fixtures (packet '{"round":7}', P1_CONTEXT, default models). An unstated reasoning spec
#: must leave both transports' bodies byte for byte what that baseline sent.
RM_BASE_OPENAI_P1_BODY_SHA256 = "9105d3d80e9da91aa861e5bc01bc037e9d7aaae49dc0abf3fd6c0f6d8cedf986"
RM_BASE_ANTHROPIC_P1_BODY_SHA256 = "ba15938e2b3e30c41965904199e28f4536c6138c457c55eb9217234f37bc233b"
REASONING_MODEL_KEYS = {"model", "max_completion_tokens", "reasoning_effort", "messages"}


def canonical_sha(body):
    """Return the sha256 of the canonical JSON form of the given object."""
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def anthropic_client(replies, *, thinking_blocks=0):
    """An injected Messages-API client answering `replies` in order; every reply is preceded by
    `thinking_blocks` thinking blocks, as an adaptive-thinking model may answer."""
    calls = []

    def create(**body):
        """Fake client entry point: record the request and return the canned reply."""
        calls.append(body)
        text = replies[min(len(calls) - 1, len(replies) - 1)]
        content = [SimpleNamespace(type="thinking", thinking="", signature=f"test-only-signature-{i}")
                   for i in range(thinking_blocks)]
        content.append(SimpleNamespace(type="text", text=text))
        return SimpleNamespace(
            content=content, id=f"test-only-{len(calls)}", model="returned-test-model", stop_reason="end_turn",
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 9, "output_tokens": 30}))

    return SimpleNamespace(messages=SimpleNamespace(create=create)), calls


def test_rm_openai_default_body_is_the_pre_rm_body_byte_for_byte(tmp_path):
    """Check that rm openai default body is the pre rm body byte for byte."""
    client, calls = scripted_openai([openai_reply(one_object())])
    _, record = OpenAICountProposer(tmp_path / "plain", client=client).propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert list(calls[0]) == ["model", "max_tokens", "messages"] and calls[0]["max_tokens"] == 1600
    assert canonical_sha(calls[0]) == RM_BASE_OPENAI_P1_BODY_SHA256
    assert record["decoding"] == {"temperature": "not_sent/provider_default", "max_tokens": 1600}
    assert "thinking_blocks" not in record["responses"][0]


def test_rm_openai_effort_sends_the_reasoning_model_shape_on_rounds_and_bursts(tmp_path):
    """Check that rm openai effort sends the reasoning model shape on rounds and bursts."""
    client, calls = scripted_openai([openai_reply(one_object())])
    proposer = OpenAICountProposer(tmp_path / "effort", client=client, reasoning="effort=medium",
                                   max_tokens=4000)
    _, record = proposer.propose_one('{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert set(calls[0]) == REASONING_MODEL_KEYS
    assert calls[0]["max_completion_tokens"] == 4000 and calls[0]["reasoning_effort"] == "medium"
    assert record["decoding"] == {"temperature": "not_sent/provider_default", "max_tokens": 4000,
                                  "reasoning": {"effort": "medium"},
                                  "token_parameter": "max_completion_tokens", "reasoning_effort": "medium"}
    stored = json.loads((tmp_path / "effort" / "round_07.request.json").read_text())
    assert stored["bodies"][0] == calls[0] and stored["identity"]["decoding"] == record["decoding"]
    # The P4 burst path sends the same shape, with the transport's 1600 when no budget is stated.
    fixture_proposer, burst_calls, context, public, pool = openai_fixture(tmp_path)
    burst = OpenAICountProposer(tmp_path / "burst", client=fixture_proposer.client, reasoning="effort=medium")
    _, metadata = burst.propose(public, pool, context=context, initial_family="binomial")
    assert set(burst_calls[0]) == REASONING_MODEL_KEYS and burst_calls[0]["max_completion_tokens"] == 1600
    assert metadata["decoding"]["token_parameter"] == "max_completion_tokens"


@pytest.mark.parametrize("spec, message", [
    ("max_tokens=512", "effort=<level> only"), ("effort=low,exclude", "effort=<level> only"),
    ("effort=low,thinking=adaptive", "effort=<level> only"),
])
def test_rm_openai_refuses_the_compat_and_anthropic_fields(tmp_path, spec, message):
    """Check that rm openai refuses the compat and anthropic fields."""
    with pytest.raises(ValueError, match=message):
        OpenAICountProposer(tmp_path / "refused", client=object(), reasoning=spec)
    assert not (tmp_path / "refused").exists()


def test_rm_anthropic_default_body_is_the_pre_rm_body_byte_for_byte(tmp_path):
    """Check that rm anthropic default body is the pre rm body byte for byte."""
    proposer, calls, context = p1_fixture(tmp_path, [one_object()])
    _, record = proposer.propose_one('{"round":7}', context=context, initial_family="binomial")
    assert list(calls[0]) == ["model", "max_tokens", "system", "messages"]
    assert canonical_sha(calls[0]) == RM_BASE_ANTHROPIC_P1_BODY_SHA256
    assert record["decoding"] == {"temperature": "not_sent/provider_default", "max_tokens": 1600}
    assert "thinking_blocks" not in record["responses"][0]


def test_rm_anthropic_effort_and_adaptive_thinking_reach_the_body_and_only_text_is_parsed(tmp_path):
    """Check that rm anthropic effort and adaptive thinking reach the body and only text is parsed."""
    client, calls = anthropic_client([one_object()], thinking_blocks=2)
    proposer = AnthropicCountProposer(tmp_path / "adaptive", client=client,
                                      reasoning="effort=high,thinking=adaptive")
    models, record = proposer.propose_one('{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert len(models) == 1 and record["status"] == "parsed"
    assert list(calls[0]) == ["model", "max_tokens", "system", "messages", "output_config", "thinking"]
    assert calls[0]["output_config"] == {"effort": "high"} and calls[0]["thinking"] == {"type": "adaptive"}
    assert calls[0]["max_tokens"] == 1600
    assert record["decoding"] == {"temperature": "not_sent/provider_default", "max_tokens": 1600,
                                  "reasoning": {"effort": "high", "thinking": "adaptive"},
                                  "token_parameter": "max_tokens", "output_config": {"effort": "high"},
                                  "thinking": {"type": "adaptive"}}
    assert record["responses"][0]["thinking_blocks"] == 2
    assert record["responses"][0]["raw_text"] == one_object()
    stored = json.loads((tmp_path / "adaptive" / "round_07.request.json").read_text())
    assert stored["bodies"][0] == calls[0]
    # effort alone adds output_config and nothing else.
    client2, calls2 = anthropic_client([one_object()])
    AnthropicCountProposer(tmp_path / "effort_only", client=client2, reasoning="effort=medium").propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert list(calls2[0]) == ["model", "max_tokens", "system", "messages", "output_config"]
    # The P4 burst path carries the same keys and counts the thinking blocks too.
    fixture_proposer, burst_calls, context, public, pool = fixture(tmp_path)
    burst = AnthropicCountProposer(tmp_path / "burst", client=fixture_proposer.client,
                                   reasoning="effort=high,thinking=adaptive")
    _, metadata = burst.propose(public, pool, context=context, initial_family="binomial")
    assert burst_calls[0]["output_config"] == {"effort": "high"}
    assert burst_calls[0]["thinking"] == {"type": "adaptive"}
    assert metadata["decoding"]["output_config"] == {"effort": "high"} and metadata["thinking_blocks"] == 0


@pytest.mark.parametrize("spec, message", [
    ("effort=none", "none and minimal"), ("effort=minimal", "none and minimal"),
    ("max_tokens=512", "OpenRouter"), ("effort=high,exclude", "OpenRouter"),
])
def test_rm_anthropic_refuses_levels_and_fields_it_cannot_send(tmp_path, spec, message):
    """Check that rm anthropic refuses levels and fields it cannot send."""
    with pytest.raises(ValueError, match=message):
        AnthropicCountProposer(tmp_path / "refused", client=object(), reasoning=spec)
    assert not (tmp_path / "refused").exists()


def test_rm_the_repair_turn_carries_the_same_settings_and_echoes_text_only(tmp_path):
    """Check that rm the repair turn carries the same settings and echoes text only."""
    client, calls = anthropic_client(["not a candidate", one_object()], thinking_blocks=1)
    proposer = AnthropicCountProposer(tmp_path / "repair", client=client,
                                      reasoning="effort=high,thinking=adaptive")
    models, record = proposer.propose_one('{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert len(calls) == 2 and len(models) == 1 and record["repairs"] == 1
    for body in calls:
        assert body["output_config"] == {"effort": "high"} and body["thinking"] == {"type": "adaptive"}
    # The assistant turn is re-sent as text only - no thinking block goes back (allowed outside
    # tool use: platform.claude.com/docs/en/build-with-claude/thinking, Preserving thinking blocks).
    assert [message["role"] for message in calls[1]["messages"]] == ["user", "assistant", "user"]
    assert calls[1]["messages"][1] == {"role": "assistant", "content": "not a candidate"}
    assert [reply["thinking_blocks"] for reply in record["responses"]] == [1, 1]
    oclient, ocalls = scripted_openai([openai_reply("not a candidate", index=1),
                                       openai_reply(one_object(), index=2)])
    OpenAICountProposer(tmp_path / "orepair", client=oclient, reasoning="effort=medium").propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert len(ocalls) == 2 and all(set(body) == REASONING_MODEL_KEYS for body in ocalls)
    assert [message["role"] for message in ocalls[1]["messages"]] == ["system", "user", "assistant", "user"]


def test_rm_decoding_identity_adds_no_key_without_a_spec_and_names_what_was_sent_with_one():
    """Check that rm decoding identity adds no key without a spec and names what was sent with one."""
    plain = {"temperature": "not_sent/provider_default", "max_tokens": 1600}
    for provider in (None, "openai", "anthropic", "openai_compat"):
        assert count_proposer.decoding_identity(1600, None, provider=provider) == plain
    assert count_proposer.decoding_identity(1600, {"effort": "low"}, provider="openai_compat") == {
        **plain, "reasoning": {"effort": "low"}}
    assert count_proposer.decoding_identity(1600, {"effort": "low"}, provider="openai") == {
        **plain, "reasoning": {"effort": "low"}, "token_parameter": "max_completion_tokens",
        "reasoning_effort": "low"}
    assert count_proposer.decoding_identity(1600, {"effort": "low", "thinking": "adaptive"},
                                            provider="anthropic") == {
        **plain, "reasoning": {"effort": "low", "thinking": "adaptive"}, "token_parameter": "max_tokens",
        "output_config": {"effort": "low"}, "thinking": {"type": "adaptive"}}


@pytest.mark.parametrize("text, expected", [
    ("effort=high,thinking=adaptive", {"effort": "high", "thinking": "adaptive"}),
    (" thinking = adaptive , effort=low ", {"effort": "low", "thinking": "adaptive"}),
    ("exclude,thinking=adaptive,effort=max", {"effort": "max", "exclude": True, "thinking": "adaptive"}),
])
def test_rm_the_grammar_takes_thinking_adaptive_in_canonical_order(text, expected):
    """Check that rm the grammar takes thinking adaptive in canonical order."""
    spec = count_proposer.parse_reasoning_spec(text)
    assert spec == expected and list(spec) == list(expected)
    assert count_proposer.normalize_reasoning(dict(reversed(list(spec.items())))) == expected


@pytest.mark.parametrize("text", [
    "thinking=adaptive", "max_tokens=5,thinking=adaptive", "thinking=high,effort=low",
    "effort=low,thinking", "effort=low,thinking=adaptive,thinking=adaptive",
])
def test_rm_the_grammar_refuses_a_thinking_field_off_the_rule(text):
    """Check that rm the grammar refuses a thinking field off the rule."""
    with pytest.raises(ValueError):
        count_proposer.parse_reasoning_spec(text)


@pytest.mark.parametrize("provider, spec, message", [
    ("openai", {"effort": "high"}, None), ("openai", {"max_tokens": 5}, "effort=<level> only"),
    ("anthropic", {"effort": "high", "thinking": "adaptive"}, None),
    ("anthropic", {"effort": "none"}, "none and minimal"),
    ("anthropic", {"effort": "high", "exclude": True}, "OpenRouter"),
    ("openai_compat", {"max_tokens": 5, "exclude": True}, None),
    ("openai_compat", {"effort": "high", "thinking": "adaptive"}, "Anthropic field"),
    ("mock", {"effort": "high"}, "unsupported provider"),
])
def test_rm_the_provider_rule_is_one_table(provider, spec, message):
    """Check that rm the provider rule is one table."""
    if message is None:
        assert count_proposer.validate_provider_reasoning(provider, spec) == spec
    else:
        with pytest.raises(ValueError, match=message):
            count_proposer.validate_provider_reasoning(provider, spec)
    assert count_proposer.validate_provider_reasoning(provider, None) is None


def test_rm_the_option_plumbing_hands_the_reasoning_to_every_provider():
    """Check that rm the option plumbing hands the reasoning to every provider."""
    args = SimpleNamespace(proposer="openai", llm_reasoning={"effort": "high"}, llm_max_tokens=4000,
                           llm_provider_order=["alpha"])
    assert count_proposer.compat_option_kwargs(args, compat=False) == {
        "max_tokens": 4000, "reasoning": {"effort": "high"}}
    assert count_proposer.compat_option_kwargs(args, compat=True) == {
        "max_tokens": 4000, "reasoning": {"effort": "high"}, "provider_order": ["alpha"]}
    assert count_proposer.compat_option_kwargs(SimpleNamespace(), compat=False) == {}
    values = count_proposer.compat_config(SimpleNamespace(proposer="anthropic", llm_reasoning={"effort": "high"}))
    assert values["llm_reasoning"] == {"effort": "high"} and "llm_provider" not in values
    with pytest.raises(ValueError, match="require proposer openai_compat"):
        count_proposer.compat_config(SimpleNamespace(proposer="openai", llm_base_url="https://host.invalid/v1"))


# --------------------------------------------------------------------- thinking summary (`test_ts_*`)
@pytest.mark.parametrize("text", [
    "effort=medium,thinking=adaptive,display=summarized",
    "display=summarized,thinking=adaptive,effort=medium",
    " thinking = adaptive , display = summarized , effort = medium ",
])
@pytest.mark.parametrize("exclude", [False, True])
def test_ts_display_is_canonical_for_text_and_dict(text, exclude):
    """Check that ts display is canonical for text and dict."""
    expected = {"effort": "medium"}
    if exclude:
        text = "exclude," + text
        expected["exclude"] = True
    expected.update(thinking="adaptive", display="summarized")
    spec = count_proposer.parse_reasoning_spec(text)
    assert spec == expected and list(spec) == list(expected)
    normalized = count_proposer.normalize_reasoning(dict(reversed(list(spec.items()))))
    assert normalized == expected and list(normalized) == list(expected)


@pytest.mark.parametrize("text", [
    "display=summarized", "effort=medium,display=summarized",
    "effort=medium,thinking=adaptive,display=omitted",
    "effort=medium,thinking=adaptive,display=updates",
    "effort=medium,thinking=adaptive,display=summarized,display=summarized",
    "max_tokens=512,thinking=adaptive,display=summarized",
    "effort=medium,thinking=adaptive,display",
])
def test_ts_display_refuses_invalid_grammar(text):
    """Check that ts display refuses invalid grammar."""
    with pytest.raises(ValueError):
        count_proposer.parse_reasoning_spec(text)


@pytest.mark.parametrize("provider", ["openai", "openai_compat"])
def test_ts_non_anthropic_display_is_refused_before_client_or_output(tmp_path, provider):
    """Check that ts non anthropic display is refused before client or output."""
    options = {"reasoning": "effort=medium,thinking=adaptive,display=summarized"}
    proposer_type = OpenAICountProposer
    if provider == "openai_compat":
        proposer_type = count_proposer.OpenAICompatCountProposer
        options.update(base_url="https://synthetic.example/v1", model="synthetic-model")
    with pytest.raises(ValueError, match="display= is an Anthropic field"):
        proposer_type(tmp_path / "refused", client=object(), **options)
    assert not (tmp_path / "refused").exists()


def test_ts_body_and_decoding_record_the_same_display_in_order(tmp_path):
    """Check that ts body and decoding record the same display in order."""
    spec = count_proposer.parse_reasoning_spec("display=summarized,effort=medium,thinking=adaptive")
    assert count_proposer.validate_provider_reasoning("anthropic", spec) == spec
    client, calls = anthropic_client([one_object()])
    proposer = AnthropicCountProposer(tmp_path / "display", client=client, reasoning=spec)
    _, record = proposer.propose_one('{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    expected = {"type": "adaptive", "display": "summarized"}
    assert calls[0]["thinking"] == expected
    assert list(calls[0]["thinking"]) == ["type", "display"]
    assert record["decoding"]["thinking"] == expected
    assert list(record["decoding"]["thinking"]) == ["type", "display"]
    assert record["decoding"] == count_proposer.decoding_identity(1600, spec, provider="anthropic")
    request = json.loads((tmp_path / "display/round_07.request.json").read_text())
    assert request["bodies"][0] == calls[0]
    assert request["identity"]["decoding"] == record["decoding"]


@pytest.mark.parametrize("registered_display", [False, True])
@pytest.mark.parametrize("stated_display", [False, True])
def test_ts_production_binds_display_presence_in_both_directions(
    tmp_path, registered_display, stated_display,
):
    """Check that ts production binds display presence in both directions."""
    registered = {"effort": "medium", "thinking": "adaptive"}
    stated = dict(registered)
    if registered_display:
        registered["display"] = "summarized"
    if stated_display:
        stated["display"] = "summarized"
    receipt = synthetic_manifest(tmp_path, {"llm_reasoning": registered})
    if registered_display == stated_display:
        assert count_proposer.require_registered_decoding({"llm_reasoning": stated}, receipt, "core") is None
    else:
        with pytest.raises(SystemExit) as refusal:
            count_proposer.require_registered_decoding({"llm_reasoning": stated}, receipt, "core")
        assert refusal.value.code == 2


# --------------------------------------------------------------------- JSON mode (`test_em_*`)
def test_em_json_mode_adds_response_format_on_the_stock_path_only(tmp_path):
    """Check that em json mode adds response format on the stock path only."""
    client, calls = scripted_openai([openai_reply(one_object())])
    proposer = OpenAICountProposer(tmp_path / "json", client=client, response_format="json_object")
    _, record = proposer.propose_one('{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert list(calls[0]) == ["model", "max_tokens", "messages", "response_format"]
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert record["decoding"] == {"temperature": "not_sent/provider_default", "max_tokens": 1600,
                                  "response_format": {"type": "json_object"}}
    stored = json.loads((tmp_path / "json" / "round_07.request.json").read_text())
    assert stored["bodies"][0] == calls[0]
    # With a reasoning spec the reasoning-model shape carries it too, and so does the P4 burst.
    client2, calls2 = scripted_openai([openai_reply(one_object())])
    OpenAICountProposer(tmp_path / "json_effort", client=client2, reasoning="effort=low",
                        response_format="json_object").propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert set(calls2[0]) == {*REASONING_MODEL_KEYS, "response_format"}
    fixture_proposer, burst_calls, context, public, pool = openai_fixture(tmp_path)
    burst = OpenAICountProposer(tmp_path / "json_burst", client=fixture_proposer.client,
                                response_format="json_object")
    _, metadata = burst.propose(public, pool, context=context, initial_family="binomial")
    assert burst_calls[0]["response_format"] == {"type": "json_object"}
    assert metadata["decoding"]["response_format"] == {"type": "json_object"}
    # Unstated: no key anywhere (the RM byte pins above cover the body).
    client3, calls3 = scripted_openai([openai_reply(one_object())])
    _, plain = OpenAICountProposer(tmp_path / "plain", client=client3).propose_one(
        '{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert "response_format" not in calls3[0] and "response_format" not in plain["decoding"]


def test_em_json_mode_is_refused_by_the_other_transports_and_by_a_mock_run(tmp_path):
    """Check that em json mode is refused by the other transports and by a mock run."""
    with pytest.raises(ValueError, match="openai_compat does not take it"):
        count_proposer.OpenAICompatCountProposer(
            tmp_path / "compat", base_url="https://compat.example.invalid/api/v1",
            key_name="SYNTHETIC_COMPAT_CREDENTIAL", model="synthetic/vendor-edition", client=object(),
            response_format="json_object")
    with pytest.raises(ValueError, match="anthropic takes no response format"):
        AnthropicCountProposer(tmp_path / "anthropic", client=object(), response_format="json_object")
    with pytest.raises(ValueError, match="llm-response-format must be one of"):
        OpenAICountProposer(tmp_path / "bad", client=object(), response_format="json_schema")
    assert not any((tmp_path / name).exists() for name in ("compat", "anthropic", "bad"))
    with pytest.raises(ValueError, match="requires an LLM proposer"):
        count_proposer.compat_config(SimpleNamespace(proposer="mock", llm_response_format="json_object"))
    values = count_proposer.compat_config(SimpleNamespace(proposer="openai", llm_response_format="json_object"))
    assert values["llm_response_format"] == "json_object"
    args = SimpleNamespace(llm_response_format="json_object")
    assert count_proposer.compat_option_kwargs(args, compat=False) == {"response_format": "json_object"}


def test_em_json_mode_refuses_before_any_call_when_no_message_carries_the_word_json(tmp_path, monkeypatch):
    """OpenAI's JSON mode requires the word JSON in the messages; the packaged prompt carries it
    ("JSON belief packet"), a synthetic system text without it is refused before the client is
    called."""
    monkeypatch.setattr(count_proposer, "composed_system_prompt",
                        lambda *args, **kwargs: ("Propose one model as an object.",
                                                 {"prompt_version": "synthetic", "prompt_sha256": "0" * 64}))
    client, calls = scripted_openai([openai_reply(one_object())])
    proposer = OpenAICountProposer(tmp_path / "noword", client=client, response_format="json_object")
    with pytest.raises(ValueError, match="requires the word JSON"):
        proposer.propose_one('{"round":7}', context=P1_CONTEXT, initial_family="binomial")
    assert calls == [] and not list((tmp_path / "noword").iterdir())
    assert "JSON" in count_proposer.SYSTEM
