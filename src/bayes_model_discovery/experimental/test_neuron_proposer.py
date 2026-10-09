"""Synthetic transport/packet probes only. No fitting and no live API clients."""
import copy
import hashlib
import inspect
import json
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from ..provenance.canonical import is_hex64
from . import count_proposer as count
from . import neuron_forward as forward
from . import neuron_proposer as p

PRIVATE = "synthetic-private-target-never-in-packet"
WINDOWS = {"E_mV": [-150, 150], "mvh_mV": [-150, 100], "mk_mV": [-100, 100],
           "mtau_ms": [0.1, 500], "hvh_mV": [-150, 100], "hk_mV": [-100, 100],
           "htau_ms": [0.1, 1000]}
DESIGNS = [("synthetic_first", [(1.0, 1.0)]), ("synthetic_second", [(1.0, 2.0)])]


class PublicWorld:
    """Test helper class PublicWorld."""
    def __init__(self, hidden=PRIVATE):
        """Test double method __init__."""
        self.hidden = hidden

    def problem(self):
        """Test helper or fixture: problem."""
        return {"text_prior": "An unknown modification changes the response.",
                "reference_model": {"extra": [], "slow_na": False},
                "protocols": DESIGNS, "test_protocol_labels": ["public_test_label"],
                "budget_rule": "each protocol once", "ignored_hidden": self.hidden}

    def __getattr__(self, key):
        """Test double method __getattr__."""
        if key in ("test_protocols", "_truth_name", "_truth_kwargs", "discriminator"):
            raise AssertionError("private world access")
        raise AttributeError(key)


def observations():
    """Test helper or fixture: observations."""
    return [{"round": 0, "design_index": i, "label": design[0],
             "features_mV": [-60.123456, -70.0, -58.0], "spike_count": 1,
             "cost": 1, "test_start_observed": 2, "seed": PRIVATE,
             "held_out_targets": PRIVATE} for i, design in enumerate(DESIGNS)]


def packet(cell="100", fits=(), archive=()):
    """Test helper or fixture: packet."""
    return p.build_neuron_belief_packet(PublicWorld(), observations(), fits, archive, cell,
                                       parameter_windows=WINDOWS)


def text_packet(**kwargs):
    """Test helper or fixture: text packet."""
    return p.serialize_packet(packet(**kwargs))


def reply(mechanism="h_current", **extra):
    """Test helper or fixture: reply."""
    recipe = p.mechanism_vocabulary()[mechanism]
    return {"mechanism": mechanism, **copy.deepcopy(recipe), **extra}


def parse(value, *, constraints=None, known_ids=()):
    """Test helper or fixture: parse."""
    return p.parse_neuron_candidate(json.dumps(value), constraints=constraints or packet()["constraints"],
                                    initial_designs=DESIGNS, known_ids=known_ids)


@pytest.fixture
def probe(monkeypatch):
    """Test helper or fixture: probe."""
    calls = []
    def finite(hypothesis, theta, design):
        calls.append((hypothesis, theta, design))
        return np.array([-60.0, -70.0, -59.0])
    monkeypatch.setattr(forward, "predict_features", finite)
    return calls


def test_public_packet_is_deterministic_and_private_state_independent():
    """Check that public packet is deterministic and private state independent."""
    left = p.build_neuron_belief_packet(PublicWorld("one-secret"), observations(), [], [], "100",
                                       parameter_windows=WINDOWS)
    right = p.build_neuron_belief_packet(PublicWorld("another-secret"), observations(), [], [], "100",
                                        parameter_windows=WINDOWS)
    text = p.serialize_packet(left)
    assert text == p.serialize_packet(right)
    assert PRIVATE not in text and "one-secret" not in text and "another-secret" not in text
    assert left["observations"][0]["features_mV"][0] == -60.12
    assert left["interface"]["test_protocol_labels"] == ["public_test_label"]
    assert set(left) == {"interface", "observations", "pool", "archive", "constraints",
                         "prompt_version", "prompt_sha256"}
    assert p.packet_digest(text) == {"packet_bytes": len(text.encode()),
                                    "packet_sha256": hashlib.sha256(text.encode()).hexdigest()}
    forbidden = ("held_out", "evaluation", "score", "truth", "seed")
    assert not any(word in key for word in forbidden
                   for key in inspect.signature(p.build_neuron_belief_packet).parameters)


def test_pool_and_acquisition_allowlists_and_posterior_summaries():
    """Check that pool and acquisition allowlists and posterior summaries."""
    fit = NS(hypothesis=forward.HHHypothesis("leak"), log_evidence=-2.,
             particles={"conductance": np.array([1., 3.]), "sigma": np.array([2., 4.]),
                        "private": PRIVATE}, private=PRIVATE)
    got = p.build_neuron_belief_packet(PublicWorld(), observations(), [fit], [], "101",
        parameter_windows=WINDOWS, acquisition=[{"round": 1, "selected": 0, "available": [0, 1],
                                                "mode": "joint", "rows": PRIVATE}])
    member, = got["pool"]
    assert member["posterior"] == {"conductance": {"mean": 2., "std": 1.},
                                    "sigma": {"mean": 3., "std": 1.}}
    assert member["weight"] == 1.
    assert PRIVATE not in p.serialize_packet(got)
    assert set(got["acquisition"][0]) == {"round", "selected", "available", "mode"}


@pytest.mark.parametrize("cell", [f"{n:03b}" for n in range(8)])
def test_structure_bit_is_only_mode_and_family_always_inapplicable(cell):
    """Check that structure bit is only mode and family always inapplicable."""
    got = packet(cell)["constraints"]
    assert p.structure_mode(cell) == ("free" if cell[0] == "1" else "catalog")
    assert got["structure_expansion"] == (cell[0] == "1")
    assert got["family_expansion"] is False


def test_windows_must_be_explicit_and_initial_mechanisms_public():
    """Check that windows must be explicit and initial mechanisms public."""
    with pytest.raises(ValueError, match="explicit parameter_windows"):
        p.build_neuron_belief_packet(PublicWorld(), observations(), [], [], "100", parameter_windows={})
    broken = copy.deepcopy(WINDOWS)
    broken["mk_mV"] = [5, -5]
    with pytest.raises(ValueError, match="finite ordered"):
        p.build_neuron_belief_packet(PublicWorld(), observations(), [], [], "100", parameter_windows=broken)


@pytest.mark.parametrize("raw,stage,message", [
    ("not-json", "json", "reply is not valid JSON:"),
    ("[]", "shape", p.OBJECT_REFUSAL),
    ("null", "shape", p.NON_OBJECT_REFUSAL),
    ("17", "shape", p.NON_OBJECT_REFUSAL),
    ("{}", "shape", p.REQUIRED_KEYS_REFUSAL),
    (json.dumps({"mechanism": "unknown", "kind": "bad", "parameters": []}), "shape", p.PARAMETERS_REFUSAL),
    (json.dumps({"mechanism": "unknown", "kind": "bad", "parameters": {}}), "vocabulary", p.MECHANISM_VOCABULARY_REFUSAL_REASON),
])
def test_validation_precedence_json_shape_vocabulary(raw, stage, message, probe):
    """Check that validation precedence json shape vocabulary."""
    with pytest.raises(p.CandidateValidationError) as error:
        p.parse_neuron_candidate(raw, constraints=packet()["constraints"], initial_designs=DESIGNS)
    assert error.value.stage == stage
    assert str(error.value).startswith(message)
    assert probe == []


@pytest.mark.parametrize("change", ["kind", "missing_htau", "inactive_nonnull", "extra_key", "base_parameters"])
def test_kind_parameter_agreement_before_numeric_windows(change, probe):
    """Check that kind parameter agreement before numeric windows."""
    obj = reply("rebound_inward_slow")
    if change == "kind": obj["kind"] = "base_conductance"
    elif change == "missing_htau": del obj["parameters"]["htau_ms"]
    elif change == "inactive_nonnull": obj["parameters"]["hvh_mV"] = None
    elif change == "extra_key": obj["parameters"]["junk"] = 1
    else: obj = reply("leak"); obj["parameters"]["junk"] = 1
    with pytest.raises(p.CandidateValidationError, match=p.KIND_REFUSAL) as error:
        parse(obj)
    assert error.value.stage == "kind" and not probe


@pytest.mark.parametrize("key,value", [("mk_mV", 0), ("mtau_ms", 0), ("E_mV", float("inf")),
    ("mvh_mV", True), ("E_mV", {"value": 1}), ("hvh_mV", -200), ("htau_ms", None),
    ("hk_mV", 0), ("mpow", 3)])
def test_numeric_windows_precede_cell_and_duplicate_checks(key, value, probe):
    """Check that numeric windows precede cell and duplicate checks."""
    obj = reply("rebound_inward_slow")
    obj["parameters"][key] = value
    constraints = packet("000")["constraints"]
    with pytest.raises(p.CandidateValidationError, match=p.WINDOW_REFUSAL) as error:
        parse(obj, constraints=constraints)
    assert error.value.stage == "numeric" and not probe


def test_powers_are_integer_and_negative_activation_slope_is_valid(probe):
    """Check that powers are integer and negative activation slope is valid."""
    obj = reply()
    obj["parameters"]["mpow"] = 1.0
    with pytest.raises(p.CandidateValidationError, match=p.POWER_REFUSAL): parse(obj)
    candidate, _ = parse(reply())
    assert candidate.parameters["mk_mV"] < 0 and len(probe) == 2


def test_s0_restricts_each_candidate_but_f0_never_rejects(probe):
    """Check that s0 restricts each candidate but f0 never rejects."""
    constraints = packet("000")["constraints"]
    with pytest.raises(p.CandidateValidationError, match=p.S0_REFUSAL_REASON):
        parse(reply(), constraints=constraints)
    hyp, _ = parse(reply("leak"), constraints=constraints)
    assert hyp.model_id == forward.HHHypothesis("leak").model_id
    hyp, _ = parse(reply("rebound_inward_slow"))
    assert hyp.parameters["htau_ms"] == 130


def test_duplicate_precedes_finiteness_and_identity_includes_shape(probe):
    """Check that duplicate precedes finiteness and identity includes shape."""
    original, _ = parse(reply())
    probe.clear()
    with pytest.raises(p.CandidateValidationError, match=p.DUPLICATE_REFUSAL):
        parse(reply(), known_ids={original.model_id})
    assert not probe
    obj = reply(); obj["parameters"]["mvh_mV"] += 1
    changed, _ = parse(obj, known_ids={original.model_id})
    assert changed.model_id != original.model_id
    obj = reply(); del obj["parameters"]["hk_mV"]
    same, _ = parse(obj)
    assert same.model_id == original.model_id


def test_extra_root_keys_and_long_note_are_ignored(probe):
    """Check that extra root keys and long note are ignored."""
    hyp, metadata = parse(reply(note="x" * 201, synthetic_extra={"nested": 7}))
    assert metadata["ignored_keys"] == {"note": "x" * 201, "synthetic_extra": {"nested": 7}}
    assert hyp.model_id == parse(reply())[0].model_id
    _, metadata = parse(reply(note="short synthetic note"))
    assert metadata["note"] == "short synthetic note"


def test_finiteness_probe_calls_existing_forward_on_two_initial_designs(probe):
    """Check that finiteness probe calls existing forward on two initial designs."""
    hyp, _ = parse(reply())
    assert [tuple(call[2]) for call in probe] == DESIGNS
    assert all(call[0] is hyp and call[1] == [1.] for call in probe)


def test_divergent_forward_is_rejected_before_any_fit(monkeypatch):
    """Check that divergent forward is rejected before any fit."""
    monkeypatch.setattr(forward, "predict_features", lambda *args: np.array([0., np.inf, 0.]))
    with pytest.raises(p.CandidateValidationError, match=p.FINITENESS_REFUSAL) as error:
        parse(reply())
    assert error.value.stage == "finiteness"
    assert "fit(" not in inspect.getsource(p.finiteness_probe)


@pytest.mark.usefixtures("require_neuronbench_source")
def test_native_forward_adapter_preserves_all_inactivation_parameters(monkeypatch):
    """Check that native forward adapter preserves all inactivation parameters."""
    from neuronbench import worlds
    seen = []
    def simulate(current, start, **kwargs):
        seen.append(kwargs)
        return 0, np.full(len(current), -65.)
    monkeypatch.setattr(worlds, "simulate", simulate)
    obj = reply("rebound_inward_slow"); obj["parameters"]["hpow"] = 2
    _hyp, _ = parse(obj)
    assert len(seen) == 2
    channel = seen[0]["extra"][0]
    assert (channel.E, channel.mvh, channel.mk, channel.mtau, channel.mpow,
            channel.hvh, channel.hk, channel.htau, channel.hpow) == tuple(obj["parameters"].values())
    assert channel.g == 1. and set(seen[0]) == {"extra", "trace"}


@pytest.mark.usefixtures("require_neuronbench_source")
def test_real_native_forward_probe_without_fitting():
    """Check that real native forward probe without fitting."""
    hyp, _ = parse(reply())
    assert np.isfinite(forward.predict_features(hyp, [1.], DESIGNS[0])).all()


class Client:
    """Test helper class Client."""
    def __init__(self, provider, responses):
        """Test double method __init__."""
        self.provider, self.responses, self.calls = provider, iter(responses), []
        self.messages = self
        self.chat = NS(completions=self)

    def create(self, **body):
        """Test helper or fixture: create."""
        self.calls.append(copy.deepcopy(body))
        response = next(self.responses)
        if isinstance(response, Exception): raise response
        return response


def response(provider, body=None, *, usage=True, stop=None):
    """Test helper or fixture: response."""
    raw = json.dumps(reply() if body is None else body) if not isinstance(body, str) else body
    common = {"model": "synthetic_returned_model", "id": "synthetic_response_id",
              "_request_id": "synthetic_http_id"}
    if provider == "anthropic":
        return NS(**common, content=[NS(type="text", text=raw)], stop_reason=stop or "end_turn",
                  usage=NS(model_dump=lambda **kw: {"input_tokens": 11, "output_tokens": 7}) if usage else None)
    return NS(**common, choices=[NS(message=NS(content=raw), finish_reason=stop or "stop")],
              usage=NS(prompt_tokens=11, completion_tokens=7) if usage else None)


def context(round=1):
    """Test helper or fixture: context."""
    return {"run_id": "synthetic_transport_run", "world_seed": 123,
            "cell": "100", "proposal_rep": 0, "round": round}


def proposer(tmp_path, provider, responses, **kwargs):
    """Test helper or fixture: proposer."""
    client = Client(provider, responses)
    instance = p.NeuronProposer(tmp_path / "records", world="synthetic_world", cell="100",
                                provider=provider, client=client, **kwargs)
    return instance, client


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_injected_clients_success_and_provenance(tmp_path, provider, probe):
    """Check that injected clients success and provenance."""
    instance, client = proposer(tmp_path, provider, [response(provider)])
    candidate, record = instance.propose_one(text_packet(), context=context())
    assert candidate is not None and record["status"] == "parsed"
    assert record["calls"] == instance.attempts == 1
    assert instance.paid_api_calls == record["paid_api_calls"] == 0
    assert record["client_injected"] and record["api_request_attempted"]
    assert record["prompt_sha256"] == hashlib.sha256(p.PROMPT_FILE.read_bytes()).hexdigest()
    assert record["prompt_version"] == "v3.0"
    assert record["world"] == "synthetic_world" and record["recovery_mode"] == "freeform_discovery"
    assert record["usage_total"]["input_tokens"] == 11 and record["usage_total_complete"]
    assert record["responses"][0]["http_request_id"] == "synthetic_http_id"
    assert record["mechanism_vocabulary_sha256"] == p.packet_digest(p.serialize_packet(
        packet()["constraints"]["allowed_mechanisms"]))["packet_sha256"]
    assert "temperature" not in client.calls[0]
    assert record["decoding"]["temperature"] == "not_sent/provider_default"
    assert "component" not in record
    files = sorted(instance.records.iterdir())
    assert [f.name for f in files] == ["round_01.request.json", "round_01.response.json"]
    assert json.loads(files[1].read_text()) == record


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_one_repair_uses_same_conversation_and_exact_error(tmp_path, provider, probe):
    """Check that one repair uses same conversation and exact error."""
    instance, client = proposer(tmp_path, provider, [response(provider, []), response(provider)])
    candidate, record = instance.propose_one(text_packet(), context=context())
    assert candidate and record["calls"] == 2 and record["repairs"] == 1
    assert record["usage_total"]["input_tokens"] == 22
    turns = client.calls[1]["messages"]
    assert turns[-2] == {"role": "assistant", "content": "[]"}
    assert turns[-1]["content"] == p.REPAIR_INSTRUCTION.format(error=p.OBJECT_REFUSAL)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_two_contract_failures_leave_pool_unchanged(tmp_path, provider, probe):
    """Check that two contract failures leave pool unchanged."""
    instance, _client = proposer(tmp_path, provider, [response(provider, []), response(provider, [])])
    before = text_packet()
    candidate, record = instance.propose_one(before, context=context())
    assert candidate is None and record["status"] == "proposal_failed"
    assert record["calls"] == 2 and len(record["rejected_candidates"]) == 2
    assert not record["silent_fallback"] and text_packet() == before


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("kind", ["api", "object", "usage"])
def test_transport_or_missing_usage_never_spends_a_repair(tmp_path, provider, kind, probe):
    """Check that transport or missing usage never spends a repair."""
    bad = (ValueError("synthetic API failure") if kind == "api" else NS() if kind == "object"
           else response(provider, [], usage=False))
    instance, _client = proposer(tmp_path, provider, [bad])
    candidate, record = instance.propose_one(text_packet(), context=context())
    assert candidate is None and record["calls"] == 1 and record["repairs"] == 0
    assert not record["rejected_candidates"][0]["repair_eligible"]
    if kind == "usage":
        assert not record["usage_total_complete"] and record["usage_missing_indices"] == [0]


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_valid_reply_with_unknown_usage_is_accepted_without_retry(tmp_path, provider, probe):
    """Check that valid reply with unknown usage is accepted without retry."""
    instance, _client = proposer(tmp_path, provider, [response(provider, usage=False)])
    candidate, record = instance.propose_one(text_packet(), context=context())
    assert candidate and record["calls"] == 1 and not record["usage_total_complete"]
    assert record["usage_missing_indices"] == [0]


def test_vocabulary_refusal_archives_constant_and_adds_known_identity(tmp_path, probe):
    """Check that vocabulary refusal archives constant and adds known identity."""
    obj = {"mechanism": "synthetic_outside_vocabulary", "kind": "extra_channel", "parameters": {}}
    instance, _client = proposer(tmp_path, "anthropic", [response("anthropic", obj)] * 2)
    candidate, record = instance.propose_one(text_packet(), context=context())
    assert candidate is None and len(record["archive_entries"]) == 2
    for row in record["archive_entries"]:
        assert row["outcome"] == "rejected"
        assert row["reason"] == {"validation_message": p.MECHANISM_VOCABULARY_REFUSAL_REASON}
        assert row["model_id"] in instance.known_ids


def test_duplicate_archive_retirement_note_and_probe_not_called(tmp_path, probe):
    """Check that duplicate archive retirement note and probe not called."""
    hyp, _ = parse(reply()); probe.clear()
    retired = {"model_id": hyp.model_id, "mechanism": hyp.mechanism,
               "outcome": "retired_by_evidence", "reason": {
                   "log_evidence_at_retirement": -3., "gap_to_cutoff_nats": 1.,
                   "active_rounds": [0], "computed_round": 1, "n_obs": 2}}
    instance, _client = proposer(tmp_path, "anthropic", [response("anthropic")] * 2)
    candidate, record = instance.propose_one(text_packet(archive=[retired]), context=context())
    assert candidate is None and not probe
    for row in record["archive_entries"]:
        assert row["outcome"] == "duplicate"
        assert set(row["reason"]) == p.ARCHIVE_REASON_KEYS["duplicate"]
        assert row["note"] == "refused: matched a model retired by evidence"


def test_finiteness_refusal_enters_rejected_archive(tmp_path, monkeypatch):
    """Check that finiteness refusal enters rejected archive."""
    monkeypatch.setattr(forward, "predict_features", lambda *args: np.full(3, np.nan))
    instance, _ = proposer(tmp_path, "anthropic", [response("anthropic")], repair_attempts=0)
    candidate, record = instance.propose_one(text_packet(), context=context())
    assert candidate is None
    assert record["archive_entries"][0]["reason"] == {"validation_message": p.FINITENESS_REFUSAL}


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_live_gate_uses_existing_authorization_sentence(tmp_path, provider):
    """Check that live gate uses existing authorization sentence."""
    name = "Anthropic" if provider == "anthropic" else "OpenAI"
    with pytest.raises(ValueError, match=f"{name} proposals require explicit --allow-paid-api authorization"):
        p.NeuronProposer(tmp_path / "records", world="synthetic", cell="100", provider=provider)
    assert not (tmp_path / "records").exists()


def test_s0_factory_constructs_nothing_and_direct_constructor_refuses(tmp_path):
    """Check that s0 factory constructs nothing and direct constructor refuses."""
    assert p.make_neuron_proposer("anthropic", tmp_path / "records", cell="000") is None
    with pytest.raises(ValueError, match="S0 forbids"):
        p.AnthropicNeuronProposer(tmp_path / "records", world="synthetic", cell="001", allow_live=True)
    assert not (tmp_path / "records").exists()


@pytest.mark.parametrize("repairs", [-1, 2, None, True])
def test_invalid_repair_count_refused_before_client_or_directory(tmp_path, repairs):
    """Check that invalid repair count refused before client or directory."""
    with pytest.raises(ValueError, match="repair_attempts must be 0 or 1"):
        p.NeuronProposer(tmp_path / "records", world="synthetic", cell="100", repair_attempts=repairs)
    assert not (tmp_path / "records").exists()


def test_context_rejects_component_and_repeat_record_before_new_call(tmp_path, probe):
    """Check that context rejects component and repeat record before new call."""
    instance, client = proposer(tmp_path, "anthropic", [response("anthropic")])
    with pytest.raises(ValueError, match="identity required"):
        instance.propose_one(text_packet(), context={**context(), "component": "y"})
    assert not client.calls
    instance.propose_one(text_packet(), context=context())
    before = {f.name: f.read_bytes() for f in instance.records.iterdir()}
    with pytest.raises(FileExistsError): instance.propose_one(text_packet(), context=context())
    assert len(client.calls) == 1
    assert {f.name: f.read_bytes() for f in instance.records.iterdir()} == before


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_masking_covers_capture_repair_returned_record_and_files(tmp_path, provider, probe):
    """Check that masking covers capture repair returned record and files."""
    secret = "sk-syntheticfixturecredential123"
    bad = {"mechanism": secret, "kind": "extra_channel", "parameters": {}}
    good = reply(note=secret, extra={secret: secret})
    instance, client = proposer(tmp_path, provider, [response(provider, bad), response(provider, good)])
    hyp, record = instance.propose_one(text_packet(), context=context())
    assert hyp and secret not in json.dumps(record)
    assert secret not in json.dumps(client.calls[1])
    assert all(secret not in f.read_text() for f in instance.records.iterdir())
    assert "<OPENAI_API_KEY:masked>" in json.dumps(record)


def test_prompt_is_fixed_general_text_and_count_helpers_are_reused():
    """Check that prompt is fixed general text and count helpers are reused."""
    assert hashlib.sha256(p.PROMPT_FILE.read_bytes()).hexdigest() == "8d578e80b4d6cb83cc16eaeb2dd5681e179f3e3efff04c6ad0264babab140ef8"
    text = p.PROMPT_FILE.read_text()
    forbidden = [PRIVATE, "synthetic_transport_run", "synthetic_world", "synthetic_first",
                 "z_rebound", "h_sag", "na_fatigue", "ca_rebound", "d_type", "textbook_M"]
    assert all(word not in text for word in forbidden)
    assert all(name not in text for name in p.mechanism_vocabulary())
    assert p._write_once is count._write_once
    assert p.mask_secrets is count.mask_secrets and p.mask_secrets_deep is count.mask_secrets_deep
    before = count.prompt_identity()
    p.neuron_prompt_identity()
    assert count.prompt_identity() == before


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_request_hash_is_of_the_masked_packet_actually_sent(tmp_path, provider, probe):
    """Check that request hash is of the masked packet actually sent."""
    packet_obj = packet()
    packet_obj["interface"]["text_prior"] = "sk-syntheticrequestcredential123"
    original = p.serialize_packet(packet_obj)
    instance, client = proposer(tmp_path, provider, [response(provider)])
    _, record = instance.propose_one(original, context=context())
    wire = next(turn["content"] for turn in client.calls[0]["messages"] if turn["role"] == "user")
    assert "sk-syntheticrequestcredential123" not in wire
    assert record["packet_sha256"] == hashlib.sha256(wire.encode()).hexdigest()
    assert record["packet_bytes"] == len(wire.encode())


def test_no_response_has_unknown_usage_instead_of_complete_zero(tmp_path):
    """Check that no response has unknown usage instead of complete zero."""
    instance, _ = proposer(tmp_path, "anthropic", [RuntimeError("synthetic unavailable")])
    _, record = instance.propose_one(text_packet(), context=context())
    assert record["responses"][0]["usage"] is None
    assert record["usage_total_complete"] is False
    assert record["usage_missing_indices"] == [0]
    assert record["calls"] == 1


def test_pool_accepts_a_fit_iterator_without_silently_dropping_candidates():
    """Check that pool accepts a fit iterator without silently dropping candidates."""
    fit = NS(hypothesis=forward.HHHypothesis("leak"), log_evidence=-2.,
             particles={"conductance": np.array([1., 3.]), "sigma": np.array([2., 4.])})
    assert packet(fits=iter([fit])) == packet(fits=[fit])


@pytest.mark.parametrize("cls", [p.AnthropicNeuronProposer, p.OpenAINeuronProposer])
def test_provider_aliases_select_the_correct_wire_contract(tmp_path, cls, probe):
    """Check that provider aliases select the correct wire contract."""
    provider = "anthropic" if cls is p.AnthropicNeuronProposer else "openai"
    client = Client(provider, [response(provider)])
    instance = cls(tmp_path / "records", world="synthetic", cell="100", client=client)
    candidate, record = instance.propose_one(text_packet(), context=context())
    assert candidate and record["provider"] == provider


@pytest.mark.parametrize("round_number", [0, 6, True, 1.5])
def test_context_horizon_is_validated_before_any_request(tmp_path, round_number):
    """Check that context horizon is validated before any request."""
    instance, client = proposer(tmp_path, "anthropic", [])
    with pytest.raises(ValueError, match="identity required"):
        instance.propose_one(text_packet(), context=context(round_number))
    assert not client.calls and list(instance.records.iterdir()) == []


@pytest.mark.parametrize("mechanism", ["h_current", "m_current", "inward"])
@pytest.mark.usefixtures("require_neuronbench_source")
def test_legacy_catalog_identity_preserves_the_exact_native_prediction(mechanism):
    """Check that legacy catalog identity preserves the exact native prediction."""
    recipe = reply(mechanism)
    candidate = p.NeuronHypothesis(mechanism, recipe["kind"],
                                   tuple(sorted(recipe["parameters"].items())))
    old = forward.HHHypothesis(mechanism)
    assert candidate.model_id == old.model_id
    assert np.array_equal(candidate.predict([1.25], DESIGNS[0]),
                          old.predict([1.25], DESIGNS[0]))


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_d1_empty_reply_repairs_with_one_user_turn_and_no_echo(tmp_path, provider, probe):
    """A body that is absent or blank is its own reply-contract
    failure, and the repair turn quotes nothing - an empty assistant turn is
    what both wire shapes refuse."""
    instance, client = proposer(tmp_path, provider, [response(provider, "   ", stop="length"),
                                                     response(provider)])
    candidate, record = instance.propose_one(text_packet(), context=context())
    assert candidate is not None and record["calls"] == 2 and record["repairs"] == 1
    rejected = record["rejected_candidates"][0]
    assert rejected["reason"] == "empty response body (finish_reason=length)"
    assert rejected["repair_kind"] == "empty_body" and rejected["repair_eligible"]
    messages = client.calls[1]["messages"]
    expected = ["user", "user"] if provider == "anthropic" else ["system", "user", "user"]
    assert [message["role"] for message in messages] == expected
    assert messages[-1]["content"] == p.REPAIR_INSTRUCTION_EMPTY.format(finish_reason="length")


def test_d1_stated_reasoning_and_budget_reach_the_compatible_body(tmp_path, probe):
    """Check that d1 stated reasoning and budget reach the compatible body."""
    instance, client = proposer(tmp_path, "openai_compat", [response("openai")],
                                base_url="https://compat.example.invalid/api/v1",
                                key_name="SYNTHETIC_COMPAT_CREDENTIAL", model="synthetic/edition",
                                reasoning="effort=none", max_tokens=4000)
    candidate, record = instance.propose_one(text_packet(), context=context())
    body = client.calls[0]
    assert candidate is not None and body["max_tokens"] == 4000
    assert body["extra_body"] == {"reasoning": {"effort": "none"}}
    assert record["decoding"] == {"temperature": "not_sent/provider_default",
                                 "max_tokens": 4000, "reasoning": {"effort": "none"}}


# ---------------------------------------------------------------------------
# Which observation features a packet carries is one descriptor
# (`FeatureSpace`) rather than three hardcoded millivolt names. The default
# descriptor is the earlier behaviour, pinned below by the serialized packet's
# own digest; a six-feature descriptor travels the same code with no branch on
# how many features there are or what they are called.
# ---------------------------------------------------------------------------

#: `serialize_packet(packet())` on the tree before the descriptor (exported with `git
#: archive` and hashed there). The default descriptor must still produce these exact bytes.
PRE_N2C_PACKET_SHA256 = "7025c5c404f5d1b40d3d324959242ec8fb77e0842f4401a18753b2e57ff51f56"
PRE_N2C_PACKET_BYTES = 2470
PRE_N2C_OBSERVATION_KEYS = ("round", "design_index", "label", "features_mV", "spike_count",
                            "cost", "test_start_observed")
PRE_N2C_OBSERVABLE = ["test-window mean mV", "overall minimum mV", "final mV"]
PRE_N2C_OBSERVATION_REFUSAL = "observation must identify one public protocol and three features"

#: The `paper_neuron` profile's descriptor, exactly as `neuron_core.proposer_feature_space`
#: builds it: the benchmark's six evaluation features, their scales and their forward.
EVAL6 = p.FeatureSpace(record_key="features", names=forward.EVAL_FEATURE_NAMES,
                       probe=forward.predict_eval_features,
                       feature_scales=forward.EVAL_FEATURE_SCALES)


def eval_observations(length=6):
    """Test helper or fixture: eval observations."""
    return [{"round": 0, "design_index": i, "label": design[0],
             "features": [float(value) for value in range(length)], "spike_count": 1,
             "cost": 1, "test_start_observed": 2, "seed": PRIVATE}
            for i, design in enumerate(DESIGNS)]


def stub_space(space, values):
    """The same descriptor with an injected probe, to drive the probe's own rules."""
    return replace(space, probe=lambda *args: np.asarray(values, dtype=float))


def test_n2c_the_default_descriptor_reproduces_the_pre_n2c_packet_byte_for_byte():
    """Check that n2c the default descriptor reproduces the pre n2c packet byte for byte."""
    text = text_packet()
    assert is_hex64(PRE_N2C_PACKET_SHA256)
    assert hashlib.sha256(text.encode()).hexdigest() == PRE_N2C_PACKET_SHA256
    assert p.packet_digest(text) == {"packet_bytes": PRE_N2C_PACKET_BYTES,
                                     "packet_sha256": PRE_N2C_PACKET_SHA256}
    assert p.OBSERVATION_KEYS == PRE_N2C_OBSERVATION_KEYS
    assert p.observation_keys(p.DEFAULT_FEATURE_SPACE) == PRE_N2C_OBSERVATION_KEYS
    assert p.DEFAULT_FEATURE_SPACE.message == PRE_N2C_OBSERVATION_REFUSAL
    assert p.DEFAULT_FEATURE_SPACE.record_key == "features_mV"
    assert list(p.DEFAULT_FEATURE_SPACE.names) == PRE_N2C_OBSERVABLE
    interface = packet()["interface"]
    assert interface["observable"] == PRE_N2C_OBSERVABLE
    assert interface["noise_unit_mV"] == 20.0 and "feature_scales" not in interface


def test_n2c_the_default_refusal_message_and_length_are_unchanged():
    """Check that n2c the default refusal message and length are unchanged."""
    short = observations()
    short[0]["features_mV"] = [-60.0, -70.0]
    with pytest.raises(ValueError, match=PRE_N2C_OBSERVATION_REFUSAL):
        p.build_neuron_belief_packet(PublicWorld(), short, [], [], "100",
                                     parameter_windows=WINDOWS)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_n2c_the_default_descriptor_leaves_the_transport_identity_untouched(
        tmp_path, provider, probe):
    """Check that n2c the default descriptor leaves the transport identity untouched."""
    instance, _client = proposer(tmp_path, provider, [response(provider)])
    candidate, record = instance.propose_one(text_packet(), context=context())
    assert candidate is not None and instance.feature_space == p.DEFAULT_FEATURE_SPACE
    assert "feature_space" not in record
    identity = json.loads((instance.records / "round_01.request.json").read_text())["identity"]
    assert "feature_space" not in identity


def test_n2c_a_six_feature_descriptor_publishes_its_own_names_scales_and_record_key():
    """Check that n2c a six feature descriptor publishes its own names scales and record key."""
    got = p.build_neuron_belief_packet(PublicWorld(), eval_observations(), [], [], "100",
                                       parameter_windows=WINDOWS, feature_space=EVAL6)
    interface = got["interface"]
    assert interface["observable"] == list(forward.EVAL_FEATURE_NAMES)
    assert len(interface["observable"]) == 6
    assert interface["feature_scales"] == list(forward.EVAL_FEATURE_SCALES)
    assert "noise_unit_mV" not in interface
    assert p.observation_keys(EVAL6) == ("round", "design_index", "label", "features",
                                         "spike_count", "cost", "test_start_observed")
    row = got["observations"][0]
    assert set(row) == set(p.observation_keys(EVAL6))
    assert len(row["features"]) == 6 and "features_mV" not in row
    assert PRIVATE not in p.serialize_packet(got)


@pytest.mark.parametrize("length", [5, 7])
def test_n2c_a_row_of_the_wrong_length_is_refused_by_the_declared_vector(length):
    """Check that n2c a row of the wrong length is refused by the declared vector."""
    with pytest.raises(ValueError, match=p.DECLARED_FEATURES_REFUSAL):
        p.build_neuron_belief_packet(PublicWorld(), eval_observations(length), [], [], "100",
                                     parameter_windows=WINDOWS, feature_space=EVAL6)


def test_n2c_a_row_from_the_other_feature_space_is_a_value_error_not_a_key_error():
    """Check that n2c a row from the other feature space is a value error not a key error."""
    with pytest.raises(ValueError, match=p.DECLARED_FEATURES_REFUSAL):
        p.build_neuron_belief_packet(PublicWorld(), observations(), [], [], "100",
                                     parameter_windows=WINDOWS, feature_space=EVAL6)
    with pytest.raises(ValueError, match=PRE_N2C_OBSERVATION_REFUSAL):
        p.build_neuron_belief_packet(PublicWorld(), eval_observations(), [], [], "100",
                                     parameter_windows=WINDOWS)


@pytest.mark.usefixtures("require_neuronbench_source")
def test_n2c_the_finiteness_probe_runs_the_descriptors_own_forward(monkeypatch):
    """Check that n2c the finiteness probe runs the descriptors own forward."""
    from neuronbench import worlds
    monkeypatch.setattr(worlds, "build_I", lambda segments: (np.zeros(30), 10))
    monkeypatch.setattr(worlds, "simulate",
                        lambda current, start, **kwargs: (0, np.full(len(current), -65.)))
    recipe = p.mechanism_vocabulary()["h_current"]
    candidate = p.NeuronHypothesis("h_current", recipe["kind"],
                                   tuple(sorted(recipe["parameters"].items())))
    # The real six-feature forward on a real (synthetic) trace: six finite values, accepted.
    assert p.finiteness_probe(candidate, DESIGNS, EVAL6) is None
    assert np.asarray(forward.predict_eval_features(candidate, [1.], DESIGNS[0])).shape == (6,)
    with pytest.raises(p.CandidateValidationError, match=p.FINITENESS_REFUSAL):
        p.finiteness_probe(candidate, DESIGNS,
                           stub_space(EVAL6, [0., np.inf, 0., 0., 0., 0.]))
    # Three finite values are the WRONG length for this descriptor, and refused as such.
    with pytest.raises(p.CandidateValidationError, match=p.FINITENESS_REFUSAL):
        p.finiteness_probe(candidate, DESIGNS, stub_space(EVAL6, [0., 0., 0.]))
    # The default descriptor is unchanged: three finite values pass, six do not.
    assert p.finiteness_probe(candidate, DESIGNS,
                              stub_space(p.DEFAULT_FEATURE_SPACE, [0., 0., 0.])) is None
    with pytest.raises(p.CandidateValidationError, match=p.FINITENESS_REFUSAL):
        p.finiteness_probe(candidate, DESIGNS,
                           stub_space(p.DEFAULT_FEATURE_SPACE, [0.] * 6))


def test_n2c_parse_and_the_transport_thread_the_descriptor_to_the_probe(tmp_path):
    """Check that n2c parse and the transport thread the descriptor to the probe."""
    seen = []

    def watched(candidate, theta, design):
        seen.append((candidate, theta, design))
        return np.zeros(6)

    space = replace(EVAL6, probe=watched)
    hypothesis, _meta = p.parse_neuron_candidate(json.dumps(reply()),
                                                 constraints=packet()["constraints"],
                                                 initial_designs=DESIGNS, feature_space=space)
    assert [tuple(call[2]) for call in seen] == DESIGNS
    assert all(call[0] is hypothesis and call[1] == [1.] for call in seen)
    instance, _client = proposer(tmp_path, "anthropic", [response("anthropic")],
                                 feature_space=space)
    candidate, record = instance.propose_one(text_packet(), context=context())
    assert candidate is not None and len(seen) == 4
    named = {"record_key": "features", "names": list(forward.EVAL_FEATURE_NAMES)}
    assert record["feature_space"] == named
    identity = json.loads((instance.records / "round_01.request.json").read_text())["identity"]
    assert identity["feature_space"] == named


def test_n2c_make_neuron_proposer_carries_the_descriptor_and_the_gate_is_unchanged(tmp_path):
    """Check that n2c make neuron proposer carries the descriptor and the gate is unchanged."""
    instance = p.make_neuron_proposer("anthropic", tmp_path / "free", cell="100",
                                      world="synthetic_world", client=Client("anthropic", []),
                                      feature_space=EVAL6)
    assert instance.feature_space is EVAL6
    assert p.make_neuron_proposer("anthropic", tmp_path / "catalog", cell="000",
                                  world="synthetic_world", client=Client("anthropic", []),
                                  feature_space=EVAL6) is None
