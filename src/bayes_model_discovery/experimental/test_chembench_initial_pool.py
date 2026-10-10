"""ChemBench initial-pool synthetic waves; no provider calls."""

import hashlib
import json
import sys
import threading
import time

import pytest

from bayes_model_discovery.experimental import chembench_core as cc
from bayes_model_discovery.experimental import count_proposer
from bayes_model_discovery.experimental import test_chembench_core as old
from bayes_model_discovery.experimental.count_models import Hypothesis, judge_candidate
from bayes_model_discovery.experimental.test_chembench_mda_options import stub_fits
from bayes_model_discovery.experimental.test_neuron_proposer import Client, response


def _json(path):
    return json.loads(path.read_text())


def _wave_entry(output, wave, calls, accepted):
    """One `waves[]` entry with its eleven keys, for a wave whose packet
    showed an empty pool (catalogue off, nothing admitted before it) and that wrote no duplicate,
    probe or fit refusal; the packet digest is read off the packet file the wave wrote."""
    packet = (output / "packets" / f"t00_burst_w{wave}.json").read_bytes()
    return {"wave": wave, "calls": calls, "accepted": accepted, "duplicates": 0,
            "rejected_prior_probe": 0, "fit_failed": 0, "pool_visible": wave > 0,
            "paragraph": "burst_refill" if wave else "burst",
            "packet_digest": hashlib.sha256(packet).hexdigest(), "pool_size_shown": 0,
            "pool_digest_shown": hashlib.sha256(b"[]").hexdigest()}


def _candidate(index, names, role="signed"):
    return Hypothesis(f"b0+b1*tanh(f0z+{index}/20)", "lognormal",
                      (("b0", "intercept"), ("b1", role)), names)


class SyntheticProposer:
    """Synthetic proposer that answers each call with a candidate chosen by `choose`; no provider calls."""
    def __init__(self, choose):
        self.choose = choose
        self.calls = []
        self.attempts = 0
        self.lock = threading.Lock()

    def propose_one(self, packet_text, **kwargs):
        """Record the call, ask `choose` for a candidate, admit it, and return the candidates with a transport record."""
        with self.lock:
            self.calls.append((json.loads(packet_text), kwargs))
        time.sleep(0.01)
        names = kwargs["initial_structures"][0].input_names
        candidate = self.choose(kwargs["context"], names)
        record = {"calls": 0, "repairs": 0, "status": "parsed",
                  "rejected_candidates": [], "mock": True,
                  "api_request_attempted": False, "note": "synthetic transport"}
        if candidate is None:
            return [], record
        try:
            kwargs["admit"](candidate)
        except cc.PriorProbeRejection as exc:
            record.update(status="rejected", rejected_candidates=[{"reason": str(exc)}])
            return [], record
        return [candidate], record


def _wave(tmp_path, monkeypatch, choose=None, **overrides):
    stub_fits(monkeypatch)
    proposer = SyntheticProposer(choose or (lambda context, names:
        _candidate(context["burst_call_index"], names)))
    monkeypatch.setattr(count_proposer, "AnthropicCountProposer", lambda *a, **k: proposer)
    output = tmp_path / "cell"
    kwargs = {"cell": "100", "structure_proposals": "free", "rounds": 0,
              "initial_proposals": 10, "initial_catalogue": "off", "proposer": "anthropic",
              "allow_paid_api": True, "run_label": "development"}
    kwargs.update(overrides)
    cc.run(old._args(output, old._manifest(tmp_path), **kwargs))
    return output, proposer


def test_ten_parallel_initial_candidates_use_existing_pipeline_and_core_shape(tmp_path, monkeypatch):
    """Ten parallel initial candidates use the existing pipeline and the CORE record shape."""
    output, proposer = _wave(tmp_path, monkeypatch)
    burst = _json(output / "proposals/t00_burst.json")
    assert burst["requested"] == 10 and burst["initial_catalogue"] == "off"
    assert burst["model_id_schema"] == cc.MODEL_ID_SCHEMA
    assert burst["before_ids"] == [] and len(burst["after_ids"]) == 10
    # A wave entry carries eleven keys; K is filled, so there is no refill wave.
    assert burst["waves"] == [_wave_entry(output, 0, 10, 10)]
    assert 1 < burst["max_in_flight"] <= cc.DEFAULT_BURST_CONCURRENCY
    assert burst["fit_attempt_count"] == burst["fit_success_count"] == 10
    assert all(set(row) == {"expression", "family", "parameters"}
               for row in burst["raw_response"])
    assert set(burst["after_ids"]) == {row["model_id"] for row in burst["events"]}
    assert len(list((output / "fits").glob("t00_*.json"))) == 10
    assert _json(output / "frozen_d0_policy.json")["models"] == []
    for packet, call in proposer.calls:
        assert packet["pool"] == []
        assert call["context"]["round"] == 0 and call["burst"] is True
        assert call["allowed_families"] == cc.CHEM_ALLOWED_FAMILIES
        assert call["repair_attempts"] == 1
    summary = _json(output / "summary.json")
    assert summary["proposal_accounting"]["initial_pool_size"] == 10
    assert summary["paid_api_calls"] == 0


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_real_transport_parser_with_injected_client_builds_ten_without_network(
        tmp_path, monkeypatch, provider):
    """The real transport parser with an injected client builds ten candidates without network access."""
    stub_fits(monkeypatch)
    candidates = [{"expression": f"b0+b1*tanh(f0z+{index}/20)", "family": "lognormal",
                   "parameters": {"b0": "intercept", "b1": "signed"}} for index in range(10)]
    client = Client(provider, [response(provider, candidate) for candidate in [*candidates, candidates[0]]])
    name = "AnthropicCountProposer" if provider == "anthropic" else "OpenAICountProposer"
    original = getattr(count_proposer, name)
    monkeypatch.setattr(count_proposer, name, lambda records, **kwargs: original(records, client=client))
    output = tmp_path / "parsed"
    cc.run(old._args(output, old._manifest(tmp_path), cell="100", structure_proposals="free",
                    rounds=1, initial_proposals=10, initial_catalogue="off", proposer=provider,
                    allow_paid_api=True))
    burst = _json(output / "proposals/t00_burst.json")
    assert len(client.calls) == 11 and len(burst["after_ids"]) == 10
    assert all(row["transport"]["status"] == "parsed" for row in burst["transports"])
    receipt = _json(output / "initial_pool_receipt.json")
    assert receipt["calls"] == 10 and receipt["usage_total_complete"] is True
    assert receipt["usage_total"]["input_tokens"] == 110
    assert receipt["usage_total"]["output_tokens"] == 70


@pytest.mark.parametrize("role", ["positive", "shape"])
def test_burst_roles_and_ids_round_trip_through_shared_hypothesis(tmp_path, monkeypatch, role):
    """Burst roles and model ids round-trip through the shared hypothesis class."""
    output, proposer = _wave(tmp_path, monkeypatch,
                            lambda context, names: _candidate(0, names, role), initial_proposals=1)
    burst = _json(output / "proposals/t00_burst.json")
    event, = burst["events"]
    names = proposer.calls[0][1]["initial_structures"][0].input_names
    rebuilt = Hypothesis(event["expression"], event["family"],
                         tuple(event["parameters"].items()), names)
    assert dict(rebuilt.parameters) == {"b0": "intercept", "b1": role}
    assert rebuilt.model_id == event["model_id"] == burst["after_ids"][0]
    assert judge_candidate(event, "100", (), "lognormal", input_names=names,
                           allowed_families=cc.CHEM_ALLOWED_FAMILIES) == (True, None)


def test_existing_admit_refuses_bad_prior_before_fit_and_refills_once(tmp_path, monkeypatch):
    """The admit step refuses a bad prior before any fit, and the burst refills once."""
    def choose(context, names):
        """Return an invalid-prior candidate in wave 0 and a valid one afterwards."""
        if context["burst_wave"] == 0:
            return Hypothesis("b0-2000*square(f0-f1)", "lognormal",
                              (("b0", "intercept"),), names)
        return _candidate(0, names)
    output, proposer = _wave(tmp_path, monkeypatch, choose, initial_proposals=1)
    burst = _json(output / "proposals/t00_burst.json")
    # Eleven keys per wave; the refill packet shows the (still empty) pool
    # under the refill paragraph, and the refused call is a transport rejection, not an event.
    assert burst["waves"] == [_wave_entry(output, 0, 1, 0), _wave_entry(output, 1, 1, 1)]
    assert burst["fit_attempt_count"] == 1
    archive = proposer.calls[1][0]["archive"]
    assert any(row["outcome"] == "rejected" and "non-finite/nonpositive mean"
               in row["reason"]["validation_message"] for row in archive)


def test_catalogue_identity_detects_duplicate_in_grammar_space(tmp_path, monkeypatch):
    """Catalogue identity detects a duplicate in grammar space."""
    output, _ = _wave(tmp_path, monkeypatch,
        lambda context, names: Hypothesis("b0", "lognormal", (("b0", "intercept"),), names),
        initial_proposals=1, initial_catalogue="on")
    burst = _json(output / "proposals/t00_burst.json")
    # Wave 0 and the two refill waves each ask for the one missing
    # candidate and each gets the catalogue duplicate, so three duplicates, not two.
    assert [(e["calls"], e["duplicates"]) for e in burst["waves"]] == [(1, 1), (1, 1), (1, 1)]
    assert burst["duplicate_count"] == 3 and burst["fit_attempt_count"] == 0
    assert len(burst["after_ids"]) == 4 and burst["admitted_ids"] == []


def test_duplicate_survivor_events_remain_visible_for_strict_reader_refusal(tmp_path, monkeypatch):
    """Duplicate survivor events stay visible so a strict reader can refuse the file."""
    output, _ = _wave(tmp_path, monkeypatch,
                      lambda context, names: _candidate(0, names), initial_proposals=2)
    burst = _json(output / "proposals/t00_burst.json")
    survivor, = burst["after_ids"]
    matching = [event for event in burst["events"] if event["model_id"] == survivor]
    # The strict reader refuses multiple event bindings. Do not silently deduplicate
    # this transcript to make a duplicate-containing generated file pass that gate.
    # Wave 0's second call and both refill waves' calls are duplicates.
    assert len(matching) == 4
    assert [event["status"] for event in matching] == [
        "fitted", "duplicate_not_refitted", "duplicate_not_refitted", "duplicate_not_refitted"]


def test_midc_commuted_proposals_share_identity_without_a_second_fit(tmp_path, monkeypatch):
    """Commuted proposals share one model identity and are fitted only once."""
    def choose(context, names):
        """Return two commuted spellings of the same model for the first and later calls."""
        expression = "b0+b1*tanh(f0z)" if context["burst_call_index"] == 0 else "b1*tanh(f0z)+b0"
        return Hypothesis(expression, "lognormal", (("b0", "intercept"), ("b1", "signed")), names)
    output, _ = _wave(tmp_path, monkeypatch, choose, initial_proposals=2)
    burst = _json(output / "proposals/t00_burst.json")
    assert burst["model_id_schema"] == "ast_roles_family_ac_v2"  # model-id schema ac_v2
    assert len(burst["after_ids"]) == burst["fit_attempt_count"] == 1
    assert burst["events"][0]["expression"] != burst["events"][1]["expression"]
    assert burst["events"][0]["model_id"] == burst["events"][1]["model_id"]


def test_round_twenty_one_mock_refusal_precedes_wave_and_outputs(tmp_path):
    """A mock run asking for round 21 is refused before any wave or output."""
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="fixed at 20 rounds"):
        cc.run(old._args(output, old._manifest(tmp_path), cell="100", structure_proposals="free",
                        rounds=21, initial_proposals=10, initial_catalogue="off"))
    assert cc.MOCK_SCHEDULE_ROUNDS == 20 and not output.exists()


def test_mock_ten_uses_existing_schedule_without_new_templates_or_horizon(tmp_path, monkeypatch):
    """The mock ten-candidate burst uses the existing schedule with no new templates or horizon."""
    stub_fits(monkeypatch)
    output = tmp_path / "mock"
    cc.run(old._args(output, old._manifest(tmp_path), cell="100", structure_proposals="free",
                    rounds=1, initial_proposals=10, initial_catalogue="off"))
    burst = _json(output / "proposals/t00_burst.json")
    expected = cc._mock_schedule(old.FAKE_NAMES, cc.CHEM_ALLOWED_FAMILIES)
    assert [(event["expression"], event["family"]) for event in burst["events"][:10]] == list(expected[:10])
    assert len(burst["after_ids"]) == 2
    # Wave 0 admits two of ten, and each of the two refill waves asks for
    # the eight still missing: 10 + 8 + 8 calls, one event each.
    assert [wave["calls"] for wave in burst["waves"]] == [10, 8, 8]
    assert sum(wave["calls"] for wave in burst["waves"]) == len(burst["events"]) == 26
    later = _json(output / "proposals/t01.json")
    assert later["proposed_ids"] == [burst["events"][0]["model_id"]]
    assert len((output / "observations.jsonl").read_text().splitlines()) == 5


def test_generation_receipt_binds_survivors_and_file_without_mutating_preflight(tmp_path, monkeypatch):
    """The generation receipt binds the survivors and the pool file without mutating the preflight."""
    output, _ = _wave(tmp_path, monkeypatch,
                      lambda context, names: _candidate(0, names), initial_proposals=2)
    config = _json(output / "config.json")
    receipt = _json(output / "initial_pool_receipt.json")
    pool_hash = hashlib.sha256((output / "proposals/t00_burst.json").read_bytes()).hexdigest()
    assert config["model_id_schema"] == receipt["model_id_schema"] == cc.MODEL_ID_SCHEMA
    for record in (config, receipt):
        assert record["initial_pool_source"] == "generated"
        assert record["initial_pool_count"] == 1
        assert record["initial_pool_sha256"] == pool_hash
    assert config["initial_pool"] == 1
    assert config["initial_pool_receipt"]["sha256"] == hashlib.sha256(
        (output / "initial_pool_receipt.json").read_bytes()).hexdigest()
    settings = config["label_receipt"]["binding"]["settings"]
    assert settings["model_id_schema"] == cc.MODEL_ID_SCHEMA
    assert settings["initial_proposals"] == receipt["requested"] == 2
    assert settings["initial_pool_source"] == "generated"
    assert "initial_pool_count" not in settings and "initial_pool_sha256" not in settings
    assert cc.recheck_manifest(config["label_receipt"]) == config["label_receipt"]
    assert receipt["calls"] == receipt["repairs"] == 0
    assert receipt["usage_total"] == {"input_tokens": 0, "output_tokens": 0}
    assert receipt["usage_total_complete"] is True


@pytest.mark.parametrize("tokens, expected", [([], {}), (
    ["--initial-proposals", "10", "--initial-catalogue", "off"],
    {"initial_proposals": 10, "initial_catalogue": "off"})])
def test_cli_new_flags_are_optional(monkeypatch, tokens, expected):
    """The new command-line flags are optional."""
    captured = []
    monkeypatch.setattr(cc, "run", captured.append)
    monkeypatch.setattr(sys, "argv", ["chembench_core", "--manifest", "/synthetic.json",
                                     "--output", "/synthetic", "--noise", "0.01", *tokens])
    cc.main()
    assert {key: getattr(captured[0], key) for key in ("initial_proposals", "initial_catalogue")
            if hasattr(captured[0], key)} == expected


@pytest.mark.parametrize("requested", [11, 12])
def test_mock_refill_wraps_existing_schedule_without_reset_or_new_cap(tmp_path, monkeypatch, requested):
    """Mock refill wraps the existing schedule without a reset or a new cap."""
    stub_fits(monkeypatch)
    monkeypatch.setattr(cc, "_prior_probe", lambda *a, **k: (
        {"passed": False, "detail": "synthetic rejection to exercise refill wrap"}, None))
    output = tmp_path / "mock_wrap"
    cc.run(old._args(output, old._manifest(tmp_path), cell="100", structure_proposals="free",
                    rounds=0, initial_proposals=requested, initial_catalogue="on"))
    burst = _json(output / "proposals/t00_burst.json")
    schedule = cc._mock_schedule(old.FAKE_NAMES, cc.CHEM_ALLOWED_FAMILIES)
    calls = sum(wave["calls"] for wave in burst["waves"])
    assert calls > 20
    assert [wave["wave"] for wave in burst["waves"]] == [0, 1, 2]  # three waves
    assert [(event["expression"], event["family"]) for event in burst["events"]] == [
        schedule[index % cc.MOCK_SCHEDULE_ROUNDS] for index in range(calls)]
    assert burst["shortfall_after_refill"] == requested - burst["fit_success_count"]
    assert cc.MOCK_SCHEDULE_ROUNDS == 20


def test_initial_wave_reuses_active_cap_and_records_retired_pool(tmp_path, monkeypatch):
    """The initial wave reuses the active-model cap and records the retired pool."""
    output, _ = _wave(tmp_path, monkeypatch, initial_proposals=12, initial_catalogue="on")
    burst = _json(output / "proposals/t00_burst.json")
    assert burst["fit_success_count"] == 12
    assert len(burst["before_ids"]) == 4
    assert len(burst["after_ids"]) == cc.ACTIVE_MAXIMUM == 12
    assert len(burst["retired_ids"]) == 4
    assert burst["shortfall_after_refill"] == 0
    assert _json(output / "summary.json")["proposal_accounting"]["retired_candidates"] == 4


def test_outcome_usage_includes_rate_limit_attempt_before_success(tmp_path, monkeypatch):
    """Outcome usage includes a rate-limited attempt made before the successful call."""
    stub_fits(monkeypatch)
    proposer = SyntheticProposer(lambda context, names: _candidate(0, names))
    original = proposer.propose_one
    def retrying(packet_text, **kwargs):
        """Fail the first attempt with a rate-limit record, then delegate to the original proposer."""
        if not kwargs["context"].get("burst_retry"):
            return [], {"calls": 1, "repairs": 0, "status": "failed",
                        "rejected_candidates": [{"http_status": 429}],
                        "usage_total": {"input_tokens": 3, "output_tokens": 2},
                        "usage_total_complete": True, "responses": [{}]}
        candidates, record = original(packet_text, **kwargs)
        record.update(calls=1, usage_total={"input_tokens": 5, "output_tokens": 7},
                      usage_total_complete=True, responses=[{}])
        return candidates, record
    monkeypatch.setattr(proposer, "propose_one", retrying)
    monkeypatch.setattr(count_proposer, "AnthropicCountProposer", lambda *a, **k: proposer)
    monkeypatch.setattr(cc.time, "sleep", lambda _: None)
    output = tmp_path / "retry"
    cc.run(old._args(output, old._manifest(tmp_path), cell="100", structure_proposals="free",
                    rounds=0, initial_proposals=1, initial_catalogue="off", proposer="anthropic",
                    allow_paid_api=True))
    receipt = _json(output / "initial_pool_receipt.json")
    assert receipt["calls"] == 2 and receipt["usage_total_complete"] is True
    assert receipt["usage_total"] == {"input_tokens": 8, "output_tokens": 9}
    assert [attempt["retry"] for attempt in receipt["attempts"]] == [0, 1]
    assert _json(output / "summary.json")["proposal_accounting"]["llm_calls"] == 2


@pytest.mark.parametrize("options", [
    {"initial_proposals": 0}, {"initial_proposals": 25},
    {"initial_proposals": cc.ACTIVE_MAXIMUM + 1}, {"initial_catalogue": "off"},
])
def test_invalid_wave_options_fail_before_output(tmp_path, options):
    """Invalid wave options fail before any output is written."""
    output = tmp_path / "refused"
    with pytest.raises(ValueError):
        cc.run(old._args(output, old._manifest(tmp_path), cell="100",
                        structure_proposals="free", **options))
    assert not output.exists()
