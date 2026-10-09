"""The round-0 initial burst (K independent single-candidate calls in
parallel, one refill wave for duplicates, no catalogue when asked) and the
active-pool cap.

Synthetic replies through an injected client, the label policy with synthetic
settings, and short BoxingGym runs driven in process (mock, and an injected
Anthropic client with the real sequential Monte Carlo (SMC)); no credentials, no network, no run text and
no results folder.
"""

import argparse
import hashlib
import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from . import count_proposer as cp
from .belief_packet import ARCHIVE_OPTIONAL_KEYS, _archive_entry, build_belief_packet
from .count_models import INITIAL_BURST_REFILL_POLICY
from .prior_probe_detail import PriorProbeRejection
from .run_label_policy import (
    BOXINGGYM_ACTIVE_MAXIMUM,
    BURST_REFILL_KEYS,
    BURST_REFILL_POLICY_KEY,
    bmd_runtime_settings,
)
from .test_run_label_policy import write_manifest

CANDIDATE = {"family": "binomial", "parameters": {"b0": "intercept", "b1": "signed"}}


def reply(expression):
    """Build a proposer reply payload for one expression, using the shared candidate fields."""
    return {"expression": expression, **CANDIDATE}


# --------------------------------------------------------------------------
# 1. the burst request text and its identity
# --------------------------------------------------------------------------


def test_a_burst_request_is_the_file_plus_the_burst_paragraph_under_its_own_key():
    """Test: a burst request is the file plus the burst paragraph under its own key."""
    text, identity = cp.composed_system_prompt(burst=True)
    file_text = cp.system_prompt_bytes().decode()
    assert text == file_text + cp.BURST_PARAGRAPH
    assert identity == {**cp.prompt_identity(),
                        "burst_prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}
    # The per-round request and identity are untouched by the burst's existence.
    assert cp.composed_system_prompt(1) == (file_text, cp.prompt_identity())


def test_the_burst_paragraph_is_general_and_asks_for_one_candidate():
    """Test: the burst paragraph is general and asks for one candidate."""
    paragraph = cp.BURST_PARAGRAPH
    assert "round 0" in paragraph and "one candidate" in paragraph and "in parallel" in paragraph
    assert "constraints.initial_family" in paragraph
    for word in ("death", "peregrine", "dugong", "michaelis", "b0", "tz", "=", "already proposed"):
        assert word not in paragraph


# --------------------------------------------------------------------------
# 2. the proposer: a burst call is a one-candidate call at round 0 with its own record
# --------------------------------------------------------------------------

CONTEXT = {"run_id": "test-only-run", "world_seed": 11, "cell": "100", "proposal_rep": 0, "round": 0}


def injected(tmp_path, replies, name="transport"):
    """Build a count proposer over a fake client that replays the given replies; return it with the recorded calls."""
    calls, queue, lock = [], list(replies), threading.Lock()

    def create(**body):
        """Fake client call: record the request body and return the next queued reply."""
        with lock:
            calls.append(body)
            text = queue.pop(0)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(text))],
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 12, "output_tokens": 3}),
            id="test-only-response", model="returned-test-model", stop_reason="end_turn",
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return cp.AnthropicCountProposer(tmp_path / name, client=client), calls


def test_a_burst_call_parses_one_candidate_and_writes_its_own_record(tmp_path):
    """Test: a burst call parses one candidate and writes its own record."""
    proposer, calls = injected(tmp_path, [reply("b0 + b1*z")])
    models, record = proposer.propose_one(
        "packet", context={**CONTEXT, "burst_call_index": 3, "burst_wave": 0},
        initial_family="binomial", burst=True)
    assert [m.expression for m in models] == ["b0 + b1*z"]
    assert record["status"] == "parsed" and record["calls"] == 1 and record["repairs"] == 0
    assert record["burst_call_index"] == 3 and record["burst_wave"] == 0 and record["round"] == 0
    assert record["burst_prompt_sha256"] == cp.composed_system_prompt(burst=True)[1]["burst_prompt_sha256"]
    assert calls[0]["system"] == cp.composed_system_prompt(burst=True)[0]
    assert (tmp_path / "transport" / "round_00__burst_w0_c03.request.json").exists()


def test_a_retried_burst_call_writes_a_record_of_its_own(tmp_path):
    """Test: a retried burst call writes a record of its own."""
    proposer, _ = injected(tmp_path, [reply("b0 + b1*z")])
    proposer.propose_one(
        "packet", context={**CONTEXT, "burst_call_index": 0, "burst_wave": 1, "burst_retry": 2},
        initial_family="binomial", burst=True)
    assert (tmp_path / "transport" / "round_00__burst_w1_c00_r2.request.json").exists()


def test_a_parallel_call_needs_its_keys_and_an_ordinary_call_refuses_them(tmp_path):
    """Test: a parallel call needs its keys and an ordinary call refuses them."""
    proposer, _ = injected(tmp_path, [reply("b0 + b1*z")] * 4)
    with pytest.raises(ValueError):
        proposer.propose_one("packet", context=CONTEXT, initial_family="binomial", burst=True)
    # A parallel call at a later round is the per-round path.
    _, record = proposer.propose_one("packet", context={**CONTEXT, "round": 1, "burst_call_index": 0,
                                                                "burst_wave": 0}, initial_family="binomial",
                                          burst=True)
    assert record["round"] == 1 and (tmp_path / "transport" / "round_01__burst_w0_c00.request.json").exists()
    with pytest.raises(ValueError):
        proposer.propose_one("packet", context={**CONTEXT, "burst_call_index": 0, "burst_wave": 2},
                             initial_family="binomial", burst=True)
    with pytest.raises(ValueError):
        proposer.propose_one("packet", context={**CONTEXT, "round": 1, "burst_call_index": 0,
                                                "burst_wave": 0}, initial_family="binomial")


def test_a_probe_refusal_inside_a_burst_call_earns_the_one_repair_turn(tmp_path):
    """Test: a probe refusal inside a burst call earns the one repair turn."""
    proposer, calls = injected(tmp_path, [reply("b0 + b1*sqrt(u)"), reply("b0 + b1*z")])
    seen = []

    def admit(hypothesis):
        """Probe stub used to admit or refuse a hypothesis."""
        seen.append(hypothesis.expression)
        if hypothesis.expression == "b0 + b1*sqrt(u)":
            raise PriorProbeRejection("synthetic probe sentence")

    models, record = proposer.propose_one(
        "packet", context={**CONTEXT, "burst_call_index": 0, "burst_wave": 0},
        initial_family="binomial", burst=True, admit=admit)
    assert [m.expression for m in models] == ["b0 + b1*z"]
    assert record["calls"] == 2 and record["repairs"] == 1
    assert record["rejected_candidates"][0]["repair_kind"] == "prior_probe"
    assert "synthetic probe sentence" in calls[1]["messages"][-1]["content"]
    assert seen == ["b0 + b1*sqrt(u)", "b0 + b1*z"]


def test_the_array_path_keeps_the_repair_turn_rejections_beside_the_per_candidate_ones(tmp_path):
    """The fixup: a k > 1 reply's per-candidate rejections are appended to the
    record, never written over the repair turn's."""
    bad_then_good = [[{"expression": "b0 + b9*z", **CANDIDATE}],
                     [reply("b0 + b1*z"), {"expression": "b0 + b8*z", **CANDIDATE}]]
    proposer, _ = injected(tmp_path, bad_then_good)
    models, record = proposer.propose_one(
        "packet", context={**CONTEXT, "round": 1}, initial_family="binomial", proposal_limit=2)
    assert [m.expression for m in models] == ["b0 + b1*z"]
    assert record["repairs"] == 1 and len(record["rejected_candidates"]) == 2
    assert record["rejected_candidates"][0]["repair_kind"] == "malformed"


# --------------------------------------------------------------------------
# 3. the packet, the archive key and the label policy
# --------------------------------------------------------------------------


def test_the_archive_carries_the_burst_source_only_when_present():
    """Test: the archive carries the burst source only when present."""
    assert "source" in ARCHIVE_OPTIONAL_KEYS
    base = {"model_id": "m", "expression": "b0", "family": "binomial", "proposed_round": 0,
            "outcome": "active", "reason": {"admitted_round": 0, "active_rounds": [0]}}
    assert "source" not in _archive_entry(base)
    assert _archive_entry({**base, "source": "initial_burst"})["source"] == "initial_burst"


def test_a_round_zero_packet_may_carry_an_empty_pool_and_no_later_round_may():
    """Test: a round zero packet may carry an empty pool and no later round may."""
    from .belief_packet import cell_constraints, core_interface

    constraints = cell_constraints("100", "binomial")
    observations = [{"obs_id": i, "round": 0, "u": 0.1 * (i + 1), "y": 3} for i in range(4)]
    packet = build_belief_packet(observations, [], [], constraints, core_interface(), 0)
    assert packet["pool"] == [] and packet["best_model"] is None
    assert packet["best_model_id"] is None and packet["best_model_residuals"] == []
    json.dumps(packet, allow_nan=False)
    with pytest.raises(ValueError, match="non-empty active pool"):
        build_belief_packet(observations, [], [], constraints, core_interface(), 1)


def settings(**extra):
    """Build the runtime settings for a BoxingGym run with a mock proposer and the given extra options."""
    args = argparse.Namespace(profile="dev", proposer="mock", cell="100", rounds=20, seed=11,
                              prompt_version=cp.PROMPT_VERSION,
                              prompt_sha256=cp.prompt_identity()["prompt_sha256"], **extra)
    return bmd_runtime_settings("boxinggym", args)


#: A burst with initial proposals also states the two refill digests
#: and the refill rule, exactly as the runner states them.
REFILL_KEYS = (*BURST_REFILL_KEYS, BURST_REFILL_POLICY_KEY)


def refill_settings():
    """Return the refill digests and refill rule exactly as the runner states them."""
    identity = cp.burst_refill_identity()
    return {**{key: identity[key] for key in BURST_REFILL_KEYS},
            BURST_REFILL_POLICY_KEY: INITIAL_BURST_REFILL_POLICY}


def test_the_receipt_binds_the_burst_the_catalogue_switch_and_the_cap_only_when_declared():
    """Test: the receipt binds the burst the catalogue switch and the cap only when declared."""
    plain = settings()
    # The refill keys are absent without a burst, too.
    for key in ("initial_proposals", "burst_prompt_sha256", "active_maximum", "initial_catalogue",
                *REFILL_KEYS):
        assert key not in plain
    digest = cp.composed_system_prompt(burst=True)[1]["burst_prompt_sha256"]
    # The burst states its refill digests and rule; each is bound as stated.
    bound = settings(initial_proposals=12, burst_prompt_sha256=digest, active_maximum=16,
                     initial_catalogue="off", **refill_settings())
    rounds = settings(parallel_proposals=3, burst_prompt_sha256=digest)
    assert rounds["parallel_proposals"] == 3 and rounds["burst_prompt_sha256"] == digest
    assert "initial_proposals" not in rounds
    assert not set(REFILL_KEYS) & set(rounds)  # a later-round wave binds none
    assert bound["initial_proposals"] == 12 and bound["burst_prompt_sha256"] == digest
    assert bound["active_maximum"] == 16 and bound["initial_catalogue"] == "off"
    assert {key: bound[key] for key in REFILL_KEYS} == refill_settings()
    # The refill keys join the burst's own as the only additions.
    extra = {"initial_proposals", "burst_prompt_sha256", "active_maximum", "initial_catalogue",
             *REFILL_KEYS}
    assert {k: v for k, v in bound.items() if k not in extra} == plain
    assert settings(active_maximum=BOXINGGYM_ACTIVE_MAXIMUM) == plain
    assert settings(initial_catalogue="on") == plain
    with pytest.raises(ValueError):
        settings(initial_proposals=0, burst_prompt_sha256=digest)
    with pytest.raises(ValueError):
        settings(active_maximum=0)
    with pytest.raises(ValueError):
        settings(initial_catalogue="off")
    with pytest.raises(ValueError):
        settings(parallel_proposals=7, burst_prompt_sha256=digest)
    with pytest.raises(ValueError):
        settings(burst_prompt_sha256=digest)


@pytest.mark.parametrize("wrong", [
    {"initial_proposals": 12}, {"burst_prompt_sha256": "0" * 64}, {"active_maximum": "sixteen"},
    {"initial_proposals": 0, "burst_prompt_sha256": "0" * 64}, {"initial_catalogue": "off"},
    {"initial_catalogue": "on", "initial_proposals": 12, "burst_prompt_sha256": "0" * 64},
    {"parallel_proposals": 3}, {"parallel_proposals": 7, "burst_prompt_sha256": "0" * 64},
])
def test_a_registered_burst_carries_both_halves_and_the_switch_needs_a_burst(tmp_path, wrong):
    """Test: a registered burst carries both halves and the switch needs a burst."""
    from .test_run_label_policy import registration, resolve_label

    ref, context, manifest = registration(tmp_path, profile="dev", proposer="mock",
                                          bundle="boxinggym", task="death")
    manifest["runtime_settings"]["boxinggym"].update(wrong)
    ref = write_manifest(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        resolve_label(None, manifest_ref=ref, **context)


# --------------------------------------------------------------------------
# 4. the runner: parallel burst, refill wave, concurrency gauge, catalogue switch
# --------------------------------------------------------------------------


def burst_records(out):
    """Load the burst record, archive entries, config and proposal accounting written by a run."""
    record = json.loads((out / "proposals" / "t00_burst.json").read_text())
    archive = json.loads((out / "archive.json").read_text())
    entries = archive if isinstance(archive, list) else list(archive.values())
    config = json.loads((out / "config.json").read_text())
    accounting = json.loads((out / "summary.json").read_text())["proposal_accounting"]
    return record, entries, config, accounting


def burst_wave(out, wave, calls, accepted, shown=(), **counts):
    """One round-0 `waves[]` entry in the refill rule's shape.

    The three keys these tests pinned before, the wave's own `duplicates`,
    `rejected_prior_probe` and `fit_failed` (zero unless `counts` states them), whether its
    packet showed the pool (a refill wave), its paragraph, the sha256 of the bytes
    packets/t00_burst_w<wave>.json holds, and the size and sorted-id digest of the model ids
    `shown` in that packet's pool - ids the caller derives from the run's events, never from
    the packet itself.
    """
    text = (out / "packets" / f"t00_burst_w{wave}.json").read_bytes()
    shown_digest = hashlib.sha256(json.dumps(sorted(shown), separators=(",", ":")).encode()).hexdigest()
    return {"wave": wave, "calls": calls, "accepted": accepted,
            "duplicates": 0, "rejected_prior_probe": 0, "fit_failed": 0, **counts,
            "pool_visible": wave > 0, "paragraph": "burst_refill" if wave else "burst",
            "packet_digest": hashlib.sha256(text).hexdigest(),
            "pool_size_shown": len(shown), "pool_digest_shown": shown_digest}


TWELVE = ["b0+b1*tz", "b0+b1*tz+b2*square(tz)", "b0+b1*tanh(3*tz)", "b0+b1*tz+b2*tanh(3*tz)",
          "b0+b1*tz+b2*t01", "b0+b1*log(1+t01)", "b0+b1*tz+b2*exp(t01)", "b0+b1*sqrt(t01)",
          "b0+b1*tz+b2*square(t01)", "b0+b1*tz+b2*power(t01,3)", "b0+b1*tanh(2*tz)",
          "b0+b1*tz+b2*tanh(2*tz)"]
REFILL = ["b0+b1*tz+b2*log(1+t01)", "b0+b1*tz+b2*sqrt(t01)", "b0+b1*t01"]


def death_reply(expression):
    """Build a binomial reply for the given expression, with roles for the coefficients it uses."""
    coefficients = sorted({token for token in ("b0", "b1", "b2") if token in expression})
    roles = {name: ("intercept" if name == "b0" else "signed") for name in coefficients}
    return {"expression": expression, "family": "binomial", "parameters": roles}


def injected_burst_proposer(monkeypatch, first, refill, *, delay=0.05):
    """An Anthropic client whose replies are handed out in call order, with a gauge
    of the calls in flight at once. The runner's factory is monkeypatched at the
    module attribute the runner imports, as the CORE transport test does."""
    lock = threading.Lock()
    state = {"queue": [*first, *refill], "in_flight": 0, "max_in_flight": 0, "calls": 0}

    def create(**body):
        """Fake client call: record the request body and return the next queued reply."""
        with lock:
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            state["calls"] += 1
            text = state["queue"].pop(0)
        time.sleep(delay)
        with lock:
            state["in_flight"] -= 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(death_reply(text)))],
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 12, "output_tokens": 3}),
            id="test-only-response", model="returned-test-model", stop_reason="end_turn",
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    real = cp.AnthropicCountProposer
    monkeypatch.setattr(cp, "AnthropicCountProposer",
                        lambda records, **kwargs: real(records, client=client))
    return state


@pytest.mark.usefixtures("require_boxinggym_source")
def test_a_death_cell_bursts_twelve_parallel_calls_refills_duplicates_and_seeds_no_catalogue(tmp_path, monkeypatch):
    """K = 12, concurrency 4, catalogue off: twelve single-candidate calls run in
    parallel (never more than four in flight), three of them duplicate earlier
    ones, one refill wave of three follows, and the round-0 pool is LLM
    candidates only, retained under the cap of 16."""
    bg = pytest.importorskip("bayes_model_discovery.experimental.boxinggym_core")
    from .test_death_kl import bg_run_args

    first = [*TWELVE[:9], TWELVE[0], TWELVE[1], TWELVE[2]]  # three duplicates of earlier calls
    # 15 burst replies, then one more for round 1's ordinary proposal call.
    gauge = injected_burst_proposer(monkeypatch, first, [*REFILL, TWELVE[9]])
    out = tmp_path / "burst"
    bg.run(bg_run_args(out, cell="100", rounds=1, proposer="anthropic", allow_paid_api=True,
                       llm_model="synthetic-model", initial_proposals=12, active_maximum=16,
                       burst_concurrency=4, initial_catalogue="off"))
    record, entries, config, accounting = burst_records(out)
    assert gauge["calls"] == 16 and gauge["max_in_flight"] <= 4  # 15 burst + 1 round-1 call
    # Each wave entry carries the refill rule's eight keys; the refill wave's packet
    # shows the nine models wave 0 admitted (the first twelve events are wave 0's).
    first_fitted = [e["model_id"] for e in record["events"][:12] if e["status"] == "fitted"]
    assert len(first_fitted) == 9
    assert record["waves"] == [burst_wave(out, 0, 12, 9, duplicates=3),
                               burst_wave(out, 1, 3, 3, first_fitted)]
    assert record["requested"] == 12 and record["max_in_flight"] <= 4 and record["concurrency"] == 4
    assert record["initial_catalogue"] == "off" and record["before_ids"] == []
    statuses = [e["status"] for e in record["events"]]
    assert statuses.count("duplicate_not_refitted") == 3 and statuses.count("fitted") == 12
    assert len(record["after_ids"]) == 12 and len(record["admitted_ids"]) == 12
    catalogue = set(bg.TASKS["death"].initial_structures())
    assert not any(e["expression"] in catalogue and e["outcome"] == "active" and e.get("source") is None
                   for e in entries)
    burst = [e for e in entries if e.get("source") == "initial_burst"]
    assert len(burst) == 15 and all(e["proposed_round"] == 0 for e in burst)
    assert sorted(e["outcome"] for e in burst).count("active") == 12
    assert config["initial_proposals"] == 12 and config["burst_concurrency"] == 4
    assert config["initial_catalogue"] == "off" and config["active_maximum"] == 16
    assert config["burst_prompt_sha256"] == cp.composed_system_prompt(burst=True)[1]["burst_prompt_sha256"]
    assert accounting["burst_calls"] == 15 and accounting["burst_refill_calls"] == 3
    assert accounting["burst_duplicates"] == 3 and accounting["burst_admitted"] == 12
    assert accounting["burst_max_in_flight"] <= 4 and accounting["burst_failed_calls"] == 0
    assert accounting["llm_calls"] == 16
    transports = record["transports"]
    assert [(t["wave"], t["index"]) for t in transports] == [(0, i) for i in range(12)] + [(1, i) for i in range(3)]
    # The refill wave sends the refill request, so its transports name the
    # refill digest config binds, and only wave 0's name the burst digest.
    assert all(t["transport"]["burst_prompt_sha256"] == config["burst_prompt_sha256"]
               for t in transports if t["wave"] == 0)
    assert all(t["transport"]["burst_refill_prompt_sha256"] == config["burst_refill_prompt_sha256"]
               and "burst_prompt_sha256" not in t["transport"] for t in transports if t["wave"])
    packet = json.loads((out / "packets" / "t00_burst.json").read_text())
    assert packet["pool"] == [] and packet["best_model"] is None and packet["round"] == 0
    assert (out / "llm_transport" / "round_00__burst_w0_c11.request.json").exists()
    assert (out / "llm_transport" / "round_00__burst_w1_c02.request.json").exists()
    assert (out / "proposals" / "t01.json").exists()


@pytest.mark.usefixtures("require_boxinggym_source")
def test_a_death_cell_runs_every_proposal_round_as_three_parallel_calls_with_one_refill(tmp_path, monkeypatch):
    """--parallel-proposals 3 without a burst: round 1 asks three parallel calls; two
    of the replies duplicate the catalogue, the refill wave asks for two more with the
    archive updated, and round 2 does the same. The --proposal-limit option is untouched."""
    bg = pytest.importorskip("bayes_model_discovery.experimental.boxinggym_core")
    from .test_death_kl import bg_run_args

    catalogue = bg.TASKS["death"].initial_structures()
    round_one = [TWELVE[2], catalogue[1], catalogue[2]]      # one fresh, two catalogue duplicates
    refill_one = [TWELVE[3], TWELVE[4]]
    # TWELVE[5] is itself a catalogue member (log(1+t01)), so round 2 uses later templates.
    round_two = [TWELVE[6], TWELVE[7], TWELVE[2]]            # the last duplicates round 1's admission
    refill_two = [TWELVE[8]]
    gauge = injected_burst_proposer(monkeypatch, [*round_one, *refill_one], [*round_two, *refill_two])
    out = tmp_path / "parallel"
    bg.run(bg_run_args(out, cell="100", rounds=2, proposer="anthropic", allow_paid_api=True,
                       llm_model="synthetic-model", parallel_proposals=3, burst_concurrency=3))
    assert gauge["calls"] == 9 and gauge["max_in_flight"] <= 3
    config = json.loads((out / "config.json").read_text())
    assert config["parallel_proposals"] == 3 and config["proposal_limit"] == 1
    assert "initial_proposals" in config and config["initial_proposals"] == 0
    assert config["burst_prompt_sha256"] == cp.composed_system_prompt(burst=True)[1]["burst_prompt_sha256"]
    assert not (out / "proposals" / "t00_burst.json").exists()
    first = json.loads((out / "proposals" / "t01.json").read_text())
    assert first["requested"] == 3 and first["waves"] == [{"wave": 0, "calls": 3, "accepted": 1},
                                                          {"wave": 1, "calls": 2, "accepted": 2}]
    statuses = [e["status"] for e in first["events"]]
    assert statuses.count("duplicate_not_refitted") == 2 and statuses.count("fitted") == 3
    assert len(first["before_ids"]) == 4 and len(first["after_ids"]) == 7 and len(first["admitted_ids"]) == 3
    assert all(e["source"] == "parallel_wave" for e in first["events"])
    assert [(x["wave"], x["index"]) for x in first["transports"]] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    assert all(x["transport"]["round"] == 1 for x in first["transports"])
    second = json.loads((out / "proposals" / "t02.json").read_text())
    assert second["waves"] == [{"wave": 0, "calls": 3, "accepted": 2}, {"wave": 1, "calls": 1, "accepted": 1}]
    assert len(second["after_ids"]) == 10
    archive = json.loads((out / "archive.json").read_text())
    entries = archive if isinstance(archive, list) else list(archive.values())
    assert sum(e.get("source") == "parallel_wave" for e in entries) == 9
    accounting = json.loads((out / "summary.json").read_text())["proposal_accounting"]
    assert accounting["burst_calls"] == 9 and accounting["burst_refill_calls"] == 3
    assert accounting["burst_duplicates"] == 3 and accounting["proposal_rounds_executed"] == 2
    assert accounting["valid_candidates"] == 9 and "burst_admitted" in accounting
    assert (out / "llm_transport" / "round_01__burst_w1_c01.request.json").exists()
    assert (out / "llm_transport" / "round_02__burst_w0_c02.request.json").exists()
    # The refill wave's packet carried the first wave's admissions in the archive.
    packet = json.loads((out / "packets" / "t01.json").read_text())
    assert packet["round"] == 1 and len(packet["pool"]) == 4


@pytest.mark.usefixtures("require_boxinggym_source")
def test_a_mock_death_cell_with_a_burst_keeps_the_catalogue_when_asked(tmp_path):
    """Mock proposer, catalogue on, K = 4 under a cap of 8: the burst walks the mock
    schedule, whose first two templates are catalogue members (duplicates), and
    the other two are fitted and admitted beside the four catalogue fits; the
    refill wave asks for the two the duplicates cost, and a second
    refill wave asks for what the first one's own duplicate cost."""
    bg = pytest.importorskip("bayes_model_discovery.experimental.boxinggym_core")
    from .test_death_kl import bg_run_args

    out = tmp_path / "burst"
    bg.run(bg_run_args(out, cell="100", rounds=1, initial_proposals=4, active_maximum=8))
    record, _, config, accounting = burst_records(out)
    statuses = [e["status"] for e in record["events"]]
    # The mock schedule's first two templates are catalogue members; the refill wave
    # walks on from the fifth template and may meet a catalogue member again, so the
    # duplicate count is at least two and the fitted count is what the seven calls left.
    assert statuses.count("duplicate_not_refitted") >= 2 and statuses.count("fitted") >= 3
    # The first refill wave meets a catalogue member again, so a third
    # wave asks for the one still missing: 4 + 2 + 1 calls, one event each.
    assert len(record["events"]) == 7
    assert len(record["before_ids"]) == 4 and len(record["after_ids"]) == 4 + statuses.count("fitted")
    assert [w["calls"] for w in record["waves"]] == [4, 2, 1]  # three waves
    assert config["initial_catalogue"] == "on" and config["burst_concurrency"] == bg.DEFAULT_BURST_CONCURRENCY
    assert accounting["burst_calls"] == 0 and accounting["burst_admitted"] == statuses.count("fitted")


@pytest.mark.usefixtures("require_boxinggym_source")
def test_a_mock_death_cell_without_the_options_writes_no_burst_key_or_file(tmp_path):
    """Test: a mock death cell without the options writes no burst key or file."""
    bg = pytest.importorskip("bayes_model_discovery.experimental.boxinggym_core")
    from .test_death_kl import bg_run_args

    out = tmp_path / "plain"
    bg.run(bg_run_args(out, cell="100", rounds=1))
    config = json.loads((out / "config.json").read_text())
    for key in ("initial_proposals", "burst_prompt_sha256", "burst_concurrency", "initial_catalogue"):
        assert key not in config
    assert config["active_maximum"] == bg.ACTIVE_MAXIMUM
    assert not list((out / "proposals").glob("t00_burst*"))
    accounting = json.loads((out / "summary.json").read_text())["proposal_accounting"]
    assert not any(key.startswith("burst_") for key in accounting)


@pytest.mark.parametrize("extra", [
    ["--initial-proposals", "12", "--active-maximum", "8"],
    ["--initial-catalogue", "off"],
    ["--burst-concurrency", "0", "--initial-proposals", "2"],
    ["--initial-proposals", "25"],
    ["--parallel-proposals", "3", "--proposal-limit", "2"],
    ["--parallel-proposals", "7"],
])
def test_the_cli_refuses_an_inconsistent_burst_configuration(monkeypatch, extra):
    """Test: the cli refuses an inconsistent burst configuration."""
    bg = pytest.importorskip("bayes_model_discovery.experimental.boxinggym_core")
    monkeypatch.setattr(sys, "argv", ["boxinggym_core", "--output", "never", "--task", "death",
                                      "--cell", "100", *extra])
    with pytest.raises(SystemExit) as refused:
        bg.main()
    assert refused.value.code == 2
