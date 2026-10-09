"""The round-0 burst refills in at most three waves, and each refill wave sees the pool.

Synthetic replies only - the runner's own mock templates through injected clients, and the
synthetic ChemBench proposer - with stubbed fits: no network, no saved run and no stored
response. The byte comparison of the later-round waves with the base commit runs only with
BMD_BASELINE_COMPARE=1.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from . import core_count as core
from . import count_proposer as cp
from .count_models import INITIAL_BURST_REFILL_POLICY, read_initial_pool_file
from .model_identity import MODEL_ID_SCHEMA
from .run_label_policy import BURST_REFILL_KEYS, bmd_runtime_settings, verify_manifest
from .test_core_d0_policy import _synthetic_real_fits
from .test_core_parallel_wave import cell_args, injected_replies
from .test_count_proposer import stub_fits
from .test_initial_burst import CONTEXT, TWELVE, injected, reply
from .test_run_label_policy import registration, runner_args, write_manifest

# The base commit whose sources the opt-in byte comparison extracts with `git archive`; it skips when this
# commit is absent from the checkout's history.
BASE_COMMIT = "b43bf8ae5245a1010594ef53cf51d6358c759ee4"
#: sha256(b"[]"): the pool digest of a packet that shows no pool (wave 0, catalogue off).
EMPTY_POOL_DIGEST = "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
WAVE_KEYS = {"wave", "calls", "accepted", "duplicates", "rejected_prior_probe", "fit_failed",
             "pool_visible", "paragraph", "packet_digest", "pool_size_shown", "pool_digest_shown"}
POOL_FILE_KEYS = ("initial_burst_refill_policy", "base_prompt_sha256", "burst_refill_paragraph_sha256")
CONFIG_KEYS = (*POOL_FILE_KEYS, "burst_refill_prompt_sha256")
FORMS = core.P1_MOCK_EXPRESSIONS


def sha(data):
    """Helper `sha` of the burst-refill tests: sha."""
    return hashlib.sha256(data).hexdigest()


def expected_identity():
    """The refill digests recomputed from the prompt file's bytes and the constant."""
    prompt, paragraph = cp.system_prompt_bytes(), cp.BURST_REFILL_PARAGRAPH.encode("utf-8")
    return {"base_prompt_sha256": sha(prompt), "burst_refill_paragraph_sha256": sha(paragraph),
            "burst_refill_prompt_sha256": sha(prompt + paragraph),
            "initial_burst_refill_policy": "pool_visible_refill_waves_max_3"}


def outputs(out):
    """Helper `outputs` of the burst-refill tests: outputs."""
    record = json.loads((out / "proposals" / "t00_burst.json").read_text())
    config = json.loads((out / "config.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    return record, config, summary


def check_burst(out, record, config, requested):
    """The refill rule and its records, read off one burst's outputs (every runner)."""
    waves = record["waves"]
    assert 1 <= len(waves) <= 3 and [entry["wave"] for entry in waves] == list(range(len(waves)))
    remaining = requested
    for index, entry in enumerate(waves):
        assert set(entry) == WAVE_KEYS
        # Every wave asks for exactly the shortfall: no extra requests, ever.
        assert entry["calls"] == remaining > 0
        assert entry["pool_visible"] is (index > 0)
        assert entry["paragraph"] == ("burst_refill" if index else "burst")
        text = (out / "packets" / f"t00_burst_w{index}.json").read_bytes()
        assert entry["packet_digest"] == sha(text)
        shown = [member["model_id"] for member in json.loads(text)["pool"]]
        assert entry["pool_size_shown"] == len(shown)
        assert entry["pool_digest_shown"] == sha(json.dumps(sorted(shown), separators=(",", ":")).encode())
        remaining -= entry["accepted"]
    # Stop as soon as K is reached, otherwise after the third wave.
    assert all(requested - sum(e["accepted"] for e in waves[:index + 1]) > 0
               for index in range(len(waves) - 1))
    assert remaining == 0 or len(waves) == 3
    assert record["shortfall_after_refill"] == remaining == requested - record["accepted_total"]
    assert record["accepted_total"] == sum(entry["accepted"] for entry in waves)
    for key in ("duplicates", "rejected_prior_probe", "fit_failed"):
        assert record[key] == sum(entry[key] for entry in waves)
    assert not (out / "packets" / f"t00_burst_w{len(waves)}.json").exists()
    assert ((out / "packets" / "t00_burst.json").read_bytes()
            == (out / "packets" / "t00_burst_w0.json").read_bytes())
    transports = record["transports"]
    assert [item["wave"] for item in transports] == [
        entry["wave"] for entry in waves for _ in range(entry["calls"])]
    identity = expected_identity()
    for item in transports:
        transport = item["transport"]
        if "packet_sha256" in transport:
            assert transport["packet_sha256"] == waves[item["wave"]]["packet_digest"]
        if "prompt_sha256" in transport:
            if item["wave"]:
                assert transport["burst_refill_prompt_sha256"] == identity["burst_refill_prompt_sha256"]
                assert "burst_prompt_sha256" not in transport
            else:
                assert transport["burst_prompt_sha256"] == config["burst_prompt_sha256"]
                assert "burst_refill_prompt_sha256" not in transport
    assert {key: record[key] for key in POOL_FILE_KEYS} == {key: identity[key] for key in POOL_FILE_KEYS}
    assert {key: config[key] for key in CONFIG_KEYS} == {key: identity[key] for key in CONFIG_KEYS}
    assert config["base_prompt_sha256"] == config["prompt_sha256"]
    return waves


def check_accounting(record, summary):
    """PWC2's wave keys, `wave_event_counts` and the burst counters add up (CORE, BoxingGym)."""
    accounting = summary["proposal_accounting"]
    transports = record["transports"]
    assert accounting["burst_wave_count"] == len(record["waves"])
    assert accounting["burst_calls"] == sum(item["transport"]["calls"] for item in transports)
    assert accounting["burst_refill_calls"] == sum(item["transport"]["calls"] for item in transports
                                                   if item["wave"])
    assert accounting["burst_failed_calls"] == sum(item["transport"]["status"] != "parsed"
                                                   for item in transports)
    counts = core.wave_event_counts(record["events"], transports, record["requested"])
    assert set(counts) == set(core.PWC2_WAVE_COUNT_KEYS)
    assert {key: record[key] for key in counts} == counts
    for key in core.PWC2_WAVE_COUNT_KEYS:
        if key != "duplicates":
            assert accounting[key] == record[key]
    assert accounting["burst_duplicates"] == record["duplicates"] == record["duplicate_count"]


def packet_of(state, wave):
    """The packets the injected client saw in one wave (all calls of a wave see one packet)."""
    seen = [packet for context, packet in state["packets"]
            if context.get("round") == 0 and context.get("burst_wave") == wave]
    assert seen and all(packet == seen[0] for packet in seen)
    return seen[0]


def request_system(out, wave, index=0):
    """Helper `request_system` of the burst-refill tests: request system."""
    request = json.loads((out / "llm_transport" / f"round_00__burst_w{wave}_c{index:02d}.request.json")
                         .read_text())
    return request["bodies"][0]["system"]


# --------------------------------------------------------------------------
# 1. the refill request: its paragraph, its identity, the proposer's contract
# --------------------------------------------------------------------------


def test_a_refill_request_is_the_file_plus_the_refill_paragraph_under_its_own_key():
    """Test: a refill request is the file plus the refill paragraph under its own key."""
    text, identity = cp.composed_system_prompt(burst="refill")
    file_text = cp.system_prompt_bytes().decode()
    assert text == file_text + cp.BURST_REFILL_PARAGRAPH
    assert identity == {**cp.prompt_identity(),
                        "burst_refill_prompt_sha256": sha(text.encode())}
    # The first wave's request and the per-round request are untouched.
    assert cp.composed_system_prompt(burst=True)[0] == file_text + cp.BURST_PARAGRAPH
    assert cp.composed_system_prompt(1) == (file_text, cp.prompt_identity())
    expected = expected_identity()
    assert cp.burst_refill_identity() == {key: expected[key] for key in (
        "base_prompt_sha256", "burst_refill_paragraph_sha256", "burst_refill_prompt_sha256")}
    with pytest.raises(ValueError):
        cp.composed_system_prompt(burst="another")


def test_the_refill_paragraph_is_general_and_names_no_form_or_benchmark():
    """Test: the refill paragraph is general and names no form or benchmark."""
    paragraph = cp.BURST_REFILL_PARAGRAPH
    assert paragraph == (
        "This request is one of several independent requests made in parallel for the same "
        "round; each answers with one candidate and none sees the others. The pool in this "
        "packet already holds the candidates admitted by earlier requests of this round. "
        "Propose one candidate whose functional form is not already in the pool; a candidate "
        "with the same canonical identity as a pool member is discarded as a duplicate. Keep "
        "the family constraints.initial_family; every other rule applies unchanged.\n")
    for word in ("one candidate", "in parallel", "pool", "canonical identity", "constraints.initial_family"):
        assert word in paragraph
    lowered = paragraph.lower()
    for word in ("death", "peregrine", "dugong", "michaelis", "chembench", "boxinggym", "core",
                 "gauss", "b0", "b1", "tz", "t01", "tanh", "exp(", "log(", "sqrt", "square",
                 "=", "e.g.", "for example", "such as", "round 0"):
        assert word not in lowered


def test_a_refill_call_is_a_round_zero_wave_one_or_two_call_with_its_own_record(tmp_path):
    """Test: a refill call is a round zero wave one or two call with its own record."""
    proposer, calls = injected(tmp_path, [reply("b0 + b1*z")])
    models, record = proposer.propose_one(
        "packet", context={**CONTEXT, "burst_call_index": 1, "burst_wave": 2},
        initial_family="binomial", burst="refill")
    assert [m.expression for m in models] == ["b0 + b1*z"]
    assert record["burst_wave"] == 2 and record["round"] == 0 and record["status"] == "parsed"
    assert record["burst_refill_prompt_sha256"] == expected_identity()["burst_refill_prompt_sha256"]
    assert "burst_prompt_sha256" not in record
    assert calls[0]["system"] == cp.composed_system_prompt(burst="refill")[0]
    assert (tmp_path / "transport" / "round_00__burst_w2_c01.request.json").exists()


@pytest.mark.parametrize("burst,context", [
    ("refill", {"burst_call_index": 0, "burst_wave": 0}),
    ("refill", {"burst_call_index": 0, "burst_wave": 1, "round": 1}),
    ("refill", {"burst_call_index": 0, "burst_wave": 3}),
    (True, {"burst_call_index": 0, "burst_wave": 2}),
    ("another", {"burst_call_index": 0, "burst_wave": 1}),
])
def test_the_proposer_refuses_a_refill_call_outside_round_zero_waves_one_and_two(tmp_path, burst, context):
    """Test: the proposer refuses a refill call outside round zero waves one and two."""
    proposer, calls = injected(tmp_path, [reply("b0 + b1*z")])
    with pytest.raises(ValueError):
        proposer.propose_one("packet", context={**CONTEXT, **context},
                             initial_family="binomial", burst=burst)
    assert calls == []


# --------------------------------------------------------------------------
# 2. CORE
# --------------------------------------------------------------------------


def core_burst(tmp_path, monkeypatch, forms, *, requested, family="binomial", **options):
    """Helper `core_burst` of the burst-refill tests: core burst."""
    if family == "normal":
        _synthetic_real_fits(monkeypatch)
        options = {"structure": "gauss_den", "truth_family": "normal", "initial_family": "normal",
                   **options}
    else:
        stub_fits(monkeypatch)
    state = injected_replies(monkeypatch, forms, family=family)
    out = tmp_path / "core"
    core.run(cell_args(out, initial_proposals=requested, initial_catalogue="off",
                       burst_concurrency=1, run_label="development", **options))
    return out, state


def test_core_a_first_wave_that_fills_k_runs_no_refill(tmp_path, monkeypatch):
    """Test: core a first wave that fills k runs no refill."""
    out, state = core_burst(tmp_path, monkeypatch, FORMS[:3], requested=3, proposal_rounds="2")
    record, config, summary = outputs(out)
    waves = check_burst(out, record, config, 3)
    check_accounting(record, summary)
    assert state["queue"] == [] and state["calls"] == 3
    assert [(e["calls"], e["accepted"]) for e in waves] == [(3, 3)]
    assert waves[0]["pool_size_shown"] == 0 and waves[0]["pool_digest_shown"] == EMPTY_POOL_DIGEST
    assert summary["proposal_accounting"]["burst_wave_count"] == 1
    assert summary["proposal_accounting"]["burst_refill_calls"] == 0
    # B7: the reader accepts the pool file the writer produced, key included.
    pool = read_initial_pool_file(out / "proposals" / "t00_burst.json", cell="100",
                                  initial_family="binomial")
    assert [h.model_id for h in pool.hypotheses] == record["after_ids"]


@pytest.mark.parametrize("family", ["binomial", "normal"])
def test_core_a_refill_wave_sees_the_first_waves_admissions_under_the_refill_paragraph(
        tmp_path, monkeypatch, family):
    # Wave 0: three calls, one form -> one admission, two duplicates; wave 1 asks for the two.
    """Test: core a refill wave sees the first waves admissions under the refill paragraph."""
    forms = [FORMS[0], FORMS[0], FORMS[0], FORMS[1], FORMS[2], FORMS[3]]  # the last: round 1
    out, state = core_burst(tmp_path, monkeypatch, forms, requested=3, family=family)
    record, config, summary = outputs(out)
    waves = check_burst(out, record, config, 3)
    check_accounting(record, summary)
    assert state["queue"] == [] and state["calls"] == 6
    assert [(e["calls"], e["accepted"], e["duplicates"]) for e in waves] == [(3, 1, 2), (2, 2, 0)]
    first_fitted = [e["model_id"] for e in record["events"][:3] if e["status"] == "fitted"]
    refill = packet_of(state, 1)
    assert [member["model_id"] for member in refill["pool"]] == first_fitted
    assert refill == json.loads((out / "packets" / "t00_burst_w1.json").read_text())
    assert packet_of(state, 0)["pool"] == []
    assert waves[1]["pool_size_shown"] == 1 and waves[1]["paragraph"] == "burst_refill"
    # The refill packet's pool is in the form a round >= 1 packet carries.
    later = json.loads((out / "packets" / "t01.json").read_text())
    assert set(refill["pool"][0]) == set(later["pool"][0])
    assert refill["best_model_id"] == first_fitted[0] and refill["best_model_residuals"]
    assert refill["constraints"]["initial_structures"] == []
    assert request_system(out, 0) == cp.composed_system_prompt(burst=True)[0]
    assert request_system(out, 1) == request_system(out, 1, 1) == cp.composed_system_prompt(burst="refill")[0]
    assert summary["proposal_accounting"]["burst_refill_calls"] == 2


def test_core_three_waves_still_short_stop_and_record_the_shortfall(tmp_path, monkeypatch):
    # s1 = 3 after wave 0, s2 = 2 after wave 1, and wave 2 admits nothing: calls = K + s1 + s2.
    """Test: core three waves still short stop and record the shortfall."""
    forms = [FORMS[0]] * 4 + [FORMS[0], FORMS[0], FORMS[1]] + [FORMS[0], FORMS[1]]
    out, state = core_burst(tmp_path, monkeypatch, forms, requested=4, proposal_rounds="2")
    record, config, summary = outputs(out)
    waves = check_burst(out, record, config, 4)
    check_accounting(record, summary)
    assert state["queue"] == [] and state["calls"] == 4 + 3 + 2
    assert [(e["calls"], e["accepted"]) for e in waves] == [(4, 1), (3, 1), (2, 0)]
    assert [e["pool_size_shown"] for e in waves] == [0, 1, 2]
    assert waves[0]["pool_digest_shown"] == EMPTY_POOL_DIGEST
    assert record["shortfall_after_refill"] == 2 == summary["proposal_accounting"]["shortfall_after_refill"]
    assert summary["proposal_accounting"]["burst_wave_count"] == 3
    assert summary["proposal_accounting"]["burst_refill_calls"] == 5
    assert len(record["after_ids"]) == 2
    wave_two = packet_of(state, 2)
    assert sorted(m["model_id"] for m in wave_two["pool"]) == sorted(record["after_ids"])
    assert sorted(p.name for p in (out / "packets").glob("t00_burst*.json")) == [
        "t00_burst.json", "t00_burst_w0.json", "t00_burst_w1.json", "t00_burst_w2.json"]


def test_core_a_catalogue_pool_is_shown_to_wave_zero_and_kept_in_the_refill_pool(tmp_path, monkeypatch):
    """Mock proposer, catalogue on: wave 0 shows the catalogue fits, a refill wave shows
    `retain(catalogue + fitted, active_maximum)`, and every mock transport names its text."""
    stub_fits(monkeypatch)
    out = tmp_path / "core"
    core.run(cell_args(out, proposer="mock", allow_paid_api=False, initial_proposals=4,
                       active_maximum=5, proposal_rounds="2"))
    record, config, summary = outputs(out)
    waves = check_burst(out, record, config, 4)
    check_accounting(record, summary)
    assert waves[0]["pool_size_shown"] == 4 and len(waves) >= 2
    catalogue = set(record["before_ids"])
    refill = json.loads((out / "packets" / "t00_burst_w1.json").read_text())
    shown = {member["model_id"] for member in refill["pool"]}
    assert len(shown) == min(5, 4 + waves[0]["accepted"]) and shown & catalogue


def test_core_without_a_burst_writes_no_refill_key_file_or_counter(tmp_path, monkeypatch):
    """Test: core without a burst writes no refill key file or counter."""
    stub_fits(monkeypatch)
    for name, options in (("plain", {}), ("parallel", {"parallel_proposals": 2})):
        out = tmp_path / name
        core.run(cell_args(out, proposer="mock", allow_paid_api=False, **options))
        config = json.loads((out / "config.json").read_text())
        assert not set(CONFIG_KEYS) & set(config)
        assert ("burst_prompt_sha256" in config) == bool(options)
        assert not list(out.rglob("t00_burst*"))
        accounting = json.loads((out / "summary.json").read_text())["proposal_accounting"]
        assert "burst_wave_count" not in accounting
        for path in (out / "proposals").glob("t*.json"):
            record = json.loads(path.read_text())
            assert not set(POOL_FILE_KEYS) & set(record)
            assert all(set(entry) == {"wave", "calls", "accepted"} for entry in record.get("waves", []))


# --------------------------------------------------------------------------
# 3. BoxingGym
# --------------------------------------------------------------------------


def boxinggym_burst(tmp_path, monkeypatch, forms, *, requested):
    """Helper `boxinggym_burst` of the burst-refill tests: boxinggym burst."""
    from . import boxinggym_core as bg
    from .test_death_kl import bg_run_args

    stub_fits(monkeypatch)
    monkeypatch.setattr(bg, "count_fit", core.fit)
    state = injected_replies(monkeypatch, forms)
    out = tmp_path / "boxinggym"
    bg.run(bg_run_args(out, cell="100", rounds=1, proposer="anthropic", allow_paid_api=True,
                       llm_model="synthetic-model", initial_proposals=requested,
                       initial_catalogue="off", burst_concurrency=1))
    return out, state


@pytest.mark.usefixtures("require_boxinggym_source")
def test_boxinggym_a_first_wave_that_fills_k_runs_no_refill(tmp_path, monkeypatch):
    """Test: boxinggym a first wave that fills k runs no refill."""
    out, state = boxinggym_burst(tmp_path, monkeypatch, [*TWELVE[:3], TWELVE[3]], requested=3)
    record, config, summary = outputs(out)
    waves = check_burst(out, record, config, 3)
    check_accounting(record, summary)
    assert state["queue"] == [] and state["calls"] == 4  # three burst calls and round 1's
    assert [(e["calls"], e["accepted"]) for e in waves] == [(3, 3)]
    assert waves[0]["pool_digest_shown"] == EMPTY_POOL_DIGEST


@pytest.mark.usefixtures("require_boxinggym_source")
def test_boxinggym_a_refill_wave_sees_the_first_waves_admissions_under_the_refill_paragraph(
        tmp_path, monkeypatch):
    """Test: boxinggym a refill wave sees the first waves admissions under the refill paragraph."""
    forms = [TWELVE[0], TWELVE[0], TWELVE[0], TWELVE[1], TWELVE[2], TWELVE[3]]
    out, state = boxinggym_burst(tmp_path, monkeypatch, forms, requested=3)
    record, config, summary = outputs(out)
    waves = check_burst(out, record, config, 3)
    check_accounting(record, summary)
    assert state["queue"] == [] and state["calls"] == 6
    assert [(e["calls"], e["accepted"], e["duplicates"]) for e in waves] == [(3, 1, 2), (2, 2, 0)]
    first_fitted = [e["model_id"] for e in record["events"][:3] if e["status"] == "fitted"]
    refill = packet_of(state, 1)
    assert [member["model_id"] for member in refill["pool"]] == first_fitted
    assert refill == json.loads((out / "packets" / "t00_burst_w1.json").read_text())
    later = json.loads((out / "packets" / "t01.json").read_text())
    assert set(refill["pool"][0]) == set(later["pool"][0])
    assert request_system(out, 1) == cp.composed_system_prompt(burst="refill")[0]
    assert request_system(out, 0) == cp.composed_system_prompt(burst=True)[0]


@pytest.mark.usefixtures("require_boxinggym_source")
def test_boxinggym_three_waves_still_short_stop_and_record_the_shortfall(tmp_path, monkeypatch):
    """Test: boxinggym three waves still short stop and record the shortfall."""
    forms = [TWELVE[0]] * 4 + [TWELVE[0], TWELVE[0], TWELVE[1]] + [TWELVE[0], TWELVE[1]] + [TWELVE[2]]
    out, state = boxinggym_burst(tmp_path, monkeypatch, forms, requested=4)
    record, config, summary = outputs(out)
    waves = check_burst(out, record, config, 4)
    check_accounting(record, summary)
    assert state["queue"] == [] and state["calls"] == 4 + 3 + 2 + 1  # plus round 1's call
    assert [(e["calls"], e["accepted"]) for e in waves] == [(4, 1), (3, 1), (2, 0)]
    assert [e["pool_size_shown"] for e in waves] == [0, 1, 2]
    assert record["shortfall_after_refill"] == 2 == summary["proposal_accounting"]["shortfall_after_refill"]
    assert summary["proposal_accounting"]["burst_wave_count"] == 3


@pytest.mark.usefixtures("require_boxinggym_source")
def test_boxinggym_without_a_burst_writes_no_refill_key_or_file(tmp_path, monkeypatch):
    """Test: boxinggym without a burst writes no refill key or file."""
    from . import boxinggym_core as bg
    from .test_death_kl import bg_run_args

    stub_fits(monkeypatch)
    monkeypatch.setattr(bg, "count_fit", core.fit)
    out = tmp_path / "plain"
    bg.run(bg_run_args(out, cell="100", rounds=1))
    config = json.loads((out / "config.json").read_text())
    assert not set(CONFIG_KEYS) & set(config)
    assert not list(out.rglob("t00_burst*"))
    assert "burst_wave_count" not in json.loads((out / "summary.json").read_text())["proposal_accounting"]


# --------------------------------------------------------------------------
# 4. ChemBench (CBIP)
# --------------------------------------------------------------------------


def chembench_burst(tmp_path, monkeypatch, choose, *, requested):
    """Helper `chembench_burst` of the burst-refill tests: chembench burst."""
    from .test_chembench_initial_pool import _wave

    return _wave(tmp_path, monkeypatch, choose, initial_proposals=requested)


def chembench_candidate(index, names):
    """Helper `chembench_candidate` of the burst-refill tests: chembench candidate."""
    from .test_chembench_initial_pool import _candidate

    return _candidate(index, names)


def test_chembench_a_first_wave_that_fills_k_runs_no_refill(tmp_path, monkeypatch):
    """Test: chembench a first wave that fills k runs no refill."""
    out, proposer = chembench_burst(tmp_path, monkeypatch, lambda context, names:
                                    chembench_candidate(context["burst_call_index"], names), requested=3)
    record, config, summary = outputs(out)
    waves = check_burst(out, record, config, 3)
    assert [(e["calls"], e["accepted"]) for e in waves] == [(3, 3)]
    assert waves[0]["pool_digest_shown"] == EMPTY_POOL_DIGEST
    assert len(proposer.calls) == 3 and all(call["burst"] is True for _, call in proposer.calls)
    assert summary["proposal_accounting"]["burst_wave_count"] == 1


def test_chembench_a_refill_wave_sees_the_first_waves_admissions_under_the_refill_paragraph(
        tmp_path, monkeypatch):
    """Test: chembench a refill wave sees the first waves admissions under the refill paragraph."""
    def choose(context, names):
        wave = context["burst_wave"]
        return chembench_candidate(0 if wave == 0 else 10 + context["burst_call_index"], names)

    out, proposer = chembench_burst(tmp_path, monkeypatch, choose, requested=3)
    record, config, summary = outputs(out)
    waves = check_burst(out, record, config, 3)
    assert [(e["calls"], e["accepted"], e["duplicates"]) for e in waves] == [(3, 1, 2), (2, 2, 0)]
    first_fitted = [e["model_id"] for e in record["events"][:3] if e["status"] == "fitted"]
    refill = [(packet, call) for packet, call in proposer.calls if call["context"]["burst_wave"] == 1]
    assert len(refill) == 2
    for packet, call in refill:
        assert call["burst"] == "refill"
        assert [member["model_id"] for member in packet["pool"]] == first_fitted
        assert packet == json.loads((out / "packets" / "t00_burst_w1.json").read_text())
    assert all(call["burst"] is True for _, call in proposer.calls if call["context"]["burst_wave"] == 0)
    assert summary["proposal_accounting"]["burst_wave_count"] == 2


def test_chembench_three_waves_still_short_stop_and_record_the_shortfall(tmp_path, monkeypatch):
    """Test: chembench three waves still short stop and record the shortfall."""
    def choose(context, names):
        wave, index = context["burst_wave"], context["burst_call_index"]
        return chembench_candidate(index if wave == 1 else 0, names)

    out, proposer = chembench_burst(tmp_path, monkeypatch, choose, requested=3)
    record, config, summary = outputs(out)
    waves = check_burst(out, record, config, 3)
    assert len(proposer.calls) == 3 + 2 + 1
    assert [(e["calls"], e["accepted"]) for e in waves] == [(3, 1), (2, 1), (1, 0)]
    assert [e["pool_size_shown"] for e in waves] == [0, 1, 2]
    assert record["shortfall_after_refill"] == 1
    assert summary["proposal_accounting"]["burst_wave_count"] == 3
    receipt = json.loads((out / "initial_pool_receipt.json").read_text())
    assert receipt["waves"] == waves


def test_chembench_mock_burst_records_both_requests_per_wave(tmp_path, monkeypatch):
    """Test: chembench mock burst records both requests per wave."""
    from . import chembench_core as cc
    from .test_chembench_core import _args, _manifest
    from .test_chembench_mda_options import stub_fits as stub_chembench_fits

    stub_chembench_fits(monkeypatch)
    out = tmp_path / "mock"
    cc.run(_args(out, _manifest(tmp_path), cell="100", structure_proposals="free", rounds=1,
                 initial_proposals=10, initial_catalogue="off"))
    record, config, summary = outputs(out)
    waves = check_burst(out, record, config, 10)
    assert len(waves) >= 2 and all("prompt_sha256" in item["transport"] for item in record["transports"])
    assert summary["proposal_accounting"]["burst_wave_count"] == len(waves)


def test_chembench_refill_calls_send_and_record_the_refill_request(tmp_path, monkeypatch):
    """Test: chembench refill calls send and record the refill request."""
    from . import chembench_core as cc
    from .test_chembench_core import _args, _manifest
    from .test_chembench_mda_options import stub_fits as stub_chembench_fits
    from .test_neuron_proposer import Client, response

    stub_chembench_fits(monkeypatch)
    replies = [{"expression": f"b0+b1*tanh(f0z+{index}/20)", "family": "lognormal",
                "parameters": {"b0": "intercept", "b1": "signed"}} for index in (0, 0, 1, 2)]
    client = Client("anthropic", [response("anthropic", body) for body in replies])
    original = cp.AnthropicCountProposer
    monkeypatch.setattr(cp, "AnthropicCountProposer",
                        lambda records, **kwargs: original(records, client=client))
    out = tmp_path / "parsed"
    cc.run(_args(out, _manifest(tmp_path), cell="100", structure_proposals="free", rounds=1,
                 initial_proposals=2, initial_catalogue="off", proposer="anthropic", allow_paid_api=True))
    record, config, _ = outputs(out)
    waves = check_burst(out, record, config, 2)
    assert [(e["calls"], e["accepted"]) for e in waves] == [(2, 1), (1, 1)]
    assert len(client.calls) == 4  # two first-wave calls, one refill call, round 1's call
    assert request_system(out, 0) == request_system(out, 0, 1) == cp.composed_system_prompt(burst=True)[0]
    assert request_system(out, 1) == cp.composed_system_prompt(burst="refill")[0]


def test_chembench_without_a_burst_writes_no_refill_key_or_file(tmp_path, monkeypatch):
    """Test: chembench without a burst writes no refill key or file."""
    from . import chembench_core as cc
    from .test_chembench_core import _args, _manifest
    from .test_chembench_mda_options import stub_fits as stub_chembench_fits

    stub_chembench_fits(monkeypatch)
    out = tmp_path / "plain"
    cc.run(_args(out, _manifest(tmp_path), cell="100", structure_proposals="free", rounds=1))
    config = json.loads((out / "config.json").read_text())
    assert not set(CONFIG_KEYS) & set(config)
    assert not list(out.rglob("t00_burst*"))
    assert "burst_wave_count" not in json.loads((out / "summary.json").read_text())["proposal_accounting"]


# --------------------------------------------------------------------------
# 5. the pool-file reader (IPR): the refill rule is part of what a frozen burst binds
# --------------------------------------------------------------------------


def synthetic_burst(**changes):
    """Helper `synthetic_burst` of the burst-refill tests: synthetic burst."""
    from .test_initial_pool_file import burst

    payload = {**burst(), "initial_burst_refill_policy": INITIAL_BURST_REFILL_POLICY}
    payload.update(changes)
    return {key: value for key, value in payload.items() if value is not None}


def read_synthetic(tmp_path, payload):
    """Helper `read_synthetic` of the burst-refill tests: read synthetic."""
    path = tmp_path / "synthetic_pool.json"
    path.write_text(json.dumps(payload) + "\n")
    return read_initial_pool_file(path, cell="100", initial_family="binomial")


def test_the_reader_accepts_a_burst_under_the_refill_rule(tmp_path):
    """Test: the reader accepts a burst under the refill rule."""
    payload = synthetic_burst()
    pool = read_synthetic(tmp_path, payload)
    assert pool.source == "file_burst" and [h.model_id for h in pool.hypotheses] == payload["after_ids"]


@pytest.mark.parametrize("value", [None, "two_waves_empty_pool", ""])
def test_the_reader_refuses_a_burst_without_the_refill_rule(tmp_path, value):
    """Test: the reader refuses a burst without the refill rule."""
    payload = synthetic_burst(initial_burst_refill_policy=value)
    if value is None:
        assert "initial_burst_refill_policy" not in payload
    expected = ("initial pool burst requires initial_burst_refill_policy "
                f"'pool_visible_refill_waves_max_3'; got {value!r}")
    with pytest.raises(ValueError) as refused:
        read_synthetic(tmp_path, payload)
    assert str(refused.value) == expected


def test_the_reader_checks_the_schema_before_the_refill_rule_and_leaves_arrays_alone(tmp_path):
    """Test: the reader checks the schema before the refill rule and leaves arrays alone."""
    with pytest.raises(ValueError, match="model_id_schema mismatch"):
        read_synthetic(tmp_path, synthetic_burst(model_id_schema="unknown", initial_burst_refill_policy=None))
    from .test_initial_pool_file import candidates

    pool = read_synthetic(tmp_path, candidates())
    assert pool.source == "file_candidates" and len(pool.hypotheses) == 2
    assert synthetic_burst()["model_id_schema"] == MODEL_ID_SCHEMA


# --------------------------------------------------------------------------
# 6. the registered settings: both refill digests, only with initial proposals
# --------------------------------------------------------------------------


def burst_args(bundle="core", **changes):
    """Helper `burst_args` of the burst-refill tests: burst args."""
    identity = cp.burst_refill_identity()
    # A burst with initial proposals states its refill rule, which the
    # settings now bind beside the two digests (the three runners state it the same way).
    values = {"initial_proposals": 4, "initial_catalogue": "off",
              "burst_prompt_sha256": cp.composed_system_prompt(burst=True)[1]["burst_prompt_sha256"],
              **{key: identity[key] for key in BURST_REFILL_KEYS},
              "initial_burst_refill_policy": INITIAL_BURST_REFILL_POLICY,
              **({"initial_pool_source": "generated"} if bundle == "chembench" else {})}
    values.update(changes)
    return runner_args(**{key: value for key, value in values.items() if value is not None})


@pytest.mark.parametrize("bundle", ["core", "boxinggym", "chembench"])
def test_a_burst_binds_both_refill_digests_and_nothing_else_does(bundle):
    """Test: a burst binds both refill digests and nothing else does."""
    identity = cp.burst_refill_identity()
    bound = bmd_runtime_settings(bundle, burst_args(bundle))
    assert {key: bound[key] for key in BURST_REFILL_KEYS} == {key: identity[key] for key in BURST_REFILL_KEYS}
    assert not set(BURST_REFILL_KEYS) & set(bmd_runtime_settings(bundle, runner_args()))
    if bundle != "chembench":
        parallel = bmd_runtime_settings(bundle, runner_args(
            parallel_proposals=3, burst_prompt_sha256=bound["burst_prompt_sha256"]))
        assert not set(BURST_REFILL_KEYS) & set(parallel)
    for key in BURST_REFILL_KEYS:
        with pytest.raises(ValueError, match="lowercase SHA256"):
            bmd_runtime_settings(bundle, burst_args(bundle, **{key: None}))
        with pytest.raises(ValueError, match="needs a burst with initial proposals"):
            bmd_runtime_settings(bundle, runner_args(**{key: identity[key]}))


def test_the_core_registration_refuses_a_missing_altered_or_unexpected_refill_digest(tmp_path):
    """Test: the core registration refuses a missing altered or unexpected refill digest."""
    settings = bmd_runtime_settings("core", burst_args())
    (tmp_path / "bound").mkdir()
    ref, context, manifest = registration(tmp_path / "bound", settings=settings)
    verify_manifest(ref, **context)
    for key in BURST_REFILL_KEYS:
        for change in ("removed", "altered"):
            tampered = json.loads(json.dumps(manifest))
            if change == "removed":
                tampered["runtime_settings"]["core"].pop(key)
            else:
                tampered["runtime_settings"]["core"][key] = "0" * 64
            other = write_manifest(tmp_path / "bound" / f"{key}_{change}.json", tampered)
            with pytest.raises(ValueError, match="Manifest runtime settings mismatch"):
                verify_manifest(other, **context)
    plain = bmd_runtime_settings("core", runner_args())
    (tmp_path / "plain").mkdir()
    ref, context, manifest = registration(tmp_path / "plain", settings=plain)
    verify_manifest(ref, **context)
    for key in BURST_REFILL_KEYS:
        stray = json.loads(json.dumps(manifest))
        stray["runtime_settings"]["core"][key] = settings[key]
        other = write_manifest(tmp_path / "plain" / f"{key}_stray.json", stray)
        with pytest.raises(ValueError, match="Manifest runtime settings mismatch"):
            verify_manifest(other, **context)


def test_a_core_burst_receipt_binds_the_digests_its_refill_calls_send(tmp_path, monkeypatch):
    """Test: a core burst receipt binds the digests its refill calls send."""
    out, _ = core_burst(tmp_path, monkeypatch, [FORMS[0]] * 3 + [FORMS[1], FORMS[2]], requested=3,
                        proposal_rounds="2")
    record, config, _ = outputs(out)
    settings = config["label_receipt"]["binding"]["settings"]
    assert {key: settings[key] for key in BURST_REFILL_KEYS} == {key: config[key] for key in BURST_REFILL_KEYS}
    refill = [item["transport"] for item in record["transports"] if item["wave"]]
    assert refill and all(t["burst_refill_prompt_sha256"] == settings["burst_refill_prompt_sha256"]
                          for t in refill)


# --------------------------------------------------------------------------
# 7. the later-round parallel waves are the base commit's, byte for byte
# --------------------------------------------------------------------------

# Conditions 2 and 3 of the re-pinning rule: the changes stacked after the refill rule add
# these record keys on this path; they are removed by name from both runs, any other difference fails, and the
# science is compared as raw bytes on its own.
#: The three engine limits every packet's `constraints` now states.
PCX1_PACKET_CONSTRAINT_KEYS = ("max_distinct_mean_coefficients", "reserved_names",
                               "positive_mean_required_families")
#: The digest pair a transport record keeps of the packet text it sent, moved by the
#: keys above; each run's pair is checked against that run's own request text before it is removed.
PCX1_PACKET_DIGEST_KEYS = ("packet_sha256", "packet_bytes")
#: `failure_class` on each candidate event, `parameters` on each archive.json
#: entry, and the run's `failure_class_counts` in summary `proposal_accounting`.
PCX2_EVENT_KEYS = ("failure_class",)
PCX2_ARCHIVE_KEYS = ("parameters",)
PCX2_ACCOUNTING_KEYS = ("failure_class_counts",)


def request_text(run, name):
    """The one packet text an llm_transport request sent (one body, one message)."""
    (body,) = json.loads((run / "llm_transport" / name).read_text())["bodies"]
    (message,) = body["messages"]
    return message["content"].encode()


def pop_packet_pair(record, text, name):
    """Remove a record's packet digest pair once it binds `text`."""
    assert tuple(record.pop(key) for key in PCX1_PACKET_DIGEST_KEYS) == (sha(text), len(text)), name


def without_unit_keys(run, folder, name):
    """One B8 record, parsed, with the `PCX1_*` and `PCX2_*` named keys removed."""
    record = json.loads((run / folder / name).read_text())
    if folder == "packets":
        for key in PCX1_PACKET_CONSTRAINT_KEYS:
            record["constraints"].pop(key, None)
    elif name.endswith(".request.json"):
        text = request_text(run, name)
        pop_packet_pair(record["identity"], text, name)
        packet = json.loads(text)
        for key in PCX1_PACKET_CONSTRAINT_KEYS:
            packet["constraints"].pop(key, None)
        record["bodies"][0]["messages"][0]["content"] = packet
    elif name.endswith(".response.json"):
        pop_packet_pair(record, request_text(run, name.replace(".response.", ".request.")), name)
    else:
        for event in record["events"]:
            for key in PCX2_EVENT_KEYS:
                event.pop(key, None)
        for item in record["transports"]:
            transport = item["transport"]
            request = (f"round_{transport['round']:02d}__burst_w{transport['burst_wave']}"
                       f"_c{transport['burst_call_index']:02d}.request.json")
            pop_packet_pair(transport, request_text(run, request), name)
    return record


PARITY_HARNESS = r'''
import sys
import time
from pathlib import Path
from pytest import MonkeyPatch
from bayes_model_discovery.experimental.test_count_proposer import stub_fits
from bayes_model_discovery.experimental.test_core_parallel_wave import cell_args, injected_replies
from bayes_model_discovery.experimental.test_initial_burst import TWELVE
with MonkeyPatch.context() as patch:
    core = stub_fits(patch)
    patch.setattr(time, "perf_counter", lambda: 0.0)
    out, suite = Path(sys.argv[1]), sys.argv[2]
    if suite == "core":
        catalogue = [h.expression for h in core.initial_hypotheses("binomial")]
        forms = core.P1_MOCK_EXPRESSIONS
        injected_replies(patch, [forms[2], catalogue[1], catalogue[2], forms[3], forms[4],
                                 forms[6], forms[7], forms[2], forms[8]])
        core.run(cell_args(out, rounds=2, parallel_proposals=3, burst_concurrency=1))
    else:
        from bayes_model_discovery.experimental import boxinggym_core as bg
        from bayes_model_discovery.experimental.test_death_kl import bg_run_args
        patch.setattr(bg, "count_fit", core.fit)
        catalogue = bg.TASKS["death"].initial_structures()
        injected_replies(patch, [TWELVE[2], catalogue[1], catalogue[2], TWELVE[3], TWELVE[4],
                                 TWELVE[6], TWELVE[7], TWELVE[2], TWELVE[8]])
        bg.run(bg_run_args(out, cell="100", rounds=2, proposer="anthropic", allow_paid_api=True,
                           llm_model="synthetic-model", parallel_proposals=3, burst_concurrency=1))
'''


@pytest.mark.parametrize("suite", ["core", "boxinggym"])
def test_later_round_parallel_waves_match_the_base_commit_byte_for_byte(tmp_path, request, suite):
    """CORE's parallel_proposals and BoxingGym's parallel rounds under proposal policy P1 (t >= 1) keep their two
    waves, packets, records and transports exactly as BASE_COMMIT (the base commit the refill rule is stacked on) wrote
    them, with injected replies, stubbed fits and a zero clock in both trees.

    The science is compared byte for byte; the records are compared with the keys the
    `PCX1_*` and `PCX2_*` constants name, removed by name, and nothing else may differ."""
    if os.environ.get("BMD_BASELINE_COMPARE") != "1":
        pytest.skip("set BMD_BASELINE_COMPARE=1 for the byte comparison with the base commit")
    if suite == "boxinggym":
        request.getfixturevalue("require_boxinggym_source")
    current = Path(__file__).resolve().parents[3]
    archive = subprocess.run(["git", "-C", str(current), "archive", BASE_COMMIT, "src"],
                             capture_output=True, check=False)
    if archive.returncode:
        pytest.skip(f"base commit {BASE_COMMIT} is not in this checkout's history")
    base = tmp_path / "base_tree"
    base.mkdir()
    subprocess.run(["tar", "-x", "-C", str(base)], input=archive.stdout, check=True)
    external = os.environ.get("BMD_BOXINGGYM_SRC", "")
    same, runs = tmp_path / "same_output", []
    for label, source in (("base", base), ("current", current)):
        env = dict(os.environ, PYTHONPATH=f"{source}/src:{external}", PYTHONDONTWRITEBYTECODE="1",
                   OPENAI_API_KEY="", ANTHROPIC_API_KEY="", OPENAI_COMPAT_API_KEY="", OPENROUTER_API_KEY="")
        result = subprocess.run([sys.executable, "-B", "-c", PARITY_HARNESS, str(same), suite],
                                cwd=source, env=env, capture_output=True, text=True, check=False,
                                timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr
        target = tmp_path / label
        same.rename(target)
        runs.append(target)
    left, right = runs
    # The science as raw bytes (observations, scores, chosen
    # designs, fitted and predicted arrays) and each round's admitted ids, strictly.
    for name in ("observations.jsonl", "scores.jsonl"):
        assert (left / name).read_bytes() == (right / name).read_bytes(), name
    for folder, pattern in (("acquisition", "*.json"), ("fits", "*.npz"), ("predictions", "*.npz")):
        names = [sorted(path.name for path in (run / folder).glob(pattern)) for run in runs]
        assert names[0] == names[1] and names[0]
        for name in names[0]:
            assert (left / folder / name).read_bytes() == (right / folder / name).read_bytes(), name
    for round_number in (1, 2):
        admitted = [json.loads((run / "proposals" / f"t{round_number:02d}.json").read_text())["admitted_ids"]
                    for run in runs]
        assert admitted[0] == admitted[1]
    # The records, with the `PCX1_*` and `PCX2_*` keys removed by name;
    # weakened from raw bytes to parsed JSON (packets, proposals, llm_transport, archive) for that.
    for folder in ("proposals", "packets", "llm_transport"):
        names = [sorted(path.name for path in (run / folder).iterdir()) for run in runs]
        assert names[0] == names[1] and names[0]
        for name in names[0]:
            assert without_unit_keys(left, folder, name) == without_unit_keys(right, folder, name), name
    archives = [json.loads((run / "archive.json").read_text()) for run in runs]
    for entry in (entry for archive in archives for entry in archive):
        for key in PCX2_ARCHIVE_KEYS:
            entry.pop(key, None)
    assert archives[0] == archives[1]
    accounting = [json.loads((run / "summary.json").read_text())["proposal_accounting"] for run in runs]
    for record in accounting:
        for key in PCX2_ACCOUNTING_KEYS:
            record.pop(key, None)
    assert accounting[0] == accounting[1]
    for round_number in (1, 2):
        record = json.loads((right / "proposals" / f"t{round_number:02d}.json").read_text())
        assert [entry["wave"] for entry in record["waves"]] == [0, 1]  # a later-round refill ran
        assert all(set(entry) == {"wave", "calls", "accepted"} for entry in record["waves"])
    assert not list(right.rglob("t00_burst*"))
