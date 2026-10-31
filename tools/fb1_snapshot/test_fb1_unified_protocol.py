"""The unified protocol's ForceBench options. No paid call, no network, no key.

FB1_INITIAL_PROPOSALS=<K> (round-0 burst), FB1_PROPOSAL_LIMIT=3 (one call answering three candidates per round,
COMPOSITION v2), FB1_ARCHIVE_REVIEW_ROUNDS (ARCHIVE_REVIEW v1 rounds with LLM-chosen revivals), FB1_DESIGN_POLICY
(joint / Model / Parameter expected information gain (EIG), expected predictive information gain (EPIG), random;
EIG parts and point disagreement recorded), FB1_PACKET_ARCHIVE.

Run: cd tools/fb1_snapshot && env -u FORCE_COLOR JAX_PLATFORMS=cpu DP_SCIENCEAGENT_DIR=<ScienceAgent dir> \
      FB1_PYTHON=<jax venv python> FB1_OUT_DIR=/tmp/x/ \
      <jax venv python> -m pytest -q --confcutdir=. test_fb1_unified_protocol.py
Injected scripted clients and synthetic packets/posteriors only. The runner's option checks import it in a child
interpreter (test_fb1_label_rule.native_import, skipped without FB1_PYTHON); the record checks run the runner with
FB1_T_EXTRA=0.
"""
import ast
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

SNAPSHOT = Path(__file__).resolve().parent
ROOT = SNAPSHOT.parents[1]
sys.path.insert(0, str(SNAPSHOT))
os.environ.setdefault("FB1_OUT_DIR", "/tmp/fb1_unified_protocol_test_out/")
os.environ.setdefault("FB1_WORLD", "gravity")
# DiscoverPhysics @ 33b7fa9: DP_SCIENCEAGENT_DIR, or a checkout beside the repository
os.environ.setdefault("DP_SCIENCEAGENT_DIR",
                      str(Path(__file__).resolve().parents[3] / "DiscoverPhysics" / "ScienceAgent"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fb1_core as core  # noqa: E402
import fb1_design_search as fds  # noqa: E402
import fb1_grammar as fbg  # noqa: E402
import fb1_proposer as fp  # noqa: E402
import run_gravity as rg  # noqa: E402
from test_fb1_label_rule import native_import  # noqa: E402
from test_fb1_proposer import _state  # noqa: E402

TIMES = np.linspace(0.5, 5.0, 10)
CONTEXT = {"run_id": "t", "world": "gravity", "arm": "111", "policy": "composed_3", "event": 0, "round": 1,
           "n_designs": 3, "candidate_index": 0, "packet_file": "t03_c0.json"}
VALID = [{"expression": "c*p1/(r**2 + d)", "family": "gauss", "parameters": {"c": "positive", "d": "positive"}},
         {"expression": "c*p1/r**3", "family": "gauss", "parameters": {"c": "positive"}},
         {"expression": "c*p1*tanh(r/d)", "family": "student_t4", "parameters": {"c": "positive", "d": "positive"}}]
M_OUTER, N_INNER = 32, 24
PLAIN = lambda c, D: (lambda u, a, b: core.make_predict(c)(u, a, b, rg.POS0, rg.VEL0, rg.TIMES))  # noqa: E731


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Autouse fixture: unset run-affecting FB1 environment variables and reset the prompt variant and input-name mask afterwards."""
    for name in ("FB1_LLM_REASONING", "FB1_MAX_PAID_CALLS", "FB1_LLM_MAX_TOKENS", "FB1_LLM_PROVIDER",
                 "FB1_PROMPT_VARIANT", "FB1_MASK_INPUT_NAMES"):
        monkeypatch.delenv(name, raising=False)
    yield
    fp.set_prompt_variant(None)
    fp.set_input_name_mask(None)


def free_state():
    """Return the synthetic (rows, members, free-form constraints, interface) state used by the proposer tests."""
    rows, members, _constraints, interface = _state()
    constraints = fp.free_constraints("gauss", family_expansion=True, times=TIMES)
    return rows, members, constraints, interface


def free_packet():
    """Return a serialized belief packet and its free-form constraints for the proposer tests."""
    rows, members, constraints, interface = free_state()
    return fp.serialize_packet(fp.build_belief_packet(rows, members, [], constraints, interface, 1)), constraints


def _count_proposer_constants():
    """Read the burst paragraph constants from the count proposer's source without importing it; skip if the source is absent."""
    path = ROOT / "src/bayes_model_discovery/experimental/count_proposer.py"
    if not path.is_file():
        pytest.skip("the engine source is not in this checkout")
    tree = ast.parse(path.read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("BURST_PARAGRAPH", "BURST_REFILL_PARAGRAPH"):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    return found


def test_the_burst_paragraphs_are_the_count_runners_bytes():
    """The burst paragraphs equal the count runner's constants byte for byte."""
    assert _count_proposer_constants() == {"BURST_PARAGRAPH": fp.BURST_PARAGRAPH,
                                           "BURST_REFILL_PARAGRAPH": fp.BURST_REFILL_PARAGRAPH}


def test_request_prompt_default_is_the_stored_file_and_each_kind_appends_its_paragraphs():
    """The default prompt is the stored file; each request kind appends its paragraphs and records their digests; invalid kind/limit pairs raise."""
    base = fp.system_prompt_bytes().decode()
    assert fp.request_prompt(None) == (base, fp.prompt_identity())
    for kind, paragraph, key in (("burst", fp.BURST_PARAGRAPH, "burst_prompt_sha256"),
                                 ("burst_refill", fp.BURST_REFILL_PARAGRAPH, "burst_refill_prompt_sha256")):
        text, identity = fp.request_prompt(kind)
        assert text == base + paragraph
        assert identity == {**fp.prompt_identity(), key: hashlib.sha256(text.encode()).hexdigest()}
    v2 = fp.COMPOSITION_PARAGRAPH_V2.replace("{limit}", "3")
    review = fp.ARCHIVE_REVIEW_PARAGRAPH_V1.replace("{limit}", "3")
    text, identity = fp.request_prompt("composed", 3)
    assert text == base + v2 and "exactly 3 candidate objects" in text and "{limit}" not in text
    assert identity == {**fp.prompt_identity(), "proposal_limit": 3,
                        "composition_paragraph_sha256": fp.COMPOSITION_PARAGRAPH_V2_SHA256,
                        "composed_prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}
    text, identity = fp.request_prompt("composed_review", 3)
    assert text == base + v2 + review and '{"candidates": [...], "revive": [...]}' in text
    assert identity["archive_review_paragraph_sha256"] == fp.ARCHIVE_REVIEW_PARAGRAPH_V1_SHA256
    assert identity["composed_prompt_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    for bad in (("composed", None), ("composed", 2), ("composed_review", 4), ("burst", 3), (None, 3),
                ("sequential", None)):
        with pytest.raises(ValueError):
            fp.request_prompt(*bad)
    with pytest.raises(ValueError, match="limit 3 only"):
        fp.fill_limit(fp.COMPOSITION_PARAGRAPH_V2, 2)


def test_a_variant_file_is_composed_and_named_beside_the_paragraph_digests(tmp_path):
    """A prompt variant file is composed into the request and its identity is recorded next to the paragraph digests."""
    variant = tmp_path / "variant.txt"
    variant.write_text("# prompt_version: vT.synthetic\nPropose one model as one JSON object.\n")
    fp.set_prompt_variant(str(variant))
    text, identity = fp.request_prompt("composed_review", 3)
    assert text.startswith(variant.read_text()) and identity["prompt_version"] == "vT.synthetic"
    assert identity["prompt_sha256"] == hashlib.sha256(variant.read_bytes()).hexdigest()
    assert identity["composed_prompt_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    ids = fp.request_prompt_identities()
    assert ids["base_prompt_sha256"] == identity["prompt_sha256"]
    assert ids["composed_review_prompt_sha256"] == identity["composed_prompt_sha256"]


# ================================================================== the proposer: composed calls
def test_a_composed_call_sends_the_v2_paragraph_and_keeps_every_accepted_object(tmp_path):
    """A composed call sends the v2 paragraph and keeps every accepted candidate while recording rejected ones."""
    packet, constraints = free_packet()
    reply = [VALID[0], {**VALID[1], "family": "poisson"}, VALID[2]]
    client = fp.ScriptedClient([json.dumps(reply)])
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client, structure_mode="free")
    got, record = proposer.propose_one(packet, context=CONTEXT, constraints=constraints,
                                       request="composed", proposal_limit=3)
    assert [c["expression"] for c in got] == [VALID[0]["expression"], VALID[2]["expression"]]
    assert record["status"] == "parsed" and record["calls"] == 1 and record["valid_candidate_count"] == 2
    assert [r["index"] for r in record["rejected_candidates"]] == [1]
    assert record["notes"] == ["", ""] and record["proposal_limit"] == 3
    assert client.bodies[0]["system"].endswith(fp.COMPOSITION_PARAGRAPH_V2.replace("{limit}", "3"))
    assert record["composition_paragraph_sha256"] == fp.COMPOSITION_PARAGRAPH_V2_SHA256
    request = json.loads((tmp_path / "records" / "event_00_round_01_c0.request.json").read_text())
    assert request["identity"]["composed_prompt_sha256"] == record["composed_prompt_sha256"]


def test_a_composed_batch_failure_earns_one_array_repair_turn(tmp_path):
    """A whole-batch failure in a composed call earns exactly one array repair turn."""
    packet, constraints = free_packet()
    client = fp.ScriptedClient([json.dumps(VALID[0]), json.dumps(VALID[:2])])
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client, structure_mode="free")
    got, record = proposer.propose_one(packet, context=CONTEXT, constraints=constraints,
                                       request="composed", proposal_limit=3)
    assert len(got) == 2 and record["repairs"] == 1
    assert record["rejected_candidates"][0]["reason"] == "reply must be a JSON array of 3 candidates"
    assert "Reply with exactly one JSON array of 3 objects" in client.bodies[1]["messages"][-1]["content"]


def test_an_object_reply_off_a_review_round_earns_the_array_repair_turn(tmp_path):
    """An object reply outside a review round is repaired once with an array-only instruction."""
    packet, constraints = free_packet()
    client = fp.ScriptedClient([json.dumps({"candidates": VALID, "revive": []}), json.dumps(VALID)])
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client, structure_mode="free")
    got, record = proposer.propose_one(packet, context=CONTEXT, constraints=constraints,
                                       request="composed", proposal_limit=3)
    assert len(got) == 3 and record["repairs"] == 1
    assert record["rejected_candidates"][0]["reason"] == "reply must be a JSON array of 3 candidates"
    assert "Reply with exactly one JSON array of 3 objects" in client.bodies[1]["messages"][-1]["content"]


@pytest.mark.parametrize(("text", "message"), [
    ("not json", "reply is not valid JSON"),
    (json.dumps(VALID[0]), "reply must be a JSON array of 3 candidates"),
    ("[]", "reply must hold 1..3 candidates"),
    (json.dumps(VALID + VALID[:1]), "reply must hold 1..3 candidates"),
    (json.dumps([{**VALID[0], "family": "poisson"}, 7]), "no candidate in the array was accepted"),
])
def test_the_array_parser_refuses_only_whole_batch_failures(text, message):
    """The candidate-array parser raises only for whole-batch failures."""
    constraints = fp.free_constraints("gauss", family_expansion=True, times=TIMES)
    with pytest.raises(ValueError, match=message):
        fp.parse_candidate_array(text, constraints, 3)


def test_the_review_reply_parser_keeps_the_revive_list_as_written_and_reads_a_bare_array():
    """The review reply parser keeps the revive list as written and accepts a bare array with an empty revive list."""
    constraints = fp.free_constraints("gauss", family_expansion=True, times=TIMES)
    got, ignored = fp.parse_review_reply(json.dumps({"candidates": VALID, "revive": ["a" * 20, 5], "why": "x"}),
                                         constraints, 3)
    assert len(got) == 3 and ignored["revive_requested"] == ["a" * 20, 5] and ignored["reply_shape"] == "object"
    assert ignored["ignored_review_keys"] == {"why": "x"} and ignored["revive_present"] is True
    got, ignored = fp.parse_review_reply(json.dumps({"candidates": VALID[:1]}), constraints, 3)
    assert ignored["revive_requested"] == [] and ignored["revive_present"] is False
    got, ignored = fp.parse_review_reply(json.dumps({"candidates": VALID[:1], "revive": "a" * 20}), constraints, 3)
    assert ignored["revive_requested"] == ["a" * 20] and ignored["revive_shape"] == "str"
    got, ignored = fp.parse_review_reply(json.dumps(VALID), constraints, 3)     # accepted, revive empty
    assert len(got) == 3 and ignored["reply_shape"] == "array" and ignored["revive_requested"] == []
    assert ignored["revive_present"] is False
    for text, message in ((json.dumps({"revive": []}), "one JSON object"),
                          (json.dumps({"candidates": VALID[0]}), "candidates must be a JSON array of 3"),
                          ("x", "not valid JSON")):
        with pytest.raises(ValueError, match=message):
            fp.parse_review_reply(text, constraints, 3)


def test_a_masked_review_call_is_judged_in_the_engines_names_and_maps_the_revive_ids_back(tmp_path):
    """A masked review call is judged in the engine's names and the revive ids are mapped back to engine ids."""
    mask = fp.InputNameMask()
    fp.set_input_name_mask(mask)
    rows, members, constraints, interface = free_state()
    archived = fp.model_id("c*p1/r**2", "gauss", {"c": "positive"})
    archive = [{"model_id": archived, "expression": "c*p1/r**2", "family": "gauss", "proposed_round": 0,
                "outcome": "retired_by_evidence",
                "reason": {"log_evidence_at_retirement": -9.0, "gap_to_cutoff_nats": 1.0, "active_rounds": [0],
                           "computed_round": 1, "n_obs": 3}}]
    shown_packet = mask.mask_value(fp.build_belief_packet(rows, members, archive, constraints, interface, 5))
    shown_id = shown_packet["archive"][0]["model_id"]
    assert shown_id != archived and "X1" in shown_packet["archive"][0]["expression"]
    shown_candidates = [json.loads(mask.mask_reply(json.dumps(v))) for v in VALID]
    reply = json.dumps({"candidates": shown_candidates, "revive": [shown_id, "f" * 20]})
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=fp.ScriptedClient([reply]), structure_mode="free",
                                     name_mask=mask)
    got, record = proposer.propose_one(fp.serialize_packet(shown_packet), context=CONTEXT, constraints=constraints,
                                       request="composed_review", proposal_limit=3)
    assert [c["expression"] for c in got] == [v["expression"] for v in VALID]
    assert record["revive_requested"] == [shown_id, "f" * 20]
    assert record["revive_requested_engine_ids"] == [archived, "f" * 20]
    assert record["responses"][0]["raw_text"] == reply


# ================================================================== the proposer: burst calls
BURST_CONTEXT = {**CONTEXT, "policy": "initial_burst", "round": 0, "n_designs": 2, "packet_file": "t02_burst_w0.json"}


def test_a_burst_call_needs_its_wave_and_index_and_names_its_records(tmp_path):
    """A burst call requires its wave and call index and names its record files by them."""
    packet, constraints = free_packet()
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=fp.ScriptedClient([json.dumps(VALID[0])] * 2),
                                     structure_mode="free")
    with pytest.raises(ValueError, match="identity required"):
        proposer.propose_one(packet, context=BURST_CONTEXT, constraints=constraints, request="burst")
    with pytest.raises(ValueError, match="identity required"):
        proposer.propose_one(packet, context={**CONTEXT, "burst_wave": 0, "burst_call_index": 0},
                             constraints=constraints)
    got, record = proposer.propose_one(packet, context={**BURST_CONTEXT, "burst_wave": 1, "burst_call_index": 3},
                                       constraints=constraints, request="burst_refill")
    assert len(got) == 1 and "burst_refill_prompt_sha256" in record
    assert (tmp_path / "records" / "round_00__burst_w1_c03.response.json").is_file()


def test_the_unified_mock_answers_each_kind_and_reads_the_revive_list_off_the_packet(tmp_path):
    """The mock transport answers each request kind and builds the revive list from the packet archive, masked or not."""
    rows, members, constraints, interface = free_state()
    retired = [{"model_id": f"{i:020x}", "expression": "c*p1/r", "family": "gauss", "proposed_round": 0,
                "outcome": "retired_by_evidence",
                "reason": {"log_evidence_at_retirement": -9.0, "gap_to_cutoff_nats": 1.0, "active_rounds": [0],
                           "computed_round": 1, "n_obs": 3}} for i in (1, 2, 3)]
    packet = fp.serialize_packet(fp.build_belief_packet(rows, members, retired, constraints, interface, 5))
    for mask in (None, fp.InputNameMask()):
        proposer = fp.Fb1CatalogProposer(tmp_path / f"records_{mask is not None}", structure_mode="free",
                                         name_mask=mask)
        assert proposer.transport == "mock"
        text = packet if mask is None else fp.serialize_packet(mask.mask_value(json.loads(packet)))
        first, _ = proposer.propose_one(text, context={**BURST_CONTEXT, "burst_wave": 0, "burst_call_index": 0},
                                        constraints=constraints, request="burst")
        composed, record = proposer.propose_one(text, context=CONTEXT, constraints=constraints,
                                                request="composed", proposal_limit=3)
        review, review_record = proposer.propose_one(text, context={**CONTEXT, "candidate_index": 1, "packet_file": "y"},
                                                     constraints=constraints, request="composed_review",
                                                     proposal_limit=3)
        default, _ = proposer.propose_one(text, context={**CONTEXT, "candidate_index": 2, "packet_file": "x"},
                                          constraints=constraints)
        assert first[0]["expression"] == fp.BURST_MOCK_SCHEDULE[0][0]
        assert [c["expression"] for c in composed] == [e for e, _f, _p in fp.COMPOSED_MOCK_SCHEDULE[0]]
        assert [c["expression"] for c in review] == [e for e, _f, _p in fp.COMPOSED_MOCK_SCHEDULE[1]]
        assert default[0]["expression"] == fp.FREE_MOCK_SCHEDULE[0][0]   # the default mock queue is untouched
        shown_ids = [row["model_id"] for row in json.loads(text)["archive"]]
        assert review_record["revive_requested"] == [shown_ids[0], fp.MOCK_UNKNOWN_MODEL_ID, shown_ids[1]]
        assert ("X1" in record["responses"][0]["raw_text"]) == (mask is not None)


def test_the_mock_revives_the_latest_nearest_miss_first_then_an_unknown_then_an_over_limit_id():
    """The mock revive list orders the latest nearest misses first, then an unknown id, then an over-limit id."""
    def row(i, computed, gap, outcome="retired_by_evidence"):
        return {"model_id": f"{i:020x}", "outcome": outcome,
                "reason": {"computed_round": computed, "gap_to_cutoff_nats": gap}}
    archive = [row(1, 1, 0.1), row(2, 3, 2.0), row(3, 3, 0.5), row(4, 3, 0.5), row(5, 4, 0.1, "active"),
               row(6, 2, float("nan"))]
    assert fp.mock_revive_list(json.dumps({"archive": archive})) == [f"{3:020x}", fp.MOCK_UNKNOWN_MODEL_ID,
                                                                     f"{1:020x}"]
    assert fp.mock_revive_list(json.dumps({"archive": [row(5, 4, 0.1, "active")]})) == [fp.MOCK_UNKNOWN_MODEL_ID]
    assert fp.mock_revive_list("not a packet") == [fp.MOCK_UNKNOWN_MODEL_ID]


def test_the_mock_schedules_are_valid_grammar_with_declared_outcomes_and_a_paid_cells_load():
    """The mock schedules are valid grammar with declared outcomes and cover the load of a full paid cell."""
    assert len(fp.BURST_MOCK_EXPECTED_OUTCOMES) == len(fp.BURST_MOCK_SCHEDULE) >= 14
    assert all(len(triple) == 3 for triple in fp.COMPOSED_MOCK_SCHEDULE)
    generated = [entry for triple in fp.COMPOSED_MOCK_SCHEDULE[8:] for entry in triple]
    assert len(fp.COMPOSED_MOCK_SCHEDULE) - 8 >= 29                # rounds 9..30 of a 30-round run stay new
    ids = [fbg.structure_model_id(e, p) for e, _f, p in generated]
    assert len(set(ids)) == len(ids)
    times = np.asarray(rg.TIMES)
    for expression, _family, parameters in fp.BURST_MOCK_SCHEDULE + [e for t in fp.COMPOSED_MOCK_SCHEDULE for e in t]:
        struct, record = fbg.compile_structure(expression, parameters)
        if expression in ("c*p1*log(r - 5)",):
            continue                                                # the declared non-finite burst entry
        probe = fbg.finiteness_probe(struct, record["model_id"], times)
        assert not probe["nonfinite_points"] and probe["finite"], (expression, probe.get("reason"))


def test_the_trig_mock_forms_compile_and_stay_finite_under_the_trig_grammar(tmp_path):
    """The trigonometric mock forms compile and stay finite under the trig grammar (checked in a child interpreter)."""
    code = ("import numpy as np, fb1_grammar as g, fb1_proposer as p\n"
            "bad = []\n"
            "for e, f, q in p.TRIG_MOCK_FORMS:\n"
            "    s, rec = g.compile_structure(e, q)\n"
            "    pr = g.finiteness_probe(s, rec['model_id'], r.TIMES)\n"
            "    bad += [] if (pr['finite'] and not pr['nonfinite_points']) else [e]\n"
            "print(json.dumps(bad))")
    proc = native_import(tmp_path, {**FREE, "FB1_GRAMMAR_TRIG": "1"}, suffix=code)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout.splitlines()[0]) == []


class SlowClient(fp.ScriptedClient):
    """Thread-safe replies after a short pause, so parallel calls overlap; the provider name keeps the budget rule."""

    provider = "anthropic"

    def __init__(self, text):
        super().__init__([])
        self.text, self.lock, self.in_flight, self.max_in_flight = text, threading.Lock(), 0, 0

    def _create(self, **body):
        with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            index = self.calls
            self.calls += 1
        time.sleep(0.05)
        with self.lock:
            self.in_flight -= 1
        return fp._Response(self.text, model="slow", response_id=f"slow_{index:03d}",
                            usage={"input_tokens": 1, "output_tokens": 1})


def test_parallel_burst_calls_count_every_attempt_once_and_stop_at_the_cap(tmp_path):
    """Parallel burst calls count every attempt once and stop at the paid-call cap."""
    from concurrent.futures import ThreadPoolExecutor
    packet, constraints = free_packet()
    client = SlowClient(json.dumps(VALID[0]))
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client, structure_mode="free", max_paid_calls=12)

    def call(index):
        return proposer.propose_one(packet, context={**BURST_CONTEXT, "burst_wave": 0, "burst_call_index": index},
                                    constraints=constraints, request="burst")
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(call, range(12)))
    assert client.max_in_flight > 1 and proposer.attempts == 12 == client.calls
    assert len(list((tmp_path / "records").glob("round_00__burst_w0_c*.response.json"))) == 12
    assert all(len(got) == 1 for got, _ in results)
    with pytest.raises(fp.PaidCallBudgetExhausted):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: proposer.propose_one(
                packet, context={**BURST_CONTEXT, "burst_wave": 1, "burst_call_index": i},
                constraints=constraints, request="burst_refill"), range(4)))
    assert proposer.attempts == 12 and (tmp_path / "records" / fp.STOP_SUMMARY_FILE).is_file()


# ================================================================== the empty-pool packet
def test_the_empty_pool_packet_has_the_belief_packets_keys_and_no_trajectory():
    """The empty-pool packet has the belief packet's keys, no trajectory and no evaluation information."""
    rows, members, constraints, interface = free_state()
    full = fp.build_belief_packet(rows, members, [], constraints, interface, 0)
    empty = fp.build_empty_pool_packet(rows, [], constraints, interface, 0)
    assert set(empty) == set(full) and empty["pool"] == [] and empty["best_model_residuals"] == []
    assert empty["observations"] == full["observations"] and empty["best_model"] is None
    assert "y" not in json.dumps(empty["observations"])
    names = list(inspect.signature(fp.build_empty_pool_packet).parameters)
    assert names == ["public_observations", "archive", "constraints", "interface", "round"]
    body = inspect.getsource(fp.build_empty_pool_packet).replace(fp.build_empty_pool_packet.__doc__, "")
    for word in ("heldout", "evaluation", "Q_HEAD", "Q_SHIFT", "coverage90", "truth"):
        assert word not in body


def test_the_runner_unit_helpers_take_no_evaluation_arguments():
    """The runner's helper functions take no held-out or evaluation arguments and reference no evaluation names."""
    for function in (rg._llm_burst, rg._llm_propose_composed, rg._validate_revive, rg._admit_llm_candidate,
                     rg._packet_archive,
                     rg.design_policy_record_ascent, rg.design_policy_record_random, rg.point_disagreement,
                     rg._archive_retirement, rg._archive_resurrection, rg._top_by_evidence):
        names = list(inspect.signature(function).parameters)
        assert not [n for n in names for f in ("heldout", "held_out", "evaluation", "truth", "score", "nll")
                    if f in n.lower()], (function, names)
        body = inspect.getsource(function)
        for word in ("TR_H", "TR_S", "YH_H", "YH_S", "Q_HEAD", "Q_SHIFT", "score_set", "VAR_GT"):
            assert word not in body, (function.__name__, word)


# ================================================================== protocol U2 in the runner (in this process)
@pytest.fixture
def free_runner(monkeypatch):
    """run_gravity's helpers as a free-structure LLM run sees them (this process imported it with the defaults)."""
    monkeypatch.setattr(rg, "FREE_STRUCTURES", True)
    monkeypatch.setattr(rg, "fbp", fp, raising=False)
    monkeypatch.setattr(rg, "fbg", fbg, raising=False)
    yield rg
    fbg.reset_registry()


def prior_fits(structures, seed=3, particles=40):
    """Build prior-sampled (candidate, particles) fits for the given (structure, family) pairs."""
    key = jax.random.PRNGKey(seed)
    fits = []
    for struct, family in structures:
        key, k = jax.random.split(key)
        cand = core.Candidate(struct, family, 2)
        fits.append((cand, core.sample_prior_u(k, particles, cand.kinds())))
    return fits


def composed_round(tmp_path, reply, *, review=False, archive_rows=()):
    """Run one composed proposal round through the runner with a scripted client; return its result and the client."""
    fits = prior_fits([("inv_r2", "gauss"), ("linear", "gauss")])
    client = fp.ScriptedClient([json.dumps(reply)])
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client, structure_mode="free")
    constraints = fp.free_constraints("gauss", family_expansion=True, times=np.asarray(rg.TIMES))
    interface = fp.forcebench_interface(measurement_times=rg.TIMES, dt=rg.DT, p1_range=rg.P1_RANGE,
                                        p2_range=rg.P2_RANGE, integration_steps=rg._integration_steps())
    rng = np.random.default_rng(5)
    designs = [[1.0, 1.0], [2.0, 3.0], [4.0, 2.0]]
    ys = [rng.normal(size=(rg.T_OBS, 2)) for _ in designs]
    (tmp_path / "packets").mkdir()
    result = rg._llm_propose_composed(
        proposer, n_designs=3, round_t=1, event_index=0, designs=designs, ys=ys, fits=fits,
        weights=np.array([0.7, 0.3]), log_evidence=np.array([-1.0, -2.0]), one_step={},
        archive_rows=list(archive_rows), pool_keys={("inv_r2", "gauss"), ("linear", "gauss")},
        constraints=constraints, interface=interface, packets_dir=str(tmp_path / "packets"), run_id="t",
        world="gravity", arm="111", policy="composed_3", review=review)
    return result, client


def test_one_composed_call_admits_in_reply_order_and_records_exact_duplicates(tmp_path, free_runner):
    """One composed call admits candidates in reply order and records exact duplicates with their matched models."""
    reply = [{"expression": "c*p1/(r**2 + d)", "family": "gauss", "parameters": {"c": "positive", "d": "positive"}},
             {"expression": "p1*c/(r**2 + d)", "family": "gauss", "parameters": {"c": "positive", "d": "positive"}},
             {"expression": "c*p1/r**2", "family": "gauss", "parameters": {"c": "positive"}}]
    (pairs, event, additions, revive), client = composed_round(tmp_path, reply)
    assert len(client.bodies) == 1 and client.bodies[0]["system"].endswith(
        fp.COMPOSITION_PARAGRAPH_V2.replace("{limit}", "3"))
    assert event["outcomes"] == ["active", "duplicate", "duplicate"] and event["admitted"] == 1 == len(pairs)
    assert additions[1]["reason"]["matched_model_id"] == pairs[0][0]   # a commuted transcription: numerically equal
    assert additions[2]["reason"]["matched_model_id"] == "inv_r2"      # the pool's catalogue law
    assert event["kind"] == "composed" and revive["requested"] == [] == revive["requested_engine_ids"]
    assert sorted(p.name for p in (tmp_path / "packets").iterdir()) == ["t03_c0.json"]


def test_a_review_round_returns_the_revive_request_and_validation_keeps_two_retired_ids(tmp_path, free_runner):
    """A review round returns the revive request and validation keeps at most two retired ids, labelling the rest."""
    retired = {f"{i:020x}": {"model_id": f"{i:020x}", "expression": "c*p1/r", "family": "gauss",
                             "proposed_round": 0, "outcome": "retired_by_evidence",
                             "reason": {"log_evidence_at_retirement": -9.0, "gap_to_cutoff_nats": 1.0,
                                        "active_rounds": [0], "computed_round": 1, "n_obs": 3}} for i in (1, 2, 3)}
    active = {"a" * 20: {**retired[f"{1:020x}"], "model_id": "a" * 20, "outcome": "active",
                         "reason": {"admitted_round": 0, "active_rounds": [0, 1]}}}
    archive = {**retired, **active}
    names = list(retired)
    reply = {"candidates": [{"expression": "c*p1/r**3", "family": "gauss", "parameters": {"c": "positive"}}],
             "revive": [names[0], "f" * 20, "a" * 20, names[0], 7, names[1], names[2]]}
    (pairs, event, _adds, revive), client = composed_round(tmp_path, reply, review=True,
                                                          archive_rows=archive.values())
    assert client.bodies[0]["system"].endswith(fp.ARCHIVE_REVIEW_PARAGRAPH_V1.replace("{limit}", "3"))
    assert event["kind"] == "composed_review" and event["reply_shape"] == "object" and len(pairs) == 1
    assert revive["requested"] == reply["revive"] == revive["requested_engine_ids"]
    valid, invalid = rg._validate_revive(revive["requested"], revive["requested_engine_ids"], archive, 2)
    assert valid == names[:1] and [item["why"] for item in invalid] == ["unknown"] + ["over_limit"] * 5
    valid, invalid = rg._validate_revive(["a" * 20], ["a" * 20], archive, 2)
    assert valid == [] and invalid[0]["why"] == "active"
    duplicate_row = {**retired[names[1]], "model_id": "b" * 20, "outcome": "duplicate",
                     "reason": {"matched_model_id": "x", "matched_expression": "y"}}
    archive = {**archive, "b" * 20: duplicate_row}
    for ids, expected_valid, expected_why in (
            ([names[0], names[1], names[2]], names[:2], ["over_limit"]),
            ([names[0], names[0]], names[:1], ["repeated"]),
            (["b" * 20, names[2]], names[2:], ["not_retired"]),
            ([7, names[1]], names[1:2], ["unknown"]),
            ([], [], [])):
        valid, invalid = rg._validate_revive(ids, ids, archive, 2)
        assert valid == expected_valid and [item["why"] for item in invalid] == expected_why
        assert all(item["why"] in rg.REVIVE_INVALID_CODES for item in invalid)


def test_the_pool_cut_keeps_the_top_by_evidence_with_ties_to_the_earlier_and_non_finite_last():
    """The pool cut keeps the top models by evidence, breaking ties toward the earlier and ranking non-finite values last."""
    assert rg._top_by_evidence([-3.0, -1.0, -2.0, -1.0], 2) == ([1, 3], [0, 2])
    # both non-finite values rank last, the earlier of them first
    assert rg._top_by_evidence([float("nan"), -5.0, float("-inf"), -4.0], 3) == ([0, 1, 3], [2])
    assert rg._top_by_evidence([float("nan"), -5.0, float("-inf"), -4.0], 2) == ([1, 3], [0, 2])
    assert rg._top_by_evidence([-1.0, -2.0], 12) == ([0, 1], [])


def test_retirement_and_resurrection_rows_are_valid_packet_archive_rows(free_runner):
    """Retirement and resurrection rows are valid packet archive rows."""
    archive = {}
    cand = core.Candidate("screened", "gauss", rg.D0 + 3)
    mid = rg._archive_retirement(archive, cand, -12.5, -10.0, rg.D0 + 4)
    row = archive[mid]
    assert row["outcome"] == "retired_by_evidence" and row["reason"]["gap_to_cutoff_nats"] == 2.5
    assert row["proposed_round"] == 3 and fp._archive_entry(row)["outcome"] == "retired_by_evidence"
    rg._archive_resurrection(archive, mid, 5)
    again = archive[mid]
    assert again["outcome"] == "active" and again["reason"] == {"admitted_round": 3, "active_rounds": [5]}
    assert again["resurrected_rounds"] == [5] and fp._archive_entry(again)["outcome"] == "active"
    rg._archive_retirement(archive, cand, -20.0, -11.0, rg.D0 + 7)
    assert archive[mid]["resurrected_rounds"] == [5] and archive[mid]["reason"]["active_rounds"] == [5]


def test_the_compact_packet_archive_drops_only_the_two_diagnostic_blocks(monkeypatch):
    """The compact packet archive drops only the two diagnostic blocks and leaves the run's own archive intact."""
    rows = [{"model_id": "a" * 20, "expression": "c*p1/r", "family": "gauss", "proposed_round": 1,
             "outcome": "duplicate", "reason": {"matched_model_id": "b", "matched_expression": "x"},
             "reason_code": "duplicate_by_numerical_equivalence", "numerical_equivalence": {"big": 1},
             "prior_probe": {"big": 2}, "note": "n"}]
    assert rg._packet_archive(rows) == rows                            # the default is every key
    monkeypatch.setattr(rg, "PACKET_ARCHIVE", "compact")
    compact = rg._packet_archive(rows)
    assert compact == [{k: v for k, v in rows[0].items() if k not in ("numerical_equivalence", "prior_probe")}]
    assert rows[0]["prior_probe"] == {"big": 2}                       # the run's own archive keeps them


def test_the_catalogue_switch_takes_the_law_catalogue_out_of_what_the_model_reads():
    """The catalogue switch removes the law catalogue from the constraints the model reads and leaves the input untouched."""
    rows, members, _c, interface = _state()
    on = fp.free_constraints("gauss", family_expansion=True, times=TIMES)
    off = fp.catalogue_off_constraints(on)
    assert set(off) == set(on) and off["initial_structures"] == [] and off["structure_parameters"] == {}
    assert on["initial_structures"] and on["grammar"]["catalogue_expressible"]           # the input is untouched
    assert not set(fp.CATALOGUE_GRAMMAR_KEYS) & set(off["grammar"])
    assert not set(fp.CATALOGUE_DUPLICATE_KEYS) & set(off["grammar"]["duplicate_detection"])
    assert {k: v for k, v in off["grammar"].items() if k != "duplicate_detection"} == {
        k: v for k, v in on["grammar"].items() if k not in fp.CATALOGUE_GRAMMAR_KEYS and k != "duplicate_detection"}
    packet = fp.serialize_packet(fp.InputNameMask().mask_value(
        fp.build_belief_packet(rows, members, [], off, interface, 2)))
    for word in ("K11", "K12", "initial_structures\":[\"", "catalogue_expressible"):
        assert word not in packet
    candidate, _ = fp.parse_candidate_object(json.dumps(VALID[0]), off)    # the judge does not need the catalogue
    assert candidate["expression"] == VALID[0]["expression"]
    with pytest.raises(ValueError, match="free-form constraints only"):
        fp.catalogue_off_constraints(fp.catalog_constraints())
    # second commit: no grammar text names the catalogue any more; the numerical check keeps its other words
    assert not re.search("catalog", json.dumps(off), re.IGNORECASE)
    assert off["grammar"]["n_ops_max"] == on["grammar"]["n_ops_max"] and "n_ops_max_basis" not in off["grammar"]
    stage_2 = on["grammar"]["duplicate_detection"]["stage_2_numerical"]
    assert fp.CATALOGUE_NUMERICAL_PHRASE in stage_2
    assert off["grammar"]["duplicate_detection"]["stage_2_numerical"] == stage_2.replace(
        fp.CATALOGUE_NUMERICAL_PHRASE, "")
    assert off["grammar"]["duplicate_detection"]["reason_codes"] == {
        k: v for k, v in on["grammar"]["duplicate_detection"]["reason_codes"].items() if k != "catalog_reference"}


def test_with_the_catalogue_off_the_pool_and_archive_carry_no_catalogue_trace(monkeypatch):
    """With the catalogue off, the pool and archive in the packet carry no trace of the catalogue."""
    rows, members, _c, interface = _state()            # a pool of two catalogue laws (inv_r2, screened)
    off = fp.catalogue_off_constraints(fp.free_constraints("gauss", family_expansion=True, times=TIMES))
    inv_r2 = fp.structure_expression("inv_r2")
    inv_r2_id = fp.structure_candidate_identity("inv_r2", "gauss")
    assert "^" in inv_r2 and inv_r2_id in {m.model_id for m in members}

    def link(basis):
        return {"structure": "inv_r2", "basis": basis, "priors_match": True, "catalog_expression": inv_r2,
                "free_expression_as_proposed": "c*p1/r**2", "counts_as_new_structure": False, "decision": "d"}
    compared = {"comparisons": [{"target": "screened", "target_kind": "catalog", "pairing": {"c": "c"},
                                 "max_relative_difference": 0.5}], "matched": None, "matched_kind": None}
    archive = [
        {"model_id": inv_r2_id, "expression": inv_r2, "family": "gauss", "proposed_round": 0, "outcome": "active",
         "reason": {"admitted_round": 0, "active_rounds": [0, 1]}, "reason_code": fbg.REASON_CATALOG_REFERENCE,
         "catalog_reference": link("canonical_id"), "numerical_equivalence": compared,
         "prior_probe": {"finite": True, "n_points": np.int64(266240)}, "note": "n"},   # numpy, as a runner row may
        {"model_id": "a" * 20, "expression": "p1*c/r**2", "family": "gauss", "proposed_round": 1,
         "outcome": "duplicate", "reason": {"matched_model_id": "inv_r2", "matched_expression": inv_r2},
         "reason_code": fbg.REASON_CATALOG_REFERENCE, "catalog_reference": link("canonical_id"), "note": "n"},
        {"model_id": "b" * 20, "expression": "c*p1*r/r**3", "family": "gauss", "proposed_round": 1,
         "outcome": "duplicate", "reason": {"matched_model_id": "inv_r2", "matched_expression": inv_r2},
         "reason_code": fbg.REASON_CATALOG_REFERENCE, "catalog_reference": link("probe_grid"), "note": "n"},
        {"model_id": "c" * 20, "expression": "c*p1/r**3", "family": "gauss", "proposed_round": 1,
         "outcome": "active", "reason": {"admitted_round": 1, "active_rounds": [1]}, "reason_code": None,
         "numerical_equivalence": compared, "prior_probe": {"finite": True}, "note": "n"},
    ]
    packet = fp.build_belief_packet(rows, members, archive, off, interface, 2)
    before = fp.serialize_packet(packet)
    view = fp.catalogue_off_packet(packet)
    assert fp.serialize_packet(packet) == before                          # a new packet; the argument is untouched
    text = fp.serialize_packet(fp.InputNameMask().mask_value(view))
    assert "^" not in text and not re.search("catalog", text, re.IGNORECASE)
    assert not re.search(r"\bK(0[1-9]|1[0-4])\b", text)                     # no masked catalogue name either
    shown = {row["model_id"]: row for row in view["archive"]}
    assert shown[inv_r2_id]["expression"] == inv_r2.replace("^", "**") and shown[inv_r2_id]["reason_code"] is None
    assert shown[inv_r2_id]["prior_probe"] == {"finite": True, "n_points": 266240}
    assert "numerical_equivalence" not in shown["c" * 20]
    assert shown["a" * 20]["reason_code"] == fbg.REASON_DUPLICATE_CANONICAL
    assert shown["b" * 20]["reason_code"] == fbg.REASON_DUPLICATE_NUMERICAL
    assert shown["a" * 20]["reason"] == {"matched_model_id": inv_r2_id, "matched_expression": inv_r2.replace("^", "**")}
    assert [m["model_id"] for m in view["pool"]] == [m["model_id"] for m in packet["pool"]]
    assert [m["expression"] for m in view["pool"]] == [m["expression"].replace("^", "**") for m in packet["pool"]]
    assert view["best_model"] == packet["best_model"].replace("^", "**")
    assert {k: v for k, v in view.items() if k not in ("pool", "archive", "best_model")} == {
        k: v for k, v in packet.items() if k not in ("pool", "archive", "best_model")}
    # the runner applies it only when the switch is off: on, the packet object passes through unchanged
    monkeypatch.setattr(rg, "fbp", fp, raising=False)
    monkeypatch.setattr(rg, "CATALOGUE", "on")
    assert rg._catalogue_view(packet) is packet
    monkeypatch.setattr(rg, "CATALOGUE", "off")
    assert rg._catalogue_view(packet) == view


def test_the_unified_one_step_definition_says_when_a_candidate_is_first_scored():
    """The unified one-step score definition states when a candidate is first scored and differs from the base definitions only there."""
    changed = fp.unified_summary_definitions()
    assert changed["pool"]["one_step_log_scores"] == fp.UNIFIED_ONE_STEP_DEFINITION
    assert {k: v for k, v in changed["pool"].items() if k != "one_step_log_scores"} == {
        k: v for k, v in fp.FIELD_DEFINITIONS["pool"].items() if k != "one_step_log_scores"}
    assert fp.FIELD_DEFINITIONS["pool"]["one_step_log_scores"].endswith("first score appears at round t+2")
    assert fp.InputNameMask().mask_text(fp.UNIFIED_ONE_STEP_DEFINITION) == fp.UNIFIED_ONE_STEP_DEFINITION


# ================================================================== the EIG parts and the disagreement
@pytest.fixture(scope="module")
def posterior():
    """Module-scoped fixture: prior-sampled fits for three candidates and their model weights."""
    key = jax.random.PRNGKey(20261005)
    cands = [core.Candidate("inv_r2", "gauss", 0), core.Candidate("screened", "gauss", 0),
             core.Candidate("power", "student_t4", 0)]
    fits = []
    for c in cands:
        key, k = jax.random.split(key)
        fits.append((c, core.sample_prior_u(k, 60, c.kinds())))
    return fits, np.array([0.6, 0.3, 0.1])


def designs_of(seed, count):
    """Return `count` random (p1, p2) designs drawn with the given seed."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0.1, 10, count), rng.uniform(0.1, 10, count)])


def test_the_joint_column_of_the_parts_is_eig_pure_and_the_parts_add_up(posterior):
    """The joint column of the EIG parts equals the pure EIG and the model and parameter parts add up to it."""
    fits, w = posterior
    draws = fds.eig_draws(jax.random.PRNGKey(5), fits, w, M_OUTER, N_INNER)
    designs = jnp.asarray(designs_of(7, 6))
    joint, _ = fds.eig_pure(designs, fits, w, draws, PLAIN, rg.log_dens, rg.T_OBS)
    parts = fds.eig_parts_pure(designs, fits, w, draws, PLAIN, rg.log_dens, rg.T_OBS)
    assert np.array_equal(np.asarray(parts["values"][0]), np.asarray(joint))
    total = np.asarray(parts["values"][1]) + np.asarray(parts["values"][2])
    assert np.max(np.abs(total - np.asarray(joint))) < 1e-10
    rows = fds.eig_parts_rows(np.asarray(designs), parts, M_OUTER)
    assert max(r["identity_gap"] for r in rows) < 1e-10 and all(r["model_eig_se"] >= 0 for r in rows)


def test_one_model_has_no_model_eig_and_identical_particles_have_no_parameter_eig(posterior):
    """A single model has zero model EIG and identical particles have zero parameter EIG."""
    fits, _ = posterior
    draws = fds.eig_draws(jax.random.PRNGKey(6), fits[:1], np.array([1.0]), M_OUTER, N_INNER)
    parts = fds.eig_parts_pure(jnp.asarray(designs_of(8, 3)), fits[:1], np.array([1.0]), draws, PLAIN,
                               rg.log_dens, rg.T_OBS)
    assert np.max(np.abs(np.asarray(parts["values"][1]))) < 1e-12
    same = [(c, jnp.repeat(u[:1], 40, axis=0)) for c, u in fits]
    w = np.array([0.5, 0.3, 0.2])
    draws = fds.eig_draws(jax.random.PRNGKey(7), same, w, M_OUTER, N_INNER)
    parts = fds.eig_parts_pure(jnp.asarray(designs_of(9, 3)), same, w, draws, PLAIN, rg.log_dens, rg.T_OBS)
    assert np.max(np.abs(np.asarray(parts["values"][2]))) < 1e-9
    assert np.all(np.asarray(parts["values"][1]) > 0)


@pytest.mark.parametrize("objective", ["model_eig", "parameter_eig"])
def test_a_part_objective_ascends_that_part_and_chooses_its_holdout_argmax(posterior, objective):
    """A part objective ascends that EIG part and picks the holdout argmax of the same column."""
    fits, w = posterior
    rec = fds.run_design_search(jax.random.PRNGKey(11), np.random.default_rng(11), fits, w, objective=objective,
                                p1_range=(0.1, 10.0), p2_range=(0.1, 10.0), m_outer=M_OUTER, n_inner=N_INNER,
                                predict=PLAIN, log_dens=rg.log_dens, t_obs=rg.T_OBS, steps=4)[2]
    parts = rec["eig_parts_holdout_rows"]
    column = [r[objective] for r in parts]
    assert [r["objective"] for r in rec["holdout_rows"]] == column
    assert rec["chosen_restart_index"] == int(np.argmax(column)) + 1 and rec["objective"] == objective
    assert all(np.isfinite(t["arrival_value"]) for t in rec["trajectories"])


def test_record_parts_leaves_the_joint_search_unchanged_and_adds_the_parts(posterior):
    """Recording the parts leaves the joint search result unchanged and only adds the part records."""
    fits, w = posterior
    common = dict(objective="eig", p1_range=(0.1, 10.0), p2_range=(0.1, 10.0), m_outer=M_OUTER, n_inner=N_INNER,
                  predict=PLAIN, log_dens=rg.log_dens, t_obs=rg.T_OBS, steps=3)
    plain = fds.run_design_search(jax.random.PRNGKey(12), np.random.default_rng(12), fits, w, **common)
    parts = fds.run_design_search(jax.random.PRNGKey(12), np.random.default_rng(12), fits, w, record_parts=True,
                                  **common)
    extra = {k: parts[2][k] for k in set(parts[2]) - set(plain[2])}
    assert set(extra) == {"eig_parts_holdout_rows", "eig_parts_draws"}
    trimmed = dict(parts[2], objective_evaluations={k: v for k, v in parts[2]["objective_evaluations"].items()
                                                    if k != "eig_parts_holdout_calls"})
    assert {k: v for k, v in trimmed.items() if k not in extra} == plain[2]
    assert np.array_equal(plain[0], parts[0]) and plain[1] == parts[1]
    gap = max(abs(r["joint_eig"] - h["objective"]) for r, h in zip(extra["eig_parts_holdout_rows"],
                                                                    plain[2]["holdout_rows"]))
    assert gap < 1e-12


def test_the_epig_search_records_the_eig_parts_on_its_own_holdout_draws(posterior):
    """The EPIG search records the EIG parts on its own holdout draws."""
    fits, w = posterior
    targets = rg.sample_epig_targets(13, count=3)[0]
    rec = fds.run_design_search(jax.random.PRNGKey(13), np.random.default_rng(13), fits, w, objective="epig",
                                p1_range=(0.1, 10.0), p2_range=(0.1, 10.0), m_outer=M_OUTER, n_inner=N_INNER,
                                predict=PLAIN, log_dens=rg.log_dens, t_obs=rg.T_OBS, target_points=targets, steps=2,
                                record_parts=True)[2]
    assert rec["objective"] == "epig" and len(rec["eig_parts_holdout_rows"]) == fds.RESTARTS
    assert rec["chosen_restart_index"] == int(np.argmax([r["objective"] for r in rec["holdout_rows"]])) + 1


def test_the_disagreement_is_the_weighted_variance_of_the_mean_forecasts():
    """Point disagreement equals the weighted variance of the mean forecasts, None for non-finite input, and rejects bad weights."""
    rng = np.random.default_rng(3)
    means = rng.normal(size=(4, 2, 5, 2))
    w = np.array([0.1, 0.2, 0.3, 0.4])
    mbar = np.einsum("k,kdta->dta", w, means)
    expected = np.einsum("k,kdta->d", w, (means - mbar) ** 2) / 10
    assert np.allclose(fds.point_disagreement_values(means, w), expected, rtol=0, atol=1e-15)
    one = fds.point_disagreement_values(means[:, :1, :1, :1].repeat(2, axis=-1), w)
    assert one == pytest.approx([float(np.dot(w, (means[:, 0, 0, 0] - np.dot(w, means[:, 0, 0, 0])) ** 2))],
                                rel=1e-12)
    means[2, 1, 0, 0] = np.inf
    assert fds.point_disagreement_values(means, w)[1] is None
    with pytest.raises(ValueError):
        fds.point_disagreement_values(means, w * 2)


def test_the_disagreement_rule_is_the_shared_count_definition_per_coordinate():
    """Point disagreement matches the shared count-task definition applied per coordinate (skipped if the engine package is not importable)."""
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from bayes_model_discovery.experimental.design_policies import point_disagreement as shared
    except Exception as exc:  # noqa: BLE001 - the engine package needs its own interpreter
        pytest.skip(f"design_policies not importable here: {type(exc).__name__}")
    finally:
        sys.path.remove(str(ROOT / "src"))
    rng = np.random.default_rng(4)
    means = rng.normal(size=(3, 2, 4, 2))
    w = np.array([0.5, 0.25, 0.25])
    per_coordinate = shared(means.reshape(3, -1), w).reshape(2, 4, 2).mean(axis=(1, 2))
    assert np.allclose(fds.point_disagreement_values(means, w), per_coordinate, rtol=0, atol=1e-15)


# ================================================================== the runner's options (child interpreter)
FREE = {"FB1_PROPOSER": "llm", "FB1_STRUCTURE_PROPOSALS": "free"}


@pytest.mark.parametrize(("extra", "message"), [
    ({"FB1_INITIAL_PROPOSALS": "12"}, "requires FB1_PROPOSER=llm and FB1_STRUCTURE_PROPOSALS=free"),
    ({**FREE, "FB1_INITIAL_PROPOSALS": "13"}, "FB1_INITIAL_PROPOSALS must be an integer in 1..12"),
    ({**FREE, "FB1_INITIAL_PROPOSALS": "-1"}, "FB1_INITIAL_PROPOSALS must be an integer in 1..12"),
    ({**FREE, "FB1_INITIAL_PROPOSALS": "012"}, "FB1_INITIAL_PROPOSALS must be an integer in 1..12"),
    ({**FREE, "FB1_ACTIVE_MAXIMUM": "10"}, "FB1_ACTIVE_MAXIMUM must be 12"),
    ({"FB1_PROPOSER": "llm", "FB1_PROPOSAL_LIMIT": "3"}, "requires FB1_PROPOSER=llm and FB1_STRUCTURE"),
    ({**FREE, "FB1_PROPOSAL_LIMIT": "2"}, "FB1_PROPOSAL_LIMIT must be 3"),
    ({**FREE, "FB1_PROPOSAL_LIMIT": "1"}, "FB1_PROPOSAL_LIMIT must be 3"),
    ({**FREE, "FB1_PROPOSAL_LIMIT": "3", "FB1_PROPOSAL_POLICY": "P4"}, "unset FB1_PROPOSAL_POLICY"),
    ({**FREE, "FB1_ARCHIVE_REVIEW_ROUNDS": "5,10"}, "set FB1_PROPOSAL_LIMIT=3 with it"),
    ({**FREE, "FB1_PROPOSAL_LIMIT": "3", "FB1_ARCHIVE_REVIEW_ROUNDS": "0,5"}, "entries must be 1..16"),
    ({**FREE, "FB1_PROPOSAL_LIMIT": "3", "FB1_ARCHIVE_REVIEW_ROUNDS": "5,17"}, "entries must be 1..16"),
    ({**FREE, "FB1_PROPOSAL_LIMIT": "3", "FB1_ARCHIVE_REVIEW_ROUNDS": "five"}, "must list integers"),
    ({**FREE, "FB1_PACKET_ARCHIVE": "short"}, "FB1_PACKET_ARCHIVE must be full or compact"),
    ({"FB1_PACKET_ARCHIVE": "compact"}, "requires FB1_PROPOSER=llm and FB1_STRUCTURE_PROPOSALS=free"),
    ({**FREE, "FB1_CATALOGUE": "no"}, "FB1_CATALOGUE must be on or off"),
    ({"FB1_CATALOGUE": "off"}, "requires FB1_PROPOSER=llm and FB1_STRUCTURE_PROPOSALS=free"),
    ({"FB1_DESIGN_POLICY": "structure_eig"}, "FB1_DESIGN_POLICY must be one of"),
    ({"FB1_DESIGN_POLICY": "random"}, "needs arm design digit 0"),
    ({"FB1_ARM": "110", "FB1_DESIGN_POLICY": "joint_eig"}, "needs arm design digit 1"),
    ({"FB1_ARM": "111", "FB1_DESIGN_POLICY": "epig", "FB1_DESIGN_SEARCH": "ascent"}, "needs arm design digit 2"),
    ({"FB1_DESIGN_POLICY": "model_eig"}, "requires FB1_DESIGN_SEARCH=ascent"),
    ({"FB1_ARM": "112", "FB1_DESIGN_POLICY": "epig"}, "requires FB1_DESIGN_SEARCH=ascent"),
])
def test_a_bad_or_inconsistent_option_is_refused_before_any_output(tmp_path, extra, message):
    """A bad or inconsistent option combination makes the runner exit with the expected message before any output."""
    proc = native_import(tmp_path, extra)
    assert proc.returncode != 0 and message in proc.stderr


def settings_of(proc):
    """Return the recorded binding settings from a child run's last stdout line; assert the child succeeded."""
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])["binding"]["settings"]


U3_OPTIONS = {"FB1_INITIAL_PROPOSALS": "12", "FB1_ACTIVE_MAXIMUM": "12", "FB1_PROPOSAL_LIMIT": "3",
              "FB1_ARCHIVE_REVIEW_ROUNDS": "5,10,15", "FB1_DESIGN_POLICY": "model_eig"}


def test_the_options_are_declared_to_the_label_rule_after_every_earlier_key(tmp_path):
    """The new options are declared to the label rule after every earlier settings key and change only the expected entries."""
    plain = settings_of(native_import(tmp_path, {**FREE, "FB1_DESIGN_SEARCH": "ascent"}))
    on = settings_of(native_import(tmp_path, {**FREE, "FB1_DESIGN_SEARCH": "ascent", **U3_OPTIONS}))
    assert list(on)[:len(plain)] == list(plain)
    changed = {k for k in plain if on[k] != plain[k]}
    assert changed == {"proposal_policy", "proposal_events"}
    assert on["proposal_policy"] == "composed_3" and on["proposal_events"] == list(range(3, 19))
    assert on["candidates_per_event"] == 3
    assert {k: on[k] for k in list(on)[len(plain):]} == {
        "initial_proposals": 12, "active_maximum": 12, "proposal_limit": 3,
        "archive_review_rounds": [5, 10, 15], "design_policy": "model_eig"}
    epig = settings_of(native_import(tmp_path, {"FB1_ARM": "112", "FB1_DESIGN_SEARCH": "ascent"}))
    flagged = settings_of(native_import(tmp_path, {"FB1_ARM": "112", "FB1_DESIGN_SEARCH": "ascent",
                                                   "FB1_DESIGN_POLICY": "epig"}))
    assert flagged == epig   # the EPIG arm already declares design_policy "epig"
    off = settings_of(native_import(tmp_path, {**FREE, "FB1_DESIGN_SEARCH": "ascent", **U3_OPTIONS,
                                               "FB1_CATALOGUE": "off"}))
    assert off == {**on, "catalogue": "off"}
    thirty = settings_of(native_import(tmp_path, {**FREE, "FB1_DESIGN_SEARCH": "ascent", **U3_OPTIONS,
                                                  "FB1_T_EXTRA": "30", "FB1_ARCHIVE_REVIEW_ROUNDS": "5 10 15 20 25 30"}))
    assert thirty["proposal_events"] == list(range(3, 33)) and thirty["archive_review_rounds"] == [5, 10, 15, 20, 25, 30]


def test_the_command_line_flags_set_the_same_variables(tmp_path):
    """The command-line flags set the same environment variables and conflicting flags are refused."""
    flags = ["--initial-proposals", "12", "--active-maximum", "12", "--proposal-limit", "3",
             "--archive-review-rounds", "5,10,15", "--design-policy", "random", "--packet-archive", "compact",
             "--catalogue", "off"]
    show = (f"r._apply_cli_flags({flags!r}); print(json.dumps({{name: os.environ.get(name) for _f, name in r.CLI_FLAGS}}))")
    proc = native_import(tmp_path, {}, prefix="import os", suffix=show)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout.splitlines()[0]) == {
        "FB1_INITIAL_PROPOSALS": "12", "FB1_ACTIVE_MAXIMUM": "12", "FB1_PROPOSAL_LIMIT": "3",
        "FB1_ARCHIVE_REVIEW_ROUNDS": "5,10,15", "FB1_DESIGN_POLICY": "random", "FB1_PACKET_ARCHIVE": "compact",
        "FB1_CATALOGUE": "off"}
    clash = native_import(tmp_path, {"FB1_DESIGN_POLICY": "random", "FB1_ARM": "110"}, prefix="import os",
                          suffix="r._apply_cli_flags(['--design-policy', 'joint_eig'])")
    assert clash.returncode != 0 and "contradicts FB1_DESIGN_POLICY=random" in clash.stderr


# ================================================================== whole-run records (child runs, FB1_T_EXTRA=0)


def run_child(snapshot, out_dir, extra, timeout=1500):
    """Run the runner in a child process with a scrubbed environment (no API keys) in the given snapshot; return the completed process."""
    dp = os.environ.get("DP_SCIENCEAGENT_DIR", "")
    if not (Path(dp) / "scienceagent").is_dir():
        pytest.skip("set DP_SCIENCEAGENT_DIR for runner checks")
    env = {k: v for k, v in os.environ.items() if not k.startswith("FB1_")}
    env.update(JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENAI_API_KEY="", ANTHROPIC_API_KEY="", OPENAI_COMPAT_API_KEY="",
               DP_SCIENCEAGENT_DIR=dp, FB1_OUT_DIR=str(out_dir) + os.sep, **extra)
    env.pop("FORCE_COLOR", None)
    return subprocess.run([sys.executable, "-B", "run_gravity.py"], cwd=snapshot, env=env, capture_output=True,
                          text=True, timeout=timeout, check=False)


ZERO_ROUND = {"FB1_WORLD": "gravity", "FB1_ARM": "111", "FB1_SEED": "31", "FB1_T_EXTRA": "0", **FREE,
              "FB1_MASK_INPUT_NAMES": "1", "FB1_RUN_LABEL": "development", "FB1_DESIGN_SEARCH": "ascent"}
RECORD = "FB1_RUN_gravity_111_seed31_v3_free.json"


@pytest.fixture(scope="module")
def zero_round_runs(tmp_path_factory):
    """Module-scoped fixture: run the default and burst zero-round child runs and return their output folders."""
    folders = {name: tmp_path_factory.mktemp(f"zero_round_{name}") for name in ("default", "burst")}
    for name, snapshot, extra in (("default", SNAPSHOT, {}),
                                  ("burst", SNAPSHOT, {"FB1_INITIAL_PROPOSALS": "12"})):
        done = run_child(snapshot, folders[name], {**ZERO_ROUND, **extra})
        assert done.returncode == 0, (name, done.stderr[-3000:])
    return folders


def test_the_burst_zero_round_run_admits_twelve_and_records_the_burst(zero_round_runs):
    """The burst zero-round run admits twelve distinct models and records the burst waves, archive and packets."""
    folder = zero_round_runs["burst"]
    record = json.loads((folder / RECORD).read_text())
    burst = record["initial_burst"]
    assert burst["requested"] == 12 and burst["admitted"] == 12 and burst["shortfall"] == 0
    assert burst["requested_per_wave"] == [12, 2] and burst["accepted_per_wave"] == [10, 2]
    assert len(record["initial_pool"]) == 12 and record["checkpoints"]["2"]["active"] == [
        f"{s}|{f}" for s, f in record["initial_pool"]]
    assert record["arm_config"]["unified_protocol"]["initial_proposals"] == 12
    assert len({tuple(pair) for pair in record["initial_pool"]}) == 12          # twelve distinct (structure, family)
    assert [len(e["proposed"]) for e in record["proposer"]["initial_burst_events"]] == [10, 2]
    outcomes = [a["outcome"] for a in record["proposer"]["archive"] if a["proposed_round"] == 0]
    assert sorted(outcomes) == sorted(fp.BURST_MOCK_EXPECTED_OUTCOMES[:14])
    records = sorted(p.name for p in folder.rglob("round_00__burst_w*_c*.response.json"))
    assert len(records) == 14 and records[-1] == "round_00__burst_w1_c01.response.json"
    packets = sorted(p.name for p in folder.rglob("t02_burst_w*.json"))
    assert packets == ["t02_burst_w0.json", "t02_burst_w1.json"]
    wave0 = json.loads(next(folder.rglob("t02_burst_w0.json")).read_text())
    wave1 = json.loads(next(folder.rglob("t02_burst_w1.json")).read_text())
    assert wave0["pool"] == [] and len(wave1["pool"]) == 10 and "X1" in json.dumps(wave1["constraints"])
    assert record["code_sha256"]["fb1_proposer.py"]
