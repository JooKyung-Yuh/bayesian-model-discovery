"""ChemBench records every candidate event after round 0, and the round-0 burst's refill policy
is a registered runtime setting (a project decision of 2026-09-23).

Synthetic replies, the runner's mock path and stubbed fits only: no network, no saved run, no
stored response, and no expression or seed taken from a real run. The sealed proposal-shape tool
is read and executed from its own file and never written to; the byte comparison with the base
commit runs only with BMD_BASELINE_COMPARE=1.
"""

import glob
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from . import chembench_core as cc
from . import count_proposer as cp
from .count_inference import FitTimeout
from .count_models import FAILURE_CLASSES, INITIAL_BURST_REFILL_POLICY, model_id_preimage
from .run_label_policy import (
    BURST_REFILL_KEYS,
    BURST_REFILL_POLICY_KEY,
    bmd_runtime_settings,
    verify_manifest,
)
from .test_chembench_core import FAKE_NAMES, _args, _manifest
from .test_chembench_mda_options import stub_fits as stub_chembench_fits
from .test_failure_class_and_archive_parameters import (
    FORMS,
    STATUS_CLASSES,
    ZERO,
    check_classes,
    keyed_client,
    plan_for,
    recount,
    run_chembench,
)
from .test_run_label_policy import registration, runner_args, write_manifest

KEY = BURST_REFILL_POLICY_KEY
# The base commit whose sources the opt-in byte comparison extracts; it skips when this commit is absent from the
# checkout's history.
BASE_COMMIT = "147dcd944afa2755215c91cc7ff24918f54c3e7f"
#: The sealed analysis tool (prompt optimization, 2026-09-20). Read and executed, never written.
#: Its path comes from this environment variable - an external document, so no machine path is
#: written in this file - and the test that reads it skips when unset.
SEALED_TOOL_ENV = "BMD_PROMPT_SHAPE_TOOL"
SEALED_TOOL_SHA256 = "433ef767b8dd7966fa1c535c8f31ad295676d930cfecc7646ef81fd0d76720f7"
PAIR = {"b0": "intercept", "b1": "signed"}


def form(index, name="f0z"):
    """A synthetic two-coefficient form over one grammar input (b0 the intercept, b1 signed)."""
    return f"b0+b1*tanh({name}+{index}/20)"


def classed(out):
    """Every later round's record, in round order (t00_burst.json is round 0's)."""
    return [json.loads(path.read_text()) for path in sorted((out / "proposals").glob("t[0-9][0-9].json"))]


def check_event_identity(event):
    """An event names its candidate: the model_id is the hash of the preimage it records."""
    assert event["model_id_preimage"] == model_id_preimage(
        event["expression"], event["family"], tuple(event["parameters"].items()))
    assert hashlib.sha256(event["model_id_preimage"].encode()).hexdigest()[:20] == event["model_id"]


# --------------------------------------------------------------------------
# 1. every later round records one event per candidate (the mock path)
# --------------------------------------------------------------------------

A, B, C, D = (form(index) for index in (1, 2, 3, 4))
GAMMA = (form(1, "f1z"), form(2, "f1z"))  # the F0 cell refuses the family
REFUSED = (form(1, "f2z"), form(2, "f2z"))  # the prior probe refuses them
TIMED_OUT = (form(1, "f3z"), form(2, "f3z"))  # their fit times out
#: K = 3. Wave 0: A, A again, a gamma form; wave 1 asks for the two missing and gets a refused
#: prior and a timed-out fit; wave 2 asks for the two again and gets B and C.
BURST_SCHEDULE = [(A, "lognormal"), (A, "lognormal"), (GAMMA[0], "gamma"),
                  (REFUSED[0], "lognormal"), (TIMED_OUT[0], "lognormal"),
                  (B, "lognormal"), (C, "lognormal")]
BURST_STATUSES = ["fitted", "duplicate_not_refitted", "rejected_family", "rejected_prior_probe",
                  "fit_failed", "fitted", "fitted"]
#: Rounds 1..5, one candidate each: every outcome a later round can have on this path.
ROUND_SCHEDULE = [(A, "lognormal"), (GAMMA[1], "gamma"), (REFUSED[1], "lognormal"),
                  (TIMED_OUT[1], "lognormal"), (D, "lognormal")]
ROUND_STATUSES = ["duplicate_not_refitted", "rejected_family", "rejected_prior_probe", "fit_failed",
                  "fitted"]


class PhasedSchedule:
    """The runner's 20-entry mock schedule, indexed exactly as the runner indexes it, answering
    from the burst's forms until the burst record is written and from the rounds' after it."""

    def __init__(self, burst, rounds):
        """Pad the burst and round tables to the runner's mock schedule length, starting in the burst phase."""
        pad = [(form(10 + index), "lognormal") for index in range(cc.MOCK_SCHEDULE_ROUNDS)]
        self.tables = {"burst": (list(burst) + pad)[:cc.MOCK_SCHEDULE_ROUNDS],
                       "rounds": (list(rounds) + pad)[:cc.MOCK_SCHEDULE_ROUNDS]}
        self.phase = "burst"

    def __len__(self):
        """Return the runner's fixed mock schedule length."""
        return cc.MOCK_SCHEDULE_ROUNDS

    def __getitem__(self, index):
        """Return the entry at `index` from the table of the current phase."""
        return self.tables[self.phase][index]


def refuse_priors(monkeypatch, expressions):
    """Patch the prior probe so it refuses every hypothesis whose expression is in `expressions`."""
    original = cc._prior_probe

    def probe(hypothesis, *args, **kwargs):
        """Run the real prior probe, then mark a listed expression as failed."""
        record, predictive = original(hypothesis, *args, **kwargs)
        if hypothesis.expression in expressions:
            record = {**record, "passed": False, "detail": "synthetic prior-probe refusal"}
        return record, predictive

    monkeypatch.setattr(cc, "_prior_probe", probe)


def time_out_fits(monkeypatch, expressions):
    """Patch the fit so it raises FitTimeout for every hypothesis whose expression is listed."""
    original = cc.fit

    def fit(hypothesis, *args, **kwargs):
        """Raise FitTimeout for a listed expression, otherwise call the real fit."""
        if getattr(hypothesis, "expression", None) in expressions:
            raise FitTimeout(2.0, 1.0)
        return original(hypothesis, *args, **kwargs)

    monkeypatch.setattr(cc, "fit", fit)


def run_phased(tmp_path, monkeypatch):
    """Run the ChemBench mock path with the phased burst and round schedules; return the output folder."""
    stub_chembench_fits(monkeypatch)
    schedule = PhasedSchedule(BURST_SCHEDULE, ROUND_SCHEDULE)
    monkeypatch.setattr(cc, "_mock_schedule", lambda names, allowed: schedule)
    write = cc.write_final_proposal_json

    def write_then_switch(path, value):
        """Write the proposal record, then switch the schedule to the rounds once round 0 is written."""
        write(path, value)
        if path.name == "t00_burst.json":
            schedule.phase = "rounds"

    monkeypatch.setattr(cc, "write_final_proposal_json", write_then_switch)
    refuse_priors(monkeypatch, set(REFUSED))
    time_out_fits(monkeypatch, set(TIMED_OUT))
    out = tmp_path / "phased"
    cc.run(_args(out, _manifest(tmp_path), cell="100", structure_proposals="free",
                 rounds=len(ROUND_SCHEDULE), initial_proposals=3, initial_catalogue="off"))
    return out


def test_every_later_round_records_each_candidate_with_the_burst_event_keys(tmp_path, monkeypatch):
    """Each later round records one event per candidate, with the burst event's keys minus `source`."""
    out = run_phased(tmp_path, monkeypatch)
    burst = json.loads((out / "proposals" / "t00_burst.json").read_text())
    assert [event["status"] for event in burst["events"]] == BURST_STATUSES
    assert [(entry["calls"], entry["accepted"]) for entry in burst["waves"]] == [(3, 1), (2, 0), (2, 2)]
    # One key set per status among the burst's events, `source` aside.
    burst_keys = {}
    for event in burst["events"]:
        assert event["source"] == "initial_burst"
        keys = set(event) - {"source"}
        assert burst_keys.setdefault(event["status"], keys) == keys
    records = classed(out)
    assert [record["round"] for record in records] == [1, 2, 3, 4, 5]
    for record, (expression, family), status in zip(records, ROUND_SCHEDULE, ROUND_STATUSES, strict=True):
        event, = record["events"]
        assert [event["model_id"]] == record["proposed_ids"]
        assert (event["expression"], event["family"], event["parameters"]) == (expression, family, PAIR)
        assert (event["status"], event["failure_class"]) == (status, STATUS_CLASSES[status])
        assert "source" not in event and "initial_burst" not in json.dumps(record)
        # The burst event's key set on disk, minus `source` - model_id_preimage included.
        assert set(event) == burst_keys[status]
        check_event_identity(event)
    assert records[-1]["admitted_ids"] == [records[-1]["events"][0]["model_id"]]
    summary = json.loads((out / "summary.json").read_text())
    counts = summary["proposal_accounting"]["failure_class_counts"]
    # Each classed event counts once, the burst's and every later round's; the two
    # candidate_failures copies of the timed-out fits are not counted again.
    assert counts == recount(out) == {**ZERO, "parse_format": 2, "prior_probe": 2, "fit_failed": 2,
                                      "duplicate": 2}
    failures = summary["candidate_failures"]
    assert [(entry["round"], entry["failure_class"]) for entry in failures] == [(0, "fit_failed"),
                                                                              (4, "fit_failed")]


def test_a_round_with_two_candidates_records_one_event_each_in_proposal_order(tmp_path, monkeypatch):
    """The mock path without a burst, two candidates a round: the SAME events, in the order the
    fixed schedule proposes them."""
    stub_chembench_fits(monkeypatch)
    out = tmp_path / "two"
    cc.run(_args(out, _manifest(tmp_path), cell="100", structure_proposals="free", rounds=3,
                 proposal_limit=2))
    schedule = cc._mock_schedule(FAKE_NAMES, cc.CHEM_ALLOWED_FAMILIES)
    records = classed(out)
    statuses = []
    for t, record in enumerate(records, start=1):
        events = record["events"]
        assert [event["model_id"] for event in events] == record["proposed_ids"] and len(events) == 2
        assert [(event["expression"], event["family"]) for event in events] == [schedule[t - 1], schedule[t]]
        for event in events:
            assert "source" not in event and event["failure_class"] == STATUS_CLASSES[event["status"]]
            check_event_identity(event)
        statuses.append([event["status"] for event in events])
    assert statuses == [["rejected_family", "fitted"], ["duplicate_not_refitted", "rejected_family"],
                        ["duplicate_not_refitted", "fitted"]]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["proposal_accounting"]["failure_class_counts"] == recount(out) == {
        **ZERO, "parse_format": 2, "duplicate": 2}


def test_the_llm_path_records_and_counts_the_round_one_duplicate(tmp_path, monkeypatch):
    """The scripted LLM path: the round-1 reply (an empty body repaired into A) is a duplicate
    that the round's record names and the failure-class count includes."""
    forms = FORMS["chembench"]
    plan = plan_for(forms)
    seen = keyed_client(monkeypatch, plan)
    out, _, _, summary = run_chembench(tmp_path, monkeypatch, forms, initial_proposals=6,
                                       initial_catalogue="off")
    assert sorted(seen, key=repr) == sorted(plan, key=repr)
    statuses = check_classes(out)
    assert statuses.count("duplicate_not_refitted") == 2
    burst = json.loads((out / "proposals" / "t00_burst.json").read_text())
    later, = classed(out)
    event, = later["events"]
    assert (event["status"], event["failure_class"], event["expression"]) == (
        "duplicate_not_refitted", "duplicate", forms["A"])
    assert [event["model_id"]] == later["proposed_ids"] and "source" not in event
    duplicate = next(item for item in burst["events"] if item["status"] == "duplicate_not_refitted")
    assert set(event) == set(duplicate) - {"source"}
    check_event_identity(event)
    expected = {**ZERO, "parse_format": 2, "reserved_name": 1, "coefficient_count": 1,
                "prior_probe": 1, "fit_failed": 1, "duplicate": 2}
    assert summary["proposal_accounting"]["failure_class_counts"] == recount(out) == expected


# --------------------------------------------------------------------------
# 2. the sealed proposal-shape tool reads such a cell exactly as before
# --------------------------------------------------------------------------


def load_sealed_tool(monkeypatch):
    """Load the sealed proposal-shape tool from the environment path after checking its sha256."""
    # The path is read from the environment. Unset (or empty) skips;
    # set, it must name an existing file holding exactly the sealed bytes, otherwise the test fails.
    location = os.environ.get(SEALED_TOOL_ENV)
    if not location:
        pytest.skip(f"{SEALED_TOOL_ENV} is unset: set it to the sealed proposal-shape tool "
                    "(proposal_shape_metrics.py, prompt optimization 2026-09-20) to run this test")
    tool = Path(location)
    if not tool.is_file():
        pytest.fail(f"{SEALED_TOOL_ENV} names no file: {tool}")
    digest = hashlib.sha256(tool.read_bytes()).hexdigest()
    if digest != SEALED_TOOL_SHA256:
        pytest.fail(f"{SEALED_TOOL_ENV} is not the sealed proposal-shape tool: sha256 {digest}, "
                    f"sealed {SEALED_TOOL_SHA256}")
    # Loaded without writing a byte next to it: no __pycache__ entry is created or refreshed.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location("sealed_proposal_shape_metrics", tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reply(expression, parameters):
    """A synthetic JSON proposal reply for a lognormal candidate."""
    return json.dumps({"expression": expression, "family": "lognormal", "parameters": parameters})


def shape_event(expression, parameters, status, **extra):
    """A later-round event in the shape the runner now writes."""
    hypothesis = cc.ScalarHypothesis(expression, "lognormal", tuple(parameters.items()),
                                     ("f0", "f1", "f0z", "f1z"))
    return {**hypothesis.as_dict(), **extra, "status": status,
            "failure_class": STATUS_CLASSES[status],
            "model_id_preimage": hypothesis.model_id_preimage}


def shape_cell(root, with_events):
    """One synthetic ChemBench-like published cell: no archive (ChemBench writes none), config
    constraints, a burst record and two later rounds with transport records - round 1 answered
    on its first reply, round 2 refused once and repaired. `with_events` adds the two later
    records' per-candidate events and changes nothing else."""
    cell = root / "cell"
    (cell / "proposals").mkdir(parents=True)
    (cell / "config.json").write_text(json.dumps(
        {"constraints": {"initial_structures": ["b0", "b0+b1*f0z"]}}, indent=2) + "\n")
    first, repaired = "b0+b1*tanh(f0z)", "b0+b1*f0z+b2*square(f1z)"
    three = {**PAIR, "b2": "signed"}
    (cell / "proposals" / "t00_burst.json").write_text(json.dumps({
        "transports": [{"wave": 0, "index": 0, "transport": {"round": 0, "burst_wave": 0,
                                                             "status": "parsed", "repairs": 0}}],
        "events": [{**shape_event("b0+b1*f1z", PAIR, "fitted", log_evidence=-2.0),
                    "source": "initial_burst"}]}, indent=2) + "\n")
    rounds = {
        1: ({"round": 1, "status": "parsed", "calls": 1, "repairs": 0, "rejected_candidates": [],
             "responses": [{"attempt": 0, "raw_text": reply(first, PAIR)}]},
            [shape_event(first, PAIR, "fitted", log_evidence=-2.0)]),
        2: ({"round": 2, "status": "parsed", "calls": 2, "repairs": 1,
             "rejected_candidates": [{"attempt": 0, "reason": "reply is not valid JSON: synthetic",
                                      "error_type": "ValueError", "repair_eligible": True,
                                      "failure_class": "parse_format"}],
             "responses": [{"attempt": 0, "raw_text": "not json"},
                           {"attempt": 1, "raw_text": reply(repaired, three)}]},
            [shape_event(repaired, three, "duplicate_not_refitted")]),
    }
    for number, (transport, events) in rounds.items():
        record = {"round": number, "transport": transport, "proposed_ids": [e["model_id"] for e in events],
                  "admitted_ids": [], "retired_ids": [], "active_pool_after": []}
        if with_events:
            record["events"] = events
        (cell / "proposals" / f"t{number:02d}.json").write_text(json.dumps(record, indent=2) + "\n")
    return cell


def test_the_sealed_shape_tool_reads_a_cell_the_same_with_or_without_round_events(
        tmp_path, monkeypatch, capsys):
    """The sealed shape tool reads a cell identically whether or not its records carry round events."""
    tool = load_sealed_tool(monkeypatch)
    cells = {flag: shape_cell(tmp_path / ("with" if flag else "without"), flag) for flag in (False, True)}
    for number in (1, 2):
        name = Path("proposals") / f"t{number:02d}.json"
        stripped = json.loads((cells[True] / name).read_text())
        assert stripped.pop("events")
        assert stripped == json.loads((cells[False] / name).read_text())
    measured = {flag: [json.dumps(tool.measure([str(cell)], fallback), indent=2, ensure_ascii=False)
                       for fallback in (False, True)] for flag, cell in cells.items()}
    assert measured[True] == measured[False]
    # Not a vacuous comparison: the tool read both replies of the transport records.
    replies = json.loads(measured[True][1])
    assert (replies["cells_with_transport"], replies["first_reply"]["proposals"],
            replies["repair_reply"]["proposals"], replies["proposals"]) == (1, 1, 1, 2)
    printed = {}
    for flag, cell in cells.items():
        # A cell without an archive is read through the transport fallback ("+" arm prefix).
        assert tool.main([f"+arm={glob.escape(str(cell))}"]) == 0
        printed[flag] = capsys.readouterr()
    assert printed[True].out.encode() == printed[False].out.encode() and printed[True].out
    assert printed[True].err == printed[False].err == ""


# --------------------------------------------------------------------------
# 3. every other record is the base commit's, byte for byte
# --------------------------------------------------------------------------

PARITY_HARNESS = r'''
import sys
import time
from pathlib import Path
from pytest import MonkeyPatch
from bayes_model_discovery.experimental import chembench_core as cc
from bayes_model_discovery.experimental import test_failure_class_and_archive_parameters as pcx
from bayes_model_discovery.experimental.test_chembench_core import _args, _manifest
from bayes_model_discovery.experimental.test_chembench_mda_options import stub_fits
with MonkeyPatch.context() as patch:
    patch.setattr(time, "perf_counter", lambda: 0.0)
    # One burst call at a time, so max_in_flight and the call order are the same in both trees.
    patch.setattr(cc, "DEFAULT_BURST_CONCURRENCY", 1)
    root, suite = Path(sys.argv[1]), sys.argv[2]
    root.mkdir()
    if suite == "llm":
        forms = pcx.FORMS["chembench"]
        pcx.keyed_client(patch, pcx.plan_for(forms))
        pcx.run_chembench(root, patch, forms, initial_proposals=6, initial_catalogue="off")
    else:
        stub_fits(patch)
        cc.run(_args(root / "chembench", _manifest(root), cell="100", structure_proposals="free",
                     rounds=3, initial_proposals=3, initial_catalogue="off"))
'''
TIMING_SIDECARS = {"fit_timings.jsonl", "acquisition_timings.jsonl", "prediction_pool_summary.jsonl"}
CHANGED_SOURCES = {"chembench_core.py", "core_count.py"}


def normalized(path, tree):
    """config.json or summary.json with its tree's own path, the two changed source digests,
    the label policy's digest and the one new setting set aside (each checked by the caller)."""
    record = json.loads(path.read_text().replace(str(tree), "<tree>"))
    for key in ("wall_seconds", "user_cpu_seconds", "system_cpu_seconds", "timing_seconds"):
        record.pop(key, None)
    digests = record.get("source_sha256", {})
    changed = {name: digests.pop(name) for name in CHANGED_SOURCES & set(digests)}
    receipt = record.get("label_receipt") or {}
    policy = receipt.pop("policy_source_sha256", None)
    bound = receipt.get("binding", {}).get("settings", {}).pop(KEY, None)
    return record, changed, policy, bound


@pytest.mark.parametrize("suite", ["mock", "llm"])
def test_every_other_chembench_record_matches_the_base_commit_byte_for_byte(tmp_path, suite):
    """The burst record, packets, transports, fits, observations, admissions and identity are
    the base commit's bytes; a later round's record differs by its `events` alone, the count by exactly
    their classes, and the receipt by the one new setting."""
    if os.environ.get("BMD_BASELINE_COMPARE") != "1":
        pytest.skip("set BMD_BASELINE_COMPARE=1 for the byte comparison with the base commit")
    current = Path(__file__).resolve().parents[3]
    archive = subprocess.run(["git", "-C", str(current), "archive", BASE_COMMIT, "src", "adapters"],
                             capture_output=True, check=False)
    if archive.returncode:
        pytest.skip(f"base commit {BASE_COMMIT} is not in this checkout's history")
    base = tmp_path / "base_tree"
    base.mkdir()
    subprocess.run(["tar", "-x", "-C", str(base)], input=archive.stdout, check=True)
    external = [entry for entry in os.environ.get("PYTHONPATH", "").split(":")
                if entry and not entry.startswith(str(current))]
    same, runs = tmp_path / "same_root", {}
    for label, source in (("base", base), ("current", current)):
        env = dict(os.environ, PYTHONPATH=":".join([f"{source}/src", str(source), *external]),
                   PYTHONDONTWRITEBYTECODE="1", OPENAI_API_KEY="", ANTHROPIC_API_KEY="",
                   OPENAI_COMPAT_API_KEY="", OPENROUTER_API_KEY="")
        result = subprocess.run([sys.executable, "-B", "-c", PARITY_HARNESS, str(same), suite],
                                cwd=source, env=env, capture_output=True, text=True, check=False,
                                timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr
        target = tmp_path / label
        same.rename(target)
        runs[label] = (target / "chembench", source)
    (left, left_tree), (right, right_tree) = runs["base"], runs["current"]
    files = [sorted(str(path.relative_to(run)) for path in run.rglob("*") if path.is_file())
             for run in (left, right)]
    assert files[0] == files[1] and "proposals/t00_burst.json" in files[0]
    later = [name for name in files[0] if re.fullmatch(r"proposals/t[0-9]{2}\.json", name)]
    assert later
    added = dict.fromkeys(FAILURE_CLASSES, 0)
    for name in later:
        base_record, record = (json.loads((run / name).read_text()) for run in (left, right))
        events = record.pop("events")
        assert "events" not in base_record and record == base_record, name
        assert [event["model_id"] for event in events] == record["proposed_ids"]
        for event in events:
            if event["failure_class"] is not None:
                added[event["failure_class"]] += 1
    base_config, changed_base, policy_base, bound_base = normalized(left / "config.json", left_tree)
    config, changed, policy, bound = normalized(right / "config.json", right_tree)
    assert base_config == config
    assert set(changed) == set(changed_base) == CHANGED_SOURCES
    assert all(changed[name] != changed_base[name] for name in CHANGED_SOURCES)
    assert policy != policy_base and (bound_base, bound) == (None, INITIAL_BURST_REFILL_POLICY)
    base_summary, *_ = normalized(left / "summary.json", left_tree)
    summary, *_ = normalized(right / "summary.json", right_tree)
    base_counts = base_summary["proposal_accounting"].pop("failure_class_counts")
    counts = summary["proposal_accounting"].pop("failure_class_counts")
    assert counts == {kind: base_counts[kind] + added[kind] for kind in FAILURE_CLASSES}
    assert base_summary == summary
    special = {"config.json", "summary.json", "points.csv", *later}
    for name in files[0]:
        if name in special or Path(name).name in TIMING_SIDECARS:
            continue
        if name.endswith(".npz"):
            with np.load(left / name) as a, np.load(right / name) as b:
                assert a.files == b.files and all(
                    a[key].dtype == b[key].dtype and a[key].tobytes() == b[key].tobytes()
                    for key in a.files), name
            continue
        assert (left / name).read_bytes() == (right / name).read_bytes(), name
    points = (right / "points.csv").read_bytes().replace(
        changed["chembench_core.py"].encode(), changed_base["chembench_core.py"].encode())
    assert (left / "points.csv").read_bytes() == points


# --------------------------------------------------------------------------
# 4. the refill policy is a registered runtime setting
# --------------------------------------------------------------------------


def policy_args(bundle="core", **changes):
    """A runner-shaped namespace for a burst with initial proposals, stating the two refill
    digests and the refill rule exactly as the three runners state them; a None change removes
    that field."""
    identity = cp.burst_refill_identity()
    values = {"initial_proposals": 4, "initial_catalogue": "off",
              "burst_prompt_sha256": cp.composed_system_prompt(burst=True)[1]["burst_prompt_sha256"],
              **{key: identity[key] for key in BURST_REFILL_KEYS}, KEY: INITIAL_BURST_REFILL_POLICY,
              **({"initial_pool_source": "generated"} if bundle == "chembench" else {})}
    values.update(changes)
    return runner_args(**{key: value for key, value in values.items() if value is not None})


@pytest.mark.parametrize("bundle", ["core", "boxinggym", "chembench"])
def test_a_burst_binds_the_refill_policy_it_states_and_nothing_else_does(bundle):
    """A burst with initial proposals binds the refill policy it states; any other launch binds none."""
    bound = bmd_runtime_settings(bundle, policy_args(bundle))
    assert bound[KEY] == INITIAL_BURST_REFILL_POLICY
    # Bound as stated: the registration's exact comparison decides whether it is the registered rule.
    assert bmd_runtime_settings(bundle, policy_args(bundle, **{KEY: "synthetic_other_rule"}))[KEY] == (
        "synthetic_other_rule")
    for stated in (None, "", 3, [INITIAL_BURST_REFILL_POLICY]):
        with pytest.raises(ValueError, match="declares its refill policy"):
            bmd_runtime_settings(bundle, policy_args(bundle, **{KEY: stated}))
    assert KEY not in bmd_runtime_settings(bundle, runner_args())
    with pytest.raises(ValueError, match="refill policy needs a burst with initial proposals"):
        bmd_runtime_settings(bundle, runner_args(**{KEY: INITIAL_BURST_REFILL_POLICY}))
    if bundle != "chembench":
        parallel = {"parallel_proposals": 3, "burst_prompt_sha256": bound["burst_prompt_sha256"]}
        assert KEY not in bmd_runtime_settings(bundle, runner_args(**parallel))
        with pytest.raises(ValueError, match="refill policy needs a burst with initial proposals"):
            bmd_runtime_settings(bundle, runner_args(**parallel, **{KEY: INITIAL_BURST_REFILL_POLICY}))


@pytest.mark.parametrize("bundle", ["core", "boxinggym", "chembench"])
def test_the_registration_refuses_a_missing_altered_or_unexpected_refill_policy(tmp_path, bundle):
    """A registration whose refill policy is missing, altered or unexpected is refused."""
    settings = bmd_runtime_settings(bundle, policy_args(bundle))
    (tmp_path / "bound").mkdir()
    ref, context, manifest = registration(tmp_path / "bound", bundle=bundle, settings=settings)
    verify_manifest(ref, **context)
    for change in ("removed", "altered"):
        tampered = json.loads(json.dumps(manifest))
        if change == "removed":
            tampered["runtime_settings"][bundle].pop(KEY)
        else:
            tampered["runtime_settings"][bundle][KEY] = "synthetic_other_rule"
        other = write_manifest(tmp_path / "bound" / f"{change}.json", tampered)
        with pytest.raises(ValueError, match="Manifest runtime settings mismatch"):
            verify_manifest(other, **context)
    # A launch stating another rule is refused against the registered one.
    stated = bmd_runtime_settings(bundle, policy_args(bundle, **{KEY: "synthetic_other_rule"}))
    with pytest.raises(ValueError, match="Manifest runtime settings mismatch"):
        verify_manifest(ref, **{**context, "settings": stated})
    # Without a burst the key is on neither side, and a registration that adds it is refused.
    plain = bmd_runtime_settings(bundle, runner_args())
    (tmp_path / "plain").mkdir()
    ref, context, manifest = registration(tmp_path / "plain", bundle=bundle, settings=plain)
    verify_manifest(ref, **context)
    assert KEY not in manifest["runtime_settings"][bundle]
    stray = json.loads(json.dumps(manifest))
    stray["runtime_settings"][bundle][KEY] = INITIAL_BURST_REFILL_POLICY
    other = write_manifest(tmp_path / "plain" / "stray.json", stray)
    with pytest.raises(ValueError, match="Manifest runtime settings mismatch"):
        verify_manifest(other, **context)


def capture_receipts(monkeypatch, module):
    """Every receipt a runner resolves, recorded as it is returned."""
    seen, original = [], module.bmd_label_receipt

    def receipt(bundle, args, **kwargs):
        """Call the original receipt resolver and record its result."""
        record = original(bundle, args, **kwargs)
        seen.append(record)
        return record

    monkeypatch.setattr(module, "bmd_label_receipt", receipt)
    return seen


def test_a_chembench_burst_receipt_binds_the_refill_policy(tmp_path, monkeypatch):
    """A ChemBench run with a burst writes the refill policy into its config, summary and receipts."""
    stub_chembench_fits(monkeypatch)
    out = tmp_path / "burst"
    cc.run(_args(out, _manifest(tmp_path), cell="100", structure_proposals="free", rounds=1,
                 initial_proposals=2, initial_catalogue="off"))
    config = json.loads((out / "config.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    burst = json.loads((out / "proposals" / "t00_burst.json").read_text())
    for receipt in (config["label_receipt"], summary["label_receipt"]):
        settings = receipt["binding"]["settings"]
        assert settings[KEY] == INITIAL_BURST_REFILL_POLICY == config[KEY] == burst[KEY]
        assert {key: settings[key] for key in BURST_REFILL_KEYS} == {key: config[key] for key in BURST_REFILL_KEYS}
    assert cc.recheck_manifest(config["label_receipt"]) == config["label_receipt"]
    # A run without a burst binds none.
    plain = tmp_path / "plain"
    cc.run(_args(plain, _manifest(tmp_path), cell="100", structure_proposals="free", rounds=1,
                 run_label="development"))
    assert KEY not in json.loads((plain / "config.json").read_text())["label_receipt"]["binding"]["settings"]


@pytest.mark.parametrize("runner", ["core", "boxinggym"])
def test_the_core_and_boxinggym_burst_receipts_bind_the_refill_policy(tmp_path, monkeypatch, request,
                                                                     runner):
    """The CORE and BoxingGym burst receipts bind the refill policy like ChemBench's."""
    from . import boxinggym_core as bg
    from . import core_count as core
    from .test_burst_refill_waves import FORMS as CORE_FORMS
    from .test_burst_refill_waves import boxinggym_burst, core_burst
    from .test_initial_burst import TWELVE

    if runner == "core":
        seen = capture_receipts(monkeypatch, core)
        out, _ = core_burst(tmp_path, monkeypatch, CORE_FORMS[:3], requested=3, proposal_rounds="2")
    else:
        request.getfixturevalue("require_boxinggym_source")
        seen = capture_receipts(monkeypatch, bg)
        out, _ = boxinggym_burst(tmp_path, monkeypatch, [*TWELVE[:3], TWELVE[3]], requested=3)
    receipt, = seen
    config = json.loads((out / "config.json").read_text())
    burst = json.loads((out / "proposals" / "t00_burst.json").read_text())
    assert receipt["binding"]["settings"][KEY] == INITIAL_BURST_REFILL_POLICY == config[KEY] == burst[KEY]


class StopAtReceipt(Exception):
    """Raised in place of the runner's label receipt, so nothing after it can run."""


def test_a_chembench_burst_launch_is_verified_against_the_registered_policy_before_output(
        tmp_path, monkeypatch):
    """A ChemBench burst launch is checked against the registered policy before any output is written."""
    stub_chembench_fits(monkeypatch)
    oracle = _manifest(tmp_path)
    options = {"cell": "100", "structure_proposals": "free", "rounds": 1, "initial_proposals": 2,
               "initial_catalogue": "off", "task": "fake_task__development"}
    captured, original = {}, cc.bmd_label_receipt

    def stop(bundle, args, *, task, metadata):
        """Capture the receipt request and stop the run."""
        captured.update(args=args, task=task)
        raise StopAtReceipt

    monkeypatch.setattr(cc, "bmd_label_receipt", stop)
    with pytest.raises(StopAtReceipt):
        cc.run(_args(tmp_path / "probe", oracle, **options))
    monkeypatch.setattr(cc, "bmd_label_receipt", original)
    assert not (tmp_path / "probe").exists()
    # The runner's own declaration, registered: a development-profile mock row.
    settings = bmd_runtime_settings("chembench", captured["args"])
    assert settings[KEY] == INITIAL_BURST_REFILL_POLICY
    folder = tmp_path / "registration"
    folder.mkdir()
    ref, _, manifest = registration(folder, profile="dev", proposer="mock", bundle="chembench",
                                    task=captured["task"], seed=11, cell="100", label="development",
                                    settings=settings)
    for change in ("removed", "altered"):
        tampered = json.loads(json.dumps(manifest))
        if change == "removed":
            tampered["runtime_settings"]["chembench"].pop(KEY)
        else:
            tampered["runtime_settings"]["chembench"][KEY] = "synthetic_other_rule"
        refused = tmp_path / f"refused_{change}"
        with pytest.raises(ValueError, match="Manifest runtime settings mismatch"):
            cc.run(_args(refused, oracle, manifest_ref=json.dumps(
                write_manifest(folder / f"{change}.json", tampered)), **options))
        assert not refused.exists()
    out = tmp_path / "registered"
    cc.run(_args(out, oracle, manifest_ref=json.dumps(ref), **options))
    receipt = json.loads((out / "config.json").read_text())["label_receipt"]
    assert receipt["binding"]["settings"] == settings and receipt["manifest_sha256"] == ref["sha256"]
    assert receipt["matched_row"]["task"] == "fake_task__development"
