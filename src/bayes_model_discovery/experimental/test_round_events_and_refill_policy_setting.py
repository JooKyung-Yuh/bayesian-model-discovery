"""ChemBench records every candidate event after round 0, and the round-0 burst's refill policy
is a registered runtime setting (a project decision of 2026-09-23).

Synthetic replies, the runner's mock path and stubbed fits only: no network, no saved run, no
stored response, and no expression or seed taken from a real run.
"""

import hashlib
import json

import pytest

from . import chembench_core as cc
from . import count_proposer as cp
from .count_inference import FitTimeout
from .count_models import INITIAL_BURST_REFILL_POLICY, model_id_preimage
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
# 2. the refill policy is a registered runtime setting
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
