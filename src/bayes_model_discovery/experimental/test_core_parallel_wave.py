"""Parallel-wave CORE: synthetic replies, native fits and two-runner invariants."""

import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import core_count as core
from . import count_proposer as cp
from .count_models import initial_hypotheses
from .run_label_policy import CORE_ACTIVE_MAXIMUM, bmd_runtime_settings, resolve_label
from .test_count_proposer import run_args, stub_fits
from .test_initial_burst import REFILL_KEYS, burst_records, burst_wave, refill_settings
from .test_run_label_policy import registration, runner_args, write_manifest

# Synthetic schedules only; no stored run responses are read by these tests.
FORMS = core.P1_MOCK_EXPRESSIONS


def injected_replies(monkeypatch, forms, *, family="binomial"):
    """Patch the proposer client to return scripted candidate forms and track call counts and concurrency."""
    original = cp.AnthropicCountProposer
    lock = threading.Lock()
    state = {"queue": list(forms), "calls": 0, "in_flight": 0, "max_in_flight": 0,
             "packets": [], "structures": []}

    def create(**body):
        """Fake client call that pops the next scripted form and returns a synthetic response."""
        with lock:
            form = state["queue"].pop(0)
            state["calls"] += 1
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        time.sleep(0.01)
        with lock:
            state["in_flight"] -= 1
        candidate = {"expression": form, "family": family,
                     "parameters": dict(core.mock_parameters(form))}
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(candidate))],
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 12, "output_tokens": 3}),
            id="synthetic-response", model="synthetic-model", stop_reason="end_turn")

    def factory(records, **kwargs):
        """Build a proposer wrapped to record each packet and its initial structures."""
        proposer = original(records, client=SimpleNamespace(messages=SimpleNamespace(create=create)))
        propose = proposer.propose_one

        def call(packet_text, **options):
            """Record the packet and options, then delegate to the original proposal method."""
            with lock:
                state["packets"].append((dict(options["context"]), json.loads(packet_text)))
                state["structures"].append(options.get("initial_structures"))
            return propose(packet_text, **options)

        proposer.propose_one = call
        return proposer

    monkeypatch.setattr(cp, "AnthropicCountProposer", factory)
    return state


def cell_args(out, **options):
    """Build runner arguments for a one-round synthetic cell with a paid-API-shaped proposer."""
    return run_args(out, **{"seed": 31, "cell": "100", "rounds": 1,
                           "proposer": "anthropic", "allow_paid_api": True,
                           "llm_model": "synthetic-model", **options})


def packet(out, stem):
    """Load the saved packet JSON with the given file stem from a run directory."""
    return json.loads((out / "packets" / f"{stem}.json").read_text())


@pytest.mark.parametrize("structure,family", [("out", "binomial"), ("gauss_den", "normal")])
def test_native_core_burst_twelve_refills_once_and_hides_only_the_round_zero_catalogue(
        tmp_path, monkeypatch, structure, family):
    """A twelve-proposal burst refills once and hides the catalogue only in the round-zero packet."""
    first = [*FORMS[:9], FORMS[0], FORMS[1], FORMS[2]]
    state = injected_replies(monkeypatch, [*first, *FORMS[12:15], FORMS[9]], family=family)
    out = tmp_path / "cell"
    core.run(cell_args(out, structure=structure, truth_family=family, initial_family=family,
                       initial_proposals=12, active_maximum=16, burst_concurrency=4,
                       initial_catalogue="off"))
    record, entries, config, accounting = burst_records(out)
    assert state["queue"] == [] and state["calls"] == 16 and state["max_in_flight"] <= 4
    # Each wave entry carries the eight standard wave keys; the refill wave's packet
    # shows the nine models wave 0 admitted (the first twelve events are wave 0's).
    first_fitted = [e["model_id"] for e in record["events"][:12] if e["status"] == "fitted"]
    assert len(first_fitted) == 9
    assert record["waves"] == [burst_wave(out, 0, 12, 9, duplicates=3),
                               burst_wave(out, 1, 3, 3, first_fitted)]
    assert record["max_in_flight"] <= 4 and record["concurrency"] == 4
    assert record["before_ids"] == [] and len(record["admitted_ids"]) == 12
    assert record["duplicate_count"] == 3 and record["fit_success_count"] == 12
    assert len([entry for entry in entries if entry.get("source") == "initial_burst"]) == 15
    assert config["active_maximum"] == 16 and config["initial_catalogue"] == "off"
    assert config["burst_prompt_sha256"] == cp.composed_system_prompt(burst=True)[1]["burst_prompt_sha256"]
    assert accounting["burst_calls"] == 15 and accounting["burst_refill_calls"] == 3
    assert accounting["burst_duplicates"] == 3 and accounting["burst_admitted"] == 12
    zero, one = packet(out, "t00_burst"), packet(out, "t01")
    assert zero["pool"] == [] and zero["best_model"] is None and zero["best_model_residuals"] == []
    assert zero["constraints"]["initial_structures"] == []
    assert one["constraints"]["initial_structures"] == [h.expression for h in initial_hypotheses(family)]
    assert all(len(structures) == 4 for structures in state["structures"][:-1])
    expected = {f"round_00__burst_w0_c{i:02d}.request.json" for i in range(12)}
    expected |= {f"round_00__burst_w1_c{i:02d}.request.json" for i in range(3)}
    assert {path.name for path in (out / "llm_transport").glob("round_00*.request.json")} == expected


def test_core_parallel_rounds_deduplicate_catalogue_and_previous_admissions(tmp_path, monkeypatch):
    """Parallel rounds drop candidates duplicating the catalogue or earlier admissions."""
    catalogue = [h.expression for h in initial_hypotheses("binomial")]
    sequence = [FORMS[2], catalogue[1], catalogue[2], FORMS[3], FORMS[4],
                FORMS[6], FORMS[7], FORMS[2], FORMS[8]]
    state = injected_replies(monkeypatch, sequence)
    out = tmp_path / "cell"
    core.run(cell_args(out, rounds=2, parallel_proposals=3, burst_concurrency=3))
    records = [json.loads((out / "proposals" / f"t{t:02d}.json").read_text()) for t in (1, 2)]
    assert state["calls"] == 9 and state["max_in_flight"] <= 3
    assert records[0]["waves"] == [{"wave": 0, "calls": 3, "accepted": 1},
                                   {"wave": 1, "calls": 2, "accepted": 2}]
    assert records[1]["waves"] == [{"wave": 0, "calls": 3, "accepted": 2},
                                   {"wave": 1, "calls": 1, "accepted": 1}]
    assert [r["duplicate_count"] for r in records] == [2, 1]
    assert [len(r["after_ids"]) for r in records] == [7, 10]
    assert all(e["source"] == "parallel_wave" for r in records for e in r["events"])
    assert not (out / "proposals" / "t00_burst.json").exists()
    assert (out / "llm_transport" / "round_02__burst_w1_c00.request.json").is_file()


def test_mock_burst_keeps_catalogue_and_applies_the_declared_cap(tmp_path, monkeypatch):
    """A mock burst keeps the catalogue and applies the declared active-model cap."""
    stub_fits(monkeypatch)
    out = tmp_path / "cell"
    core.run(cell_args(out, proposer="mock", allow_paid_api=False,
                       initial_proposals=4, active_maximum=5))
    record, _, config, accounting = burst_records(out)
    assert len(record["before_ids"]) == 4 and len(record["after_ids"]) == 5
    # The first refill wave falls one short again, so a third wave asks
    # for exactly that one.
    assert [wave["calls"] for wave in record["waves"]] == [4, 2, 1]
    assert record["duplicate_count"] >= 2 and record["fit_success_count"] >= 3
    assert len(packet(out, "t00_burst")["constraints"]["initial_structures"]) == 4
    assert config["active_maximum"] == 5 and config["burst_concurrency"] == 6
    assert accounting["burst_calls"] == 0 and accounting["llm_calls"] == 0


def test_parallel_call_keeps_cores_declared_horizon_past_twenty(tmp_path, monkeypatch):
    """A parallel call still works at a declared horizon beyond round twenty."""
    stub_fits(monkeypatch)
    state = injected_replies(monkeypatch, [FORMS[2]])
    out = tmp_path / "cell"
    core.run(cell_args(out, rounds=21, proposal_rounds="21", parallel_proposals=1))
    assert state["queue"] == [] and state["calls"] == 1
    record = json.loads((out / "proposals" / "t21.json").read_text())
    assert record["waves"] == [{"wave": 0, "calls": 1, "accepted": 1}]


@pytest.mark.parametrize("kind,expected", [("numeric", "fit_or_predict_failed"),
                                          ("guarded", "fit_failed"), ("timeout", "fit_timeout")])
def test_fit_failure_refills_once_without_losing_core_failure_provenance(tmp_path, monkeypatch, kind, expected):
    """A candidate fit failure triggers one refill and keeps its failure status in the record."""
    from .count_inference import CountInferenceFailure, FitTimeout

    stub_fits(monkeypatch)
    original = core.fit

    def fit(hypothesis, *args, **kwargs):
        """Raise a synthetic failure for the chosen candidate; otherwise delegate to the original fit."""
        if hypothesis.expression == FORMS[2]:
            if kind == "timeout":
                raise FitTimeout(2.0, 1.0)
            raise CountInferenceFailure("nonfinite_smc_weights" if kind == "guarded" else "synthetic_numeric",
                                        "sampling", "synthetic candidate fit failure")
        return original(hypothesis, *args, **kwargs)

    monkeypatch.setattr(core, "fit", fit)
    monkeypatch.setattr(core, "loglik_prior_probe", lambda *a, **k: {"passed": True})
    state = injected_replies(monkeypatch, [FORMS[2], FORMS[3]])
    out = tmp_path / "cell"
    core.run(cell_args(out, parallel_proposals=1, **({"fit_guards": "v1"} if kind == "guarded" else {})))
    record = json.loads((out / "proposals" / "t01.json").read_text())
    assert state["calls"] == 2
    assert record["waves"] == [{"wave": 0, "calls": 1, "accepted": 0},
                               {"wave": 1, "calls": 1, "accepted": 1}]
    assert [e["status"] for e in record["events"]] == [expected, "fitted"]
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["refit_failed_models"]) == 1
    assert record["fit_attempt_count"] == 2 and record["fit_success_count"] == 1


@pytest.mark.parametrize("flags", [
    ["--initial-proposals", "25"], ["--initial-proposals", "0"],
    ["--initial-proposals", "13"], ["--initial-catalogue", "off"],
    ["--burst-concurrency", "0"], ["--burst-concurrency", "25"],
    ["--parallel-proposals", "7"], ["--parallel-proposals", "0"],
    ["--parallel-proposals", "3", "--proposal-limit", "2"],
    ["--initial-proposals", "5", "--active-maximum", "4"],
    ["--active-maximum", "0"], ["--active-maximum", "33"],
    ["--initial-catalogue", "off", "--initial-proposals", "2", "--cell", "000"],
    ["--initial-proposals", "2", "--proposal-policy", "P4"],
])
def test_wave_cli_refusals_leave_no_output(tmp_path, flags):
    """Invalid wave command-line flags are refused with exit code 2 and leave no output."""
    out = tmp_path / "refused"
    result = subprocess.run([sys.executable, "-B", "-m", core.__name__, "--output", str(out),
                             "--cell", "100", *flags], capture_output=True, text=True,
                            check=False, timeout=60)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not out.exists()


def test_core_label_binds_only_stated_nondefault_wave_settings(tmp_path):
    """The run label binds wave settings only when they are stated and non-default."""
    plain = bmd_runtime_settings("core", runner_args())
    digest = cp.composed_system_prompt(burst=True)[1]["burst_prompt_sha256"]
    # A burst with initial proposals states its two refill digests and its
    # refill rule too; they are the only other keys it adds, bound as stated.
    bound = bmd_runtime_settings("core", runner_args(
        initial_proposals=12, active_maximum=16, initial_catalogue="off", burst_prompt_sha256=digest,
        **refill_settings()))
    added = {"initial_proposals", "active_maximum", "initial_catalogue", "burst_prompt_sha256",
             *REFILL_KEYS}
    assert set(bound) - set(plain) == added
    assert {key: bound[key] for key in REFILL_KEYS} == refill_settings()  # refill settings are bound as stated
    assert {k: v for k, v in bound.items() if k not in added} == plain
    assert bound["initial_catalogue"] == "off" and bound["active_maximum"] == 16
    assert bmd_runtime_settings("core", runner_args(active_maximum=CORE_ACTIVE_MAXIMUM,
                                                    initial_catalogue="on")) == plain
    ref, context, _ = registration(tmp_path, settings=bound)
    context["settings"] = bound
    assert resolve_label("production", manifest_ref=ref, **context)["run_label"] == "production"


@pytest.mark.parametrize("wrong", [{"active_maximum": 0}, {"active_maximum": "16"},
                                    {"initial_catalogue": "off"}, {"initial_catalogue": "on"}])
def test_core_registration_refuses_invalid_wave_settings(tmp_path, wrong):
    """A registration with invalid wave settings is refused."""
    ref, context, manifest = registration(tmp_path)
    manifest["runtime_settings"]["core"].update(wrong)
    ref = write_manifest(Path(ref["path"]), manifest)
    with pytest.raises(ValueError):
        resolve_label("production", manifest_ref=ref, **context)


@pytest.mark.usefixtures("require_boxinggym_source")
def test_both_runners_share_wave_invariants_including_probe_refusal_and_refill(tmp_path):
    """Both runners satisfy the same wave invariants, including prior-probe refusal and refill."""
    from . import boxinggym_core as bg
    from .test_death_kl import bg_run_args

    outputs, records = [], []
    sequence = ["b0+b1*z", "b0+b1*z", "b0+b1*sqrt(z)",
                "b0+b1*z+b2*square(z)", "b0+b1*exp(z)", "b0+b1*tanh(3*z)", "b0+b1*u"]
    for module in (core, bg):
        boxing = module is bg
        forms = [re.sub(r"\b(z|u)\b", lambda m: {"z": "tz", "u": "t01"}[m[0]], form)
                 for form in sequence] if boxing else sequence
        with pytest.MonkeyPatch.context() as patch:
            state = injected_replies(patch, forms)
            out = tmp_path / ("boxinggym" if boxing else "core")
            options = {"cell": "100", "rounds": 1, "proposer": "anthropic", "allow_paid_api": True,
                       "llm_model": "synthetic-model", "initial_proposals": 4, "active_maximum": 8,
                       "burst_concurrency": 1, "initial_catalogue": "off"}
            module.run(bg_run_args(out, **options) if boxing else cell_args(out, **options))
        record, entries, _, accounting = burst_records(out)
        assert state["queue"] == []
        # Each wave entry carries the eight standard wave keys - wave 0 counts its
        # duplicate and its repaired prior-probe refusal, and the refill wave's packet shows the
        # three models wave 0 admitted (the first four events are wave 0's).
        first_fitted = [e["model_id"] for e in record["events"][:4] if e["status"] == "fitted"]
        assert len(first_fitted) == 3
        assert record["waves"] == [burst_wave(out, 0, 4, 3, duplicates=1, rejected_prior_probe=1),
                                   burst_wave(out, 1, 1, 1, first_fitted)]
        assert record["duplicate_count"] == 1 and record["fit_success_count"] == 4
        assert record["max_in_flight"] <= record["concurrency"] == 1
        assert accounting["prior_probe_rejections"] == 1 and accounting["burst_repairs"] == 1
        assert len([e for e in entries if e["outcome"] == "rejected"]) == 1
        assert {e["source"] for e in record["events"]} == {"initial_burst"}
        first = next(p for c, p in state["packets"] if c.get("burst_wave") == 0)
        refill = next(p for c, p in state["packets"] if c.get("burst_wave") == 1)
        assert first["pool"] == [] and len(refill["archive"]) > len(first["archive"])
        assert len([e for e in refill["archive"] if e["outcome"] == "active"]) == 3
        assert first["constraints"]["initial_structures"] == []
        assert len(packet(out, "t01")["constraints"]["initial_structures"]) == 4
        assert all(len(s) == 4 for s in state["structures"] if s is not None)
        assert record["fit_success_count"] > 0  # input-name metadata still reaches the judge
        outputs.append(out)
        records.append(record)
    assert records[0].keys() == records[1].keys()
    assert {p.name for p in (outputs[0] / "llm_transport").iterdir()} == {
        p.name for p in (outputs[1] / "llm_transport").iterdir()}


@pytest.mark.usefixtures("require_boxinggym_source")
def test_boxinggym_catalogue_on_retains_forms_in_both_packets(tmp_path):
    """With the catalogue on, BoxingGym retains the initial forms in both packets."""
    from . import boxinggym_core as bg
    from .test_death_kl import bg_run_args

    out = tmp_path / "cell"
    bg.run(bg_run_args(out, rounds=1, cell="100", initial_proposals=4, initial_catalogue="on"))
    expected = list(bg.TASKS["death"].initial_structures())
    assert packet(out, "t00_burst")["constraints"]["initial_structures"] == expected
    assert packet(out, "t01")["constraints"]["initial_structures"] == expected
