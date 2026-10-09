"""Synthetic frozen pools only; no saved runs, network calls, or sequential Monte Carlo (SMC)."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from . import core_count as core
from . import count_models as models
from . import gaussian_core
from .model_identity import MODEL_ID_SCHEMA
from .test_core_d0_policy import _points_with_authentic_identity, _synthetic_real_fits
from .test_core_parallel_wave import injected_replies
from .test_count_proposer import run_args, stub_fits

# The parity base is BASE_COMMIT (the last deliberate science change); the opt-in comparison skips when a
# checkout of it is absent. On these opt-in paths every later change alters records only: the keys each one
# adds are named below and removed by name from both runs, and any other difference fails.
BASE_COMMIT = "b43bf8ae5245a1010594ef53cf51d6358c759ee4"
#: Per initial family, the files the stack changed after the base among
#: those config.json's `source_files_sha256` records (the normal family also records the
#: real-outcome modules, of which only positive_inference.py moved).
CHANGED_SOURCES = {
    "binomial": ("belief_packet.py", "core_count.py", "count_models.py", "count_proposer.py"),
    "normal": ("belief_packet.py", "core_count.py", "count_models.py", "count_proposer.py",
               "positive_inference.py"),
}
#: Burst refill, with a burst only: config.json's refill digests and rule, the three the
#: burst pool file states, the eight keys a `waves[]` entry gains, the two refill digests the
#: receipt settings bind, the wave count in summary `proposal_accounting`, and the wave-0 packet
#: file, which must be a byte copy of packets/t00_burst.json.
BRF_CONFIG_KEYS = ("initial_burst_refill_policy", "base_prompt_sha256",
                   "burst_refill_paragraph_sha256", "burst_refill_prompt_sha256")
BRF_POOL_FILE_KEYS = ("initial_burst_refill_policy", "base_prompt_sha256",
                      "burst_refill_paragraph_sha256")
BRF_WAVE_KEYS = ("duplicates", "rejected_prior_probe", "fit_failed", "pool_visible", "paragraph",
                 "packet_digest", "pool_size_shown", "pool_digest_shown")
BRF_BURST_SETTINGS_KEYS = ("burst_refill_prompt_sha256", "burst_refill_paragraph_sha256")
BRF_ACCOUNTING_KEYS = ("burst_wave_count",)
BRF_WAVE_PACKET = "t00_burst_w0.json"
#: Burst receipt settings: the refill rule a burst's receipt settings bind.
PCX2B_BURST_SETTINGS_KEYS = ("initial_burst_refill_policy",)
#: Packet constraints: the three engine limits every packet's `constraints` now states.
PCX1_PACKET_CONSTRAINT_KEYS = ("max_distinct_mean_coefficients", "reserved_names",
                               "positive_mean_required_families")
#: Same unit: the digest pair a proposal's transport keeps of the packet it sent, moved by the keys
#: above; each run's pair is checked against that run's own packet file before it is removed.
PCX1_PACKET_DIGEST_KEYS = ("packet_sha256", "packet_bytes")
#: Failure class: `failure_class` on each candidate event, `parameters` on each archive.json
#: entry, and the run's `failure_class_counts` in summary `proposal_accounting`.
PCX2_EVENT_KEYS = ("failure_class",)
PCX2_ARCHIVE_KEYS = ("parameters",)
PCX2_ACCOUNTING_KEYS = ("failure_class_counts",)
#: The id sets each proposals record admits, compared strictly.
ADMITTED_ID_KEYS = ("admitted_ids", "after_ids")


def candidates(family="binomial"):
    """Return two synthetic candidate dicts (expression, family, parameter roles) for the given family."""
    return [
        {"expression": "b0+b1*u+b2*square(z)", "family": family,
         "parameters": {"b0": "intercept", "b1": "positive", "b2": "shape"}},
        {"expression": "b0+b1*z", "family": family,
         "parameters": {"b0": "intercept", "b1": "signed"}},
    ]


def event(item):
    """Turn a candidate dict into a fitted burst event carrying its computed model_id."""
    hypothesis = models.Hypothesis(item["expression"], item["family"], tuple(item["parameters"].items()))
    return {**item, "model_id": hypothesis.model_id, "status": "fitted"}


def burst(family="binomial"):
    """Build a synthetic burst pool payload with two selected events and one rejected, unselected event."""
    selected = [event(item) for item in candidates(family)]
    rejected = {"expression": "synthetic_unselected_invalid_expression", "family": family,
                "parameters": {}, "model_id": "synthetic_unselected_rejection", "status": "rejected"}
    # A burst pool file states the refill rule it was written under.
    return {"model_id_schema": MODEL_ID_SCHEMA, "initial_catalogue": "off",
            "initial_burst_refill_policy": models.INITIAL_BURST_REFILL_POLICY,
            "after_ids": [entry["model_id"] for entry in selected],
            "events": [selected[1], rejected, selected[0]],
            "raw_response": [{"expression": "synthetic_transcript_is_not_the_pool"}]}


def write_pool(tmp_path, value):
    """Write a synthetic pool payload as JSON under tmp_path and return the file path."""
    path = tmp_path / "synthetic_pool.json"
    path.write_text(json.dumps(value) + "\n")
    return path


def read_pool(path, **options):
    """Read a pool file with the initial-pool reader, defaulting to cell 100 and the binomial family."""
    return models.read_initial_pool_file(path, **{"cell": "100", "initial_family": "binomial", **options})


@pytest.mark.parametrize("family", ("binomial", "normal"))
@pytest.mark.parametrize("shape", ("burst", "candidates"))
def test_reader_binds_selected_ids_and_preserves_roles(tmp_path, family, shape):
    """The reader binds the selected model ids and keeps parameter roles and priors for burst and array pools."""
    payload = burst(family) if shape == "burst" else candidates(family)
    path = write_pool(tmp_path, payload)
    kwargs = {"initial_family": family, "allowed_families": (family,)}
    pool = read_pool(path, **kwargs)
    assert pool.source == "file_" + shape
    assert pool.metadata() == {"initial_pool_source": pool.source,
        "initial_pool_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "initial_pool_count": 2}
    assert [dict(h.parameters) for h in pool.hypotheses] == [item["parameters"] for item in candidates(family)]
    assert models.coefficient_prior(pool.hypotheses[0], "b1", "positive") == models.PRIOR_RULES["positive"]
    assert models.coefficient_prior(pool.hypotheses[0], "b2", "shape") == models.PRIOR_RULES["shape"]
    if shape == "burst":
        assert [h.model_id for h in pool.hypotheses] == payload["after_ids"]
    if family == "normal":
        assert [gaussian_core.as_real(h).model_id for h in pool.hypotheses] == [h.model_id for h in pool.hypotheses]


@pytest.mark.parametrize("defect,match", [
    ("catalogue_on", "catalogue models"), ("catalogue_missing", "catalogue models"),
    ("missing_event", "found 0"), ("two_fitted_events", "exactly one fitted event"),
    ("fitted_plus_fit_failed", "non-duplicate events"), ("fitted_plus_rejected", "non-duplicate events"),
    ("fitted_plus_unstatused", "non-duplicate events"),
    ("changed_id", "model_id mismatch"), ("missing_ids", "after_ids"),
    ("bad_events", "candidate events"), ("empty", "no hypotheses"),
])
def test_burst_refuses_ambiguous_or_unbound_state(tmp_path, defect, match):
    """Each defective burst payload is refused with its expected error message."""
    payload = burst()
    if defect == "catalogue_on":
        payload["initial_catalogue"] = "on"
    elif defect == "catalogue_missing":
        payload.pop("initial_catalogue")
    elif defect == "missing_event":
        payload["events"].pop()
    elif defect == "two_fitted_events":
        payload["events"].append({**payload["events"][-1], "status": "fitted"})
    elif defect == "fitted_plus_fit_failed":
        payload["events"].append({**payload["events"][-1], "status": "fit_failed"})
    elif defect == "fitted_plus_rejected":
        payload["events"].append({**payload["events"][-1], "status": "rejected_prior_probe"})
    elif defect == "fitted_plus_unstatused":
        payload["events"].append({key: value for key, value in payload["events"][-1].items() if key != "status"})
    elif defect == "changed_id":
        payload["events"][-1]["expression"] = "b0+b1*z+b2*square(u)"
    elif defect == "missing_ids":
        payload.pop("after_ids")
    elif defect == "bad_events":
        payload["events"] = [None]
    else:
        payload["after_ids"] = []
    with pytest.raises(ValueError, match=match):
        read_pool(write_pool(tmp_path, payload))


@pytest.mark.parametrize("payload,match", [
    (None, "burst object or a candidate array"), ([], "no hypotheses"),
    ([None], "requires expression/family/parameters"),
    ([{"expression": "b0", "family": "binomial", "parameters": []}], "requires expression"),
    ([{"expression": "b0+unknown", "family": "binomial", "parameters": {"b0": "intercept"}}], "unknown"),
    ([{"expression": "b0", "family": "binomial", "parameters": {"b0": "invented"}}], "role"),
    ([{"expression": "b0", "family": "normal", "parameters": {"b0": "intercept"}}], "unsupported"),
    ([{"expression": "b0", "family": "beta_binomial", "parameters": {"b0": "intercept"}}], "F0 requires"),
])
def test_plain_array_reads_every_entry_and_refuses_invalid_candidates(tmp_path, payload, match):
    """A plain array pool reads every entry, and invalid candidates are refused."""
    with pytest.raises(ValueError, match=match):
        read_pool(write_pool(tmp_path, payload))


def test_reader_applies_the_existing_s0_cell_check(tmp_path):
    """The reader applies the existing cell check for the given cell."""
    with pytest.raises(ValueError, match="S0 requires"):
        read_pool(write_pool(tmp_path, candidates()), cell="000")


def test_invalid_json_is_refused(tmp_path):
    """A pool file that is not valid JSON is refused."""
    path = tmp_path / "synthetic_pool.json"
    path.write_text("[")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_pool(path)


@pytest.mark.parametrize("family", ("binomial", "normal"))
@pytest.mark.parametrize("shape", ("candidates", "burst"))
def test_core_injects_exact_file_pool_and_records_binding(tmp_path, monkeypatch, family, shape):
    """The core run uses the file pool exactly and records its binding in the run files."""
    if family == "normal":
        _synthetic_real_fits(monkeypatch)
    else:
        stub_fits(monkeypatch)
    payload = burst(family) if shape == "burst" else candidates(family)
    path = write_pool(tmp_path, payload)
    out = tmp_path / "run"
    core.run(run_args(out, rounds=0, cell="100", initial_catalogue="off", initial_pool_file=str(path),
                      structure="gauss_den" if family == "normal" else "out",
                      truth_family=family, initial_family=family))
    expected = read_pool(path, initial_family=family, allowed_families=(family,))
    first = json.loads((out / "scores.jsonl").read_text().splitlines()[0])
    assert first["active_ids"] == [h.model_id for h in expected.hypotheses]
    config = json.loads((out / "config.json").read_text())
    for record in (config, config["label_receipt"]["binding"]["settings"]):
        assert {key: record[key] for key in expected.metadata()} == expected.metadata()
    assert config["initial_pool"] == 2
    for hypothesis in expected.hypotheses:
        fit = json.loads((out / "fits" / f"t00_{hypothesis.model_id}.json").read_text())
        assert fit["parameters"] == dict(hypothesis.parameters)
    assert not (out / "proposals" / "t00_burst.json").exists()
    assert json.loads((out / "summary.json").read_text())["proposal_accounting"]["llm_calls"] == 0


def test_arm_prompts_share_identical_round_zero_state(tmp_path, monkeypatch):
    """All arms see an identical round-zero state in their prompts."""
    stub_fits(monkeypatch)
    versions = ("v2.0", "v2.6.7")
    schedules = (("b0+b1*tanh(2*z)", "b0+b1*u", "b0+b1*square(z)", "b0+b1*z"),
                 ("b0+b1*z+b2*square(z)", "b0+b1*u", "b0+b1*tanh(3*z)", "b0+b1*z+b2*tanh(2*z)"))
    generated = [tmp_path / "generated_a", tmp_path / "generated_b"]
    controls = []
    for out, version, forms in zip(generated, versions, schedules, strict=True):
        with monkeypatch.context() as patch:
            state = injected_replies(patch, forms)
            core.run(run_args(out, rounds=1, proposal_rounds="2", cell="100", initial_catalogue="off",
                initial_proposals=4, burst_concurrency=1, proposer="anthropic", allow_paid_api=True,
                llm_model="synthetic-model", prompt_version=version))
            assert state["calls"] == 4 and state["queue"] == []
        controls.append(json.loads((out / "scores.jsonl").read_text().splitlines()[0]))
    assert set(controls[0]["active_ids"]) != set(controls[1]["active_ids"])
    assert controls[0]["bma"]["nll"] != controls[1]["bma"]["nll"]
    # Freeze the writer's actual synthetic output bytes, without deriving an array.
    path = generated[0] / "proposals/t00_burst.json"
    frozen = path.read_bytes()
    outputs = [tmp_path / "arm_a", tmp_path / "arm_b"]
    for out, version in zip(outputs, versions, strict=True):
        core.run(run_args(out, rounds=0, cell="100", initial_catalogue="off",
                          initial_pool_file=str(path), prompt_version=version))
        score = json.loads((out / "scores.jsonl").read_text().splitlines()[0])
        assert set(score["active_ids"]) == set(controls[0]["active_ids"])
        assert score["bma"]["nll"] == controls[0]["bma"]["nll"]
    assert path.read_bytes() == frozen
    for name in ("observations.jsonl", "scores.jsonl", "archive.json"):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
    for left in (outputs[0] / "fits").glob("*.json"):
        assert json.loads(left.read_text()) == json.loads((outputs[1] / "fits" / left.name).read_text())
    for left in (outputs[0] / "predictions").glob("*.npz"):
        with np.load(left) as a, np.load(outputs[1] / "predictions" / left.name) as b:
            assert a.files == b.files
            for name in a.files:
                np.testing.assert_array_equal(a[name], b[name])
    configs = [json.loads((out / "config.json").read_text()) for out in outputs]
    assert configs[0]["prompt_sha256"] != configs[1]["prompt_sha256"]
    for config in configs:
        assert config["initial_catalogue"] == "off"
        assert config["initial_pool_source"] == "file_burst"
        assert config["initial_pool_sha256"] == hashlib.sha256(frozen).hexdigest()
        assert config["initial_pool_count"] == 4


@pytest.mark.parametrize("options,match", [
    ({}, "requires --initial-catalogue off"),
    ({"initial_catalogue": "on"}, "requires --initial-catalogue off"),
    ({"initial_catalogue": "off", "initial_proposals": 2}, "no --initial-proposals"),
    ({"initial_catalogue": "off"}, "unknown"),
])
def test_core_refuses_before_any_output_or_proposer(tmp_path, monkeypatch, options, match):
    """The core run refuses a bad pool file before writing any output or calling the proposer."""
    path = write_pool(tmp_path, [{"expression": "b0+unknown", "family": "binomial",
                                  "parameters": {"b0": "intercept"}}])
    calls = []
    monkeypatch.setattr(core, "bmd_label_receipt", lambda *a, **k: calls.append("receipt"))
    out = tmp_path / "refused"
    with pytest.raises(ValueError, match=match):
        core.run(run_args(out, rounds=0, cell="100", initial_pool_file=str(path), **options))
    assert not out.exists() and calls == []


def test_core_cli_accepts_file_flag_and_refuses_malformed_file_without_output(tmp_path):
    # The file states the refill rule, so the catalogue refusal is reached.
    """The core CLI accepts the pool-file flag and refuses a malformed file without writing output."""
    path = write_pool(tmp_path, {"model_id_schema": MODEL_ID_SCHEMA, "initial_catalogue": "on",
                                 "initial_burst_refill_policy": models.INITIAL_BURST_REFILL_POLICY})
    out = tmp_path / "refused"
    proc = subprocess.run([sys.executable, "-B", "-m", core.__name__, "--output", str(out),
        "--cell", "100", "--initial-pool-file", str(path), "--initial-catalogue", "off"],
        capture_output=True, text=True, check=False, timeout=60)
    assert proc.returncode != 0 and not out.exists()
    assert "catalogue models" in proc.stderr


@pytest.mark.parametrize("catalogue", ("on", "off"))
def test_core_cli_refuses_empty_file_path_without_catalogue_fallback(tmp_path, catalogue):
    """The core CLI refuses an empty pool-file path instead of falling back to the catalogue."""
    out = tmp_path / "refused"
    proc = subprocess.run([sys.executable, "-B", "-m", core.__name__, "--output", str(out),
        "--initial-pool-file", "", "--initial-catalogue", catalogue],
        capture_output=True, text=True, check=False, timeout=60)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "nonempty path" in proc.stderr and not out.exists()


@pytest.mark.parametrize("schema", ("missing", None, "ast_roles_family_ordered_legacy", "unknown"))
def test_legacy_schema_is_refused_by_reader_and_real_cli_before_output(tmp_path, schema):
    """A pool file with the legacy model-id schema is refused by the reader and the real CLI before any output."""
    payload = burst()
    if schema == "missing":
        payload.pop("model_id_schema")
    else:
        payload["model_id_schema"] = schema
    path = write_pool(tmp_path, payload)
    with pytest.raises(ValueError, match="model_id_schema mismatch"):
        read_pool(path)
    out = tmp_path / "refused"
    proc = subprocess.run([sys.executable, "-B", "-m", core.__name__, "--output", str(out),
        "--cell", "100", "--initial-pool-file", str(path), "--initial-catalogue", "off"],
        capture_output=True, text=True, check=False, timeout=60)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "model_id_schema mismatch" in proc.stderr
    assert not out.exists() and not list(tmp_path.rglob("cost_state*.json"))


def test_synthetic_transcript_markers_never_enter_prompt_sources(tmp_path, monkeypatch):
    """Transcript markers in a synthetic pool file never reach the prompt sources."""
    root = Path(core.__file__).parent
    markers = ("synthetic_transcript_is_not_the_pool", "synthetic_unselected_invalid_expression")
    for path in [root / "count_proposer.py", root / "belief_packet.py", *(root / "prompts").glob("*.txt")]:
        assert not any(marker in path.read_text() for marker in markers)
    stub_fits(monkeypatch)
    path = write_pool(tmp_path, burst())
    out = tmp_path / "synthetic_context"
    core.run(run_args(out, rounds=1, cell="100", initial_catalogue="off", initial_pool_file=str(path)))
    packet = (out / "packets/t01.json").read_text()
    assert not any(marker in packet for marker in markers)


@pytest.mark.parametrize("mode", ("catalogue", "generated"))
@pytest.mark.parametrize("family", ("binomial", "normal"))
def test_opt_in_catalogue_and_generation_match_hub_baseline(tmp_path, mode, family):
    """Opt-in parity check: catalogue and generated runs match the frozen baseline checkout apart from the named record keys."""
    if os.environ.get("BMD_BASELINE_COMPARE") != "1" or not os.environ.get("BMD_IPF2_BASELINE_DIR"):
        pytest.skip("set BMD_BASELINE_COMPARE=1 and BMD_IPF2_BASELINE_DIR for frozen hub parity")
    base = Path(os.environ["BMD_IPF2_BASELINE_DIR"])
    current = Path(__file__).resolve().parents[3]
    # The baseline checkout is BASE_COMMIT (the pinned parity base).
    assert subprocess.check_output(["git", "-C", str(base), "rev-parse", "HEAD"], text=True).strip() == BASE_COMMIT
    assert subprocess.check_output(["git", "-C", str(base), "status", "--porcelain", "-uall"], text=True) == ""
    harness = """
import sys
from types import SimpleNamespace
from pytest import MonkeyPatch
from bayes_model_discovery.experimental.test_count_proposer import run_args, stub_fits
from bayes_model_discovery.experimental.test_core_d0_policy import _synthetic_real_fits
with MonkeyPatch.context() as patch:
    core = stub_fits(patch)
    if sys.argv[3] == "normal":
        _synthetic_real_fits(patch)
    patch.setattr(core.time, "perf_counter", lambda: 0.0)
    patch.setattr(core.resource, "getrusage", lambda *args: SimpleNamespace(ru_utime=0.0, ru_stime=0.0))
    options = ({"initial_catalogue": "on"} if sys.argv[2] == "catalogue" else
               {"initial_catalogue": "off", "initial_proposals": 2, "burst_concurrency": 1})
    if sys.argv[3] == "normal":
        options.update(structure="gauss_den", truth_family="normal", initial_family="normal")
    core.run(run_args(sys.argv[1], cell="100", rounds=1, run_label="development", **options))
"""
    outputs, configs = [], []
    for label, source in (("base", base), ("current", current)):
        output = tmp_path / label / "synthetic_run"
        env = dict(os.environ, PYTHONPATH=f"{source}/src:{source}", GIT_OPTIONAL_LOCKS="0")
        proc = subprocess.run([sys.executable, "-B", "-c", harness, str(output), mode, family],
                              env=env, cwd=source, capture_output=True, text=True, check=False, timeout=120)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        outputs.append(output)
        config = json.loads((output / "config.json").read_text())
        for name, digest in config["source_files_sha256"].items():
            assert digest == hashlib.sha256((source / "src/bayes_model_discovery/experimental" / name).read_bytes()).hexdigest()
        receipt = config["label_receipt"]
        assert receipt["policy_source_sha256"] == hashlib.sha256(
            (source / "src/bayes_model_discovery/experimental/run_label_policy.py").read_bytes()).hexdigest()
        configs.append(config)
    for config in configs:  # the keys added by the burst-refill and receipt-settings changes, removed by name
        _drop_unit_record_keys("config.json", config)
    assert configs[0].keys() == configs[1].keys()
    # Exactly this family's listed sources moved.
    assert {name for name in configs[0]["source_files_sha256"]
            if configs[0]["source_files_sha256"][name] != configs[1]["source_files_sha256"][name]} == set(
        CHANGED_SOURCES[family])
    assert _points_with_authentic_identity(outputs[0], configs[0], base) == (
        _points_with_authentic_identity(outputs[1], configs[1], current))
    identity = {"output", "created_utc", "source_files_sha256"}
    for config in configs:
        for key in identity:
            config.pop(key)
        config["label_receipt"].pop("policy_source_sha256")
    assert configs[0] == configs[1]
    # Science bytes stay strict on their own; the
    # archive is weakened from raw bytes to parsed JSON with the failure-class `parameters` removed by name.
    for name in ("observations.jsonl", "scores.jsonl"):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
    for path in sorted(path.relative_to(outputs[0]) for path in (outputs[0] / "acquisition").glob("*.json")):
        assert (outputs[0] / path).read_bytes() == (outputs[1] / path).read_bytes(), path
    archives = [json.loads((output / "archive.json").read_bytes()) for output in outputs]
    for archive in archives:
        _drop_unit_record_keys("archive.json", archive)
    assert archives[0] == archives[1]
    for folder in ("fits", "packets", "proposals", "acquisition", "truncation"):
        paths = [sorted(path.relative_to(output) for path in (output / folder).glob("*.json"))
                 for output in outputs]
        if folder == "packets" and mode == "generated":
            # The burst-refill change writes the burst's wave-0 packet file beside the burst
            # packet, as its byte copy; one wave ran, so there is no other wave file.
            wave_file = Path(folder) / BRF_WAVE_PACKET
            assert (outputs[1] / wave_file).read_bytes() == (outputs[1] / folder / "t00_burst.json").read_bytes()
            paths[1].remove(wave_file)
        assert paths[0] == paths[1]
        for path in paths[0]:
            records = [json.loads((output / path).read_bytes()) for output in outputs]
            if folder == "proposals":  # the admitted id sets, strictly
                admitted = [{key: record[key] for key in ADMITTED_ID_KEYS} for record in records]
                assert admitted[0] == admitted[1], path
            for output, record in zip(outputs, records, strict=True):
                if folder == "fits":
                    record.pop("fit_seconds", None)
                    record.get("diagnostics", {}).pop("sampler_attrs", None)
                if folder == "proposals":
                    for transport in record.get("transports", []):
                        assert transport["transport"].pop("run_id") == str(output)
                if folder in ("packets", "proposals"):  # keys added by later changes, removed by name
                    _drop_unit_record_keys(str(path), record, output)
            records = [json.dumps(record, sort_keys=True) for record in records]
            assert records[0] == records[1], path
    summaries = []
    for output, source in zip(outputs, (base, current), strict=True):
        summary = json.loads((output / "summary.json").read_bytes())
        for key in ("timing_seconds", "wall_seconds", "user_cpu_seconds", "system_cpu_seconds"):
            summary.pop(key)
        for key in ("initial_pool_fit", "rounds_fit"):
            summary[key].pop("seconds")
        for key in ("initial_pool_llm", "rounds_llm"):
            summary[key].pop("wall_seconds")
        assert summary["label_receipt"].pop("policy_source_sha256") == hashlib.sha256(
            (source / "src/bayes_model_discovery/experimental/run_label_policy.py").read_bytes()).hexdigest()
        _drop_unit_record_keys("summary.json", summary)  # keys added by later changes, removed by name
        summaries.append(json.dumps(summary, sort_keys=True))
    assert summaries[0] == summaries[1]
    for left in outputs[0].rglob("*.npz"):
        with np.load(left) as a, np.load(outputs[1] / left.relative_to(outputs[0])) as b:
            assert a.files == b.files
            for name in a.files:
                np.testing.assert_array_equal(a[name], b[name])


def _drop_unit_record_keys(name, record, output=None):
    """The record keys named for each later change, removed by name from one run's
    parsed config, summary, archive, packet or proposals record in place; nothing else."""
    if name in ("config.json", "summary.json"):
        if name == "config.json":
            for key in BRF_CONFIG_KEYS:
                record.pop(key, None)
        else:
            for key in (*BRF_ACCOUNTING_KEYS, *PCX2_ACCOUNTING_KEYS):
                record["proposal_accounting"].pop(key, None)
        settings = record["label_receipt"]["binding"]["settings"]
        for key in (*BRF_BURST_SETTINGS_KEYS, *PCX2B_BURST_SETTINGS_KEYS):
            settings.pop(key, None)
    elif name == "archive.json":
        for entry in record:
            for key in PCX2_ARCHIVE_KEYS:
                entry.pop(key, None)
    elif name.startswith("packets/"):
        for key in PCX1_PACKET_CONSTRAINT_KEYS:
            record["constraints"].pop(key, None)
    else:
        for key in BRF_POOL_FILE_KEYS:
            record.pop(key, None)
        for entry in record.get("waves", []):
            for key in BRF_WAVE_KEYS:
                entry.pop(key, None)
        for event in record["events"]:
            for key in PCX2_EVENT_KEYS:
                event.pop(key, None)
        # A burst's transports all sent wave 0's packet (one wave ran); a round's, its own.
        packet = (output / "packets" / Path(name).name).read_bytes()
        transports = ([item["transport"] for item in record["transports"]] if "transports" in record
                      else [record["transport"]])
        for transport in transports:
            pair = tuple(transport.pop(key) for key in PCX1_PACKET_DIGEST_KEYS)
            assert pair == (hashlib.sha256(packet).hexdigest(), len(packet)), name


def test_burst_duplicate_events_under_a_retained_id_are_not_a_defect(tmp_path):
    """The real writer emits `duplicate_not_refitted` events under the admitted id (registration edition 45:
    all three regenerated pools carried them); the reader binds the id to its fitted event."""
    payload = burst()
    fitted = payload["events"][-1]
    payload["events"].append({**fitted, "status": "duplicate_not_refitted"})
    payload["events"].append({**fitted, "status": "duplicate_not_refitted"})
    pool = read_pool(write_pool(tmp_path, payload))
    assert pool.source == "file_burst"
    assert pool.metadata()["initial_pool_count"] == len(payload["after_ids"])
    assert [h.model_id for h in pool.hypotheses] == payload["after_ids"]

