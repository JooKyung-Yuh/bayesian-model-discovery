"""Synthetic frozen pools only; no saved runs, network calls, or sequential Monte Carlo (SMC)."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from . import core_count as core
from . import count_models as models
from . import gaussian_core
from .model_identity import MODEL_ID_SCHEMA
from .test_core_d0_policy import _synthetic_real_fits
from .test_core_parallel_wave import injected_replies
from .test_count_proposer import run_args, stub_fits


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

