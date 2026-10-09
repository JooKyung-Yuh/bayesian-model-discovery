"""Synthetic timeout/recovery receipts; no stored research outputs or providers."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from ..provenance.canonical import is_hex64
from . import core_count as cc
from . import count_inference as ci
from . import neuron_core as nc
from . import vector_inference as vi
from .test_neuron_core_cli import _fake_smc_result, transport_world  # noqa: F401

ADDED_NEURON_CONFIG_KEYS = (
    "forward_backend", "numpy_cpu_features_disabled", "numpy_cpu_features_enabled",
)


def _normalize_added_neuron_config(before, after):
    # Later commits added backend/CPU provenance after these parity bases.
    # Validate actual dispatch provenance before excluding only these added keys.
    original, current = json.loads(before), json.loads(after)
    for key in ADDED_NEURON_CONFIG_KEYS:
        assert key not in original, ("baseline already contains", key)
        assert key in current, ("current config is missing", key)
    backend = current["forward_backend"]
    assert isinstance(backend, str) and backend in vi.FORWARD_BACKENDS, "forward_backend"
    disabled = current["numpy_cpu_features_disabled"]
    assert isinstance(disabled, str), "numpy_cpu_features_disabled must be raw text"
    assert disabled == nc.numpy_cpu_features_disabled(), "numpy_cpu_features_disabled"
    enabled = current["numpy_cpu_features_enabled"]
    assert isinstance(enabled, list), "numpy_cpu_features_enabled must be a list"
    assert all(isinstance(name, str) and name for name in enabled), "numpy_cpu_features_enabled"
    assert len(enabled) == len(set(enabled)), "numpy_cpu_features_enabled contains duplicates"
    assert enabled == nc.numpy_cpu_features_enabled(), "numpy_cpu_features_enabled"
    for key in ADDED_NEURON_CONFIG_KEYS:
        current.pop(key)
    return (json.dumps(current, indent=2, sort_keys=True) + "\n").encode()


def _synthetic_neuron_config():
    return {
        "unchanged": {"value": 7},
        "forward_backend": "native",
        "numpy_cpu_features_disabled": nc.numpy_cpu_features_disabled(),
        "numpy_cpu_features_enabled": nc.numpy_cpu_features_enabled(),
    }


def test_added_neuron_config_comparison_preserves_artifact_bytes(tmp_path):
    """Check that added neuron config comparison preserves artifact bytes."""
    before = (json.dumps({"unchanged": {"value": 7}}, indent=2, sort_keys=True) + "\n").encode()
    after = (json.dumps(_synthetic_neuron_config(), indent=2, sort_keys=True) + "\n").encode()
    original, current = tmp_path / "before.json", tmp_path / "after.json"
    original.write_bytes(before)
    current.write_bytes(after)
    assert _normalize_added_neuron_config(original.read_bytes(), current.read_bytes()) == before
    assert original.read_bytes() == before
    assert current.read_bytes() == after
    assert set(ADDED_NEURON_CONFIG_KEYS) <= json.loads(current.read_bytes()).keys()


@pytest.mark.parametrize("key", ADDED_NEURON_CONFIG_KEYS)
def test_added_neuron_config_refuses_missing_provenance(key):
    """Check that added neuron config refuses missing provenance."""
    current = _synthetic_neuron_config()
    current.pop(key)
    with pytest.raises(AssertionError, match=key):
        _normalize_added_neuron_config(b"{}", json.dumps(current).encode())


@pytest.mark.parametrize("key,value", [
    ("forward_backend", "unregistered"),
    ("forward_backend", None),
    ("numpy_cpu_features_disabled", []),
    ("numpy_cpu_features_disabled", None),
    ("numpy_cpu_features_disabled", "not the raw request"),
    ("numpy_cpu_features_enabled", None),
    ("numpy_cpu_features_enabled", "SSE2"),
    ("numpy_cpu_features_enabled", [""]),
    ("numpy_cpu_features_enabled", [3]),
    ("numpy_cpu_features_enabled", ["SSE2", "SSE2"]),
    ("numpy_cpu_features_enabled", ["not an enabled feature"]),
])
def test_added_neuron_config_refuses_malformed_provenance(key, value):
    """Check that added neuron config refuses malformed provenance."""
    current = _synthetic_neuron_config()
    current[key] = value
    with pytest.raises(AssertionError, match=key):
        _normalize_added_neuron_config(b"{}", json.dumps(current).encode())


@pytest.mark.parametrize("key", ADDED_NEURON_CONFIG_KEYS)
def test_added_neuron_config_refuses_a_base_that_already_has_the_key(key):
    """Check that added neuron config refuses a base that already has the key."""
    current = _synthetic_neuron_config()
    with pytest.raises(AssertionError, match=key):
        _normalize_added_neuron_config(json.dumps({key: current[key]}).encode(), json.dumps(current).encode())


def test_added_neuron_config_keeps_every_other_field():
    """Check that added neuron config keeps every other field."""
    current = _synthetic_neuron_config()
    current["unexpected"] = "must still fail parity"
    normalized = json.loads(_normalize_added_neuron_config(b"{}", json.dumps(current).encode()))
    assert normalized == {"unchanged": {"value": 7}, "unexpected": "must still fail parity"}


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _slow_workers(receipt):
    # Three generations share the isolated fit's group; none catches SIGKILL.
    def record():
        """Record a call and return its result."""
        with Path(receipt).open("a") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "pgid": os.getpgrp()}) + "\n")
    record()
    child = os.fork()
    if child == 0:
        record()
        grandchild = os.fork()
        if grandchild == 0:
            record()
        time.sleep(60)
        os._exit(0)
    time.sleep(60)


def _assert_reaped(receipt):
    processes = _rows(receipt)
    assert len(processes) == 3
    group = processes[0]["pgid"]
    assert {row["pgid"] for row in processes} == {group}
    for row in processes:
        assert not Path(f"/proc/{row['pid']}").exists(), "worker still exists, including a zombie"
    residual = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == group:
            residual.append(path.parent.name)
    assert residual == [], f"residual children in sampler process group: {residual}"


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, False])
def test_timeout_refuses_invalid_values(value):
    """Check that timeout refuses invalid values."""
    with pytest.raises(ValueError, match="finite and positive"):
        ci.validate_fit_timeout(value)


def test_unbounded_helper_preserves_identity_and_in_process_side_effects():
    """Check that unbounded helper preserves identity and in process side effects."""
    calls = []
    result = object()
    def fit_without_timeout_kwarg(value):
        """Stand-in fit function that accepts no timeout keyword."""
        calls.append(value)
        return result
    assert ci.run_with_fit_timeout(fit_without_timeout_kwarg, 7) is result
    assert calls == [7]


def test_timeout_reaps_worker_grandchildren_and_next_fit_succeeds(tmp_path):
    """Check that timeout reaps worker grandchildren and next fit succeeds."""
    before_group = os.getpgrp()
    receipt = tmp_path / "processes.jsonl"
    with pytest.raises(ci.FitTimeout) as error:
        ci.run_with_fit_timeout(_slow_workers, receipt, fit_timeout_seconds=0.3)
    record = error.value.as_record()
    assert record["stage"] == "sampling"
    assert record["cause_code"] == "wall_clock_timeout"
    assert record["status"] == "fit_timeout"
    assert record["elapsed_seconds"] >= record["timeout_seconds"] == 0.3
    _assert_reaped(receipt)
    assert os.getpgrp() == before_group
    assert ci.run_with_fit_timeout(lambda: {"next": "completed"}, fit_timeout_seconds=2) == {"next": "completed"}


@pytest.mark.parametrize("error", [ci.CountInferenceFailure("synthetic", "sampling", "failed"),
                                     ValueError("shape mismatch"), OSError(28, "disk full")])
def test_isolated_fit_keeps_exception_identity_and_traceback(error):
    """Check that isolated fit keeps exception identity and traceback."""
    def broken():
        """Stand-in that always fails, for exercising error handling."""
        raise error
    with pytest.raises(type(error)) as caught:
        ci.run_with_fit_timeout(broken, fit_timeout_seconds=2)
    assert caught.value.args == error.args
    assert caught.value.__dict__ == error.__dict__
    assert "broken" in str(caught.value.__cause__)
    if isinstance(error, OSError):
        assert caught.value.errno == 28


def _count_result(hypothesis):
    particles = {name: np.array([-0.1, 0.1]) for name, _ in hypothesis.parameters}
    particles.update({name: np.array([0.1, 0.2]) for name in hypothesis.noise_parameters})
    return ci.Fit(hypothesis, particles, 0.0, [0.0, 0.0], {}, 0.0)


def _run_core(monkeypatch, out, *extra):
    monkeypatch.setattr(sys, "argv", ["core_count", "--output", str(out), "--profile", "dev",
                                     "--rounds", "2", "--cell", "110", "--proposer", "mock", *extra])
    cc.main()


@pytest.mark.parametrize("scope", ["active", "candidate"])
def test_core_timeout_retires_forever_continues_and_rejects_reproposal(tmp_path, monkeypatch, scope):
    """Check that core timeout retires forever continues and rejects reproposal."""
    from .count_models import Hypothesis
    victim = (cc.initial_hypotheses("binomial")[0] if scope == "active"
              else Hypothesis("b0+b1*square(u)", "binomial", (("b0", "intercept"), ("b1", "signed"))))
    receipt = tmp_path / "core-workers.jsonl"
    attempts = tmp_path / "core-attempts.jsonl"
    def injected(h, u, y, *args, **kwargs):
        """Stand-in injected into the engine under test."""
        with attempts.open("a") as stream:
            stream.write(json.dumps({"model_id": h.model_id, "n": len(y)}) + "\n")
        if h.model_id == victim.model_id:
            _slow_workers(receipt)
        return _count_result(h)
    monkeypatch.setattr(cc, "fit", injected)
    monkeypatch.setattr(cc, "mock_proposal_p1", lambda *args: [victim])
    out = tmp_path / "core-timeout"
    _run_core(monkeypatch, out, "--fit-timeout", "0.3", "--archive-reevaluation-rounds", "2")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "completed"
    failures = [row for row in summary["candidate_failures"] if row["status"] == "fit_timeout"]
    assert len(failures) == 1 and failures[0]["model_id"] == victim.model_id
    assert failures[0]["timeout_seconds"] == 0.3
    assert len([row for row in _rows(attempts) if row["model_id"] == victim.model_id]) == 1
    assert all(victim.model_id not in row["active_ids"] for row in _rows(out / "scores.jsonl"))
    events = [event for path in sorted((out / "proposals").glob("*.json"))
              for event in json.loads(path.read_text())["events"]]
    assert any(row["status"] == "duplicate_not_refitted" for row in events)
    assert any(row["round"] == 2 and row["status"] == "completed" for row in _rows(out / "fit_timings.jsonl"))
    assert json.loads((out / "config.json").read_text())["fit_timeout_seconds"] == 0.3
    _assert_reaped(receipt)


@pytest.mark.parametrize("scope", ["active", "candidate"])
def test_neuron_timeout_retires_forever_continues_and_rejects_reproposal(
        tmp_path, monkeypatch, transport_world, scope):  # noqa: F811 - Pytest injects the imported fixture by its parameter name.
    """Check that neuron timeout retires forever continues and rejects reproposal."""
    victim = nc.initial_hypotheses()[0] if scope == "active" else nc.mock_proposals(1)[0]
    receipt = tmp_path / "neuron-workers.jsonl"
    attempts = tmp_path / "neuron-attempts.jsonl"
    def injected(h, *args, **kwargs):
        """Stand-in injected into the engine under test."""
        with attempts.open("a") as stream:
            stream.write(json.dumps({"model_id": h.model_id}) + "\n")
        if h.model_id == victim.model_id:
            _slow_workers(receipt)
        return _fake_smc_result(h)
    monkeypatch.setattr(nc, "fit", injected)
    monkeypatch.setattr(nc, "mock_proposals", lambda *args: [victim])
    out = tmp_path / "neuron-timeout"
    args = nc.build_parser().parse_args(["--output", str(out), "--rounds", "2", "--cell", "110",
                                       "--fit-timeout", "0.3"])
    nc.run(args)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "completed_deterministic_development"
    assert summary["active_count"] == (3 if scope == "active" else 4)
    assert summary["candidate_failures"][0]["status"] == "fit_timeout"
    assert summary["candidate_failures"][0]["timeout_seconds"] == 0.3
    assert len([row for row in _rows(attempts) if row["model_id"] == victim.model_id]) == 1
    assert all(victim.model_id not in row["active_ids"] for row in _rows(out / "scores.jsonl"))
    events = [event for row in _rows(out / "proposals.jsonl") for event in row["events"]]
    assert [row["status"] for row in events] == (["duplicate", "duplicate"] if scope == "active"
                                                 else ["fit_timeout", "duplicate"])
    assert any(row["round"] == 2 and row["status"] == "completed" for row in _rows(out / "fit_timings.jsonl"))
    _assert_reaped(receipt)


def test_default_core_has_no_timeout_keys_or_sidecar(tmp_path, monkeypatch):
    """Check that default core has no timeout keys or sidecar."""
    monkeypatch.setattr(cc, "fit", lambda h, *args, **kwargs: _count_result(h))
    out = tmp_path / "core-default"
    _run_core(monkeypatch, out)
    assert "fit_timeout_seconds" not in json.loads((out / "config.json").read_text())
    assert not (out / "fit_timings.jsonl").exists()


def test_default_neuron_has_no_timeout_keys_or_sidecar(tmp_path, transport_world):  # noqa: F811 - Pytest injects the imported fixture by its parameter name.
    """Check that default neuron has no timeout keys or sidecar."""
    out = tmp_path / "neuron-default"
    args = nc.build_parser().parse_args(["--output", str(out), "--cell", "000"])
    assert not hasattr(args, "fit_timeout_seconds")
    nc.run(args)
    assert "fit_timeout_seconds" not in json.loads((out / "config.json").read_text())
    assert not (out / "fit_timings.jsonl").exists()


@pytest.mark.parametrize("kind", ["count", "vector"])
def test_native_fit_timeout_then_real_smc_recovers(kind, monkeypatch, tmp_path):
    """Check that native fit timeout then real smc recovers."""
    import pymc as pm
    sample_smc = pm.sample_smc
    receipt = tmp_path / f"{kind}-workers.jsonl"
    monkeypatch.setattr(pm, "sample_smc", lambda **kwargs: _slow_workers(receipt))
    def forward(h, theta, design):
        """Stand-in forward model used by the test."""
        return np.array([theta[0] * design])
    if kind == "count":
        hypothesis = cc.initial_hypotheses("binomial")[0]
        arguments = (hypothesis, [0.2, 0.8], [20, 30], 821, ci.SMCSettings(16, 2, 2))
        fit = ci.fit
        extra = {}
    else:
        from types import SimpleNamespace
        arguments = (SimpleNamespace(model_id="synthetic-vector"), [1.0, 2.0], [[1.0], [2.1]],
                     forward, 821, ci.SMCSettings(16, 2, 2))
        fit = vi.fit
        extra = {"scale": 1.0}
    import pytensor
    # Real SMC with two real worker processes. Python execution keeps this
    # lifecycle test independent of C compiler cost/locks in other test jobs.
    with pytensor.config.change_flags(cxx=""):
        with pytest.raises(ci.FitTimeout):
            fit(*arguments, **extra, fit_timeout_seconds=2)
        _assert_reaped(receipt)
        monkeypatch.setattr(pm, "sample_smc", sample_smc)
        result = fit(*arguments, **extra, fit_timeout_seconds=60)
    assert np.isfinite(result.log_evidence)
    assert len(result.per_chain_log_evidence) == 2


DEFAULT_BASELINE_HARNESS = r'''
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location(
    "bayes_model_discovery.experimental._timeout_parity_fixture", sys.argv[1])
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2001, 1, 1, tzinfo=tz)
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(fixture.cc.time, "perf_counter", lambda: 0.0)
    patch.setattr(fixture.cc.resource, "getrusage", lambda *args: SimpleNamespace(ru_utime=0.0, ru_stime=0.0))
    patch.setattr(fixture.cc, "datetime", FrozenDatetime)
    patch.setattr(fixture.nc, "datetime", FrozenDatetime)
    out = Path(sys.argv[2])
    if sys.argv[3] == "core":
        patch.setattr(fixture.cc, "fit", lambda h, *args, **kwargs: fixture._count_result(h))
        fixture._run_core(patch, out)
    else:
        fixture.transport_world.__wrapped__(patch)
        args = fixture.nc.build_parser().parse_args([
            "--output", str(out), "--rounds", "2", "--cell", "110"])
        fixture.nc.run(args)
'''


@pytest.mark.parametrize("runner", ["core", "neuron"])
def test_opt_in_default_bytes_match_verified_base(tmp_path, runner):
    """Check that opt in default bytes match verified base."""
    if runner == "core":
        pytest.skip("parity with the baseline recorded before the model-identity change crosses the ID/RNG boundary; a successor parity test is still owed")
    if os.environ.get("BMD_BASELINE_COMPARE") != "1":
        pytest.skip("set BMD_BASELINE_COMPARE=1 and BMD_BASELINE_DIR for the read-only base comparison")
    configured = os.environ.get("BMD_BASELINE_DIR")
    if not configured:
        pytest.skip("BMD_BASELINE_DIR must name the read-only declared-base checkout")
    import hashlib
    import subprocess

    baseline = Path(configured).resolve()
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=baseline,
                              check=True, capture_output=True, text=True).stdout.strip()
    assert revision == "c6fe27c76515065a943df33f122529eb181aac04"
    output = tmp_path / "same_output"
    saved = []
    configs = []
    for label, source in (("base", baseline), ("current", root)):
        environment = os.environ.copy()
        environment.update(PYTHONPATH=f"{source}/src:{source}:" + os.environ.get("PYTHONPATH", ""),
                           PYTHONDONTWRITEBYTECODE="1", OPENAI_API_KEY="", ANTHROPIC_API_KEY="",
                           OPENAI_COMPAT_API_KEY="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1")
        completed = subprocess.run(
            [sys.executable, "-B", "-c", DEFAULT_BASELINE_HARNESS, str(Path(__file__).resolve()),
             str(output), runner], cwd=source, env=environment, capture_output=True, text=True, timeout=90, check=False)
        (tmp_path / f"{label}.log").write_text(completed.stdout + completed.stderr)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        config = json.loads((output / "config.json").read_text())
        assert "fit_timeout_seconds" not in config
        assert not (output / "fit_timings.jsonl").exists()
        key = "source_files_sha256" if runner == "core" else "experimental_source_sha256"
        for name, digest in config[key].items():
            assert digest == hashlib.sha256((source / "src/bayes_model_discovery/experimental" / name).read_bytes()).hexdigest()
        configs.append(config)
        destination = tmp_path / label
        output.rename(destination)
        saved.append(destination)
    left, right = saved
    observational_sidecars = {"fit_timings.jsonl", "acquisition_timings.jsonl",
                              "prediction_pool_summary.jsonl"}
    names = sorted(str(path.relative_to(left)) for path in left.rglob("*")
                   if path.is_file() and path.name not in observational_sidecars)
    assert names == sorted(str(path.relative_to(right)) for path in right.rglob("*")
                           if path.is_file() and path.name not in observational_sidecars)
    key = "source_files_sha256" if runner == "core" else "experimental_source_sha256"
    source_pairs = [(configs[1][key][name], configs[0][key][name]) for name in configs[0][key]]
    receipt = []
    for name in names:
        before, after = (left / name).read_bytes(), (right / name).read_bytes()
        added_fields = []
        if name.endswith(".npz"):
            with np.load(left / name) as first, np.load(right / name) as second:
                assert first.files == second.files, name
                for field in first.files:
                    assert first[field].dtype == second[field].dtype, (name, field)
                    assert first[field].tobytes() == second[field].tobytes(), (name, field)
        else:
            if runner == "neuron" and name == "config.json":
                raw_after = after
                after = _normalize_added_neuron_config(before, after)
                added_fields.extend(ADDED_NEURON_CONFIG_KEYS)
                assert (right / name).read_bytes() == raw_after
                assert set(ADDED_NEURON_CONFIG_KEYS) <= json.loads((right / name).read_bytes()).keys()
            if runner != "neuron" and name.startswith("acquisition/") and name.endswith(".json"):
                # A later change added two candidate-pool provenance keys after the base commit;
                # they must be present and well-formed here, absent there, and are
                # removed only after that check. Every other key still compares byte for byte.
                original, current_record = json.loads(before), json.loads(after)
                for key in ("candidate_set_sha256", "candidate_generator"):
                    assert key not in original, (name, key)
                    assert key in current_record, (name, key)
                assert is_hex64(current_record.pop("candidate_set_sha256")), name
                generator = current_record.pop("candidate_generator")
                assert generator["hash_scope"] == "generated_pool_before_history_filters", name
                after = (json.dumps(current_record, indent=2, sort_keys=True) + "\n").encode()
            for current, previous in source_pairs:
                after = after.replace(current.encode(), previous.encode())
            after = after.replace(str(root).encode(), str(baseline).encode())
            assert after == before, name
        receipt.append({"artifact": name, "identical_after_verified_source_provenance_normalization": True,
                        "explicitly_verified_added_fields": added_fields})
    (tmp_path / "parity_report.json").write_text(json.dumps(receipt, indent=2) + "\n")


@pytest.mark.parametrize("runner", ["core", "neuron"])
def test_timeout_stops_only_when_the_pool_becomes_empty(tmp_path, monkeypatch, transport_world, runner):  # noqa: F811 - Pytest injects the imported fixture by its parameter name.
    """Check that timeout stops only when the pool becomes empty."""
    def slow(*args, **kwargs):
        """Stand-in that runs long enough to trigger the timeout."""
        time.sleep(60)
    out = tmp_path / "empty"
    if runner == "core":
        monkeypatch.setattr(cc, "fit", slow)
        with pytest.raises(cc.PoolEmpty, match="active pool empty"):
            _run_core(monkeypatch, out, "--fit-timeout", "0.05", "--cell", "000")
    else:
        monkeypatch.setattr(nc, "fit", slow)
        args = nc.build_parser().parse_args(["--output", str(out), "--cell", "000", "--fit-timeout", "0.05"])
        with pytest.raises(RuntimeError, match="active pool empty"):
            nc.run(args)
    assert not (out / "summary.json").exists()
    receipt = json.loads((out / "failure.json").read_text())
    failures = receipt["refit_failed_models" if runner == "core" else "candidate_failures"]
    assert len(failures) == 4 and all(row["status"] == "fit_timeout" for row in failures)
    assert [row["status"] for row in _rows(out / "fit_timings.jsonl")] == ["fit_timeout"] * 4


def _cancel_nested_caller(receipt, terminate_signal, whole_group):
    import signal
    caller = os.fork()
    if caller == 0:
        try:
            os.setpgid(0, 0)
            ci.run_with_fit_timeout(_slow_workers, receipt, fit_timeout_seconds=60)
        finally:
            os._exit(0)
    records = []
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if receipt.exists():
                records = _rows(receipt)
                if len(records) == 3:
                    break
            time.sleep(0.01)
        assert len(records) == 3, "nested worker tree did not start"
        (os.killpg if whole_group else os.kill)(caller, terminate_signal)
        os.waitpid(caller, 0)
        deadline = time.monotonic() + 3
        remaining = [row["pid"] for row in records]
        while remaining and time.monotonic() < deadline:
            remaining = [pid for pid in remaining if Path(f"/proc/{pid}").exists()]
            time.sleep(0.01)
        return remaining
    finally:
        # Bounded cleanup even if the implementation under test regresses.
        try:
            os.kill(caller, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(caller, 0)
        except ChildProcessError:
            pass
        if records:
            try:
                os.killpg(records[0]["pgid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("terminate_signal", [15, 9], ids=["sigterm", "sigkill"])
@pytest.mark.parametrize("whole_group", [False, True], ids=["caller", "caller_group"])
def test_killed_caller_still_reaps_sampler_descendants(tmp_path, terminate_signal, whole_group):
    """Check that killed caller still reaps sampler descendants."""
    receipt = tmp_path / "cancelled-workers.jsonl"
    # The outer supervisor reaps the nested supervisor after its caller dies;
    # the pytest process itself never changes its child-reaping policy.
    remaining = ci.run_with_fit_timeout(_cancel_nested_caller, receipt, terminate_signal, whole_group,
                                       fit_timeout_seconds=10)
    assert remaining == []
    _assert_reaped(receipt)
