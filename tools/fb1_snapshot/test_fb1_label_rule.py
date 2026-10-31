"""Synthetic label registrations and native import checks; no live API calls."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SNAPSHOT = Path(__file__).resolve().parent
ROOT = SNAPSHOT.parents[1]
sys.path.insert(0, str(SNAPSHOT))


def native_import(tmp_path, extra=None, prefix="", suffix=""):
    """Import the runner in a child interpreter with a controlled environment; return the finished process."""
    python = os.environ.get("FB1_PYTHON", "")
    dp = os.environ.get("DP_SCIENCEAGENT_DIR", "")
    if not python or not Path(python).is_file() or not (Path(dp) / "scienceagent").is_dir():
        pytest.skip("set FB1_PYTHON and DP_SCIENCEAGENT_DIR for native import checks")
    env = {k: v for k, v in os.environ.items() if not k.startswith("FB1_")}
    env.update(PYTHONPATH=str(SNAPSHOT), PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu",
        OPENAI_API_KEY="", ANTHROPIC_API_KEY="", OPENAI_COMPAT_API_KEY="", SYNTHETIC_KEY="",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        FB1_OUT_DIR=str(tmp_path / "output"))
    env.update(extra or {})
    code = prefix + "\nimport json, run_gravity as r\n" + suffix + "\nprint(json.dumps(r.LABEL_RECEIPT))"
    proc = subprocess.run([python, "-B", "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=120)
    assert not (tmp_path / "output").exists(), "validation must run before output creation"
    return proc


def test_snapshot_policy_is_the_exact_portable_canonical_module():
    """The snapshot label policy is byte-identical to the canonical module."""
    canonical = ROOT / "src/bayes_model_discovery/experimental/run_label_policy.py"
    assert (SNAPSHOT / "run_label_policy.py").read_bytes() == canonical.read_bytes()


@pytest.mark.parametrize("arm", ["", "0", "1000", "10x", " 100", "111\n"])
def test_bad_arm_fails_before_indexing_or_output(tmp_path, arm):
    """A malformed arm code is refused before any indexing or output."""
    proc = native_import(tmp_path, {"FB1_ARM": arm})
    assert proc.returncode != 0 and "FB1_ARM must match [01][01][012]" in proc.stderr


@pytest.mark.parametrize("extra", [
    {}, {"FB1_LLM_PROVIDER": "openai"},
    {"FB1_PROPOSER": "llm", "FB1_ALLOW_PAID_API": "0"},
])
def test_mock_cannot_get_production_from_default_provider(tmp_path, extra):
    """A mock run cannot obtain the production label from the default provider."""
    proc = native_import(tmp_path, {**extra, "FB1_RUN_LABEL": "production"})
    assert proc.returncode != 0 and "production requires" in proc.stderr


def test_omitted_and_explicit_development_are_development(tmp_path):
    """An omitted or explicit development label yields a development receipt."""
    for index, extra in enumerate([{}, {"FB1_RUN_LABEL": "development"}]):
        path = tmp_path / str(index)
        path.mkdir()
        proc = native_import(path, extra)
        assert proc.returncode == 0, proc.stderr
        receipt = json.loads(proc.stdout)
        assert receipt["run_label"] == "development" and receipt["manifest_path"] is None
        assert receipt["binding"]["profile"] == "dev"


@pytest.mark.parametrize("requested_label", [None, "development", "production"])
def test_supplied_invalid_reference_always_refused(tmp_path, requested_label):
    """A supplied but invalid manifest reference is refused for every requested label."""
    extra = {"FB1_MANIFEST_REF": "{}"}
    if requested_label is not None:
        extra["FB1_RUN_LABEL"] = requested_label
    proc = native_import(tmp_path, extra)
    assert proc.returncode != 0 and "manifest_ref must contain exactly" in proc.stderr


def test_unknown_label_is_input_error_not_a_silent_downgrade(tmp_path):
    """An unknown label is an input error, not a silent downgrade."""
    proc = native_import(tmp_path, {"FB1_RUN_LABEL": "prod"})
    assert proc.returncode != 0 and "Unknown run label" in proc.stderr


