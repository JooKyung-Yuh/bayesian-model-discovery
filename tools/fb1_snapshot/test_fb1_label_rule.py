"""Synthetic label registrations and native import checks; no live API calls."""
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SNAPSHOT = Path(__file__).resolve().parent
ROOT = SNAPSHOT.parents[1]
sys.path.insert(0, str(SNAPSHOT))
import run_label_policy as policy


def wave_module():
    """Load the research-wave registration driver, skipping the test when it is absent."""
    directory = ROOT / "tools/bai_smoke_2h"
    if not (directory / "make_research_wave.py").is_file():
        pytest.skip("the research-wave registration driver is not part of this release")
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("synthetic_fb_label_wave", directory / "make_research_wave.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registration(tmp_path, provider="anthropic", model="synthetic-model", host="synthetic.invalid"):
    """Build a synthetic registration under tmp_path; return the manifest reference and the manifest."""
    wave = wave_module()
    options = {} if provider != "openai_compat" else dict(
        base_url=f"https://{host}/v1", key_name="SYNTHETIC_KEY", provider_order="supplier-a,supplier-b")
    payloads, manifest = wave.build_payloads(source_root=ROOT,
        rates={bundle: .01 for bundle in wave.BUNDLES}, generated_at="2000-01-01T00:00:00Z",
        generation_commit="0" * 40, generator_sha256="1" * 64,
        run_label="production", artifact_directory=str(tmp_path),
        provider=provider, model=model, **options)
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        (tmp_path / name).write_bytes(data)
    path = tmp_path / "research_wave_manifest.json"
    ref = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return ref, manifest


def paper_env(ref, provider="anthropic", model="synthetic-model"):
    """Return the environment variables of a registered production cell for the given provider and model."""
    env = dict(FB1_ARM="100", FB1_WORLD="gravity", FB1_SEED="11",
        FB1_PROPOSER="llm", FB1_ALLOW_PAID_API="1", FB1_LLM_PROVIDER=provider,
        FB1_LLM_MODEL=model, FB1_RUN_LABEL="production", FB1_MANIFEST_REF=json.dumps(ref),
        FB1_D0="mda1", FB1_CHECKPOINTS="all", FB1_T_EXTRA="16",
        FB1_INTEGRATOR="match_truth", FB1_DESIGN_GATE="dt_halving",
        FB1_DESIGN_GATE_FALLBACK="min_delta_k", FB1_DESIGN_GATE_FALLBACK_K="8")
    if provider == "openai_compat":
        env.update(FB1_LLM_BASE_URL="https://synthetic.invalid/v1", FB1_LLM_KEY_NAME="SYNTHETIC_KEY",
                   FB1_LLM_PROVIDER_ORDER="supplier-a,supplier-b")
    return env


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


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openai_compat"])
def test_matching_registered_llm_is_production_without_making_calls(tmp_path, provider):
    """A registered, matching LLM cell resolves to production without making any call."""
    ref, _ = registration(tmp_path / "registration", provider)
    proc = native_import(tmp_path, paper_env(ref, provider))
    assert proc.returncode == 0, proc.stderr
    receipt = json.loads(proc.stdout)
    assert receipt["run_label"] == "production"
    assert receipt["manifest_sha256"] == ref["sha256"]
    assert receipt["binding"]["settings"]["particles"] == 500
    assert receipt["binding"]["settings"]["chains"] == 2
    assert receipt["matched_row"]["cell"] == "100"


@pytest.mark.parametrize("change", [
    {"FB1_LLM_MODEL": "different-model"}, {"FB1_ARM": "000"},
    {"FB1_T_EXTRA": "1"}, {"FB1_CHECKPOINTS": "default"},
    {"FB1_D0": "random2"}, {"FB1_INTEGRATOR": "legacy"},
    {"FB1_DESIGN_GATE": "off"}, {"FB1_PROPOSAL_POLICY": "P1"},
    {"FB1_STRUCTURE_PROPOSALS": "free"},
    {"FB1_DESIGN_GATE_FALLBACK": "off"}, {"FB1_DESIGN_GATE_FALLBACK_K": "7"},
])
def test_registration_mismatch_fails_before_output(tmp_path, change):
    """Any mismatch with the registered cell fails before output exists."""
    ref, _ = registration(tmp_path / "registration")
    proc = native_import(tmp_path, {**paper_env(ref), **change})
    assert proc.returncode != 0 and "ValueError" in proc.stderr


def test_changed_particle_profile_cannot_inherit_paper(tmp_path):
    """A changed particle count cannot inherit the paper profile."""
    ref, _ = registration(tmp_path / "registration")
    proc = native_import(tmp_path, paper_env(ref), prefix="import fb1_core; fb1_core.N_PARTICLES = 4")
    assert proc.returncode != 0 and "runtime settings mismatch" in proc.stderr


def test_manifest_changed_between_import_and_main_refused_before_output(tmp_path):
    """A manifest modified between import and main is refused before output."""
    ref, _ = registration(tmp_path / "registration")
    suffix = "\nfrom pathlib import Path\np = Path(r.LABEL_RECEIPT['manifest_path'])\np.write_bytes(p.read_bytes() + b' ')\nr.main()"
    proc = native_import(tmp_path, paper_env(ref), suffix=suffix)
    assert proc.returncode != 0 and "whole-file SHA mismatch" in proc.stderr


def synthetic_run(receipt=None):
    # These values are handwritten data, not copied from any experiment.
    """Build a synthetic run record, bound to the given label receipt when one is supplied."""
    from test_fb1_proposer import _minimal_run_json_29fb
    head = dict(nmse_mda=1., nll_per_obs=.2, coverage90=.9, interval_width90=.4,
                pred_mean=[[[1., 1.]]], kl={"value": .3})
    run = _minimal_run_json_29fb(head, copy.deepcopy(head))
    if receipt is None:
        return run
    binding = receipt["binding"]
    settings = binding["settings"]
    final_n = settings["d0_size"] + settings["t_extra"]
    run.update(world=binding["task"], arm=binding["cell"], label=copy.deepcopy(receipt))
    run["seeds"]["run"] = binding["seed"]
    run["budget"].update(D0=settings["d0_size"], extra=settings["t_extra"], total=final_n,
        checkpoints=settings["checkpoint_values"], checkpoints_option=settings["checkpoints"],
        d0_policy=settings["d0_policy"], d0_points=settings["d0_points"])
    run["smc"].update(particles_per_chain=settings["particles"], chains=settings["chains"])
    run["dt_forward"] = settings["dt"]
    run["arm_config"].update(expand=binding["cell"][0] == "1", numerics=dict(
        mode=settings["integrator"], design_gate=settings["design_gate"], noise_inflate=settings["noise_inflate"]))
    for key in ("design_gate_fallback", "fallback_k"):
        if key in settings:
            run["arm_config"]["numerics"][key] = settings[key]
    run["steps"][0]["designs"] = [[1., 1.]] * final_n
    run["checkpoints"] = {str(final_n): run["checkpoints"]["2"]}
    if binding["proposer"] != "mock":
        run["proposer"] = dict(proposer="llm", requested_provider=binding["proposer"],
            requested_model=binding["model"], allow_paid_api=True, paid_api_calls=0,
            transport=binding["proposer"], policy=settings["proposal_policy"],
            proposal_events=settings["proposal_events"], candidates_per_event=settings["candidates_per_event"],
            prompt_version="synthetic", prompt_sha256="0" * 64,
            label=receipt["run_label"], label_receipt=copy.deepcopy(receipt))
        if binding["proposer"] == "openai_compat":
            run["proposer"].update(llm_base_url_host=binding["base_url_host"],
                                    provider_order=binding["provider_order"])
    return run


def production_receipt(tmp_path):
    """Resolve a production label receipt against a synthetic registration."""
    ref, manifest = registration(tmp_path)
    settings = dict(manifest["runtime_settings"]["forcebench"])
    settings.pop("profile")
    return policy.resolve_label("production", "paper", "anthropic", cell="100", manifest_ref=ref,
        model="synthetic-model", bundle="forcebench", task="gravity", seed=11, settings=settings)



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


