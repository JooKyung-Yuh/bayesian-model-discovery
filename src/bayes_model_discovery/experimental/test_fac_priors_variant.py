"""The FA-C prompt variant that lets the proposer write numeric coefficient priors, end to end
with `--prompt-variant <variant> --priors llm`. Synthetic replies through an injected Anthropic client; 0 USD.

The failure this guards against (2026-10-04): config said priors=llm, but the FA-C variant told the proposer "never write
numeric priors", so no reply carried a prior and no fit used one."""

import hashlib
import json
from pathlib import Path

import pytest

from . import boxinggym_core as bg
from . import count_inference as ci
from .test_named_inputs import bg_fake_anthropic_client, bg_inject_client, bg_run_args, bg_stub_fits

ROOT = Path(__file__).resolve().parents[3]
VARIANT = ROOT / "tools" / "fb1_launch_c" / "proposer_system_v2_9_FA-C_priors.txt"
FAC = ROOT / "tools" / "fb1_launch_c" / "proposer_system_v2_9_FA-C.txt"
V22 = Path(__file__).resolve().parent / "prompts" / "proposer_system_v2_2.txt"
V20 = Path(__file__).resolve().parent / "prompts" / "proposer_system_v2_0.txt"
ROLES = {"b0": "intercept", "b1": "positive"}
PRIOR = {"b1": {"dist": "LogNormal", "params": [1.0, 0.5]}}


@pytest.fixture(autouse=True)
def restore_process_switches():
    """A run selects the prompt variant and the proposed-prior path for the whole process; put both back."""
    from . import count_models as cm
    from . import count_proposer as cp

    yield
    cp.set_prompt_variant(None)
    cm.set_llm_priors(False)


def reply(priors=None, expression="b0+b1*(dR-iR)"):
    """Build a synthetic proposer JSON reply, optionally carrying priors."""
    body = {"expression": expression, "family": "bernoulli", "parameters": ROLES,
            "predicted_diagnostic": {"input": "dR", "low": 1, "high": 300}, "closest_archived_model": "b0"}
    if priors is not None:
        body["priors"] = priors
    return json.dumps(body)


def system_text(body):
    """Return the system prompt text of a recorded request body."""
    system = body["system"]
    return system if isinstance(system, str) else "".join(block["text"] for block in system)


def run_with(tmp_path, monkeypatch, replies, name, **overrides):
    """Run one stubbed paid cell with the given replies and return its output folder, calls and fitted hypotheses."""
    adapter = bg_stub_fits(monkeypatch)
    fitted = []
    stub = adapter.count_fit

    def recording_fit(hypothesis, *args, **kwargs):
        """Record the hypothesis and delegate to the stubbed fit."""
        fitted.append(hypothesis)
        return stub(hypothesis, *args, **kwargs)

    monkeypatch.setattr(adapter, "count_fit", recording_fit)
    client, calls = bg_fake_anthropic_client(replies)
    bg_inject_client(monkeypatch, client)
    out = tmp_path / name
    adapter.run(bg_run_args(out, task="hyperbolic", cell="111", rounds=1, proposer="anthropic", allow_paid_api=True,
                            run_label="development", **overrides))
    return out, calls, fitted


def test_variant_differs_from_fac_only_in_its_prior_instruction():
    """Check that variant differs from fac only in its prior instruction."""
    fac = FAC.read_text().splitlines(keepends=True)
    variant = VARIANT.read_text().splitlines(keepends=True)
    assert fac[0] == "# prompt_version: v2.9.FA-C\n" and variant[0] == "# prompt_version: v2.9.FA-C-priors\n"
    changed = [i for i, (a, b) in enumerate(zip(fac, variant)) if a != b]
    assert changed == [0, 17, 19]                                  # tag, the prior sentence, the reply keys
    assert "never write numeric priors" in fac[17] and "never write numeric priors" not in variant[17]
    assert "intercept, signed, positive, shape" in variant[17]      # role labels stay the fallback
    assert '"priors"' in variant[19] and '"priors"' not in fac[19]
    block = "".join(V22.read_text().splitlines(keepends=True)[12:19])
    assert "".join(variant[len(fac):]) == "\n" + block               # the v2.2 prompt file's priors block, verbatim


def test_variant_carries_v22_priors_block_verbatim_and_no_numbers_from_runs():
    """Check that variant carries v22 priors block verbatim and no numbers from runs."""
    text = VARIANT.read_text()
    block = "".join(V22.read_text().splitlines(keepends=True)[12:19])
    assert text.endswith("\n" + block)
    assert "never write numeric priors" not in text


@pytest.mark.usefixtures("require_boxinggym_source")
def test_end_to_end_priors_reach_the_fit_and_the_receipt_binds_the_variant(tmp_path, monkeypatch):
    """Check that end to end priors reach the fit and the receipt binds the variant."""
    out, calls, fitted = run_with(tmp_path, monkeypatch, [reply(PRIOR)], "variant",
                                  priors="llm", prompt_variant=str(VARIANT))
    # the system prompt sent is the variant, byte for byte
    assert system_text(calls[0]) == VARIANT.read_text()
    # the packet carries the prior menu block
    packet = json.loads((out / "packets" / "t01.json").read_text())
    assert set(packet["constraints"]["priors"]["allowed_by_role"]["positive"]) == {"LogNormal", "Gamma", "Beta"}
    # the reply is parsed with its priors
    raw = json.loads((out / "proposals" / "t01.json").read_text())["raw_response"]
    assert raw[0]["prior_source"] == "llm" and raw[0]["proposed_priors"] == {"b1": ["LogNormal", 1.0, 0.5]}
    # coefficient_priors reach the fit, and the model the fit builds uses them
    candidate = next(h for h in fitted if h.model_id == raw[0]["model_id"])
    assert candidate.coefficient_priors == (("b1", ("LogNormal", 1.0, 0.5)),)
    seen = {}
    real_prior = ci.coefficient_prior

    def spy(hypothesis, name, role):
        """Spy wrapper that records arguments and delegates to the original function."""
        seen[name] = real_prior(hypothesis, name, role)
        return seen[name]

    monkeypatch.setattr(ci, "coefficient_prior", spy)
    import numpy as np
    x = np.asarray([[1.0, 100.0, 10.0], [50.0, 200.0, 30.0]])
    model = ci.build_model(candidate, x, np.asarray([1.0, 0.0]), 1, bg.TASKS["hyperbolic"].interface())
    assert seen["b1"] == ("LogNormal", 1.0, 0.5) and seen["b0"][0] == "Normal"
    assert type(model["b1"].owner.op).__name__ == "LogNormalRV"
    # the fit record says so
    fit = json.loads((out / "fits" / f"t01_{candidate.model_id}.json").read_text())
    assert fit["prior_source"] == "llm" and fit["proposed_priors"] == {"b1": ["LogNormal", 1.0, 0.5]}
    # the label receipt binds the variant
    config = json.loads((out / "config.json").read_text())
    settings = config["label_receipt"]["binding"]["settings"]
    assert settings["prompt_version"] == "v2.9.FA-C-priors"
    assert settings["prompt_sha256"] == hashlib.sha256(VARIANT.read_bytes()).hexdigest()
    assert config["priors"] == "llm" and config["prior_source"] == "llm"


@pytest.mark.usefixtures("require_boxinggym_source")
def test_an_invalid_prior_is_rejected_with_its_reason_and_the_repair_is_admitted(tmp_path, monkeypatch):
    """Check that an invalid prior is rejected with its reason and the repair is admitted."""
    invalid = {"b1": {"dist": "Normal", "params": [0.0, 1.0]}}   # Normal is outside a positive role's support
    out, calls, _fitted = run_with(tmp_path, monkeypatch, [reply(invalid), reply(PRIOR)], "invalid",
                                  priors="llm", prompt_variant=str(VARIANT))
    assert len(calls) == 2
    transport = json.loads((out / "proposals" / "t01.json").read_text())["transport"]
    assert transport["status"] == "parsed" and transport["repairs"] == 1
    reason = transport["rejected_candidates"][0]["reason"]
    assert reason.startswith("proposed prior rejected - ") and "b1" in reason
    raw = json.loads((out / "proposals" / "t01.json").read_text())["raw_response"]
    assert raw[0]["proposed_priors"] == {"b1": ["LogNormal", 1.0, 0.5]}


@pytest.mark.usefixtures("require_boxinggym_source")
def test_without_the_two_options_the_default_path_is_untouched(tmp_path, monkeypatch):
    """Check that without the two options the default path is untouched."""
    out, calls, fitted = run_with(tmp_path, monkeypatch, [reply(PRIOR)], "default")
    assert system_text(calls[0]) == V20.read_text()
    packet = json.loads((out / "packets" / "t01.json").read_text())
    assert "priors" not in packet["constraints"]
    raw = json.loads((out / "proposals" / "t01.json").read_text())["raw_response"]
    assert "prior_source" not in raw[0] and "proposed_priors" not in raw[0]
    assert all(not h.coefficient_priors for h in fitted)
    text = "".join(p.read_text() for p in (out / "fits").glob("*.json"))
    assert "prior_source" not in text


@pytest.mark.usefixtures("require_boxinggym_source")
def test_end_to_end_with_masked_input_names_as_the_paid_cells_run(tmp_path, monkeypatch):
    """Check that end to end with masked input names as the paid cells run."""
    out, calls, _fitted = run_with(tmp_path, monkeypatch, [reply(PRIOR, expression="b0+b1*(X2-X1)")], "masked",
                                  priors="llm", prompt_variant=str(VARIANT), mask_input_names=True)
    assert system_text(calls[0]) == VARIANT.read_text()
    raw = json.loads((out / "proposals" / "t01.json").read_text())["raw_response"]
    assert raw[0]["expression"] == "b0+b1*(X2-X1)" and raw[0]["prior_source"] == "llm"
    fit = json.loads((out / "fits" / f"t01_{raw[0]['model_id']}.json").read_text())
    assert fit["prior_source"] == "llm" and fit["proposed_priors"] == {"b1": ["LogNormal", 1.0, 0.5]}
    settings = json.loads((out / "config.json").read_text())["label_receipt"]["binding"]["settings"]
    assert settings["prompt_sha256"] == hashlib.sha256(VARIANT.read_bytes()).hexdigest()


def test_the_variant_is_a_registered_prompt_version():
    """Check that the variant is a registered prompt version."""
    from .count_proposer import REPOSITORY_PROMPT_FILES, prompt_version_file

    path, digest = REPOSITORY_PROMPT_FILES["v2.9.FA-C-priors"]
    assert ROOT / path == VARIANT and hashlib.sha256(VARIANT.read_bytes()).hexdigest() == digest
    assert prompt_version_file("v2.9.FA-C-priors").read_bytes() == VARIANT.read_bytes()
