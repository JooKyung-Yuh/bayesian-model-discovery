"""Development-only prompt file selection; synthetic text and injected clients only."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from . import chembench_core as cc
from . import count_proposer as cp
from . import test_chembench_core as fake
from .test_chembench_mda_options import stub_fits


@pytest.fixture(autouse=True)
def isolated_prompt(monkeypatch):
    """Fixture isolating the prompt-selection state for one test."""
    stub_fits(monkeypatch)
    cp.select_prompt_version(None)
    cp.set_prompt_variant(None)
    yield
    cp.select_prompt_version(None)
    cp.set_prompt_variant(None)


@pytest.fixture
def variant(tmp_path):
    """Fixture writing a prompt variant file and returning its path."""
    path = tmp_path / "synthetic_variant.txt"
    path.write_text("# prompt_version: synthetic-gs-variant\nReturn a candidate model.\n")
    return path


def argv(tmp_path, extra=()):
    """Build the command-line arguments for a run, with optional extras."""
    return ["chembench_core", "--manifest", str(fake._manifest(tmp_path)),
            "--output", str(tmp_path / "cell"), "--noise", "0.01", "--rounds", "1",
            "--particles", "16", "--q", "8", *extra]


def test_cli_variant_sends_the_whole_file_and_records_its_real_identity(
    tmp_path, monkeypatch, variant,
):
    """Check that cli variant sends the whole file and records its real identity."""
    calls = []

    def create(**body):
        """Stub client call that records the request body and returns a canned reply."""
        calls.append(body)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps({
                "expression": "b0", "family": "lognormal", "parameters": {"b0": "intercept"},
            }))],
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 1, "output_tokens": 1}),
            id="synthetic-response", model="synthetic-model", stop_reason="end_turn",
        )

    original = cp.AnthropicCountProposer
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(cp, "AnthropicCountProposer",
                        lambda records, **kwargs: original(records, client=client))
    monkeypatch.setattr(sys, "argv", argv(tmp_path, [
        "--cell", "100", "--proposer", "anthropic", "--allow-paid-api",
        "--run-label", "development", "--prompt-variant", str(variant),
    ]))
    assert cc.main() is None
    assert calls and all(call["system"] == variant.read_text() for call in calls)
    identity = {"prompt_version": "synthetic-gs-variant",
                "prompt_sha256": hashlib.sha256(variant.read_bytes()).hexdigest()}
    out = tmp_path / "cell"
    config = json.loads((out / "config.json").read_text())
    transport = json.loads((out / "proposals/t01.json").read_text())["transport"]
    for record in (config, transport):
        assert {key: record[key] for key in identity} == identity
    assert config["prompt_variant"] == str(variant)
    requests = list((out / "llm_transport").glob("*.request.json"))
    assert requests
    for path in requests:
        record = json.loads(path.read_text())
        assert {key: record["identity"][key] for key in identity} == identity


def test_missing_tag_is_refused_before_output(tmp_path, monkeypatch):
    """Check that missing tag is refused before output."""
    path = tmp_path / "untagged.txt"
    path.write_text("Synthetic system text without a tag.\n")
    monkeypatch.setattr(sys, "argv", argv(tmp_path, ["--prompt-variant", str(path)]))
    with pytest.raises(ValueError, match="tag on line 1"):
        cc.main()
    assert not (tmp_path / "cell").exists()


def test_ts_cli_records_display_and_summary_with_an_injected_anthropic_client(tmp_path, monkeypatch):
    """Check that ts cli records display and summary with an injected anthropic client."""
    calls = []

    def create(**body):
        """Stub client call that records the request body and returns a canned reply."""
        calls.append(body)
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking="synthetic summary", signature="sig"),
                     SimpleNamespace(type="text", text=json.dumps({
                         "expression": "b0", "family": "lognormal", "parameters": {"b0": "intercept"},
                     }))],
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 1, "output_tokens": 1}),
            id="synthetic-ts-response", model="synthetic-model", stop_reason="end_turn",
        )

    original = cp.AnthropicCountProposer
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(cp, "AnthropicCountProposer",
                        lambda records, **kwargs: original(records, client=client, **kwargs))
    monkeypatch.setattr(sys, "argv", argv(tmp_path, [
        "--cell", "100", "--proposer", "anthropic", "--allow-paid-api",
        "--run-label", "development", "--llm-reasoning",
        "effort=medium,thinking=adaptive,display=summarized",
    ]))
    assert cc.main() is None
    expected = {"effort": "medium", "thinking": "adaptive", "display": "summarized"}
    out = tmp_path / "cell"
    assert json.loads((out / "config.json").read_text())["llm_reasoning"] == expected
    assert calls and all(call["thinking"] == {"type": "adaptive", "display": "summarized"} for call in calls)
    responses = list((out / "llm_transport").glob("*.response.json"))
    assert responses
    for path in responses:
        record = json.loads(path.read_text())
        assert record["decoding"]["reasoning"] == expected
        assert all(reply["thinking_text"] == ["synthetic summary"] for reply in record["responses"])


def test_numbered_version_and_variant_are_refused_before_output(tmp_path, monkeypatch, variant):
    """Check that numbered version and variant are refused before output."""
    monkeypatch.setattr(sys, "argv", argv(tmp_path, [
        "--prompt-variant", str(variant), "--prompt-version", "v2.0",
    ]))
    with pytest.raises(ValueError, match="state at most one"):
        cc.main()
    assert not (tmp_path / "cell").exists()


def test_even_a_verified_production_receipt_cannot_enable_a_variant(tmp_path, monkeypatch, variant):
    # Isolate the runner's new refusal from the existing missing-manifest refusal.
    """Check that even a verified production receipt cannot enable a variant."""
    monkeypatch.setattr(cc, "bmd_label_receipt", lambda *a, **k: {"run_label": "production"})
    monkeypatch.setattr(sys, "argv", argv(tmp_path, [
        "--prompt-variant", str(variant), "--run-label", "production",
    ]))
    with pytest.raises(ValueError, match="development-only"):
        cc.main()
    assert not (tmp_path / "cell").exists()
    assert not cp.prompt_variant_active()


def test_pool_generator_and_variant_remain_exclusive(tmp_path, monkeypatch, variant):
    """Check that pool generator and variant remain exclusive."""
    monkeypatch.setattr(sys, "argv", argv(tmp_path, [
        "--prompt-variant", str(variant), "--pool-generator", "pool_generator_v1",
    ]))
    with pytest.raises(ValueError, match="excludes --prompt-version, --prompt-variant"):
        cc.main()
    assert not (tmp_path / "cell").exists()


def test_omitting_variant_leaves_no_namespace_or_config_key(tmp_path, monkeypatch):
    """Check that omitting variant leaves no namespace or config key."""
    original = cc.run
    captured = []

    def recording(args):
        """Stub entry point that records the parsed arguments."""
        captured.append(vars(args).copy())
        return original(args)

    monkeypatch.setattr(cc, "run", recording)
    monkeypatch.setattr(sys, "argv", argv(tmp_path))
    assert cc.main() is None
    assert "prompt_variant" not in captured[0]
    assert "prompt_variant" not in json.loads((tmp_path / "cell/config.json").read_text())
