"""RPR2: synthetic manifests/clients only; the real allow-list remains empty."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import count_proposer as cp
from . import run_label_policy as policy
from .test_count_proposer import one_object, openai_reply, run_args, scripted_openai
from .test_run_label_policy import registration, runner_args, write_manifest

GIB = 1024 ** 3


def launch_contract(**changes):
    """A synthetic launch contract: every key `launch_contract.validate_launch_contract` requires."""
    options = {"per_host_parallel_cap_policy": "memory_derived",
               "baseline_cgroup_memory_bytes": 8 * GIB,
               "per_cell_peak_increment_bytes": 24 * GIB,
               "memory_safety_reserve_bytes": 8 * GIB,
               "parallel_requested": 2, "track_global_inflight_cap": 4,
               "global_budget_ledger": "synthetic/global_budget.json",
               "per_cell_output_bytes": 1024, "disk_safety_reserve_bytes": 1024}
    options.update(changes)
    return options


def authority(tmp_path, *, track="X", arm="A8", provider="openai", settings=None):
    """Write a synthetic registered manifest and resolve a production label receipt for it; return the receipt and a context dict."""
    settings = settings or policy.bmd_runtime_settings(
        "core", runner_args(rounds=7, repair_attempts=2, cell="111"))
    reference, identity, manifest = registration(
        tmp_path, cell="111", proposer=provider, settings=settings)
    manifest["core_track"] = {"track_id": track, "arm": arm}
    manifest.update(launch_contract(), contract_status="registered",
                    execution_commit_rule={"rule": "synthetic test fixture only"},
                    cell_timeout_minutes={"core": 5})
    reference = write_manifest(Path(reference["path"]), manifest)
    receipt = policy.resolve_label("production", manifest_ref=reference, **identity)
    return receipt, {"run_id": "synthetic", "world_seed": 11, "cell": "111",
                     "proposal_rep": 0, "round": 7}


def injected(tmp_path, provider):
    """Build a proposer with a scripted client; return it with the list of recorded calls."""
    texts = ["not-json", "still-not-json", one_object()]
    if provider == "openai":
        client, calls = scripted_openai([openai_reply(text, index=i + 1)
                                         for i, text in enumerate(texts)])
        return cp.OpenAICountProposer(tmp_path, client=client, model="gpt-4o"), calls
    calls = []

    def create(**body):
        """Record the call body and return the next scripted reply."""
        calls.append(body)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=texts[len(calls) - 1])],
            usage=SimpleNamespace(model_dump=lambda **kw: {"input_tokens": 12, "output_tokens": 3}),
            id="synthetic-response", model="synthetic-model", stop_reason="end_turn")

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return cp.AnthropicCountProposer(tmp_path, client=client, model="test-model"), calls


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_unlisted_track_refuses_even_verified_a8_before_api_or_output(tmp_path, provider):
    """Test that unlisted track refuses even verified a8 before api or output."""
    assert policy.REPAIR_TWO_ALLOWED_TRACKS == ("prompt_ladder_stage1_20260921",)
    # The fixture's track is "X", which the shipped allow-list does not name.
    receipt, context = authority(tmp_path, provider=provider)
    output = tmp_path / "transport"
    proposer, calls = injected(output, provider)
    with pytest.raises(ValueError, match="verified CORE A8"):
        proposer.propose_one("synthetic packet", context=context, initial_family="binomial",
                             repair_attempts=2, repair_receipt=receipt)
    assert calls == []
    # The existing transport constructor makes its directory; refusal writes no files.
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_synthetic_registered_a8_performs_exactly_two_repairs(tmp_path, monkeypatch, provider):
    """Test that synthetic registered a8 performs exactly two repairs."""
    monkeypatch.setattr(policy, "REPAIR_TWO_ALLOWED_TRACKS", ("X",))
    receipt, context = authority(tmp_path, provider=provider)
    proposer, calls = injected(tmp_path / "transport", provider)
    models, record = proposer.propose_one(
        "synthetic packet", context=context, initial_family="binomial",
        repair_attempts=2, repair_receipt=receipt)
    assert len(models) == 1
    assert len(calls) == record["calls"] == 3
    assert record["repairs"] == 2
    assert len(record["responses"]) == 3
    assert len(record["rejected_candidates"]) == 2
    assert all(row["repair_eligible"] and row["reason"] for row in record["rejected_candidates"])
    assert record["usage_total_complete"] is True
    assert record["usage_total"] == {
        key: sum(row["usage"].get(key, 0) for row in record["responses"])
        for key in record["usage_total"]}
    assert receipt["binding"]["settings"]["scientific_options"]["repair_attempts"] == 2


@pytest.mark.parametrize("arm,track", [(f"A{i}", "X") for i in range(8)] + [("A8", "Y")])
def test_other_arms_and_tracks_refuse_before_requests(tmp_path, monkeypatch, arm, track):
    """Test that other arms and tracks refuse before requests."""
    monkeypatch.setattr(policy, "REPAIR_TWO_ALLOWED_TRACKS", ("X",))
    receipt, context = authority(tmp_path, arm=arm, track=track)
    out = tmp_path / "transport"
    proposer, calls = injected(out, "openai")
    with pytest.raises(ValueError, match="verified CORE A8"):
        proposer.propose_one("synthetic packet", context=context, initial_family="binomial",
                             repair_attempts=2, repair_receipt=receipt)
    assert calls == [] and list(out.iterdir()) == []


@pytest.mark.parametrize("mismatch", ["provider", "model", "initial_family"])
def test_authority_cannot_be_reused_by_a_different_request(tmp_path, monkeypatch, mismatch):
    """Test that authority cannot be reused by a different request."""
    monkeypatch.setattr(policy, "REPAIR_TWO_ALLOWED_TRACKS", ("X",))
    receipt, context = authority(tmp_path)
    proposer, calls = injected(tmp_path / "transport", "anthropic" if mismatch == "provider" else "openai")
    if mismatch == "model":
        proposer.model = "synthetic-other-model"
    with pytest.raises(ValueError, match="proposer identity"):
        proposer.propose_one("synthetic packet", context=context,
            initial_family="lognormal" if mismatch == "initial_family" else "binomial",
            repair_attempts=2, repair_receipt=receipt)
    assert calls == [] and list(proposer.records.iterdir()) == []


@pytest.mark.parametrize("damage", ["manifest", "rows", "receipt", "context", "burst", "three"])
def test_authority_is_rechecked_and_does_not_authorize_bursts(tmp_path, monkeypatch, damage):
    """Test that authority is rechecked and does not authorize bursts."""
    monkeypatch.setattr(policy, "REPAIR_TWO_ALLOWED_TRACKS", ("X",))
    receipt, context = authority(tmp_path)
    if damage in ("manifest", "rows"):
        path = Path(receipt["manifest_path"]) if damage == "manifest" else tmp_path / "rows.txt"
        path.write_bytes(path.read_bytes() + b"\n")
    elif damage == "receipt":
        receipt["binding"]["seed"] = 17
    elif damage == "context":
        context["world_seed"] = 17
    with pytest.raises(ValueError):
        cp._validate_repair_attempts(3 if damage == "three" else 2,
            receipt=receipt, context=context, burst=damage == "burst")


@pytest.mark.parametrize("missing", ["global_budget_ledger", "per_cell_output_bytes",
                                      "execution_commit_rule", "cell_timeout_minutes"])
def test_label_manifest_alone_is_not_launch_authority(tmp_path, monkeypatch, missing):
    """Test that label manifest alone is not launch authority."""
    monkeypatch.setattr(policy, "REPAIR_TWO_ALLOWED_TRACKS", ("X",))
    receipt, _ = authority(tmp_path)
    path = Path(receipt["manifest_path"])
    manifest = json.loads(path.read_text())
    manifest.pop(missing)
    reference = write_manifest(path, manifest)
    binding = receipt["binding"].copy()
    # Re-resolve authentic label metadata; only the launch contract is incomplete.
    fresh = policy.resolve_label("production", manifest_ref=reference, **binding)
    with pytest.raises(ValueError, match="verified CORE A8"):
        policy.validate_repair_attempts(2, bundle="core", receipt=fresh)


@pytest.mark.parametrize("bundle", ["core", "chembench"])
@pytest.mark.parametrize("value", [2, 3, -1])
def test_cli_refuses_unbound_amplification_before_output(tmp_path, monkeypatch, capsys, bundle, value):
    """Test that cli refuses unbound amplification before output."""
    from . import chembench_core, core_count

    out = tmp_path / "no-output"
    argv = [bundle, "--output", str(out), "--repair-attempts", str(value)]
    if bundle == "chembench":
        argv += ["--manifest", str(tmp_path / "not-read.json"), "--noise", "0.01"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        (core_count if bundle == "core" else chembench_core).main()
    assert caught.value.code == 2
    assert "repair" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("stated", [None, 0, 1])
def test_chembench_omitted_setting_adds_no_key(tmp_path, monkeypatch, stated):
    """Test that chembench omitted setting adds no key."""
    from . import chembench_core

    argv = ["chembench", "--manifest", str(tmp_path / "not-read.json"),
            "--noise", "0.01", "--output", str(tmp_path / "cell")]
    if stated is not None:
        argv += ["--repair-attempts", str(stated)]
    captured = []
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(chembench_core, "run", captured.append)
    chembench_core.main()
    args = captured[0]
    assert getattr(args, "repair_attempts", 1) == (1 if stated is None else stated)
    args.prompt_version, args.prompt_sha256 = "v9.9", "a" * 64
    settings = policy.bmd_runtime_settings("chembench", args)
    assert ("repair_attempts" in vars(args)) == (stated is not None)
    assert ("repair_attempts" in settings) == (stated is not None)
    if stated is not None:
        assert settings["repair_attempts"] == stated


def test_core_runner_reads_the_same_allow_list_before_mkdir(tmp_path, monkeypatch):
    """Test that core runner reads the same allow list before mkdir."""
    from . import core_count

    out = tmp_path / "no-output"
    args = run_args(out, rounds=7, cell="111", profile="paper", repair_attempts=2,
                    proposer="openai", llm_model="gpt-4o", allow_paid_api=True,
                    run_label="production")
    _, prompt = core_count.select_run_prompt(args, llm_priors=False)
    declared = SimpleNamespace(**{**vars(args), **prompt})
    receipt, _ = authority(tmp_path, settings=policy.bmd_runtime_settings("core", declared))
    args.manifest_ref = json.dumps({"path": receipt["manifest_path"], "sha256": receipt["manifest_sha256"]})
    with pytest.raises(ValueError, match="verified CORE A8"):
        core_count.run(args)
    assert not out.exists()
    monkeypatch.setattr(policy, "REPAIR_TWO_ALLOWED_TRACKS", ("X",))
    original = Path.mkdir

    class ReachedOutput(Exception):
        """Raised when the run reaches its output directory."""

    def mkdir(path, *a, **kw):
        """Raise when the guarded output directory is reached; otherwise delegate."""
        if path == out:
            raise ReachedOutput
        return original(path, *a, **kw)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    with pytest.raises(ReachedOutput):
        core_count.run(args)
    assert not out.exists()
