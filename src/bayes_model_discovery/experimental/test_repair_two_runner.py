"""Exercise the scoped repair transport through a synthetic complete CORE cell."""

import json
from types import SimpleNamespace

from . import count_proposer as cp
from . import run_label_policy as policy
from .test_count_proposer import one_object, openai_reply, run_args, scripted_openai, stub_fits
from .test_repair_two_authority import authority


def test_synthetic_a8_cell_binds_and_accounts_two_repairs(tmp_path, monkeypatch):
    """Test that synthetic a8 cell binds and accounts two repairs."""
    runner = stub_fits(monkeypatch)
    monkeypatch.setattr(policy, "REPAIR_TWO_ALLOWED_TRACKS", ("X",))
    out = tmp_path / "cell"
    args = run_args(out, rounds=1, cell="111", profile="paper", repair_attempts=2,
                    proposer="openai", llm_model="gpt-4o", allow_paid_api=True,
                    run_label="production")
    _, prompt = runner.select_run_prompt(args, llm_priors=False)
    declared = SimpleNamespace(**{**vars(args), **prompt})
    receipt, _ = authority(tmp_path, settings=policy.bmd_runtime_settings("core", declared))
    args.manifest_ref = json.dumps({"path": receipt["manifest_path"],
                                    "sha256": receipt["manifest_sha256"]})
    expression = "b0+b1*tanh(3*z)"
    client, calls = scripted_openai([
        openai_reply("not-json", index=1),
        openai_reply("still-not-json", index=2),
        openai_reply(one_object(expression, "beta_binomial"), index=3),
    ])
    real_proposer = cp.OpenAICountProposer

    def injected(records, **kwargs):
        """Build the real proposer with the scripted client injected."""
        return real_proposer(records, client=client, **kwargs)

    monkeypatch.setattr(cp, "OpenAICountProposer", injected)
    runner.run(args)

    config = json.loads((out / "config.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    event = json.loads((out / "proposals/t01.json").read_text())
    transport = event["transport"]
    assert config["repair_attempts"] == 2
    assert config["label_receipt"]["binding"]["settings"]["scientific_options"]["repair_attempts"] == 2
    assert len(calls) == transport["calls"] == summary["proposal_accounting"]["llm_calls"] == 3
    assert transport["repairs"] == summary["proposal_accounting"]["llm_repairs"] == 2
    assert len(transport["responses"]) == 3
    assert len(transport["rejected_candidates"]) == 2
    assert all(row["reason"] for row in transport["rejected_candidates"])
    assert transport["usage_total_complete"] is True
    assert transport["usage_total"] == {
        key: sum(row["usage"].get(key, 0) for row in transport["responses"])
        for key in transport["usage_total"]}
    # This counter records the injected transport attempts, not actual billed requests.
    assert summary["paid_api_calls"] == 3
    assert summary["admissions"] == 1
    assert event["events"][0]["status"] == "fitted"
    assert event["events"][0]["expression"] == expression
