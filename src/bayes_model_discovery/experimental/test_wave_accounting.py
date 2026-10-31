"""Synthetic record-derived wave accounting, with no provider requests."""

import json
import re
import threading
from types import SimpleNamespace

import pytest

from . import core_count as core
from . import count_proposer as cp
from .test_count_proposer import run_args, stub_fits

CURVE_KEYS = ("active_count_by_round", "proposed_by_round", "admitted_by_round", "retired_by_round")
COST_KEYS = ("initial_pool_llm", "initial_pool_fit", "rounds_llm", "rounds_fit")
ADDED_KEYS = (*CURVE_KEYS, *COST_KEYS)


@pytest.fixture(params=["core", "boxinggym"])
def runner(request, monkeypatch):
    """Provide the (module, argument factory) pair for each benchmark runner under test."""
    stub_fits(monkeypatch)
    if request.param == "core":
        return core, run_args
    request.getfixturevalue("require_boxinggym_source")
    from . import boxinggym_core as bg
    from .test_death_kl import bg_run_args
    monkeypatch.setattr(bg, "count_fit", core.fit)
    return bg, bg_run_args


def _args(factory, output, **kwargs):
    return factory(output, **{"seed": 31, "cell": "100", "rounds": 3, "proposer": "mock",
                               "initial_family": "binomial", "design_policy": "joint_eig", **kwargs})


def _records(output):
    summary = json.loads((output / "summary.json").read_text())
    scores = [json.loads(line) for line in (output / "scores.jsonl").read_text().splitlines()]
    proposals = [(int(path.stem.split("_", 1)[0][1:]), json.loads(path.read_text()))
                 for path in sorted((output / "proposals").glob("t*.json"))]
    return summary, scores, proposals


def _assert_curves(module, output):
    summary, scores, proposals = _records(output)
    assert module.PWC2_ADDED_KEYS == ADDED_KEYS
    assert module.PWC2_SCORE_ADDED_KEYS == ("active_count",)
    active = {}
    for row in scores:
        assert row["active_count"] == len(row["active_ids"])
        active[str(row["round"])] = len(row["active_ids"])
    assert "0" in active
    expected = {key: dict.fromkeys(active, 0) for key in CURVE_KEYS}
    expected["active_count_by_round"] = active
    for t, record in proposals:
        for target, source in (("proposed_by_round", "events"), ("admitted_by_round", "admitted_ids"),
                               ("retired_by_round", "retired_ids")):
            expected[target][str(t)] += len(record.get(source, []))
    for key in CURVE_KEYS:
        assert summary.get("proposal_accounting", summary)[key] == expected[key]


def _assert_costs(output):
    summary, _, _ = _records(output)
    assert set(COST_KEYS) <= set(summary)
    totals = {key: {"calls": 0, "repairs": 0, "failed_calls": 0, "input_tokens": 0,
                    "output_tokens": 0, "usage_complete": True, "wall_seconds": 0.0}
              for key in ("initial_pool_llm", "rounds_llm")}
    for path in sorted((output / "llm_transport").glob("*.response.json")):
        record = json.loads(path.read_text())
        block = totals["initial_pool_llm" if record["round"] == 0 else "rounds_llm"]
        for key in ("calls", "repairs"):
            block[key] += record[key]
        block["failed_calls"] += record["status"] != "parsed"
        for key in ("input_tokens", "output_tokens"):
            block[key] += record["usage_total"].get(key, 0)
        block["usage_complete"] &= (record["usage_total_complete"]
                                      and len(record["responses"]) == record["calls"])
        block["wall_seconds"] += record["elapsed_seconds"]
    for key, expected in totals.items():
        assert summary[key] == expected
    assert sum(summary[key]["calls"] for key in totals) == summary["paid_api_calls"]
    fits = [json.loads(path.read_text()) for path in sorted((output / "fits").glob("*.json"))]
    for key, zero in (("initial_pool_fit", True), ("rounds_fit", False)):
        selected = [record for record in fits if (record["diagnostics"]["round"] == 0) == zero]
        assert summary[key]["count"] == len(selected)
        assert summary[key]["seconds"] == sum(record["fit_seconds"] for record in selected)
    initial = summary["initial_pool_fit"]
    wave_ids = set()
    for path in (output / "proposals").glob("t00_burst*.json"):
        record = json.loads(path.read_text())
        wave_ids.update((record.get("component"), event["model_id"])
                        for event in record["events"] if event["status"] == "fitted")
    zero_fits = [record for record in fits if record["diagnostics"]["round"] == 0]
    wave_fits = [record for record in zero_fits
                 if (record["diagnostics"].get("component"), record["model_id"]) in wave_ids]
    assert initial["wave_count"] == len(wave_fits)
    assert initial["catalogue_count"] == len(zero_fits) - len(wave_fits)
    assert initial["count"] == initial["catalogue_count"] + initial["wave_count"]
    assert initial["count"] + summary["rounds_fit"]["count"] == len(fits)
    return summary


@pytest.mark.parametrize("wave", [False, True])
def test_three_round_curves_are_derived_on_both_option_paths(tmp_path, runner, wave):
    """Round curves and cost blocks are derived with and without the parallel-proposal wave option."""
    module, factory = runner
    output = tmp_path / "cell"
    module.run(_args(factory, output, **({"parallel_proposals": 2} if wave else {})))
    _assert_curves(module, output)
    _assert_costs(output)


def test_legacy_p4_derives_curves_without_an_accounting_container(tmp_path, monkeypatch):
    """The legacy P4 policy derives curves from records even without a proposal-accounting container."""
    stub_fits(monkeypatch)
    output = tmp_path / "cell"
    core.run(_args(run_args, output, rounds=6, proposal_policy="P4", proposal_limit=3))
    _assert_curves(core, output)
    summary = _assert_costs(output)
    assert "proposal_accounting" not in summary
    assert summary["proposed_by_round"]["5"] == 3
    assert set(summary["active_count_by_round"]) == {str(t) for t in range(7)}


@pytest.mark.parametrize("source", ["catalogue", "wave", "both"])
def test_mock_cost_partition_includes_every_round_zero_fit(tmp_path, runner, source):
    """The mock cost partition counts every round-zero fit for catalogue, wave, and mixed pools."""
    module, factory = runner
    output = tmp_path / "cell"
    # The catalogue-only initial pool has no round-zero proposal calls.
    options = {} if source == "catalogue" else {"initial_proposals": 4}
    if source == "wave":
        options["initial_catalogue"] = "off"
    module.run(_args(factory, output, **options))
    summary = _assert_costs(output)
    initial = summary["initial_pool_fit"]
    assert initial["catalogue_count"] == (0 if source == "wave" else 4)
    assert (initial["wave_count"] > 0) == (source != "catalogue")
    for key in ("initial_pool_llm", "rounds_llm"):
        assert summary[key]["calls"] == summary[key]["input_tokens"] == summary[key]["output_tokens"] == 0
        assert summary[key]["usage_complete"] is True


def _fault_client(monkeypatch, module):
    boxing = module is not core
    mapped = lambda form: re.sub(r"\b(z|u)\b", lambda m: {"z": "tz", "u": "t01"}[m[0]], form) if boxing else form
    forms = [mapped(core.P1_MOCK_EXPRESSIONS[index]) for index in (0, 2, 3, 4, 5, 6, 16)]
    good, repaired, loglik_bad, fourth, fifth, sixth, prior_bad = forms
    original_probe = module.prior_probe

    def prior(hypothesis, *args, **kwargs):
        record, predictive = original_probe(hypothesis, *args, **kwargs)
        if hypothesis.expression == prior_bad:
            record = {**record, "passed": False, "detail": "synthetic prior-probe refusal"}
        return record, predictive

    monkeypatch.setattr(module, "prior_probe", prior)
    monkeypatch.setattr(module, "loglik_prior_probe", lambda hypothesis, *a, **k: {
        "passed": hypothesis.expression != loglik_bad, "reason": "synthetic loglik-probe refusal"})
    original = cp.AnthropicCountProposer
    local = threading.local()

    def create(**body):
        wave, index = local.context.get("burst_wave"), local.context.get("burst_call_index")
        if wave == 0 and index == 4:
            raise RuntimeError("synthetic transport failure")
        if wave == 0:
            form = [good, good, prior_bad, loglik_bad, None, fourth][index]
            if index == 2 and len(body["messages"]) > 1:
                form = repaired
        elif wave == 1:
            form = [good, fifth, sixth][index]
        elif wave == 2:
            # The burst's third wave repeats `good`, as the
            # duplicate call of each earlier wave does, so the burst stops after three waves still
            # one short, and the later-round call below keeps a reply the burst never admitted.
            form = good
        else:
            form = mapped(core.P1_MOCK_EXPRESSIONS[14])
        candidate = {"expression": form, "family": "binomial", "parameters": dict(core.mock_parameters(form))}
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(candidate))],
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 12, "output_tokens": 3}),
            id="synthetic-accounting", model="synthetic-model", stop_reason="end_turn")

    def factory(records, **kwargs):
        instance = original(records, client=SimpleNamespace(messages=SimpleNamespace(create=create)))
        propose = instance.propose_one

        def call(packet, **options):
            local.context = options["context"]
            return propose(packet, **options)

        instance.propose_one = call
        return instance

    monkeypatch.setattr(cp, "AnthropicCountProposer", factory)


def test_mixed_wave_failures_are_counted_from_existing_events_and_transports(tmp_path, monkeypatch, runner):
    """Wave failure counts are derived from existing events and transport records."""
    module, factory = runner
    _fault_client(monkeypatch, module)
    output = tmp_path / "cell"
    module.run(_args(factory, output, rounds=1, proposal_rounds="2", proposer="anthropic",
                     allow_paid_api=True, llm_model="synthetic-model", initial_proposals=6,
                     initial_catalogue="off", burst_concurrency=3, fit_guards="v1"))
    summary = _assert_costs(output)
    record = json.loads((output / "proposals/t00_burst.json").read_text())
    statuses = [event["status"] for event in record["events"]]
    prior_rejections = sum(item.get("error_type") == "PriorProbeRejection"
                           for entry in record["transports"]
                           for item in entry["transport"]["rejected_candidates"])
    expected = {"accepted_total": statuses.count("fitted"),
                "shortfall_after_refill": 6 - statuses.count("fitted"),
                "duplicates": statuses.count("duplicate_not_refitted"),
                "rejected_prior_probe": statuses.count("rejected_prior_probe") + prior_rejections,
                "rejected_loglik_probe": statuses.count("rejected_loglik_prior_probe"),
                "fit_failed": sum(status in ("fit_failed", "fit_timeout", "fit_or_predict_failed")
                                  for status in statuses),
                "failed_calls": sum(entry["transport"]["status"] != "parsed" for entry in record["transports"])}
    # Wave 1 leaves one short, so a third wave asks for that one; its reply is
    # a third duplicate, so the burst still ends one short. The other counts are unchanged.
    assert expected == {"accepted_total": 5, "shortfall_after_refill": 1, "duplicates": 3,
                        "rejected_prior_probe": 1, "rejected_loglik_probe": 1, "fit_failed": 0, "failed_calls": 1}
    for key, value in expected.items():
        assert record[key] == summary["proposal_accounting"][key] == value
    assert [wave["wave"] for wave in record["waves"]] == [0, 1, 2]  # three waves
    assert sum(wave["accepted"] for wave in record["waves"]) == record["accepted_total"]
    assert record["initial_catalogue"] == "off" and len(record["after_ids"]) == 5
    assert record["max_in_flight"] <= 3
    assert summary["initial_pool_llm"]["calls"] == 11  # 6 + 1 repair + 3 + 1
    assert summary["initial_pool_llm"]["usage_complete"] is False


def test_failed_fit_statuses_and_repaired_prior_refusals_are_distinct():
    """Failed-fit statuses and repaired prior-probe refusals are counted separately."""
    events = [{"status": value} for value in ("fit_failed", "fit_timeout", "fit_or_predict_failed",
                                             "rejected_prior_probe", "rejected_loglik_prior_probe")]
    transports = [{"transport": {"status": "parsed", "rejected_candidates": [
        {"error_type": "PriorProbeRejection"}, {"error_type": "ValueError"}]}}]
    assert core.wave_event_counts(events, transports, 5) == {
        "accepted_total": 0, "shortfall_after_refill": 5, "rejected_prior_probe": 2,
        "rejected_loglik_probe": 1, "fit_failed": 3, "failed_calls": 0, "duplicates": 0}


def test_initial_and_later_llm_cost_axes_do_not_share_calls(tmp_path, monkeypatch, runner):
    """Initial-pool and later-round LLM cost axes never count the same call twice."""
    module, factory = runner
    _fault_client(monkeypatch, module)
    output = tmp_path / "cell"
    module.run(_args(factory, output, rounds=1, proposer="anthropic", allow_paid_api=True,
                     llm_model="synthetic-model", initial_proposals=6, initial_catalogue="off",
                     burst_concurrency=3, fit_guards="v1"))
    summary = _assert_costs(output)
    # The burst's third wave adds one round-0 call (6 + 1 repair + 3 + 1);
    # round 1 keeps its one call, so the paid total is 11 + 1.
    assert summary["initial_pool_llm"]["calls"] == 11
    assert summary["rounds_llm"]["calls"] == 1
    assert summary["rounds_llm"]["input_tokens"] == 12
    assert summary["rounds_llm"]["output_tokens"] == 3
    assert summary["rounds_llm"]["usage_complete"] is True
    assert summary["paid_api_calls"] == 12  # 11 + 1


def test_cost_index_includes_all_retry_files_and_legacy_usage(tmp_path):
    """The cost index covers every retry response file and legacy usage records."""
    transport = tmp_path / "llm_transport"
    transport.mkdir()
    retry = {"round": 0, "calls": 1, "repairs": 0, "status": "proposal_failed",
             "usage_total": {}, "usage_total_complete": True, "responses": [], "elapsed_seconds": 2.0}
    accepted = {"round": 0, "calls": 1, "repairs": 0, "status": "parsed",
                "usage_total": {"prompt_tokens": 7, "completion_tokens": 3},
                "usage_total_complete": True, "responses": [{"usage_known": True}], "elapsed_seconds": 4.0}
    legacy = {"event": 5, "api_request_attempted": True, "status": "parsed",
              "usage": {"input_tokens": 11, "output_tokens": 5}, "usage_known": True, "elapsed_seconds": 8.0}
    for name, record in (("round_00__burst_w0_c00", retry),
                         ("round_00__burst_w0_c00_r1", accepted), ("event_05", legacy)):
        (transport / f"{name}.response.json").write_text(json.dumps(record))
    actual = core.wave_cost_blocks(tmp_path)
    assert actual["initial_pool_llm"] == {"calls": 2, "repairs": 0, "failed_calls": 1,
        "input_tokens": 7, "output_tokens": 3, "usage_complete": False, "wall_seconds": 6.0}
    assert actual["rounds_llm"] == {"calls": 1, "repairs": 0, "failed_calls": 0,
        "input_tokens": 11, "output_tokens": 5, "usage_complete": True, "wall_seconds": 8.0}


def test_component_pool_counts_preserve_the_component_units():
    """Pool counts keep the per-component units alongside the total."""
    row = {"components": [{"active_ids": ["a", "b"]}, {"active_ids": ["c"]}]}
    core.add_recorded_pool_count(row)
    assert row["active_count"] == 3
    assert [component["active_count"] for component in row["components"]] == [2, 1]


def test_initial_fit_sources_preserve_component_identity(tmp_path):
    """Initial fit sources keep component identity when model ids coincide."""
    (tmp_path / "fits").mkdir()
    (tmp_path / "proposals").mkdir()
    wave = {"component": "first", "events": [{"status": "fitted", "model_id": "same-model"}]}
    (tmp_path / "proposals/t00_burst_first.json").write_text(json.dumps(wave))
    for component in ("first", "second"):
        record = {"model_id": "same-model", "fit_seconds": 0.25,
                  "diagnostics": {"round": 0, "component": component}}
        (tmp_path / f"fits/t00_{component}.json").write_text(json.dumps(record))
    assert core.wave_cost_blocks(tmp_path)["initial_pool_fit"] == {
        "count": 2, "seconds": 0.5, "catalogue_count": 1, "wave_count": 1}


