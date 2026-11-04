"""Unit ICL-K: the ten-call projection by request kind.

Synthetic clients and bodies only; the one replay test reads the recorded accounting rows of the
stopped c5 easy run from its folder at test time (skipped where that folder is absent) and puts
no recorded value in this file.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import tools.icl_baseline.icl_baseline as module
from tools.icl_baseline.icl_baseline import CLOSED, FIXED, VISIBLE, Unit
from tools.icl_baseline.protocol import render
from tools.icl_baseline.tests.test_fix_round import install_closed_session
from tools.icl_baseline.tests.test_icl import synthetic_history
from tools.icl_baseline.transport import (
    KINDS,
    PROJECTION_RULE,
    Budget,
    BudgetStopped,
    load_prices,
    request_body,
    settings,
)

STOPPED_RUN = Path("<development path>")
TASK_PLAN = {"prediction": 138, "selection": 180}


def bodies():
    observations = synthetic_observations()
    selection = request_body(render("selection", observations)[0])
    prediction = request_body(render("prediction", observations, queries=np.full((4, 7), 1.0))[0])
    return {"selection": selection, "prediction": prediction}


def synthetic_observations():
    from tools.icl_baseline.protocol import Observation
    return [Observation(index, (float(index + 1),) * 7, float(index + 2)) for index in range(3)]


def usage_for(cost, prices, input_tokens=1000):
    """Token counts whose usage-based cost is `cost` to the cent (4 and 20 USD per million)."""
    output = round((cost * 1_000_000 - input_tokens * prices["input_usd_per_1m"]) / prices["output_usd_per_1m"])
    return {"input_tokens": input_tokens, "output_tokens": output}


def feed(budget, rows, remaining):
    """Reserve and record `rows` (kind, usage or None) in order, decrementing `remaining` by kind."""
    body = bodies()
    for kind, usage in rows:
        reserve = budget.reserve(body[kind], remaining, kind)
        budget.record(usage, reserve, kind=kind, request_id=f"{kind}", attempt=0)
        remaining[kind] -= 1
    return remaining


def test_the_recorded_stopped_run_projects_under_the_cap_by_kind():
    if not (STOPPED_RUN / "receipt.json").is_file():
        pytest.skip("the stopped c5 easy run is not on this host")
    receipt = json.loads((STOPPED_RUN / "receipt.json").read_text())
    assert receipt["stop_reason"] == "registered_ten_call_projection_exceeds_cap" and receipt["call_count"] == 10
    prices = load_prices()
    budget = Budget(receipt["settings"]["cap_usd"], prices)
    remaining = dict(TASK_PLAN)
    rows = [(row["kind"], {"input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"]})
            for row in receipt["calls"]]
    feed(budget, rows, remaining)
    # The prices reproduce the recorded costs, and the spent total.
    assert [row["cost_usd"] for row in budget.calls] == [row["cost_usd"] for row in receipt["calls"]]
    assert budget.spent == pytest.approx(receipt["spent_usd"], abs=1e-9)
    costs = {kind: [row["cost_usd"] for row in receipt["calls"] if row["kind"] == kind] for kind in KINDS}
    # The eleventh request: the new rule projects below the cap and does not stop.
    reserve = budget.reserve(bodies()["prediction"], remaining, "prediction")
    projection = budget.projection
    assert budget.stopped_reason is None and projection["exceeds_cap"] is False
    assert projection["remaining_by_kind"] == remaining == {kind: TASK_PLAN[kind] - len(costs[kind]) for kind in KINDS}
    assert projection["c_hi_by_kind"] == {kind: max(costs[kind]) * 1.25 for kind in KINDS}
    assert reserve["c_hi_kind_usd"] == max(costs["prediction"]) * 1.25
    expected = budget.spent + math.fsum(remaining[kind] * max(costs[kind]) * 1.25 for kind in KINDS)
    assert projection["projected_usd"] == pytest.approx(expected, abs=1e-9)
    assert projection["projected_usd"] == pytest.approx(72.52, abs=0.01)
    # The rule before this unit, computed from the same rows: one c_hi for every remaining request.
    single = budget.spent + sum(remaining.values()) * max(costs["prediction"] + costs["selection"]) * 1.25
    assert single == pytest.approx(159.84, abs=0.01) and single > budget.cap


@pytest.mark.parametrize("cap,stops", [(40.0, False), (30.0, True)])
def test_an_expensive_prediction_kind_reduces_the_official_layer_and_stops_only_if_still_over(cap, stops):
    prices = load_prices()
    reductions = []

    def reduce_plan():
        reductions.append(dict(remaining))
        remaining["prediction"] -= 90
        return remaining

    budget = Budget(cap, prices, reduce_plan=reduce_plan)
    remaining = dict(TASK_PLAN)
    rows = [("prediction", usage_for(0.6, prices))] * 4 + [("selection", usage_for(0.008, prices))] * 6
    feed(budget, rows, remaining)
    c_hi = {"prediction": 0.6 * 1.25, "selection": 0.008 * 1.25}
    before = budget.spent + 134 * c_hi["prediction"] + 174 * c_hi["selection"]
    after = budget.spent + 44 * c_hi["prediction"] + 174 * c_hi["selection"]
    assert before > cap and (after > cap) == stops
    if stops:
        with pytest.raises(BudgetStopped, match="registered_ten_call_projection_exceeds_cap"):
            budget.reserve(bodies()["prediction"], remaining, "prediction")
    else:
        budget.reserve(bodies()["prediction"], remaining, "prediction")
        assert budget.stopped_reason is None
    assert reductions == [{"prediction": 134, "selection": 174}]
    projection = budget.projection
    assert projection["projected_usd"] == pytest.approx(before) and projection["exceeds_cap"]
    assert projection["remaining_by_kind_after_reduction"] == {"prediction": 44, "selection": 174}
    assert projection["projected_usd_after_reduction"] == pytest.approx(after)
    assert projection["exceeds_cap_after_reduction"] is stops
    assert projection["rule"] == PROJECTION_RULE and projection["cap_usd"] == cap


def test_a_kind_without_a_usage_cost_in_the_first_ten_calls_keeps_the_reservation_rule():
    prices = load_prices()
    body = bodies()
    # Ten selections with usage; no prediction has been sent. The cap is raised so that the
    # projection (138 predictions at their full reservation) does not stop and can be read.
    budget = Budget(200, prices)
    remaining = feed(budget, [("selection", usage_for(0.008, prices))] * 10, dict(TASK_PLAN))
    reserve = budget.reserve(body["prediction"], remaining, "prediction")
    c_hi = budget.projection["c_hi_by_kind"]
    assert c_hi["selection"] == pytest.approx(0.008 * 1.25)
    assert c_hi["prediction"] == reserve["reservation_usd"] == reserve["c_hi_kind_usd"]
    assert budget.projection["projected_usd"] == pytest.approx(budget.spent + 138 * reserve["reservation_usd"] + 170 * 0.008 * 1.25)
    # Predictions sent but with unknown usage: their largest recorded reservation when the current
    # request is a selection, the current request's own reservation when it is a prediction.
    budget = Budget(90, prices)
    remaining = feed(budget, [("prediction", None)] * 2 + [("selection", usage_for(0.008, prices))] * 8, dict(TASK_PLAN))
    recorded = max(row["reservation_usd"] for row in budget.calls if row["kind"] == "prediction")
    assert budget.c_hi_by_kind(0.5, "selection")["prediction"] == recorded
    assert budget.c_hi_by_kind(0.5, "prediction")["prediction"] == 0.5
    assert budget.c_hi_by_kind(0.5, "selection")["selection"] == pytest.approx(0.008 * 1.25)


def test_the_request_kind_is_checked_and_an_int_remaining_means_the_requests_own_kind():
    budget = Budget(90, load_prices())
    body = bodies()["selection"]
    with pytest.raises(ValueError, match="request kind"):
        budget.reserve(body, 1, "repair")
    reserve = budget.reserve(body, 5, "selection")
    assert set(reserve) == {"reservation_usd", "c_hi_kind_usd", "input_tokens_reserved"}
    for _ in range(10):
        budget.record(usage_for(0.01, load_prices()), reserve, kind="selection")
    budget.reserve(body, 5, "selection")
    assert budget.projection["remaining_by_kind"] == {"selection": 5}


@pytest.mark.parametrize("unit_id", [FIXED, CLOSED, VISIBLE])
def test_settings_declare_the_rule_and_receipts_carry_the_plan_by_kind(tmp_path, monkeypatch, unit_id):
    assert settings(unit_id)["projection_rule"] == PROJECTION_RULE
    groups = ("v2",) if unit_id == VISIBLE else ("v2", "fac")
    histories = [synthetic_history(tmp_path, group, replicate) for group in groups for replicate in (1, 2, 3)]
    if unit_id == CLOSED:
        install_closed_session(monkeypatch, histories)
    unit = Unit(unit_id, tmp_path / "output", histories, mock=True, client=module.SyntheticClient())
    receipt = unit.run()
    assert receipt["status"] == "complete" and receipt["settings"]["projection_rule"] == PROJECTION_RULE
    planned = {"prediction": 138, "selection": 180} if unit_id == CLOSED else {"prediction": receipt["planned_initial_requests"], "selection": 0}
    assert receipt["planned_initial_requests_by_kind"] == receipt["completed_initial_requests_by_kind"] == planned
    assert sum(planned.values()) == receipt["planned_initial_requests"]
    first_ten = {kind: sum(row["kind"] == kind for row in receipt["calls"][:10]) for kind in KINDS}
    projection = receipt["projection"]
    assert projection["rule"] == PROJECTION_RULE and projection["calls_completed"] == 10
    assert projection["remaining_by_kind"] == {kind: planned[kind] - first_ten[kind] for kind in KINDS}
    assert projection["exceeds_cap"] is False and "remaining_by_kind_after_reduction" not in projection
    if unit_id == CLOSED:
        assert first_ten == {"prediction": 4, "selection": 6}
    assert all("c_hi_kind_usd" in row and "c_hi_usd" not in row for row in receipt["calls"])
    starting = json.loads((tmp_path / "output" / "starting_receipt.json").read_text())
    assert starting["settings"]["projection_rule"] == PROJECTION_RULE


@pytest.mark.parametrize("output_tokens,expected_status", [(15000, "complete"), (24000, "budget_stopped")])
def test_the_visible_reduction_reports_its_counts_by_kind(tmp_path, output_tokens, expected_status):
    from types import SimpleNamespace

    class Client(module.SyntheticClient):
        def create(self, **body):
            response = super().create(**body)
            response.usage = SimpleNamespace(model_dump=lambda **_: {"input_tokens": 100, "output_tokens": output_tokens})
            return response

    histories = [synthetic_history(tmp_path, "v2", replicate) for replicate in (1, 2, 3)]
    unit = Unit(VISIBLE, tmp_path / "visible", histories, mock=True, client=Client())
    receipt = unit.run()
    assert receipt["status"] == expected_status and receipt["reduced_visible_plan"]
    projection = receipt["projection"]
    assert projection["remaining_by_kind"] == {"prediction": 86, "selection": 0}
    assert projection["remaining_by_kind_after_reduction"] == {"prediction": 38, "selection": 0}
    assert projection["exceeds_cap_after_reduction"] is (expected_status == "budget_stopped")
    assert receipt["planned_initial_requests_by_kind"] == {"prediction": 48, "selection": 0}
