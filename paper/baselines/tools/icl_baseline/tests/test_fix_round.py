"""F1-F5 regression cases use synthetic histories and clients only."""

import copy
import json
import math
import re
from types import SimpleNamespace

import pytest

import tools.icl_baseline.icl_baseline as module
from tools.icl_baseline.icl_baseline import CLOSED, FIXED, VISIBLE, History, SyntheticClient, Unit
from tools.icl_baseline.protocol import (
    DOMAIN_WORDS,
    OFFICIAL_TIMES,
    ROOT,
    TIMES,
    VISIBLE_NAMES,
    json_bytes,
    render,
    repair_request,
    validate,
)
from tools.icl_baseline.tests.test_icl import synthetic_history
from tools.icl_baseline.transport import Budget, Transport, load_prices, request_body, settings


@pytest.mark.parametrize("summary,expected", [
    ("ping-pong", True), ("ping pong", True), ("pingpong", True),
    ("Ping-Pong", True), ("shipping", False),
])
def test_domain_incidence_matches_e6_search(tmp_path, summary, expected):
    scorer_rule = re.compile(r"enzyme|substrate|michaelis|inhibit|ping[- ]?pong|kinetic|catalys", re.IGNORECASE)
    assert bool(DOMAIN_WORDS.search(summary)) == bool(scorer_rule.search(summary)) == expected

    class Client(SyntheticClient):
        def create(self, **body):
            response = super().create(**body)
            response.content[0].thinking = summary
            return response

    transport = Transport(Client(), tmp_path, Budget(45, load_prices()), synthetic=True)
    prompt, _ = render("selection", [])
    transport.request(prompt, "synthetic", "selection", [])
    assert transport.counts["selection"] == {"summaries": 1, "with_domain_word": int(expected)}


@pytest.mark.parametrize("text,stop,expected_error", [
    ("", "end_turn", "empty text"), (" \n\t ", "end_turn", "empty text"),
    ("", "max_tokens", "response stopped at max_tokens"),
    (" \t", "max_tokens", "response stopped at max_tokens"),
])
def test_empty_first_reply_resends_identical_request(tmp_path, text, stop, expected_error):
    class Client(SyntheticClient):
        def __init__(self):
            super().__init__()
            self.bodies = []

        def create(self, **body):
            self.bodies.append(copy.deepcopy(body))
            response = super().create(**body)
            if len(self.bodies) == 1:
                response.content[-1].text = text
                response.stop_reason = stop
            return response

    client = Client()
    transport = Transport(client, tmp_path, Budget(45, load_prices()), synthetic=True)
    prompt, _ = render("selection", [])
    result = transport.request(prompt, "empty", "selection", [])
    assert len(client.bodies) == 2
    assert json_bytes(client.bodies[0]) == json_bytes(client.bodies[1])
    assert result["attempt_statuses"] == ["retryable_validation_failure", "valid"]
    validation = json.loads((tmp_path / "transport/empty.attempt0.validation.json").read_text())
    assert validation == {"status": "retryable_validation_failure", "errors": [expected_error]}
    assert [row["status"] for row in transport.budget.calls] == result["attempt_statuses"]


@pytest.mark.parametrize("stop", ["end_turn", "max_tokens"])
def test_nonempty_failure_keeps_original_repair_body(tmp_path, stop):
    class Client(SyntheticClient):
        def __init__(self):
            super().__init__()
            self.bodies = []

        def create(self, **body):
            self.bodies.append(copy.deepcopy(body))
            response = super().create(**body)
            if len(self.bodies) == 1:
                response.content[-1].text = "{}"
                response.stop_reason = stop
            return response

    client = Client()
    prompt, _ = render("selection", [])
    transport = Transport(client, tmp_path, Budget(45, load_prices()), synthetic=True)
    transport.request(prompt, "repair", "selection", [])
    errors = validate("{}", "selection", stop_reason=stop).errors
    expected = request_body(repair_request(prompt, "{}", "; ".join(errors), "selection", 0))
    assert json_bytes(client.bodies[1]) == json_bytes(expected)
    assert client.bodies[1]["messages"][1] == {"role": "assistant", "content": "{}"}


def test_empty_replies_still_consume_exactly_two_attempts(tmp_path):
    class Client(SyntheticClient):
        def create(self, **body):
            response = super().create(**body)
            response.content[-1].text = ""
            return response

    transport = Transport(Client(), tmp_path, Budget(45, load_prices()), synthetic=True)
    prompt, _ = render("selection", [])
    result = transport.request(prompt, "empty-twice", "selection", [])
    assert result["status"] == "selection_fallback"
    assert result["attempt_statuses"] == ["retryable_validation_failure"] * 2
    assert len(transport.budget.calls) == 2


def test_nontext_validation_retains_invalid_json_refusal():
    assert validate(None, "selection").errors == ["invalid JSON object"]


def test_unknown_usage_is_charged_but_not_used_for_observed_projection():
    body = request_body(render("selection", [])[0])
    budget = Budget(45, load_prices())
    reserve = budget.reserve(body, 270, "selection")
    unknown = budget.record({}, reserve, status="transport_failure", kind="selection")
    known = [budget.record({"input_tokens": 100, "output_tokens": 1000}, reserve, kind="selection") for _ in range(9)]
    next_reserve = budget.reserve(body, 260, "selection")
    assert budget.projection_done and budget.stopped_reason is None
    assert next_reserve["c_hi_kind_usd"] == max(row["cost_usd"] for row in known) * 1.25
    assert unknown["cost_usd"] == reserve["reservation_usd"]
    assert budget.spent == math.fsum([reserve["reservation_usd"], *(r["cost_usd"] for r in known)])
    assert budget.spent + 260 * unknown["cost_usd"] * 1.25 > budget.cap


def test_zero_known_usage_falls_back_to_current_reservation_without_inflation():
    body = request_body(render("selection", [])[0])
    budget = Budget(100, load_prices())
    reserve = budget.reserve(body, 20, "selection")
    for _ in range(10):
        budget.record({}, reserve, kind="selection")
    next_reserve = budget.reserve(body, 10, "selection")
    assert next_reserve["c_hi_kind_usd"] == next_reserve["reservation_usd"]
    assert budget.spent == math.fsum([reserve["reservation_usd"]] * 10)


@pytest.mark.parametrize("cache_field", ["cache_creation_input_tokens", "cache_read_input_tokens"])
def test_cache_usage_remains_unknown_for_accounting_and_projection(cache_field):
    body = request_body(render("selection", [])[0])
    budget = Budget(45, load_prices())
    reserve = budget.reserve(body, 270, "selection")
    known = budget.record({"input_tokens": 100, "output_tokens": 1000}, reserve, kind="selection")
    for _ in range(9):
        row = budget.record({"input_tokens": 100, "output_tokens": 1000, cache_field: 10}, reserve, kind="selection")
        assert row["cost_basis"] == "maximum_reservation_unknown_usage"
        assert row["cost_usd"] == reserve["reservation_usd"]
    assert budget.reserve(body, 260, "selection")["c_hi_kind_usd"] == known["cost_usd"] * 1.25


def test_mock_fixed_cli_scores_all_six_synthetic_histories(tmp_path, monkeypatch):
    histories = [synthetic_history(tmp_path, g, r) for g in ("v2", "fac") for r in (1, 2, 3)]
    monkeypatch.setattr(History, "load", lambda path, group, replicate: next(h for h in histories if h.root == path))
    monkeypatch.setattr(History, "files", lambda self: [])
    monkeypatch.setattr(module, "dependencies", list)
    monkeypatch.setattr(module, "live_client", lambda *a, **kw: pytest.fail("no live client"))
    output = tmp_path / "fixed"
    args = ["--unit", FIXED, "--mock", "--output", str(output), "--registration", str(ROOT / "prices.json")]
    for history in histories:
        args += ["--history" if history.group == "v2" else "--fac-history", str(history.root)]
    assert module.main(args) == 0
    data = json.loads((output / "scores.json").read_text())
    assert len(data["cells"]) == 66
    assert {row["group"] for row in data["cells"]} == {"v2", "fac"}
    expected = {(g, n, 256) for g in ("v2", "fac") for n in TIMES}
    expected |= {(g, n, 1000) for g in ("v2", "fac") for n in OFFICIAL_TIMES}
    assert {(r["group"], r["n"], r["evaluation_size"]) for r in data["timepoint_directions"]} == expected
    for row in data["cells"]:
        if row["evaluation_size"] == 1000:
            assert row["ours"]["noiseless_observation"] is None
            assert row["direction"] == "no directional statement"
    assert all(r["direction"] == "no directional statement" for r in data["timepoint_directions"] if r["evaluation_size"] == 1000)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["completed_initial_requests"] == 270 and receipt["paid_api_calls"] == 0
    assert not receipt["reduced_official_layer"]


def test_closed_finish_compares_each_reference_group_separately(tmp_path, monkeypatch):
    histories = [synthetic_history(tmp_path, g, r) for g in ("v2", "fac") for r in (1, 2, 3)]

    def ours(history, size, n):
        value = 2.0 if history.group == "v2" else .5
        return {"primary_rate": value, "noiseless_observation": value if size == 256 else None}

    monkeypatch.setattr(History, "ours", ours)
    unit = Unit(CLOSED, tmp_path / "closed", histories, mock=True, client=SyntheticClient())
    unit.results = [{"n": 10, "evaluation_size": size, "replicate": r,
                     "status": "valid", "design_status": "valid",
                     "icl": {"primary_rate": 1., "noiseless_observation": 1.}}
                    for size in (256, 1000) for r in (1, 2, 3)]
    unit.finish("complete")
    rows = json.loads((unit.output / "scores.json").read_text())["timepoint_directions"]
    assert {(r["group"], r["evaluation_size"]): r["direction"] for r in rows} == {
        ("v2", 256): "ICL side", ("fac", 256): "our side",
        ("v2", 1000): "no directional statement", ("fac", 1000): "no directional statement",
    }


class CostlyClient(SyntheticClient):
    def __init__(self, fail_tenth=False):
        super().__init__()
        self.calls = 0
        self.fail_tenth = fail_tenth

    def create(self, **body):
        self.calls += 1
        response = super().create(**body)
        response.usage = SimpleNamespace(model_dump=lambda **_: {"input_tokens": 100, "output_tokens": 10000})
        if self.fail_tenth and self.calls == 10:
            response.content[-1].text = "{}"
        return response


def install_closed_session(monkeypatch, histories):
    import adapters.chembench_observation as adapter
    import bayes_model_discovery.experimental.chembench_core as engine

    for history in histories:
        history.config = {"task": {"domain": "synthetic", "difficulty": "synthetic", "law_version": "synthetic"},
                          "noise": .01, "d0_conditions": {}}

    class Session:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def schema(self):
            return {"input_names": VISIBLE_NAMES}

        def observe(self, x):
            self.calls += 1
            return {"observation_index": self.calls, "x": x, "y": float(self.calls)}

        def finish(self):
            pass

    monkeypatch.setattr(adapter, "ChemBenchSession", Session)
    monkeypatch.setattr(engine, "fixed3_designs", lambda _: [dict(zip(VISIBLE_NAMES, o.x)) for o in histories[0].observations[:3]])


@pytest.mark.parametrize("unit_id,cap,expected_status,planned,completed,sent", [
    (FIXED, 40, "complete", 94, 94, 4), (FIXED, 15, "budget_stopped", 94, 10, 4),
    (CLOSED, 65, "complete", 228, 228, 0), (CLOSED, 40, "budget_stopped", 228, 10, 0),
])
def test_official_projection_reduction_counts_and_preserves_sent_bytes(
        tmp_path, monkeypatch, unit_id, cap, expected_status, planned, completed, sent):
    histories = [synthetic_history(tmp_path, g, r) for g in ("v2", "fac") for r in (1, 2, 3)]
    if unit_id == CLOSED:
        install_closed_session(monkeypatch, histories)
    client = CostlyClient()
    unit = Unit(unit_id, tmp_path / "output", histories, mock=True, client=client)
    unit.budget.cap = cap
    before = {}
    reduce = unit.budget.reduce_plan

    def capture_then_reduce():
        for pattern in ("corpus/**/*.json", "transport/*.request.json"):
            before.update({p: p.read_bytes() for p in unit.output.glob(pattern)})
        return reduce()

    unit.budget.reduce_plan = capture_then_reduce
    receipt = unit.run()
    assert receipt["status"] == expected_status and receipt["reduced_official_layer"]
    assert receipt["planned_initial_requests"] == planned
    assert receipt["completed_initial_requests"] == receipt["call_count"] == client.calls == completed
    assert all(path.read_bytes() == data for path, data in before.items())
    assert before
    snapshot = receipt["official_layer_reduction"]
    original = 270 if unit_id == FIXED else 318
    # the same counts by kind; the first ten CLOSED calls are 4 predictions and 6 selections.
    by_kind_before = ({"prediction": 138 - 4, "selection": 180 - 6} if unit_id == CLOSED
                      else {"prediction": original - 10, "selection": 0})
    by_kind_after = {**by_kind_before, "prediction": by_kind_before["prediction"] - (original - planned)}
    assert snapshot == {
        "calls_completed": 10, "completed_initial_requests": 10,
        "planned_before": original, "remaining_before": original - 10,
        "official_sent_requests": sent, "official_completed_requests": sent,
        "official_unsent_requests_removed": original - planned,
        "planned_after": planned, "remaining_after": planned - 10,
        "remaining_by_kind_before": by_kind_before, "remaining_by_kind_after": by_kind_after,
    }
    assert receipt["projection"]["rule"] == "per_kind_max_cost_x1.25_after_10_calls"
    assert receipt["projection"]["remaining_by_kind"] == by_kind_before
    assert receipt["projection"]["remaining_by_kind_after_reduction"] == by_kind_after
    assert receipt["projection"]["exceeds_cap"] and receipt["projection"]["exceeds_cap_after_reduction"] == (
        expected_status == "budget_stopped")
    by_id = {r["request_id"]: r for r in unit.requests}
    assert sum(by_id[c["request_id"]].get("evaluation_size") == 1000 for c in unit.budget.calls) == sent
    assert receipt["spent_usd"] <= cap and receipt["paid_api_calls"] == 0
    scores = json.loads((unit.output / "scores.json").read_text())
    assert all(r["evaluation_size"] == 256 for r in scores["cells"])
    if unit_id == FIXED and expected_status == "complete":
        assert receipt["incomplete_official_timepoints"][0]["completed_batches"] == 4
        skipped = [json.loads(p.read_text()) for p in (unit.output / "responses").glob("*.json")
                   if json.loads(p.read_text()).get("status") == "not_sent_official_layer_reduction"]
        assert len(skipped) == 1 and skipped[0]["attempt_statuses"] == []
    if unit_id == CLOSED and expected_status == "complete":
        assert sum(c["kind"] == "selection" for c in unit.budget.calls) == 180


def test_reduction_during_retry_keeps_already_sent_official_request(tmp_path):
    histories = [synthetic_history(tmp_path, g, r) for g in ("v2", "fac") for r in (1, 2, 3)]
    unit = Unit(FIXED, tmp_path / "retry", histories, mock=True, client=CostlyClient(fail_tenth=True))
    unit.budget.cap = 40
    receipt = unit.run()
    assert receipt["status"] == "complete"
    assert receipt["planned_initial_requests"] == receipt["completed_initial_requests"] == 94
    assert receipt["call_count"] == 95
    snapshot = receipt["official_layer_reduction"]
    assert snapshot["completed_initial_requests"] == 9
    assert snapshot["official_completed_requests"] == 3 and snapshot["official_sent_requests"] == 4
    assert snapshot["remaining_after"] == 85
    a, b = unit.budget.calls[9:11]
    assert a["request_id"] == b["request_id"] and (a["attempt"], b["attempt"]) == (0, 1)
    assert (a["status"], b["status"]) == ("retryable_validation_failure", "valid")
    assert receipt["incomplete_official_timepoints"][0]["completed_batches"] == 4


@pytest.mark.parametrize("unit_id", [FIXED, CLOSED, VISIBLE])
def test_no_projection_request_and_science_bytes_unchanged_by_reduction_hook(tmp_path, monkeypatch, unit_id):
    groups = ("v2",) if unit_id == VISIBLE else ("v2", "fac")
    histories = [synthetic_history(tmp_path, g, r) for g in groups for r in (1, 2, 3)]
    if unit_id == CLOSED:
        install_closed_session(monkeypatch, histories)
    enabled = Unit(unit_id, tmp_path / "enabled", histories, mock=True, client=SyntheticClient())
    disabled = Unit(unit_id, tmp_path / "disabled", histories, mock=True, client=SyntheticClient())
    disabled.budget.reduce_plan = None
    a, b = enabled.run(), disabled.run()
    assert a["status"] == b["status"] == "complete"
    assert a["planned_initial_requests"] == b["planned_initial_requests"]
    assert not a["reduced_official_layer"] and a["official_layer_reduction"] is None
    for pattern in ("corpus/**/*.json", "transport/*.request.json", "responses/*.json", "scores.json", "closed_rep*.json"):
        files = {p.relative_to(enabled.output): p.read_bytes() for p in enabled.output.glob(pattern)}
        other = {p.relative_to(disabled.output): p.read_bytes() for p in disabled.output.glob(pattern)}
        assert files == other


@pytest.mark.parametrize("unit_id,cap", [(FIXED, 90.0), (CLOSED, 90.0), (VISIBLE, 25.0)])
def test_approved_cap_bound_to_budget_settings_and_receipts(tmp_path, unit_id, cap):
    unit = Unit(unit_id, tmp_path / "cap", [], mock=True, client=SyntheticClient())
    unit.planned = 0
    unit.run_closed = lambda: None
    assert settings(unit_id)["cap_usd"] == unit.budget.cap == cap
    receipt = unit.run()
    start = json.loads((unit.output / "starting_receipt.json").read_text())
    assert start["settings"]["cap_usd"] == start["starting_balance_usd"] == cap
    assert receipt["settings"]["cap_usd"] == receipt["starting_balance_usd"] == receipt["remaining_usd"] == cap
    assert receipt["paid_api_calls"] == 0
