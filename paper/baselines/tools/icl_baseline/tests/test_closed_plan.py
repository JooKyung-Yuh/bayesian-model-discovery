"""the CLOSED plan options (--rounds, --checkpoints, --official-checkpoints, --cap-usd). The oracle is the
real zero-cost simulator and the client is the tool's fake one; no provider call. No published response is a fixture."""

import json

import pytest

import tools.icl_baseline.icl_baseline as module
from tools.icl_baseline import transport
from tools.icl_baseline.icl_baseline import CLOSED, FIXED, TaskSource, Unit
from tools.icl_baseline.protocol import OFFICIAL_TIMES, ROOT, TIMES

pytestmark = pytest.mark.usefixtures("require_autoscilab_source")

TASK = "c0_michaelis_menten__easy__v0"
PRICES = ROOT / "prices.json"
EVERY = tuple(range(3, 34))
EVERY_FLAGS = ["--rounds", "30", "--checkpoints", ",".join(str(x) for x in EVERY), "--official-checkpoints", "none",
               "--cap-usd", "200"]


def corpus(folder):
    rows = [json.loads(p.read_text()) for p in (folder / "corpus").rglob("*.json")]
    return sorted(rows, key=lambda r: r["request_id"])


def test_the_released_plan_is_the_default_and_leaves_the_settings_as_released():
    plan = module.closed_plan()
    assert plan == module.RELEASED_PLAN == {"rounds": 60, "checkpoints": TIMES, "official_checkpoints": OFFICIAL_TIMES,
                                            "cap_usd": 90.0}
    for unit in (CLOSED, FIXED):
        assert module.plan_settings(unit, 32000, plan) == module.plan_settings(unit, 32000) == transport.settings(unit, 32000)
    args = module.parser().parse_args(["--unit", CLOSED, "--output", "o", "--dry-run", "--registration", "r"])
    assert (args.rounds, args.checkpoints, args.official_checkpoints, args.cap_usd) == (60, TIMES, OFFICIAL_TIMES, None)
    stated = module.closed_plan(30, EVERY, (), 200)
    assert module.plan_settings(CLOSED, 32000, stated) == {
        **transport.settings(CLOSED, 32000), "cap_usd": 200.0, "rounds": 30, "checkpoints": list(EVERY),
        "official_checkpoints": []}
    assert module.integer_list("none") == () and module.integer_list("3,4") == (3, 4)


@pytest.mark.parametrize("flags", [
    ["--rounds", "61", "--checkpoints", "3,64"], ["--rounds", "0", "--checkpoints", "3"],
    ["--rounds", "30"],                                                    # the released checkpoints end at 63
    ["--rounds", "30", "--checkpoints", "4,33"], ["--rounds", "30", "--checkpoints", "3,20,10,33"],
    ["--rounds", "30", "--checkpoints", "3,33", "--official-checkpoints", "10"],   # 10 is not a forecast checkpoint
    ["--rounds", "30", "--checkpoints", "3,10,25,33", "--official-checkpoints", "25,10"],
    ["--cap-usd", "0"], ["--cap-usd", "400.5"], ["--cap-usd", "nan"], ["--checkpoints", "3,x"],
])
def test_a_bad_plan_is_refused_before_any_output(tmp_path, monkeypatch, flags):
    monkeypatch.setattr(module, "live_client", lambda *a, **k: pytest.fail("client must not be constructed"))
    output = tmp_path / "o"
    with pytest.raises(SystemExit) as error:
        module.main(["--unit", CLOSED, "--dry-run", "--output", str(output), "--task", TASK, "--registration", str(PRICES), *flags])
    assert error.value.code == 2 and not output.exists()


def test_only_closed_takes_another_plan(tmp_path):
    with pytest.raises(ValueError, match="only CLOSED"):
        Unit(FIXED, tmp_path / "f", [TaskSource(TASK, 1)], dry_run=True, plan=module.closed_plan(30, EVERY, (), 200))


def test_an_every_round_dry_run_forecasts_after_every_observation_and_keeps_the_released_requests(tmp_path, capsys):
    released, every = tmp_path / "released", tmp_path / "every"
    assert module.main(["--unit", CLOSED, "--dry-run", "--output", str(released), "--task", TASK,
                        "--registration", str(PRICES)]) == 0
    assert module.main(["--unit", CLOSED, "--dry-run", "--output", str(every), "--task", TASK,
                        "--registration", str(PRICES), *EVERY_FLAGS]) == 0
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert printed == [{"status": "dry_run_complete", "requests": 318, "paid_api_calls": 0},
                       {"status": "dry_run_complete", "requests": 276, "paid_api_calls": 0}]
    receipt = json.loads((every / "receipt.json").read_text())
    plan = module.closed_plan(30, EVERY, (), 200)
    assert receipt["planned_initial_requests_by_kind"] == receipt["completed_initial_requests_by_kind"] == {
        "prediction": 186, "selection": 90}
    assert receipt["settings"] == module.plan_settings(CLOSED, 32000, plan) and receipt["starting_balance_usd"] == 200.0
    assert json.loads((released / "receipt.json").read_text())["settings"] == transport.settings(CLOSED, 32000)
    cells = json.loads((every / "scores.json").read_text())["cells"]
    assert sorted({(c["n"], c["evaluation_size"]) for c in cells}) == [(n, 256) for n in EVERY]
    for replicate in (1, 2, 3):
        record = json.loads((every / f"closed_rep{replicate}.json").read_text())
        assert len(record["observations"]) == 33 and record["completed_selection_count"] == 30
    # Forecasts are side queries: up to round 30, without the added forecasts and the official layer, the requests
    # are the released run's.
    kept = {3, 5, 10, 15, 20, 25}
    def first_thirty(rows, official):
        return [r["body"] for r in rows if (r["round"] <= 30 if r["request_id"].endswith("selection")
                                            else r["n"] in kept and (official or r["evaluation_size"] == 256))]
    assert first_thirty(corpus(every), True) == first_thirty(corpus(released), False)
    draft = json.loads((every / "seal_draft.json").read_text())
    assert draft["d0_prediction_requests"] == json.loads((released / "seal_draft.json").read_text())["d0_prediction_requests"]


def test_a_stated_plan_seal_opens_only_that_plan(tmp_path):
    plan = module.closed_plan(30, EVERY, (), 200)
    histories = [TaskSource(TASK, replicate) for replicate in (1, 2, 3)]
    dependencies = module.dependencies()
    draft = module.seal_draft(CLOSED, 32000, histories, PRICES, dependencies, plan)
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps({**draft, "status": "sealed"}))
    required = module.input_files(histories) + dependencies + [PRICES]

    def factory(**kwargs):
        return module.SyntheticClient()

    assert isinstance(transport.live_client(True, seal, module.plan_settings(CLOSED, 32000, plan), required, factory),
                      module.SyntheticClient)
    with pytest.raises(ValueError, match="exactly matching settings"):
        transport.live_client(True, seal, transport.settings(CLOSED, 32000), required, factory)


def test_the_v6_driver_states_the_registration_and_its_seal_helper_seals_only_that_plan(tmp_path, capsys):
    from tools.icl_baseline.v6 import run_v6_task, seal_v6
    args = run_v6_task.arguments("dry", TASK, tmp_path / "x")
    assert args[args.index("--checkpoints") + 1] == ",".join(str(x) for x in EVERY) and args[-1] == "--dry-run"
    assert [args[args.index(flag) + 1] for flag in ("--rounds", "--official-checkpoints", "--cap-usd", "--registration")] == [
        "30", "none", "200", str(run_v6_task.REGISTRATION)]
    assert run_v6_task.arguments("paid", TASK, "o", "seal.json")[-3:] == ["--allow-paid-api", "--seal-manifest", "seal.json"]
    for bad in (("paid", TASK, "o"), ("dry", "c9_unknown__easy__v0", "o"), ("dry", TASK, "o", "seal.json")):
        with pytest.raises(SystemExit):
            run_v6_task.arguments(*bad)
    output = tmp_path / "every"
    assert module.main(run_v6_task.arguments("dry", TASK, output)) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "dry_run_complete", "requests": 276, "paid_api_calls": 0}
    summary = seal_v6.seal(TASK, output, tmp_path / "seal.json")
    assert summary["verify_seal"] == "passed" and summary["checkpoints"] == list(EVERY) and summary["cap_usd"] == 200.0
    assert seal_v6.seal(TASK, "build", tmp_path / "built.json")["verify_seal"] == "passed"
    assert json.loads((tmp_path / "built.json").read_text()) == json.loads((tmp_path / "seal.json").read_text())
    with pytest.raises(SystemExit, match="not this task's draft"):
        seal_v6.seal("c2_product_inhibition__easy__v0", output, tmp_path / "other.json")
