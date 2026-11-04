"""BoxingGym ICL (+LLM design). Synthetic observations and clients, the tasks' own
zero-cost environments; no provider call, no published response as a fixture."""

import json
import math
import re
from types import SimpleNamespace

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("boxing_gym") is None:
    pytest.skip("needs the BoxingGym checkout on the path (BMD_BOXINGGYM_SRC)", allow_module_level=True)

from bayes_model_discovery.experimental import boxinggym_core as bg  # noqa: E402
from bayes_model_discovery.experimental.core_count import seed_for
from tools.icl_boxinggym import icl_boxinggym as module
from tools.icl_boxinggym import protocol
from tools.icl_boxinggym.protocol import (
    CHECKPOINTS,
    TASKS,
    Observation,
    brier,
    fallback_value,
    forbidden_words,
    interface_for,
    nmse_truth,
    prediction_result,
    render_prediction,
    render_selection,
    validate_prediction,
    validate_selection,
)

NATIVE = {"death": ("t",), "dugongs": ("age",), "peregrines": ("tt",), "hyperbolic": ("iR", "dR", "D"),
          "survival": ("time", "metastasized")}


def observations(k, n=4):
    return [Observation(i, tuple(float(i + 1 + j) for j in range(k)), float(2 * i + 1)) for i in range(n)]


@pytest.mark.parametrize("task", TASKS)
def test_the_interface_masks_the_declared_inputs_in_order_and_knows_the_outcome(task):
    spec = bg.TASKS[task]
    interface = interface_for(spec)
    assert interface.native_names == NATIVE[task] == tuple(spec.input_names)
    assert interface.names == tuple(f"X{i}" for i in range(1, len(NATIVE[task]) + 1))
    assert interface.input_name_map == dict(zip(interface.names, NATIVE[task]))
    assert interface.bounds == tuple(tuple(float(v) for v in item["range"]) for item in spec.inputs)
    assert interface.outcome_type == spec.outcome_type
    low, high = interface.support()
    expected = {"death": (0.0, 50.0), "peregrines": (0.0, None), "dugongs": ("open_zero", None),
                "hyperbolic": (0.0, 1.0), "survival": (0.0, 1.0)}[task]
    assert (low, high) == expected


@pytest.mark.parametrize("task", TASKS)
def test_rendered_requests_carry_no_task_word_and_no_native_name(task):
    interface = interface_for(bg.TASKS[task])
    k = len(interface.names)
    rows = observations(k)
    prompt, mapping = render_prediction(interface, rows, [[0.5 * (j + 1) for j in range(k)] for _ in range(3)])
    text = prompt["system"] + "\n" + prompt["messages"][0]["content"]
    assert forbidden_words(text, task) == [] and "<" not in text.replace("<number>", "")
    assert [row["id"] for row in mapping] == ["q1", "q2", "q3"] and "New inputs (3):" in text
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    assert not tokens & set(NATIVE[task])
    assert ("X1 is a variable name" in text) == (k == 1) and ("are variable names" in text) == (k > 1)
    selection, ids = render_selection(interface, rows, np.array([[1.0] * k, [2.0] * k]), repeats_allowed=True)
    text = selection["system"] + "\n" + selection["messages"][0]["content"]
    assert forbidden_words(text, task) == [] and "<" not in text.replace("<candidate id>", "")
    assert [row["id"] for row in ids] == ["c1", "c2"] and "Candidate experiments (2)" in text
    assert "You may repeat an earlier experiment." in text
    excluded, _ = render_selection(interface, rows, np.array([[1.0] * k]), repeats_allowed=False)
    assert "Experiments already run are not listed." in excluded["messages"][0]["content"]


@pytest.mark.parametrize(("task", "bad", "good"), [
    ("death", [60, -1, "x"], [0, 50, 12.5]), ("peregrines", [-0.5, float("nan")], [0, 1e6]),
    ("dugongs", [0, -1], [1e-9, 3.2]), ("hyperbolic", [1.5, -0.1], [0, 1, 0.5]), ("survival", [2], [0.25]),
])
def test_prediction_values_are_checked_against_the_outcome_support(task, bad, good):
    interface = interface_for(bg.TASKS[task])
    for value in bad:
        text = ('{"predictions": {"q1": NaN}}' if isinstance(value, float) and math.isnan(value)
                else json.dumps({"predictions": {"q1": value}}))
        result = validate_prediction(text, interface, ("q1",))
        assert not result.valid and result.errors[0].startswith("q1:")
    for value in good:
        assert validate_prediction(json.dumps({"predictions": {"q1": value}}), interface, ("q1",)).values == {"q1": float(value)}
    assert validate_prediction('{"predictions": {"q1": 1, "q1": 2}}', interface, ("q1",)).errors == ["q1: expected exactly once"]
    assert validate_prediction('{"predictions": {"q1": 1, "q9": 2}}', interface, ("q1",)).errors == ["unexpected keys"]
    assert validate_prediction("", interface, ("q1",)).errors == ["empty text"]
    assert validate_prediction('{"predictions": {"q1": 1}}', interface, ("q1",), stop_reason="max_tokens").errors == ["response stopped at max_tokens"]
    assert validate_prediction("[1]", interface, ("q1",)).errors == ["invalid JSON object"]


def test_selection_replies_name_one_listed_candidate():
    assert validate_selection('{"choice": "c2"}', ("c1", "c2")).values == {"choice": "c2"}
    assert validate_selection('{"choice": "c9"}', ("c1", "c2")).errors == ["choice must be one listed candidate id"]
    assert validate_selection('{"choice": 2}', ("c1", "c2")).errors == ["choice must be one listed candidate id"]
    assert validate_selection('{"x": "c1"}', ("c1",)).errors == ["root must contain only one choice string"]
    assert validate_selection('```json\n{"choice": "c1"}\n```', ("c1",)).valid


def test_fallbacks_are_the_median_or_for_a_binary_outcome_the_mean():
    rows = [Observation(i, (float(i),), y) for i, y in enumerate([0.0, 0.0, 1.0, 1.0, 1.0])]
    assert fallback_value(interface_for(bg.TASKS["survival"]), rows) == 0.6
    assert fallback_value(interface_for(bg.TASKS["dugongs"]), rows) == 1.0
    result = prediction_result([protocol.Validation({"q1": 2.0}, [])], ("q1", "q2"), interface_for(bg.TASKS["dugongs"]), rows)
    assert result == {"values": [2.0, 1.0], "valid": [True, False], "fallback_ids": ["q2"],
                      "status": "prediction_fallback", "fallback_value": 1.0}


def test_scores_are_the_engines():
    rng = np.random.default_rng(3)
    truth = rng.uniform(1, 5, size=40)
    prediction = truth + rng.normal(0, 0.3, size=40)
    expected = bg.mda_score_row(7, prediction, truth, None)["nmse_truth"]
    assert nmse_truth(prediction, truth) == expected
    assert nmse_truth(prediction, np.full(40, 2.0)) is None and nmse_truth(prediction, None) is None
    p = rng.uniform(0, 1, size=40)
    y = (rng.uniform(size=40) < 0.5).astype(float)
    assert brier(p, y) == float(np.mean((p - y) ** 2))
    with pytest.raises(ValueError):
        nmse_truth(prediction[:10], truth)


def test_batches_deduplicate_repeated_evaluation_rows():
    rows = np.array([[1.0], [2.0], [1.0], [3.0], [2.0]])
    world = SimpleNamespace(evaluation_rows=rows, unobserved_mask=lambda: np.array([True, True, True, False, True]))
    assert module.batches_of(world, []) == [(0, [(1.0,), (2.0,)])]
    big = SimpleNamespace(evaluation_rows=np.arange(300.0)[:, None], unobserved_mask=lambda: np.ones(300, dtype=bool))
    plan = module.batches_of(big, [])
    assert [offset for offset, _ in plan] == [0, 128, 256] and [len(b) for _, b in plan] == [128, 128, 44]


def test_the_world_is_the_engines_start_pool_and_evaluation_set():
    # The engine's own calls, in the engine's order, on a fresh global numpy stream; the World is
    # built afterwards and must reproduce them (its construction reseeds the stream the same way).
    task, seed = "death", 31
    spec = bg.TASKS[task]
    world_seed = seed_for("boxinggym_world", task, seed)
    with bg.env_seed_injection(world_seed):
        environment = bg.TaskEnvironment(spec, world_seed)
        x, y = environment.evaluation_points(256, seed_for("boxinggym_evaluation", task, seed, "paper"))
        pool = bg.build_design_pool(spec, environment, bg.design_pool_target(spec, 100, 20))
        d0 = bg.select_d0_designs(spec, pool, "random2", seed)
        truth = spec.truth_mean(environment, x)
        start_y = [environment.observe(h) for _, h in d0]
    world = module.World(task, seed)
    try:
        np.testing.assert_array_equal(world.evaluation_x, x)
        np.testing.assert_array_equal(world.heldout, y)
        np.testing.assert_array_equal(world.pool.rows, pool.rows)
        assert [h for _, h in world.d0] == [h for _, h in d0]
        np.testing.assert_array_equal(world.truth, truth)
        started = world.start()
        assert [o.x for o in started] == [tuple(map(float, np.atleast_1d(r))) for r, _ in d0]
        assert [o.y for o in started] == start_y
        assert world.candidate_indices() == list(range(100)) and world.unobserved_mask().sum() == 256
        assert world.observe_index(5).x == tuple(map(float, np.atleast_1d(pool.rows[5])))
        assert world.observed_indices == [pool.index_of(r) for r, _ in d0] + [5]
    finally:
        world.close()


def test_synthetic_client_answers_every_query_and_a_listed_candidate():
    interface = interface_for(bg.TASKS["hyperbolic"])
    rows = observations(3)
    prompt, mapping = render_prediction(interface, rows, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    reply = json.loads(module.SyntheticClient().create(**module.request_body(prompt)).content[-1].text)
    assert reply == {"predictions": {"q1": 0.5, "q2": 0.5}}
    prompt, mapping = render_selection(interface, rows, np.array([[1.0, 2.0, 3.0]] * 7), repeats_allowed=True)
    reply = json.loads(module.SyntheticClient().create(**module.request_body(prompt)).content[-1].text)
    assert reply["choice"] in {row["id"] for row in mapping}


def test_a_dry_run_of_death_sends_eighty_four_requests_that_leak_nothing(tmp_path, capsys):
    output = tmp_path / "death"
    assert module.main(["--task", "death", "--dry-run", "--output", str(output),
                        "--registration", str(module.ROOT.parent / "icl_baseline" / "prices.json")]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "dry_run_complete", "requests": 84, "paid_api_calls": 0}
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["task"] == "death" and receipt["settings"] == module.settings("death")
    assert receipt["completed_initial_requests_by_kind"] == {"prediction": 24, "selection": 60}
    assert receipt["planned_initial_requests_by_kind_upper_bound"] == {"prediction": 24, "selection": 60}
    assert receipt["projection"]["rule"] == "per_kind_max_cost_x1.25_after_10_calls" and receipt["stop_reason"] is None
    assert all(row["status"] == "valid" for row in receipt["calls"]) and receipt["paid_api_calls"] == 0
    texts = []
    for path in sorted([*(output / "corpus").glob("*.json"), *(output / "transport").glob("*.request.json")]):
        body = json.loads(path.read_text())["body"]
        assert set(body) == {"model", "max_tokens", "system", "messages", "output_config", "thinking"}
        texts.append(body["system"] + "\n" + "\n".join(m["content"] for m in body["messages"]))
    assert len(texts) == 2 * 84
    for text in texts:
        assert forbidden_words(text, "death") == []
        assert not re.search(r"\bt\b|\btheta\b|\bN\b", text.split("Observed experiments", 1)[1] if "Observed experiments" in text else text)
    scores = json.loads((output / "scores.json").read_text())
    assert [(c["replicate"], c["x"]) for c in scores["cells"]] == [(r, x) for r in (1, 2, 3) for x in CHECKPOINTS]
    assert all(c["q_scored"] == 256 and c["batches"] == 2 and c["brier"] is None for c in scores["cells"])
    points = (output / "points_icl.csv").read_text().splitlines()
    assert points[0] == "replicate,x,nmse_truth,brier,best_so_far_nmse_truth" and len(points) == 13
    for replicate, seed in zip((1, 2, 3), module.SEEDS):
        record = json.loads((output / f"closed_rep{replicate}.json").read_text())
        spec = bg.TASKS["death"]
        world_seed = seed_for("boxinggym_world", "death", seed)
        with bg.env_seed_injection(world_seed):
            environment = bg.TaskEnvironment(spec, world_seed)
            environment.evaluation_points(256, seed_for("boxinggym_evaluation", "death", seed, "paper"))
            pool = bg.build_design_pool(spec, environment, bg.design_pool_target(spec, 100, 20))
            d0 = bg.select_d0_designs(spec, pool, "random2", seed)
        assert record["d0_designs"] == [list(map(float, np.atleast_1d(r))) for r, _ in d0]
        assert len(record["observations"]) == 22 and len(record["selections"]) == 20
        assert record["input_name_map"] == {"X1": "t"} and record["provenance"] == "SYNTHETIC"
    draft = json.loads((output / "seal_draft.json").read_text())
    assert draft["task"] == "death" and len(draft["d0_prediction_requests"]) == 6
    assert draft["settings"] == module.settings("death") and str(module.ENGINE) in draft["files"]


def test_cli_refusals(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "live_client", lambda *a, **k: pytest.fail("client must not be constructed"))
    with pytest.raises(SystemExit) as error:
        module.main(["--task", "irt", "--dry-run", "--output", str(tmp_path / "a"), "--registration", "r"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        module.main(["--task", "death", "--allow-paid-api", "--output", str(tmp_path / "b"),
                     "--registration", str(module.ROOT.parent / "icl_baseline" / "prices.json")])
    assert error.value.code == 2 and not (tmp_path / "b").exists()
    with pytest.raises(ValueError, match="verified client"):
        module.Unit("death", tmp_path / "c")


def test_the_transport_counts_the_tasks_own_domain_words(tmp_path):
    transport = module.BoxingTransport(module.SyntheticClient(), tmp_path, module.Budget(1.0, module.load_prices()), task="dugongs")
    assert transport.domain_words.search("a dugong grows") and not transport.domain_words.search("an enzyme")


def sealed_from(draft, path):
    seal = {**draft, "status": "sealed"}
    path.write_text(json.dumps(seal))
    return seal


def test_the_draft_binds_every_file_the_shared_verifier_requires_and_a_sealed_draft_passes(tmp_path):
    from tools.icl_baseline import transport
    registration = module.ROOT.parent / "icl_baseline" / "prices.json"
    draft = module.seal_draft("death", 32000, registration)
    shared = {str(p.resolve()) for p in transport.ROOT.rglob("*")
              if p.is_file() and p.suffix in (".py", ".txt", ".json") and "tests" not in p.parts}
    assert shared <= set(draft["files"]) and len(shared) >= 8
    assert {str(p) for p in module.dependencies()} <= set(draft["files"])
    seal = tmp_path / "seal.json"
    sealed_from(draft, seal)
    factory = lambda **kwargs: module.SyntheticClient()
    client = transport.live_client(True, seal, module.settings("death"), [*module.dependencies(), registration], factory)
    assert isinstance(client, module.SyntheticClient)
    # One missing shared file (the ChemBench driver, which this tool never imports) is refused.
    short = {**draft, "files": {k: v for k, v in draft["files"].items() if not k.endswith("icl_baseline/icl_baseline.py")}}
    sealed_from(short, tmp_path / "short.json")
    with pytest.raises(ValueError, match="seal is missing"):
        transport.live_client(True, tmp_path / "short.json", module.settings("death"), [*module.dependencies(), registration], factory)
    with pytest.raises(ValueError, match="exactly matching settings"):
        transport.live_client(True, seal, module.settings("dugongs"), [*module.dependencies(), registration], factory)


def test_the_live_path_verifies_the_seal_and_runs_to_the_client_boundary(tmp_path, monkeypatch, capsys):
    """`--allow-paid-api --seal-manifest`: the real verifier on a seal made from the draft, then the
    whole run against a synthetic client in place of the provider's."""
    from tools.icl_baseline import transport
    registration = module.ROOT.parent / "icl_baseline" / "prices.json"
    draft = module.seal_draft("death", 32000, registration)
    seal = tmp_path / "seal.json"
    sealed_from(draft, seal)
    calls = []

    def verified(allow, manifest, expected_settings, required_inputs, factory=None):
        calls.append((allow, str(manifest)))
        return transport.live_client(allow, manifest, expected_settings, required_inputs,
                                     factory=lambda **kwargs: module.SyntheticClient())

    monkeypatch.setattr(module, "live_client", verified)
    output = tmp_path / "live"
    assert module.main(["--task", "death", "--allow-paid-api", "--seal-manifest", str(seal), "--output", str(output),
                        "--registration", str(registration)]) == 0
    assert calls == [(True, str(seal))]
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "complete" and receipt["accounting_kind"] == "live"
    assert receipt["completed_initial_requests"] == 84 and receipt["paid_api_calls"] == 84
    assert json.loads(capsys.readouterr().out) == {"status": "complete", "requests": 84, "paid_api_calls": 84}
    # Another task's seal, or a seal without the shared files, never reaches the client.
    sealed_from({**draft, "task": "dugongs"}, tmp_path / "other.json")
    with pytest.raises(SystemExit) as error:
        module.main(["--task", "death", "--allow-paid-api", "--seal-manifest", str(tmp_path / "other.json"),
                     "--output", str(tmp_path / "x"), "--registration", str(registration)])
    assert error.value.code == 2 and not (tmp_path / "x").exists()
    short = {**draft, "files": {k: v for k, v in draft["files"].items() if not k.endswith("icl_baseline/precheck.py")}}
    sealed_from(short, tmp_path / "short.json")
    with pytest.raises(SystemExit) as error:
        module.main(["--task", "death", "--allow-paid-api", "--seal-manifest", str(tmp_path / "short.json"),
                     "--output", str(tmp_path / "y"), "--registration", str(registration)])
    assert error.value.code == 2 and not (tmp_path / "y").exists()


# ---------------------------------------------------------------- budget, checkpoints, worlds
PRICES = module.ROOT.parent / "icl_baseline" / "prices.json"
# A 30-round plan whose checkpoints extend the 2026-10-02 ones (so a 20-round run is a prefix of it); the plan
# registration v6 states (2, 7, 12, 17, 22, 27, 32) is pinned by the driver test at the end.
R30 = {"rounds": 30, "checkpoints": (2, 7, 12, 22, 32)}
R30_FLAGS = ["--rounds", "30", "--checkpoints", "2,7,12,22,32"]


def test_unstated_run_parameters_are_the_2026_10_02_registration():
    assert module.run_parameters() == {"rounds": 20, "checkpoints": (2, 7, 12, 22), "seeds": (31, 37, 41), "cap_usd": 20.0}
    args = module.parser().parse_args(["--task", "death", "--dry-run", "--output", "o", "--registration", "r"])
    assert (args.rounds, args.checkpoints, args.seeds) == (20, (2, 7, 12, 22), (31, 37, 41))
    settings = module.settings("death")
    assert (settings["rounds"], settings["checkpoints"], settings["seeds"]) == (20, [2, 7, 12, 22], [31, 37, 41])
    assert module.settings("death", **module.run_parameters()) == settings
    stated = module.settings("survival", rounds=30, checkpoints=(2, 7, 12, 22, 32), seeds=(53, 61, 43))
    assert {k: v for k, v in stated.items() if k not in ("rounds", "checkpoints", "seeds")} == \
        {k: v for k, v in module.settings("survival").items() if k not in ("rounds", "checkpoints", "seeds")}
    assert (stated["rounds"], stated["checkpoints"], stated["seeds"]) == (30, [2, 7, 12, 22, 32], [53, 61, 43])


@pytest.mark.parametrize("flags", [
    ["--rounds", "31", "--checkpoints", "2,33"], ["--rounds", "0", "--checkpoints", "2"],
    ["--rounds", "30"],                                   # the last checkpoint 22 would leave rounds 21..30 unscored
    ["--rounds", "30", "--checkpoints", "7,12,22,32"],    # no forecast at the start
    ["--rounds", "30", "--checkpoints", "2,12,7,32"], ["--rounds", "30", "--checkpoints", "2,7,7,32"],
    ["--seeds", "31,31,41"], ["--seeds=-1,37"], ["--checkpoints", "2,x"],
    ["--cap-usd", "0"], ["--cap-usd", "100.5"], ["--cap-usd", "nan"],
])
def test_a_plan_with_an_unscored_round_or_a_bad_seed_is_refused_before_any_output(tmp_path, monkeypatch, flags):
    monkeypatch.setattr(module, "live_client", lambda *a, **k: pytest.fail("client must not be constructed"))
    monkeypatch.setattr(module, "seal_draft", lambda *a, **k: pytest.fail("no world is built for a refused plan"))
    with pytest.raises(SystemExit) as error:
        module.main(["--task", "death", "--dry-run", "--output", str(tmp_path / "o"), "--registration", str(PRICES), *flags])
    assert error.value.code == 2 and not (tmp_path / "o").exists()


def test_thirty_rounds_leave_every_paper_pool_as_it_was():
    # The engine sizes a pool with the run's --rounds; at the paper grid no task's pool changes.
    for task in TASKS:
        spec = bg.TASKS[task]
        assert bg.design_pool_target(spec, 100, 30) == bg.design_pool_target(spec, 100, 20) == 100


def corpus_bodies(folder):
    rows = [json.loads(path.read_text()) for path in sorted((folder / "corpus").glob("*.json"))]
    return {r: [row["body"] for row in rows if row["replicate"] == r] for r in (1, 2, 3)}


def test_thirty_rounds_append_ten_selections_and_one_forecast_to_each_twenty_round_replicate(tmp_path, capsys):
    short, long = tmp_path / "r20", tmp_path / "r30"
    assert module.main(["--task", "death", "--dry-run", "--output", str(short), "--registration", str(PRICES)]) == 0
    assert module.main(["--task", "death", "--dry-run", "--output", str(long), "--registration", str(PRICES), *R30_FLAGS]) == 0
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert printed == [{"status": "dry_run_complete", "requests": 84, "paid_api_calls": 0},
                       {"status": "dry_run_complete", "requests": 120, "paid_api_calls": 0}]
    receipt = json.loads((long / "receipt.json").read_text())
    assert receipt["settings"] == module.settings("death", **R30)
    assert json.loads((long / "starting_receipt.json").read_text())["settings"] == module.settings("death", **R30)
    assert receipt["completed_initial_requests_by_kind"] == {"prediction": 30, "selection": 90}
    assert receipt["planned_initial_requests_by_kind_upper_bound"] == {"prediction": 30, "selection": 90}
    scores = json.loads((long / "scores.json").read_text())
    assert scores["checkpoints"] == [2, 7, 12, 22, 32]
    assert [(c["replicate"], c["x"]) for c in scores["cells"]] == [(r, x) for r in (1, 2, 3) for x in (2, 7, 12, 22, 32)]
    assert len((long / "points_icl.csv").read_text().splitlines()) == 16
    draft = json.loads((long / "seal_draft.json").read_text())
    assert draft["settings"] == module.settings("death", **R30) and draft["seeds"] == [31, 37, 41]
    assert draft["d0_prediction_requests"] == json.loads((short / "seal_draft.json").read_text())["d0_prediction_requests"]
    # Each replicate's first 28 requests (8 forecasts at x = 2..22 and 20 selections) are the 20-round run's.
    before, after = corpus_bodies(short), corpus_bodies(long)
    for replicate in (1, 2, 3):
        assert len(before[replicate]) == 28 and len(after[replicate]) == 40
        assert after[replicate][:28] == before[replicate]
        record = json.loads((long / f"closed_rep{replicate}.json").read_text())
        assert len(record["observations"]) == 32 and len(record["selections"]) == 30
        assert record["selections"][:20] == json.loads((short / f"closed_rep{replicate}.json").read_text())["selections"]


def test_stated_seeds_are_the_engines_worlds_and_reach_every_record(tmp_path, capsys):
    output = tmp_path / "seeds"
    assert module.main(["--task", "death", "--dry-run", "--output", str(output), "--registration", str(PRICES),
                        "--seeds", "41,31", *R30_FLAGS]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "dry_run_complete", "requests": 80, "paid_api_calls": 0}
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["settings"]["seeds"] == [41, 31]
    assert receipt["completed_initial_requests_by_kind"] == {"prediction": 20, "selection": 60}
    draft = json.loads((output / "seal_draft.json").read_text())
    assert draft["seeds"] == [41, 31] and [r["replicate"] for r in draft["d0_prediction_requests"]] == [1, 1, 2, 2]
    spec = bg.TASKS["death"]
    for replicate, seed in ((1, 41), (2, 31)):
        world_seed = seed_for("boxinggym_world", "death", seed)
        with bg.env_seed_injection(world_seed):
            environment = bg.TaskEnvironment(spec, world_seed)
            environment.evaluation_points(256, seed_for("boxinggym_evaluation", "death", seed, "paper"))
            pool = bg.build_design_pool(spec, environment, bg.design_pool_target(spec, 100, 30))
            d0 = bg.select_d0_designs(spec, pool, "random2", seed)
        record = json.loads((output / f"closed_rep{replicate}.json").read_text())
        assert record["seed"] == seed and record["d0_designs"] == [list(map(float, np.atleast_1d(r))) for r, _ in d0]
    assert json.loads((output / "scores.json").read_text())["replicates"] == [
        {"replicate": 1, "seed": 41, "selection_fallback_count": 0}, {"replicate": 2, "seed": 31, "selection_fallback_count": 0}]


def test_a_thirty_round_seal_opens_only_the_thirty_round_plan(tmp_path, monkeypatch, capsys):
    from tools.icl_baseline import transport
    draft = module.seal_draft("death", 32000, PRICES, **module.run_parameters(**R30))
    seal = tmp_path / "v6.json"
    sealed_from(draft, seal)
    required = [*module.dependencies(), PRICES]
    factory = lambda **kwargs: module.SyntheticClient()
    assert isinstance(transport.live_client(True, seal, module.settings("death", **R30), required, factory), module.SyntheticClient)
    for other in (module.settings("death"), module.settings("death", **R30, seeds=(53, 61, 43))):
        with pytest.raises(ValueError, match="exactly matching settings"):
            transport.live_client(True, seal, other, required, factory)

    def verified(allow, manifest, expected_settings, required_inputs, factory=None):
        return transport.live_client(allow, manifest, expected_settings, required_inputs,
                                     factory=lambda **kwargs: module.SyntheticClient())

    monkeypatch.setattr(module, "live_client", verified)
    output = tmp_path / "live"
    assert module.main(["--task", "death", "--allow-paid-api", "--seal-manifest", str(seal), "--output", str(output),
                        "--registration", str(PRICES), *R30_FLAGS]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "complete", "requests": 120, "paid_api_calls": 120}
    assert json.loads((output / "receipt.json").read_text())["accounting_kind"] == "live"
    # The same seal never opens the 20-round plan (no flags) or another world set.
    for flags, name in (([], "x"), ([*R30_FLAGS, "--seeds", "53,61,43"], "y")):
        with pytest.raises(SystemExit) as error:
            module.main(["--task", "death", "--allow-paid-api", "--seal-manifest", str(seal), "--output", str(tmp_path / name),
                         "--registration", str(PRICES), *flags])
        assert error.value.code == 2 and not (tmp_path / name).exists()




