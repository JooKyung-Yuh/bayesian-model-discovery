"""Unit ICL-F: ForceBench ICL (+LLM design). Synthetic clients, the runner's own DiscoverPhysics
worlds (venv_dp, DP_SCIENCEAGENT_DIR); no provider call, no published response as a fixture. A
recorded mock cell of the runner is read at test time when ICLF_REFERENCE_CELL names its out/
folder (never copied into this file)."""

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from tools.icl_forcebench import icl_forcebench as module
from tools.icl_forcebench.protocol import (
    CHECKPOINTS,
    T_OBS,
    WORLDS,
    Context,
    Observation,
    fallback_series,
    forbidden_words,
    nmse,
    prediction_result,
    render_prediction,
    render_selection,
    series_text,
    validate_prediction,
    validate_selection,
)

pytest.importorskip("jax")
if not os.environ.get("DP_SCIENCEAGENT_DIR"):
    pytest.skip("DP_SCIENCEAGENT_DIR names the DiscoverPhysics/ScienceAgent directory", allow_module_level=True)
REGISTRATION = module.ROOT.parent / "icl_baseline" / "prices.json"
CONTEXT = Context(p_range=(0.1, 10.0), times=tuple(0.5 * k for k in range(1, 11)), pos0=(-2.5, -6.0), vel0=(-0.1, 2.0))


def observations(n=3):
    return [Observation(i, (1.0 + i, 2.0 + i), tuple((float(k) + i, -float(k) * 0.5) for k in range(T_OBS)), "d0" if i < 2 else f"round {i - 1}")
            for i in range(n)]


def test_series_render_as_numbers_and_requests_leak_nothing():
    assert series_text([[1.0, -2.0]] * T_OBS) == ",".join(["1.0,-2.0"] * T_OBS)
    with pytest.raises(ValueError):
        series_text([[1.0, 2.0]] * 3)
    rows = observations()
    queries = [("head_1", (1.0, 1.0)), ("shift_4", (4.0, 5.0))]
    prompt, mapping = render_prediction(rows, queries, CONTEXT)
    text = prompt["system"] + "\n" + prompt["messages"][0]["content"]
    assert forbidden_words(text) == [] and not re.search(r"\bp[12]\b", text) and "head_1" not in text and "shift" not in text
    assert [row["id"] for row in mapping] == ["q1", "q2"] and [row["label"] for row in mapping] == ["head_1", "shift_4"]
    assert "Held-out input settings (2):" in text and "q2,4.0,5.0" in text and "t1..t10 = 0.5, 1.0, 1.5" in text
    assert "(u0, v0) = (-2.5, -6.0)" in text and "(-0.1, 2.0) per unit time" in text and "[0.1, 10.0]" in text
    assert "<" not in text.replace("<candidate id>", "")
    assert "1,1.0,2.0,0.0,-0.0,1.0,-0.5," in text  # observation row: id, X2, X3, u1, v1, u2, v2, ...
    selection, ids = render_selection(rows, np.array([[0.5, 9.5], [3.0, 3.0], [7.25, 0.125]]), CONTEXT)
    stext = selection["system"] + "\n" + selection["messages"][0]["content"]
    assert forbidden_words(stext) == [] and [row["id"] for row in ids] == ["c1", "c2", "c3"] and [row["candidate_index"] for row in ids] == [0, 1, 2]
    assert "Candidate input settings (3)" in stext and "c3,7.25,0.125" in stext and "<" not in stext.replace("<candidate id>", "")
    with pytest.raises(ValueError):
        render_selection(rows, np.zeros((0, 2)), CONTEXT)
    with pytest.raises(ValueError):
        render_prediction(rows, [], CONTEXT)


def test_forecast_values_are_ten_pairs_of_finite_numbers():
    good = [[float(k), -float(k)] for k in range(T_OBS)]
    ok = validate_prediction(json.dumps({"predictions": {"q1": good, "q2": good}}), ("q1", "q2"))
    assert ok.valid and ok.values["q1"] == good
    assert validate_prediction(json.dumps({"predictions": {"q1": good[:9], "q2": good}}), ("q1", "q2")).errors == ["q1: requires a list of 10 [u, v] pairs"]
    assert validate_prediction(json.dumps({"predictions": {"q1": [[1.0]] * T_OBS, "q2": good}}), ("q1", "q2")).errors == ["q1: each element must be a pair [u, v]"]
    assert validate_prediction('{"predictions": {"q1": [[1.0, NaN]' + ", [1.0, 1.0]" * 9 + ', "q2": [[1.0, 1.0]' + ", [1.0, 1.0]" * 9 + "]}}", ("q1", "q2")).errors == ["invalid JSON object"]  # NaN is not JSON
    assert validate_prediction(json.dumps({"predictions": {"q1": good, "q2": good, "q3": good}}), ("q1", "q2")).errors == ["unexpected keys"]
    assert validate_prediction('{"predictions": {"q1": ' + json.dumps(good) + ', "q1": ' + json.dumps(good) + "}}", ("q1",)).errors == ["q1: expected exactly once"]
    assert validate_prediction(json.dumps({"predictions": {"q1": good}}), ("q1",), "max_tokens").errors == ["response stopped at max_tokens"]
    assert validate_prediction("", ("q1",)).errors == ["empty text"] and validate_prediction("[1]", ("q1",)).errors == ["invalid JSON object"]
    assert validate_prediction('{"a": 1}', ("q1",)).errors == ["root must contain only one predictions object"]
    assert validate_selection('{"choice": "c2"}', ("c1", "c2")).values == {"choice": "c2"}
    assert validate_selection('```json\n{"choice": "c1"}\n```', ("c1", "c2")).valid
    assert validate_selection('{"choice": "c9"}', ("c1", "c2")).errors == ["choice must be one listed candidate id"]


def test_fallback_is_the_per_value_median():
    rows = observations(3)
    median = fallback_series(rows)
    assert median[0] == [1.0, 0.0] and median[3] == [4.0, -1.5] and len(median) == T_OBS
    result = prediction_result([validate_prediction(json.dumps({"predictions": {"q1": median}}), ("q1", "q2"))], ("q1", "q2"), rows)
    assert result["status"] == "prediction_fallback" and result["fallback_ids"] == ["q2"] and result["values"][1] == median and result["valid"] == [True, False]


def test_nmse_is_the_runners_formula():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(0)
    predicted, truth, var_gt = rng.normal(size=(9, T_OBS, 2)), rng.normal(size=(9, T_OBS, 2)), 4.283
    expected = float(jnp.mean((jnp.asarray(predicted) - jnp.asarray(truth)) ** 2) / var_gt)   # run_gravity.py score_set
    assert nmse(predicted, truth, var_gt) == pytest.approx(expected, rel=1e-12)
    assert nmse(truth, truth, var_gt) == 0.0
    with pytest.raises(ValueError):
        nmse(predicted[:, :5], truth[:, :5], var_gt)


def reference_cell():
    folder = os.environ.get("ICLF_REFERENCE_CELL")
    if not folder or not Path(folder).is_dir():
        pytest.skip("ICLF_REFERENCE_CELL names a recorded runner cell's out/ folder")
    runs = glob.glob(str(Path(folder) / "FB1_RUN_*_v3.json"))
    npz = glob.glob(str(Path(folder) / "FB1_RUN_*_evaluation_only.npz"))
    if len(runs) != 1 or len(npz) != 1:
        pytest.skip("the reference folder holds one run JSON and one evaluation npz")
    return json.loads(Path(runs[0]).read_text()), np.load(npz[0])


def test_the_world_is_the_runners_cell_without_its_inference():
    cell, z = reference_cell()
    world, seed = str(z["world"]), int(cell["seeds"]["run"])
    assert world in WORLDS
    w = module.World(world, seed)
    started = w.start()
    designs = np.asarray(cell["observations"]["designs"], dtype=float)
    y = np.asarray(cell["observations"]["y"], dtype=float)
    assert np.array_equal(w.d0, designs[:2]) and [o.design for o in started] == [tuple(d) for d in designs[:2].tolist()]
    assert np.array_equal(np.asarray([o.series for o in started], dtype=float), y[:2]) and [o.source for o in started] == ["d0", "d0"]
    columns, rows = cell["steps"][0]["eig_rows_columns"], cell["steps"][0]["eig_rows"]
    recorded = np.asarray([[r[columns.index("p1")], r[columns.index("p2")]] for r in rows], dtype=float)
    assert np.array_equal(w.candidates(), recorded) and len(recorded) == 64
    assert np.array_equal(w.truth, np.concatenate([z["TR_H"], z["TR_S"]])) and w.truth.shape == (9, T_OBS, 2)
    assert [d for _l, d in w.queries] == [tuple(map(float, q)) for q in np.concatenate([z["Q_HEAD"], z["Q_SHIFT"]])]
    assert float(w.rg.VAR_GT) == float(z["var_gt"]) and float(w.rg.SIGMA) == float(z["sigma_world"])
    assert 20260906 + (seed - 11) == int(cell["seeds"]["observation"]) and w.context.times == tuple(float(t) for t in z["measurement_times"])
    other = module.World(world, seed + 1)
    assert not np.array_equal(other.d0, w.d0) and np.array_equal(other.truth, w.truth)


def test_the_runner_module_takes_the_world_and_seed_and_refuses_option_overrides(monkeypatch):
    rg = module.load_runner("oscillator", 37)
    assert rg.WORLD == "oscillator" and rg.RUN_SEED == 37 and rg.D0 == 2 and rg.T_EXTRA == 16
    again = module.load_runner("oscillator", 37)
    assert again is rg
    monkeypatch.setenv("FB1_D0", "mda1")
    with pytest.raises(ValueError, match="FB1_D0"):
        module.load_runner("oscillator", 37)
    monkeypatch.delenv("FB1_D0")
    with pytest.raises(ValueError, match="world must be one of"):
        module.load_runner("fractional", 37)
    constants = module.runner_constants("oscillator")
    assert constants["d0_size"] == 2 and constants["n_random_designs"] == 64 and len(constants["q_head"]) == 3 and len(constants["q_shift"]) == 6
    assert len(constants["measurement_times"]) == T_OBS and "dp_git_commit" in constants["runner_input_provenance"]


def test_synthetic_client_answers_every_query_and_a_listed_candidate():
    client = module.SyntheticClient()
    rows = observations()
    prompt, mapping = render_prediction(rows, [("head_1", (1.0, 1.0)), ("head_2", (2.0, 1.0))], CONTEXT)
    reply = client.create(**module.request_body(prompt, 32000))
    result = validate_prediction(reply.content[1].text, tuple(m["id"] for m in mapping))
    assert result.valid and result.values["q1"] == [list(p) for p in rows[0].series]
    selection, ids = render_selection(rows, np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]), CONTEXT)
    choice = validate_selection(client.create(**module.request_body(selection, 32000)).content[1].text, tuple(m["id"] for m in ids))
    assert choice.valid and choice.values["choice"] in {"c1", "c2", "c3", "c4"}


def test_a_dry_run_sends_sixty_three_requests_that_leak_nothing(tmp_path, capsys):
    # an nbody world (0.2 s per simulated design); the field worlds gravity/yukawa take ~5 s each
    output = tmp_path / "x"
    assert module.main(["--world", "extra_dimensions", "--dry-run", "--output", str(output), "--registration", str(REGISTRATION)]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "dry_run_complete", "requests": 63, "paid_api_calls": 0}
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["completed_initial_requests_by_kind"] == receipt["planned_initial_requests_by_kind"] == {"prediction": 15, "selection": 48}
    assert receipt["settings"] == module.settings("extra_dimensions") and all(row["status"] == "valid" for row in receipt["calls"])
    assert receipt["projection"]["rule"] == "per_kind_max_cost_x1.25_after_10_calls" and receipt["stop_reason"] is None
    texts = []
    for path in sorted([*(output / "corpus").glob("*.json"), *(output / "transport").glob("*.request.json")]):
        body = json.loads(path.read_text())["body"]
        assert set(body) == {"model", "max_tokens", "system", "messages", "output_config", "thinking"}
        texts.append(body["system"] + "\n" + "\n".join(m["content"] for m in body["messages"]))
    assert len(texts) == 2 * 63 and all(forbidden_words(t) == [] for t in texts) and not any(re.search(r"\bp[12]\b|extra|dimension", t, re.IGNORECASE) for t in texts)
    scores = json.loads((output / "scores.json").read_text())
    assert [(c["replicate"], c["x"]) for c in scores["cells"]] == [(r, x) for r in (1, 2, 3) for x in CHECKPOINTS]
    assert all(c["evaluation_designs"] == 9 and c["nmse_head"] >= 0 and c["nmse_shift"] >= 0 and c["nll"] is None for c in scores["cells"])
    for replicate in (1, 2, 3):
        record = json.loads((output / f"closed_rep{replicate}.json").read_text())
        assert len(record["observations"]) == 18 and [o["source"] for o in record["observations"][:3]] == ["d0", "d0", "round 1"]
        assert len(record["candidate_rounds"]) == 16 and all(len(r) == 64 for r in record["candidate_rounds"])
        assert all(s["design"] == record["candidate_rounds"][s["round"] - 1][s["candidate_index"]] for s in record["selections"])
        assert record["seeds"]["observation"] == 20260906 + (record["seed"] - 11)
    draft = json.loads((output / "seal_draft.json").read_text())
    assert draft["world"] == "extra_dimensions" and len(draft["d0_prediction_requests"]) == 3 and str(module.RUNNER) in draft["files"]
    points = (output / "points_icl.csv").read_text().splitlines()
    assert points[0] == "replicate,x,nmse_head,nmse_shift,nmse_all,nll,coverage90,width90,best_so_far_nmse_head" and len(points) == 16


def sealed_from(draft, path):
    path.write_text(json.dumps({**draft, "status": "sealed"}))


def test_the_live_path_verifies_a_sealed_draft_and_runs_to_the_client_boundary(tmp_path, monkeypatch, capsys):
    from tools.icl_baseline import transport
    draft = module.seal_draft("coulomb_easy", 32000, REGISTRATION)
    shared = {str(p.resolve()) for p in transport.ROOT.rglob("*") if p.is_file() and p.suffix in (".py", ".txt", ".json") and "tests" not in p.parts}
    assert shared <= set(draft["files"]) and {str(p) for p in module.dependencies()} <= set(draft["files"])
    seal = tmp_path / "seal.json"
    sealed_from(draft, seal)
    monkeypatch.setattr(module, "live_client", lambda allow, manifest, expected, required, factory=None: transport.live_client(
        allow, manifest, expected, required, factory=lambda **kwargs: module.SyntheticClient()))
    output = tmp_path / "live"
    assert module.main(["--world", "coulomb_easy", "--allow-paid-api", "--seal-manifest", str(seal), "--output", str(output),
                        "--registration", str(REGISTRATION)]) == 0
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "complete" and receipt["accounting_kind"] == "live" and receipt["paid_api_calls"] == 63
    assert json.loads(capsys.readouterr().out) == {"status": "complete", "requests": 63, "paid_api_calls": 63}
    sealed_from({**draft, "world": "yukawa"}, tmp_path / "other.json")
    with pytest.raises(SystemExit) as error:
        module.main(["--world", "coulomb_easy", "--allow-paid-api", "--seal-manifest", str(tmp_path / "other.json"),
                     "--output", str(tmp_path / "x"), "--registration", str(REGISTRATION)])
    assert error.value.code == 2 and not (tmp_path / "x").exists()
    with pytest.raises(SystemExit) as error:
        module.main(["--world", "coulomb_easy", "--allow-paid-api", "--output", str(tmp_path / "y"), "--registration", str(REGISTRATION)])
    assert error.value.code == 2 and not (tmp_path / "y").exists()
    with pytest.raises(ValueError, match="verified client"):
        module.Unit("coulomb_easy", tmp_path / "c")
    assert module.ForceTransport(module.SyntheticClient(), tmp_path, module.Budget(1.0, module.load_prices()), world="coulomb_easy").domain_words.search("the gravitational force")
    assert re.fullmatch(r"[0-9a-f]{64}", draft["settings"]["engine_sha256"])


# ---------------------------------------------------------------- unit ICL-X: forecasts at every observation count
EVERY = tuple(range(2, 19))
EVERY_FLAGS = ["--checkpoints", ",".join(str(x) for x in EVERY)]


def corpus_bodies(folder, keep):
    """Request bodies in order; forecasts only at the observation counts in `keep`."""
    rows = [json.loads(p.read_text()) for p in sorted((folder / "corpus").glob("*.json"))]
    return [row["body"] for row in rows if row["request_id"].endswith("selection") or row["n"] in keep]


def test_unstated_checkpoints_are_the_released_plan_and_a_bad_plan_is_refused_before_any_output(tmp_path):
    assert module.checkpoint_plan() == CHECKPOINTS == (2, 4, 8, 16, 18)
    args = module.parser().parse_args(["--world", "gravity", "--dry-run", "--output", "o", "--registration", "r"])
    assert args.checkpoints == CHECKPOINTS
    plans = ["3,18", "2,17", "2,8,4,18", "2,4,4,18", "2,x",
             ",".join(str(x) for x in EVERY if x != 4)]                  # drops the runner's own checkpoint 4
    for index, plan in enumerate(plans):
        output = tmp_path / f"o{index}"
        with pytest.raises(SystemExit) as error:
            module.main(["--world", "extra_dimensions", "--dry-run", "--output", str(output), "--registration", str(REGISTRATION),
                         "--checkpoints", plan])
        assert error.value.code == 2 and not output.exists()


def test_an_unstated_cap_is_the_released_15_usd_and_a_bad_cap_is_refused_before_any_output(tmp_path):
    assert module.cap_plan() == module.CAP_USD == 15.0 and module.settings("gravity")["cap_usd"] == 15.0
    args = module.parser().parse_args(["--world", "gravity", "--dry-run", "--output", "o", "--registration", "r"])
    assert args.cap_usd == 15.0
    assert module.settings("gravity", cap_usd=module.cap_plan("16")) == {**module.settings("gravity"), "cap_usd": 16.0}
    for index, cap in enumerate(("0", "100.5", "nan", "inf")):
        output = tmp_path / f"c{index}"
        with pytest.raises(SystemExit) as error:
            module.main(["--world", "extra_dimensions", "--dry-run", "--output", str(output), "--registration", str(REGISTRATION),
                         "--cap-usd", cap])
        assert error.value.code == 2 and not output.exists()


def test_an_every_round_dry_run_adds_forecasts_and_keeps_the_released_requests(tmp_path, capsys):
    released, every = tmp_path / "released", tmp_path / "every"
    assert module.main(["--world", "extra_dimensions", "--dry-run", "--output", str(released), "--registration", str(REGISTRATION)]) == 0
    assert module.main(["--world", "extra_dimensions", "--dry-run", "--output", str(every), "--registration", str(REGISTRATION),
                        *EVERY_FLAGS]) == 0
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert printed == [{"status": "dry_run_complete", "requests": 63, "paid_api_calls": 0},
                       {"status": "dry_run_complete", "requests": 99, "paid_api_calls": 0}]
    receipt = json.loads((every / "receipt.json").read_text())
    assert receipt["planned_initial_requests_by_kind"] == receipt["completed_initial_requests_by_kind"] == {"prediction": 51, "selection": 48}
    assert receipt["settings"] == module.settings("extra_dimensions", checkpoints=EVERY)
    assert json.loads((released / "receipt.json").read_text())["settings"] == module.settings("extra_dimensions")
    scores = json.loads((every / "scores.json").read_text())
    assert scores["checkpoints"] == list(EVERY) and [c["x"] for c in scores["cells"]] == list(EVERY) * 3
    # Forecasts are side queries: without the added ones, the requests and the scores are the released run's.
    assert corpus_bodies(every, CHECKPOINTS) == corpus_bodies(released, CHECKPOINTS)
    assert [c for c in scores["cells"] if c["x"] in CHECKPOINTS] == json.loads((released / "scores.json").read_text())["cells"]
    draft = json.loads((every / "seal_draft.json").read_text())
    assert draft["settings"]["checkpoints"] == list(EVERY)
    assert draft["d0_prediction_requests"] == json.loads((released / "seal_draft.json").read_text())["d0_prediction_requests"]


def test_an_every_round_seal_opens_only_the_every_round_plan(tmp_path):
    from tools.icl_baseline import transport
    draft = module.seal_draft("extra_dimensions", 32000, REGISTRATION, checkpoints=EVERY)
    seal = tmp_path / "every.json"
    sealed_from(draft, seal)
    required = [*module.dependencies(), REGISTRATION]
    def factory(**kwargs):
        return module.SyntheticClient()

    assert isinstance(transport.live_client(True, seal, module.settings("extra_dimensions", checkpoints=EVERY), required, factory),
                      module.SyntheticClient)
    with pytest.raises(ValueError, match="exactly matching settings"):
        transport.live_client(True, seal, module.settings("extra_dimensions"), required, factory)


def test_the_v2_driver_states_every_round_and_its_seal_helper_seals_only_that_plan(tmp_path, capsys):
    from tools.icl_forcebench.v2 import run_v2_world, seal_v2
    args = run_v2_world.arguments("dry", "extra_dimensions", tmp_path / "x")
    assert args[args.index("--checkpoints") + 1] == ",".join(str(x) for x in EVERY) and args[-1] == "--dry-run"
    assert args[args.index("--registration") + 1] == str(run_v2_world.REGISTRATION)
    assert args[args.index("--cap-usd") + 1] == "16"
    assert run_v2_world.arguments("paid", "gravity", "o", "seal.json")[-3:] == ["--allow-paid-api", "--seal-manifest", "seal.json"]
    for bad in (("paid", "gravity", "o"), ("dry", "fractional", "o"), ("dry", "gravity", "o", "seal.json")):
        with pytest.raises(SystemExit):
            run_v2_world.arguments(*bad)
    output = tmp_path / "every"
    assert module.main(run_v2_world.arguments("dry", "extra_dimensions", output)) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "dry_run_complete", "requests": 99, "paid_api_calls": 0}
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["settings"]["cap_usd"] == 16.0 and receipt["starting_balance_usd"] == 16.0
    summary = seal_v2.seal("extra_dimensions", output, tmp_path / "seal.json")
    assert summary["verify_seal"] == "passed" and summary["checkpoints"] == list(EVERY) and summary["cap_usd"] == 16.0
    with pytest.raises(SystemExit, match="not this world's draft"):
        seal_v2.seal("gravity", output, tmp_path / "other.json")
    released = tmp_path / "released"
    assert module.main(["--world", "extra_dimensions", "--dry-run", "--output", str(released), "--registration", str(REGISTRATION)]) == 0
    with pytest.raises(SystemExit, match="not this tree's registration-v2 settings"):
        seal_v2.seal("extra_dimensions", released, tmp_path / "released_seal.json")
