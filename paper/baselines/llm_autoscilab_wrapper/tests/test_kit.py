#!/usr/bin/env python3
"""Kit checks on the development host (0 USD): leak block, same-world parity, scorer parity with OUR scorers,
adapter rules, and that the agent configuration equals what the released runners build.

  env -u FORCE_COLOR PYTHONDONTWRITEBYTECODE=1 PYTENSOR_FLAGS=base_compiledir=<scratch>/pytensor \
    PYTHONPATH=<repository root>/src:<LLM-AutoSciLab checkout> python -B tests/test_kit.py
"""
import dataclasses
import json
import math
import os
import sys
import tempfile

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREE = "<development path>"
NOLEAK_MANIFEST = ("<development path>"
                   "p4_chembench_noleak_oracle_blocked_b40_gpt4omini_5runs/manifest.json")
sys.path[:0] = [os.path.join(KIT, "tests", "stub"), KIT]
from asl_kit import paths  # noqa: E402

paths.import_checkout()
sys.path[:0] = [TREE, TREE + "/src", "<development path>"]

import numpy as np  # noqa: E402

from asl_kit import boxinggym as BG, chem, forcebench as FB, protocol, runner  # noqa: E402


# ---------------------------------------------------------------- ChemBench
def test_chem_leak_block_and_same_observations():
    from autoscilab.oracle.chembench import ChemBenchOracle
    from adapters.chembench_observation import ChemBenchSession
    for domain, diff in (("c0_michaelis_menten", "easy"), ("c5_pingpong_bisubstrate", "hard")):
        o = chem.make_oracle(domain, diff)
        assert isinstance(o, ChemBenchOracle) and type(o).ASL_BLOCKED
        before = protocol.BLOCK["attempts"]
        for call in (lambda: o.evaluate_law("def discovered_law(*a):\n    return 1.0\n"), o.get_ground_truth_law_str):
            try:
                call()
                raise AssertionError("not blocked")
            except RuntimeError as exc:
                assert protocol.BLOCK_MESSAGE in str(exc)
        assert protocol.BLOCK["attempts"] == before + 2
        session = ChemBenchSession(domain, diff, "v0", noise_level=0.01, budget=10)
        for x in chem.fixed3_designs():
            assert session.observe(dict(x))["y"] == o.run(dict(x)).measurement   # identical world and noise


def test_chem_scorer_matches_noleak_variances_and_truth():
    manifest = json.load(open(NOLEAK_MANIFEST))["var_log1p_truth"]
    from autoscilab.oracle.chembench import ChemBenchOracle
    from bayes_model_discovery.experimental.chembench_core import log_error_metrics
    for key, ref in manifest.items():
        domain, diff, ver = key.split("__")
        s = chem.ChemScorer(domain, diff, ver)
        assert abs(s.var_log1p_primary_1000 - ref) < 1e-9, (key, s.var_log1p_primary_1000, ref)
        truth = ChemBenchOracle(domain, diff, 0.01, ver, "universal").get_ground_truth_law_str()
        row = s.score(truth)
        assert row["official_rmsle_1000"] < 1e-12 and row["nmsle_1000_primary"] < 1e-20, row
        assert row["nmsle_256_primary"] < 1e-20 and 0 < row["nmsle_256_noiseless_observation"] < 1e-2, row
        # the 256-point statistic equals OUR engine's log_error_metrics on a positive prediction
        law = truth.replace("return ", "return 1.37 * ")
        mine = s.score(law)
        fn = chem.compile_law(law)
        pred = np.asarray([fn(*(x[k] for k in s._vars)) for x in s.q256])
        for tname, target in s.targets256.items():
            ours = log_error_metrics(pred, np.log1p(target), float(np.var(np.log1p(target))))["nmsle"]
            assert abs(mine[f"nmsle_256_{tname}"] - ours) <= 1e-12 * max(1.0, ours), (key, tname)


def test_chem_eval_sets_equal_a_recorded_run():
    """Our recorded c5 easy s31 cell (read only): 256 inputs, both targets and the official 1,000 inputs."""
    c = "<development path>"
    names = ["C_A", "C_I", "C_B", "C_P", "Enz", "T", "pH"]
    s = chem.ChemScorer("c5_pingpong_bisubstrate", "easy", "v0")
    x256 = np.load(c + "/evaluation_inputs.npz")["x"]
    assert np.array_equal(x256, np.asarray([[d[n] for n in names] for d in s.q256]))
    ref = np.load(c + "/reference_after_finish.npz")
    assert np.array_equal(ref["primary_rate"], s.targets256["primary"])
    assert np.array_equal(ref["noiseless_observation"], s.targets256["noiseless_observation"])
    x1000 = np.load(c + "/official_evaluation_inputs.npz")["x"]
    assert np.array_equal(x1000, np.asarray([[d[n] for n in names] for d in chem.sample_designs(1000, 42)]))


def test_blocked_final_selection_cannot_use_gt():
    """discovery.py's own final GT-selection block, executed verbatim (as the 9/6 no-leak demo did): with the
    native oracle the GT-best candidate B replaces A; with the kit's blocked subclass A stays."""
    import textwrap
    import types
    from autoscilab.loop import discovery as D
    from autoscilab.oracle.chembench import ChemBenchOracle
    src = open(D.__file__).read().splitlines()
    s0 = next(i for i, l in enumerate(src) if "# GT-evaluate any promising final candidates" in l)
    e0 = next(i for i, l in enumerate(src) if i > s0 and "final_eq = self._best_gt_equation" in l)
    block = textwrap.dedent("\n".join(src[s0:e0 + 1]))
    names = "C_A, C_I, C_B, C_P, Enz, T, pH"
    A = types.SimpleNamespace(law_str=f"def discovered_law({names}):\n    return 4.2*Enz*C_A**0.9\n", skeleton_str="A")
    B = types.SimpleNamespace(law_str=f"def discovered_law({names}):\n    return 5.0*Enz*C_A/(1.0+C_A)\n", skeleton_str="B")
    out = {}
    for mode, oracle in (("as_released", ChemBenchOracle("c0_michaelis_menten", "easy", 0.01, "v0", "universal")),
                         ("blocked", chem.make_oracle("c0_michaelis_menten", "easy"))):
        me = types.SimpleNamespace(_objective_type="equation", _oracle=oracle, _best_gt_equation=None, _best_gt_eval_rmsle=None)
        ns = {"self": me, "deduped_final_candidates": [A, B], "final_eq": A, "console": D.console, "np": np}
        exec(block, ns)
        out[mode] = ns["final_eq"].skeleton_str
    assert out == {"as_released": "B", "blocked": "A"}, out


def test_prompt_audit_detects_released_gt_strings():
    """The audit must flag the released texts that carry ground truth (check the checker)."""
    from autoscilab.llm.prompts import build_validation_prompt
    from autoscilab.llm.memory import LLMMemory
    cell = tempfile.mkdtemp(prefix="asl_audit_")
    gt = "  GT RMSLE: 0.01234 (GOOD)\n  Exact accuracy: 1.0"
    p = build_validation_prompt(goal="g", domain_id="d", param_names=["X1"], param_description="-", function_signature="f",
                                data_table="t", best_hypothesis="h", budget_remaining=3, gt_metrics_str=gt)
    mem = LLMMemory()
    mem.add(iteration=1, n_data=10, equation="C0*X1", r2=0.99, rmsle=0.01, bootstrap_conf=0.5, gt_rmsle=0.004)
    with open(os.path.join(cell, "llm_requests.jsonl"), "w") as fh:
        fh.write(json.dumps({"call_index": 1, "system": "s", "messages": [{"role": "user", "content": p}]}) + "\n")
        fh.write(json.dumps({"call_index": 2, "system": "s", "messages": [{"role": "user", "content": mem.to_prompt_str()}]}) + "\n")
    a = runner._audit_prompts(cell, [])
    assert not a["clean"] and a["forbidden_strings"]["gt rmsle"] >= 1 and a["forbidden_strings"]["ground-truth"] >= 2, a


# ---------------------------------------------------------------- BoxingGym
def test_bg_scorer_parity_with_mda_score_row():
    from bayes_model_discovery.experimental import boxinggym_core as B
    rng = np.random.default_rng(5)
    for task, seed in (("death", 31), ("dugongs", 37), ("peregrines", 41), ("hyperbolic", 31), ("survival", 53)):
        e = BG.load_entry(task, seed)
        truth = np.asarray(e["q"]["truth_mean"], float)
        pred = truth * (1 + 0.1 * rng.standard_normal(truth.shape)) + 0.01
        assert BG.nmse_truth(pred, truth) == B.mda_score_row(2, pred, truth, None)["nmse_truth"]
        qx = np.asarray(e["q"]["x"], float)
        obs = [list(qx[3]), list(qx[7])] + [list(r) for r in e["d0"]["x"]]
        a = BG.unobserved_q_mask(qx[:, 0] if qx.shape[1] == 1 else qx, obs)
        b = B.unobserved_q_mask(qx[:, 0] if qx.shape[1] == 1 else qx, obs)
        # Q can repeat a point (peregrines' times are rounded to 0.1), so count matches independently
        expected = sum(1 for row in qx if any(np.array_equal(row, np.asarray(o, float)) for o in obs))
        assert np.array_equal(a, b) and int((~a).sum()) == expected and expected >= 2, (task, int((~a).sum()), expected)


def test_bg_adapter_rules_and_no_constants_in_prompt_strings():
    for task, seed in (("death", 31), ("dugongs", 31), ("peregrines", 31), ("hyperbolic", 31), ("survival", 43)):
        e = BG.load_entry(task, seed)
        o = BG.make_oracle(e, 123, names="masked")
        assert o.parameter_names == [f"X{i + 1}" for i in range(len(e["input_names"]))]
        text = " ".join([o.domain, o.param_description, o.function_signature, str(o.parameter_bounds)])
        for k, v in e["world"].items():
            if isinstance(v, float):
                assert repr(v) not in text and f"{v:.4g}" not in text, (task, k)
        import re
        for name in e["input_names"]:
            assert not re.search(rf"\b{re.escape(name)}\b", text), (task, name)
        d0 = o.d0_results()
        assert [r.measurement for r in d0] == e["d0"]["y"]
        try:
            o.evaluate_law("def discovered_law(*a):\n    return 0\n")
            raise AssertionError("not blocked")
        except RuntimeError:
            pass
        if task == "hyperbolic":
            r = o.run({"X1": 250.4, "X2": 20.6, "X3": 30.2})
            assert r.params == {"X1": 21.0, "X2": 250.0, "X3": 30.0} and r.metadata["swapped"]
        if task == "survival":
            recs = e["world"]["records"]
            seen = set(int(h) for h in e["d0"]["handles"])
            r = o.run({"X1": 5.0, "X2": 0.8})
            i = r.metadata["patient"]
            assert i not in seen and int(recs[i][2]) == 1 and r.measurement == recs[i][0]
            assert r.params == {"X1": recs[i][1], "X2": float(recs[i][2])}
            r2 = o.run({"X1": 5.0, "X2": 0.8})
            assert r2.metadata["patient"] != i


def test_bg_observation_means():
    """The numpy observation law against the pack's noise-free mean (boxinggym_core's own truth formula,
    computed by the engine) and against the analytic variance of the environment's step() law."""
    from scipy.stats import norm
    for task in ("death", "dugongs", "peregrines", "hyperbolic"):
        e = BG.load_entry(task, 31)
        w = e["world"]
        o = BG.make_oracle(e, 7)
        qx = np.asarray(e["q"]["x"], float)
        for j in (0, 100, 200):
            x = list(qx[j])
            draws = np.asarray([o._observe(o._legalize(list(x))[0], {}) for _ in range(4000)], float)
            mu = e["q"]["truth_mean"][j]
            se = draws.std(ddof=1) / math.sqrt(len(draws)) + 1e-12
            assert abs(draws.mean() - mu) < 5 * se + 1e-9, (task, j, draws.mean(), mu, se)
            if task == "death":
                p = 1 - math.exp(-w["theta"] * x[0])
                var = w["N"] * p * (1 - p)
            elif task == "dugongs":
                m = w["a"] - w["b"] * abs(w["l"]) ** x[0]
                var = m * m + w["sigma_obs"] ** 2 - mu * mu
            elif task == "peregrines":
                var = mu
            else:
                iR, dR, D = x
                p = w["epsilon"] + (1 - 2 * w["epsilon"]) * norm.cdf((dR / (1 + w["k"] * D) - iR) / w["alpha"])
                assert abs(p - mu) < 1e-12
                var = p * (1 - p)
            if var > 1e-6:
                # 5 standard errors of the sample variance, from the draws' own fourth central moment
                # (a fixed relative tolerance is too tight for a rare Bernoulli outcome)
                s2 = draws.var(ddof=1)
                m4 = float(np.mean((draws - draws.mean()) ** 4))
                se_var = math.sqrt(max(m4 - s2 * s2, 0.0) / len(draws))
                assert abs(s2 - var) < 5 * se_var + 1e-12, (task, j, s2, var, se_var)


# ---------------------------------------------------------------- ForceBench
def test_fb_floors_and_oracle():
    for world in ("gravity", "yukawa", "coulomb_easy", "oscillator", "extra_dimensions"):
        wp = FB.load_world(world)
        law = FB.force_law(world, wp["law_constants"])
        s = FB.scores_of_forecast(FB.forecast(law, wp, world == "oscillator"), wp)
        for k, ref in wp["floor_oracle_law"].items():   # built under numpy 2.5 / scipy 1.18, re-run here
            assert abs(s[k] - ref) <= 1e-12 + 1e-9 * abs(ref), (world, k, s[k], ref)
        assert FB.make_oracle(world, wp, 31).noise_mode == FB.DEFAULT_NOISE == "relative"   # main 2026-10-06
        cli = runner.build_parser().parse_args(["run", "--benchmark", "forcebench", "--task", world, "--seed", "31",
                                                "--budget", "16", "--out", "unused", "--cap-folder-usd", "1"])
        assert cli.force_noise == FB.DEFAULT_NOISE
        o = FB.make_oracle(world, wp, 31, noise="additive_var")
        names = o.parameter_names
        assert names == (["X1", "X2", "X3", "X4"] if world == "oscillator" else ["X1", "X2", "X3"])
        d0 = o.d0_results()
        assert [[r.params["X2"], r.params["X3"]] for r in d0] == wp["d0_knobs"]["31"]
        x = {n: (lo + hi) / 2 for n, (lo, hi) in o.parameter_bounds.items()}
        ys = [o.run(x).measurement for _ in range(3000)]
        args = [x[n] for n in names]
        f = law(*args)
        assert abs(np.mean(ys) - f) < 5 * wp["sigma_f"] / math.sqrt(3000) and abs(np.std(ys) - wp["sigma_f"]) < 0.05 * wp["sigma_f"]
        rel = FB.make_oracle(world, wp, 31, noise="relative")
        rel_runs = [rel.run(x) for _ in range(3000)]
        yr = [res.measurement for res in rel_runs]
        assert abs(np.std(yr) - 0.05 * abs(f)) < 0.05 * 0.05 * abs(f) + 1e-12
        assert all(res.noise_level == 0.05 for res in rel_runs[:3]) and o.run(x).noise_level == wp["sigma_f"]
        # an AutoSciLab-style law string for the true law scores the floor
        if world == "coulomb_easy":
            sc = FB.ForceBenchScorer(world, wp).score("def discovered_law(X1, X2, X3):\n    return X2*X3/X1**2\n")
            assert sc["status"] == "scored" and sc["nmse_shift"] < 1e-15, sc


def test_fb_d0_knobs_equal_recorded_runs():
    """Our recorded ForceBench cells (read only): the first two designs are the pack's D0 knobs."""
    import glob
    checked = 0
    for world in ("gravity", "yukawa", "coulomb_easy", "oscillator", "extra_dimensions"):
        wp = FB.load_world(world)
        for seed in (31, 37, 41):
            for f in sorted(glob.glob(f"<development path>{world}__*__s{seed}/FB1_RUN_{world}_*_seed{seed}_v3_free.json")):
                designs = json.load(open(f))["observations"]["designs"][:2]
                assert designs == wp["d0_knobs"][str(seed)], (f, designs, wp["d0_knobs"][str(seed)])
                checked += 1
    assert checked >= 15, checked


# ---------------------------------------------------------------- released configuration
class _Captured(Exception):
    pass


def _capture_configs(fn, modules=()):
    """Replace DiscoveryLoop (in the discovery module and in each module that imported the name) with a
    class that records the config and raises - the released code never reaches an LLM client."""
    from autoscilab.loop import discovery as D
    seen = []
    targets = [D, *modules]
    real = [m.DiscoveryLoop for m in targets]

    class Capture:
        def __init__(self, config, *a, **k):
            seen.append(config)
            raise _Captured()
    for m in targets:
        m.DiscoveryLoop = Capture
    try:
        fn()
    except _Captured:
        pass
    finally:
        for m, r in zip(targets, real):
            m.DiscoveryLoop = r
    assert seen, "no DiscoveryConfig captured"
    return seen


def test_released_chembench_config_equals_ours():
    import dotenv
    dotenv.load_dotenv = lambda *a, **k: False            # never read a .env file in this test
    import run_chembench_llm_autoscilab_budget as RB
    import run_chembench_comparison as RCC
    RB.load_dotenv = dotenv.load_dotenv
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or "sk-test-not-a-key"
    out = tempfile.mkdtemp(prefix="asl_cfg_")
    task = {"id": "t", "domain": "c2_product_inhibition", "difficulty": "medium", "law_version": "v0"}
    ns = type("A", (), dict(main_model="gpt-4o-mini", ensemble_model="Qwen/Qwen2.5-7B-Instruct",
                            ensemble_url="http://localhost:8001/v1", main_url=None, max_per_iter=5))()
    released = _capture_configs(lambda: RB._run_one(task, 30, ns, __import__("pathlib").Path(out)))
    # _run_one swallows the capture exception into error.json; the config was still captured
    ours = _capture_configs(lambda: RCC.run_llm_pipeline("c2_product_inhibition", "medium", "v0", budget=30,
                                                         model="claude-opus-5-5", noise=0.01,
                                                         main_url=runner.CLAUDE_SENTINEL_URL,
                                                         results_dir=__import__("pathlib").Path(out),
                                                         **runner.RELEASED_CHEM_KWARGS))
    a, b = dataclasses.asdict(released[0]), dataclasses.asdict(ours[0])
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"llm_model", "noise_level", "results_dir"}, diff
    assert a["noise_level"] == 0.0 and b["noise_level"] == 0.01


def test_released_newton_config_equals_ours():
    import dotenv
    dotenv.load_dotenv = lambda *a, **k: False
    import run_paper_sweep as RPS
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or "sk-test-not-a-key"

    class FakeOracle:
        def __init__(self, *a, **k):
            pass
    real = RPS.NewtonBenchOracle
    RPS.NewtonBenchOracle = FakeOracle
    out = tempfile.mkdtemp(prefix="asl_cfg_")
    try:
        for budget in (16, 30):
            m = max(1, budget // 5)   # run_newton_llm_autoscilab_budget.py: max_per_iter, eq_every
            released = _capture_configs(lambda: RPS.run_one("m0_gravity", "easy", "v0", 0, "gpt-4o-mini", budget, m, m,
                                                            0.0, 800, __import__("pathlib").Path(out), None),
                                        modules=(RPS,))
            from autoscilab.loop.discovery import DiscoveryConfig
            ours = DiscoveryConfig(domain_id="m0_gravity", goal=released[0].goal, llm_model="claude-opus-5-5",
                                   results_dir=__import__("pathlib").Path(out), **runner.newton_config_kwargs(budget))
            a, b = dataclasses.asdict(released[0]), dataclasses.asdict(ours)
            diff = {k for k in a if a[k] != b[k]}
            assert diff == {"llm_model"}, (budget, diff)
    finally:
        RPS.NewtonBenchOracle = real


# ---------------------------------------------------------------- evaluation schedule (released | ef | er)
def _released_runner_configs(out):
    """The configs the two released runners build (ChemBench budget runner, NewtonBench sweep runner) under the
    schedule installed in this process."""
    import dotenv
    dotenv.load_dotenv = lambda *a, **k: False
    import run_chembench_llm_autoscilab_budget as RB
    import run_chembench_comparison as RCC
    import run_paper_sweep as RPS
    RB.load_dotenv = dotenv.load_dotenv
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or "sk-test-not-a-key"
    chem = _capture_configs(lambda: RCC.run_llm_pipeline("c2_product_inhibition", "medium", "v0", budget=30,
                                                         model="claude-opus-5-5", noise=0.01,
                                                         main_url=runner.CLAUDE_SENTINEL_URL,
                                                         results_dir=__import__("pathlib").Path(out),
                                                         **runner.RELEASED_CHEM_KWARGS))[0]

    class FakeOracle:
        def __init__(self, *a, **k):
            pass
    real = RPS.NewtonBenchOracle
    RPS.NewtonBenchOracle = FakeOracle
    try:
        from autoscilab.loop import discovery as D
        newton = D.DiscoveryConfig(domain_id="m0_gravity", goal="g", llm_model="claude-opus-5-5",
                                   results_dir=__import__("pathlib").Path(out), **runner.newton_config_kwargs(30))
    finally:
        RPS.NewtonBenchOracle = real
    return dataclasses.asdict(chem), dataclasses.asdict(newton)


def test_schedule_modes_override_exactly_their_fields_in_both_runners():
    from asl_kit import schedule
    from autoscilab.loop import discovery as D
    out = tempfile.mkdtemp(prefix="asl_sched_")
    released_cls = D.DiscoveryConfig
    try:
        schedule.install("released")
        assert D.DiscoveryConfig is released_cls
        chem0, newton0 = _released_runner_configs(out)
        assert chem0["eq_learner_every"] == 20 and chem0["min_data_for_eq"] == 10 and chem0["max_experiments_per_iter"] == 5
        assert newton0["eq_learner_every"] == 6 and newton0["max_experiments_per_iter"] == 6 and newton0["min_data_for_eq"] == 10
        for mode in ("ef", "er"):
            cls = schedule.install(mode)
            assert D.DiscoveryConfig is cls and cls is not released_cls and issubclass(cls, released_cls)
            chem, newton = _released_runner_configs(out)
            for before, after in ((chem0, chem), (newton0, newton)):
                diff = {k for k in before if before[k] != after[k]}
                assert diff == set(schedule.OVERRIDES[mode]), (mode, diff)
                for k, v in schedule.OVERRIDES[mode].items():
                    assert after[k] == v, (mode, k, after[k])
            # the released value is recorded beside the applied one, per constructed config
            rec = schedule.STATE["applied"]
            assert len(rec) == 2 and rec[0]["eq_learner_every"] == {"released": 20, "applied": 1}
            assert rec[1]["eq_learner_every"] == {"released": 6, "applied": 1}
            if mode == "er":
                assert rec[0]["max_experiments_per_iter"] == {"released": 5, "applied": 1}
            block = schedule.receipt_block(mode)
            assert block["mode"] == mode and block["configs_constructed"] == 2 and block["overrides"] == schedule.OVERRIDES[mode]
        # the released schedule can be put back in the same process, and a second install is idempotent
        schedule.install("released")
        assert D.DiscoveryConfig is released_cls and schedule.receipt_block("released")["released_values"] is None
        schedule.install("er")
        schedule.install("er")
        assert D.DiscoveryConfig._released_config is released_cls
    finally:
        schedule.install("released")


def test_schedule_er_gate_fits_every_iteration_from_the_learner_floor():
    """Under er the released gate (oracle_calls >= min_data_for_eq and oracle_calls % eq_learner_every == 0) is true at every
    iteration; the released learner's own floor (fit returns None below 5 stored points) is what remains."""
    from asl_kit import schedule
    from autoscilab.loop import discovery as D
    import inspect
    src = inspect.getsource(D.DiscoveryLoop.run)
    assert "self._oracle_calls >= cfg.min_data_for_eq" in src and "self._oracle_calls % cfg.eq_learner_every == 0" in src
    for mode, calls_fitting in (("er", list(range(1, 31))), ("ef", list(range(1, 31))), ("released", [20])):
        o = schedule.OVERRIDES[mode]
        every, floor = o.get("eq_learner_every", 20), o.get("min_data_for_eq", 10)
        assert [c for c in range(1, 31) if c >= floor and c % every == 0] == calls_fitting, mode
    from autoscilab.equation_learner import learner as L
    assert "if len(store) < 5:" in inspect.getsource(L.EquationLearner.fit)


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name, flush=True)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                import traceback
                print("FAIL", name, type(exc).__name__, exc, flush=True)
                traceback.print_exc()
    print("RESULT", "PASS" if not failed else f"FAIL ({failed})")
    sys.exit(1 if failed else 0)
