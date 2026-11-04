#!/usr/bin/env python3
"""Kit tests (0 USD, no network): scorer equality with the LLM-AutoSciLab wrapper's scorer, scorer rules, the pack, the in-memory
retry patch v3, the world construction of all 15 cells against the pack, the D0 wording, the Q-to-question mapping, the
curve rules, the Claude-system augment, the env exception guard, and the retry patch v3 retry flow through the released code.

  env -u FORCE_COLOR PYTHONDONTWRITEBYTECODE=1 <venv python> -B tests/test_kit.py
APPR_ASL_KIT names the LLM-AutoSciLab wrapper folder
used for the scorer comparison; the test is skipped (reported SKIP) where that kit is absent.
"""
import contextlib
import json
import math
import os
import sys
import tempfile
from types import SimpleNamespace

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
sys.dont_write_bytecode = True
os.environ.setdefault("PYTENSOR_FLAGS", "base_compiledir=" + tempfile.mkdtemp(prefix="appr_kit_test_ptc_"))
os.environ.pop("ANTHROPIC_API_KEY", None)

from appr_kit import paths  # noqa: E402

paths.import_checkout()

import numpy as np  # noqa: E402

from appr_kit import claude_path as CP  # noqa: E402
from appr_kit import scorer, scripted, shim, worlds  # noqa: E402

ASL_KIT = os.environ.get("APPR_ASL_KIT", "<development path>")
GIT_APPLY_SHA256 = "98168c70a0d4e9128c9dbe389e2a5acb66008bee47b2b4336193a5c1da98fa8f"   # `git apply` of the patch on blob 92cdabe
SKIPPED = []


class Skip(Exception):
    pass


def _asl_boxinggym():
    if not os.path.isfile(os.path.join(ASL_KIT, "asl_kit", "boxinggym.py")):
        raise Skip(f"AutoSciLab kit not found at {ASL_KIT}")
    sys.path.insert(0, ASL_KIT)
    try:
        from asl_kit import boxinggym as ASL
    finally:
        sys.path.remove(ASL_KIT)
    return ASL


def test_scorer_equals_autoscilab_scorer():
    ASL = _asl_boxinggym()
    laws = {
        "dugongs": "def discovered_law(X1):\n    return 2.4 - 1.1 * np.exp(-0.9 * X1)\n",
        "hyperbolic": "def discovered_law(X1, X2, X3):\n    return 1.0 / (1.0 + np.exp(-(X2 - X1) / 40.0 + X3 / 200.0))\n",
        "survival": "def discovered_law(X1, X2):\n    return 1.0 / (1.0 + np.exp(-(0.3 * X1 + 1.5 * X2 - 1.0)))\n",
        "peregrines": "def discovered_law(X1):\n    return math.exp(4.0 + 0.9 * X1 - 0.2 * X1 ** 2)\n",
    }
    seeds = {"dugongs": 31, "hyperbolic": 37, "survival": 53, "peregrines": 41}
    compared = 0
    for task, law in laws.items():
        entry = ASL.load_entry(task, seeds[task], pack_path=paths.PACK_PATH)
        qx = np.asarray(entry["q"]["x"], dtype=float)
        fn = ASL.compile_law(law)
        pred_all = np.asarray([fn(*row) for row in qx], dtype=float)
        for observed in ([], entry["d0"]["x"], entry["d0"]["x"] + [list(qx[3]), list(qx[100])]):
            ref = ASL.BoxingGymScorer(entry).score(law, observed=observed)
            got = scorer.score_predictions(pred_all, {}, qx, entry["q"]["truth_mean"], entry["q"]["heldout"],
                                           entry["outcome_type"] == "binary", observed)
            assert got == ref, (task, observed, got, ref)
            compared += 1
    assert compared == 12
    # the copied rule functions are byte-identical in behaviour on a random case
    rng = np.random.default_rng(5)
    q, known = rng.integers(0, 4, size=(50, 2)).astype(float), rng.integers(0, 4, size=(6, 2)).astype(float)
    assert np.array_equal(scorer.unobserved_q_mask(q, known), ASL.unobserved_q_mask(q, known))
    a, b = rng.normal(size=40), rng.normal(size=40)
    assert scorer.nmse_truth(a, b) == ASL.nmse_truth(a, b)


def test_scorer_rules():
    qx = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    truth, heldout = np.asarray([0.0, 1.0, 2.0, 3.0]), np.asarray([0.0, 1.0, 1.0, 0.0])
    pred = np.asarray([0.5, np.nan, 2.0, 3.5])
    assert scorer.score_predictions(pred, {}, qx, truth, heldout, False, [])["status"] == "law_error"
    row = scorer.score_predictions(pred, {}, qx, truth, heldout, False, [[1.0]])
    assert row["status"] == "scored" and row["q_points"] == 3 and row["q_excluded"] == 1
    assert math.isclose(row["nmse_truth"], np.mean((np.asarray([0.5, 2.0, 3.5]) - [0, 2, 3]) ** 2) / np.var([0, 2, 3]))
    err = scorer.score_predictions(np.asarray([0.5, 1.0, 2.0, 3.5]), {2: "ValueError: x"}, qx, truth, heldout, True, [])
    assert err == {"status": "law_error", "error": "ValueError: x"}
    ok = scorer.score_predictions(np.asarray([0.5, 1.0, 2.0, 3.5]), {2: "ValueError: x"}, qx, truth, heldout, True, [[2.0]])
    assert ok["status"] == "scored" and "brier" in ok


def test_pack_and_patch():
    assert paths.sha256_file(paths.PACK_PATH) == paths.PACK_SHA256
    assert paths.sha256_file(paths.PATCH_PATH) == paths.PATCH_SHA256
    RE, info = paths.load_run_experiment()
    assert info["hunks_applied"] == 2 and info["patched_sha256"] == GIT_APPLY_SHA256, info
    assert info["source_git_blob"] == info["checkout_tree_blob"] == paths.RUN_EXPERIMENT_BLOB
    assert RE.CCR5B_MAX_RETRIES == 2 and RE.MAX_TRIES == 3 and callable(RE._ccr5b_placeholder)


def test_vendored_run_experiment_is_the_released_blob():
    import subprocess
    raw = open(paths.VENDOR_RUN_EXPERIMENT, "rb").read()
    got = subprocess.run(["git", "hash-object", paths.VENDOR_RUN_EXPERIMENT], capture_output=True, text=True).stdout.strip()
    assert paths.git_blob_id(raw) == got == paths.RUN_EXPERIMENT_BLOB
    st = paths.checkout_state()
    assert st["run_experiment_blob_in_tree"] == paths.RUN_EXPERIMENT_BLOB == st["run_experiment_blob_vendored"]
    if st["run_experiment_in_worktree"]:
        assert st["run_experiment_blob_worktree"] == paths.RUN_EXPERIMENT_BLOB
    paths.require_clean_checkout()


def test_peregrines_47_matches_the_16_entry_pack():
    """2026-10-08: the pack gains peregrines seed 47 (the site registry's seed); the same checks as the 15 cells."""
    with contextlib.ExitStack() as stack:
        w = worlds.World("peregrines", 47).build(stack)
        assert all(v for v in w.checks.values() if isinstance(v, bool)), w.checks
        assert len(w.goal.eval_points) == 256 and w.checks["truth_max_abs_diff"] == 0.0
        assert w.pack_sha256 == paths.PACK_SHA256


def test_all_15_worlds_match_the_pack():
    cells = [(t, s) for t in ("death", "dugongs", "peregrines", "hyperbolic") for s in (31, 37, 41)] + \
            [("survival", s) for s in (43, 53, 61)]
    for task, seed in cells:
        with contextlib.ExitStack() as stack:
            w = worlds.World(task, seed).build(stack)
            assert all(v for v in w.checks.values() if isinstance(v, bool)), (task, seed, w.checks)
            assert len(w.goal.eval_points) == 256 and w.checks["truth_max_abs_diff"] == 0.0


def test_d0_block_wording():
    with contextlib.ExitStack() as stack:
        w = worlds.World("death", 31).build(stack)
        assert w.d0_block() == ("Before you begin, 2 observations have already been made in this environment (given to "
                                "you as prior data):\nInput: 0.4248181818181818 -> Result: 14\nInput: 1.999 -> Result: 46\n"
                                "Use them. You will now make your own observations.")


def test_question_mapping():
    assert worlds.question_from_row("survival", [6.5, 1.0], [False, True]) == (1, 6.5)
    q = worlds.question_from_row("hyperbolic", [25.0, 294.0, 161.0], [True, True, True])
    assert q == (25, 294, 161) and all(isinstance(v, int) for v in q)
    assert worlds.question_from_row("death", [0.25], [False]) == (0.25,)


def _fake_recorder(task="death", seed=31):
    with contextlib.ExitStack() as stack:
        w = worlds.World(task, seed).build(stack)
    rec = shim.Recorder(tempfile.mkdtemp(), tempfile.mkdtemp(), w, [0, 2, 5], SimpleNamespace(calls=0), 3)
    return rec, w


def test_curve_rules():
    rec, w = _fake_recorder()
    q3 = list(w.qx[3])
    for b in range(1, 6):
        rec.start_experiment()
        rec.add_try("0.5" if b != 4 else str(q3[0]), True, 10, [0.5] if b != 4 else q3, {}, None)
    pred = np.asarray(w.truth) + 1.0
    rec.checkpoint_rows = [{"b": 0, "pred_all": pred, "errors": {}, "skeleton": "s0", "unavailable_reason": None},
                           {"b": 2, "pred_all": None, "errors": {}, "skeleton": None, "unavailable_reason": "CCR-5b v3"},
                           {"b": 5, "pred_all": pred, "errors": {}, "skeleton": "s5", "unavailable_reason": None}]
    assert shim.last_complete_budget(rec) == 5
    rows = shim.build_curve(rec, 5)
    assert [r["source"] for r in rows] == ["checkpoint_0", "carried_from_checkpoint_0", "checkpoint_2",
                                           "carried_from_checkpoint_2", "carried_from_checkpoint_2", "checkpoint_5"]
    assert [r["status"] for r in rows] == ["scored", "scored", "law_error", "law_error", "law_error", "scored"]
    assert rows[0]["q_excluded"] == 0 and rows[5]["q_excluded"] == 1 and rows[5]["n_obs"] == 7
    assert math.isclose(rows[0]["nmse_truth"], 1.0 / float(np.var(w.truth)), rel_tol=1e-9)
    rec.checkpoint_rows = rec.checkpoint_rows[1:]
    assert shim.last_complete_budget(rec) == -1          # checkpoint 0 missing: no row at all
    rec2, _ = _fake_recorder()
    rec2.checkpoints = [2, 5]
    rec2.start_experiment(); rec2.add_try("0.5", True, 1, [0.5], {}, None)
    rec2.start_experiment(); rec2.add_try("0.5", True, 1, [0.5], {}, None)
    rec2.checkpoint_rows = [{"b": 2, "pred_all": pred, "errors": {}, "skeleton": "s", "unavailable_reason": None}]
    rows2 = shim.build_curve(rec2, shim.last_complete_budget(rec2))
    assert [r["status"] for r in rows2] == ["no_law", "no_law", "scored"]


def test_augment_builds_claude_system_from_base():
    RE, _info = paths.load_run_experiment()
    rec, w = _fake_recorder("dugongs", 31)
    shim.install(RE, rec, w)
    sci = SimpleNamespace(system="BASE", _appr_base_system="BASE",
                          messages=[{"role": "user", "content": [{"type": "text", "text": "first user turn"}]}])
    progs = [[{"str_prob_prog": "PROGRAM-A"}]]
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        RE.augment_scientist_with_ppl(sci, progs, [], critic_mode=False)
    first = sci.system
    assert first.startswith("BASE\n \nTo help guide your experimentation") and "Program: PROGRAM-A" in first
    assert sci.messages[0]["content"][0]["text"] == "first user turn"
    progs.append([{"str_prob_prog": "PROGRAM-B"}])
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        RE.augment_scientist_with_ppl(sci, progs, [[{"str_hypotheses": "H", "synthesis": "S"}]], critic_mode=True)
    assert "PROGRAM-A" not in sci.system and "PROGRAM-B" in sci.system and "Here is criticism of the previous model" in sci.system
    assert sci.system.startswith("BASE\n \n")


def test_env_exception_guard():
    with contextlib.ExitStack() as stack:
        w = worlds.World("hyperbolic", 31).build(stack)
        rec = shim.Recorder(tempfile.mkdtemp(), tempfile.mkdtemp(), w, [0, 2], SimpleNamespace(calls=0), 3)
        shim.instrument_env(w.env, rec, "hyperbolic")
        rec.start_experiment()
        value, ok = w.env.run_experiment("[10, 20, 5] (note)")
        assert not ok and value.startswith("Error: invalid input (ValueError") and len(rec.guard_events) == 1
        value, ok = w.env.run_experiment("[10, 20, 5]")
        assert ok and rec.experiments[1]["executed"] == [10.0, 20.0, 5.0] and rec.experiments[1]["completed"]
        assert rec.n_obs_at(1) == 3


class _FailingThenOK(scripted.ScriptedClient):
    """Prior-mode programs that fail at prediction time (wrong return arity) for the first `fail` proposals."""

    def __init__(self, task, fail):
        super().__init__(task)
        self.fail, self.proposals = fail, 0

    def create(self, *, model, max_tokens, system, messages, thinking=None, temperature=None, extra_body=None):
        resp = super().create(model=model, max_tokens=max_tokens, system=system, messages=messages, thinking=thinking,
                              temperature=temperature, extra_body=extra_body)
        if "You are a brilliant statistician modeling a dataset" in system:
            self.proposals += 1
            if self.proposals <= self.fail:
                text = resp.content[1].text.replace("return model, prior_predictive", "return model, prior_predictive, None")
                resp.content[1].text = text
        return resp


def _ccr5b_run(fail):
    from hydra import initialize_config_dir
    RE, _info = paths.load_run_experiment()
    cell = tempfile.mkdtemp(prefix="appr_ccr5b_")
    work = os.path.join(cell, "work")
    os.makedirs(work)
    ledger = CP.UsageLedger(cell, 5.0, 50.0, mock=True)
    client = _FailingThenOK("death", fail)
    path = CP.ClaudePath(ledger, client)
    CP.bind_box_loop(path)
    with contextlib.ExitStack() as stack:
        w = worlds.World("death", 31).build(stack)
        w.env.include_prior = True
        rec = shim.Recorder(cell, work, w, [0], ledger, RE.MAX_TRIES)
        shim.install(RE, rec, w)
        here = os.getcwd()
        with initialize_config_dir(config_dir=os.path.join(paths.checkout_dir(), "conf"), version_base=None):
            os.chdir(work)
            try:
                with contextlib.redirect_stdout(open(os.devnull, "w")):
                    out = RE.ppl_evaluate("x", w.goal, None, 10, True, [[]], [], prior_mode=True, critic_mode=False)
            finally:
                os.chdir(here)
    return out, rec, client


def test_ccr5b_v3_unavailable_after_two_reproposals():
    out, rec, client = _ccr5b_run(fail=99)
    assert out[0] == (None, None) and client.proposals == 3, (out[0], client.proposals)
    cp = rec.checkpoint_rows[0]
    assert cp["fit_status"] == "measurement_unavailable" and cp["re_proposals"] == 2 and cp["pred_all"] is None
    assert cp["ccr5b_events"][-1].get("outcome") == "measurement_unavailable"
    assert shim.build_curve(rec, 0)[0]["status"] == "law_error"


def test_ccr5b_v3_success_after_one_reproposal():
    out, rec, client = _ccr5b_run(fail=1)
    assert out[0] != (None, None) and client.proposals == 2
    cp = rec.checkpoint_rows[0]
    assert cp["fit_status"] == "accepted" and cp["re_proposals"] == 1 and cp["q_errors"] == 0
    assert np.isfinite(cp["pred_all"]).all() and shim.build_curve(rec, 0)[0]["status"] == "scored"


# ---------------------------------------------------------------- 2026-10-08: prompt variant no_thought, batched Q prediction
def test_prompt_filter_removes_every_thought_tag_of_the_released_texts_and_nothing_else():
    from appr_kit import prompt_filter as PF
    import inspect
    from boxing_gym.agents import agent as agent_module
    # the follow-up texts of the released scientist (agent.py 111 / 113), captured from a stand-in llm
    captured = []

    class Capture:
        def __init__(self):
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kw):
            content = kw["messages"][-1]["content"]   # the released Claude branch sends text blocks
            captured.append(content if isinstance(content, str) else "".join(b["text"] for b in content))
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="<observe>1.0</observe>")])
    saved = agent_module.anthropic
    agent_module.anthropic = type("NoRealClient", (), {"Anthropic": staticmethod(lambda: None)})
    try:
        sci = agent_module.LMExperimenter(model_name="claude-opus-5-5", temperature=0.0, max_tokens=512)
    finally:
        agent_module.anthropic = saved
    sci.llm = Capture()
    sci.set_system_message("s")
    sci.generate_actions(None)
    sci.generate_actions("12")
    assert len(captured) == 2 and all("<thought>" in c for c in captured)
    texts = {"follow_up_first": captured[0], "follow_up_result": captured[1]}
    for task in sorted(worlds.TASKS):
        with contextlib.ExitStack() as stack:
            w = worlds.World(task, 43 if task == "survival" else 31)   # the pack's seeds (survival 43/53/61)
            w.build(stack, stdout_path=os.devnull)
            texts[f"system_{task}"] = w.goal.get_system_message(True)
    for name, text in texts.items():
        released, c0 = PF.apply(text, "released")
        assert released == text and not any(c0.values())
        out, counts = PF.apply(text, "no_thought")
        assert PF.remaining_tags(out) == 0, (name, out)
        assert sum(counts.values()) == text.count("<thought>"), (name, counts, text.count("<thought>"))
        # only <thought> material left: every other line is still there, in order
        kept = [l for l in text.splitlines() if "<thought>" not in l]
        kept_out = [l for l in out.splitlines() if "<thought>" not in l]
        if name == "follow_up_first":
            kept = [l.replace(" Articulate your strategy for choosing measurements in <thought>.", "") for l in text.splitlines()
                    if "<thought>" not in l or "Articulate" in l]
        assert "<observe>" in out or "answer" in out or name.startswith("follow_up"), name
        assert "<thought>" not in out
        for line in ("<observe>", "<answer>"):
            assert out.count(line) == text.count(line), (name, line)
    # the two follow-ups keep their instruction and format lines
    first, _ = PF.apply(texts["follow_up_first"], "no_thought")
    assert first == ("Think about where to observe next.\nProvide a new measurement point in the format:\n"
                     "<observe> your observation</observe>\nMake an observation now."), first
    second, _ = PF.apply(texts["follow_up_result"], "no_thought")
    assert second == ("Result: 12\nThink about where to observe next.\nProvide a new measurement point in the format:\n"
                      "<observe> your observation (remember the type of inputs accepted)</observe>"), second
    # the Claude path applies it at send time and refuses a request that still carries the tag
    cell = tempfile.mkdtemp(prefix="appr_pf_")
    ledger = CP.UsageLedger(cell, 1.0, 1.0, mock=True)
    client = scripted.ScriptedClient("death")
    path = CP.ClaudePath(ledger, client, prompt_variant="no_thought")
    sys_text = texts["system_death"]
    answer, _u, row = path.call("scientist", sys_text, [{"role": "user", "content": texts["follow_up_first"]}], 512)
    assert "<observe>" in answer and row["prompt_variant"] == "no_thought" and row["prompt_filter"]["changed"]
    sent = client.requests[-1]
    assert "<thought>" not in sent["system"] and "<thought>" not in sent["messages"][0]["content"]
    assert path.filter_calls == 1 and sum(path.filter_totals.values()) == sum(row["prompt_filter"]["counts"].values())
    released_path = CP.ClaudePath(ledger, scripted.ScriptedClient("death"))
    _a, _u, row2 = released_path.call("scientist", sys_text, [{"role": "user", "content": texts["follow_up_first"]}], 512)
    assert row2["prompt_variant"] == "released" and not row2["prompt_filter"]["changed"]
    assert "<thought>" in released_path.client.requests[-1]["system"]


def test_batched_q_prediction_matches_the_per_point_released_path():
    """The scripted death program, prior and posterior mode: the batched predictive mean per Q point equals the released
    per-point get_ppl_prediction mean up to Monte Carlo error (both average the same number of draws)."""
    import numpy as np
    from appr_kit import paths as P
    RE, _info = P.load_run_experiment(P.checkout_dir())
    with contextlib.ExitStack() as stack:
        w = worlds.World("death", 31)
        w.build(stack, stdout_path=os.devnull)
        env = w.env
        env.include_prior = True   # the driver sets it from conf/config.yaml before the loop
        # a few observations so the posterior program has data (through the released env, equal to the pack's D0)
        for x in ("1.0", "2.0", "3.0"):
            env.run_experiment(x)
        env.get_df()
        cell = tempfile.mkdtemp(prefix="appr_q_")
        rec = shim.Recorder(cell, cell, w, [0, 30], CP.UsageLedger(cell, 1.0, 1.0, mock=True), 3, q_method="batched")
        for prior in (True, False):
            program = scripted.program("death", prior)
            if prior:
                program_dict = {"str_prob_prog": program}
            else:
                gen = RE.get_gen_model(program)
                env.get_df()                       # the released loop calls it before the box loop; it sets env.df
                model, _ppc, trace = gen(env.df)
                program_dict = {"str_prob_prog": program, "model": model, "trace": trace}
            batched = rec._q_batched(RE, env, program_dict, prior)
            per_point, errors, _wall = rec._q_per_point(RE, env, program_dict, prior)
            assert not errors and batched.shape == per_point.shape == (len(w.questions),)
            draws = 1000 if prior else 3000
            # Monte Carlo error of a mean of `draws` Binomial(50, p) draws: at most sqrt(50/4 / draws) ~ 0.11 / 0.065
            tol = 8 * np.sqrt(12.5 / draws)
            diff = float(np.max(np.abs(batched - per_point)))
            assert diff <= tol, (prior, diff, tol)
            print(f"  batched vs per_point ({'prior' if prior else 'posterior'}): max abs diff {diff:.4f}, tolerance {tol:.4f}")
        assert rec.q_fallbacks == []
        # the public entry point under q_method batched reports no errors and no fallback
        pred, errs, wall = rec.q_predictions(RE, env, program_dict, False)
        assert errs == {} and pred.shape == (len(w.questions),) and rec.q_fallbacks == []
        rec_pp = shim.Recorder(cell, cell, w, [0, 30], CP.UsageLedger(cell, 1.0, 1.0, mock=True), 3)
        assert rec_pp.q_method == "per_point"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Skip as exc:
                SKIPPED.append(name)
                print("SKIP", name, exc)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", name, type(exc).__name__, exc)
    print("RESULT", "PASS" if not failed else f"FAIL ({failed})", f"skipped {len(SKIPPED)}")
    sys.exit(1 if failed else 0)
