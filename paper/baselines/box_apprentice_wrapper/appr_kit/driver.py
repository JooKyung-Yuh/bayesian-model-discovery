"""Run one Box's Apprentice cell on one of our BoxingGym worlds and score it per budget.

  python -m appr_kit.driver --task dugongs --seed 31 --budget 30 --mode stub|mock-transport|paid --out <cell> \
      --cap-cell-usd 3 --cap-folder-usd <F> [--checkpoints 0,2,5,10,20,30] [--prompt-variant released|no_thought]
      [--q-prediction per_point|batched]
  python -m appr_kit.driver score --cell <cell>       (re-score a finished cell offline from predictions/*.npz)

The released loop is run_experiment.iterative_experiment (use_ppl=True, naive_agent=None, include_prior=True) from the
checkout's run_experiment.py with the retry patch v3 applied in memory; see shim.py for the changes and the README.
Outputs in <cell>/: receipt.json, curve.json, curve.csv, laws.jsonl, experiments.jsonl, bg_native_scores.json,
llm_usage.jsonl, llm_requests.jsonl, llm_responses.jsonl, scientist_messages.json, box_loop.log, predictions/,
world_build_stdout.txt; work/ (the released code's CWD: logs/, ccr5b_retries.json, oed_llms_stan/); ptc/ (PyTensor).
Exit code: 0 completed, 5 completed_with_cap_refusals, 4 failed.
"""
import argparse
import contextlib
import csv
import json
import os
import platform
import random
import signal
import socket
import sys
import time
import traceback

KIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KIT_DIR not in sys.path:
    sys.path.insert(0, KIT_DIR)
sys.dont_write_bytecode = True

from appr_kit import paths  # noqa: E402

SCHEMA = "appr_cell_receipt_v1"
DEFAULT_CHECKPOINTS = "0,2,5,10,20,30"
UNROUTABLE = "http://127.0.0.1:9"
PACKAGES = ("numpy", "scipy", "pandas", "pymc", "pytensor", "arviz", "hydra-core", "omegaconf", "anthropic", "httpx",
            "openai", "matplotlib", "xarray")

INTERFACE_CHANGES = [
    {"id": "box_loop_llm", "what": "model_search.AsyncOpenAIGPT4V (hard-coded gpt-4o, OpenAI chat completions) is replaced by "
     "AnthropicAsOpenAI: claude-opus-5-5 on the Messages API; the system-role text becomes the top-level system string, user and "
     "assistant text pass through byte for byte, image blocks are refused (use_vision false), n choices = n sequential calls; "
     "it returns .model / .choices[i].message.content / .usage.prompt_tokens (input incl. cache) / .completion_tokens"},
    {"id": "cost_table", "what": "base_agent.MODEL_COST_PER_INPUT / MODEL_COST_PER_OUTPUT gain claude-opus-5-5 = 4e-6 / 2e-5 USD per "
     "token (BaseAgent.calc_cost only; the kit's ledger is the spend record)"},
    {"id": "scientist_llm", "what": "LMExperimenter is built with model claude-opus-5-5 (conf/llms/openai.yaml says gpt-4o) and the "
     "conf's temperature 0.0 and max_tokens 512; its .llm is the kit's ScientistClient (the released Claude branch reads "
     ".content[0].text, which is the first TEXT block of the reply)"},
    {"id": "claude_request", "what": "every call: thinking {type: adaptive}, output_config {effort: medium} (extra_body on anthropic "
     "0.71.0), max_tokens = max(requested, 16000) (the released 512 / 4096 would be eaten by thinking), temperature dropped and "
     "counted, refusal recorded then raised (no fallback model), answer = first text block; caps checked before every call"},
    {"id": "program_in_system", "what": "augment_scientist_with_ppl: the released function builds the text on a stand-in object; "
     "the result (base system + latest program [+ critic]) is sent as the Claude system message, rebuilt from the base each "
     "time as on the gpt-4o path. The released function writes scientist.messages[0], which on the Claude branch is the first "
     "user turn (and does not exist yet at checkpoint 0); that write is not done"},
    {"id": "d0_prior_data", "what": "our 2 random start designs and their observations (env.run_experiment, equal to the pack) "
     "are given in the BASE-1 wording, appended to the scientist's system message after one newline; the box loop sees them "
     "through env.get_df() from the first posterior checkpoint on"},
    {"id": "ccr5b_v3", "what": "BoxingGym-Baseline-Evals baselines/patches/ccr5b_retry_v3.patch is applied IN MEMORY to the "
     "released run_experiment.py source (no file is written): at most 2 re-proposals with the error fed back, then "
     "'measurement unavailable' for that checkpoint (curve status law_error)"},
    {"id": "run_experiment_source", "what": "the released run_experiment.py is read from the kit's verbatim copy "
     "vendor/run_experiment.py (git blob 92cdabe2), checked against the checkout's tree entry at 06c790aa (and its worktree "
     "file where present): the development hosts' sparse checkouts (src, conf) have no root files"},
    {"id": "checkpoints", "what": "num_experiments: conf/exp/oed.yaml [0, 1, 3, 5, 7, 10] -> [0, 2, 5, 10, 20, 30] (protocol the shared protocol: "
     "budget 30 experiments after D0); checkpoint 0 is the box loop's prior mode"},
    {"id": "worlds", "what": "env and goal are built the engine's way (worlds.py: env_seed_injection(world_seed) for the whole "
     "cell, np.random.seed(world_seed), constructor defaults = conf env_params, 256 goal eval points under evaluation_seed, "
     "D0 through env.run_experiment) instead of hydra's config seed; checked against the pack before any call. Survival's "
     "eval points are loaded from the pack (recorded from the engine) instead of re-drawn"},
    {"id": "native_eval_points", "what": "BoxingGym's own score uses the goal's first num_evals = 10 stored eval points: for "
     "death, dugongs, peregrines and hyperbolic the goal's own generator under our evaluation_seed (= Q[:10]); for survival "
     "the pack's Q[:10]. The released run would have drawn 10 points at the first checkpoint from the then-current RNG"},
    {"id": "env_exception_guard", "what": "an exception raised inside env.run_experiment (released code: hyperbolic int(float()) "
     "on a non-numeric item, survival ID 100) is returned to the loop as (error text, False), the released invalid-design path "
     "(MAX_TRIES = 3), instead of ending the cell"},
    {"id": "empty_turns", "what": "the Messages API rejects a turn without text: the released prior-mode proposal's user message "
     "is empty (box_loop_prompts.get_stan_user_prompt_prior returns ''), so it is sent as 'Please proceed.' (checkpoint 0 "
     "only); an assistant turn without text in the scientist's history would be sent as '(no text in this reply)'; both are "
     "counted in usage (empty_user_turns_replaced, empty_assistant_turns_replaced)"},
    {"id": "consecutive_turns", "what": "consecutive messages of the same role (the released critic sends one user message per "
     "program after its user prompt) are sent as one turn with one text block per released message, which is how the "
     "Messages API combines consecutive turns; counted in usage (merged_consecutive_turns)"},
    {"id": "working_folders", "what": "CWD = <cell>/work (the box loop's ./logs and ccr5b_retries.json), checkpoint_dir = "
     "<cell>/work/oed_llms_stan (box_loop_experiment.py:38 hard-codes /tmp/oed_llms/stan/), PYTENSOR base_compiledir = <cell>/ptc"},
    {"id": "hydra", "what": "hydra initialize_config_dir(<checkout>/conf) around the loop so run_box_loop's compose('box_loop') "
     "works (the released main runs under @hydra.main)"},
    {"id": "logging", "what": "the boxing_gym.agents loggers write INFO to <cell>/box_loop.log (no console); no effect on behaviour"},
    {"id": "scoring", "what": "after each checkpoint the kit predicts our 256 Q points through the released get_ppl_prediction "
     "with the accepted program, under rng isolation (global numpy / random states restored, pymc's unwrapped draw functions, "
     "posterior-predictive progress bar off); nothing flows back into the loop"},
    {"id": "python_random_seed", "what": "random.seed(<world seed index>) as the released main() does with config.seed"},
]
# Added to interface_changes only when the option is on (2026-10-08).
OPTIONAL_CHANGES = {
    "prompt_variant": {"id": "prompt_variant_no_thought", "what": "the released `<thought>` instruction is removed from the system "
                       "string and every user turn at send time (appr_kit/prompt_filter.py, three rules: the `<thought> your thought "
                       "</thought>` example lines of the environment prompts, the sentence 'Articulate your strategy for choosing "
                       "measurements in <thought>.' and the line 'Thought: <thought>' of the scientist's follow-ups); the "
                       "`<observe>` / `<answer>` format and the released parser are untouched. Reason: claude-opus-5-5 refuses the "
                       "released text (stop_reason refusal, 2026-10-08 canary). Counts per rule in llm_usage.jsonl and the receipt"},
    "q_prediction": {"id": "q_prediction_batched", "what": "the kit's scorer predicts the 256 Q points in one predictive sampling run "
                     "(construct_features / pm.set_data over all rows, mean over draws per row) instead of one released "
                     "get_ppl_prediction call per point; same program, trace and draw count, different Monte Carlo draws; a "
                     "program that cannot take 256 rows falls back to the per-point path for that checkpoint (receipt q_fallbacks). "
                     "Scoring only; nothing flows back into the loop"},
}


def _read_yaml(path):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def released_config(task, checkout):
    from appr_kit import worlds
    conf = os.path.join(checkout, "conf")
    env_conf, env_path = worlds.conf_env(task, checkout)
    import importlib
    import inspect
    module_name, env_class = worlds.TASKS[task][0], worlds.TASKS[task][1]
    sig = inspect.signature(getattr(importlib.import_module(module_name), env_class).__init__)
    defaults = {k: p.default for k, p in sig.parameters.items() if p.default is not inspect.Parameter.empty}
    env_params = env_conf.get("env_params") or {}
    return {
        "files": {"config": _read_yaml(os.path.join(conf, "config.yaml")),
                  "llms/openai": _read_yaml(os.path.join(conf, "llms", "openai.yaml")),
                  "box_loop": _read_yaml(os.path.join(conf, "box_loop.yaml")),
                  "exp/oed": _read_yaml(os.path.join(conf, "exp", "oed.yaml")),
                  f"envs/{os.path.basename(env_path)[:-5]}": env_conf},
        "taken": {"config.include_prior": "used (true)", "config.use_ppl": "used (true)",
                  "llms.temperature": "passed to LMExperimenter; dropped on the wire (Opus 5.5)",
                  "llms.max_tokens": "passed to LMExperimenter (512); sent as max(512, 16000)",
                  "llms.model_name": "replaced by claude-opus-5-5",
                  "box_loop.*": "used as is through compose('box_loop') (llm GPT4-V -> AnthropicAsOpenAI, max_tokens 4096 "
                                "sent as 16000, temperatures 0.0 dropped, num_rounds 1, num_proposals 1, language_synthesize, "
                                "critic, warm_start false)",
                  "exp.num_experiments": "replaced by the checkpoints of this run",
                  "envs.num_evals": "used (BoxingGym-native score)", "envs.com_limit": "passed (unused: no naive agent)",
                  "envs.env_params": "not passed: the engine builds the env with constructor defaults",
                  "envs.goal_name": "direct (the engine's goal class)"},
        "env_params_equal_constructor_defaults": all(defaults.get(k) == v for k, v in env_params.items()),
        "constructor_defaults": {k: v for k, v in defaults.items() if isinstance(v, (int, float, str, bool))},
    }


def classify(exc):
    from appr_kit import claude_path
    if isinstance(exc, claude_path.CostCapExceeded):
        return "cost_cap_refusal"
    if isinstance(exc, claude_path.ClaudeRefusal):
        return "claude_refusal"
    if isinstance(exc, SystemExit) and "SIGTERM" in str(exc):
        return "sigterm_running_stop"
    if isinstance(exc, SystemExit) and "differs from the pack" in str(exc):
        return "world_check_failed"
    return "error"


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")


def write_curve(cell, rows):
    with open(os.path.join(cell, "curve.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=str)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(cell, "curve.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys or ["b"])
        w.writeheader()
        w.writerows(rows)


def curve_summary(rows):
    scored = [r for r in rows if r.get("status") == "scored"]
    return {"rows": len(rows), "scored": len(scored), "no_law": sum(1 for r in rows if r.get("status") == "no_law"),
            "law_error": sum(1 for r in rows if r.get("status") == "law_error"),
            "first_scored_b": scored[0]["b"] if scored else None}


def make_client(mode, task):
    if mode == "stub":
        from appr_kit import scripted
        client = scripted.ScriptedClient(task)
        return client, client
    if mode == "mock-transport":
        from appr_kit import mock_transport
        handler = mock_transport.MessagesAPIMock(task)
        return mock_transport.mock_client(handler), handler
    import anthropic   # official SDK; the key comes from ANTHROPIC_API_KEY in the environment
    return anthropic.Anthropic(max_retries=2, timeout=600.0), None


def run(args):
    cell = os.path.abspath(args.out)
    if os.path.isdir(cell) and os.listdir(cell):
        raise SystemExit(f"cell folder exists and is not empty (a previous attempt is never reused): {cell}")
    checkpoints = [int(c) for c in args.checkpoints.split(",")]
    if checkpoints != sorted(set(checkpoints)) or checkpoints[-1] != args.budget:
        raise SystemExit(f"checkpoints {checkpoints} must be increasing and end at the budget {args.budget}")
    interface_changes = list(INTERFACE_CHANGES)
    if args.prompt_variant != "released":
        interface_changes.append(OPTIONAL_CHANGES["prompt_variant"])
    if args.q_prediction != "per_point":
        interface_changes.append(OPTIONAL_CHANGES["q_prediction"])
    want = f"base_compiledir={os.path.join(cell, 'ptc')}"
    flags = os.environ.get("PYTENSOR_FLAGS", "")
    if not flags:
        os.environ["PYTENSOR_FLAGS"] = want
    elif want not in flags.split(","):
        raise SystemExit(f"PYTENSOR_FLAGS={flags!r} must contain {want}")
    if args.mode in ("stub", "mock-transport"):
        for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "GPT4_API_KEY"):
            os.environ.pop(k, None)
        os.environ["ANTHROPIC_BASE_URL"] = UNROUTABLE
        os.environ["OPENAI_BASE_URL"] = UNROUTABLE
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("--mode paid needs ANTHROPIC_API_KEY in the environment (read by name, never printed)")
    t0 = time.time()
    started = time.strftime("%Y-%m-%d %H:%M:%S KST")
    checkout = paths.checkout_dir()
    checkout_before = paths.require_clean_checkout(checkout)
    os.makedirs(cell, exist_ok=True)
    work = os.path.join(cell, "work")
    os.makedirs(work, exist_ok=True)
    paths.import_checkout(checkout)

    import importlib.metadata as md
    import numpy as np
    from appr_kit import claude_path, prompt_filter, shim, worlds   # shim captures pymc's functions before env_seed_injection
    RE, re_info = paths.load_run_experiment(checkout)
    from boxing_gym.agents import agent as agent_module
    from hydra import initialize_config_dir

    random.seed(args.seed)
    ledger = claude_path.UsageLedger(cell, args.cap_cell_usd, args.cap_folder_usd, mock=args.mode != "paid")
    client, transport = make_client(args.mode, args.task)
    path = claude_path.ClaudePath(ledger, client, model=args.model, prompt_variant=args.prompt_variant)
    claude_path.bind_box_loop(path)
    world = worlds.World(args.task, args.seed)
    env_conf, _ = worlds.conf_env(args.task, checkout)
    llm_conf = _read_yaml(os.path.join(checkout, "conf", "llms", "openai.yaml"))
    top_conf = _read_yaml(os.path.join(checkout, "conf", "config.yaml"))
    receipt = {"schema": SCHEMA, "args": {**vars(args), "benchmark": "boxinggym", "names": "shown (include_prior=True)"},
               "cell": cell, "mode": args.mode, "model": args.model, "effort": claude_path.EFFORT,
               "thinking": "adaptive", "max_tokens_floor": claude_path.MAX_TOKENS_FLOOR,
               "prices_usd_per_m": {"input": claude_path.PRICE_INPUT_PER_M, "output": claude_path.PRICE_OUTPUT_PER_M},
               "checkpoints": checkpoints, "budget": args.budget, "started": started, "host": socket.gethostname(),
               "prompt_variant": args.prompt_variant, "prompt_filter": prompt_filter.describe(args.prompt_variant),
               "q_prediction": args.q_prediction,
               "python": platform.python_version(),
               "packages": {p: (md.version(p) if p else None) for p in PACKAGES},
               "checkout": {**checkout_before, "path": checkout, "head": checkout_before["head"]},
               "run_experiment": re_info, "kit_files_sha256": paths.kit_file_hashes(),
               "released_config": released_config(args.task, checkout), "interface_changes": interface_changes,
               "pytensor_flags": os.environ.get("PYTENSOR_FLAGS")}
    status, error, termination, rec, scientist = "completed", None, None, None, None

    def _on_sigterm(signum, frame):   # a stop signal from the queue: close the cell with a receipt
        raise SystemExit("SIGTERM (running stop)")

    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        with contextlib.ExitStack() as stack:
            world.build(stack, stdout_path=os.path.join(cell, "world_build_stdout.txt"))
            env, goal = world.env, world.goal
            env.include_prior = bool(top_conf.get("include_prior", True))
            rec = shim.Recorder(cell, work, world, checkpoints, ledger, RE.MAX_TRIES, q_method=args.q_prediction)
            receipt["wrapped"] = sorted(shim.install(RE, rec, world))
            saved_anthropic = agent_module.anthropic
            agent_module.anthropic = type("NoRealClient", (), {"Anthropic": staticmethod(lambda: None)})
            try:
                scientist = RE.LMExperimenter(model_name=args.model, temperature=llm_conf["temperature"],
                                              max_tokens=llm_conf["max_tokens"])
            finally:
                agent_module.anthropic = saved_anthropic
            scientist.llm = claude_path.ScientistClient(path)
            base_system = goal.get_system_message(env.include_prior) + "\n" + world.d0_block()
            scientist.set_system_message(base_system)
            scientist._appr_base_system = base_system
            receipt["d0_block"] = world.d0_block()
            receipt["scientist_base_system_sha256"] = paths.sha256_bytes(base_system.encode())
            shim.instrument_scientist(scientist, rec)
            shim.instrument_env(env, rec, args.task)
            with initialize_config_dir(config_dir=os.path.join(checkout, "conf"), version_base=None):
                here = os.getcwd()
                os.chdir(work)
                try:
                    RE.iterative_experiment(goal, scientist, checkpoints, int(env_conf["num_evals"]),
                                            env.include_prior, None, env_conf.get("com_limit"), False,
                                            bool(top_conf.get("use_ppl", True)))
                finally:
                    os.chdir(here)
            termination = "budget_exhausted"
    except BaseException as exc:  # noqa: BLE001 - recorded, then the cell is closed with what it has
        status, termination = "failed", classify(exc)
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    usage = ledger.totals()
    if status == "completed" and usage["refused_by_cap"]:
        status = "completed_with_cap_refusals"
    rows, laws, experiments, b_end = [], [], [], -1
    if rec is not None:
        b_end = shim.last_complete_budget(rec)
        rows = shim.build_curve(rec, b_end) if b_end >= 0 else []
        for cp in rec.checkpoint_rows:
            laws.append({k: v for k, v in cp.items() if k not in ("pred_all", "errors")})
        experiments = [rec.experiments[b] for b in sorted(rec.experiments)]
        for e in experiments:
            last = e["tries"][-1] if e["tries"] else {}
            e["requested"] = last.get("requested")
    write_jsonl(os.path.join(cell, "laws.jsonl"), laws)
    write_jsonl(os.path.join(cell, "experiments.jsonl"), experiments)
    with open(os.path.join(cell, "bg_native_scores.json"), "w") as fh:
        json.dump({"note": "BoxingGym's own goal.evaluate_predictions on the goal's first num_evals stored eval points, "
                           "predictions from get_ppl_prediction (str of the predictive mean), at every checkpoint",
                   "metric_name": "accuracy" if world.binary else "mse", "num_evals": int(env_conf["num_evals"]),
                   "checkpoints": rec.native if rec else []}, fh, indent=1, default=str)
    if scientist is not None:
        with open(os.path.join(cell, "scientist_messages.json"), "w") as fh:
            json.dump({"system_final": scientist.system, "all_messages": scientist.all_messages}, fh, indent=1, default=str)
    write_curve(cell, rows)
    accepted = [cp for cp in (rec.checkpoint_rows if rec else []) if cp["fit_status"] == "accepted" and not cp["placeholder"]]
    checkout_after = paths.checkout_state(checkout)
    receipt.update(
        status=status, error=error, wall_s=round(time.time() - t0, 1), termination=termination,
        ended=time.strftime("%Y-%m-%d %H:%M:%S KST"), n_llm_calls=usage["calls"],
        final_law=accepted[-1]["program"] if accepted else None,
        final_law_checkpoint=accepted[-1]["b"] if accepted else None,
        usage=usage, world={**world.world_record(), "checks": world.checks},
        world_checks=world.checks, b_end=b_end, curve_summary=curve_summary(rows),
        experiments_summary={"iterations": len(experiments), "successful": sum(1 for e in experiments if e["success"]),
                             "design_attempts": sum(len(e["tries"]) for e in experiments),
                             "guard_events": len(rec.guard_events) if rec else 0},
        guard_events=rec.guard_events if rec else [],
        checkpoints_summary=[{k: cp[k] for k in ("b", "mode", "fit_status", "placeholder", "re_proposals", "llm_calls",
                                                  "box_loop_and_native_wall_s", "q_wall_s", "q_errors", "native_metric")}
                             for cp in (rec.checkpoint_rows if rec else [])],
        checkout_after={"head": checkout_after["head"], "porcelain_lines": checkout_after["porcelain_lines"],
                        "run_paths_porcelain_lines": checkout_after["run_paths_porcelain_lines"],
                        "unchanged": checkout_after["porcelain_lines"] == checkout_before["porcelain_lines"]},
        prompt_filter_totals={"calls_changed": path.filter_calls,
                              "substitutions_over_all_calls": path.filter_totals,   # every call re-sends the history, so the counts accumulate
                              "thought_tags_in_system_and_user_turns_sent": 0 if args.prompt_variant != "released" else None,
                              "thought_tags_in_assistant_turns_sent": path.assistant_thought_tags if args.prompt_variant != "released" else None},
        q_fallbacks=(rec.q_fallbacks if rec is not None else []),
        mock_requests_seen=(len(transport.requests) if transport is not None else None),
        mock_violations=(transport.violations if transport is not None else None),
        box_loop_clients_constructed=claude_path._STATE["constructed"])
    with open(os.path.join(cell, "receipt.json"), "w") as fh:
        json.dump(receipt, fh, indent=1, default=str)
    print(json.dumps({"cell": cell, "status": status, "termination": termination, "wall_s": receipt["wall_s"],
                      "usage": {k: usage[k] for k in ("calls", "refused_by_cap", "errors", "usd", "mock")},
                      "curve": receipt["curve_summary"]}, default=str), flush=True)
    return {"completed": 0, "completed_with_cap_refusals": 5}.get(status, 4)


def score(cell):
    """Re-score curve rows from the saved predictions (no LLM, no fit): prints the rows' (b, status, nmse_truth)."""
    from appr_kit import scorer, worlds
    import numpy as np
    receipt = json.load(open(os.path.join(cell, "receipt.json")))
    world = worlds.World(receipt["args"]["task"], receipt["args"]["seed"])
    rows = json.load(open(os.path.join(cell, "curve.json")))
    exps = [json.loads(l) for l in open(os.path.join(cell, "experiments.jsonl")) if l.strip()]
    d0 = [[float(v) for v in x] for x in receipt["world"]["d0_x"]]
    out = []
    for r in rows:
        if "checkpoint" not in r or r["status"] == "no_law":
            out.append((r["b"], r["status"], None))
            continue
        p = os.path.join(cell, "predictions", f"b{int(r['checkpoint']):02d}.npz")
        if not os.path.exists(p):
            out.append((r["b"], "law_error", None))
            continue
        z = np.load(p)
        errors = {int(i): "error" for i in np.flatnonzero(z["error_mask"])}
        observed = d0 + [e["executed"] for e in exps if e["success"] and int(e["b"]) <= int(r["b"])]
        s = scorer.score_predictions(z["pred"], errors, world.qx, world.truth, world.heldout, world.binary, observed)
        out.append((r["b"], s["status"], s.get("nmse_truth")))
    for row in out:
        print(json.dumps(row))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="run", choices=("run", "score"))
    ap.add_argument("--task", choices=("death", "dugongs", "peregrines", "hyperbolic", "survival"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--checkpoints", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--prompt-variant", choices=("released", "no_thought"), default="released",
                    help="no_thought: the released <thought> instruction removed at send time (appr_kit/prompt_filter.py)")
    ap.add_argument("--q-prediction", choices=("per_point", "batched"), default="per_point",
                    help="how the kit scores the 256 Q points at a checkpoint (scoring only; see shim.Recorder.q_predictions)")
    ap.add_argument("--mode", choices=("stub", "mock-transport", "paid"))
    ap.add_argument("--out")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--benchmark", default="boxinggym", choices=("boxinggym",))
    ap.add_argument("--cap-cell-usd", type=float, default=3.0)
    ap.add_argument("--cap-folder-usd", type=float)
    ap.add_argument("--cell", help="for `score`")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "score":
        return score(os.path.abspath(args.cell))
    missing = [k for k in ("task", "seed", "mode", "out", "cap_folder_usd") if getattr(args, k) is None]
    if missing:
        raise SystemExit(f"missing arguments: {missing}")
    rc = run(args)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)   # a box-loop fit that hit the released 8-minute alarm leaves a non-daemon worker thread behind


if __name__ == "__main__":
    main()
