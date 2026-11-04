"""Run one AutoSciLab cell through the interface overlay, then score its per-experiment laws.

  python -m asl_kit.runner run --benchmark chembench  --task c0_michaelis_menten:easy:v0 --seed 31 --budget 30 --out <cell>
  python -m asl_kit.runner run --benchmark boxinggym  --task death --seed 31 --budget 30 --out <cell>
  python -m asl_kit.runner run --benchmark forcebench --task gravity --seed 31 --budget 16 --out <cell>
  python -m asl_kit.runner score --cell <cell>          (re-score a finished cell offline)
Options: --mode released|ef|er (evaluation schedule, see asl_kit/schedule.py; default released = the tip
behaviour byte for byte), --cap-cell-usd, --cap-folder-usd (folder = the cell's parent), --names masked|shown (BoxingGym/
ForceBench), --d0 ours|none, --force-noise relative|additive_var (ForceBench; default relative = NewtonBench's
5 % relative noise, decided 2026-10-06; recorded in every receipt), --model (default claude-opus-5-5),
--stub (mock Messages API + fake PySR).

Agent configuration (as released, see README):
  ChemBench   scripts/run_chembench_llm_autoscilab_budget.py `_run_one` -> run_llm_pipeline(...), called
              with the same arguments except noise (0.0 -> 0.01, our world) and main_url (a sentinel the
              Claude path consumes; it never reaches a network client)
  BoxingGym / ForceBench   scripts/run_paper_sweep.py `run_one` DiscoveryConfig (the NewtonBench runner):
              max_experiments_per_iter = eq_learner_every = max(1, budget // 5), pysr 800 iterations,
              families_v3, strict generator/discriminator loop, analysis feedback, negative evidence,
              stratified scoring, full-store promotion gate, mid_run_gt_eval (blocked).
"""
import argparse
import csv
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
import traceback

KIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KIT_DIR not in sys.path:
    sys.path.insert(0, KIT_DIR)

from asl_kit import paths, schedule  # noqa: E402

CLAUDE_SENTINEL_URL = "anthropic-messages://claude-path"   # consumed by install_claude_path, never dialled
DEFAULT_MODEL = "claude-opus-5-5"
# Matched case-insensitively. The released texts that carry ground truth into a prompt are the validation
# prompt's "GROUND-TRUTH EVALUATION (mid-run)" + "GT RMSLE: ..." / "Exact accuracy: ..." (prompts.py 668,
# discovery.py 1227-1228) and the memory's "Ground-truth RMSLE (mid-run eval)" / "... GROUND TRUTH DETECTED"
# (memory.py 401-406); with the leak block none of them can be filled, so every count must be 0.
FORBIDDEN_PROMPT_STRINGS = ("gt rmsle", "gt_rmsle", "exact accuracy:", "ground-truth", "ground truth", "held-out evaluator")

# scripts/run_chembench_llm_autoscilab_budget.py::_run_one, argument for argument
RELEASED_CHEM_KWARGS = dict(
    use_domain_tags=False, hypothesis_grammar_source="universal", strong_model=None, strong_model_calls=0,
    ensemble_mode=False, ensemble_k=5, ensemble_model="Qwen/Qwen2.5-7B-Instruct",
    ensemble_url="http://localhost:8001/v1", ensemble_every=2, max_completion_tokens=4096,
    ensemble_adaptive=True, ensemble_k_max=20, ensemble_stability_threshold=0.1,
    ensemble_confidence_gate=0.7, confidence_threshold=1.1, max_experiments_per_iter=5)


def newton_config_kwargs(budget):
    """scripts/run_paper_sweep.py::run_one's DiscoveryConfig, with run_newton_llm_autoscilab_budget.py's
    max_per_iter = max(1, budget // 5) and eq_every = max_per_iter, pysr_iters default 800, noise 0.0."""
    m = max(1, int(budget) // 5)
    return dict(budget=int(budget), noise_level=0.0, difficulty="easy", law_version="v0",
                max_experiments_per_iter=m, eq_learner_every=m, pysr_n_iterations=800, policy="families_v3",
                analysis_feedback_enabled=True, negative_evidence_enabled=True, stratified_hypothesis_scoring=True,
                promotion_full_store_gate=True, strict_generator_discriminator_loop=True, mid_run_gt_eval=True)


def cell_seed(benchmark, task, seed):
    return int(hashlib.sha256(f"asl|{benchmark}|{task}|{seed}".encode()).hexdigest()[:8], 16)


def _setup(args, cell):
    paths.require_clean_checkout()
    if args.fake_pysr:
        sys.path.insert(0, paths.STUB_DIR)        # the stub pysr/together shadow any installed ones
    elif args.mock_llm:
        sys.path.append(paths.STUB_DIR)           # only mock_messages is taken from the stub dir
    paths.import_checkout()
    schedule.install(getattr(args, "mode", "released"))   # released: a no-op; ef / er: the DiscoveryConfig subclass
    s = cell_seed(args.benchmark, args.task, args.seed)
    random.seed(s)
    import numpy as np
    np.random.seed(s)
    from asl_kit import claude_client
    ledger = claude_client.UsageLedger(cell, args.cap_cell_usd, args.cap_folder_usd, mock=args.mock_llm)
    factory = None
    handler = None
    if args.mock_llm:
        from mock_messages import MockMessages, mock_client
        handler = MockMessages()
        factory = lambda: mock_client(handler)   # noqa: E731
    claude_client.install_claude_path(ledger, client_factory=factory)
    import pysr
    receipt_env = {"pysr_version": getattr(pysr, "__version__", "?"), "pysr_is_stub": bool(getattr(pysr, "STUB", False)),
                   "mock_llm": bool(args.mock_llm)}
    return s, ledger, handler, receipt_env


def _build(args, cell):
    """oracle (leak-blocked, what the agent sees), scorer (what the agent never sees), D0 results."""
    if args.benchmark == "chembench":
        from asl_kit import chem
        domain, difficulty, law_version = (args.task.split(":") + ["easy", "v0"])[:3]
        oracle = chem.make_oracle(domain, difficulty, law_version)
        scorer = chem.ChemScorer(domain, difficulty, law_version)
        d0 = [oracle.run(x) for x in chem.fixed3_designs()] if args.d0 == "ours" else []
        return oracle, scorer, d0, {"domain": domain, "difficulty": difficulty, "law_version": law_version,
                                    "noise_level": chem.NOISE_LEVEL, "grammar_mode": chem.GRAMMAR_MODE,
                                    "names": "as_released (C_A..pH, universal grammar, goal sentence, domain id); --names ignored"}
    if args.benchmark == "boxinggym":
        from asl_kit import boxinggym as bg
        entry = bg.load_entry(args.task, args.seed)
        oracle = bg.make_oracle(entry, cell_seed("boxinggym_obs", args.task, args.seed), names=args.names)
        scorer = bg.BoxingGymScorer(entry)
        d0 = oracle.d0_results() if args.d0 == "ours" else []
        return oracle, scorer, d0, {"world_seed": entry["world_seed"], "evaluation_seed": entry["evaluation_seed"],
                                    "d0_x": entry["d0"]["x"], "d0_y": entry["d0"]["y"]}
    if args.benchmark == "forcebench":
        from asl_kit import forcebench as fb
        wp = fb.load_world(args.task)
        noise = getattr(args, "force_noise", fb.DEFAULT_NOISE)
        oracle = fb.make_oracle(args.task, wp, args.seed, names=args.names, noise=noise)
        scorer = fb.ForceBenchScorer(args.task, wp)
        d0 = oracle.d0_results() if args.d0 == "ours" else []
        return oracle, scorer, d0, {"force_noise": noise, "force_noise_rule": fb.NOISE_RULES[noise],
                                    "force_noise_default": fb.DEFAULT_NOISE, "sigma_f": wp["sigma_f"],
                                    "floor_oracle_law": wp["floor_oracle_law"], "method": wp["method"],
                                    "d0_knobs": wp["d0_knobs"].get(str(args.seed))}
    raise SystemExit(f"unknown benchmark {args.benchmark}")


def _goal(args):
    if args.names == "masked" or args.benchmark == "forcebench":
        label = "the unknown system" if args.names == "masked" else "the force between two bodies"
    else:
        from asl_kit.boxinggym import SHOWN_TEXT
        label = SHOWN_TEXT[args.task][0]
    return f"Discover the governing law for {label}."


def _run_agent(args, cell, oracle):
    from autoscilab.loop import discovery as D
    results_dir = os.path.join(cell, "autoscilab_results")
    if args.benchmark == "chembench":
        import run_chembench_comparison as RCC
        expected = (oracle.domain, oracle._difficulty, oracle._law_version)

        def get_oracle(domain_id, difficulty, noise_level, law_version, grammar_mode, **_):
            if (domain_id, difficulty, law_version, grammar_mode) != expected + ("universal",):
                raise RuntimeError(f"unexpected oracle request {(domain_id, difficulty, law_version, grammar_mode)}")
            return oracle

        D.get_oracle = get_oracle
        domain, difficulty, law_version = expected
        summary = RCC.run_llm_pipeline(domain, difficulty, law_version, budget=args.budget, model=args.model,
                                       noise=oracle._noise_level, main_url=CLAUDE_SENTINEL_URL,
                                       results_dir=__import__("pathlib").Path(results_dir), **RELEASED_CHEM_KWARGS)
        if summary.get("error"):
            raise RuntimeError(f"run_llm_pipeline refused: {summary}")
        return summary
    cfg = D.DiscoveryConfig(domain_id=oracle.domain, goal=_goal(args), llm_model=args.model,
                            results_dir=__import__("pathlib").Path(results_dir), **newton_config_kwargs(args.budget))
    loop = D.DiscoveryLoop(cfg, api_key=None, oracle=oracle, base_url=None)
    loop.run()
    return {"goal": cfg.goal}


def _audit_prompts(cell, secret_numbers):
    counts = {s: 0 for s in FORBIDDEN_PROMPT_STRINGS}
    numbers = {s: 0 for s in secret_numbers}
    n = 0
    path = os.path.join(cell, "llm_requests.jsonl")
    if os.path.exists(path):
        for line in open(path):
            if not line.strip():
                continue
            n += 1
            text = line
            low = line.lower()
            for s in counts:
                counts[s] += low.count(s)
            for s in numbers:
                numbers[s] += text.count(s)
    return {"requests_scanned": n, "forbidden_strings": counts, "truth_constant_strings": numbers,
            "clean": all(v == 0 for v in counts.values()) and all(v == 0 for v in numbers.values())}


def _secret_numbers(args):
    """World constants with >= 5 significant digits, formatted as Python repr and as %.6g."""
    out = set()
    if args.benchmark == "boxinggym":
        from asl_kit import boxinggym as bg
        w = bg.load_entry(args.task, args.seed)["world"]
        vals = [v for k, v in w.items() if isinstance(v, float)]
    elif args.benchmark == "forcebench":
        from asl_kit import forcebench as fb
        vals = [v for v in fb.load_world(args.task)["law_constants"].values() if isinstance(v, float)]
    else:
        vals = []
    for v in vals:
        for s in (repr(v), f"{v:.6g}"):
            if len(s.replace("-", "").replace(".", "").lstrip("0")) >= 5:
                out.add(s)
    return sorted(out)


def score_cell(cell, scorer=None, args=None):
    """laws.jsonl x scorer -> curve.json / curve.csv (b = experiments after D0)."""
    receipt = json.load(open(os.path.join(cell, "receipt.json")))
    if scorer is None:
        a = argparse.Namespace(**receipt["args"])
        paths.import_checkout()
        _o, scorer, _d0, _m = _build(a, cell)
    laws = [json.loads(l) for l in open(os.path.join(cell, "laws.jsonl")) if l.strip()]
    exps = [json.loads(l) for l in open(os.path.join(cell, "experiments.jsonl")) if l.strip()]
    d0_x = receipt.get("d0_executed_x") or []
    names = receipt["parameter_names"]
    cache, rows = {}, []
    for row in laws:
        b = int(row["b"])
        observed = d0_x + [[e["executed"][n] for n in names] for e in exps if int(e["b"]) <= b]
        key = (row.get("law_str"), len(observed) if receipt["args"]["benchmark"] == "boxinggym" else 0)
        if key not in cache:
            cache[key] = scorer.score(row.get("law_str"), observed=observed)
        rows.append({"b": b, "n_obs": row.get("n_obs"), "source": row.get("source"), "skeleton": row.get("skeleton"),
                     **cache[key]})
    with open(os.path.join(cell, "curve.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(cell, "curve.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return rows


def run(args):
    cell = os.path.abspath(args.out)
    if os.path.isdir(cell) and os.listdir(cell):
        raise SystemExit(f"cell folder exists and is not empty (a previous attempt?): {cell}")
    os.makedirs(cell, exist_ok=True)
    t0 = time.time()
    s, ledger, handler, receipt_env = _setup(args, cell)
    from asl_kit import protocol, claude_client
    oracle, scorer, d0, world_meta = _build(args, cell)
    recorder = protocol.LawRecorder(cell, args.budget, len(d0))
    context = protocol.RunContext(d0, recorder)
    protocol.install_run_wrapper(context)
    receipt = {"schema": "asl_cell_receipt_v1", "args": vars(args), "cell": cell, "python_seed": s,
               "environment": receipt_env, "python": sys.version.split()[0],
               "started_unix": t0, "model": args.model, "effort": claude_client.EFFORT,
               "max_tokens_floor": claude_client.MAX_TOKENS_FLOOR, "thinking": "adaptive",
               "prices_usd_per_m": {"input": claude_client.PRICE_INPUT_PER_M, "output": claude_client.PRICE_OUTPUT_PER_M},
               "checkout": paths.checkout_state(), "kit_files_sha256": paths.kit_file_hashes(),
               "parameter_names": oracle.parameter_names, "parameter_bounds": oracle.parameter_bounds,
               "domain_shown_to_agent": oracle.domain, "param_description_shown_to_agent": oracle.param_description,
               "function_signature": oracle.function_signature, "world": world_meta, "d0_count": len(d0),
               "d0_executed_x": [[r.params[n] for n in oracle.parameter_names] for r in d0],
               "d0_y": [r.measurement for r in d0]}
    status, error, agent_summary = "completed", None, None

    def _on_sigterm(signum, frame):   # a stop signal from the queue: close the cell with a receipt
        raise SystemExit("SIGTERM (running stop)")
    import signal
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        agent_summary = _run_agent(args, cell, oracle)
    except BaseException as exc:   # recorded, then the cell is closed with what it has
        status, error = "error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    res = context.result
    loop = context.loop
    usage = ledger.totals()
    termination = getattr(res, "termination_reason", None)
    laws_with_a_law = sorted(b for b, row in recorder.rows.items() if row.get("law_str"))
    carried = sum(1 for row in recorder.rows.values() if str(row.get("source", "")).endswith("_carried_after_termination"))
    if status == "completed" and usage["refused_by_cap"]:
        # a cap refusal inside one of the released loop's try/except sites (generator, analysis, final
        # arbitration) lets the run finish on a fallback path - not a clean released run
        status = "completed_with_cap_refusals"
    receipt.update(
        status=status, error=error, agent_summary=agent_summary, wall_s=round(time.time() - t0, 1),
        termination=getattr(res, "termination_reason", None), n_oracle_calls=getattr(res, "n_oracle_calls", None),
        n_llm_calls_agent=getattr(res, "n_llm_calls", None),
        final_law=getattr(getattr(res, "best_equation", None), "law_str", None),
        final_evaluation_in_run=getattr(res, "final_evaluation", None),
        released_config=({k: str(v) for k, v in dataclasses.asdict(loop._cfg).items()} if loop is not None else None),
        schedule=schedule.receipt_block(args.mode),
        first_fit_b=(laws_with_a_law[0] if laws_with_a_law else None),
        early_termination={"terminated_before_budget": bool(termination is not None and termination != "budget"),
                           "termination": termination, "carried_rows": carried},
        fits={"count": len(recorder.fits), "seconds_total": round(sum(f["seconds"] for f in recorder.fits), 1),
              "returned_none": sum(1 for f in recorder.fits if not f["returned"]),
              "failed": sum(1 for f in recorder.fits if f.get("error")), "rows": recorder.fits},
        blocked_attempts=protocol.BLOCK["attempts"], blocked_where=sorted(set(protocol.BLOCK["where"])),
        usage=usage, prompt_audit=_audit_prompts(cell, _secret_numbers(args)),
        mock_requests_seen=(len(handler.requests) if handler is not None else None))
    with open(os.path.join(cell, "receipt.json"), "w") as fh:
        json.dump(receipt, fh, indent=1, default=str)
    rows = score_cell(cell, scorer=scorer)
    scored = [r for r in rows if r.get("status") == "scored"]
    receipt["curve_summary"] = {"rows": len(rows), "scored": len(scored),
                                "no_law": sum(1 for r in rows if r.get("status") == "no_law"),
                                "law_error": sum(1 for r in rows if r.get("status") == "law_error"),
                                "first_scored_b": scored[0]["b"] if scored else None}
    with open(os.path.join(cell, "receipt.json"), "w") as fh:
        json.dump(receipt, fh, indent=1, default=str)
    print(json.dumps({"cell": cell, "status": status, "usage": receipt["usage"], "blocked": receipt["blocked_attempts"],
                      "audit_clean": receipt["prompt_audit"]["clean"], "curve": receipt["curve_summary"]}, default=str))
    return {"completed": 0, "completed_with_cap_refusals": 5}.get(status, 4)


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--benchmark", required=True, choices=("chembench", "boxinggym", "forcebench"))
    r.add_argument("--task", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--budget", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--mode", choices=schedule.MODES, default="released",
                   help="evaluation schedule: released (default) | ef (fit after every iteration) | er (1 experiment per round)")
    r.add_argument("--names", choices=("masked", "shown"), default="masked")
    r.add_argument("--d0", choices=("ours", "none"), default="ours")
    r.add_argument("--force-noise", choices=("relative", "additive_var"), default="relative",
                   help="ForceBench measurement noise: relative (default) or the brief's additive_var mapping")
    r.add_argument("--cap-cell-usd", type=float, default=5.0)
    r.add_argument("--cap-folder-usd", type=float, required=True)
    r.add_argument("--mock-llm", action="store_true", help="answer the Messages API from an httpx MockTransport (0 USD)")
    r.add_argument("--fake-pysr", action="store_true", help="least-squares stand-in for PySR (the development host has no Julia)")
    r.add_argument("--stub", action="store_true", help="= --mock-llm --fake-pysr")
    sc = sub.add_parser("score")
    sc.add_argument("--cell", required=True)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        if args.stub:
            args.mock_llm = args.fake_pysr = True
        return run(args)
    rows = score_cell(os.path.abspath(args.cell))
    print(json.dumps(rows[-1], default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
