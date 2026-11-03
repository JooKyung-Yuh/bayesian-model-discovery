"""Harvest the <current_law> blocks of a `--protocol current_law` run and fit their constants with DP's own fitting code.

Input: the run log DP wrote (`--store_output` -> <stem>.json: rounds[*].llm_reply, action, experiment_output, final_law) and,
when present, <stem>.results.json (DP's end-of-mission evaluation).
Output: <stem>.current_laws.json with one row per b = 0..B, where b = the number of experiments that had actually been EXECUTED
when the law was written (rounds with an experiment_output; a round whose <run_experiment> was refused or failed, a `no_tag`
round and a `warning` round add no experiment). The law stated in round r has seen the outputs of the executed experiments
of rounds 1..r-1, so b(r) = executed experiments in rounds < r. When several statements share one b (a round without an
experiment between two experiments), the latest statement is the row's law; `statements` counts them.

One fitting rule for every row of the curve (2026-10-08, REVIEW-ER): the law's declared constants (fit_parameters()) are
refitted on ALL executed experiments the agent had seen at that b, with DP's evaluator functions (_compile_law,
_compile_fit_parameters, _validate_fit_spec, _fit_law_parameters and mse_fitting._LOSS_FNS[world], the functions
mse_fitting.fit_law uses), reading the experiments from the run log (what DP's trajectory CSV would hold; the wrapper runs
DP with --no-trajectory-csv). This holds for b = B too: the final law (DP's <final_law>) is refitted on all B experiments
(`fit: harvest_all`). DP's own end-of-mission fit is different - it fits only the worlds in run_discovery.py's _FIT_WORLDS
(coulomb_easy is not among them) and on a 4 trajectories x 5 times subsample (evaluator.py FIT_MAX_TRAJECTORIES /
FIT_MAX_TIMES_PER_TRAJ) - so its constants are recorded beside, as `dp_final_fitted_params` / `dp_final_fit`, for the
as-released arm; the ER curve uses the harvest rule at every b. b = 0 is the round-1 law at its initial constants (no data;
`fit: none`). A b without any statement is `missing`; the scorer carries the previous law (status carried).
Nothing here is fed back to the agent; the harvest runs after the mission ended.
Usage (also called by dp_agent_throttled.py at the end of a current_law cell):
  dp_current_law.py <stem>.json <stem>.current_laws.json <world> <budget>
"""
import functools
import hashlib
import json
import os
import re
import sys
import time

TAG = "current_law"
SCHEMA = "dp_current_laws_v2"


def extract_tag(text, tag=TAG):
    if not text:
        return None
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1) if m else None


def executed_pairs(r):
    """(input, output) pairs of the experiments a round actually executed (an output with pos2 exists)."""
    if r.get("action") != "experiment" or r.get("experiment_error"):
        return []
    inputs, outputs = r.get("experiment_input") or [], r.get("experiment_output") or []
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return []
    return [(inp, out) for inp, out in zip(inputs, outputs)
            if isinstance(inp, dict) and isinstance(out, dict) and "pos2" in out]


def training_from_rounds(rounds, upto_round):
    """DP's loss-function sample shape ({"input", "output"} per experiment) from the experiments EXECUTED in rounds < upto_round."""
    return [{"input": inp, "output": out} for r in rounds if r.get("round", 0) < upto_round for inp, out in executed_pairs(r)]


def fit_constants(law_source, world, training):
    """DP's fit of the law's declared constants on `training` (mse_fitting.fit_law without the CSV detour)."""
    from scienceagent import evaluator as EV
    from scienceagent.mse_fitting import _LOSS_FNS
    result = {"error": None, "declared_params": {}, "fitted_params": {}, "loss_before": None, "loss_after": None,
              "n_training": len(training)}
    loss_fn = _LOSS_FNS.get(world)
    if loss_fn is None:
        result["error"] = f"no loss function for world {world!r}"
        return result
    try:
        law = EV._compile_law(law_source)
    except Exception as exc:  # noqa: BLE001 - recorded
        result["error"] = f"compile_error: {type(exc).__name__}: {exc}"[:300]
        return result
    fit_fn = EV._compile_fit_parameters(law_source)
    spec = []
    if fit_fn is not None:
        try:
            spec = EV._validate_fit_spec(fit_fn())
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"invalid_fit_parameters: {type(exc).__name__}: {exc}"[:300]
            return result
        result["declared_params"] = {name: {"init": init, "bounds": list(bounds)} for name, init, bounds in spec}
    init = {name: i for name, i, _b in spec}
    result["fitted_params"] = dict(init)
    if not training:
        result["fit"] = "none (no experiments yet; initial constants kept)"
        return result
    try:
        result["loss_before"] = _finite(loss_fn(functools.partial(law, **init) if init else law, training))
    except Exception as exc:  # noqa: BLE001
        result["loss_before"] = None
        result["loss_before_error"] = f"{type(exc).__name__}: {exc}"[:200]
    if not spec:
        result["fit"] = "harvest_all (no fit_parameters(); law scored as written)"
        result["loss_after"] = result["loss_before"]
        return result
    try:
        fitted = EV._fit_law_parameters(law, spec, training, loss_fn)
        result["fitted_params"] = {k: float(v) for k, v in fitted.items()}
        result["loss_after"] = _finite(loss_fn(functools.partial(law, **fitted), training))
        result["fit"] = "harvest_all"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"optimizer_failure: {type(exc).__name__}: {exc}"[:300]
        result["fit"] = "failed; initial constants kept"
    return result


def _finite(x):
    try:
        x = float(x)
    except Exception:  # noqa: BLE001
        return None
    return x if x == x and abs(x) != float("inf") else None


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def harvest(run_json, out_path, world, budget):
    t0 = time.time()
    run = json.load(open(run_json))
    rounds = sorted(run.get("rounds", []), key=lambda r: r.get("round", 0))
    rows, executed = {}, 0
    log = []   # one entry per round: what the harvest saw
    for r in rounds:
        b = executed
        src = extract_tag(r.get("llm_reply") or "")
        pairs = executed_pairs(r)
        log.append({"round": r.get("round"), "action": r.get("action"), "b_before": b, "executed_here": len(pairs),
                    "stated": src is not None, "experiment_error": bool(r.get("experiment_error"))})
        if src is not None and b <= budget:
            src = src.strip("\n")
            training = training_from_rounds(rounds, int(r["round"]))
            assert len(training) == b, (len(training), b)
            fit = fit_constants(src, world, training)
            prev = rows.get(b)
            rows[b] = {"b": b, "round": r["round"], "status": "stated", "law": src, "law_sha256": _sha(src),
                       "statements": (prev["statements"] + 1) if prev and prev["status"] == "stated" else 1, **fit}
        executed += len(pairs)
    final_round = next((r for r in rounds if r.get("action") == "final_law"), None)
    final_src = (run.get("final_law") or (final_round or {}).get("final_law") or "").strip("\n")
    results_path = run_json[:-len(".json")] + ".results.json"
    final = {"b": budget, "round": final_round["round"] if final_round else None, "status": "final", "law": final_src or None,
             "law_sha256": _sha(final_src) if final_src else None, "executed_total": executed,
             "also_stated_at_b": rows.get(budget, {}).get("round")}
    if final_src:
        final.update(fit_constants(final_src, world, training_from_rounds(rounds, 10 ** 9)))
    else:
        final["status"] = "no_final_law"
    final["dp_final_fit"] = "no results.json"
    final["dp_final_fitted_params"] = None
    if os.path.exists(results_path):
        res = json.load(open(results_path))
        fit = (res.get("evaluation") or {}).get("fit") or {}
        final["dp_final_fitted_params"] = (fit.get("fitted_params") if isinstance(fit, dict) else None) or {}
        final["dp_final_fit"] = ("DP end-of-mission fit (results.json; _FIT_WORLDS only, 4 x 5 subsample)" if fit.get("fitted_params")
                                 else f"DP did not fit: {fit.get('error') or 'no fit block'}")
        if res.get("law") and final_src and res["law"].strip("\n") != final_src:
            final["note"] = "results.json law differs from the run log's final_law; the run log's is used"
    rows[budget] = final
    for b in range(0, budget + 1):
        rows.setdefault(b, {"b": b, "round": None, "status": "missing", "law": None})
    out = {"schema": SCHEMA, "world": world, "budget": budget, "run_json": os.path.basename(run_json),
           "rule": ("b = executed experiments before the statement; every row (b = B included) refitted on all executed "
                    "experiments with DP's _fit_law_parameters (harvest_all); DP's own final constants recorded as dp_final_*"),
           "executed_total": executed, "stated": sum(1 for r in rows.values() if r["status"] == "stated"),
           "missing": sum(1 for r in rows.values() if r["status"] == "missing"),
           "fit_errors": sum(1 for r in rows.values() if r.get("error")),
           "harvest_seconds": round(time.time() - t0, 1), "rounds_seen": log,
           "rows": [rows[b] for b in sorted(rows)]}
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, out_path)
    return {k: out[k] for k in ("executed_total", "stated", "missing", "fit_errors", "harvest_seconds")}


if __name__ == "__main__":
    print(json.dumps(harvest(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))))
