"""Per-round scores of `--protocol current_law` DP-Agent cells on OUR ForceBench designs (the development host, 0 USD).

Reuses the final-law scorer as a module (its world objects, forecast and nmse rules; nothing re-implemented):
  --scorer  path of score_dp_ourq.py (default: the the development host analysis folder)
For every <stem>.current_laws.json in the given folders it writes one row per b = 0..B:
  status scored            the law stated at that b, constants from dp_current_law's harvest rule (refit on all executed
                           experiments so far), forecast on the 9 designs
  status carried           no <current_law> at that b: the previous scored b's law and scores, carried_from = b'
  status no_law            no law has been stated yet (b before the first stated law)
  status law_error         the law raised or gave a non-finite forecast on our designs (not repaired)
b = B is DP's final law refitted by the same harvest rule (one rule for the whole ER curve). The as-released point - the final
law with DP's own end-of-mission constants, what score_dp_ourq scores - is reported beside it as `as_released_final`.
Usage: score_dp_rounds.py <runs folder> [<runs folder> ...] --out <scores.json> [--scorer <score_dp_ourq.py>]
"""
import argparse
import functools
import glob
import importlib.util
import json
import os
import sys

DEFAULT_SCORER = "<development path>"


def load_scorer(path):
    spec = importlib.util.spec_from_file_location("score_dp_ourq", path)
    module = importlib.util.module_from_spec(spec)
    sys.argv = [path]            # the scorer module parses nothing at import; keep argv clean anyway
    spec.loader.exec_module(module)
    return module


def score_law(SC, w, law_src, fitted):
    import numpy as np
    try:
        law = SC._compile_law(law_src)
        if fitted:
            law = functools.partial(law, **fitted)
        pred = SC.forecast(law, w)
        if pred.shape != w.truth.shape or not np.isfinite(pred).all():
            return {"status": "law_error", "error": f"shape {pred.shape} finite {bool(np.isfinite(pred).all())}"}
        return {"status": "scored", **SC.scores(pred, w)}
    except Exception as exc:  # noqa: BLE001 - a law that fails on our designs is recorded, not repaired
        return {"status": "law_error", "error": f"{type(exc).__name__}: {exc}"[:500]}


def score_cell(SC, path):
    cl = json.load(open(path))
    w = SC.world_obj(cl["world"])
    stem = path[:-len(".current_laws.json")]
    rows, last = [], None
    for r in cl["rows"]:
        base = {"b": r["b"], "round": r.get("round"), "law_status": r["status"], "fit": r.get("fit"), "fit_error": r.get("error"),
                "n_training": r.get("n_training"), "law_sha256": r.get("law_sha256"), "statements": r.get("statements")}
        if r["status"] in ("stated", "final") and r.get("law"):
            s = score_law(SC, w, r["law"], r.get("fitted_params") or {})
            row = {**base, **s, "fitted_params": r.get("fitted_params") or {}}
            if r["status"] == "final":
                # the as-released point: DP's own end-of-mission constants (score_dp_ourq's final-law row)
                dp = score_law(SC, w, r["law"], r.get("dp_final_fitted_params") or {})
                row["as_released_final"] = {**dp, "dp_final_fit": r.get("dp_final_fit"),
                                            "dp_final_fitted_params": r.get("dp_final_fitted_params")}
            if s["status"] == "scored":
                last = row
        elif last is not None:
            row = {**base, **{k: last[k] for k in ("status", "nmse_head", "nmse_shift", "nmse_all", "law_sha256", "fitted_params")},
                   "status": "carried", "carried_from": last["b"]}
        else:
            row = {**base, "status": "no_law"}
        rows.append(row)
    pr = json.load(open(stem + ".protocol.json")) if os.path.exists(stem + ".protocol.json") else {}
    us = json.load(open(stem + ".usage.json")) if os.path.exists(stem + ".usage.json") else {}
    return {"stem": os.path.basename(stem), "world": cl["world"], "budget": cl["budget"], "seed": pr.get("seed"),
            "schema": cl.get("schema"), "rule": cl.get("rule"), "executed_total": cl.get("executed_total"),
            "mode": pr.get("mode"), "cell_status": pr.get("status"), "experiments_run": pr.get("experiments_run"),
            "usd_estimate": us.get("usd_estimate_total"), "calls": us.get("calls"),
            "stated": cl["stated"], "missing": cl["missing"], "fit_errors": cl["fit_errors"], "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scorer", default=DEFAULT_SCORER)
    a = ap.parse_args()
    SC = load_scorer(a.scorer)
    cells = []
    for folder in a.folders:
        for p in sorted(glob.glob(os.path.join(folder, "*.current_laws.json"))):
            cells.append(score_cell(SC, p))
    payload = {"scorer": a.scorer, "scorer_sha256": SC.sha(a.scorer), "rounds_scorer_sha256": SC.sha(__file__),
               "folders": a.folders, "cells": cells}
    json.dump(payload, open(a.out, "w"), indent=1)
    for c in cells:
        counts = {}
        for r in c["rows"]:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        print(c["stem"], c["cell_status"], "stated", c["stated"], "missing", c["missing"], counts,
              "final nmse_shift", next((r.get("nmse_shift") for r in reversed(c["rows"]) if r["status"] == "scored"), None))


if __name__ == "__main__":
    main()
