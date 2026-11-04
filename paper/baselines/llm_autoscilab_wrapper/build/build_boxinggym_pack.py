#!/usr/bin/env python3
"""Build the BoxingGym world pack for the AutoSciLab adapter (the development host only, 0 USD, no LLM).

The pack holds, per (task, seed), exactly what our BoxingGym runs fixed before round 1:
the world parameters, the evaluation set Q (|Q| = 256, the goal's own held-out questions),
the noise-free mean E[Y|x] at Q, the held-out outcomes at Q, the design pool, the two D0
designs (D0 policy random2) and the two D0 observations.

How each item is produced (all through OUR engine's own functions, frozen tree 5d88aaf, the
tree that ran the 10/2 cohort and the 10/4 survival re-run; the functions used here are
byte-identical to the B1 tree efa5bad, checked by AST hash):
  world_seed      = seed_for("boxinggym_world", task, seed)
  evaluation_seed = seed_for("boxinggym_evaluation", task, seed, "paper")
  under env_seed_injection(world_seed):
    TaskEnvironment(spec, world_seed)            -> world parameters (and survival's 100 records)
    evaluation_points(256, evaluation_seed)      -> Q, held-out outcomes
    spec.truth_mean(environment, Q)              -> noise-free mean at Q
    build_design_pool(spec, env, design_pool_target(spec, 100, rounds=30)) -> pool
    select_d0_designs(spec, pool, "random2", seed)                         -> D0 designs
D0 observations are NOT re-simulated: they are read from the recorded round-0 lines of our
own runs (observations.jsonl), and every recorded arm of the same seed must agree, and the
recorded D0 x must equal the D0 designs computed here. Q and the pool are cross-checked
against each recorded cell's evaluation_scoring_only.npz when that cell excluded no Q point.

--q-from-records: survival's Q is read from a recorded cell that excluded no Q point instead of being
re-drawn (the survival goal re-draws held-out patients by rejection sampling until the outcomes are
balanced - survival_analysis.py:28-46 - which costs tens of minutes per world; the recorded array is that
same draw, written by run() before any observation). Every other task's Q is regenerated and compared.

Usage (the development host):
  env -u FORCE_COLOR PYTHONDONTWRITEBYTECODE=1 PYTENSOR_FLAGS=base_compiledir=<scratch>/pytensor \
    python -B build_boxinggym_pack.py <out.json> [--q-from-records]
"""
import contextlib
import glob
import hashlib
import json
import os
import sys

TREE = "<development path>"
BOXING_GYM_SRC = "<development path>"
sys.path[:0] = [TREE + "/src", BOXING_GYM_SRC]

import numpy as np  # noqa: E402

from bayes_model_discovery.experimental import boxinggym_core as B  # noqa: E402
from bayes_model_discovery.experimental.core_count import seed_for  # noqa: E402

TASK_SEEDS = {
    "death": (31, 37, 41),
    "dugongs": (31, 37, 41),
    "peregrines": (31, 37, 41),
    "hyperbolic": (31, 37, 41),
    # protocol U: survival runs on the new worlds 43/53/61 (old 31/37/41 worlds were flat)
    "survival": (43, 53, 61),
}
PROFILE, Q_COUNT, GRID_SIZE, ROUNDS = "paper", 256, 100, 30
R10 = "<development path>"
RECORD_GLOBS = {
    "death": [R10 + "/final_20261002_boxinggym_paper_masked_mda*/a compute host/*__death__*__s{seed}/cell"],
    "dugongs": [R10 + "/final_20261002_boxinggym_paper_masked_mda*/a compute host/*__dugongs__*__s{seed}/cell"],
    "peregrines": [R10 + "/final_20261002_boxinggym_paper_masked_mda/a compute host/*__peregrines__*__s{seed}/cell",
                   R10 + "/final_20261002_boxinggym_paper_masked_mda_random/a compute host/*__peregrines__*__s{seed}/cell"],
    "hyperbolic": [R10 + "/final_20261002_boxinggym_paper_masked_mda*/a compute host/*__hyperbolic__*__s{seed}/cell",
                   "<development path>*__wide__s{seed}/cell"],
    "survival": ["<development path>*__default__s{seed}/cell"],
}


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def as_list(a):
    return np.asarray(a, dtype=float).tolist()


def recorded_q(task, seed):
    """Q and held-out outcomes from a recorded cell of this task+seed that excluded no Q point (else None)."""
    for pattern in RECORD_GLOBS[task]:
        for cell in sorted(glob.glob(pattern.format(seed=seed))):
            cfg = json.load(open(os.path.join(cell, "config.json")))
            if int(cfg.get("q_excluded_count") or 0) == 0:
                npz = np.load(os.path.join(cell, "evaluation_scoring_only.npz"))
                return np.asarray(npz["x"], float), np.asarray(npz["y"], float), cell
    return None


def build_one(task, seed, q_from_records=False):
    spec = B.TASKS[task]
    world_seed = seed_for("boxinggym_world", spec.name, seed)
    evaluation_seed = seed_for("boxinggym_evaluation", spec.name, seed, PROFILE)
    design_pool_seed = seed_for("boxinggym_design_pool", spec.name, seed)
    q_source = "regenerated: TaskEnvironment.evaluation_points(256, evaluation_seed)"
    with contextlib.ExitStack() as stack:
        stack.enter_context(B.env_seed_injection(world_seed))
        environment = B.TaskEnvironment(spec, world_seed)
        rec = recorded_q(task, seed) if q_from_records else None
        if rec is not None:
            # survival's goal draws its held-out patients by rejection sampling until the outcomes are
            # balanced (survival_analysis.py:28-46), tens of minutes per world; the recorded Q of an
            # unexcluded cell IS that draw (written by the same run() before any observation)
            ex, heldout, cell = rec
            q_source = f"recorded: {cell}/evaluation_scoring_only.npz (q_excluded_count 0)"
        else:
            ex, heldout = environment.evaluation_points(Q_COUNT, evaluation_seed)
        truth = spec.truth_mean(environment, ex)
        if spec.environment_pool_source == "sampled":
            np.random.seed(design_pool_seed)
        pool = B.build_design_pool(spec, environment, B.design_pool_target(spec, GRID_SIZE, ROUNDS))
        d0 = B.select_d0_designs(spec, pool, "random2", seed)
        params = environment.world_parameters()
        env = environment.env
        extra = {}
        if task == "death":
            extra = {"N": int(env.N), "theta": float(env.theta)}
        elif task == "dugongs":
            extra = {"a": float(env.a), "b": float(env.b), "l": float(env.l), "sigma_obs": 0.25}
        elif task == "peregrines":
            extra = {"alpha": float(env.alpha), "beta1": float(env.beta1), "beta2": float(env.beta2),
                     "beta3": float(env.beta3)}
        elif task == "hyperbolic":
            k, alpha = env.truth
            extra = {"k": float(k), "alpha": float(alpha), "epsilon": float(env.epsilon)}
        elif task == "survival":
            extra = {"lambda0": float(env.lambda0), "beta": float(env.beta),
                     # [outcome, time, metastasized] per patient, the environment's own stored records
                     "records": [[int(r[0]), float(r[1]), int(r[2])] for r in env.outcomes]}
    rows = np.asarray(pool.rows, dtype=float)
    d0_rows = [as_list(np.atleast_1d(np.asarray(row, dtype=float))) for row, _handle in d0]
    d0_handles = [str(handle) for _row, handle in d0]
    entry = {
        "task": task, "seed": seed, "profile": PROFILE,
        "world_seed": int(world_seed), "evaluation_seed": int(evaluation_seed),
        "design_pool_seed": int(design_pool_seed),
        "input_names": list(spec.input_names),
        "input_ranges": [list(map(float, item["range"])) for item in spec.inputs],
        "design_range": list(map(float, spec.design_range)) if spec.design_range else None,
        "integer_inputs": list(spec.integer_inputs),
        "outcome_type": spec.outcome_type, "support": list(spec.support), "trials": spec.trials,
        "world_parameters_engine": params, "world": extra,
        "q": {"x": as_list(ex if ex.ndim == 2 else ex[:, None]), "truth_mean": as_list(truth),
              "heldout": as_list(heldout), "source": q_source},
        "pool_rows": as_list(rows), "pool_handles": list(map(str, pool.handles)),
        "d0": {"x": d0_rows, "handles": d0_handles},
    }
    return entry


def crosscheck(entry):
    """Recorded cells of the same task+seed: D0 (x, y), world parameters, Q, pool."""
    task, seed = entry["task"], entry["seed"]
    cells = sorted({c for pattern in RECORD_GLOBS[task] for c in glob.glob(pattern.format(seed=seed))})
    if not cells:
        raise SystemExit(f"no recorded cell for {task} s{seed}")
    report, d0_y = [], None
    single = len(entry["input_names"]) == 1
    for cell in cells:
        cfg = json.load(open(os.path.join(cell, "config.json")))
        rows = [json.loads(line) for line in open(os.path.join(cell, "observations.jsonl")) if line.strip()]
        r0 = [r for r in rows if r.get("round") == 0]
        x_rec = [[float(r["x"])] if single else [float(v) for v in r["x"]] for r in r0]
        y_rec = [float(r["y"]) for r in r0]
        ok_x = np.allclose(np.asarray(x_rec), np.asarray(entry["d0"]["x"]), rtol=0, atol=0)
        ok_seed = int(cfg["world_seed"]) == entry["world_seed"] and int(cfg["evaluation_seed"]) == entry["evaluation_seed"]
        wp = cfg.get("world_parameters") or {}
        ok_params = all(np.allclose(np.asarray(v, float), np.asarray(entry["world_parameters_engine"][k], float), rtol=0, atol=0)
                        for k, v in wp.items() if k in entry["world_parameters_engine"])
        npz = np.load(os.path.join(cell, "evaluation_scoring_only.npz"))
        q_excluded = int(cfg.get("q_excluded_count") or 0)
        qx = np.asarray(entry["q"]["x"], float)
        rec_x = np.asarray(npz["x"], float)
        rec_x = rec_x[:, None] if rec_x.ndim == 1 else rec_x
        ok_q = (q_excluded > 0) or (rec_x.shape == qx.shape and np.array_equal(rec_x, qx)
                                    and np.array_equal(np.asarray(npz["y"], float), np.asarray(entry["q"]["heldout"], float)))
        ok_pool = np.array_equal(np.asarray(npz["designs"], float).reshape(len(entry["pool_rows"]), -1),
                                 np.asarray(entry["pool_rows"], float))
        if d0_y is None:
            d0_y = y_rec
        ok_y = y_rec == d0_y
        report.append({"cell": cell, "config_sha256": sha256(os.path.join(cell, "config.json")),
                       "observations_sha256": sha256(os.path.join(cell, "observations.jsonl")),
                       "d0_x_equal": bool(ok_x), "d0_y_equal_across_cells": bool(ok_y),
                       "seeds_equal": bool(ok_seed), "world_parameters_equal": bool(ok_params),
                       "q_equal": ("skipped: q_excluded_count=%d" % q_excluded) if q_excluded else bool(ok_q),
                       "pool_equal": bool(ok_pool), "d0_y": y_rec})
        if not (ok_x and ok_y and ok_seed and ok_params and ok_q and ok_pool):
            raise SystemExit(f"cross-check failed for {task} s{seed}: {report[-1]}")
    if task == "survival":
        # the recorded D0 outcome must be the stored record of the D0 patient
        recs = entry["world"]["records"]
        for handle, y in zip(entry["d0"]["handles"], d0_y):
            if float(recs[int(handle)][0]) != float(y):
                raise SystemExit(f"survival D0 outcome mismatch {handle}")
    entry["d0"]["y"] = d0_y
    entry["crosscheck"] = report
    return entry


def main():
    out = sys.argv[1]
    # --q-from-records: take survival's Q from a recorded unexcluded cell instead of re-running the
    # goal's balanced rejection sampler (every other task regenerates Q and is cross-checked)
    q_rec_tasks = {"survival"} if "--q-from-records" in sys.argv[2:] else set()
    pack = {"schema": "asl_boxinggym_world_pack_v1", "tree": TREE, "boxing_gym_src": BOXING_GYM_SRC,
            "tree_boxinggym_core_sha256": sha256(TREE + "/src/bayes_model_discovery/experimental/boxinggym_core.py"),
            "q_count": Q_COUNT, "grid_size": GRID_SIZE, "rounds_for_pool_target": ROUNDS,
            "d0_policy": "random2", "q_policy": "exclude_observed", "q_from_records_tasks": sorted(q_rec_tasks),
            "entries": []}
    for task, seeds in TASK_SEEDS.items():
        for seed in seeds:
            entry = crosscheck(build_one(task, seed, q_from_records=task in q_rec_tasks))
            pack["entries"].append(entry)
            print(task, seed, "d0", entry["d0"]["x"], entry["d0"]["y"], "cells", len(entry["crosscheck"]), flush=True)
    with open(out, "x") as fh:
        json.dump(pack, fh, indent=None, separators=(",", ":"))
    print("wrote", out, sha256(out))


if __name__ == "__main__":
    main()
