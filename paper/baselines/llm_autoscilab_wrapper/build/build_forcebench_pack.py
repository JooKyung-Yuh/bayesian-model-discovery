#!/usr/bin/env python3
"""Build the ForceBench truth pack for the AutoSciLab force oracle (the development host only, 0 USD, no LLM).

Per world: the fixed IC (pos0, vel0, measurement times), VAR_GT, inertia and softening, the scoring
designs Q_HEAD + Q_SHIFT and their noise-free truth series - all taken from score_dp_ourq.py's own
world_obj() (the frozen ICL tree's World(world, 31), the arrays the DP-agent and ICL baselines are
scored against) - plus the force-law constants, sigma_F, our D0 knobs for seeds 31/37/41 and floors.

Checks written into the pack:
  scorer_parity   score_dp_ourq.scores() on the kit integrator's forecast == the kit's own nmse
  law_check       the stationary-particle law scored by the kit equals LAW_CHECK_v2.json
  floors          the kit score of the oracle's own law (and of the declared law for gravity/yukawa)

Usage (the ForceBench environment):
  env -u FORCE_COLOR PYTHONDONTWRITEBYTECODE=1 python -B build_forcebench_pack.py <out.json>
"""
import hashlib
import importlib.util
import json
import math
import os
import sys

import numpy as np

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORER = "<development path>"
LAW_CHECK = "<development path>"
WORLDS = ("gravity", "yukawa", "coulomb_easy", "oscillator", "extra_dimensions")
SEEDS = (31, 37, 41)


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    out = sys.argv[1]
    spec = importlib.util.spec_from_file_location("score_dp_ourq", SCORER)
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)          # sets DP paths and chdir(ROOT) as the scorer always does
    sys.path.insert(0, KIT)
    from asl_kit import forcebench as F  # noqa: E402

    law_check = json.load(open(LAW_CHECK))
    pack = {"schema": "asl_forcebench_truth_pack_v1", "scorer": SCORER, "scorer_sha256": sha256(SCORER),
            "icl_tree": S.ROOT, "r_range": F.R_RANGE, "p_range": F.P_RANGE, "t_range_oscillator": F.T_RANGE,
            "worlds": {}}
    for world in WORLDS:
        w = S.world_obj(world)
        rg = w.rg
        cfg = rg.WORLD_CFG[world]
        consts = {}
        if world == "gravity":
            consts = {"A": float(rg._GRAV_EFC["A"]), "B": float(rg._GRAV_EFC["B"]), "source": rg._GRAV_EFC["source"]}
        elif world == "yukawa":
            consts = {"A": float(rg._YUK_EFC["A"]), "source": rg._YUK_EFC["source"]}
        truth_numerics = rg.fwd.TRUTH_NUMERICS[world]
        if truth_numerics["method"] != F.TRUTH_METHOD[world] or truth_numerics["dt"] != F.TRUTH_DT:
            raise SystemExit(f"{world}: kit stepper {F.TRUTH_METHOD[world]} differs from TRUTH_NUMERICS {truth_numerics}")
        wp = {"times": [float(t) for t in rg.TIMES], "pos0": [float(v) for v in np.asarray(rg.POS0)],
              "vel0": [float(v) for v in np.asarray(rg.VEL0)], "var_gt": float(rg.VAR_GT),
              "inertia": cfg["inertia"], "soft": float(cfg["soft"]), "engine": rg.IC["engine"],
              "method": F.TRUTH_METHOD[world], "dt": F.TRUTH_DT, "truth_numerics": truth_numerics,
              "q_head": [[float(a), float(b)] for a, b in rg.Q_HEAD],
              "q_shift": [[float(a), float(b)] for a, b in rg.Q_SHIFT],
              "truth": np.asarray(w.truth, dtype=float).tolist(), "law_constants": consts,
              "world_cfg_law_str": cfg["law_str"], "sigma_trajectory": float(rg.SIGMA),
              "d0_knobs": {str(s): np.asarray(rg._initial_designs(np.random.default_rng(s)), float).tolist() for s in SEEDS}}
        # sigma_F = 0.05 * sqrt(var of F along the nine noise-free scoring trajectories)
        law = F.force_law(world, consts)
        timed = world == "oscillator"
        vals = []
        for (p1, p2), series in zip(wp["q_head"] + wp["q_shift"], wp["truth"]):
            for t, (x, y) in zip(wp["times"], series):
                r = math.hypot(x, y)
                vals.append(law(r, p1, p2, t) if timed else law(r, p1, p2))
        wp["var_f"] = float(np.var(vals))
        wp["sigma_f"] = 0.05 * math.sqrt(wp["var_f"])
        wp["force_range_on_scoring_trajectories"] = [float(min(vals)), float(max(vals))]
        # floors and parity with score_dp_ourq.scores
        pred = F.forecast(law, wp, timed)
        mine = F.scores_of_forecast(pred, wp)
        theirs = S.scores(pred, w)
        wp["scorer_parity"] = {k: {"kit": mine[k], "score_dp_ourq": theirs[k], "equal": mine[k] == theirs[k]} for k in mine}
        wp["floor_oracle_law"] = mine
        if world in ("gravity", "yukawa"):
            wp["floor_declared_law"] = F.scores_of_forecast(F.forecast(F.force_law(world, consts, "declared"), wp, False), wp)
        # the same oracle law through run_gravity._integrate_true's velocity Verlet, for the record
        wp["floor_oracle_law_leapfrog_reference"] = F.scores_of_forecast(
            np.asarray([F.integrate(law, wp, p1, p2, timed, "leapfrog") for p1, p2 in wp["q_head"] + wp["q_shift"]]), wp)
        still = np.broadcast_to(np.asarray(wp["pos0"], float), np.asarray(wp["truth"]).shape)
        ls = F.scores_of_forecast(still, wp)
        wp["law_check"] = {k: {"kit": ls[k], "LAW_CHECK_v2": law_check[world][k], "equal": abs(ls[k] - law_check[world][k]) <= 1e-12 * max(1.0, abs(ls[k]))}
                           for k in ls}
        pack["worlds"][world] = wp
        print(world, "sigma_f", wp["sigma_f"], "floor", mine, "declared", wp.get("floor_declared_law"),
              "parity", all(v["equal"] for v in wp["scorer_parity"].values()),
              "law_check", all(v["equal"] for v in wp["law_check"].values()), flush=True)
    with open(out, "x") as fh:
        json.dump(pack, fh, separators=(",", ":"))
    print("wrote", out, sha256(out))


if __name__ == "__main__":
    main()
