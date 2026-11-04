#!/usr/bin/env python3
"""The kit's world construction against the engine's own, cell by cell (0 USD; needs an engine tree, else SKIP).

  APPR_ENGINE_TREE=<frozen tree with tools/icl_boxinggym, default main_verify_iclb_a75c32c> \
  env -u FORCE_COLOR PYTHONDONTWRITEBYTECODE=1 <venv python> -B tests/test_engine_equality.py
Death, dugongs, peregrines, hyperbolic: tools.icl_boxinggym.icl_boxinggym.World(task, seed) (the design's reference):
world parameters, the 256 evaluation points and held-out outcomes, the noise-free truth, and the D0 observations of
World.start() must equal the kit's World (worlds.py) exactly. Survival: the ICL World would re-draw the 256 evaluation
points with the goal's rejection sampler (tens of minutes), so the engine's TaskEnvironment is built the way World
builds it (env_seed_injection(world_seed), TaskEnvironment(spec, world_seed)) and its world parameters, patient records
and the D0 observations through TaskEnvironment.observe are compared.
"""
import contextlib
import os
import sys
import tempfile

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
sys.dont_write_bytecode = True
os.environ.setdefault("PYTENSOR_FLAGS", "base_compiledir=" + tempfile.mkdtemp(prefix="appr_engine_test_ptc_"))
TREE = os.environ.get("APPR_ENGINE_TREE", "<development path>")

from appr_kit import paths, worlds  # noqa: E402

paths.import_checkout()
import numpy as np  # noqa: E402


def main():
    if not os.path.isfile(os.path.join(TREE, "tools", "icl_boxinggym", "icl_boxinggym.py")):
        print(f"SKIP engine tree not found: {TREE}")
        print("RESULT SKIP")
        return 0
    sys.path[:0] = [TREE, os.path.join(TREE, "src")]
    from tools.icl_boxinggym import icl_boxinggym as I
    bg = I.bg
    print(f"engine {bg.__file__}")
    failed = 0
    cells = [(t, s) for t in ("death", "dugongs", "peregrines", "hyperbolic") for s in (31, 37, 41)] + \
            [("survival", s) for s in (43, 53, 61)]
    for task, seed in cells:
        with contextlib.ExitStack() as stack:
            ours = worlds.World(task, seed).build(stack)
            mine = {"wp": worlds.world_parameters(ours.env), "d0": [float(r["value"]) for r in ours.d0_records],
                    "d0x": [r["executed"] for r in ours.d0_records],
                    "qx": [[float(p[i]) for i in worlds.TASKS[task][3]] for p in ours.goal.eval_points[:256]],
                    "qy": [float(p[-1]) for p in ours.goal.eval_points[:256]]}
            records = [[int(o), float(t), int(m)] for o, t, m in ours.env.outcomes] if task == "survival" else None
        checks = {}
        if task != "survival":
            w = I.World(task, seed)
            try:
                obs = w.start()
                rows = w.evaluation_rows
                checks["world_parameters"] = w.environment.world_parameters() == mine["wp"]
                checks["evaluation_x"] = np.array_equal(np.asarray(rows, dtype=float), np.asarray(mine["qx"]))
                checks["heldout"] = np.array_equal(np.asarray(w.heldout, dtype=float), np.asarray(mine["qy"]))
                checks["truth_vs_pack"] = np.array_equal(np.asarray(w.truth, dtype=float), ours.truth)
                checks["d0_y"] = [o.y for o in obs] == mine["d0"]
                checks["d0_x"] = [list(o.x) for o in obs] == mine["d0x"]
            finally:
                w.close()
        else:
            spec = bg.TASKS[task]
            entry = ours.entry
            with bg.env_seed_injection(int(entry["world_seed"])):
                environment = bg.TaskEnvironment(spec, int(entry["world_seed"]))
                ys = [environment.observe(h) for h in entry["d0"]["handles"]]
                checks["world_parameters"] = environment.world_parameters() == mine["wp"]
                checks["records"] = [[int(o), float(t), int(m)] for o, t, m in environment.env.outcomes] == records
                checks["d0_y"] = [float(y) for y in ys] == mine["d0"]
        bad = [k for k, v in checks.items() if not v]
        failed += bool(bad)
        print(("PASS" if not bad else "FAIL"), task, seed, " ".join(f"{k}={v}" for k, v in checks.items()))
    print("RESULT", "PASS" if not failed else f"FAIL ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
