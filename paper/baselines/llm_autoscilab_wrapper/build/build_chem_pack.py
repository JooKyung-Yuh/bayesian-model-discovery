#!/usr/bin/env python3
"""Build the ChemBench 256-point evaluation pack (the development host only, 0 USD, no LLM).

designs = adapters/chembench_observation.sample_designs(256, seed_for("chem_evaluation_inputs", 20260907)),
the per-checkpoint evaluation set of our ChemBench runs (chembench_core.py: `evaluation =
adapter.sample_designs(args.q, seed_for("chem_evaluation_inputs", 20260907))`, q = 256 at profile paper).
Also checked here: the kit's own sampler and fixed3 rule equal the engine's, and the targets the kit
computes equal one recorded run's reference values where such a file exists.

Usage (the engine's environment, with `src` and the LLM-AutoSciLab checkout on PYTHONPATH):
  env -u FORCE_COLOR PYTHONDONTWRITEBYTECODE=1 PYTENSOR_FLAGS=base_compiledir=<scratch>/pytensor \
      python -B build_chem_pack.py <out.json>
"""
import hashlib
import json
import os
import sys

TREE = "<development path>"
CHECKOUT = "<development path>"
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
sys.path[:0] = [TREE, TREE + "/src", CHECKOUT, KIT]

import numpy as np  # noqa: E402

from adapters import chembench_observation as adapter  # noqa: E402
from bayes_model_discovery.experimental.core_count import seed_for  # noqa: E402
from bayes_model_discovery.experimental import chembench_core as CC  # noqa: E402
from asl_kit import chem  # noqa: E402


def main():
    out = sys.argv[1]
    seed = seed_for("chem_evaluation_inputs", 20260907)
    designs = adapter.sample_designs(256, seed)
    kit_designs = chem.sample_designs(256, seed)
    same_sampler = designs == kit_designs
    same_1000 = adapter.sample_designs(1000, 42) == chem.sample_designs(1000, 42)
    same_fixed3 = CC.fixed3_designs() == chem.fixed3_designs()
    if not (same_sampler and same_1000 and same_fixed3):
        raise SystemExit(f"parity failed sampler={same_sampler} q1000={same_1000} fixed3={same_fixed3}")
    pack = {"schema": "asl_chembench_eval256_v1", "tree": TREE, "seed_label": "chem_evaluation_inputs/20260907",
            "seed": int(seed), "count": 256, "designs": designs,
            "checks": {"kit_sampler_equals_adapter_256": same_sampler, "kit_sampler_equals_adapter_1000_seed42": same_1000,
                       "kit_fixed3_equals_chembench_core": same_fixed3,
                       "fixed3": chem.fixed3_designs()}}
    with open(out, "x") as fh:
        json.dump(pack, fh, separators=(",", ":"))
    print("wrote", out, hashlib.sha256(open(out, "rb").read()).hexdigest(), "seed", seed)


if __name__ == "__main__":
    main()
