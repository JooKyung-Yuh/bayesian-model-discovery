"""ChemBench: the native AutoSciLab oracle (leak-blocked), our D0 (fixed3) and our scorers.

World. ChemBenchOracle(domain, difficulty, law_version="v0", noise_level=0.01,
grammar_mode="universal") - the same class, noise level and grammar mode our engine's
ChemBenchSession uses (adapters/chembench_observation.py), so an identical design gives an
identical observation (the native noise is SHA-256-seeded per design). The released budget
runner uses noise 0.0; we use 0.01 because that is the world our runs observe.

D0 (fixed3, our rule; chembench_core.fixed3_designs): C_A in {0.03, 0.3, 3.0} with C_I 0, C_P 0,
C_B and Enz at the geometric middle of their bounds, T 310 K, pH 7.

Scores of a law string (after the run, by an unblocked scorer instance):
  official_rmsle_1000  native evaluate_law (1,000 points, sampler seed 42, clean primary rate r0,
                       native |.| + finite mask); official_exact = native rule rmsle < 0.05
  nmsle_1000_primary   official_rmsle^2 / Var_Q1000[log1p r0]   (MDA Eq. 50, as the 9/6 no-leak runs)
  nmsle_256_noiseless_observation, nmsle_256_primary
                       our 256-point set (sample_designs(256, seed_for("chem_evaluation_inputs",
                       20260907)), from packs/chembench_eval256.json), squared log1p error / Var_Q256
                       of the target, the two targets of our figures; predictions get the native
                       evaluate_law convention (|.|, non-finite points masked, count recorded)
"""
import json
import math
import os

import numpy as np

from .paths import PACK_DIR

NOISE_LEVEL = 0.01
GRAMMAR_MODE = "universal"
FIXED3_CONCENTRATIONS = (0.03, 0.3, 3.0)


def fixed3_designs():
    from autoscilab.oracle.chembench import CHEM_INPUT_BOUNDS, _T_REF
    values = {"C_I": 0.0, "C_P": 0.0,
              "C_B": float(np.sqrt(np.prod(CHEM_INPUT_BOUNDS["C_B"]))),
              "Enz": float(np.sqrt(np.prod(CHEM_INPUT_BOUNDS["Enz"]))),
              "T": float(_T_REF), "pH": 7.0}
    return [{"C_A": c, **{k: values[k] for k in sorted(values)}} for c in FIXED3_CONCENTRATIONS]


def sample_designs(count, seed):
    """Byte-for-byte the sampler of adapters/chembench_observation.py (count=1000, seed=42 = evaluate_law's)."""
    from autoscilab.oracle.chembench import CHEM_INPUT_BOUNDS, CHEM_LOG_VARS
    rng = np.random.default_rng(seed)
    columns = {}
    for name, (lo, hi) in CHEM_INPUT_BOUNDS.items():
        columns[name] = (np.exp(rng.uniform(np.log(lo), np.log(hi), count))
                         if name in CHEM_LOG_VARS else rng.uniform(lo, hi, count))
    return [{name: float(values[i]) for name, values in columns.items()} for i in range(count)]


def make_oracle(domain, difficulty, law_version="v0"):
    from .protocol import blocked_chembench_class
    cls = blocked_chembench_class()
    return cls(domain_id=domain, difficulty=difficulty, noise_level=NOISE_LEVEL,
               law_version=law_version, grammar_mode=GRAMMAR_MODE)


def compile_law(law_str):
    g = {"np": np, "math": math, "__builtins__": __builtins__}
    exec(law_str, g)
    fn = g.get("discovered_law")
    if fn is None:
        raise ValueError("no discovered_law in law string")
    return fn


class ChemScorer:
    def __init__(self, domain, difficulty, law_version="v0"):
        from autoscilab.oracle.chembench import ChemBenchOracle, CHEM_INPUT_VARS, _secondary_effects
        self._native = ChemBenchOracle(domain_id=domain, difficulty=difficulty, noise_level=NOISE_LEVEL,
                                       law_version=law_version, grammar_mode=GRAMMAR_MODE)
        self._evaluate_law = ChemBenchOracle.evaluate_law   # the released, unblocked method
        self._vars = list(CHEM_INPUT_VARS)
        rate = lambda x: self._native._rate_fn(self._native._law_params, *(x[k] for k in self._vars))
        q1000 = sample_designs(1000, 42)
        r1000 = np.asarray([rate(x) for x in q1000], dtype=float)
        self.var_log1p_primary_1000 = float(np.var(np.log1p(r1000)))
        pack = json.load(open(os.path.join(PACK_DIR, "chembench_eval256.json")))
        self.q256 = pack["designs"]
        primary = np.asarray([rate(x) for x in self.q256], dtype=float)
        noiseless = primary * np.asarray([_secondary_effects(x["T"], x["pH"]) for x in self.q256])
        self.targets256 = {"primary": primary, "noiseless_observation": noiseless}

    def score(self, law_str, observed=None):
        if not law_str:
            return {"status": "no_law"}
        ev = self._evaluate_law(self._native, law_str)
        r = ev.get("rmsle")
        ok = r is not None and math.isfinite(float(r))
        row = {"status": "scored" if ok else "law_error", "error": ev.get("error"),
               "official_rmsle_1000": float(r) if ok else None,
               "official_exact": ev.get("exact_accuracy"),
               "nmsle_1000_primary": float(r) ** 2 / self.var_log1p_primary_1000 if ok else None}
        try:
            fn = compile_law(law_str)
            pred = np.asarray([fn(*(x[k] for k in self._vars)) for x in self.q256], dtype=float)
        except Exception as exc:  # a law that fails on our set is recorded, not repaired
            row.update(status="law_error", error=f"{type(exc).__name__}: {exc}"[:300])
            return row
        yp = np.abs(pred)
        for name, target in self.targets256.items():
            mask = np.isfinite(yp) & np.isfinite(target) & (target > 0)
            sq = float(np.mean((np.log1p(yp[mask]) - np.log1p(target[mask])) ** 2)) if mask.any() else float("nan")
            row[f"nmsle_256_{name}"] = sq / float(np.var(np.log1p(target)))
            row[f"masked_256_{name}"] = int((~mask).sum())
        return row
