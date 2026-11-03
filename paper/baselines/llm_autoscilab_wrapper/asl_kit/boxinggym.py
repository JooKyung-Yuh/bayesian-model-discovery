"""BoxingGym adapter: the released AutoSciLab oracle interface over our BoxingGym worlds.

World. One entry of packs/boxinggym_worlds.json (built by build/build_boxinggym_pack.py with our
engine's own functions from the frozen tree 5d88aaf, and cross-checked against our recorded runs):
the world parameters, Q (256 held-out questions), the noise-free mean at Q, the held-out outcomes,
the D0 designs and the D0 observations our runs saw. Seeds: 31/37/41, survival 43/53/61.

Parameters. The task inputs in declared order with their ranges (design_range where declared).
Names are X1..Xn by default (our masking: "inputs X1..Xn, original ranges, output type and
support only"); --names shown uses the native names and short task descriptions.

Measurement. The scalar observation y as a real number (counts and binary 0/1 outcomes are passed
as reals; AutoSciLab regresses them like any other measurement). A new observation is drawn from
the same observation law as the environment's step(), with numpy instead of the environment's
scipy/PyMC draw (same distribution, different random stream - our recorded D0 values are reused
verbatim):
  death       y ~ Binomial(N=50, 1 - exp(-theta t))                 (death_process.py:319-324)
  dugongs     y = |Normal(a - b |l|^age, 0.25)|                    (dugongs.py:252-259)
  peregrines  y ~ Poisson(exp(alpha + b1 t + b2 t^2 + b3 t^3))      (peregrines.py:256-272)
  hyperbolic  y ~ Bernoulli(eps + (1-2 eps) Phi((dR/(1+kD) - iR)/alpha)) (hyperbolic_temporal_discount.py:415-427)
  survival    y = the stored outcome of the executed patient record (survival_analysis.py:299-301)
Legal designs (the environment's own validators): hyperbolic rounds each input to an integer and,
when iR > dR, swaps iR and dR (validate_input rejects iR > dR); survival executes the nearest
not-yet-observed patient of the fixed 100-record pool (same metastasized value when one is left,
then nearest time) and reports that patient's covariates. The data table always shows the
EXECUTED design. evaluate_law is blocked (no in-run evaluator).

Score (offline, by BoxingGymScorer). On Q minus the observed designs (q_policy exclude_observed,
exact equality, as unobserved_q_mask): nmse_truth = mean((law(x) - E[Y|x])^2) / Var_Q(E[Y|x])
(boxinggym_core.mda_score_row's rule); binary tasks also brier = mean((clip(law(x),0,1) - y_heldout)^2).
A law that raises or returns a non-finite value at any Q point is recorded as law_error, not scored.
"""
import json
import math
import os

import numpy as np

from .paths import PACK_DIR
from .protocol import blocked

SHOWN_TEXT = {
    "death": ("the death process (infections over time)",
              {"t": "time since the start of the outbreak; must be greater than 0 and less than 2"},
              "number of infected individuals out of a population of 50"),
    "dugongs": ("dugong length versus age", {"age": "age of a dugong (sea cow)"}, "length of the dugong"),
    "peregrines": ("peregrine falcon population over time", {"tt": "time"},
                   "population count of peregrine falcons"),
    "hyperbolic": ("choices between an immediate and a delayed reward",
                   {"iR": "immediate reward (integer)", "dR": "delayed reward (integer, at least iR)",
                    "D": "delay in days (positive integer)"},
                   "1 if the person chooses the delayed reward, else 0"),
    "survival": ("patient survival after surgery",
                 {"time": "time since surgery", "metastasized": "1 if the cancer metastasized, else 0"},
                 "1 if the patient died, else 0"),
}
OUTPUT_TEXT = {
    "bounded_count": "a non-negative integer count between {lo} and {hi}",
    "unbounded_count": "a non-negative integer count",
    "positive": "a positive real number",
    "binary": "a binary outcome, 0 or 1",
}


def load_entry(task, seed, pack_path=None):
    pack = json.load(open(pack_path or os.path.join(PACK_DIR, "boxinggym_worlds.json")))
    for e in pack["entries"]:
        if e["task"] == task and int(e["seed"]) == int(seed):
            return e
    raise KeyError(f"no world pack entry for {task} seed {seed}")


def _bounds(entry):
    if entry["design_range"] and len(entry["input_names"]) == 1:
        return [tuple(entry["design_range"])]
    return [tuple(r) for r in entry["input_ranges"]]


_CLASS = {}


def make_oracle(entry, rng_seed, names="masked"):
    """An instance of the adapter, a subclass of the released BaseOracle (imported from the checkout)."""
    if "cls" not in _CLASS:
        from autoscilab.oracle.base import BaseOracle
        _CLASS["cls"] = type("BoxingGymOracle", (_BoxingGymOracleMixin, BaseOracle), {"ASL_BLOCKED": True})
    return _CLASS["cls"](entry, rng_seed, names)


class _BoxingGymOracleMixin:
    def __init__(self, entry, rng_seed, names="masked"):
        from autoscilab.oracle.base import OracleResult
        self._OracleResult = OracleResult
        self.__entry = entry
        self.__task = entry["task"]
        self.__rng = np.random.default_rng(int(rng_seed))
        native = list(entry["input_names"])
        self.__native = native
        self.__names = [f"X{i + 1}" for i in range(len(native))] if names == "masked" else native
        self.__mode = names
        self.__bounds = _bounds(entry)
        self.__observed_patients = set()   # survival: filled by d0_results() and by every executed request

    # --- the released BaseOracle interface ------------------------------------------------
    @property
    def domain(self):
        return "unknown_system" if self.__mode == "masked" else self.__task

    @property
    def parameter_names(self):
        return list(self.__names)

    @property
    def parameter_bounds(self):
        return {n: (float(lo), float(hi)) for n, (lo, hi) in zip(self.__names, self.__bounds)}

    @property
    def function_signature(self):
        return f"def discovered_law({', '.join(self.__names)}):"

    @property
    def param_description(self):
        e = self.__entry
        lo, hi = (e["support"] + [None, None])[:2]
        out = OUTPUT_TEXT[e["outcome_type"]].format(lo=lo, hi=hi)
        integer = set(e["integer_inputs"])
        lines = []
        for name, native, (blo, bhi) in zip(self.__names, self.__native, self.__bounds):
            kind = "integer" if native in integer else "real number"
            what = "" if self.__mode == "masked" else f" ({SHOWN_TEXT[self.__task][1][native]})"
            lines.append(f"- {name}{what}: {kind} in [{blo:g}, {bhi:g}]")
        if self.__task == "hyperbolic":
            a, b = self.__names[0], self.__names[1]
            lines.append(f"Constraint: {a} <= {b}; a request with {a} > {b} is executed with {a} and {b} swapped, "
                         "and inputs are rounded to integers.")
        if self.__task == "survival":
            lines.append("Inputs are the covariates of a fixed set of 100 candidate units; a request is executed on "
                         f"the nearest not-yet-observed unit (same {self.__names[1]} value when one is left) and the "
                         "data table shows that unit's covariates.")
        noun = "y" if self.__mode == "masked" else f"y ({SHOWN_TEXT[self.__task][2]})"
        lines.append(f"Output {noun}: {out}; each measurement is a noisy draw. "
                     "The discovered law should return the expected value of y.")
        return "\n".join(lines)

    def run(self, params):
        x = [float(params[n]) for n in self.__names]
        executed, meta = self._legalize(x)
        y = self._observe(executed, meta)
        return self._OracleResult(params={n: float(v) for n, v in zip(self.__names, executed)},
                                  measurement=float(y), domain=self.domain, noise_level=0.0,
                                  metadata={"requested": [float(v) for v in x], **meta})

    def evaluate_law(self, law_str):
        blocked("BoxingGymOracle.evaluate_law")

    def get_ground_truth_law_str(self):
        blocked("BoxingGymOracle.get_ground_truth_law_str")

    # --- the adapter ----------------------------------------------------------------------
    def d0_results(self):
        """Our D0: the designs and the observations our runs recorded (round 0), reused verbatim."""
        e = self.__entry
        if self.__task == "survival":
            self.__observed_patients.update(int(h) for h in e["d0"]["handles"])
        return [self._OracleResult(params={n: float(v) for n, v in zip(self.__names, x)}, measurement=float(y),
                                   domain=self.domain, noise_level=0.0, metadata={"d0": True})
                for x, y in zip(e["d0"]["x"], e["d0"]["y"])]

    def _legalize(self, x):
        t = self.__task
        x = [min(max(v, lo), hi) for v, (lo, hi) in zip(x, self.__bounds)]
        if t == "hyperbolic":
            iR, dR, D = (int(round(v)) for v in x)
            swapped = iR > dR
            if swapped:
                iR, dR = dR, iR
            return [float(iR), float(dR), float(D)], {"swapped": swapped}
        if t == "survival":
            recs = self.__entry["world"]["records"]
            want_t, want_m = x[0], int(round(x[1]))
            free = [i for i in range(len(recs)) if i not in self.__observed_patients]
            if not free:
                raise RuntimeError("survival pool exhausted")
            same = [i for i in free if int(recs[i][2]) == want_m]
            cands = same or free
            i = min(cands, key=lambda j: (abs(float(recs[j][1]) - want_t) / 10.0 + abs(int(recs[j][2]) - want_m), j))
            self.__observed_patients.add(i)
            return [float(recs[i][1]), float(recs[i][2])], {"patient": i}
        return x, {}

    def _observe(self, x, meta):
        w, t, rng = self.__entry["world"], self.__task, self.__rng
        if t == "death":
            return rng.binomial(int(w["N"]), 1.0 - math.exp(-w["theta"] * x[0]))
        if t == "dugongs":
            return abs(rng.normal(w["a"] - w["b"] * abs(w["l"]) ** x[0], w["sigma_obs"]))
        if t == "peregrines":
            tt = x[0]
            return rng.poisson(math.exp(w["alpha"] + w["beta1"] * tt + w["beta2"] * tt ** 2 + w["beta3"] * tt ** 3))
        if t == "hyperbolic":
            from scipy.stats import norm
            iR, dR, D = x
            p = w["epsilon"] + (1 - 2 * w["epsilon"]) * norm.cdf((dR / (1 + w["k"] * D) - iR) / w["alpha"])
            return rng.binomial(1, p)
        if t == "survival":
            return int(w["records"][meta["patient"]][0])
        raise ValueError(t)


def compile_law(law_str):
    g = {"np": np, "math": math, "__builtins__": __builtins__}
    exec(law_str, g)
    fn = g.get("discovered_law")
    if fn is None:
        raise ValueError("no discovered_law in law string")
    return fn


def unobserved_q_mask(evaluation_x, observed):
    """boxinggym_core.unobserved_q_mask, verbatim rule: exact equality in declared input units."""
    q = np.asarray(evaluation_x, dtype=float)
    q = q[:, None] if q.ndim == 1 else q
    known = np.asarray(observed, dtype=float)
    if known.size == 0:
        return np.ones(len(q), dtype=bool)
    known = known.reshape(-1, q.shape[1])
    return ~np.any(np.all(q[:, None, :] == known[None, :, :], axis=2), axis=1)


def nmse_truth(predictions, truth_mean):
    """boxinggym_core.mda_score_row's nmse_truth for one output component."""
    truth = np.asarray(truth_mean, dtype=float)
    means = np.asarray(predictions, dtype=float)
    variance = np.var(truth, axis=0)
    if truth.size and np.isfinite(truth).all() and np.all(variance > 0):
        return float(np.mean(np.mean((means - truth) ** 2, axis=0) / variance))
    return None


class BoxingGymScorer:
    def __init__(self, entry):
        self.entry = entry
        self.qx = np.asarray(entry["q"]["x"], dtype=float)
        self.truth = np.asarray(entry["q"]["truth_mean"], dtype=float)
        self.heldout = np.asarray(entry["q"]["heldout"], dtype=float)
        self.binary = entry["outcome_type"] == "binary"

    def score(self, law_str, observed=None):
        if not law_str:
            return {"status": "no_law"}
        keep = unobserved_q_mask(self.qx, observed if observed is not None else [])
        try:
            fn = compile_law(law_str)
            pred = np.asarray([fn(*row) for row in self.qx[keep]], dtype=float)
        except Exception as exc:
            return {"status": "law_error", "error": f"{type(exc).__name__}: {exc}"[:300]}
        if pred.shape != (int(keep.sum()),) or not np.isfinite(pred).all():
            return {"status": "law_error", "error": f"non-finite or misshaped predictions ({int((~np.isfinite(pred)).sum())} non-finite)"}
        row = {"status": "scored", "nmse_truth": nmse_truth(pred, self.truth[keep]), "q_points": int(keep.sum()),
               "q_excluded": int((~keep).sum())}
        if self.binary:
            row["brier"] = float(np.mean((np.clip(pred, 0.0, 1.0) - self.heldout[keep]) ** 2))
        return row
