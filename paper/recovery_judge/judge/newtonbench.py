"""NewtonBench judge .  The decision rules are core.py's (rule c5-judge-1.3), unchanged.

Truth (read only).  The task's law function in NewtonBench (HKUST-KnowComp/NewtonBench 912a4ba5f4356ddd06acc16e44460ca30be4abc2,
the copy LLM-AutoSciLab 9a37255 vendors), loaded through this repository's adapter
`bayes_model_discovery.adapters.newtonbench_observation` (the adapter the cells record as config.json adapter_sha256).  The adapter executes laws.py by file path after its own
sha256 check and never imports a NewtonBench package.  Each declared law is transcribed below from the law function's
return statement; when a target is built, the transcription is compared with the adapter's vanilla-branch value
(`_Law(...).value`, noise 0) on the box grid and the box corners, and a relative deviation above TRANSCRIPTION_TOL
refuses the target.

Target family.  The same pipeline as ChemBench: the mean after the log link, compared up to a positive overall factor
that is not judged.  The law's literal numbers (exponents, numeric coefficients) are fixed, because they are the law's
structure: they are what tells one NewtonBench difficulty or version from another.  Its named module constants
(CONSTANT, CONSTANT2, HIDDEN_CONSTANT) are free, and a named constant that multiplies the whole law is the overall
factor.  Inputs are the engine's masked X1..Xn (config.json input_name_map), all positive on the box.  The admissible
domain is the task's box: config.json benchmark_adapter input_bounds, that is AutoSciLab's DOMAIN_REGISTRY box, or for
m5 NewtonBench's evaluate_law ranges.  The box is cross-checked against the adapter's task_box and is log-uniform on
the log-sampled inputs.  Diagnostic members: the truth, truth x 0.6 and truth x 1.7 (every family coefficient).

Candidate (presentation P, read from config.json).  mean = response_unit * exp(expression), the log link.  The
expression is written over X1..Xn, the z-scored Xiz = (Xi - centre_i)/scale_i (config.json transform), and the
log-centred Xil = log(Xi) - c_i (config.json benchmark_presentation.log_centred_inputs; each definition is parsed as
written and checked against its recorded centre).  The geometric-mean response unit is recorded; like any positive
constant it is absorbed by the overall factor, which is not judged.

Representability per task (the nine depth tasks of the NewtonBench plans):
  exact law in the grammar   m2, m3, m5, m8, m9, m11.  Each spelling is checked against the grammar v2 caps
                             (<= 6 coefficients, <= 15 operators, calls exp/log/sqrt/tanh/square/power), and
                             test_newtonbench.py judges every spelling equivalent to its target.
  not_representable          m7_malus_law__medium__v0: I_0 (2 sin theta + cos theta)^2 needs sin and cos, and grammar
                             v2 has no trigonometric call.  Section 13 checked m7 only through a least-squares quartic
                             surrogate whose error, 0.011, is above the 0.01 noise.
  identifiable surrogate     m4_snell_law__medium__v2: atan(n1 tan(theta)/n2) with theta in degrees on [0.01, 1.5]; the
                             surrogate is the small-angle form n1 theta / n2.
                             m10_be_distribution__medium__v1: 1/(exp(C sqrt(omega) T) - 1) with C sqrt(omega) T <= 1.06e-3
                             on the box; the surrogate is 1/(C sqrt(omega) T).
                             On the box the nonlinear form is not identifiable at noise 0.01.  The largest relative
                             difference between law and surrogate on the box grid and corners is recorded and must be
                             below the noise.
                             RULE: a candidate is equivalent (E_SURROGATE) only when its family is equivalent to the
                             surrogate family by the canonical pipeline.  Every other candidate is unresolved with code
                             U_NOT_IDENTIFIABLE.  The surrogate comparison's own state and code are kept in
                             `surrogate_comparison`, and containment and world_law_member are those of that comparison
                             (basis prefixed 'surrogate:').  m10's exact law is in the grammar too, and
                             `true_law_match` records, as a diagnostic only, how the candidate relates symbolically to it.
A task without a declaration raises: the cell is reported as a record error, never judged silently.
"""
import json
import math
import sys

import numpy as np
import sympy as sp

from . import core as C

BENCH = "newtonbench"
TARGET_VERSION = "v1"
#: the adapter of the paper's NewtonBench runs, `bayes_model_discovery.adapters.newtonbench_observation` of this
#: repository (its recorded sha256 400ded4f64e0bcf21c5432b862c0a9f72426d0189afe81f23c7860758d032716 covers the
#: development copy; the public copy adds the BMD_NEWTONBENCH lookup and a docstring)
NEWTONBENCH_COMMIT = "912a4ba5f4356ddd06acc16e44460ca30be4abc2"
NOISE = 0.01                    # the noise level of every NewtonBench cell (relative, NewtonBench's inject_noise)
TRANSCRIPTION_TOL = 1e-9        # largest relative deviation of a transcribed law from the adapter's value
CALLS_V2 = ("exp", "log", "sqrt", "tanh", "square", "power")
CALLS_V3 = CALLS_V2 + ("abs", "sigmoid", "expm1", "log1p", "minimum", "maximum")
MAX_COEFFICIENTS, MAX_OPERATORS = 6, 15     # count_models.py MAX_DISTINCT_MEAN_COEFFICIENTS, MEAN_OPERATOR_LIMIT
R = sp.Rational


def _n(name):
    return sp.Symbol(name, positive=True)


#: task id -> declaration.  'law' is the law in the native argument names with the module constants (a dict of floats)
#: written in; 'shape' the family without its overall factor, with free named constants as symbols ('free': name ->
#: truth value from the module constants); 'scale' the overall factor's truth value; 'kind' law | surrogate |
#: not_representable; 'spelling' the grammar spelling over X1..Xn and Xil (presentation P); for a surrogate task
#: 'surrogate' says which form the family is and 'true_shape' / 'true_free' / 'true_scale' / 'true_spelling' declare
#: the exact law's family when the grammar can write it.
DECLARATIONS = {
    "m2_magnetic_force__medium__v0": {
        "kind": "law", "source": "modules/m2_magnetic_force/laws.py:15 _ground_truth_law_medium_v0",
        "law": lambda N, K: K["CONSTANT"] * (N["current1"] * N["current2"]) ** R(3, 2) / N["distance"] ** 3,
        "shape": lambda N, F: (N["current1"] * N["current2"]) ** R(3, 2) / N["distance"] ** 3,
        "free": {}, "scale": lambda K: K["CONSTANT"],
        "spelling": "a + 1.5*X1l + 1.5*X2l - 3*X3l"},
    "m3_fourier_law__medium__v0": {
        "kind": "law", "source": "modules/m3_fourier_law/laws.py:37 _ground_truth_law_medium_v0",
        "law": lambda N, K: N["k"] * N["A"] ** R(1, 2) * N["delta_T"] / N["d"] ** 2,
        "shape": lambda N, F: N["k"] * N["A"] ** R(1, 2) * N["delta_T"] / N["d"] ** 2,
        "free": {}, "scale": lambda K: 1.0,
        "spelling": "a + X1l + 0.5*X2l + X3l - 2*X4l"},
    "m4_snell_law__medium__v2": {
        "kind": "surrogate", "source": "modules/m4_snell_law/laws.py:59 _ground_truth_law_medium_v2",
        "law_text": "degrees(atan(n1 tan(radians(theta)) / n2))",
        "surrogate": "n1 theta / n2 (the small-angle form; theta and the result in degrees)",
        "shape": lambda N, F: N["refractive_index_1"] * N["incidence_angle"] / N["refractive_index_2"],
        "free": {}, "scale": lambda K: 1.0,
        "spelling": "a + X1l - X2l + X3l",
        "true_in_grammar": "no: atan and tan are not grammar v2 calls"},
    "m5_radioactive_decay__medium__v0": {
        "kind": "law", "source": "modules/m5_radioactive_decay/laws.py:40 _ground_truth_law_medium_v0",
        "law": lambda N, K: N["N0"] ** R(6, 5) * sp.exp(-N["lambda_constant"] * N["t"] ** R(3, 2)),
        "shape": lambda N, F: N["N0"] ** R(6, 5) * sp.exp(-N["lambda_constant"] * N["t"] ** R(3, 2)),
        "free": {}, "scale": lambda K: 1.0,
        "spelling": "a + 1.2*X1l - X2*power(X3, 1.5)"},
    "m7_malus_law__medium__v0": {
        "kind": "not_representable", "source": "modules/m7_malus_law/laws.py:53 _ground_truth_law_medium_v0",
        "law": lambda N, K: N["I_0"] * (2 * sp.sin(N["theta"]) + sp.cos(N["theta"])) ** 2,
        "reason": "the true law I_0 (2 sin theta + cos theta)^2 needs sin and cos; grammar v2 (calls exp, log, sqrt, "
                  "tanh, square, power) has no trigonometric call (STUDY_NEWTONBENCH.md section 13: 'Grammar v2 has no "
                  "trigonometric function'; m7 was checked only through a least-squares quartic surrogate whose "
                  "error 0.011 is above the 0.01 noise)"},
    "m8_sound_speed__medium__v0": {
        "kind": "law", "source": "modules/m8_sound_speed/laws.py:18 _ground_truth_law_medium_v0",
        "law": lambda N, K: sp.sqrt(N["gamma"] * K["HIDDEN_CONSTANT"] * N["T"] ** 2 / N["M"] ** R(3, 2)),
        "shape": lambda N, F: N["gamma"] ** R(1, 2) * N["T"] * N["M"] ** R(-3, 4),
        "free": {}, "scale": lambda K: math.sqrt(K["HIDDEN_CONSTANT"]),
        "spelling": "a + 0.5*X1l + X2l - 0.75*X3l"},
    "m9_hooke_law__medium__v0": {
        "kind": "law", "source": "modules/m9_hooke_law/laws.py:41 _ground_truth_law_medium_v0",
        "law": lambda N, K: 2 * K["CONSTANT"] * N["x"] ** 2 + K["CONSTANT2"] * N["x"],
        "shape": lambda N, F: N["x"] ** 2 + F["c"] * N["x"],
        "free": {"c": lambda K: K["CONSTANT2"] / (2 * K["CONSTANT"])}, "scale": lambda K: 2 * K["CONSTANT"],
        "spelling": "a + log(square(X1) + c*X1)"},
    "m10_be_distribution__medium__v1": {
        "kind": "surrogate", "source": "modules/m10_be_distribution/laws.py:61 _ground_truth_law_medium_v1",
        "law_text": "1/(exp(C sqrt(omega) T) - 1)",
        "surrogate": "1/(C sqrt(omega) T) (the first term of 1/(e^x - 1) = 1/x - 1/2 + x/12 - ...)",
        "shape": lambda N, F: N["omega"] ** R(-1, 2) / N["T"],
        "free": {}, "scale": lambda K: 1.0 / K["HIDDEN_CONSTANT"],
        "spelling": "a - 0.5*X1l - X2l",
        "true_shape": lambda N, F: 1 / (sp.exp(F["c"] * N["omega"] ** R(1, 2) * N["T"]) - 1),
        "true_free": {"c": lambda K: K["HIDDEN_CONSTANT"]}, "true_scale": lambda K: 1.0,
        "true_spelling": "a - log(exp(c*sqrt(X1)*X2) - 1)",
        "true_in_grammar": "yes: a - log(exp(c*sqrt(X1)*X2) - 1)"},
    "m11_heat_transfer__medium__v1": {
        "kind": "law", "source": "modules/m11_heat_transfer/laws.py:52 _ground_truth_law_medium_v1",
        "law": lambda N, K: N["c"] / (N["m"] ** R(5, 2) * N["delta_T"]),
        "shape": lambda N, F: N["c"] * N["m"] ** R(-5, 2) / N["delta_T"],
        "free": {}, "scale": lambda K: 1.0,
        "spelling": "a - 2.5*X1l + X2l - X3l"},
}
#: the m10 exact law (for its transcription check; the surrogate's family is what is judged)
EXACT_LAWS = {
    "m4_snell_law__medium__v2": None,   # trigonometric: checked through the adapter's value only (surrogate error)
    "m10_be_distribution__medium__v1": lambda N, K: 1 / (sp.exp(K["HIDDEN_CONSTANT"] * N["omega"] ** R(1, 2) * N["T"]) - 1),
}
MODULE_CONSTANTS = ("CONSTANT", "CONSTANT2", "CONSTANT3", "HIDDEN_CONSTANT", "C", "K", "R")


# ------------------------------------------------------------------------------------------------ the NewtonBench adapter
_ADAPTER = {}


def adapter():
    """The NewtonBench adapter module of this repository (law files read by path after their sha256 check)."""
    if "mod" not in _ADAPTER:
        import os
        root = os.environ.get("BMD_AUTOSCILAB")
        if root and root not in sys.path:
            sys.path.insert(0, root)      # optional: the AutoSciLab registry box of the paper's tasks
        from bayes_model_discovery.adapters import newtonbench_observation as mod
        _ADAPTER["mod"] = mod
    return _ADAPTER["mod"]


def module_constants(domain):
    """The named module constants of the domain's laws.py (as loaded by the adapter)."""
    ad = adapter()
    laws = ad._load(f"{domain}/laws.py", f"bmd_newtonbench_laws_{domain}")
    return {k: float(getattr(laws, k)) for k in MODULE_CONSTANTS if isinstance(getattr(laws, k, None), (int, float))}


# ------------------------------------------------------------------------------------------------ representability
def count_operators(text):
    import ast
    return sum(isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call)) for nd in ast.walk(ast.parse(text, mode="eval")))


def spelling_check(text, input_names, calls=CALLS_V2):
    """(ok, detail): the spelling stays within the grammar caps (coefficients, operators, calls)."""
    import ast
    tree = ast.parse(text, mode="eval")
    used_calls = {nd.func.id for nd in ast.walk(tree) if isinstance(nd, ast.Call) and isinstance(nd.func, ast.Name)}
    names = {nd.id for nd in ast.walk(tree) if isinstance(nd, ast.Name)} - used_calls
    coefs = sorted(names - set(input_names))
    ops = count_operators(text)
    ok = len(coefs) <= MAX_COEFFICIENTS and ops <= MAX_OPERATORS and used_calls <= set(calls)
    return ok, (f"spelling {text!r}: {len(coefs)} coefficients, {ops} operators, calls {sorted(used_calls)} "
                f"(caps {MAX_COEFFICIENTS} and {MAX_OPERATORS}, calls {', '.join(calls)})")


# ------------------------------------------------------------------------------------------------ target
class Target:
    """The declared target of one NewtonBench cell, built from its config.json (task, box, presentation P)."""

    def __init__(self, config):
        ba = config.get("benchmark_adapter") or {}
        if (ba.get("benchmark") or config.get("benchmark")) != BENCH:
            raise ValueError("config.json is not a NewtonBench cell (benchmark_adapter.benchmark)")
        task = config.get("task") or {}
        self.world = task.get("id") or config.get("task_requested")
        self.benchmark = BENCH
        self.domain, self.difficulty, self.law_version = ba["domain"], ba["difficulty"], ba["law_version"]
        if self.world != f"{self.domain}__{self.difficulty}__{self.law_version}":
            raise ValueError(f"task id {self.world!r} differs from the adapter record {ba}")
        if self.world not in DECLARATIONS:
            raise ValueError(f"no declared C5 target for NewtonBench task {self.world!r} (declared: "
                             f"{', '.join(sorted(DECLARATIONS))})")
        decl = DECLARATIONS[self.world]
        self.kind = decl["kind"]
        self.target_id = f"{BENCH}/{self.world}/{TARGET_VERSION}"
        self.source = decl["source"]
        # ---- inputs, names and box (config) cross-checked against the adapter's task_box
        name_map = dict(config["input_name_map"])
        self.inputs_masked = sorted(name_map, key=lambda x: int(x[1:]))
        natives = [name_map[x] for x in self.inputs_masked]
        if natives != list(ba["input_names"]):
            raise ValueError(f"input_name_map {name_map} is not the adapter's law-argument order {ba['input_names']}")
        ad = adapter()
        names, bounds, log_names = ad.task_box({"domain": self.domain, "input_box": ba["input_box"],
                                                "system": ba.get("system", "vanilla_equation")})
        if list(names) != natives or any([float(bounds[n][0]), float(bounds[n][1])] != [float(v) for v in ba["input_bounds"][n]]
                                         for n in names) or sorted(log_names) != sorted(ba["log_sampled_inputs"]):
            raise ValueError("config.json benchmark_adapter box differs from the adapter's task_box")
        self.name_map, self.natives, self.input_box = name_map, natives, ba["input_box"]
        self.X = {x: sp.Symbol(x, positive=True) for x in self.inputs_masked}
        native_sym = {name_map[x]: self.X[x] for x in self.inputs_masked}     # the law written straight in X1..Xn
        spec = {self.X[x]: (float(bounds[name_map[x]][0]), float(bounds[name_map[x]][1]),
                            "log" if name_map[x] in log_names else "lin") for x in self.inputs_masked}
        self.box_spec = {x: list(spec[self.X[x]]) for x in self.inputs_masked}
        self.box = C.Box(spec, f"{self.target_id}|{json.dumps(self.box_spec, sort_keys=True)}")
        self.law_file = (ba.get("law_files") or {}).get(f"{self.domain}/laws.py") or {}
        self.adapter_sha256_recorded = config.get("adapter_sha256")
        # ---- presentation (P): derived inputs and response unit
        grammar = str(config.get("grammar_version") or "v2")
        self.calls = CALLS_V3 if grammar == "v3" else CALLS_V2
        self.grammar = grammar
        self.derived, self.transform_record = self._derived(config)
        pres = config.get("benchmark_presentation") or {}
        self.presentation_edition = pres.get("edition") or "registered (no benchmark_presentation)"
        self.response_unit = pres.get("response_unit")
        self.response_unit_rule = pres.get("response_unit_rule") or "arithmetic_mean_of_d0 (no benchmark_presentation)"
        self.transform_key = json.dumps(self.transform_record, sort_keys=True)
        # ---- the law, its family and its representability
        K = module_constants(self.domain)
        self.module_constants = K
        self.law_function = ba.get("law_function")
        if self.kind == "not_representable":
            self.representable = False
            self.not_representable_reason = decl["reason"]
            self.representability_detail = decl["reason"]
            self.family = None
            self.truth, self.members = None, []
            self.check = self._check_law(decl["law"](native_sym, K), strict=False)    # recorded, not used to judge
            self.judged_against = "not_representable"
            self.true_family = None
            return
        ok, detail = spelling_check(decl["spelling"], list(self.inputs_masked) + list(self.derived), self.calls)
        self.representable = ok
        self.not_representable_reason = None if ok else detail
        self.representability_detail = detail
        self.family, self.truth = self._family(decl["shape"], decl["free"], decl["scale"], native_sym, K, "T_")
        self.members = [self.truth] + [{k: v * f for k, v in self.truth.items()} for f in (0.6, 1.7)]
        if self.kind == "law":
            self.check = self._check_law(None, family=self.family, truth=self.truth)
            self.check["law_text_check"] = self._check_law(decl["law"](native_sym, K))
            self.judged_against = "law"
            self.true_family = None
        else:
            self.check = self._check_surrogate()
            if not self.check["surrogate_max_rel_error"] < NOISE:
                raise ValueError(f"{self.world}: the surrogate differs from the law by "
                                 f"{self.check['surrogate_max_rel_error']:.3g} on the box, not below the noise {NOISE}")
            self.judged_against = "identifiable_surrogate"
            self.surrogate_text, self.law_text = decl["surrogate"], decl["law_text"]
            self.true_in_grammar = decl.get("true_in_grammar")
            self.true_family = None
            if decl.get("true_shape") is not None:
                self.true_family, self.true_truth = self._family(decl["true_shape"], decl["true_free"],
                                                                 decl["true_scale"], native_sym, K, "L_")
                # diagnostic only (exp(x) - 1 at x ~ 1e-8 loses digits in float64): recorded, never refusing
                self.check["true_law_transcription"] = self._check_law(None, family=self.true_family,
                                                                       truth=self.true_truth, strict=False)
                self.check["true_law_text_check"] = self._check_law(EXACT_LAWS[self.world](native_sym, K), strict=False)
                ok2, d2 = spelling_check(decl["true_spelling"], list(self.inputs_masked) + list(self.derived), self.calls)
                self.check["true_law_spelling"] = d2 if ok2 else "outside the caps: " + d2

    # -- helpers
    def _derived(self, config):
        """Xiz from config.json transform, Xil from benchmark_presentation.log_centred_inputs (definition parsed)."""
        out, record = {}, {"z": {}, "l": {}}
        tr = config.get("transform") or {}
        centres, scales = tr.get("center"), tr.get("scale")
        if centres is not None and scales is not None:
            if len(centres) != len(self.inputs_masked) or len(scales) != len(self.inputs_masked):
                raise ValueError("config.json transform does not have one centre and scale per input")
            for x, c, s in zip(self.inputs_masked, centres, scales):
                out[x + "z"] = (self.X[x] - C._literal(float(c))) / C._literal(float(s))
                record["z"][x + "z"] = [repr(float(c)), repr(float(s))]
        pres = config.get("benchmark_presentation") or {}
        for name, d in sorted((pres.get("log_centred_inputs") or {}).items()):
            x = d["input"]
            if x not in self.X or name != x + "l":
                raise ValueError(f"log-centred input {name} names an unknown input {x}")
            expr = C.parse_expression(d["definition"], dict(self.X), calls=("log",))
            if sp.simplify(expr - (sp.log(self.X[x]) - C._literal(float(d["centre"])))) != 0:
                raise ValueError(f"{name}: definition {d['definition']!r} differs from log({x}) - {d['centre']!r}")
            out[name] = expr
            record["l"][name] = d["definition"]
        return out, record

    def _family(self, shape_fn, free, scale_fn, native_sym, K, prefix):
        free_syms = {nm: C.param_symbol(nm, "positive", prefix=prefix) for nm in free}
        shape = shape_fn(native_sym, free_syms)
        s = C.param_symbol("scale", "positive", prefix=prefix)
        params = [s] + [free_syms[nm] for nm in sorted(free_syms)]
        fam = C.Family(s * shape, params, {p: "positive" for p in params}, [self.X[x] for x in self.inputs_masked],
                       num=sp.log(s) + sp.expand_log(sp.log(shape), force=True))
        truth = {s: float(scale_fn(K))}
        for nm, fn in free.items():
            truth[free_syms[nm]] = float(fn(K))
        return fam, truth

    def _points(self):
        """The box's diagnostic grid plus its corners (the inputs at which the law is checked)."""
        syms = self.box.symbols()
        grid = self.box.grid(C.N_DIAG, "diag")
        pts = [[float(col[i]) for col in grid] for i in range(len(grid[0]))]
        corners = [[]]
        for s in syms:
            lo, hi, _ = self.box.spec[s]
            corners = [c + [v] for c in corners for v in (lo, hi)]
        return syms, pts + corners

    def _law_values(self, syms, pts):
        """The adapter's noise-free vanilla-branch value (its preprocessing included) at each point; the point's
        coordinates are in the order of `syms` (the box symbols X1..Xn), mapped to the law's argument names."""
        ad = adapter()
        law = ad._Law(self.domain, self.difficulty, self.law_version, list(self.natives), 0.0)
        native_of = {self.X[x]: self.name_map[x] for x in self.inputs_masked}
        return np.asarray([law.value({native_of[s]: v for s, v in zip(syms, p)}) for p in pts], dtype=float)

    def _check_law(self, law_expr, family=None, truth=None, strict=True):
        """Largest relative deviation of a transcription (the law itself, or a family at the truth) from the
        adapter's value on the box grid and corners; above TRANSCRIPTION_TOL a strict check refuses the target."""
        syms, pts = self._points()
        ref = self._law_values(syms, pts)
        expr = law_expr if family is None else family.sym.xreplace({p: sp.Float(repr(v)) for p, v in truth.items()})
        f = sp.lambdify(syms, expr, modules="numpy")
        with np.errstate(all="ignore"):
            mine = np.asarray([float(f(*p)) for p in pts], dtype=float)
        ok = np.isfinite(ref) & (ref != 0)
        dev = float(np.max(np.abs(mine[ok] - ref[ok]) / np.abs(ref[ok]))) if ok.any() else math.inf
        out = {"points": len(pts), "finite_reference_points": int(ok.sum()), "max_rel_deviation": float(f"{dev:.6g}"),
               "transcription_tol": TRANSCRIPTION_TOL}
        if strict and not dev <= TRANSCRIPTION_TOL:
            raise ValueError(f"{self.world}: the transcribed law deviates from the adapter's value by {dev:.3g}")
        return out

    def _check_surrogate(self):
        """Largest relative difference between the adapter's law value and the surrogate at the truth, on the box
        grid and corners (the identifiability statement: it must be below the noise)."""
        syms, pts = self._points()
        ref = self._law_values(syms, pts)
        expr = self.family.sym.xreplace({p: sp.Float(repr(v)) for p, v in self.truth.items()})
        f = sp.lambdify(syms, expr, modules="numpy")
        with np.errstate(all="ignore"):
            mine = np.asarray([float(f(*p)) for p in pts], dtype=float)
        ok = np.isfinite(ref) & (ref != 0)
        if not ok.all():
            raise ValueError(f"{self.world}: the law is not finite and nonzero on the whole box")
        dev = float(np.max(np.abs(mine - ref) / np.abs(ref)))
        return {"points": len(pts), "surrogate_max_rel_error": float(f"{dev:.6g}"), "noise": NOISE}

    def describe(self):
        d = {"target_id": self.target_id, "world": self.world, "kind": self.kind, "representable": self.representable,
             "representability": self.representability_detail, "judged_against": self.judged_against,
             "law_source": f"NewtonBench {NEWTONBENCH_COMMIT[:7]} {self.source}", "law_function": self.law_function,
             "law_file_sha256": self.law_file.get("sha256"), "module_constants": self.module_constants,
             "box": self.box_spec, "presentation_edition": self.presentation_edition,
             "derived_inputs": self.transform_record, "response_unit": self.response_unit,
             "response_unit_rule": self.response_unit_rule, "grammar": self.grammar,
             "adapter": {"module": "bayes_model_discovery.adapters.newtonbench_observation",
                         "cell_recorded_sha256": self.adapter_sha256_recorded},
             "transcription_check": self.check}
        if self.family is not None:
            d["family"] = str(self.family.sym)
            d["truth"] = {k.name: v for k, v in self.truth.items()}
        if self.kind == "surrogate":
            d["surrogate"] = self.surrogate_text
            d["true_law"] = self.law_text
            d["true_law_in_grammar"] = self.true_in_grammar
            if self.true_family is not None:
                d["true_law_family"] = str(self.true_family.sym)
                d["true_law_truth"] = {k.name: v for k, v in self.true_truth.items()}
        return d


# ------------------------------------------------------------------------------------------------ candidate
def candidate_family(candidate, target):
    names = dict(target.X)
    names.update(target.derived)
    roles = dict(candidate["parameters"])
    params = {nm: C.param_symbol(nm, r) for nm, r in roles.items()}
    clash = set(params) & set(names)
    if clash:
        raise ValueError(f"coefficient names shadow inputs: {sorted(clash)}")
    names.update(params)
    eta = C.parse_expression(candidate["expression"], names, calls=target.calls)
    sig = C.param_symbol("overall_factor", "positive", prefix="S_")
    plist = [params[nm] for nm in roles] + [sig]
    domains = {params[nm]: C.role_domain(r) for nm, r in roles.items()}
    domains[sig] = "positive"
    fam = C.Family(sig * sp.exp(eta), plist, domains, [target.X[x] for x in target.inputs_masked],
                   num=sp.log(sig) + eta, num_from_sym=lambda e: sp.expand_log(sp.log(e), force=True), scale=sig)
    return fam, eta


FIELDS = ("containment", "containment_basis", "containment_witness", "numeric_containment_dev", "numeric_members_evaluated",
          "world_law_member", "world_member_basis", "world_member_dev", "truth_reachable", "domain_note", "dependence",
          "dims", "target_tree")


def judge(candidate, target):
    """candidate: {expression, family, parameters: {name: role}, model_id?}; target: Target."""
    expr = candidate.get("expression")
    extra = {"benchmark": BENCH, "model_id": candidate.get("model_id"), "family": candidate.get("family"),
             "roles": candidate.get("parameters"), "noise_family_match": None, "judged_against": target.judged_against,
             "presentation": target.presentation_edition}
    if not target.representable:
        return C.record("not_representable", target.target_id, expr, "R_NOT_REPRESENTABLE",
                        target.not_representable_reason, **extra)
    if not expr or candidate.get("parameters") is None:
        return C.record("unresolved", target.target_id, expr, "U_MISSING_EXPRESSION",
                        "the record holds no expression or no coefficient roles for this model id", **extra)
    try:
        fam, _ = candidate_family(candidate, target)
    except Exception as exc:  # noqa: BLE001
        return C.record("unresolved", target.target_id, expr, "U_PARSE", f"{type(exc).__name__}: {str(exc)[:160]}", **extra)
    key = json.dumps([target.target_id, target.world, target.transform_key, expr,
                      sorted((candidate.get("parameters") or {}).items())])
    out = C.compare(fam, target.family, target.box, truth=target.truth, members=target.members, key=key)
    if target.kind == "surrogate":
        comp = {"state": out["state"], "code": out["reason"]["code"], "detail": out["reason"]["detail"]}
        err = target.check["surrogate_max_rel_error"]
        if out["state"] == "equivalent":
            rec = C.record("equivalent", target.target_id, expr, "E_SURROGATE",
                           f"equivalent to the identifiable surrogate {target.surrogate_text} ({out['reason']['detail']}); "
                           f"the true law {target.law_text} is not identifiable on the box (largest relative "
                           f"difference from the surrogate {err:.3g}, noise {NOISE:g}); rule: equivalent only when the "
                           "family matches the surrogate", **extra)
        else:
            rec = C.record("unresolved", target.target_id, expr, "U_NOT_IDENTIFIABLE",
                           f"not equivalent to the identifiable surrogate {target.surrogate_text} ({comp['code']}); the "
                           f"true law {target.law_text} is not identifiable on the box (largest relative difference "
                           f"from the surrogate {err:.3g}, noise {NOISE:g}), so a form other than the surrogate is "
                           "left unresolved", **extra)
        rec["surrogate_comparison"] = comp
        if target.true_family is not None:
            try:
                mm = C.match_families(fam, target.true_family, target.box, key + "|true_law")
                rec["true_law_match"] = mm.get("kind") or "no_canonical_match"
            except Exception as exc:  # noqa: BLE001 - a diagnostic only
                rec["true_law_match"] = f"error: {type(exc).__name__}"
    else:
        rec = C.record(out["state"], target.target_id, expr, out["reason"]["code"], out["reason"]["detail"], **extra)
    rec["canonical_form"] = out.get("canonical_tree")
    rec["parameter_mapping"] = out.get("parameter_mapping")
    for k in FIELDS:
        rec[k] = out.get(k)
    if target.kind == "surrogate":
        for k in ("containment_basis", "world_member_basis"):
            if rec.get(k):
                rec[k] = "surrogate:" + str(rec[k])
    return rec
