"""ChemBench judge (the recovery judge).

Truth: the ChemBench oracle the runs used (LLM-AutoSciLab at 9a37255ff5432a7155b6cde8985e9654f2f8f272 = the manifest's
native_commit; the MIT-licensed 467fa184 differs from it only in LICENSE and README): observation = r0(C_A, C_I, C_B, C_P, Enz[, T])
x s(T, pH) x (1 + noise), with the primary rate law r0 of the domain and the fixed background nuisance
s(T, pH) = exp(-1000/R (1/T - 1/310)) (1 - 0.0025 (pH - 7)^2) (chembench.py _secondary_effects; no free constants).

Target family (declared from the oracle's rate functions, transcribed below and checked numerically against
chembench._RATE_FNS by test_chembench.py): the PRIMARY law r0 as a family with every named constant free, up to a
positive overall factor (kcat x response unit x nuisance value at the reference point are not judged - the same
scale-free reading as the scale-free reading of the error metric).  Arrhenius domains: T is a primary input and
exp(-Ea/R (1/T - 1/T_ref)) is the family exp(-E/T) up to the overall factor.

Nuisance handling : the candidate is judged on
the slice where the nuisance inputs sit at the oracle's reference point - pH = 7 always, T = T_ref = 310 K unless
the domain is Arrhenius (then T stays a primary input).  Allowed transformation: a positive overall factor.
Whether the candidate's log-mean splits into a primary-input part plus a nuisance-input part (as the truth's does)
is recorded separately (separable) and is not part of the state.  Candidate mean = response_unit * exp(expression)
(lognormal "with mean", the cell's events "mean" field).
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import sympy as sp

from . import core as C

BENCH = "chembench"
TARGET_VERSION = "v1"
ORACLE = None   # the LLM-AutoSciLab checkout: BMD_AUTOSCILAB (scripts/provision.sh --with chembench)
CALLS = ("exp", "log", "sqrt", "tanh", "square", "power")
R_GAS = 8.314
T_REF = 310.0
PH_REF = 7.0
INPUT_NAMES = {"X1": "C_A", "X2": "C_I", "X3": "C_B", "X4": "C_P", "X5": "Enz", "X6": "T", "X7": "pH"}
BOUNDS = {"X1": (0.01, 100.0, "log"), "X2": (0.0, 50.0, "lin"), "X3": (0.01, 100.0, "log"), "X4": (0.0, 20.0, "lin"),
          "X5": (0.01, 10.0, "log"), "X6": (278.0, 368.0, "lin"), "X7": (4.0, 10.0, "lin")}
XS = {n: sp.Symbol(n, positive=True) for n in INPUT_NAMES}
A, I, B, P, E, T = XS["X1"], XS["X2"], XS["X3"], XS["X4"], XS["X5"], XS["X6"]


def _p(name):
    return C.param_symbol(name, "positive", prefix="T_")


Km, Ki, Kp, KmA, KmB, KiA, Kh, n, al, be, Kth, Kmet, Ec = (_p(x) for x in (
    "Km", "Ki", "Kp", "KmA", "KmB", "KiA", "K_half", "n", "alpha", "beta", "Kthresh", "Km_met", "Ec"))
ARR = sp.exp(-Ec / T)
PING = A * B / (KmA * B + KmB * A + A * B)
ORD = A * B / (KiA * KmB + KmB * A + KmA * B + A * B)

#: domain -> (primary rate law without kcat (sympy, in A I B P E T), {oracle parameter name: symbol or (symbol, map)})
LAWS = {
    "c0_michaelis_menten": (E * A / (Km + A), {"Km": Km}),
    "c1_competitive_inhibition": (E * A / (Km * (1 + I / Ki) + A), {"Km": Km, "Ki": Ki}),
    "c2_product_inhibition": (E * A / (Km * (1 + P / Kp) + A), {"Km": Km, "Kp": Kp}),
    "c5_pingpong_bisubstrate": (E * PING, {"KmA": KmA, "KmB": KmB}),
    "c6_uncompetitive_inhibition": (E * A / (Km + A * (1 + I / Ki)), {"Km": Km, "Ki": Ki}),
    "c7_substrate_inhibition": (E * A / (Km + A + A ** 2 / Ki), {"Km": Km, "Ki": Ki}),
    "c8_hill_cooperativity": (E * A ** n / (Kh ** n + A ** n), {"K_half": Kh, "n": n}),
    "c9_noncompetitive_inhibition": (E * A / ((1 + I / Ki) * (Km + A)), {"Km": Km, "Ki": Ki}),
    "c14_mm_noncompetitive_arrhenius": (ARR * E * A / ((Km + A) * (1 + I / Ki)), {"Km": Km, "Ki": Ki, "Ea": Ec}),
    "c23_pingpong_arrhenius": (ARR * E * PING, {"KmA": KmA, "KmB": KmB, "Ea": Ec}),
    "c26_pingpong_noncompetitive": (E * PING / (1 + I / Ki), {"KmA": KmA, "KmB": KmB, "Ki": Ki}),
    "c33_hill_competitive": (E * A ** n / ((Kh * (1 + I / Ki)) ** n + A ** n), {"K_half": Kh, "n": n, "Ki": Ki}),
    "c35_hill_noncompetitive": (E * A ** n / ((Kh ** n + A ** n) * (1 + I / Ki)), {"K_half": Kh, "n": n, "Ki": Ki}),
    "c65_ordered_bi_bi": (E * ORD, {"KiA": KiA, "KmA": KmA, "KmB": KmB}),
    "c68_anticoop_hill": (E * A ** n / (Kh ** n + A ** n), {"K_half": Kh, "n": n}),
    "c69_fractal_kinetics": (E * A ** al, {"alpha": al}),
    "c71_coop_inhibition": (E * A / ((Km + A) * (1 + (I / Ki) ** n)), {"Km": Km, "Ki": Ki, "n_inh": n}),
    "c73_metal_activation": (E * A / (KmA + A) * B ** n / (Kmet ** n + B ** n), {"KmA": KmA, "Km_met": Kmet, "n_met": n}),
    "c74_product_activation": (E * A / (Km + A) * (Kth + P) / Kth, {"Km": Km, "Kthresh": Kth}),
    "c75_two_substrate_inhibition": (E * A / ((Km * (1 + I / Ki) + A) * (1 + P / Kp)), {"Km": Km, "Ki": Ki, "Kp": Kp}),
    "c76_ordered_bi_bi_arrhenius": (ARR * E * ORD, {"KiA": KiA, "KmA": KmA, "KmB": KmB, "Ea": Ec}),
    "c79_anticoop_arrhenius": (ARR * E * A ** n / (Kh ** n + A ** n), {"K_half": Kh, "n": n, "Ea": Ec}),
    "c89_ordered_bi_bi_competitive": (E * A * B / (KiA * (1 + I / Ki) * KmB + KmB * A + KmA * B + A * B),
                                      {"KiA": KiA, "KmA": KmA, "KmB": KmB, "Ki": Ki}),
    "c90_ordered_bi_bi_noncomp": (E * ORD / (1 + I / Ki), {"KiA": KiA, "KmA": KmA, "KmB": KmB, "Ki": Ki}),
    "c93_fractal_competitive": (E * A ** al / (1 + (I / Ki) ** be), {"alpha": al, "Ki": Ki, "beta": be}),
    "c94_metal_act_competitive": (E * A / (KmA * (1 + I / Ki) + A) * B ** n / (Kmet ** n + B ** n),
                                  {"KmA": KmA, "Ki": Ki, "Km_met": Kmet, "n_met": n}),
    "c98_two_sub_inh_arrhenius": (ARR * E * A / ((Km * (1 + I / Ki) + A) * (1 + P / Kp)),
                                  {"Km": Km, "Ki": Ki, "Kp": Kp, "Ea": Ec}),
}
ARRHENIUS = {d for d, (law, _) in LAWS.items() if law.has(Ec)}
#: a compact grammar spelling of each primary law (log of the mean, any extra literal-free scale in `a`), used to
#: check that the target is representable: <= 6 coefficients, <= 15 operators (count_models.validate_expression's
#: rule: every BinOp, UnaryOp and call counts one), calls exp/log/sqrt/tanh/square/power only.
SPELLINGS = {
    "c0_michaelis_menten": "a + log(X5*X1/(K + X1))",
    "c1_competitive_inhibition": "a + log(X5*X1/(K*(1 + X2/K2) + X1))",
    "c2_product_inhibition": "a + log(X5*X1/(K*(1 + X4/K2) + X1))",
    "c5_pingpong_bisubstrate": "a + log(X5*X1*X3/(K1*X3 + K2*X1 + X1*X3))",
    "c6_uncompetitive_inhibition": "a + log(X5*X1/(K + X1*(1 + X2/K2)))",
    "c7_substrate_inhibition": "a + log(X5*X1/(K + X1 + square(X1)/K2))",
    "c8_hill_cooperativity": "a + log(X5*power(X1, h)/(K + power(X1, h)))",
    "c9_noncompetitive_inhibition": "a + log(X5*X1/((1 + X2/K2)*(K + X1)))",
    "c14_mm_noncompetitive_arrhenius": "a - E/X6 + log(X5*X1/((K + X1)*(1 + X2/K2)))",
    "c23_pingpong_arrhenius": "a - E/X6 + log(X5*X1*X3/(K1*X3 + K2*X1 + X1*X3))",
    "c26_pingpong_noncompetitive": "a + log(X5*X1*X3/((K1*X3 + K2*X1 + X1*X3)*(1 + X2/K3)))",
    "c33_hill_competitive": "a + log(X5*power(X1, h)/(power(K*(1 + X2/K2), h) + power(X1, h)))",
    "c35_hill_noncompetitive": "a + log(X5*power(X1, h)/((K + power(X1, h))*(1 + X2/K2)))",
    "c65_ordered_bi_bi": "a + log(X5*X1*X3/(K1 + K2*X1 + K3*X3 + X1*X3))",
    "c68_anticoop_hill": "a + log(X5*power(X1, h)/(K + power(X1, h)))",
    "c69_fractal_kinetics": "a + log(X5) + h*log(X1)",
    "c71_coop_inhibition": "a + log(X5*X1/((K + X1)*(1 + power(X2/K2, h))))",
    "c73_metal_activation": "a + log(X5*X1*power(X3, h)/((K + X1)*(K2 + power(X3, h))))",
    "c74_product_activation": "a + log(X5*X1*(1 + X4/K2)/(K + X1))",
    "c75_two_substrate_inhibition": "a + log(X5*X1/((K*(1 + X2/K2) + X1)*(1 + X4/K3)))",
    "c76_ordered_bi_bi_arrhenius": "a - E/X6 + log(X5*X1*X3/(K1 + K2*X1 + K3*X3 + X1*X3))",
    "c79_anticoop_arrhenius": "a - E/X6 + log(X5*power(X1, h)/(K + power(X1, h)))",
    "c89_ordered_bi_bi_competitive": "a + log(X5*X1*X3/(K1*(1 + X2/K4) + K2*X1 + K3*X3 + X1*X3))",
    "c90_ordered_bi_bi_noncomp": "a + log(X5*X1*X3/((K1 + K2*X1 + K3*X3 + X1*X3)*(1 + X2/K4)))",
    "c93_fractal_competitive": "a + log(X5*power(X1, h)/(1 + power(X2/K2, g)))",
    "c94_metal_act_competitive": "a + log(X5*X1*power(X3, h)/((K*(1 + X2/K3) + X1)*(K2 + power(X3, h))))",
    "c98_two_sub_inh_arrhenius": "a - E/X6 + log(X5*X1/((K*(1 + X2/K2) + X1)*(1 + X4/K3)))",
}


def oracle():
    import os
    root = ORACLE or os.environ.get("BMD_AUTOSCILAB")
    if not root:
        raise ImportError("the ChemBench judge needs the LLM-AutoSciLab checkout: set BMD_AUTOSCILAB "
                          "(scripts/provision.sh --with chembench)")
    if root not in sys.path:
        sys.path.insert(0, root)
    with contextlib.redirect_stdout(io.StringIO()):
        from autoscilab.oracle import chembench as cb
    return cb


def count_operators(text):
    import ast
    return sum(isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call)) for nd in ast.walk(ast.parse(text, mode="eval")))


def representability(domain):
    """(representable, detail): the domain's spelling validates under the ChemBench grammar caps."""
    text = SPELLINGS.get(domain)
    if text is None:
        return False, "no transcription of this domain"
    import ast
    names = {nd.id for nd in ast.walk(ast.parse(text, mode="eval")) if isinstance(nd, ast.Name)}
    coefs = sorted(names - set(INPUT_NAMES) - {"log", "exp", "power", "square", "sqrt", "tanh"})
    ops = count_operators(text)
    ok = len(coefs) <= 6 and ops <= 15
    return ok, f"spelling {text!r}: {len(coefs)} coefficients, {ops} operators (caps 6 and 15)"


class Target:
    """The declared primary-law target of one ChemBench world (domain, difficulty, law version)."""

    def __init__(self, world, transform):
        self.world = world
        self.benchmark = BENCH
        domain, difficulty, version = world.split("__")
        self.domain, self.difficulty, self.version = domain, difficulty, version
        self.target_id = f"{BENCH}/{domain}/{TARGET_VERSION}"
        self.transform = transform
        self.arrhenius = domain in ARRHENIUS
        law, pmap = LAWS[domain]
        ok, detail = representability(domain)
        self.representable = ok
        self.not_representable_reason = None if ok else detail
        self.representability_detail = detail
        self.primary_inputs = sorted({s.name for s in law.free_symbols if s in XS.values()})
        self.signature_inputs = ["X1", "X2", "X3", "X4", "X5"] + (["X6"] if self.arrhenius else [])
        self.nuisance_inputs = (["X7"] if self.arrhenius else ["X6", "X7"])
        s = C.param_symbol("scale", "positive", prefix="T_")
        params = [s] + sorted({v for v in pmap.values()}, key=lambda z: z.name)
        self.family = C.Family(s * law, params, {p: "positive" for p in params},
                               [XS[x] for x in self.signature_inputs], num=sp.log(s) + sp.log(law))
        spec = {XS[x]: BOUNDS[x] for x in self.signature_inputs}
        self.box = C.Box(spec, f"{self.target_id}|{','.join(self.signature_inputs)}")
        cb = oracle()
        true = cb._PARAMS[domain][difficulty][version]
        self.oracle_params = dict(true)
        truth = {s: float(true.get("kcat", true.get("kcat_ref", 1.0)))}
        for name, sym in pmap.items():
            truth[sym] = float(true[name]) / (R_GAS if name == "Ea" else 1.0)
        self.truth = truth
        self.members = [truth] + [{k: v * f for k, v in truth.items()} for f in (0.6, 1.7)]

    def describe(self):
        return {"target_id": self.target_id, "world": self.world, "representable": self.representable,
                "representability": self.representability_detail, "primary_law": str(self.family.sym),
                "primary_inputs": self.primary_inputs, "signature_inputs": self.signature_inputs,
                "nuisance_inputs_fixed": {x: (T_REF if x == "X6" else PH_REF) for x in self.nuisance_inputs},
                "truth": {k.name: v for k, v in self.truth.items()}, "oracle_params": self.oracle_params}


def derived_names(transform):
    names = dict(XS)
    for i, x in enumerate(["X1", "X2", "X3", "X4", "X5", "X6", "X7"]):
        c, s = C._literal(float(transform["center"][i])), C._literal(float(transform["scale"][i]))
        names[x + "z"] = (XS[x] - c) / s
    return names


def candidate_eta(candidate, transform):
    names = derived_names(transform)
    roles = dict(candidate["parameters"])
    params = {nm: C.param_symbol(nm, r) for nm, r in roles.items()}
    names.update(params)
    eta = C.parse_expression(candidate["expression"], names, calls=CALLS)
    return eta, params, roles


def separability(eta, target):
    """True when the log-mean is a sum of primary-signature-only, nuisance-only and constant terms; False when a
    term couples the two groups; the nuisance part as text."""
    e = sp.expand(sp.expand_log(sp.expand(eta, power_exp=True, mul=True, multinomial=False, force=True), force=True))
    prim = {XS[x] for x in target.signature_inputs}
    nuis = {XS[x] for x in target.nuisance_inputs}
    coupled, nuis_terms = [], []
    for t in sp.Add.make_args(e):
        fs = t.free_symbols
        if fs & prim and fs & nuis:
            coupled.append(t)
        elif fs & nuis:
            nuis_terms.append(t)
    return (not coupled), (str(sp.Add(*nuis_terms)) if nuis_terms else "0"), [str(c) for c in coupled[:3]]


def candidate_family(candidate, target):
    eta, params, roles = candidate_eta(candidate, target.transform)
    fixed = {XS[x]: C._literal(T_REF if x == "X6" else PH_REF) for x in target.nuisance_inputs}
    eta_p = eta.xreplace(fixed)
    sig = C.param_symbol("overall_factor", "positive", prefix="S_")
    plist = [params[nm] for nm in roles] + [sig]
    domains = {params[nm]: C.role_domain(r) for nm, r in roles.items()}
    domains[sig] = "positive"
    fam = C.Family(sig * sp.exp(eta_p), plist, domains, [XS[x] for x in target.signature_inputs],
                   num=sp.log(sig) + eta_p, num_from_sym=lambda e: sp.expand_log(sp.log(e), force=True), scale=sig)
    return fam, eta


def judge(candidate, target):
    expr = candidate.get("expression")
    extra = {"benchmark": BENCH, "model_id": candidate.get("model_id"), "family": candidate.get("family"),
             "roles": candidate.get("parameters"), "noise_family_match": None,
             "nuisance_rule": f"pH = {PH_REF:g}" + ("" if target.arrhenius else f", T = {T_REF:g} K") +
                              "; a positive overall factor is not judged"}
    if not target.representable:
        return C.record("not_representable", target.target_id, expr, "R_NOT_REPRESENTABLE",
                        target.not_representable_reason, **extra)
    if not expr or candidate.get("parameters") is None:
        return C.record("unresolved", target.target_id, expr, "U_MISSING_EXPRESSION",
                        "the record holds no expression or no coefficient roles for this model id", **extra)
    try:
        fam, eta = candidate_family(candidate, target)
        sep, nuis, coupled = separability(eta, target)
    except Exception as exc:  # noqa: BLE001
        return C.record("unresolved", target.target_id, expr, "U_PARSE", f"{type(exc).__name__}: {str(exc)[:160]}", **extra)
    extra.update(separable=sep, nuisance_part=nuis, coupled_terms=coupled)
    key = json.dumps([target.target_id, target.world, sorted(target.transform["center"]), expr,
                      sorted((candidate.get("parameters") or {}).items())])
    out = C.compare(fam, target.family, target.box, truth=target.truth, members=target.members, key=key)
    rec = C.record(out["state"], target.target_id, expr, out["reason"]["code"], out["reason"]["detail"], **extra)
    rec["canonical_form"] = out.get("canonical_tree")
    rec["parameter_mapping"] = out.get("parameter_mapping")
    for k in ("containment", "containment_basis", "containment_witness", "numeric_containment_dev",
              "numeric_members_evaluated", "world_law_member", "world_member_basis", "world_member_dev",
              "truth_reachable", "domain_note", "dependence", "dims", "target_tree"):
        rec[k] = out.get(k)
    return rec


# ------------------------------------------------------------------------------------------------ records
def model_table(cell):
    """{model_id: {expression, family, parameters}} from the cell's proposal events (burst, rounds; revived models
    are earlier events), plus any revival record's entries."""
    out = {}
    pdir = Path(cell) / "proposals"
    files = sorted(pdir.glob("t[0-9][0-9]_burst.json")) + sorted(pdir.glob("t[0-9][0-9].json"))
    for f in files:
        d = json.loads(f.read_text())
        for ev in d.get("events") or []:
            mid = ev.get("model_id")
            if mid and ev.get("expression") and isinstance(ev.get("parameters"), dict) and mid not in out:
                out[mid] = {"expression": ev["expression"], "family": ev.get("family"), "parameters": ev["parameters"],
                            "source": f"proposals/{f.name}"}
    return out
