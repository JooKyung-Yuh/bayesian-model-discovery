"""ForceBench judge (the recovery judge).

Truth: DiscoverPhysics (local clone <development checkout> at 33b7fa9df96d = the runs'
dp_commit), ScienceAgent/scienceagent/worlds.py WORLDS and executor.py; the run record's truth_law.  The judged object is
the full law of motion of particle 2 - its acceleration as a function of the distance r, the time t and BOTH control
inputs p1, p2 (coupling) - because the engine integrates a candidate force F as acceleration F / p2
(fb1_forward.integrate: acc = -(f / p2) x / r, the protected file of the ForceBench kit tree
<development checkout>, commit 33f2cf7).

Target families (every named constant free; declared values in brackets):
  gravity           a = c p1 / (p2 r)                          [c = 1/(2 pi): 2-D Poisson force p1/(2 pi r), inertia p2]
  coulomb_easy      a = c p1 p2 / r^2                          [c = k = 1; inertia 1 (worlds.py: "particle 1 mobile, inertia 1")]
  oscillator        a = (A cos(w t) + B sin(w t)) p1 / (p2 r)  [G(t) = G0 cos(w t + phi), G0 = 5, w = pi/2, phi = 0:
                                                                 A = G0 cos(phi)/(2 pi), B = -G0 sin(phi)/(2 pi)]
  yukawa            NOT REPRESENTABLE: the 2-D Helmholtz force is proportional to K1(r/lambda) (modified Bessel), not an
                    elementary function; the grammar's calls are exp, log, sqrt, tanh, square, power, cos, sin.
  extra_dimensions  NOT REPRESENTABLE: the Kaluza-Klein image sum (record law_str: 41 terms, the world defines the infinite
                    tower) has no elementary closed form and its finite truncation exceeds the free-form caps (at most 3
                    free constants and 10 operators, fb1_grammar.MAX_FREE_CONSTANTS / N_OPS_MAX).
Not part of the family: the simulator's force softening (nbody worlds evaluate the kernel at sqrt(r^2 + 0.05^2),
fb1_forward.TRUTH_SOFTENING) and the field worlds' periodic-box background - a candidate with a free softening or
background term is a strictly larger family (containment), not the declared law.  Noise family (gauss / student_t4)
is recorded, not judged.
"""
import json
import math
from pathlib import Path

import sympy as sp

from . import core as C

BENCH = "forcebench"
TARGET_VERSION = "v1"
CALLS = ("exp", "log", "sqrt", "tanh", "square", "power", "cos", "sin")
#: fb1_grammar.CATALOG_GRAMMAR_EXPRESSIONS of the ForceBench kit tree (catalogue ids that a free proposal can transcribe to)
CATALOGUE = {
    "inv_r2": ("c*p1/r**2", {"c": "positive"}),
    "linear": ("c*p1*r", {"c": "positive"}),
    "const": ("c*p1", {"c": "positive"}),
    "screened": ("c*p1*exp(-r/lam)/r", {"c": "positive", "lam": "positive"}),
    "inv_r": ("c*p1/r", {"c": "positive"}),
    "power": ("c*p1*r**(-a)", {"c": "positive", "a": "signed"}),
    "inv_r_plus_c": ("c*p1/r + b*p1", {"c": "positive", "b": "positive"}),
    "logr": ("c*p1*log(1+k*r)/r", {"c": "positive", "k": "positive"}),
    "inv_r_p1sq": ("c*p1**2/r", {"c": "positive"}),
    "inv_r_nomass": ("c*p1*p2/r", {"c": "positive"}),
    "inv_r_bg": ("c*p1/r - b*p1*r", {"c": "positive", "b": "positive"}),
    "prod_coupling": ("c*p1*p2**2/r**k", {"c": "positive", "k": "positive"}),
    "crossover": ("c1*p1/r + c2*p1*exp(-r/R)/r**2", {"c1": "positive", "c2": "positive", "R": "positive"}),
    "inv_r_tmod": ("c*p1/r*(1 + A1*cos(w*t) + A2*sin(w*t))",
                   {"c": "positive", "A1": "signed", "A2": "signed", "w": "positive"}),
}
NOT_REPRESENTABLE = {
    "yukawa": ("the 2-D Helmholtz (screened) force is proportional to the modified Bessel function K1(r/lambda), which is "
               "not elementary; no finite composition of exp, log, sqrt, tanh, square, power, cos, sin equals it"),
    "extra_dimensions": ("the Kaluza-Klein periodic-image sum has no elementary closed form and its 41-term truncation "
                         "(record law_str) exceeds the free-form caps of 3 free constants and 10 operators"),
}
R = sp.Symbol("r", positive=True)
P1 = sp.Symbol("p1", positive=True)
P2 = sp.Symbol("p2", positive=True)
TT = sp.Symbol("t", positive=True)
BOX = {R: (0.3, 10.0, "log"), P1: (0.1, 10.0, "lin"), P2: (0.1, 10.0, "lin"), TT: (0.0, 5.0, "lin")}


class Target:
    def __init__(self, world):
        self.world = world
        self.benchmark = BENCH
        self.target_id = f"{BENCH}/{world}/{TARGET_VERSION}"
        self.representable = world not in NOT_REPRESENTABLE
        self.not_representable_reason = NOT_REPRESENTABLE.get(world)
        if not self.representable:
            return
        P = lambda n, r: C.param_symbol(n, r, prefix="T_")
        if world == "gravity":
            c = P("c", "positive")
            self.family = C.Family(c * P1 / (P2 * R), [c], {c: "positive"}, [R, P1, P2])
            self.truth = {c: 1 / (2 * math.pi)}
            self.members = [self.truth, {c: 0.5}, {c: 2.0}]
            inputs = [R, P1, P2]
        elif world == "coulomb_easy":
            c = P("c", "positive")
            self.family = C.Family(c * P1 * P2 / R ** 2, [c], {c: "positive"}, [R, P1, P2])
            self.truth = {c: 1.0}
            self.members = [self.truth, {c: 0.4}, {c: 2.5}]
            inputs = [R, P1, P2]
        elif world == "oscillator":
            a, b, w = P("A", "signed"), P("B", "signed"), P("w", "positive")
            f = (a * sp.cos(w * TT) + b * sp.sin(w * TT)) * P1 / (P2 * R)
            self.family = C.Family(f, [a, b, w], {a: "real", b: "real", w: "positive"}, [R, P1, P2, TT])
            self.truth = {a: 5.0 / (2 * math.pi), b: 0.0, w: math.pi / 2}
            self.members = [self.truth, {a: 0.5, b: 0.3, w: 1.0}, {a: -0.4, b: 0.6, w: 2.2}]
            inputs = [R, P1, P2, TT]
        else:
            raise ValueError(world)
        self.box = C.Box({s: BOX[s] for s in [R, P1, P2, TT]}, f"{self.target_id}|box")
        self.inputs = inputs

    def describe(self):
        return {"target_id": self.target_id, "world": self.world, "representable": self.representable,
                "not_representable_reason": self.not_representable_reason,
                "family_acceleration": None if not self.representable else str(self.family.sym),
                "truth": None if not self.representable else {k.name: v for k, v in self.truth.items()}}


def resolve(model_id, registered):
    """(structure id, noise family, expression, roles, source) of a pool id 'structure|family'."""
    sid, _, fam = model_id.partition("|")
    if sid in registered:
        e = registered[sid]
        return sid, fam, e.get("expression"), dict(e.get("parameters") or {}), "registered"
    if sid in CATALOGUE:
        expr, roles = CATALOGUE[sid]
        return sid, fam, expr, dict(roles), "catalogue"
    return sid, fam, None, None, "unknown"


def candidate_family(candidate):
    roles = dict(candidate["parameters"])
    params = {n: C.param_symbol(n, r) for n, r in roles.items()}
    names = {"r": R, "p1": P1, "p2": P2, "t": TT, **params}
    force = C.parse_expression(candidate["expression"], names, calls=CALLS)
    acc = force / P2
    plist = [params[n] for n in roles]
    return C.Family(acc, plist, {params[n]: C.role_domain(r) for n, r in roles.items()}, [R, P1, P2, TT])


def judge(candidate, target):
    expr = candidate.get("expression")
    extra = {"benchmark": BENCH, "model_id": candidate.get("model_id"), "family": candidate.get("family"),
             "roles": candidate.get("parameters"), "noise_family_match": None,
             "structure_source": candidate.get("source"), "judged_object": "acceleration F/p2"}
    if not target.representable:
        return C.record("not_representable", target.target_id, expr, "R_NOT_REPRESENTABLE",
                        target.not_representable_reason, **extra)
    if not expr:
        return C.record("unresolved", target.target_id, expr, "U_MISSING_EXPRESSION",
                        "the run record holds no expression for this structure id", **extra)
    try:
        fam = candidate_family(candidate)
    except Exception as exc:  # noqa: BLE001
        return C.record("unresolved", target.target_id, expr, "U_PARSE", f"{type(exc).__name__}: {str(exc)[:160]}", **extra)
    key = json.dumps([target.target_id, expr, sorted((candidate.get("parameters") or {}).items())])
    out = C.compare(fam, target.family, target.box, truth=target.truth, members=target.members, key=key)
    rec = C.record(out["state"], target.target_id, expr, out["reason"]["code"], out["reason"]["detail"], **extra)
    rec["canonical_form"] = out.get("canonical_tree")
    rec["parameter_mapping"] = out.get("parameter_mapping")
    for k in ("containment", "containment_basis", "containment_witness", "numeric_containment_dev",
              "numeric_members_evaluated", "world_law_member", "world_member_basis", "world_member_dev",
              "truth_reachable", "domain_note", "dependence", "dims", "target_tree"):
        rec[k] = out.get(k)
    # the two factors the site's form-class reading separates, beside the full judgment
    rec.update(_factors(fam, target))
    return rec


def _factors(fam, target):
    """rt_slice: the canonical relation of the candidate to the target at p1 = p2 = 1 (the dependence on r and t
    alone); coupling_same: whether A(p1, p2) / A(1, 1) equals the target's coupling factor exactly."""
    out = {"rt_slice": None, "coupling_same": None}
    try:
        sl = {P1: sp.Integer(1), P2: sp.Integer(1)}
        c = C.Family(fam.sym.xreplace(sl), fam.params, fam.domains, [R, TT]).with_sub({})
        t = C.Family(target.family.sym.xreplace(sl), target.family.params, target.family.domains, [R, TT]).with_sub({})
        box = C.Box({R: BOX[R], TT: BOX[TT]}, target.target_id + "|rt-slice")
        out["rt_slice"] = C.match_families(c, t, box, target.target_id + "|rt|" + str(fam.sym))["kind"] or "different"
    except Exception as exc:  # noqa: BLE001
        out["rt_slice"] = f"error: {type(exc).__name__}"
    try:
        ratio_c = C._tidy(fam.sym / fam.sym.xreplace({P1: sp.Integer(1), P2: sp.Integer(1)}))
        ratio_t = C._tidy(target.family.sym / target.family.sym.xreplace({P1: sp.Integer(1), P2: sp.Integer(1)}))
        out["coupling_same"] = bool(C._is_zero(ratio_c - ratio_t))
    except Exception:  # noqa: BLE001
        out["coupling_same"] = None
    return out


# ------------------------------------------------------------------------------------------------ records
def run_record(folder, world, arm_code, seed):
    """(record dict, path) of a ForceBench run folder: the snapshot's u3_fb_compact.json, else the full record."""
    compact = Path(folder) / "u3_fb_compact.json"
    if compact.is_file():
        return json.loads(compact.read_text()), compact
    hits = sorted(Path(folder).glob(f"FB1_RUN_{world}_*_seed{seed}_v3_free.json"))
    if len(hits) == 1:
        return json.loads(hits[0].read_text()), hits[0]
    return None, None


def model_table(run):
    reg = {e["structure"]: e for e in ((run.get("arm_config") or {}).get("structure_proposals") or {}).get("registered") or []
           if e.get("structure")}
    return reg
