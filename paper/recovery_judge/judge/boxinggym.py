"""BoxingGym judge (the recovery judge).

Target families (declared from the environments' generating code before any pool was read; the same truth
families as <development checkout> and the engine's
boxinggym_core._truth_* in <development checkout>):

  death       p(x) = 1 - exp(-theta x), theta > 0; observation Binomial(50, p) (mean 50 p = noise-free mean).
              Comparison scale: -log(1 - p) (= theta x for the truth; = log(1 + exp(eta)) for a logit-link
              candidate); numerics on p.
  dugongs     latent mean a - b c^x, a and b real, c = |l| > 0.  The environment returns |Normal(mean, 0.25)|, so the
              noise-free observation mean is the folded-Normal mean, which is NOT in the grammar; the judged target
              is the latent generating law (as bg_pools.py since 2026-10-05 20:4x).  Candidate mean U exp(eta)
              (U = the frozen D0 response unit).  Comparison scale: the mean itself.
  peregrines  Poisson mean exp(alpha + b1 x + b2 x^2 + b3 x^3), all real.  Comparison scale: log mean
              (= log U + eta for a candidate; log(1 - pi) added for zi_poisson).
  survival    p = expit(lambda0 * t * exp(beta m)), lambda0 > 0, beta real, m in {0, 1}.  Comparison scale:
              the logit, as the pair of slices m = 0 and m = 1 (the binary input is substituted).
  hyperbolic  p = eps + (1 - 2 eps) Phi((dR/(1 + k D) - iR)/alpha): NOT REPRESENTABLE - Phi (the normal CDF)
              is not an elementary function, so no finite composition of the grammar's calls (exp, log, sqrt,
              tanh, square, power) equals it, and the lapse eps is fixed by the environment.

Link handling: logit link -> p = expit(eta) (binomial, beta_binomial, bernoulli; zi_binomial multiplies by
1 - pi, pi a free coefficient in (0, 1)); log link -> mean = U exp(eta) (poisson, negative_binomial, lognormal,
gamma - the engine's "with mean" parametrisations, positive_inference.py lines 313-329; zi_poisson multiplies
by 1 - pi).  The noise family is NOT part of the judged family; noise_family_match records it separately.
"""
import json
from pathlib import Path

import sympy as sp

from . import core as C

BENCH = "boxinggym"
TARGET_VERSION = "v1"
CALLS = ("exp", "log", "sqrt", "tanh", "square", "power")
#: bg_pools.py THETAS (declared 2026-10-05 before any pool was read): the diagnostic target members
THETAS = {"death": [{"theta": 0.5}, {"theta": 1.0}, {"theta": 2.0}],
          "dugongs": [{"a": 2.0, "b": 1.5, "l": 0.4}, {"a": 2.5, "b": 1.0, "l": 0.6}, {"a": 3.0, "b": 2.0, "l": 0.3}],
          "peregrines": [{"alpha": 4.5, "beta1": 1.2, "beta2": 0.07, "beta3": -0.24},
                         {"alpha": 4.6, "beta1": 1.1, "beta2": 0.08, "beta3": -0.20},
                         {"alpha": 4.4, "beta1": 1.3, "beta2": 0.06, "beta3": -0.28}],
          "survival": [{"lambda0": 0.5, "beta": 1.0}, {"lambda0": 0.1, "beta": 2.0}, {"lambda0": 1.0, "beta": 0.5}]}
TRUTH_NOISE = {"death": "binomial", "peregrines": "poisson", "dugongs": "folded_normal(0.25)", "survival": "bernoulli",
               "hyperbolic": "bernoulli_with_lapse"}
NOT_REPRESENTABLE = {"hyperbolic": ("the truth eps + (1 - 2 eps) Phi((dR/(1 + k D) - iR)/alpha) uses the normal CDF Phi, "
                                    "which is not elementary; no finite composition of the grammar calls exp, log, sqrt, "
                                    "tanh, square, power equals it (and the lapse eps = 0.01 is fixed by the environment)")}


def _rat(v):
    return sp.Rational(repr(float(v)))


class Target:
    """The declared target of one BoxingGym world (task, seed: each seed draws its own parameters)."""

    def __init__(self, task, world_parameters, interface, response_unit=None):
        self.task, self.benchmark = task, BENCH
        self.target_id = f"{BENCH}/{task}/{TARGET_VERSION}"
        self.world_parameters = dict(world_parameters or {})
        self.interface = interface
        self.response_unit = response_unit
        self.representable = task not in NOT_REPRESENTABLE
        self.not_representable_reason = NOT_REPRESENTABLE.get(task)
        self.truth_noise = TRUTH_NOISE.get(task)
        self.latent_equals_observation_mean = task != "dugongs"
        ins = interface["inputs"]
        self.x = sp.Symbol("X1", positive=True)
        self.binary = None
        if task == "survival":
            self.binary = sp.Symbol("X2", real=True)
        if not self.representable:
            return
        x = self.x
        lo, hi = float(ins[0]["range"][0]), float(ins[0]["range"][1])
        self.box = C.Box({x: (max(lo, 1e-9) if lo <= 0 else lo, hi, "lin")}, f"{self.target_id}|box|{lo}|{hi}")
        P = lambda n, r: C.param_symbol(n, r, prefix="T_")
        if task == "death":
            th = P("theta", "positive")
            self.family = C.Family(th * x, [th], {th: "positive"}, [x], num=1 - sp.exp(-th * x))
            conv = lambda w: {th: float(w["theta"])}
        elif task == "dugongs":
            a, b, c = P("a", "signed"), P("b", "signed"), P("c", "positive")
            f = a - b * c ** x
            self.family = C.Family(f, [a, b, c], {a: "real", b: "real", c: "positive"}, [x])
            conv = lambda w: {a: float(w["a"]), b: float(w["b"]), c: abs(float(w["l"]))}
        elif task == "peregrines":
            al, b1, b2, b3 = (P(n, "signed") for n in ("alpha", "beta1", "beta2", "beta3"))
            f = al + b1 * x + b2 * x ** 2 + b3 * x ** 3
            self.family = C.Family(f, [al, b1, b2, b3], {p: "real" for p in (al, b1, b2, b3)}, [x])
            conv = lambda w: {al: float(w["alpha"]), b1: float(w["beta1"]), b2: float(w["beta2"]), b3: float(w["beta3"])}
        elif task == "survival":
            l0, be = P("lambda0", "positive"), P("beta", "signed")
            f = sp.Tuple(l0 * x, l0 * sp.exp(be) * x)
            self.family = C.Family(f, [l0, be], {l0: "positive", be: "real"}, [x])
            conv = lambda w: {l0: float(w["lambda0"]), be: float(w["beta"])}
        else:
            raise ValueError(f"no BoxingGym target for {task}")
        self.truth = conv(self.world_parameters)
        self.members = [self.truth] + [conv(m) for m in THETAS[task]]

    def describe(self):
        return {"target_id": self.target_id, "task": self.task, "representable": self.representable,
                "not_representable_reason": self.not_representable_reason,
                "family": None if not self.representable else str(self.family.sym),
                "truth": None if not self.representable else {k.name: v for k, v in self.truth.items()},
                "truth_noise_family": self.truth_noise,
                "latent_equals_observation_mean": self.latent_equals_observation_mean}


def candidate_family(candidate, target):
    """The candidate's mean on the target's comparison scale (core.Family)."""
    itf = target.interface
    x = target.x
    names = {"X1": x}
    if target.binary is not None:
        names["X2"] = target.binary
    for i, inp in enumerate(itf["inputs"][1:], start=2):
        names.setdefault(inp["name"], sp.Symbol(inp["name"], positive=True))
    for dn, formula in (itf.get("derived_inputs") or {}).items():
        names[dn] = C.parse_expression(formula, dict(names), calls=())
    roles = dict(candidate["parameters"])
    params = {n: C.param_symbol(n, r) for n, r in roles.items()}
    names.update(params)
    eta = C.parse_expression(candidate["expression"], names, calls=CALLS)
    family = candidate["family"]
    plist = [params[n] for n in roles]
    domains = {params[n]: C.role_domain(r) for n, r in roles.items()}
    pi = None
    if family in ("zi_binomial", "zi_poisson"):
        pi = C.param_symbol("zero_inflation_pi", "unit", prefix="S_")
        plist.append(pi)
        domains[pi] = "unit"
    link = itf.get("link")
    task = target.task
    if task == "death":
        if link != "logit":
            raise ValueError("death candidate without a logit link")
        p = 1 / (1 + sp.exp(-eta))
        if pi is None:
            sym = sp.log(1 + sp.exp(eta))
        else:
            p = (1 - pi) * p
            sym = -sp.log(1 - p)
        return C.Family(sym, plist, domains, [x], num=p, num_from_sym=lambda e: 1 - sp.exp(-e))
    if task == "survival":
        if link != "logit":
            raise ValueError("survival candidate without a logit link")
        b = target.binary
        if pi is not None:
            raise ValueError("zero inflation on a binary outcome")
        sl = sp.Tuple(eta.xreplace({b: 0}), eta.xreplace({b: 1}))
        return C.Family(sl, plist, domains, [x])
    U = C._literal(float(target.response_unit)) if target.response_unit is not None else sp.Integer(1)
    if link != "log":
        raise ValueError(f"{task} candidate without a log link")
    if task == "dugongs":
        m = U * sp.exp(eta) * (1 if pi is None else (1 - pi))
        return C.Family(m, plist, domains, [x])
    if task == "peregrines":
        lm = sp.log(U) + eta + (0 if pi is None else sp.log(1 - pi))
        return C.Family(lm, plist, domains, [x])
    raise ValueError(task)


def judge(candidate, target):
    """candidate: {expression, family, parameters: {name: role}, model_id?}; target: Target."""
    expr = candidate.get("expression")
    extra = {"benchmark": BENCH, "model_id": candidate.get("model_id"), "family": candidate.get("family"),
             "roles": candidate.get("parameters"), "noise_family_match": candidate.get("family") == target.truth_noise,
             "link": target.interface.get("link")}
    if not target.representable:
        return C.record("not_representable", target.target_id, expr, "R_NOT_REPRESENTABLE",
                        target.not_representable_reason, **extra)
    if not expr:
        return C.record("unresolved", target.target_id, expr, "U_MISSING_EXPRESSION",
                        "the record holds no expression for this model id", **extra)
    try:
        fam = candidate_family(candidate, target)
    except Exception as exc:  # noqa: BLE001
        return C.record("unresolved", target.target_id, expr, "U_PARSE", f"{type(exc).__name__}: {str(exc)[:160]}", **extra)
    key = json.dumps([target.target_id, sorted((k.name, v) for k, v in target.truth.items()),
                      str(target.response_unit), expr, candidate.get("family"),
                      sorted((candidate.get("parameters") or {}).items())])
    out = C.compare(fam, target.family, target.box, truth=target.truth, members=target.members, key=key)
    rec = C.record(out["state"], target.target_id, expr, out["reason"]["code"], out["reason"]["detail"], **extra)
    _fill(rec, out)
    return rec


def _fill(rec, out):
    rec["canonical_form"] = out.get("canonical_tree")
    rec["parameter_mapping"] = out.get("parameter_mapping")
    for k in ("containment", "containment_basis", "containment_witness", "numeric_containment_dev",
              "world_law_member", "world_member_basis", "world_member_dev", "truth_reachable", "domain_note",
              "dependence", "dims", "target_tree"):
        rec[k] = out.get(k)


# ------------------------------------------------------------------------------------------------ records
def response_unit_of(cell, config):
    """The frozen D0 response unit of a positive (log-link) task: the mean of the round-0 observations, the half-count
    rule when all are zero and the config records it applied (boxinggym_core.d0_response_unit)."""
    if (config.get("interface") or {}).get("link") != "log":
        return None
    p = Path(cell) / "frozen_d0_policy.json"
    if p.is_file():
        d = json.loads(p.read_text())
        if d.get("response_unit") is not None:
            return float(d["response_unit"])
    ys = []
    for line in (Path(cell) / "observations.jsonl").read_text().splitlines():
        r = json.loads(line)
        if int(r.get("round", -1)) == 0:
            ys.append(float(r["y"]))
    if not ys:
        return None
    if config.get("zero_count_unit_applied") and all(y == 0.0 for y in ys):
        return (sum(ys) + 0.5) / len(ys)
    return sum(ys) / len(ys)


def model_table(cell):
    """{model_id: {expression, family, parameters}} from cell/archive.json (every proposed model, retired included),
    with the fit records (fits/tNN_<id>.json) filling any id the archive lacks."""
    out = {}
    a = json.loads((Path(cell) / "archive.json").read_text())
    for e in a if isinstance(a, list) else []:
        if isinstance(e, dict) and e.get("model_id") and e.get("expression") and e["model_id"] not in out:
            out[e["model_id"]] = {"expression": e["expression"], "family": e.get("family"),
                                  "parameters": e.get("parameters") or {}, "source": "archive.json"}
    for f in sorted((Path(cell) / "fits").glob("t[0-9][0-9]_*.json")) if (Path(cell) / "fits").is_dir() else []:
        mid = f.stem.split("_", 1)[1]
        if mid in out:
            continue
        d = json.loads(f.read_text())
        if isinstance(d.get("parameters"), dict) and d.get("expression"):
            out[mid] = {"expression": d["expression"], "family": d.get("family"), "parameters": d["parameters"],
                        "source": f"fits/{f.name}"}
    return out
