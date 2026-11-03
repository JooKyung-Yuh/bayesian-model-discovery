"""the recovery judge core: expression parsing, numeric family diagnostics, canonical-tree unification and the
generic family comparison every benchmark module calls.  Deterministic: the decision rules never read a clock,
every random draw comes from a numpy Generator seeded by sha256 of the target id and the candidate, and the batch
runner sets PYTHONHASHSEED=0 so sympy's set iteration order is fixed.  No language model is used anywhere.

Decision rules (rule_version below; REPORT_C5_JUDGE.md states them in prose).  Both functions are first written on
the target's comparison scale (benchmark modules: link, nuisance slice, coupling), then:

  equivalent      the two reduce to the same canonical tree (deterministic identities: exp of sums split, logs of
                  products split, one fraction with a normalised denominator, leading coefficients pulled out of sums
                  under log and powers, cos/sin of sums expanded; sums and products compared as multisets; every
                  maximal input-free subexpression is one slot) and the slot equations solve BOTH ways (one unknown
                  at a time), a 30-digit spot test agrees, and the coefficient domains agree: the candidate's roles
                  reach the world's own parameters and every declared diagnostic member (else N_DOMAIN_NARROWER), and
                  no finite candidate member maps outside the target's declared parameter domain (else
                  N_DOMAIN_WIDER, which is also containment).
  non_equivalent  N_DEPENDENCE (the two depend on different inputs: the function changes when one input's values
                  are permuted), N_DIMENSION (different effective dimension = generic rank of the Jacobian w.r.t. the
                  coefficients on the diagnostic inputs), N_NUMERIC_WITNESS (same inputs and dimension, no canonical
                  match, and the candidate does not reproduce a target member within WITNESS_TOL after STARTS
                  least-squares starts), N_DOMAIN* (same canonical family, different coefficient domain), or a
                  benchmark rule (NeuronBench shape and kind).
  unresolved      U_PARSE / U_EVAL / U_MISSING_*: the candidate could not be read or evaluated; U_SYMBOLIC_UNPROVEN:
                  same inputs and dimension, the candidate reproduces every target member within EXACT_TOL, but no
                  canonical proof was found; U_NUMERIC_AMBIGUOUS: the deviation lies between EXACT_TOL and
                  WITNESS_TOL.  Never counted as recovered.
  not_representable  the target family is outside the benchmark's grammar or vocabulary (a world-level fact decided
                  before any candidate is read); every candidate of that world carries it.

  containment     (separate field; the candidate family strictly contains the target family) basis "symbolic": a
                  bounded substitution (a coefficient set to 0, 1 or infinity, or two set equal; the refit's special
                  values first, then single and paired substitutions, at most MAX_SUB_ATTEMPTS comparisons) turns the
                  candidate into a family equivalent to the target that reaches every declared member; "symbolic_mapping":
                  the target parameters solve into the candidate's for every member (target inside candidate) though
                  the reverse fails; "domain": N_DOMAIN_WIDER; "numeric": no symbolic witness, but the candidate
                  reproduces every target member within EXACT_TOL on the check inputs after fitting on the separate
                  diagnostic inputs (plan p.11's procedure).
  world_law_member (separate field) the candidate family holds the world's own law (the target at the true
                  parameters): from the solved mapping, a containment witness, a restricted family's fixed slots, or a
                  refit to the true member (EXACT_TOL).
"""
import ast
import hashlib
import itertools
import math
import warnings

import mpmath  # sympy 1.14.0 and mpmath 1.4.1 are the versions the paper's judgments were made with (the judge extra)
import numpy as np
import sympy as sp
from scipy.optimize import least_squares

RULE_VERSION = "c5-judge-1.3"
#: 1.0 -> 1.1 (2026-10-09, after the first full run, before any result was quoted): the canonical-form and solver
#: machinery only - numerators regrouped by input part, exp(k log u) written u**k, common monomial factors taken out
#: of numerator and denominator, sin/cos of sums by the addition formulas only (expand_trig's multiple-angle
#: expansion of tanh(100000 x) hung), positivity kept in the solving dummies, two-term power equations solved as logs,
#: no cubic-or-higher radical solves, bounded equation growth, no general limit algorithm in substitutions.  No
#: threshold, tolerance, target family or decision rule changed.
#: 1.1 -> 1.2 (2026-10-09 10:1x KST, after the second full run, before any result was quoted): no Lambert-type solves -
#: an equation whose unknown sits both inside a function or an exponent and outside it (u*exp(u) = c) is not handed
#: to sympy's solver (its Lambert W path ran past the 180 s guard twice in the second run, which made those two
#: judgments depend on time).  Solver machinery only; no threshold, tolerance, target family or decision rule changed.
#: 1.2 -> 1.3 (2026-10-09 10:5x KST; the 1.2 run was stopped at cell 33 of 396 when the gaps below were found in the
#: second run's unresolved list, before any result was quoted):
#:   machinery - powers of products are split before fractions are combined (1/(1 + (K/x)^n) and x^n/(K^n + x^n)
#:   now reach the same tree), powers of one input are merged (x^n x^(c k) = x^(n + c k)), and the zero test also
#:   compares log A with log B for a two-term A - B whose terms sympy knows to be positive (no force=True), an equation
#:   that has become an identity after the earlier solutions is dropped instead of solved, a rational factor stays
#:   outside a frozen constant (3100 exp(7) -> 3100 k, so cancel() sees the relation), and the 30-digit spot check
#:   draws only coefficient values inside the declared domains (mapped coefficients included) and needs at least
#:   one finite comparison;
#:   decision rule - one solved direction is no longer read as non-equivalence: with equal dimensions,
#:   N_DOMAIN_NARROWER needs a target member the candidate does not reproduce (> WITNESS_TOL) and N_DOMAIN_WIDER a
#:   finite candidate member the target family does not reproduce (> WITNESS_TOL); otherwise the state is
#:   unresolved (U_SYMBOLIC_UNPROVEN at <= EXACT_TOL, U_NUMERIC_AMBIGUOUS between).  No threshold or tolerance value
#:   and no target family changed.
STATES = ("equivalent", "non_equivalent", "unresolved", "not_representable")

# ------------------------------------------------------------------------------------------------ thresholds
RANK_REL = 1e-8          # singular value / largest singular value above this = an independent direction
RANK_AMBIG = 1e-11       # a ratio in [RANK_AMBIG, RANK_REL] makes the rank ambiguous
DEP_REL = 1e-9           # an input matters when permuting its values changes the function by more than this
EXACT_TOL = 1e-6         # numeric reproduction on the check inputs (containment basis "numeric", membership)
GATE_TOL = 1e-2          # numeric gate before the symbolic containment search (a boundary limit converges slowly)
WITNESS_TOL = 1e-3       # numeric non-reproduction witness (N_NUMERIC_WITNESS)
N_DIAG, N_CHECK = 96, 96
STARTS = 8
MAX_NFEV_PER_PARAM = 40
EARLY_EXIT = 1e-10       # a start that reproduces the target this closely ends the multi-start search
MAX_SUB_ATTEMPTS = 60    # canonical comparisons one containment search may spend
MAX_SOLVE_OPS = 120      # largest slot equation handed to sympy's solver
MAX_SOLVE_DEGREE = 2     # highest polynomial degree in the unknown the solver is asked to invert
MAX_FORM_OPS = 150       # largest candidate expression the canonical normal form is attempted on
MAX_TIDY_OPS = 400       # largest expression cancel() is applied to
MAX_SIMPLIFY_OPS = 80    # largest expression simplify() is tried on (a zero proof)
BIG_INT = 10 ** 6        # a rational with a larger numerator or denominator becomes a constant symbol
DRAWS = 3
MAX_UNIFIERS = 24
SPOT_DIGITS = 30
SPOT_POINTS = 5


def seed_of(*parts):
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16], 16)


# ------------------------------------------------------------------------------------------------ parsing
def _literal(value):
    if isinstance(value, bool):
        raise ValueError("boolean literal")
    if isinstance(value, int):
        return sp.Integer(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite literal")
        return sp.Rational(repr(value))           # the written decimal, exactly
    raise ValueError(f"literal of type {type(value).__name__}")


CALLS = {
    "exp": (1, sp.exp), "log": (1, sp.log), "sqrt": (1, sp.sqrt), "tanh": (1, sp.tanh),
    "square": (1, lambda a: a ** 2), "power": (2, lambda a, b: a ** b), "abs": (1, sp.Abs),
    "sigmoid": (1, lambda a: 1 / (1 + sp.exp(-a))), "expm1": (1, lambda a: sp.exp(a) - 1),
    "log1p": (1, lambda a: sp.log(1 + a)), "minimum": (2, sp.Min), "maximum": (2, sp.Max),
    "cos": (1, sp.cos), "sin": (1, sp.sin),
}


def parse_expression(text, names, calls=None):
    """Grammar text -> sympy expression.  `names` maps every allowed identifier (inputs, derived inputs,
    coefficients) to a sympy expression; `calls` restricts the call vocabulary (default: every call above)."""
    allowed = CALLS if calls is None else {k: CALLS[k] for k in calls}
    tree = ast.parse(str(text), mode="eval")

    def conv(node):
        if isinstance(node, ast.Expression):
            return conv(node.body)
        if isinstance(node, ast.Constant):
            return _literal(node.value)
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            if node.id == "pi":
                return sp.pi
            raise ValueError(f"unknown name {node.id}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = conv(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            a, b = conv(node.left), conv(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.Pow):
                return a ** b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            spec = allowed.get(node.func.id)
            if spec and len(node.args) == spec[0]:
                return spec[1](*[conv(arg) for arg in node.args])
        raise ValueError(f"unsupported syntax: {ast.dump(node)[:120]}")
    return conv(tree)


def param_symbol(name, role, prefix="p_"):
    """A coefficient symbol: positive and shape roles are positive, intercept and signed are real, unit is (0, 1).
    Candidate coefficients carry the prefix p_, target parameters T_, the judge's own auxiliary coefficients S_, so a
    candidate coefficient never shares a sympy symbol with a target parameter of the same written name."""
    clean = prefix + "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
    if role in ("positive", "shape", "unit"):
        return sp.Symbol(clean, positive=True)
    return sp.Symbol(clean, real=True)


def role_domain(role):
    return {"positive": "positive", "shape": "positive", "unit": "unit"}.get(role, "real")


# ------------------------------------------------------------------------------------------------ families
class Family:
    """A function family on the comparison scale.

    sym   sympy expression (or sympy Tuple of expressions: one per value of a binary input) used for the canonical
          tree; num the expression used for numerics (the same function or a fixed bijection of it, e.g. a log);
    params / domains  coefficient symbols and their domains ('positive', 'real', 'unit');
    inputs  the input symbols the expressions are written in."""

    def __init__(self, sym, params, domains, inputs, num=None, label="", num_from_sym=None, scale=None):
        self.sym = sym
        self.num = sym if num is None else num
        self.params = tuple(params)
        self.domains = {p: domains[p] for p in self.params}
        self.inputs = tuple(inputs)
        self.label = label
        self.num_from_sym = num_from_sym if num is not None else (lambda e: e)
        self.scale = scale if scale in self.params else None    # a free positive overall factor of sym, if any
        self._f = None
        self._g = None

    def parts(self, which="num"):
        e = self.num if which == "num" else self.sym
        return list(e) if isinstance(e, sp.Tuple) else [e]

    def compile(self):
        if self._f is None:
            syms = list(self.params) + list(self.inputs)
            self._f = [sp.lambdify(syms, part, modules=_MODULES) for part in self.parts()]
        return self._f

    def compile_grad(self):
        if self._g is None:
            syms = list(self.params) + list(self.inputs)
            self._g = [[sp.lambdify(syms, sp.diff(part, p), modules=_MODULES) for p in self.params]
                       for part in self.parts()]
        return self._g

    def values(self, a, X):
        n = len(X[0]) if X else 1
        out = []
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for f in self.compile():
                try:
                    v = np.asarray(f(*a, *X), dtype=float)
                except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                    v = np.full(n, np.nan)
                out.append(np.broadcast_to(v, (n,)).astype(float))
        return np.concatenate(out)

    def jacobian(self, a, X):
        n = len(X[0]) if X else 1
        cols = []
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grads = self.compile_grad()
            for j in range(len(self.params)):
                col = []
                for g in grads:
                    try:
                        v = np.asarray(g[j](*a, *X), dtype=float)
                    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                        v = np.full(n, np.nan)
                    col.append(np.broadcast_to(v, (n,)))
                cols.append(np.concatenate(col))
        return np.stack(cols, axis=1) if cols else np.zeros((n * len(self.parts()), 0))

    def with_sub(self, mapping, label=None):
        """The family after substituting coefficient values/expressions (removed coefficients drop out)."""
        sym = self.sym.xreplace(mapping) if mapping else self.sym
        num = self.num.xreplace(mapping) if mapping else self.num
        free = sym.free_symbols | (num.free_symbols if hasattr(num, "free_symbols") else set())
        params = [p for p in self.params if p in free]
        return Family(sym, params, {p: self.domains[p] for p in params}, self.inputs, num=num,
                      label=label or self.label, num_from_sym=self.num_from_sym, scale=self.scale)


_MODULES = [{"Min": np.minimum, "Max": np.maximum, "Abs": np.abs}, "numpy"]


class Box:
    """Input box of a target: {symbol: (lo, hi, kind)} with kind 'lin', 'log' or 'binary' (0/1)."""

    def __init__(self, spec, key):
        self.spec = dict(spec)
        self.key = key

    def grid(self, n, which):
        rng = np.random.default_rng(seed_of("grid", self.key, which))
        cols = []
        for sym, (lo, hi, kind) in self.spec.items():
            if kind == "log":
                cols.append(np.exp(rng.uniform(math.log(lo), math.log(hi), n)))
            elif kind == "binary":
                cols.append(rng.integers(0, 2, n).astype(float))
            else:
                cols.append(rng.uniform(lo, hi, n))
        return cols

    def symbols(self):
        return list(self.spec)


def draw_params(fam, rng):
    out = []
    for p in fam.params:
        d = fam.domains[p]
        if d == "positive":
            out.append(float(np.exp(rng.uniform(-1.0, 1.0))))
        elif d == "unit":
            out.append(float(rng.uniform(0.15, 0.85)))
        else:
            out.append(float(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.5)))
    return out


def _from_u(fam, u):
    a = []
    for p, x in zip(fam.params, u):
        d = fam.domains[p]
        a.append(math.exp(min(x, 700.0)) if d == "positive" else (1.0 / (1.0 + math.exp(-x)) if d == "unit" else x))
    return a


def _da_du(fam, a):
    out = []
    for p, x in zip(fam.params, a):
        d = fam.domains[p]
        out.append(x if d == "positive" else (x * (1.0 - x) if d == "unit" else 1.0))
    return np.asarray(out)


def _to_u(fam, a):
    u = []
    for p, x in zip(fam.params, a):
        d = fam.domains[p]
        if d == "positive":
            u.append(math.log(x) if x > 0 else None)
        elif d == "unit":
            u.append(math.log(x / (1.0 - x)) if 0 < x < 1 else None)
        else:
            u.append(x)
    return u


# ------------------------------------------------------------------------------------------------ numerics
def dependence(fam, box, rng, draws=DRAWS):
    """{input name: True/False/None}: does the family's function change when only that input's values are
    permuted (for some admissible coefficient draw)?  None when no finite draw was found."""
    X = box.grid(N_DIAG, "diag")
    syms = box.symbols()
    out = {s.name: False for s in syms}
    finite_draws = 0
    for _ in range(draws * 6):
        if finite_draws >= draws:
            break
        a = draw_params(fam, rng)
        base = fam.values(a, [X[syms.index(s)] for s in fam.inputs])
        if not np.all(np.isfinite(base)):
            continue
        finite_draws += 1
        scale = max(float(np.max(np.abs(base))), 1e-300)
        for s in syms:
            if s not in fam.inputs or out[s.name]:
                continue
            Xp = list(X)
            Xp[syms.index(s)] = X[syms.index(s)][rng.permutation(len(X[0]))]
            pert = fam.values(a, [Xp[syms.index(t)] for t in fam.inputs])
            if np.all(np.isfinite(pert)) and float(np.max(np.abs(pert - base))) / scale > DEP_REL:
                out[s.name] = True
    if finite_draws == 0:
        return None
    return out


def rank(fam, box, rng, draws=DRAWS):
    """(generic rank, ambiguous flag, smallest kept/largest dropped singular-value ratios) of the Jacobian of the
    family's function w.r.t. its coefficients on the diagnostic inputs; None when no finite draw was found."""
    if not fam.params:
        return 0, False, []
    X = box.grid(N_DIAG, "diag")
    syms = box.symbols()
    Xi = [X[syms.index(s)] for s in fam.inputs]
    best, ambiguous, ratios, finite_draws = 0, False, [], 0
    for _ in range(draws * 6):
        if finite_draws >= draws:
            break
        a = draw_params(fam, rng)
        J = fam.jacobian(a, Xi)
        if not np.all(np.isfinite(J)):
            continue
        finite_draws += 1
        norms = np.linalg.norm(J, axis=0)
        keep = norms > 0
        if not np.any(keep):
            continue
        s = np.linalg.svd(J[:, keep] / norms[keep], compute_uv=False)
        r = s / s[0]
        k = int(np.sum(r > RANK_REL))
        if np.any((r >= RANK_AMBIG) & (r <= RANK_REL)):
            ambiguous = True
        ratios.append([float(x) for x in r])
        best = max(best, k)
    if finite_draws == 0:
        return None
    return best, ambiguous, ratios


def refit(cand, target_values, box, rng, starts=STARTS):
    """Fit the candidate's coefficients to the target values on the diagnostic inputs (least squares, relative
    to max|target|), report the largest relative deviation on the separate check inputs.
    target_values(Xd, Xc) -> (yd, yc).  Returns (dev, fitted coefficient vector or None)."""
    syms = box.symbols()
    Xd_all, Xc_all = box.grid(N_DIAG, "diag"), box.grid(N_CHECK, "check")
    yd, yc = target_values(Xd_all, Xc_all)
    if not (np.all(np.isfinite(yd)) and np.all(np.isfinite(yc))):
        return None, None
    Xd = [Xd_all[syms.index(s)] for s in cand.inputs]
    Xc = [Xc_all[syms.index(s)] for s in cand.inputs]
    scale_d = max(float(np.max(np.abs(yd))), 1e-300)
    scale_c = max(float(np.max(np.abs(yc))), 1e-300)
    if not cand.params:
        v = cand.values([], Xc)
        return (float(np.max(np.abs(v - yc))) / scale_c if np.all(np.isfinite(v)) else math.inf), []

    def resid(u):
        a = _from_u(cand, u)
        r = (cand.values(a, Xd) - yd) / scale_d
        return np.where(np.isfinite(r), r, 1e3)

    def jac(u):
        a = _from_u(cand, u)
        J = cand.jacobian(a, Xd) * _da_du(cand, a)[None, :] / scale_d
        return np.where(np.isfinite(J), J, 0.0)

    best_dev, best_a = math.inf, None
    k = len(cand.params)
    u0s = [rng.normal(0.0, 1.5, k) for _ in range(starts)]   # drawn up front: the draws do not depend on early exits
    for u0 in u0s:
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = least_squares(resid, u0, jac=jac, method="trf", max_nfev=MAX_NFEV_PER_PARAM * (k + 1))
        except Exception:  # noqa: BLE001 - a failed start is a failed start
            continue
        a = _from_u(cand, res.x)
        v = cand.values(a, Xc)
        if not np.all(np.isfinite(v)):
            continue
        dev = float(np.max(np.abs(v - yc))) / scale_c
        if dev < best_dev:
            best_dev, best_a = dev, a
        if best_dev <= EARLY_EXIT:
            break
    return best_dev, best_a


def member_values(target, theta, box):
    """target_values closure for refit: the target family at one parameter vector, on its own inputs of the box."""
    syms = box.symbols()
    idx = [syms.index(s) for s in target.inputs]

    def tv(Xd_all, Xc_all, _t=target, _th=theta):
        return _t.values(_th, [Xd_all[i] for i in idx]), _t.values(_th, [Xc_all[i] for i in idx])
    return tv


# ------------------------------------------------------------------------------------------------ normal form
def _has(e, inputs):
    return bool(e.free_symbols & set(inputs))


def _rewrite_input_exponents(e, inputs):
    """b**q with an input-dependent exponent and an input-free base -> exp(q*log(b))."""
    def rule(x):
        return x.is_Pow and _has(x.exp, inputs) and not _has(x.base, inputs) and x.base != sp.E
    return e.replace(rule, lambda x: sp.exp(x.exp * sp.log(x.base)))


def _leading(add, inputs):
    """(coefficient, rest) of the leading term of an input-dependent sum: the input-dependent term whose
    structural key is largest (coefficients are not part of the key)."""
    groups, order = {}, []
    for t in sp.Add.make_args(add):
        c, r = t.as_independent(*inputs, as_Add=False)
        if r == 1:
            continue
        if r not in groups:
            groups[r] = sp.Integer(0)
            order.append(r)
        groups[r] = groups[r] + c
    best = None
    for r in order:
        c = _tidy(groups[r]) if groups[r].is_Add else groups[r]
        if c == 0:
            continue
        k = shape_key(build_core(r, inputs)) + "|" + sp.srepr(r)
        if best is None or k > best[0]:
            best = (k, c, r)
    return (best[1], best[2]) if best else (None, None)


def _normalize_inner(e, inputs):
    """log(A) -> log(c) + log(A/c) and A**q -> c**q * (A/c)**q for an input-dependent sum A whose leading
    coefficient c is positive (so both rewrites are identities on the domain)."""
    if e.is_Atom:
        return e
    args = [_normalize_inner(a, inputs) for a in e.args]
    e = e.func(*args)
    if isinstance(e, sp.log) and e.args[0].is_Add and _has(e.args[0], inputs):
        c, _ = _leading(e.args[0], inputs)
        if c is not None and c != 1 and c.is_positive:
            return sp.log(c) + sp.log(sp.expand(e.args[0] / c))
    if e.is_Pow and e.base.is_Add and _has(e.base, inputs) and not _has(e.exp, inputs):
        c, _ = _leading(e.base, inputs)
        if c is not None and c != 1 and (c.is_positive or (e.exp.is_integer is True)):
            return c ** e.exp * sp.expand(e.base / c) ** e.exp
    return e


#: exact values of the constant symbols freeze_numbers() introduces (name -> sympy number); a symbol k_<hash> stands
#: for one irrational number (e.g. 310**(3/8) after the nuisance input is fixed) so that sympy never does radical
#: arithmetic on it, and every numeric check puts the exact value back (constants())
CONSTANTS = {}


def freeze_numbers(e):
    """Every maximal symbol-free subexpression that is not a rational number becomes a constant symbol k_<sha>
    (its exact value kept in CONSTANTS)."""
    if e.is_Atom:
        if e.is_Rational and not e.is_Integer and (abs(e.p) > BIG_INT or e.q > BIG_INT):
            return _const_symbol(e, float(e))
        return e
    if not e.free_symbols:
        if e.is_Rational:
            return e
        c, rest = e.as_coeff_Mul()
        if c.is_Rational and c != 1 and rest != 1 and not rest.is_Rational:
            # rule 1.3: 3100*exp(7) -> 3100*k (a rational factor stays outside, so cancel() still sees k*3100 and
            # k*31 as related)
            return freeze_numbers(c) * freeze_numbers(rest)
        try:
            v = complex(sp.N(e, 30))
        except Exception:  # noqa: BLE001
            return e
        if not math.isfinite(v.real) or abs(v.imag) > 0:
            return e
        return _const_symbol(e, v.real)
    return e.func(*[freeze_numbers(a) for a in e.args])


def _const_symbol(e, value):
    """The constant symbol standing for the exact number e (positive or negative by its value)."""
    if value == 0:
        return sp.Integer(0)
    name = "k_" + hashlib.sha256(sp.srepr(e).encode()).hexdigest()[:10]
    sym = sp.Symbol(name, positive=True) if value > 0 else sp.Symbol(name, negative=True)
    CONSTANTS[sym] = e
    return sym


def constants(e):
    """Put the exact values of constant symbols back."""
    if not hasattr(e, "free_symbols"):
        return e
    hit = {k: v for k, v in CONSTANTS.items() if k in e.free_symbols}
    return e.xreplace(hit) if hit else e


def _regroup(e, inputs):
    """A sum with one term per distinct input-dependent part: c1*m + c2*m -> (c1 + c2)*m, the coefficient tidied."""
    groups, order = {}, []
    for t in sp.Add.make_args(e):
        c, r = t.as_independent(*inputs, as_Add=False)
        if r not in groups:
            groups[r] = sp.Integer(0)
            order.append(r)
        groups[r] = groups[r] + c
    terms = []
    for r in order:
        c = _tidy(groups[r]) if groups[r].is_Add else groups[r]
        if c != 0:
            terms.append(c * r)
    return sp.Add(*terms)


def _exp_log_to_pow(e, inputs):
    """exp(k*log(u)) -> u**k for an input-dependent u (sympy leaves the exp form once it is a separate factor)."""
    def rule(x):
        if not isinstance(x, sp.exp):
            return False
        c, r = x.args[0].as_independent(*inputs, as_Add=False)
        return isinstance(r, sp.log) and _has(r, inputs)
    def conv(x):
        c, r = x.args[0].as_independent(*inputs, as_Add=False)
        return r.args[0] ** c
    return e.replace(rule, conv)


def _merge_input_powers(e, inputs):
    """x**a * x**b -> x**(a + b) for an input x in every product (expand()'s power_exp hint leaves the two factors
    apart, so X1**(n + c k) and X1**alpha would otherwise have different trees)."""
    if e.is_Atom:
        return e
    args = [_merge_input_powers(a, inputs) for a in e.args]
    if not e.is_Mul:
        return e.func(*args)
    groups, order, rest = {}, [], []
    for f in args:
        b, q = f.as_base_exp()
        if b in inputs:
            if b not in groups:
                groups[b] = []
                order.append(b)
            groups[b].append(q)
        else:
            rest.append(f)
    return sp.Mul(*(rest + [b ** sp.Add(*groups[b]) for b in order]))


def _split_sums(x, inputs):
    """(product of the factors that are not input-dependent sums, [the input-dependent sum factors and their powers])."""
    mono, sums = [], []
    for f in sp.Mul.make_args(x):
        base, ex = f.as_base_exp()
        if base.is_Add and _has(base, inputs) and (ex.is_Integer and ex > 0):
            sums.append(f)
        else:
            mono.append(f)
    return sp.Mul(*mono), sums


def _trig_sum(x):
    """sin/cos of a sum by the addition formulas only (no multiple-angle expansion: expand_trig would turn
    tanh(100000 x) or sin(1000 x) into enormous polynomials)."""
    terms = sp.Add.make_args(x.args[0])
    if len(terms) < 2:
        return x
    a, b = terms[0], sp.Add(*terms[1:])
    if isinstance(x, sp.sin):
        return sp.sin(a) * _trig_sum(sp.cos(b)) + sp.cos(a) * _trig_sum(sp.sin(b))
    return sp.cos(a) * _trig_sum(sp.cos(b)) - sp.sin(a) * _trig_sum(sp.sin(b))


def normal_form(e, inputs):
    """A deterministic chain of identities on the domain: input exponents as exp-log, cos/sin of sums
    expanded, exp of sums split, logs of products split (inputs and positive coefficients), one fraction with
    expanded numerator and denominator, leading coefficients pulled out of sums under log and powers."""
    if isinstance(e, sp.Tuple):
        return sp.Tuple(*[normal_form(x, inputs) for x in e])
    e = freeze_numbers(e)
    e = _rewrite_input_exponents(e, inputs)
    e = e.replace(lambda x: isinstance(x, (sp.sin, sp.cos)), lambda x: x.func(sp.expand(x.args[0])))
    e = e.replace(lambda x: isinstance(x, (sp.sin, sp.cos)) and x.args[0].is_Add, _trig_sum)
    e = sp.expand(e, power_exp=True, log=True, mul=False, multinomial=False, power_base=True, force=True)
    e = sp.expand_log(e, force=True)
    e = _exp_log_to_pow(e, inputs)
    e = sp.together(e)
    n, d = sp.fraction(e)
    # common factors out of numerator and denominator (an exp(-E/T) or x^k that multiplies every term is one factor,
    # wherever together() put it); the remaining input-dependent sums are expanded and regrouped
    n_mono, n_sums = _split_sums(sp.factor_terms(sp.expand(n, force=True)), inputs)
    d_mono, d_sums = _split_sums(sp.factor_terms(sp.expand(d, force=True)), inputs)
    mono = n_mono / d_mono
    num = _regroup(sp.expand(sp.Mul(*n_sums), force=True), inputs) if n_sums else sp.Integer(1)
    den = _regroup(sp.expand(sp.Mul(*d_sums), force=True), inputs) if d_sums else sp.Integer(1)
    if den.is_Add and _has(den, inputs):
        c, _ = _leading(den, inputs)
        if c is not None and c != 1 and c != 0:
            mono = mono / c
            den = _regroup(sp.expand(den / c), inputs)
    # the monomial multiplies each numerator term as it is (expand() would also expand its denominator)
    num = _regroup(sp.Add(*[mono * t for t in sp.Add.make_args(num)]), inputs)
    e = num / den if den != 1 else num
    e = _normalize_inner(e, inputs)
    e = _merge_input_powers(e, inputs)
    return freeze_numbers(e)


# ------------------------------------------------------------------------------------------------ canonical tree
def build(e, inputs):
    if isinstance(e, sp.Tuple):
        return ("tuple", tuple(build(x, inputs) for x in e))
    return build_sum(e, inputs)


def build_sum(e, inputs):
    groups = {}
    order = []
    for t in sp.Add.make_args(e):
        c, r = t.as_independent(*inputs, as_Add=False)
        if r not in groups:
            groups[r] = sp.Integer(0)
            order.append(r)
        groups[r] = groups[r] + c
    children = []
    for r in order:
        c = groups[r]
        if c.is_Add:
            c = _tidy(c)
        if c == 0:
            continue
        core = ("one",) if r == 1 else build_core(r, inputs)
        children.append(("term", ("slot", c), core))
    return ("add", tuple(sorted(children, key=shape_key)))


def build_core(r, inputs):
    if r.is_Symbol:
        return ("in", r.name) if r in inputs else ("slot", r)
    if not _has(r, inputs):
        return ("slot", r)
    if r.is_Add:
        return build_sum(r, inputs)
    if r.is_Mul:
        c, rest = r.as_independent(*inputs, as_Add=False)
        factors = tuple(sorted((build_core(f, inputs) for f in sp.Mul.make_args(rest)), key=shape_key))
        node = ("mul", factors)
        return node if c == 1 else ("mul", (("slot", c),) + factors)
    if r.is_Pow:
        b, q = r.base, r.exp
        if b == sp.E:
            return ("fn", "exp", (build_sum(sp.expand(q), inputs),))
        base = build_sum(b, inputs) if b.is_Add else build_core(b, inputs)
        exp = ("slot", q) if not _has(q, inputs) else build_sum(q, inputs)
        return ("pow", base, exp)
    if isinstance(r, sp.exp):
        return ("fn", "exp", (build_sum(sp.expand(r.args[0]), inputs),))
    if isinstance(r, sp.Function) or hasattr(r, "func"):
        name = type(r).__name__
        return ("fn", name, tuple(build_sum(sp.expand(a), inputs) for a in r.args))
    return ("opaque", sp.srepr(r))


def shape_key(node):
    """The node's structure with every slot shown as S (coefficients do not enter the key)."""
    kind = node[0]
    if kind == "slot":
        return "S"
    if kind in ("in",):
        return f"in:{node[1]}"
    if kind == "one":
        return "1"
    if kind == "opaque":
        return "opaque:" + node[1]
    if kind == "term":
        return "T(" + shape_key(node[1]) + "*" + shape_key(node[2]) + ")"
    if kind in ("add", "mul", "tuple"):
        return kind + "[" + ",".join(shape_key(c) for c in node[1]) + "]"
    if kind == "pow":
        return "pow(" + shape_key(node[1]) + "," + shape_key(node[2]) + ")"
    if kind == "fn":
        return node[1] + "(" + ",".join(shape_key(c) for c in node[2]) + ")"
    raise ValueError(kind)


def tree_text(node):
    """Readable canonical form with slot values shown."""
    kind = node[0]
    if kind == "slot":
        return "{" + str(constants(node[1])) + "}"
    if kind == "in":
        return node[1]
    if kind == "one":
        return "1"
    if kind == "opaque":
        return "<" + node[1][:60] + ">"
    if kind == "term":
        return tree_text(node[1]) + ("" if node[2] == ("one",) else "*" + tree_text(node[2]))
    if kind == "add":
        return "(" + " + ".join(tree_text(c) for c in node[1]) + ")"
    if kind == "mul":
        return "*".join(tree_text(c) for c in node[1])
    if kind == "tuple":
        return "<" + "; ".join(tree_text(c) for c in node[1]) + ">"
    if kind == "pow":
        return "(" + tree_text(node[1]) + ")^" + tree_text(node[2])
    if kind == "fn":
        return node[1] + "(" + ", ".join(tree_text(c) for c in node[2]) + ")"
    return str(node)


def _numeric(x):
    return not (x.free_symbols - set(CONSTANTS))


def unify(a, b, eqs, out, budget):
    """All ways the two canonical trees match; each way appends its slot equations to `out` (bounded)."""
    if len(out) >= budget[0]:
        return
    ka, kb = a[0], b[0]
    if ka != kb:
        return
    if ka == "slot":
        x, y = a[1], b[1]
        if _numeric(x) and _numeric(y):
            if x.is_Rational and y.is_Rational:
                same = x == y
            else:
                try:
                    vx, vy = complex(sp.N(constants(x), 30)), complex(sp.N(constants(y), 30))
                    same = abs(vx - vy) <= 1e-12 * max(1.0, abs(vx), abs(vy))
                except Exception:  # noqa: BLE001
                    same = False
            if same:
                out.append(eqs)
            return
        out.append(eqs + [(x, y)])
        return
    if ka in ("in", "one", "opaque"):
        if a == b:
            out.append(eqs)
        return
    if ka == "term":
        partial = []
        unify(a[1], b[1], eqs, partial, budget)
        for e in partial:
            unify(a[2], b[2], e, out, budget)
        return
    if ka == "pow":
        partial = []
        unify(a[1], b[1], eqs, partial, budget)
        for e in partial:
            unify(a[2], b[2], e, out, budget)
        return
    if ka == "fn":
        if a[1] != b[1] or len(a[2]) != len(b[2]):
            return
        states = [eqs]
        for ca, cb in zip(a[2], b[2]):
            nxt = []
            for e in states:
                unify(ca, cb, e, nxt, budget)
            states = nxt
            if not states:
                return
        out.extend(states)
        return
    if ka == "tuple":
        if len(a[1]) != len(b[1]):
            return
        states = [eqs]
        for ca, cb in zip(a[1], b[1]):
            nxt = []
            for e in states:
                unify(ca, cb, e, nxt, budget)
            states = nxt
            if not states:
                return
        out.extend(states)
        return
    if ka in ("add", "mul"):
        ca, cb = list(a[1]), list(b[1])
        if len(ca) != len(cb):
            return
        if sorted(shape_key(c) for c in ca) != sorted(shape_key(c) for c in cb):
            return
        _unify_multiset(ca, cb, eqs, out, budget)
        return


def _unify_multiset(ca, cb, eqs, out, budget):
    if len(out) >= budget[0]:
        return
    if not ca:
        out.append(eqs)
        return
    first, rest = ca[0], ca[1:]
    key = shape_key(first)
    for j, other in enumerate(cb):
        if shape_key(other) != key:
            continue
        partial = []
        unify(first, other, eqs, partial, budget)
        for e in partial:
            _unify_multiset(rest, cb[:j] + cb[j + 1:], e, out, budget)
            if len(out) >= budget[0]:
                return


# ------------------------------------------------------------------------------------------------ slot equations
def _plain(symbols):
    """Dummies for solving that keep positivity (so powers and logs of positive coefficients simplify) and are real
    otherwise; finer coefficient domains are checked numerically afterwards."""
    return {s: sp.Symbol("u_" + s.name, positive=True) if s.is_positive else sp.Symbol("u_" + s.name, real=True)
            for s in symbols}


def _tidy(x):
    """An input-free coefficient brought to one cancelled fraction (exact identities such as a/(1+a) + 1/(1+a) = 1);
    numbers that would make polynomial arithmetic slow are constant symbols first, and a large expression is only
    put over one denominator."""
    try:
        x = freeze_numbers(x)
        t = sp.together(x)
        if sp.count_ops(t) > MAX_TIDY_OPS:
            return t
        return sp.cancel(t)
    except Exception:  # noqa: BLE001
        return x


def _is_zero(x):
    """Exact zero test: cancel first; a cheap 30-digit probe rejects a clearly nonzero expression; a two-term
    log comparison (_zero_by_logs) or simplify (bounded size) is the proof for the rest.  False means 'not shown
    to be zero'."""
    x0 = x
    x = _tidy(x)
    if x == 0:
        return True
    try:
        rng = np.random.default_rng(seed_of("zero", sp.srepr(x)))
        free = sorted(x.free_symbols, key=lambda z: z.name)
        for _ in range(3):
            vals = {z: (constants(CONSTANTS[z]) if z in CONSTANTS else
                        sp.Float(float(np.exp(rng.uniform(-0.7, 0.7))) if z.is_positive else
                                 float(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.3)), 30)) for z in free}
            terms = sp.Add.make_args(sp.expand(sp.numer(x))) if sp.count_ops(x) <= MAX_TIDY_OPS else (x,)
            mags = [abs(complex(sp.N(t.xreplace(vals), 40))) for t in terms]
            v = complex(sp.N(sp.numer(x).xreplace(vals), 40))
            scale = max([1.0] + [m for m in mags if math.isfinite(m)])
            if math.isfinite(v.real) and math.isfinite(v.imag) and abs(v) > 1e-20 * scale:
                return False
    except Exception:  # noqa: BLE001
        pass
    for form in (sp.numer(x), x0):      # the tidied numerator, then the expression as given
        try:
            if _zero_by_logs(form):
                return True
        except Exception:  # noqa: BLE001
            pass
    if sp.count_ops(x) <= MAX_SIMPLIFY_OPS:
        try:
            return sp.simplify(x) == 0
        except Exception:  # noqa: BLE001
            return False
    return False


def _log_split(z):
    """log z with products, powers and exp split by sympy's own (assumption-checked) rules, every sum under a log
    written with its common factor taken out first (log(310 c + 310) = log 310 + log(c + 1) for c > 0)."""
    z = sp.expand_log(sp.log(z))
    z = z.replace(lambda w: isinstance(w, sp.log) and w.args[0].is_Add,
                  lambda w: sp.expand_log(sp.log(sp.factor_terms(w.args[0]))))
    return sp.expand(z)


def _zero_by_logs(x):
    """A two-term sum A - B where sympy knows A and B to be positive (or both negative) is zero when log A - log B
    expands to zero (rule 1.3; sound: no force=True, every split is checked by sympy's assumptions)."""
    terms = sp.Add.make_args(x)
    if len(terms) != 2 or sp.count_ops(x) > MAX_TIDY_OPS:
        return False
    a, b = constants(terms[0]), constants(-terms[1])
    if a.is_negative and b.is_negative:
        a, b = -a, -b
    if not (a.is_positive and b.is_positive):
        return False
    d = _log_split(a) - _log_split(b)
    if d == 0:
        return True
    return sp.count_ops(d) <= MAX_SIMPLIFY_OPS and sp.simplify(d) == 0


def _pick_root(roots, symbols):
    """The first root that is real when every remaining symbol is 1.3 (a fixed probe), else the first root."""
    probe = {z: sp.Float("1.3") for z in symbols}
    for r in roots:
        try:
            v = complex(sp.N(r.xreplace({z: probe[z] for z in r.free_symbols if z in probe})))
        except Exception:  # noqa: BLE001
            continue
        if math.isfinite(v.real) and abs(v.imag) <= 1e-12 * max(1.0, abs(v.real)):
            return r
    return roots[0]


def _degree_in(e, u):
    """Polynomial degree of e in u read off the expression tree without expanding it (math.inf when u sits inside a
    function or a non-integer power)."""
    if not e.has(u):
        return 0
    if e == u:
        return 1
    if e.is_Add:
        return max(_degree_in(a, u) for a in e.args)
    if e.is_Mul:
        return sum(_degree_in(a, u) for a in e.args)
    if e.is_Pow:
        if e.exp.has(u):
            return math.inf
        k = e.exp
        if k.is_Integer and k >= 0:
            return int(k) * _degree_in(e.base, u)
        return math.inf
    return math.inf


def _linear_in(e, u):
    """True when e is of degree 1 in u (solving for u is a division)."""
    return _degree_in(e, u) == 1


_LAMBERT_MASK = sp.Symbol("lambert_mask_")


def _lambert_type(e, u):
    """True when u sits both inside a function application (or an exponent) and outside of it, e.g. u*exp(u) - c or
    u + log(u) - c: such equations need the Lambert W function, and sympy's Lambert path can run for minutes
    (rule 1.2: they are not solved, which is no proof).  Equations with u only inside functions (exp(u) - c) or only
    outside (u**2 - c, sqrt(u) - c) are not affected."""
    inner = {f for f in e.atoms(sp.Function) if f.has(u)}
    inner |= {p for p in e.atoms(sp.Pow) if p.exp.has(u)}
    if not inner:
        return False
    return e.xreplace({f: _LAMBERT_MASK for f in inner}).has(u)


def _solve_as_logs(e, u):
    """A two-term equation A - B = 0 of products and powers (e.g. p**n - T**n) solved as log A - log B = 0."""
    terms = sp.Add.make_args(e)
    if len(terms) != 2:
        return []
    t1, t2 = terms
    for a, b in ((t1, -t2), (-t1, t2)):
        if any(f.is_Add for f in sp.Mul.make_args(a) + sp.Mul.make_args(b)):
            continue
        try:
            eq = sp.expand_log(sp.log(a) - sp.log(b), force=True)
            if _lambert_type(eq, u):
                continue
            roots = sp.solve(eq, u)
        except Exception:  # noqa: BLE001
            continue
        if roots:
            return roots
    return []


def solve_direction(eqs, unknowns, givens):
    """Solve the slot equations for `unknowns` as functions of `givens`, one equation and one unknown at a time
    (the equation with the fewest unknowns first, then the unknown with the fewest occurrences): every call to
    sympy's solver has a single unknown, which keeps it fast and its result order fixed.  Returns a substitution
    dict (unknown -> expression in givens and possibly leftover unknowns) or None."""
    if not eqs:
        return {}
    plain = _plain(list(unknowns) + list(givens))
    back = {v: k for k, v in plain.items()}
    unk = set(plain[u] for u in unknowns)
    exprs = []
    for x, y in eqs:
        d = _tidy((x - y).xreplace(plain))
        if d == 0:
            continue
        num = sp.fraction(d)[0]
        if not (num.free_symbols & unk):
            if _is_zero(num):
                continue
            return None            # a constraint among the givens alone: not an identity
        exprs.append(num)
    if not exprs:
        return {}
    # non-integer rational literals become symbols while solving (exp(23026/10000*u) would otherwise be turned into a
    # polynomial of degree 11513 in exp(u/5000)); they are put back in the solution
    rats = sorted({r for e in exprs for r in e.atoms(sp.Rational) if r.q != 1}, key=lambda r: (r.q, r.p))
    rmap = {r: sp.Symbol(f"rat_{i}", positive=bool(r > 0), negative=bool(r < 0)) for i, r in enumerate(rats)}
    rback = {v: k for k, v in rmap.items()}
    work = [e.xreplace(rmap) for e in exprs]
    sol, remaining, left = {}, list(work), set(unk)
    while remaining:
        if any(sp.count_ops(x) > MAX_SOLVE_OPS for x in remaining):
            return None            # substitutions made an equation too large for a bounded solve: no proof
        choices = []
        for i, e in enumerate(remaining):
            us = e.free_symbols & left
            if not us:
                continue
            for u in us:
                kind = 0 if _linear_in(e, u) else (2 if _lambert_type(e, u) else 1)
                choices.append((len(us), kind, e.count(u), sp.count_ops(e), u.name, i, u))
        if not choices:
            break
        choices.sort(key=lambda z: z[:5])
        _, _, _, _, _, i, u = choices[0]
        e = remaining.pop(i)
        if sp.count_ops(e) > MAX_SOLVE_OPS:
            return None            # too large for a bounded single-unknown solve: no proof
        deg = _degree_in(e, u)
        if deg != math.inf and deg > MAX_SOLVE_DEGREE:
            return None            # a cubic or higher in the unknown: radical solutions are not attempted (no proof)
        if _lambert_type(e, u):
            return None            # u inside and outside a function: a Lambert W solve is not attempted (no proof)
        try:
            roots = sp.solve(e, u)
        except Exception:  # noqa: BLE001 - an unsolved equation is no proof
            roots = []
        if not roots:
            roots = _solve_as_logs(e, u)
        if not roots:
            if _is_zero(e):
                continue           # an identity once the earlier solutions are put in: nothing left to solve here
            return None
        r = _pick_root(roots, set(plain.values()))
        sol = {k: v.xreplace({u: r}) for k, v in sol.items()}
        sol[u] = r
        left.discard(u)
        remaining = [x.xreplace({u: r}) for x in remaining]
        for x in remaining:
            if not (x.free_symbols & left) and not _is_zero(x):
                return None
    sol = {k: v.xreplace(rback) for k, v in sol.items()}
    if not all(_is_zero(e.xreplace(sol)) for e in exprs):
        return None
    return {back[k]: v.xreplace(back) for k, v in sol.items() if k in back}


def _in_domain(v, d):
    if not (math.isfinite(v.real) and math.isfinite(v.imag)) or abs(v.imag) > 1e-12 * max(1.0, abs(v.real)):
        return False
    return not ((d == "positive" and not v.real > 0) or (d == "unit" and not 0 < v.real < 1))


def spot_check(lhs, rhs, params_l, params_r, mapping, box, key, domains=None):
    """30-digit check of lhs(params_l) == rhs(params_r with `mapping` applied) at SPOT_POINTS random inputs and
    coefficient draws.  Leftover symbols get random values.  Rule 1.3: draws whose coefficients - drawn or mapped -
    all lie in their declared domains (`domains`; the canonical-form identities hold on the domain) are used first;
    only when no such draw is found among SPOT_POINTS * 20 are unrestricted draws used; at least one finite
    comparison must be made.  True / False."""
    for restricted in (True, False):
        verdict = _spot_pass(lhs, rhs, mapping, box, key, domains if restricted else None, restricted)
        if verdict is not None:
            return verdict
    return False


def _spot_pass(lhs, rhs, mapping, box, key, domains, restricted):
    """One spot-check pass: True / False, or None when no usable comparison was made."""
    rng = np.random.default_rng(seed_of("spot", key, "restricted" if restricted else "free"))
    domains = domains or {}
    mapped_exprs = {k: constants(sp.sympify(v)) for k, v in mapping.items()} if mapping else {}
    mapped = rhs.xreplace(mapped_exprs) if mapping else rhs
    lhs = constants(lhs)
    lhs_parts = list(lhs) if isinstance(lhs, sp.Tuple) else [lhs]
    rhs_parts = list(mapped) if isinstance(mapped, sp.Tuple) else [mapped]
    if len(lhs_parts) != len(rhs_parts):
        return False
    mpmath.mp.dps = SPOT_DIGITS
    syms = box.symbols()
    free = set()
    for part in lhs_parts + rhs_parts:
        free |= part.free_symbols
    for ex in mapped_exprs.values():
        free |= ex.free_symbols
    compared = 0
    for _ in range(SPOT_POINTS * 20):
        if compared >= SPOT_POINTS:
            break
        vals = {}
        for s, (lo, hi, kind) in box.spec.items():
            if kind == "binary":
                vals[s] = int(rng.integers(0, 2))
            elif kind == "log":
                vals[s] = mpmath.mpf(float(np.exp(rng.uniform(math.log(lo), math.log(hi)))))
            else:
                vals[s] = mpmath.mpf(float(rng.uniform(lo, hi)))
        for s in sorted(free - set(syms), key=lambda z: z.name):
            if domains.get(s) == "unit":
                vals[s] = mpmath.mpf(float(rng.uniform(0.15, 0.85)))
            elif s.is_positive:
                vals[s] = mpmath.mpf(float(np.exp(rng.uniform(-0.7, 0.7))))
            else:
                vals[s] = mpmath.mpf(float(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.3)))
        fv = {k: sp.Float(str(v), SPOT_DIGITS) for k, v in vals.items()}
        if restricted:
            try:
                if not all(_in_domain(complex(sp.N(ex.xreplace(fv), 20)), domains.get(p, "real"))
                           for p, ex in mapped_exprs.items()):
                    continue
            except Exception:  # noqa: BLE001
                continue
        point_ok = False
        for lp, rp in zip(lhs_parts, rhs_parts):
            try:
                lv = complex(sp.N(lp.xreplace(fv), SPOT_DIGITS))
                rv = complex(sp.N(rp.xreplace(fv), SPOT_DIGITS))
            except Exception:  # noqa: BLE001
                return False
            if not (math.isfinite(lv.real) and math.isfinite(rv.real)):
                continue
            if abs(lv - rv) > 1e-18 * max(1.0, abs(lv), abs(rv)):
                return False
            point_ok = True
        compared += point_ok
    return True if compared > 0 else None


def feasible(fam, mapping, values, rng, tries=(1.0, 0.5, 2.0, 3.0)):
    """Can the family's coefficients take the values the mapping gives at `values` (a dict of the other side's
    symbols), within their domains?  Leftover symbols of the mapping are tried at a few values."""
    if mapping is None:
        return False
    for t in tries:
        ok = True
        subs = dict(values)
        for p in fam.params:
            if p not in mapping:
                subs.setdefault(p, sp.Float(t))
        for p in fam.params:
            expr = mapping.get(p, subs.get(p))
            try:
                v = complex(sp.N(constants(sp.sympify(expr)).xreplace(subs)))
            except Exception:  # noqa: BLE001
                ok = False
                break
            if abs(v.imag) > 1e-9 * max(1.0, abs(v.real)) or not math.isfinite(v.real):
                ok = False
                break
            x = v.real
            d = fam.domains[p]
            if (d == "positive" and not x > 0) or (d == "unit" and not 0 < x < 1):
                ok = False
                break
        if ok:
            return True
    return False


def match_families(cand, targ, box, key):
    """Symbolic equivalence attempt.  Returns dict(kind=..., c_to_t=..., t_to_c=..., tree=...) where kind is
    'equivalent' (both directions solved and spot-checked), 'cand_in_target' (only target parameters as
    functions of the candidate's), 'target_in_cand', or None (trees differ or nothing solved)."""
    if sp.count_ops(cand.sym) > MAX_FORM_OPS:
        return {"kind": None, "tree": None, "target_tree": None, "c_to_t": None, "t_to_c": None,
                "skipped": f"candidate larger than {MAX_FORM_OPS} operations"}
    nc = normal_form(cand.sym, cand.inputs)
    nt = normal_form(targ.sym, targ.inputs)
    if sp.count_ops(nc) > 4 * MAX_FORM_OPS:
        return {"kind": None, "tree": None, "target_tree": None, "c_to_t": None, "t_to_c": None,
                "skipped": f"normal form larger than {4 * MAX_FORM_OPS} operations"}
    tc, tt = build(nc, cand.inputs), build(nt, targ.inputs)
    result = {"kind": None, "tree": tree_text(tc), "target_tree": tree_text(tt), "c_to_t": None, "t_to_c": None}
    unifiers = []
    unify(tc, tt, [], unifiers, [MAX_UNIFIERS])
    if not unifiers:
        return result
    best = None
    for eqs in unifiers:
        c_to_t = solve_direction(eqs, targ.params, cand.params)
        t_to_c = solve_direction(eqs, cand.params, targ.params)
        domains = {**cand.domains, **targ.domains}
        if c_to_t is not None and not spot_check(cand.sym, targ.sym, cand.params, targ.params, c_to_t, box,
                                                 key + "|ct", domains):
            c_to_t = None
        if t_to_c is not None and not spot_check(targ.sym, cand.sym, targ.params, cand.params, t_to_c, box,
                                                 key + "|tc", domains):
            t_to_c = None
        kind = ("equivalent" if c_to_t is not None and t_to_c is not None else
                "cand_in_target" if c_to_t is not None else "target_in_cand" if t_to_c is not None else None)
        rank_of = {"equivalent": 3, "target_in_cand": 2, "cand_in_target": 1, None: 0}
        if best is None or rank_of[kind] > rank_of[best[0]]:
            best = (kind, c_to_t, t_to_c, eqs)
        if kind == "equivalent":
            break
    result.update(kind=best[0], c_to_t=best[1], t_to_c=best[2],
                  equations=[f"{constants(x)} = {constants(y)}" for x, y in best[3]])
    return result


# ------------------------------------------------------------------------------------------------ substitutions
def _limit(expr, p, to, scale_free):
    """expr as the coefficient p goes to 0 or to infinity (to = 0 | 'inf').  When the plain limit is 0 or infinite
    and the family carries a free positive overall factor (scale_free), the leading term in the limit variable is
    kept and its power absorbed by that factor (a joint limit p -> 0 or infinity with the factor compensating).
    Returns (expression, compensated) or (None, False)."""
    q = sp.Symbol("q_lim", positive=True)
    e = sp.together(expr.xreplace({p: (q if to == 0 else 1 / q)}))
    try:
        if sp.count_ops(e) <= MAX_TIDY_OPS:
            e = sp.cancel(e)
        v = e.xreplace({q: 0})      # no general limit algorithm: only what cancelling exposes (bounded work)
    except Exception:  # noqa: BLE001
        return None, False
    bad = v.has(sp.zoo, sp.nan, sp.oo, -sp.oo)
    if (bad or v == 0) and scale_free:
        n, d = sp.fraction(e)
        try:
            pn, pd = sp.Poly(n, q), sp.Poly(d, q)
        except Exception:  # noqa: BLE001
            return None, False
        lo_n = min(m[0] for m in pn.monoms())
        lo_d = min(m[0] for m in pd.monoms())
        v = sp.cancel(pn.coeff_monomial(q ** lo_n) / pd.coeff_monomial(q ** lo_d))
        if v == 0 or v.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            return None, False
        return v, True
    if bad:
        return None, False
    return v, False


def apply_substitution(fam, sub):
    """sub = (p, value) with value 0, 1, 'inf' or another coefficient symbol.  Returns the reduced family or None;
    a limit that needed the overall factor's compensation is marked on the family (attribute compensated)."""
    p, v = sub
    params = [x for x in fam.params if x != p]
    if v in (0, "inf") and not isinstance(v, sp.Symbol):
        scale_free = fam.scale is not None and fam.scale != p
        if isinstance(fam.sym, sp.Tuple):
            parts = [_limit(x, p, v, False) for x in fam.sym]
            if any(x[0] is None for x in parts):
                return None
            sym, comp = sp.Tuple(*[x[0] for x in parts]), False
        else:
            sym, comp = _limit(fam.sym, p, v, scale_free)
            if sym is None:
                return None
        if v == 0 and not comp:
            num = fam.num.xreplace({p: sp.Integer(0)})
            if num.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
                num = fam.num_from_sym(sym)
        else:
            num = fam.num_from_sym(sym)
        free = sym.free_symbols | num.free_symbols
        params = [x for x in params if x in free]
        red = Family(sym, params, {x: fam.domains[x] for x in params}, fam.inputs, num=num, label=fam.label,
                     num_from_sym=fam.num_from_sym, scale=fam.scale)
        red.compensated = comp
        return red
    value = v if isinstance(v, sp.Symbol) else sp.Integer(v)
    try:
        red = fam.with_sub({p: value})
    except Exception:  # noqa: BLE001
        return None
    if any(x.has(sp.zoo, sp.nan) for x in red.parts("sym")):
        return None
    red.compensated = False
    return red


def guided_substitutions(fam, fitted):
    """Coefficient values a numeric refit drove to a special point: (p, 0), (p, 'inf'), (p, 1), (p, q)."""
    if fitted is None:
        return []
    subs = []
    for p, x in zip(fam.params, fitted):
        d = fam.domains[p]
        if d == "positive":
            if x < 1e-4:
                subs.append((p, 0))
            elif x > 1e4:
                subs.append((p, "inf"))
            elif abs(x - 1) < 1e-3:
                subs.append((p, 1))
        elif d == "real":
            if abs(x) < 1e-4:
                subs.append((p, 0))
            elif abs(x - 1) < 1e-3:
                subs.append((p, 1))
    for (p, x), (q, y) in itertools.combinations(list(zip(fam.params, fitted)), 2):
        if abs(x - y) < 1e-3 * max(1.0, abs(x), abs(y)):
            subs.append((p, q))
    return subs


def all_single_substitutions(fam):
    subs = []
    for p in fam.params:
        subs += [(p, 0), (p, 1), (p, "inf")]
    for p, q in itertools.combinations(fam.params, 2):
        subs.append((p, q))
    return subs


def containment_search(cand, targ, box, key, fitted_list, members, max_depth=2, max_attempts=MAX_SUB_ATTEMPTS):
    """A substitution that turns the candidate into a family equivalent to the target and reaching every target
    member, or None.  Order: the refit's special values (all at once, then one, then two at a time), every single
    substitution, then pairs of single substitutions on two different coefficients; at most max_attempts
    canonical comparisons (a count, not a clock, so the result is reproducible)."""
    tried = set()
    budget = [max_attempts]

    def attempt(subs):
        sig = tuple(sorted((x[0].name, str(x[1])) for x in subs))
        if sig in tried or budget[0] <= 0:
            return None
        tried.add(sig)
        params = [x[0] for x in subs]
        if len(set(params)) != len(params):
            return None
        fam = cand
        for x in subs:
            if x[0] not in fam.params or (isinstance(x[1], sp.Symbol) and x[1] not in fam.params):
                return None
            fam = apply_substitution(fam, x)
            if fam is None:
                return None
        budget[0] -= 1
        mm = match_families(fam, targ, box, key + "|sub|" + "|".join(f"{a}={b}" for a, b in sig))
        if mm["kind"] != "equivalent":
            return None
        mm["compensated"] = bool(getattr(fam, "compensated", False))
        rng = np.random.default_rng(seed_of("subfeas", key, sig))
        if not all(feasible(fam, mm["t_to_c"], {p: sp.Float(repr(float(v[p]))) for p in targ.params}, rng)
                   for v in members):
            return None
        return {"substitution": {a: b for a, b in sig}, "mapping": mm}

    for fitted in fitted_list:
        guided = guided_substitutions(cand, fitted)
        if not guided:
            continue
        by_param = {}
        for x in guided:
            by_param.setdefault(x[0], x)
        hit = attempt(list(by_param.values()))
        if hit:
            return hit
        for x in guided:
            hit = attempt([x])
            if hit:
                return hit
        for x, y in itertools.combinations(guided, 2):
            hit = attempt([x, y])
            if hit:
                return hit
    singles = all_single_substitutions(cand)
    for x in singles:
        hit = attempt([x])
        if hit:
            return hit
    if max_depth >= 2:
        for x, y in itertools.combinations(singles, 2):
            involved = {x[0], y[0]} | {z for z in (x[1], y[1]) if isinstance(z, sp.Symbol)}
            if x[0] == y[0] or (isinstance(x[1], sp.Symbol) and x[1] == y[0]) or (isinstance(y[1], sp.Symbol) and y[1] == x[0]):
                continue
            if len(involved) < 2:
                continue
            hit = attempt([x, y])
            if hit:
                return hit
            if budget[0] <= 0:
                break
    return None


# ------------------------------------------------------------------------------------------------ comparison
def _fmt_map(mapping):
    if mapping is None:
        return None
    return {k.name if hasattr(k, "name") else str(k): str(constants(sp.sympify(v)))
            for k, v in sorted(mapping.items(), key=lambda kv: str(kv[0]))}


class _Repro:
    """Lazy refits of the candidate to the target members (member 0 is the world's own law), one seed per member."""

    def __init__(self, cand, targ, box, key, vectors):
        self.cand, self.targ, self.box, self.key, self.vectors = cand, targ, box, key, vectors
        self.devs, self.fits = {}, {}

    def member(self, i):
        if i not in self.devs:
            dev, fit = refit(self.cand, member_values(self.targ, self.vectors[i], self.box), self.box,
                             np.random.default_rng(seed_of("refit", self.key, i)))
            self.devs[i] = math.inf if dev is None else dev
            self.fits[i] = fit
        return self.devs[i]

    def sequential(self, stop_above):
        """Members in order until one exceeds stop_above: (largest deviation seen, every member evaluated)."""
        worst = 0.0
        for i in range(len(self.vectors)):
            d = self.member(i)
            worst = max(worst, d)
            if d > stop_above:
                return worst, False
        return worst, True


def _outside_dev(cand, targ, box, key, draws=5):
    """The largest relative deviation of the target family refitted to finite candidate members (draws of the
    candidate's coefficients whose function is finite on the diagnostic inputs), stopping at the first above
    WITNESS_TOL; None when no finite draw was found."""
    rng = np.random.default_rng(seed_of("outside", key))
    syms = box.symbols()
    Xi = [box.grid(N_DIAG, "diag")[syms.index(s)] for s in cand.inputs]
    worst, finite = 0.0, 0
    for k in range(draws * 4):
        if finite >= draws:
            break
        a = draw_params(cand, rng)
        if not np.all(np.isfinite(cand.values(a, Xi))):
            continue
        finite += 1
        dev, _ = refit(targ, member_values(cand, a, box), box, np.random.default_rng(seed_of("outside-fit", key, k)))
        dev = math.inf if dev is None else dev
        worst = max(worst, dev)
        if worst > WITNESS_TOL:
            break
    return worst if finite else None


def _maps_outside(cand, targ, c_to_t, box, key, draws=5):
    """Do finite candidate members map (through the solved target-from-candidate expressions) to target parameter
    values outside the target's declared domain?  Candidate draws whose function is not finite on the diagnostic
    inputs are not family members and are skipped."""
    if not c_to_t:
        return False
    rng = np.random.default_rng(seed_of("wide", key))
    syms = box.symbols()
    X = box.grid(N_DIAG, "diag")
    Xi = [X[syms.index(s)] for s in cand.inputs]
    finite = 0
    for _ in range(draws * 4):
        if finite >= draws:
            break
        a = draw_params(cand, rng)
        v = cand.values(a, Xi)
        if not np.all(np.isfinite(v)):
            continue
        finite += 1
        subs = {p: sp.Float(repr(x)) for p, x in zip(cand.params, a)}
        for p in targ.params:
            expr = c_to_t.get(p)
            if expr is None:
                continue
            e = constants(sp.sympify(expr)).xreplace(subs)
            e = e.xreplace({s: sp.Integer(1) for s in e.free_symbols})
            try:
                val = complex(sp.N(e))
            except Exception:  # noqa: BLE001
                return True
            if not math.isfinite(val.real) or abs(val.imag) > 1e-9 * max(1.0, abs(val.real)):
                return True
            d = targ.domains[p]
            if (d == "positive" and not val.real > 0) or (d == "unit" and not 0 < val.real < 1):
                return True
    return False


def compare(cand, targ, box, *, truth, members, key, want_membership=True):
    """The generic decision for one candidate family against one representable target family.
    truth: dict target-symbol -> value (the world's own parameters); members: list of such dicts (diagnostic
    target members, the truth first).  Returns the state, reason and the separate fields."""
    out = {"containment": False, "containment_basis": None, "containment_witness": None,
           "numeric_containment_dev": None, "numeric_members_evaluated": None, "world_law_member": None,
           "world_member_basis": None, "world_member_dev": None, "truth_reachable": None, "domain_note": None,
           "parameter_mapping": None, "canonical_tree": None, "target_tree": None}
    rng = np.random.default_rng(seed_of("draws", key))
    dep_c, dep_t = dependence(cand, box, rng), dependence(targ, box, rng)
    rk_c, rk_t = rank(cand, box, rng), rank(targ, box, rng)
    out["dependence"] = {"candidate": dep_c, "target": dep_t}
    out["dims"] = {"candidate": None if rk_c is None else rk_c[0], "target": None if rk_t is None else rk_t[0],
                   "ambiguous": bool((rk_c and rk_c[1]) or (rk_t and rk_t[1]))}
    if dep_c is None or rk_c is None:
        return _done(out, "unresolved", "U_EVAL", "the candidate is not finite on the diagnostic inputs for any "
                     "admissible coefficient draw")
    if dep_t is None or rk_t is None:
        return _done(out, "unresolved", "U_EVAL_TARGET", "the target is not finite on the diagnostic inputs")
    set_c = {k for k, v in dep_c.items() if v}
    set_t = {k for k, v in dep_t.items() if v}
    dim_c, dim_t = rk_c[0], rk_t[0]
    ambiguous = out["dims"]["ambiguous"]
    repro = _Repro(cand, targ, box, key, [[float(m[p]) for p in targ.params] for m in members])
    truth_subs = {p: sp.Float(repr(float(truth[p]))) for p in targ.params}

    # ---- symbolic attempt whenever the input sets agree
    m = None
    if set_c == set_t:
        try:
            m = match_families(cand, targ, box, key)
        except Exception as exc:  # noqa: BLE001 - a failed normalisation is no proof either way
            m = {"kind": None, "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        out["canonical_tree"] = m.get("tree")
        out["target_tree"] = m.get("target_tree")
        kind = m.get("kind")
        if kind in ("equivalent", "target_in_cand", "cand_in_target"):
            out["parameter_mapping"] = {"target_from_candidate": _fmt_map(m.get("c_to_t")),
                                        "candidate_from_target": _fmt_map(m.get("t_to_c")),
                                        "slot_equations": m.get("equations"), "relation": kind}
        member_ok = None
        if kind in ("equivalent", "target_in_cand"):
            # narrower = the candidate's coefficient roles miss the world's own law or a declared diagnostic member
            member_ok = [feasible(cand, m["t_to_c"], {p: sp.Float(repr(float(v[p]))) for p in targ.params}, rng)
                         or repro.member(i) <= EXACT_TOL for i, v in enumerate(members)]
            out["truth_reachable"] = member_ok[0]
        wider = _maps_outside(cand, targ, m["c_to_t"], box, key) if kind in ("equivalent", "cand_in_target") else None
        if kind == "equivalent":
            narrower = not all(member_ok)
            out["world_law_member"], out["world_member_basis"] = member_ok[0], "symbolic_mapping"
            if not narrower and not wider:
                return _done(out, "equivalent", "E_CANONICAL", "identical canonical tree; slot equations solved both "
                             "ways and spot-checked at 30 digits; coefficient domains agree on the target members "
                             "and on finite candidate draws")
            if wider and not narrower:
                out["containment"], out["containment_basis"] = True, "domain"
                return _done(out, "non_equivalent", "N_DOMAIN_WIDER", "same canonical family, but finite candidate "
                             "coefficient values reach functions outside the target's declared parameter domain")
            if narrower and not wider:
                return _done(out, "non_equivalent", "N_DOMAIN_NARROWER", "same canonical family, but the candidate's "
                             "coefficient roles exclude some target parameter values"
                             + ("" if member_ok[0] else ", including the world's own"))
            return _done(out, "non_equivalent", "N_DOMAIN", "same canonical family, but the coefficient domains differ "
                         "both ways")
        if kind == "target_in_cand" and all(member_ok):
            out["world_law_member"], out["world_member_basis"] = True, "symbolic_mapping"
            witness = {"candidate_from_target": _fmt_map(m["t_to_c"])}
            if dim_c != dim_t or ambiguous:
                out["containment"], out["containment_basis"], out["containment_witness"] = True, "symbolic_mapping", witness
            else:
                # rule 1.3: one solved direction is no proof that the other fails - the candidate is wider only when a
                # finite candidate member lies outside the target family (the target refitted to it)
                wdev = _outside_dev(cand, targ, box, key)
                out["outside_check_dev"] = _r(wdev) if wdev is not None else None
                if wdev is not None and wdev > WITNESS_TOL:
                    out["containment"], out["containment_basis"], out["containment_witness"] = True, "symbolic_mapping", witness
                    return _done(out, "non_equivalent", "N_DOMAIN_WIDER", "the candidate's coefficients solve from the "
                                 "target's (every target member is a candidate member) and a finite candidate member "
                                 f"lies outside the target family (refit deviation {wdev:.3g} > {WITNESS_TOL:g})")
                if wdev is not None and wdev <= EXACT_TOL:
                    return _done(out, "unresolved", "U_SYMBOLIC_UNPROVEN", "the candidate's coefficients solve from the "
                                 "target's and the target family reproduces every finite candidate draw within "
                                 f"{EXACT_TOL:g}, but the reverse mapping was not found")
                return _done(out, "unresolved", "U_NUMERIC_AMBIGUOUS", "the candidate's coefficients solve from the "
                             "target's, the reverse mapping was not found and the outside check is not decisive"
                             + ("" if wdev is None else f" (refit deviation {wdev:.3g})"))
        if kind == "cand_in_target" and dim_c == dim_t and not ambiguous:
            # rule 1.3: narrower only with a target member the candidate does not reproduce
            dev, complete = repro.sequential(WITNESS_TOL)
            out["numeric_containment_dev"] = _r(dev)
            out["numeric_members_evaluated"] = len(repro.devs)
            if dev > WITNESS_TOL:
                if want_membership:
                    _membership(out, cand, targ, box, key, repro, [], m)
                return _done(out, "non_equivalent", "N_DOMAIN_NARROWER", "the candidate is the target family on a "
                             "smaller coefficient domain: target parameters solve from the candidate's, and target "
                             f"member {len(repro.devs) - 1} is not reproduced (relative deviation "
                             f"{repro.devs[len(repro.devs) - 1]:.3g} > {WITNESS_TOL:g} after {STARTS} starts)")
            if dev <= EXACT_TOL:
                out["world_law_member"], out["world_member_basis"], out["world_member_dev"] = True, "numeric", _r(repro.devs[0])
                return _done(out, "unresolved", "U_SYMBOLIC_UNPROVEN", "target parameters solve from the candidate's "
                             f"and the candidate reproduces every target member within {EXACT_TOL:g}, but the reverse "
                             "mapping was not found")
            if want_membership:
                _membership(out, cand, targ, box, key, repro, [], m)
            return _done(out, "unresolved", "U_NUMERIC_AMBIGUOUS", "target parameters solve from the candidate's; "
                         f"numeric deviation {dev:.3g} lies between {EXACT_TOL:g} and {WITNESS_TOL:g}")

    # ---- decisions that do not need the symbolic proof
    if set_c != set_t:
        missing, extra = sorted(set_t - set_c), sorted(set_c - set_t)
        if not missing:
            _containment(out, cand, targ, box, key, repro)
        if want_membership:
            _membership(out, cand, targ, box, key, repro, missing, m)
        return _done(out, "non_equivalent", "N_DEPENDENCE", f"input dependence differs: missing {missing}, extra {extra}")
    if not ambiguous and dim_c != dim_t:
        if dim_c > dim_t:
            _containment(out, cand, targ, box, key, repro)
        if want_membership:
            _membership(out, cand, targ, box, key, repro, [], m)
        return _done(out, "non_equivalent", "N_DIMENSION", f"effective dimension differs: candidate {dim_c}, target {dim_t}")

    # ---- same inputs, same (or ambiguous) dimension, no symbolic proof: numeric witness or unresolved
    dev, complete = repro.sequential(WITNESS_TOL)
    out["numeric_containment_dev"] = _r(dev)
    out["numeric_members_evaluated"] = len(repro.devs)
    if dev > WITNESS_TOL:
        if want_membership:
            _membership(out, cand, targ, box, key, repro, [], m)
        return _done(out, "non_equivalent", "N_NUMERIC_WITNESS", f"same inputs and dimension {dim_c}; no "
                     f"candidate member reproduces target member {len(repro.devs) - 1} (relative deviation "
                     f"{repro.devs[len(repro.devs) - 1]:.3g} > {WITNESS_TOL:g} after {STARTS} starts)")
    if dev <= EXACT_TOL:
        out["world_law_member"], out["world_member_basis"], out["world_member_dev"] = True, "numeric", _r(repro.devs[0])
        return _done(out, "unresolved", "U_SYMBOLIC_UNPROVEN", f"same inputs and dimension {dim_c}; the "
                     f"candidate reproduces every target member within {EXACT_TOL:g} numerically, but no canonical "
                     "proof was found" + ("" if not ambiguous else "; rank ambiguous"))
    if want_membership:
        _membership(out, cand, targ, box, key, repro, [], m)
    return _done(out, "unresolved", "U_NUMERIC_AMBIGUOUS", f"same inputs and dimension {dim_c}; numeric "
                 f"deviation {dev:.3g} lies between {EXACT_TOL:g} and {WITNESS_TOL:g}")


def _r(x):
    return None if x is None or not math.isfinite(x) else float(f"{x:.6g}")


def _containment(out, cand, targ, box, key, repro):
    """Strict containment of the target family: refit every target member in turn (stop at the first one above
    GATE_TOL); when all are within GATE_TOL, search a symbolic witness; else the numeric basis."""
    dev, complete = repro.sequential(GATE_TOL)
    out["numeric_containment_dev"] = _r(dev)
    out["numeric_members_evaluated"] = len(repro.devs)
    if complete and dev <= GATE_TOL:
        try:
            hit = containment_search(cand, targ, box, key, [repro.fits[i] for i in sorted(repro.fits)
                                                             if repro.fits[i] is not None],
                                     [dict(zip(targ.params, vec)) for vec in repro.vectors])
        except Exception as exc:  # noqa: BLE001
            hit = None
            out["containment_search_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        if hit:
            out["containment"], out["containment_basis"] = True, "symbolic"
            out["containment_witness"] = {"substitution": {k: str(v) for k, v in hit["substitution"].items()},
                                          "overall_factor_compensates": hit["mapping"].get("compensated", False),
                                          "target_from_candidate": _fmt_map(hit["mapping"]["c_to_t"]),
                                          "candidate_from_target": _fmt_map(hit["mapping"]["t_to_c"])}
            return
        if dev <= EXACT_TOL:
            out["containment"], out["containment_basis"] = True, "numeric"


def _membership(out, cand, targ, box, key, repro, missing, m):
    """Does the candidate family contain the world's own law (the target at the true parameters)?"""
    if out.get("world_law_member") is not None:
        return
    if missing:
        out["world_law_member"], out["world_member_basis"] = False, "dependence"
        return
    if out["containment"] and out["containment_basis"] == "symbolic":
        out["world_law_member"], out["world_member_basis"] = True, "containment"
        return
    if m and m.get("kind") == "cand_in_target":
        # a restricted member family of the target holds the world's law exactly when its fixed slot values equal
        # the true parameter values
        truth = {p: complex(v) for p, v in zip(targ.params, repro.vectors[0])}
        mapping = m["c_to_t"]
        ok = True
        for p in targ.params:
            if p in mapping:
                expr = mapping[p]
                if expr.free_symbols & set(cand.params):
                    continue
                try:
                    v = complex(sp.N(constants(expr)))
                except Exception:  # noqa: BLE001
                    ok = None
                    break
                if abs(v - truth[p]) > 1e-9 * max(1.0, abs(truth[p])):
                    ok = False
                    break
        if ok is False:
            out["world_law_member"], out["world_member_basis"] = False, "symbolic_restriction"
            return
    dev = repro.member(0)
    out["world_member_dev"] = _r(dev)
    if dev <= EXACT_TOL:
        out["world_law_member"], out["world_member_basis"] = True, "numeric"
    elif dev > WITNESS_TOL:
        out["world_law_member"], out["world_member_basis"] = False, "numeric"
    else:
        out["world_law_member"], out["world_member_basis"] = None, "numeric_ambiguous"


def _done(out, state, code, detail):
    out["state"] = state
    out["reason"] = {"code": code, "detail": detail}
    return out


def record(state, target_id, candidate_expression, reason_code, reason_detail, **extra):
    """The common output record of every benchmark's judge()."""
    rec = {"state": state, "rule_version": RULE_VERSION, "target_id": target_id,
           "candidate_expression": candidate_expression, "canonical_form": None, "parameter_mapping": None,
           "containment": False, "reason": {"code": reason_code, "detail": reason_detail}}
    rec.update(extra)
    assert rec["state"] in STATES
    return rec
