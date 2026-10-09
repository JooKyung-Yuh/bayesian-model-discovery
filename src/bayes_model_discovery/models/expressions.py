"""Expression machinery: whitelisted-AST compilation, canonicalization, fitting.

Every model in this system is a closed-form expression y_hat(x; theta) written
as text - by the preregistered library, or by an LLM. This module is the only
place that text becomes executable, and the only place two texts are decided to
be the same model.

WHAT IS HERE, AND WHY EACH PIECE IS NEEDED
------------------------------------------
`compile_form`      whitelisted-AST compile to a vectorized callable. There is
                    no `eval` of user text: the code object handed to `eval` is
                    compiled from an AST this module has already walked and
                    accepted node by node, and it runs with empty builtins.
`canonical`         the preregistered canonical form v2 - semantic rewrites
                    (1/exp, c**x, exp(u)**k), constant folding to 6 significant
                    digits, +/* chain flattening with NAME-INVARIANT operand
                    sorting, p1/p2/p3 renaming by first appearance in the
                    sorted text, iterated to a fixed point. This is what makes
                    `model_spec_id` a stable identity: two spellings of one
                    model must hash the same, or the pool silently carries
                    duplicates.
`is_reference_form_structural` and its AST helpers - the structural test for
                    the exact reference family C*(1-exp(-A*x)).
`fit_candidate`     the preregistered deterministic multi-start MLE.

WHY THE FITTER TRAVELLED WITH THE GRAMMAR
-----------------------------------------
`fit_candidate` is not expression machinery, and on scope alone it would have
stayed behind with the rest of the proposal layer. It is here because
`inference.laplace.LaplaceMember.start_candidates` uses its solution as ONE of
the deterministic optimizer starts, inside a `try/except Exception: pass`. Had
this function been left in the archive, that except clause would have absorbed
the ImportError and the Laplace backend would have quietly run with a smaller
start set - a numerical change disguised as an unchanged file. The Laplace
migration is required to keep every number identical, so the start set has to
be identical, so this function had to come too. `is_reference_form_structural`
and its AST helpers are here for the same reason: `fit_candidate` calls it to
choose the parameter box, and the box decides the fit.

WHAT STAYED IN THE ARCHIVE (model_proposal.py, disposition: split)
-----------------------------------------------------------------
`ModelPool` and its BIC pseudo-weights, the BMA prediction surface, the
reference baselines (`spline_baseline`, `true_form_mle`, `oracle_predict`,
`constant_baseline`, `library_baseline`), the candidate evaluator
(`evaluate_candidates`, `prediction_errors`, `classify_form`,
`is_reference_form`, `saturating_increasing`), the prompt/ledger text layer
(`parse_model_tags`, `MODEL_TAG_RE`), the older `fit_mle` / `binom_loglik` /
`bic` scoring trio, the Phase-1 grid helpers and the two module self-tests.
None of them is on the inference path, and none is imported from here.
"""
import ast
import math
import re

import numpy as np
from scipy import optimize
from scipy.stats import binom

from ..numerics import EPS_P

MAX_PARAMS = 3          # a proposed form may use at most this many free parameters

# Phase-1 preregistered grammar: functions exp / log / sqrt ONLY. "ln" is
# accepted as a spelling ALIAS of log (natural log, semantically identical -
# declared preprocessing rule, recorded in the Day-1 evidence file). log2/log10
# were dropped from the earlier draft grammar to match the preregistration.
_ALLOWED_FUNCS = {
    "exp": np.exp, "ln": np.log, "log": np.log,
    "sqrt": np.sqrt,
}
_ALLOWED_CONSTS = {"e": math.e, "pi": math.pi}
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARY = (ast.USub, ast.UAdd)


# ---------------------------------------------------------------- expression parsing

class FormError(ValueError):
    """Raised when a proposed expression is outside the whitelisted grammar."""


def _walk_validate(node, names):
    if isinstance(node, ast.Expression):
        _walk_validate(node.body, names)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        _walk_validate(node.left, names)
        _walk_validate(node.right, names)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
        _walk_validate(node.operand, names)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        pass
    elif isinstance(node, ast.Name):
        if node.id == "y":   # reserved output symbol - an external variable, not a parameter
            raise FormError("'y' is the output symbol, not an allowed parameter/variable")
        names.add(node.id)
    elif isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS):
            raise FormError(f"function not allowed: {ast.dump(node.func)}")
        if len(node.args) != 1 or node.keywords:
            raise FormError("functions take exactly one positional argument")
        _walk_validate(node.args[0], names)
    else:
        raise FormError(f"node not allowed: {type(node).__name__}")


def compile_form(expr_str):
    """Whitelisted-AST compile of a proposed expression into a vectorized callable.

    Returns (fn, param_names) where fn(x_array, param_vector) -> np.array and param_names is the
    sorted list of free-parameter symbols (everything that is not x / e / pi / a function name).
    Raises FormError for anything outside the grammar or with > MAX_PARAMS parameters.
    """
    cleaned = expr_str.strip().replace("^", "**")
    cleaned = re.sub(r"\bln\b", "log", cleaned)
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as e:
        raise FormError(f"syntax: {e}") from e
    names = set()
    _walk_validate(tree, names)
    params = sorted(n for n in names if n not in ("x",) and n not in _ALLOWED_CONSTS
                    and n not in _ALLOWED_FUNCS)
    if len(params) > MAX_PARAMS:
        raise FormError(f"too many parameters ({len(params)} > {MAX_PARAMS}): {params}")
    code = compile(tree, "<proposed-form>", "eval")

    def fn(x, theta):
        """Evaluate the compiled form at inputs `x` with parameter vector `theta`.

        Returns a float array with the shape of `x`."""
        env = {"x": np.asarray(x, dtype=float)}
        env.update(_ALLOWED_FUNCS)
        env.update(_ALLOWED_CONSTS)
        env.update({p: float(v) for p, v in zip(params, theta)})
        with np.errstate(all="ignore"):
            # the code object came from OUR whitelisted AST, and builtins are empty
            out = eval(code, {"__builtins__": {}}, env)
        return np.broadcast_to(np.asarray(out, dtype=float), np.shape(env["x"])).copy()

    return fn, params


def _round_sig(v, sig=6):
    """Round a float to `sig` significant digits (preregistered constant normalization)."""
    if v == 0 or not math.isfinite(v):
        return v
    # the int() is redundant on CPython (math.floor already returns int) and is
    # kept so the rounding expression stays byte-identical to the archived one
    return round(v, -int(math.floor(math.log10(abs(v)))) + (sig - 1))  # noqa: RUF046


# ---- canonicalization v2: chain flattening, name-invariant sorting,
# ---- rename-after-sort to p1/p2/p3, exp rewrites, fixed-point iteration.

class _Rewriter(ast.NodeTransformer):
    """Semantic-preserving rewrites applied BEFORE rendering:
      - fold named constants e/pi and negative constants; reduce double negation;
      - a / exp(u)      ->  a * exp(-u)          (covers the 1/exp reference rewrite)
      - c ** g(x), c>0  ->  exp(g(x) * ln(c))    (covers 2**(-b*x) style rewrites)
      - exp(u) ** k     ->  exp(k * u)
      - round numeric constants to 6 significant digits."""

    def visit_Name(self, node):
        """Replace the named constants e/pi by their rounded numeric value; keep other names."""
        if node.id in _ALLOWED_CONSTS:
            return ast.copy_location(ast.Constant(value=_round_sig(_ALLOWED_CONSTS[node.id])), node)
        return node

    def visit_Constant(self, node):
        """Round a numeric constant to 6 significant digits (as a float); keep other constants."""
        if isinstance(node.value, (int, float)):
            return ast.copy_location(ast.Constant(value=_round_sig(float(node.value))), node)
        return node

    def visit_UnaryOp(self, node):
        """Normalize unary signs: drop unary plus, cancel double negation, fold negative
        constants and push a minus sign into */ chains. Returns the rewritten node."""
        self.generic_visit(node)
        if isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.UnaryOp) and isinstance(node.operand.op, ast.USub):
                return node.operand.operand
            if isinstance(node.operand, ast.Constant):
                return ast.copy_location(ast.Constant(value=-node.operand.value), node)
            # push the sign into */ chains so -(a*x), (-a)*x, a*(-x) all normalize alike,
            # including inside function arguments like exp(-(b*x))
            if isinstance(node.operand, ast.BinOp) and isinstance(node.operand.op,
                                                                  (ast.Mult, ast.Div)):
                inner = node.operand
                new_left = self.visit(ast.UnaryOp(op=ast.USub(), operand=inner.left))
                pushed = ast.BinOp(left=new_left, op=inner.op, right=inner.right)
                ast.fix_missing_locations(pushed)
                return self.visit_BinOp(pushed)
        if isinstance(node.op, ast.UAdd):
            return node.operand
        return node

    def visit_BinOp(self, node):
        """Apply the exp rewrites a/exp(u), exp(u)**k and c**g to a binary operation.

        Returns the rewritten node (unchanged when no rewrite applies)."""
        self.generic_visit(node)
        # a / exp(u) -> a * exp(-u)
        if isinstance(node.op, ast.Div) and isinstance(node.right, ast.Call) \
                and isinstance(node.right.func, ast.Name) and node.right.func.id == "exp":
            neg = ast.UnaryOp(op=ast.USub(), operand=node.right.args[0])
            new_exp = ast.Call(func=ast.Name(id="exp", ctx=ast.Load()), args=[neg], keywords=[])
            out = ast.BinOp(left=node.left, op=ast.Mult(), right=new_exp)
            ast.fix_missing_locations(out)
            return self.visit_BinOp(out)
        if isinstance(node.op, ast.Pow):
            # exp(u) ** k -> exp(k * u)
            if isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Name) \
                    and node.left.func.id == "exp":
                prod = ast.BinOp(left=node.right, op=ast.Mult(), right=node.left.args[0])
                out = ast.Call(func=ast.Name(id="exp", ctx=ast.Load()), args=[prod], keywords=[])
                ast.fix_missing_locations(out)
                return out
            # c ** g  (c positive constant != 1) -> exp(g * ln c)
            if isinstance(node.left, ast.Constant) and isinstance(node.left.value, (int, float)) \
                    and node.left.value > 0 and node.left.value != 1.0:
                lnc = ast.Constant(value=_round_sig(math.log(float(node.left.value))))
                prod = ast.BinOp(left=node.right, op=ast.Mult(), right=lnc)
                out = ast.Call(func=ast.Name(id="exp", ctx=ast.Load()), args=[prod], keywords=[])
                ast.fix_missing_locations(out)
                return out
        return node


def _is_param_name(name):
    return name not in ("x",) and name not in _ALLOWED_CONSTS and name not in _ALLOWED_FUNCS


def _render(node):
    """Render an AST node to canonical text with flattened, sorted +/* chains.

    Returns (text, key) where key is the NAME-INVARIANT sort key (every parameter symbol
    replaced by 'P'), so `a*x` and `b*x` sort identically and renaming cannot perturb order.
    Additive chains flatten through +/- (a-b == a + -b); multiplicative chains flatten
    through *. Division and power stay binary. Operands with internal structure are
    parenthesized so the output re-parses unambiguously."""
    def atom(t):
        """True when rendered text `t` contains no space, i.e. needs no parentheses."""
        return " " not in t

    def paren(t):
        """Return rendered text `t`, wrapped in parentheses unless it is an atom."""
        return t if atom(t) else f"({t})"

    if isinstance(node, ast.Expression):
        return _render(node.body)
    if isinstance(node, ast.Constant):
        t = repr(node.value)
        return t, t
    if isinstance(node, ast.Name):
        return node.id, ("P" if _is_param_name(node.id) else node.id)
    if isinstance(node, ast.Call):
        t, k = _render(node.args[0])
        return f"{node.func.id}({t})", f"{node.func.id}({k})"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        t, k = _render(node.operand)
        return f"-{paren(t)}", f"-{paren(k)}"
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            terms = []

            def collect_add(n, negate=False):
                """Flatten an additive chain into `terms` as (key, text) pairs, tracking sign."""
                if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
                    collect_add(n.left, negate)
                    collect_add(n.right, negate)
                elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub):
                    collect_add(n.left, negate)
                    collect_add(n.right, not negate)
                elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
                    collect_add(n.operand, not negate)
                else:
                    t, k = _render(n)
                    if negate:
                        t, k = f"-{paren(t)}", f"-{paren(k)}"
                    terms.append((k, t))

            collect_add(node)
            terms.sort(key=lambda p: p[0])
            text = " + ".join(t for _, t in terms)
            key = " + ".join(k for k, _ in terms)
            return text, key
        if isinstance(node.op, ast.Mult):
            factors, sign = [], [1]

            def collect_mul(n):
                """Flatten a multiplicative chain into `factors`, accumulating the sign in `sign`."""
                # sign is extracted to the CHAIN level so -(a*b), (-a)*b and a*(-b) all
                # canonicalize identically
                if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
                    collect_mul(n.left)
                    collect_mul(n.right)
                elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
                    sign[0] *= -1
                    collect_mul(n.operand)
                elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) \
                        and n.value < 0:
                    sign[0] *= -1
                    if -n.value != 1.0:               # drop the identity factor 1.0
                        factors.append(_render(ast.Constant(value=-n.value)))
                elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) \
                        and n.value == 1.0:
                    pass                               # multiplicative identity
                else:
                    factors.append(_render(n))

            collect_mul(node)
            if not factors:
                factors = [("1.0", "1.0")]
            factors.sort(key=lambda p: p[1])
            neg = "-" if sign[0] < 0 else ""
            text = neg + " * ".join(paren(t) for t, _ in factors)
            key = neg + " * ".join(paren(k) for _, k in factors)
            return text, key
        op = {ast.Div: "/", ast.Pow: "**"}[type(node.op)]
        lt, lk = _render(node.left)
        rt, rk = _render(node.right)
        return f"{paren(lt)} {op} {paren(rt)}", f"{paren(lk)} {op} {paren(rk)}"
    raise FormError(f"render: node not allowed: {type(node).__name__}")


def _canonical_once(text):
    tree = ast.parse(text, mode="eval")
    tree = _Rewriter().visit(tree)
    ast.fix_missing_locations(tree)
    rendered, _ = _render(tree)
    # rename parameters to p1/p2/p3 in order of first appearance in the SORTED text
    seen, out_tokens = {}, []
    for tok in re.split(r"([A-Za-z_][A-Za-z_0-9]*)", rendered):
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", tok or "") and _is_param_name(tok):
            if tok not in seen:
                seen[tok] = f"p{len(seen) + 1}"
            tok = seen[tok]
        out_tokens.append(tok)
    return "".join(out_tokens)


def canonical(expr_str):
    """Preregistered canonical form v2: rewrites (1/exp, c**x, exp**k), constant folding to
    6 significant digits, +/* chain flattening with NAME-INVARIANT operand sorting, then
    p1/p2/p3 renaming by first appearance in the sorted text, iterated to a fixed point.
    Falls back to the cleaned raw string when parsing fails (callers treat as invalid)."""
    cleaned = expr_str.strip().replace("^", "**")
    cleaned = re.sub(r"\bln\b", "log", cleaned)   # declared alias: ln == log (natural log)
    try:
        cur = _canonical_once(cleaned)
        for _ in range(3):                          # fixed-point iteration
            nxt = _canonical_once(cur)
            if nxt == cur:
                break
            cur = nxt
        return cur
    except (SyntaxError, FormError):
        return cleaned


# ---------------- structural classification (deterministic, AST-based)

def _contains_x(node):
    return any(isinstance(n, ast.Name) and n.id == "x" for n in ast.walk(node))


def _calls_of(tree, fname):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == fname]


def _is_linear_in_x(node):
    """True if the (sub)expression is affine in x: a*x, a*x+b, x/c ... (a,b x-free)."""
    if not _contains_x(node):
        return True
    if isinstance(node, ast.Name) and node.id == "x":
        return True
    if isinstance(node, ast.UnaryOp):
        return _is_linear_in_x(node.operand)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return _is_linear_in_x(node.left) and _is_linear_in_x(node.right)
        if isinstance(node.op, ast.Mult):
            lx, rx = _contains_x(node.left), _contains_x(node.right)
            if lx and rx:
                return False
            return _is_linear_in_x(node.left) if lx else _is_linear_in_x(node.right)
        if isinstance(node.op, ast.Div):
            return (not _contains_x(node.right)) and _is_linear_in_x(node.left)
    return False


def _affine_in_single_exp(tree):
    """True if expr == A + B * exp(linear(x)) with A, B free of x and exactly one exp whose
    argument contains x linearly -- the structural signature of the reference family
    C*(1-exp(-A*x)) and its algebraic rewrites."""
    exps = _calls_of(tree, "exp")
    if len(exps) != 1:
        return False
    e = exps[0]
    if not (_contains_x(e.args[0]) and _is_linear_in_x(e.args[0])):
        return False

    def affine(node):   # affine in the exp-node `e`, coefficients free of x
        """True if `node` is affine in the single exp node `e` with x-free coefficients."""
        if node is e:
            return True
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, (ast.Name,)):
            return node.id != "x"
        if isinstance(node, ast.UnaryOp):
            return affine(node.operand)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return affine(node.left) and affine(node.right)
            if isinstance(node.op, ast.Mult):
                le = _node_contains(node.left, e)
                re_ = _node_contains(node.right, e)
                if le and re_:
                    return False
                dep, free = (node.left, node.right) if le else (node.right, node.left)
                return affine(dep) and not _contains_x(free) and not _node_contains(free, e)
            if isinstance(node.op, ast.Div):
                return affine(node.left) and not _contains_x(node.right) \
                    and not _node_contains(node.right, e)
        return False

    return affine(tree.body)


def _node_contains(node, target):
    return any(n is target for n in ast.walk(node))


def structural_zero_at_origin(expr_str):
    """True if f(0) == 0 for ALL parameter values (checked at 3 deterministic draws) --
    the structural signature separating the exact reference family C*(1-exp(-A*x))
    (which passes through the origin by construction) from offset variants a*exp(b*x)+c."""
    try:
        fn, params = compile_form(canonical(expr_str))
    except FormError:
        return False
    draws = [[0.7, 1.3, 2.1], [1.9, 0.3, 0.8], [2.6, 1.1, 0.45]]
    for d in draws:
        with np.errstate(all="ignore"):
            v = fn(np.array([0.0]), d[:len(params)])
        if not np.isfinite(v[0]) or abs(v[0]) > 1e-9:
            return False
    return True


def _family_exp_affine(expr_str):
    """Structural membership of the exp-affine family A + B*exp(linear(x)) -- the superset
    containing the reference family and its rewrites. Orientation (growth vs saturation)
    is NOT decided here; that requires parameter values."""
    try:
        tree = ast.parse(canonical(expr_str), mode="eval")
    except SyntaxError:
        return False
    return _affine_in_single_exp(tree)


def is_reference_form_structural(expr_str):
    """EXACT reference family, structural half: exp-affine AND structurally zero at the
    origin (i.e. algebraically reducible to C*(1-exp(-A*x))). Offset forms like
    a*exp(b*x)+c are NOT exact even when fitted signs make them behaviorally equivalent --
    those count at the family/class level instead."""
    return _family_exp_affine(expr_str) and structural_zero_at_origin(expr_str)


# ---------------- preregistered fitter (deterministic multistart)

_STARTS_PER_DIM = {1: 9, 2: 5, 3: 3}    # -> 9 / 25 / 27 total starts


# Deterministic per-dimension start values (preregistered COUNTS 9/25/27; the VALUES are an
# implementation choice). The axes exclude 0 -- a 0 start is a saddle for exp-affine forms
# (the scale parameter kills the rate gradient), which made the fitter miss the global
# optimum on noisy draws (verified: seed-0 offset form -287.6 vs -263.7). Each axis mixes
# scale-sized and rate-sized magnitudes of both signs.
_START_AXES = {
    1: [-100.0, -25.0, -5.0, -1.0, 0.2, 1.0, 5.0, 25.0, 100.0],   # 9 starts
    2: [-50.0, -1.0, 0.5, 5.0, 50.0],                              # 5^2 = 25 starts
    3: [-50.0, -1.0, 50.0],                                        # 3^3 = 27 starts
}


def _start_grid(k, lo, hi, log_space):
    per = _STARTS_PER_DIM[k]
    if log_space:
        axis = np.geomspace(max(lo, 1e-6), hi, per)
    else:
        axis = np.array(_START_AXES[k])
    grids = np.meshgrid(*([axis] * k), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def fit_candidate(expr_str, D, objective="count", N=50):
    """Preregistered fit of one candidate form on the fixed batch D=[(x,y),...].

    objective="count": maximize count-style likelihood -- predicted mean clipped to [0, N],
        scored as Binomial(N, clip(mean)/N).  (primary reported fit)
    objective="ls"   : minimize squared error between predicted mean and realized outputs.
        (fitter-neutrality ablation)
    Bounds: [-100, 100]; positivity classes (exponential_saturation) use [1e-6, 100].
    Deterministic multistart: 9/25/27 starts for 1/2/3 parameters. Fit fails when all
    starts fail or predictions are non-finite. k=0 forms are evaluated directly.
    """
    try:
        canon = canonical(expr_str)
        fn, params = compile_form(canon)
    except FormError as e:
        return {"ok": False, "why": f"invalid ({e})", "params": None, "score": None}
    xs = np.array([d[0] for d in D], float)
    ys = np.array([d[1] for d in D], float)

    def count_score(theta):
        """Binomial(N, clip(mean)/N) log-likelihood of the batch at `theta` (-inf if non-finite)."""
        with np.errstate(all="ignore"):
            yhat = fn(xs, theta)
        if not np.all(np.isfinite(yhat)):
            return -np.inf
        p = np.clip(np.clip(yhat, 0.0, float(N)) / N, EPS_P, 1.0 - EPS_P)
        s = binom.logpmf(ys.astype(int), N, p).sum()
        return float(s) if np.isfinite(s) else -np.inf

    def ls_score(theta):
        """Negative sum of squared errors of the batch at `theta` (-inf if non-finite)."""
        with np.errstate(all="ignore"):
            yhat = fn(xs, theta)
        if not np.all(np.isfinite(yhat)):
            return -np.inf
        return -float(np.sum((yhat - ys) ** 2))

    score_fn = count_score if objective == "count" else ls_score
    if not params:
        s = score_fn([])
        return {"ok": np.isfinite(s), "params": [], "score": (s if np.isfinite(s) else None),
                "canonical": canon}

    # positivity bounds ONLY for the literal saturation parameterization C*(1-exp(-A*x));
    # the wider exp-affine family (e.g. a*exp(b*x)+c) needs NEGATIVE coefficients to reach
    # its reference-equivalent fit and gets the default box.
    positivity = is_reference_form_structural(canon)
    lo, hi = (1e-6, 100.0) if positivity else (-100.0, 100.0)
    k = len(params)
    starts = _start_grid(k, lo, hi, log_space=positivity)

    best_x, best_s = None, -np.inf
    for s0 in starts:
        try:
            res = optimize.minimize(lambda t: -score_fn(t), s0, method="L-BFGS-B",
                                    bounds=[(lo, hi)] * k)
        except Exception:                            # noqa: BLE001, S112 - a start that
            continue                                 # fails is dropped, not reported
        if np.isfinite(res.fun) and -res.fun > best_s:
            best_x, best_s = res.x, -res.fun
    if best_x is None or not np.isfinite(best_s):
        return {"ok": False, "why": "all starts failed", "params": None, "score": None,
                "canonical": canon}
    return {"ok": True, "params": [float(v) for v in best_x], "score": float(best_s),
            "canonical": canon}
