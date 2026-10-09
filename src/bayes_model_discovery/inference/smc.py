"""Sequential Monte Carlo: the PRIMARY inference backend.

Every log evidence and every particle set in the frozen acquisition run came
out of this module. It builds a PyMC model that reproduces the production joint
density exactly, samples it with `pm.sample_smc`, and extracts the evidence and
the SMC health diagnostics that the run stores as raw material.

WHY THE TWO GATES EXIST
-----------------------
The model here is written twice: once as a numpy callable (`compile_form`, the
production compiler) and once as a pytensor graph (`symbolic_mu`), because a
free-form challenger expression cannot be hand-translated per form. Two
spellings of one model is exactly the failure mode where an audit ends up
auditing a different model than the one that ran, so both spellings are
checked against each other before any sampling happens:

  `validate_symbolic`  the pytensor mean function must agree with the
                       production compiler at 6 probe points, INCLUDING the
                       pattern of non-finite values (SYMBOLIC_TOL = 1e-8).
  `validate_logp`      the PyMC joint logp must equal
                       `scoring.count_loglik` + the policy prior, at 4 probe
                       points, to 1e-7, and at least ONE probe must have been
                       comparable — a gate whose every reference value came out
                       non-finite has compared nothing and says so instead of
                       returning 0.0.

`run_smc` runs BOTH gates itself, on every call, before sampling. A gate that
only runs in a separate audit script is a gate the production path can skip,
and running only one of the two leaves the other in exactly that position.
Neither gate touches global RNG state - each probes through its own
`np.random.default_rng(<fixed seed>)` and the sampler is seeded explicitly - so
adding them to the production path cannot move a sampled value.

WHAT THE GATES CANNOT SEE, AND WHAT DOES
----------------------------------------
The gates compare two SPELLINGS of one model, so they are blind to a
disagreement both spellings implement faithfully. The registered example is
arity: `build_model` passes prior mu/sigma arrays of length len(prior_rows)
with `shape=len(param_names)`, and PyMC broadcasts one row into d i.i.d.
priors; `validate_logp` broadcasts the same rows with numpy and agrees. So
`check_inference_inputs` runs FIRST, before any model exists, and refuses the
disagreement structurally: the three per-parameter sequences must be the same
length, the names must be distinct, and every prior must be finite with a
positive sigma.

THE EVIDENCE COMBINATION
------------------------
Two levels, the same rule at both, and the rule is an average in EVIDENCE
space, not in log space:

  chains  `_extract_logz` takes the last finite per-stage log marginal
          likelihood of each chain and combines the SMC_CHAINS = 2 of them.
  seeds   the caller combines the SMC_SEEDS = 2 per-seed values the same way.

Both are `combine_log_evidence`: logsumexp(values) - log(len(values)). With two
values that is "logsumexp minus log 2". Averaging in log space instead would
report the geometric mean of two evidence estimates, which is not an unbiased
estimate of anything the model posterior wants.

PyMC 5.13 stores unequal per-chain SMC stage counts as an object array whose
cells are the stage lists, so indexing by the first axis is only correct in the
equal-length case; `_chain_units` handles both layouts and refuses anything it
cannot map onto exactly `n_chains` units rather than guessing.

FAILURES ARE RECORDED, NEVER DROPPED
------------------------------------
`run_smc` returns `{"ok": False, "failure_reason": ...}, None` instead of
raising. A caller that treats a missing result as "this model scored badly"
would silently reshape the pool; the caller's job is to halt on `ok=False`, and
the recorded reason is what makes that halt explainable afterwards.

Notes from porting this module from the earlier prototype
---------------------------------------------------------
PARTIAL: this module is the inference half of the archived audit script; the
other half (particle views for design scoring) is not part of this package.
Every numeric constant, tolerance, probe count, probe seed, rounding position
and comparison operator below is unchanged from the archived file. Edits: the
sys.path bootstrap is gone; imports are rewritten to this package
(`from mp_refit import count_loglik` -> `from .scoring import count_loglik`,
`from model_proposal import compile_form` -> `from ..models.expressions import
compile_form`); `AuditError` is renamed `SMCError`, because that name belonged
to the audit driver and this module is not it; and the chain/seed combination
is named `combine_log_evidence` instead of being written out twice.

`batch_mu_fn` here is the PYTENSOR evaluator. It exists solely so
`validate_symbolic` can compare the graph that the sampler will actually use
against the production compiler (a numpy callable): keeping both spellings is
the point, because if they ever disagree, the gate fires.
"""
import ast
import math
import time

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from ..models.expressions import compile_form
from ..models.schema import TRANSFORMS
from ..numerics import EPS_P
from .evidence import EvidenceError
from .evidence import combine_log_evidence as _combine_log_evidence
from .scoring import count_loglik

N_TRIALS = 50
SMC_DRAWS = 2000
SMC_CHAINS = 2
SMC_SEEDS = (20260726, 20260727)      # two INDEPENDENT seeds, fixed here
SYMBOLIC_TOL = 1e-8                   # production-vs-symbolic agreement gate


class SMCError(RuntimeError):
    """A registered invariant broke - inference stops rather than degrade."""


def _r(v, n=6):
    if v is None:
        return None
    v = float(v)
    return None if not math.isfinite(v) else round(v, n)


# ------------------------------------------------------- structural fail-closed

def check_inference_inputs(param_names, transforms, prior_rows):
    """Refuse an inference request whose per-parameter sequences disagree, or
    whose transforms are not in the registered vocabulary.

    Runs BEFORE any model is built, because every check downstream compares two
    spellings of one model and a broadcast prior is faithfully implemented by
    both. The three sequences describe the SAME parameter vector, position by
    position, so any difference in their lengths means the caller does not know
    what it is asking for. Silent broadcasting turns that into a run on a prior
    nobody specified, which is unrecoverable afterwards: the stored evidence
    looks ordinary."""
    n_p, n_t, n_r = len(param_names), len(transforms), len(prior_rows)
    if not (n_p == n_t == n_r):
        raise SMCError(
            f"parameter arity disagreement: {n_p} parameter names, {n_t} "
            f"transforms, {n_r} prior rows — the three describe one parameter "
            "vector position by position and must have equal length")
    # Structural for the same reason: every phi -> theta map on this path -
    # `_theta_of`, `batch_mu_fn`, `validate_symbolic`, `validate_logp`,
    # `theta_summary` - is written as `exp(phi) if t == "log" else phi`, so an
    # unregistered transform is applied as the IDENTITY by all of them at once.
    # Both gates therefore agree, exactly as they agreed on the broadcast prior,
    # and the run samples a model nobody asked for while its sidecars record the
    # transform that was requested. Naming the value and the position is the
    # whole message: the caller's next question is which parameter it was.
    for i, t in enumerate(transforms):
        if t not in TRANSFORMS:
            raise SMCError(
                f"unknown transform {t!r} at position {i} "
                f"({param_names[i]!r}): the registered vocabulary is "
                f"{sorted(TRANSFORMS)} — an unregistered transform is applied "
                "as the identity by every phi->theta map here, so the request "
                "would be answered by a different model than the one asked for")
    seen, dupes = set(), []
    for nm in param_names:
        if nm in seen and nm not in dupes:
            dupes.append(nm)
        seen.add(nm)
    if dupes:
        raise SMCError(f"duplicate parameter name(s) {dupes} in {list(param_names)} — "
                       "a repeated name makes the phi index ambiguous")
    for i, row in enumerate(prior_rows):
        where = f"prior row {i} ({param_names[i]!r})"
        for key in ("phi_prior_mu", "phi_prior_sigma"):
            if key not in row:
                raise SMCError(f"{where}: no {key}")
            try:
                val = float(row[key])
            except (TypeError, ValueError):
                raise SMCError(f"{where}: {key} {row[key]!r} is not a number")
            if not math.isfinite(val):
                raise SMCError(f"{where}: non-finite {key} {row[key]!r}")
        if float(row["phi_prior_sigma"]) <= 0.0:
            raise SMCError(f"{where}: phi_prior_sigma "
                           f"{row['phi_prior_sigma']!r} is not positive")
    return n_p


# ------------------------------------------------- symbolic model translation

_PT_FUNCS = {"exp": pt.exp, "log": pt.log, "sqrt": pt.sqrt}
_CONSTS = {"e": math.e, "pi": math.pi}


def symbolic_mu(canonical, param_names, theta_syms, x_sym):
    """Translate a canonical expression into pytensor over the SAME whitelisted
    grammar the production compiler accepts (+ - * / ** , exp/log/sqrt, e/pi).

    A general translator is required: the challengers are free-form, so a
    per-form hand-written symbolic model cannot cover them."""
    env = dict(zip(param_names, theta_syms))
    env["x"] = x_sym

    def walk(node):
        """Recursively translate one AST node into a pytensor expression (a
        plain float for a numeric literal or e/pi); raise SMCError on any node
        outside the whitelisted grammar."""
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left), walk(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.Div):
                return a / b
            if isinstance(op, ast.Pow):
                return a ** b
            raise SMCError(f"binop outside the grammar: {ast.dump(op)}")
        if isinstance(node, ast.UnaryOp):
            v = walk(node.operand)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return v
            raise SMCError("unary op outside the grammar")
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in _CONSTS:
                return _CONSTS[node.id]
            raise SMCError(f"unknown name {node.id!r}")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name)
                    and node.func.id in _PT_FUNCS):
                raise SMCError("call outside the grammar")
            return _PT_FUNCS[node.func.id](walk(node.args[0]))
        raise SMCError(f"node outside the grammar: {type(node).__name__}")

    src = canonical.strip().replace("^", "**")
    return walk(ast.parse(src, mode="eval"))


def _theta_of(phi, transforms):
    return pt.stack([pt.exp(phi[i]) if t == "log" else phi[i]
                     for i, t in enumerate(transforms)])


def batch_mu_fn(canonical, param_names, transforms):
    """Compile phi-matrix (n, d) x design vector (m,) -> mu (n, m).

    This is the PYTENSOR evaluator, and its only caller is `validate_symbolic`:
    it exists so the gate compares the production compiler against the same
    symbolic path the sampler uses. The EIG code in `eig.joint` does not call
    it; it evaluates means through the numpy production compiler."""
    phi = pt.matrix("phi")
    xs = pt.vector("xs")
    # dimshuffle('x') marks the new axis as genuinely broadcastable; a plain
    # slice leaves the length unknown and pytensor >=2.20 then refuses to
    # runtime-broadcast a (n, 1) against a (1, m)
    cols = [phi[:, i].dimshuffle(0, "x") for i in range(len(transforms))]
    theta = [pt.exp(c) if t == "log" else c
             for c, t in zip(cols, transforms)]
    x_row = xs.dimshuffle("x", 0)
    mu = symbolic_mu(canonical, param_names, theta, x_row)
    # a form that ignores x (or ignores a parameter) yields a partially shaped
    # result; fill it out to the full (n, m) grid so every caller sees one shape
    mu = pt.second(cols[0] + x_row, mu)
    return pytensor.function([phi, xs], mu, on_unused_input="ignore")


def validate_symbolic(canonical, param_names, transforms, xs, n_probe=6,
                      seed=13):
    """Gate: the symbolic mean function must agree with the production
    compiler. Without this the sampler would be sampling a different model."""
    fn_prod, names = compile_form(canonical)
    if list(names) != list(param_names):
        raise SMCError(f"{canonical}: parameter names disagree")
    f = batch_mu_fn(canonical, param_names, transforms)
    rng = np.random.default_rng(seed)
    phis = rng.normal(0.0, 1.0, size=(n_probe, len(param_names)))
    got = np.asarray(f(phis, np.asarray(xs, float)), dtype=float)
    worst = 0.0
    for i, phi in enumerate(phis):
        theta = [math.exp(v) if t == "log" else v
                 for v, t in zip(phi, transforms)]
        with np.errstate(all="ignore"):
            want = np.asarray(fn_prod(np.asarray(xs, float), theta), float)
        both = np.isfinite(want) & np.isfinite(got[i])
        if not np.array_equal(np.isfinite(want), np.isfinite(got[i])):
            raise SMCError(f"{canonical}: finiteness pattern differs")
        if both.any():
            worst = max(worst, float(np.max(np.abs(want[both] - got[i][both]))))
    if worst > SYMBOLIC_TOL:
        raise SMCError(f"{canonical}: symbolic/production mismatch {worst}")
    return worst


# ------------------------------------------------------------- the PyMC model

def _log_binom_coeff(ys, n):
    from scipy.special import gammaln
    ys = np.asarray(ys, float)
    return float(np.sum(gammaln(n + 1) - gammaln(ys + 1) - gammaln(n - ys + 1)))


def build_model(canonical, param_names, transforms, prior_rows, D):
    """A PyMC model that reproduces the production joint density exactly:
    per-parameter Normal priors in phi space (the common policy), the same
    log/identity transforms, the same clip-then-divide link, and the same
    Binomial(50, p) likelihood including its combinatorial constant."""
    xs = np.array([float(x) for x, _ in D])
    ys = np.array([float(y) for _, y in D])
    mu0 = np.array([r["phi_prior_mu"] for r in prior_rows], float)
    sd0 = np.array([r["phi_prior_sigma"] for r in prior_rows], float)
    const = _log_binom_coeff(ys, N_TRIALS)

    with pm.Model() as model:
        phi = pm.Normal("phi", mu=mu0, sigma=sd0, shape=len(param_names))
        theta = _theta_of(phi, transforms)
        mu_raw = symbolic_mu(canonical, param_names,
                             [theta[i] for i in range(len(param_names))],
                             pt.as_tensor_variable(xs))
        mu_raw = pt.second(pt.as_tensor_variable(xs), mu_raw)
        p = pt.clip(pt.clip(mu_raw, 0.0, float(N_TRIALS)) / float(N_TRIALS),
                    EPS_P, 1.0 - EPS_P)
        ll = const + pt.sum(ys * pt.log(p) + (N_TRIALS - ys) * pt.log1p(-p))
        # production returns -inf when ANY predicted mean is non-finite;
        # mirroring that exactly matters for the evidence
        finite = pt.all(pt.bitwise_not(pt.isnan(mu_raw) | pt.isinf(mu_raw)))
        ll = pt.switch(finite, ll, -np.inf)
        pm.Potential("likelihood", ll)
    return model


def validate_logp(model, canonical, param_names, transforms, prior_rows, D,
                  n_probe=4, seed=29):
    """Gate: the PyMC joint logp must equal production loglik + policy prior.

    A probe whose production reference is non-finite carries no information and
    is skipped. If EVERY probe is skipped the gate has compared nothing, and
    returning 0.0 there would report the strongest possible agreement on the
    strength of zero evidence — so a run of `n_probe` skips is a refusal."""
    fn, _ = compile_form(canonical)
    mu0 = np.array([r["phi_prior_mu"] for r in prior_rows], float)
    sd0 = np.array([r["phi_prior_sigma"] for r in prior_rows], float)
    logp = model.compile_logp()
    rng = np.random.default_rng(seed)
    worst = 0.0
    n_compared = 0
    for _ in range(n_probe):
        phi = rng.normal(0.0, 0.8, size=len(param_names))
        theta = [math.exp(v) if t == "log" else v
                 for v, t in zip(phi, transforms)]
        want = count_loglik(fn, theta, list(D), N_TRIALS)
        want += float(np.sum(-0.5 * ((phi - mu0) / sd0) ** 2
                             - np.log(sd0) - 0.5 * math.log(2 * math.pi)))
        got = float(logp({"phi": phi}))
        if not math.isfinite(want):
            continue
        n_compared += 1
        worst = max(worst, abs(got - want))
    if n_compared == 0:
        raise SMCError(
            f"{canonical}: the logp gate compared 0 of {n_probe} probes — every "
            "production reference value was non-finite, so the gate passed "
            "vacuously rather than agreeing with anything")
    if worst > 1e-7:
        raise SMCError(f"{canonical}: PyMC/production logp mismatch {worst}")
    return worst


# --------------------------------------------------- evidence and diagnostics

def combine_log_evidence(values):
    """Compatibility surface over the shared dependency-light implementation."""
    try:
        return _combine_log_evidence(values)
    except EvidenceError as exc:
        raise SMCError(str(exc)) from exc


def _extract_logz(idata):
    """Per-chain SMC evidence, combined by averaging in EVIDENCE space.

    (PyMC 5.13 stores unequal per-chain stage counts as an object array; the
    handling below maps both layouts onto exactly `n_chains` units, the same
    rule `_chain_units` applies, and refuses anything else.)"""
    raw = np.asarray(idata.sample_stats["log_marginal_likelihood"].values,
                     dtype=object)
    n_chains = int(idata.posterior.sizes["chain"])

    def finite(v):
        """Yield every finite float found in a possibly nested array, list or
        tuple, in order, skipping None and non-finite values."""
        if isinstance(v, np.ndarray):
            for it in v.ravel():
                yield from finite(it)
            return
        if isinstance(v, (list, tuple)):
            for it in v:
                yield from finite(it)
            return
        if v is None:
            return
        f = float(v)
        if math.isfinite(f):
            yield f

    units = ([raw[i] for i in range(n_chains)] if raw.shape[0] == n_chains
             else list(raw.ravel()))
    if len(units) != n_chains:
        raise SMCError(f"cannot map SMC stats {raw.shape} to {n_chains}")
    per_chain = []
    for u in units:
        vals = list(finite(u))
        if not vals:
            raise SMCError("SMC chain produced no finite evidence")
        per_chain.append(vals[-1])
    return combine_log_evidence(per_chain), per_chain


def _chain_units(raw, n_chains):
    """Map a sample_stats array onto per-chain sequences.

    PyMC 5.13 represents unequal per-chain SMC stage counts as a (1, chains)
    object array whose cells are the stage lists, so indexing by the first
    axis is only correct in the equal-length case."""
    if raw.shape and raw.shape[0] == n_chains:
        return [raw[i] for i in range(n_chains)]
    cells = list(raw.ravel())
    if len(cells) != n_chains:
        raise SMCError(f"cannot map SMC stats {raw.shape} to {n_chains}")
    return cells


def _extract_diagnostics(idata):
    """SMC health, which `sample_stats` exposes and which was previously
    discarded: the tempering schedule, the per-stage acceptance rate and the
    per-stage evidence. These are the only evidence that the reference
    inference behaved, and they cannot be recovered without re-running."""
    ss = idata.sample_stats
    n_chains = int(idata.posterior.sizes["chain"])

    def as_lists(name, keep_gaps):
        """`keep_gaps` preserves the None positions, which are meaningful for
        the per-stage evidence (early stages legitimately have none) and are
        pure padding for the tempering schedule and acceptance rate."""
        if name not in ss:
            return []
        raw = np.asarray(ss[name].values, dtype=object)
        out = []
        for unit in _chain_units(raw, n_chains):
            vals = []
            for v in np.asarray(unit, dtype=object).ravel():
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    f = None
                if f is not None and not math.isfinite(f):
                    f = None
                if f is None and not keep_gaps:
                    continue
                vals.append(f)
            out.append(vals)
        return out

    beta = as_lists("beta", keep_gaps=False)
    return {
        "beta_schedule": beta,
        "accept_rate": as_lists("accept_rate", keep_gaps=False),
        "log_marginal_likelihood_per_stage": as_lists(
            "log_marginal_likelihood", keep_gaps=True),
        "n_stages": max((len(b) for b in beta), default=0),
        "sampler_attrs": {k: v for k, v in dict(ss.attrs).items()
                          if k in ("threshold", "correlation_threshold",
                                   "_n_tune", "_n_draws", "_t_sampling")},
        "pymc_version": pm.__version__,
    }


# ---------------------------------------------------------------- one SMC run

def run_smc(canonical, param_names, transforms, prior_rows, D, seed, *,
            draws=SMC_DRAWS, chains=SMC_CHAINS):
    """One SMC run. A failure is RECORDED, never silently dropped.

    Returns (info, phi) where phi is the (draws * chains, d) particle matrix, or
    (info_with_ok_False, None). The caller decides what a failure means; this
    function's contract is only that it never turns one into a missing model.

    `draws` and `chains` are KEYWORD-ONLY and default to the archived constants,
    so every existing call site and every equivalence test samples exactly what
    it sampled before. They are parameters so that a reduced numerical setting
    (for example a 200-draw test) can pass its own sample size: with a module
    constant, such a run would record 200 draws in its identity while drawing
    2000."""
    t0 = time.time()
    try:
        # structural first: the gates below compare two spellings of one model,
        # so they cannot see a disagreement both spellings implement
        check_inference_inputs(param_names, transforms, prior_rows)
        # both gates, on the production path, before anything is sampled.
        # Neither consumes global RNG state: validate_symbolic probes through
        # default_rng(13), validate_logp through default_rng(29), and
        # pm.sample_smc is seeded explicitly below - so the sampler sees the
        # identical seed path it saw when the gates were not here.
        sym_err = validate_symbolic(canonical, param_names, transforms,
                                    [float(x) for x, _ in D])
        model = build_model(canonical, param_names, transforms, prior_rows, D)
        logp_err = validate_logp(model, canonical, param_names, transforms,
                                 prior_rows, D)
        with model:
            idata = pm.sample_smc(draws=draws, chains=chains,
                                  random_seed=seed, progressbar=False,
                                  compute_convergence_checks=False)
        logz, per_chain = _extract_logz(idata)
        diag = _extract_diagnostics(idata)
        phi = np.asarray(idata.posterior["phi"].values, float)
        phi = phi.reshape((-1, phi.shape[-1]))
        if not np.all(np.isfinite(phi)):
            raise SMCError("non-finite particles")
        return {"ok": True, "seed": seed, "log_evidence": logz,
                "per_chain_log_evidence": per_chain,
                "smc_diagnostics": diag,
                "n_particles": int(phi.shape[0]),
                "symbolic_check_max_abs_error": sym_err,
                "logp_check_max_abs_error": logp_err,
                "seconds": round(time.time() - t0, 2)}, phi
    except Exception as e:                                    # noqa: BLE001
        return {"ok": False, "seed": seed,
                "failure_reason": f"{type(e).__name__}: {e}",
                "seconds": round(time.time() - t0, 2)}, None


# ------------------------------------------------------------------ summaries

def theta_summary(phi, transforms, param_names):
    """Posterior quantiles in THETA space, per parameter.

    The transform vocabulary is checked FIRST, before a single number is
    produced. This is the last phi -> theta map on the path and it was the one
    still written as a bare `exp(phi) if t == "log" else phi`: reached with an
    unregistered transform it would have summarised the phi column as though it
    were theta and returned a perfectly ordinary-looking median and interval.
    `run_smc` refuses such a request upstream, but this function can also be
    called directly - on already-drawn particles, or by anything
    post-processing a stored particle file - so the refusal has to be its own,
    not one inherited from a caller that may not be there."""
    for i, t in enumerate(transforms):
        if t not in TRANSFORMS:
            raise SMCError(
                f"unknown transform {t!r} at position {i} "
                f"({list(param_names)[i]!r}): the registered vocabulary is "
                f"{sorted(TRANSFORMS)} — summarising phi under an unregistered "
                "transform would report the untransformed column as theta")
    theta = np.empty_like(phi)
    for j, t in enumerate(transforms):
        theta[:, j] = np.exp(phi[:, j]) if t == "log" else phi[:, j]
    out = {}
    for j, nm in enumerate(param_names):
        col = theta[:, j]
        q = np.percentile(col, [2.5, 16.0, 50.0, 84.0, 97.5])
        out[nm] = {"median": _r(q[2]), "ci68": [_r(q[1]), _r(q[3])],
                   "ci95": [_r(q[0]), _r(q[4])]}
    return out
