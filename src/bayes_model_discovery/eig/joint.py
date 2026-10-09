"""Joint expected information gain over structure AND parameters (§9).

Registered definition (NOT point disagreement):

    I((M, Phi); Y_x | H_t)
      = H[ p(Y_x | H_t) ]
        - sum_m w_m E_{Phi_m | H_t, M_m} H[ p(Y_x | Phi_m, M_m) ]

Death-process observations are discrete counts 0..50, so both entropies are
computed by EXHAUSTIVE summation over y - no sampling of y, no plug-in.

Numerical discipline (registered before the run):
  * one Gauss-Hermite node bank per model, built ONCE from that model's
    posterior and REUSED for every design (common random numbers across
    designs, so design comparisons are not corrupted by resampling noise);
  * every design is scored with the identical rule;
  * a non-finite value anywhere HALTS the computation explicitly - nothing is
    silently dropped;
  * ties are broken by the SMALLEST design value (fixed here, before any run).

Notes from porting this module from the earlier prototype
---------------------------------------------------------
The initial package migration made four non-numerical changes:

1. The `sys.path.insert` block that made the flat prototype directory
   importable is gone; this is a package now.

2. `GH_NODES` is renamed `EIG_GH_NODES`. The VALUE IS UNCHANGED (7). The old
   name collides with an identically-named constant in the Laplace backend,
   which is a different quantity - the Laplace module's node count controls a
   diagnostic quadrature that never decides anything, whereas this one is on
   the decision path. Two unrelated numbers under one name in one package is
   how a diagnostic knob silently ends up steering a design choice, so the
   collision is resolved here, before the Laplace module is migrated, rather
   than by whoever gets to that module second.

3. `compile_form` is imported lazily inside `ModelEIGView.__init__` instead of
   at module level. It originally pointed at the unmigrated `model_proposal.py`
   and raised a clear error when the Gauss-Hermite view was constructed; the
   form compiler has since arrived as `models.expressions`, so the lazy import
   now resolves and the Gauss-Hermite path works. The import stays function-
   local rather than moving to module scope: the particle path (`eig.views`)
   never touches `compile_form`, and keeping it deferred means `eig.views` does
   not acquire a scipy-optimize import through `eig.joint` merely to score
   designs from particles it already has.

4. HARDENING, not a transcription: `ModelEIGView.__init__` refuses a transform
   outside `models.schema.TRANSFORMS`. The theta line it guards is unchanged;
   what changes is that an unregistered transform now stops the view instead of
   being applied as the identity by the `else` branch and scoring every design
   against a posterior read in the wrong parameterization. No registered
   transform reaches a different result, so the archived numbers are untouched.

P1 performance migration
------------------------
The decision rule is unchanged, but two implementations are deliberately no
longer byte-for-byte copies of the archive. Row entropies are vectorized over
outcomes and agree bit-exactly with the scalar loop. Binomial probabilities are
evaluated from the closed-form log mass instead of ``scipy.stats.binom.pmf``;
the declared reference bound is ``BINOM_PMF_REFERENCE_ATOL``. G25 and G200
tests compare the complete EIG and decomposition tables, including a synthetic
tie-boundary case, before accepting either change. ``EPS_P`` now comes from the
dependency-free :mod:`bayes_model_discovery.numerics` module shared by every
Likelihood and EIG backend.
"""
import math

import numpy as np
from scipy.special import gammaln

from ..models.schema import TRANSFORMS
from ..numerics import EPS_P

EIG_GH_NODES = 7             # per dimension; 7^3 = 343 nodes for a 3-param model
TIE_RULE = "smallest_design_value"
# One tolerance for every production argmax.  It is also reconciled against
# NumericalSpec before a run starts; a spec may not record one boundary while
# the implementation applies another.
TIE_ATOL = 1e-12
# Tolerance for the chain-rule decomposition: both parts must be non-negative
# and must sum to the total to within this. Entropies here are O(1) nats summed
# over 51 outcomes, so the accumulated rounding is many orders below 1e-9.
DECOMP_TOL = 1e-9
# Predeclared validation bound against scipy.stats.binom.pmf. The production
# observation model uses n_trials=50; tests also cover 1, 2, 7, and 101 over
# the registered open probability interval. This bound is a comparison
# contract, not a tolerance used to modify or clamp production values.
BINOM_PMF_REFERENCE_ATOL = 1e-13


class EIGError(RuntimeError):
    """A non-finite quantity appeared - the computation halts, never drops."""


def _gh_bank(mean, cov, n_nodes=EIG_GH_NODES):
    """Deterministic Gauss-Hermite node bank for one model's Gaussian
    posterior in phi-space. Built once, reused for every design."""
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / math.sqrt(2.0 * math.pi)
    d = len(mean)
    grids = np.meshgrid(*([x] * d), indexing="ij")
    Z = np.stack([g.ravel() for g in grids], axis=1)
    wgrids = np.meshgrid(*([w] * d), indexing="ij")
    W = np.prod(np.stack([g.ravel() for g in wgrids], axis=1), axis=1)
    L = np.linalg.cholesky(0.5 * (np.asarray(cov) + np.asarray(cov).T)
                           + 1e-10 * np.eye(d))
    phis = np.asarray(mean)[None, :] + Z @ L.T
    W = W / W.sum()                      # normalise the quadrature weights
    return phis, W


class ModelEIGView:
    """Everything the EIG needs from one active model."""

    def __init__(self, model_id, canonical, transforms, mean, cov, weight,
                 n_trials=50, n_nodes=EIG_GH_NODES):
        # Deferred until construction: the particle path in `eig.views` does not
        # need the form compiler and should not pull it in. See migration note 3.
        from ..models.expressions import compile_form
        self.model_id = model_id
        self.canonical = canonical
        self.transforms = list(transforms)
        self.weight = float(weight)
        self.n_trials = int(n_trials)
        self._fn, names = compile_form(canonical)
        if len(names) != len(self.transforms):
            raise EIGError(f"{model_id}: arity mismatch")
        # The theta line below is `math.exp(v) if t == "log" else v`, so a transform
        # outside the registered vocabulary is applied as the identity and this view
        # scores designs against a posterior it has read wrongly. Refused at
        # construction, against the one declaration in models.schema.
        for i, t in enumerate(self.transforms):
            if t not in TRANSFORMS:
                raise EIGError(
                    f"{model_id}: unknown transform {t!r} at position {i} — the "
                    f"registered vocabulary is {sorted(TRANSFORMS)}, and anything "
                    "else is applied as the identity by the phi->theta map here")
        self.phis, self.node_w = _gh_bank(mean, cov, n_nodes)
        self.thetas = [[math.exp(v) if t == "log" else v
                        for v, t in zip(phi, self.transforms)]
                       for phi in self.phis]

    def success_probs(self, x):
        """p(x) per posterior node, clipped exactly as the likelihood does."""
        out = np.empty(len(self.thetas))
        for i, th in enumerate(self.thetas):
            try:
                v = float(self._fn(x, th))
            except Exception as e:                            # noqa: BLE001
                raise EIGError(f"{self.model_id}: forward failed at x={x}: {e}")
            if not math.isfinite(v):
                raise EIGError(f"{self.model_id}: non-finite mean at x={x}")
            out[i] = min(max(v, 0.0), float(self.n_trials)) / self.n_trials
        return np.clip(out, EPS_P, 1 - EPS_P)


def _entropy(p):
    p = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(p)):
        raise EIGError("non-finite pmf")
    q = np.clip(p, 1e-300, None)
    return float(-np.sum(p * np.log(q)))


def _row_entropies(pmf, model_id, x):
    """Return ``H[row / row.sum()]`` for every row in one reduction.

    ``pmf`` is C-contiguous and the reduction stays on the last axis, so each
    row is summed in the same order as the former one-dimensional call to
    :func:`_entropy`. Rows never share an accumulator. This removes one Python
    call per posterior node without changing the floating-point operation
    order within a row.

    The finiteness check deliberately remains after normalisation, where the
    scalar implementation performed it. A finite all-zero row becomes NaN at
    this point and must still halt rather than silently contribute an entropy.
    """
    q = np.asarray(pmf, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = q / q.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(q)):
        raise EIGError(f"{model_id}: non-finite pmf at x={x}")
    return -np.sum(q * np.log(np.clip(q, 1e-300, None)), axis=1)


_LOG_BINOMIAL_COEFFICIENTS = {}


def _log_binomial_coefficients(n_trials):
    """Return read-only ``log(C(n, y))`` for ``y=0..n``, cached by ``n``.

    The coefficients depend only on ``n_trials``. Keeping a mutable cached
    array would let one caller corrupt every later design, so the cached value
    is immutable and tests exercise different request orders.
    """
    if isinstance(n_trials, bool) or not isinstance(n_trials, (int, np.integer)):
        raise EIGError(f"n_trials must be a non-negative integer, got {n_trials!r}")
    n_trials = int(n_trials)
    if n_trials < 0:
        raise EIGError(f"n_trials must be a non-negative integer, got {n_trials!r}")
    if n_trials not in _LOG_BINOMIAL_COEFFICIENTS:
        ys = np.arange(n_trials + 1)
        values = (
            gammaln(n_trials + 1)
            - gammaln(ys + 1)
            - gammaln(n_trials - ys + 1)
        )
        values.setflags(write=False)
        _LOG_BINOMIAL_COEFFICIENTS[n_trials] = values
    return _LOG_BINOMIAL_COEFFICIENTS[n_trials]


def _binomial_pmf_rows(success_probabilities, n_trials, model_id, x):
    """Evaluate one Binomial PMF row per posterior node.

    The closed form is valid only for finite probabilities strictly inside the
    unit interval. Both production views enforce that interval with ``EPS_P``;
    this second guard keeps the kernel fail-closed if a new view bypasses the
    clipping contract. Finiteness is checked first because comparisons do not
    reject NaN.
    """
    probabilities = np.asarray(success_probabilities, dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise EIGError(
            f"{model_id}: success probabilities must be a non-empty vector at x={x}"
        )
    if not np.all(np.isfinite(probabilities)):
        raise EIGError(f"{model_id}: non-finite success probability at x={x}")
    minimum = float(probabilities.min())
    maximum = float(probabilities.max())
    if minimum <= 0.0 or maximum >= 1.0:
        raise EIGError(
            f"{model_id}: success probability outside (0,1) at x={x}: "
            f"min {minimum!r} max {maximum!r}"
        )

    ys = np.arange(n_trials + 1)
    probabilities = probabilities[:, None]
    log_pmf = (
        _log_binomial_coefficients(n_trials)[None, :]
        + ys[None, :] * np.log(probabilities)
        + (n_trials - ys)[None, :] * np.log1p(-probabilities)
    )
    pmf = np.exp(log_pmf)
    if not np.all(np.isfinite(pmf)):
        raise EIGError(f"{model_id}: non-finite pmf at x={x}")
    return pmf


def binomial_pmf_rows(success_probabilities, n_trials, model_id="predictive", x=None):
    """Public guarded Binomial PMF kernel for offline posterior prediction.

    EIG and evaluator-side predictive metrics must use one probability kernel.
    Keeping the guard here prevents an analyser from accepting boundary, NaN,
    or infinite probabilities that the acquisition path refuses.
    """
    return _binomial_pmf_rows(success_probabilities, n_trials, model_id, x)


def design_eig(views, x, n_trials=50):
    """Joint EIG at one design, plus its chain-rule decomposition.

    The decomposition is computed HERE, in the one function both the
    Gauss-Hermite path (`ModelEIGView`) and the particle path
    (`eig.views.ParticleEIGView` -> `eig.views.eig_table`) already call, so
    the total and its two parts can never drift apart into two numerical
    definitions of the same quantity:

        I((M,Phi); Y) = I(M; Y)  +  E_M[ I(Phi; Y | M) ]
          total          structure       parameter

    with, writing p(y|m) = E_{Phi|m} p(y|phi,m) for a model's posterior
    predictive,

        total     = H[p(y)]        - sum_m w_m E_{Phi|m} H[p(y|phi,m)]
        structure = H[p(y)]        - sum_m w_m H[p(y|m)]
        parameter = sum_m w_m H[p(y|m)] - sum_m w_m E_{Phi|m} H[p(y|phi,m)]

    `sum_m w_m H[p(y|m)]` is the only quantity that was not already being
    formed, and it needs NO extra sampling - `model_pred` is the per-model
    predictive that the mixture is built from anyway. So this adds one
    entropy per model per design and changes no existing number: `eig` is
    still `H_mix - exp_cond_H`, computed from the same accumulators in the
    same order, so the argmax and the tie rule are untouched.

    Both parts are non-negative in exact arithmetic (entropy is concave, so a
    mixture is at least as uncertain as its averaged components). A value
    below -`DECOMP_TOL` is a numerical failure and HALTS. Values inside the
    tolerance are returned RAW rather than clamped to zero: clamping would
    buy a cosmetically tidy sign at the cost of breaking the exact identity
    `total == structure + parameter`, which is the property the tests check.
    """
    ys = np.arange(n_trials + 1)
    mix = np.zeros(n_trials + 1)
    exp_cond_H = 0.0
    exp_model_pred_H = 0.0
    per_model = {}
    for v in views:
        ps = v.success_probs(x)
        # pmf per node: (n_nodes, n_trials+1)
        pmf = _binomial_pmf_rows(ps, n_trials, v.model_id, x)
        if not np.all(np.isfinite(pmf)):
            raise EIGError(f"{v.model_id}: non-finite pmf at x={x}")
        model_pred = v.node_w @ pmf                    # posterior predictive
        s = model_pred.sum()
        if not math.isfinite(s) or s <= 0:
            raise EIGError(f"{v.model_id}: degenerate predictive at x={x}")
        model_pred = model_pred / s
        node_H = _row_entropies(pmf, v.model_id, x)
        cond_H = float(v.node_w @ node_H)
        pred_H = _entropy(model_pred)          # H[p(y|m)]
        mix += v.weight * model_pred
        exp_cond_H += v.weight * cond_H
        exp_model_pred_H += v.weight * pred_H
        # per-model parameter information: how much this model alone expects
        # to learn about its OWN parameters here. Without it the pool total
        # cannot be attributed back to the model that earned it.
        per_model[v.model_id] = {"conditional_entropy": round(cond_H, 8),
                                 "predictive_entropy": round(pred_H, 8),
                                 "parameter_eig": round(pred_H - cond_H, 8),
                                 "predictive_mean": round(
                                     float(model_pred @ ys), 6)}
    tot = mix.sum()
    if not math.isfinite(tot) or tot <= 0:
        raise EIGError(f"degenerate mixture predictive at x={x}")
    mix = mix / tot
    H_mix = _entropy(mix)
    eig = H_mix - exp_cond_H
    if not math.isfinite(eig):
        raise EIGError(f"non-finite EIG at x={x}")
    structure = H_mix - exp_model_pred_H
    parameter = exp_model_pred_H - exp_cond_H
    for nm, val in (("structure_eig", structure), ("parameter_eig", parameter)):
        if not math.isfinite(val):
            raise EIGError(f"non-finite {nm} at x={x}")
        if val < -DECOMP_TOL:
            raise EIGError(
                f"{nm} is negative beyond tolerance at x={x}: {val!r}. "
                "Entropy concavity makes this impossible in exact arithmetic, "
                "so the quadrature or the weights are wrong. Halting rather "
                "than reporting a negative information quantity.")
    if abs(eig - (structure + parameter)) > DECOMP_TOL:
        raise EIGError(
            f"decomposition identity broken at x={x}: total {eig!r} != "
            f"structure {structure!r} + parameter {parameter!r}")
    return {"design": x, "eig": eig, "mixture_entropy": H_mix,
            "expected_conditional_entropy": exp_cond_H,
            "expected_model_predictive_entropy": exp_model_pred_H,
            "structure_eig": structure, "parameter_eig": parameter,
            "total_eig": eig,
            "mixture_predictive_mean": float(mix @ ys),
            "per_model": per_model}


def choose_design(rows, key="eig", *, tie_atol=TIE_ATOL):
    """The registered argmax + tie rule, applied to UNROUNDED objective values.

    Single authority for the selection decision: ``select_design``,
    ``views.eig_table`` and ``arms.select`` all route through here, so the
    executed tie window is ``tie_atol`` and never the storage quantum
    (``numerics.EIG_STORAGE_DECIMALS``). Callers must pass unrounded rows;
    storage rounding happens strictly after the decision.
    """
    if not rows:
        raise EIGError("empty design table")
    missing = [i for i, r in enumerate(rows) if key not in r or "design" not in r]
    if missing:
        raise EIGError(f"design table row {missing[0]} lacks {key!r} or 'design'")
    vals = [float(r[key]) for r in rows]
    if not all(math.isfinite(v) for v in vals):
        raise EIGError(f"{key}: a non-finite value is present. nothing is dropped silently")
    top = max(vals)
    # the archived window form, kept to the bit: abs(v - top) <= tie_atol
    tied = [r["design"] for r, v in zip(rows, vals, strict=True)
            if abs(v - top) <= tie_atol]
    return {
        "chosen_design": min(tied),        # the registered tie rule
        "n_tied_at_max": len(tied),
        "top_value": top,
        "tie_rule": TIE_RULE,
        "tie_atol": tie_atol,
    }


def select_design(views, design_grid, n_trials=50):
    """Score EVERY design with the same node banks, then argmax with the
    registered tie rule. Returns (chosen, full table)."""
    if not views:
        raise EIGError("empty active pool")
    w = sum(v.weight for v in views)
    if not math.isfinite(w) or abs(w - 1.0) > 1e-6:
        raise EIGError(f"model weights do not sum to 1 (got {w})")
    table = [design_eig(views, float(x), n_trials) for x in design_grid]
    # tie rule fixed before the run: smallest design value wins - the decision
    # IS choose_design's, not a local recomputation free to drift from it
    decision = choose_design(table, "eig")
    chosen = next(r for r in table if r["design"] == decision["chosen_design"])
    return {"chosen_design": chosen["design"], "chosen_eig": chosen["eig"],
            "n_tied_at_max": decision["n_tied_at_max"], "tie_rule": TIE_RULE,
            "gh_nodes_per_dim": EIG_GH_NODES,
            # unrounded decision inputs, same shape as views.eig_table emits -
            # a selection may lawfully be taken over these rows only
            "selection_rows": [{"design": r["design"], "eig": r["eig"],
                                "structure_eig": r["structure_eig"],
                                "parameter_eig": r["parameter_eig"]}
                               for r in table],
            # per_model is carried into the summary table as well: without it
            # the stored result answers "how informative was this design" but
            # not "which model made it informative", and that cannot be
            # reconstructed afterwards from the totals alone
            "table": [{"design": r["design"], "eig": round(r["eig"], 8),
                       "structure_eig": round(r["structure_eig"], 8),
                       "parameter_eig": round(r["parameter_eig"], 8),
                       "mixture_entropy": round(r["mixture_entropy"], 8),
                       "expected_conditional_entropy": round(
                           r["expected_conditional_entropy"], 8),
                       "expected_model_predictive_entropy": round(
                           r["expected_model_predictive_entropy"], 8),
                       "mixture_predictive_mean": round(
                           r["mixture_predictive_mean"], 6),
                       "per_model": r["per_model"]}
                      for r in table]}, table
