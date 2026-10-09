"""The count objective: log Binomial(y; N, clip(yhat/N)) summed over a batch.

This is the ONE likelihood every layer of the system is scored against, so it
lives on its own rather than inside any backend. The Laplace member evaluates
it per observation, the PyMC SMC model reproduces it symbolically, and the
logp gate in `inference.smc` compares the two against THIS function - so if it
ever drifts, the gate fires instead of the two backends silently agreeing on a
different likelihood.

The clip is not incidental. `clip(yhat, 0, N) / N` then `clip(., 1e-9, 1-1e-9)`
is what makes an out-of-range predicted mean a finite (bad) score instead of a
domain error, and it is the reason a flat plateau exists at theta = 0 for
identity-role parameters - the behaviour the Laplace backend's deterministic
multi-start exists to escape. A non-finite predicted mean anywhere returns
-inf for the WHOLE batch, never a partial sum over the finite points.

Migration note (analysis/phase1/mp_refit.py -> inference/scoring.py)
-------------------------------------------------------------------
SPLIT, not a move: only `count_loglik` was extracted, byte-for-byte in its
arithmetic. The rest of mp_refit.py - the Phase-1 raw-vs-refit machinery
(`score_raw_refit`, `primary_pair`, `secondary_pair`, `_pred_errors`), the
constant-classification and substitution layer (`classify_constants`,
`substitute_constants`) and the Phase-1 validation harness - stays in the
archive and is not on this package's path.
"""
import numpy as np
from scipy.stats import binom

from ..numerics import EPS_P


def count_loglik(fn, params, D, N):
    """The evaluator's count objective, evaluated directly.

    `fn` is a compiled form from `models.expressions.compile_form`, `params`
    the parameter vector in THETA space (already transformed), D a list of
    (x, y) pairs and N the number of trials. Equivalence to the stored
    evaluator scores was asserted (<= 1e-6 over all 92 finite stored candidate
    scores) by the Phase-1 test harness before anything else.
    """
    xs = np.array([d[0] for d in D], float)
    ys = np.array([d[1] for d in D], float)
    with np.errstate(all="ignore"):
        yhat = fn(xs, params)
    if not np.all(np.isfinite(yhat)):
        return -np.inf
    p = np.clip(np.clip(yhat, 0.0, float(N)) / N, EPS_P, 1.0 - EPS_P)
    s = binom.logpmf(ys.astype(int), N, p).sum()
    return float(s) if np.isfinite(s) else -np.inf
