"""ForceBench design variant: resolution floor.

Declared in advance (decided on 2026-09-07; not a tuning on results; basis = the force-curve diagnosis):
  among the design candidates, those whose **predicted trajectory under the current model (Bayesian model
  averaging, BMA)** comes closer than two grid cells (`min_t r(t) < 2*dx = 0.625`) are excluded **before** the
  expected information gain (EIG) computation, and the argmax is taken over the remaining candidates only.

  dx = domain_size / grid = 20.0 / 64 = 0.3125.
  The only basis is the executor's public metadata: the default arguments grid_size=(64, 64),
  domain_size=20.0 of SimulationExecutor in scienceagent/executor.py (the field-engine worlds gravity,
  yukawa and fractional all use this class as is).  The true simulator's output does **not** enter this
  path: predictions are made only from our forward model (fb1_forward) and the current posterior particles.

Scope (declared in advance):
  * field worlds only.  The nbody worlds have no periodic box or grid, so no resolution floor is defined.
  * active arms only.  The random arm uses cands[0] as is and never calls this module.
  * if every candidate is excluded, the candidate with the largest predicted min r is chosen and recorded
    as a fallback.
"""
import sys
import numpy as np, jax, jax.numpy as jnp

# Executor public metadata (default arguments of SimulationExecutor.__init__, scienceagent/executor.py:257-262)
DOMAIN_SIZE = 20.0
GRID = 64
DX = DOMAIN_SIZE / GRID              # 0.3125
FLOOR = 2.0 * DX                     # 0.625 - two grid cells
N_PARTICLES_SUBSAMPLE = 64           # particle subsample size per candidate (declared in advance)


def _runner():
    """Return the runner module that is already loaded.

    Run directly as `python run_gravity.py`, the runner is registered as sys.modules["__main__"].
    An `import run_gravity` in that state loads the same file **a second time**, with its own
    _PR_CACHE, TIMERS and IC; the 24-cell run starts exactly that way, so the loaded module is looked up first.
    """
    m = sys.modules.get("run_gravity")
    if m is not None:
        return m
    m = sys.modules.get("__main__")
    if m is not None and str(getattr(m, "__file__", "")).endswith("run_gravity.py"):
        return m
    import run_gravity as R            # neither (unit tests etc.): a regular import
    return R


def predicted_min_r(fits, weights, cands, key, n_particles=N_PARTICLES_SUBSAMPLE):
    """min_t ||x_t|| of the **BMA predictive mean trajectory** per design.  Returns a (D,) numpy array.

    fits    : [(Candidate, u)] - u is the (P, k) posterior particles
    weights : (C,) posterior weights of the candidates
    cands   : (D, 2) design candidates (p1, p2)
    key     : jax PRNGKey - for the per-candidate particle subsample

    From each candidate, n_particles particles are drawn at random and run_gravity.predict_for(cand, D) gives
    the (D, n, T, 2) predictions; they are averaged over particles, then mixed by the candidate weights.
    The true values are not used.
    """
    R = _runner()                    # avoids a circular import and a second load of the runner
    cands = np.asarray(cands, float)
    D = cands.shape[0]
    p1d, p2d = jnp.array(cands[:, 0]), jnp.array(cands[:, 1])
    w = np.asarray(weights, float)
    pred_mean = None
    for i, (cand, u) in enumerate(fits):
        key, ks = jax.random.split(key)
        P = int(u.shape[0])
        k = min(int(n_particles), P)
        idx = jax.random.choice(ks, P, (k,), replace=False)
        mu = R.predict_for(cand, D)(u[idx], p1d, p2d)        # (D, k, T, 2)
        m = mu.mean(axis=1)                                   # (D, T, 2)
        pred_mean = w[i] * m if pred_mean is None else pred_mean + w[i] * m
    r = jnp.linalg.norm(pred_mean, axis=-1)                   # (D, T)
    return np.asarray(jnp.min(r, axis=1))                     # (D,)


def resolution_floor_filter(min_r, floor=FLOOR):
    """Resolution floor filter.  Returns (mask (D,) bool, n_excluded int, fallback bool).

    Only candidates with min_r >= floor are kept.  If all are excluded (mask all False), **only the one**
    candidate with the largest predicted min r is kept and fallback=True is returned.
    n_excluded is the number of candidates caught by the floor - unchanged even under fallback.
    """
    min_r = np.asarray(min_r, float)
    mask = min_r >= floor
    n_excluded = int((~mask).sum())
    fallback = not bool(mask.any())
    if fallback:
        mask = np.zeros(min_r.shape, bool)
        mask[int(np.argmax(min_r))] = True
    return mask, n_excluded, fallback
