"""FB-1 (ForceBench) core: candidate grammar, tempered sequential Monte Carlo (SMC), joint expected information
gain (EIG), scoring.

The design every run follows:
  - The DiscoverPhysics (DP) executor is the true world (observations and noiseless true values only).  predict() = our jax integrator.
  - sigma_world = 0.05*sqrt(Var_GT)
  - fixed prior rules: signed N(0,1) · positive LogNormal(0,1) · sigma HalfNormal(1)
        SMC 500 x 2 chains · complexity: coefficients<=6, operators<=15 · 12 active
"""
import math, time
from dataclasses import dataclass, field
from typing import Callable
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.scipy.special import logsumexp, gammaln
import fb1_forward as fwd

DT = 0.02                 # dt rationale: on prior draws of the 8 structures, max|dlogL| vs dt=0.005 is <= 1.2e-3 nats (measured)
N_PARTICLES = 500         # per chain
N_CHAINS = 2
ESS_TARGET = 0.5
MH_STEPS = 3
MAX_ACTIVE = 12
MAX_COEFFS, MAX_OPS = 6, 15
# ---------------- candidate grammar (all of it ours) ----------------
# param kind: "pos" = LogNormal(0,1) [u~N(0,1), x=exp u] · "sig" = signed N(0,1)
@dataclass(frozen=True)
class Structure:
    """One candidate force structure: name, formula text, force function, free-constant kinds and complexity counts."""
    name: str
    expr: str
    force: Callable
    kinds: tuple          # kind of each free constant of the structure
    n_coeff: int
    n_ops: int

STRUCTURES = {
    "inv_r2":      Structure("inv_r2",  "F = c*p1/r^2",              fwd.f_inv_r2,      ("pos",),        1, 3),
    "linear":      Structure("linear",  "F = c*p1*r",                fwd.f_linear,      ("pos",),        1, 2),
    "const":       Structure("const",   "F = c*p1",                  fwd.f_const,       ("pos",),        1, 1),
    "screened":    Structure("screened","F = c*p1*exp(-r/lam)/r",    fwd.f_screened,    ("pos", "pos"),  2, 6),
    "inv_r":       Structure("inv_r",   "F = c*p1/r",                fwd.f_inv_r,       ("pos",),        1, 2),
    "power":       Structure("power",   "F = c*p1*r^(-a)",           fwd.f_power,       ("pos", "sig"),  2, 4),
    "inv_r_plus_c":Structure("inv_r_plus_c","F = c*p1/r + b*p1",     fwd.f_inv_r_plus_c,("pos", "pos"),  2, 5),
    "logr":        Structure("logr",    "F = c*p1*log1p(k*r)/r",     fwd.f_log,         ("pos", "pos"),  2, 7),
    "inv_r_p1sq":  Structure("inv_r_p1sq",  "F = c*p1^2/r",          fwd.f_inv_r_p1sq,  ("pos",),        1, 3),
    "inv_r_nomass":Structure("inv_r_nomass","F = c*p1*p2/r",         fwd.f_inv_r_nomass,("pos",),        1, 3),
    "inv_r_bg":    Structure("inv_r_bg","F = c*p1/r - b*p1*r",        fwd.f_inv_r_bg,    ("pos", "pos_bg"),  2, 5),
    "inv_r_tmod":  Structure("inv_r_tmod","F = c*p1/r * (1 + A1 cos(wt) + A2 sin(wt))",
                             fwd.f_inv_r_tmod, ("pos", "sig", "sig", "pos_w"), 4, 8),
    "prod_coupling": Structure("prod_coupling", "a = c*p1*p2/r^k  (F = c*p1*p2^2/r^k)",
                               fwd.f_prod_coupling, ("pos", "pos"), 2, 5),
    "crossover":   Structure("crossover", "F = c1*p1/r + c2*p1*exp(-r/R)/r^2",
                             fwd.f_crossover, ("pos", "pos", "pos"), 3, 8),
}
FAMILIES = {"gauss": ("sig_hn",), "student_t4": ("sig_hn",)}   # one noise free constant, HalfNormal(1)

@dataclass
class Candidate:
    """A candidate model: a structure name, a noise family name and the step at which it entered the pool."""
    struct: str
    family: str
    born_step: int
    def key(self):
        """Return the (structure, family) tuple that identifies the candidate."""
        return (self.struct, self.family)
    def label(self):
        """Return the display label "structure|family"."""
        return f"{self.struct}|{self.family}"
    def kinds(self):
        """Return the kinds of all free constants: the structure's, then the noise family's."""
        return STRUCTURES[self.struct].kinds + FAMILIES[self.family]
    def dim(self):
        """Return the number of free constants (the dimension of u)."""
        return len(self.kinds())

# ---------------- prior (u space) ----------------
_LOG_HN = math.log(math.sqrt(2.0 / math.pi))
# Scale taken from public metadata.  domain_size=20.0, periodic_boundaries=True (executor.py:262)
# -> half-width R=10, natural scale of the neutralizing background b0 = 1/(2*pi*R^2).  prior = LogNormal(log b0, 1).
B0_BACKGROUND = 1.0 / (2.0 * math.pi * 10.0**2)
# Angular-frequency scale of the time modulation.  Measurement times 0.5..5.0 → time span 5.0 (public metadata) → w0 = 2*pi/5.
W0_MODULATION = 2.0 * math.pi / 5.0
def log_prior_u(u, kinds):
    """Log prior density of u: (P, d), including the Jacobian of each kind's transform; returns (P,)."""
    tot = jnp.zeros(u.shape[0])
    for j, k in enumerate(kinds):
        uj = u[:, j]
        if k in ("pos", "sig", "pos_bg", "pos_w"):  # LogNormal(scale,1) in log space = N(0,1) · signed N(0,1)
            tot = tot - 0.5 * uj**2 - 0.5 * math.log(2 * math.pi)
        elif k == "sig_hn":                # x=exp(u), x~HalfNormal(1)
            tot = tot + _LOG_HN - 0.5 * jnp.exp(2 * uj) + uj
        else:
            raise ValueError(k)
    return tot

def u_to_x(u, kinds):
    """Map unconstrained u (P, d) to free-constant values x (P, d) by each column's kind."""
    cols = []
    for j, k in enumerate(kinds):
        if k == "sig":      cols.append(u[:, j])
        elif k == "pos_bg": cols.append(B0_BACKGROUND * jnp.exp(u[:, j]))
        elif k == "pos_w":  cols.append(W0_MODULATION * jnp.exp(u[:, j]))
        else:               cols.append(jnp.exp(u[:, j]))
    return jnp.stack(cols, axis=1)

def sample_prior_u(key, n, kinds):
    """Per-kind prior sampler.  Only `sig_hn` columns (x=exp(u)~HalfNormal(1)) become u = log(max(|z|, tiny)),
    z~N(0,1); every other kind's column is array_equal to the output of the same `jax.random.normal` call on this
    `key` (before an earlier correction the `sig_hn` column was also left as z -- LogNormal(0,1) instead of the
    declared HalfNormal(1))."""
    u = jax.random.normal(key, (n, len(kinds)))
    for j, k in enumerate(kinds):
        if k == "sig_hn":
            z = u[:, j]
            u = u.at[:, j].set(jnp.log(jnp.maximum(jnp.abs(z), jnp.finfo(z.dtype).tiny)))
    return u

# ---------------- likelihood ----------------
def _log_t4(z):
    nu = 4.0
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(nu * math.pi)
            - (nu + 1) / 2 * jnp.log1p(z**2 / nu))

def make_loglik(cand):
    """(u, p1, p2, y) -> (P,).  y: (D,T,2)  p1,p2: (D,)"""
    st = STRUCTURES[cand.struct]
    kinds, k_struct, fam = cand.kinds(), len(STRUCTURES[cand.struct].kinds), cand.family
    def loglik(u, p1, p2, y, pos0, vel0, times):
        """Log likelihood of the observations y under each particle u; returns (P,)."""
        x = u_to_x(u, kinds)
        mu = fwd.integrate(st.force, x[:, :k_struct], p1, p2, pos0, vel0, times, DT)  # (D,P,T,2)
        s = x[:, k_struct][None, :, None, None]
        z = (y[:, None, :, :] - mu) / s
        ll = -0.5 * z**2 - 0.5 * math.log(2 * math.pi) - jnp.log(s) if fam == "gauss" \
             else _log_t4(z) - jnp.log(s)
        tot = ll.sum(axis=(0, 2, 3))
        return jnp.where(jnp.isfinite(tot), tot, -1e30)   # guard against diverging trajectories
    return loglik

def make_predict(cand):
    """Build predict(u, p1, p2, pos0, vel0, times) -> noiseless trajectories (D,P,T,2) for one candidate."""
    st, kinds, k = STRUCTURES[cand.struct], cand.kinds(), len(STRUCTURES[cand.struct].kinds)
    def predict(u, p1, p2, pos0, vel0, times):
        """Integrate each particle's force law at the designs; returns (D,P,T,2)."""
        x = u_to_x(u, kinds)
        return fwd.integrate(st.force, x[:, :k], p1, p2, pos0, vel0, times, DT)
    return predict

def noise_scale(u, cand):
    """Return each particle's noise scale (P,): exp of the u column after the structure's constants."""
    return jnp.exp(u[:, len(STRUCTURES[cand.struct].kinds)])

# ---------------- tempered SMC ----------------
def _systematic(key, w, n):
    u0 = jax.random.uniform(key)
    pos = (u0 + jnp.arange(n)) / n
    return jnp.searchsorted(jnp.cumsum(w), pos)

def _ess(logw):
    w = jnp.exp(logw - logsumexp(logw))
    return 1.0 / jnp.sum(w**2)

def smc_chain(key, cand, loglik_fn, n=None, mh_steps=None):
    """Single-chain tempered SMC.  Returns (u, logZ, diagnostics)."""
    n = N_PARTICLES if n is None else n
    mh_steps = MH_STEPS if mh_steps is None else mh_steps
    kinds = cand.kinds()
    key, k0 = jax.random.split(key)
    u = sample_prior_u(k0, n, kinds)
    ll = loglik_fn(u)
    lam, logZ, stages, accs = 0.0, 0.0, 0, []
    while lam < 1.0:
        lo, hi = 0.0, 1.0 - lam                      # bisection for the next temperature increment (ESS = 0.5n)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if _ess(mid * ll) < ESS_TARGET * n: hi = mid
            else: lo = mid
        d = max(lo, 1e-6)
        if lam + d > 1.0: d = 1.0 - lam
        logw = d * ll
        logZ += float(logsumexp(logw) - math.log(n))
        w = jnp.exp(logw - logsumexp(logw))
        lam += d; stages += 1
        key, kr = jax.random.split(key)
        idx = _systematic(kr, w, n)
        u, ll = u[idx], ll[idx]
        cov = jnp.cov(u.T).reshape(u.shape[1], u.shape[1]) + 1e-8 * jnp.eye(u.shape[1])
        L = jnp.linalg.cholesky(cov) * (2.38 / math.sqrt(u.shape[1]))
        lp = log_prior_u(u, kinds)
        acc_tot = 0.0
        for _ in range(mh_steps):
            key, kp, ka = jax.random.split(key, 3)
            prop = u + jax.random.normal(kp, u.shape) @ L.T
            ll_p, lp_p = loglik_fn(prop), log_prior_u(prop, kinds)
            logr = (lp_p + lam * ll_p) - (lp + lam * ll)
            acc = jnp.log(jax.random.uniform(ka, (n,))) < logr
            u = jnp.where(acc[:, None], prop, u); ll = jnp.where(acc, ll_p, ll); lp = jnp.where(acc, lp_p, lp)
            acc_tot += float(acc.mean())
        accs.append(acc_tot / mh_steps)
    return u, logZ, {"stages": stages, "mean_accept": float(np.mean(accs)) if accs else None}

def smc_fit(key, cand, loglik_fn, n=None, chains=None, mh_steps=None):
    """Run `chains` independent SMC chains; returns pooled particles u, the chain-averaged logZ and diagnostics."""
    n = N_PARTICLES if n is None else n; chains = N_CHAINS if chains is None else chains
    us, logZs, diag = [], [], []
    for c in range(chains):
        key, kc = jax.random.split(key)
        u, lz, d = smc_chain(kc, cand, loglik_fn, n, mh_steps)
        us.append(u); logZs.append(lz); diag.append(d)
    u = jnp.concatenate(us, axis=0)
    logZ = float(logsumexp(jnp.array(logZs)) - math.log(chains))     # chain average (in log space)
    return u, logZ, {"logZ_per_chain": [float(z) for z in logZs],
                     "logZ_chain_spread": float(max(logZs) - min(logZs)),
                     "stages": [d["stages"] for d in diag],
                     "mean_accept": [d["mean_accept"] for d in diag]}
