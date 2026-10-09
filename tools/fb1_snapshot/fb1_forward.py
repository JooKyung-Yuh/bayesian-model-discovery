"""FB-1 (ForceBench) forward model: our jax integrator (not the true world).

The DiscoverPhysics (DP) executor is the true world; proposed force laws are integrated here, by us.
State layout = (design D, particle P, 2).  Particle 1 is fixed at the origin (pos1 variation measured as 0.000e+00 in
appendix B).
"""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from functools import partial
import math
import numpy as np

# Effective get_world/executor defaults at DP commit 33b7fa9, not the raw
# NBodySampler defaults (leapfrog/.01). World kwargs and runner requests do not
# override these settings. Matching time stepping does not change force physics.
TRUTH_COMMIT = "33b7fa9df96de9c35744efd181ca7e5a8dd60ad5"
TRUTH_NUMERICS = {
    "gravity": dict(engine="field", method="symplectic_euler", dt=.005,
        executor="SimulationExecutor", source="ScienceAgent/scienceagent/executor.py:257-263"),
    "yukawa": dict(engine="field", method="symplectic_euler", dt=.005,
        executor="SimulationExecutor", source="ScienceAgent/scienceagent/executor.py:257-263"),
    "coulomb_easy": dict(engine="nbody", method="yoshida4", dt=.005,
        executor="NBodyCoulombEasyExecutor", source="ScienceAgent/scienceagent/executor.py:2440-2449"),
    "oscillator": dict(engine="nbody", method="yoshida4", dt=.005,
        executor="NBodyOscillatorExecutor", source="ScienceAgent/scienceagent/executor.py:2146-2155"),
    "extra_dimensions": dict(engine="nbody", method="yoshida4", dt=.005,
        executor="NBodyExtraDimensionsExecutor", source="ScienceAgent/scienceagent/executor.py:2321-2330"),
}
INTEGRATION_METHOD = "leapfrog"
ALIGN_MEASUREMENT_TIMES = False
# Optional force softening in the candidate forward model (added 2026-10-02), default 0 = byte-identical. The
# simulator's nbody executors evaluate the kernel at r_eff = sqrt(r^2 + eps^2) with eps 0.05
# (PhysicsSchool nbody_sampler.py:95-101, 128-141; ScienceAgent executor.py:1085 _NBODY_SOFTENING_DEFAULT) and keep the
# direction x/r; the field executors have no softening. FB1_SOFTENING = 0 | world | <number>.
TRUTH_SOFTENING = {"gravity": 0.0, "yukawa": 0.0, "coulomb_easy": 0.05, "oscillator": 0.05, "extra_dimensions": 0.05}
FORCE_SOFTENING = 0.0
# RMS time-discretization error <= 10% of observation SD contributes <= 1%
# of its variance. This world-independent heuristic is not a convergence proof.
DESIGN_GATE_TAU = 0.1


def numerical_config(world, environ):
    """Resolve explicit options without consulting observations or held-out truth."""
    mode = environ.get("FB1_INTEGRATOR", "legacy")
    if mode not in ("legacy", "match_truth"):
        raise ValueError("FB1_INTEGRATOR must be legacy or match_truth")
    truth = TRUTH_NUMERICS[world]
    method, dt = ("leapfrog", .02) if mode == "legacy" else (truth["method"], truth["dt"])
    if "FB1_DT" in environ:
        try:
            dt = float(environ["FB1_DT"])
        except (TypeError, ValueError) as exc:
            raise ValueError("FB1_DT must be a finite positive number") from exc
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("FB1_DT must be a finite positive number")
    gate = environ.get("FB1_DESIGN_GATE", "off")
    if gate not in ("off", "dt_halving"):
        raise ValueError("FB1_DESIGN_GATE must be off or dt_halving")
    fallback = environ.get("FB1_DESIGN_GATE_FALLBACK", "off")
    if fallback not in ("off", "min_delta_k"):
        raise ValueError("FB1_DESIGN_GATE_FALLBACK must be off or min_delta_k")
    raw_k = environ.get("FB1_DESIGN_GATE_FALLBACK_K", "8")
    if (not isinstance(raw_k, str) or not raw_k.isascii()
            or not raw_k.isdecimal() or int(raw_k) <= 0):
        raise ValueError("FB1_DESIGN_GATE_FALLBACK_K must be a positive integer")
    fallback_k = int(raw_k)
    if fallback != "off" and gate != "dt_halving":
        raise ValueError("FB1_DESIGN_GATE_FALLBACK=min_delta_k requires FB1_DESIGN_GATE=dt_halving")
    inflate = environ.get("FB1_DESIGN_NOISE_INFLATE", "0")
    if inflate not in ("0", "1"):
        raise ValueError("FB1_DESIGN_NOISE_INFLATE must be 0 or 1")
    if inflate == "1" and gate != "dt_halving":
        raise ValueError("FB1_DESIGN_NOISE_INFLATE=1 requires FB1_DESIGN_GATE=dt_halving")
    softening = parse_softening(world, environ.get("FB1_SOFTENING", "0"))
    config = dict(mode=mode, method=method, dt=dt, dt_override="FB1_DT" in environ,
                design_gate=gate, noise_inflate=inflate == "1", tau=DESIGN_GATE_TAU,
                align_measurement_times=mode == "match_truth" or "FB1_DT" in environ)
    # Disabled safeguards preserve the existing config/provenance key set.
    if fallback != "off":
        config.update(design_gate_fallback=fallback, fallback_k=fallback_k)
    if softening != 0.0:   # the default keeps the key set (and every default record) byte-identical
        config.update(softening=softening, softening_option=environ.get("FB1_SOFTENING"))
    return config


def parse_softening(world, raw):
    """FB1_SOFTENING: "0" (default) | "world" (the simulator's softening of this world, 0 for the field worlds) | a
    finite non-negative number."""
    if raw in (None, "", "0", "0.0"):
        return 0.0
    if raw == "world":
        if world not in TRUTH_SOFTENING:
            raise ValueError(f"FB1_SOFTENING=world: unknown world {world!r}")
        return TRUTH_SOFTENING[world]
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("FB1_SOFTENING must be 0, world or a finite non-negative number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("FB1_SOFTENING must be 0, world or a finite non-negative number")
    return value


def numerical_provenance(world, config):
    """Return the serializable numerics record: the effective config, every world's truth time stepping and sources."""
    table = [dict(world=name, **truth, forward_method=("leapfrog" if config["mode"] == "legacy"
             else truth["method"]), forward_dt=(config["dt"] if config["dt_override"]
             or config["mode"] == "legacy" else truth["dt"]))
             for name, truth in TRUTH_NUMERICS.items()]
    if config.get("softening"):
        for row in table:
            row["forward_softening"] = config["softening"]
            row["truth_softening"] = TRUTH_SOFTENING[row["world"]]
    return dict(schema="fb1-numerics-1", world=world, effective_config=dict(config),
        truth_reference_commit=TRUTH_COMMIT, truth_table=table,
        dispatch_source="ScienceAgent/scienceagent/worlds.py:905-955",
        nbody_method_source="PhysicsSchool/physchool/worlds/nbody_sampler.py:184-189,247-262",
        nbody_default_source="ScienceAgent/scienceagent/executor.py:1083",
        field_method_source="PhysicsSchool/physchool/worlds/field_sampler.py:207-231",
        raw_sampler_defaults=dict(nbody_method="leapfrog", nbody_dt=.01, field_dt=.01),
        truth_path="World.observe and World.truth use the same get_world executor; only noise is disabled for truth",
        limitations=["Time stepping only: forward forces remain free-space candidate laws; field FFT/CIC and periodic wrapping are not reproduced.",
                    "NBody truth softening .05 is not inserted into candidate force laws.",
                    "dt halving is a posterior-dependent numerical sensitivity heuristic, not a formal convergence guarantee."])


def assess_design_gate(coarse, fine, sigma, inflate=False, *, fallback="off", fallback_k=8):
    """Return keep mask, per-design RMS delta and serializable audit record.

    Both arrays contain the same top-BMA model's full posterior bank (D,P,T,2).
    Inflation retains all finite candidates and adds every measured delta^2,
    including errors below tau; nonfinite trajectories can never be inflated.
    """
    if fallback not in ("off", "min_delta_k"):
        raise ValueError("fallback must be off or min_delta_k")
    if type(fallback_k) is not int or fallback_k <= 0:
        raise ValueError("fallback_k must be a positive integer")
    coarse, fine = np.asarray(coarse), np.asarray(fine)
    if coarse.shape != fine.shape or coarse.ndim != 4:
        raise ValueError("gate trajectories must have matching (D,P,T,2) shapes")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("gate sigma must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.sqrt(np.mean((coarse - fine) ** 2, axis=(1, 2, 3)))
    relative = delta / sigma
    finite = np.isfinite(relative)
    over = finite & (relative > DESIGN_GATE_TAU)
    keep = finite if inflate else finite & ~over
    fallback_used = fallback == "min_delta_k" and not np.any(keep) and np.any(finite)
    if fallback_used:
        finite_indices = np.flatnonzero(finite)
        # Stable sorting resolves equal deltas by the original candidate index.
        selected = finite_indices[np.argsort(relative[finite_indices], kind="stable")[:fallback_k]]
        keep[selected] = True
    values = relative[finite]
    record = dict(method="dt_halving", tau=DESIGN_GATE_TAU, sigma=float(sigma),
        delta_definition="RMS over full posterior particles, measurement times and coordinates of the highest-weight model",
        candidate_count=len(delta), excluded_count=int((~keep).sum()),
        inflated_count=int((finite & (delta > 0)).sum()) if inflate else 0,
        above_tau_count=int(over.sum()), nonfinite_count=int((~finite).sum()),
        delta_sigma_quantiles=(dict(zip(("min", "q25", "median", "q75", "max"),
            map(float, np.quantile(values, [0, .25, .5, .75, 1])))) if len(values) else None),
        delta_sigma=[float(v) if ok else None for v, ok in zip(relative, finite)],
        delta=[float(v) if ok else None for v, ok in zip(delta, finite)],
        kept=keep.tolist(), noise_inflate=bool(inflate))
    if fallback != "off":
        kept_indices = selected if fallback_used else np.flatnonzero(keep)
        record.update(fallback_used=bool(fallback_used), fallback=fallback,
                      fallback_k=fallback_k, kept_indices=kept_indices.tolist(),
                      kept_delta_sigma=relative[kept_indices].tolist())
    return keep, delta, record


def inflated_noise_scale(scale, delta, family):
    """Add delta^2 to actual noise variance (Student t4 variance is 2*scale^2)."""
    if family not in ("gauss", "student_t4"):
        raise ValueError("unknown noise family")
    variance_factor = 1.0 if family == "gauss" else 2.0
    return jnp.sqrt(jnp.asarray(scale) ** 2 + jnp.asarray(delta) ** 2 / variance_factor)


def integrate(force, theta, p1, p2, pos0, vel0, times, dt, method=None, align_times=None, softening=None):
    """Selected time stepper; implicit legacy keeps velocity Verlet unchanged.

    theta  (P, k)      p1,p2  (D,)      pos0,vel0  (2,)      times  (T,) ascending
    returns  (D, P, T, 2)
    """
    method = INTEGRATION_METHOD if method is None else method
    align_times = ALIGN_MEASUREMENT_TIMES if align_times is None else align_times
    if method not in ("leapfrog", "symplectic_euler", "yoshida4"):
        raise ValueError("unsupported forward integration method")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("integration dt must be finite and positive")
    D, P = p1.shape[0], theta.shape[0]
    x = jnp.broadcast_to(pos0, (D, P, 2))
    v = jnp.broadcast_to(vel0, (D, P, 2))
    p1b, p2b = p1[:, None], p2[:, None]                       # (D,1)

    soft = FORCE_SOFTENING if softening is None else softening

    def acc(x, t):
        """Acceleration (D,P,2) at positions x and time t: -(F/p2) along the unit vector x/r."""
        r = jnp.linalg.norm(x, axis=-1)                       # (D,P)
        r = jnp.maximum(r, 1e-9)
        # With softening the kernel sees r_soft = sqrt(r^2 + eps^2) (the simulator's convention); the direction
        # x/r is unchanged. soft == 0 takes the original line, so the default graph is identical.
        f = force(jnp.sqrt(r * r + soft * soft) if soft > 0 else r, p1b, p2b, theta, t)   # (D,P)
        return -(f / p2b)[..., None] * (x / r[..., None])

    def step(carry, _):
        """One velocity-Verlet (leapfrog) step of length dt; carry = (x, v, a, t)."""
        x, v, a, t = carry
        v = v + 0.5 * dt * a
        x = x + dt * v
        t = t + dt
        a = acc(x, t)
        v = v + 0.5 * dt * a
        return (x, v, a, t), None

    # Keep the legacy velocity-Verlet path and operation order intact.
    if method == "symplectic_euler":
        def step(carry, _):
            """One symplectic Euler step of length dt; carry = (x, v, a, t)."""
            x, v, _, t = carry
            v = v + dt * acc(x, t)
            x = x + dt * v
            return (x, v, jnp.zeros_like(x), t + dt), None
    elif method == "yoshida4":
        w = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
        c = jnp.asarray((.5*w, .5*(1-w), .5*(1-w), .5*w))
        d = jnp.asarray((w, 1-2*w, w))
        def step(carry, _):
            """One fourth-order Yoshida step of length dt; carry = (x, v, a, t)."""
            x, v, _, t = carry
            # Native drift/kick composition, including fractional force times.
            for i in range(3):
                x = x + c[i] * dt * v
                t = t + c[i] * dt
                v = v + d[i] * dt * acc(x, t)
            x = x + c[-1] * dt * v
            t = t + c[-1] * dt
            return (x, v, jnp.zeros_like(x), t), None

    def remainder_step(carry, h):
        """One step of length h with the selected method: the final short step up to a measurement time."""
        # Only explicit overrides/match_truth use this branch. A final short
        # step prevents cumulative timestamp drift for arbitrary positive dt.
        x, v, a, t = carry
        if method == "leapfrog":
            v = v + .5 * h * a
            x = x + h * v
            t = t + h
            a = acc(x, t)
            v = v + .5 * h * a
        elif method == "symplectic_euler":
            v = v + h * acc(x, t)
            x = x + h * v
            t = t + h
            a = jnp.zeros_like(x)
        else:
            for i in range(3):
                x = x + c[i] * h * v
                t = t + c[i] * h
                v = v + d[i] * h * acc(x, t)
            x = x + c[-1] * h * v
            t = t + c[-1] * h
            a = jnp.zeros_like(x)
        return x, v, a, t

    outs, t_prev = [], 0.0
    carry = (x, v, acc(x, 0.0), 0.0)
    for tq in times:
        n = int(round((float(tq) - t_prev) / dt))
        remainder = 0.0
        interval = float(tq) - t_prev
        if align_times and not math.isclose(n * dt, interval, rel_tol=1e-12, abs_tol=1e-12):
            n = int(math.floor(interval / dt))
            remainder = interval - n * dt
        carry, _ = jax.lax.scan(step, carry, None, length=n)
        if remainder > 0:
            carry = remainder_step(carry, remainder)
        outs.append(carry[0])
        t_prev = float(tq)
    return jnp.stack(outs, axis=2)


# ---- candidate force laws: all of them our own grammar, unrelated to the DP operator menu ----
# A force takes (r, p1, p2, theta).  The acceleration is always a = F/p2 (Newton's second law), so a candidate with a
# wrong **source coupling** (dependence on p1, p2) writes that dependence differently inside F.
def f_inv_r(r, p1, p2, th, t):
    """F = c*p1/r."""
    return th[:, 0] * p1 / r
def f_inv_r2(r, p1, p2, th, t):
    """F = c*p1/r^2."""
    return th[:, 0] * p1 / r**2
def f_linear(r, p1, p2, th, t):
    """F = c*p1*r."""
    return th[:, 0] * p1 * r
def f_const(r, p1, p2, th, t):
    """F = c*p1 (independent of r)."""
    return th[:, 0] * p1 * jnp.ones_like(r)
def f_screened(r, p1, p2, th, t):
    """F = c*p1*exp(-r/lam)/r."""
    return th[:, 0] * p1 * jnp.exp(-r / th[:, 1]) / r
def f_power(r, p1, p2, th, t):
    """F = c*p1*r^(-a)."""
    return th[:, 0] * p1 * r ** (-th[:, 1])
def f_inv_r_plus_c(r, p1, p2, th, t):
    """F = c*p1/r + b*p1."""
    return th[:, 0] * p1 / r + th[:, 1] * p1
def f_log(r, p1, p2, th, t):
    """F = c*p1*log1p(k*r)/r."""
    return th[:, 0] * p1 * jnp.log1p(th[:, 1] * r) / r
# variants with a wrong coupling -- they are what makes the design (p1,p2) actually informative for identification
def f_inv_r_p1sq(r, p1, p2, th, t):
    """F = c*p1^2/r (wrong source coupling)."""
    return th[:, 0] * p1**2 / r
def f_inv_r_nomass(r, p1, p2, th, t):
    """F = c*p1*p2/r (wrong coupling: the acceleration does not depend on p2)."""
    return th[:, 0] * p1 * p2 / r      # a = F/p2 = c*p1/r → independent of mass


# measured running totals of forward-model calls (jit execution time; updated at the call sites via block_until_ready)
FORWARD_SECONDS = 0.0
FORWARD_CALLS = 0

def timed(fn):
    """Decorator: time every call of fn (blocking until the result is ready) into FORWARD_SECONDS / FORWARD_CALLS."""
    import time as _t
    def wrapper(*a, **k):
        """Call fn, block until its result is ready and add the elapsed time and one call to the module totals."""
        global FORWARD_SECONDS, FORWARD_CALLS
        t0 = _t.perf_counter(); r = fn(*a, **k); jax.block_until_ready(r)
        FORWARD_SECONDS += _t.perf_counter() - t0; FORWARD_CALLS += 1
        return r
    return wrapper

# neutralizing-background term of the periodic box -- F = c*p1/r - b*p1*r  (matches the force-curve diagnostic to 0.68%)
def f_inv_r_bg(r, p1, p2, th, t):
    """F = c*p1/r - b*p1*r (inverse-r law with the periodic box's background term)."""
    return th[:, 0] * p1 / r - th[:, 1] * p1 * r

# time-modulation template (adopted from a proposal of a parallel line of work):
# radial law x (1 + A1 cos(w t) + A2 sin(w t)).  With A1=A2=0 it reduces to the static law -- a superset model that
# contains the static law.
def f_inv_r_tmod(r, p1, p2, th, t):
    """F = c*p1/r * (1 + A1 cos(w t) + A2 sin(w t))."""
    return (th[:, 0] * p1 / r) * (1.0 + th[:, 1] * jnp.cos(th[:, 3] * t)
                                      + th[:, 2] * jnp.sin(th[:, 3] * t))

# --- further templates (proposed in the same round for every world) ---
# (a) product coupling: for worlds whose acceleration must be a = c*p1*p2/r^k (coulomb_easy: p1 and p2 are both
#     charges, mass 1).  The integrator always uses a = F/p2, so setting **F = c*p1*p2^2/r^k** gives a = c*p1*p2/r^k.
def f_prod_coupling(r, p1, p2, th, t):
    """F = c*p1*p2^2/r^k, i.e. acceleration a = c*p1*p2/r^k."""
    return th[:, 0] * p1 * p2**2 / r ** th[:, 1]

# (b2) crossover template: joins a far-field 1/r and a near-field 1/r^2 with an exponential cutoff (for the KK sum of
#      extra_dimensions).  The plain two-term c1/r + c2/r^2 failed at 30-49% against the KK sum; this form passed at 2.1-4.2%.
def f_crossover(r, p1, p2, th, t):
    """F = c1*p1/r + c2*p1*exp(-r/R)/r^2."""
    return th[:, 0] * p1 / r + th[:, 1] * p1 * jnp.exp(-r / th[:, 2]) / r**2
