"""FB-1 (ForceBench) design search by gradient ascent (added 2026-10-02).

FB1_DESIGN_SEARCH=ascent replaces the 64-candidate draw + argmax of run_gravity.py with: RESTARTS starts drawn uniform on
the linear box from the run rng (the same measure as random64), ascent of a jax-pure, differentiable estimate of the
active objective (joint expected information gain (EIG) for arm digit 1, joint expected predictive information gain
(EPIG) for digit 2) with the round's random draws held fixed across steps (one key per round: common random
numbers), a fixed step budget, and the choice of the endpoint with the highest
objective re-estimated on an independent key at the same M_OUTER / N_INNER. Everything is recorded per restart: the
start, every step (point and objective), the arrival, the re-estimated value and the chosen index, in the shape of the
ChemBench acquisition record ("trajectories").

The estimators here draw exactly what run_gravity.joint_eig / joint_epig draw, in the same key order (inner subsample
with ka, numpy outer component draw seeded from kb, per-group splits of kc), so at a fixed design they equal the runner's
estimators to floating-point round-off (probe: 8.9e-16 on 8 designs); the difference is only that the design-independent
draws are made once and the design enters a pure jnp computation that jax can differentiate.

Parametrisation: z in R^2 unconstrained, t = sigmoid(z) in (0, 1), design = exp(log lo + (log hi - log lo) * t) — log scale
onto [lo, hi]. The sigmoid flattens near the bounds, so the record counts how many endpoints end within EDGE_TOL of a bound
in unit log coordinate (a check requested on 2026-10-02).
"""
import math

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import fb1_core as core

# ---- optimiser constants (fixed in this module; recorded in the run)
RESTARTS = 5
STEPS = 40                      # fixed step budget per round, no early stop
OPTIMIZER = "adam"
LEARNING_RATE = 0.15            # in z (unconstrained) units
BETA1, BETA2, ADAM_EPS = 0.9, 0.999, 1e-8
TRANSFORM = "log_sigmoid"       # design = exp(log lo + (log hi - log lo) * sigmoid(z))
START_DISTRIBUTION = "uniform_linear_box_from_run_rng"   # same measure as random64's candidate draw
EDGE_TOL = 0.01                 # endpoint counted as "at a bound" when its unit log coordinate is within 0.01 of 0 or 1
HOLDOUT_FOLD = 0x41534345       # "ASCE": fold_in constant for the independent re-estimation key
UNIT_CLIP = 1e-6                # starts are mapped to z through logit of the unit log coordinate clipped to [eps, 1-eps]


def design_search_constants():
    """Return the fixed optimiser constants and search rules as a dict for the run record."""
    return {"restarts": RESTARTS, "steps": STEPS, "optimizer": OPTIMIZER, "learning_rate": LEARNING_RATE,
            "beta1": BETA1, "beta2": BETA2, "adam_eps": ADAM_EPS, "transform": TRANSFORM,
            "start_distribution": START_DISTRIBUTION, "edge_tolerance_unit_log": EDGE_TOL,
            "holdout_fold_in": HOLDOUT_FOLD, "unit_clip": UNIT_CLIP, "common_random_numbers": "one draw set per round, shared by every step and start",
            "choice": "endpoint with the highest objective re-estimated on an independent key (fold_in of the round key)",
            "stop_rule": "fixed step budget"}


# ---------------------------------------------------------------- transform
class LogSigmoidTransform:
    """Map unconstrained z in R^2 onto the design box [lo, hi] on a log scale through a sigmoid, and back."""
    def __init__(self, p1_range, p2_range):
        self.lo = jnp.asarray([p1_range[0], p2_range[0]], float)
        self.hi = jnp.asarray([p1_range[1], p2_range[1]], float)
        self.log_lo, self.log_hi = jnp.log(self.lo), jnp.log(self.hi)

    def design(self, z):                       # (S,2) -> (S,2) in the box
        """Map z (S,2) to designs (S,2) inside the box."""
        return jnp.exp(self.log_lo + (self.log_hi - self.log_lo) * jax.nn.sigmoid(z))

    def unit_log(self, design):                # (S,2) -> (S,2) in [0,1]
        """Map designs (S,2) to their unit log coordinate (S,2) in [0,1]."""
        return (jnp.log(jnp.asarray(design, float)) - self.log_lo) / (self.log_hi - self.log_lo)

    def z_from_design(self, design):
        """Inverse map: designs (S,2) to z (S,2), with the unit log coordinate clipped to [UNIT_CLIP, 1-UNIT_CLIP]."""
        t = jnp.clip(self.unit_log(design), UNIT_CLIP, 1.0 - UNIT_CLIP)
        return jnp.log(t) - jnp.log1p(-t)

    def record(self):
        """Return the transform's description (rule, ranges, clip) for the run record."""
        return {"rule": TRANSFORM, "p1_range": [float(self.lo[0]), float(self.hi[0])], "p2_range": [float(self.lo[1]), float(self.hi[1])],
                "unit_coordinate": "(log d - log lo) / (log hi - log lo)", "unit_clip_for_start_logit": UNIT_CLIP}


# ---------------------------------------------------------------- jax-pure estimators (draws mirror run_gravity)
def eig_draws(key, fits, weights, m_outer, n_inner):
    """Every design-independent draw of run_gravity.joint_eig, in its order: inner subsample (ka, shared by all
    candidates), the numpy outer component draw seeded from kb, and per component group (k1 particle indices, k2 noise)."""
    key, ka, kb, kc = jax.random.split(key, 4)
    inner_idx = [jax.random.choice(ka, u.shape[0], (n_inner,), replace=False) for _, u in fits]
    w = np.asarray(weights, float)
    rng_o = np.random.default_rng(int(jax.random.randint(kb, (), 0, 2**31)))
    comp = rng_o.choice(len(fits), m_outer, p=w)
    groups, key_o = [], kc
    for cj in np.unique(comp):
        js = np.where(comp == cj)[0]
        _, u_j = fits[int(cj)]
        key_o, k1, k2 = jax.random.split(key_o, 3)
        idx = jax.random.randint(k1, (len(js),), 0, u_j.shape[0])
        groups.append({"component": int(cj), "js": js, "particle_idx": idx, "noise_key": k2})
    return {"inner_idx": inner_idx, "groups": groups, "m_outer": m_outer, "n_inner": n_inner, "component_draw": comp}


def eig_pure(designs, fits, weights, draws, predict, log_dens, t_obs):
    """joint EIG at `designs` (D,2) with the draws fixed; returns (eig (D,), per-outer contributions (D, M))."""
    D = designs.shape[0]
    p1d, p2d = designs[:, 0], designs[:, 1]
    w = np.asarray(weights, float)
    mus_in = [predict(c, D)(u[idx], p1d, p2d) for (c, u), idx in zip(fits, draws["inner_idx"])]
    s_in = [core.noise_scale(u[idx], c) for (c, u), idx in zip(fits, draws["inner_idx"])]
    m_outer, n_inner = draws["m_outer"], draws["n_inner"]
    lp_true = jnp.zeros((D, m_outer))
    lp_marg = jnp.zeros((D, m_outer))
    for g in draws["groups"]:
        cand_j, u_j = fits[g["component"]]
        js = g["js"]
        uj = u_j[g["particle_idx"]]
        mu_j = predict(cand_j, D)(uj, p1d, p2d)
        sj = core.noise_scale(uj, cand_j)
        eps = (jax.random.normal(g["noise_key"], (D, len(js), t_obs, 2)) if cand_j.family == "gauss"
               else jax.random.t(g["noise_key"], 4.0, (D, len(js), t_obs, 2)))
        y_j = mu_j + sj[None, :, None, None] * eps
        sj_design = jnp.broadcast_to(sj, (D, len(js)))
        lp_true = lp_true.at[:, js].set(log_dens(y_j, mu_j, sj_design, cand_j.family))
        terms = [logsumexp(log_dens(y_j[:, :, None], mus_in[ci][:, None, :], s_in[ci][None, None, :], ci_c.family), axis=2)
                 - math.log(n_inner) + math.log(max(w[ci], 1e-300)) for ci, (ci_c, _) in enumerate(fits)]
        lp_marg = lp_marg.at[:, js].set(logsumexp(jnp.stack(terms, axis=2), axis=2))
    contrib = lp_true - lp_marg
    return contrib.mean(axis=1), contrib


#: Added 2026-10-05 for the unified protocol: the chain-rule parts of the joint EIG, I((M,theta);Y) = I(M;Y) +
#: E_M I(theta;Y|M), with the definitions used for BoxingGym and NeuronBench (positive_inference.joint_eig(decompose=True),
#: vector_inference.joint_eig): model (= "structure_eig" there, MDA v4 Eq. 26) = E[log p_k(y) - log p(y)] and
#: parameter = E[log p(y|theta,k) - log p_k(y)], where k is the outer draw's own model and p_k its inner-particle predictive.
EIG_PART_OBJECTIVES = ("model_eig", "parameter_eig")
EIG_PART_COLUMNS = ("joint_eig", "model_eig", "parameter_eig")


def eig_parts_pure(designs, fits, weights, draws, predict, log_dens, t_obs):
    """joint EIG and its two parts at `designs` (D,2) from the draws of `eig_draws` (or `epig_draws`, a superset).

    The outer draws, inner subsamples and noise are exactly eig_pure's: for an outer draw of model j,
    log p(y|theta,j) is lp_true, log p_j(y) = logsumexp over j's inner particles - log n_inner, and log p(y) = lp_marg.
    Returns {"values": (joint, model, parameter) each (D,), "contributions": the three (D, M) arrays}; joint = model +
    parameter for every draw up to floating-point round-off.
    """
    D = designs.shape[0]
    p1d, p2d = designs[:, 0], designs[:, 1]
    w = np.asarray(weights, float)
    mus_in = [predict(c, D)(u[idx], p1d, p2d) for (c, u), idx in zip(fits, draws["inner_idx"])]
    s_in = [core.noise_scale(u[idx], c) for (c, u), idx in zip(fits, draws["inner_idx"])]
    m_outer, n_inner = draws["m_outer"], draws["n_inner"]
    lp_true = jnp.zeros((D, m_outer))
    lp_own = jnp.zeros((D, m_outer))
    lp_marg = jnp.zeros((D, m_outer))
    for g in draws["groups"]:
        cand_j, u_j = fits[g["component"]]
        js = g["js"]
        uj = u_j[g["particle_idx"]]
        mu_j = predict(cand_j, D)(uj, p1d, p2d)
        sj = core.noise_scale(uj, cand_j)
        eps = (jax.random.normal(g["noise_key"], (D, len(js), t_obs, 2)) if cand_j.family == "gauss"
               else jax.random.t(g["noise_key"], 4.0, (D, len(js), t_obs, 2)))
        y_j = mu_j + sj[None, :, None, None] * eps
        sj_design = jnp.broadcast_to(sj, (D, len(js)))
        lp_true = lp_true.at[:, js].set(log_dens(y_j, mu_j, sj_design, cand_j.family))
        own = [logsumexp(log_dens(y_j[:, :, None], mus_in[ci][:, None, :], s_in[ci][None, None, :], ci_c.family), axis=2)
               - math.log(n_inner) for ci, (ci_c, _) in enumerate(fits)]
        terms = [own[ci] + math.log(max(w[ci], 1e-300)) for ci in range(len(fits))]
        lp_own = lp_own.at[:, js].set(own[g["component"]])
        lp_marg = lp_marg.at[:, js].set(logsumexp(jnp.stack(terms, axis=2), axis=2))
    joint, model, parameter = lp_true - lp_marg, lp_own - lp_marg, lp_true - lp_own
    return {"values": (joint.mean(axis=1), model.mean(axis=1), parameter.mean(axis=1)),
            "contributions": (joint, model, parameter)}


def eig_parts_rows(designs, parts, m_outer):
    """Record rows of `eig_parts_pure` output: the three values and their Monte Carlo standard errors per design."""
    designs = np.asarray(designs, float)
    values = [np.asarray(v, float) for v in parts["values"]]
    errors = [np.asarray(jnp.std(c, axis=1)) / math.sqrt(m_outer) for c in parts["contributions"]]
    return [{"design": [float(designs[i, 0]), float(designs[i, 1])],
             **{name: float(values[k][i]) for k, name in enumerate(EIG_PART_COLUMNS)},
             **{f"{name}_se": float(errors[k][i]) for k, name in enumerate(EIG_PART_COLUMNS)},
             "identity_gap": float(abs(values[0][i] - values[1][i] - values[2][i]))}
            for i in range(designs.shape[0])]


def point_disagreement_values(means, weights):
    """Evidence-weighted variance across models of the posterior-mean forecast, per design.

    `means` (K, D, T, 2): model k's posterior-mean trajectory at design d (the mean over ALL of its particles of the
    predicted positions at the T measurement times); `weights` (K,): the evidence weights. For each scalar coordinate
    (t, axis) the value is sum_k w_k (m_k - mbar)^2 with mbar = sum_k w_k m_k (design_policies.point_disagreement, the
    BoxingGym definition); the design's value is the mean over the T times and both axes (units: position squared).
    A design whose means are not all finite gets None (a record field never stops a run; the BoxingGym function raises).
    """
    means = np.asarray(means, float)
    w = np.asarray(weights, float)
    if means.ndim != 4 or means.shape[0] != len(w) or means.shape[-1] != 2:
        raise ValueError("means must be (model, design, time, 2) with one weight per model")
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0, atol=1e-12, rtol=0):
        raise ValueError("model weights must be nonnegative and normalized")
    out = []
    for d in range(means.shape[1]):
        m = means[:, d]
        if not np.all(np.isfinite(m)):
            out.append(None)
            continue
        centred = m - np.tensordot(w, m, axes=1)
        out.append(float(np.mean(np.tensordot(w, centred ** 2, axes=1))))
    return out


def epig_draws(key, fits, weights, m_outer, n_inner):
    """run_gravity.joint_epig's draws in order (its per-group k2 noise key also seeds the target noise via fold_in)."""
    base = eig_draws(key, fits, weights, m_outer, n_inner)
    for g in base["groups"]:
        g["target_noise_key"] = jax.random.fold_in(g["noise_key"], 0x45504947)
    return base


def epig_pure(designs, target_points, fits, weights, draws, predict, log_dens, t_obs):
    """Mean-target joint EPIG at `designs` (D,2) for fixed `target_points` (K,2); returns (epig (D,), contributions)."""
    designs = jnp.asarray(designs); target_points = jnp.asarray(target_points, float)
    D, K = designs.shape[0], target_points.shape[0]
    p1d, p2d = designs[:, 0], designs[:, 1]
    p1t, p2t = target_points[:, 0], target_points[:, 1]
    w = np.asarray(weights, float)
    m_outer, n_inner = draws["m_outer"], draws["n_inner"]
    mus_in = [predict(c, D)(u[idx], p1d, p2d) for (c, u), idx in zip(fits, draws["inner_idx"])]
    targets_in = [predict(c, K)(u[idx], p1t, p2t) for (c, u), idx in zip(fits, draws["inner_idx"])]
    scales_in = [core.noise_scale(u[idx], c) for (c, u), idx in zip(fits, draws["inner_idx"])]
    log_weights = [math.log(max(v, 1e-300)) - math.log(n_inner) for v in w]
    contributions = jnp.zeros((D, m_outer))
    for g in draws["groups"]:
        cand_j, u_j = fits[g["component"]]
        js = g["js"]
        uj = u_j[g["particle_idx"]]
        mu_j = predict(cand_j, D)(uj, p1d, p2d)
        mu_target = predict(cand_j, K)(uj, p1t, p2t)
        sj = core.noise_scale(uj, cand_j)
        k2, kt = g["noise_key"], g["target_noise_key"]
        eps = (jax.random.normal(k2, (D, len(js), t_obs, 2)) if cand_j.family == "gauss" else jax.random.t(k2, 4.0, (D, len(js), t_obs, 2)))
        eps_target = (jax.random.normal(kt, (K, len(js), t_obs, 2)) if cand_j.family == "gauss" else jax.random.t(kt, 4.0, (K, len(js), t_obs, 2)))
        sj_design = jnp.broadcast_to(sj, (D, len(js)))
        y = mu_j + sj_design[:, :, None, None] * eps
        y_target = mu_target + sj[None, :, None, None] * eps_target
        query_density, query_terms = [], []
        for ci, (cand_i, _) in enumerate(fits):
            ld = log_dens(y[:, :, None], mus_in[ci][:, None, :], scales_in[ci][None, None, :], cand_i.family)
            query_density.append(ld)
            query_terms.append(logsumexp(ld, axis=2) + log_weights[ci])
        log_query = logsumexp(jnp.stack(query_terms, axis=2), axis=2)
        acc = jnp.zeros((D, len(js)))
        for ti in range(K):
            joint_terms, target_terms = [], []
            for ci, (cand_i, _) in enumerate(fits):
                ld_target = log_dens(y_target[ti, :, None], targets_in[ci][ti, None, :], scales_in[ci][None, :], cand_i.family)
                joint_terms.append(logsumexp(query_density[ci] + ld_target[None], axis=2) + log_weights[ci])
                target_terms.append(logsumexp(ld_target, axis=1) + log_weights[ci])
            log_joint = logsumexp(jnp.stack(joint_terms, axis=2), axis=2)
            log_target = logsumexp(jnp.stack(target_terms, axis=1), axis=1)
            acc = acc + (log_joint - log_query - log_target[None]) / K
        contributions = contributions.at[:, js].set(acc)
    return contributions.mean(axis=1), contributions


# ---------------------------------------------------------------- the search
def _rows(designs, values, contrib, phase, step, transform, m_outer):
    se = np.asarray(jnp.std(contrib, axis=1)) / math.sqrt(m_outer)
    unit = np.asarray(transform.unit_log(designs))
    return [{"restart_index": i + 1, "phase": phase, "step": int(step), "design": [float(designs[i, 0]), float(designs[i, 1])],
             "unit_log": unit[i].tolist(), "objective": float(values[i]), "mc_standard_error": float(se[i])} for i in range(designs.shape[0])]


def run_design_search(key, rng, fits, weights, *, objective, p1_range, p2_range, m_outer, n_inner, predict, log_dens, t_obs,
                      target_points=None, restarts=RESTARTS, steps=STEPS, learning_rate=LEARNING_RATE, record_parts=False):
    """One round of ascent. `key` is the round's acquisition key (what joint_eig/joint_epig would have received); `rng` is the
    run's numpy Generator (consumed only for the starts). Returns (chosen design (2,), chosen restart index, record).

    `objective` may also be "model_eig" or "parameter_eig" (`eig_parts_pure` on the same draws as "eig"), and
    `record_parts=True` adds `eig_parts_holdout_rows`: the joint, model and parameter EIG of every endpoint on the
    independent holdout draws (for "epig" the EPIG holdout draws, whose outer and inner draws are eig_draws'). With
    objective "eig" or "epig" and record_parts False (the default) nothing here differs from the original ascent search."""
    if objective not in ("eig", "epig") + EIG_PART_OBJECTIVES:
        raise ValueError("design search objective must be eig or epig")
    if objective == "epig" and target_points is None:
        raise ValueError("EPIG design search needs target_points")
    transform = LogSigmoidTransform(p1_range, p2_range)
    starts = np.column_stack([rng.uniform(*p1_range, restarts), rng.uniform(*p2_range, restarts)])   # linear box, run rng
    S = restarts
    key_hold = jax.random.fold_in(key, HOLDOUT_FOLD)
    if objective == "eig":
        draws_a, draws_c = eig_draws(key, fits, weights, m_outer, n_inner), eig_draws(key_hold, fits, weights, m_outer, n_inner)
        def estimate(designs, draws):
            """Joint EIG (values, contributions) at `designs` on the given draws."""
            return eig_pure(designs, fits, weights, draws, predict, log_dens, t_obs)
    elif objective in EIG_PART_OBJECTIVES:
        draws_a, draws_c = eig_draws(key, fits, weights, m_outer, n_inner), eig_draws(key_hold, fits, weights, m_outer, n_inner)
        part = EIG_PART_COLUMNS.index(objective)
        def estimate(designs, draws):
            """The chosen EIG part (values, contributions) at `designs` on the given draws."""
            parts = eig_parts_pure(designs, fits, weights, draws, predict, log_dens, t_obs)
            return parts["values"][part], parts["contributions"][part]
    else:
        draws_a, draws_c = epig_draws(key, fits, weights, m_outer, n_inner), epig_draws(key_hold, fits, weights, m_outer, n_inner)
        def estimate(designs, draws):
            """Joint EPIG (values, contributions) at `designs` on the given draws."""
            return epig_pure(designs, target_points, fits, weights, draws, predict, log_dens, t_obs)

    def objective_z(z):
        """Sum of the objective over all restarts at z (the ascent target), with (values, contributions) as aux."""
        values, contrib = estimate(transform.design(z), draws_a)
        return jnp.sum(values), (values, contrib)
    value_and_grad = jax.jit(jax.value_and_grad(objective_z, has_aux=True))
    z = transform.z_from_design(jnp.asarray(starts))
    m = jnp.zeros_like(z); v = jnp.zeros_like(z)
    trajectories = [{"restart_index": i + 1, "start": starts[i].tolist(), "start_unit_log": np.asarray(transform.unit_log(jnp.asarray(starts)))[i].tolist(),
                     "steps": []} for i in range(S)]
    (_, (vals0, contrib0)), grad0 = value_and_grad(z)
    start_rows = _rows(np.asarray(transform.design(z)), np.asarray(vals0), contrib0, "start", 0, transform, m_outer)
    grad = grad0
    for step in range(1, steps + 1):
        # Adam ascent in z; the gradient used is the one evaluated at the current point
        m = BETA1 * m + (1 - BETA1) * grad
        v = BETA2 * v + (1 - BETA2) * grad ** 2
        m_hat = m / (1 - BETA1 ** step); v_hat = v / (1 - BETA2 ** step)
        z = z + learning_rate * m_hat / (jnp.sqrt(v_hat) + ADAM_EPS)
        (_, (vals, contrib)), grad = value_and_grad(z)
        d = np.asarray(transform.design(z)); g = np.asarray(grad); vals_np = np.asarray(vals)
        for i in range(S):
            trajectories[i]["steps"].append({"step": step, "design": d[i].tolist(), "z": np.asarray(z)[i].tolist(), "objective": float(vals_np[i]),
                                             "gradient_z": g[i].tolist(), "gradient_norm": float(np.linalg.norm(g[i]))})
    arrival_designs = np.asarray(transform.design(z))
    arrival_rows = _rows(arrival_designs, np.asarray(vals), contrib, "arrival", steps, transform, m_outer)
    hold_parts = None
    if objective in EIG_PART_OBJECTIVES:
        # One holdout evaluation gives the chosen part (the choice) and all three columns (the record).
        hold_parts = eig_parts_pure(jnp.asarray(arrival_designs), fits, weights, draws_c, predict, log_dens, t_obs)
        hold_vals, hold_contrib = hold_parts["values"][part], hold_parts["contributions"][part]
    else:
        hold_vals, hold_contrib = estimate(jnp.asarray(arrival_designs), draws_c)
    holdout_rows = _rows(arrival_designs, np.asarray(hold_vals), hold_contrib, "independent_holdout", steps, transform, m_outer)
    hold_np = np.asarray(hold_vals)
    chosen = int(np.argmax(hold_np))
    unit_end = np.asarray(transform.unit_log(jnp.asarray(arrival_designs)))
    at_edge = np.any((unit_end < EDGE_TOL) | (unit_end > 1 - EDGE_TOL), axis=1)
    for i in range(S):
        trajectories[i].update(arrival=arrival_designs[i].tolist(), arrival_unit_log=unit_end[i].tolist(), arrival_value=float(np.asarray(vals)[i]),
                               holdout_value=float(hold_np[i]), value_climbed_seed_a=float(np.asarray(vals)[i] - np.asarray(vals0)[i]),
                               at_edge=bool(at_edge[i]), stop_reason="max_steps")
    record = {"design_search": "ascent", "candidate_source": "ascent_endpoints", "objective": objective, "constants": design_search_constants(),
              "transform": transform.record(), "starts": starts.tolist(), "start_distribution": START_DISTRIBUTION,
              "round_key": [int(x) for x in np.asarray(jax.random.key_data(key)).ravel()], "holdout_key_fold_in": HOLDOUT_FOLD,
              "component_draw_seed_a": draws_a["component_draw"].tolist(), "component_draw_seed_c": draws_c["component_draw"].tolist(),
              "start_rows": start_rows, "arrival_rows": arrival_rows, "holdout_rows": holdout_rows, "trajectories": trajectories,
              "chosen": arrival_designs[chosen].tolist(), "chosen_restart_index": chosen + 1, "chosen_by": "independent_holdout_value",
              "chosen_holdout_value": float(hold_np[chosen]), "chosen_arrival_value": float(np.asarray(vals)[chosen]),
              "seed_a_argmax_restart_index": int(np.argmax(np.asarray(vals))) + 1,
              "endpoints_at_edge": int(at_edge.sum()), "endpoints_at_edge_rule": f"unit log coordinate within {EDGE_TOL} of 0 or 1 on either axis",
              "objective_evaluations": {"value_and_grad_calls": steps + 1, "holdout_calls": 1, "designs_per_call": S}}
    if record_parts or hold_parts is not None:
        # The three EIG columns of every endpoint, on the draws the choice above was made with.
        if hold_parts is None:
            hold_parts = eig_parts_pure(jnp.asarray(arrival_designs), fits, weights, draws_c, predict, log_dens, t_obs)
            record["objective_evaluations"]["eig_parts_holdout_calls"] = 1
        rows = eig_parts_rows(arrival_designs, hold_parts, m_outer)
        for i, row in enumerate(rows):
            row["restart_index"] = i + 1
        record["eig_parts_holdout_rows"] = rows
        record["eig_parts_draws"] = "independent_holdout"
    return arrival_designs[chosen], chosen, record
