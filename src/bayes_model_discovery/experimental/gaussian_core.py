"""CORE runner adapters for the existing identity-link scalar inference engine."""

from functools import partial

import numpy as np

from . import positive_inference as scalar
from .belief_packet import interface_grid
from .boxinggym_truth_law import kl_against_pool, normal_law
from .count_inference import loglik_prior_probe, model_logweights
from .design_policies import TARGET, random_draw, select_space_filling
from .real_inference import RealHypothesis

_PRIOR_NUMERIC_FAILURES = (
    "non-finite mean expression; no clipping fallback",
    "non-finite/nonpositive predictive scale; no clipping fallback",
)


def as_real(hypothesis):
    """Return `hypothesis` as a RealHypothesis; refuses non-normal families and proposed priors."""
    if hypothesis.family != "normal":
        raise ValueError("Gaussian CORE candidates must use family normal")
    if isinstance(hypothesis, RealHypothesis):
        return hypothesis
    if getattr(hypothesis, "proposed_priors", ()):
        raise ValueError("Gaussian CORE uses fixed role priors, not proposed priors")
    return RealHypothesis(hypothesis.expression, "normal", hypothesis.parameters,
                          hypothesis.input_names)


def calibrate(public):
    """Same frozen, public-D0 calibration as the BoxingGym real path."""
    values = np.asarray([row["y"] for row in public], dtype=float)
    center, unit = float(values.mean()), float(values.std(ddof=1))
    if not np.isfinite(center) or not np.isfinite(unit) or unit <= 0:
        raise ValueError("D0 response calibration is not finite/positive; no fallback")
    return center, unit


def fit_core(hypothesis, x, y, seed, settings, interface, n_trials=None, *,
             response_unit, response_center):
    """Fit one candidate with the scalar engine (identity link, no transform); returns the engine's fit result."""
    return scalar.fit(as_real(hypothesis), x, y, seed, transform=None,
                       response_unit=response_unit, settings=settings,
                       interface=interface, response_center=response_center)


def prior_probe(hypothesis, interface, count, response_unit, response_center, grid_points):
    """Prior predictive mean/sd on the interface grid; returns (verdict dict, summary dict or None if it fails)."""
    from .boxinggym_core import prior_particles

    hypothesis = as_real(hypothesis)
    grid = interface_grid(interface, grid_points)
    particles = prior_particles(hypothesis, count)
    result = {"n_samples": int(count), "passed": True, "grid_points": grid_points}
    try:
        mean, sd = scalar.mixture_moments(hypothesis, particles, grid, interface,
                                          response_unit, response_center)
    except ValueError as exc:
        # Only the two numerical verdicts in real_inference's mean/scale
        # evaluators are refusals. Shape, family and programming errors still
        # raise; the scalar path has no count-PMF normalization warning.
        if str(exc) not in _PRIOR_NUMERIC_FAILURES:
            raise
        return {**result, "passed": False, "detail": f"{type(exc).__name__}: {exc}"}, None
    finite = bool(np.all(np.isfinite(mean)) and np.all(np.isfinite(sd)))
    result["passed"] = finite
    if not finite:
        result["detail"] = "non-finite scalar prior predictive mean or sd"
        return result, None
    return result, {"grid": grid.tolist(), "mean": mean.tolist(), "sd": sd.tolist(),
                    "n_prior_samples": int(count)}


def guarded_loglik_probe(hypothesis, public, interface, response_unit, response_center, count):
    """Model logp of the public data at prior draws, through the shared count-path probe; returns its verdict dict."""
    from .boxinggym_core import prior_particles

    x, y = np.asarray([row["u"] for row in public]), np.asarray([row["y"] for row in public])
    builder = partial(scalar.build_model, hypothesis, x, y, None,
                      response_unit, interface, response_center)
    return loglik_prior_probe(hypothesis, x, y, interface,
                              prior_draws=prior_particles(hypothesis, count),
                              model_builder=builder, n_trials=None)


def score(fits, inputs, heldout, truth, seed):
    """Held-out density scores and the existing normal-truth quadrature KL."""
    summary = scalar.predictive_summary(fits, inputs, seed, 2048)
    logpdf, _ = scalar.log_predictive(fits, inputs, heldout)
    law = normal_law(*truth)
    divergence = kl_against_pool(law, fits, inputs)
    mean, lower, upper = summary["mean"], summary["lower90"][:, None], summary["upper90"][:, None]
    mse = float(np.mean((mean[:, None] - heldout) ** 2))
    scores = {"nll": -float(logpdf.mean()),
              "expected_log_score": -float(0.5 * np.log(2 * np.pi * np.e * truth[1]**2)) - divergence["kl"],
              "kl": divergence["kl"], "kl_method": divergence["kl_method"],
              "kl_truncated_mass": divergence["kl_truncated_mass"],
              "kl_quadrature_error": divergence["kl_quadrature_error"],
              "mean_rmse": float(np.sqrt(np.mean((mean - truth[0]) ** 2))),
              "rmse": float(np.sqrt(mse)), "mse": mse,
              "coverage90": float(((heldout >= lower) & (heldout <= upper)).mean()),
              "width90": float((upper - lower).mean())}
    return scores, summary


def checkpoint(fits, inputs, heldout, truth, seed_for, seed, t, phase):
    """Score the BMA pool and the top-weight single model; returns (record row, arrays to save, per-model means)."""
    weights = model_logweights(fits)
    best = min(range(len(fits)), key=lambda i: (-weights[i], fits[i].hypothesis.model_id))
    row = {"model_weights": np.exp(weights).tolist(),
           "single_model_id": fits[best].hypothesis.model_id}
    arrays = {"model_logweights": weights}
    for name, pool in (("bma", fits), ("single", [fits[best]])):
        scores, summary = score(pool, inputs, heldout, truth,
                                seed_for("predictive_samples", seed, t, phase, name))
        row[name] = scores
        arrays.update({name + "_" + key: value for key, value in summary.items()
                       if key not in ("single_model_id", "measure")})
    _, components = scalar.log_predictive(fits, inputs, heldout)
    row["per_model_nll"] = {result.hypothesis.model_id: -float(components[i].mean())
                            for i, result in enumerate(fits)}
    models = [{"model_id": result.hypothesis.model_id, "weight": float(np.exp(weights[i])),
               "means": scalar.predictive_means(result, inputs).mean(0)}
              for i, result in enumerate(fits)]
    return row, arrays, models


def select_design(fits, designs, seed, policy, samples, interface, n_trials=None, *,
                  observed_designs=None, bounds=None):
    """Pick the next design under `policy`; returns (chosen design, criterion rows, decision record)."""
    from bayes_model_discovery.eig.joint import choose_design

    if policy == "space_filling":
        return select_space_filling(designs, observed_designs, bounds)
    if policy == "joint_eig":
        estimate = scalar.joint_eig(fits, designs, seed, samples)
        key, values = "eig", estimate["eig"]
        extra = {"mc_standard_error": estimate["mc_standard_error"].tolist(),
                 "outer_samples": estimate["outer_samples"], "negative_estimates_clipped": False}
    elif policy == "point_disagreement":
        key, values = policy, scalar.point_disagreement(fits, designs)
        extra = {"uses_rng": False}
    elif policy == "uniform_random":
        _, index = random_draw(designs, seed)
        key, values = policy, np.eye(1, len(designs), index).ravel()
        extra = {"random_draw_index": index}
    else:
        raise ValueError(f"Gaussian CORE has no registered scalar {policy} adapter")
    rows = [{"design": float(x), key: float(value)} for x, value in zip(designs, values, strict=True)]
    decision = choose_design(rows, key)
    chosen = decision["chosen_design"]
    return chosen, rows, {"design_policy": policy, "objective_column": key,
                           "target": TARGET[policy], **decision, **extra,
                           "criteria": rows, "criteria_columns": [key]}


def same_round_truncation(before, active, inputs, round_number, observation_count,
                          interface=None, n_trials=None):
    """Read the two same-data scalar banks, never historical retired posteriors."""
    before_ids, active_ids = ([fit.hypothesis.model_id for fit in pool] for pool in (before, active))
    if not set(active_ids) <= set(before_ids) or len(active_ids) > 12:
        raise ValueError("active pool must be a subset of this pre-truncation pool")
    hashes = {fit.diagnostics.get("public_data_sha256") for fit in [*before, *active]}
    if len(hashes) != 1 or None in hashes:
        raise ValueError("truncation diagnostic requires the same public dataset hash")
    for fit in [*before, *active]:
        if (fit.diagnostics.get("round") != round_number or
                fit.diagnostics.get("observation_count") != observation_count):
            raise ValueError("truncation diagnostic requires the same round and public dataset")
    seed = int(next(iter(hashes))[:8], 16)
    samples = scalar.predictive_summary(before, inputs, seed, 2048)["samples"]
    p, _ = scalar.log_predictive(before, inputs, samples)
    q, _ = scalar.log_predictive(active, inputs, samples)
    metadata = {"round": round_number, "observation_count": observation_count,
                "public_data_sha256": next(iter(hashes)),
                "scope": "same_round_same_public_data_before_truncation_vs_active",
                "historical_retired_posteriors_used": False,
                "before_ids": before_ids, "active_ids": active_ids,
                "before_fit_files": [f"fits/t{round_number:02d}_{key}.npz" for key in before_ids],
                "active_fit_files": [f"fits/t{round_number:02d}_{key}.npz" for key in active_ids],
                "method": "posterior_predictive_monte_carlo", "draws": 2048, "seed": seed,
                "negative_estimates_clipped": False,
                "mean_predictive_kl_before_to_active": float((p - q).mean()),
                "mean_predictive_total_variation": float(0.5 * np.abs(1 - np.exp(q - p)).mean()),
                "retained_pre_truncation_mass": float(np.exp(model_logweights(before))[
                    [key in active_ids for key in before_ids]].sum())}
    return metadata, {"inputs": np.asarray(inputs), "samples": samples,
                       "before_logpdf": p, "active_logpdf": q}
