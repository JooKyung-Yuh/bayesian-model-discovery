"""Posterior-only diagnostics with an explicit common-data boundary."""

import numpy as np
from scipy.special import xlogy

from .count_inference import predict


def same_round_truncation(
    before, active, inputs, round_number, observation_count, interface=None, n_trials=50
):
    """Compare candidates just before truncation to retained candidates, not history."""
    before_ids = [f.hypothesis.model_id for f in before]
    active_ids = [f.hypothesis.model_id for f in active]
    if not set(active_ids) <= set(before_ids) or len(active_ids) > 12:
        raise ValueError("active pool must be a subset of this pre-truncation pool")
    data_hashes = {result.diagnostics.get("public_data_sha256") for result in [*before, *active]}
    if len(data_hashes) != 1 or None in data_hashes:
        raise ValueError("truncation diagnostic requires the same public dataset hash")
    for result in [*before, *active]:
        if (
            result.diagnostics.get("round") != round_number
            or result.diagnostics.get("observation_count") != observation_count
        ):
            raise ValueError("truncation diagnostic requires the same round and public dataset")
    predictions = [predict(pool, inputs, interface, n_trials) for pool in (before, active)]
    p, q = [np.exp(row["bma_logpmf"]) for row in predictions]
    metadata = {
        "round": round_number,
        "observation_count": observation_count,
        "public_data_sha256": next(iter(data_hashes)),
        "scope": "same_round_same_public_data_before_truncation_vs_active",
        "historical_retired_posteriors_used": False,
        "before_ids": before_ids,
        "active_ids": active_ids,
        "before_fit_files": [f"fits/t{round_number:02d}_{key}.npz" for key in before_ids],
        "active_fit_files": [f"fits/t{round_number:02d}_{key}.npz" for key in active_ids],
        "mean_predictive_kl_before_to_active": float(
            (xlogy(p, p) - p * predictions[1]["bma_logpmf"]).sum(-1).mean()
        ),
        "mean_predictive_total_variation": float(0.5 * np.abs(p - q).sum(-1).mean()),
        "retained_pre_truncation_mass": float(
            np.exp(predictions[0]["model_logweights"])[
                [key in active_ids for key in before_ids]
            ].sum()
        ),
    }
    arrays = {"inputs": np.asarray(inputs)}
    for prefix, prediction in zip(("before", "active"), predictions, strict=True):
        arrays.update(
            {
                f"{prefix}_{key}": value
                for key, value in prediction.items()
                if key != "single_model_id"
            }
        )
    return metadata, arrays
