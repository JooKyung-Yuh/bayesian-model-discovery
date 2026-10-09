"""Synthetic corner acceptance, independent of recorded runs."""

import json

import numpy as np
import pytest

from . import design_search as search
from .core_count import seed_for

DIMENSIONS = 7
FAR_THRESHOLD = 0.25


def corner_objective(relative_noise):
    """Build a seven-input objective: a linear trend plus a smooth seed-fixed random field of the given relative noise."""
    weights = np.linspace(1.0, 0.5, DIMENSIONS)
    fields = {}

    def evaluate(points, seed):
        """Evaluate the corner objective at the given points for this seed."""
        if seed not in fields:
            rng = np.random.default_rng(seed)
            omega = rng.normal(0.0, 1.0 / 0.35, (48, DIMENSIONS))
            phase = rng.uniform(0.0, 2.0 * np.pi, 48)
            coefficients = rng.normal(0.0, 1.0, 48)
            reference = rng.random((4096, DIMENSIONS))
            raw = np.cos(reference @ omega.T + phase) @ coefficients
            scale = relative_noise * np.std(reference @ weights) / np.std(raw)
            fields[seed] = omega, phase, coefficients, scale
        omega, phase, coefficients, scale = fields[seed]
        values = points @ weights + scale * (np.cos(points @ omega.T + phase) @ coefficients)
        return [{"design": point.tolist(), "eig": float(value)}
                for point, value in zip(points, values, strict=True)]

    return evaluate


@pytest.mark.parametrize("relative_noise", [0.0, 0.15])
@pytest.mark.parametrize("seed", range(8))
def test_corner_selected_endpoint_meets_declared_acceptance(tmp_path, seed, relative_noise):
    """The selected restart endpoint meets the declared corner acceptance checks; the evidence record is written."""
    transform = search.SearchTransform(
        tuple(f"x{index}" for index in range(DIMENSIONS)), ((0.0, 1.0),) * DIMENSIONS, ())
    record = search.projected_spsa(
        corner_objective(relative_noise), transform, global_seed=seed, cell="001", round_index=1,
        acquisition_seed=seed_for("synthetic_acquisition", seed, 1),
        acquisition_stream="synthetic_acquisition",
    )
    trajectories = record["trajectories"]
    distances = [float(np.linalg.norm(np.asarray(row["end_unit"]) - 1.0)) for row in trajectories]
    chosen_index = record["chosen_restart_index"] - 1
    chosen = trajectories[chosen_index]
    checks = {
        "selected_stopped_after_gain": chosen["stop_reason"] == "stopped_after_gain",
        "selected_value_climbed_positive": chosen["value_climbed"] > 0,
        "selected_distance_at_most_declared_far_threshold": distances[chosen_index] <= FAR_THRESHOLD,
        "parameters_unchanged": record["parameters"] == search.PARAMETER_DEFAULTS,
        "clipping_exercised": any(step["clip_count"] for row in trajectories for step in row["steps"]),
    }
    evidence = {
        "seed": seed, "relative_noise": relative_noise, "dimensions": DIMENSIONS,
        "objective": "sum(x * linspace(1.0, 0.5, 7))", "optimum": [1.0] * DIMENSIONS,
        "noise_rule": "smooth seed-fixed field; sd = relative_noise * sd(objective)",
        "far_threshold": FAR_THRESHOLD, "distance_by_restart": distances,
        "chosen_distance": distances[chosen_index], "checks": checks, "record": record,
    }
    (tmp_path / "corner_record.json").write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
    assert all(checks.values()), {"seed": seed, "relative_noise": relative_noise,
                                 "distance": distances[chosen_index], "checks": checks}
