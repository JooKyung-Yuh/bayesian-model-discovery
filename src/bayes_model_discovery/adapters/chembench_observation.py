"""ChemBench observation bridge; no fitting, proposal service or LLM calls.

The trusted runner owns this object. Pass only schema/observation dictionaries
and the observe callback to a learner. This Python interface is not an OS sandbox.
Save predictions during discovery; request reference values only after finish().
Native noise is deterministic per design, so repeats are not independent draws.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from typing import Callable

import numpy as np


def sample_designs(count: int, seed: int) -> list[dict[str, float]]:
    """Use native input bounds/scales with a local RNG (no global RNG changes).

    count=1000, seed=42 reproduces the native evaluate_law input sampler.
    Training, acquisition and evaluation callers must use separate streams.
    """
    from autoscilab.oracle.chembench import CHEM_INPUT_BOUNDS, CHEM_LOG_VARS

    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    columns = {}
    for name, (lo, hi) in CHEM_INPUT_BOUNDS.items():
        columns[name] = (
            np.exp(rng.uniform(np.log(lo), np.log(hi), count))
            if name in CHEM_LOG_VARS else rng.uniform(lo, hi, count)
        )
    return [{name: float(values[i]) for name, values in columns.items()}
            for i in range(count)]


def predict_rate(candidate: Callable[..., float], designs: list[dict]) -> np.ndarray:
    """Evaluate an already constructed candidate, never the true oracle law.

    Invalid predictions fail the call; no abs(), clipping or subset scoring.
    Candidate code construction/validation belongs to the common model layer.
    """
    from autoscilab.oracle.chembench import CHEM_INPUT_VARS

    values = np.asarray([
        candidate(*(float(x[name]) for name in CHEM_INPUT_VARS)) for x in designs
    ], dtype=float)
    if values.shape != (len(designs),) or not np.isfinite(values).all():
        raise ValueError("candidate must return one finite scalar per design")
    if (values < 0).any():
        raise ValueError("rate predictions must be nonnegative")
    return values


class ChemBenchSession:
    """Budgeted observations plus explicitly closed-run reference export.

    The learner sees all six native observables, including scalar target r0.
    Domain labels, true parameters, grammar tags and reference values are not
    included in those dictionaries. A scalar likelihood may use only r0; record
    that choice when comparing with a baseline using the auxiliary observables.
    """

    def __init__(self, domain: str, difficulty: str, law_version: str,
                 *, noise_level: float, budget: int):
        from autoscilab.oracle.chembench import ChemBenchOracle

        if budget < 1 or not np.isfinite(noise_level) or noise_level < 0:
            raise ValueError("require positive budget and finite nonnegative noise")
        # The native constructor prints mechanism names; do not put them in a
        # learner-visible stream. Upstream sources and classes remain unchanged.
        with redirect_stdout(StringIO()):
            self._oracle = ChemBenchOracle(
                domain_id=domain, difficulty=difficulty, law_version=law_version,
                noise_level=noise_level, grammar_mode="universal")
        self._budget, self._calls, self._finished = budget, 0, False

    @property
    def calls(self) -> int:
        """Number of observations spent so far (failed native calls included)."""
        return self._calls

    def schema(self) -> dict:
        """The learner-visible schema: target, input names and bounds, log-sampled inputs, observable names."""
        from autoscilab.oracle.chembench import (
            CHEM_INPUT_BOUNDS, CHEM_INPUT_VARS, CHEM_LOG_VARS, CHEM_OUTPUT_VARS)

        return {
            "benchmark": "chembench", "target": "r0",
            "input_names": list(CHEM_INPUT_VARS),
            "input_bounds": {k: list(v) for k, v in CHEM_INPUT_BOUNDS.items()},
            "log_sampled_inputs": sorted(CHEM_LOG_VARS),
            "observable_names": list(CHEM_OUTPUT_VARS),
        }

    def observe(self, design: dict[str, float]) -> dict:
        """Spend one observation on `design` (all seven native inputs, inside bounds); returns x, y and all observables."""
        from autoscilab.oracle.chembench import CHEM_INPUT_BOUNDS, CHEM_OUTPUT_VARS

        if self._finished:
            raise RuntimeError("discovery is finished")
        if self._calls >= self._budget:
            raise RuntimeError("observation budget exhausted")
        if set(design) != set(CHEM_INPUT_BOUNDS):
            raise ValueError("provide exactly the seven native input variables")
        x = {k: float(design[k]) for k in CHEM_INPUT_BOUNDS}
        for name, (lo, hi) in CHEM_INPUT_BOUNDS.items():
            if not np.isfinite(x[name]) or not lo <= x[name] <= hi:
                raise ValueError(f"input outside native bounds: {name}")
        self._calls += 1  # failed native calls also consume the attempted call
        result = self._oracle.run(x)
        outputs = {k: float(result.metadata["outputs"][k]) for k in CHEM_OUTPUT_VARS}
        if not np.isfinite(list(outputs.values())).all():
            raise ValueError("native oracle returned nonfinite observables")
        return {"observation_index": self._calls, "x": x,
                "y": float(result.measurement), "outputs": outputs}

    def finish(self) -> None:
        """Close discovery: no further observations; reference values become available."""
        self._finished = True

    def reference_values(self, designs: list[dict]) -> dict[str, np.ndarray]:
        """Evaluator-only targets, after discovery; never candidate selection.

        primary_rate is the target of upstream evaluate_law. In contrast,
        noiseless_observation includes the native background T/pH factor, and
        heldout_observation additionally includes the configured native noise.
        Those definitions must remain distinct in scores and plot labels.
        """
        from autoscilab.oracle.chembench import CHEM_INPUT_VARS, _secondary_effects

        if not self._finished:
            raise RuntimeError("reference values are unavailable during discovery")
        oracle = self._oracle
        primary = np.asarray([
            oracle._rate_fn(oracle._law_params, *(x[k] for k in CHEM_INPUT_VARS))
            for x in designs], dtype=float)
        noiseless = primary * np.asarray([
            _secondary_effects(x["T"], x["pH"]) for x in designs])
        observed = np.asarray([oracle.run(x).measurement for x in designs])
        if not np.isfinite([primary, noiseless, observed]).all():
            raise ValueError("nonfinite reference values")
        return {"primary_rate": primary, "noiseless_observation": noiseless,
                "heldout_observation": observed}
