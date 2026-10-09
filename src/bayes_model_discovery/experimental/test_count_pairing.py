"""Small paired-random and same-dataset diagnostic contracts; no sequential Monte Carlo (SMC) execution."""

import numpy as np
import pytest

from .core_count import D0, CoreWorld, acquisition_seed_for
from .count_diagnostics import same_round_truncation
from .count_inference import Fit, retain
from .count_models import Hypothesis


def test_random_designs_are_shared_across_cells_not_replicates():
    """Random designs repeat across cells of one seed and differ across replicates."""
    grid = np.linspace(0, 1, 100)
    streams = {}
    for cell in ("000", "010", "100", "110"):
        streams[cell] = [
            float(np.random.default_rng(acquisition_seed_for(11, 0, t)).choice(grid))
            for t in range(1, 21)
        ]
    assert all(v == streams["000"] for v in streams.values())
    assert acquisition_seed_for(11, 0, 5) != acquisition_seed_for(11, 1, 5)
    worlds = [CoreWorld(11, "out", "beta_binomial") for _ in streams]
    data = [
        [world.observe(u) for u in [*D0, *streams[cell]]]
        for cell, world in zip(streams, worlds, strict=True)
    ]
    assert all(rows == data[0] for rows in data)


def test_truncation_compares_only_same_round_fits():
    """Truncation diagnostics refuse fits from different rounds or datasets."""
    fits = [
        Fit(
            Hypothesis(f"b0+{i}", "binomial", (("b0", "intercept"),)),
            {"b0": np.zeros(4)},
            float(i),
            diagnostics={"round": 15, "observation_count": 19, "public_data_sha256": "a" * 64},
        )
        for i in range(13)
    ]
    active, _ = retain(fits, 12)
    metadata, arrays = same_round_truncation(fits, active, [0.2, 0.8], 15, 19)
    assert len(metadata["before_ids"]) == 13 and len(metadata["active_ids"]) == 12
    assert not metadata["historical_retired_posteriors_used"]
    assert arrays["before_bma_logpmf"].shape == (2, 51)
    fits[0].diagnostics["observation_count"] = 24
    with pytest.raises(ValueError, match="same round and public dataset"):
        same_round_truncation(fits, active, [0.2, 0.8], 15, 19)
    fits[0].diagnostics["observation_count"] = 19
    fits[0].diagnostics["public_data_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="same public dataset hash"):
        same_round_truncation(fits, active, [0.2, 0.8], 15, 19)


def test_cell_contract_and_observation_replicates():
    """Cell candidate validation and per-replicate observation seeds behave as specified."""
    import pytest

    from .core_count import D0, CoreWorld, mock_proposals, validate_cell_candidates
    from .count_models import Hypothesis, initial_hypotheses

    data = [(0.5, 25)] * 9
    for cell in (f"{i:03b}" for i in range(8)):
        candidates = mock_proposals(data, 5, cell, "binomial")
        validate_cell_candidates(candidates, cell, "binomial")
    with pytest.raises(ValueError, match="forbids proposal"):
        validate_cell_candidates(initial_hypotheses()[:1], "000", "binomial")
    with pytest.raises(ValueError, match="initial likelihood"):
        validate_cell_candidates(initial_hypotheses("beta_binomial")[:1], "100", "binomial")
    novel = Hypothesis("b0+tanh(3*u)", "binomial", (("b0", "intercept"),))
    with pytest.raises(ValueError, match="initial pool"):
        validate_cell_candidates([novel], "010", "binomial")
    with pytest.raises(ValueError, match="three"):
        validate_cell_candidates(initial_hypotheses(), "110", "binomial")
    first = CoreWorld(11, "out", "beta_binomial", 0)
    second = CoreWorld(11, "out", "beta_binomial", 1)
    assert [first.observe(u)["observation_seed"] for u in D0] != [
        second.observe(u)["observation_seed"] for u in D0
    ]
