"""Synthetic discrete expected predictive information gain (EPIG) coarsening witnesses; no sequential Monte Carlo (SMC) or provider calls."""
import json
import weakref
from dataclasses import replace

import numpy as np
import pytest

from . import count_inference as inference
from .count_epig import predictive_information
from .test_epig_unbounded import count_fit, reference_distribution

FAMILIES = ("poisson", "negative_binomial", "zi_poisson")


def _scores(bank, candidates, weights):
    """Helper: scores."""
    return predictive_information(bank[:, candidates:], bank[:, :candidates], weights,
                                  np.full(bank.shape[1] - candidates, 1 / (bank.shape[1] - candidates)))


def _refusal(count, resource="acquisition_arrays"):
    """Helper: refusal."""
    return {"resource": resource, "cutoffs": [5000] * count,
            "estimated_working_bytes": 2**40, "max_working_bytes": 256 * 1024**2}


@pytest.mark.parametrize("family", FAMILIES)
def test_under_cap_keeps_integer_banks_and_never_calls_quantile_fallback(monkeypatch, family):
    """Check that under cap keeps integer banks and never calls quantile fallback."""
    def forbidden(*args, **kwargs):
        """Helper: forbidden."""
        raise AssertionError("an under-cap integer bank must not be replaced")
    monkeypatch.setattr(inference, "_epig_discrete_coarsened_bank", forbidden)
    result = count_fit(family)
    _, rows = inference.choose_epig([result], [0., .5], 31, target_points=[.8, 1.])
    subsets, weights = inference._epig_subsample([result], 31, 128)
    banks, info = inference._epig_discrete_banks(subsets, np.array([0., .5, .8, 1.]), 2)
    expected = inference._epig_discrete_information(banks, 2, weights)
    assert rows.outcome_info == info and info["epig_discrete_coarsened"] is False
    np.testing.assert_array_equal([row["epig"] for row in rows], expected["epig"])
    np.testing.assert_array_equal([row["joint_eig"] for row in rows], expected["joint_eig"])


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("resource,means", [
    ("acquisition_arrays", (8000., 10000.)),
    ("integer_support", (200000., 250000.)),
])
def test_high_mean_resource_fallback_is_deterministic_and_records_original_cutoffs(family, resource, means):
    """Check that high mean resource fallback is deterministic and records original cutoffs."""
    result = count_fit(family, means=means, slopes=(0., .02))
    if family == "negative_binomial":
        result = replace(result, particles={**result.particles, "alpha": np.array([2000., 3000.])})
    chosen, rows = inference.choose_epig([result], [0., .3], 71, target_points=[.6, .9])
    info = rows.outcome_info
    assert info["epig_discrete_coarsened"] is True
    assert info["epig_discrete_coarsening_rule"] == "bma_predictive_quantile_intervals_64_with_tails"
    assert np.asarray(info["epig_discrete_bin_boundaries"]).shape == (4, 63)
    assert info["epig_discrete_enumeration_refusal"]["resource"] == resource
    assert len(info["epig_discrete_enumeration_refusal"]["cutoffs"]) == 4
    assert min(info["epig_discrete_enumeration_refusal"]["cutoffs"]) >= 1000
    assert info["epig_discrete_enumeration_cutoffs_known"] is True
    assert info["epig_discrete_exact_original_mi"] is False
    assert info["epig_discrete_mi_interpretation"] == "exact_MI_of_recorded_coarsened_observations"
    assert info["epig_discrete_probabilities_renormalized"] is False
    assert info["epig_discrete_mass_dropped"] is False
    assert info["epig_discrete_coarsened_estimated_working_bytes"] <= 256 * 1024**2
    for row in rows:
        assert -1e-10 <= row["epig"] <= row["joint_eig"] + 1e-10
    repeated, repeated_rows = inference.choose_epig([result], [0., .3], 71, target_points=[.6, .9])
    assert json.dumps(inference.epig_acquisition_record(chosen, rows, 71), sort_keys=True) == json.dumps(
        inference.epig_acquisition_record(repeated, repeated_rows, 71), sort_keys=True)


@pytest.mark.parametrize("family", FAMILIES)
def test_one_byte_estimate_boundary_switches_paths_deterministically(monkeypatch, family):
    """Check that one byte estimate boundary switches paths deterministically."""
    result = count_fit(family)
    inputs = np.array([0., .4, .8, 1.])
    subsets, _ = inference._epig_subsample([result], 37, 128)
    _, info = inference._epig_discrete_banks(subsets, inputs, 2)
    estimate = info["epig_discrete_estimated_working_bytes"]
    original_preflight = inference._epig_discrete_preflight
    # Move only this fixture's enumeration budget around its computed estimate.
    # Production constants and the fallback's own allocation cap stay unchanged.
    assert inference.EPIG_DISCRETE_MAX_WORKING_BYTES == 256 * 1024**2
    for budget, expected_coarsening in ((estimate, False), (estimate - 1, True)):
        def budgeted(*args, budget=budget, **kwargs):
            """Helper: budgeted."""
            return original_preflight(*args, **kwargs, max_working_bytes=budget)
        with monkeypatch.context() as patch:
            patch.setattr(inference, "_epig_discrete_preflight", budgeted)
            first, first_rows = inference.choose_epig([result], inputs[:2], 37, target_points=inputs[2:])
            again, again_rows = inference.choose_epig([result], inputs[:2], 37, target_points=inputs[2:])
        assert first_rows.outcome_info["epig_discrete_coarsened"] is expected_coarsening
        assert json.dumps(inference.epig_acquisition_record(first, first_rows, 37), sort_keys=True) == json.dumps(
            inference.epig_acquisition_record(again, again_rows, 37), sort_keys=True)
        if expected_coarsening:
            refusal = first_rows.outcome_info["epig_discrete_enumeration_refusal"]
            assert refusal["estimated_working_bytes"] == estimate
            assert refusal["max_working_bytes"] == estimate - 1


def test_mixed_model_interval_masses_match_full_family_cdfs_in_particle_order():
    """Check that mixed model interval masses match full family cdfs in particle order."""
    fits = [count_fit(family, evidence=np.log(weight))
            for family, weight in zip(FAMILIES, (.2, .5, .3), strict=True)]
    inputs = np.array([0., .3, .8, 1.])
    subsets, weights = inference._epig_subsample(fits, 97, 128)
    bank, boundaries = inference._epig_continuous_bank(subsets, inputs, weights, 64, discrete=True)
    assert bank.shape == (6, 4, 64) and boundaries.shape == (4, 63)
    expected = []
    for result in subsets:
        per_input = []
        for index, value in enumerate(inputs):
            cdf = reference_distribution(result, value).cdf(boundaries[index])
            if result.hypothesis.family == "zi_poisson":
                pi = result.particles["pi"][:, None]
                cdf = (1 - pi) * cdf + pi * (boundaries[index] >= 0)
            per_input.append(np.diff(np.concatenate([np.zeros((2, 1)), cdf, np.ones((2, 1))], axis=1)))
        expected.append(np.stack(per_input, axis=1))
    np.testing.assert_allclose(bank, np.concatenate(expected), atol=2e-14, rtol=0)
    np.testing.assert_allclose(bank.sum(-1), 1., atol=1e-13, rtol=0)
    assert np.all(bank >= 0)


def test_zip_atom_changes_quantiles_and_is_not_replaced_by_poisson():
    """Check that zip atom changes quantiles and is not replaced by poisson."""
    zipped = count_fit("zi_poisson", means=(1000., 1200.), slopes=(0., 0.))
    zipped = replace(zipped, particles={**zipped.particles, "pi": np.array([.9, .95])})
    inputs, weights = np.array([0., 1.]), np.array([.5, .5])
    bank, boundaries = inference._epig_continuous_bank([zipped], inputs, weights, 64, discrete=True)
    np.testing.assert_allclose(bank[:, :, 0], np.broadcast_to(np.array([.9, .95])[:, None], (2, 2)),
                               atol=1e-15, rtol=0)
    # Fixed bisection may leave a positive sub-integer boundary at the atom.
    assert np.all(np.floor(boundaries[:, :59]) == 0.)
    assert np.all(bank[:, :, 1:59] == 0.)
    poisson_fit = count_fit("poisson", means=(1000., 1200.), slopes=(0., 0.))
    plain, plain_boundaries = inference._epig_continuous_bank([poisson_fit], inputs, weights, 64, discrete=True)
    assert np.all(plain_boundaries > 0.)
    assert np.max(plain[:, :, 0]) < .1
    assert not np.array_equal(bank, plain)


@pytest.mark.parametrize("family", FAMILIES)
def test_dyadic_bins_nest_and_obey_data_processing_against_integer_reference(monkeypatch, tmp_path, family):
    """Check that dyadic bins nest and obey data processing against integer reference."""
    result = count_fit(family, means=(2400., 2800.), slopes=(.015, -.01))
    if family == "negative_binomial":
        result = replace(result, particles={**result.particles, "alpha": np.array([2000., 3000.])})
    inputs = np.array([0., .4, .8, 1.])
    subsets, weights = inference._epig_subsample([result], 53, 128)
    original_preflight = inference._epig_discrete_preflight
    def generous(*args, **kwargs):
        """Helper: generous."""
        return original_preflight(*args, **kwargs, max_working_bytes=1024**3)
    with monkeypatch.context() as patch:
        patch.setattr(inference, "_epig_discrete_preflight", generous)
        banks, exact_info = inference._epig_discrete_banks(subsets, inputs, 2)
    exact = inference._epig_discrete_information(banks, 2, weights)
    coarse, coarse_boundaries = inference._epig_continuous_bank(subsets, inputs, weights, 64, discrete=True)
    fine, fine_boundaries = inference._epig_continuous_bank(subsets, inputs, weights, 128, discrete=True)
    np.testing.assert_array_equal(coarse_boundaries, fine_boundaries[:, 1::2])
    np.testing.assert_allclose(coarse, fine.reshape(2, 4, 64, 2).sum(-1), atol=2e-15, rtol=0)
    cutoffs = exact_info["epig_discrete_candidate_cutoffs"] + exact_info["epig_discrete_target_cutoffs"]
    assert np.all(fine_boundaries <= np.asarray(cutoffs)[:, None])
    coarse_score, fine_score = _scores(coarse, 2, weights), _scores(fine, 2, weights)
    assert np.all(coarse_score["epig"] <= fine_score["epig"] + 1e-10)
    assert np.all(fine_score["epig"] <= exact["epig"] + 1e-10)
    assert np.all(coarse_score["epig"] <= coarse_score["joint_eig"] + 1e-10)
    # The unmodified production cap refuses this same ordered particle bank.
    _, rows = inference.choose_epig([result], inputs[:2], 53, target_points=inputs[2:])
    refusal = rows.outcome_info["epig_discrete_enumeration_refusal"]
    assert refusal["resource"] == "acquisition_arrays"
    assert refusal["estimated_working_bytes"] > refusal["max_working_bytes"] == 256 * 1024**2
    assert rows.outcome_info["epig_discrete_enumeration_cutoffs_known"] is True
    np.testing.assert_array_equal(rows.outcome_info["epig_discrete_bin_boundaries"], coarse_boundaries)
    np.testing.assert_array_equal([row["epig"] for row in rows], coarse_score["epig"])
    (tmp_path / "coarsened_exact_comparison.json").write_text(json.dumps({
        "family": family, "same_ordered_particle_bank": True,
        "reference": "integers_0_to_K_plus_survival_tail", "reference_cap_bytes": 1024**3,
        "production_enumeration_refusal": refusal,
        "quantile_boundaries_within_enumerated_K": True, "dyadic_boundaries_nested": True,
        "epig_64": coarse_score["epig"].tolist(), "epig_128": fine_score["epig"].tolist(),
        "epig_integer_tail": exact["epig"].tolist(),
        "joint_eig_64": coarse_score["joint_eig"].tolist(),
    }, indent=2) + "\n")


def test_chunking_keeps_full_input_quantile_boundaries_and_particle_masses_identical():
    """Check that chunking keeps full input quantile boundaries and particle masses identical."""
    fits = [count_fit(family, evidence=np.log(weight))
            for family, weight in zip(FAMILIES, (.2, .5, .3), strict=True)]
    inputs = np.linspace(0, 1, 35)
    subsets, weights = inference._epig_subsample(fits, 101, 128)
    expected_bank, expected_boundaries = inference._epig_continuous_bank(subsets, inputs, weights, 64,
                                                                       discrete=True)
    bank, info = inference._epig_discrete_coarsened_bank(subsets, inputs, weights, 3, _refusal(35))
    assert info["epig_discrete_input_chunk_size"] == 16
    np.testing.assert_array_equal(bank, expected_bank)
    np.testing.assert_array_equal(info["epig_discrete_bin_boundaries"], expected_boundaries)


def test_larger_particle_bank_has_bounded_chunk_workspace(monkeypatch, tmp_path):
    """Check that larger particle bank has bounded chunk workspace."""
    result = count_fit()
    result = replace(result, particles={name: np.tile(values, 768) for name, values in result.particles.items()})
    inputs, weights = np.linspace(0, 1, 132), np.full(1536, 1 / 1536)
    visited = []
    def synthetic_intervals(fits, points, weights, grid_size, *, discrete):
        """Helper: synthetic intervals."""
        assert discrete is True and len(weights) == 1536 and grid_size == 64
        visited.append(points.copy())
        bank = np.zeros((1536, len(points), 64))
        bank[:, :, 0] = 1.
        return bank, np.broadcast_to(np.arange(1., 64.), (len(points), 63)).copy()
    monkeypatch.setattr(inference, "_epig_continuous_bank", synthetic_intervals)
    bank, info = inference._epig_discrete_coarsened_bank([result], inputs, weights, 100, _refusal(132))
    np.testing.assert_array_equal(np.concatenate(visited), inputs)
    assert max(map(len, visited)) <= 16
    assert bank.nbytes == 1536 * 132 * 64 * 8
    assert np.all(bank[:, :, 0] == 1.) and np.all(bank[:, :, 1:] == 0.)
    estimate = info["epig_discrete_coarsened_estimated_working_bytes"]
    assert bank.nbytes + 1536 * 100 * 64 * 8 <= estimate <= 256 * 1024**2
    phases = info["epig_discrete_coarsened_estimated_array_bytes"]
    assert estimate == max(sum(phase.values()) for phase in phases.values())
    (tmp_path / "coarsened_resource_estimate.json").write_text(json.dumps({
        "particles": 1536, "input_count": 132, "candidate_count": 100,
        "stored_bank_bytes": bank.nbytes, "estimated_working_bytes": estimate,
        "cap_bytes": 256 * 1024**2, "input_chunk_size": info["epig_discrete_input_chunk_size"],
        "estimated_array_bytes": phases, "estimate_is_rss_guarantee": False,
    }, indent=2) + "\n")


def test_second_resource_refusal_retains_original_detail_before_quantile_allocation(monkeypatch):
    """Check that second resource refusal retains original detail before quantile allocation."""
    result = count_fit()
    result = replace(result, particles={name: np.tile(values, 1536) for name, values in result.particles.items()})
    def forbidden(*args, **kwargs):
        """Helper: forbidden."""
        raise AssertionError("quantile allocation must follow the fallback preflight")
    monkeypatch.setattr(inference, "_epig_continuous_bank", forbidden)
    refusal = _refusal(132)
    with pytest.raises(inference.EpigResourceLimit, match="coarsened_acquisition_arrays") as caught:
        inference._epig_discrete_coarsened_bank([result], np.linspace(0, 1, 132),
                                                np.full(3072, 1 / 3072), 100, refusal)
    assert caught.value.detail["enumeration_refusal"] == refusal
    assert caught.value.detail["estimated_working_bytes"] > caught.value.detail["max_working_bytes"]


def test_floor_refusal_reports_unknown_predictive_cutoffs(monkeypatch):
    """Check that floor refusal reports unknown predictive cutoffs."""
    refusal = {**_refusal(2), "cutoffs": [0, 0]}
    def refuse(*args):
        """Helper: refuse."""
        raise inference.EpigResourceLimit(refusal)
    monkeypatch.setattr(inference, "_epig_discrete_banks", refuse)
    _, rows = inference.choose_epig([count_fit()], [0.], 4, target_points=[1.])
    assert rows.outcome_info["epig_discrete_enumeration_refusal"] == refusal
    assert rows.outcome_info["epig_discrete_enumeration_cutoffs_known"] is False
    assert "unknown" in rows.outcome_info["epig_discrete_enumeration_cutoff_note"]


def test_refused_enumeration_temporaries_are_released_before_fallback(monkeypatch):
    """Check that refused enumeration temporaries are released before fallback."""
    class Temporary:
        """Synthetic Temporary fixture used by the surrounding test."""
    refs = []
    original = inference._epig_discrete_coarsened_bank
    def refuse(*args):
        """Helper: refuse."""
        temporary = Temporary()
        refs.append(weakref.ref(temporary))
        raise inference.EpigResourceLimit(_refusal(2))
    def fallback(*args):
        """Helper: fallback."""
        assert refs[-1]() is None
        return original(*args)
    monkeypatch.setattr(inference, "_epig_discrete_banks", refuse)
    monkeypatch.setattr(inference, "_epig_discrete_coarsened_bank", fallback)
    inference.choose_epig([count_fit()], [0.], 4, target_points=[1.])


@pytest.mark.parametrize("error", [ValueError("invalid cutoff"), inference.EpigResourceLimit({"resource": "other"})])
def test_nonresource_failures_are_not_replaced(monkeypatch, error):
    """Check that nonresource failures are not replaced."""
    def refuse(*args):
        """Helper: refuse."""
        raise error
    monkeypatch.setattr(inference, "_epig_discrete_banks", refuse)
    with pytest.raises(type(error)) as caught:
        inference.choose_epig([count_fit()], [0.], 5, target_points=[1.])
    assert caught.value is error


def test_nonfinite_fallback_quantiles_fail_without_a_second_substitute(monkeypatch):
    """Check that nonfinite fallback quantiles fail without a second substitute."""
    class InvalidQuantiles:
        """Synthetic InvalidQuantiles fixture used by the surrounding test."""
        def ppf(self, probabilities):
            """Helper: ppf."""
            return np.full((2, 2, len(probabilities)), np.nan)
    def refuse(*args):
        """Helper: refuse."""
        raise inference.EpigResourceLimit(_refusal(2))
    monkeypatch.setattr(inference, "_epig_discrete_banks", refuse)
    monkeypatch.setattr(inference, "_epig_scalar_distribution", lambda *args: InvalidQuantiles())
    with pytest.raises(ValueError, match="non-finite posterior predictive quantile bracket"):
        inference.choose_epig([count_fit()], [0.], 5, target_points=[1.])


@pytest.mark.parametrize("pi", [0., 1., np.nan])
def test_zip_interval_path_rejects_invalid_zero_inflation(pi):
    """Check that zip interval path rejects invalid zero inflation."""
    result = count_fit("zi_poisson")
    result = replace(result, particles={**result.particles, "pi": np.array([pi, .5])})
    with pytest.raises(ValueError, match="open unit interval"):
        inference._epig_continuous_bank([result], np.array([0., 1.]), np.array([.5, .5]), 64,
                                        discrete=True)


def test_boxing_synthetic_count_runner_records_coarsening_in_two_rounds(tmp_path, monkeypatch):
    """Check that boxing synthetic count runner records coarsening in two rounds."""
    from types import SimpleNamespace

    from . import boxinggym_core as bg
    from .test_named_inputs import bg_run_args, bg_scalar_stub_fits

    class SyntheticCountGoal:
        """Synthetic SyntheticCountGoal fixture used by the surrounding test."""
        # Native-goal shape used by the target sampler: constructed from a
        # realised environment copy, appends one point per goal question.
        def __init__(self, env):
            """Helper: init  ."""
            self.env = env
            self.eval_points = []

        def get_goal_eval_question(self, include_prior=True):
            """Helper: get goal eval question."""
            x = float(np.random.uniform(.1, 4.9))
            self.eval_points.append((x, float(np.rint(1000. * np.exp(.03 * x)))))

    class SyntheticCountEnvironment:
        """Synthetic SyntheticCountEnvironment fixture used by the surrounding test."""
        def __init__(self, spec, seed):
            """Helper: init  ."""
            self.spec = spec
            self.env = SimpleNamespace(alpha=np.log(1000.), beta1=.03, beta2=0., beta3=0.)
            self.goal = SyntheticCountGoal(self.env)
            self.observation_calls = 0
        def evaluation_points(self, count, seed):
            """Helper: evaluation points."""
            x = np.linspace(.1, 4.9, count)
            y = np.rint(1000. * np.exp(.03 * x))
            self.goal.eval_points = list(zip(x, y, strict=True))
            return x, y
        def eval_inputs(self, point):
            """Helper: eval inputs."""
            return [point[0]]
        def world_parameters(self):
            """Helper: world parameters."""
            return {"synthetic_count_world": True}
        def refused_designs(self):
            """Helper: refused designs."""
            return frozenset()
        def accepts(self, handle):
            """Helper: accepts."""
            return 0 <= float(handle) <= 5
        def observe(self, handle):
            """Helper: observe."""
            self.observation_calls += 1
            return float(round(1000. * np.exp(.03 * float(handle))))
        def evaluate(self, predictions):
            """Helper: evaluate."""
            predictions = np.array([float(value) for value in predictions])
            measurements = np.array([point[-1] for point in self.goal.eval_points])
            return {"metric": float(np.mean((predictions - measurements)**2)), "spread": 0.}

    # The runner hashes its declared environment source; this fixture owns it.
    monkeypatch.setitem(bg.TASKS, "peregrines", replace(bg.TASKS["peregrines"], module=__name__))
    monkeypatch.setattr(bg, "TaskEnvironment", SyntheticCountEnvironment)
    bg_scalar_stub_fits(monkeypatch)
    original_fit = bg.scalar_fit
    def high_count_fit(hypothesis, inputs, labels, seed, transform, response_unit, settings,
                       interface=None, response_center=0.):
        """Helper: high count fit."""
        result = original_fit(hypothesis, inputs, labels, seed, transform, response_unit, settings,
                              interface, response_center)
        particles = dict(result.particles)
        for name, _ in hypothesis.parameters:
            particles[name] = (np.log(8000. / response_unit) + np.linspace(-.1, .1, 8)
                               if name == "b0" else np.zeros(8))
        return replace(result, particles=particles)
    monkeypatch.setattr(bg, "scalar_fit", high_count_fit)
    output = tmp_path / "synthetic_runner"
    bg.run(bg_run_args(output, task="peregrines", cell="102", rounds=2))
    for index in (1, 2):
        record = json.loads((output / "acquisition" / f"t{index:02d}.json").read_text())
        info = record["outcome_info"]
        assert info["epig_discrete_coarsened"] is True
        assert info["epig_discrete_path_fixed_at_start"] is True
        assert info["epig_discrete_path"] == "coarsened"
        if index == 1:
            assert info["epig_discrete_enumeration_refusal"]["resource"] == "acquisition_arrays"
            outcome_keys = set(info) - {"epig_discrete_boundary_repairs"}
        else:
            assert info["epig_discrete_enumeration_refusal"] is None
            assert info["epig_discrete_enumeration_cutoffs_known"] is False
            assert info["epig_discrete_enumeration_cutoff_note"] == "not_recomputed: coarsened_path_fixed_at_D0"
            assert set(info) - {"epig_discrete_boundary_repairs"} == outcome_keys
        assert len(info["epig_discrete_bin_boundaries"]) == len(record["rows"]) + 32
        assert all(len(bounds) == 63 for bounds in info["epig_discrete_bin_boundaries"])
        assert record["rows"][record["policy"]["chosen_index"]]["design"] == record["chosen"]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["paid_api_calls"] == 0
    assert summary["environment_calls"] == 6


def test_invalid_interval_masses_fail_without_clipping_or_renormalization(monkeypatch):
    """Check that invalid interval masses fail without clipping or renormalization."""
    class InvalidCDF:
        """Synthetic InvalidCDF fixture used by the surrounding test."""
        def ppf(self, probabilities):
            """Helper: ppf."""
            return np.broadcast_to(probabilities * 10, (2, 2, len(probabilities))).copy()
        def cdf(self, boundaries):
            """Helper: cdf."""
            result = np.broadcast_to(np.linspace(.1, .9, boundaries.shape[-1]),
                                     (2, 2, boundaries.shape[-1])).copy()
            result[:, :, 31] = 0.
            return result
    monkeypatch.setattr(inference, "_epig_scalar_distribution", lambda *args: InvalidCDF())
    with pytest.raises(ValueError, match="interval normalization failed; no renormalization"):
        inference._epig_continuous_bank([count_fit()], np.array([0., 1.]), np.array([.5, .5]), 64,
                                        discrete=True)
