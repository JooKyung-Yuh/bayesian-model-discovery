"""Approved unbounded-count expected predictive information gain (EPIG) rules; no inference or provider calls."""
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import xlogy
from scipy.stats import nbinom, poisson

from bayes_model_discovery.experimental import count_inference as inference
from bayes_model_discovery.experimental.belief_packet import task_interface
from bayes_model_discovery.experimental.positive_inference import ScalarHypothesis


def count_fit(family="poisson", *, means=(1., 5.), slopes=(.8, -.2), evidence=0.):
    """Build a synthetic count-family fit with two particles."""
    count = len(means)
    hypothesis = ScalarHypothesis("b0+b1*u", family, (("b0", "intercept"), ("b1", "signed")))
    particles = {"b0": np.log(means), "b1": np.asarray(slopes)}
    if family == "negative_binomial":
        particles["alpha"] = np.linspace(1., 3., count)
    elif family == "zi_poisson":
        particles["pi"] = np.linspace(.2, .7, count)
    interface = task_interface("unbounded_count", [{"name": "u", "range": [0., 1.]}], {}, (0, None))
    return inference.Fit(hypothesis, particles, evidence, diagnostics={"interface": interface,
        "response_unit": 1., "input_transform": None, "observation_count": 4,
        "public_data_sha256": "synthetic-unbounded-proposal"})


def reference_distribution(result, value):
    """Return the reference SciPy distribution for one input value of a count fit."""
    means = np.exp(result.particles["b0"] + result.particles["b1"] * value)
    if result.hypothesis.family == "negative_binomial":
        alpha = result.particles["alpha"]
        return nbinom(alpha[:, None], alpha[:, None] / (alpha + means)[:, None])
    return poisson(means[:, None])


def reference_bank(result, value, cutoff):
    """Build the reference probability bank: integer masses up to the cutoff plus the aggregated survival tail."""
    distribution = reference_distribution(result, value)
    integers = distribution.pmf(np.arange(cutoff + 1))
    tail = distribution.sf(cutoff)
    if result.hypothesis.family == "zi_poisson":
        pi = result.particles["pi"]
        integers *= 1 - pi[:, None]
        integers[:, 0] += pi
        tail *= 1 - pi[:, None]
    return np.concatenate([integers, tail], axis=1)


@pytest.mark.parametrize("family", ["poisson", "negative_binomial", "zi_poisson"])
def test_every_integer_and_sf_tail_are_retained_without_normalization(family):
    """Check that every integer and sf tail are retained without normalization."""
    result = count_fit(family)
    inputs = np.array([0., .5, 1.])
    banks, info = inference._epig_discrete_banks([result], inputs, 2)
    cutoffs = info["epig_discrete_candidate_cutoffs"] + info["epig_discrete_target_cutoffs"]
    assert info["epig_discrete_global_cutoff"] == max(cutoffs)
    for value, cutoff, bank in zip(inputs, cutoffs, banks, strict=True):
        assert bank.shape == (2, cutoff + 2)
        np.testing.assert_array_equal(bank, reference_bank(result, value, cutoff))
        np.testing.assert_allclose(bank.sum(1), 1., atol=1e-13, rtol=0)
        assert np.all(bank[:, -1] <= 1e-12) and np.all(bank[:, -1] > 0)
    assert info["epig_discrete_max_particle_tail_mass"] == max(bank[:, -1].max() for bank in banks)
    assert info["epig_discrete_probabilities_renormalized"] is False
    assert info["epig_discrete_mass_dropped"] is False


@pytest.mark.parametrize("family", ["poisson", "negative_binomial", "zi_poisson"])
def test_coarsened_mi_matches_larger_support_reference_and_obeys_bound(family):
    """Check that coarsened mi matches larger support reference and obeys bound."""
    result = count_fit(family)
    designs, targets = np.array([0., .5]), np.array([.8, 1.])
    chosen, rows = inference.choose_epig([result], designs, 71, target_points=targets)
    bank = [reference_bank(result, value, 400) for value in np.r_[designs, targets]]
    entropy = lambda probabilities: -np.sum(xlogy(probabilities, probabilities))
    for index, row in enumerate(rows):
        candidate = bank[index]
        candidate_mix = candidate.mean(0)
        eig = entropy(candidate_mix) - np.mean([entropy(p) for p in candidate])
        references = []
        for target in bank[len(designs):]:
            joint = sum(np.outer(x, q) / len(candidate) for x, q in zip(candidate, target, strict=True))
            references.append(entropy(candidate_mix) + entropy(target.mean(0)) - entropy(joint))
        # This is an empirical comparison for these synthetic laws, not an
        # asserted universal MI-error bound implied by the residual tolerance.
        assert row["epig"] == pytest.approx(np.mean(references), abs=2e-10)
        assert row["joint_eig"] == pytest.approx(eig, abs=2e-10)
        assert -1e-12 <= row["epig"] <= row["joint_eig"] + 1e-12
    assert chosen in designs


def test_mixed_count_families_share_particle_order_and_record_deterministic_cutoffs():
    """Check that mixed count families share particle order and record deterministic cutoffs."""
    fits = [count_fit(family, evidence=np.log(weight))
            for family, weight in [("poisson", .2), ("negative_binomial", .5), ("zi_poisson", .3)]]
    chosen, rows = inference.choose_epig(fits, [0., .5], 19, target_points=[.8, 1.])
    subsets, weights = inference._epig_subsample(fits, 19, 128)
    banks, info = inference._epig_discrete_banks(subsets, np.array([0., .5, .8, 1.]), 2)
    from bayes_model_discovery.experimental.count_epig import predictive_information
    for index, row in enumerate(rows):
        pairwise = [predictive_information(target[:, None], banks[index][:, None], weights, [1.])
                    for target in banks[2:]]
        assert row["epig"] == pytest.approx(np.mean([pair["epig"][0] for pair in pairwise]), abs=1e-14)
    record = inference.epig_acquisition_record(chosen, rows, 19)
    assert record["outcome_info"] == info
    assert info["epig_discrete_exact_original_mi"] is False
    assert info["epig_discrete_tail_tolerance_is_mi_error_bound"] is False
    chosen_again, rows_again = inference.choose_epig(fits, [0., .5], 19, target_points=[.8, 1.])
    assert json.dumps(record, sort_keys=True) == json.dumps(
        inference.epig_acquisition_record(chosen_again, rows_again, 19), sort_keys=True)


def test_zero_atom_alone_can_satisfy_tail_criterion_without_invalid_inverse_cdf():
    """Check that zero atom alone can satisfy tail criterion without invalid inverse cdf."""
    result = count_fit("zi_poisson", means=(100.,), slopes=(0.,))
    result = replace(result, particles={**result.particles, "pi": np.array([1 - 5e-13])})
    banks, info = inference._epig_discrete_banks([result], np.array([0., 1.]), 1)
    assert info["epig_discrete_global_cutoff"] == 0
    assert all(bank.shape == (1, 2) and bank[0, 1] <= 1e-12 for bank in banks)
    assert all(bank[0, 0] == result.particles["pi"][0] for bank in banks)


@pytest.mark.parametrize("particles,cutoffs,candidates", [(128, [100000] * 132, 100), (2, [10000, 10000], 1)])
def test_allocation_shape_refuses_huge_joint_tables_without_allocating(particles, cutoffs, candidates):
    """Check that allocation shape refuses huge joint tables without allocating."""
    with pytest.raises(inference.EpigResourceLimit, match="acquisition_arrays") as caught:
        inference._epig_discrete_preflight(particles, cutoffs, candidates)
    assert caught.value.detail["estimated_working_bytes"] > 256 * 1024 * 1024
    assert caught.value.detail["tail_tolerance"] == 1e-12
    assert caught.value.detail["cutoffs"] == cutoffs


def test_support_cap_refuses_instead_of_lowering_cutoff():
    """Check that support cap refuses instead of lowering cutoff."""
    with pytest.raises(inference.EpigResourceLimit, match="integer_support") as caught:
        inference._epig_discrete_preflight(2, [100001, 3], 1)
    assert caught.value.detail["requested_cutoff"] == 100001
    assert caught.value.detail["support_cap"] == 100000


def test_choose_refuses_memory_before_any_pmf_is_allocated(monkeypatch):
    """Check that choose refuses memory before any pmf is allocated."""
    # Enumeration still refuses before PMF allocation; its declared
    # replacement must now compute and record CDF interval masses.
    native = inference._epig_scalar_distribution(count_fit(), np.array([0., 1.]))
    class HugeDistribution:
        """Stand-in distribution whose implied support is too large to enumerate."""
        def isf(self, probability):
            """Return a huge inverse survival value for every particle and input."""
            return np.full((2, 2, 1), 10000.)
        def sf(self, cutoff):
            """Return a tiny survival value for every particle and input."""
            return np.full((2, 2, 1), 1e-13)
        def pmf(self, outcomes):
            """Fail the test if a PMF is ever requested."""
            raise AssertionError("PMF allocation must be unreachable")
        def ppf(self, probabilities):
            """Delegate the quantile function to the native distribution."""
            return native.ppf(probabilities)
        def cdf(self, boundaries):
            """Delegate the cumulative distribution function to the native distribution."""
            return native.cdf(boundaries)
    monkeypatch.setattr(inference, "_epig_scalar_distribution", lambda *args: HugeDistribution())
    _, rows = inference.choose_epig([count_fit()], [0.], 31, target_points=[1.])
    assert rows.outcome_info["epig_discrete_coarsened"] is True
    assert rows.outcome_info["epig_discrete_enumeration_refusal"]["resource"] == "acquisition_arrays"


def test_tail_residual_violation_is_refused_instead_of_renormalized(monkeypatch):
    """Check that tail residual violation is refused instead of renormalized."""
    class InaccurateInverse:
        """Stand-in distribution whose inverse survival function leaves too much tail mass."""
        def isf(self, probability):
            """Return a fixed cutoff for every particle and input."""
            return np.full((2, 2, 1), 10.)
        def sf(self, cutoff):
            """Return a tail mass above the tolerance for every particle and input."""
            return np.full((2, 2, 1), 2e-12)
        def pmf(self, outcomes):
            """Fail the test if a PMF is built before residuals are verified."""
            raise AssertionError("PMF construction must follow verified per-particle residuals")
    monkeypatch.setattr(inference, "_epig_scalar_distribution", lambda *args: InaccurateInverse())
    with pytest.raises(ValueError, match="per-particle tail tolerance"):
        inference.choose_epig([count_fit()], [0.], 31, target_points=[1.])


def test_config_declares_fixed_tail_and_resource_rules():
    """Check that config declares fixed tail and resource rules."""
    config = inference.epig_config(np.linspace(0, 1, 32), 91, "synthetic_uniform")
    assert config["epig_discrete_outcome_rule"] == "enumerate_integers_then_aggregated_tail"
    assert config["epig_discrete_tail_tolerance"] == 1e-12
    assert config["epig_discrete_support_cap"] == 100000
    assert config["epig_discrete_max_working_bytes"] == 256 * 1024 * 1024
    assert config["epig_discrete_tail_tolerance_is_mi_error_bound"] is False


@pytest.mark.parametrize("inputs,cutoff,candidates,retained_arrays,fixed_limit", [
    (1600, 10, 1568, 3, 256 * 1024 * 1024),
    (288, 71, 256, 3, 256 * 1024 * 1024),
    (132, 10, 100, 4, None),
])
def test_retained_particle_input_arrays_cannot_bypass_budget(
        inputs, cutoff, candidates, retained_arrays, fixed_limit):
    """Check that retained particle input arrays cannot bypass budget."""
    # These are independent lower bounds for arrays that coexist in _banks:
    # stored PMFs, quantiles, SF tail tables, concatenated residuals, and (for
    # the 132-input case) one frozen full-input distribution parameter array.
    # No probability table is allocated; only Python integer shapes are used.
    particles = 1536
    width = cutoff + 2
    bank_bytes = particles * inputs * width * 8
    known_retained_minimum = bank_bytes + retained_arrays * particles * inputs * 8
    limit = fixed_limit if fixed_limit is not None else known_retained_minimum - 1
    old_estimate = 8 * (particles * inputs * width + 4 * width**2
                        + 4 * particles * width + 8 * inputs * width)
    assert old_estimate <= limit < known_retained_minimum
    with pytest.raises(inference.EpigResourceLimit, match="acquisition_arrays") as caught:
        inference._epig_discrete_preflight(particles, [cutoff] * inputs, candidates,
                                          max_working_bytes=limit)
    detail = caught.value.detail
    assert detail["estimated_working_bytes"] >= known_retained_minimum
    assert detail["max_working_bytes"] == limit
    assert sum(detail["estimated_array_bytes"].values()) == detail["estimated_working_bytes"]
    assert detail["memory_estimate_rule"] == "retained_input_arrays_and_workspace_v2"


def test_retained_array_floor_is_checked_before_full_input_distributions(monkeypatch):
    """Check that retained array floor is checked before full input distributions."""
    # Broadcasting a 1536-particle fit is tiny; the requested 3200 inputs would
    # require too much retained workspace even with the smallest possible K.
    result = count_fit()
    result = replace(result, particles={key: np.tile(value, 768)
                                       for key, value in result.particles.items()})
    def must_not_construct(*args):
        """Fail the test if a full-input distribution is constructed."""
        raise AssertionError("full-input distribution allocation must be unreachable")
    monkeypatch.setattr(inference, "_epig_scalar_distribution", must_not_construct)
    with pytest.raises(inference.EpigResourceLimit, match="acquisition_arrays") as caught:
        inference._epig_discrete_banks([result], np.linspace(0, 1, 3200), 3168)
    assert caught.value.detail["estimated_array_bytes"]["retained_residuals"] == 1536 * 3200 * 8


def test_high_mean_negative_binomial_records_each_input_cutoff_and_tail():
    """Check that high mean negative binomial records each input cutoff and tail."""
    result = count_fit("negative_binomial", means=(1000., 1200.), slopes=(0., .02))
    result = replace(result, particles={**result.particles, "alpha": np.array([2000., 3000.])})
    inputs = np.array([0., 1.])
    assert np.exp(result.particles["b0"]).max() >= 1000
    banks, info = inference._epig_discrete_banks([result], inputs, 1)
    cutoffs = info["epig_discrete_candidate_cutoffs"] + info["epig_discrete_target_cutoffs"]
    assert len(set(cutoffs)) == 2 and min(cutoffs) > 1000
    assert info["epig_discrete_global_cutoff"] == max(cutoffs)
    assert info["epig_discrete_max_particle_tail_mass"] <= 1e-12
    for value, cutoff, bank in zip(inputs, cutoffs, banks, strict=True):
        np.testing.assert_array_equal(bank, reference_bank(result, value, cutoff))
        np.testing.assert_allclose(bank.sum(1), 1., atol=1e-12, rtol=0)


def test_high_mean_dispersed_negative_binomial_refuses_over_budget_joint():
    """Check that high mean dispersed negative binomial refuses over budget joint."""
    result = count_fit("negative_binomial", means=(1000., 1200.), slopes=(0., 0.))
    subsets, _ = inference._epig_subsample([result], 83, 128)
    with pytest.raises(inference.EpigResourceLimit, match="acquisition_arrays") as caught:
        inference._epig_discrete_banks(subsets, np.array([0., 1.]), 1)
    assert max(caught.value.detail["cutoffs"]) > 1000
    assert caught.value.detail["tail_tolerance"] == 1e-12
    _, rows = inference.choose_epig([result], [0.], 83, target_points=[1.])
    assert rows.outcome_info["epig_discrete_coarsened"] is True
    assert rows.outcome_info["epig_discrete_enumeration_refusal"] == caught.value.detail
