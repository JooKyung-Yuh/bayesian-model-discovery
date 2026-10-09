"""Native forward and small Gaussian mixture/sequential Monte Carlo (SMC) examples, not benchmark claims."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from .count_inference import SMCSettings
from .neuron_forward import HHHypothesis, native_test_counts, predict_features
from .vector_inference import fit, joint_eig, log_predictive, predictive_summary


@pytest.mark.usefixtures("require_neuronbench_source")
def test_native_candidate_forward_and_boundary():
    """Check that native candidate forward and boundary."""
    from neuronbench.worlds import build_I, simulate

    design = ("public brief step", [(40.0, 12.0)])
    current, start = build_I(design[1])
    count, reference = simulate(current, start, trace=True)
    h = HHHypothesis("leak")
    np.testing.assert_array_equal(h.predict([1.0], design), reference)
    assert native_test_counts(reference, start)[0] == count
    assert native_test_counts(np.array([-1.0, 1.0, -1.0]), 1)[0] == 1
    assert not np.array_equal(h.predict([1.4], design), reference)
    assert predict_features(h, [1.0], design).shape == (3,)


def test_gaussian_vector_mixture_not_plugin_and_units():
    """Check that gaussian vector mixture not plugin and units."""
    means = np.array([[[-1.0, 2.0]], [[2.0, 3.0]]])
    bank = {"means": means, "sigmas": np.array([0.5, 2.0]), "logweights": np.log([0.25, 0.75])}
    y = np.array([[0.0, 1.0]])
    expected = logsumexp(
        norm.logpdf(y, means, bank["sigmas"][:, None, None]).sum(-1)[:, 0] + bank["logweights"]
    )
    np.testing.assert_allclose(log_predictive(bank, y), [expected], atol=1e-12)
    unit = 20.0
    converted = {**bank, "means": means / unit, "sigmas": bank["sigmas"] / unit}
    np.testing.assert_allclose(
        log_predictive(converted, y / unit) - 2 * np.log(unit), [expected], atol=1e-12
    )


def test_joint_eig_contains_parameter_information():
    # A single model with a finite Gaussian-mean posterior, known sigma=1.
    """Check that joint eig contains parameter information."""
    means = norm.ppf((np.arange(128) + 0.5) / 128).reshape(128, 1, 1)
    bank = {"means": means, "sigmas": np.ones(128), "logweights": np.full(128, -np.log(128))}
    _, rows = joint_eig(bank, 749, samples=8192)
    assert abs(rows[0]["eig"] - 0.5 * np.log(2)) < 0.03
    bank["means"] = np.zeros_like(means)
    _, rows = joint_eig(bank, 749, samples=64)
    assert abs(rows[0]["eig"]) < 1e-12


def test_future_feature_interval_includes_noise():
    """Check that future feature interval includes noise."""
    bank = {"means": np.zeros((1, 2, 3)), "sigmas": np.array([20.0]), "logweights": np.zeros(1)}
    result = predictive_summary(bank)
    np.testing.assert_allclose(result["feature_predictive_mean"], 0)
    np.testing.assert_allclose(result["feature_marginal_lower90"], 20 * norm.ppf(0.05))
    np.testing.assert_allclose(result["feature_marginal_upper90"], 20 * norm.ppf(0.95))


def test_real_vector_smc_original_prior_path():
    """Check that real vector smc original prior path."""
    def forward(hypothesis, theta, design):
        """Forward."""
        return np.array([theta[0] * design])

    result = fit(
        SimpleNamespace(model_id="linear_fixture"),
        [1.0, 2.0],
        [[1.1], [2.0]],
        forward,
        671,
        SMCSettings(32, 2, 2),
        scale=1.0,
    )
    assert np.isfinite(result.log_evidence)
    assert len(result.per_chain_log_evidence) == 2
    assert len(result.particles["conductance"]) == 64
    assert result.diagnostics["original_prior_refit"]


# Synthetic fixtures exercise the public channel equations; no saved run output.
INACTIVATING_CANDIDATES = (
    "rebound_inward_slow",
    "rebound_inward_fast",
    "rebound_outward",
)


def _capture_candidate_channels(monkeypatch, *, omit_htau=False):
    from neuronbench import worlds

    from . import neuron_forward

    original = worlds.Chan
    channels = []

    def capture(name, conductance, *shape):
        """Capture."""
        if omit_htau and name in INACTIVATING_CANDIDATES:
            # The historical seven-field recipe silently takes htau=1e9.
            shape = shape[:7]
        channel = original(name, conductance, *shape)
        channels.append(channel)
        return channel

    monkeypatch.setattr(worlds, "Chan", capture)
    neuron_forward._trace.cache_clear()
    return channels


def _assert_native_deinactivation(hypothesis, channels):
    from neuronbench.worlds import ADVANCED, TEXTBOOK, _hinf, build_I

    def gate_at_test(design):
        """Gate at test."""
        _, start = build_I(design[1])
        voltage = hypothesis.predict([1.0], design)
        channel = channels[-1]
        initial = _hinf(-65.0, channel.hvh, channel.hk)
        # Reconstruct the native h recurrence from its actual pre-update
        # voltages. This measures the integrated gate, not just h_inf(V).
        previous = np.concatenate(([-65.0], voltage[:start - 1]))
        targets = _hinf(previous, channel.hvh, channel.hk)
        decay = 1.0 - 0.01 / channel.htau
        powers = decay ** np.arange(start - 1, -1, -1)
        recovered = decay**start * initial + (0.01 / channel.htau) * (powers @ targets)
        assert voltage.shape == (len(build_I(design[1])[0]),)
        assert np.isfinite(voltage).all()
        return initial, recovered

    initial, unconditioned = gate_at_test(TEXTBOOK[0])
    _, conditioned = gate_at_test(ADVANCED[0])
    assert conditioned > initial + 0.15, "inactivation gate stayed frozen during conditioning"
    assert conditioned > unconditioned + 0.15, "hyperpolarisation did not release inactivation"


@pytest.mark.parametrize("mechanism", INACTIVATING_CANDIDATES)
@pytest.mark.usefixtures("require_neuronbench_source")
def test_inactivating_candidate_forward_shape_and_conductance(mechanism, monkeypatch):
    """Check that inactivating candidate forward shape and conductance."""
    from neuronbench.worlds import build_I

    channels = _capture_candidate_channels(monkeypatch)
    hypothesis = HHHypothesis(mechanism)
    design = ("synthetic short conditioning", [(30.0, -30.0), (40.0, 12.0)])
    weak = hypothesis.predict([0.5], design)
    strong = hypothesis.predict([2.0], design)
    assert weak.shape == strong.shape == build_I(design[1])[0].shape
    assert np.isfinite(weak).all() and np.isfinite(strong).all()
    assert not np.array_equal(weak, strong)
    assert [channel.g for channel in channels] == [0.5, 2.0]
    assert all(channel.hvh is not None and channel.htau < 1000 for channel in channels)
    assert predict_features(hypothesis, [2.0], design).shape == (3,)


@pytest.mark.parametrize("mechanism", INACTIVATING_CANDIDATES)
@pytest.mark.usefixtures("require_neuronbench_source")
def test_native_inactivation_gate_recovers_after_conditioning(mechanism, monkeypatch):
    """Check that native inactivation gate recovers after conditioning."""
    channels = _capture_candidate_channels(monkeypatch)
    _assert_native_deinactivation(HHHypothesis(mechanism), channels)


@pytest.mark.parametrize("mechanism", INACTIVATING_CANDIDATES)
@pytest.mark.usefixtures("require_neuronbench_source")
def test_seven_tuple_mutation_fails_native_deinactivation(mechanism, monkeypatch):
    """Check that seven tuple mutation fails native deinactivation."""
    from . import neuron_forward

    channels = _capture_candidate_channels(monkeypatch, omit_htau=True)
    try:
        with pytest.raises(AssertionError, match="inactivation gate stayed frozen"):
            _assert_native_deinactivation(HHHypothesis(mechanism), channels)
        assert channels and all(channel.htau == 1e9 for channel in channels)
    finally:
        # Mutated traces must never survive into another test or a real run.
        neuron_forward._trace.cache_clear()


def test_initial_pool_and_mock_schedule_remain_world_independent():
    """Check that initial pool and mock schedule remain world independent."""
    from .neuron_forward import initial_hypotheses, mock_proposals

    assert [h.mechanism for h in initial_hypotheses()] == [
        "leak", "sodium", "potassium", "slow_na",
    ]
    for event in (1, 2, 3, 4):
        assert [h.mechanism for h in mock_proposals(event)] == [
            "h_current", "m_current", "inward",
        ]
    assert not set(INACTIVATING_CANDIDATES) & {h.mechanism for h in initial_hypotheses()}


# ---------------------------------------------------------------------------
# Batch forward backend: the feature-level prefill cache that
# `predict_features` consults. The batch integrator itself, the SMC kernel
# that fills the cache and the equivalence report live in
# test_neuron_forward_batch.py.
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("require_neuronbench_source")
def test_prefilled_features_are_served_from_the_cache_and_misses_are_counted():
    """Check that prefilled features are served from the cache and misses are counted."""
    from .neuron_forward import clear_prefill, prefill_features, prefill_stats

    hypothesis = HHHypothesis("potassium")
    design = ("synthetic prefill probe", [(6.0, 11.0)])
    thetas = [0.7, 1.0, 1.9]
    clear_prefill()
    try:
        native = [predict_features(hypothesis, [theta], design) for theta in thetas]
        # An empty cache changes nothing: every lookup misses and answers natively.
        assert prefill_stats() == {"entries": 0, "hits": 0, "misses": len(thetas)}

        clear_prefill()
        rows = prefill_features(hypothesis, thetas, design)
        assert prefill_stats() == {"entries": len(thetas), "hits": 0, "misses": 0}
        for index, theta in enumerate(thetas):
            served = predict_features(hypothesis, [theta], design)
            assert np.array_equal(served.view(np.uint64), native[index].view(np.uint64))
            assert np.array_equal(served, rows[index])
        assert prefill_stats() == {"entries": len(thetas), "hits": len(thetas), "misses": 0}

        # A cached row is handed out as a copy: a caller that scales it in place (the runner
        # does not, but nothing stops one) must not corrupt the next particle's features.
        served = predict_features(hypothesis, [thetas[0]], design)
        served *= 0.0
        assert np.array_equal(
            predict_features(hypothesis, [thetas[0]], design), native[0]
        )

        # A multiplier nobody prefilled is computed natively and counted as a miss; so is a
        # prefilled multiplier under a design nobody prefilled.
        unprefilled = predict_features(hypothesis, [2.4], design)
        assert unprefilled.shape == (3,) and np.isfinite(unprefilled).all()
        assert not np.array_equal(unprefilled, native[0])
        other = ("synthetic unprefilled probe", [(7.0, 11.0)])
        predict_features(hypothesis, [thetas[0]], other)
        assert prefill_stats()["misses"] == 2
        assert prefill_stats()["entries"] == len(thetas)  # a miss never fills the cache

        clear_prefill()
        assert prefill_stats() == {"entries": 0, "hits": 0, "misses": 0}
        assert np.array_equal(
            predict_features(hypothesis, [thetas[0]], design).view(np.uint64),
            native[0].view(np.uint64),
        )
        assert prefill_stats()["misses"] == 1
    finally:
        clear_prefill()


# ---------------------------------------------------------------------------
# Tests for the `paper_neuron` profile's kernel, feature space and noise. The
# CLI side lives in test_neuron_core_cli.py and the batch kernel's equivalence
# in test_neuron_forward_batch.py; everything here is a pure-function or
# small-synthetic-SMC claim about `vector_inference`.
# ---------------------------------------------------------------------------


def _toy_smc_model(observations=(0.3, -0.2, 0.9, 1.4)):
    """A two-parameter Gaussian toy: enough for a real tempering schedule, no forward at all."""
    import pymc as pm

    with pm.Model() as model:
        location = pm.Normal("location", 0, 1)
        spread = pm.HalfNormal("spread", 1)
        pm.Normal("observed", mu=location, sigma=spread, observed=np.asarray(observations))
    return model


def _drive_kernel(kernel_class, seed, **kernel_kwargs):
    """PyMC 5.13.1's own SMC driver loop (`pymc.smc.sampling._sample_smc_int`), in process.

    `pm.sample_smc` runs every chain in a worker process even at `cores=1`, which a monkeypatch
    in this process would never reach. The loop below is what that worker runs, so the two
    kernels can be compared - and PyMC's own `Pearson` stopping rule replaced - directly.
    """
    kernel = kernel_class(draws=64, start=None, model=_toy_smc_model(), random_seed=seed,
                          threshold=0.5, **kernel_kwargs)
    kernel._initialize_kernel()
    kernel.setup_kernel()
    steps = []
    while kernel.beta < 1:
        kernel.update_beta_and_weights()
        kernel.resample()
        kernel.tune()
        kernel.mutate()
        steps.append(kernel.n_steps)
    return kernel, steps


class _ThreeStepPearson:
    """PyMC's `Pearson` replaced by one that makes `MH.mutate` stop after exactly three steps.

    `MH.mutate` continues while the correlation has fallen by more than `correlation_threshold`
    for over 90 % of the dimensions. Returning a strictly falling value twice and then a repeat
    keeps it going through steps 1 and 2 and stops it at step 3, without touching the kernel's
    random numbers - `Pearson` consumes none.
    """

    def __init__(self, a):
        self.calls = 0
        self.dimensions = a.shape[1]

    def get(self, b):
        """Get."""
        self.calls += 1
        return np.full(self.dimensions, 1.0 if self.calls == 1 else 0.0)


def test_fixed_steps_mh_takes_exactly_three_steps_and_reproduces_pymc_forced_to_three(monkeypatch):
    """Check that fixed steps mh takes exactly three steps and reproduces pymc forced to three."""
    from pymc.smc import kernels

    from .vector_inference import MH_ESS_THRESHOLD, MH_STEPS_PER_STAGE, FixedStepsMH

    fixed, fixed_steps = _drive_kernel(FixedStepsMH, 12345, steps_per_stage=MH_STEPS_PER_STAGE)
    assert fixed_steps and set(fixed_steps) == {3}  # every stage, not just the first
    settings = fixed.sample_settings()
    assert settings["_n_tune"] == 3
    assert settings["threshold"] == MH_ESS_THRESHOLD == 0.5
    assert settings["steps_per_stage"] == 3
    # The dead knob is not recorded: this kernel never reads a correlation threshold.
    assert "correlation_threshold" not in settings

    monkeypatch.setattr(kernels, "Pearson", _ThreeStepPearson)
    reference, reference_steps = _drive_kernel(kernels.MH, 12345)
    # PyMC's counter starts at 1 and is incremented once per step, so three steps read as four.
    assert reference_steps == [step + 1 for step in fixed_steps]
    assert np.array_equal(
        fixed.tempered_posterior.view(np.uint64), reference.tempered_posterior.view(np.uint64)
    )
    assert np.array_equal(fixed.chain_acc_rate, reference.chain_acc_rate)
    assert np.array_equal(fixed.prior_logp, reference.prior_logp)
    assert np.array_equal(fixed.likelihood_logp, reference.likelihood_logp)


def test_fixed_steps_mh_refuses_a_step_count_that_is_not_a_positive_integer():
    """Check that fixed steps mh refuses a step count that is not a positive integer."""
    from .vector_inference import FixedStepsMH

    for bad in (0, -1, 2.5, True, "3"):
        with pytest.raises(ValueError, match="positive integer"):
            FixedStepsMH(draws=8, start=None, model=_toy_smc_model(), random_seed=1,
                         steps_per_stage=bad)


def test_kernel_policy_is_checked_and_imh_adaptive_passes_no_kernel_keyword():
    """Check that kernel policy is checked and imh adaptive passes no kernel keyword."""
    from .vector_inference import _kernel_options

    assert _kernel_options("imh_adaptive", "native") == {}
    assert set(_kernel_options("imh_adaptive", "batch")) == {"kernel"}
    options = _kernel_options("mh_fixed_steps", "native")
    assert options["kernel"].__name__ == "FixedStepsMH"
    assert options["threshold"] == 0.5 and options["steps_per_stage"] == 3
    assert _kernel_options("mh_fixed_steps", "batch")["kernel"].__name__ == "BatchFixedStepsMH"
    for bad in ("MH", "mh", "", None):
        with pytest.raises(ValueError, match="imh_adaptive or mh_fixed_steps"):
            _kernel_options(bad, "native")


def _linear_fit(**kwargs):
    def forward(hypothesis, theta, design):
        """Forward."""
        return np.array([theta[0] * design, theta[0] * design * 0.5])

    return fit(
        SimpleNamespace(model_id="linear_fixture"),
        [1.0, 2.0],
        [[1.1, 0.5], [2.0, 1.1]],
        forward,
        671,
        SMCSettings(32, 2, 2),
        scale=1.0,
        **kwargs,
    )


def test_explicit_imh_adaptive_is_the_unchanged_default_path():
    """Check that explicit imh adaptive is the unchanged default path."""
    default, explicit = _linear_fit(), _linear_fit(kernel_policy="imh_adaptive")
    for name in ("conductance", "sigma"):
        assert np.array_equal(default.particles[name], explicit.particles[name])
    assert default.log_evidence == explicit.log_evidence
    assert set(default.diagnostics) == set(explicit.diagnostics)
    assert "kernel_policy" not in default.diagnostics
    assert "steps_per_stage" not in default.diagnostics
    assert "ess_threshold" not in default.diagnostics
    assert default.diagnostics["sigma_unit_mV"] == 1.0


def test_mh_fixed_steps_fit_records_its_kernel_and_the_six_feature_noise():
    """Check that mh fixed steps fit records its kernel and the six feature noise."""
    fixed = _linear_fit(kernel_policy="mh_fixed_steps")
    baseline = _linear_fit()
    assert fixed.diagnostics["kernel_policy"] == "mh_fixed_steps"
    assert fixed.diagnostics["steps_per_stage"] == 3
    assert fixed.diagnostics["ess_threshold"] == 0.5
    assert fixed.diagnostics["sampler_attrs"]["_n_tune"] == 3
    assert fixed.diagnostics["sampler_attrs"]["threshold"] == 0.5
    assert "correlation_threshold" not in fixed.diagnostics["sampler_attrs"]
    # A different kernel is a different posterior; the point is that it ran, not that it agrees.
    assert set(fixed.diagnostics) == set(baseline.diagnostics) | {
        "kernel_policy", "steps_per_stage", "ess_threshold",
    }

    scaled = _linear_fit(feature_scales=[2.0, 5.0])
    assert scaled.diagnostics["feature_scales_mV_or_count"] == [2.0, 5.0]
    assert "sigma_unit_mV" not in scaled.diagnostics  # replaced, not left beside the scales
    assert np.all(scaled.particles["sigma"] > 0)
    for bad in ([1.0], [1.0, 0.0], [1.0, -2.0], [1.0, np.nan], [[1.0, 2.0]]):
        with pytest.raises(ValueError, match="one positive finite scale per feature"):
            _linear_fit(feature_scales=bad)


def test_mh_fixed_steps_fit_records_one_acceptance_rate_per_stage_and_chain():
    # PyMC's MH kernel reports `mean_accept_rate` where IMH reports `accept_rate`; the fit
    # must still record one acceptance rate per tempering stage for every chain.
    """Check that mh fixed steps fit records one acceptance rate per stage and chain."""
    fixed = _linear_fit(kernel_policy="mh_fixed_steps")
    rates, betas = fixed.diagnostics["accept_rate"], fixed.diagnostics["beta_schedule"]
    assert rates and len(rates) == len(betas)
    for chain_rates, chain_betas in zip(rates, betas, strict=True):
        assert len(chain_rates) == len(chain_betas) >= 1
        assert all(0.0 <= value <= 1.0 for value in chain_rates)
    baseline = _linear_fit()
    assert baseline.diagnostics["accept_rate"] and len(baseline.diagnostics["accept_rate"]) == len(
        baseline.diagnostics["beta_schedule"])


def test_per_feature_sigmas_leave_the_single_sigma_path_bit_identical():
    """Check that per feature sigmas leave the single sigma path bit identical."""
    from .vector_inference import joint_epig, particle_sigmas

    rng = np.random.default_rng(20260918)
    particles, designs, features = 23, 4, 3
    means = rng.normal(size=(particles, designs, features))
    sigma = rng.uniform(0.3, 2.0, particles)
    logweights = np.full(particles, -np.log(particles))
    flat = {"means": means, "sigmas": sigma, "logweights": logweights}
    # (K, 1) is the same scalar sigma written as one column; (K, F) repeats it per feature.
    column = {**flat, "sigmas": sigma[:, None]}
    spread = {**flat, "sigmas": np.repeat(sigma[:, None], features, axis=1)}
    y = rng.normal(size=(designs, features))

    assert particle_sigmas(sigma).shape == (particles, 1, 1)
    assert particle_sigmas(sigma[:, None]).shape == (particles, 1, 1)
    assert particle_sigmas(np.repeat(sigma[:, None], features, axis=1)).shape == (
        particles, 1, features)
    with pytest.raises(ValueError, match=r"\(K,\) or \(K, features\)"):
        particle_sigmas(means)

    for bank in (column, spread):
        assert np.array_equal(
            log_predictive(flat, y).view(np.uint64), log_predictive(bank, y).view(np.uint64)
        )
        assert joint_eig(flat, 91, samples=64) == joint_eig(bank, 91, samples=64)
        assert joint_epig(flat, flat, 91, samples=32) == joint_epig(bank, bank, 91, samples=32)
        reference, generalised = predictive_summary(flat), predictive_summary(bank)
        for key in reference:
            assert np.array_equal(
                reference[key].view(np.uint64), generalised[key].view(np.uint64)
            ), key


def _epig_banks(candidate_means, target_means, sigma=1.0):
    particles = len(candidate_means)
    logweights = np.full(particles, -np.log(particles))
    return (
        {"means": np.asarray(candidate_means, dtype=float),
         "sigmas": np.full(particles, float(sigma)), "logweights": logweights},
        {"means": np.asarray(target_means, dtype=float),
         "sigmas": np.full(particles, float(sigma)), "logweights": logweights},
    )


def test_joint_epig_is_zero_when_no_particle_disagrees():
    """Check that joint epig is zero when no particle disagrees."""
    from .vector_inference import joint_epig

    # Every particle predicts the same thing everywhere, so an outcome tells us nothing about
    # another outcome: the estimator must return exactly zero, not merely something small.
    candidate = np.tile([[[0.4], [-1.0]]], (6, 1, 1))
    target = np.tile([[[2.0], [0.1]]], (6, 1, 1))
    chosen, rows = joint_epig(*_epig_banks(candidate, target), 5)
    assert chosen in (0, 1)  # every candidate is worth the same; which one wins is arithmetic
    assert [row["design_index"] for row in rows] == [0, 1]
    for row in rows:
        assert row["policy"] == "epig"
        assert abs(row["epig"]) < 1e-9
        assert row["samples"] == 128 and row["targets"] == 2
        assert len(row["per_target"]) == 2 and all(abs(v) < 1e-9 for v in row["per_target"])

    # An exact tie - two candidates with the identical mean column, so the two estimates are
    # the same arithmetic on the same numbers - falls to the lowest design index.
    informative = (np.arange(6) * 50.0)[:, None, None]
    tied = np.concatenate([np.zeros((6, 1, 1)), informative, informative], axis=1)
    chosen, rows = joint_epig(*_epig_banks(tied, np.repeat(informative, 2, axis=1)), 5)
    assert rows[1]["epig"] == rows[2]["epig"] > rows[0]["epig"]
    assert chosen == 1


def test_joint_epig_recovers_the_component_entropy_of_a_separating_design():
    """Check that joint epig recovers the component entropy of a separating design."""
    from .vector_inference import joint_epig

    # Four well-separated components, unit noise. Candidate 0 cannot tell them apart; candidate
    # 1 resolves them exactly, and so do the targets, so the shared information is the
    # component entropy log 4 - which is also the ceiling for a K-component mixture.
    particles = 4
    candidate = np.zeros((particles, 2, 1))
    candidate[:, 1, 0] = np.arange(particles) * 100.0
    target = np.repeat((np.arange(particles) * 100.0)[:, None, None], 2, axis=1)
    chosen, rows = joint_epig(*_epig_banks(candidate, target), 11)
    assert chosen == 1
    assert abs(rows[0]["epig"]) < 1e-9
    assert 0 < rows[1]["epig"] <= np.log(particles) + 1e-9
    assert abs(rows[1]["epig"] - np.log(particles)) < 1e-6
    assert rows[1]["mc_standard_error"] >= 0


def test_joint_epig_uses_common_random_numbers_and_ignores_candidate_order():
    """Check that joint epig uses common random numbers and ignores candidate order."""
    from .vector_inference import joint_epig

    rng = np.random.default_rng(4)
    candidate = rng.normal(size=(9, 5, 2)) * 3
    target = rng.normal(size=(9, 3, 2)) * 3
    banks = _epig_banks(candidate, target)
    first = joint_epig(*banks, 77)
    assert first == joint_epig(*banks, 77)  # same seed, same numbers
    assert first != joint_epig(*banks, 78)

    order = [3, 0, 4, 1, 2]
    permuted = ({**banks[0], "means": candidate[:, order, :]}, banks[1])
    _, rows = joint_epig(*permuted, 77)
    reference = {row["design_index"]: row for row in first[1]}
    for index, row in enumerate(rows):
        original = reference[order[index]]
        assert row["epig"] == original["epig"]
        assert row["mc_standard_error"] == original["mc_standard_error"]
        assert row["per_target"] == original["per_target"]


def test_joint_epig_refuses_non_finite_values_and_mismatched_particle_clouds():
    """Check that joint epig refuses non finite values and mismatched particle clouds."""
    from .vector_inference import joint_epig

    candidate, target = _epig_banks(np.zeros((4, 2, 1)), np.ones((4, 3, 1)))
    broken = {**candidate, "means": candidate["means"].copy()}
    broken["means"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite joint EPIG"):
        joint_epig(broken, target, 1)
    with pytest.raises(ValueError, match="share one particle cloud"):
        joint_epig(candidate, {**target, "logweights": target["logweights"] + 0.5}, 1)
    with pytest.raises(ValueError, match="share one particle cloud"):
        joint_epig(candidate, {**target, "means": target["means"][:3],
                               "logweights": target["logweights"][:3]}, 1)
    with pytest.raises(ValueError, match="at least one candidate and one target"):
        joint_epig({**candidate, "means": candidate["means"][:, :0, :]}, target, 1)


@pytest.mark.usefixtures("require_neuronbench_source")
def test_declared_feature_scales_are_the_benchmarks_own():
    """Check that declared feature scales are the benchmarks own."""
    from neuronbench.evaluator import _EVAL_FEAT_SCALES

    from .neuron_forward import EVAL_FEATURE_NAMES, EVAL_FEATURE_SCALES

    # Repeated rather than imported (see the note beside the constant), so this is what keeps
    # the two copies from drifting: a package change must be a failure here, never a silent
    # change of the `paper_neuron` likelihood.
    assert list(EVAL_FEATURE_SCALES) == list(np.asarray(_EVAL_FEAT_SCALES, dtype=float))
    assert len(EVAL_FEATURE_NAMES) == len(EVAL_FEATURE_SCALES) == 6


@pytest.mark.usefixtures("require_neuronbench_source")
def test_eval_feature_vector_measures_the_observed_trace_with_the_packages_own_function():
    """Check that eval feature vector measures the observed trace with the packages own function."""
    import neuronbench as nb
    from neuronbench.evaluator import eval_features, true_mechanism
    from neuronbench.worlds import build_I, simulate

    from .neuron_forward import eval_feature_vector

    world = nb.load_world("h_sag", stochastic=False, seed=11)
    protocol = world.protocol_pool[0]
    observation = world.run(protocol)
    features = eval_feature_vector(observation.voltage, observation.test_start)
    assert features.shape == (6,)
    assert np.array_equal(
        features, eval_features(observation.voltage, observation.test_start)
    )

    # ... and the trace it measured really is the benchmark's sub-sampled observable: rebuild it
    # from the package's own integrator at the world's true mechanism and the same sub-sampling
    # `World.run` applies, then measure that.
    _, truth = true_mechanism("h_sag")
    current, test_start = build_I(protocol[1])
    _, voltage = simulate(current, test_start, **truth, trace=True)
    index = np.arange(0, len(voltage), nb._SUB)
    assert np.array_equal(observation.voltage, voltage[index].astype(float))
    assert np.array_equal(
        features, eval_features(voltage[index], int(np.searchsorted(index, test_start)))
    )
    # The full-resolution trace is a different measurement, which is why the sub-sampling is
    # part of the contract rather than an implementation detail.
    assert not np.array_equal(features, eval_features(voltage, test_start))

    for bad in ((np.zeros((2, 4)), 0), (np.zeros(4), 4), (np.zeros(4), -1)):
        with pytest.raises(ValueError, match="invalid voltage trace or test window"):
            eval_feature_vector(*bad)
