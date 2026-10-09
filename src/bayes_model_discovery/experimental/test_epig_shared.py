"""Independent finite-law identities and scalar quadrature checks for shared expected predictive information gain (EPIG)."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import logit, xlogy

from . import count_inference as inference
from .belief_packet import task_interface
from .count_inference import Fit
from .count_models import Hypothesis
from .positive_inference import RegressionHypothesis
from .real_inference import RealHypothesis


def two_latent_fit(target_latent):
    """Build a synthetic binomial fit with two latent bits and return it with its per-particle probabilities."""
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 0, 1])
    latent = b if target_latent == "b" else a
    probabilities = np.stack([.01 + .98*a, .2 + .6*b, .05 + .9*latent], axis=1)
    basis = np.column_stack([np.ones(3), [0., .5, 1.], np.square([0., .5, 1.])])
    coefficients = np.linalg.solve(basis, logit(probabilities).T).T
    hypothesis = Hypothesis("b0+b1*u+b2*square(u)", "binomial",
                            (("b0", "intercept"), ("b1", "signed"), ("b2", "signed")))
    return Fit(hypothesis, {f"b{i}": coefficients[:, i] for i in range(3)}, 0.), probabilities


def direct_information(probabilities):
    """Compute reference EPIG and EIG values directly from per-particle Bernoulli probabilities."""
    conditional = np.stack([1-probabilities, probabilities], axis=-1)
    epig, eig = [], []
    entropy = lambda p: -np.sum(xlogy(p, p))
    for index in (0, 1):
        joint = sum(np.outer(row[index], row[2])/len(conditional) for row in conditional)
        marginal_product = np.outer(joint.sum(1), joint.sum(0))
        epig.append(np.sum(xlogy(joint, joint/marginal_product)))
        eig.append(entropy(conditional[:, index].mean(0)) -
                   np.mean([entropy(row[index]) for row in conditional]))
    return np.array(epig), np.array(eig)


@pytest.mark.parametrize("target_latent,expected", [("b", .5), ("a", 0.)])
def test_exact_epig_same_and_different_choice_from_eig(target_latent, expected):
    """Check that exact epig same and different choice from eig."""
    result, probabilities = two_latent_fit(target_latent)
    targets = np.ones(inference.EPIG_TARGET_POINTS)
    selected, rows = inference.choose_epig([result], [0., .5], 41,
                                           n_trials=1, target_points=targets)
    eig_selected, eig_rows = inference.choose([result], [0., .5], 41, n_trials=1)
    epig, eig = direct_information(probabilities)
    np.testing.assert_allclose([row["epig"] for row in rows], epig, atol=1e-13)
    np.testing.assert_allclose([row["joint_eig"] for row in rows], eig, atol=1e-13)
    np.testing.assert_allclose([row["eig"] for row in eig_rows], eig, atol=1e-13)
    assert selected == expected and eig_selected == 0.
    assert np.all(epig >= -1e-14) and np.all(epig <= eig + 1e-14)
    if target_latent == "a":
        assert probabilities[:, 0].var() > probabilities[:, 1].var()
    assert inference.choose_epig([result], [0., .5], 41,
                                  n_trials=1, target_points=targets) == (selected, rows)


def test_single_parameter_particle_has_no_predictive_information():
    """Check that single parameter particle has no predictive information."""
    result, _ = two_latent_fit("b")
    result = replace(result, particles={key: value[:1] for key, value in result.particles.items()})
    _, rows = inference.choose_epig([result], [0., .5], 37,
                                    n_trials=1, target_points=np.ones(32))
    np.testing.assert_allclose([row["epig"] for row in rows], 0., atol=1e-14)
    np.testing.assert_allclose([row["joint_eig"] for row in rows], 0., atol=1e-14)


def test_particle_subsampling_matches_existing_eig_and_keeps_joint_pairing(monkeypatch):
    """Check that particle subsampling matches existing eig and keeps joint pairing."""
    result, _ = two_latent_fit("b")
    result = replace(result, particles={key: np.tile(value, 40) for key, value in result.particles.items()})
    original = inference.particle_logpmf
    calls = []
    def record(hypothesis, particles, inputs, n_trials, interface):
        """Record a copy of the particles and inputs, then delegate to the original log-pmf function."""
        calls.append(({key: value.copy() for key, value in particles.items()}, np.asarray(inputs).copy()))
        return original(hypothesis, particles, inputs, n_trials, interface)
    monkeypatch.setattr(inference, "particle_logpmf", record)
    inference.choose([result], [0., .5], 61, 128, n_trials=1)
    inference.choose_epig([result], [0., .5], 61, 128,
                          n_trials=1, target_points=np.ones(32))
    assert len(calls) == 2
    for key in result.particles:
        np.testing.assert_array_equal(calls[0][0][key], calls[1][0][key])
    np.testing.assert_array_equal(calls[1][1], np.r_[0., .5, np.ones(32)])


def scalar_fit(family="normal", values=(-1., 1.), evidence=0.):
    """Build a synthetic one-input scalar fit for the given response family."""
    values = np.asarray(values)
    if family in ("normal", "student_t"):
        hypothesis = RealHypothesis("b0+b1*u", family, (("b0", "intercept"), ("b1", "signed")))
        particles = {"b0": values, "b1": values.copy(), "sigma": np.full(len(values), .7)}
        interface = task_interface("real", [{"name": "u", "range": [0., 1.]}],
                                   {"z": "2*u-1"}, (None, None))
        diagnostics = {"interface": interface, "response_unit": 1., "response_center": 0.,
                       "input_transform": None, "observation_count": 4, "public_data_sha256": "synthetic"}
    else:
        hypothesis = RegressionHypothesis(family)
        particles = {"b0": values}
        particles["shape" if family == "gamma" else "sigma"] = np.full(len(values), 2. if family == "gamma" else .7)
        diagnostics = {"response_unit": 1., "input_transform": {"center": (0.,), "scale": (1.,)},
                       "observation_count": 4, "public_data_sha256": "synthetic"}
    return Fit(hypothesis, particles, evidence, diagnostics=diagnostics)


@pytest.mark.parametrize("family", ["normal", "student_t", "lognormal", "gamma"])
def test_scalar_grid_is_64_normalized_quantile_events_with_tails(family):
    """Check that scalar grid is 64 normalized quantile events with tails."""
    result = scalar_fit(family)
    inputs = np.array([[0.], [.5], [1.]])
    weights = np.full(2, .5)
    bank = inference._epig_continuous_bank([result], inputs, weights, 64)
    assert bank.shape == (2, 3, 64)
    assert np.all(np.isfinite(bank)) and np.all(bank >= 0)
    np.testing.assert_allclose(bank.sum(-1), 1., atol=1e-13)
    np.testing.assert_allclose(np.einsum("k,kxy->xy", weights, bank), 1/64, atol=1e-12)
    _, rows = inference.choose_epig([result], inputs, 41, target_points=np.ones((32, 1)))
    assert all(-1e-12 <= row["epig"] <= row["joint_eig"]+1e-12 for row in rows)


def test_bma_quantile_grid_uses_declared_model_weights_and_common_edges():
    """Check that bma quantile grid uses declared model weights and common edges."""
    fits = [scalar_fit("normal", (-2.,), np.log(.2)), scalar_fit("student_t", (2.,), np.log(.8))]
    bank = inference._epig_continuous_bank(fits, np.array([[0.], [1.]]), np.array([.2, .8]), 64)
    np.testing.assert_allclose(np.einsum("k,kxy->xy", [.2, .8], bank), 1/64, atol=1e-12)
    assert not np.allclose(bank[0], bank[1])


@pytest.mark.parametrize("cell", ["002", "012", "102", "112"])
def test_epig_cell_resolves_without_changing_three_character_identity(cell):
    """Check that epig cell resolves without changing three character identity."""
    assert inference.resolve_epig_policy(cell) == "epig"
    assert inference.resolve_epig_policy(cell, "epig") == "epig"
    with pytest.raises(ValueError, match="requires"):
        inference.resolve_epig_policy(cell, "joint_eig")


@pytest.mark.parametrize("cell", ["000", "001", "100", "101"])
def test_existing_cells_keep_default_policy_and_reject_epig_mismatch(cell):
    """Check that existing cells keep default policy and reject epig mismatch."""
    assert inference.resolve_epig_policy(cell) == "joint_eig"
    with pytest.raises(ValueError, match="requires"):
        inference.resolve_epig_policy(cell, "epig")


@pytest.mark.parametrize("cell", ["103", "12", "1002", "202", "abc", 102])
def test_invalid_cell_syntax_is_rejected(cell):
    """Check that invalid cell syntax is rejected."""
    with pytest.raises(ValueError, match="three-character"):
        inference.resolve_epig_policy(cell)


def test_config_records_fixed_targets_measure_grid_seed_and_no_redraw():
    """Check that config records fixed targets measure grid seed and no redraw."""
    targets = np.random.default_rng(42).uniform(size=(32, 3))
    record = inference.epig_config(targets, 42, "synthetic_public_box_uniform")
    assert record["design_policy"] == "epig" and record["design_policy_target"] == "predictive_target_points"
    assert record["epig_target_points"] == 32 and record["epig_y_grid"] == 64
    assert record["epig_targets_seed"] == 42 and record["epig_targets_seed_stream"] == "epig_targets"
    assert record["epig_targets"] == targets.tolist() and record["epig_units"] == "nats"
    assert record["epig_target_redraw"] == "once_at_run_start"
    assert inference.epig_config(targets, 42, "synthetic_public_box_uniform") == record
    with pytest.raises(ValueError, match="32 finite"):
        inference.epig_config(targets[:-1], 42, "synthetic_public_box_uniform")


def test_real_vector_inputs_keep_particle_and_input_axes_separate():
    """Check that real vector inputs keep particle and input axes separate."""
    hypothesis = RealHypothesis("b0+b1*u+b2*v", "normal",
                                (("b0", "intercept"), ("b1", "signed"), ("b2", "signed")),
                                input_names=("u", "v", "z"))
    interface = task_interface("real", [{"name": "u", "range": [0., 1.]},
                                        {"name": "v", "range": [0., 1.]}],
                               {"z": "2*u-1"}, (None, None))
    result = Fit(hypothesis, {"b0": np.array([-.5, .5]), "b1": np.array([-.8, .8]),
                              "b2": np.array([.4, -.4]), "sigma": np.array([.6, .6])}, 0.,
                 diagnostics={"interface": interface, "response_unit": 1., "response_center": 0.,
                              "input_transform": None, "observation_count": 4,
                              "public_data_sha256": "synthetic"})
    chosen, rows = inference.choose_epig([result], [[0., 0.], [.8, .3]], 21,
                                        target_points=np.tile([.9, .1], (32, 1)))
    assert chosen in ([0., 0.], [.8, .3])
    assert all(np.isfinite(row["epig"]) and row["epig"] <= row["joint_eig"]+1e-12 for row in rows)


def test_continuous_epig_refinement_obeys_information_coarsening():
    """Check that continuous epig refinement obeys information coarsening."""
    result = scalar_fit("normal")
    inputs = np.array([0., 1.])
    targets = np.ones(32)
    _, coarse = inference.choose_epig([result], inputs, 17, target_points=targets, y_grid=64)
    _, refined = inference.choose_epig([result], inputs, 17, target_points=targets, y_grid=128)
    # BMA quantile boundaries at multiples of 1/64 are also 128-bin boundaries.
    # Both future outcomes are coarsened, so refinement cannot lose information.
    for left, right in zip(coarse, refined, strict=True):
        assert left["epig"] <= right["epig"] + 1e-12
        assert right["epig"] <= right["joint_eig"] + 1e-12


def test_positive_grammar_singleton_vector_matches_scalar_design_encoding():
    """Check that positive grammar singleton vector matches scalar design encoding."""
    from .positive_inference import ScalarHypothesis
    hypothesis = ScalarHypothesis("b0+b1*u", "gamma", (("b0", "intercept"), ("b1", "signed")))
    interface = task_interface("positive", [{"name": "u", "range": [0., 1.]}],
                               {"z": "2*u-1"}, (0., None))
    result = Fit(hypothesis, {"b0": np.array([-.2, .2]), "b1": np.array([-.4, .4]),
                              "shape": np.array([2., 2.])}, 0.,
                 diagnostics={"interface": interface, "response_unit": 1., "input_transform": None,
                              "observation_count": 4, "public_data_sha256": "synthetic"})
    _, scalar = inference.choose_epig([result], [0., 1.], 31, target_points=np.ones(32))
    _, vector = inference.choose_epig([result], [[0.], [1.]], 31, target_points=np.ones((32, 1)))
    np.testing.assert_allclose([row["epig"] for row in scalar], [row["epig"] for row in vector], atol=1e-13)
