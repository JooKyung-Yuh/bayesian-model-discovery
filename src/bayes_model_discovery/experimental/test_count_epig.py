"""Acceptance is an information identity, not a model/strategy performance claim."""

import numpy as np

from .count_epig import predictive_information, uniform_quadrature


def test_single_particle_concentration_has_zero_predictive_information():
    """Concentrating on a single particle gives zero predictive information."""
    q = np.array([[[0.2, 0.8], [0.6, 0.4]], [[0.9, 0.1], [0.1, 0.9]]])
    x = np.array([[[0.3, 0.7], [0.4, 0.6]], [[0.1, 0.9], [0.8, 0.2]]])
    result = predictive_information(q, x, [1.0, 0.0], [0.5, 0.5])
    np.testing.assert_allclose(result["epig"], 0, atol=1e-14)
    np.testing.assert_allclose(result["joint_eig"], 0, atol=1e-14)


def test_predictive_information_is_nonnegative_and_retains_particle_dependence():
    """Predictive information is positive when particles differ and zero for mixed marginals."""
    bank = np.array([[[0.99, 0.01]], [[0.01, 0.99]]])
    result = predictive_information(bank, bank, [0.5, 0.5], [1.0])
    assert result["epig"][0] > 0.5
    # A product of mixed marginals would be independent and incorrectly give zero.
    mixed = bank.mean(0, keepdims=True)
    wrong = predictive_information(mixed, mixed, [1.0], [1.0])
    np.testing.assert_allclose(wrong["epig"], 0, atol=1e-14)


def test_epig_does_not_exceed_joint_eig_and_fixed_quadrature_is_normalized():
    """EPIG is nonnegative and at most the joint EIG, and the fixed quadrature is normalized."""
    rng = np.random.default_rng(7301)
    q = rng.dirichlet(np.ones(5), size=(7, 6))
    x = rng.dirichlet(np.ones(5), size=(7, 8))
    w = rng.dirichlet(np.ones(7))
    result = predictive_information(q, x, w, np.full(6, 1 / 6))
    assert np.all(result["epig"] >= -1e-12)
    assert np.all(result["epig"] <= result["joint_eig"] + 1e-12)
    for count in (32, 128):
        nodes, weights = uniform_quadrature(count)
        np.testing.assert_allclose(
            [weights.sum(), weights @ nodes, weights @ nodes**2], [1.0, 0.5, 1 / 3], atol=1e-14
        )
