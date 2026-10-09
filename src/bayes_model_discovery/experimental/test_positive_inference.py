"""Small numerical examples for the public multivariate scalar API."""

import numpy as np
import pytest
from scipy.integrate import quad

from .count_inference import Fit, SMCSettings
from .positive_inference import (
    InputTransform,
    RegressionHypothesis,
    build_model,
    conditional_loglik,
    fit,
    initial_catalogue,
    joint_eig,
    log_predictive,
    particle_means,
    predictive_summary,
)


def example(family, feature=False):
    """Build a small example input for the tests below."""
    hypothesis = RegressionHypothesis(family, (1,) if feature else ())
    particles = {"b0": np.array([np.log(2.0)])}
    if feature:
        particles["b1"] = np.array([0.5])
    if family == "lognormal":
        particles["sigma"] = np.array([0.8])
    if family == "gamma":
        particles["shape"] = np.array([2.0])
    diagnostics = {
        "input_transform": {"center": (0.0, 0.0, 0.0), "scale": (1.0, 1.0, 1.0)},
        "response_unit": 3.0,
        "observation_count": 5,
        "public_data_sha256": "example",
    }
    return Fit(hypothesis, particles, 0.0, diagnostics=diagnostics)


@pytest.mark.parametrize("family", ("poisson", "lognormal", "gamma"))
def test_conditional_normalization_mean_and_pymc_logp(family):
    """Check that conditional normalization mean and pymc logp."""
    result = example(family)
    x = np.zeros((1, 3))
    assert particle_means(result, x)[0, 0] == pytest.approx(6.0)
    if family == "poisson":
        ys = np.arange(200.0)
        density = np.exp(conditional_loglik(result, x, ys[None, :]))[0, 0]
        mass, mean = density.sum(), density @ ys
    else:
        density = lambda y: np.exp(conditional_loglik(result, x, np.array([y])))[0, 0, 0]
        mass = quad(density, 0, np.inf, epsabs=1e-8)[0]
        mean = quad(lambda y: y * density(y), 0, np.inf, epsabs=1e-7)[0]
    assert mass == pytest.approx(1, abs=1e-7)
    assert mean == pytest.approx(6, abs=1e-6)
    transform = InputTransform(**result.diagnostics["input_transform"])
    y = np.array([2.0, 7.0])
    model = build_model(result.hypothesis, np.zeros((2, 3)), y, transform, 3.0)
    point = model.initial_point()
    point["b0"] = np.array(np.log(2.0))
    if family == "lognormal":
        point["sigma_log__"] = np.array(np.log(0.8))
    if family == "gamma":
        point["shape_log__"] = np.array(np.log(2.0))
    pm_value = model.compile_logp(vars=[model["observed"]])(point)
    np_value = conditional_loglik(result, np.zeros((2, 3)), y).sum()
    assert pm_value == pytest.approx(np_value, abs=1e-7)


def test_multivariate_inputs_predictive_noise_and_eig():
    """Check that multivariate inputs predictive noise and eig."""
    assert joint_eig.__defaults__ == (64,)
    x = np.array([[0.0, -1.0, 9.0], [4.0, 1.0, -2.0]])
    for family in ("poisson", "lognormal", "gamma"):
        result = example(family, feature=True)
        np.testing.assert_allclose(particle_means(result, x)[0], 6 * np.exp([-0.5, 0.5]))
        prediction = predictive_summary([result], x, 11, 2000)
        assert np.all(prediction["upper90"] > prediction["lower90"])
        assert prediction["measure"] == ("PMF" if family == "poisson" else "PDF")
        np.testing.assert_allclose(joint_eig([result], x, 11, 16)["eig"], 0, atol=1e-12)
    with pytest.raises(ValueError, match="PMF and PDF"):
        log_predictive([example("poisson"), example("gamma")], x, [1.0, 2.0])


def test_original_prior_support_and_public_d0_catalogue():
    """Check that original prior support and public d0 catalogue."""
    x = np.arange(20.0).reshape(5, 4)
    transform = InputTransform.from_public_pool(x)
    y = np.arange(1.0, 6.0)
    pool, policy = initial_catalogue(transform.apply(x), y, "poisson")
    assert len({h.model_id for h in pool}) == 4
    assert policy["test_labels_used"] == 0 and policy["public_observation_count"] == 5
    with pytest.raises(ValueError, match="integer"):
        build_model(pool[0], x, y + 0.1, transform, 3.0)
    with pytest.raises(ValueError, match="y > 0"):
        build_model(RegressionHypothesis("gamma"), x, y - 1, transform, 3.0)
    with pytest.raises(ValueError, match="response_unit"):
        build_model(pool[0], x, y, transform, 0.0)


@pytest.mark.parametrize("family", ("poisson", "lognormal", "gamma"))
def test_actual_smc_scalar_path(family):
    """Check that actual smc scalar path."""
    x = np.array(
        [[-1.0, 0.0, 0.0], [-0.5, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, -1.0, 0.0], [1.0, 0.0, -1.0]]
    )
    y = np.array([2.0, 3.0, 4.0, 6.0, 8.0])
    transform = InputTransform.from_public_pool(x)
    result = fit(
        RegressionHypothesis(family, (0,)), x, y, 912, transform, y.mean(), SMCSettings(32, 2, 2)
    )
    assert np.isfinite(result.log_evidence) and len(result.per_chain_log_evidence) == 2
    assert len(result.particles["b0"]) == 64
    predicted, _ = log_predictive([result], x, y)
    assert predicted.shape == (5, 1) and np.all(np.isfinite(predicted))
    acquisition = joint_eig([result], x, 19, 32)
    assert np.all(np.isfinite(acquisition["eig"]))
