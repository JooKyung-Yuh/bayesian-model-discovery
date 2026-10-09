"""Binary, real and zero-inflated-Poisson runner contract.

Five things are pinned here:

1. the paper's task table's binary row (logit / Bernoulli / no revisions) and
   real row (identity / Normal / Student-t nu=4 / heteroscedastic Normal) are
   transcribed into the menus, and the unbounded-count row's zero-inflated
   Poisson is gated by its own "only where zeros are natural" caveat;
2. every new family is NORMALIZED - the discrete ones sum to 1 over their
   support, the continuous ones integrate to 1 on a grid - and PyMC's own joint
   log density agrees with our scipy likelihood to 1e-7;
3. the sequential Monte Carlo (SMC) evidence of a Bernoulli and of a Normal model agrees with the
   corresponding low-dimensional quadrature;
4. the paper's prior table's rows are the priors actually used, including the
   pre-declaration that the heteroscedastic scale coefficients are the same
   "intercept" and "standardised slope" rows the mean coefficients use;
5. the held-out-set leakage invariance holds for the new packets, and our
   submitted predictions survive the environments' own parsers.

The environment tests skip when the read-only `boxing_gym` package is not on
PYTHONPATH, so the five MANIFEST checks can run without it.
"""

import numpy as np
import pytest
from scipy.integrate import dblquad, quad
from scipy.special import expit
from scipy.stats import beta as beta_dist
from scipy.stats import binom, halfnorm, norm, poisson
from scipy.stats import t as student_t_dist

from .belief_packet import (
    EXHAUSTIVE_COUNT_OUTCOMES,
    PoolMember,
    build_belief_packet,
    cell_constraints,
    interface_families,
    serialize_packet,
    task_interface,
)
from .count_inference import Fit, SMCSettings
from .count_inference import build_model as count_build_model
from .count_inference import fit as count_fit
from .count_inference import predict as count_predict
from .count_models import (
    BINARY_FAMILIES,
    BINARY_TRIALS,
    IMPLEMENTED_FAMILIES,
    OUTCOME_MENUS,
    PRIOR_RULES,
    Hypothesis,
    interface_input_names,
    particle_logpmf,
)
from .positive_inference import (
    SCALAR_FAMILIES,
    SCALAR_NOISE,
    ScalarHypothesis,
    conditional_loglik,
    mixture_moments,
    predictive_means,
)
from .positive_inference import build_model as scalar_build_model
from .positive_inference import fit as scalar_fit
from .real_inference import (
    HETEROSCEDASTIC_SCALE_PARAMETERS,
    REAL_FAMILIES,
    REAL_NOISE,
    REAL_NOISE_PRIORS,
    STUDENT_T_DF,
    RealHypothesis,
    conditional_cdf,
    default_scale_expression,
)

BINARY_INTERFACE = task_interface(
    "binary", [{"name": "u", "range": [0.0, 1.0]}], {"z": "2*u-1"}, (0, 1), BINARY_TRIALS
)
REAL_INTERFACE = task_interface(
    "real", [{"name": "u", "range": [0.0, 1.0]}], {"z": "2*u-1"}, (None, None)
)
COUNT_INTERFACE = task_interface(
    "unbounded_count", [{"name": "u", "range": [0.0, 1.0]}], {"z": "2*u-1"}, (0, None)
)
DESIGNS = np.array([0.0, 0.3, 1.0])


def make_fit(hypothesis, particles, interface, unit=1.0, center=0.0):
    """A Fit with only the bindings the predictive path reads; no SMC is run."""
    return Fit(
        hypothesis,
        particles,
        -1.0,
        [-1.0],
        {
            "interface": interface,
            "response_unit": float(unit),
            "response_center": float(center),
            "input_transform": None,
            "observation_count": 3,
            "public_data_sha256": "test",
        },
        0.0,
    )


# --------------------------------------------------------------------------
# 1. Table 2 / Table 3 transcription for the rows implemented here
# --------------------------------------------------------------------------
def test_table_2_binary_and_real_rows_are_transcribed_and_runnable():
    """The binary and real rows of the paper's family table are in the menus."""
    # "| Binary | logit | Bernoulli | none (probit/cloglog are link
    #  alternatives, not families; not counted as F) | |"
    assert BINARY_FAMILIES == ("bernoulli",)
    assert OUTCOME_MENUS["binary"] == {
        "initial_family": "bernoulli",
        "allowed_families": ("bernoulli",),
    }
    assert interface_families("binary", True) == ("bernoulli", ["bernoulli"])
    assert BINARY_TRIALS == 1
    # "| Real outcome | identity | Normal(mu,sigma) | Student-t (nu=4);
    #  heteroscedastic Normal, log sigma(x) = a0 + a1 z_j | ... |"
    assert REAL_FAMILIES == ("normal", "student_t", "heteroscedastic_normal")
    assert OUTCOME_MENUS["real"]["initial_family"] == "normal"
    assert interface_families("real", True)[1] == list(REAL_FAMILIES)
    # "| Unbounded count | log | Poisson(mu) | Negative Binomial(alpha);
    #  zero-inflated Poisson only where zeros are natural | |"
    assert "zi_poisson" in SCALAR_FAMILIES
    assert set(REAL_FAMILIES) | {"bernoulli", "zi_poisson"} <= set(IMPLEMENTED_FAMILIES)
    assert "binary" in EXHAUSTIVE_COUNT_OUTCOMES


def test_table_3_priors_of_every_new_parameter_are_the_declared_rows():
    """Priors of every new parameter equal the declared prior-table rows."""
    # "| Student-$t$ degrees of freedom | fixed at 4 |"
    assert STUDENT_T_DF == 4.0
    # "| zero-inflation $\pi$ | Beta$(1,4)$ |" - the same row the ZI-Binomial
    # already uses, so zi_poisson introduces no new prior value.
    assert PRIOR_RULES["pi"] == ("Beta", 1.0, 4.0)
    assert SCALAR_NOISE["zi_poisson"] == ("pi",)
    # "| output scale $\sigma$ (standardised outputs) | HalfNormal$(1)$ |"
    assert REAL_NOISE_PRIORS["sigma"] == ("HalfNormal", 1.0, None)
    assert REAL_NOISE["normal"] == REAL_NOISE["student_t"] == ("sigma",)
    # The pre-declaration for the heteroscedastic scale: Table 2 names a_0 and
    # a_1, Table 3 fixes "intercept | N(0,2)" and "standardised slope | N(0,1)",
    # and this change introduces NO other value for them.
    assert HETEROSCEDASTIC_SCALE_PARAMETERS == (("a0", "intercept"), ("a1", "signed"))
    assert REAL_NOISE_PRIORS["a0"] == PRIOR_RULES["intercept"] == ("Normal", 0.0, 2.0)
    assert REAL_NOISE_PRIORS["a1"] == PRIOR_RULES["signed"] == ("Normal", 0.0, 1.0)
    assert REAL_NOISE["heteroscedastic_normal"] == ("a0", "a1")


def test_the_new_priors_add_no_value_outside_the_two_tables():
    """The new priors introduce no value outside the two paper tables."""
    declared = {value for value in REAL_NOISE_PRIORS.values()}
    assert declared == {("HalfNormal", 1.0, None), ("Normal", 0.0, 2.0), ("Normal", 0.0, 1.0)}


# --------------------------------------------------------------------------
# 2. Per-family normalization
# --------------------------------------------------------------------------
def test_bernoulli_pmf_sums_to_one_and_is_the_binomial_with_one_trial():
    """The Bernoulli pmf sums to one and equals a one-trial binomial."""
    h = Hypothesis("b0+b1*z", "bernoulli", (("b0", "intercept"), ("b1", "signed")))
    particles = {"b0": np.array([-0.4, 0.2]), "b1": np.array([1.0, 1.5])}
    rows = particle_logpmf(h, particles, DESIGNS, BINARY_TRIALS, BINARY_INTERFACE)
    assert rows.shape == (2, 3, 2)
    np.testing.assert_allclose(np.exp(rows).sum(-1), 1.0, atol=1e-14)
    p = expit(particles["b0"][:, None] + particles["b1"][:, None] * (2 * DESIGNS - 1))
    np.testing.assert_allclose(np.exp(rows)[:, :, 1], p, atol=1e-14)
    np.testing.assert_allclose(np.exp(rows)[:, :, 0], 1 - p, atol=1e-14)
    assert h.noise_parameters == ()


def test_a_bernoulli_likelihood_refuses_more_than_one_trial():
    """A Bernoulli likelihood refuses more than one trial."""
    h = Hypothesis("b0", "bernoulli", (("b0", "intercept"),))
    with pytest.raises(ValueError, match="exactly one trial"):
        particle_logpmf(h, {"b0": np.array([0.1])}, DESIGNS, 50, BINARY_INTERFACE)
    with pytest.raises(ValueError, match="exactly one trial"):
        count_build_model(h, DESIGNS, np.array([0, 1, 0]), 50, BINARY_INTERFACE)


def test_zi_poisson_pmf_sums_to_one_and_carries_the_declared_moments():
    """The zero-inflated Poisson pmf sums to one and has the declared mean and variance."""
    h = ScalarHypothesis("b0+b1*z", "zi_poisson", (("b0", "intercept"), ("b1", "signed")))
    particles = {
        "b0": np.array([0.4, 1.1]),
        "b1": np.array([0.5, -0.3]),
        "pi": np.array([0.2, 0.45]),
    }
    result = make_fit(h, particles, COUNT_INTERFACE)
    support = np.arange(300)
    grid = np.broadcast_to(support, (len(DESIGNS), len(support)))
    table = np.exp(conditional_loglik(result, DESIGNS, grid))
    np.testing.assert_allclose(table.sum(-1), 1.0, atol=1e-10)
    mu = np.exp(particles["b0"][:, None] + particles["b1"][:, None] * (2 * DESIGNS - 1))
    pi = particles["pi"][:, None]
    # Table 2's ZI-Binomial note, applied symmetrically: the predictive mean is
    # (1 - pi) mu and is scored as such.
    np.testing.assert_allclose(predictive_means(result, DESIGNS), (1 - pi) * mu, rtol=1e-12)
    np.testing.assert_allclose(table @ support, (1 - pi) * mu, rtol=1e-9)
    mean, sd = mixture_moments(h, particles, DESIGNS, COUNT_INTERFACE)
    conditional, variance = (1 - pi) * mu, (1 - pi) * mu * (1 + pi * mu)
    np.testing.assert_allclose(mean, conditional.mean(0), rtol=1e-12)
    second = (variance + conditional**2).mean(0)
    np.testing.assert_allclose(sd, np.sqrt(second - mean**2), rtol=1e-12)


@pytest.mark.parametrize("family", REAL_FAMILIES)
def test_a_real_density_integrates_to_one_on_a_grid(family):
    """Each real-valued density integrates to one, by quadrature and on a grid."""
    h = RealHypothesis(
        "b0+b1*z",
        family,
        (("b0", "intercept"), ("b1", "signed")),
        ("u", "z"),
        "a0+a1*z" if family == "heteroscedastic_normal" else None,
    )
    particles = {
        "b0": np.array([-0.4, 0.6]),
        "b1": np.array([1.0, -0.5]),
        "sigma": np.array([0.7, 1.3]),
        "a0": np.array([-0.2, 0.3]),
        "a1": np.array([0.4, -0.1]),
    }
    result = make_fit(h, particles, REAL_INTERFACE, unit=2.0, center=5.0)
    # Exact normalization, per particle and per input, over the whole real line.
    for particle in range(2):
        for index in range(len(DESIGNS)):

            def density(value, particle=particle, index=index):
                """Conditional density at `value` for one particle and one input."""
                point = np.asarray([[value]])
                return float(
                    np.exp(conditional_loglik(result, DESIGNS[index : index + 1], point))[
                        particle, 0, 0
                    ]
                )

            total, error = quad(density, -np.inf, np.inf, limit=400)
            assert error < 1e-7
            assert total == pytest.approx(1.0, abs=1e-8)
    # And the same statement on a uniform grid, which is what the predictive
    # summaries and the EIG actually sample over.
    grid = np.broadcast_to(np.linspace(-60.0, 70.0, 200001), (len(DESIGNS), 200001))
    mass = np.trapz(np.exp(conditional_loglik(result, DESIGNS, grid)), grid[0], axis=-1)
    assert np.all(mass > 0.99) and np.all(mass <= 1.0 + 1e-9)


def real_fit(family):
    """One fitted real hypothesis of `family`, the shared fixture below."""
    hypothesis = RealHypothesis(
        "b0+b1*z",
        family,
        (("b0", "intercept"), ("b1", "signed")),
        ("u", "z"),
        "a0+a1*z" if family == "heteroscedastic_normal" else None,
    )
    particles = {
        "b0": np.array([-0.4, 0.6]),
        "b1": np.array([1.0, -0.5]),
        "sigma": np.array([0.7, 1.3]),
        "a0": np.array([-0.2, 0.3]),
        "a1": np.array([0.4, -0.1]),
    }
    return make_fit(hypothesis, particles, REAL_INTERFACE, unit=2.0, center=5.0)


@pytest.mark.parametrize("family", REAL_FAMILIES)
def test_the_conditional_cdf_is_the_integral_of_the_conditional_density(family):
    """`conditional_cdf` is `conditional_loglik`'s own law, integrated.

    The emotion Likert cells are now a difference of two of these values
    instead of a quadrature of that density (`boxinggym_truth_law`), so the two
    functions have to describe the same distribution: the CDF between two
    points is the density's integral there, and the two tails sum to 1.
    """
    result = real_fit(family)
    edges = np.broadcast_to(np.array([-4.0, 1.0, 5.0, 9.0, 14.0]), (len(DESIGNS), 5))
    lower = conditional_cdf(result, DESIGNS, edges)
    upper = conditional_cdf(result, DESIGNS, edges, "upper")
    np.testing.assert_allclose(lower + upper, 1.0, atol=1e-12)
    for start, stop in zip(edges[0, :-1], edges[0, 1:], strict=True):
        grid = np.broadcast_to(np.linspace(start, stop, 40001), (len(DESIGNS), 40001))
        integral = np.trapz(np.exp(conditional_loglik(result, DESIGNS, grid)), grid[0], axis=-1)
        index = int(np.flatnonzero(edges[0] == start)[0])
        np.testing.assert_allclose(
            lower[:, :, index + 1] - lower[:, :, index], integral, atol=1e-8
        )


def test_the_conditional_survival_keeps_resolution_where_one_minus_cdf_has_none():
    """Why `tail="upper"` exists: `1 - cdf` is exactly 0 far below a Normal."""
    result = real_fit("normal")
    far = np.broadcast_to(np.array([30.0]), (len(DESIGNS), 1))
    lower = conditional_cdf(result, DESIGNS, far)
    upper = conditional_cdf(result, DESIGNS, far, "upper")
    assert np.all(1.0 - lower == 0.0)
    assert np.all(upper > 0.0)
    with pytest.raises(ValueError):
        conditional_cdf(result, DESIGNS, far, "both")


def test_student_t_predictive_variance_is_twice_the_squared_scale_at_nu_4():
    """At nu = 4 the Student-t variance is twice the squared scale."""
    particles = {"b0": np.array([0.3]), "sigma": np.array([0.9])}
    moments = {}
    for family in ("normal", "student_t"):
        h = RealHypothesis("b0", family, (("b0", "intercept"),), ("u", "z"))
        moments[family] = mixture_moments(h, particles, DESIGNS, REAL_INTERFACE, 2.0, 5.0)
    np.testing.assert_allclose(moments["normal"][0], moments["student_t"][0], rtol=1e-12)
    # Table 2's own note: "sigma of the Student-t is its scale, so its variance
    # is 2 sigma^2 at nu = 4".
    np.testing.assert_allclose(
        moments["student_t"][1] ** 2, 2.0 * moments["normal"][1] ** 2, rtol=1e-12
    )
    np.testing.assert_allclose(moments["normal"][1], 2.0 * 0.9, rtol=1e-12)


def test_real_mixture_moments_match_scipy_moments():
    """Mixture moments of a heteroscedastic Normal match scipy's moments."""
    h = RealHypothesis(
        "b0+b1*z",
        "heteroscedastic_normal",
        (("b0", "intercept"), ("b1", "signed")),
        ("u", "z"),
        "a0+a1*z",
    )
    particles = {
        "b0": np.array([0.1, -0.7]),
        "b1": np.array([0.5, 0.2]),
        "a0": np.array([-0.3, 0.1]),
        "a1": np.array([0.2, -0.4]),
    }
    unit, center = 1.7, -0.5
    mean, sd = mixture_moments(h, particles, DESIGNS, REAL_INTERFACE, unit, center)
    z = 2 * DESIGNS - 1
    mu = center + unit * (particles["b0"][:, None] + particles["b1"][:, None] * z)
    scale = unit * np.exp(particles["a0"][:, None] + particles["a1"][:, None] * z)
    reference = [norm(loc=mu, scale=scale).mean(), norm(loc=mu, scale=scale).var()]
    np.testing.assert_allclose(mean, reference[0].mean(0), rtol=1e-10)
    second = (reference[1] + reference[0] ** 2).mean(0)
    np.testing.assert_allclose(sd, np.sqrt(second - mean**2), rtol=1e-10)


# --------------------------------------------------------------------------
# 3. PyMC's own joint log density against our scipy likelihood
# --------------------------------------------------------------------------
def test_bernoulli_pymc_joint_logp_matches_the_scipy_likelihood():
    """PyMC's joint log density of the Bernoulli model matches the scipy likelihood."""
    h = Hypothesis("b0+b1*z", "bernoulli", (("b0", "intercept"), ("b1", "signed")))
    particles = {"b0": np.array([-0.4]), "b1": np.array([1.0])}
    y = np.array([0, 1, 1])
    rows = particle_logpmf(h, particles, DESIGNS, BINARY_TRIALS, BINARY_INTERFACE)
    model = count_build_model(h, DESIGNS, y, BINARY_TRIALS, BINARY_INTERFACE)
    point = {"b0": np.array(-0.4), "b1": np.array(1.0)}
    prior = norm.logpdf(-0.4, 0, 2) + norm.logpdf(1.0, 0, 1)
    likelihood = rows[0, np.arange(3), y].sum()
    assert model.compile_logp()(point) == pytest.approx(likelihood + prior, abs=1e-7)


def test_zi_poisson_pymc_joint_logp_matches_the_scipy_likelihood():
    """PyMC's joint log density of the zero-inflated Poisson model matches the scipy likelihood."""
    h = ScalarHypothesis("b0+b1*z", "zi_poisson", (("b0", "intercept"), ("b1", "signed")))
    particles = {"b0": np.array([0.4]), "b1": np.array([0.5]), "pi": np.array([0.2])}
    y = np.array([0.0, 3.0, 7.0])
    result = make_fit(h, particles, COUNT_INTERFACE)
    likelihood = conditional_loglik(result, DESIGNS, y)[0, np.arange(3), 0].sum()
    model = scalar_build_model(h, DESIGNS, y, None, 1.0, COUNT_INTERFACE)
    point = {"b0": np.array(0.4), "b1": np.array(0.5), "pi": np.array(0.2)}
    prior = norm.logpdf(0.4, 0, 2) + norm.logpdf(0.5, 0, 1) + beta_dist.logpdf(0.2, 1, 4)
    assert model.compile_logp()(point) == pytest.approx(likelihood + prior, abs=1e-7)
    mu = np.exp(0.4 + 0.5 * (2 * DESIGNS - 1))
    manual = np.where(
        y == 0,
        np.logaddexp(poisson.logpmf(y, mu) + np.log1p(-0.2), np.log(0.2)),
        poisson.logpmf(y, mu) + np.log1p(-0.2),
    )
    assert manual.sum() == pytest.approx(likelihood, abs=1e-12)


@pytest.mark.parametrize("family", REAL_FAMILIES)
def test_real_pymc_joint_logp_matches_the_scipy_likelihood(family):
    """PyMC's joint log density of each real-valued model matches the scipy likelihood."""
    scale_expression = "a0+a1*z" if family == "heteroscedastic_normal" else None
    h = RealHypothesis(
        "b0+b1*z", family, (("b0", "intercept"), ("b1", "signed")), ("u", "z"), scale_expression
    )
    particles = {
        "b0": np.array([0.25]),
        "b1": np.array([-0.6]),
        "sigma": np.array([0.8]),
        "a0": np.array([-0.15]),
        "a1": np.array([0.35]),
    }
    unit, center = 1.3, 2.5
    y = np.array([1.0, 3.5, -0.25])
    result = make_fit(h, particles, REAL_INTERFACE, unit, center)
    likelihood = conditional_loglik(result, DESIGNS, y)[0, np.arange(3), 0].sum()
    model = scalar_build_model(h, DESIGNS, y, None, unit, REAL_INTERFACE, center)
    point = {"b0": np.array(0.25), "b1": np.array(-0.6)}
    prior = norm.logpdf(0.25, 0, 2) + norm.logpdf(-0.6, 0, 1)
    if family == "heteroscedastic_normal":
        point |= {"a0": np.array(-0.15), "a1": np.array(0.35)}
        prior += norm.logpdf(-0.15, 0, 2) + norm.logpdf(0.35, 0, 1)
    else:
        point |= {"sigma": np.array(0.8)}
        prior += halfnorm.logpdf(0.8, 0, 1)
    assert model.compile_logp()(point) == pytest.approx(likelihood + prior, abs=1e-6)
    z = 2 * DESIGNS - 1
    mu = center + unit * (0.25 - 0.6 * z)
    spread = unit * (np.exp(-0.15 + 0.35 * z) if scale_expression else 0.8)
    manual = (
        student_t_dist.logpdf(y, STUDENT_T_DF, loc=mu, scale=spread)
        if family == "student_t"
        else norm.logpdf(y, loc=mu, scale=spread)
    )
    assert manual.sum() == pytest.approx(likelihood, abs=1e-12)


# --------------------------------------------------------------------------
# 4. SMC against low-dimensional quadrature
# --------------------------------------------------------------------------
def test_bernoulli_smc_evidence_matches_the_one_dimensional_integral():
    """The SMC evidence of a Bernoulli model matches the one-dimensional integral."""
    h = Hypothesis("b0", "bernoulli", (("b0", "intercept"),))
    u, y = np.array([0.1, 0.7, 0.4, 0.9]), np.array([0, 1, 1, 1])
    exact, error = quad(
        lambda b: np.exp(binom.logpmf(y, 1, expit(b)).sum() + norm.logpdf(b, 0, 2)),
        -40,
        40,
        epsabs=1e-13,
    )
    assert error < 1e-9
    result = count_fit(h, u, y, 20260907, SMCSettings(256, 2, 2), BINARY_INTERFACE, BINARY_TRIALS)
    # A stochastic estimator check, not an equality claim or posterior tuning.
    assert result.log_evidence == pytest.approx(np.log(exact), abs=0.5)
    assert len(result.particles["b0"]) == 512
    # The exhaustive predictive table is the Bernoulli PMF and sums to one.
    prediction = count_predict([result], u, BINARY_INTERFACE, BINARY_TRIALS)
    np.testing.assert_allclose(np.exp(prediction["bma_logpmf"]).sum(-1), 1.0, atol=1e-12)


def test_normal_smc_evidence_matches_the_two_dimensional_integral():
    """The SMC evidence of a Normal model matches the two-dimensional integral."""
    h = RealHypothesis("b0", "normal", (("b0", "intercept"),), ("u", "z"))
    u, y = np.array([0.1, 0.7, 0.4]), np.array([0.4, -0.2, 0.9])
    exact, error = dblquad(
        lambda s, b: np.exp(
            norm.logpdf(y, b, s).sum() + norm.logpdf(b, 0, 2) + halfnorm.logpdf(s, 0, 1)
        ),
        -30,
        30,
        0,
        30,
        epsabs=1e-12,
    )
    assert error < 1e-8
    result = scalar_fit(
        h, u, y, 20260907, None, 1.0, SMCSettings(256, 2, 2), REAL_INTERFACE, 0.0
    )
    assert result.log_evidence == pytest.approx(np.log(exact), abs=0.5)
    assert result.diagnostics["response_center"] == 0.0
    assert set(result.particles) == {"b0", "sigma"}


def test_the_identity_link_is_centred_and_scaled_by_the_frozen_d0_calibration():
    """The identity link is centred and scaled by the frozen initial-data calibration."""
    h = RealHypothesis("b0", "normal", (("b0", "intercept"),), ("u", "z"))
    particles = {"b0": np.array([0.0, 1.0]), "sigma": np.array([1.0, 1.0])}
    result = make_fit(h, particles, REAL_INTERFACE, unit=3.0, center=10.0)
    np.testing.assert_allclose(
        predictive_means(result, DESIGNS), np.array([[10.0] * 3, [13.0] * 3]), rtol=1e-14
    )


# --------------------------------------------------------------------------
# 5. Identity, packets and the environments' own parsers
# --------------------------------------------------------------------------
def test_the_heteroscedastic_scale_expression_is_part_of_the_identity():
    """The heteroscedastic scale expression is part of the model identity."""
    common = ("b0", "heteroscedastic_normal", (("b0", "intercept"),), ("u", "z", "w", "wz"))
    a = RealHypothesis(*common, "a0+a1*z")
    b = RealHypothesis(*common, "a0+a1*wz")
    assert a.model_id != b.model_id
    assert a.identity_family == "heteroscedastic_normal|log_sigma=a0+a1*z"
    plain = RealHypothesis("b0", "normal", (("b0", "intercept"),), ("u", "z"))
    assert plain.model_id != a.model_id and plain.identity_family == "normal"
    with pytest.raises(ValueError, match="no scale expression"):
        RealHypothesis("b0", "normal", (("b0", "intercept"),), ("u", "z"), "a0+a1*z")
    with pytest.raises(ValueError, match="collides"):
        RealHypothesis(
            "a0+a1*z",
            "heteroscedastic_normal",
            (("a0", "intercept"), ("a1", "signed")),
            ("u", "z"),
            "a0+a1*z",
        )


def test_the_default_scale_expression_uses_the_first_standardized_input():
    """The default scale expression uses the first standardized input."""
    interface = task_interface(
        "real",
        [{"name": "x1", "range": [-4.0, 4.0]}, {"name": "x2", "range": [-4.0, 4.0]}],
        {"x101": "(x1+4.0)/8.0", "x1z": "2*(x1--4.0)/(4.0--4.0)-1", "x2z": "2*(x2--4.0)/8.0-1"},
        (None, None),
    )
    assert default_scale_expression(interface) == "a0+a1*x1z"
    assert default_scale_expression(None) == "a0+a1*z"


def adapter():
    """Import the BoxingGym adapter module, skipping the test when it is unavailable."""
    return pytest.importorskip(
        "bayes_model_discovery.experimental.boxinggym_core",
        reason="adapter imports the read-only boxing_gym package lazily",
    )


@pytest.mark.parametrize("task", ["hyperbolic", "location"])
def test_packet_is_byte_identical_when_only_the_heldout_set_changes(task):
    """Held-out-set invariance on the binary and real packets."""
    bg = adapter()
    spec = bg.TASKS[task]
    pytest.importorskip(spec.module)
    interface = spec.interface()
    namespace = interface_input_names(interface)
    structures = spec.initial_structures()
    initial_family, allowed = interface_families(spec.outcome_type, True)
    constraints = cell_constraints(
        "111", initial_family, allowed_families=allowed, initial_structures=list(structures)
    )
    rng = np.random.default_rng(21)
    members = []
    for index, form in enumerate(structures):
        parameters = bg.mock_parameters(form)
        if spec.outcome_type == "binary":
            hypothesis = Hypothesis(form, "bernoulli", parameters, namespace)
        else:
            hypothesis = RealHypothesis(form, "normal", parameters, namespace)
        particles = {name: rng.normal(size=16) for name, _ in parameters}
        particles |= {name: np.abs(rng.normal(size=16)) + 0.5 for name in hypothesis.noise_parameters}
        members.append(
            PoolMember(hypothesis, particles, -float(index + 3), response_unit=2.0,
                       response_center=0.5)
        )
    names = spec.input_names
    public = [
        {
            "obs_id": i,
            "round": 0 if i < 4 else i - 3,
            **{
                name: float(low + (high - low) * (0.1 + 0.15 * i))
                for name, (low, high) in zip(names, [tuple(x["range"]) for x in interface["inputs"]], strict=True)
            },
            "y": (i % 2) if spec.outcome_type == "binary" else 0.5 + 0.25 * i,
        }
        for i in range(5)
    ]
    archive = [
        {
            "model_id": m.hypothesis.model_id,
            "expression": m.hypothesis.expression,
            "family": m.hypothesis.family,
            "proposed_round": 0,
            "outcome": "active",
            "reason": {"admitted_round": 0, "active_rounds": [0, 1]},
        }
        for m in members
    ]
    texts = []
    environment = bg.TaskEnvironment(spec, 20260907)
    for count, seed in ((8, 111), (24, 222)):
        environment.goal.eval_points.clear()
        environment.goal.eval_pointer = 0
        environment.evaluation_points(count, seed)
        texts.append(
            serialize_packet(
                build_belief_packet(public, members, archive, constraints, interface, 1)
            )
        )
    assert texts[0] == texts[1]
    assert "eval" not in texts[0] and "heldout" not in texts[0]


@pytest.mark.parametrize("task", ["survival", "hyperbolic"])
def test_binary_predictions_survive_the_environment_parser(task):
    """Submission format: the evaluator's own parser, cited per task."""
    bg = adapter()
    spec = bg.TASKS[task]
    module = pytest.importorskip(spec.module)
    goal = getattr(module, spec.goal_class)
    # `evaluate_predictions` tests '"1" in p' BEFORE '"0" in p', so a decimal
    # string is misread; our submitted strings are exactly "0" and "1".
    submitted = bg.format_predictions(spec, np.array([0.2, 0.5, 0.7, 0.99]))
    assert submitted == ["0", "0", "1", "1"]
    parsed = [1 if "1" in p else 0 if "0" in p else None for p in submitted]
    assert parsed == [0, 0, 1, 1]
    accuracy, _ = goal.evaluate_predictions(goal, submitted, [0, 0, 1, 1])
    assert accuracy == 1.0
    # The trap this format avoids, recorded so it cannot regress silently.
    assert [1 if "1" in p else 0 for p in ["0.1", "1.0"]] == [1, 1]


def test_real_predictions_survive_the_environment_parser():
    """Real-valued predictions survive the environment's own parser."""
    bg = adapter()
    spec = bg.TASKS["location"]
    module = pytest.importorskip(spec.module)
    values = np.array([1.234567890123, -0.5, 12.0])
    submitted = bg.format_predictions(spec, values)
    assert [float(p) for p in submitted] == list(values)
    goal = module.DirectGoal
    mse, _ = goal.evaluate_predictions(goal, submitted, list(values))
    assert mse == 0.0


def test_every_new_task_design_pool_is_legal_and_d0_comes_from_it():
    """Every new task's design pool is legal and its initial design comes from that pool."""
    bg = adapter()
    for task in ("hyperbolic", "location"):
        spec = bg.TASKS[task]
        pytest.importorskip(spec.module)
        environment = bg.TaskEnvironment(spec, 20260907)
        pool = bg.build_design_pool(spec, environment, 16)
        assert pool.rows.shape[1] == len(spec.input_names)
        assert len(pool.rows) <= 16
        for row, handle in zip(pool.rows, pool.handles, strict=True):
            assert environment.accepts(handle), (task, row)
            assert pool.index_of(row) == list(pool.handles).index(handle)
        for index in pool.d0_indices():
            assert 0 <= index < len(pool.rows)
        assert len(set(pool.d0_indices())) == 4


def test_the_eval_tuple_reader_matches_each_environment_layout():
    """The evaluation tuple reader matches each environment's tuple layout."""
    bg = adapter()
    # survival_analysis.py:42 `(metastasized, time_since_surgery, outcome)`;
    # the declared input order is (time, metastasized).
    assert bg.TASKS["survival"].eval_inputs((1, 4.25, 0)) == [4.25, 1.0]
    # hyperbolic_temporal_discount.py:50 `(ir, dr, days, choice)`.
    assert bg.TASKS["hyperbolic"].eval_inputs((10, 200, 30, 1)) == [10.0, 200.0, 30.0]
    # location_finding.py:35 `(loc, choice)` with `loc` the whole 2-vector.
    assert bg.TASKS["location"].eval_inputs((np.array([0.5, -1.25]), 3.1)) == [0.5, -1.25]
    # death_process.py: the earlier single-input layout is unchanged.
    assert bg.TASKS["death"].eval_inputs((0.75, 12)) == [0.75]
