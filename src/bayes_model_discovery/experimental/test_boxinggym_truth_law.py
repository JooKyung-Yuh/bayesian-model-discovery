"""The per-task true observation laws and the Kullback-Leibler divergence (KL) from the truth to the
Bayesian model averaging (BMA) predictive, KL(truth || BMA), on them.

Every check here is analytic and synthetic - no environment, no fit, no paid
call. The pool's predictive log-density is injected (the pattern
`chembench_core.censored_normal_kl_quadrature`'s own docstring allows: "a test
may inject any callable here - a closed-form comparator"), so each family's KL
is compared against a value written out by hand:

  * binomial / Bernoulli / Poisson / categorical: the exact discrete sum;
  * Normal: log(s'/s) + (s^2 + (m - m')^2) / (2 s'^2) - 1/2;
  * folded Normal with m >> s: the same Normal formula, since the reflected
    term is exp(-2 (m/s)^2) - here below 1e-80.

and every family is additionally checked at truth == model, where the KL must
be 0 to within the recorded quadrature error.
"""

import numpy as np
import pytest
from scipy.special import xlogy
from scipy.stats import binom, norm, poisson

from . import boxinggym_truth_law as law_module
from .boxinggym_truth_law import (
    EMOTION_CELL_EDGES,
    bernoulli_law,
    bernoulli_truth_mean_law,
    death_law,
    discrete_kl,
    folded_normal_law,
    kl_against_pool,
    likert_cells_law,
    normal_law,
    poisson_law,
    subset_law,
    truth_law,
)

FITS = ("one-model pool sentinel",)


class Spec:
    """The two TaskSpec fields the builders read."""

    def __init__(self, truth_mean=None, truth_law=None):
        self.truth_mean = truth_mean
        self.truth_law = truth_law


class Environment:
    """Minimal environment exposing the given attributes as `env`."""
    def __init__(self, **attributes):
        self.env = type("Env", (), attributes)()


def inject(monkeypatch, density):
    """Make the pool's BMA log-density `density(y)`, evaluated element-wise."""
    monkeypatch.setattr(law_module, "particle_count", lambda result: 64)
    monkeypatch.setattr(
        law_module, "log_predictive",
        lambda fits, inputs, outcomes: (density(np.asarray(outcomes, dtype=float)), None),
    )


def inject_cdf(monkeypatch, cumulative):
    """Make the pool one unit-weight member with CDF/survival `cumulative(y, tail)`.

    The emotion path (the later exact-cell revision) asks the pool for a MASS, not a density: the
    two injected callables are `positive_inference.validate_pool`'s log weights
    and `real_inference.conditional_cdf`'s [particle, input, draw] values, the
    single particle standing for the mean over particles.
    """
    monkeypatch.setattr(law_module, "validate_pool", lambda fits: np.zeros(len(fits)))
    monkeypatch.setattr(
        law_module, "conditional_cdf",
        lambda fit, inputs, outcomes, tail="lower": cumulative(
            np.asarray(outcomes, dtype=float), tail
        )[None, :, :],
    )


def normal_pool(mean, sigma):
    """One Normal component, as the log density AND the CDF/survival pair."""
    def density(y):
        """Log density of the pool's Normal component."""
        return norm.logpdf(y, mean[:, None], sigma)

    def cumulative(y, tail):
        """CDF (or survival function for tail='upper') of the pool's Normal component."""
        if tail == "upper":
            return norm.sf(y, mean[:, None], sigma)
        return norm.cdf(y, mean[:, None], sigma)

    return density, cumulative


def hand_written_cells(cumulative, queries):
    """The nine Likert cell masses of that pool, written out from its own CDF.

    `sf` above the median and never `1 - cdf`: the two expressions are the same
    number in exact arithmetic, but a difference of CDFs that both round to 1
    is 0 where the mass is merely very small, which would make the reference KL
    infinite (the earlier quadrature test wrote the top cell this way for the same reason).
    """
    edges = np.tile(np.asarray(EMOTION_CELL_EDGES, dtype=float), (queries, 1))
    below, above = cumulative(edges, "lower"), cumulative(edges, "upper")
    interior = np.where(
        below[:, 1:] < 0.5, np.diff(below, axis=1), -np.diff(above, axis=1)
    )
    return np.concatenate([below[:, :1], interior, above[:, -1:]], axis=1)


def categorical_kl(truth, cells):
    """mean_Q sum_k p log(p/q), the same two lines `discrete_kl` is."""
    return float(np.sum(xlogy(truth, truth) - truth * np.log(cells), axis=-1).mean())


def quadrature_cell_kl(law, inputs, component, nodes):
    """The retired quadrature rule, still reachable (private) for the comparison test."""
    log_model = law_module._cell_log_model(FITS, inputs, law, component, nodes)
    return law_module.discrete_kl(law["probability"][:, component, :], log_model), np.exp(log_model)


def normal_kl(mean, sigma, model_mean, model_sigma):
    """The closed-form KL(Normal(mean, sigma) || Normal(model_mean, model_sigma))."""
    return float(np.mean(
        np.log(model_sigma / sigma)
        + (sigma**2 + (mean - model_mean) ** 2) / (2 * model_sigma**2)
        - 0.5
    ))


# --------------------------------------------------------------------------
# death: the new builder reproduces the earlier array and its two numbers
# --------------------------------------------------------------------------


def test_death_law_is_the_same_array_as_death_truth_pmf():
    """The death law is the same array as the runner's death truth pmf."""
    boxinggym_core = pytest.importorskip("bayes_model_discovery.experimental.boxinggym_core")
    environment = Environment(N=50, theta=0.73)
    times = np.array([0.001, 0.4, 1.1, 1.999])
    built = death_law(Spec(), environment, times)
    np.testing.assert_array_equal(built["pmf"], boxinggym_core.death_truth_pmf(environment, times))
    assert built["pmf"].shape == (4, 51)
    np.testing.assert_allclose(built["pmf"].sum(axis=-1), 1.0, atol=1e-12)


def test_discrete_kl_is_score_prediction_bit_for_bit_on_deaths_own_arrays():
    """`kl`/`expected_log_score` must be the SAME float, not merely close."""
    core_count = pytest.importorskip("bayes_model_discovery.experimental.core_count")
    environment = Environment(N=50, theta=0.41)
    times = np.array([0.2, 0.9, 1.5])
    truth = death_law(Spec(), environment, times)["pmf"]
    logpmf = np.log(binom.pmf(np.arange(51)[None, :], 50, 0.3))
    logpmf = np.repeat(logpmf, len(times), axis=0)
    reference, _ = core_count.score_prediction(
        logpmf, np.zeros(len(times), dtype=int), truth, np.random.default_rng(7)
    )
    mine = discrete_kl(truth, logpmf)
    assert mine["kl"] == reference["kl"]
    assert mine["expected_log_score"] == reference["expected_log_score"]


def test_score_prediction_cannot_be_called_for_a_two_column_truth():
    """Why `discrete_kl` exists: `mean_rmse` hard-codes `np.arange(51)`."""
    core_count = pytest.importorskip("bayes_model_discovery.experimental.core_count")
    truth = bernoulli_law([0.3, 0.6])["pmf"]
    with pytest.raises(ValueError):
        core_count.score_prediction(
            np.log(truth), np.zeros(2, dtype=int), truth, np.random.default_rng(1)
        )


# --------------------------------------------------------------------------
# the four binary tasks
# --------------------------------------------------------------------------


def test_bernoulli_law_sums_to_one_and_matches_the_hand_written_kl():
    """The Bernoulli law sums to one and its KL equals the hand-written value."""
    p = np.array([0.05, 0.5, 0.92])
    q = np.array([0.2, 0.4, 0.7])
    built = bernoulli_law(p)
    np.testing.assert_allclose(built["pmf"].sum(axis=-1), 1.0, atol=1e-15)
    np.testing.assert_allclose(built["pmf"][:, 1], p)
    logpmf = np.log(np.column_stack([1 - q, q]))
    scores = kl_against_pool(built, {"logpmf": logpmf})
    closed = np.mean(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))
    assert scores["kl"] == pytest.approx(closed, abs=1e-12)
    assert scores["expected_log_score"] == pytest.approx(
        float(np.mean(p * np.log(q) + (1 - p) * np.log(1 - q))), abs=1e-12
    )
    assert scores["kl_method"] == "discrete_exact"
    assert "kl_truncated_mass" not in scores and "kl_quadrature_error" not in scores


def test_bernoulli_kl_is_zero_when_the_pool_is_the_truth():
    """The Bernoulli KL is zero when the pool equals the truth."""
    p = np.array([0.1, 0.35, 0.8])
    built = bernoulli_law(p)
    scores = kl_against_pool(built, {"logpmf": np.log(built["pmf"])})
    assert abs(scores["kl"]) <= 1e-9


def test_binary_builder_reuses_truth_mean_and_passes_none_through():
    """The binary builder reuses the truth mean and passes None through."""
    spec = Spec(truth_mean=lambda environment, x: np.asarray(x, dtype=float) / 10.0)
    built = bernoulli_truth_mean_law(spec, None, [1.0, 5.0])
    np.testing.assert_allclose(built["pmf"][:, 1], [0.1, 0.5])
    assert bernoulli_truth_mean_law(Spec(truth_mean=lambda e, x: None), None, [1.0]) is None


# --------------------------------------------------------------------------
# peregrines: enumeration depth, truncated mass, closed-form Poisson KL
# --------------------------------------------------------------------------


def test_poisson_law_enumerates_to_the_declared_tail_and_sums_to_one():
    """The Poisson law enumerates to the declared tail and sums to one."""
    rate = np.array([3.0, 40.0, 250.0])
    built = poisson_law(rate)
    assert built["kind"] == "count_grid"
    np.testing.assert_allclose(built["pmf"].sum(axis=-1), 1.0, atol=1e-11)
    assert float(np.max(built["excluded"])) < 1e-10
    assert built["grid"][0] == 0.0 and built["grid"][-1] == len(built["grid"]) - 1


def test_poisson_kl_matches_the_closed_form_and_vanishes_at_truth(monkeypatch):
    """The Poisson KL matches the closed form and vanishes at the truth."""
    rate = np.array([3.0, 12.0])
    model = np.array([5.0, 12.0])
    built = poisson_law(rate)
    inputs = np.zeros((2, 1))
    inject(monkeypatch, lambda y: poisson.logpmf(y, model[:, None]))
    scores = kl_against_pool(built, FITS, inputs)
    closed = float(np.mean(model - rate + rate * np.log(rate / model)))
    assert scores["kl"] == pytest.approx(closed, abs=1e-11)
    assert scores["kl_truncated_mass"] < 1e-10
    inject(monkeypatch, lambda y: poisson.logpmf(y, rate[:, None]))
    assert abs(kl_against_pool(built, FITS, inputs)["kl"]) <= 1e-9


def test_blocked_log_predictive_equals_one_call(monkeypatch):
    """Blocked evaluation of the log predictive gives the same KL as a single call."""
    rate = np.array([7.0, 9.0])
    built = poisson_law(rate)
    inputs = np.zeros((2, 1))
    inject(monkeypatch, lambda y: poisson.logpmf(y, 8.0))
    whole = kl_against_pool(built, FITS, inputs)["kl"]
    monkeypatch.setattr(law_module, "KL_PREDICTIVE_BLOCK_CELLS", 512)
    assert kl_against_pool(built, FITS, inputs)["kl"] == whole


# --------------------------------------------------------------------------
# dugongs / location: quadrature against the closed-form Normal KL
# --------------------------------------------------------------------------


def test_normal_law_kl_matches_the_closed_form(monkeypatch):
    """The Normal-law KL matches the closed form."""
    mean, sigma = np.array([1.5, -0.4, 3.0]), 0.5
    model_mean, model_sigma = mean + 0.3, 0.8
    built = normal_law(mean, sigma)
    inject(monkeypatch, lambda y: norm.logpdf(y, model_mean[:, None], model_sigma))
    scores = kl_against_pool(built, FITS, mean[:, None])
    closed = normal_kl(mean, sigma, model_mean, model_sigma)
    assert scores["kl"] == pytest.approx(closed, abs=max(1e-9, scores["kl_quadrature_error"]))
    assert scores["kl_truncated_mass"] < 1e-14
    assert scores["kl_quadrature_error"] < 1e-9
    assert "expected_log_score" not in scores


def test_normal_law_kl_vanishes_at_truth(monkeypatch):
    """The Normal-law KL vanishes at the truth."""
    mean, sigma = np.array([0.0, 2.2]), 0.5
    inject(monkeypatch, lambda y: norm.logpdf(y, mean[:, None], sigma))
    scores = kl_against_pool(normal_law(mean, sigma), FITS, mean[:, None])
    assert abs(scores["kl"]) <= 1e-9


def test_folded_normal_kl_matches_the_normal_closed_form_far_from_zero(monkeypatch):
    """m/s = 20 here, so the reflected term is exp(-800) and the law is Normal."""
    mean, sigma = np.array([5.0, 4.0]), 0.25
    model_mean, model_sigma = mean + 0.1, 0.4
    built = folded_normal_law(mean, sigma)
    inject(monkeypatch, lambda y: norm.logpdf(y, model_mean[:, None], model_sigma))
    scores = kl_against_pool(built, FITS, mean[:, None])
    closed = normal_kl(mean, sigma, model_mean, model_sigma)
    assert scores["kl"] == pytest.approx(closed, abs=max(1e-9, scores["kl_quadrature_error"]))
    assert scores["kl_truncated_mass"] < 1e-14


def test_folded_normal_kl_vanishes_at_truth_even_against_zero(monkeypatch):
    """A location parameter inside one sigma of 0: both reflections matter."""
    mean, sigma = np.array([0.2, 0.05]), 0.25

    def folded(y):
        """Log density of the folded Normal law (both reflections)."""
        return np.logaddexp(
            norm.logpdf(y, mean[:, None], sigma), norm.logpdf(y, -mean[:, None], sigma)
        )

    inject(monkeypatch, folded)
    scores = kl_against_pool(folded_normal_law(mean, sigma), FITS, mean[:, None])
    assert abs(scores["kl"]) <= 1e-9
    assert scores["kl_truncated_mass"] < 1e-13


# --------------------------------------------------------------------------
# emotion: nine cells, cell integration, the composite's per-component law
# --------------------------------------------------------------------------


def test_likert_cells_sum_to_one_and_follow_the_int_clip_coding():
    """Likert cell probabilities sum to one and follow the integer-clip coding."""
    mean = np.array([[4.4, 1.2], [8.9, 5.0]])
    sigma = np.array([0.7, 1.3])
    built = likert_cells_law(mean, sigma)
    assert built["probability"].shape == (2, 2, 9)
    np.testing.assert_allclose(built["probability"].sum(axis=-1), 1.0, atol=1e-12)
    assert np.all(built["probability"] >= 0.0)
    # cell 1 is everything below 2 (int() truncation then clip to 1) and cell 9
    # everything from 9 up, written out independently of the builder.
    np.testing.assert_allclose(
        built["probability"][:, :, 0], norm.cdf((2.0 - mean) / sigma), atol=1e-14
    )
    np.testing.assert_allclose(
        built["probability"][:, :, -1], norm.sf((9.0 - mean) / sigma), atol=1e-14
    )


def test_likert_cell_law_agrees_with_the_runners_own_truth_mean():
    """sum_k k p_k must be `_truth_emotion`'s 1 + sum_thresholds Phi."""
    boxinggym_core = pytest.importorskip("bayes_model_discovery.experimental.boxinggym_core")
    rows = np.array([[10.0, 20.0, 70.0, 0.2, 0.3, 0.5, 20.0],
                     [5.0, 50.0, 95.0, 0.5, 0.25, 0.25, 95.0]])
    attributes = {
        "alpha": np.linspace(1.5, 5.0, 8),
        "beta_win": np.linspace(0.01, 0.04, 8),
        "beta_PE": np.linspace(-0.03, 0.02, 8),
        "beta_absPE": np.linspace(0.005, 0.03, 8),
        "sigma": np.linspace(0.6, 1.4, 8),
    }
    environment = Environment(**attributes)
    built = law_module.emotion_law(Spec(), environment, rows)
    codes = np.arange(1, 10, dtype=float)
    np.testing.assert_allclose(
        built["probability"] @ codes,
        boxinggym_core._truth_emotion(environment, rows),
        atol=1e-10,
    )


def test_emotion_cell_kl_matches_a_hand_written_categorical_kl(monkeypatch):
    """Exact-cell revision: the cell masses are CDF differences, so this is EXACT.

    Flipped from the earlier quadrature version, which read the same value off a 32-node-per-panel
    quadrature and compared it within its own `kl_quadrature_error`; the two
    diagnostics that described that approximation are gone.
    """
    mean = np.array([[4.4], [6.1]])
    sigma = np.array([0.8])
    model_mean, model_sigma = mean[:, 0] + 0.5, 1.1
    built = likert_cells_law(mean, sigma)
    _, cumulative = normal_pool(model_mean, model_sigma)
    inject_cdf(monkeypatch, cumulative)
    scores = kl_against_pool(built, FITS, np.zeros((2, 1)), component=0)
    closed = categorical_kl(built["probability"][:, 0, :], hand_written_cells(cumulative, 2))
    assert scores["kl"] == pytest.approx(closed, abs=1e-12)
    assert scores["kl_method"] == "cell_cdf_exact"
    assert "kl_quadrature_error" not in scores and "kl_truncated_mass" not in scores
    assert np.isfinite(scores["expected_log_score"])


def test_emotion_end_cells_resolve_a_narrow_pool_far_from_the_edge(monkeypatch):
    """The end cells, where the earlier panelled quadrature was only approximate.

    The truth is wide (sigma 2, so the lower end cell runs from -14 up to 2)
    and the pool is narrow and centred at 5, so all of the pool's mass in that
    cell sits in its last fraction. The quadrature matched the hand-written categorical KL
    to a relative 1e-3 here; the CDF difference matches it exactly.
    """
    mean, sigma = np.array([[4.5]]), np.array([2.0])
    model_mean, model_sigma = np.array([5.0]), 0.4
    built = likert_cells_law(mean, sigma)
    _, cumulative = normal_pool(model_mean, model_sigma)
    inject_cdf(monkeypatch, cumulative)
    scores = kl_against_pool(built, FITS, np.zeros((1, 1)), component=0)
    closed = categorical_kl(built["probability"][:, 0, :], hand_written_cells(cumulative, 1))
    assert scores["kl"] == pytest.approx(closed, rel=1e-12)


def test_emotion_cell_masses_agree_with_the_retired_quadrature_within_its_error(monkeypatch):
    """Condition set during review (2026-09-18): same quantity, computed exactly.

    On an ordinary pool - one whose mass lies inside the truth's own +-8 sigma
    window, which is the case the quadrature was designed for - the exact cell
    masses and the 32-node-per-panel masses agree, and the two KLs agree
    within the `kl_quadrature_error` the quadrature itself reported.
    """
    mean, sigma = np.array([[4.4], [6.1]]), np.array([0.8])
    model_mean, model_sigma = mean[:, 0] + 0.5, 1.1
    built = likert_cells_law(mean, sigma)
    inputs = np.zeros((2, 1))
    density, cumulative = normal_pool(model_mean, model_sigma)
    inject(monkeypatch, density)
    coarse, _ = quadrature_cell_kl(built, inputs, 0, law_module.KL_QUADRATURE_CELL_NODES)
    fine, quadrature_masses = quadrature_cell_kl(
        built, inputs, 0, 2 * law_module.KL_QUADRATURE_CELL_NODES)
    reported_error = abs(fine["kl"] - coarse["kl"])
    inject_cdf(monkeypatch, cumulative)
    scores = kl_against_pool(built, FITS, inputs, component=0)
    masses = law_module._cell_mass_model(FITS, inputs)
    np.testing.assert_allclose(masses.sum(axis=-1), 1.0, atol=1e-12)
    assert reported_error > 0.0
    assert abs(scores["kl"] - fine["kl"]) <= reported_error
    assert np.max(np.abs(masses - quadrature_masses)) < 1e-6


def test_a_wide_mis_placed_pool_separates_the_exact_cells_from_the_quadrature(monkeypatch):
    """The one place the two rules must NOT agree (a review note).

    The truth is narrow (sigma 0.3 at 5.0), so its +-8 sigma window stops at
    11.4; the pool is wide and centred far above it, so most of its mass in the
    top cell [9, inf) lies OUTSIDE that window. The quadrature integrates only
    the part inside and therefore understates the pool's mass there, while the
    CDF difference takes the whole tail - a gap that `kl_quadrature_error`, a
    32-versus-64-node difference, does not bound. The exact value is the
    hand-written categorical KL from scipy's own CDFs.
    """
    mean, sigma = np.array([[5.0]]), np.array([0.3])
    built = likert_cells_law(mean, sigma)
    inputs = np.zeros((1, 1))
    model_mean, model_sigma = np.array([40.0]), 30.0
    density, cumulative = normal_pool(model_mean, model_sigma)
    inject(monkeypatch, density)
    coarse, _ = quadrature_cell_kl(built, inputs, 0, law_module.KL_QUADRATURE_CELL_NODES)
    fine, quadrature_masses = quadrature_cell_kl(
        built, inputs, 0, 2 * law_module.KL_QUADRATURE_CELL_NODES)
    inject_cdf(monkeypatch, cumulative)
    scores = kl_against_pool(built, FITS, inputs, component=0)
    masses = law_module._cell_mass_model(FITS, inputs)
    exact_cells = hand_written_cells(cumulative, 1)
    np.testing.assert_allclose(masses, exact_cells, atol=1e-15)
    assert scores["kl"] == pytest.approx(
        categorical_kl(built["probability"][:, 0, :], exact_cells), abs=1e-12
    )
    # the truncation is in the TOP cell, and it is large - not a rounding gap.
    assert quadrature_masses[0, -1] < 0.2 * masses[0, -1]
    assert abs(scores["kl"] - fine["kl"]) > 100 * abs(fine["kl"] - coarse["kl"])


def test_emotion_cell_kl_vanishes_when_the_pool_is_the_latent_truth(monkeypatch):
    """The emotion cell KL vanishes when the pool is the latent truth."""
    mean = np.array([[3.0, 7.5]])
    sigma = np.array([0.9, 0.6])
    built = likert_cells_law(mean, sigma)
    for component in (0, 1):
        _, cumulative = normal_pool(mean[:, component], sigma[component])
        inject_cdf(monkeypatch, cumulative)
        scores = kl_against_pool(built, FITS, np.zeros((1, 1)), component=component)
        assert abs(scores["kl"]) <= 1e-9


def test_a_cell_the_pool_gives_no_mass_is_an_infinite_kl_not_a_nan(monkeypatch):
    """`_cell_scores`' one departure from `discrete_kl`: 0 * -inf is not nan."""
    truth = np.array([[0.0, 0.5, 0.5]])
    mass = np.array([[0.0, 0.5, 0.5]])
    assert law_module._cell_scores(truth, mass)["kl"] == pytest.approx(0.0, abs=1e-15)
    assert law_module._cell_scores(np.array([[0.5, 0.5, 0.0]]), mass)["kl"] == np.inf


def test_emotion_builder_returns_none_without_the_world_parameters():
    """The emotion law builder returns None without the world parameters."""
    assert law_module.emotion_law(Spec(), Environment(alpha=np.zeros(8)), np.zeros((1, 7))) is None


# --------------------------------------------------------------------------
# exclude_observed: the law follows Q
# --------------------------------------------------------------------------


def test_subset_law_keeps_every_kind_aligned_with_q():
    """subset_law keeps every law kind aligned with the retained Q points."""
    keep = np.array([True, False, True])
    pmf = bernoulli_law([0.1, 0.5, 0.9])
    np.testing.assert_allclose(subset_law(pmf, keep)["pmf"], pmf["pmf"][keep])
    counts = poisson_law([2.0, 60.0, 4.0])
    subset = subset_law(counts, keep)
    np.testing.assert_array_equal(subset["grid"], counts["grid"])  # the grid is NOT per-query
    np.testing.assert_allclose(subset["pmf"], counts["pmf"][keep])
    np.testing.assert_allclose(subset["excluded"], counts["excluded"][keep])
    density = normal_law([1.0, 2.0, 3.0], 0.5)
    np.testing.assert_allclose(subset_law(density, keep)["mean"], [1.0, 3.0])
    cells = likert_cells_law(np.array([[3.0], [5.0], [7.0]]), np.array([1.0]))
    subset = subset_law(cells, keep)
    np.testing.assert_allclose(subset["probability"], cells["probability"][keep])
    np.testing.assert_allclose(subset["lower"], cells["lower"][keep])
    assert subset_law(None, keep) is None


def test_subset_law_leaves_a_subset_kl_equal_to_the_kl_of_the_subset(monkeypatch):
    """The KL of a subset law equals the KL computed directly on the subset."""
    p = np.array([0.2, 0.75, 0.45, 0.6])
    keep = np.array([True, False, True, False])
    built = bernoulli_law(p)
    logpmf = np.log(bernoulli_law([0.3, 0.3, 0.3, 0.3])["pmf"])
    subset = kl_against_pool(subset_law(built, keep), {"logpmf": logpmf[keep]})
    direct = kl_against_pool(bernoulli_law(p[keep]), {"logpmf": logpmf[keep]})
    assert subset["kl"] == direct["kl"]


# --------------------------------------------------------------------------
# the dispatcher
# --------------------------------------------------------------------------


def test_truth_law_dispatches_through_the_spec_and_tolerates_a_lawless_task():
    """truth_law dispatches through the spec and returns None for a lawless task."""
    spec = Spec(truth_mean=lambda environment, x: np.full(len(x), 0.25),
                truth_law=bernoulli_truth_mean_law)
    np.testing.assert_allclose(truth_law(spec, None, [0.0, 1.0])["pmf"][:, 1], 0.25)
    assert truth_law(Spec(), None, [0.0]) is None
