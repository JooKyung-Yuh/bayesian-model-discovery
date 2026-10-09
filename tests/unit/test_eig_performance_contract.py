"""Numerical contracts for the expected information gain (EIG) kernel optimisations.

The reference functions in this file preserve the arithmetic from before the optimisations. They are
used only in tests; production retains one implementation. Every comparison is
deterministic, archive-free, and performs no inference or external calls.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import binom

from bayes_model_discovery.eig import joint as J

G25 = [round(0.08 * k, 2) for k in range(1, 26)]
G200 = [round(0.01 * k, 2) for k in range(1, 201)]


def _scalar_row_entropies(pmf, _model_id, _x):
    """The exact pre-optimisation Python loop, retained as the test oracle."""
    return np.array([J._entropy(row / row.sum()) for row in pmf])


def _scipy_binomial_pmf(probabilities, n_trials, _model_id, _x):
    probabilities = np.asarray(probabilities, dtype=float)
    ys = np.arange(n_trials + 1)
    return binom.pmf(ys[None, :], n_trials, probabilities[:, None])


class DesignView:
    """Test group: DesignView."""
    def __init__(self, model_id, base, slope, weight):
        self.model_id = model_id
        self.base = np.asarray(base, dtype=float)
        self.slope = np.asarray(slope, dtype=float)
        self.weight = float(weight)
        self.node_w = np.full(len(self.base), 1.0 / len(self.base))
        self.n_trials = 50

    def success_probs(self, x):
        """Helper or fixture: success probs."""
        return np.clip(self.base + float(x) * self.slope, 1e-9, 1.0 - 1e-9)


def _pool():
    rng = np.random.default_rng(20260818)
    weights = (0.17, 0.31, 0.52)
    return [
        DesignView(
            f"m{i}",
            rng.uniform(0.08 + 0.12 * i, 0.28 + 0.12 * i, 64),
            rng.uniform(-0.025, 0.025, 64),
            weight,
        )
        for i, weight in enumerate(weights)
    ]


def _raw_table(views, grid):
    return [J.design_eig(views, x, n_trials=50) for x in grid]


def _selection(rows):
    best = max(row["eig"] for row in rows)
    tied = [row for row in rows if abs(row["eig"] - best) <= J.TIE_ATOL]
    return min(row["design"] for row in tied), len(tied)


def _synthetic_eig_row(design, eig):
    """Build a complete row whose decision value is controlled exactly."""
    return {
        "design": float(design),
        "eig": float(eig),
        "mixture_entropy": 1.0,
        "expected_conditional_entropy": 1.0 - float(eig),
        "expected_model_predictive_entropy": 1.0,
        "structure_eig": 0.0,
        "parameter_eig": float(eig),
        "total_eig": float(eig),
        "mixture_predictive_mean": 25.0,
        "per_model": {},
    }


def test_vectorised_row_entropy_is_bit_exact_to_the_scalar_loop():
    """Check that vectorised row entropy is bit exact to the scalar loop."""
    probabilities = np.r_[
        1e-9,
        np.linspace(0.001, 0.999, 257),
        1.0 - 1e-9,
    ]
    ys = np.arange(51)
    pmf = np.ascontiguousarray(binom.pmf(ys[None, :], 50, probabilities[:, None]))
    expected = _scalar_row_entropies(pmf, "m", 0.4)
    actual = J._row_entropies(pmf, "m", 0.4)
    assert np.array_equal(actual, expected)


def test_vectorised_row_entropy_keeps_the_post_normalisation_guard():
    """Check that vectorised row entropy keeps the post normalisation guard."""
    pmf = np.array([[0.5, 0.5], [0.0, 0.0]])
    with pytest.raises(J.EIGError, match="m: non-finite pmf at x=0.4"):
        J._row_entropies(pmf, "m", 0.4)


@pytest.mark.parametrize("grid", [G25, G200], ids=["G25", "G200"])
def test_entropy_vectorisation_keeps_full_eig_table_bit_exact(monkeypatch, grid):
    """Check that entropy vectorisation keeps full eig table bit exact."""
    optimised = _raw_table(_pool(), grid)
    monkeypatch.setattr(J, "_row_entropies", _scalar_row_entropies)
    reference = _raw_table(_pool(), grid)

    assert optimised == reference
    assert _selection(optimised) == _selection(reference)
    for row in optimised:
        assert row["total_eig"] == row["structure_eig"] + row["parameter_eig"]


def test_entropy_vectorisation_preserves_selection_inside_the_tie_boundary(monkeypatch):
    """The former probability perturbation left rounding noise some builds round to zero."""
    TIE_DELTA = 2.0 ** -42

    def _offset_at(inner, x_hit=0.2, delta=TIE_DELTA):
        def wrapped(pmf, model_id, x):
            return inner(pmf, model_id, x) + (delta if x == x_hit else 0.0)
        return wrapped

    def view():
        return SimpleNamespace(
            model_id="near-tie",
            weight=1.0,
            node_w=np.array([0.5, 0.5]),
            n_trials=50,
            success_probs=lambda x: np.array([0.2, 0.8]),
        )

    grid = [0.2, 0.1]
    monkeypatch.setattr(J, "_row_entropies", _offset_at(J._row_entropies))
    optimised_summary, optimised_rows = J.select_design([view()], grid)
    delta = abs(optimised_rows[0]["eig"] - optimised_rows[1]["eig"])
    assert 0.0 < delta <= J.TIE_ATOL, "the fixture must exercise the tie tolerance"
    assert abs(delta - TIE_DELTA) <= 16 * np.finfo(float).eps

    monkeypatch.setattr(J, "_row_entropies", _offset_at(_scalar_row_entropies))
    reference_summary, reference_rows = J.select_design([view()], grid)

    assert optimised_rows == reference_rows
    assert optimised_summary == reference_summary
    assert optimised_summary["chosen_design"] == 0.1
    assert optimised_summary["n_tied_at_max"] == 2


@pytest.mark.parametrize("n_trials", [1, 2, 7, 50, 101])
def test_closed_form_binomial_pmf_stays_within_the_predeclared_scipy_bound(n_trials):
    """Check that closed form binomial pmf stays within the predeclared scipy bound."""
    probabilities = np.unique(
        np.r_[
            1e-9,
            np.geomspace(1e-9, 0.5, 500),
            1.0 - np.geomspace(1e-9, 0.5, 500),
            1.0 - 1e-9,
        ]
    )
    expected = _scipy_binomial_pmf(probabilities, n_trials, "m", 0.8)
    actual = J._binomial_pmf_rows(probabilities, n_trials, "m", 0.8)
    assert np.max(np.abs(actual - expected)) <= J.BINOM_PMF_REFERENCE_ATOL


@pytest.mark.parametrize("bad", [0.0, 1.0, np.nan, np.inf, -np.inf])
def test_closed_form_binomial_pmf_rejects_boundaries_and_non_finite_values(bad):
    """Check that closed form binomial pmf rejects boundaries and non finite values."""
    with pytest.raises(J.EIGError, match=r"m: .* at x=0.8"):
        J._binomial_pmf_rows(np.array([0.2, bad]), 50, "m", 0.8)


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_binomial_coefficient_cache_rejects_invalid_trial_counts(bad):
    """Check that binomial coefficient cache rejects invalid trial counts."""
    with pytest.raises(J.EIGError, match="n_trials must be a non-negative integer"):
        J._log_binomial_coefficients(bad)


def test_binomial_coefficient_cache_is_immutable_and_call_order_independent():
    """Check that binomial coefficient cache is immutable and call order independent."""
    J._LOG_BINOMIAL_COEFFICIENTS.clear()
    first_order = {
        n: J._log_binomial_coefficients(n).copy() for n in (101, 1, 50, 2, 7)
    }
    cached = J._log_binomial_coefficients(50)
    assert not cached.flags.writeable
    with pytest.raises(ValueError):
        cached[0] = -1.0

    J._LOG_BINOMIAL_COEFFICIENTS.clear()
    second_order = {
        n: J._log_binomial_coefficients(n).copy() for n in (7, 2, 50, 1, 101)
    }
    for n, expected in first_order.items():
        assert np.array_equal(expected, second_order[n])


@pytest.mark.parametrize("grid", [G25, G200], ids=["G25", "G200"])
def test_closed_form_pmf_keeps_full_eig_table_and_decomposition(monkeypatch, grid):
    """Check that closed form pmf keeps full eig table and decomposition."""
    optimised = _raw_table(_pool(), grid)
    monkeypatch.setattr(J, "_binomial_pmf_rows", _scipy_binomial_pmf)
    reference = _raw_table(_pool(), grid)

    columns = (
        "eig",
        "mixture_entropy",
        "expected_conditional_entropy",
        "expected_model_predictive_entropy",
        "structure_eig",
        "parameter_eig",
        "total_eig",
        "mixture_predictive_mean",
    )
    for actual_row, expected_row in zip(optimised, reference, strict=True):
        assert actual_row["design"] == expected_row["design"]
        for column in columns:
            assert abs(actual_row[column] - expected_row[column]) <= 1e-12
        assert actual_row["per_model"] == expected_row["per_model"]
        assert abs(
            actual_row["total_eig"]
            - actual_row["structure_eig"]
            - actual_row["parameter_eig"]
        ) <= J.DECOMP_TOL

    assert _selection(optimised) == _selection(reference)


def test_closed_form_pmf_preserves_the_near_tie_decision(monkeypatch):
    """Keep both PMF implementations inside the registered tie tolerance.

    The former fixture perturbed one member of a symmetric posterior by
    1e-13. SciPy rounded that perturbation out of the resulting EIG entirely,
    so its ``delta > 0`` self-check depended on floating-point accident rather
    than on the decision contract. This asymmetric posterior produces a
    non-zero signal with ample distance from machine precision in both paths.
    The row-level tests below verify the tie rule independently of either PMF.
    """

    def view():
        return SimpleNamespace(
            model_id="near-tie",
            weight=1.0,
            node_w=np.array([0.3, 0.7]),
            n_trials=50,
            success_probs=lambda x: np.array(
                [0.17 + (5e-11 if x == 0.2 else 0.0), 0.63]
            ),
        )

    grid = [0.2, 0.1]
    optimised_summary, optimised_rows = J.select_design([view()], grid)
    optimised_delta = abs(optimised_rows[0]["eig"] - optimised_rows[1]["eig"])

    monkeypatch.setattr(J, "_binomial_pmf_rows", _scipy_binomial_pmf)
    reference_summary, reference_rows = J.select_design([view()], grid)
    reference_delta = abs(reference_rows[0]["eig"] - reference_rows[1]["eig"])

    assert 0.0 < optimised_delta <= J.TIE_ATOL
    assert 0.0 < reference_delta <= J.TIE_ATOL
    assert optimised_summary["chosen_design"] == reference_summary["chosen_design"] == 0.1
    assert optimised_summary["n_tied_at_max"] == reference_summary["n_tied_at_max"] == 2


def test_select_design_applies_the_tie_rule_to_deterministic_near_tie_rows(
    monkeypatch,
):
    """Exercise the tolerance without deriving its delta from PMF rounding."""
    lower = 0.25
    higher = lower + 5e-13
    delta = higher - lower
    assert 0.0 < delta <= J.TIE_ATOL, "the fixture must cross the near-tie path"

    rows = {
        0.2: _synthetic_eig_row(0.2, higher),
        0.1: _synthetic_eig_row(0.1, lower),
    }
    monkeypatch.setattr(
        J,
        "design_eig",
        lambda _views, x, _n_trials=50: dict(rows[float(x)]),
    )

    summary, emitted_rows = J.select_design(
        [SimpleNamespace(weight=1.0)],
        [0.2, 0.1],
    )

    assert emitted_rows == [rows[0.2], rows[0.1]]
    assert summary["chosen_design"] == 0.1
    assert summary["n_tied_at_max"] == 2


def test_select_design_applies_the_tie_rule_to_exact_tie_rows(monkeypatch):
    """An exact tie is valid and must also choose the smallest design."""
    rows = {
        0.2: _synthetic_eig_row(0.2, 0.25),
        0.1: _synthetic_eig_row(0.1, 0.25),
    }
    monkeypatch.setattr(
        J,
        "design_eig",
        lambda _views, x, _n_trials=50: dict(rows[float(x)]),
    )

    summary, emitted_rows = J.select_design(
        [SimpleNamespace(weight=1.0)],
        [0.2, 0.1],
    )

    assert emitted_rows == [rows[0.2], rows[0.1]]
    assert summary["chosen_design"] == 0.1
    assert summary["chosen_eig"] == 0.25
    assert summary["n_tied_at_max"] == 2
