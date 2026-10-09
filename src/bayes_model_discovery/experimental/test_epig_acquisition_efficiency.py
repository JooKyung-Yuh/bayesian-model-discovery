"""Bank reference and efficiency tests. The original arithmetic is preserved below."""

import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import xlogy

from bayes_model_discovery.eig.joint import TIE_ATOL, choose_design
from bayes_model_discovery.experimental import boxinggym_core as _path_box_37
from bayes_model_discovery.experimental import chembench_core as _path_chem_37
from bayes_model_discovery.experimental import core_count as _path_core_37
from bayes_model_discovery.experimental import count_epig as bank_epig
from bayes_model_discovery.experimental import count_inference as bank_inference
from bayes_model_discovery.experimental import test_chembench_core as _path_chem_fixture_37
from bayes_model_discovery.experimental import test_chembench_mda_options as _path_chem_mocks_37
from bayes_model_discovery.experimental import test_named_inputs as _path_box_mocks_37
from bayes_model_discovery.experimental.count_epig import _bank, _weights
from bayes_model_discovery.experimental.test_epig_boundary_monotone import _witness_fit
from bayes_model_discovery.experimental.test_epig_positivity_guard import grammar_fit
from bayes_model_discovery.experimental.test_epig_unbounded import count_fit
from bayes_model_discovery.experimental.test_prewave_fit_timeout import (
    _count_result as _path_count_result_37,
)


def reference_information_37(q_pmfs, x_pmfs, particle_weights, q_weights):
    """Sum_q omega_q I(Y_q;Y_x|D), exact in the supplied finite particle bank."""
    q_pmfs, x_pmfs = _bank(q_pmfs), _bank(x_pmfs)
    if q_pmfs.shape[0] != x_pmfs.shape[0]:
        raise ValueError("q and x must use the same ordered posterior particles")
    w = _weights(particle_weights, len(q_pmfs), "particle")
    omega = _weights(q_weights, q_pmfs.shape[1], "quadrature")
    entropy = lambda p: -np.sum(xlogy(p, p), axis=-1)
    q_mixture = np.einsum("k,kqy->qy", w, q_pmfs)
    x_mixture = np.einsum("k,kxy->xy", w, x_pmfs)
    hq, hx = entropy(q_mixture), entropy(x_mixture)
    joint_eig = hx - np.einsum("k,kx->x", w, entropy(x_pmfs))
    pairwise = []
    for qi in range(q_pmfs.shape[1]):
        # Multiply conditional PMFs inside k, then mix k. Never outer-product marginals.
        joint = np.einsum("k,ky,kxz->xyz", w, q_pmfs[:, qi, :], x_pmfs, optimize=True)
        if not np.allclose(joint.sum(axis=(1, 2)), 1, atol=1e-10, rtol=0):
            raise ValueError("joint predictive normalization failed")
        h_joint = -np.sum(xlogy(joint, joint), axis=(1, 2))
        pairwise.append(hq[qi] + hx - h_joint)
    per_q = np.stack(pairwise)
    epig = omega @ per_q
    tolerance = 1e-9
    if (
        np.any(per_q < -tolerance)
        or np.any(joint_eig < -tolerance)
        or np.any(per_q > joint_eig[None, :] + tolerance)
        or not np.all(np.isfinite(epig))
    ):
        raise ValueError("predictive information violated nonnegativity/data-processing bound")
    return {
        "epig": epig,
        "per_q_information": per_q,
        "joint_eig": joint_eig,
        "roundoff_tolerance": tolerance,
        "negative_values_clipped": False,
    }


def reference_bank_37(fits, inputs, weights, grid_size, *, discrete=False):
    """CDF masses on BMA quantile intervals, including the two infinite tails.

    MI of the two binned future observations approximates continuous EPIG.
    A common grid at each input keeps every model and particle on one measure.
    Discrete mode includes the ZIP atom and returns the right-closed interval
    boundaries alongside the bank. Empty intervals retain zero probability.
    """
    if type(grid_size) is not int or grid_size < 2:
        raise ValueError("EPIG grid_size must be an integer of at least two")
    distributions = [bank_inference._epig_scalar_distribution(result, inputs) for result in fits]
    if discrete:
        distributions = [
            bank_inference._EpigZeroInflatedQuantileDistribution(
                distribution, result.particles["pi"]
            )
            if result.hypothesis.family == "zi_poisson"
            else distribution
            for result, distribution in zip(fits, distributions, strict=True)
        ]
    probabilities = np.arange(1, grid_size, dtype=float) / grid_size
    lower = np.min(np.concatenate([d.ppf(probabilities) for d in distributions]), axis=0)
    upper = np.max(np.concatenate([d.ppf(probabilities) for d in distributions]), axis=0)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("non-finite posterior predictive quantile bracket")
    # Fixed iterations make the result independent of convergence dispatch.
    for _ in range(56):
        middle = lower + (upper - lower) / 2
        cdf = np.concatenate([d.cdf(middle[None, :, :]) for d in distributions])
        below = np.einsum("k,kxy->xy", weights, cdf) < probabilities
        lower, upper = np.where(below, middle, lower), np.where(below, upper, middle)
    boundaries = lower + (upper - lower) / 2
    monotone = np.maximum.accumulate(boundaries, axis=-1)
    repaired_rows = np.any(monotone > boundaries, axis=-1)
    repairs = None
    if np.any(repaired_rows):
        repairs = {
            "rows_repaired": int(np.count_nonzero(repaired_rows)),
            "max_shift": float(np.max(monotone[repaired_rows] - boundaries[repaired_rows])),
            "path": "discrete" if discrete else "continuous",
        }
        boundaries = monotone
    # Already-monotone boundary bytes, including signed zeros, stay unchanged.
    del monotone
    banks = []
    for distribution in distributions:
        cdf = distribution.cdf(boundaries[None, :, :])
        banks.append(
            np.diff(
                np.concatenate(
                    [np.zeros_like(cdf[:, :, :1]), cdf, np.ones_like(cdf[:, :, :1])], axis=-1
                ),
                axis=-1,
            )
        )
    bank = np.concatenate(banks)
    if repairs is not None:
        bank = bank.view(bank_inference._EpigBoundaryBank)
        bank.boundary_repairs = repairs
    if discrete:
        if (
            np.any(~np.isfinite(bank))
            or np.any(bank < 0)
            or not np.allclose(bank.sum(-1), 1.0, atol=1e-10, rtol=0)
        ):
            raise ValueError("EPIG discrete interval normalization failed; no renormalization")
        return bank, boundaries
    return bank


def reference_ragged_information_37(banks, candidate_count, weights):
    """Reuse the established finite MI identity on one ragged pair at a time."""

    target_count = len(banks) - candidate_count
    epig, joint_eig = [], []
    for candidate in banks[:candidate_count]:
        per_target = []
        for target in banks[candidate_count:]:
            pair = reference_information_37(
                target[:, None, :], candidate[:, None, :], weights, [1.0]
            )
            per_target.append(float(pair["epig"][0]))
        epig.append(sum(per_target) / target_count)
        joint_eig.append(float(pair["joint_eig"][0]))
    return {"epig": np.asarray(epig), "joint_eig": np.asarray(joint_eig)}


@pytest.mark.parametrize(
    "family", ("normal", "lognormal", "poisson", "negative_binomial", "zi_poisson")
)
def test_37_bank_fixed_point_keeps_full56_boundaries_masses_and_repairs(family):
    """Check: 37 bank fixed point keeps full56 boundaries masses and repairs."""
    discrete = family not in ("normal", "lognormal")
    fit = _witness_fit(family) if discrete else grammar_fit(family, intercepts=(-2.0, 0.7))
    inputs = np.array([0.0, 0.25, 0.5, 1.0])
    weights = np.full(len(fit.particles["b0"]), 1 / len(fit.particles["b0"]))
    expected = reference_bank_37([fit], inputs, weights, 64, discrete=discrete)
    actual = bank_inference._epig_continuous_bank([fit], inputs, weights, 64, discrete=discrete)
    if discrete:
        expected, expected_boundaries = expected
        actual, actual_boundaries = actual
        assert actual_boundaries.tobytes() == expected_boundaries.tobytes()
    assert actual.tobytes() == expected.tobytes()
    assert getattr(actual, "boundary_repairs", None) == getattr(expected, "boundary_repairs", None)


def test_37_ppf_called_once_and_fixed_bracket_reuses_exact_final_state(monkeypatch):
    """Check: 37 ppf called once and fixed bracket reuses exact final state."""
    fit = count_fit("poisson", means=(40.0, 40.0), slopes=(0.0, 0.0))
    inputs, weights = np.array([0.0, 1.0]), np.full(2, 0.5)
    original = bank_inference._epig_scalar_distribution
    calls = {"ppf": 0, "cdf": 0}

    class CountCalls:
        """Test helper class CountCalls."""
        def __init__(self, distribution):
            self.distribution = distribution

        def ppf(self, values):
            """Helper ppf used by the tests in this module."""
            calls["ppf"] += 1
            return self.distribution.ppf(values)

        def cdf(self, values):
            """Helper cdf used by the tests in this module."""
            calls["cdf"] += 1
            return self.distribution.cdf(values)

    monkeypatch.setattr(
        bank_inference, "_epig_scalar_distribution", lambda *args: CountCalls(original(*args))
    )
    actual, boundaries = bank_inference._epig_continuous_bank(
        [fit], inputs, weights, 64, discrete=True
    )
    assert calls == {"ppf": 1, "cdf": 2}  # one fixed-point update and final interval masses
    expected, expected_boundaries = reference_bank_37([fit], inputs, weights, 64, discrete=True)
    assert actual.tobytes() == expected.tobytes()
    assert boundaries.tobytes() == expected_boundaries.tobytes()


def ragged_banks_37():
    """Helper ragged banks used by the tests in this module."""
    rng = np.random.default_rng(713)
    return [rng.dirichlet(np.ones(size), size=11) for size in (4, 9, 6, 7, 3, 10, 5)]


def decisions_37(scores):
    """Helper decisions used by the tests in this module."""
    return choose_design(
        [{"design": float(j), "epig": float(value)} for j, value in enumerate(scores)], "epig"
    )


@pytest.mark.parametrize("family", ("poisson", "negative_binomial", "zi_poisson"))
def test_37_ragged_family_scores_choices_and_ties_match_reference(family):
    """Check: 37 ragged family scores choices and ties match reference."""
    fit, inputs = count_fit(family), np.array([0.0, 0.3, 0.6, 0.7, 1.0])
    banks, _ = bank_inference._epig_discrete_banks([fit], inputs, 3)
    weights = np.full(2, 0.5)
    expected = reference_ragged_information_37(banks, 3, weights)
    actual = bank_inference._epig_discrete_information(banks, 3, weights)
    for key in ("epig", "joint_eig"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=0)
    for key in ("chosen_design", "n_tied_at_max", "tie_atol", "tie_rule"):
        assert decisions_37(actual["epig"])[key] == decisions_37(expected["epig"])[key]


def test_37_ragged_targets_batch_under_fixed_memory_and_validate_bank_once(monkeypatch):
    """Check: 37 ragged targets batch under fixed memory and validate bank once."""
    banks, weights = ragged_banks_37(), np.full(11, 1 / 11)
    expected = reference_ragged_information_37(banks, 3, weights)
    calls = []
    original = bank_epig._bank

    def checked(values):
        """Helper checked used by the tests in this module."""
        calls.append(values.shape)
        return original(values)

    monkeypatch.setattr(bank_epig, "_bank", checked)
    actual = bank_inference._epig_discrete_information(banks, 3, weights)
    assert len(calls) == len(banks)
    calls.clear()
    retained = (
        sum(b.nbytes for b in banks)
        + 8 * 8 * sum(b.shape[1] for b in banks)
        + 8 * 8 * (len(weights) + len(banks))
    )
    # Fit only the widest one-target workspace. This forces batching; the
    # production constant remains untouched outside this isolated test scope.
    one_target = 8 * (2 * 11 * 10 + 2 * 9 * 10 + 2 * 11 * 9)
    monkeypatch.setattr(bank_inference, "EPIG_DISCRETE_MAX_WORKING_BYTES", retained + one_target)
    batched = bank_inference._epig_discrete_information(banks, 3, weights)
    assert len(calls) == len(banks)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=0)
        np.testing.assert_allclose(batched[key], expected[key], rtol=1e-10, atol=0)
    assert (
        decisions_37(actual["epig"])["chosen_design"]
        == decisions_37(expected["epig"])["chosen_design"]
    )


@pytest.mark.parametrize(
    "kind", ("nan", "negative", "unnormalized", "particle_mismatch", "bad_weights")
)
def test_37_ragged_validation_is_not_weakened(kind):
    """Check: 37 ragged validation is not weakened."""
    banks, weights = ragged_banks_37(), np.full(11, 1 / 11)
    if kind == "nan":
        banks[-1][0, 0] = np.nan
    if kind == "negative":
        banks[-1][0, 0] = -0.1
    if kind == "unnormalized":
        banks[-1][0] *= 2
    if kind == "particle_mismatch":
        banks[-1] = banks[-1][1:]
    if kind == "bad_weights":
        weights[0] = -0.1
    with pytest.raises(ValueError):
        bank_inference._epig_discrete_information(banks, 3, weights)


def test_37_ragged_exact_tie_has_same_chosen_design_and_tie_count():
    """Check: 37 ragged exact tie has same chosen design and tie count."""
    banks = ragged_banks_37()
    banks[1] = banks[0].copy()
    banks[2] = banks[0].copy()
    weights = np.full(11, 1 / 11)
    expected = reference_ragged_information_37(banks, 3, weights)
    actual = bank_inference._epig_discrete_information(banks, 3, weights)
    assert (
        decisions_37(actual["epig"])["n_tied_at_max"]
        == decisions_37(expected["epig"])["n_tied_at_max"]
        == 3
    )
    assert decisions_37(actual["epig"])["chosen_design"] == 0


@pytest.mark.parametrize("excess", (0.0, 5e-13))
def test_37_zero_weight_and_accepted_weight_tolerance_preserve_reference(excess):
    """Check: 37 zero weight and accepted weight tolerance preserve reference."""
    banks = ragged_banks_37()
    weights = np.zeros(11)
    weights[1], weights[-1] = 0.4, 0.6 + excess
    expected = reference_ragged_information_37(banks, 3, weights)
    actual = bank_inference._epig_discrete_information(banks, 3, weights)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=0)


@pytest.mark.parametrize("difference,ties,chosen", ((0.5e-12, 2, 0.0), (2e-12, 1, 1.0)))
def test_37_ragged_near_registered_tie_window_matches_reference(difference, ties, chosen):
    """Check: 37 ragged near registered tie window matches reference."""
    weights = np.array([0.5, 0.5])
    target = np.array([[0.85, 0.15], [0.15, 0.85]])
    left = np.array([[0.7, 0.3], [0.3, 0.7]])
    base = reference_ragged_information_37([left, left, target], 2, weights)["epig"][0]
    lower, upper = 0.0, 1e-8
    for _ in range(40):
        step = (lower + upper) / 2
        right = left + np.array([[step, -step], [-step, step]])
        delta = reference_ragged_information_37([left, right, target], 2, weights)["epig"][1] - base
        if delta < difference:
            lower = step
        else:
            upper = step
    right = left + np.array([[upper, -upper], [-upper, upper]])
    expected = reference_ragged_information_37([left, right, target], 2, weights)
    actual = bank_inference._epig_discrete_information([left, right, target], 2, weights)
    for result in (expected, actual):
        decision = decisions_37(result["epig"])
        assert decision["n_tied_at_max"] == ties
        assert decision["chosen_design"] == chosen


def test_37_ragged_releases_previous_target_batch_before_allocating_next(monkeypatch):
    """Check: 37 ragged releases previous target batch before allocating next."""
    import weakref

    banks, weights = ragged_banks_37(), np.full(11, 1 / 11)
    retained = (
        sum(b.nbytes for b in banks)
        + 8 * 8 * sum(b.shape[1] for b in banks)
        + 8 * 8 * (len(weights) + len(banks))
    )
    monkeypatch.setattr(
        bank_inference,
        "EPIG_DISCRETE_MAX_WORKING_BYTES",
        retained + 8 * (2 * 11 * 10 + 2 * 9 * 10 + 2 * 11 * 9),
    )
    original, refs = np.concatenate, []

    def tracked(*args, **kwargs):
        """Helper tracked used by the tests in this module."""
        assert not refs or refs[-1]() is None
        value = original(*args, **kwargs)
        refs.append(weakref.ref(value))
        return value

    monkeypatch.setattr(bank_inference.np, "concatenate", tracked)
    bank_inference._epig_discrete_information(banks, 3, weights)
    assert len(refs) > 3
    assert all(reference() is None for reference in refs)


@pytest.mark.parametrize("particles", (1, 2, 5))
@pytest.mark.parametrize("layout", ("C", "F", "strided"))
def test_37_ragged_singleton_and_noncontiguous_layout_keep_reference_zero_roundoff(
    particles, layout
):
    """Check: 37 ragged singleton and noncontiguous layout keep reference zero roundoff."""
    rng = np.random.default_rng(816)
    for width in (1, 3, 7, 17):
        banks = [rng.dirichlet(np.ones(k), size=particles) for k in (width, 3, 2, 1, 4, 17)]
        if layout == "F":
            banks = [np.asfortranarray(bank) for bank in banks]
        elif layout == "strided":
            banks = [np.repeat(np.repeat(bank, 2, axis=0), 2, axis=1)[::2, ::2] for bank in banks]
        weights = np.full(particles, 1 / particles)
        expected = reference_ragged_information_37(banks, 3, weights)
        actual = bank_inference._epig_discrete_information(banks, 3, weights)
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=0)
            if particles == 1 or layout != "C":
                assert actual[key].tobytes() == expected[key].tobytes()


def test_37_ragged_exact_zero_is_not_clipped_or_changed():
    """Check: 37 ragged exact zero is not clipped or changed."""
    banks = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), np.array([[1.0]])]
    expected = reference_ragged_information_37(banks, 2, [1.0])
    actual = bank_inference._epig_discrete_information(banks, 2, [1.0])
    for key in actual:
        assert np.all(expected[key] == 0.0)
        assert actual[key].tobytes() == expected[key].tobytes()


def _gemm_banks(shape, seed=73, layout="C"):
    k, q, x, y, z = shape
    rng = np.random.default_rng(seed)
    target = rng.dirichlet(np.full(y, 0.3), size=(k, q))
    candidate = rng.dirichlet(np.full(z, 0.3), size=(k, x))
    if layout == "F":
        target, candidate = np.asfortranarray(target), np.asfortranarray(candidate)
    elif layout == "slice":
        target = target[:, ::-1, :]
        candidate = candidate[:, ::-1, :]
    return target, candidate, rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(q))


def _gemm_assert_same(reference, actual, designs):
    assert actual.keys() == reference.keys()
    for key in ("epig", "per_q_information", "joint_eig"):
        np.testing.assert_allclose(actual[key], reference[key], rtol=1e-10, atol=0)
    for key in ("roundoff_tolerance", "negative_values_clipped"):
        assert actual[key] == reference[key]
    for key in ("epig", "joint_eig"):
        old = choose_design(
            [
                {"design": float(x), key: float(v)}
                for x, v in zip(designs, reference[key], strict=True)
            ],
            key,
        )
        new = choose_design(
            [
                {"design": float(x), key: float(v)}
                for x, v in zip(designs, actual[key], strict=True)
            ],
            key,
        )
        assert new["chosen_design"] == old["chosen_design"]
        assert new["n_tied_at_max"] == old["n_tied_at_max"]


@pytest.mark.parametrize(
    "shape",
    [
        (7, 3, 5, 4, 6),
        (8, 3, 1, 64, 2),
        (8, 3, 2, 64, 8),
        (8, 3, 1, 64, 64),
        (1, 3, 4, 3, 5),
        (1, 2, 1, 64, 2),
        (128, 32, 100, 64, 64),
    ],
)
@pytest.mark.parametrize("layout", ["C", "F", "slice"])
def test_gemm_matches_legacy_weight_side_and_layout(shape, layout):
    """Check: gemm matches legacy weight side and layout."""
    inputs = _gemm_banks(shape, layout=layout)
    before = [array.copy() for array in inputs]
    reference = reference_information_37(*inputs)
    actual = bank_epig.predictive_information(*inputs)
    _gemm_assert_same(reference, actual, np.arange(shape[2]))
    for old, new in zip(before, inputs, strict=True):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize("delta,expected_ties", [(1e-13, 2), (1e-8, 1)])
def test_gemm_preserves_near_registered_tie_and_zero_weights(delta, expected_ties):
    """Check: gemm preserves near registered tie and zero weights."""
    target = np.array(
        [[[0.8, 0.2], [0.6, 0.4]], [[0.2, 0.8], [0.4, 0.6]], [[1.0, 0.0], [0.0, 1.0]]]
    )
    candidate = np.array(
        [
            [[0.7, 0.3], [0.7 + delta, 0.3 - delta]],
            [[0.3, 0.7], [0.3 - delta, 0.7 + delta]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    inputs = (target, candidate, np.array([0.5, 0.5, 0.0]), np.array([1.0, 0.0]))
    old = reference_information_37(*inputs)
    new = bank_epig.predictive_information(*inputs)
    _gemm_assert_same(old, new, [0.0, 1.0])
    selected = choose_design(
        [{"design": float(i), "epig": float(v)} for i, v in enumerate(new["epig"])], "epig"
    )
    assert selected["n_tied_at_max"] == expected_ties
    assert (abs(new["epig"][1] - new["epig"][0]) <= TIE_ATOL) == (expected_ties == 2)


@pytest.mark.parametrize(
    "mutation",
    [
        "q_nan",
        "x_inf",
        "negative",
        "q_sum",
        "x_sum",
        "particle_sum",
        "quadrature_sum",
        "particle_negative",
        "quadrature_negative",
        "q_shape",
        "x_shape",
        "particle_shape",
    ],
)
def test_gemm_preserves_validation_errors(mutation):
    """Check: gemm preserves validation errors."""
    target, candidate, weights, omega = _gemm_banks((4, 3, 2, 3, 2))
    if mutation == "q_nan":
        target[0, 0, 0] = np.nan
    elif mutation == "x_inf":
        candidate[0, 0, 0] = np.inf
    elif mutation == "negative":
        target[0, 0, 0] = -0.1
    elif mutation == "q_sum":
        target[0, 0] *= 0.9
    elif mutation == "x_sum":
        candidate[0, 0] *= 0.9
    elif mutation == "particle_sum":
        weights *= 0.9
    elif mutation == "quadrature_sum":
        omega *= 0.9
    elif mutation == "particle_negative":
        weights[0] = -0.1
    elif mutation == "quadrature_negative":
        omega[0] = -0.1
    elif mutation == "q_shape":
        target = target[0]
    elif mutation == "x_shape":
        candidate = candidate[:-1]
    elif mutation == "particle_shape":
        weights = weights[:-1]
    for implementation in (reference_information_37, bank_epig.predictive_information):
        with pytest.raises(ValueError):
            implementation(target, candidate, weights, omega)


def test_gemm_preserves_joint_normalization_check():
    """Check: gemm preserves joint normalization check."""
    target = np.array([[[0.5, 0.5]]]) * (1 + 7e-11)
    candidate = target.copy()
    for implementation in (reference_information_37, bank_epig.predictive_information):
        with pytest.raises(ValueError, match="joint predictive normalization failed"):
            implementation(target, candidate, [1.0], [1.0])


def test_gemm_halves_targets_and_handles_singleton_tail(monkeypatch):
    """Check: gemm halves targets and handles singleton tail."""
    inputs = _gemm_banks((8, 5, 3, 4, 5))
    reference = reference_information_37(*inputs)
    # Independent retained-array bound: the 5-target joint is too large at this
    # synthetic cap; two targets and all retained inputs fit. 5 -> 2 -> tail 1.
    monkeypatch.setattr(bank_inference, "EPIG_DISCRETE_MAX_WORKING_BYTES", 5960)
    empty = np.empty
    allocated = []

    def observed(shape, *args, **kwargs):
        """Helper observed used by the tests in this module."""
        allocated.append(shape)
        return empty(shape, *args, **kwargs)

    monkeypatch.setattr(np, "empty", observed)
    actual = bank_epig.predictive_information(*inputs)
    assert [shape for shape in allocated if shape[0] == 8] == [(8, 8), (8, 8), (8, 4)]
    _gemm_assert_same(reference, actual, np.arange(3))


def test_gemm_refuses_before_workspace_allocation_when_one_target_cannot_fit(monkeypatch):
    """Check: gemm refuses before workspace allocation when one target cannot fit."""
    inputs = _gemm_banks((8, 5, 3, 4, 5))
    monkeypatch.setattr(bank_inference, "EPIG_DISCRETE_MAX_WORKING_BYTES", 1)

    def forbidden(*args, **kwargs):
        """Helper forbidden used by the tests in this module."""
        pytest.fail("workspace allocation preceded the one-target resource refusal")

    monkeypatch.setattr(np, "empty", forbidden)
    with pytest.raises(
        bank_inference.EpigResourceLimit, match="predictive_information_arrays"
    ) as caught:
        bank_epig.predictive_information(*inputs)
    assert (
        caught.value.detail["estimated_working_bytes"]
        > caught.value.detail["max_working_bytes"]
        == 1
    )


@pytest.mark.parametrize(
    "case",
    [
        "normal",
        "lognormal",
        "poisson_enumeration",
        "poisson_coarsened",
        "multiple_models",
        "multiple_components",
    ],
)
def test_37_complete_acquisition_matches_original_values_choice_ties_and_metadata(
    monkeypatch, case
):
    """Check: 37 complete acquisition matches original values choice ties and metadata."""
    candidates, targets = np.array([0.0, 0.15, 0.5]), np.array([0.3, 0.8, 1.0])
    if case == "multiple_components":
        components = [
            [grammar_fit("normal")],
            [grammar_fit("normal", intercepts=(-0.6, 0.2), slopes=(0.7, -0.4))],
        ]
    elif case == "multiple_models":
        components = [
            [
                grammar_fit("lognormal", evidence=np.log(0.3)),
                grammar_fit("gamma", evidence=np.log(0.7)),
            ]
        ]
    elif case.startswith("poisson"):
        means = (10000.0, 12000.0) if case.endswith("coarsened") else (2.0, 5.0)
        components = [[count_fit("poisson", means=means)]]
    else:
        components = [[grammar_fit(case)]]

    def calculate():
        """Helper calculate used by the tests in this module."""
        records = []
        for index, fits in enumerate(components):
            chosen, rows = bank_inference.choose_epig(
                fits, candidates, 170 + index, target_points=targets
            )
            records.append(bank_inference.epig_acquisition_record(chosen, rows, 170 + index))
        total = np.sum([[row["epig"] for row in record["rows"]] for record in records], axis=0)
        return records, decisions_37(total)

    with monkeypatch.context() as patch:
        patch.setattr(bank_inference, "_epig_continuous_bank", reference_bank_37)
        patch.setattr(bank_inference, "_epig_discrete_information", reference_ragged_information_37)
        patch.setattr(bank_epig, "predictive_information", reference_information_37)
        expected, expected_decision = calculate()
    actual, actual_decision = calculate()
    for key in ("chosen_design", "n_tied_at_max", "tie_rule", "tie_atol"):
        assert actual_decision[key] == expected_decision[key]
    for before, after in zip(expected, actual, strict=True):
        assert after["chosen"] == before["chosen"]
        assert after["policy"]["n_tied_at_max"] == before["policy"]["n_tied_at_max"]
        for key in ("epig", "joint_eig"):
            np.testing.assert_allclose(
                [row[key] for row in after["rows"]],
                [row[key] for row in before["rows"]],
                rtol=1e-10,
                atol=0.0,
            )
        assert after.get("outcome_info") == before.get("outcome_info")
        assert after.get("epig_discrete_boundary_repairs") == before.get(
            "epig_discrete_boundary_repairs"
        )


@pytest.mark.parametrize("particles", [2, 7, 128])
@pytest.mark.parametrize("shape", [(3, 4, 3, 5), (5, 1, 64, 2)])
@pytest.mark.parametrize("mode", ["identical_particles", "one_hot_weight"])
@pytest.mark.parametrize("layout", ["C", "F", "slice"])
def test_gemm_degenerate_multi_particle_roundoff_is_exact(particles, shape, mode, layout):
    """Check: gemm degenerate multi particle roundoff is exact."""
    target, candidate, weights, omega = _gemm_banks((particles, *shape), seed=973)
    if mode == "identical_particles":
        target[:] = target[0]
        candidate[:] = candidate[0]
    else:
        weights[:] = 0
        weights[particles // 2] = 1
    if layout == "F":
        target, candidate = np.asfortranarray(target), np.asfortranarray(candidate)
    elif layout == "slice":
        target, candidate = target[:, ::-1, :], candidate[:, ::-1, :]
    inputs = (target, candidate, weights, omega)
    old = reference_information_37(*inputs)
    new = bank_epig.predictive_information(*inputs)
    _gemm_assert_same(old, new, np.arange(shape[1]))
    for key in ("epig", "per_q_information", "joint_eig"):
        assert old[key].tobytes() == new[key].tobytes()


def test_gemm_entropy_phase_does_not_reserve_the_later_matrix_copy(monkeypatch):
    """Check: gemm entropy phase does not reserve the later matrix copy."""
    inputs = _gemm_banks((12, 2, 10, 2, 2), layout="slice")
    old = reference_information_37(*inputs)
    # This cap accommodates retained inputs plus the entropy workspace, then
    # the later x copy and one-target GEMM. Reserving both phases at once would
    # incorrectly reject the bank even at one target.
    monkeypatch.setattr(bank_inference, "EPIG_DISCRETE_MAX_WORKING_BYTES", 7500)
    new = bank_epig.predictive_information(*inputs)
    _gemm_assert_same(old, new, np.arange(10))


@pytest.mark.parametrize("constant_bank", ["target", "candidate"])
def test_gemm_zero_weight_outlier_keeps_positive_particle_degeneracy_exact(constant_bank):
    """Check: gemm zero weight outlier keeps positive particle degeneracy exact."""
    target, candidate, weights, omega = _gemm_banks((128, 3, 4, 3, 5), seed=973)
    bank = target if constant_bank == "target" else candidate
    # Only one input is constant on positive support; other inputs and the
    # zero-weight particle remain nonconstant. Do not let that outlier hide MI0.
    bank[1:, 0, :] = bank[1, 0, :]
    weights[0] = 0
    weights /= weights.sum()
    inputs = (np.asfortranarray(target), np.asfortranarray(candidate), weights, omega)
    old = reference_information_37(*inputs)
    new = bank_epig.predictive_information(*inputs)
    _gemm_assert_same(old, new, np.arange(4))
    for key in ("epig", "per_q_information", "joint_eig"):
        assert old[key].tobytes() == new[key].tobytes()


# Explicit runner-owned lifetime; no global or fit-attached state.
_PATH_KEYS_37 = {'epig_discrete_path_fixed_at_start', 'epig_discrete_path'}


def _path_acquire_37(fits, state, seed=71, candidates=(0., .3), targets=(.6, .9)):
    chosen, rows = bank_inference.choose_epig(fits, candidates, seed, target_points=targets,
                                       discrete_path_state=state)
    return bank_inference.epig_acquisition_record(chosen, rows, seed)


def _path_poisson_fit_37(big=False):
    return grammar_fit('poisson', intercepts=tuple(np.log([200000., 250000.])) if big else (0., .7),
                       slopes=(0., .02))


@pytest.mark.parametrize('big,path', [(False, 'enumeration'), (True, 'coarsened')])
def test_37_path_stable_route_keeps_scores_choice_and_ties(big, path):
    """Check: 37 path stable route keeps scores choice and ties."""
    state = bank_inference.EpigDiscretePathState()
    for seed in (71, 72):
        fits = [_path_poisson_fit_37(big)]
        baseline = _path_acquire_37(fits, None, seed)
        actual = _path_acquire_37(fits, state, seed)
        assert state.path == path
        assert actual['chosen'] == baseline['chosen']
        assert actual['policy']['n_tied_at_max'] == baseline['policy']['n_tied_at_max']
        for key in ('epig', 'joint_eig'):
            np.testing.assert_allclose([r[key] for r in actual['rows']],
                                       [r[key] for r in baseline['rows']], rtol=1e-10, atol=0.)
        info = actual['outcome_info']
        assert info['epig_discrete_path_fixed_at_start'] is True
        assert info['epig_discrete_path'] == path
        assert set(actual) == set(baseline)
        assert set(info) == set(baseline['outcome_info']) | _PATH_KEYS_37
        assert set(info) - set(baseline['outcome_info']) == _PATH_KEYS_37
        if path == 'coarsened' and seed == 72:
            assert info['epig_discrete_enumeration_refusal'] is None
            assert info['epig_discrete_enumeration_cutoffs_known'] is False
            assert info['epig_discrete_enumeration_cutoff_note'] == 'not_recomputed: coarsened_path_fixed_at_D0'
        else:
            assert {key: value for key, value in info.items() if key not in _PATH_KEYS_37} == baseline['outcome_info']


def test_37_path_enumeration_failure_after_initial_choice_propagates(monkeypatch):
    """Check: 37 path enumeration failure after initial choice propagates."""
    state = bank_inference.EpigDiscretePathState()
    _path_acquire_37([_path_poisson_fit_37()], state)
    def forbidden(*args, **kwargs):
        """Helper forbidden used by the tests in this module."""
        raise AssertionError('fixed enumeration cannot switch to coarsened')
    monkeypatch.setattr(bank_inference, '_epig_discrete_coarsened_bank', forbidden)
    with pytest.raises(bank_inference.EpigResourceLimit) as caught:
        _path_acquire_37([_path_poisson_fit_37(True)], state)
    assert caught.value.detail['resource'] == 'integer_support'
    assert state.path == 'enumeration'


def test_37_path_coarsened_keeps_initial_route_without_replanning_or_stale_cutoffs(monkeypatch):
    """Check: 37 path coarsened keeps initial route without replanning or stale cutoffs."""
    state = bank_inference.EpigDiscretePathState()
    first = _path_acquire_37([_path_poisson_fit_37(True)], state)
    assert first['outcome_info']['epig_discrete_enumeration_refusal']['resource'] == 'integer_support'
    def forbidden(*args, **kwargs):
        """Helper forbidden used by the tests in this module."""
        raise AssertionError('fixed coarsened route must not replan enumeration')
    monkeypatch.setattr(bank_inference, '_epig_discrete_banks', forbidden)
    # New fit object and shorter candidate list exercise changing posterior/input shapes.
    next_record = _path_acquire_37([_path_poisson_fit_37()], state, seed=72, candidates=(.3,))
    info = next_record['outcome_info']
    assert state.path == info['epig_discrete_path'] == 'coarsened'
    assert set(next_record) == set(first)
    assert set(info) == set(first['outcome_info'])
    assert info['epig_discrete_path_fixed_at_start'] is True
    assert info['epig_discrete_enumeration_refusal'] is None
    assert info['epig_discrete_enumeration_cutoffs_known'] is False
    assert info['epig_discrete_enumeration_cutoff_note'] == 'not_recomputed: coarsened_path_fixed_at_D0'
    assert np.asarray(info['epig_discrete_bin_boundaries']).shape == (3, 63)


def test_37_path_independent_cell_states_do_not_share_routes():
    """Check: 37 path independent cell states do not share routes."""
    left, right = bank_inference.EpigDiscretePathState(), bank_inference.EpigDiscretePathState()
    _path_acquire_37([_path_poisson_fit_37()], left)
    _path_acquire_37([_path_poisson_fit_37(True)], right)
    _path_acquire_37([_path_poisson_fit_37()], right)
    assert left.path == 'enumeration'
    assert right.path == 'coarsened'


def test_37_path_fixed_coarse_resource_failure_preserves_existing_exception(monkeypatch):
    """Check: 37 path fixed coarse resource failure preserves existing exception."""
    state = bank_inference.EpigDiscretePathState()
    _path_acquire_37([_path_poisson_fit_37(True)], state)
    error = bank_inference.EpigResourceLimit({'resource': 'coarsened_acquisition_arrays', 'enumeration_refusal': None})
    def fail(*args, **kwargs):
        """Helper fail used by the tests in this module."""
        assert args[4] is None
        raise error
    monkeypatch.setattr(bank_inference, '_epig_discrete_coarsened_bank', fail)
    with pytest.raises(bank_inference.EpigResourceLimit) as caught:
        _path_acquire_37([_path_poisson_fit_37()], state)
    assert caught.value is error
    assert state.path == 'coarsened'


def test_37_path_path_state_rejects_reassignment_and_invalid_state():
    """Check: 37 path path state rejects reassignment and invalid state."""
    state = bank_inference.EpigDiscretePathState()
    state._fix('enumeration')
    with pytest.raises(ValueError, match='already fixed'):
        state._fix('coarsened')
    with pytest.raises(ValueError, match='invalid EPIG'):
        state._fix('invalid')
    with pytest.raises(TypeError, match='EpigDiscretePathState'):
        _path_acquire_37([_path_poisson_fit_37()], {})


class _PathLifecycleWitness37:
    """Observe the exact kwargs entering the runner's choose_epig boundary."""

    def __init__(self, path_by_component):
        self.path_by_component = path_by_component
        self.events = {}

    def __call__(self, fits, designs, seed, *args, **kwargs):
        component = fits[0].diagnostics.get('component', 'scalar')
        events = self.events.setdefault(component, [])
        state = kwargs['discrete_path_state']
        assert isinstance(state, bank_inference.EpigDiscretePathState)
        expected = self.path_by_component[component]
        # The first actual runner acquisition must use its D0 fit. Later calls
        # must see freshly rebuilt fits and the identical runner-owned state.
        assert {fit.diagnostics['observation_count'] for fit in fits} == {4 + len(events)}
        if events:
            assert state is events[0]['state']
            assert state.path == expected
            assert not ({id(fit) for fit in fits} &
                        {id(fit) for fit in events[-1]['fits']})
        else:
            assert state.path is None
            state._fix(expected)
        events.append({'state': state, 'fits': tuple(fits), 'seed': seed,
                       'targets': np.asarray(kwargs['target_points']).copy()})
        table = [
            {'design': np.asarray(value).tolist(), 'epig': float(index + 1),
             'joint_eig': float(index + 2)}
            for index, value in enumerate(designs)
        ]
        # This synthetic acquisition isolates lifecycle/wiring. Engine-level
        # path transitions/resource failures are checked in the helper tests above.
        return table[-1]['design'], bank_inference.EpigRows(table, {
            'epig_discrete_path_fixed_at_start': True,
            'epig_discrete_path': expected,
        })


def _assert_three_rounds_37(witness, components):
    assert set(witness.events) == set(components)
    states = []
    for component in components:
        events = witness.events[component]
        assert len(events) == 3
        assert all(event['state'] is events[0]['state'] for event in events)
        for event in events:
            np.testing.assert_array_equal(event['targets'], events[0]['targets'])
        states.append(events[0]['state'])
    assert len({id(state) for state in states}) == len(components)
    return states


def _assert_new_cells_independent_37(first, second, components):
    old_states = _assert_three_rounds_37(first, components)
    new_states = _assert_three_rounds_37(second, components)
    assert not ({id(state) for state in old_states} & {id(state) for state in new_states})
    for component in components:
        assert first.events[component][0]['state'].path == first.path_by_component[component]
        assert second.events[component][0]['state'].path == second.path_by_component[component]


@pytest.mark.parametrize('runner', ('core', 'chembench'))
def test_37_path_runner_keeps_cell_state_across_new_fit_objects(
        tmp_path, monkeypatch, runner):
    """Check: 37 path runner keeps cell state across new fit objects."""
    module = {'core': _path_core_37, 'chembench': _path_chem_37}[runner]
    if runner == 'core':
        def fake_fit(hypothesis, inputs, labels, *args, **kwargs):
            """Helper fake fit used by the tests in this module."""
            result = _path_count_result_37(hypothesis)
            result.diagnostics['observation_count'] = len(labels)
            return result
        monkeypatch.setattr(module, 'fit', fake_fit)
    else:
        # Reuse the existing fake adapter and fit recipe. EPIG at
        # d >= 3 is refused, so the path state is exercised on the two-input fake.
        _path_chem_mocks_37.stub_fits(monkeypatch)
        _path_chem_fixture_37.use_two_input_fake(monkeypatch)
        original_fit = module.fit
        def fake_fit(hypothesis, inputs, labels, *args, **kwargs):
            """Helper fake fit used by the tests in this module."""
            result = original_fit(hypothesis, inputs, labels, *args, **kwargs)
            result.diagnostics['observation_count'] = len(labels)
            return result
        monkeypatch.setattr(module, 'fit', fake_fit)
        monkeypatch.setattr(module, 'OFFICIAL_EVALUATION_COUNT', 8)
        monkeypatch.setitem(module.PROFILE_GRID, 'dev',
                            {**module.PROFILE_GRID['dev'], 'design_pool': 8, 'eig_samples': 8})
    witnesses = []
    for cell_index, path in enumerate(('enumeration', 'coarsened')):
        output = tmp_path / f'{runner}_cell_{cell_index}'
        witness = _PathLifecycleWitness37( {'scalar': path})
        monkeypatch.setattr(module, 'choose_epig', witness)
        if runner == 'core':
            monkeypatch.setattr(sys, 'argv', ['core', '--output', str(output),
                                '--profile', 'dev', '--cell', '002', '--rounds', '3',
                                '--proposer', 'mock'])
            module.main()
        else:
            module.run(_path_chem_fixture_37._args(
                output, _path_chem_fixture_37._manifest(tmp_path), cell='002', rounds=3,
                design_policy=None, structure_proposals=None))
        assert json.loads((output / 'summary.json').read_text())['paid_api_calls'] == 0
        witnesses.append(witness)
    _assert_new_cells_independent_37(*witnesses, ('scalar',))


def _install_synthetic_box_world_37(module, monkeypatch, multi):
    """Same synthetic-world pattern as the existing unit35 emotion fixture."""
    components = module.TASKS['emotion'].components if multi else ()
    spec = replace(module.TASKS['emotion'], module=module.__name__,
                   components=components, openai_narration=False, truth_mean=None)
    bounds = [spec.input_range, *[item['range'] for item in spec.extra_inputs]]
    designs = np.array([[lo + fraction * (hi - lo) for lo, hi in bounds]
                        for fraction in (.2, .4, .6, .8)])

    class SyntheticWorld:
        """Test helper class SyntheticWorld."""
        def __init__(self, *_):
            self.goal = SimpleNamespace(eval_points=[])
            self.observation_calls = 0

        def evaluation_points(self, count, seed):
            """Helper evaluation points used by the tests in this module."""
            points = np.tile(designs, (int(np.ceil(count / len(designs))), 1))[:count]
            self.goal.eval_points = list(points)
            outcomes = (np.tile(np.arange(1., len(components) + 1), (count, 1))
                        if multi else np.full(count, 2.))
            return points, outcomes

        def eval_inputs(self, point):
            """Helper eval inputs used by the tests in this module."""
            return point

        def refused_designs(self):
            """Helper refused designs used by the tests in this module."""
            return set()

        def world_parameters(self):
            """Helper world parameters used by the tests in this module."""
            return {'synthetic': True}

        def observe(self, handle):
            """Helper observe used by the tests in this module."""
            self.observation_calls += 1
            return (np.arange(1., len(components) + 1).tolist() if multi
                    else 2. + .1 * self.observation_calls)

        def evaluate(self, submitted):
            """Helper evaluate used by the tests in this module."""
            assert len(submitted) == len(self.goal.eval_points)
            return {'metric': 0., 'std': 0.}

    _path_box_mocks_37.bg_scalar_stub_fits(monkeypatch)
    monkeypatch.setitem(module.TASKS, 'emotion', spec)
    monkeypatch.setattr(module, 'TaskEnvironment', SyntheticWorld)
    monkeypatch.setattr(module, 'env_seed_injection', lambda *_: nullcontext([]))
    monkeypatch.setattr(module, 'build_design_pool',
                        lambda *_: module.DesignPool(designs, tuple(range(len(designs))), False))
    monkeypatch.setattr(module, 'sample_epig_targets', lambda *_, **__: np.tile(designs, (8, 1)))
    monkeypatch.setattr(module, 'prior_probe', lambda *args, **kwargs: ({}, None))
    return components or ('scalar',)


@pytest.mark.parametrize('multi', (False, True))
def test_37_path_box_state_is_per_component_and_per_new_cell(
        tmp_path, monkeypatch, multi):
    """Check: 37 path box state is per component and per new cell."""
    module = _path_box_37
    components = _install_synthetic_box_world_37(module, monkeypatch, multi)
    witnesses = []
    for cell_index in range(2):
        paths = {name: ('coarsened' if (index + cell_index) % 2 else 'enumeration')
                 for index, name in enumerate(components)}
        witness = _PathLifecycleWitness37( paths)
        monkeypatch.setattr(module, 'choose_epig', witness)
        output = tmp_path / f'box_{multi}_cell_{cell_index}'
        module.run(_path_box_mocks_37.bg_run_args(output, task='emotion', cell='002', rounds=3))
        assert json.loads((output / 'summary.json').read_text())['paid_api_calls'] == 0
        witnesses.append(witness)
    _assert_new_cells_independent_37(*witnesses, components)


@pytest.mark.parametrize('family', ('normal', 'student_t', 'lognormal', 'gamma'))
def test_37_path_continuous_has_no_new_keys_and_exact_record_bytes(family):
    """Check: 37 path continuous has no new keys and exact record bytes."""
    fits = [grammar_fit(family)]
    state = bank_inference.EpigDiscretePathState()
    expected = _path_acquire_37(fits, None)
    for seed in (71, 72):
        actual = _path_acquire_37(fits, state, seed)
        baseline = expected if seed == 71 else _path_acquire_37(fits, None, seed)
        assert state.path is None
        assert 'outcome_info' not in actual
        assert not (_PATH_KEYS_37 & set(actual))
        assert json.dumps(actual, sort_keys=True).encode() == json.dumps(baseline, sort_keys=True).encode()


@pytest.mark.parametrize('kind', ('bounded', 'enumeration', 'coarsened', 'continuous'))
def test_37_path_explicit_none_keeps_default_record_bytes(kind):
    """Check: 37 path explicit none keeps default record bytes."""
    if kind == 'bounded':
        from bayes_model_discovery.experimental.test_epig_positivity_guard import bounded_fit
        fits = [bounded_fit()]
    elif kind == 'continuous':
        fits = [grammar_fit('normal')]
    else:
        fits = [_path_poisson_fit_37(kind == 'coarsened')]
    chosen, rows = bank_inference.choose_epig(fits, (0., .3), 71, target_points=(.6, .9))
    omitted = bank_inference.epig_acquisition_record(chosen, rows, 71)
    explicit = _path_acquire_37(fits, None)
    assert json.dumps(explicit, sort_keys=True).encode() == json.dumps(omitted, sort_keys=True).encode()
    assert not (_PATH_KEYS_37 & set(explicit.get('outcome_info', {})))
    if kind == 'coarsened':
        assert isinstance(explicit['outcome_info']['epig_discrete_enumeration_refusal'], dict)
