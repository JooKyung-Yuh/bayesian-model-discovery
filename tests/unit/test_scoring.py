"""The count objective, checked against a hand-written Binomial sum.

`count_loglik` is the single likelihood the whole system is scored against, so
it is worth having one test that does not go through any backend: build the
Binomial log-pmf by hand and require agreement, then pin the two behaviours the
rest of the stack depends on - the [0, N] clip that turns an out-of-range mean
into a finite bad score, and the all-or-nothing -inf on a non-finite mean.
"""
import math

import numpy as np
import pytest

from bayes_model_discovery.inference.scoring import count_loglik
from bayes_model_discovery.models.expressions import compile_form

N = 50
D = [(0.2, 9), (0.5, 21), (0.9, 30)]


def _hand_written(fn, theta, D, N):
    total = 0.0
    for x, y in D:
        mu = float(fn(np.array([float(x)]), theta)[0])
        p = min(max(min(max(mu, 0.0), float(N)) / N, 1e-9), 1 - 1e-9)
        total += (math.lgamma(N + 1) - math.lgamma(y + 1) - math.lgamma(N - y + 1)
                  + y * math.log(p) + (N - y) * math.log1p(-p))
    return total


def test_matches_a_hand_written_binomial_sum():
    """Check that matches a hand written binomial sum."""
    fn, names = compile_form("p1 * (1 - exp(-p2 * x))")
    assert names == ["p1", "p2"]
    theta = [45.0, 1.3]
    got = count_loglik(fn, theta, D, N)
    assert abs(got - _hand_written(fn, theta, D, N)) < 1e-9


def test_predicted_mean_above_n_is_clipped_not_rejected():
    """A mean of 500 on a 50-trial Binomial is a terrible model, not an error.
    The clip is what makes it score finitely - and what creates the flat
    plateau the Laplace multi-start exists to escape."""
    fn, _ = compile_form("p1")
    high = count_loglik(fn, [500.0], D, N)
    ceiling = count_loglik(fn, [float(N)], D, N)
    assert math.isfinite(high)
    assert high == ceiling


def test_predicted_mean_below_zero_is_clipped_to_the_floor():
    """Check that predicted mean below zero is clipped to the floor."""
    fn, _ = compile_form("p1")
    low = count_loglik(fn, [-10.0], D, N)
    floor = count_loglik(fn, [0.0], D, N)
    assert math.isfinite(low)
    assert low == floor


def test_a_single_non_finite_mean_takes_the_whole_batch_to_minus_infinity():
    """Not a partial sum over the finite points: a form that blows up anywhere
    on the data is scored -inf everywhere, so it can never win by being
    evaluated on a convenient subset."""
    fn, _ = compile_form("log(p1 + x) * p2")
    got = count_loglik(fn, [-1.0, 10.0], [(1.0, 25)], N)
    assert got == -np.inf


@pytest.mark.parametrize("theta,expected_order", [([10.0], 0), ([25.0], 1), ([30.0], 2)])
def test_scores_order_sensibly_around_the_data(theta, expected_order):
    """Check that scores order sensibly around the data."""
    fn, _ = compile_form("p1")
    scores = [count_loglik(fn, [v], D, N) for v in (10.0, 25.0, 30.0)]
    assert all(math.isfinite(s) for s in scores)
    # the middle constant sits closest to the observed counts (9, 21, 30)
    assert max(range(3), key=lambda i: scores[i]) == 1
    assert math.isfinite(count_loglik(fn, theta, D, N))
