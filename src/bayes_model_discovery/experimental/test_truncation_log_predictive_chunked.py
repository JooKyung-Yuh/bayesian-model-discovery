"""Chunked `log_predictive`: `positive_inference.log_predictive` split along the draw axis, byte for byte.

Synthetic pools only; no run output, seed or response of any real cell is read.

* Unit: the block budget is in no record or frozen header, and on this numpy a one-column block of a
  one-input layout sums differently from the whole array (the reason a block is never narrower than two
  columns there).
"""

import hashlib

import numpy as np
import pytest
from scipy.special import logsumexp

from . import boxinggym_truth_law as truth_law
from . import frozen_kl_rescore
from . import positive_inference as current
from .belief_packet import core_real_interface
from .count_inference import Fit
from .real_inference import RealHypothesis

BUDGET = "_LOG_PREDICTIVE_BLOCK_CELLS"
REAL = ("normal", "student_t", "heteroscedastic_normal")
COUNT = ("poisson", "negative_binomial", "zi_poisson")
EXPRESSIONS = ("b0+b1*z", "b0+b1*u", "b0+b1*log(1+u)", "b0+b1*u*u")
DATA_SHA = hashlib.sha256(b"synthetic public data").hexdigest()


# --------------------------------------------------------------------------
# synthetic pools
# --------------------------------------------------------------------------


def _diagnostics(interface, **extra):
    return {"interface": interface, "response_unit": 1.7, "observation_count": 7,
            "public_data_sha256": DATA_SHA, "round": 3, **extra}


def real_fit(family, particles, seed, expression=EXPRESSIONS[0]):
    """Build a synthetic real-valued `Fit` of the given family with the given number of particles."""
    rng = np.random.default_rng(seed)
    hypothesis = RealHypothesis(expression, family, (("b0", "intercept"), ("b1", "signed")))
    draws = {"b0": rng.normal(0.2, 0.3, particles), "b1": rng.normal(-0.4, 0.3, particles)}
    if family == "heteroscedastic_normal":
        draws.update(a0=rng.normal(-1.0, 0.3, particles), a1=rng.normal(0.4, 0.2, particles))
    else:
        draws["sigma"] = rng.gamma(4.0, 0.1, particles)
    return Fit(hypothesis, draws, float(rng.normal()), [],
               _diagnostics(core_real_interface(), response_center=0.35))


def design(fits, rows, seed=0):
    """Return synthetic input points for the pool: a permuted grid, or uniform rows for a catalogue pool."""
    rng = np.random.default_rng(seed)
    if isinstance(fits[0].hypothesis, current.RegressionHypothesis):
        return rng.uniform(0.0, 1.0, (rows, 3))
    return rng.permutation(np.linspace(0.02, 0.98, rows))


def outcomes_for(fits, inputs, draws, edges, seed=5):
    """The pool's own predictive draws [input, draw]; `edges` adds support-edge cells (zero,
    non-integer counts, nonpositive positives, far outliers) so some blocks hold -inf cells."""
    y = current.predictive_summary(fits, inputs, seed, max(draws, 2))["samples"][:, :draws].copy()
    if edges:
        family = fits[0].hypothesis.family
        cells = [(i % y.shape[0], (7 * i + 3) % draws) for i in range(0, 3 * draws, 5)]
        for index, (row, column) in enumerate(cells):
            if family in REAL:
                y[row, column] = (-40.0, 40.0)[index % 2]
            elif family in COUNT:
                y[row, column] = (0.0, 2.5)[index % 2]
            else:
                y[row, column] = (0.0, -1.0)[index % 2]
    return y


# --------------------------------------------------------------------------
# the source change is exactly log_predictive
# --------------------------------------------------------------------------


def test_the_budget_is_in_no_record_or_frozen_header():
    """Check that the block budget constant appears in no record or frozen header."""
    assert not hasattr(truth_law, BUDGET)
    assert BUDGET not in frozen_kl_rescore.FROZEN_CONSTANT_NAMES
    assert all(BUDGET not in str(value) for value in frozen_kl_rescore.frozen_constants())


# --------------------------------------------------------------------------
# log_predictive: the one-input block rule
# --------------------------------------------------------------------------


def test_rule_r2_is_load_bearing_on_this_numpy():
    """The trap the one-input block-width rule avoids: with one input, a one-column block is a single particle vector
    and numpy sums it pairwise, so some columns round differently from the whole array's."""
    fit = real_fit("normal", 1000, 31)
    inputs = design([fit], 1)
    loglik = current.conditional_loglik(fit, inputs, outcomes_for([fit], inputs, 64, False))
    whole = logsumexp(loglik, axis=0)
    columns = np.concatenate([logsumexp(np.ascontiguousarray(loglik[:, :, j:j + 1]), axis=0)
                              for j in range(64)], axis=1)
    pairs = np.concatenate([logsumexp(np.ascontiguousarray(loglik[:, :, j:j + 2]), axis=0)
                            for j in range(0, 64, 2)], axis=1)
    assert np.array_equal(pairs, whole)
    differing = int(np.sum(columns != whole))
    print(f"chunked log-predictive, one input, 1000 particles, 64 draws: {differing} of 64 one-column "
          f"reductions differ from the whole array's; two-column blocks: 0 differ")
    if not differing:
        pytest.skip("this numpy reduces a one-column block like the whole array: the one-column rule is inert")
