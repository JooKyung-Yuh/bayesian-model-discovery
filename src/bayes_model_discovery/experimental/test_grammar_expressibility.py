"""What the ONE shared grammar already admits, pinned.

The brief for this test opens on a claim it refuses to assume - that the true laws of the
positive-outcome benchmark tasks lie outside `count_models.validate_expression`
- and requires the claim to be TESTED before any grammar is extended. The probe
that answered it took each task's true rate law from the read-only benchmark
source, wrote it in the runner's own namespace, and passed it through
`validate_expression`, `judge_candidate` and one real sequential Monte Carlo (SMC) fit; the answer was
that every one of them is inside the v2 grammar.

This module pins that answer in the repository's own terms. It does NOT carry a
benchmark's true law or its parameter values: those belong to the read-only
source and stay there, so that nothing a proposer could ever be shown is
written into this tree. What is pinned instead is the STRUCTURE the probe
turned on, as synthetic expressions over a synthetic seven-input positive
interface built exactly the way the ChemBench runner builds its own
(`belief_packet.task_interface` over base inputs plus the z-scored derived
inputs of a frozen `positive_inference.InputTransform`):

- a saturating rational, one coefficient in the denominator;
- the same with the denominator modulated by a second input, through a
  coefficient that is itself in a denominator;
- a two-factor rational whose denominator is a sum of three products.

Each is written under one `log` call, because the positive path's link is
`mean = response_unit * exp(eta)` (`positive_inference.build_model`), so a law
that is a rational function of the inputs is a `log` of that rational in `eta`.
The three facts the probe reported are asserted here for each shape: the v2
gate accepts it, the per-candidate judge admits it in the free-form cell under
the positive family menu, and one real SMC fit on a synthetic world drawn from
it produces only finite particles and recovers its coefficients.

The complementary half of the same table - the calls the gate does NOT have -
is pinned too, because it is the part of the audit that survived step 1 and is
what the grammar work that follows is scoped to.
"""

import numpy as np
import pytest

from .belief_packet import task_interface
from .count_inference import SMCSettings
from .count_models import (
    CALL_ARITY,
    interface_categorical_levels,
    interface_input_names,
    judge_candidate,
    validate_expression,
)
from .positive_inference import InputTransform, ScalarHypothesis, fit

# The free-form cell: structure and family expanded, acquisition off - the cell
# whose candidates `judge_candidate` judges without the S0/F0 restrictions.
FREE_FORM_CELL = "110"
POSITIVE_FAMILY = "lognormal"
POSITIVE_FAMILIES = ("lognormal", "gamma")
INPUT_COUNT = 7

# Synthetic shapes, named for what they are rather than for any benchmark law.
SHAPES = {
    "saturating_rational": (
        "log(a*x1*x0/(b+x0))",
        {"a": "positive", "b": "positive"},
    ),
    "modulated_denominator_rational": (
        "log(a*x1*x0/(b*(1+x2/c)+x0))",
        {"a": "positive", "b": "positive", "c": "positive"},
    ),
    "two_factor_rational": (
        "log(a*x1*x0*x3/(b*x3+c*x0+x0*x3))",
        {"a": "positive", "b": "positive", "c": "positive"},
    ),
}


def synthetic_interface(rng):
    """A positive-outcome interface shaped like the runner's own.

    Base inputs `x0..x6` over synthetic positive ranges, plus one z-scored
    derived input per base input, standardized by a frozen transform fitted on
    a synthetic public pool - the same two-block interface
    `chembench_core.run` hands `count_models` for a free-form run.
    """
    bounds = [(0.01, 100.0)] * 2 + [(0.1, 50.0)] * 2 + [(0.01, 10.0)] * 3
    pool = np.column_stack([
        np.exp(rng.uniform(np.log(low), np.log(high), 256)) for low, high in bounds
    ])
    transform = InputTransform.from_public_pool(pool)
    interface = task_interface(
        "positive",
        [{"name": f"x{index}", "range": list(bound)}
         for index, bound in enumerate(bounds)],
        {f"x{index}z": f"(x{index}-{float(center)!r})/{float(scale)!r}"
         for index, (center, scale)
         in enumerate(zip(transform.center, transform.scale, strict=True))},
        (0.0, None),
    )
    return interface, transform, bounds


@pytest.fixture
def positive_interface():
    """Fixture: a synthetic positive-outcome interface with its frozen transform and bounds."""
    return synthetic_interface(np.random.default_rng(20260920))


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_v2_gate_admits_a_coefficient_in_a_denominator_and_under_a_log(
    shape, positive_interface
):
    """Step 1 (a): `validate_expression` refuses none of the three shapes.

    The gate constrains the coefficient COUNT, the role vocabulary, the
    operator count, the call vocabulary and the namespace - never the position
    a coefficient occupies, so a coefficient in a denominator and a whole
    rational under `log` pass the v2 grammar as it stands.
    """
    interface, _, _ = positive_interface
    expression, parameters = SHAPES[shape]
    names = interface_input_names(interface)
    assert len(names) == 2 * INPUT_COUNT
    validate_expression(
        expression,
        tuple(parameters.items()),
        names,
        interface_categorical_levels(interface),
    )


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_per_candidate_judge_admits_each_shape_in_the_free_form_cell(
    shape, positive_interface
):
    """Step 1 (b): `judge_candidate` returns (True, None) for each shape.

    Judged the way the positive-outcome runner judges a proposal: the
    free-form cell, the runner's own positive family menu, and the run's
    intercept-only seed structure as the initial pool.
    """
    interface, _, _ = positive_interface
    expression, parameters = SHAPES[shape]
    names = interface_input_names(interface)
    seed_structure = ScalarHypothesis(
        "b0", POSITIVE_FAMILY, (("b0", "intercept"),), names
    )
    admitted, reason = judge_candidate(
        {"expression": expression,
         "family": POSITIVE_FAMILY,
         "parameters": dict(parameters)},
        FREE_FORM_CELL,
        [seed_structure],
        POSITIVE_FAMILY,
        input_names=names,
        allowed_families=POSITIVE_FAMILIES,
    )
    assert (admitted, reason) == (True, None)


def test_one_real_smc_fit_on_a_synthetic_saturating_world_is_finite_and_identified():
    """Step 1 (c): the fit path carries a coefficient in a denominator.

    A synthetic world whose true law is the saturating rational shape, drawn
    with lognormal observation noise so the fitted family is the world's own.
    The assertion is the probe's own three readings: no non-finite particle
    (the fit would raise on one), a posterior narrow enough to identify each
    coefficient, and the truth inside the posterior's 90% interval.
    """
    rng = np.random.default_rng(20260920)
    interface, transform, bounds = synthetic_interface(rng)
    names = interface_input_names(interface)
    designs = np.column_stack([
        np.exp(rng.uniform(np.log(low), np.log(high), 48)) for low, high in bounds
    ])
    truth = {"a": 1.25, "b": 0.4}
    rate = truth["a"] * designs[:, 1] * designs[:, 0] / (truth["b"] + designs[:, 0])
    sigma = 0.05
    observations = rate * np.exp(rng.normal(0.0, sigma, len(rate)) - sigma**2 / 2)

    expression, parameters = SHAPES["saturating_rational"]
    hypothesis = ScalarHypothesis(
        expression, POSITIVE_FAMILY, tuple(parameters.items()), names
    )
    result = fit(
        hypothesis,
        designs,
        observations,
        20260920,
        transform,
        1.0,
        SMCSettings(64, 2, 2),
        interface,
    )
    for name, value in truth.items():
        particles = result.particles[name]
        assert np.all(np.isfinite(particles))
        assert np.std(particles) < abs(value)
        low, high = np.quantile(particles, (0.05, 0.95))
        assert low <= value <= high
    assert np.all(np.isfinite(result.particles["sigma"]))
    assert np.isfinite(result.log_evidence)


def test_the_call_vocabulary_the_v2_gate_does_not_have(positive_interface):
    """The half of the step-1 audit that step 1 did not overturn.

    The gate's calls are exactly `CALL_ARITY`; an element-wise function outside
    it is refused with the one literal reason, whatever role its coefficients
    declare. This is what scopes the grammar work that follows: the extension
    needed is the call vocabulary and the domain of the positions already
    parsed, not the positions themselves.
    """
    interface, _, _ = positive_interface
    names = interface_input_names(interface)
    assert set(CALL_ARITY) == {"exp", "log", "sqrt", "tanh", "square", "power"}
    for call in ("abs", "sigmoid", "expm1", "log1p"):
        with pytest.raises(ValueError, match="call outside expression grammar"):
            validate_expression(
                f"a*{call}(x0)", (("a", "signed"),), names,
                interface_categorical_levels(interface),
            )
