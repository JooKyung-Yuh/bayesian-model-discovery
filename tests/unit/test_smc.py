"""The sequential Monte Carlo (SMC) backend's deterministic parts: the two gates, the chain mapping,
the evidence combination.

Nothing here samples. `pm.sample_smc` is exercised against the frozen run in
tests/equivalence/test_inference_round1.py, which is marked `heavy`; what is
checked here is everything that decides whether a sample would have been of the
RIGHT model, plus the arithmetic that turns sampler output into an evidence.

The two gates are the reason this module can be trusted at all: the model is
written twice (numpy production compiler, pytensor graph), and these tests are
what say the two spellings are the same model.
"""
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from bayes_model_discovery.inference.smc import (
    SMC_CHAINS,
    SMC_DRAWS,
    SMC_SEEDS,
    SYMBOLIC_TOL,
    SMCError,
    _chain_units,
    build_model,
    combine_log_evidence,
    symbolic_mu,
    theta_summary,
    validate_logp,
    validate_symbolic,
)

# The four members of the frozen run's initial pool, with the transforms the
# run's prior policy assigned them.
POOL = [
    ("p1", ["p1"], ["identity"]),
    ("p1 + p2 * x", ["p1", "p2"], ["identity", "identity"]),
    ("p1 * sqrt(x)", ["p1"], ["identity"]),
    ("(p1 * x) / (p2 + x)", ["p1", "p2"], ["identity", "log"]),
]
XS = [0.08, 0.4, 0.72, 1.04, 1.36, 1.68, 2.0]
D = [(0.08, 5), (0.4, 20), (0.72, 32), (1.04, 38), (1.36, 43), (1.68, 47), (2.0, 47)]
PRIOR_ROWS = {
    "p1": [{"phi_prior_mu": 0.0, "phi_prior_sigma": 25.0}],
    "p1 + p2 * x": [{"phi_prior_mu": 0.0, "phi_prior_sigma": 25.0},
                    {"phi_prior_mu": 0.0, "phi_prior_sigma": 25.0}],
    "p1 * sqrt(x)": [{"phi_prior_mu": 0.0, "phi_prior_sigma": 25.0}],
    "(p1 * x) / (p2 + x)": [{"phi_prior_mu": 0.0, "phi_prior_sigma": 25.0},
                            {"phi_prior_mu": 0.0, "phi_prior_sigma": 1.0}],
}


# --------------------------------------------------------- registered constants

def test_the_registered_sampler_configuration_is_unchanged():
    """These four numbers ARE the archived run's inference identity. A change
    to any of them makes new evidence incomparable with the stored evidence."""
    assert SMC_DRAWS == 2000
    assert SMC_CHAINS == 2
    assert SMC_SEEDS == (20260726, 20260727)
    assert SYMBOLIC_TOL == 1e-8


# ------------------------------------------------------ the symbolic gate

@pytest.mark.parametrize("canonical,names,transforms", POOL,
                         ids=[p[0] for p in POOL])
def test_the_pytensor_graph_agrees_with_the_production_compiler(canonical, names,
                                                                transforms):
    """Check that the pytensor graph agrees with the production compiler."""
    worst = validate_symbolic(canonical, names, transforms, XS)
    assert worst <= SYMBOLIC_TOL


def test_the_symbolic_gate_refuses_a_parameter_name_mismatch():
    """Check that the symbolic gate refuses a parameter name mismatch."""
    with pytest.raises(SMCError, match="parameter names disagree"):
        validate_symbolic("p1 + p2 * x", ["p2", "p1"], ["identity", "identity"], XS)


@pytest.mark.parametrize("expr", ["p1 % x", "sin(p1 * x)", "p1 if x else 0"])
def test_symbolic_mu_refuses_anything_outside_the_grammar(expr):
    """Check that symbolic mu refuses anything outside the grammar."""
    import pytensor.tensor as pt

    with pytest.raises((SMCError, SyntaxError)):
        symbolic_mu(expr, ["p1"], [pt.scalar("p1")], pt.vector("x"))


# ----------------------------------------------------------- the logp gate

@pytest.mark.parametrize("canonical,names,transforms", POOL,
                         ids=[p[0] for p in POOL])
def test_the_pymc_joint_logp_equals_production_loglik_plus_prior(canonical, names,
                                                                 transforms):
    """Check that the pymc joint logp equals production loglik plus prior."""
    rows = PRIOR_ROWS[canonical]
    model = build_model(canonical, names, transforms, rows, D)
    worst = validate_logp(model, canonical, names, transforms, rows, D)
    assert worst <= 1e-7


def test_the_logp_gate_fires_when_the_model_is_built_on_a_different_prior():
    """A gate that has never been seen to fire is an assumption. The model is
    built with sigma = 25 and validated against a claimed sigma of 0.5, so the
    prior term disagrees and the gate must refuse rather than return a number."""
    canonical, names, transforms = "p1", ["p1"], ["identity"]
    model = build_model(canonical, names, transforms, PRIOR_ROWS[canonical], D)
    wrong = [{"phi_prior_mu": 0.0, "phi_prior_sigma": 0.5}]
    with pytest.raises(SMCError, match="logp mismatch"):
        validate_logp(model, canonical, names, transforms, wrong, D)


def test_the_logp_gate_refuses_to_pass_on_zero_comparable_probes():
    """A gate that skips every probe has compared nothing.

    `validate_logp` skips a probe whose production reference is non-finite,
    which is right - the reference carries no information there. But the skip
    used to leave `worst` at its initial 0.0, so a form that is non-finite at
    EVERY probe reported the strongest possible agreement on the strength of no
    comparison at all. Here x is negative, so sqrt(x) is nan for every phi and
    all four probes are skipped."""
    canonical, names, transforms = "p1 * sqrt(x)", ["p1"], ["identity"]
    rows = PRIOR_ROWS[canonical]
    nowhere = [(-1.0, 5), (-2.0, 20)]
    model = build_model(canonical, names, transforms, rows, nowhere)
    with pytest.raises(SMCError, match="compared 0 of 4 probes"):
        validate_logp(model, canonical, names, transforms, rows, nowhere)


def test_the_logp_gate_still_passes_when_only_some_probes_are_comparable():
    """The refusal above must not become "every probe must be comparable" - a
    form that is non-finite on part of its prior support is ordinary, and the
    gate's job is to compare where a comparison exists."""
    canonical, names, transforms = "p1 * sqrt(x)", ["p1"], ["identity"]
    rows = PRIOR_ROWS[canonical]
    assert validate_logp(build_model(canonical, names, transforms, rows, D),
                         canonical, names, transforms, rows, D) <= 1e-7


def test_the_logp_gate_fires_when_the_model_is_built_on_different_data():
    """Check that the logp gate fires when the model is built on different data."""
    canonical, names, transforms = "p1 * sqrt(x)", ["p1"], ["identity"]
    rows = PRIOR_ROWS[canonical]
    model = build_model(canonical, names, transforms, rows, D)
    other = [(x, min(y + 3, 50)) for x, y in D]
    with pytest.raises(SMCError, match="logp mismatch"):
        validate_logp(model, canonical, names, transforms, rows, other)


# ------------------------------------------------- the evidence combination

def test_combining_averages_in_evidence_space_not_log_space():
    """Check that combining averages in evidence space not log space."""
    vals = [-87.26131388400839, -87.2990735672698]
    got = combine_log_evidence(vals)
    assert got == float(logsumexp(vals) - math.log(2))
    # strictly above the arithmetic mean of the logs: exp is convex, so the
    # evidence-space average is pulled toward the larger value
    assert got > sum(vals) / 2


def test_combining_one_value_is_the_identity():
    """Check that combining one value is the identity."""
    assert combine_log_evidence([-12.5]) == pytest.approx(-12.5, abs=1e-12)


def test_combining_nothing_is_refused():
    """Check that combining nothing is refused."""
    with pytest.raises(SMCError):
        combine_log_evidence([])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_combining_non_finite_evidence_is_refused(value):
    """Check that combining non finite evidence is refused."""
    with pytest.raises(SMCError, match="non-finite"):
        combine_log_evidence([-12.5, value])


def test_the_two_level_combination_is_associative_in_evidence_space():
    """chains-then-seeds must equal one flat average over all four chain
    values, because both are the same mean of exponentials."""
    seed_a = [-87.26131388400839, -87.2990735672698]
    seed_b = [-87.21, -87.2599]
    two_level = combine_log_evidence([combine_log_evidence(seed_a),
                                      combine_log_evidence(seed_b)])
    flat = combine_log_evidence(seed_a + seed_b)
    assert abs(two_level - flat) < 1e-12


# ----------------------------------------------- per-chain stats extraction

def test_chain_units_handles_the_equal_length_layout():
    """Check that chain units handles the equal length layout."""
    raw = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=object)
    units = _chain_units(raw, 2)
    assert len(units) == 2
    assert list(units[0]) == [1.0, 2.0]


def test_chain_units_handles_the_pymc_513_object_cell_layout():
    """PyMC 5.13 stores unequal per-chain stage counts as a (1, chains) object
    array whose cells are the stage lists. Indexing axis 0 would return one
    row, not two chains."""
    raw = np.empty((1, 2), dtype=object)
    raw[0, 0] = [1.0, 2.0, 3.0]
    raw[0, 1] = [1.0, 2.0]
    units = _chain_units(raw, 2)
    assert len(units) == 2
    assert units[0] == [1.0, 2.0, 3.0]
    assert units[1] == [1.0, 2.0]


def test_chain_units_refuses_a_layout_it_cannot_map():
    """Check that chain units refuses a layout it cannot map."""
    raw = np.empty((1, 3), dtype=object)
    for i in range(3):
        raw[0, i] = [1.0]
    with pytest.raises(SMCError, match="cannot map SMC stats"):
        _chain_units(raw, 2)


# ------------------------------------------------------------- theta summary

def test_theta_summary_applies_the_role_transform_before_quantiles():
    """Check that theta summary applies the role transform before quantiles."""
    phi = np.linspace(-1.0, 1.0, 101).reshape(-1, 1)
    identity = theta_summary(phi, ["identity"], ["p1"])
    logged = theta_summary(phi.copy(), ["log"], ["p1"])
    assert identity["p1"]["median"] == pytest.approx(0.0, abs=1e-9)
    assert logged["p1"]["median"] == pytest.approx(1.0, abs=1e-9)
    assert logged["p1"]["ci95"][0] > 0.0
