"""Acceptance tests for the chain-rule decomposition of the joint expected information gain (EIG).

    I((M,Phi); Y) = I(M; Y) + E_M[ I(Phi; Y | M) ]
      total_eig     structure_eig    parameter_eig

Everything here exercises the PRODUCTION function `eig.joint.design_eig` - the
one both the Gauss-Hermite path and the particle path already call. There is no
second copy of the EIG formula in this file, so a test passing here is a
statement about the code the runs actually use.

Observation model: the death process emits Binomial(n_trials, p) counts, so the
tests use n_trials=1 (Bernoulli) and n_trials=50 (the production Binomial).
Poisson is deliberately out of scope - it is not the current observation model.

Ported from the earlier prototype's joint-EIG decomposition check script. Two of
the archived checks are NOT ported here and the omissions are deliberate:

  * `t_real_model_view_path` builds a `ModelEIGView`, which needs `compile_form`
    from the not-yet-migrated `model_proposal.py`;
  * `t_regression_stored_run_unchanged` reads the stored run. Its job is done -
    better - by tests/equivalence/test_acquisition_round1.py, which recomputes
    a frozen round end to end instead of spot-checking three states.

Everything below is synthetic and seeded: no archive data, no external calls.
"""
import math

import numpy as np
import pytest

from bayes_model_discovery.eig.joint import (
    DECOMP_TOL,
    EIG_GH_NODES,
    TIE_RULE,
    EIGError,
    design_eig,
    select_design,
)


class StubView:
    """The exact duck-typed interface `design_eig` consumes, with the
    success probabilities supplied directly. Both production views
    (`joint.ModelEIGView`, `views.ParticleEIGView`) expose this same surface;
    fixing the probabilities is what lets a test construct the degenerate
    cases (identical predictives, point-mass posteriors) exactly rather than
    approximately."""

    def __init__(self, model_id, probs, weight, n_trials):
        self.model_id = model_id
        self.weight = float(weight)
        self.n_trials = int(n_trials)
        self._p = np.asarray(probs, float)
        self.node_w = np.full(len(self._p), 1.0 / len(self._p))

    def success_probs(self, x):
        """Helper or fixture: success probs."""
        return self._p


# ------------------------------------------------------ 0. the constant rename

def test_gh_node_constant_kept_its_value_through_the_rename():
    """`GH_NODES` became `EIG_GH_NODES` to clear a name collision with the
    Laplace backend's unrelated node count. A rename that also moved the value
    would be a silent change of the registered quadrature."""
    assert EIG_GH_NODES == 7
    assert TIE_RULE == "smallest_design_value"
    assert DECOMP_TOL == 1e-9


# --------------------------------------------------------------- 1. identity

def test_identity_bernoulli():
    """total == structure + parameter at EVERY design, Bernoulli."""
    views = [StubView("a", [0.10, 0.30, 0.55], 0.4, 1),
             StubView("b", [0.60, 0.75, 0.95], 0.6, 1)]
    for x in np.linspace(0.01, 2.0, 25):
        r = design_eig(views, float(x), n_trials=1)
        d = abs(r["total_eig"] - (r["structure_eig"] + r["parameter_eig"]))
        assert d <= DECOMP_TOL, f"x={x}: identity off by {d}"
        assert r["total_eig"] == r["eig"], "total_eig must alias eig"


def test_identity_binomial():
    """Same, at the production n_trials=50."""
    rng = np.random.default_rng(11)
    views = [StubView("a", rng.uniform(0.05, 0.45, 24), 0.25, 50),
             StubView("b", rng.uniform(0.30, 0.70, 24), 0.35, 50),
             StubView("c", rng.uniform(0.55, 0.95, 24), 0.40, 50)]
    for x in np.linspace(0.01, 2.0, 25):
        r = design_eig(views, float(x), n_trials=50)
        d = abs(r["total_eig"] - (r["structure_eig"] + r["parameter_eig"]))
        assert d <= DECOMP_TOL, f"x={x}: identity off by {d}"


# ------------------------------------------------- 2. structure_eig == 0 cases

@pytest.mark.parametrize("n_trials", [1, 50])
def test_structure_zero_identical_predictives(n_trials):
    """Two models with the SAME posterior predictive carry no structure
    information: observing y cannot tell them apart."""
    p = [0.2, 0.4, 0.6, 0.8]
    views = [StubView("a", p, 0.5, n_trials),
             StubView("b", p, 0.5, n_trials)]
    r = design_eig(views, 1.0, n_trials=n_trials)
    assert abs(r["structure_eig"]) <= DECOMP_TOL, \
        f"n_trials={n_trials}: structure {r['structure_eig']} != 0"
    # and then the whole total must be parameter information
    assert abs(r["total_eig"] - r["parameter_eig"]) <= DECOMP_TOL


def test_structure_zero_unequal_weights():
    """Identical predictives with LOPSIDED weights: still exactly zero.
    A structure term that secretly measured weight imbalance rather than
    predictive difference would fail here and pass the equal-weight test."""
    p = [0.15, 0.5, 0.85]
    views = [StubView("a", p, 0.97, 50), StubView("b", p, 0.03, 50)]
    r = design_eig(views, 1.0, n_trials=50)
    assert abs(r["structure_eig"]) <= DECOMP_TOL, r["structure_eig"]


def test_structure_positive_when_models_differ():
    """Guard against a structure term that is zero for a trivial reason (e.g.
    always returning 0): models with different predictives MUST give > 0."""
    views = [StubView("a", [0.05], 0.5, 50), StubView("b", [0.95], 0.5, 50)]
    r = design_eig(views, 1.0, n_trials=50)
    assert r["structure_eig"] > 1e-3, r["structure_eig"]


# ------------------------------------------------- 3. parameter_eig == 0 cases

@pytest.mark.parametrize("n_trials", [1, 50])
def test_parameter_zero_point_mass(n_trials):
    """A point-mass posterior (one particle) has nothing left to learn about
    its parameters, so the whole total is structure information."""
    views = [StubView("a", [0.2], 0.5, n_trials),
             StubView("b", [0.8], 0.5, n_trials)]
    r = design_eig(views, 1.0, n_trials=n_trials)
    assert abs(r["parameter_eig"]) <= DECOMP_TOL, \
        f"n_trials={n_trials}: parameter {r['parameter_eig']} != 0"
    assert abs(r["total_eig"] - r["structure_eig"]) <= DECOMP_TOL


def test_single_model_pool_total_is_parameter():
    """With one model in the pool there is no structural uncertainty at all:
    total == parameter and structure == 0."""
    views = [StubView("only", [0.1, 0.35, 0.6, 0.9], 1.0, 50)]
    r = design_eig(views, 1.0, n_trials=50)
    assert abs(r["structure_eig"]) <= DECOMP_TOL, r["structure_eig"]
    assert abs(r["total_eig"] - r["parameter_eig"]) <= DECOMP_TOL
    assert r["parameter_eig"] > 1e-6, "a spread posterior must learn something"


def test_single_model_point_mass_is_fully_zero():
    """One model, one particle: nothing to learn either way. Both parts and
    the total must be zero - this is the corner where a sign error hides."""
    views = [StubView("only", [0.42], 1.0, 50)]
    r = design_eig(views, 1.0, n_trials=50)
    for k in ("structure_eig", "parameter_eig", "total_eig"):
        assert abs(r[k]) <= DECOMP_TOL, f"{k} = {r[k]}"


# ----------------------------------------------- 4. finite and non-negative

def test_both_parts_finite_and_nonnegative_over_60_random_pools():
    """Over a wide sweep of pools and designs, both parts stay finite and
    non-negative (entropy concavity). Extremes included."""
    rng = np.random.default_rng(2026)
    for trial in range(60):
        k = int(rng.integers(1, 5))
        views = []
        w = rng.dirichlet(np.ones(k))
        for j in range(k):
            n_part = int(rng.integers(1, 16))
            lo = float(rng.uniform(1e-6, 0.9))
            views.append(StubView(f"m{j}",
                                  rng.uniform(lo, min(lo + 0.3, 1 - 1e-6),
                                              n_part),
                                  w[j], 50))
        r = design_eig(views, float(rng.uniform(0.01, 2.0)), n_trials=50)
        for key in ("structure_eig", "parameter_eig", "total_eig"):
            v = r[key]
            assert math.isfinite(v), f"trial {trial}: {key} not finite"
            assert v >= -DECOMP_TOL, f"trial {trial}: {key} negative ({v})"
        d = abs(r["total_eig"] - (r["structure_eig"] + r["parameter_eig"]))
        assert d <= DECOMP_TOL, f"trial {trial}: identity off by {d}"


def test_per_model_parameter_terms_average_to_pool_parameter():
    """The pool parameter term must be the weight-average of the per-model
    ones. Without this the per-model column could be decorative."""
    rng = np.random.default_rng(5)
    views = [StubView("a", rng.uniform(0.1, 0.4, 9), 0.3, 50),
             StubView("b", rng.uniform(0.4, 0.9, 9), 0.7, 50)]
    r = design_eig(views, 1.0, n_trials=50)
    got = sum(v.weight * r["per_model"][v.model_id]["parameter_eig"]
              for v in views)
    assert abs(got - r["parameter_eig"]) < 1e-7, (got, r["parameter_eig"])


# -------------------------------------------------------- 5. fail-closed

def test_negative_part_halts():
    """The guard must FAIL CLOSED. Feeding weights that do not form a mixture
    drives structure_eig negative; that must raise, not be reported."""
    views = [StubView("a", [0.2], -2.0, 50), StubView("b", [0.8], 3.0, 50)]
    with pytest.raises(EIGError, match="negative beyond tolerance"):
        design_eig(views, 1.0, n_trials=50)


def test_negative_parameter_term_halts():
    """A view whose node weights do not sum to 1 inflates the expected
    conditional entropy without touching the (renormalised) predictive, which
    drives the PARAMETER term negative. Distinct from the structure case above:
    the two guards are separate branches and both must fail closed."""
    v = StubView("a", [0.2, 0.5, 0.8], 1.0, 50)
    v.node_w = v.node_w * 1.5
    with pytest.raises(EIGError, match="parameter_eig is negative"):
        design_eig([v], 1.0, n_trials=50)


def test_non_finite_success_probability_halts():
    """A NaN anywhere halts. Nothing is dropped silently."""
    v = StubView("a", [0.2, float("nan")], 1.0, 50)
    with pytest.raises(EIGError):
        design_eig([v], 1.0, n_trials=50)


def test_select_design_refuses_an_empty_pool():
    """Check that select design refuses an empty pool."""
    with pytest.raises(EIGError, match="empty active pool"):
        select_design([], [0.1, 0.2])


def test_select_design_refuses_weights_that_are_not_a_distribution():
    """Check that select design refuses weights that are not a distribution."""
    views = [StubView("a", [0.2], 0.3, 50), StubView("b", [0.8], 0.3, 50)]
    with pytest.raises(EIGError, match="do not sum to 1"):
        select_design(views, [0.1, 0.2])


# ---------------------------------------------------------- 6. the tie rule

def test_tie_rule_picks_the_smallest_design_value():
    """Every view here is flat in x, so every design in the grid scores
    identically and the whole grid ties. The registered rule (fixed before any
    run) then has to return the SMALLEST design, not the first one scored and
    not the last one to survive a max()."""
    views = [StubView("a", [0.2, 0.5], 0.5, 50),
             StubView("b", [0.6, 0.9], 0.5, 50)]
    grid = [1.2, 0.4, 2.0, 0.08, 0.7]
    summary, table = select_design(views, grid)
    assert summary["n_tied_at_max"] == len(grid), "the setup must produce a full tie"
    assert summary["chosen_design"] == 0.08
    assert summary["tie_rule"] == TIE_RULE
    assert summary["gh_nodes_per_dim"] == EIG_GH_NODES
    # the grid is scored in the order given, so the winner is NOT the first row
    assert table[0]["design"] == 1.2


def test_select_design_returns_one_row_per_design_in_grid_order():
    """Check that select design returns one row per design in grid order."""
    views = [StubView("a", [0.2, 0.5], 0.5, 50),
             StubView("b", [0.6, 0.9], 0.5, 50)]
    grid = [0.5, 0.1, 0.9]
    summary, table = select_design(views, grid)
    assert [r["design"] for r in table] == grid
    assert [r["design"] for r in summary["table"]] == grid
