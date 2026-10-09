"""Design-policy arms: definitions, identities and the two byte-level anchors.

No PyMC here. `Fit` is a plain dataclass, so a posterior bank can be built
directly and every claim about the arms is checked on exact numbers.
"""

import numpy as np
import pytest
from scipy.special import expit

from ..eig.joint import TIE_ATOL, choose_design
from . import design_policies as dp
from .core_count import acquisition_seed_for, points_variant
from .count_inference import Fit, choose, information_from_components, model_logweights
from .count_models import Hypothesis, evaluate, initial_hypotheses

DESIGNS = np.linspace(0, 1, 16)


def make_fits(seed=7, count=4, particles=64, family=None):
    """Four CORE structures with a synthetic but valid posterior bank."""
    rng = np.random.default_rng(seed)
    fits = []
    for index, hypothesis in enumerate(initial_hypotheses("binomial")[:count]):
        if family is not None:
            hypothesis = Hypothesis(hypothesis.expression, family, hypothesis.parameters)
        values = {
            name: rng.normal(0.0, 1.0, particles) for name, _ in hypothesis.parameters
        }
        for name in hypothesis.noise_parameters:
            values[name] = rng.beta(1.0, 9.0, particles)
        fits.append(Fit(hypothesis, values, float(-30.0 - index)))
    return fits


# --------------------------------------------------------------- definitions


def synthetic_banks(models=3, particles=5, designs=7, outcomes=11, seed=3):
    """Normalized [particle, design, outcome] PMF banks, one per model."""
    rng = np.random.default_rng(seed)
    banks = []
    for _ in range(models):
        raw = rng.gamma(1.5, 1.0, (particles, designs, outcomes))
        banks.append(raw / raw.sum(-1, keepdims=True))
    return banks


@pytest.mark.parametrize("seed", range(6))
def test_structure_is_between_zero_and_joint(seed):
    """Check that structure is between zero and joint."""
    banks = synthetic_banks(seed=seed)
    weights = np.full(len(banks), 1 / len(banks))
    total, structure, parameter = information_from_components(banks, weights)
    assert np.all(structure >= 0)
    assert np.all(structure <= total + 1e-12)
    assert np.allclose(parameter, total - structure, atol=0, rtol=0)


def test_parameter_is_exactly_joint_minus_structure():
    """Check that parameter is exactly joint minus structure."""
    banks = synthetic_banks(models=4, seed=11)
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    total, structure, parameter = information_from_components(banks, weights)
    assert np.array_equal(parameter, total - structure)


def test_single_model_has_zero_structure():
    """Check that single model has zero structure."""
    banks = synthetic_banks(models=1, seed=5)
    total, structure, parameter = information_from_components(banks, np.array([1.0]))
    assert np.array_equal(structure, np.zeros_like(structure))
    assert np.array_equal(parameter, total)


def test_identical_means_zero_disagreement():
    """Plan Q-L2 (i): equal means, different dispersion -> exactly 0.

    Both PMFs are exact binary fractions with mean 1, and the weights sum to
    exactly 1, so the corollary holds at the bit level, not to a tolerance.
    Structure information is meanwhile strictly positive on the same pair: the
    statistic is blind to precisely what the model term sees.
    """
    tight = np.array([0.25, 0.5, 0.25])
    wide = np.array([0.5, 0.0, 0.5])
    weights = np.array([0.5, 0.5])
    banks = [np.tile(p, (3, 4, 1)) for p in (tight, wide)]
    means = dp.predictive_means(banks, n_trials=2)
    assert np.array_equal(means, np.ones((2, 4)))
    spread = dp.point_disagreement(means, weights)
    assert np.array_equal(spread, np.zeros(4))
    assert not np.any(spread)
    _, structure, _ = information_from_components(banks, weights)
    assert np.all(structure > 0.1)


def test_disagreement_matches_the_plan_formula():
    """Check that disagreement matches the plan formula."""
    means = np.array([[1.0, 4.0], [3.0, 4.0], [5.0, 4.0]])
    weights = np.array([0.25, 0.5, 0.25])
    expected = np.array([0.25 * 4 + 0.5 * 0 + 0.25 * 4, 0.0])
    assert np.array_equal(dp.point_disagreement(means, weights), expected)


def test_disagreement_rejects_bad_weights():
    """Check that disagreement rejects bad weights."""
    with pytest.raises(ValueError):
        dp.point_disagreement(np.zeros((2, 3)), np.array([0.5, 0.6]))
    with pytest.raises(ValueError):
        dp.point_disagreement(np.zeros((2, 3)), np.array([-0.5, 1.5]))


def test_predictive_mean_is_the_count_mean():
    """`sum_y y*pbar_k(y|u)` equals n*pbar for a binomial bank, family-free."""
    fits = make_fits(count=1)
    banks = dp._banks(fits, DESIGNS, 1234, 32, None, 50)
    result = fits[0]
    rng = np.random.default_rng(1234)
    indices = rng.choice(len(result.particles["b0"]), size=32, replace=False)
    params = {name: result.particles[name][indices][:, None] for name, _ in
              result.hypothesis.parameters}
    p = expit(evaluate(result.hypothesis.expression, params, DESIGNS[None, :], np, None))
    assert np.allclose(dp.predictive_means(banks, 50), 50 * p.mean(axis=0), atol=1e-10, rtol=0)


def test_structure_eig_from_draws_matches_the_discrete_definition():
    """The sample-based positive/real form converges to the exact discrete one."""
    rng = np.random.default_rng(19)
    weights = np.array([0.3, 0.7])
    pmfs = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    exact = float(
        -np.sum((weights @ pmfs) * np.log(weights @ pmfs))
        + np.sum(weights * np.sum(pmfs * np.log(pmfs), axis=1))
    )
    draws = 400000
    model_of_draw = rng.choice(2, draws, p=weights)
    outcome = np.array([rng.choice(3, p=pmfs[m]) for m in model_of_draw])
    component = np.log(pmfs[:, outcome])[:, None, :]
    mixture = np.log(weights @ pmfs[:, outcome])[None, :]
    estimate = dp.structure_eig_from_draws(component, mixture, model_of_draw, weights)
    assert estimate.shape == (1,)
    assert abs(float(estimate[0]) - exact) < 5e-3


# ----------------------------------------------------- the joint-path anchor


def test_bank_reproduces_the_joint_rows():
    """The disagreement bank IS `count_inference.choose`'s bank, exactly.

    `choose` never returns its bank, so `_banks` rebuilds it from the same seed.
    Recomputing the registered information triple on the rebuilt bank must
    reproduce `choose`'s own rows bit for bit; if the two ever drifted apart the
    columns of one acquisition record would describe two different banks.
    """
    fits = make_fits()
    chosen, rows = choose(fits, DESIGNS, 4242, 32, None, 50)
    banks = dp._banks(fits, DESIGNS, 4242, 32, None, 50)
    total, structure, parameter = information_from_components(
        banks, np.exp(model_logweights(fits))
    )
    for row, t, m, p in zip(rows, total, structure, parameter, strict=True):
        assert row["eig"] == float(t)
        assert row["structure_eig"] == float(m)
        assert row["parameter_eig"] == float(p)
    assert dp.criteria_table(fits, DESIGNS, 4242, 32, None, 50)[0] == rows
    assert dp.select(fits, DESIGNS, 4242, "joint_eig", 32, None, 50)[0] == chosen


def test_joint_arm_rows_are_the_v2_rows():
    """The default policy adds columns to a NEW block and edits none of the old."""
    fits = make_fits()
    _, rows = choose(fits, DESIGNS, 99, 32, None, 50)
    _, kept, record = dp.select(fits, DESIGNS, 99, "joint_eig", 32, None, 50)
    assert kept == rows
    assert all(set(row) == {"design", "eig", "structure_eig", "parameter_eig"} for row in kept)
    assert record["target"] == "joint_model_parameter"
    for row, entry in zip(rows, record["criteria"], strict=True):
        assert {k: entry[k] for k in row} == row


# ------------------------------------------------------------ the random arm


@pytest.mark.parametrize("world", (11, 17, 23))
@pytest.mark.parametrize("replicate", (0, 1, 3))
@pytest.mark.parametrize("round_number", (1, 5, 20))
def test_random_arm_is_the_a0_branch_draw(world, replicate, round_number):
    """Expression for expression, the A=0 branch of core_count's random draw."""
    seed = acquisition_seed_for(world, replicate, round_number)
    expected = float(np.random.default_rng(seed).choice(DESIGNS))
    value, index = dp.random_draw(DESIGNS, seed)
    assert value == expected
    assert DESIGNS[index] == expected


def test_random_arm_selection_equals_the_draw():
    """Check that random arm selection equals the draw."""
    fits = make_fits()
    for round_number in range(1, 8):
        seed = acquisition_seed_for(11, 0, round_number)
        expected = float(np.random.default_rng(seed).choice(DESIGNS))
        chosen, _, record = dp.select(fits, DESIGNS, seed, "uniform_random", 32, None, 50)
        assert chosen == expected
        assert record["random_draw"] == expected
        assert record["n_tied_at_max"] == 1
        assert record["target"] == "random"
        assert sum(row["uniform_random"] for row in record["criteria"]) == 1.0


def test_random_arm_is_reproducible_and_consumes_no_run_rng():
    """Same seed -> same design; the run's own RNG is neither read nor advanced."""
    fits = make_fits()
    seed = acquisition_seed_for(17, 2, 4)
    np.random.seed(20260907)
    before = np.random.get_state()
    first = dp.select(fits, DESIGNS, seed, "uniform_random", 32, None, 50)[0]
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.random.seed(1)  # a completely different global stream
    second = dp.select(fits, DESIGNS, seed, "uniform_random", 32, None, 50)[0]
    assert first == second


def test_random_arm_needs_a_duplicate_free_grid():
    """Check that random arm needs a duplicate free grid."""
    with pytest.raises(ValueError):
        dp.random_draw(np.array([0.5, 0.5]), 3)
    with pytest.raises(ValueError):
        dp.random_draw(np.zeros((2, 2)), 3)


# --------------------------------------------------------- policy plumbing


def test_every_arm_selects_and_records_all_criteria():
    """Check that every arm selects and records all criteria."""
    fits = make_fits()
    seed = acquisition_seed_for(11, 0, 3)
    chosen = {}
    legacy_policies = tuple(policy for policy in dp.POLICIES
                            if policy not in ("epig", "space_filling"))
    for policy in legacy_policies:
        design, rows, record = dp.select(fits, DESIGNS, seed, policy, 32, None, 50)
        chosen[policy] = design
        assert record["design_policy"] == policy
        assert record["objective_column"] == dp.OBJECTIVE_COLUMN[policy]
        assert record["criteria"][record["chosen_index"]]["design"] == design
        assert len(record["criteria"]) == len(DESIGNS)
        for entry in record["criteria"]:
            assert set(entry) == {
                "design", "eig", "structure_eig", "parameter_eig",
                "point_disagreement", "uniform_random",
            }
        assert rows == dp.criteria_table(fits, DESIGNS, seed, 32, None, 50)[0]
        assert record["reason"]
    # the criteria table is identical across arms - only the column read changes
    tables = [
        dp.select(fits, DESIGNS, seed, policy, 32, None, 50)[2]["criteria"]
        for policy in legacy_policies
    ]
    assert all(table == tables[0] for table in tables)
    assert set(chosen) == set(legacy_policies)


def test_arms_disagree_about_the_design():
    """At least two arms pick different designs on the same state."""
    for seed in range(1, 25):
        fits = make_fits(seed=seed)
        picks = {
            policy: dp.select(fits, DESIGNS, 555, policy, 32, None, 50,
                              target_points=np.linspace(0, 1, 32))[0]
            for policy in dp.POLICIES if policy != "space_filling"
        }
        if len(set(picks.values())) > 1:
            return
    raise AssertionError("no state separated the arms")


def test_tie_rule_is_the_registered_one():
    """Ties inside TIE_ATOL go to the smallest design, as `eig/joint.py` fixes."""
    rows = [
        {"design": 0.9, "point_disagreement": 1.0},
        {"design": 0.2, "point_disagreement": 1.0 - TIE_ATOL / 2},
        {"design": 0.5, "point_disagreement": 0.5},
    ]
    decision = choose_design(rows, "point_disagreement")
    assert decision["chosen_design"] == 0.2
    assert decision["n_tied_at_max"] == 2
    assert decision["tie_atol"] == TIE_ATOL


def test_unregistered_policy_is_refused():
    """Check that unregistered policy is refused."""
    fits = make_fits(count=2)
    with pytest.raises(ValueError, match="unregistered design policy"):
        dp.select(fits, DESIGNS, 1, "max_variance", 32, None, 50)
    with pytest.raises(ValueError, match="unregistered design policy"):
        dp.check_policy("111", "max_variance")


def test_a0_cells_take_no_policy():
    """Check that a0 cells take no policy."""
    for cell in ("000", "010", "100", "110"):
        assert dp.check_policy(cell, "joint_eig") is False
        with pytest.raises(ValueError, match="acquisition bit"):
            dp.check_policy(cell, "uniform_random")
    for cell in ("001", "011", "101", "111"):
        assert dp.check_policy(cell, "point_disagreement") is True


def test_variant_tokens_sort_and_combine():
    """Check that variant tokens sort and combine."""
    assert points_variant("off") == ""
    assert points_variant("off", "joint_eig", True) == ""
    assert points_variant("off", "uniform_random", False) == ""
    assert points_variant("t10_20") == "rev10_20"
    assert points_variant("off", "structure_eig", True) == "design_structure_eig"
    assert points_variant("t10_20", "structure_eig", True) == "design_structure_eig+rev10_20"
    assert (
        points_variant("t10_20", "point_disagreement", True)
        == "design_point_disagreement+rev10_20"
    )
    assert points_variant("t5", "uniform_random", True) == "design_uniform_random+rev5"
    for value in (
        points_variant("t10_20", "structure_eig", True),
        points_variant("t10_20", "uniform_random", True),
    ):
        assert value.split("+") == sorted(value.split("+"))


def test_epig_select_requires_fixed_targets_and_uses_shared_prediction_information():
    """Check that epig select requires fixed targets and uses shared prediction information."""
    from .count_inference import choose_epig

    fits = make_fits(count=2)
    with pytest.raises(ValueError, match="explicit fixed target_points"):
        dp.select(fits, DESIGNS, 19, "epig")
    targets = np.linspace(0, 1, 32)
    chosen, rows, record = dp.select(fits, DESIGNS, 19, "epig", target_points=targets)
    assert (chosen, rows) == choose_epig(fits, DESIGNS, 19, target_points=targets)
    assert record["objective_column"] == "epig" and record["units"] == "nats"
    assert record["criteria"][record["chosen_index"]]["design"] == chosen
    assert dp.check_policy("102", "epig") is True
    assert dp.variant_token("epig", True) == ""
