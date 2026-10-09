"""Named multi-input grammar + BoxingGym adapter contract.

Four things are pinned here:

1. the grammar accepts EXACTLY the interface's names and rejects everything
   else with the verbatim reason "unknown input name <id>";
2. `model_id` separates models written over different input names, while CORE's
   twelve initial ids are bit-identical to those of the earlier implementation
   (the literals below are the ones that implementation printed);
3. the paper's task table is transcribed correctly into `LINK_BY_OUTCOME` /
   `OUTCOME_MENUS`, and the link is never an argument;
4. the BoxingGym adapter's environment evaluator and our own scoring read the
   SAME held-out points, and our predictions survive the environment's parser
   unchanged.

A fifth section: the `--proposer
anthropic` wiring in `boxinggym_core.run` - the --allow-paid-api refusal gate,
an injected-client end-to-end round for "death" (including the one-repair
case), the "never touches the client" guarantee of the no-proposal family, and the S0/open-family
catalog-exhaustion contract, all proven with an injected fake client - never a
real `anthropic.Anthropic()`.

A sixth section: with `count_proposer.
judge_candidate` parametrized by the packet's `constraints.allowed_families`,
the injected-client anthropic path admits a candidate for peregrines (poisson,
its initial family; negative_binomial, a revision) and dugongs (lognormal,
its initial family; gamma, a revision) for the first time - the earlier wiring could only
ever admit "death", because the family check was fixed to CORE's own three
families regardless of task. A family outside a task's own menu (declared and
implemented, just not on THAT task's menu) is still rejected, with the same
literal reason Hypothesis has always used for a family it does not recognize
at all.

The environment tests skip when the read-only `boxing_gym` package is not on
PYTHONPATH, so the five MANIFEST checks can run without it.
"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from . import count_proposer
from .belief_packet import (
    build_belief_packet,
    cell_constraints,
    core_interface,
    interface_families,
    interface_grid,
    task_interface,
)
from .count_inference import Fit
from .count_models import (
    CORE_DERIVED_INPUTS,
    CORE_INITIAL_STRUCTURES,
    CORE_INPUTS,
    DEFAULT_INPUT_NAMES,
    IMPLEMENTED_FAMILIES,
    LINK_BY_OUTCOME,
    OUTCOME_MENUS,
    Hypothesis,
    initial_hypotheses,
    interface_input_names,
    judge_candidate,
    named_input_values,
    particle_logpmf,
    standardized_expression,
)
from .count_proposer import AnthropicCountProposer, OpenAICountProposer

# Printed by the earlier implementation (belief-packet scratch directory) for
# initial_hypotheses(f) over the three families, in that order.
V23_CORE_MODEL_IDS = [
    "fd2262c8fc36fdd43a04",
    "e2a71965fd9c8137e3b5",
    "99421d6d64f55e88bcdd",
    "837b460994280db4ae69",
    "9a066b5937460b61a720",
    "1a62faf77a57512e4213",
    "aa57e7f6aae6db2210eb",
    "d5888f3bf52b4416f210",
    "1ce460ad08b379296794",
    "656068784fc9da5e8d89",
    "bb0c0784b4c95f861202",
    "bbe8d00c332b35821e4b",
]
# Successor anchors of the model-identity unit; the predecessor table above remains immutable.
# Re-pin step of the model-identity unit: the POW rule covers square(z), so the three b0+b1*z+b2*square(z)
# anchors (positions 2, 6 and 10) are recomputed with the current canonical_model_id (schema
# ast_roles_family_ac_v2). Their v1 values belonged to schema ast_roles_family_ac_v1:
# 99421d6d64f55e88bcdd, aa57e7f6aae6db2210eb and bb0c0784b4c95f861202. The other nine are unchanged.
MIDC_CORE_MODEL_IDS = [
    "fd2262c8fc36fdd43a04", "e2a71965fd9c8137e3b5", "2bf731df9a5e18b9dce7",
    "725dc31cf0bbf6abb59f", "9a066b5937460b61a720", "1a62faf77a57512e4213",
    "371fecf7c183b441df82", "bf2e102828c6e503ac01", "1ce460ad08b379296794",
    "656068784fc9da5e8d89", "62c8053f618756669cab", "65e7963d07a9981eb9bd",
]
TASK_NAMESPACE = ("t", "t01", "tz")
TASK_INTERFACE = {
    "inputs": [{"name": "t", "range": [0.001, 2.0]}],
    "derived_inputs": {
        "t01": "(t-0.001)/1.999",
        "tz": standardized_expression("t", 0.001, 2.0),
    },
}


def core_pair():
    """Return the CORE grammar pair used by the tests below."""
    return (("b0", "intercept"), ("b1", "signed"))


# --------------------------------------------------------------------------
# 1. the grammar namespace
# --------------------------------------------------------------------------


def test_an_identifier_outside_the_interface_is_rejected_with_the_named_reason():
    """Check that an identifier outside the interface is rejected with the named reason."""
    with pytest.raises(ValueError, match=r"^unknown input name z$"):
        Hypothesis("b0+b1*z", "binomial", core_pair(), TASK_NAMESPACE)
    structures = initial_hypotheses("binomial", ("b0", "b0+b1*tz"), TASK_NAMESPACE)
    ok, reason = judge_candidate(
        {"expression": "b0+b1*z", "family": "binomial", "parameters": dict(core_pair())},
        "111",
        structures,
        "binomial",
    )
    assert (ok, reason) == (False, "unknown input name z")


def test_declared_and_derived_names_are_accepted():
    """Check that declared and derived names are accepted."""
    for expression in ("b0+b1*t", "b0+b1*t01", "b0+b1*tz", "b0+b1*tz+b2*log(1+t01)"):
        parameters = tuple(
            (name, "intercept" if name == "b0" else "signed")
            for name in sorted({"b0", "b1"} | ({"b2"} if "b2" in expression else set()))
        )
        assert Hypothesis(expression, "binomial", parameters, TASK_NAMESPACE).expression


def test_a_coefficient_may_not_shadow_an_interface_input_name():
    """Check that a coefficient may not shadow an interface input name."""
    with pytest.raises(ValueError, match="invalid or reserved coefficient name"):
        Hypothesis("tz+b1*t", "binomial", (("tz", "intercept"), ("b1", "signed")), TASK_NAMESPACE)


def test_the_complexity_limits_are_unchanged():
    """Check that the complexity limits are unchanged."""
    seven = tuple((f"b{i}", "signed" if i else "intercept") for i in range(7))
    with pytest.raises(ValueError, match="require 1..6 distinct mean coefficients"):
        Hypothesis("+".join(f"b{i}*tz" for i in range(7)), "binomial", seven, TASK_NAMESPACE)
    long = "b0" + "+b1*tz" * 16
    with pytest.raises(ValueError, match="more than 15 mean operators"):
        Hypothesis(long, "binomial", core_pair(), TASK_NAMESPACE)
    with pytest.raises(ValueError, match="coefficient role outside fixed prior policy"):
        Hypothesis("b0+b1*tz", "binomial", (("b0", "intercept"), ("b1", "slope")), TASK_NAMESPACE)


# --------------------------------------------------------------------------
# 2. identity
# --------------------------------------------------------------------------


def test_model_id_separates_models_written_over_different_input_names():
    """Check that model id separates models written over different input names."""
    core = Hypothesis("b0+b1*z", "binomial", core_pair())
    task = Hypothesis("b0+b1*tz", "binomial", core_pair(), TASK_NAMESPACE)
    assert core.model_id != task.model_id
    assert core.expression_inputs == ("z",) and task.expression_inputs == ("tz",)


def test_core_model_ids_are_unchanged_from_v2_3():
    """Legacy expectation is retained below; runtime now pins the model-identity unit's schema."""
    ids = [
        h.model_id
        for family in ("binomial", "beta_binomial", "zi_binomial")
        for h in initial_hypotheses(family)
    ]
    assert ids == MIDC_CORE_MODEL_IDS


def test_v2_3_predecessor_anchors_are_preserved():
    """Check that v2 3 predecessor anchors are preserved."""
    from .test_model_id_commutativity import legacy_model_id

    assert [legacy_model_id(h.expression, h.family, h.parameters)
            for family in ("binomial", "beta_binomial", "zi_binomial")
            for h in initial_hypotheses(family)] == V23_CORE_MODEL_IDS


def test_the_declared_namespace_is_not_part_of_the_identity():
    # Two hypotheses whose expressions are identical stay one model even when
    # the interface that admitted them declares more names.
    """Check that the declared namespace is not part of the identity."""
    narrow = Hypothesis("b0+b1*tz", "binomial", core_pair(), ("tz",))
    wide = Hypothesis("b0+b1*tz", "binomial", core_pair(), TASK_NAMESPACE)
    assert narrow.model_id == wide.model_id


# --------------------------------------------------------------------------
# 3. the interface, the standardization and Table 2
# --------------------------------------------------------------------------


def test_core_interface_matches_the_grammar_defaults():
    """Check that core interface matches the grammar defaults."""
    interface = core_interface()
    assert [item["name"] for item in interface["inputs"]] == [i["name"] for i in CORE_INPUTS]
    assert interface["derived_inputs"] == CORE_DERIVED_INPUTS
    assert interface_input_names(interface) == DEFAULT_INPUT_NAMES
    assert [h.expression for h in initial_hypotheses()] == list(CORE_INITIAL_STRUCTURES)


def test_core_standardization_matches_the_general_formula_bit_for_bit():
    """Check that core standardization matches the general formula bit for bit."""
    u = np.linspace(0.0, 1.0, 257)
    short = named_input_values(core_interface(), u)["z"]
    general = named_input_values(
        {
            "inputs": [{"name": "u", "range": [0, 1]}],
            "derived_inputs": {"z": standardized_expression("u", 0, 1)},
        },
        u,
    )["z"]
    assert short.tobytes() == general.tobytes()
    assert short.tobytes() == (2 * u - 1).tobytes()


def test_the_named_input_path_reproduces_the_core_path_bit_for_bit():
    """Check that the named input path reproduces the core path bit for bit."""
    hypothesis = initial_hypotheses()[2]
    particles = {"b0": np.linspace(-1, 1, 8), "b1": np.linspace(-2, 2, 8), "b2": np.zeros(8)}
    grid = np.linspace(0, 1, 11)
    plain = particle_logpmf(hypothesis, particles, grid)
    named = particle_logpmf(hypothesis, particles, grid, 50, core_interface())
    assert plain.tobytes() == named.tobytes()


def test_link_is_fixed_by_outcome_type_and_never_an_argument():
    # The paper's task table, column "Link g".
    """Check that link is fixed by outcome type and never an argument."""
    assert LINK_BY_OUTCOME == {
        "bounded_count": "logit",
        "binary": "logit",
        "unbounded_count": "log",
        "positive": "log",
        "real": "identity",
        "trajectory": "identity",
    }
    built = task_interface("unbounded_count", TASK_INTERFACE["inputs"], {}, (0, None))
    assert built["link"] == "log" and built["trials"] is None
    assert core_interface()["link"] == LINK_BY_OUTCOME["bounded_count"]


def test_outcome_menus_transcribe_table_2():
    # Rows 1-6 of Table 2: initial family, then the revisions it may introduce.
    """Check that outcome menus transcribe table 2."""
    assert OUTCOME_MENUS["bounded_count"] == {
        "initial_family": "binomial",
        "allowed_families": ("binomial", "beta_binomial", "zi_binomial"),
    }
    assert OUTCOME_MENUS["unbounded_count"]["initial_family"] == "poisson"
    assert "negative_binomial" in OUTCOME_MENUS["unbounded_count"]["allowed_families"]
    # Row 4's revisions are "Gamma with mean $\\mu$; heteroscedastic LogNormal"
    # and row 3's are "Student-$t$ ($\\nu=4$); heteroscedastic Normal,
    # $\\log\\sigma(x)=a_0+a_1 z_j$"; both transcriptions are complete.
    assert OUTCOME_MENUS["positive"] == {
        "initial_family": "lognormal",
        "allowed_families": ("lognormal", "gamma", "heteroscedastic_lognormal"),
    }
    assert OUTCOME_MENUS["real"] == {
        "initial_family": "normal",
        "allowed_families": ("normal", "student_t", "heteroscedastic_normal"),
    }
    assert OUTCOME_MENUS["binary"]["allowed_families"] == ("bernoulli",)


def test_interface_families_intersects_the_menu_with_what_is_runnable():
    """Check that interface families intersects the menu with what is runnable."""
    initial, allowed = interface_families("unbounded_count", True)
    assert initial == "poisson"
    # The zi_poisson runner is available; the Table 2 caveat "only where
    # zeros are natural" is the `zero_inflation` gate, declared per task.
    assert allowed == ["poisson", "negative_binomial", "zi_poisson"]
    assert interface_families("unbounded_count", True, zero_inflation=False)[1] == [
        "poisson",
        "negative_binomial",
    ]
    # The bounded-count row carries no such caveat and is never gated.
    assert interface_families("bounded_count", True, zero_inflation=False)[1] == [
        "binomial",
        "beta_binomial",
        "zi_binomial",
    ]
    assert set(allowed) <= set(IMPLEMENTED_FAMILIES)
    assert interface_families("positive", False)[1] == ["lognormal"]
    # `heteroscedastic_lognormal` is declared by Table 2 but has no runner here.
    assert interface_families("positive", True)[1] == ["lognormal", "gamma"]
    assert interface_families("real", True) == (
        "normal",
        ["normal", "student_t", "heteroscedastic_normal"],
    )
    assert interface_families("binary", True) == ("bernoulli", ["bernoulli"])
    with pytest.raises(ValueError, match="not runnable"):
        interface_families("trajectory", True)


def test_a_task_interface_rejects_a_name_outside_its_own_inputs():
    """Check that a task interface rejects a name outside its own inputs."""
    with pytest.raises(ValueError, match="unknown input name"):
        task_interface("bounded_count", TASK_INTERFACE["inputs"], {"bad": "2*q-1"}, (0, 50), 50)
    with pytest.raises(ValueError, match="shadows an input name"):
        task_interface("bounded_count", TASK_INTERFACE["inputs"], {"t": "t"}, (0, 50), 50)
    with pytest.raises(ValueError, match="invalid or reserved input name"):
        task_interface("bounded_count", [{"name": "rho", "range": [0, 1]}], {}, (0, 50), 50)
    with pytest.raises(ValueError, match="finite range"):
        task_interface("bounded_count", [{"name": "t", "range": [1, 1]}], {}, (0, 50), 50)


def test_interface_grid_handles_one_and_several_inputs():
    """Check that interface grid handles one and several inputs."""
    assert interface_grid(core_interface(), 5).shape == (5,)
    two = task_interface(
        "real",
        [{"name": "x1", "range": [-4, 4]}, {"name": "x2", "range": [0, 1]}],
        {},
        (None, None),
    )
    grid = interface_grid(two, 7)
    assert grid.shape == (7, 2) and grid[0].tolist() == [-4.0, 0.0]


def test_the_packet_refuses_a_link_that_contradicts_the_outcome_type():
    """Check that the packet refuses a link that contradicts the outcome type."""
    from .belief_packet import PoolMember

    member = PoolMember(initial_hypotheses()[1], {"b0": np.zeros(4), "b1": np.ones(4)}, -1.0)
    packet = {
        "public_observations": [{"obs_id": 0, "round": 0, "u": 0.5, "y": 25}],
        "active_fits": [member],
        "archive": [],
        "constraints": cell_constraints("111", "binomial"),
        "round": 1,
    }
    assert build_belief_packet(interface=core_interface(), **packet)["packet_version"] == "v2"
    with pytest.raises(ValueError, match="link is fixed by the outcome type"):
        build_belief_packet(interface={**core_interface(), "link": "log"}, **packet)


# --------------------------------------------------------------------------
# 4. the BoxingGym adapter
# --------------------------------------------------------------------------


def adapter():
    """Return a task adapter built from the given task specification."""
    return pytest.importorskip(
        "bayes_model_discovery.experimental.boxinggym_core",
        reason="adapter imports the read-only boxing_gym package lazily",
    )


def test_every_task_spec_generates_a_valid_interface_and_pool():
    """Check that every task spec generates a valid interface and pool."""
    bg = adapter()
    for name, spec in bg.TASKS.items():
        interface = spec.interface()
        namespace = interface_input_names(interface)
        assert interface["link"] == LINK_BY_OUTCOME[spec.outcome_type], name
        for expression in (*spec.initial_structures(), *spec.mock_templates()):
            if spec.outcome_type != "bounded_count":
                continue
            bg.Hypothesis(expression, "binomial", bg.mock_parameters(expression), namespace)
        # The additive-in-every-input structure joins the pool only
        # while 1 + d fits the six-coefficient cap (ADDITIVE_STRUCTURE_CAP_RULE),
        # and a task with declared forms (irt, which has no numeric input to
        # retarget) ships exactly the forms it declares.
        numeric = len(spec.numeric_input_names)
        expected = len(spec.initial_forms) if spec.initial_forms else (
            5 if 1 < numeric <= 5 else 4
        )
        assert len(set(spec.initial_structures())) == expected, name
        low, high = spec.design_range or spec.input_range
        assert all(low <= d <= high for d in spec.d0()), name


def test_retarget_rewrites_inputs_and_leaves_coefficients_alone():
    """Check that retarget rewrites inputs and leaves coefficients alone."""
    bg = adapter()
    assert bg.retarget("b0+b1*z+b2*log(1+u)", {"u": "t01", "z": "tz"}) == "b0+b1*tz+b2*log(1+t01)"
    assert bg.retarget("b0", {"u": "t01", "z": "tz"}) == "b0"


def test_predictions_survive_the_environment_parsers_unchanged():
    """Check that predictions survive the environment parsers unchanged."""
    bg = adapter()
    counts = np.array([6.4, 17.5, 42.9])
    submitted = bg.format_predictions(bg.TASKS["death"], counts)
    # death_process.py:46 parses int(float(pred)); our rounding must be the value
    # the evaluator then scores, not a truncation of a mean.
    assert [int(float(p)) for p in submitted] == [round(float(v)) for v in counts]
    lengths = np.array([1.234567890123, 2.5])
    submitted = bg.format_predictions(bg.TASKS["dugongs"], lengths)
    assert [float(p) for p in submitted] == lengths.tolist()


def test_the_environment_evaluator_and_our_scoring_use_the_same_points():
    """Check that the environment evaluator and our scoring use the same points."""
    bg = adapter()
    pytest.importorskip("boxing_gym.envs.death_process")
    environment = bg.TaskEnvironment(bg.TASKS["death"], 20260907)
    x, y = environment.evaluation_points(6, 4242)
    stored_x = [float(p[0]) for p in environment.goal.eval_points]
    stored_y = [float(p[1]) for p in environment.goal.eval_points]
    assert stored_x == x.tolist() and stored_y == y.tolist()
    # Feeding the truth back gives the environment's own zero.
    assert environment.evaluate([str(int(v)) for v in y])["metric"] == 0.0


def test_only_lotka_volterra_is_specification_only():
    # Survival/hyperbolic/location were first specifications with no runner; the
    # Bernoulli and Normal/Student-t runners make them executable. Exactly one
    # task is non-runnable: lotka_volterra has NO likelihood family in the paper's
    # task table (a deterministic ODE with no observation noise) and would
    # need an ODE grammar this tree does not have. `run` must refuse it with
    # that reason rather than fail inside a fit.
    """Check that only lotka volterra is specification only."""
    bg = adapter()
    not_runnable = {name for name, spec in bg.TASKS.items() if not spec.runnable}
    assert not_runnable == {"lotka_volterra"}
    note = bg.TASKS["lotka_volterra"].note
    assert "specification_only" in note
    assert "deterministic" in note and "ODE grammar" in note
    with pytest.raises(ValueError, match="specification-only"):
        bg.run(SimpleNamespace(task="lotka_volterra", output="/dev/null/nope"))


def test_the_task_registry_records_the_declared_outcome_types():
    """Check that the task registry records the declared outcome types."""
    bg = adapter()
    declared = {name: spec.outcome_type for name, spec in bg.TASKS.items()}
    assert declared == {
        "death": "bounded_count",
        "peregrines": "unbounded_count",
        "dugongs": "positive",
        "survival": "binary",
        "hyperbolic": "binary",
        "location": "real",
        "irt": "binary",
        "moral_machines": "binary",
        "emotion": "real",
        "lotka_volterra": "trajectory",
    }
    assert json.loads(json.dumps(bg.TASKS["death"].interface()))["trials"] == 50


def test_task_packet_is_byte_identical_when_only_the_heldout_set_changes():
    """Held-out-set invariance on the named-input path: vary only the evaluation set."""
    bg = adapter()
    pytest.importorskip("boxing_gym.envs.death_process")
    from .belief_packet import PoolMember, packet_digest, serialize_packet

    spec = bg.TASKS["death"]
    interface = spec.interface()
    namespace = interface_input_names(interface)
    structures = spec.initial_structures()
    constraints = cell_constraints(
        "111",
        "binomial",
        allowed_families=list(OUTCOME_MENUS["bounded_count"]["allowed_families"]),
        initial_structures=list(structures),
    )
    rng = np.random.default_rng(7)
    members = []
    for index, form in enumerate(structures):
        hypothesis = Hypothesis(form, "binomial", bg.mock_parameters(form), namespace)
        particles = {name: rng.normal(size=16) for name, _ in hypothesis.parameters}
        members.append(PoolMember(hypothesis, particles, -float(index + 3)))
    public = [
        {"obs_id": i, "round": 0 if i < 4 else i - 3, "t": 0.2 + 0.4 * i, "y": 5 + 7 * i}
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
    assert packet_digest(texts[0]) == packet_digest(texts[1])
    for forbidden in ("heldout", "evaluation", "truth", "coverage90", "eval_points"):
        assert forbidden not in texts[0]


@pytest.mark.parametrize("family", ["poisson", "negative_binomial", "lognormal", "gamma"])
def test_scalar_mixture_moments_match_scipy_per_particle_moments(family):
    """The closed-form predictive mean/sd the packet reports, against scipy's own."""
    from scipy.stats import gamma as gamma_dist
    from scipy.stats import lognorm, nbinom, poisson

    from .positive_inference import ScalarHypothesis, grammar_means, mixture_moments

    interface = task_interface(
        "positive", TASK_INTERFACE["inputs"], TASK_INTERFACE["derived_inputs"], (0.0, None)
    )
    names = interface_input_names(interface)
    hypothesis = ScalarHypothesis(
        "b0+b1*tz", family, (("b0", "intercept"), ("b1", "signed")), names
    )
    rng = np.random.default_rng(19)
    particles = {"b0": rng.normal(0, 0.3, 256), "b1": rng.normal(0, 0.3, 256)}
    for name in hypothesis.noise_parameters:
        particles[name] = rng.gamma(4.0, 0.25, 256)
    designs = np.linspace(0.01, 1.9, 5)
    mean, sd = mixture_moments(hypothesis, particles, designs, interface, 7.0)
    mu = grammar_means(hypothesis, particles, designs, interface, 7.0)
    if family == "poisson":
        distribution = poisson(mu)
    elif family == "negative_binomial":
        alpha = particles["alpha"][:, None]
        distribution = nbinom(alpha, alpha / (alpha + mu))
    elif family == "lognormal":
        sigma = particles["sigma"][:, None]
        distribution = lognorm(sigma, scale=mu * np.exp(-(sigma**2) / 2))
    else:
        shape = particles["shape"][:, None]
        distribution = gamma_dist(shape, scale=mu / shape)
    component_mean, component_var = distribution.mean(), distribution.var()
    expected_mean = component_mean.mean(axis=0)
    expected_sd = np.sqrt(component_var.mean(axis=0) + component_mean.var(axis=0, ddof=0))
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-10)
    np.testing.assert_allclose(sd, expected_sd, rtol=1e-10)


# --------------------------------------------------------------------------
# 5. environment seed injection
# --------------------------------------------------------------------------
#
# peregrines.py/dugongs.py draw their world parameters (`reset`, via
# `pymc.draw`) and every observation (`step`, via `pymc.sample_prior_predictive`)
# with no `random_seed`, so PyMC reseeds those calls' shared RNGs from OS
# entropy on every call - independently of `np.random.seed()`. `env_seed_
# injection` closes that gap for the duration of one call. death_process.py
# never touches PyMC (theta and every observation come from scipy, which
# already obeys `np.random.seed()`), so it must log nothing.


def test_env_seed_injection_restores_pymc_on_a_clean_exit():
    """Check that env seed injection restores pymc on a clean exit."""
    bg = adapter()
    import pymc

    original_draw, original_spp = pymc.draw, pymc.sample_prior_predictive
    with bg.env_seed_injection(11) as log:
        assert pymc.draw is not original_draw
        assert pymc.sample_prior_predictive is not original_spp
        assert log == []
    assert pymc.draw is original_draw
    assert pymc.sample_prior_predictive is original_spp


def test_env_seed_injection_restores_pymc_after_an_exception():
    """Check that env seed injection restores pymc after an exception."""
    bg = adapter()
    import pymc

    original_draw = pymc.draw
    with pytest.raises(ValueError, match="boom"), bg.env_seed_injection(11):
        raise ValueError("boom")
    assert pymc.draw is original_draw


def test_env_seed_injection_leaves_an_explicit_random_seed_alone():
    """Check that env seed injection leaves an explicit random seed alone."""
    bg = adapter()
    import pymc

    with pymc.Model():
        x = pymc.Normal("x")
        with bg.env_seed_injection(11) as log:
            a = pymc.draw(x, random_seed=777)
            b = pymc.draw(x, random_seed=777)
    assert a == b
    assert log == []  # an explicit seed is never logged - the caller already pinned it


def test_env_seed_injection_is_a_no_op_for_death():
    """Check that env seed injection is a no op for death."""
    bg = adapter()
    pytest.importorskip("boxing_gym.envs.death_process")
    with bg.env_seed_injection(4242) as log:
        environment = bg.TaskEnvironment(bg.TASKS["death"], 4242)
        environment.observe(0.5)
        environment.observe(1.2)
    assert log == []  # DeathProcess draws theta/infected_num from scipy, not PyMC


def test_env_seed_injection_makes_peregrines_world_reproducible_and_seed_dependent():
    """Check that env seed injection makes peregrines world reproducible and seed dependent."""
    bg = adapter()
    pytest.importorskip("boxing_gym.envs.peregrines")

    def world(seed):
        """Build a seeded world for the test and return it."""
        with bg.env_seed_injection(seed) as log:
            environment = bg.TaskEnvironment(bg.TASKS["peregrines"], seed)
        return environment.world_parameters(), log

    params_a, log_a = world(4242)
    params_b, log_b = world(4242)
    # same seed -> byte-identical world (reset() is 4 pm.draw calls: alpha..beta3)
    assert params_a == params_b
    assert log_a == log_b
    assert [entry["function"] for entry in log_a] == ["draw"] * 4
    assert log_a[0]["injected_seed"] == int(np.random.SeedSequence([4242, 0]).generate_state(1)[0])

    params_c, _ = world(99)
    assert params_c != params_a  # different seed -> different world


def test_env_seed_injection_makes_dugongs_world_reproducible_and_seed_dependent():
    """Check that env seed injection makes dugongs world reproducible and seed dependent."""
    bg = adapter()
    pytest.importorskip("boxing_gym.envs.dugongs")

    def world(seed):
        """Build a seeded world for the test and return it."""
        with bg.env_seed_injection(seed) as log:
            environment = bg.TaskEnvironment(bg.TASKS["dugongs"], seed)
        return environment.world_parameters(), log

    params_a, log_a = world(4242)
    params_b, log_b = world(4242)
    assert params_a == params_b  # reset() is 3 pm.draw calls: a, b, l
    assert log_a == log_b
    assert [entry["function"] for entry in log_a] == ["draw"] * 3

    params_c, _ = world(99)
    assert params_c != params_a


def test_env_seed_injection_makes_peregrines_observations_reproducible():
    """Check that env seed injection makes peregrines observations reproducible."""
    bg = adapter()
    pytest.importorskip("boxing_gym.envs.peregrines")

    def observe(seed):
        """Return the observations of the seeded world."""
        with bg.env_seed_injection(seed) as log:
            environment = bg.TaskEnvironment(bg.TASKS["peregrines"], seed)
            ys = [environment.observe(x) for x in (0.5, 1.5, 2.5)]
        return ys, log

    ys_a, log_a = observe(4242)
    ys_b, log_b = observe(4242)
    assert ys_a == ys_b
    assert log_a == log_b
    # 4 reset() draws, then one sample_prior_predictive call per observe()
    assert [entry["function"] for entry in log_a] == ["draw"] * 4 + ["sample_prior_predictive"] * 3

    ys_c, _ = observe(99)
    assert ys_c != ys_a  # different seed -> different observations


# --------------------------------------------------------------------------
# 5. the LLM proposer wiring
# --------------------------------------------------------------------------


def bg_run_args(output, **overrides):
    """The full argparse namespace boxinggym_core.run expects, with default option values."""
    return SimpleNamespace(
        **{
            "output": str(output),
            "task": "death",
            "profile": "dev",
            "proposer": "mock",
            "llm_model": "claude-opus-4-7",
            "allow_paid_api": False,
            "rounds": 1,
            "seed": 11,
            "cell": "111",
            # The runner's two D8 prior-ablation switches
            "prior_template": "default",
            "priors": "off",
            # The runner's three death-regime switches plus the development-only
            # prompt variant.
            "initial_family": None,
            "truth_in_pool": False,
            "inject_oracle_at": None,
            "prompt_variant": None,
            **overrides,
        }
    )


def bg_stub_fits(monkeypatch):
    """Deterministic stand-in for SMC (no PyMC, no real fit): same recipe as
    test_count_proposer.stub_fits, ported to the bounded (death) path's own
    `count_fit` name binding inside boxinggym_core (a top-of-file import, so
    monkeypatching the module attribute reaches every `refit()` call)."""
    bg = adapter()

    def fake_fit(hypothesis, u, y, seed, settings, interface=None, n_trials=50):
        """Stand-in fit that returns a canned result without running SMC."""
        rng = np.random.default_rng(seed % 2**32)
        particles = {name: rng.normal(0.3, 0.4, 8) for name, _ in hypothesis.parameters}
        particles.update({name: rng.beta(2, 8, 8) for name in hypothesis.noise_parameters})
        offset = (int(hypothesis.model_id[:6], 16) % 500) / 100.0
        return Fit(
            hypothesis,
            particles,
            -float(len(y)) - offset,
            [-float(len(y)) - offset] * 2,
            {"observation_count": len(y), "seed": seed},
        )

    monkeypatch.setattr(bg, "count_fit", fake_fit)
    return bg


def bg_scalar_stub_fits(monkeypatch):
    """Deterministic stand-in for SMC on the unbounded/positive/real path
    (peregrines, dugongs, and location): same
    recipe as bg_stub_fits, ported to `scalar_fit`'s own binding inside
    boxinggym_core (`from .positive_inference import fit as scalar_fit`, a
    top-of-file import, so monkeypatching the module attribute reaches every
    `refit()` call). `positive_inference.fit`'s positional shape is (hypothesis,
    inputs, labels, seed, transform, response_unit, settings, interface,
    response_center); the fake accepts exactly that - `refit()` always calls it
    with transform=None. Every peregrines/dugongs/location run fits its four
    initial structures through this same name at round 0 (`fits = [refit(make(
    form, initial_family), 0) ...`), mock or anthropic, so this stub is
    required for ANY test of any of those tasks, not just the LLM-admission
    ones below."""
    bg = adapter()

    def fake_scalar_fit(
        hypothesis,
        inputs,
        labels,
        seed,
        transform,
        response_unit,
        settings,
        interface=None,
        response_center=0.0,
    ):
        """Stand-in scalar fit that returns a canned result without running SMC."""
        rng = np.random.default_rng(seed % 2**32)
        particles = {name: rng.normal(0.3, 0.4, 8) for name, _ in hypothesis.parameters}
        particles.update({name: rng.gamma(4.0, 0.25, 8) for name in hypothesis.noise_parameters})
        offset = (int(hypothesis.model_id[:6], 16) % 500) / 100.0
        return Fit(
            hypothesis,
            particles,
            -float(len(labels)) - offset,
            [-float(len(labels)) - offset] * 2,
            # `interface`/`response_unit` are read back out of diagnostics by
            # positive_inference.particle_means/grammar_means at prediction time
            # (real positive_inference.fit always records both) - omitting them
            # is invisible until the first predictive/checkpoint call, which
            # every round makes.
            {
                "observation_count": len(labels),
                "seed": seed,
                "interface": interface,
                "response_unit": float(response_unit),
            },
        )

    monkeypatch.setattr(bg, "scalar_fit", fake_scalar_fit)
    return bg


def bg_fake_anthropic_client(replies):
    """Injected client that replies with `replies` in order; no credentials, no network."""
    calls = []

    def create(**body):
        """Fake client call: record the request body and return a canned reply."""
        calls.append(body)
        text = replies[min(len(calls) - 1, len(replies) - 1)]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(
                model_dump=lambda **kwargs: {"input_tokens": 9, "output_tokens": 3}
            ),
            id=f"test-only-{len(calls)}",
            model="returned-test-model",
            stop_reason="end_turn",
        )

    return SimpleNamespace(messages=SimpleNamespace(create=create)), calls


def bg_inject_client(monkeypatch, client):
    """Same trick core_count's own tests use: monkeypatch the count_proposer
    module attribute (not boxinggym_core's), because AnthropicCountProposer is
    imported locally inside run() - a fresh `from .count_proposer import
    AnthropicCountProposer` on every call, which resolves against whatever
    count_proposer.AnthropicCountProposer currently is. A real
    anthropic.Anthropic() is never constructed in this file."""
    monkeypatch.setattr(
        count_proposer,
        "AnthropicCountProposer",
        lambda records, **kwargs: AnthropicCountProposer(records, client=client),
    )


def test_boxinggym_run_refuses_anthropic_proposer_without_allow_paid_api(tmp_path):
    """Refused exactly as CORE, before --output is even created."""
    bg = adapter()
    out = tmp_path / "refused"
    with pytest.raises(ValueError, match="allow-paid-api"):
        bg.run(bg_run_args(out, proposer="anthropic"))
    assert not out.exists()


def test_boxinggym_run_admits_an_injected_anthropic_candidate_for_death(tmp_path, monkeypatch):
    """Packet out as the user message, SYSTEM = the prompt file bytes,
    transport record complete, candidate judged/admitted through the normal path."""
    bg = bg_stub_fits(monkeypatch)
    pytest.importorskip("boxing_gym.envs.death_process")
    assert (
        hashlib.sha256(count_proposer.PROMPT_FILE.read_bytes()).hexdigest()
        == "f095edf1adb3b6533a9e68e73f98ca1dee3f23cd077e424b092393cb32b93aef"
    )
    # NOT one of the four initial structures (bg.TASKS["death"].initial_structures()
    # includes "b0+b1*tz" itself) - a genuinely new structure, so archive.json has
    # exactly one entry for this expression, unambiguous to look up below. Mirrors
    # how CORE's own test_p1_llm_path_sends_the_packet_and_admits_the_candidate
    # uses "b0+b1*tanh(3*z)" rather than plain "b0+b1*z" for the same reason.
    form = "b0+b1*tanh(tz)"
    reply = json.dumps(
        {
            "expression": form,
            "family": "beta_binomial",
            "parameters": dict(bg.mock_parameters(form)),
            "note": "dispersion",
        }
    )
    client, calls = bg_fake_anthropic_client([reply])
    bg_inject_client(monkeypatch, client)
    out = tmp_path / "p1llm"
    bg.run(bg_run_args(out, proposer="anthropic", allow_paid_api=True))
    assert len(calls) == 1
    packet_text = (out / "packets/t01.json").read_text()
    assert calls[0]["messages"] == [{"role": "user", "content": packet_text}]
    assert calls[0]["system"] == count_proposer.PROMPT_FILE.read_text()
    event = json.loads((out / "proposals/t01.json").read_text())
    assert not event["mock"]
    transport = event["transport"]
    assert transport["status"] == "parsed"
    assert transport["calls"] == 1 and transport["repairs"] == 0
    assert transport["prompt_version"] == "v2.0"
    assert transport["prompt_sha256"] == count_proposer.prompt_identity()["prompt_sha256"]
    assert transport["packet_sha256"] == hashlib.sha256(packet_text.encode()).hexdigest()
    assert transport["packet_bytes"] == len(packet_text.encode())
    assert transport["responses"][0]["raw_text"] == reply
    assert transport["note"] == "dispersion"
    assert transport["rejected_candidates"] == []
    assert event["events"][0]["status"] == "fitted" and event["events"][0]["note"] == "dispersion"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["proposal_accounting"]["llm_calls"] == 1
    assert summary["proposal_accounting"]["llm_repairs"] == 0
    assert summary["paid_api_calls"] == 1 and summary["admissions"] == 1
    assert summary["billing_status"] == "usage_in_llm_transport_records"
    config = json.loads((out / "config.json").read_text())
    assert config["proposer"] == "anthropic" and config["paid_api_calls"] is None
    archived = json.loads((out / "archive.json").read_text())
    admitted = next(item for item in archived if item["expression"] == form)
    assert admitted["note"] == "dispersion" and admitted["outcome"] == "active"


def test_boxinggym_same_round_duplicate_gets_a_collision_safe_archive_key(tmp_path, monkeypatch):
    """An injected proposer that repeats one model_id inside
    a single round must not collide archive keys - entry count equals event
    count, the first entry survives untouched, and only the second (third,
    ...) gets a `_k{n}` suffix. `boxinggym_core.run` has no `proposal_limit`
    cap at all (unlike CORE's `core_count.run`, where P4 fixes it at 3), so
    an injected proposer returning several proposals from one call is by
    itself enough to put more than one candidate in a single round - the
    general "any path that admits several candidates per round" case."""
    bg = bg_stub_fits(monkeypatch)
    pytest.importorskip("boxing_gym.envs.death_process")
    calls = []
    real_key_fn = bg.duplicate_archive_key

    def spy(archive, t, key):
        """Record the call and delegate to the wrapped function."""
        result = real_key_fn(archive, t, key)
        calls.append((archive, t, key, result))
        return result

    monkeypatch.setattr(bg, "duplicate_archive_key", spy)

    form = "b0+b1*tanh(tz)"
    spec = bg.TASKS["death"]
    grammar_names = tuple(spec.input_names) + tuple(spec.derived_inputs)
    first = bg.Hypothesis(
        form, "beta_binomial", bg.mock_parameters(form), input_names=grammar_names
    )

    class FixedProposer:
        """Proposer stub that always returns the same fixed proposal."""
        def __init__(self):
            self.round = 0
            self.attempts = 0

        def propose_one(self, packet_text, **kwargs):
            """Return the fixed proposal for the given packet text."""
            self.round += 1
            self.attempts += 1
            proposals = [first] if self.round == 1 else [first, first, first]
            transport_record = {
                "provider": "none",
                "mock": False,
                "api_request_attempted": True,
                "policy": "P1",
                "status": "parsed",
                "calls": 1,
                "repairs": 0,
                "rejected_candidates": [],
                "note": "",
            }
            return proposals, transport_record

    monkeypatch.setattr(count_proposer, "AnthropicCountProposer", lambda *a, **k: FixedProposer())
    out = tmp_path / "dup"
    bg.run(bg_run_args(out, proposer="anthropic", allow_paid_api=True, rounds=2))
    events = json.loads((out / "proposals/t02.json").read_text())["events"]
    duplicate_events = [e for e in events if e["status"] == "duplicate_not_refitted"]
    # `first` was already admitted (active) after round 1, so all three of its
    # round-2 repeats are duplicates of an already-seen model.
    assert len(duplicate_events) == 3
    archive = json.loads((out / "archive.json").read_text())
    duplicate_entries = [
        item
        for item in archive
        if item["outcome"] == "duplicate" and item["model_id"] == first.model_id
    ]
    # Pre-fix this list has length 1: the second and third duplicates' inserts
    # each collide with and overwrite the previous one under the same key.
    assert len(duplicate_entries) == 3
    round2 = [c for c in calls if c[1] == 2]
    assert len(round2) == 3
    base_key = f"duplicate_t02_{first.model_id}"
    assert [c[3] for c in round2] == [base_key, f"{base_key}_k2", f"{base_key}_k3"]
    assert round2[0][0] is round2[1][0] is round2[2][0]  # same dict, mutated in place
    live_archive = round2[0][0]
    assert live_archive[base_key]["outcome"] == "duplicate"
    assert f"{base_key}_k2" in live_archive and f"{base_key}_k3" in live_archive
    summary = json.loads((out / "summary.json").read_text())
    assert summary["proposal_accounting"]["duplicates"] == 3


def test_boxinggym_run_repairs_once_then_admits_for_death(tmp_path, monkeypatch):
    """A malformed reply then a valid one gives calls=2, repairs=1."""
    bg = bg_stub_fits(monkeypatch)
    pytest.importorskip("boxing_gym.envs.death_process")
    form = "b0+b1*tanh(tz)"  # see the sibling test for why not an initial structure
    reply = json.dumps(
        {
            "expression": form,
            "family": "beta_binomial",
            "parameters": dict(bg.mock_parameters(form)),
        }
    )
    client, calls = bg_fake_anthropic_client(["not an object", reply])
    bg_inject_client(monkeypatch, client)
    out = tmp_path / "p1llm_repair"
    bg.run(bg_run_args(out, proposer="anthropic", allow_paid_api=True))
    assert len(calls) == 2
    transport = json.loads((out / "proposals/t01.json").read_text())["transport"]
    assert transport["status"] == "parsed"
    assert transport["calls"] == 2 and transport["repairs"] == 1
    assert len(transport["rejected_candidates"]) == 1
    assert "not valid JSON" in transport["rejected_candidates"][0]["reason"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["proposal_accounting"]["llm_calls"] == 2
    assert summary["proposal_accounting"]["llm_repairs"] == 1
    assert summary["paid_api_calls"] == 2 and summary["admissions"] == 1


def test_boxinggym_run_never_calls_the_anthropic_client_for_s0f0(tmp_path, monkeypatch):
    """The no-proposal family forbids proposals outright (unchanged from earlier versions) and must never
    reach the proposer, mock or anthropic - a poison client proves it here."""
    bg = bg_stub_fits(monkeypatch)
    pytest.importorskip("boxing_gym.envs.death_process")

    def poison(**body):
        """Fail the test if the client is ever called."""
        raise AssertionError("the client must never be called for the fixed-pool cell")

    client = SimpleNamespace(messages=SimpleNamespace(create=poison))
    bg_inject_client(monkeypatch, client)
    out = tmp_path / "s0f0_anthropic"
    bg.run(bg_run_args(out, proposer="anthropic", allow_paid_api=True, cell="000"))
    event = json.loads((out / "proposals/t01.json").read_text())
    # The top-level "mock" key tracks the RUN's proposer choice (False: this run
    # asked for anthropic), same as CORE; the per-round transport["mock"] tracks
    # whether THIS round made a live call (True: it never did, the no-proposal family short-circuits
    # before the proposer, mock or anthropic, is ever consulted).
    assert not event["mock"]
    assert event["transport"]["mock"] is True
    assert event["transport"]["status"] == "no_candidate"
    assert event["valid_candidate_count"] == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["proposal_accounting"]["llm_calls"] == 0


def test_boxinggym_catalog_exhaustion_for_s0_family_open_via_injected_client(
    tmp_path, monkeypatch
):
    """core_count.candidate_catalog's S0 contract, for this task's own menu: once
    every (initial structure, allowed family) pair is known - active or archived -
    no further call is made. An injected client that raises if over-called proves
    the short-circuit happens before the client is ever touched, not just that the
    accounting matches."""
    bg = bg_stub_fits(monkeypatch)
    pytest.importorskip("boxing_gym.envs.death_process")
    structures = bg.TASKS["death"].initial_structures()
    assert len(structures) == 4
    alternatives = ["beta_binomial", "zi_binomial"]  # allowed minus the initial "binomial"
    pairs = [(form, family) for family in alternatives for form in structures]
    assert len(pairs) == 8
    replies = [
        json.dumps(
            {"expression": form, "family": family, "parameters": dict(bg.mock_parameters(form))}
        )
        for form, family in pairs
    ]
    calls = []

    def create(**body):
        """Fake client call: record the request body and return a canned reply."""
        if len(calls) >= len(replies):
            raise AssertionError("propose_one must not be called once the S0 catalog is exhausted")
        calls.append(body)
        text = replies[len(calls) - 1]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(
                model_dump=lambda **kwargs: {"input_tokens": 9, "output_tokens": 3}
            ),
            id=f"test-only-{len(calls)}",
            model="returned-test-model",
            stop_reason="end_turn",
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    bg_inject_client(monkeypatch, client)
    out = tmp_path / "s0f1_catalog"
    bg.run(bg_run_args(out, rounds=10, cell="011", proposer="anthropic", allow_paid_api=True))
    assert len(calls) == 8
    summary = json.loads((out / "summary.json").read_text())
    assert summary["catalog_exhausted_rounds"] == [9, 10]
    accounting = summary["proposal_accounting"]
    assert accounting["catalog_exhausted"] == 2
    assert accounting["llm_calls"] == 8 and accounting["valid_candidates"] == 8
    assert accounting["proposal_rounds_executed"] == 10
    transport9 = json.loads((out / "proposals/t09.json").read_text())["transport"]
    assert transport9["status"] == "catalog_exhausted" and transport9["calls"] == 0
    assert transport9["catalog_size"] == 12
    assert summary["paid_api_calls"] == 8


# --------------------------------------------------------------------------
# 5b. the OpenAI proposer wiring - mirrors section 5
# above one for one: same gate, same injected-client admission, plus the
# runner's label-forcing rule and its key masking.
# --------------------------------------------------------------------------


def bg_fake_openai_client(replies):
    """Injected OpenAI client that replies with `replies` in order; mirrors
    bg_fake_anthropic_client above - no credentials, no network."""
    calls = []

    def create(**body):
        """Fake client call: record the request body and return a canned reply."""
        calls.append(body)
        text = replies[min(len(calls) - 1, len(replies) - 1)]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")
            ],
            usage=SimpleNamespace(
                prompt_tokens=9,
                completion_tokens=3,
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
            id=f"test-only-openai-{len(calls)}",
            model="returned-test-model",
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def bg_inject_openai_client(monkeypatch, client):
    """Same trick as bg_inject_client, for the count_proposer.OpenAICountProposer
    module attribute - a real openai.OpenAI() is never constructed in this file."""
    monkeypatch.setattr(
        count_proposer,
        "OpenAICountProposer",
        lambda records, **kwargs: OpenAICountProposer(records, client=client),
    )


def test_boxinggym_run_refuses_openai_proposer_without_allow_paid_api(tmp_path):
    """OpenAI: refused exactly as anthropic, before --output
    is even created."""
    bg = adapter()
    out = tmp_path / "refused_openai"
    with pytest.raises(ValueError, match="allow-paid-api"):
        bg.run(bg_run_args(out, proposer="openai"))
    assert not out.exists()


def test_boxinggym_run_admits_an_injected_openai_candidate_for_death_and_forces_development_label(
    tmp_path, monkeypatch
):
    """Same contract as test_boxinggym_run_admits_an_injected_anthropic_
    candidate_for_death, over the OpenAI transport, plus the label-forcing
    rule added here: config.json/summary.json/points.csv all read
    "development" for an openai run (bg_run_args defaults to --profile dev
    already, so this also cross-checks against the anthropic run above having
    no "label" key in config.json/summary.json at all - the key is added only
    when it is actually forced, never unconditionally)."""
    bg = bg_stub_fits(monkeypatch)
    pytest.importorskip("boxing_gym.envs.death_process")
    form = "b0+b1*tanh(tz)"
    reply = json.dumps(
        {
            "expression": form,
            "family": "beta_binomial",
            "parameters": dict(bg.mock_parameters(form)),
            "note": "dispersion",
        }
    )
    client, calls = bg_fake_openai_client([reply])
    bg_inject_openai_client(monkeypatch, client)
    out = tmp_path / "p1llm_openai"
    bg.run(bg_run_args(out, proposer="openai", llm_model="gpt-4o", allow_paid_api=True))
    assert len(calls) == 1
    packet_text = (out / "packets/t01.json").read_text()
    assert calls[0]["messages"] == [
        {"role": "system", "content": count_proposer.PROMPT_FILE.read_text()},
        {"role": "user", "content": packet_text},
    ]
    event = json.loads((out / "proposals/t01.json").read_text())
    assert not event["mock"]
    transport = event["transport"]
    assert transport["status"] == "parsed" and transport["provider"] == "openai"
    assert transport["calls"] == 1 and transport["repairs"] == 0
    last_call = {"input_tokens": 9, "output_tokens": 3, "cached_tokens": 0}
    assert transport["last_call_usage"] == last_call
    assert transport["usage_total"] == last_call and transport["usage_total_complete"]
    assert transport["completion_id"] == "test-only-openai-1"
    assert transport["note"] == "dispersion"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["proposal_accounting"]["llm_calls"] == 1
    assert summary["paid_api_calls"] == 1 and summary["admissions"] == 1
    config = json.loads((out / "config.json").read_text())
    assert config["proposer"] == "openai" and config["paid_api_calls"] is None
    assert config["label"] == "development" and summary["label"] == "development"
    points = (out / "points.csv").read_text().splitlines()
    header = points[0].split(",")
    label_col = header.index("label")
    assert points[1:] and all(row.split(",")[label_col] == "development" for row in points[1:])

    # Byte-invariance cross-check: a plain mock run never gets a "label" key
    # in config.json/summary.json at all - the key is added only when it is
    # actually forced, never unconditionally.
    mock_out = tmp_path / "mock_death"
    bg.run(bg_run_args(mock_out, rounds=1))
    mock_config = json.loads((mock_out / "config.json").read_text())
    mock_summary = json.loads((mock_out / "summary.json").read_text())
    assert "label" not in mock_config and "label" not in mock_summary


def test_boxinggym_run_never_leaks_the_openai_api_key_across_any_output_file(
    tmp_path, monkeypatch
):
    """Key-masking requirement (21g), full pipeline for the BoxingGym runner:
    walks every file under a completed boxinggym_core.run output directory
    and asserts the raw OPENAI_API_KEY value is absent from all of them,
    while the masked placeholder shows up at least once. The injected
    client's `create()` raises with the key embedded (simulating an SDK
    error echoing a header) - one call, no repair (API-call-level failures
    never retry - see the CORE-chain contract in test_count_proposer.py);
    the run itself still completes (a failed proposal round is recorded in
    candidate_failures, not fatal)."""
    bg = bg_stub_fits(monkeypatch)
    pytest.importorskip("boxing_gym.envs.death_process")
    fake_key = "sk-test-FAKEKEY-21g-bgfullrun"
    monkeypatch.setenv("OPENAI_API_KEY", fake_key)
    calls = []

    def create(**body):
        """Fake client call: record the request body and return a canned reply."""
        calls.append(body)
        raise RuntimeError(f"upstream echoed Authorization: Bearer {fake_key}")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    bg_inject_openai_client(monkeypatch, client)
    out = tmp_path / "leak_walk_bg"
    bg.run(bg_run_args(out, rounds=1, proposer="openai", llm_model="gpt-4o", allow_paid_api=True))
    assert len(calls) == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["candidate_failures"] and summary["label"] == "development"
    found_masked = False
    for path in out.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        assert fake_key not in text, f"leaked key in {path}"
        if "<OPENAI_API_KEY:masked>" in text:
            found_masked = True
    assert found_masked


# --------------------------------------------------------------------------
# 6. allowed-family parametrization
# --------------------------------------------------------------------------


def bg_admits_family(monkeypatch, tmp_path, task, form, family, out_name):
    """Shared body for the four admission tests below: one injected reply for
    `task`'s packet proposing (`form`, `family`), cell="111" (both structure and
    family expansion open, so neither S0 nor F0 can produce a competing
    rejection reason - the only question is whether `family` clears the task's
    own allowed-family menu). Returns (summary, transport, archived-entry)."""
    bg = bg_scalar_stub_fits(monkeypatch)
    pytest.importorskip(f"boxing_gym.envs.{task}")  # peregrines/dugongs: module stem == task name
    reply = json.dumps(
        {"expression": form, "family": family, "parameters": dict(bg.mock_parameters(form))}
    )
    client, calls = bg_fake_anthropic_client([reply])
    bg_inject_client(monkeypatch, client)
    out = tmp_path / out_name
    bg.run(bg_run_args(out, task=task, proposer="anthropic", allow_paid_api=True))
    assert len(calls) == 1
    transport = json.loads((out / "proposals/t01.json").read_text())["transport"]
    summary = json.loads((out / "summary.json").read_text())
    archived = json.loads((out / "archive.json").read_text())
    admitted = next(item for item in archived if item["expression"] == form)
    return summary, transport, admitted


def test_boxinggym_run_admits_an_injected_anthropic_poisson_candidate_for_peregrines(
    tmp_path, monkeypatch
):
    """Peregrines' initial family (poisson) - earlier versions could never
    admit this at all, because count_proposer's Hypothesis fixed CORE's three
    count families regardless of task; that check is now parametrized by the
    packet's constraints.allowed_families (here ["poisson", "negative_binomial"],
    see test_interface_families_intersects_the_menu_with_what_is_runnable)."""
    summary, transport, admitted = bg_admits_family(
        monkeypatch, tmp_path, "peregrines", "b0+b1*tanh(ttz)", "poisson", "p1llm_peregrines_poisson"
    )
    assert transport["status"] == "parsed"
    assert transport["calls"] == 1 and transport["repairs"] == 0
    assert transport["rejected_candidates"] == []
    assert admitted["family"] == "poisson" and admitted["outcome"] == "active"
    assert summary["admissions"] == 1 and summary["paid_api_calls"] == 1


def test_boxinggym_run_admits_an_injected_anthropic_negative_binomial_candidate_for_peregrines(
    tmp_path, monkeypatch
):
    """Same as above for the REVISION family: negative_binomial is not
    peregrines' initial family, so this also exercises F0 staying open on
    cell="111" (family_expansion=1) - and proves negative_binomial is actually
    implemented end to end (positive_inference.build_model's pm.NegativeBinomial
    branch, SCALAR_NOISE["negative_binomial"] == ("alpha",)), not merely
    declared by the paper's task table."""
    summary, transport, admitted = bg_admits_family(
        monkeypatch,
        tmp_path,
        "peregrines",
        "b0+b1*tanh(2*ttz)",
        "negative_binomial",
        "p1llm_peregrines_negbinom",
    )
    assert transport["status"] == "parsed"
    assert transport["calls"] == 1 and transport["repairs"] == 0
    assert transport["rejected_candidates"] == []
    assert admitted["family"] == "negative_binomial" and admitted["outcome"] == "active"
    assert summary["admissions"] == 1 and summary["paid_api_calls"] == 1


def test_boxinggym_run_admits_an_injected_anthropic_lognormal_candidate_for_dugongs(
    tmp_path, monkeypatch
):
    """dugongs' initial family (lognormal); allowed_families here is
    ["lognormal", "gamma"] (positive outcome, the paper's task table row 4)."""
    summary, transport, admitted = bg_admits_family(
        monkeypatch, tmp_path, "dugongs", "b0+b1*tanh(agez)", "lognormal", "p1llm_dugongs_lognormal"
    )
    assert transport["status"] == "parsed"
    assert transport["calls"] == 1 and transport["repairs"] == 0
    assert transport["rejected_candidates"] == []
    assert admitted["family"] == "lognormal" and admitted["outcome"] == "active"
    assert summary["admissions"] == 1 and summary["paid_api_calls"] == 1


def test_boxinggym_run_admits_an_injected_anthropic_gamma_candidate_for_dugongs(
    tmp_path, monkeypatch
):
    """dugongs' revision family."""
    summary, transport, admitted = bg_admits_family(
        monkeypatch, tmp_path, "dugongs", "b0+b1*tanh(2*agez)", "gamma", "p1llm_dugongs_gamma"
    )
    assert transport["status"] == "parsed"
    assert transport["calls"] == 1 and transport["repairs"] == 0
    assert transport["rejected_candidates"] == []
    assert admitted["family"] == "gamma" and admitted["outcome"] == "active"
    assert summary["admissions"] == 1 and summary["paid_api_calls"] == 1


def test_boxinggym_run_rejects_a_family_outside_the_tasks_allowed_menu(tmp_path, monkeypatch):
    """The negative case: "gamma" is implemented (IMPLEMENTED_FAMILIES)
    and would fit peregrines' own grammar fine, but it is not in peregrines'
    OWN allowed menu (["poisson", "negative_binomial"] - unbounded_count, not
    positive). judge_candidate's new allowed_families pre-check rejects it with
    the SAME literal reason Hypothesis.__post_init__ has always raised for a
    family it does not recognize AT ALL ("unsupported count
    family: <family>") - the repair reply repeats the same mistake, so this
    also proves the run survives to proposal_failed rather than crashing."""
    bg = bg_scalar_stub_fits(monkeypatch)
    pytest.importorskip("boxing_gym.envs.peregrines")
    form = "b0+b1*tanh(3*ttz)"  # NOT one of the four initial structures; grammar is fine
    reply = json.dumps(
        {"expression": form, "family": "gamma", "parameters": dict(bg.mock_parameters(form))}
    )
    client, calls = bg_fake_anthropic_client([reply])
    bg_inject_client(monkeypatch, client)
    out = tmp_path / "p1llm_peregrines_disallowed_family"
    bg.run(bg_run_args(out, task="peregrines", proposer="anthropic", allow_paid_api=True))
    assert len(calls) == 2  # original attempt + the one repair, both rejected
    transport = json.loads((out / "proposals/t01.json").read_text())["transport"]
    assert transport["status"] == "proposal_failed"
    assert transport["calls"] == 2 and transport["repairs"] == 1
    assert len(transport["rejected_candidates"]) == 2
    for rejection in transport["rejected_candidates"]:
        assert rejection["reason"] == "unsupported count family: gamma"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["admissions"] == 0 and summary["paid_api_calls"] == 2
    archived = json.loads((out / "archive.json").read_text())
    assert not any(item["expression"] == form for item in archived)


# ---------------------------------------------------------------------------
# A multi-component task calls the
# proposer once per component in the SAME round, so the transport record name
# has to carry the component or the second component collides on a write-once
# file. Single-outcome tasks keep the earlier names and record keys exactly.
# ---------------------------------------------------------------------------

def bg_packet_driven_client():
    """Injected client whose candidate is read off the packet it is sent.

    Nothing here is a stored transcript: the packet states its own
    `constraints.initial_structures` and `constraints.allowed_families`, so the
    reply is legal for whatever task, namespace and cell the run uses, and the
    test does not have to hard-code one task's grammar.
    """
    calls = []

    def create(**body):
        """Fake client call: record the request body and return a canned reply."""
        calls.append(body)
        constraints = json.loads(body["messages"][0]["content"])["constraints"]
        expression = list(constraints["initial_structures"])[(len(calls) - 1) % 4]
        reply = json.dumps({
            "expression": expression,
            "family": constraints["allowed_families"][0],
            "parameters": dict(bg_adapter_mock_parameters(expression)),
        })
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=reply)],
            usage=SimpleNamespace(
                model_dump=lambda **kwargs: {"input_tokens": 9, "output_tokens": 3}),
            id=f"test-only-{len(calls)}",
            model="returned-test-model",
            stop_reason="end_turn",
        )

    return SimpleNamespace(messages=SimpleNamespace(create=create)), calls


def bg_adapter_mock_parameters(expression):
    """Return mock parameters for the given expression, for the BoxingGym adapter stub."""
    return adapter().mock_parameters(expression)


@pytest.mark.parametrize("cell", ["100", "101"])
def test_emotion_writes_one_transport_record_per_round_and_component(tmp_path, monkeypatch, cell):
    """The reproduction case at runner level: 8 components x 2 rounds, 0 collisions."""
    bg = bg_scalar_stub_fits(monkeypatch)
    pytest.importorskip("boxing_gym.envs.emotion")
    client, calls = bg_packet_driven_client()
    bg_inject_client(monkeypatch, client)
    out = tmp_path / f"emotion{cell}"
    bg.run(bg_run_args(out, task="emotion", cell=cell, rounds=2,
                       proposer="anthropic", allow_paid_api=True))
    components = bg.TASKS["emotion"].components
    assert len(components) == 8
    expected = sorted(
        f"round_{t:02d}__{component}.{kind}.json"
        for t in (1, 2) for component in components for kind in ("request", "response")
    )
    assert sorted(p.name for p in (out / "llm_transport").iterdir()) == expected
    assert len(calls) == 16
    for t in (1, 2):
        for component in components:
            record = json.loads(
                (out / "llm_transport" / f"round_{t:02d}__{component}.response.json").read_text())
            assert record["component"] == component and record["round"] == t
    summary = json.loads((out / "summary.json").read_text())
    assert summary["paid_api_calls"] == 16
    assert summary["proposal_accounting"]["llm_calls"] == 16


def test_a_single_outcome_task_keeps_the_pre_v2_20_record_names(tmp_path, monkeypatch):
    """death: `round_01.*` byte for byte, and no `component` key anywhere."""
    bg = bg_stub_fits(monkeypatch)
    pytest.importorskip("boxing_gym.envs.death_process")
    client, calls = bg_packet_driven_client()
    bg_inject_client(monkeypatch, client)
    out = tmp_path / "death"
    bg.run(bg_run_args(out, task="death", cell="100", rounds=1,
                       proposer="anthropic", allow_paid_api=True))
    assert sorted(p.name for p in (out / "llm_transport").iterdir()) == [
        "round_01.request.json", "round_01.response.json",
    ]
    assert len(calls) == 1
    record = json.loads((out / "llm_transport" / "round_01.response.json").read_text())
    assert "component" not in record
    request = json.loads((out / "llm_transport" / "round_01.request.json").read_text())
    assert "component" not in request["identity"]
    assert "component" not in json.loads((out / "proposals/t01.json").read_text())


def test_emotion_is_the_only_task_that_declares_several_output_components():
    """A future multi-component task must be noticed, not silently added."""
    bg = adapter()
    multi = {name: spec.components for name, spec in bg.TASKS.items() if len(spec.components) > 1}
    assert set(multi) == {"emotion"}, multi
    assert multi["emotion"] == (
        "happiness", "sadness", "anger", "surprise", "fear", "disgust",
        "contentment", "disappointment",
    )
    assert all(len(spec.components) in (0, 1) or name == "emotion"
               for name, spec in bg.TASKS.items())
