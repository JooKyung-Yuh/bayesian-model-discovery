"""Categorical inputs, and the three new adapters.

What is under test, in order:

  1. the ONE grammar addition - an indexed coefficient `a[<categorical input>]`:
     parse, level-count validation, the shared role prior, the coefficient-slot
     accounting, the model identity, and every rejection reason;
  2. the `irt` adapter, including the pre-declared `design_exclusion="observed"`
     rule and the explicit removal of the cells the environment REFUSES
     (`restricted_pair`);
  3. the `moral_machines` adapter's seven attribute contrasts and its 1/2 choice
     coding;
  4. the `emotion` adapter's eight component-wise real runners and the composite
     score, plus the exact prediction string its evaluator's regex needs;
  5. leakage invariance (the packet must not depend on the held-out set) for all three new packets;
  6. that the environment's own evaluator and our held-out scoring read the SAME
     points, asserted rather than assumed;
  7. that `lotka_volterra` is registered and refused with its stated reason
     (in test_named_inputs.py, next to the rest of the registry assertions).

Every environment call here goes through the real boxing-gym environments,
read-only, with `openai.OpenAI` replaced by the adapter's own local stub - so
this module makes zero paid API calls and needs no API key.
"""

import contextlib
import hashlib
import importlib
import json
import re

import numpy as np
import pytest

from .belief_packet import (
    OPTIONAL_CONSTRAINT_KEYS,
    PoolMember,
    build_belief_packet,
    cell_constraints,
    interface_grid,
    packet_digest,
    serialize_packet,
    task_interface,
)
from .count_models import (
    CATEGORICAL_KIND,
    INDEXED_COEFFICIENT_ROLE,
    PRIOR_RULES,
    Hypothesis,
    canonical_model_id,
    evaluate,
    indexed_parameters,
    input_index_codes,
    interface_categorical_levels,
    interface_input_names,
    judge_candidate,
    particle_logpmf,
    validate_expression,
)


def adapter():
    """Helper: adapter; returns the object the tests use."""
    pytest.importorskip("pymc")
    return importlib.import_module(f"{__package__}.boxinggym_core")


def two_categorical(levels=6):
    """A 6x6 all-categorical binary interface - irt's own shape."""
    return task_interface(
        "binary",
        [
            {
                "name": "student",
                "range": [0, levels - 1],
                "kind": CATEGORICAL_KIND,
                "levels": list(range(levels)),
            },
            {
                "name": "question",
                "range": [0, levels - 1],
                "kind": CATEGORICAL_KIND,
                "levels": list(range(levels)),
            },
        ],
        {},
        (0, 1),
        1,
    )


def mixed_interface():
    """One categorical and one numeric input, for the non-categorical-index test."""
    return task_interface(
        "binary",
        [
            {"name": "arm", "range": [0, 2], "kind": CATEGORICAL_KIND, "levels": [0, 1, 2]},
            {"name": "dose", "range": [0.0, 10.0]},
        ],
        {"dosez": "2*(dose-0.0)/(10.0-0.0)-1"},
        (0, 1),
        1,
    )


# --------------------------------------------------------------------------
# 1. the grammar
# --------------------------------------------------------------------------
def test_indexed_coefficient_parses_and_is_the_only_added_form():
    """Test: indexed coefficient parses and is the only added form."""
    levels = interface_categorical_levels(two_categorical())
    assert levels == (("student", 6), ("question", 6))
    names = interface_input_names(two_categorical())
    for expression in (
        "b0+b1[student]",
        "b0+b1[question]",
        "b0+b1[student]+b2[question]",
        "b0+b1[student]*b2[question]",
        "b0+tanh(b1[student])",
        "b1[student]",
        "b0+b1[student]*(b2[student]-b3[question])",
    ):
        used = sorted(set(re.findall(r"b\d+", expression)))
        roles = tuple((name, "intercept" if name == "b0" else "signed") for name in used)
        validate_expression(expression, roles, names, levels)


def test_the_grammar_is_unchanged_when_no_input_is_categorical():
    """The default empty block is what keeps CORE and every earlier task byte-identical.

    With no categorical input declared, a subscript is still REJECTED - earlier
    versions rejected it as `TypeError: unsupported expression syntax`, the
    current grammar rejects it by naming the index instead. The admissibility
    verdict is identical; only the reason STRING of an expression no earlier
    task can write differs, which is a deliberate change.
    """
    with pytest.raises(ValueError, match="non-categorical index u"):
        validate_expression("b0+b1[u]", (("b0", "intercept"), ("b1", "signed")))
    with pytest.raises(ValueError, match="unknown input name w"):
        validate_expression("b0+b1[w]", (("b0", "intercept"), ("b1", "signed")))
    with pytest.raises(ValueError, match="an indexed coefficient must be a declared coefficient"):
        validate_expression("b0+u[z]", (("b0", "intercept"),))
    # and every CORE expression still validates with the default arguments
    for form in ("b0", "b0+b1*z", "b0+b1*z+b2*square(z)", "b0+b1*log(1+u)"):
        used = sorted(set(re.findall(r"b\d+", form)))
        validate_expression(
            form, tuple((n, "intercept" if n == "b0" else "signed") for n in used)
        )


def test_a_categorical_input_may_only_index_a_coefficient():
    """Test: a categorical input may only index a coefficient."""
    levels = interface_categorical_levels(two_categorical())
    names = interface_input_names(two_categorical())
    with pytest.raises(ValueError, match="categorical input student may only index a coefficient"):
        validate_expression(
            "b0+b1*student", (("b0", "intercept"), ("b1", "signed")), names, levels
        )


def test_a_non_categorical_index_is_rejected_by_name():
    """The verbatim reason the per-item judge turns into a candidate rejection."""
    interface = mixed_interface()
    levels, names = interface_categorical_levels(interface), interface_input_names(interface)
    assert levels == (("arm", 3),)
    validate_expression("b0+b1[arm]", (("b0", "intercept"), ("b1", "signed")), names, levels)
    for index in ("dose", "dosez"):
        with pytest.raises(ValueError, match=f"non-categorical index {index}"):
            validate_expression(
                f"b0+b1[{index}]", (("b0", "intercept"), ("b1", "signed")), names, levels
            )
    with pytest.raises(ValueError, match="unknown input name nope"):
        validate_expression(
            "b0+b1[nope]", (("b0", "intercept"), ("b1", "signed")), names, levels
        )
    with pytest.raises(TypeError, match="a coefficient index must be a categorical input name"):
        validate_expression("b0+b1[2]", (("b0", "intercept"), ("b1", "signed")), names, levels)


def test_an_indexed_coefficient_must_carry_the_shared_signed_role():
    """Test: an indexed coefficient must carry the shared signed role."""
    levels = interface_categorical_levels(two_categorical())
    names = interface_input_names(two_categorical())
    assert INDEXED_COEFFICIENT_ROLE == "signed"
    for role in ("intercept", "positive", "shape"):
        with pytest.raises(ValueError, match="an indexed coefficient must have role signed"):
            validate_expression("b1[student]", (("b1", role),), names, levels)


def test_a_coefficient_cannot_be_both_indexed_and_bare():
    """Test: a coefficient cannot be both indexed and bare."""
    levels = interface_categorical_levels(two_categorical())
    names = interface_input_names(two_categorical())
    with pytest.raises(ValueError, match="b1 is used both indexed and unindexed"):
        validate_expression(
            "b0+b1[student]+b1", (("b0", "intercept"), ("b1", "signed")), names, levels
        )


# --------------------------------------------------------------------------
# A coefficient re-indexed by a SECOND, different
# categorical input must be a candidate/format rejection AT VALIDATION TIME -
# the same stage as every other rejection above - not a `ValueError` first
# reachable from `Hypothesis.as_dict()` after `judge_candidate`/
# `parse_candidate_object` already said the candidate was fine. Two
# contrasting cases, both from the defect's own repro:
# `{"expression":"a[student]-a[question]","family":"bernoulli",
# "parameters":{"a":"signed"}}`.
# --------------------------------------------------------------------------
def test_repeating_the_same_index_is_not_a_conflict():
    """`a[student]+a[student]` binds `a` to `student` twice - still one input."""
    levels = interface_categorical_levels(two_categorical())
    names = interface_input_names(two_categorical())
    roles = (("a", "signed"),)
    validate_expression("a[student]+a[student]", roles, names, levels)
    hypothesis = Hypothesis("a[student]+a[student]", "bernoulli", roles, names, levels)
    assert hypothesis.indexed_parameters == (("a", "student", 6),)
    assert hypothesis.as_dict()["indexed_parameters"] == {
        "a": {"input": "student", "n_levels": 6, "prior_shared_across_levels": True}
    }
    structures = [Hypothesis("b0", "bernoulli", (("b0", "intercept"),), names, levels)]
    ok, reason = judge_candidate(
        {"expression": "a[student]+a[student]", "family": "bernoulli", "parameters": {"a": "signed"}},
        "111",
        structures,
        "bernoulli",
        allowed_families=["bernoulli"],
    )
    assert (ok, reason) == (True, None)


def test_a_coefficient_indexed_by_two_different_inputs_is_rejected_at_validation():
    """The repro: rejected by `validate_expression`, `judge_candidate` and
    `parse_candidate_object` alike - never reaching `as_dict()` unrejected.
    """
    levels = interface_categorical_levels(two_categorical())
    names = interface_input_names(two_categorical())
    roles = (("a", "signed"),)
    reason_text = "coefficient a is indexed by two different inputs"
    structures = [Hypothesis("b0", "bernoulli", (("b0", "intercept"),), names, levels)]
    for expression in ("a[student]+a[question]", "a[student]-a[question]"):
        with pytest.raises(ValueError, match=re.escape(reason_text)):
            validate_expression(expression, roles, names, levels)
        with pytest.raises(ValueError, match=re.escape(reason_text)):
            Hypothesis(expression, "bernoulli", roles, names, levels)
        ok, reason = judge_candidate(
            {"expression": expression, "family": "bernoulli", "parameters": {"a": "signed"}},
            "111",
            structures,
            "bernoulli",
            allowed_families=["bernoulli"],
        )
        assert (ok, reason) == (False, reason_text), (expression, ok, reason)
        from .count_proposer import parse_candidate_object

        payload = json.dumps(
            {"expression": expression, "family": "bernoulli", "parameters": {"a": "signed"}}
        )
        with pytest.raises(ValueError, match=re.escape(reason_text)):
            parse_candidate_object(
                payload, "111", "bernoulli", structures, allowed_families=["bernoulli"]
            )


def test_level_count_validation_of_the_interface_block():
    """Test: level count validation of the interface block."""
    ok = two_categorical()
    assert interface_input_names(ok) == ("student", "question")
    def block(**over):
        item = {
            "name": "student",
            "range": [0, 5],
            "kind": CATEGORICAL_KIND,
            "levels": [0, 1, 2, 3, 4, 5],
        }
        item.update(over)
        return {**ok, "inputs": [item]}

    with pytest.raises(ValueError, match="needs at least two levels"):
        interface_input_names(block(levels=[0], range=[0, 1]))
    with pytest.raises(ValueError, match="duplicate levels"):
        interface_input_names(block(levels=[0, 0, 2]))
    with pytest.raises(ValueError, match=r"must declare range \[0, n_levels-1\]"):
        interface_input_names(block(range=[0, 9]))
    with pytest.raises(ValueError, match=r"must declare range \[0, n_levels-1\]"):
        interface_input_names(block(range=[1, 6]))
    with pytest.raises(ValueError, match="unknown kind"):
        interface_input_names(block(kind="ordinal"))
    with pytest.raises(ValueError, match="numeric input student must not declare levels"):
        interface_input_names({**ok, "inputs": [{"name": "student", "range": [0, 5], "levels": [0, 1]}]})
    # a categorical level code is a label, so no derived input may transform it
    with pytest.raises(ValueError, match="categorical input student cannot enter derived input"):
        interface_input_names({**ok, "derived_inputs": {"s2": "student*2"}})


def test_undeclared_level_is_rejected_with_that_phrase():
    """Test: undeclared level is rejected with that phrase."""
    interface = two_categorical()
    codes = input_index_codes(interface, np.array([[0.0, 5.0], [3.0, 1.0]]))
    assert codes["student"].tolist() == [0, 3]
    assert codes["question"].tolist() == [5, 1]
    for bad in ([[0.0, 6.0]], [[-1.0, 0.0]], [[0.5, 0.0]]):
        with pytest.raises(ValueError, match="undeclared level for categorical input"):
            input_index_codes(interface, np.asarray(bad))
    # a numeric-only interface has no codes at all, which is what makes every
    # earlier task's `evaluate` call the earlier, unindexed one.
    assert input_index_codes(task_interface("binary", [{"name": "d", "range": [0, 1]}], {}, (0, 1), 1), np.array([0.5])) == {}


def test_one_indexed_coefficient_is_one_of_the_six_slots():
    """Cap accounting: DECLARED coefficients, not fitted parameters."""
    interface = task_interface(
        "binary",
        [
            {
                "name": f"g{i}",
                "range": [0, 5],
                "kind": CATEGORICAL_KIND,
                "levels": list(range(6)),
            }
            for i in range(7)
        ],
        {},
        (0, 1),
        1,
    )
    names = interface_input_names(interface)
    levels = interface_categorical_levels(interface)
    six = "+".join(f"b{i}[g{i}]" for i in range(6))
    roles = tuple((f"b{i}", "signed") for i in range(6))
    # six indexed coefficients = six slots = 36 fitted parameters, and it passes
    validate_expression(six, roles, names, levels)
    hypothesis = Hypothesis(six, "bernoulli", roles, names, levels)
    assert len(hypothesis.parameters) == 6
    assert sum(n for _, n in hypothesis.parameter_shapes) == 36
    # a seventh is the SAME cap violation an unindexed seventh coefficient is
    seven = six + "+b6[g6]"
    with pytest.raises(ValueError, match=r"require 1\.\.6 distinct mean coefficients"):
        validate_expression(seven, (*roles, ("b6", "signed")), names, levels)


def test_model_id_carries_the_indexed_coefficient_and_its_input():
    """Test: model id carries the indexed coefficient and its input."""
    levels = interface_categorical_levels(two_categorical())
    names = interface_input_names(two_categorical())
    roles = (("b0", "intercept"), ("b1", "signed"))
    a = Hypothesis("b0+b1[student]", "bernoulli", roles, names, levels)
    b = Hypothesis("b0+b1[question]", "bernoulli", roles, names, levels)
    assert a.model_id != b.model_id
    # the canonical form is explicit about both halves
    assert canonical_model_id("b0+b1[student]", "bernoulli", roles) == a.model_id
    # an alpha-rename keeps the identity, a different index does not
    assert (
        Hypothesis("b0+c1[student]", "bernoulli", (("b0", "intercept"), ("c1", "signed")), names, levels).model_id
        == a.model_id
    )
    # and the identity separates an indexed coefficient from an unindexed one
    # written over a NUMERIC input of the same name
    numeric = task_interface("binary", [{"name": "student", "range": [0, 5]}], {}, (0, 1), 1)
    plain = Hypothesis(
        "b0+b1*student", "bernoulli", roles, interface_input_names(numeric), ()
    )
    assert plain.model_id != a.model_id
    assert indexed_parameters("b0+b1[student]", levels) == (("b1", "student", 6),)
    assert indexed_parameters("b0+b1*z", ()) == ()


def test_core_model_ids_are_bit_identical_to_the_v2_9_values():
    """The categorical-input grammar must not move a single CORE identity."""
    assert canonical_model_id(
        "b0+b1*z", "binomial", (("b0", "intercept"), ("b1", "signed"))
    ) == "e2a71965fd9c8137e3b5"
    assert Hypothesis("b0", "binomial", (("b0", "intercept"),)).indexed_parameters == ()
    assert Hypothesis("b0", "binomial", (("b0", "intercept"),)).parameter_shapes == ()
    assert "indexed_parameters" not in Hypothesis(
        "b0+b1*z", "binomial", (("b0", "intercept"), ("b1", "signed"))
    ).as_dict()


def test_the_per_level_offsets_share_one_fixed_prior():
    """Table 3 row 1, shared across levels; no new prior, no changed prior."""
    bg = adapter()
    levels = interface_categorical_levels(two_categorical())
    names = interface_input_names(two_categorical())
    hypothesis = Hypothesis(
        "b0+b1[student]",
        "bernoulli",
        (("b0", "intercept"), ("b1", "signed")),
        names,
        levels,
    )
    record = hypothesis.as_dict()
    assert record["fixed_priors"]["b1"] == PRIOR_RULES["signed"] == ("Normal", 0.0, 1.0)
    assert record["indexed_parameters"] == {
        "b1": {"input": "student", "n_levels": 6, "prior_shared_across_levels": True}
    }
    draws = bg.prior_particles(hypothesis, 20000)
    assert draws["b0"].shape == (20000,)
    assert draws["b1"].shape == (20000, 6)
    # every level column is a Normal(0,1) sample from the SAME rule
    assert np.allclose(draws["b1"].mean(axis=0), 0.0, atol=0.05)
    assert np.allclose(draws["b1"].std(axis=0, ddof=1), 1.0, atol=0.05)


def test_the_pymc_graph_gives_the_indexed_coefficient_one_vector_variable():
    """Test: the pymc graph gives the indexed coefficient one vector variable."""
    pytest.importorskip("pymc")
    from .count_inference import build_model

    interface = two_categorical()
    levels, names = interface_categorical_levels(interface), interface_input_names(interface)
    hypothesis = Hypothesis(
        "b0+b1[student]+b2[question]",
        "bernoulli",
        (("b0", "intercept"), ("b1", "signed"), ("b2", "signed")),
        names,
        levels,
    )
    x = np.array([[0.0, 1.0], [3.0, 5.0], [5.0, 0.0], [2.0, 2.0]])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    model = build_model(hypothesis, x, y, 1, interface)
    shapes = {v.name: tuple(model.initial_point()[f"{v.name}"].shape) for v in model.free_RVs}
    assert shapes == {"b0": (), "b1": (6,), "b2": (6,)}


def test_numpy_and_the_gather_agree_with_an_explicit_per_level_sum():
    """Test: numpy and the gather agree with an explicit per level sum."""
    interface = two_categorical()
    levels, names = interface_categorical_levels(interface), interface_input_names(interface)
    hypothesis = Hypothesis(
        "b0+b1[student]+b2[question]",
        "bernoulli",
        (("b0", "intercept"), ("b1", "signed"), ("b2", "signed")),
        names,
        levels,
    )
    rng = np.random.default_rng(3)
    particles = {
        "b0": rng.normal(size=5),
        "b1": rng.normal(size=(5, 6)),
        "b2": rng.normal(size=(5, 6)),
    }
    designs = np.array([[0.0, 4.0], [2.0, 1.0], [5.0, 5.0]])
    logpmf = particle_logpmf(hypothesis, particles, designs, 1, interface)
    assert logpmf.shape == (5, 3, 2)
    assert np.allclose(np.exp(logpmf).sum(-1), 1.0)
    codes = input_index_codes(interface, designs)
    expected = (
        particles["b0"][:, None]
        + particles["b1"][:, codes["student"]]
        + particles["b2"][:, codes["question"]]
    )
    from scipy.special import expit

    assert np.allclose(np.exp(logpmf[:, :, 1]), expit(expected))
    # the same gather through `evaluate` directly, both layouts
    direct = evaluate(
        "b1[student]",
        {"b1": particles["b1"]},
        np.zeros((1, 3)),
        np,
        {"student": designs[:, 0][None, :], "question": designs[:, 1][None, :]},
        codes,
    )
    assert np.allclose(direct, particles["b1"][:, codes["student"]])
    with pytest.raises(ValueError, match="no level codes supplied"):
        evaluate("b1[student]", {"b1": particles["b1"]}, np.zeros((1, 3)), np, {}, None)


def test_the_reference_grid_only_ever_holds_declared_level_codes():
    """Test: the reference grid only ever holds declared level codes."""
    interface = two_categorical()
    grid = interface_grid(interface, 16)
    assert grid.shape == (16, 2)
    # every value is a declared code, so the prior probe cannot be handed an
    # undeclared level by the grid itself
    input_index_codes(interface, grid)
    assert set(grid[:, 0].tolist()) <= {float(v) for v in range(6)}


# --------------------------------------------------------------------------
# 2. the packet: the rule the byte-frozen SYSTEM prompt cannot carry
# --------------------------------------------------------------------------
def test_the_constraints_state_the_indexed_form_and_its_slot_cost():
    """Test: the constraints state the indexed form and its slot cost."""
    levels = interface_categorical_levels(two_categorical())
    constraints = cell_constraints(
        "111",
        "bernoulli",
        allowed_families=["bernoulli"],
        initial_structures=["b0", "b0+b1[student]"],
        categorical_inputs=levels,
    )
    assert set(OPTIONAL_CONSTRAINT_KEYS) <= set(constraints)
    assert constraints["indexed_coefficients"]["student"] == {
        "n_levels": 6,
        "form": "<coefficient>[student]",
        "role_required": "signed",
        "counts_as_coefficient_slots": 1,
        "parameters_contributed": 6,
    }
    assert "at most six declared coefficient slots" in constraints["coefficient_slot_rule"]
    # and a task with no categorical input emits NEITHER key, which is what
    # keeps every CORE and earlier-task packet byte-identical
    plain = cell_constraints("111", "binomial")
    assert not set(OPTIONAL_CONSTRAINT_KEYS) & set(plain)


def test_the_system_prompt_file_is_untouched():
    """Test: the system prompt file is untouched."""
    from .count_proposer import system_prompt_bytes

    assert (
        hashlib.sha256(system_prompt_bytes()).hexdigest()
        == "f095edf1adb3b6533a9e68e73f98ca1dee3f23cd077e424b092393cb32b93aef"
    )


def test_a_categorical_packet_must_state_its_indexed_block():
    """Test: a categorical packet must state its indexed block."""
    levels = interface_categorical_levels(two_categorical())
    hypothesis = Hypothesis(
        "b0+b1[student]",
        "bernoulli",
        (("b0", "intercept"), ("b1", "signed")),
        interface_input_names(two_categorical()),
        levels,
    )
    rng = np.random.default_rng(1)
    member = PoolMember(
        hypothesis, {"b0": rng.normal(size=8), "b1": rng.normal(size=(8, 6))}, -3.0
    )
    public = [
        {"obs_id": i, "round": 0, "student": float(i % 6), "question": float((i + 2) % 6), "y": i % 2}
        for i in range(6)
    ]
    without = cell_constraints(
        "111", "bernoulli", allowed_families=["bernoulli"], initial_structures=["b0"]
    )
    with pytest.raises(ValueError, match="must state its indexed-coefficient block"):
        build_belief_packet(public, [member], [], without, two_categorical(), 1)
    good = cell_constraints(
        "111",
        "bernoulli",
        allowed_families=["bernoulli"],
        initial_structures=["b0"],
        categorical_inputs=levels,
    )
    packet = build_belief_packet(public, [member], [], good, two_categorical(), 1)
    assert packet["interface"]["inputs"][0]["kind"] == CATEGORICAL_KIND
    assert packet["interface"]["inputs"][0]["levels"] == [0, 1, 2, 3, 4, 5]
    # the offsets are reported PER LEVEL, not as one scalar over the vector
    coefficients = packet["pool"][0]["coefficients"]
    assert coefficients["b0"].keys() == {"mean", "sd"}
    assert coefficients["b1"]["n_levels"] == 6
    assert len(coefficients["b1"]["mean"]) == 6 and len(coefficients["b1"]["sd"]) == 6
    serialize_packet(packet)


def test_the_judge_rejects_a_bad_index_per_item_and_admits_a_good_one():
    """Test: the judge rejects a bad index per item and admits a good one."""
    levels = interface_categorical_levels(two_categorical())
    names = interface_input_names(two_categorical())
    structures = [
        Hypothesis("b0", "bernoulli", (("b0", "intercept"),), names, levels),
        Hypothesis(
            "b0+b1[student]", "bernoulli", (("b0", "intercept"), ("b1", "signed")), names, levels
        ),
    ]
    ok, reason = judge_candidate(
        {
            "expression": "b0+b1[student]+b2[question]",
            "family": "bernoulli",
            "parameters": {"b0": "intercept", "b1": "signed", "b2": "signed"},
        },
        "111",
        structures,
        "bernoulli",
        allowed_families=["bernoulli"],
    )
    assert (ok, reason) == (True, None)
    for expression, parameters, expected in (
        ("b0+b1*student", {"b0": "intercept", "b1": "signed"},
         "categorical input student may only index a coefficient"),
        ("b0+b1[nope]", {"b0": "intercept", "b1": "signed"}, "unknown input name nope"),
        ("b0+b1[student]", {"b0": "intercept", "b1": "positive"},
         "an indexed coefficient must have role signed"),
    ):
        ok, reason = judge_candidate(
            {"expression": expression, "family": "bernoulli", "parameters": parameters},
            "111",
            structures,
            "bernoulli",
            allowed_families=["bernoulli"],
        )
        assert not ok and reason == expected, (expression, reason)


# --------------------------------------------------------------------------
# 3. the irt adapter
# --------------------------------------------------------------------------
def irt_environment(seed=20260908):
    """Helper: irt environment; returns the object the tests use."""
    bg = adapter()
    pytest.importorskip("boxing_gym.envs.irt")
    return bg, bg.TaskEnvironment(bg.TASKS["irt"], seed)


def test_irt_declares_two_categorical_inputs_and_a_36_cell_grid():
    """Test: irt declares two categorical inputs and a 36 cell grid."""
    bg = adapter()
    spec = bg.TASKS["irt"]
    assert spec.categorical_inputs == (("student", 6), ("question", 6))
    interface = spec.interface()
    assert [item["kind"] for item in interface["inputs"]] == [CATEGORICAL_KIND] * 2
    assert interface["derived_inputs"] == {}
    assert spec.numeric_input_names == ()
    assert spec.design_exclusion == "observed"
    assert spec.eval_point_limit == 10
    assert spec.initial_structures() == (
        "b0",
        "b0+b1[student]",
        "b0+b1[question]",
        "b0+b1[student]+b2[question]",
    )
    # every declared form and every mock template is grammar-valid for this task
    levels, names = spec.categorical_levels(), spec.input_names
    for expression in (*spec.initial_structures(), *spec.mock_templates()):
        Hypothesis(expression, "bernoulli", bg.mock_parameters(expression), names, levels)
    assert len(spec.mock_templates()) == 20


def test_irt_pool_excludes_every_refused_cell_and_still_covers_the_run():
    """Test: irt pool excludes every refused cell and still covers the run."""
    bg, environment = irt_environment()
    spec = bg.TASKS["irt"]
    refused = environment.refused_designs()
    # the goal registers ten evaluation cells, possibly with repeats
    assert len(environment.goal.eval_points) == 10
    assert 1 <= len(refused) <= 10
    pool = bg.build_design_pool(spec, environment, bg.design_pool_target(spec, 16, 20))
    rows = {tuple(int(v) for v in row) for row in pool.rows}
    assert len(rows) == len(pool.rows) == 36 - len(refused)
    assert not rows & refused
    # the sizing requirement: 36 - k >= D0 4 + 20 rounds = 24
    assert len(pool.rows) >= len(bg.D0_FRACTIONS) + 20
    assert bg.design_pool_target(spec, 16, 20) <= len(pool.rows)
    # D0 is inside the pool, distinct, and outside the refused set
    d0 = pool.d0_indices()
    assert len(set(d0)) == len(d0) == 4
    assert not {tuple(int(v) for v in pool.rows[i]) for i in d0} & refused


def test_irt_refusal_never_becomes_an_observation():
    """The environment answers a refused cell with a STRING and ok=False."""
    _, environment = irt_environment()
    refused = sorted(environment.refused_designs())
    student, question = refused[0]
    with pytest.raises(ValueError, match="environment rejected design"):
        environment.observe(f"[{student}, {question}]")
    assert environment.observation_calls == 0
    assert environment.env.observed_data == []
    # and the validator alone would have ACCEPTED it - which is exactly why the
    # pool has to remove the refused cells explicitly (irt.py:378-:391 vs :412)
    assert environment.accepts(f"[{student}, {question}]")


def test_irt_eval_points_are_the_environments_own_ten_and_ours_match():
    """Test: irt eval points are the environments own ten and ours match."""
    bg, environment = irt_environment()
    spec = bg.TASKS["irt"]
    x, y = environment.evaluation_points(10, 4242)
    assert x.shape == (10, 2) and y.shape == (10,)
    stored = np.asarray(
        [environment.eval_inputs(p) for p in environment.goal.eval_points], dtype=float
    )
    assert np.array_equal(stored, x)
    assert [float(p[-1]) for p in environment.goal.eval_points] == y.tolist()
    # the environment's own evaluator reads the same stored points
    submitted = bg.format_predictions(spec, np.zeros(10))
    assert submitted == ["0"] * 10
    score = environment.evaluate(submitted)
    assert score["metric"] == pytest.approx(np.mean(y != 0.0))


def test_the_round_loop_never_redraws_or_refuses_an_irt_design():
    """Inject refused cells, drive the exclusion loop.

    This exercises exactly what `run()` does per round - build the pool, take
    D0, then remove every observed index before the next draw - with an
    INJECTED `restricted_pair` so the refused set is chosen by the test rather
    than by the goal's own sampling. Every observation must succeed (no refusal
    string is ever counted) and no design may repeat.
    """
    bg, environment = irt_environment()
    spec = bg.TASKS["irt"]
    injected = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (0, 1), (1, 0), (2, 3), (3, 2)]
    environment.env.restricted_pair = list(injected)
    assert environment.refused_designs() == frozenset(injected)
    pool = bg.build_design_pool(spec, environment, bg.design_pool_target(spec, 16, 20))
    assert len(pool.rows) == 36 - len(set(injected)) == 26
    observed, seen_handles = set(), []
    for index in pool.d0_indices():
        observed.add(index)
        seen_handles.append(pool.handles[index])
        environment.observe(pool.handles[index])
    assert len(observed) == 4
    rng = np.random.default_rng(5)
    for _ in range(20):
        available = pool.available(observed)
        chosen = available[rng.integers(len(available))]
        index = pool.index_of(chosen)
        assert index not in observed
        observed.add(index)
        seen_handles.append(pool.handles[index])
        environment.observe(pool.handles[index])
    assert len(observed) == len(seen_handles) == 24
    assert len(set(seen_handles)) == 24
    assert environment.observation_calls == 24
    assert len(environment.env.observed_data) == 24
    cells = {(s, q) for s, q, _ in environment.env.observed_data}
    assert len(cells) == 24
    assert not cells & set(injected)


# --------------------------------------------------------------------------
# 4. the moral_machines adapter
# --------------------------------------------------------------------------
def test_the_narration_stub_absorbs_the_call_and_restores_openai():
    """Test: the narration stub absorbs the call and restores openai."""
    bg = adapter()
    openai = pytest.importorskip("openai")
    original = openai.OpenAI
    environment_module = pytest.importorskip(bg.TASKS["moral_machines"].module)
    with bg.openai_narration_stub(environment_module.__name__) as log:
        assert openai.OpenAI is original
        client = environment_module.openai.OpenAI()
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user"}])
        assert response.choices[0].message.content
        assert log == [{"model": "gpt-4o", "messages": 1}]
    assert openai.OpenAI is original


def test_moral_contrasts_are_the_environments_own_likelihood_features():
    """Test: moral contrasts are the environments own likelihood features."""
    bg = adapter()
    pytest.importorskip("boxing_gym.envs.moral_machines")
    with bg.openai_narration_stub(bg.TASKS["moral_machines"].module):
        environment = bg.TaskEnvironment(bg.TASKS["moral_machines"], 20260908)
    env = environment.env
    group1, group2, intervention = ("boy", "girl", "dog"), ("old_man",), "swerve"
    row = bg.moral_contrast_row(env, group1, group2, intervention)
    assert len(row) == len(bg.MORAL_CONTRAST_NAMES) == 7
    # recomputed here exactly as moral_machines.py:471-:484 does
    human = set(env.characters[:-2])
    assert row[0] == float(
        len([c for c in group1 if c in human]) - len([c for c in group2 if c in human])
    )
    a1 = [env._get_char_attr(c) for c in group1]
    a2 = [env._get_char_attr(c) for c in group2]
    for offset, key in enumerate(
        ("gender", "age", "social_status", "fitness", "species"), start=1
    ):
        assert row[offset] == float(sum(a[key] for a in a1) - sum(a[key] for a in a2))
    assert row[6] == 1.0
    assert bg.moral_contrast_row(env, group1, group2, "stay")[6] == -1.0
    # and the handle round-trips through the environment's OWN parser
    handle = bg.moral_handle(group1, group2, intervention)
    parsed = env.validate_input(handle)
    assert parsed == {
        "group1": list(group1),
        "group2": list(group2),
        "intervention": intervention,
    }
    assert environment.accepts(handle)


def test_moral_choice_is_recoded_to_bernoulli_and_back():
    """Test: moral choice is recoded to bernoulli and back."""
    bg = adapter()
    spec = bg.TASKS["moral_machines"]
    assert spec.outcome_offset == -1.0
    assert spec.observation_source == "observed_data" and spec.observed_data_index == 3
    assert len(spec.input_names) == 7
    # the environment's 1/2 code becomes the modelled 0/1 ...
    pytest.importorskip("boxing_gym.envs.moral_machines")
    with bg.openai_narration_stub(bg.TASKS["moral_machines"].module):
        environment = bg.TaskEnvironment(spec, 20260908)
    assert environment.outcome(1) == 0.0
    assert environment.outcome(2) == 1.0
    # ... and a submitted prediction is the environment's own code again
    assert bg.format_predictions(spec, np.array([0.1, 0.9])) == ["1", "2"]
    # which is what its own evaluator parses (moral_machines.py:61-:74)
    metric, _ = environment.goal.evaluate_predictions(["1", "2"], [1, 2])
    assert metric == 1.0


def test_moral_eval_points_are_read_through_the_same_feature_map():
    """The environment's evaluator and our scorer share the points."""
    bg = adapter()
    pytest.importorskip("boxing_gym.envs.moral_machines")
    spec = bg.TASKS["moral_machines"]
    with bg.openai_narration_stub(bg.TASKS["moral_machines"].module), bg.env_seed_injection(4242):
        environment = bg.TaskEnvironment(spec, 20260908)
        x, y = environment.evaluation_points(3, 4242)
        stored = np.asarray(
            [environment.eval_inputs(p) for p in environment.goal.eval_points], dtype=float
        )
        assert x.shape == (3, 7) and y.shape == (3,)
        assert np.array_equal(stored, x)
        # the modelled outcome is the environment's 1/2 choice minus one
        assert [float(p[-1]) - 1.0 for p in environment.goal.eval_points] == y.tolist()
        assert set(y.tolist()) <= {0.0, 1.0}
        # the environment's own evaluator scores exactly those stored points
        score = environment.evaluate(bg.format_predictions(spec, y))
        assert score["metric"] == 1.0  # we submitted the truth, so accuracy is 1


def test_the_declared_moral_ranges_contain_every_generator_draw():
    """A design outside its declared range would be refused by `check_designs`."""
    bg = adapter()
    pytest.importorskip("boxing_gym.envs.moral_machines")
    from .count_inference import check_designs

    spec = bg.TASKS["moral_machines"]
    interface = spec.interface()
    with bg.openai_narration_stub(bg.TASKS["moral_machines"].module):
        environment = bg.TaskEnvironment(spec, 20260908)
        np.random.seed(5)
        rows = np.asarray(
            [environment.design_features(environment.env.sample_random_input())[0]
             for _ in range(600)]
        )
    check_designs(rows, interface)
    for index, item in enumerate(interface["inputs"]):
        low, high = item["range"]
        assert low <= rows[:, index].min() and rows[:, index].max() <= high, item["name"]
    # the attainable bounds, from the environment's own attribute map
    assert [item["range"] for item in interface["inputs"]] == [
        [-4.0, 4.0], [-4.0, 4.0], [-4.0, 4.0], [-6.0, 6.0], [-8.0, 8.0], [-8.0, 8.0], [-1.0, 1.0]
    ]


def test_moral_pool_comes_from_the_environments_own_generator():
    """Test: moral pool comes from the environments own generator."""
    bg = adapter()
    pytest.importorskip("boxing_gym.envs.moral_machines")
    spec = bg.TASKS["moral_machines"]
    with bg.openai_narration_stub(bg.TASKS["moral_machines"].module):
        environment = bg.TaskEnvironment(spec, 20260908)
        np.random.seed(101)
        pool = bg.build_design_pool(spec, environment, 16)
    assert len(pool.rows) == 16
    assert pool.rows.shape[1] == 7
    assert len(set(pool.handles)) == 16
    for handle in pool.handles:
        assert environment.accepts(handle)
    assert bg.design_grid_rule(spec) == bg.DESIGN_GRID_RULE_SAMPLED


# --------------------------------------------------------------------------
# 5. the emotion adapter
# --------------------------------------------------------------------------
def test_emotion_declares_eight_components_over_seven_shared_inputs():
    """Test: emotion declares eight components over seven shared inputs."""
    bg = adapter()
    spec = bg.TASKS["emotion"]
    assert spec.input_names == bg.EMOTION_INPUT_NAMES == ("v1", "v2", "v3", "p1", "p2", "p3", "win")
    assert spec.components == (
        "happiness",
        "sadness",
        "anger",
        "surprise",
        "fear",
        "disgust",
        "contentment",
        "disappointment",
    )
    assert spec.outcome_type == "real"
    # seven numeric inputs cannot carry the additive structure inside the
    # six-slot cap, so the pool is the four rotated CORE forms
    assert len(spec.initial_structures()) == 4
    assert bg.ADDITIVE_STRUCTURE_CAP_RULE
    assert spec.eval_point_limit is None


def test_emotion_predictions_parse_under_the_environments_own_regex():
    """Test: emotion predictions parse under the environments own regex."""
    bg = adapter()
    pytest.importorskip("boxing_gym.envs.emotion")
    spec = bg.TASKS["emotion"]
    means = np.array([[4.5, 2.0], [3.0, 1.5], [1.7, 1.0], [3.3, 2.2],
                      [1.4, 1.1], [1.8, 1.2], [3.7, 2.8], [2.0, 6.0]])
    submitted = bg.format_predictions(spec, means[0], component_means=means)
    assert len(submitted) == 2
    with bg.openai_narration_stub(bg.TASKS["emotion"].module):
        environment = bg.TaskEnvironment(spec, 20260908)
    truth = [
        dict(zip(spec.components, [5, 3, 2, 3, 1, 2, 4, 2], strict=True)),
        dict(zip(spec.components, [2, 2, 1, 2, 1, 1, 3, 6], strict=True)),
    ]
    mean, _ = environment.goal.evaluate_predictions(submitted, truth)
    expected = np.mean(
        [
            np.mean(np.abs(means[:, j] - np.array([truth[j][k] for k in spec.components])))
            for j in range(2)
        ]
    )
    assert mean == pytest.approx(expected)


def test_emotion_outcomes_and_eval_points_are_read_as_eight_vectors():
    """Test: emotion outcomes and eval points are read as eight vectors."""
    bg = adapter()
    pytest.importorskip("boxing_gym.envs.emotion")
    spec = bg.TASKS["emotion"]
    with bg.openai_narration_stub(bg.TASKS["emotion"].module), bg.env_seed_injection(4242):
        environment = bg.TaskEnvironment(spec, 20260908)
        x, y = environment.evaluation_points(3, 4242)
        assert x.shape == (3, 7) and y.shape == (3, 8)
        stored = np.asarray(
            [environment.eval_inputs(p) for p in environment.goal.eval_points], dtype=float
        )
        assert np.array_equal(stored, x)
        np.random.seed(77)
        pool = bg.build_design_pool(spec, environment, 8)
        value = environment.observe(pool.handles[0])
    assert len(value) == 8
    assert environment.observation_calls == 1
    # the recorded observation, not the narration string
    recorded = environment.env.observed_data[-1]
    assert list(value) == [float(v) for v in recorded[3]]
    # every emotion is the environment's own 1-9 Likert coding (emotion.py:411-:414)
    assert all(1.0 <= v <= 9.0 for v in value)
    assert bg.EMOTION_WIN_UNIT_DEFECT.startswith("emotion.py:464")


def test_emotion_handle_round_trips_through_the_environments_validator():
    """Test: emotion handle round trips through the environments validator."""
    bg = adapter()
    pytest.importorskip("boxing_gym.envs.emotion")
    spec = bg.TASKS["emotion"]
    with bg.openai_narration_stub(bg.TASKS["emotion"].module):
        environment = bg.TaskEnvironment(spec, 20260908)
    handle = bg.emotion_handle([10.0, 20.0, 70.0], [0.2, 0.3, 0.5], 2)
    prizes, probs, win = environment.env.validate_input(handle)
    assert prizes.tolist() == [10.0, 20.0, 70.0]
    assert probs.tolist() == [0.2, 0.3, 0.5]
    # validate_input hands `step` the PRIZE VALUE, which is the declared input
    assert win == 70.0
    assert bg.emotion_input_row([10.0, 20.0, 70.0], [0.2, 0.3, 0.5], win) == [
        10.0, 20.0, 70.0, 0.2, 0.3, 0.5, 70.0
    ]
    # a triple whose float sum is not exactly 1 is refused by the environment
    assert not environment.accepts(
        bg.emotion_handle([1.0, 2.0, 3.0], [0.3, 0.3, 0.3], 0)
    )


# --------------------------------------------------------------------------
# 6. leakage invariance (held-out-set invariance) on the three new packets
# --------------------------------------------------------------------------
@pytest.mark.parametrize("task", ["irt", "moral_machines", "emotion"])
def test_new_task_packet_is_byte_identical_when_only_the_heldout_set_changes(task):
    """Held-out-set invariance on the three categorical-input tasks: vary ONLY the evaluation set.

    Same shape as `test_named_inputs.test_task_packet_is_byte_identical_when_
    only_the_heldout_set_changes`: the environment's own held-out set is
    replaced between two packet builds while the public observations, the pool
    and the archive are held fixed. The packet bytes must not move, and no
    evaluation vocabulary may appear in them.
    """
    bg = adapter()
    pytest.importorskip("pymc")
    spec = bg.TASKS[task]
    interface = spec.interface()
    namespace = interface_input_names(interface)
    levels = spec.categorical_levels()
    structures = spec.initial_structures()
    initial_family = {"binary": "bernoulli", "real": "normal"}[spec.outcome_type]
    constraints = cell_constraints(
        "111",
        initial_family,
        allowed_families=[initial_family],
        initial_structures=list(structures),
        categorical_inputs=levels,
    )
    rng = np.random.default_rng(11)
    members = []
    for index, form in enumerate(structures[:2]):
        roles = bg.mock_parameters(form)
        if spec.outcome_type == "binary":
            hypothesis = Hypothesis(form, initial_family, roles, namespace, levels)
            shapes = dict(hypothesis.parameter_shapes)
            particles = {
                name: rng.normal(size=(12, shapes[name]) if name in shapes else 12)
                for name, _ in roles
            }
        else:
            from .real_inference import RealHypothesis

            hypothesis = RealHypothesis(form, initial_family, roles, namespace, None)
            particles = {name: rng.normal(size=12) for name, _ in roles}
            particles["sigma"] = np.abs(rng.normal(size=12)) + 0.5
        members.append(
            PoolMember(hypothesis, particles, -float(index + 3), response_unit=1.0)
        )
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
    public = []
    for i in range(6):
        row = {"obs_id": i, "round": 0 if i < 4 else i - 3}
        for j, name in enumerate(spec.input_names):
            low, high = interface["inputs"][j]["range"]
            if interface["inputs"][j].get("kind") == CATEGORICAL_KIND:
                row[name] = float((i + j) % len(interface["inputs"][j]["levels"]))
            else:
                row[name] = float(low + (high - low) * ((i + j) % 4) / 4.0)
        row["y"] = float(i % 2) if spec.outcome_type == "binary" else 1.0 + 0.5 * i
        public.append(row)

    # the builder takes no held-out argument at all - asserted statically
    import inspect

    assert set(inspect.signature(build_belief_packet).parameters) == {
        "public_observations",
        "active_fits",
        "archive",
        "constraints",
        "interface",
        "round",
    }

    texts = []
    pytest.importorskip(spec.module)
    narration = (bg.openai_narration_stub(spec.openai_narration_module)
                 if spec.openai_narration_module else contextlib.nullcontext())
    with narration, bg.env_seed_injection(4242):
        environment = bg.TaskEnvironment(spec, 20260908)
        if task == "irt":
            # irt's ten points are fixed at construction, so the held-out set
            # is REPLACED outright - two disjoint sets of cells and outcomes.
            held_out_sets = [
                [(s, (s + 1) % 6, s % 2) for s in range(6)],
                [(s, (s + 4) % 6, (s + 1) % 2) for s in range(6)],
            ]
            for points in held_out_sets:
                environment.goal.eval_points = list(points)
                environment.goal.eval_pointer = 0
                texts.append(
                    serialize_packet(
                        build_belief_packet(
                            public, members, archive, constraints, interface, 1
                        )
                    )
                )
        else:
            for count, seed in ((2, 111), (4, 222)):
                environment.goal.eval_points.clear()
                environment.goal.eval_pointer = 0
                environment.evaluation_points(count, seed)
                texts.append(
                    serialize_packet(
                        build_belief_packet(
                            public, members, archive, constraints, interface, 1
                        )
                    )
                )
    assert texts[0] == texts[1]
    assert packet_digest(texts[0]) == packet_digest(texts[1])
    for forbidden in ("heldout", "evaluation", "truth", "coverage90", "eval_points"):
        assert forbidden not in texts[0]
    assert json.loads(texts[0])["n_public"] == len(public)


# --------------------------------------------------------------------------
# 7. the declared rules, as constants (so a silent edit fails a test)
# --------------------------------------------------------------------------
def test_the_declared_v2_10_rules_are_recorded_verbatim():
    """Test: the declared v2 10 rules are recorded verbatim."""
    bg = adapter()
    assert bg.DESIGN_GRID_RULE_CATEGORICAL == (
        "full_product_grid_of_the_declared_categorical_levels_minus_environment_refused_designs"
    )
    assert bg.DESIGN_GRID_RULE_SAMPLED == (
        "the_environments_own_sample_random_input_draws_filtered_by_its_own_validate_input"
    )
    assert bg.MULTI_COMPONENT_EIG_RULE == "sum_of_the_per_component_eig_tables"
    assert bg.MULTI_COMPONENT_CALIBRATION_RULE == (
        "per_component_frozen_D0_mean_center_with_one_pooled_D0_sample_sd_scale"
    )
    assert bg.COMPOSITE_SCORE_RULE == "mean_over_the_declared_output_components"
    assert "openai.OpenAI" in bg.OPENAI_NARRATION_STUB_RULE
    assert bg.design_grid_rule(bg.TASKS["death"]) == bg.DESIGN_GRID_RULE
    assert bg.design_grid_rule(bg.TASKS["irt"]) == bg.DESIGN_GRID_RULE_CATEGORICAL
    assert bg.design_grid_rule(bg.TASKS["emotion"]) == bg.DESIGN_GRID_RULE_SAMPLED
    # every earlier task keeps its own grid rule value
    for name in ("death", "peregrines", "dugongs", "survival", "hyperbolic", "location"):
        assert bg.design_grid_rule(bg.TASKS[name]) == bg.DESIGN_GRID_RULE
