"""Model identity: synthetic identity controls, immutable predecessor anchors, no API."""

import ast
import hashlib
import json
from itertools import permutations

import pytest

from . import core_count as core
from .count_models import canonical_model_id, initial_hypotheses
from .model_identity import LEGACY_MODEL_ID_SCHEMA, MODEL_ID_SCHEMA, require_model_id_schema

PAIR = (("b0", "intercept"), ("b1", "signed"))
TRIPLE = (*PAIR, ("b2", "signed"))
BEFORE_AFTER = (
    ("b0+b1*z", "ce80109ed97b168ed85c", "ce80109ed97b168ed85c"),
    ("b1*z+b0", "820ec1c839662842543d", "ce80109ed97b168ed85c"),
    ("b0+z*b1", "efbca5a0dc89d7c46764", "ce80109ed97b168ed85c"),
    # The POW rule covers square(z), so these four after-anchors are
    # recomputed with the current canonical_model_id (schema ast_roles_family_ac_v2). The v1
    # values belonged to schema ast_roles_family_ac_v1: 7b7d022f01abaebdf3a6 (the first two
    # rows), 520faffe9b05a6675e23 and 8e777af28ad3351c9e9c. The before-anchors are unchanged.
    ("b0+b1*square(z)", "7b7d022f01abaebdf3a6", "881cd38b487c43232fa0"),
    ("b0+square(z)*b1", "3c93fe71901bd426e3bf", "881cd38b487c43232fa0"),
    ("b0+b1*z+b2*square(z)", "520faffe9b05a6675e23", "b6599bdb2570cf5e1de0"),
    ("b2*square(z)+b1*z+b0", "56c169112a734c45496a", "5db2d6c3aa91362fab27"),
    ("b0-b1*z", "2cc5a1eb40b592cdf639", "2cc5a1eb40b592cdf639"),
    ("b0+b1/z", "efd0d7e61b4c94a77449", "efd0d7e61b4c94a77449"),
    ("b0+z/b1", "fc9ac9180a3d24ab9990", "fc9ac9180a3d24ab9990"),
)


def legacy_model_id(expression, family, parameters, coefficient_priors=()):
    """Predecessor oracle, copied from the predecessor commit; tests only, never a runner switch."""
    roles, aliases = dict(parameters), {}

    def canonical(node):
        """Return the canonical form of an AST node (predecessor oracle)."""
        if isinstance(node, ast.Expression):
            return canonical(node.body)
        if isinstance(node, ast.Name):
            if node.id in roles:
                aliases.setdefault(node.id, len(aliases))
                return ["parameter", aliases[node.id], roles[node.id]]
            return ["input", node.id]
        if isinstance(node, ast.Subscript):
            return ["index", canonical(node.value), ["input", node.slice.id]]
        if isinstance(node, ast.Constant):
            return ["number", float(node.value)]
        if isinstance(node, ast.BinOp):
            return [type(node.op).__name__, canonical(node.left), canonical(node.right)]
        if isinstance(node, ast.UnaryOp):
            return [type(node.op).__name__, canonical(node.operand)]
        return [node.func.id, *[canonical(arg) for arg in node.args]]

    identity = [family, canonical(ast.parse(expression, mode="eval"))]
    if coefficient_priors:
        identity.append([[name, list(rule)] for name, rule in coefficient_priors])
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()[:20]


@pytest.mark.parametrize("expression,before,after", BEFORE_AFTER)
def test_declared_before_and_after_anchors(expression, before, after):
    """Declared before- and after-anchors for model ids hold."""
    assert legacy_model_id(expression, "normal", TRIPLE) == before
    assert canonical_model_id(expression, "normal", TRIPLE) == after


@pytest.mark.parametrize("left,right", [
    ("b0+b1*z", "b1*z+b0"), ("b0+b1*z", "b0+z*b1"),
    ("b0*b1", "b1*b0"), ("tanh(b0+b1*z)", "tanh(b1*z+b0)"),
])
def test_local_add_mult_and_call_arguments_commute(left, right):
    """Addition, multiplication and call arguments commute locally in the model id."""
    assert canonical_model_id(left, "binomial", PAIR) == canonical_model_id(right, "binomial", PAIR)


def test_roles_family_input_and_index_remain_bound():
    """Roles, family, input and index stay bound in the model id."""
    current = canonical_model_id("b0+b1*z", "normal", PAIR)
    assert current != canonical_model_id("b0+b1*z", "normal", (("b0", "signed"), ("b1", "intercept")))
    assert current != canonical_model_id("b0+b1*z", "binomial", PAIR)
    assert current != canonical_model_id("b0+b1*u", "normal", PAIR)
    assert canonical_model_id("b0+b1[student]", "normal", PAIR) != canonical_model_id(
        "b0+b1[question]", "normal", PAIR)
    assert canonical_model_id("b0+b1[student]", "normal", PAIR) != canonical_model_id(
        "b0+b1*student", "normal", PAIR)


def test_proposed_priors_follow_the_coefficient_through_sort_and_renaming():
    """Proposed priors follow their coefficient through sorting and renaming."""
    prior = (("b1", ("Normal", 0.0, 0.4)),)
    current = canonical_model_id("b0+b1*z", "normal", PAIR, prior)
    assert current == canonical_model_id("b1*z+b0", "normal", PAIR, prior)
    assert current == canonical_model_id("s*z+i", "normal", (("i", "intercept"), ("s", "signed")),
                                        (("s", ("Normal", 0.0, 0.4)),))
    assert current != canonical_model_id("b0+b1*z", "normal", PAIR)
    assert current != canonical_model_id("b0+b1*z", "normal", PAIR, (("b1", ("Normal", 0.0, 0.8)),))


def test_tied_role_keys_preserve_shared_coefficient_structure():
    """Tied role keys preserve the shared-coefficient structure."""
    roles = tuple((n, "signed") for n in ("a", "b", "c"))
    expected = canonical_model_id("a*b+b*c", "normal", roles)
    for names in permutations(("a", "b", "c")):
        a, b, c = names
        assert canonical_model_id(f"{b}*{c}+{a}*{b}", "normal", roles) == expected
    assert expected != canonical_model_id("a*b+c*c", "normal", roles)


def test_associativity_scope_guard_keeps_binary_nesting():
    """The associativity scope guard keeps binary nesting."""
    assert canonical_model_id("b0+b1*z+b2*square(z)", "normal", TRIPLE) != canonical_model_id(
        "b2*square(z)+b1*z+b0", "normal", TRIPLE)
    # Distinct roles isolate nesting from the pre-existing alpha equivalence.
    roles = (("a", "intercept"), ("b", "signed"), ("c", "positive"))
    assert canonical_model_id("a+b+c", "normal", roles) != canonical_model_id("a+(b+c)", "normal", roles)


@pytest.mark.parametrize("operator", ["-", "/", "**"])
def test_noncommutative_operators_keep_operand_order(operator):
    """Non-commutative operators keep their operand order."""
    assert canonical_model_id(f"b0{operator}b1", "normal", PAIR) != canonical_model_id(
        f"b1{operator}b0", "normal", PAIR)
    assert canonical_model_id(f"u{operator}z", "normal", ()) != canonical_model_id(
        f"z{operator}u", "normal", ())


@pytest.mark.parametrize("function", ["minimum", "maximum"])
def test_v3_function_order_is_untouched(function):
    """Function-argument order of the v3 schema is untouched."""
    assert canonical_model_id(f"{function}(u,z)", "normal", ()) != canonical_model_id(
        f"{function}(z,u)", "normal", ())


@pytest.mark.parametrize("record", [{}, {"model_id_schema": None}, {"model_id_schema": "old"}, []])
def test_legacy_schema_is_refused_including_absent_field(record, tmp_path):
    """The legacy schema, including an absent schema field, is refused."""
    output = tmp_path / "cell"
    reservation = tmp_path / "reservation.json"
    with pytest.raises(ValueError, match="model_id_schema mismatch"):
        require_model_id_schema(record)
    assert not output.exists() and not reservation.exists()


def test_matching_schema_is_accepted_without_modifying_record():
    """A matching schema is accepted and the record is not modified."""
    record = {"model_id_schema": MODEL_ID_SCHEMA, "events": []}
    before = json.dumps(record)
    assert require_model_id_schema(record) == MODEL_ID_SCHEMA
    assert json.dumps(record) == before


def synthetic_pool(identity, schema):
    """Build a synthetic pool record whose model ids come from the given identity function and schema label."""
    forms = ("b0+b1*z", "b1*z+b0", "b0+b1*square(z)", "b0+b1*tanh(z)")
    events = [{"expression": form, "family": "normal", "parameters": dict(PAIR),
               "model_id": identity(form, "normal", PAIR)} for form in forms]
    return {"model_id_schema": schema, "seed": 7, "d0_designs": [0.1, 0.35, 0.65, 0.9],
            "prompt_sha256": hashlib.sha256(b"synthetic identity-change control").hexdigest(),
            "requested": 4, "initial_catalogue": "off", "events": events,
            "raw_response": [{k: e[k] for k in ("expression", "family", "parameters")} for e in events],
            "after_ids": list(dict.fromkeys(e["model_id"] for e in events))}


def test_synthetic_pool_control_four_becomes_three_with_same_declared_inputs(tmp_path):
    """A synthetic pool of four forms collapses to three under the new identity with the same declared inputs."""
    before = synthetic_pool(legacy_model_id, LEGACY_MODEL_ID_SCHEMA)
    after = synthetic_pool(canonical_model_id, MODEL_ID_SCHEMA)
    for name, pool in (("before", before), ("after", after)):
        path = tmp_path / f"{name}_pool.json"
        path.write_text(json.dumps(pool))
        assert json.loads(path.read_text()) == pool
    assert len(before["after_ids"]) == 4 and len(after["after_ids"]) == 3
    for field in ("seed", "d0_designs", "prompt_sha256", "requested", "initial_catalogue", "raw_response"):
        assert before[field] == after[field]
    assert [{k: v for k, v in e.items() if k != "model_id"} for e in before["events"]] == [
        {k: v for k, v in e.items() if k != "model_id"} for e in after["events"]]


def test_native_core_mock_records_schema_and_duplicate_collapse(tmp_path, monkeypatch):
    """A mock native core run records the schema and the duplicate collapse."""
    from .test_core_parallel_wave import cell_args, injected_replies
    from .test_count_proposer import stub_fits

    stub_fits(monkeypatch)
    forms = ["b0+b1*z", "b1*z+b0", "b0+b1*square(z)", "b0+b1*tanh(z)", "b0+b1*z"]
    state = injected_replies(monkeypatch, forms)
    out = tmp_path / "cell"
    core.run(cell_args(out, proposal_rounds="2", initial_proposals=4,
                       initial_catalogue="off", burst_concurrency=1, run_label="development"))
    pool = json.loads((out / "proposals/t00_burst.json").read_text())
    config = json.loads((out / "config.json").read_text())
    assert state["calls"] == 5 and len(pool["after_ids"]) == 3
    assert pool["duplicate_count"] == 2
    assert require_model_id_schema(pool) == MODEL_ID_SCHEMA
    assert config["model_id_schema"] == MODEL_ID_SCHEMA
    assert config["label_receipt"]["binding"]["settings"]["model_id_schema"] == MODEL_ID_SCHEMA
    assert "model_identity.py" in config["source_files_sha256"]


def test_schema_labels_only_the_runners_using_this_identity():
    """The schema label is written only by the runners that use this identity."""
    from .run_label_policy import bmd_runtime_settings
    from .test_run_label_policy import runner_args

    for bundle in ("core", "boxinggym", "chembench"):
        settings = bmd_runtime_settings(bundle, runner_args())
        assert settings["model_id_schema"] == MODEL_ID_SCHEMA


def test_initial_core_anchors_still_describe_four_distinct_models():
    """The initial core anchors still describe four distinct models."""
    for family in ("binomial", "beta_binomial", "zi_binomial", "normal"):
        assert len({h.model_id for h in initial_hypotheses(family)}) == 4
