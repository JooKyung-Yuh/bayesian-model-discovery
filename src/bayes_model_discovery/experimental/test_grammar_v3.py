"""The v3 expression grammar, and v2's unchanged default.

A first step established that positional admission needs no change - a
coefficient in a denominator, inside a call or as an exponent already passes -
so v3 adds exactly two things and nothing else:

- a DOMAIN PROOF over the interface's declared input ranges and the coefficient
  roles, refusing a division whose denominator it cannot prove nonzero, a
  logarithm whose argument it cannot prove positive, a root or fractional power
  whose base it cannot prove nonnegative (`domain_unprovable`), and a
  subexpression that reduces to a coefficient-only value
  (`constant_degenerate_subexpression`);
- six general element-wise calls: abs, sigmoid, expm1, log1p, minimum, maximum.

Everything here is synthetic: a synthetic positive-outcome interface and
synthetic expressions. No benchmark law, no benchmark parameter, no run text.
"""

import ast

import numpy as np
import pytest

from .belief_packet import task_interface
from .count_models import (
    CALL_ARITY,
    CONSTANT_DEGENERATE_RULE,
    DEFAULT_GRAMMAR_VERSION,
    DOMAIN_UNPROVABLE_RULE,
    GRAMMAR_VERSIONS,
    SIGN_ANY,
    SIGN_NONNEGATIVE,
    SIGN_POSITIVE,
    V3_ADDED_CALL_ARITY,
    active_grammar_version,
    active_input_signs,
    evaluate,
    interface_input_names,
    interface_input_signs,
    restore_grammar_state,
    set_grammar_version,
    validate_expression,
)

# A synthetic interface: one strictly positive input, one input whose range
# straddles zero, one nonnegative input, and a z-scored derived twin of each.
POSITIVE_INPUT, SIGNED_INPUT, NONNEGATIVE_INPUT = "x0", "x1", "x2"
BOUNDS = {POSITIVE_INPUT: (0.5, 20.0), SIGNED_INPUT: (-4.0, 4.0), NONNEGATIVE_INPUT: (0.0, 9.0)}


@pytest.fixture
def interface():
    """Fixture: a synthetic positive-outcome interface with base inputs and z-scored derived twins."""
    return task_interface(
        "positive",
        [{"name": name, "range": list(bound)} for name, bound in BOUNDS.items()],
        {f"{name}z": f"({name}-1.5)/2.5" for name in BOUNDS},
        (0.0, None),
    )


@pytest.fixture
def names(interface):
    """Fixture: the input names declared by the interface."""
    return interface_input_names(interface)


@pytest.fixture
def signs(interface):
    """Fixture: the sign table derived from the interface's declared input ranges."""
    return interface_input_signs(interface)


def accept(expression, parameters, names, signs, version="v3"):
    """Validate an expression under the given grammar version; raises if it is refused."""
    validate_expression(
        expression, tuple(parameters.items()), names, (),
        grammar_version=version, input_signs=signs,
    )


# ---------------------------------------------------------------------------
# The switch itself, and v2 as the untouched default
# ---------------------------------------------------------------------------

def test_the_default_grammar_is_v2_and_the_switch_refuses_an_unregistered_name():
    """Test: the default grammar is v2 and the switch refuses an unregistered name."""
    assert DEFAULT_GRAMMAR_VERSION == "v2"
    assert GRAMMAR_VERSIONS == ("v2", "v3")
    assert active_grammar_version() == "v2"
    assert active_input_signs() == {}
    with pytest.raises(ValueError, match="unregistered grammar version"):
        set_grammar_version("v4")
    assert active_grammar_version() == "v2"


def test_the_switch_registers_the_interface_and_restores(interface, signs):
    """Test: the switch registers the interface and restores."""
    previous = set_grammar_version("v3", interface)
    try:
        assert previous == ("v2", {})
        assert active_grammar_version() == "v3"
        assert active_input_signs() == signs
    finally:
        version, registered = previous
        set_grammar_version(version)
        assert active_grammar_version() == "v2"
        assert active_input_signs() == registered


@pytest.mark.parametrize("expression,parameters", (
    ("a/(b+x1)", {"a": "positive", "b": "positive"}),
    ("log(a+x1)", {"a": "positive"}),
    ("a*sqrt(x1)", {"a": "signed"}),
    ("a/b*x0", {"a": "positive", "b": "positive"}),
))
def test_v2_admits_what_v3_refuses(expression, parameters, names, signs):
    """The v2 gate is unchanged: none of these domain rules exist under it."""
    validate_expression(expression, tuple(parameters.items()), names, (), grammar_version="v2")
    with pytest.raises(ValueError):
        accept(expression, parameters, names, signs)


def test_v2_keeps_its_own_reserved_names_and_call_table(names):
    """A coefficient named after a v3 call is legal under v2 and refused under v3."""
    validate_expression("abs*x0", (("abs", "signed"),), names, (), grammar_version="v2")
    with pytest.raises(ValueError, match="invalid or reserved coefficient name"):
        validate_expression("abs*x0", (("abs", "signed"),), names, (), grammar_version="v3")
    with pytest.raises(ValueError, match="call outside expression grammar"):
        validate_expression("a*abs(x1)", (("a", "signed"),), names, (), grammar_version="v2")


def test_an_unregistered_version_argument_is_refused(names):
    """Test: an unregistered version argument is refused."""
    with pytest.raises(ValueError, match="unregistered grammar version"):
        validate_expression("a*x0", (("a", "signed"),), names, (), grammar_version="v2.5")


# ---------------------------------------------------------------------------
# The domain proof: provable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expression,parameters", (
    # A denominator proved nonzero by a strictly positive input.
    ("a/x0", {"a": "signed"}),
    # A denominator proved nonzero by a positive coefficient plus a nonnegative input.
    ("a*x1/(b+x2)", {"a": "signed", "b": "positive"}),
    # A saturating rational: the coefficient stands in the denominator.
    ("a*x0/(b+x0)", {"a": "positive", "b": "positive"}),
    # A logarithm whose argument is a product of strictly positive factors.
    ("log(a*x0/(b+x0))", {"a": "positive", "b": "positive"}),
    # A root of a nonnegative input, and of a square that cannot be negative.
    ("a*sqrt(x2)+b*sqrt(square(x1))", {"a": "signed", "b": "signed"}),
    # A coefficient as the exponent of a strictly positive base.
    ("power(x0, a)", {"a": "signed"}),
    # A fractional literal exponent on a strictly positive base.
    ("a*x0**0.5", {"a": "signed"}),
    # A negative integer exponent on a base proved nonzero.
    ("a*x0**-2", {"a": "signed"}),
    # An even integer exponent needs no proof at all.
    ("a*x1**2", {"a": "signed"}),
    # exp and sigmoid are strictly positive, so a log of them is provable.
    ("log(exp(a*x1)+sigmoid(b*x1))", {"a": "signed", "b": "signed"}),
    # log1p of a nonnegative argument.
    ("a*log1p(x2)", {"a": "signed"}),
    # minimum/maximum carry the union of their arguments' signs.
    ("a/maximum(x0, b)", {"a": "signed", "b": "positive"}),
))
def test_v3_admits_a_provable_domain(expression, parameters, names, signs):
    """Test: v3 admits a provable domain."""
    accept(expression, parameters, names, signs)


# ---------------------------------------------------------------------------
# The domain proof: domain_unprovable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expression,parameters,what", (
    ("a/x1", {"a": "signed"}, "a division"),
    ("a/(b+x1)", {"a": "positive", "b": "positive"}, "a division"),
    ("a/x2", {"a": "signed"}, "a division"),
    ("log(a*x1)", {"a": "positive"}, "a logarithm"),
    ("log(x2+a*x2)", {"a": "positive"}, "a logarithm"),
    ("a*sqrt(x1)", {"a": "signed"}, "a square root"),
    ("a*x1**0.5", {"a": "signed"}, "a power"),
    ("power(x1, a)", {"a": "signed"}, "a power call"),
    ("a*x1**-1", {"a": "signed"}, "a power"),
    ("a*log1p(x1)", {"a": "signed"}, "a log1p"),
    # A signed coefficient never proves a denominator nonzero, even alone.
    ("x0/a", {"a": "signed"}, "a division"),
))
def test_v3_refuses_an_unprovable_domain(expression, parameters, what, names, signs):
    """Test: v3 refuses an unprovable domain."""
    with pytest.raises(ValueError) as raised:
        accept(expression, parameters, names, signs)
    message = str(raised.value)
    assert message.startswith(f"{DOMAIN_UNPROVABLE_RULE}: {what} requires ")
    # A general sentence: it names the node class and the role that would make
    # the node provable, and quotes no expression.
    assert "role positive or shape" in message
    assert expression not in message


def test_an_unregistered_namespace_makes_the_proof_harder_never_easier(names):
    """With no registered ranges every input is unsigned, so nothing is provable."""
    with pytest.raises(ValueError, match=DOMAIN_UNPROVABLE_RULE):
        validate_expression(
            "a/x0", (("a", "signed"),), names, (), grammar_version="v3", input_signs={},
        )


# ---------------------------------------------------------------------------
# The domain proof: constant_degenerate_subexpression
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expression,parameters", (
    ("a/b*x0", {"a": "positive", "b": "positive"}),
    ("(a+b)*x0", {"a": "signed", "b": "signed"}),
    ("a*b*x0", {"a": "positive", "b": "positive"}),
    ("exp(a)*x0", {"a": "signed"}),
    ("x0*(a+1.5)", {"a": "signed"}),
    ("x0*power(a, 2)", {"a": "positive"}),
))
def test_v3_refuses_a_constant_degenerate_subexpression(expression, parameters, names, signs):
    """Test: v3 refuses a constant degenerate subexpression."""
    with pytest.raises(ValueError) as raised:
        accept(expression, parameters, names, signs)
    assert str(raised.value).startswith(f"{CONSTANT_DEGENERATE_RULE}: ")


@pytest.mark.parametrize("expression,parameters", (
    ("b0", {"b0": "intercept"}),
    ("-a*x0", {"a": "positive"}),
    ("a*x0+b*x1", {"a": "signed", "b": "signed"}),
))
def test_v3_keeps_a_bare_coefficient_and_a_sign_flip(expression, parameters, names, signs):
    """Test: v3 keeps a bare coefficient and a sign flip."""
    accept(expression, parameters, names, signs)


# ---------------------------------------------------------------------------
# The six added calls
# ---------------------------------------------------------------------------

def test_the_six_added_calls_and_their_arities():
    """Test: the six added calls and their arities."""
    assert set(V3_ADDED_CALL_ARITY) == {
        "abs", "sigmoid", "expm1", "log1p", "minimum", "maximum"
    }
    assert set(V3_ADDED_CALL_ARITY) & set(CALL_ARITY) == set()
    assert [V3_ADDED_CALL_ARITY[name] for name in
            ("abs", "sigmoid", "expm1", "log1p", "minimum", "maximum")] == [1, 1, 1, 1, 2, 2]


# `log1p` is the one added call with a domain of its own, so the generic
# vocabulary tests feed it the nonnegative input and everything else the
# straddling one. Its domain rule has its own tests above.
CALL_ARGUMENT = {"log1p": NONNEGATIVE_INPUT}


@pytest.mark.parametrize("call,arity", sorted(V3_ADDED_CALL_ARITY.items()))
def test_each_added_call_passes_v3_and_is_refused_by_v2(call, arity, names, signs):
    """Test: each added call passes v3 and is refused by v2."""
    arguments = ", ".join([CALL_ARGUMENT.get(call, SIGNED_INPUT)] * arity)
    expression = f"a*{call}({arguments})"
    accept(expression, {"a": "signed"}, names, signs)
    with pytest.raises(ValueError, match="call outside expression grammar"):
        validate_expression(expression, (("a", "signed"),), names, (), grammar_version="v2")


@pytest.mark.parametrize("call,arity", sorted(V3_ADDED_CALL_ARITY.items()))
def test_each_added_call_is_refused_at_the_wrong_arity(call, arity, names, signs):
    """Test: each added call is refused at the wrong arity."""
    arguments = ", ".join([CALL_ARGUMENT.get(call, SIGNED_INPUT)] * (arity + 1))
    with pytest.raises(ValueError, match="wrong expression call arity"):
        accept(f"a*{call}({arguments})", {"a": "signed"}, names, signs)


def test_every_added_call_evaluates_on_the_numpy_backend():
    """Test: every added call evaluates on the numpy backend."""
    u = np.linspace(-3.0, 3.0, 9)
    inputs = {"x1": u}
    expected = {
        "abs": np.abs(u),
        "sigmoid": 1.0 / (1.0 + np.exp(-u)),
        "expm1": np.expm1(u),
        "log1p": np.log1p(np.abs(u)),
        "minimum": np.minimum(u, u),
        "maximum": np.maximum(u, u),
    }
    for call in sorted(V3_ADDED_CALL_ARITY):
        argument = "abs(x1)" if call == "log1p" else "x1"
        arguments = ", ".join([argument] * V3_ADDED_CALL_ARITY[call])
        value = evaluate(f"{call}({arguments})", {}, u, np, inputs)
        assert np.allclose(value, expected[call])
        assert np.all(np.isfinite(value))


def test_sigmoid_saturates_rather_than_returning_a_non_finite_value():
    """Test: sigmoid saturates rather than returning a non finite value."""
    u = np.array([-800.0, 0.0, 800.0])
    value = evaluate("sigmoid(x1)", {}, u, np, {"x1": u})
    assert np.all(np.isfinite(value))
    assert np.allclose(value, [0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# The sign table the proof is built on
# ---------------------------------------------------------------------------

def test_interface_input_signs_reads_the_declared_ranges(interface, signs):
    """Test: interface input signs reads the declared ranges."""
    assert signs[POSITIVE_INPUT] == SIGN_POSITIVE
    assert signs[SIGNED_INPUT] == SIGN_ANY
    assert signs[NONNEGATIVE_INPUT] == SIGN_NONNEGATIVE
    # A z-scored derived input is an affine shift and straddles zero.
    for name in BOUNDS:
        assert signs[f"{name}z"] == SIGN_ANY
    assert set(signs) == set(interface_input_names(interface))


def test_a_derived_input_the_analysis_cannot_clear_becomes_unsigned():
    """A derived block is the interface's own declaration, never a refusal."""
    interface = task_interface(
        "positive",
        [{"name": "x0", "range": [-1.0, 1.0]}],
        {"x0r": "1/x0"},
        (0.0, None),
    )
    assert interface_input_signs(interface)["x0r"] == SIGN_ANY


def test_the_proof_visits_every_operator_node(names, signs):
    """An unprovable node deep inside an otherwise provable expression is caught."""
    expression = "a*x0+b*exp(x0/x1)"
    ast.parse(expression, mode="eval")
    with pytest.raises(ValueError, match=DOMAIN_UNPROVABLE_RULE):
        accept(expression, {"a": "signed", "b": "signed"}, names, signs)


def test_restoring_the_state_puts_back_the_signs_a_version_name_alone_would_drop(interface, signs):
    """`set_grammar_version`'s return value can be put back exactly.

    The switch derives the signs from an interface, so feeding it the version
    name alone clears them - which is what let a runner-driven test leak CORE's
    interface into every test that followed it.
    """
    registered = set_grammar_version("v3", interface)
    try:
        assert active_input_signs() == signs
        # The version name alone is NOT a restore: it drops the signs.
        set_grammar_version("v3")
        assert active_input_signs() == {}
        restore_grammar_state(("v3", signs))
        assert active_input_signs() == signs
        assert active_grammar_version() == "v3"
    finally:
        restore_grammar_state(registered)
    assert (active_grammar_version(), active_input_signs()) == registered
