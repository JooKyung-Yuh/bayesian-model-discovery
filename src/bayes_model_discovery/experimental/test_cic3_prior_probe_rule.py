"""Tests for the ChemBench packet's `prior_probe_rule` key (a project decision).

The key states the prior-probe admission contract as engine mechanics only: it is bound to
the probe's own constants and to the packet's declared priors and inputs, it is identical
for every arm (the helper takes no arm, prompt, seed or pool input), and its sentence
carries no strategy, preference, example, input name or count from any run.
"""

import inspect
import json
import re

import pytest

from .belief_packet import (
    CONSTRAINT_KEYS,
    OPTIONAL_CONSTRAINT_KEYS,
    PRIOR_PROBE_RULE_CONSTRAINT_KEYS,
    cell_constraints,
)
from .chembench_core import (
    CHEM_ALLOWED_FAMILIES,
    CONSTRAINTS_DISCLOSURE,
    PRIOR_PROBE_GRID_POINTS,
    PRIOR_PROBE_RULE,
    PRIOR_PROBE_SAMPLES,
    _disclosed_constraints,
    _prior_probe_rule,
)


def _interface(names=("xa", "xb", "T"), link="log"):
    inputs = [{"name": name, "range": [0.5 * i + 0.1, 10.0 * (i + 1)]} for i, name in enumerate(names)]
    derived = {f"{name}z": f"({name}-1.0)/2.0" for name in names}
    return {"inputs": inputs, "derived_inputs": derived, "link": link,
            "outcome_type": "positive", "support": [0.0, None], "trials": None}


def _constraints(interface, initial_structures=()):
    names = tuple(item["name"] for item in interface["inputs"]) + tuple(interface["derived_inputs"])
    return cell_constraints("101", "lognormal", allowed_families=CHEM_ALLOWED_FAMILIES,
                            initial_structures=list(initial_structures), input_names=names), names


def test_the_disclosure_constant_names_cic3():
    """Check that the disclosure constant names cic3."""
    assert CONSTRAINTS_DISCLOSURE == "cic3_v1"


def test_the_key_is_bound_to_the_probe_constants_and_the_packet_values():
    """Check that the key is bound to the probe constants and the packet values."""
    interface = _interface()
    constraints, names = _constraints(interface)
    disclosed = _disclosed_constraints(constraints, interface, names)
    probe = disclosed["prior_probe_rule"]
    assert probe["rule"] == PRIOR_PROBE_RULE
    assert probe["prior_draws"] == PRIOR_PROBE_SAMPLES == 64
    assert probe["checked_points"] == {
        "diagonal_grid_points": PRIOR_PROBE_GRID_POINTS,
        "range_corners": 2 ** len(interface["inputs"]),
        "disclosed_observation_inputs": True,
    }
    assert PRIOR_PROBE_GRID_POINTS == 16
    assert probe["coefficient_priors"] == constraints["roles"]
    assert probe["link"] == "log" and probe["mean"] == "response_unit * exp(expression)"
    # every other key is the previous packet version's own
    assert set(disclosed) == set(constraints) | {"coefficient_name_rule", "prior_probe_rule"}
    for key in constraints:
        assert disclosed[key] == constraints[key]


def test_the_sentence_states_the_numbers_the_probe_uses_and_nothing_else():
    """Check that the sentence states the numbers the probe uses and nothing else."""
    numbers = set(re.findall(r"\d+", PRIOR_PROBE_RULE))
    assert numbers == {str(PRIOR_PROBE_SAMPLES), str(PRIOR_PROBE_GRID_POINTS)}
    # mechanics, not strategy: no preference, recommendation, example or scale advice
    lowered = PRIOR_PROBE_RULE.lower()
    for word in ("standardi", "prefer", "recommend", "should", "instead", "example", "such as",
                 "raw ", "overflow", "canary", "arm"):
        assert word not in lowered, word
    # no formula fragments in the sentence (the formula sits in the structured `mean` field)
    assert not any(ch in PRIOR_PROBE_RULE for ch in "=*+/")
    # the probe checks declared points only; the sentence says so and claims no more
    assert "only" in PRIOR_PROBE_RULE and "finite and strictly positive" in PRIOR_PROBE_RULE


@pytest.mark.parametrize("names", [("C_A", "C_I", "C_B", "C_P", "Enz", "T", "pH"), ("x", "y")])
def test_the_sentence_names_no_interface_input(names):
    """Check that the sentence names no interface input."""
    interface = _interface(names)
    constraints, grammar = _constraints(interface)
    probe = _disclosed_constraints(constraints, interface, grammar)["prior_probe_rule"]
    for name in grammar:
        assert re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", probe["rule"]) is None, name


def test_the_helper_reads_no_arm_prompt_seed_or_pool():
    """Check that the helper reads no arm prompt seed or pool."""
    assert tuple(inspect.signature(_disclosed_constraints).parameters) == (
        "constraints", "interface", "grammar_names")
    assert tuple(inspect.signature(_prior_probe_rule).parameters) == ("constraints", "interface")
    # the code objects (not the docstrings) reference no run, arm, prompt, seed or pool name
    for code in (_disclosed_constraints.__code__, _prior_probe_rule.__code__):
        referenced = " ".join(code.co_names + code.co_varnames).lower()
        for token in ("args", "prompt", "seed", "pool", "arm", "cell", "round", "wave"):
            assert re.search(rf"(?<![a-z_]){token}(?![a-z_])", referenced) is None, token
    # same bytes whatever the round's structures or the grammar-name order
    interface = _interface()
    c0, names = _constraints(interface)
    c1, _ = _constraints(interface, initial_structures=("b0",))
    a = json.dumps(_disclosed_constraints(c0, interface, names)["prior_probe_rule"], sort_keys=True)
    b = json.dumps(_disclosed_constraints(c1, interface, names)["prior_probe_rule"], sort_keys=True)
    c = json.dumps(_disclosed_constraints(c0, interface, tuple(reversed(names)))["prior_probe_rule"],
                   sort_keys=True)
    assert a == b == c


def test_a_non_log_link_is_refused_not_described():
    """Check that a non log link is refused not described."""
    interface = _interface(link="identity")
    constraints, names = _constraints(interface)
    with pytest.raises(ValueError, match="log link"):
        _disclosed_constraints(constraints, interface, names)


def test_the_key_lives_outside_the_optional_set_so_no_other_bundle_grows_it():
    """Check that the key lives outside the optional set so no other bundle grows it."""
    assert PRIOR_PROBE_RULE_CONSTRAINT_KEYS == {"prior_probe_rule"}
    assert not PRIOR_PROBE_RULE_CONSTRAINT_KEYS & (CONSTRAINT_KEYS | OPTIONAL_CONSTRAINT_KEYS)
