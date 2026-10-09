"""The two prior ablation switches: the role prior template and LLM-proposed priors.

Both switches OFF is the frozen default behaviour, and that is what most of this
file pins: the default template IS the paper's prior table, a `priors` key in a reply
is still an ignored key, the packet publishes no menu, no record gains a field
and the SYSTEM prompt is still the byte-frozen v2.0 file. The rest pins the two
switches ON: the exact hyperparameter tables, the menu/window/role validator
with a per-item reason, the identity change a proposed prior causes, and the fact
that the proposed prior actually reaches the PyMC graph and the prior probe
rather than only the record.

No credentials, no network: every transport test injects a fake client, exactly
as `test_count_proposer.py` does.
"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from . import count_proposer
from .belief_packet import cell_constraints, core_interface
from .core_count import points_variant, prior_particles, prior_probe
from .count_inference import build_model
from .count_models import (
    DEFAULT_PRIOR_TEMPLATE,
    LLM_PRIORS_VARIANT_TOKEN,
    PRIOR_CONCENTRATION_RANGE,
    PRIOR_LOCATION_ABS_MAX,
    PRIOR_MENU,
    PRIOR_MENU_BY_ROLE,
    PRIOR_RULES,
    PRIOR_SCALE_RANGE,
    PRIOR_TEMPLATE_NAMES,
    PRIOR_TEMPLATES,
    PRIOR_VARIANT_TOKEN,
    Hypothesis,
    active_prior_template,
    coefficient_prior,
    dispersion_prior,
    llm_priors_enabled,
    prior_menu_constraints,
    prior_template_table,
    set_llm_priors,
    set_prior_template,
    validate_proposed_prior,
    validate_proposed_priors,
)
from .count_proposer import (
    AnthropicCountProposer,
    OpenAICountProposer,
    parse_candidate_object,
)

# The prior-template ablation, quoted from the decision document
# that lists the ablations (translated):
#   "replace the whole role prior template by x2 (wide: intercept N(0,4),
#    signed N(0,2), positive LogNormal(0,2), shape Gamma(2,0.5), rho Beta(1,4),
#    pi Beta(1,2)) and by 0.5x (narrow: N(0,1), N(0,0.5), LogNormal(0,0.5),
#    Gamma(2,2), Beta(1,19), Beta(1,9))"
# written out here as the independent copy the code is checked against.
D8_TABLES = {
    "default": {
        "intercept": ("Normal", 0.0, 2.0),
        "signed": ("Normal", 0.0, 1.0),
        "positive": ("LogNormal", 0.0, 1.0),
        "shape": ("Gamma", 2.0, 1.0),
        "rho": ("Beta", 1.0, 9.0),
        "pi": ("Beta", 1.0, 4.0),
    },
    "wide": {
        "intercept": ("Normal", 0.0, 4.0),
        "signed": ("Normal", 0.0, 2.0),
        "positive": ("LogNormal", 0.0, 2.0),
        "shape": ("Gamma", 2.0, 0.5),
        "rho": ("Beta", 1.0, 4.0),
        "pi": ("Beta", 1.0, 2.0),
    },
    "narrow": {
        "intercept": ("Normal", 0.0, 1.0),
        "signed": ("Normal", 0.0, 0.5),
        "positive": ("LogNormal", 0.0, 0.5),
        "shape": ("Gamma", 2.0, 2.0),
        "rho": ("Beta", 1.0, 19.0),
        "pi": ("Beta", 1.0, 9.0),
    },
}
CONTEXT = {
    "run_id": "test-only-run",
    "world_seed": 11,
    "cell": "111",
    "proposal_rep": 0,
    "round": 1,
}


@pytest.fixture(autouse=True)
def restore_switches():
    """Every test leaves the process on the production defaults."""
    template, priors = active_prior_template(), llm_priors_enabled()
    yield
    set_prior_template(template)
    set_llm_priors(priors)


def hypothesis(expression, family, parameters, coefficient_priors=()):
    """Build a `Hypothesis` identically on both chains.

    The BoxingGym chain's class carries two extra fields between `parameters`
    and `coefficient_priors` (`input_names`, `categorical_levels`)
    that the CORE chain has no counterpart for, so this file passes
    the new field BY KEYWORD and is byte-identical on both trees.
    """
    return Hypothesis(expression, family, parameters, coefficient_priors=coefficient_priors)


def reply(priors=None, expression="b0+b1*z", parameters=None):
    """Return a JSON candidate reply, optionally carrying a `priors` block."""
    payload = {
        "expression": expression,
        "family": "binomial",
        "parameters": parameters or {"b0": "intercept", "b1": "signed"},
    }
    if priors is not None:
        payload["priors"] = priors
    return json.dumps(payload)


def anthropic_client(text):
    """Return a fake Anthropic-style client and the list recording its calls."""
    calls = []

    def create(**body):
        """Record the request body and return a canned response object."""
        calls.append(body)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text.pop(0) if text else "{}")],
            usage=SimpleNamespace(
                model_dump=lambda **kwargs: {"input_tokens": 1, "output_tokens": 1}
            ),
            id="test-only-response",
            model="returned-test-model",
            stop_reason="end_turn",
        )

    return SimpleNamespace(messages=SimpleNamespace(create=create)), calls


def openai_client(text):
    """Return a fake OpenAI-style client and the list recording its calls."""
    calls = []

    def create(**body):
        """Record the request body and return a canned response object."""
        calls.append(body)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text.pop(0) if text else "{}"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
            id="test-only-response",
            model="returned-test-model",
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


# ---------------------------------------------------------------- the prior templates


def test_prior_templates_are_exactly_the_three_d8_tables():
    """Check that prior templates are exactly the three d8 tables."""
    assert PRIOR_TEMPLATE_NAMES == ("default", "wide", "narrow")
    assert PRIOR_TEMPLATES == D8_TABLES
    for name, table in D8_TABLES.items():
        assert tuple(PRIOR_TEMPLATES[name]) == tuple(table), name


def test_default_is_the_process_default_and_reproduces_table_3():
    """Check that default is the process default and reproduces table 3."""
    assert active_prior_template() == DEFAULT_PRIOR_TEMPLATE == "default"
    assert dict(PRIOR_RULES) == D8_TABLES["default"]
    # The four values the tree's own tests already hard-code for Table 3.
    assert PRIOR_RULES["intercept"] == ("Normal", 0.0, 2.0)
    assert PRIOR_RULES["signed"] == ("Normal", 0.0, 1.0)
    assert PRIOR_RULES["rho"] == ("Beta", 1.0, 9.0)
    assert PRIOR_RULES["pi"] == ("Beta", 1.0, 4.0)


def test_prior_rules_is_a_live_view_that_follows_the_selected_template():
    """Check that prior rules is a live view that follows the selected template."""
    for name in PRIOR_TEMPLATE_NAMES:
        previous = set_prior_template(name)
        assert active_prior_template() == name
        assert dict(PRIOR_RULES) == D8_TABLES[name]
        # every operation the tree performs on PRIOR_RULES, on the live view
        assert PRIOR_RULES["signed"] == D8_TABLES[name]["signed"]
        assert PRIOR_RULES.get("positive") == D8_TABLES[name]["positive"]
        assert PRIOR_RULES.get("nonexistent-role") is None
        assert "shape" in PRIOR_RULES and len(PRIOR_RULES) == 6
        assert dict(PRIOR_RULES.items()) == D8_TABLES[name]
        assert set_prior_template(previous) == name
    with pytest.raises(TypeError):
        PRIOR_RULES["signed"] = ("Normal", 0.0, 99.0)


def test_set_prior_template_refuses_an_unregistered_name_and_keeps_the_active_one():
    """Check that set prior template refuses an unregistered name and keeps the active one."""
    with pytest.raises(ValueError, match="unregistered prior template"):
        set_prior_template("wide_ish")
    assert active_prior_template() == DEFAULT_PRIOR_TEMPLATE
    assert dict(PRIOR_RULES) == D8_TABLES["default"]


def test_prior_template_table_is_json_ready_and_names_the_active_template():
    """Check that prior template table is json ready and names the active template."""
    assert prior_template_table() == {
        role: [rule[0], rule[1], rule[2]] for role, rule in D8_TABLES["default"].items()
    }
    assert prior_template_table("narrow")["rho"] == ["Beta", 1.0, 19.0]
    assert json.loads(json.dumps(prior_template_table("wide")))["shape"] == ["Gamma", 2.0, 0.5]
    set_prior_template("wide")
    assert prior_template_table() == prior_template_table("wide")


def test_dispersion_prior_follows_the_template_and_refuses_a_non_beta_row():
    """Check that dispersion prior follows the template and refuses a non beta row."""
    assert dispersion_prior("rho") == (1.0, 9.0) and dispersion_prior("pi") == (1.0, 4.0)
    set_prior_template("narrow")
    assert dispersion_prior("rho") == (1.0, 19.0) and dispersion_prior("pi") == (1.0, 9.0)
    set_prior_template("default")
    with pytest.raises(ValueError, match="must stay a Beta row"):
        dispersion_prior("shape")


def test_variant_tokens_are_registered_alphabetical_and_empty_by_default():
    """Check that variant tokens are registered alphabetical and empty by default."""
    assert PRIOR_VARIANT_TOKEN == {"default": "", "wide": "prior_wide", "narrow": "prior_narrow"}
    assert LLM_PRIORS_VARIANT_TOKEN == "prior_llm"
    assert points_variant("off") == ""
    assert points_variant("off", prior_template="wide") == "prior_wide"
    assert points_variant("off", prior_template="narrow") == "prior_narrow"
    assert points_variant("off", llm_priors=True) == "prior_llm"
    assert points_variant("t10", prior_template="narrow", llm_priors=True) == (
        "prior_llm+prior_narrow+rev10"
    )
    assert points_variant("t10") == "rev10"


def test_record_and_packet_follow_the_template_without_moving_the_identity():
    """Check that record and packet follow the template without moving the identity."""
    model = hypothesis("b0+b1*z", "beta_binomial", (("b0", "intercept"), ("b1", "signed")))
    default_id, default_record = model.model_id, model.as_dict()
    assert default_record["fixed_priors"] == {
        "b0": ("Normal", 0.0, 2.0),
        "b1": ("Normal", 0.0, 1.0),
        "rho": ("Beta", 1.0, 9.0),
    }
    assert cell_constraints("111", "binomial")["roles"]["signed"] == "Normal(0,1)"
    set_prior_template("wide")
    assert model.as_dict()["fixed_priors"] == {
        "b0": ("Normal", 0.0, 4.0),
        "b1": ("Normal", 0.0, 2.0),
        "rho": ("Beta", 1.0, 4.0),
    }
    assert cell_constraints("111", "binomial")["roles"]["signed"] == "Normal(0,2)"
    # The template is a RUN-level setting, not part of model identity: the template
    # arms are separate runs and the pairwise comparison needs the same
    # model_id in both, which is also what keeps the default byte-identical.
    assert model.model_id == default_id


# ------------------------------------------------------- the prior validator


def test_menu_windows_and_role_menu_are_the_pre_declared_d8_ranges():
    """Check that menu windows and role menu are the pre declared d8 ranges."""
    assert PRIOR_MENU == ("Normal", "LogNormal", "Gamma", "Beta")
    assert PRIOR_LOCATION_ABS_MAX == 5.0
    assert PRIOR_SCALE_RANGE == (0.1, 5.0)
    assert PRIOR_CONCENTRATION_RANGE == (0.5, 20.0)
    # role consistency is SUPPORT CONTAINMENT,
    # not support equality, so `positive`/`shape` (support (0,inf)) additionally
    # accept `Beta` (support (0,1), a subset of (0,inf)), and `intercept`/
    # `signed` (support the whole real line) accept all four menu distributions.
    assert PRIOR_MENU_BY_ROLE == {
        "intercept": ("Normal", "LogNormal", "Gamma", "Beta"),
        "signed": ("Normal", "LogNormal", "Gamma", "Beta"),
        "positive": ("LogNormal", "Gamma", "Beta"),
        "shape": ("LogNormal", "Gamma", "Beta"),
        "rho": ("Beta",),
        "pi": ("Beta",),
    }
    block = prior_menu_constraints()
    assert block["menu"] == list(PRIOR_MENU) and block["optional"] is True
    assert block["ranges"] == {
        "mu": [-5.0, 5.0],
        "sigma": [0.1, 5.0],
        "alpha": [0.5, 20.0],
        "beta": [0.5, 20.0],
    }
    assert "Normal" not in block["allowed_by_role"]["positive"]
    assert "Normal" not in block["allowed_by_role"]["shape"]
    assert "Beta" in block["allowed_by_role"]["positive"] and "Beta" in block["allowed_by_role"]["shape"]
    assert "Beta" in block["allowed_by_role"]["intercept"] and "Beta" in block["allowed_by_role"]["signed"]


@pytest.mark.parametrize(
    "role,spec,expected",
    [
        ("signed", {"dist": "Normal", "params": [0.0, 1.5]}, ("Normal", 0.0, 1.5)),
        ("intercept", {"dist": "Normal", "params": [-5, 5]}, ("Normal", -5.0, 5.0)),
        ("positive", {"dist": "LogNormal", "params": [1.0, 0.1]}, ("LogNormal", 1.0, 0.1)),
        ("positive", {"dist": "Gamma", "params": [0.5, 20]}, ("Gamma", 0.5, 20.0)),
        ("shape", {"dist": "Gamma", "params": [2, 2]}, ("Gamma", 2.0, 2.0)),
        # the `role_mismatch` scenario an earlier version
        # used to demonstrate a rejection - `signed` (real support)
        # now ACCEPTS a LogNormal, since (0,inf) is a subset of (-inf,inf).
        ("signed", {"dist": "LogNormal", "params": [0.0, 1.0]}, ("LogNormal", 0.0, 1.0)),
    ],
)
def test_a_prior_inside_the_menu_windows_and_role_support_is_accepted(role, spec, expected):
    """Check that a prior inside the menu windows and role support is accepted."""
    assert validate_proposed_prior(role, spec) == (expected, None)


@pytest.mark.parametrize(
    "role,spec,fragment",
    [
        ("signed", {"dist": "Cauchy", "params": [0, 1]}, "outside the menu"),
        ("signed", {"dist": "Normal", "params": [5.1, 1]}, "|mu| must be at most 5"),
        ("signed", {"dist": "Normal", "params": [0, 0.05]}, "sigma must be in [0.1, 5]"),
        ("signed", {"dist": "Normal", "params": [0, 5.5]}, "sigma must be in [0.1, 5]"),
        ("shape", {"dist": "Gamma", "params": [0.4, 2]}, "alpha must be in [0.5, 20]"),
        ("shape", {"dist": "Gamma", "params": [2, 20.5]}, "beta must be in [0.5, 20]"),
        ("positive", {"dist": "Normal", "params": [0, 1]}, "positive support"),
        # support CONTAINMENT, not equality, so
        # `signed`/`intercept` (real support) now accept all four distributions
        # - there is no coefficient-role rejection left for them. The only
        # remaining role-support rejections are Normal-on-positive/shape (above)
        # and a direct call naming `rho`/`pi` (unit support, Beta-only) - the
        # dispersion roles a proposal can never reach through the parsed-reply
        # path (`validate_proposed_priors` filters them out first).
        ("rho", {"dist": "Gamma", "params": [2.0, 2.0]}, "unit support"),
        ("pi", {"dist": "Normal", "params": [0.0, 1.0]}, "unit support"),
        ("signed", {"dist": "Normal", "params": [0]}, "exactly two parameters"),
        ("signed", {"dist": "Normal", "params": ["0", "1"]}, "finite real numbers"),
        ("signed", {"dist": "Normal", "params": [0, float("inf")]}, "finite real numbers"),
        ("signed", {"dist": "Normal", "params": [True, 1]}, "finite real numbers"),
        ("signed", {"dist": "Normal"}, 'prior must be {"dist"'),
        ("signed", ["Normal", 0, 1], 'prior must be {"dist"'),
    ],
)
def test_a_prior_outside_the_menu_windows_or_role_support_is_rejected(role, spec, fragment):
    """Check that a prior outside the menu windows or role support is rejected."""
    rule, reason = validate_proposed_prior(role, spec)
    assert rule is None and fragment in reason


def test_a_whole_block_reports_one_reason_per_offending_item_and_never_falls_back():
    """Check that a whole block reports one reason per offending item and never falls back."""
    roles = {"b0": "intercept", "b1": "signed", "c": "positive"}
    accepted, reason = validate_proposed_priors(
        {
            "b1": {"dist": "Normal", "params": [0.0, 1.5]},
            "b0": {"dist": "Normal", "params": [9.0, 1.0]},
            "c": {"dist": "Normal", "params": [0.0, 1.0]},
            "rho": {"dist": "Beta", "params": [1.0, 9.0]},
            "zz": {"dist": "Normal", "params": [0.0, 1.0]},
        },
        roles,
    )
    # Nothing is accepted and nothing silently keeps its role prior: the whole
    # candidate is rejected, with every offending item named once.
    assert accepted == ()
    assert reason.startswith("proposed prior rejected - ")
    assert "b0: |mu| must be at most 5" in reason
    assert "c: role 'positive' has positive support" in reason
    assert "rho: a family-owned dispersion parameter" in reason
    assert "zz: not a declared coefficient" in reason
    assert reason.count(";") == 3
    # ... and the same block without the four offenders is accepted, sorted.
    accepted, reason = validate_proposed_priors(
        {
            "b1": {"dist": "Normal", "params": [0.0, 1.5]},
            "b0": {"dist": "Normal", "params": [0.0, 3.0]},
        },
        roles,
    )
    assert reason is None
    assert accepted == (("b0", ("Normal", 0.0, 3.0)), ("b1", ("Normal", 0.0, 1.5)))


def test_hypothesis_refuses_a_non_canonical_duplicate_or_undeclared_proposed_prior():
    """Check that hypothesis refuses a non canonical duplicate or undeclared proposed prior."""
    parameters = (("b0", "intercept"), ("b1", "signed"))
    with pytest.raises(ValueError, match="sorted by coefficient name"):
        hypothesis(
            "b0+b1*z",
            "binomial",
            parameters,
            (("b1", ("Normal", 0.0, 1.0)), ("b0", ("Normal", 0.0, 1.0))),
        )
    with pytest.raises(ValueError, match="at most one proposed prior per coefficient"):
        hypothesis(
            "b0+b1*z",
            "binomial",
            parameters,
            (("b0", ("Normal", 0.0, 1.0)), ("b0", ("Normal", 0.0, 2.0))),
        )
    with pytest.raises(ValueError, match="undeclared coefficient"):
        hypothesis("b0+b1*z", "binomial", parameters, (("b2", ("Normal", 0.0, 1.0)),))
    with pytest.raises(ValueError, match="triple"):
        hypothesis("b0+b1*z", "binomial", parameters, (("b0", ("Normal", 0.0)),))
    with pytest.raises(ValueError, match="sigma must be in"):
        hypothesis("b0+b1*z", "binomial", parameters, (("b0", ("Normal", 0.0, 99.0)),))


def test_a_proposed_prior_changes_the_identity_and_the_record_but_not_the_default():
    """Check that a proposed prior changes the identity and the record but not the default."""
    parameters = (("b0", "intercept"), ("b1", "signed"))
    plain = hypothesis("b0+b1*z", "binomial", parameters)
    proposed = hypothesis("b0+b1*z", "binomial", parameters, (("b1", ("Normal", 0.0, 1.5)),))
    other = hypothesis("b0+b1*z", "binomial", parameters, (("b1", ("Normal", 0.0, 2.5)),))
    assert len({plain.model_id, proposed.model_id, other.model_id}) == 3
    assert plain.as_dict() == {
        "model_id": plain.model_id,
        "expression": "b0+b1*z",
        "family": "binomial",
        "parameters": {"b0": "intercept", "b1": "signed"},
        "noise_parameters": (),
        "fixed_priors": {"b0": ("Normal", 0.0, 2.0), "b1": ("Normal", 0.0, 1.0)},
    }
    record = proposed.as_dict()
    assert record["fixed_priors"] == {"b0": ("Normal", 0.0, 2.0), "b1": ("Normal", 0.0, 1.5)}
    assert record["prior_source"] == "llm"
    assert record["proposed_priors"] == {"b1": ["Normal", 0.0, 1.5]}
    assert coefficient_prior(proposed, "b1", "signed") == ("Normal", 0.0, 1.5)
    assert coefficient_prior(proposed, "b0", "intercept") == ("Normal", 0.0, 2.0)


def test_prior_source_is_recorded_for_every_model_only_while_the_switch_is_on():
    """Check that prior source is recorded for every model only while the switch is on."""
    plain = hypothesis("b0+b1*z", "binomial", (("b0", "intercept"), ("b1", "signed")))
    assert "prior_source" not in plain.as_dict()
    set_llm_priors(True)
    assert plain.as_dict()["prior_source"] == "role_template"
    set_llm_priors(False)
    assert "prior_source" not in plain.as_dict()


# ------------------------------------------------- where the prior is APPLIED


def test_build_model_uses_the_proposed_prior_and_the_template_dispersion_row():
    """Check that build model uses the proposed prior and the template dispersion row."""
    parameters = (("b0", "intercept"), ("b1", "signed"))
    u = np.linspace(0.05, 0.95, 8)
    y = np.full(8, 25)

    def priors_of(model):
        """Return the op name and parameters of the b0, b1 and rho priors in a built model."""
        return {
            name: (
                type(model[name].owner.op).__name__,
                [float(np.asarray(arg.eval())) for arg in model[name].owner.inputs[3:5]],
            )
            for name in ("b0", "b1", "rho")
        }

    plain = priors_of(build_model(hypothesis("b0+b1*z", "beta_binomial", parameters), u, y))
    assert plain["b0"] == ("NormalRV", [0.0, 2.0])
    assert plain["b1"] == ("NormalRV", [0.0, 1.0])
    assert plain["rho"] == ("BetaRV", [1.0, 9.0])
    proposed = priors_of(
        build_model(
            hypothesis(
                "b0+b1*z", "beta_binomial", parameters, (("b1", ("Normal", 0.0, 0.25)),)
            ),
            u,
            y,
        )
    )
    assert proposed["b1"] == ("NormalRV", [0.0, 0.25])
    assert proposed["b0"] == plain["b0"] and proposed["rho"] == plain["rho"]
    set_prior_template("narrow")
    narrow = priors_of(build_model(hypothesis("b0+b1*z", "beta_binomial", parameters), u, y))
    assert narrow["b0"] == ("NormalRV", [0.0, 1.0])
    assert narrow["b1"] == ("NormalRV", [0.0, 0.5])
    assert narrow["rho"] == ("BetaRV", [1.0, 19.0])


def test_prior_particles_draw_from_the_template_and_from_a_proposed_prior():
    """Check that prior particles draw from the template and from a proposed prior."""
    parameters = (("b0", "intercept"), ("b1", "signed"))
    plain = prior_particles(hypothesis("b0+b1*z", "binomial", parameters), 4096)
    assert 1.7 < plain["b0"].std() < 2.3 and 0.85 < plain["b1"].std() < 1.15
    proposed = prior_particles(
        hypothesis("b0+b1*z", "binomial", parameters, (("b1", ("Normal", 0.0, 0.25)),)), 4096
    )
    assert 0.2 < proposed["b1"].std() < 0.3
    set_prior_template("wide")
    wide = prior_particles(hypothesis("b0+b1*z", "binomial", parameters), 4096)
    assert 3.5 < wide["b0"].std() < 4.5


# ------------------------------------------------------- the reply and prompt


def test_a_priors_key_is_an_ignored_key_while_the_switch_is_off():
    """Check that a priors key is an ignored key while the switch is off."""
    assert llm_priors_enabled() is False
    block = {"b1": {"dist": "Normal", "params": [0.0, 1.5]}}
    model, ignored = parse_candidate_object(reply(block), "111", "binomial")
    assert ignored["ignored_keys"] == {"priors": block}
    assert model.coefficient_priors == ()
    assert model.as_dict()["fixed_priors"]["b1"] == ("Normal", 0.0, 1.0)
    # even a block that would be REJECTED while on is merely ignored while off
    bad = {"b1": {"dist": "Cauchy", "params": [0.0, 99.0]}}
    model, ignored = parse_candidate_object(reply(bad), "111", "binomial")
    assert ignored["ignored_keys"] == {"priors": bad}
    assert model.model_id == parse_candidate_object(reply(), "111", "binomial")[0].model_id


def test_a_priors_block_is_accepted_and_applied_while_the_switch_is_on():
    """Check that a priors block is accepted and applied while the switch is on."""
    set_llm_priors(True)
    block = {"b1": {"dist": "Normal", "params": [0.0, 1.5]}}
    model, ignored = parse_candidate_object(reply(block), "111", "binomial")
    assert ignored["ignored_keys"] == {}
    assert model.coefficient_priors == (("b1", ("Normal", 0.0, 1.5)),)
    assert model.as_dict()["fixed_priors"]["b1"] == ("Normal", 0.0, 1.5)
    plain, _ = parse_candidate_object(reply(), "111", "binomial")
    assert model.model_id != plain.model_id
    assert plain.as_dict()["prior_source"] == "role_template"
    assert model.as_dict()["prior_source"] == "llm"


@pytest.mark.parametrize(
    "block,fragment,parameters",
    [
        ({"b1": {"dist": "Normal", "params": [0.0, 99.0]}}, "sigma must be in", None),
        ({"b0": {"dist": "Normal", "params": [42.0, 1.0]}}, "|mu| must be at most", None),
        # `b1`'s default role is `signed` (real
        # support), which now accepts all four distributions, so this role
        # mismatch is reproduced by reinterpreting `b1` as `positive` instead -
        # the expression b0+b1*z is unchanged, only the declared role differs.
        (
            {"b1": {"dist": "Normal", "params": [0.0, 1.0]}},
            "positive support",
            {"b0": "intercept", "b1": "positive"},
        ),
        ({"b1": {"dist": "Wishart", "params": [1.0, 1.0]}}, "outside the menu", None),
        ({"nope": {"dist": "Normal", "params": [0.0, 1.0]}}, "not a declared coefficient", None),
        ({"pi": {"dist": "Beta", "params": [1.0, 4.0]}}, "family-owned dispersion parameter", None),
        ({}, "non-empty object", None),
    ],
)
def test_an_invalid_priors_block_rejects_the_candidate_while_the_switch_is_on(
    block, fragment, parameters
):
    """Check that an invalid priors block rejects the candidate while the switch is on."""
    set_llm_priors(True)
    with pytest.raises(ValueError, match="proposed prior rejected"):
        parse_candidate_object(reply(block, parameters=parameters), "111", "binomial")
    try:
        parse_candidate_object(reply(block, parameters=parameters), "111", "binomial")
    except ValueError as exc:
        assert fragment in str(exc)


def test_the_explicit_priors_argument_overrides_the_process_switch_both_ways():
    """Check that the explicit priors argument overrides the process switch both ways."""
    block = {"b1": {"dist": "Normal", "params": [0.0, 1.5]}}
    model, ignored = parse_candidate_object(reply(block), "111", "binomial", priors=True)
    assert model.coefficient_priors == (("b1", ("Normal", 0.0, 1.5)),) and ignored["ignored_keys"] == {}
    set_llm_priors(True)
    model, ignored = parse_candidate_object(reply(block), "111", "binomial", priors=False)
    assert model.coefficient_priors == () and ignored["ignored_keys"] == {"priors": block}


def test_the_v2_2_prompt_is_loaded_only_while_the_switch_is_on_and_supersedes_v2_1():
    """Check that the v2 2 prompt is loaded only while the switch is on and supersedes v2 1."""
    # v2.1's role-mapping sentence baked in
    # support EQUALITY, not containment, and was never used in a
    # paid call, so it is SUPERSEDED before any use rather than edited in
    # place - its bytes stay frozen on disk, unread by any switch.
    frozen = "f095edf1adb3b6533a9e68e73f98ca1dee3f23cd077e424b092393cb32b93aef"
    v2_1_frozen_sha = "2edb4e0c11ba00cb6f07be4ddef48d4da75893a717c3c22ca767dfc2bd33a273"
    v2_0 = count_proposer.PROMPT_FILE.read_bytes()
    v2_1_file = count_proposer.PROMPT_FILE_LLM_PRIORS.with_name("proposer_system_v2_1.txt")
    v2_1 = v2_1_file.read_bytes()
    v2_2 = count_proposer.PROMPT_FILE_LLM_PRIORS.read_bytes()
    assert hashlib.sha256(v2_1).hexdigest() == v2_1_frozen_sha
    assert count_proposer.prompt_identity() == {
        "prompt_version": "v2.0",
        "prompt_sha256": frozen,
    }
    assert count_proposer.system_prompt_bytes() == v2_0
    set_llm_priors(True)
    identity = count_proposer.prompt_identity()
    assert identity["prompt_version"] == "v2.2"
    assert identity["prompt_sha256"] not in (frozen, v2_1_frozen_sha)
    assert count_proposer.system_prompt_bytes() == v2_2
    # v2.2 is v2.0's bytes plus the same one appended section v2.1 had, and
    # nothing else.
    assert v2_2.startswith(v2_0)
    added = v2_2[len(v2_0) :].decode()
    assert added.startswith("\nOptional priors block\n")
    assert "priors" in added and "|mu| at most 5" in added
    # v2.1 and v2.2 differ in EXACTLY one line (18, 0-indexed 17): the
    # support-containment sentence, and the packet's menu table agrees with it.
    v2_1_lines, v2_2_lines = v2_1.decode().splitlines(), v2_2.decode().splitlines()
    assert len(v2_1_lines) == len(v2_2_lines) == 19
    differing = [i for i, (a, b) in enumerate(zip(v2_1_lines, v2_2_lines)) if a != b]
    assert differing == [17]
    assert "support must be contained in" in v2_2_lines[17]
    assert "support must be contained in" not in v2_1_lines[17]
    assert "LogNormal, Gamma or Beta" in v2_2_lines[17] and "take any of the four" in v2_2_lines[17]
    assert tuple(sorted(PRIOR_MENU_BY_ROLE["positive"])) == tuple(sorted(("LogNormal", "Gamma", "Beta")))
    assert tuple(sorted(PRIOR_MENU_BY_ROLE["intercept"])) == tuple(sorted(PRIOR_MENU))
    set_llm_priors(False)
    assert count_proposer.system_prompt_bytes() == v2_0


def test_the_packet_publishes_the_menu_only_while_the_switch_is_on():
    """Check that the packet publishes the menu only while the switch is on."""
    off = cell_constraints("111", "binomial")
    assert "priors" not in off
    set_llm_priors(True)
    on = cell_constraints("111", "binomial")
    assert set(on) - set(off) == {"priors"}
    assert on["priors"] == prior_menu_constraints()
    assert cell_constraints("111", "binomial", llm_priors=False) == off


# ------------------------------------------------- through an injected client


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_an_injected_client_reply_with_priors_is_accepted_while_the_switch_is_on(
    tmp_path, provider
):
    """Check that an injected client reply with priors is accepted while the switch is on."""
    set_llm_priors(True)
    text = [reply({"b1": {"dist": "Normal", "params": [0.0, 1.5]}})]
    if provider == "anthropic":
        client, calls = anthropic_client(text)
        proposer = AnthropicCountProposer(tmp_path / "on", client=client)
    else:
        client, calls = openai_client(text)
        proposer = OpenAICountProposer(tmp_path / "on", client=client)
    models, record = proposer.propose_one("{}", context=dict(CONTEXT), initial_family="binomial")
    assert record["status"] == "parsed" and len(models) == 1
    assert models[0].coefficient_priors == (("b1", ("Normal", 0.0, 1.5)),)
    assert record["prompt_version"] == "v2.2"
    assert record["ignored_keys"] == {}
    assert record["client_injected"] is True and record["calls"] == 1
    # the v2.2 text, not the v2.0 one, went on the wire
    system = calls[0]["system"] if provider == "anthropic" else calls[0]["messages"][0]["content"]
    assert "Optional priors block" in system


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_an_injected_client_reply_with_priors_is_ignored_while_the_switch_is_off(
    tmp_path, provider
):
    """Check that an injected client reply with priors is ignored while the switch is off."""
    block = {"b1": {"dist": "Normal", "params": [0.0, 1.5]}}
    text = [reply(block)]
    if provider == "anthropic":
        client, calls = anthropic_client(text)
        proposer = AnthropicCountProposer(tmp_path / "off", client=client)
    else:
        client, calls = openai_client(text)
        proposer = OpenAICountProposer(tmp_path / "off", client=client)
    models, record = proposer.propose_one("{}", context=dict(CONTEXT), initial_family="binomial")
    assert record["status"] == "parsed" and models[0].coefficient_priors == ()
    assert record["ignored_keys"] == {"priors": block}
    assert record["prompt_version"] == "v2.0"
    system = calls[0]["system"] if provider == "anthropic" else calls[0]["messages"][0]["content"]
    assert "Optional priors block" not in system


def test_an_injected_client_out_of_range_prior_is_rejected_then_repaired_once(tmp_path):
    """Check that an injected client out of range prior is rejected then repaired once."""
    set_llm_priors(True)
    text = [
        reply({"b1": {"dist": "Normal", "params": [0.0, 99.0]}}),
        reply({"b1": {"dist": "Normal", "params": [0.0, 0.5]}}),
    ]
    client, calls = anthropic_client(text)
    proposer = AnthropicCountProposer(tmp_path / "repair", client=client)
    models, record = proposer.propose_one("{}", context=dict(CONTEXT), initial_family="binomial")
    assert record["calls"] == 2 and record["repairs"] == 1
    assert record["status"] == "parsed"
    assert models[0].coefficient_priors == (("b1", ("Normal", 0.0, 0.5)),)
    rejected = record["rejected_candidates"]
    assert len(rejected) == 1 and rejected[0]["repair_eligible"] is True
    assert "sigma must be in [0.1, 5]" in rejected[0]["reason"]
    # the verbatim reason is what the single repair turn re-sends
    assert "sigma must be in [0.1, 5]" in calls[1]["messages"][-1]["content"]


def test_an_injected_client_role_mismatch_ends_the_round_without_a_candidate(tmp_path):
    """Check that an injected client role mismatch ends the round without a candidate."""
    set_llm_priors(True)
    # `b1`'s default role `signed` (real
    # support) now accepts all four menu distributions, so the role mismatch is
    # reproduced with `b1` reinterpreted as `positive` instead (same
    # expression, `b0+b1*z`) proposing a Normal, which (0,inf) support rejects.
    bad = reply(
        {"b1": {"dist": "Normal", "params": [0.0, 1.0]}},
        parameters={"b0": "intercept", "b1": "positive"},
    )
    client, _ = anthropic_client([bad, bad])
    proposer = AnthropicCountProposer(tmp_path / "role", client=client)
    models, record = proposer.propose_one("{}", context=dict(CONTEXT), initial_family="binomial")
    assert models == [] and record["status"] == "proposal_failed"
    assert len(record["rejected_candidates"]) == 2
    assert all("positive support" in item["reason"] for item in record["rejected_candidates"])


# ------------------------------------------- ablation addendum: Beta reachable


def test_a_positive_coefficient_accepts_a_beta_proposal_and_reaches_the_graph_and_the_probe():
    """Check that a positive coefficient accepts a beta proposal and reaches the graph and the probe."""
    # An earlier change note said `Beta` was unreachable for ANY coefficient role, because the earlier version read
    # role consistency as support EQUALITY. The CONTAINMENT fix makes Beta's
    # (0,1) support a valid subset of `positive`'s (0,inf), so it is reachable
    # here for the first time - and it reaches the PyMC graph and the prior
    # probe exactly like any other accepted proposal.
    parameters = (("b0", "intercept"), ("b1", "positive"))
    plain = hypothesis("b0+b1*z", "binomial", parameters)
    proposed = hypothesis("b0+b1*z", "binomial", parameters, (("b1", ("Beta", 2.0, 5.0)),))
    assert validate_proposed_prior("positive", {"dist": "Beta", "params": [2.0, 5.0]}) == (
        ("Beta", 2.0, 5.0),
        None,
    )
    assert plain.model_id != proposed.model_id
    assert proposed.as_dict()["fixed_priors"]["b1"] == ("Beta", 2.0, 5.0)
    u = np.linspace(0.05, 0.95, 8)
    y = np.full(8, 25)
    model = build_model(proposed, u, y)
    assert type(model["b1"].owner.op).__name__ == "BetaRV"
    args = [float(np.asarray(arg.eval())) for arg in model["b1"].owner.inputs[3:5]]
    assert args == [2.0, 5.0]
    probe, predictive = prior_probe(proposed, core_interface())
    assert probe["passed"] is True and predictive is not None
    particles = prior_particles(proposed, 512)
    assert np.all((particles["b1"] > 0.0) & (particles["b1"] < 1.0))


def test_a_signed_coefficient_also_accepts_a_beta_proposal():
    """Check that a signed coefficient also accepts a beta proposal."""
    # `signed` is real-valued, which contains (0,1) too - the same acceptance
    # that `positive`/`shape` get also holds for `intercept`/`signed`.
    parameters = (("b0", "intercept"), ("b1", "signed"))
    plain = hypothesis("b0+b1*z", "binomial", parameters)
    proposed = hypothesis("b0+b1*z", "binomial", parameters, (("b1", ("Beta", 1.0, 9.0)),))
    assert validate_proposed_prior("signed", {"dist": "Beta", "params": [1.0, 9.0]}) == (
        ("Beta", 1.0, 9.0),
        None,
    )
    assert plain.model_id != proposed.model_id
    u = np.linspace(0.05, 0.95, 8)
    y = np.full(8, 25)
    model = build_model(proposed, u, y)
    assert type(model["b1"].owner.op).__name__ == "BetaRV"
    probe, predictive = prior_probe(proposed, core_interface())
    assert probe["passed"] is True and predictive is not None


def test_a_signed_coefficient_accepts_a_lognormal_proposal_the_flipped_role_mismatch():
    """Check that a signed coefficient accepts a lognormal proposal the flipped role mismatch."""
    # the earlier `role_mismatch` fake-client
    # scenario proposed a LogNormal on a `signed` coefficient and treated it
    # as a rejection ("real support" is not `positive`'s support under the
    # old EQUALITY rule). Under the CONTAINMENT rule `signed` (real support)
    # accepts a LogNormal ((0,inf) is a subset of (-inf,inf)) - this is now an
    # ACCEPTED proposal, not a mismatch; see the fake-client development run for
    # the same flip through the full transport layer.
    parameters = (("b0", "intercept"), ("b1", "signed"))
    plain = hypothesis("b0+b1*z", "binomial", parameters)
    proposed = hypothesis("b0+b1*z", "binomial", parameters, (("b1", ("LogNormal", 0.0, 1.0)),))
    assert validate_proposed_prior("signed", {"dist": "LogNormal", "params": [0.0, 1.0]}) == (
        ("LogNormal", 0.0, 1.0),
        None,
    )
    assert plain.model_id != proposed.model_id
    u = np.linspace(0.05, 0.95, 8)
    y = np.full(8, 25)
    model = build_model(proposed, u, y)
    assert type(model["b1"].owner.op).__name__ == "LogNormalRV"
    probe, predictive = prior_probe(proposed, core_interface())
    assert probe["passed"] is True and predictive is not None


def test_a_positive_coefficient_still_rejects_a_normal_proposal():
    """Check that a positive coefficient still rejects a normal proposal."""
    # Regression pin: the real line is not inside (0,inf), so this is
    # the one role-support rejection that survives the containment fix.
    rule, reason = validate_proposed_prior("positive", {"dist": "Normal", "params": [0.0, 1.0]})
    assert rule is None and "positive support" in reason
    with pytest.raises(ValueError, match="positive support"):
        hypothesis(
            "b0+b1*z",
            "binomial",
            (("b0", "intercept"), ("b1", "positive")),
            (("b1", ("Normal", 0.0, 1.0)),),
        )


def test_an_injected_client_accepted_beta_proposal_scenario(tmp_path):
    """Check that an injected client accepted beta proposal scenario."""
    # Fake-client scenario mirroring the fake-client development run's pattern:
    # a Beta proposal on a `positive` coefficient, accepted
    # end to end through the transport layer. No network, no paid API.
    set_llm_priors(True)
    text = [
        reply(
            {"b1": {"dist": "Beta", "params": [2.0, 5.0]}},
            parameters={"b0": "intercept", "b1": "positive"},
        )
    ]
    client, _ = openai_client(text)
    proposer = OpenAICountProposer(tmp_path / "beta_accepted", client=client)
    models, record = proposer.propose_one("{}", context=dict(CONTEXT), initial_family="binomial")
    assert record["status"] == "parsed" and len(models) == 1
    assert models[0].coefficient_priors == (("b1", ("Beta", 2.0, 5.0)),)
    assert record["prompt_version"] == "v2.2"
    assert record["ignored_keys"] == {}
    plain = hypothesis("b0+b1*z", "binomial", (("b0", "intercept"), ("b1", "positive")))
    assert models[0].model_id != plain.model_id
