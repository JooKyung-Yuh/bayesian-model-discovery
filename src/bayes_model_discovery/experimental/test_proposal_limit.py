"""How many candidates a round asks for, and what that binds.

`--proposal-limit 1..3` opens on the two runners that had it fixed at one. The
whole point of the option is that k = 1 is untouched: the request bytes, the
transport record's identity block, the config keys and the label receipt are all
exactly what they were, and the packaged prompt files are never edited at any k.

Above one, the runner COMPOSES the request - the selected prompt file's bytes,
then one general paragraph naming the reply shape and the composition rule - and
the sha256 of that composed text is bound into the run's config, its label
receipt and the manifest's runtime settings, beside the file's own digest.

Every candidate here is synthetic.
"""

import json

import pytest

from .count_models import Hypothesis
from .count_proposer import (
    COMPOSITION_PARAGRAPH,
    PROPOSAL_LIMIT_RANGE,
    composed_prompt_identity,
    composed_system_prompt,
    parse_candidate_array,
    parse_candidate_object,
    prompt_identity,
    system_prompt_bytes,
    validate_proposal_limit,
)

CELL = "111"
FAMILY = "binomial"


def candidate(expression, parameters, family=FAMILY):
    """Build a synthetic candidate dict from an expression, its parameter kinds and a family."""
    return {"expression": expression, "family": family, "parameters": dict(parameters)}


def structures():
    """Return the initial hypotheses of the default family."""
    from .count_models import initial_hypotheses

    return initial_hypotheses(FAMILY)


# ---------------------------------------------------------------------------
# The bound
# ---------------------------------------------------------------------------

def test_the_bound_is_one_to_three():
    """Test that the bound is one to three."""
    assert PROPOSAL_LIMIT_RANGE == (1, 3)
    for limit in (1, 2, 3):
        assert validate_proposal_limit(limit) == limit
    for bad in (0, 4, -1, 1.0, True, None, "2"):
        with pytest.raises(ValueError, match="proposal limit must be 1..3"):
            validate_proposal_limit(bad)


# ---------------------------------------------------------------------------
# k = 1 keeps every byte
# ---------------------------------------------------------------------------

def test_one_candidate_sends_the_packaged_file_and_nothing_else():
    """Test that one candidate sends the packaged file and nothing else."""
    text, identity = composed_system_prompt(1)
    assert text == system_prompt_bytes().decode()
    assert identity == prompt_identity()
    assert composed_prompt_identity(1) == prompt_identity()
    assert COMPOSITION_PARAGRAPH.format(limit=1) not in text


@pytest.mark.parametrize("limit", (2, 3))
def test_a_composed_request_is_the_file_plus_one_paragraph(limit):
    """Test that a composed request is the file plus one paragraph."""
    text, identity = composed_system_prompt(limit)
    stored = system_prompt_bytes().decode()
    assert text.startswith(stored)
    assert text[len(stored):] == COMPOSITION_PARAGRAPH.format(limit=limit)
    assert identity["prompt_version"] == prompt_identity()["prompt_version"]
    assert identity["prompt_sha256"] == prompt_identity()["prompt_sha256"]
    assert identity["proposal_limit"] == limit
    import hashlib

    assert identity["composed_prompt_sha256"] == hashlib.sha256(text.encode()).hexdigest()


def test_a_composed_request_for_two_composes_the_g1_paragraph():
    """Test that a composed request for two composes the g1 paragraph."""
    # COMPOSITION_PARAGRAPH v2 says "three" in words, so it composes for 3 alone;
    # k = 2 (no kit of the unified protocol uses it) keeps paragraph v1, byte for byte as before,
    # and the BoxingGym runner selects v2 for k = 3 only.
    from .count_proposer import COMPOSITION_PARAGRAPH_V1, set_composition_paragraph

    text, identity = composed_system_prompt(2)
    stored = system_prompt_bytes().decode()
    assert text == stored + COMPOSITION_PARAGRAPH_V1.format(limit=2)
    assert identity["proposal_limit"] == 2
    assert "composition_paragraph_version" not in identity
    previous = set_composition_paragraph("v2")
    try:
        assert composed_system_prompt(3)[1]["composition_paragraph_version"] == "v2"
        with pytest.raises(ValueError, match="three candidates"):
            composed_system_prompt(2, archive_review=True)
    finally:
        set_composition_paragraph(previous)


def test_the_paragraph_names_no_benchmark_and_no_example():
    """Test that the paragraph names no benchmark and no example."""
    from .count_proposer import COMPOSITION_PARAGRAPH_V2

    paragraph = COMPOSITION_PARAGRAPH.format(limit=3)
    assert "differ from one another" in paragraph
    assert "contains another as a sub-expression" in paragraph
    v2 = COMPOSITION_PARAGRAPH_V2.replace("{limit}", "3")
    assert "distinct, non-equivalent hypotheses" in v2
    assert "primary functional composition" in v2
    assert "sub-expression" not in v2
    for text in (paragraph, v2):
        lowered = text.lower()
        for forbidden in ("michaelis", "menten", "inhibition", "pingpong", "enzyme",
                          "chembench", "boxinggym", "c_a", "kcat", "km"):
            assert forbidden not in lowered
        # A rule, not an example: no expression syntax is quoted.
        assert not any(character in text for character in "*/^")


def test_the_packaged_prompt_files_are_never_written():
    """Test that the packaged prompt files are never written."""
    from pathlib import Path

    prompts = Path(__file__).resolve().parent / "prompts"
    before = {path: path.read_bytes() for path in sorted(prompts.glob("proposer_system_*.txt"))}
    from .count_proposer import set_composition_paragraph

    for limit in (1, 2, 3):
        composed_system_prompt(limit)
    previous = set_composition_paragraph("v2")
    try:
        composed_system_prompt(3)
        composed_system_prompt(3, archive_review=True)
    finally:
        set_composition_paragraph(previous)
    assert {path: path.read_bytes() for path in sorted(prompts.glob("proposer_system_*.txt"))} \
        == before


# ---------------------------------------------------------------------------
# Parsing a k > 1 reply
# ---------------------------------------------------------------------------

def test_an_array_of_three_is_parsed_into_three_hypotheses():
    """Test that an array of three is parsed into three hypotheses."""
    payload = json.dumps([
        candidate("b0", {"b0": "intercept"}),
        candidate("b0+b1*u", {"b0": "intercept", "b1": "signed"}),
        candidate("b0+b1*u+b2*square(u)",
                  {"b0": "intercept", "b1": "signed", "b2": "signed"}),
    ])
    models, ignored = parse_candidate_array(
        payload, CELL, FAMILY, structures(), proposal_limit=3
    )
    assert [type(model) for model in models] == [Hypothesis] * 3
    assert [model.expression for model in models] == [
        "b0", "b0+b1*u", "b0+b1*u+b2*square(u)"
    ]
    assert ignored["rejected_candidates"] == []
    assert len(ignored["ignored_parameters"]) == 3


def test_one_bad_candidate_is_rejected_and_the_others_are_kept():
    """Test that one bad candidate is rejected and the others are kept."""
    payload = json.dumps([
        candidate("b0", {"b0": "intercept"}),
        candidate("b0+b1*undeclared_input", {"b0": "intercept", "b1": "signed"}),
    ])
    models, ignored = parse_candidate_array(
        payload, CELL, FAMILY, structures(), proposal_limit=3
    )
    assert [model.expression for model in models] == ["b0"]
    assert [entry["index"] for entry in ignored["rejected_candidates"]] == [1]
    assert ignored["rejected_candidates"][0]["reason"] == "unknown input name undeclared_input"


@pytest.mark.parametrize("payload,reason", (
    ("not json at all", "reply is not valid JSON"),
    ('{"expression": "b0"}', "reply must be a JSON array of 2 candidates"),
    ("[]", "reply must hold 1..2 candidates"),
    (('[{"expression":"b0","family":"binomial","parameters":{"b0":"intercept"}},'
      '{"expression":"b0","family":"binomial","parameters":{"b0":"intercept"}},'
      '{"expression":"b0","family":"binomial","parameters":{"b0":"intercept"}}]'),
     "reply must hold 1..2 candidates"),
))
def test_a_batch_level_failure_raises_for_the_one_repair_call(payload, reason):
    """Test that a batch level failure raises for the one repair call."""
    with pytest.raises(ValueError) as raised:
        parse_candidate_array(payload, CELL, FAMILY, structures(), proposal_limit=2)
    assert reason in str(raised.value)


def test_an_array_in_which_nothing_is_accepted_is_a_batch_failure():
    """Test that an array in which nothing is accepted is a batch failure."""
    payload = json.dumps([candidate("b0+b1*undeclared_input",
                                    {"b0": "intercept", "b1": "signed"})])
    with pytest.raises(ValueError, match="no candidate in the array was accepted"):
        parse_candidate_array(payload, CELL, FAMILY, structures(), proposal_limit=2)


def test_the_one_object_reply_still_refuses_an_array_with_its_own_message():
    """Test that the one object reply still refuses an array with its own message."""
    payload = json.dumps([candidate("b0", {"b0": "intercept"})])
    with pytest.raises(ValueError, match="reply must be one JSON object, not an array"):
        parse_candidate_object(payload, CELL, FAMILY, structures())


def test_both_reply_shapes_judge_one_candidate_identically():
    """The array path is the same judge, not a second implementation."""
    item = candidate("b0+b1*undeclared_input", {"b0": "intercept", "b1": "signed"})
    with pytest.raises(ValueError) as single:
        parse_candidate_object(json.dumps(item), CELL, FAMILY, structures())
    _, ignored = parse_candidate_array(
        json.dumps([candidate("b0", {"b0": "intercept"}), item]),
        CELL, FAMILY, structures(), proposal_limit=2,
    )
    assert ignored["rejected_candidates"][0]["reason"] == str(single.value)


# ---------------------------------------------------------------------------
# What the runners open, and what they bind
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_name", ("boxinggym_core", "chembench_core"))
def test_both_runners_open_the_option_and_default_to_one(module_name):
    """Test that both runners open the option and default to one."""
    import importlib

    module = importlib.import_module(f".{module_name}", __package__)
    text = open(module.__file__).read()  # noqa: SIM115 - read once, for one assertion
    assert '"--proposal-limit", type=int, default=argparse.SUPPRESS' in text
    assert '--proposal-limit must be 1..3' in text
    assert 'proposal_limit=proposal_limit' in text


def test_the_label_receipt_binds_the_composed_digest_only_above_one():
    """Test that the label receipt binds the composed digest only above one."""
    from types import SimpleNamespace

    from . import run_label_policy

    def settings(**overrides):
        """Resolve boxinggym runtime settings from synthetic arguments plus overrides."""
        args = SimpleNamespace(
            profile="dev", rounds=5, cell="111", seed=11, proposer="mock", llm_model=None,
            run_label=None, prompt_version="v2.0", prompt_sha256="0" * 64,
            d0_policy="random2", q_policy="native", **overrides,
        )
        return run_label_policy.bmd_runtime_settings("boxinggym", args)

    plain = settings()
    assert "composed_prompt_sha256" not in plain
    assert "proposal_limit" not in plain
    composed = composed_prompt_identity(3)
    bound = settings(proposal_limit=3,
                     composed_prompt_sha256=composed["composed_prompt_sha256"])
    assert bound["proposal_limit"] == 3
    assert bound["composed_prompt_sha256"] == composed["composed_prompt_sha256"]
    with pytest.raises(ValueError, match="proposal limit of 2..3"):
        settings(proposal_limit=1, composed_prompt_sha256=composed["composed_prompt_sha256"])
