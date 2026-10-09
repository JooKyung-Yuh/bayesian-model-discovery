"""Unified protocol, third revision (2026-10-06): COMPOSITION_PARAGRAPH v2 and archive review rounds (BoxingGym, count transport).

The two paragraphs are bound to their registered bytes; the k = 3 request composes v2 (and, on a review
round, ARCHIVE_REVIEW v1 after it) and records the digests beside the file's; the review reply's object
shape is parsed and its `revive` list handed to the runner as given; the runner's flag is SUPPRESS and
checked before any output. The mock cells of the unit's report exercise the refit / retain / record loop.
"""
import hashlib
import json
import sys

import pytest

from . import boxinggym_core as bg
from . import count_proposer as cp
from .test_count_proposer import fixture, item

V2_SHA256 = "782a30b74cd25adb6b51351096cf97e31a7375d8630d726ee8eb7d7949c35be6"
REVIEW_SHA256 = "8b4c5359d81b6b3e5afed570c1e3c3f1bb217a57e70977bcbdb42d1238cd6cde"


@pytest.fixture(autouse=True)
def composition_v2():
    """The BoxingGym runner selects paragraph v2 for its k = 3 request (`set_composition_paragraph`)."""
    previous = cp.set_composition_paragraph("v2")
    yield
    cp.set_composition_paragraph(previous)


def test_both_paragraphs_are_the_registered_bytes():
    """Both paragraphs have the registered length and sha256 digest and carry the limit placeholder."""
    v2 = cp.COMPOSITION_PARAGRAPH_V2.encode("utf-8")
    review = cp.ARCHIVE_REVIEW_PARAGRAPH.encode("utf-8")
    assert len(v2) == 525 and hashlib.sha256(v2).hexdigest() == V2_SHA256 == cp.COMPOSITION_PARAGRAPH_V2_SHA256
    assert len(review) == 675 and hashlib.sha256(review).hexdigest() == REVIEW_SHA256 == cp.ARCHIVE_REVIEW_PARAGRAPH_SHA256
    assert cp.ARCHIVE_REVIEW_PARAGRAPH_VERSION == "v1"
    assert "{limit}" in cp.COMPOSITION_PARAGRAPH_V2 and "{limit}" in cp.ARCHIVE_REVIEW_PARAGRAPH
    assert "sub-expression" not in cp.COMPOSITION_PARAGRAPH_V2 and "sub-expression" in cp.COMPOSITION_PARAGRAPH_V1
    assert cp.COMPOSITION_PARAGRAPH_V1 is cp.COMPOSITION_PARAGRAPH
    filled = cp.fill_limit(cp.ARCHIVE_REVIEW_PARAGRAPH, 3)
    assert "your 3 new candidates" in filled and '{"candidates": [...], "revive": [...]}' in filled


def test_composed_requests_and_their_identity():
    """Composed requests and their identity records are correct for k = 1, 2 and 3, with and without review."""
    text, identity = cp.composed_system_prompt(3)
    assert text == cp.system_prompt_bytes().decode() + cp.fill_limit(cp.COMPOSITION_PARAGRAPH_V2, 3)
    assert identity["composition_paragraph_version"] == "v2"
    assert identity["composition_paragraph_sha256"] == V2_SHA256
    assert identity["composed_prompt_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    review_text, review_identity = cp.composed_system_prompt(3, archive_review=True)
    assert review_text == text + cp.fill_limit(cp.ARCHIVE_REVIEW_PARAGRAPH, 3)
    # the digest of the whole text sent, under both names (the ChemBench and the BoxingGym key)
    assert review_identity["composed_prompt_sha256"] == hashlib.sha256(review_text.encode()).hexdigest()
    assert review_identity["archive_review_paragraph_sha256"] == REVIEW_SHA256
    assert review_identity["archive_review_prompt_sha256"] == hashlib.sha256(review_text.encode()).hexdigest()
    assert cp.composed_prompt_identity(3, archive_review=True) == review_identity
    # v2 is written for three; k = 2 is composed with v1 (the runner selects v1 for it), and no
    # review rides on k = 2 or k = 1
    with pytest.raises(ValueError, match="three candidates"):
        cp.composed_system_prompt(2)
    with pytest.raises(ValueError, match="three candidates"):
        cp.composed_system_prompt(2, archive_review=True)
    with pytest.raises(ValueError, match="k > 1 request"):
        cp.composed_system_prompt(1, archive_review=True)
    cp.set_composition_paragraph("v1")
    two, two_identity = cp.composed_system_prompt(2)
    assert two == cp.system_prompt_bytes().decode() + cp.COMPOSITION_PARAGRAPH_V1.format(limit=2)
    assert "composition_paragraph_version" not in two_identity
    cp.set_composition_paragraph("v2")
    plain, plain_identity = cp.composed_system_prompt(1)
    assert plain == cp.system_prompt_bytes().decode() and plain_identity == cp.prompt_identity()


CELL, FAMILY = "111", "beta_binomial"


def _structures():
    """Return the single seed structure used by the parser tests."""
    from .count_models import Hypothesis
    return (Hypothesis("b0", FAMILY, {"b0": "intercept"}, ("X1",)),)


def test_review_reply_object_is_parsed_and_revive_handed_over():
    """A review reply object is parsed and its revive list is passed on unchanged."""
    body = {"candidates": [item(), {"expression": "b0+b1*X1", "family": FAMILY, "parameters": {"b0": "intercept", "b1": "signed"}}],
            "revive": ["abc123", "def456", 7]}
    models, ignored = cp.parse_candidate_array(json.dumps(body), CELL, FAMILY, _structures(), proposal_limit=3,
                                               archive_review=True)
    assert len(models) == 2
    assert ignored["revive_requested"] == ["abc123", "def456", 7] and ignored["reply_shape"] == "object"
    bare, bare_ignored = cp.parse_candidate_array(json.dumps([item()]), CELL, FAMILY, _structures(), proposal_limit=3,
                                                  archive_review=True)
    assert len(bare) == 1 and bare_ignored["revive_requested"] == [] and bare_ignored["reply_shape"] == "array"
    with pytest.raises(ValueError, match=r'one JSON object \{"candidates"'):
        cp.parse_candidate_array(json.dumps({"revive": []}), CELL, FAMILY, _structures(), proposal_limit=3,
                                 archive_review=True)
    with pytest.raises(ValueError, match="JSON array of 3 candidates"):
        cp.parse_candidate_array(json.dumps({"candidates": [item()]}), CELL, FAMILY, _structures(), proposal_limit=3)
    _plain, plain_ignored = cp.parse_candidate_array(json.dumps([item()]), CELL, FAMILY, _structures(), proposal_limit=3)
    assert "revive_requested" not in plain_ignored and "reply_shape" not in plain_ignored


def test_a_review_round_request_composes_both_paragraphs_and_records_the_revive_list(tmp_path):
    """A review-round request composes both paragraphs and records the revive list."""
    reply = json.dumps({"candidates": [item()], "revive": ["abc123"]})
    proposer, calls, *_ = fixture(tmp_path, text=reply)
    context = {"run_id": "synthetic", "world_seed": 7, "cell": CELL, "proposal_rep": 0, "round": 5}
    proposed, record = proposer.propose_one(reply and "{\"packet\": 1}", context=context, initial_family=FAMILY,
                                            proposal_limit=3, archive_review=True)
    assert record["status"] == "parsed" and len(proposed) == 1
    review_text, identity = cp.composed_system_prompt(3, archive_review=True)
    assert calls[0]["system"] == review_text
    assert record["archive_review_prompt_sha256"] == identity["archive_review_prompt_sha256"]
    assert record["composition_paragraph_version"] == "v2" and record["archive_review_paragraph_version"] == "v1"
    assert record["revive_requested"] == ["abc123"] and record["reply_shape"] == "object"
    request = json.loads((tmp_path / "transport" / "round_05.request.json").read_text())
    assert request["identity"]["archive_review_paragraph_sha256"] == REVIEW_SHA256
    with pytest.raises(ValueError, match="k > 1 request"):
        proposer.propose_one("{\"packet\": 1}", context={**context, "round": 6}, initial_family=FAMILY,
                             proposal_limit=1, archive_review=True)


def test_an_ordinary_k3_request_is_v2_without_the_review_paragraph(tmp_path):
    """An ordinary k = 3 request uses paragraph v2 and carries no review paragraph."""
    proposer, calls, *_ = fixture(tmp_path, text=json.dumps([item()]))
    context = {"run_id": "synthetic", "world_seed": 7, "cell": CELL, "proposal_rep": 0, "round": 4}
    _, record = proposer.propose_one("{\"packet\": 1}", context=context, initial_family=FAMILY, proposal_limit=3)
    text, identity = cp.composed_system_prompt(3)
    assert calls[0]["system"] == text and record["composed_prompt_sha256"] == identity["composed_prompt_sha256"]
    assert "archive_review_prompt_sha256" not in record and "revive_requested" not in record


def test_an_object_on_a_non_review_round_is_the_contract_failure_with_one_repair(tmp_path):
    # The object shape is accepted on review rounds only; on any other round it is
    # the existing reply-contract failure - one repair turn - after which the array is taken.
    """An object reply on a non-review round is a contract failure with one repair turn."""
    from .test_count_proposer import p1_fixture
    wrong = json.dumps({"candidates": [item()], "revive": []})
    right = json.dumps([item()])
    proposer, calls, context = p1_fixture(tmp_path, [wrong, right])
    proposed, record = proposer.propose_one("{\"packet\": 1}", context=context, initial_family=FAMILY, proposal_limit=3)
    assert len(calls) == 2 and record["repairs"] == 1 and record["status"] == "parsed" and len(proposed) == 1
    assert calls[0]["system"] == cp.composed_system_prompt(3)[0]
    assert "JSON array of 3 candidates" in json.dumps(calls[1]["messages"])
    assert "revive_requested" not in record and "reply_shape" not in record


def _parse(monkeypatch, tmp_path, *extra):
    seen = []
    monkeypatch.setattr(bg, "run", seen.append)
    monkeypatch.setattr(sys, "argv", ["boxinggym_core", "--output", str(tmp_path / "out"),
                                      "--task", "dugongs", "--proposer", "mock", *extra])
    bg.main()
    return seen[0]


def test_runner_flag_is_suppressed_unless_stated(monkeypatch, tmp_path):
    """The archive-review flag is absent from the namespace unless it is stated."""
    unstated = _parse(monkeypatch, tmp_path, "--cell", "111")
    assert "archive_review_rounds" not in vars(unstated)
    stated = _parse(monkeypatch, tmp_path, "--cell", "111", "--proposal-limit", "3", "--rounds", "30",
                    "--archive-review-rounds", "5,10,15,20,25,30")
    assert stated.archive_review_rounds == "5,10,15,20,25,30"


class _Reached(Exception):
    """Sentinel exception raised when the run gets past the option checks."""


@pytest.mark.parametrize("extra,match", [
    (("--archive-review-rounds", "5", "--rounds", "30"), "proposal-limit 3"),
    (("--archive-review-rounds", "40", "--rounds", "30", "--proposal-limit", "3"), "round list entries"),
])
def test_bad_flags_are_refused_before_any_output(monkeypatch, tmp_path, extra, match):
    """Invalid archive-review flags are refused before any output exists."""
    run = bg.run
    args = _parse(monkeypatch, tmp_path, "--cell", "111", *extra)
    with pytest.raises(ValueError, match=match):
        run(args)
    assert not (tmp_path / "out").exists()


def test_good_flags_pass_the_checks(monkeypatch, tmp_path):
    """Valid flags pass the option checks and the run proceeds to its first later call."""
    run = bg.run
    args = _parse(monkeypatch, tmp_path, "--cell", "111", "--rounds", "30", "--proposal-limit", "3",
                  "--initial-proposals", "12", "--active-maximum", "12", "--initial-catalogue", "off",
                  "--archive-review-rounds", "5,10,15,20,25,30")

    def reached(*_):
        """Stand-in that stops the run at the first call after the option checks."""
        raise _Reached

    monkeypatch.setattr(bg, "compat_config", reached)  # the first call after the checks
    with pytest.raises(_Reached):
        run(args)
    assert not (tmp_path / "out").exists()
    assert bg.ARCHIVE_REVIEW_RULE.startswith("on a listed round")
    # The recorded rule names the array acceptance and the four refusal reasons.
    for word in ("revive_missing", "active / not_retired / unknown / over_limit", "no repair turn"):
        assert word in bg.ARCHIVE_REVIEW_RULE
    assert bg.ARCHIVE_REVIVE_LIMIT == 2
