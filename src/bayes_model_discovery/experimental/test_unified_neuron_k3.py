"""Unified protocol, k candidates and criteria record: NeuronBench k candidates per call and the design criteria record.

Synthetic replies through the injected client (the proposer tests' fixtures); no fitting, no live client.
"""
import hashlib
import json

import numpy as np
import pytest

from . import design_policies as dp
from . import neuron_core as nc
from . import neuron_proposer as p
from . import vector_inference as vi
from .test_neuron_proposer import (  # noqa: F401
    context,
    probe,
    proposer,
    reply,
    response,
    text_packet,
)


@pytest.mark.usefixtures("probe")
def test_limit_one_is_the_transport_as_before(tmp_path):
    """With proposal_limit 1 the request and record are those of the plain single-candidate transport."""
    instance, client = proposer(tmp_path, "anthropic", [response("anthropic")])
    assert instance.proposal_limit == 1
    candidate, record = instance.propose_one(text_packet(), context=context())
    assert candidate is not None and not isinstance(candidate, list)
    assert client.calls[0]["system"] == p.PROMPT_FILE.read_text()
    assert "proposal_limit" not in record and "composed_prompt_sha256" not in record
    assert record["valid_candidate_count"] == 1
    (tmp_path / "explicit").mkdir()
    explicit, _ = proposer(tmp_path / "explicit", "anthropic", [response("anthropic")], proposal_limit=1)
    assert explicit.proposal_limit == 1


@pytest.mark.parametrize("limit", (0, 4, True, 2.0))
def test_other_limits_are_refused(tmp_path, limit):
    """Proposal limits outside the integers 1..3 are refused with a ValueError."""
    with pytest.raises(ValueError, match="proposal_limit must be an integer 1..3"):
        proposer(tmp_path, "anthropic", [], proposal_limit=limit)


def test_composed_prompt_identity():
    """The composed system prompt and its identity record match the registered paragraph bytes."""
    text, identity = p.composed_system_prompt(1)
    assert text == p.PROMPT_FILE.read_text() and identity == {}
    composed, identity = p.composed_system_prompt(3)
    paragraph = p.COMPOSITION_PARAGRAPH.replace("{limit}", "3")
    assert composed == p.PROMPT_FILE.read_text() + paragraph
    assert identity == {"proposal_limit": 3, "composition_paragraph_version": "v2",
                        "composition_paragraph_sha256": p.COMPOSITION_PARAGRAPH_SHA256,
                        "composed_prompt_sha256": hashlib.sha256(composed.encode()).hexdigest()}
    assert "exactly 3 candidate objects" in paragraph and "distinct, non-equivalent" in paragraph
    # v2 is the registered bytes (composition_paragraph_v2.txt) and composes for 3 alone.
    raw = p.COMPOSITION_PARAGRAPH.encode("utf-8")
    assert len(raw) == 525 and hashlib.sha256(raw).hexdigest() == p.COMPOSITION_PARAGRAPH_SHA256
    assert p.COMPOSITION_PARAGRAPH_SHA256 == "782a30b74cd25adb6b51351096cf97e31a7375d8630d726ee8eb7d7949c35be6"
    with pytest.raises(ValueError, match="written for 3 candidates"):
        p.composed_system_prompt(2)


@pytest.mark.usefixtures("probe")
def test_an_array_of_three_gives_three_candidates(tmp_path):
    """A reply array of three valid objects yields three candidates and a complete record."""
    body = json.dumps([reply("h_current"), reply("m_current"), reply("inward")])
    instance, client = proposer(tmp_path, "anthropic", [response("anthropic", body)], proposal_limit=3)
    candidates, record = instance.propose_one(text_packet(), context=context())
    assert isinstance(candidates, list) and [c.mechanism for c in candidates] == ["h_current", "m_current", "inward"]
    assert record["status"] == "parsed" and record["valid_candidate_count"] == 3 and record["repairs"] == 0
    assert record["proposal_limit"] == 3 and record["array_length"] == 3
    assert record["candidate_model_ids"] == [c.model_id for c in candidates]
    composed, identity = p.composed_system_prompt(3)
    assert client.calls[0]["system"] == composed
    assert record["composed_prompt_sha256"] == identity["composed_prompt_sha256"]
    request = json.loads((tmp_path / "records" / "round_01.request.json").read_text())
    assert request["identity"]["composed_prompt_sha256"] == identity["composed_prompt_sha256"]
    assert request["bodies"][0]["system"] == composed
    assert {c.model_id for c in candidates} <= instance.known_ids


@pytest.mark.usefixtures("probe")
def test_an_invalid_item_is_recorded_and_the_valid_ones_kept(tmp_path):
    """An invalid array item is recorded as rejected while the valid items are kept."""
    bad = {**reply("m_current"), "mechanism": "not_a_mechanism"}
    body = json.dumps([reply("h_current"), bad, reply("inward")])
    instance, _ = proposer(tmp_path, "anthropic", [response("anthropic", body)], proposal_limit=3)
    candidates, record = instance.propose_one(text_packet(), context=context())
    assert [c.mechanism for c in candidates] == ["h_current", "inward"]
    assert record["status"] == "parsed" and record["valid_candidate_count"] == 2 and record["repairs"] == 0
    (rejected,) = record["rejected_candidates"]
    assert rejected["array_index"] == 1 and rejected["stage"] == "vocabulary" and rejected["repair_eligible"] is False
    (entry,) = record["archive_entries"]
    assert entry["outcome"] == "rejected" and entry["mechanism"] == "not_a_mechanism"


@pytest.mark.usefixtures("probe")
def test_a_lone_object_at_limit_three_is_repaired_into_an_array(tmp_path):
    """A single object at limit 3 triggers one repair turn that yields an array of three."""
    first = response("anthropic", json.dumps(reply("h_current")))
    second = response("anthropic", json.dumps([reply("h_current"), reply("m_current"), reply("inward")]))
    instance, client = proposer(tmp_path, "anthropic", [first, second], proposal_limit=3)
    candidates, record = instance.propose_one(text_packet(), context=context())
    assert len(candidates) == 3 and record["status"] == "parsed" and record["repairs"] == 1
    assert record["rejected_candidates"][0]["stage"] == "shape"
    assert record["rejected_candidates"][0]["reason"] == p.ARRAY_REFUSAL.format(limit=3)
    repair_turn = client.calls[1]["messages"][-1]["content"]
    assert "JSON array of exactly 3 objects" in repair_turn and "{limit}" not in repair_turn


@pytest.mark.usefixtures("probe")
def test_no_valid_item_at_all_is_a_repairable_failure(tmp_path):
    """An array with no valid item triggers one repair turn and then succeeds."""
    bad = json.dumps([{**reply("h_current"), "mechanism": "nope"}])
    good = json.dumps([reply("h_current")])
    instance, _ = proposer(tmp_path, "anthropic", [response("anthropic", bad), response("anthropic", good)],
                           proposal_limit=3)
    candidates, record = instance.propose_one(text_packet(), context=context())
    assert [c.mechanism for c in candidates] == ["h_current"] and record["repairs"] == 1
    assert any(r["reason"] == p.NO_VALID_ARRAY_CANDIDATE for r in record["rejected_candidates"])


@pytest.mark.usefixtures("probe")
def test_too_many_items_are_refused(tmp_path):
    """An array longer than the limit is refused and the proposal fails."""
    body = json.dumps([reply("h_current"), reply("m_current"), reply("inward"), reply("h_current")])
    instance, _ = proposer(tmp_path, "anthropic", [response("anthropic", body)], proposal_limit=3)
    candidate, record = instance.propose_one(text_packet(), context=context())
    assert candidate is None and record["status"] == "proposal_failed"
    assert record["rejected_candidates"][0]["reason"] == p.ARRAY_REFUSAL.format(limit=3)


def test_runner_options_are_suppressed_unless_stated_and_bounded(tmp_path):
    """Runner options stay out of the namespace unless stated, and an out-of-range limit is refused before output."""
    unstated = nc.build_parser().parse_args(["--output", "x"])
    assert "proposal_limit" not in vars(unstated) and "record_design_criteria" not in vars(unstated)
    stated = nc.build_parser().parse_args(["--output", "x", "--proposal-limit", "3", "--record-design-criteria"])
    assert stated.proposal_limit == 3 and stated.record_design_criteria is True
    out = tmp_path / "never"
    args = nc.build_parser().parse_args(["--output", str(out), "--cell", "111", "--proposal-limit", "4"])
    with pytest.raises(ValueError, match="proposal_limit must be an integer 1..3"):
        nc.run(args)
    assert not out.exists()


def _bank(seed=0):
    """Build a small synthetic particle bank (means, sigmas, log weights, model indices) from a seed."""
    rng = np.random.default_rng(seed)
    per_model, designs, features = (3, 2), 4, 2
    means, logweights, indices = [], [], []
    model_logw = np.log(np.array([0.7, 0.3]))
    for m, n in enumerate(per_model):
        means.append(rng.normal(size=(n, designs, features)))
        logweights += [model_logw[m] - np.log(n)] * n
        indices += [m] * n
    return {"means": np.concatenate(means), "sigmas": np.full(sum(per_model), 0.5),
            "logweights": np.asarray(logweights), "model_indices": np.asarray(indices)}


def test_point_disagreement_is_the_core_rule_per_feature_summed():
    """Point disagreement equals the core rule applied per feature and summed over features."""
    bank = _bank()
    result = vi.point_disagreement(bank)
    means = bank["means"]
    model_means = np.stack([means[:3].mean(0), means[3:].mean(0)])  # [M, D, F]
    weights = np.array([0.7, 0.3])
    expected = np.stack([dp.point_disagreement(model_means[:, :, f], weights) for f in range(2)], axis=1)
    np.testing.assert_allclose(result["per_feature"], expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result["point_disagreement"], expected.sum(1), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result["model_weights"], weights, rtol=1e-12, atol=0)
    assert nc.DESIGN_CRITERIA_DEFINITIONS["point_disagreement_definition"] == vi.POINT_DISAGREEMENT_DEFINITION


def test_decompose_adds_the_parts_without_moving_the_joint_rows():
    """Decomposing the joint EIG adds structure and parameter parts without changing the joint rows."""
    bank = _bank(3)
    chosen, rows = vi.joint_eig(bank, 7, 128)
    chosen_d, rows_d = vi.joint_eig(bank, 7, 128, decompose=True)
    assert chosen == chosen_d
    for row, part in zip(rows, rows_d, strict=True):
        assert {k: part[k] for k in row} == row
        assert abs(part["structure_eig"] + part["parameter_eig"] - part["eig"]) <= 1e-12
