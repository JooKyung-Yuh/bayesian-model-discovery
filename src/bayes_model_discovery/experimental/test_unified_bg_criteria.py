"""Unified protocol, design-criteria record: BoxingGym `--record-design-criteria`, `--rounds` up to 30.

The record (acquisition/tNN_criteria.json: joint, Model and Parameter expected information gain (EIG) and point disagreement of every
candidate) is opt-in; unstated, the parser namespace, config.json and every file are as before. The mock
smokes of the unit's report exercise the record itself; here are the parser, the refusals and the
definitions.
"""
import sys

import numpy as np
import pytest

from . import boxinggym_core as bg
from . import design_policies as dp
from . import positive_inference


def _parse(monkeypatch, tmp_path, *extra):
    """The runner's own parser, with `run` replaced by a recorder (the pattern of the other runner-option tests)."""
    seen = []
    monkeypatch.setattr(bg, "run", seen.append)
    monkeypatch.setattr(sys, "argv", ["boxinggym_core", "--output", str(tmp_path / "out"),
                                      "--task", "dugongs", "--proposer", "mock", *extra])
    bg.main()
    return seen[0]


def test_rounds_accepts_up_to_thirty_and_refuses_beyond(monkeypatch, tmp_path):
    """The --rounds option accepts 1..30 and exits with code 2 for 31, 0 and negatives."""
    assert _parse(monkeypatch, tmp_path, "--rounds", "30").rounds == 30
    assert _parse(monkeypatch, tmp_path, "--rounds", "20").rounds == 20
    for text in ("31", "0", "-1"):
        with pytest.raises(SystemExit) as refused:
            _parse(monkeypatch, tmp_path, "--rounds", text)
        assert refused.value.code == 2


def test_the_record_option_leaves_no_key_unless_stated(monkeypatch, tmp_path):
    """The record option adds no namespace key unless it is stated, and changes nothing else."""
    unstated = _parse(monkeypatch, tmp_path, "--cell", "111")
    assert "record_design_criteria" not in vars(unstated)
    stated = _parse(monkeypatch, tmp_path, "--cell", "111", "--record-design-criteria")
    assert stated.record_design_criteria is True
    assert {k: v for k, v in vars(stated).items() if k != "record_design_criteria"} == vars(unstated)


def test_the_record_is_refused_on_a_multi_component_task_before_any_output(monkeypatch, tmp_path):
    """Recording design criteria on a multi-component task is refused before any output exists."""
    run = bg.run
    seen = []
    monkeypatch.setattr(bg, "run", seen.append)
    monkeypatch.setattr(sys, "argv", ["boxinggym_core", "--output", str(tmp_path / "out"), "--task", "emotion",
                                      "--proposer", "mock", "--cell", "111", "--record-design-criteria"])
    bg.main()
    if not bg.TASKS["emotion"].runnable:
        pytest.skip("emotion is specification-only in this tree; the refusal is not reachable")
    with pytest.raises(ValueError, match="single-component tasks only"):
        run(seen[0])
    assert not (tmp_path / "out").exists()


def test_protocol_u_flags_parse_together(monkeypatch, tmp_path):
    """All unified-protocol flags parse together into the expected namespace values."""
    args = _parse(monkeypatch, tmp_path, "--cell", "111", "--rounds", "30", "--initial-proposals", "10",
                  "--proposal-limit", "3", "--active-maximum", "12", "--initial-catalogue", "off",
                  "--record-design-criteria", "--design-policy", "structure_eig")
    assert (args.rounds, args.initial_proposals, args.proposal_limit, args.active_maximum) == (30, 10, 3, 12)
    assert args.initial_catalogue == "off" and args.record_design_criteria is True
    assert args.design_policy == "structure_eig"


def test_definitions_name_the_core_rule():
    """The recorded definitions match the shared core rule, and point disagreement equals the evidence-weighted variance."""
    assert bg.DESIGN_CRITERIA_DEFINITIONS["point_disagreement_definition"] == positive_inference.POINT_DISAGREEMENT_DEFINITION
    assert bg.DESIGN_CRITERIA_DEFINITIONS["structure_eig_definition"] == bg.EIG_PART_DEFINITION["structure_eig"]
    assert bg.DESIGN_CRITERIA_DEFINITIONS["parameter_eig_definition"] == bg.EIG_PART_DEFINITION["parameter_eig"]
    assert bg.DESIGN_CRITERIA_FILE == "acquisition/tNN_criteria.json"
    # The record's disagreement is the shared core rule, the evidence-weighted variance of the
    # posterior-mean prediction across models: sum_m W_m (m_m(u) - sum_k W_k m_k(u))^2.
    means = np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 0.0], [1.0, 2.0, 6.0]])
    weights = np.array([0.5, 0.3, 0.2])
    centre = weights @ means
    expected = weights @ (means - centre) ** 2
    np.testing.assert_allclose(dp.point_disagreement(means, weights), expected, rtol=0, atol=1e-15)
    assert expected[0] == 0.0 and expected[1] > 0 and expected[2] > 0
