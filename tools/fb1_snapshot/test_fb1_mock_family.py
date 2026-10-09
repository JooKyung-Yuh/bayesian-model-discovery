"""Family-fixing tests: the family check on the mock admission path.

Background: on the LLM path the constraint block carries
`allowed_families=[initial_family]` (built by `fb1_proposer.catalog_constraints`
and `free_constraints`, which the runner calls in `run_gravity.py` with
`family_expansion=ARM[1]=="1"`), and the judge (`judge_candidate` and
`_judge_free_candidate`) refuses any other family. The mock path did not pass
through that judge, so the student_t4 candidate of the pre-declared schedule
(`run_gravity.MOCK_PROPOSALS_V3`) was also admitted on the family-fixed arms
(ARM[1]=="0", cells 100 and 101). This file verifies the correction: the same
rule and the same reason sentence.

Run: <jax venv python> -m pytest -q test_fb1_mock_family.py
No paid calls (the transport path is never invoked).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = str(Path(__file__).resolve().parent)
sys.path.insert(0, _HERE)
# Same safeguard as test_fb1_core_prior.py: keep the run_gravity import away from the
# shared output path (importing alone writes nothing, but the default is the run output root).
os.environ.setdefault("FB1_OUT_DIR", "/tmp/fb1_mock_family_test_out/")

import fb1_proposer as fbp  # noqa: E402
import run_gravity as rg  # noqa: E402

# Two pairs that actually appear in the pre-declared mock schedule (first event of run_gravity.MOCK_PROPOSALS_V3).
IN_FAMILY = ("power", "gauss")
OUT_OF_FAMILY = ("inv_r2", "student_t4")
FIXED_FAMILY_REASON = "family must equal constraints.initial_family when family_expansion is false"
NOT_ALLOWED_REASON = "family must be one of constraints.allowed_families"


@pytest.fixture
def fixed_family(monkeypatch):
    """Module state of the family-fixed arms (ARM[1]=="0", cells 100 and 101).

    ARM is fixed at import time (environment variable), so the test sets the three
    resulting values directly. The import test below checks separately that those
    three values really follow from ARM[1].
    """
    monkeypatch.setattr(rg, "FAMILY_EXPANSION", False)
    monkeypatch.setattr(rg, "INITIAL_FAMILY", "gauss")
    monkeypatch.setattr(rg, "ALLOWED_FAMILIES", ["gauss"])


@pytest.fixture
def expanded_family(monkeypatch):
    """Family-expansion arms (ARM[1]=="1", cells 110 and 111): the setting that produced the existing outputs."""
    monkeypatch.setattr(rg, "FAMILY_EXPANSION", True)
    monkeypatch.setattr(rg, "INITIAL_FAMILY", "gauss")
    monkeypatch.setattr(rg, "ALLOWED_FAMILIES", list(rg.FAMILIES))


def _pool_keys():
    """Return the set of (structure, family) pairs in the initial pool."""
    return {(s, f) for s, f in rg.INITIAL_POOL}


def test_mock_candidate_outside_allowed_families_is_refused_and_recorded(fixed_family):
    """A mock candidate outside the allowed families is refused and the refusal is recorded."""
    ok, record = rg.admission_record(*OUT_OF_FAMILY, _pool_keys(), True)
    assert ok is False
    assert record == {"pair": list(OUT_OF_FAMILY), "admitted": False,
                      "reason": FIXED_FAMILY_REASON}


def test_mock_candidate_inside_allowed_families_is_admitted(fixed_family):
    """A mock candidate inside the allowed families is admitted."""
    ok, record = rg.admission_record(*IN_FAMILY, _pool_keys(), True)
    assert ok is True
    assert record == {"pair": list(IN_FAMILY), "admitted": True, "reason": "ok"}


def test_family_expansion_arm_still_admits_the_second_family(expanded_family):
    # The mock output of cells 110 and 111 must be byte-identical before and after this correction.
    """The family-expansion arm still admits the second family."""
    ok, record = rg.admission_record(*OUT_OF_FAMILY, _pool_keys(), True)
    assert ok is True
    assert record == {"pair": list(OUT_OF_FAMILY), "admitted": True, "reason": "ok"}


def test_llm_path_admission_does_not_re_judge_the_family(fixed_family):
    # An LLM candidate has already passed the family check in the judge (judge_candidate).
    # The admission step does not judge it again: check_family=False.
    """Admission on the LLM path does not judge the family a second time."""
    ok, record = rg.admission_record(*OUT_OF_FAMILY, _pool_keys(), False)
    assert ok is True
    assert record["reason"] == "ok"


def test_duplicate_reason_is_unchanged(fixed_family):
    """The duplicate-refusal reason string is unchanged."""
    ok, record = rg.admission_record("inv_r2", "gauss", _pool_keys(), True)
    assert ok is False
    assert record["reason"] == "duplicate"


def test_refusal_reasons_are_the_judges_own_strings(fixed_family, monkeypatch):
    """Compare against the reason sentences read directly from fb1_proposer (never written separately in two places)."""
    constraints = fbp.catalog_constraints("gauss", family_expansion=False)
    assert constraints["allowed_families"] == ["gauss"]
    expression = constraints["initial_structures"][0]
    declared = constraints["structure_parameters"][expression]
    candidate = {"expression": expression, "family": "student_t4",
                 "parameters": {name: spec["role"] for name, spec in declared.items()}}
    ok, reason = fbp.judge_candidate(candidate, constraints)
    assert ok is False
    assert reason == FIXED_FAMILY_REASON
    assert rg.family_refusal("student_t4") == reason

    # The second sentence (family expansion on, but outside the list) also equals the judge's.
    open_constraints = fbp.catalog_constraints("gauss", family_expansion=True)
    ok2, reason2 = fbp.judge_candidate(dict(candidate, family="cauchy"), open_constraints)
    assert ok2 is False
    assert reason2 == NOT_ALLOWED_REASON
    monkeypatch.setattr(rg, "FAMILY_EXPANSION", True)
    monkeypatch.setattr(rg, "ALLOWED_FAMILIES", list(rg.FAMILIES))
    assert rg.family_refusal("cauchy") == reason2


def _allowed_families_for_arm(arm, tmp_path):
    """Re-import the runner with only FB1_ARM changed and read the three family-fixing globals."""
    code = ("import json, run_gravity as rg; print(json.dumps("
            "[rg.ARM, rg.FAMILY_EXPANSION, rg.INITIAL_FAMILY, rg.ALLOWED_FAMILIES]))")
    env = {**os.environ, "FB1_ARM": arm, "FB1_OUT_DIR": str(tmp_path) + "/",
           "JAX_PLATFORMS": "cpu"}
    proc = subprocess.run([sys.executable, "-B", "-c", code], cwd=_HERE, env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("arm,expansion", [("101", False), ("111", True)])
def test_allowed_families_follow_the_family_bit(arm, expansion, tmp_path):
    """The allowed families follow the family bit of the arm code."""
    got_arm, got_expansion, initial_family, allowed = _allowed_families_for_arm(arm, tmp_path)
    assert got_arm == arm
    assert got_expansion is expansion
    assert initial_family == "gauss"
    assert allowed == (list(rg.FAMILIES) if expansion else ["gauss"])
    # Only the same list as the constraint block the LLM path actually uses makes this the "same rule".
    assert allowed == fbp.catalog_constraints(
        initial_family, family_expansion=expansion)["allowed_families"]
