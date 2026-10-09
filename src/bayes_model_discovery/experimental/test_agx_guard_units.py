"""Particle-guard rules: the ChemBench ascent acquisition's particle guard on synthetic banks.

All-valid banks: the guard returns the very list object passed in and None, uses no RNG, and the
guarded objective's rows are the pre-guard objective's rows exactly, from the same joint_eig call on
the same bank object. Invalid banks: the retained bank is the hand-filtered bank, a model
with no valid particle is excluded, joint_eig's weights are the evidence renormalized over the
retained models, an all-excluded call is explicit invalid rows, and the exclusion record exists
only when something was excluded.

Every bank is synthetic: two declared inputs on [0.1, 1.0], grammar hypotheses with hand-set
particles, positive families. No fit, no provider, no recorded response.
"""

import ast
import importlib.util
import json
import os

import numpy as np
import pytest
from scipy.special import logsumexp

from . import chembench_core, design_search, positive_inference
from . import test_agx_ascent_nonfinite_mean as prep
from . import test_chembench_core as fake
from .belief_packet import task_interface
from .count_inference import Fit
from .count_models import DEFAULT_GRAMMAR_VERSION, interface_input_names, restore_grammar_state
from .positive_inference import RegressionHypothesis, ScalarHypothesis

NAMES = ("x0", "x1")
CENTER, SCALE = 0.55, 0.25
SEED = 20260925
EXPRESSION = "b0+b1*x0z+b2*x1z"
ROLES = (("b0", "intercept"), ("b1", "signed"), ("b2", "signed"))
NOISE = {"lognormal": ("sigma", 0.4), "gamma": ("shape", 3.0)}
#: A slope this steep overflows or underflows the log link wherever |x0z| > 709/800.
STEEP = 800.0


def interface():
    """Return the two-input positive-response task interface used by the synthetic banks."""
    return task_interface("positive", [{"name": name, "range": [0.1, 1.0]} for name in NAMES],
                          {f"{name}z": f"({name}-{CENTER!r})/{SCALE!r}" for name in NAMES},
                          (0.0, None))


def diagnostics():
    """Return the synthetic fit diagnostics shared by every bank."""
    return {"interface": interface(), "response_unit": 1.3, "response_center": 0.0,
            "input_transform": {"center": (CENTER, CENTER), "scale": (SCALE, SCALE)},
            "observation_count": 5, "public_data_sha256": "synthetic-agx-guard"}


def grammar_fit(family, b0, b1, b2, evidence, expression=EXPRESSION):
    """A grammar fit whose declared coefficients are exactly those `expression` uses."""
    used = {node.id for node in ast.walk(ast.parse(expression, mode="eval"))
            if isinstance(node, ast.Name)}
    roles = tuple((name, role) for name, role in ROLES if name in used)
    hypothesis = ScalarHypothesis(expression, family, roles, interface_input_names(interface()))
    values = {"b0": b0, "b1": b1, "b2": b2}
    particles = {name: np.asarray(values[name], dtype=float) for name, _ in roles}
    key, value = NOISE[family]
    particles[key] = np.full(len(particles["b0"]), value)
    return Fit(hypothesis, particles, float(evidence), [float(evidence)] * 2, diagnostics())


def regression_fit(family, b0, b1, evidence):
    """A catalogue regression fit with hand-set particles."""
    hypothesis = RegressionHypothesis(family, (0,))
    key, value = NOISE[family]
    particles = {"b0": np.asarray(b0, dtype=float), "b1": np.asarray(b1, dtype=float)}
    particles[key] = np.full(len(particles["b0"]), value)
    return Fit(hypothesis, particles, float(evidence), [float(evidence)] * 2, diagnostics())


def valid_bank(family):
    """Four grammar models and one catalogue model, every particle modest: all-valid."""
    rng = np.random.default_rng(7)
    fits = [grammar_fit(family, rng.normal(0, 0.3, 12), rng.normal(0, 0.4, 12),
                        rng.normal(0, 0.4, 12), evidence, expression)
            for evidence, expression in ((-3.1, EXPRESSION), (-2.4, "b0+b1*x0z"),
                                         (-4.0, "b0+b1*x1z+b2*x0z*x1z"),
                                         (-2.9, "b0+b1*square(x0z)+b2*x1"))]
    fits.append(regression_fit(family, rng.normal(0, 0.3, 12), rng.normal(0, 0.4, 12), -3.3))
    assert len({result.hypothesis.model_id for result in fits}) == len(fits)
    return fits


def design_points(count):
    """Return `count` reproducible random design points in the input box."""
    return np.random.default_rng(100 + count).uniform(0.1, 1.0, size=(count, len(NAMES)))


def old_objective_rows(fits, points, seed, eig_samples):
    """The pre-guard ascent objective's joint_eig branch, verbatim (chembench_core.py:2068-2071)."""
    table = positive_inference.joint_eig(fits, points, seed, eig_samples)
    return [{"design": point.tolist(), "eig": float(table["eig"][i]),
             "mc_standard_error": float(table["mc_standard_error"][i])}
            for i, point in enumerate(points)]


class StopAtAscent(Exception):
    """Raised in place of projected_spsa once the ascent objective has been captured."""


@pytest.fixture(scope="module")
def ascent_objective(tmp_path_factory):
    """chembench_core.run's own ascent objective closure (cell 001, the fake world), taken at
    round 1's projected_spsa call. Its bank, call log and ordinal are keyword defaults, so each
    test calls it on a synthetic bank with its own log. Its eig_samples is the dev profile's."""
    # The per-test `require_autoscilab_source` checks the pinned source; a module fixture
    # cannot use it, so it only skips when the source is absent altogether.
    if not os.environ.get("BMD_AUTOSCILAB") or importlib.util.find_spec("autoscilab") is None:
        pytest.skip("autoscilab requires BMD_AUTOSCILAB and importable source")
    captured = {}

    def capture(objective, *args, **kwargs):
        captured["objective"] = objective
        raise StopAtAscent

    base = tmp_path_factory.mktemp("objective")
    restore_grammar_state((DEFAULT_GRAMMAR_VERSION, {}))
    with pytest.MonkeyPatch.context() as patch:
        prep.install_world(patch)
        patch.setattr(design_search, "projected_spsa", capture)
        with pytest.raises(StopAtAscent):
            chembench_core.run(fake._args(
                base / "cell", fake._manifest(base), cell="001", seed=31, rounds=1,
                particles=16, q=8, structure_proposals=None, proposer="mock",
                design_policy="joint_eig", design_search="ascent"))
    restore_grammar_state((DEFAULT_GRAMMAR_VERSION, {}))
    return captured["objective"]


def call(objective, fits, points, log):
    """Call the ascent objective on a synthetic bank, logging exclusions into `log`."""
    return objective(points, SEED, fit_bank=fits, exclusion_calls=log, objective_calls=[0])


def spy_joint_eig(patch):
    """Patch joint_eig to record the banks it receives; return that list."""
    calls = []
    real = positive_inference.joint_eig

    def recorded(fits, inputs, seed, outer_samples=64):
        calls.append(fits)
        return real(fits, inputs, seed, outer_samples)

    patch.setattr(chembench_core, "joint_eig", recorded)
    return calls


def forbid_rng(patch):
    """Patch numpy so that creating a generator raises."""
    def refused(*args, **kwargs):
        raise AssertionError("the acquisition guard must not create a generator")

    patch.setattr(np.random, "default_rng", refused)
    patch.setattr(np.random, "Generator", refused)


# ---------------------------------------------------------------------------
# T2: all-valid banks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ("lognormal", "gamma"))
@pytest.mark.parametrize("count", (1, 2, 2 * design_search.RESTARTS))
def test_all_valid_bank_is_the_same_object_with_no_record_and_no_rng(family, count, monkeypatch):
    """An all-valid bank comes back as the same object with no record and no RNG use."""
    fits = valid_bank(family)
    points = design_points(count)
    before = [{name: values.copy() for name, values in result.particles.items()} for result in fits]
    state = json.dumps(np.random.get_state()[1].tolist())
    with monkeypatch.context() as patch:
        forbid_rng(patch)
        bank, info = chembench_core._acquisition_bank_for_points(fits, points)
    assert bank is fits and info is None
    assert all(a is b for a, b in zip(bank, fits, strict=True))
    assert json.dumps(np.random.get_state()[1].tolist()) == state
    for result, particles in zip(fits, before, strict=True):
        assert all(np.array_equal(result.particles[name], values)
                   for name, values in particles.items())


@pytest.mark.usefixtures("require_autoscilab_source")
@pytest.mark.parametrize("family", ("lognormal", "gamma"))
@pytest.mark.parametrize("count", (1, 2, 2 * design_search.RESTARTS))
def test_all_valid_rows_equal_the_pre_agx_objective_from_the_same_bank_object(
        family, count, monkeypatch, ascent_objective):
    """With an all-valid bank the guarded objective's rows equal the pre-guard rows exactly."""
    fits = valid_bank(family)
    points = design_points(count)
    samples = chembench_core.PROFILE_GRID["dev"]["eig_samples"]
    expected = old_objective_rows(fits, points, SEED, samples)
    calls = spy_joint_eig(monkeypatch)
    log = []
    rows = call(ascent_objective, fits, points, log)
    assert log == []
    assert len(calls) == 1 and calls[0] is fits
    assert [list(row) for row in rows] == [list(row) for row in expected] == [
        ["design", "eig", "mc_standard_error"]] * count
    assert [row["design"] for row in rows] == [row["design"] for row in expected]
    for key in ("eig", "mc_standard_error"):
        assert np.array_equal(np.array([row[key] for row in rows]),
                              np.array([row[key] for row in expected]))
    assert json.dumps(rows) == json.dumps(expected)


# ---------------------------------------------------------------------------
# T3: invalid banks
# ---------------------------------------------------------------------------

def mixed_bank(family):
    """Model 0: all valid. Model 1: particles 1 and 3 steep in x0z. Model 2: every particle
    steep. Model 3: particle 0 steep. The points below reach x0z = +/-1.4, where a steep
    particle's mean is inf or 0, and a modest one's is finite and positive."""
    modest = [0.1, -0.2, 0.3, 0.05, -0.1]
    fits = [
        grammar_fit(family, modest, [0.2, -0.3, 0.1, 0.4, -0.2], [0.1] * 5, -2.0),
        grammar_fit(family, modest, [0.2, STEEP, 0.1, -STEEP, -0.2], [0.1] * 5, -1.5,
                    "b0+b1*x0z+b2*square(x1z)"),
        grammar_fit(family, modest, [STEEP] * 5, [0.1] * 5, -0.5, "b0+b1*x0z"),
        grammar_fit(family, modest, [-STEEP, 0.3, 0.1, 0.4, -0.2], [0.1] * 5, -3.0,
                    "b0+b1*x0z+b2*x1"),
    ]
    assert len({result.hypothesis.model_id for result in fits}) == len(fits)
    return fits


#: x0 = 0.2 and 0.9 give x0z = -1.4 and +1.4, where a steep slope's |eta| is about 1120: exp
#: overflows to inf one way and underflows to 0 the other, so both steep signs are invalid.
MIXED_POINTS = np.array([[0.2, 0.5], [0.9, 0.3], [0.55, 0.7]])
#: x0 = 0.775 gives x0z = 0.9 and eta = +/-720: +800 overflows (inf), while -800 leaves a
#: positive subnormal mean (exp(-720) ~ 1e-313), which is finite and > 0, so valid under the
#: mean rule. Guard v2 also excludes it from a lognormal model (its scale is not
#: representable), so the tests of the mean rule below use gamma, which v2 leaves unchanged.
EDGE_POINT = np.array([[0.775, 0.5]])


@pytest.mark.parametrize("family", ("lognormal", "gamma"))
def test_retained_bank_is_the_hand_filtered_bank_and_a_fully_invalid_model_is_excluded(
        family, monkeypatch):
    """The retained bank equals the hand-filtered bank, and a model with no valid particle is excluded."""
    fits = mixed_bank(family)
    with monkeypatch.context() as patch:
        forbid_rng(patch)
        bank, info = chembench_core._acquisition_bank_for_points(fits, MIXED_POINTS)
    kept = {0: np.arange(5), 1: np.array([0, 2, 4]), 3: np.array([1, 2, 3, 4])}
    assert [result.hypothesis.model_id for result in bank] == [
        fits[index].hypothesis.model_id for index in kept]
    assert bank[0] is fits[0]
    for result, (index, indices) in zip(bank, kept.items(), strict=True):
        source = fits[index]
        assert result.hypothesis is source.hypothesis
        assert result.log_evidence == source.log_evidence
        assert result.per_chain_log_evidence is source.per_chain_log_evidence
        assert result.diagnostics is source.diagnostics
        assert list(result.particles) == list(source.particles)
        for name, values in source.particles.items():
            assert result.particles[name].dtype == values.dtype
            assert np.array_equal(result.particles[name], values[indices])
    ids = [result.hypothesis.model_id for result in fits]
    assert info == {
        "excluded_particles": {ids[1]: {"excluded": 2, "of": 5}, ids[2]: {"excluded": 5, "of": 5},
                               ids[3]: {"excluded": 1, "of": 5}},
        "excluded_models": [ids[2]], "invalid": False}
    # The caller's fits are untouched.
    assert np.array_equal(fits[1].particles["b1"], [0.2, STEEP, 0.1, -STEEP, -0.2])


def test_mask_is_per_particle_over_every_point_of_the_call(monkeypatch):
    """The validity mask is per particle over every point of the call."""
    fits = mixed_bank("gamma")
    # At the centre point alone every particle is valid, so the same bank comes back.
    centre = MIXED_POINTS[2:]
    bank, info = chembench_core._acquisition_bank_for_points(fits, centre)
    assert bank is fits and info is None
    # At the edge point only the +800 slopes are invalid: model 1's particle 1 and all of model
    # 2; model 1's particle 3 and model 3's particle 0 (-800) keep a positive subnormal mean.
    _, info = chembench_core._acquisition_bank_for_points(fits, EDGE_POINT)
    ids = [result.hypothesis.model_id for result in fits]
    assert info == {"excluded_particles": {ids[1]: {"excluded": 1, "of": 5},
                                           ids[2]: {"excluded": 5, "of": 5}},
                    "excluded_models": [ids[2]], "invalid": False}
    # The union over the call's points: adding the edge point to the centre changes nothing
    # else, and adding x0z = -1.4 also excludes model 1's particle 3 and model 3's particle 0.
    _, both = chembench_core._acquisition_bank_for_points(fits, np.vstack((centre, EDGE_POINT)))
    assert both == info
    _, union = chembench_core._acquisition_bank_for_points(
        fits, np.vstack((EDGE_POINT, MIXED_POINTS[:1])))
    assert union["excluded_particles"] == {ids[1]: {"excluded": 2, "of": 5},
                                           ids[2]: {"excluded": 5, "of": 5},
                                           ids[3]: {"excluded": 1, "of": 5}}
    # Guard v2: in the lognormal bank the edge point alone already excludes the -800 slopes'
    # positive subnormal means too (guard v2).
    lognormal = mixed_bank("lognormal")
    ids = [result.hypothesis.model_id for result in lognormal]
    _, edge = chembench_core._acquisition_bank_for_points(lognormal, EDGE_POINT)
    assert edge == {"excluded_particles": {ids[1]: {"excluded": 2, "of": 5},
                                           ids[2]: {"excluded": 5, "of": 5},
                                           ids[3]: {"excluded": 1, "of": 5}},
                    "excluded_models": [ids[2]], "invalid": False}


@pytest.mark.usefixtures("require_autoscilab_source")
@pytest.mark.parametrize("family", ("lognormal", "gamma"))
def test_joint_eig_weights_are_the_evidence_renormalized_over_the_retained_models(
        family, monkeypatch, ascent_objective):
    """joint_eig's weights are the evidence renormalized over the retained models."""
    fits = mixed_bank(family)
    samples = chembench_core.PROFILE_GRID["dev"]["eig_samples"]
    seen = []
    real_validate = positive_inference.validate_pool

    def recorded(pool):
        weights = real_validate(pool)
        seen.append(([result.hypothesis.model_id for result in pool], weights))
        return weights

    calls = spy_joint_eig(monkeypatch)
    monkeypatch.setattr(positive_inference, "validate_pool", recorded)
    log = []
    rows = call(ascent_objective, fits, MIXED_POINTS, log)
    [entry] = log
    assert entry["objective_call_index"] == 0 and entry["point_count"] == len(MIXED_POINTS)
    assert entry["invalid"] is False
    bank = calls[0]
    assert [result.hypothesis.model_id for result in bank] == [
        fits[i].hypothesis.model_id for i in (0, 1, 3)]
    logz = np.array([fits[i].log_evidence for i in (0, 1, 3)])
    expected = logz - logsumexp(logz)
    ids, weights = seen[0]
    assert ids == [result.hypothesis.model_id for result in bank]
    assert np.array_equal(weights, expected)
    assert np.array_equal(np.exp(weights), np.exp(logz - logsumexp(logz)))
    # The rows are joint_eig on the hand-filtered bank, bit for bit.
    hand = [fits[0]] + [Fit(fits[i].hypothesis, {name: values[keep] for name, values
                                                 in fits[i].particles.items()},
                            fits[i].log_evidence, fits[i].per_chain_log_evidence,
                            fits[i].diagnostics)
                        for i, keep in ((1, [0, 2, 4]), (3, [1, 2, 3, 4]))]
    monkeypatch.setattr(positive_inference, "validate_pool", real_validate)
    assert json.dumps(rows) == json.dumps(old_objective_rows(hand, MIXED_POINTS, SEED, samples))
    assert all(np.isfinite(row["eig"]) for row in rows)


@pytest.mark.usefixtures("require_autoscilab_source")
def test_every_model_excluded_is_explicit_invalid_rows_without_joint_eig(
        monkeypatch, ascent_objective):
    """When every model is excluded the call returns explicit invalid rows without calling joint_eig."""
    fits = [mixed_bank("lognormal")[2],
            grammar_fit("lognormal", [0.0] * 3, [-STEEP] * 3, [0.0] * 3, -1.0)]
    calls = spy_joint_eig(monkeypatch)
    log = []
    rows = call(ascent_objective, fits, MIXED_POINTS[:2], log)
    assert calls == []
    [info] = log
    assert info["invalid"] is True
    assert info["excluded_models"] == [result.hypothesis.model_id for result in fits]
    assert rows == [{"design": point.tolist(), "invalid": True,
                     "invalid_reason": "nonfinite_or_nonpositive_mean"}
                    for point in MIXED_POINTS[:2]]
    assert not any({"eig", "mc_standard_error"} & set(row) for row in rows)
    json.dumps(rows, allow_nan=False)
    # The row vocabulary is design_search's registered one.
    assert chembench_core.ACQUISITION_INVALID_REASON in design_search.INVALID_REASONS


def test_exclusion_record_exists_only_for_exclusions_and_sums_the_call_log():
    """The exclusion record exists only when something was excluded and sums the call log."""
    fits = mixed_bank("gamma")
    ids = [result.hypothesis.model_id for result in fits]
    calls = []
    # Call 1 (all three points): model 1 loses 2, model 2 all 5, model 3 one. Call 2 (the edge
    # point): model 1 loses 1 and model 2 all 5. Calls 0 and 3 (the centre) exclude nothing.
    for call_index, points in enumerate((MIXED_POINTS[2:], MIXED_POINTS, EDGE_POINT,
                                         MIXED_POINTS[2:])):
        _, info = chembench_core._acquisition_bank_for_points(fits, points)
        if info is not None:
            calls.append({"objective_call_index": call_index, "point_count": len(points), **info})
    assert [call["objective_call_index"] for call in calls] == [1, 2]
    record = {"objective_call_count": 4}
    keys = chembench_core._eig_exclusion_record(fits, calls, record)
    assert list(keys) == ["acquisition_status", "eig_excluded_particles", "eig_excluded_models",
                          "eig_exclusion_calls", "eig_exclusion_summary", "eig_exclusion_guard"]
    # Guard v2: the record restates the guard and its registered constants.
    assert keys["eig_exclusion_guard"] == design_search.acquisition_guard_constants()
    assert keys["acquisition_status"] == "ascent_with_exclusions"
    assert keys["eig_excluded_particles"] == {
        ids[1]: {"of": 5, "calls": 2, "excluded_total": 3, "excluded_max": 2},
        ids[2]: {"of": 5, "calls": 2, "excluded_total": 10, "excluded_max": 5},
        ids[3]: {"of": 5, "calls": 1, "excluded_total": 1, "excluded_max": 1}}
    assert list(keys["eig_excluded_particles"]) == [ids[1], ids[2], ids[3]]
    assert keys["eig_excluded_models"] == {ids[2]: 2}
    assert keys["eig_exclusion_calls"] is calls
    assert keys["eig_exclusion_summary"] == {
        "objective_call_count": 4, "calls_with_excluded_particles": 2,
        "invalid_objective_call_count": 0, "excluded_particle_count": 14,
        "excluded_model_count": 2,
        "retained_model_weighting": "model_logweights_over_retained_models"}
    # design_search's own status wins when it reported one.
    assert chembench_core._eig_exclusion_record(
        fits, calls, {**record, "acquisition_status": "all_designs_invalid"})[
        "acquisition_status"] == "all_designs_invalid"
    json.dumps(keys, allow_nan=False)


@pytest.mark.parametrize("valid", (True, False))
def test_fallback_checks_its_one_point_once_and_marks_failure(valid, monkeypatch):
    # Valid: models 0 and 1 at x0z = 0, where every particle is valid. Invalid: model 2 alone
    # (every particle steep) at x0z = 1.4. The design dict's own key order is not names order.
    """The fallback checks its one point once and marks failure when the bank is empty."""
    fits = mixed_bank("lognormal")[:2] if valid else [mixed_bank("lognormal")[2]]
    designs = [{"x0": 0.3, "x1": 0.4}, {"x1": 0.25, "x0": 0.55 if valid else 0.9}]
    calls = []
    real = chembench_core._acquisition_bank_for_points

    def counted(bank, points):
        calls.append(points.copy())
        return real(bank, points)

    monkeypatch.setattr(chembench_core, "_acquisition_bank_for_points", counted)
    record = {"acquisition_status": "all_designs_invalid"}
    design = chembench_core._acquisition_fallback(record, fits, designs, 1, list(NAMES),
                                                  design_search.FALLBACK_POLICY)
    assert len(calls) == 1 and calls[0].tolist() == [[designs[1]["x0"], designs[1]["x1"]]]
    assert record["fallback_policy"] == "fixed_random_designs_from_native_input_distribution"
    assert record["fallback_design_index"] == 1
    assert record["fallback_design"] == [designs[1]["x0"], designs[1]["x1"]]
    assert record["fallback_valid"] is valid
    if valid:
        assert design is designs[1]
        assert record["acquisition_status"] == "all_designs_invalid"
        assert "fallback_excluded_particles" not in record
    else:
        assert design is None
        assert record["acquisition_status"] == "acquisition_failed"
        assert record["fallback_excluded_models"] == [fits[0].hypothesis.model_id]
        assert record["fallback_excluded_particles"] == {
            fits[0].hypothesis.model_id: {"excluded": 5, "of": 5}}
