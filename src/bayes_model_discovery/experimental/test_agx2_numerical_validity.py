"""Guard v2 keeps a lognormal particle only where its scale is representable, checked in log
space; the twelve fixtures come from the project's specification.

Guard v1 kept a particle whose mean was finite and > 0 at every point of the objective call.
A positive subnormal mean passes that test, and `conditional_loglik`'s scale
mean * exp(-sigma^2 / 2) then underflows to 0 and `lognorm.logpdf` returns NaN: the round-1
crash of the registered easy s31 cells. Guard v2 also requires, per point,
REPRESENTABILITY_LOWER <= log_scale - 8 sigma and log_scale + 8 sigma <= REPRESENTABILITY_UPPER
with log_scale = log(mean) - 0.5 sigma^2, and mean, sigma, sigma^2 and log_scale finite.

Fixtures 1-11 are synthetic banks in test_agx_guard_units' two-input interface: expression
b0 + b1*x0z with b1 = 0, so each particle's mean is 1.3 * exp(b0) at every point and its log
mean is set exactly. Fixture 12 is the registered easy s31 pool member that crashed, with
hand-set particles at a point just below its divergence (Tz -> 0-). No fit, no provider.
"""

import json
import math
import sys

import numpy as np
import pytest

from . import chembench_core, design_search, positive_inference
from . import test_agx_guard_units as units
from .count_inference import Fit
from .count_models import interface_input_names
from .positive_inference import ScalarHypothesis
from .test_agx_guard_units import ascent_objective  # noqa: F401  (module-scoped fixture)

ROLES = (("b0", "intercept"), ("b1", "signed"))
EXPRESSIONS = ("b0+b1*x0z", "b0+b1*x1z", "b0+b1*x0z*x1z")
POINTS = units.design_points(3)
LOWER, UPPER = design_search.REPRESENTABILITY_LOWER, design_search.REPRESENTABILITY_UPPER
TAIL = design_search.LOGNORMAL_TAIL_SIGMA
#: design_search_constants_sha256 of guard v2 (v1's was e4a7ceb8...).
AGX2_CONSTANTS_SHA256 = "827a55876ae3fb2753bedb335dd15e31ab29fe7d963928057d83f3b959d4a2cf"

#: (log mean, sigma) per fixture, and whether guard v2 keeps the particle. Every mean below is
#: finite and > 0, so guard v1 kept all of them.
PARTICLES = {
    "normal": (0.0, 0.4, True),                        # 1: an ordinary finite positive mean
    "subnormal_mean": (-712.0, 0.4, False),            # 2: mean ~ 1e-309, below float64 tiny
    "scale_underflow": (-744.0, 2.0, False),           # 3: mean * exp(-sigma^2/2) == 0
    "near_tiny_excluded": (-700.0, 1.0, False),        # 4: normal mean and scale, but the
    "near_tiny_kept": (-699.5, 1.0, True),             #    8-sigma lower tail leaves float64
    "near_max_excluded": (709.5, 0.4, False),          # 5: mean ~ 1.3e308 < float64 max, but
    "near_max_kept": (700.0, 1.0, True),               #    the 8-sigma upper tail overflows
    "sigma_squared_overflow": (0.0, 1e155, False),     # 6: sigma^2 = inf
}


def b0_for(log_mean):
    """Return the intercept b0 that gives a particle the requested log mean."""
    return log_mean - math.log(units.diagnostics()["response_unit"])


def lognormal_fit(names, evidence=-2.0, expression=EXPRESSIONS[0]):
    """One lognormal model whose particles are the named PARTICLES entries, in order."""
    hypothesis = ScalarHypothesis(expression, "lognormal", ROLES,
                                  interface_input_names(units.interface()))
    b0 = np.array([b0_for(PARTICLES[name][0]) for name in names])
    particles = {"b0": b0, "b1": np.zeros_like(b0),
                 "sigma": np.array([PARTICLES[name][1] for name in names])}
    return Fit(hypothesis, particles, float(evidence), [float(evidence)] * 2, units.diagnostics())


def v1_valid(result, points=POINTS):
    """Guard v1's per-particle decision: a finite, positive mean at every point."""
    return positive_inference.particle_means_masked(result, points)[1].all(axis=1)


@pytest.mark.parametrize("name", sorted(PARTICLES))
def test_each_fixture_particle_is_kept_exactly_when_its_scale_is_representable(name):
    """Fixtures 1-6: one particle each; guard v1 kept every one, guard v2 keeps the listed."""
    fit = lognormal_fit(["normal", name])
    assert v1_valid(fit).all()
    bank, info = chembench_core._acquisition_bank_for_points([fit], POINTS)
    if PARTICLES[name][2]:
        assert bank == [fit] and bank[0] is fit and info is None
    else:
        assert info == {"excluded_particles": {fit.hypothesis.model_id: {"excluded": 1, "of": 2}},
                        "excluded_models": [], "invalid": False}
        assert np.array_equal(bank[0].particles["b0"], fit.particles["b0"][:1])
        assert np.array_equal(bank[0].particles["sigma"], fit.particles["sigma"][:1])


def test_the_fixture_values_sit_where_their_names_say():
    """The fixtures test what they claim: the underflow particle's formed scale is exactly 0, the
    near-boundary pairs straddle a bound by less than one sigma, and sigma^2 overflows."""
    fit = lognormal_fit(list(PARTICLES))
    mean = positive_inference.particle_means_masked(fit, POINTS)[0]
    sigma = fit.particles["sigma"][:, None]
    with np.errstate(over="ignore", under="ignore"):
        formed = mean * np.exp(-(sigma**2) / 2)
    row = {name: index for index, name in enumerate(PARTICLES)}
    assert np.all(mean > 0) and np.all(np.isfinite(mean))
    assert np.all(mean[row["subnormal_mean"]] < np.finfo(np.float64).tiny)
    assert np.all(formed[row["scale_underflow"]] == 0.0)
    log_scale = {name: PARTICLES[name][0] - 0.5 * PARTICLES[name][1] ** 2
                 for name in ("near_tiny_excluded", "near_tiny_kept", "near_max_excluded",
                              "near_max_kept")}
    assert log_scale["near_tiny_excluded"] - TAIL * 1.0 < LOWER <= (
        log_scale["near_tiny_kept"] - TAIL * 1.0)
    assert np.all(np.isfinite(formed[row["near_tiny_excluded"]]))
    assert np.all(formed[row["near_tiny_excluded"]] >= np.finfo(np.float64).tiny)
    assert log_scale["near_max_excluded"] + TAIL * 0.4 > UPPER >= log_scale["near_max_kept"] + TAIL
    with np.errstate(over="ignore"):
        assert np.isinf(np.float64(PARTICLES["sigma_squared_overflow"][1]) ** 2)


def test_the_check_is_log_space_and_never_forms_the_scale(monkeypatch):
    """The representability check works in log space and never exponentiates to form the scale."""
    fit = lognormal_fit(list(PARTICLES))
    mean = positive_inference.particle_means_masked(fit, POINTS)[0]

    def refused(*args, **kwargs):
        raise AssertionError("guard v2 must not exponentiate: the scale is never formed")

    monkeypatch.setattr(chembench_core.np, "exp", refused)
    keep = chembench_core._lognormal_scale_representable(fit, mean)
    monkeypatch.undo()
    assert keep.shape == (len(PARTICLES), len(POINTS))
    assert keep.all(axis=1).tolist() == [kept for _, _, kept in PARTICLES.values()]


def test_the_unguarded_bank_is_the_crash_and_the_guarded_bank_runs():
    """Fixture 3 end to end: joint_eig on the v1 bank raises the crash's error; on the v2 bank it
    returns finite values; NaN, inf and a zero scale never reach it."""
    fit = lognormal_fit(["normal", "near_tiny_kept", "scale_underflow"])
    assert v1_valid(fit).all()
    with pytest.raises(ValueError, match="invalid conditional likelihood"):
        positive_inference.joint_eig([fit], POINTS, units.SEED, 16)
    bank, info = chembench_core._acquisition_bank_for_points([fit], POINTS)
    assert info["excluded_particles"] == {fit.hypothesis.model_id: {"excluded": 1, "of": 3}}
    table = positive_inference.joint_eig(bank, POINTS, units.SEED, 16)
    assert np.all(np.isfinite(table["eig"])) and np.all(np.isfinite(table["mc_standard_error"]))


def test_some_particles_and_a_whole_model_invalid_leave_the_hand_filtered_bank():
    """Fixtures 7 and 8: a partially invalid model becomes a copy of its kept particles, a model
    with none left is excluded, and a fully valid model stays the same object."""
    valid = lognormal_fit(["normal", "near_max_kept"], -2.5, EXPRESSIONS[0])
    partial = lognormal_fit(["subnormal_mean", "normal", "near_max_excluded", "near_tiny_kept"],
                            -1.5, EXPRESSIONS[1])
    gone = lognormal_fit(["scale_underflow", "sigma_squared_overflow"], -0.5, EXPRESSIONS[2])
    fits = [valid, partial, gone]
    assert all(v1_valid(result).all() for result in fits)
    bank, info = chembench_core._acquisition_bank_for_points(fits, POINTS)
    ids = [result.hypothesis.model_id for result in fits]
    assert [result.hypothesis.model_id for result in bank] == ids[:2]
    assert bank[0] is valid
    assert bank[1].hypothesis is partial.hypothesis and bank[1].diagnostics is partial.diagnostics
    assert bank[1].log_evidence == partial.log_evidence
    for name, values in partial.particles.items():
        assert np.array_equal(bank[1].particles[name], values[[1, 3]])
    assert info == {"excluded_particles": {ids[1]: {"excluded": 2, "of": 4},
                                           ids[2]: {"excluded": 2, "of": 2}},
                    "excluded_models": [ids[2]], "invalid": False}


@pytest.mark.usefixtures("require_autoscilab_source")
def test_a_whole_objective_call_invalid_under_v2_alone_is_explicit_invalid_rows(
        monkeypatch, ascent_objective):  # noqa: F811
    """Fixture 9: every particle has a finite positive mean (v1 would call joint_eig and crash),
    and v2 leaves no model, so the call is the registered invalid rows without joint_eig."""
    fits = [lognormal_fit(["subnormal_mean", "scale_underflow"], -1.0, EXPRESSIONS[0]),
            lognormal_fit(["near_max_excluded"], -2.0, EXPRESSIONS[1])]
    assert all(v1_valid(result).all() for result in fits)
    calls = units.spy_joint_eig(monkeypatch)
    log = []
    rows = units.call(ascent_objective, fits, POINTS, log)
    assert calls == []
    [info] = log
    assert info["invalid"] is True
    assert info["excluded_models"] == [result.hypothesis.model_id for result in fits]
    assert rows == [{"design": point.tolist(), "invalid": True,
                     "invalid_reason": "nonfinite_or_nonpositive_mean"} for point in POINTS]
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize("valid", (True, False))
def test_the_fallback_design_is_rechecked_with_guard_v2(valid, monkeypatch):
    """Fixture 10: the all_designs_invalid fallback checks its one point with the same v2 guard;
    a bank that v1 would accept there but v2 empties makes the round acquisition_failed."""
    fits = [lognormal_fit(["normal"] if valid else ["subnormal_mean", "scale_underflow"])]
    designs = [{"x0": 0.3, "x1": 0.4}, {"x1": 0.25, "x0": 0.55}]
    point = np.array([[designs[1]["x0"], designs[1]["x1"]]])
    assert v1_valid(fits[0], point).all()
    record = {"acquisition_status": "all_designs_invalid"}
    design = chembench_core._acquisition_fallback(record, fits, designs, 1, list(units.NAMES),
                                                  design_search.FALLBACK_POLICY)
    assert record["fallback_valid"] is valid
    if valid:
        assert design is designs[1] and record["acquisition_status"] == "all_designs_invalid"
        assert "fallback_excluded_particles" not in record
    else:
        assert design is None and record["acquisition_status"] == "acquisition_failed"
        model_id = fits[0].hypothesis.model_id
        assert record["fallback_excluded_models"] == [model_id]
        assert record["fallback_excluded_particles"] == {model_id: {"excluded": 2, "of": 2}}


@pytest.mark.parametrize("family", ("lognormal", "gamma"))
def test_a_valid_bank_is_the_same_object_with_no_record_and_no_rng(family, monkeypatch):
    """Fixture 11: guard v2 leaves an all-valid bank exactly as v1 did (the same list object,
    no copy, no RNG), so the objective's joint_eig inputs and outputs are unchanged; gamma
    models never meet the lognormal condition."""
    fits = units.valid_bank(family) + [lognormal_fit(["normal", "near_tiny_kept",
                                                      "near_max_kept"], -2.2, EXPRESSIONS[2])]
    state = json.dumps(np.random.get_state()[1].tolist())
    with monkeypatch.context() as patch:
        units.forbid_rng(patch)
        bank, info = chembench_core._acquisition_bank_for_points(fits, POINTS)
    assert bank is fits and info is None
    assert json.dumps(np.random.get_state()[1].tolist()) == state


def test_the_guard_constants_are_registered_and_restated_by_an_exclusion_record():
    """The guard constants are registered and restated by the exclusion record."""
    constants = design_search.design_search_constants()
    guard = {"acquisition_guard": "particle_conditioning_v2", "lognormal_tail_sigma": 8,
             "representability_lower": -708.3964185322641,
             "representability_upper": 709.782712893384}
    assert {key: constants[key] for key in guard} == guard == (
        design_search.acquisition_guard_constants())
    assert guard["representability_lower"] == math.log(sys.float_info.min)
    assert guard["representability_upper"] == math.log(sys.float_info.max)
    assert sys.float_info.min == np.finfo(np.float64).tiny
    assert design_search.constants_sha256(constants) == AGX2_CONSTANTS_SHA256
    fits = [lognormal_fit(["normal", "subnormal_mean"])]
    _, info = chembench_core._acquisition_bank_for_points(fits, POINTS)
    record = chembench_core._eig_exclusion_record(
        fits, [{"objective_call_index": 0, "point_count": len(POINTS), **info}],
        {"objective_call_count": 1})
    assert record["eig_exclusion_guard"] == guard
    json.dumps(record, allow_nan=False)
