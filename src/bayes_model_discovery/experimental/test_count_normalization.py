"""A predictive row that misses the normalization tolerance.

Before the rule under test a single PMF row whose mass summed to 1 only to within a few times
1e-10 ended the whole run with "likelihood normalization failed". The saved fit
replayed in section 4 is exactly that: ONE particle of a beta-binomial model, at
kappa ~ 1.8e5, drifting by at most 6.02e-10 - a thousandfold below the loss a
defective model produces, and with no NaN cell anywhere.

The rule separates the two by the SIZE of the loss:

  dev <= 1e-10   nothing happens, the array is the one the earlier code returned
  dev <= 1e-6    floating-point drift: the row is rescaled by its own sum
  otherwise      the particle lost real mass: it is given zero weight

and counts every event, so neither branch is ever silent. A row with a NaN cell
that still fails to normalize keeps the earlier refusal untouched
(`test_count_nan_robustness` pins that path and is not weakened here).
"""

import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit, logsumexp
from scipy.stats import betabinom, binom

from . import count_epig, count_models, design_policies
from .count_inference import Fit, choose, predict
from .count_models import (
    DRIFT_TOL,
    NORMALIZATION_ATOL,
    Hypothesis,
    nan_substitution_counter,
    normalization_event_counter,
    particle_logpmf,
)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "fe0_t18"
# One intercept, one dispersion: every particle's p and kappa are set directly,
# so each row below sits on exactly one side of exactly one threshold.
FLAT = Hypothesis("b0", "beta_binomial", (("b0", "intercept"),))
DESIGNS = np.array([0.25, 0.75])
# rho -> kappa = 1/rho - 1. The deviations are scipy's own, measured in section
# 1's first test, not asserted from a formula: (a) an ordinary dispersion and
# (b) a tiny kappa normalize exactly; (c) kappa = 1e8 drifts by ~4e-7, inside
# DRIFT_TOL; (d) kappa = 1e13 loses ~6% of the mass, far outside it.
EXACT_RHO, TINY_KAPPA_RHO, DRIFTING_RHO, DEFECTIVE_RHO = 0.5, 0.999, 1e-8, 1e-13


def logit(p):
    """Return the logit log(p / (1 - p)) of a probability as a float."""
    return float(np.log(p) - np.log1p(-p))


def particles_for(rows):
    """One particle per (p, rho) pair, in the given order."""
    return {
        "b0": np.array([logit(p) for p, _ in rows]),
        "rho": np.array([rho for _, rho in rows]),
    }


def raw_rows(particles, designs=DESIGNS, n_trials=50):
    """The log pmf table BEFORE any normalization rule - scipy's own bytes."""
    p = expit(particles["b0"])[:, None, None] + 0 * np.asarray(designs)[None, :, None]
    kappa = (1 / particles["rho"] - 1)[:, None, None]
    counts = np.arange(n_trials + 1)[None, None, :]
    return betabinom.logpmf(counts, n_trials, p * kappa, (1 - p) * kappa)


def deviations(rows):
    """Absolute deviation from 1 of each row's total probability mass."""
    return np.abs(np.exp(rows).sum(axis=-1) - 1.0)


# --------------------------------------------------------------------------
# 1. the three branches of the rule
# --------------------------------------------------------------------------


def test_the_synthetic_particles_really_do_straddle_the_two_thresholds():
    """The fixture's premise, measured rather than assumed.

    Every later assertion is about which branch a row takes, so the deviations
    that decide it are read off scipy here, once, before the rule is involved.
    """
    assert (NORMALIZATION_ATOL, DRIFT_TOL) == (1e-10, 1e-6)
    measured = deviations(raw_rows(particles_for([
        (0.5, EXACT_RHO), (0.5, TINY_KAPPA_RHO), (0.9, DRIFTING_RHO), (0.1, DEFECTIVE_RHO),
    ])))[:, 0]
    assert measured[0] <= NORMALIZATION_ATOL
    assert measured[1] <= NORMALIZATION_ATOL
    assert NORMALIZATION_ATOL < measured[2] <= DRIFT_TOL
    assert measured[2] == pytest.approx(4.0e-7, rel=0.1)
    assert measured[3] > DRIFT_TOL
    assert measured[3] == pytest.approx(6.4e-2, rel=0.1)
    assert not np.isnan(raw_rows(particles_for([(0.1, DEFECTIVE_RHO)]))).any()


def test_a_row_inside_the_tolerance_is_returned_exactly_as_before():
    """Rows within the tolerance pass through bit-identically and record no event."""
    particles = particles_for([(0.5, EXACT_RHO), (0.5, TINY_KAPPA_RHO)])
    before = normalization_event_counter()
    rows = particle_logpmf(FLAT, particles, DESIGNS)
    after = normalization_event_counter()
    assert after == before
    # bit-identical to the untouched scipy table: the rule did not run at all
    np.testing.assert_array_equal(rows, raw_rows(particles))


def test_floating_point_drift_is_renormalized_in_place_and_counted():
    """Drifting rows are rescaled by their own sum and counted as renormalized."""
    particles = particles_for([(0.5, EXACT_RHO), (0.9, DRIFTING_RHO)])
    raw = raw_rows(particles)
    before = normalization_event_counter()
    rows = particle_logpmf(FLAT, particles, DESIGNS)
    after = normalization_event_counter()
    # one row per (particle, input): the drifting particle at both designs
    assert after["renormalized_rows"] - before["renormalized_rows"] == len(DESIGNS)
    assert after["zeroed_rows"] - before["zeroed_rows"] == 0
    assert after["calls"] - before["calls"] == 1
    assert after["max_deviation"] >= deviations(raw)[1].max()
    # the row is its own sum divided out - nothing else moved
    np.testing.assert_array_equal(rows[0], raw[0])
    np.testing.assert_allclose(
        rows[1], raw[1] - np.log(np.exp(raw[1]).sum(axis=-1))[:, None], rtol=0, atol=0
    )
    assert np.all(deviations(rows) <= NORMALIZATION_ATOL)


def test_a_particle_that_lost_real_mass_is_given_zero_weight_and_counted():
    """A particle that lost real mass is zeroed at every input and counted."""
    particles = particles_for([(0.5, EXACT_RHO), (0.1, DEFECTIVE_RHO)])
    raw = raw_rows(particles)
    before = normalization_event_counter()
    rows = particle_logpmf(FLAT, particles, DESIGNS)
    after = normalization_event_counter()
    # the whole particle leaves, not one of its two inputs: a particle that lost
    # mass at one design is not trusted at the other
    assert after["zeroed_rows"] - before["zeroed_rows"] == 1
    assert after["renormalized_rows"] - before["renormalized_rows"] == 0
    assert after["calls"] - before["calls"] == 1
    assert after["max_deviation"] >= deviations(raw)[1].max()
    assert np.all(np.isneginf(rows[1]))
    np.testing.assert_array_equal(rows[0], raw[0])
    assert count_models.zeroed_particles(rows).tolist() == [False, True]
    np.testing.assert_array_equal(count_models.active_particle_rows(rows), rows[:1])


def test_drift_and_defect_in_one_call_take_their_own_branches():
    """Drift and defect in a single call are handled by their separate branches."""
    particles = particles_for([
        (0.5, EXACT_RHO), (0.9, DRIFTING_RHO), (0.1, DEFECTIVE_RHO), (0.5, TINY_KAPPA_RHO),
    ])
    before = normalization_event_counter()
    rows = particle_logpmf(FLAT, particles, DESIGNS)
    after = normalization_event_counter()
    assert after["renormalized_rows"] - before["renormalized_rows"] == len(DESIGNS)
    assert after["zeroed_rows"] - before["zeroed_rows"] == 1
    assert after["calls"] - before["calls"] == 1
    np.testing.assert_array_equal(np.isneginf(rows).all(axis=(1, 2)), [False, False, True, False])
    np.testing.assert_array_equal(rows[[0, 3]], raw_rows(particles)[[0, 3]])
    assert np.all(deviations(rows[[0, 1, 3]]) <= NORMALIZATION_ATOL)


def test_every_particle_zeroed_keeps_the_old_refusal_and_its_message():
    """A model with no particle left has no predictive at all (by design)."""
    particles = particles_for([(0.1, DEFECTIVE_RHO), (0.5, 1e-15)])
    before = normalization_event_counter()
    with pytest.raises(ValueError, match="likelihood normalization failed"):
        particle_logpmf(FLAT, particles, DESIGNS)
    after = normalization_event_counter()
    # counted even though it refused: the events happened
    assert after["zeroed_rows"] - before["zeroed_rows"] == 2
    assert count_models.NORMALIZATION_FAILURE == "likelihood normalization failed"


def test_active_particle_rows_refuses_when_no_particle_survives():
    """Selecting active particles raises when every particle has been zeroed."""
    zeroed = np.full((3, 2, 4), -np.inf)
    with pytest.raises(ValueError, match="likelihood normalization failed"):
        count_models.active_particle_rows(zeroed)


def test_a_nan_cell_that_breaks_normalization_still_refuses_as_invalid_likelihood(monkeypatch):
    """The earlier rule is untouched: a NaN that carried mass is still a refusal.

    Not the drift branch and not the zero-weight branch - the same message, so
    `core_count.PRIOR_PROBE_NONFINITE` classifies it exactly as before, and the
    normalization counter records nothing at all.
    """
    particles = particles_for([(0.5, EXACT_RHO)])
    original = betabinom.logpmf

    def poisoned(*args, **kwargs):
        """Wrap betabinom.logpmf and set the modal-count cell to NaN."""
        value = np.array(original(*args, **kwargs), dtype=float)
        if value.ndim == 3:
            value[:, :, 25] = np.nan  # p = 0.5, kappa = 1: the modal count
        return value

    monkeypatch.setattr(betabinom, "logpmf", poisoned)
    before, nan_before = normalization_event_counter(), nan_substitution_counter()
    with pytest.raises(ValueError, match="invalid likelihood"):
        particle_logpmf(FLAT, particles, DESIGNS)
    assert normalization_event_counter() == before
    assert nan_substitution_counter()["cells"] - nan_before["cells"] == len(DESIGNS)


def test_an_ordinary_binomial_table_is_untouched_by_the_rule():
    """The path every death/binomial run takes: no deviation, no event, no copy."""
    hypothesis = Hypothesis("b0+b1*u", "binomial", (("b0", "intercept"), ("b1", "signed")))
    particles = {"b0": np.array([-1.0, 0.0, 2.0]), "b1": np.array([0.5, -1.5, 0.25])}
    before = normalization_event_counter()
    rows = particle_logpmf(hypothesis, particles, DESIGNS)
    assert normalization_event_counter() == before
    expected = binom.logpmf(
        np.arange(51)[None, None, :],
        50,
        expit(particles["b0"][:, None] + particles["b1"][:, None] * DESIGNS[None, :])[:, :, None],
    )
    np.testing.assert_array_equal(rows, expected)


# --------------------------------------------------------------------------
# 2. the consumers: a zeroed particle is out of the average and out of the bank
# --------------------------------------------------------------------------


def fits_with(rows, log_evidence=-10.0):
    """Build a one-model fit list from (p, rho) particle rows."""
    return [Fit(FLAT, particles_for(rows), log_evidence)]


def test_predict_averages_over_the_particles_that_kept_weight():
    """The zeroed particle is out of the sum AND out of the divisor.

    Checked against the predictive of the surviving particles alone, not
    against a reimplementation of the average.
    """
    kept = [(0.5, EXACT_RHO), (0.5, TINY_KAPPA_RHO)]
    with_defect = predict(fits_with([*kept, (0.1, DEFECTIVE_RHO)]), DESIGNS)
    without = predict(fits_with(kept), DESIGNS)
    np.testing.assert_array_equal(with_defect["bma_logpmf"], without["bma_logpmf"])
    np.testing.assert_allclose(np.exp(with_defect["bma_logpmf"]).sum(-1), 1.0, atol=1e-10, rtol=0)


def test_predict_is_unchanged_when_no_row_fails():
    """Without any failing row the predictive equals the plain particle mixture."""
    rows = [(0.5, EXACT_RHO), (0.5, TINY_KAPPA_RHO)]
    table = particle_logpmf(FLAT, particles_for(rows), DESIGNS)
    np.testing.assert_array_equal(
        predict(fits_with(rows), DESIGNS)["bma_logpmf"],
        logsumexp(table, axis=0) - np.log(len(table)),
    )


def test_the_eig_bank_drops_the_zeroed_row_before_the_bank_is_validated():
    """An all-zero PMF row is an "invalid component PMF bank"; it never gets there."""
    fits = fits_with([(0.5, EXACT_RHO), (0.5, TINY_KAPPA_RHO), (0.1, DEFECTIVE_RHO)])
    chosen, rows = choose(fits, DESIGNS, seed=0, posterior_samples=3)
    assert chosen in [float(u) for u in DESIGNS]
    assert all(np.isfinite(row["eig"]) for row in rows)
    # design_policies rebuilds the SAME bank from the same seed, so it must drop
    # the same particle - test_design_policies pins the two against each other.
    banks = design_policies._banks(fits, DESIGNS, 0, 3, None, 50)
    assert [len(bank) for bank in banks] == [2]
    np.testing.assert_allclose(banks[0].sum(-1), 1.0, atol=1e-10, rtol=0)


def test_the_epig_banks_drop_the_particle_from_the_targets_and_the_designs_together():
    """Y_q and Y_x share one particle, so one exclusion covers both banks."""
    fits = fits_with([(0.5, EXACT_RHO), (0.5, TINY_KAPPA_RHO), (0.1, DEFECTIVE_RHO)])
    answer = count_epig.from_fits(fits, DESIGNS, quadrature_count=4)
    assert answer["particle_inventory"][0]["indices"] == [0, 1]
    assert np.all(np.isfinite(answer["epig"]))


# --------------------------------------------------------------------------
# 3. the checkpoint record
# --------------------------------------------------------------------------


def adapter():
    """Import the BoxingGym adapter module, skipping the test when it is unavailable."""
    return pytest.importorskip(
        "bayes_model_discovery.experimental.boxinggym_core",
        reason="adapter imports the read-only boxing_gym package lazily",
    )


def test_no_event_writes_no_key_at_all():
    """Identical before/after counters produce no normalization record."""
    bg = adapter()
    counter = normalization_event_counter()
    assert bg.predictive_normalization_record(counter, counter) is None


def test_an_event_is_recorded_with_the_tolerance_that_judged_it():
    """A counter difference is recorded together with the drift tolerance."""
    bg = adapter()
    before = {"renormalized_rows": 4, "max_deviation": 1e-12, "zeroed_rows": 1, "calls": 2}
    after = {"renormalized_rows": 9, "max_deviation": 6.02e-10, "zeroed_rows": 3, "calls": 4}
    assert bg.predictive_normalization_record(before, after) == {
        "renormalized_rows": 5,
        "max_deviation_cumulative": 6.02e-10,
        "zeroed_rows": 2,
        "drift_tolerance": 1e-6,
    }


def bg_run_args(bg, output, **overrides):
    """The argparse namespace `boxinggym_core.run` expects, with default option values."""
    return SimpleNamespace(**{
        "output": str(output), "task": "death", "profile": "dev", "proposer": "mock",
        "llm_model": None, "allow_paid_api": False, "rounds": 1, "seed": 11, "cell": "111",
        "prior_template": bg.DEFAULT_PRIOR_TEMPLATE, "priors": "off", "initial_family": None,
        "truth_in_pool": False, "inject_oracle_at": None, "prompt_variant": None, **overrides,
    })


def test_a_run_whose_rows_all_normalize_writes_no_predictive_normalization(tmp_path):
    """A clean run leaves no predictive_normalization key in scores or summary."""
    bg = adapter()
    pytest.importorskip("pymc")
    pytest.importorskip(bg.TASKS["death"].module)
    out = tmp_path / "clean"
    bg.run(bg_run_args(bg, out))
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    assert rows and not any("predictive_normalization" in row for row in rows)
    assert "predictive_normalization" not in json.loads((out / "summary.json").read_text())


def test_a_drifting_run_records_the_event_in_the_score_row_and_the_summary(tmp_path, monkeypatch):
    """Every binomial row drifts by ~4e-8, which is inside DRIFT_TOL.

    The run must finish, the checkpoint that renormalized rows must say so, and
    the summary must carry the run total under the same condition.
    """
    bg = adapter()
    pytest.importorskip("pymc")
    pytest.importorskip(bg.TASKS["death"].module)
    real = count_models.binom

    class _Drifting:
        """scipy's binomial, with a deviation of 4e-8 added to every log pmf."""

        @staticmethod
        def logpmf(*args, **kwargs):
            """Return scipy's binomial log pmf shifted by 4e-8."""
            return real.logpmf(*args, **kwargs) + 4e-8

    monkeypatch.setattr(count_models, "binom", _Drifting)
    out = tmp_path / "drifting"
    bg.run(bg_run_args(bg, out))
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    recorded = [row["predictive_normalization"] for row in rows if "predictive_normalization" in row]
    assert recorded, "a renormalized predictive must say so in its own score row"
    for record in recorded:
        assert record["renormalized_rows"] > 0
        assert record["zeroed_rows"] == 0
        assert record["drift_tolerance"] == DRIFT_TOL
        # a running maximum, not this checkpoint's own (see the record's
        # docstring): all it can promise is that the rule fired at all
        assert record["max_deviation_cumulative"] > NORMALIZATION_ATOL
    summary = json.loads((out / "summary.json").read_text())
    total = summary["predictive_normalization"]
    assert total["checkpoints"] == len(recorded)
    assert total["renormalized_rows"] == sum(r["renormalized_rows"] for r in recorded)
    assert total["zeroed_rows"] == 0
    assert total["drift_tolerance"] == DRIFT_TOL
    assert total["max_deviation_cumulative"] >= max(record["max_deviation_cumulative"] for record in recorded)


# --------------------------------------------------------------------------
# 4. the regression: the saved fit that ended a run
# --------------------------------------------------------------------------


def saved_fit():
    """The t18 beta-binomial fit, particles and evaluation inputs only."""
    bg = adapter()
    model = json.loads((FIXTURE / "model.json").read_text())
    interface = bg.TASKS["death"].interface()
    namespace = (tuple(item["name"] for item in interface["inputs"])
                 + tuple(interface["derived_inputs"]))
    hypothesis = Hypothesis(
        model["expression"], model["family"], tuple(model["parameters"].items()), namespace, {}
    )
    with np.load(FIXTURE / "particles.npz") as saved:
        particles = {name: saved[name] for name in saved.files}
    return (Fit(hypothesis, particles, model["log_evidence"]),
            np.load(FIXTURE / "evaluation_x.npy"), interface)


def raw_named_rows(hypothesis, particles, x, interface):
    """The saved fit's log pmf table BEFORE the rule, written out here.

    The mean expression goes through the module's own `evaluate` (there is no
    second parser), but the likelihood is this file's own call, so the
    deviations below are not the ones the rule itself measured.
    """
    named = {name: np.asarray(value, dtype=float)[None, :]
             for name, value in count_models.named_input_values(interface, x).items()}
    params = {name: np.asarray(particles[name], dtype=float)[:, None]
              for name, _ in hypothesis.parameters}
    logits = count_models.evaluate(
        hypothesis.expression, params, named[interface["inputs"][0]["name"]], np, named, {}
    )
    p = expit(logits)[:, :, None]
    kappa = (1 / np.asarray(particles["rho"], dtype=float) - 1)[:, None, None]
    counts = np.arange(interface["trials"] + 1)[None, None, :]
    return betabinom.logpmf(counts, interface["trials"], p * kappa, (1 - p) * kappa)


def test_the_saved_fit_that_ended_a_run_now_replays_as_drift():
    """One particle of 1000, at kappa ~ 1.8e5, drifting by at most 6.02e-10.

    Before the rule existed this call raised "likelihood normalization failed" and the run
    ended at round 18 with no summary. It is drift, not a defect: no particle is
    zeroed, no cell is NaN, and every row normalizes afterwards.
    """
    fit, evaluation_x, interface = saved_fit()
    assert fit.hypothesis.family == "beta_binomial"
    assert float(fit.particles["rho"].min()) == pytest.approx(5.6e-6, rel=0.01)
    # what the saved particles do before the rule sees them
    untouched = deviations(raw_named_rows(fit.hypothesis, fit.particles, evaluation_x, interface))
    missing = untouched > NORMALIZATION_ATOL
    assert untouched.max() == pytest.approx(6.02e-10, rel=0.05)
    assert untouched.max() <= DRIFT_TOL  # drift, not a defect
    assert missing.any(axis=1).sum() == 1  # exactly one particle of 1000
    before = normalization_event_counter()
    rows = particle_logpmf(fit.hypothesis, fit.particles, evaluation_x,
                           interface["trials"], interface)
    after = normalization_event_counter()
    assert after["renormalized_rows"] - before["renormalized_rows"] == int(missing.sum())
    assert 0 < int(missing.sum()) <= len(evaluation_x)  # one particle, at some inputs
    assert after["zeroed_rows"] - before["zeroed_rows"] == 0
    # a running maximum, so it can only be read as "at least this much"
    assert after["max_deviation"] >= untouched.max()
    assert not np.isnan(rows).any()
    assert np.all(deviations(rows) <= NORMALIZATION_ATOL)


def test_predict_on_the_saved_fit_completes_and_keeps_a_normalized_mixture():
    """Prediction on the saved fit completes and the mixture stays normalized."""
    fit, evaluation_x, interface = saved_fit()
    before = normalization_event_counter()
    prediction = predict([fit], evaluation_x, interface, interface["trials"])
    assert normalization_event_counter()["renormalized_rows"] > before["renormalized_rows"]
    logpmf = prediction["bma_logpmf"]
    assert logpmf.shape == (len(evaluation_x), interface["trials"] + 1)
    assert np.all(np.isfinite(logpmf))
    np.testing.assert_allclose(np.exp(logpmf).sum(-1), 1.0, atol=1e-10, rtol=0)
    assert prediction["single_model_id"] == fit.hypothesis.model_id
