"""The NaN numeric rule and refit exception capture.

The defect these tests pin down killed two paid runs outright: one active
model's sequential Monte Carlo (SMC) refit raised

    ValueError: cannot convert float NaN to integer

inside `pymc/smc/kernels.py:273`, `ESS = int(np.exp(-logsumexp(log_weights*2)))`,
and `core_count.run` had no handler around the ACTIVE refit, so the whole run
ended with no `summary.json`. `NAN_LIKELIHOOD_HYPOTHESIS` below reproduces that
exact PyMC failure from first principles: every prior draw overflows the logit,
so `p` is exactly 0 or 1, so the beta-binomial shape parameter is 0, so every
particle's likelihood log-density is -inf, so `logsumexp` of the unnormalized log
weights is -inf, so every normalized log weight is `-inf - (-inf) = NaN`, and the
integer conversion of the effective sample size raises.
"""

import json
import sys

import numpy as np
import pytest
from scipy.stats import betabinom

from . import core_count
from .core_count import D0, PRIOR_PROBE_NONFINITE
from .count_inference import SMCSettings, build_model
from .count_inference import fit as real_fit
from .count_models import (
    Hypothesis,
    initial_hypotheses,
    nan_substitution_counter,
    particle_logpmf,
)

# Every prior draw sends the logit to +-inf, so p is exactly 0 or 1 and the
# beta-binomial alpha or beta is 0: the likelihood of EVERY particle is -inf.
NAN_LIKELIHOOD_HYPOTHESIS = Hypothesis(
    "b0+b1*exp(exp(exp(exp(u))))", "beta_binomial", (("b0", "intercept"), ("b1", "signed"))
)
# log of a negative argument: the mean expression itself is undefined (NaN).
UNDEFINED_LOGIT_HYPOTHESIS = Hypothesis(
    "b0+b1*log(u+b2)",
    "beta_binomial",
    (("b0", "intercept"), ("b1", "signed"), ("b2", "signed")),
)
SMALL = SMCSettings(particles=16, chains=2, cores=2)
INITIAL_HYPOTHESIS = initial_hypotheses("binomial")[1]
U = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
Y = np.array([1, 5, 25, 45, 49])


# --------------------------------------------------------------------------
# (1) the numeric rule: a NaN particle log pmf is -inf
# --------------------------------------------------------------------------


def test_nan_cell_becomes_negative_infinity_and_the_row_still_normalizes(monkeypatch):
    """A NaN in a NEGLIGIBLE cell yields -inf there instead of a rejection.

    White box on purpose: every NaN scipy actually produces for these three
    families covers a whole count row (see the next test), so the only way to
    exhibit the substitution itself is to inject one NaN into a cell whose true
    probability is far below the 1e-10 normalization tolerance. Earlier versions refused such
    a table outright ("invalid likelihood"); now the undefined cell is defined as
    impossible and a usable, normalized table is returned.
    """
    hypothesis = Hypothesis("b0+b1*z", "beta_binomial", (("b0", "intercept"), ("b1", "signed")))
    particles = {"b0": np.array([0.0]), "b1": np.array([0.0]), "rho": np.array([0.001])}
    clean = particle_logpmf(hypothesis, particles, np.array([0.5]))
    assert np.all(np.isfinite(clean))
    # at p = 0.5 and kappa = 999 the extreme count carries ~3e-15 of the mass,
    # far below the 1e-10 normalization tolerance
    negligible = int(np.argmin(clean[0, 0]))
    assert np.exp(clean[0, 0, negligible]) < 1e-11

    original = betabinom.logpmf

    def poisoned(*args, **kwargs):
        """Wrap betabinom.logpmf and set one negligible-mass cell to NaN."""
        value = np.array(original(*args, **kwargs), dtype=float)
        if value.ndim == 3:
            value[:, :, negligible] = np.nan
        return value

    monkeypatch.setattr(betabinom, "logpmf", poisoned)
    before = nan_substitution_counter()
    result = particle_logpmf(hypothesis, particles, np.array([0.5]))
    after = nan_substitution_counter()
    # rule (1): the substitution is COUNTED, never silent
    assert after["cells"] - before["cells"] == 1
    assert after["calls"] - before["calls"] == 1
    assert not np.any(np.isnan(result))
    assert np.isneginf(result[0, 0, negligible])
    np.testing.assert_array_equal(
        np.delete(result[0, 0], negligible), np.delete(clean[0, 0], negligible)
    )
    assert np.isclose(np.exp(result).sum(-1), 1.0, atol=1e-10, rtol=0)


def test_real_nan_table_keeps_the_v2_4_rejection_message_for_the_prior_probe():
    """The reachable NaN covers the whole row, so the table cannot normalize.

    Such a table is still rejected, and with the SAME message, so
    `core_count.PRIOR_PROBE_NONFINITE` classifies it exactly as before - which
    is why the `P1`/`P4` proposal-policy anchors are byte-identical.
    """
    hypothesis = Hypothesis("b0+b1*z", "beta_binomial", (("b0", "intercept"), ("b1", "signed")))
    # smallest positive double: kappa = 1/rho - 1 overflows, scipy returns NaN
    particles = {"b0": np.array([0.0]), "b1": np.array([3.0]), "rho": np.array([5e-324])}
    with pytest.raises(ValueError) as excinfo:
        particle_logpmf(hypothesis, particles, np.array([0.1, 0.5, 0.9]))
    assert "invalid likelihood" in str(excinfo.value)
    assert any(marker in str(excinfo.value) for marker in PRIOR_PROBE_NONFINITE)


def test_particle_logpmf_never_returns_nan_for_the_undefined_logit_model():
    """A NaN mean expression is still caught upstream, before the pmf exists."""
    particles = {
        "b0": np.array([0.0]),
        "b1": np.array([1.0]),
        "b2": np.array([-0.5]),
        "rho": np.array([0.1]),
    }
    with pytest.raises(ValueError, match="non-finite mean expression"):
        particle_logpmf(UNDEFINED_LOGIT_HYPOTHESIS, particles, np.array([0.1, 0.2]))


@pytest.mark.parametrize("family", ("binomial", "beta_binomial", "zi_binomial"))
def test_pymc_model_logp_is_minus_infinity_not_nan_where_the_logit_is_undefined(family):
    """OBSERVED behaviour of PyMC 5.13.1, not a guard of ours.

    `build_model` carries no NaN guard. This records what the pinned PyMC
    actually does with an undefined logit - `check_parameters` yields -inf, not
    NaN - which is why an earlier in-graph guard bought nothing and was removed.
    It is a pinned-stack observation; the contract that a run RELIES on is the
    exception path (`CountInferenceFailure` -> structured record -> retirement).
    """
    hypothesis = Hypothesis(
        "b0+b1*log(u+b2)", family, (("b0", "intercept"), ("b1", "signed"), ("b2", "signed"))
    )
    model = build_model(hypothesis, U, Y)
    with model:
        data_logp = model.compile_fn(model.datalogp, point_fn=True)
    extra = {"beta_binomial": {"rho": np.array(0.1)}, "zi_binomial": {"pi": np.array(0.1)}}
    point = {"b0": np.array(0.0), "b1": np.array(1.0), **extra.get(family, {})}
    undefined = data_logp({**point, "b2": np.array(-0.5)})  # log of a negative argument
    defined = data_logp({**point, "b2": np.array(0.5)})
    assert float(undefined) == -np.inf
    assert np.isfinite(float(defined))


def test_fit_on_a_nan_producing_model_does_not_crash_the_process():
    """`fit` either returns finite evidence or raises a plain catchable exception."""
    try:
        result = real_fit(UNDEFINED_LOGIT_HYPOTHESIS, U, Y, seed=11, settings=SMALL)
    except Exception as exc:  # noqa: BLE001 - a catchable failure is the accepted outcome
        assert isinstance(exc, Exception) and not isinstance(exc, (SystemExit, KeyboardInterrupt))
    else:
        assert np.isfinite(result.log_evidence)


def test_the_observed_failure_mode_reproduces_and_stays_catchable():
    """The exact PyMC failure that ended the two paid runs, raised and caught.

    `fit` re-raises it as the TYPED `CountInferenceFailure`, keeping
    the original message, so `core_count` can retire a model on this and on
    nothing else. The original ValueError stays as `__cause__`.
    """
    from .count_inference import CountInferenceFailure

    with pytest.raises(CountInferenceFailure) as excinfo:
        real_fit(NAN_LIKELIHOOD_HYPOTHESIS, U, Y, seed=11, settings=SMALL)
    assert "cannot convert float NaN to integer" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)
    # named by the classifier, and the original is preserved durably
    assert excinfo.value.cause_code == "smc_weight_normalization_nonfinite"
    assert excinfo.value.stage == "sampling"
    assert excinfo.value.original["class"] == "ValueError"
    assert "pymc/smc/kernels.py" in excinfo.value.original["traceback"]


# --------------------------------------------------------------------------
# (2)/(3) exception capture around every refit, through a real run
# --------------------------------------------------------------------------

INITIAL_IDS = frozenset(h.model_id for h in initial_hypotheses("binomial"))


def _inject(monkeypatch, should_fail):
    """Replace the fit of the selected models by a REAL SMC fit of a model whose
    every particle has likelihood -inf, so the run sees the genuine PyMC
    `cannot convert float NaN to integer` - never a stubbed exception."""

    def patched(hypothesis, u, y, seed, *rest, **kwargs):
        """Fit the NaN-likelihood model in place of the selected hypotheses."""
        # *rest/**kwargs so the same test rides along with later signatures of
        # count_inference.fit (interface / n_trials) without being rewritten.
        if should_fail(hypothesis, len(y)):
            return real_fit(NAN_LIKELIHOOD_HYPOTHESIS, u, y, seed, *rest, **kwargs)
        return real_fit(hypothesis, u, y, seed, *rest, **kwargs)

    monkeypatch.setattr(core_count, "fit", patched)


def _run(tmp_path, monkeypatch, name="run"):
    out = tmp_path / name
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "core_count",
            "--output",
            str(out),
            "--profile",
            "dev",
            "--rounds",
            "1",
            "--seed",
            "11",
            "--cell",
            "111",
            "--proposer",
            "mock",
            "--proposal-policy",
            "P1",
        ],
    )
    core_count.main()
    return out


def _json(path):
    return json.loads(path.read_text())


def test_active_refit_failure_retires_that_model_and_the_run_completes(tmp_path, monkeypatch):
    """An active-model refit failure retires that model, records it, and the run completes."""
    victim = min(INITIAL_IDS)
    _inject(monkeypatch, lambda h, n: h.model_id == victim and n > len(D0))  # round >= 1 only
    out = _run(tmp_path, monkeypatch)

    summary = _json(out / "summary.json")
    assert summary["status"] == "completed"
    (entry,) = summary["active_refit_failures"]
    assert entry["round"] == 1
    assert entry["retired_model_ids"] == [victim]
    assert entry["failure_count"] == 1
    assert entry["active_after"] == len(INITIAL_IDS) - 1
    assert entry["pool_empty"] is False

    # rule (3): identity of every NaN-retired model, in summary.json
    (record,) = summary["refit_failed_models"]
    assert record == entry["failures"][0]
    assert record["model_id"] == victim and record["round"] == 1
    assert record["scope"] == "active"
    assert record["family"] and record["expression"]
    assert "cannot convert float NaN to integer" in record["exception"]
    assert set(record) == {
        "model_id",
        "expression",
        "family",
        "round",
        "scope",
        "stage",
        "retirement_kind",
        "cause_code",
        "exception",
        "original_exception",
        "fit_seed",
        "nan_substitutions",
    }
    assert record["stage"] == "fit"
    # wording and durability
    assert record["retirement_kind"] == "inference_failure"
    assert record["cause_code"] == "smc_weight_normalization_nonfinite"
    assert record["fit_seed"] is not None
    original = record["original_exception"]
    assert original["class"] == "ValueError"
    assert "cannot convert float NaN to integer" in original["message"]
    assert "pymc/smc/kernels.py" in original["traceback"]
    assert summary["predict_failures"] == []
    assert set(summary["nan_substitutions"]) == {"cells", "calls", "attempts_with_substitutions"}

    archive = {item["model_id"]: item for item in _json(out / "archive.json")}
    assert archive[victim]["outcome"] == "fit_failed"
    # the archive reason keeps the original shape belief_packet.py validates
    assert set(archive[victim]["reason"]) == {"error"}
    assert archive[victim]["reason"]["error"] == record["exception"]
    # the run continued with the rest of the pool and scored it
    assert summary["active_count"] >= len(INITIAL_IDS) - 1
    assert (out / "points.csv").exists()


def test_all_active_failing_ends_the_run_with_status_pool_empty(tmp_path, monkeypatch):
    """Pre-declared EMPTY-POOL RULE: terminate at that round, no re-seed.

    No summary.json, no points.csv, no round-1 checkpoint, and NOTHING carried
    forward from round 0 - the later checkpoints are simply missing.
    """
    _inject(monkeypatch, lambda h, n: n > len(D0) and h.model_id in INITIAL_IDS)
    with pytest.raises(core_count.PoolEmpty, match="active pool empty at round 1"):
        _run(tmp_path, monkeypatch)

    out = tmp_path / "run"
    assert not (out / "summary.json").exists()
    assert not (out / "points.csv").exists()
    # Round 0 was checkpointed when it happened and legitimately stays on disk.
    # Round 1 has NO checkpoint: nothing is imputed and round 0's prediction is
    # not carried forward into it.
    assert (out / "predictions" / "t00_initial.npz").exists()
    assert not list((out / "predictions").glob("t01*"))
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    assert [row["round"] for row in rows] == [0]

    failure = _json(out / "failure.json")
    assert failure["status"] == "pool_empty"
    assert failure["rounds_completed"] == 1
    assert "no re-seed" in failure["error"]
    (entry,) = failure["active_refit_failures"]
    assert entry["round"] == 1
    assert entry["pool_empty"] is True
    assert entry["active_after"] == 0
    assert entry["failure_count"] == len(INITIAL_IDS)
    assert {r["model_id"] for r in failure["refit_failed_models"]} == INITIAL_IDS
    assert all(r["scope"] == "active" for r in failure["refit_failed_models"])
    # no model was refit a second time: the initial pool is never re-seeded
    assert len(failure["refit_failed_models"]) == len(INITIAL_IDS)


def test_candidate_refit_failure_is_archived_and_the_run_completes(tmp_path, monkeypatch):
    """The candidate path: same NaN failure, same structured record, run completes."""
    _inject(monkeypatch, lambda h, n: h.model_id not in INITIAL_IDS)
    out = _run(tmp_path, monkeypatch)

    summary = _json(out / "summary.json")
    assert summary["status"] == "completed"
    assert summary["active_refit_failures"] == []
    accounting = summary["proposal_accounting"]
    assert accounting["fit_attempts"] >= 1
    assert accounting["fit_successes"] == 0
    assert summary["candidate_failures"], "the failed candidate must be reported"
    assert all(
        "cannot convert float NaN to integer" in item["error"]
        for item in summary["candidate_failures"]
    )
    # rule (1): the candidate path carries the SAME structured record
    assert summary["refit_failed_models"]
    for record in summary["refit_failed_models"]:
        assert record["scope"] == "candidate"
        assert record["round"] == 1
        assert "cannot convert float NaN to integer" in record["exception"]
        assert record["expression"] and record["family"]
    assert all(item["status"] == "fit_failed" for item in summary["candidate_failures"])
    assert summary["predict_failures"] == []
    assert all(r["stage"] == "fit" for r in summary["refit_failed_models"])
    failed = [e for e in _json(out / "archive.json") if e["outcome"] == "fit_failed"]
    assert failed and all(set(e["reason"]) == {"error"} for e in failed)
    assert {e["model_id"] for e in failed} == {
        r["model_id"] for r in summary["refit_failed_models"]
    }


# --------------------------------------------------------------------------
# What a retirement MEANS
#
# An earlier version retired a model on ANY exception raised anywhere inside `refit` - the
# SMC fit, but also the NPZ/JSON writes, the prior probe and the diagnostics.
# A full disk or a typo therefore looked exactly like "this model's likelihood
# is numerically impossible", and four of those in one round looked like
# `pool_empty`. The catch is now narrowed to ONE typed exception raised by
# `count_inference.fit` for EXPECTED numerical failures; everything else is a
# run-level failure, as it always should have been.
# --------------------------------------------------------------------------


def _inject_raise(monkeypatch, exc, only=lambda h, n: True):
    """Make `count_inference.fit` raise `exc` for the selected models."""

    def patched(hypothesis, u, y, seed, *rest, **kwargs):
        """Raise the chosen exception for selected models, otherwise fit normally."""
        if only(hypothesis, len(y)):
            raise exc
        return real_fit(hypothesis, u, y, seed, *rest, **kwargs)

    monkeypatch.setattr(core_count, "fit", patched)


@pytest.mark.parametrize(
    "exc",
    [
        OSError(28, "No space left on device"),
        MemoryError("out of memory"),
    ],
    ids=["oserror", "memoryerror"],
)
def test_infrastructure_error_is_not_model_retirement(tmp_path, monkeypatch, exc):
    """A full disk is not an impossible likelihood.

    An infrastructure failure must reach the run's own failure path, NOT retire
    the model that happened to be fitting when it struck, and NEVER be dressed
    up as `pool_empty`.
    """
    _inject_raise(monkeypatch, exc)
    with pytest.raises(type(exc)):
        _run(tmp_path, monkeypatch)

    out = tmp_path / "run"
    assert not (out / "summary.json").exists()
    failure = _json(out / "failure.json")
    assert failure["status"] == "failed", "an infrastructure error is not pool_empty"
    assert failure["active_refit_failures"] == []
    assert failure["refit_failed_models"] == []
    assert not [e for e in _json(out / "archive.json")] if (out / "archive.json").exists() else True


@pytest.mark.parametrize(
    "exc",
    [AssertionError("invariant broken"), TypeError("bad argument"), KeyError("missing")],
    ids=["assertion", "typeerror", "keyerror"],
)
def test_programming_error_is_not_model_retirement(tmp_path, monkeypatch, exc):
    """A bug in our own code must surface as a bug, not as a retired model."""
    _inject_raise(monkeypatch, exc)
    with pytest.raises(type(exc)):
        _run(tmp_path, monkeypatch)

    failure = _json(tmp_path / "run" / "failure.json")
    assert failure["status"] == "failed"
    assert failure["refit_failed_models"] == []
    assert failure["active_refit_failures"] == []


def test_artifact_write_error_is_not_model_retirement(tmp_path, monkeypatch):
    """A failure AFTER the fit (artifact write) is run-level, not a retirement."""
    real_savez = core_count.np.savez_compressed

    def exploding(path, **arrays):
        """Raise a disk-full error when saving a per-fit archive, else save normally."""
        # only the per-fit particle archive, so the failure is unambiguously a
        # POST-fit artifact write inside `refit`, not a setup error
        if "fits" in str(path):
            raise OSError(28, "No space left on device")
        return real_savez(path, **arrays)

    monkeypatch.setattr(core_count.np, "savez_compressed", exploding)
    with pytest.raises(OSError):
        _run(tmp_path, monkeypatch)
    failure = _json(tmp_path / "run" / "failure.json")
    assert failure["status"] == "failed"
    assert failure["refit_failed_models"] == []
    assert real_savez is not None


def test_predict_failure_is_not_recorded_as_refit_failure(tmp_path, monkeypatch):
    """A candidate whose PREDICT fails is not a fit failure.

    `refit_failed_models` is the NaN/numerical-retirement ledger; a predict
    failure belongs in its own list with `stage="predict"` so a report can never
    read it as "the likelihood was impossible".
    """
    calls = {"n": 0}
    real_predict = core_count.predict

    def patched_predict(fits, designs, *rest, **kwargs):
        """Fail the predict of single non-initial candidates with a numerical error."""
        calls["n"] += 1
        if len(fits) == 1 and fits[0].hypothesis.model_id not in INITIAL_IDS:
            # a NUMERICAL scoring failure, in the words our own code uses
            raise ValueError("non-finite evidence")
        return real_predict(fits, designs, *rest, **kwargs)

    monkeypatch.setattr(core_count, "predict", patched_predict)
    out = _run(tmp_path, monkeypatch)

    summary = _json(out / "summary.json")
    assert summary["status"] == "completed"
    assert summary["refit_failed_models"] == [], "a predict failure is not a refit failure"
    assert summary["active_refit_failures"] == []
    assert summary["predict_failures"], "the predict failure must be recorded somewhere"
    for record in summary["predict_failures"]:
        assert record["stage"] == "predict"
        assert record["scope"] == "candidate"
        assert "non-finite evidence" in record["exception"]
        assert record["model_id"] and record["expression"] and record["family"]
    assert summary["candidate_failures"]
    assert all(item["status"] == "predict_failed" for item in summary["candidate_failures"])
    # the FIT succeeded, and is counted as such
    assert summary["proposal_accounting"]["fit_successes"] >= 1
    archived = [e for e in _json(out / "archive.json") if e["outcome"] == "fit_failed"]
    assert archived and all(
        "stage=predict; fit succeeded" in e["reason"]["error"] for e in archived
    )


def test_pymc_numerical_failure_is_a_typed_structured_record(tmp_path, monkeypatch):
    """Policy: the PyMC side is covered by the EXCEPTION, not by a graph guard.

    `build_model` carries no NaN guard. The abnormal PyMC path - every
    particle at -inf, which is the actual incident - is observed exactly once, as
    a typed `CountInferenceFailure` turned into a structured failure record and a
    model retirement, with the NaN counter honestly reading zero because the
    NumPy substitution rule never fired.
    """
    from .count_inference import CountInferenceFailure

    with pytest.raises(CountInferenceFailure) as excinfo:
        real_fit(NAN_LIKELIHOOD_HYPOTHESIS, U, Y, seed=11, settings=SMALL)
    assert "cannot convert float NaN to integer" in str(excinfo.value)

    # and there is no in-graph guard left to hide it
    source = build_model.__code__.co_consts
    assert not any(isinstance(c, str) and "undefined_logit_is_impossible" in c for c in source), (
        "the v2.4a PyMC Potential guard must be gone"
    )

    victim = min(INITIAL_IDS)
    _inject(monkeypatch, lambda h, n: h.model_id == victim and n > len(D0))
    out = _run(tmp_path, monkeypatch, name="run2")
    summary = _json(out / "summary.json")
    (record,) = summary["refit_failed_models"]
    assert record["model_id"] == victim
    assert record["stage"] == "fit" and record["scope"] == "active"
    assert "cannot convert float NaN to integer" in record["exception"]
    # the numeric rule did not fire here: the counter says so instead of guessing
    assert record["nan_substitutions"] == {"cells": 0, "calls": 0}
    assert summary["nan_substitutions"]["cells"] == 0


# --------------------------------------------------------------------------
# The three catches OUTSIDE the refit path
#
# `prior_probe`, `one_step_log_score` and the candidate `predict` each swallowed
# ANY exception and turned it into a domain verdict - "rejected under its own
# prior", "this model scores null this round", "this candidate could not be
# scored". They pre-date the refit-path narrowing above, and under the same rule
# applied to the refit path they are wrong for the same reason: a full disk is
# not a rejection, a bug is not a null score. All three are now narrowed to the
# numerical family and everything else fails the run.
# --------------------------------------------------------------------------

DISK_FULL = OSError(28, "No space left on device")


def test_infrastructure_error_in_the_prior_probe_is_not_a_rejection(tmp_path, monkeypatch):
    """An OSError inside the prior probe fails the run, it does not reject.

    The probe is the only caller that passes `n_trials` positionally, which makes
    it the one call this injection can hit.
    """
    real = core_count.particle_logpmf

    def exploding(hypothesis, particles, u, *rest, **kwargs):
        """Raise a disk-full error when the prior probe passes the trial count."""
        if rest:  # prior_probe passes interface["trials"]; nobody else does
            raise DISK_FULL
        return real(hypothesis, particles, u, *rest, **kwargs)

    monkeypatch.setattr(core_count, "particle_logpmf", exploding)
    with pytest.raises(OSError):
        _run(tmp_path, monkeypatch)
    out = tmp_path / "run"
    assert not (out / "summary.json").exists()
    assert _json(out / "failure.json")["status"] == "failed"


def test_infrastructure_error_in_the_one_step_score_is_not_a_null_score(tmp_path, monkeypatch):
    """An OSError inside the one-step scorer fails the run, it does not score None.

    `one_step_log_score` is the only caller that asks for a SINGLE design.
    """
    real = core_count.particle_logpmf

    def exploding(hypothesis, particles, u, *rest, **kwargs):
        """Raise a disk-full error for single-design calls from the one-step scorer."""
        if len(u) == 1:
            raise DISK_FULL
        return real(hypothesis, particles, u, *rest, **kwargs)

    monkeypatch.setattr(core_count, "particle_logpmf", exploding)
    with pytest.raises(OSError):
        _run(tmp_path, monkeypatch)
    out = tmp_path / "run"
    assert not (out / "summary.json").exists()
    assert _json(out / "failure.json")["status"] == "failed"


def test_infrastructure_error_in_candidate_predict_is_not_predict_failed(tmp_path, monkeypatch):
    """An OSError from the candidate predict fails the run, not `predict_failed`."""
    real_predict = core_count.predict

    def patched_predict(fits, designs, *rest, **kwargs):
        """Raise a disk-full error for single non-initial candidates."""
        if len(fits) == 1 and fits[0].hypothesis.model_id not in INITIAL_IDS:
            raise DISK_FULL
        return real_predict(fits, designs, *rest, **kwargs)

    monkeypatch.setattr(core_count, "predict", patched_predict)
    with pytest.raises(OSError):
        _run(tmp_path, monkeypatch)
    out = tmp_path / "run"
    assert not (out / "summary.json").exists()
    failure = _json(out / "failure.json")
    assert failure["status"] == "failed"
    assert failure["predict_failures"] == []
    assert failure["refit_failed_models"] == []


# --------------------------------------------------------------------------
# The classifier boundary (external design review)
#
# An exception CLASS is not evidence. PyMC 5.13.1 raises plain ValueError for a
# bad threshold, a broadcast mismatch, a malformed seed and a wrong return
# format just as readily as for the SMC weight collapse, and
# `pymc.exceptions.IncorrectArgumentsError` is a ValueError subclass. These pin
# down that only a NAMED numerical cause becomes a retirement.
# --------------------------------------------------------------------------


def _classify(exc, stage="sampling"):
    from .count_inference import classify_inference_failure

    try:
        raise exc
    except type(exc) as raised:
        return classify_inference_failure(raised, stage)


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("could not broadcast input array from shape (3,) into shape (4,)"),
        ValueError("The number of samples must be a positive integer"),
        ValueError("Invalid seed"),
        # the SAME message from somewhere that is NOT the SMC kernel
        ValueError("cannot convert float NaN to integer"),
        OSError(28, "No space left on device"),
        MemoryError("out of memory"),
        AssertionError("invariant broken"),
        TypeError("bad argument"),
        KeyError("missing"),
        RuntimeError("worker died"),
        ZeroDivisionError("division by zero"),
    ],
    ids=[
        "broadcast",
        "config",
        "seed",
        "same_message_wrong_frames",
        "oserror",
        "memoryerror",
        "assertion",
        "typeerror",
        "keyerror",
        "runtime",
        "zerodivision",
    ],
)
def test_classifier_refuses_to_name_a_non_numerical_cause(exc):
    """None means "propagate": these must never become a model retirement."""
    assert _classify(exc) is None


def test_classifier_names_the_real_incident_only_from_the_smc_kernel_frames():
    """The one message that IS the incident, but only with the kernel frames."""
    from .count_inference import CountInferenceFailure, classify_inference_failure

    with pytest.raises(CountInferenceFailure) as excinfo:
        real_fit(NAN_LIKELIHOOD_HYPOTHESIS, U, Y, seed=11, settings=SMALL)
    original = excinfo.value.__cause__
    assert classify_inference_failure(original, "sampling") == "smc_weight_normalization_nonfinite"
    assert "pymc/smc/kernels.py" in excinfo.value.original["traceback"]


def test_a_broken_sampler_return_contract_is_a_run_failure_not_a_retirement(monkeypatch):
    """A malformed InferenceData is a library/version problem, never a verdict."""
    import pymc as pm

    from .count_inference import CountInferenceFailure

    class Broken:
        """A sampler return value lacking the expected InferenceData structure."""

        def __init__(self):
            self.posterior = {}
            self.sample_stats = {}

    monkeypatch.setattr(pm, "sample_smc", lambda *a, **k: Broken())
    with pytest.raises(RuntimeError) as excinfo:
        real_fit(INITIAL_HYPOTHESIS, U, Y, seed=11, settings=SMALL)
    assert not isinstance(excinfo.value, CountInferenceFailure)
    assert "missing posterior variables" in str(excinfo.value)


def test_nonfinite_particles_in_a_well_formed_result_are_a_named_numerical_failure(monkeypatch):
    """The value check runs only AFTER the contract check, and is named."""
    import pymc as pm

    from .count_inference import CountInferenceFailure

    real_sample = pm.sample_smc

    def poisoned(*args, **kwargs):
        """Run the real SMC sampler, then overwrite the b0 posterior with NaN."""
        idata = real_sample(*args, **kwargs)
        idata.posterior["b0"].values[...] = np.nan
        return idata

    monkeypatch.setattr(pm, "sample_smc", poisoned)
    with pytest.raises(CountInferenceFailure) as excinfo:
        real_fit(INITIAL_HYPOTHESIS, U, Y, seed=11, settings=SMALL)
    assert excinfo.value.cause_code == "nonfinite_posterior"
    assert excinfo.value.stage == "output_validation"


# --------------------------------------------------------------------------
# The FINAL-evidence path
#
# `pymc/smc/kernels.py:316` records
#     "log_marginal_likelihood": self.log_marginal_likelihood if self.beta == 1 else np.nan
# so EVERY non-terminal stage is NaN by construction and the terminal stage is
# the one with beta == 1. The shared `_extract_logz` takes the LAST FINITE value
# per chain, which coincides with the terminus only while the terminus is finite.
# The reviewer reproduced both ways that breaks: an all-NaN column escapes as a shared
# `SMCError` instead of a typed numerical failure, and a non-finite terminal
# value silently returns an EARLIER stage's evidence as if the fit had succeeded.
# These six cases are the reviewer's, as mock-InferenceData unit tests.
# --------------------------------------------------------------------------

TERMINAL_HYPOTHESIS = Hypothesis("b0+b1*z", "binomial", (("b0", "intercept"), ("b1", "signed")))
TERMINAL_SETTINGS = SMCSettings(particles=16, chains=2, cores=1)
# the wrong answer an earlier version returned for every non-finite terminal case
V24C_SUBSTITUTED_LOG_EVIDENCE = -3.3798854930417224


def _mock_idata(evidence, beta=((0.5, 1.0), (0.5, 1.0)), posterior_nan=False):
    import arviz as az

    particles = np.zeros((2, 16))
    if posterior_nan:
        particles[0, -1] = np.nan
    return az.from_dict(
        posterior={"b0": particles, "b1": np.zeros((2, 16))},
        sample_stats={
            "log_marginal_likelihood": np.asarray(evidence, dtype=float),
            "beta": np.asarray(beta, dtype=float),
        },
    )


def _mock_idata_missing_beta(evidence):
    """The reviewer's counterexample shape: no `beta` key in
    `sample_stats` at all.

    pymc 5.13.1 always writes `beta` alongside `log_marginal_likelihood`
    (`kernels.py:316-317`), so this return never happens against the installed
    version - it IS the return-contract violation, tested directly.
    """
    import arviz as az

    return az.from_dict(
        posterior={"b0": np.zeros((2, 16)), "b1": np.zeros((2, 16))},
        sample_stats={"log_marginal_likelihood": np.asarray(evidence, dtype=float)},
    )


def _fit_returning(monkeypatch, idata):
    """Run `fit` against a fixed sampler return; no posterior is sampled."""
    from contextlib import nullcontext

    import pymc as pm

    from . import count_inference as ci

    monkeypatch.setattr(ci, "build_model", lambda *a, **k: nullcontext())
    monkeypatch.setattr(pm, "sample_smc", lambda *a, **k: idata)
    return ci.fit(TERMINAL_HYPOTHESIS, [0.2, 0.8], [10, 30], 11, TERMINAL_SETTINGS)


def test_intermediate_nan_before_the_final_stage_is_normal(monkeypatch):
    """Reviewer case 1: the shape EVERY healthy pymc 5.13.1 fit has."""
    fitted = _fit_returning(monkeypatch, _mock_idata([[np.nan, -3.0], [np.nan, -4.0]]))
    assert fitted.per_chain_log_evidence == [-3.0, -4.0]
    assert np.isfinite(fitted.log_evidence)


def test_all_stage_evidence_nonfinite_is_a_typed_numerical_failure(monkeypatch):
    """Reviewer case 2: one model's numerical failure, not a whole-run failure."""
    from .count_inference import CountInferenceFailure

    with pytest.raises(CountInferenceFailure) as excinfo:
        _fit_returning(monkeypatch, _mock_idata([[np.nan, np.nan], [np.nan, np.nan]]))
    assert excinfo.value.cause_code == "nonfinite_final_evidence"
    assert excinfo.value.stage == "output_validation"
    original = excinfo.value.original
    assert original["final_stage_index"] == [1, 1]
    assert original["chains"] == [0, 1]
    assert all(np.isnan(v) for v in original["final_log_marginal_likelihood"])


@pytest.mark.parametrize("terminal", [np.nan, -np.inf, np.inf], ids=["nan", "neg_inf", "pos_inf"])
def test_a_nonfinite_terminal_value_is_never_replaced_by_an_earlier_stage(monkeypatch, terminal):
    """Reviewer cases 3-5: an earlier version returned -3.3798... here. That must never happen."""
    from .count_inference import CountInferenceFailure

    with pytest.raises(CountInferenceFailure) as excinfo:
        _fit_returning(monkeypatch, _mock_idata([[-3.0, terminal], [-4.0, terminal]]))
    assert excinfo.value.cause_code == "nonfinite_final_evidence"
    assert excinfo.value.original["final_stage_index"] == [1, 1]
    assert V24C_SUBSTITUTED_LOG_EVIDENCE not in excinfo.value.original.get(
        "final_log_marginal_likelihood", []
    )


def test_a_nonfinite_posterior_is_still_its_own_named_failure(monkeypatch):
    """Reviewer case 6: valid evidence, NaN particle - a different cause_code."""
    from .count_inference import CountInferenceFailure

    with pytest.raises(CountInferenceFailure) as excinfo:
        _fit_returning(
            monkeypatch, _mock_idata([[np.nan, -3.0], [np.nan, -4.0]], posterior_nan=True)
        )
    assert excinfo.value.cause_code == "nonfinite_posterior"


def test_ragged_padding_after_the_final_stage_is_tolerated(monkeypatch):
    """A chain that finished early is padded with NaN; the terminus is still beta == 1."""
    fitted = _fit_returning(
        monkeypatch,
        _mock_idata(
            [[np.nan, -3.0, np.nan], [np.nan, np.nan, -4.0]],
            beta=((0.5, 1.0, np.nan), (0.3, 0.7, 1.0)),
        ),
    )
    assert fitted.per_chain_log_evidence == [-3.0, -4.0]


def test_a_final_stage_that_never_reached_beta_one_is_a_run_level_contract_failure(monkeypatch):
    """Tempering that did not terminate is a version/contract problem, not a verdict."""
    from .count_inference import CountInferenceFailure

    with pytest.raises(RuntimeError) as excinfo:
        _fit_returning(
            monkeypatch,
            _mock_idata([[np.nan, -3.0], [np.nan, -4.0]], beta=((0.5, 0.9), (0.5, 0.9))),
        )
    assert not isinstance(excinfo.value, CountInferenceFailure)
    assert "beta" in str(excinfo.value)


# --------------------------------------------------------------------------
# A missing `beta` is a run failure, not a
# degradation
#
# The external review of the preceding change found the one branch that
# still substituted an old value for a broken return contract: when `beta` is
# absent entirely, `_final_stage_log_evidence` fell back to the shared
# extractor's last-finite-value rule instead of failing the run, so a chain
# with SOME finite evidence was returned as if that were the terminal value -
# exactly the substitution the preceding change had just forbidden for the beta-PRESENT
# case. That fallback is now removed outright: without `beta` the terminus
# cannot be identified at all, so there is no old per-chain value this
# function is entitled to reuse, and the whole run fails instead.
# --------------------------------------------------------------------------


def test_a_beta_missing_return_with_an_earlier_finite_stage_is_a_run_level_failure(monkeypatch):
    """The earlier declared degradation returned SUCCESS here: per-chain -3.0/-4.0,
    quietly substituting the earlier finite stage for a terminus it could not
    identify. That is exactly the substitution the run-level contract forbids.
    """
    from .count_inference import CountInferenceFailure

    with pytest.raises(RuntimeError) as excinfo:
        _fit_returning(monkeypatch, _mock_idata_missing_beta([[-3.0, np.nan], [-4.0, np.nan]]))
    assert not isinstance(excinfo.value, CountInferenceFailure)
    assert "sample_stats.beta" in str(excinfo.value)
    assert "2 chain" in str(excinfo.value)
    assert "[2, 2]" in str(excinfo.value)  # stage count per chain


def test_a_beta_missing_return_with_no_ragged_stage_is_still_a_run_level_failure(monkeypatch):
    """Same counterexample without the padding column: [-3.]/[-4.] alone was
    already a quiet "success" under the old degradation. It must fail all the
    same - the absence of `beta` is decided before any per-chain value is read.
    """
    from .count_inference import CountInferenceFailure

    with pytest.raises(RuntimeError) as excinfo:
        _fit_returning(monkeypatch, _mock_idata_missing_beta([[-3.0], [-4.0]]))
    assert not isinstance(excinfo.value, CountInferenceFailure)
    assert "sample_stats.beta" in str(excinfo.value)
    assert "2 chain" in str(excinfo.value)
    assert "[1, 1]" in str(excinfo.value)


def test_a_beta_missing_return_with_all_nan_evidence_is_run_level_not_typed(monkeypatch):
    """Under the earlier fallback this ONE beta-missing shape happened to already be typed
    numerically (the "no finite evidence at any stage" safety valve inside the
    declared degradation). The whole fallback is now removed, so it is a
    run-level RuntimeError like its two siblings above, never the typed
    `CountInferenceFailure` reserved for model retirement.
    """
    from .count_inference import CountInferenceFailure

    with pytest.raises(RuntimeError) as excinfo:
        _fit_returning(monkeypatch, _mock_idata_missing_beta([[np.nan], [np.nan]]))
    assert not isinstance(excinfo.value, CountInferenceFailure)
    assert "sample_stats.beta" in str(excinfo.value)
    assert "[1, 1]" in str(excinfo.value)


def test_a_stats_mapping_failure_stays_run_level(monkeypatch):
    """Not every SMCError is numerical: a shape mapping failure is a contract error."""
    from ..inference.smc import SMCError
    from .count_inference import CountInferenceFailure

    idata = _mock_idata([[np.nan, -3.0], [np.nan, -4.0]])
    with pytest.raises(Exception) as excinfo:
        _fit_returning(
            monkeypatch,
            _mock_idata(
                [[np.nan, -3.0], [np.nan, -4.0], [np.nan, -5.0]],
                beta=((0.5, 1.0), (0.5, 1.0), (0.5, 1.0)),
            ),
        )
    assert not isinstance(excinfo.value, CountInferenceFailure)
    assert isinstance(excinfo.value, (SMCError, RuntimeError, ValueError))
    assert idata is not None


def test_the_terminal_rule_agrees_with_the_registered_extractor_on_a_real_fit():
    """On a healthy fit the new rule returns exactly what `_extract_logz` returns.

    This is the anchor-identity argument as a test: every non-terminal stage is
    NaN by construction, so "last finite" and "the beta == 1 terminus" are the
    same value - until the terminus is not finite, which is the case the terminal-stage rule fixes.
    """
    import pymc as pm

    from ..inference.smc import _extract_logz
    from .count_inference import build_model

    captured = {}
    real_sample = pm.sample_smc

    def capturing(*args, **kwargs):
        """Run the real SMC sampler and keep its InferenceData."""
        captured["idata"] = real_sample(*args, **kwargs)
        return captured["idata"]

    model = build_model(TERMINAL_HYPOTHESIS, U, Y)
    with model:
        idata = pm.sample_smc(
            draws=16,
            chains=2,
            cores=2,
            random_seed=11,
            progressbar=False,
            compute_convergence_checks=False,
        )
    logz, per_chain = _extract_logz(idata)
    result = real_fit(TERMINAL_HYPOTHESIS, U, Y, seed=11, settings=SMALL)
    assert result.per_chain_log_evidence == per_chain
    assert result.log_evidence == logz
    assert capturing is not None
