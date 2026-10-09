"""Synthetic prior/sequential Monte Carlo (SMC) hang guards and observational sidecar contracts."""
import json
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from . import count_inference as ci
from .belief_packet import task_interface
from .count_models import Hypothesis


@pytest.fixture(autouse=True)
def python_backend():
    """Test helper `python_backend`: stand-in or builder used by the surrounding tests."""
    import pytensor

    with pytensor.config.change_flags(cxx=""):
        yield


def hypothesis(expression="a", family="binomial"):
    """Test helper `hypothesis`: stand-in or builder used by the surrounding tests."""
    return Hypothesis(expression, family, (("a", "intercept"),))


def fake_idata():
    """Test helper `fake_idata`: stand-in or builder used by the surrounding tests."""
    import arviz as az

    return az.from_dict(
        posterior={"a": np.zeros((2, 16))},
        sample_stats={"log_marginal_likelihood": np.full((2, 1), -2.),
                      "beta": np.ones((2, 1))},
    )


def synthetic_warning_kernel():
    """Only the traceback filename is synthetic; no real SMC kernel runs."""
    warnings.warn("invalid value encountered in divide", RuntimeWarning, stacklevel=1)
    return 19


# The failure classifier deliberately requires an SMC frame. Give this tiny
# warning emitter that path so local and remote traceback tests exercise the
# same evidence requirement without depending on numerical kernel behavior.
synthetic_warning_kernel.__code__ = synthetic_warning_kernel.__code__.replace(
    co_filename="synthetic/pymc/smc/kernels.py")


def fake_fork_sampler(**kwargs):
    """Test helper `fake_fork_sampler`: stand-in or builder used by the surrounding tests."""
    with ProcessPoolExecutor(max_workers=kwargs["cores"],
                             mp_context=multiprocessing.get_context("fork")) as executor:
        assert executor.submit(synthetic_warning_kernel).result(timeout=15) == 19
    return fake_idata()


@pytest.mark.parametrize("values,reason,finite,nans,infs", [
    ([np.nan, 0.], "nonfinite_loglik_at_prior", .5, 1, 0),
    ([-np.inf, -np.inf], "degenerate_loglik_at_prior", 0., 0, 2),
    ([-np.inf, 0.], None, .5, 0, 1),
    ([np.inf, 0.], "nonfinite_loglik_at_prior", .5, 0, 0),
])
def test_probe_classifies_values_without_sampling(monkeypatch, values, reason, finite, nans, infs):
    """Check that probe classifies values without sampling."""
    import pymc as pm

    def forbidden(**kwargs):
        """Test helper `forbidden`: stand-in or builder used by the surrounding tests."""
        raise AssertionError("a prior logp probe must never sample SMC")

    monkeypatch.setattr(pm, "sample_smc", forbidden)
    model = SimpleNamespace(value_vars=[SimpleNamespace(name="a")],
                            compile_logp=lambda: lambda point: values[int(point["a"])])
    record = ci.loglik_prior_probe(hypothesis(), [0.2], [10], count=2,
                                  prior_draws={"a": np.arange(2)}, model_builder=lambda: model)
    assert record["passed"] is (reason is None)
    assert record["reason"] == reason
    assert record["finite_fraction"] == finite
    assert record["nan_count"] == nans and record["neg_inf_count"] == infs
    assert record["pos_inf_count"] == sum(np.isposinf(values))


def test_probe_actual_nan_model_then_finite_model_recovers():
    """Check that probe actual nan model then finite model recovers."""
    from .real_inference import RealHypothesis, build_model

    draws = {"a": np.linspace(-1., 1., 64), "sigma": np.ones(64)}
    h = RealHypothesis("log(a-u)", "normal", (("a", "intercept"),))
    interface = task_interface("real", [{"name": "u", "range": [0., 1.]}], {}, (None, None))
    bad = ci.loglik_prior_probe(
        h, [.5, .8], [20., 30.], prior_draws=draws,
        model_builder=lambda: build_model(h, [.5, .8], [20., 30.], interface, 0., 1.),
    )
    good = ci.loglik_prior_probe(hypothesis(), [.5, .8], [20, 30],
                                prior_draws={"a": draws["a"]})
    assert not bad["passed"] and bad["reason"] == "nonfinite_loglik_at_prior"
    assert bad["nan_count"] > 0
    assert good["passed"] and good["finite_fraction"] == 1.


def test_probe_actual_some_negative_infinity_is_allowed_and_logged():
    """Check that probe actual some negative infinity is allowed and logged."""
    draws = {"a": np.array([0.] * 32 + [1.] * 32)}
    record = ci.loglik_prior_probe(hypothesis("900*a"), [.2], [20], prior_draws=draws)
    assert record["passed"] and record["finite_fraction"] == .5
    assert record["neg_inf_count"] == 32 and record["nan_count"] == 0
    degenerate = ci.loglik_prior_probe(hypothesis("900*a"), [.2], [20],
                                     prior_draws={"a": np.ones(64)})
    assert not degenerate["passed"]
    assert degenerate["reason"] == "degenerate_loglik_at_prior"


@pytest.mark.parametrize("expression", ["a+u**3", "a+exp(u)"])
def test_raw_input_overflow_rejects_and_standardization_passes(expression):
    """Check that raw input overflow rejects and standardization passes."""
    from .positive_inference import ScalarHypothesis, build_model

    interface = task_interface("unbounded_count", [{"name": "u", "range": [1., 365.]}],
                               {"z": "(u-183)/182"}, (0, None))
    inputs, outcomes = [180., 365.], [2., 4.]
    draws = {"a": np.random.default_rng(427).normal(0., 2., 64)}
    records = []
    for expr in (expression, expression.replace("u", "z")):
        h = ScalarHypothesis(expr, "poisson", (("a", "intercept"),))
        records.append(ci.loglik_prior_probe(
            h, inputs, outcomes, interface, prior_draws=draws,
            model_builder=lambda h=h: build_model(h, inputs, outcomes, None, 1., interface),
        ))
    assert not records[0]["passed"]
    assert records[0]["reason"] in ("nonfinite_loglik_at_prior", "degenerate_loglik_at_prior")
    assert records[0]["finite_fraction"] == 0.
    assert records[1]["passed"] and records[1]["finite_fraction"] == 1.


def test_probe_reuses_exact_runner_prior_points_and_does_not_shift_rng():
    """Check that probe reuses exact runner prior points and does not shift rng."""
    from .core_count import prior_particles

    h = hypothesis()
    draws = prior_particles(h, 64)
    state = np.random.get_state()
    result = ci.loglik_prior_probe(h, [.2, .8], [20, 30], prior_draws=draws)
    expected = ci.build_model(h, [.2, .8], [20, 30]).compile_logp()
    assert result["finite_fraction"] == np.mean([
        np.isfinite(expected({"a": value})) for value in draws["a"]])
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    for name, values in prior_particles(h, 64).items():
        np.testing.assert_array_equal(values, draws[name])


def test_probe_default_draws_is_seeded_and_accepts_actual_model():
    """Check that probe default draws is seeded and accepts actual model."""
    first = ci.loglik_prior_probe(hypothesis(), [.2, .8], [20, 30], seed=591)
    second = ci.loglik_prior_probe(hypothesis(), [.2, .8], [20, 30], seed=591)
    assert first == second and first["passed"] and first["n_samples"] == 64


@pytest.mark.parametrize("failure", [TypeError("compiler contract"), OSError("backend")])
def test_probe_does_not_reclassify_infrastructure_or_programming_errors(failure):
    """Check that probe does not reclassify infrastructure or programming errors."""
    def builder():
        """Test helper `builder`: stand-in or builder used by the surrounding tests."""
        raise failure

    with pytest.raises(type(failure), match=str(failure)):
        ci.loglik_prior_probe(hypothesis(), [.2], [10], model_builder=builder)


@pytest.mark.parametrize("cores", [1, 2])
def test_promoted_warning_becomes_typed_failure_and_filter_is_restored(monkeypatch, cores):
    """Check that promoted warning becomes typed failure and filter is restored."""
    import pymc as pm

    def sample(**kwargs):
        """Test helper `sample`: stand-in or builder used by the surrounding tests."""
        synthetic_warning_kernel()
        return fake_idata()

    monkeypatch.setattr(ci, "build_model", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(pm, "sample_smc", sample)
    filters = warnings.filters.copy()
    with pytest.raises(ci.CountInferenceFailure) as caught:
        ci.fit(hypothesis(), [.2], [10], 42, ci.SMCSettings(16, 2, cores), fit_guards="v1")
    assert caught.value.cause_code == "nonfinite_smc_weights"
    assert caught.value.stage == "sampling"
    assert "pymc/smc/kernels.py" in caught.value.original["traceback"]
    assert warnings.filters == filters
    with pytest.warns(RuntimeWarning, match="invalid value encountered"):
        healthy = ci.fit(hypothesis(), [.2], [10], 42, ci.SMCSettings(16, 2, cores))
    assert np.isfinite(healthy.log_evidence)
    monkeypatch.setattr(pm, "sample_smc", lambda **kwargs: fake_idata())
    assert np.isfinite(ci.fit(hypothesis(), [.2], [10], 42,
                             ci.SMCSettings(16, 2, cores), fit_guards="v1").log_evidence)


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="fork required")
def test_actual_fork_worker_inherits_filter_and_keeps_remote_traceback(monkeypatch):
    """Check that actual fork worker inherits filter and keeps remote traceback."""
    import pymc as pm

    monkeypatch.setattr(ci, "build_model", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(pm, "sample_smc", fake_fork_sampler)
    with pytest.raises(ci.CountInferenceFailure) as caught:
        ci.fit(hypothesis(), [.2], [10], 42, ci.SMCSettings(16, 2, 2), fit_guards="v1")
    assert caught.value.cause_code == "nonfinite_smc_weights"
    assert "RemoteTraceback" in caught.value.original["traceback"]
    assert "synthetic/pymc/smc/kernels.py" in caught.value.original["traceback"]
    # A following call with guards disabled creates a fresh fork whose warning
    # remains a warning. The child, not the pytest parent, emits it.
    healthy = ci.fit(hypothesis(), [.2], [10], 42, ci.SMCSettings(16, 2, 2))
    assert np.isfinite(healthy.log_evidence)


def test_same_warning_outside_smc_is_not_a_model_retirement():
    """Check that same warning outside smc is not a model retirement."""
    def outside():
        """Test helper `outside`: stand-in or builder used by the surrounding tests."""
        warnings.warn("invalid value encountered in divide", RuntimeWarning, stacklevel=1)

    with pytest.warns(RuntimeWarning, match="invalid value encountered"):
        ci.run_with_fit_guards(outside, fit_guards="v1")


def test_sampling_filter_does_not_promote_unrelated_warning():
    """Check that sampling filter does not promote unrelated warning."""
    def unrelated():
        """Test helper `unrelated`: stand-in or builder used by the surrounding tests."""
        warnings.warn("unrelated numerical warning", RuntimeWarning, stacklevel=1)
        return 4

    with pytest.warns(RuntimeWarning, match="unrelated"):
        assert ci.run_with_fit_guards(unrelated, fit_guards="v1") == 4


def test_guarded_native_fit_with_timeout_completes_real_16_by_2_smc():
    """Check that guarded native fit with timeout completes real 16 by 2 smc."""
    result = ci.fit(hypothesis(), [.2, .8], [19, 29], 745,
                    ci.SMCSettings(16, 2, 2), fit_timeout_seconds=60, fit_guards="v1")
    assert np.isfinite(result.log_evidence)
    assert np.asarray(result.particles["a"]).shape == (32,)
    assert len(result.per_chain_log_evidence) == 2
    assert result.diagnostics["chains"] == 2


def test_off_wrapper_is_exact_direct_call_and_does_not_install_filter(monkeypatch):
    """Check that off wrapper is exact direct call and does not install filter."""
    sentinel = object()
    called = []

    def receiver(*args, **kwargs):
        """Test helper `receiver`: stand-in or builder used by the surrounding tests."""
        called.append((args, kwargs))
        return sentinel

    def forbidden(*args, **kwargs):
        """Test helper `forbidden`: stand-in or builder used by the surrounding tests."""
        raise AssertionError("off must not install a warning context")

    monkeypatch.setattr(warnings, "catch_warnings", forbidden)
    assert ci.run_with_fit_guards(receiver, 1, word=2) is sentinel
    assert called == [((1,), {"word": 2})]


def test_acquisition_timing_appends_rounds_without_touching_scientific_record(tmp_path):
    """Check that acquisition timing appends rounds without touching scientific record."""
    acquisition = tmp_path / "acquisition.json"
    payload = b'{"chosen_design":0.25}'
    acquisition.write_bytes(payload)
    for t in (1, 2, 3):
        ci.record_acquisition_timing(tmp_path, t, "epig", t / 10., 21)
    rows = [json.loads(line) for line in (tmp_path / "acquisition_timings.jsonl").read_text().splitlines()]
    assert [row["round"] for row in rows] == [1, 2, 3]
    assert all(row["design_policy"] == "epig" and row["elapsed_seconds"] >= 0
               and row["n_candidates"] == 21 for row in rows)
    assert acquisition.read_bytes() == payload


def test_prediction_summary_identifies_dominant_model_and_q_index(tmp_path):
    """Check that prediction summary identifies dominant model and q index."""
    models = [
        {"model_id": "ordinary", "weight": .8, "means": np.array([1., 2., 3.])},
        {"model_id": "large-mean", "weight": .2, "means": np.array([2., 1e90, 2.])},
        {"model_id": "excluded", "weight": 0., "excluded": True},
    ]
    ci.record_prediction_pool_summary(tmp_path, 7, "official", 11, models)
    row = json.loads((tmp_path / "prediction_pool_summary.jsonl").read_text())
    assert row["round"] == 7 and row["checkpoint"] == "official"
    assert row["observation_count"] == 11
    assert row["models"][1] == {"model_id": "large-mean", "weight": .2,
                                 "mean_max": 1e90, "mean_argmax_q_index": 1,
                                 "mean_median": 2.}
    assert row["models"][2]["excluded"] and row["models"][2]["mean_max"] is None


def test_prediction_summary_preserves_partial_exclusion_and_original_q_index(tmp_path):
    """Check that prediction summary preserves partial exclusion and original q index."""
    models = [{"model_id": "partial", "weight": .5, "excluded": True,
               "means": [np.nan, 5., 19., 2.], "excluded_q_indices": [0],
               "weights_by_q": [0., .7, .8, .9]}]
    ci.record_prediction_pool_summary(tmp_path, 2, "official", 6, models)
    row = json.loads((tmp_path / "prediction_pool_summary.jsonl").read_text())["models"][0]
    assert row["mean_argmax_q_index"] == 2 and row["mean_max"] == 19.
    assert row["mean_median"] == 5. and row["mean_summary_scope"] == "retained_q"
    assert row["excluded_q_indices"] == [0] and row["weights_by_q"] == [0., .7, .8, .9]


def test_nonfinite_retained_means_are_explicit_not_silently_filtered(tmp_path):
    """Check that nonfinite retained means are explicit not silently filtered."""
    ci.record_prediction_pool_summary(tmp_path, 0, "prediction", 4, [
        {"model_id": "bad", "weight": 1., "means": [1., np.inf]},
    ])
    row = json.loads((tmp_path / "prediction_pool_summary.jsonl").read_text())["models"][0]
    assert row["mean_max"] is None and row["mean_argmax_q_index"] is None
    assert row["nonfinite_mean_count"] == 1


def test_prediction_summary_large_finite_median_does_not_overflow(tmp_path):
    """Check that prediction summary large finite median does not overflow."""
    ci.record_prediction_pool_summary(tmp_path, 1, "official", 5, [
        {"model_id": "finite-extreme", "component": 2, "weight": 1.,
         "means": [1e308, 1.5e308]},
    ])
    row = json.loads((tmp_path / "prediction_pool_summary.jsonl").read_text())["models"][0]
    assert row["mean_median"] == 1.25e308 and row["mean_max"] == 1.5e308
    assert row["component"] == 2


def _runner_case(monkeypatch, output, bundle, *, guards="v1", rounds=2, policy="random"):
    from .count_models import Hypothesis
    from .test_count_proposer import run_args, stub_fits
    from .test_named_inputs import bg_run_args, bg_stub_fits

    if bundle == "core":
        module = stub_fits(monkeypatch)
        victim = Hypothesis("b0+b1*square(u)", "binomial",
                            (("b0", "intercept"), ("b1", "signed")))
        monkeypatch.setattr(module, "mock_proposal_p1", lambda *args: [victim])
        cell = "112" if policy == "epig" else "111" if policy == "joint_eig" else "110"
        args = run_args(output, cell=cell, rounds=rounds,
                        design_policy="joint_eig" if policy == "random" else policy, fit_guards=guards)
        fit_name = "fit"
    else:
        module = bg_stub_fits(monkeypatch)
        expression = "b0+b1*square(t01)"
        victim = Hypothesis(expression, "binomial", module.mock_parameters(expression),
                            ("t", "t01", "tz"))
        monkeypatch.setattr(module.TaskSpec, "mock_templates", lambda self: (expression,))
        cell = "102" if policy == "epig" else "101" if policy == "joint_eig" else "100"
        args = bg_run_args(output, cell=cell, rounds=rounds, fit_guards=guards,
                           **({"design_policy": "epig"} if policy == "epig" else {}))
        fit_name = "count_fit"
    return module, args, victim, fit_name


@pytest.mark.parametrize("bundle", ["core", "boxinggym"])
def test_candidate_source_is_durable_before_fit_even_on_interruption(tmp_path, monkeypatch, bundle):
    """Check that candidate source is durable before fit even on interruption."""
    if bundle == "boxinggym":
        pytest.importorskip("boxing_gym.envs.death_process")
    output = tmp_path / bundle
    module, args, victim, fit_name = _runner_case(monkeypatch, output, bundle, guards="off")
    original = getattr(module, fit_name)

    def interrupted(hypothesis, *arguments, **kwargs):
        """Test helper `interrupted`: stand-in or builder used by the surrounding tests."""
        if hypothesis.expression == victim.expression:
            proposal = json.loads((output / "proposals" / "t01.json").read_text())
            assert proposal["status"] == "proposed"
            assert proposal["raw_response"][0]["expression"] == hypothesis.expression
            assert "transport" in proposal and "fit_results" not in proposal
            raise KeyboardInterrupt("synthetic interruption during candidate fit")
        return original(hypothesis, *arguments, **kwargs)

    monkeypatch.setattr(module, fit_name, interrupted)
    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        module.run(args)
    saved = json.loads((output / "proposals" / "t01.json").read_text())
    assert saved["status"] == "proposed"
    assert saved["raw_response"][0]["expression"] == victim.expression


@pytest.mark.parametrize("bundle", ["core", "boxinggym"])
@pytest.mark.parametrize("reason,nans,neg_inf", [
    ("nonfinite_loglik_at_prior", 1, 0), ("degenerate_loglik_at_prior", 0, 64)])
def test_candidate_loglik_rejection_is_recorded_once_and_cell_continues(
        tmp_path, monkeypatch, bundle, reason, nans, neg_inf):
    """Check that candidate loglik rejection is recorded once and cell continues."""
    if bundle == "boxinggym":
        pytest.importorskip("boxing_gym.envs.death_process")
    output = tmp_path / bundle
    module, args, victim, fit_name = _runner_case(monkeypatch, output, bundle)
    original = getattr(module, fit_name)
    probe_calls, fit_calls = [], []

    def rejected(hypothesis, x, y, interface, **kwargs):
        """Test helper `rejected`: stand-in or builder used by the surrounding tests."""
        proposal = json.loads((output / "proposals" / "t01.json").read_text())
        assert proposal["status"] == "proposed"
        assert len(y) == 5  # Only public D0 plus the just-revealed observation.
        assert all(len(draws) == 64 for draws in kwargs["prior_draws"].values())
        probe_calls.append(hypothesis.expression)
        return {"passed": False, "reason": reason, "n_samples": 64,
                "finite_fraction": (64 - nans - neg_inf) / 64,
                "nan_count": nans, "neg_inf_count": neg_inf, "pos_inf_count": 0}

    def recording(hypothesis, *arguments, **kwargs):
        """Test helper `recording`: stand-in or builder used by the surrounding tests."""
        fit_calls.append(hypothesis.expression)
        return original(hypothesis, *arguments, **kwargs)

    monkeypatch.setattr(module, "loglik_prior_probe", rejected)
    monkeypatch.setattr(module, fit_name, recording)
    module.run(args)
    assert probe_calls == [victim.expression]  # No initial or duplicate probes.
    assert victim.expression not in fit_calls
    first = json.loads((output / "proposals" / "t01.json").read_text())
    second = json.loads((output / "proposals" / "t02.json").read_text())
    assert first["events"][0]["status"] == "rejected_loglik_prior_probe"
    assert first["events"][0]["prior_probe"]["loglik_prior_probe"]["reason"] == reason
    assert second["events"][0]["status"] == "duplicate_not_refitted"
    assert json.loads((output / "summary.json").read_text())["status"] == "completed"
    archives = json.loads((output / "archive.json").read_text())
    entries = archives if isinstance(archives, list) else archives.values()
    assert any(row.get("outcome") == "rejected" and
               row["reason"]["validation_message"] == reason for row in entries)


@pytest.mark.parametrize("bundle", ["core", "boxinggym"])
def test_smc_warning_retirement_continues_to_the_next_round(tmp_path, monkeypatch, bundle):
    """Check that smc warning retirement continues to the next round."""
    if bundle == "boxinggym":
        pytest.importorskip("boxing_gym.envs.death_process")
    output = tmp_path / bundle
    module, args, victim, fit_name = _runner_case(monkeypatch, output, bundle)
    original = getattr(module, fit_name)
    attempts = []

    def injected(hypothesis, *arguments, **kwargs):
        """Test helper `injected`: stand-in or builder used by the surrounding tests."""
        attempts.append(hypothesis.expression)
        if hypothesis.expression == victim.expression:
            raise ci.CountInferenceFailure("nonfinite_smc_weights", "sampling", "synthetic warning")
        return original(hypothesis, *arguments, **kwargs)

    monkeypatch.setattr(module, fit_name, injected)
    module.run(args)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert attempts.count(victim.expression) == 1
    ledger = summary["refit_failed_models"] if bundle == "core" else summary["candidate_failures"]
    assert any(row["retirement_kind"] == "nonfinite_smc_weights" for row in ledger)
    assert json.loads((output / "proposals" / "t02.json").read_text())["events"][0]["status"] == "duplicate_not_refitted"


@pytest.mark.parametrize("bundle", ["core", "boxinggym"])
@pytest.mark.parametrize("policy", ["random", "joint_eig", "epig"])
def test_sidecars_cover_every_round_and_score_without_changing_acquisition_fields(
        tmp_path, monkeypatch, bundle, policy):
    """Check that sidecars cover every round and score without changing acquisition fields."""
    if bundle == "boxinggym":
        pytest.importorskip("boxing_gym.envs.death_process")
    output = tmp_path / bundle
    module, args, _victim, _fit_name = _runner_case(
        monkeypatch, output, bundle, guards="off", policy=policy)

    def must_not_probe(*args, **kwargs):
        """Test helper `must_not_probe`: stand-in or builder used by the surrounding tests."""
        raise AssertionError("disabled guards must not probe log likelihoods")

    monkeypatch.setattr(module, "loglik_prior_probe", must_not_probe)
    module.run(args)
    timings = [json.loads(row) for row in (output / "acquisition_timings.jsonl").read_text().splitlines()]
    assert [row["round"] for row in timings] == [1, 2]
    assert all(row["elapsed_seconds"] >= 0 and row["n_candidates"] > 0 for row in timings)
    assert all(row["design_policy"] == policy for row in timings)
    scores = (output / "scores.jsonl").read_text().splitlines()
    summaries = [json.loads(row) for row in (output / "prediction_pool_summary.jsonl").read_text().splitlines()]
    assert len(summaries) == len(scores)
    assert [r["round"] for r in summaries] == [json.loads(r)["round"] for r in scores]
    for row in summaries:
        assert row["checkpoint"] == "official"
        assert sum(model["weight"] for model in row["models"]) == pytest.approx(1)
        assert all(model["mean_max"] >= model["mean_median"] for model in row["models"])
    for path in (output / "acquisition").glob("*.json"):
        assert "elapsed_seconds" not in json.loads(path.read_text())
    assert "fit_guards" not in json.loads((output / "config.json").read_text())
    assert "status" not in json.loads((output / "proposals" / "t01.json").read_text())


def test_atomic_proposal_finalization_preserves_json_bytes_and_refuses_completed_files(tmp_path):
    """Check that atomic proposal finalization preserves json bytes and refuses completed files."""
    from .core_count import write_final_proposal_json, write_json, write_proposed_json
    target, reference = tmp_path / "proposal.json", tmp_path / "reference.json"
    proposal = {"raw_response": [{"expression": "a+b*x"}], "transport": {"mock": True}}
    final = {**proposal, "events": [{"status": "fitted"}], "values": np.array([1., 2.])}
    write_proposed_json(target, proposal)
    write_final_proposal_json(target, final)
    write_json(reference, final)
    assert target.read_bytes() == reference.read_bytes()
    with pytest.raises(FileExistsError):
        write_final_proposal_json(target, {"status": "changed"})
    with pytest.raises(FileExistsError):
        write_proposed_json(target, proposal)
    assert target.read_bytes() == reference.read_bytes()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["proposal.json", "reference.json"]


def test_smc_filename_in_warning_message_is_not_frame_evidence():
    """Check that smc filename in warning message is not frame evidence."""
    def caller_warning():
        """Test helper `caller_warning`: stand-in or builder used by the surrounding tests."""
        warnings.warn("invalid value encountered while describing pymc/smc/kernels.py",
                      RuntimeWarning, stacklevel=1)

    with pytest.warns(RuntimeWarning, match="invalid value encountered"):
        ci.run_with_fit_guards(caller_warning, fit_guards="v1")
    try:
        raise RuntimeWarning("invalid value encountered while describing pymc/smc/kernels.py")
    except RuntimeWarning as exc:
        assert ci.classify_inference_failure(exc, "sampling") is None


def test_timeout_remote_warning_keeps_actual_smc_frame_classification(monkeypatch):
    """Check that timeout remote warning keeps actual smc frame classification."""
    import pymc as pm

    monkeypatch.setattr(pm, "sample_smc", synthetic_warning_kernel)
    with pytest.raises(ci.CountInferenceFailure) as caught:
        ci.run_with_fit_guards(ci.run_with_fit_timeout, lambda: pm.sample_smc(),
                               fit_timeout_seconds=10, fit_guards="v1")
    assert caught.value.cause_code == "nonfinite_smc_weights"
    assert caught.value.stage == "sampling"
    assert "synthetic/pymc/smc/kernels.py" in caught.value.original["traceback"]
    assert "isolated fit traceback" in str(caught.value.__cause__)


def test_generic_fit_only_promotes_sampler_warning_and_restores_sampler(monkeypatch):
    """Check that generic fit only promotes sampler warning and restores sampler."""
    import pymc as pm

    original_sampler = lambda **kwargs: 37
    monkeypatch.setattr(pm, "sample_smc", original_sampler)
    filters = warnings.filters.copy()

    def adapter():
        """Test helper `adapter`: stand-in or builder used by the surrounding tests."""
        warnings.warn("invalid value encountered in compile-time log", RuntimeWarning, stacklevel=1)
        return pm.sample_smc()

    with pytest.warns(RuntimeWarning, match="compile-time"):
        assert ci.run_with_fit_guards(adapter, fit_guards="v1") == 37
    assert pm.sample_smc is original_sampler and warnings.filters == filters


def test_generic_fit_nested_interception_restores_after_error(monkeypatch):
    """Check that generic fit nested interception restores after error."""
    import pymc as pm

    original_sampler = synthetic_warning_kernel
    monkeypatch.setattr(pm, "sample_smc", original_sampler)
    filters = warnings.filters.copy()
    enclosing = []

    def outer():
        """Test helper `outer`: stand-in or builder used by the surrounding tests."""
        enclosing.append(pm.sample_smc)
        try:
            return ci.run_with_fit_guards(lambda: pm.sample_smc(), fit_guards="v1")
        finally:
            assert pm.sample_smc is enclosing[0]

    with pytest.raises(ci.CountInferenceFailure) as caught:
        ci.run_with_fit_guards(outer, fit_guards="v1")
    assert caught.value.cause_code == "nonfinite_smc_weights"
    assert pm.sample_smc is original_sampler and warnings.filters == filters
    monkeypatch.setattr(pm, "sample_smc", lambda: 41)
    assert ci.run_with_fit_guards(lambda: pm.sample_smc(), fit_guards="v1") == 41


def test_generic_fit_restores_sampler_on_non_sampling_failure(monkeypatch):
    """Check that generic fit restores sampler on non sampling failure."""
    import pymc as pm

    original_sampler = lambda: 9
    monkeypatch.setattr(pm, "sample_smc", original_sampler)

    def bad_compile():
        """Test helper `bad_compile`: stand-in or builder used by the surrounding tests."""
        raise TypeError("synthetic compile contract")

    with pytest.raises(TypeError, match="compile contract"):
        ci.run_with_fit_guards(bad_compile, fit_guards="v1")
    assert pm.sample_smc is original_sampler


@pytest.mark.parametrize("invalid", (np.nan, np.inf, -np.inf))
def test_neuron_nonfinite_forward_is_a_probe_rejection_without_smc(monkeypatch, invalid):
    """Check that neuron nonfinite forward is a probe rejection without smc."""
    import pymc as pm

    from . import neuron_core as neuron

    monkeypatch.setattr(pm, "sample_smc", lambda *a, **k: pytest.fail("probe invoked SMC"))
    monkeypatch.setattr(neuron, "predict_features", lambda *a: np.full(3, invalid))
    result = neuron._neuron_loglik_probe(
        SimpleNamespace(model_id="synthetic-nonfinite-forward"), [.25, .75], np.ones((2, 3)))
    assert result["passed"] is False
    assert result["reason"] == "nonfinite_loglik_at_prior"
    assert result["n_samples"] == result["nan_count"] == 64
    assert result["finite_fraction"] == 0.
    assert result["neg_inf_count"] == result["pos_inf_count"] == 0


def test_neuron_prior_probe_wrong_forward_shape_remains_a_program_error(monkeypatch):
    """Check that neuron prior probe wrong forward shape remains a program error."""
    from . import neuron_core as neuron

    monkeypatch.setattr(neuron, "predict_features", lambda *a: np.ones(4))
    with pytest.raises(ValueError, match="invalid forward shape"):
        neuron._neuron_loglik_probe(
            SimpleNamespace(model_id="synthetic-wrong-shape"), [.25, .75], np.ones((2, 3)))


def test_neuron_prior_probe_forward_oserror_propagates(monkeypatch):
    """Check that neuron prior probe forward oserror propagates."""
    from . import neuron_core as neuron

    def unavailable(*args):
        """Test helper `unavailable`: stand-in or builder used by the surrounding tests."""
        raise OSError("synthetic forward infrastructure failure")

    monkeypatch.setattr(neuron, "predict_features", unavailable)
    with pytest.raises(OSError, match="synthetic forward infrastructure failure"):
        neuron._neuron_loglik_probe(
            SimpleNamespace(model_id="synthetic-forward-oserror"), [.25], np.ones((1, 3)))


@pytest.mark.parametrize("invalid", (np.nan, np.inf))
def test_neuron_actual_probe_rejects_candidate_and_cell_continues(
        tmp_path, monkeypatch, request, invalid):
    """Check that neuron actual probe rejects candidate and cell continues."""
    from . import neuron_core as neuron
    from .test_neuron_core_cli import transport_world

    request.getfixturevalue("require_neuronbench_source")
    transport_world.__wrapped__(monkeypatch)
    initial_ids = {h.model_id for h in neuron.initial_hypotheses()}
    native_forward, fake_fit = neuron.predict_features, neuron.fit
    attempted = []

    def forward(h, *args):
        """Test helper `forward`: stand-in or builder used by the surrounding tests."""
        return native_forward(h, *args) if h.model_id in initial_ids else np.full(3, invalid)

    def fit(h, *args, **kwargs):
        """Test helper `fit`: stand-in or builder used by the surrounding tests."""
        attempted.append(h.model_id)
        assert h.model_id in initial_ids, "invalid candidate reached fit"
        return fake_fit(h, *args, **kwargs)

    monkeypatch.setattr(neuron, "predict_features", forward)
    monkeypatch.setattr(neuron, "fit", fit)
    out = tmp_path / "nonfinite-candidates"
    neuron.run(neuron.build_parser().parse_args([
        "--output", str(out), "--cell", "100", "--rounds", "2", "--fit-guards", "v1"]))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "completed_deterministic_development"
    assert summary["active_count"] == 4 and summary["candidate_failures"] == []
    assert set(attempted) == initial_ids
    proposals = [json.loads(line) for line in (out / "proposals.jsonl").read_text().splitlines()]
    probes = [probe for record in proposals for probe in record["loglik_prior_probes"].values()]
    assert len(probes) == 3
    assert all(not p["passed"] and p["nan_count"] == 64 for p in probes)
    archive = json.loads((out / "archive.json").read_text())
    rejected = [entry for entry in archive if entry["outcome"] == "rejected"]
    assert len(rejected) == 3
    assert all(entry["reason"]["validation_message"] == "nonfinite_loglik_at_prior" for entry in rejected)
