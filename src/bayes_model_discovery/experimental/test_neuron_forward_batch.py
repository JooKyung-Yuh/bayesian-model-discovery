"""The particle-vectorised HH forward, its equivalence report, and the batch sequential Monte Carlo (SMC) kernel.

Every claim here is an EQUALITY claim - the batch path must return what the scalar path returns,
bit for bit - so nothing in this file has a tolerance. Bit identity holds only in an interpreter
whose numpy does not dispatch `power` to a SIMD implementation (see `neuron_forward_batch`'s
module docstring); `power_matches_scalar` decides that at run time, and the tests that need it
SKIP, with the environment variable spelled out, when numpy reports AVX-512 enabled and the
self-check fails. On a machine with no AVX-512 at all a failing self-check is a real defect, not
a missing environment variable, so there the tests run and are allowed to fail.

The two SMC tests drive real `pm.sample_smc` fits (16 particles, two short synthetic designs)
and cost roughly a minute each; they are module-scoped so the four fits happen once.
"""

import hashlib
import json

import numpy as np
import pytest

from ..provenance.canonical import canonical_json, is_hex64, sha256_file
from . import neuron_forward_batch as nfb
from .count_inference import SMCSettings
from .neuron_forward import (
    HHHypothesis,
    clear_prefill,
    native_test_counts,
    predict_features,
    prefill_stats,
)
from .vector_inference import BatchIMH, fit, prediction_bank

WORLD_WITH_TEST_PROTOCOLS = "h_sag"
# Two short synthetic probes (~10,500 samples each once build_I adds its fixed 20 ms base and
# 80 ms tail), so the SMC fixtures below stay in the "minute", not the "hour", range.
SYNTHETIC_DESIGNS = [("synthetic probe A", [(5.0, 12.0)]), ("synthetic probe B", [(8.0, 6.0)])]
SYNTHETIC_MECHANISM = "leak"
SYNTHETIC_TRUTH = 1.3
SMC_SEED = 4321


def _bit_identity_skip_reason():
    """Why a bit-identity test cannot run here, or "" when it can."""
    from numpy.core._multiarray_umath import __cpu_features__

    if nfb.power_matches_scalar():
        return ""
    if not __cpu_features__.get("AVX512F"):
        # No AVX-512 to disable: a mismatch is a genuine defect, so run and let it fail.
        return ""
    return (
        "numpy dispatches power to AVX-512 here, which differs from the scalar libm pow the "
        'benchmark uses; run with NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_KNL '
        'AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"'
    )


_SKIP_REASON = _bit_identity_skip_reason()
needs_scalar_power = pytest.mark.skipif(bool(_SKIP_REASON), reason=_SKIP_REASON)


def _assert_bit_identical(mechanism, protocols, scales):
    """One batched call per protocol against one scalar `_trace` per (protocol, multiplier)."""
    from neuronbench.worlds import build_I

    from .neuron_forward import _trace

    pairs = 0
    for _label, segments in protocols:
        key = tuple((float(duration), float(amplitude)) for duration, amplitude in segments)
        current, start = build_I(key)
        voltage, spikes = nfb.simulate_batch(current, start, mechanism, np.asarray(scales))
        for index, scale in enumerate(scales):
            reference = np.asarray(_trace(mechanism, float(scale), key))
            batched = np.ascontiguousarray(voltage[index])
            assert np.array_equal(batched.view(np.uint64), reference.view(np.uint64)), (
                mechanism, _label, scale)
            assert int(spikes[index]) == int(native_test_counts(reference, start)[0])
            pairs += 1
        _trace.cache_clear()
    return pairs


# The grid is split across the three tests below rather than run as 10 mechanisms x 9 pool
# protocols x 3 multipliers in each: the batch integrator costs ~150 us per step whatever the
# particle count, so that product is about 12 minutes of test time, almost all of it re-proving
# the same shared arithmetic. What varies BY MECHANISM is the kwargs mapping, which one short
# protocol settles; what varies BY PROTOCOL is the integrator itself (the conditioning
# pre-pulse, the moved test window, the paired-pulse gap), which one mechanism settles. The
# full 10 x 37 x 3 grid is the equivalence report's job, and the report is generated - and
# checked into the run's config.json by digest - on every machine that runs the batch backend.


@needs_scalar_power
@pytest.mark.parametrize("mechanism", nfb.KINDS)
@pytest.mark.usefixtures("require_neuronbench_source")
def test_batch_traces_are_bit_identical_for_every_mechanism(mechanism):
    """Every catalogue mechanism's conductance mapping, on the shortest pool protocol."""
    from neuronbench.worlds import POOL

    assert _assert_bit_identical(mechanism, POOL[:1], nfb.SCALES) == len(nfb.SCALES)


@needs_scalar_power
@pytest.mark.usefixtures("require_neuronbench_source")
def test_batch_traces_are_bit_identical_over_the_whole_design_pool():
    """The nine pool protocols on the richest path: an inactivating extra channel (mpow 2)."""
    from neuronbench.worlds import POOL

    assert _assert_bit_identical("rebound_inward_slow", POOL, nfb.SCALES) == (
        len(POOL) * len(nfb.SCALES)
    )


@needs_scalar_power
@pytest.mark.usefixtures("require_neuronbench_source")
def test_batch_traces_are_bit_identical_on_a_world_test_protocol_set():
    """The held-out protocols the scoring block integrates - not in POOL, and their
    conditioning pre-pulse moves the test window, hence the spike count - on the other
    branch-carrying path (slow Na inactivation's voltage-dependent time constant)."""
    from neuronbench.worlds import WORLDS

    protocols = WORLDS[WORLD_WITH_TEST_PROTOCOLS]["test"]
    assert _assert_bit_identical("slow_na", protocols, nfb.SCALES) == (
        len(protocols) * len(nfb.SCALES)
    )


@needs_scalar_power
@pytest.mark.usefixtures("require_neuronbench_source")
def test_predict_batch_and_predict_features_batch_equal_the_per_particle_calls():
    """Check that predict batch and predict features batch equal the per particle calls."""
    hypothesis = HHHypothesis("h_current")
    design = ("synthetic conditioning probe", [(15.0, -30.0), (10.0, 12.0)])
    thetas = [0.4, 1.0, 2.5]
    traces = nfb.predict_batch(hypothesis, thetas, design)
    features = nfb.predict_features_batch(hypothesis, thetas, design)
    assert traces.shape[0] == features.shape[0] == len(thetas)
    assert features.shape[1] == 3
    for index, theta in enumerate(thetas):
        reference = hypothesis.predict([theta], design)
        assert np.array_equal(
            np.ascontiguousarray(traces[index]).view(np.uint64), reference.view(np.uint64)
        )
        assert np.array_equal(
            features[index].view(np.uint64),
            predict_features(hypothesis, [theta], design).view(np.uint64),
        )


def test_simulate_batch_refuses_unknown_mechanisms_and_invalid_multipliers():
    """Check that simulate batch refuses unknown mechanisms and invalid multipliers."""
    pytest.importorskip("neuronbench")
    current = np.zeros(4)
    with pytest.raises(ValueError, match="unknown candidate mechanism"):
        nfb.simulate_batch(current, 0, "not_a_mechanism", np.array([1.0]))
    for bad in (np.array([0.0]), np.array([-1.0]), np.array([np.nan])):
        with pytest.raises(ValueError, match="finite and positive"):
            nfb.simulate_batch(current, 0, "leak", bad)
    with pytest.raises(ValueError, match="non-empty"):
        nfb.simulate_batch(current, 0, "leak", np.array([]))


# ----------------------------------------------------------------- SMC equivalence


def _synthetic_problem():
    hypothesis = HHHypothesis(SYNTHETIC_MECHANISM)
    values = np.asarray(
        [predict_features(hypothesis, [SYNTHETIC_TRUTH], design) for design in SYNTHETIC_DESIGNS]
    )
    return hypothesis, values


def _native_and_batch_fits(chains):
    hypothesis, values = _synthetic_problem()
    settings = SMCSettings(16, chains, chains)
    clear_prefill()
    native = fit(hypothesis, SYNTHETIC_DESIGNS, values, predict_features, SMC_SEED, settings)
    batch = fit(hypothesis, SYNTHETIC_DESIGNS, values, predict_features, SMC_SEED, settings,
                forward_backend="batch")
    return native, batch


@pytest.fixture(scope="module")
def single_chain_fits():
    """One native and one batch fit of the same synthetic problem at the same seed."""
    pytest.importorskip("neuronbench")
    return _native_and_batch_fits(1)


@pytest.fixture(scope="module")
def two_chain_fits():
    """The same pair with two chains on two cores: the kernel subclass really is cloudpickled
    into worker processes, and each worker keeps its own prefill cache and miss counter."""
    pytest.importorskip("neuronbench")
    return _native_and_batch_fits(2)


def _assert_same_posterior(native, batch, chains):
    for name in ("conductance", "sigma"):
        assert np.array_equal(native.particles[name], batch.particles[name]), name
    assert batch.log_evidence == native.log_evidence
    assert batch.per_chain_log_evidence == native.per_chain_log_evidence
    assert len(batch.per_chain_log_evidence) == chains
    assert batch.diagnostics["forward_backend"] == "batch"
    assert batch.diagnostics["forward_cache_misses"] == 0
    # The native fit's diagnostics are untouched; the batch fit's are those two keys more.
    assert "forward_backend" not in native.diagnostics
    assert "forward_cache_misses" not in native.diagnostics
    assert set(batch.diagnostics) == set(native.diagnostics) | {
        "forward_backend", "forward_cache_misses",
    }
    assert native.diagnostics["beta_schedule"] == batch.diagnostics["beta_schedule"]
    assert native.diagnostics["accept_rate"] == batch.diagnostics["accept_rate"]


@needs_scalar_power
def test_batch_smc_reproduces_the_native_posterior_exactly(single_chain_fits):
    """Check that batch smc reproduces the native posterior exactly."""
    _assert_same_posterior(*single_chain_fits, chains=1)


@needs_scalar_power
def test_batch_smc_reproduces_the_native_posterior_across_worker_processes(two_chain_fits):
    """Check that batch smc reproduces the native posterior across worker processes."""
    _assert_same_posterior(*two_chain_fits, chains=2)


def test_batch_kernel_is_an_importable_module_attribute_for_the_worker_processes():
    # pm.sample_smc cloudpickles the kernel CLASS to its workers; a class defined inside a
    # function would not survive that round trip by reference.
    """Check that batch kernel is an importable module attribute for the worker processes."""
    import cloudpickle

    from . import vector_inference

    assert vector_inference.BatchIMH is BatchIMH
    assert cloudpickle.loads(cloudpickle.dumps(BatchIMH)) is BatchIMH


@needs_scalar_power
def test_prediction_bank_is_identical_under_both_backends(single_chain_fits):
    """Check that prediction bank is identical under both backends."""
    native, batch = single_chain_fits
    designs = [*SYNTHETIC_DESIGNS, ("synthetic probe C", [(6.0, 9.0)])]
    native_bank = prediction_bank([native], designs, predict_features)
    batch_bank = prediction_bank([batch], designs, predict_features, forward_backend="batch")
    assert set(native_bank) == set(batch_bank)
    for key in native_bank:
        assert np.array_equal(native_bank[key], batch_bank[key]), key
    # Every row of that bank came out of the prefill cache.
    assert prefill_stats()["misses"] == 0


def test_backends_are_checked_and_batch_requires_the_native_feature_forward():
    """Check that backends are checked and batch requires the native feature forward."""
    for backend in ("Batch", "jax", ""):
        with pytest.raises(ValueError, match="native or batch"):
            prediction_bank([], [], predict_features, forward_backend=backend)
    with pytest.raises(ValueError, match="native feature forward"):
        prediction_bank([], [], lambda *a: None, forward_backend="batch")
    with pytest.raises(ValueError, match="native feature forward"):
        fit(None, [], [], lambda *a: None, 1, forward_backend="batch")


# ----------------------------------------------------------------- equivalence report

REPORT_KEYS = frozenset({
    "generated_utc", "neuronbench_version", "grid", "pairs", "bit_identical",
    "max_abs_delta_mV", "spike_counts_equal", "features_equal", "by_mechanism",
    "first_mismatch", "machine", "equivalence_semantic_sha256",
})


@pytest.fixture(scope="module")
def small_reports(tmp_path_factory):
    """The cheap grid, generated twice into two files, for the determinism check."""
    pytest.importorskip("neuronbench")
    base = tmp_path_factory.mktemp("n1_reports")
    return [nfb.write_equivalence_report(base / f"report_{i}.json", "small") for i in (0, 1)], base


@needs_scalar_power
def test_small_equivalence_report_is_complete_and_bit_identical(small_reports):
    """Check that small equivalence report is complete and bit identical."""
    (report, _), base = small_reports
    assert set(report) == REPORT_KEYS
    assert report["pairs"] == (
        len(nfb.SMALL_KINDS) * nfb.SMALL_PROTOCOL_COUNT * len(nfb.SCALES)
    )
    assert report["bit_identical"] == report["pairs"]
    assert report["max_abs_delta_mV"] == 0.0
    assert report["spike_counts_equal"] is True
    assert report["features_equal"] is True
    assert report["first_mismatch"] is None
    assert [row["mechanism"] for row in report["by_mechanism"]] == list(nfb.SMALL_KINDS)
    assert report["machine"]["power_matches_scalar"] is True
    assert report["machine"]["numpy_cpu_features_enabled"]
    assert isinstance(report["machine"]["numpy_cpu_features_disabled"], str)
    assert report["grid"]["name"] == "small"

    path = base / "report_0.json"
    stored = json.loads(path.read_text())
    assert stored == report
    assert path.read_text() == canonical_json(stored) + "\n"
    # The file carries no digest of itself: the digest the runner binds is over these bytes.
    digest = sha256_file(path)
    assert is_hex64(digest)
    assert digest not in path.read_text()


@needs_scalar_power
def test_equivalence_report_repeats_exactly_apart_from_its_timestamp(small_reports):
    """Check that equivalence report repeats exactly apart from its timestamp."""
    (first, second), _ = small_reports
    assert first["generated_utc"] != second["generated_utc"]
    assert {k: v for k, v in first.items() if k != "generated_utc"} == {
        k: v for k, v in second.items() if k != "generated_utc"
    }


def test_report_protocols_cover_the_pool_and_every_world_test_set():
    """Check that report protocols cover the pool and every world test set."""
    pytest.importorskip("neuronbench")
    from neuronbench.worlds import POOL, WORLDS

    protocols = nfb.report_protocols()
    segments = [tuple(tuple(pair) for pair in row["segments"]) for row in protocols]
    assert len(segments) == len(set(segments))  # deduplicated on the segment list
    for _label, segment_list in POOL:
        assert tuple((float(d), float(a)) for d, a in segment_list) in segments
    for spec in WORLDS.values():
        for _label, segment_list in spec["test"]:
            assert tuple((float(d), float(a)) for d, a in segment_list) in segments
    assert all(row["samples"] > row["test_start"] >= 0 for row in protocols)


def test_report_cli_writes_the_named_file(tmp_path, capsys):
    """Check that report cli writes the named file."""
    pytest.importorskip("neuronbench")
    path = tmp_path / "cli_report.json"
    nfb.main(["--report", str(path), "--grid", "small"])
    report = json.loads(path.read_text())
    assert report["pairs"] > 0
    assert f"pairs={report['pairs']}" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        nfb.main(["--report", str(tmp_path / "unused.json"), "--grid", "enormous"])
    with pytest.raises(ValueError, match="small or full"):
        nfb.equivalence_report("enormous")


@pytest.mark.usefixtures("require_neuronbench_source")
def test_predict_batch_bank_is_c_contiguous_and_means_like_the_native_bank():
    """Check that predict batch bank is c contiguous and means like the native bank."""
    from .neuron_forward import HHHypothesis
    from .neuron_forward_batch import predict_batch

    design = ("public brief step", [(40.0, 12.0)])
    hypothesis = HHHypothesis("leak")
    thetas = np.array([0.5, 1.0, 2.0, 1.3, 0.8])
    batch = predict_batch(hypothesis, thetas, design)
    native = np.asarray([hypothesis.predict([c], design) for c in thetas])
    assert batch.flags["C_CONTIGUOUS"] and native.flags["C_CONTIGUOUS"]
    assert np.array_equal(batch.view(np.uint64), native.view(np.uint64))
    assert np.array_equal(batch.mean(axis=0).view(np.uint64), native.mean(axis=0).view(np.uint64))



# ---------------------------------------------------------------------------
# The six evaluation features as a second forward, its own prefill
# cache, and the fixed-step MH kernel's batch twin. The equality claims here
# are the ones above over again for the second feature space.
# ---------------------------------------------------------------------------


@needs_scalar_power
@pytest.mark.usefixtures("require_neuronbench_source")
def test_predict_eval_features_is_the_packages_own_function_on_the_sub_sampled_prediction():
    """Check that predict eval features is the packages own function on the sub sampled prediction."""
    from neuronbench.evaluator import eval_features
    from neuronbench.worlds import build_I

    from .neuron_forward import predict_eval_features

    hypothesis = HHHypothesis("sodium")
    design = ("synthetic conditioning probe", [(15.0, -30.0), (10.0, 12.0)])
    _, start = build_I(design[1])
    for theta in (0.6, 1.0, 1.7):
        voltage = hypothesis.predict([theta], design)
        observed_start = int(np.searchsorted(np.arange(0, len(voltage), 10), start))
        expected = eval_features(voltage[::10], observed_start)
        served = predict_eval_features(hypothesis, [theta], design)
        assert served.shape == (6,)
        assert np.array_equal(served.view(np.uint64), expected.view(np.uint64))
        # The three-feature forward is a different measurement of the same trace.
        assert len(predict_features(hypothesis, [theta], design)) == 3


@needs_scalar_power
@pytest.mark.usefixtures("require_neuronbench_source")
def test_eval_features_batch_equals_the_per_particle_calls():
    """Check that eval features batch equals the per particle calls."""
    from .neuron_forward import predict_eval_features

    hypothesis = HHHypothesis("h_current")
    design = ("synthetic conditioning probe", [(15.0, -30.0), (10.0, 12.0)])
    thetas = [0.4, 1.0, 2.5]
    rows = nfb.predict_eval_features_batch(hypothesis, thetas, design)
    assert rows.shape == (len(thetas), 6)
    clear_prefill()
    try:
        for index, theta in enumerate(thetas):
            assert np.array_equal(
                rows[index].view(np.uint64),
                predict_eval_features(hypothesis, [theta], design).view(np.uint64),
            )
    finally:
        clear_prefill()


@pytest.mark.usefixtures("require_neuronbench_source")
def test_the_two_feature_spaces_keep_separate_prefill_rows():
    """Check that the two feature spaces keep separate prefill rows."""
    from .neuron_forward import (
        predict_eval_features,
        prefill_eval_features,
        prefill_features,
    )

    hypothesis = HHHypothesis("potassium")
    design = ("synthetic two-space probe", [(6.0, 11.0)])
    thetas = [0.7, 1.4]
    clear_prefill()
    try:
        # Filling one space serves that space and MISSES in the other: a row measured three
        # ways must never be handed to the forward that wanted six.
        prefill_features(hypothesis, thetas, design)
        assert prefill_stats() == {"entries": 2, "hits": 0, "misses": 0}
        assert len(predict_features(hypothesis, [thetas[0]], design)) == 3
        assert prefill_stats()["hits"] == 1 and prefill_stats()["misses"] == 0
        assert len(predict_eval_features(hypothesis, [thetas[0]], design)) == 6
        assert prefill_stats()["misses"] == 1  # answered natively, and counted

        eval_rows = prefill_eval_features(hypothesis, thetas, design)
        assert prefill_stats()["entries"] == 4  # both spaces, side by side
        for index, theta in enumerate(thetas):
            served = predict_eval_features(hypothesis, [theta], design)
            assert np.array_equal(served, eval_rows[index])
            assert np.array_equal(
                predict_features(hypothesis, [theta], design).view(np.uint64),
                nfb.predict_features_batch(hypothesis, [theta], design)[0].view(np.uint64),
            )
        assert prefill_stats()["misses"] == 1  # still just the one, from before the prefill
    finally:
        clear_prefill()


# ----------------------------------------------------------------- fixed-step MH equivalence

EVAL_FEATURE_SCALES = [3.0, 3.0, 3.0, 3.0, 12.0, 8.0]


def _eval_problem():
    from .neuron_forward import predict_eval_features

    hypothesis = HHHypothesis(SYNTHETIC_MECHANISM)
    values = np.asarray([predict_eval_features(hypothesis, [SYNTHETIC_TRUTH], design)
                         for design in SYNTHETIC_DESIGNS])
    return hypothesis, values


def _fixed_steps_fits(chains):
    """The `paper_neuron` fit - fixed-step MH, six features, per-feature noise - both backends."""
    from .neuron_forward import predict_eval_features

    hypothesis, values = _eval_problem()
    settings = SMCSettings(16, chains, chains)
    options = {"kernel_policy": "mh_fixed_steps", "feature_scales": EVAL_FEATURE_SCALES}
    clear_prefill()
    native = fit(hypothesis, SYNTHETIC_DESIGNS, values, predict_eval_features, SMC_SEED,
                 settings, **options)
    batch = fit(hypothesis, SYNTHETIC_DESIGNS, values, predict_eval_features, SMC_SEED,
                settings, forward_backend="batch", **options)
    return native, batch


@pytest.fixture(scope="module")
def fixed_steps_single_chain_fits():
    """Test helper or fixture: fixed steps single chain fits."""
    pytest.importorskip("neuronbench")
    return _fixed_steps_fits(1)


@pytest.fixture(scope="module")
def fixed_steps_two_chain_fits():
    """Two chains on two cores: `BatchFixedStepsMH` really is cloudpickled into workers."""
    pytest.importorskip("neuronbench")
    return _fixed_steps_fits(2)


def _assert_same_fixed_steps_posterior(native, batch, chains):
    for name in ("conductance", "sigma"):
        assert np.array_equal(native.particles[name], batch.particles[name]), name
    assert batch.log_evidence == native.log_evidence
    assert batch.per_chain_log_evidence == native.per_chain_log_evidence
    assert len(batch.per_chain_log_evidence) == chains
    assert batch.diagnostics["forward_cache_misses"] == 0
    assert batch.diagnostics["forward_backend"] == "batch"
    assert set(batch.diagnostics) == set(native.diagnostics) | {
        "forward_backend", "forward_cache_misses",
    }
    for result in (native, batch):
        assert result.diagnostics["kernel_policy"] == "mh_fixed_steps"
        assert result.diagnostics["steps_per_stage"] == 3
        assert result.diagnostics["ess_threshold"] == 0.5
        assert result.diagnostics["sampler_attrs"]["_n_tune"] == 3
        assert result.diagnostics["sampler_attrs"]["threshold"] == 0.5
        assert result.diagnostics["feature_scales_mV_or_count"] == EVAL_FEATURE_SCALES
        assert result.diagnostics["feature_count"] == 6
        assert "sigma_unit_mV" not in result.diagnostics


@needs_scalar_power
def test_batch_fixed_steps_mh_reproduces_the_native_posterior_exactly(
    fixed_steps_single_chain_fits,
):
    """Check that batch fixed steps mh reproduces the native posterior exactly."""
    _assert_same_fixed_steps_posterior(*fixed_steps_single_chain_fits, chains=1)


@needs_scalar_power
def test_batch_fixed_steps_mh_reproduces_it_across_worker_processes(fixed_steps_two_chain_fits):
    """Check that batch fixed steps mh reproduces it across worker processes."""
    _assert_same_fixed_steps_posterior(*fixed_steps_two_chain_fits, chains=2)


def test_fixed_step_kernels_are_importable_module_attributes_for_the_worker_processes():
    """Check that fixed step kernels are importable module attributes for the worker processes."""
    import cloudpickle

    from . import vector_inference

    for kernel in (vector_inference.FixedStepsMH, vector_inference.BatchFixedStepsMH):
        assert cloudpickle.loads(cloudpickle.dumps(kernel)) is kernel
    # The prefill really is shared rather than re-implemented.
    assert issubclass(vector_inference.BatchFixedStepsMH, vector_inference._BatchPrefill)
    assert issubclass(BatchIMH, vector_inference._BatchPrefill)
    assert not issubclass(vector_inference.FixedStepsMH, vector_inference._BatchPrefill)


@needs_scalar_power
def test_six_feature_prediction_bank_is_identical_under_both_backends(
    fixed_steps_single_chain_fits,
):
    """Check that six feature prediction bank is identical under both backends."""
    from .neuron_forward import predict_eval_features

    native, batch = fixed_steps_single_chain_fits
    designs = [*SYNTHETIC_DESIGNS, ("synthetic probe C", [(6.0, 9.0)])]
    options = {"feature_scales": EVAL_FEATURE_SCALES}
    native_bank = prediction_bank([native], designs, predict_eval_features, **options)
    batch_bank = prediction_bank([batch], designs, predict_eval_features,
                                 forward_backend="batch", **options)
    assert native_bank["means"].shape == (len(native.particles["sigma"]), len(designs), 6)
    assert native_bank["sigmas"].shape == (len(native.particles["sigma"]), 6)
    np.testing.assert_array_equal(
        native_bank["sigmas"],
        np.asarray(EVAL_FEATURE_SCALES)[None, :] * native.particles["sigma"][:, None],
    )
    for key in native_bank:
        assert np.array_equal(native_bank[key], batch_bank[key]), key
    assert prefill_stats()["misses"] == 0


# --------------------------------------------------------------------------
# The semantic digest the post-run comparison uses
# --------------------------------------------------------------------------


def test_the_semantic_digest_ignores_the_timestamp_and_nothing_else(small_reports):
    """Check that the semantic digest ignores the timestamp and nothing else."""
    _, base = small_reports
    first, second = (json.loads((base / f"report_{i}.json").read_text()) for i in (0, 1))
    assert first["generated_utc"] != second["generated_utc"] or first == second
    assert nfb.equivalence_semantic_sha256(first) == nfb.equivalence_semantic_sha256(second)
    assert first["equivalence_semantic_sha256"] == nfb.equivalence_semantic_sha256(first)
    # the file digest still moves with the timestamp; the semantic one does not.
    changed = dict(first, generated_utc="2000-01-01T00:00:00+00:00")
    assert nfb.equivalence_semantic_sha256(changed) == first["equivalence_semantic_sha256"]
    for key, value in (("max_abs_delta_mV", first["max_abs_delta_mV"] + 1.0),
                       ("bit_identical", first["bit_identical"] - 1),
                       ("machine", {**first["machine"], "synthetic_field": "changed"}),
                       ("neuronbench_version", "0.0.0-synthetic")):
        assert nfb.equivalence_semantic_sha256(dict(first, **{key: value})) != first["equivalence_semantic_sha256"], key
    payload = nfb.equivalence_semantic_payload(first)
    assert "generated_utc" not in payload and "equivalence_semantic_sha256" not in payload
    assert set(payload) >= {"neuronbench_version", "grid", "pairs", "bit_identical", "max_abs_delta_mV",
                            "spike_counts_equal", "features_equal", "by_mechanism", "first_mismatch", "machine"}


def test_the_semantic_digest_is_sorted_key_compact_json_of_the_payload():
    """Check that the semantic digest is sorted key compact json of the payload."""
    synthetic = {"generated_utc": "2026-01-01T00:00:00+00:00", "neuronbench_version": "synthetic",
                 "grid": {"name": "small", "mechanisms": ["leak"], "scales": [1.0], "protocols": []},
                 "pairs": 1, "bit_identical": 1, "max_abs_delta_mV": 0.0, "spike_counts_equal": True,
                 "features_equal": True, "by_mechanism": [], "first_mismatch": None, "machine": {"host": "synthetic"}}
    expected = hashlib.sha256(json.dumps({k: v for k, v in synthetic.items() if k != "generated_utc"},
                                         sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    assert nfb.equivalence_semantic_sha256(synthetic) == expected
    assert nfb.equivalence_semantic_sha256({**synthetic, "equivalence_semantic_sha256": expected}) == expected
