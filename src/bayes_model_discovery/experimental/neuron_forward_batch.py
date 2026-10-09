"""Particle-vectorised replica of ``neuronbench.worlds.simulate`` (forward Euler, trace=True).

One particle is one positive conductance multiplier for one catalogue mechanism, exactly as
``neuron_forward._trace`` maps it onto ``simulate`` kwargs. The inner loop keeps the ORIGINAL
scalar loop's operation order, expressions, ``np.exp`` calls, ``+1e-9`` regulariser, clipping and
gate initialisation - nothing is rearranged or fused - so every particle's trace is bit-identical
to the scalar ``simulate(..., trace=True)`` run for that multiplier. The alpha/beta and steady
state helpers are IMPORTED from the benchmark rather than copied, so they cannot drift.

BIT-IDENTITY REQUIREMENT (numpy 1.26.4): the scalar path evaluates ``m**3`` / ``nn**4`` /
``x**mpow`` on numpy float64 SCALARS, which calls libm ``pow``. numpy's AVX-512 array loop for
``power`` differs from libm ``pow`` by up to 1 ulp, and the array ``x**2`` operator takes an
``x*x`` fast path which libm ``pow`` does not reproduce either. Run the interpreter with

    NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX
                              AVX512_CNL AVX512_ICL"

(one space-separated string) so numpy's power loop falls back to libm ``pow``.
:func:`power_matches_scalar` checks this at run time and is the startup self-check the runner
performs before it accepts ``--forward-backend batch``. That environment also changes what
``np.exp`` returns, so the batch path, the scalar path and the benchmark's own truth generation
must all run under the SAME environment; the equivalence report records which one it was.

``python -m bayes_model_discovery.experimental.neuron_forward_batch --report OUT.json`` writes
that report.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..provenance.canonical import canonical_json
from .neuron_forward import _trace, eval_feature_vector, native_test_counts, summarize_voltage

# The candidate catalogue's frozen generic channel shapes (E, mvh, mk, mtau[, mpow, hvh, hk,
# htau]), i.e. the table `neuron_forward._trace` builds its `Chan` from. It is repeated here
# rather than shared because `neuron_forward` takes additions only; the ten-mechanism
# bit-identity test is what keeps the two copies from drifting apart.
SHAPES = {
    "h_current": (-35.0, -85.0, -8.0, 100.0),
    "m_current": (-77.0, -40.0, 12.0, 80.0),
    "inward": (50.0, -45.0, 8.0, 10.0),
    "rebound_inward_slow": (120.0, -57.0, 5.0, 4.0, 2, -88.0, 4.0, 130.0),
    "rebound_inward_fast": (120.0, -54.0, 6.0, 2.0, 2, -87.0, 4.0, 22.0),
    "rebound_outward": (-77.0, -30.0, 10.0, 3.0, 1, -80.0, 5.0, 200.0),
}
KINDS = ("leak", "sodium", "potassium", "slow_na", *SHAPES)
SCALES = (0.5, 1.0, 2.0)
#: The mechanisms and protocol sources the cheap ``--grid small`` report covers.
SMALL_KINDS = ("leak", "slow_na", "rebound_outward")
SMALL_PROTOCOL_COUNT = 1


def _clip01(x):
    """``np.clip(x, 0, 1)`` - the benchmark's scalar ``_clip(x, 0, 1)``, as a min/max pair."""
    return np.minimum(np.maximum(x, 0), 1)


def _ipow(x, p):
    """libm ``pow(x, p)`` elementwise - what the scalar loop's ``x**p`` does on a float64."""
    return np.power(x, float(p))


def power_matches_scalar(n=2048, seed=0):
    """True iff the array power loop reproduces the scalar (libm) ``pow`` bit for bit here.

    False means numpy is dispatching ``power`` to a SIMD implementation and the batch forward
    is NOT bit-identical to the scalar one; the caller is expected to refuse rather than to
    silently produce a near-identical trace.
    """
    x = np.random.default_rng(seed).uniform(0.0, 1.0, n)
    for p in (3.0, 4.0, 2.0, 1.0):
        reference = np.array([np.float64(t) ** p for t in x])
        if not np.array_equal(_ipow(x, p).view(np.uint64), reference.view(np.uint64)):
            return False
    return True


def batch_kwargs(kind, scales):
    """Map a (P,) array of positive multipliers onto per-particle ``simulate`` kwargs."""
    from neuronbench.worlds import Chan

    scales = np.asarray(scales, dtype=float)
    if scales.ndim != 1 or scales.size == 0:
        raise ValueError("conductance multipliers must be a non-empty 1-D array")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("conductance multiplier must be finite and positive")
    count = scales.size
    gNa = np.full(count, 120.0)
    gK = np.full(count, 36.0)
    gL = np.full(count, 0.3)
    slow_na, channel = False, None
    if kind == "leak":
        gL = 0.3 * scales
    elif kind == "sodium":
        gNa = 120.0 * scales
    elif kind == "potassium":
        gK = 36.0 * scales
    elif kind == "slow_na":
        slow_na, gL = True, 0.3 * scales
    elif kind in SHAPES:
        channel = Chan(kind, scales, *SHAPES[kind])
    else:
        raise ValueError("unknown candidate mechanism")
    return gNa, gK, gL, slow_na, channel


def simulate_batch(I, test_start, kind, scales, dt=0.01, V0=-65.0):
    """Vectorise the native integrator over P conductance multipliers.

    Returns ``(voltage, counts)`` with ``voltage`` a (P, len(I)) float64 array - a transposed
    view of the time-major working buffer, so no second copy of a 300 MB trace block is made -
    and ``counts`` the (P,) test-window spike counts. Blockers and stochastic worlds are not
    supported: this is the candidate forward hook, which never sees either.
    """
    from neuronbench.worlds import EK, EL, ENa, _ab, _hinf, _minf

    gNa, gK, gL, slow_na, c = batch_kwargs(kind, scales)
    count = gNa.size
    n = len(I)
    v = np.full(count, float(V0))
    am, bm, ah, bh, an, bn = _ab(v)
    m = am / (am + bm)
    h = ah / (ah + bh)
    nn = an / (an + bn)
    s = np.full(count, 1.0)  # slow Na inactivation (only if slow_na)
    xs = _minf(v, c.mvh, c.mk) if c is not None else None
    hs = _hinf(v, c.hvh, c.hk) if c is not None and c.hvh is not None else np.full(count, 1.0)
    voltage = np.empty((n, count))  # time-major; transposed on return
    spikes = np.zeros(count, dtype=np.int64)
    pv = v
    for i in range(n):
        am, bm, ah, bh, an, bn = _ab(v)
        m = m + dt * (am * (1 - m) - bm * m)
        h = h + dt * (ah * (1 - h) - bh * h)
        nn = nn + dt * (an * (1 - nn) - bn * nn)
        m = _clip01(m)
        h = _clip01(h)
        nn = _clip01(nn)
        if slow_na:
            s_inf = 1.0 / (1.0 + np.exp((v + 45) / 4.0))
            tau_s = np.where(v > -55.0, 380.0, 1500.0)
            s = s + dt * (s_inf - s) / tau_s
            s = _clip01(s)
        cur = (
            gNa * _ipow(m, 3) * h * (s if slow_na else 1.0) * (v - ENa)
            + gK * _ipow(nn, 4) * (v - EK)
            + gL * (v - EL)
        )
        if c is not None:
            xs = xs + dt * (_minf(v, c.mvh, c.mk) - xs) / c.mtau
            if c.hvh is not None:
                hs = hs + dt * (_hinf(v, c.hvh, c.hk) - hs) / c.htau
            xs = _clip01(xs)
            hs = _clip01(hs)
            cur = cur + c.g * _ipow(xs, c.mpow) * _ipow(hs, c.hpow) * (v - c.E)
        v = v + dt * (I[i] - cur)
        v = np.minimum(np.maximum(v, -95), 80)
        voltage[i] = v
        if i >= test_start:
            spikes += (pv < 0) & (0 <= v)
        pv = v
    if not np.all(np.isfinite(voltage)):
        raise ValueError("non-finite native HH output")
    return voltage.T, spikes


def _multipliers(thetas):
    values = np.asarray(thetas, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("exactly one conductance multiplier required per particle")
    return values


def predict_batch(hypothesis, thetas, design):
    """The (P, T) native traces for P conductance multipliers under one design."""
    from neuronbench.worlds import build_I

    current, start = build_I(design[1])
    voltage, _ = simulate_batch(current, start, hypothesis.mechanism, _multipliers(thetas))
    # C-contiguous like the native `np.asarray([predict(...) for c in particles])` bank, so
    # reductions over the particle axis (`bank.mean(axis=0)` in the scoring block) add in
    # the same memory order and stay bit-identical; the transposed time-major view would
    # differ by an ulp or two.
    return np.ascontiguousarray(voltage)


def predict_features_batch(hypothesis, thetas, design):
    """The (P, 3) feature rows ``predict_features`` returns, one per conductance multiplier."""
    from neuronbench.worlds import build_I

    current, start = build_I(design[1])
    voltage, _ = simulate_batch(current, start, hypothesis.mechanism, _multipliers(thetas))
    # Match the observable deterministic sampling interval (native dt 0.01 ms), exactly as
    # `neuron_forward.predict_features` does for a single particle.
    observed_start = int(np.searchsorted(np.arange(0, voltage.shape[1], 10), start))
    return np.asarray([summarize_voltage(row[::10], observed_start) for row in voltage])


def predict_eval_features_batch(hypothesis, thetas, design):
    """The (P, 6) rows ``neuron_forward.predict_eval_features`` returns, one per multiplier."""
    from neuronbench.worlds import build_I

    current, start = build_I(design[1])
    voltage, _ = simulate_batch(current, start, hypothesis.mechanism, _multipliers(thetas))
    observed_start = int(np.searchsorted(np.arange(0, voltage.shape[1], 10), start))
    return np.asarray([eval_feature_vector(row[::10], observed_start) for row in voltage])


# ----------------------------------------------------------------- equivalence report


def report_protocols():
    """The design pool plus every world's held-out test protocol, deduplicated on segments."""
    from neuronbench.worlds import POOL, WORLDS, build_I

    rows, seen = [], {}
    sources = [("POOL", label, segments) for label, segments in POOL]
    sources += [("test:" + world, label, segments)
                for world, spec in WORLDS.items() for label, segments in spec["test"]]
    for source, label, segments in sources:
        key = tuple((float(duration), float(amplitude)) for duration, amplitude in segments)
        if key in seen:
            rows[seen[key]]["aliases"].append(source + "/" + label)
            continue
        seen[key] = len(rows)
        current, start = build_I(key)
        rows.append({"label": label, "source": source, "segments": [list(p) for p in key],
                     "aliases": [], "samples": len(current), "test_start": int(start)})
    return rows


def numpy_cpu_features_enabled():
    """The CPU features numpy is actually dispatching on, in numpy's own order."""
    from numpy.core._multiarray_umath import __cpu_features__

    return [name for name, enabled in __cpu_features__.items() if enabled]


def numpy_cpu_features_disabled():
    """The raw NPY_DISABLE_CPU_FEATURES request, verbatim ("" when it was not set).

    Recorded as text rather than parsed: numpy accepts names this machine does not have and
    only warns, so the request and the result are two different facts and both are kept.
    """
    return os.environ.get("NPY_DISABLE_CPU_FEATURES", "")


def machine_fingerprint():
    """Everything about this interpreter that decides the floating-point result."""
    import importlib.metadata

    return {
        "numpy_cpu_features_enabled": numpy_cpu_features_enabled(),
        "numpy_cpu_features_disabled": numpy_cpu_features_disabled(),
        "power_matches_scalar": power_matches_scalar(),
        "numpy": importlib.metadata.version("numpy"),
        "pytensor": importlib.metadata.version("pytensor"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }


def _cell(kind, protocol, scales):
    """One (mechanism, protocol) cell: the scalar traces for every multiplier vs one batch call."""
    from neuronbench.evaluator import eval_features
    from neuronbench.worlds import build_I

    segments = tuple(tuple(pair) for pair in protocol["segments"])
    current, start = build_I(segments)
    native = [np.asarray(_trace(kind, float(scale), segments)) for scale in scales]
    voltage, spikes = simulate_batch(current, start, kind, np.asarray(scales, dtype=float))
    observed_start = int(np.searchsorted(np.arange(0, voltage.shape[1], 10), start))
    rows = []
    for index, scale in enumerate(scales):
        batched = np.ascontiguousarray(voltage[index])
        reference = native[index]
        difference = np.flatnonzero(batched.view(np.uint64) != reference.view(np.uint64))
        identical = difference.size == 0
        delta = float(np.max(np.abs(batched - reference)))
        counts_equal = bool(
            int(spikes[index]) == int(native_test_counts(reference, start)[0])
        )
        features_equal = bool(
            np.array_equal(summarize_voltage(batched[::10], observed_start),
                           summarize_voltage(reference[::10], observed_start))
            and np.array_equal(eval_features(batched, start), eval_features(reference, start))
        )
        rows.append({
            "mechanism": kind, "protocol": protocol["label"], "source": protocol["source"],
            "scale": float(scale), "bit_identical": identical, "abs_delta_mV": delta,
            "spike_counts_equal": counts_equal, "features_equal": features_equal,
            "first_difference_index": None if identical else int(difference[0]),
        })
    return rows


def equivalence_report(grid="full"):
    """Compare the batch integrator against the production scalar path over a fixed grid.

    ``full`` is the ten catalogue mechanisms x the deduplicated design pool and held-out test
    protocols x the three multipliers; ``small`` is the cheap subset the tests use. Everything
    but ``generated_utc`` is a deterministic function of the grid and the machine.
    """
    import neuronbench as nb

    if grid not in ("small", "full"):
        raise ValueError("grid must be small or full")
    kinds = KINDS if grid == "full" else SMALL_KINDS
    protocols = report_protocols()
    if grid == "small":
        protocols = protocols[:SMALL_PROTOCOL_COUNT]
    rows = []
    for kind in kinds:
        for protocol in protocols:
            rows.extend(_cell(kind, protocol, SCALES))
        # Bound the scalar path's trace cache: a full grid is 1,110 traces of up to 76,000
        # samples, and the cache is a speed aid, never state anything depends on.
        _trace.cache_clear()
    by_mechanism = []
    for kind in kinds:
        cells = [row for row in rows if row["mechanism"] == kind]
        by_mechanism.append({
            "mechanism": kind,
            "pairs": len(cells),
            "bit_identical": sum(row["bit_identical"] for row in cells),
            "max_abs_delta_mV": max(row["abs_delta_mV"] for row in cells),
            "spike_counts_equal": all(row["spike_counts_equal"] for row in cells),
            "features_equal": all(row["features_equal"] for row in cells),
        })
    mismatches = [row for row in rows if not row["bit_identical"]]
    return {
        "generated_utc": datetime.now(UTC).isoformat(),
        "neuronbench_version": nb.__version__,
        "grid": {
            "name": grid,
            "mechanisms": list(kinds),
            "scales": list(SCALES),
            "protocols": protocols,
        },
        "pairs": len(rows),
        "bit_identical": sum(row["bit_identical"] for row in rows),
        "max_abs_delta_mV": max(row["abs_delta_mV"] for row in rows),
        "spike_counts_equal": all(row["spike_counts_equal"] for row in rows),
        "features_equal": all(row["features_equal"] for row in rows),
        "by_mechanism": by_mechanism,
        "first_mismatch": mismatches[0] if mismatches else None,
        "machine": machine_fingerprint(),
    }


SEMANTIC_EXCLUDED_KEYS = ("generated_utc", "equivalence_semantic_sha256")


def equivalence_semantic_payload(report):
    """The report without its timestamp and without its own semantic digest.

    Everything left is a deterministic function of the grid and the machine
    (a project decision of 2026-09-20): neuronbench_version,
    the full grid, pairs, bit_identical, max_abs_delta_mV, spike_counts_equal,
    features_equal, by_mechanism, first_mismatch and machine.
    """
    return {key: value for key, value in report.items() if key not in SEMANTIC_EXCLUDED_KEYS}


def equivalence_semantic_sha256(report):
    """SHA-256 of the semantic payload as sorted-key compact JSON.

    Two reports of one grid on one machine that differ only in ``generated_utc``
    share this digest; a change in any result or machine field changes it. The
    file digest the runner binds (the SHA-256 of the written bytes) is unchanged
    and still moves with the timestamp; the post-run comparison the project
    decided on is made on this digest.
    """
    payload = json.dumps(equivalence_semantic_payload(report), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_equivalence_report(path, grid="full"):
    """Write the report as canonical JSON, carrying its own semantic digest under
    ``equivalence_semantic_sha256`` (computed over the payload without that key and
    without the timestamp). The file carries no hash of its whole bytes: the digest the
    runner binds into config.json is the SHA-256 of these bytes."""
    report = equivalence_report(grid)
    report["equivalence_semantic_sha256"] = equivalence_semantic_sha256(report)
    Path(path).write_text(canonical_json(report) + "\n")
    return report


def main(argv=None):
    """Command-line entry: write the batch/scalar equivalence report to ``--report`` for the
    chosen ``--grid`` and print a one-line summary of its counts and semantic digest."""
    parser = argparse.ArgumentParser(description="Batch/scalar HH forward equivalence report")
    parser.add_argument("--report", required=True)
    parser.add_argument("--grid", choices=("small", "full"), default="full")
    args = parser.parse_args(argv)
    report = write_equivalence_report(args.report, args.grid)
    print(
        f"pairs={report['pairs']} bit_identical={report['bit_identical']} "
        f"max_abs_delta_mV={report['max_abs_delta_mV']} "
        f"spike_counts_equal={report['spike_counts_equal']} "
        f"features_equal={report['features_equal']} "
        f"semantic_sha256={report['equivalence_semantic_sha256']}"
    )


if __name__ == "__main__":
    main()
