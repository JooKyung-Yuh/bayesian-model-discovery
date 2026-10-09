"""Candidate Hodgkin-Huxley (HH) forward hook; no world/truth access and no observation noise."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class HHHypothesis:
    """One candidate Hodgkin-Huxley mechanism from the fixed catalogue, with one conductance multiplier."""

    mechanism: str

    def __post_init__(self):
        if self.mechanism not in (
            "leak",
            "sodium",
            "potassium",
            "slow_na",
            "h_current",
            "m_current",
            "inward",
            "rebound_inward_slow",
            "rebound_inward_fast",
            "rebound_outward",
        ):
            raise ValueError("unknown candidate mechanism")

    @property
    def model_id(self):
        """Stable model key of this candidate."""
        return "hh_" + self.mechanism + "_gaussian_features"

    def predict(self, theta, design):
        """theta=[positive conductance multiplier]; output is full native trace."""
        theta = np.asarray(theta)
        if theta.shape != (1,):
            raise ValueError("exactly one conductance multiplier required")
        scale = float(theta[0])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("conductance multiplier must be finite and positive")
        segments = tuple((float(t), float(i)) for t, i in design[1])
        return _trace(self.mechanism, scale, segments).copy()


@lru_cache(maxsize=512)
def _trace(kind, scale, segments):
    from neuronbench.worlds import Chan, build_I, simulate

    # Candidate catalogue, not the task's hidden alternative or discriminator.
    kwargs = {}
    if kind in ("leak", "sodium", "potassium"):
        name, unit = {"leak": ("gL", 0.3), "sodium": ("gNa", 120.0), "potassium": ("gK", 36.0)}[
            kind
        ]
        kwargs[name] = unit * scale
    elif kind == "slow_na":
        kwargs = {"slow_na": True, "gL": 0.3 * scale}
    else:
        # Frozen generic channel shapes; only conductance is inferred initially.
        shape = {
            "h_current": (-35.0, -85.0, -8.0, 100.0),
            "m_current": (-77.0, -40.0, 12.0, 80.0),
            "inward": (50.0, -45.0, 8.0, 10.0),
            # Public channel definitions; E, mvh, mk, mtau, mpow, hvh, hk, htau.
            # The eighth value is essential: omitting it freezes h at htau=1e9.
            "rebound_inward_slow": (120.0, -57.0, 5.0, 4.0, 2, -88.0, 4.0, 130.0),
            "rebound_inward_fast": (120.0, -54.0, 6.0, 2.0, 2, -87.0, 4.0, 22.0),
            "rebound_outward": (-77.0, -30.0, 10.0, 3.0, 1, -80.0, 5.0, 200.0),
        }[kind]
        kwargs["extra"] = [Chan(kind, scale, *shape)]
    current, start = build_I(segments)
    _, voltage = simulate(current, start, trace=True, **kwargs)
    if not np.all(np.isfinite(voltage)):
        raise ValueError("non-finite native HH output")
    voltage.flags.writeable = False
    return voltage


def summarize_voltage(voltage, test_start):
    """Fixed physical features [test mean, overall minimum, final voltage], mV."""
    voltage = np.asarray(voltage, dtype=float)
    if voltage.ndim != 1 or not 0 <= test_start < len(voltage):
        raise ValueError("invalid voltage trace or test window")
    return np.array([voltage[test_start:].mean(), voltage.min(), voltage[-1]])


def native_test_counts(voltage, test_start):
    """Match native deterministic crossing at i >= test_start, including boundary."""
    voltage = np.atleast_2d(voltage)
    previous = np.concatenate([np.full((len(voltage), 1), -65.0), voltage[:, :-1]], axis=1)
    return ((previous < 0) & (voltage >= 0))[:, test_start:].sum(axis=1)


# --------------------------------------------------------------------------
# The benchmark's OWN evaluation feature map as a second, optional
# training feature space (the `paper_neuron` profile; `dev`/`paper` keep the
# three physical features above). The features are computed by
# `neuronbench.evaluator.eval_features` itself - never reimplemented here -
# on exactly the trace the benchmark hands out, so an observation and a
# candidate's prediction are compared on the same function.
# --------------------------------------------------------------------------
#: The names of the six values `eval_feature_vector` returns, in order.
EVAL_FEATURE_NAMES = ("n_test", "n_pre", "rundown", "adaptation", "V_min_mV", "V_end_mV")
#: The benchmark's per-feature standardisation for those six
#: (`neuronbench.evaluator._EVAL_FEAT_SCALES`): counts for the first four, mV for the last two.
#: Repeated here rather than imported from a private package name, exactly as
#: `neuron_forward_batch.SHAPES` repeats the channel table - and, as there, a test is what keeps
#: the two copies from drifting apart, so a package change is a visible failure rather than a
#: silent change of this profile's likelihood.
EVAL_FEATURE_SCALES = (3.0, 3.0, 3.0, 3.0, 12.0, 8.0)


def eval_feature_vector(voltage, test_start):
    """The benchmark's six evaluation features of an observed (0.1 ms) trace, its own function.

    `[n_test, n_pre, n_pre - n_test, within-test adaptation, V_min, V_end]` - four spike counts
    and two voltages. The input is the sub-sampled trace `World.run` returns and the test-window
    index it returns beside it, which is what makes this the same measurement the benchmark's
    own evaluator makes.
    """
    from neuronbench.evaluator import eval_features

    voltage = np.asarray(voltage, dtype=float)
    if voltage.ndim != 1 or not 0 <= test_start < len(voltage):
        raise ValueError("invalid voltage trace or test window")
    return eval_features(voltage, int(test_start))


# --------------------------------------------------------------------------
# Batch forward: an OPTIONAL feature-level prefill cache. A batch run
# computes one whole particle cloud in a single vectorised call and deposits
# the three-feature rows here; `predict_features` then answers from the cache
# instead of integrating that particle again. The cache is keyed on exactly
# what decides the trace - (mechanism, conductance multiplier, design
# segments) - and holds FEATURES, never traces: a 500-particle trace block for
# one design is ~300 MB, three floats per particle is not.
#
# Nothing here changes the default path. An empty cache (every native run)
# means every lookup misses and `predict_features` computes what it always
# computed, bit for bit; only the miss counter moves.
#
# The second feature space (the six evaluation features above) is also cached, so
# the key carries the space as well and the two never share a row: a forward
# reads only what its own prefill wrote.
# --------------------------------------------------------------------------
_FEATURE_PREFILL = {}
_PREFILL_HITS = 0
_PREFILL_MISSES = 0

#: Cache-key tags for the two feature spaces (`predict_features`, `predict_eval_features`).
FEATURE_SPACE_3D = "features3"
FEATURE_SPACE_EVAL6 = "eval6"


def _prefill_key(mechanism, scale, design, space):
    return (mechanism, float(scale),
            tuple((float(t), float(i)) for t, i in design[1]), space)


def _store_prefill(hypothesis, thetas, design, space, rows):
    for scale, row in zip(np.asarray(thetas, dtype=float).reshape(-1), rows, strict=True):
        _FEATURE_PREFILL[_prefill_key(hypothesis.mechanism, scale, design, space)] = row
    return rows


def _lookup_prefill(hypothesis, theta, design, space):
    """The cached row for one particle, or None - counting the hit or the miss either way."""
    global _PREFILL_HITS, _PREFILL_MISSES

    if theta.shape == (1,):
        cached = _FEATURE_PREFILL.get(
            _prefill_key(hypothesis.mechanism, theta[0], design, space)
        )
        if cached is not None:
            _PREFILL_HITS += 1
            return cached.copy()
    _PREFILL_MISSES += 1
    return None


def prefill_features(hypothesis, thetas, design):
    """Compute one design's features for a whole particle cloud and cache them.

    Returns the (P, 3) rows, which are the same values `predict_features` returns one particle
    at a time (the batch integrator is bit-identical to the scalar one under the declared
    environment; see neuron_forward_batch).
    """
    from .neuron_forward_batch import predict_features_batch

    return _store_prefill(hypothesis, thetas, design, FEATURE_SPACE_3D,
                          predict_features_batch(hypothesis, thetas, design))


def prefill_eval_features(hypothesis, thetas, design):
    """The same for the benchmark's six evaluation features: the (P, 6) rows, cached."""
    from .neuron_forward_batch import predict_eval_features_batch

    return _store_prefill(hypothesis, thetas, design, FEATURE_SPACE_EVAL6,
                          predict_eval_features_batch(hypothesis, thetas, design))


def clear_prefill():
    """Drop every prefilled row and zero the counters."""
    global _PREFILL_HITS, _PREFILL_MISSES

    _FEATURE_PREFILL.clear()
    _PREFILL_HITS = 0
    _PREFILL_MISSES = 0


def prefill_stats():
    """Entries currently cached, and how many lookups have hit and missed since the last clear.

    A batch fit asserts `misses == 0`: a miss is answered natively, so it is a performance and
    provenance fault, never a wrong number.
    """
    return {"entries": len(_FEATURE_PREFILL), "hits": _PREFILL_HITS, "misses": _PREFILL_MISSES}


def predict_features(hypothesis, theta, design):
    """The three physical features of a candidate's predicted trace, sub-sampled to the observed interval (cache first)."""
    from neuronbench.worlds import build_I

    theta = np.asarray(theta)
    cached = _lookup_prefill(hypothesis, theta, design, FEATURE_SPACE_3D)
    if cached is not None:
        return cached
    _, start = build_I(design[1])
    voltage = hypothesis.predict(theta, design)
    # Match the observable deterministic sampling interval (native dt 0.01 ms).
    return summarize_voltage(
        voltage[::10], int(np.searchsorted(np.arange(0, len(voltage), 10), start))
    )


def predict_eval_features(hypothesis, theta, design):
    """The benchmark's six evaluation features of a candidate's predicted trace.

    The same sub-sampling `predict_features` applies, then the same `eval_feature_vector` the
    observation went through - the only difference between the two sides is whose trace it is.
    """
    from neuronbench.worlds import build_I

    theta = np.asarray(theta)
    cached = _lookup_prefill(hypothesis, theta, design, FEATURE_SPACE_EVAL6)
    if cached is not None:
        return cached
    _, start = build_I(design[1])
    voltage = hypothesis.predict(theta, design)
    return eval_feature_vector(
        voltage[::10], int(np.searchsorted(np.arange(0, len(voltage), 10), start))
    )


#: Which prefill fills the cache each forward reads. `vector_inference` looks a forward up here
#: rather than naming one feature space, so the batch backend follows the profile's forward.
BATCH_FORWARDS = {
    predict_features: prefill_features,
    predict_eval_features: prefill_eval_features,
}


def initial_hypotheses():
    """The initial candidate pool: leak, sodium, potassium and slow sodium."""
    return [HHHypothesis(name) for name in ("leak", "sodium", "potassium", "slow_na")]


def mock_proposals(event):
    """Fixed mock proposals for proposal events 1 to 4: h, M and inward currents."""
    if event not in (1, 2, 3, 4):
        raise ValueError("proposal event outside Neuron schedule")
    return [HHHypothesis(name) for name in ("h_current", "m_current", "inward")]
