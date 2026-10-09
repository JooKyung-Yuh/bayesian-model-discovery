"""Real PyMC sequential Monte Carlo (SMC) and discrete predictive mixtures for the new paper CORE."""

import math
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp, xlogy

from ..eig.joint import choose_design
from .count_models import (
    BINARY_TRIALS,
    Hypothesis,
    active_particle_rows,
    coefficient_prior,
    dispersion_prior,
    evaluate,
    input_index_codes,
    named_input_values,
    particle_logpmf,
)


class CountInferenceFailure(RuntimeError):
    """THIS INFERENCE ATTEMPT failed numerically.

    Read it as "the SMC run for this hypothesis, on this data, with this seed,
    did not produce a usable posterior". It is NOT a statement that the model is
    false, and it is not a claim that its evidence is -inf: it is a failed
    attempt, which is why `core_count` records it as
    `retirement_kind: "inference_failure"`.

    Raised by `fit` and by nothing else, and only for a cause the classifier
    below could actually NAME (`cause_code`). An OSError, a MemoryError, an
    AssertionError, a TypeError, a KeyError, a shape/broadcast/seed/config
    ValueError, a compile or worker failure, or any exception the classifier
    cannot place is NOT a statement about the model and propagates unchanged, so
    the run fails instead of quietly retiring a model.

    Carries, for the durable record `core_count` writes: `cause_code`, `stage`,
    and `original` - the original exception's module, class, message and the
    FULL chained traceback text. A live `__cause__` is not enough: the run
    continues after a retirement and the traceback object is gone by the time the
    summary is written.
    """

    def __init__(self, cause_code, stage, message, original=None):
        super().__init__(f"[{cause_code}] {message}")
        self.cause_code = cause_code
        self.stage = stage
        self.original = original or {}

    def as_record(self):
        """Return the failure as a JSON-ready dict with `cause_code`, `stage` and `original`."""
        return {"cause_code": self.cause_code, "stage": self.stage, "original": self.original}


class FitTimeout(CountInferenceFailure):
    """A single fit exceeded its declared wall-clock budget."""

    def __init__(self, elapsed_seconds, timeout_seconds):
        self.elapsed_seconds = float(elapsed_seconds)
        self.timeout_seconds = float(timeout_seconds)
        super().__init__(
            "wall_clock_timeout", "sampling",
            f"fit exceeded {self.timeout_seconds:g} seconds "
            f"(elapsed {self.elapsed_seconds:.6f} seconds)",
        )

    def as_record(self):
        """Return the base failure record plus the timeout status and elapsed/limit seconds."""
        return {**super().as_record(), "status": "fit_timeout",
                "elapsed_seconds": self.elapsed_seconds,
                "timeout_seconds": self.timeout_seconds}


def validate_fit_timeout(value):
    """Return a fit timeout as positive finite float seconds, or None when omitted.

    Raises ValueError for booleans, non-finite or non-positive values.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("fit timeout must be finite and positive, not boolean")  # noqa: TRY004 - Preserve the public ValueError validation contract.
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("fit timeout must be finite and positive")
    return seconds


def record_fit_timing(output_dir, hypothesis, round_index, elapsed_seconds, status):
    """Append the opt-in sidecar without changing any existing artifact keys."""
    import json
    from pathlib import Path

    row = {"model_id": getattr(hypothesis, "model_id", hypothesis),
           "round": int(round_index), "elapsed_seconds": float(elapsed_seconds),
           "status": status}
    with (Path(output_dir) / "fit_timings.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")



def record_acquisition_timing(output_dir, round_index, design_policy, elapsed_seconds,
                              n_candidates):
    """Append observational timing separately from reproducible acquisition records."""
    import json
    from pathlib import Path

    seconds = float(elapsed_seconds)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("acquisition elapsed time must be finite and nonnegative")
    row = {"round": int(round_index), "design_policy": design_policy,
           "elapsed_seconds": seconds, "n_candidates": int(n_candidates)}
    with (Path(output_dir) / "acquisition_timings.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def record_prediction_pool_summary(output_dir, round_index, checkpoint, observation_count,
                                   models):
    """Summarize existing per-model predictive means, without changing predictions.

    Each input row supplies model_id, weight and a one-dimensional ``means``
    vector over the checkpoint's evaluation inputs. An excluded model may omit
    means. Explicit point-local exclusions retain their original indices and
    point-local weights; summaries cover only retained points. Nonfinite values
    among retained points produce null summaries and a diagnostic count, never
    silent filtering. Only summaries are serialized, never evaluation inputs.
    """
    import json
    from pathlib import Path

    if checkpoint not in ("official", "prediction"):
        raise ValueError("unknown prediction checkpoint")
    summaries = []
    for model in models:
        weight = float(model["weight"])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("prediction summary weights must be finite and nonnegative")
        summary = {"model_id": model["model_id"], "weight": weight,
                   "mean_max": None, "mean_argmax_q_index": None, "mean_median": None}
        if "component" in model:
            summary["component"] = model["component"]
        if model.get("excluded", False):
            summary["excluded"] = True
        if model.get("means") is not None:
            means = np.asarray(model["means"], dtype=float)
            if means.ndim != 1 or not means.size:
                raise ValueError("prediction summary needs a nonempty vector of per-input means")
            indices = np.arange(len(means))
            if "excluded_q_indices" in model:
                excluded = np.asarray(model["excluded_q_indices"])
                if (excluded.ndim != 1 or (excluded.size and excluded.dtype.kind not in "iu")
                        or np.any(excluded < 0) or np.any(excluded >= len(means))
                        or len(np.unique(excluded)) != len(excluded)):
                    raise ValueError("excluded prediction indices must be distinct valid Q indices")
                indices = np.setdiff1d(indices, excluded)
                summary["excluded_q_indices"] = excluded.astype(int).tolist()
                summary["mean_summary_scope"] = "retained_q"
            if "weights_by_q" in model:
                weights = np.asarray(model["weights_by_q"], dtype=float)
                if (weights.shape != means.shape or not np.all(np.isfinite(weights))
                        or np.any(weights < 0)):
                    raise ValueError("point-local prediction weights must match the Q vector")
                summary["weights_by_q"] = weights.tolist()
            retained = means[indices]
            nonfinite = int(np.count_nonzero(~np.isfinite(retained)))
            if nonfinite:
                summary["nonfinite_mean_count"] = nonfinite
            elif retained.size:
                ordered = np.sort(retained)
                middle = len(ordered) // 2
                median = float(ordered[middle])
                if len(ordered) % 2 == 0:
                    low, high = float(ordered[middle - 1]), median
                    # The ordinary (low + high) / 2 can overflow even when
                    # both means and their median are representable.
                    median = (low + (high - low) / 2 if (low < 0) == (high < 0)
                              else low / 2 + high / 2)
                summary.update(mean_max=float(np.max(retained)),
                               mean_argmax_q_index=int(indices[np.argmax(retained)]),
                               mean_median=median)
        elif not model.get("excluded", False):
            raise ValueError("a retained model needs predictive means")
        summaries.append(summary)
    row = {"round": int(round_index), "checkpoint": checkpoint,
           "observation_count": int(observation_count), "models": summaries}
    with (Path(output_dir) / "prediction_pool_summary.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _run_guarded_smc_call(function, *args, **kwargs):
    """Keep warning promotion scoped to the sampler, including its fork workers."""
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="invalid value encountered",
                                    category=RuntimeWarning)
            return function(*args, **kwargs)
    except CountInferenceFailure:
        raise
    except RuntimeWarning as exc:
        cause = classify_inference_failure(exc, "sampling")
        if cause != "nonfinite_smc_weights":
            raise
        raise CountInferenceFailure(
            cause, "sampling", f"{type(exc).__name__}: {exc}",
            {"module": type(exc).__module__, "class": type(exc).__qualname__,
             "message": str(exc), "traceback": _chained_traceback(exc)},
        ) from exc


def run_with_fit_guards(function, *args, fit_guards="off", **kwargs):
    """Promote invalid-arithmetic warnings only during the actual SMC call.

    Generic scalar/vector fit adapters retain their native builders and result
    handling. A temporary sampler wrapper installs the filter only when those
    adapters call pm.sample_smc; build/compile warnings retain their ordinary
    behavior. Linux fork workers inherit the sampler's temporary filter.

    Fitting is serial within each runner process; this process-global sampler
    wrapper, like Python warning filters, is not a thread-safe fit API. Nested
    calls restore the immediately enclosing sampler, and all exit paths restore
    it. The off path remains the original direct call.
    """
    if fit_guards == "off":
        return function(*args, **kwargs)
    if fit_guards != "v1":
        raise ValueError("fit_guards must be off or v1")
    import pymc as pm

    if function is pm.sample_smc:
        return _run_guarded_smc_call(function, *args, **kwargs)
    original_sampler = pm.sample_smc

    def guarded_sampler(*sampler_args, **sampler_kwargs):
        """Call the original pm.sample_smc under the warning guard."""
        return _run_guarded_smc_call(original_sampler, *sampler_args, **sampler_kwargs)

    pm.sample_smc = guarded_sampler
    try:
        return function(*args, **kwargs)
    finally:
        pm.sample_smc = original_sampler


def run_with_fit_timeout(function, *args, fit_timeout_seconds=None, **kwargs):
    """Bound a fit and reap its SMC workers before returning to the runner.

    The omitted-option path is a direct call. Opt-in execution uses a Linux
    subreaper supervisor and a separate process group for the fit. Killing just
    the fit would orphan PyMC's multiprocessing workers; the supervisor instead
    kills the group and waits for every adopted descendant, including zombies.
    Neither the caller's process group nor its child-reaping policy is changed.
    """
    seconds = validate_fit_timeout(fit_timeout_seconds)
    if seconds is None:
        return function(*args, **kwargs)

    import ctypes
    import os
    import pickle
    import signal
    import sys
    from multiprocessing import Pipe

    if sys.platform != "linux":
        raise RuntimeError("fit timeout requires Linux process-group/subreaper support")
    started = time.monotonic()
    receiver, sender = Pipe(duplex=False)
    caller_pid = os.getpid()
    supervisor = os.fork()
    if supervisor == 0:
        receiver.close()
        worker = None
        result_reader = result_writer = None
        message = None
        try:
            # An outer killpg(SIGKILL) must not kill this cleanup supervisor.
            # Its caller's death is delivered separately through prctl below.
            os.setsid()
            signal.signal(signal.SIGINT, signal.default_int_handler)
            signal.signal(signal.SIGTERM, signal.default_int_handler)
            # Both prctl settings are scoped to this short-lived supervisor.
            # Parent-death notification also covers an outer driver SIGKILL;
            # compare ppid after setting it to close the early-death race.
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "cannot enable fit caller death notification")
            if os.getppid() != caller_pid:
                raise KeyboardInterrupt("fit caller exited")
            if libc.prctl(36, 1, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "cannot enable fit worker subreaper")
            result_reader, result_writer = Pipe(duplex=False)
            # Do not allow cancellation between fork and storing the worker PID:
            # cleanup must always know the process group it owns.
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            worker = os.fork()
            if worker == 0:
                try:
                    sender.close()
                    result_reader.close()
                    os.setpgid(0, 0)
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                    try:
                        result = function(*args, **kwargs)
                        payload = ("result", result)
                    except BaseException as exc:  # noqa: BLE001 - Relay all worker failures and cancellation while preserving cleanup.
                        payload = ("error", type(exc), exc.args, exc.__dict__, _chained_traceback(exc))
                    result_writer.send_bytes(pickle.dumps(payload))
                finally:
                    result_writer.close()
                    os._exit(0)
            result_writer.close()
            # Set from both processes so even a near-zero limit has a group.
            try:
                os.setpgid(worker, worker)
            except ProcessLookupError:
                pass
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            remaining = max(0.0, seconds - (time.monotonic() - started))
            if result_reader.poll(remaining):
                message = ("payload", result_reader.recv_bytes())
            else:
                message = ("timeout", time.monotonic() - started)
        except BaseException as exc:  # noqa: BLE001 - Relay all worker failures and cancellation while preserving cleanup.
            message = ("payload", pickle.dumps(
                ("error", type(exc), exc.args, exc.__dict__, _chained_traceback(exc))))
        finally:
            # A second outer cancellation must not interrupt descendant reaping.
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            if worker is not None:
                try:
                    os.killpg(worker, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                # Adopted descendants are reaped here, including zombies.
                while True:
                    try:
                        os.waitpid(-1, 0)
                    except InterruptedError:
                        continue
                    except ChildProcessError:
                        break
            if result_reader is not None:
                result_reader.close()
        try:
            sender.send(message)
        finally:
            sender.close()
            os._exit(0)
    sender.close()
    try:
        message = receiver.recv()
    except BaseException:
        # SIGINT unwinds the supervisor's finally and reaps its worker group.
        try:
            os.kill(supervisor, signal.SIGINT)
        except ProcessLookupError:
            pass
        raise
    finally:
        receiver.close()
        while True:
            try:
                os.waitpid(supervisor, 0)
                break
            except InterruptedError:
                continue
    if message[0] == "timeout":
        raise FitTimeout(message[1], seconds)
    payload = pickle.loads(message[1])
    if payload[0] == "result":
        return payload[1]
    _, exception_type, exception_args, attributes, remote_traceback = payload
    # Typed failure constructors require more than their RuntimeError args.
    try:
        exception = exception_type(*exception_args)
    except TypeError:
        exception = exception_type.__new__(exception_type)
        BaseException.__init__(exception, *exception_args)
    exception.__dict__.update(attributes)
    raise exception from RuntimeError("isolated fit traceback:\n" + remote_traceback)


# Where the classifier is allowed to look for a NUMERICAL cause. A frame outside
# these is a frame in our own code or in the harness, and a failure there is a
# bug or an environment problem, not a numerical failure of the model.
SMC_NUMERIC_FRAMES = ("pymc/smc/kernels.py", "pymc/smc/sampling.py", "pymc/step_methods")


def _chained_traceback(exc):
    """Full chained traceback TEXT, including a remote worker's.

    `pm.sample_smc(cores=2)` runs the kernel in a worker process, so the frames
    that actually failed exist only as the text of a
    `concurrent.futures.process._RemoteTraceback` on the cause chain. Reading the
    text is therefore the only way to see `pymc/smc/kernels.py` at all.
    """
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, chain=True))



def _has_smc_warning_frame(exc):
    """Check frame filenames, not an incidental filename in a warning message.

    Fork sampling and the fit-timeout supervisor relay remote frame text through
    their own exception wrappers. Parse only those wrappers' File entries;
    ordinary exception messages and source-code lines are not frame evidence.
    """
    import re

    pending, seen = [exc], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for frame in traceback.extract_tb(current.__traceback__):
            if any(path in frame.filename for path in SMC_NUMERIC_FRAMES):
                return True
        remote = (type(current).__module__ == "concurrent.futures.process"
                  and type(current).__name__ == "_RemoteTraceback")
        supervisor = (type(current) is RuntimeError
                      and str(current).startswith("isolated fit traceback:\n"))
        if remote or supervisor:
            for line in str(current).splitlines():
                match = re.match(r'^\s*File "([^"]+)", line [0-9]+, in ', line)
                if match and any(path in match.group(1) for path in SMC_NUMERIC_FRAMES):
                    return True
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif not current.__suppress_context__ and current.__context__ is not None:
            pending.append(current.__context__)
    return False


def classify_inference_failure(exc, stage):
    """Name the numerical cause of `exc`, or return None to let it propagate.

    An exception CLASS is not
    evidence. PyMC 5.13.1 raises plain `ValueError` for a bad `threshold`, a
    broadcast mismatch, a malformed seed and a wrong return format just as
    readily as for the SMC weight collapse, and `pymc.exceptions
    .IncorrectArgumentsError` is itself a ValueError subclass. So the class is
    only ever a precondition here; the decision is made on the FRAMES the failure
    came from plus the concrete cause in the message.
    """
    text = _chained_traceback(exc)
    message = str(exc)
    in_smc = any(frame in text for frame in SMC_NUMERIC_FRAMES)
    module = type(exc).__module__

    if (stage == "sampling" and isinstance(exc, RuntimeWarning)
            and message.startswith("invalid value encountered") and _has_smc_warning_frame(exc)):
        return "nonfinite_smc_weights"
    # The original incident: every particle at -inf collapses the normalized log
    # weights to NaN and the effective-sample-size integer conversion raises.
    if (
        isinstance(exc, ValueError)
        and "cannot convert float NaN to integer" in message
        and "pymc/smc/kernels.py" in text
        and ("update_beta_and_weights" in text or "ESS = int(" in text)
    ):
        return "smc_weight_normalization_nonfinite"
    # The installed SMC kernels raise this when the tuned proposal covariance is
    # not finite. Same file, different numerical failure.
    if isinstance(exc, ValueError) and "Sample covariances not valid" in message and in_smc:
        return "smc_covariance_nonfinite"
    if module in ("numpy.linalg", "scipy.linalg") and in_smc:
        return "smc_linear_algebra_failure"
    if isinstance(exc, FloatingPointError) and in_smc:
        return "smc_floating_point_failure"
    if isinstance(exc, OverflowError) and in_smc:
        return "smc_numeric_overflow"
    if (
        type(exc).__name__ == "SamplingError"
        and module.startswith("pymc")
        and ("non-finite" in message or "not finite" in message or "Initial evaluation" in message)
    ):
        return "sampling_nonfinite_state"
    if type(exc).__name__ == "ParameterValueError" and module.startswith("pymc"):
        return "likelihood_parameter_domain"
    # Our own explicit output checks, raised below AFTER the return contract of
    # the sampler result has been verified. `stage` is what separates them from a
    # same-message failure raised anywhere else.
    if stage == "output_validation" and isinstance(exc, _OutputValueFailure):
        return exc.cause_code
    return None


class _OutputValueFailure(ValueError):
    """A VALUE in a structurally valid sampler result is not usable.

    `detail` is merged into the durable record so a value failure -
    which has no original exception to preserve - still says exactly WHICH
    numbers, on which chain, at which stage, failed the condition.
    """

    def __init__(self, cause_code, message, detail=None):
        super().__init__(message)
        self.cause_code = cause_code
        self.detail = detail or {}


@dataclass(frozen=True)
class SMCSettings:
    """SMC sampler settings: particles per chain, number of chains and worker cores (1 or 2)."""

    particles: int = 500
    chains: int = 2
    cores: int = 2

    def __post_init__(self):
        if self.particles < 16 or self.chains < 1 or not 1 <= self.cores <= 2:
            raise ValueError("invalid SMC particles/chains/cores")


@dataclass
class Fit:
    """One hypothesis's SMC result: posterior particles, log evidence (total and per chain),
    diagnostics and wall-clock seconds."""

    hypothesis: Hypothesis
    particles: dict[str, np.ndarray]
    log_evidence: float
    per_chain_log_evidence: list[float] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    seconds: float = 0.0


def check_designs(u, interface):
    """Validate designs against the interface's declared input ranges.

    Returns the named-input mapping the expression will be evaluated over, or
    None for the CORE namespace, in which case the original `[0,1]` check applies
    unchanged.
    """
    if interface is None:
        if u.ndim != 1 or not np.all(np.isfinite(u)) or np.any((u < 0) | (u > 1)):
            raise ValueError("CORE input outside [0,1]")
        return None
    base = interface["inputs"]
    if u.ndim != (1 if len(base) == 1 else 2) or not np.all(np.isfinite(u)):
        raise ValueError("designs must be a finite vector or [observation, input] matrix")
    columns = u[:, None] if u.ndim == 1 else u
    for index, item in enumerate(base):
        low, high = (float(v) for v in item["range"])
        if np.any((columns[:, index] < low) | (columns[:, index] > high)):
            raise ValueError(f"input {item['name']} outside its declared range")
    return named_input_values(interface, u)


def build_model(hypothesis, u, y, n_trials=50, interface=None):
    """Build the PyMC model for a count hypothesis on designs `u` and observed counts `y`.

    Returns a `pm.Model` with the role priors, the sigmoid-link mean and the hypothesis's
    count likelihood (Bernoulli, binomial, beta-binomial or zero-inflated binomial).
    """
    import pymc as pm
    import pytensor.tensor as pt

    u, y = np.asarray(u, dtype=float), np.asarray(y)
    expected = 1 if interface is None or len(interface["inputs"]) == 1 else 2
    if u.ndim != expected or y.ndim != 1 or len(u) != len(y) or not len(y):
        raise ValueError("nonempty scalar observation vectors required")
    named = check_designs(u, interface)
    anchor = pt.as_tensor(u if named is None else np.asarray(named[interface["inputs"][0]["name"]]))
    # `{}` for every interface without a categorical input, so the model
    # graph below is unchanged for CORE and every earlier BoxingGym task.
    codes = {} if interface is None else input_index_codes(interface, u)
    shapes = dict(getattr(hypothesis, "parameter_shapes", ()) or ()) if codes else {}
    if not np.all(np.isfinite(y)) or np.any((y != np.floor(y)) | (y < 0) | (y > n_trials)):
        raise ValueError("observations outside integer count support")
    with pm.Model() as model:
        parameters = {}
        for name, role in hypothesis.parameters:
            # An ACCEPTED proposed prior
            # replaces this coefficient's role prior, and only for the
            # coefficient it was proposed for. With no proposal - the default
            # path, and every hypothesis a mock or a reply without proposed priors produces -
            # `coefficient_prior` IS `PRIOR_RULES[role]`, so this graph is
            # the pre-ablation one down to the argument values.
            family, a, b = coefficient_prior(hypothesis, name, role)
            distribution = getattr(pm, family)
            if name in shapes:
                # One length-n_levels vector coefficient whose entries SHARE the
                # single fixed role prior (Table 3 "standardised slope",
                # N(0,1)); no prior value changes, only the shape.
                parameters[name] = distribution(
                    name, a, b, shape=shapes[name], transform=None
                )
            else:
                parameters[name] = distribution(name, a, b, transform=None)
        # NO in-graph NaN guard. An earlier version carried one, and it was
        # removed because it was not worth its own claim: it caught only a NaN in
        # the MEAN EXPRESSION, it did not prevent a NaN observed log density, it
        # did nothing at all about the every-particle-at-minus-infinity case that
        # actually killed the two paid runs, and its firing was never recorded -
        # so a run could not tell whether it had fired. The honest contract is
        # the one stated here:
        #
        #   NumPy `count_models.particle_logpmf`: a NaN cell IS -inf, and every
        #     substitution is counted (`nan_substitution_counter`).
        #   PyMC: nothing is guarded in the graph. A non-finite or
        #     all-particles--inf SMC run raises, `fit` re-raises it as
        #     `CountInferenceFailure`, and `core_count` turns that into a
        #     structured failure record plus a model retirement.
        p = pm.math.sigmoid(
            evaluate(hypothesis.expression, parameters, anchor, pt, named, codes)
        )
        if hypothesis.family == "bernoulli":
            # The paper's task table, row 5: "| Binary | logit | Bernoulli |
            # none (probit/cloglog are link alternatives, not families; not
            # counted as F) |". No dispersion parameter, no revision, one trial.
            if n_trials != BINARY_TRIALS:
                raise ValueError("a Bernoulli likelihood has exactly one trial")
            pm.Bernoulli("observed", p=p, observed=y)
        elif hypothesis.family == "binomial":
            pm.Binomial("observed", n=n_trials, p=p, observed=y)
        elif hypothesis.family == "beta_binomial":
            # The dispersion rows follow the ACTIVE role-prior
            # template like every other row, instead of restating Table 3 here.
            # `default` is Beta(1,9), which is what this line has always built.
            rho = pm.Beta("rho", *dispersion_prior("rho"), transform=None)
            kappa = 1 / rho - 1
            pm.BetaBinomial(
                "observed", n=n_trials, alpha=p * kappa, beta=(1 - p) * kappa, observed=y
            )
        else:
            # Same role-prior template decision as for `rho`; `default` is Beta(1,4).
            pi = pm.Beta("pi", *dispersion_prior("pi"), transform=None)
            pm.ZeroInflatedBinomial("observed", n=n_trials, p=p, psi=1 - pi, observed=y)
    return model



def loglik_prior_probe(hypothesis, u, y, interface=None, count=64, seed=None, *,
                       prior_draws=None, model_builder=None, n_trials=50):
    """Evaluate the actual model logp at prior points, without running SMC.

    ``prior_draws`` lets a runner reuse its existing identity-seeded probe
    samples exactly, including vector coefficients and dispersion parameters.
    ``model_builder`` is a zero-argument adapter for positive/real likelihoods;
    omitted, the count builder is used. Compile, input and shape errors are not
    converted into scientific rejection. Model logp includes the declared prior
    as requested by the shared PyMC compile_logp contract. The diagnostic uses
    only the observed data passed by the caller, never held-out data.
    """
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("prior probe count must be a positive integer")
    model = (build_model(hypothesis, u, y, n_trials, interface)
             if model_builder is None else model_builder())
    if prior_draws is None:
        import pymc as pm

        drawn = pm.draw(model.free_RVs, draws=int(count), random_seed=seed)
        prior_draws = {rv.name: values for rv, values in zip(model.free_RVs, drawn, strict=True)}
    names = [variable.name for variable in model.value_vars]
    if set(prior_draws) != set(names):
        raise ValueError("prior probe draws must match the model value-variable coordinates")
    draws = {name: np.asarray(prior_draws[name]) for name in names}
    if any(values.ndim == 0 or len(values) != count for values in draws.values()):
        raise ValueError("prior probe draws must have count entries on their first axis")
    logp = model.compile_logp()
    values = np.asarray([float(logp({name: draws[name][i] for name in names}))
                         for i in range(int(count))])
    nan_count = int(np.count_nonzero(np.isnan(values)))
    neg_inf_count = int(np.count_nonzero(np.isneginf(values)))
    pos_inf_count = int(np.count_nonzero(np.isposinf(values)))
    reason = None
    if nan_count or pos_inf_count:
        reason = "nonfinite_loglik_at_prior"
    elif neg_inf_count == count:
        reason = "degenerate_loglik_at_prior"
    return {"passed": reason is None, "reason": reason, "n_samples": int(count),
            "finite_fraction": float(np.mean(np.isfinite(values))),
            "nan_count": nan_count, "neg_inf_count": neg_inf_count,
            "pos_inf_count": pos_inf_count}


def _final_stage_log_evidence(idata):
    """Each chain's evidence AT ITS OWN FINAL SMC STAGE.

    `pymc/smc/kernels.py` (5.13.1) records, once per stage,

        "log_marginal_likelihood": self.log_marginal_likelihood if self.beta == 1 else np.nan
        "beta": self.beta

    so every NON-terminal stage is NaN BY CONSTRUCTION and the terminal stage is
    the one at which beta reached 1. The shared `inference/smc._extract_logz`
    takes the last FINITE value per chain, which coincides with the terminus only
    while the terminus is finite - and silently returns an EARLIER stage's
    evidence when it is not. That is the defect this function removes; the shared
    extractor itself is untouched and still owns the combination.

    Returns the per-chain terminal values. Raises:
      * `SMCError` / `RuntimeError` - the RETURN CONTRACT is broken: `beta` is
        absent so no terminus can be identified at all (a counterexample
        showed that an earlier version's declared degradation, which
        substituted an earlier per-chain value here instead of failing the
        run, was wrong), stats cannot be mapped to chains, a chain has no finite beta, or
        its last finite beta is not 1.0, i.e. tempering never terminated.
        Run-level: not a statement about the model.
      * `_OutputValueFailure("nonfinite_final_evidence")` - the contract holds and
        the terminal VALUE is NaN or +-inf on at least one chain. Typed.
    """
    from ..inference.smc import _chain_units

    n_chains = int(idata.posterior.sizes["chain"])
    evidence = _chain_units(
        np.asarray(idata.sample_stats["log_marginal_likelihood"].values, dtype=object), n_chains
    )

    def stages(unit):
        """Flatten one chain's per-stage values to floats, with None read as NaN."""
        return [
            float(v) if v is not None else np.nan for v in np.asarray(unit, dtype=object).ravel()
        ]

    if "beta" not in idata.sample_stats:
        # A design review found a counterexample: an earlier version
        # left a "declared degradation" here that fell back to the shared
        # extractor's last-finite-value rule - so a chain with SOME finite
        # evidence was returned as though that were the terminal value, exactly
        # the substitution that version had just forbidden for the beta-PRESENT case.
        # pymc 5.13.1 always records `beta` alongside the evidence
        # (kernels.py:316-317), so this shape never occurs against the
        # installed version; it IS the return-contract violation. Without
        # `beta` no terminus can be identified at all, so there is no old
        # per-chain value this function is entitled to reuse - the whole run
        # fails instead, never the typed `CountInferenceFailure` reserved for
        # model retirement.
        stage_counts = [len(stages(unit)) for unit in evidence]
        raise RuntimeError(
            "SMC result is missing sample_stats.beta: cannot identify the final "
            f"stage of any of its {n_chains} chain(s) (stage counts per chain: "
            f"{stage_counts}) - a return-contract violation is a run failure, "
            "never a substitution of an earlier per-chain evidence value"
        )

    betas = _chain_units(np.asarray(idata.sample_stats["beta"].values, dtype=object), n_chains)
    finals, indices = [], []
    for chain, (beta_unit, evidence_unit) in enumerate(zip(betas, evidence, strict=True)):
        beta_stages, evidence_stages = stages(beta_unit), stages(evidence_unit)
        if len(beta_stages) != len(evidence_stages):
            raise RuntimeError(
                f"chain {chain}: {len(beta_stages)} beta stages vs "
                f"{len(evidence_stages)} evidence stages"
            )
        # the last FINITE beta: ragged padding after the final stage is NaN
        finite = [i for i, value in enumerate(beta_stages) if math.isfinite(value)]
        if not finite:
            raise RuntimeError(f"chain {chain}: no finite beta stage in the SMC result")
        index = finite[-1]
        if beta_stages[index] != 1.0:
            raise RuntimeError(
                f"chain {chain}: the final SMC stage has beta={beta_stages[index]!r}, not 1.0 - "
                "tempering did not terminate, so there is no final evidence to read"
            )
        finals.append(evidence_stages[index])
        indices.append(index)

    if not all(math.isfinite(value) for value in finals):
        raise _OutputValueFailure(
            "nonfinite_final_evidence",
            "SMC returned non-finite evidence at the final stage "
            f"(beta=1) of chain(s) {[c for c, v in enumerate(finals) if not math.isfinite(v)]}",
            {
                "chains": list(range(len(finals))),
                "final_stage_index": indices,
                "final_log_marginal_likelihood": finals,
                "failed_condition": "math.isfinite(log_marginal_likelihood[beta == 1]) for every chain",
            },
        )
    return finals


def _check_return_contract(idata, names, settings):
    """Is the sampler result SHAPED the way this version promises?

    Deliberately raises a plain exception: a broken return contract is a
    library/version/environment problem, so it must reach the run-level failure
    path, never `CountInferenceFailure`. It runs BEFORE any value check so that
    "non-finite posterior" can only ever be said about a result we could read.
    """
    if not hasattr(idata, "posterior") or not hasattr(idata, "sample_stats"):
        raise RuntimeError("SMC result is missing the posterior/sample_stats groups")
    missing = [name for name in names if name not in idata.posterior]
    if missing:
        raise RuntimeError(f"SMC result is missing posterior variables {missing}")
    if "log_marginal_likelihood" not in idata.sample_stats:
        raise RuntimeError("SMC result is missing sample_stats.log_marginal_likelihood")
    chains = int(idata.posterior.sizes["chain"])
    if chains != settings.chains:
        raise RuntimeError(f"SMC returned {chains} chains, expected {settings.chains}")


def fit(hypothesis, u, y, seed, settings=None, interface=None, n_trials=50,
        fit_timeout_seconds=None, fit_guards="off"):
    """Fit one count hypothesis to (u, y) with PyMC SMC and return a `Fit`.

    Optionally runs under a wall-clock timeout and/or warning guards. A numerical
    failure the classifier can name becomes `CountInferenceFailure`; any other
    exception propagates unchanged.
    """
    if fit_guards not in ("off", "v1"):
        raise ValueError("fit_guards must be off or v1")
    if fit_timeout_seconds is not None:
        return run_with_fit_timeout(
            fit, hypothesis, u, y, seed, settings, interface, n_trials,
            fit_timeout_seconds=fit_timeout_seconds,
            **({"fit_guards": fit_guards} if fit_guards != "off" else {}),
        )
    import pymc as pm

    from ..inference.smc import _extract_diagnostics, _extract_logz

    settings = SMCSettings() if settings is None else settings
    started = time.perf_counter()
    model = build_model(hypothesis, u, y, n_trials, interface)
    # ONLY these two stages may end in a typed
    # `CountInferenceFailure`, and only for a cause `classify_inference_failure`
    # can NAME. Everything the caller does afterwards - diagnostics, the prior
    # probe, the NPZ/JSON artifacts - is outside on purpose, and an unclassified
    # exception is re-raised in its original form so the run fails.
    #
    #   stage "sampling"          - the SMC call itself
    #   stage "output_validation" - the VALUES in a structurally valid result,
    #                               checked only after the return CONTRACT
    #                               (groups, names, shapes) has been verified;
    #                               a contract violation is a run-level failure,
    #                               never a statement about the model.
    names = tuple(name for name, _ in hypothesis.parameters) + hypothesis.noise_parameters
    stage = "sampling"
    try:
        with model:
            if fit_guards == "off":
                idata = pm.sample_smc(
                    draws=settings.particles,
                    chains=settings.chains,
                    cores=settings.cores,
                    random_seed=seed,
                    progressbar=False,
                    compute_convergence_checks=False,
                )
            else:
                idata = run_with_fit_guards(
                    pm.sample_smc, fit_guards=fit_guards,
                    draws=settings.particles, chains=settings.chains, cores=settings.cores,
                    random_seed=seed, progressbar=False, compute_convergence_checks=False,
                )
        _check_return_contract(idata, names, settings)
        stage = "output_validation"
        # The terminal values are established FIRST, so a non-finite
        # final evidence is a typed numerical failure instead of escaping as the
        # shared extractor's `SMCError`, and an earlier finite stage can never be
        # substituted for it. Only once every terminal value is finite does the
        # shared extractor own the combination - and its per-chain values must
        # then be exactly the terminal ones, or the contract is broken.
        finals = _final_stage_log_evidence(idata)
        logz, per_chain = _extract_logz(idata)
        if finals is not None and list(per_chain) != finals:
            raise RuntimeError(
                f"the shared extractor returned {list(per_chain)} where the final SMC "
                f"stage of each chain holds {finals}"
            )
        # An INDEXED coefficient's posterior is [chain, draw, level], so
        # it is flattened over chain/draw ONLY - a plain `reshape(-1)` would
        # interleave levels into the particle axis. `parameter_shapes` is empty
        # for every hypothesis without an indexed coefficient, so this is the
        # unchanged line for CORE and every earlier task.
        shapes = dict(getattr(hypothesis, "parameter_shapes", ()) or ())
        particles = {
            name: (
                np.asarray(idata.posterior[name]).reshape(-1, shapes[name])
                if name in shapes
                else np.asarray(idata.posterior[name]).reshape(-1)
            )
            for name in names
        }
        if not all(np.all(np.isfinite(values)) for values in particles.values()):
            raise _OutputValueFailure("nonfinite_posterior", "SMC returned non-finite particles")
        if not np.isfinite(logz):
            raise _OutputValueFailure(
                "nonfinite_final_evidence", "SMC returned non-finite final evidence"
            )
    except CountInferenceFailure:
        raise  # already typed: never double-wrap, never lose the first traceback
    except Exception as exc:
        cause_code = classify_inference_failure(exc, stage)
        if cause_code is None:
            raise  # unclassified: propagate in its ORIGINAL form, fail the run
        raise CountInferenceFailure(
            cause_code,
            stage,
            f"{type(exc).__name__}: {exc}",
            {
                "module": type(exc).__module__,
                "class": type(exc).__qualname__,
                "message": str(exc),
                "traceback": _chained_traceback(exc),
                **getattr(exc, "detail", {}),
            },
        ) from exc
    diag = _extract_diagnostics(idata)
    diag.update(
        seed=int(seed),
        particles_per_chain=settings.particles,
        chains=settings.chains,
        cores=settings.cores,
        refit="original_prior_and_all_public_observations",
        observation_count=len(y),
        particle_weighting="equal_within_and_across_chains",
    )
    return Fit(hypothesis, particles, float(logz), per_chain, diag, time.perf_counter() - started)


def model_logweights(fits):
    """Return normalized log posterior model weights from the fits' log evidences (uniform model prior)."""
    if not fits or len({f.hypothesis.model_id for f in fits}) != len(fits):
        raise ValueError("empty pool or duplicated canonical hypothesis")
    logz = np.asarray([f.log_evidence for f in fits])
    if not np.all(np.isfinite(logz)):
        raise ValueError("non-finite evidence")
    # Uniform over distinct active (expression, family) pairs, conditional on this pool.
    return logz - logsumexp(logz)


def mixture_logpmf(component_logpmfs, logweights):
    """Combine per-model [input, count] log-PMF tables into the weighted mixture log-PMF."""
    logweights = np.asarray(logweights)
    if not np.all(np.isfinite(logweights)) or not np.isclose(logsumexp(logweights), 0):
        raise ValueError("mixture weights must be finite and normalized")
    table = np.stack(component_logpmfs)
    if len(logweights) != table.shape[0]:
        raise ValueError("weight and component count mismatch")
    return logsumexp(table + logweights[:, None, None], axis=0)


def predict(fits, u, interface=None, n_trials=50):
    """Posterior predictive log-PMFs at inputs `u`: the BMA mixture, the top-evidence
    single model, and every model's own log-PMF and log weight."""
    logweights = model_logweights(fits)
    components = []
    for result in fits:
        # Zero-weight particle rule: a particle the normalization rule gave zero weight is not in
        # the posterior average at all, so it is out of both the sum and the
        # divisor. With no zeroed particle this is the same array and the same
        # count as before, hence the same floats.
        rows = active_particle_rows(
            particle_logpmf(result.hypothesis, result.particles, u, n_trials, interface)
        )
        components.append(logsumexp(rows, axis=0) - np.log(len(rows)))
    mixed = mixture_logpmf(components, logweights)
    # Stable ID breaks evidence ties, independent of test scores.
    best = min(range(len(fits)), key=lambda i: (-logweights[i], fits[i].hypothesis.model_id))
    return {
        "bma_logpmf": mixed,
        "single_logpmf": components[best],
        "single_model_id": fits[best].hypothesis.model_id,
        "model_logpmfs": np.stack(components),
        "model_logweights": logweights,
    }


def predictive_summary(logpmf, rng, sample_count=4):
    """Summarize count predictive PMFs per input: mean, 90% interval bounds and a few draws."""
    pmf = np.exp(logpmf)
    cdf = np.cumsum(pmf, axis=-1)
    draws = np.minimum(
        (rng.random((len(pmf), sample_count, 1)) > cdf[:, None, :]).sum(-1), pmf.shape[-1] - 1
    )
    return {
        "mean": pmf @ np.arange(pmf.shape[-1]),
        "lower90": (cdf < 0.05).sum(-1),
        "upper90": (cdf < 0.95).sum(-1),
        "samples": draws,
    }


def information_from_components(component_pmfs, model_weights):
    """Each component is [particle, input, count]; sum over the complete support."""

    def entropy(p):
        """Shannon entropy in nats over the last (count) axis."""
        return -np.sum(xlogy(p, p), axis=-1)

    weights = np.asarray(model_weights)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("invalid model weights")
    for bank in component_pmfs:
        if not np.all(np.isfinite(bank)) or np.any(bank < 0) or not np.allclose(bank.sum(-1), 1):
            raise ValueError("invalid component PMF bank")
    model_predictions = np.stack([bank.mean(axis=0) for bank in component_pmfs])
    mixture = np.einsum("m,mxy->xy", weights, model_predictions)
    conditional = sum(
        w * entropy(bank).mean(axis=0) for w, bank in zip(weights, component_pmfs, strict=True)
    )
    model_entropy = weights @ entropy(model_predictions)
    total = entropy(mixture) - conditional
    structure = entropy(mixture) - model_entropy
    parameter = model_entropy - conditional
    if any(np.any(~np.isfinite(v)) or np.any(v < -1e-9) for v in (total, structure, parameter)):
        raise ValueError("non-finite or negative information gain")
    return total, structure, parameter


def choose(fits, designs, seed, posterior_samples=128, interface=None, n_trials=50):
    """Score candidate designs by joint, structure and parameter EIG on subsampled particles.

    Returns the chosen design and one row per candidate.
    """
    rng = np.random.default_rng(seed)
    banks = []
    for result in fits:
        size = len(next(iter(result.particles.values())))
        indices = rng.choice(size, size=min(size, posterior_samples), replace=False)
        subset = {k: v[indices] for k, v in result.particles.items()}
        # Zero-weight particle rule: a zero-weight particle is an all-zero PMF row, which
        # `information_from_components` refuses as an invalid bank - it leaves
        # the bank here, before the check, and the model's own average is over
        # the particles that remain.
        banks.append(
            np.exp(
                active_particle_rows(
                    particle_logpmf(result.hypothesis, subset, designs, n_trials, interface)
                )
            )
        )
    total, structure, parameter = information_from_components(banks, np.exp(model_logweights(fits)))
    rows = [
        {
            "design": [float(v) for v in np.atleast_1d(u)] if np.ndim(u) else float(u),
            "eig": float(t),
            "structure_eig": float(m),
            "parameter_eig": float(p),
        }
        for u, t, m, p in zip(designs, total, structure, parameter, strict=True)
    ]
    return choose_design(rows)["chosen_design"], rows


def retain(fits, maximum=12):
    """Split fits into the `maximum` highest-evidence ones (ties by model ID) and the rest."""
    model_logweights(fits)
    ordered = sorted(fits, key=lambda f: (-f.log_evidence, f.hypothesis.model_id))
    return ordered[:maximum], ordered[maximum:]

# EPIG acquisition: fixed once-per-run predictive targets; no evaluation data enters here.
EPIG_TARGET_POINTS = 32
EPIG_Y_GRID = 64
EPIG_TARGETS_SEED_STREAM = "epig_targets"
EPIG_DISCRETE_TAIL_TOLERANCE = 1e-12
EPIG_DISCRETE_SUPPORT_CAP = 100000
EPIG_DISCRETE_MAX_WORKING_BYTES = 256 * 1024 * 1024


def resolve_epig_policy(cell, policy=None):
    """Resolve an omitted policy and reject ambiguous acquisition declarations."""
    import re

    if not isinstance(cell, str) or re.fullmatch(r"[01][01][012]", cell) is None:
        raise ValueError("cell must be a three-character string matching [01][01][012]")
    effective = ("epig" if cell[2] == "2" else "joint_eig") if policy is None else policy
    if (cell[2] == "2") != (effective == "epig"):
        raise ValueError("design policy epig requires cell[2]=2, and cell[2]=2 requires epig")
    return effective


COORDINATE_SHA256_RULE = (
    "sha256(coordinates_v1\n + shape_as_ascii_nxd + \n + "
    "ordered_C_little_endian_float64_rows); scalar_inputs_as_n_by_1; signed_zero_normalized"
)


def coordinate_set_sha256(values):
    """Hash ordered coordinate rows, preserving multiplicity and feature order."""
    import hashlib

    rows = np.asarray(values, dtype="<f8")
    if rows.ndim == 1:
        rows = rows[:, None]
    if rows.ndim != 2 or not np.all(np.isfinite(rows)):
        raise ValueError("coordinate hash requires a finite point-by-input array")
    rows = np.array(rows, dtype="<f8", order="C", copy=True)
    rows[rows == 0] = 0.0
    header = f"coordinates_v1\n{rows.shape[0]}x{rows.shape[1]}\n".encode("ascii")
    return hashlib.sha256(header + rows.tobytes(order="C")).hexdigest()


CANDIDATE_GENERATORS = {
    "core": {
        "name": "numpy.linspace(0,1,profile.grid_size)",
        "seed_stream": "none; deterministic_grid",
        "sha256_rule": COORDINATE_SHA256_RULE,
        "hash_scope": "generated_pool_before_history_filters",
        "selection_rule": "all_three_policies_select_from_the_same_grid",
    },
    "boxinggym": {
        "name": "boxinggym_core.build_design_pool",
        "seed_stream": "sampled:boxinggym_design_pool; fixed_or_categorical:boxinggym_world; grid:none",
        "sha256_rule": COORDINATE_SHA256_RULE,
        "hash_scope": "generated_pool_before_history_filters",
        "selection_rule": "task_design_exclusion_applies_to_each_policy_own_observation_history",
    },
    "chembench": {
        "name": "adapters.chembench_observation.sample_designs(profile.design_pool)",
        "seed_stream": "chem_design_pool_inputs",
        "sha256_rule": COORDINATE_SHA256_RULE,
        "hash_scope": "generated_pool_before_history_filters",
        "selection_rule": "EIG_EPIG_exclude_observed_pool_indices; random_uses_separate_chem_observation_designs_schedule",
        "random_schedule": "separate_predrawn_same_measure",
    },
}


def candidate_set_record(values, bundle):
    """Two explicit metadata fields for the generated pool, not its filtered subset."""
    return {"candidate_set_sha256": coordinate_set_sha256(values),
            "candidate_generator": dict(CANDIDATE_GENERATORS[bundle])}


def epig_config(targets, target_seed, sampling_rule, *, evaluation_points=None,
                target_measure=None, target_measure_note=None, target_count=EPIG_TARGET_POINTS):
    """Validate the run's EPIG target inputs and return the EPIG configuration record.

    With `evaluation_points`, the record also counts target/evaluation overlaps.
    """
    # Configurable EPIG target count (2026-10-05): `target_count` is the run's stated EPIG target count
    # (BoxingGym `--epig-target-points`); every existing caller keeps the default 32,
    # so its check, message and record are byte-identical to before.
    targets = np.asarray(targets, dtype=float)
    if len(targets) != target_count or not np.all(np.isfinite(targets)):
        raise ValueError(f"EPIG requires {target_count} finite target inputs sampled once per run")
    record = {
        "design_policy": "epig",
        "design_policy_target": "predictive_target_points",
        "epig_target_points": target_count,
        "epig_y_grid": EPIG_Y_GRID,
        "epig_targets_seed_stream": EPIG_TARGETS_SEED_STREAM,
        "epig_targets_seed": int(target_seed),
        "epig_targets_seed_rule": "seed_for('epig_targets', seed)",
        "epig_targets": targets.tolist(),
        "epig_targets_sampling_rule": sampling_rule,
        "epig_target_measure": target_measure or sampling_rule,
        "epig_target_seed_stream": EPIG_TARGETS_SEED_STREAM,
        "epig_targets_sha256": coordinate_set_sha256(targets),
        "epig_targets_sha256_rule": COORDINATE_SHA256_RULE,
        "epig_target_redraw": "once_at_run_start",
        "epig_units": "nats",
        "epig_outcome_grid": "64_BMA_predictive_quantile_intervals_including_both_tails_for_PDF",
        "epig_particle_rule": "same_seed_and_per_model_subsample_as_count_choose",
        "epig_finite_outcome_rule": "complete_bounded_count_or_binary_support",
        "epig_discrete_rule_scope": "unbounded_count_only",
        "epig_discrete_outcome_rule": "enumerate_integers_then_aggregated_tail",
        "epig_discrete_tail_tolerance": EPIG_DISCRETE_TAIL_TOLERANCE,
        "epig_discrete_support_cap": EPIG_DISCRETE_SUPPORT_CAP,
        "epig_discrete_max_working_bytes": EPIG_DISCRETE_MAX_WORKING_BYTES,
        "epig_discrete_memory_rule": "ragged_banks_and_one_candidate_target_pair_at_a_time",
        "epig_discrete_memory_estimate_rule": "retained_input_arrays_and_workspace_v2",
        "epig_discrete_tail_tolerance_is_mi_error_bound": False,
    }
    if evaluation_points is not None:
        target_rows = targets.reshape(len(targets), -1)
        evaluation = np.asarray(evaluation_points, dtype=float)
        evaluation = evaluation.reshape(len(evaluation), -1)
        if target_rows.shape[1] != evaluation.shape[1] or not np.all(np.isfinite(evaluation)):
            raise ValueError("EPIG target/evaluation inputs must share finite coordinate columns")
        evaluation_keys = {tuple(row) for row in evaluation}
        overlap = sum(tuple(row) in evaluation_keys for row in target_rows)
        record.update(epig_target_disjoint_from_evaluation=overlap == 0,
                      epig_target_evaluation_overlap_count=int(overlap),
                      epig_target_overlap_rule="exact_coordinate_rows; count_target_draws; no_resampling")
    if target_measure_note is not None:
        record["epig_target_measure_note"] = target_measure_note
    return record


def _epig_subsample(fits, seed, posterior_samples):
    """The exact count.choose particle stream, also shared across x and x*."""
    from dataclasses import replace

    if type(posterior_samples) is not int or posterior_samples < 1:
        raise ValueError("posterior_samples must be a positive integer")
    rng = np.random.default_rng(seed)
    subsets, weights = [], []
    for result, weight in zip(fits, np.exp(model_logweights(fits)), strict=True):
        size = len(next(iter(result.particles.values())))
        indices = rng.choice(size, size=min(size, posterior_samples), replace=False)
        subsets.append(replace(result, particles={k: v[indices] for k, v in result.particles.items()}))
        weights.append(np.full(len(indices), weight / len(indices)))
    return subsets, np.concatenate(weights)


def _epig_scalar_distribution(result, inputs):
    """Use the scalar path's native means/scales and full normalized families."""
    from scipy.stats import gamma, lognorm, nbinom, norm, poisson, t

    from .positive_inference import particle_means_masked

    interface = result.diagnostics.get("interface")
    if (interface is not None and len(interface["inputs"]) == 1
            and np.asarray(inputs).ndim == 2 and np.asarray(inputs).shape[1] == 1):
        inputs = np.asarray(inputs)[:, 0]
    mean, valid = particle_means_masked(result, inputs)
    # choose_epig has already applied one common-input exclusion mask. Keep
    # direct private-bank callers fail-closed rather than passing invalid
    # distribution parameters to SciPy.
    if not np.all(valid):
        raise ValueError("non-finite/nonpositive mean; no clipping fallback")
    mean = mean[:, :, None]
    family = result.hypothesis.family
    if family in ("normal", "student_t", "heteroscedastic_normal"):
        from .real_inference import STUDENT_T_DF, scale_values
        binding = result.diagnostics
        scale = scale_values(result.hypothesis, result.particles, inputs,
                             binding["interface"], binding["response_unit"])[:, :, None]
        return t(STUDENT_T_DF, loc=mean, scale=scale) if family == "student_t" else norm(mean, scale)
    if family in ("poisson", "zi_poisson"):
        return poisson(mean)
    if family == "negative_binomial":
        alpha = result.particles["alpha"][:, None, None]
        return nbinom(alpha, alpha / (alpha + mean))
    if family == "lognormal":
        sigma = result.particles["sigma"][:, None, None]
        return lognorm(sigma, scale=mean * np.exp(-sigma**2 / 2))
    if family == "gamma":
        shape = result.particles["shape"][:, None, None]
        return gamma(shape, scale=mean / shape)
    raise ValueError(f"EPIG has no scalar distribution for {family!r}")


class _EpigZeroInflatedQuantileDistribution:
    """The full ZIP law for interval CDFs; enumeration keeps its SF-aware path."""

    def __init__(self, distribution, pi):
        self.distribution = distribution
        self.pi = np.asarray(pi, dtype=float)[:, None, None]
        if np.any(~np.isfinite(self.pi)) or np.any((self.pi <= 0) | (self.pi >= 1)):
            raise ValueError("EPIG zero-inflation pi must lie in the open unit interval")

    def cdf(self, values):
        """CDF of the zero-inflated law: mixture of the base CDF and an atom at zero."""
        values = np.asarray(values, dtype=float)
        return (1 - self.pi) * self.distribution.cdf(values) + self.pi * (values >= 0)

    def ppf(self, probabilities):
        """Quantile function of the zero-inflated law (0 at or below the zero atom's mass)."""
        probabilities = np.asarray(probabilities, dtype=float)
        above_atom = probabilities > self.pi
        adjusted = np.where(above_atom, (probabilities - self.pi) / (1 - self.pi), .5)
        return np.where(above_atom, self.distribution.ppf(adjusted), 0.)


class _EpigBoundaryBank(np.ndarray):
    """Array-compatible bank carrying a nonempty quantile-repair receipt."""


def _epig_continuous_bank(fits, inputs, weights, grid_size, *, discrete=False):
    """CDF masses on BMA quantile intervals, including the two infinite tails.

    MI of the two binned future observations approximates continuous EPIG.
    A common grid at each input keeps every model and particle on one measure.
    Discrete mode includes the ZIP atom and returns the right-closed interval
    boundaries alongside the bank. Empty intervals retain zero probability.
    """
    if type(grid_size) is not int or grid_size < 2:
        raise ValueError("EPIG grid_size must be an integer of at least two")
    distributions = [_epig_scalar_distribution(result, inputs) for result in fits]
    if discrete:
        distributions = [
            _EpigZeroInflatedQuantileDistribution(distribution, result.particles["pi"])
            if result.hypothesis.family == "zi_poisson" else distribution
            for result, distribution in zip(fits, distributions, strict=True)
        ]
    probabilities = np.arange(1, grid_size, dtype=float) / grid_size
    quantiles = np.concatenate([d.ppf(probabilities) for d in distributions])
    lower, upper = np.min(quantiles, axis=0), np.max(quantiles, axis=0)
    del quantiles
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("non-finite posterior predictive quantile bracket")
    # Width alone is unsafe at discrete CDF jumps. Stop only once an update
    # leaves both float64 brackets bit-identical: every remaining deterministic
    # update would be identical too, including signed zeros and repair receipts.
    for _ in range(56):
        middle = lower + (upper - lower) / 2
        cdf = np.concatenate([d.cdf(middle[None, :, :]) for d in distributions])
        below = np.einsum("k,kxy->xy", weights, cdf) < probabilities
        next_lower, next_upper = np.where(below, middle, lower), np.where(below, upper, middle)
        converged = (np.max(next_upper - next_lower)
                     <= 1e-12 * max(1., float(np.max(np.abs(next_upper)))))
        fixed = (converged
                 and np.array_equal(next_lower.view(np.uint64), lower.view(np.uint64))
                 and np.array_equal(next_upper.view(np.uint64), upper.view(np.uint64)))
        lower, upper = next_lower, next_upper
        if fixed:
            break
    boundaries = lower + (upper - lower) / 2
    monotone = np.maximum.accumulate(boundaries, axis=-1)
    repaired_rows = np.any(monotone > boundaries, axis=-1)
    repairs = None
    if np.any(repaired_rows):
        repairs = {
            "rows_repaired": int(np.count_nonzero(repaired_rows)),
            "max_shift": float(np.max(monotone[repaired_rows] - boundaries[repaired_rows])),
            "path": "discrete" if discrete else "continuous",
        }
        boundaries = monotone
    # Already-monotone boundary bytes, including signed zeros, stay unchanged.
    del monotone
    banks = []
    for distribution in distributions:
        cdf = distribution.cdf(boundaries[None, :, :])
        banks.append(np.diff(np.concatenate([np.zeros_like(cdf[:, :, :1]), cdf,
                                            np.ones_like(cdf[:, :, :1])], axis=-1), axis=-1))
    bank = np.concatenate(banks)
    if repairs is not None:
        bank = bank.view(_EpigBoundaryBank)
        bank.boundary_repairs = repairs
    if discrete:
        if (np.any(~np.isfinite(bank)) or np.any(bank < 0)
                or not np.allclose(bank.sum(-1), 1., atol=1e-10, rtol=0)):
            raise ValueError("EPIG discrete interval normalization failed; no renormalization")
        return bank, boundaries
    return bank


class EpigResourceLimit(ValueError):
    """A declared acquisition allocation/support bound failed, with audit detail."""

    def __init__(self, detail):
        import json
        self.detail = detail
        super().__init__("EPIG resource refusal: " + json.dumps(detail, sort_keys=True))


def _epig_guard_particles(fits, inputs, weights):
    """Condition this acquisition's subsample on valid means at all x and x*.

    Positive/count scalar laws require finite positive link means; real-valued
    laws retain their finite-only contract. No fit, evidence, prior, or caller
    array is changed. An all-valid pool returns the original objects, avoiding
    any additional normalization or arithmetic change on the legacy path.
    """
    from dataclasses import replace

    from .positive_inference import particle_means_masked

    masks, excluded_particles, excluded_models = [], {}, []
    for result in fits:
        size = len(next(iter(result.particles.values())))
        keep = np.ones(size, dtype=bool)
        interface = result.diagnostics.get("interface")
        # Bound the guard's mean arrays before the discrete bank preflight.
        # Chunking only accumulates one round-wide mask; it never chooses a
        # different particle bank for different candidates or targets.
        chunk_size = min(16, max(1, EPIG_DISCRETE_MAX_WORKING_BYTES // (8 * 8 * size)))
        for start in range(0, len(inputs), chunk_size):
            mean_inputs = inputs[start:start + chunk_size]
            if (interface is not None and len(interface["inputs"]) == 1
                    and mean_inputs.ndim == 2 and mean_inputs.shape[1] == 1):
                mean_inputs = mean_inputs[:, 0]
            means, valid = particle_means_masked(result, mean_inputs)
            if (valid.ndim != 2 or valid.shape[0] != size
                    or valid.shape[1] not in (1, len(mean_inputs))):
                raise ValueError("EPIG particle mean mask must align with particles and inputs")
            keep &= np.all(valid, axis=1)
            del means, valid
        masks.append(keep)
        excluded = int(size - np.count_nonzero(keep))
        if excluded:
            model_id = result.hypothesis.model_id
            excluded_particles[model_id] = {
                "excluded": excluded, "of": size,
                "reason": "nonfinite_or_nonpositive_mean",
            }
            if excluded == size:
                excluded_models.append(model_id)
    if not excluded_particles:
        return fits, weights, {}
    info = {"epig_excluded_particles": excluded_particles}
    if excluded_models:
        info["epig_excluded_models"] = excluded_models
    if len(excluded_models) == len(fits):
        raise EpigResourceLimit({"resource": "positivity", **info,
                                 "epig_all_candidates_missing": True})
    retained = [result if np.all(keep) else replace(
        result, particles={name: values[keep] for name, values in result.particles.items()})
        for result, keep in zip(fits, masks, strict=True) if np.any(keep)]
    # Reuse the same evidence-derived model weights when only particles change.
    # With removed models, log-space normalization also handles survivors whose
    # former model mass underflowed to zero; no replacement or clipping is used.
    model_weights = np.exp(model_logweights(retained if excluded_models else fits))
    adjusted = np.concatenate([
        np.full(len(next(iter(result.particles.values()))), weight /
                len(next(iter(result.particles.values()))))
        for result, weight in zip(retained, model_weights, strict=True)
    ])
    return retained, adjusted, info


@dataclass
class EpigDiscretePathState:
    """One cell/component's route, selected from its first D0 acquisition bank.

    The runner owns this state across refits. Standalone diagnostic calls may
    omit it; no global cache or particle identity stands in for a cell boundary.
    """

    _path: str | None = field(default=None, init=False)

    @property
    def path(self):
        """The fixed discrete route ("enumeration" or "coarsened"), or None before the first bank."""
        return self._path

    def _fix(self, path):
        if path not in {"enumeration", "coarsened"}:
            raise ValueError("invalid EPIG discrete path")
        if self._path is not None and self._path != path:
            raise ValueError("EPIG discrete path is already fixed")
        self._path = path



class EpigRows(list):
    """List-compatible candidate rows carrying the declared outcome coarsening."""

    def __init__(self, rows, outcome_info, exclusion_info=None):
        super().__init__(rows)
        self.outcome_info = outcome_info
        self.exclusion_info = exclusion_info


def _epig_discrete_preflight(particle_count, cutoffs, candidate_count, *,
                             support_cap=EPIG_DISCRETE_SUPPORT_CAP,
                             max_working_bytes=EPIG_DISCRETE_MAX_WORKING_BYTES):
    """Plan finite tables without allocating a PMF bank or a joint outcome table.

    This bounds estimated acquisition arrays, not total process resident memory.
    Ragged banks retain each input's own K; one x/x* pair is scored at a time.
    The estimate explicitly includes live full-input quantiles, tail tables,
    residuals, and frozen distribution parameters. Additional allowances cover
    full-input calculations, PMF construction, and one pair's MI workspace.
    It is an acquisition-array estimate, not a total-process RSS guarantee.
    """
    if (type(particle_count) is not int or particle_count < 1
            or type(candidate_count) is not int or not 0 < candidate_count < len(cutoffs)
            or any(type(k) is not int or k < 0 for k in cutoffs)):
        raise ValueError("invalid EPIG discrete allocation shape")
    if max(cutoffs) > support_cap:
        raise EpigResourceLimit({"resource": "integer_support", "requested_cutoff": max(cutoffs),
                                 "support_cap": support_cap, "cutoffs": list(cutoffs),
                                 "tail_tolerance": EPIG_DISCRETE_TAIL_TOLERANCE})
    sizes = [k + 2 for k in cutoffs]  # Integers 0..K plus one aggregated tail.
    largest_pair = max(sizes[:candidate_count]) * max(sizes[candidate_count:])
    # All entries are float64-equivalent bytes, calculated with Python ints.
    # quantiles, tail_tables, and residuals coexist with the growing PMF banks.
    # Native frozen Poisson/NB distributions retain full-input parameters; two
    # P*N arrays cover those parameters plus one additional array allowance.
    # Four more P*N arrays cover quantile/SF/mean calculation and concatenation
    # workspace. Their peak may precede bank construction; summing is deliberate.
    full_input_bytes = 8 * particle_count * len(cutoffs)
    estimated_array_bytes = {
        "stored_pmf_banks": 8 * particle_count * sum(sizes),
        "retained_quantiles": full_input_bytes,
        "retained_tail_tables": full_input_bytes,
        "retained_residuals": full_input_bytes,
        "full_input_distribution_parameters": 2 * full_input_bytes,
        "full_input_calculation_workspace": 4 * full_input_bytes,
        "particle_pmf_workspace": 8 * 8 * particle_count * max(sizes),
        "pair_mi_workspace": 8 * 4 * largest_pair,
        "marginal_arrays": 8 * 8 * sum(sizes),
        "particle_vectors_and_input_summaries": 8 * 8 * (particle_count + len(cutoffs)),
    }
    estimated = sum(estimated_array_bytes.values())
    if estimated > max_working_bytes:
        raise EpigResourceLimit({"resource": "acquisition_arrays", "estimated_working_bytes": estimated,
                                 "max_working_bytes": max_working_bytes,
                                 "particle_count": particle_count, "cutoffs": list(cutoffs),
                                 "candidate_count": candidate_count,
                                 "estimated_array_bytes": estimated_array_bytes,
                                 "memory_estimate_rule": "retained_input_arrays_and_workspace_v2",
                                 "tail_tolerance": EPIG_DISCRETE_TAIL_TOLERANCE})
    return estimated


def _epig_discrete_banks(fits, inputs, candidate_count):
    """Integers 0..K and SF(K), without dropping mass or renormalizing particles.

    K is shared across every selected particle/model at one input. A survival
    probability is used directly, including for the zero-inflated component;
    1-CDF subtraction would erase tiny but declared tail probabilities.
    """
    from .positive_inference import COUNT_SCALAR_FAMILIES

    # K=0 gives a lower bound before constructing any full-input distributions
    # or quantiles. The final known-K preflight below checks the full request.
    particle_count = sum(len(next(iter(result.particles.values()))) for result in fits)
    _epig_discrete_preflight(particle_count, [0] * len(inputs), candidate_count)
    distributions, nonzero_weights, quantiles = [], [], []
    for result in fits:
        if result.hypothesis.family not in COUNT_SCALAR_FAMILIES:
            raise ValueError("EPIG requires a common supported unbounded count measure")
        distribution = _epig_scalar_distribution(result, inputs)
        count = len(next(iter(result.particles.values())))
        nonzero = np.ones((count, 1, 1))
        if result.hypothesis.family == "zi_poisson":
            pi = np.asarray(result.particles["pi"], dtype=float)[:, None, None]
            if np.any(~np.isfinite(pi)) or np.any((pi <= 0) | (pi >= 1)):
                raise ValueError("EPIG zero-inflation pi must lie in the open unit interval")
            nonzero = 1 - pi
        # For an atom at zero larger than 1-tolerance, K=0 already satisfies
        # the tail criterion. Otherwise isf is the stable inverse-CDF form.
        survival = EPIG_DISCRETE_TAIL_TOLERANCE / nonzero
        quantile = distribution.isf(np.where(survival < 1, survival, .5))
        quantile = np.where(survival < 1, quantile, 0.)
        if (np.any(~np.isfinite(quantile)) or np.any(quantile < 0)
                or np.any(quantile != np.floor(quantile))):
            raise ValueError("non-finite or invalid EPIG discrete predictive cutoff")
        distributions.append(distribution)
        nonzero_weights.append(nonzero)
        quantiles.append(quantile[..., 0])
    cutoffs = np.max(np.concatenate(quantiles, axis=0), axis=0)
    # Validate integer support before conversion/allocation; never lower K.
    if np.any(cutoffs > EPIG_DISCRETE_SUPPORT_CAP):
        raise EpigResourceLimit({"resource": "integer_support", "requested_cutoff": float(cutoffs.max()),
                                 "support_cap": EPIG_DISCRETE_SUPPORT_CAP,
                                 "cutoffs": cutoffs.tolist(),
                                 "tail_tolerance": EPIG_DISCRETE_TAIL_TOLERANCE})
    cutoffs = [int(k) for k in cutoffs]
    tail_tables = [nonzero * distribution.sf(np.asarray(cutoffs)[None, :, None])
                   for distribution, nonzero in zip(distributions, nonzero_weights, strict=True)]
    residuals = np.concatenate(tail_tables, axis=0)[..., 0]
    if (np.any(~np.isfinite(residuals)) or np.any(residuals < 0)
            or np.any(residuals > EPIG_DISCRETE_TAIL_TOLERANCE)):
        raise ValueError("EPIG predictive cutoff failed the per-particle tail tolerance; no fallback")
    particles = residuals.shape[0]
    estimated = _epig_discrete_preflight(particles, cutoffs, candidate_count)
    banks = []
    # Allocate only after both the cutoff and memory preflight have passed.
    for input_index, cutoff in enumerate(cutoffs):
        bank = np.empty((particles, cutoff + 2), dtype=float)
        start = 0
        for result, nonzero, tails in zip(fits, nonzero_weights, tail_tables, strict=True):
            # A one-input frozen distribution avoids evaluating a large padded
            # support for all the other inputs in this ragged representation.
            distribution = _epig_scalar_distribution(result, inputs[input_index:input_index + 1])
            component = nonzero[:, 0, :] * distribution.pmf(np.arange(cutoff + 1))[..., 0, :]
            if result.hypothesis.family == "zi_poisson":
                component[:, 0] += result.particles["pi"]
            end = start + len(component)
            bank[start:end, :-1] = component
            bank[start:end, -1] = tails[:, input_index, 0]
            start = end
        if (np.any(~np.isfinite(bank)) or np.any(bank < 0)
                or not np.allclose(bank.sum(-1), 1., atol=1e-10, rtol=0)):
            raise ValueError("EPIG discrete PMF normalization failed; no renormalization")
        banks.append(bank)
    maxima = residuals.max(axis=0)
    info = {
        "epig_discrete_outcome_rule": "enumerate_integers_then_aggregated_tail",
        "epig_discrete_coarsened": False,
        "epig_discrete_exact_original_mi": False,
        "epig_discrete_mi_interpretation": "exact_MI_of_recorded_coarsened_observations",
        "epig_discrete_tail_tolerance": EPIG_DISCRETE_TAIL_TOLERANCE,
        "epig_discrete_tail_tolerance_is_mi_error_bound": False,
        "epig_discrete_support_cap": EPIG_DISCRETE_SUPPORT_CAP,
        "epig_discrete_max_working_bytes": EPIG_DISCRETE_MAX_WORKING_BYTES,
        "epig_discrete_estimated_working_bytes": estimated,
        "epig_discrete_memory_estimate_rule": "retained_input_arrays_and_workspace_v2",
        "epig_discrete_global_cutoff": max(cutoffs),
        "epig_discrete_candidate_cutoffs": cutoffs[:candidate_count],
        "epig_discrete_target_cutoffs": cutoffs[candidate_count:],
        "epig_discrete_candidate_max_particle_tail_mass": maxima[:candidate_count].tolist(),
        "epig_discrete_target_max_particle_tail_mass": maxima[candidate_count:].tolist(),
        "epig_discrete_max_particle_tail_mass": float(maxima.max()),
        "epig_discrete_mass_dropped": False,
        "epig_discrete_probabilities_renormalized": False,
    }
    return banks, info


def _epig_discrete_coarsened_bank(fits, inputs, weights, candidate_count,
                                   enumeration_refusal, grid_size=EPIG_Y_GRID):
    """Record CDF bins on the same selected particles after enumeration refusal.

    The chunked-bank allowance is an acquisition-array estimate, not an RSS
    guarantee. No particles, candidates, targets, or quantile bins are removed
    to satisfy it. The original enumeration detail is retained even if this
    second allocation plan must also refuse.
    """
    from .positive_inference import COUNT_SCALAR_FAMILIES

    if type(grid_size) is not int or grid_size < 2:
        raise ValueError("EPIG grid_size must be an integer of at least two")
    if any(result.hypothesis.family not in COUNT_SCALAR_FAMILIES for result in fits):
        raise ValueError("EPIG requires a common supported unbounded count measure")
    particles, count = len(weights), len(inputs)
    if not 0 < candidate_count < count:
        raise ValueError("invalid EPIG coarsened allocation shape")
    bank_bytes = 8 * particles * count * grid_size
    boundary_bytes = 8 * count * (grid_size - 1)
    vector_bytes = 8 * 8 * (particles + count)
    # Quantile/CDF temporaries only span one input chunk. This does not change
    # any posterior draw, coordinate, probability level, or final CDF boundary.
    per_chunk_input_bytes = 8 * (6 * particles * grid_size + 3 * particles + 8 * grid_size)
    available = EPIG_DISCRETE_MAX_WORKING_BYTES - bank_bytes - boundary_bytes - vector_bytes
    chunk_size = min(count, 16, max(1, available // per_chunk_input_bytes))
    construction_bytes = {
        "stored_interval_bank": bank_bytes, "stored_boundaries": boundary_bytes,
        "quantile_cdf_and_mass_workspace": chunk_size * per_chunk_input_bytes,
        "particle_vectors_and_input_summaries": vector_bytes,
    }
    information_bytes = {
        "stored_interval_bank": bank_bytes, "stored_boundaries": boundary_bytes,
        "candidate_entropy_workspace": 8 * particles * candidate_count * grid_size,
        "particle_entropy_summaries": 8 * 4 * particles * candidate_count,
        "mi_joint_workspace": 8 * 4 * candidate_count * grid_size**2,
        "marginal_workspace": 8 * 8 * count * grid_size,
        "particle_vectors_and_input_summaries": vector_bytes,
    }
    estimated_array_bytes = {"construction": construction_bytes, "information": information_bytes}
    estimated = max(sum(phase.values()) for phase in estimated_array_bytes.values())
    if estimated > EPIG_DISCRETE_MAX_WORKING_BYTES:
        raise EpigResourceLimit({
            "resource": "coarsened_acquisition_arrays",
            "estimated_working_bytes": estimated,
            "max_working_bytes": EPIG_DISCRETE_MAX_WORKING_BYTES,
            "estimated_array_bytes": estimated_array_bytes,
            "memory_estimate_rule": "chunked_quantile_arrays_and_mi_workspace_v1",
            "particle_count": particles, "input_count": count,
            "candidate_count": candidate_count, "bin_count": grid_size,
            "enumeration_refusal": enumeration_refusal,
        })
    bank = np.empty((particles, count, grid_size), dtype=float)
    boundaries = np.empty((count, grid_size - 1), dtype=float)
    boundary_repairs = None
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        chunk_bank, chunk_boundaries = _epig_continuous_bank(
            fits, inputs[start:stop], weights, grid_size, discrete=True)
        bank[:, start:stop] = chunk_bank
        boundaries[start:stop] = chunk_boundaries
        chunk_repairs = getattr(chunk_bank, "boundary_repairs", None)
        if chunk_repairs:
            if boundary_repairs is None:
                boundary_repairs = dict(chunk_repairs)
            else:
                boundary_repairs["rows_repaired"] += chunk_repairs["rows_repaired"]
                boundary_repairs["max_shift"] = max(
                    boundary_repairs["max_shift"], chunk_repairs["max_shift"])
        del chunk_bank, chunk_boundaries
    cutoffs = None if enumeration_refusal is None else enumeration_refusal.get("cutoffs")
    # A memory refusal with all-zero K occurred before quantiles were computed.
    # Fixed-coarsened rounds do not replan enumeration: None is not a new refusal
    # and avoids presenting D0 cutoffs as estimates for later posterior banks.
    cutoffs_known = (enumeration_refusal is not None
                     and (enumeration_refusal["resource"] == "integer_support"
                          or bool(cutoffs and any(cutoffs))))
    info = {
        "epig_discrete_coarsened": True,
        "epig_discrete_coarsening_rule": f"bma_predictive_quantile_intervals_{grid_size}_with_tails",
        "epig_discrete_bin_count": grid_size,
        "epig_discrete_bin_boundaries": boundaries.tolist(),
        "epig_discrete_bin_interval_rule": "right_closed_CDF_intervals_with_both_infinite_tails",
        "epig_discrete_bin_input_order": "candidate_inputs_then_target_inputs",
        "epig_discrete_enumeration_refusal": enumeration_refusal,
        "epig_discrete_enumeration_cutoffs_known": cutoffs_known,
        "epig_discrete_enumeration_cutoff_note": (
            "not_recomputed: coarsened_path_fixed_at_D0" if enumeration_refusal is None else
            "computed_per_input_cutoffs" if cutoffs_known else "unknown; recorded_K_is_planning_lower_bound"
        ),
        "epig_discrete_mi_interpretation": "exact_MI_of_recorded_coarsened_observations",
        "epig_discrete_exact_original_mi": False,
        "epig_discrete_tail_tolerance_is_mi_error_bound": False,
        "epig_discrete_support_cap": EPIG_DISCRETE_SUPPORT_CAP,
        "epig_discrete_max_working_bytes": EPIG_DISCRETE_MAX_WORKING_BYTES,
        "epig_discrete_coarsened_estimated_working_bytes": estimated,
        "epig_discrete_coarsened_estimated_array_bytes": estimated_array_bytes,
        "epig_discrete_coarsened_memory_estimate_rule": "chunked_quantile_arrays_and_mi_workspace_v1",
        "epig_discrete_input_chunk_size": chunk_size,
        "epig_discrete_mass_dropped": False,
        "epig_discrete_probabilities_renormalized": False,
    }
    if boundary_repairs is not None:
        info["epig_discrete_boundary_repairs"] = boundary_repairs
    return bank, info


def _epig_discrete_information(banks, candidate_count, weights):
    """Finite MI on ragged banks, with one validation and bounded target batches."""
    from scipy.special import xlogy

    from .count_epig import _bank, _weights

    if type(candidate_count) is not int or not 0 < candidate_count < len(banks):
        raise ValueError("invalid EPIG discrete information shape")
    # Reuse the shared validation contracts, once per bank rather than per pair.
    checked = [_bank(np.asarray(bank)[:, None, :])[:, 0, :] for bank in banks]
    particles = len(checked[0])
    if any(len(bank) != particles for bank in checked):
        raise ValueError("q and x must use the same ordered posterior particles")
    weights = _weights(weights, particles, "particle")
    mixtures = [np.einsum("k,kqy->qy", weights, bank[:, None, :])[0] for bank in checked]
    entropies = [-np.sum(xlogy(mixture, mixture)) for mixture in mixtures]
    target_count = len(checked) - candidate_count
    target_banks = checked[candidate_count:]
    # Joint, log-joint, reductions and concatenated weighted PMFs remain under
    # the fixed acquisition-array cap. Stored banks and marginal arrays stay live.
    retained_bytes = (sum(bank.nbytes for bank in checked)
                      + 8 * 8 * sum(bank.shape[1] for bank in checked)
                      + 8 * 8 * (particles + len(checked)))
    available = EPIG_DISCRETE_MAX_WORKING_BYTES - retained_bytes
    epig, joint_eig = [], []
    tolerance = 1e-9
    for candidate_index, candidate in enumerate(checked[:candidate_count]):
        hx = entropies[candidate_index]
        eig = hx - np.einsum(
            "k,kx->x", weights,
            -np.sum(xlogy(candidate[:, None, :], candidate[:, None, :]), axis=-1))[0]
        per_target = []
        first = 0
        while first < target_count:
            # Concatenate ragged targets without padding. The split offsets
            # recover each separate target's entropy and normalization check.
            stop, outcomes = first, 0
            weight_candidate = target_banks[first].shape[1] > candidate.shape[1]
            legacy_pair = (particles == 1 or not candidate.flags.c_contiguous
                           or not target_banks[first].flags.c_contiguous)
            while stop < target_count:
                if stop > first and (legacy_pair or not target_banks[stop].flags.c_contiguous):
                    break
                if (target_banks[stop].shape[1] > candidate.shape[1]) != weight_candidate:
                    break
                next_outcomes = outcomes + target_banks[stop].shape[1]
                # A multi-target x-weighted batch needs one contiguous pair
                # copy to preserve the legacy entropy reduction layout.
                pair_arrays = 3 if weight_candidate and stop > first else 2
                workspace = 8 * (2 * particles * next_outcomes
                                 + pair_arrays * candidate.shape[1] * next_outcomes
                                 + 2 * particles * candidate.shape[1])
                if workspace > available:
                    break
                outcomes, stop = next_outcomes, stop + 1
            if stop == first:
                raise EpigResourceLimit({
                    "resource": "acquisition_arrays",
                    "estimated_working_bytes": retained_bytes + workspace,
                    "max_working_bytes": EPIG_DISCRETE_MAX_WORKING_BYTES,
                    "particle_count": particles, "candidate_count": candidate_count,
                })
            if legacy_pair:
                # Singleton and non-C banks use NumPy's original contraction
                # and three-dimensional reduction layout, including exact zero
                # MI roundoff. Bank and weight validation is still done once.
                joint = np.einsum("k,ky,kxz->xyz", weights, target_banks[first],
                                  candidate[:, None, :], optimize=True)
                if not np.allclose(joint.sum(axis=(1, 2)), 1., atol=1e-10, rtol=0):
                    raise ValueError("joint predictive normalization failed")
                h_joint = -np.sum(xlogy(joint, joint), axis=(1, 2))
                information = entropies[candidate_count + first] + hx - h_joint[0]
                if (information < -tolerance or eig < -tolerance
                        or information > eig + tolerance or not np.isfinite(information)):
                    raise ValueError("predictive information violated nonnegativity/data-processing bound")
                per_target.append(float(information))
                del joint
                first = stop
                continue
            targets = np.concatenate(target_banks[first:stop], axis=1)
            if weight_candidate:
                joint = (weights[:, None] * candidate).T @ targets
            else:
                joint = (weights[:, None] * targets).T @ candidate
            offset = 0
            for target_index in range(first, stop):
                width = target_banks[target_index].shape[1]
                # einsum's original contraction uses a different contiguous
                # dimension on either side of this width comparison. Preserve
                # that pair layout for the exact normalization/entropy sum.
                pair = (np.ascontiguousarray(joint[:, offset:offset + width]).T
                        if weight_candidate else joint[offset:offset + width])
                if not np.allclose(pair.sum(), 1., atol=1e-10, rtol=0):
                    raise ValueError("joint predictive normalization failed")
                information = entropies[candidate_count + target_index] + hx + np.sum(xlogy(pair, pair))
                if (information < -tolerance or eig < -tolerance
                        or information > eig + tolerance or not np.isfinite(information)):
                    raise ValueError("predictive information violated nonnegativity/data-processing bound")
                per_target.append(float(information))
                offset += width
                del pair
            # Do not retain the previous batch while allocating its successor.
            del targets, joint
            first = stop
        # Keep the established target-order Python sum, including its roundoff.
        epig.append(sum(per_target) / target_count)
        joint_eig.append(float(eig))
    return {"epig": np.asarray(epig), "joint_eig": np.asarray(joint_eig)}


def choose_epig(fits, designs, seed, posterior_samples=128, interface=None, n_trials=50,
                *, target_points, y_grid=EPIG_Y_GRID, discrete_path_state=None):
    """Mean_x* I(Y_x;Y_x*), in nats, on one common posterior particle bank.

    Bounded count outcomes are exhaustively enumerated. Unbounded counts use
    integers 0..K plus an explicitly recorded survival-tail category, giving
    coarsened-observation MI rather than exact original-law MI. Resource-limited
    integer enumeration instead uses 64 recorded BMA quantile intervals on the
    same selected particles, with both tails and the full zero-inflated law.
    Continuous outcomes retain their predictive quantile-interval calculation.
    Targets are explicit run-start inputs; no evaluation points are accepted
    implicitly and no target or posterior RNG is consumed by legacy EIG.
    """
    from .count_epig import predictive_information

    if discrete_path_state is not None and not isinstance(discrete_path_state, EpigDiscretePathState):
        raise TypeError("discrete_path_state must be an EpigDiscretePathState")
    designs, targets = np.asarray(designs, dtype=float), np.asarray(target_points, dtype=float)
    if (designs.ndim not in (1, 2) or targets.ndim != designs.ndim
            or designs.shape[1:] != targets.shape[1:] or not len(designs) or not len(targets)
            or not np.all(np.isfinite(designs)) or not np.all(np.isfinite(targets))):
        raise ValueError("candidate and target inputs must be finite compatible nonempty arrays")
    subsets, weights = _epig_subsample(fits, seed, posterior_samples)
    inputs = np.concatenate([designs, targets])
    outcome_info = None
    exclusion_info = {}
    if all(isinstance(result.hypothesis, Hypothesis) for result in subsets):
        blocks = [active_particle_rows(particle_logpmf(result.hypothesis, result.particles,
                                                      inputs, n_trials, interface))
                  for result in subsets]
        bank = np.concatenate([np.exp(block) for block in blocks])
        if len(bank) != len(weights):
            # Zero-weight particle rule: zero-weight particles left the bank, so each model's own
            # evidence mass is redistributed over the particles it still has -
            # the same repair `_epig_guard_particles` makes when a scalar mean
            # is non-finite. An untouched bank keeps the original weights array.
            weights = np.concatenate([
                np.full(len(block), weight / len(block))
                for block, weight in zip(blocks, np.exp(model_logweights(subsets)), strict=True)])
    else:
        from .positive_inference import validate_pool
        validate_pool(subsets)
        subsets, weights, exclusion_info = _epig_guard_particles(subsets, inputs, weights)
        if subsets[0].hypothesis.measure == "PMF":
            fixed_path = None if discrete_path_state is None else discrete_path_state.path
            enumeration_refusal = None
            if fixed_path != "coarsened":
                try:
                    banks, outcome_info = _epig_discrete_banks(subsets, inputs, len(designs))
                except EpigResourceLimit as error:
                    if (fixed_path == "enumeration"
                            or error.detail.get("resource") not in {"acquisition_arrays", "integer_support"}):
                        raise
                    enumeration_refusal = dict(error.detail)
            # Leave the exception scope before coarsening: its traceback can
            # otherwise retain the rejected enumeration's planning arrays.
            path = ("coarsened" if fixed_path == "coarsened" or enumeration_refusal is not None
                    else "enumeration")
            if discrete_path_state is not None:
                discrete_path_state._fix(path)
            if path == "enumeration":
                score = _epig_discrete_information(banks, len(designs), weights)
            else:
                bank, outcome_info = _epig_discrete_coarsened_bank(
                    subsets, inputs, weights, len(designs), enumeration_refusal, EPIG_Y_GRID)
                score = predictive_information(bank[:, len(designs):], bank[:, :len(designs)], weights,
                                               np.full(len(targets), 1 / len(targets)))
            if discrete_path_state is not None:
                outcome_info.update(epig_discrete_path_fixed_at_start=True,
                                    epig_discrete_path=discrete_path_state.path)
        else:
            bank = _epig_continuous_bank(subsets, inputs, weights, y_grid)
    if outcome_info is None:
        score = predictive_information(bank[:, len(designs):], bank[:, :len(designs)], weights,
                                       np.full(len(targets), 1 / len(targets)))
    rows = [{"design": [float(v) for v in np.atleast_1d(x)] if np.ndim(x) else float(x),
             "epig": float(epig), "joint_eig": float(eig)}
            for x, epig, eig in zip(designs, score["epig"], score["joint_eig"], strict=True)]
    if outcome_info is not None or exclusion_info:
        rows = EpigRows(rows, outcome_info, exclusion_info or None)
    boundary_repairs = (getattr(bank, "boundary_repairs", None) if outcome_info is None
                        else outcome_info.get("epig_discrete_boundary_repairs"))
    if boundary_repairs:
        if not isinstance(rows, EpigRows):
            rows = EpigRows(rows, None)
        rows.boundary_repairs = boundary_repairs
    return choose_design(rows, "epig")["chosen_design"], rows


def epig_acquisition_record(chosen, rows, seed):
    """Same candidate-row/choice structure as EIG, with an explicit EPIG key."""
    decision = choose_design(rows, "epig")
    record = {"seed": int(seed), "chosen": chosen, "rows": rows, "epig_rows": rows,
            "target": "predictive_target_points", "policy": {
                "design_policy": "epig", "objective_column": "epig",
                "chosen_design": chosen,
                "chosen_index": next(i for i, row in enumerate(rows) if row["design"] == chosen),
                "chosen_value": decision["top_value"], "n_tied_at_max": decision["n_tied_at_max"],
                "tie_rule": decision["tie_rule"], "tie_atol": decision["tie_atol"],
                "units": "nats", "criteria": rows}}

    if getattr(rows, "outcome_info", None) is not None:
        record["outcome_info"] = rows.outcome_info
    if getattr(rows, "exclusion_info", None):
        record.update(rows.exclusion_info)
    if getattr(rows, "boundary_repairs", None):
        record["epig_discrete_boundary_repairs"] = rows.boundary_repairs
    return record
