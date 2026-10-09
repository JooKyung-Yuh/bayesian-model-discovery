"""Run one new-paper CORE world with explicit mock/Anthropic proposals and real inference.

Not a registered BoxingGym run. Evaluation observations are generated separately
and never passed to fitting, acquisition, or the proposal callable.
"""

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import os
import resource
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp, xlogy
from scipy.stats import betabinom, binom

from ..provenance.canonical import canonical_json, sha256_text
from .belief_packet import (
    PoolMember,
    build_belief_packet,
    cell_constraints,
    core_interface,
    core_real_interface,
    interface_grid,
    packet_digest,
    serialize_packet,
)
from .count_inference import (
    EPIG_TARGET_POINTS,
    EPIG_TARGETS_SEED_STREAM,
    CountInferenceFailure,
    EpigDiscretePathState,
    FitTimeout,
    SMCSettings,
    candidate_set_record,
    choose_epig,
    epig_acquisition_record,
    epig_config,
    fit,
    loglik_prior_probe,
    model_logweights,
    predict,
    predictive_summary,
    record_acquisition_timing,
    record_fit_timing,
    record_prediction_pool_summary,
    resolve_epig_policy,
    retain,
    run_with_fit_guards,
    run_with_fit_timeout,
    validate_fit_timeout,
)
from .count_models import (
    DEFAULT_GRAMMAR_VERSION,
    DEFAULT_PRIOR_TEMPLATE,
    DEFAULT_VALIDATION_MESSAGES,
    FAILURE_CLASSES,
    FAMILIES,
    GRAMMAR_VERSIONS,
    INITIAL_BURST_REFILL_POLICY,
    LLM_PRIORS_VARIANT_TOKEN,
    PRIOR_TEMPLATE_NAMES,
    PRIOR_VARIANT_TOKEN,
    VALIDATION_MESSAGE_VERSIONS,
    Hypothesis,
    active_particle_rows,
    coefficient_prior,
    failure_class,
    initial_hypotheses,
    judge_candidate,
    nan_substitution_counter,
    particle_logpmf,
    prior_template_table,
    read_initial_pool_file,
    set_grammar_version,
    set_llm_priors,
    set_prior_template,
    set_validation_messages,
)
from .count_proposer import (
    PAID_CALL_BUDGET_EXIT_CODE,
    RESPONSE_FORMATS,
    PaidCallBudgetExhausted,
    budget_exhausted_summary,
    burst_refill_identity,
    compat_config,
    compat_option_kwargs,
    compat_points_metadata,
    composed_system_prompt,
    normalize_provider_order,
    parse_max_paid_calls,
    parse_max_tokens,
    parse_reasoning_spec,
    require_registered_decoding,
    select_run_prompt,
)
from .design_policies import (
    DEFAULT_POLICY,
    POLICIES,
    SPACE_FILLING_DEFINITION,
    TARGET,
    check_policy,
    variant_token,
)
from .design_policies import select as select_design
from .model_identity import MODEL_ID_SCHEMA, NOTATION_NORMAL_FORM
from .prior_probe_detail import (
    PriorProbeRejection,
    first_nonfinite_subexpression,
    nonfinite_probe_sentence,
    prior_probe_reason,
)
from .run_label_policy import (
    PROFILES,
    RUN_LABEL_CHOICES,
    bmd_label_receipt,
    recheck_manifest,
    resolve_label,
)

WORLD_COEFFICIENTS = {
    11: (-0.8, 1.4),
    17: (-0.2, 1.0),
    23: (0.5, 0.8),
    # Development world, derived as the mean of the three registered settings;
    # never used in a claim. The three research worlds above are untouched -
    # this dict is only ever read by `WORLD_COEFFICIENTS[self.seed]` and by
    # `--seed`'s `choices=tuple(WORLD_COEFFICIENTS)`, so a new key can add a
    # seed but can never change an existing one. D0, the evaluation set and the
    # noise rule stay exactly as the existing code derives them from the seed.
    99: (-0.17, 1.07),
    # Three more development worlds: new keys only; registered keys above are immutable.
    31: (-0.50, 1.20),
    37: (0.15, 0.90),
    41: (-0.15, 1.10),
}
PROPOSAL_ROUNDS = (5, 10, 15, 20)
P1_PROPOSAL_ROUNDS = tuple(range(1, 21))
D0 = (0.1, 0.35, 0.65, 0.9)
D0_RULE = "fixed4"
CEP_ADDED_KEYS = ("candidate_exclusion_policy",)
PWC2_ADDED_KEYS = ("active_count_by_round", "proposed_by_round", "admitted_by_round",
                   "retired_by_round", "initial_pool_llm", "initial_pool_fit", "rounds_llm", "rounds_fit")
PWC2_SCORE_ADDED_KEYS = ("active_count",)
PWC2_WAVE_COUNT_KEYS = ("accepted_total", "shortfall_after_refill", "rejected_prior_probe",
                       "rejected_loglik_probe", "fit_failed", "failed_calls", "duplicates")
PWC2_COST_KEYS = ("initial_pool_llm", "initial_pool_fit", "rounds_llm", "rounds_fit")
EVALUATION_SEEDS = {"dev": 20260906, "paper": 20260907}
ACTIVE_MAXIMUM = 12
DEFAULT_BURST_CONCURRENCY = 6
BURST_RETRIES = 4
BURST_BACKOFF_SECONDS = 1.0
# Three catches OUTSIDE the refit
# path - the prior probe, the one-step score and the candidate predict - each
# turn an exception into a DOMAIN VERDICT: "rejected under its own prior", "this
# model scores null this round", "this candidate could not be scored". A verdict
# is a statement about a model, so only a NUMERICAL failure may produce one, and
# an exception CLASS is not evidence of that - `ValueError` is also what a
# broadcast mismatch, a bad index and a malformed argument raise. What identifies
# a numerical verdict here is that OUR OWN code stated it: these are the exact
# messages `count_models.particle_logpmf` and `count_inference` raise when a
# number comes out wrong. Anything else - OSError, MemoryError, AssertionError,
# TypeError, KeyError, an unrecognized ValueError - propagates and fails the run.
PARTICLE_LOGPMF_NUMERIC = (
    "non-finite mean expression",
    "invalid likelihood",
    "outside open unit interval",
    "likelihood normalization failed",
)
PREDICT_NUMERIC = PARTICLE_LOGPMF_NUMERIC + (
    "non-finite evidence",
    "mixture weights must be finite and normalized",
)


def numeric_verdict(exc, markers):
    """Did our own numerical code state this, or is it something else?"""
    return isinstance(exc, ValueError) and any(marker in str(exc) for marker in markers)


class PoolEmpty(RuntimeError):
    """Every active model failed NUMERICALLY in one round (the empty-pool terminal state).

    Pre-declared EMPTY-POOL RULE: the run TERMINATES at that
    round. There is no re-seed, no imputation, no carry-forward of the last
    successful round's predictions - the later checkpoints simply do not exist,
    and the run is recorded as the failure path with `status: "pool_empty"`.

    What can cause it is narrow: ONLY `CountInferenceFailure`, the typed
    exception `count_inference.fit` raises for an expected numerical failure. A
    full disk, an out-of-memory, a serialization error or a bug in our own code
    empties nothing - it propagates and the run fails as `failed`, because none
    of those is a statement about a model.
    """


FAMILY_CODES = {"binomial": "F1", "beta_binomial": "F2", "zi_binomial": "F3", "normal": "N1"}
GAUSSIAN_STRUCTURES = ("gauss_den", "gauss_pow", "gauss_call")
# The plotting contract's columns plus the v1.1 `variant` column that separates an
# archive-re-evaluation run from the default one on the same axes.
POINTS_COLUMNS = (
    "suite",
    "task",
    "method",
    "proposer",
    "seed",
    "x",
    "x_unit",
    "y_primary",
    "y_name",
    "nll",
    "coverage90",
    "width90",
    "crps",
    "is_0.1",
    "kl",
    "status",
    "label",
    "run_id",
    "source_path",
    "code_sha",
    "variant",
)


# The plotting contract:
# **25 columns in one fixed order** - the contract's own columns 1..21 (ending
# `variant`, exactly the contract's section 2 numbering), then
# `code_sha` (22), `brier` (23), `nmsle` (24), `rmsle`
# (25). A column with no definition for a suite is written as an EMPTY STRING,
# never 0.
#
# `POINTS_COLUMNS` above is deliberately LEFT at its v1.2 21-column shape:
# `boxinggym_core.py` still imports and uses it, and that file is owned by
# another task, so migrating that one runner to this tuple is that
# task's own one-line change. `core_count.write_points` and
# `chembench_core.write_points` emit the v1.3 order below.
CONTRACT_POINTS_COLUMNS = (
    "suite",
    "task",
    "method",
    "proposer",
    "seed",
    "x",
    "x_unit",
    "y_primary",
    "y_name",
    "y_label",
    "nll",
    "coverage90",
    "width90",
    "crps",
    "is_0.1",
    "kl",
    "status",
    "label",
    "run_id",
    "source_path",
    "variant",
    "code_sha",
    "brier",
    "nmsle",
    "rmsle",
)
# The human-readable name of CORE's own primary metric (contract column
# `y_label`; the machine key stays `y_name` = "nll", per the contract's
# section 3 CORE row, "held-out NLL(nats/obs), KL").
CORE_Y_LABEL = "held-out NLL (nats/obs)"


def seed_for(*parts):
    """Return a deterministic 32-bit seed derived from the JSON-encoded `parts`."""
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:4], "little")


def initial_designs(seed, policy="fixed4"):
    """One seed-only initial design set, shared by every regime and arm."""
    if policy == "fixed4":
        return D0
    if policy != "stratified4":
        raise ValueError("CORE d0_policy must be fixed4 or stratified4")
    rng = np.random.default_rng(seed_for("core_d0", seed))
    return tuple(float(value) for value in 0.25 * np.arange(4) + 0.25 * rng.random(4))


def acquisition_seed_for(world_seed, replicate, round_number):
    """Common random-arm design stream across cells of one world/replicate."""
    return seed_for("acquisition", world_seed, replicate, round_number)


def wave_event_counts(events, transports, requested):
    """Indexes of existing admission events and per-call terminal records."""
    records = [item["transport"] for item in transports]
    accepted = sum(event["status"] == "fitted" for event in events)
    return {
        "accepted_total": accepted,
        "shortfall_after_refill": requested - accepted,
        "rejected_prior_probe": sum(event["status"] == "rejected_prior_probe" for event in events)
        + sum(rejection.get("error_type") == "PriorProbeRejection"
              for record in records for rejection in record.get("rejected_candidates", [])),
        "rejected_loglik_probe": sum(event["status"] == "rejected_loglik_prior_probe" for event in events),
        "fit_failed": sum(event["status"] in ("fit_failed", "fit_timeout", "fit_or_predict_failed")
                          for event in events),
        "failed_calls": sum(record["status"] != "parsed" for record in records),
        "duplicates": sum(event["status"] == "duplicate_not_refitted" for event in events),
    }


def failure_class_counts(events, transports):
    """Every failure class with the number of records that name it, zeros included.

    One per record carrying a non-null `failure_class`: each `rejected_candidates` entry of the
    transport records (`transports` is `wave_event_counts`' `[{"transport": record}, ...]`) and
    each per-candidate outcome event. Their projections - archive rows and candidate_failures
    copies - are never counted again, and a refused attempt counts whatever the next attempt did.
    """
    counts = dict.fromkeys(FAILURE_CLASSES, 0)
    entries = [entry for item in transports for entry in item["transport"].get("rejected_candidates", [])]
    for record in (*entries, *events):
        if record.get("failure_class") is not None:
            counts[record["failure_class"]] += 1
    return counts


#: The round-0 burst runs at most these waves (wave 0, then refill waves 1 and 2,
#: each asking for exactly the shortfall). A later round's parallel waves keep (0, 1).
INITIAL_BURST_WAVES = (0, 1, 2)


def burst_refill_pool(pool, maximum):
    """The pool a round-0 refill wave's packet shows - the kept part of
    `retain(pool, maximum)`, the same function the round's final retention applies, over
    the round's starting pool plus the candidates fitted so far - or none yet."""
    return retain(pool, maximum)[0] if pool else []


def pool_digest(model_ids):
    """sha256 of the sorted model_id list as compact
    JSON, UTF-8 - the identity of the pool one burst wave's packet showed."""
    return hashlib.sha256(json.dumps(sorted(model_ids), separators=(",", ":")).encode("utf-8")).hexdigest()


def burst_wave_record(wave, calls, accepted, events, transports, packet_text):
    """One round-0 burst wave's `waves[]` entry.

    `events` and `transports` are this wave's own, counted by `wave_event_counts`'
    definitions, so a burst's entries add up to its record-level counts. Wave 0 is the
    independent burst over the round's starting packet (`BURST_PARAGRAPH`); waves 1 and 2
    are refill waves whose packet shows the intermediate retained pool
    (`BURST_REFILL_PARAGRAPH`). The packet fields are read off the exact text the wave
    sent: its digest (`packet_digest`'s sha256, the bytes packets/t00_burst_w<wave> holds)
    and the size and `pool_digest` of the pool members it showed.
    """
    counts = wave_event_counts(events, transports, calls)
    shown = [member["model_id"] for member in json.loads(packet_text)["pool"]]
    return {"wave": wave, "calls": calls, "accepted": accepted,
            "duplicates": counts["duplicates"], "rejected_prior_probe": counts["rejected_prior_probe"],
            "fit_failed": counts["fit_failed"], "pool_visible": wave > 0,
            "paragraph": "burst_refill" if wave else "burst",
            "packet_digest": packet_digest(packet_text)["packet_sha256"],
            "pool_size_shown": len(shown), "pool_digest_shown": pool_digest(shown)}


REJECTION_KINDS = (
    "transport_failure", "engine_rejection", "repair_rejection",
    "duplicate_archive_projection",
)


def project_transport_rejections(archive, record, prior_rows, round_number):
    """Link this call's hook refusals; keep non-candidate rows in the transport.

    The provider's write-once response is untouched. Only the runner's copy gains
    these named projection fields. A hook raises once per refused attempt, so its
    call-local row queue corresponds to PriorProbeRejection entries in order, not
    to a guess based on expression or round. No admission or repair decision lives
    here.
    """
    pending = iter(prior_rows)
    rejections = record.get("rejected_candidates", [])
    for index, rejected in enumerate(rejections):
        attempt = rejected.get("attempt", record.get("repairs", 0))
        prior = (next(pending, None)
                 if rejected.get("error_type") == "PriorProbeRejection" else None)
        expression = rejected.get("expression")
        readable = isinstance(expression, str) and bool(expression.strip())
        if prior is not None:
            key, row = prior
        else:
            key = f"rejected_t{round_number:02d}_{index}"
            row = {
                "model_id": rejected.get("model_id", ""),
                "expression": expression if readable else "",
                "family": rejected.get("family", ""),
                "proposed_round": round_number,
                "outcome": "rejected" if readable else "transport_failure",
                "reason": {"validation_message": rejected["reason"]},
                # An array rejection's own declaration, copied; a row with no
                # candidate (a whole-reply failure) declares none.
                "parameters": rejected.get("parameters") if readable else None,
            }
            if readable:
                archive[key] = row
            else:
                # A failed/unreadable call is not a hypothesis in the belief packet.
                # Preserve its empty-expression row beside the original rejection.
                record.setdefault("transport_failure_rows", []).append(row)
        kind = ("repair_rejection" if attempt else "engine_rejection") if (
            prior is not None or readable) else "transport_failure"
        row.update(archive_key=key, attempt=attempt, rejection_kind=kind)
        rejected.update(archive_key=key, rejection_kind=kind, rejection_attempt=attempt)
    if rejections:
        record["rejection_taxonomy"] = {"schema_version": "rrt/1", "kinds": list(REJECTION_KINDS)}


def add_recorded_pool_count(row):
    """Add `active_count` to a recorded pool row (in place) from its recorded active ids or components."""
    if "active_ids" in row:
        row["active_count"] = len(row["active_ids"])
    else:
        for component in row["components"]:
            component["active_count"] = len(component["active_ids"])
        row["active_count"] = sum(component["active_count"] for component in row["components"])


def proposal_accounting_indexes(out, score_rows):
    """Derive final-per-round pool sizes and proposal counts without new decisions."""
    active = {str(row["round"]): row["active_count"] for row in score_rows}
    indexes = {key: dict.fromkeys(active, 0) for key in PWC2_ADDED_KEYS if key not in PWC2_COST_KEYS}
    indexes["active_count_by_round"] = active
    for path in sorted((out / "proposals").glob("t*.json")):
        record = json.loads(path.read_text())
        round_key = str(int(path.stem.split("_", 1)[0][1:]))
        for target, source in (("proposed_by_round", "events"), ("admitted_by_round", "admitted_ids"),
                               ("retired_by_round", "retired_ids")):
            indexes[target][round_key] = indexes[target].get(round_key, 0) + len(record.get(source, []))
    return indexes


def wave_cost_blocks(out):
    """Index recorded transports and fits on disjoint round-zero/later axes."""
    llm = {key: {"calls": 0, "repairs": 0, "failed_calls": 0, "input_tokens": 0,
                 "output_tokens": 0, "usage_complete": True, "wall_seconds": 0.0}
           for key in ("initial_pool_llm", "rounds_llm")}
    for path in sorted((out / "llm_transport").glob("*.response.json")):
        record = json.loads(path.read_text())
        if "round" in record:
            round_number, calls, repairs = record["round"], record["calls"], record["repairs"]
            usage = record["usage_total"]
            complete = (record["usage_total_complete"] and len(record["responses"]) == calls)
        else:
            # Legacy P4 stores one response per event rather than a P1 call bundle.
            round_number, calls, repairs = record["event"], int(record["api_request_attempted"]), 0
            usage = record.get("usage") or {}
            complete = record.get("usage_known", False)
        block = llm["initial_pool_llm" if round_number == 0 else "rounds_llm"]
        block["calls"] += calls
        block["repairs"] += repairs
        block["failed_calls"] += record["status"] != "parsed"
        block["input_tokens"] += usage.get("input_tokens", usage.get("prompt_tokens", 0))
        block["output_tokens"] += usage.get("output_tokens", usage.get("completion_tokens", 0))
        block["usage_complete"] &= complete
        block["wall_seconds"] += record["elapsed_seconds"]
    wave_ids = set()
    for path in sorted((out / "proposals").glob("t00_burst*.json")):
        record = json.loads(path.read_text())
        wave_ids.update((record.get("component"), event["model_id"])
                        for event in record["events"] if event["status"] == "fitted")
    initial = {"count": 0, "seconds": 0.0, "catalogue_count": 0, "wave_count": 0}
    rounds = {"count": 0, "seconds": 0.0}
    for path in sorted((out / "fits").glob("*.json")):
        record = json.loads(path.read_text())
        at_zero = record["diagnostics"]["round"] == 0
        block = initial if at_zero else rounds
        block["count"] += 1
        block["seconds"] += record["fit_seconds"]
        if at_zero:
            identity = (record["diagnostics"].get("component"), record["model_id"])
            initial["wave_count" if identity in wave_ids else "catalogue_count"] += 1
    return {**llm, "initial_pool_fit": initial, "rounds_fit": rounds}


def write_json(path, value):
    """Write `value` to `path` as sorted-key JSON, converting numpy arrays and scalars."""
    def default(v):
        """JSON fallback encoder for numpy arrays and scalars."""
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)

    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=default, allow_nan=False)
        stream.write("\n")



def _atomic_proposal_json(path, value):
    """Replace a live proposal record atomically using the original JSON format."""
    def default(v):
        """JSON fallback encoder for numpy arrays and scalars."""
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, default=default, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_proposed_json(path, value):
    """Persist the current transport and proposed expressions before candidate fitting."""
    if path.exists():
        raise FileExistsError(path)
    _atomic_proposal_json(path, {**value, "status": "proposed"})


def write_final_proposal_json(path, value):
    """Finalize only this run's temporary proposal; completed results stay immutable."""
    if not path.exists():
        write_json(path, value)
        return
    if json.loads(path.read_text()).get("status") != "proposed":
        raise FileExistsError(path)
    _atomic_proposal_json(path, value)


def append_json(path, value):
    """Append `value` to `path` as one sorted-key JSON line."""
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


class CoreWorld:
    """A synthetic bounded-count world with a known structure, family and coefficients."""
    outcome_type = "bounded_count"

    def __new__(cls, seed, structure, family, replicate=0, *, design_grid=None):
        if cls is CoreWorld and structure in ("gauss_den", "gauss_pow", "gauss_call"):
            return object.__new__(GaussianCoreWorld)
        return object.__new__(cls)

    def __init__(self, seed, structure, family, replicate=0):
        self.seed, self.structure, self.family = seed, structure, family
        self.replicate = replicate
        self.repeats = {}

    def probability(self, u):
        """Return the true success probability of the count world at input `u`."""
        b0, b1 = WORLD_COEFFICIENTS[self.seed]
        z = 2 * np.asarray(u) - 1
        g = b0 + b1 * z
        if self.structure == "out":
            g = g + 0.8 * np.tanh(3 * z)
        elif self.structure == "den":
            g = b0 + b1 * z / (1.6 + z)
        elif self.structure == "pow":
            g = b0 + b1 * (np.asarray(u) + 0.25) ** 2.4
        return expit(g)

    def sample(self, u, rng, size=None):
        """Draw count outcomes at input `u` from the world's family."""
        p = self.probability(u)
        if self.family == "beta_binomial":
            p = rng.beta(p * 9, (1 - p) * 9, size=size)
            return rng.binomial(50, p)
        y = rng.binomial(50, p, size=size)
        if self.family == "zi_binomial":
            y = np.where(rng.random(np.shape(y)) < 0.2, 0, y)
        return y

    def observe(self, u):
        """Draw one observation at `u`, using a seed that depends on the repetition number."""
        key = float(u).hex()
        repetition = self.repeats.get(key, 0)
        self.repeats[key] = repetition + 1
        seed = seed_for("observation", self.seed, self.structure, self.family, key, repetition)
        if self.replicate:
            seed = seed_for("observation_replicate", seed, self.replicate)
        return {
            "u": float(u),
            "y": int(self.sample(u, np.random.default_rng(seed))),
            "design_id": key,
            "repetition": repetition,
            "observation_seed": seed,
        }

    def evaluation(self, count, input_seed=20260906):
        """Return evaluation inputs, outcomes and true probabilities for `count` held-out points."""
        u = np.random.default_rng(seed_for("evaluation_inputs", input_seed)).uniform(0, 1, count)
        rng = np.random.default_rng(
            seed_for("evaluation_outcomes", self.seed, self.structure, self.family)
        )
        y = self.sample(u[:, None], rng, size=(count, 4))
        p, ys = self.probability(u)[:, None], np.arange(51)[None, :]
        if self.family == "beta_binomial":
            pmf = betabinom.pmf(ys, 50, p * 9, (1 - p) * 9)
        else:
            pmf = binom.pmf(ys, 50, p)
            if self.family == "zi_binomial":
                pmf *= 0.8
                pmf[:, 0] += 0.2
        return u, y, pmf


class GaussianCoreWorld(CoreWorld):
    """Identity-link world; its one truth sigma never enters candidate fitting."""

    outcome_type = "real"

    def __init__(self, seed, structure, family, replicate=0, *, design_grid=None):
        if structure not in GAUSSIAN_STRUCTURES or family != "normal":
            raise ValueError("Gaussian CORE requires a gauss_* structure and truth family normal")
        super().__init__(seed, structure, family, replicate)
        self.design_grid = np.asarray(
            np.linspace(0.0, 1.0, 100) if design_grid is None else design_grid, dtype=float
        ).copy()
        if (self.design_grid.ndim != 1 or len(self.design_grid) < 2 or
                not np.array_equal(self.design_grid, np.linspace(0.0, 1.0, len(self.design_grid)))):
            raise ValueError("Gaussian sigma requires the run's linspace(0, 1, grid_size)")
        self.sigma = float(0.05 * np.std(self.mean(self.design_grid), ddof=1))
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Gaussian truth sigma must be finite and positive; no floor")
        self.grid_bytes = json.dumps(self.design_grid.tolist(), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()

    def mean(self, u):
        """Return the true Gaussian mean at input `u`."""
        from .gaussian_core_constants import GAUSS_CALL_K, GAUSS_DEN_C, GAUSS_POW_E

        b0, b1 = WORLD_COEFFICIENTS[self.seed]
        u = np.asarray(u)
        if self.structure == "gauss_den":
            return b0 + b1 * u / (GAUSS_DEN_C + u)
        if self.structure == "gauss_pow":
            return b0 + b1 * u ** GAUSS_POW_E
        return b0 + b1 * np.tanh(GAUSS_CALL_K * (2 * u - 1))

    def probability(self, u):
        """Raise ValueError: a Gaussian world has a density, not a count probability."""
        raise ValueError("Gaussian CORE has a density, not a count probability")

    def sample(self, u, rng, size=None):
        """Draw Gaussian outcomes at input `u`."""
        return rng.normal(self.mean(u), self.sigma, size=size)

    def observe(self, u):
        """Draw one Gaussian observation at `u`, using a seed that depends on the repetition number."""
        key = float(u).hex()
        repetition = self.repeats.get(key, 0)
        self.repeats[key] = repetition + 1
        seed = seed_for("observation", self.seed, self.structure, self.family, key, repetition)
        if self.replicate:
            seed = seed_for("observation_replicate", seed, self.replicate)
        return {"u": float(u), "y": float(self.sample(u, np.random.default_rng(seed))),
                "design_id": key, "repetition": repetition, "observation_seed": seed}

    def evaluation(self, count, input_seed=20260906):
        """Return evaluation inputs, outcomes and the true (mean, sigma) for `count` held-out points."""
        u = np.random.default_rng(seed_for("evaluation_inputs", input_seed)).uniform(0, 1, count)
        rng = np.random.default_rng(seed_for("evaluation_outcomes", self.seed, self.structure, self.family))
        y = self.sample(u[:, None], rng, size=(count, 4))
        return u, y, (self.mean(u), self.sigma)

    def noise_record(self):
        """Return the record describing how the noise level was set."""
        return {"rule": f"0.05 * sd_x(m) on linspace(0, 1, {len(self.design_grid)})",
                "sigma": self.sigma, "sigma_6sf": format(self.sigma, ".6g"), "ddof": 1,
                "grid_size": len(self.design_grid),
                "grid_canonical_json": self.grid_bytes.decode(),
                "grid_sha256": hashlib.sha256(self.grid_bytes).hexdigest()}


def mock_proposals(public_observations, event, cell, initial_family):
    """Fixed non-LLM responses, no truth/Q/test arguments and no API imports."""
    if len(public_observations) != 4 + event or event not in PROPOSAL_ROUNDS:
        raise ValueError("proposal event/public observation count mismatch")
    s, f, _ = (int(v) for v in cell)
    if not s and not f:
        return []
    initial = initial_hypotheses(initial_family)
    forms = ("b0+b1*z+b2*tanh(3*z)", "b0+b1*z+b2*square(u)", "b0+b1*tanh(3*z)", "b0+b1*z+b2*exp(u)")
    # The template schedule is frozen before looking at any score.
    event_index = PROPOSAL_ROUNDS.index(event)
    form = forms[event_index] if s else initial[event_index].expression
    names = ("b0", "b1", "b2") if "b2" in form else ("b0", "b1") if "b1" in form else ("b0",)
    params = tuple((n, "intercept" if n == "b0" else "signed") for n in names)
    if f:
        return [Hypothesis(form, family, params) for family in FAMILIES]
    alternatives = (form, form.replace("3*z", "2*z"), form + "+b3*u")
    return [
        Hypothesis(text, initial_family, params + ((("b3", "signed"),) if "b3" in text else ()))
        for text in alternatives
    ]


# Policy P1 mock schedule (an internal design decision): AT MOST one new candidate is
# evaluated per round, and admission is only ever by evidence rank - proposing
# does not admit. Twenty templates, declared here before any run and never tuned
# after seeing a score, exactly like the P4 schedule above. Cell-aware: S0 keeps
# an initial structure verbatim and changes only the family; F0 keeps the
# initial family and changes only the structure; the structure-and-family-frozen variant proposes nothing.
P1_MOCK_EXPRESSIONS = (
    "b0+b1*z",
    "b0+b1*z+b2*square(z)",
    "b0+b1*tanh(3*z)",
    "b0+b1*z+b2*tanh(3*z)",
    "b0+b1*z+b2*u",
    "b0+b1*log(1+u)",
    "b0+b1*z+b2*exp(u)",
    "b0+b1*sqrt(u)",
    "b0+b1*z+b2*square(u)",
    "b0+b1*z+b2*power(u,3)",
    "b0+b1*tanh(2*z)",
    "b0+b1*z+b2*tanh(2*z)",
    "b0+b1*z+b2*log(1+u)",
    "b0+b1*z+b2*sqrt(u)",
    "b0+b1*u",
    "b0+b1*z+b2*z*u",
    "b0+b1*exp(z)",
    "b0+b1*z+b2*square(z)+b3*tanh(3*z)",
    "b0+b1*power(z,3)",
    "b0+b1*z+b2*u+b3*square(u)",
)
P1_MOCK_FAMILIES = tuple(FAMILIES[(index + 1) % len(FAMILIES)] for index in range(20))

# Prior finiteness probe / prior predictive (revived model_spec.py prior probe).
# Both draw from the SAME dedicated stream, seeded per model, so neither touches
# the run's own numpy RNG, the SMC seeds or the acquisition stream - which the
# byte-identical P4 anchor demonstrates rather than merely asserts.
PRIOR_PROBE_SEED = 20260907
PRIOR_PROBE_SAMPLES = 64
PRIOR_PROBE_GRID_POINTS = 16
# The only count_models.particle_logpmf failures that mean "non-finite" and so
# reject a candidate. Every other failure (the normalization tolerance) is a
# far-prior-tail numerical warning on a well-defined model, and is recorded.
PRIOR_PROBE_NONFINITE = (
    "non-finite mean expression",
    "invalid likelihood",
    "outside open unit interval",
)


def mock_parameters(form):
    """Role assignment for a template expression: b0 intercept, the rest signed."""
    names = sorted(
        {
            node.id
            for node in ast.walk(ast.parse(form))
            if isinstance(node, ast.Name) and node.id.startswith("b")
        }
    )
    return tuple((name, "intercept" if name == "b0" else "signed") for name in names)


def mock_proposal_p1(public_observations, t, cell, initial_family, max_rounds=20):
    """One fixed non-LLM candidate for round t; no truth/Q/test arguments, no API imports."""
    if len(public_observations) != 4 + t or not 1 <= t <= max_rounds:
        raise ValueError("proposal round/public observation count mismatch")
    structure, family_free, _ = (int(bit) for bit in cell)
    if not structure and not family_free:
        return []
    index = (t - 1) % len(P1_MOCK_EXPRESSIONS)
    if structure:
        form = P1_MOCK_EXPRESSIONS[index]
        family = P1_MOCK_FAMILIES[index] if family_free else initial_family
    else:
        # S0: an initial structure verbatim, family changed. Structure varies
        # fastest so the eight reachable pairs are covered once each in eight
        # rounds - after which candidate_catalog reports the cell exhausted.
        initial = initial_hypotheses(initial_family)
        alternatives = [name for name in FAMILIES if name != initial_family]
        form = initial[index % len(initial)].expression
        family = alternatives[(index // len(initial)) % len(alternatives)]
    return [Hypothesis(form, family, mock_parameters(form))]


def candidate_catalog(cell, initial_family):
    """Every model_id the cell can still express, or None when the catalog is infinite.

    S0 fixes the expression to `initial_structures`, so the catalog is that list
    times `allowed_families` - at most twelve pairs, of which the four initial
    ones are already in the pool. Whenever structure expansion is allowed the
    expression grammar is unbounded and there is no catalog to exhaust; F0 alone
    is likewise unbounded because only the family is pinned.
    """
    if cell[0] == "1":
        return None
    families = FAMILIES if cell[1] == "1" else (initial_family,)
    return {
        Hypothesis(item.expression, family, item.parameters).model_id: (item.expression, family)
        for item in initial_hypotheses(initial_family)
        for family in families
    }


def prior_particles(hypothesis, count=PRIOR_PROBE_SAMPLES):
    """Draw coefficient and dispersion vectors from the fixed role priors.

    Dedicated stream: seeded from the model identity alone, never from the run
    seed, and drawn from its own Generator, so no probe can shift the SMC seeds,
    the acquisition stream or any later draw.
    """
    rng = np.random.default_rng(PRIOR_PROBE_SEED ^ int(hypothesis.model_id, 16))
    draws = {}
    names = [*hypothesis.parameters, *((name, name) for name in hypothesis.noise_parameters)]
    for name, role in names:
        # LLM-proposed-prior ablation: the probe draws from the prior the FIT will use, so
        # an accepted proposed prior enters the prior predictive too - which is
        # what the coverage diagnostic ("prior predictive coverage of the observed
        # range") is computed from. No proposal: `PRIOR_RULES[role]`, unchanged.
        family, a, b = coefficient_prior(hypothesis, name, role)
        if family == "Normal":
            draws[name] = rng.normal(a, b, count)
        elif family == "LogNormal":
            draws[name] = rng.lognormal(a, b, count)
        elif family == "Gamma":
            draws[name] = rng.gamma(a, 1.0 / b, count)
        elif family == "Beta":
            draws[name] = rng.beta(a, b, count)
        else:
            raise ValueError(f"prior probe cannot sample {family}")
    return draws


def prior_probe(hypothesis, interface, count=PRIOR_PROBE_SAMPLES):
    """Finiteness gate plus prior predictive summary, from one shared prior sample.

    Evaluates expression -> link -> log pmf on the interface's reference grid.
    A candidate whose mean expression or likelihood is NON-FINITE anywhere under
    its own prior is rejected before any SMC fit is attempted.

    The gate is finiteness and nothing else. `particle_logpmf` also enforces a
    1e-10 normalization tolerance, which a perfectly well-defined model can trip
    in the far prior tail (scipy's beta-binomial at kappa ~ 1e9, say). Failing a
    candidate for that would silently ban whole families, so such a run is
    recorded as `normalization_warning` and PASSES; only the messages in
    PRIOR_PROBE_NONFINITE reject. The tolerance itself rarely
    raises: drift up to `DRIFT_TOL` is renormalized and a prior draw that lost
    real mass is dropped from the summary below, so only a sample in which EVERY
    draw lost mass still arrives here as a normalization warning.
    """
    grid = interface_grid(interface, PRIOR_PROBE_GRID_POINTS)
    particles = prior_particles(hypothesis, count)
    result = {"n_samples": int(count), "passed": True, "grid_points": PRIOR_PROBE_GRID_POINTS}
    try:
        logpmf = particle_logpmf(hypothesis, particles, grid, interface["trials"])
    except ValueError as exc:
        # Only a numerical verdict OUR OWN likelihood code stated may
        # become a probe result. A second model reproduced the alternative on earlier versions:
        # a TypeError injected inside the probe came back as `passed=true` with a
        # normalization warning, so narrowing the refit catch alone left the hole
        # open one level down. An unrecognized failure re-raises unchanged.
        if not numeric_verdict(exc, PARTICLE_LOGPMF_NUMERIC):
            raise
        detail = f"{type(exc).__name__}: {exc}"
        finite = not any(marker in str(exc) for marker in PRIOR_PROBE_NONFINITE)
        if not finite:
            # Name the sub-expression, input value and range that went
            # non-finite, on the same particles and grid, in the CORE namespace.
            found = first_nonfinite_subexpression(
                hypothesis.expression, particles, {"u": grid, "z": 2 * grid - 1}, interface
            )
            if found is not None:
                return {**result, "passed": False, "normalization_warning": False,
                        "detail": nonfinite_probe_sentence(found), "exception": detail,
                        "nonfinite": found}, None
        return {**result, "passed": finite, "normalization_warning": finite, "detail": detail}, None
    # A prior draw given zero weight is not part of the prior
    # predictive average; with none of them this is the same mean as before.
    pmf = np.exp(active_particle_rows(logpmf)).mean(axis=0)
    counts = np.arange(pmf.shape[-1])
    cumulative = np.cumsum(pmf, axis=-1)
    predictive = {
        "grid": grid.tolist(),
        "mean": (pmf @ counts).tolist(),
        "q05": (cumulative < 0.05).sum(-1).tolist(),
        "q50": (cumulative < 0.50).sum(-1).tolist(),
        "q95": (cumulative < 0.95).sum(-1).tolist(),
        "n_prior_samples": int(count),
    }
    return result, predictive


def packet_observations(public):
    """Public rows as the packet states them: identity and round only, no seeds.

    The runner's own rows also carry `design_id`, `repetition` and
    `observation_seed`; none of those reach the packet. `round` is exact because
    D0 is observed at round 0 and every later round adds exactly one row.
    """
    return [
        {
            "obs_id": index,
            "round": 0 if index < len(D0) else index - len(D0) + 1,
            "u": row["u"],
            "y": row["y"],
        }
        for index, row in enumerate(public)
    ]


def one_step_log_score(result, u, y, interface=None):
    """log p(y_new) under a model's posterior BEFORE the update.

    log( sum_i w_i p(y_new | theta_i) ) over the equally weighted SMC particles
    - the log of the mean predictive probability, NOT the mean of the
    per-particle log probabilities. Public data only; nothing here sees Q.
    """
    if interface is not None and interface["outcome_type"] == "real":
        from .positive_inference import log_predictive

        return float(log_predictive([result], np.asarray([u]), np.asarray([[y]]))[0][0, 0])
    try:
        # The mean predictive probability is over the particles that
        # carry weight, so a zeroed one leaves the sum and the divisor together.
        rows = active_particle_rows(
            particle_logpmf(result.hypothesis, result.particles, np.asarray([float(u)]))
        )
    except ValueError as exc:
        # A NUMERICALLY degenerate model scores null; anything else is not
        # a score, it is a failure, and it propagates.
        if not numeric_verdict(exc, PARTICLE_LOGPMF_NUMERIC):
            raise
        return None
    return float(logsumexp(rows[:, 0, int(y)]) - np.log(rows.shape[0]))


def parse_round_list(text, default, maximum=20):
    """Comma or space separated round list bounded by the run horizon."""
    if text is None or not str(text).strip():
        return tuple(default)
    values = tuple(sorted({int(item) for item in str(text).replace(",", " ").split()}))
    if any(not 1 <= value <= maximum for value in values):
        raise ValueError(f"round list entries must be 1..{maximum}")
    return values


def archive_reevaluation_label(rounds):
    """Return the short label of an archive re-evaluation round list (`off` when empty)."""
    return "off" if not rounds else "t" + "_".join(str(value) for value in rounds)


def points_variant(
    label,
    design_policy=DEFAULT_POLICY,
    design_policy_applies=False,
    prior_template=DEFAULT_PRIOR_TEMPLATE,
    llm_priors=False,
):
    """The plotting contract's `variant` column: tokens sorted alphabetically, joined by `+`.

    Empty means the runner default. `rev<rounds>` is the archive
    re-evaluation switch; the `design_*` token is added only when the
    design policy differs from the cell default, which is `joint_eig` on an A=1
    cell and the fixed random draw on an A=0 cell (where no policy applies and
    no token is emitted). `method` stays the cell key - the plot contract
    forbids putting a policy name in the fixed method list.

    The prior-scale and LLM-proposed-prior ablations add `prior_wide`/`prior_narrow` for a
    replaced role-prior template and `prior_llm` for the proposed-prior path. The
    default template and the off switch emit nothing, so a default run's column
    is unchanged.
    """
    tokens = [] if label == "off" else ["rev" + label[1:]]
    token = variant_token(design_policy, design_policy_applies)
    if token:
        tokens.append(token)
    if PRIOR_VARIANT_TOKEN[prior_template]:
        tokens.append(PRIOR_VARIANT_TOKEN[prior_template])
    if llm_priors:
        tokens.append(LLM_PRIORS_VARIANT_TOKEN)
    return "+".join(sorted(tokens))


# The plot contract's `label` column (and the `config`/`summary` echoes of it)
# is shared by all three runners - core_count, boxinggym_core and
# chembench_core - and used to be three separately hand-written copies of
# "development if profile == dev or proposer == openai else production". That
# inline rule mislabels a PAPER-profile MOCK run as production: mock is
# neither "dev" nor "openai", so the "else" branch fired. A paper x
# ChemBench smoke (2026-09-08, cells 001/111, --proposer mock, --profile
# paper) hit exactly that: its config.json/points.csv both recorded
# `production` for a run that made zero paid API calls.
#
# Current rule: production is explicit and verified against the registered cohort.
# Historic points without a bound receipt default to development.
def points_label(profile, proposer):
    """Default label only; explicit production needs resolve_run_label context.

    `paper_neuron` joins the two original profiles and is registered in
    `run_label_policy.py`; like them it derives `development` and nothing else.
    """
    if profile not in PROFILES:
        raise ValueError(f"points_label: unknown profile {profile!r}")
    if proposer not in ("mock", "anthropic", "openai", "openai_compat"):
        raise ValueError(f"points_label: unknown proposer {proposer!r}")
    return "development"


def resolve_run_label(requested, profile, proposer, **context):
    """Compatibility entry point returning the shared policy's resolved label."""
    points_label(profile, proposer)
    if requested not in (None, *RUN_LABEL_CHOICES):
        raise ValueError(f"resolve_run_label: unknown run label {requested!r}")
    return resolve_label(requested, profile, proposer, **context)["run_label"]


def write_points(path, config, rows, status, variant):
    """The plotting contract's point table for this run, one row per score row."""
    proposer = config["proposer"]
    task = (
        f"core__structure_{config['structure']}"
        f"__truth_{FAMILY_CODES[config['truth_family']]}__s{config['seed']}"
    )
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=(*CONTRACT_POINTS_COLUMNS, *compat_points_metadata(config)), lineterminator="\n"
        )
        writer.writeheader()
        for line, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    **compat_points_metadata(config),
                    "suite": "core",
                    "task": task,
                    "method": config["cell"],
                    "proposer": "mock" if "mock" in proposer else f"llm:{config['llm_model']}",
                    "seed": config["seed"],
                    "x": row["observation_count"],
                    "x_unit": "observations",
                    "y_primary": row["bma"]["nll"],
                    "y_name": "nll",
                    # The plotting contract.
                    "y_label": CORE_Y_LABEL,
                    "nll": row["bma"]["nll"],
                    "coverage90": row["bma"]["coverage90"],
                    "width90": row["bma"]["width90"],
                    "crps": "",
                    "is_0.1": "",
                    "kl": row["bma"]["kl"],
                    "status": status,
                    # Shared helper, see
                    # `points_label`'s own docstring for the rule and why.
                    # A label the RUN already resolved (--run-label,
                    # recorded in config) wins; the helper is the fallback for a
                    # config written before --run-label existed (and for callers that build a
                    # config dict by hand, such as this module's own tests).
                    "label": config.get("label")
                    or points_label(
                        config["profile"], "mock" if "mock" in proposer else proposer
                    ),
                    "run_id": Path(config["output"]).name,
                    "source_path": (
                        f"{config['output']}/scores.jsonl#L{line}"
                        f"(round={row['round']},phase={row['phase']})"
                    ),
                    "code_sha": config["source_files_sha256"]["core_count.py"],
                    "variant": variant,
                    # The plotting contract's columns 23-25. None of the three has a
                    # definition on the CORE path - the outcome is a bounded
                    # count, not a binary event (`brier`) and not a positive
                    # continuous rate scored in log1p space (`nmsle`/`rmsle`) -
                    # so all three are BLANK, never 0 (an internal design decision).
                    "brier": "",
                    "nmsle": "",
                    "rmsle": "",
                }
            )


def validate_cell_candidates(candidates, cell, initial_family, *, allowed_families=None):
    """Validate expansion permissions independently of the proposal transport.

    All-or-nothing batch check (unchanged from earlier versions): raises on the first
    invalid candidate. Used by the mock-proposal path, where a violation is a
    programming error, not an LLM mistake to tolerate - contrast
    count_proposer.parse_candidates, which judges the LLM's own candidates
    per-item via the same judge_candidate this delegates to.
    """
    if cell not in {f"{s}{f}{a}" for s in "01" for f in "01" for a in "012"} or len(candidates) > 3:
        raise ValueError("invalid cell or more than three proposal pairs")
    structure, family = (int(bit) for bit in cell[:2])
    if not structure and not family and candidates:
        raise ValueError("the fixed-pool cell (no structure and no family expansion) forbids proposals")
    initial_structures = initial_hypotheses(initial_family)
    for hypothesis in candidates:
        from .real_inference import RealHypothesis

        if not isinstance(hypothesis, Hypothesis) and not (
                allowed_families == ("normal",) and isinstance(hypothesis, RealHypothesis)):
            raise TypeError("proposal must be a validated Hypothesis")
        item = {
            "expression": hypothesis.expression,
            "family": hypothesis.family,
            "parameters": dict(hypothesis.parameters),
        }
        ok, reason = judge_candidate(item, cell, initial_structures, initial_family,
                                      **({"allowed_families": allowed_families} if allowed_families else {}))
        if not ok:
            raise ValueError(reason)


def score_prediction(logpmf, heldout, truth, rng):
    """Score predictive log-probabilities against held-out counts and the true distribution; returns a summary dict."""
    summary = predictive_summary(logpmf, rng)
    indices = np.arange(len(heldout))[:, None]
    nll = -float(logpmf[indices, heldout].mean())
    expected_log_score = float(np.sum(truth * logpmf, axis=-1).mean())
    kl = float(np.sum(xlogy(truth, truth) - truth * logpmf, axis=-1).mean())
    lower, upper = summary["lower90"][:, None], summary["upper90"][:, None]
    return {
        "nll": nll,
        "expected_log_score": expected_log_score,
        "kl": kl,
        "mean_rmse": float(np.sqrt(np.mean((summary["mean"] - truth @ np.arange(51)) ** 2))),
        "coverage90": float(((heldout >= lower) & (heldout <= upper)).mean()),
        "width90": float((upper - lower).mean()),
    }, summary


def duplicate_archive_key(archive, t, key):
    """Collision-safe archive key for a same-round duplicate.

    Unchanged from earlier versions when this is the first duplicate of `key` archived
    this round: `duplicate_t{t:02d}_{key}`, so a run with no same-round
    repeat of one model_id keeps byte-identical archive keys. A second (or
    third, ...) duplicate of the SAME model_id in the SAME round would
    otherwise collide with and silently overwrite the first; instead it gets
    a `_k{n}` suffix (n = 2, 3, ...), so every duplicate event keeps its own
    archive entry and the entry count equals the event count.
    """
    base = f"duplicate_t{t:02d}_{key}"
    if base not in archive:
        return base
    n = 2
    while f"{base}_k{n}" in archive:
        n += 1
    return f"{base}_k{n}"


def validate_world_options(args):
    """Validate the world-related command-line options; return whether the outcome is Gaussian."""
    real = args.structure in GAUSSIAN_STRUCTURES
    if real != (args.truth_family == "normal"):
        raise ValueError("gauss_* structures require --truth-family normal; count structures forbid normal")
    if args.initial_family is None:
        args.initial_family = "normal" if real else "binomial"
    if real != (args.initial_family == "normal"):
        raise ValueError("Gaussian CORE initial family must be normal; count structures forbid normal")
    if real and (args.cell[1] != "0" or args.priors != "off"):
        raise ValueError("Gaussian CORE requires family_expansion=false and fixed role priors")
    if real and getattr(args, "design_policy", None) == "structure_eig":
        raise ValueError("Gaussian CORE has no registered scalar structure_eig adapter")
    if real and getattr(args, "run_label", None) == "production":
        raise ValueError("Gaussian CORE is development-only")
    return real


def validate_wave_options(args):
    """Validate the proposal-wave command-line options and raise ValueError when out of range."""
    for name, maximum in (("initial_proposals", 24), ("parallel_proposals", 6),
                          ("active_maximum", 32), ("burst_concurrency", 24)):
        if hasattr(args, name) and not 1 <= getattr(args, name) <= maximum:
            raise ValueError(f"--{name.replace('_', '-')} must be 1..{maximum}")
    initial = getattr(args, "initial_proposals", 0)
    parallel = getattr(args, "parallel_proposals", 0)
    if initial > getattr(args, "active_maximum", ACTIVE_MAXIMUM):
        raise ValueError("--initial-proposals must not exceed --active-maximum")
    if parallel and args.proposal_limit != 1:
        raise ValueError("--parallel-proposals and --proposal-limit are alternative arms; state one")
    catalogue = getattr(args, "initial_catalogue", "on")
    if catalogue not in ("on", "off"):
        raise ValueError("--initial-catalogue must be on or off")
    pool_file = getattr(args, "initial_pool_file", None)
    if pool_file is not None and not pool_file:
        raise ValueError("--initial-pool-file must be a nonempty path")
    if pool_file and (catalogue != "off" or initial):
        raise ValueError("--initial-pool-file requires --initial-catalogue off and no --initial-proposals")
    if not pool_file and catalogue == "off" and (not initial or args.cell[:2] == "00"):
        raise ValueError("--initial-catalogue off needs --initial-proposals on a cell that proposes")
    if (initial or parallel) and args.proposal_policy != "P1":
        raise ValueError("parallel proposal waves require policy P1")


def run(args):
    """Run one CORE world end to end with the given parsed arguments and write its run files."""
    validate_wave_options(args)
    real_outcome = validate_world_options(args)
    candidate_exclusion_policy = getattr(
        args, "candidate_exclusion_policy", "observed" if real_outcome else "none")
    if candidate_exclusion_policy not in ("none", "observed"):
        raise ValueError("candidate_exclusion_policy must be none or observed")
    if not real_outcome and candidate_exclusion_policy == "observed":
        raise ValueError("count world forbids candidate_exclusion_policy=observed")
    d0_policy = getattr(args, "d0_policy", "fixed4")
    d0_designs = initial_designs(args.seed, d0_policy)
    if real_outcome:
        from . import gaussian_core as gaussian
        from . import gaussian_core_constants

    args.design_policy = resolve_epig_policy(args.cell, getattr(args, "design_policy", None))
    fit_timeout_seconds = validate_fit_timeout(getattr(args, "fit_timeout_seconds", None))
    fit_guards = getattr(args, "fit_guards", "off")
    if fit_guards not in ("off", "v1"):
        raise ValueError("fit_guards must be off or v1")
    start = time.perf_counter()
    if args.proposer == "anthropic" and not args.allow_paid_api:
        raise ValueError("Anthropic proposals require explicit --allow-paid-api authorization")
    if args.proposer in ("openai", "openai_compat") and not args.allow_paid_api:
        raise ValueError("OpenAI proposals require explicit --allow-paid-api authorization")
    # --run-label. Resolved HERE,
    # before any output directory exists, so an inadmissible label leaves
    # nothing behind. `args.run_label` itself is NOT overwritten: it stays the
    # caller's raw request (None when the flag was not typed), because the
    # byte-invariance rule below keys off "was a label actually
    # stated" and because `**vars(args)` should record what was asked for.
    label_metadata = compat_config(args)
    stated_run_label = getattr(args, "run_label", None)
    # The system prompt is SELECTED here, before the receipt, so the label binds
    # the text this run actually sends instead of the module default. `--priors llm` is
    # applied further down (with the prior template, as the ablation plan requires), so its decision is
    # stated to `select_run_prompt` rather than read off a switch that is not set yet; the
    # `**prompt_identity()` in `config` below recomputes the same pair from the same source.
    # The two keys ride a receipt-only namespace, exactly as the neuron runner's `active_maximum` does, so
    # `compat_config(args)` keeps the key set every earlier run recorded.
    _, label_prompt = select_run_prompt(args, llm_priors=args.priors == "llm")
    initial_pool_file = None
    if getattr(args, "initial_pool_file", None):
        pool_interface = core_real_interface() if real_outcome else core_interface()
        set_prior_template(args.prior_template)
        set_llm_priors(args.priors == "llm")
        set_validation_messages(getattr(args, "validation_messages", DEFAULT_VALIDATION_MESSAGES))
        set_grammar_version(getattr(args, "grammar_version", DEFAULT_GRAMMAR_VERSION), pool_interface)
        pool_constraints = cell_constraints(args.cell, args.initial_family,
            **({"allowed_families": ("normal",)} if real_outcome else {}))
        initial_pool_file = read_initial_pool_file(args.initial_pool_file, cell=args.cell,
            initial_family=args.initial_family, allowed_families=pool_constraints["allowed_families"])
        label_prompt = {**label_prompt, **initial_pool_file.metadata()}
    initial_proposals = int(getattr(args, "initial_proposals", 0))
    active_maximum = int(getattr(args, "active_maximum", ACTIVE_MAXIMUM))
    burst_concurrency = int(getattr(args, "burst_concurrency", DEFAULT_BURST_CONCURRENCY))
    initial_catalogue = getattr(args, "initial_catalogue", "on")
    parallel_proposals = int(getattr(args, "parallel_proposals", 0))
    if initial_proposals or parallel_proposals:
        label_prompt = {**label_prompt, "burst_prompt_sha256":
                        composed_system_prompt(burst=True)[1]["burst_prompt_sha256"]}
    # A round-0 burst's refill waves send their own composed request; the receipt
    # binds its digest and the refill paragraph's (a project decision), and config and the
    # burst pool file record these with the prompt file's own. The receipt also
    # binds the refill rule beside them.
    refill_identity = burst_refill_identity() if initial_proposals else {}
    if initial_proposals:
        label_prompt = {**label_prompt, **{key: refill_identity[key] for key in (
            "burst_refill_prompt_sha256", "burst_refill_paragraph_sha256")},
            "initial_burst_refill_policy": INITIAL_BURST_REFILL_POLICY}
    label_args = argparse.Namespace(**{**vars(args), **label_prompt})
    label_receipt = bmd_label_receipt("core", label_args, task=f"structure_{args.structure}__truth_{FAMILY_CODES[args.truth_family]}", metadata=label_metadata)
    from .run_label_policy import validate_repair_attempts
    validate_repair_attempts(args.repair_attempts, bundle="core", receipt=label_receipt)
    run_label = label_receipt["run_label"]
    # A production run states a decoding option only if the manifest
    # registered the same value - here, before any output exists.
    require_registered_decoding(label_metadata, label_receipt, "core")
    label_explicit = (stated_run_label is not None or getattr(args, "manifest_ref", None) is not None
                      or initial_pool_file is not None)
    from .count_proposer import (
        mask_secrets,
        mask_secrets_deep,
        masked_exception,
        prompt_identity,
    )

    # Prior-scale and LLM-proposed-prior ablations. Selected ONCE, here, before any
    # hypothesis, packet, probe or fit exists, because every reader of the role
    # priors and of the proposed-prior path resolves them at use time:
    # `count_models.PRIOR_RULES`, `count_inference.build_model`,
    # `prior_particles`, `belief_packet.cell_constraints`, `Hypothesis.as_dict`
    # and `count_proposer`'s prompt choice. The defaults are the paper's prior table and
    # the v2.0 prompt, so a run that passes neither flag is unchanged.
    set_prior_template(args.prior_template)
    set_llm_priors(args.priors == "llm")
    # The validation-message wording, selected once with the other engine switches;
    # unstated is v1, the sentence every run has produced.
    set_validation_messages(getattr(args, "validation_messages", DEFAULT_VALIDATION_MESSAGES))

    policy = args.proposal_policy
    if policy == "P4":
        proposal_rounds, proposal_limit, repair_attempts = PROPOSAL_ROUNDS, 3, 0
    else:
        # The proposal horizon follows `--rounds` past the 20 the P1 list was
        # written for, and never below it - a run of 20 rounds or fewer records and
        # accepts exactly the list it always did (the registered waves and every short
        # development run keep their bytes), and a longer run extends it to its own
        # last round.
        horizon = max(args.rounds, P1_PROPOSAL_ROUNDS[-1])
        proposal_rounds = parse_round_list(
            args.proposal_rounds, tuple(range(1, horizon + 1)), maximum=horizon
        )
        proposal_limit, repair_attempts = int(args.proposal_limit), int(args.repair_attempts)
    reevaluation_rounds = parse_round_list(
        args.archive_reevaluation_rounds, (), maximum=max(args.rounds, P1_PROPOSAL_ROUNDS[-1])
    )
    reevaluation_label = archive_reevaluation_label(reevaluation_rounds)
    design_policy = args.design_policy
    design_policy_applies = check_policy(args.cell, design_policy)
    variant = points_variant(
        reevaluation_label,
        design_policy,
        design_policy_applies,
        args.prior_template,
        args.priors == "llm",
    )
    interface = core_real_interface() if real_outcome else core_interface()
    # The expression grammar is selected ONCE, here, with this
    # task's own interface, so the v3 domain proof reads the declared input
    # ranges this run actually uses. `getattr` for the same reason every other
    # late-added switch uses it - an args namespace built before this change
    # existed still means the default, which is the gate it was judged by.
    grammar_version = getattr(args, "grammar_version", DEFAULT_GRAMMAR_VERSION)
    set_grammar_version(grammar_version, interface)
    constraints = cell_constraints(args.cell, args.initial_family,
                                   **({"allowed_families": ("normal",)} if real_outcome else {}))
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    children = ["fits", "predictions", "acquisition", "proposals", "truncation"]
    if policy == "P1":
        children.append("packets")
    if reevaluation_rounds:
        children.append("resurrection")
    for child in children:
        (out / child).mkdir()
    particles, q, grid_size, eig_samples = (
        (64, 32, 16, 32) if args.profile == "dev" else (500, 256, 100, 128)
    )
    settings = SMCSettings(particles, 2, 2)
    proposer = None
    # The proposer - and with it the SDK client
    # constructor - is built INSIDE a recording try. Before this, a
    # constructor error that echoed the credential (an auth/proxy/header
    # message) propagated verbatim out of run() with its str, repr and
    # traceback intact, and no failure.json was written at all because the
    # construction sat outside run's recording try (a code review found this).
    # Now the same masked failure record is written and the exception is
    # re-raised masked. Only a constructor failure is caught here; nothing
    # about the numerical path changes.
    try:
        if args.proposer == "anthropic":
            from .count_proposer import AnthropicCountProposer

            proposer = AnthropicCountProposer(
                out / "llm_transport", allow_live=True, model=args.llm_model,
                **compat_option_kwargs(args, compat=False)
            )
        elif args.proposer == "openai":
            from .count_proposer import OpenAICountProposer

            proposer = OpenAICountProposer(
                out / "llm_transport", allow_live=True, model=args.llm_model,
                **compat_option_kwargs(args, compat=False)
            )
        elif args.proposer == "openai_compat":
            from .count_proposer import OpenAICompatCountProposer

            proposer = OpenAICompatCountProposer(
                out / "llm_transport", allow_live=True, model=args.llm_model,
                base_url=getattr(args, "llm_base_url", None),
                key_name=getattr(args, "llm_key_name", "OPENAI_COMPAT_API_KEY"),
                **compat_option_kwargs(args, compat=True)
            )
    except BaseException as exc:  # noqa: BLE001 - recorded masked, then re-raised masked
        write_json(
            out / "failure.json",
            {
                "status": "failed",
                "stage": "proposer_construction",
                "error": mask_secrets(f"{type(exc).__name__}: {exc}"),
                "traceback": mask_secrets(traceback.format_exc()),
                "rounds_completed": 0,
                "active_refit_failures": [],
                "refit_failed_models": [],
                "predict_failures": [],
                "nan_substitutions": {
                    "cells": 0,
                    "calls": 0,
                    "attempts_with_substitutions": 0,
                },
                "timing_seconds": {},
            },
        )
        raise masked_exception(exc) from None
    config = {
        **compat_config(args),
        "model_id_schema": MODEL_ID_SCHEMA,
        "notation_normal_form": NOTATION_NORMAL_FORM,
        "output": str(out),
        "smc": asdict(settings),
        "q": q,
        "evaluation_input_seed": EVALUATION_SEEDS[args.profile],
        "evaluation_outcome_seed": seed_for(
            "evaluation_outcomes", args.seed, args.structure, args.truth_family
        ),
        "design_count": grid_size,
        "d0_policy": D0_RULE if d0_policy == "fixed4" else d0_policy,
        "d0_designs": d0_designs,
        "eig_posterior_samples": eig_samples,
        "initial_pool": len(initial_pool_file.hypotheses) if initial_pool_file else 4,
        **({**initial_pool_file.metadata(), "initial_catalogue": initial_catalogue}
           if initial_pool_file else {}),
        "active_maximum": active_maximum,
        **({"initial_proposals": initial_proposals, "burst_concurrency": burst_concurrency,
            "initial_catalogue": initial_catalogue, "parallel_proposals": parallel_proposals,
            "burst_prompt_sha256": label_prompt["burst_prompt_sha256"]}
           if initial_proposals or parallel_proposals else {}),
        # Present only when a round-0 burst runs.
        **({**refill_identity, "initial_burst_refill_policy": INITIAL_BURST_REFILL_POLICY}
           if initial_proposals else {}),
        "proposal_policy": policy,
        "proposal_rounds": proposal_rounds,
        "proposal_limit": proposal_limit,
        "repair_attempts": repair_attempts,
        "proposal_cadence": (
            "at_most_one_new_candidate_per_round_admission_only_by_evidence_rank"
            if policy == "P1"
            else "up_to_three_candidates_at_four_events"
        ),
        "archive_reevaluation": reevaluation_label,
        "archive_reevaluation_rounds": reevaluation_rounds,
        "design_policy": design_policy,
        "design_policy_applies": design_policy_applies,
        "design_policy_target": TARGET[design_policy] if design_policy_applies else "random",
        "points_variant": variant,
        "belief_packet": "v2" if policy == "P1" else "none",
        "interface": interface,
        **prompt_identity(),
        "prior_probe": {
            "samples": PRIOR_PROBE_SAMPLES,
            "grid_points": PRIOR_PROBE_GRID_POINTS,
            "seed_rule": "dedicated_generator_seeded_by_model_id_only",
        },
        "model_prior": "uniform_distinct_active_pairs",
        "evidence": "full_public_dataset_under_original_fixed_prior_each_fit",
        "proposer": "mock_fixed_template_responses_NOT_LLM" if proposer is None else args.proposer,
        "paid_api_calls": 0 if proposer is None else None,
        "seed_fields": {
            "world": args.seed,
            "obs": "observations.jsonl:observation_seed; design/repeat keyed, cell independent",
            "proposal_rep": seed_for("proposal_rep", args.seed, args.replicate),
            "smc": "fits/*.json:diagnostics.seed; one recorded seed per actual fit",
        },
        "proposal_rep_seed_consumed": False,
        "random_order_contract": "acquisition seed = (world seed, replicate, additional round); no cell",
        "source_files_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "core_count.py",
                "count_inference.py",
                "count_models.py",
                "model_identity.py",
                "count_diagnostics.py",
                "count_proposer.py",
                "belief_packet.py",
                "design_policies.py",
            )
        },
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.executable,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pymc", "pytensor")
        },
        "evaluation_use": "scoring_only_never_fit_choose_or_propose",
    }
    if d0_policy == "stratified4":
        config["d0_coordinates_sha256"] = sha256_text(canonical_json(d0_designs))
    # The prior-scale ablation requires the template NAME and the ACTUAL hyperparameter
    # table in config; `config = {**compat_config(args), ...}` already carries the two flag
    # names. The TABLE and `prior_source` are added only when a switch is on: on
    # the default path they would restate a frozen code constant while enlarging
    # every existing run's config.json, and the experiment anchor comparison
    # accepts a config diff of new CLI keys + self-hash + timing only.
    if args.prior_template != DEFAULT_PRIOR_TEMPLATE or args.priors == "llm":
        config["prior_hyperparameters"] = prior_template_table()
        config["prior_source"] = "llm" if args.priors == "llm" else "role_template"
    # The run-label rule binds explicit labels to the cohort; omitted legacy key sets stay unchanged.
    if label_explicit or args.proposer in ("openai", "openai_compat"):
        config["label"] = run_label
    if label_explicit:
        config["label_receipt"] = label_receipt
    if fit_timeout_seconds is None:
        config.pop("fit_timeout_seconds", None)
    else:
        config["fit_timeout_seconds"] = fit_timeout_seconds
    if fit_guards == "off":
        config.pop("fit_guards", None)
    else:
        config["fit_guards"] = fit_guards
    epig_targets = None
    epig_path_state = EpigDiscretePathState() if design_policy == "epig" else None
    if design_policy == "space_filling":
        config["design_policy_definition"] = SPACE_FILLING_DEFINITION
        config["design_normalization_bounds"] = [item["range"] for item in interface["inputs"]]
    if design_policy == "epig":
        target_seed = seed_for(EPIG_TARGETS_SEED_STREAM, args.seed)
        epig_targets = np.random.default_rng(target_seed).uniform(0.0, 1.0, EPIG_TARGET_POINTS)
    world = CoreWorld(args.seed, args.structure, args.truth_family, args.replicate)
    if real_outcome:
        config.update(outcome_type="real", noise=world.noise_record(),
                      constant_selection_sha256=hashlib.sha256(Path(gaussian_core_constants.__file__).read_bytes()).hexdigest(),
                      allowed_families=["normal"], family_expansion=False,
                      design_proxy_name=gaussian_core_constants.DESIGN_PROXY_NAME,
                      gauss_den_c=gaussian_core_constants.GAUSS_DEN_C,
                      gauss_pow_e=gaussian_core_constants.GAUSS_POW_E,
                      gauss_call_k=gaussian_core_constants.GAUSS_CALL_K)
        from . import boxinggym_truth_law, real_inference
        for module in (gaussian, gaussian_core_constants, gaussian.scalar,
                       real_inference, boxinggym_truth_law):
            path = Path(module.__file__)
            config["source_files_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if design_policy != "epig":
        write_json(out / "config.json", config)
    public = [world.observe(u) for u in d0_designs]
    response_center, response_unit = gaussian.calibrate(public) if real_outcome else (0.0, 1.0)
    if real_outcome:
        write_json(out / "frozen_d0_policy.json", {
            "response_center": response_center, "response_unit": response_unit,
            "policy": "frozen_D0_mean_and_sample_sd_of_the_four_initial_observations",
            "prior_recalibration_after_d0": False,
        })
    for row in public:
        append_json(out / "observations.jsonl", {"round": 0, **row})
    evaluation_u, heldout, truth = world.evaluation(q, EVALUATION_SEEDS[args.profile])
    if design_policy == "epig":
        config.update(epig_config(
            epig_targets, target_seed, "uniform_unit_interval_same_as_evaluation",
            evaluation_points=evaluation_u))
        write_json(out / "config.json", config)
    if real_outcome:
        np.savez_compressed(out / "evaluation_scoring_only.npz", u=evaluation_u, y=heldout,
                            mean=truth[0], sigma=truth[1])
    else:
        np.savez_compressed(out / "evaluation_scoring_only.npz", u=evaluation_u, y=heldout, pmf=truth)
    timings = {"fit": 0.0, "proposal": 0.0, "eig": 0.0, "forward": 0.0, "scoring": 0.0}
    fits, admissions, failures = [], 0, []
    transport_failure_rows = []
    timed_out_ids = set()
    # Run-level bookkeeping for the belief packet and the archive. `archive` is
    # keyed by model_id (or a synthetic key for a candidate that never became a
    # valid Hypothesis) and holds one structured entry per model ever seen.
    archive, hypotheses, score_rows, candidate_meta = {}, {}, [], {}
    one_step, born_round, active_rounds, evidence_stamp = {}, {}, {}, {}
    resurrection_events, catalog_exhausted_rounds = [], []
    # Refit-failure bookkeeping. `active_refit_failures` holds one
    # entry per ROUND in which an active model's refit raised; `refit_failed_models`
    # holds one flat, structured record per model lost that way, ACTIVE or
    # CANDIDATE, so a report can answer "did a retirement change the winning
    # family/structure?" without reopening the archive. Both are `[]` on every run
    # the engine could complete before this rule existed. `nan_substitutions` is the run-level count of
    # NaN -> -inf substitutions made by count_models.particle_logpmf, so the rule
    # is never silent: many substitutions is a bug signal, none is the norm.
    active_refit_failures, refit_failed_models, predict_failures = [], [], []
    nan_substitutions = {"cells": 0, "calls": 0, "attempts_with_substitutions": 0}
    accounting = {
        "llm_calls": 0,
        "llm_repairs": 0,
        "proposal_rounds_executed": 0,
        "catalog_exhausted": 0,
        "valid_candidates": 0,
        "duplicates": 0,
        "rejected_candidates": 0,
        "prior_probe_rejections": 0,
        "fit_attempts": 0,
        "fit_successes": 0,
        "reevaluation_refits": 0,
        # `failure_class_counts` over every transport and outcome record of the run.
        "failure_class_counts": dict.fromkeys(FAILURE_CLASSES, 0),
    }
    if initial_proposals or parallel_proposals:
        accounting.update(burst_calls=0, burst_repairs=0, burst_failed_calls=0, burst_candidates=0,
                          burst_duplicates=0, burst_refill_calls=0, burst_admitted=0,
                          burst_max_in_flight=0)
        accounting.update({key: 0 for key in PWC2_WAVE_COUNT_KEYS if key != "duplicates"})
    if initial_proposals:
        # The round-0 burst waves executed (1..3).
        accounting["burst_wave_count"] = 0

    def register(hypothesis, t):
        """Record a model in the run-level history the first time it is seen."""
        key = hypothesis.model_id
        hypotheses.setdefault(key, hypothesis)
        if key not in one_step:
            one_step[key] = [None] * max(0, t)
            born_round[key] = t
            active_rounds[key] = []
        return key

    def mark_active(hypothesis, t):
        """Move a model to outcome `active`; a model already active keeps its entry.

        `reason.active_rounds` aliases the live list, so a model that simply
        stays in the pool needs no rewrite - its rounds keep accumulating.
        """
        key = register(hypothesis, t)
        entry = archive.get(key)
        if entry is not None and entry["outcome"] == "active":
            return
        archive[key] = {
            **candidate_meta.get(key, {}),
            "model_id": key,
            "expression": hypothesis.expression,
            "family": hypothesis.family,
            # The role mapping the engine judged, recorded; no reader decides on it.
            "parameters": dict(hypothesis.parameters),
            "proposed_round": born_round[key],
            "outcome": "active",
            "reason": {"admitted_round": t, "active_rounds": active_rounds[key]},
        }

    def stamp_active(t):
        """Append round t to every active model's history exactly once."""
        for result in fits:
            history = active_rounds[register(result.hypothesis, t)]
            if not history or history[-1] != t:
                history.append(t)

    def nan_since(before):
        """NaN -> -inf substitutions made since the snapshot `before`.

        Pure: it reads the counter and subtracts, and folds nothing into the run
        totals. `account_nan` does that, exactly once per refit ATTEMPT, so an
        attempt that fails after a partly-counted refit is never counted twice.
        """
        now = nan_substitution_counter()
        return {key: now[key] - before[key] for key in before}

    def account_nan(before):
        """Fold one refit attempt's substitutions into the run totals, once."""
        delta = nan_since(before)
        if delta["cells"]:
            nan_substitutions["cells"] += delta["cells"]
            nan_substitutions["calls"] += delta["calls"]
            nan_substitutions["attempts_with_substitutions"] += 1
        return delta

    def probe_candidate(hypothesis, interface):
        """Draw the prior-predictive probe of a candidate hypothesis."""
        if real_outcome:
            return gaussian.prior_probe(hypothesis, interface, PRIOR_PROBE_SAMPLES,
                                         response_unit, response_center, PRIOR_PROBE_GRID_POINTS)
        return prior_probe(hypothesis, interface)

    def refit(hypothesis, t):
        """Refit one hypothesis with SMC at round `t` and return its fit result."""
        seed = seed_for("smc", args.seed, args.cell, t, hypothesis.model_id)
        if args.replicate:
            seed = seed_for("smc_replicate", seed, args.replicate)
        nan_before = nan_substitution_counter()
        fit_started = time.monotonic() if fit_timeout_seconds is not None else None
        fit_status = "completed"
        try:
            fit_impl = fit
            if real_outcome:
                from functools import partial
                fit_impl = partial(gaussian.fit_core, response_unit=response_unit,
                                   response_center=response_center)
            fit_callable = fit_impl
            if fit_timeout_seconds is not None:
                from functools import partial
                fit_callable = partial(run_with_fit_timeout, fit_impl,
                                       fit_timeout_seconds=fit_timeout_seconds)
            if fit_guards == "v1":
                from functools import partial
                fit_callable = partial(run_with_fit_guards, fit_callable, fit_guards=fit_guards)
            result = fit_callable(
                hypothesis,
                [d["u"] for d in public],
                [d["y"] for d in public],
                seed,
                settings,
                interface,
                interface.get("trials"),
            )
        except CountInferenceFailure as exc:
            fit_status = "fit_timeout" if isinstance(exc, FitTimeout) else "fit_failed"
            if isinstance(exc, FitTimeout) or exc.cause_code == "nonfinite_smc_weights":
                timed_out_ids.add(hypothesis.model_id)
            # The seed is part of "which attempt failed"; only `refit`
            # knows it. Attach and re-raise bare - no re-wrapping, no lost frames.
            exc.fit_seed = int(seed)
            raise
        except BaseException:
            fit_status = "fit_failed"
            raise
        finally:
            if fit_started is not None:
                record_fit_timing(out, hypothesis, t, time.monotonic() - fit_started, fit_status)
        result.diagnostics["round"] = t
        result.diagnostics["public_data_sha256"] = hashlib.sha256(
            json.dumps([(d["u"], d["y"]) for d in public]).encode()
        ).hexdigest()
        timings["fit"] += result.seconds
        stem = f"t{t:02d}_{hypothesis.model_id}"
        np.savez_compressed(out / "fits" / f"{stem}.npz", **result.particles)
        # Prior predictive of the OUTCOME (through the family, not just the link
        # mean) from the dedicated per-model prior stream; a record, not a gate.
        probe, predictive = probe_candidate(hypothesis, interface)
        # Recorded in the fit's own diagnostics ONLY when the numeric rule
        # actually fired, so a run in which it never fires keeps the bytes it had before this rule.
        substitutions = nan_since(nan_before)
        if substitutions["cells"]:
            result.diagnostics["nan_substitutions"] = substitutions
        evidence_stamp[hypothesis.model_id] = {"computed_round": t, "n_obs": len(public)}
        write_json(
            out / "fits" / f"{stem}.json",
            {
                **hypothesis.as_dict(),
                "log_evidence": result.log_evidence,
                "per_chain_log_evidence": result.per_chain_log_evidence,
                "diagnostics": result.diagnostics,
                "fit_seconds": result.seconds,
                "prior_predictive": {**probe, **(predictive or {})},
            },
        )
        print(
            f"FIT t={t} model={hypothesis.model_id} family={hypothesis.family} "
            f"logZ={result.log_evidence:.6f} seconds={result.seconds:.2f}",
            flush=True,
        )
        return result

    def pool_signature():
        """Identity of the active pool INCLUDING the fit objects.

        A set, not a sequence: `retain` re-sorts the same fits by evidence every
        round, and a pure reordering leaves the mixture unchanged, so it must not
        force a rescore. `id(v)` makes "the same fit object" - hence the same
        particles and the same weight - part of the identity.
        """
        return frozenset((v.hypothesis.model_id, id(v), v.log_evidence) for v in fits)

    checkpoint_models = []

    def checkpoint(t, phase, scoring_phase=None, one_step_row=None):
        """Fit and score the active pool at round `t`; return the scores and checkpoint arrays."""
        nonlocal checkpoint_models
        tic = time.perf_counter()
        if real_outcome:
            scores, arrays, checkpoint_models = gaussian.checkpoint(
                fits, evaluation_u, heldout, truth, seed_for, args.seed, t, phase)
            row = {"round": t, "phase": phase, "observation_count": len(public),
                   "active_ids": [v.hypothesis.model_id for v in fits], **scores}
            if scoring_phase is not None:
                row.update(scoring_phase=scoring_phase, post_equals_pre=False)
                if one_step_row is not None:
                    row["one_step_log_score"] = one_step_row
            np.savez_compressed(out / "predictions" / f"t{t:02d}_{phase}.npz", u=evaluation_u, **arrays)
            record_prediction_pool_summary(out, t, "official", len(public), checkpoint_models)
            add_recorded_pool_count(row)
            append_json(out / "scores.jsonl", row)
            score_rows.append(row)
            timings["scoring"] += time.perf_counter() - tic
            print(f"SCORE t={t} phase={phase} pool={len(fits)} NLL={row['bma']['nll']:.6f}", flush=True)
            return row, pool_signature()
        prediction = predict(fits, evaluation_u, interface, interface["trials"])
        timings["forward"] += time.perf_counter() - tic
        tic = time.perf_counter()
        row = {
            "round": t,
            "phase": phase,
            "observation_count": len(public),
            "active_ids": [v.hypothesis.model_id for v in fits],
            "model_weights": np.exp(prediction["model_logweights"]).tolist(),
            "single_model_id": prediction["single_model_id"],
        }
        if scoring_phase is not None:
            row["scoring_phase"] = scoring_phase
            row["post_equals_pre"] = False
            if one_step_row is not None:
                row["one_step_log_score"] = one_step_row
        arrays = {k: v for k, v in prediction.items() if k != "single_model_id"}
        for name in ("bma", "single"):
            rng = np.random.default_rng(seed_for("predictive_samples", args.seed, t, phase, name))
            scores, summary = score_prediction(prediction[name + "_logpmf"], heldout, truth, rng)
            row[name] = scores
            arrays.update({name + "_" + key: value for key, value in summary.items()})
        row["per_model_nll"] = {
            v.hypothesis.model_id: -float(
                prediction["model_logpmfs"][i][np.arange(q)[:, None], heldout].mean()
            )
            for i, v in enumerate(fits)
        }
        np.savez_compressed(out / "predictions" / f"t{t:02d}_{phase}.npz", u=evaluation_u, **arrays)
        checkpoint_models = [
            {"model_id": result.hypothesis.model_id,
             "weight": float(np.exp(prediction["model_logweights"][index])),
             "means": np.exp(prediction["model_logpmfs"][index]) @
                      np.arange(prediction["model_logpmfs"].shape[-1])}
            for index, result in enumerate(fits)
        ]
        record_prediction_pool_summary(out, t, "official", len(public), checkpoint_models)
        add_recorded_pool_count(row)
        append_json(out / "scores.jsonl", row)
        score_rows.append(row)
        timings["scoring"] += time.perf_counter() - tic
        print(f"SCORE t={t} phase={phase} pool={len(fits)} NLL={row['bma']['nll']:.6f}", flush=True)
        return row, pool_signature()

    def repeat_checkpoint(previous, phase, scoring_phase):
        """Record the unchanged score without recomputing it.

        Only reached when the active model_id set AND every fit object are the
        same as at the previous scoring, so the prediction is identical by
        construction: no forward pass, no new predictions array. The two
        order-dependent bookkeeping fields are still refreshed from the current
        pool, because `retain` may have re-sorted the same fits; both come from
        `model_logweights`, which is a softmax over evidence and costs nothing.
        """
        row = {key: value for key, value in previous.items() if key != "one_step_log_score"}
        row.update(
            phase=phase,
            scoring_phase=scoring_phase,
            post_equals_pre=True,
            predictions_from=previous["phase"],
            active_ids=[v.hypothesis.model_id for v in fits],
            model_weights=np.exp(model_logweights(fits)).tolist(),
        )
        record_prediction_pool_summary(out, row["round"], "official", len(public), checkpoint_models)
        add_recorded_pool_count(row)
        append_json(out / "scores.jsonl", row)
        score_rows.append(row)
        print(f"SCORE t={row['round']} phase={phase} post_equals_pre=True", flush=True)
        return row

    def score_after(previous, signature, phase, scoring_phase):
        """Score the pool after a round, reusing the previous checkpoint when the pool is unchanged."""
        if policy == "P4":
            return checkpoint(previous["round"], phase)
        if pool_signature() == signature:
            return repeat_checkpoint(previous, phase, scoring_phase), signature
        return checkpoint(previous["round"], phase, scoring_phase)

    def pool_members(t, members=None):
        """Return the pool members of round `t` (the active fits unless `members` is given)."""
        # `members` (a round-0 refill wave's intermediate pool) replaces the
        # active `fits`; None - every other caller - is the active pool as always.
        return [
            PoolMember(
                v.hypothesis,
                v.particles,
                v.log_evidence,
                tuple(one_step[v.hypothesis.model_id]),
                tuple(range(1, t + 1)),
                born_round[v.hypothesis.model_id],
                born_round[v.hypothesis.model_id],
                tuple(active_rounds[v.hypothesis.model_id]),
                evidence_stamp[v.hypothesis.model_id]["computed_round"],
                evidence_stamp[v.hypothesis.model_id]["n_obs"],
                **({"response_unit": response_unit, "response_center": response_center}
                   if real_outcome else {}),
            )
            for v in (fits if members is None else members)
        ]

    def retire(retired, t, cutoff):
        """Move models retired at round `t` into the archive with their evidence cutoff."""
        for result in retired:
            key = result.hypothesis.model_id
            archive[key] = {
                **candidate_meta.get(key, {}),
                "model_id": key,
                "expression": result.hypothesis.expression,
                "family": result.hypothesis.family,
                "parameters": dict(result.hypothesis.parameters),
                "proposed_round": born_round[key],
                "outcome": "retired_by_evidence",
                "reason": {
                    "log_evidence_at_retirement": float(result.log_evidence),
                    "gap_to_cutoff_nats": float(cutoff - result.log_evidence),
                    "active_rounds": active_rounds[key],
                    "computed_round": evidence_stamp[key]["computed_round"],
                    "n_obs": evidence_stamp[key]["n_obs"],
                },
            }

    def record_refit_failure(hypothesis, exc, t, scope, substitutions, stage="fit"):
        """One structured record per model lost to a failed refit.

        The ARCHIVE keeps the original `fit_failed` reason shape (`{"error": ...}`) -
        `belief_packet.ARCHIVE_REASON_KEYS` fixes it and belief_packet.py is
        unchanged - so the structured detail lives in `summary.json` instead:
        identity (model_id/expression/family), when (round), which path (scope),
        which STAGE, what happened (exception), and how many NaN -> -inf
        substitutions the numeric rule made while attempting this model.

        `refit_failed_models` is the NUMERICAL-failure ledger and
        holds `stage="fit"` records only - `scope` says whether the model was an
        `active` one (retired from the pool), a `candidate` (never admitted) or a
        `reevaluation` revival (stays retired_by_evidence). A `stage="predict"`
        record goes to the separate `predict_failures` list and is NOT a
        statement about the likelihood, so it never counts as a NaN retirement.
        """
        typed = isinstance(exc, CountInferenceFailure)
        record = {
            "model_id": hypothesis.model_id,
            "expression": hypothesis.expression,
            "family": hypothesis.family,
            "round": t,
            "scope": scope,
            "stage": stage,
            # A numerical failure means THIS ATTEMPT failed, not
            # that the model is false and not that its evidence is -inf.
            "retirement_kind": "inference_failure",
            "cause_code": exc.cause_code if typed else None,
            "exception": f"{type(exc).__name__}: {exc}",
            # The run continues after this, so the traceback object is
            # gone by the time the summary is written. Keep the original
            # exception's identity and its FULL chained traceback text here, not
            # only as a live `__cause__`.
            "original_exception": (exc.original if typed else None)
            or {
                "module": type(exc).__module__,
                "class": type(exc).__qualname__,
                "message": str(exc),
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__, chain=True)
                ),
            },
            "fit_seed": getattr(exc, "fit_seed", None),
            "nan_substitutions": substitutions,
        }
        if typed and exc.cause_code == "nonfinite_smc_weights":
            record["retirement_kind"] = "nonfinite_smc_weights"
        if isinstance(exc, FitTimeout):
            record.update(exc.as_record(), retirement_kind="fit_timeout")
            if scope != "candidate":
                # The candidate_failures copy names its class; the refit
                # ledgers (`refit_failed_models`, `active_refit_failures`) are unchanged.
                failures.append({**record, "failure_class": failure_class(status=record["status"])})
        (predict_failures if stage == "predict" else refit_failed_models).append(record)
        print(
            f"REFIT_FAILED t={t} scope={scope} stage={stage} model={record['model_id']} "
            f"family={record['family']} cause={record['cause_code']} "
            f"nan_cells={substitutions['cells']} error={record['exception']}",
            flush=True,
        )
        return record

    def refit_all(pool, t):
        """Refit every hypothesis in `pool`, capturing per-model failures.

        Returns (fits, failures) and never raises FOR A NUMERICAL FAILURE: a model
        whose SMC fit raises `CountInferenceFailure` is archived as `fit_failed`
        for that round, exactly like a failed candidate, and the run continues
        with the models that did fit. This is the whole reason this rule exists - a
        single PyMC SMC `ValueError: cannot convert float NaN to integer` inside
        one ACTIVE model's refit used to end the entire run with no summary.

        The catch is narrowed to that ONE type, which `count_inference.fit`
        raises around the sampling block and nowhere else. Everything else that
        can go wrong inside `refit` - the NPZ and JSON artifact writes, the
        diagnostics, the prior probe, a bug in this file - is NOT caught and
        fails the run, because a full disk is not an impossible likelihood.
        """
        survivors, failed = [], []
        for hypothesis in pool:
            nan_before = nan_substitution_counter()
            try:
                result = refit(hypothesis, t)
            except CountInferenceFailure as exc:
                key = hypothesis.model_id
                record = record_refit_failure(hypothesis, exc, t, "active", account_nan(nan_before))
                failed.append(record)
                archive[key] = {
                    **candidate_meta.get(key, {}),
                    "model_id": key,
                    "expression": hypothesis.expression,
                    "family": hypothesis.family,
                    "parameters": dict(hypothesis.parameters),
                    "proposed_round": born_round.get(key, t),
                    "outcome": "fit_failed",
                    "reason": {"error": record["exception"]},
                }
            else:
                account_nan(nan_before)
                survivors.append(result)
        return survivors, failed

    def refit_active(pool, t):
        """Refit the active pool for round t under the refit-failure continuation rule.

        EMPTY-POOL RULE (pre-declared): if EVERY active model
        fails NUMERICALLY in the same round the run TERMINATES at that round. There is no
        re-seed, nothing is imputed, and the last successful round's predictions
        are NOT carried forward - `summary.json` and `points.csv` are never
        written, the later checkpoints simply do not exist, and the run is
        recorded through the failure path with `status: "pool_empty"`. At least
        one survivor and the run continues with the survivors.
        """
        survivors, failed = refit_all(pool, t)
        if failed:
            active_refit_failures.append(
                {
                    "round": t,
                    "failure_count": len(failed),
                    "retired_model_ids": [item["model_id"] for item in failed],
                    "failures": failed,
                    "active_after": len(survivors),
                    "pool_empty": not survivors,
                }
            )
        if not survivors:
            raise PoolEmpty(
                f"active pool empty at round {t}: all {len(failed)} active refits failed "
                f"numerically (CountInferenceFailure); "
                "the run stops here by the pre-declared empty-pool rule (no re-seed, "
                "no carried-forward predictions)"
            )
        return survivors

    def reevaluate(t, previous, signature):
        """Archive re-evaluation: refit ONLY the inactive retired models, from the original prior.

        The active fits are reused exactly as they are - an old posterior is
        never multiplied by the data a second time - and every retired model is
        refit from the original prior on all current public data before the
        merged pool is ranked again and truncated to the top twelve.
        """
        nonlocal fits
        retired_ids = [
            key for key, entry in archive.items()
            if entry["outcome"] == "retired_by_evidence" and key not in timed_out_ids
        ]
        revived, refit_failures = [], []
        for key in retired_ids:
            nan_before = nan_substitution_counter()
            try:
                revived.append(refit(hypotheses[key], t))
            except CountInferenceFailure as exc:
                # Only a NUMERICAL failure is a failed revival;
                # it is recorded with scope="reevaluation" in the same structured
                # ledger (the model was already retired_by_evidence, so nothing
                # is retired here). Anything else propagates and fails the run.
                record = record_refit_failure(
                    hypotheses[key], exc, t, "reevaluation", account_nan(nan_before)
                )
                refit_failures.append({"model_id": key, "error": record["exception"]})
                if isinstance(exc, FitTimeout):
                    archive[key].update(outcome="fit_failed", reason={"error": record["exception"]})
            else:
                account_nan(nan_before)
        accounting["reevaluation_refits"] += len(retired_ids)
        before_ids = [v.hypothesis.model_id for v in fits]
        fits, retired = retain(fits + revived, active_maximum)
        stamp_active(t)
        for result in fits:
            mark_active(result.hypothesis, t)
        retire(retired, t, fits[-1].log_evidence)
        row, updated = score_after(previous, signature, "after_reevaluation", "post_reevaluation")
        event = {
            "round": t,
            "refit_ids": retired_ids,
            "refit_count": len(retired_ids),
            "refit_failures": refit_failures,
            "refit_basis": "original_prior_and_all_public_observations",
            "resurrected_ids": sorted({v.hypothesis.model_id for v in fits} - set(before_ids)),
            "retired_ids": [v.hypothesis.model_id for v in retired],
            "before_ids": before_ids,
            "after_ids": [v.hypothesis.model_id for v in fits],
            "pre_scores": {"phase": previous["phase"], "bma": previous["bma"]},
            "post_scores": {"phase": row["phase"], "bma": row["bma"]},
            "post_equals_pre": bool(row.get("post_equals_pre")),
        }
        write_json(out / "resurrection" / f"t{t:02d}.json", event)
        resurrection_events.append(event)
        return row, updated

    def make_admit(t, known, prior_probe_cache, prior_rows=None):
        """The same prior-probe hook serves single calls and locked wave calls."""
        def admit(hypothesis):
            """Admit a proposed hypothesis into the pool if it is new and valid."""
            key = hypothesis.model_id
            if key in known:
                return
            if real_outcome:
                hypothesis = gaussian.as_real(hypothesis)
            probe, _ = probe_candidate(hypothesis, interface)
            if probe["passed"]:
                prior_probe_cache[key] = probe
                return
            register(hypothesis, t)
            accounting["prior_probe_rejections"] += 1
            known.add(key)
            archive[key] = {
                "model_id": key, "expression": hypothesis.expression,
                "family": hypothesis.family, "parameters": dict(hypothesis.parameters),
                "proposed_round": born_round[key],
                "prior_probe": probe, "note": "", "outcome": "rejected",
                "reason": prior_probe_reason(probe),
            }
            if prior_rows is not None:
                prior_rows.append((key, archive[key]))
            raise PriorProbeRejection(probe["detail"])

        return admit

    def round_packet(t, members=None):
        """Build the belief packet shown to the proposer at round `t`."""
        packet_constraints = ({**constraints, "initial_structures": []}
                              if t == 0 and initial_catalogue == "off" else constraints)
        packet = build_belief_packet(packet_observations(public), pool_members(t, members),
                                     list(archive.values()), packet_constraints, interface, t)
        text = serialize_packet(packet)
        return text, packet_digest(text)

    def parallel_wave(t, count):
        """BoxingGym's single-candidate wave, over CORE's own fit/archive state."""
        nonlocal fits
        tic = time.perf_counter()
        stem = "t00_burst" if t == 0 else f"t{t:02d}"
        lock = threading.Lock()
        gauge = {"in_flight": 0, "max_in_flight": 0,
                 "mock_offset": (t - 1) % len(P1_MOCK_EXPRESSIONS) if t else 0}
        previous_ids = [f.hypothesis.model_id for f in fits]
        known = set(archive) | set(previous_ids)
        events, fitted, transports, waves = [], [], [], []
        # Each event's own `model_id_preimage`, added to the burst record only (the
        # same event dicts also reach summary.json's candidate_failures, which must not change).
        event_preimages = []
        prior_probe_cache, wave_packets = {}, []
        fit_attempt_count = 0

        def admit_locked(hook):
            """Wrap an admission hook so that calls are serialized under the lock."""
            def admit(candidate):
                """Call the hook on a candidate while holding the lock."""
                with lock:
                    hook(candidate)
            return admit

        def call(wave, index, *, packet_text, digest, admit, burst=True):
            """Send one proposal request of a wave and return its transport record."""
            # `burst` is "refill" for a round-0 refill wave, True otherwise.
            context = {
                "run_id": str(out), "world_seed": args.seed, "cell": args.cell,
                "proposal_rep": args.replicate, "round": t,
                "burst_call_index": index, "burst_wave": wave,
            }
            if proposer is None:
                with lock:
                    offset = gauge["mock_offset"]
                    gauge["mock_offset"] += 1
                if args.cell[0] == "1":
                    form = P1_MOCK_EXPRESSIONS[offset % len(P1_MOCK_EXPRESSIONS)]
                    family = (P1_MOCK_FAMILIES[offset % len(P1_MOCK_FAMILIES)]
                              if args.cell[1] == "1" else args.initial_family)
                else:
                    initial = initial_hypotheses(args.initial_family)
                    alternatives = [name for name in FAMILIES if name != args.initial_family]
                    form = initial[offset % len(initial)].expression
                    family = alternatives[(offset // len(initial)) % len(alternatives)]
                candidates = ([Hypothesis(form, family, mock_parameters(form))]
                              if args.cell[:2] != "00" else [])
                if real_outcome:
                    candidates = [gaussian.as_real(h) for h in candidates]
                return candidates, {
                    "provider": "none", "mock": True, "api_request_attempted": False,
                    "policy": "P1", "status": "parsed", "calls": 0, "repairs": 0,
                    "rejected_candidates": [], "note": "", **context,
                    **composed_system_prompt(burst=burst)[1], **digest,
                }
            for retry in range(BURST_RETRIES + 1):
                with lock:
                    gauge["in_flight"] += 1
                    gauge["max_in_flight"] = max(gauge["max_in_flight"], gauge["in_flight"])
                try:
                    candidates, record = proposer.propose_one(
                        packet_text,
                        context={**context, **({"burst_retry": retry} if retry else {})},
                        initial_family=args.initial_family,
                        initial_structures=initial_hypotheses(args.initial_family),
                        repair_attempts=1, allowed_families=("normal",) if real_outcome else FAMILIES,
                        max_rounds=args.rounds, admit=admit, burst=burst,
                    )
                finally:
                    with lock:
                        gauge["in_flight"] -= 1
                rate_limited = record["status"] != "parsed" and any(
                    item.get("http_status") == 429 for item in record["rejected_candidates"])
                if not rate_limited or retry == BURST_RETRIES:
                    break
                time.sleep(BURST_BACKOFF_SECONDS * 2 ** retry)
            if real_outcome:
                candidates = [gaussian.as_real(candidate) for candidate in candidates]
            return candidates, record

        def admit_candidate(hypothesis, note):
            """Admit a candidate hypothesis, fit it, and return its event record."""
            nonlocal fit_attempt_count
            item = hypothesis.as_dict()
            key = register(hypothesis, t)
            probe = prior_probe_cache.pop(key, None)
            if probe is None:
                probe, _ = probe_candidate(hypothesis, interface)
            loglik_probe = None
            if fit_guards == "v1" and probe["passed"] and key not in known:
                loglik_probe = gaussian.guarded_loglik_probe(
                    hypothesis, public, interface, response_unit, response_center,
                    PRIOR_PROBE_SAMPLES) if real_outcome else loglik_prior_probe(
                    hypothesis, [d["u"] for d in public], [d["y"] for d in public],
                    interface, prior_draws=prior_particles(hypothesis),
                    n_trials=interface["trials"],
                )
                probe = {**probe, "loglik_prior_probe": loglik_probe}
            source = "initial_burst" if t == 0 else "parallel_wave"
            item.update(prior_probe=probe, note=note, source=source)
            candidate_meta[key] = {"prior_probe": probe, "note": note, "source": source}
            stub = {"model_id": key, "expression": hypothesis.expression,
                    "family": hypothesis.family, "parameters": dict(hypothesis.parameters),
                    "proposed_round": born_round[key],
                    "prior_probe": probe, "note": note, "source": source}
            accepted = False
            if key in known:
                item["status"] = "duplicate_not_refitted"
                accounting["duplicates"] += 1
                accounting["burst_duplicates"] += 1
                matched = next((f.hypothesis for f in fits if f.hypothesis.model_id == key),
                               hypotheses.get(key, hypothesis))
                archive[duplicate_archive_key(archive, t, key)] = {
                    **stub, "proposed_round": t, "outcome": "duplicate",
                    "reason": {"matched_model_id": key, "matched_expression": matched.expression},
                }
            elif not probe["passed"]:
                item["status"] = "rejected_prior_probe"
                accounting["prior_probe_rejections"] += 1
                known.add(key)
                archive[key] = {**stub, "outcome": "rejected", "reason": prior_probe_reason(probe)}
            elif loglik_probe is not None and not loglik_probe["passed"]:
                item["status"] = "rejected_loglik_prior_probe"
                accounting["prior_probe_rejections"] += 1
                known.add(key)
                archive[key] = {**stub, "outcome": "rejected",
                                "reason": {"validation_message": loglik_probe["reason"]}}
            else:
                known.add(key)
                accounting["fit_attempts"] += 1
                fit_attempt_count += 1
                nan_before = nan_substitution_counter()
                try:
                    fitted.append(refit(hypothesis, t))
                except CountInferenceFailure as exc:
                    record = record_refit_failure(
                        hypothesis, exc, t, "candidate", account_nan(nan_before))
                    if isinstance(exc, FitTimeout):
                        item.update(exc.as_record(), model_id=key, round=t)
                    elif fit_guards == "v1" and exc.cause_code == "nonfinite_smc_weights":
                        item.update(status="fit_failed", cause_code=exc.cause_code,
                                    retirement_kind="nonfinite_smc_weights", stage="sampling",
                                    model_id=key, round=t, error=record["exception"])
                    else:
                        item.update(status="fit_or_predict_failed", error=record["exception"])
                    failures.append(item)
                    archive[key] = {**stub, "outcome": "fit_failed",
                                    "reason": {"error": record["exception"]}}
                except Exception as exc:  # noqa: BLE001 - same per-candidate wave boundary as BoxingGym
                    account_nan(nan_before)
                    item.update(status="fit_or_predict_failed", error=f"{type(exc).__name__}: {exc}")
                    failures.append(item)
                    archive[key] = {**stub, "outcome": "fit_failed",
                                    "reason": {"error": f"{type(exc).__name__}: {exc}"}}
                else:
                    account_nan(nan_before)
                    accepted = True
                    accounting["fit_successes"] += 1
                    archive[key] = {
                        **stub, "outcome": "active",
                        "reason": {"admitted_round": t, "active_rounds": active_rounds[key]},
                    }
                    item.update(status="fitted", log_evidence=fitted[-1].log_evidence)
            # The event names its class (None when fitted); a candidate_failures
            # entry that is this same dict names it too.
            item["failure_class"] = failure_class(status=item["status"])
            events.append(item)
            event_preimages.append(hypothesis.model_id_preimage)
            return accepted

        shortfall = count
        # The round-0 burst runs at most three waves; a later round keeps its two.
        for wave in (INITIAL_BURST_WAVES if t == 0 else (0, 1)):
            if shortfall <= 0:
                break
            # A round-0 refill wave's packet shows the round's intermediate retained pool
            # (the active pool plus this round's fits so far); `fits` itself is untouched.
            refill = t == 0 and wave > 0
            shown = burst_refill_pool(fits + fitted, active_maximum) if refill else None
            packet_text, digest = round_packet(t, shown)
            wave_packets.append(packet_text)
            admit = admit_locked(make_admit(t, known, prior_probe_cache))
            with ThreadPoolExecutor(max_workers=max(1, min(burst_concurrency, shortfall))) as pool_:
                results = list(pool_.map(
                    partial(call, wave, packet_text=packet_text, digest=digest, admit=admit,
                            burst="refill" if refill else True),
                    range(shortfall)))
            accepted, wave_events, wave_transports = 0, len(events), len(transports)
            for index, (returned, record) in enumerate(results):
                transports.append({"wave": wave, "index": index, "transport": record})
                accounting["burst_calls"] += record["calls"]
                accounting["burst_repairs"] += record["repairs"]
                accounting["llm_calls"] += record["calls"]
                accounting["llm_repairs"] += record["repairs"]
                if wave:
                    accounting["burst_refill_calls"] += record["calls"]
                if record["status"] != "parsed":
                    accounting["burst_failed_calls"] += 1
                    failures.append({"round": t, "burst_wave": wave, "burst_call_index": index,
                                     "status": "proposal_failed", "transport": record})
                accounting["burst_candidates"] += len(returned)
                for candidate in returned:
                    accepted += admit_candidate(candidate, record.get("note", ""))
            waves.append(burst_wave_record(wave, shortfall, accepted, events[wave_events:],
                                           transports[wave_transports:], packet_text)
                         if t == 0 else {"wave": wave, "calls": shortfall, "accepted": accepted})
            shortfall -= accepted
        (out / "packets" / f"{stem}.json").write_text(wave_packets[0])
        if t == 0:
            # Every executed wave's packet; t00_burst.json stays wave 0's copy.
            for wave, text in enumerate(wave_packets):
                (out / "packets" / f"t00_burst_w{wave}.json").write_text(text)
        accounting["burst_max_in_flight"] = max(accounting["burst_max_in_flight"], gauge["max_in_flight"])
        timings["proposal"] += time.perf_counter() - tic
        if t:
            accounting["proposal_rounds_executed"] += 1
            accounting["valid_candidates"] += len(events)
        fits, retired = retain(fits + fitted, active_maximum)
        if not fits:
            raise ValueError("no models remain after the parallel proposal wave")
        new_ids = sorted({f.hypothesis.model_id for f in fits} - set(previous_ids))
        if t == 0:
            accounting["burst_admitted"] += len(new_ids)
            accounting["burst_wave_count"] += len(waves)
        stamp_active(t)
        for result in fits:
            mark_active(result.hypothesis, t)
        retire(retired, t, fits[-1].log_evidence)
        counts = wave_event_counts(events, transports, count)
        for key, value in counts.items():
            if key != "duplicates":
                accounting[key] += value
        for kind, value in failure_class_counts(events, transports).items():
            accounting["failure_class_counts"][kind] += value
        write_final_proposal_json(
            out / "proposals" / f"{stem}.json",
            {"model_id_schema": MODEL_ID_SCHEMA, "notation_normal_form": NOTATION_NORMAL_FORM,
             "public_observations": public, "mock": proposer is None, "transports": transports,
             "cell": args.cell,
             "raw_response": [{k: v for k, v in e.items() if k in ("expression", "family", "parameters")}
                              for e in events],
             "events": [{**e, "model_id_preimage": preimage}
                        for e, preimage in zip(events, event_preimages, strict=True)],
             "valid_candidate_count": len(events),
             "duplicate_count": sum(e["status"] == "duplicate_not_refitted" for e in events),
             "fit_attempt_count": fit_attempt_count, "fit_success_count": len(fitted),
             "waves": waves, "requested": count, **counts,
             "max_in_flight": gauge["max_in_flight"], "concurrency": burst_concurrency,
             **({"initial_catalogue": initial_catalogue,
                 "initial_burst_refill_policy": INITIAL_BURST_REFILL_POLICY,
                 "base_prompt_sha256": refill_identity["base_prompt_sha256"],
                 "burst_refill_paragraph_sha256": refill_identity["burst_refill_paragraph_sha256"]}
                if t == 0 else {}),
             "before_ids": previous_ids, "after_ids": [f.hypothesis.model_id for f in fits],
             "admitted_ids": new_ids, "retired_ids": [f.hypothesis.model_id for f in retired]},
        )
        return new_ids

    # The round in progress, for a summary the paid-call guard may have to write.
    current_round = 0
    try:
        if initial_pool_file is not None:
            initial = initial_pool_file.hypotheses
            fits = refit_active([gaussian.as_real(h) for h in initial] if real_outcome else initial, 0)
            stamp_active(0)
            for result in fits:
                mark_active(result.hypothesis, 0)
        elif initial_catalogue != "off":
            initial = initial_hypotheses(args.initial_family)
            fits = refit_active([gaussian.as_real(h) for h in initial] if real_outcome else initial, 0)
            stamp_active(0)
            for result in fits:
                mark_active(result.hypothesis, 0)
        if initial_proposals:
            parallel_wave(0, initial_proposals)
        pre_row, signature = checkpoint(0, "initial", "initial" if policy == "P1" else None)
        designs = np.linspace(0, 1, grid_size)
        candidate_metadata = candidate_set_record(designs, "core")
        if hasattr(args, "candidate_exclusion_policy"):
            candidate_metadata["candidate_exclusion_policy"] = candidate_exclusion_policy
        for t in range(1, args.rounds + 1):
            current_round = t
            if real_outcome and candidate_exclusion_policy == "observed":
                observed = {row["u"] for row in public}
                designs = np.asarray([u for u in np.linspace(0, 1, grid_size)
                                      if float(u) not in observed])
                if not len(designs):
                    raise ValueError(f"Gaussian CORE design pool exhausted: profile={args.profile}, grid_size={grid_size}")
                candidate_metadata = candidate_set_record(designs, "core")
                if hasattr(args, "candidate_exclusion_policy"):
                    candidate_metadata["candidate_exclusion_policy"] = candidate_exclusion_policy
            tic = time.perf_counter()
            acquisition_seed = acquisition_seed_for(args.seed, args.replicate, t)
            acquisition_started = time.perf_counter()
            if design_policy == "epig":
                u, rows = choose_epig(fits, designs, acquisition_seed, eig_samples, interface,
                                      interface.get("trials"), target_points=epig_targets,
                                      discrete_path_state=epig_path_state)
                acquisition_elapsed = time.perf_counter() - acquisition_started
                acquisition_count = len(rows)
                write_json(out / "acquisition" / f"t{t:02d}.json",
                           {**epig_acquisition_record(u, rows, acquisition_seed), **candidate_metadata})
            elif design_policy_applies:
                # `rows` and `target` keep their original shape and values under the
                # default policy, so a joint_eig run's record is byte-identical
                # to the one earlier versions wrote; everything new is inside the `policy` block.
                u, rows, policy_record = (gaussian.select_design if real_outcome else select_design)(
                    fits,
                    designs,
                    acquisition_seed,
                    design_policy,
                    eig_samples,
                    interface,
                    interface.get("trials"),
                    **({"observed_designs": [row["u"] for row in public],
                        "bounds": config["design_normalization_bounds"]}
                       if design_policy == "space_filling" else {}),
                )
                acquisition_elapsed = time.perf_counter() - acquisition_started
                acquisition_count = len(rows) if design_policy_applies else len(designs)
                write_json(
                    out / "acquisition" / f"t{t:02d}.json",
                    {
                        **candidate_metadata,
                        "seed": acquisition_seed,
                        "chosen": u,
                        "rows": rows,
                        "target": TARGET[design_policy],
                        "policy": policy_record,
                    },
                )
            else:
                u = float(np.random.default_rng(acquisition_seed).choice(designs))
                acquisition_elapsed = time.perf_counter() - acquisition_started
                acquisition_count = len(rows) if design_policy_applies else len(designs)
                write_json(
                    out / "acquisition" / f"t{t:02d}.json",
                    {**candidate_metadata, "seed": acquisition_seed, "chosen": u, "target": "random"},
                )
            timings["eig"] += time.perf_counter() - tic
            record_acquisition_timing(out, t, design_policy if design_policy_applies else "random",
                                      acquisition_elapsed, acquisition_count)
            observation = world.observe(u)
            one_step_row = {}
            if policy == "P1":
                # Score the new observation BEFORE it is incorporated, under
                # each active model's t-1 posterior. Public data
                # only; a model born later keeps null for this round.
                for history in one_step.values():
                    history.append(None)
                for result in fits:
                    key = result.hypothesis.model_id
                    value = one_step_log_score(
                        result, observation["u"], observation["y"], interface
                    )
                    one_step[key][t - 1] = value
                    one_step_row[key] = value
            public.append(observation)
            append_json(out / "observations.jsonl", {"round": t, **observation})
            fits = refit_active([previous.hypothesis for previous in fits], t)
            stamp_active(t)
            proposal_due = t in proposal_rounds and args.cell[:2] != "00"
            pre_row, signature = checkpoint(
                t,
                "before_proposal" if proposal_due else "updated",
                "pre_proposal" if policy == "P1" else None,
                one_step_row or None,
            )
            # --- archive re-evaluation rounds (default off) ---
            if t in reevaluation_rounds:
                pre_row, signature = reevaluate(t, pre_row, signature)
            # --- end archive re-evaluation ---
            if proposal_due and parallel_proposals:
                admissions += len(parallel_wave(t, parallel_proposals))
                pre_row, signature = score_after(
                    pre_row, signature, "after_proposal", "post_proposal")
            elif proposal_due:
                tic = time.perf_counter()
                packet_text, digest = "", {}
                if policy == "P1":
                    packet_text, digest = round_packet(t)
                    (out / "packets" / f"t{t:02d}.json").write_text(packet_text)
                catalog = (
                    candidate_catalog(args.cell, args.initial_family) if policy == "P1" else None
                )
                known = set(archive) | {f.hypothesis.model_id for f in fits}
                prior_probe_cache = {}

                prior_rows = []
                admit = make_admit(t, known, prior_probe_cache, prior_rows)

                exhausted = catalog is not None and set(catalog) <= known
                if exhausted:
                    # The cell's catalog is finite and spent; the
                    # LLM cannot say anything new, so no call is made at all.
                    proposals = []
                    transport_record = {
                        "provider": "none",
                        "mock": proposer is None,
                        "api_request_attempted": False,
                        "policy": policy,
                        "status": "catalog_exhausted",
                        "calls": 0,
                        "repairs": 0,
                        "rejected_candidates": [],
                        "note": "",
                        "catalog_size": len(catalog),
                        **prompt_identity(),
                        **digest,
                    }
                    catalog_exhausted_rounds.append(t)
                    accounting["catalog_exhausted"] += 1
                elif proposer is None:
                    proposals = (
                        # The horizon goes positionally: the timeout and hang-guard tests
                        # stand in for this function with a `lambda *args`.
                        mock_proposal_p1(public, t, args.cell, args.initial_family, args.rounds)
                        if policy == "P1"
                        else mock_proposals(
                            tuple((d["u"], d["y"]) for d in public),
                            t,
                            args.cell,
                            args.initial_family,
                        )
                    )
                    transport_record = {
                        "provider": "none",
                        "mock": True,
                        "api_request_attempted": False,
                    }
                    if policy == "P1":
                        transport_record.update(
                            policy=policy,
                            status="parsed" if proposals else "no_candidate",
                            calls=0,
                            repairs=0,
                            rejected_candidates=[],
                            note="",
                            **prompt_identity(),
                            **digest,
                        )
                elif policy == "P1":
                    proposals, transport_record = proposer.propose_one(
                        packet_text,
                        context={
                            "run_id": str(out),
                            "world_seed": args.seed,
                            "cell": args.cell,
                            "proposal_rep": args.replicate,
                            "round": t,
                        },
                        initial_family=args.initial_family,
                        repair_attempts=repair_attempts,
                        **({"repair_receipt": label_receipt} if repair_attempts == 2 else {}),
                        # The cell's family menu, exactly as the round-0 burst
                        # passes it above. Without it `judge_candidate` falls back to
                        # `FAMILIES` (the count three), so on a Gaussian cell the packet
                        # asks the model for `normal` (count_proposer: the menu is
                        # `[initial_family]` whenever cell[1] == "0") and the parser then
                        # refuses it as `unsupported count family: normal` - every round
                        # from 1 on, deterministically. An earlier version parametrized this check for
                        # BoxingGym and deliberately left CORE's default alone while CORE
                        # was binomial-only; the Gaussian worlds made that default wrong
                        # and the burst port added the argument only to its own new call.
                        allowed_families=("normal",) if real_outcome else FAMILIES,
                        max_rounds=args.rounds,
                        admit=admit,
                    )
                    accounting["llm_calls"] += transport_record["calls"]
                    accounting["llm_repairs"] += transport_record["repairs"]
                    if transport_record["status"] != "parsed":
                        failures.append(
                            {"round": t, "status": "proposal_failed", "transport": transport_record}
                        )
                else:
                    proposals, transport_record = proposer.propose(
                        public,
                        [result.hypothesis.as_dict() for result in fits],
                        context={
                            "run_id": str(out),
                            "world_seed": args.seed,
                            "cell": args.cell,
                            "proposal_rep": args.replicate,
                            "event": t,
                        },
                        initial_family=args.initial_family,
                    )
                    if transport_record["status"] != "parsed":
                        failures.append(
                            {"round": t, "status": "proposal_failed", "transport": transport_record}
                        )
                if real_outcome:
                    proposals = [gaussian.as_real(h) for h in proposals]
                project_transport_rejections(archive, transport_record, prior_rows, t)
                transport_failure_rows.extend(transport_record.get("transport_failure_rows", []))
                write_proposed_json(
                    out / "proposals" / f"t{t:02d}.json",
                    {"public_observations": public, "mock": proposer is None,
                     "transport": transport_record, "cell": args.cell,
                     "raw_response": [h.as_dict() for h in proposals]},
                )
                validate_cell_candidates(proposals, args.cell, args.initial_family,
                                           **({"allowed_families": ("normal",)} if real_outcome else {}))
                timings["proposal"] += time.perf_counter() - tic
                accounting["proposal_rounds_executed"] += 1
                accounting["valid_candidates"] += len(proposals)
                accounting["rejected_candidates"] += len(transport_record.get("rejected_candidates", []))
                events, candidates, fit_attempt_count = [], [], 0
                # "Already seen" is the whole run history, not just
                # the live pool - `known` is the same active-plus-archive set the
                # exhaustion check above uses. A retired, fit-failed or
                # probe-rejected model_id that is proposed again is therefore a
                # duplicate, never a refit; an inactive model returns to the pool
                # only through the archive re-evaluation path.
                seen = set(known)
                for hypothesis in proposals[:proposal_limit]:
                    item = hypothesis.as_dict()
                    key = register(hypothesis, t)
                    probe = prior_probe_cache.pop(key, None)
                    if probe is None:
                        probe, _ = probe_candidate(hypothesis, interface)
                    loglik_probe = None
                    if fit_guards == "v1" and probe["passed"] and key not in seen:
                        loglik_probe = gaussian.guarded_loglik_probe(
                            hypothesis, public, interface, response_unit, response_center,
                            PRIOR_PROBE_SAMPLES) if real_outcome else loglik_prior_probe(
                            hypothesis, [d["u"] for d in public], [d["y"] for d in public],
                            interface, prior_draws=prior_particles(hypothesis),
                            n_trials=interface["trials"],
                        )
                        probe = {**probe, "loglik_prior_probe": loglik_probe}
                    note = transport_record.get("note", "")
                    item["prior_probe"] = probe
                    item["note"] = note
                    candidate_meta[key] = {"prior_probe": probe, "note": note}
                    stub = {
                        "model_id": key,
                        "expression": hypothesis.expression,
                        "family": hypothesis.family,
                        "parameters": dict(hypothesis.parameters),
                        "proposed_round": born_round[key],
                        "prior_probe": probe,
                        "note": note,
                    }
                    if key in seen:
                        # The proposal repeats a model the run has already seen -
                        # active, or archived as retired / fit-failed / rejected.
                        # That entry keeps its own outcome and reason untouched;
                        # the duplicate is archived separately under a synthetic
                        # key and linked to it. No fit is attempted,
                        # so a retired model cannot re-enter the pool this way.
                        item["status"] = "duplicate_not_refitted"
                        accounting["duplicates"] += 1
                        matched = next(
                            (f.hypothesis for f in fits if f.hypothesis.model_id == key),
                            hypotheses.get(key, hypothesis),
                        )
                        archive[duplicate_archive_key(archive, t, key)] = {
                            **stub,
                            "proposed_round": t,
                            "outcome": "duplicate",
                            "reason": {
                                "matched_model_id": key,
                                "matched_expression": matched.expression,
                            },
                        }
                    elif not probe["passed"]:
                        # Prior finiteness probe: no SMC fit is attempted for a
                        # candidate that is non-finite under its own prior
                        # anywhere on the interface reference grid.
                        item["status"] = "rejected_prior_probe"
                        accounting["prior_probe_rejections"] += 1
                        seen.add(key)
                        archive[key] = {
                            **stub,
                            "outcome": "rejected",
                            "reason": prior_probe_reason(probe),
                        }
                    elif loglik_probe is not None and not loglik_probe["passed"]:
                        item["status"] = "rejected_loglik_prior_probe"
                        accounting["prior_probe_rejections"] += 1
                        seen.add(key)
                        archive[key] = {
                            **stub, "outcome": "rejected",
                            "reason": {"validation_message": loglik_probe["reason"]},
                        }
                    else:
                        seen.add(key)
                        # Counted here and nowhere else: one increment per actual
                        # fit call, so a duplicate or a probe-rejected candidate
                        # never shows up as a fit attempt.
                        accounting["fit_attempts"] += 1
                        fit_attempt_count += 1
                        nan_before = nan_substitution_counter()
                        # The FIT and the PREDICT are separate
                        # events with separate ledgers. Only a typed
                        # `CountInferenceFailure` from the fit is a numerical
                        # failure of the model; a predict failure is recorded as
                        # `stage="predict"` and never counts as a NaN retirement;
                        # any other exception fails the run. The archive `reason`
                        # keeps its original `{"error": ...}` shape in both cases.
                        try:
                            result = refit(hypothesis, t)
                        except CountInferenceFailure as exc:
                            record = record_refit_failure(
                                hypothesis, exc, t, "candidate", account_nan(nan_before)
                            )
                            item.update(status="fit_failed", error=record["exception"])
                            if isinstance(exc, FitTimeout):
                                item.update(exc.as_record())
                            failures.append(item)
                            archive[key] = {
                                **stub,
                                "outcome": "fit_failed",
                                "reason": {"error": record["exception"]},
                            }
                        else:
                            # A successful FIT is counted here, before the
                            # predict, so "fit succeeded and predict failed" can
                            # never be folded back into the fit failure rate.
                            accounting["fit_successes"] += 1
                            substitutions = account_nan(nan_before)
                            try:
                                if real_outcome:
                                    gaussian.scalar.predictive_summary([result], designs, acquisition_seed)
                                else:
                                    predict([result], designs, interface, interface["trials"])
                            except ValueError as exc:
                                # Only a numerical scoring failure is a
                                # `predict_failed` candidate; an OSError, a
                                # TypeError or an unrecognized ValueError is a run
                                # failure and must not hide behind a candidate.
                                if not numeric_verdict(exc, PREDICT_NUMERIC):
                                    raise
                                record = record_refit_failure(
                                    hypothesis, exc, t, "candidate", substitutions, "predict"
                                )
                                item.update(status="predict_failed", error=record["exception"])
                                failures.append(item)
                                archive[key] = {
                                    **stub,
                                    "outcome": "fit_failed",
                                    # the archive enum is fixed by
                                    # belief_packet.ARCHIVE_REASON_KEYS, so the
                                    # distinction is carried IN the reason text -
                                    # the belief packet must not read this as a
                                    # failed fit.
                                    "reason": {
                                        "error": "stage=predict; fit succeeded; "
                                        + record["exception"]
                                    },
                                }
                            else:
                                candidates.append(result)
                                item.update(status="fitted", log_evidence=result.log_evidence)
                    # See parallel_wave - the event names its class.
                    item["failure_class"] = failure_class(status=item["status"])
                    events.append(item)
                for kind, value in failure_class_counts(
                        events, [{"transport": transport_record}]).items():
                    accounting["failure_class_counts"][kind] += value
                previous_ids = [f.hypothesis.model_id for f in fits]
                before_truncation = fits + candidates
                fits, retired = retain(before_truncation, active_maximum)
                if retired:
                    from .count_diagnostics import same_round_truncation
                    if real_outcome:
                        same_round_truncation = gaussian.same_round_truncation

                    tic = time.perf_counter()
                    diagnostic, arrays = same_round_truncation(
                        before_truncation,
                        fits,
                        evaluation_u,
                        t,
                        len(public),
                        interface,
                        interface.get("trials"),
                    )
                    timings["forward"] += time.perf_counter() - tic
                    write_json(out / "truncation" / f"t{t:02d}.json", diagnostic)
                    np.savez_compressed(out / "truncation" / f"t{t:02d}.npz", **arrays)
                new_ids = sorted({f.hypothesis.model_id for f in fits} - set(previous_ids))
                admissions += len(new_ids)
                stamp_active(t)
                for result in fits:
                    mark_active(result.hypothesis, t)
                retire(retired, t, fits[-1].log_evidence)
                write_final_proposal_json(
                    out / "proposals" / f"t{t:02d}.json",
                    {
                        "public_observations": public,
                        "mock": proposer is None,
                        "transport": transport_record,
                        "proposal_rep_seed": config["seed_fields"]["proposal_rep"],
                        "proposal_rep_seed_consumed": False,
                        "cell": args.cell,
                        "raw_response": [h.as_dict() for h in proposals],
                        "events": events,
                        "valid_candidate_count": len(proposals),
                        "duplicate_count": sum(
                            e["status"] == "duplicate_not_refitted" for e in events
                        ),
                        "fit_attempt_count": fit_attempt_count,
                        "fit_success_count": len(candidates),
                        "proposed_families": [h.family for h in proposals],
                        "before_ids": previous_ids,
                        "before_truncation_ids": [f.hypothesis.model_id for f in before_truncation],
                        "truncation_scope": "same_round_same_public_data_before_truncation_vs_active",
                        "after_ids": [f.hypothesis.model_id for f in fits],
                        "admitted_ids": new_ids,
                        "retired_ids": [f.hypothesis.model_id for f in retired],
                        "model_prior": "uniform_on_distinct_active_pairs_conditional_on_current_pool",
                    },
                )
                pre_row, signature = score_after(
                    pre_row, signature, "after_proposal", "post_proposal"
                )
            elif t in proposal_rounds:
                validate_cell_candidates([], args.cell, args.initial_family)
                write_json(
                    out / "proposals" / f"t{t:02d}.json",
                    {
                        "cell": args.cell,
                        "round": t,
                        "observation_count": len(public),
                        "valid_candidate_count": 0,
                        "duplicate_count": 0,
                        "fit_attempt_count": 0,
                        "fit_success_count": 0,
                        "raw_response": [],
                        "reason": "fixed_pool_no_proposal",
                    },
                )
        own, child = (
            resource.getrusage(resource.RUSAGE_SELF),
            resource.getrusage(resource.RUSAGE_CHILDREN),
        )
        summary = {
            "status": "completed",
            "profile": args.profile,
            "rounds": args.rounds,
            "observations": len(public),
            "admissions": admissions,
            "active_count": len(fits),
            "candidate_failures": failures,
            # Four refit-failure keys, all empty/zero on any run that earlier
            # versions could complete: the per-ROUND record of active models
            # retired because their fit failed numerically, the flat identity
            # list of every model lost that way (scope active | candidate |
            # reevaluation, always stage="fit"), the separate stage="predict"
            # ledger, and the run-level NaN -> -inf count.
            "active_refit_failures": active_refit_failures,
            "refit_failed_models": refit_failed_models,
            "predict_failures": predict_failures,
            "nan_substitutions": nan_substitutions,
            "timing_seconds": timings,
            "wall_seconds": time.perf_counter() - start,
            "user_cpu_seconds": own.ru_utime + child.ru_utime,
            "system_cpu_seconds": own.ru_stime + child.ru_stime,
            "paid_api_calls": 0 if proposer is None else proposer.attempts,
            "api_request_attempts": 0 if proposer is None else proposer.attempts,
            "billing_status": "no_calls" if proposer is None else "usage_in_llm_transport_records",
            "registered_run": False,
        }
        indexes = proposal_accounting_indexes(out, score_rows)
        accounting.update(indexes)
        summary.update(wave_cost_blocks(out))
        if policy == "P1":
            # The LLM budget is not equal across cells, so the
            # per-cell counts are reported, never assumed.
            summary["proposal_accounting"] = accounting
            summary["catalog_exhausted_rounds"] = catalog_exhausted_rounds
            # Real call failures remain in the archival do-no-harm denominator,
            # but are not hypotheses in the strict belief-packet candidate view.
            # An admitted model's entry (active or retired by evidence) carries its
            # model_id_preimage in this file only; the in-memory archive the packets read does not.
            write_json(out / "archive.json", [
                *({**entry, "model_id_preimage": hypotheses[entry["model_id"]].model_id_preimage}
                  if entry["outcome"] in ("active", "retired_by_evidence") else entry
                  for entry in archive.values()),
                *transport_failure_rows])
        else:
            # P4 has no accounting container; add only the declared index keys.
            summary.update(indexes)
        if reevaluation_rounds:
            summary["resurrection_events"] = resurrection_events
        if label_explicit or args.proposer in ("openai", "openai_compat"):
            # Shared helper - see
            # `points_label` and `config["label"]` above. The
            # resolved --run-label, under the same "only when forced" condition
            # as config, so summary and config never disagree and an unforced
            # run keeps the key set it had without --run-label.
            summary["label"] = run_label
        # Every duplicate EVENT (`accounting["duplicates"]`,
        # incremented once per repeated model_id seen this run) must have
        # exactly one archive entry - the collision-safe key above guarantees
        # it even under policy P4, which never serializes archive.json. A
        # bare assert: it writes nothing, so a passing run's output bytes
        # are exactly what they were before this check existed.
        assert (
            sum(1 for entry in archive.values() if entry["outcome"] == "duplicate")
            == accounting["duplicates"]
        ), "archive duplicate-entry count must equal duplicate-event count"
        # The top-level summary is masked as well as
        # the transport records it quotes, so no output file can carry a
        # credential even if a future field forwards provider text.
        recheck_manifest(label_receipt)
        if label_explicit:
            summary["label_receipt"] = label_receipt
        write_json(out / "summary.json", mask_secrets_deep(summary))
        write_points(out / "points.csv", config, score_rows, "completed", variant)
    except PaidCallBudgetExhausted:
        # The guard refused the call that would pass max_paid_calls. Every record
        # written so far stays; the summary says why and where, and main() exits nonzero.
        write_json(out / "summary.json", mask_secrets_deep({
            **budget_exhausted_summary(proposer, current_round),
            "api_request_attempts": proposer.attempts,
            "profile": args.profile,
            "rounds": args.rounds,
            "observations": len(public),
            "timing_seconds": timings,
            "wall_seconds": time.perf_counter() - start,
            **({"label": run_label}
               if label_explicit or args.proposer in ("openai", "openai_compat") else {}),
            **({"label_receipt": label_receipt} if label_explicit else {}),
        }))
        raise
    except BaseException as exc:  # noqa: BLE001 - recorded masked, re-raised masked
        write_json(
            out / "failure.json",
            {
                # The pre-declared empty-pool terminal state is its own
                # status. Nothing after it is written - no summary.json, no
                # points.csv, no imputed checkpoint, no carried-forward
                # prediction - so a consumer sees the run stop exactly there.
                "status": "pool_empty" if isinstance(exc, PoolEmpty) else "failed",
                "error": mask_secrets(f"{type(exc).__name__}: {exc}"),
                "traceback": mask_secrets(traceback.format_exc()),
                "rounds_completed": len(public) - len(D0),
                "active_refit_failures": active_refit_failures,
                "refit_failed_models": refit_failed_models,
                "predict_failures": predict_failures,
                "nan_substitutions": nan_substitutions,
                "timing_seconds": timings,
            },
        )
        # The re-raise is masked ONLY when the
        # exception text really carries a credential (`masked_exception`
        # returns the original otherwise), so an expected numerical failure -
        # PoolEmpty above included - still propagates as its own type with its
        # own traceback.
        raise masked_exception(exc) from None


def main():
    """Command-line entry point: parse arguments and run one CORE world."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("dev", "paper"), default="paper")
    parser.add_argument("--proposer", choices=("mock", "anthropic", "openai", "openai_compat"), default="mock")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=argparse.SUPPRESS)
    parser.add_argument("--llm-key-name", default=argparse.SUPPRESS)
    parser.add_argument("--llm-provider-order", type=normalize_provider_order, default=argparse.SUPPRESS)
    # Both stay absent from the namespace unless typed, so an
    # omitted flag leaves the request body, the config and the points columns
    # byte-identical. `--llm-reasoning` is openai_compat only (compat_config
    # refuses it elsewhere); `--llm-max-tokens` is a decoding setting for
    # every provider. An invalid value ends the command line with status 2.
    parser.add_argument("--llm-reasoning", type=parse_reasoning_spec, default=argparse.SUPPRESS)
    # OpenAI JSON mode (stock provider only) and the validation-message wording; both
    # SUPPRESS, so an omitted option leaves the request, config and receipt byte-identical.
    parser.add_argument("--llm-response-format", choices=RESPONSE_FORMATS, default=argparse.SUPPRESS)
    parser.add_argument("--validation-messages", choices=VALIDATION_MESSAGE_VERSIONS,
                        default=argparse.SUPPRESS)
    parser.add_argument("--llm-max-tokens", type=parse_max_tokens, default=argparse.SUPPRESS)
    parser.add_argument("--allow-paid-api", action="store_true")
    # The per-cell paid-call guard. SUPPRESS, so an omitted option leaves every
    # record byte-identical; stated, the proposer refuses the call that would pass it.
    parser.add_argument("--max-paid-calls", type=parse_max_paid_calls, default=argparse.SUPPRESS)
    parser.add_argument("--fit-guards", choices=("off", "v1"), default=argparse.SUPPRESS)
    parser.add_argument("--fit-timeout", dest="fit_timeout_seconds",
                        type=validate_fit_timeout, default=argparse.SUPPRESS)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, choices=tuple(WORLD_COEFFICIENTS), default=11)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--d0-policy", choices=("fixed4", "stratified4"), default=argparse.SUPPRESS)
    parser.add_argument("--candidate-exclusion", dest="candidate_exclusion_policy",
                        choices=("none", "observed"), default=argparse.SUPPRESS)
    parser.add_argument("--cell", choices=tuple(f"{s}{f}{a}" for s in "01" for f in "01" for a in "012"), default="111")
    parser.add_argument("--structure", choices=("in", "out", "den", "pow", *GAUSSIAN_STRUCTURES), default="out")
    parser.add_argument("--truth-family", choices=(*FAMILIES, "normal"), default="beta_binomial")
    parser.add_argument("--initial-family", choices=(*FAMILIES, "normal"), default=None)
    parser.add_argument("--proposal-policy", choices=("P1", "P4"), default="P1")
    parser.add_argument("--proposal-rounds", default="")
    parser.add_argument("--proposal-limit", type=int, default=1)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--initial-proposals", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--active-maximum", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--burst-concurrency", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--initial-catalogue", choices=("on", "off"), default=argparse.SUPPRESS)
    parser.add_argument("--initial-pool-file", default=argparse.SUPPRESS)
    parser.add_argument("--parallel-proposals", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--archive-reevaluation-rounds", default="")
    # Prior-scale and LLM-proposed-prior ablations. Both default to the frozen
    # production behaviour: the Table 3 role priors and no proposed-prior path.
    parser.add_argument(
        "--prior-template", choices=PRIOR_TEMPLATE_NAMES, default=DEFAULT_PRIOR_TEMPLATE
    )
    parser.add_argument("--priors", choices=("off", "llm"), default="off")
    # The expression grammar this run's candidates are judged
    # against. SUPPRESS for the same reason `--prompt-version` uses it - an
    # omitted option adds no config.json key at all, so a command line that
    # never states it records exactly what it always did, and the finished v2
    # waves keep their grammar and their bytes. Unstated means
    # DEFAULT_GRAMMAR_VERSION, read through `getattr` in `run`.
    parser.add_argument(
        "--grammar-version", choices=GRAMMAR_VERSIONS, default=argparse.SUPPRESS
    )
    # The numbered system prompt this run sends. SUPPRESS, so an omitted
    # option adds no config.json key at all and a command line that never states it
    # records exactly what it always did.
    parser.add_argument("--prompt-version", default=argparse.SUPPRESS)
    parser.add_argument("--design-policy", choices=POLICIES, default=argparse.SUPPRESS)
    # Omitted labels stay development; production needs registration.
    parser.add_argument("--run-label", choices=RUN_LABEL_CHOICES, default=None)
    parser.add_argument("--manifest-ref", default=argparse.SUPPRESS,
                        help="JSON object containing manifest path and whole-file sha256")
    args = parser.parse_args()
    if args.proposer == "openai_compat":
        compat_config(args)
    if args.llm_model is None:
        # Per-provider default: tests/smoke/development use gpt-4o,
        # main runs use claude-opus-4-7; unchanged for mock (never dereferenced).
        args.llm_model = "gpt-4o" if args.proposer == "openai" else "claude-opus-4-7"
    if not 1 <= args.rounds <= 80:
        parser.error("--rounds must be 1..80")
    if args.replicate < 0:
        parser.error("--replicate must be nonnegative")
    if (not 1 <= args.proposal_limit <= 3 or args.repair_attempts not in (0, 1, 2)
            or (args.repair_attempts == 2 and not getattr(args, "manifest_ref", None))):
        # "At most one format repair" is an invariant
        # of policy P1, not a tunable - the CLI used to accept 3 (and the
        # proposer then made four calls). The sole exception is registered A8:
        # run() verifies its manifest before outputs; the proposer rechecks the
        # same receipt before requests. This is not a global widening.
        parser.error("--proposal-limit must be 1..3 and --repair-attempts 0 or 1; "
                     "2 requires a verified CORE A8 launch contract; 3 or more is forbidden")
    overrides = args.proposal_rounds or args.archive_reevaluation_rounds
    if args.proposal_policy == "P4" and overrides:
        parser.error("policy P4 is the frozen legacy path; it takes no round overrides")
    try:
        validate_wave_options(args)
        validate_world_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    from .run_label_policy import RepairAuthorizationError
    try:
        run(args)
    except RepairAuthorizationError as exc:
        parser.error(str(exc))
    except PaidCallBudgetExhausted as exc:
        print(f"STOP {exc}; summary.json records stop_reason budget_exhausted", file=sys.stderr)
        raise SystemExit(PAID_CALL_BUDGET_EXIT_CODE) from None


if __name__ == "__main__":
    main()
