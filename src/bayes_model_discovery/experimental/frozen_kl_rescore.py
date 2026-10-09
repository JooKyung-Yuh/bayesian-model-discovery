"""One frozen evaluator, re-scoring completed cells' last checkpoint.

A completed cell directory is read (never written) and the KL divergence
KL(truth || BMA) from the true law to the Bayesian model averaging (BMA) predictive of
its FINAL checkpoint - the last `scores.jsonl` record, round `config.rounds`,
phase `after_proposal` - is computed again by a single fixed evaluator, so that
cells produced by different code generations are compared on one number rather
than on whatever each run happened to record. Nothing is re-implemented here:
the arithmetic is `boxinggym_truth_law.truth_law`/`kl_against_pool`,
`core_count.score_prediction` and `count_inference.predict`, imported and
called; this module only rebuilds the inputs those functions need (the world,
the true law, and the checkpoint's pool of fits) from the cell's own artifacts.

The 112 pilot's judging rule
----------------------------
A 112 pilot cell's judgement value is the number THIS module recomputes
after the cell completes - not the `kl` the run recorded online - and the
absolute difference against that online value is reported beside it as an
audit column, where a disagreement is an incident to be investigated rather
than a number to be averaged away. Of the 27 control cells 24 recorded no
`kl` at all, so their column is `recomputed_only` and this module's value is
the only value they have; the three death control cells (seeds 11, 17, 23)
carry a `bma.kl` from an earlier code path, so their rows are `reproduced` or
`mismatch` like the pilot's, and a mismatch between the two generations there
is an incident to investigate before any judgement is read.

Statuses
--------
reproduced       the record carried a `kl` and |recomputed - recorded| <=
                 REPRODUCTION_TOLERANCE.
mismatch         the record carried a `kl` and the difference exceeds it. The
                 row is kept and the run continues; the exit code becomes 1.
cross_check_mismatch
                 the two implementations of death's sum (`score_prediction`
                 and `kl_against_pool`) differ by more than the tolerance;
                 the number is not trusted whatever the record stored, the
                 row is kept and the exit code becomes 1.
law_unavailable  the task has no true observation law (`kl_truth_law_available`
                 false, as for a latent-variable task); nothing is scored, the
                 row stays in the table with a null `kl`, and the status is
                 neutral to the exit code. Only an input defect is `rejected`.
recomputed_only  the record carried no `kl` (an older generation, or a run
                 whose task was KL-optional and unflagged). There is no stored
                 value to agree or disagree with.
rejected         the cell does not meet a precondition of the rescore (not a
                 final after_proposal checkpoint, an evaluation set that does
                 not match the recorded `q`, a world whose parameters differ,
                 a missing fit file, a rebuilt hypothesis whose `model_id`
                 does not match the stored one).
failed           an unexpected exception; the type and message are recorded
                 and surfaced on stderr, and the remaining cells still run.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import boxinggym_truth_law
from .belief_packet import EXHAUSTIVE_COUNT_OUTCOMES
from .core_count import EVALUATION_SEEDS, CoreWorld, score_prediction, seed_for
from .count_inference import Fit, model_logweights
from .count_inference import predict as count_predict
from .count_models import Hypothesis
from .positive_inference import ScalarHypothesis
from .real_inference import RealHypothesis

SCHEMA = "frozen_kl_rescore_v1"

# The whole judgement rule of this module, as one constant: a recorded `kl` is
# reproduced when the frozen evaluator lands within this of it. It is NOT a CLI
# option - a table whose tolerance the caller could choose is not a frozen
# comparison - and it is written into the output header so a table states the
# rule it was judged by. Both paths this module scores are deterministic
# closed-form arithmetic over the same arrays the run itself used, so the
# expected difference is exactly 0.0 and this bound only absorbs a difference
# no floating-point summation order could produce.
REPRODUCTION_TOLERANCE = 1e-12

# The evaluator's own identity: the files whose bytes decide every number
# below, hashed into the header so two tables can be compared at all.
MODULE_FILES = (
    "frozen_kl_rescore.py",
    "boxinggym_truth_law.py",
    "boxinggym_core.py",
    "core_count.py",
    "count_inference.py",
    "count_models.py",
    "positive_inference.py",
    "real_inference.py",
    "belief_packet.py",
    "gaussian_core_constants.py",
)

# Every function that produces a recorded number, named where it lives. None of
# them is re-implemented in this file.
EVALUATOR_FUNCTIONS = (
    "boxinggym_truth_law.truth_law",
    "boxinggym_truth_law.kl_against_pool",
    "boxinggym_core.TaskEnvironment",
    "boxinggym_core.env_seed_injection",
    "boxinggym_core.death_truth_pmf",
    "boxinggym_core.composite_value",
    "core_count.CoreWorld.evaluation",
    "core_count.score_prediction",
    "core_count.seed_for",
    "count_inference.predict",
    "count_inference.model_logweights",
    "count_inference.Fit",
    "count_models.Hypothesis",
    "positive_inference.ScalarHypothesis",
    "real_inference.RealHypothesis",
)

# The evaluator's numeric settings, read from `boxinggym_truth_law` itself so
# the header can never drift from the module that used them.
FROZEN_CONSTANT_NAMES = (
    "KL_QUADRATURE_BASE_NODES",
    "KL_QUADRATURE_CELL_NODES",
    "KL_TRUNCATION_SIGMAS",
    "KL_POISSON_TAIL_MASS",
    "EMOTION_CELL_EDGES",
    "EMOTION_CELL_METHOD",
    "KL_EMOTION_TAIL_PANELS",
    "DUGONGS_OBSERVATION_SIGMA",
    "KL_PREDICTIVE_BLOCK_CELLS",
)

# CORE's outcome is the bounded count 0..50 and `score_prediction` sums over it
# exactly, so the label is the same one `boxinggym_truth_law.death_law` writes
# for the other exactly-summed discrete law. `score_prediction` returns no
# label of its own, which is why this one is stated here.
CORE_KL_METHOD = "discrete_exact"

# The keys a scorer may contribute to a record; a diagnostic the task's own
# law did not produce stays null, so every row carries the same key set.
KL_RESULT_KEYS = (
    "kl",
    "kl_method",
    "expected_log_score",
    "kl_truncated_mass",
    "kl_quadrature_error",
    "kl_cross_check",
)

REQUIRED_CELL_FILES = ("config.json", "scores.jsonl")

# The config.json keys the boxinggym and CORE scorers require; every other key they
# read is optional (`config.get`). A finished cell of an older generation carries
# these and is rescored on them alone.
BOXINGGYM_CONFIG_KEYS_REQUIRED = (
    "task", "seed", "cell", "profile", "q", "q_policy", "q_excluded_count", "world_seed",
    "evaluation_seed", "source_files_sha256", "rounds", "world_parameters",
)
CORE_CONFIG_KEYS_REQUIRED = (
    "seed", "cell", "profile", "q", "structure", "truth_family", "replicate",
    "evaluation_input_seed", "source_files_sha256", "rounds", "interface",
)

OK_STATUSES = ("reproduced", "recomputed_only", "law_unavailable")

# Every key a record carries, in every row of every table: a sealed table's
# schema does not depend on the bundle, the task's law or how far a cell got.
RECORD_KEYS = (
    "cell",
    "cell_path",
    "task",
    "seed",
    "cell_code",
    "profile",
    "round",
    "phase",
    "observation_count",
    "inputs",
    "evaluation_x_sha256",
    "evaluation_seed",
    "run_source_files_sha256",
    "kl_recorded",
    "kl_truth_law_available",
    "narration_calls_absorbed",
    "structure",
    "truth_family",
    "replicate",
    "evaluation_truth_match",
    "q",
    "q_policy",
    "q_excluded_count",
    "world_seed",
    "world_parameters_match",
    "world_parameters_stored",
    "world_parameters_recomputed",
    "active_ids",
    "weights_stored",
    "weights_recomputed",
    "max_abs_weight_delta",
    "components",
    "composite_rule",
    "kl",
    "kl_method",
    "expected_log_score",
    "kl_truncated_mass",
    "kl_quadrature_error",
    "kl_cross_check",
    "cross_check_delta",
    "kl_nonfinite",
    "expected_log_score_nonfinite",
    "kl_cross_check_nonfinite",
    "stored_kl_nonfinite",
    "delta_nonfinite",
    "cross_check_delta_nonfinite",
    "stored_kl",
    "delta",
    "tolerance",
    "status",
    "error_type",
    "error_message",
)


class Rejection(Exception):
    """A cell that does not meet a precondition of the rescore; the reason is the message."""


# --------------------------------------------------------------------------
# reading a cell (open-only; nothing below ever writes inside a cell)
# --------------------------------------------------------------------------


class CellReader:
    """Opens files under one cell directory and records each one's SHA-256.

    Every byte this module reads passes through here, so the `inputs` list of a
    record is the complete, ordered audit trail of what the number was computed
    from - each entry a path RELATIVE to the cell directory, never an absolute
    one, so the table's bytes do not depend on where the cell tree sits.
    """

    def __init__(self, cell):
        self.cell = Path(cell)
        self.inputs = []

    def read(self, relative, missing=None):
        """Return the bytes of a cell file; raise Rejection (with `missing` as message) if absent."""
        path = self.cell / relative
        if not path.is_file():
            raise Rejection(missing or f"file missing: {relative}")
        data = path.read_bytes()
        self.inputs.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest()})
        return data

    def json(self, relative, missing=None):
        """Parse a JSON file of the cell."""
        return json.loads(self.read(relative, missing).decode())

    def jsonl(self, relative, missing=None):
        """Parse a JSON-lines file of the cell into a list of records."""
        lines = self.read(relative, missing).decode().splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def npz(self, relative, missing=None):
        """Load an npz file of the cell into a dict of arrays."""
        with np.load(io.BytesIO(self.read(relative, missing))) as saved:
            return {name: saved[name] for name in saved.files}


def is_cell_directory(path):
    """A directory carrying the two files every bundle's cell must have."""
    return path.is_dir() and all((path / name).is_file() for name in REQUIRED_CELL_FILES)


def array_sha256(array):
    """SHA-256 of one evaluation array's own bytes, contiguous and as stored."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# --------------------------------------------------------------------------
# the evaluator's identity
# --------------------------------------------------------------------------


def module_digests():
    """Return the sha256 of each module of the frozen evaluator."""
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in MODULE_FILES
    }


def frozen_constants():
    """Return the frozen constants of the truth-law module."""
    values = {name: getattr(boxinggym_truth_law, name) for name in FROZEN_CONSTANT_NAMES}
    return {
        **{
            name: list(value) if isinstance(value, tuple) else value
            for name, value in values.items()
        },
        "REPRODUCTION_TOLERANCE": REPRODUCTION_TOLERANCE,
    }


def package_commit():
    """`(commit, commit_source)` of the checkout this package lives in."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None, "absent"
    commit = completed.stdout.strip()
    return (commit, "git_rev_parse_head") if commit else (None, "absent")


def evaluator_header():
    """Return the provenance header (commit, digests, constants) of this evaluator."""
    commit, commit_source = package_commit()
    return {
        "commit": commit,
        "commit_source": commit_source,
        "module_sha256": module_digests(),
        "functions": list(EVALUATOR_FUNCTIONS),
        "constants": frozen_constants(),
    }


# --------------------------------------------------------------------------
# one cell record
# --------------------------------------------------------------------------


def blank_record(cell_name, cell_path):
    """Every key a record can carry (RECORD_KEYS), null until a scorer fills it,
    so the table's key set does not depend on how far a cell got."""
    record = dict.fromkeys(RECORD_KEYS)
    record.update(cell=cell_name, cell_path=cell_path, inputs=[], tolerance=REPRODUCTION_TOLERANCE)
    return record


def _check_keys(record):
    """A scorer that wrote a key outside RECORD_KEYS would change the sealed schema."""
    unknown = sorted(set(record) - set(RECORD_KEYS))
    if unknown:
        raise RuntimeError(f"record carries keys outside RECORD_KEYS: {unknown}")


# The value columns a scorer may leave non-finite. A KL of +inf and an expected log
# score of -inf are honest values (a pool that puts zero mass where the
# truth has mass), so they are recorded, not averaged away or refused: the value
# column is null and the sibling `<field>_nonfinite` column says which infinity.
NONFINITE_FIELDS = ("kl", "expected_log_score", "kl_cross_check", "stored_kl", "delta", "cross_check_delta")


def _agree(left, right):
    """Whether two recomputed numbers agree: within the tolerance when both are
    finite, or the same infinity. A nan agrees with nothing."""
    if math.isfinite(left) and math.isfinite(right):
        return abs(left - right) <= REPRODUCTION_TOLERANCE
    return math.isinf(left) and math.isinf(right) and (left > 0) == (right > 0)


def _judge(record, stored_kl):
    if stored_kl is None:
        record["status"] = "recomputed_only"
    else:
        record["stored_kl"] = float(stored_kl)
        record["delta"] = float(record["kl"] - float(stored_kl))
        record["status"] = "reproduced" if _agree(record["kl"], float(stored_kl)) else "mismatch"
    if record["kl_cross_check"] is not None:
        # Two implementations of one sum must agree before either number is
        # trusted; the comparison against the stored value is still recorded
        # above, but the status is the cross-check's when it fails.
        record["cross_check_delta"] = float(record["kl"] - record["kl_cross_check"])
        if not _agree(record["kl"], record["kl_cross_check"]):
            record["status"] = "cross_check_mismatch"


def _nonfinite_label(value):
    return "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")


def _split_nonfinite(mapping, fields=None):
    """Move every non-finite float of `mapping` into its `<field>_nonfinite` sibling.

    The value column becomes null and the sibling names the infinity (or nan), so the
    table stays valid JSON under `allow_nan=False` while recording exactly what the
    evaluator computed. A record's own six fields always carry the sibling (null when
    the value is finite); a component entry carries one only for a field that needed it.
    """
    for key in (fields if fields is not None else [k for k in list(mapping) if not k.endswith("_nonfinite")]):
        value = mapping.get(key)
        if isinstance(value, float) and not math.isfinite(value):
            mapping[key] = None
            mapping[key + "_nonfinite"] = _nonfinite_label(value)
        elif fields is not None:
            mapping.setdefault(key + "_nonfinite", None)
    return mapping


def _record_nonfinite(record):
    _split_nonfinite(record, NONFINITE_FIELDS)
    for entry in record.get("components") or []:
        if isinstance(entry, dict):
            _split_nonfinite(entry)
    return record


def _carry(record, scores):
    for key in KL_RESULT_KEYS:
        if key in scores:
            record[key] = scores[key] if key == "kl_method" else float(scores[key])


def _weights(fits, stored):
    recomputed = np.exp(model_logweights(fits))
    stored_array = np.asarray(stored, dtype=float)
    if recomputed.shape != stored_array.shape:
        raise Rejection(
            f"the checkpoint recorded {stored_array.shape[0]} model weights but the pool "
            f"rebuilt from its fits has {recomputed.shape[0]}"
        )
    return recomputed.tolist(), float(np.max(np.abs(recomputed - stored_array)))


def _final_record(rows, config_rounds, round_option):
    """The last `scores.jsonl` record, once it is the final after_proposal one."""
    if not rows:
        raise Rejection("scores.jsonl carries no record")
    last = rows[-1]
    if last.get("phase") != "after_proposal" or last.get("round") != config_rounds:
        raise Rejection("last record is not the final after_proposal checkpoint")
    if round_option is not None and round_option != last["round"]:
        raise Rejection(
            f"--round {round_option} is not this cell's final checkpoint round {last['round']}"
        )
    return last


# --------------------------------------------------------------------------
# bundle boxinggym
# --------------------------------------------------------------------------


def _boxinggym():
    """The runner module, imported here rather than at module import time: it
    reaches the read-only benchmark tree, which a CORE-only rescore must not
    need on the path."""
    from . import boxinggym_core

    return boxinggym_core


def _boxinggym_hypothesis(model, interface, namespace, categorical):
    """`boxinggym_core.make`'s own rule (:2643-2665), on a stored record."""
    parameters = tuple(model["parameters"].items())
    outcome = interface["outcome_type"]
    if outcome in EXHAUSTIVE_COUNT_OUTCOMES:
        return Hypothesis(model["expression"], model["family"], parameters, namespace, categorical)
    if outcome == "real":
        return RealHypothesis(
            model["expression"],
            model["family"],
            parameters,
            namespace,
            model.get("scale_expression"),
        )
    return ScalarHypothesis(model["expression"], model["family"], parameters, namespace)


def _boxinggym_fits(reader, interface, namespace, categorical, round_number, component, model_ids):
    """The checkpoint's pool, rebuilt from `fits/` under the runner's own stem rule."""
    prefix = f"t{round_number:02d}" + ("" if component is None else f"_{component}")
    fits = []
    for model_id in model_ids:
        stem = f"fits/{prefix}_{model_id}"
        model = reader.json(f"{stem}.json", missing=f"fit file missing: {stem}.json")
        particles = reader.npz(f"{stem}.npz", missing=f"fit file missing: {stem}.npz")
        hypothesis = _boxinggym_hypothesis(model, interface, namespace, categorical)
        if hypothesis.model_id != model["model_id"]:
            raise Rejection(
                f"rebuilt hypothesis is model {hypothesis.model_id}, not the stored "
                f"{model['model_id']}"
            )
        fits.append(
            Fit(
                hypothesis,
                particles,
                model["log_evidence"],
                model["per_chain_log_evidence"],
                model["diagnostics"],
                model["fit_seconds"],
            )
        )
    return fits


def _boxinggym_scores(spec, interface, environment, law, fits, x, y, seed, observations, component):
    """This task's KL, on the branch the runner's `our_scores` itself takes."""
    if interface["outcome_type"] in EXHAUSTIVE_COUNT_OUTCOMES:
        bma_logpmf = count_predict(fits, x, interface, interface["trials"])["bma_logpmf"]
        if spec.name == "death":
            # The earlier code path that produced the stored number: CORE's own
            # scorer against death's own true PMF, with the runner's rng seed.
            # `kl_against_pool` on the SAME law and the SAME log-PMF is the
            # cross-check - two independent implementations of one sum, whose
            # agreement is recorded rather than assumed.
            truth = _boxinggym().death_truth_pmf(environment, x)
            rng = np.random.default_rng(
                seed_for("boxinggym_death_kl", spec.name, seed, observations)
            )
            scores, _ = score_prediction(bma_logpmf, y.astype(int), truth, rng)
            cross = boxinggym_truth_law.kl_against_pool(law, {"logpmf": bma_logpmf})
            return {
                "kl": scores["kl"],
                "expected_log_score": scores["expected_log_score"],
                "kl_method": cross["kl_method"],
                "kl_cross_check": cross["kl"],
            }
        return boxinggym_truth_law.kl_against_pool(law, {"logpmf": bma_logpmf}, x)
    return boxinggym_truth_law.kl_against_pool(law, fits, x, component)


def rescore_boxinggym(reader, record, round_option):
    """Re-score a BoxingGym cell into `record` and return it."""
    boxinggym_core = _boxinggym()
    config = reader.json("config.json")
    record.update(
        task=config["task"],
        seed=config["seed"],
        cell_code=config["cell"],
        profile=config["profile"],
        q=config["q"],
        q_policy=config["q_policy"],
        q_excluded_count=config["q_excluded_count"],
        world_seed=config["world_seed"],
        evaluation_seed=config["evaluation_seed"],
        # Both keys were added later; the cells of an earlier research wave predate
        # them and are still rescored, with the two columns null.
        kl_recorded=config.get("kl_recorded"),
        kl_truth_law_available=config.get("kl_truth_law_available"),
        # The cell's OWN evaluator generation, beside this module's in the
        # header: which of the two produced the stored number is the whole
        # question a frozen rescore exists to answer.
        run_source_files_sha256=config["source_files_sha256"],
    )
    last = _final_record(reader.jsonl("scores.jsonl"), config["rounds"], round_option)
    record.update(
        round=last["round"], phase=last["phase"], observation_count=last["observation_count"]
    )

    # The evaluation set as the run left it. `exclude_observed` rewrites both
    # config.json and this npz every time it drops a point
    # (boxinggym_core.py:3386-3388), so the stored array IS the final Q and the
    # recorded counts must account for it exactly.
    arrays = reader.npz("evaluation_scoring_only.npz")
    x, y = arrays["x"], arrays["y"]
    record["evaluation_x_sha256"] = array_sha256(x)
    expected = config["q"] - config["q_excluded_count"]
    if len(x) != expected:
        raise Rejection(
            f"q mismatch: the stored evaluation set has {len(x)} points, "
            f"q - q_excluded_count is {expected}"
        )

    spec = boxinggym_core.TASKS[config["task"]]
    world_seed = seed_for("boxinggym_world", spec.name, config["seed"])
    if world_seed != config["world_seed"]:
        raise Rejection(
            f"world_seed mismatch: the task and seed give {world_seed}, "
            f"config.json records {config['world_seed']}"
        )
    with contextlib.ExitStack() as stack:
        # The runner's own order (boxinggym_core.run): the narration stub is
        # installed BEFORE the environment is constructed, because moral_machines
        # and emotion call `openai.OpenAI()` in `__init__`; then the seed injection,
        # then the environment. It stays installed for the whole rescore, as it does
        # for the whole run, so no key and no network are needed here either.
        narration_calls = (
            stack.enter_context(
                boxinggym_core.openai_narration_stub(spec.openai_narration_module)
            )
            if spec.openai_narration
            else None
        )
        stack.enter_context(boxinggym_core.env_seed_injection(world_seed))
        environment = boxinggym_core.TaskEnvironment(spec, world_seed)
        parameters = json.loads(json.dumps(environment.world_parameters()))
        record["world_parameters_match"] = parameters == config["world_parameters"]
        if not record["world_parameters_match"]:
            record["world_parameters_stored"] = config["world_parameters"]
            record["world_parameters_recomputed"] = parameters
            raise Rejection("world_parameters mismatch")

        law = boxinggym_truth_law.truth_law(spec, environment, x)
        if law is None:
            # A task without a true observation law (`kl_truth_law_available`
            # false) is not an input defect: the row stays, nothing is scored.
            record["status"] = "law_unavailable"
            return

        interface = spec.interface()
        namespace = tuple(item["name"] for item in interface["inputs"]) + tuple(
            interface["derived_inputs"]
        )
        categorical = spec.categorical_levels()
        components = tuple(spec.components) or ("y",)

        if len(components) == 1:
            fits = _boxinggym_fits(
                reader, interface, namespace, categorical, last["round"], None, last["active_ids"]
            )
            recomputed, delta = _weights(fits, last["model_weights"])
            record.update(
                active_ids=list(last["active_ids"]),
                weights_stored=list(last["model_weights"]),
                weights_recomputed=recomputed,
                max_abs_weight_delta=delta,
            )
            scores = _boxinggym_scores(
                spec,
                interface,
                environment,
                law,
                fits,
                x,
                y,
                config["seed"],
                last["observation_count"],
                None,
            )
        else:
            # One law, one Q, one component index per declared output: the runner
            # scores each component's own pool separately and averages the numbers
            # with `composite_value`, and so does this.
            per_component, entries, deltas = [], [], []
            for entry in last["components"]:
                index = components.index(entry["component"])
                fits = _boxinggym_fits(
                    reader,
                    interface,
                    namespace,
                    categorical,
                    last["round"],
                    entry["component"],
                    entry["active_ids"],
                )
                recomputed, delta = _weights(fits, entry["model_weights"])
                scores = _boxinggym_scores(
                    spec,
                    interface,
                    environment,
                    law,
                    fits,
                    x,
                    y,
                    config["seed"],
                    last["observation_count"],
                    index,
                )
                per_component.append(scores)
                deltas.append(delta)
                component_record = {
                    "component": entry["component"],
                    "component_index": index,
                    "active_ids": list(entry["active_ids"]),
                    "weights_stored": list(entry["model_weights"]),
                    "weights_recomputed": recomputed,
                    "max_abs_weight_delta": delta,
                    "stored_kl": entry["bma"].get("kl"),
                }
                _carry(component_record, scores)
                entries.append(component_record)
            record["components"] = entries
            record["composite_rule"] = boxinggym_core.COMPOSITE_SCORE_RULE
            record["max_abs_weight_delta"] = max(deltas)
            scores = {
                key: boxinggym_core.composite_value([score[key] for score in per_component])
                for key in per_component[0]
            }

        _carry(record, scores)
        _judge(record, last["bma"].get("kl"))
        record["narration_calls_absorbed"] = (
            None if narration_calls is None else len(narration_calls)
        )


# --------------------------------------------------------------------------
# bundle core
# --------------------------------------------------------------------------


def rescore_core(reader, record, round_option):
    """Re-score a CORE cell into `record` and return it."""
    config = reader.json("config.json")
    record.update(
        task="core",
        seed=config["seed"],
        cell_code=config["cell"],
        profile=config["profile"],
        q=config["q"],
        structure=config["structure"],
        truth_family=config["truth_family"],
        replicate=config["replicate"],
        evaluation_seed=config["evaluation_input_seed"],
        run_source_files_sha256=config["source_files_sha256"],
    )
    last = _final_record(reader.jsonl("scores.jsonl"), config["rounds"], round_option)
    record.update(
        round=last["round"], phase=last["phase"], observation_count=last["observation_count"]
    )
    if config["structure"].startswith("gauss_"):
        return rescore_gaussian_core(reader, record, config, last)

    arrays = reader.npz("evaluation_scoring_only.npz")
    u, y, pmf = arrays["u"], arrays["y"], arrays["pmf"]
    record["evaluation_x_sha256"] = array_sha256(u)
    world = CoreWorld(
        config["seed"], config["structure"], config["truth_family"], config["replicate"]
    )
    _, _, truth = world.evaluation(config["q"], EVALUATION_SEEDS[config["profile"]])
    record["evaluation_truth_match"] = bool(np.array_equal(truth, pmf))
    if not record["evaluation_truth_match"]:
        raise Rejection("the rebuilt world's true PMF is not the one the cell stored")

    interface = config["interface"]
    fits = []
    for model_id in last["active_ids"]:
        stem = f"fits/t{last['round']:02d}_{model_id}"
        model = reader.json(f"{stem}.json", missing=f"fit file missing: {stem}.json")
        particles = reader.npz(f"{stem}.npz", missing=f"fit file missing: {stem}.npz")
        hypothesis = Hypothesis(
            model["expression"], model["family"], tuple(model["parameters"].items())
        )
        if hypothesis.model_id != model["model_id"]:
            raise Rejection(
                f"rebuilt hypothesis is model {hypothesis.model_id}, not the stored "
                f"{model['model_id']}"
            )
        fits.append(
            Fit(
                hypothesis,
                particles,
                model["log_evidence"],
                model["per_chain_log_evidence"],
                model["diagnostics"],
                model["fit_seconds"],
            )
        )
    recomputed, delta = _weights(fits, last["model_weights"])
    record.update(
        active_ids=list(last["active_ids"]),
        weights_stored=list(last["model_weights"]),
        weights_recomputed=recomputed,
        max_abs_weight_delta=delta,
    )

    bma_logpmf = count_predict(fits, u, interface, interface["trials"])["bma_logpmf"]
    rng = np.random.default_rng(
        seed_for("predictive_samples", config["seed"], last["round"], last["phase"], "bma")
    )
    scores, _ = score_prediction(bma_logpmf, y.astype(int), pmf, rng)
    record["kl"] = float(scores["kl"])
    record["kl_method"] = CORE_KL_METHOD
    record["expected_log_score"] = float(scores["expected_log_score"])
    _judge(record, last["bma"].get("kl"))


def gaussian_world_from_config(config):
    """Reconstruct one truth sigma, then require exact agreement with its record."""
    from . import gaussian_core_constants
    from .belief_packet import core_real_interface

    if config.get("outcome_type") != "real" or config["interface"] != core_real_interface():
        raise Rejection("Gaussian CORE requires the real identity-link interface")
    selection_hash = hashlib.sha256(Path(gaussian_core_constants.__file__).read_bytes()).hexdigest()
    if config.get("constant_selection_sha256") != selection_hash:
        raise Rejection("Gaussian constant-selection module differs from the run")
    for name in ("GAUSS_DEN_C", "GAUSS_POW_E", "GAUSS_CALL_K"):
        if config.get(name.lower()) != getattr(gaussian_core_constants, name):
            raise Rejection("Gaussian frozen literal differs from the run: " + name.lower())
    noise = config.get("noise", {})
    try:
        grid = np.asarray(json.loads(noise["grid_canonical_json"]), dtype=float)
        world = CoreWorld(config["seed"], config["structure"], config["truth_family"],
                          config["replicate"], design_grid=grid)
    except (KeyError, TypeError, ValueError) as exc:
        raise Rejection(f"Gaussian noise/grid record is invalid: {exc}") from exc
    if noise != world.noise_record():
        raise Rejection("Gaussian truth sigma/grid differs from the recorded rule")
    return world


def rescore_gaussian_core(reader, record, config, last):
    """The normal truth law and stored scalar posteriors, with no new fitting."""
    world = gaussian_world_from_config(config)
    u, y, truth = world.evaluation(config["q"], EVALUATION_SEEDS[config["profile"]])
    arrays = reader.npz("evaluation_scoring_only.npz")
    if not all(np.array_equal(arrays.get(key), value)
               for key, value in (("u", u), ("y", y), ("mean", truth[0]), ("sigma", truth[1]))):
        raise Rejection("Gaussian evaluation inputs/outcomes/truth differ from the run")
    record["evaluation_x_sha256"] = array_sha256(u)
    record["evaluation_truth_match"] = True
    namespace = tuple(item["name"] for item in config["interface"]["inputs"]) + tuple(
        config["interface"]["derived_inputs"])
    fits = _boxinggym_fits(reader, config["interface"], namespace, (),
                          last["round"], None, last["active_ids"])
    weights, delta = _weights(fits, last["model_weights"])
    record.update(active_ids=list(last["active_ids"]), weights_stored=last["model_weights"],
                  weights_recomputed=weights, max_abs_weight_delta=delta)
    _carry(record, boxinggym_truth_law.kl_against_pool(
        boxinggym_truth_law.normal_law(*truth), fits, u))
    record["expected_log_score"] = -float(0.5 * np.log(2 * np.pi * np.e * world.sigma**2)) - record["kl"]
    _judge(record, last["bma"].get("kl"))


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------


def rescore_cell(cell, cell_path, bundle, round_option):
    """One record. A precondition failure or an exception becomes a status, never a crash."""
    reader = CellReader(cell)
    record = blank_record(Path(cell).name, cell_path)
    scorer = rescore_boxinggym if bundle == "boxinggym" else rescore_core
    try:
        scorer(reader, record, round_option)
        _record_nonfinite(record)
        _check_keys(record)
    except Rejection as exc:
        record.update(status="rejected", error_type="rejected", error_message=str(exc))
    except Exception as exc:  # noqa: BLE001 - one bad cell must not lose the table
        record.update(status="failed", error_type=type(exc).__name__, error_message=str(exc))
        print(f"FE2 {Path(cell).name}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    record["inputs"] = reader.inputs
    return record


def rescore(cells, bundle, round_option, cells_root=None):
    """The whole table: header plus one record per cell, in the given order.

    The header's `round` is the `--round` the caller STATED (null when it
    stated none, which is the default "each cell's own final round"); the round
    actually scored is each record's own `round`, because a table may hold
    cells of different lengths.
    """
    return {
        "schema": SCHEMA,
        "evaluator": evaluator_header(),
        "bundle": bundle,
        "round": round_option,
        "cells_root": None if cells_root is None else Path(cells_root).name,
        "cells": [rescore_cell(cell, path, bundle, round_option) for cell, path in cells],
    }


def _assert_finite(value, path="table"):
    """Refuse to render a non-finite number anywhere: every one a scorer can produce
    was moved into its `_nonfinite` column, so a leftover is a programming error."""
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number left in the table at {path}")


def render(table):
    """Serialize the table to canonical JSON text; rejects non-finite values."""
    _assert_finite(table)
    return json.dumps(table, sort_keys=True, allow_nan=False, indent=2) + "\n"


def write_table(table, output):
    """Write the rendered table to the output path."""
    output = Path(output)
    body = render(table)
    output.write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n")
    return digest


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _resolve_cells(args, parser):
    """`(directory, recorded path)` pairs, in `--cell` order or sorted by name."""
    if args.cells_root is None:
        return [(Path(cell).resolve(), Path(cell).name) for cell in args.cell], None
    root = Path(args.cells_root).resolve()
    if not root.is_dir():
        parser.error(f"--cells-root is not a directory: {args.cells_root}")
    found = sorted(child for child in root.iterdir() if is_cell_directory(child))
    if not found:
        parser.error(f"--cells-root holds no cell directory: {args.cells_root}")
    return [(child, child.name) for child in found], root


def main(argv=None):
    """Command-line entry point for the frozen re-scoring."""
    parser = argparse.ArgumentParser(description="Re-score completed cells with one frozen KL evaluator.")
    parser.add_argument("--bundle", choices=("boxinggym", "core"), required=True)
    parser.add_argument("--cell", action="append", default=[], metavar="DIR")
    parser.add_argument("--cells-root", default=None, metavar="ROOT")
    parser.add_argument("--output", required=True, metavar="PATH.json")
    parser.add_argument("--round", type=int, default=None, metavar="N")
    args = parser.parse_args(argv)
    if bool(args.cell) == bool(args.cells_root):
        parser.error("state either --cell (repeatable) or --cells-root, not both and not neither")

    cells, root = _resolve_cells(args, parser)
    output = Path(args.output).resolve()
    for cell, _ in cells:
        if output == cell or cell in output.parents:
            # A cell directory is read-only for the whole life of this module;
            # writing the table into one would make it its own input.
            parser.error(f"--output is inside the cell directory {cell.name}; write it elsewhere")

    table = rescore(cells, args.bundle, args.round, root)
    write_table(table, output)
    statuses = [record["status"] for record in table["cells"]]
    print(
        "FE2 " + " ".join(f"{name}={statuses.count(name)}" for name in sorted(set(statuses))),
        file=sys.stderr,
        flush=True,
    )
    return 0 if all(status in OK_STATUSES for status in statuses) else 1


if __name__ == "__main__":
    sys.exit(main())
