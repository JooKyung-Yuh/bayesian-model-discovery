"""Frozen ICL prompts, strict response parsing, and separate-target scoring."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bayes_model_discovery.experimental.belief_packet import round_significant
from bayes_model_discovery.experimental.chembench_core import log_error_metrics
from bayes_model_discovery.experimental.core_count import seed_for

ROOT = Path(__file__).resolve().parent
OUTCOME_POLICIES = {
    "bounded_count": None, "binary": None, "unbounded_count": None,
    "positive": {"phrase": "a positive outcome y", "validation": "finite_positive"},
    "real": None, "trajectory": None,
}
MASKED_NAMES = tuple(f"X{i}" for i in range(1, 8))
VISIBLE_NAMES = ("C_A", "C_I", "C_B", "C_P", "Enz", "T", "pH")
BOUNDS = ((0.01, 100.0), (0.0, 50.0), (0.01, 100.0), (0.0, 20.0),
          (0.01, 10.0), (278.0, 368.0), (4.0, 10.0))
TIMES = (3, 5, 10, 15, 20, 25, 40, 63)
OFFICIAL_TIMES = (10, 25, 63)
DOMAIN_WORDS = re.compile(
    r"enzyme|substrate|michaelis|inhibit|ping[- ]?pong|kinetic|catalys", re.IGNORECASE,
)
FORBIDDEN_WORDS = ("C_A", "C_B", "C_I", "C_P", "Enz", "pH", "conversion", "r0",
                   "enzyme", "substrate", "inhibitor", "product", "rate", "kinetic",
                   "catalys", "Michaelis", "temperature", "log link", "pool", "archive",
                   "residual", "expression", "coefficient", "family", "X1z", "lognormal")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def number(value):
    return json.dumps(round_significant(float(value)), allow_nan=False)


def prompt_parts(kind, visible=False):
    filename = "prediction_visible" if visible else kind
    text = (ROOT / "prompts" / f"{filename}.txt").read_text()
    system, rest = text.removeprefix("[system]\n").split("\n\n[user]\n", 1)
    user, repair = rest.split("\n\n[appended once, only after a validation failure]\n", 1)
    return system, user, repair.rstrip("\n")


@dataclass(frozen=True)
class Observation:
    obs_id: int
    x: tuple[float, ...]
    y: float


def observations_from_rows(rows, declared_names):
    """Discard auxiliary outputs at the boundary, before any prompt construction."""
    observations = []
    for row in rows:
        if set(row["x"]) != set(declared_names):
            raise ValueError("observation inputs disagree with interface declaration")
        index = row.get("obs_id", row.get("observation_index", 0) - 1)
        x = tuple(float(row["x"][name]) for name in declared_names)
        y = float(row["y"])
        if index < 0 or not all(math.isfinite(v) for v in (*x, y)) or y <= 0:
            raise ValueError("invalid observation")
        observations.append(Observation(index, x, y))
    if len({o.obs_id for o in observations}) != len(observations):
        raise ValueError("duplicate observation id")
    return observations


def observation_table(observations):
    return "\n".join(
        ",".join([str(o.obs_id + 1), *(number(v) for v in o.x), number(o.y)])
        for o in sorted(observations, key=lambda o: (*o.x, o.obs_id))
    )


def render(kind, observations, *, queries=None, visible=False, outcome_type="positive",
           names=MASKED_NAMES, bounds=BOUNDS):
    """Selection accepts observations/ranges only; no general context dictionary."""
    if outcome_type not in OUTCOME_POLICIES or OUTCOME_POLICIES[outcome_type] is None:
        raise NotImplementedError(f"unregistered outcome policy: {outcome_type}")
    expected_names = VISIBLE_NAMES if visible else MASKED_NAMES
    if tuple(names) != expected_names or tuple(map(tuple, bounds)) != BOUNDS:
        raise ValueError("interface is outside the frozen ICL prompt registration")
    if kind not in ("prediction", "selection") or (kind == "selection" and visible):
        raise ValueError("unregistered prompt variant")
    if kind == "selection" and queries is not None:
        raise ValueError("selection input allowlist excludes evaluation queries")
    system, user, _ = prompt_parts(kind, visible)
    if kind == "selection":
        _, prediction, _ = prompt_parts("prediction")
        prefix = prediction.split("\n\nNew inputs", 1)[0]
        user = user.replace("(the same first two lines and observation table as 5.1)", prefix)
    user = user.replace("a positive outcome y", OUTCOME_POLICIES[outcome_type]["phrase"])
    user = user.replace("<n>", str(len(observations))).replace(
        "<one row per experiment>", observation_table(observations))
    mapping = []
    if kind == "prediction":
        if queries is None or len(queries) == 0:
            raise ValueError("prediction requires query coordinates")
        for index, values in enumerate(queries):
            if len(values) != 7 or not all(math.isfinite(float(v)) for v in values):
                raise ValueError("invalid query coordinates")
            mapping.append({"id": f"q{index + 1}", "coordinates": list(map(float, values))})
        table = "\n".join(",".join([row["id"], *(number(v) for v in row["coordinates"])])
                          for row in mapping)
        user = user.replace("<one row per new input, q = q1..q<Q>>", table)
        user = user.replace("<Q>", str(len(queries)))
    return {"system": system, "messages": [{"role": "user", "content": user}]}, mapping


def repair_request(original, first_text, error, kind, query_count=0):
    _, _, repair = prompt_parts(kind)
    repair = repair.replace("<error>", error).replace("<Q>", str(query_count))
    return {"system": original["system"], "messages": [
        *original["messages"], {"role": "assistant", "content": first_text},
        {"role": "user", "content": repair},
    ]}


class Pairs(list):
    """Keep duplicate JSON keys visible rather than silently choosing a value."""


def parse_object(text):
    match = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    parsed = json.loads(text, object_pairs_hook=Pairs)
    if not isinstance(parsed, Pairs):
        raise TypeError("root must be an object")
    return parsed


def finite_number(value, positive=False):
    try:
        return (type(value) in (int, float) and math.isfinite(value)
                and (not positive or value > 0))
    except OverflowError:
        return False


@dataclass
class Validation:
    values: dict
    errors: list[str]

    @property
    def valid(self):
        return not self.errors


def validate(text, kind, *, query_ids=(), stop_reason="end_turn", names=MASKED_NAMES, bounds=BOUNDS):
    if stop_reason == "max_tokens":
        return Validation({}, ["response stopped at max_tokens"])
    if isinstance(text, str) and not text.strip():
        return Validation({}, ["empty text"])
    try:
        pairs = parse_object(text)
    except (ValueError, TypeError, OverflowError, RecursionError):
        return Validation({}, ["invalid JSON object"])
    field = "predictions" if kind == "prediction" else "x"
    if len(pairs) != 1 or pairs[0][0] != field or not isinstance(pairs[0][1], Pairs):
        return Validation({}, [f"root must contain only one {field} object"])
    entries = pairs[0][1]
    expected = tuple(query_ids) if kind == "prediction" else tuple(names)
    found = {}
    errors = []
    for key, value in entries:
        found.setdefault(key, []).append(value)
    values = {}
    for key in expected:
        items = found.get(key, [])
        if len(items) != 1:
            errors.append(f"{key}: expected exactly once")
            continue
        value = items[0]
        if not finite_number(value, positive=kind == "prediction"):
            errors.append(f"{key}: requires a finite {'positive ' if kind == 'prediction' else ''}JSON number")
        elif kind == "selection" and not bounds[names.index(key)][0] <= value <= bounds[names.index(key)][1]:
            errors.append(f"{key}: outside declared bounds")
        else:
            values[key] = float(value)
    if set(found) - set(expected):
        errors.append("unexpected keys")
    if kind == "selection" and errors:
        values = {}
    return Validation(values, errors)


def prediction_result(validations, query_ids, observations):
    accepted = {}
    for validation in validations:
        for key, value in validation.values.items():
            accepted.setdefault(key, value)
    median = float(np.median([o.y for o in observations]))
    fallback = [key for key in query_ids if key not in accepted]
    return {"values": [accepted.get(key, median) for key in query_ids],
            "valid": [key in accepted for key in query_ids], "fallback_ids": fallback,
            "status": "prediction_fallback" if fallback else "valid", "fallback_value": median}


def selection_fallback(unit_id, replicate, round_index, bounds=BOUNDS):
    rng = np.random.default_rng(seed_for("icl_selection_fallback", unit_id, replicate, round_index))
    return [float(rng.uniform(lo, hi)) for lo, hi in bounds]


def timepoint_status(batches):
    requested = sum(len(b["values"]) for b in batches)
    fallback = sum(len(b["fallback_ids"]) for b in batches)
    failed_batches = sum(bool(b["fallback_ids"]) for b in batches)
    ratio = fallback / requested
    return {"status": "invalid_timepoint" if ratio > .2 else "valid",
            "requested_query_count": requested, "fallback_query_count": fallback,
            "fallback_query_ratio": ratio, "fallback_batch_count": failed_batches,
            "requested_batch_count": len(batches), "fallback_batch_ratio": failed_batches / len(batches)}


def metrics(predictions, truth):
    if len(truth) == 0:
        return {"status": "undefined", "nmsle": None, "rmsle": None, "count": 0}
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(predictions, dtype=float)
    if not np.all(np.isfinite(truth)) or np.any(truth < 0) or not np.all(np.isfinite(prediction)) or np.any(prediction <= 0):
        raise ValueError("score inputs must be finite with positive predictions and nonnegative targets")
    log_truth = np.log1p(truth)
    variance = float(np.var(log_truth))
    if variance == 0:
        return {"status": "undefined", "nmsle": None,
                "rmsle": float(np.sqrt(np.mean((np.log1p(prediction) - log_truth) ** 2))), "count": len(truth)}
    result = log_error_metrics(prediction, log_truth, variance)
    return {"status": "defined", "nmsle": result["nmsle"], "rmsle": result["rmsle"], "count": len(truth)}


def score_timepoint(batches, references, observations):
    prediction = np.asarray([v for b in batches for v in b["values"]])
    valid = np.asarray([v for b in batches for v in b["valid"]], dtype=bool)
    constant = np.full(len(prediction), float(np.median([o.y for o in observations])))
    targets = {}
    for name in ("primary_rate", "noiseless_observation"):
        truth = np.asarray(references[name])
        if len(truth) != len(prediction):
            raise ValueError("query/reference length mismatch")
        masks = {"below_0.01": truth < .01, "0.01_to_1": (truth >= .01) & (truth <= 1), "above_1": truth > 1}
        targets[name] = {"operational": metrics(prediction, truth),
                         "conditional_valid": metrics(prediction[valid], truth[valid]),
                         "constant": metrics(constant, truth),
                         "magnitude_bins": {key: metrics(prediction[mask], truth[mask]) for key, mask in masks.items()}}
    return {**timepoint_status(batches), "targets": targets}


def cell_direction(icl, ours):
    pairs = [(icl.get(t), ours.get(t)) for t in ("primary_rate", "noiseless_observation")]
    if any(a is None or b is None or not math.isfinite(a) or not math.isfinite(b) for a, b in pairs):
        return "undefined"
    signs = [np.sign(a - b) for a, b in pairs]
    if all(s < 0 for s in signs):
        return "ICL side"
    if all(s > 0 for s in signs):
        return "our side"
    return "target sensitive" if -1 in signs and 1 in signs else "tie"


def fixed_direction(paired_cells):
    if len(paired_cells) != 3 or any(c["status"] != "valid" for c in paired_cells):
        return "no directional statement"
    directions = [cell_direction(c["icl"], c["ours"]) for c in paired_cells]
    return directions[0] if len(set(directions)) == 1 and directions[0] in ("ICL side", "our side") else "no directional statement"


def closed_direction(icl_cells, our_cells):
    if len(icl_cells) != 3 or len(our_cells) != 3 or any(c["status"] != "valid" for c in [*icl_cells, *our_cells]):
        return "no directional statement"
    directions = []
    for target in ("primary_rate", "noiseless_observation"):
        a, b = [c.get(target) for c in icl_cells], [c.get(target) for c in our_cells]
        if any(v is None or not math.isfinite(v) for v in a + b):
            return "no directional statement"
        directions.append(-1 if max(a) < min(b) else 1 if min(a) > max(b) else 0)
    return "ICL side" if directions == [-1, -1] else "our side" if directions == [1, 1] else "no directional statement"


def forbidden_words(text, visible=False):
    allowed = set(VISIBLE_NAMES) if visible else set()
    return [word for word in FORBIDDEN_WORDS if word not in allowed and re.search(
        r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])", text, flags=re.IGNORECASE)]
