"""ICL (+LLM design) for BoxingGym: frozen prompts, masked interfaces, strict parsing, scoring.

The model never sees a task name, a native input name or a domain word: every task is "an unknown
process" over X1..Xk. The scoring is the engine's (nMSE against the noise-free mean on Q, the
Brier score against the held-out 0/1 outcomes for a binary task)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bayes_model_discovery.experimental.belief_packet import round_significant
from bayes_model_discovery.experimental.core_count import seed_for
from tools.icl_baseline.protocol import Pairs, finite_number, parse_object

ROOT = Path(__file__).resolve().parent
TASKS = ("death", "dugongs", "peregrines", "hyperbolic", "survival")
#: Observation counts at which the forecaster is scored: MDA budgets 0, 5, 10, 20 after the two
#: random2 start designs (x = 2 + b), the checkpoints of the Ours v4 table.
CHECKPOINTS = (2, 7, 12, 22)
ROUNDS = 20
QUERY_BATCH = 128
#: What the model is told about the outcome and how a prediction is checked, by the engine's
#: outcome type. `support` is the closed interval a prediction must lie in (None = open side).
OUTCOME_RULES = {
    "bounded_count": {"phrase": "a count outcome y between 0 and <trials>", "estimand": "the expected value of y",
                      "rule": "Each value must be a finite JSON number between 0 and <trials>, bounds included.",
                      "low": 0.0, "high": "trials", "fallback": "median"},
    "unbounded_count": {"phrase": "a nonnegative count outcome y", "estimand": "the expected value of y",
                        "rule": "Each value must be a finite nonnegative JSON number.",
                        "low": 0.0, "high": None, "fallback": "median"},
    "positive": {"phrase": "a positive outcome y", "estimand": "y",
                 "rule": "Each value must be a positive finite JSON number.",
                 "low": "open_zero", "high": None, "fallback": "median"},
    "binary": {"phrase": "a binary outcome y (0 or 1)", "estimand": "the probability that y = 1",
               "rule": "Each value must be a finite JSON number between 0 and 1, bounds included.",
               "low": 0.0, "high": 1.0, "fallback": "mean"},
}
#: Domain stems that must not reach the model, by task (matched at a word start, so "infect"
#: covers "infection"); the task's native input names are matched as whole words.
FORBIDDEN_WORDS = {
    "death": ("death", "infect", "disease", "epidemic", "mortal", "time"),
    "dugongs": ("dugong", "age", "length", "sea cow", "growth"),
    "peregrines": ("peregrine", "falcon", "bird", "population", "year"),
    "hyperbolic": ("hyperbolic", "discount", "reward", "delay", "money", "dollar", "day"),
    "survival": ("survival", "patient", "cancer", "metasta", "tumor", "time", "death", "alive"),
}
COMMON_FORBIDDEN = ("BoxingGym", "boxing", "benchmark", "PyMC", "posterior", "prior", "goal")
NATIVE_NAMES = {"death": ("t",), "dugongs": ("age",), "peregrines": ("tt",), "hyperbolic": ("iR", "dR", "D"),
                "survival": ("time", "metastasized")}


def forbidden_pattern(task):
    """One case-insensitive pattern: the task's stems at a word start, its native input names
    and the common words as whole words."""
    stems = [r"\b" + re.escape(word) for word in FORBIDDEN_WORDS[task]]
    words = [r"\b" + re.escape(word) + r"\b" for word in (*NATIVE_NAMES[task], *COMMON_FORBIDDEN)]
    return re.compile("|".join(stems + words), re.IGNORECASE)


@dataclass(frozen=True)
class Interface:
    """One task's model-facing interface: masked names in declared order, ranges, outcome rule."""

    task: str
    native_names: tuple
    bounds: tuple
    outcome_type: str
    trials: int | None
    integer_inputs: tuple

    @property
    def names(self):
        return tuple(f"X{i}" for i in range(1, len(self.native_names) + 1))

    @property
    def input_name_map(self):
        return dict(zip(self.names, self.native_names))

    @property
    def rule(self):
        return OUTCOME_RULES[self.outcome_type]

    def text(self, key):
        value = self.rule[key]
        return value.replace("<trials>", str(self.trials)) if isinstance(value, str) else value

    def support(self):
        low = self.rule["low"]
        high = self.rule["high"]
        high = float(self.trials) if high == "trials" else high
        return low, high

    def check(self, value):
        """A finite JSON number inside the outcome's support."""
        if not finite_number(value):
            return "requires a finite JSON number"
        low, high = self.support()
        if low == "open_zero":
            if value <= 0:
                return "requires a positive number"
        elif value < low or (high is not None and value > high):
            return f"outside {low} to {high if high is not None else 'infinity'}"
        return None


def interface_for(spec):
    """The masked interface of one engine TaskSpec (names in declared order, declared ranges)."""
    return Interface(spec.name, tuple(spec.input_names),
                     tuple(tuple(float(v) for v in item["range"]) for item in spec.inputs),
                     spec.outcome_type, spec.trials, tuple(spec.integer_inputs))


@dataclass(frozen=True)
class Observation:
    obs_id: int
    x: tuple
    y: float


def number(value):
    return json.dumps(round_significant(float(value)), allow_nan=False)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def prompt_parts(kind):
    text = (ROOT / "prompts" / f"{kind}.txt").read_text()
    system, rest = text.removeprefix("[system]\n").split("\n\n[user]\n", 1)
    user, repair = rest.split("\n\n[appended once, only after a validation failure]\n", 1)
    return system, user, repair.rstrip("\n")


def observation_table(observations):
    return "\n".join(",".join([str(o.obs_id + 1), *(number(v) for v in o.x), number(o.y)])
                     for o in sorted(observations, key=lambda o: (*o.x, o.obs_id)))


def _fill(text, interface, observations):
    names = interface.names
    ranges = ", ".join(f"{name} [{number(lo)}, {number(hi)}]" for name, (lo, hi) in zip(names, interface.bounds))
    header = ",".join(names)
    naming = (f"{names[0]} is a variable name" if len(names) == 1
              else f"{names[0]} through {names[-1]} are variable names")
    return (text.replace("<input names> are variable names", naming)
                .replace("<input names>", (names[0] if len(names) == 1 else f"{names[0]} through {names[-1]}"))
                .replace("<k>", str(len(names))).replace("<input word>", "input" if len(names) == 1 else "inputs")
                .replace("<outcome phrase>", interface.text("phrase")).replace("<input ranges>", ranges)
                .replace("<input header>", header).replace("<n>", str(len(observations)))
                .replace("<one row per experiment>", observation_table(observations)))


def render_prediction(interface, observations, queries):
    """The forecast request for `queries` (rows in declared input order); returns (prompt, mapping)."""
    queries = [tuple(float(v) for v in row) for row in queries]
    if not queries or any(len(row) != len(interface.names) or not all(math.isfinite(v) for v in row) for row in queries):
        raise ValueError("invalid query coordinates")
    system, user, _ = prompt_parts("prediction")
    mapping = [{"id": f"q{index + 1}", "coordinates": list(row)} for index, row in enumerate(queries)]
    table = "\n".join(",".join([row["id"], *(number(v) for v in row["coordinates"])]) for row in mapping)
    # The table placeholder carries a <Q> of its own, so it is filled before <Q> is.
    user = _fill(user, interface, observations).replace("<one row per new input, q = q1..q<Q>>", table)
    user = user.replace("<Q>", str(len(queries)))
    user = user.replace("<estimand>", interface.text("estimand")).replace("<value rule>", interface.text("rule"))
    system = _fill(system, interface, observations)
    return {"system": system, "messages": [{"role": "user", "content": user}]}, mapping


def render_selection(interface, observations, candidates, *, repeats_allowed):
    """The design request over the candidate rows (declared input order); returns (prompt, mapping)."""
    if not len(candidates):
        raise ValueError("no candidate design")
    system, user, _ = prompt_parts("selection")
    mapping = [{"id": f"c{index + 1}", "coordinates": [float(v) for v in row]} for index, row in enumerate(candidates)]
    table = "\n".join(",".join([row["id"], *(number(v) for v in row["coordinates"])]) for row in mapping)
    user = _fill(user, interface, observations).replace("<one row per candidate, c = c1..c<C>>", table)
    user = user.replace("<C>", str(len(mapping)))
    user = user.replace("<repeat rule>", " You may repeat an earlier experiment." if repeats_allowed
                        else " Experiments already run are not listed.")
    system = _fill(system, interface, observations)
    return {"system": system, "messages": [{"role": "user", "content": user}]}, mapping


def repair_request(original, first_text, error, kind, interface, count):
    _, _, repair = prompt_parts(kind)
    repair = repair.replace("<error>", error).replace("<Q>", str(count)).replace("<value rule>", interface.text("rule"))
    return {"system": original["system"], "messages": [*original["messages"],
                                                       {"role": "assistant", "content": first_text},
                                                       {"role": "user", "content": repair}]}


@dataclass
class Validation:
    values: dict
    errors: list

    @property
    def valid(self):
        return not self.errors


def validate_prediction(text, interface, query_ids, stop_reason="end_turn"):
    if stop_reason == "max_tokens":
        return Validation({}, ["response stopped at max_tokens"])
    if isinstance(text, str) and not text.strip():
        return Validation({}, ["empty text"])
    try:
        pairs = parse_object(text)
    except (ValueError, TypeError, OverflowError, RecursionError):
        return Validation({}, ["invalid JSON object"])
    if len(pairs) != 1 or pairs[0][0] != "predictions" or not isinstance(pairs[0][1], Pairs):
        return Validation({}, ["root must contain only one predictions object"])
    found = {}
    for key, value in pairs[0][1]:
        found.setdefault(key, []).append(value)
    values, errors = {}, []
    for key in query_ids:
        items = found.get(key, [])
        if len(items) != 1:
            errors.append(f"{key}: expected exactly once")
            continue
        problem = interface.check(items[0])
        if problem:
            errors.append(f"{key}: {problem}")
        else:
            values[key] = float(items[0])
    if set(found) - set(query_ids):
        errors.append("unexpected keys")
    return Validation(values, errors)


def validate_selection(text, candidate_ids, stop_reason="end_turn"):
    if stop_reason == "max_tokens":
        return Validation({}, ["response stopped at max_tokens"])
    if isinstance(text, str) and not text.strip():
        return Validation({}, ["empty text"])
    try:
        pairs = parse_object(text)
    except (ValueError, TypeError, OverflowError, RecursionError):
        return Validation({}, ["invalid JSON object"])
    if len(pairs) != 1 or pairs[0][0] != "choice":
        return Validation({}, ["root must contain only one choice string"])
    choice = pairs[0][1]
    if not isinstance(choice, str) or choice not in candidate_ids:
        return Validation({}, ["choice must be one listed candidate id"])
    return Validation({"choice": choice}, [])


def fallback_value(interface, observations):
    ys = np.asarray([o.y for o in observations], dtype=float)
    return float(ys.mean()) if interface.rule["fallback"] == "mean" else float(np.median(ys))


def prediction_result(validations, query_ids, interface, observations):
    accepted = {}
    for validation in validations:
        for key, value in validation.values.items():
            accepted.setdefault(key, value)
    constant = fallback_value(interface, observations)
    missing = [key for key in query_ids if key not in accepted]
    return {"values": [accepted.get(key, constant) for key in query_ids],
            "valid": [key in accepted for key in query_ids], "fallback_ids": missing,
            "status": "prediction_fallback" if missing else "valid", "fallback_value": constant}


def selection_fallback(task, replicate, round_index, candidate_count):
    rng = np.random.default_rng(seed_for("icl_boxinggym_selection_fallback", task, replicate, round_index))
    return int(rng.integers(candidate_count))


# ---------------------------------------------------------------- scoring (the engine's rules)
def primary_metric(task):
    """The Ours v4 table's rule: Brier for survival (its truth probabilities barely vary over Q,
    so the nMSE normalizer is ~0 and the nMSE is not meaningful), nMSE against the truth otherwise."""
    return "brier" if task == "survival" else "nmse_truth"


def nmse_truth(predictions, truth):
    """`mda_score_row`'s nMSE: mean squared error against the noise-free mean on Q, over the
    variance of that mean on Q; None where the variance is zero or the truth is undefined."""
    if truth is None:
        return None
    truth = np.asarray(truth, dtype=float)
    means = np.asarray(predictions, dtype=float)
    if means.shape != truth.shape:
        raise ValueError("predictions and truth must have the same shape")
    variance = float(np.var(truth, axis=0))
    if not truth.size or not np.isfinite(truth).all() or not variance > 0:
        return None
    return float(np.mean(np.mean((means - truth) ** 2, axis=0) / variance))


def brier(probabilities, heldout):
    """The engine's Brier score for a binary outcome: mean squared error of P(y = 1) against the
    held-out 0/1 outcomes at Q."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(heldout, dtype=float)
    if p.shape != y.shape:
        raise ValueError("probabilities and outcomes must have the same shape")
    return float(np.mean((p - y) ** 2))


def forbidden_words(text, task):
    """The forbidden matches found in model-facing text (lower-cased, deduplicated)."""
    return sorted({match.group(0).lower() for match in forbidden_pattern(task).finditer(text)})
