"""ICL (+LLM design) for ForceBench: frozen prompts, series rendering, strict parsing, the
runner's score. The model sees two masked control inputs (X2, X3 for the runner's p1, p2 - the
input-name mask's order r, p1, p2, t), the fixed measurement times and initial state, and the
observed two-component output series as numbers; never a world name or a physics word."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bayes_model_discovery.experimental.belief_packet import round_significant
from bayes_model_discovery.experimental.core_count import seed_for
from tools.icl_baseline.protocol import Pairs, finite_number, parse_object

ROOT = Path(__file__).resolve().parent
WORLDS = ("gravity", "yukawa", "coulomb_easy", "oscillator", "extra_dimensions")
#: The runner's design knobs under the input-name mask's names (fb1_grammar.INPUT_NAMES order
#: r, p1, p2, t -> X1..X4); r and t never appear in an ICL table.
INPUT_NAME_MAP = {"X2": "p1", "X3": "p2"}
COMPONENTS = ("u", "v")
T_OBS = 10
#: The runner's rounds (T_EXTRA 16) and candidate count (N_RANDOM_DESIGNS 64).
ROUNDS = 16
N_CANDIDATES = 64
#: Observation counts at which the forecaster is scored: the runner's CHECKPOINTS (2, 4, 8, 16)
#: and its final checkpoint D0 + T_EXTRA = 18.
CHECKPOINTS = (2, 4, 8, 16, 18)
#: World and physics words that must not reach the model (matched at a word start).
FORBIDDEN_STEMS = ("gravit", "yukawa", "coulomb", "oscillat", "extra_dimension", "extra dimension", "kaluza", "klein",
                   "newton", "kepler", "inertia", "mass", "charge", "spring", "pendulum", "orbit", "planet", "potential",
                   "acceler", "veloc", "momentum", "energy", "friction", "drag", "force", "physic", "trajector",
                   "position", "particle", "simulat", "integrat", "verlet", "dynamic", "mechanic", "forcebench",
                   "discoverphysics", "scienceagent")
FORBIDDEN_WORDS = ("p1", "p2", "r", "law", "field", "nbody", "n-body", "world", "benchmark", "posterior", "prior")


def forbidden_pattern():
    stems = [r"\b" + re.escape(word) for word in FORBIDDEN_STEMS]
    words = [r"\b" + re.escape(word) + r"\b" for word in FORBIDDEN_WORDS]
    return re.compile("|".join(stems + words), re.IGNORECASE)


FORBIDDEN = forbidden_pattern()


def forbidden_words(text):
    """The forbidden matches in model-facing text."""
    return sorted({m.group(0).lower() for m in FORBIDDEN.finditer(text)})


# ---------------------------------------------------------------- the anonymized-name interface (2026-10-11)
#: The model-facing interface: "default" (the released prompts, byte for byte) or "neutral" (prompts/neutral/: the engine's
#: neutral wording of tools/fb1_snapshot/fb1_neutral.py fb_interface_neutral_v1 -- a fixed simulator, the derived inputs X1
#: and X4, the start state s0a/s0b, X4 values at the measurements; no mechanics word). The numbers are the same.
#: Unit the full-interface ForceBench edition (2026-10-12): or "full" (see full_text below).
INTERFACES = ("default", "neutral", "full")
NEUTRAL_VERSION = "icl_fb_interface_neutral_v1"
ENGINE_NEUTRAL_VERSION = "fb_interface_neutral_v1"
#: In the neutral prompts the engine's word "simulator" is part of the stated interface; every other stem stays forbidden,
#: and every text must also have 0 hits under the engine's vocabulary rule (audits/fb_vocabulary_audit.py hits_in).
NEUTRAL_ALLOWED_STEMS = ("simulat",)
NEUTRAL_FORBIDDEN = re.compile("|".join([r"\b" + re.escape(word) for word in FORBIDDEN_STEMS if word not in NEUTRAL_ALLOWED_STEMS]
                                        + [r"\b" + re.escape(word) + r"\b" for word in FORBIDDEN_WORDS]), re.IGNORECASE)


def interface_plan(interface="default"):
    if interface not in INTERFACES:
        raise ValueError(f"interface must be one of {INTERFACES}")
    return interface


# ---------------------------------------------------------------- unit the full-interface ForceBench edition (2026-10-12): the full-metadata interface
#: "full" = the released (default) prompts with the input-name mask off (X1..X4 -> r, p1, p2, t, the order of
#: fb1_grammar.INPUT_NAMES; the default prompts name only X2 and X3, the runner's p1 and p2) and ONE shared description
#: block prepended to the system text: the file's text (it ends in a newline), one more newline, then the default system
#: text. The file is byte-identical to the engine branch's copy; its sha256 is computed from the bytes at run time.
#: The masked prompts' sentence that the names carry no meaning (FULL_DROPPED_SENTENCE) belongs to the mask and goes with
#: it. Everything else -- the numbers, the tables (with native column names), the candidates, the scores, the plan --
#: is the default's.
FULL_VERSION = "icl_fb_interface_full_v1"
#: The description file is published with the paper prompts at tools/paper_prompts/ of the repository root.
FULL_DESCRIPTION = ROOT.parents[3] / "tools" / "paper_prompts" / "FB_FULL_DESCRIPTION.md"
FULL_NAME_MAP = {"X1": "r", "X2": "p1", "X3": "p2", "X4": "t"}
MASKED_TOKEN = re.compile(r"(?<![A-Za-z0-9_])X[1-4](?![A-Za-z0-9_])")
FULL_DROPPED_SENTENCE = " X2 and X3 are variable names with no additional meaning provided."
#: In full mode mechanics words and the native names are allowed; law names, world names and catalogue words are not, and
#: no masked name X1..X4 may remain (a whole token, the rule of audits/fb_full_prompt_audit.py).
FULL_FORBIDDEN_STEMS = ("gravit", "yukawa", "coulomb", "oscillat", "extra_dimension", "extra dimension", "kaluza", "klein",
                        "newton", "kepler", "hooke", "harmonic", "spring", "inverse square", "inverse-square", "catalog",
                        "forcebench", "discoverphysics", "scienceagent", "nbody", "n-body")
FULL_FORBIDDEN = re.compile("|".join(r"\b" + re.escape(word) for word in FULL_FORBIDDEN_STEMS), re.IGNORECASE)


def interface_violations(text, interface="default"):
    """Default: the released forbidden list. Neutral: that list without "simulat", plus the engine's vocabulary hits.
    Full: law, world and catalogue words, and any masked name X1..X4 left as a whole token."""
    if interface_plan(interface) == "default":
        return forbidden_words(text)
    if interface == "full":
        return sorted({m.group(0).lower() for m in FULL_FORBIDDEN.finditer(text)} | set(MASKED_TOKEN.findall(text)))
    from tools.icl_forcebench.audits.fb_vocabulary_audit import hits_in
    return sorted({m.group(0).lower() for m in NEUTRAL_FORBIDDEN.finditer(text)} | set(hits_in(text)))


def full_description():
    """The description block's text, read from the file (it ends in a newline)."""
    text = FULL_DESCRIPTION.read_bytes().decode("utf-8")
    if not text.endswith("\n"):
        raise ValueError(f"{FULL_DESCRIPTION} must end in a newline")
    return text


def full_text(text):
    """The input-name mask off: every whole token X1..X4 becomes r, p1, p2, t."""
    return MASKED_TOKEN.sub(lambda match: FULL_NAME_MAP[match.group(0)], text)


def prompt_path(kind, interface="default"):
    folder = ROOT / "prompts" if interface_plan(interface) in ("default", "full") else ROOT / "prompts" / "neutral"
    return folder / f"{kind}.txt"


@dataclass(frozen=True)
class Observation:
    obs_id: int
    design: tuple          # (p1, p2)
    series: tuple          # ((u_1, v_1), ..., (u_10, v_10))
    source: str            # "d0" or "round <k>"


def number(value):
    return json.dumps(round_significant(float(value)), allow_nan=False)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def series_text(series):
    """A (10, 2) output series as `u1,v1,...,u10,v10` (numbers only)."""
    series = np.asarray(series, dtype=float)
    if series.shape != (T_OBS, 2):
        raise ValueError(f"a series has shape (T_OBS, 2); got {series.shape}")
    return ",".join(number(v) for v in series.reshape(-1))


def design_text(design):
    p1, p2 = design
    return f"{number(p1)},{number(p2)}"


def prompt_parts(kind, interface="default"):
    text = prompt_path(kind, interface).read_text()
    system, rest = text.removeprefix("[system]\n").split("\n\n[user]\n", 1)
    user, repair = rest.split("\n\n[appended once, only after a validation failure]\n", 1)
    if interface == "full":
        # Unit the full-interface ForceBench edition: the description block, then one newline, then the default system text; the mask off.
        if user.count(FULL_DROPPED_SENTENCE) != 1:
            raise ValueError("the default prompt no longer holds the masked-name sentence")
        return full_description() + "\n" + system, full_text(user.replace(FULL_DROPPED_SENTENCE, "")), full_text(repair.rstrip("\n"))
    return system, user, repair.rstrip("\n")


def observation_table(observations):
    return "\n".join(",".join([str(o.obs_id + 1), design_text(o.design), series_text(o.series)]) for o in observations)


@dataclass(frozen=True)
class Context:
    """What every request states once: the input range, the measurement times, the initial state."""
    p_range: tuple
    times: tuple
    pos0: tuple
    vel0: tuple

    def fill(self, user):
        return (user.replace("<lo>", number(self.p_range[0])).replace("<hi>", number(self.p_range[1]))
                .replace("<times>", ", ".join(number(t) for t in self.times))
                .replace("<u0>", number(self.pos0[0])).replace("<v0>", number(self.pos0[1]))
                .replace("<du0>", number(self.vel0[0])).replace("<dv0>", number(self.vel0[1])))


def _fill(user, observations, context):
    user = context.fill(user)
    return user.replace("<n>", str(len(observations))).replace("<one row per experiment>", observation_table(observations))


def render_prediction(observations, queries, context, interface="default"):
    """`queries` = [(label, (p1, p2)), ...] the held-out evaluation designs."""
    if not queries:
        raise ValueError("prediction requires held-out designs")
    system, user, _ = prompt_parts("prediction", interface)
    mapping = [{"id": f"q{i + 1}", "label": label, "design": [float(p1), float(p2)]}
               for i, (label, (p1, p2)) in enumerate(queries)]
    table = "\n".join(",".join([row["id"], design_text(row["design"])]) for row in mapping)
    user = _fill(user, observations, context).replace("<one row per held-out setting, q = q1..q<Q>>", table)
    user = user.replace("<Q>", str(len(mapping)))
    return {"system": system, "messages": [{"role": "user", "content": user}]}, mapping


def render_selection(observations, candidates, context, interface="default"):
    """`candidates` = the round's random designs, an array of shape (C, 2)."""
    candidates = np.asarray(candidates, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 2 or len(candidates) == 0:
        raise ValueError("candidates are a non-empty (C, 2) array")
    system, user, _ = prompt_parts("selection", interface)
    mapping = [{"id": f"c{i + 1}", "candidate_index": i, "design": [float(p1), float(p2)]}
               for i, (p1, p2) in enumerate(candidates)]
    table = "\n".join(",".join([row["id"], design_text(row["design"])]) for row in mapping)
    user = _fill(user, observations, context).replace("<one row per candidate, c = c1..c<C>>", table)
    user = user.replace("<C>", str(len(mapping)))
    return {"system": system, "messages": [{"role": "user", "content": user}]}, mapping


def repair_request(original, first_text, error, kind, count, interface="default"):
    _, _, repair = prompt_parts(kind, interface)
    repair = repair.replace("<error>", error).replace("<Q>", str(count))
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


def check_series(value):
    if not isinstance(value, list) or len(value) != T_OBS:
        return f"requires a list of {T_OBS} [u, v] pairs"
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            return "each element must be a pair [u, v]"
        if not all(finite_number(v) for v in pair):
            return "each pair must hold two finite JSON numbers"
    return None


def validate_prediction(text, query_ids, stop_reason="end_turn"):
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
        problem = check_series(items[0])
        if problem:
            errors.append(f"{key}: {problem}")
        else:
            values[key] = [[float(u), float(v)] for u, v in items[0]]
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


def fallback_series(observations):
    """The per-time, per-component median of the observed series (the ChemBench rule, per value)."""
    table = np.asarray([o.series for o in observations], dtype=float)
    return [[float(u), float(v)] for u, v in np.median(table, axis=0)]


def prediction_result(validations, query_ids, observations):
    accepted = {}
    for validation in validations:
        for key, value in validation.values.items():
            accepted.setdefault(key, value)
    constant = fallback_series(observations)
    missing = [key for key in query_ids if key not in accepted]
    return {"values": [accepted.get(key, constant) for key in query_ids],
            "valid": [key in accepted for key in query_ids], "fallback_ids": missing,
            "status": "prediction_fallback" if missing else "valid", "fallback_value": constant}


def selection_fallback(world, replicate, round_index, candidate_count):
    rng = np.random.default_rng(seed_for("icl_forcebench_selection_fallback", world, replicate, round_index))
    return int(rng.integers(candidate_count))


# ---------------------------------------------------------------- scoring (the runner's rule)
def nmse(predicted, truth, var_gt):
    """The runner's `nmse_mda` (run_gravity.py score_set: mean over designs x times x components of
    the squared error of the point forecast against the noise-free truth, over VAR_GT), here on
    the model's forecast series instead of the posterior predictive mean."""
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.ndim != 3 or predicted.shape[1:] != (T_OBS, 2):
        raise ValueError(f"series arrays of shape (D, {T_OBS}, 2) are required; got {predicted.shape} and {truth.shape}")
    return float(np.mean((predicted - truth) ** 2) / float(var_gt))
