"""ICL (+LLM design) for NewtonBench, 2026-10-09: the registration-v1 prompt assets for a task with n inputs,
their rendering, the repair text, the forbidden-word precheck and the domain-word count for thinking summaries.

The two assets (prompts/prediction.txt, prompts/selection.txt) are the ChemBench ICL assets of registration v5 section 5
(tools/icl_baseline/prompts) with only the interface turned into placeholders: the input count and names, the input
ranges, the derived inputs the engine's interface declares (presentation P's log-centred Xil), the table headers, the
selection's JSON fields and the input phrase of the selection repair sentence. Every other sentence is the ChemBench
asset's. The ChemBench selection asset's "(the same first two lines and observation table as 5.1)" becomes "(the same data
block and observation table as the forecast request)": the data block now has a third line, the derived inputs. The
renderer fills the placeholders from the engine's own masked interface (`icl_ng.interface_of`); the observation and query
tables reuse the ChemBench tool's `observation_table` and `number` (4 significant digits, `belief_packet.round_significant`,
the rounding the engine's packets use)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from tools.icl_baseline.protocol import number, observation_table

ROOT = Path(__file__).resolve().parent
NUMBER_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
DATA_BLOCK_MARKER = "(the same data block and observation table as the forecast request)"
DERIVED_LINE = "<derived inputs line>"
#: The placeholders an asset may still hold after rendering: the JSON value marker of the answer formats.
ALLOWED_LEFTOVERS = {"<number>"}

#: The forbidden-word precheck (every request, before it is written or sent). Word starts (stems) of the twelve
#: NewtonBench domains, their observables and their native input names, plus the engine words a forecaster prompt must not
#: carry (the ChemBench list's model-machinery words, and the Xiz names: the z-scored inputs are not shown, as in the
#: ChemBench ICL registrations). Single-letter native names (k, A, d, t, T, M, x, m, c) are left out: as whole words they
#: would match ordinary text, and the prompts carry no native name at all (masking X1..Xn).
FORBIDDEN_STEMS = (
    "magnet", "ampere", "fourier", "conduct", "snell", "refract", "radioactiv", "decay", "half-life", "malus", "polariz",
    "polaris", "sound", "acoustic", "hooke", "spring", "bose", "einstein", "boson", "planck", "occupation", "heat",
    "thermal", "gravit", "coulomb", "newton", "physic", "electr", "current", "charge", "force", "power", "energy",
    "intensity", "activity", "speed", "velocity", "temperature", "distance", "angle", "frequency", "damp", "oscillat",
    "harmonic", "photon", "light", "lognormal", "residual", "coefficient", "posterior", "archive", "response unit",
    "response_unit", "benchmark",
)
FORBIDDEN_WORDS = (
    "current1", "current2", "delta_T", "refractive_index_1", "refractive_index_2", "incidence_angle", "N0",
    "lambda_constant", "lambda", "I_0", "theta", "gamma", "omega", "mass1", "mass2", "q1q2", "law", "laws", "pool",
    "family", "expression", "prior", "log link", "link", *(f"X{i}z" for i in range(1, 10)),
)
FORBIDDEN = re.compile("|".join([r"(?<![A-Za-z0-9_])" + re.escape(stem) for stem in FORBIDDEN_STEMS]
                                + [r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])" for word in FORBIDDEN_WORDS]),
                       re.IGNORECASE)
#: NewtonBench domain words in a thinking summary (counted, never acted on; the ChemBench tool counts its own words).
DOMAIN_WORDS = re.compile(
    r"magnet|amp[ee]re|fourier|conduct|snell|refract|radioactiv|half[- ]?life|malus|polari[sz]|sound|acoustic|hooke|"
    r"spring|bose|einstein|boson|planck|occupation|heat|thermal|gravit|coulomb|newton", re.IGNORECASE)


@dataclass(frozen=True)
class Interface:
    """A masked NewtonBench task as the engine's public interface declares it: the inputs X1..Xn in declaration order,
    their ranges, and the derived inputs ((name, definition), ...) in the engine's order (the log-centred Xil)."""

    names: tuple
    bounds: tuple
    derived: tuple

    def __post_init__(self):
        n = len(self.names)
        if not 1 <= n <= len(NUMBER_WORDS) or tuple(self.names) != tuple(f"X{i}" for i in range(1, n + 1)):
            raise ValueError("an interface declares X1..Xn with 1 <= n <= 9")
        if len(self.bounds) != n or not all(len(b) == 2 and all(math.isfinite(float(v)) for v in b) and b[0] < b[1]
                                            for b in self.bounds):
            raise ValueError("every input needs one finite range [low, high] with low < high")
        if any(not re.fullmatch(r"X[1-9]l", name) or "<" in definition or ">" in definition
               for name, definition in self.derived):
            raise ValueError("derived inputs are the engine's Xil definitions")

    def placeholders(self):
        names, n = self.names, len(self.names)
        word = NUMBER_WORDS[n - 1]
        if n == 1:
            sentence, inputs = f"{names[0]} is a variable name", f"one controllable input {names[0]}"
            repair = f"with {names[0]} inside its range"
        else:
            sentence = (f"{names[0]} and {names[1]} are variable names" if n == 2
                        else f"{names[0]} through {names[-1]} are variable names")
            inputs, repair = f"{word} controllable inputs {names[0]}..{names[-1]}", f"with all {word} inputs inside their ranges"
        return {
            "<variable names sentence>": sentence + " with no additional meaning provided.",
            "<inputs>": inputs,
            "<input ranges>": ", ".join(f"{name} [{json.dumps(float(lo))}, {json.dumps(float(hi))}]"
                                        for name, (lo, hi) in zip(names, self.bounds)),
            "<input header>": ",".join(names),
            "<selection fields>": ", ".join(f'"{name}": <number>' for name in names),
            "<selection inputs phrase>": repair,
        }

    def derived_line(self):
        """The engine's `interface.derived_inputs` Xil entries, written `name = definition` (no columns are added)."""
        return "Derived inputs: " + ", ".join(f"{name} = {definition}" for name, definition in self.derived) + "." \
            if self.derived else ""


def prompt_parts(kind):
    text = (ROOT / "prompts" / f"{kind}.txt").read_text()
    system, rest = text.removeprefix("[system]\n").split("\n\n[user]\n", 1)
    user, repair = rest.split("\n\n[appended once, only after a validation failure]\n", 1)
    return system, user, repair.rstrip("\n")


def fill(text, interface):
    for key, value in interface.placeholders().items():
        text = text.replace(key, value)
    line = interface.derived_line()
    text = text.replace(DERIVED_LINE + "\n", line + "\n" if line else "")
    if DERIVED_LINE in text:
        raise ValueError("the derived-inputs placeholder must stand on its own line")
    return text


def check_rendered(text):
    leftovers = set(re.findall(r"<[A-Za-z][^<>\n]*>", text)) - ALLOWED_LEFTOVERS
    if leftovers:
        raise ValueError(f"unfilled prompt placeholders: {sorted(leftovers)}")


def render(kind, observations, interface, *, queries=None):
    """Selection accepts observations and the interface only (no evaluation queries, as in the ChemBench tool)."""
    if kind not in ("prediction", "selection"):
        raise ValueError("unregistered prompt variant")
    if kind == "selection" and queries is not None:
        raise ValueError("selection input allowlist excludes evaluation queries")
    if any(len(o.x) != len(interface.names) for o in observations):
        raise ValueError("observation width differs from the interface")
    system, user, _ = prompt_parts(kind)
    if kind == "selection":
        _, prediction, _ = prompt_parts("prediction")
        if user.count(DATA_BLOCK_MARKER) != 1:
            raise ValueError("the selection asset must hold the data-block marker once")
        user = user.replace(DATA_BLOCK_MARKER, prediction.split("\n\nNew inputs", 1)[0])
    system, user = fill(system, interface), fill(user, interface)
    user = user.replace("<n>", str(len(observations))).replace("<one row per experiment>", observation_table(observations))
    mapping = []
    if kind == "prediction":
        if queries is None or len(queries) == 0:
            raise ValueError("prediction requires query coordinates")
        for index, values in enumerate(queries):
            if len(values) != len(interface.names) or not all(math.isfinite(float(v)) for v in values):
                raise ValueError("invalid query coordinates")
            mapping.append({"id": f"q{index + 1}", "coordinates": list(map(float, values))})
        table = "\n".join(",".join([row["id"], *(number(v) for v in row["coordinates"])]) for row in mapping)
        user = user.replace("<one row per new input, q = q1..q<Q>>", table).replace("<Q>", str(len(queries)))
    check_rendered(system + "\n" + user)
    return {"system": system, "messages": [{"role": "user", "content": user}]}, mapping


def repair_request(original, first_text, error, kind, interface, query_count=0):
    """The ChemBench repair conversation (tools/icl_baseline/protocol.repair_request) with this asset's repair text."""
    _, _, repair = prompt_parts(kind)
    repair = fill(repair, interface).replace("<error>", error).replace("<Q>", str(query_count))
    return {"system": original["system"], "messages": [
        *original["messages"], {"role": "assistant", "content": first_text},
        {"role": "user", "content": repair},
    ]}


def forbidden_words(text, extra=()):
    """Forbidden matches in model-facing text: the lists above plus `extra` (a task's own multi-letter native input and
    observable names), each at word boundaries, case-insensitive."""
    found = {match.group(0).lower() for match in FORBIDDEN.finditer(text)}
    for word in extra:
        if len(word) > 1 and re.search(r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])", text, re.IGNORECASE):
            found.add(word.lower())
    return sorted(found)
