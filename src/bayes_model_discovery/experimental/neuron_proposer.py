"""Public-only mechanism proposals; standalone transport, with no runner wiring or fits.

Supply declared parameter windows explicitly. The fixed prompt contains no task data.
Use make_neuron_proposer for the structure-bit gate; injected clients never count as
paid calls. A proposed hypothesis freezes its channel shape and fits only the existing
positive conductance multiplier through neuron_forward.predict_features.
"""
import hashlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import neuron_forward
from .belief_packet import (
    ARCHIVE_OUTCOMES,
    ARCHIVE_REASON_KEYS,
    packet_digest,
    round_significant,
    serialize_packet,
)
from .count_proposer import (
    BUDGET_EXHAUSTED,
    DEFAULT_MODEL,
    DEFAULT_MODEL_OPENAI,
    EmptyResponseBody,
    PaidCallBudgetExhausted,
    _anthropic_usage,
    _http_request_id,
    _openai_usage,
    _reserve_paid_call,
    _thinking_block_count,
    _usage_total,
    _validate_max_rounds,
    _validate_repair_attempts,
    _write_once,
    anthropic_thinking_record,
    apply_extra_body,
    blank_reply,
    compat_client_kwargs,
    compat_response_metadata,
    create_anthropic_message,
    create_openai_completion,
    decoding_identity,
    install_secret_log_filter,
    mask_secrets,
    mask_secrets_deep,
    masked_exception,
    normalize_provider_order,
    normalize_reasoning,
    parse_max_paid_calls,
    parse_max_tokens,
    repair_kind,
    repair_messages,
    validate_openai_compat,
    validate_provider_reasoning,
)

PROMPT_VERSION_MECHANISM = "v3.0"
PROMPT_FILE = Path(__file__).with_name("prompts") / "proposer_system_v3_0_mechanism.txt"
STRUCTURE_MODES = {"0": "catalog", "1": "free"}
RECOVERY_MODES = {"catalog": "catalog_selection", "free": "freeform_discovery"}
MECHANISM_VOCABULARY_REFUSAL_REASON = "mechanism is outside the declared vocabulary"
F0_REFUSAL_REASON = "F0 requires the initial likelihood family"  # Not applicable; never fired.
S0_REFUSAL_REASON = "S0 requires an initial mechanism"
JSON_REFUSAL = "reply is not valid JSON: {error}"
#: Opt-in code-fence reply rule `--llm-accept-code-fence`. A reply that is exactly one
#: Markdown code fence (```json or a bare ```), with nothing but whitespace outside it, is parsed
#: from the fence's inside; anything else - text before or after the fence, several fences, no
#: fence - is parsed exactly as written, so it is refused for the same reason as without the
#: option. Without the option nothing is stripped and every record keeps its bytes.
CODE_FENCE_POLICY = "single_outer_fence_v1"
_OUTER_FENCE = re.compile(r"\s*```(?:json|JSON)?[ \t]*\r?\n(.*?)\r?\n?[ \t]*```\s*", re.DOTALL)


def strip_outer_code_fence(text):
    """(text to parse, whether a fence was removed) under `CODE_FENCE_POLICY`."""
    if not isinstance(text, str):
        return text, False
    match = _OUTER_FENCE.fullmatch(text)
    return (match.group(1), True) if match else (text, False)
OBJECT_REFUSAL = "reply must be one JSON object, not an array"
NON_OBJECT_REFUSAL = "reply must be one JSON object"
REQUIRED_KEYS_REFUSAL = "candidate requires mechanism, kind, and parameters"
PARAMETERS_REFUSAL = "parameters must be a mapping"
KIND_REFUSAL = "kind and parameters must agree with the declared mechanism"
WINDOW_REFUSAL = "parameters must be finite numbers inside the declared windows"
POWER_REFUSAL = "mpow and hpow must be integers in {1, 2}"
DUPLICATE_REFUSAL = "candidate model_id is already known"
FINITENESS_REFUSAL = "non-finite native HH output in initial-design probe"
REPAIR_INSTRUCTION = (
    "Your previous reply failed validation with this error:\n{error}\n"
    'Reply with exactly one JSON object {{"mechanism": ..., "kind": ..., '
    '"parameters": {{...}}}} and nothing else.\n'
    "Every value in parameters is a number or null, not an object."
)
# Empty-reply repair rule: the same rule as count_proposer.REPAIR_INSTRUCTION_EMPTY, with this
# transport's own JSON-shape sentence - an empty reply is never echoed back.
REPAIR_INSTRUCTION_EMPTY = (
    "Your previous reply was empty (finish_reason={finish_reason}).\n"
    'Reply with exactly one JSON object {{"mechanism": ..., "kind": ..., '
    '"parameters": {{...}}}} and nothing else.\n'
    "Every value in parameters is a number or null, not an object."
)
#: Multi-candidate proposals (added 2026-10-05 for the unified protocol): k > 1 candidates in ONE
#: call. The system text is the v3.0 prompt file's bytes plus this paragraph, the count transport's
#: COMPOSITION_PARAGRAPH semantics (a JSON array of exactly k objects of the same shape, mutually
#: different). Its sub-expression clause has no meaning for a closed mechanism vocabulary, so
#: "whose mechanisms differ" stands in its place. k = 1 composes nothing and keeps every request,
#: parse and record byte.
#: Final unified protocol (2026-10-06): COMPOSITION_PARAGRAPH v2, the registered bytes of
#: composition_paragraph_v2.txt (sha256 COMPOSITION_PARAGRAPH_SHA256, 525 bytes, the final
#: wording fixed on 2026-10-06; bound by a test), the same text every engine appends;
#: `{limit}` is filled by the runner and the text is written for three candidates, so a composed
#: request asks for exactly COMPOSITION_LIMIT. The "primary functional composition" it speaks of
#: is, for this benchmark, the mechanism and kind of each candidate.
COMPOSITION_PARAGRAPH = (
    "Return a JSON array of exactly {limit} candidate objects, each with exactly the keys "
    "specified above and nothing else. Propose three distinct, non-equivalent hypotheses. At "
    "least two of the three candidates should differ meaningfully in their primary functional "
    "composition, rather than merely in parameterization or by adding or removing a correction "
    "term. At most one candidate may be a nested or incremental refinement of a high-posterior "
    "model, and only if such a refinement is specifically supported by the diagnostics.\n"
)
COMPOSITION_PARAGRAPH_VERSION = "v2"
COMPOSITION_PARAGRAPH_SHA256 = "782a30b74cd25adb6b51351096cf97e31a7375d8630d726ee8eb7d7949c35be6"
COMPOSITION_LIMIT = 3
ARRAY_REFUSAL = "reply must be a JSON array of 1..{limit} candidate objects"
NO_VALID_ARRAY_CANDIDATE = "no candidate of the array passed validation"
REPAIR_INSTRUCTION_ARRAY = (
    "Your previous reply failed validation with this error:\n{error}\n"
    'Reply with a JSON array of exactly {limit} objects {{"mechanism": ..., "kind": ..., '
    '"parameters": {{...}}}} and nothing else.\n'
    "Every value in parameters is a number or null, not an object."
)
REPAIR_INSTRUCTION_EMPTY_ARRAY = (
    "Your previous reply was empty (finish_reason={finish_reason}).\n"
    'Reply with a JSON array of exactly {limit} objects {{"mechanism": ..., "kind": ..., '
    '"parameters": {{...}}}} and nothing else.\n'
    "Every value in parameters is a number or null, not an object."
)


def validate_proposal_limit(limit):
    """Candidates asked for per round, 1..3: the count transport's own bound."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 3:
        raise ValueError("proposal_limit must be an integer 1..3")
    return limit


def composed_system_prompt(proposal_limit=1):
    """The system text sent and its identity: the prompt file alone at k = 1 (no identity keys,
    so every k = 1 record keeps its bytes); the file plus COMPOSITION_PARAGRAPH at k > 1."""
    validate_proposal_limit(proposal_limit)
    text = PROMPT_FILE.read_text()
    if proposal_limit == 1:
        return text, {}
    if proposal_limit != COMPOSITION_LIMIT:
        # Final unified protocol: v2 says "three" in words, so it is composed for exactly three.
        raise ValueError(f"COMPOSITION_PARAGRAPH {COMPOSITION_PARAGRAPH_VERSION} is written for "
                         f"{COMPOSITION_LIMIT} candidates, not {proposal_limit}")
    paragraph = COMPOSITION_PARAGRAPH.replace("{limit}", str(proposal_limit))
    composed = text + paragraph
    return composed, {"proposal_limit": proposal_limit,
                      "composition_paragraph_version": COMPOSITION_PARAGRAPH_VERSION,
                      "composition_paragraph_sha256": COMPOSITION_PARAGRAPH_SHA256,
                      "composed_prompt_sha256": hashlib.sha256(composed.encode()).hexdigest()}


PARAMETER_NAMES = ("E_mV", "mvh_mV", "mk_mV", "mtau_ms", "mpow", "hvh_mV",
                   "hk_mV", "htau_ms", "hpow")
CONTINUOUS_PARAMETERS = tuple(k for k in PARAMETER_NAMES if k not in ("mpow", "hpow"))
BASE_PARAMETERS = set(PARAMETER_NAMES[:6])
INACTIVATION_PARAMETERS = set(PARAMETER_NAMES[6:])
#: The observation record keys either side of the feature vector the space names.
OBSERVATION_KEYS_HEAD = ("round", "design_index", "label")
OBSERVATION_KEYS_TAIL = ("spike_count", "cost", "test_start_observed")
OBSERVATION_REFUSAL = "observation must identify one public protocol and three features"
DECLARED_FEATURES_REFUSAL = ("observation must identify one public protocol and its "
                             "declared feature vector")


def _default_probe(candidate, theta, design):
    """The three-feature forward, resolved when the probe RUNS so that a replaced
    `neuron_forward.predict_features` is honoured rather than a reference captured here."""
    return neuron_forward.predict_features(candidate, theta, design)


@dataclass(frozen=True)
class FeatureSpace:
    """Which observation feature vector a packet carries, checks, publishes and probes.

    One object decides all of it: the key an observation row spells the vector under, the
    names the packet publishes, what the interface discloses about the observation noise
    (one millivolt unit for physical features, per-feature scales for counted ones), the
    forward the finiteness probe calls, and the message a malformed row is refused with.
    Nothing downstream asks how many features there are or what they are called - the
    length is always `len(names)` and everything else is read off these fields - so a
    further feature space is a new descriptor rather than a new branch.
    """

    record_key: str
    names: tuple
    probe: object
    noise_unit_mV: float = None
    feature_scales: tuple = None
    message: str = DECLARED_FEATURES_REFUSAL

    @property
    def noise_disclosure(self):
        """What the packet interface states the observation noise is measured in."""
        if self.feature_scales is None:
            return {"noise_unit_mV": self.noise_unit_mV}
        return {"feature_scales": list(self.feature_scales)}


#: The three physical millivolt features every packet carried before selectable feature spaces
#: were added, unchanged: its record key, names, noise unit, probe and refusal message are the
#: literals from before that change.
DEFAULT_FEATURE_SPACE = FeatureSpace(
    record_key="features_mV",
    names=("test-window mean mV", "overall minimum mV", "final mV"),
    probe=_default_probe, noise_unit_mV=20.0, message=OBSERVATION_REFUSAL)


def observation_keys(feature_space):
    """The keys one observation row contributes, with that space's record key in place."""
    return (*OBSERVATION_KEYS_HEAD, feature_space.record_key, *OBSERVATION_KEYS_TAIL)


OBSERVATION_KEYS = observation_keys(DEFAULT_FEATURE_SPACE)


def structure_mode(cell):
    """Return the structure mode ("catalog" or "free") of a three-character cell code [01][01][012].

    Raises ValueError for a malformed cell code."""
    if (not isinstance(cell, str) or len(cell) != 3
            or set(cell[:2]) - {"0", "1"} or cell[2] not in "012"):
        raise ValueError("cell must match [01][01][012]")
    return STRUCTURE_MODES[cell[0]]


def neuron_prompt_identity():
    """Return the mechanism prompt's version and the sha256 of its file bytes, for records."""
    return {"prompt_version": PROMPT_VERSION_MECHANISM,
            "prompt_sha256": hashlib.sha256(PROMPT_FILE.read_bytes()).hexdigest()}


def _shape(values):
    parameters = dict(zip(PARAMETER_NAMES, values))
    if parameters["hvh_mV"] is None:
        parameters.update(hk_mV=None, htau_ms=None, hpow=None)
    else:
        parameters.setdefault("hpow", 1)
    return parameters


def mechanism_vocabulary():
    """Public candidate recipes, independent of any selected world's hidden state."""
    result = {name: {"kind": "base_conductance", "parameters": {}}
              for name in ("leak", "sodium", "potassium")}
    result["slow_na"] = {"kind": "slow_inactivation_flag", "parameters": {}}
    shapes = {
        "h_current": (-35.0, -85.0, -8.0, 100.0, 1, None),
        "m_current": (-77.0, -40.0, 12.0, 80.0, 1, None),
        "inward": (50.0, -45.0, 8.0, 10.0, 1, None),
        # Public catalogue definitions shared with the independent worlds unit.
        "rebound_inward_slow": (120.0, -57.0, 5.0, 4.0, 2, -88.0, 4.0, 130.0),
        "rebound_inward_fast": (120.0, -54.0, 6.0, 2.0, 2, -87.0, 4.0, 22.0),
        "rebound_outward": (-77.0, -30.0, 10.0, 3.0, 1, -80.0, 5.0, 200.0),
    }
    result.update({name: {"kind": "extra_channel", "parameters": _shape(shape)}
                   for name, shape in shapes.items()})
    return result


def _windows(windows):
    if not isinstance(windows, dict) or not set(CONTINUOUS_PARAMETERS) <= set(windows):
        raise ValueError("explicit parameter_windows required for every continuous parameter")
    result = {}
    for key in CONTINUOUS_PARAMETERS:
        bounds = windows[key]
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]):
            raise ValueError("parameter windows must be finite ordered [minimum, maximum] pairs")
        result[key] = list(bounds)
    result.update(mpow=[1, 2], hpow=[1, 2])
    return result


def _archive_row(row):
    outcome = row["outcome"]
    if outcome not in ARCHIVE_OUTCOMES or set(row["reason"]) != ARCHIVE_REASON_KEYS[outcome]:
        raise ValueError("archive outcome/reason keys must follow the shared archive contract")
    return {key: row[key] for key in ("model_id", "mechanism", "kind", "parameters",
                                     "proposed_round", "outcome", "reason", "note") if key in row}


def build_neuron_belief_packet(world, observations, fits, archive, cell, *, parameter_windows,
                               allowed_mechanisms=None, initial_mechanisms=None, acquisition=None,
                               feature_space=DEFAULT_FEATURE_SPACE):
    """Return public interface/observations/pool/archive/constraints/prompt identity.

    world.problem() is the benchmark's public boundary. No evaluator, score, hidden
    protocol segments, private world attributes or observation seed is accessed.
    Fits are the existing vector fit results; only their hypothesis and posterior
    conductance/sigma summaries enter the packet. Use serialize_packet/packet_digest.
    `feature_space` is the observation feature space the run trains on; the
    default is the three physical features, whose packet is unchanged byte for byte.
    """
    public = world.problem()
    fits = list(fits)
    vocabulary = mechanism_vocabulary() if allowed_mechanisms is None else allowed_mechanisms
    initial = ([hyp.mechanism for hyp in neuron_forward.initial_hypotheses()]
               if initial_mechanisms is None else list(initial_mechanisms))
    if not isinstance(vocabulary, dict) or not set(initial) <= set(vocabulary):
        raise ValueError("initial mechanisms must belong to the declared vocabulary")
    interface = {
        "text_prior": public["text_prior"],
        "reference_model": {key: public["reference_model"][key] for key in ("extra", "slow_na")},
        "protocols": [(label, [(float(duration), float(current)) for duration, current in segments])
                      for label, segments in public["protocols"]],
        "test_protocol_labels": list(public["test_protocol_labels"]),
        "budget_rule": public["budget_rule"],
        "observable": list(feature_space.names),
        **feature_space.noise_disclosure,
        "forward": "native neuronbench.worlds.simulate dt=0.01 ms without truth kwargs",
    }
    keys = observation_keys(feature_space)
    obs = []
    for row in observations:
        # A row measured in another feature space is a refusal, not a KeyError.
        if feature_space.record_key not in row:
            raise ValueError(feature_space.message)
        obs.append({key: row[key] for key in keys})
    for row in obs:
        index = row["design_index"]
        if (isinstance(index, bool) or not isinstance(index, int)
                or not 0 <= index < len(interface["protocols"])
                or row["label"] != interface["protocols"][index][0]
                or np.asarray(row[feature_space.record_key]).shape
                != (len(feature_space.names),)):
            raise ValueError(feature_space.message)
    logz = np.asarray([result.log_evidence for result in fits], dtype=float)
    weights = np.exp(logz - logz.max()) if len(logz) else np.asarray([])
    if len(weights):
        weights /= weights.sum()
    pool = []
    for result, weight in zip(fits, weights):
        hyp = result.hypothesis
        recipe = vocabulary[hyp.mechanism]
        pool.append({"model_id": hyp.model_id, "mechanism": hyp.mechanism,
                     "kind": recipe["kind"],
                     "parameters": getattr(hyp, "parameters", recipe["parameters"]),
                     "log_evidence": result.log_evidence, "weight": float(weight),
                     "posterior": {name: {"mean": float(np.mean(result.particles[name])),
                                           "std": float(np.std(result.particles[name]))}
                                   for name in ("conductance", "sigma")}})
    packet = {"interface": interface, "observations": obs, "pool": pool,
              "archive": [_archive_row(row) for row in archive],
              "constraints": {"structure_expansion": structure_mode(cell) == "free",
                              "family_expansion": False, "initial_mechanisms": initial,
                              "allowed_mechanisms": vocabulary,
                              "parameter_windows": _windows(parameter_windows)},
              **neuron_prompt_identity()}
    if acquisition is not None:
        packet["acquisition"] = [{key: row[key] for key in ("round", "selected", "available", "mode")}
                                 for row in acquisition]
    return round_significant(packet)


def _identity(mechanism, kind, parameters):
    # Float/int spellings and absent/null inactive-gate values have one identity.
    canonical = {key: None if value is None else float(value) for key, value in parameters.items()}
    payload = json.dumps([mechanism, kind, canonical], sort_keys=True, separators=(",", ":"))
    recipe = mechanism_vocabulary().get(mechanism)
    if recipe and kind == recipe["kind"] and canonical == recipe["parameters"]:
        return "hh_" + mechanism + "_gaussian_features"
    return "hh_proposed_" + hashlib.sha256(payload.encode()).hexdigest()[:24] + "_gaussian_features"


@dataclass(frozen=True)
class NeuronHypothesis:
    """A proposed mechanism: name, kind and frozen channel-shape parameters (sorted item tuple).

    Only the positive conductance multiplier is fitted; the shape stays fixed."""

    mechanism: str
    kind: str
    parameter_items: tuple

    @property
    def parameters(self):
        """The channel-shape parameters as a dict."""
        return dict(self.parameter_items)

    @property
    def model_id(self):
        """Stable model identifier: built from the catalogue name for an unchanged recipe, else
        from a hash of mechanism, kind and parameters."""
        return _identity(self.mechanism, self.kind, self.parameters)

    @property
    def expression(self):
        """Sorted-key JSON text of mechanism, kind and parameters, used in duplicate records."""
        return json.dumps({"mechanism": self.mechanism, "kind": self.kind,
                           "parameters": self.parameters}, sort_keys=True)

    def predict(self, theta, design):
        """Same forward interface as HHHypothesis; shape is fixed, conductance fitted."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (1,) or not np.isfinite(theta[0]) or theta[0] <= 0:
            raise ValueError("exactly one finite positive conductance multiplier required")
        if self.kind != "extra_channel":
            return neuron_forward.HHHypothesis(self.mechanism).predict(theta, design)
        from neuronbench.worlds import Chan, build_I, simulate
        p = self.parameters
        channel = Chan(self.mechanism, float(theta[0]), p["E_mV"], p["mvh_mV"],
                       p["mk_mV"], p["mtau_ms"], p["mpow"], p["hvh_mV"],
                       p["hk_mV"] if p["hvh_mV"] is not None else 4.0,
                       p["htau_ms"] if p["hvh_mV"] is not None else 1e9,
                       p["hpow"] if p["hvh_mV"] is not None else 1)
        current, start = build_I(design[1])
        _, voltage = simulate(current, start, extra=[channel], trace=True)
        if not np.all(np.isfinite(voltage)):
            raise ValueError("non-finite native HH output")
        return np.asarray(voltage).copy()


class CandidateValidationError(ValueError):
    """A refused candidate: the message, the validation stage that refused it, and optionally
    the candidate and its model id (used for archive entries)."""

    def __init__(self, message, stage, *, candidate=None, model_id=None):
        super().__init__(message)
        self.stage = stage
        self.candidate = candidate
        self.model_id = model_id


def finiteness_probe(candidate, initial_designs, feature_space=DEFAULT_FEATURE_SPACE):
    """Run the feature forward on the two initial designs at multiplier 1.0; raise
    CandidateValidationError (stage "finiteness") on a wrong shape, non-finite output or any error."""
    if len(initial_designs) != 2:
        raise ValueError("exactly two initial designs required for the finiteness probe")
    try:
        for design in initial_designs:
            features = feature_space.probe(candidate, [1.0], design)
            if (np.asarray(features).shape != (len(feature_space.names),)
                    or not np.all(np.isfinite(features))):
                raise ValueError(FINITENESS_REFUSAL)
    except Exception as exc:  # noqa: BLE001 - Preserve response and transport failure handling with masked diagnostics.
        raise CandidateValidationError(FINITENESS_REFUSAL, "finiteness", candidate=candidate,
                                       model_id=candidate.model_id) from masked_exception(exc)


def parse_neuron_candidate(text, *, constraints, initial_designs, known_ids=(),
                           feature_space=DEFAULT_FEATURE_SPACE, accept_code_fence=False):
    """Validate in the canonical order (JSON, shape, vocabulary, kind, numeric windows, cell,
    duplicate, finiteness); return (hypothesis, ignored/note metadata).

    F0 is deliberately not checked: this benchmark has no family expansion axis.
    A direct parser can validate S0; the transport factory never creates an S0 client.
    `accept_code_fence` (opt-in) first removes one outer fence under `CODE_FENCE_POLICY`.
    """
    if accept_code_fence:
        text, _ = strip_outer_code_fence(text)
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise CandidateValidationError(JSON_REFUSAL.format(error=exc), "json") from None
    if not isinstance(value, dict):
        raise CandidateValidationError(OBJECT_REFUSAL if isinstance(value, list)
                                       else NON_OBJECT_REFUSAL, "shape")
    if not {"mechanism", "kind", "parameters"} <= set(value):
        raise CandidateValidationError(REQUIRED_KEYS_REFUSAL, "shape")
    if not isinstance(value["parameters"], dict):
        raise CandidateValidationError(PARAMETERS_REFUSAL, "shape")
    original = value
    mechanism, kind, p = value["mechanism"], value["kind"], dict(value["parameters"])
    vocabulary = constraints["allowed_mechanisms"]
    if not isinstance(mechanism, str) or mechanism not in vocabulary:
        rejected_id = "hh_rejected_" + hashlib.sha256(
            json.dumps([mechanism, kind, p], sort_keys=True).encode()).hexdigest()[:24]
        raise CandidateValidationError(MECHANISM_VOCABULARY_REFUSAL_REASON, "vocabulary",
                                       candidate=value, model_id=rejected_id)
    if kind != vocabulary[mechanism]["kind"] or kind not in (
            "base_conductance", "slow_inactivation_flag", "extra_channel"):
        raise CandidateValidationError(KIND_REFUSAL, "kind")
    if kind == "extra_channel":
        active_h = p.get("hvh_mV") is not None
        required = BASE_PARAMETERS | (INACTIVATION_PARAMETERS if active_h else set())
        if (not required <= set(p) or set(p) - (BASE_PARAMETERS | INACTIVATION_PARAMETERS)
                or not active_h and any(p.get(key) is not None for key in INACTIVATION_PARAMETERS)):
            raise CandidateValidationError(KIND_REFUSAL, "kind")
        if not active_h:
            p.update({key: None for key in INACTIVATION_PARAMETERS})
    elif p:
        raise CandidateValidationError(KIND_REFUSAL, "kind")
    windows = constraints["parameter_windows"]
    for key, value in p.items():
        if value is None and p.get("hvh_mV") is None and key in ("hvh_mV", "hk_mV", "htau_ms", "hpow"):
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or key not in windows
                or not windows[key][0] <= value <= windows[key][1]
                or key in ("mk_mV", "hk_mV") and value == 0
                or key in ("mtau_ms", "htau_ms") and value <= 0):
            raise CandidateValidationError(WINDOW_REFUSAL, "numeric")
        if key in ("mpow", "hpow") and (not isinstance(value, int) or value not in (1, 2)):
            raise CandidateValidationError(POWER_REFUSAL, "numeric")
    if not constraints["structure_expansion"] and mechanism not in constraints["initial_mechanisms"]:
        raise CandidateValidationError(S0_REFUSAL_REASON, "cell")
    candidate = NeuronHypothesis(mechanism, kind, tuple(sorted(p.items())))
    if candidate.model_id in known_ids:
        raise CandidateValidationError(DUPLICATE_REFUSAL, "duplicate", candidate=candidate,
                                       model_id=candidate.model_id)
    finiteness_probe(candidate, initial_designs, feature_space)
    # Extra root fields never enter the scientific identity.
    ignored = {key: original[key] for key in original.keys() - {"mechanism", "kind", "parameters", "note"}}
    meta = {"ignored_keys": mask_secrets_deep(ignored)}
    note = original.get("note")
    if isinstance(note, str) and len(note) <= 200:
        meta["note"] = mask_secrets(note)
    elif "note" in original:
        meta["ignored_keys"]["note"] = mask_secrets_deep(note)
    return candidate, meta


class NeuronProposer:
    """LLM mechanism-proposal transport for free-structure cells: one request per round, with
    repair turns, a paid-call guard and write-once request/response records under `records`."""

    def __init__(self, records, *, world, cell, provider="anthropic", allow_live=False,
                 model=None, client=None, repair_attempts=1, max_rounds=5,
                 known_ids=None, archive=None, base_url=None, key_name="OPENAI_COMPAT_API_KEY",
                 provider_order=None, reasoning=None, max_tokens=None, max_paid_calls=None,
                 feature_space=DEFAULT_FEATURE_SPACE, accept_code_fence=False, proposal_limit=1):
        self.mode = structure_mode(cell)
        # Candidates asked for per call; 1 is the single-candidate transport, byte for byte.
        self.proposal_limit = validate_proposal_limit(proposal_limit)
        if self.mode != "free":
            raise ValueError("S0 forbids constructing a neuron proposer")
        if provider not in ("anthropic", "openai", "openai_compat"):
            raise ValueError("provider must be anthropic, openai or openai_compat")
        self.compat_metadata = {}
        self.provider_order = normalize_provider_order(provider_order)
        # Reasoning is accepted on the compatibility endpoint and on Anthropic (validated below),
        # never on plain OpenAI; the reply budget is statable on every provider and stays 1600 unstated.
        self.reasoning = normalize_reasoning(reasoning)
        self.max_tokens = 1600 if max_tokens is None else parse_max_tokens(max_tokens)
        if provider == "openai_compat":
            host = validate_openai_compat(base_url, key_name, model)
            self.compat_metadata = {"llm_base_url_host": host, "llm_key_name": key_name}
            if self.provider_order is not None:
                self.compat_metadata["provider_order"] = self.provider_order
        elif (base_url is not None or key_name != "OPENAI_COMPAT_API_KEY"
                or provider_order is not None or (reasoning is not None and provider != "anthropic")):
            raise ValueError("compatibility options require provider openai_compat")
        elif provider == "anthropic":
            # Anthropic reasoning: `effort=<level>[,thinking=adaptive[,display=summarized]]` goes out as
            # `output_config.effort` and `thinking`, exactly as the count transport sends it;
            # unstated, the request keeps its bytes.
            self.reasoning = validate_provider_reasoning("anthropic", self.reasoning)
        # Paid-call cap (the same guard as the count transport's): None is no guard.
        self.max_paid_calls = None if max_paid_calls is None else parse_max_paid_calls(max_paid_calls)
        # Code-fence acceptance is opt-in; False keeps the request, the parse and every record byte-identical.
        if not isinstance(accept_code_fence, bool):
            raise ValueError("accept_code_fence must be a boolean")  # noqa: TRY004 - tested as ValueError
        self.accept_code_fence = accept_code_fence
        self.repair_attempts = _validate_repair_attempts(repair_attempts)
        self.max_rounds = _validate_max_rounds(max_rounds)
        self.records, self.world, self.cell, self.provider = Path(records), world, cell, provider
        self.model = model or (DEFAULT_MODEL if provider == "anthropic" else DEFAULT_MODEL_OPENAI)
        self.injected_client = client is not None
        self.attempts = self.paid_api_calls = 0
        self.known_ids = set() if known_ids is None else known_ids
        self.feature_space = feature_space
        self.archive = [] if archive is None else archive
        self._lock = threading.Lock()
        if client is None:
            if not allow_live:
                name = "Anthropic" if provider == "anthropic" else "OpenAI"
                raise ValueError(f"{name} proposals require explicit --allow-paid-api authorization")
            try:
                install_secret_log_filter()
                if provider == "anthropic":
                    import anthropic
                    client = anthropic.Anthropic(max_retries=0, timeout=120.0)
                else:
                    import openai
                    client = (openai.OpenAI(**compat_client_kwargs(base_url, key_name, model))
                              if provider == "openai_compat" else
                              openai.OpenAI(max_retries=0, timeout=120.0))
            except BaseException as exc:  # noqa: BLE001 - Preserve response and transport failure handling with masked diagnostics.
                raise masked_exception(exc) from None
        self.client = client
        self.records.mkdir(exist_ok=False)

    def propose_one(self, packet_text, *, context):
        """Request one proposal for a serialized belief packet and a run/world/cell/round context.

        Returns (candidate, record); at k > 1 the candidate is the list of parsed candidates."""
        with self._lock:
            return self._propose_one(packet_text, context)

    def _propose_one(self, packet_text, context):
        keys = {"run_id", "world_seed", "cell", "proposal_rep", "round"}
        if (set(context) != keys or context["cell"] != self.cell
                or isinstance(context["round"], bool) or not isinstance(context["round"], int)
                or not 1 <= context["round"] <= self.max_rounds):
            raise ValueError("explicit run/world/cell/proposal-replicate/round identity required")
        if not isinstance(packet_text, str) or not packet_text:
            raise ValueError("belief packet text required")
        packet_text = mask_secrets(packet_text)
        packet = json.loads(packet_text)
        constraints = packet["constraints"]
        if constraints["structure_expansion"] is not True or constraints["family_expansion"] is not False:
            raise ValueError("packet must declare structure expansion and no family expansion")
        observations, protocols = packet["observations"], packet["interface"]["protocols"]
        if len(observations) < 2:
            raise ValueError("two initial public observations required before proposal")
        designs = [protocols[row["design_index"]] for row in observations[:2]]
        for row in packet["pool"] + packet["archive"]:
            self.known_ids.add(row["model_id"])
        prefix = self.records / f"round_{context['round']:02d}"
        request_path, response_path = prefix.with_suffix(".request.json"), prefix.with_suffix(".response.json")
        if request_path.exists() or response_path.exists():
            raise FileExistsError("proposal round record already exists")
        identity = {**context, **self.compat_metadata, "provider": self.provider, "requested_model": self.model,
                    "policy": "P1", "decoding": decoding_identity(
                        self.max_tokens, self.reasoning,
                        **({"provider": "anthropic"} if self.provider == "anthropic" else {})),
                    **neuron_prompt_identity(), **packet_digest(packet_text), "world": self.world,
                    "mechanism_vocabulary_sha256": packet_digest(serialize_packet(
                        constraints["allowed_mechanisms"]))["packet_sha256"],
                    "recovery_mode": RECOVERY_MODES[self.mode]}
        if self.accept_code_fence:
            # Code-fence policy: named only when the option is on, so a default record keeps its bytes.
            identity["reply_code_fence"] = CODE_FENCE_POLICY
        if self.feature_space != DEFAULT_FEATURE_SPACE:
            # Named only when the run trains on another observation feature space, so a
            # three-feature record keeps its exact earlier bytes.
            identity["feature_space"] = {"record_key": self.feature_space.record_key,
                                         "names": list(self.feature_space.names)}
        system, composed = composed_system_prompt(self.proposal_limit)
        if composed:
            # Multi-candidate identity: named only at k > 1 (proposal_limit, composed_prompt_sha256,
            # composition_paragraph_sha256), so a k = 1 record keeps its bytes.
            identity.update(composed)
        record = {**identity, "status": "not_returned", "returned_model": None,
                  **(compat_response_metadata() if self.provider == "openai_compat" else {}),
                  "api_request_attempted": False, "client_injected": self.injected_client,
                  "sdk_retries": 0, "temperature_sent": False, "calls": 0, "repairs": 0,
                  "rejected_candidates": [], "responses": [], "archive_entries": []}
        messages = [{"role": "user", "content": mask_secrets(packet_text)}]
        bodies, candidate = [], None
        budget_stop = None
        started = time.perf_counter()
        for attempt in range(1 + self.repair_attempts):
            body = {"model": self.model, "max_tokens": self.max_tokens}
            if self.provider == "anthropic":
                body.update(system=system, messages=list(messages))
                if self.reasoning is not None:
                    body["output_config"] = {"effort": self.reasoning["effort"]}
                    if "thinking" in self.reasoning:
                        body["thinking"] = {"type": self.reasoning["thinking"]}
                        if "display" in self.reasoning:
                            body["thinking"]["display"] = self.reasoning["display"]
            else:
                body["messages"] = [{"role": "system", "content": system}, *messages]
                apply_extra_body(body, self.provider_order, self.reasoning)
            # Paid-call cap: counted, or refused before the client is called. A refused first call
            # writes nothing; a refused repair turn keeps the paid call's record below.
            try:
                _reserve_paid_call(self)
            except PaidCallBudgetExhausted as exc:
                if not attempt:
                    raise
                budget_stop = exc
                break
            bodies.append(body)
            self.paid_api_calls += int(not self.injected_client)
            record["calls"] += 1
            record["repairs"] += int(attempt > 0)
            record["api_request_attempted"] = True
            persisted, usage_known, raw_text = False, False, None
            try:
                if self.provider == "anthropic":
                    response = create_anthropic_message(self.client, body)
                    raw_text = mask_secrets("\n".join(b.text for b in response.content if b.type == "text"))
                    stop, expected_stop = response.stop_reason, "end_turn"
                    usage, usage_known = _anthropic_usage(getattr(response, "usage", None))
                else:
                    response = create_openai_completion(
                        self.client, body, capture_http_request=self.provider == "openai_compat")
                    raw_text = mask_secrets(response.choices[0].message.content)
                    stop, expected_stop = response.choices[0].finish_reason, "stop"
                    usage, usage_known = _openai_usage(getattr(response, "usage", None))
                if raw_text is not None and not isinstance(raw_text, str):
                    raise TypeError("provider response text must be a string")
                record["responses"].append({"attempt": attempt, "returned_model": response.model,
                    **(compat_response_metadata(response) if self.provider == "openai_compat" else {}),
                    "response_id": response.id, "http_request_id": _http_request_id(response),
                    "stop_reason": stop, "usage": usage, "usage_known": usage_known, "raw_text": raw_text,
                    # Thinking blocks are counted under a stated reasoning spec and
                    # their summaries recorded when the reply carried any; never parsed.
                    **({"thinking_blocks": _thinking_block_count(response)}
                       if self.provider == "anthropic" and self.reasoning is not None else {}),
                    **(anthropic_thinking_record(response) if self.provider == "anthropic" else {})})
                record["returned_model"] = response.model
                if self.provider == "openai_compat":
                    record.update(compat_response_metadata(response))
                persisted = True
                # Empty-reply rule: an empty body is its own reply-contract failure,
                # whatever the finish/stop reason says.
                if blank_reply(raw_text):
                    raise EmptyResponseBody(stop)
                if stop != expected_stop:
                    raise ValueError(f"response did not finish with {expected_stop}")
                if self.accept_code_fence:
                    # Code-fence policy: which replies were parsed from inside a fence, on the response itself.
                    record["responses"][-1]["code_fence_stripped"] = strip_outer_code_fence(raw_text)[1]
                if self.proposal_limit == 1:
                    candidate, meta = parse_neuron_candidate(raw_text, constraints=constraints,
                                                            initial_designs=designs, known_ids=self.known_ids,
                                                            feature_space=self.feature_space,
                                                            accept_code_fence=self.accept_code_fence)
                else:
                    candidate, meta = self._parse_array(raw_text, constraints, designs, context, packet, record)
            except Exception as exc:  # noqa: BLE001 - Preserve response and transport failure handling with masked diagnostics.
                # A failed request still has unknown usage, never a complete zero.
                if not persisted:
                    record["responses"].append({"attempt": attempt, "returned_model": None,
                        **(compat_response_metadata() if self.provider == "openai_compat" else {}),
                        "response_id": None, "http_request_id": None, "stop_reason": None,
                        "usage": None, "usage_known": False, "raw_text": raw_text})
                reason = mask_secrets(str(exc))
                eligible = persisted and usage_known and isinstance(exc, ValueError)
                rejected = {"attempt": attempt, "reason": reason,
                    "error_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None),
                    "repair_eligible": eligible, "stage": getattr(exc, "stage", "transport")}
                if eligible:
                    rejected["repair_kind"] = repair_kind(exc)
                record["rejected_candidates"].append(rejected)
                if isinstance(exc, CandidateValidationError) and exc.model_id:
                    is_duplicate = exc.stage == "duplicate"
                    mechanism = (exc.candidate.mechanism if isinstance(exc.candidate, NeuronHypothesis)
                                 else exc.candidate.get("mechanism"))
                    entry = {"model_id": exc.model_id, "mechanism": mechanism,
                             "proposed_round": context["round"],
                             "outcome": "duplicate" if is_duplicate else "rejected",
                             "reason": ({"matched_model_id": exc.model_id,
                                         "matched_expression": exc.candidate.expression}
                                        if is_duplicate else {"validation_message": reason})}
                    if is_duplicate and any(row["model_id"] == exc.model_id
                            and row["outcome"] == "retired_by_evidence" for row in packet["archive"] + self.archive):
                        entry["note"] = "refused: matched a model retired by evidence"
                    entry = mask_secrets_deep(entry)
                    self.archive.append(entry)
                    self.known_ids.add(exc.model_id)
                    record["archive_entries"].append(entry)
                if not eligible:
                    break
                if self.proposal_limit == 1:
                    instruction, empty_instruction = REPAIR_INSTRUCTION, REPAIR_INSTRUCTION_EMPTY
                else:
                    # Multi-candidate repair: the array shape, with k written in; {error}/{finish_reason} stay for
                    # repair_messages to fill.
                    limit = str(self.proposal_limit)
                    instruction = REPAIR_INSTRUCTION_ARRAY.replace("{limit}", limit)
                    empty_instruction = REPAIR_INSTRUCTION_EMPTY_ARRAY.replace("{limit}", limit)
                messages = repair_messages(messages, exc, raw_text, reason,
                                           instruction=instruction, empty_instruction=empty_instruction)
                continue
            if self.proposal_limit == 1:
                self.known_ids.add(candidate.model_id)
            record.update(meta, status="parsed")
            break
        if budget_stop is not None:
            # Paid-call cap: the paid call(s) this proposal made stay recorded, under a stop status.
            record.update(status=BUDGET_EXHAUSTED, max_paid_calls=budget_stop.max_paid_calls)
        elif candidate is None:
            record["status"] = "proposal_failed"
        totals, complete, missing = _usage_total(record["responses"])
        record.update(usage_total=totals, usage_total_complete=complete,
                      usage_missing_indices=missing, elapsed_seconds=time.perf_counter()-started,
                      valid_candidate_count=(len(candidate) if isinstance(candidate, list)
                                             else int(candidate is not None)),
                      silent_fallback=False,
                      paid_api_calls=0 if self.injected_client else record["calls"])
        _write_once(request_path, {"identity": identity, "bodies": bodies,
                                   "client_injected": self.injected_client})
        record = mask_secrets_deep(record)
        _write_once(response_path, record)
        if budget_stop is not None:
            raise budget_stop
        # At k > 1 the first value is the list of parsed candidates (None when none).
        return candidate, record

    def _parse_array(self, raw_text, constraints, designs, context, packet, record):
        """A k > 1 reply is a JSON array of 1..k candidate objects, each validated exactly
        as a k = 1 reply is (`parse_neuron_candidate`); the valid ones are returned, a failing one is
        recorded under rejected_candidates (with its array index) and, when the refusal carries a
        model id (vocabulary, duplicate or finiteness), in the archive, as the k = 1 path records
        its failure. No valid candidate at all raises, so the
        repair turn runs as for a k = 1 failure."""
        text = strip_outer_code_fence(raw_text)[0] if self.accept_code_fence else raw_text
        try:
            payload = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise CandidateValidationError(JSON_REFUSAL.format(error=exc), "json") from None
        if not isinstance(payload, list) or not 1 <= len(payload) <= self.proposal_limit:
            raise CandidateValidationError(ARRAY_REFUSAL.format(limit=self.proposal_limit), "shape")
        candidates, metas, known = [], [], set(self.known_ids)
        for index, item in enumerate(payload):
            try:
                candidate, meta = parse_neuron_candidate(json.dumps(item), constraints=constraints,
                                                        initial_designs=designs, known_ids=known,
                                                        feature_space=self.feature_space)
            except CandidateValidationError as exc:
                reason = mask_secrets(str(exc))
                record["rejected_candidates"].append({"attempt": record["calls"] - 1, "array_index": index,
                                                      "reason": reason, "error_type": type(exc).__name__,
                                                      "http_status": None, "repair_eligible": False,
                                                      "stage": exc.stage})
                if exc.model_id:
                    is_duplicate = exc.stage == "duplicate"
                    mechanism = (exc.candidate.mechanism if isinstance(exc.candidate, NeuronHypothesis)
                                 else exc.candidate.get("mechanism"))
                    entry = {"model_id": exc.model_id, "mechanism": mechanism,
                             "proposed_round": context["round"],
                             "outcome": "duplicate" if is_duplicate else "rejected",
                             "reason": ({"matched_model_id": exc.model_id,
                                         "matched_expression": exc.candidate.expression}
                                        if is_duplicate else {"validation_message": reason})}
                    if is_duplicate and any(row["model_id"] == exc.model_id
                            and row["outcome"] == "retired_by_evidence" for row in packet["archive"] + self.archive):
                        entry["note"] = "refused: matched a model retired by evidence"
                    entry = mask_secrets_deep(entry)
                    self.archive.append(entry)
                    known.add(exc.model_id)
                    record["archive_entries"].append(entry)
                continue
            known.add(candidate.model_id)
            candidates.append(candidate)
            metas.append(meta)
        self.known_ids.update(known)
        if not candidates:
            raise CandidateValidationError(NO_VALID_ARRAY_CANDIDATE, "array")
        return candidates, {"array_length": len(payload), "candidate_metas": metas,
                            "candidate_model_ids": [c.model_id for c in candidates]}


class AnthropicNeuronProposer(NeuronProposer):
    """NeuronProposer with the provider fixed to Anthropic."""

    def __init__(self, records, **kwargs):
        super().__init__(records, provider="anthropic", **kwargs)


class OpenAINeuronProposer(NeuronProposer):
    """NeuronProposer with the provider fixed to OpenAI."""

    def __init__(self, records, **kwargs):
        super().__init__(records, provider="openai", **kwargs)


def make_neuron_proposer(provider, records, *, cell, **kwargs):
    """The caller uses this gate before building any client or writing any packet.

    Returns None for a catalog (S0) cell, else a NeuronProposer for `provider`."""
    if structure_mode(cell) == "catalog":
        return None
    return NeuronProposer(records, provider=provider, cell=cell, **kwargs)
