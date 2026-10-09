"""Explicitly enabled Anthropic transport; no scientific/evaluation imports."""

import hashlib
import inspect
import json
import logging
import math
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit

from .count_models import (
    ARRAY_REFUSAL_PREFIX,
    FAMILIES,
    PRIOR_RULES,
    Hypothesis,
    failure_class,
    initial_hypotheses,
    judge_candidate,
    llm_priors_enabled,
    validate_proposed_priors,
)
from .prior_probe_detail import PriorProbeRejection

DEFAULT_MODEL = "claude-opus-4-7"
# OpenAI is the tests/smoke/development transport only;
# main runs stay on DEFAULT_MODEL via Anthropic. Never used as
# a silent fallback - the runner FORCES label "development" whenever this
# provider is selected (core_count.run/boxinggym_core.run), so an OpenAI
# transport record can never be mistaken for a main-run result.
DEFAULT_MODEL_OPENAI = "gpt-4o"


_COMPAT_SECRET_NAMES = set()
_COMPAT_RESERVED_ENV_NAMES = frozenset([
    'HOME', 'PATH', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'PYTENSOR_FLAGS', 'JAX_PLATFORMS',
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'LLM_PROVIDER', 'LLM_MODEL', 'LLM_MODEL_EFFECTIVE', 'LLM_BASE_URL', 'LLM_BASE_URL_HOST',
    'LLM_KEY_NAME', 'LLM_PROVIDER_ORDER', 'LLM_PROVIDER_ORDER_SET', 'NEURONBENCH_SRC',
    'DP_SCIENCEAGENT_DIR', 'FB1_DATA_DIR', 'FB1_WORLD', 'FB1_ARM', 'FB1_SEED', 'FB1_LIST',
    'FB1_T_EXTRA', 'FB1_CHECKPOINTS', 'FB1_INTEGRATOR', 'FB1_DESIGN_GATE', 'FB1_D0',
    'FB1_PROPOSER', 'FB1_LLM_PROVIDER', 'FB1_LLM_MODEL', 'FB1_LLM_BASE_URL', 'FB1_LLM_KEY_NAME',
    'FB1_LLM_PROVIDER_ORDER', 'FB1_ALLOW_PAID_API', 'FB1_STRUCTURE_PROPOSALS', 'FB1_RUN_LABEL',
    'FB1_OUT_DIR', 'BMD_PYTHON', 'FB1_PYTHON', 'FB1_DIR', 'REPO_ROOT', 'BMD_BOXINGGYM_SRC',
    'BMD_AUTOSCILAB', 'CHEMBENCH_MANIFEST', 'PROFILE', 'PTC_DIR', 'ROUNDS_BOXINGGYM',
    'ROUNDS_CHEMBENCH', 'ROUNDS_CORE', 'ROUNDS_NEURON', 'selected_key', 'key_to_clear',
    'effective_proposer',
])



def validate_openai_compat(base_url, key_name, model):
    """Validate explicit transport settings without echoing caller input."""
    if not isinstance(key_name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_name):
        raise ValueError("llm-key-name must be an environment variable name")
    if key_name in _COMPAT_RESERVED_ENV_NAMES:
        raise ValueError("llm-key-name must not name a runtime setting")
    _COMPAT_SECRET_NAMES.add(key_name)
    if (not isinstance(model, str) or not model.strip() or model != model.strip()
            or any(ord(c) < 32 or ord(c) == 127 for c in model) or "<" in model or ">" in model
            or model.casefold() in {"name", "model", "your_model", "your-model", "todo", "tbd", "placeholder"}):
        raise ValueError("openai_compat requires an explicit actual llm-model")
    try:
        if not isinstance(base_url, str) or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in base_url):
            raise ValueError
        parsed = urlsplit(base_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment
                or "?" in base_url or "#" in base_url):
            raise ValueError
        parsed.port  # noqa: B018 - Access validates the port and raises ValueError for invalid values.
    except (ValueError, TypeError):
        raise ValueError("llm-base-url must be HTTPS with a host and no credentials, query or fragment") from None
    return parsed.hostname


def compat_client_kwargs(base_url, key_name, model):
    """Validate the compatibility settings and return the OpenAI SDK constructor keyword arguments
    (base URL, the key read from the named environment variable, no SDK retries, 120 s timeout)."""
    validate_openai_compat(base_url, key_name, model)
    key = os.environ.get(key_name)
    if not key:
        raise ValueError("openai_compat credential environment variable is empty")
    return {"base_url": base_url, "api_key": key, "max_retries": 0, "timeout": 120.0}


def normalize_provider_order(value):
    """Normalize an explicit compatible-provider order without echoing input."""
    if value is None:
        return None
    items = value.split(",") if isinstance(value, str) else value
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("llm-provider-order must contain nonempty distinct provider names")
    order = []
    for item in items:
        if (not isinstance(item, str) or "," in item
                or any(ord(c) < 32 or 127 <= ord(c) <= 159 or c in "\u2028\u2029" for c in item)):
            raise ValueError("llm-provider-order contains an invalid provider name")
        name = item.strip()
        if not name or name in order:
            raise ValueError("llm-provider-order must contain nonempty distinct provider names")
        order.append(name)
    return order


# The OpenAI-compatible `reasoning` extension. `effort` and
# `max_tokens` are alternatives, never both (provider documentation read
# 2026-09-17), and `exclude` only says whether the reasoning text comes back -
# it is billed and counted against the reply budget either way. This is the
# ONE parser for the package; the ForceBench snapshot is a separate file and
# implements the same grammar, with a test pinning the two to equal results.
REASONING_EFFORTS = ("max", "xhigh", "high", "medium", "low", "minimal", "none")
#: The levels the Anthropic Messages API takes in `output_config.effort`
#: (platform.claude.com/docs/en/build-with-claude/effort); `none` and `minimal` are not levels
#: there. The one value `thinking=` takes is `adaptive` (`thinking: {"type": "adaptive"}`).
ANTHROPIC_EFFORTS = ("max", "xhigh", "high", "medium", "low")
THINKING_MODES = ("adaptive",)
THINKING_DISPLAYS = ("summarized",)
REASONING_FIELD_ERROR = (f"llm-reasoning takes effort=<{'|'.join(REASONING_EFFORTS)}>, "
                         "max_tokens=<positive integer> and the optional flag exclude"
                         " (and, for provider anthropic, thinking=adaptive with optional display=summarized)")
MAX_TOKENS_LIMIT = 32768


def parse_reasoning_spec(text):
    """`effort=<level>` XOR `max_tokens=<n>`, plus optional `exclude`, plus the
    optional `thinking=adaptive`, with optional `display=summarized`.

    Comma separated, order free, no repeats. The returned dict is canonically
    ordered (the alternative first, `exclude`, `thinking`, then `display`) so the same request is the
    same bytes however the option was typed. Raises ValueError - the runner
    CLIs pass this as an argparse `type`, so a bad spec ends the command line
    before any output directory exists. Which fields a provider's transport can
    send is `validate_provider_reasoning`'s question, asked where the proposer is built.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(REASONING_FIELD_ERROR)
    fields = {}
    for field in text.split(","):
        key, separator, value = field.strip().partition("=")
        key, value = key.strip(), value.strip()
        if key in fields:
            raise ValueError("llm-reasoning must not repeat a field")
        if key == "exclude" and not separator:
            fields[key] = True
        elif key == "effort" and separator:
            if value not in REASONING_EFFORTS:
                raise ValueError("llm-reasoning effort must be one of " + "|".join(REASONING_EFFORTS))
            fields[key] = value
        elif key == "max_tokens" and separator:
            if re.fullmatch(r"[0-9]+", value) is None or int(value) <= 0:
                raise ValueError("llm-reasoning max_tokens must be a positive integer")
            fields[key] = int(value)
        elif key == "thinking" and separator:
            if value not in THINKING_MODES:
                raise ValueError("llm-reasoning thinking must be " + "|".join(THINKING_MODES))
            fields[key] = value
        elif key == "display" and separator:
            if value not in THINKING_DISPLAYS:
                raise ValueError("llm-reasoning display must be " + "|".join(THINKING_DISPLAYS))
            fields[key] = value
        else:
            raise ValueError(REASONING_FIELD_ERROR)
    if ("effort" in fields) == ("max_tokens" in fields):
        raise ValueError("llm-reasoning requires exactly one of effort= and max_tokens=")
    if "thinking" in fields and "effort" not in fields:
        raise ValueError("llm-reasoning thinking=adaptive rides with an effort level")
    if "display" in fields and "thinking" not in fields:
        raise ValueError("llm-reasoning display=summarized requires thinking=adaptive")
    spec = {"effort": fields["effort"]} if "effort" in fields else {"max_tokens": fields["max_tokens"]}
    if fields.get("exclude"):
        spec["exclude"] = True
    if "thinking" in fields:
        spec["thinking"] = fields["thinking"]
    if "display" in fields:
        spec["display"] = fields["display"]
    return spec


def normalize_reasoning(value):
    """A stated spec (text or an already parsed dict) as one canonical dict.

    None is "not stated": the request body and the recorded identity keep the
    bytes they had before this option existed.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return parse_reasoning_spec(value)
    if not isinstance(value, dict) or not value:
        raise ValueError(REASONING_FIELD_ERROR)
    # Re-rendered through the one parser, so a dict can never take a shortcut
    # past the grammar the command line has to obey.
    fields = []
    for key, item in value.items():
        if key == "exclude":
            if item is not True:
                raise ValueError(REASONING_FIELD_ERROR)
            fields.append("exclude")
        else:
            fields.append(f"{key}={item}")
    return parse_reasoning_spec(",".join(fields))


def validate_provider_reasoning(provider, spec):
    """The stated reasoning spec a provider's transport can send, or a ValueError
    that says why not.

    openai: `effort=<level>` only - it is sent as `reasoning_effort` and the reply budget as
    `max_completion_tokens` (the reasoning-model shape); `max_tokens=` and `exclude` are
    OpenRouter extensions of the compatibility endpoint and `thinking=` is Anthropic's field.
    anthropic: `effort=<low|medium|high|xhigh|max>` (sent as `output_config.effort`) plus the
    optional `thinking=adaptive` (sent as `thinking: {"type": "adaptive"}`); `none` and
    `minimal` are not Anthropic effort levels. openai_compat: the D1 grammar as before,
    without `thinking=`. None (not stated) passes through: the request keeps its pre-RM bytes.
    """
    if spec is None:
        return None
    if provider in ("openai", "openai_compat") and "display" in spec:
        raise ValueError(f"display= is an Anthropic field; provider {provider} does not accept it")
    if provider == "openai":
        if set(spec) != {"effort"}:
            raise ValueError("provider openai takes llm-reasoning effort=<level> only: max_tokens= "
                             "and exclude are openai_compat (OpenRouter) extensions and thinking= "
                             "is Anthropic's field")
    elif provider == "anthropic":
        if "max_tokens" in spec or "exclude" in spec:
            raise ValueError("provider anthropic takes llm-reasoning effort=<level>[,thinking=adaptive]: "
                             "max_tokens= and exclude are openai_compat (OpenRouter) extensions")
        if spec["effort"] not in ANTHROPIC_EFFORTS:
            raise ValueError("provider anthropic takes effort=<" + "|".join(ANTHROPIC_EFFORTS)
                             + ">; none and minimal are not Anthropic effort levels")
    elif provider == "openai_compat":
        if "thinking" in spec:
            raise ValueError("thinking= is an Anthropic field; provider openai_compat takes "
                             "effort=<level> or max_tokens=<n>, plus the optional exclude")
    else:
        raise ValueError("unsupported provider for llm-reasoning")
    return spec


def _thinking_block_count(response):
    """How many thinking blocks (summarized or redacted) an Anthropic reply carried.
    Only the text blocks are parsed; this count is what the record keeps of the rest."""
    content = getattr(response, "content", None) or []
    return sum(1 for block in content if getattr(block, "type", None) in ("thinking", "redacted_thinking"))


def anthropic_thinking_record(response):
    """What an Anthropic reply's thinking blocks said, for the record only.

    `thinking_text` is the summarized thinking of every `thinking` block, in reply order, each
    through `mask_secrets`; `redacted_thinking_blocks` is the count of `redacted_thinking`
    blocks and exists only when there is one. A reply with no block of either kind gives an
    empty mapping, so the record of a run without thinking keeps every byte it had. Nothing
    here is parsed, judged or re-sent: the repair turn echoes the text blocks only, and the
    request bodies do not change."""
    content = getattr(response, "content", None) or []
    texts = [mask_secrets(str(getattr(block, "thinking", ""))) for block in content
             if getattr(block, "type", None) == "thinking"]
    redacted = sum(1 for block in content if getattr(block, "type", None) == "redacted_thinking")
    if not texts and not redacted:
        return {}
    return {"thinking_text": texts, **({"redacted_thinking_blocks": redacted} if redacted else {})}


#: The per-cell paid-call guard. A runner given `--max-paid-calls N` hands N to its
#: proposer, which counts every paid request (repair turns included) and refuses the one that
#: would make it N + 1 before the client is called. The runner then writes summary.json with
#: `stop_reason` "budget_exhausted" and exits with this status: distinct from 0/1/2 (success,
#: an uncaught exception, a refusal), from `timeout`'s 124-127 and from 128+ signal statuses,
#: so the smoke driver records the cell as failed while still reading its summary's calls.
PAID_CALL_BUDGET_EXIT_CODE = 4
BUDGET_EXHAUSTED = "budget_exhausted"
MAX_PAID_CALLS_ERROR = "max-paid-calls must be an integer of at least 1"


class PaidCallBudgetExhausted(BaseException):
    """The next paid request would exceed this cell's `max_paid_calls`.

    Raised at the request site before the attempt counter moves and before the client is
    called, so `attempts` never passes `max_paid_calls`. A BaseException on purpose: no
    `except Exception` boundary - a candidate failure, a failed burst call, a repair rule -
    can read it as something the cell survives; only the runner's own handler stops on it.
    """

    def __init__(self, attempts, max_paid_calls):
        super().__init__(f"paid-call budget exhausted: {attempts} of {max_paid_calls} paid "
                         "calls made; the next call would exceed max_paid_calls")
        self.attempts = attempts
        self.max_paid_calls = max_paid_calls


def parse_max_paid_calls(value):
    """A stated per-cell paid-call cap: an integer of at least 1 (text or int).

    The runner CLIs pass this as an argparse `type` with `argparse.SUPPRESS`, so an omitted
    option is absent from the namespace, the proposer gets no guard and every record keeps
    its bytes.
    """
    if isinstance(value, str):
        if re.fullmatch(r"[0-9]+", value.strip()) is None:
            raise ValueError(MAX_PAID_CALLS_ERROR)
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(MAX_PAID_CALLS_ERROR)
    return value


def _reserve_paid_call(proposer):
    """Count one paid request, or refuse it, under the one attempt lock.

    The check and the increment share `_ATTEMPT_LOCK`, so parallel burst calls can never
    pass the cap together. No cap (None) is the counter exactly as it always was.
    """
    with _ATTEMPT_LOCK:
        limit = getattr(proposer, "max_paid_calls", None)
        if limit is not None and proposer.attempts + 1 > limit:
            raise PaidCallBudgetExhausted(proposer.attempts, limit)
        proposer.attempts += 1


def budget_exhausted_summary(proposer, stopped_round):
    """The keys each runner's summary.json carries when its guard stops the cell.

    `paid_api_calls` is the proposer's own counter, which the guard holds at
    `max_paid_calls`. The stopped round's partial records stay on disk beside every
    earlier round's complete ones; `last_completed_round` is None when round 0 itself
    did not finish.
    """
    return {
        "status": BUDGET_EXHAUSTED,
        "stop_reason": BUDGET_EXHAUSTED,
        "paid_api_calls": proposer.attempts,
        "max_paid_calls": proposer.max_paid_calls,
        "stopped_round": stopped_round,
        "last_completed_round": stopped_round - 1 if stopped_round > 0 else None,
        "billing_status": "usage_in_llm_transport_records",
        "registered_run": False,
    }


def parse_max_tokens(value):
    """An explicitly stated reply budget, 1..32768 (text or int).

    Unstated is NOT this function's business: every transport keeps its own
    1600 literal when the option is absent, so an omitted flag leaves the
    request body and the record byte-identical.
    """
    if isinstance(value, str):
        if re.fullmatch(r"[0-9]+", value.strip()) is None:
            raise ValueError(f"llm-max-tokens must be an integer in 1..{MAX_TOKENS_LIMIT}")
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_TOKENS_LIMIT:
        raise ValueError(f"llm-max-tokens must be an integer in 1..{MAX_TOKENS_LIMIT}")
    return value


#: The one response format the stock OpenAI transport can ask for - JSON mode,
#: `response_format: {"type": "json_object"}` on every request. OpenAI requires the word "JSON"
#: somewhere in the messages when it is on, so the transport refuses, before any call, a
#: request whose system and user texts do not carry it. The compatibility endpoint and
#: Anthropic do not take it (Anthropic's structured outputs need a JSON schema with
#: `additionalProperties: false`, which our coefficient-keyed `parameters` map cannot state).
RESPONSE_FORMATS = ("json_object",)
JSON_MODE_WORD = "JSON"


def parse_response_format(value):
    """A stated response format (`json_object`), or None for not stated."""
    if value is None:
        return None
    if value not in RESPONSE_FORMATS:
        raise ValueError("llm-response-format must be one of " + "|".join(RESPONSE_FORMATS))
    return value


def decoding_identity(max_tokens, reasoning=None, *, provider=None, response_format=None):
    """The decoding declaration stored in every transport record. An option
    that was never stated leaves no key behind.

    With a stated reasoning spec the record also names what the request carried -
    `token_parameter` (the wire name of the reply budget: `max_completion_tokens` for provider
    openai, `max_tokens` for anthropic), `reasoning_effort` (openai) or `output_config` and, when
    stated, `thinking` (anthropic). openai_compat keeps exactly its original keys. A
    stated response format is recorded as the `response_format` object the request carried.
    """
    decoding = {"temperature": "not_sent/provider_default", "max_tokens": max_tokens}
    if reasoning is not None:
        decoding["reasoning"] = dict(reasoning)
        if provider == "openai":
            decoding["token_parameter"] = "max_completion_tokens"
            decoding["reasoning_effort"] = reasoning["effort"]
        elif provider == "anthropic":
            decoding["token_parameter"] = "max_tokens"
            decoding["output_config"] = {"effort": reasoning["effort"]}
            if "thinking" in reasoning:
                decoding["thinking"] = {"type": reasoning["thinking"]}
                if "display" in reasoning:
                    decoding["thinking"]["display"] = reasoning["display"]
    if response_format is not None:
        decoding["response_format"] = {"type": response_format}
    return decoding


def apply_extra_body(body, provider_order=None, reasoning=None):
    """Attach the stated OpenAI-compatible extensions - routing order and
    reasoning share the one `extra_body` object - and nothing else."""
    extra = {}
    if provider_order is not None:
        extra["provider"] = {"order": list(provider_order), "allow_fallbacks": False}
    if reasoning is not None:
        extra["reasoning"] = dict(reasoning)
    if extra:
        body["extra_body"] = extra
    return body


def compat_response_metadata(response=None):
    """The record field naming the upstream provider a compatible endpoint reports for `response`
    (`returned_provider`, None when absent or when no response exists yet)."""
    provider = getattr(response, "provider", None)
    return {"returned_provider": provider if isinstance(provider, str) else None}


def compat_config(args):
    """Native namespaces remain unchanged; compatible records contain only host/name."""
    values = vars(args).copy()
    provider = values.get("requested_proposer", values.get("proposer"))
    base_url, key_name = values.pop("llm_base_url", None), values.pop("llm_key_name", "OPENAI_COMPAT_API_KEY")
    supplied_order = "llm_provider_order" in values
    provider_order = values.pop("llm_provider_order", None)
    if provider == "openai_compat":
        values.update(llm_provider=provider, llm_base_url_host=validate_openai_compat(
            base_url, key_name, values.get("llm_model")), llm_key_name=key_name)
        if supplied_order:
            values["provider_order"] = normalize_provider_order(provider_order)
    elif "llm_base_url" in vars(args) or "llm_key_name" in vars(args) or supplied_order:
        raise ValueError("compatibility options require proposer openai_compat")
    # `llm_reasoning` (parsed spec) and `llm_max_tokens` stay exactly as stated:
    # an unstated option is argparse.SUPPRESS, so the config keeps no key at all.
    # The reasoning control is no longer a compatibility-only option - every LLM
    # transport takes it and validates it for its provider (`validate_provider_reasoning`);
    # a mock run makes no call, so a stated spec is still refused there, before any output.
    if provider == "mock" and "llm_reasoning" in values:
        raise ValueError("llm-reasoning requires an LLM proposer (a mock run makes no call)")
    # The same for a stated response format.
    if provider == "mock" and "llm_response_format" in values:
        raise ValueError("llm-response-format requires an LLM proposer (a mock run makes no call)")
    return values


def compat_points_metadata(config):
    """The compatibility-endpoint fields of a run config (provider, base URL host, key name), or
    an empty mapping when the run does not use provider openai_compat."""
    if config.get("llm_provider") != "openai_compat":
        return {}
    return {key: config[key] for key in ("llm_provider", "llm_base_url_host", "llm_key_name")}


def compat_option_kwargs(args, *, compat):
    """Transport options the command line actually stated, and nothing else.

    An unstated option is `argparse.SUPPRESS`, so it is absent from the
    namespace, absent here, and absent from the constructor call - which is
    what keeps an omitted flag byte-identical all the way to the request.
    `max_tokens` is a decoding setting every provider takes; routing order is a
    compatibility-endpoint extension. `max_paid_calls` is the per-cell
    paid-call guard every provider takes. The reasoning control reaches every
    LLM transport, which validates it for its provider (`validate_provider_reasoning`).
    """
    kwargs = {"max_tokens": args.llm_max_tokens} if hasattr(args, "llm_max_tokens") else {}
    if hasattr(args, "max_paid_calls"):
        kwargs["max_paid_calls"] = args.max_paid_calls
    if hasattr(args, "llm_reasoning"):
        kwargs["reasoning"] = args.llm_reasoning
    # The response format likewise reaches the transport, which takes or refuses it.
    if hasattr(args, "llm_response_format"):
        kwargs["response_format"] = args.llm_response_format
    if compat and hasattr(args, "llm_provider_order"):
        kwargs["provider_order"] = args.llm_provider_order
    return kwargs


def _registered_decoding_equal(declared, stated):
    """Equality for a registered decoding setting - a Boolean is never a count."""
    if isinstance(stated, bool) or isinstance(declared, bool):
        return False
    if isinstance(stated, int):
        return type(declared) is int and declared == stated
    return isinstance(declared, dict) and declared == stated


def require_registered_decoding(config, receipt, bundle):
    """A production run may state a decoding option only if it was registered.

    `run_label_policy._registered_settings` compares a fixed list of keys, so
    the policy module does not bind an arbitrary one; the formal binding
    arrives with the registration unit that revises that file (and its source
    hash) on purpose. Until then the check lives HERE - in the runner, right
    after the label receipt and before any output directory exists - and
    refuses with exit status 2 when the manifest does not register exactly the
    stated `llm_reasoning` / `llm_max_tokens` for this bundle. A development
    run states whatever it likes; an unstated option checks nothing.
    """
    stated = {key: config[key] for key in ("llm_reasoning", "llm_max_tokens") if key in config}
    if not stated or receipt.get("run_label") != "production":
        return
    path, settings = receipt.get("manifest_path"), None
    if isinstance(path, str):
        try:
            data = Path(path).read_bytes()
            if hashlib.sha256(data).hexdigest() == receipt.get("manifest_sha256"):
                registered = json.loads(data).get("runtime_settings", {}).get(bundle)
                settings = registered if isinstance(registered, dict) else None
        except (OSError, ValueError):
            settings = None
    for key, value in stated.items():
        if not _registered_decoding_equal(None if settings is None else settings.get(key), value):
            print(f"ERROR --run-label production requires the registered {bundle} runtime setting "
                  f"{key} to equal the stated one", file=sys.stderr)
            raise SystemExit(2)


def sdk_call_kwargs(create, body):
    """The body as the installed SDK's `create` can take it - by name where it has
    the parameter, under `extra_body` where it does not.

    openai==1.40.3 (the hard pin, pyproject.toml) has no `max_completion_tokens` or
    `reasoning_effort` parameter, so the reasoning-model body raised
    `TypeError` inside the SDK before any HTTP request; the same keys given through
    `extra_body` land on the wire JSON's top level, which is what the record already says was
    sent. The split is decided from the signature at the call: a `create` that takes
    `**kwargs` (an injected fake client) gets the body exactly as before; a body whose keys are
    all parameters (every default-path request) is passed exactly as before, with no
    `extra_body` added; only a key the SDK does not name moves. A key stated both in the body
    and in its `extra_body` is refused before the call, and so is an SDK whose `create` takes
    neither the key nor `extra_body`. The recorded body is never touched - the split works on a
    copy."""
    try:
        parameters = inspect.signature(create).parameters
    except (TypeError, ValueError):
        return dict(body)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(body)
    unknown = {key: value for key, value in body.items() if key not in parameters}
    if not unknown:
        return dict(body)
    if "extra_body" not in parameters:
        raise ValueError(f"the installed SDK's create takes neither {sorted(unknown)} nor extra_body")
    kwargs = {key: value for key, value in body.items() if key in parameters}
    extra = dict(kwargs.get("extra_body") or {})
    clash = sorted(set(extra) & set(unknown))
    if clash:
        raise ValueError(f"request keys {clash} are stated both in the body and in extra_body")
    kwargs["extra_body"] = {**extra, **unknown}
    return kwargs


def create_anthropic_message(client, body):
    """The Messages API call with the same signature-decided split - anthropic 0.71.0
    (an older SDK some hosts run) has no `output_config` parameter, 0.120.2 (the pin) has it; both carry the
    key on the wire, by name or through `extra_body`."""
    return client.messages.create(**sdk_call_kwargs(client.messages.create, body))


def create_openai_completion(client, body, *, capture_http_request=False):
    """Capture the HTTP request id separately from the completion id on older SDKs.

    The call arguments come from `sdk_call_kwargs`, decided on the signature of the
    installed `create` (the raw-response wrapper forwards to the same method)."""
    completions = client.chat.completions
    kwargs = sdk_call_kwargs(completions.create, body)
    raw_api = getattr(completions, "with_raw_response", None) if capture_http_request else None
    if raw_api is None:
        return completions.create(**kwargs)
    raw = raw_api.create(**kwargs)
    response = raw.parse()
    response._request_id = raw.headers.get("x-request-id")
    return response

# Family variance/dispersion parameters: count_models.Hypothesis assigns these
# automatically per family (rho for beta_binomial, pi for zi_binomial) and
# reserves both names outright (see count_models.py's reserved-name check).
# Fail-soft contract: a candidate that mistakenly places one of these in
# `parameters` is not rejected for it - see parse_candidates.
# The round-identity window is the
# RUN's own horizon, handed in by the runner that owns `--rounds` (BoxingGym
# 20, ChemBench 60 since 2026-09-12), not a constant baked into the transport
# layer. The default is the earlier fixed literal, so a caller that does not pass
# `max_rounds` - `core_count.run` and `boxinggym_core.run`, whose own CLIs already refuse
# `--rounds` above 20 - keeps byte-identical behaviour AND a byte-identical
# transport record. `max_rounds` is deliberately NOT written into the record:
# it is a call-site bound, not an observation, and recording it would change
# every CORE/BoxingGym record for no new information; the runner records its
# own horizon in its own config.json instead (`chembench_core.run`'s
# `max_proposal_rounds`).
DEFAULT_MAX_PROPOSAL_ROUNDS = 20


def _validate_max_rounds(max_rounds):
    """A positive integer round horizon - never widened by a bad value.

    A `None`, 0, negative or float `max_rounds` is a runner wiring mistake,
    not a request for "no cap": it raises here, before any call, exactly like
    `_validate_repair_attempts`.
    """
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError("max_rounds must be a positive integer round horizon")
    return int(max_rounds)


# The OPTIONAL per-output-component identity of a round-based proposal.
# A task that declares several output components (only `emotion` today, eight
# of them - boxinggym_core.TASKS) calls `propose_one` once per component in
# the SAME round, and `_write_once` opens its records with mode "x", so the
# second component of a round used to die on FileExistsError. The component
# makes the record name unique; it is absent for every single-outcome task, so
# those keep the `round_NN` names and the exact identity key set they have
# always written.
#
# A DOT is deliberately NOT in the set below. The record paths are built with
# `Path.with_suffix(".request.json")`, which replaces everything after the LAST
# dot: `round_01__a.b` and `round_01__a.c` would both become
# `round_01__a.request.json` and collide again on `_write_once`'s "x" mode -
# the very failure this identity exists to remove. A component is an
# identifier out of `TaskSpec.components`, so a dotted one is a wiring mistake,
# and it is refused here rather than silently renamed.
COMPONENT_NAME_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
)


def _validate_component(context):
    """The context's optional `component`, or None when it is absent.

    Absent is today's behaviour byte for byte. Present must be a non-empty
    string that is a bare name: it is used VERBATIM in a file name (it is an
    identifier out of `TaskSpec.components`), so a separator, a dot or any
    other path character is a wiring mistake and raises here rather than
    writing - or overwriting - a record the caller did not mean.
    """
    if "component" not in context:
        return None
    component = context["component"]
    if not isinstance(component, str) or not component:
        raise ValueError("component identity must be a non-empty string")
    if set(component) - COMPONENT_NAME_CHARACTERS:
        raise ValueError("component identity must be a bare name usable in a record file name")
    return component


#: Burst calls run in parallel threads; the attempt counter is shared.
_ATTEMPT_LOCK = threading.Lock()
BURST_CONTEXT_KEYS = {"burst_call_index", "burst_wave", "burst_retry"}
#: A burst call is `True`, or "refill" for a round-0 refill wave (1 or 2), whose
#: request carries BURST_REFILL_PARAGRAPH in place of BURST_PARAGRAPH.
BURST_MODES = (True, "refill")


def _validate_burst(context, burst):
    """A parallel single-candidate call carries `burst_call_index` (0-based),
    `burst_wave` (0 = the first wave, 1 = the refill) and optionally `burst_retry`
    (1-based) in the context, at round 0 (the initial burst) or any proposal round
    (`--parallel-proposals`); an ordinary call carries none of them.

    A round-0 refill call (`burst="refill"`) is wave 1 or 2 at round 0; wave 2
    exists only there. Returns the burst mode (False for an ordinary call)."""
    present = BURST_CONTEXT_KEYS & set(context)
    if not burst:
        if present:
            raise ValueError("burst context keys need a burst call")
        return False
    if burst not in BURST_MODES:
        raise ValueError("a burst call is True or 'refill'")
    refill = burst == "refill"
    index, wave, retry = context.get("burst_call_index"), context.get("burst_wave"), context.get("burst_retry", 0)
    def whole(value, low):
        """True when `value` is an integer (not a bool) of at least `low`."""
        return not isinstance(value, bool) and isinstance(value, int) and value >= low
    if (not (whole(index, 0) and wave in ((1, 2) if refill else (0, 1)) and whole(retry, 0)
             and whole(context.get("round"), 0)) or (refill and context.get("round") != 0)):
        raise ValueError("a parallel call carries a call index, a wave of 0 or 1 (a round-0 "
                         "refill: 1 or 2) and an optional retry")
    return burst


def _round_record_prefix(records, context, component):
    """`round_NN` for a single-outcome task, `round_NN__<component>` otherwise; a parallel
    call is `round_NN__burst_w<wave>_c<index>[_r<retry>]`, then the component."""
    stem = f"round_{context['round']:02d}"
    if "burst_call_index" in context:
        stem = f"{stem}__burst_w{context['burst_wave']}_c{context['burst_call_index']:02d}"
        if context.get("burst_retry"):
            stem = f"{stem}_r{context['burst_retry']}"
    return records / (stem if component is None else f"{stem}__{component}")


RESERVED_VARIANCE_PARAMETER_NAMES = ("rho", "pi")
REQUIRED_CANDIDATE_KEYS = {"expression", "family", "parameters"}

# Policy P1 response contract: the object may also carry a short free-text
# `note`. It is ALLOWED and RECORDED (transport record, archive entry) and the
# engine never reads it - no gate, no ranking, no seed depends on it. A `note`
# that is not a string of at most NOTE_MAX_CHARS characters is not an error: it
# falls through to `ignored_keys` like any other extra key (the fail-soft rule above).
OPTIONAL_CANDIDATE_KEYS = {"note"}
NOTE_MAX_CHARS = 200
# LLM-proposed priors: the one further key a reply may carry
# WHILE `--priors llm` is on. Off - the default and every production arm - it is
# not in the accepted set, so it falls through to `ignored_keys` like any other
# extra key and no proposed prior is ever applied.
PRIORS_CANDIDATE_KEYS = {"priors"}

# Policy P1. One system text, byte-identical across every
# task and suite: no persona, no problem narrative, no domain words, and no
# task-interface prose. All task-specific content travels in the packet's
# `interface` block instead (belief_packet.core_interface). The file is the
# scientific record of the prompt, so its version and sha256 go into config.json
# and into every transport record.
PROMPT_VERSION = "v2.0"
#: The one directory every numbered prompt file lives in, named once so a tag
#: resolves the same way here and in `prompt_version_file` below.
PROMPT_DIRECTORY = Path(__file__).with_name("prompts")
PROMPT_FILE = PROMPT_DIRECTORY / "proposer_system_v2_0.txt"
# LLM-proposed priors: the PRE-DECLARED alternative system text.
# `proposer_system_v2_1.txt` was v2.0's full text plus ONE added section - the
# optional `priors` block, its menu, its hyperparameter windows, the role
# mapping, and that it may be omitted - and nothing else.
# Its role-mapping sentence baked in the WRONG rule
# (support EQUALITY, not containment - the same bug `count_models.
# PRIOR_MENU_BY_ROLE` had); v2.1 was never used in
# a paid call (fake-client tests only), so v2.2 supersedes it
# before any use rather than editing it in place: v2.1's bytes stay frozen on
# disk (sha 2edb4e0c...) and `proposer_system_v2_2.txt` = v2.1 with ONLY that
# one sentence replaced by the corrected support-containment rule.
# It is loaded ONLY while
# a run declares `--priors llm`; the default, and every production arm, loads
# the v2.0 bytes. Neither file is ever rendered or assembled: the run records
# which one it read.
PROMPT_VERSION_LLM_PRIORS = "v2.2"
PROMPT_FILE_LLM_PRIORS = PROMPT_DIRECTORY / "proposer_system_v2_2.txt"
REPAIR_INSTRUCTION = (
    "Your previous reply failed validation with this error:\n{error}\n"
    "Reply with exactly one JSON object "
    '{{"expression": ..., "family": ..., "parameters": {{...}}}} and nothing else.'
)
# A reply with no text at all cannot be echoed back. An assistant turn
# whose content is empty is refused by the compatible wire shape itself (HTTP
# 400 "Invalid assistant message: content or tool_calls must be set"), and the
# Anthropic shape refuses an empty content block the same way, so the repair
# call that was meant to rescue the round used to fail transport-side instead.
# This turn therefore states what happened once, as the user, and quotes
# nothing. Same JSON-shape sentence as REPAIR_INSTRUCTION above.
REPAIR_INSTRUCTION_EMPTY = (
    "Your previous reply was empty (finish_reason={finish_reason}).\n"
    "Reply with exactly one JSON object "
    '{{"expression": ..., "family": ..., "parameters": {{...}}}} and nothing else.'
)


class EmptyResponseBody(ValueError):
    """A persisted reply that carries no text at all.

    A ValueError, so the repair-eligibility rule is exactly the one that was
    already there (a reply-contract failure after the response was persisted).
    The dedicated type is what tells the repair turn to send no echo, and what
    the record reports as `repair_kind: "empty_body"`. No provider or model
    name is involved anywhere in the rule.
    """

    def __init__(self, finish_reason):
        super().__init__(f"empty response body (finish_reason={finish_reason})")
        self.finish_reason = finish_reason


def blank_reply(text):
    """True when a persisted reply carries no text: absent, or whitespace only.

    Any other non-string is left to the checks that were already there - this
    is about a provider that spent the whole budget on reasoning tokens and
    returned an empty body, not about an SDK contract change.
    """
    return text is None or (isinstance(text, str) and not text.strip())


def repair_messages(messages, exc, raw_text, masked_reason, *, instruction=REPAIR_INSTRUCTION,
                    empty_instruction=REPAIR_INSTRUCTION_EMPTY):
    """The one repair turn appended to the conversation.

    An empty reply gets a single user turn (see REPAIR_INSTRUCTION_EMPTY); a
    reply that exists but failed validation keeps the echo + validation
    message, both already masked by their callers.
    """
    if isinstance(exc, EmptyResponseBody):
        return [*messages, {"role": "user", "content": empty_instruction.format(
            finish_reason=mask_secrets(str(exc.finish_reason)))}]
    return [
        *messages,
        {"role": "assistant", "content": raw_text},
        {"role": "user", "content": instruction.format(error=masked_reason)},
    ]


def repair_kind(exc):
    """Which repair turn a repair-eligible failure earns."""
    if isinstance(exc, EmptyResponseBody):
        return "empty_body"
    return "prior_probe" if isinstance(exc, PriorProbeRejection) else "malformed"


# Protocol change of 2026-10-06 for k > 1 requests: the repair turn of a k > 1 request asks again for the shape that
# request asked for. REPAIR_INSTRUCTION and REPAIR_INSTRUCTION_EMPTY above ask for one object -
# right for k = 1 and the round-0 burst, which keep them byte for byte - but a k > 1 round parses
# an array (an archive review round: the {"candidates", "revive"} object), so a reply that obeyed
# them failed again and the round was lost. `{limit}` is written in by `k_repair_texts` (a plain
# replace); `{error}` / `{finish_reason}` are left to `repair_messages`, as for the k = 1 texts.
REPAIR_INSTRUCTION_ARRAY = (
    "Your previous reply failed validation with this error:\n{error}\n"
    "Reply with a JSON array of exactly {limit} candidate objects, each with the keys specified "
    "above, and nothing else."
)
REPAIR_INSTRUCTION_ARRAY_EMPTY = (
    "Your previous reply was empty (finish_reason={finish_reason}).\n"
    "Reply with a JSON array of exactly {limit} candidate objects, each with the keys specified "
    "above, and nothing else."
)
REPAIR_INSTRUCTION_REVIEW = (
    "Your previous reply failed validation with this error:\n{error}\n"
    'Reply with one raw JSON object {{"candidates": [...], "revive": [...]}} and nothing else: '
    "candidates is a JSON array of exactly {limit} candidate objects, each with the keys specified "
    "above, and revive lists up to two archived model_id strings exactly as given, or is empty."
)
REPAIR_INSTRUCTION_REVIEW_EMPTY = (
    "Your previous reply was empty (finish_reason={finish_reason}).\n"
    'Reply with one raw JSON object {{"candidates": [...], "revive": [...]}} and nothing else: '
    "candidates is a JSON array of exactly {limit} candidate objects, each with the keys specified "
    "above, and revive lists up to two archived model_id strings exactly as given, or is empty."
)
REPAIR_INSTRUCTION_ARRAY_VERSION = "repair_array_v1"
REPAIR_INSTRUCTION_REVIEW_VERSION = "repair_review_v1"


def k_repair_texts(proposal_limit, *, review=False):
    """The repair texts of a k > 1 request, `{limit}` written in: an array round's, or an archive
    review round's ({"candidates", "revive"} object). `repair_messages` keyword arguments."""
    if type(proposal_limit) is not int or proposal_limit < 2:
        raise ValueError("k = 1 keeps REPAIR_INSTRUCTION; the k-aware texts are for k > 1")
    pair = ((REPAIR_INSTRUCTION_REVIEW, REPAIR_INSTRUCTION_REVIEW_EMPTY) if review
            else (REPAIR_INSTRUCTION_ARRAY, REPAIR_INSTRUCTION_ARRAY_EMPTY))
    return {"instruction": pair[0].replace("{limit}", str(proposal_limit)),
            "empty_instruction": pair[1].replace("{limit}", str(proposal_limit))}


def repair_texts_for(exc, proposal_limit=1, *, burst=False, review=False, slots=None):
    """The one repair turn's texts for this request, and the record fields that name them.

    k = 1 and the round-0 burst: exactly what was sent before this change - the caller's slot texts
    (`StageSlots.repair_texts()`), or none, so `repair_messages` uses REPAIR_INSTRUCTION /
    REPAIR_INSTRUCTION_EMPTY - and no record field: not one byte moves.
    k > 1: `k_repair_texts` (through `slots.repair_texts(k, review=...)` when the caller has slots),
    and the fields `repair_instruction_version` and `repair_instruction_sha256` - the sha256 of the
    template this failure gets (the empty-body one for an empty reply), with `{limit}` written in
    and `{error}` / `{finish_reason}` still open, so one wording has one digest in every round.
    Returns (repair_messages keyword arguments, record fields).
    """
    if burst or proposal_limit == 1:
        return ({} if slots is None else slots.repair_texts()), {}
    texts = (k_repair_texts(proposal_limit, review=review) if slots is None
             else slots.repair_texts(proposal_limit, review=review))
    empty = isinstance(exc, EmptyResponseBody)
    version = REPAIR_INSTRUCTION_REVIEW_VERSION if review else REPAIR_INSTRUCTION_ARRAY_VERSION
    used = texts["empty_instruction"] if empty else texts["instruction"]
    return texts, {"repair_instruction_version": version + ("_empty" if empty else ""),
                   "repair_instruction_sha256": hashlib.sha256(used.encode("utf-8")).hexdigest()}


def admit_candidates(proposed, admit, *, proposal_limit=1, burst=False, rejected=()):
    """The runner's admission hook (the prior probe) over one parsed reply.

    k = 1 and the round-0 burst: unchanged - every candidate goes through `admit` in turn and a
    PriorProbeRejection propagates, so the one repair turn carries its sentence.
    k > 1 (protocol change of 2026-10-06; an array round or an archive review round): a candidate the
    hook refuses with PriorProbeRejection is refused ALONE - the hook has written its own archive
    row - and is returned as a per-candidate rejection under its payload index, with error_type
    "PriorProbeRejection", failure_class "prior_probe" and repair_eligible False; the others are
    admitted. Only when no candidate is left does the reply fail: PriorProbeRejection with
    ARRAY_REFUSAL_PREFIX and every refusal's reason, the parser's included, which earns the one
    repair turn. `rejected` is the parser's per-candidate rejections of the same reply (payload
    indices); each accepted candidate's payload index follows from them. Any other exception the
    hook raises propagates as before.

    Returns (the admitted candidates in payload order, the probe refusals).
    """
    if proposal_limit == 1 or burst:
        for candidate in proposed:
            admit(candidate)
        return list(proposed), []
    skipped = {entry["index"] for entry in rejected}
    payload_indices = [index for index in range(len(proposed) + len(skipped))
                       if index not in skipped]
    admitted, refused = [], []
    for index, candidate in zip(payload_indices, proposed, strict=True):
        try:
            admit(candidate)
        except PriorProbeRejection as exc:
            reason = mask_secrets(str(exc))
            refused.append({
                "index": index, "expression": candidate.expression, "family": candidate.family,
                "reason": reason, "parameters": dict(candidate.parameters),
                "failure_class": failure_class(error_type="PriorProbeRejection", reason=reason),
                "error_type": "PriorProbeRejection", "repair_eligible": False,
            })
            continue
        admitted.append(candidate)
    if not admitted:
        reasons = [entry["reason"]
                   for entry in sorted([*rejected, *refused], key=lambda entry: entry["index"])]
        raise PriorProbeRejection(ARRAY_REFUSAL_PREFIX + "; ".join(reasons))
    return admitted, refused

SYSTEM = """Propose at most three distinct probabilistic count models as a JSON array.
Each candidate object has exactly these three keys: expression, family, parameters.
Do not add any other key (for example, do not add a noise_parameters key).
expression is a logit; p=sigmoid(expression), u is in [0,1], z=2*u-1, and count
support is 0..50. parameters maps each mean-function coefficient name that
appears in expression to a fixed prior role, not a numeric prior. The only
allowed roles are exactly these four: intercept, signed, positive, shape.
Family variance/dispersion parameters (rho for beta_binomial, pi for
zi_binomial) are owned by family and assigned automatically from a fixed
prior outside this message; never add rho, pi, or any other name to parameters.
Use only declared coefficients, u/z, real literals, arithmetic, and
exp/log/sqrt/tanh/square/power. At most 6 coefficients and 15 mean operators.
Do not return Python code, a truth parameter, prior modifications, or prose.
Respect the cell restrictions supplied in the request as structure_expansion
and family_expansion. When structure_expansion is false, expression must be
exactly one of the strings in initial_structures (verbatim); only family may
change. When family_expansion is false, family must be exactly the initial
family supplied in the request; only expression may change.
These are proposals based only on the public training observations, not test results.
"""


# A DEVELOPMENT-ONLY override of the system
# prompt, for trying a candidate prompt revision before it earns its own
# PROMPT_FILE/PROMPT_VERSION constant. None - the default, and the only value a
# run whose label would be "production" may hold (boxinggym_core.run refuses
# any other value in that case) - reproduces the v2.0/v2.2 selection below
# byte-for-byte. Set ONCE per run by the runner, exactly like
# `count_models.set_prior_template`/`set_llm_priors`; never written to by this
# module itself.
_PROMPT_VARIANT_FILE = None
_PROMPT_VARIANT_VERSION = None
PROMPT_VARIANT_TAG_PREFIX = "# prompt_version:"


def set_prompt_variant(path):
    """Select (or clear) a development-only SYSTEM prompt file; returns the previous (path, version).

    `path` None clears the override - `system_prompt_bytes`/`prompt_identity`
    then read the byte-frozen v2.0/v2.2 files exactly as they did before this
    change. A non-None path must be a text file whose FIRST LINE is exactly
    `{PROMPT_VARIANT_TAG_PREFIX} <tag>`; `<tag>` becomes this run's
    `prompt_version`, so a dev experiment's config.json/transport records
    still say WHICH text was actually sent instead of a bare "v2.0" it never
    used. Raises rather than silently falling back on a missing/empty tag,
    exactly as `count_models.set_prior_template` raises on an unregistered
    template name.
    """
    global _PROMPT_VARIANT_FILE, _PROMPT_VARIANT_VERSION
    previous = (_PROMPT_VARIANT_FILE, _PROMPT_VARIANT_VERSION)
    if path is None:
        _PROMPT_VARIANT_FILE, _PROMPT_VARIANT_VERSION = None, None
        return previous
    text = Path(path).read_text()
    first_line = text.splitlines()[0] if text else ""
    if not first_line.startswith(PROMPT_VARIANT_TAG_PREFIX):
        raise ValueError(
            "prompt variant file must declare its tag on line 1 as "
            f"{PROMPT_VARIANT_TAG_PREFIX!r} <tag>; {path!r} does not"
        )
    tag = first_line[len(PROMPT_VARIANT_TAG_PREFIX) :].strip()
    if not tag:
        raise ValueError(f"prompt variant file {path!r} declares an empty tag")
    _PROMPT_VARIANT_FILE, _PROMPT_VARIANT_VERSION = str(path), tag
    return previous


def prompt_variant_active():
    """True while a development prompt-variant override is selected."""
    return _PROMPT_VARIANT_FILE is not None


#: The packaged init-slot generator texts. A generator is
#: not a prompt arm and not a development variant: only a runner's dedicated generation
#: option selects one (ChemBench `--pool-generator`), never `--prompt-version` or
#: `--prompt-variant`. Its file is sent exactly as stored; the name is identity metadata
#: (`prompt_identity`, transport records) and never request text.
POOL_GENERATOR_FILES = {"pool_generator_v1": PROMPT_DIRECTORY / "pool_generator_v1.txt"}
_POOL_GENERATOR = None


def select_pool_generator(name):
    """Select a packaged pool generator (None clears it); returns the previous name."""
    global _POOL_GENERATOR
    if name is not None and name not in POOL_GENERATOR_FILES:
        raise ValueError(f"unknown pool generator {name!r}; packaged: {sorted(POOL_GENERATOR_FILES)}")
    if name is not None and _PROMPT_VARIANT_FILE is not None:
        raise ValueError("a pool generator and a prompt variant are alternative system texts")
    previous, _POOL_GENERATOR = _POOL_GENERATOR, name
    return previous


def system_prompt_file(llm_priors=None):
    """The (path, version) pair this run reads - the single source both records below use.

    The v2.0 file unless this run declared `--priors llm` (LLM-proposed priors),
    in which case the pre-declared v2.2 file is the one read (v2.2 supersedes
    v2.1 before any use). All three are byte-frozen files on
    disk. A selected prompt-variant file (development-only,
    see `set_prompt_variant`) overrides BOTH of those, exactly as the v2.1/v2.2
    switch itself is an override - never combined with it.

    `llm_priors` states the `--priors` decision for a caller that has
    to know the identity BEFORE `count_models.set_llm_priors` applies it - the
    label receipt, which a runner resolves before any output exists. None (every
    other caller) reads the live switch, so nothing else changes.
    """
    if _PROMPT_VARIANT_FILE is not None:
        return Path(_PROMPT_VARIANT_FILE), _PROMPT_VARIANT_VERSION
    if _POOL_GENERATOR is not None:
        # A packaged generator, read as stored (see `select_pool_generator`).
        return POOL_GENERATOR_FILES[_POOL_GENERATOR], _POOL_GENERATOR
    if llm_priors is None:
        llm_priors = llm_priors_enabled()
    if llm_priors:
        return PROMPT_FILE_LLM_PRIORS, PROMPT_VERSION_LLM_PRIORS
    return PROMPT_FILE, PROMPT_VERSION


def system_prompt_bytes():
    """The stored system text exactly as stored; the run records its hash, not a rendering."""
    return system_prompt_file()[0].read_bytes()


def prompt_identity(llm_priors=None):
    """Version and sha256 of the system prompt file, for config.json and every record.

    This pair is what PROVES which prompt a run used: `--priors llm` records
    v2.2 and its sha, a selected prompt variant records the
    tag its own first line declares, and everything else records v2.0 and
    f095edf1... . The sha is always of the bytes actually read.
    """
    path, version = system_prompt_file(llm_priors)
    return {"prompt_version": version, "prompt_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


# Proposals per round. The packaged prompt files ask for ONE
# candidate object; a run that asks for k > 1 needs the reply contract and the
# composition rule stated, and the packaged bytes are frozen, so the runner
# COMPOSES the request instead of editing the file: the file's bytes, then this
# paragraph. The sentence is general - it names no benchmark, no law and no
# example expression - and the sha256 of the COMPOSED text is what a run records
# and a registration binds, beside (never instead of) the file's own digest.
PROPOSAL_LIMIT_RANGE = (1, 3)
COMPOSITION_PARAGRAPH = (
    "Return a JSON array of exactly {limit} candidate objects, each with the "
    "three keys above and nothing else. Propose {limit} candidates whose "
    "operator structures differ from one another and none of which contains "
    "another as a sub-expression.\n"
)


#: The initial burst. At round 0 the runner may ask for K candidates in K
#: INDEPENDENT calls of one candidate each (the per-round contract, judge and repair
#: turn apply unchanged); every call sends the selected prompt file plus this one
#: general paragraph. It names no benchmark, no example and no other candidate.
BURST_PARAGRAPH = (
    "This request is one of several independent requests made in parallel for the same "
    "round; each answers with one candidate and none sees the others. When the pool is "
    "empty (round 0) there is no residual pattern yet: propose one candidate model for "
    "the process the interface and observations describe. Keep the family "
    "constraints.initial_family; every other rule applies unchanged.\n"
)

#: A round-0 refill wave (waves 1 and 2, each asking for exactly the shortfall)
#: sends the selected prompt file plus this paragraph in place of BURST_PARAGRAPH; its
#: packet's pool holds the candidates the earlier waves of the round admitted. Like the
#: burst paragraph it names no benchmark, no example and no candidate.
BURST_REFILL_PARAGRAPH = (
    "This request is one of several independent requests made in parallel for the same "
    "round; each answers with one candidate and none sees the others. The pool in this "
    "packet already holds the candidates admitted by earlier requests of this round. "
    "Propose one candidate whose functional form is not already in the pool; a candidate "
    "with the same canonical identity as a pool member is discarded as a duplicate. Keep "
    "the family constraints.initial_family; every other rule applies unchanged.\n"
)


#: COMPOSITION_PARAGRAPH v2, selected by a runner through
#: `set_composition_paragraph("v2")` (v1 above stays the default). Canonical bytes: the registered file
#: composition_paragraph_v2.txt of the unified protocol
#: (525 bytes with its newline); `{limit}` is filled by the composer and v2 is written for 3.
COMPOSITION_PARAGRAPH_V2 = (
    "Return a JSON array of exactly {limit} candidate objects, each with exactly the keys "
    "specified above and nothing else. Propose three distinct, non-equivalent hypotheses. At "
    "least two of the three candidates should differ meaningfully in their primary functional "
    "composition, rather than merely in parameterization or by adding or removing a correction "
    "term. At most one candidate may be a nested or incremental refinement of a high-posterior "
    "model, and only if such a refinement is specifically supported by the diagnostics.\n"
)
COMPOSITION_PARAGRAPH_V2_SHA256 = "782a30b74cd25adb6b51351096cf97e31a7375d8630d726ee8eb7d7949c35be6"
COMPOSITION_PARAGRAPHS = {"v1": COMPOSITION_PARAGRAPH, "v2": COMPOSITION_PARAGRAPH_V2}
#: An archive review round appends this after v2. Canonical bytes: the registered
#: file archive_review_paragraph_v1.txt of the unified protocol
#: (675 bytes with its newline). It carries literal braces, so `{limit}` is filled by replace.
ARCHIVE_REVIEW_PARAGRAPH = (
    "This is an archive review round. Besides your {limit} new candidates, you may ask for up to "
    "two archived models to be re-fit: choose only among archive entries whose outcome is "
    "retired_by_evidence and whose form you now judge likely given all current observations "
    "(they were retired on less data). Each model you name is re-fit from its prior on all "
    "current public data and competes for the pool on evidence like any other model. Reply with "
    'one raw JSON object {"candidates": [...], "revive": [...]}: candidates is the array '
    "described above, and revive lists the chosen archive model_id strings exactly as given, or "
    "is empty when nothing in the archive deserves another fit.\n"
)
ARCHIVE_REVIEW_PARAGRAPH_VERSION = "v1"
ARCHIVE_REVIEW_PARAGRAPH_SHA256 = "8b4c5359d81b6b3e5afed570c1e3c3f1bb217a57e70977bcbdb42d1238cd6cde"
_COMPOSITION_VERSION = "v1"
#: The names the BoxingGym runner and its tests use: v1 under its own
#: name, the one limit v2 is written for, and how many archive models a review may name.
COMPOSITION_PARAGRAPH_V1 = COMPOSITION_PARAGRAPH
COMPOSITION_PARAGRAPH_V1_SHA256 = hashlib.sha256(COMPOSITION_PARAGRAPH_V1.encode("utf-8")).hexdigest()
COMPOSITION_LIMIT = 3
ARCHIVE_REVIVE_LIMIT = 2


def fill_limit(paragraph, limit):
    """`{limit}` written in; a plain replace, because the review paragraph quotes JSON braces."""
    return paragraph.replace("{limit}", str(int(limit)))


def set_composition_paragraph(version):
    """Select the k > 1 composition paragraph ("v1", the default, or "v2"); returns the previous.

    A runner selects once per run, before its receipt; v1 composes every request exactly as
    before this option existed."""
    global _COMPOSITION_VERSION
    if version not in COMPOSITION_PARAGRAPHS:
        raise ValueError(f"composition paragraph must be one of {sorted(COMPOSITION_PARAGRAPHS)}")
    previous, _COMPOSITION_VERSION = _COMPOSITION_VERSION, version
    return previous


def validate_proposal_limit(limit):
    """The candidates-per-round bound, 1..3 (core_count.py's own CLI rule)."""
    low, high = PROPOSAL_LIMIT_RANGE
    if type(limit) is not int or not low <= limit <= high:
        raise ValueError(f"proposal limit must be {low}..{high}")
    return limit



class StageSlots:
    """Prompt texts a caller binds to packaged slot files.

    The four texts replace the module constants BURST_PARAGRAPH, BURST_REFILL_PARAGRAPH,
    REPAIR_INSTRUCTION and REPAIR_INSTRUCTION_EMPTY for that caller's requests; in this
    edition they are byte-identical copies of those constants (a test asserts it), so no
    request byte moves. `metadata` names the slots and their digests and is written to the
    transport record only - it never enters a system text, a packet or a message. A caller
    that passes no StageSlots (CORE, BoxingGym, every mock path) is unchanged.
    """

    __slots__ = (
        "burst_paragraph",
        "burst_refill_paragraph",
        "metadata",
        "repair_instruction",
        "repair_instruction_empty",
    )

    def __init__(self, *, burst_paragraph, burst_refill_paragraph, repair_instruction,
                 repair_instruction_empty, metadata):
        for name, text in (("burst_paragraph", burst_paragraph),
                           ("burst_refill_paragraph", burst_refill_paragraph),
                           ("repair_instruction", repair_instruction),
                           ("repair_instruction_empty", repair_instruction_empty)):
            if not isinstance(text, str) or not text:
                raise ValueError(f"stage slot {name} must be a nonempty text")
        if not isinstance(metadata, dict):
            raise ValueError("stage slot metadata must be a dict")  # noqa: TRY004 - callers catch ValueError
        self.burst_paragraph = burst_paragraph
        self.burst_refill_paragraph = burst_refill_paragraph
        self.repair_instruction = repair_instruction
        self.repair_instruction_empty = repair_instruction_empty
        self.metadata = dict(metadata)

    def burst_paragraphs(self):
        """The (burst, round-0 refill) paragraph pair `composed_system_prompt` takes."""
        return self.burst_paragraph, self.burst_refill_paragraph

    def repair_texts(self, proposal_limit=1, *, review=False):
        """k = 1 (and the burst): the two slot texts. k > 1 (protocol change of 2026-10-06): `k_repair_texts`,
        which no slot file holds in this edition."""
        if proposal_limit == 1:
            return {"instruction": self.repair_instruction,
                    "empty_instruction": self.repair_instruction_empty}
        return k_repair_texts(proposal_limit, review=review)

    def record_fields(self, burst):
        """The record-only slot block of one request: which stage it came from."""
        return {**self.metadata, "origin_stage": "init" if burst else "expand"}


def repair_text_kind(kind):
    """The slot-level `repair_kind` (validation|empty_body) a project decision fixed, for an engine repair_kind."""
    return "empty_body" if kind == "empty_body" else "validation"


def composed_system_prompt(proposal_limit=1, *, burst=False, burst_paragraphs=None, archive_review=False):
    """The system text this run sends, and the identity block that names it.

    `proposal_limit` 1 - the default and what every finished wave ran - returns
    the stored file's text unchanged and the identity block `prompt_identity`
    has always produced, so not one byte of the k = 1 request moves. Above 1 the
    composition paragraph is appended and the block gains
    `composed_prompt_sha256` and `proposal_limit`; `prompt_version` and
    `prompt_sha256` keep naming the FILE, because that is what they mean.

    `burst="refill"` composes the file plus BURST_REFILL_PARAGRAPH the same way
    and names that text's digest under `burst_refill_prompt_sha256`.

    With paragraph v2 selected (`set_composition_paragraph`) the k = 3 request is
    the file plus v2, and the block also names v2 (`composition_paragraph_version`,
    `composition_paragraph_sha256`, the digest of the canonical bytes); `archive_review=True` (an
    archive review round, v2 only) appends ARCHIVE_REVIEW_PARAGRAPH and names it the same way.
    `composed_prompt_sha256` is always the digest of the whole text sent; a review round also
    names that digest `archive_review_prompt_sha256`, the key the BoxingGym records use.
    """
    if archive_review and (burst or proposal_limit == 1 or _COMPOSITION_VERSION != "v2"):
        raise ValueError("an archive review round is a k > 1 request with composition paragraph v2")
    if burst:
        if burst not in BURST_MODES:
            raise ValueError("a burst request is True or 'refill'")
        # A burst call sends the file plus the burst paragraph and names
        # that text's digest under its own key, so a run with a burst and k = 1
        # rounds still records the per-round request exactly as a run without one.
        paragraph, key = ((BURST_REFILL_PARAGRAPH, "burst_refill_prompt_sha256") if burst == "refill"
                          else (BURST_PARAGRAPH, "burst_prompt_sha256"))
        if burst_paragraphs is not None:
            # A caller's slot texts (byte-identical copies in this edition).
            paragraph = burst_paragraphs[1] if burst == "refill" else burst_paragraphs[0]
        composed = system_prompt_bytes().decode() + paragraph
        return composed, {
            **prompt_identity(),
            key: hashlib.sha256(composed.encode()).hexdigest(),
        }
    validate_proposal_limit(proposal_limit)
    text = system_prompt_bytes().decode()
    identity = prompt_identity()
    if proposal_limit == 1:
        return text, identity
    if _COMPOSITION_VERSION == "v2":
        if proposal_limit != COMPOSITION_LIMIT:
            raise ValueError("composition paragraph v2 is written for three candidates")
        composed = text + COMPOSITION_PARAGRAPH_V2.replace("{limit}", str(proposal_limit))
        extra = {"composition_paragraph_version": "v2",
                 "composition_paragraph_sha256": COMPOSITION_PARAGRAPH_V2_SHA256}
        if archive_review:
            composed += ARCHIVE_REVIEW_PARAGRAPH.replace("{limit}", str(proposal_limit))
            extra.update(archive_review_paragraph_version=ARCHIVE_REVIEW_PARAGRAPH_VERSION,
                         archive_review_paragraph_sha256=ARCHIVE_REVIEW_PARAGRAPH_SHA256,
                         archive_review_prompt_sha256=hashlib.sha256(composed.encode()).hexdigest())
        return composed, {
            **identity,
            "proposal_limit": proposal_limit,
            "composed_prompt_sha256": hashlib.sha256(composed.encode()).hexdigest(),
            **extra,
        }
    composed = text + COMPOSITION_PARAGRAPH.format(limit=proposal_limit)
    return composed, {
        **identity,
        "proposal_limit": proposal_limit,
        "composed_prompt_sha256": hashlib.sha256(composed.encode()).hexdigest(),
    }


def burst_refill_identity():
    """The round-0 refill request's digests - the
    selected prompt file's bytes alone, BURST_REFILL_PARAGRAPH alone (UTF-8), and the
    composed text a refill call sends (`composed_system_prompt(burst="refill")`)."""
    return {
        "base_prompt_sha256": hashlib.sha256(system_prompt_bytes()).hexdigest(),
        "burst_refill_paragraph_sha256": hashlib.sha256(BURST_REFILL_PARAGRAPH.encode("utf-8")).hexdigest(),
        "burst_refill_prompt_sha256": composed_system_prompt(burst="refill")[1]["burst_refill_prompt_sha256"],
    }


def composed_prompt_identity(proposal_limit=1, *, archive_review=False):
    """`composed_system_prompt`'s identity block alone, for a caller that sends no text.

    The mock path and the catalogue-exhausted path write a transport record but
    make no request; they still have to name the prompt the run is bound to, and
    at k = 1 this returns exactly what `prompt_identity()` returns.
    """
    return composed_system_prompt(proposal_limit, archive_review=archive_review)[1]


#: A numbered prompt file is named by the version it declares, so a tag can only
#: ever mean one packaged file beside this module. The shape is pinned here rather than
#: spelled out at each call site, and it forbids a path of any kind.
PROMPT_VERSION_TAG = re.compile(r"v[0-9]+(?:\.[0-9]+)+(?:\.[A-Za-z][A-Za-z0-9-]*)?")
#: Public-release overhaul (2026-10-07): numbered prompt files that live outside `prompts/` (whose file set the tests pin), by tag:
#: the source checkout's single copy, relative to the repository root, and its sha256. The tag is refused when the
#: file is absent (an installed package without the checkout) or its bytes differ.
REPOSITORY_PROMPT_FILES = {
    "v2.9.FA-C": ("tools/fb1_launch_c/proposer_system_v2_9_FA-C.txt",
                  "196f82b7a0f94e12a7cb8be2d07e776b1536aed11bf8ebd4bea1872c7002953c"),
    # FA-C with v2.2's numeric priors block, for `--priors llm` (FA-C itself tells the proposer never to write priors)
    "v2.9.FA-C-priors": ("tools/fb1_launch_c/proposer_system_v2_9_FA-C_priors.txt",
                         "dc03d1a61dc484c6bc28aa216887e82469f00ec791c568b01f96fb65b9779324"),
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def prompt_version_file(tag):
    """The packaged prompt file a version tag names; refused unless it exists."""
    if not isinstance(tag, str) or PROMPT_VERSION_TAG.fullmatch(tag) is None:
        raise ValueError(f"a prompt version tag looks like {PROMPT_VERSION!r}; {tag!r} does not")
    if tag in REPOSITORY_PROMPT_FILES:
        relative, digest = REPOSITORY_PROMPT_FILES[tag]
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise ValueError(f"prompt version {tag!r} is the repository file {relative}, absent here")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"prompt version {tag!r}: {relative} does not have the registered sha256 {digest}")
        return path
    path = PROMPT_DIRECTORY / f"proposer_system_{tag.replace('.', '_')}.txt"
    if not path.is_file():
        raise ValueError(f"no packaged prompt file for version {tag!r} ({path.name})")
    return path


def select_prompt_version(tag):
    """Select the packaged prompt file a run declares; returns its path, or None for the default.

    `PROMPT_VERSION` is what `system_prompt_file` already reads (and its v2.2
    companion under `--priors llm`), so the default tag is exactly the absent option and
    selects nothing. Any other tag resolves to its numbered file and goes through
    `set_prompt_variant`, and the file's own first line must declare the tag the caller
    asked for: a run can then never record a version whose bytes it did not send. Checked
    before anything is applied, so a refused selection leaves no override behind.
    """
    if tag is None or tag == PROMPT_VERSION:
        return None
    path = prompt_version_file(tag)
    lines = path.read_text().splitlines()
    declared = lines[0][len(PROMPT_VARIANT_TAG_PREFIX):].strip() if (
        lines and lines[0].startswith(PROMPT_VARIANT_TAG_PREFIX)) else None
    if declared != tag:
        raise ValueError(
            f"prompt file {path.name} must declare {PROMPT_VARIANT_TAG_PREFIX} {tag} on line 1; "
            f"it declares {declared!r}"
        )
    set_prompt_variant(path)
    return path


def select_run_prompt(args, *, llm_priors=False):
    """Apply this run's prompt selection and return the identity its label receipt binds.

    A numbered `--prompt-version` is the registerable selection; `--prompt-variant
    PATH` stays the free-form development override with its own production refusal in the
    runner. One run sends one system text, so stating both is refused here rather than
    resolved by precedence. `--priors llm` selects the v2.2 file by itself, so a
    version stated beside it is refused the same way: honouring the tag would send
    v2.2 under another name, and `v2.0` would silently record v2.2 - neither is a record of
    what was sent. When no numbered file is selected the development variant is applied
    here - or, stated as None, a selection an earlier in-process run left behind is cleared
    - BEFORE the identity is read, so the receipt binds the text this run actually sends on
    every path instead of a default that a later application would then replace.
    """
    tag = getattr(args, "prompt_version", None)
    variant = getattr(args, "prompt_variant", None)
    if tag is not None and variant is not None:
        raise ValueError(
            "--prompt-version selects a packaged prompt file and --prompt-variant a "
            "development one; state at most one"
        )
    if tag is not None and llm_priors:
        raise ValueError(
            "--prompt-version selects a packaged prompt file and --priors llm selects the "
            f"{PROMPT_VERSION_LLM_PRIORS} one by itself; state at most one prompt determinant"
        )
    # A generator an earlier in-process run selected never outlives that run.
    select_pool_generator(None)
    selected = select_prompt_version(tag)
    if selected is None:
        set_prompt_variant(variant)
    return selected, prompt_identity(llm_priors)


def _write_once(path, payload):
    """Write-once record. Every string reachable from `payload` goes through
    `mask_secrets_deep` first: the request bodies, the response
    record, the ignored fields and the note are all provider text, so this is
    the single choke point that keeps a credential out of a written file."""
    with path.open("x") as stream:
        json.dump(mask_secrets_deep(payload), stream, indent=2)
        stream.write("\n")


# API keys are read only from these environment variables (never a file - see
# AnthropicCountProposer/OpenAICountProposer.__init__) and never deliberately
# placed in a record. `mask_secrets` is defense in depth for the one place an
# SDK's own exception text is stored verbatim: an auth/network error CAN echo
# request or header detail, and on a real (never-tested-here) live call that
# text might contain the live key. Every `str(exc)` captured into a written
# record passes through this first - see the two providers' propose_one.
SECRET_ENV_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
# Masking is no longer limited to `reason`.
# Two sources are masked - the VALUE of either environment variable above, and
# any `sk-`/`sk-ant-` prefixed token even when it is not the current
# environment's value (a key echoed back from a header or another process is
# masked just the same). The replacement is ALWAYS exactly one of the two
# placeholders below and never reveals the length, the prefix or any substring
# of what it replaced (design addendum 2026-09-08 item 2).
SECRET_PATTERNS = (
    ("ANTHROPIC_API_KEY", re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")),
    ("OPENAI_API_KEY", re.compile(r"sk-(?!ant-)[A-Za-z0-9_\-]{8,}")),
)
MASK_PLACEHOLDERS = tuple(f"<{name}:masked>" for name in SECRET_ENV_VARS)
# Library loggers whose records can carry credential text. `openai` and
# `anthropic` log through CHILD loggers (`openai._base_client`), and a
# `logging.Filter` attached to a logger is NOT applied to records made by its
# children (`logging.Logger.handle` runs only that logger's filters, and
# `callHandlers` walks ancestors for HANDLERS, not filters) - so the filter is
# attached to every logger that already exists under these roots AND a
# LogRecord factory covers the ones the SDK creates later.
MASKED_LOGGER_ROOTS = ("openai", "anthropic", "httpx", "httpcore")


def mask_secrets(text):
    """Replace every credential occurrence in `text` with its `<NAME:masked>`
    placeholder. A secret env var value shorter than 8 characters is never
    treated as a secret on the stock path. Explicit compatibility credential
    names mask every nonempty value, including shorter values. Anything that is
    not a string is returned unchanged."""
    if not isinstance(text, str):
        return text
    for name in (*SECRET_ENV_VARS, *sorted(_COMPAT_SECRET_NAMES)):
        value = os.environ.get(name)
        if value and (len(value) >= 8 or name in _COMPAT_SECRET_NAMES) and value in text:
            text = text.replace(value, f"<{name}:masked>")
    for name, pattern in SECRET_PATTERNS:
        text = pattern.sub(f"<{name}:masked>", text)
    return text


def mask_secrets_deep(value):
    """`mask_secrets` applied to every string reachable from `value`, keeping
    dict/list/tuple/set structure. Dict KEYS are masked too - a provider can
    echo a header the other way round. Anything that is neither a string nor
    one of those containers is returned unchanged (numbers, None, arrays)."""
    if isinstance(value, str):
        return mask_secrets(value)
    if isinstance(value, bytes) and _COMPAT_SECRET_NAMES:
        for name in sorted(_COMPAT_SECRET_NAMES):
            secret = os.environ.get(name)
            if secret:
                value = value.replace(secret.encode(), f"<{name}:masked>".encode())
        return value
    if isinstance(value, dict):
        return {mask_secrets_deep(key): mask_secrets_deep(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_secrets_deep(item) for item in value]
    if isinstance(value, tuple):
        return tuple(mask_secrets_deep(item) for item in value)
    if isinstance(value, set):
        return {mask_secrets_deep(item) for item in value}
    return value


def carries_secret(text):
    """True when `mask_secrets` would change `text`. This is the test that
    keeps the masked stand-in below OFF every path with nothing to hide."""
    return isinstance(text, str) and mask_secrets(text) != text


class MaskedCredentialError(RuntimeError):
    """Stand-in for an exception whose own text carried a credential.

    Raised INSTEAD of the original only at the API transport boundary and only
    when the original really did carry one, so the numerical rule - an expected
    numerical failure propagates verbatim, original type and traceback - is
    untouched.
    """


def masked_exception(exc):
    """`exc` itself unless its str, repr or formatted traceback would leak a
    credential; then a `MaskedCredentialError` carrying the masked text and the
    original type name. Raise the result with `from None` so the original is
    not printed as the exception context."""
    parts = [str(exc), repr(exc)]
    try:
        parts.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:  # noqa: BLE001 - a formatting failure must not lose the mask
        parts.append("")
    if not any(carries_secret(part) for part in parts):
        return exc
    return MaskedCredentialError(f"{type(exc).__name__}: {mask_secrets(str(exc))}")


class SecretMaskingFilter(logging.Filter):
    """Masks credential text in a LogRecord's message, args and traceback.

    `exc_text` is pre-filled with the MASKED traceback because
    `logging.Formatter.format` uses `record.exc_text` verbatim when it is
    already set and only formats `exc_info` when it is not - which is how the
    SDK's own `exc_info=True` DEBUG log (openai/_base_client.py) is stopped
    from printing an exception message that echoes a key.
    """

    def filter(self, record):
        """Mask `record` in place and always keep it (returns True)."""
        record.msg = mask_secrets_deep(record.msg)
        if record.args:
            record.args = mask_secrets_deep(record.args)
        if getattr(record, "exc_text", None):
            record.exc_text = mask_secrets(record.exc_text)
        elif record.exc_info:
            try:
                record.exc_text = mask_secrets(
                    "".join(traceback.format_exception(*record.exc_info))
                )
            except Exception:  # noqa: BLE001 - never break logging over masking
                record.exc_text = "<traceback withheld: masking failed>"
        return True


_SECRET_LOG_FILTER = SecretMaskingFilter()


def install_secret_log_filter():
    """Attach `SecretMaskingFilter` to the SDK/HTTP loggers, and install a
    LogRecord factory so loggers those libraries create LATER are covered too
    (they are not imported yet when this module is). Idempotent, and called
    once at import so no caller has to remember it."""
    for name in list(logging.Logger.manager.loggerDict) + list(MASKED_LOGGER_ROOTS):
        if name.split(".", 1)[0] not in MASKED_LOGGER_ROOTS:
            continue
        logger = logging.getLogger(name)
        if _SECRET_LOG_FILTER not in logger.filters:
            logger.addFilter(_SECRET_LOG_FILTER)
    factory = logging.getLogRecordFactory()
    if getattr(factory, "masks_secrets", False):
        return factory

    def masking_factory(*args, **kwargs):
        """Build a LogRecord with the previous factory and mask it when it comes from an SDK/HTTP logger."""
        record = factory(*args, **kwargs)
        if record.name.split(".", 1)[0] in MASKED_LOGGER_ROOTS:
            _SECRET_LOG_FILTER.filter(record)
        return record

    masking_factory.masks_secrets = True
    logging.setLogRecordFactory(masking_factory)
    return masking_factory


install_secret_log_filter()


# --- Tolerated parameter shapes (mirroring the fb1 rule) ---------------
# The request publishes every role's fixed prior (`fixed_prior_roles`), so a
# model can mirror that nesting back and send `{"b0": {"role": "intercept"}}`
# where the contract asks for `{"b0": "intercept"}`. The ROLE is right and only
# the shape is wrong, so the shape is normalized and recorded instead of being
# reported as a role violation - which is what the earlier judge did
# ("coefficient role outside fixed prior policy"), losing a
# candidate whose role was correct behind a message that reads as a role error.
PARAMETER_SHAPE_MESSAGE = 'parameters values must be role strings, e.g. {"c": "positive"} (got %s)'
PRIOR_RESTATED_MESSAGE = "prior is fixed by the catalogue; do not restate it"
TOLERATED_PARAMETER_OBJECT_KEYS = ({"role"}, {"role", "prior"})


def _json_type_name(value):
    """The JSON type word used in PARAMETER_SHAPE_MESSAGE."""
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _prior_matches_declaration(role, prior):
    """True when `prior` is exactly the fixed prior already declared for
    `role` (`count_models.PRIOR_RULES[role]`, which travels in the request as a
    JSON array, so a list and the stored tuple compare equal)."""
    declared = PRIOR_RULES.get(role)
    if declared is None:
        return False
    if isinstance(prior, (list, tuple)):
        return list(prior) == list(declared)
    return prior == declared


def normalize_parameter_roles(parameters):
    """Normalize a candidate's `parameters` mapping to plain role strings.

    Returns (roles, normalized, reason). `roles` maps each coefficient name to
    a role STRING, `normalized` is True when at least one value arrived as the
    tolerated object form, and `reason` is None on success or else the verbatim
    rejection message.

    Accepted per value: a string (kept as is); `{"role": X}`; and
    `{"role": X, "prior": P}` when P is exactly the declaration already
    published for role X - a DIFFERENT prior is a prior modification, which the
    contract forbids outright, so it is rejected with PRIOR_RESTATED_MESSAGE
    rather than silently dropped. Every other object or non-string value is
    rejected with PARAMETER_SHAPE_MESSAGE. An unknown role normalizes and is
    left to the judge, which owns the role vocabulary.
    """
    roles, normalized = {}, False
    for name, value in parameters.items():
        if isinstance(value, str):
            roles[name] = value
            continue
        if isinstance(value, dict) and set(value) in TOLERATED_PARAMETER_OBJECT_KEYS:
            role = value["role"]
            if not isinstance(role, str):
                return roles, normalized, PARAMETER_SHAPE_MESSAGE % _json_type_name(role)
            if (
                "prior" in value
                and role in PRIOR_RULES
                and not _prior_matches_declaration(role, value["prior"])
            ):
                return roles, normalized, PRIOR_RESTATED_MESSAGE
            roles[name] = role
            normalized = True
            continue
        return roles, normalized, PARAMETER_SHAPE_MESSAGE % _json_type_name(value)
    return roles, normalized, None


def parse_candidate_object(
    text, cell, initial_family, initial_structures=None, allowed_families=None, priors=None
):
    """Parse ONE candidate JSON object (policy P1's response contract).

    The v2.0 prompt's contract is a single object, not an array: the model proposes one
    challenger per round. Judgement is delegated unchanged to
    count_models.judge_candidate, the same per-item judge parse_candidates (policy P4)
    and core_count.validate_cell_candidates use, so grammar, role and S0/F0
    rules cannot drift between the two policies.

    `initial_structures` is the task's own initial pool; its input names
    are the grammar namespace the candidate is validated against. It defaults to
    CORE's four structures over `("u", "z")`.

    `allowed_families` is this task/cell's admissible family menu,
    passed straight through to `count_models.judge_candidate`; None keeps
    CORE's own three-family menu (see judge_candidate's docstring). It does not
    change WHICH class is returned - this always builds a `count_models.
    Hypothesis`, whatever `initial_structures` was built from - a caller whose
    task uses a different native hypothesis class (`boxinggym_core.run`'s
    `ScalarHypothesis` for peregrines/dugongs) converts afterward.

    `priors` (LLM-proposed priors) is whether this run accepts
    the optional `priors` block; None - the default - means "whatever
    `count_models.llm_priors_enabled()` says", which is False unless the runner
    turned the switch on. OFF, a `priors` key is an extra key and lands in
    `ignored_keys` exactly as in earlier versions. ON, it is validated against the menu,
    the hyperparameter windows and the coefficient's role, and a violation
    REJECTS this candidate with a per-item reason - the role prior is never
    silently substituted.

    Returns (hypothesis, ignored) where `ignored` holds the tolerated
    fields plus the recorded note ({"ignored_parameters": {...},
    "ignored_keys": {...}, "note": "..."}; the note is "" when absent). Raises
    ValueError with the verbatim validation message on any failure - the caller
    (propose_one) appends that message to a single repair call.
    """
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise ValueError(f"reply is not valid JSON: {exc}") from exc
    if isinstance(payload, list):
        raise ValueError("reply must be one JSON object, not an array")  # noqa: TRY004 - a ValueError is what earns the repair turn
    return build_candidate(
        payload, cell, initial_family, initial_structures, allowed_families, priors
    )


def build_candidate(
    payload, cell, initial_family, initial_structures=None, allowed_families=None, priors=None
):
    """Judge ONE already decoded candidate object; the body both reply shapes share.

    Split out of `parse_candidate_object` so that the
    k > 1 array reply judges each of its items by exactly the same rules, in
    exactly the same order, with the same verbatim messages - there is no second
    implementation of the per-candidate contract. `parse_candidate_object` keeps
    its own "not an array" shape message, which belongs to the one-object reply
    contract and to nothing else.
    """
    shape = None
    if not isinstance(payload, dict):
        shape = "reply must be one JSON object"
    elif not REQUIRED_CANDIDATE_KEYS <= set(payload):
        shape = "proposal candidate must include expression/family/parameters"
    elif not isinstance(payload["parameters"], dict):
        shape = "parameters must map coefficient names to fixed roles"
    if shape is not None:
        raise ValueError(shape)
    roles, normalized_shape, shape_reason = normalize_parameter_roles(payload["parameters"])
    if shape_reason is not None:
        raise ValueError(shape_reason)
    dropped_parameters = {
        name: role for name, role in roles.items() if name in RESERVED_VARIANCE_PARAMETER_NAMES
    }
    kept = {
        name: role for name, role in roles.items() if name not in RESERVED_VARIANCE_PARAMETER_NAMES
    }
    candidate_note = payload.get("note", "")
    recorded = isinstance(candidate_note, str) and len(candidate_note) <= NOTE_MAX_CHARS
    priors_on = llm_priors_enabled() if priors is None else bool(priors)
    accepted = REQUIRED_CANDIDATE_KEYS | (OPTIONAL_CANDIDATE_KEYS if recorded else set())
    if priors_on:
        accepted = accepted | PRIORS_CANDIDATE_KEYS
    dropped_keys = {k: v for k, v in payload.items() if k not in accepted}
    judged = {"expression": payload["expression"], "family": payload["family"], "parameters": kept}
    structures = (
        initial_hypotheses(initial_family)
        if initial_structures is None
        else list(initial_structures)
    )
    names = tuple(structures[0].input_names)
    # The task's categorical block travels with its own initial
    # structures (empty for every task that declares no categorical input), so
    # an indexed coefficient `a[student]` an LLM proposes for a categorical task
    # is judged and reconstructed against the SAME namespace the initial pool
    # was built with.
    levels = tuple(getattr(structures[0], "categorical_levels", ()) or ())
    ok, reason = judge_candidate(
        judged, cell, structures, initial_family, names, allowed_families, levels
    )
    if not ok:
        raise ValueError(reason)
    # LLM-proposed priors: validated AFTER the judge, so a grammar/role/cell
    # violation keeps its exact earlier message and precedence, and so the roles
    # this validator reads are already known-good.
    coefficient_priors = ()
    if priors_on and "priors" in payload:
        coefficient_priors, priors_reason = validate_proposed_priors(payload["priors"], kept)
        if priors_reason is not None:
            raise ValueError(priors_reason)
    hypothesis = Hypothesis(
        payload["expression"],
        payload["family"],
        tuple(kept.items()),
        names,
        levels,
        coefficient_priors,
    )
    return hypothesis, {
        "ignored_parameters": dropped_parameters,
        "ignored_keys": dropped_keys,
        "normalized_parameter_shape": normalized_shape,
        "note": candidate_note if recorded else "",
    }


def candidate_rejection(index, item, reason):
    """One per-candidate `rejected_candidates` entry of an array reply (policy P4 and k > 1).

    The four keys these entries have always carried, plus (failure classification) the refusal's
    `failure_class` and `parameters`: the item's own `parameters` value verbatim
    when it is an object, else None. Record fields only - nothing reads them to decide.
    """
    mapping = item if isinstance(item, dict) else {}
    parameters = mapping.get("parameters")
    return {"index": index, "expression": mapping.get("expression"),
            "family": mapping.get("family"), "reason": reason,
            "parameters": parameters if isinstance(parameters, dict) else None,
            "failure_class": failure_class(reason=reason)}


def parse_candidate_array(
    text, cell, initial_family, initial_structures=None, allowed_families=None,
    priors=None, proposal_limit=1, archive_review=False,
):
    """Parse a k > 1 reply: a JSON array of at most `proposal_limit` candidates.

    Per-candidate fail-soft, the same contract the batch
    path has always had and the same per-item judge the one-object reply uses
    (`build_candidate`): a grammar, role or cell violation rejects THAT
    candidate and is recorded with its index, and the accepted ones are returned
    in payload order. Batch-level failure - which the caller turns into one
    repair call, exactly as it does for a malformed single object - is only:
    text that is not JSON, a payload that is not an array, an array longer than
    the run's own limit, an empty array, and an array in which nothing was
    accepted.

    Returns `(models, ignored)` where `ignored` carries the per-candidate lists
    plus `rejected_candidates`, so the caller's transport record keeps the shape
    it already writes.

    `archive_review=True` reads an archive review round's reply through
    `parse_review_reply`.
    """
    if archive_review:
        return parse_review_reply(text, cell, initial_family, initial_structures, allowed_families,
                                  priors, proposal_limit=proposal_limit)
    validate_proposal_limit(proposal_limit)
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise ValueError(f"reply is not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"reply must be a JSON array of {proposal_limit} candidates")  # noqa: TRY004 - repair-eligible
    if not payload or len(payload) > proposal_limit:
        raise ValueError(f"reply must hold 1..{proposal_limit} candidates")
    models, notes, rejected = [], [], []
    for index, item in enumerate(payload):
        try:
            hypothesis, ignored = build_candidate(
                item, cell, initial_family, initial_structures, allowed_families, priors
            )
        except Exception as exc:  # noqa: BLE001 - per-candidate rejection, never a raise
            rejected.append(candidate_rejection(index, item, str(exc)))
            continue
        models.append(hypothesis)
        notes.append(ignored)
    if not models:
        raise ValueError(
            ARRAY_REFUSAL_PREFIX
            + "; ".join(entry["reason"] for entry in rejected)
        )
    return models, {
        "ignored_parameters": [note["ignored_parameters"] for note in notes],
        "ignored_keys": [note["ignored_keys"] for note in notes],
        "normalized_parameter_shape": [note["normalized_parameter_shape"] for note in notes],
        "note": next((note["note"] for note in notes if note["note"]), ""),
        "rejected_candidates": rejected,
    }


def parse_review_reply(
    text, cell, initial_family, initial_structures=None, allowed_families=None,
    priors=None, proposal_limit=3,
):
    """An archive review round's reply, {"candidates": [...], "revive": [...]}.

    `candidates` is judged exactly as `parse_candidate_array` judges an array reply - the same
    batch-level failures, and so the same single repair turn. `revive` is optional and never
    earns a repair: a missing key, a value that is not an array, or a bare array reply (the
    candidates without the object) leaves it empty and says why in `revive_note`. The list is
    passed on as given (`revive_requested`); which ids are valid is the runner's to decide,
    since only the runner holds the archive.
    """
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise ValueError(f"reply is not valid JSON: {exc}") from exc
    note, extra_keys = None, []
    if isinstance(payload, dict) and "candidates" in payload:
        shape, candidates, revive = "object", payload["candidates"], payload.get("revive")
        extra_keys = sorted(set(payload) - {"candidates", "revive"})
        if "revive" not in payload:
            revive, note = [], "no revive key; nothing revived"
        elif not isinstance(revive, list):
            revive, note = [], f"revive is a JSON {_json_type_name(revive)}, not an array; ignored"
    elif isinstance(payload, list):
        shape, candidates, revive = "array", payload, []
        note = "the reply is the candidates array alone; nothing revived"
    else:
        raise ValueError(
            'reply must be one JSON object {"candidates": [...], "revive": [...]} with '
            f"{proposal_limit} candidates")  # repair-eligible, like every shape failure
    models, ignored = parse_candidate_array(
        json.dumps(candidates), cell, initial_family, initial_structures, allowed_families,
        priors, proposal_limit=proposal_limit)
    return models, {
        # `reply_shape` is the same value under the name the BoxingGym records use.
        **ignored, "review_reply_shape": shape, "reply_shape": shape, "revive_requested": revive,
        # True when the reply carried no revive array at all (an explicit [] is a choice).
        "revive_missing": note is not None,
        **({"revive_note": note} if note else {}),
        **({"review_ignored_keys": extra_keys} if extra_keys else {}),
    }


def parse_candidates(text, cell, initial_family, allowed_families=None):
    """Parse an LLM proposal array into validated Hypotheses.

    Per-candidate fail-soft contract: grammar, allowed-role, and S0/F0
    cell-constraint violations reject ONLY that candidate (judged by
    count_models.judge_candidate, shared with core_count.validate_cell_
    candidates), recorded per event as `rejected_candidates`
    ([{index, expression, family, reason, parameters, failure_class}, ...] -
    `candidate_rejection` - index into the raw payload
    list). Candidates that judge_candidate accepts proceed to admission;
    `models` contains only those, in payload order, so `len(models)` is the
    accepted count (see AnthropicCountProposer.propose's
    valid_candidate_count). Batch-level failure - the whole event raising,
    which the caller turns into transport status "proposal_failed" - remains
    ONLY for: text is not parseable JSON, the parsed payload is not a list,
    or the list has more than three items. A candidate missing a required key
    or with a non-dict `parameters` is also rejected per-candidate (it is not
    one of those three batch cases, and judge_candidate assumes shape-valid
    input - see its docstring - so this shape check stays here).

    Unchanged from earlier versions: a reserved variance-parameter name (rho, pi) found
    inside a candidate's `parameters`, or any candidate key beyond
    expression/family/parameters, is ignored rather than rejected, and
    recorded per accepted candidate in the returned ignored-fields mapping
    (each a list aligned to `models`, entries are {name: original_value}
    dicts, empty when nothing was dropped for that candidate).

    `allowed_families` is threaded straight through to
    count_models.judge_candidate for every item in the batch; None keeps
    CORE's own three-family menu, so `AnthropicCountProposer.propose` (which
    never passes this argument) is unchanged.
    """
    payload = json.loads(text)
    if not isinstance(payload, list) or len(payload) > 3:
        raise ValueError("proposal must be an array of at most three models")
    initial_structures = initial_hypotheses(initial_family)
    models, ignored_parameters, ignored_keys, rejected_candidates = [], [], [], []
    normalized_shapes = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or not REQUIRED_CANDIDATE_KEYS <= set(item):
            reason = "proposal candidate must include expression/family/parameters"
        elif not isinstance(item["parameters"], dict):
            reason = "parameters must map coefficient names to fixed roles"
        else:
            roles, normalized_shape, reason = normalize_parameter_roles(item["parameters"])
            if reason is not None:
                rejected_candidates.append(candidate_rejection(index, item, reason))
                continue
            dropped_params = {
                name: role
                for name, role in roles.items()
                if name in RESERVED_VARIANCE_PARAMETER_NAMES
            }
            kept_params = {
                name: role
                for name, role in roles.items()
                if name not in RESERVED_VARIANCE_PARAMETER_NAMES
            }
            dropped_keys = {k: v for k, v in item.items() if k not in REQUIRED_CANDIDATE_KEYS}
            judged = {
                "expression": item["expression"],
                "family": item["family"],
                "parameters": kept_params,
            }
            ok, reason = judge_candidate(
                judged, cell, initial_structures, initial_family, allowed_families=allowed_families
            )
            if ok:
                models.append(
                    Hypothesis(item["expression"], item["family"], tuple(kept_params.items()))
                )
                ignored_parameters.append(dropped_params)
                ignored_keys.append(dropped_keys)
                normalized_shapes.append(normalized_shape)
                continue
        rejected_candidates.append(candidate_rejection(index, item, reason))
    return models, {
        "ignored_parameters": ignored_parameters,
        "ignored_keys": ignored_keys,
        "normalized_parameter_shape": normalized_shapes,
        "rejected_candidates": rejected_candidates,
    }


class AnthropicCountProposer:
    """LLM proposer over the Anthropic Messages API: writes one request record and one response
    record per proposal (a repair turn shares its proposal's pair) under `records` and returns
    validated count-model hypotheses (policies P1 and P4)."""

    def __init__(self, records, *, allow_live=False, model=DEFAULT_MODEL, client=None,
                 max_tokens=None, max_paid_calls=None, reasoning=None, response_format=None):
        self.records = Path(records)
        self.model = model
        # An unstated budget is the 1600 this transport always sent.
        self.max_tokens = 1600 if max_tokens is None else parse_max_tokens(max_tokens)
        # `effort=<level>[,thinking=adaptive]` becomes `output_config.effort` and
        # `thinking: {"type": "adaptive"}` on every request; unstated, the body keeps its bytes.
        self.reasoning = validate_provider_reasoning("anthropic", normalize_reasoning(reasoning))
        # JSON mode is the stock OpenAI transport's; Anthropic takes no response format.
        if parse_response_format(response_format) is not None:
            raise ValueError("response_format json_object is the stock OpenAI transport's; "
                             "provider anthropic takes no response format")
        # None is no guard, the counter exactly as it always was.
        self.max_paid_calls = None if max_paid_calls is None else parse_max_paid_calls(max_paid_calls)
        self.attempts = 0
        self.injected_client = client is not None
        if client is None:
            if not allow_live:
                raise ValueError("live API disabled; explicit paid-call authorization required")
            # The SDK constructor call is INSIDE a recording try
            # so an auth/proxy/header error that echoes the credential is
            # re-raised with its str, repr and traceback masked, instead of
            # propagating verbatim out of __init__ (review section 2, last row).
            try:
                import anthropic

                client = anthropic.Anthropic(max_retries=0, timeout=120.0)
            except BaseException as exc:  # noqa: BLE001 - masked, then re-raised
                raise masked_exception(exc) from None
        self.client = client
        self.records.mkdir(exist_ok=False)

    def _body(self, system, messages):
        """The Messages API request: the original four keys, plus `output_config.effort`
        and `thinking` when a reasoning spec was stated - appended, so an unstated spec leaves
        the body byte for byte what it was."""
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": messages}
        if self.reasoning is not None:
            body["output_config"] = {"effort": self.reasoning["effort"]}
            if "thinking" in self.reasoning:
                body["thinking"] = {"type": self.reasoning["thinking"]}
                if "display" in self.reasoning:
                    body["thinking"]["display"] = self.reasoning["display"]
        return body

    def propose_one(
        self,
        packet_text,
        *,
        context,
        initial_family,
        repair_attempts=1,
        repair_receipt=None,
        initial_structures=None,
        allowed_families=None,
        max_rounds=DEFAULT_MAX_PROPOSAL_ROUNDS,
        admit=None,
        proposal_limit=1,
        burst=False,
        slots=None,
        archive_review=False,
    ):
        """Policy P1: one candidate per round, at most one format-repair re-call.

        `packet_text` is the already serialized belief packet (built by
        core_count from public data only - this module never sees a fit, a
        score or the evaluation set). The system text is the selected stored prompt
        file (v2.0 by default; see `composed_system_prompt` for what k > 1, a burst or
        an archive review round appends), byte-identical across tasks; its sha256 goes
        into the record.

        On a validation failure the same conversation is continued once with
        the verbatim validation message appended, and no further. Two failures
        leave status "proposal_failed" with both messages in
        `rejected_candidates`.

        `admit` is an optional callable the runner supplies; it is
        called with each parsed candidate before the candidate is returned and
        may raise PriorProbeRejection, which earns the same single repair turn
        as a contract failure. None (the default and every mock path) changes
        nothing.

        `allowed_families` is passed straight through to
        `parse_candidate_object`/`count_models.judge_candidate`; None keeps
        CORE's own three-family menu, so CORE's call (which never passes this
        argument) is unchanged. See `count_models.judge_candidate`'s docstring.
        """
        keys = {"run_id", "world_seed", "cell", "proposal_rep", "round"}
        max_rounds = _validate_max_rounds(max_rounds)
        # Exactly the five identity keys, or those five plus the
        # optional `component` a multi-output task adds - nothing else.
        component = _validate_component(context)
        burst = _validate_burst(context, burst)
        if (set(context) - {"component"} - (BURST_CONTEXT_KEYS if burst else set()) != keys
                or not (0 if burst else 1) <= context["round"] <= max_rounds):
            raise ValueError("explicit run/world/cell/proposal-replicate/round identity required")
        cell = context["cell"]
        if cell not in {f"{s}{f}{a}" for s in "01" for f in "01" for a in "012"} or cell[:2] == "00":
            raise ValueError("invalid cell, or the fixed-pool cell (no structure and no family expansion) forbids proposal calls")
        if not isinstance(packet_text, str) or not packet_text:
            raise ValueError("belief packet text required")
        repair_attempts = _validate_repair_attempts(
            repair_attempts, receipt=repair_receipt, context=context, burst=burst,
            request_identity={"proposer": "anthropic", "model": self.model,
                              "initial_family": initial_family})
        # k = 1 returns the stored file's text and the
        # identity block this call has always written, byte for byte.
        system, prompt_block = (composed_system_prompt(
            burst=burst, burst_paragraphs=None if slots is None else slots.burst_paragraphs())
            if burst else composed_system_prompt(proposal_limit, archive_review=archive_review))
        messages = [{"role": "user", "content": packet_text}]
        identity = {
            **context,
            "provider": "anthropic",
            "requested_model": self.model,
            "policy": "P1",
            "decoding": decoding_identity(self.max_tokens, self.reasoning, provider="anthropic"),
            **prompt_block,
            "packet_bytes": len(packet_text.encode()),
            "packet_sha256": hashlib.sha256(packet_text.encode()).hexdigest(),
        }
        prefix = _round_record_prefix(self.records, context, component)
        started = time.perf_counter()
        record = {
            **identity,
            "status": "not_returned",
            "returned_model": None,
            "api_request_attempted": True,
            "client_injected": self.injected_client,
            "sdk_retries": 0,
            "temperature_sent": False,
            "calls": 0,
            "repairs": 0,
            "rejected_candidates": [],
            "responses": [],
        }
        if slots is not None:
            # Record-only stage block (never sent).
            record.update(slots.record_fields(burst))
        bodies, models = [], []
        budget_stop = None
        for attempt in range(1 + repair_attempts):
            body = self._body(system, list(messages))
            # Counted, or refused before the client is called. A refused first
            # call writes nothing; a refused repair turn keeps the paid call's record below.
            try:
                _reserve_paid_call(self)
            except PaidCallBudgetExhausted as exc:
                if not attempt:
                    raise
                budget_stop = exc
                break
            bodies.append(body)
            record["calls"] += 1
            if attempt:
                record["repairs"] += 1
            raw_text = None
            persisted = False
            try:
                response = create_anthropic_message(self.client, body)
                # The reply text is masked AT CAPTURE, so the stored
                # raw_text, the note, the ignored fields and the assistant turn
                # that the repair call re-sends are all masked by construction.
                raw_text = mask_secrets(
                    "\n".join(block.text for block in response.content if block.type == "text")
                )
                usage, usage_known = _anthropic_usage(getattr(response, "usage", None))
                http_request_id = _http_request_id(response)
                # Identity, finish state and usage(known|unknown) are
                # persisted BEFORE anything is parsed, so a reply that fails
                # validation still leaves its ids and token counts behind.
                record["responses"].append(
                    {
                        "attempt": attempt,
                        "returned_model": response.model,
                        "response_id": response.id,
                        "http_request_id": http_request_id,
                        "stop_reason": response.stop_reason,
                        "usage": usage,
                        "usage_known": usage_known,
                        "raw_text": raw_text,
                        # Thinking blocks are counted, never parsed; the key exists
                        # only under a stated reasoning spec, so other records keep their shape.
                        **({"thinking_blocks": _thinking_block_count(response)}
                           if self.reasoning is not None else {}),
                        # The thinking summaries, recorded (masked) when the reply
                        # carried any; a reply without them adds no key.
                        **anthropic_thinking_record(response),
                    }
                )
                if slots is not None:
                    # Which turn of the conversation this reply answers.
                    record["responses"][-1]["turn_type"] = "repair" if attempt else "proposal"
                record["returned_model"] = response.model
                record["http_request_id"] = http_request_id
                persisted = True
                # An empty body is its own reply-contract failure,
                # whatever the stop reason says, in every transport.
                if blank_reply(raw_text):
                    raise EmptyResponseBody(response.stop_reason)
                if response.stop_reason != "end_turn":
                    raise ValueError("response did not finish with end_turn")
                if archive_review:
                    # An archive review round's object reply; its optional
                    # `revive` list never earns a repair (parse_review_reply).
                    proposed, ignored = parse_review_reply(
                        raw_text, cell, initial_family, initial_structures,
                        allowed_families, proposal_limit=proposal_limit,
                    )
                elif proposal_limit == 1 or burst:
                    hypothesis, ignored = parse_candidate_object(
                        raw_text, cell, initial_family, initial_structures, allowed_families
                    )
                    proposed = [hypothesis]
                else:
                    proposed, ignored = parse_candidate_array(
                        raw_text, cell, initial_family, initial_structures,
                        allowed_families, proposal_limit=proposal_limit,
                    )
                if admit is not None:
                    # The runner's admission hook (the prior probe). It
                    # returns for an admitted or already-known candidate and raises
                    # PriorProbeRejection (a ValueError, so repair-eligible) with the
                    # engine's sentence for a candidate that is non-finite under its
                    # own prior; that sentence is what the one repair turn carries.
                    # With k > 1 a refused candidate is refused alone, and the reply fails
                    # only when none is left (see `admit_candidates`).
                    proposed, refused = admit_candidates(
                        proposed, admit, proposal_limit=proposal_limit, burst=burst,
                        rejected=ignored.get("rejected_candidates", ()))
                    if refused:
                        ignored = {**ignored, "rejected_candidates": [
                            *ignored.get("rejected_candidates", []), *refused]}
            except Exception as exc:  # noqa: BLE001 - one repair, then the pool is kept
                # ONLY a reply-contract failure is repairable - a
                # ValueError raised after the response was persisted (invalid
                # JSON, the object/shape rules, the parameters-value shape, a
                # judge_candidate grammar/role/cell message, a finish/stop
                # reason that is not the completed one). An API or transport
                # error, an SDK object-contract error and a missing usage
                # object are NOT: they are recorded and end the round without
                # a second paid call (review section 3).
                repair_eligible = persisted and isinstance(exc, ValueError)
                masked_reason = mask_secrets(str(exc))
                rejected = {
                    "attempt": attempt,
                    "reason": masked_reason,
                    "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "status_code", None),
                    "repair_eligible": repair_eligible,
                }
                if repair_eligible:
                    # Which repair turn this failure earns - the key
                    # exists only where a repair turn is actually built.
                    rejected["repair_kind"] = repair_kind(exc)
                    if slots is not None:
                        rejected["repair_text_kind"] = repair_text_kind(rejected["repair_kind"])
                    # Failure classification: a judged reply's refusal names its class; a transport
                    # error (not repair-eligible) is no candidate record and names none.
                    rejected["failure_class"] = failure_class(
                        error_type=rejected["error_type"], reason=masked_reason,
                        repair_eligible=True)
                    # k > 1 requests (protocol change of 2026-10-06): the repair turn asks again for this request's own shape
                    # (k = 1 and the burst: today's texts and no new key).
                    texts, text_fields = repair_texts_for(
                        exc, proposal_limit, burst=burst, slots=slots,
                        review="archive_review_paragraph_sha256" in prompt_block)
                    rejected.update(text_fields)
                record["rejected_candidates"].append(rejected)
                if not repair_eligible:
                    break
                # Both halves of the repair turn are masked BEFORE
                # they go on the wire - the echoed assistant reply (masked at
                # capture) and the validation message. An empty reply
                # is never echoed back.
                messages = repair_messages(messages, exc, raw_text, masked_reason, **texts)
                continue
            models = proposed
            # An array reply's per-candidate rejections are appended
            # to the repair-turn rejections already on the record, never in their place.
            record["rejected_candidates"] = [*record["rejected_candidates"],
                                            *ignored.get("rejected_candidates", [])]
            record.update({k: v for k, v in ignored.items() if k != "rejected_candidates"})
            record["status"] = "parsed"
            break
        if budget_stop is not None:
            # The paid call(s) this proposal made stay recorded, under a stop status.
            record.update(status=BUDGET_EXHAUSTED, max_paid_calls=budget_stop.max_paid_calls)
        elif not models:
            record["status"] = "proposal_failed"
        # responses[*].usage summed exactly once. An unknown usage is
        # never counted as zero - it lands in `usage_missing_indices` and makes
        # `usage_total_complete` false, so a cost ledger reserves instead of
        # under-counting. NEVER add the top-level last_call_usage to this.
        usage_total, usage_complete, usage_missing = _usage_total(record["responses"])
        record["usage_total"] = usage_total
        record["usage_total_complete"] = usage_complete
        record["usage_missing_indices"] = usage_missing
        _write_once(
            prefix.with_suffix(".request.json"),
            {"identity": identity, "bodies": bodies, "client_injected": self.injected_client},
        )
        record.update(
            elapsed_seconds=time.perf_counter() - started,
            valid_candidate_count=len(models),
            silent_fallback=False,
        )
        # The RETURNED record is masked too, not just the written
        # file - the caller copies it into proposals/*.json and summary.json.
        record = mask_secrets_deep(record)
        _write_once(prefix.with_suffix(".response.json"), record)
        if budget_stop is not None:
            raise budget_stop
        return models, record

    def propose(self, public_observations, current_models, *, context, initial_family):
        """Policy P4: one batch call of up to three candidates at a CORE event (5, 10, 15 or 20).

        Inputs are the public training observations and the current models; returns (accepted
        hypotheses, masked response metadata). No retry: a failure records the error type and
        HTTP status, never the error text.
        """
        keys = {"run_id", "world_seed", "cell", "proposal_rep", "event"}
        if set(context) != keys or context["event"] not in (5, 10, 15, 20):
            raise ValueError("explicit run/world/cell/proposal-replicate/event identity required")
        cell = context["cell"]
        if cell not in {f"{s}{f}{a}" for s in "01" for f in "01" for a in "012"} or cell[:2] == "00":
            raise ValueError("invalid cell, or the fixed-pool cell (no structure and no family expansion) forbids proposal calls")
        if len(public_observations) != 4 + context["event"]:
            raise ValueError("proposal must follow the registered additional observation count")
        public = []
        for row in public_observations:
            u, y = float(row["u"]), float(row["y"])
            if (
                not math.isfinite(u)
                or not 0 <= u <= 1
                or not math.isfinite(y)
                or not 0 <= y <= 50
                or y != int(y)
            ):
                raise ValueError("public training input/count outside CORE support")
            public.append({"u": u, "y": int(y)})
        prompt = json.dumps(
            {
                "observations": public,
                "current_models": current_models,
                "cell": cell,
                "structure_expansion": cell[0] == "1",
                "allowed_families": list(FAMILIES) if cell[1] == "1" else [initial_family],
                "initial_structures": [
                    item.as_dict() for item in initial_hypotheses(initial_family)
                ],
                "fixed_prior_roles": {
                    role: rule
                    for role, rule in PRIOR_RULES.items()
                    if role not in RESERVED_VARIANCE_PARAMETER_NAMES
                },
                "max_pairs": 3,
            },
            sort_keys=True,
        )
        body = self._body(SYSTEM, [{"role": "user", "content": prompt}])
        identity = {
            **context,
            "provider": "anthropic",
            "requested_model": self.model,
            "decoding": decoding_identity(self.max_tokens, self.reasoning, provider="anthropic"),
            "prompt_sha256": hashlib.sha256((SYSTEM + prompt).encode()).hexdigest(),
        }
        prefix = self.records / f"event_{context['event']:02d}"
        # Counted (or refused) before the request record exists, so a refused
        # call leaves no record of a request that was never sent.
        _reserve_paid_call(self)
        _write_once(
            prefix.with_suffix(".request.json"),
            {"identity": identity, "body": body, "client_injected": self.injected_client},
        )
        started = time.perf_counter()
        metadata = {
            **identity,
            "status": "not_returned",
            "returned_model": None,
            "api_request_attempted": True,
            "client_injected": self.injected_client,
            "sdk_retries": 0,
            "temperature_sent": False,
        }
        models = []
        try:
            response = create_anthropic_message(self.client, body)
            texts = [block.text for block in response.content if block.type == "text"]
            # Masked at capture (see propose_one).
            raw_text = mask_secrets("\n".join(texts))
            usage, usage_known = _anthropic_usage(getattr(response, "usage", None))
            metadata.update(
                returned_model=response.model,
                response_id=response.id,
                http_request_id=_http_request_id(response),
                stop_reason=response.stop_reason,
                usage=usage,
                usage_known=usage_known,
                raw_text=raw_text,
                **({"thinking_blocks": _thinking_block_count(response)}
                   if self.reasoning is not None else {}),
                # The thinking summaries, as on the round path.
                **anthropic_thinking_record(response),
            )
            if blank_reply(raw_text):
                raise EmptyResponseBody(response.stop_reason)
            if response.stop_reason != "end_turn":
                raise ValueError("response did not finish with end_turn")
            models, ignored = parse_candidates(raw_text, cell, initial_family)
            metadata.update(ignored)
            metadata["status"] = "parsed"
        except Exception as exc:  # noqa: BLE001 - no retry; caller retains the current pool
            # NOTE: propose() deliberately records the error
            # TYPE and no error text at all - the pre-declared P4 contract
            # asserted by test_cell_restrictions_stop_reason_and_transport_
            # error_are_explicit ("not exposed" not in the record). Masking
            # would have been enough, but not recording is stronger, so the
            # existing contract stands and only propose_one carries a reason.
            metadata.update(
                status="proposal_failed",
                error_type=type(exc).__name__,
                http_status=getattr(exc, "status_code", None),
            )
        metadata.update(
            elapsed_seconds=time.perf_counter() - started,
            valid_candidate_count=len(models),
            silent_fallback=False,
        )
        # The returned metadata is masked too (see propose_one).
        metadata = mask_secrets_deep(metadata)
        _write_once(prefix.with_suffix(".response.json"), metadata)
        return models, metadata


def _numeric(value):
    """True for a real token count (a bool is not one)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _openai_usage(usage_obj):
    """Normalize the OpenAI SDK usage shape into ({input_tokens,
    output_tokens, cached_tokens}, known). `usage_obj` is `response.usage`:
    `.prompt_tokens` and `.completion_tokens` are normally present;
    `.prompt_tokens_details.cached_tokens` is only present for some
    models/requests, so both the details object and the attribute are optional
    (0 when absent) - matches both the real openai>=1.40 SDK response and the
    injected fakes in test_count_proposer.py.

    A MISSING usage object
    (`response.usage is None`) or one whose token counts are not numbers is NOT
    an error and NOT a reason to re-call. It returns (None, False) so the
    record can say "unknown" and the reply can still be parsed - the pre-patch
    `usage_obj.prompt_tokens` AttributeError fell into the broad except and,
    because `raw_text` already existed, spent a second paid call on a reply
    that was perfectly valid.
    """
    if usage_obj is None:
        return None, False
    try:
        details = getattr(usage_obj, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details is not None else 0
        usage = {
            "input_tokens": usage_obj.prompt_tokens,
            "output_tokens": usage_obj.completion_tokens,
            "cached_tokens": cached or 0,
        }
        # Reasoning tokens are part of `completion_tokens` and of the
        # reply budget, so a reply can finish on length with an empty body.
        # The key exists ONLY when the provider reported the count, which is
        # why a response without the details object records exactly the three
        # keys it recorded before (and `_usage_total` sums whatever is there).
        completion_details = getattr(usage_obj, "completion_tokens_details", None)
        reasoning = getattr(completion_details, "reasoning_tokens", None)
        if _numeric(reasoning):
            usage["reasoning_tokens"] = reasoning
    except Exception:  # noqa: BLE001 - an SDK object contract change is not a format failure
        return None, False
    if not _numeric(usage["input_tokens"]) or not _numeric(usage["output_tokens"]):
        return None, False
    return usage, True


def _anthropic_usage(usage_obj):
    """({provider dump}, known) for the Anthropic usage object - the same
    contract as `_openai_usage`: a missing usage object, a dump that fails, or
    a dump with no numeric field at all is recorded as unknown instead of
    raising into the repair path."""
    if usage_obj is None:
        return None, False
    try:
        usage = usage_obj.model_dump(mode="json")
    except Exception:  # noqa: BLE001 - an SDK object contract change is not a format failure
        return None, False
    if not isinstance(usage, dict) or not any(_numeric(value) for value in usage.values()):
        return None, False
    return usage, True


def _http_request_id(response):
    """The HTTP request id (`x-request-id`) of the call when the SDK exposes
    it, else None.

    NOT the completion id: `response.id` is `chatcmpl_...` (review section 5),
    which is why the record now names them apart. openai 1.40.3 - the version
    this tree runs - does not attach `_request_id` to the parsed model at all,
    so None is the normal, honest answer there; newer SDKs (1.109 in the fb1
    venv) do attach it, and a raw-response wrapper carries the header itself.
    Never raises and never fails a call over an id.
    """
    try:
        value = getattr(response, "_request_id", None)
        if isinstance(value, str) and value:
            return value
        for attribute in ("_raw_response", "http_response", "raw_response"):
            headers = getattr(getattr(response, attribute, None), "headers", None)
            if headers is None:
                continue
            value = headers.get("x-request-id")
            if isinstance(value, str) and value:
                return value
    except Exception:  # noqa: BLE001 - an id is never worth failing a call for
        return None
    return None


def _usage_total(responses):
    """Sum `responses[*]["usage"]` exactly ONCE each (review section 4).

    Returns (total, complete, missing): `missing` holds the indices into
    `responses` whose usage is unknown. An unknown usage is NEVER counted as
    zero - it makes `complete` False and shows up in `missing`, so a cost
    ledger reserves a conservative budget instead of under-counting. The
    top-level `last_call_usage` is the LAST response's usage alone and must
    never be added to this total.
    """
    total, missing = {}, []
    for index, item in enumerate(responses):
        usage = item.get("usage")
        if not item.get("usage_known", False) or not isinstance(usage, dict):
            missing.append(index)
            continue
        for key, value in usage.items():
            if _numeric(value):
                total[key] = total.get(key, 0) + value
    return total, not missing, missing


def _validate_repair_attempts(repair_attempts, *, receipt=None, context=None, burst=False,
                              request_identity=None):
    """Policy P1 permits 0 or 1; 2 only under a verified, registered CORE A8 launch contract.

    "at most one format repair" is an invariant of policy P1, not a tunable:
    an earlier `range(1 + max(0, int(repair_attempts)))` silently accepted 3
    and made four calls. The exception requires the same verified receipt and
    track allow-list as the runner; 3 is still refused before any request.
    """
    from .run_label_policy import validate_repair_attempts

    return validate_repair_attempts(
        repair_attempts, bundle="core" if receipt is not None else None,
        receipt=receipt, context=context, burst=burst, request_identity=request_identity)


class OpenAICountProposer:
    """Same policy P1/P4 contracts as AnthropicCountProposer, over the OpenAI
    Chat Completions transport. Differences are wire
    shape only: SYSTEM travels as the first `messages` entry (role "system")
    instead of a separate field; the reply text is `response.choices[0].
    message.content`; the finish signal is `choices[0].finish_reason ==
    "stop"` (Anthropic's "end_turn"); usage and the ids are normalized (see
    `_openai_usage`, `_http_request_id`) rather than stored as the provider's
    raw dump - those record keys were renamed to `last_call_usage` /
    `usage_total` and `completion_id` / `http_request_id`, because the old
    `usage` read like a run total and the old `request_id` held the completion
    id.
    The stock OpenAI path lets the SDK read `OPENAI_API_KEY`. The explicit
    compatibility path passes its named environment credential directly to
    the SDK constructor; transport records contain the name only.
    """

    def __init__(self, records, *, allow_live=False, model=DEFAULT_MODEL_OPENAI, client=None,
                 provider="openai", base_url=None, key_name="OPENAI_COMPAT_API_KEY", provider_order=None,
                 reasoning=None, max_tokens=None, max_paid_calls=None, response_format=None):
        if provider not in ("openai", "openai_compat"):
            raise ValueError("unsupported OpenAI transport provider")
        self.provider = provider
        self.compat_metadata = {}
        self.provider_order = normalize_provider_order(provider_order)
        # JSON mode (`response_format: {"type": "json_object"}`) on the stock path only.
        self.response_format = parse_response_format(response_format)
        if self.response_format is not None and provider != "openai":
            raise ValueError("response_format json_object is the stock OpenAI transport's; "
                             "provider openai_compat does not take it")
        # `reasoning` is validated for the provider - the stock path takes
        # `effort=<level>` only and sends the reasoning-model shape (`max_completion_tokens`,
        # `reasoning_effort`); the compatibility endpoint keeps the `extra_body` grammar. The
        # budget is statable everywhere and keeps its 1600 literal when it is not stated.
        self.reasoning = validate_provider_reasoning(provider, normalize_reasoning(reasoning))
        self.max_tokens = 1600 if max_tokens is None else parse_max_tokens(max_tokens)
        # None is no guard, the counter exactly as it always was.
        self.max_paid_calls = None if max_paid_calls is None else parse_max_paid_calls(max_paid_calls)
        if provider == "openai_compat":
            host = validate_openai_compat(base_url, key_name, model)
            self.compat_metadata = {"llm_base_url_host": host, "llm_key_name": key_name}
            if self.provider_order is not None:
                self.compat_metadata["provider_order"] = self.provider_order
        elif (base_url is not None or key_name != "OPENAI_COMPAT_API_KEY"
                or provider_order is not None):
            raise ValueError("compatibility options require provider openai_compat")
        self.records = Path(records)
        self.model = model
        self.attempts = 0
        self.injected_client = client is not None
        if client is None:
            if not allow_live:
                raise ValueError("live API disabled; explicit paid-call authorization required")
            # See AnthropicCountProposer.__init__ - the SDK
            # constructor is inside a recording try and re-raised masked.
            try:
                import openai

                client = (openai.OpenAI(**compat_client_kwargs(base_url, key_name, model))
                          if provider == "openai_compat" else
                          openai.OpenAI(max_retries=0, timeout=120.0))
            except BaseException as exc:  # noqa: BLE001 - masked, then re-raised
                raise masked_exception(exc) from None
        self.client = client
        self.records.mkdir(exist_ok=False)

    def _body(self, messages):
        """Provider openai with a stated effort sends the reasoning-model shape -
        `max_completion_tokens` and `reasoning_effort`, no `max_tokens`. Every other case is the
        original body: `max_tokens`, and for openai_compat the D1 `extra_body` extensions.
        A stated response format is appended last, after the messages were checked for
        the word OpenAI's JSON mode requires - refused here, before any call."""
        if self.provider == "openai" and self.reasoning is not None:
            body = {
                "model": self.model,
                "max_completion_tokens": self.max_tokens,
                "reasoning_effort": self.reasoning["effort"],
                "messages": messages,
            }
        else:
            body = apply_extra_body(
                {"model": self.model, "max_tokens": self.max_tokens, "messages": messages},
                self.provider_order,
                self.reasoning,
            )
        if self.response_format is not None:
            texts = [message.get("content") for message in messages
                     if message.get("role") in ("system", "user")]
            if not any(isinstance(text, str) and JSON_MODE_WORD in text for text in texts):
                raise ValueError(
                    f"OpenAI JSON mode requires the word {JSON_MODE_WORD} in the system or user "
                    "messages; none of this request's messages carries it, so it was not sent")
            body["response_format"] = {"type": self.response_format}
        return body

    def propose_one(
        self,
        packet_text,
        *,
        context,
        initial_family,
        repair_attempts=1,
        repair_receipt=None,
        initial_structures=None,
        allowed_families=None,
        max_rounds=DEFAULT_MAX_PROPOSAL_ROUNDS,
        admit=None,
        proposal_limit=1,
        burst=False,
        slots=None,
        archive_review=False,
    ):
        """Policy P1: one candidate per round, at most one format-repair re-call.
        See AnthropicCountProposer.propose_one for the shared identity/packet
        contract - unchanged here down to the field names, except `provider`,
        the wire shape of `body`/`messages`, and the normalized usage/request
        id (see class docstring).
        """
        keys = {"run_id", "world_seed", "cell", "proposal_rep", "round"}
        max_rounds = _validate_max_rounds(max_rounds)
        # Exactly the five identity keys, or those five plus the
        # optional `component` a multi-output task adds - nothing else.
        component = _validate_component(context)
        burst = _validate_burst(context, burst)
        if (set(context) - {"component"} - (BURST_CONTEXT_KEYS if burst else set()) != keys
                or not (0 if burst else 1) <= context["round"] <= max_rounds):
            raise ValueError("explicit run/world/cell/proposal-replicate/round identity required")
        cell = context["cell"]
        if cell not in {f"{s}{f}{a}" for s in "01" for f in "01" for a in "012"} or cell[:2] == "00":
            raise ValueError("invalid cell, or the fixed-pool cell (no structure and no family expansion) forbids proposal calls")
        if not isinstance(packet_text, str) or not packet_text:
            raise ValueError("belief packet text required")
        repair_attempts = _validate_repair_attempts(
            repair_attempts, receipt=repair_receipt, context=context, burst=burst,
            request_identity={"proposer": self.provider, "model": self.model,
                              "initial_family": initial_family})
        # k = 1 returns the stored file's text and the
        # identity block this call has always written, byte for byte.
        system, prompt_block = (composed_system_prompt(
            burst=burst, burst_paragraphs=None if slots is None else slots.burst_paragraphs())
            if burst else composed_system_prompt(proposal_limit, archive_review=archive_review))
        messages = [{"role": "user", "content": packet_text}]
        identity = {
            **context,
            "provider": self.provider,
            **self.compat_metadata,
            "requested_model": self.model,
            "policy": "P1",
            "decoding": decoding_identity(self.max_tokens, self.reasoning, provider=self.provider,
                                          response_format=self.response_format),
            **prompt_block,
            "packet_bytes": len(packet_text.encode()),
            "packet_sha256": hashlib.sha256(packet_text.encode()).hexdigest(),
        }
        prefix = _round_record_prefix(self.records, context, component)
        started = time.perf_counter()
        record = {
            **identity,
            "status": "not_returned",
            "returned_model": None,
            **(compat_response_metadata() if self.provider == "openai_compat" else {}),
            "api_request_attempted": True,
            "client_injected": self.injected_client,
            "sdk_retries": 0,
            "temperature_sent": False,
            "calls": 0,
            "repairs": 0,
            "rejected_candidates": [],
            "responses": [],
            # `last_call_usage` is the LAST response's usage (the
            # old key was named `usage`, which read like a run total - review
            # section 4); `usage_total` below is the sum over responses[*].
            # `completion_id` is `chatcmpl_...` (the old key was named
            # `request_id`, which read like the HTTP x-request-id - section 5);
            # `http_request_id` is that header when the SDK exposes it.
            "last_call_usage": None,
            "last_call_usage_known": False,
            "completion_id": None,
            "http_request_id": None,
        }
        if slots is not None:
            # Record-only stage block (never sent).
            record.update(slots.record_fields(burst))
        bodies, models = [], []
        budget_stop = None
        for attempt in range(1 + repair_attempts):
            body = self._body([{"role": "system", "content": system}, *messages])
            # See the Anthropic transport - counted, or refused before the call.
            try:
                _reserve_paid_call(self)
            except PaidCallBudgetExhausted as exc:
                if not attempt:
                    raise
                budget_stop = exc
                break
            bodies.append(body)
            record["calls"] += 1
            if attempt:
                record["repairs"] += 1
            raw_text = None
            persisted = False
            try:
                response = create_openai_completion(
                    self.client, body, capture_http_request=self.provider == "openai_compat")
                choice = response.choices[0]
                # Masked at capture - see the Anthropic transport.
                raw_text = mask_secrets(choice.message.content)
                usage, usage_known = _openai_usage(getattr(response, "usage", None))
                http_request_id = _http_request_id(response)
                # Persist identity/finish/usage BEFORE parsing.
                record["responses"].append(
                    {
                        "attempt": attempt,
                        "returned_model": response.model,
                        **(compat_response_metadata(response) if self.provider == "openai_compat" else {}),
                        "completion_id": response.id,
                        "http_request_id": http_request_id,
                        "finish_reason": choice.finish_reason,
                        "usage": usage,
                        "usage_known": usage_known,
                        "raw_text": raw_text,
                    }
                )
                if slots is not None:
                    # Which turn of the conversation this reply answers.
                    record["responses"][-1]["turn_type"] = "repair" if attempt else "proposal"
                record["returned_model"] = response.model
                if self.provider == "openai_compat":
                    record.update(compat_response_metadata(response))
                record["completion_id"] = response.id
                record["http_request_id"] = http_request_id
                record["last_call_usage"] = usage
                record["last_call_usage_known"] = usage_known
                persisted = True
                # An empty body is its own reply-contract failure,
                # whatever the finish reason says, in every transport.
                if blank_reply(raw_text):
                    raise EmptyResponseBody(choice.finish_reason)
                if choice.finish_reason != "stop":
                    raise ValueError("response did not finish with stop")
                if archive_review:
                    # An archive review round's object reply; its optional
                    # `revive` list never earns a repair (parse_review_reply).
                    proposed, ignored = parse_review_reply(
                        raw_text, cell, initial_family, initial_structures,
                        allowed_families, proposal_limit=proposal_limit,
                    )
                elif proposal_limit == 1 or burst:
                    hypothesis, ignored = parse_candidate_object(
                        raw_text, cell, initial_family, initial_structures, allowed_families
                    )
                    proposed = [hypothesis]
                else:
                    proposed, ignored = parse_candidate_array(
                        raw_text, cell, initial_family, initial_structures,
                        allowed_families, proposal_limit=proposal_limit,
                    )
                if admit is not None:
                    # The runner's admission hook (the prior probe). It
                    # returns for an admitted or already-known candidate and raises
                    # PriorProbeRejection (a ValueError, so repair-eligible) with the
                    # engine's sentence for a candidate that is non-finite under its
                    # own prior; that sentence is what the one repair turn carries.
                    # With k > 1 a refused candidate is refused alone, and the reply fails
                    # only when none is left (see `admit_candidates`).
                    proposed, refused = admit_candidates(
                        proposed, admit, proposal_limit=proposal_limit, burst=burst,
                        rejected=ignored.get("rejected_candidates", ()))
                    if refused:
                        ignored = {**ignored, "rejected_candidates": [
                            *ignored.get("rejected_candidates", []), *refused]}
            except Exception as exc:  # noqa: BLE001 - one repair, then the pool is kept
                # ONLY a reply-contract failure is repairable - a
                # ValueError raised after the response was persisted (invalid
                # JSON, the object/shape rules, the parameters-value shape, a
                # judge_candidate grammar/role/cell message, a finish/stop
                # reason that is not the completed one). An API or transport
                # error, an SDK object-contract error and a missing usage
                # object are NOT: they are recorded and end the round without
                # a second paid call (review section 3).
                repair_eligible = persisted and isinstance(exc, ValueError)
                masked_reason = mask_secrets(str(exc))
                rejected = {
                    "attempt": attempt,
                    "reason": masked_reason,
                    "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "status_code", None),
                    "repair_eligible": repair_eligible,
                }
                if repair_eligible:
                    # Which repair turn this failure earns - the key
                    # exists only where a repair turn is actually built.
                    rejected["repair_kind"] = repair_kind(exc)
                    if slots is not None:
                        rejected["repair_text_kind"] = repair_text_kind(rejected["repair_kind"])
                    # Failure classification: a judged reply's refusal names its class; a transport
                    # error (not repair-eligible) is no candidate record and names none.
                    rejected["failure_class"] = failure_class(
                        error_type=rejected["error_type"], reason=masked_reason,
                        repair_eligible=True)
                    # k > 1 requests (protocol change of 2026-10-06): the repair turn asks again for this request's own shape
                    # (k = 1 and the burst: today's texts and no new key).
                    texts, text_fields = repair_texts_for(
                        exc, proposal_limit, burst=burst, slots=slots,
                        review="archive_review_paragraph_sha256" in prompt_block)
                    rejected.update(text_fields)
                record["rejected_candidates"].append(rejected)
                if not repair_eligible:
                    break
                # Both halves of the repair turn are masked BEFORE
                # they go on the wire - the echoed assistant reply (masked at
                # capture) and the validation message. An empty reply
                # is never echoed back.
                messages = repair_messages(messages, exc, raw_text, masked_reason, **texts)
                continue
            models = proposed
            # An array reply's per-candidate rejections are appended
            # to the repair-turn rejections already on the record, never in their place.
            record["rejected_candidates"] = [*record["rejected_candidates"],
                                            *ignored.get("rejected_candidates", [])]
            record.update({k: v for k, v in ignored.items() if k != "rejected_candidates"})
            record["status"] = "parsed"
            break
        if budget_stop is not None:
            # The paid call(s) this proposal made stay recorded, under a stop status.
            record.update(status=BUDGET_EXHAUSTED, max_paid_calls=budget_stop.max_paid_calls)
        elif not models:
            record["status"] = "proposal_failed"
        # responses[*].usage summed exactly once. An unknown usage is
        # never counted as zero - it lands in `usage_missing_indices` and makes
        # `usage_total_complete` false, so a cost ledger reserves instead of
        # under-counting. NEVER add the top-level last_call_usage to this.
        usage_total, usage_complete, usage_missing = _usage_total(record["responses"])
        record["usage_total"] = usage_total
        record["usage_total_complete"] = usage_complete
        record["usage_missing_indices"] = usage_missing
        _write_once(
            prefix.with_suffix(".request.json"),
            {"identity": identity, "bodies": bodies, "client_injected": self.injected_client},
        )
        record.update(
            elapsed_seconds=time.perf_counter() - started,
            valid_candidate_count=len(models),
            silent_fallback=False,
        )
        # The RETURNED record is masked too, not just the written
        # file - the caller copies it into proposals/*.json and summary.json.
        record = mask_secrets_deep(record)
        _write_once(prefix.with_suffix(".response.json"), record)
        if budget_stop is not None:
            raise budget_stop
        return models, record

    def propose(self, public_observations, current_models, *, context, initial_family):
        """Policy P4 over the OpenAI wire shape; see AnthropicCountProposer.propose.

        Returns (accepted hypotheses, masked response metadata); no retry on failure.
        """
        keys = {"run_id", "world_seed", "cell", "proposal_rep", "event"}
        if set(context) != keys or context["event"] not in (5, 10, 15, 20):
            raise ValueError("explicit run/world/cell/proposal-replicate/event identity required")
        cell = context["cell"]
        if cell not in {f"{s}{f}{a}" for s in "01" for f in "01" for a in "012"} or cell[:2] == "00":
            raise ValueError("invalid cell, or the fixed-pool cell (no structure and no family expansion) forbids proposal calls")
        if len(public_observations) != 4 + context["event"]:
            raise ValueError("proposal must follow the registered additional observation count")
        public = []
        for row in public_observations:
            u, y = float(row["u"]), float(row["y"])
            if (
                not math.isfinite(u)
                or not 0 <= u <= 1
                or not math.isfinite(y)
                or not 0 <= y <= 50
                or y != int(y)
            ):
                raise ValueError("public training input/count outside CORE support")
            public.append({"u": u, "y": int(y)})
        prompt = json.dumps(
            {
                "observations": public,
                "current_models": current_models,
                "cell": cell,
                "structure_expansion": cell[0] == "1",
                "allowed_families": list(FAMILIES) if cell[1] == "1" else [initial_family],
                "initial_structures": [
                    item.as_dict() for item in initial_hypotheses(initial_family)
                ],
                "fixed_prior_roles": {
                    role: rule
                    for role, rule in PRIOR_RULES.items()
                    if role not in RESERVED_VARIANCE_PARAMETER_NAMES
                },
                "max_pairs": 3,
            },
            sort_keys=True,
        )
        body = self._body([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        identity = {
            **context,
            "provider": self.provider,
            **self.compat_metadata,
            "requested_model": self.model,
            "decoding": decoding_identity(self.max_tokens, self.reasoning, provider=self.provider,
                                          response_format=self.response_format),
            "prompt_sha256": hashlib.sha256((SYSTEM + prompt).encode()).hexdigest(),
        }
        prefix = self.records / f"event_{context['event']:02d}"
        # Counted (or refused) before the request record exists.
        _reserve_paid_call(self)
        _write_once(
            prefix.with_suffix(".request.json"),
            {"identity": identity, "body": body, "client_injected": self.injected_client},
        )
        started = time.perf_counter()
        metadata = {
            **identity,
            "status": "not_returned",
            "returned_model": None,
            **(compat_response_metadata() if self.provider == "openai_compat" else {}),
            "api_request_attempted": True,
            "client_injected": self.injected_client,
            "sdk_retries": 0,
            "temperature_sent": False,
        }
        models = []
        try:
            response = create_openai_completion(
                    self.client, body, capture_http_request=self.provider == "openai_compat")
            choice = response.choices[0]
            # Masked at capture (see propose_one).
            raw_text = mask_secrets(choice.message.content)
            usage, usage_known = _openai_usage(getattr(response, "usage", None))
            if self.provider == "openai_compat":
                metadata.update(compat_response_metadata(response))
            metadata.update(
                returned_model=response.model,
                completion_id=response.id,
                http_request_id=_http_request_id(response),
                finish_reason=choice.finish_reason,
                usage=usage,
                usage_known=usage_known,
                raw_text=raw_text,
            )
            if blank_reply(raw_text):
                raise EmptyResponseBody(choice.finish_reason)
            if choice.finish_reason != "stop":
                raise ValueError("response did not finish with stop")
            models, ignored = parse_candidates(raw_text, cell, initial_family)
            metadata.update(ignored)
            metadata["status"] = "parsed"
        except Exception as exc:  # noqa: BLE001 - no retry; caller retains the current pool
            # NOTE: propose() deliberately records the error
            # TYPE and no error text at all - the pre-declared P4 contract
            # asserted by test_cell_restrictions_stop_reason_and_transport_
            # error_are_explicit ("not exposed" not in the record). Masking
            # would have been enough, but not recording is stronger, so the
            # existing contract stands and only propose_one carries a reason.
            metadata.update(
                status="proposal_failed",
                error_type=type(exc).__name__,
                http_status=getattr(exc, "status_code", None),
            )
        metadata.update(
            elapsed_seconds=time.perf_counter() - started,
            valid_candidate_count=len(models),
            silent_fallback=False,
        )
        # The returned metadata is masked too (see propose_one).
        metadata = mask_secrets_deep(metadata)
        _write_once(prefix.with_suffix(".response.json"), metadata)
        return models, metadata


class OpenAICompatCountProposer(OpenAICountProposer):
    """Explicit OpenAI-compatible endpoint; model and endpoint never fall back."""
    def __init__(self, records, *, base_url, key_name="OPENAI_COMPAT_API_KEY", model=None,
                 allow_live=False, client=None, provider_order=None, reasoning=None,
                 max_tokens=None, max_paid_calls=None, response_format=None):
        super().__init__(records, provider="openai_compat", base_url=base_url,
                         key_name=key_name, model=model, allow_live=allow_live, client=client,
                         provider_order=provider_order, reasoning=reasoning, max_tokens=max_tokens,
                         max_paid_calls=max_paid_calls, response_format=response_format)
