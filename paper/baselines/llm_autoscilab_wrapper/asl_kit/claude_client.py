"""The Claude path for the released AutoSciLab LLMClient (the only change that can alter the agent).

The released client (autoscilab/llm/client.py, unchanged) talks to an OpenAI-shaped object:
`self._oa_client.chat.completions.create(model=..., messages=[system, user, ...], max_tokens=...,
[temperature=...], [response_format={"type": "json_object"}])` and reads
`response.choices[0].message.content`. `install_claude_path()` wraps `LLMClient.__init__`: for a
model name starting with "claude" it builds the released client's fields and puts an
`AnthropicChatShim` where the OpenAI client would be; every other model name runs the released
`__init__` untouched. `complete()` and `complete_json()` (prompts, schema-in-system text, the
fence stripping, the JSON parsing and the retry loop) run exactly as released.

What the shim sends (Anthropic Messages API, official `anthropic` SDK):
  model       = the requested model (claude-opus-5-5)
  system      = the released system message, byte for byte
  messages    = the released user/assistant turns, byte for byte
  thinking    = {"type": "adaptive"}             (Opus 5.5 cannot disable thinking)
  output_config = {"effort": "medium"}
  max_tokens  = max(requested, 16000)            (thinking tokens count against max_tokens; the
                released 4096/8192/16 caps would cut the answer or leave it empty)
Dropped, and counted per call: `temperature` (Opus 5.5 rejects sampling parameters; the released
configs leave it unset anyway) and `response_format` (no json_object mode on the Messages API;
the released client already puts the schema in the system prompt). When json_object was requested
and the reply text is not valid JSON, the outermost {...} span is returned if it parses, which is
what json_object mode guarantees on the OpenAI side; every such call is flagged `json_extracted`.
A `refusal` stop reason raises (non-retryable) after its usage is recorded. No fallback model.

Every call is appended to <cell>/llm_usage.jsonl (tokens, USD at 4 / 20 USD per 1M input /
output tokens, input including cache tokens - the ledger rule of 2026-10-02), and the request
text and reply text to llm_requests.jsonl / llm_responses.jsonl for the leak audit. Before each
call the cell total and the folder total (all <folder>/*/llm_usage.jsonl) plus the worst case of
the next call (input estimate at 1.9 characters per token + max_tokens of output) must stay under
the caps, otherwise CostCapExceeded is raised, no request is sent, and a `refused_by_cap` row (0 USD)
is written so that a refusal swallowed by one of the released loop's try/except sites stays visible.
"""
import inspect
import json
import math
import os
import time
from types import SimpleNamespace

PRICE_INPUT_PER_M = 4.0     # USD per 1M input tokens, claude-opus-5-5
PRICE_OUTPUT_PER_M = 20.0   # USD per 1M output tokens, claude-opus-5-5
MAX_TOKENS_FLOOR = 16000
EFFORT = "medium"
CHARS_PER_TOKEN_ESTIMATE = 1.9   # measured 1.942 on 533 Opus 5.5 requests (bg_rerun_20261004), rounded down
DEFAULT_MODEL = "claude-opus-5-5"


class CostCapExceeded(RuntimeError):
    """Raised before a call that could cross a cap; the released retry loop does not retry it."""


class ClaudeRefusal(RuntimeError):
    """stop_reason == "refusal"; recorded, then raised (not retried)."""


def usd_of(usage):
    tin = (usage.get("input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0) \
        + (usage.get("cache_read_input_tokens") or 0)
    return (tin * PRICE_INPUT_PER_M + (usage.get("output_tokens") or 0) * PRICE_OUTPUT_PER_M) / 1e6


def _read_usd(path):
    total = 0.0
    try:
        with open(path) as fh:
            for line in fh:
                try:
                    total += float(json.loads(line).get("usd") or 0.0)
                except (ValueError, AttributeError):
                    continue   # a line being written by another process
    except FileNotFoundError:
        pass
    return total


class UsageLedger:
    """Per-cell usage file plus cell and folder caps (folder = parent of the cell directory)."""

    def __init__(self, cell_dir, cap_cell_usd, cap_folder_usd, mock=False):
        self.cell_dir = os.path.abspath(cell_dir)
        self.folder_dir = os.path.dirname(self.cell_dir)
        os.makedirs(self.cell_dir, exist_ok=True)
        self.path = os.path.join(self.cell_dir, "llm_usage.jsonl")
        self.requests_path = os.path.join(self.cell_dir, "llm_requests.jsonl")
        self.responses_path = os.path.join(self.cell_dir, "llm_responses.jsonl")
        self.cap_cell_usd = float(cap_cell_usd)
        self.cap_folder_usd = float(cap_folder_usd)
        self.mock = bool(mock)
        self.calls = 0

    def spent_cell(self):
        return _read_usd(self.path)

    def spent_folder(self):
        total = 0.0
        for name in os.listdir(self.folder_dir):
            p = os.path.join(self.folder_dir, name, "llm_usage.jsonl")
            if os.path.isfile(p):
                total += _read_usd(p)
        return total

    def check(self, worst_usd):
        cell, folder = self.spent_cell(), self.spent_folder()
        if cell + worst_usd > self.cap_cell_usd:
            raise CostCapExceeded(f"cell cap {self.cap_cell_usd} USD: spent {cell:.4f} + worst next {worst_usd:.4f}")
        if folder + worst_usd > self.cap_folder_usd:
            raise CostCapExceeded(f"folder cap {self.cap_folder_usd} USD: spent {folder:.4f} + worst next {worst_usd:.4f}")
        return cell, folder

    def _append(self, path, row):
        with open(path, "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def record(self, row, request=None, reply_text=None):
        self._append(self.path, row)
        if request is not None:
            self._append(self.requests_path, {"call_index": row["call_index"], **request})
        if reply_text is not None:
            self._append(self.responses_path, {"call_index": row["call_index"], "text": reply_text})

    def totals(self):
        rows = []
        if os.path.isfile(self.path):
            with open(self.path) as fh:
                rows = [json.loads(l) for l in fh if l.strip()]
        keys = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        out = {k: int(sum(int(r.get(k) or 0) for r in rows)) for k in keys}
        out.update(calls=sum(1 for r in rows if not r.get("refused_by_cap")),
                   refused_by_cap=sum(1 for r in rows if r.get("refused_by_cap")),
                   errors=sum(1 for r in rows if r.get("error") and not r.get("refused_by_cap")),
                   usd=round(sum(float(r.get("usd") or 0.0) for r in rows), 6),
                   json_extracted=sum(1 for r in rows if r.get("json_extracted")),
                   max_tokens_stops=sum(1 for r in rows if r.get("stop_reason") == "max_tokens"),
                   refusals=sum(1 for r in rows if r.get("stop_reason") == "refusal"),
                   dropped_temperature=sum(1 for r in rows if "temperature" in (r.get("dropped") or [])),
                   dropped_response_format=sum(1 for r in rows if "response_format" in (r.get("dropped") or [])),
                   price_rule="(input + cache_creation + cache_read) x 4 + output x 20 USD per 1M tokens",
                   mock=self.mock)
        return out


def sdk_call_kwargs(create, body):
    """Split the body on the installed SDK's create() signature; unknown keys ride in extra_body.

    anthropic 0.71.0 (a compute host) has no `output_config` parameter, 0.120.2 (the development host) has; either way
    the JSON body on the wire is the same (checked by the MockTransport tests).
    """
    params = inspect.signature(create).parameters
    direct, extra = {}, {}
    for key, value in body.items():
        (direct if key in params else extra)[key] = value
    if extra:
        direct["extra_body"] = extra
    return direct


def _valid_json_object(text):
    try:
        return isinstance(json.loads(text), dict)
    except (ValueError, TypeError):
        return False


def extract_json_object(text):
    """json_object-mode emulation: the outermost {...} span if (and only if) it parses as an object."""
    s = text.strip()
    if _valid_json_object(s):
        return s, False
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        candidate = s[i:j + 1]
        if _valid_json_object(candidate):
            return candidate, True
    return text, False


class AnthropicChatShim:
    """OpenAI-shaped `.chat.completions.create(**kw)` over the Anthropic Messages API."""

    ALLOWED = {"model", "messages", "max_tokens", "temperature", "response_format"}

    def __init__(self, ledger, client=None, effort=EFFORT, max_tokens_floor=MAX_TOKENS_FLOOR,
                 sdk_max_retries=2, timeout=600.0):
        if client is None:
            import anthropic   # official SDK; key from ANTHROPIC_API_KEY in the environment
            client = anthropic.Anthropic(max_retries=sdk_max_retries, timeout=timeout)
        self._client = client
        self._ledger = ledger
        self._effort = effort
        self._floor = int(max_tokens_floor)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        unknown = set(kw) - self.ALLOWED
        if unknown:
            raise TypeError(f"unexpected arguments from the released client: {sorted(unknown)}")
        msgs = kw["messages"]
        system = "\n\n".join(m["content"] for m in msgs if m["role"] == "system")
        convo = [{"role": m["role"], "content": m["content"]} for m in msgs if m["role"] in ("user", "assistant")]
        if not convo or convo[0]["role"] != "user" or convo[-1]["role"] != "user":
            raise ValueError("conversation must start and end with a user turn")
        requested = kw.get("max_tokens")
        sent = max(int(requested or 0), self._floor)
        body = {"model": kw["model"], "max_tokens": sent, "system": system, "messages": convo,
                "thinking": {"type": "adaptive"}, "output_config": {"effort": self._effort}}
        dropped = [k for k in ("temperature", "response_format") if kw.get(k) is not None]
        json_mode = (kw.get("response_format") or {}).get("type") == "json_object"
        chars = len(system) + sum(len(m["content"]) for m in convo)
        worst = math.ceil(chars / CHARS_PER_TOKEN_ESTIMATE) * PRICE_INPUT_PER_M / 1e6 + sent * PRICE_OUTPUT_PER_M / 1e6
        self._ledger.calls += 1
        idx = self._ledger.calls
        try:
            spent_cell, spent_folder = self._ledger.check(worst)
        except CostCapExceeded as exc:
            # recorded, so a refusal inside one of the released loop's try/except call sites
            # (generator, analysis, final arbitration) stays visible in the receipt
            self._ledger.record({"t": time.time(), "call_index": idx, "model": kw["model"], "refused_by_cap": True,
                                 "error": str(exc), "usd": 0.0, "worst_usd_checked": round(worst, 6),
                                 "mock": self._ledger.mock})
            raise
        request_record = {"model": kw["model"], "system": system, "messages": convo}
        row = {"t": time.time(), "call_index": idx, "model": kw["model"], "max_tokens_requested": requested,
               "max_tokens_sent": sent, "effort": self._effort, "thinking": "adaptive", "dropped": dropped,
               "json_mode_requested": json_mode, "request_chars": chars, "worst_usd_checked": round(worst, 6),
               "spent_cell_before": round(spent_cell, 6), "spent_folder_before": round(spent_folder, 6),
               "mock": self._ledger.mock}
        t0 = time.time()
        try:
            resp = self._client.messages.create(**sdk_call_kwargs(self._client.messages.create, body))
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {str(exc)[:300]}", status_code=getattr(exc, "status_code", None),
                       usd=0.0, elapsed_s=round(time.time() - t0, 3))
            self._ledger.record(row, request=request_record)
            raise
        usage = {k: getattr(resp.usage, k, None) for k in
                 ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")
        extracted = False
        if json_mode:
            text, extracted = extract_json_object(text)
        row.update(usage, usd=round(usd_of(usage), 8), returned_model=getattr(resp, "model", None),
                   request_id=getattr(resp, "_request_id", None), stop_reason=getattr(resp, "stop_reason", None),
                   thinking_blocks=sum(1 for b in resp.content if getattr(b, "type", "") in ("thinking", "redacted_thinking")),
                   text_chars=len(text), json_extracted=extracted, elapsed_s=round(time.time() - t0, 3))
        self._ledger.record(row, request=request_record, reply_text=text)
        if row["stop_reason"] == "refusal":
            category = getattr(getattr(resp, "stop_details", None), "category", None)
            raise ClaudeRefusal(f"claude refusal stop (category={category})")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=row["stop_reason"])],
            usage=SimpleNamespace(prompt_tokens=usage["input_tokens"], completion_tokens=usage["output_tokens"]),
            model=row["returned_model"])


_STATE = {"ledger": None, "client_factory": None, "constructed": 0}


def install_claude_path(ledger, client_factory=None):
    """Route LLMClient(model="claude...") to the shim; other model names keep the released __init__."""
    from autoscilab.llm import client as LC
    _STATE["ledger"], _STATE["client_factory"] = ledger, client_factory
    current = LC.LLMClient.__init__
    if getattr(current, "_asl_claude_path", False):
        return
    released_init = current

    def __init__(self, model=LC.DEFAULT_MODEL, api_key=None, max_retries=3, base_url=None,
                 max_completion_tokens=8192, temperature=None):
        if not str(model).startswith("claude"):
            return released_init(self, model=model, api_key=api_key, max_retries=max_retries, base_url=base_url,
                                 max_completion_tokens=max_completion_tokens, temperature=temperature)
        # the released fields, set the way the released __init__ sets them
        self._model = model
        self._max_retries = max_retries
        self._max_completion_tokens = max_completion_tokens
        self._temperature = temperature
        self._is_thinking = LC._is_thinking_model(model)   # False for claude-* (released heuristic)
        client = _STATE["client_factory"]() if _STATE["client_factory"] else None
        self._oa_client = AnthropicChatShim(_STATE["ledger"], client=client)
        self._tg_client = None
        self._use_openai = True
        _STATE["constructed"] += 1

    __init__._asl_claude_path = True
    __init__._released_init = released_init
    LC.LLMClient.__init__ = __init__
