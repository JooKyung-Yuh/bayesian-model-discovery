"""The Claude path for both LLM roles of the released Box's Apprentice (the only change that can alter the agent).

Conventions are the AutoSciLab kit's (unified_run_20261005/autoscilab_kit/asl_kit/claude_client.py):
  official anthropic SDK, Messages API, model claude-opus-5-5
  thinking      = {"type": "adaptive"}       (Opus 5.5 cannot switch thinking off)
  output_config = {"effort": "medium"}       (rides in extra_body when the installed SDK has no such argument)
  max_tokens    = max(requested, 16000)      (thinking shares max_tokens; the released 512 / 4096 would truncate)
  temperature   dropped and recorded          (Opus 5.5 rejects sampling parameters)
  refusal       recorded, then raised (ClaudeRefusal); no fallback model
  answer        = the FIRST text block of the reply (thinking blocks come first)
Every call appends one row to <cell>/llm_usage.jsonl (tokens; usd at 4 USD per 1M input + cache_creation +
cache_read tokens and 20 USD per 1M output tokens) and the request / reply to llm_requests.jsonl /
llm_responses.jsonl. Before every call, the cell total and the folder total (<folder>/*/llm_usage.jsonl) plus the
worst case of the next call (request characters / 1.9 tokens at 4 USD per 1M + max_tokens of output at 20 USD per
1M) must stay within the caps; otherwise CostCapExceeded is raised, nothing is sent, and a `refused_by_cap` row
(0 USD) is written.

Two adapters put the released code on this path:
  ScientistClient   replaces LMExperimenter.llm (the released Claude branch calls
                    `self.llm.messages.create(model, system, messages, max_tokens, temperature).content[0].text`)
  AnthropicAsOpenAI replaces model_search.AsyncOpenAIGPT4V (the box loop's OpenAI chat-completions client): the
                    `system` role becomes the top-level system string, user / assistant text passes through byte for
                    byte, an image block is refused, `n` choices are `n` sequential calls.
"""
import inspect
import json
import math
import os
import time
from types import SimpleNamespace

from . import prompt_filter

PRICE_INPUT_PER_M = 4.0
PRICE_OUTPUT_PER_M = 20.0
MAX_TOKENS_FLOOR = 16000
EFFORT = "medium"
CHARS_PER_TOKEN_ESTIMATE = 1.9   # the AutoSciLab kit's rule (1.942 measured on 533 Opus 5.5 requests, rounded down)
DEFAULT_MODEL = "claude-opus-5-5"
PRICE_RULE = "(input + cache_creation + cache_read) x 4 + output x 20 USD per 1M tokens"
# The Messages API rejects a turn without text. The released prior-mode proposal sends an EMPTY user message
# (box_loop_prompts.get_stan_user_prompt_prior returns ""), which gpt-4o accepted; it is sent as EMPTY_USER_PLACEHOLDER.
EMPTY_ASSISTANT_PLACEHOLDER = "(no text in this reply)"
EMPTY_USER_PLACEHOLDER = "Please proceed."


class CostCapExceeded(RuntimeError):
    """Raised before a call that could cross a cap; no request is sent."""


class ClaudeRefusal(RuntimeError):
    """stop_reason == "refusal"; recorded, then raised (not retried, no fallback model)."""


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

    def record(self, row, request=None, reply=None):
        self._append(self.path, row)
        if request is not None:
            self._append(self.requests_path, {"call_index": row["call_index"], "role": row.get("role"), **request})
        if reply is not None:
            self._append(self.responses_path, {"call_index": row["call_index"], "role": row.get("role"), **reply})

    def rows(self):
        if not os.path.isfile(self.path):
            return []
        with open(self.path) as fh:
            return [json.loads(l) for l in fh if l.strip()]

    def totals(self):
        rows = self.rows()
        keys = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        out = {k: int(sum(int(r.get(k) or 0) for r in rows)) for k in keys}
        sent = [r for r in rows if not r.get("refused_by_cap")]
        out.update(calls=len(sent),
                   refused_by_cap=sum(1 for r in rows if r.get("refused_by_cap")),
                   errors=sum(1 for r in sent if r.get("error")),
                   usd=round(sum(float(r.get("usd") or 0.0) for r in rows), 6),
                   calls_by_role={role: sum(1 for r in sent if r.get("role") == role)
                                  for role in sorted({r.get("role") for r in sent if r.get("role")})},
                   max_tokens_stops=sum(1 for r in rows if r.get("stop_reason") == "max_tokens"),
                   refusals=sum(1 for r in rows if r.get("stop_reason") == "refusal"),
                   replies_without_text=sum(1 for r in sent if not r.get("error") and not r.get("text_blocks")),
                   replies_with_several_text_blocks=sum(1 for r in sent if (r.get("text_blocks") or 0) > 1),
                   dropped_temperature=sum(1 for r in rows if "temperature" in (r.get("dropped") or [])),
                   empty_user_turns_replaced=sum(int(r.get("empty_user_turns_replaced") or 0) for r in rows),
                   empty_assistant_turns_replaced=sum(int(r.get("empty_assistant_turns_replaced") or 0) for r in rows),
                   merged_consecutive_turns=sum(int(r.get("merged_consecutive_turns") or 0) for r in rows),
                   price_rule=PRICE_RULE, mock=self.mock)
        return out


def sdk_call_kwargs(create, body):
    """Split the body on the installed SDK's create() signature; unknown keys ride in extra_body.

    anthropic 0.71.0 has no `output_config` parameter; the JSON body on the wire is the same either way (checked by
    tests/test_claude_wire.py through httpx.MockTransport).
    """
    params = inspect.signature(create).parameters
    direct, extra = {}, {}
    for key, value in body.items():
        (direct if key in params else extra)[key] = value
    if extra:
        direct["extra_body"] = extra
    return direct


def _block_text(block):
    if isinstance(block, dict):
        if block.get("type") != "text":
            raise ValueError(f"only text content blocks are sent (use_vision is false); got {block.get('type')!r}")
        return block.get("text") or ""
    raise ValueError(f"unexpected content block {type(block).__name__}")


def content_text_len(content):
    if isinstance(content, str):
        return len(content)
    return sum(len(_block_text(b)) for b in content)


def _as_blocks(content):
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content)


def _normalise_turn(role, content):
    """(turn as sent, replaced-empty flag). Text passes through byte for byte; image blocks are refused; a turn
    whose text is empty or whitespace is sent as the placeholder of its role."""
    placeholder = EMPTY_ASSISTANT_PLACEHOLDER if role == "assistant" else EMPTY_USER_PLACEHOLDER
    if isinstance(content, str):
        if not content.strip():
            return {"role": role, "content": placeholder}, True
        return {"role": role, "content": content}, False
    blocks = [{"type": "text", "text": _block_text(b)} for b in content]
    if not "".join(b["text"] for b in blocks).strip():
        return {"role": role, "content": [{"type": "text", "text": placeholder}]}, True
    return {"role": role, "content": blocks}, False


class ClaudePath:
    """One metered Messages API call per `call()`; both roles of the released agent come through here."""

    def __init__(self, ledger, client, model=DEFAULT_MODEL, effort=EFFORT, max_tokens_floor=MAX_TOKENS_FLOOR,
                 prompt_variant="released"):
        self.ledger = ledger
        self.client = client
        self.model = model
        self.effort = effort
        self.floor = int(max_tokens_floor)
        if prompt_variant not in prompt_filter.VARIANTS:
            raise ValueError(f"prompt variant {prompt_variant!r}")
        self.prompt_variant = prompt_variant
        self.filter_totals = {rule: 0 for rule, _p, _r in prompt_filter.RULES}
        self.filter_calls = 0      # calls in which the filter changed at least one text
        self.assistant_thought_tags = 0   # <thought> in the model's OWN earlier replies (history), left as the released code leaves them

    def _filter_prompt(self, system, convo):
        """prompt_filter on the system string and every user turn (released: unchanged, zero counts)."""
        counts = {rule: 0 for rule, _p, _r in prompt_filter.RULES}
        if self.prompt_variant == "released":
            return system, convo, {"variant": "released", "changed": False, "counts": counts}
        before_sha = prompt_filter.sha256(system)
        system_out, c = prompt_filter.apply(system, self.prompt_variant)
        for k in counts:
            counts[k] += c[k]
        out = []
        for turn in convo:
            if turn["role"] != "user":
                out.append(turn)
                continue
            if isinstance(turn["content"], str):
                content, c = prompt_filter.apply(turn["content"], self.prompt_variant)
                for k in counts:
                    counts[k] += c[k]
            else:
                content = []
                for block in turn["content"]:
                    text, c = prompt_filter.apply(block["text"], self.prompt_variant)
                    content.append({**block, "text": text})
                    for k in counts:
                        counts[k] += c[k]
            out.append({"role": "user", "content": content})
        def tags(turns, role):
            return sum(prompt_filter.remaining_tags(t["content"]) if isinstance(t["content"], str)
                       else sum(prompt_filter.remaining_tags(b["text"]) for b in t["content"]) for t in turns if t["role"] == role)
        remaining = prompt_filter.remaining_tags(system_out) + tags(out, "user")
        assistant_tags = tags(out, "assistant")   # the model's own earlier replies pass through byte for byte (released history)
        self.assistant_thought_tags += assistant_tags
        if remaining:
            raise ValueError(f"prompt variant {self.prompt_variant}: {remaining} <thought> tag(s) still in the request after "
                             "the filter; not sent")
        changed = any(counts.values())
        for k in counts:
            self.filter_totals[k] += counts[k]
        self.filter_calls += int(changed)
        return system_out, out, {"variant": self.prompt_variant, "changed": changed, "counts": counts,
                                 "thought_tags_in_assistant_turns": assistant_tags,
                                 "system_sha256_before": before_sha, "system_sha256_after": prompt_filter.sha256(system_out)}

    def call(self, role, system, messages, requested_max_tokens, dropped=(), model=None):
        model = model or self.model
        convo, replaced, merged = [], {"user": 0, "assistant": 0}, 0
        for m in messages:
            if m.get("role") not in ("user", "assistant"):
                raise ValueError(f"role {m.get('role')!r} inside messages (the system text is the top-level `system`)")
            turn, rep = _normalise_turn(m["role"], m["content"])
            replaced[m["role"]] += int(rep)
            if convo and convo[-1]["role"] == turn["role"]:
                # consecutive same-role messages (the released critic sends one user message per program): one turn
                # with one text block per released message, which is how the Messages API combines such turns itself
                convo[-1] = {"role": turn["role"], "content": _as_blocks(convo[-1]["content"]) + _as_blocks(turn["content"])}
                merged += 1
            else:
                convo.append(turn)
        if not convo or convo[0]["role"] != "user" or convo[-1]["role"] != "user":
            raise ValueError("conversation must start and end with a user turn")
        if not isinstance(system, str):
            raise ValueError("system must be one string")
        system, convo, filtered = self._filter_prompt(system, convo)
        sent = max(int(requested_max_tokens or 0), self.floor)
        body = {"model": model, "max_tokens": sent, "system": system, "messages": convo,
                "thinking": {"type": "adaptive"}, "output_config": {"effort": self.effort}}
        chars = len(system) + sum(content_text_len(m["content"]) for m in convo)
        worst = math.ceil(chars / CHARS_PER_TOKEN_ESTIMATE) * PRICE_INPUT_PER_M / 1e6 + sent * PRICE_OUTPUT_PER_M / 1e6
        self.ledger.calls += 1
        idx = self.ledger.calls
        base = {"t": time.time(), "call_index": idx, "role": role, "model": model, "mock": self.ledger.mock}
        try:
            spent_cell, spent_folder = self.ledger.check(worst)
        except CostCapExceeded as exc:
            self.ledger.record({**base, "refused_by_cap": True, "error": str(exc), "usd": 0.0,
                                "worst_usd_checked": round(worst, 6)})
            raise
        request_record = {"model": model, "system": system, "messages": convo, "max_tokens": sent}
        row = {**base, "max_tokens_requested": requested_max_tokens, "max_tokens_sent": sent, "effort": self.effort,
               "thinking": "adaptive", "dropped": sorted(set(dropped)), "request_chars": chars,
               "worst_usd_checked": round(worst, 6), "spent_cell_before": round(spent_cell, 6),
               "spent_folder_before": round(spent_folder, 6), "empty_user_turns_replaced": replaced["user"],
               "empty_assistant_turns_replaced": replaced["assistant"], "merged_consecutive_turns": merged,
               "prompt_variant": self.prompt_variant, "prompt_filter": filtered}
        t0 = time.time()
        try:
            resp = self.client.messages.create(**sdk_call_kwargs(self.client.messages.create, body))
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {str(exc)[:300]}", status_code=getattr(exc, "status_code", None),
                       usd=0.0, elapsed_s=round(time.time() - t0, 3))
            self.ledger.record(row, request=request_record)
            raise
        usage = {k: getattr(resp.usage, k, None) for k in
                 ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
        blocks = list(resp.content or [])
        texts = [getattr(b, "text", "") or "" for b in blocks if getattr(b, "type", None) == "text"]
        answer = texts[0] if texts else ""
        row.update(usage, usd=round(usd_of(usage), 8), returned_model=getattr(resp, "model", None),
                   request_id=getattr(resp, "_request_id", None), stop_reason=getattr(resp, "stop_reason", None),
                   block_types=[getattr(b, "type", None) for b in blocks], text_blocks=len(texts),
                   thinking_blocks=sum(1 for b in blocks if getattr(b, "type", "") in ("thinking", "redacted_thinking")),
                   answer_chars=len(answer), elapsed_s=round(time.time() - t0, 3))
        self.ledger.record(row, request=request_record,
                           reply={"text": answer, "other_text_blocks": texts[1:], "stop_reason": row["stop_reason"]})
        if row["stop_reason"] == "refusal":
            category = getattr(getattr(resp, "stop_details", None), "category", None)
            raise ClaudeRefusal(f"claude refusal stop (category={category})")
        return answer, usage, row


class ScientistClient:
    """Stands in for LMExperimenter.llm (anthropic.Anthropic in the released Claude branch, agent.py:16 and :61)."""

    def __init__(self, path):
        self._path = path
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, model, system, messages, max_tokens, temperature=None):
        dropped = ["temperature"] if temperature is not None else []
        answer, usage, _row = self._path.call("scientist", system, messages, max_tokens, dropped=dropped, model=model)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=answer)],
                               usage=SimpleNamespace(**{k: usage.get(k) for k in usage}))


_STATE = {"path": None, "constructed": 0}


def bind_box_loop(path):
    _STATE["path"] = path


def _role_of(system):
    # the released box-loop prompts open with these sentences (box_loop_prompts.py:22, :123, :164, :289)
    return "box_loop_critic" if "specializing in critiquing" in system else "box_loop_proposal"


class AnthropicAsOpenAI:
    """Drop-in for model_search.AsyncOpenAIGPT4V (openai_wrapper.py:8-42): same constructor and async call."""

    def __init__(self, max_tokens=512):
        if _STATE["path"] is None:
            raise RuntimeError("AnthropicAsOpenAI used before bind_box_loop()")
        self.max_tokens = max_tokens
        self._path = _STATE["path"]
        _STATE["constructed"] += 1

    @property
    def llm_type(self):
        return "claude-opus-5-5 via AnthropicAsOpenAI"

    async def __call__(self, messages, **kwargs):
        unknown = set(kwargs) - {"temperature", "n"}
        if unknown:
            raise TypeError(f"unexpected arguments from the released box loop: {sorted(unknown)}")
        n = int(kwargs.get("n") or 1)
        dropped = ["temperature"] if kwargs.get("temperature") is not None else []
        system_parts, convo = [], []
        for m in messages:
            if m["role"] == "system":
                c = m["content"]
                system_parts.append(c if isinstance(c, str) else "".join(_block_text(b) for b in c))
            else:
                convo.append({"role": m["role"], "content": m["content"]})
        system = "\n\n".join(system_parts)
        role = _role_of(system)
        choices, tin, tout = [], 0, 0
        for _ in range(n):
            answer, usage, row = self._path.call(role, system, convo, self.max_tokens, dropped=dropped)
            choices.append(SimpleNamespace(message=SimpleNamespace(content=answer), finish_reason=row.get("stop_reason")))
            tin += sum(int(usage.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            tout += int(usage.get("output_tokens") or 0)
        # `.model` is the requested id: BaseAgent.calc_cost looks it up in the cost table (the returned id is in the ledger)
        return SimpleNamespace(model=self._path.model, choices=choices,
                               usage=SimpleNamespace(prompt_tokens=tin, completion_tokens=tout))
