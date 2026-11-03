"""Metered two-attempt transport. Tests inject clients; live use requires a seal."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from bayes_model_discovery.experimental.count_proposer import (
    SecretMaskingFilter,
    _anthropic_usage,
    anthropic_thinking_record,
    mask_secrets_deep,
    masked_exception,
    sdk_call_kwargs,
)
from tools.icl_baseline.protocol import (
    DOMAIN_WORDS,
    ROOT,
    json_bytes,
    prediction_result,
    repair_request,
    selection_fallback,
    validate,
)

MODEL = "claude-opus-5-5"
CAPS = {"ICL-MASKED-FIXED-HISTORY": 90.0, "ICL-MASKED-CLOSED-LOOP": 90.0,
        "ICL-NAME-VISIBLE-EXPLORATORY": 25.0}
RESERVATION_RULE = "utf8_request_bytes_plus_1024_input_tokens_and_max_output_tokens"
# Unit ICL-K: the ten-call projection multiplies each request kind's remaining count by that
# kind's own c_hi (a selection costs about 1/40 of a prediction), instead of one c_hi for all.
PROJECTION_RULE = "per_kind_max_cost_x1.25_after_10_calls"
KINDS = ("prediction", "selection")


def utc():
    return datetime.now(UTC).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json_bytes(value))


def request_body(prompt, max_tokens=32000):
    if max_tokens not in (16000, 32000):
        raise ValueError("max_tokens must match a registered transport path")
    return {"model": MODEL, "max_tokens": max_tokens, **prompt,
            "output_config": {"effort": "medium"},
            "thinking": {"type": "adaptive", "display": "summarized"}}


def settings(unit, max_tokens=32000):
    return {"unit_id": unit, "model": MODEL, "max_tokens": max_tokens,
            "stream": max_tokens == 32000, "timeout_seconds": 300, "max_retries": 0,
            "output_config": {"effort": "medium"},
            "thinking": {"type": "adaptive", "display": "summarized"},
            "cap_usd": CAPS[unit], "reservation_rule": RESERVATION_RULE,
            "projection_rule": PROJECTION_RULE,
            "selection_fallback_stream": "icl_selection_fallback",
            "mock_selection_stream": "icl_synthetic_selection",
            "max_attempts_per_request": 2}


def input_token_reservation(body):
    # A deliberately conservative local bound, not a provider token measurement.
    # No remote token-counting endpoint is called by dry runs or mock runs.
    return len(json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")) + 1024


class BudgetStopped(RuntimeError):
    pass


class RequestSkipped(RuntimeError):
    """A projection reduction removed this request before its first attempt."""


@dataclass
class Budget:
    cap: float
    prices: dict
    calls: list = field(default_factory=list)
    projection_done: bool = False
    stopped_reason: str | None = None
    reduce_plan: object = None
    projection: dict | None = None

    @property
    def spent(self):
        return math.fsum(row["cost_usd"] for row in self.calls)

    def cost(self, input_tokens, output_tokens):
        return (input_tokens * self.prices["input_usd_per_1m"]
                + output_tokens * self.prices["output_usd_per_1m"]) / 1_000_000

    def c_hi_by_kind(self, reservation, kind):
        """Each kind's registered upper cost per call. After ten calls: that kind's largest
        usage-based cost times 1.25. A kind without one keeps the reservation rule: the current
        request's reservation for the request's own kind, the largest reservation recorded for
        the other kind, or the current reservation when that kind has no recorded call."""
        result = {}
        for name in KINDS:
            rows = [row for row in self.calls if row.get("kind") == name]
            known = [row["cost_usd"] for row in rows if row["cost_basis"] == "usage"]
            if len(self.calls) >= 10 and known:
                result[name] = max(known) * 1.25
            elif name == kind or not rows:
                result[name] = reservation
            else:
                result[name] = max(row["reservation_usd"] for row in rows)
        return result

    def projected(self, remaining, c_hi):
        return self.spent + math.fsum(remaining.get(name, 0) * c_hi[name] for name in KINDS)

    def reserve(self, body, remaining_initial_requests, kind):
        """`remaining_initial_requests` is {kind: count} (an int means the request's own kind)."""
        if kind not in KINDS:
            raise ValueError(f"request kind must be one of {KINDS}")
        remaining = ({kind: remaining_initial_requests} if isinstance(remaining_initial_requests, int)
                     else dict(remaining_initial_requests))
        reservation = self.cost(input_token_reservation(body), body["max_tokens"])
        c_hi = self.c_hi_by_kind(reservation, kind)
        reason = None
        if self.spent + reservation > self.cap:
            reason = "next_call_maximum_reservation_exceeds_cap"
        elif self.spent + c_hi[kind] > self.cap:
            reason = "registered_c_hi_exceeds_cap"
        elif len(self.calls) >= 10 and not self.projection_done:
            self.projection_done = True
            projection = self.projected(remaining, c_hi)
            self.projection = {"rule": PROJECTION_RULE, "calls_completed": len(self.calls),
                               "spent_usd": self.spent, "remaining_by_kind": remaining,
                               "c_hi_by_kind": c_hi, "projected_usd": projection, "cap_usd": self.cap,
                               "exceeds_cap": projection > self.cap}
            if projection > self.cap:
                if self.reduce_plan is not None:
                    remaining = dict(self.reduce_plan())
                    projection = self.projected(remaining, c_hi)
                    self.projection.update(remaining_by_kind_after_reduction=remaining,
                                           projected_usd_after_reduction=projection,
                                           exceeds_cap_after_reduction=projection > self.cap)
                if projection > self.cap:
                    reason = "registered_ten_call_projection_exceeds_cap"
        if reason:
            self.stopped_reason = reason
            raise BudgetStopped(reason)
        return {"reservation_usd": reservation, "c_hi_kind_usd": c_hi[kind],
                "input_tokens_reserved": input_token_reservation(body)}

    def record(self, usage, reservation, **identity):
        usage = usage or {}
        known = all(type(usage.get(key)) is int and usage[key] >= 0 for key in ("input_tokens", "output_tokens"))
        input_tokens = usage.get("input_tokens") if known else None
        output_tokens = usage.get("output_tokens") if known else None
        # No cache controls are sent. Unexpected cached usage is fail-conservative.
        cache = any(usage.get(key, 0) for key in ("cache_creation_input_tokens", "cache_read_input_tokens"))
        known = known and not cache
        cost = self.cost(input_tokens, output_tokens) if known else reservation["reservation_usd"]
        details = usage.get("output_tokens_details") or {}
        row = {**identity, **reservation, "input_tokens": input_tokens, "output_tokens": output_tokens,
               "thinking_tokens": details.get("thinking_tokens"), "cost_usd": cost,
               "cost_basis": "usage" if known else "maximum_reservation_unknown_usage",
               "cumulative_usd": self.spent + cost}
        self.calls.append(row)
        if cost > reservation["reservation_usd"]:
            self.stopped_reason = "provider_usage_exceeded_conservative_reservation"
        return row


def load_prices():
    data = json.loads((ROOT / "prices.json").read_text())
    return data[MODEL]


def verify_seal(path, expected_settings, required_inputs):
    seal = json.loads(Path(path).read_text())
    if seal.get("status") != "sealed" or seal.get("settings") != expected_settings:
        raise ValueError("a sealed manifest with exactly matching settings is required")
    files = seal.get("files", {})
    required = {str(p.resolve()) for p in ROOT.rglob("*") if p.is_file() and
                (p.suffix in (".py", ".txt", ".json")) and "tests" not in p.parts}
    required.update(str(Path(p).resolve()) for p in required_inputs)
    if not required.issubset(files):
        raise ValueError("seal is missing code, prompt, price, or input files")
    for filename, expected in files.items():
        if sha(filename) != expected:
            raise ValueError(f"seal hash mismatch: {filename}")
    return seal


def live_client(allow_paid_api, seal_manifest, expected_settings, required_inputs, factory=None):
    if not allow_paid_api or seal_manifest is None:
        raise ValueError("live client locked: --allow-paid-api AND --seal-manifest are required")
    verify_seal(seal_manifest, expected_settings, required_inputs)
    if factory is None:
        from anthropic import Anthropic
        factory = Anthropic
    for name in ("httpx", "httpcore", "anthropic._base_client"):
        logging.getLogger(name).addFilter(SecretMaskingFilter())
    return factory(max_retries=0, timeout=300.0)


class Transport:
    def __init__(self, client, output, budget, *, max_tokens=32000, synthetic=False):
        self.client, self.output, self.budget = client, Path(output), budget
        self.max_tokens, self.synthetic = max_tokens, synthetic
        self.counts = {kind: {"summaries": 0, "with_domain_word": 0} for kind in ("prediction", "selection")}

    def call(self, prompt, request_id, attempt, kind, remaining, skip_unstarted=None):
        body = request_body(prompt, self.max_tokens)
        if self.budget.stopped_reason:
            raise BudgetStopped(self.budget.stopped_reason)
        reserved = self.budget.reserve(body, remaining, kind)
        # A reduction can happen inside reserve. Never send a removed initial call,
        # but preserve retries of requests that have already consumed an attempt.
        if attempt == 0 and skip_unstarted is not None and skip_unstarted():
            raise RequestSkipped(request_id)
        stem = self.output / "transport" / f"{request_id}.attempt{attempt}"
        write_new(str(stem) + ".request.json", {"body": body, "request_id": request_id,
                                               "attempt": attempt, "synthetic": self.synthetic})
        response = None
        error = None
        try:
            if self.max_tokens == 32000:
                with self.client.messages.stream(**sdk_call_kwargs(self.client.messages.stream, body)) as stream:
                    response = stream.get_final_message()
            else:
                response = self.client.messages.create(**sdk_call_kwargs(self.client.messages.create, body))
        except Exception as exc:  # noqa: BLE001 - provider/SDK failures consume one attempt
            error = masked_exception(exc)
        usage, _ = _anthropic_usage(getattr(response, "usage", None))
        usage = usage or {}
        text = "".join(getattr(block, "text", "") for block in getattr(response, "content", [])
                       if getattr(block, "type", None) == "text")
        thinking = anthropic_thinking_record(response) if response is not None else {}
        for summary in thinking.get("thinking_text", []):
            self.counts[kind]["summaries"] += 1
            self.counts[kind]["with_domain_word"] += int(DOMAIN_WORDS.search(summary) is not None)
        stop = getattr(response, "stop_reason", None)
        row = self.budget.record(usage, reserved, request_id=request_id, attempt=attempt,
                                 status="transport_failure" if error else "response_received", kind=kind)
        record = mask_secrets_deep({"request_id": request_id, "attempt": attempt, "text": text,
                                   "stop_reason": stop, "usage": usage, **thinking,
                                   "error": str(error) if error else None, "accounting": row,
                                   "synthetic": self.synthetic})
        write_new(str(stem) + ".response.json", record)
        return text, stop, error, stem

    def request(self, prompt, request_id, kind, observations, *, mapping=(), replicate=0,
                round_index=0, unit_id="", remaining=0, skip_unstarted=None):
        query_ids = tuple(row["id"] for row in mapping)
        validations = []
        request = prompt
        attempts = []
        for attempt in range(2):
            text, stop, error, stem = self.call(request, request_id, attempt, kind, remaining, skip_unstarted)
            if error:
                attempts.append("transport_failure")
                # Transport retries resend the original independent conversation.
                request = prompt
                continue
            result = validate(text, kind, query_ids=query_ids, stop_reason=stop)
            validations.append(result)
            status = "valid" if result.valid else "retryable_validation_failure"
            self.budget.calls[-1]["status"] = status
            attempts.append(status)
            write_new(str(stem) + ".validation.json", {"status": status, "errors": result.errors})
            if result.valid:
                break
            request = (repair_request(prompt, text, "; ".join(result.errors), kind, len(query_ids))
                       if text.strip() else prompt)
        if kind == "prediction":
            return {**prediction_result(validations, query_ids, observations), "attempt_statuses": attempts}
        for result in validations:
            if result.valid:
                return {"x": list(result.values.values()), "status": "valid", "attempt_statuses": attempts}
        return {"x": selection_fallback(unit_id, replicate, round_index),
                "status": "selection_fallback", "attempt_statuses": attempts}
