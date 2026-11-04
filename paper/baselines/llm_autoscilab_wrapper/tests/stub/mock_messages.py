"""STUB ONLY - an httpx.MockTransport handler that answers POST /v1/messages like the Messages API.

Used by the wire tests and by `runner.py run --stub`. The official anthropic SDK builds and sends
the real request; this handler sees the bytes on the wire, records them, and returns a schema-valid
synthetic reply (no data from any real run: the expressions are generic forms over the parameter
names the request itself declares). Usage numbers in the reply are synthetic and flagged mock.
"""
import json
import re

import httpx

FIELDS_RE = re.compile(r"single valid JSON object with these fields: (.*?)\. ")


def _params_from_prompt(text):
    m = re.search(r"^PARAMETERS: (.*)$", text, re.M)
    if m:
        return [p.strip() for p in m.group(1).split(",") if p.strip()]
    m = re.search(r"def discovered_law\(([^)]*)\)", text)
    return [p.strip() for p in m.group(1).split(",")] if m else ["x"]


def _bounds_from_prompt(text, names):
    out = {}
    for n in names:
        m = re.search(rf"- {re.escape(n)}\b[^\n]*?\[([-+0-9.eE]+), ([-+0-9.eE]+)\]", text)
        if m:
            out[n] = [float(m.group(1)), float(m.group(2))]
    return out


def synthetic_reply(system, user, call_index):
    if "Respond with ONLY a single integer" in system:
        return "0"
    m = FIELDS_RE.search(system)
    fields = [f.strip().strip('"') for f in m.group(1).split(",")] if m else ["reasoning"]
    names = _params_from_prompt(user)
    bounds = _bounds_from_prompt(user, names)
    if len(bounds) < len(names):   # ChemBench prints ranges in one line; fall back to the full box
        rng = re.search(r"Inputs: (.*)$", user, re.M)
        for n in names:
            bounds.setdefault(n, [0.5, 1.5])
            if rng:
                mm = re.search(rf"{re.escape(n)} \[([0-9.]+)[--]([0-9.]+)", rng.group(1))
                if mm:
                    bounds[n] = [float(mm.group(1)), float(mm.group(2))]
    prod = " * ".join(f"{n}**C{i + 1}" for i, n in enumerate(names))
    obj = {}
    for f in fields:
        if f == "reasoning":
            obj[f] = f"synthetic stub reasoning #{call_index}"
        elif f == "hypothesis":
            obj[f] = f"C0 * {prod}"
        elif f == "alternate_hypotheses":
            obj[f] = [f"C0 + C1 * {names[0]}", f"C0 * exp(C1 * {names[0]})"]
        elif f == "search_regions":
            obj[f] = [{"bounds": {n: bounds[n] for n in names}, "n_experiments": 4, "priority": "high",
                       "rationale": "synthetic full-box region"}]
        elif f == "phase":
            obj[f] = "discovery"
        elif f == "confidence":
            obj[f] = 0.3
        elif f == "done":
            obj[f] = False
        else:
            obj[f] = "stub"
    return json.dumps(obj)


class MockMessages:
    """Callable httpx handler; keeps every request body it saw."""

    def __init__(self, reply_fn=synthetic_reply, stop_reason="end_turn", status=200, prefix_text=""):
        self.reply_fn = reply_fn
        self.stop_reason = stop_reason
        self.status = status
        self.prefix_text = prefix_text
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        self.requests.append({"url": str(request.url), "method": request.method,
                              "headers": {k.lower(): v for k, v in request.headers.items()}, "body": body})
        if self.status != 200:
            return httpx.Response(self.status, json={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
        user = body["messages"][-1]["content"]
        text = self.prefix_text + self.reply_fn(body.get("system", ""), user, len(self.requests))
        chars = len(body.get("system", "")) + sum(len(m["content"]) for m in body["messages"])
        msg = {"id": f"msg_stub_{len(self.requests)}", "type": "message", "role": "assistant",
               "model": body["model"], "stop_reason": self.stop_reason, "stop_sequence": None,
               "content": [{"type": "thinking", "thinking": "", "signature": "stub-signature"},
                           {"type": "text", "text": text}],
               "usage": {"input_tokens": chars // 2, "output_tokens": 300,
                         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
        return httpx.Response(200, json=msg, headers={"request-id": f"req_stub_{len(self.requests)}"})


def mock_client(handler):
    import anthropic
    return anthropic.Anthropic(api_key="stub-not-a-key", max_retries=0,
                               http_client=httpx.Client(transport=httpx.MockTransport(handler)))
