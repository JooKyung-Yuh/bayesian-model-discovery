"""CHECK ONLY - --mode mock-transport: the installed anthropic SDK over an httpx.MockTransport (0 USD, no network).

The SDK builds and sends the real request; `MessagesAPIMock` sees the bytes on the wire, records them, enforces
scripted.check_request (fields, system placement, no temperature, max_tokens >= 16000, adaptive thinking,
output_config) plus the HTTP method, path and headers, and answers like the Messages API: a thinking block first,
then the scripted text block, with synthetic usage (flagged mock in the ledger).
"""
import json

import httpx

from . import scripted


class MessagesAPIMock:
    def __init__(self, task, stop_reason="end_turn", status=200, text_fn=None, extra_text_block=None):
        self.task = task
        self.stop_reason = stop_reason
        self.status = status
        self.text_fn = text_fn
        self.extra_text_block = extra_text_block
        self.requests = []
        self.violations = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        headers = {k.lower(): v for k, v in request.headers.items()}
        self.requests.append({"url": str(request.url), "method": request.method, "headers": headers, "body": body})
        bad = scripted.check_request(body)
        if request.method != "POST" or not str(request.url).endswith("/v1/messages"):
            bad.append(f"{request.method} {request.url}")
        if not headers.get("x-api-key") or not headers.get("anthropic-version"):
            bad.append("x-api-key / anthropic-version header missing")
        if bad:
            self.violations.append(bad)
            return httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request_error",
                                                                         "message": "mock transport: " + "; ".join(bad)}})
        if self.status != 200:
            return httpx.Response(self.status, json={"type": "error", "error": {"type": "overloaded_error",
                                                                                "message": "Overloaded"}})
        text = (self.text_fn or (lambda s, m: scripted.reply_text(self.task, s, m)))(body["system"], body["messages"])
        msg = scripted.message_json(body, text, len(self.requests), stop_reason=self.stop_reason)
        if self.extra_text_block is not None:
            msg["content"].append({"type": "text", "text": self.extra_text_block})
        return httpx.Response(200, json=msg, headers={"request-id": f"req_mock_{len(self.requests)}"})


def mock_client(handler):
    import anthropic
    return anthropic.Anthropic(api_key="mock-not-a-key", max_retries=0,
                               http_client=httpx.Client(transport=httpx.MockTransport(handler)))
