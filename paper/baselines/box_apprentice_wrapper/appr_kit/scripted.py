"""CHECK ONLY - scripted replies for --mode stub and --mode mock-transport (0 USD).

Written for this kit's plumbing check; nothing here comes from a recorded run (no responses, programs, designs or
critic texts of any campaign). Per task there is one fixed <observe> design and one small generalized linear model
in the released program template (prior-mode and posterior-mode variant). These are deliberately generic forms
(logistic / linear / log-linear in the inputs), not the task's generating law.

`check_request(body)` holds the request rules both test transports enforce on every call: the fields sent, the
top-level system string, no system role inside messages, alternating non-empty text turns that start and end with the
user, no temperature, max_tokens >= 16000, adaptive thinking and output_config {"effort": "medium"}.
"""
import json
from types import SimpleNamespace

OBSERVE = {"death": "1.0", "dugongs": "2.5", "peregrines": "2.5", "hyperbolic": "[100, 200, 30]", "survival": "5"}

_HEAD = '''import pymc as pm
import numpy as np


def gen_model(observed_data):
'''
_SEEDS = '''    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(314)
'''
_FIT = '''        trace = pm.sample(1000, tune=500, target_accept=0.90, chains=3, cores=1, random_seed=rng1, idata_kwargs={"log_likelihood": True})
        posterior_predictive = pm.sample_posterior_predictive(trace, random_seed=rng2, return_inferencedata=False)
        return model, posterior_predictive, trace
'''
_PRIOR = '''        prior_predictive = pm.sample_prior_predictive(samples=1000, random_seed=rng2, return_inferencedata=False)
        return model, prior_predictive
'''

# (feature columns, outcome column, body lines inside the model; {OBS} is replaced by y or None)
_MODELS = {
    "death": (["x"], "infection_num", '''        a = pm.Normal("a", mu=0.0, sigma=2.0)
        b = pm.Normal("b", mu=0.0, sigma=2.0)
        p = pm.math.invlogit(a + b * x_data)
        y_obs = pm.Binomial("y_obs", n=50, p=p, observed={OBS}, dims="obs_id")
'''),
    "dugongs": (["x"], "Y", '''        a = pm.Normal("a", mu=1.0, sigma=2.0)
        b = pm.Normal("b", mu=0.0, sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        y_obs = pm.Normal("y_obs", mu=a + b * x_data, sigma=sigma, observed={OBS}, dims="obs_id")
'''),
    "peregrines": (["year"], "C", '''        a = pm.Normal("a", mu=3.0, sigma=1.0)
        b = pm.Normal("b", mu=0.0, sigma=0.5)
        y_obs = pm.Poisson("y_obs", mu=pm.math.exp(a + b * year_data), observed={OBS}, dims="obs_id")
'''),
    "hyperbolic": (["ir", "dr", "days"], "Choice", '''        a = pm.Normal("a", mu=0.0, sigma=2.0)
        b = pm.Normal("b", mu=0.0, sigma=2.0)
        c = pm.Normal("c", mu=0.0, sigma=2.0)
        p = pm.math.invlogit(a + b * (dr_data - ir_data) / 100.0 + c * days_data / 100.0)
        y_obs = pm.Bernoulli("y_obs", p=p, observed={OBS}, dims="obs_id")
'''),
    "survival": (["metastasized_status", "time_surgery"], "outcome", '''        a = pm.Normal("a", mu=0.0, sigma=2.0)
        b = pm.Normal("b", mu=0.0, sigma=2.0)
        c = pm.Normal("c", mu=0.0, sigma=2.0)
        p = pm.math.invlogit(a + b * time_surgery_data / 10.0 + c * metastasized_status_data)
        y_obs = pm.Bernoulli("y_obs", p=p, observed={OBS}, dims="obs_id")
'''),
}


def program(task, prior):
    features, outcome, body = _MODELS[task]
    lines = [_HEAD]
    for f in features:
        lines.append(f'    {f} = observed_data["{f}"].to_numpy()\n')
    if not prior:
        lines.append(f'    y = observed_data["{outcome}"].to_numpy()\n')
    lines.append(_SEEDS)
    lines.append("    with pm.Model() as model:\n")
    for f in features:
        lines.append(f'        {f}_data = pm.MutableData("{f}", {f}, dims="obs_id")\n')
    lines.append(body.replace("{OBS}", "None" if prior else "y"))
    lines.append(_PRIOR if prior else _FIT)
    return "".join(lines)


CRITIC_REPLY = ("``` Hypotheses\n* Scripted hypothesis for the kit check (no model criticism was performed).\n```\n\n"
                "```Synthesis\nScripted synthesis for the kit check.\n```")


def reply_text(task, system, messages):
    """The scripted answer text for one request, chosen by the released prompt it answers."""
    if "You are a brilliant statistician modeling a dataset" in system:
        prior = "Your goal is not to perform inference but just sample from the prior" in system
        return ("Sketch: a small generalized linear model written for the kit's plumbing check.\n"
                "```python\n" + program(task, prior) + "```\n")
    if "specializing in critiquing" in system:
        return CRITIC_REPLY
    return f"<thought>Scripted reply for the kit check.</thought>\n<observe>{OBSERVE[task]}</observe>"


SENT_FIELDS = {"model", "max_tokens", "system", "messages", "thinking", "output_config"}


def _text_of(content):
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def check_request(body, model="claude-opus-5-5"):
    """Violations of the request rules (an empty list when the request is as specified)."""
    v = []
    if set(body) != SENT_FIELDS:
        v.append(f"fields {sorted(body)} != {sorted(SENT_FIELDS)}")
    if "temperature" in body:
        v.append("temperature sent")
    if body.get("model") != model:
        v.append(f"model {body.get('model')!r}")
    if not isinstance(body.get("max_tokens"), int) or body["max_tokens"] < 16000:
        v.append(f"max_tokens {body.get('max_tokens')!r} < 16000")
    if body.get("thinking") != {"type": "adaptive"}:
        v.append(f"thinking {body.get('thinking')!r}")
    if body.get("output_config") != {"effort": "medium"}:
        v.append(f"output_config {body.get('output_config')!r}")
    if not isinstance(body.get("system"), str) or not body["system"].strip():
        v.append("system is not a non-empty top-level string")
    msgs = body.get("messages") or []
    if not msgs or msgs[0].get("role") != "user" or msgs[-1].get("role") != "user":
        v.append("messages must start and end with a user turn")
    for prev, m in zip(msgs, msgs[1:]):
        if prev.get("role") == m.get("role"):
            v.append("two consecutive turns with the same role")
            break
    for m in msgs:
        if m.get("role") not in ("user", "assistant"):
            v.append(f"role {m.get('role')!r} inside messages")
        if not _text_of(m.get("content")).strip():
            v.append(f"empty {m.get('role')} turn")
        if isinstance(m.get("content"), list) and any(b.get("type") != "text" for b in m["content"]):
            v.append("non-text content block")
    return v


def synthetic_usage(body, text):
    chars = len(body.get("system", "")) + sum(len(_text_of(m["content"])) for m in body.get("messages", []))
    return {"input_tokens": chars // 2, "output_tokens": 300 + len(text) // 2,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


def message_json(body, text, index, stop_reason="end_turn"):
    """A Messages API reply: a thinking block first, then the text block."""
    return {"id": f"msg_scripted_{index}", "type": "message", "role": "assistant", "model": body["model"],
            "stop_reason": stop_reason, "stop_sequence": None,
            "content": [{"type": "thinking", "thinking": "(scripted check)", "signature": "scripted-signature"},
                        {"type": "text", "text": text}],
            "usage": synthetic_usage(body, text)}


class ScriptedClient:
    """--mode stub: answers in-process with the SDK's response shape; no SDK, no HTTP, no network."""

    def __init__(self, task):
        self.task = task
        self.requests = []
        self.violations = []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, *, model, max_tokens, system, messages, thinking=None, temperature=None, extra_body=None):
        body = {"model": model, "max_tokens": max_tokens, "system": system, "messages": messages}
        if thinking is not None:
            body["thinking"] = thinking
        if temperature is not None:
            body["temperature"] = temperature
        body.update(extra_body or {})
        body = json.loads(json.dumps(body))   # what would go on the wire
        self.requests.append(body)
        bad = check_request(body)
        if bad:
            self.violations.append(bad)
            raise AssertionError(f"request violates the Claude path rules: {bad}")
        text = reply_text(self.task, body["system"], body["messages"])
        msg = message_json(body, text, len(self.requests))
        blocks = [SimpleNamespace(**b) for b in msg["content"]]
        return SimpleNamespace(content=blocks, usage=SimpleNamespace(**msg["usage"]), stop_reason=msg["stop_reason"],
                               model=msg["model"], _request_id=f"req_scripted_{len(self.requests)}")
