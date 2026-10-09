"""Synthetic clients and one loopback TLS server for the compatible runners; no external calls or fits."""
import argparse
import csv
import importlib
import json
import shutil
import ssl
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace as NS

import pytest

from bayes_model_discovery.experimental import count_proposer as count
from bayes_model_discovery.experimental import neuron_proposer as neuron

URL = "https://compat.example.invalid/api/v1"
MODEL = "synthetic/vendor-edition"
KEY_NAME = "SYNTHETIC_COMPAT_CREDENTIAL"
CONTEXT = {"run_id": "synthetic-only", "world_seed": 11, "cell": "100", "proposal_rep": 0, "round": 1}


def response(text=None):
    """Build a synthetic chat-completion reply object, optionally with the given text."""
    return NS(model="different-returned-model", id="synthetic-completion-id", _request_id="synthetic-request-id",
              choices=[NS(message=NS(content=text or json.dumps({"expression": "b0", "family": "binomial", "parameters": {"b0": "intercept"}})), finish_reason="stop")],
              usage=NS(prompt_tokens=2, completion_tokens=3))


def fake_client(replies):
    """Return a fake chat client replaying the given replies, plus the list of recorded request bodies."""
    calls = []
    def create(**body):
        """Record the request body and return or raise the next scripted reply."""
        calls.append(body)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
    return NS(chat=NS(completions=NS(create=create))), calls


@pytest.mark.parametrize("url", [None, "", "http://localhost:9/v1", "https:///v1", "https://u:p@host/v1",
    "https://host/v1?token=x", "https://host/v1#secret", "https://host:bad/v1", "https://host/a b", "https://host/\x00bad", "https://host/\x7fbad"])
def test_compat_url_refused_without_records(tmp_path, url):
    """Malformed or unsafe base URLs are refused before any record directory is created."""
    with pytest.raises(ValueError):
        count.OpenAICompatCountProposer(tmp_path / "out", base_url=url, model=MODEL, client=object())
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("model", [None, "", "NAME", "<MODEL>", "TODO", " model ", "line\nbreak"])
def test_compat_requires_actual_model(tmp_path, model):
    """A placeholder or malformed model name is refused by the compatible proposer."""
    with pytest.raises(ValueError, match="actual llm-model"):
        count.OpenAICompatCountProposer(tmp_path / "out", base_url=URL, model=model, client=object())


def test_explicit_sdk_options_and_custom_secret_masking(tmp_path, monkeypatch):
    """The SDK client gets explicit options only, and the custom credential never reaches records."""
    import openai
    secret = "synthetic-compat-credential-not-an-api-key"
    monkeypatch.setenv(KEY_NAME, secret)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.invalid/v1")
    captured = []
    client, _ = fake_client([response()])
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: captured.append(kwargs) or client)
    proposer = count.OpenAICompatCountProposer(tmp_path / "count", base_url=URL, key_name=KEY_NAME,
                                              model=MODEL, allow_live=True)
    assert captured == [{"base_url": URL, "api_key": secret, "max_retries": 0, "timeout": 120.0}]
    models, record = proposer.propose_one("synthetic public packet", context=CONTEXT, initial_family="binomial")
    assert len(models) == 1 and record["status"] == "parsed"
    assert (record["requested_model"], record["returned_model"], record["completion_id"], record["http_request_id"]) == (
        MODEL, "different-returned-model", "synthetic-completion-id", "synthetic-request-id")
    assert record["provider"] == "openai_compat"
    assert record["llm_base_url_host"] == "compat.example.invalid" and record["llm_key_name"] == KEY_NAME
    assert secret not in json.dumps(record) and URL not in json.dumps(record)
    for path in (tmp_path / "count").iterdir():
        assert secret not in path.read_text() and URL not in path.read_text()
    assert secret not in str(count.masked_exception(RuntimeError(secret)))
    monkeypatch.setenv(KEY_NAME, "tiny")
    assert count.mask_secrets("contains tiny") == f"contains <{KEY_NAME}:masked>"


def test_compat_repair_and_transport_error_accounting(tmp_path, monkeypatch):
    """Repair calls and transport errors are counted without leaking the credential."""
    secret = "synthetic-custom-credential"
    monkeypatch.setenv(KEY_NAME, secret)
    client, calls = fake_client([response("invalid JSON "+secret), response()])
    proposer = count.OpenAICompatCountProposer(tmp_path / "repair", base_url=URL, key_name=KEY_NAME, model=MODEL, client=client)
    models, record = proposer.propose_one("synthetic packet", context=CONTEXT, initial_family="binomial")
    assert len(models) == 1 and record["calls"] == 2 and record["repairs"] == 1
    assert secret not in json.dumps(calls) and secret not in json.dumps(record)
    assert record["usage_total"]["input_tokens"] == 4
    broken, attempts = fake_client([RuntimeError("synthetic failure "+secret)])
    p = count.OpenAICompatCountProposer(tmp_path / "failed", base_url=URL, key_name=KEY_NAME, model=MODEL, client=broken)
    models, record = p.propose_one("synthetic packet", context=CONTEXT, initial_family="binomial")
    assert not models and len(attempts) == 1 and record["repairs"] == 0
    assert secret not in json.dumps(record)


@pytest.mark.parametrize("name", ["core_count", "boxinggym_core", "chembench_core", "neuron_core"])
def test_runner_cli_forwards_explicit_settings_without_default_keys(tmp_path, monkeypatch, name):
    """Runner command lines forward compatible-provider settings only when they are typed."""
    module = importlib.import_module("bayes_model_discovery.experimental."+name)
    received = []
    monkeypatch.setattr(module, "run", lambda args: received.append(args))
    extras = {"core_count": [], "boxinggym_core": ["--task", "death"], "chembench_core": ["--manifest", str(tmp_path / "synthetic-manifest.json"), "--noise", "0.1"], "neuron_core": []}[name]
    argv = [name, "--output", str(tmp_path / name), *extras]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert "llm_base_url" not in vars(received[-1]) and "llm_key_name" not in vars(received[-1])
    assert count.compat_config(received[-1]) == vars(received[-1])
    monkeypatch.setattr(sys, "argv", [*argv, "--proposer", "openai_compat", "--llm-base-url", URL,
                                    "--llm-key-name", KEY_NAME, "--llm-model", MODEL, "--allow-paid-api"])
    module.main()
    args = received[-1]
    assert args.llm_base_url == URL and args.llm_key_name == KEY_NAME and args.llm_model == MODEL
    recorded = count.compat_config(args)
    assert recorded["llm_provider"] == "openai_compat"
    assert recorded["llm_base_url_host"] == "compat.example.invalid" and "llm_base_url" not in recorded
    from bayes_model_discovery.experimental.core_count import points_label, resolve_run_label
    assert points_label("paper", "openai_compat") == "development"
    with pytest.raises(ValueError):
        resolve_run_label("production", "paper", "openai_compat")


def test_count_runner_constructor_forwards_and_masks_before_any_fit(tmp_path, monkeypatch):
    """The count runner forwards constructor options and masks errors before any fit starts."""
    from bayes_model_discovery.experimental import core_count
    from bayes_model_discovery.experimental.test_count_proposer import run_args
    secret = "synthetic-error-credential"
    monkeypatch.setenv(KEY_NAME, secret)
    captured = []
    def fail(records, **kwargs):
        """Capture the constructor keywords, then raise an error that contains the credential."""
        captured.append(kwargs)
        raise RuntimeError("constructor failed "+secret)
    monkeypatch.setattr(count, "OpenAICompatCountProposer", fail)
    args = run_args(tmp_path / "runner", proposer="openai_compat", llm_model=MODEL, allow_paid_api=True)
    args.llm_base_url, args.llm_key_name = URL, KEY_NAME
    with pytest.raises(count.MaskedCredentialError) as error:
        core_count.run(args)
    assert secret not in str(error.value)
    assert captured == [{"allow_live": True, "model": MODEL, "base_url": URL, "key_name": KEY_NAME}]
    assert list((tmp_path / "runner" / "fits").iterdir()) == []
    assert secret not in (tmp_path / "runner" / "failure.json").read_text()


def test_native_transport_reuses_openai_parse_and_zero_injected_cost(tmp_path, monkeypatch):
    """The neuron proposer reuses the OpenAI-shaped parse and records zero paid calls for an injected client."""
    import numpy as np

    from bayes_model_discovery.experimental import neuron_forward
    from bayes_model_discovery.experimental.test_neuron_proposer import reply, text_packet
    monkeypatch.setattr(neuron_forward, "predict_features", lambda *args: np.array([-60., -70., -59.]))
    client, calls = fake_client([response(json.dumps(reply()))])
    proposer = neuron.NeuronProposer(tmp_path / "neuron", world="synthetic", cell="100", provider="openai_compat",
        base_url=URL, key_name=KEY_NAME, model=MODEL, client=client)
    candidate, record = proposer.propose_one(text_packet(), context=CONTEXT)
    assert candidate is not None and len(calls) == 1 and record["paid_api_calls"] == 0
    assert record["provider"] == "openai_compat" and record["requested_model"] == MODEL
    assert record["responses"][0]["response_id"] == "synthetic-completion-id"
    assert record["responses"][0]["http_request_id"] == "synthetic-request-id"
    assert neuron.make_neuron_proposer("openai_compat", tmp_path / "forbidden", cell="000") is None
    assert not (tmp_path / "forbidden").exists()


def test_localhost_fake_http_receipt_keeps_distinct_model_and_ids(tmp_path, monkeypatch):
    """One local TLS request through the actual SDK; all credentials are synthetic."""
    if shutil.which("openssl") is None:
        pytest.skip("local TLS fixture requires openssl")
    import httpx
    import openai
    cert, private = tmp_path / "cert.pem", tmp_path / "private.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(private),
                    "-out", str(cert), "-days", "1", "-subj", "/CN=localhost"],
                   check=True, capture_output=True)
    requests = []
    class Handler(BaseHTTPRequestHandler):
        """Minimal local HTTP handler that answers one chat-completion request."""
        def log_message(self, *args):
            """Silence the default request logging."""
        def do_POST(self):
            """Record the POST request and reply with a synthetic chat completion."""
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            payload = json.dumps({"id": "local-completion", "model": "local-returned-edition", "object": "chat.completion", "created": 1,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response().choices[0].message.content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("x-request-id", "local-http-request")
            self.end_headers()
            self.wfile.write(payload)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, private)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    secret = "synthetic-loopback-key-only"
    monkeypatch.setenv(KEY_NAME, secret)
    endpoint = f"https://127.0.0.1:{server.server_port}/api/v1"
    try:
        with httpx.Client(verify=False, trust_env=False) as http_client, \
                openai.OpenAI(**count.compat_client_kwargs(endpoint, KEY_NAME, MODEL), http_client=http_client) as client:
            proposer = count.OpenAICompatCountProposer(tmp_path / "transport", base_url=endpoint, key_name=KEY_NAME, model=MODEL, client=client)
            models, record = proposer.propose_one("synthetic packet", context=CONTEXT, initial_family="binomial")
        assert len(models) == len(requests) == 1 and record["client_injected"] is True
        assert requests[0][0] == "/api/v1/chat/completions" and requests[0][1] == "Bearer "+secret
        assert requests[0][2]["model"] == record["requested_model"] == MODEL
        assert record["returned_model"] == "local-returned-edition"
        assert record["http_request_id"] == "local-http-request" and record["completion_id"] == "local-completion"
        assert secret not in json.dumps(record)
        assert all(secret not in p.read_text() for p in (tmp_path / "transport").iterdir())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("kind", ["count", "neuron"])
def test_both_sdk_factories_receive_explicit_options(tmp_path, monkeypatch, kind):
    """Both proposers pass only explicit options to the SDK client factory."""
    import openai
    secret = "synthetic-arbitrary-credential"
    monkeypatch.setenv("COHORT_ACCESS", secret)
    captured = []
    client, _ = fake_client([])
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: captured.append(kwargs) or client)
    common = {'base_url': URL, 'key_name': "COHORT_ACCESS", 'model': MODEL, 'allow_live': True}
    if kind == "count":
        count.OpenAICompatCountProposer(tmp_path / "records", **common)
    else:
        neuron.NeuronProposer(tmp_path / "records", world="synthetic", cell="100", provider="openai_compat", **common)
    assert captured == [{"base_url": URL, "api_key": secret, "max_retries": 0, "timeout": 120.0}]
    assert secret not in count.mask_secrets(secret)


@pytest.mark.parametrize("key_name", ["PATH", "HOME", "PYTHONPATH", "FB1_LLM_MODEL", "LLM_MODEL", "CHEMBENCH_MANIFEST", "bad-name"])
def test_reserved_or_invalid_key_names_are_refused_before_registration(tmp_path, key_name):
    """Reserved or malformed credential names are refused and never registered for masking."""
    before = count._COMPAT_SECRET_NAMES.copy()
    with pytest.raises(ValueError):
        count.OpenAICompatCountProposer(tmp_path / "out", base_url=URL, model=MODEL, key_name=key_name, client=object())
    assert count._COMPAT_SECRET_NAMES == before
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("provider", ["mock", "openai_compat"])
def test_points_metadata_is_host_only_and_default_columns_unchanged(tmp_path, provider):
    """Points files carry only the host of the base URL, and default columns stay unchanged."""
    from bayes_model_discovery.experimental import core_count
    config = {"profile": "paper", "proposer": provider, "llm_model": MODEL, "cell": "100", "seed": 11,
              "structure": "out", "truth_family": "beta_binomial", "output": str(tmp_path / "run"),
              "source_files_sha256": {"core_count.py": "synthetic-sha"}}
    if provider == "openai_compat":
        config.update(count.compat_config(argparse.Namespace(proposer=provider, llm_model=MODEL,
                      llm_base_url=URL, llm_key_name=KEY_NAME)))
    row = {"observation_count": 9, "bma": {"nll": 1., "coverage90": .9, "width90": 1., "kl": .1},
           "round": 1, "phase": "after_proposal"}
    path = tmp_path / "points.csv"
    core_count.write_points(path, config, [row], "completed", "")
    reader = csv.DictReader(path.read_text().splitlines())
    result, = list(reader)
    if provider == "mock":
        assert reader.fieldnames == list(core_count.CONTRACT_POINTS_COLUMNS)
    else:
        assert result["llm_provider"] == provider and result["llm_base_url_host"] == "compat.example.invalid"
        assert result["llm_key_name"] == KEY_NAME and URL not in path.read_text()
    assert result["label"] == "development"



def test_sdk_debug_byte_headers_mask_custom_credentials(monkeypatch, caplog):
    """Debug logging of byte headers and dicts masks custom credentials."""
    import logging
    secret = "synthetic-byte-header-credential"
    monkeypatch.setenv("COHORT_ACCESS", secret)
    count.validate_openai_compat(URL, "COHORT_ACCESS", MODEL)
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("httpcore.synthetic").debug("headers=%r", [(b"authorization", ("Bearer "+secret).encode())])
        logging.getLogger("openai.synthetic").debug({"authorization": secret})
    assert secret not in caplog.text
    assert "COHORT_ACCESS:masked" in caplog.text


# The provider-order tests use only synthetic provider names and injected HTTP responses.
@pytest.mark.parametrize("value", ["", " ", ",a", "a,", "a,,b", "a,a", "a, a ",
    "a\n,b", "a\t,b", "a\r,b", "a\x00,b", "a\x7f,b", "a\x85,b", "a\u2028,b",
    [], ["a", ""], ["a", "a"], ["a,b"], ["a", None], 1, True, {"a": 1}])
def test_provider_order_refuses_ambiguous_input_before_records(tmp_path, value):
    """Ambiguous provider-order values are refused before any records are written."""
    with pytest.raises(ValueError):
        count.OpenAICompatCountProposer(tmp_path / "invalid", base_url=URL, model=MODEL,
                                       provider_order=value, client=object())
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("value", [" alpha , beta ", [" alpha ", "beta"], ("alpha", " beta ")])
def test_provider_order_normalizes_a_defensive_copy(value):
    """Provider-order normalization returns a list that later mutation of the input cannot change."""
    assert count.normalize_provider_order(value) == ["alpha", "beta"]
    if isinstance(value, list):
        result = count.normalize_provider_order(value)
        value.append("later")
        assert result == ["alpha", "beta"]


@pytest.mark.parametrize("name", ["core_count", "boxinggym_core", "chembench_core", "neuron_core"])
def test_provider_order_cli_config_is_optional_and_compat_only(tmp_path, monkeypatch, name):
    """The provider-order option is optional and accepted only for the compatible proposer."""
    module = importlib.import_module("bayes_model_discovery.experimental." + name)
    seen = []
    monkeypatch.setattr(module, "run", lambda args: seen.append(args))
    extras = {"core_count": [], "boxinggym_core": ["--task", "death"],
              "chembench_core": ["--manifest", str(tmp_path / "synthetic-manifest.json"), "--noise", "0.1"],
              "neuron_core": []}[name]
    argv = [name, "--output", str(tmp_path / name), *extras]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert "llm_provider_order" not in vars(seen[-1])
    assert "provider_order" not in count.compat_config(seen[-1])
    compat = [*argv, "--proposer", "openai_compat", "--llm-base-url", URL,
              "--llm-key-name", KEY_NAME, "--llm-model", MODEL, "--allow-paid-api"]
    monkeypatch.setattr(sys, "argv", compat)
    module.main()
    assert "provider_order" not in count.compat_config(seen[-1])
    monkeypatch.setattr(sys, "argv", [*compat, "--llm-provider-order", " alpha , beta "])
    module.main()
    assert seen[-1].llm_provider_order == ["alpha", "beta"]
    cfg = count.compat_config(seen[-1])
    assert cfg["provider_order"] == ["alpha", "beta"] and "llm_provider_order" not in cfg
    for provider in ("mock", "anthropic", "openai"):
        with pytest.raises(ValueError, match="require proposer openai_compat"):
            count.compat_config(argparse.Namespace(proposer=provider, llm_provider_order=["alpha"]))


@pytest.mark.parametrize("kind", ["count", "neuron"])
def test_native_provider_cannot_accept_order(tmp_path, kind):
    """Native providers refuse a provider order."""
    with pytest.raises(ValueError, match="require provider openai_compat"):
        if kind == "count":
            count.OpenAICountProposer(tmp_path / "out", client=object(), provider_order=["alpha"])
        else:
            neuron.NeuronProposer(tmp_path / "out", world="synthetic", cell="100",
                                  provider="openai", client=object(), provider_order=["alpha"])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("key_name", ["LLM_PROVIDER_ORDER", "FB1_LLM_PROVIDER_ORDER", "LLM_PROVIDER_ORDER_SET",
                                     "selected_key", "key_to_clear", "effective_proposer"])
def test_provider_order_settings_cannot_be_credentials(tmp_path, key_name):
    """Runtime setting names are refused as credential names."""
    with pytest.raises(ValueError, match="runtime setting"):
        count.OpenAICompatCountProposer(tmp_path / "out", base_url=URL, model=MODEL,
                                       key_name=key_name, client=object())
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["count", "neuron"])
@pytest.mark.parametrize("wire_provider", ["synthetic-returned-provider", None, "__missing__"])
def test_real_sdk_records_provider_and_preserves_order_on_repair(tmp_path, monkeypatch, kind, wire_provider):
    """Actual SDK serialization/parsing with injected HTTP transport, no network."""
    import httpx
    import numpy as np
    import openai

    from bayes_model_discovery.experimental import neuron_forward
    from bayes_model_discovery.experimental.test_neuron_proposer import reply, text_packet
    monkeypatch.setattr(neuron_forward, "predict_features", lambda *args: np.array([-60., -70., -59.]))
    secret = "synthetic-stage2-credential"
    monkeypatch.setenv(KEY_NAME, secret)
    sent = []
    good = response().choices[0].message.content if kind == "count" else json.dumps(reply())
    def handle(request):
        """Mock HTTP transport handler: check the credential header and answer with a synthetic completion."""
        sent.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer " + secret
        payload = {"id": "completion-" + str(len(sent)), "model": "different-returned-model",
                   "object": "chat.completion", "created": 1,
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": "invalid JSON" if len(sent) == 1 else good},
                                "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}
        if wire_provider != "__missing__":
            payload["provider"] = wire_provider
        return httpx.Response(200, json=payload, headers={"x-request-id": "http-" + str(len(sent))})
    with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as http_client, \
            openai.OpenAI(**count.compat_client_kwargs(URL, KEY_NAME, MODEL), http_client=http_client) as client:
        kwargs = {'base_url': URL, 'key_name': KEY_NAME, 'model': MODEL, 'client': client, 'provider_order': " alpha , beta "}
        if kind == "count":
            proposer = count.OpenAICompatCountProposer(tmp_path / "records", **kwargs)
            candidate, record = proposer.propose_one("synthetic public packet", context=CONTEXT, initial_family="binomial")
        else:
            proposer = neuron.NeuronProposer(tmp_path / "records", world="synthetic", cell="100", provider="openai_compat", **kwargs)
            candidate, record = proposer.propose_one(text_packet(), context=CONTEXT)
    assert candidate and record["status"] == "parsed" and record["calls"] == 2 and record["repairs"] == 1
    assert len(sent) == 2
    assert all(body["provider"] == {"order": ["alpha", "beta"], "allow_fallbacks": False} for body in sent)
    assert all("extra_body" not in body for body in sent)  # SDK merges extension at the HTTP top level.
    expected = None if wire_provider == "__missing__" else wire_provider
    assert record["provider_order"] == ["alpha", "beta"] and record["returned_provider"] == expected
    assert [r["returned_provider"] for r in record["responses"]] == [expected, expected]
    assert record["requested_model"] == MODEL and record["returned_model"] == "different-returned-model"
    assert [r["http_request_id"] for r in record["responses"]] == ["http-1", "http-2"]
    id_key = "completion_id" if kind == "count" else "response_id"
    assert [r[id_key] for r in record["responses"]] == ["completion-1", "completion-2"]
    requests = json.loads((tmp_path / "records" / "round_01.request.json").read_text())
    assert requests["identity"]["provider_order"] == ["alpha", "beta"]
    assert all(b["extra_body"]["provider"]["order"] == ["alpha", "beta"] for b in requests["bodies"])
    assert secret not in json.dumps(record)
    assert all(secret not in p.read_text() for p in (tmp_path / "records").iterdir())


@pytest.mark.parametrize("kind", ["count", "neuron"])
def test_omitted_order_keeps_native_and_compat_request_bytes(tmp_path, monkeypatch, kind):
    """Without a provider order, native and compatible requests are byte-identical."""
    import numpy as np

    from bayes_model_discovery.experimental import neuron_forward
    from bayes_model_discovery.experimental.test_neuron_proposer import reply, text_packet
    monkeypatch.setattr(neuron_forward, "predict_features", lambda *args: np.array([-60., -70., -59.]))
    all_calls, all_records = [], []
    for provider in ("openai", "openai_compat"):
        text = response().choices[0].message.content if kind == "count" else json.dumps(reply())
        client, calls = fake_client([response(text)])
        kwargs = {'model': MODEL, 'client': client}
        if provider == "openai_compat":
            kwargs.update(base_url=URL, key_name=KEY_NAME)
        if kind == "count":
            proposer = count.OpenAICountProposer(tmp_path / provider, provider=provider, **kwargs)
            _, record = proposer.propose_one("synthetic packet", context=CONTEXT, initial_family="binomial")
        else:
            proposer = neuron.NeuronProposer(tmp_path / provider, world="synthetic", cell="100", provider=provider, **kwargs)
            _, record = proposer.propose_one(text_packet(), context=CONTEXT)
        all_calls.append(json.dumps(calls, sort_keys=True))
        all_records.append(record)
        assert "extra_body" not in calls[0] and "provider_order" not in record
        if provider == "openai":
            assert "returned_provider" not in record and "returned_provider" not in record["responses"][0]
        else:
            assert record["returned_provider"] is None and record["responses"][0]["returned_provider"] is None
    assert all_calls[0] == all_calls[1]


def test_p4_compat_order_and_returned_provider_are_recorded(tmp_path):
    """The event-based proposal path records the provider order and the returned provider."""
    candidate = json.loads(response().choices[0].message.content)
    reply = response(json.dumps([candidate]))
    reply.provider = "synthetic-returned-provider"
    client, calls = fake_client([reply])
    proposer = count.OpenAICompatCountProposer(tmp_path / "p4", base_url=URL, model=MODEL,
                                               client=client, provider_order=["alpha"])
    context = {k: v for k, v in CONTEXT.items() if k != "round"}
    context["event"] = 5
    _, record = proposer.propose([{"u": .5, "y": 1}] * 9, [], context=context, initial_family="binomial")
    assert calls[0]["extra_body"] == {"provider": {"order": ["alpha"], "allow_fallbacks": False}}
    assert record["provider_order"] == ["alpha"] and record["returned_provider"] == "synthetic-returned-provider"
    request = json.loads((tmp_path / "p4" / "event_05.request.json").read_text())
    assert request["body"] == calls[0]


@pytest.mark.parametrize("name", ["core_count", "boxinggym_core"])
def test_runner_provider_order_reaches_constructor_before_fit(tmp_path, monkeypatch, name):
    """The provider order reaches the proposer constructor before any fit starts."""
    from bayes_model_discovery.experimental.test_count_proposer import run_args
    module = importlib.import_module("bayes_model_discovery.experimental." + name)
    seen = []
    def fail(records, **kwargs):
        """Capture the constructor keywords, then stop the run with a synthetic error."""
        seen.append(kwargs)
        raise RuntimeError("synthetic constructor stop")
    monkeypatch.setattr(count, "OpenAICompatCountProposer", fail)
    args = run_args(tmp_path / "run", proposer="openai_compat", llm_model=MODEL, allow_paid_api=True)
    args.llm_base_url, args.llm_key_name, args.llm_provider_order = URL, KEY_NAME, ["alpha", "beta"]
    with pytest.raises(RuntimeError, match="synthetic constructor stop"):
        if name == "core_count":
            module.run(args)
        else:
            module.build_count_proposer(args, tmp_path)
    assert seen[0]["provider_order"] == ["alpha", "beta"]
    assert not list((tmp_path / "run").glob("fits/*"))



def test_chem_provider_order_reaches_constructor_with_synthetic_adapter(tmp_path, monkeypatch):
    """The ChemBench runner forwards the provider order to the constructor, using a synthetic adapter."""
    from bayes_model_discovery.experimental import chembench_core as module
    from bayes_model_discovery.experimental import test_chembench_core as fixture
    monkeypatch.setattr(fixture.real_adapter, "ChemBenchSession", fixture._FakeSession)
    monkeypatch.setattr(fixture.real_adapter, "sample_designs", fixture._fake_sample_designs)
    seen = []
    def stop(records, **kwargs):
        """Capture the constructor keywords, then stop the run with a synthetic error."""
        seen.append(kwargs)
        raise RuntimeError("synthetic constructor stop")
    monkeypatch.setattr(count, "OpenAICompatCountProposer", stop)
    args = fixture._args(tmp_path / "out", fixture._manifest(tmp_path), proposer="openai_compat",
                         llm_model=MODEL, allow_paid_api=True, cell="100", structure_proposals="free")
    args.llm_base_url, args.llm_key_name, args.llm_provider_order = URL, KEY_NAME, ["alpha", "beta"]
    with pytest.raises(RuntimeError, match="synthetic constructor stop"):
        module.run(args)
    assert seen[0]["provider_order"] == ["alpha", "beta"]
    assert list((tmp_path / "out" / "fits").iterdir()) == []


def test_neuron_provider_order_reaches_constructor_with_synthetic_world(tmp_path, monkeypatch):
    """The neuron runner forwards the provider order to the constructor, using a synthetic world."""
    from types import ModuleType

    from bayes_model_discovery.experimental import neuron_core as module
    fake = ModuleType("neuronbench")
    fake.load_world = lambda *args, **kwargs: NS(protocol_pool=[], test_protocols=[])
    worlds = ModuleType("neuronbench.worlds")
    worlds.build_I = lambda *args: None
    monkeypatch.setitem(sys.modules, "neuronbench", fake)
    monkeypatch.setitem(sys.modules, "neuronbench.worlds", worlds)
    seen = []
    def stop(provider, records, **kwargs):
        """Capture the constructor keywords, then stop the run with a synthetic error."""
        assert provider == "openai_compat"
        seen.append(kwargs)
        raise RuntimeError("synthetic constructor stop")
    monkeypatch.setattr(module, "make_neuron_proposer", stop)
    args = module.build_parser().parse_args(["--output", str(tmp_path / "out"), "--cell", "100",
        "--proposer", "openai_compat", "--llm-model", MODEL, "--llm-base-url", URL,
        "--llm-key-name", KEY_NAME, "--llm-provider-order", " alpha , beta ", "--allow-paid-api"])
    with pytest.raises(RuntimeError, match="synthetic constructor stop"):
        module.run(args)
    assert seen[0]["provider_order"] == ["alpha", "beta"]
    assert list((tmp_path / "out" / "fits").iterdir()) == []


# ---------------------------------------------------------------- reasoning and max-token options
@pytest.mark.parametrize("order", [["alpha", "beta"], None])
@pytest.mark.parametrize("kind", ["count", "count_p4", "neuron"])
def test_d1_reasoning_travels_in_extra_body_beside_the_provider_key(tmp_path, monkeypatch, kind, order):
    """The routing key and the reasoning spec share one `extra_body`; the
    identity records the decoding actually sent. Four OpenAI-shape transports
    are covered here and in the ForceBench snapshot's own suite."""
    import numpy as np

    from bayes_model_discovery.experimental import neuron_forward
    from bayes_model_discovery.experimental.test_neuron_proposer import reply, text_packet
    monkeypatch.setattr(neuron_forward, "predict_features", lambda *args: np.array([-60., -70., -59.]))
    kwargs = {"base_url": URL, "key_name": KEY_NAME, "model": MODEL, "reasoning": "max_tokens=512,exclude",
              **({"provider_order": order} if order is not None else {})}
    if kind == "neuron":
        client, calls = fake_client([response(json.dumps(reply()))])
        proposer = neuron.NeuronProposer(tmp_path / "records", world="synthetic", cell="100",
                                         provider="openai_compat", client=client, **kwargs)
        _, record = proposer.propose_one(text_packet(), context=CONTEXT)
    elif kind == "count":
        client, calls = fake_client([response()])
        proposer = count.OpenAICompatCountProposer(tmp_path / "records", client=client, **kwargs)
        _, record = proposer.propose_one("synthetic packet", context=CONTEXT, initial_family="binomial")
    else:
        client, calls = fake_client([response(json.dumps([json.loads(response().choices[0].message.content)]))])
        proposer = count.OpenAICompatCountProposer(tmp_path / "records", client=client, **kwargs)
        context = {**{k: v for k, v in CONTEXT.items() if k != "round"}, "event": 5}
        _, record = proposer.propose([{"u": .5, "y": 1}] * 9, [], context=context, initial_family="binomial")
    expected = {"reasoning": {"max_tokens": 512, "exclude": True}}
    if order is not None:
        expected = {"provider": {"order": order, "allow_fallbacks": False}, **expected}
    assert calls[0]["extra_body"] == expected
    assert list(calls[0]["extra_body"]) == list(expected)
    assert record["decoding"]["reasoning"] == {"max_tokens": 512, "exclude": True}
    assert record["decoding"]["max_tokens"] == 1600


@pytest.mark.parametrize("kind", ["count", "neuron"])
def test_d1_options_passed_as_none_keep_the_stated_request_bytes(tmp_path, monkeypatch, kind):
    """Byte invariance: `reasoning=None, max_tokens=None` is the same
    request as never mentioning either option."""
    import numpy as np

    from bayes_model_discovery import experimental
    from bayes_model_discovery.experimental import neuron_forward
    from bayes_model_discovery.experimental.test_neuron_proposer import reply, text_packet
    monkeypatch.setattr(neuron_forward, "predict_features", lambda *args: np.array([-60., -70., -59.]))
    assert experimental is not None
    recorded = []
    for extra in ({}, {"reasoning": None, "max_tokens": None}):
        text = response().choices[0].message.content if kind == "count" else json.dumps(reply())
        client, calls = fake_client([response(text)])
        kwargs = {"base_url": URL, "key_name": KEY_NAME, "model": MODEL, "client": client, **extra}
        directory = tmp_path / f"{kind}_{len(recorded)}"
        if kind == "count":
            proposer = count.OpenAICompatCountProposer(directory, **kwargs)
            _, record = proposer.propose_one("synthetic packet", context=CONTEXT, initial_family="binomial")
        else:
            proposer = neuron.NeuronProposer(directory, world="synthetic", cell="100",
                                             provider="openai_compat", **kwargs)
            _, record = proposer.propose_one(text_packet(), context=CONTEXT)
        assert "extra_body" not in calls[0] and calls[0]["max_tokens"] == 1600
        assert record["decoding"] == {"temperature": "not_sent/provider_default", "max_tokens": 1600}
        recorded.append(json.dumps(calls, sort_keys=True))
    assert recorded[0] == recorded[1]


@pytest.mark.parametrize("name", ["core_count", "boxinggym_core", "chembench_core", "neuron_core"])
def test_d1_runner_cli_states_the_two_options_only_when_typed(tmp_path, monkeypatch, name):
    """Runner command lines state the reasoning and max-token options only when they are typed."""
    module = importlib.import_module("bayes_model_discovery.experimental." + name)
    seen = []
    monkeypatch.setattr(module, "run", lambda args: seen.append(args))
    extras = {"core_count": [], "boxinggym_core": ["--task", "death"],
              "chembench_core": ["--manifest", str(tmp_path / "synthetic-manifest.json"), "--noise", "0.1"],
              "neuron_core": []}[name]
    argv = [name, "--output", str(tmp_path / name), *extras]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert "llm_reasoning" not in vars(seen[-1]) and "llm_max_tokens" not in vars(seen[-1])
    assert count.compat_option_kwargs(seen[-1], compat=False) == {}
    assert count.compat_option_kwargs(seen[-1], compat=True) == {}
    compat = [*argv, "--proposer", "openai_compat", "--llm-base-url", URL,
              "--llm-key-name", KEY_NAME, "--llm-model", MODEL, "--allow-paid-api"]
    monkeypatch.setattr(sys, "argv", [*compat, "--llm-reasoning", "exclude,effort=none",
                                     "--llm-max-tokens", "4000"])
    module.main()
    args = seen[-1]
    assert args.llm_reasoning == {"effort": "none", "exclude": True} and args.llm_max_tokens == 4000
    config = count.compat_config(args)
    assert config["llm_reasoning"] == {"effort": "none", "exclude": True}
    assert config["llm_max_tokens"] == 4000
    assert count.compat_option_kwargs(args, compat=True) == {
        "max_tokens": 4000, "reasoning": {"effort": "none", "exclude": True}}
    # The reasoning spec reaches every LLM transport's constructor, which validates it for its
    # provider (the stock OpenAI path would refuse this compat-only `exclude` there).
    assert count.compat_option_kwargs(args, compat=False) == {
        "max_tokens": 4000, "reasoning": {"effort": "none", "exclude": True}}
    # The budget is a decoding setting for every provider; so is the spec since the reasoning-transport work, for
    # every LLM provider - a mock run makes no call and still refuses a stated spec.
    monkeypatch.setattr(sys, "argv", [*argv, "--llm-max-tokens", "4000"])
    module.main()
    assert count.compat_config(seen[-1])["llm_max_tokens"] == 4000
    for provider in ("anthropic", "openai"):
        values = count.compat_config(argparse.Namespace(proposer=provider, llm_reasoning={"effort": "high"}))
        assert values["llm_reasoning"] == {"effort": "high"} and "llm_provider" not in values
    with pytest.raises(ValueError, match="requires an LLM proposer"):
        count.compat_config(argparse.Namespace(proposer="mock", llm_reasoning={"effort": "high"}))


@pytest.mark.parametrize("flags", [["--llm-reasoning", "effort=none,max_tokens=512"],
                                   ["--llm-reasoning", "effort=louder"], ["--llm-reasoning", "exclude"],
                                   ["--llm-max-tokens", "0"], ["--llm-max-tokens", "32769"],
                                   ["--llm-max-tokens", "1.5"]])
def test_d1_invalid_option_values_end_the_command_line_before_any_output(tmp_path, monkeypatch, flags):
    """Invalid reasoning or max-token values end the command line before any output is written."""
    from bayes_model_discovery.experimental import core_count
    monkeypatch.setattr(core_count, "run", lambda args: pytest.fail("run must not be reached"))
    monkeypatch.setattr(sys, "argv", ["core_count", "--output", str(tmp_path / "refused"),
                                      "--proposer", "openai_compat", "--llm-base-url", URL,
                                      "--llm-key-name", KEY_NAME, "--llm-model", MODEL,
                                      "--allow-paid-api", *flags])
    with pytest.raises(SystemExit) as refusal:
        core_count.main()
    assert refusal.value.code == 2
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("name", ["core_count", "boxinggym_core"])
def test_d1_stated_options_reach_the_constructor_before_any_fit(tmp_path, monkeypatch, name):
    """Stated reasoning and max-token options reach the constructor before any fit starts."""
    from bayes_model_discovery.experimental.test_count_proposer import run_args
    module = importlib.import_module("bayes_model_discovery.experimental." + name)
    seen = []

    def stop(records, **kwargs):
        """Capture the constructor keywords, then stop the run with a synthetic error."""
        seen.append(kwargs)
        raise RuntimeError("synthetic constructor stop")

    monkeypatch.setattr(count, "OpenAICompatCountProposer", stop)
    args = run_args(tmp_path / "run", proposer="openai_compat", llm_model=MODEL, allow_paid_api=True)
    args.llm_base_url, args.llm_key_name = URL, KEY_NAME
    args.llm_reasoning, args.llm_max_tokens = {"effort": "none"}, 4000
    with pytest.raises(RuntimeError, match="synthetic constructor stop"):
        if name == "core_count":
            module.run(args)
        else:
            module.build_count_proposer(args, tmp_path)
    assert seen[0]["reasoning"] == {"effort": "none"} and seen[0]["max_tokens"] == 4000
    assert not list((tmp_path / "run").glob("fits/*"))


@pytest.mark.parametrize("proposer", ["openai_compat", "anthropic"])
def test_d1_neuron_runner_forwards_both_options_exactly_once(tmp_path, monkeypatch, proposer):
    """The compatibility branch and the decoding branch must not both name
    `max_tokens` - a duplicated keyword would fail the constructor call."""
    from types import ModuleType

    from bayes_model_discovery.experimental import neuron_core as module
    fake = ModuleType("neuronbench")
    fake.load_world = lambda *args, **kwargs: NS(protocol_pool=[], test_protocols=[])
    worlds = ModuleType("neuronbench.worlds")
    worlds.build_I = lambda *args: None
    monkeypatch.setitem(sys.modules, "neuronbench", fake)
    monkeypatch.setitem(sys.modules, "neuronbench.worlds", worlds)
    seen = []

    def stop(requested, records, **kwargs):
        """Capture the constructor keywords, then stop the run with a synthetic error."""
        seen.append(kwargs)
        raise RuntimeError("synthetic constructor stop")

    monkeypatch.setattr(module, "make_neuron_proposer", stop)
    options = ["--proposer", proposer, "--llm-model", MODEL, "--llm-max-tokens", "4000",
               "--allow-paid-api"]
    if proposer == "openai_compat":
        options += ["--llm-base-url", URL, "--llm-key-name", KEY_NAME,
                    "--llm-reasoning", "effort=none"]
    args = module.build_parser().parse_args(["--output", str(tmp_path / "out"), "--cell", "100",
                                             *options])
    with pytest.raises(RuntimeError, match="synthetic constructor stop"):
        module.run(args)
    assert seen[0]["max_tokens"] == 4000
    assert seen[0].get("reasoning") == ({"effort": "none"} if proposer == "openai_compat" else None)
    assert list((tmp_path / "out" / "fits").iterdir()) == []
