"""Synthetic ForceBench compatibility transport checks; no provider requests."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SNAPSHOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SNAPSHOT))
import fb1_proposer as fp


def response(text=None):
    """Build a synthetic chat-completion reply carrying the given text or the default mock text."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=fp.mock_response_text(0) if text is None else text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
        id="completion-synthetic", model="returned-different-model")


def fake_client(reply=None):
    """Return a fake chat client and the list of request bodies it receives."""
    calls = []
    def create(**body):
        """Record the request body and return the configured or default reply."""
        calls.append(body)
        return reply if reply is not None else response()
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls


def context():
    """Return a synthetic proposal context."""
    return dict(run_id="synthetic", world="gravity", arm="100", policy="P1", event=1,
                round=1, n_designs=2, candidate_index=0, packet_file="synthetic.json")


def test_injected_compat_records_model_difference_and_separate_ids(tmp_path, monkeypatch):
    """The compatibility transport records the returned model and keeps request and response ids separate."""
    monkeypatch.setenv("CUSTOM_ROUTER_TOKEN", "fake-long-compat-credential")
    client, calls = fake_client()
    raw = SimpleNamespace(parse=response, headers={"x-request-id": "request-synthetic"})
    client.with_raw_response = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **body: (calls.append(body), raw)[1])))
    p = fp.Fb1CatalogProposer(tmp_path / "records", provider="openai_compat", client=client,
        model="synthetic/requested-model", base_url="https://example.invalid/private/path", key_name="CUSTOM_ROUTER_TOKEN")
    candidates, record = p.propose_one("{}", context=context(), constraints=fp.catalog_constraints())
    assert record["status"] == "parsed" and candidates
    assert record["provider"] == record["transport"] == "openai_compat"
    assert record["requested_model"] == "synthetic/requested-model"
    assert record["returned_model"] == "returned-different-model"
    assert record["http_request_id"] == "request-synthetic"
    assert record["response_id"] == record["completion_id"] == "completion-synthetic"
    assert record["llm_base_url_host"] == "example.invalid"
    assert record["llm_key_name"] == "CUSTOM_ROUTER_TOKEN"
    assert calls[0]["messages"][0]["role"] == "system"
    assert "system" not in calls[0]
    for file in (tmp_path / "records").glob("*.json"):
        text = file.read_text()
        assert "fake-long-compat-credential" not in text
        assert "/private/path" not in text


def test_sdk_constructor_receives_explicit_url_and_named_environment_key(tmp_path, monkeypatch):
    """The SDK constructor receives the declared URL and the key read from the named environment variable."""
    sdk = pytest.importorskip("openai")
    monkeypatch.setenv("SYNTHETIC_ROUTER_KEY", "fake-router-token")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.invalid/ignored")
    seen = []
    client, _ = fake_client()
    monkeypatch.setattr(sdk, "OpenAI", lambda **kw: (seen.append(kw), client)[1])
    p = fp.Fb1CatalogProposer(tmp_path / "records", allow_live=True, provider="openai_compat",
        model="synthetic/model", base_url="https://declared.invalid/v1", key_name="SYNTHETIC_ROUTER_KEY")
    assert seen == [dict(base_url="https://declared.invalid/v1", api_key="fake-router-token", max_retries=0, timeout=120.0)]
    assert p.transport == "openai_compat"


def test_short_dynamic_secret_and_constructor_errors_are_masked(tmp_path, monkeypatch):
    """A short credential and constructor errors are masked in the raised message."""
    sdk = pytest.importorskip("openai")
    monkeypatch.setenv("SYNTHETIC_SHORT_KEY", "abc")
    def broken(**kwargs):
        """Raise an error that echoes the credential."""
        raise RuntimeError("provider echoed abc")
    monkeypatch.setattr(sdk, "OpenAI", broken)
    with pytest.raises(Exception) as exc:
        fp.Fb1CatalogProposer(tmp_path / "records", allow_live=True, provider="openai_compat",
            model="synthetic", base_url="https://example.invalid/v1", key_name="SYNTHETIC_SHORT_KEY")
    assert "abc" not in str(exc.value)
    assert "SYNTHETIC_SHORT_KEY:masked" in str(exc.value)
    assert not (tmp_path / "records").exists()


@pytest.mark.parametrize("url", ["http://example.invalid/v1", "https://user:password@example.invalid/v1",
    "https://example.invalid/v1?key=fake", "https://example.invalid/#fake", "https://example.invalid:invalid/v1"])
def test_invalid_compat_url_refused_without_echo_or_output(tmp_path, url):
    """An invalid base URL is refused without being echoed and without output."""
    client, _ = fake_client()
    with pytest.raises(ValueError) as exc:
        fp.Fb1CatalogProposer(tmp_path / "records", provider="openai_compat", client=client,
            model="synthetic", base_url=url, key_name="SYNTHETIC_KEY")
    assert url not in str(exc.value)
    assert not (tmp_path / "records").exists()


def test_compat_needs_explicit_model_and_real_credential_for_constructed_client(tmp_path, monkeypatch):
    """A constructed compatibility client needs an explicit model and a configured credential."""
    client, _ = fake_client()
    with pytest.raises(ValueError, match="explicit model"):
        fp.Fb1CatalogProposer(tmp_path / "missing_model", provider="openai_compat", client=client,
                             base_url="https://example.invalid/v1")
    monkeypatch.delenv("UNCONFIGURED_SYNTHETIC_KEY", raising=False)
    with pytest.raises(ValueError, match="credential is not configured"):
        fp.Fb1CatalogProposer(tmp_path / "missing_key", allow_live=True, provider="openai_compat",
            model="synthetic", base_url="https://example.invalid/v1", key_name="UNCONFIGURED_SYNTHETIC_KEY")
    assert not (tmp_path / "missing_model").exists() and not (tmp_path / "missing_key").exists()


def test_runner_records_compatibility_metadata_and_uses_verified_v2_label_rule():
    """The runner records compatibility metadata and resolves labels through the label policy."""
    source = (SNAPSHOT / "run_gravity.py").read_text()
    tree = ast.parse(source)
    factory = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "Fb1CatalogProposer")
    supplied = next(node.value for node in factory.keywords if node.arg is None)
    assert isinstance(supplied, ast.IfExp)
    assert {key.value for key in supplied.body.keys} == {"base_url", "key_name", "provider_order"}
    assert "**proposer.compat_metadata" in source
    assert 'RUN_LABEL = LABEL_RECEIPT["run_label"]' in source
    assert "label_policy.resolve_label(" in source



def test_custom_secret_masks_nested_bytes_in_sdk_debug_records(monkeypatch):
    """A custom secret is masked inside nested bytes of SDK debug log records."""
    import logging
    monkeypatch.setenv("SYNTHETIC_HEADER_CREDENTIAL", "fake-header-token")
    fp.validate_compat_options("https://example.invalid/v1", "SYNTHETIC_HEADER_CREDENTIAL", "synthetic")
    for msg, args in (("headers=%s", ({b"authorization": b"Bearer fake-header-token"},)),
                      ({"headers": [b"fake-header-token"]}, ())):
        record = logging.LogRecord("httpcore.http11", logging.DEBUG, "synthetic", 1, msg, args, None)
        fp.SecretMaskingFilter().filter(record)
        text = logging.Formatter().format(record)
        assert "fake-header-token" not in text
        assert "SYNTHETIC_HEADER_CREDENTIAL:masked" in text


def test_config_environment_name_cannot_be_a_credential_alias(tmp_path):
    """A configuration variable name cannot be used as the credential name."""
    client, _ = fake_client()
    with pytest.raises(ValueError, match="must not name a runtime setting"):
        fp.Fb1CatalogProposer(tmp_path / "records", provider="openai_compat", client=client,
            model="synthetic", base_url="https://example.invalid/v1", key_name="CHEMBENCH_MANIFEST")
    assert not (tmp_path / "records").exists()


@pytest.mark.parametrize("returned", ["reported-supplier", None, {"unexpected": "shape"}])
def test_provider_order_and_reported_provider_are_recorded_without_identity_rejection(tmp_path, returned):
    """The provider order and the reported provider are recorded; the reported provider is not used to reject a reply."""
    reply = response()
    reply.provider = returned
    client, calls = fake_client(reply)
    p = fp.Fb1CatalogProposer(tmp_path / "records", provider="openai_compat", client=client,
        model="synthetic/requested", base_url="https://example.invalid/v1", key_name="TEST_ROUTER_CREDENTIAL",
        provider_order=" supplier-a , supplier-b ")
    candidates, record = p.propose_one("{}", context=context(), constraints=fp.catalog_constraints())
    assert candidates and record["status"] == "parsed"
    expected = returned if isinstance(returned, str) else None
    assert calls[0]["extra_body"] == {"provider": {"order": ["supplier-a", "supplier-b"], "allow_fallbacks": False}}
    assert record["provider_order"] == ["supplier-a", "supplier-b"]
    assert record["returned_provider"] == record["responses"][0]["returned_provider"] == expected
    assert record["provider"] == "openai_compat"
    assert record["requested_model"] == "synthetic/requested"
    assert record["returned_model"] == "returned-different-model"
    saved = json.loads(next((tmp_path / "records").glob("*.response.json")).read_text())
    assert saved["returned_provider"] == expected


def test_omitted_order_keeps_openai_request_body_and_null_missing_reported_provider(tmp_path):
    """Without a provider order the request body matches the native one and a missing reported provider is recorded as null."""
    native, native_calls = fake_client()
    native.provider = "openai"
    compat, compat_calls = fake_client()
    a = fp.Fb1CatalogProposer(tmp_path / "native", provider="openai", client=native, model="synthetic/model")
    b = fp.Fb1CatalogProposer(tmp_path / "compat", provider="openai_compat", client=compat,
        model="synthetic/model", base_url="https://example.invalid/v1", key_name="TEST_ROUTER_CREDENTIAL")
    _, native_record = a.propose_one("{}", context=context(), constraints=fp.catalog_constraints())
    _, compat_record = b.propose_one("{}", context=context(), constraints=fp.catalog_constraints())
    assert native_calls == compat_calls and "extra_body" not in compat_calls[0]
    assert "returned_provider" not in native_record and "provider_order" not in compat_record
    assert compat_record["returned_provider"] is None
    assert compat_record["responses"][0]["returned_provider"] is None


def test_repair_retains_order_and_each_response_provider(tmp_path):
    """A repair request keeps the provider order and each response keeps its reported provider."""
    replies = [response("not json"), response()]
    replies[0].provider, replies[1].provider = "reported-first", "reported-second"
    calls = []
    def create(**body):
        """Record the request body and return the next reply of the sequence."""
        calls.append(body)
        return replies[len(calls) - 1]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    p = fp.Fb1CatalogProposer(tmp_path / "records", provider="openai_compat", client=client,
        model="synthetic/model", base_url="https://example.invalid/v1", key_name="TEST_ROUTER_CREDENTIAL",
        provider_order=["supplier-a"])
    _, record = p.propose_one("{}", context=context(), constraints=fp.catalog_constraints())
    assert record["status"] == "parsed" and record["repairs"] == 1
    assert len(calls) == 2 and calls[0]["extra_body"] == calls[1]["extra_body"]
    assert [r["returned_provider"] for r in record["responses"]] == ["reported-first", "reported-second"]
    assert record["returned_provider"] == "reported-second"


@pytest.mark.parametrize("value", ["", [], "a,,b", "a,a", ["a", " a"], "a\nb", "a\tb", "a"+chr(128), ["a,b"], [2], 2])
def test_invalid_provider_order_refuses_before_output(tmp_path, value):
    """An invalid provider order is refused before any output."""
    client, _ = fake_client()
    with pytest.raises(ValueError, match="[Pp]rovider order"):
        fp.Fb1CatalogProposer(tmp_path / "records", provider="openai_compat", client=client,
            model="synthetic/model", base_url="https://example.invalid/v1", key_name="TEST_ROUTER_CREDENTIAL",
            provider_order=value)
    assert not (tmp_path / "records").exists()


def test_provider_order_is_for_compatibility_transport_only(tmp_path):
    """A provider order is accepted only by the compatibility transport."""
    client, _ = fake_client()
    client.provider = "openai"
    with pytest.raises(ValueError, match="requires the openai_compat"):
        fp.Fb1CatalogProposer(tmp_path / "records", provider="openai", client=client,
                             model="synthetic/model", provider_order="supplier-a")
    assert not (tmp_path / "records").exists()


@pytest.mark.parametrize("key_name", ["selected_key", "key_to_clear", "effective_proposer",
                                     "LLM_PROVIDER_ORDER", "LLM_PROVIDER_ORDER_SET", "FB1_LLM_PROVIDER_ORDER"])
def test_compat_key_cannot_shadow_routing_or_credential_gate_variables(tmp_path, key_name):
    """The credential name cannot shadow routing or credential-gate variables."""
    client, _ = fake_client()
    with pytest.raises(ValueError, match="must not name a runtime setting"):
        fp.Fb1CatalogProposer(tmp_path / "records", provider="openai_compat", client=client,
            model="synthetic/model", base_url="https://example.invalid/v1", key_name=key_name)
    assert not (tmp_path / "records").exists()


def scripted(replies):
    """An injected compatible client that answers `replies` in order."""
    calls = []

    def create(**body):
        """Record the request body and return the next scripted reply."""
        calls.append(body)
        return replies[min(len(calls) - 1, len(replies) - 1)]

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def empty_response(text, finish_reason):
    """Build a synthetic reply whose message content and finish reason are given."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                                    finish_reason=finish_reason)],
                           usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
                           id="completion-synthetic", model="returned-different-model")


def compat_proposer(records, **kwargs):
    """Build a compatibility-transport proposer with a fixed synthetic model and URL."""
    return fp.Fb1CatalogProposer(records, provider="openai_compat",
                                 model="synthetic/requested-model",
                                 base_url="https://example.invalid/private/path",
                                 key_name="CUSTOM_ROUTER_TOKEN", **kwargs)


# --------------------------------------------------------- reasoning option and token budget
@pytest.mark.parametrize("spec", [
    "effort=none", "effort=max", "max_tokens=512", "effort=high,exclude", "exclude,effort=high",
    " max_tokens = 512 , exclude ",
])
def test_d1_reasoning_grammar_matches_the_package_parser(spec):
    """The snapshot is a separate file, so the grammar is implemented twice;
    this pins the two implementations to the same result."""
    sys.path.insert(0, str(SNAPSHOT.parents[1] / "src"))
    package = pytest.importorskip("bayes_model_discovery.experimental.count_proposer")
    assert fp.parse_reasoning_spec(spec) == package.parse_reasoning_spec(spec)
    assert list(fp.parse_reasoning_spec(spec)) == list(package.parse_reasoning_spec(spec))


@pytest.mark.parametrize("spec", [
    "", "effort", "effort=none,max_tokens=512", "max_tokens=0", "effort=louder",
    "effort=none,effort=low", "exclude=true", "effort=none,unknown",
])
def test_d1_invalid_reasoning_grammar_is_refused_in_both_implementations(spec):
    """An invalid reasoning specification is refused by both parser implementations."""
    sys.path.insert(0, str(SNAPSHOT.parents[1] / "src"))
    package = pytest.importorskip("bayes_model_discovery.experimental.count_proposer")
    with pytest.raises(ValueError):
        fp.parse_reasoning_spec(spec)
    with pytest.raises(ValueError):
        package.parse_reasoning_spec(spec)


@pytest.mark.parametrize("order", [["alpha", "beta"], None])
def test_d1_reasoning_and_budget_reach_the_body_beside_the_provider_key(tmp_path, order):
    """The reasoning option and the token budget reach the request body beside the provider key."""
    client, calls = scripted([response()])
    proposer = compat_proposer(tmp_path / "records", client=client, reasoning="effort=none",
                               max_tokens=4000,
                               **({"provider_order": order} if order is not None else {}))
    candidates, record = proposer.propose_one("{}", context=context(),
                                              constraints=fp.catalog_constraints())
    expected = {"reasoning": {"effort": "none"}}
    if order is not None:
        expected = {"provider": {"order": order, "allow_fallbacks": False}, **expected}
    assert candidates and calls[0]["max_tokens"] == 4000
    assert calls[0]["extra_body"] == expected and list(calls[0]["extra_body"]) == list(expected)
    assert record["decoding"] == {"temperature": "not_sent/provider_default",
                                  "max_tokens": 4000, "reasoning": {"effort": "none"}}


def test_d1_unstated_options_keep_the_request_and_the_record(tmp_path):
    """Unstated options leave the request and the record unchanged."""
    bytes_seen = []
    for extra in ({}, {"reasoning": None, "max_tokens": None}):
        client, calls = scripted([response()])
        proposer = compat_proposer(tmp_path / f"records_{len(bytes_seen)}", client=client, **extra)
        _, record = proposer.propose_one("{}", context=context(),
                                         constraints=fp.catalog_constraints())
        assert "extra_body" not in calls[0] and calls[0]["max_tokens"] == fp.MAX_TOKENS
        assert record["decoding"] == {"temperature": "not_sent/provider_default",
                                      "max_tokens": fp.MAX_TOKENS}
        bytes_seen.append(json.dumps(calls, sort_keys=True))
    assert bytes_seen[0] == bytes_seen[1]


def test_d1_environment_names_are_read_when_the_caller_states_nothing(tmp_path, monkeypatch):
    """FB1_LLM_REASONING / FB1_LLM_MAX_TOKENS, read exactly where the base URL
    and key name are read, so the driver's cell environment reaches the wire."""
    monkeypatch.setenv("FB1_LLM_REASONING", "max_tokens=512,exclude")
    monkeypatch.setenv("FB1_LLM_MAX_TOKENS", "6000")
    client, calls = scripted([response()])
    proposer = compat_proposer(tmp_path / "records", client=client)
    proposer.propose_one("{}", context=context(), constraints=fp.catalog_constraints())
    assert calls[0]["extra_body"] == {"reasoning": {"max_tokens": 512, "exclude": True}}
    assert calls[0]["max_tokens"] == 6000
    monkeypatch.setenv("FB1_LLM_REASONING", "effort=louder")
    with pytest.raises(ValueError):
        compat_proposer(tmp_path / "refused", client=client)
    assert not (tmp_path / "refused").exists()


def test_d1_mock_transport_ignores_a_stray_budget_in_the_environment(tmp_path, monkeypatch):
    """The mock transport ignores a stray token budget in the environment."""
    monkeypatch.setenv("FB1_LLM_MAX_TOKENS", "6000")
    proposer = fp.Fb1CatalogProposer(tmp_path / "mock_records")
    assert proposer.transport == "mock" and proposer.max_tokens == fp.MAX_TOKENS


# --------------------------------------------------------- empty-reply repair
@pytest.mark.parametrize("empty", ["", "   ", None])
def test_d1_empty_reply_repairs_with_one_user_turn_and_no_echo(tmp_path, empty):
    """An empty reply is repaired with one extra user turn and no echo of the reply."""
    client, calls = scripted([empty_response(empty, "length"), response()])
    proposer = compat_proposer(tmp_path / "records", client=client)
    candidates, record = proposer.propose_one("{}", context=context(),
                                              constraints=fp.catalog_constraints())
    assert candidates and record["calls"] == 2 and record["repairs"] == 1
    assert [message["role"] for message in calls[1]["messages"]] == ["system", "user", "user"]
    assert calls[1]["messages"][2]["content"] == fp.REPAIR_INSTRUCTION_EMPTY.format(
        finish_reason="length")
    rejected = record["rejected_candidates"][0]
    assert rejected["reason"] == "empty response body (finish_reason=length)"
    assert rejected["repair_kind"] == "empty_body" and rejected["repair_eligible"] is True


def test_d1_a_truncated_reply_keeps_the_echo_and_is_marked_malformed(tmp_path):
    """A truncated reply keeps the echo in the repair and is marked malformed."""
    client, calls = scripted([empty_response('{"expression": "r**-2"', "length"), response()])
    proposer = compat_proposer(tmp_path / "records", client=client)
    candidates, record = proposer.propose_one("{}", context=context(),
                                              constraints=fp.catalog_constraints())
    assert candidates and [m["role"] for m in calls[1]["messages"]] == [
        "system", "user", "assistant", "user"]
    assert record["rejected_candidates"][0]["repair_kind"] == "malformed"


def test_d1_reasoning_tokens_are_recorded_only_when_reported(tmp_path):
    """Reasoning tokens are recorded only when the provider reports them."""
    reported = response()
    reported.usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=64)
    client, _ = scripted([reported])
    _, record = compat_proposer(tmp_path / "reported", client=client).propose_one(
        "{}", context=context(), constraints=fp.catalog_constraints())
    assert record["last_call_usage"]["reasoning_tokens"] == 64
    assert record["usage_total"]["reasoning_tokens"] == 64
    plain, _ = scripted([response()])
    _, silent = compat_proposer(tmp_path / "silent", client=plain).propose_one(
        "{}", context=context(), constraints=fp.catalog_constraints())
    assert "reasoning_tokens" not in silent["last_call_usage"]
