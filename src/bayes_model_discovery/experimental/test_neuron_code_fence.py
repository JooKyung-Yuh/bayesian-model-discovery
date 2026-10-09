"""Tests for the opt-in reply rule --llm-accept-code-fence (accepting a reply wrapped in one code fence). Synthetic replies only; no fitting, no live client."""
import argparse
import json

import pytest

from . import neuron_core as core
from . import neuron_proposer as p
from . import run_label_policy as policy
from .test_neuron_core_cli import _transport_reply, _TransportClient, transport_world  # noqa: F401
from .test_neuron_proposer import (  # noqa: F401
    DESIGNS,
    context,
    packet,
    probe,
    proposer,
    reply,
    response,
    text_packet,
)

FENCED = "```json\n{body}\n```"
BARE = "```\n{body}\n```"


def parse_text(text, *, accept):
    """Parse text."""
    return p.parse_neuron_candidate(text, constraints=packet()["constraints"], initial_designs=DESIGNS,
                                    accept_code_fence=accept)


@pytest.mark.parametrize("template", [FENCED, BARE, "  \n" + FENCED + "\n\n", "```JSON\n{body}\n```"])
def test_fence_alone_is_stripped(template):
    """Check that fence alone is stripped."""
    body = json.dumps(reply())
    inner, stripped = p.strip_outer_code_fence(template.format(body=body))
    assert stripped is True and inner == body


@pytest.mark.parametrize("text", [
    "Here is my proposal:\n" + FENCED,          # text before the fence
    FENCED + "\nThis channel explains the sag.",  # text after the fence
    "```json {body}```",                          # no newline after the opening fence
    "{body}",                                     # no fence at all
])
def test_anything_but_one_lone_fence_is_left_as_written(text):
    """Check that anything but one lone fence is left as written."""
    text = text.format(body=json.dumps(reply()))
    assert p.strip_outer_code_fence(text) == (text, False)


@pytest.mark.usefixtures("probe")
def test_fenced_json_passes_only_with_the_option():
    """Check that fenced json passes only with the option."""
    text = FENCED.format(body=json.dumps(reply()))
    candidate, _ = parse_text(text, accept=True)
    assert candidate.mechanism == "h_current"
    with pytest.raises(p.CandidateValidationError) as refused:
        parse_text(text, accept=False)
    assert refused.value.stage == "json"


@pytest.mark.usefixtures("probe")
def test_plain_json_parses_the_same_with_and_without_the_option():
    """Check that plain json parses the same with and without the option."""
    text = json.dumps(reply())
    on, meta_on = parse_text(text, accept=True)
    off, meta_off = parse_text(text, accept=False)
    assert on == off and meta_on == meta_off


@pytest.mark.usefixtures("probe")
def test_broken_json_inside_a_fence_is_refused_for_the_same_reason():
    """Check that broken json inside a fence is refused for the same reason."""
    broken = '{"mechanism": "h_current", "kind": '
    with pytest.raises(p.CandidateValidationError) as fenced:
        parse_text(FENCED.format(body=broken), accept=True)
    with pytest.raises(p.CandidateValidationError) as plain:
        parse_text(broken, accept=False)
    assert fenced.value.stage == plain.value.stage == "json"
    assert str(fenced.value) == str(plain.value)


@pytest.mark.usefixtures("probe")
def test_text_outside_the_fence_is_refused_as_without_the_option():
    """Check that text outside the fence is refused as without the option."""
    text = "Proposal follows.\n" + FENCED.format(body=json.dumps(reply()))
    with pytest.raises(p.CandidateValidationError) as on:
        parse_text(text, accept=True)
    with pytest.raises(p.CandidateValidationError) as off:
        parse_text(text, accept=False)
    assert on.value.stage == off.value.stage == "json" and str(on.value) == str(off.value)


@pytest.mark.usefixtures("probe")
def test_transport_records_name_the_rule_only_when_on(tmp_path):
    """Check that transport records name the rule only when on."""
    for sub in ("on", "off", "plain_on"):
        (tmp_path / sub).mkdir()
    fenced = response("anthropic", FENCED.format(body=json.dumps(reply())))
    on, _client_on = proposer(tmp_path / "on", "anthropic", [fenced], accept_code_fence=True)
    candidate, record = on.propose_one(text_packet(), context=context())
    assert candidate is not None and record["status"] == "parsed"
    assert record["reply_code_fence"] == p.CODE_FENCE_POLICY
    assert record["responses"][0]["code_fence_stripped"] is True
    request = json.loads((tmp_path / "on" / "records" / "round_01.request.json").read_text())
    assert request["identity"]["reply_code_fence"] == p.CODE_FENCE_POLICY

    plain = response("anthropic")
    off, client_off = proposer(tmp_path / "off", "anthropic", [plain])
    _, record_off = off.propose_one(text_packet(), context=context())
    assert "reply_code_fence" not in record_off
    assert all("code_fence_stripped" not in r for r in record_off["responses"])
    # the option changes no request byte
    plain_on, client_plain_on = proposer(tmp_path / "plain_on", "anthropic", [response("anthropic")],
                                         accept_code_fence=True)
    plain_on.propose_one(text_packet(), context=context())
    assert client_plain_on.calls == client_off.calls


@pytest.mark.usefixtures("probe")
def test_default_proposer_repairs_a_fenced_reply_as_before(tmp_path):
    """Check that default proposer repairs a fenced reply as before."""
    fenced = response("anthropic", FENCED.format(body=json.dumps(reply())))
    instance, _client = proposer(tmp_path, "anthropic", [fenced, response("anthropic")])
    candidate, record = instance.propose_one(text_packet(), context=context())
    assert candidate is not None and record["calls"] == 2 and record["repairs"] == 1
    assert record["rejected_candidates"][0]["stage"] == "json"


def test_option_must_be_boolean(tmp_path):
    """Check that option must be boolean."""
    with pytest.raises(ValueError, match="accept_code_fence must be a boolean"):
        proposer(tmp_path, "anthropic", [], accept_code_fence="yes")


def test_command_line_flag_is_absent_unless_typed():
    """Check that command line flag is absent unless typed."""
    parser = core.build_parser()
    assert not hasattr(parser.parse_args(["--output", "x"]), "llm_accept_code_fence")
    assert parser.parse_args(["--output", "x", "--llm-accept-code-fence"]).llm_accept_code_fence is True


def test_label_policy_registers_the_rule_only_when_stated():
    """Check that label policy registers the rule only when stated."""
    base = {"profile": "dev", "rounds": 1, "particles": None, "forward_backend": "native", "cell": "111"}
    without = policy.bmd_runtime_settings("neuron", argparse.Namespace(**base))
    stated = policy.bmd_runtime_settings("neuron", argparse.Namespace(**base, llm_accept_code_fence=True))
    assert "llm_accept_code_fence" not in without
    assert stated == {**without, "llm_accept_code_fence": True}


class _FencedClient(_TransportClient):
    """test_neuron_core_cli's synthetic Anthropic client with its text reply inside one ```json fence."""

    def create(self, **kwargs):
        """Create."""
        message = super().create(**kwargs)
        message.content[0].text = FENCED.format(body=message.content[0].text)
        return message


def _fenced_run(out, *extra):
    client = _FencedClient("anthropic", [_transport_reply()] * 3)
    args = core.build_parser().parse_args([
        "--output", str(out), "--cell", "111", "--proposer", "anthropic", "--allow-paid-api",
        "--llm-model", "synthetic_requested_model", "--rounds", "1", "--run-label", "development", *extra])
    core.run(args, client=client)
    transport = {path.name: json.loads(path.read_text()) for path in sorted((out / "llm_transport").glob("*.json"))}
    return (client, json.loads((out / "config.json").read_text()), json.loads((out / "summary.json").read_text()),
            transport)


@pytest.mark.usefixtures("transport_world")
def test_runner_llm_path_parses_a_fenced_reply_and_records_the_rule_only_when_on(tmp_path):
    """Check that runner llm path parses a fenced reply and records the rule only when on."""
    client, config, summary, transport = _fenced_run(tmp_path / "on", "--llm-accept-code-fence")
    request, response_record = transport["round_01.request.json"], transport["round_01.response.json"]
    assert config["llm_accept_code_fence"] is True
    assert config["label_receipt"]["binding"]["settings"]["llm_accept_code_fence"] is True
    assert request["identity"]["reply_code_fence"] == response_record["reply_code_fence"] == p.CODE_FENCE_POLICY
    assert response_record["status"] == "parsed" and response_record["calls"] == 1
    assert [row["code_fence_stripped"] for row in response_record["responses"]] == [True]
    assert response_record["responses"][0]["raw_text"].startswith("```json\n")
    assert summary["admitted"] == ["hh_m_current_gaussian_features"] and len(client.calls) == 1

    client, config, summary, transport = _fenced_run(tmp_path / "off")
    text = json.dumps(config) + json.dumps(transport)
    assert "accept_code_fence" not in text and "reply_code_fence" not in text and "code_fence_stripped" not in text
    response_record = transport["round_01.response.json"]
    # without the option the fenced reply is a JSON failure, repaired once (the engine's one repair), then the round fails
    assert response_record["status"] == "proposal_failed"
    assert response_record["calls"] == 2 and response_record["repairs"] == 1
    assert [row["stage"] for row in response_record["rejected_candidates"]] == ["json", "json"]
    assert summary["admitted"] == [] and len(client.calls) == 2
