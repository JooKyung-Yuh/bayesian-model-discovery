"""A development prompt variant and input-name masking for the ForceBench proposer.

Injected scripted clients and synthetic packets only: no key, no network, no paid call. The wire
test uses the installed `anthropic` SDK over `httpx.MockTransport` with a synthetic key. The
runner checks import `run_gravity` in a child interpreter and stop before any output exists.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")  # fb1_proposer imports fb1_core, which needs the ForceBench interpreter

SNAPSHOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SNAPSHOT))

import fb1_grammar as fbg
import fb1_proposer as fp
from test_fb1_label_rule import native_import
from test_fb1_proposer import _state
from test_fb1_reasoning_and_paid_call_guard import CONTEXT as PRE_UNIT_CONTEXT
from test_fb1_reasoning_and_paid_call_guard import record_hashes

#: The prompt the count proposers use: the repository copy beside the fb1_launch_c kit.
FA_C = Path(__file__).resolve().parents[1] / "fb1_launch_c" / "proposer_system_v2_9_FA-C.txt"
FA_C_SHA256 = "196f82b7a0f94e12a7cb8be2d07e776b1536aed11bf8ebd4bea1872c7002953c"
FA_C_EXTRA_KEYS = {"predicted_diagnostic": {"input": "X1", "low": 1.0, "high": 2.0},
                   "closest_archived_model": "c*X2/X1^2"}
SYNTHETIC_VARIANT = "# prompt_version: vT.synthetic\nPropose one model as one JSON object.\n"
CONTEXT = {**PRE_UNIT_CONTEXT, "run_id": "synthetic"}
TOKEN = re.compile(r"[A-Za-z0-9_]+")
REAL_NAMES = frozenset(fbg.INPUT_NAMES)
TIMES = np.linspace(0.5, 5.0, 10)

# One synthetic reply per kind and mode, written in the engine's names.
FREE_VALID = {"expression": "c*p1/(r**2 + d)", "family": "gauss",
              "parameters": {"c": "positive", "d": "positive"}}
REPLIES = {
    "catalog": {
        "valid": json.loads(fp.mock_response_text(0)),
        "bad_family": {**json.loads(fp.mock_response_text(0)), "family": "poisson"},
        "refused_expression": {"expression": "c*p1/r**2", "family": "gauss", "parameters": {"c": "positive"}},
    },
    "free": {
        "valid": FREE_VALID,
        "bad_family": {**FREE_VALID, "family": "poisson"},
        "refused_expression": {**FREE_VALID, "expression": "c*p1*q/(r**2 + d)"},
    },
}
#: The request paths of one proposal and the number of requests each makes.
PATHS = {"first_proposal": (["valid"], 1),
         "repair": (["bad_family", "valid"], 2),
         "repair_after_a_refused_expression": (["refused_expression", "valid"], 2)}


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Clear the option environment variables around each test and reset the module-level options afterwards."""
    for name in ("FB1_LLM_REASONING", "FB1_MAX_PAID_CALLS", "FB1_LLM_MAX_TOKENS", "FB1_LLM_PROVIDER",
                 "FB1_PROMPT_VARIANT", "FB1_MASK_INPUT_NAMES"):
        monkeypatch.delenv(name, raising=False)
    yield
    fp.set_prompt_variant(None)
    fp.set_input_name_mask(None)


def real_name_tokens(text):
    """Tokens with a name segment that is one of the engine's input names."""
    return sorted({token for token in TOKEN.findall(text) if REAL_NAMES & set(token.split("_"))})


def engine_written(body):
    """What the engine wrote in one request: the system text and every user turn."""
    return [body["system"], *[turn["content"] for turn in body["messages"] if turn["role"] == "user"]]


def state(mode):
    """Build a belief packet and its constraints for the given structure mode."""
    rows, members, constraints, interface = _state()
    if mode == "free":
        constraints = fp.free_constraints("gauss", family_expansion=True, times=TIMES)
    return fp.build_belief_packet(rows, members, [], constraints, interface, 1), constraints


def run(tmp_path, mode, kinds, *, mask=None, client=None, extra=None, **options):
    """One proposal over scripted replies; a masked run is given the replies in the shown names."""
    packet, constraints = state(mode)
    texts = [json.dumps({**REPLIES[mode][kind], **(extra or {})}, sort_keys=True) for kind in kinds]
    if mask is not None:
        packet, texts = mask.mask_value(packet), [mask.mask_reply(text) for text in texts]
    client = fp.ScriptedClient(texts) if client is None else client
    proposer = fp.Fb1CatalogProposer(tmp_path, client=client, structure_mode=mode,
                                     **({"name_mask": mask} if mask is not None else {}), **options)
    candidates, record = proposer.propose_one(fp.serialize_packet(packet), context=CONTEXT,
                                              constraints=constraints)
    return candidates, record, client


# ------------------------------------------------------------------ the prompt variant
def test_without_a_variant_the_frozen_prompt_is_sent(tmp_path):
    """Without a variant the frozen prompt is sent and recorded."""
    assert not fp.prompt_variant_active()
    assert fp.system_prompt_bytes() == fp.PROMPT_FILE.read_bytes()
    assert fp.prompt_identity() == {"prompt_version": "v2.0", "prompt_sha256": fp.PROMPT_SHA256}
    _candidates, record, client = run(tmp_path / "records", "catalog", ["valid"])
    assert client.bodies[0]["system"] == fp.PROMPT_FILE.read_text()
    assert record["prompt_version"] == "v2.0" and record["prompt_sha256"] == fp.PROMPT_SHA256


def test_a_variant_file_is_sent_as_stored_and_named_in_the_records(tmp_path):
    """A variant file is sent exactly as stored and its tag and hash are recorded."""
    variant = tmp_path / "variant.txt"
    variant.write_text(SYNTHETIC_VARIANT)
    assert fp.set_prompt_variant(variant) is None and fp.prompt_variant_active()
    sha = hashlib.sha256(SYNTHETIC_VARIANT.encode()).hexdigest()
    assert fp.prompt_identity() == {"prompt_version": "vT.synthetic", "prompt_sha256": sha}
    candidates, record, client = run(tmp_path / "records", "catalog", ["valid"])
    assert len(candidates) == 1 and client.bodies[0]["system"] == SYNTHETIC_VARIANT
    assert record["prompt_version"] == "vT.synthetic" and record["prompt_sha256"] == sha
    stored = json.loads((tmp_path / "records" / "event_00_round_01_c0.request.json").read_text())
    assert stored["identity"]["prompt_version"] == "vT.synthetic" and stored["identity"]["prompt_sha256"] == sha
    assert stored["bodies"][0]["system"] == SYNTHETIC_VARIANT
    assert fp.set_prompt_variant(None) == (str(variant), "vT.synthetic")
    assert fp.prompt_identity() == {"prompt_version": "v2.0", "prompt_sha256": fp.PROMPT_SHA256}


@pytest.mark.parametrize(("content", "message"), [
    ("Propose one model.\n", "must declare its tag on line 1"),
    ("", "must declare its tag on line 1"),
    ("# prompt_version:   \nPropose one model.\n", "declares an empty tag"),
])
def test_a_variant_file_without_a_tag_is_refused(tmp_path, content, message):
    """A variant file without a valid tag line is refused."""
    variant = tmp_path / "variant.txt"
    variant.write_text(content)
    with pytest.raises(ValueError, match=message):
        fp.set_prompt_variant(variant)
    assert not fp.prompt_variant_active()
    with pytest.raises(FileNotFoundError):
        fp.set_prompt_variant(tmp_path / "absent.txt")


def test_the_selection_leaves_the_default_path_bytes(tmp_path):
    """Selecting and then clearing a variant leaves the default-path bytes unchanged."""
    variant = tmp_path / "variant.txt"
    variant.write_text(SYNTHETIC_VARIANT)
    fp.set_prompt_variant(variant)
    fp.set_prompt_variant(None)
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=fp.ScriptedClient([fp.mock_response_text(0)]))
    packet, constraints = state("catalog")
    proposer.propose_one(fp.serialize_packet(packet), context=PRE_UNIT_CONTEXT, constraints=constraints)
    # The hashes from before the reasoning-transport change, which the tests of that change pin for this same proposal.
    from test_fb1_reasoning_and_paid_call_guard import PRE_UNIT_DEFAULT
    assert record_hashes(tmp_path / "records") == PRE_UNIT_DEFAULT["scripted"]


@pytest.mark.parametrize("mode", ["catalog", "free"])
def test_fa_c_is_sent_byte_for_byte_and_its_two_extra_reply_keys_are_recorded_and_dropped(tmp_path, mode):
    """The FA-C prompt is sent byte for byte and its two extra reply keys are recorded and dropped."""
    if not FA_C.is_file():
        pytest.skip("the FA-C prompt file is not on this host")
    assert hashlib.sha256(FA_C.read_bytes()).hexdigest() == FA_C_SHA256
    fp.set_prompt_variant(FA_C)
    assert fp.prompt_identity() == {"prompt_version": "v2.9.FA-C", "prompt_sha256": FA_C_SHA256}
    three_key, _record, _client = run(tmp_path / "three", mode, ["valid"])
    candidates, record, client = run(tmp_path / "five", mode, ["valid"], extra=FA_C_EXTRA_KEYS)
    assert client.bodies[0]["system"].encode() == FA_C.read_bytes()
    assert record["status"] == "parsed" and record["repairs"] == 0
    assert record["prompt_version"] == "v2.9.FA-C" and record["prompt_sha256"] == FA_C_SHA256
    assert record["ignored_keys"] == FA_C_EXTRA_KEYS
    assert candidates == three_key and not set(FA_C_EXTRA_KEYS) & set(candidates[0])


def test_the_fa_c_text_reaches_the_wire_through_the_installed_sdk(tmp_path):
    """The FA-C prompt reaches the wire unchanged through the installed SDK."""
    if not FA_C.is_file():
        pytest.skip("the FA-C prompt file is not on this host")
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")
    sent = []
    mask = fp.InputNameMask()
    reply = json.dumps({**json.loads(mask.mask_reply(json.dumps(REPLIES["catalog"]["valid"]))), **FA_C_EXTRA_KEYS})

    def handler(request):
        """Capture the request body and return a synthetic message reply."""
        sent.append(json.loads(request.content))
        return httpx.Response(200, headers={"request-id": "synthetic-request-id"}, json={
            "id": "msg_synthetic", "type": "message", "role": "assistant", "model": "synthetic-returned-model",
            "content": [{"type": "text", "text": reply}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 10}})

    fp.set_prompt_variant(FA_C)
    client = anthropic.Anthropic(api_key="sk-ant-synthetic-not-a-key", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    candidates, record, _client = run(tmp_path / "records", "catalog", [], mask=mask, client=client,
                                      provider="anthropic", model="synthetic-model")
    assert len(candidates) == 1 and record["status"] == "parsed" and len(sent) == 1
    assert sent[0]["system"].encode() == FA_C.read_bytes()
    assert record["prompt_version"] == "v2.9.FA-C" and record["ignored_keys"] == FA_C_EXTRA_KEYS
    assert real_name_tokens(sent[0]["system"] + sent[0]["messages"][0]["content"]) == []
    stored = json.loads((tmp_path / "records" / "event_00_round_01_c0.request.json").read_text())
    assert sent[0] == stored["bodies"][0]  # the wire JSON is the recorded body, key for key


# ------------------------------------------------------------------ the name mask
def test_the_map_follows_the_declared_input_order():
    """The name map follows the declared input order and swapping is its own inverse."""
    mask = fp.InputNameMask()
    assert tuple(fbg.INPUT_NAMES) == ("r", "p1", "p2", "t")
    assert mask.input_name_map == {"X1": "r", "X2": "p1", "X3": "p2", "X4": "t"}
    assert list(mask.structure_name_map) == list(fbg.CATALOG_STRUCTURE_NAMES)
    assert list(mask.structure_name_map.values()) == [f"K{i:02d}" for i in range(1, len(fbg.CATALOG_STRUCTURE_NAMES) + 1)]
    for text in ("c*p1*p2/r**2 + X1*t", "min_r", "r_points", "X9 + q"):
        assert mask.swap_names(mask.swap_names(text)) == text
    assert mask.swap_names("c*p1*p2/r**2 + X1*t") == "c*X2*X3/X1**2 + r*X4"


def numbers(value):
    """Collect every numeric leaf of a nested value, excluding booleans."""
    if isinstance(value, dict):
        return [n for item in value.values() for n in numbers(item)]
    if isinstance(value, list):
        return [n for item in value for n in numbers(item)]
    return [value] if isinstance(value, (int, float)) and not isinstance(value, bool) else []


@pytest.mark.parametrize("mode", ["catalog", "free"])
def test_a_masked_packet_has_no_input_name_and_keeps_every_number(mode):
    """A masked packet shows no input name and keeps every number."""
    packet, constraints = state(mode)
    mask = fp.InputNameMask()
    masked = mask.mask_value(packet)
    assert real_name_tokens(fp.serialize_packet(packet))  # the check sees the names when they are there
    assert real_name_tokens(fp.serialize_packet(masked)) == []
    assert numbers(masked) == numbers(packet)
    # The pool's identifiers are replaced, and a second pass leaves a replaced one as it is.
    shown_ids = [row["model_id"] for row in masked["pool"]]
    assert not set(shown_ids) & {row["model_id"] for row in packet["pool"]}
    assert [row["model_id"] for row in mask.mask_value(masked)["pool"]] == shown_ids
    assert [sorted(row) for row in masked["observations"]] != [sorted(row) for row in packet["observations"]]
    if mode == "catalog":
        # Every catalogue expression the model can copy comes back as the engine's own string.
        shown = masked["constraints"]["initial_structures"]
        assert len(set(shown)) == len(shown) == len(constraints["initial_structures"])
        for original, text in zip(constraints["initial_structures"], shown):
            reply = json.dumps({"expression": text, "family": "gauss", "parameters": {}})
            assert json.loads(mask.unmask_reply(reply, constraints))["expression"] == original
        assert mask.mask_expression("c*p1/r * (1 + A1 cos(wt) + A2 sin(wt))") == (
            "c*X2/X1 * (1 + A1 cos(w*X4) + A2 sin(w*X4))")


def test_the_free_grammar_block_shows_no_catalogue_label_and_keeps_the_call_names():
    """The free-grammar block shows no catalogue label and keeps the call names."""
    constraints = fp.free_constraints("gauss", family_expansion=True, times=TIMES)
    mask = fp.InputNameMask()
    masked = mask.mask_value(constraints)
    text = json.dumps(masked)
    assert masked["grammar"]["calls"] == constraints["grammar"]["calls"] and "power" in masked["grammar"]["calls"]
    labels = set(fbg.CATALOG_STRUCTURE_NAMES)
    assert [token for token in TOKEN.findall(text) if token in labels] == ["power"]  # the call name
    assert real_name_tokens(text) == []
    assert sorted(masked["grammar"]["catalogue_expressible"]) == sorted(
        mask.structure_name_map[name] for name in constraints["grammar"]["catalogue_expressible"])
    assert set(masked["grammar"]["duplicate_detection"]["catalogue_prior_match"]) <= set(
        mask.structure_name_map.values())
    assert "at round k is first fit at round k+1" in json.dumps(mask.mask_value(state("free")[0]))


def test_the_three_sentences_that_name_a_structure_are_relabelled():
    """The sentences that name a structure are relabelled; other text is untouched."""
    mask = fp.InputNameMask()
    assert mask.mask_text("force is x from power (max relative difference 1e-09 < 1e-06 over 9 probe points)") == (
        "force is x from K06 (max relative difference 1e-09 < 1e-06 over 9 probe points)")
    assert mask.mask_text("canonical id matches the catalogue law power with the same priors") == (
        "canonical id matches the catalogue law K06 with the same priors")
    assert mask.mask_text("power: force is x from crossover (max") == "K06: force is x from K14 (max"
    assert mask.mask_text("power(x, k) raises x to the power k") == "power(x, k) raises x to the power k"
    assert mask.mask_structure_name("power") == "K06" and mask.mask_structure_name("inv_r2") == "K01"


def test_a_reply_is_judged_in_the_engines_names():
    """A reply is translated back to the engine's names before it is judged."""
    mask = fp.InputNameMask()
    free = fp.free_constraints("gauss", family_expansion=True, times=TIMES)
    reply = {"expression": "c*X2/(X1**2 + r)", "family": "gauss",
             "parameters": {"c": "positive", "r": "positive"}, **FA_C_EXTRA_KEYS}
    back = json.loads(mask.unmask_reply(json.dumps(reply), free))
    # A coefficient the model calls `r` is the engine's `X1`: the swap is its own inverse.
    assert back["expression"] == "c*p1/(r**2 + X1)" and back["parameters"] == {"c": "positive", "X1": "positive"}
    assert {key: back[key] for key in FA_C_EXTRA_KEYS} == FA_C_EXTRA_KEYS  # other keys as written
    candidate, ignored = fp.parse_candidate_object(json.dumps(back), free)
    assert candidate["expression"] == "c*p1/(r**2 + X1)" and ignored["ignored_keys"] == FA_C_EXTRA_KEYS
    for text in ("not json", "[1, 2]", '"text"'):
        assert mask.unmask_reply(text, free) == text
    with pytest.raises(ValueError, match="unknown input name X2"):
        fp.parse_candidate_object(mask.unmask_reply(json.dumps({**reply, "expression": "c*p1/X1"}), free), free)
    assert mask.mask_text("unknown input name X2") == "unknown input name p1"  # the model's own token


def test_identifiers_shown_to_the_model_are_the_masked_ones_and_the_pairs_are_kept():
    """Identifiers shown to the model are the masked ones, and the pairs are kept."""
    mask = fp.InputNameMask()
    silent = fp.model_id("c*p1/r^2", "gauss", {"c": "positive"})
    assert mask.pairs()[0] != {"model_id": silent} and all("rng_model_id" not in row for row in mask.pairs())
    fp.set_input_name_mask(mask)
    original = fp.model_id("c*p1/r^2", "gauss", {"c": "positive"})
    assert original == silent  # the engine's identifier does not depend on the option
    expected = hashlib.sha256(b"c*X2/X1^2|c:positive|gauss").hexdigest()[:20]
    assert mask.mask_text(f"matched {original}") == f"matched {expected}"
    assert mask.mask_value({"matched_model_id": original}) == {"matched_model_id": expected}
    assert mask.mask_text(expected) == expected
    other = "0123456789abcdef0123"  # an identifier model_id did not compute
    relabelled = mask.mask_text(other)
    assert relabelled != other and re.fullmatch(r"[0-9a-f]{20}", relabelled) and mask.mask_text(relabelled) == relabelled
    free_name = mask.mask_text("g_0123456789ab")
    assert free_name != "g_0123456789ab" and mask.mask_text(free_name) == free_name
    pairs = mask.pairs()
    assert {"model_id": expected, "rng_model_id": original} in pairs
    assert {"model_id": relabelled, "rng_model_id": other} in pairs
    assert {"structure": free_name, "rng_structure": "g_0123456789ab"} in pairs
    assert {"structure": "K01", "rng_structure": "inv_r2"} in pairs


@pytest.mark.parametrize("mode", ["catalog", "free"])
@pytest.mark.parametrize("path", sorted(PATHS))
def test_no_request_of_a_masked_proposal_carries_an_input_name(tmp_path, mode, path):
    """No request of a masked proposal carries an input name."""
    kinds, requests = PATHS[path]
    plain_candidates, plain, plain_client = run(tmp_path / "plain", mode, kinds)
    mask = fp.InputNameMask()
    fp.set_input_name_mask(mask)
    candidates, record, client = run(tmp_path / "masked", mode, kinds, mask=mask)
    assert len(client.bodies) == len(plain_client.bodies) == requests
    # The same check finds the names in every request of the unmasked proposal.
    assert all(real_name_tokens("\n".join(engine_written(body))) for body in plain_client.bodies)
    for body in client.bodies:
        assert real_name_tokens("\n".join(engine_written(body))) == []
        assert real_name_tokens("\n".join(turn["content"] for turn in body["messages"])) == []
    stored = (tmp_path / "masked" / "event_00_round_01_c0.request.json").read_text()
    assert real_name_tokens(stored) == []
    # The engine receives what it receives without the option.
    assert candidates == plain_candidates and len(candidates) == 1
    assert record["status"] == plain["status"] == "parsed" and record["repairs"] == plain["repairs"] == requests - 1
    assert [mask.mask_text(item["reason"]) for item in plain["rejected_candidates"]] == [
        item["reason"] for item in record["rejected_candidates"]]
    assert not set(candidates[0]["model_id"]) - set("0123456789abcdef")
    assert candidates[0]["model_id"] not in stored


def test_a_name_the_model_was_never_shown_is_refused_and_named_back_as_written(tmp_path):
    """A name the model was never shown is refused and reported back as written."""
    mask = fp.InputNameMask()
    shown = [json.dumps({**FREE_VALID, "expression": "c*p1/(X1**2 + d)"}),  # the model wrote `p1` itself
             mask.mask_reply(json.dumps(FREE_VALID))]
    packet, constraints = state("free")
    client = fp.ScriptedClient(shown)
    proposer = fp.Fb1CatalogProposer(tmp_path / "records", client=client, structure_mode="free", name_mask=mask)
    candidates, record = proposer.propose_one(fp.serialize_packet(mask.mask_value(packet)),
                                              context=CONTEXT, constraints=constraints)
    assert len(candidates) == 1 and record["repairs"] == 1 and len(client.bodies) == 2
    assert record["rejected_candidates"][0]["reason"] == "unknown input name p1"
    repair = client.bodies[1]["messages"]
    assert repair[1] == {"role": "assistant", "content": shown[0]}
    # The only input name in the engine's turns is the token the model wrote, reported as unknown.
    assert real_name_tokens(repair[0]["content"]) == [] and real_name_tokens(repair[2]["content"]) == ["p1"]


@pytest.mark.parametrize("mode", ["catalog", "free"])
def test_the_built_in_mock_answers_in_the_names_it_is_shown(tmp_path, mode):
    """The built-in mock answers in the names it is shown."""
    packet, constraints = state(mode)
    plain = fp.Fb1CatalogProposer(tmp_path / "plain", structure_mode=mode)
    plain_candidates, plain_record = plain.propose_one(fp.serialize_packet(packet), context=CONTEXT,
                                                       constraints=constraints)
    mask = fp.InputNameMask()
    masked = fp.Fb1CatalogProposer(tmp_path / "masked", structure_mode=mode, name_mask=mask)
    candidates, record = masked.propose_one(fp.serialize_packet(mask.mask_value(packet)), context=CONTEXT,
                                            constraints=constraints)
    assert masked.transport == plain.transport == "mock" and candidates == plain_candidates
    assert real_name_tokens(plain_record["responses"][0]["raw_text"])
    assert real_name_tokens(record["responses"][0]["raw_text"]) == []


# ------------------------------------------------------------------ the runner's options
def runner(tmp_path, extra, suffix=""):
    """Import the runner with the LLM proposer in a child interpreter and return the process."""
    return native_import(tmp_path, {"FB1_PROPOSER": "llm", **extra}, suffix=suffix)


def test_the_runner_reads_both_options_and_records_nothing_when_they_are_off(tmp_path):
    """The runner reads both options and reports defaults when they are off."""
    variant = tmp_path / "variant.txt"
    variant.write_text(SYNTHETIC_VARIANT)
    show = "print(json.dumps([r.PROMPT_VARIANT, r.MASK_INPUT_NAMES, r.fbp.prompt_identity()['prompt_version']]))"
    off = runner(tmp_path, {}, show)
    assert off.returncode == 0, off.stderr[-2000:]
    assert json.loads(off.stdout.splitlines()[0]) == [None, False, "v2.0"]
    on = runner(tmp_path, {"FB1_PROMPT_VARIANT": str(variant), "FB1_MASK_INPUT_NAMES": "1"}, show)
    assert on.returncode == 0, on.stderr[-2000:]
    assert json.loads(on.stdout.splitlines()[0]) == [str(variant), True, "vT.synthetic"]


@pytest.mark.parametrize(("extra", "message"), [
    ({"FB1_MASK_INPUT_NAMES": "yes"}, "FB1_MASK_INPUT_NAMES must be 0 or 1"),
    ({"FB1_MASK_INPUT_NAMES": "1", "FB1_PROPOSER": "mock"}, "require FB1_PROPOSER=llm"),
    ({"FB1_PROMPT_VARIANT": "variant.txt", "FB1_PROPOSER": "mock"}, "require FB1_PROPOSER=llm"),
    ({"FB1_PROMPT_VARIANT": "untagged.txt"}, "must declare its tag on line 1"),
])
def test_the_runner_refuses_a_bad_option_before_any_output(tmp_path, extra, message):
    """The runner refuses a bad option value before any output."""
    (tmp_path / "variant.txt").write_text(SYNTHETIC_VARIANT)
    (tmp_path / "untagged.txt").write_text("Propose one model.\n")
    extra = {key: str(tmp_path / value) if key == "FB1_PROMPT_VARIANT" else value for key, value in extra.items()}
    proc = runner(tmp_path, extra)
    assert proc.returncode != 0 and message in proc.stderr


