"""ChemBench stage slots are a provenance structure only.

Each slot file is a byte-identical copy of the constant it replaces, so a ChemBench request is
the same bytes with or without slots; the slot names and digests land in config.json and the
transport records and never in a request; CORE and BoxingGym pass no slots and are unchanged.
Synthetic replies through injected clients only - no network, no saved run.
"""

import ast
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import chembench_core
from . import count_proposer as cp
from .count_models import PRIOR_TEMPLATES

PROMPTS = cp.PROMPT_DIRECTORY
#: The digests of this edition's slot files (= the constants' UTF-8 bytes) and of the
#: pool_generator_v1 text, fixed by a project decision.
EXPECTED_SHA256 = {
    "slot_burst_paragraph_v1.txt": "3155653c9968352ba8f5758ba39ac6a75e20b466139543cc9a1a5dd377638ba2",
    "slot_burst_refill_paragraph_v1.txt": "7581cd50c981e9ae83e8171e6b98c63f3fde88440ca96a5136af12446225ff82",
    "slot_repair_v1.txt": "8a31a92c58f35c61e566eda7f83f1e9c27a2e7bb4fbe822644a69f62e6f937af",
    "slot_repair_v1_empty.txt": "f6d969239867cffc98b06b165d0e15028caefddd415273ef6bc2ecc5d44c5260",
    "slot_fixed_role_prior_v1.json": "82a432fe3f7615ca5d1f76bbda27cc70d81c0801cc94f28397ca388aa27de984",
    "pool_generator_v1.txt": "a4466c0f6a54e0380f630481f6501802e5a75a0e981d290f310a64834eecc1a4",
}
#: The eight ChemBench arm files (fixed by a project decision) - bytes unchanged by the stage-slot change.
ARM_SHA256 = {
    "proposer_system_v2_0.txt": "f095edf1adb3b6533a9e68e73f98ca1dee3f23cd077e424b092393cb32b93aef",
    "proposer_system_v2_6_1.txt": "ba5faab3136111b9523e9ac4b01bc6b3a26640aa139ba781abdbc38544ec2961",
    "proposer_system_v2_6_2.txt": "13bf1390990aa0036384d8d8f94341201b11a0f613b0be576efd6182922e8ed3",
    "proposer_system_v2_6_3.txt": "843ee2a64774ae935fd49fc75dd9b2bfb9768cb8d1c21794323880b6608622ad",
    "proposer_system_v2_6_4.txt": "81380b6e5a1efb8570cb4fd1137df4d2e494fd5a4a2bba0c954ce5c827e18fb2",
    "proposer_system_v2_6_5.txt": "c21da86c44afb826f96dde109451ba9ce397325ac9be8df96b4042348b57170e",
    "proposer_system_v2_6_6.txt": "e7cb43b039769ac91751806e0675de547ab2ca9e4d62c9e56c661e0f9a5a7d6c",
    "proposer_system_v2_6_7_1.txt": "5d21ea8534d6aeae3f8f97caedc3ad1d60bc5dc5565caa6c19a535fc9667f207",
}
#: Strings that would mean slot metadata leaked into a request.
LEAK_MARKERS = ("stage_slot", "origin_stage", "turn_type", "repair_slot", "repair_v1",
                "prior_slot", "fixed_role_prior", "pool_generator", "in_cell_arm_burst",
                "role_prior_contract", "expand_slot", "init_slot")


def sha(data):
    """Return the hex SHA-256 digest of the given bytes."""
    return hashlib.sha256(data).hexdigest()


def slots():
    """Build the ChemBench stage slots and their record for the current prompt identity."""
    return chembench_core._chembench_stage_slots(cp.prompt_identity(), init_slot="in_cell_arm_burst")


# ------------------------------------------------------------------ files are the constants


@pytest.mark.parametrize("name,constant", [
    ("slot_burst_paragraph_v1.txt", "BURST_PARAGRAPH"),
    ("slot_burst_refill_paragraph_v1.txt", "BURST_REFILL_PARAGRAPH"),
    ("slot_repair_v1.txt", "REPAIR_INSTRUCTION"),
    ("slot_repair_v1_empty.txt", "REPAIR_INSTRUCTION_EMPTY"),
])
def test_each_slot_file_is_the_exact_bytes_of_its_constant(name, constant):
    """Test that each slot file is the exact bytes of its constant."""
    raw = (PROMPTS / name).read_bytes()
    assert raw == getattr(cp, constant).encode("utf-8")
    assert sha(raw) == EXPECTED_SHA256[name]


def test_the_role_prior_contract_names_the_engine_role_priors():
    """Test that the role prior contract names the engine role priors."""
    raw = (PROMPTS / "slot_fixed_role_prior_v1.json").read_bytes()
    assert sha(raw) == EXPECTED_SHA256["slot_fixed_role_prior_v1.json"]
    contract = json.loads(raw)
    assert contract["contract"] == "fixed_role_prior_v1" and contract["prior_template"] == "default"
    assert contract["coefficient_roles"] == {
        role: list(PRIOR_TEMPLATES["default"][role]) for role in ("intercept", "signed", "positive", "shape")}


def test_the_pool_generator_text_is_the_base_prompt_without_its_search_goal_sentence():
    """Test that the pool generator text is the base prompt without its search goal sentence."""
    generator = (PROMPTS / "pool_generator_v1.txt").read_bytes()
    assert sha(generator) == EXPECTED_SHA256["pool_generator_v1.txt"]
    base = (PROMPTS / "proposer_system_v2_0.txt").read_bytes().decode("utf-8").splitlines(keepends=True)
    assert generator.decode("utf-8") == "".join(base[:4] + base[6:])
    assert "what the pool misses" not in generator.decode("utf-8")


def test_the_eight_arm_files_are_unchanged():
    """Test that the eight arm files are unchanged."""
    for name, digest in ARM_SHA256.items():
        assert sha((PROMPTS / name).read_bytes()) == digest, name


def test_the_slot_record_binds_every_file_and_the_arm_identity():
    """Test that the slot record binds every file and the arm identity."""
    stage, record = slots()
    identity = cp.prompt_identity()
    assert record == {
        "stage_slot_schema": "chembench_stage_slots_v1",
        "init_slot": "in_cell_arm_burst",
        "burst_paragraph_sha256": EXPECTED_SHA256["slot_burst_paragraph_v1.txt"],
        "burst_refill_paragraph_sha256": EXPECTED_SHA256["slot_burst_refill_paragraph_v1.txt"],
        "expand_slot": identity["prompt_version"],
        "expand_prompt_sha256": identity["prompt_sha256"],
        "repair_slot": "repair_v1",
        "repair_prompt_sha256": EXPECTED_SHA256["slot_repair_v1.txt"],
        "repair_empty_prompt_sha256": EXPECTED_SHA256["slot_repair_v1_empty.txt"],
        "prior_slot": "none",
        "role_prior_contract": "fixed_role_prior_v1",
        "role_prior_contract_sha256": EXPECTED_SHA256["slot_fixed_role_prior_v1.json"],
    }
    assert stage.metadata == record


# ------------------------------------------------------------------ composed text parity


@pytest.mark.parametrize("burst", [True, "refill"])
def test_a_burst_request_composed_from_slots_is_the_same_text_and_identity(burst):
    """Test that a burst request composed from slots is the same text and identity."""
    stage, _ = slots()
    assert cp.composed_system_prompt(burst=burst, burst_paragraphs=stage.burst_paragraphs()) == \
        cp.composed_system_prompt(burst=burst)


# ------------------------------------------------------------------ requests through proposers

CANDIDATE = {"expression": "b0 + b1*z", "family": "binomial",
             "parameters": {"b0": "intercept", "b1": "signed"}}


def openai_injected(directory, texts):
    """Build an OpenAI proposer over an injected fake client that replies with the given texts in order."""
    calls, queue, lock = [], list(texts), threading.Lock()

    def create(**body):
        with lock:
            calls.append(json.loads(json.dumps(body)))
            text = queue.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
            id="test-only-response", model="returned-test-model")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return cp.OpenAICountProposer(directory, client=client), calls


def anthropic_injected(directory, texts):
    """Build an Anthropic proposer over an injected fake client that replies with the given texts in order."""
    calls, queue, lock = [], list(texts), threading.Lock()

    def create(**body):
        with lock:
            calls.append(json.loads(json.dumps(body)))
            text = queue.pop(0)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 12, "output_tokens": 3}),
            id="test-only-response", model="returned-test-model", stop_reason="end_turn")

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return cp.AnthropicCountProposer(directory, client=client), calls


VALID = json.dumps(CANDIDATE)
SCENARIOS = {
    "first_reply_valid": [VALID],
    "validation_repair": ["not a json object", VALID],
    "empty_body_repair": ["", VALID],
}
ROUND_CONTEXT = {"run_id": "test-only-run", "world_seed": 11, "cell": "100", "proposal_rep": 0,
                 "round": 1}
BURST_CONTEXT = {**ROUND_CONTEXT, "round": 0, "burst_call_index": 0, "burst_wave": 0}


def call(build, directory, scenario, burst, stage):
    """Run one propose_one call for a scenario and return the models, the record and the captured requests."""
    proposer, calls = build(directory, SCENARIOS[scenario])
    context = BURST_CONTEXT if burst else ROUND_CONTEXT
    models, record = proposer.propose_one(
        "packet", context=context, initial_family="binomial", burst=burst,
        **({} if stage is None else {"slots": stage}))
    return models, record, calls


@pytest.mark.parametrize("build", [openai_injected, anthropic_injected])
@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("burst", [False, True])
def test_requests_are_byte_identical_with_and_without_slots(tmp_path, build, scenario, burst):
    """Test that requests are byte identical with and without slots."""
    stage, record_block = slots()
    plain_models, plain, plain_calls = call(build, tmp_path / "plain", scenario, burst, None)
    slot_models, slotted, slot_calls = call(build, tmp_path / "slots", scenario, burst, stage)
    # Every request the provider saw is the same, byte for byte (system, messages, repair turn).
    assert json.dumps(slot_calls, sort_keys=True) == json.dumps(plain_calls, sort_keys=True)
    assert [m.expression for m in slot_models] == [m.expression for m in plain_models]
    assert len(slot_calls) == (1 if scenario == "first_reply_valid" else 2)
    wire = json.dumps(slot_calls)
    for marker in LEAK_MARKERS:
        assert marker not in wire, marker
    # The record gains exactly the slot block, the origin stage and the per-turn labels.
    assert {k: slotted[k] for k in record_block} == record_block
    assert slotted["origin_stage"] == ("init" if burst else "expand")
    assert [r["turn_type"] for r in slotted["responses"]] == (
        ["proposal"] if scenario == "first_reply_valid" else ["proposal", "repair"])
    repaired = [r for r in slotted["rejected_candidates"] if "repair_kind" in r]
    if scenario == "first_reply_valid":
        assert repaired == []
    else:
        assert [r["repair_text_kind"] for r in repaired] == [
            "empty_body" if scenario == "empty_body_repair" else "validation"]
    extra = set(slotted) - set(plain)
    assert extra == set(record_block) | {"origin_stage"}
    assert all("turn_type" not in r for r in plain["responses"])
    assert all("repair_text_kind" not in r for r in plain["rejected_candidates"])


# ------------------------------------------------------------------ runner wiring


def _propose_one_calls(filename):
    """Return the `propose_one` call nodes found in the named source file next to chembench_core."""
    tree = ast.parse((Path(chembench_core.__file__).parent / filename).read_text())
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "propose_one"]


def test_both_chembench_proposer_calls_pass_the_run_slots_and_no_other_runner_does():
    """Test that both chembench proposer calls pass the run slots and no other runner does."""
    calls = _propose_one_calls("chembench_core.py")
    assert len(calls) == 2
    for node in calls:
        values = [k.value for k in node.keywords if k.arg == "slots"]
        assert len(values) == 1 and ast.unparse(values[0]) == "stage_slots"
    for filename in ("core_count.py", "boxinggym_core.py"):
        assert all(k.arg != "slots" for node in _propose_one_calls(filename) for k in node.keywords)


def test_a_chembench_run_records_its_slots_and_no_packet_carries_them(tmp_path, monkeypatch):
    """Test that a chembench run records its slots and no packet carries them."""
    from .test_packet_constraints_explicit import packet_files, run_chembench

    out = run_chembench(tmp_path / "chem", tmp_path, monkeypatch)
    config = json.loads((out / "config.json").read_text())
    stage = config["stage_slots"]
    expected_init = "in_cell_arm_burst" if config.get("initial_proposals") else "catalogue"
    _, record = chembench_core._chembench_stage_slots(cp.prompt_identity(), init_slot=expected_init)
    assert stage == record
    for path in packet_files(out):
        text = path.read_text()
        for marker in LEAK_MARKERS:
            assert marker not in text, (path.name, marker)


def test_core_and_boxinggym_configs_carry_no_stage_slots(tmp_path, monkeypatch):
    """Test that core and boxinggym configs carry no stage slots."""
    from .test_named_inputs import bg_run_args, bg_scalar_stub_fits
    from .test_packet_constraints_explicit import run_core

    core_out = run_core(tmp_path / "core", monkeypatch)
    assert "stage_slots" not in json.loads((core_out / "config.json").read_text())
    bg = bg_scalar_stub_fits(monkeypatch)
    task = next(iter(bg.TASKS))
    pytest.importorskip(bg.TASKS[task].module)
    bg_out = tmp_path / "bg"
    bg.run(bg_run_args(bg_out, task=task, cell="100", rounds=1))
    assert "stage_slots" not in json.loads((bg_out / "config.json").read_text())
