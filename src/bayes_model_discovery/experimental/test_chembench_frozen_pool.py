"""ChemBench frozen initial pools, tested offline.

A `--pool-generator` run makes one pool file (the round-0 burst alone, arm-neutral generator
text); a registered cell reads it with `--initial-pool-file` and fits its members at round 0
with no LLM call. Real AutoSciLab simulator (free, local) with the sequential Monte Carlo (SMC) fit stubbed, synthetic
replies through injected clients only - no network, no saved run, no paid call.
"""

import copy
import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from . import chembench_core as cc
from . import count_proposer as cp
from .count_inference import Fit
from .test_chembench_core import _args
from .test_stage_slots_chembench import LEAK_MARKERS

pytestmark = pytest.mark.usefixtures("require_autoscilab_source")


@pytest.fixture(autouse=True)
def _default_system_text_after_each_test():
    """A run here selects the generator or an arm file in this interpreter; every later test
    reads the default system text again, whatever this one ran."""
    from bayes_model_discovery.experimental import count_proposer

    yield
    count_proposer.select_pool_generator(None)
    count_proposer.set_prompt_variant(None)

REPO = Path(cc.__file__).resolve().parents[3]
DEV_MANIFEST = REPO / "manifests" / "chembench_dev_manifest.json"
#: The fixed generator text (the packaged file's digest).
GENERATOR_SHA256 = "a4466c0f6a54e0380f630481f6501802e5a75a0e981d290f310a64834eecc1a4"
#: The pool receipt fields and the init-prompt digests the project decision required.
POOL_RECEIPT_FIELDS = (
    "pool_id", "task", "stratum", "seed", "generator_prompt_sha256", "generator_code_commit",
    "packet_constraints_sha256", "pool_file_sha256", "pool_size", "provider", "model", "host",
    "generation_started_at", "generation_finished_at", "paid_calls", "paid_usd",
    "pool_generator_sha256", "burst_paragraph_sha256", "burst_refill_paragraph_sha256",
    "coefficient_name_rule_sha256", "prior_probe_rule_sha256", "composed_init_prompt_sha256")
#: The registered-cell fields the project decision required.
CELL_FIELDS = ("pool_id", "pool_file_sha256", "initial_pool_source", "initial_pool_generation",
               "round0_llm_calls_inside_cell", "arm_type", "prompt_version")
CANDIDATE = {"expression": "b0 + b1*C_Az", "family": "lognormal",
             "parameters": {"b0": "intercept", "b1": "signed"}}


def stub_fit(patch, calls=None):
    """A seed-determined stand-in for the SMC fit; `calls` records each fit's inputs."""
    def fit(hypothesis, x, y, seed, transform, response_unit, settings, **kwargs):
        """Seed-determined stand-in for the SMC fit; records its inputs in `calls` when given."""
        if calls is not None:
            calls.append({"model_id": hypothesis.model_id, "seed": int(seed),
                          "x": np.asarray(x, dtype=float).tobytes(),
                          "y": np.asarray(y, dtype=float).tobytes(),
                          "response_unit": float(response_unit)})
        rng = np.random.default_rng(int(seed) % 2 ** 32)
        particles = {name: rng.normal(0.0, 0.05, 16) for name, _ in hypothesis.parameters}
        particles.update({name: np.full(16, 0.4 if name == "sigma" else 2.0)
                          for name in hypothesis.noise_parameters})
        return Fit(hypothesis, particles, -float(len(hypothesis.parameters)), [-1.0, -1.0],
                   {"input_transform": {"center": transform.center, "scale": transform.scale},
                    "interface": kwargs.get("interface"), "response_unit": float(response_unit)},
                   0.0)
    patch.setattr(cc, "fit", fit)


class InjectedOpenAI:
    """A thread-safe synthetic chat client: `reply(index, body)` gives each reply's text."""

    def __init__(self, reply):
        self.reply, self.calls, self.lock = reply, [], threading.Lock()
        self.chat = SimpleNamespace(completions=self)

    def create(self, **body):
        """Return a synthetic chat completion whose text comes from `reply`; record the request body."""
        with self.lock:
            index = len(self.calls)
            self.calls.append(copy.deepcopy(body))
        return SimpleNamespace(
            model="synthetic_returned_model", id="synthetic_response_id",
            _request_id="synthetic_http_id",
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply(index, body)),
                                     finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7))


def inject(patch, reply):
    """Patch the proposer with an injected synthetic chat client; return the client and the list of rounds proposed."""
    client = InjectedOpenAI(reply)
    original = cp.OpenAICountProposer
    # The run's own paid-call cap reaches the proposer; None when none is stated.
    patch.setattr(cp, "OpenAICountProposer",
                  lambda records, **kwargs: original(
                      records, client=client, max_paid_calls=kwargs.get("max_paid_calls")))
    rounds = []
    propose_one = original.propose_one
    patch.setattr(original, "propose_one",
                  lambda self, packet_text, **kwargs: (
                      rounds.append(kwargs["context"]["round"]),
                      propose_one(self, packet_text, **kwargs))[1])
    return client, rounds


def run_chem(out, **overrides):
    """Run one mock ChemBench cell into `out` with default options overridden by `overrides`; return `out`."""
    options = {"cell": "100", "seed": 31, "rounds": 1, "particles": 16, "q": 8,
               "structure_proposals": "free", "proposer": "mock", "d0_policy": "fixed3"}
    options.update(overrides)
    cc.run(_args(out, DEV_MANIFEST, **options))
    return out


def generate(out, **overrides):
    """Run a pool-generation cell (round 0 only) into `out`; return `out`."""
    return run_chem(out, initial_proposals=10, initial_catalogue="off",
                    pool_generator="pool_generator_v1", **overrides)


def pool_cell(out, pool, **overrides):
    """Run a registered cell that reads the frozen pool file `pool`; return `out`."""
    return run_chem(out, initial_catalogue="off", initial_pool_file=str(pool), **overrides)


def read(path):
    """Read and parse a JSON file."""
    return json.loads(Path(path).read_text())


def sha(data):
    """Return the SHA-256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def mock_pool(tmp_path, monkeypatch):
    """Fixture: generate one frozen pool with the SMC fit stubbed and return the pool file path."""
    stub_fit(monkeypatch)
    return generate(tmp_path / "generation") / "proposals" / "t00_burst.json"


def reader_arguments(**overrides):
    """Build the frozen-pool interface and the reader arguments for cell 100; return both."""
    interface = cc.frozen_pool_interface()
    names = cc.interface_input_names(interface)
    task = cc.select_task(read(DEV_MANIFEST), None)
    arguments = {"task": task, "seed": 31, "cell": "100", "family": "lognormal",
                 "d0_policy": "fixed3", "d0_fixed_inputs": None, "input_names": names}
    arguments.update(overrides)
    return interface, arguments


def read_pool(path, **overrides):
    """Read a frozen pool file through the reader under the default grammar and restore the grammar afterwards."""
    interface, arguments = reader_arguments(**overrides)
    previous = cc.set_grammar_version(cc.DEFAULT_GRAMMAR_VERSION, interface)
    try:
        return cc.read_chembench_initial_pool_file(path, **arguments)
    finally:
        from .count_models import restore_grammar_state

        restore_grammar_state(previous)


# ------------------------------------------------------------------ the generation run


def test_the_generation_run_writes_one_pool_file_and_stops_after_round_zero(mock_pool):
    """The generation run writes one pool file with its receipt and stops after round 0."""
    out = mock_pool.parents[1]
    burst = read(mock_pool)
    block = burst["frozen_pool"]
    assert block["init_slot"] == "pool_generator_v1"
    assert block["generator_prompt_sha256"] == GENERATOR_SHA256
    assert block["pool_id"] == "chembench_frozen_pool_v1__c0_michaelis_menten__easy__s31"
    assert block["constraints_disclosure"] == cc.CONSTRAINTS_DISCLOSURE == "cic3_v1"
    assert block["stage_slot_schema"] == cc.STAGE_SLOT_SCHEMA
    # Round 0 only: no later proposal, no prediction, no acquisition, no finish.
    assert sorted(p.name for p in (out / "proposals").iterdir()) == ["t00_burst.json"]
    assert not any((out / "predictions").iterdir())
    assert not (out / "scores_after_finish.json").exists()
    summary = read(out / "summary.json")
    assert summary["status"] == "initial_pool_generated"
    assert summary["pool_id"] == block["pool_id"]
    receipt = read(out / "initial_pool_generation_receipt.json")
    assert set(POOL_RECEIPT_FIELDS) <= set(receipt)
    assert receipt["pool_file_sha256"] == sha(mock_pool.read_bytes())
    assert receipt["pool_size"] == len(burst["after_ids"]) == summary["initial_pool_count"]
    assert receipt["generator_prompt_sha256"] == receipt["pool_generator_sha256"] == GENERATOR_SHA256
    assert receipt["paid_usd"] is None and receipt["paid_usd_source"] == "driver_cost_ledger"
    assert receipt["generator_code_commit"] is None
    assert receipt["generator_code_commit_source"] == "absent"
    constraints = read(out / "packets" / "t00_burst.json")["constraints"]
    assert receipt["packet_constraints_sha256"] == cc._canonical_sha256(constraints)
    assert receipt["coefficient_name_rule_sha256"] == cc._canonical_sha256(
        constraints["coefficient_name_rule"])
    assert receipt["prior_probe_rule_sha256"] == cc._canonical_sha256(constraints["prior_probe_rule"])
    config = read(out / "config.json")
    assert config["prompt_version"] == "pool_generator_v1"
    assert config["prompt_sha256"] == GENERATOR_SHA256
    assert config["stage_slots"]["init_slot"] == "pool_generator_v1"
    assert config["stage_slots"]["init_prompt_sha256"] == GENERATOR_SHA256
    assert config["stage_slots"]["expand_slot"] is None
    assert config["stage_slots"]["prior_slot"] == "none"
    assert config["frozen_pool_generation"] == block


def test_the_generation_requests_send_the_generator_file_as_stored(tmp_path, monkeypatch):
    """Generation requests send the generator file as stored, with one repair turn and no leaked markers."""
    stub_fit(monkeypatch)
    # Every call answers the same candidate, so wave 0 admits one and both refill waves run;
    # the first call's reply fails validation once, so one init-stage repair turn is sent.
    client, rounds = inject(monkeypatch, lambda index, body: (
        "not a json object" if index == 0 else json.dumps(CANDIDATE)))
    out = generate(tmp_path / "generation", proposer="openai", allow_paid_api=True)
    generator = (cp.PROMPT_DIRECTORY / "pool_generator_v1.txt").read_text()
    systems = {call["messages"][0]["content"] for call in client.calls}
    assert systems == {generator + cp.BURST_PARAGRAPH, generator + cp.BURST_REFILL_PARAGRAPH}
    assert set(rounds) == {0}
    wire = json.dumps(client.calls)
    for marker in (*LEAK_MARKERS, "prompt_version"):
        assert marker not in wire, marker
    burst = read(out / "proposals" / "t00_burst.json")
    assert len(burst["waves"]) == 3
    records = [row["transport"] for row in burst["transports"]]
    assert {record["origin_stage"] for record in records} == {"init"}
    assert {record["prompt_version"] for record in records} == {"pool_generator_v1"}
    turns = [response["turn_type"] for record in records for response in record["responses"]]
    assert turns.count("repair") == 1
    receipt = read(out / "initial_pool_generation_receipt.json")
    assert receipt["provider"] == "openai" and receipt["paid_calls"] == len(client.calls)
    assert receipt["returned_models"] == ["synthetic_returned_model"]


def test_a_generation_run_stops_at_its_max_paid_calls_and_leaves_no_pool(tmp_path, monkeypatch):
    """A generation run stops at its paid-call cap and leaves no pool file."""
    stub_fit(monkeypatch)
    client, _ = inject(monkeypatch, lambda index, body: json.dumps(CANDIDATE))
    out = tmp_path / "generation"
    # Wave 0 alone asks for ten calls; the guard refuses the fourth before it is sent.
    with pytest.raises(cp.PaidCallBudgetExhausted):
        generate(out, proposer="openai", allow_paid_api=True, max_paid_calls=3)
    assert len(client.calls) == 3
    summary = read(out / "summary.json")
    assert {key: summary[key] for key in ("status", "stop_reason", "paid_api_calls",
                                          "max_paid_calls", "stopped_round",
                                          "last_completed_round")} == {
        "status": "budget_exhausted", "stop_reason": "budget_exhausted", "paid_api_calls": 3,
        "max_paid_calls": 3, "stopped_round": 0, "last_completed_round": None}
    assert not (out / "proposals" / "t00_burst.json").exists()
    assert not (out / "initial_pool_generation_receipt.json").exists()
    # The generator selection ended with the run, as it does after a finished generation.
    assert cp.prompt_identity() == {"prompt_version": cp.PROMPT_VERSION,
                                    "prompt_sha256": sha(cp.PROMPT_FILE.read_bytes())}


@pytest.mark.parametrize("overrides,message", [
    ({"prompt_version": "v2.6.2"}, "excludes --prompt-version"),
    ({"prompt_variant": "synthetic.txt"}, "excludes --prompt-version"),
    ({"priors": "llm"}, "excludes --prompt-version"),
    ({"initial_pool_file": "synthetic.json"}, "cannot also read one"),
    ({"initial_catalogue": "on"}, "needs --initial-catalogue off"),
    ({"d0_policy": "random4"}, "requires --d0-policy fixed3"),
])
def test_the_generation_option_refuses_every_other_system_text_before_any_output(
        tmp_path, overrides, message):
    """Generation option combinations that select another system text are refused before any output."""
    options = {"initial_proposals": 10, "initial_catalogue": "off",
               "pool_generator": "pool_generator_v1", **overrides}
    with pytest.raises(ValueError, match=message):
        run_chem(tmp_path / "refused", **options)
    assert not (tmp_path / "refused").exists()


def test_no_slot_or_generator_selection_can_switch_on_a4_priors_or_another_text():
    """Neither a slot nor a generator selection can switch on LLM priors or another text."""
    with pytest.raises(ValueError, match="unknown pool generator"):
        cp.select_pool_generator("v2.2")
    assert set(cp.POOL_GENERATOR_FILES) == {"pool_generator_v1"}
    try:
        cp.select_pool_generator("pool_generator_v1")
        assert cp.prompt_identity() == {"prompt_version": "pool_generator_v1",
                                        "prompt_sha256": GENERATOR_SHA256}
        # Every runner's prompt selection clears a generator an earlier run left behind.
        _, identity = cp.select_run_prompt(SimpleNamespace())
        assert identity == {"prompt_version": cp.PROMPT_VERSION,
                            "prompt_sha256": sha(cp.PROMPT_FILE.read_bytes())}
    finally:
        cp.select_pool_generator(None)


# ------------------------------------------------------------------ the registered cell


def test_a_pool_file_cell_fits_the_members_at_round_zero_without_any_llm_call(
        tmp_path, monkeypatch, mock_pool):
    """A pool-file cell fits the pool members at round 0 without any LLM call."""
    fits = []
    stub_fit(monkeypatch, fits)
    client, rounds = inject(monkeypatch, lambda index, body: json.dumps(CANDIDATE))
    out = pool_cell(tmp_path / "cell", mock_pool, proposer="openai", allow_paid_api=True)
    pool = read(mock_pool)
    # No proposer call at round 0; the round-1 call sends the arm file itself (expand slot).
    assert rounds == [1]
    assert [call["messages"][0]["content"] for call in client.calls] == [
        cp.PROMPT_FILE.read_text()]
    transport = sorted(p.name for p in (out / "llm_transport").iterdir())
    assert transport and not any(name.startswith("round_00") for name in transport)
    assert not (out / "proposals" / "t00_burst.json").exists()
    assert not any(p.name.startswith("t00") for p in (out / "packets").iterdir())
    round0 = read(out / "proposals" / "t00_initial_pool.json")
    assert round0["file_after_ids"] == pool["after_ids"] == round0["after_ids"]
    assert [event["status"] for event in round0["events"]] == ["fitted"] * len(pool["after_ids"])
    assert {event["source"] for event in round0["events"]} == {"initial_pool_file"}
    assert round0["round0_llm_calls_inside_cell"] == 0
    first = [row for row in fits if row["model_id"] in pool["after_ids"]][:len(pool["after_ids"])]
    assert [row["seed"] for row in first] == [
        cc.seed_for("chem_smc_proposed", 31, 0, model_id) for model_id in pool["after_ids"]]
    config = read(out / "config.json")
    assert {key: config[key] for key in CELL_FIELDS} == {
        "pool_id": pool["frozen_pool"]["pool_id"], "pool_file_sha256": sha(mock_pool.read_bytes()),
        "initial_pool_source": "file_burst", "initial_pool_generation": "external_frozen_stage",
        "round0_llm_calls_inside_cell": 0, "arm_type": "random_baseline",
        "prompt_version": cp.PROMPT_VERSION}
    assert config["initial_pool_count"] == config["initial_pool"] == len(pool["after_ids"])
    receipt_bytes = (out / "initial_pool_receipt.json").read_bytes()
    assert config["initial_pool_receipt"]["sha256"] == sha(receipt_bytes)
    assert read(out / "initial_pool_receipt.json")["member_ids"] == pool["after_ids"]
    settings = config["label_receipt"]["binding"]["settings"]
    assert {key: settings[key] for key in ("initial_pool_source", "initial_pool_sha256",
                                            "initial_pool_count", "initial_catalogue")} == {
        "initial_pool_source": "file_burst", "initial_pool_sha256": sha(mock_pool.read_bytes()),
        "initial_pool_count": len(pool["after_ids"]), "initial_catalogue": "off"}
    assert "initial_proposals" not in settings and "burst_prompt_sha256" not in settings
    stage = config["stage_slots"]
    assert stage["init_slot"] == "pool_generator_v1"
    assert stage["init_prompt_sha256"] == GENERATOR_SHA256
    assert stage["expand_slot"] == cp.PROMPT_VERSION
    record = read(out / "proposals" / "t01.json")["transport"]
    assert record["origin_stage"] == "expand"
    for path in (out / "packets").iterdir():
        text = path.read_text()
        for marker in LEAK_MARKERS:
            assert marker not in text, (path.name, marker)


def test_a_round_one_repair_turn_keeps_its_expand_origin(tmp_path, monkeypatch, mock_pool):
    """A round-1 repair turn keeps its expand-stage origin."""
    stub_fit(monkeypatch)
    inject(monkeypatch, lambda index, body: "not a json object" if index == 0 else json.dumps(CANDIDATE))
    out = pool_cell(tmp_path / "cell", mock_pool, proposer="openai", allow_paid_api=True)
    record = read(out / "proposals" / "t01.json")["transport"]
    assert record["origin_stage"] == "expand"
    assert [response["turn_type"] for response in record["responses"]] == ["proposal", "repair"]
    assert [row["repair_text_kind"] for row in record["rejected_candidates"]
            if "repair_kind" in row] == ["validation"]


def test_two_arms_and_both_cells_start_from_the_same_round_zero_pool(tmp_path, monkeypatch, mock_pool):
    """Two arms and both cells start from the same round-0 pool and fit inputs."""
    runs = {}
    for name, overrides in {
        "A0_101": {"cell": "101", "design_search": "ascent"},
        "A2_101": {"cell": "101", "design_search": "ascent", "prompt_version": "v2.6.2"},
        "A0_100": {"cell": "100"},
    }.items():
        fits = []
        with monkeypatch.context() as patch:
            stub_fit(patch, fits)
            out = pool_cell(tmp_path / name, mock_pool, **overrides)
        round0 = read(out / "proposals" / "t00_initial_pool.json")["events"]
        runs[name] = {
            "pool_sha": read(out / "config.json")["pool_file_sha256"],
            "ids": [event["model_id"] for event in round0],
            "inputs": fits[:len(round0)],
            "fit_files": {p.name: p.read_bytes() for p in (out / "fits").glob("t00_*")},
            "arm_type": read(out / "config.json")["arm_type"],
            "prompt": read(out / "config.json")["prompt_version"],
        }
    first = runs["A0_101"]
    for name, run in runs.items():
        assert run["pool_sha"] == first["pool_sha"] == sha(mock_pool.read_bytes()), name
        assert run["ids"] == first["ids"], name
        assert run["inputs"] == first["inputs"], name
        assert run["fit_files"] == first["fit_files"], name
    assert [runs[name]["prompt"] for name in runs] == ["v2.0", "v2.6.2", "v2.0"]
    assert [runs[name]["arm_type"] for name in runs] == ["prompt", "prompt", "random_baseline"]


def test_a_pool_smaller_than_ten_is_used_as_it_is(tmp_path, monkeypatch, mock_pool):
    """A pool with fewer than ten members is used as it is."""
    stub_fit(monkeypatch)
    pool = read(mock_pool)
    assert 1 <= len(pool["after_ids"]) < 10
    out = pool_cell(tmp_path / "cell", mock_pool)
    summary = read(out / "summary.json")
    assert summary["proposal_accounting"]["initial_pool_size"] == len(pool["after_ids"])
    assert "burst_wave_count" not in summary["proposal_accounting"]
    assert read(out / "proposals" / "t00_initial_pool.json")["after_ids"] == pool["after_ids"]


def test_the_round_zero_fits_equal_the_generation_runs_own(tmp_path, monkeypatch, mock_pool):
    """The round-0 fits of a pool cell equal the generation run's own fits byte for byte."""
    stub_fit(monkeypatch)
    out = pool_cell(tmp_path / "cell", mock_pool)
    generation = mock_pool.parents[1] / "fits"
    for path in sorted(generation.glob("t00_*.npz")):
        assert (out / "fits" / path.name).read_bytes() == path.read_bytes(), path.name


def test_a_member_that_fails_to_fit_fails_the_cell_and_names_itself(tmp_path, monkeypatch, mock_pool):
    """A member that fails to fit fails the cell and the failure names that member."""
    stub_fit(monkeypatch)
    victim = read(mock_pool)["after_ids"][-1]
    stubbed = cc.fit

    def failing(hypothesis, *args, **kwargs):
        """Raise for the victim model and delegate to the stubbed fit otherwise."""
        if hypothesis.model_id == victim:
            raise ValueError("synthetic fit failure")
        return stubbed(hypothesis, *args, **kwargs)

    monkeypatch.setattr(cc, "fit", failing)
    with pytest.raises(cc.FrozenPoolMemberFitFailure):
        pool_cell(tmp_path / "cell", mock_pool)
    failure = read(tmp_path / "cell" / "failure.json")
    assert failure["frozen_pool_failed_model_id"] == victim
    assert victim in failure["error"]


def test_other_d0_observations_than_the_pools_are_refused(tmp_path, monkeypatch, mock_pool):
    """Different D0 observations than the pool's are refused."""
    stub_fit(monkeypatch)
    with pytest.raises(ValueError, match="public observations differ"):
        pool_cell(tmp_path / "cell", mock_pool, noise=0.02)


@pytest.mark.parametrize("overrides,message", [
    ({"initial_proposals": 10}, "requires --initial-catalogue off and no --initial-proposals"),
    ({"initial_catalogue": "on"}, "requires --initial-catalogue off and no --initial-proposals"),
])
def test_a_pool_file_refuses_a_burst_or_a_catalogue_before_any_output(
        tmp_path, mock_pool, overrides, message):
    """A pool-file cell refuses a burst size or a catalogue before any output."""
    with pytest.raises(ValueError, match=message):
        run_chem(tmp_path / "refused", initial_pool_file=str(mock_pool),
                 **{"initial_catalogue": "off", **overrides})
    assert not (tmp_path / "refused").exists()


# ------------------------------------------------------------------ the reader


def mutate(tmp_path, path, change):
    """Apply `change` to a copy of the JSON file at `path`, write it under `tmp_path`, and return the new path."""
    payload = read(path)
    change(payload)
    target = tmp_path / "mutated.json"
    target.write_text(json.dumps(payload))
    return target


def _first_fitted(payload):
    return next(event for event in payload["events"] if event["model_id"] == payload["after_ids"][0]
                and event["status"] == "fitted")


READER_REFUSALS = {
    "ordinary_burst": (lambda p: p.pop("frozen_pool"), "no frozen_pool block"),
    "other_init_slot": (lambda p: p["frozen_pool"].update(init_slot="in_cell_arm_burst"),
                        "init_slot must be 'pool_generator_v1'"),
    "other_generator_bytes": (lambda p: p["frozen_pool"].update(generator_prompt_sha256="0" * 64),
                              "generator_prompt_sha256"),
    "other_disclosure": (lambda p: p["frozen_pool"].update(constraints_disclosure="cic2_v1"),
                         "constraints_disclosure"),
    "other_slot_schema": (lambda p: p["frozen_pool"].update(stage_slot_schema="other"),
                          "stage_slot_schema"),
    "other_pool_id": (lambda p: p["frozen_pool"].update(pool_id="other"), "pool_id"),
    "refill_policy": (lambda p: p.update(initial_burst_refill_policy="other"),
                      "initial_burst_refill_policy"),
    "catalogue_on": (lambda p: p.update(initial_catalogue="on"), "initial_catalogue off"),
    "model_id_schema": (lambda p: p.update(model_id_schema="ast_roles_family_ac_v1"),
                        "model_id_schema mismatch"),
    "no_after_ids": (lambda p: p.update(after_ids=[]), "after_ids"),
    "two_fitted": (lambda p: p["events"].append(copy.deepcopy(_first_fitted(p))),
                   "exactly one fitted event"),
    "stray_status": (lambda p: p["events"].append({**copy.deepcopy(_first_fitted(p)),
                                                   "status": "rejected_prior_probe"}),
                     "non-duplicate events"),
    "unknown_input": (lambda p: _first_fitted(p).update(expression="b0 + b1*w_unknown"),
                      "unknown input name"),
    "other_family": (lambda p: _first_fitted(p).update(family="gamma"), "frozen pool entry 0"),
    "outside_family": (lambda p: _first_fitted(p).update(family="poisson"),
                       "is not a ChemBench family"),
    "id_mismatch": (lambda p: _first_fitted(p).update(expression="b0 + b1*C_Bz"),
                    "reconstructed model_id mismatch"),
    "bad_parameters": (lambda p: _first_fitted(p).update(parameters=["b0"]),
                       "requires expression/family/parameters"),
}


@pytest.mark.parametrize("case", sorted(READER_REFUSALS))
def test_the_reader_refuses_the_whole_file_on_any_bad_entry(tmp_path, mock_pool, case):
    """The reader refuses the whole file on any bad entry."""
    change, message = READER_REFUSALS[case]
    with pytest.raises(ValueError, match=message):
        read_pool(mutate(tmp_path, mock_pool, change))


@pytest.mark.parametrize("overrides,message", [
    ({"seed": 37}, "pool_id"),
    ({"d0_policy": "random4"}, "d0_policy"),
    ({"family": "gamma"}, "family"),
])
def test_the_reader_refuses_a_pool_made_for_another_cell(mock_pool, overrides, message):
    """The reader refuses a pool made for another cell."""
    with pytest.raises(ValueError, match=message):
        read_pool(mock_pool, **overrides)


def test_the_reader_accepts_the_generation_file_exactly_and_rebuilds_every_member(mock_pool):
    """The reader accepts the generation file exactly and rebuilds every member."""
    pool = read_pool(mock_pool)
    payload = read(mock_pool)
    assert [h.model_id for h in pool.hypotheses] == payload["after_ids"]
    assert pool.sha256 == sha(mock_pool.read_bytes())
    assert pool.metadata() == {"initial_pool_source": "file_burst",
                               "initial_pool_sha256": pool.sha256,
                               "initial_pool_count": len(payload["after_ids"])}
    assert pool.block == payload["frozen_pool"]


def test_the_frozen_pool_interface_is_the_runs_own(tmp_path, monkeypatch, mock_pool):
    """The frozen-pool interface is the one the run itself uses."""
    stub_fit(monkeypatch)
    captured = []
    real = cc.task_interface
    monkeypatch.setattr(cc, "task_interface", lambda *a, **k: captured.append(real(*a, **k)) or captured[-1])
    pool_cell(tmp_path / "cell", mock_pool)
    monkeypatch.setattr(cc, "task_interface", real)
    assert captured and captured[0] == cc.frozen_pool_interface()


# ------------------------------------------------------------------ binding and preflight


def test_a_changed_pool_file_is_refused_against_its_registered_digest(tmp_path, mock_pool):
    """A changed pool file is refused against its registered digest."""
    from .run_label_policy import bmd_runtime_settings, resolve_label
    from .test_run_label_policy import registration, runner_args

    binding = read_pool(mock_pool).metadata()
    registered = bmd_runtime_settings("chembench", runner_args(
        initial_catalogue="off", **binding))
    assert {key: registered[key] for key in binding} == binding
    ref, context, _ = registration(tmp_path, bundle="chembench", cell="100", settings=registered)
    resolve_label("production", manifest_ref=ref, **context)
    changed = mutate(tmp_path, mock_pool, lambda p: p.update(note="one more byte"))
    actual = bmd_runtime_settings("chembench", runner_args(
        initial_catalogue="off", **read_pool(changed).metadata()))
    with pytest.raises(ValueError):
        resolve_label("production", manifest_ref=ref, **{**context, "settings": actual})


def test_the_driver_preflight_binds_what_the_runner_binds(tmp_path, monkeypatch, mock_pool):
    """The driver preflight binds the same settings the runner binds."""
    stub_fit(monkeypatch)
    out = pool_cell(tmp_path / "cell", mock_pool)
    settings = read(out / "config.json")["label_receipt"]["binding"]["settings"]
    task = cc.select_task(read(DEV_MANIFEST), None)
    cells = tmp_path / "cells.txt"
    cells.write_text(f"chembench {task['id']} 100 31 6.00\nchembench {task['id']} 101 31 6.00\n")
    script = REPO / "tools" / "bai_smoke_2h" / "initial_pool_preflight.py"
    env_path = [str(REPO / "src"), str(REPO), *sys.path]
    result = subprocess.run(
        [sys.executable, "-B", str(script), "--file", str(mock_pool), "--cells", str(cells),
         "--chembench-manifest", str(DEV_MANIFEST), "--chembench-d0-policy", "fixed3"],
        capture_output=True, text=True, check=False,
        env={"PYTHONPATH": ":".join(env_path), "PYTHONDONTWRITEBYTECODE": "1",
             "PATH": "/usr/bin:/bin"})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {key: settings[key] for key in (
        "initial_pool_source", "initial_pool_sha256", "initial_pool_count")}


def test_a_generation_run_leaves_the_default_system_text_behind(mock_pool):
    """A generation run leaves the default system text selected afterwards."""
    assert read(mock_pool)["frozen_pool"]["init_slot"] == "pool_generator_v1"
    assert cp.prompt_identity() == {"prompt_version": cp.PROMPT_VERSION,
                                    "prompt_sha256": sha(cp.PROMPT_FILE.read_bytes())}
