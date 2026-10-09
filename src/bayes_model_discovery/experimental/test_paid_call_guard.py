"""The per-cell paid-call guard, fail-closed.

Synthetic replies through injected clients only - no SDK, no credentials, no network and no
saved run output. The proposer refuses the request that would make its paid-call count
N + 1 before the client is called; each runner stops the cell there, writes summary.json
with the stop and exits 4; an omitted option leaves every record and setting as before.
"""

import json
import re
import subprocess
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import count_proposer as cp
from .run_label_policy import bmd_runtime_settings, resolve_label
from .test_count_proposer import run_args, stub_fits
from .test_run_label_policy import registration, runner_args, write_manifest

REPOSITORY = Path(__file__).resolve().parents[3]
#: The base commit the omitted-option parity test reads its transport from; the test skips when this commit is
#: absent from the checkout.
BASE_COMMIT = "45feb07357c9b7c71be57b245c8b2cb9f82e0ee4"
CANDIDATE = {"expression": "b0 + b1*z", "family": "binomial",
             "parameters": {"b0": "intercept", "b1": "signed"}}


def context(round_=1, **extra):
    """Build a synthetic proposal context for the given round."""
    return {"run_id": "synthetic-pcg-run", "world_seed": 11, "cell": "100",
            "proposal_rep": 0, "round": round_, **extra}


def fake_client(provider, replies=(), *, delay=0.0):
    """A client whose create() hands out `replies` in order (then CANDIDATE), and counts."""
    lock = threading.Lock()
    state = {"creates": 0, "queue": list(replies), "bodies": []}

    def create(**body):
        with lock:
            state["creates"] += 1
            state["bodies"].append(body)
            reply = state["queue"].pop(0) if state["queue"] else CANDIDATE
        if delay:
            time.sleep(delay)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        if provider == "anthropic":
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 12, "output_tokens": 3}),
                id="synthetic-response", model="synthetic-model")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
            id="synthetic-response", model="synthetic-model")

    if provider == "anthropic":
        return SimpleNamespace(messages=SimpleNamespace(create=create)), state
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), state


def make_proposer(records, provider, replies=(), *, delay=0.0, module=cp, **kwargs):
    """Build a proposer of the given provider over a fake client; return it with the client state."""
    client, state = fake_client(provider, replies, delay=delay)
    if provider == "anthropic":
        return module.AnthropicCountProposer(records, client=client, **kwargs), state
    if provider == "openai":
        return module.OpenAICountProposer(records, client=client, **kwargs), state
    return module.OpenAICompatCountProposer(
        records, client=client, base_url="https://synthetic.invalid/v1",
        key_name="PCG_SYNTHETIC_KEY", model="synthetic-model", **kwargs), state


PROVIDERS = ("anthropic", "openai", "openai_compat")


# --------------------------------------------------------------------------
# 1. the proposer: one choke point, under one lock
# --------------------------------------------------------------------------


@pytest.mark.parametrize("provider", PROVIDERS)
def test_the_call_that_would_make_it_n_plus_one_is_never_made(tmp_path, provider):
    """The call that would make the paid-call count N + 1 is never made."""
    records = tmp_path / "records"
    proposer, state = make_proposer(records, provider, max_paid_calls=2)
    for round_ in (1, 2):
        _, record = proposer.propose_one("packet", context=context(round_), initial_family="binomial")
        assert record["status"] == "parsed"
    with pytest.raises(cp.PaidCallBudgetExhausted) as stop:
        proposer.propose_one("packet", context=context(3), initial_family="binomial")
    assert state["creates"] == 2 and proposer.attempts == 2
    assert (stop.value.attempts, stop.value.max_paid_calls) == (2, 2)
    # Nothing was sent for round 3, so nothing is recorded for it.
    assert not list(records.glob("round_03*"))
    # Every later request is refused the same way, and still sends nothing.
    with pytest.raises(cp.PaidCallBudgetExhausted):
        proposer.propose_one("packet", context=context(4), initial_family="binomial")
    assert state["creates"] == 2 and proposer.attempts == 2


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_budget_of_exactly_n_allows_n_calls(tmp_path, provider):
    """A budget of exactly N allows N calls."""
    proposer, state = make_proposer(tmp_path / "records", provider, max_paid_calls=3)
    for round_ in (1, 2, 3):
        _, record = proposer.propose_one("packet", context=context(round_), initial_family="binomial")
        assert record["status"] == "parsed" and record["calls"] == 1
    assert state["creates"] == proposer.attempts == 3


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_repair_turn_is_a_paid_call_and_a_refused_one_keeps_the_paid_calls_record(tmp_path, provider):
    # Budget 2: the malformed first reply and its repair are two paid calls.
    """A repair turn is a paid call; a refused one keeps the record of the paid calls."""
    allowed, state = make_proposer(tmp_path / "allowed", provider, ["not json"], max_paid_calls=2)
    _, record = allowed.propose_one("packet", context=context(1), initial_family="binomial")
    assert record["status"] == "parsed" and record["calls"] == 2 and record["repairs"] == 1
    assert state["creates"] == allowed.attempts == 2
    # Budget 1: the repair turn is refused before it is sent. The first call was paid, so its
    # request and response stay recorded, under the stop status, before the stop propagates.
    records = tmp_path / "refused"
    refused, state = make_proposer(records, provider, ["not json"], max_paid_calls=1)
    with pytest.raises(cp.PaidCallBudgetExhausted):
        refused.propose_one("packet", context=context(1), initial_family="binomial")
    assert state["creates"] == refused.attempts == 1
    request = json.loads((records / "round_01.request.json").read_text())
    response = json.loads((records / "round_01.response.json").read_text())
    assert len(request["bodies"]) == 1
    assert response["status"] == cp.BUDGET_EXHAUSTED and response["max_paid_calls"] == 1
    assert response["calls"] == 1 and response["repairs"] == 0 and len(response["responses"]) == 1
    assert response["rejected_candidates"][0]["repair_eligible"] is True
    assert response["valid_candidate_count"] == 0


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_the_legacy_event_path_is_guarded_before_its_request_record(tmp_path, provider):
    """The legacy event path is guarded before its request record is written."""
    records = tmp_path / "records"
    proposer, state = make_proposer(records, provider, [json.dumps([CANDIDATE])], max_paid_calls=1)
    public = [{"u": 0.1 * (i % 10), "y": i % 50} for i in range(9)]
    event = {"run_id": "synthetic-pcg-run", "world_seed": 11, "cell": "100", "proposal_rep": 0}
    _, metadata = proposer.propose(public, [], context={**event, "event": 5}, initial_family="binomial")
    assert metadata["status"] == "parsed"
    with pytest.raises(cp.PaidCallBudgetExhausted):
        proposer.propose([*public, *public[:5]], [], context={**event, "event": 10},
                         initial_family="binomial")
    assert state["creates"] == proposer.attempts == 1
    assert not (records / "event_10.request.json").exists()


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_parallel_burst_calls_never_pass_the_remaining_budget(tmp_path, provider):
    """Two of five calls are spent; then 24 burst calls start at once on a slow client.
    Exactly the three remaining are made, the other 21 are refused, none is sent."""
    records = tmp_path / "records"
    proposer, state = make_proposer(records, provider, delay=0.05, max_paid_calls=5)
    for round_ in (1, 2):
        proposer.propose_one("packet", context=context(round_), initial_family="binomial")
    parallel = 24
    barrier = threading.Barrier(parallel)

    def one(index):
        barrier.wait()
        try:
            proposer.propose_one("packet", context=context(0, burst_call_index=index, burst_wave=0),
                                 initial_family="binomial", burst=True)
        except cp.PaidCallBudgetExhausted:
            return "refused"
        return "made"

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        outcomes = list(pool.map(one, range(parallel)))
    assert outcomes.count("made") == 3 and outcomes.count("refused") == parallel - 3
    assert state["creates"] == proposer.attempts == 5
    assert len(list(records.glob("round_00__burst_*.response.json"))) == 3


@pytest.mark.parametrize("provider", PROVIDERS)
def test_no_cap_is_the_counter_exactly_as_before(tmp_path, provider):
    """Without a cap the call counter behaves exactly as before."""
    proposer, state = make_proposer(tmp_path / "records", provider)
    assert proposer.max_paid_calls is None
    for round_ in range(1, 8):
        proposer.propose_one("packet", context=context(round_), initial_family="binomial")
    assert state["creates"] == proposer.attempts == 7


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "0", "", "x", "1e3"])
def test_an_invalid_cap_is_refused_before_the_records_directory_exists(tmp_path, value):
    """An invalid cap is refused before the records directory exists."""
    with pytest.raises(ValueError, match="max-paid-calls"):
        make_proposer(tmp_path / "records", "anthropic", max_paid_calls=value)
    assert not (tmp_path / "records").exists()


def test_the_stop_is_not_an_exception_a_candidate_boundary_could_swallow():
    """The stop derives from BaseException so a per-candidate `except Exception` cannot swallow it."""
    assert issubclass(cp.PaidCallBudgetExhausted, BaseException)
    assert not issubclass(cp.PaidCallBudgetExhausted, Exception)
    caught = None
    try:
        try:
            raise cp.PaidCallBudgetExhausted(3, 3)
        except Exception:  # noqa: BLE001 - the boundary the stop must pass through
            caught = "swallowed"
    except cp.PaidCallBudgetExhausted as stop:
        caught = (stop.attempts, stop.max_paid_calls)
    assert caught == (3, 3)
    assert cp.PAID_CALL_BUDGET_EXIT_CODE == 4


def base_transport():
    """The base commit's count_proposer, loaded beside this package so its relative imports
    resolve to the same (unchanged) count_models and prior_probe_detail."""
    name = "src/bayes_model_discovery/experimental/count_proposer.py"
    try:
        source = subprocess.run(["git", "-C", str(REPOSITORY), "show", f"{BASE_COMMIT}:{name}"],
                                check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"base commit {BASE_COMMIT[:12]} is not readable here: {exc}")
    module = types.ModuleType("bayes_model_discovery.experimental._pcg_base_count_proposer")
    module.__file__ = cp.__file__  # the packaged prompts directory resolves beside it
    module.__package__ = "bayes_model_discovery.experimental"
    exec(compile(source, cp.__file__, "exec"), module.__dict__)  # noqa: S102 - pinned local source only
    return module


def written(records):
    """Every record file's bytes, with the one wall-clock field zeroed."""
    return {path.name: re.sub(rb'"elapsed_seconds": [-0-9.eE+]+', b'"elapsed_seconds": 0',
                              path.read_bytes())
            for path in sorted(records.iterdir())}


@pytest.mark.parametrize("provider", PROVIDERS)
def test_the_omitted_option_writes_the_base_commits_transport_bytes(tmp_path, provider):
    """With the option omitted, the transport writes the same bytes as the base commit."""
    base = base_transport()
    public = [{"u": 0.1 * (i % 10), "y": i % 50} for i in range(9)]
    event = {"run_id": "synthetic-pcg-run", "world_seed": 11, "cell": "100", "proposal_rep": 0,
             "event": 5}
    results = []
    for label, module in (("base", base), ("candidate", cp)):
        records = tmp_path / label
        replies = [CANDIDATE, "not json", CANDIDATE, "not json", "still not json",
                   json.dumps([CANDIDATE])]
        proposer, state = make_proposer(records, provider, replies, module=module)
        returned = []
        # A parsed call, a repaired one, a twice-failed one, and the legacy event path.
        for round_ in (1, 2, 3):
            _, record = proposer.propose_one("packet", context=context(round_),
                                             initial_family="binomial")
            returned.append({k: v for k, v in record.items() if k != "elapsed_seconds"})
        _, metadata = proposer.propose(public, [], context=event, initial_family="binomial")
        returned.append({k: v for k, v in metadata.items() if k != "elapsed_seconds"})
        results.append((written(records), returned, state["bodies"], proposer.attempts))
    assert results[0] == results[1]
    assert [record["status"] for record in results[1][1]] == [
        "parsed", "parsed", "proposal_failed", "parsed"]
    assert results[1][3] == 1 + 2 + 2 + 1


# --------------------------------------------------------------------------
# 2. the runners: the cell stops, keeps its records, and exits 4
# --------------------------------------------------------------------------


def guarded_factory(monkeypatch, provider_class, forms, *, family="binomial", parameters=None):
    """Replace a transport class, as the runners import it, by the real class over an
    injected client; the runner's constructor keywords are recorded and max_paid_calls is
    forwarded, so the guard under test is the runner's own."""
    real = getattr(cp, provider_class)
    replies = [{"expression": form, "family": family,
                "parameters": parameters(form) if parameters else dict(CANDIDATE["parameters"])}
               for form in forms]
    client, state = fake_client("anthropic", replies)
    state["kwargs"] = []

    def factory(records, **kwargs):
        state["kwargs"].append(kwargs)
        return real(records, client=client, max_paid_calls=kwargs.get("max_paid_calls"))

    monkeypatch.setattr(cp, provider_class, factory)
    return state


def run_main(monkeypatch, module, argv):
    """Run a runner's main() with the given argv and return its exit code."""
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as stop:
        module.main()
    return stop.value.code


def assert_stopped(out, *, budget, stopped_round, state):
    """Assert a cell stopped at its budget as recorded in summary and config; return the summary."""
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == summary["stop_reason"] == "budget_exhausted"
    assert summary["paid_api_calls"] == summary["max_paid_calls"] == budget
    assert summary["stopped_round"] == stopped_round
    assert summary["last_completed_round"] == stopped_round - 1
    # The receipt that binds the run carries the same cap.
    assert summary["label_receipt"]["binding"]["settings"]["max_paid_calls"] == budget
    assert json.loads((out / "config.json").read_text())["max_paid_calls"] == budget
    assert state["creates"] == budget
    assert [kwargs["max_paid_calls"] for kwargs in state["kwargs"]] == [budget]
    # A stopped cell publishes no point table and is not a crash.
    assert not (out / "points.csv").exists() and not (out / "failure.json").exists()
    return summary


def test_core_cell_stops_at_its_budget_keeps_earlier_rounds_and_exits_4(tmp_path, monkeypatch):
    """A CORE cell stops at its budget, keeps earlier rounds and exits with code 4."""
    from . import core_count as core

    stub_fits(monkeypatch)
    state = guarded_factory(monkeypatch, "AnthropicCountProposer", core.P1_MOCK_EXPRESSIONS[2:6],
                            parameters=lambda form: dict(core.mock_parameters(form)))
    out = tmp_path / "core_cell"
    code = run_main(monkeypatch, core, [
        "core_count", "--output", str(out), "--profile", "dev", "--rounds", "4", "--seed", "11",
        "--cell", "100", "--initial-family", "binomial", "--proposer", "anthropic",
        "--allow-paid-api", "--llm-model", "synthetic-model", "--max-paid-calls", "2",
        "--run-label", "development"])
    assert code == cp.PAID_CALL_BUDGET_EXIT_CODE == 4
    summary = assert_stopped(out, budget=2, stopped_round=3, state=state)
    assert summary["api_request_attempts"] == 2 and summary["observations"] == 4 + 3
    # Rounds 1 and 2 are complete and untouched; round 3 made its observation and refit
    # before the refused call, and nothing after that point was written.
    for t in (1, 2):
        record = json.loads((out / "proposals" / f"t{t:02d}.json").read_text())
        assert record["transport"]["status"] == "parsed" and record["after_ids"]
    assert not (out / "proposals" / "t03.json").exists()
    phases = [(row["round"], row["phase"]) for row in map(json.loads, (out / "scores.jsonl").read_text().splitlines())]
    assert (2, "after_proposal") in phases and (3, "before_proposal") in phases
    assert (3, "after_proposal") not in phases
    assert len(list((out / "llm_transport").glob("*.response.json"))) == 2


def test_core_burst_race_spends_exactly_the_budget(tmp_path, monkeypatch):
    """Twelve parallel round-0 calls (six in flight) against a budget of five."""
    from . import core_count as core

    stub_fits(monkeypatch)
    state = guarded_factory(monkeypatch, "AnthropicCountProposer", core.P1_MOCK_EXPRESSIONS[:12],
                            parameters=lambda form: dict(core.mock_parameters(form)))
    out = tmp_path / "core_burst"
    with pytest.raises(cp.PaidCallBudgetExhausted):
        core.run(run_args(out, seed=31, cell="100", rounds=1, proposer="anthropic",
                          allow_paid_api=True, llm_model="synthetic-model", initial_proposals=12,
                          active_maximum=12, burst_concurrency=6, initial_catalogue="off",
                          max_paid_calls=5))
    summary = json.loads((out / "summary.json").read_text())
    assert state["creates"] == summary["paid_api_calls"] == summary["max_paid_calls"] == 5
    assert summary["stopped_round"] == 0 and summary["last_completed_round"] is None
    assert len(list((out / "llm_transport").glob("round_00__burst_*.response.json"))) == 5
    assert not (out / "proposals" / "t00_burst.json").exists()


def test_chembench_cell_stops_at_its_budget_keeps_earlier_rounds_and_exits_4(tmp_path, monkeypatch):
    """A ChemBench cell stops at its budget, keeps earlier rounds and exits with code 4."""
    from . import chembench_core as cc
    from .test_chembench_core import _manifest
    from .test_chembench_mda_options import stub_fits as chembench_stub_fits

    chembench_stub_fits(monkeypatch)
    forms = [f"b0+b1*tanh(f0z+{index}/20)" for index in range(1, 5)]
    state = guarded_factory(monkeypatch, "AnthropicCountProposer", forms, family="lognormal")
    out = tmp_path / "chembench_cell"
    code = run_main(monkeypatch, cc, [
        "chembench_core", "--manifest", str(_manifest(tmp_path)), "--output", str(out),
        "--rounds", "3", "--particles", "16", "--q", "8", "--noise", "0.01", "--cell", "100",
        "--proposer", "anthropic", "--allow-paid-api", "--llm-model", "synthetic-model",
        "--max-paid-calls", "2", "--run-label", "development"])
    assert code == 4
    assert_stopped(out, budget=2, stopped_round=3, state=state)
    for t in (1, 2):
        record = json.loads((out / "proposals" / f"t{t:02d}.json").read_text())
        assert record["transport"]["status"] == "parsed"
        assert (out / "predictions" / f"t{t:02d}.npz").exists()
    assert not (out / "proposals" / "t03.json").exists()
    assert not (out / "predictions" / "t03.npz").exists()


@pytest.mark.usefixtures("require_boxinggym_source")
def test_boxinggym_cell_stops_at_its_budget_keeps_earlier_rounds_and_exits_4(tmp_path, monkeypatch):
    """A BoxingGym cell stops at its budget, keeps earlier rounds and exits with code 4."""
    bg = pytest.importorskip("bayes_model_discovery.experimental.boxinggym_core")
    core = stub_fits(monkeypatch)
    monkeypatch.setattr(bg, "count_fit", core.fit)
    forms = ["b0+b1*tz+b2*exp(t01)", "b0+b1*sqrt(t01)", "b0+b1*tz+b2*square(t01)"]
    state = guarded_factory(
        monkeypatch, "AnthropicCountProposer", forms,
        parameters=lambda form: {name: ("intercept" if name == "b0" else "signed")
                                 for name in ("b0", "b1", "b2") if name in form})
    out = tmp_path / "boxinggym_cell"
    code = run_main(monkeypatch, bg, [
        "boxinggym_core", "--output", str(out), "--task", "death", "--profile", "dev",
        "--rounds", "3", "--seed", "11", "--cell", "100", "--proposer", "anthropic",
        "--allow-paid-api", "--llm-model", "synthetic-model", "--max-paid-calls", "2",
        "--run-label", "development"])
    assert code == 4
    summary = assert_stopped(out, budget=2, stopped_round=3, state=state)
    assert summary["suite"] == "boxinggym" and summary["task"] == "death"
    for t in (1, 2):
        assert json.loads((out / "proposals" / f"t{t:02d}.json").read_text())["after_ids"]
    assert not (out / "proposals" / "t03.json").exists()


def test_a_mock_cell_makes_no_paid_call_and_never_meets_the_guard(tmp_path, monkeypatch):
    """A mock cell makes no paid call and never meets the guard."""
    from . import core_count as core

    stub_fits(monkeypatch)
    out = tmp_path / "mock_cell"
    core.run(run_args(out, rounds=3, max_paid_calls=1))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "completed" and summary["paid_api_calls"] == 0
    assert "stop_reason" not in summary and (out / "points.csv").exists()
    assert json.loads((out / "config.json").read_text())["max_paid_calls"] == 1


def test_the_omitted_option_adds_no_key_to_config_summary_receipt_or_settings(tmp_path, monkeypatch):
    """The omitted option adds no key to config, summary, receipt or settings."""
    from . import core_count as core

    stub_fits(monkeypatch)
    out = tmp_path / "plain_cell"
    core.run(run_args(out, rounds=2, run_label="development"))
    for name in ("config.json", "summary.json"):
        text = (out / name).read_text()
        assert "max_paid_calls" not in text and "budget_exhausted" not in text
        assert "stop_reason" not in text
    receipt = json.loads((out / "summary.json").read_text())["label_receipt"]
    assert "max_paid_calls" not in receipt["binding"]["settings"]
    for bundle in ("core", "chembench", "boxinggym"):
        assert "max_paid_calls" not in bmd_runtime_settings(bundle, runner_args())
    assert "max_paid_calls" not in cp.compat_option_kwargs(SimpleNamespace(), compat=False)


# --------------------------------------------------------------------------
# 3. the command lines and the registered comparison
# --------------------------------------------------------------------------


def runner_argv(runner, out, manifest):
    """Return a minimal command line for the named runner."""
    if runner == "core":
        return ["core_count", "--output", str(out), "--cell", "100"]
    if runner == "chembench":
        return ["chembench_core", "--manifest", str(manifest), "--output", str(out),
                "--noise", "0.01", "--cell", "100"]
    return ["boxinggym_core", "--output", str(out), "--task", "death", "--cell", "100"]


@pytest.mark.parametrize("runner", ["core", "chembench", "boxinggym"])
def test_each_cli_states_the_cap_only_when_typed_and_refuses_a_bad_one(tmp_path, monkeypatch, runner, capsys):
    """Each CLI states the cap only when typed and refuses a bad value with exit code 2."""
    if runner == "boxinggym":
        module = pytest.importorskip("bayes_model_discovery.experimental.boxinggym_core")
    else:
        from importlib import import_module
        module = import_module(f"bayes_model_discovery.experimental.{runner}_{'count' if runner == 'core' else 'core'}")
    out, manifest = tmp_path / "never", tmp_path / "synthetic_manifest.json"
    captured = []
    monkeypatch.setattr(module, "run", captured.append)
    for extra, expected in (([], None), (["--max-paid-calls", "180"], 180)):
        monkeypatch.setattr(sys, "argv", [*runner_argv(runner, out, manifest), *extra])
        module.main()
        assert getattr(captured[-1], "max_paid_calls", None) == expected
        assert hasattr(captured[-1], "max_paid_calls") is (expected is not None)
    for bad in ("0", "-3", "1.5", "many"):
        assert run_main(monkeypatch, module, [*runner_argv(runner, out, manifest),
                                              "--max-paid-calls", bad]) == 2
        assert "max-paid-calls" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("bundle", ["core", "chembench", "boxinggym"])
def test_the_settings_bind_the_cap_exactly_when_stated(bundle):
    """The runtime settings bind the cap exactly when it is stated."""
    plain = bmd_runtime_settings(bundle, runner_args())
    bound = bmd_runtime_settings(bundle, runner_args(max_paid_calls=180))
    assert bound == {**plain, "max_paid_calls": 180}
    for wrong in (0, -1, True, 1.5, "180"):
        with pytest.raises(ValueError, match="max_paid_calls"):
            bmd_runtime_settings(bundle, runner_args(max_paid_calls=wrong))


def test_a_registration_binds_the_cap_and_refuses_another_or_a_malformed_one(tmp_path):
    """A registration binds the cap and refuses a different or malformed one."""
    bound = bmd_runtime_settings("core", runner_args(max_paid_calls=107))
    ref, binding, manifest = registration(tmp_path, settings=bound)
    assert resolve_label("production", manifest_ref=ref, **binding)["run_label"] == "production"
    for stated in (bmd_runtime_settings("core", runner_args(max_paid_calls=108)),
                   bmd_runtime_settings("core", runner_args())):
        with pytest.raises(ValueError, match="runtime settings mismatch"):
            resolve_label("production", manifest_ref=ref, **{**binding, "settings": stated})
    # A float or text 107 would compare equal to the stated 107; it is refused instead.
    for wrong in (107.0, "107", True, 0):
        manifest["runtime_settings"]["core"]["max_paid_calls"] = wrong
        ref = write_manifest(Path(ref["path"]), manifest)
        with pytest.raises(ValueError, match="max_paid_calls"):
            resolve_label("production", manifest_ref=ref, **binding)
