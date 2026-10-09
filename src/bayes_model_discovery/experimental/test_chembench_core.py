"""Acceptance tests for chembench_core.py's `--structure-proposals` switch.

The real `adapters.chembench_observation` (BMD-external, network-free but
dependent on the `autoscilab` package for its real oracle) is never touched
here: `chembench_core.run` imports it as `from adapters import
chembench_observation as adapter` *inside* `run`, exactly so a test can
monkeypatch the module's `ChemBenchSession`/`sample_designs` attributes with
a small in-repo fake before `run` ever looks them up - the same reason
`boxinggym_core.run`'s own `AnthropicCountProposer` import is local (see
`count_proposer`/`boxinggym_core.py`). No credentials, no network, no real
LLM/oracle client anywhere in this file.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp as _logsumexp
from scipy.stats import gamma as _gamma_dist
from scipy.stats import lognorm as _lognorm_dist
from scipy.stats import norm as _norm_dist

# `adapters/` lives at the scratch ROOT (sibling of `src/`), not under
# `src/`, so it is not on sys.path merely because `bayes_model_discovery` is
# importable (PYTHONPATH=src). Make this test self-contained regardless of
# how the caller set PYTHONPATH for the rest of the suite.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import bayes_model_discovery.adapters.chembench_observation as real_adapter
from bayes_model_discovery.experimental import chembench_core as cc
from bayes_model_discovery.experimental import count_proposer
from bayes_model_discovery.experimental.count_proposer import AnthropicCountProposer

# Captured at IMPORT time, i.e. before the autouse fixture below replaces
# them: the one test that needs the real `autoscilab` oracle (the
# baseline cross-check) must not get the fake.
REAL_SAMPLE_DESIGNS = real_adapter.sample_designs
REAL_SESSION = real_adapter.ChemBenchSession

FAKE_NAMES = ("f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7")
FAKE_BOUNDS = {n: (0.1, 1.0) for n in FAKE_NAMES}
# An exact constant, chosen inside the same [0.1, 1.0] range every other
# design uses (so a model can still be evaluated at it without over/underflow
# - unlike an out-of-range sentinel, which would break the ordinary
# evaluation-grid PREDICTION step every round already does, in catalog mode
# too, and so would prove nothing about leakage specifically). A continuous
# `rng.uniform` draw lands on this exact 6-decimal value with probability
# ~0, so its presence anywhere can only mean the EVALUATION seed's rows
# reached somewhere they must never reach.
EVALUATION_SENTINEL = 0.543210


def _fake_sample_designs(count, seed):
    """Deterministic fake standing in for the real adapter's own sampler.

    Returns the sentinel row for exactly the seed `chembench_core.run` uses
    for the evaluation set (`seed_for("chem_evaluation_inputs", 20260907)`,
    a fixed, already-deterministic call this module makes unconditionally),
    and ordinary pseudo-random rows for every other stream (observation
    designs, the public pool for the input transform) - mirroring the real
    adapter's contract (one row per count, `FAKE_NAMES`-keyed) without
    needing `autoscilab` installed at all.
    """
    if seed == cc.seed_for("chem_evaluation_inputs", 20260907):
        return [dict.fromkeys(FAKE_NAMES, EVALUATION_SENTINEL) for _ in range(count)]
    rng = np.random.default_rng(seed)
    rows = rng.uniform(0.1, 1.0, size=(count, len(FAKE_NAMES)))
    return [{n: float(v) for n, v in zip(FAKE_NAMES, row, strict=True)} for row in rows]


class _FakeSession:
    """Stands in for `adapters.chembench_observation.ChemBenchSession`.

    A simple deterministic log-linear rate law over f0/f1 with small
    multiplicative noise - just enough structure that SMC has something
    non-degenerate to fit; this is not a claim about ChemBench's real law.
    """

    def __init__(self, domain, difficulty, law_version, *, noise_level, budget):
        if budget < 1 or not np.isfinite(noise_level) or noise_level < 0:
            raise ValueError("require positive budget and finite nonnegative noise")
        self.domain, self.difficulty, self.law_version = domain, difficulty, law_version
        self.noise_level, self.budget = noise_level, budget
        self.calls = 0
        self._rng = np.random.default_rng(12345)
        self._finished = False

    def schema(self):
        """Return the fake benchmark schema."""
        return {
            "benchmark": "fake_for_tests",
            "target": "r0",
            "input_names": list(FAKE_NAMES),
            "input_bounds": {n: list(b) for n, b in FAKE_BOUNDS.items()},
            "log_sampled_inputs": [],
            "observable_names": ["r0"],
        }

    def _rate(self, design):
        eta = 0.1 + 0.3 * design["f0"] - 0.15 * design["f1"]
        return float(np.exp(eta))

    def observe(self, design):
        """Record one observation at the given design."""
        if self._finished:
            raise RuntimeError("discovery is finished")
        if self.calls >= self.budget:
            raise RuntimeError("observation budget exhausted")
        self.calls += 1
        y = self._rate(design) * (1.0 + self.noise_level * self._rng.standard_normal())
        y = float(max(y, 1e-6))
        return {"observation_index": self.calls, "x": dict(design), "y": y, "outputs": {"r0": y}}

    def finish(self):
        """Mark the fake session as finished."""
        self._finished = True

    def reference_values(self, designs):
        """Return noiseless reference values for the given designs."""
        if not self._finished:
            raise RuntimeError("reference values are unavailable during discovery")
        primary = np.array([self._rate(d) for d in designs])
        return {
            "primary_rate": primary,
            "noiseless_observation": primary,
            "heldout_observation": primary,
        }


def _manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [{"id": "fake_task__development", "domain": "d0", "difficulty": "easy", "law_version": "v0"}]
        )
    )
    return manifest


def _args(output, manifest, **overrides):
    base = {
        "manifest": str(manifest),
        "output": str(output),
        "seed": 11,
        "rounds": 1,
        "particles": 16,
        "q": 8,
        "noise": 0.01,
        "family": "lognormal",
        "structure_proposals": "catalog",
        "proposer": "mock",
        "llm_model": None,
        "allow_paid_api": False,
        # This runner takes the prior-template switch only
        "prior_template": "default",
        # The three new axes.
        # `task` None reproduces the earlier single-row behaviour, `cell`
        # "000" the earlier recorded value, `profile` the new default. The
        # explicit `particles`/`q` above override the profile grid, exactly as
        # they did before `--profile` existed.
        "task": None,
        "cell": "000",
        "profile": "dev",
        # The acquisition axis. The shared default, which is also the A=0 cells' only
        # legal value, so every pre-existing case keeps its earlier behaviour.
        "design_policy": "joint_eig",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _patch_adapter(monkeypatch):
    monkeypatch.setattr(real_adapter, "ChemBenchSession", _FakeSession)
    monkeypatch.setattr(real_adapter, "sample_designs", _fake_sample_designs)


# The pool-path mechanics run on a two-input variant of the fake. The pool EIG/EPIG path is
# refused at input dimension d >= 3, and positive_inference.initial_catalogue refuses d < 3
# ("at least three input columns required"), so no dimension reaches that path through the
# production catalogue. The
# two-input world therefore also carries that catalogue's own rule applied to its two columns.
FAKE_NAMES_2D = FAKE_NAMES[:2]
FAKE_BOUNDS_2D = {n: FAKE_BOUNDS[n] for n in FAKE_NAMES_2D}


def _fake_sample_designs_2d(count, seed):
    """`_fake_sample_designs` projected onto the first two names (sentinel rows included)."""
    return [{n: row[n] for n in FAKE_NAMES_2D} for row in _fake_sample_designs(count, seed)]


class _FakeSession2D(_FakeSession):
    """`_FakeSession` (its rate law reads f0 and f1 only) declaring the two inputs."""

    def schema(self):
        """Return the base schema extended with the two-input names and bounds."""
        return {**super().schema(), "input_names": list(FAKE_NAMES_2D),
                "input_bounds": {n: list(b) for n, b in FAKE_BOUNDS_2D.items()}}


def _two_input_catalogue(inputs_d0, labels_d0, family):
    """Test-only substitute catalogue (positive_inference.initial_catalogue rule applied to two
    columns; production never reaches the pool branch): rank by absolute D0 Pearson
    correlation, then column index, and nest (0, 1 and 2 features: three members instead of
    four)."""
    from bayes_model_discovery.experimental.positive_inference import RegressionHypothesis

    x = np.asarray(inputs_d0, dtype=float)
    y = np.asarray(labels_d0, dtype=float)
    xc, yc = x - x.mean(0), y - y.mean()
    norms = np.linalg.norm(xc, axis=0) * np.linalg.norm(yc)
    correlations = np.divide(np.abs(xc.T @ yc), norms, out=np.zeros_like(norms), where=norms > 0)
    selected = sorted(range(x.shape[1]), key=lambda i: (-correlations[i], i))
    return [RegressionHypothesis(family, tuple(selected[:n])) for n in range(len(selected) + 1)], {
        "rule": "absolute_D0_Pearson_correlation_then_column_index",
        "features": selected,
        "correlations": correlations.tolist(),
        "public_observation_count": len(y),
        "test_labels_used": 0,
    }


def use_two_input_fake(patch):
    """Put `patch` (a MonkeyPatch) on the two-input world, pool path intact, with the
    test-only substitute catalogue (positive_inference.initial_catalogue rule applied to two
    columns; production never reaches the pool branch)."""
    patch.setattr(real_adapter, "ChemBenchSession", _FakeSession2D)
    patch.setattr(real_adapter, "sample_designs", _fake_sample_designs_2d)
    patch.setattr(cc, "initial_catalogue", _two_input_catalogue)


def _strip_volatile(config):
    for key in ("output", "manifest", "manifest_sha256"):
        config.pop(key, None)
    return config


def _strip_timing(summary):
    for key in ("wall_seconds", "user_cpu_seconds", "system_cpu_seconds", "timing_seconds"):
        summary.pop(key, None)
    return summary


# ---------------------------------------------------------------------------
# catalog (default) mode: no proposer/packet artifacts, deterministic,
# byte-identical between two independent runs of the same args.
# ---------------------------------------------------------------------------


def test_chem_catalog_mode_is_deterministic_and_untouched_by_the_switch(tmp_path):
    """Catalog mode is deterministic and unaffected by the new switch."""
    manifest = _manifest(tmp_path)
    run_a, run_b = tmp_path / "run_a", tmp_path / "run_b"
    cc.run(_args(run_a, manifest))
    cc.run(_args(run_b, manifest))

    for run_dir in (run_a, run_b):
        assert not (run_dir / "packets").exists()
        assert not (run_dir / "proposals").exists()
        assert not (run_dir / "llm_transport").exists()

    config_a = _strip_volatile(json.loads((run_a / "config.json").read_text()))
    config_b = _strip_volatile(json.loads((run_b / "config.json").read_text()))
    assert config_a == config_b
    assert config_a["structure_proposals"] == "catalog"

    summary_a = _strip_timing(json.loads((run_a / "summary.json").read_text()))
    summary_b = _strip_timing(json.loads((run_b / "summary.json").read_text()))
    assert summary_a == summary_b
    assert summary_a["active_pool"] == 4
    assert summary_a["proposals"] == 0
    assert summary_a["fit_count"] == 4 * (1 + 1)  # rounds=1 -> t in {0, 1}, 4 models each
    assert summary_a["paid_api_calls"] == 0
    assert "proposal_accounting" not in summary_a

    pred_a = json.loads((run_a / "predictions" / "t01.json").read_text())
    pred_b = json.loads((run_b / "predictions" / "t01.json").read_text())
    assert pred_a == pred_b


# ---------------------------------------------------------------------------
# free mode: paid-API guard, mock schedule completion, and the leakage
# invariant (evaluation designs never reach the packet or the proposer).
# ---------------------------------------------------------------------------


def test_chem_free_mode_refuses_anthropic_without_allow_paid_api(tmp_path):
    """Free mode refuses the Anthropic provider unless paid API use is allowed."""
    manifest = _manifest(tmp_path)
    out = tmp_path / "refused"
    with pytest.raises(ValueError, match="Anthropic proposals require"):
        cc.run(_args(out, manifest, structure_proposals="free", proposer="anthropic"))
    assert not out.exists()


def test_chem_free_mode_mock_schedule_completes_and_admits_candidates(tmp_path):
    """Free mode with a mock schedule completes and admits candidates."""
    manifest = _manifest(tmp_path)
    out = tmp_path / "free_mock"
    # `cell="111"` is what the earlier code EXECUTED for every free run (the
    # hardcoded `FREE_CELL`). Nothing here reads the
    # acquisition, and pool EIG on this eight-input fake is refused, so the run
    # moves to `110`, the same free route with the third digit 0.
    cc.run(_args(out, manifest, rounds=2, structure_proposals="free", proposer="mock", cell="110"))

    summary = json.loads((out / "summary.json").read_text())
    assert summary["paid_api_calls"] == 0
    assert summary["proposals"] == 2  # rounds 1 and 2 each propose once
    assert summary["proposal_accounting"]["admitted_candidates"] >= 1
    assert summary["proposal_accounting"]["llm_calls"] == 0  # mock: no transport call at all
    assert summary["active_pool"] > 4
    assert summary["proposal_accounting"]["initial_pool_remains_in_active_pool"] is True
    config = json.loads((out / "config.json").read_text())
    assert config["structure_proposals"] == "free"
    assert config["points_variant"] == "structure_free"
    for t in (1, 2):
        transport = json.loads((out / "proposals" / f"t{t:02d}.json").read_text())["transport"]
        assert transport["mock"] is True
        assert transport["status"] == "parsed"


def test_chem_evaluation_designs_never_reach_the_packet_or_proposer(tmp_path, monkeypatch):
    """The leakage invariant: `chem_evaluation_inputs` (seed 20260907) rows
    never reach the belief packet or the proposer transport, checked with an
    injected FAKE Anthropic client (never a real one) whose every request
    body is captured and scanned for the sentinel.

    Cell `111` keeps its acquisition route (a pool design
    could otherwise carry an evaluation row into the observations), so the run
    uses the two-input fake; its sampler projects the sentinel rows too."""
    use_two_input_fake(monkeypatch)
    manifest = _manifest(tmp_path)
    calls = []

    def create(**body):
        """Fake client call: record the request body and return a canned reply."""
        calls.append(body)
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text=json.dumps(
                        {"expression": "b0", "family": "lognormal", "parameters": {"b0": "intercept"}}
                    ),
                )
            ],
            usage=SimpleNamespace(
                model_dump=lambda **kwargs: {"input_tokens": 1, "output_tokens": 1}
            ),
            id="test-only-response",
            model="test-only-model",
            stop_reason="end_turn",
        )

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(
        count_proposer,
        "AnthropicCountProposer",
        lambda records, **kwargs: AnthropicCountProposer(records, client=fake_client),
    )

    out = tmp_path / "leak_check"
    cc.run(
        _args(
            out, manifest, rounds=2, structure_proposals="free", proposer="anthropic",
            allow_paid_api=True, cell="111",
        )
    )

    assert len(calls) >= 2, "the proposer must actually have been called (real wiring, fake client)"
    sentinel_text = repr(EVALUATION_SENTINEL)
    for body in calls:
        packet_text = body["messages"][0]["content"]
        assert sentinel_text not in packet_text
        packet = json.loads(packet_text)
        assert "evaluation" not in json.dumps(packet).lower()

    summary = json.loads((out / "summary.json").read_text())
    assert summary["paid_api_calls"] == len(calls)  # real transport, fake client - still counted
    assert summary["proposal_accounting"]["llm_calls"] == len(calls)

    # Regression coverage: this fake client always proposes "b0"/lognormal,
    # exactly the catalogue's OWN "features=()" member once re-expressed via
    # `_packet_view` - it must be recognised as a duplicate at round 1 (not
    # crash `model_logweights` at round 2 with two pool entries sharing one
    # canonical id, which is what happened before this was fixed).
    proposals_t01 = json.loads((out / "proposals" / "t01.json").read_text())
    assert proposals_t01["admitted_ids"] == []

    # Never in the packets/proposals/transport files this run wrote either.
    for pattern in ("packets/*.json", "proposals/*.json", "llm_transport/*.json"):
        for path in out.glob(pattern):
            assert sentinel_text not in path.read_text(), f"sentinel leaked into {path}"

    # The evaluation file itself IS allowed to (and must) hold the sentinel -
    # this proves the sentinel technique actually exercises the real seed,
    # rather than the fake never being asked for that stream at all.
    evaluation_inputs = np.load(out / "evaluation_inputs.npz")
    assert np.all(evaluation_inputs["x"] == EVALUATION_SENTINEL)


# ---------------------------------------------------------------------------
# Every test below uses the same fake session/sampler fixture as the four
# above: no credentials, no network, no real oracle or LLM client. The one
# exception is `test_official_evaluation_set_reproduces_the_baseline_...`,
# which needs the real `autoscilab` oracle and SKIPS when it is unavailable.
# ---------------------------------------------------------------------------

PAPER_MANIFEST = (
    Path(__file__).resolve().parents[3] / "manifests" / "chembench_paper_manifest.json"
)
# The no-leak ChemBench baseline's own nine task ids, copied verbatim from the
# evaluation repository of the comparison method, path
# results/llm-autoscilab-chembench/p4_chembench_noleak_oracle_blocked_b40_
# gpt4omini_5runs/tables/per_seed_budget.csv (column `task_id`, the 224
# contract rows). Written out here so the manifest cannot drift silently.
BASELINE_TASK_IDS = (
    "c0_michaelis_menten__easy__v0",
    "c2_product_inhibition__easy__v0",
    "c5_pingpong_bisubstrate__easy__v0",
    "c0_michaelis_menten__hard__v0",
    "c0_michaelis_menten__medium__v0",
    "c2_product_inhibition__hard__v0",
    "c2_product_inhibition__medium__v0",
    "c5_pingpong_bisubstrate__hard__v0",
    "c5_pingpong_bisubstrate__medium__v0",
)
# The nine per-task Var[log1p(primary_rate)] values the baseline recorded for
# exactly this evaluation set, same source commit, .../manifest.json key
# `var_log1p_truth`. Reproducing them proves our official evaluation set IS
# the baseline's scoring set and that `Var` is the population variance.
BASELINE_LOG1P_TRUTH_VARIANCE = {
    "c0_michaelis_menten__easy__v0": 0.9094754605582429,
    "c0_michaelis_menten__medium__v0": 1.1158810282876033,
    "c0_michaelis_menten__hard__v0": 1.2743716793762507,
    "c2_product_inhibition__easy__v0": 0.7954841283298765,
    "c2_product_inhibition__medium__v0": 0.5335658127595951,
    "c2_product_inhibition__hard__v0": 0.3976869022554729,
    "c5_pingpong_bisubstrate__easy__v0": 0.5288303095523867,
    "c5_pingpong_bisubstrate__medium__v0": 0.49642097924229084,
    "c5_pingpong_bisubstrate__hard__v0": 0.32565354377474415,
}


def _points_rows(run_dir):
    import csv

    with (run_dir / "points.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames), list(reader)


def _fake_anthropic_response(expression="b0+b1*f0z", family="lognormal", parameters=None):
    parameters = parameters or {"b0": "intercept", "b1": "signed"}
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text=json.dumps(
                    {"expression": expression, "family": family, "parameters": parameters}
                ),
            )
        ],
        usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 1, "output_tokens": 1}),
        id="test-only-response",
        model="test-only-model",
        stop_reason="end_turn",
    )


def test_22a_catalog_mode_is_byte_identical_with_the_new_flags_at_their_defaults(tmp_path):
    """Anchor, in test form ("nothing changes at the defaults").

    One run leaves `--task`/`--cell`/`--profile` at their defaults, the other
    states all three explicitly at the values that reproduce the earlier
    behaviour. Every artifact must be byte-identical except the two things a
    second run into a second directory is allowed to change - the embedded
    output path and wall-clock timing.
    """
    manifest = _manifest(tmp_path)
    # Same basename in two different parents: `points.csv`'s `run_id` column is
    # `Path(output).name`, so two differently-NAMED directories would differ
    # there for a reason that has nothing to do with this change.
    default_run, explicit_run = tmp_path / "defaults" / "run", tmp_path / "explicit" / "run"
    cc.run(_args(default_run, manifest))
    cc.run(
        _args(
            explicit_run,
            manifest,
            task="fake_task__development",
            cell="000",
            profile="dev",
            particles=16,
            q=8,
        )
    )

    # round_timings.jsonl is wall time like the other sidecars.
    observational_sidecars = {"fit_timings.jsonl", "acquisition_timings.jsonl",
                              "prediction_pool_summary.jsonl", "round_timings.jsonl"}
    files = sorted(p.relative_to(default_run) for p in default_run.rglob("*")
                   if p.is_file() and p.name not in observational_sidecars)
    assert files == sorted(
        p.relative_to(explicit_run) for p in explicit_run.rglob("*")
        if p.is_file() and p.name not in observational_sidecars
    )
    npz = [p for p in files if p.suffix == ".npz"]
    assert len(npz) >= 4
    for relative in npz:
        assert (default_run / relative).read_bytes() == (explicit_run / relative).read_bytes(), (
            f"{relative} is not byte-identical"
        )
    for relative in (p for p in files if p.suffix in (".json", ".jsonl", ".csv")):
        left = (default_run / relative).read_text().replace(str(default_run), "<out>")
        right = (explicit_run / relative).read_text().replace(str(explicit_run), "<out>")
        if relative.name in ("config.json", "summary.json") or relative.parts[0] == "fits":
            left_json, right_json = json.loads(left), json.loads(right)
            for volatile in (
                "wall_seconds",
                "user_cpu_seconds",
                "system_cpu_seconds",
                "timing_seconds",
                "fit_seconds",
                "diagnostics",
                # The ONE key that is meant to differ here: it records what
                # the command line ASKED for (`None` = "the manifest's only
                # row"), which is exactly the difference under test. The
                # RESOLVED row (`config["task"]`) is compared, and equal.
                "task_requested",
            ):
                left_json.pop(volatile, None)
                right_json.pop(volatile, None)
            assert left_json == right_json, f"{relative} differs beyond timing"
            if relative.name == "config.json":
                assert json.loads(left)["task_requested"] is None
                assert json.loads(right)["task_requested"] == "fake_task__development"
                assert json.loads(left)["task"] == json.loads(right)["task"]
        else:
            assert left == right, f"{relative} differs beyond the output path"


def test_22a_paper_manifest_carries_the_nine_baseline_task_ids_and_each_is_selectable(tmp_path):
    """The nine ids are the baseline's, verbatim, and `--task`
    reaches each one. `select_task` is the function `run` uses, so this is the
    same selection path a run takes - and the earlier `rows[0]` fixation is
    gone: a nine-row manifest without `--task` is refused, not silently
    resolved to row 0."""
    rows = json.loads(PAPER_MANIFEST.read_text())
    assert [row["id"] for row in rows] == list(BASELINE_TASK_IDS)
    assert {row["label"] for row in rows} == {"paper"}
    for row in rows:
        assert row["id"] == f"{row['domain']}__{row['difficulty']}__{row['law_version']}"

    for task_id in BASELINE_TASK_IDS:
        selected = cc.select_task(rows, task_id)
        assert selected["id"] == task_id
        assert selected is rows[BASELINE_TASK_IDS.index(task_id)]

    with pytest.raises(ValueError, match="--task is required for a 9-row manifest"):
        cc.select_task(rows, None)
    with pytest.raises(ValueError, match="is not in the manifest"):
        cc.select_task(rows, "c0_michaelis_menten__easy__v0__development")
    # A one-row manifest still needs no --task (the anchor's own command line).
    single = [rows[0]]
    assert cc.select_task(single, None) is rows[0]

    # Through `run`, both refusals happen BEFORE the output directory exists.
    for task_id, message in ((None, "--task is required"), ("nope", "is not in the manifest")):
        out = tmp_path / f"refused_{task_id}"
        with pytest.raises(ValueError, match=message):
            cc.run(_args(out, PAPER_MANIFEST, task=task_id))
        assert not out.exists()


@pytest.mark.parametrize("task_id", BASELINE_TASK_IDS)
def test_22a_each_paper_manifest_task_enters_a_full_run_through_task(tmp_path, task_id):
    """End to end, once per manifest row: `--task <id>` runs THAT
    row - its domain/difficulty reach the session, and its id is what the
    config, the summary and the points table record. Nine separate runs, so
    "the runner silently used row 0" cannot pass."""
    row = next(r for r in json.loads(PAPER_MANIFEST.read_text()) if r["id"] == task_id)
    out = tmp_path / "run"
    cc.run(_args(out, PAPER_MANIFEST, task=task_id))

    config = json.loads((out / "config.json").read_text())
    assert config["task"]["id"] == task_id
    assert config["task_requested"] == task_id
    assert config["task"] == row
    assert config["task"]["label"] == "paper"
    assert json.loads((out / "summary.json").read_text())["task_id"] == task_id
    _, points = _points_rows(out)
    assert {r["task"] for r in points} == {task_id}
    assert {r["suite"] for r in points} == {"chembench"}


def test_22a_cell_is_recorded_where_it_is_executed_and_free_mode_never_says_000(tmp_path):
    """The recorded cell is the executed cell in all three places
    (config, summary, points table), a free run never records "000" again,
    and a cell inconsistent with `--structure-proposals` is refused before any
    output exists."""
    manifest = _manifest(tmp_path)

    catalog = tmp_path / "catalog_001"
    # The `001` leg asserts that the cell executes its acquisition,
    # so it runs on the two-input fake; the `110` leg below keeps the eight-input one.
    with pytest.MonkeyPatch.context() as patch:
        use_two_input_fake(patch)
        cc.run(_args(catalog, manifest, cell="001"))
    assert json.loads((catalog / "config.json").read_text())["cell"] == "001"
    assert json.loads((catalog / "summary.json").read_text())["cell"] == "001"
    _, points = _points_rows(catalog)
    assert {row["method"] for row in points} == {"001"}
    # The other half of that rule, as the acquisition-axis decision
    # settles it: the recorded axis is the ACTIVE POLICY NAME, and `001` really
    # does execute it. Earlier versions of this assertion expected `"absent_stage1"`.
    catalog_config = json.loads((catalog / "config.json").read_text())
    assert catalog_config["design_axis"] == "joint_eig"
    assert catalog_config["design_policy"] == "joint_eig"
    assert catalog_config["design_policy_applies"] is True
    assert (catalog / "acquisition" / "t01.json").is_file()

    free = tmp_path / "free_110"
    cc.run(_args(free, manifest, structure_proposals="free", cell="110"))
    free_config = json.loads((free / "config.json").read_text())
    free_summary = json.loads((free / "summary.json").read_text())
    assert free_config["cell"] == "110"
    assert free_summary["cell"] == "110"
    _, free_points = _points_rows(free)
    assert {row["method"] for row in free_points} == {"110"}
    assert "000" not in {free_config["cell"], free_summary["cell"]}
    # The cell the packet's own constraints were built from, too - the bits
    # come from `cell_constraints(args.cell, ...)`, so a "000" cell here would
    # have forbidden proposals outright instead of being merely mis-recorded.
    packet = json.loads((free / "packets" / "t01.json").read_text())
    assert packet["constraints"]["structure_expansion"] == 1
    assert packet["constraints"]["family_expansion"] == 1

    for cell, proposals in (("000", "free"), ("110", "catalog")):
        refused = tmp_path / f"refused_{cell}_{proposals}"
        with pytest.raises(ValueError, match="inconsistent with --structure-proposals"):
            cc.run(_args(refused, manifest, structure_proposals=proposals, cell=cell))
        assert not refused.exists()
    # `010`/`011`/`100`/`101` are valid, so the only cells left to
    # refuse are the ones that are not three bits at all.
    assert cc.validate_cell("002", "catalog") == "002"
    for bad in ("00", "0000", "003", "abc", "1 1"):
        with pytest.raises(ValueError, match="--cell must be one of"):
            cc.validate_cell(bad, "free")


def test_22a_profile_grid_is_the_declared_tuple_and_dev_is_the_default(tmp_path):
    """One table gives `(particles, q, design_pool, eig_samples)`,
    the default profile is `dev`, and `--particles`/`--q` remain explicit
    overrides. Also the labelling rule in this runner: a `dev` run's contract
    label is `development`."""
    assert cc.PROFILE_GRID["dev"] == {
        "particles": 64,
        "q": 32,
        "design_pool": 32,
        "eig_samples": 32,
    }
    assert cc.PROFILE_GRID["paper"] == {
        "particles": 500,
        "q": 256,
        "design_pool": 256,
        "eig_samples": 128,
    }
    assert cc.DEFAULT_PROFILE == "dev"

    manifest = _manifest(tmp_path)
    out = tmp_path / "profile_dev"
    cc.run(_args(out, manifest, profile="dev", particles=None, q=None))
    config = json.loads((out / "config.json").read_text())
    assert (config["particles"], config["q"]) == (64, 32)
    assert config["smc"]["particles"] == 64
    assert config["profile"] == "dev"
    assert config["profile_grid"]["design_pool"] == 32
    assert config["profile_grid"]["eig_samples"] == 32
    assert config["particles_source"] == "profile"
    assert config["q_source"] == "profile"
    assert config["points_label"] == "development"
    assert json.loads((out / "summary.json").read_text())["evaluation_points_after_finish"] == 32
    _, points = _points_rows(out)
    assert {row["label"] for row in points} == {"development"}

    override = tmp_path / "profile_dev_override"
    cc.run(_args(override, manifest, profile="dev", particles=16, q=8))
    override_config = json.loads((override / "config.json").read_text())
    assert (override_config["particles"], override_config["q"]) == (16, 8)
    assert override_config["particles_source"] == "explicit"
    assert override_config["q_source"] == "explicit"

    # `--profile paper` is what a paid run must say, and it must give the
    # earlier defaults (500/256) without running them here.
    parser_defaults = cc.PROFILE_GRID["paper"]
    assert (parser_defaults["particles"], parser_defaults["q"]) == (500, 256)


def test_22a_proposer_round_window_is_the_runs_own_rounds_not_a_baked_in_20():
    """At the layer that changed: `count_proposer.propose_one`
    accepts round 21..40 when the runner hands it `max_rounds=40`, and still
    refuses them at the earlier default. Injected fake client only - no real
    Anthropic/OpenAI client is ever constructed."""
    from bayes_model_discovery.experimental.count_proposer import OpenAICountProposer

    assert count_proposer.DEFAULT_MAX_PROPOSAL_ROUNDS == 20
    packet = json.dumps({"round": 1})
    seed = cc.ScalarHypothesis("b0", "lognormal", (("b0", "intercept"),), ("f0", "f0z"))

    def context(round_number):
        """Return a minimal run context for the given round."""
        return {
            "run_id": "test-only",
            "world_seed": 11,
            "cell": "110",
            "proposal_rep": 0,
            "round": round_number,
        }

    anthropic_client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **body: _fake_anthropic_response())
    )
    for round_number in (21, 30, 40):
        import tempfile

        records = Path(tempfile.mkdtemp()) / f"anthropic_{round_number}"
        proposer = AnthropicCountProposer(records, client=anthropic_client)
        proposals, record = proposer.propose_one(
            packet,
            context=context(round_number),
            initial_family="lognormal",
            initial_structures=(seed,),
            allowed_families=cc.CHEM_ALLOWED_FAMILIES,
            max_rounds=40,
        )
        assert record["status"] == "parsed", record
        assert record["round"] == round_number
        assert len(proposals) == 1  # at most one candidate per round
        # ... and the same call is refused at the earlier default window.
        refused = AnthropicCountProposer(Path(tempfile.mkdtemp()) / "refused", client=anthropic_client)
        with pytest.raises(ValueError, match="round identity required"):
            refused.propose_one(
                packet,
                context=context(round_number),
                initial_family="lognormal",
                initial_structures=(seed,),
                allowed_families=cc.CHEM_ALLOWED_FAMILIES,
            )
    # A bad horizon is a wiring mistake, never "no cap".
    for bad in (None, 0, -1, 2.5, True):
        with pytest.raises(ValueError, match="positive integer round horizon"):
            count_proposer._validate_max_rounds(bad)
    # The OpenAI transport carries the identical change.
    assert "max_rounds" in OpenAICountProposer.propose_one.__code__.co_varnames


def test_22a_chembench_hands_its_own_rounds_to_the_proposer_and_mock_refuses_round_21(
    tmp_path, monkeypatch
):
    """Wiring: the runner passes its `--rounds` as the proposal
    window, and the FIXED 20-entry mock schedule is refused past its end at
    ENTRY - before any output directory exists - instead of quietly cycling."""
    manifest = _manifest(tmp_path)
    seen = []
    real_propose_one = AnthropicCountProposer.propose_one

    def spy(self, packet_text, **kwargs):
        """Record the max_rounds argument, then delegate to the real proposer."""
        seen.append(kwargs.get("max_rounds"))
        return real_propose_one(self, packet_text, **kwargs)

    monkeypatch.setattr(AnthropicCountProposer, "propose_one", spy)
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **body: _fake_anthropic_response())
    )
    monkeypatch.setattr(
        count_proposer,
        "AnthropicCountProposer",
        lambda records, **kwargs: AnthropicCountProposer(records, client=fake_client),
    )
    out = tmp_path / "window"
    # The round window does not read the acquisition, so the run moves
    # from `111` (pool EIG on this eight-input fake, now refused) to `110`.
    cc.run(
        _args(
            out, manifest, rounds=2, structure_proposals="free", proposer="anthropic",
            allow_paid_api=True, cell="110",
        )
    )
    assert seen == [2, 2], seen
    assert json.loads((out / "config.json").read_text())["max_proposal_rounds"] == 2
    # Again, on the transport's own record: the cell the proposer was
    # called with is the cell the run states, not a hardcoded FREE_CELL.
    records = [
        json.loads((out / "proposals" / f"t{round_number:02d}.json").read_text())
        for round_number in (1, 2)
    ]
    for round_number, record in zip((1, 2), records, strict=True):
        assert record["transport"]["cell"] == "110"
        assert record["transport"]["round"] == round_number
    # Pinned here too: a proposal EVERY round, at most one
    # candidate per round, and no revision-point schedule anywhere - the
    # proposal gate is still `if free and t`, so the rounds are 1..R with no
    # gaps and no flag selecting them.
    assert [record["round"] for record in records] == [1, 2]
    for record in records:
        assert len(record["proposed_ids"]) == 1
        assert record["transport"]["policy"] == "P1"
    assert "revision" not in Path(cc.__file__).read_text().lower()

    # The mock schedule: exactly 20 fixed entries, and no silent extension.
    assert cc.MOCK_SCHEDULE_ROUNDS == 20
    schedule = cc._mock_schedule(list(FAKE_NAMES), cc.CHEM_ALLOWED_FAMILIES)
    assert len(schedule) == 20
    refused = tmp_path / "mock_21"
    with pytest.raises(ValueError, match="fixed at 20 rounds"):
        cc.run(
            _args(refused, manifest, rounds=21, structure_proposals="free", proposer="mock",
                  cell="110")
        )
    assert not refused.exists()
    # ... and the catalog path is unaffected by the mock horizon (it never
    # proposes at all), so a long catalog run is not refused here.
    assert cc.validate_cell("000", "catalog") == "000"


def test_22a_free_mode_retires_by_evidence_at_k_max_and_never_re_proposes(tmp_path, monkeypatch):
    """The CORE retirement rule. `K_max` is 12 as declared; the
    mechanism itself is exercised with a deliberately small maximum so that a
    three-round run trips it, then checked for all four properties the rule
    states - evidence-top K kept, ties broken by the stable id, the loser
    archived as `retired_by_evidence`, and never proposable again."""
    assert cc.ACTIVE_MAXIMUM == 12  # K_max, from core_count

    manifest = _manifest(tmp_path)
    monkeypatch.setattr(cc, "ACTIVE_MAXIMUM", 5)
    out = tmp_path / "retire"
    cc.run(
        _args(out, manifest, rounds=4, structure_proposals="free", proposer="mock", cell="110")
    )

    summary = json.loads((out / "summary.json").read_text())
    accounting = summary["proposal_accounting"]
    assert accounting["active_maximum"] == 5
    assert summary["active_pool"] <= 5
    assert accounting["retired_candidates"] >= 1
    retired_ids = {item["model_id"] for item in accounting["retirements"]}

    rounds = [
        json.loads((out / "proposals" / f"t{t:02d}.json").read_text()) for t in (1, 2, 3, 4)
    ]
    assert retired_ids == {mid for record in rounds for mid in record["retired_ids"]}
    for record in rounds:
        assert len(record["active_pool_after"]) <= 5
        # Evidence-top K: nothing retired is still in the pool afterwards.
        assert not set(record["retired_ids"]) & set(record["active_pool_after"])

    # Archived as retired_by_evidence, with the reason key set
    # belief_packet.ARCHIVE_REASON_KEYS fixes for that outcome, and reachable
    # from the belief packet's own archive block (so the model is told).
    packet = json.loads((out / "packets" / "t04.json").read_text())
    archived = {
        entry["model_id"]: entry
        for entry in packet["archive"]
        if entry["outcome"] == "retired_by_evidence"
    }
    retired_before_last_packet = {mid for record in rounds[:3] for mid in record["retired_ids"]}
    assert retired_before_last_packet, "the small maximum must have retired something by round 3"
    assert retired_before_last_packet <= set(archived)
    assert archived
    for entry in archived.values():
        assert set(entry["reason"]) == {
            "log_evidence_at_retirement",
            "gap_to_cutoff_nats",
            "active_rounds",
            "computed_round",
            "n_obs",
        }
        assert entry["reason"]["gap_to_cutoff_nats"] >= 0

    # Never re-proposed: a retired id never reappears as an admission in any
    # later round, and it is still in the pool's "already known" set (which is
    # what forbids the re-proposal in the first place).
    for index, record in enumerate(rounds):
        for mid in record["retired_ids"]:
            for later in rounds[index + 1 :]:
                assert mid not in later["admitted_ids"]
                assert mid not in later["active_pool_after"]


def test_22a_primary_and_auxiliary_metric_definitions(tmp_path):
    """The primary metric IS the baseline's own statistic.

        nmse_mda = nmsle = mean_Q[(log1p(pred) - log1p(truth))^2] / Var_Q[log1p(truth)]
        rmsle    = sqrt(mean_Q[(log1p(pred) - log1p(truth))^2])

    with `Var` the population variance (ddof=0). Checked three ways: against
    an independently computed value, against the baseline's own published
    (rmsle, Var, nmse_mda) triple for its first contract row, and for the
    identity `nmsle == nmse_mda`.
    """
    rng = np.random.default_rng(2026)
    truth = rng.lognormal(0.0, 1.0, 500)
    predicted = truth * rng.lognormal(0.0, 0.2, 500)
    log_truth = np.log1p(truth)
    variance = float(log_truth.var())  # ddof=0
    expected_squared = float(np.mean((np.log1p(predicted) - log_truth) ** 2))

    scores = cc.log_error_metrics(predicted, log_truth, variance)
    assert scores["rmsle"] == pytest.approx(np.sqrt(expected_squared), rel=1e-15)
    assert scores["nmse_mda"] == pytest.approx(expected_squared / variance, rel=1e-15)
    assert scores["nmsle"] == scores["nmse_mda"]  # one statistic, two names
    assert scores["nmse_mda"] == pytest.approx(scores["rmsle"] ** 2 / variance, rel=1e-12)

    # The baseline's own first contract row (shared read-only repository
    # evaluation repository of the comparison method, .../p4_chembench_noleak_oracle_blocked_
    # b40_gpt4omini_5runs/tables/per_seed_budget.csv row `1,run1,
    # c0_michaelis_menten__easy__v0,easy,True,20,4`): bench_metric_0 (rmsle) =
    # 0.06907901335977809 and nmse_mda = 0.00524688 at Var = 0.9094754605582429.
    baseline_rmsle = 0.06907901335977809
    baseline_variance = BASELINE_LOG1P_TRUTH_VARIANCE["c0_michaelis_menten__easy__v0"]
    assert baseline_rmsle**2 / baseline_variance == pytest.approx(0.00524688, rel=1e-6)

    # A perfect prediction is exactly 0 under both names, and a run's own
    # recorded definition text names the same formula.
    perfect = cc.log_error_metrics(truth, log_truth, variance)
    assert perfect == {"nmse_mda": 0.0, "nmsle": 0.0, "rmsle": 0.0}

    manifest = _manifest(tmp_path)
    out = tmp_path / "metrics"
    cc.run(_args(out, manifest))
    official = json.loads((out / "official_scores_after_finish.json").read_text())
    assert official["evaluation_set"]["count"] == 1000
    assert official["evaluation_set"]["seed"] == 42
    assert official["evaluation_set"]["primary_target"] == "primary_rate"
    assert official["reference_used_for_fit_or_selection"] is False
    for row in official["rounds"]:
        assert row["nmsle"] == row["nmse_mda"]
        assert row["rmsle"] ** 2 == pytest.approx(
            row["nmse_mda"] * official["evaluation_set"]["log1p_primary_variance"], rel=1e-12
        )
        assert 0.0 <= row["coverage90"] <= 1.0
        assert row["width90"] >= 0.0


def test_22a_points_csv_carries_the_contract_columns_with_code_sha(tmp_path):
    """The plotting contract point table.

    The first twenty-two columns are the contract's own, in the collector's
    order, `code_sha` among them; the two declared auxiliary columns follow
    (the ForceBench pass-through precedent). Undefined metrics are BLANK,
    never 0, and x/x_unit are the baseline's own axis.
    """
    manifest = _manifest(tmp_path)
    out = tmp_path / "points"
    cc.run(_args(out, manifest, rounds=2))

    header, rows = _points_rows(out)
    # The plotting contract: 25 columns in one
    # declared order, shared with the CORE emitter (one definition, in
    # core_count), the contract's 1..21 ending `variant` then `code_sha` (22),
    # `brier` (23), `nmsle` (24), `rmsle` (25).
    assert tuple(header) == cc.CONTRACT_POINTS_COLUMNS
    assert len(cc.CONTRACT_POINTS_COLUMNS) == 25
    assert header[20:] == ["variant", "code_sha", "brier", "nmsle", "rmsle"]

    config = json.loads((out / "config.json").read_text())
    assert len(rows) == 3  # t = 0, 1, 2
    assert [int(row["x"]) for row in rows] == [4, 5, 6]  # D0 4 + one per round
    for row in rows:
        assert row["suite"] == "chembench"
        assert row["task"] == "fake_task__development"
        assert row["method"] == "000"
        assert row["x_unit"] == "oracle_calls"
        assert row["y_name"] == "nmse_mda"
        assert row["y_label"]
        assert row["status"] == "completed"
        assert row["label"] == "development"
        assert row["variant"] == ""
        assert row["proposer"] == "none"  # catalog mode constructs no proposer
        # The full 64-hex sha256 stays the recorded value.
        assert row["code_sha"] == config["source_sha256"]["chembench_core.py"]
        assert len(row["code_sha"]) == 64
        assert row["run_id"] == "points"
        # Undefined metrics are blank, never 0. `brier` joins the two: this runner's
        # outcome is a positive continuous rate, never a binary event. `kl` IS
        # defined, but only on the FINAL checkpoint row - the official
        # evaluation set opens only after `finish()` - so every earlier row
        # still leaves it blank.
        assert row["crps"] == "" and row["is_0.1"] == ""
        if row is rows[-1]:
            assert row["kl"] != "" and row["kl"] != "0"
            float(row["kl"])
        else:
            assert row["kl"] == ""
        assert row["brier"] == "" and row["brier"] != "0"
        # Defined metrics are computed from our own predictive samples.
        for column in ("y_primary", "nll", "coverage90", "width90", "nmsle", "rmsle"):
            assert row[column] != ""
            float(row[column])
        assert row["nmsle"] == row["y_primary"]

    # The rows are exactly the official scoring rows, in round order.
    official = json.loads((out / "official_scores_after_finish.json").read_text())
    assert [row["round"] for row in official["rounds"]] == [0, 1, 2]
    assert [float(row["y_primary"]) for row in rows] == [
        row["nmse_mda"] for row in official["rounds"]
    ]

    # A free run records the LLM model instead, and its own variant token.
    free = tmp_path / "points_free"
    cc.run(_args(free, manifest, structure_proposals="free", proposer="mock", cell="110"))
    _, free_rows = _points_rows(free)
    assert {row["proposer"] for row in free_rows} == {"mock"}
    assert {row["variant"] for row in free_rows} == {"structure_free"}
    assert {row["method"] for row in free_rows} == {"110"}


def test_22a_official_evaluation_set_reproduces_the_baselines_own_truth_variance():
    """With the real oracle: `sample_designs(1000, 42)` IS the
    set the no-leak baseline scored on.

    The baseline recorded `Var[log1p(y_true)]` per task (derived there from two
    constant laws through `evaluate_law`, i.e. E[L^2] - E[L]^2 - the ddof = 0
    form). Recomputing it directly from the adapter's `primary_rate` on
    `sample_designs(1000, 42)` must give the same nine numbers; if either the
    sampler, the count, the seed, the truth target or the variance convention
    differed, they would not match to last-ULP. Skipped where `autoscilab` is
    not on PYTHONPATH (it is never installed).
    """
    pytest.importorskip("autoscilab.oracle.chembench")
    designs = REAL_SAMPLE_DESIGNS(cc.OFFICIAL_EVALUATION_COUNT, cc.OFFICIAL_EVALUATION_SEED)
    assert len(designs) == 1000
    for row in json.loads(PAPER_MANIFEST.read_text()):
        session = REAL_SESSION(
            row["domain"], row["difficulty"], row["law_version"], noise_level=0.01, budget=1
        )
        session.finish()
        reference = session.reference_values(designs)
        variance = float(np.log1p(reference[cc.OFFICIAL_PRIMARY_TARGET]).var())
        assert variance == pytest.approx(
            BASELINE_LOG1P_TRUTH_VARIANCE[row["id"]], rel=1e-14
        ), row["id"]


# Nine distinct free-form candidates for the K_max boundary test below, all
# non-linear so none of them can collide with the fixed catalogue's own
# packet view (a `RegressionHypothesis` re-expresses as a purely LINEAR
# `b0+b1*<f>z+...` form - see `chembench_core._packet_view`), and all over the
# standardized derived inputs the interface declares.
BOUNDARY_EXPRESSIONS = (
    "b0+b1*square(f0z)",
    "b0+b1*tanh(f0z)",
    "b0+b1*square(f1z)",
    "b0+b1*tanh(f1z)",
    "b0+b1*square(f2z)",
    "b0+b1*tanh(f2z)",
    "b0+b1*square(f3z)",
    "b0+b1*tanh(f3z)",
    "b0+b1*square(f4z)",
)


def test_22a_at_the_real_k_max_12_the_13th_admission_retires_the_weakest_and_it_is_refused(
    tmp_path, monkeypatch
):
    """The retirement rule at its PRODUCTION threshold.

    `ACTIVE_MAXIMUM` is NOT reduced here. The run grows the pool to exactly 12
    active candidates (4 fixed catalogue + 8 admitted), then admits a 13th,
    which must retire the SINGLE lowest-evidence candidate; the next round
    re-offers exactly that retired candidate, which must be refused and never
    re-admitted.

    The re-offer is arranged with a dynamic mock schedule: rounds 1..9 hand out
    nine distinct expressions, and round 10 looks up whichever model the run
    itself retired (observed through `_archive_entry`, whose every call this
    test records). That way the boundary is exercised in ONE run and the test
    never has to predict which candidate will lose.
    """
    assert cc.ACTIVE_MAXIMUM == 12, "K_max must be the production value here"

    archived = []
    real_archive_entry = cc._archive_entry

    def recording_archive_entry(candidate, round_index, outcome, reason, note=None):
        """Record each archive entry, then delegate to the real function."""
        entry = real_archive_entry(candidate, round_index, outcome, reason, note=note)
        archived.append(entry)
        return entry

    monkeypatch.setattr(cc, "_archive_entry", recording_archive_entry)

    class _Schedule:
        """A 20-entry schedule whose entry for round 10 is decided at read
        time, from what the run has already archived."""

        def __len__(self):
            return cc.MOCK_SCHEDULE_ROUNDS

        def __getitem__(self, index):
            if index < len(BOUNDARY_EXPRESSIONS):
                return (BOUNDARY_EXPRESSIONS[index], "lognormal")
            retired = [e for e in archived if e["outcome"] == "retired_by_evidence"]
            assert retired, "the 13th admission must have retired something by now"
            return (retired[0]["expression"], retired[0]["family"])

    monkeypatch.setattr(cc, "_mock_schedule", lambda names, allowed: _Schedule())

    manifest = _manifest(tmp_path)
    out = tmp_path / "k_max_boundary"
    cc.run(
        _args(out, manifest, rounds=10, structure_proposals="free", proposer="mock", cell="110")
    )

    rounds = [
        json.loads((out / "proposals" / f"t{t:02d}.json").read_text()) for t in range(1, 11)
    ]
    summary = json.loads((out / "summary.json").read_text())
    accounting = summary["proposal_accounting"]
    assert accounting["active_maximum"] == 12

    # Rounds 1..8 grow the pool to exactly K_max with nothing retired.
    for index, record in enumerate(rounds[:8], start=1):
        assert record["admitted_ids"], f"round {index} admitted nothing"
        assert record["retired_ids"] == [], f"round {index} retired too early"
        assert len(record["active_pool_after"]) == 4 + index
    assert len(rounds[7]["active_pool_after"]) == 12  # exactly K_max, round 8

    # Round 9 admits the 13th and must retire exactly ONE candidate.
    ninth = rounds[8]
    assert len(ninth["admitted_ids"]) == 1
    assert len(ninth["retired_ids"]) == 1
    retired_id = ninth["retired_ids"][0]
    assert len(ninth["active_pool_after"]) == 12
    assert retired_id not in ninth["active_pool_after"]
    assert accounting["retired_candidates"] == 1
    assert summary["active_pool"] == 12
    assert [item["model_id"] for item in accounting["retirements"]] == [retired_id]

    # ... and the one retired is the LOWEST-evidence member of the 13. The
    # thirteen round-9 fits on disk are the evidence the rule ranked on.
    evidence = {}
    for path in (out / "fits").glob("t09_*.json"):
        body = json.loads(path.read_text())
        evidence[path.stem.split("_", 1)[1]] = body["log_evidence"]
    assert len(evidence) == 13, sorted(evidence)
    retirement = accounting["retirements"][0]
    assert retirement["log_evidence_at_retirement"] == min(evidence.values())
    assert retirement["cutoff_log_evidence"] == sorted(evidence.values())[1]

    # Archived as retired_by_evidence with the fixed reason key set, and
    # visible in the packet the model itself is shown at round 10.
    packet = json.loads((out / "packets" / "t10.json").read_text())
    retired_entries = [
        e for e in packet["archive"] if e["outcome"] == "retired_by_evidence"
    ]
    assert [e["model_id"] for e in retired_entries] == [retired_id]
    assert set(retired_entries[0]["reason"]) == {
        "log_evidence_at_retirement",
        "gap_to_cutoff_nats",
        "active_rounds",
        "computed_round",
        "n_obs",
    }
    assert retired_entries[0]["reason"]["gap_to_cutoff_nats"] >= 0

    # Round 10 re-offers exactly that model: refused, not re-admitted, not
    # back in the pool, and the archive SAYS the match was a retirement.
    tenth = rounds[9]
    assert tenth["proposed_ids"] == [retired_id]
    assert tenth["admitted_ids"] == []
    assert tenth["retired_ids"] == []
    assert retired_id not in tenth["active_pool_after"]
    assert len(tenth["active_pool_after"]) == 12
    refusals = [
        e
        for e in archived
        if e["outcome"] == "duplicate"
        and e["model_id"] == retired_id
        and e["proposed_round"] == 10
    ]
    assert len(refusals) == 1
    assert refusals[0]["note"] == cc.RETIRED_REFUSAL_NOTE
    assert refusals[0]["reason"]["matched_model_id"] == retired_id

    # Membership is unchanged by the refused re-offer, and the twelve that
    # stayed are exactly the twelve highest evidences of the thirteen: the one
    # retired is the minimum and every survivor is at or above the cutoff.
    #
    # Compared on VALUES, not ids, deliberately: `fits/*.json` is named by the
    # raw `hypothesis.model_id`, while the pool/archive/`known_ids` bookkeeping
    # speaks the grammar identity space (`_packet_view`), and for the four
    # fixed-catalogue members those two ids differ by construction (see
    # `_packet_view`'s docstring).
    assert sorted(tenth["active_pool_after"]) == sorted(ninth["active_pool_after"])
    ranked = sorted(evidence.values())
    assert ranked[0] == retirement["log_evidence_at_retirement"]
    assert all(value >= retirement["cutoff_log_evidence"] for value in ranked[1:])
    assert len(ranked) == 13 and len(tenth["active_pool_after"]) == 12


# ---------------------------------------------------------------------------
# The acquisition axis.
#
# `001`/`111` choose each round's design from the pre-declared `sample_designs`
# candidate pool by the shared joint-EIG policy; `000`/`110` keep the pre-drawn
# `chem_observation_designs` sequence and are the byte-identical anchor.
# ---------------------------------------------------------------------------

# The pool draw and the official-set draw are the two `sample_designs` streams
# the acquisition-axis decision has to keep apart, so a stub that makes them COLLIDE is the only way
# to exercise the disjointness refusal (two independent continuous draws never
# collide on their own - which is the point of the check).
COLLIDING_DESIGN = dict.fromkeys(FAKE_NAMES, 0.25)


def _colliding_sample_designs(count, seed):
    rows = _fake_sample_designs(count, seed)
    if seed in (
        cc.OFFICIAL_EVALUATION_SEED,
        cc.seed_for(cc.DESIGN_POOL_STREAM, cc.DESIGN_POOL_SEED),
    ):
        rows[0] = dict(COLLIDING_DESIGN)
    return rows


def _acquisition_records(run_dir):
    return [
        json.loads(path.read_text())
        for path in sorted((run_dir / "acquisition").glob("t*.json"))
    ]


@pytest.mark.parametrize(
    ("cell", "structure_proposals"), [("001", "catalog"), ("111", "free")]
)
def test_22a2_the_eig_design_choice_is_reproducible_under_a_fixed_seed(
    tmp_path, monkeypatch, cell, structure_proposals
):
    """The acquisition record of two runs of the same arguments is
    BYTE-identical, on both A=1 cells.

    The choice runs on its own dedicated stream, `seed_for("chem_acquisition",
    seed, t)` - the same shape as `core_count.acquisition_seed_for` and
    `boxinggym_core`'s own acquisition seed - so nothing about it depends on how
    much of any other generator the round happened to consume. The observations
    that follow from the choice are compared too: a reproducible record over a
    non-reproducible observation would be worthless.

    The pool EIG path runs on the two-input fake.
    """
    use_two_input_fake(monkeypatch)
    manifest = _manifest(tmp_path)
    first, second = tmp_path / "a" / "run", tmp_path / "b" / "run"
    for out in (first, second):
        cc.run(
            _args(
                out,
                manifest,
                cell=cell,
                structure_proposals=structure_proposals,
                rounds=3,
            )
        )

    names = sorted((first / "acquisition").glob("t*.json"))
    assert [path.name for path in names] == ["t01.json", "t02.json", "t03.json"]
    for path in names:
        assert path.read_bytes() == (second / "acquisition" / path.name).read_bytes(), (
            f"{path.name} is not reproducible"
        )
    assert (first / "observations.jsonl").read_bytes() == (
        second / "observations.jsonl"
    ).read_bytes()

    records = _acquisition_records(first)
    assert [record["seed"] for record in records] == [
        cc.seed_for(cc.ACQUISITION_STREAM, 11, t) for t in (1, 2, 3)
    ]
    # A real decision, not a constant: one winner per round, each the argmax of
    # its own table under the ONE registered tie rule and tolerance.
    for record in records:
        assert record["target"] == "joint_model_parameter"
        assert record["policy"]["design_policy"] == "joint_eig"
        assert record["policy"]["tie_rule"] == "smallest_design_value"
        assert record["policy"]["tie_atol"] == 1e-12
        assert record["policy"]["objective_column"] == "eig"
        top = max(row["eig"] for row in record["rows"])
        assert record["policy"]["chosen_value"] == top
        assert record["chosen"] == min(
            row["design"] for row in record["rows"] if abs(row["eig"] - top) <= 1e-12
        )
    # ... and the designs it picked are the ones the session was actually asked
    # to observe, in order.
    observations = [
        json.loads(line) for line in (first / "observations.jsonl").read_text().splitlines()
    ]
    for index, record in enumerate(records):
        observed = observations[4 + index]["x"]
        assert [float(observed[name]) for name in FAKE_NAMES_2D] == record["chosen"]


def test_22a2_design_pool_size_and_seed_are_the_pre_declared_dev_32_paper_256(tmp_path, monkeypatch):
    """`PROFILE_GRID`'s pre-declared `design_pool`/`eig_samples`, and
    the pool a run actually scores, are the same thing.

    The sizes were pre-declared for
    exactly this, and the SEED is pre-declared here as one constant, fixed for
    every task, seed and cell - so the arms of one task compare over one
    candidate set. The last assertion recomputes the pool from the adapter and
    requires the scored rows to be it, row for row.

    The pool EIG path runs on the two-input fake.
    """
    use_two_input_fake(monkeypatch)
    assert cc.PROFILE_GRID["dev"]["design_pool"] == 32
    assert cc.PROFILE_GRID["paper"]["design_pool"] == 256
    assert cc.PROFILE_GRID["dev"]["eig_samples"] == 32
    assert cc.PROFILE_GRID["paper"]["eig_samples"] == 128
    assert cc.DESIGN_POOL_SEED == 20260908
    assert cc.DESIGN_POOL_STREAM == "chem_design_pool_inputs"

    manifest = _manifest(tmp_path)
    for profile, size, samples in (("dev", 32, 32), ("paper", 256, 128)):
        out = tmp_path / profile / "run"
        cc.run(_args(out, manifest, cell="001", profile=profile, rounds=2))
        config = json.loads((out / "config.json").read_text())
        pool = config["design_pool"]
        assert pool["count"] == size
        assert pool["count_source"] == "profile_grid.design_pool"
        assert pool["declared_seed"] == cc.DESIGN_POOL_SEED
        assert pool["stream"] == cc.DESIGN_POOL_STREAM
        assert pool["sampler_seed"] == cc.seed_for(cc.DESIGN_POOL_STREAM, cc.DESIGN_POOL_SEED)
        assert pool["eig_samples"] == samples
        assert pool["read_by_this_run"] is True
        assert pool["supported_policies"] == ["joint_eig"]
        assert pool["registered_policies"] == [policy for policy in cc.POLICIES
                                                if policy not in ("epig", "space_filling")]

        first, second = _acquisition_records(out)
        assert first["policy"]["pool_size"] == size
        assert first["policy"]["candidate_count"] == len(first["rows"]) == size
        assert first["policy"]["eig_samples"] == samples
        # Round 2 scores the pool MINUS round 1's design: the adapter's native
        # noise is deterministic per design, so a repeat carries no new
        # information (the `boxinggym_core` `design_exclusion="observed"` rule).
        assert second["policy"]["excluded_pool_indices"] == [
            first["policy"]["chosen_pool_index"]
        ]
        assert second["policy"]["candidate_count"] == len(second["rows"]) == size - 1
        assert (
            second["policy"]["chosen_pool_index"] != first["policy"]["chosen_pool_index"]
        )

        drawn = real_adapter.sample_designs(size, pool["sampler_seed"])
        assert [[float(row[name]) for name in FAKE_NAMES_2D] for row in drawn] == [
            row["design"] for row in first["rows"]
        ]
        assert [row["pool_index"] for row in first["rows"]] == list(range(size))


def test_22a2_the_candidate_pool_is_disjoint_from_the_official_evaluation_set(
    tmp_path, monkeypatch
):
    """Verified and recorded per run, and it HALTS when it fails.

    Part one is the recorded proof, on an A=0 cell as well as an A=1 cell (the
    pool is drawn and checked on every cell, so the declaration and its proof
    are in every run's config.json). Part two makes the two streams collide on
    purpose and requires the run to refuse BEFORE any output exists - the check
    lives in the same entry-refusal zone as `--cell`/`--profile`/`--rounds`.
    """
    manifest = _manifest(tmp_path)
    # The A=1 run keeps its pool on the two-input fake, and the A=0
    # run shares that fake so both proofs are about one sampler.
    use_two_input_fake(monkeypatch)
    for cell in ("000", "001"):
        out = tmp_path / cell / "run"
        cc.run(_args(out, manifest, cell=cell))
        pool = json.loads((out / "config.json").read_text())["design_pool"]
        assert pool["official_evaluation_overlap"] == 0
        assert pool["disjoint_from_official_evaluation"] is True
        assert pool["observed_designs_leave_the_candidate_set"] is True

    # The real adapter's own two streams, at both declared sizes, are disjoint
    # from the official set - checked here on the sampler itself.
    official = {
        cc._design_key(row)
        for row in _fake_sample_designs_2d(
            cc.OFFICIAL_EVALUATION_COUNT, cc.OFFICIAL_EVALUATION_SEED
        )
    }
    for size in (32, 256):
        drawn = _fake_sample_designs_2d(
            size, cc.seed_for(cc.DESIGN_POOL_STREAM, cc.DESIGN_POOL_SEED)
        )
        assert not official & {cc._design_key(row) for row in drawn}

    monkeypatch.setattr(real_adapter, "sample_designs", _colliding_sample_designs)
    refused = tmp_path / "refused" / "run"
    with pytest.raises(ValueError, match="disjoint from the scored set"):
        cc.run(_args(refused, manifest, cell="001"))
    assert not refused.exists()


@pytest.mark.parametrize(
    ("cell", "structure_proposals"), [("000", "catalog"), ("110", "free")]
)
def test_22a2_a_zero_cells_keep_the_pre_drawn_stream_and_write_no_acquisition_record(
    tmp_path, cell, structure_proposals
):
    """The anchor claim, in test form: nothing on the A=0 path reads the pool.

    Every round's design is still `sample_designs(4 + rounds, seed_for(
    "chem_observation_designs", seed))[3 + t]`, position for position, the pool
    is recorded but not read, no `acquisition/` artifact is produced at all, and
    `design_axis` says so. The byte-level anchor against the earlier version of the code is the
    development run; this test pins
    the mechanism that makes it hold.
    """
    manifest = _manifest(tmp_path)
    out = tmp_path / "run"
    cc.run(
        _args(out, manifest, cell=cell, structure_proposals=structure_proposals, rounds=3)
    )

    assert not (out / "acquisition").exists()
    config = json.loads((out / "config.json").read_text())
    assert config["design_axis"] == cc.DESIGN_AXIS_ABSENT == "fixed_random_designs"
    assert config["design_policy"] == "joint_eig"
    assert config["design_policy_applies"] is False
    assert config["design_policy_target"] == "random"
    assert config["design_pool"]["read_by_this_run"] is False
    assert config["label_policy"] == "fixed_random_designs_from_native_input_distribution"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["design_axis"] == cc.DESIGN_AXIS_ABSENT
    # The acquisition timing bucket exists only where an acquisition happens,
    # so an A=0 summary keeps its earlier key set exactly.
    assert "acquisition" not in summary["timing_seconds"]

    expected = _fake_sample_designs(
        4 + 3, cc.seed_for("chem_observation_designs", 11)
    )
    observed = [
        json.loads(line)["x"]
        for line in (out / "observations.jsonl").read_text().splitlines()
    ]
    assert len(observed) == 7
    assert [cc._design_key(row) for row in observed] == [
        cc._design_key(row) for row in expected
    ]


def test_22a2_points_method_is_the_cell_and_variant_follows_the_v2_6_token_rule(
    tmp_path, monkeypatch
):
    """`method` is the executed cell; `variant` follows the design-token rule.

    `design_policies.variant_token` emits a `design_*` token only when the
    policy differs from the CELL DEFAULT, and the cell default on an A=1 cell IS
    `joint_eig` - so a `001`/`111` joint-EIG run emits NO design token and is
    told apart from `000`/`110` by `method`, which is what the plot contract's
    fixed method list requires (a policy name may not enter it). The
    non-default tokens are pinned on the shared helper, alphabetically joined.

    The run-level half depends on `001`/`111` executing their
    acquisition, so all four cells run on the two-input fake.
    """
    use_two_input_fake(monkeypatch)
    manifest = _manifest(tmp_path)
    expected = {"000": "", "001": "", "110": "structure_free", "111": "structure_free"}
    for cell, variant in expected.items():
        out = tmp_path / cell / "run"
        cc.run(_args(out, manifest, cell=cell, structure_proposals=cc.structure_mode(cell)))
        config = json.loads((out / "config.json").read_text())
        assert config["points_variant"] == variant
        header, rows = _points_rows(out)
        assert rows, cell
        for row in rows:
            assert row["method"] == cell
            assert row["variant"] == variant
        assert header == list(cc.CONTRACT_POINTS_COLUMNS)
        assert len(header) == 25

    # The token rule itself, on the shared helper the runner calls.
    assert cc._points_variant("catalog", "default", "joint_eig", True) == ""
    assert cc._points_variant("free", "default", "joint_eig", True) == "structure_free"
    assert (
        cc._points_variant("catalog", "default", "structure_eig", True)
        == "design_structure_eig"
    )
    assert (
        cc._points_variant("free", "default", "point_disagreement", True)
        == "design_point_disagreement+structure_free"
    )
    # A=0: no policy applies, so no token, whatever the (default) policy says.
    assert cc._points_variant("catalog", "default", "joint_eig", False) == ""
    # and the two-argument form the earlier call sites use is unchanged
    assert cc._points_variant("free") == "structure_free"


def test_22a2_an_unwired_or_misplaced_design_policy_is_refused_before_any_output(tmp_path):
    """The shared refusals, plus the one this path adds.

    `design_policies.check_policy` is the SHARED rule and is reused verbatim: an
    unregistered name is refused, and ANY non-default policy on an A=0 cell is
    refused because those cells are the fixed-draw anchor. On top of it, the
    three count-bank arms are registered but not WIRED here - a ChemBench rate
    is positive and continuous, so `design_policies.criteria_table`'s bounded-
    count bank does not exist on this path - and saying so at entry beats
    failing inside the estimator.
    """
    manifest = _manifest(tmp_path)
    assert cc.CHEM_DESIGN_POLICIES == (cc.DEFAULT_POLICY, "epig", "point_disagreement", "space_filling")
    for cell in ("000", "001"):
        out = tmp_path / f"misplaced_epig_{cell}"
        with pytest.raises(ValueError, match="epig requires cell"):
            cc.run(_args(out, manifest, cell=cell, design_policy="epig"))
        assert not out.exists()
    for policy in cc.POLICIES:
        if policy in cc.CHEM_DESIGN_POLICIES:
            continue
        out = tmp_path / f"a0_{policy}" / "run"
        with pytest.raises(ValueError, match="needs an acquisition bit"):
            cc.run(_args(out, manifest, cell="000", design_policy=policy))
        assert not out.exists()
        out = tmp_path / f"a1_{policy}" / "run"
        with pytest.raises(ValueError, match="not wired on this positive-rate path"):
            cc.run(_args(out, manifest, cell="001", design_policy=policy))
        assert not out.exists()
    out = tmp_path / "unregistered" / "run"
    with pytest.raises(ValueError, match="unregistered design policy"):
        cc.run(_args(out, manifest, cell="001", design_policy="greedy_variance"))
    assert not out.exists()
    # More rounds than the pool can offer DISTINCT designs is a sizing error,
    # refused at entry rather than found out in round `design_pool + 1`.
    out = tmp_path / "too_many_rounds" / "run"
    with pytest.raises(ValueError, match="design pool holds 32"):
        cc.run(_args(out, manifest, cell="001", rounds=33))
    assert not out.exists()
    # The same run on an A=0 cell is unaffected: it never reads the pool.
    assert cc.PROFILE_GRID["dev"]["design_pool"] == 32


def test_22a2_the_arms_share_d0_and_then_diverge(tmp_path, monkeypatch):
    """The reason the axis exists: `001` and `000` start from the same
    four D0 observations and separate from round 1 on.

    D0 is still the first four entries of the pre-drawn
    `chem_observation_designs` sequence on BOTH cells (the draw's `count` is
    unchanged at `4 + rounds`, so its rows are unchanged), which is what makes
    the two arms comparable; every later design is the policy's on `001` and the
    pre-drawn one on `000`.

    Both arms run on the two-input fake, so the `001` arm keeps
    its pool EIG path and the two still share one world.
    """
    use_two_input_fake(monkeypatch)
    manifest = _manifest(tmp_path)
    fixed, chosen = tmp_path / "fixed" / "run", tmp_path / "chosen" / "run"
    cc.run(_args(fixed, manifest, cell="000", rounds=3))
    cc.run(_args(chosen, manifest, cell="001", rounds=3))

    def observations(run_dir):
        """Return the design keys of the observations in a run directory."""
        return [
            cc._design_key(json.loads(line)["x"])
            for line in (run_dir / "observations.jsonl").read_text().splitlines()
        ]

    left, right = observations(fixed), observations(chosen)
    assert left[:4] == right[:4]
    assert left[4:] != right[4:]
    # The chosen designs come from the pool; the fixed ones do not.
    pool = {
        cc._design_key(row)
        for row in _fake_sample_designs_2d(
            32, cc.seed_for(cc.DESIGN_POOL_STREAM, cc.DESIGN_POOL_SEED)
        )
    }
    assert set(right[4:]) <= pool
    assert not set(left[4:]) & pool
    assert len(set(right[4:])) == 3
    # ... and the fixed arm's own npz artifacts are untouched by the pool being
    # drawn at all: the D0 policy file is identical between the two runs.
    assert (fixed / "frozen_d0_policy.json").read_bytes() == (
        chosen / "frozen_d0_policy.json"
    ).read_bytes()


# ---------------------------------------------------------------------------
# KL(truth || BMA) on the
# official evaluation set. The first four tests exercise `censored_normal_kl_quadrature`
# directly - the "quadrature routine" - with no
# PyMC fit and no `positive_inference.log_predictive` involved at all, since
# its only pluggable seam, `log_q_batch`, accepts any callable. The tests after
# those four (the first of them reusing the points-row-format test) exercise
# the real production wiring inside `run`.
# ---------------------------------------------------------------------------


def test_29_kl_quadrature_matches_closed_form_normal_vs_normal():
    """Quadrature routine vs a closed-form Normal-Normal KL, <=1e-10.

    `nu` sits 100 relative standard deviations from 0 (KL_TRUNCATION_SIGMAS is
    8), so `N+`'s censoring moves no measurable mass and the plain closed form
    KL(N(mu1,s1^2)||N(mu2,s2^2)) = log(s2/s1) + (s1^2+(mu1-mu2)^2)/(2 s2^2) - 1/2
    applies to the quadrature's own truth density.
    """
    nu, relative_sd = 100.0, 0.01
    sigma = relative_sd * nu
    mu2, sigma2 = 101.5, 1.3
    closed_form = np.log(sigma2 / sigma) + (sigma**2 + (nu - mu2) ** 2) / (2 * sigma2**2) - 0.5

    def log_q_batch(y):
        """Log density of the fake predictive at the observations y."""
        return _norm_dist.logpdf(y, loc=mu2, scale=sigma2)

    result = cc.censored_normal_kl_quadrature(
        [nu], relative_sd, log_q_batch, cc.KL_QUADRATURE_BASE_NODES
    )
    assert abs(float(result["kl"][0]) - closed_form) <= 1e-10


def test_29_kl_quadrature_matches_a_fine_riemann_sum_against_a_lognormal_mixture():
    """A 2-component lognormal mixture model density (standing in
    for a realistic BMA), checked against a 1e6-point Riemann sum, <=1e-8.
    """
    nu, relative_sd = 100.0, 0.01
    sigma = relative_sd * nu
    w1, w2 = 0.4, 0.6
    s1, s2 = 0.02, 0.03
    m1, m2 = nu * 0.98, nu * 1.01

    def component_logpdf(y, mean, shape_param):
        """Log density of one lognormal mixture component."""
        return _lognorm_dist.logpdf(y, shape_param, scale=mean * np.exp(-(shape_param**2) / 2))

    def log_q_batch(y):
        """Log density of the two-component lognormal mixture at y."""
        l1 = np.log(w1) + component_logpdf(y, m1, s1)
        l2 = np.log(w2) + component_logpdf(y, m2, s2)
        return np.logaddexp(l1, l2)

    n = cc.KL_QUADRATURE_BASE_NODES
    result = cc.censored_normal_kl_quadrature([nu], relative_sd, log_q_batch, n)

    lo, hi = float(result["lo"][0]), float(result["hi"][0])
    yy = np.linspace(lo, hi, 1_000_000)
    dy = yy[1] - yy[0]
    log_p_fine = _norm_dist.logpdf(yy, loc=nu, scale=sigma)
    riemann = float(np.sum(np.exp(log_p_fine) * (log_p_fine - log_q_batch(yy))) * dy)

    assert abs(float(result["kl"][0]) - riemann) <= 1e-8


def test_29_kl_is_near_zero_when_the_model_density_equals_the_truth_density():
    """A fake pool with the true density AS a component.

    No lognormal/gamma component can equal a Normal exactly, so "the pool
    recovered the truth" is injected at `official_kl_summary`'s only seam,
    `log_q_batch`, as the literal truth log-density; KL must be ~0 (<=1e-12),
    not merely small, at several node counts.
    """
    nu, relative_sd = 57.0, 0.01

    def log_q_batch(y):
        """Log density of the relative-sd normal predictive at y."""
        return _norm_dist.logpdf(y, loc=nu, scale=relative_sd * nu)

    for n in (8, cc.KL_QUADRATURE_BASE_NODES, 2 * cc.KL_QUADRATURE_BASE_NODES):
        result = cc.censored_normal_kl_quadrature([nu], relative_sd, log_q_batch, n)
        assert abs(float(result["kl"][0])) <= 1e-12


def test_29_truncated_mass_and_atom_mass_are_genuine_computed_values():
    """`kl_truncated_mass` is a real, present VALUE - never a blank
    or a hardcoded 0 - and matches the 8-sigma window's own closed-form tail
    mass `2*(1-Phi(KL_TRUNCATION_SIGMAS))`. The y=0 atom mass, `Phi(-1/noise)`,
    is likewise computed via `scipy.stats.norm.logcdf` (never hardcoded) and
    happens to underflow to exactly 0.0 at noise=0.01 - the log-space
    computation behind it is finite and very negative, not skipped.
    """
    relative_sd = 0.01
    nu = np.array([12.5, 340.0, 0.003])  # scale-invariant: identical relative window

    def log_q_batch(y):
        """Batched log density of the relative-sd normal predictive at y."""
        return _norm_dist.logpdf(y, loc=nu[:, None], scale=(relative_sd * nu)[:, None])

    result = cc.censored_normal_kl_quadrature(
        nu, relative_sd, log_q_batch, cc.KL_QUADRATURE_BASE_NODES
    )
    expected_truncated = 2.0 * (1.0 - _norm_dist.cdf(cc.KL_TRUNCATION_SIGMAS))
    assert expected_truncated > 0.0
    for value in result["truncated_mass"]:
        assert isinstance(float(value), float)
        assert abs(float(value) - expected_truncated) <= 1e-18
    for value in result["atom_mass"]:
        assert float(value) == 0.0
    log_atom = _norm_dist.logcdf(-1.0 / relative_sd)
    assert np.isfinite(log_atom) and log_atom < -1000.0


def test_29_official_kl_wiring_points_row_config_summary_and_recompute_artifact(tmp_path):
    """The production wiring of the official KL in one real (fake-session) run
    (reusing the points-row-format test): `kl` on the FINAL
    points.csv row only, the config.json/summary.json disclosure keys, and an
    INDEPENDENT recomputation of `kl` using ONLY the recompute artifact
    (`official_kl_particle_summary_after_finish.npz`) plus
    `official_evaluation_inputs.npz`/`official_reference_after_finish.npz` -
    never the live `fits` objects `run` itself used - confirming the artifact
    actually supports recomputation, not merely storage.
    """
    manifest = _manifest(tmp_path)
    out = tmp_path / "run"
    cc.run(_args(out, manifest, cell="000", structure_proposals="catalog", rounds=2))

    header, rows = _points_rows(out)
    assert header == list(cc.CONTRACT_POINTS_COLUMNS)
    assert len(rows) == 3  # t = 0, 1, 2
    for row in rows[:-1]:
        assert row["kl"] == ""
    assert rows[-1]["kl"] not in ("", None)
    kl_from_points = float(rows[-1]["kl"])

    config = json.loads((out / "config.json").read_text())
    assert config["kl_truth_center"] == "noiseless_observation"
    assert config["nmse_mda_target"] == "primary_rate"
    assert config["kl_truth_density"]["family"] == "relative_normal_truncated_at_0"
    assert config["kl_truth_density"]["relative_sd"] == config["noise"]

    summary = json.loads((out / "summary.json").read_text())
    assert summary["kl_truth_center"] == "noiseless_observation"
    assert summary["nmse_mda_target"] == "primary_rate"
    assert summary["kl_n_nodes"] == 2 * cc.KL_QUADRATURE_BASE_NODES
    assert summary["kl"] == pytest.approx(kl_from_points, abs=1e-9)
    assert 0.0 <= summary["kl_quadrature_error"] < 1e-6
    assert 0.0 <= summary["kl_truncated_mass"] < 1e-10
    per_q = summary["kl_per_q_summary"]
    assert per_q["min"] <= per_q["median"] <= per_q["max"]

    # The recompute artifact.
    artifact = np.load(out / "official_kl_particle_summary_after_finish.npz")
    assert str(artifact["format"]) == "particle_mean_scale"
    families = artifact["families"]
    mean, scale, weights = artifact["mean"], artifact["scale"], artifact["weights"]
    n_candidates, n_particles = mean.shape[0], mean.shape[1]
    assert scale.shape == (n_candidates, n_particles)
    assert weights.shape == (n_candidates,)
    assert abs(float(weights.sum()) - 1.0) <= 1e-9

    official_inputs = np.load(out / "official_evaluation_inputs.npz")
    n_q = official_inputs["x"].shape[0]
    assert n_q == cc.OFFICIAL_EVALUATION_COUNT
    assert mean.shape == (n_candidates, n_particles, n_q)

    official_reference = np.load(out / "official_reference_after_finish.npz")
    noiseless = np.asarray(official_reference["noiseless_observation"], dtype=float)
    noise = config["noise"]

    def log_q_from_artifact(y):
        """Rebuilds `log_predictive`'s double mixture from ONLY the artifact."""
        n_nodes = y.shape[1]
        components = np.empty((n_candidates, n_q, n_nodes))
        for c in range(n_candidates):
            m = mean[c][:, :, None]  # [particle, query, 1]
            s = scale[c][:, None, None]  # [particle, 1, 1]
            yy = y[None, :, :]  # [1, query, node]
            if families[c] == "lognormal":
                conditional = _lognorm_dist.logpdf(yy, s, scale=m * np.exp(-(s**2) / 2))
            else:
                conditional = _gamma_dist.logpdf(yy, s, scale=m / s)
            components[c] = _logsumexp(conditional, axis=0) - np.log(n_particles)
        return _logsumexp(components + np.log(weights)[:, None, None], axis=0)

    base_n = cc.KL_QUADRATURE_BASE_NODES
    fine = cc.censored_normal_kl_quadrature(noiseless, noise, log_q_from_artifact, 2 * base_n)
    recomputed_kl = float(np.mean(fine["kl"]))
    assert recomputed_kl == pytest.approx(summary["kl"], abs=1e-9)


def test_29_particle_summary_falls_back_to_quadrature_nodes_when_oversized(tmp_path):
    """The size gate genuinely switches format; it is not always the small
    (particle_mean_scale) representation regardless of the projected size.
    """
    from bayes_model_discovery.experimental.positive_inference import RegressionHypothesis

    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 1.0, size=(20, 3))
    y = np.maximum(np.exp(0.2 + 0.1 * x[:, 0]) * (1.0 + 0.01 * rng.standard_normal(20)), 1e-6)
    transform = cc.InputTransform.from_public_pool(x)
    result = cc.fit(
        RegressionHypothesis("lognormal", ()),
        x,
        y,
        123,
        transform,
        float(y.mean()),
        settings=cc.SMCSettings(16, 2, 2),
    )
    fits = [result]
    official_x = rng.uniform(0.1, 1.0, size=(50, 3))
    noiseless = np.exp(0.2 + 0.1 * official_x[:, 0])
    kl_summary = cc.official_kl_summary(fits, official_x, noiseless, 0.01)

    small_cap = cc.official_kl_particle_summary(
        fits, official_x, kl_summary["quadrature"], max_bytes=1_000_000_000
    )
    assert small_cap["format"] == "particle_mean_scale"

    forced = cc.official_kl_particle_summary(
        fits, official_x, kl_summary["quadrature"], max_bytes=1
    )
    assert forced["format"] == "mixture_log_density_at_quadrature_nodes"
    assert forced["nodes"].shape == forced["log_density"].shape
    assert forced["projected_bytes"] > forced["max_bytes"]


def test_29_frozen_adapter_is_byte_identical_to_the_baseline_hash():
    """The frozen-adapter requirement: byte-identical, always.

    The digest is pinned to this tree's copy of the adapter, whose docstrings are in English;
    its syntax tree is identical to the development tree's (`tools/release/ast_equiv.py`).
    """
    import hashlib

    path = Path(cc.__file__).resolve().parents[1] / "adapters" / "chembench_observation.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == "9097d8cdb6f875f5bf42463e2acca5da6d99724197ec4bee12aa22b800f15459"


# ---------------------------------------------------------------------------
# The shared `core_count.points_label`
# helper - a finding from an earlier smoke run. `chembench_core`'s own
# earlier rule ("development" if args.profile == "dev" or args.proposer ==
# "openai" else "production") mislabelled a paper-profile MOCK run as
# "production" - exactly what a paper x ChemBench acquisition smoke run
# (profile paper, proposer mock, family lognormal, noise 0.01,
# structure-proposals catalog) recorded. Two things are pinned: `points_label` is
# not reimplemented locally, and the paper+mock flip itself.
# ---------------------------------------------------------------------------


def test_22a3_points_label_is_the_shared_helper_not_a_reimplementation():
    """`chembench_core.points_label` is not a second, independently
    maintained copy of the rule - it is the exact same function object
    `core_count` defines (the import binds the name, nothing wraps it)."""
    from bayes_model_discovery.experimental import core_count

    assert cc.points_label is core_count.points_label


def test_22a3_paper_profile_mock_proposer_is_development_not_production(tmp_path):
    """A paper-profile MOCK run's `points_label` is
    "development", not "production" - the exact defect a smoke run
    found. This reproduces the smoke's config at a tiny scale (fake adapter,
    small particles/q/rounds, cell "000" so no acquisition cost) - profile
    and proposer are the only two fields `points_label` reads, so cutting
    every other dimension down for speed cannot change the answer."""
    manifest = _manifest(tmp_path)
    out = tmp_path / "paper_mock"
    cc.run(_args(out, manifest, profile="paper", proposer="mock"))
    config = json.loads((out / "config.json").read_text())
    assert config["profile"] == "paper"
    assert config["proposer"] == "mock"
    assert config["points_label"] == "development"
    _, points = _points_rows(out)
    assert points  # the run produced at least one point row to check
    assert {row["label"] for row in points} == {"development"}


# ---------------------------------------------------------------------------
# `--run-label` on chembench_core (all-benchmark 2 h smoke).
# ---------------------------------------------------------------------------


def test_18_chem_refuses_run_label_production_before_any_output(tmp_path):
    """REFUSE: a paper-profile MOCK run asking for "production" is refused at
    run() entry - before `out.mkdir`, so nothing is left behind."""
    manifest = _manifest(tmp_path)
    out = tmp_path / "should_not_exist"
    with pytest.raises(ValueError, match="--run-label production"):
        cc.run(_args(out, manifest, profile="paper", proposer="mock", run_label="production"))
    assert not out.exists()


def test_18_chem_accepts_run_label_development_and_records_it(tmp_path):
    """ACCEPT: development always runs; the chosen label reaches config.json
    (both the `run_label` CLI echo and the `points_label` key this runner
    already published) and every points.csv row."""
    manifest = _manifest(tmp_path)
    out = tmp_path / "dev_label"
    cc.run(_args(out, manifest, profile="paper", proposer="mock", run_label="development"))
    config = json.loads((out / "config.json").read_text())
    assert config["run_label"] == "development"
    assert config["points_label"] == "development"
    _, points = _points_rows(out)
    assert points and {row["label"] for row in points} == {"development"}


# ---------------------------------------------------------------------------
# F0 factorial: the first two cell bits move
# independently, so the F0 structure-expansion route `100`/`101` exists - the
# proposer expands the mean function while the likelihood family stays the one
# `--family` designates. The bits mean here what they mean in
# `boxinggym_core`/`core_count`; there is no ChemBench-specific wording.
#
# Anchor invariance for `000`/`001`/`110`/`111` is asserted OUTSIDE this file,
# against the run trees the smoke recorded; what the tests below pin is that the four newly
# reachable cells behave, and that the four old ones are reached by the same
# command lines as before.
# ---------------------------------------------------------------------------


def test_19_all_eight_cells_are_legal_and_cell0_alone_decides_the_structure_mode():
    """The binding to `--structure-proposals` is gone: cell[0] decides.

    `structure_mode` is the single reading of the bit, the flag is an optional
    override that must AGREE (so every earlier command line stays valid and a
    contradicting one is still refused), and omitting it means "take the cell".
    """
    assert cc.CELL_CHOICES == ("000", "001", "002", "010", "011", "012",
                               "100", "101", "102", "110", "111", "112")
    for cell in cc.CELL_CHOICES:
        expected = "free" if cell[0] == "1" else "catalog"
        assert cc.structure_mode(cell) == expected
        # omitted -> derived; stated and agreeing -> accepted, cell returned
        assert cc.validate_cell(cell) == cell
        assert cc.validate_cell(cell, expected) == cell
        other = "catalog" if expected == "free" else "free"
        with pytest.raises(ValueError, match="inconsistent with --structure-proposals"):
            cc.validate_cell(cell, other)


def test_19_cell_100_runs_the_proposer_and_refuses_out_of_family_candidates_one_by_one(
    tmp_path,
):
    """The F0 route itself, on the mock path, with `--structure-proposals` NOT
    stated - the shape a `100` cell is actually launched in.

    The fixed mock schedule alternates the two task families
    (`_mock_schedule`: gamma, lognormal, gamma, lognormal for rounds 1..4), so
    one four-round run exercises both outcomes. Each gamma candidate is refused
    ALONE - the round still proposes, the round still runs, and the lognormal
    rounds still admit and fit - which is the per-candidate rule of an earlier experiment phase, not a
    batch refusal.
    """
    manifest = _manifest(tmp_path)
    out = tmp_path / "f0_100"
    cc.run(_args(out, manifest, cell="100", rounds=4, structure_proposals=None))

    config = json.loads((out / "config.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    # The RECORDED mode is the EXECUTED mode, derived from cell[0].
    assert config["structure_proposals"] == "free"
    assert config["cell"] == summary["cell"] == "100"
    assert config["points_variant"] == "structure_free"
    # A=0, so no acquisition record and the pre-drawn design sequence.
    assert config["design_axis"] == cc.DESIGN_AXIS_ABSENT
    assert not (out / "acquisition").exists()

    schedule = cc._mock_schedule(list(FAKE_NAMES), cc.CHEM_ALLOWED_FAMILIES)
    families = [schedule[t - 1][1] for t in range(1, 5)]
    assert families == ["gamma", "lognormal", "gamma", "lognormal"]

    admitted, refused = [], []
    for t in range(1, 5):
        record = json.loads((out / "proposals" / f"t{t:02d}.json").read_text())
        assert len(record["proposed_ids"]) == 1, t  # the proposer DID run every round
        (admitted if record["admitted_ids"] else refused).append(t)
    assert admitted == [2, 4] and refused == [1, 3]

    # The refusal is recorded, per candidate, in the same words the LLM path's
    # own judge uses - and the archive says which model and which family.
    archive = json.loads((out / "packets" / "t04.json").read_text())["archive"]
    rejected = [e for e in archive if e["outcome"] == "rejected"]
    assert [e["family"] for e in rejected] == ["gamma", "gamma"]
    assert [e["expression"] for e in rejected] == [schedule[0][0], schedule[2][0]]
    assert {e["reason"]["validation_message"] for e in rejected} == {
        "F0 requires the initial likelihood family"
    }
    assert [e["proposed_round"] for e in rejected] == [1, 3]
    # ... and the admitted ones really are the designated family, really fitted.
    active = [e for e in archive if e["outcome"] == "active"]
    assert [e["family"] for e in active] == ["lognormal"]
    assert summary["active_pool"] == 4 + 2  # the 4-member catalogue plus 2 admissions
    assert summary["proposals"] == 4
    assert summary["proposal_accounting"]["admitted_candidates"] == 2
    assert len(list((out / "fits").glob("*.npz"))) > 0

    # The packet the proposer is handed states the same restriction up front.
    constraints = json.loads((out / "packets" / "t01.json").read_text())["constraints"]
    assert constraints["structure_expansion"] == 1
    assert constraints["family_expansion"] == 0
    assert constraints["allowed_families"] == ["lognormal"]
    assert constraints["initial_family"] == "lognormal"


def test_19_cell_101_is_the_same_route_with_the_acquisition_bit_on(tmp_path, monkeypatch):
    """`101` = `100` plus the third bit: the same F0 admissions and refusals,
    and the acquisition record the A=1 cells write.

    Both cells run on the two-input fake. Its catalogue
    already holds the two linear forms the mock schedule offers on the
    lognormal rounds, so the per-round outcome is compared with the same fake's
    `100` run instead of being pinned to the eight-input admissions [2, 4]
    (still pinned by test_19_cell_100 above)."""
    use_two_input_fake(monkeypatch)
    manifest = _manifest(tmp_path)
    runs = {cell: tmp_path / f"f0_{cell}" for cell in ("100", "101")}
    for cell, run_dir in runs.items():
        cc.run(_args(run_dir, manifest, cell=cell, rounds=4, structure_proposals=None))
    out = runs["101"]

    config = json.loads((out / "config.json").read_text())
    assert config["structure_proposals"] == "free"
    assert config["design_axis"] == "joint_eig"
    assert config["design_policy_applies"] is True
    assert (out / "acquisition" / "t01.json").is_file()
    assert not (runs["100"] / "acquisition").exists()

    constraints = json.loads((out / "packets" / "t01.json").read_text())["constraints"]
    assert constraints["family_expansion"] == 0
    assert constraints["allowed_families"] == ["lognormal"]

    def outcomes(run_dir):
        """Return the per-round proposal records of a run directory."""
        records = [json.loads((run_dir / "proposals" / f"t{t:02d}.json").read_text())
                   for t in range(1, 5)]
        return [(record["proposed_ids"], record["admitted_ids"]) for record in records]

    assert outcomes(out) == outcomes(runs["100"])
    assert all(len(proposed) == 1 for proposed, _ in outcomes(out))
    # The gamma rounds are refused one candidate at a time, on 101 as on 100.
    archive = json.loads((out / "packets" / "t04.json").read_text())["archive"]
    rejected = [e for e in archive if e["outcome"] == "rejected"]
    assert [e["proposed_round"] for e in rejected] == [1, 3]
    assert [e["family"] for e in rejected] == ["gamma", "gamma"]
    assert {e["reason"]["validation_message"] for e in rejected} == {
        "F0 requires the initial likelihood family"
    }
    _, points = _points_rows(out)
    assert {row["method"] for row in points} == {"101"}
    assert {row["variant"] for row in points} == {"structure_free"}


def test_19_the_family_bit_is_the_only_difference_between_100_and_110(tmp_path):
    """The two cells differ in exactly the family bit, so the same schedule
    admits 4 candidates on `110` and 2 on `100`, and the two admissions `100`
    does make are the same models `110` makes."""
    manifest = _manifest(tmp_path)
    f0, f1 = tmp_path / "cell_100", tmp_path / "cell_110"
    cc.run(_args(f0, manifest, cell="100", rounds=4, structure_proposals=None))
    cc.run(_args(f1, manifest, cell="110", rounds=4, structure_proposals="free"))

    def admitted(out):
        """Return the admitted candidate ids per round of a run."""
        return {
            t: json.loads((out / "proposals" / f"t{t:02d}.json").read_text())["admitted_ids"]
            for t in range(1, 5)
        }

    a0, a1 = admitted(f0), admitted(f1)
    assert all(a1[t] for t in (1, 2, 3, 4))
    assert a0[1] == [] and a0[3] == []
    assert a0[2] == a1[2] and a0[4] == a1[4]
    assert json.loads((f0 / "summary.json").read_text())["active_pool"] == 6
    assert json.loads((f1 / "summary.json").read_text())["active_pool"] == 8


def test_19_the_f0_refusal_is_the_shared_count_models_rule_not_a_local_copy():
    """`family_refusal` is `count_models.judge_candidate`'s F0 branch: the same
    predicate and the same reason string, taken from the same constant, so the
    two cannot drift into two different words for one rule."""
    from bayes_model_discovery.experimental import count_models

    assert cc.F0_REFUSAL_REASON is count_models.F0_REFUSAL_REASON
    assert count_models.F0_REFUSAL_REASON == "F0 requires the initial likelihood family"
    # The judge still returns exactly this for an out-of-family candidate.
    ok, reason = count_models.judge_candidate(
        {"expression": "b0", "family": "zi_binomial", "parameters": {"b0": "intercept"}},
        "100",
        count_models.initial_hypotheses("binomial"),
        "binomial",
    )
    assert ok is False and reason == count_models.F0_REFUSAL_REASON
    # ... and the runner's own per-candidate form agrees on every combination.
    for cell in cc.CELL_CHOICES:
        assert cc.family_refusal(cell, "lognormal", "lognormal") is None
        expected = None if cell[1] == "1" else count_models.F0_REFUSAL_REASON
        assert cc.family_refusal(cell, "gamma", "lognormal") == expected


def test_19_paid_api_calls_in_summary_comes_from_the_proposer_on_an_f0_cell(tmp_path, monkeypatch):
    """`summary.json`'s `paid_api_calls` is the proposer's own attempt count on
    a `100` cell, not a literal 0 - asserted through the real transport with an
    INJECTED FAKE client (no credentials, no network, no real request), the
    same construction `test_chem_evaluation_designs_never_reach_the_packet_or_
    proposer` uses. The fake answers `gamma`, which this cell must refuse one
    at a time while still counting the call."""
    manifest = _manifest(tmp_path)
    calls = []

    def create(**body):
        """Fake client call: record the request body and return a canned reply."""
        calls.append(body)
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text=json.dumps(
                        {
                            "expression": "b0+b1*f0z",
                            "family": "gamma",
                            "parameters": {"b0": "intercept", "b1": "signed"},
                        }
                    ),
                )
            ],
            usage=SimpleNamespace(
                model_dump=lambda **kwargs: {"input_tokens": 1, "output_tokens": 1}
            ),
            id="test-only-response",
            model="test-only-model",
            stop_reason="end_turn",
        )

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(
        count_proposer,
        "AnthropicCountProposer",
        lambda records, **kwargs: AnthropicCountProposer(records, client=fake_client),
    )

    out = tmp_path / "f0_llm"
    cc.run(
        _args(
            out, manifest, cell="100", rounds=2, proposer="anthropic",
            allow_paid_api=True, structure_proposals=None,
        )
    )
    summary = json.loads((out / "summary.json").read_text())
    assert calls, "the proposer must actually have been called (real wiring, fake client)"
    assert summary["paid_api_calls"] == len(calls) > 0
    assert summary["proposal_accounting"]["llm_calls"] == len(calls)
    assert summary["billing_status"] == "usage_in_llm_transport_records"

    # The out-of-family candidate never entered the pool, and the refusal is
    # recorded per candidate by the shared judge, in the shared words.
    record = json.loads((out / "proposals" / "t01.json").read_text())
    assert record["admitted_ids"] == []
    # One entry per attempt: the transport's repair-once contract asks again
    # after the judge refuses, and this fake answers gamma both times, so every
    # entry carries the same per-candidate F0 refusal - never one batch verdict.
    # (`propose_one`'s own rejection shape is
    # {attempt, reason, error_type, http_status, repair_eligible} - count_
    # proposer.py:810-814 - so the family is read off the reason, not a key.)
    rejected = record["transport"]["rejected_candidates"]
    assert rejected
    assert {e["reason"] for e in rejected} == {"F0 requires the initial likelihood family"}
    assert {e["error_type"] for e in rejected} == {"ValueError"}
    assert summary["active_pool"] == 4  # the catalogue alone; nothing was admitted


# ---------------------------------------------------------------------------
# The paid ChemBench horizon moved from 40 to 60 rounds. The CLI bound is now derived from the profile grid rather than
# being a literal, so nothing below re-states "60"; the run-level guards that
# decide whether a horizon is actually runnable are unchanged and still first.
# ---------------------------------------------------------------------------

def _rounds_argv(rounds):
    return ["chembench_core", "--output", "/dev/null/never", "--manifest", "m",
            "--noise", "0.01", "--rounds", str(rounds)]


def test_the_rounds_cli_bound_is_the_largest_design_pool_not_a_literal(monkeypatch):
    """60 is admitted, and the bound follows PROFILE_GRID rather than a number."""
    import sys as _sys

    bound = max(profile["design_pool"] for profile in cc.PROFILE_GRID.values())
    assert bound == 256
    for rounds in (1, 40, 60, bound):
        captured = []
        monkeypatch.setattr(cc, "run", captured.append)
        monkeypatch.setattr(_sys, "argv", _rounds_argv(rounds))
        cc.main()
        (args,) = captured
        assert args.rounds == rounds
    for rounds in (0, -1, bound + 1):
        monkeypatch.setattr(cc, "run", lambda args: pytest.fail("run() must not be reached"))
        monkeypatch.setattr(_sys, "argv", _rounds_argv(rounds))
        with pytest.raises(SystemExit):
            cc.main()


def test_a_60_round_horizon_reaches_the_proposer_as_its_own_window(tmp_path):
    """The transport accepts round 41..60 when the runner hands it max_rounds=60.

    Same shape as the test above, moved out to 60: the horizon belongs to
    the run, not to the transport - `DEFAULT_MAX_PROPOSAL_ROUNDS` is untouched
    and still refuses those rounds. Injected fake client only.
    """
    assert count_proposer.DEFAULT_MAX_PROPOSAL_ROUNDS == 20
    packet = json.dumps({"round": 1})
    seed = cc.ScalarHypothesis("b0", "lognormal", (("b0", "intercept"),), ("f0", "f0z"))
    client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **body: _fake_anthropic_response())
    )

    def context(round_number):
        """Return a minimal run context for the given round."""
        return {"run_id": "test-only", "world_seed": 11, "cell": "110",
                "proposal_rep": 0, "round": round_number}

    for round_number in (41, 60):
        proposer = AnthropicCountProposer(tmp_path / f"r{round_number}", client=client)
        _, record = proposer.propose_one(
            packet, context=context(round_number), initial_family="lognormal",
            initial_structures=(seed,), allowed_families=cc.CHEM_ALLOWED_FAMILIES,
            max_rounds=60,
        )
        assert record["status"] == "parsed" and record["round"] == round_number
        refused = AnthropicCountProposer(tmp_path / f"refused{round_number}", client=client)
        with pytest.raises(ValueError, match="round identity required"):
            refused.propose_one(
                packet, context=context(round_number), initial_family="lognormal",
                initial_structures=(seed,), allowed_families=cc.CHEM_ALLOWED_FAMILIES,
            )
    over = AnthropicCountProposer(tmp_path / "over", client=client)
    with pytest.raises(ValueError, match="round identity required"):
        over.propose_one(
            packet, context=context(61), initial_family="lognormal",
            initial_structures=(seed,), allowed_families=cc.CHEM_ALLOWED_FAMILIES,
            max_rounds=60,
        )


def test_the_run_level_round_guards_are_unchanged(monkeypatch):
    """Widening the CLI must not widen the two checks that refuse a real run."""
    assert cc.MOCK_SCHEDULE_ROUNDS == 20
    assert cc.PROFILE_GRID["dev"]["design_pool"] == 32
    assert cc.PROFILE_GRID["paper"]["design_pool"] == 256


# ---------------------------------------------------------------------------
# K2: `--kl-every-checkpoint` (scoring output only) and `pool_weights.jsonl`.
# ---------------------------------------------------------------------------

def _npz_contents(path):
    """Every array of an .npz, as (name -> dtype, shape, raw bytes).

    `np.savez_compressed` stamps each zip member with the wall-clock time of
    the write, so two runs seconds apart never produce identical FILE bytes;
    this is the identity that actually matters - the same names, the same
    dtypes, the same shapes, the same bits in every array.
    """
    with np.load(path) as data:
        return {name: (data[name].dtype.str, data[name].shape, data[name].tobytes())
                for name in data.files}


def _pool_weight_lines(run_dir):
    return [json.loads(line)
            for line in (run_dir / "pool_weights.jsonl").read_text().splitlines()]


def test_k2_kl_every_checkpoint_scores_every_row_and_leaves_the_final_value_identical(
    tmp_path, monkeypatch
):
    """With the option on, `kl` is filled on EVERY checkpoint row and
    is that round's own pool re-scored (recomputed here directly, from the
    run's published official inputs/truth and the pool the run handed the
    scorer); with it off, exactly one call is made and only the last row
    carries a value. The final row, summary.json's whole `kl` family and the
    recompute artifact are identical between the two runs - the option
    widens which rows are scored and changes nothing about the final one.
    """
    calls = []
    real = cc.official_kl_summary

    def recording(fits, official_x, noiseless, relative_sd, **kwargs):
        """Record the fit pool and KL, then delegate to the real function."""
        result = real(fits, official_x, noiseless, relative_sd, **kwargs)
        calls.append({"fits": list(fits), "kl": result["kl"]})
        return result

    monkeypatch.setattr(cc, "official_kl_summary", recording)
    manifest = _manifest(tmp_path)
    off, on = tmp_path / "off", tmp_path / "on"
    cc.run(_args(off, manifest, rounds=3))
    off_calls = len(calls)
    cc.run(_args(on, manifest, rounds=3, kl_every_checkpoint=True))
    on_calls = calls[off_calls:]

    assert off_calls == 1  # the final pool alone, as before
    assert len(on_calls) == 4  # t = 0, 1, 2, 3

    off_header, off_rows = _points_rows(off)
    on_header, on_rows = _points_rows(on)
    assert on_header == off_header == list(cc.CONTRACT_POINTS_COLUMNS)
    assert [row["kl"] for row in off_rows[:-1]] == ["", "", ""]
    assert all(row["kl"] not in ("", None) for row in on_rows)
    assert len({row["kl"] for row in on_rows}) > 1  # genuinely per-round, not copied

    inputs = np.load(on / "official_evaluation_inputs.npz")["x"]
    truth = np.load(on / "official_reference_after_finish.npz")
    noiseless = np.asarray(truth[cc.OFFICIAL_KL_TRUTH_CENTER], dtype=float)
    noise = json.loads((on / "config.json").read_text())["noise"]
    for row, call in zip(on_rows, on_calls, strict=True):
        # The pool handed to the scorer is the pool of THAT round: it was fit
        # on exactly the observations the row reports.
        assert call["fits"][0].diagnostics["observation_count"] == int(row["x"])
        assert float(row["kl"]) == real(call["fits"], inputs, noiseless, noise)["kl"]

    assert on_rows[-1]["kl"] == off_rows[-1]["kl"]
    off_summary = json.loads((off / "summary.json").read_text())
    on_summary = json.loads((on / "summary.json").read_text())
    for key in ("kl", "kl_quadrature_error", "kl_truncated_mass", "kl_n_nodes",
                "kl_per_q_summary", "kl_truth_center", "nmse_mda_target"):
        assert on_summary[key] == off_summary[key]
    artifact = "official_kl_particle_summary_after_finish.npz"
    assert _npz_contents(on / artifact) == _npz_contents(off / artifact)

    # The two auxiliary values reach official_scores_after_finish.json only.
    on_scores = json.loads((on / "official_scores_after_finish.json").read_text())
    off_scores = json.loads((off / "official_scores_after_finish.json").read_text())
    for entry in on_scores["rounds"]:
        assert {"kl", "kl_quadrature_error", "kl_truncated_mass"} <= set(entry)
    assert [set(entry) & {"kl", "kl_quadrature_error", "kl_truncated_mass"}
            for entry in off_scores["rounds"]] == [set(), set(), set(), {"kl"}]
    assert on_scores["rounds"][-1]["kl"] == off_scores["rounds"][-1]["kl"]


def test_k2_an_omitted_option_leaves_the_namespace_config_and_metric_text_untouched(
    tmp_path, monkeypatch
):
    """Items 2 and 3: `default=argparse.SUPPRESS` keeps the flag out of
    `vars(args)`, so an unstated option adds no config.json key at all, and
    the `metric_definitions.kl` sentence is the pre-K2 one; stating it records
    the key and switches the sentence.
    """
    import sys as _sys

    captured = []
    argv = ["chembench_core", "--output", "/dev/null/never", "--manifest", "m",
            "--noise", "0.01"]
    # A scoped context: the autouse adapter fixture shares this test's
    # `monkeypatch`, so these two patches must be lifted on their own.
    with monkeypatch.context() as scoped:
        scoped.setattr(cc, "run", captured.append)
        scoped.setattr(_sys, "argv", argv)
        cc.main()
        assert "kl_every_checkpoint" not in vars(captured[-1])
        scoped.setattr(_sys, "argv", [*argv, "--kl-every-checkpoint"])
        cc.main()
        assert vars(captured[-1])["kl_every_checkpoint"] is True

    manifest = _manifest(tmp_path)
    off, on = tmp_path / "off", tmp_path / "on"
    cc.run(_args(off, manifest, rounds=1))
    cc.run(_args(on, manifest, rounds=1, kl_every_checkpoint=True))

    assert "kl_every_checkpoint" not in json.loads((off / "config.json").read_text())
    assert json.loads((on / "config.json").read_text())["kl_every_checkpoint"] is True

    off_text = json.loads(
        (off / "official_scores_after_finish.json").read_text()
    )["metric_definitions"]["kl"]
    on_text = json.loads(
        (on / "official_scores_after_finish.json").read_text()
    )["metric_definitions"]["kl"]
    assert "defined ONLY for the FINAL checkpoint row" in off_text
    assert "kl_every_checkpoint" not in off_text
    assert "defined for EVERY checkpoint row (kl_every_checkpoint)" in on_text
    assert "defined ONLY for the FINAL checkpoint row" not in on_text
    # Nothing else in the definition block moved.
    off_block = json.loads(
        (off / "official_scores_after_finish.json").read_text()
    )["metric_definitions"]
    on_block = json.loads(
        (on / "official_scores_after_finish.json").read_text()
    )["metric_definitions"]
    assert {k: v for k, v in on_block.items() if k != "kl"} == {
        k: v for k, v in off_block.items() if k != "kl"
    }


def test_k2_an_undefined_truth_center_stays_a_string_on_every_scored_row(tmp_path, monkeypatch):
    """A non-positive truth center makes `official_kl_summary` return
    the `"undefined_truth_center"` STRING instead of a number. That string has
    to survive `write_points`' blank/str rule on every row the option scores -
    never becoming a blank, a 0 or a crash - and the recompute npz stays unwritten
    because there is no quadrature to record.
    """
    class _UndefinedCenterSession(_FakeSession):
        def reference_values(self, designs):
            """Return the base reference values with an undefined center injected."""
            values = super().reference_values(designs)
            centered = np.array(values["noiseless_observation"], dtype=float)
            centered[0] = 0.0  # a single non-positive center is enough
            return {**values, "noiseless_observation": centered}

    monkeypatch.setattr(real_adapter, "ChemBenchSession", _UndefinedCenterSession)
    manifest = _manifest(tmp_path)
    off, on = tmp_path / "off", tmp_path / "on"
    cc.run(_args(off, manifest, rounds=2))
    cc.run(_args(on, manifest, rounds=2, kl_every_checkpoint=True))

    _, off_rows = _points_rows(off)
    _, on_rows = _points_rows(on)
    assert [row["kl"] for row in off_rows] == ["", "", "undefined_truth_center"]
    assert [row["kl"] for row in on_rows] == ["undefined_truth_center"] * 3
    for run_dir in (off, on):
        assert not (run_dir / "official_kl_particle_summary_after_finish.npz").exists()
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["kl"] == "undefined_truth_center"
        assert summary["kl_status"] == "undefined_truth_center"
        assert summary["kl_undefined_truth_center_points"] == [0]
    on_scores = json.loads((on / "official_scores_after_finish.json").read_text())
    for entry in on_scores["rounds"]:
        assert entry["kl"] == "undefined_truth_center"
        assert entry["kl_quadrature_error"] is None
        assert entry["kl_truncated_mass"] is None


@pytest.mark.parametrize(
    ("mode", "cell"), [("catalog", "000"), ("free", "110")]
)
def test_k2_pool_weights_jsonl_records_every_checkpoints_whole_pool(tmp_path, mode, cell):
    """The round-by-round pool record ChemBench had no equivalent of.

    Written unconditionally (the option is not consulted), one line per
    checkpoint in checkpoint order, each line carrying the BMA weights the
    round actually mixed with. The last line's argmax is the final pool's
    single best model: summary.json has no `single_model_id` of its own on
    this runner, so the rule falls through to the maximum-log-evidence member
    of the final pool - which is the same model, because the weights ARE the
    normalized exponential of those log evidences.
    """
    manifest = _manifest(tmp_path)
    out = tmp_path / "run"
    cc.run(_args(out, manifest, rounds=2, structure_proposals=mode, cell=cell))

    lines = _pool_weight_lines(out)
    scores = json.loads((out / "official_scores_after_finish.json").read_text())
    assert len(lines) == len(scores["rounds"]) == 3  # t = 0, 1, 2
    assert [line["round"] for line in lines] == [entry["round"] for entry in scores["rounds"]]
    for line, entry in zip(lines, scores["rounds"], strict=True):
        assert line["observation_count"] == entry["observation_count"]
        assert line["active_pool"] == entry["active_pool"] == len(line["models"])
        assert abs(sum(model["weight"] for model in line["models"]) - 1.0) <= 1e-12
        for model in line["models"]:
            assert set(model) == {"model_id", "structure", "logZ", "weight"}
            assert isinstance(model["structure"], str) and model["structure"]
            assert np.isfinite(model["logZ"]) and 0.0 <= model["weight"] <= 1.0

    summary = json.loads((out / "summary.json").read_text())
    final = lines[-1]
    by_weight = max(final["models"], key=lambda m: m["weight"])
    assert final["single_model_id"] == by_weight["model_id"]
    if summary.get("single_model_id") is not None:
        assert final["single_model_id"] == summary["single_model_id"]
    else:
        assert final["single_model_id"] == max(
            final["models"], key=lambda m: m["logZ"]
        )["model_id"]

    predictions = json.loads((out / "predictions" / "t02.json").read_text())
    if isinstance(predictions["single_model_id"], str):  # no per-query exclusions
        assert final["single_model_id"] == predictions["single_model_id"]


def test_k2_pool_weights_are_the_weights_the_predictive_mixture_used(tmp_path, monkeypatch):
    """The file is not a re-derivation under some other rule: each line's
    weights are `positive_inference.validate_pool`'s own normalized evidence
    weights - the array `log_predictive`/`predictive_summary` mix with.
    """
    from bayes_model_discovery.experimental.positive_inference import validate_pool

    pools = []
    real = cc.official_checkpoint_scores

    def recording(fits, *args, **kwargs):
        """Record the pool of fits, then delegate to the real function."""
        pools.append(list(fits))
        return real(fits, *args, **kwargs)

    monkeypatch.setattr(cc, "official_checkpoint_scores", recording)
    manifest = _manifest(tmp_path)
    out = tmp_path / "run"
    cc.run(_args(out, manifest, rounds=2))

    lines = _pool_weight_lines(out)
    assert len(pools) == len(lines) == 3
    for fits, line in zip(pools, lines, strict=True):
        expected = np.exp(validate_pool(fits))
        assert [model["model_id"] for model in line["models"]] == [
            result.hypothesis.model_id for result in fits
        ]
        assert [model["weight"] for model in line["models"]] == [
            float(w) for w in expected
        ]
        assert [model["logZ"] for model in line["models"]] == [
            float(result.log_evidence) for result in fits
        ]
        assert [model["structure"] for model in line["models"]] == [
            cc._packet_expression(result.hypothesis, list(FAKE_NAMES)) for result in fits
        ]

    # And that string is exactly the one the archive/proposal records spell:
    # `_packet_view` builds its belief-packet member out of the SAME text.
    grammar_names = ("f0", "f0z", "f1", "f1z")
    for hypothesis in (
        pools[-1][0].hypothesis,
        cc.ScalarHypothesis("b0+b1*f0z", "lognormal",
                            (("b0", "intercept"), ("b1", "signed")), grammar_names),
    ):
        view = cc._packet_view(hypothesis, list(FAKE_NAMES), grammar_names)
        assert cc._packet_expression(hypothesis, list(FAKE_NAMES)) == view.expression
