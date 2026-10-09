"""Synthetic expected predictive information gain (EPIG) dispatch, once-only targets and legacy artifact preservation."""
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from . import boxinggym_core as bg
from . import chembench_core as chem
from . import core_count as core
from . import neuron_core as neuron
from .test_chembench_core import _args, _manifest
from .test_chembench_mda_options import stub_fits
from .test_named_inputs import bg_run_args, bg_stub_fits
from .test_prewave_fit_timeout import _count_result


def _run(runner, output, patch, cell="002", rounds=2, two_inputs=False):
    """Helper: run."""
    if runner == "core":
        patch.setattr(core, "fit", lambda hypothesis, *args: _count_result(hypothesis))
        patch.setattr(sys, "argv", ["core", "--output", str(output), "--profile", "dev",
                                   "--cell", cell, "--rounds", str(rounds), "--proposer", "mock"])
        core.main()
    elif runner == "chembench":
        stub_fits(patch)
        if two_inputs:
            # The pool EIG/EPIG path runs on the two-input fake. Imported
            # here, not at module level: the parity harness below loads this file against a
            # base checkout whose test_chembench_core has no two-input fake.
            from .test_chembench_core import use_two_input_fake
            use_two_input_fake(patch)
        chem.run(_args(output, _manifest(output.parent), cell=cell,
                       design_policy=None if cell[2] == "2" else "joint_eig",
                       structure_proposals=None, rounds=rounds))
    elif runner == "boxinggym":
        bg_stub_fits(patch)
        bg.run(bg_run_args(output, cell=cell, rounds=rounds))
    else:
        from .test_neuron_core_cli import transport_world
        transport_world.__wrapped__(patch)
        neuron.run(neuron.build_parser().parse_args([
            "--output", str(output), "--cell", cell, "--rounds", str(rounds)]))


@pytest.mark.parametrize("runner", ["core", "boxinggym", "chembench"])
@pytest.mark.parametrize("cell", ["002", "102"])
def test_epig_runs_keep_fixed_targets_and_record_choice(tmp_path, monkeypatch, runner, cell, request):
    """Check that epig runs keep fixed targets and record choice."""
    if runner == "boxinggym":
        request.getfixturevalue("require_boxinggym_source")
    module = {"core": core, "boxinggym": bg, "chembench": chem}[runner]
    original = module.choose_epig
    seen = []
    def capture(*args, **kwargs):
        """Helper: capture."""
        seen.append(np.asarray(kwargs["target_points"]).copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "choose_epig", capture)
    output = tmp_path / "run"
    _run(runner, output, monkeypatch, cell=cell, two_inputs=runner == "chembench")
    config = json.loads((output / "config.json").read_text())
    assert config["design_policy"] == "epig"
    assert config["design_policy_target"] == "predictive_target_points"
    assert config["epig_target_points"] == 32
    assert config["epig_y_grid"] == 64
    assert config["epig_targets_seed_stream"] == "epig_targets"
    assert config["epig_targets_seed"] == core.seed_for("epig_targets", config["seed"])
    if runner == "chembench":
        assert config["design_pool"]["eig_estimator"] == "count_inference.choose_epig"
        assert config["design_pool"]["objective_column"] == "epig"
    if runner == "boxinggym":
        assert "multi_component_eig_rule" not in config["task_spec"]
        assert "multi_component_epig_rule" in config["task_spec"]
    assert len(seen) == 2
    for targets in seen:
        np.testing.assert_array_equal(targets, config["epig_targets"])
    for path in sorted((output / "acquisition").glob("t*.json")):
        record = json.loads(path.read_text())
        assert record["target"] == "predictive_target_points"
        assert record["rows"] == record["epig_rows"]
        assert record["epig_rows"][record["policy"]["chosen_index"]]["design"] == record["chosen"]
        assert record["policy"]["units"] == "nats"
    with (output / "points.csv").open() as stream:
        points = list(csv.DictReader(stream))
    assert points and {row["method"] for row in points} == {cell}
    assert all("design_epig" not in row.get("variant", "") for row in points)
    assert json.loads((output / "summary.json").read_text())["paid_api_calls"] == 0


@pytest.mark.parametrize("cell", ["002", "012", "102", "112"])
def test_neuron_executes_the_epig_cell_unit_24_only_recognised(tmp_path, monkeypatch, cell):
    """An EPIG cell is recognised by `resolve_acquisition` and the run executes the acquisition.
    This test used to assert the opposite (a refusal as "not implemented"); the refusal is gone,
    so the expectation was flipped rather than the test deleted. `resolve_acquisition` itself is
    unchanged, which the first two assertions pin."""
    assert neuron.resolve_acquisition(cell, None) == "epig"
    assert neuron.resolve_acquisition(cell, "epig") == "epig"
    output = tmp_path / "run"
    _run("neuron", output, monkeypatch, cell=cell, rounds=1)
    config = json.loads((output / "config.json").read_text())
    assert config["acquisition"] == "epig"
    assert config["EPIG_predictive_samples"] == 128
    assert config["epig_targets"] == "world.test_protocols (public evaluation input measure)"
    records = [json.loads(line)
               for line in (output / "acquisition.jsonl").read_text().splitlines()]
    assert records
    for record in records:
        assert record["mode"] == "epig"
        assert record["selected"] in record["available"]
        assert len(record["rows"]) == len(record["available"])
        assert [row["design_index"] for row in record["rows"]] == list(
            range(len(record["available"]))
        )
        for row in record["rows"]:
            assert row["policy"] == "epig"
            assert row["samples"] == 128
            assert row["targets"] == len(row["per_target"]) == len(config["q_protocol_labels"])
            assert np.isfinite(row["epig"]) and row["mc_standard_error"] >= 0


@pytest.mark.parametrize("cell", ["002", "012", "102", "112"])
def test_neuron_axis_config_accepts_epig_cell(cell):
    """Check that neuron axis config accepts epig cell."""
    assert neuron.cell_axis_config(cell)["acquisition_from_cell"] is True


def test_boxing_target_sampler_determinism_and_rng_isolation():
    """Check that boxing target sampler determinism and rng isolation."""
    from types import SimpleNamespace
    spec = bg.TASKS["death"]
    class NativeGoal:
        """Synthetic NativeGoal fixture used by the surrounding test."""
        def __init__(self, env):
            """Helper: init  ."""
            self.eval_points = []

        def get_goal_eval_question(self, include_prior):
            """Helper: get goal eval question."""
            assert include_prior
            self.eval_points.append((float(np.random.normal(10, 1)), 0))

    native_env = SimpleNamespace()
    environment = SimpleNamespace(goal=NativeGoal(native_env), env=native_env,
                                  eval_inputs=lambda point: [point[0]])
    pool = SimpleNamespace(single_input=True)
    np.random.seed(551)
    before = np.random.get_state()
    first = bg.sample_epig_targets(spec, environment, pool, 115)
    second = bg.sample_epig_targets(spec, environment, pool, 115)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 32 and len(np.unique(first)) == 32
    # The native goal covariates replace the old legal-range uniform targets with native goal
    # covariates, even where the goal measure is outside acquisition bounds.
    assert (first > (spec.design_range or spec.input_range)[1]).all()
    assert environment.goal.eval_points == []
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


PARITY_HARNESS = r'''
from datetime import datetime
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
spec = importlib.util.spec_from_file_location("bayes_model_discovery.experimental._epig_fixture", sys.argv[1])
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2001, 1, 1, tzinfo=tz)
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(fixture.core.time, "perf_counter", lambda: 0.0)
    patch.setattr(fixture.core.resource, "getrusage", lambda *_: SimpleNamespace(ru_utime=0.0, ru_stime=0.0))
    for module in (fixture.core, fixture.bg, fixture.chem, fixture.neuron):
        if hasattr(module, "datetime"):
            patch.setattr(module, "datetime", FrozenDatetime)
    fixture._run(sys.argv[3], Path(sys.argv[2]), patch, cell=sys.argv[4], rounds=1)
'''


@pytest.mark.parametrize("runner", ["core", "boxinggym", "chembench", "neuron"])
@pytest.mark.parametrize("cell", ["000", "001", "100", "101"])
def test_opt_in_all_legacy_cells_preserve_artifact_bytes(tmp_path, runner, cell):
    """Check that opt in all legacy cells preserve artifact bytes."""
    if runner != "neuron":
        pytest.skip("parity with the baseline recorded before the model-identity change crosses the ID/RNG boundary; a successor parity test is still owed")
    if os.environ.get("BMD_BASELINE_COMPARE") != "1":
        pytest.skip("set BMD_BASELINE_COMPARE=1 and BMD_EPIG_BASELINE_DIR for verified-base parity")
    value = os.environ.get("BMD_EPIG_BASELINE_DIR")
    if not value:
        pytest.skip("BMD_EPIG_BASELINE_DIR must name a read-only checkout of the baseline commit")
    root = Path(__file__).resolve().parents[3]
    baseline = Path(value).resolve()
    # The base commit this opt-in comparison pins; it skips when no checkout of it is supplied.
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=baseline, text=True).strip() == \
        "cf169c5b17f45931814c82c22e4bcfdf2ad142b8"
    output, saved, configurations = tmp_path / "same_output", [], []
    for label, source in (("baseline", baseline), ("current", root)):
        env = os.environ.copy()
        env.update(PYTHONPATH=f"{source}/src:{source}:" + env.get("PYTHONPATH", ""),
                   PYTHONDONTWRITEBYTECODE="1", OPENAI_API_KEY="", ANTHROPIC_API_KEY="",
                   OPENAI_COMPAT_API_KEY="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        process = subprocess.run([sys.executable, "-B", "-c", PARITY_HARNESS, str(Path(__file__).resolve()),
                                  str(output), runner, cell], cwd=source, env=env,
                                 capture_output=True, text=True, timeout=120, check=False)
        assert process.returncode == 0, process.stdout + process.stderr
        configurations.append(json.loads((output / "config.json").read_text()))
        saved.append(tmp_path / label)
        output.rename(saved[-1])
    before, after = saved
    # round_timings.jsonl is ChemBench's per-round wall time, absent at the base.
    observational_sidecars = {"fit_timings.jsonl", "acquisition_timings.jsonl",
                              "prediction_pool_summary.jsonl", "round_timings.jsonl"}
    paths = lambda directory: sorted(str(p.relative_to(directory)) for p in directory.rglob("*")
                                     if p.is_file() and p.name not in observational_sidecars)
    assert paths(before) == paths(after)
    provenance = {"core": "source_files_sha256", "boxinggym": "source_files_sha256",
                  "chembench": "source_sha256", "neuron": "experimental_source_sha256"}[runner]
    replacements = []
    for index, source in enumerate((baseline, root)):
        for name, digest in configurations[index][provenance].items():
            assert digest == hashlib.sha256((source / "src/bayes_model_discovery/experimental" / name).read_bytes()).hexdigest()
    for name in configurations[0][provenance]:
        replacements.append((configurations[1][provenance][name].encode(), configurations[0][provenance][name].encode()))
    from .count_inference import candidate_set_record
    if runner == "core":
        candidate_metadata = candidate_set_record(np.linspace(0, 1, configurations[1]["design_count"]), runner)
    elif runner == "boxinggym":
        with np.load(after / "evaluation_scoring_only.npz") as values:
            candidate_metadata = candidate_set_record(values["designs"], runner)
    elif runner == "chembench":
        from .test_chembench_core import FAKE_NAMES, _fake_sample_designs
        pool_config = configurations[1]["design_pool"]
        rows = _fake_sample_designs(pool_config["count"], pool_config["sampler_seed"])
        candidate_metadata = candidate_set_record([[row[n] for n in FAKE_NAMES] for row in rows], runner)
    report = []
    for name in paths(before):
        left, right = (before / name).read_bytes(), (after / name).read_bytes()
        added_fields = []
        if runner == "neuron" and name == "config.json":
            from .test_prewave_fit_timeout import (
                ADDED_NEURON_CONFIG_KEYS,
                _normalize_added_neuron_config,
            )

            raw_right = right
            right = _normalize_added_neuron_config(left, right)
            added_fields.extend(ADDED_NEURON_CONFIG_KEYS)
            assert (after / name).read_bytes() == raw_right
            assert set(ADDED_NEURON_CONFIG_KEYS) <= json.loads((after / name).read_bytes()).keys()
        if ((runner != "neuron" and name.startswith("acquisition/") and name.endswith(".json"))
                or (runner == "chembench" and cell[2] == "0" and name == "config.json")):
            original, current_record = json.loads(left), json.loads(right)
            for key in ("candidate_set_sha256", "candidate_generator"):
                assert key not in original
                assert current_record.pop(key) == candidate_metadata[key]
                added_fields.append(key)
            right = (json.dumps(current_record, indent=2, sort_keys=True) + "\n").encode()
        for current, previous in replacements:
            right = right.replace(current, previous)
        right = right.replace((str(root) + "/").encode(), (str(baseline) + "/").encode())
        assert left == right, name
        report.append({"artifact": name, "raw_bytes_identical": (before / name).read_bytes() == (after / name).read_bytes(),
                       "verified_source_provenance_only_normalization": not added_fields,
                       "explicitly_verified_added_fields": added_fields})
    (tmp_path / "parity_report.json").write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.usefixtures("require_boxinggym_source")
@pytest.mark.parametrize("task", ["peregrines", "dugongs", "location"])
def test_boxing_scalar_epig_runs_and_discloses_its_outcome_rule(tmp_path, monkeypatch, task):
    """Check that boxing scalar epig runs and discloses its outcome rule."""
    from .test_named_inputs import bg_scalar_stub_fits

    bg_scalar_stub_fits(monkeypatch)
    output = tmp_path / task
    bg.run(bg_run_args(output, task=task, cell="002", rounds=1))
    config = json.loads((output / "config.json").read_text())
    record = json.loads((output / "acquisition/t01.json").read_text())
    assert config["design_policy"] == "epig" and len(config["epig_targets"]) == 32
    assert record["policy"]["chosen_design"] == record["chosen"]
    if task == "peregrines":
        info = record["outcome_info"]
        assert info["epig_discrete_exact_original_mi"] is False
        assert info["epig_discrete_max_particle_tail_mass"] <= 1e-12
        assert info["epig_discrete_global_cutoff"] <= 100000
        assert info["epig_discrete_estimated_working_bytes"] <= 256 * 1024 * 1024
        assert len(info["epig_discrete_target_cutoffs"]) == 32
        assert len(info["epig_discrete_candidate_cutoffs"]) == len(record["epig_rows"])
    else:
        assert "outcome_info" not in record
        assert config["epig_y_grid"] == 64
    assert json.loads((output / "summary.json").read_text())["paid_api_calls"] == 0


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("policy", ["p1", "p4"])
def test_epig_cell_reaches_each_injected_provider_context(tmp_path, provider, policy):
    """Check that epig cell reaches each injected provider context."""
    from .test_count_proposer import (
        fixture,
        item,
        one_object,
        openai_fixture,
        p1_fixture,
        p1_openai_fixture,
    )
    if policy == "p1":
        factory = p1_fixture if provider == "anthropic" else p1_openai_fixture
        proposer, calls, context = factory(tmp_path, [one_object(family="binomial")], cell="102")
        models, record = proposer.propose_one(
            '{"round":7}', context=context, initial_family="binomial")
    else:
        factory = fixture if provider == "anthropic" else openai_fixture
        proposer, calls, context, public, pool = factory(tmp_path, json.dumps([item("binomial")]))
        models, record = proposer.propose(
            public, pool, context={**context, "cell": "102"}, initial_family="binomial")
    assert len(calls) == 1 and len(models) == 1
    assert record["cell"] == "102" and record["status"] == "parsed"
    assert record["client_injected"] is True


@pytest.mark.parametrize("runner", ["core", "boxinggym", "chembench"])
def test_three_policies_share_generated_pool_with_explicit_selector_scope(
        tmp_path, monkeypatch, runner, request):
    """Check that three policies share generated pool with explicit selector scope."""
    from .count_inference import coordinate_set_sha256

    if runner == "boxinggym":
        request.getfixturevalue("require_boxinggym_source")
    module = {"core": core, "boxinggym": bg, "chembench": chem}[runner]
    original = module.candidate_set_record
    captured = []

    def capture(values, bundle):
        """Helper: capture."""
        captured.append(np.asarray(values, dtype=float).reshape(len(values), -1).copy())
        return original(values, bundle)

    monkeypatch.setattr(module, "candidate_set_record", capture)
    evidence = []
    for cell in ("000", "001", "002"):
        output = tmp_path / cell
        _run(runner, output, monkeypatch, cell=cell, rounds=3, two_inputs=runner == "chembench")
        config = json.loads((output / "config.json").read_text())
        pool = captured[-1]
        pool_hash = coordinate_set_sha256(pool)
        if runner == "chembench" and cell[2] == "0":
            # A shared unused pool is not an actual random selection invariant.
            assert config["candidate_set_sha256"] == pool_hash
            assert config["design_pool"]["read_by_this_run"] is False
            assert "separate_chem_observation_designs" in config["candidate_generator"]["selection_rule"]
            from .test_chembench_core import FAKE_NAMES_2D
            observations = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
            for t in range(1, 4):
                chosen = [observations[3 + t]["x"][name] for name in FAKE_NAMES_2D]
                assert not np.any(np.all(pool == chosen, axis=1))
                evidence.append({"cell": cell, "round": t, "candidate_set_sha256": pool_hash,
                                 "actual_candidates_sha256": None,
                                 "selection_scope": "separate_predrawn_schedule"})
            assert not (output / "acquisition").exists()
            continue
        for path in sorted((output / "acquisition").glob("t*.json")):
            record = json.loads(path.read_text())
            assert record["candidate_set_sha256"] == pool_hash
            assert record["candidate_generator"]["hash_scope"] == "generated_pool_before_history_filters"
            chosen = np.asarray(record["chosen"]).reshape(-1)
            assert np.any(np.all(pool == chosen, axis=1))
            actual = [row["design"] for row in record["rows"]] if "rows" in record else pool
            actual_hash = coordinate_set_sha256(actual)
            if runner != "chembench":
                # CORE and this no-exclusion BoxingGym fixture actually expose
                # the same entire pool in all independently executed histories.
                assert actual_hash == pool_hash
            evidence.append({"cell": cell, "round": int(path.stem[1:]),
                             "candidate_set_sha256": pool_hash,
                             "actual_candidates_sha256": actual_hash,
                             "selection_scope": record["candidate_generator"]["selection_rule"]})
    assert len(captured) == 3
    for pool in captured[1:]:
        np.testing.assert_array_equal(pool, captured[0])
    assert len(evidence) == 9
    assert len({row["candidate_set_sha256"] for row in evidence}) == 1
    if runner == "chembench":
        initial = [row["actual_candidates_sha256"] for row in evidence
                   if row["round"] == 1 and row["cell"] != "000"]
        assert initial[0] == initial[1]
        assert any(len({row["actual_candidates_sha256"] for row in evidence
                        if row["round"] == t and row["cell"] != "000"}) > 1 for t in (2, 3))
    (tmp_path / "Q3_candidates.json").write_text(json.dumps({
        "fixture": "synthetic_or_native_death_world; mock_fits; independent_policy_histories",
        "runner": runner, "shared_generated_pool_equal": True,
        "actual_three_policy_candidate_invariant": runner != "chembench", "rows": evidence,
    }, indent=2) + "\n")


@pytest.mark.usefixtures("require_boxinggym_source")
@pytest.mark.parametrize("task", ["survival", "irt"])
def test_boxing_finite_pool_hashes_preserve_history_dependent_actual_candidates(
        tmp_path, monkeypatch, task):
    """Check that boxing finite pool hashes preserve history dependent actual candidates."""
    from .count_inference import Fit, coordinate_set_sha256

    def fake_fit(hypothesis, inputs, labels, seed, settings, interface=None, n_trials=50):
        """Helper: fake fit."""
        rng = np.random.default_rng(seed % 2**32)
        shapes = dict(hypothesis.parameter_shapes)
        particles = {name: rng.normal(0.3, 0.4, (8, shapes[name]) if name in shapes else 8)
                     for name, _ in hypothesis.parameters}
        particles.update({name: rng.beta(2, 8, 8) for name in hypothesis.noise_parameters})
        weight = -(int(hypothesis.model_id[:6], 16) % 500) / 100.0
        return Fit(hypothesis, particles, weight, [weight, weight], {"interface": interface})

    monkeypatch.setattr(bg, "count_fit", fake_fit)
    original_available = bg.DesignPool.available
    actual_by_call = []

    def available(pool, observed):
        """Helper: available."""
        values = original_available(pool, observed)
        actual_by_call.append(np.asarray(values).copy())
        return values

    monkeypatch.setattr(bg.DesignPool, "available", available)
    evidence = []
    for cell in ("000", "001", "002"):
        output = tmp_path / cell
        bg.run(bg_run_args(output, task=task, cell=cell, rounds=3))
        actual = actual_by_call[-3:]
        assert len(actual) == 3
        for t, candidates in enumerate(actual, 1):
            record = json.loads((output / "acquisition" / f"t{t:02d}.json").read_text())
            assert np.any(np.all(candidates == record["chosen"], axis=1))
            if "rows" in record:
                np.testing.assert_array_equal([row["design"] for row in record["rows"]], candidates)
            evidence.append({"task": task, "cell": cell, "round": t,
                             "candidate_set_sha256": record["candidate_set_sha256"],
                             "actual_candidates_sha256": coordinate_set_sha256(candidates),
                             "actual_candidate_count": len(candidates)})
    assert len(actual_by_call) == 9
    assert len({row["candidate_set_sha256"] for row in evidence}) == 1
    assert len({row["actual_candidates_sha256"] for row in evidence if row["round"] == 1}) == 1
    assert any(len({row["actual_candidates_sha256"] for row in evidence if row["round"] == t}) > 1
               for t in (2, 3))
    (tmp_path / "Q3_finite_candidates.json").write_text(json.dumps({
        "fixture": "native_finite_world; mock_posterior_fits; independent_policy_histories",
        "shared_generated_pool_equal": True,
        "actual_candidate_equality": "same_history_only; later_policy_histories_diverge",
        "rows": evidence,
    }, indent=2) + "\n")
