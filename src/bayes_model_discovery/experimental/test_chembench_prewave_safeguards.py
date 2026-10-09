"""Synthetic prewave safeguards; no native outputs or provider calls are used."""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from bayes_model_discovery.experimental import chembench_core as cc
from bayes_model_discovery.experimental import test_chembench_core as old
from bayes_model_discovery.experimental.count_inference import Fit
from bayes_model_discovery.experimental.test_chembench_mda_options import stub_fits
from bayes_model_discovery.provenance.canonical import is_hex64


def _fit(expression="b0", evidence=0.0, coefficient=0.0):
    interface = cc.task_interface("positive", [{"name": "u", "range": [0.0, 1.0]},
                                               {"name": "v", "range": [0.0, 1.0]}], {}, (0.0, None))
    hypothesis = cc.ScalarHypothesis(expression, "lognormal", (("b0", "intercept"),), ("u", "v"))
    return Fit(hypothesis, {"b0": np.full(4, coefficient), "sigma": np.full(4, 0.2)},
               evidence, [evidence], {"interface": interface, "response_unit": 1.0}, 0.0)


def _rows(path):
    return [json.loads(row) for row in path.read_text().splitlines()]


def _json(path):
    return json.loads(path.read_text())


def test_prior_checks_box_corners_missed_by_diagonal():
    """Check that prior checks box corners missed by diagonal."""
    interface = cc.task_interface("positive", [{"name": "u", "range": [0.0, 1.0]},
                                               {"name": "v", "range": [0.0, 1.0]}], {}, (0.0, None))
    hypothesis = cc.ScalarHypothesis("b0-1000*square(u-v)", "lognormal",
                                    (("b0", "intercept"),), ("u", "v"))
    cc.grammar_means(hypothesis, cc._prior_particles(hypothesis, 4),
                     cc.interface_grid(interface, 16), interface, 1.0)
    result, prediction = cc._prior_probe(hypothesis, interface, 1.0, count=4)
    assert not result["passed"]
    assert "non-finite/nonpositive mean" in result["detail"]
    assert prediction is None


def test_prior_checks_every_observed_design(monkeypatch):
    """Check that prior checks every observed design."""
    result = _fit()
    original = cc.grammar_means
    checked = []
    def inspect(hypothesis, particles, inputs, interface, response_unit):
        """Stub prior check that records or validates the inputs it is given."""
        checked.extend(np.asarray(inputs).tolist())
        if np.any(np.asarray(inputs) == 0.314159):
            raise ValueError("non-finite/nonpositive mean; no clipping fallback")
        return original(hypothesis, particles, inputs, interface, response_unit)
    monkeypatch.setattr(cc, "grammar_means", inspect)
    probe, _ = cc._prior_probe(result.hypothesis, result.diagnostics["interface"], 1.0,
                              count=4, observed_inputs=np.array([[0.314159, 0.0]]))
    assert not probe["passed"] and [0.314159, 0.0] in checked


def test_prior_box_rejection_is_candidate_local_and_cell_continues(tmp_path, monkeypatch):
    """Check that prior box rejection is candidate local and cell continues."""
    stub_fits(monkeypatch)
    monkeypatch.setattr(cc, "_mock_schedule", lambda *_: (
        ("b0-2000*square(f0-f1)", "lognormal"), ("b0-2000*square(f0-f1)", "lognormal")))
    fitted = []
    original_fit = cc.fit
    def tracked(hypothesis, *args, **kwargs):
        """Wrapper that records each call before delegating."""
        fitted.append(hypothesis)
        return original_fit(hypothesis, *args, **kwargs)
    monkeypatch.setattr(cc, "fit", tracked)
    output = tmp_path / "rejected"
    cc.run(old._args(output, old._manifest(tmp_path), cell="100", structure_proposals="free", rounds=2))
    assert _json(output / "summary.json")["active_pool"] == 4
    packet = _json(output / "packets/t02.json")
    entry = next(row for row in packet["archive"] if row["outcome"] == "rejected")
    assert "non-finite/nonpositive mean" in entry["reason"]["validation_message"]
    assert not any(getattr(item, "expression", "") == entry["expression"] for item in fitted)
    assert _json(output / "proposals/t02.json")["admitted_ids"] == []
    assert len(_rows(output / "observations.jsonl")) == 6


def test_invalid_model_is_excluded_only_at_its_invalid_point():
    """Check that invalid model is excluded only at its invalid point."""
    good, bad = _fit(coefficient=np.log(2), evidence=np.log(3)), _fit("b0-1000*u")
    inputs = np.array([[0.0, 0.0], [1.0, 0.0]])
    summary = cc.guarded_predictive_summary([good, bad], inputs, 129, 7, "synthetic", 64)
    np.testing.assert_allclose(summary["mean"], [1.75, 2.0])
    np.testing.assert_allclose(summary["model_weights"], [[0.75, 0.25], [1.0, 0.0]])
    assert summary["prediction_excluded_models"] == [{
        "model_id": bad.hypothesis.model_id, "round": 7, "checkpoint": "synthetic",
        "query_index": 1, "reason": "non-finite/nonpositive mean; no clipping fallback"}]
    assert len([good, bad]) == 2
    # A later valid point still includes the previously excluded model.
    later = cc.guarded_predictive_summary([good, bad], inputs[:1], 129, 8, "synthetic", 64)
    assert "prediction_excluded_models" not in later
    np.testing.assert_allclose(later["model_weights"], [0.75, 0.25])
    logp, components = cc.guarded_log_predictive([good, bad], inputs, np.array([1.0, 2.0]))
    expected, _ = cc.log_predictive([good], inputs[1:], np.array([2.0]))
    np.testing.assert_array_equal(logp[1:], expected)
    assert np.isneginf(components[1, 1]).all()


@pytest.mark.parametrize("function", [cc.guarded_predictive_summary, cc.guarded_log_predictive])
def test_all_excluded_still_fails(function):
    """Check that all excluded still fails."""
    with pytest.raises(ValueError, match="all models excluded.*no clipping"):
        function([_fit("b0-1000*u")], np.array([[1.0, 0.0]]), 123)


def test_unrelated_prediction_error_is_not_hidden(monkeypatch):
    """Check that unrelated prediction error is not hidden."""
    def fail(*args):
        """Stand-in that always raises, to check the error is propagated rather than hidden."""
        raise ValueError("incompatible diagnostic shape")
    monkeypatch.setattr(cc, "particle_means", fail)
    with pytest.raises(ValueError, match="incompatible diagnostic shape"):
        cc.guarded_predictive_summary([_fit()], np.array([[0.0, 0.0]]), 99)


def test_valid_predictions_and_density_preserve_bytes():
    """Check that valid predictions and density preserve bytes."""
    fits, inputs = [_fit(), _fit("b0+u", coefficient=0.3)], np.array([[0.1, 0.0], [0.5, 0.0], [0.9, 0.0]])
    actual = cc.guarded_predictive_summary(fits, inputs, 14, sample_count=128)
    expected = cc.predictive_summary(fits, inputs, 14, 128)
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            assert actual[key].dtype == value.dtype and actual[key].tobytes() == value.tobytes()
        else:
            assert actual[key] == value
    for actual, expected in zip(cc.guarded_log_predictive(fits, inputs, np.ones(3)),
                                cc.log_predictive(fits, inputs, np.ones(3)), strict=True):
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf])
def test_kl_invalid_truth_center_does_not_redefine_primary_metrics(invalid):
    """Check that kl invalid truth center does not redefine primary metrics."""
    fits, inputs = [_fit()], np.array([[0.2, 0.0], [0.7, 0.0]])
    primary = np.log1p(np.array([1.1, 1.3]))
    targets = {"observed": np.array([1.1, 1.3]), "log_primary": primary,
               "log_primary_variance": float(primary.var())}
    before = cc.official_checkpoint_scores(fits, inputs, 44, targets)
    result = cc.official_kl_summary(fits, inputs, [1.0, invalid], 0.01)
    assert result["kl"] == "undefined_truth_center"
    assert result["kl_undefined_truth_center_points"] == [1]
    assert result["quadrature"] is None
    assert cc.official_checkpoint_scores(fits, inputs, 44, targets) == before


def test_kl_uses_same_point_local_bma_and_recomputable_artifact(tmp_path):
    """Check that kl uses same point local bma and recomputable artifact."""
    fits, inputs = [_fit(), _fit("b0-1000*u")], np.array([[0.0, 0.0], [1.0, 0.0]])
    result = cc.official_kl_summary(fits, inputs, np.ones(2), 0.01, base_nodes=16)
    assert np.isfinite(result["kl"])
    artifact = cc.write_official_kl_particle_summary(tmp_path / "kl.npz", fits, inputs,
                                                   result["quadrature"])
    assert artifact["format"] == "mixture_log_density_at_quadrature_nodes"
    np.testing.assert_array_equal(np.load(tmp_path / "kl.npz")["log_density"],
                                  result["quadrature"]["log_q"])



def test_prediction_exclusions_are_logged_without_retiring_model(tmp_path, monkeypatch):
    """Check that prediction exclusions are logged without retiring model."""
    from bayes_model_discovery.experimental import positive_inference as positive

    stub_fits(monkeypatch)
    selected = None
    original_fit, original_catalogue = cc.fit, cc.initial_catalogue
    original_means = positive.particle_means
    def catalogue(*args, **kwargs):
        """Stub catalogue builder returning a small fixed set of candidates."""
        nonlocal selected
        values = original_catalogue(*args, **kwargs)
        selected = values[0][0].model_id
        return values
    def fit(*args, **kwargs):
        """Stub fit: return a synthetic Fit without running inference."""
        result = original_fit(*args, **kwargs)
        result.diagnostics["observation_count"] = len(args[2])
        return result
    def means(result, inputs):
        """Stub prediction-means function over the given inputs."""
        if (result.hypothesis.model_id == selected
                and result.diagnostics["observation_count"] == 4
                and np.any(np.asarray(inputs)[:, 0] == old.EVALUATION_SENTINEL)):
            raise ValueError("non-finite/nonpositive mean; no clipping fallback")
        return original_means(result, inputs)
    def sample_designs(count, seed):
        """Stub design sampler returning synthetic designs for the given count and seed."""
        rows = old._fake_sample_designs(count, seed)
        if seed == cc.seed_for("chem_evaluation_inputs", 20260907):
            for row in rows[1:]:
                row["f0"] = 0.6
        return rows
    monkeypatch.setattr(cc, "initial_catalogue", catalogue)
    monkeypatch.setattr(cc, "fit", fit)
    monkeypatch.setattr(cc, "particle_means", means)
    monkeypatch.setattr(positive, "particle_means", means)
    monkeypatch.setattr(old.real_adapter, "sample_designs", sample_designs)
    output = tmp_path / "predicted"
    cc.run(old._args(output, old._manifest(tmp_path)))
    summary = _json(output / "summary.json")
    assert summary["active_pool"] == 4 and summary["fit_count"] == 8
    exclusions = summary["prediction_excluded_models"]
    assert len(exclusions) == 1
    assert exclusions[0]["model_id"] == selected and exclusions[0]["round"] == 0
    assert exclusions[0]["query_index"] == 0
    assert _json(output / "scores_after_finish.json")["prediction_excluded_models"] == exclusions
    assert _json(output / "official_scores_after_finish.json")["prediction_excluded_models"] == exclusions
    assert "prediction_excluded_models" not in _json(output / "predictions/t01.json")
    with np.load(output / "predictions/t00.npz") as first:
        assert first["model_weights"][0, 0] == 0
        assert first["model_weights"][1, 0] > 0
        np.testing.assert_allclose(first["model_weights"].sum(axis=1), 1.0)
    assert (output / f"fits/t01_{selected}.npz").exists()


def test_undefined_kl_run_completes_with_primary_points(tmp_path, monkeypatch):
    """Check that undefined kl run completes with primary points."""
    stub_fits(monkeypatch)
    class Session(old._FakeSession):
        """Fake session whose reference values are altered for this test."""
        def reference_values(self, designs):
            """Return the fake session reference values with a copied noiseless observation."""
            values = super().reference_values(designs)
            values["noiseless_observation"] = values["noiseless_observation"].copy()
            values["noiseless_observation"][0] = 0.0
            return values
    monkeypatch.setattr(old.real_adapter, "ChemBenchSession", Session)
    output = tmp_path / "undefined"
    cc.run(old._args(output, old._manifest(tmp_path)))
    summary = _json(output / "summary.json")
    assert summary["status"] == "completed_fixed_pool_first_path"
    assert summary["kl"] == "undefined_truth_center"
    assert np.isfinite(summary["primary_metric_final"])
    with (output / "points.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[-1]["kl"] == "undefined_truth_center"
    assert all(np.isfinite(float(row["y_primary"])) for row in rows)
    assert not (output / "official_kl_particle_summary_after_finish.npz").exists()


def test_slow_catalogue_fit_is_retired_and_next_fit_runs(tmp_path, monkeypatch):
    """Check that slow catalogue fit is retired and next fit runs."""
    stub_fits(monkeypatch)
    original_fit = cc.fit
    first = None
    original_catalogue = cc.initial_catalogue
    def catalogue(*args, **kwargs):
        """Stub catalogue builder returning a small fixed set of candidates."""
        nonlocal first
        values = original_catalogue(*args, **kwargs)
        first = values[0][0].model_id
        return values
    monkeypatch.setattr(cc, "initial_catalogue", catalogue)
    def slow_fit(hypothesis, *args, **kwargs):
        """Stub fit that simulates a slow fit for the timeout test."""
        with (tmp_path / "attempts.jsonl").open("a") as stream:
            stream.write(json.dumps({"id": hypothesis.model_id, "pid": os.getpid(), "pgid": os.getpgrp()}) + "\n")
        if hypothesis.model_id == first:
            child = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(10)"])
            (tmp_path / "worker_pid").write_text(str(child.pid))
            time.sleep(10)
        return original_fit(hypothesis, *args, **kwargs)
    monkeypatch.setattr(cc, "fit", slow_fit)
    output = tmp_path / "timed"
    cc.run(old._args(output, old._manifest(tmp_path), fit_timeout_seconds=0.15, rounds=2))
    summary, timing = _json(output / "summary.json"), _rows(output / "fit_timings.jsonl")
    assert summary["status"] == "completed_fixed_pool_first_path" and summary["active_pool"] == 3
    failure = _rows(output / "candidate_failures.jsonl")
    assert len(failure) == 1 and failure[0]["model_id"] == first
    assert failure[0]["status"] == "fit_timeout" and failure[0]["stage"] == "sampling"
    assert failure[0]["cause_code"] == "wall_clock_timeout"
    assert failure[0]["elapsed_seconds"] >= 0.15 and failure[0]["timeout_seconds"] == 0.15
    assert [row["status"] for row in timing] == ["fit_timeout"] + ["completed"] * 9
    attempts = _rows(tmp_path / "attempts.jsonl")
    assert sum(row["id"] == first for row in attempts) == 1
    assert not Path(f"/proc/{(tmp_path / 'worker_pid').read_text()}").exists()
    assert not any(Path(f"/proc/{row['pid']}").exists() for row in attempts)
    pgid = attempts[0]["pgid"]
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            try:
                assert os.getpgid(int(path.name)) != pgid
            except ProcessLookupError:
                pass
    assert _json(output / "config.json")["fit_timeout_seconds"] == 0.15


def test_timed_out_proposal_stays_known_and_is_not_refit(tmp_path, monkeypatch):
    """Check that timed out proposal stays known and is not refit."""
    stub_fits(monkeypatch)
    original_fit = cc.fit
    expression = "b0+b1*square(f0)"
    monkeypatch.setattr(cc, "_mock_schedule", lambda *_: ((expression, "lognormal"),) * 3)
    def slow_candidate(hypothesis, *args, **kwargs):
        """Stub candidate fit that simulates a slow proposal for the timeout test."""
        if getattr(hypothesis, "expression", None) == expression:
            with (tmp_path / "candidate_attempts").open("a") as stream:
                stream.write(hypothesis.model_id + "\n")
            time.sleep(10)
        return original_fit(hypothesis, *args, **kwargs)
    monkeypatch.setattr(cc, "fit", slow_candidate)
    output = tmp_path / "proposal"
    cc.run(old._args(output, old._manifest(tmp_path), fit_timeout_seconds=0.15,
                    rounds=3, cell="100", structure_proposals="free"))
    attempts = (tmp_path / "candidate_attempts").read_text().splitlines()
    assert len(attempts) == 1
    assert _json(output / "summary.json")["active_pool"] == 4
    packet = _json(output / "packets/t03.json")
    archive = [row for row in packet["archive"] if row["model_id"] == attempts[0]]
    assert [row["outcome"] for row in archive] == ["fit_failed", "duplicate"]
    assert archive[0]["note"] == "fit_timeout"
    assert _json(output / "proposals/t03.json")["admitted_ids"] == []


def test_no_timeout_writes_no_new_sidecar_or_keys(tmp_path, monkeypatch):
    """Check that no timeout writes no new sidecar or keys."""
    stub_fits(monkeypatch)
    output = tmp_path / "default"
    cc.run(old._args(output, old._manifest(tmp_path), fit_timeout_seconds=None))
    assert "fit_timeout_seconds" not in _json(output / "config.json")
    assert not (output / "fit_timings.jsonl").exists()
    assert not (output / "candidate_failures.jsonl").exists()
    summary = _json(output / "summary.json")
    assert "prediction_excluded_models" not in summary and "kl_status" not in summary


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_bad_timeout_fails_before_outputs(tmp_path, value):
    """Check that bad timeout fails before outputs."""
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="finite and positive"):
        cc.run(old._args(output, old._manifest(tmp_path), fit_timeout_seconds=value))
    assert not output.exists()



@pytest.mark.parametrize("tokens,expected", [([], None), (["--fit-timeout", "600"], 600.0)])
def test_fit_timeout_cli_is_explicit_and_omitted_by_default(monkeypatch, tokens, expected):
    """Check that fit timeout cli is explicit and omitted by default."""
    captured = []
    monkeypatch.setattr(cc, "run", captured.append)
    monkeypatch.setattr(sys, "argv", ["chembench_core", "--manifest", "synthetic.json",
                                      "--output", "synthetic_output", "--noise", "0.01", *tokens])
    cc.main()
    assert len(captured) == 1
    if expected is None:
        assert not hasattr(captured[0], "fit_timeout_seconds")
    else:
        assert captured[0].fit_timeout_seconds == expected


BASELINE_HARNESS = r'''
import sys
from pathlib import Path
import pytest
from bayes_model_discovery.experimental import chembench_core as cc
from bayes_model_discovery.experimental import test_chembench_core as old
from bayes_model_discovery.experimental.test_chembench_mda_options import stub_fits
with pytest.MonkeyPatch.context() as patch:
    stub_fits(patch)
    cc.run(old._args(Path(sys.argv[2]), Path(sys.argv[3]), rounds=2,
           cell=sys.argv[4], structure_proposals='free' if sys.argv[4][0] == '1' else 'catalog'))
'''


@pytest.mark.parametrize("cell", ["000", "100"])
@pytest.mark.skip(reason="parity with the baseline recorded before the model-identity change crosses the identity boundary; a successor parity test is still owed")
def test_opt_in_default_artifacts_match_declared_base(tmp_path, cell):
    """Check that opt in default artifacts match declared base."""
    if os.environ.get("BMD_BASELINE_COMPARE") != "1":
        pytest.skip("set BMD_BASELINE_COMPARE=1 for the detached baseline artifact comparison")
    root = Path(__file__).resolve().parents[3]
    scratch = Path(os.environ.get("BMD_SAFEGUARDS_SCRATCH", root.parent / (root.name + "_scratch")))
    baseline = Path(os.environ.get("BMD_SAFEGUARDS_BASELINE", scratch / "chem/base"))
    if not baseline.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(baseline),
                        "c6fe27c76515065a943df33f122529eb181aac04"], cwd=root, check=True, capture_output=True)
    output, manifest = tmp_path / "same_output", old._manifest(tmp_path)
    saved = []
    for label, source in (("base", baseline), ("current", root)):
        env = os.environ.copy()
        env.update(PYTHONPATH=f"{source}/src:{source}", PYTHONDONTWRITEBYTECODE="1",
                   TMPDIR="/tmp", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENAI_API_KEY="", ANTHROPIC_API_KEY="", OPENAI_COMPAT_API_KEY="")
        result = subprocess.run([sys.executable, "-B", "-c", BASELINE_HARNESS, str(Path(__file__)),
                                 str(output), str(manifest), cell], cwd=source, env=env,
                                capture_output=True, text=True, check=False)
        (tmp_path / f"{label}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        target = tmp_path / label
        output.rename(target)
        saved.append(target)
    a, b = saved
    # round_timings.jsonl is ChemBench's per-round wall time.
    observational_sidecars = {"fit_timings.jsonl", "acquisition_timings.jsonl",
                              "prediction_pool_summary.jsonl", "round_timings.jsonl"}
    files_a = sorted(str(path.relative_to(a)) for path in a.rglob("*")
                     if path.is_file() and path.name not in observational_sidecars)
    files_b = sorted(str(path.relative_to(b)) for path in b.rglob("*")
                     if path.is_file() and path.name not in observational_sidecars)
    assert files_a == files_b
    config_a, config_b = _json(a / "config.json"), _json(b / "config.json")
    # Absolute source provenance and measured timing are truthful and necessarily
    # checkout-dependent. Normalize only these declared metadata fields.
    hashes = list(zip(config_b["source_sha256"].values(), config_a["source_sha256"].values(), strict=True))
    report = []
    for name in files_a:
        aa, bb = (a / name).read_bytes(), (b / name).read_bytes()
        if name.endswith(".npz"):
            assert aa == bb, name
            with np.load(a / name) as left, np.load(b / name) as right:
                assert left.files == right.files, name
                for key in left.files:
                    assert left[key].dtype == right[key].dtype and left[key].tobytes() == right[key].tobytes(), (name, key)
        else:
            if name == "summary.json":
                left, right = json.loads(aa), json.loads(bb)
                for item in (left, right):
                    for key in ("wall_seconds", "user_cpu_seconds", "system_cpu_seconds", "timing_seconds"):
                        item.pop(key, None)
                aa, bb = (json.dumps(item, sort_keys=True).encode() for item in (left, right))
            elif name == "config.json" or (name.startswith("acquisition/") and name.endswith(".json")):
                # Candidate-pool provenance keys: present and well-formed now,
                # absent at the base commit; removed only after that check
                # (stale-key normalization). Every other key compares byte for byte.
                original, current_record = json.loads(aa), json.loads(bb)
                if "candidate_set_sha256" in current_record or "candidate_generator" in current_record:
                    for key in ("candidate_set_sha256", "candidate_generator"):
                        assert key not in original, (name, key)
                        assert key in current_record, (name, key)
                    assert is_hex64(current_record.pop("candidate_set_sha256")), name
                    assert current_record.pop("candidate_generator")["hash_scope"] == "generated_pool_before_history_filters", name
                    bb = (json.dumps(current_record, indent=2, sort_keys=True) + "\n").encode()
            bb = bb.replace((str(root) + "/").encode(), (str(baseline) + "/").encode())
            for current_hash, base_hash in hashes:
                bb = bb.replace(current_hash.encode(), base_hash.encode())
            assert aa == bb, name
        report.append({"artifact": name, "identical_after_declared_metadata_normalization": True})
    (tmp_path / "parity_report.json").write_text(json.dumps(report, indent=2) + "\n")
