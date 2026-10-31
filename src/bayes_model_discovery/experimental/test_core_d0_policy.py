"""Synthetic CORE initial-design policy checks; no saved responses or paid calls."""

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ..provenance.canonical import canonical_json, is_hex64
from . import core_count as core
from . import gaussian_core as gaussian
from .count_inference import Fit
from .run_label_policy import (
    RECEIPT_D0_POLICY,
    _registered_settings,
    bmd_label_receipt,
    bmd_runtime_settings,
    resolve_label,
)
from .test_count_proposer import run_args
from .test_run_label_policy import registration, runner_args, write_manifest

#: The points.csv columns that name the run and its code (read from the record, not compared).
CSV_IDENTITY_FIELDS = ("source_path", "code_sha")


@pytest.mark.parametrize("seed", [31, 37, 41])
def test_stratified_coordinates_equal_the_declared_seed_only_draw(seed):
    """Stratified4 coordinates equal the draw declared from the seed alone and are sorted, one per quarter."""
    rng = np.random.default_rng(core.seed_for("core_d0", seed))
    expected = tuple(0.25 * k + 0.25 * value for k, value in enumerate(rng.random(4)))
    actual = core.initial_designs(seed, "stratified4")
    assert actual == expected
    assert actual == tuple(sorted(actual))
    assert all(k / 4 <= value < (k + 1) / 4 for k, value in enumerate(actual))
    assert actual == core.initial_designs(seed, "stratified4")


def test_different_seeds_have_different_initial_coordinates():
    """Different seeds give different stratified4 initial coordinates."""
    assert len({core.initial_designs(seed, "stratified4") for seed in (31, 37, 41)}) == 3


@pytest.mark.parametrize("seed", [31, 37, 41])
def test_all_regimes_share_seed_coordinates(seed):
    """All structure and family regimes observe the same seed-only initial designs."""
    designs = core.initial_designs(seed, "stratified4")
    observations = []
    for structure, family in (("out", "binomial"), ("den", "beta_binomial"),
                               ("gauss_den", "normal")):
        world = core.CoreWorld(seed, structure, family)
        observations.append([world.observe(u) for u in designs])
    assert all(tuple(row["u"] for row in rows) == designs for rows in observations)


@pytest.mark.parametrize("seed", [31, 37, 41])
@pytest.mark.parametrize("cell", ["100", "102"])
@pytest.mark.parametrize("structure,family", [
    ("out", "binomial"), ("den", "beta_binomial"), ("gauss_den", "normal"),
])
def test_runner_records_the_same_seed_only_designs_before_inference(
        tmp_path, monkeypatch, seed, cell, structure, family):
    """The runner records the seed-only initial designs and their hash before any inference starts."""
    class InitialDataRecorded(Exception):
        """Sentinel raised to stop the runner once the initial data are recorded."""

    def stop_before_inference(*args, **kwargs):
        """Replacement for the hypothesis initializer that raises the sentinel exception."""
        raise InitialDataRecorded

    monkeypatch.setattr(core, "initial_hypotheses", stop_before_inference)
    output = tmp_path / "synthetic_cell"
    args = run_args(output, seed=seed, cell=cell, structure=structure,
                    truth_family=family, initial_family="normal" if family == "normal" else "binomial",
                    d0_policy="stratified4", design_policy=None)
    with pytest.raises(InitialDataRecorded):
        core.run(args)
    config = json.loads((output / "config.json").read_text())
    rows = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    expected = list(core.initial_designs(seed, "stratified4"))
    assert config["d0_policy"] == "stratified4"
    assert config["d0_designs"] == expected == [row["u"] for row in rows]
    assert len(rows) == 4 and {row["round"] for row in rows} == {0}
    assert config["d0_coordinates_sha256"] == hashlib.sha256(canonical_json(expected).encode()).hexdigest()


def test_draw_is_one_local_generator_call_and_never_uses_global_rng(monkeypatch):
    """The initial draw uses exactly one local generator call and never touches the global RNG."""
    constructor = np.random.default_rng
    observed = []

    class LocalGenerator:
        """Recording stand-in for the numpy generator that logs its seed and draw size."""
        def __init__(self, seed):
            observed.append(("seed", seed))
            self.rng = constructor(seed)

        def random(self, size):
            """Log the requested size and draw from the wrapped generator."""
            observed.append(("size", size))
            return self.rng.random(size)

    def forbidden(*args, **kwargs):
        """Fail the test if the global RNG is used."""
        raise AssertionError("global RNG must not be used")

    monkeypatch.setattr(np.random, "default_rng", LocalGenerator)
    monkeypatch.setattr(np.random, "random", forbidden)
    monkeypatch.setattr(np.random, "seed", forbidden)
    core.initial_designs(31, "stratified4")
    assert observed == [("seed", core.seed_for("core_d0", 31)), ("size", 4)]


def test_fixed4_is_the_unchanged_tuple_and_needs_no_rng(monkeypatch):
    """The fixed4 policy returns the unchanged constant tuple and draws no random numbers."""
    def forbidden(*args, **kwargs):
        """Fail the test if the global RNG is used."""
        raise AssertionError("fixed4 does not draw")
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    assert core.D0 == (0.1, 0.35, 0.65, 0.9)
    assert core.initial_designs(31) is core.D0
    assert core.initial_designs(41, "fixed4") is core.D0


def test_unknown_policy_refused_before_output(tmp_path, monkeypatch):
    """An unknown initial-design policy is refused before any output is created."""
    output = tmp_path / "absent"
    monkeypatch.setattr(sys, "argv", ["core_count", "--output", str(output),
                                     "--d0-policy", "unknown"])
    with pytest.raises(SystemExit) as stopped:
        core.main()
    assert stopped.value.code == 2
    assert not output.exists()
    with pytest.raises(ValueError, match="d0_policy"):
        core.initial_designs(31, "unknown")


def test_default_and_explicit_fixed4_keep_the_existing_receipt_settings():
    """Default and explicit fixed4 yield identical receipt settings, while stratified4 differs only in the policy field."""
    assert RECEIPT_D0_POLICY == {"fixed4": "runner_default", "stratified4": "stratified4"}
    absent = bmd_runtime_settings("core", runner_args())
    stated = bmd_runtime_settings("core", runner_args(d0_policy="fixed4"))
    stratified = bmd_runtime_settings("core", runner_args(d0_policy="stratified4"))
    assert absent == stated
    assert absent["d0_policy"] == RECEIPT_D0_POLICY["fixed4"]
    assert stratified == {**absent, "d0_policy": "stratified4"}


def test_stratified_receipt_declares_a_different_policy_without_relabelling_fixed4(tmp_path):
    """A stratified4 receipt declares a different policy from fixed4 without relabelling fixed4."""
    receipts = []
    for policy in ("fixed4", "stratified4"):
        args = run_args(tmp_path / policy, d0_policy=policy)
        _, prompt = core.select_run_prompt(args, llm_priors=False)
        label_args = SimpleNamespace(**{**vars(args), **prompt})
        receipts.append(bmd_label_receipt("core", label_args, task="structure_out__truth_F2", metadata={}))
    assert receipts[0] != receipts[1]
    assert canonical_json(receipts[0]) != canonical_json(receipts[1])


@pytest.mark.parametrize("policy", ["runner_default", "fixed4", "stratified4"])
def test_registered_policy_matches_actual_settings_without_changing_legacy_receipts(tmp_path, policy):
    """A registered policy matching the actual settings is accepted and legacy receipts are unchanged."""
    actual = bmd_runtime_settings("core", runner_args(
        d0_policy="stratified4" if policy == "stratified4" else "fixed4"))
    ref, context, manifest = registration(tmp_path, settings={**actual, "d0_policy": policy})
    context["settings"] = actual
    _, normalized = _registered_settings(manifest, "core", context["cell"])
    assert normalized["d0_policy"] == actual["d0_policy"]
    assert manifest["runtime_settings"]["core"]["d0_policy"] == policy
    assert resolve_label("production", manifest_ref=ref, **context)["run_label"] == "production"


@pytest.mark.parametrize("policy", ["unknown", "random2", "", None, True])
def test_unknown_registered_core_policy_refused(tmp_path, policy):
    """A registration naming an unknown CORE initial-design policy is refused."""
    ref, context, manifest = registration(tmp_path)
    manifest["runtime_settings"]["core"]["d0_policy"] = policy
    manifest["policies"]["core"]["d0_policy"] = policy
    ref = write_manifest(Path(ref["path"]), manifest)
    with pytest.raises(ValueError, match="CORE d0_policy"):
        resolve_label("production", manifest_ref=ref, **context)


def _synthetic_real_fits(monkeypatch):
    calls = []

    def fitted(hypothesis, x, y, seed, settings, interface, n_trials=None, **kwargs):
        """Synthetic stand-in for the Gaussian CORE fit that records its calls and returns a small particle set."""
        assert set(kwargs) == {"response_unit", "response_center"}
        calls.append({"x": list(x), "y": list(y), **kwargs})
        rng = np.random.default_rng(seed)
        particles = {name: rng.normal(0.0, 0.1, 8) for name, _ in hypothesis.parameters}
        particles["sigma"] = np.ones(8)
        return Fit(hypothesis, particles, -float(len(y)), diagnostics={
            **kwargs, "interface": interface, "seed": seed, "observation_count": len(y),
        })

    monkeypatch.setattr(gaussian, "fit_core", fitted)
    return calls


def test_profiles_share_truth_sigma_but_use_their_own_acquisition_grid(tmp_path, monkeypatch):
    """The dev and paper profiles share the true noise sigma but each uses its own acquisition grid."""
    calls = _synthetic_real_fits(monkeypatch)
    configs, acquisitions = [], []
    for profile in ("dev", "paper"):
        output = tmp_path / profile
        core.run(run_args(output, profile=profile, seed=31, rounds=1, cell="001",
                          structure="gauss_pow", truth_family="normal", initial_family="normal",
                          design_policy="uniform_random", d0_policy="stratified4"))
        configs.append(json.loads((output / "config.json").read_text()))
        acquisitions.append(json.loads((output / "acquisition/t01.json").read_text()))
    dev, paper = configs
    assert dev["noise"] == paper["noise"]
    assert dev["noise"]["grid_size"] == 100
    assert dev["noise"]["rule"] == "0.05 * sd_x(m) on linspace(0, 1, 100)"
    assert dev["design_count"] == 16 and paper["design_count"] == 100
    for config, record, count in zip(configs, acquisitions, (16, 100), strict=True):
        expected = [float(u) for u in np.linspace(0, 1, count) if u not in config["d0_designs"]]
        assert [row["design"] for row in record["rows"]] == expected
        assert len(record["rows"]) == count
        assert record["candidate_set_sha256"] == core.candidate_set_record(expected, "core")["candidate_set_sha256"]
    initial = [row for row in calls if len(row["y"]) == 4]
    assert len(initial) == 8
    center = float(np.mean(initial[0]["y"]))
    unit = float(np.std(initial[0]["y"], ddof=1))
    assert unit != dev["noise"]["sigma"]
    assert all(row["response_unit"] == unit and row["response_center"] == center for row in calls)


def test_dev_gaussian_seventeenth_round_reports_profile_and_exhausted_pool(tmp_path, monkeypatch):
    """A seventeenth dev round raises an error naming the profile and the exhausted design pool."""
    _synthetic_real_fits(monkeypatch)
    output = tmp_path / "synthetic_exhaustion"
    with pytest.raises(ValueError, match="Gaussian CORE design pool exhausted: profile=dev, grid_size=16"):
        core.run(run_args(output, profile="dev", rounds=17, seed=31, cell="000",
                          structure="gauss_pow", truth_family="normal", initial_family="normal",
                          d0_policy="stratified4"))
    rows = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    acquired = [row["u"] for row in rows if row["round"] > 0]
    assert len(acquired) == len(set(acquired)) == 16
    assert sorted(acquired) == np.linspace(0, 1, 16).tolist()


def test_old_variable_sigma_grid_rule_is_refused_by_frozen_rescore():
    """The frozen rescoring refuses the old variable-sigma grid rule."""
    from .frozen_kl_rescore import Rejection, gaussian_world_from_config
    from .test_gaussian_core import config_for

    config = config_for(core.CoreWorld(31, "gauss_pow", "normal"))
    config["noise"]["rule"] = "0.05 * sd_x(m) on linspace(0, 1, grid_size)"
    with pytest.raises(Rejection, match="sigma/grid"):
        gaussian_world_from_config(config)


def _points_with_authentic_identity(output, config, source):
    with (output / "points.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        headers, rows = reader.fieldnames, list(reader)
    assert headers and len(set(headers)) == len(headers)
    assert set(CSV_IDENTITY_FIELDS) <= set(headers)
    assert rows and all(set(row) == set(headers) and None not in row.values() for row in rows)
    code = config["source_files_sha256"]["core_count.py"]
    assert is_hex64(code)
    assert {row["code_sha"] for row in rows} == {code}
    assert code == hashlib.sha256((source / "src/bayes_model_discovery/experimental/core_count.py").read_bytes()).hexdigest()
    scores = output / "scores.jsonl"
    assert scores == Path(config["output"]) / "scores.jsonl"
    lines = scores.read_text().splitlines()
    pattern = re.compile(re.escape(str(scores)) + r"#L([1-9][0-9]*)\(round=([0-9]+),phase=([A-Za-z_]+)\)")
    references = set()
    for row in rows:
        match = pattern.fullmatch(row["source_path"])
        assert match, row["source_path"]
        line, round_number, phase = match.groups()
        assert 1 <= int(line) <= len(lines)
        score = json.loads(lines[int(line) - 1])
        assert score["round"] == int(round_number) and score["phase"] == phase
        assert row["source_path"] not in references
        references.add(row["source_path"])
    return headers, [{key: value for key, value in row.items() if key not in CSV_IDENTITY_FIELDS}
                     for row in rows]
