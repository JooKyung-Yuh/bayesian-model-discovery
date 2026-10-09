"""Synthetic Gaussian-world (CORE) acceptance checks, with no paid calls or stored run fixtures."""

import ast
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from . import core_count as core
from . import gaussian_core as adapter
from . import gaussian_core_constants as constants
from .belief_packet import cell_constraints, core_interface, core_real_interface
from .count_inference import Fit
from .count_models import evaluate, initial_hypotheses, standardized_expression
from .frozen_kl_rescore import Rejection, gaussian_world_from_config, rescore_cell

REGIMES = ("gauss_den", "gauss_pow", "gauss_call")


def sha(path):
    """SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_for(world):
    """Build the run-config dict that a Gaussian-world run would record for `world`."""
    return {
        "seed": world.seed, "structure": world.structure, "truth_family": world.family,
        "replicate": world.replicate, "outcome_type": "real",
        "interface": core_real_interface(), "noise": world.noise_record(),
        "constant_selection_sha256": sha(constants.__file__),
        "gauss_den_c": constants.GAUSS_DEN_C, "gauss_pow_e": constants.GAUSS_POW_E,
        "gauss_call_k": constants.GAUSS_CALL_K,
    }


@pytest.mark.parametrize("seed", tuple(core.WORLD_COEFFICIENTS))
@pytest.mark.parametrize("structure", REGIMES)
@pytest.mark.parametrize("grid_size", (16, 100))
def test_sigma_is_one_full_precision_sample_sd(seed, structure, grid_size):
    """The noise sigma is 5% of the full-precision sample SD of the mean curve, and the record binds its grid."""
    grid = np.linspace(0, 1, grid_size)
    world = core.CoreWorld(seed, structure, "normal", design_grid=grid)
    noise = world.noise_record()
    expected = float(0.05 * np.std(world.mean(grid), ddof=1))
    assert np.isfinite(world.sigma) and world.sigma > 0
    assert noise["sigma"] == world.sigma == expected
    assert noise["sigma_6sf"] == format(expected, ".6g")
    assert noise["ddof"] == 1 and noise["grid_size"] == grid_size
    canonical = json.dumps(grid.tolist(), sort_keys=True, separators=(",", ":")).encode()
    assert noise["grid_canonical_json"].encode() == canonical
    assert noise["grid_sha256"] == hashlib.sha256(canonical).hexdigest()
    encoded = json.loads(json.dumps(config_for(world)))
    assert encoded["noise"]["sigma"] == expected


@pytest.mark.parametrize("seed", tuple(core.WORLD_COEFFICIENTS))
def test_means_equal_the_declared_literal_formulas(seed):
    """Each regime's mean curve equals its declared literal formula exactly."""
    u = np.array([0.0, 0.07, 0.25, 0.61, 1.0])
    b0, b1 = core.WORLD_COEFFICIENTS[seed]
    expected = (b0 + b1 * u / (0.07 + u),
                b0 + b1 * u**0.40,
                b0 + b1 * np.tanh(5 * (2*u - 1)))
    for regime, reference in zip(REGIMES, expected, strict=True):
        world = core.CoreWorld(seed, regime, "normal")
        np.testing.assert_array_equal(world.mean(u), reference)


@pytest.mark.parametrize("seed", tuple(core.WORLD_COEFFICIENTS))
def test_power_zero_and_monotonicity_without_derivatives(seed):
    """The power regime equals b0 at u=0 and is monotone, with no floating-point errors raised."""
    world = core.CoreWorld(seed, "gauss_pow", "normal")
    b0, b1 = core.WORLD_COEFFICIENTS[seed]
    with np.errstate(all="raise"):
        assert world.mean(0.0) == b0
        means = world.mean(np.linspace(0, 1, 1001))
    assert np.isfinite(means).all()
    assert np.all(np.diff(means) * np.sign(b1) >= 0)


def test_selection_reproduces_constants_on_the_engine_standardization():
    """Constant selection returns (0.07, 0.40, 5) on the engine's own standardization of the grid."""
    assert constants.select_constants() == (0.07, 0.40, 5)
    expression = standardized_expression("u", 0.0, 1.0)
    np.testing.assert_array_equal(constants.standardized(constants.GRID),
                                  evaluate(expression, {}, constants.GRID))
    details = constants.selection_details()
    assert [f"{row[1]:.3f}" for row in details.values()] == ["0.986", "0.958", "1.062"]
    assert [f"{row[2]:.1f}" for row in details.values()] == ["26.2", "24.8", "30.4"]
    assert constants.DENOMINATORS == tuple(round(0.02 + i * 0.01, 2) for i in range(19))
    assert 0.5 not in constants.EXPONENTS
    assert constants.CALL_SCALES == tuple(range(4, 21))


def test_selector_never_reads_benchmark_tables_or_runs_during_a_world(monkeypatch):
    """The selector imports only argparse and numpy, names no benchmark, and is never called while a world runs."""
    source = Path(constants.__file__).read_text()
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert imports == ["argparse", "numpy"]
    assert all(word not in source.lower() for word in ("autoscilab", "chembench", "playground"))
    def forbidden():
        raise AssertionError("selection is not a run-time operation")
    monkeypatch.setattr(constants, "select_constants", forbidden)
    monkeypatch.setattr(constants, "selection_details", forbidden)
    for regime in REGIMES:
        assert np.isfinite(core.CoreWorld(31, regime, "normal").mean(0.3))


def test_select_command_declares_the_development_proxy():
    """The --select command prints the chosen constants and names the development proxy."""
    result = subprocess.run(
        [sys.executable, "-B", "-m", constants.__name__, "--select"],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    for value in ("0.07", "0.4", "5", "0.986", "0.958", "1.062", "26.2", "24.8", "30.4"):
        assert value in result.stdout
    assert constants.DESIGN_PROXY_NAME in result.stdout
    assert "evidence=" not in result.stdout


def test_one_truth_sigma_in_generation_scoring_record_and_frozen_rescore(monkeypatch):
    """One truth sigma is computed per world and reused by generation, scoring, the record and the frozen rescore."""
    calls = []
    original_std = np.std
    def counted_std(*args, **kwargs):
        calls.append(kwargs.get("ddof"))
        return original_std(*args, **kwargs)
    monkeypatch.setattr(np, "std", counted_std)
    world = core.CoreWorld(31, "gauss_den", "normal")
    assert calls == [1]
    sampled = []
    class SpyRng:
        def normal(self, mean, sigma, size=None):
            sampled.append(sigma)
            return np.asarray(mean)
    world.sample(np.array([0.2]), SpyRng())
    u, y, truth = world.evaluation(3)
    assert calls == [1]
    assert sampled[0] is world.sigma and truth[1] is world.sigma
    assert world.noise_record()["sigma"] is world.sigma
    law_sigmas = []
    real_law = adapter.normal_law
    def law(mean, sigma):
        law_sigmas.append(sigma)
        return real_law(mean, sigma)
    monkeypatch.setattr(adapter, "normal_law", law)
    monkeypatch.setattr(adapter.scalar, "predictive_summary", lambda *args: {
        "mean": truth[0], "lower90": truth[0]-world.sigma, "upper90": truth[0]+world.sigma})
    monkeypatch.setattr(adapter.scalar, "log_predictive", lambda *args: (np.zeros(y.shape), None))
    monkeypatch.setattr(adapter, "kl_against_pool", lambda *args: {
        "kl": 0.0, "kl_method": "synthetic", "kl_truncated_mass": 0.0, "kl_quadrature_error": 0.0})
    adapter.score([], u, y, truth, 0)
    assert law_sigmas == [world.sigma] and law_sigmas[0] is world.sigma
    restored = gaussian_world_from_config(config_for(world))
    assert restored.sigma == world.sigma
    assert calls == [1, 1]  # Exactly once for each independent world construction.


def test_candidate_noise_scale_is_public_d0_not_the_truth(monkeypatch):
    """The candidate noise scale is calibrated on the public initial data, not on the true sigma."""
    world = core.CoreWorld(31, "gauss_call", "normal")
    public = [world.observe(u) for u in core.D0]
    center, unit = adapter.calibrate(public)
    assert unit == np.std([row["y"] for row in public], ddof=1)
    assert unit != world.sigma
    captured = {}
    def fake_fit(*args, **kwargs):
        captured.update(kwargs)
        return "sentinel"
    monkeypatch.setattr(adapter.scalar, "fit", fake_fit)
    result = adapter.fit_core(adapter.as_real(initial_hypotheses("normal")[0]),
                              [row["u"] for row in public], [row["y"] for row in public],
                              123, object(), core_real_interface(),
                              response_unit=unit, response_center=center)
    assert result == "sentinel"
    assert captured["response_unit"] == unit
    assert captured["response_center"] == center


def test_real_interface_and_family_constraints_are_literal():
    """The real-valued interface and the family constraints equal their literal specifications."""
    assert core_real_interface() == {
        "outcome_type": "real", "inputs": [{"name": "u", "range": [0, 1]}],
        "derived_inputs": {"z": "2*u-1"}, "link": "identity",
    }
    constraints = cell_constraints("100", "normal", allowed_families=("normal",))
    assert constraints["allowed_families"] == ["normal"]
    assert not constraints["family_expansion"]
    assert "support" not in core_real_interface() and "trials" not in core_real_interface()
    assert core_interface() == {
        "outcome_type": "bounded_count", "support": [0, 50], "trials": 50,
        "inputs": [{"name": "u", "range": [0, 1]}],
        "derived_inputs": {"z": "2*u-1"}, "link": "logit",
    }


@pytest.mark.parametrize("structure", REGIMES)
def test_new_observation_streams_are_separate_and_repeat_keyed(structure):
    """Gaussian-world observations use their own seed stream, keyed by repeat and separate from the count world's."""
    old = core.CoreWorld(31, "out", "binomial")
    world = core.CoreWorld(31, structure, "normal")
    first = world.observe(0.35)
    second = world.observe(0.35)
    assert first["observation_seed"] != old.observe(0.35)["observation_seed"]
    assert first["observation_seed"] != second["observation_seed"]
    assert first == core.CoreWorld(31, structure, "normal").observe(0.35)
    assert isinstance(first["y"], float)
    np.testing.assert_array_equal(world.evaluation(5)[0], old.evaluation(5)[0])


@pytest.mark.parametrize("options,reason", [
    (["--structure", "gauss_den", "--truth-family", "binomial"], "require --truth-family normal"),
    (["--structure", "out", "--truth-family", "normal"], "count structures forbid normal"),
    (["--structure", "gauss_den", "--truth-family", "normal", "--cell", "110"], "family_expansion"),
    (["--structure", "gauss_den", "--truth-family", "normal", "--priors", "llm"], "fixed role priors"),
    (["--structure", "gauss_den", "--truth-family", "normal", "--run-label", "production"], "development-only"),
])
def test_cli_refusal_leaves_no_output(tmp_path, options, reason):
    """Each invalid CLI combination exits with code 2, states its reason and writes no output."""
    out = tmp_path / "absent"
    result = subprocess.run([sys.executable, "-B", "-m", core.__name__, "--output", str(out),
                             "--profile", "dev", "--proposer", "mock", "--cell", "100", *options],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 2, result.stdout + result.stderr
    assert reason in result.stderr
    assert not out.exists()


@pytest.fixture(scope="module", params=REGIMES)
def real_mock_runs(request, tmp_path_factory):
    """Run the mock-proposer CLI twice per regime and return the two output directories."""
    root = tmp_path_factory.mktemp(request.param)
    outputs = []
    for index in range(2):
        out = root / f"run_{index}"
        command = [sys.executable, "-B", "-m", core.__name__, "--output", str(out),
                   "--profile", "dev", "--rounds", "2", "--cell", "100", "--proposer", "mock",
                   "--seed", "31", "--structure", request.param, "--truth-family", "normal"]
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=300)
        (root / f"run_{index}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(out)
    return outputs


def test_three_regimes_complete_real_smc_reproducibly(real_mock_runs):
    """All three regimes complete real SMC runs whose artifacts are byte-identical across two runs."""
    first, second = real_mock_runs
    for name in ("archive.json", "observations.jsonl", "scores.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    config = json.loads((first / "config.json").read_text())
    assert config["constant_selection_sha256"] == sha(constants.__file__)
    assert config["truth_family"] == config["initial_family"] == "normal"
    assert config["allowed_families"] == ["normal"] and not config["family_expansion"]
    for row in map(json.loads, (first / "scores.jsonl").read_text().splitlines()):
        for key in ("nll", "kl", "coverage90", "width90", "rmse", "mse"):
            assert np.isfinite(row["bma"][key]), (key, row)
    packets = list((first / "packets").glob("*.json"))
    assert len(packets) == 2
    for path in packets:
        packet = json.loads(path.read_text())
        assert packet["interface"] == core_real_interface()
        assert packet["constraints"]["allowed_families"] == ["normal"]
        assert not packet["constraints"]["family_expansion"]
    summary = json.loads((first / "summary.json").read_text())
    assert summary["paid_api_calls"] == 0
    observations = [json.loads(line) for line in (first / "observations.jsonl").read_text().splitlines()]
    assert len({row["u"] for row in observations}) == len(observations)
    policy = json.loads((first / "frozen_d0_policy.json").read_text())
    for path in (first / "fits").glob("*.json"):
        model = json.loads(path.read_text())
        assert model["family"] == "normal"
        assert model["diagnostics"]["response_unit"] == policy["response_unit"]
        assert model["diagnostics"]["response_center"] == policy["response_center"]


def test_frozen_rescore_uses_the_same_normal_truth(real_mock_runs):
    """The frozen rescore reproduces a recorded run against the same normal truth and modifies no file."""
    first = real_mock_runs[0]
    before = {str(path.relative_to(first)): sha(path) for path in first.rglob("*") if path.is_file()}
    record = rescore_cell(first, str(first), "core", None)
    assert record["status"] == "reproduced", record
    assert record["delta"] == 0.0
    assert record["evaluation_truth_match"]
    assert "gauss_legendre" in record["kl_method"]
    assert before == {str(path.relative_to(first)): sha(path) for path in first.rglob("*") if path.is_file()}


@pytest.mark.parametrize("field,value", [("sigma", 1.0), ("ddof", 0), ("grid_sha256", "bad")])
def test_frozen_rescore_refuses_a_changed_noise_binding(field, value):
    """The frozen rescore rejects a config whose noise binding (sigma, ddof or grid hash) was changed."""
    config = config_for(core.CoreWorld(31, "gauss_pow", "normal"))
    config["noise"][field] = value
    with pytest.raises(Rejection, match="sigma/grid"):
        gaussian_world_from_config(config)


@pytest.mark.parametrize("policy", ("joint_eig", "point_disagreement", "uniform_random", "space_filling"))
def test_scalar_acquisition_uses_existing_real_engine(policy):
    """Every scalar acquisition policy picks a grid design through the existing real-valued engine."""
    h = adapter.as_real(initial_hypotheses("normal")[0])
    fitted = Fit(h, {"b0": np.array([-0.5, 0.5]), "sigma": np.ones(2)}, 0.0,
                 diagnostics={"response_unit": 1.0, "response_center": 0.0,
                              "interface": core_real_interface()})
    grid = np.array([0.0, 0.5, 1.0])
    kwargs = {"observed_designs": [0.5], "bounds": [[0, 1]]} if policy == "space_filling" else {}
    chosen, rows, metadata = adapter.select_design(
        [fitted], grid, 12, policy, 16, core_real_interface(), **kwargs)
    assert chosen in grid and len(rows) == len(grid)
    assert metadata["design_policy"] == policy


def test_scalar_epig_uses_existing_real_engine():
    """Scalar EPIG picks a grid design through the existing real-valued engine."""
    from .count_inference import choose_epig
    h = adapter.as_real(initial_hypotheses("normal")[0])
    fitted = Fit(h, {"b0": np.array([-0.5, 0.5]), "sigma": np.ones(2)}, 0.0,
                 diagnostics={"response_unit": 1.0, "response_center": 0.0,
                              "interface": core_real_interface()})
    grid = np.array([0.0, 0.5, 1.0])
    chosen, rows = choose_epig([fitted], grid, 12, 16, core_real_interface(), None,
                               target_points=np.array([0.25, 0.75]))
    assert chosen in grid and len(rows) == len(grid)


def test_opt_in_count_world_baseline_methods_and_values_are_unchanged():
    """Opt-in: the count-world methods and values match the pinned baseline checkout."""
    if os.environ.get("BMD_BASELINE_COMPARE") != "1":
        pytest.skip("set BMD_BASELINE_COMPARE=1 and BMD_GW_BASELINE_DIR for count baseline")
    base = os.environ.get("BMD_GW_BASELINE_DIR")
    if not base:
        pytest.skip("BMD_GW_BASELINE_DIR is required")
    head = subprocess.run(["git", "-C", base, "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    assert head == "a58257478ef448d2c0bbbfeedbda85a2fcff95b4"
    path = Path(base) / "src/bayes_model_discovery/experimental/core_count.py"
    subprocess.run(["git", "-C", base, "diff", "--exit-code", "HEAD", "--", str(path)],
                   check=True, capture_output=True)
    source = path.read_text()
    old = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "CoreWorld")
    current = Path(core.__file__).read_text()
    new = next(node for node in ast.parse(current).body
               if isinstance(node, ast.ClassDef) and node.name == "CoreWorld")
    for name in ("probability", "sample", "observe", "evaluation"):
        before = next(node for node in old.body if getattr(node, "name", None) == name)
        after = next(node for node in new.body if getattr(node, "name", None) == name)
        assert ast.get_source_segment(source, before) == ast.get_source_segment(current, after)
    namespace = {key: getattr(core, key) for key in
                 ("WORLD_COEFFICIENTS", "seed_for", "np", "expit", "betabinom", "binom")}
    exec(compile(ast.Module(body=[old], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 - pinned local baseline class only
    old_world = namespace["CoreWorld"]
    for structure in ("in", "out", "den", "pow"):
        for family in ("binomial", "beta_binomial", "zi_binomial"):
            before, after = old_world(31, structure, family), core.CoreWorld(31, structure, family)
            u = np.array([0.0, 0.1, 0.3, 0.7, 1.0])
            np.testing.assert_array_equal(before.probability(u), after.probability(u))
            np.testing.assert_array_equal(before.sample(u, np.random.default_rng(123)),
                                          after.sample(u, np.random.default_rng(123)))
            assert json.dumps([before.observe(x) for x in (0.1, 0.7, 0.1)]) == json.dumps(
                [after.observe(x) for x in (0.1, 0.7, 0.1)])
            for left, right in zip(before.evaluation(8), after.evaluation(8), strict=True):
                np.testing.assert_array_equal(left, right)
    old_interface_source = (Path(base) / "src/bayes_model_discovery/experimental/belief_packet.py").read_text()
    old_function = next(node for node in ast.parse(old_interface_source).body
                        if getattr(node, "name", None) == "core_interface")
    assert ast.get_source_segment(old_interface_source, old_function) == inspect.getsource(core_interface).strip()


@pytest.mark.skip(reason="parity with the baseline recorded before the model-identity change crosses the ID/RNG boundary; a successor parity test is still owed")
def test_opt_in_legacy_count_runner_keeps_scientific_bytes(tmp_path):
    """Opt-in: the legacy count runner keeps its scientific output bytes (currently skipped)."""
    if os.environ.get("BMD_BASELINE_COMPARE") != "1" or not os.environ.get("BMD_GW_BASELINE_DIR"):
        pytest.skip("set BMD_BASELINE_COMPARE=1 and BMD_GW_BASELINE_DIR for count runner parity")
    base, candidate = Path(os.environ["BMD_GW_BASELINE_DIR"]), Path(__file__).resolve().parents[3]
    harness = """
import sys
from pytest import MonkeyPatch
from bayes_model_discovery.experimental.test_count_proposer import run_args, stub_fits
with MonkeyPatch.context() as patch:
    core = stub_fits(patch)
    core.run(run_args(sys.argv[1], rounds=2))
"""
    configs, outputs = [], []
    for label, root in (("baseline", base), ("candidate", candidate)):
        output = tmp_path / label / "synthetic_cell"
        env = dict(os.environ, PYTHONPATH=f"{root}/src:{root}",
                   OPENAI_API_KEY="", ANTHROPIC_API_KEY="", OPENAI_COMPAT_API_KEY="",
                   OPENROUTER_API_KEY="", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", "-c", harness, str(output)],
                                cwd=root, env=env, text=True, capture_output=True,
                                check=False, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        configs.append(json.loads((output / "config.json").read_text()))
        outputs.append(output)
    for name in ("archive.json", "observations.jsonl", "scores.jsonl"):
        if name == "scores.jsonl":
            score_rows = [
                [json.dumps({key: value for key, value in json.loads(line).items()
                             if key not in core.PWC2_SCORE_ADDED_KEYS}, sort_keys=True)
                 for line in (output / name).read_text().splitlines()]
                for output in outputs
            ]
            assert score_rows[0] == score_rows[1], name
        else:
            assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes(), name
    identity = ("source_files_sha256", "created_utc", "output", "manifest_ref", "python")
    d0s_added_keys = ("d0_policy", "d0_designs")
    left, right = configs
    assert set(right) - set(left) == set(d0s_added_keys)
    assert set(left) - set(right) == set()
    assert right["d0_policy"] == core.D0_RULE
    assert right["d0_designs"] == [0.1, 0.35, 0.65, 0.9]
    assert {k: v for k, v in left.items() if k not in identity} == {
        k: v for k, v in right.items() if k not in identity + d0s_added_keys}
    assert {name for name, digest in left["source_files_sha256"].items()
            if digest != right["source_files_sha256"][name]} == {"core_count.py", "belief_packet.py"}


def test_same_round_scalar_truncation_rejects_historical_fits():
    """Same-round truncation reports no change for identical fits and rejects fits from another round."""
    h = adapter.as_real(initial_hypotheses("normal")[0])
    result = Fit(h, {"b0": np.array([-0.5, 0.5]), "sigma": np.ones(2)}, 0.0,
                 diagnostics={"interface": core_real_interface(), "response_unit": 1.0,
                              "response_center": 0.0, "public_data_sha256": "abc123",
                              "round": 2, "observation_count": 6})
    diagnostic, _ = adapter.same_round_truncation([result], [result], np.array([0.2, 0.7]), 2, 6)
    assert diagnostic["mean_predictive_kl_before_to_active"] == 0
    assert diagnostic["mean_predictive_total_variation"] == 0
    assert not diagnostic["historical_retired_posteriors_used"]
    with pytest.raises(ValueError, match="same round"):
        adapter.same_round_truncation([result], [result], np.array([0.2, 0.7]), 3, 7)
