"""Scalar-bank disagreement and ChemBench study routing; synthetic, no provider calls."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from . import chembench_core as chem
from . import count_proposer
from . import positive_inference as positive
from .count_inference import Fit
from .count_proposer import AnthropicCountProposer
from .design_policies import point_disagreement as count_mean_disagreement
from .test_chembench_core import _args, _manifest
from .test_chembench_mda_options import stub_fits


def scalar_fit(family, means, slopes, log_weight, noise):
    """Build a synthetic scalar-regression Fit with the given means, slopes, weight and noise."""
    h = positive.RegressionHypothesis(family, (0,))
    particles = {"b0": np.log(np.asarray(means, dtype=float)),
                 "b1": np.asarray(slopes, dtype=float),
                 h.noise_parameters[0]: np.asarray(noise, dtype=float)}
    return Fit(h, particles, log_weight, [log_weight, log_weight],
               {"input_transform": {"center": [0.], "scale": [1.]},
                "response_unit": 1.}, 0.)


def banks():
    """Return two synthetic scalar fits (lognormal and gamma) used as a model pool."""
    return [scalar_fit("lognormal", [1., 3.], [.2, -.1], np.log(.25), [.1, 3.]),
            scalar_fit("gamma", [2., 4.], [-.3, .4], np.log(.75), [2., 30.])]


def test_scalar_disagreement_is_exact_count_mean_variance_definition(monkeypatch):
    """Test that scalar disagreement is exact count mean variance definition."""
    fits, inputs = banks(), np.array([[0.], [.3], [.8]])
    means = np.stack([positive.predictive_means(f, inputs).mean(0) for f in fits])
    weights = np.exp(positive.validate_pool(fits))
    expected = count_mean_disagreement(means, weights)
    monkeypatch.setattr(positive, "draw_conditional", lambda *a, **k: pytest.fail("outcome draw"))
    before = np.random.get_state()
    actual = positive.point_disagreement(fits, inputs)
    after = np.random.get_state()
    np.testing.assert_array_equal(actual, expected)
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_allclose(actual, weights[0] * weights[1] * (means[0] - means[1])**2)


def test_equal_means_ignore_within_model_particle_spread_and_noise():
    """Test that equal means ignore within model particle spread and noise."""
    fits = [scalar_fit("lognormal", [1., 3.], [0., 0.], np.log(.2), [.1, 5.]),
            scalar_fit("gamma", [2., 2.], [0., 0.], np.log(.8), [1., 50.])]
    np.testing.assert_allclose(positive.point_disagreement(fits, [[0.], [.5]]), 0., atol=1e-28)
    fits[0].particles["sigma"] *= 2
    fits[1].particles["shape"] /= 3
    np.testing.assert_allclose(positive.point_disagreement(fits, [[0.], [.5]]), 0., atol=1e-28)


def test_scalar_disagreement_uses_zero_inflated_predictive_mean(monkeypatch):
    """Test that scalar disagreement uses zero inflated predictive mean."""
    fits = [SimpleNamespace(hypothesis=SimpleNamespace(family=family, measure="PMF", model_id=family),
                            particles={"pi": np.array([.25, .75])},
                            diagnostics={}, log_evidence=0.)
            for family in ("zi_poisson", "poisson")]
    monkeypatch.setattr(positive, "particle_means", lambda *a: np.full((2, 2), 8.))
    # ZIP particle-averaged predictive mean is 4; ordinary Poisson mean is 8.
    np.testing.assert_array_equal(positive.point_disagreement(fits, [[0.], [.5]]), [4., 4.])


def test_scalar_disagreement_rejects_nonfinite_mean_without_clipping():
    """Test that scalar disagreement rejects nonfinite mean without clipping."""
    fits = banks()
    fits[0].particles["b0"][:] = np.inf
    with pytest.raises(ValueError, match="non-finite/nonpositive mean"):
        positive.point_disagreement(fits, [[0.]])


def test_scalar_disagreement_keeps_pool_binding_validation():
    """Test that scalar disagreement keeps pool binding validation."""
    fits = banks()
    fits[1].diagnostics["response_unit"] = 2.
    with pytest.raises(ValueError, match="pool must share"):
        positive.point_disagreement(fits, [[0.]])


@pytest.fixture
def synthetic_chem(monkeypatch):
    """Stub the fits and shrink the ChemBench dev profile to a tiny design pool."""
    stub_fits(monkeypatch)
    monkeypatch.setattr(chem, "OFFICIAL_EVALUATION_COUNT", 8)
    monkeypatch.setitem(chem.PROFILE_GRID, "dev", {**chem.PROFILE_GRID["dev"],
                                                 "design_pool": 8, "eig_samples": 8})


def run_chem(output, manifest, policy, **overrides):
    """Run the ChemBench runner with the given policy and return its config.json as a dict."""
    options = {"cell": "001", "structure_proposals": "catalog", "rounds": 3, "design_policy": policy}
    options.update(overrides)
    chem.run(_args(output, manifest, **options))
    return json.loads((output / "config.json").read_text())


@pytest.mark.usefixtures("synthetic_chem")
@pytest.mark.parametrize("policy", ("point_disagreement", "space_filling"))
def test_chem_study_routing_uses_existing_pool_and_never_calls_information_estimators(
        tmp_path, monkeypatch, policy):
    """Test that chem study routing uses existing pool and never calls information estimators."""
    monkeypatch.setattr(chem, "joint_eig", lambda *a, **k: pytest.fail("EIG reached by study policy"))
    monkeypatch.setattr(chem, "choose_epig", lambda *a, **k: pytest.fail("EPIG reached by study policy"))
    calls = []
    if policy == "point_disagreement":
        original = chem.point_disagreement
        def objective(fits, inputs):
            values = original(fits, inputs)
            calls.append((inputs.copy(), values.copy()))
            return values
        monkeypatch.setattr(chem, "point_disagreement", objective)
    else:
        original = chem.select_space_filling
        def objective(candidates, history, bounds):
            result = original(candidates, history, bounds)
            calls.append((candidates.copy(), history.copy(), np.asarray(bounds), result))
            return result
        monkeypatch.setattr(chem, "select_space_filling", objective)
        monkeypatch.setattr(chem, "point_disagreement", lambda *a: pytest.fail("space queried posterior"))
    output = tmp_path / policy
    config = run_chem(output, _manifest(tmp_path), policy)
    assert config["design_policy"] == policy
    assert config["design_policy_definition"]
    assert config["design_pool"]["eig_estimator"] is None
    assert config["design_pool"]["uses_rng"] is False
    records = [json.loads(p.read_text()) for p in sorted((output / "acquisition").glob("*.json"))]
    observations = [json.loads(s) for s in (output / "observations.jsonl").read_text().splitlines()]
    assert len(calls) == len(records) == 3
    selected = []
    for index, (record, call) in enumerate(zip(records, calls, strict=True)):
        assert len(record["rows"]) == 8 - index
        assert record["policy"]["excluded_pool_indices"] == sorted(selected)
        assert record["policy"]["uses_rng"] is False
        assert not any("eig" in row or "epig" in row for row in record["rows"])
        selected.append(record["policy"]["chosen_pool_index"])
        assert record["chosen"] == list(observations[4 + index]["x"].values())
        if policy == "point_disagreement":
            np.testing.assert_array_equal([r["point_disagreement"] for r in record["rows"]], call[1])
            assert record["policy"]["posterior_particles"] == "all_retained"
        else:
            expected = np.array([list(r["x"].values()) for r in observations[:4 + index]])
            np.testing.assert_array_equal(call[1], expected)
            np.testing.assert_array_equal(call[2], config["design_normalization_bounds"])
            assert record["policy"]["observed_design_count"] == 4 + index
    assert len(set(selected)) == 3


@pytest.mark.usefixtures("synthetic_chem")
@pytest.mark.parametrize("policy", ("point_disagreement", "space_filling"))
def test_chem_study_acquisition_and_observations_are_deterministic(tmp_path, policy):
    """Test that chem study acquisition and observations are deterministic."""
    manifest = _manifest(tmp_path)
    left, right = tmp_path / "left", tmp_path / "right"
    run_chem(left, manifest, policy)
    run_chem(right, manifest, policy)
    assert (left / "observations.jsonl").read_bytes() == (right / "observations.jsonl").read_bytes()
    for path in (left / "acquisition").glob("*.json"):
        assert path.read_bytes() == (right / "acquisition" / path.name).read_bytes()


@pytest.mark.usefixtures("synthetic_chem")
@pytest.mark.parametrize("policy", ("point_disagreement", "space_filling"))
@pytest.mark.parametrize("cell", ("000", "002"))
def test_chem_new_policy_refuses_random_or_epig_slot_before_output(tmp_path, policy, cell):
    """Test that chem new policy refuses random or epig slot before output."""
    output = tmp_path / "refused"
    with pytest.raises(ValueError):
        run_chem(output, _manifest(tmp_path), policy, cell=cell)
    assert not output.exists()


@pytest.mark.usefixtures("synthetic_chem")
@pytest.mark.parametrize("policy", ("point_disagreement", "space_filling"))
def test_chem_study_policy_with_injected_proposer_keeps_transport_and_archive(tmp_path, monkeypatch, policy):
    """Test that chem study policy with injected proposer keeps transport and archive."""
    calls = []
    def create(**body):
        calls.append(body)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(
                {"expression": "b0", "family": "lognormal", "parameters": {"b0": "intercept"}}))],
            usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 1, "output_tokens": 1}),
            id="synthetic-study-response", model="synthetic-study-model", stop_reason="end_turn")
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(count_proposer, "AnthropicCountProposer",
                        lambda records, **kwargs: AnthropicCountProposer(records, client=client))
    output = tmp_path / "injected"
    config = run_chem(output, _manifest(tmp_path), policy, cell="101", structure_proposals="free",
                      proposer="anthropic", allow_paid_api=True, rounds=2)
    assert calls
    assert config["points_variant"] == f"design_{policy}+structure_free"
    proposal = json.loads((output / "proposals" / "t01.json").read_text())
    assert proposal["transport"] and proposal["admitted_ids"] == []
    packet = json.loads((output / "packets" / "t02.json").read_text())
    assert any(entry["outcome"] == "duplicate" for entry in packet["archive"])
