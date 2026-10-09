"""Model expected information gain (EIG) and Parameter EIG as opt-in BoxingGym arms.

joint EIG I((M,theta);Y) = Model EIG I(M;Y) + Parameter EIG E_M I(theta;Y|M). The exhaustive
(count) path already returns all three columns; the scalar path gets them from the same draws
as the joint estimate (`joint_eig(..., decompose=True)`), so its default call is unchanged.
"""
import sys

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import poisson

from . import boxinggym_core as bg
from . import run_label_policy
from .count_inference import Fit
from .positive_inference import RegressionHypothesis, joint_eig, particle_means

DIAGNOSTICS = {
    "input_transform": {"center": (0.0, 0.0, 0.0), "scale": (1.0, 1.0, 1.0)},
    "response_unit": 3.0,
    "observation_count": 5,
    "public_data_sha256": "synthetic",
}
INPUTS = np.array([[0.0, -1.0, 9.0], [4.0, 1.0, -2.0], [0.0, 0.5, 0.0]])
BASE_KEYS = {"eig", "mc_standard_error", "outer_samples", "negative_estimates_clipped", "target", "seed"}


def _fit(family, features, log_evidence, particles=40, spread=0.3, seed=0):
    """A synthetic posterior bank: `particles` draws around fixed coefficients."""
    rng = np.random.default_rng(seed)
    bank = {"b0": np.log(2.0) + spread * rng.standard_normal(particles)}
    for index in range(len(features)):
        bank[f"b{index + 1}"] = 0.5 + spread * rng.standard_normal(particles)
    if family == "lognormal":
        bank["sigma"] = np.full(particles, 0.6)
    if family == "gamma":
        bank["shape"] = np.full(particles, 3.0)
    return Fit(RegressionHypothesis(family, features), bank, log_evidence, diagnostics=dict(DIAGNOSTICS))


def _pool(family="poisson", **kwargs):
    return [_fit(family, (1,), 0.0, seed=1, **kwargs), _fit(family, (), -0.4, seed=2, **kwargs)]


def test_default_call_is_unchanged_and_the_parts_sum_to_the_joint_estimate():
    """The default call is unchanged and the structure and parameter parts sum to the joint estimate."""
    for family in ("poisson", "lognormal", "gamma"):
        pool = _pool(family)
        default = joint_eig(pool, INPUTS, 7, 256)
        parts = joint_eig(pool, INPUTS, 7, 256, decompose=True)
        assert set(default) == BASE_KEYS
        assert set(parts) == BASE_KEYS | {"structure_eig", "parameter_eig", "structure_mc_standard_error",
                                          "parameter_mc_standard_error"}
        np.testing.assert_array_equal(parts["eig"], default["eig"])
        np.testing.assert_array_equal(parts["mc_standard_error"], default["mc_standard_error"])
        assert {k: parts[k] for k in BASE_KEYS - {"eig", "mc_standard_error"}} == {
            k: default[k] for k in BASE_KEYS - {"eig", "mc_standard_error"}}
        np.testing.assert_allclose(parts["structure_eig"] + parts["parameter_eig"], parts["eig"], rtol=0, atol=1e-12)
    assert joint_eig.__defaults__ == (64,) and joint_eig.__kwdefaults__ == {"decompose": False}


def _exact_poisson(pool, inputs, support=400):
    """The three mutual informations by summation over y = 0..support-1 (particle banks exact)."""
    y = np.arange(support)
    weights = np.exp(np.array([f.log_evidence for f in pool]) - logsumexp([f.log_evidence for f in pool]))
    conditional = [poisson.logpmf(y[None, None, :], particle_means(f, inputs)[:, :, None]) for f in pool]
    model = [logsumexp(c, axis=0) - np.log(c.shape[0]) for c in conditional]  # [input, y]
    mixed = logsumexp(np.stack(model) + np.log(weights)[:, None, None], axis=0)
    structure = sum(w * (np.exp(m) * (m - mixed)).sum(-1) for w, m in zip(weights, model, strict=True))
    joint = sum(w * (np.exp(c) * (c - mixed[None])).sum(-1).mean(0)
                for w, c in zip(weights, conditional, strict=True))
    return joint, structure, joint - structure


def test_the_parts_estimate_the_exact_mutual_informations():
    """The parts estimate the exact mutual informations of a Poisson pool."""
    pool = _pool("poisson")
    joint, structure, parameter = _exact_poisson(pool, INPUTS)
    assert np.all(parameter > 1e-3) and np.all(structure > 1e-3)  # both parts are present in this pool
    estimate = joint_eig(pool, INPUTS, 11, 20000, decompose=True)
    for key, exact, error in (("eig", joint, "mc_standard_error"),
                              ("structure_eig", structure, "structure_mc_standard_error"),
                              ("parameter_eig", parameter, "parameter_mc_standard_error")):
        assert np.all(np.abs(estimate[key] - exact) <= 5 * estimate[error] + 1e-4), key
    assert np.all(estimate["structure_eig"] < estimate["eig"])
    assert np.all(estimate["parameter_eig"] > 0)


def test_single_particle_models_carry_no_parameter_term():
    """Single-particle models carry no parameter term."""
    pool = [_fit("poisson", (1,), 0.0, particles=1), _fit("poisson", (), -0.4, particles=1, seed=3)]
    parts = joint_eig(pool, INPUTS, 5, 512, decompose=True)
    np.testing.assert_allclose(parts["parameter_eig"], 0, atol=1e-12)
    np.testing.assert_allclose(parts["structure_eig"], parts["eig"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("family", ("poisson", "lognormal", "gamma"))
def test_a_one_model_pool_carries_no_structure_term(family):
    """A one-model pool carries no structure term."""
    parts = joint_eig([_fit(family, (1,), 0.0)], INPUTS, 5, 512, decompose=True)
    np.testing.assert_allclose(parts["structure_eig"], 0, atol=1e-12)
    np.testing.assert_allclose(parts["parameter_eig"], parts["eig"], rtol=0, atol=1e-12)
    assert np.all(parts["eig"] > 0)


def _parse(monkeypatch, tmp_path, *extra):
    """The runner's own parser, with `run` replaced by a recorder."""
    seen = []
    monkeypatch.setattr(bg, "run", seen.append)
    monkeypatch.setattr(sys, "argv", ["boxinggym_core", "--output", str(tmp_path / "out"),
                                      "--task", "hyperbolic", "--proposer", "mock", *extra])
    bg.main()
    return seen[0]


class _Reached(Exception):
    pass


@pytest.mark.parametrize("policy", ("structure_eig", "parameter_eig"))
def test_the_two_arms_pass_the_checks_on_an_a1_cell(monkeypatch, tmp_path, policy):
    """Both part arms pass the checks on an A=1 cell."""
    run = bg.run
    args = _parse(monkeypatch, tmp_path, "--cell", "111", "--design-policy", policy)
    assert args.design_policy == policy

    def reached(*_):
        """Replacement that raises to mark that the checks were passed."""
        raise _Reached

    monkeypatch.setattr(bg, "compat_config", reached)  # the first call after the checks
    with pytest.raises(_Reached):
        run(args)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("policy", ("structure_eig", "parameter_eig"))
@pytest.mark.parametrize("cell", ("110", "010", "100", "000"))
def test_the_two_arms_are_refused_off_the_a1_cells_before_any_output(monkeypatch, tmp_path, policy, cell):
    """Both part arms are refused off the A=1 cells before any output exists."""
    run = bg.run
    args = _parse(monkeypatch, tmp_path, "--cell", cell, "--design-policy", policy)
    with pytest.raises(ValueError, match="needs an A=1 cell"):
        run(args)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("policy", ("structure_eig", "parameter_eig"))
def test_the_two_arms_are_refused_on_the_epig_cell(monkeypatch, tmp_path, policy):
    """Both part arms are refused on the EPIG cell."""
    run = bg.run
    args = _parse(monkeypatch, tmp_path, "--cell", "102", "--design-policy", policy)
    with pytest.raises(ValueError, match="requires cell\\[2\\]=2"):
        run(args)
    assert not (tmp_path / "out").exists()


def test_an_unstated_policy_keeps_the_joint_namespace(monkeypatch, tmp_path):
    """An unstated design policy keeps the joint namespace unchanged."""
    unstated = _parse(monkeypatch, tmp_path, "--cell", "111")
    assert "design_policy" not in vars(unstated)
    stated = _parse(monkeypatch, tmp_path, "--cell", "111", "--design-policy", "structure_eig")
    assert {k: v for k, v in vars(stated).items() if k != "design_policy"} == vars(unstated)


def test_variant_tokens():
    """Variant tokens mark only the two part arms and are distinct from other policies."""
    spec = bg.TASKS["hyperbolic"]
    base = {"prior_template": "default", "priors": "off"}
    assert bg.points_variant(base, spec) == ""
    assert bg.points_variant({**base, "design_policy": "structure_eig"}, spec) == "design_structure_eig"
    assert bg.points_variant({**base, "design_policy": "parameter_eig"}, spec) == "design_parameter_eig"
    tokens = {value[2] for value in bg.EIG_PART_POLICIES.values()}
    assert len(tokens) == 2 and not tokens & {"design_point_disagreement", "design_space_filling"}
    assert {value[0] for value in bg.EIG_PART_POLICIES.values()} == {"structure_eig", "parameter_eig"}
    assert set(bg.EIG_PART_POLICIES) == set(bg.EIG_PART_DEFINITION)


def test_label_settings_are_the_joint_arms(monkeypatch, tmp_path):
    """run_label_policy is a versioned rule file with a byte snapshot (RULE_VERSION); this change leaves
    it as it is, so the label receipt's settings of the two arms equal the joint arm's. The arm is
    stated by config.json (design_policy*), every acquisition record (`target`, `policy`) and the
    points `variant` token."""
    from types import SimpleNamespace

    from . import count_proposer as cp

    def settings(*extra):
        """Build the label-receipt runtime settings for the given extra command-line arguments."""
        args = _parse(monkeypatch, tmp_path, "--cell", "111", *extra)
        _, identity = cp.select_run_prompt(args)
        return run_label_policy.bmd_runtime_settings("boxinggym", SimpleNamespace(**{**vars(args), **identity}))

    joint = settings()
    assert "design_policy" not in joint
    for policy in ("structure_eig", "parameter_eig"):
        assert settings("--design-policy", policy) == joint


def test_the_count_path_rows_carry_the_same_identity():
    """The exhaustive path's rows (count_inference.choose) already satisfy eig = structure + parameter."""
    from .count_inference import choose
    from .test_ept_epig_target_points import _hyperbolic_fit, _synthetic_hyperbolic_rows

    fit, interface = _hyperbolic_fit()
    designs = _synthetic_hyperbolic_rows(np.random.default_rng(5), 9)
    _, rows = choose([fit], designs, 3, 64, interface, interface["trials"])
    for row in rows:
        assert abs(row["eig"] - row["structure_eig"] - row["parameter_eig"]) <= 1e-12
        assert abs(row["structure_eig"]) <= 1e-12  # one model: no structure term
        assert row["parameter_eig"] > 0
