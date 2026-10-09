"""Tests for the design-ascent unit: fb1_design_search on a synthetic posterior (prior particles), no paid calls, no world calls.

Run: cd tools/fb1_snapshot && env -u FORCE_COLOR JAX_PLATFORMS=cpu DP_SCIENCEAGENT_DIR=<dir> FB1_OUT_DIR=/tmp/x/ \
      <jax venv python> -m pytest -q test_fb1_design_search.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = str(Path(__file__).resolve().parent)
sys.path.insert(0, _HERE)
os.environ.setdefault("FB1_OUT_DIR", "/tmp/fb1_design_search_test_out/")
os.environ.setdefault("FB1_WORLD", "gravity")
# DiscoverPhysics @ 33b7fa9: DP_SCIENCEAGENT_DIR, or a checkout beside the repository
os.environ.setdefault("DP_SCIENCEAGENT_DIR",
                      str(Path(__file__).resolve().parents[3] / "DiscoverPhysics" / "ScienceAgent"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fb1_core as core  # noqa: E402
import fb1_design_search as fds  # noqa: E402
import run_gravity as rg  # noqa: E402

M_OUTER, N_INNER = 32, 24          # small for speed; the estimators take them as arguments
PLAIN = lambda c, D: (lambda u, a, b: core.make_predict(c)(u, a, b, rg.POS0, rg.VEL0, rg.TIMES))  # noqa: E731


@pytest.fixture(scope="module")
def posterior():
    """Synthetic posterior: prior particles for three candidates with fixed weights."""
    key = jax.random.PRNGKey(20261002)
    cands = [core.Candidate("inv_r2", "gauss", 0), core.Candidate("screened", "gauss", 0), core.Candidate("power", "student_t4", 0)]
    fits = []
    for c in cands:
        key, k = jax.random.split(key)
        fits.append((c, core.sample_prior_u(k, 60, c.kinds())))
    return fits, np.array([0.6, 0.3, 0.1])


def test_eig_pure_equals_joint_eig_on_the_same_key(posterior, monkeypatch):
    """Test: eig pure equals joint eig on the same key."""
    fits, w = posterior
    monkeypatch.setattr(rg, "M_OUTER", M_OUTER); monkeypatch.setattr(rg, "N_INNER", N_INNER)
    rng = np.random.default_rng(7)
    designs = np.column_stack([rng.uniform(0.1, 10, 6), rng.uniform(0.1, 10, 6)])
    key = jax.random.PRNGKey(5)
    ref, _ = rg.joint_eig(key, fits, w, designs)
    draws = fds.eig_draws(key, fits, w, M_OUTER, N_INNER)
    mine, contrib = fds.eig_pure(jnp.asarray(designs), fits, w, draws, PLAIN, rg.log_dens, rg.T_OBS)
    assert np.max(np.abs(np.asarray(mine) - np.asarray(ref))) < 1e-10
    assert contrib.shape == (6, M_OUTER)


def test_epig_pure_equals_joint_epig_on_the_same_key(posterior, monkeypatch):
    """Test: epig pure equals joint epig on the same key."""
    fits, w = posterior
    monkeypatch.setattr(rg, "M_OUTER", M_OUTER); monkeypatch.setattr(rg, "N_INNER", N_INNER)
    rng = np.random.default_rng(8)
    designs = np.column_stack([rng.uniform(0.1, 10, 4), rng.uniform(0.1, 10, 4)])
    targets, _ = rg.sample_epig_targets(31, count=6)
    key = jax.random.PRNGKey(6)
    ref, _ = rg.joint_epig(key, fits, w, designs, targets)
    draws = fds.epig_draws(key, fits, w, M_OUTER, N_INNER)
    mine, _ = fds.epig_pure(jnp.asarray(designs), targets, fits, w, draws, PLAIN, rg.log_dens, rg.T_OBS)
    assert np.max(np.abs(np.asarray(mine) - np.asarray(ref))) < 1e-10


def test_gradient_matches_finite_differences(posterior):
    """Test: gradient matches finite differences."""
    fits, w = posterior
    key = jax.random.PRNGKey(9)
    draws = fds.eig_draws(key, fits, w, M_OUTER, N_INNER)
    tr = fds.LogSigmoidTransform((0.1, 10.0), (0.1, 10.0))
    f = lambda z: jnp.sum(fds.eig_pure(tr.design(z), fits, w, draws, PLAIN, rg.log_dens, rg.T_OBS)[0])  # noqa: E731
    z = jnp.asarray([[0.3, -0.4], [1.0, 0.5]])
    g = np.asarray(jax.grad(f)(z))
    h = 1e-4
    for i in range(2):
        for j in range(2):
            fd = (float(f(z.at[i, j].add(h))) - float(f(z.at[i, j].add(-h)))) / (2 * h)
            assert abs(g[i, j] - fd) <= 1e-4 * max(1.0, abs(fd))


def _search(posterior, seed, objective="eig", steps=12):
    fits, w = posterior
    rng = np.random.default_rng(seed)
    targets = rg.sample_epig_targets(seed, count=4)[0] if objective == "epig" else None
    return fds.run_design_search(jax.random.PRNGKey(seed), rng, fits, w, objective=objective, p1_range=(0.1, 10.0), p2_range=(0.1, 10.0),
                                 m_outer=M_OUTER, n_inner=N_INNER, predict=PLAIN, log_dens=rg.log_dens, t_obs=rg.T_OBS,
                                 target_points=targets, steps=steps)


def test_ascent_raises_the_objective_from_every_start(posterior):
    """Test: ascent raises the objective from every start."""
    chosen, best, rec = _search(posterior, 11)
    for t in rec["trajectories"]:
        assert t["arrival_value"] > t["steps"][0]["objective"] - 1e-9 or t["arrival_value"] > rec["start_rows"][t["restart_index"] - 1]["objective"]
        assert t["value_climbed_seed_a"] > 0
        assert len(t["steps"]) == 12 and t["stop_reason"] == "max_steps"
    assert rec["chosen_restart_index"] == best + 1 and rec["chosen"] == list(map(float, chosen))
    assert rec["chosen_by"] == "independent_holdout_value"
    assert rec["chosen_holdout_value"] == max(r["objective"] for r in rec["holdout_rows"])
    assert all(0.1 <= v <= 10.0 for row in rec["arrival_rows"] for v in row["design"])
    assert rec["start_distribution"] == "uniform_linear_box_from_run_rng"
    assert 0 <= rec["endpoints_at_edge"] <= 5


def test_same_seed_gives_the_same_trajectories(posterior):
    """Test: same seed gives the same trajectories."""
    a = _search(posterior, 12)[2]; b = _search(posterior, 12)[2]
    assert a == b


def test_starts_come_from_the_run_rng_on_the_linear_box(posterior):
    """Test: starts come from the run rng on the linear box."""
    fits, w = posterior
    rng = np.random.default_rng(3); expected = np.column_stack([rng.uniform(0.1, 10, 5), rng.uniform(0.1, 10, 5)])
    rec = fds.run_design_search(jax.random.PRNGKey(3), np.random.default_rng(3), fits, w, objective="eig", p1_range=(0.1, 10.0), p2_range=(0.1, 10.0),
                                m_outer=M_OUTER, n_inner=N_INNER, predict=PLAIN, log_dens=rg.log_dens, t_obs=rg.T_OBS, steps=2)[2]
    assert np.allclose(np.asarray(rec["starts"]), expected)


def test_epig_ascent_path_runs(posterior):
    """Test: epig ascent path runs."""
    chosen, best, rec = _search(posterior, 13, objective="epig", steps=4)
    assert rec["objective"] == "epig" and len(rec["trajectories"]) == 5 and all(len(t["steps"]) == 4 for t in rec["trajectories"])
    assert np.all(np.isfinite([t["arrival_value"] for t in rec["trajectories"]]))


def test_constants_are_recorded():
    """Test: constants are recorded."""
    c = fds.design_search_constants()
    assert c["restarts"] == 5 and c["steps"] == 40 and c["optimizer"] == "adam" and c["learning_rate"] == 0.15 and c["transform"] == "log_sigmoid"
