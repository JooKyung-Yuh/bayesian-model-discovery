"""Tests for FB1_SOFTENING (force softening in the candidate forward model), no paid calls.

Run: cd tools/fb1_snapshot && env -u FORCE_COLOR JAX_PLATFORMS=cpu DP_SCIENCEAGENT_DIR=<dir> FB1_OUT_DIR=/tmp/x/ \
      <jax venv python> -m pytest -q --confcutdir=. test_fb1_softening.py
The world tests start one interpreter per world (run_gravity reads FB1_WORLD at import) and call the simulator's
noise-free executor once per world (DP_SCIENCEAGENT_DIR must point at the ScienceAgent checkout).
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = str(Path(__file__).resolve().parent)
sys.path.insert(0, _HERE)
os.environ.setdefault("FB1_OUT_DIR", "/tmp/fb1_softening_test_out/")
os.environ.setdefault("FB1_WORLD", "gravity")
# DiscoverPhysics @ 33b7fa9: DP_SCIENCEAGENT_DIR, or a checkout beside the repository
os.environ.setdefault("DP_SCIENCEAGENT_DIR",
                      str(Path(__file__).resolve().parents[3] / "DiscoverPhysics" / "ScienceAgent"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fb1_forward as fwd  # noqa: E402

NBODY = ("coulomb_easy", "oscillator", "extra_dimensions")
FIELD = ("gravity", "yukawa")


# ---------------------------------------------------------------- option parsing and the recorded config
def test_parse_softening_values_and_refusals():
    """Softening option parsing accepts valid values and refuses invalid ones."""
    for raw in (None, "", "0", "0.0"):
        assert fwd.parse_softening("gravity", raw) == 0.0
    assert fwd.parse_softening("coulomb_easy", "world") == 0.05
    assert fwd.parse_softening("gravity", "world") == 0.0 and fwd.parse_softening("yukawa", "world") == 0.0
    assert fwd.parse_softening("gravity", "0.1") == 0.1 and fwd.parse_softening("oscillator", "2") == 2.0
    for bad in ("-0.01", "nan", "inf", "abc", "0.05,0.1"):
        with pytest.raises(ValueError, match="FB1_SOFTENING must be 0, world or a finite non-negative number"):
            fwd.parse_softening("gravity", bad)
    with pytest.raises(ValueError, match="unknown world"):
        fwd.parse_softening("no_such_world", "world")
    assert fwd.TRUTH_SOFTENING == {"gravity": 0.0, "yukawa": 0.0, "coulomb_easy": 0.05, "oscillator": 0.05, "extra_dimensions": 0.05}
    assert fwd.FORCE_SOFTENING == 0.0


@pytest.mark.parametrize("world", NBODY + FIELD)
def test_numerical_config_default_key_set_is_unchanged_and_world_softening_is_recorded(world):
    """The default numerical-config key set is unchanged and the softening option is recorded when set."""
    plain = fwd.numerical_config(world, {})
    assert plain == fwd.numerical_config(world, {"FB1_SOFTENING": "0"})
    assert "softening" not in plain and "softening_option" not in plain
    by_world = fwd.numerical_config(world, {"FB1_SOFTENING": "world"})
    if world in FIELD:
        assert by_world == plain   # the field executors have no softening: "world" is the default there
    else:
        assert by_world == {**plain, "softening": 0.05, "softening_option": "world"}
    number = fwd.numerical_config(world, {"FB1_SOFTENING": "0.1"})
    assert number == {**plain, "softening": 0.1, "softening_option": "0.1"}
    # the option composes with match_truth (setting e of the exposure test)
    e = fwd.numerical_config(world, {"FB1_INTEGRATOR": "match_truth", "FB1_SOFTENING": "world"})
    assert e["mode"] == "match_truth" and e["method"] == fwd.TRUTH_NUMERICS[world]["method"]
    assert (e.get("softening", 0.0) == fwd.TRUTH_SOFTENING[world])
    # provenance rows carry the forward and truth softening only when the option is set
    assert all("forward_softening" not in row for row in fwd.numerical_provenance(world, plain)["truth_table"])
    rows = fwd.numerical_provenance(world, number)["truth_table"]
    assert all(row["forward_softening"] == 0.1 and row["truth_softening"] == fwd.TRUTH_SOFTENING[row["world"]] for row in rows)


# ---------------------------------------------------------------- the stepper against a numpy replica
def replica_leapfrog(force, p1, p2, pos0, vel0, times, dt, soft):
    """fb1_forward.integrate's velocity-Verlet path in numpy for one design, with the simulator's softening convention:
    the kernel sees sqrt(r^2 + soft^2), the direction is x/r."""
    x = np.asarray(pos0, float).copy(); v = np.asarray(vel0, float).copy()
    def acc(x, t):
        rn = math.hypot(x[0], x[1]); rs = math.sqrt(rn * rn + soft * soft)
        return -force(rs, p1, p2, t) / p2 * (x / rn)
    t = 0.0; a = acc(x, t); out = []
    for tq in times:
        for _ in range(int(round((tq - t) / dt))):
            v = v + 0.5 * dt * a; x = x + dt * v; t = t + dt; a = acc(x, t); v = v + 0.5 * dt * a
        t = float(tq); out.append(x.copy())
    return np.asarray(out)


def inv_r2_jnp(r, p1, p2, th, t):
    """Inverse-square force kernel in JAX with a theta-scaled coefficient."""
    return th[:, 0] * p1 / r ** 2


def inv_r2_np(r, p1, p2, t):
    """Inverse-square force kernel in numpy with a fixed coefficient of 1.3."""
    return 1.3 * p1 / r ** 2


@pytest.fixture(scope="module")
def setting():
    """Module-scoped fixture with one synthetic integration setting."""
    theta = jnp.array([[1.3]]); p1 = jnp.array([4.0, 2.5]); p2 = jnp.array([1.0, 0.8])
    pos0 = jnp.array([1.2, 0.0]); vel0 = jnp.array([0.0, 0.9]); times = np.array([0.5, 1.0, 2.0]); dt = 0.01
    return theta, p1, p2, pos0, vel0, times, dt


def test_default_integrate_matches_the_replica_and_an_explicit_zero_is_bit_identical(setting, monkeypatch):
    """Default integration matches the numpy replica and an explicit zero softening is bit-identical."""
    theta, p1, p2, pos0, vel0, times, dt = setting
    monkeypatch.setattr(fwd, "INTEGRATION_METHOD", "leapfrog"); monkeypatch.setattr(fwd, "ALIGN_MEASUREMENT_TIMES", False)
    assert fwd.FORCE_SOFTENING == 0.0
    default = np.asarray(fwd.integrate(inv_r2_jnp, theta, p1, p2, pos0, vel0, times, dt))      # (D, P, T, 2)
    explicit = np.asarray(fwd.integrate(inv_r2_jnp, theta, p1, p2, pos0, vel0, times, dt, softening=0.0))
    assert np.array_equal(default, explicit)
    for d in range(2):
        ref = replica_leapfrog(inv_r2_np, float(p1[d]), float(p2[d]), pos0, vel0, times, dt, soft=0.0)
        assert np.max(np.abs(default[d, 0] - ref)) < 1e-10


def test_softened_integrate_matches_the_softened_replica_via_the_module_default_and_the_argument(setting, monkeypatch):
    """Softened integration matches the softened replica via both the module default and the argument."""
    theta, p1, p2, pos0, vel0, times, dt = setting
    monkeypatch.setattr(fwd, "INTEGRATION_METHOD", "leapfrog"); monkeypatch.setattr(fwd, "ALIGN_MEASUREMENT_TIMES", False)
    plain = np.asarray(fwd.integrate(inv_r2_jnp, theta, p1, p2, pos0, vel0, times, dt))
    by_argument = np.asarray(fwd.integrate(inv_r2_jnp, theta, p1, p2, pos0, vel0, times, dt, softening=0.05))
    monkeypatch.setattr(fwd, "FORCE_SOFTENING", 0.05)   # what run_gravity sets from NUMERICS["softening"]
    by_module = np.asarray(fwd.integrate(inv_r2_jnp, theta, p1, p2, pos0, vel0, times, dt))
    assert np.array_equal(by_argument, by_module)
    for d in range(2):
        ref = replica_leapfrog(inv_r2_np, float(p1[d]), float(p2[d]), pos0, vel0, times, dt, soft=0.05)
        assert np.max(np.abs(by_argument[d, 0] - ref)) < 1e-10
    # softening changes the trajectory (r ~ 1 here: the kernel argument moves by ~ soft^2 / (2 r))
    assert np.max(np.abs(by_argument - plain)) > 1e-5
    # the explicit argument wins over the module default
    assert np.array_equal(np.asarray(fwd.integrate(inv_r2_jnp, theta, p1, p2, pos0, vel0, times, dt, softening=0.0)), plain)


# ---------------------------------------------------------------- setting e against the simulator (one interpreter per world)
#: One recorded EIG design per nbody world from the 2026-10-02 arm-111 runs (the design with the median setting-e gap
#: of its world in the table of the softening change); the exposure gap there was 1.2e-17 / 4.8e-24 / 8.1e-18 nats.
RECORDED_DESIGN = {
    "coulomb_easy": (9.593902087226676, 6.891930464950818),
    "oscillator": (7.73872269470021, 0.45810701592193015),
    "extra_dimensions": (8.610130522136258, 0.24655800353443816),
}


def declared_force(world):
    """The world's declared true law as an engine force F(r, p1, p2, theta, t) with theta = 1 (acceleration = F / p2
    in fb1_forward.integrate, so coulomb_easy — unit inertia in the simulator — carries p2^2)."""
    i2pi = 1.0 / (2.0 * math.pi)
    if world == "coulomb_easy":
        return lambda r, p1, p2, th, t: th[:, 0] * p1 * p2 ** 2 / r ** 2
    if world == "oscillator":
        from scienceagent import executor as EX
        g0, om, ph = float(EX.NBodyOscillatorExecutor.G_0), float(EX.NBodyOscillatorExecutor.OMEGA), float(EX.NBodyOscillatorExecutor.PHI)
        return lambda r, p1, p2, th, t: th[:, 0] * g0 * jnp.cos(om * t + ph) * p1 * i2pi / r
    if world == "extra_dimensions":
        L = 2 * math.pi * 0.5; n = jnp.arange(-20, 21, dtype=float)
        return lambda r, p1, p2, th, t: th[:, 0] * p1 * L / (4 * math.pi) * jnp.sum(
            r[..., None] / (r[..., None] ** 2 + (n * L) ** 2) ** 1.5, axis=-1)
    raise ValueError(world)


def world_gap_main():
    """Runs inside the per-world interpreter: setting e (FB1_INTEGRATOR=match_truth, FB1_SOFTENING=world) against the
    simulator's noise-free trajectory at the recorded design; prints one JSON line."""
    import run_gravity as rg
    world = rg.WORLD
    out = {"world": world, "method": fwd.INTEGRATION_METHOD, "dt": rg.DT, "align": fwd.ALIGN_MEASUREMENT_TIMES,
           "force_softening": fwd.FORCE_SOFTENING, "numerics_softening": rg.NUMERICS.get("softening"),
           "numerics_explicit": rg.NUMERICS_EXPLICIT, "label_softening": rg._label_runtime_settings().get("softening")}
    if world in RECORDED_DESIGN:                              # the gap against the simulator: nbody worlds only
        p1, p2 = RECORDED_DESIGN[world]
        force = declared_force(world); theta = jnp.ones((1, 1))
        def forward(**kw):
            res = fwd.integrate(force, theta, jnp.array([p1]), jnp.array([p2]), rg.POS0, rg.VEL0, rg.TIMES, rg.DT, **kw)
            return np.asarray(res)[0, 0]                      # (T, 2)
        truth = rg.World().truth(p1, p2)
        gap = lambda ours: float(np.sum((ours - truth) ** 2) / (2 * rg.SIGMA ** 2))
        out.update(gap_e=gap(forward()), gap_unsoftened=gap(forward(softening=0.0)))
    print(json.dumps(out))


def run_world(world, extra_env):
    """Run the per-world gap computation in a fresh interpreter and return its JSON output."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("FB1_")}
    env.update({"FB1_WORLD": world, "FB1_OUT_DIR": os.environ["FB1_OUT_DIR"], "JAX_PLATFORMS": "cpu",
                "DP_SCIENCEAGENT_DIR": os.environ["DP_SCIENCEAGENT_DIR"], "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "", **extra_env})
    done = subprocess.run([sys.executable, "-c", "import test_fb1_softening as t; t.world_gap_main()"],
                          cwd=_HERE, env=env, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("world", NBODY)
def test_setting_e_reproduces_the_simulator_at_a_recorded_eig_design(world):
    """Setting e reproduces the simulator at a recorded EIG design."""
    out = run_world(world, {"FB1_INTEGRATOR": "match_truth", "FB1_SOFTENING": "world"})
    assert out["method"] == "yoshida4" and out["dt"] == 0.005 and out["align"] is True
    assert out["force_softening"] == 0.05 == out["numerics_softening"] == out["label_softening"] and out["numerics_explicit"] is True
    assert out["gap_e"] < 1e-12                       # the exposure level recorded above (1e-17 .. 1e-24 at these designs)
    assert out["gap_unsoftened"] > 1e-4               # the same method and dt without softening does not reproduce it


def test_default_run_configuration_registers_no_softening():
    """The default run configuration registers no softening."""
    out = run_world("gravity", {})
    assert out["force_softening"] == 0.0 and out["numerics_softening"] is None and out["label_softening"] is None
    assert out["numerics_explicit"] is False and out["method"] == "leapfrog"
