"""ForceBench adapter: a NewtonBench-style scalar force oracle over our ForceBench worlds.

INFORMATION CONDITION (read this first). Our engine and our ICL baseline observe whole noisy
trajectories of the moving body for a chosen (p1, p2) and never see a force. This oracle lets
AutoSciLab measure the force itself at a chosen distance. That is an easier information condition
than ours: the law is measured directly, point by point, instead of being inferred through an
integrator from positions.

Parameters (names X1.. by default, native names with --names shown):
  r  in [0.5, 8.0]   separation (the r range our fixed-IC trajectories are required to stay in:
                     r >= 0.5 and max r <= cap 8.0, FB1_FIXED_IC_MANIFEST_v5.json accept rule)
  p1 in [0.1, 10]    our P1_RANGE      p2 in [0.1, 10]   our P2_RANGE
  t  in [0, 5]       oscillator only: time since launch (the world's force is modulated in time;
                     without t its law cannot be measured or plugged into a trajectory)
Measurement: F(r, p1, p2[, t]) + Gaussian noise (below), the signed radial force on body 2 toward
body 1 (positive = attraction). F is the force the trajectory simulator applies:
  gravity           p1 * (A/(2 pi r) - B r)    A, B = bg_A, bg_B (FB1_FORCE_CURVE_WORLD_RULING.json),
                                               WORLD_CFG["gravity"]["me_eff"] scaled by p1: the field
                                               engine's periodic box removes the k=0 mode
  yukawa            p1 * A K1(r/2) / (4 pi)    A from FB1_FORCE_CURVE_yukawa_amplitude.json (me_eff x p1)
  coulomb_easy      p1 p2 / r^2                WORLD_CFG true_force (nbody)
  oscillator        5 cos(pi t / 2) p1 / (2 pi r)   true_force x G(t), G0 = 5, omega = pi/2, phi = 0
                                                    (DiscoverPhysics NBodyOscillatorExecutor)
  extra_dimensions  p1 (2 pi 0.5)/(4 pi) sum_{n=-20..20} r / (r^2 + (n 2 pi 0.5)^2)^1.5  (true_force)
Noise (default relative, main's decision 2026-10-06 01:5x): NewtonBench's own rule (utils/noise.py
inject_noise at noise_level 0.05): F + Normal(0, max(0.05 |F|, 1e-9)). Seeded per (world, seed).
--force-noise additive_var keeps the brief's mapping as a non-default option: sigma_F = 0.05 *
sqrt(var_F), var_F = the variance of F along the nine noise-free scoring trajectories (9 designs x
10 times), the same "5 % of the target's spread" rule as our trajectory noise sigma = 0.05 *
sqrt(var_gt). Because forces are heavy-tailed where a trajectory passes close to the fixed body,
that sigma_F is 3-8 % of the median |F| for gravity, yukawa and oscillator, 28 % for
extra_dimensions and 3.9x the median |F| for coulomb_easy (one shift trajectory passes at r = 0.10),
which buries the coulomb signal - the reason for the relative default. The mode is recorded in every
receipt (args.force_noise and world.force_noise / force_noise_rule).
D0: two designs; (p1, p2) = our D0 knobs for that seed (run_gravity._initial_designs on
default_rng(seed)), r (and t) drawn uniformly in their ranges from a seeded stream.
evaluate_law is blocked.

Score (offline): the law is plugged into our forward stepper from the fixed IC (v5 manifest) at the
headline designs (1,1),(2,1),(1,2) and the shift designs {3,4,5}x{3,5}: fb1_forward.integrate's
match_truth time stepping (Yoshida-4 for the n-body worlds, symplectic Euler for the field worlds,
dt 0.005 - the truth simulator's own), DiscoverPhysics' force convention (magnitude at
sqrt(r^2 + soft^2), acceleration F / m with m = p2 or 1 per WORLD_CFG, t passed for oscillator);
nmse = mean over designs x times x components of (forecast - truth)^2 / VAR_GT
(tools/icl_forcebench/protocol.nmse, the rule score_dp_ourq.py uses; parity checked in the pack).
The truth series are the arrays score_dp_ourq.py scores against (packs/forcebench_truth.json, built
from its world_obj). Each world's floor = the score of its own oracle law through this path (in the
pack); for the field worlds it is the effective-force fit residual plus the field engine's
discretisation.
"""
import json
import math
import os

import numpy as np

from .paths import PACK_DIR
from .protocol import blocked

R_RANGE = (0.5, 8.0)
P_RANGE = (0.1, 10.0)
T_RANGE = (0.0, 5.0)
_I2PI = 1.0 / (2.0 * math.pi)
_KK_L = 2 * math.pi * 0.5
_KK_N = np.arange(-20, 21)


def force_law(world, consts, variant="effective"):
    """Scalar F(r, p1, p2[, t]) of the world (signed, attraction positive)."""
    if world == "gravity":
        if variant == "declared":
            return lambda r, p1, p2: p1 * _I2PI / r
        A, B = consts["A"], consts["B"]
        return lambda r, p1, p2: p1 * (A * _I2PI / r - B * r)
    if world == "yukawa":
        from scipy.special import kv
        A = 1.0 if variant == "declared" else consts["A"]
        return lambda r, p1, p2: p1 * A * float(kv(1, r / 2.0)) / (4.0 * math.pi)
    if world == "coulomb_easy":
        return lambda r, p1, p2: p1 * p2 / r ** 2
    if world == "oscillator":
        return lambda r, p1, p2, t: 5.0 * math.cos(math.pi / 2.0 * t) * p1 * _I2PI / r
    if world == "extra_dimensions":
        return lambda r, p1, p2: p1 * _KK_L * float(np.sum(r / (r * r + (_KK_N * _KK_L) ** 2) ** 1.5)) / (4.0 * math.pi)
    raise ValueError(world)


def load_world(world, pack_path=None):
    pack = json.load(open(pack_path or os.path.join(PACK_DIR, "forcebench_truth.json")))
    if world not in pack["worlds"]:
        raise KeyError(world)
    return pack["worlds"][world]


TRUTH_METHOD = {"gravity": "symplectic_euler", "yukawa": "symplectic_euler", "coulomb_easy": "yoshida4",
                "oscillator": "yoshida4", "extra_dimensions": "yoshida4"}   # fb1_forward.TRUTH_NUMERICS
TRUTH_DT = 0.005
_W4 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_Y4_C = (0.5 * _W4, 0.5 * (1.0 - _W4), 0.5 * (1.0 - _W4), 0.5 * _W4)
_Y4_D = (_W4, 1.0 - 2.0 * _W4, _W4)


def integrate(law, wp, p1, p2, timed, method, dt=TRUTH_DT):
    """The law as the force in our forward stepper; returns (T, 2) positions of the moving body.

    Time stepping = fb1_forward.integrate in match_truth mode (the truth simulator's own method and dt,
    fractional sub-step times for Yoshida-4, positions recorded after round((t_q - t_prev)/dt) steps).
    Force = DiscoverPhysics' pairwise convention: magnitude F(sqrt(r^2 + soft^2)), direction toward
    the fixed body, acceleration F / m with m = p2 or 1 (WORLD_CFG inertia), soft from WORLD_CFG.
    """
    m = float(p2) if wp["inertia"] == "p2" else 1.0
    s2 = float(wp["soft"]) ** 2
    x, y = (float(v) for v in wp["pos0"])
    vx, vy = (float(v) for v in wp["vel0"])

    def acc(x, y, tau):
        rn = math.hypot(x, y)
        rs = math.sqrt(rn * rn + s2)
        f = float(law(rs, p1, p2, tau) if timed else law(rs, p1, p2))
        k = -f / m / rn
        return k * x, k * y

    t, t_prev, out = 0.0, 0.0, []
    for tq in wp["times"]:
        for _ in range(int(round((float(tq) - t_prev) / dt))):
            if method == "yoshida4":
                for i in range(3):
                    x += _Y4_C[i] * dt * vx
                    y += _Y4_C[i] * dt * vy
                    t += _Y4_C[i] * dt
                    ax, ay = acc(x, y, t)
                    vx += _Y4_D[i] * dt * ax
                    vy += _Y4_D[i] * dt * ay
                x += _Y4_C[3] * dt * vx
                y += _Y4_C[3] * dt * vy
                t += _Y4_C[3] * dt
            elif method == "symplectic_euler":
                ax, ay = acc(x, y, t)
                vx += dt * ax
                vy += dt * ay
                x += dt * vx
                y += dt * vy
                t += dt
            elif method == "leapfrog":   # run_gravity._integrate_true's velocity Verlet (reference only)
                ax, ay = acc(x, y, t)
                vx += 0.5 * dt * ax
                vy += 0.5 * dt * ay
                x += dt * vx
                y += dt * vy
                t += dt
                ax, ay = acc(x, y, t)
                vx += 0.5 * dt * ax
                vy += 0.5 * dt * ay
            else:
                raise ValueError(method)
        t_prev = float(tq)
        out.append([x, y])
    return np.asarray(out)


def nmse(predicted, truth, var_gt):
    """tools/icl_forcebench/protocol.nmse."""
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if predicted.shape != truth.shape:
        raise ValueError(f"shape {predicted.shape} vs {truth.shape}")
    return float(np.mean((predicted - truth) ** 2) / float(var_gt))


def forecast(law, wp, timed, method=None):
    method = method or wp["method"]
    return np.asarray([integrate(law, wp, p1, p2, timed, method) for p1, p2 in wp["q_head"] + wp["q_shift"]])


def scores_of_forecast(pred, wp):
    truth = np.asarray(wp["truth"], dtype=float)
    h = len(wp["q_head"])
    return {"nmse_head": nmse(pred[:h], truth[:h], wp["var_gt"]),
            "nmse_shift": nmse(pred[h:], truth[h:], wp["var_gt"]),
            "nmse_all": nmse(pred, truth, wp["var_gt"])}


_CLASS = {}


NOISE_MODES = ("relative", "additive_var")
DEFAULT_NOISE = "relative"   # main's decision 2026-10-06 01:5x (NewtonBench's 5 % relative noise)
RELATIVE_NOISE_LEVEL = 0.05
NOISE_RULES = {"relative": "F + Normal(0, max(0.05 |F|, 1e-9)) (NewtonBench utils/noise.py inject_noise, level 0.05)",
               "additive_var": "F + Normal(0, sigma_F), sigma_F = 0.05 sqrt(var_F along the scoring trajectories)"}


def make_oracle(world, wp, seed, names="masked", noise=DEFAULT_NOISE):
    """noise: relative = NewtonBench's inject_noise rule at noise_level 0.05 (sigma = max(0.05 |F|, 1e-9)),
    the default; additive_var = Normal(0, sigma_F), sigma_F = 0.05 sqrt(var_F) (the brief's mapping)."""
    if noise not in NOISE_MODES:
        raise ValueError(noise)
    if "cls" not in _CLASS:
        from autoscilab.oracle.base import BaseOracle
        _CLASS["cls"] = type("ForceBenchOracle", (_ForceOracleMixin, BaseOracle), {"ASL_BLOCKED": True})
    return _CLASS["cls"](world, wp, seed, names, noise)


def _seed(*parts):
    import hashlib
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class _ForceOracleMixin:
    def __init__(self, world, wp, seed, names="masked", noise=DEFAULT_NOISE):
        from autoscilab.oracle.base import OracleResult
        self._OracleResult = OracleResult
        self.__world, self.__wp, self.__seed = world, wp, int(seed)
        self.__timed = world == "oscillator"
        self.__law = force_law(world, wp["law_constants"])
        self.__noise = noise
        self.__sigma = float(wp["sigma_f"])
        self.__rng = np.random.default_rng(_seed("asl_forcebench_noise", world, seed))
        native = ["r", "p1", "p2"] + (["t"] if self.__timed else [])
        self.__native = native
        self.__names = [f"X{i + 1}" for i in range(len(native))] if names == "masked" else native
        self.__mode = names
        self.__bounds = [R_RANGE, P_RANGE, P_RANGE] + ([T_RANGE] if self.__timed else [])

    @property
    def domain(self):
        return "unknown_system" if self.__mode == "masked" else "two_body_force"

    @property
    def parameter_names(self):
        return list(self.__names)

    @property
    def parameter_bounds(self):
        return {n: (float(lo), float(hi)) for n, (lo, hi) in zip(self.__names, self.__bounds)}

    @property
    def function_signature(self):
        return f"def discovered_law({', '.join(self.__names)}):"

    @property
    def param_description(self):
        shown = {"r": "distance between the two bodies", "p1": "parameter of the fixed body",
                 "p2": "parameter of the moving body", "t": "time since the start of the experiment"}
        lines = []
        for name, native, (lo, hi) in zip(self.__names, self.__native, self.__bounds):
            what = "" if self.__mode == "masked" else f" ({shown[native]})"
            lines.append(f"- {name}{what}: real number in [{lo:g}, {hi:g}]")
        noun = "y" if self.__mode == "masked" else "y (radial force on the moving body toward the fixed body, positive = attraction)"
        sign = "a real number (it can be negative)" if self.__timed else "a real number"
        lines.append(f"Output {noun}: {sign}; each measurement carries additive noise. "
                     "The discovered law should return the noise-free value of y.")
        return "\n".join(lines)

    def _measure(self, x):
        f = float(self.__law(*x) if self.__timed else self.__law(x[0], x[1], x[2]))
        sigma = self.__sigma if self.__noise == "additive_var" else max(abs(f) * RELATIVE_NOISE_LEVEL, 1e-9)
        return f + float(self.__rng.normal(0.0, sigma))

    @property
    def noise_mode(self):
        return self.__noise

    def run(self, params):
        x = [min(max(float(params[n]), lo), hi) for n, (lo, hi) in zip(self.__names, self.__bounds)]
        # noise_level as the released NewtonBench oracle reports it (its relative level); sigma_F for additive_var
        level = self.__sigma if self.__noise == "additive_var" else RELATIVE_NOISE_LEVEL
        return self._OracleResult(params={n: v for n, v in zip(self.__names, x)}, measurement=self._measure(x),
                                  domain=self.domain, noise_level=level, metadata={})

    def evaluate_law(self, law_str):
        blocked("ForceBenchOracle.evaluate_law")

    def get_ground_truth_law_str(self):
        blocked("ForceBenchOracle.get_ground_truth_law_str")

    def d0_results(self):
        knobs = self.__wp["d0_knobs"][str(self.__seed)]
        rng = np.random.default_rng(_seed("asl_forcebench_d0", self.__world, self.__seed))
        out = []
        for p1, p2 in knobs:
            r = float(rng.uniform(*R_RANGE))
            x = [r, float(p1), float(p2)] + ([float(rng.uniform(*T_RANGE))] if self.__timed else [])
            out.append(self._OracleResult(params={n: v for n, v in zip(self.__names, x)}, measurement=self._measure(x),
                                          domain=self.domain, noise_level=self.__sigma, metadata={"d0": True}))
        return out


def compile_law(law_str):
    g = {"np": np, "math": math, "__builtins__": __builtins__}
    exec(law_str, g)
    fn = g.get("discovered_law")
    if fn is None:
        raise ValueError("no discovered_law in law string")
    return fn


class ForceBenchScorer:
    def __init__(self, world, wp):
        self.world, self.wp = world, wp
        self.timed = world == "oscillator"

    def score(self, law_str, observed=None):
        if not law_str:
            return {"status": "no_law"}
        try:
            fn = compile_law(law_str)
            pred = forecast(fn, self.wp, self.timed)
        except Exception as exc:
            return {"status": "law_error", "error": f"{type(exc).__name__}: {exc}"[:300]}
        if not np.isfinite(pred).all():
            return {"status": "law_error", "error": "non-finite forecast"}
        return {"status": "scored", **scores_of_forecast(pred, self.wp)}
