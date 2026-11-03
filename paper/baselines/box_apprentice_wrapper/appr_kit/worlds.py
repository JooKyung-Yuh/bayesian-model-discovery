"""Our 15 BoxingGym worlds as the released env / goal objects, built the engine's way and checked against the pack.

Construction = tools/icl_boxinggym/icl_boxinggym.py `World.__init__` + `start()` (frozen tree main_verify_iclb_a75c32c),
i.e. the engine's TaskEnvironment (boxinggym_core.py 1900-1945, main_verify_bgpm_5d88aaf) in the engine's order:
  1. env_seed_injection(world_seed) is entered (and stays active for the whole cell, as in World)
  2. np.random.seed(world_seed); env = EnvClass(); goal = GoalClass(env)        (TaskSpec env_kwargs are empty)
  3. np.random.seed(evaluation_seed); 256 x goal.get_goal_eval_question(include_prior=True)   (evaluation_points)
  4. the random2 D0 designs are observed through env.run_experiment(handle), in the pack's order (World.start)
The five tasks build grid or record pools (no global RNG draw) and select_d0_designs uses its own generator, so
steps 3 -> 4 are adjacent in the global stream; the D0 handles are taken from the pack.
Survival: step 3 is replaced by loading the pack's 256 evaluation points (recorded from the engine's own
evaluation_points; re-drawing them with the goal's balanced rejection sampler costs tens of minutes per world).
Survival observations are stored patient outcomes, so nothing downstream depends on the skipped draws.

Before any LLM call `build_world` asserts against packs/boxinggym_worlds.json: the seeds (seed_for), the world
parameters (TaskEnvironment.world_parameters names) and the pack's world block, Q x and held-out outcomes (non-survival:
regenerated here and compared exactly), the noise-free truth recomputed from the world (max abs difference recorded),
and D0: the executed designs and the observations equal the pack's.
"""
import contextlib
import hashlib
import importlib
import io
import itertools
import json
import os

import numpy as np

from . import paths

# boxinggym_core.TASKS (5d88aaf, lines 1306-1468): module, env class, goal class, and the eval-tuple positions of the
# declared inputs (`eval_input_index`); conf/envs/<name>.yaml is the released config (num_evals, env_params, com_limit).
TASKS = {
    "death": ("boxing_gym.envs.death_process", "DeathProcess", "DirectDeath", (0,), "death_process_direct"),
    "dugongs": ("boxing_gym.envs.dugongs", "Dugongs", "DirectGoal", (0,), "dugongs_direct"),
    "peregrines": ("boxing_gym.envs.peregrines", "Peregrines", "DirectGoal", (0,), "peregrines_direct"),
    "hyperbolic": ("boxing_gym.envs.hyperbolic_temporal_discount", "TemporalDiscount", "DirectGoal", (0, 1, 2),
                   "hyperbolic_direct"),
    "survival": ("boxing_gym.envs.survival_analysis", "SurvivalAnalysis", "DirectGoal", (1, 0), "survival_direct"),
}
Q_COUNT = 256
PROFILE = "paper"


def seed_for(*parts):
    """core_count.seed_for (5d88aaf core_count.py:286-287), verbatim."""
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:4], "little")


@contextlib.contextmanager
def env_seed_injection(world_seed):
    """boxinggym_core.env_seed_injection (5d88aaf boxinggym_core.py:2335-2370), verbatim body.

    boxing_gym's peregrines / dugongs call pymc.draw and pymc.sample_prior_predictive without a random_seed; while
    this context is active a call with random_seed=None gets seed_k = SeedSequence([world_seed, k]) for a per-run
    counter k. A call that carries a random_seed passes through unchanged.
    """
    import pymc
    log = []
    counter = itertools.count()
    original_draw = pymc.draw
    original_sample_prior_predictive = pymc.sample_prior_predictive

    def _wrap(name, original):
        def wrapper(*args, random_seed=None, **kwargs):
            if random_seed is None:
                k = next(counter)
                random_seed = int(np.random.SeedSequence([world_seed, k]).generate_state(1)[0])
                log.append({"call": k, "function": name, "injected_seed": random_seed})
            return original(*args, random_seed=random_seed, **kwargs)

        return wrapper

    pymc.draw = _wrap("draw", original_draw)
    pymc.sample_prior_predictive = _wrap("sample_prior_predictive", original_sample_prior_predictive)
    try:
        yield log
    finally:
        pymc.draw = original_draw
        pymc.sample_prior_predictive = original_sample_prior_predictive


_WORLD_PARAMETER_NAMES = ("theta", "alpha", "beta1", "beta2", "beta3", "a", "b", "l", "lambda0", "beta", "truth",
                          "true_theta", "sigma", "m", "epsilon", "num_sources", "dim", "num_patients", "time_upper_bound")


def world_parameters(env):
    """TaskEnvironment.world_parameters (5d88aaf boxinggym_core.py:2117-2157), verbatim rule."""
    found = {}
    for name in _WORLD_PARAMETER_NAMES:
        value = getattr(env, name, None)
        if value is None or isinstance(value, str):
            continue
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        if array.size == 1:
            found[name] = float(array.reshape(-1)[0])
        elif array.ndim >= 1 and array.size <= 64:
            found[name] = array.tolist()
    return found


def truth_mean(task, env, qx):
    """The engine's _truth_* (5d88aaf boxinggym_core.py:1149-1180), used only to cross-check the pack's truth."""
    from scipy.special import expit, ndtr
    rows = np.asarray(qx, dtype=float)
    if task == "death":
        return env.N * -np.expm1(-float(env.theta) * rows[:, 0])
    if task == "peregrines":
        t = rows[:, 0]
        return np.exp(env.alpha + env.beta1 * t + env.beta2 * t ** 2 + env.beta3 * t ** 3)
    if task == "dugongs":
        mean = env.a - env.b * abs(env.l) ** rows[:, 0]
        sigma = 0.25
        return sigma * np.sqrt(2 / np.pi) * np.exp(-0.5 * (mean / sigma) ** 2) + mean * (2 * ndtr(mean / sigma) - 1)
    if task == "survival":
        return expit(rows[:, 0] * np.exp(env.beta * rows[:, 1]) * env.lambda0)
    if task == "hyperbolic":
        k, alpha = env.truth
        return env.epsilon + (1 - 2 * env.epsilon) * ndtr((rows[:, 1] / (1 + k * rows[:, 2]) - rows[:, 0]) / alpha)
    raise ValueError(task)


def load_entry(task, seed, pack_path=None):
    pack_path = pack_path or paths.PACK_PATH
    with open(pack_path, "rb") as fh:
        raw = fh.read()
    pack = json.loads(raw)
    for e in pack["entries"]:
        if e["task"] == task and int(e["seed"]) == int(seed):
            return e, paths.sha256_bytes(raw)
    raise KeyError(f"no world pack entry for {task} seed {seed}")


def question_from_row(task, row, integer_flags):
    """A declared-order Q row -> the goal's eval-question tuple (eval_input_index order, integer inputs as int)."""
    index = TASKS[task][3]
    q = [None] * len(index)
    for i, pos in enumerate(index):
        v = float(row[i])
        q[pos] = int(round(v)) if integer_flags[i] else v
    return tuple(q)


def last_executed(task, env):
    """(executed design in declared input units, extra) of the environment's latest recorded observation."""
    if task == "death":
        xi, _y = env.observation_data[-1]
        return [float(xi)], {}
    if task == "peregrines":
        ti, _y = env.observation_data[-1]
        return [float(ti)], {}
    if task == "dugongs":
        age, _y = env.observed_data[-1]
        return [float(age)], {}
    if task == "hyperbolic":
        i_r, d_r, days, _y = env.observed_data[-1]
        return [float(i_r), float(d_r), float(days)], {}
    if task == "survival":
        pid, _y = env.observed_data[-1]
        rec = env.outcomes[int(pid)]
        return [float(rec[1]), float(rec[2])], {"patient": int(pid)}
    raise ValueError(task)


class World:
    """The released env and goal for one (task, seed), plus everything the scorer needs from the pack."""

    def __init__(self, task, seed, pack_path=None):
        self.task, self.seed = task, int(seed)
        self.entry, self.pack_sha256 = load_entry(task, seed, pack_path)
        e = self.entry
        self.input_names = list(e["input_names"])
        self.integer_flags = [n in set(e["integer_inputs"]) for n in self.input_names]
        self.binary = e["outcome_type"] == "binary"
        self.qx = np.asarray(e["q"]["x"], dtype=float)
        self.truth = np.asarray(e["q"]["truth_mean"], dtype=float)
        self.heldout = np.asarray(e["q"]["heldout"], dtype=float)
        self.questions = [question_from_row(task, row, self.integer_flags) for row in self.qx]
        self.env = self.goal = None
        self.d0_records = []
        self.checks = {}
        self.seed_log = None

    def build(self, stack, stdout_path=None):
        """Enter env_seed_injection on `stack` (kept open by the caller for the whole cell), build, check, start."""
        e, task = self.entry, self.task
        module_name, env_class, goal_class, _index, _conf = TASKS[task]
        self.seed_log = stack.enter_context(env_seed_injection(int(e["world_seed"])))
        sink = open(stdout_path, "w") if stdout_path else io.StringIO()
        try:
            with contextlib.redirect_stdout(sink):
                np.random.seed(int(e["world_seed"]))
                module = importlib.import_module(module_name)
                env = getattr(module, env_class)()
                goal = getattr(module, goal_class)(env)
                if task != "survival":
                    np.random.seed(int(e["evaluation_seed"]))
                    for _ in range(Q_COUNT):
                        goal.get_goal_eval_question(include_prior=True)
                else:
                    goal.eval_points = [(int(round(m)), float(t), int(round(y)))
                                        for (t, m), y in zip(e["q"]["x"], e["q"]["heldout"])]
                d0 = []
                for handle, x_pack, y_pack in zip(e["d0"]["handles"], e["d0"]["x"], e["d0"]["y"]):
                    value, ok = env.run_experiment(handle)
                    executed, extra = last_executed(task, env)
                    d0.append({"handle": handle, "value": value, "ok": bool(ok), "executed": executed, "extra": extra,
                               "x_pack": x_pack, "y_pack": y_pack})
        finally:
            if stdout_path:
                sink.close()
        self.env, self.goal, self.d0_records = env, goal, d0
        self._check(e, env, goal, d0)
        return self

    def _check(self, e, env, goal, d0):
        task, c = self.task, {}
        c["world_seed_equal"] = seed_for("boxinggym_world", task, self.seed) == int(e["world_seed"])
        c["evaluation_seed_equal"] = seed_for("boxinggym_evaluation", task, self.seed, PROFILE) == int(e["evaluation_seed"])
        wp = world_parameters(env)
        c["world_parameters_equal"] = wp == e["world_parameters_engine"]
        w = e["world"]
        if task == "death":
            c["world_block_equal"] = env.N == w["N"] and float(env.theta) == w["theta"]
        elif task == "dugongs":
            c["world_block_equal"] = (float(env.a), float(env.b), float(env.l)) == (w["a"], w["b"], w["l"])
        elif task == "peregrines":
            c["world_block_equal"] = (float(env.alpha), float(env.beta1), float(env.beta2), float(env.beta3)) == \
                (w["alpha"], w["beta1"], w["beta2"], w["beta3"])
        elif task == "hyperbolic":
            k, alpha = env.truth
            c["world_block_equal"] = (float(k), float(alpha), float(env.epsilon)) == (w["k"], w["alpha"], w["epsilon"])
        elif task == "survival":
            recs = [[int(o), float(t), int(m)] for o, t, m in env.outcomes]
            c["world_block_equal"] = (float(env.lambda0), float(env.beta)) == (w["lambda0"], w["beta"]) and \
                recs == [[int(a), float(b), int(m)] for a, b, m in w["records"]]
        points = list(goal.eval_points[:Q_COUNT])
        index = TASKS[task][3]
        gx = np.asarray([[float(p[i]) for i in index] for p in points], dtype=float)
        gy = np.asarray([float(p[-1]) for p in points], dtype=float)
        c["q_source"] = "regenerated_by_goal" if task != "survival" else "pack_recorded"
        c["q_x_equal"] = gx.shape == self.qx.shape and bool(np.array_equal(gx, self.qx))
        c["q_heldout_equal"] = bool(np.array_equal(gy, self.heldout))
        diff = np.abs(truth_mean(task, env, self.qx) - self.truth)
        c["truth_max_abs_diff"] = float(np.max(diff))
        c["truth_close"] = bool(np.allclose(truth_mean(task, env, self.qx), self.truth, rtol=1e-12, atol=0.0))
        c["d0_ok"] = all(r["ok"] for r in d0)
        c["d0_x_equal"] = [r["executed"] for r in d0] == [[float(v) for v in r["x_pack"]] for r in d0]
        c["d0_y_equal"] = [float(r["value"]) for r in d0] == [float(r["y_pack"]) for r in d0]
        c["env_seed_injections"] = len(self.seed_log)
        self.checks = c
        failed = [k for k, v in c.items() if isinstance(v, bool) and not v]
        if failed:
            raise SystemExit(f"world {task} s{self.seed} differs from the pack: {failed} ({c})")

    def d0_block(self):
        """The BASE-1 wording (DESIGN_APPRENTICE_OPUS55.md section 2.5) with our D0's designs and observations."""
        lines = [f"Input: {r['handle']} -> Result: {r['value']}" for r in self.d0_records]
        return (f"Before you begin, {len(lines)} observations have already been made in this environment "
                f"(given to you as prior data):\n" + "\n".join(lines) + "\nUse them. You will now make your own observations.")

    def world_record(self):
        e = self.entry
        return {"task": self.task, "seed": self.seed, "world_seed": e["world_seed"], "evaluation_seed": e["evaluation_seed"],
                "d0_x": e["d0"]["x"], "d0_y": e["d0"]["y"], "d0_handles": e["d0"]["handles"],
                "input_names": self.input_names, "outcome_type": e["outcome_type"], "pack_sha256": self.pack_sha256,
                "world_parameters_engine": e["world_parameters_engine"]}


def conf_env(task, checkout=None):
    """The released conf/envs/<task>_direct.yaml (read with omegaconf, as hydra would)."""
    from omegaconf import OmegaConf
    path = os.path.join(checkout or paths.checkout_dir(), "conf", "envs", TASKS[task][4] + ".yaml")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True), path
