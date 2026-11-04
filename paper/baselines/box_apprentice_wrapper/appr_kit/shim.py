"""The in-process changes of DESIGN_APPRENTICE_OPUS55.md section 2 and the recorders around the released loop.

Changes to the released agent (each also listed in receipt["interface_changes"]):
  1. box-loop LLM      model_search.AsyncOpenAIGPT4V = claude_path.AnthropicAsOpenAI
  2. cost table        base_agent.MODEL_COST_PER_INPUT / _OUTPUT["claude-opus-5-5"] = 4e-6 / 2e-5 (calc_cost only)
  3. scientist         LMExperimenter("claude-opus-5-5", temperature, max_tokens from conf/llms/openai.yaml), its .llm =
                       claude_path.ScientistClient (driver)
  4. program in system augment_scientist_with_ppl's text is built by the RELEASED function on a stand-in object and
                       sent as the Claude system message (base system + latest program [+ critic]); the released
                       function writes scientist.messages[0], which on the Claude branch is the first user turn
  5. start data        our 2 D0 observations in the BASE-1 wording, appended to the system message (driver)
  6. failed programs   retry patch v3 (paths.load_run_experiment applies the patch in memory); wrapped here for scoring
Also needed (no effect on what the agent sees): per-cell checkpoint_dir, box-loop log file, an env.run_experiment
exception guard (an exception becomes the released invalid-design path), and the offline scorer at each checkpoint.
"""
import contextlib
import functools
import json
import logging
import os
import random
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pymc

from . import claude_path, scorer, worlds

MODEL = claude_path.DEFAULT_MODEL
COST_PER_INPUT_TOKEN = 4e-6
COST_PER_OUTPUT_TOKEN = 2e-5
Q_METHODS = ("per_point", "batched")
MAX_CONSECUTIVE_Q_ERRORS = 10

# pymc's own functions, captured before env_seed_injection wraps them (the driver imports this module first)
_PYMC_ORIGINALS = {"draw": pymc.draw, "sample_prior_predictive": pymc.sample_prior_predictive,
                   "sample_posterior_predictive": pymc.sample_posterior_predictive}
assert getattr(pymc.draw, "__name__", "") != "wrapper", "appr_kit.shim must be imported before env_seed_injection"


@contextlib.contextmanager
def rng_neutral():
    """Scoring-only isolation: the global numpy / python random states are restored afterwards, env_seed_injection's
    counter is not advanced (pymc's own draw functions are used), and the posterior-predictive progress bar is off."""
    np_state, py_state = np.random.get_state(), random.getstate()
    saved = {k: getattr(pymc, k) for k in _PYMC_ORIGINALS}
    pymc_logger = logging.getLogger("pymc")
    level = pymc_logger.level
    pymc.draw = _PYMC_ORIGINALS["draw"]
    pymc.sample_prior_predictive = _PYMC_ORIGINALS["sample_prior_predictive"]
    pymc.sample_posterior_predictive = functools.partial(_PYMC_ORIGINALS["sample_posterior_predictive"], progressbar=False)
    pymc_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(pymc, k, v)
        pymc_logger.setLevel(level)
        np.random.set_state(np_state)
        random.setstate(py_state)


def skeleton_of(program):
    """One-line summary of a PyMC program: its y_obs likelihood line (None when there is none)."""
    if not program:
        return None
    for line in str(program).splitlines():
        s = line.strip()
        if ('"y_obs"' in s or "'y_obs'" in s) and "=" in s:
            return s[:200]
    return None


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


class Recorder:
    """What happened in the released loop, budget by budget (b = agent experiments after D0)."""

    def __init__(self, cell, work, world, checkpoints, ledger, max_tries, q_method="per_point"):
        self.cell, self.work, self.world = cell, work, world
        self.checkpoints = list(checkpoints)
        self.ledger = ledger
        self.max_tries = int(max_tries)
        if q_method not in Q_METHODS:
            raise ValueError(f"q_method must be one of {Q_METHODS}, not {q_method!r}")
        self.q_method = q_method
        self.q_fallbacks = []     # batched attempts that fell back to the per-point loop (checkpoint b, error)
        self.b = 0
        self.experiments = {}
        self.checkpoint_rows = []
        self.native = []
        self.guard_events = []
        self.augments = []
        self.d0_observed = [list(map(float, r["executed"])) for r in world.d0_records]
        self.checkpoint_dir = os.path.join(work, "oed_llms_stan")
        self.ccr5b_log = os.path.join(work, "ccr5b_retries.json")

    # ---- experiments ------------------------------------------------------------------------------------------
    def start_experiment(self):
        self.b += 1
        self.experiments[self.b] = {"b": self.b, "tries": [], "success": False, "executed": None, "y": None,
                                    "completed": False, "t_start": time.time()}

    def add_try(self, requested, ok, value, executed, extra, error):
        row = self.experiments.get(self.b)
        if row is None:     # a run_experiment call outside an agent iteration (never in the released loop)
            return
        row["tries"].append({"requested": requested, "success": bool(ok),
                             "observation": _jsonable(value) if ok else None,
                             "message": None if ok else str(value), "guard_error": error})
        if ok:
            row.update(success=True, executed=executed, y=_jsonable(value), extra=extra, completed=True)
        elif len(row["tries"]) >= self.max_tries:
            row["completed"] = True

    def observed_up_to(self, b):
        out = list(self.d0_observed)
        for k in sorted(self.experiments):
            if k <= b and self.experiments[k]["success"]:
                out.append(list(map(float, self.experiments[k]["executed"])))
        return out

    def n_obs_at(self, b):
        return len(self.observed_up_to(b))

    # ---- checkpoints --------------------------------------------------------------------------------------------
    def read_ccr5b(self):
        try:
            with open(self.ccr5b_log) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return []

    def q_predictions(self, run_experiment_module, env, program_dict, prior_mode):
        """Posterior- (or prior-) predictive means at our 256 Q points.

        per_point (the kit's first edition): the released get_ppl_prediction once per Q point (one predictive sampling
        run per point; 75-85 s per checkpoint in the stub cells).
        batched (2026-10-08): the same two released steps - construct_features / pm.set_data on the program's data
        containers, then one prior- or posterior-predictive sampling - over all Q rows at once, and the mean over
        draws per row. Same model, same trace, same draw count per row; the Monte Carlo draws differ (one joint run
        instead of 256 separate ones), nothing else. A program whose data containers cannot take 256 rows raises
        inside the batched path; the checkpoint then falls back to per_point (recorded in q_fallbacks)."""
        if self.q_method == "batched":
            t0 = time.time()
            try:
                pred = self._q_batched(run_experiment_module, env, program_dict, prior_mode)
                return pred, {}, round(time.time() - t0, 1)
            except Exception as exc:  # noqa: BLE001 - recorded, then the per-point path answers this checkpoint
                self.q_fallbacks.append({"b": self.b, "prior_mode": bool(prior_mode), "error": f"{type(exc).__name__}: {exc}"[:300]})
        return self._q_per_point(run_experiment_module, env, program_dict, prior_mode)

    def _q_batched(self, run_experiment_module, env, program_dict, prior_mode):
        questions = list(self.world.questions)
        n = len(questions)
        with rng_neutral():
            if prior_mode:
                prior_model = run_experiment_module.get_gen_model(program_dict["str_prob_prog"])
                observed_df = run_experiment_module.construct_features(env, data=questions)
                _model, predictive = prior_model(observed_df)
                y = np.asarray(predictive["y_obs"], dtype=float)
            else:
                model, trace = program_dict["model"], program_dict["trace"]
                features = env.get_ordered_features()
                if any(len(features) != len(q) for q in questions):
                    raise ValueError("question length differs from the environment's ordered features")
                data_dict = {features[i]: np.asarray([q[i] for q in questions]) for i in range(len(features))}
                with model:
                    pymc.set_data(data_dict)
                    post = pymc.sample_posterior_predictive(trace, var_names=["y_obs"], return_inferencedata=False)
                y = np.asarray(post["y_obs"], dtype=float)
        if y.ndim < 2 or y.shape[-1] != n:
            raise ValueError(f"predictive draws have shape {y.shape}; expected (..., {n})")
        pred = y.reshape(-1, n).mean(axis=0)
        if not np.all(np.isfinite(pred)):
            raise ValueError("non-finite predictive mean in the batched path")
        return pred

    def _q_per_point(self, run_experiment_module, env, program_dict, prior_mode):
        n = len(self.world.questions)
        pred, errors, consecutive = np.full(n, np.nan), {}, 0
        t0 = time.time()
        with rng_neutral():
            for i, question in enumerate(self.world.questions):
                if consecutive >= MAX_CONSECUTIVE_Q_ERRORS:
                    errors[i] = f"not evaluated: {MAX_CONSECUTIVE_Q_ERRORS} consecutive errors before this point"
                    continue
                try:
                    pred[i] = float(run_experiment_module.get_ppl_prediction(env, program_dict, question, prior_mode))
                    consecutive = 0
                except Exception as exc:  # noqa: BLE001 - recorded; the checkpoint is then law_error where it matters
                    errors[i] = f"{type(exc).__name__}: {exc}"[:300]
                    consecutive += 1
        return pred, errors, round(time.time() - t0, 1)


def install(run_experiment_module, recorder, world):
    """Changes 1, 2, 4, 6 and the per-cell folders; returns the released callables it wrapped."""
    from boxing_gym.agents import base_agent, model_search

    RE = run_experiment_module
    rec = recorder
    # 1. box-loop LLM
    model_search.AsyncOpenAIGPT4V = claude_path.AnthropicAsOpenAI
    # 2. cost table (BaseAgent.calc_cost looks response.model up; an unknown id raises KeyError)
    base_agent.MODEL_COST_PER_INPUT[MODEL] = COST_PER_INPUT_TOKEN
    base_agent.MODEL_COST_PER_OUTPUT[MODEL] = COST_PER_OUTPUT_TOKEN
    # per-cell checkpoint_dir (box_loop_experiment.py:38 hard-codes /tmp/oed_llms/stan/)
    released_initialize = model_search.initialize_experiment

    def initialize_experiment(cfg, logger, log_dir, env, prior_mode):
        experiment = released_initialize(cfg=cfg, logger=logger, log_dir=log_dir, env=env, prior_mode=prior_mode)
        experiment.checkpoint_dir = rec.checkpoint_dir
        return experiment

    model_search.initialize_experiment = initialize_experiment
    # box-loop INFO log (proposals, failed-program tracebacks) to a file in the cell, not to the console
    lg = logging.getLogger("boxing_gym.agents")
    lg.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(rec.cell, "box_loop.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    lg.addHandler(handler)
    lg.propagate = False

    # 4. program (and critic) in the Claude system message
    released_augment = RE.augment_scientist_with_ppl

    def augment_scientist_with_ppl(scientist, proposed_programs_all, critic_info_all, critic_mode=False):
        stand_in = SimpleNamespace(system=scientist._appr_base_system,
                                   messages=[{"role": "system", "content": [{"type": "text", "text": ""}]}])
        released_augment(stand_in, proposed_programs_all, critic_info_all, critic_mode=critic_mode)
        scientist.system = stand_in.messages[0]["content"][0]["text"]
        rec.augments.append({"b": rec.b, "critic_mode": bool(critic_mode), "system_chars": len(scientist.system)})

    RE.augment_scientist_with_ppl = augment_scientist_with_ppl

    # 6. retry patch v3 (already in RE.ppl_evaluate) + scoring at the checkpoint
    released_ppl_evaluate = RE.ppl_evaluate

    def ppl_evaluate(final_results, goal, scientist, num_evals, include_prior, proposed_programs_all, critic_info_all,
                     prior_mode=False, critic_mode=False):
        b = 0 if prior_mode else rec.b
        t0, calls0, events0 = time.time(), rec.ledger.calls, len(rec.read_ccr5b())
        out = released_ppl_evaluate(final_results, goal, scientist, num_evals, include_prior, proposed_programs_all,
                                    critic_info_all, prior_mode=prior_mode, critic_mode=critic_mode)
        result, questions, gts, predictions = out
        events = rec.read_ccr5b()[events0:]
        unavailable = result is None or (isinstance(result, tuple) and len(result) == 2 and result[0] is None and result[1] is None)
        program_dict = proposed_programs_all[-1][0] if proposed_programs_all and proposed_programs_all[-1] else {}
        critic = critic_info_all[-1][0] if critic_info_all and critic_info_all[-1] else {}
        fit_wall = round(time.time() - t0, 1)
        native = {"b": b, "mode": "prior" if prior_mode else "posterior", "status": "unavailable" if unavailable else "ok",
                  "metric": None if unavailable else _jsonable(result[0]),
                  "spread": None if unavailable else _jsonable(result[1]),
                  "predictions": [str(p) for p in predictions], "gts": _jsonable(list(gts)),
                  "questions": [str(q) for q in questions]}
        rec.native.append(native)
        if unavailable:
            pred, errors, q_wall = None, {}, 0.0
            reason = "CCR-5b v3: measurement unavailable after the retries"
        else:
            pred, errors, q_wall = rec.q_predictions(RE, goal.env, program_dict, prior_mode)
            reason = None
        error_entries = [e for e in events if "error" in e]
        rec.checkpoint_rows.append({
            "b": b, "mode": native["mode"], "n_obs": len(goal.env.get_data()),
            "fit_status": "measurement_unavailable" if unavailable else "accepted",
            "unavailable_reason": reason,
            "placeholder": bool(program_dict.get("ccr5b_placeholder")),
            "program": program_dict.get("str_prob_prog"), "skeleton": None if unavailable else skeleton_of(program_dict.get("str_prob_prog")),
            "loo": _jsonable(program_dict.get("loo")) if not isinstance(program_dict.get("loo"), str) else None,
            "waic": _jsonable(program_dict.get("waic")) if not isinstance(program_dict.get("waic"), str) else None,
            "critic_hypotheses": critic.get("str_hypotheses"), "critic_synthesis": critic.get("synthesis"),
            "ccr5b_events": events, "re_proposals": max(0, len(error_entries) - (1 if unavailable else 0)),
            "llm_calls": rec.ledger.calls - calls0, "box_loop_and_native_wall_s": fit_wall, "q_wall_s": q_wall,
            "q_errors": len(errors), "q_first_error": next(iter(errors.values()), None),
            "pred_all": pred, "errors": errors, "native_metric": native["metric"]})
        if pred is not None:
            os.makedirs(os.path.join(rec.cell, "predictions"), exist_ok=True)
            np.savez_compressed(os.path.join(rec.cell, "predictions", f"b{b:02d}.npz"), qx=rec.world.qx, pred=pred,
                                error_mask=np.asarray([i in errors for i in range(len(pred))]))
        return out

    RE.ppl_evaluate = ppl_evaluate
    return {"initialize_experiment": released_initialize, "augment": released_augment, "ppl_evaluate": released_ppl_evaluate}


def instrument_scientist(scientist, recorder):
    """generate_actions is called once at the start of every agent iteration (run_experiment.py:102)."""
    released = scientist.generate_actions

    def generate_actions(experiment_results=None):
        recorder.start_experiment()
        return released(experiment_results)

    scientist.generate_actions = generate_actions


def instrument_env(env, recorder, task):
    """Record every design attempt; an exception inside the released run_experiment (e.g. hyperbolic's int(float())
    on a non-numeric item, survival's ID 100) is returned as an invalid design instead of ending the cell."""
    released = env.run_experiment

    def run_experiment(input_string):
        error = None
        try:
            value, ok = released(input_string)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"[:300]
            value, ok = f"Error: invalid input ({error}).", False
            recorder.guard_events.append({"b": recorder.b, "requested": str(input_string)[:300], "error": error,
                                          "traceback": traceback.format_exc()[-1500:]})
        executed, extra = (worlds.last_executed(task, env) if ok else (None, None))
        recorder.add_try(None if input_string is None else str(input_string), ok, value, executed, extra, error)
        return value, ok

    env.run_experiment = run_experiment


def build_curve(recorder, b_end):
    """curve rows for b = 0..b_end (see the README: a row between checkpoints carries the latest checkpoint's program)."""
    world, rows = recorder.world, []
    cps = {r["b"]: r for r in recorder.checkpoint_rows}
    for b in range(0, b_end + 1):
        ks = [k for k in cps if k <= b]
        n_obs = recorder.n_obs_at(b)
        if not ks:
            rows.append({"b": b, "n_obs": n_obs, "source": "no_law", "skeleton": None, "status": "no_law"})
            continue
        k = max(ks)
        cp = cps[k]
        source = f"checkpoint_{k}" if b == k else f"carried_from_checkpoint_{k}"
        if cp["pred_all"] is None:
            score = {"status": "law_error", "error": cp["unavailable_reason"]}
        else:
            score = scorer.score_predictions(cp["pred_all"], cp["errors"], world.qx, world.truth, world.heldout,
                                             world.binary, recorder.observed_up_to(b))
        rows.append({"b": b, "n_obs": n_obs, "source": source, "skeleton": cp["skeleton"], "checkpoint": k, **score})
    return rows


def last_complete_budget(recorder):
    """The largest b whose agent experiment finished and, when b is a checkpoint, whose checkpoint finished."""
    done = {r["b"] for r in recorder.checkpoint_rows}
    if 0 in recorder.checkpoints and 0 not in done:
        return -1
    b_end = 0
    for b in range(1, recorder.b + 1):
        row = recorder.experiments.get(b)
        if row is None or not row["completed"]:
            break
        if b in recorder.checkpoints and b not in done:
            break
        b_end = b
    return b_end
