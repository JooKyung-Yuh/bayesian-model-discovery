"""ICL (+LLM design) over the five ForceBench worlds. Dry runs and mock runs never construct a live client.
One run = one world, three replicates (the seeds main registers). Each replicate is the runner's own
cell without its inference: the runner module (tools/fb1_snapshot/run_gravity.py) is imported with
FB1_WORLD and FB1_SEED set, so its World (the DiscoverPhysics executor with the seed's observation
noise), its two random start designs, its 64 random candidate designs per round (the one numpy
stream the runner draws both from, run_gravity.py:140 and :1573-1574) and its fixed evaluation
designs Q_HEAD + Q_SHIFT with their noise-free truths are the runner's. The model chooses one of
the 64 candidates in each of the 16 rounds and forecasts the two-component output series at the
nine evaluation designs at 2, 4, 8, 16 and 18 observations; the score is the runner's nmse formula
on the forecast (mean squared error over designs x times x components over VAR_GT). No NLL,
coverage or KL: a point forecast has none (left blank, disclosed).

Unit ICL-X (2026-10-06): `--checkpoints` states other observation counts to forecast at, e.g. every
count 2, 3, ..., 18. The default is the runner's (2, 4, 8, 16) and 18, so an unstated option changes
nothing; stated, the runner's checkpoints must be among them and the last must be D0 + T_EXTRA. The
value is in `settings()`, which the sealed manifest must match exactly.

Unit ICL-F-NEUTRAL (2026-10-11): `--interface neutral` (or ICL_FB_INTERFACE_NEUTRAL=1) sends the prompts of
prompts/neutral/, the engine's neutral wording (fb_interface_neutral_v1): a fixed simulator, the derived inputs X1 and X4,
the start state as s0a/s0b, X4 values at the measurements; no mechanics word. The numbers, the worlds, the candidates, the
scores and the plan are unchanged. Unstated (and the variable unset), every request is byte for byte the released one and
the settings are the released ones; stated, the settings carry an `interface` block, so a neutral seal opens only a
neutral run and a default seal only a default run. The receipts and the seal draft name the mode (`interface_mode`).

Unit the full-interface ForceBench edition (2026-10-12): `--interface full` (or ICL_FB_INTERFACE_FULL=1) sends the released prompts with the input-name
mask off (r, p1, p2, t instead of X1..X4) and the shared description block tools/forcebench_full/FB_FULL_DESCRIPTION.md
prepended to the system text (protocol.prompt_parts). Full together with neutral is refused before any output. The
settings carry an `interface` block with the description's sha256 (computed from the file at run time), so a full seal
opens only a full run; the receipts and the seal draft name the mode and the sha256 (`description_sha256`).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import os
import sys
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bayes_model_discovery.experimental.core_count import seed_for
from tools.icl_baseline.transport import (
    MODEL,
    PROJECTION_RULE,
    RESERVATION_RULE,
    Budget,
    BudgetStopped,
    Transport,
    input_token_reservation,
    live_client,
    load_prices,
    request_body,
    sha,
    utc,
    write_new,
)
from tools.icl_forcebench.protocol import (
    CHECKPOINTS,
    FORBIDDEN,
    INPUT_NAME_MAP,
    N_CANDIDATES,
    ROUNDS,
    T_OBS,
    WORLDS,
    Context,
    Observation,
    ENGINE_NEUTRAL_VERSION,
    FULL_DESCRIPTION,
    FULL_DROPPED_SENTENCE,
    FULL_FORBIDDEN_STEMS,
    FULL_NAME_MAP,
    FULL_VERSION,
    INTERFACES,
    NEUTRAL_ALLOWED_STEMS,
    NEUTRAL_VERSION,
    forbidden_words,
    interface_plan,
    interface_violations,
    json_bytes,
    nmse,
    prediction_result,
    render_prediction,
    render_selection,
    repair_request,
    selection_fallback,
    validate_prediction,
    validate_selection,
)

UNIT = "ICL-FORCEBENCH-CLOSED-LOOP"
#: Unit ICL-F-NEUTRAL: the variable that states the neutral interface ("1"; unset, empty or "0" = default).
INTERFACE_ENV = "ICL_FB_INTERFACE_NEUTRAL"
#: The neutral addendum to registration v2; a neutral seal binds it (dependencies("neutral")).
NEUTRAL_ADDENDUM = Path(__file__).resolve().parent / "v2" / "ICL_FORCEBENCH_NEUTRAL_ADDENDUM_20261011.md"
VOCABULARY_AUDIT = Path(__file__).resolve().parent / "audits" / "fb_vocabulary_audit.py"
#: Unit the full-interface ForceBench edition: the variable that states the full interface ("1"; unset, empty or "0" = not stated); its addendum and the
#: prompt audit a full seal binds (dependencies("full")) with the description file.
INTERFACE_FULL_ENV = "ICL_FB_INTERFACE_FULL"
FULL_ADDENDUM = Path(__file__).resolve().parent / "v2" / "ICL_FORCEBENCH_FULL_ADDENDUM_20261012.md"
FULL_PROMPT_AUDIT = Path(__file__).resolve().parent / "audits" / "fb_full_prompt_audit.py"
#: Main's cap (2026-10-02): 15 USD per world; the checks and the per-kind projection are the
#: ChemBench tool's `Budget`.
CAP_USD = 15.0
#: Unit ICL-X: `--cap-usd` states another per-world cap, sealed in the settings; a guard against a mistyped one.
MAX_CAP_USD = 100.0
SEEDS = (31, 37, 41)
ROOT = Path(__file__).resolve().parent
#: The repository root (this file is paper/baselines/tools/icl_forcebench/icl_forcebench.py); the frozen
#: ForceBench simulator lives at tools/fb1_snapshot there.
REPO_ROOT = ROOT.parents[3]
FB1_DIR = REPO_ROOT / "tools" / "fb1_snapshot"
RUNNER = FB1_DIR / "run_gravity.py"
#: The runner's observation-noise seed rule (run_gravity.py:90), repeated here only for the record.
OBSERVATION_SEED_RULE = "20260906 + (seed - 11)"


def checkpoint_plan(checkpoints=CHECKPOINTS):
    """Unit ICL-X: the observation counts to forecast at, checked: strictly increasing from the
    start (2 designs) to the last round (2 + ROUNDS), so no paid selection goes unscored."""
    checkpoints = tuple(int(x) for x in checkpoints)
    if (not checkpoints or checkpoints[0] != 2 or checkpoints[-1] != 2 + ROUNDS
            or any(b <= a for a, b in pairwise(checkpoints))):
        raise ValueError(f"checkpoints must increase strictly from 2 to {2 + ROUNDS}")
    return checkpoints


def interface_mode(stated=None, environ=None):
    """Unit ICL-F-NEUTRAL: the interface from `--interface` and ICL_FB_INTERFACE_NEUTRAL; a contradiction is refused.
    Unit the full-interface ForceBench edition: also ICL_FB_INTERFACE_FULL; full and neutral together (flag and variables in any combination) are refused."""
    environ = os.environ if environ is None else environ
    stated_by_env = []
    for name, mode in ((INTERFACE_ENV, "neutral"), (INTERFACE_FULL_ENV, "full")):
        value = environ.get(name, "")
        if value not in ("", "0", "1"):
            raise ValueError(f"{name} must be 1, 0 or unset")
        if value == "1":
            stated_by_env.append((name, mode))
    if len(stated_by_env) > 1:
        raise ValueError(f"{INTERFACE_ENV}=1 and {INTERFACE_FULL_ENV}=1 together: full and neutral are refused together")
    from_env = stated_by_env[0][1] if stated_by_env else None
    if stated is not None and from_env is not None and interface_plan(stated) != from_env:
        raise ValueError(f"--interface {stated} contradicts {stated_by_env[0][0]}=1")
    return interface_plan(stated or from_env or "default")


def description_sha256():
    """Unit the full-interface ForceBench edition: the sha256 of the description file's bytes, computed now."""
    return sha(FULL_DESCRIPTION)


def interface_settings(interface):
    """The settings block of a neutral run (none for the default: its settings stay the released ones)."""
    if interface_plan(interface) == "default":
        return {}
    if interface == "full":
        return {"interface": {"mode": "full", "version": FULL_VERSION, "prompts": "prompts (the released ones), input-name mask off",
                              "input_names": dict(FULL_NAME_MAP), "dropped_masked_sentence": FULL_DROPPED_SENTENCE.strip(),
                              "description": "tools/forcebench_full/FB_FULL_DESCRIPTION.md",
                              "description_sha256": description_sha256(),
                              "description_prepend": "system text = description file text + one newline + default system text",
                              "forbidden_stems": list(FULL_FORBIDDEN_STEMS), "masked_names_forbidden": sorted(FULL_NAME_MAP),
                              "prompt_audit": "tools/e2/fb_full_prompt_audit.py"}}
    return {"interface": {"mode": "neutral", "version": NEUTRAL_VERSION, "engine_wording": ENGINE_NEUTRAL_VERSION,
                          "prompts": "prompts/neutral", "start_state_names": {"pos0": "s0a", "vel0": "s0b"},
                          "measurement_schedule_name": "X4 values at the measurements",
                          "allowed_stems": list(NEUTRAL_ALLOWED_STEMS), "vocabulary_rule": "tools/e2/fb_vocabulary_audit.py hits_in"}}


def cap_plan(cap_usd=CAP_USD):
    """Unit ICL-X: the per-world cap, checked: above 0 and at most MAX_CAP_USD (a NaN fails both)."""
    cap = float(cap_usd)
    if not 0 < cap <= MAX_CAP_USD:
        raise ValueError(f"cap_usd must be above 0 and at most {MAX_CAP_USD}")
    return cap


# ---------------------------------------------------------------- the runner as a module
def load_runner(world, seed, checkpoints=CHECKPOINTS):
    """Import (or reload) tools/fb1_snapshot/run_gravity.py for `world` and `seed`. The module reads
    FB1_WORLD and FB1_SEED at import; DP_SCIENCEAGENT_DIR must already name the DiscoverPhysics
    ScienceAgent directory (the runner refuses otherwise). Default design policy only (FB1_D0 unset)."""
    if world not in WORLDS:
        raise ValueError(f"world must be one of {WORLDS}")
    if not os.environ.get("DP_SCIENCEAGENT_DIR"):
        raise ValueError("DP_SCIENCEAGENT_DIR must name the DiscoverPhysics/ScienceAgent directory")
    for name in ("FB1_D0", "FB1_T_EXTRA", "FB1_CHECKPOINTS", "FB1_DESIGN_VARIANT", "FB1_SCORE_SEED", "FB1_IC_MANIFEST"):
        if os.environ.get(name):
            raise ValueError(f"{name} must not be set: the ICL tool uses the runner's defaults")
    os.environ["FB1_WORLD"] = world
    os.environ["FB1_SEED"] = str(int(seed))
    if str(FB1_DIR) not in sys.path:
        sys.path.insert(0, str(FB1_DIR))
    module = sys.modules.get("run_gravity")
    if module is None:
        module = importlib.import_module("run_gravity")
    elif module.WORLD != world or module.RUN_SEED != int(seed):
        module = importlib.reload(module)
    if module.WORLD != world or module.RUN_SEED != int(seed) or module.D0 != 2 or module.D0_POLICY == "mda1":
        raise ValueError("the runner module did not take the world, the seed or the default design policy")
    # The default plan is the runner's checkpoints and its last round, exactly; a stated plan (unit
    # ICL-X) must contain the runner's checkpoints and end at the runner's last round.
    runner_checkpoints_fit = (tuple(module.CHECKPOINTS) == CHECKPOINTS[:-1] if tuple(checkpoints) == CHECKPOINTS
                              else set(module.CHECKPOINTS) <= set(checkpoints))
    if module.T_EXTRA != ROUNDS or module.N_RANDOM_DESIGNS != N_CANDIDATES or module.T_OBS != T_OBS \
            or not runner_checkpoints_fit or module.D0 + module.T_EXTRA != checkpoints[-1]:
        raise ValueError("the runner's rounds, candidate count, measurement times or checkpoints differ from this tool's")
    return module


def runner_module(world):
    module = sys.modules.get("run_gravity")
    if module is not None and module.WORLD == world:
        return module
    return load_runner(world, SEEDS[0])


def runner_constants(world):
    rg = runner_module(world)
    return {"d0_policy": rg.D0_POLICY, "d0_size": rg.D0, "t_extra": rg.T_EXTRA, "n_random_designs": rg.N_RANDOM_DESIGNS,
            "p1_range": list(rg.P1_RANGE), "p2_range": list(rg.P2_RANGE),
            "q_head": [list(q) for q in rg.Q_HEAD], "q_shift": [list(q) for q in rg.Q_SHIFT],
            "measurement_times": [float(t) for t in rg.TIMES], "pos0": [float(v) for v in np.asarray(rg.POS0)],
            "vel0": [float(v) for v in np.asarray(rg.VEL0)], "engine": rg.IC["engine"],
            "var_gt": float(rg.VAR_GT), "sigma_world": float(rg.SIGMA), "score_seed": int(rg.SCORE_SEED),
            "runner_checkpoints": list(rg.CHECKPOINTS), "ic_manifest": str(rg.IC_MANIFEST),
            "runner_input_provenance": rg._input_provenance()}


def settings(world, max_tokens=32000, *, checkpoints=CHECKPOINTS, cap_usd=CAP_USD, interface="default"):
    return {"unit_id": UNIT, "world": world, "model": MODEL, "max_tokens": max_tokens,
            "stream": max_tokens == 32000, "timeout_seconds": 300, "max_retries": 0,
            "output_config": {"effort": "medium"}, "thinking": {"type": "adaptive", "display": "summarized"},
            "cap_usd": cap_usd, "reservation_rule": RESERVATION_RULE, "projection_rule": PROJECTION_RULE,
            "seeds": list(SEEDS), "checkpoints": list(checkpoints), "rounds": ROUNDS, "candidates_per_round": N_CANDIDATES,
            "observation_seed_rule": OBSERVATION_SEED_RULE, "input_name_map": dict(INPUT_NAME_MAP),
            "primary_metric": "nmse_head", "secondary_metrics": ["nmse_shift", "nmse_all"],
            "selection_fallback_stream": "icl_forcebench_selection_fallback",
            "mock_selection_stream": "icl_forcebench_synthetic_selection",
            "max_attempts_per_request": 2, "engine_sha256": sha(RUNNER), "runner": runner_constants(world),
            **interface_settings(interface)}


#: Noise-free evaluation truths per world (seed-independent), computed once per process.
_TRUTHS = {}


class World:
    """One world at one seed through the runner's own objects: `rg.World()` (the executor with the
    seed's observation noise), `rg._initial_designs(rng)` on `np.random.default_rng(seed)`, the
    64 candidates per round from the same stream, and the noise-free truths at Q_HEAD + Q_SHIFT
    from a scoring executor as the runner's `_q_data` builds them."""

    def __init__(self, name, seed, checkpoints=CHECKPOINTS):
        self.name, self.seed = name, int(seed)
        self.rg = rg = load_runner(name, seed, checkpoints)
        self.world = rg.World()
        score_exe = rg.get_world(rg.WORLD, engine=rg.IC["engine"], noise_std=rg.SIGMA, noise_seed=rg.SCORE_SEED)["executor"]
        self.queries = [(f"head_{i + 1}", (float(p1), float(p2))) for i, (p1, p2) in enumerate(rg.Q_HEAD)] + \
                       [(f"shift_{i + 1}", (float(p1), float(p2))) for i, (p1, p2) in enumerate(rg.Q_SHIFT)]
        requests = [{"p1": p1, "p2": p2, "pos2": [float(v) for v in np.asarray(rg.POS0)],
                     "velocity2": [float(v) for v in np.asarray(rg.VEL0)], "measurement_times": [float(t) for t in rg.TIMES]}
                    for _label, (p1, p2) in self.queries]
        if name not in _TRUTHS:
            # Noise-free and seed-independent (the runner's `_q_data` under `noise_disabled`); the
            # field engine spends ~5 s per design, so one computation per world and process.
            with score_exe.noise_disabled():
                _TRUTHS[name] = np.asarray([np.asarray(r["pos2"], float) for r in score_exe.run(requests)], dtype=float)
        self.truth = _TRUTHS[name].copy()
        self.context = Context(p_range=tuple(rg.P1_RANGE), times=tuple(float(t) for t in rg.TIMES),
                               pos0=tuple(float(v) for v in np.asarray(rg.POS0)), vel0=tuple(float(v) for v in np.asarray(rg.VEL0)))
        self.rng = np.random.default_rng(rg.RUN_SEED)
        self.d0 = np.asarray(rg._initial_designs(self.rng), dtype=float)
        self.observations = []
        self.candidate_rounds = []

    def observe(self, design, source):
        p1, p2 = (float(v) for v in design)
        y = np.asarray(self.world.observe(p1, p2), dtype=float)
        if y.shape != (T_OBS, 2) or not np.isfinite(y).all():
            raise ValueError(f"the world returned a series of shape {y.shape} with non-finite values" if y.shape == (T_OBS, 2)
                             else f"the world returned a series of shape {y.shape}")
        self.observations.append(Observation(len(self.observations), (p1, p2), tuple((float(u), float(v)) for u, v in y), source))
        return self.observations[-1]

    def start(self):
        for design in self.d0:
            self.observe(design, "d0")
        return list(self.observations)

    def candidates(self):
        """The runner's draw for the next round (run_gravity.py:1573-1574), recorded per round."""
        rg = self.rg
        cands = np.column_stack([self.rng.uniform(*rg.P1_RANGE, rg.N_RANDOM_DESIGNS),
                                 self.rng.uniform(*rg.P2_RANGE, rg.N_RANDOM_DESIGNS)])
        self.candidate_rounds.append(cands)
        return cands


class ForceTransport(Transport):
    def __init__(self, client, output, budget, *, world, max_tokens=32000, synthetic=False, interface="default"):
        super().__init__(client, output, budget, max_tokens=max_tokens, synthetic=synthetic)
        self.world_name = world
        self.interface = interface_plan(interface)
        # The thinking-summary incidence keeps the released pattern in every mode (comparable counts; in neutral mode a
        # summary may say "simulator", which the neutral prompts state; in full mode the description names mechanics).
        self.domain_words = FORBIDDEN

    def request(self, prompt, request_id, kind, *, mapping, observations, replicate, round_index, remaining):
        ids = tuple(row["id"] for row in mapping)
        validations, attempts, request = [], [], prompt
        for attempt in range(2):
            text, stop, error, stem = self.call(request, request_id, attempt, kind, remaining)
            if error:
                attempts.append("transport_failure")
                request = prompt
                continue
            result = validate_prediction(text, ids, stop) if kind == "prediction" else validate_selection(text, ids, stop)
            validations.append(result)
            status = "valid" if result.valid else "retryable_validation_failure"
            self.budget.calls[-1]["status"] = status
            attempts.append(status)
            write_new(str(stem) + ".validation.json", {"status": status, "errors": result.errors})
            if result.valid:
                break
            request = repair_request(prompt, text, "; ".join(result.errors), kind, len(ids), self.interface) if text.strip() else prompt
            if self.interface != "default":
                found = interface_violations(request["messages"][-1]["content"], self.interface)
                if found:
                    raise ValueError(f"{self.interface} repair forbidden words: {found}")
        if kind == "prediction":
            return {**prediction_result(validations, ids, observations), "attempt_statuses": attempts}
        for result in validations:
            if result.valid:
                return {"choice": result.values["choice"], "status": "valid", "attempt_statuses": attempts}
        index = selection_fallback(self.world_name, replicate, round_index, len(ids))
        return {"choice": ids[index], "status": "selection_fallback", "attempt_statuses": attempts}


class SyntheticClient:
    """Deterministic fake answers from the request alone, never published responses: every held-out
    series is the first observed series; the selection is a seeded choice."""

    def __init__(self):
        self.messages = self

    def create(self, **body):
        text = body["messages"][0]["content"]
        # Unit the full-interface ForceBench edition: the full interface's table header names p1, p2 (the mask off).
        header = "id,X2,X3,u1,v1," if "id,X2,X3,u1,v1," in text else "id,p1,p2,u1,v1,"
        first_row = text.split(header, 1)[1].split("\n", 1)[1].split("\n", 1)[0]
        numbers = [float(v) for v in first_row.split(",")[3:]]
        series = [[numbers[2 * k], numbers[2 * k + 1]] for k in range(T_OBS)]
        if "Held-out input settings (" in text:
            table = text.split("\nHeld-out input settings (", 1)[1].split("\n\n", 1)[0].splitlines()[2:]
            result = {"predictions": {row.split(",")[0]: series for row in table}}
        else:
            count = int(text.split("Candidate input settings (", 1)[1].split(")", 1)[0])
            digest = hashlib.sha256(json_bytes(body)).hexdigest()
            rng = np.random.default_rng(seed_for("icl_forcebench_synthetic_selection", digest))
            result = {"choice": f"c{int(rng.integers(count)) + 1}"}
        usage = SimpleNamespace(model_dump=lambda **_: {"input_tokens": 100, "output_tokens": 100,
                                                        "output_tokens_details": {"thinking_tokens": 10}})
        return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="Synthetic protocol check."),
                                        SimpleNamespace(type="text", text=json.dumps(result))],
                               usage=usage, stop_reason="end_turn")

    @contextmanager
    def stream(self, **body):
        response = self.create(**body)
        yield SimpleNamespace(get_final_message=lambda: response)


def dependencies(interface="default"):
    """Every file a seal must bind: this tool, the shared ChemBench ICL tool's code/prompt/price
    files (the shared verifier requires all of them), the engine modules this tool imports, the
    runner and the snapshot modules it imports, and the frozen inputs it reads. The DiscoverPhysics
    package is external: its commit is recorded in the settings (runner_input_provenance).
    Neutral (unit ICL-F-NEUTRAL): also the neutral prompts, the vocabulary rule and the neutral addendum.
    Full (unit the full-interface ForceBench edition): also the description file, the full prompt audit and the full addendum."""
    from bayes_model_discovery.experimental import belief_packet, core_count
    from tools.icl_baseline import transport
    from tools.icl_forcebench import protocol
    files = [Path(inspect.getfile(m)).resolve() for m in (belief_packet, core_count, protocol)]
    files.append(Path(__file__).resolve())
    files += sorted((ROOT / "prompts").glob("*.txt"))
    files += [FB1_DIR / name for name in ("run_gravity.py", "fb1_core.py", "fb1_forward.py", "fb1_grammar.py",
                                           "fb1_proposer.py", "world_constants.py", "run_label_policy.py")]
    files += sorted((FB1_DIR / "frozen_inputs").glob("*.json"))
    shared = Path(transport.__file__).resolve().parent
    files += [p.resolve() for p in shared.rglob("*") if p.is_file() and p.suffix in (".py", ".txt", ".json") and "tests" not in p.parts]
    if interface_plan(interface) == "neutral":
        files += sorted((ROOT / "prompts" / "neutral").glob("*.txt")) + [VOCABULARY_AUDIT, NEUTRAL_ADDENDUM]
    if interface_plan(interface) == "full":
        files += [FULL_DESCRIPTION, FULL_PROMPT_AUDIT, FULL_ADDENDUM]
    return sorted(set(files))


def full_record(interface):
    """Unit the full-interface ForceBench edition: the top-level description sha256 of a full run's receipts and seal draft (nothing in other modes)."""
    return {"description_sha256": description_sha256()} if interface_plan(interface) == "full" else {}


def seal_draft(world, max_tokens, registration, *, checkpoints=CHECKPOINTS, cap_usd=CAP_USD, interface="default"):
    d0_requests = []
    for replicate, seed in enumerate(SEEDS, 1):
        w = World(world, seed, checkpoints)
        observations = w.start()
        prompt, _ = render_prediction(observations, w.queries, w.context, interface)
        d0_requests.append({"replicate": replicate, "d0_designs": w.d0.tolist(),
                            "body_sha256": hashlib.sha256(json_bytes(request_body(prompt, max_tokens))).hexdigest()})
    files = {str(path): sha(path) for path in [*dependencies(interface), Path(registration).resolve()]}
    return {"status": "draft_not_execution_authority",
            "settings": settings(world, max_tokens, checkpoints=checkpoints, cap_usd=cap_usd, interface=interface),
            "world": world, "interface_mode": interface, **full_record(interface),
            "seeds": list(SEEDS), "files": files, "d0_prediction_requests": d0_requests,
            "synthetic_corpus_is_not_a_live_request_seal": True,
            "receipt_fields": ["unit_id", "world", "started_utc", "ended_utc", "files", "settings", "global_spent_before",
                               "track_spent_before_usd", "calls", "spent_usd", "remaining_usd", "stop_reason"]}


class Unit:
    def __init__(self, world, output, *, dry_run=False, mock=False, max_tokens=32000, client=None,
                 global_spent_before="unverified", draft=None, seeds=SEEDS, checkpoints=CHECKPOINTS, cap_usd=CAP_USD,
                 interface="default"):
        if world not in WORLDS:
            raise ValueError(f"world must be one of {WORLDS}")
        self.world_name, self.output, self.seeds = world, Path(output), tuple(seeds)
        self.checkpoints = checkpoint_plan(checkpoints)
        self.cap_usd = cap_plan(cap_usd)
        self.interface = interface_plan(interface)
        self.dry_run, self.mock, self.max_tokens = dry_run, mock, max_tokens
        self.started = utc()
        self.draft = draft or {}
        self.global_spent_before = global_spent_before
        self.budget = Budget(self.cap_usd, load_prices())
        if client is None:
            if not (mock or dry_run):
                raise ValueError("a live run needs a verified client")
            client = SyntheticClient()
        self.transport = ForceTransport(client, self.output, self.budget, world=world, max_tokens=max_tokens,
                                        synthetic=mock or dry_run, interface=self.interface)
        self.planned_by_kind = {"prediction": len(self.seeds) * len(self.checkpoints), "selection": len(self.seeds) * ROUNDS}
        self.completed_by_kind = {"prediction": 0, "selection": 0}
        self.requests, self.scores, self.replicates = [], [], []

    def remaining_by_kind(self):
        return {kind: self.planned_by_kind[kind] - self.completed_by_kind[kind] for kind in self.planned_by_kind}

    def ask(self, kind, world, replicate, round_index, prompt, mapping, metadata):
        body = request_body(prompt, self.max_tokens)
        if self.interface == "default":
            found = forbidden_words(prompt["system"] + "\n" + prompt["messages"][0]["content"])
        else:
            found = interface_violations("\n".join([prompt["system"], *(m["content"] for m in prompt["messages"] if m["role"] == "user")]),
                                         self.interface)
        if found:
            raise ValueError(f"prompt forbidden words: {found}")
        identity = f"{len(self.requests):04d}-{kind}"
        record = {"request_id": identity, "body": body, "query_mapping": mapping, "replicate": replicate,
                  "round": round_index, "n": len(world.observations), **metadata,
                  "provenance": "ACTUAL_D0" if len(world.observations) == 2 else "LIVE_DECISIONS"}
        write_new(self.output / "corpus" / f"{identity}.json", record)
        self.requests.append(record)
        result = self.transport.request(prompt, identity, kind, mapping=mapping, observations=world.observations,
                                        replicate=replicate, round_index=round_index, remaining=self.remaining_by_kind())
        self.completed_by_kind[kind] += 1
        write_new(self.output / "responses" / f"{identity}.json", {**record, "body": None, **result})
        return result

    def predict(self, world, replicate):
        n = len(world.observations)
        prompt, mapping = render_prediction(world.observations, world.queries, world.context, self.interface)
        result = self.ask("prediction", world, replicate, n - 2, prompt, mapping, {"evaluation_size": len(mapping)})
        predicted = np.asarray(result["values"], dtype=float)                      # (9, 10, 2) in query order
        head = len(world.rg.Q_HEAD)
        row = {"replicate": replicate, "seed": world.seed, "x": n, "evaluation_designs": len(mapping),
               "fallback_query_count": len(result["fallback_ids"]),
               "status": "invalid_timepoint" if len(result["fallback_ids"]) / len(mapping) > .2 else "valid",
               "nmse_head": nmse(predicted[:head], world.truth[:head], world.rg.VAR_GT),
               "nmse_shift": nmse(predicted[head:], world.truth[head:], world.rg.VAR_GT),
               "nmse_all": nmse(predicted, world.truth, world.rg.VAR_GT),
               "nll": None, "coverage90": None, "width90": None}
        self.scores.append(row)
        write_new(self.output / "predictions" / f"rep{replicate}_x{n}.json",
                  {"queries": [{"id": m["id"], "label": m["label"], "design": m["design"]} for m in mapping],
                   "predicted": predicted.tolist(), "truth": world.truth.tolist(), "var_gt": float(world.rg.VAR_GT),
                   "input_name_map": dict(INPUT_NAME_MAP)})
        return row

    def run_replicate(self, replicate, seed):
        world = World(self.world_name, seed, self.checkpoints)
        selections, fallback_count = [], 0
        try:
            world.start()
            self.predict(world, replicate)
            for round_index in range(1, ROUNDS + 1):
                candidates = world.candidates()
                prompt, mapping = render_selection(world.observations, candidates, world.context, self.interface)
                result = self.ask("selection", world, replicate, round_index, prompt, mapping, {"candidate_count": len(candidates)})
                chosen = mapping[int(result["choice"][1:]) - 1]
                fallback_count += result["status"] == "selection_fallback"
                observation = world.observe(chosen["design"], f"round {round_index}")
                selections.append({"round": round_index, "candidate_index": chosen["candidate_index"], "design": chosen["design"],
                                   "status": result["status"], "attempt_statuses": result["attempt_statuses"],
                                   "observation_id": observation.obs_id})
                if len(world.observations) in self.checkpoints:
                    self.predict(world, replicate)
        finally:
            write_new(self.output / f"closed_rep{replicate}.json", {
                "world": self.world_name, "seed": seed, "replicate": replicate,
                "provenance": "SYNTHETIC" if self.mock or self.dry_run else "LIVE",
                "seeds": {"run": seed, "observation": 20260906 + (seed - 11), "scoring": int(world.rg.SCORE_SEED)},
                "input_name_map": dict(INPUT_NAME_MAP), "d0_designs": world.d0.tolist(),
                "observations": [{"obs_id": o.obs_id, "design": list(o.design), "source": o.source, "y": [list(p) for p in o.series]}
                                 for o in world.observations],
                "candidate_rounds": [c.tolist() for c in world.candidate_rounds],
                "evaluation_designs": [{"label": label, "design": list(design)} for label, design in world.queries],
                "selections": selections, "selection_fallback_count": fallback_count,
                "status": "run_design_failure" if fallback_count >= 7 else "valid"})
        self.replicates.append({"replicate": replicate, "seed": seed, "selection_fallback_count": fallback_count})

    def finish(self, status):
        rows = sorted(self.scores, key=lambda r: (r["replicate"], r["x"]))
        best = {}
        points = []
        for row in rows:
            key = row["replicate"]
            best[key] = row["nmse_head"] if key not in best else min(best[key], row["nmse_head"])
            points.append({"replicate": key, "x": row["x"], "nmse_head": row["nmse_head"], "nmse_shift": row["nmse_shift"],
                           "nmse_all": row["nmse_all"], "nll": "", "coverage90": "", "width90": "",
                           "best_so_far_nmse_head": best[key]})
        write_new(self.output / "scores.json", {"status": status, "world": self.world_name,
                                               "synthetic": self.mock or self.dry_run, "interpret_synthetic_scores": False,
                                               "checkpoints": list(self.checkpoints), "primary_metric": "nmse_head",
                                               "secondary_metrics": ["nmse_shift", "nmse_all"], "cells": rows, "replicates": self.replicates})
        with (self.output / "points_icl.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("replicate", "x", "nmse_head", "nmse_shift", "nmse_all", "nll",
                                                        "coverage90", "width90", "best_so_far_nmse_head"), lineterminator="\n")
            writer.writeheader()
            writer.writerows(points)
        thinking = {kind: {**counts, "ratio": counts["with_domain_word"] / counts["summaries"] if counts["summaries"] else None}
                    for kind, counts in self.transport.counts.items()}
        receipt = {"unit_id": UNIT, "world": self.world_name, "interface_mode": self.interface, **full_record(self.interface),
                   "started_utc": self.started, "ended_utc": utc(),
                   "files": self.draft.get("files", {}),
                   "settings": settings(self.world_name, self.max_tokens, checkpoints=self.checkpoints, cap_usd=self.cap_usd,
                                       interface=self.interface),
                   "track_spent_before_usd": 0, "global_spent_before": self.global_spent_before,
                   "starting_balance_usd": self.cap_usd, "status": status,
                   "calls": self.budget.calls, "call_count": len(self.budget.calls),
                   "spent_usd": self.budget.spent, "remaining_usd": self.cap_usd - self.budget.spent,
                   "paid_api_calls": 0 if self.mock or self.dry_run else len(self.budget.calls),
                   "accounting_kind": "synthetic_not_paid" if self.mock or self.dry_run else "live",
                   "stop_reason": self.budget.stopped_reason,
                   "completed_initial_requests": sum(self.completed_by_kind.values()),
                   "completed_initial_requests_by_kind": dict(self.completed_by_kind),
                   "planned_initial_requests": sum(self.planned_by_kind.values()),
                   "planned_initial_requests_by_kind": dict(self.planned_by_kind),
                   "projection": self.budget.projection, "domain_word_incidence": thinking,
                   "input_tokens": sum(c["input_tokens"] or 0 for c in self.budget.calls),
                   "output_tokens": sum(c["output_tokens"] or 0 for c in self.budget.calls)}
        write_new(self.output / "receipt.json", receipt)
        planned_costs = {}
        for thinking_tokens in (700, 3000, 6000):
            total = 0
            for request in self.requests:
                output_estimate = thinking_tokens + (16 * 2 * T_OBS * len(request["query_mapping"])
                                                     if request["request_id"].endswith("prediction") else 16)
                total += self.budget.cost(len(json.dumps(request["body"])) / 4, output_estimate)
            planned_costs[str(thinking_tokens)] = total
        write_new(self.output / "cost_plan.json", {"request_count": len(self.requests), "requests_by_kind": dict(self.completed_by_kind),
                  "expected_initial_requests": sum(self.planned_by_kind.values()), "maximum_attempts": 2 * sum(self.planned_by_kind.values()),
                  "thinking_scenarios_usd": planned_costs, "planning_only_not_guard": True, "reservation_rule": RESERVATION_RULE,
                  "cap_usd": self.cap_usd, "maximum_input_tokens_reserved": max((input_token_reservation(r["body"]) for r in self.requests), default=0)})
        return receipt

    def run(self):
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "predictions").mkdir()
        if self.draft:
            write_new(self.output / "seal_draft.json", self.draft)
        write_new(self.output / "starting_receipt.json", {"unit_id": UNIT, "world": self.world_name,
                  "interface_mode": self.interface, **full_record(self.interface), "started_utc": self.started,
                  "settings": settings(self.world_name, self.max_tokens, checkpoints=self.checkpoints, cap_usd=self.cap_usd,
                                       interface=self.interface),
                  "files": self.draft.get("files", {}),
                  "global_spent_before": self.global_spent_before, "track_spent_before_usd": 0,
                  "starting_balance_usd": self.cap_usd, "mock": self.mock, "dry_run": self.dry_run})
        status = "complete"
        try:
            for replicate, seed in enumerate(self.seeds, 1):
                self.run_replicate(replicate, seed)
        except BudgetStopped:
            status = "budget_stopped"
        except Exception:
            self.finish("failed")
            raise
        if status == "complete" and self.completed_by_kind != self.planned_by_kind:
            raise ValueError(f"request count mismatch: {self.completed_by_kind} != {self.planned_by_kind}")
        return self.finish("dry_run_complete" if self.dry_run and status == "complete" else status)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--world", choices=WORLDS, required=True)
    result.add_argument("--output", type=Path, required=True)
    modes = result.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--mock", action="store_true")
    modes.add_argument("--allow-paid-api", action="store_true")
    result.add_argument("--seal-manifest", type=Path)
    result.add_argument("--registration", type=Path, required=True)
    result.add_argument("--max-tokens", type=int, choices=(16000, 32000), default=32000)
    result.add_argument("--global-spent-before", default="unverified")
    # Unit ICL-X: the observation counts to forecast at; the default is the released plan.
    result.add_argument("--checkpoints", type=integer_list, default=CHECKPOINTS, help="comma-separated, e.g. 2,3,...,18")
    # Unit ICL-X: the per-world cap; the default is the released 15 USD.
    result.add_argument("--cap-usd", type=float, default=CAP_USD, help=f"per-world cap, above 0 and at most {MAX_CAP_USD}")
    # Unit ICL-F-NEUTRAL: the model-facing interface; unstated = ICL_FB_INTERFACE_NEUTRAL (1 = neutral), else the released one.
    # Unit the full-interface ForceBench edition: or full (ICL_FB_INTERFACE_FULL=1); full and neutral together are refused.
    result.add_argument("--interface", choices=INTERFACES, default=None,
                        help=f"default (the released prompts), neutral or full; unstated, {INTERFACE_ENV}=1 states neutral and "
                             f"{INTERFACE_FULL_ENV}=1 states full (both together are refused)")
    return result


def integer_list(text):
    try:
        return tuple(int(item) for item in str(text).split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"comma-separated integers required: {text!r}") from error


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        checkpoints = checkpoint_plan(args.checkpoints)
        cap = cap_plan(args.cap_usd)
        interface = interface_mode(args.interface)
        draft = seal_draft(args.world, args.max_tokens, args.registration, checkpoints=checkpoints, cap_usd=cap,
                           interface=interface)
        client = None
        if args.allow_paid_api:
            if args.seal_manifest is None:
                raise ValueError("a live run needs --seal-manifest")
            seal = json.loads(args.seal_manifest.read_text())
            if seal.get("world") != args.world:
                raise ValueError("the sealed manifest names another world")
            if seal.get("interface_mode", "default") != interface:
                raise ValueError("the sealed manifest names another interface")
            client = live_client(True, args.seal_manifest,
                                 settings(args.world, args.max_tokens, checkpoints=checkpoints, cap_usd=cap, interface=interface),
                                 [*dependencies(interface), args.registration])
        unit = Unit(args.world, args.output, dry_run=args.dry_run, mock=args.mock, max_tokens=args.max_tokens,
                    client=client, global_spent_before=args.global_spent_before, draft=draft, checkpoints=checkpoints,
                    cap_usd=cap, interface=interface)
        receipt = unit.run()
        line = {"status": receipt["status"], "requests": receipt["completed_initial_requests"],
                "paid_api_calls": receipt["paid_api_calls"]}
        print(json.dumps({**line, "interface": interface, **full_record(interface)} if interface != "default" else line))
        return 4 if receipt["status"] == "budget_stopped" else 0
    except ValueError as error:
        cli.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
