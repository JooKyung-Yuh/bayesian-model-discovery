"""ICL-MASKED-CLOSED-LOOP in task mode for NewtonBench: `--benchmark newtonbench` (unit ICL-NG, 2026-10-09). Dry runs and mock
runs never construct a live client.

The ChemBench task-mode CLOSED protocol under registration v6 (ICL + LLM design: the model picks every design inside the
box, and forecasts the 256-point evaluation grid after every observation, two requests of 128 points) on one NewtonBench
depth task, with the engine's own NewtonBench path as oracle and evaluation:

- the adapter is bound by the engine (`chembench_core.bind_benchmark_adapter`) to its row of the depth manifest and observed
  at noise 0.01 through the engine's masking session (`InputNameMaskedSession`): the names X1..Xn and the box are the
  engine's, and the prompt states the ranges and the derived log-centred inputs Xil exactly as the engine's interface
  writes them (presentation P; the z-scored Xiz are not shown, as in the ChemBench ICL registrations);
- three replicates = seeds 31, 37 and 41; the D0 of seed s is the engine joint cell's: the first four of
  `sample_designs(4 + 30, seed_for("chem_observation_designs", s))` - the paid cells run 30 rounds, and the sampler draws
  input by input, so the D0 depends on that count (`ENGINE_CELL_ROUNDS`, whatever this run's own `--rounds`);
- the evaluation grid is the engine's q set, `sample_designs(256, seed_for("chem_evaluation_inputs", 20260907))`, and the
  score of a checkpoint is the engine's own `evaluation_grid_scores` (nMSLE against the noise-free law), run on the
  forecasts written in the engine's predictions/tNN layout; checkpoints are the observation counts 4..34 (round t = n - 4,
  the engine's t00..t30);
- the 1,000-point official layer (`sample_designs(1000, 42)`, AutoSciLab's evaluate_law points on its box) is off by
  default and can be stated for the last checkpoint only (`--official-checkpoints last`).

The D0 is pinned twice (unit ICL-NG, 2026-10-09 20:3x KST, after unit ASL-NG found that NewtonBench's per-design noise
makes D0 depend on the numpy build): the process must run numpy 1.26.4, the engine venv's (sealed in the settings), and
the 27 computed D0s must equal d0_engine_rows_22583bd.json, rows the unit proved byte-equal to the engine's own 30-round
observations.jsonl and to unit ASL-NG's pack. Either mismatch refuses the run before any output.

`--resume` continues an unfinished run in its own folder with the semantics of the ChemBench v6 resume (unit ICL-RUN,
a693e40; registration v6-r): resume_ng.py and registration v1-r (ICL_NG_REGISTRATION_V1_R_20261009.md). A run that is not
resumed goes through the shared `Transport.call` unchanged.

Everything else is the ChemBench task-mode tool's, unchanged and imported: the transport and its two-attempt rule, the
budget with its reservation, `c_hi` and per-kind ten-call projection, the official-layer reduction, the parsers, the
fallbacks and the time-point scores (tools/icl_baseline/icl_baseline.py, protocol.py, transport.py). The engine is the
ASB-P tip 22583bd7 (src/, adapters/ and manifests/ pinned file by file in engine_22583bd_files.json)."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
from contextlib import contextmanager
from functools import cache
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import adapters
from bayes_model_discovery.experimental import chembench_core as engine
from bayes_model_discovery.experimental.core_count import seed_for
from bayes_model_discovery.experimental.count_proposer import (
    _anthropic_usage,
    anthropic_thinking_record,
    mask_secrets_deep,
    masked_exception,
    sdk_call_kwargs,
)
from tools.icl_baseline import icl_baseline as base
from tools.icl_baseline.ng import resume_ng
from tools.icl_baseline.ng.protocol_ng import (
    DOMAIN_WORDS,
    Interface,
    forbidden_words,
    render,
    repair_request,
)
from tools.icl_baseline.ng.protocol_ng import ROOT as NG_ROOT
from tools.icl_baseline.ng.resume_ng import ResumeMismatch, save_mean_out, write_out, write_text_out
from tools.icl_baseline.protocol import DOMAIN_WORDS as CHEMBENCH_DOMAIN_WORDS
from tools.icl_baseline.protocol import ROOT as ICL_TOOL_ROOT
from tools.icl_baseline.protocol import (
    closed_direction,
    json_bytes,
    number,
    observations_from_rows,
    prediction_result,
    score_timepoint,
    selection_fallback,
    validate,
)
from tools.icl_baseline.transport import (
    KINDS,
    BudgetStopped,
    RequestSkipped,
    Transport,
    input_token_reservation,
    live_client,
    request_body,
    sha,
    utc,
    write_new,
)

BENCHMARK = "newtonbench"
ENGINE_ROOT = Path(engine.__file__).resolve().parents[3]
MANIFEST = ENGINE_ROOT / "manifests" / "newtonbench_depth_manifest.json"
ENGINE_PINS = NG_ROOT / "engine_22583bd_files.json"
#: Registration v1-r: the resume paragraph (v1 itself unchanged); every seal binds it and the live gate requires it.
REGISTRATION_ADDENDUM = NG_ROOT / "ICL_NG_REGISTRATION_V1_R_20261009.md"
#: The engine's D0 rows of the 9 tasks x 3 seeds (30-round draw), full precision, byte-proved against the engine's records.
D0_PINS = NG_ROOT / "d0_engine_rows_22583bd.json"
#: The engine venv's numpy (<development path>): NewtonBench's noise is seeded from the
#: full-precision design, and another numpy build moves the sampler's last bits and so the D0 observations.
REGISTERED_NUMPY = "1.26.4"
SEEDS = (31, 37, 41)
D0_COUNT = 4
#: The paid joint cell's round count (tools/newtonbench_launch/run_cell.sh ROUNDS=30): its D0 is drawn with it.
ENGINE_CELL_ROUNDS = 30
NOISE = .01
GRID = 256
OFFICIAL = engine.OFFICIAL_EVALUATION_COUNT
GRID_SEED = seed_for("chem_evaluation_inputs", 20260907)
POOL_SEED = seed_for("chem_public_pool_inputs", 20260906)
REGISTERED_ROUNDS = 30
REGISTERED_CAP_USD = 200.0
OFFICIAL_CHOICES = ("none", "last")
#: A replicate whose selections fell back this many times is a run design failure (the ChemBench tool's rule).
DESIGN_FAILURE_FALLBACKS = 7


# ---------------------------------------------------------------- engine identity
def check_engine():
    """Refuse an engine tree whose src/, adapters/ or manifests/ file differs from the pinned commit's, an adapters package
    from another tree, and AutoSciLab oracle files other than the pinned ones (the NewtonBench adapter itself refuses any
    vendored law file other than NewtonBench 912a4ba's)."""
    from autoscilab.oracle import newtonbench as asl_newtonbench
    if np.__version__ != REGISTERED_NUMPY:
        raise ValueError(f"this process runs numpy {np.__version__}; the registered D0 needs the engine venv's numpy "
                         f"{REGISTERED_NUMPY}")
    if Path(adapters.__file__).resolve().parents[1] != ENGINE_ROOT:
        raise ValueError(f"the adapters package ({adapters.__file__}) is not the engine tree's ({ENGINE_ROOT})")
    pins = json.loads(ENGINE_PINS.read_text())
    changed = [path for path, digest in pins["files"].items()
               if not (ENGINE_ROOT / path).is_file() or sha(ENGINE_ROOT / path) != digest]
    if changed:
        raise ValueError(f"engine tree {ENGINE_ROOT} differs from {pins['engine_commit']} in {len(changed)} file(s): "
                         f"{changed[:5]}")
    asl_root = Path(asl_newtonbench.__file__).resolve().parents[2]
    changed = [path for path, digest in pins["autoscilab_files"].items() if sha(asl_root / path) != digest]
    if changed:
        raise ValueError(f"AutoSciLab tree {asl_root} differs from {pins['autoscilab_commit']} in {changed}")
    return {"engine_root": str(ENGINE_ROOT), "engine_commit": pins["engine_commit"], "autoscilab_root": str(asl_root),
            **process_versions()}


def process_versions():
    return {"numpy": np.__version__, "python": platform.python_version()}


@cache
def d0_pins():
    return json.loads(D0_PINS.read_text())


# ---------------------------------------------------------------- the task through the engine's calls
def depth_tasks():
    return [row["id"] for row in json.loads(MANIFEST.read_text())]


@cache
def bind(task_id):
    """The engine's own binding of the NewtonBench adapter to the manifest row (refuses fixed3 and the ChemBench-only
    options exactly as a `--benchmark` cell does)."""
    return engine.bind_benchmark_adapter(SimpleNamespace(
        benchmark=BENCHMARK, manifest=str(MANIFEST), task=task_id, d0_policy=engine.BENCHMARK_D0_POLICY,
        d0_fixed_inputs=None, initial_pool_file=None, pool_generator=None))


def masked_session(task_id, budget):
    """The cell's session: the adapter's Session at noise 0.01 behind the engine's input-name masking."""
    bound = bind(task_id)
    row = bound.task
    native = bound.Session(row["domain"], row["difficulty"], row["law_version"], noise_level=NOISE, budget=budget)
    return engine.InputNameMaskedSession(native, bound.sample_designs)


def matrix(designs, names):
    return np.array([[row[name] for name in names] for row in designs], dtype=float)


@cache
def interface_of(task_id):
    """X1..Xn, their ranges and the Xil definitions, as `chembench_core.run` builds a benchmark cell's interface: the masked
    schema's ranges, and `benchmark_log_centred_inputs` over the 256-design public pool (seed_for("chem_public_pool_inputs",
    20260906))."""
    session = masked_session(task_id, 1)
    schema = session.schema()
    names = tuple(schema["input_names"])
    bounds = tuple(tuple(float(v) for v in schema["input_bounds"][name]) for name in names)
    pool = matrix(session.sample_designs(256, POOL_SEED), names)
    centred = engine.benchmark_log_centred_inputs(list(names), schema["log_sampled_inputs"], pool)
    return Interface(names, bounds, tuple((name, item["definition"]) for name, item in centred.items()))


def native_words(task_id):
    """The task's native input and observable names and its domain id, which the precheck refuses in any prompt."""
    provenance = bind(task_id).provenance
    return (*provenance["input_names"], provenance["observable"], provenance["domain"])


@cache
def designs_of(task_id, size):
    """The engine's evaluation points of a task: the 256-point q set or the 1,000-point official set (masked)."""
    if size not in (GRID, OFFICIAL):
        raise ValueError("unregistered evaluation layer")
    seed = GRID_SEED if size == GRID else engine.OFFICIAL_EVALUATION_SEED
    return tuple(masked_session(task_id, 1).sample_designs(size, seed))


@cache
def references_of(task_id, size):
    """Reference values at the evaluation points, from a session that has finished (the engine reads them after finish())."""
    session = masked_session(task_id, 1)
    session.finish()
    values = session.reference_values(list(designs_of(task_id, size)))
    return {key: np.asarray(values[key], dtype=float) for key in ("primary_rate", "noiseless_observation")}


class NewtonBenchTask:
    """One replicate of one task: the seed's D0 and the task's evaluation points, all by the engine's calls. The replicates
    of a task share the interface and the evaluation points and differ in their D0 (one seed each)."""

    group = "task"

    def __init__(self, task_id, replicate, seed, *, engine_rounds=ENGINE_CELL_ROUNDS):
        self.task_id, self.replicate, self.seed = task_id, int(replicate), int(seed)
        self.engine_rounds = int(engine_rounds)
        self.root = f"task:{task_id}:seed{self.seed}"
        row = bind(task_id).task
        self.config = {"task": {key: row[key] for key in ("id", "domain", "difficulty", "law_version")}, "noise": NOISE,
                       "seed": self.seed}
        self.interface = interface_of(task_id)
        session = masked_session(task_id, D0_COUNT)
        stream = session.sample_designs(D0_COUNT + self.engine_rounds, seed_for("chem_observation_designs", self.seed))
        self.d0_designs = stream[:D0_COUNT]
        self.d0_rows = [session.observe(design) for design in self.d0_designs]
        if self.engine_rounds == ENGINE_CELL_ROUNDS:
            pinned = d0_pins()["rows"].get(f"{task_id}__s{self.seed}")
            if json.loads(json.dumps(self.d0_rows)) != pinned:
                raise ValueError(f"the D0 of {task_id} seed {self.seed} differs from the engine's pinned D0 rows "
                                 f"({D0_PINS.name}; this process runs numpy {np.__version__}, the pin numpy "
                                 f"{d0_pins()['numpy']})")
        self.observations = observations_from_rows(self.d0_rows, self.interface.names)

    def session(self, budget):
        return masked_session(self.task_id, budget)

    def queries(self, size):
        return matrix(designs_of(self.task_id, size), self.interface.names)

    def references(self, size):
        return {key: value.copy() for key, value in references_of(self.task_id, size).items()}

    def ours(self, size, n):
        return {"primary_rate": None, "noiseless_observation": None,
                "unavailable_reason": "task mode reads no run of ours; the comparison is made outside this tool"}


def task_sources(task_id):
    return [NewtonBenchTask(task_id, replicate, seed) for replicate, seed in enumerate(SEEDS, 1)]


# ---------------------------------------------------------------- plan, settings, seal draft
def ng_plan(rounds=REGISTERED_ROUNDS, checkpoints=None, official_checkpoints="none", cap_usd=REGISTERED_CAP_USD):
    """The plan, checked: 1..30 rounds (at most the engine cell's); forecast checkpoints strictly increasing from the D0
    (n = 4) to the last round (n = 4 + rounds), every observation unless stated; the official layer at none or at the
    last checkpoint only; a cap in (0, 400] USD."""
    rounds = int(rounds)
    if not 1 <= rounds <= ENGINE_CELL_ROUNDS:
        raise ValueError(f"rounds must be 1..{ENGINE_CELL_ROUNDS} (the engine cell's rounds)")
    checkpoints = (tuple(range(D0_COUNT, D0_COUNT + rounds + 1)) if checkpoints is None
                   else tuple(int(x) for x in checkpoints))
    if (not checkpoints or checkpoints[0] != D0_COUNT or checkpoints[-1] != D0_COUNT + rounds
            or any(b <= a for a, b in pairwise(checkpoints))):
        raise ValueError(f"checkpoints must increase strictly from {D0_COUNT} to {D0_COUNT} + rounds")
    if official_checkpoints not in OFFICIAL_CHOICES:
        raise ValueError("official checkpoints: none, or last (the last checkpoint only)")
    cap = float(cap_usd)
    if not 0 < cap <= base.MAX_CAP_USD:
        raise ValueError(f"cap_usd must be above 0 and at most {base.MAX_CAP_USD}")
    return {"rounds": rounds, "checkpoints": checkpoints,
            "official_checkpoints": (checkpoints[-1],) if official_checkpoints == "last" else (), "cap_usd": cap}


REGISTERED_PLAN = ng_plan()


def prompt_digests():
    return {kind: sha(NG_ROOT / "prompts" / f"{kind}.txt") for kind in ("prediction", "selection")}


def ng_settings(task_id, plan, max_tokens=32000):
    """The ChemBench CLOSED transport settings with this plan stated (`icl_baseline.plan_settings`), plus what makes the
    run a NewtonBench task run. The seal must match them exactly; both receipts record them."""
    result = base.plan_settings(base.CLOSED, max_tokens, plan)
    pins = json.loads(ENGINE_PINS.read_text())
    result.update({
        "benchmark": BENCHMARK, "task": task_id, "seeds": list(SEEDS), "noise": NOISE,
        "d0": {"policy": engine.BENCHMARK_D0_POLICY, "count": D0_COUNT, "engine_cell_rounds": ENGINE_CELL_ROUNDS,
               "rule": "the first 4 of sample_designs(4 + 30, seed_for('chem_observation_designs', seed)) through the "
                       "engine's masking session: the D0 of the engine's joint cell for (task, seed)"},
        "evaluation_grid": {"points": GRID, "stream": "seed_for('chem_evaluation_inputs', 20260907)",
                            "requests_per_checkpoint": base.BATCHES[GRID], "target": "primary_rate",
                            "scorer": "chembench_core.evaluation_grid_scores"},
        "official_layer": {"points": OFFICIAL, "seed": engine.OFFICIAL_EVALUATION_SEED,
                           "requests_per_checkpoint": base.BATCHES[OFFICIAL], "allowed_checkpoints": "none or the last"},
        "presentation": {"edition": engine.BENCHMARK_PRESENTATION_EDITION, "inputs": "X1..Xn and ranges as the engine's "
                         "interface declares them", "derived_inputs": "the engine's Xil definitions", "xiz_shown": False,
                         "response_unit_shown": False},
        "prompt_assets_sha256": prompt_digests(),
        "selection_fallback_stream": "icl_selection_fallback(ICL-MASKED-CLOSED-LOOP/<task>, replicate, round), uniform in "
                                     "the task's box",
        "mock_selection_stream": "icl_ng_synthetic_selection",
        "engine_commit": pins["engine_commit"], "engine_pins_sha256": sha(ENGINE_PINS),
        "numpy_version": REGISTERED_NUMPY, "d0_pins_sha256": sha(D0_PINS),
    })
    return result


def dependencies(task_id):
    """Every file a seal binds besides the registration: the engine modules the run executes (adapter, oracle, the vendored
    law files of this task, the D0/masking/Xil/scorer module and the transport helpers), the depth manifest, and every
    code/prompt/price/pin file of tools/icl_baseline, which the shared verifier requires."""
    from autoscilab.oracle import base as asl_base
    from autoscilab.oracle import external, newtonbench

    from adapters import newtonbench_observation
    from bayes_model_discovery.experimental import (
        belief_packet,
        core_count,
        count_proposer,
        positive_inference,
    )
    modules = (newtonbench_observation, newtonbench, asl_base, external, belief_packet, engine, core_count,
               count_proposer, positive_inference)
    files = [Path(inspect.getfile(module)).resolve() for module in modules]
    files.append(MANIFEST.resolve())
    files.append(REGISTRATION_ADDENDUM.resolve())
    files += [Path(item["path"]).resolve() for item in bind(task_id).provenance["law_files"].values()]
    files += [p.resolve() for p in ICL_TOOL_ROOT.rglob("*") if p.is_file() and p.suffix in (".py", ".txt", ".json")
              and "tests" not in p.parts]
    return sorted(set(files))


def d0_request_digests(sources, max_tokens):
    digests = []
    for source in sources:
        queries = source.queries(GRID)
        for offset in range(0, GRID, GRID // base.BATCHES[GRID]):
            prompt, _ = render("prediction", source.observations, source.interface,
                               queries=queries[offset:offset + GRID // base.BATCHES[GRID]])
            digests.append({"replicate": source.replicate, "seed": source.seed, "offset": offset,
                            "body_sha256": hashlib.sha256(json_bytes(request_body(prompt, max_tokens))).hexdigest()})
    return digests


def seal_draft(task_id, max_tokens, sources, registration, plan):
    files = dependencies(task_id) + [Path(registration).resolve()]
    return {"status": "draft_not_execution_authority", "settings": ng_settings(task_id, plan, max_tokens),
            "benchmark": BENCHMARK, "task": task_id,
            "files": {str(path): sha(path) for path in sorted(set(files))},
            "history_assignment": [{"path": s.root, "group": s.group, "replicate": s.replicate, "seed": s.seed}
                                   for s in sources],
            "synthetic_corpus_is_not_a_live_request_seal": True,
            "d0_prediction_requests": d0_request_digests(sources, max_tokens),
            "masked_receipts_reviewed": False,
            "receipt_fields": ["unit_id", "benchmark", "task_id", "started_utc", "ended_utc", "files", "settings",
                               "global_spent_before", "track_spent_before_usd", "calls", "spent_usd", "remaining_usd",
                               "stop_reason"]}


# ---------------------------------------------------------------- transport and synthetic client
class NGTransport(Transport):
    """The ChemBench transport (`call` unchanged: budget, records, accounting) with the task's interface in `request`:
    selections are validated against the task's names and box, the repair text is this asset's, and a failed selection
    falls back to a uniform draw in the task's box. Thinking summaries are also counted for NewtonBench domain words."""

    def __init__(self, client, output, budget, *, interface, fallback_key, max_tokens=32000, synthetic=False):
        super().__init__(client, output, budget, max_tokens=max_tokens, synthetic=synthetic)
        self.interface, self.fallback_key = interface, fallback_key
        self.newtonbench_counts = {kind: {"summaries": 0, "with_domain_word": 0} for kind in KINDS}

    def count_newtonbench(self, kind, summaries):
        for summary in summaries or []:
            self.newtonbench_counts[kind]["summaries"] += 1
            self.newtonbench_counts[kind]["with_domain_word"] += int(DOMAIN_WORDS.search(summary) is not None)

    def call(self, prompt, request_id, attempt, kind, remaining, skip_unstarted=None):
        if resume_ng.RESUME is not None:
            return self.call_resumed(prompt, request_id, attempt, kind, remaining, skip_unstarted)
        text, stop, error, stem = super().call(prompt, request_id, attempt, kind, remaining, skip_unstarted)
        record = json.loads(Path(str(stem) + ".response.json").read_text())
        self.count_newtonbench(kind, record.get("thinking_text"))
        return text, stop, error, stem

    def call_resumed(self, prompt, request_id, attempt, kind, remaining, skip_unstarted=None):
        """`Transport.call` of 5c71800 with a693e40's resume branch (resume_ng): a recorded attempt is answered from its
        response file and nothing is sent; the first unanswered attempt is sent once every recorded one was replayed and
        the Budget holds exactly the recorded spend."""
        resume = resume_ng.RESUME
        body = request_body(prompt, self.max_tokens)
        if self.budget.stopped_reason:
            raise BudgetStopped(self.budget.stopped_reason)
        reserved = self.budget.reserve(body, remaining, kind)
        if attempt == 0 and skip_unstarted is not None and skip_unstarted():
            raise RequestSkipped(request_id)
        stem = self.output / "transport" / f"{request_id}.attempt{attempt}"
        existing = Path(str(stem) + ".request.json").exists()
        write_out(str(stem) + ".request.json", {"body": body, "request_id": request_id,
                                                "attempt": attempt, "synthetic": self.synthetic})
        recorded = resume.recorded(stem, request_id, attempt)
        if recorded is not None:
            if not existing:
                resume.fail(f"resume: {stem}.response.json has no request file")
            return self.replay(recorded, reserved, request_id, attempt, kind, stem)
        if not resume.sent:
            # The first attempt this process sends: every recorded response is replayed by now.
            if len(resume.replayed) != resume.recorded_attempts or self.budget.spent != resume.recorded_spent_usd:
                resume.fail(f"resume: {len(resume.replayed)} of {resume.recorded_attempts} recorded attempts replayed, "
                            f"spend {self.budget.spent!r} vs recorded {resume.recorded_spent_usd!r}")
            print(json.dumps({"resume_first_sent_attempt": stem.name, "request_file_reused": existing,
                              "replayed_attempts": len(resume.replayed), "spent_usd_before": self.budget.spent,
                              "last_recorded_response": resume.last_recorded_response,
                              "recorded_responses": resume.recorded_responses, "utc": utc()}), flush=True)
        resume.sent.append(stem.name)
        if existing:
            resume.reused_request_files.append({"attempt": stem.name,
                                                "reservation_upper_bound_usd": reserved["reservation_usd"]})
        response = None
        error = None
        try:
            if self.max_tokens == 32000:
                with self.client.messages.stream(**sdk_call_kwargs(self.client.messages.stream, body)) as stream:
                    response = stream.get_final_message()
            else:
                response = self.client.messages.create(**sdk_call_kwargs(self.client.messages.create, body))
        except Exception as exc:  # noqa: BLE001 - provider/SDK failures consume one attempt
            error = masked_exception(exc)
        usage, _ = _anthropic_usage(getattr(response, "usage", None))
        usage = usage or {}
        text = "".join(getattr(block, "text", "") for block in getattr(response, "content", [])
                       if getattr(block, "type", None) == "text")
        thinking = anthropic_thinking_record(response) if response is not None else {}
        for summary in thinking.get("thinking_text", []):
            self.counts[kind]["summaries"] += 1
            self.counts[kind]["with_domain_word"] += int(CHEMBENCH_DOMAIN_WORDS.search(summary) is not None)
        stop = getattr(response, "stop_reason", None)
        row = self.budget.record(usage, reserved, request_id=request_id, attempt=attempt,
                                 status="transport_failure" if error else "response_received", kind=kind)
        record = mask_secrets_deep({"request_id": request_id, "attempt": attempt, "text": text,
                                   "stop_reason": stop, "usage": usage, **thinking,
                                   "error": str(error) if error else None, "accounting": row,
                                   "synthetic": self.synthetic})
        write_out(str(stem) + ".response.json", record)
        self.count_newtonbench(kind, record.get("thinking_text"))
        return text, stop, error, stem

    def replay(self, recorded, reserved, request_id, attempt, kind, stem):
        """a693e40's `Transport.replay`: one attempt answered from its recorded response file; the recorded usage enters
        the Budget as when it arrived (the accounting row must come out equal) and its thinking summaries are counted."""
        resume = resume_ng.RESUME
        error = recorded.get("error")
        for summary in recorded.get("thinking_text", []):
            self.counts[kind]["summaries"] += 1
            self.counts[kind]["with_domain_word"] += int(CHEMBENCH_DOMAIN_WORDS.search(summary) is not None)
        self.count_newtonbench(kind, recorded.get("thinking_text"))
        row = self.budget.record(recorded.get("usage") or {}, reserved, request_id=request_id, attempt=attempt,
                                 status="transport_failure" if error else "response_received", kind=kind)
        if row != recorded.get("accounting"):
            resume.fail(f"resume: the accounting of {stem}.response.json differs from its replay")
        resume.replayed.append(stem.name)
        return recorded.get("text", ""), recorded.get("stop_reason"), error, stem

    def request(self, prompt, request_id, kind, observations, *, mapping=(), replicate=0, round_index=0, unit_id="",
                remaining=0, skip_unstarted=None):
        query_ids = tuple(row["id"] for row in mapping)
        validations, attempts, request = [], [], prompt
        for attempt in range(2):
            text, stop, error, stem = self.call(request, request_id, attempt, kind, remaining, skip_unstarted)
            if error:
                attempts.append("transport_failure")
                request = prompt
                continue
            result = validate(text, kind, query_ids=query_ids, stop_reason=stop,
                              names=self.interface.names, bounds=self.interface.bounds)
            validations.append(result)
            status = "valid" if result.valid else "retryable_validation_failure"
            self.budget.calls[-1]["status"] = status
            attempts.append(status)
            write_out(str(stem) + ".validation.json", {"status": status, "errors": result.errors})
            if result.valid:
                break
            request = (repair_request(prompt, text, "; ".join(result.errors), kind, self.interface, len(query_ids))
                       if text.strip() else prompt)
        if kind == "prediction":
            return {**prediction_result(validations, query_ids, observations), "attempt_statuses": attempts}
        for result in validations:
            if result.valid:
                return {"x": list(result.values.values()), "status": "valid", "attempt_statuses": attempts}
        return {"x": selection_fallback(self.fallback_key, replicate, round_index, bounds=self.interface.bounds),
                "status": "selection_fallback", "attempt_statuses": attempts}


class SyntheticClient:
    """Deterministic fake answers from the request alone, never published responses: 1.0 for every forecast, and a
    selection drawn uniformly inside the ranges the request itself states."""

    def __init__(self, usage=None):
        self.messages = self
        self.usage = dict(usage or {"input_tokens": 100, "output_tokens": 100,
                                    "output_tokens_details": {"thinking_tokens": 10}})

    def create(self, **body):
        text = body["messages"][0]["content"]
        if "\nNew inputs (" in text:
            table = text.split("\nNew inputs (", 1)[1].split("\n\n", 1)[0].splitlines()[2:]
            result = {"predictions": {row.split(",")[0]: 1.0 for row in table}}
        else:
            line = next(line for line in text.splitlines() if line.startswith("Input ranges: "))
            ranges = [item.split(" [") for item in line.removeprefix("Input ranges: ").removesuffix(".").split("], ")]
            digest = hashlib.sha256(json_bytes(body)).hexdigest()
            rng = np.random.default_rng(seed_for("icl_ng_synthetic_selection", digest))
            result = {"x": {}}
            for name, span in ranges:
                low, high = (float(v) for v in span.rstrip("]").split(", "))
                result["x"][name] = float(rng.uniform(low, high))
        usage = SimpleNamespace(model_dump=lambda **_: json.loads(json.dumps(self.usage)))
        return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="Synthetic protocol check."),
                                        SimpleNamespace(type="text", text=json.dumps(result))],
                               usage=usage, stop_reason="end_turn")

    @contextmanager
    def stream(self, **body):
        response = self.create(**body)
        yield SimpleNamespace(get_final_message=lambda: response)


# ---------------------------------------------------------------- the unit
def engine_layer_scores(folder, forecasts, reference, size, meta):
    """Score forecasts {n: values} with the engine's own per-round grid scorer: each forecast is written as the engine
    writes a round's prediction (predictions/tNN.npz "mean", tNN.json "observation_count", t = n - 4), then
    `chembench_core.evaluation_grid_scores(folder, reference, size)` reads them back."""
    predictions = Path(folder) / "predictions"
    predictions.mkdir(parents=True, exist_ok=resume_ng.RESUME is not None)
    for n, values in sorted(forecasts.items()):
        stem = predictions / f"t{n - D0_COUNT:02d}"
        save_mean_out(str(stem) + ".npz", values)
        write_out(str(stem) + ".json", {"observation_count": n, "round": n - D0_COUNT, **meta})
    return engine.evaluation_grid_scores(Path(folder), reference, size)


class NGUnit(base.Unit):
    """The ChemBench task-mode CLOSED unit (`icl_baseline.Unit`: plan, budget, request accounting, ten-call projection and
    official-layer reduction inherited unchanged) over a NewtonBench task: the requests render this registration's
    assets from the task's interface, D0 has four observations (round t = n - 4), and the per-checkpoint scores add the
    engine's own grid scorer."""

    def __init__(self, task_id, output, sources, *, dry_run=False, mock=False, max_tokens=32000, client=None,
                 global_spent_before="unverified", draft=None, plan=REGISTERED_PLAN, resume=False):
        if len(sources) != len(SEEDS) or {s.task_id for s in sources} != {task_id}:
            raise ValueError("one task, three replicates (seeds 31, 37, 41)")
        # `--resume`: continue the run already started in `output` (resume_ng.Resume).
        self.resume, self.resume_index = resume, None
        super().__init__(base.CLOSED, output, sources, dry_run=dry_run, mock=mock, max_tokens=max_tokens, client=client,
                         global_spent_before=global_spent_before, draft=draft, plan=plan)
        self.task_id, self.interface = task_id, sources[0].interface
        self.native_words = native_words(task_id)
        self.settings = ng_settings(task_id, plan, max_tokens)
        self.transport = NGTransport(client, self.output, self.budget, interface=self.interface,
                                     fallback_key=f"{base.CLOSED}/{task_id}", max_tokens=max_tokens, synthetic=mock)
        self.engine_scores = []

    def ask(self, kind, observations, metadata, *, queries=None, visible=False):
        prompt, mapping = render(kind, observations, self.interface, queries=queries)
        body = request_body(prompt, self.max_tokens)
        found = forbidden_words(prompt["system"] + "\n" + prompt["messages"][0]["content"], self.native_words)
        if found:
            raise ValueError(f"prompt forbidden words: {found}")
        identity = f"{len(self.requests):04d}-{kind}"
        synthetic = (self.mock or self.dry_run) and len(observations) > D0_COUNT
        corpus = "closed_synthetic" if synthetic else "closed_d0" if len(observations) == D0_COUNT else "closed_live"
        record = {"request_id": identity, "body": body, "query_mapping": mapping, **metadata,
                  "provenance": ("SYNTHETIC" if synthetic else "ACTUAL_D0" if len(observations) == D0_COUNT
                                 else "LIVE_DECISIONS")}
        write_out(self.output / "corpus" / corpus / f"{identity}.json", record)
        self.requests.append(record)
        official = metadata.get("evaluation_size") == OFFICIAL
        try:
            result = self.transport.request(prompt, identity, kind, observations, mapping=mapping,
                                            unit_id=self.unit_id, replicate=metadata["replicate"],
                                            round_index=metadata["round"], remaining=self.remaining_by_kind(),
                                            skip_unstarted=lambda: official and self.reduced_official_layer)
        except RequestSkipped:
            write_out(self.output / "responses" / f"{identity}.json", {
                **metadata, "status": "not_sent_official_layer_reduction", "attempt_statuses": [],
            })
            return None
        self.completed += 1
        self.completed_by_kind[kind] += 1
        self.official_completed += official
        write_out(self.output / "responses" / f"{identity}.json", {**metadata, **result})
        return result

    def predict(self, history, observations, n, *, visible=False):
        for size in (GRID, OFFICIAL):
            if size == OFFICIAL and (n not in self.official_times or self.reduced_official_layer):
                continue
            queries = history.queries(size)
            batch_size = size // base.BATCHES[size]
            batches = []
            for offset in range(0, size, batch_size):
                if size == OFFICIAL and self.reduced_official_layer:
                    break
                metadata = {"history": str(history.root), "group": history.group, "replicate": history.replicate,
                            "seed": history.seed, "n": n, "round": n - D0_COUNT, "evaluation_size": size,
                            "offset": offset, "visible": False}
                result = self.ask("prediction", observations, metadata, queries=queries[offset:offset + batch_size])
                if result is not None:
                    batches.append(result)
            if size == OFFICIAL and self.reduced_official_layer and len(batches) != base.BATCHES[size]:
                self.incomplete_official_timepoints.append({
                    "group": history.group, "replicate": history.replicate, "seed": history.seed, "n": n,
                    "evaluation_size": size, "completed_batches": len(batches),
                    "status": "incomplete_official_layer_reduction",
                })
            elif batches:
                self.pending_scores.append((history, list(observations), n, size, False, batches))

    def run_closed(self):
        names = self.interface.names
        for history in self.histories:
            session = history.session(D0_COUNT + self.rounds)
            if tuple(session.schema()["input_names"]) != names:
                raise ValueError("session input order differs from the interface")
            observations, selections, fallback_count = [], [], 0
            for design in history.d0_designs:
                observations.extend(observations_from_rows([session.observe(design)], names))
            if observations != history.observations:
                raise ValueError("the session's D0 differs from the task source's (the sealed D0 requests)")
            completed = False
            try:
                self.predict(history, observations, D0_COUNT)
                for round_index in range(1, self.rounds + 1):
                    selection = self.ask("selection", observations, {"replicate": history.replicate, "seed": history.seed,
                                                                     "round": round_index, "n": len(observations)})
                    fallback_count += selection["status"] == "selection_fallback"
                    values = selection["x"]
                    observations.extend(observations_from_rows([session.observe(dict(zip(names, values)))], names))
                    selections.append({**selection, "round": round_index, "executed_coordinates": values,
                                       "display_coordinates": [number(v) for v in values],
                                       "run_status": ("run_design_failure" if fallback_count >= DESIGN_FAILURE_FALLBACKS
                                                      else "valid")})
                    if len(observations) in self.checkpoints:
                        self.predict(history, observations, len(observations))
                completed = True
            finally:
                session.finish()
                if fallback_count >= DESIGN_FAILURE_FALLBACKS:
                    self.failed_design_replicates.add(history.replicate)
                previous = [o.x for o in observations[:D0_COUNT]]
                boundary = one_input = 0
                for selection in selections:
                    x = tuple(selection["x"])
                    boundary += any(v == lo or v == hi for v, (lo, hi) in zip(x, self.interface.bounds))
                    one_input += sum(a != b for a, b in zip(x, previous[-1])) == 1
                    previous.append(x)
                # A resume writes no replicate record when it found a different recorded run, nor for an unfinished
                # replicate while it is still replaying (a693e40's rule; a fresh run writes one in both cases).
                resume = resume_ng.RESUME
                if resume is None or not (resume.failed or (not completed and not resume.sent)):
                    write_out(self.output / f"closed_rep{history.replicate}.json", {
                        "provenance": "SYNTHETIC" if self.mock or self.dry_run else "LIVE",
                        "benchmark": BENCHMARK, "task_id": self.task_id, "replicate": history.replicate,
                        "seed": history.seed, "input_names": list(names), "d0_designs": list(history.d0_designs),
                        "observations": [{"obs_id": o.obs_id, "x": o.x, "y": o.y} for o in observations],
                        "selections": selections, "selection_fallback_count": fallback_count,
                        "status": "run_design_failure" if fallback_count >= DESIGN_FAILURE_FALLBACKS else "valid",
                        "completed_selection_count": len(selections), "unique_designs": len(set(previous)),
                        "repeat_count": len(previous) - len(set(previous)),
                        "boundary_selection_ratio": boundary / len(selections) if selections else None,
                        "one_input_changed_ratio": one_input / len(selections) if selections else None})

    def score_pending(self):
        # Called only after every session has finished. No truth enters prompts.
        forecasts = {}
        for history, observations, n, size, _visible, batches in self.pending_scores:
            score = score_timepoint(batches, history.references(size), observations)
            icl = {key: score["targets"][key]["operational"]["nmsle"] for key in score["targets"]}
            self.results.append({"group": history.group, "replicate": history.replicate, "seed": history.seed, "n": n,
                                 "round": n - D0_COUNT, "evaluation_size": size, "visible": False, **score, "icl": icl,
                                 "ours": history.ours(size, n), "ours_comparison": "unpaired_reference_only",
                                 "design_status": ("run_design_failure" if history.replicate in self.failed_design_replicates
                                                   else "valid"),
                                 "direction": "no directional statement"})
            forecasts.setdefault((history.replicate, size), (history, {}))[1][n] = [v for b in batches for v in b["values"]]
        for (replicate, size), (history, by_n) in sorted(forecasts.items(), key=lambda item: item[0]):
            layer = "grid" if size == GRID else "official"
            scored = engine_layer_scores(self.output / "engine_scorer" / f"{layer}_rep{replicate}", by_n,
                                         history.references(size), size,
                                         {"replicate": replicate, "seed": history.seed, "task": self.task_id,
                                          "provenance": "SYNTHETIC" if self.mock or self.dry_run else "LIVE"})
            for row in scored["rounds"]:
                cell = next(r for r in self.results if (r["replicate"], r["evaluation_size"], r["n"])
                            == (replicate, size, row["observation_count"]))
                row["timepoint_status"] = cell["status"]
                row["design_status"] = cell["design_status"]
                row["equals_time_point_nmsle_primary"] = row["nmsle_primary"] == cell["icl"]["primary_rate"]
                if size == OFFICIAL:
                    rmsle = cell["targets"]["primary_rate"]["operational"]["rmsle"]
                    row["rmsle_primary"] = rmsle
                    row["autoscilab_exact_threshold_rmsle_below_0_01"] = rmsle < .01
            self.engine_scores.append({"replicate": replicate, "seed": history.seed, "evaluation_size": size, **scored})

    def finish(self, status):
        self.score_pending()
        directions = []
        for n, size in sorted({(r["n"], r["evaluation_size"]) for r in self.results}):
            rows = [r for r in self.results if (r["n"], r["evaluation_size"]) == (n, size)]
            ours = [{"status": "valid", **h.ours(size, n)} for h in self.histories]
            directions.append({"group": "task", "n": n, "evaluation_size": size, "direction": closed_direction(
                [{"status": r["status"] if r["design_status"] == "valid" else "run_design_failure", **r["icl"]}
                 for r in rows], ours)})
        write_out(self.output / "scores.json", {
            "status": status, "synthetic": self.mock or self.dry_run, "interpret_synthetic_scores": False,
            "benchmark": BENCHMARK, "task_id": self.task_id,
            "primary_metric": "nmsle_primary of chembench_core.evaluation_grid_scores on the 256-point grid",
            "engine_scorer": self.engine_scores, "cells": self.results, "timepoint_directions": directions})
        lines = ["replicate,seed,observation_count,round,evaluation_size,nmsle_primary,timepoint_status,design_status"]
        for block in self.engine_scores:
            for row in block["rounds"]:
                lines.append(",".join(str(v) for v in (block["replicate"], block["seed"], row["observation_count"],
                                                       row["round"], block["evaluation_size"], repr(row["nmsle_primary"]),
                                                       row["timepoint_status"], row["design_status"])))
        write_text_out(self.output / "points_icl.csv", "\n".join(lines) + "\n")

        def incidence(counts):
            return {kind: {**c, "ratio": c["with_domain_word"] / c["summaries"] if c["summaries"] else None}
                    for kind, c in counts.items()}

        receipt = {"unit_id": self.unit_id, "benchmark": BENCHMARK, **self.task_fields, "seeds": list(SEEDS),
                   "process": process_versions(),
                   "started_utc": self.started, "ended_utc": utc(), "files": self.draft.get("files", {}),
                   "settings": self.settings, "track_spent_before_usd": 0,
                   "global_spent_before": self.global_spent_before, "starting_balance_usd": self.cap_usd,
                   "status": status, "calls": self.budget.calls, "call_count": len(self.budget.calls),
                   "spent_usd": self.budget.spent, "remaining_usd": self.cap_usd - self.budget.spent,
                   "paid_api_calls": 0 if self.mock or self.dry_run else len(self.budget.calls),
                   "accounting_kind": "synthetic_not_paid" if self.mock or self.dry_run else "live",
                   "stop_reason": self.budget.stopped_reason, "completed_initial_requests": self.completed,
                   "incomplete_initial_requests": self.planned - self.completed,
                   "planned_initial_requests": self.planned,
                   "domain_word_incidence": incidence(self.transport.counts),
                   "domain_word_incidence_newtonbench": incidence(self.transport.newtonbench_counts),
                   "planned_initial_requests_by_kind": dict(self.planned_by_kind),
                   "completed_initial_requests_by_kind": dict(self.completed_by_kind),
                   "projection": self.budget.projection, "reduced_official_layer": self.reduced_official_layer,
                   "official_layer_reduction": self.official_reduction,
                   "incomplete_official_timepoints": self.incomplete_official_timepoints,
                   "engine_scorer_equals_time_point_scores": (all(row["equals_time_point_nmsle_primary"]
                                                                  for block in self.engine_scores for row in block["rounds"])
                                                              if self.engine_scores else None),
                   "input_tokens": sum(c["input_tokens"] or 0 for c in self.budget.calls),
                   "output_tokens": sum(c["output_tokens"] or 0 for c in self.budget.calls)}
        if resume_ng.RESUME is not None:
            receipt["resume"] = self.resume_record()
        write_out(self.output / "receipt.json", receipt)
        planned_costs = {}
        for thinking_tokens in (700, 3000, 6000):
            total = 0
            for request in self.requests:
                output_estimate = thinking_tokens + 16 * (len(request["query_mapping"]) or len(self.interface.names))
                total += self.budget.cost(len(json.dumps(request["body"])) / 4, output_estimate)
            planned_costs[str(thinking_tokens)] = total
        write_out(self.output / "cost_plan.json", {
            "request_count": len(self.requests), "expected_initial_requests": self.planned,
            "maximum_attempts": self.planned * 2, "thinking_scenarios_usd": planned_costs, "planning_only_not_guard": True,
            "reservation_rule": self.settings["reservation_rule"],
            "maximum_input_tokens_reserved": max((input_token_reservation(r["body"]) for r in self.requests), default=0)})
        return receipt

    def resume_record(self):
        """What the receipt adds for a resumed run; `spent_usd` is recorded + sent (a693e40's record)."""
        state = resume_ng.RESUME
        replayed = self.budget.calls[:len(state.replayed)]
        sent = self.budget.calls[len(state.replayed):]
        return {"index": self.resume_index, "starting_receipt": f"starting_receipt.resume{self.resume_index}.json",
                "resumed_run_started_utc": json.loads((self.output / "starting_receipt.json").read_text())["started_utc"],
                **state.summary(), "replayed_spent_usd": math.fsum(row["cost_usd"] for row in replayed),
                "sent_spent_usd": math.fsum(row["cost_usd"] for row in sent),
                "rule": "recorded attempts replayed from their response files (nothing sent; usage counted as "
                        "recorded); every existing file matched byte for byte (an engine-scorer forecast by its array); "
                        "only unanswered attempts sent"}

    def run_resumed(self):
        """`--resume` (resume_ng.Resume): the run in `output` must be this unit, benchmark, task, settings and mode,
        started and not finished; the new starting receipt gets a resume index and the old one stays."""
        start_file = self.output / "starting_receipt.json"
        if not start_file.exists():
            raise ValueError("resume: no started run in the output folder")
        if (self.output / "receipt.json").exists():
            raise ValueError("resume: the run in the output folder is finished; nothing to resume")
        start = json.loads(start_file.read_text())
        expected = {"unit_id": self.unit_id, "benchmark": BENCHMARK, **self.task_fields, "settings": self.settings,
                    "mock": self.mock, "dry_run": self.dry_run}
        if any(start.get(key) != value for key, value in expected.items()):
            raise ValueError("resume: the started run has another unit, benchmark, task, settings or mode")
        self.resume_index = next(i for i in range(1, 1000)
                                 if not (self.output / f"starting_receipt.resume{i}.json").exists())
        resume_ng.RESUME = resume_ng.Resume(self.output)
        try:
            write_new(self.output / f"starting_receipt.resume{self.resume_index}.json", {
                "unit_id": self.unit_id, "benchmark": BENCHMARK, **self.task_fields, "seeds": list(SEEDS),
                "process": process_versions(), "started_utc": self.started, "settings": self.settings,
                "files": self.draft.get("files", {}), "global_spent_before": self.global_spent_before,
                "track_spent_before_usd": 0, "starting_balance_usd": self.cap_usd, "mock": self.mock,
                "dry_run": self.dry_run, "resume": self.resume_index, "resumes_run_started_utc": start["started_utc"],
                **{key: value for key, value in resume_ng.RESUME.summary().items()
                   if key.startswith(("recorded", "last"))}})
            status = "complete"
            try:
                self.run_closed()
            except BudgetStopped:
                if len(resume_ng.RESUME.replayed) < resume_ng.RESUME.recorded_attempts:
                    resume_ng.RESUME.fail("resume: the cap stopped the replay before every recorded attempt")
                status = "budget_stopped"
            except ResumeMismatch:
                raise
            except Exception:
                if resume_ng.RESUME.sent:  # a failure while replaying writes nothing more into the recorded run
                    self.finish("failed")
                raise
            if len(resume_ng.RESUME.replayed) != resume_ng.RESUME.recorded_attempts:
                resume_ng.RESUME.fail("resume: the run ended before replaying every recorded attempt")
            if status == "complete" and self.completed != self.planned:
                raise ValueError(f"request count mismatch: {self.completed} != {self.planned}")
            return self.finish("dry_run_complete" if self.dry_run and status == "complete" else status)
        finally:
            resume_ng.RESUME = None

    def run(self):
        if self.resume:
            return self.run_resumed()
        self.output.mkdir(parents=True, exist_ok=False)
        write_out(self.output / "starting_receipt.json", {
            "unit_id": self.unit_id, "benchmark": BENCHMARK, **self.task_fields, "seeds": list(SEEDS),
            "process": process_versions(),
            "started_utc": self.started, "settings": self.settings, "files": self.draft.get("files", {}),
            "global_spent_before": self.global_spent_before, "track_spent_before_usd": 0,
            "starting_balance_usd": self.cap_usd, "mock": self.mock, "dry_run": self.dry_run})
        status = "complete"
        try:
            self.run_closed()
        except BudgetStopped:
            status = "budget_stopped"
        except Exception:
            self.finish("failed")
            raise
        if status == "complete" and self.completed != self.planned:
            raise ValueError(f"request count mismatch: {self.completed} != {self.planned}")
        return self.finish("dry_run_complete" if self.dry_run and status == "complete" else status)


# ---------------------------------------------------------------- command line
def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--benchmark", choices=(BENCHMARK,), required=True)
    result.add_argument("--task", choices=depth_tasks(), required=True, help="a row id of the engine's depth manifest")
    result.add_argument("--output", type=Path, required=True)
    modes = result.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--mock", action="store_true")
    modes.add_argument("--allow-paid-api", action="store_true")
    result.add_argument("--seal-manifest", type=Path)
    result.add_argument("--registration", type=Path, required=True)
    result.add_argument("--max-tokens", type=int, choices=(16000, 32000), default=32000)
    result.add_argument("--global-spent-before", default="unverified")
    result.add_argument("--rounds", type=int, default=REGISTERED_ROUNDS, help=f"1..{ENGINE_CELL_ROUNDS}")
    result.add_argument("--checkpoints", type=base.integer_list, default=None,
                        help="comma-separated observation counts from 4 to 4 + rounds; default every observation")
    result.add_argument("--official-checkpoints", choices=OFFICIAL_CHOICES, default="none",
                        help="the 1,000-point official layer: none, or the last checkpoint only")
    result.add_argument("--cap-usd", type=float, default=REGISTERED_CAP_USD, help=f"above 0, at most {base.MAX_CAP_USD}")
    result.add_argument("--resume", action="store_true", help="continue the unfinished run in --output; answered "
                        "attempts are replayed from their files, only unanswered ones are sent (registration v1-r)")
    return result


def main(argv=None, *, client_factory=None):
    """`client_factory` is for the synthetic paid-path check only (tests and the unit's check script); the command line
    always builds the provider client through the shared seal gate."""
    cli = parser()
    args = cli.parse_args(argv)
    try:
        check_engine()
        if args.resume:
            # The one exception to "output must not exist": a started, unfinished run (registration v1-r).
            if not (args.output / "starting_receipt.json").exists():
                raise ValueError("resume: no started run in the output folder")
            if (args.output / "receipt.json").exists():
                raise ValueError("resume: the run in the output folder is finished; nothing to resume")
        elif args.output.exists():
            raise ValueError("output must not exist; previous artifacts are immutable")
        plan = ng_plan(args.rounds, args.checkpoints, args.official_checkpoints, args.cap_usd)
        sources = task_sources(args.task)
        draft = seal_draft(args.task, args.max_tokens, sources, args.registration, plan)
        if args.allow_paid_api:
            seal = json.loads(args.seal_manifest.read_text()) if args.seal_manifest else {}
            for key in ("benchmark", "task", "history_assignment", "d0_prediction_requests"):
                if seal.get(key) != draft[key]:
                    raise ValueError(f"sealed {key} differs")
            client = live_client(True, args.seal_manifest, ng_settings(args.task, plan, args.max_tokens),
                                 dependencies(args.task) + [args.registration], factory=client_factory)
        else:
            client = SyntheticClient()
        unit = NGUnit(args.task, args.output, sources, dry_run=args.dry_run, mock=args.mock or args.dry_run,
                      max_tokens=args.max_tokens, client=client, global_spent_before=args.global_spent_before,
                      draft=draft, plan=plan, resume=args.resume)
        receipt = unit.run()
        if args.resume:
            write_out(args.output / f"seal_draft.resume{unit.resume_index}.json", draft)
            print(json.dumps({"resume_receipt": {key: receipt["resume"][key] for key in (
                "index", "recorded_attempts", "replayed_attempts", "sent_attempts", "first_sent_attempt",
                "recorded_spent_usd", "replayed_spent_usd", "sent_spent_usd")}, "spent_usd": receipt["spent_usd"]}))
        else:
            write_out(args.output / "seal_draft.json", draft)
        print(json.dumps({"status": receipt["status"], "requests": receipt["completed_initial_requests"],
                          "paid_api_calls": receipt["paid_api_calls"]}))
        return 4 if receipt["status"] == "budget_stopped" else 0
    except (ValueError, OSError, ResumeMismatch) as exc:
        cli.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
