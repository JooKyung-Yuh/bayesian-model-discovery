"""Registered ICL units. Dry runs and mock runs never construct a live client.

Unit ICL-X (2026-10-06): CLOSED takes `--rounds`, `--checkpoints`, `--official-checkpoints` and
`--cap-usd` (e.g. 30 rounds with a forecast after every observation, x = 3, 4, ..., 33). Their
defaults are the released plan (60 rounds, TIMES, OFFICIAL_TIMES, the unit's cap), and an unstated
plan leaves every request, record and setting as released; a stated plan enters the settings, which
the sealed manifest must match exactly. The other units refuse them."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bayes_model_discovery.experimental.core_count import seed_for
from tools.icl_baseline.protocol import (
    BOUNDS,
    MASKED_NAMES,
    OFFICIAL_TIMES,
    ROOT,
    TIMES,
    VISIBLE_NAMES,
    cell_direction,
    closed_direction,
    fixed_direction,
    forbidden_words,
    json_bytes,
    metrics,
    number,
    observations_from_rows,
    render,
    score_timepoint,
)
from tools.icl_baseline.transport import (
    CAPS,
    Budget,
    BudgetStopped,
    RequestSkipped,
    Transport,
    input_token_reservation,
    live_client,
    load_prices,
    request_body,
    settings,
    sha,
    utc,
    write_new,
)

FIXED, CLOSED, VISIBLE = tuple(CAPS)
INPUT_FILES = ("config.json", "observations.jsonl", "evaluation_inputs.npz",
               "official_evaluation_inputs.npz", "reference_after_finish.npz",
               "official_reference_after_finish.npz", "official_scores_after_finish.json")
# Task mode (CLOSED without recorded histories): one of the paper manifest's task ids.
PAPER_MANIFEST = Path(__file__).resolve().parents[2] / "manifests" / "chembench_paper_manifest.json"
DEFAULT_TASK = "c5_pingpong_bisubstrate__easy__v0"
REGISTERED_NOISE = .01
#: Unit ICL-X: the CLOSED plan. The fixed3 start, the released plan, and a guard against a mistyped cap.
D0_COUNT = 3
RELEASED_ROUNDS = 60
MAX_CAP_USD = 400.0
#: Requests per forecast: the 256-point layer in batches of 128, the official 1,000-point layer in batches of 100.
BATCHES = {256: 2, 1000: 10}


def closed_plan(rounds=RELEASED_ROUNDS, checkpoints=TIMES, official_checkpoints=OFFICIAL_TIMES, cap_usd=None):
    """The CLOSED plan, checked: 1..60 rounds; forecast checkpoints strictly increasing from the
    start (x = 3) to the last round (x = 3 + rounds), so no paid selection goes unscored; official
    checkpoints strictly increasing among them; a cap in (0, MAX_CAP_USD] USD (None = the unit's)."""
    rounds, checkpoints = int(rounds), tuple(int(x) for x in checkpoints)
    official = tuple(int(x) for x in official_checkpoints)
    cap = float(CAPS[CLOSED] if cap_usd is None else cap_usd)
    if not 1 <= rounds <= RELEASED_ROUNDS:
        raise ValueError(f"rounds must be 1..{RELEASED_ROUNDS}")
    if (not checkpoints or checkpoints[0] != D0_COUNT or checkpoints[-1] != D0_COUNT + rounds
            or any(b <= a for a, b in pairwise(checkpoints))):
        raise ValueError(f"checkpoints must increase strictly from {D0_COUNT} to {D0_COUNT} + rounds")
    if not set(official) <= set(checkpoints) or any(b <= a for a, b in pairwise(official)):
        raise ValueError("official checkpoints must increase strictly and be forecast checkpoints")
    if not 0 < cap <= MAX_CAP_USD:
        raise ValueError(f"cap_usd must be above 0 and at most {MAX_CAP_USD}")
    return {"rounds": rounds, "checkpoints": checkpoints, "official_checkpoints": official, "cap_usd": cap}


RELEASED_PLAN = closed_plan()


def plan_settings(unit, max_tokens=32000, plan=None):
    """The transport settings, plus the CLOSED plan where it is not the released one: an unstated
    plan leaves them byte-identical to the released runs'; a stated one is sealed."""
    result = settings(unit, max_tokens)
    if plan is None or plan == RELEASED_PLAN:
        return result
    result["cap_usd"] = plan["cap_usd"]
    result.update(rounds=plan["rounds"], checkpoints=list(plan["checkpoints"]),
                  official_checkpoints=list(plan["official_checkpoints"]))
    return result


def paper_tasks():
    """The paper manifest's task ids in manifest order; `--task` takes one of them."""
    return [row["id"] for row in json.loads(PAPER_MANIFEST.read_text())]


@cache
def evaluation_designs(size):
    """The runner's own evaluation points: the 256-point layer (`chem_evaluation_inputs`) and
    the official 1,000-point layer, drawn by the calls chembench_core.run makes."""
    from adapters.chembench_observation import sample_designs
    from bayes_model_discovery.experimental.chembench_core import (
        OFFICIAL_EVALUATION_COUNT,
        OFFICIAL_EVALUATION_SEED,
    )
    if size == 256:
        return tuple(sample_designs(256, seed_for("chem_evaluation_inputs", 20260907)))
    if size != OFFICIAL_EVALUATION_COUNT:
        raise ValueError("unregistered evaluation layer")
    return tuple(sample_designs(OFFICIAL_EVALUATION_COUNT, OFFICIAL_EVALUATION_SEED))


@cache
def task_references(task_id, size):
    """Reference values of one task at the evaluation points, from a session that has finished."""
    source = TaskSource(task_id, 0)
    session = source.session(budget=1)
    session.finish()
    values = session.reference_values(list(evaluation_designs(size)))
    return {key: np.asarray(values[key], dtype=float) for key in ("primary_rate", "noiseless_observation")}


class TaskSource:
    """CLOSED without a recorded history. The task's own oracle declares what a history's
    files declared - the fixed3 start, the evaluation points and, after the run, the reference
    values - by the calls the runner makes. One per replicate; all three declare the same."""

    group = "task"
    names = MASKED_NAMES

    def __init__(self, task_id, replicate):
        from bayes_model_discovery.experimental.chembench_core import fixed3_designs, select_task
        row = select_task(json.loads(PAPER_MANIFEST.read_text()), task_id)
        self.task_id, self.replicate, self.root = task_id, replicate, f"task:{task_id}"
        self.config = {"task": {key: row[key] for key in ("id", "domain", "difficulty", "law_version")},
                       "noise": REGISTERED_NOISE, "d0_conditions": None}
        session = self.session(budget=3)
        schema = session.schema()
        bounds = tuple(tuple(map(float, schema["input_bounds"][name])) for name in VISIBLE_NAMES)
        if tuple(schema["input_names"]) != VISIBLE_NAMES or bounds != BOUNDS:
            raise ValueError("task interface differs from the frozen ICL prompt registration")
        self.observations = [observation for design in fixed3_designs(None)
                             for observation in observations_from_rows([session.observe(design)], VISIBLE_NAMES)]

    def session(self, budget):
        from adapters.chembench_observation import ChemBenchSession
        task = self.config["task"]
        return ChemBenchSession(task["domain"], task["difficulty"], task["law_version"],
                                noise_level=self.config["noise"], budget=budget)

    def queries(self, size):
        return np.array([[row[name] for name in VISIBLE_NAMES] for row in evaluation_designs(size)])

    def references(self, size):
        return {key: value.copy() for key, value in task_references(self.task_id, size).items()}

    def ours(self, size, n):
        return {"primary_rate": None, "noiseless_observation": None,
                "unavailable_reason": "task mode reads no run of ours; the comparison is made outside this tool"}

    def files(self):
        return [PAPER_MANIFEST]


@dataclass
class History:
    root: Path
    group: str
    replicate: int
    config: dict
    names: tuple
    observations: list

    @classmethod
    def load(cls, root, group, replicate):
        root = Path(root).resolve()
        config = json.loads((root / "config.json").read_text())
        request = json.loads((root / "llm_transport" / "round_30.request.json").read_text())
        packet = json.loads(request["bodies"][0]["messages"][0]["content"])
        names = tuple(item["name"] for item in packet["interface"]["inputs"])
        if names not in (MASKED_NAMES, VISIBLE_NAMES):
            raise ValueError("unregistered input declaration order")
        bounds = tuple(tuple(item["range"]) for item in packet["interface"]["inputs"])
        if bounds != BOUNDS or packet["interface"]["outcome_type"] != "positive":
            raise ValueError("history interface differs from registration")
        if names == MASKED_NAMES and config.get("input_name_map") != dict(zip(MASKED_NAMES, VISIBLE_NAMES)):
            raise ValueError("masked input map differs from registered declaration")
        task = config["task"]
        if (task["domain"], task["difficulty"], task["law_version"]) != ("c5_pingpong_bisubstrate", "easy", "v0"):
            raise ValueError("ICL registration is c5 easy v0 only")
        if config["noise"] != .01:
            raise ValueError("history noise differs from registration")
        rows = [json.loads(line) for line in (root / "observations.jsonl").read_text().splitlines()]
        observations = observations_from_rows(rows, names)
        if len(observations) != 63 or [o.obs_id for o in observations] != list(range(63)):
            raise ValueError("history must contain the complete ordered 63 observations")
        return cls(root, group, replicate, config, names, observations)

    def queries(self, size):
        filename = "evaluation_inputs.npz" if size == 256 else "official_evaluation_inputs.npz"
        with np.load(self.root / filename, allow_pickle=False) as data:
            if tuple(data["names"].tolist()) != self.names:
                raise ValueError("evaluation name order differs from declared interface")
            result = data["x"].copy()
        if result.shape != (size, 7):
            raise ValueError("evaluation shape differs from registration")
        return result

    def references(self, size):
        filename = "reference_after_finish.npz" if size == 256 else "official_reference_after_finish.npz"
        with np.load(self.root / filename, allow_pickle=False) as data:
            return {key: data[key].copy() for key in ("primary_rate", "noiseless_observation")}

    def ours(self, size, n):
        if size == 256:
            with np.load(self.root / "predictions" / f"t{n - 3:02d}.npz", allow_pickle=False) as data:
                prediction = data["mean"]
                return {key: metrics(prediction, truth)["nmsle"] for key, truth in self.references(size).items()}
        rows = json.loads((self.root / "official_scores_after_finish.json").read_text())["rounds"]
        found = [row for row in rows if row["observation_count"] == n]
        if len(found) != 1:
            raise ValueError("official score must contain one matching observation_count")
        return {"primary_rate": found[0]["nmsle"], "noiseless_observation": None,
                "unavailable_reason": "existing official scores contain only primary_rate"}

    def files(self):
        return [*(self.root / name for name in INPUT_FILES),
                self.root / "llm_transport" / "round_30.request.json",
                *(self.root / "predictions" / f"t{n - 3:02d}.npz" for n in TIMES)]


def verify_histories(histories):
    first = histories[0]
    for history in histories[1:]:
        if history.observations[:3] != first.observations[:3]:
            raise ValueError("D0 observations differ; n=3 response sharing is not valid")
        for size in (256, 1000):
            if not np.array_equal(history.queries(size), first.queries(size)):
                raise ValueError("evaluation coordinates differ across histories")


def input_files(histories):
    return sorted({path for history in histories for path in history.files()})


def dependencies():
    import inspect

    from autoscilab.oracle import chembench

    from adapters import chembench_observation
    from bayes_model_discovery.experimental import (
        belief_packet,
        chembench_core,
        core_count,
        count_proposer,
    )
    return [Path(inspect.getfile(module)).resolve() for module in (
        chembench_observation, chembench, belief_packet, chembench_core, core_count, count_proposer)]


def seal_draft(unit, max_tokens, histories, registration, code_dependencies=(), plan=None):
    files = input_files(histories) + list(code_dependencies)
    files += [p for p in ROOT.rglob("*") if p.is_file() and p.suffix in (".py", ".txt", ".json")
              and "tests" not in p.parts]
    files.append(Path(registration).resolve())
    d0_requests = []
    for history in histories[:3]:
        queries = history.queries(256)
        for offset in (0, 128):
            prompt, _ = render("prediction", history.observations[:3], queries=queries[offset:offset + 128])
            d0_requests.append({"replicate": history.replicate, "offset": offset,
                                "body_sha256": hashlib.sha256(json_bytes(request_body(prompt, max_tokens))).hexdigest()})
    task_id = getattr(histories[0], "task_id", None) if histories else None
    return {"status": "draft_not_execution_authority", "settings": plan_settings(unit, max_tokens, plan),
            **({"task": task_id} if task_id is not None else {}),
            "files": {str(path.resolve()): sha(path) for path in sorted(set(files))},
            "history_assignment": [{"path": str(h.root), "group": h.group, "replicate": h.replicate} for h in histories],
            "synthetic_corpus_is_not_a_live_request_seal": True,
            "d0_prediction_requests": d0_requests,
            "masked_receipts_reviewed": False,
            "receipt_fields": ["unit_id", "started_utc", "ended_utc", "files", "settings", "global_spent_before",
                               "track_spent_before_usd", "calls", "spent_usd", "remaining_usd", "stop_reason"]}


class SyntheticClient:
    """Deterministic fake answers from the request alone, never published responses."""

    def __init__(self):
        self.messages = self

    def create(self, **body):
        text = body["messages"][0]["content"]
        if "New inputs (" in text:
            table = text.split("\nNew inputs (", 1)[1].split("\n\n", 1)[0].splitlines()[2:]
            result = {"predictions": {row.split(",")[0]: 1.0 for row in table}}
        else:
            digest = hashlib.sha256(json_bytes(body)).hexdigest()
            rng = np.random.default_rng(seed_for("icl_synthetic_selection", digest))
            result = {"x": {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in zip(MASKED_NAMES, BOUNDS)}}
        usage = SimpleNamespace(model_dump=lambda **_: {"input_tokens": 100, "output_tokens": 100,
                                                        "output_tokens_details": {"thinking_tokens": 10}})
        return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="Synthetic protocol check."),
                                        SimpleNamespace(type="text", text=json.dumps(result))],
                               usage=usage, stop_reason="end_turn")

    @contextmanager
    def stream(self, **body):
        response = self.create(**body)
        yield SimpleNamespace(get_final_message=lambda: response)


class Unit:
    def __init__(self, unit_id, output, histories, *, dry_run=False, mock=False, max_tokens=32000, client=None,
                 global_spent_before="unverified", draft=None, plan=None):
        if plan is not None and unit_id != CLOSED and plan != RELEASED_PLAN:
            raise ValueError("only CLOSED takes another plan")
        self.unit_id, self.output, self.histories = unit_id, Path(output), histories
        self.plan = plan if unit_id == CLOSED and plan is not None else None
        closed = self.plan or RELEASED_PLAN
        self.rounds, self.checkpoints = closed["rounds"], closed["checkpoints"]
        self.official_times = closed["official_checkpoints"] if unit_id == CLOSED else OFFICIAL_TIMES
        self.cap_usd = closed["cap_usd"] if unit_id == CLOSED else CAPS[unit_id]
        self.dry_run, self.mock, self.max_tokens = dry_run, mock, max_tokens
        self.started = utc()
        self.draft = draft or {}
        # Task mode names its task in both receipts; a history run's receipts are unchanged.
        task_id = getattr(histories[0], "task_id", None) if histories else None
        self.task_fields = {"task_id": task_id} if task_id is not None else {}
        self.global_spent_before = global_spent_before
        self.budget = Budget(self.cap_usd, load_prices())
        self.transport = Transport(client, self.output, self.budget, max_tokens=max_tokens, synthetic=mock)
        # Unit ICL-K: the plan by request kind. FIXED and NAME-VISIBLE send predictions only;
        # CLOSED sends, per replicate, 16 (256-point) + 30 (1,000-point) predictions and 60 selections
        # in the released plan (unit ICL-X: in general 2 per checkpoint, 10 per official checkpoint
        # and 1 per round).
        closed_predictions = 3 * (BATCHES[256] * len(self.checkpoints) + BATCHES[1000] * len(self.official_times))
        self.planned = (len(histories) * 44 + 6 if unit_id == FIXED
                        else closed_predictions + 3 * self.rounds if unit_id == CLOSED else 96)
        self.planned_by_kind = ({"prediction": closed_predictions, "selection": 3 * self.rounds} if unit_id == CLOSED
                                else {"prediction": self.planned, "selection": 0})
        self.completed_by_kind = {"prediction": 0, "selection": 0}
        self.completed, self.requests, self.pending_scores = 0, [], []
        self.results, self.shared = [], {}
        self.reduced_plan = False
        self.reduced_official_layer = False
        self.official_reduction = None
        self.official_completed = 0
        self.incomplete_official_timepoints = []
        self.failed_design_replicates = set()
        if unit_id == VISIBLE:
            self.budget.reduce_plan = self.reduce_visible_plan
        else:
            self.budget.reduce_plan = self.reduce_official_plan

    def remaining_by_kind(self):
        return {kind: self.planned_by_kind[kind] - self.completed_by_kind[kind] for kind in self.planned_by_kind}

    def reduce_visible_plan(self):
        self.reduced_plan = True
        self.planned = self.planned_by_kind["prediction"] = 48
        return self.remaining_by_kind()

    def reduce_official_plan(self):
        total = len(self.histories) * 30 if self.unit_id == FIXED else 3 * BATCHES[1000] * len(self.official_times)
        sent_ids = {row["request_id"] for row in self.budget.calls}
        sent = sum(r.get("evaluation_size") == 1000 and r["request_id"] in sent_ids for r in self.requests)
        dropped = total - sent
        before = self.planned
        remaining_before = self.remaining_by_kind()
        self.reduced_official_layer = True
        self.planned -= dropped
        self.planned_by_kind["prediction"] -= dropped
        self.official_reduction = {
            "calls_completed": len(self.budget.calls), "completed_initial_requests": self.completed,
            "planned_before": before, "remaining_before": before - self.completed,
            "official_sent_requests": sent, "official_completed_requests": self.official_completed,
            "official_unsent_requests_removed": dropped,
            "planned_after": self.planned, "remaining_after": self.planned - self.completed,
            "remaining_by_kind_before": remaining_before, "remaining_by_kind_after": self.remaining_by_kind(),
        }
        return self.remaining_by_kind()

    def ask(self, kind, observations, metadata, *, queries=None, visible=False):
        prompt, mapping = render(kind, observations, queries=queries, visible=visible,
                                 names=VISIBLE_NAMES if visible else MASKED_NAMES)
        body = request_body(prompt, self.max_tokens)
        forbidden = forbidden_words(prompt["system"] + "\n" + prompt["messages"][0]["content"], visible)
        if forbidden:
            raise ValueError(f"prompt forbidden words: {forbidden}")
        identity = f"{len(self.requests):04d}-{kind}"
        synthetic = self.unit_id == CLOSED and (self.mock or self.dry_run) and len(observations) > 3
        corpus = ("closed_synthetic" if synthetic else "closed_d0" if len(observations) == 3 else "closed_live") if self.unit_id == CLOSED else self.unit_id
        record = {"request_id": identity, "body": body, "query_mapping": mapping, **metadata,
                  "provenance": ("SYNTHETIC" if synthetic else "ACTUAL_D0" if len(observations) == 3 else "LIVE_DECISIONS") if self.unit_id == CLOSED else "READ_ONLY_HISTORY"}
        write_new(self.output / "corpus" / corpus / f"{identity}.json", record)
        self.requests.append(record)
        if self.dry_run and self.unit_id != CLOSED:
            self.completed += 1
            self.completed_by_kind[kind] += 1
            return None
        official = metadata.get("evaluation_size") == 1000
        try:
            result = self.transport.request(prompt, identity, kind, observations, mapping=mapping,
                                            unit_id=self.unit_id, replicate=metadata["replicate"],
                                            round_index=metadata["round"], remaining=self.remaining_by_kind(),
                                            skip_unstarted=lambda: official and self.reduced_official_layer)
        except RequestSkipped:
            write_new(self.output / "responses" / f"{identity}.json", {
                **metadata, "status": "not_sent_official_layer_reduction", "attempt_statuses": [],
            })
            return None
        self.completed += 1
        self.completed_by_kind[kind] += 1
        self.official_completed += official
        write_new(self.output / "responses" / f"{identity}.json", {**metadata, **result})
        return result

    def predict(self, history, observations, n, *, visible=False):
        for size in (256, 1000):
            if size == 1000 and (self.unit_id == VISIBLE or n not in self.official_times or self.reduced_official_layer):
                continue
            queries = history.queries(size)
            batch_size = 128 if size == 256 else 100
            batches = []
            for offset in range(0, size, batch_size):
                if size == 1000 and self.reduced_official_layer:
                    break
                batch = queries[offset:offset + batch_size]
                metadata = {"history": str(history.root), "group": history.group, "replicate": history.replicate,
                            "n": n, "round": n - 3, "evaluation_size": size, "offset": offset,
                            "visible": visible}
                key = (history.replicate, offset)
                if self.unit_id == FIXED and n == 3 and key in self.shared:
                    old_body, result = self.shared[key]
                    prompt, _ = render("prediction", observations, queries=batch)
                    if json_bytes(request_body(prompt, self.max_tokens)) != old_body:
                        raise ValueError("n=3 request byte identity failed")
                else:
                    result = self.ask("prediction", observations, metadata, queries=batch, visible=visible)
                    if self.unit_id == FIXED and n == 3:
                        self.shared[key] = (json_bytes(self.requests[-1]["body"]), result)
                if result is not None:
                    batches.append(result)
            if size == 1000 and self.reduced_official_layer and len(batches) != size // batch_size:
                self.incomplete_official_timepoints.append({
                    "group": history.group, "replicate": history.replicate, "n": n,
                    "evaluation_size": size, "completed_batches": len(batches),
                    "status": "incomplete_official_layer_reduction",
                })
            elif batches:
                self.pending_scores.append((history, list(observations), n, size, visible, batches))

    def score_pending(self):
        # Called only after CLOSED sessions have all finished. No truth enters prompts.
        for history, observations, n, size, visible, batches in self.pending_scores:
            score = score_timepoint(batches, history.references(size), observations)
            ours = history.ours(size, n)
            icl = {key: score["targets"][key]["operational"]["nmsle"] for key in score["targets"]}
            direction = "no directional statement"
            if (self.unit_id != CLOSED and size == 256 and score["status"] == "valid"
                    and history.replicate not in self.failed_design_replicates):
                direction = cell_direction(icl, ours)
            self.results.append({"group": history.group, "replicate": history.replicate, "n": n,
                                 "evaluation_size": size, "visible": visible, **score, "icl": icl, "ours": ours,
                                 "ours_comparison": "unpaired_reference_only" if self.unit_id == CLOSED else "same_history",
                                 "design_status": "run_design_failure" if history.replicate in self.failed_design_replicates else "valid",
                                 "direction": direction})

    def run_histories(self):
        if self.unit_id == VISIBLE:
            # Execute the entire reduced plan first, so projection never discards calls.
            for times in ((3, 10, 25, 63), (5, 15, 20, 40)):
                if self.reduced_plan and times[0] == 5:
                    break
                for history in self.histories:
                    for visible in (False, True):
                        for n in times:
                            self.predict(history, history.observations[:n], n, visible=visible)
            return
        for history in self.histories:
            for n in TIMES:
                self.predict(history, history.observations[:n], n)

    def run_closed(self):
        from adapters.chembench_observation import ChemBenchSession
        from bayes_model_discovery.experimental.chembench_core import fixed3_designs
        for history in self.histories[:3]:
            task = history.config["task"]
            session = ChemBenchSession(task["domain"], task["difficulty"], task["law_version"],
                                       noise_level=history.config["noise"], budget=D0_COUNT + self.rounds)
            observations, selections = [], []
            fallback_count = 0
            native_names = tuple(session.schema()["input_names"])
            if native_names != VISIBLE_NAMES:
                raise ValueError("simulator input order differs from registration")
            for design in fixed3_designs(history.config["d0_conditions"]):
                row = session.observe(design)
                observations.extend(observations_from_rows([row], native_names))
            if observations != history.observations[:3]:
                raise ValueError("simulator D0 differs from existing registered history")
            try:
                self.predict(history, observations, 3)
                for round_index in range(1, self.rounds + 1):
                    selection = self.ask("selection", observations, {"replicate": history.replicate,
                                         "round": round_index, "n": len(observations)})
                    fallback_count += selection["status"] == "selection_fallback"
                    values = selection["x"]
                    row = session.observe(dict(zip(native_names, values)))
                    observations.extend(observations_from_rows([row], native_names))
                    selections.append({**selection, "round": round_index, "executed_coordinates": values,
                                       "display_coordinates": [number(v) for v in values],
                                       "run_status": "run_design_failure" if fallback_count >= 7 else "valid"})
                    if len(observations) in self.checkpoints:
                        self.predict(history, observations, len(observations))
            finally:
                session.finish()
                if fallback_count >= 7:
                    self.failed_design_replicates.add(history.replicate)
                previous = [o.x for o in observations[:3]]
                boundary = one_input = 0
                for selection in selections:
                    x = tuple(selection["x"])
                    boundary += any(v == lo or v == hi for v, (lo, hi) in zip(x, BOUNDS))
                    one_input += sum(a != b for a, b in zip(x, previous[-1])) == 1
                    previous.append(x)
                write_new(self.output / f"closed_rep{history.replicate}.json", {
                    "provenance": "SYNTHETIC" if self.mock or self.dry_run else "LIVE",
                    "observations": [{"obs_id": o.obs_id, "x": o.x, "y": o.y} for o in observations],
                    "selections": selections, "selection_fallback_count": fallback_count,
                    "status": "run_design_failure" if fallback_count >= 7 else "valid",
                    "completed_selection_count": len(selections), "unique_designs": len(set(previous)),
                    "repeat_count": len(previous) - len(set(previous)),
                    "boundary_selection_ratio": boundary / len(selections) if selections else None,
                    "one_input_changed_ratio": one_input / len(selections) if selections else None})

    def finish(self, status):
        self.score_pending()
        directions = []
        if self.unit_id == FIXED:
            keys = {(row["group"], row["n"], row["evaluation_size"]) for row in self.results}
            for group, n, size in sorted(keys):
                rows = [r for r in self.results if (r["group"], r["n"], r["evaluation_size"]) == (group, n, size)]
                directions.append({"group": group, "n": n, "evaluation_size": size, "direction": fixed_direction(rows)})
        elif self.unit_id == CLOSED:
            for group in sorted({h.group for h in self.histories}):
                for n, size in sorted({(r["n"], r["evaluation_size"]) for r in self.results}):
                    rows = [r for r in self.results if (r["n"], r["evaluation_size"]) == (n, size)]
                    ours = [{"status": "valid", **h.ours(size, n)} for h in self.histories if h.group == group]
                    directions.append({"group": group, "n": n, "evaluation_size": size,
                                       "direction": closed_direction([{"status": r["status"] if r["design_status"] == "valid" else "run_design_failure", **r["icl"]} for r in rows], ours)})
        write_new(self.output / "scores.json", {"status": status, "synthetic": self.mock or self.dry_run,
                                               "interpret_synthetic_scores": False, "cells": self.results,
                                               "timepoint_directions": directions})
        thinking = {kind: {**counts, "ratio": counts["with_domain_word"] / counts["summaries"] if counts["summaries"] else None}
                    for kind, counts in self.transport.counts.items()}
        receipt = {"unit_id": self.unit_id, **self.task_fields, "started_utc": self.started, "ended_utc": utc(),
                   "files": self.draft.get("files", {}), "settings": plan_settings(self.unit_id, self.max_tokens, self.plan),
                   "track_spent_before_usd": 0, "global_spent_before": self.global_spent_before,
                   "starting_balance_usd": self.cap_usd, "status": status,
                   "calls": self.budget.calls, "call_count": len(self.budget.calls),
                   "spent_usd": self.budget.spent, "remaining_usd": self.cap_usd - self.budget.spent,
                   "paid_api_calls": 0 if self.mock or self.dry_run else len(self.budget.calls),
                   "accounting_kind": "synthetic_not_paid" if self.mock or self.dry_run else "live",
                   "stop_reason": self.budget.stopped_reason, "completed_initial_requests": self.completed,
                   "incomplete_initial_requests": self.planned - self.completed,
                   "planned_initial_requests": self.planned, "domain_word_incidence": thinking,
                   "planned_initial_requests_by_kind": dict(self.planned_by_kind),
                   "completed_initial_requests_by_kind": dict(self.completed_by_kind),
                   "projection": self.budget.projection,
                   "reduced_visible_plan": self.reduced_plan,
                   "reduced_official_layer": self.reduced_official_layer,
                   "official_layer_reduction": self.official_reduction,
                   "incomplete_official_timepoints": self.incomplete_official_timepoints,
                   "input_tokens": sum(c["input_tokens"] or 0 for c in self.budget.calls),
                   "output_tokens": sum(c["output_tokens"] or 0 for c in self.budget.calls)}
        write_new(self.output / "receipt.json", receipt)
        planned_costs = {}
        for thinking_tokens in (700, 3000, 6000):
            total = 0
            for request in self.requests:
                input_estimate = len(json.dumps(request["body"])) / 4
                output_estimate = thinking_tokens + 16 * (len(request["query_mapping"]) or 7)
                total += self.budget.cost(input_estimate, output_estimate)
            planned_costs[str(thinking_tokens)] = total
        write_new(self.output / "cost_plan.json", {"request_count": len(self.requests),
                  "expected_initial_requests": self.planned, "maximum_attempts": self.planned * 2,
                  "thinking_scenarios_usd": planned_costs, "planning_only_not_guard": True,
                  "reservation_rule": settings(self.unit_id)["reservation_rule"],
                  "maximum_input_tokens_reserved": max((input_token_reservation(r["body"]) for r in self.requests), default=0)})
        return receipt

    def run(self):
        self.output.mkdir(parents=True, exist_ok=False)
        write_new(self.output / "starting_receipt.json", {"unit_id": self.unit_id, **self.task_fields,
                  "started_utc": self.started,
                  "settings": plan_settings(self.unit_id, self.max_tokens, self.plan), "files": self.draft.get("files", {}),
                  "global_spent_before": self.global_spent_before, "track_spent_before_usd": 0,
                  "starting_balance_usd": self.cap_usd, "mock": self.mock, "dry_run": self.dry_run})
        status = "complete"
        try:
            if self.unit_id == CLOSED:
                self.run_closed()
            else:
                self.run_histories()
        except BudgetStopped:
            status = "budget_stopped"
        except Exception:
            self.finish("failed")
            raise
        if status == "complete" and self.completed != self.planned:
            raise ValueError(f"request count mismatch: {self.completed} != {self.planned}")
        return self.finish("dry_run_complete" if self.dry_run and status == "complete" else status)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--unit", choices=tuple(CAPS), required=True)
    result.add_argument("--history", type=Path, action="append", default=[],
                        help="three v2 histories, ordered replicates 1/2/3; CLOSED without any runs in task mode")
    result.add_argument("--task", choices=paper_tasks(), default=DEFAULT_TASK,
                        help="paper manifest task id; any task but the default needs CLOSED without --history")
    result.add_argument("--fac-history", type=Path, action="append", default=[])
    result.add_argument("--output", type=Path, required=True)
    modes = result.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--mock", action="store_true")
    modes.add_argument("--allow-paid-api", action="store_true")
    result.add_argument("--seal-manifest", type=Path)
    result.add_argument("--registration", type=Path, required=True)
    result.add_argument("--max-tokens", type=int, choices=(16000, 32000), default=32000)
    result.add_argument("--global-spent-before", default="unverified")
    result.add_argument("--masked-receipt", type=Path, action="append", default=[])
    # Unit ICL-X: the CLOSED plan; the defaults are the released plan.
    result.add_argument("--rounds", type=int, default=RELEASED_ROUNDS)
    result.add_argument("--checkpoints", type=integer_list, default=TIMES, help="comma-separated, e.g. 3,4,...,33")
    result.add_argument("--official-checkpoints", type=integer_list, default=OFFICIAL_TIMES,
                        help="comma-separated forecast checkpoints that also forecast the official layer, or none")
    result.add_argument("--cap-usd", type=float, default=None, help=f"CLOSED cap, above 0 and at most {MAX_CAP_USD}")
    return result


def integer_list(text):
    if str(text).strip().lower() == "none":
        return ()
    try:
        return tuple(int(item) for item in str(text).split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"comma-separated integers or none required: {text!r}") from error


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        task_mode = args.unit == CLOSED and not args.history
        if task_mode and args.fac_history:
            raise ValueError("CLOSED in task mode takes no histories")
        if not task_mode and args.task != DEFAULT_TASK:
            raise ValueError("a task other than the default runs as CLOSED without --history")
        if not task_mode and (len(args.history) != 3 or len(args.fac_history) not in (0, 3)):
            raise ValueError("exactly three histories per prompt group are required")
        if args.unit == FIXED and not args.dry_run and len(args.fac_history) != 3:
            raise ValueError("FIXED execution requires all six histories; only --dry-run is allowed now")
        if args.unit == VISIBLE and args.fac_history:
            raise ValueError("NAME-VISIBLE takes the three shown histories only")
        if args.output.exists():
            raise ValueError("output must not exist; previous artifacts are immutable")
        plan = closed_plan(args.rounds, args.checkpoints, args.official_checkpoints, args.cap_usd)
        if args.unit != CLOSED and plan != RELEASED_PLAN:
            raise ValueError("--rounds, --checkpoints, --official-checkpoints and --cap-usd are CLOSED options")
        if task_mode:
            histories = [TaskSource(args.task, replicate) for replicate in (1, 2, 3)]
        else:
            histories = [History.load(path, group, i + 1)
                         for group, paths in (("v2", args.history), ("fac", args.fac_history))
                         for i, path in enumerate(paths)]
        if args.unit == VISIBLE and any(h.names != VISIBLE_NAMES for h in histories):
            raise ValueError("NAME-VISIBLE requires shown histories")
        if args.unit != VISIBLE and any(h.names != MASKED_NAMES for h in histories):
            raise ValueError("masked units require masked histories")
        verify_histories(histories)
        dep = dependencies()
        draft = seal_draft(args.unit, args.max_tokens, histories, args.registration, dep,
                           plan if args.unit == CLOSED else None)
        if args.allow_paid_api:
            seal = json.loads(args.seal_manifest.read_text()) if args.seal_manifest else {}
            if seal.get("history_assignment") != draft["history_assignment"]:
                raise ValueError("sealed history assignment differs")
            if seal.get("task") != draft.get("task"):
                raise ValueError("sealed task differs")
            if seal.get("d0_prediction_requests") != draft["d0_prediction_requests"]:
                raise ValueError("sealed D0 request hashes differ")
            if args.unit == VISIBLE:
                receipts = [json.loads(path.read_text()) for path in args.masked_receipt]
                if (len(receipts) != 2 or {r.get("unit_id") for r in receipts} != {FIXED, CLOSED}
                        or any(r.get("accounting_kind") != "live" or r.get("status") not in ("complete", "budget_stopped") for r in receipts)
                        or seal.get("masked_receipts_reviewed") is not True):
                    raise ValueError("NAME-VISIBLE live requires reviewed receipts from both masked units")
            client = live_client(True, args.seal_manifest,
                                 plan_settings(args.unit, args.max_tokens, plan if args.unit == CLOSED else None),
                                 input_files(histories) + dep + [args.registration, *args.masked_receipt])
        else:
            client = SyntheticClient()
        unit = Unit(args.unit, args.output, histories, dry_run=args.dry_run, mock=args.mock or args.dry_run,
                    max_tokens=args.max_tokens, client=client, global_spent_before=args.global_spent_before, draft=draft,
                    plan=plan if args.unit == CLOSED else None)
        receipt = unit.run()
        write_new(args.output / "seal_draft.json", draft)
        print(json.dumps({"status": receipt["status"], "requests": receipt["completed_initial_requests"],
                          "paid_api_calls": receipt["paid_api_calls"]}))
        return 4 if receipt["status"] == "budget_stopped" else 0
    except (ValueError, OSError) as exc:
        cli.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
