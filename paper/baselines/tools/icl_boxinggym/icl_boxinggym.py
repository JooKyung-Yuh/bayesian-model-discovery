"""ICL (+LLM design) over the five BoxingGym paper tasks. Dry runs and mock runs never construct a live client.

One run = one task, three replicates (seeds 31, 37, 41). Each replicate starts at the engine's own
`--d0-policy random2` designs, lets the model choose 20 experiments from the engine's paper design
pool, and forecasts the engine's 256 evaluation points (minus the observed designs, `--q-policy
exclude_observed`) at x = 2, 7, 12, 22 observations. Scores are the engine's: nMSE against the
noise-free mean on Q and, for a binary task, the Brier score against the held-out outcomes.

Unit ICL-X: `--rounds`, `--checkpoints` and `--seeds` state another budget, other scoring points
or other worlds (a seed is the engine's `--seed`: world, evaluation points, pool and start), and
`--cap-usd` another per-task cap (a plan with more requests needs one: the ten-call projection
multiplies every remaining request by its kind's largest cost x 1.25). Their defaults are the values
above, so an unstated option changes nothing; stated or not, the values are in `settings()`, which
the sealed manifest must match exactly.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bayes_model_discovery.experimental import boxinggym_core as bg
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
from tools.icl_boxinggym.protocol import (
    CHECKPOINTS,
    QUERY_BATCH,
    ROUNDS,
    TASKS,
    Observation,
    brier,
    forbidden_pattern,
    forbidden_words,
    interface_for,
    json_bytes,
    nmse_truth,
    prediction_result,
    primary_metric,
    render_prediction,
    render_selection,
    repair_request,
    selection_fallback,
    validate_prediction,
    validate_selection,
)

UNIT = "ICL-BOXINGGYM-CLOSED-LOOP"
#: Proposed per-task cap (HANDBACK section cost plan); the checks and the per-kind projection are the
#: ChemBench tool's `Budget`.
CAP_USD = 20.0
SEEDS = (31, 37, 41)
#: The Ours v4 conditions (final_panel_bai_v4.sh): paper profile, random2 start, Q 256 minus the
#: observed designs, 20 rounds, the paper design grid.
PROFILE = {"profile": "paper", "q": 256, "grid": 100, "rounds": ROUNDS, "d0_policy": "random2",
           "q_policy": "exclude_observed"}
#: Unit ICL-X: the engine's own round bound (boxinggym_core `--rounds` 1..30 since unit U-REC) and
#: the observations before round 1 (`select_d0_designs` random2: two for every task here).
MAX_ROUNDS = 30
D0_COUNT = 2
#: Unit ICL-X: the largest per-task cap `--cap-usd` accepts (a guard against a mistyped cap).
MAX_CAP_USD = 100.0
ROOT = Path(__file__).resolve().parent
ENGINE = Path(bg.__file__).resolve()


def run_parameters(rounds=ROUNDS, checkpoints=CHECKPOINTS, seeds=SEEDS, cap_usd=CAP_USD):
    """The budget, the scoring points, the worlds and the cap of one run, checked: 1..MAX_ROUNDS rounds;
    checkpoints strictly increasing from the start (x = D0_COUNT) to the last round (x = D0_COUNT
    + rounds), so no paid selection goes unscored; distinct non-negative seeds; a cap in
    (0, MAX_CAP_USD] USD."""
    rounds, checkpoints, seeds = int(rounds), tuple(int(x) for x in checkpoints), tuple(int(s) for s in seeds)
    cap_usd = float(cap_usd)
    if not 1 <= rounds <= MAX_ROUNDS:
        raise ValueError(f"rounds must be 1..{MAX_ROUNDS}")
    if (not checkpoints or checkpoints[0] != D0_COUNT or checkpoints[-1] != D0_COUNT + rounds
            or any(b <= a for a, b in zip(checkpoints, checkpoints[1:]))):
        raise ValueError(f"checkpoints must increase strictly from {D0_COUNT} to {D0_COUNT} + rounds")
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be distinct non-negative integers")
    if not 0 < cap_usd <= MAX_CAP_USD:
        raise ValueError(f"cap_usd must be above 0 and at most {MAX_CAP_USD}")
    return {"rounds": rounds, "checkpoints": checkpoints, "seeds": seeds, "cap_usd": cap_usd}


def settings(task, max_tokens=32000, *, rounds=ROUNDS, checkpoints=CHECKPOINTS, seeds=SEEDS, cap_usd=CAP_USD):
    return {"unit_id": UNIT, "task": task, "model": MODEL, "max_tokens": max_tokens,
            "stream": max_tokens == 32000, "timeout_seconds": 300, "max_retries": 0,
            "output_config": {"effort": "medium"}, "thinking": {"type": "adaptive", "display": "summarized"},
            "cap_usd": cap_usd, "reservation_rule": RESERVATION_RULE, "projection_rule": PROJECTION_RULE,
            "seeds": list(seeds), "checkpoints": list(checkpoints), "query_batch": QUERY_BATCH,
            **PROFILE, "rounds": rounds,
            "selection_fallback_stream": "icl_boxinggym_selection_fallback",
            "mock_selection_stream": "icl_boxinggym_synthetic_selection",
            "max_attempts_per_request": 2, "engine_sha256": sha(ENGINE)}


class World:
    """One task at one seed, through the engine's own calls in the engine's own order: the world,
    the evaluation points, the design pool, the random2 start; then observations on request."""

    def __init__(self, task, seed, rounds=ROUNDS):
        self.task, self.seed = task, seed
        self.spec = bg.TASKS[task]
        self.interface = interface_for(self.spec)
        self.world_seed = seed_for("boxinggym_world", task, seed)
        self.evaluation_seed = seed_for("boxinggym_evaluation", task, seed, PROFILE["profile"])
        self.design_pool_seed = seed_for("boxinggym_design_pool", task, seed)
        self.stack = contextlib.ExitStack()
        self.env_seed_log = self.stack.enter_context(bg.env_seed_injection(self.world_seed))
        self.environment = bg.TaskEnvironment(self.spec, self.world_seed)
        self.evaluation_x, self.heldout = self.environment.evaluation_points(PROFILE["q"], self.evaluation_seed)
        # The engine sizes the pool with the run's own `--rounds` (survival: at least D0 + rounds rows).
        pool_target = bg.design_pool_target(self.spec, PROFILE["grid"], rounds)
        if self.spec.environment_pool_source == "sampled":
            np.random.seed(self.design_pool_seed)
        self.pool = bg.build_design_pool(self.spec, self.environment, pool_target)
        self.d0 = bg.select_d0_designs(self.spec, self.pool, PROFILE["d0_policy"], seed)
        self.truth = self.spec.truth_mean(self.environment, self.evaluation_x) if self.spec.truth_mean else None
        self.observations = []
        self.observed_indices = []

    @property
    def evaluation_rows(self):
        q = np.asarray(self.evaluation_x, dtype=float)
        return q[:, None] if q.ndim == 1 else q

    def start(self):
        for row, handle in self.d0:
            self._observe(self.pool.index_of(row), handle)
        return list(self.observations)

    def observe_index(self, index):
        return self._observe(int(index), self.pool.handles[int(index)])

    def _observe(self, index, handle):
        y = self.environment.observe(handle)
        if isinstance(y, list):
            raise TypeError("a multi-component task is outside this tool")
        row = tuple(float(v) for v in np.atleast_1d(self.pool.rows[index]))
        self.observations.append(Observation(len(self.observations), row, float(y)))
        self.observed_indices.append(index)
        return self.observations[-1]

    def candidate_indices(self):
        """The design pool, minus the observed designs where the task excludes them (survival)."""
        if self.spec.design_exclusion == "observed":
            return [i for i in range(len(self.pool.rows)) if i not in self.observed_indices]
        return list(range(len(self.pool.rows)))

    def unobserved_mask(self):
        return bg.unobserved_q_mask(self.evaluation_x, [o.x for o in self.observations])

    def close(self):
        self.stack.close()


class BoxingTransport(Transport):
    """The ChemBench transport's metered two-attempt call with the BoxingGym protocol on top."""

    def __init__(self, client, output, budget, *, task, max_tokens=32000, synthetic=False):
        super().__init__(client, output, budget, max_tokens=max_tokens, synthetic=synthetic)
        self.task = task
        self.domain_words = forbidden_pattern(task)

    def request(self, prompt, request_id, kind, *, interface, mapping, observations, replicate,
                round_index, remaining):
        ids = tuple(row["id"] for row in mapping)
        validations, attempts, request = [], [], prompt
        for attempt in range(2):
            text, stop, error, stem = self.call(request, request_id, attempt, kind, remaining)
            if error:
                attempts.append("transport_failure")
                request = prompt
                continue
            result = (validate_prediction(text, interface, ids, stop) if kind == "prediction"
                      else validate_selection(text, ids, stop))
            validations.append(result)
            status = "valid" if result.valid else "retryable_validation_failure"
            self.budget.calls[-1]["status"] = status
            attempts.append(status)
            write_new(str(stem) + ".validation.json", {"status": status, "errors": result.errors})
            if result.valid:
                break
            request = (repair_request(prompt, text, "; ".join(result.errors), kind, interface, len(ids))
                       if text.strip() else prompt)
        if kind == "prediction":
            return {**prediction_result(validations, ids, interface, observations), "attempt_statuses": attempts}
        for result in validations:
            if result.valid:
                return {"choice": result.values["choice"], "status": "valid", "attempt_statuses": attempts}
        index = selection_fallback(self.task, replicate, round_index, len(ids))
        return {"choice": ids[index], "status": "selection_fallback", "attempt_statuses": attempts}


class SyntheticClient:
    """Deterministic fake answers from the request alone, never published responses."""

    def __init__(self):
        self.messages = self

    def create(self, **body):
        text = body["messages"][0]["content"]
        if "New inputs (" in text:
            table = text.split("\nNew inputs (", 1)[1].split("\n\n", 1)[0].splitlines()[2:]
            value = 0.5 if "between 0 and 1" in text else 1.0
            result = {"predictions": {row.split(",")[0]: value for row in table}}
        else:
            count = int(text.split("Candidate experiments (", 1)[1].split(")", 1)[0])
            digest = hashlib.sha256(json_bytes(body)).hexdigest()
            rng = np.random.default_rng(seed_for("icl_boxinggym_synthetic_selection", digest))
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


def dependencies():
    """Every file a seal must bind: this tool's modules and prompts, the engine modules, and
    every code/prompt/price file of the shared ChemBench ICL tool - the shared verifier
    (`transport.verify_seal`) requires all of the latter whatever the caller imports, so a
    draft that bound fewer files could never be sealed for a live run."""
    from bayes_model_discovery.experimental import belief_packet, core_count
    from tools.icl_baseline import transport
    from tools.icl_boxinggym import protocol
    files = [Path(inspect.getfile(module)).resolve() for module in (bg, belief_packet, core_count, protocol)]
    files.append(Path(__file__).resolve())
    files += sorted((ROOT / "prompts").glob("*.txt"))
    shared = Path(transport.__file__).resolve().parent
    files += [p.resolve() for p in shared.rglob("*")
              if p.is_file() and p.suffix in (".py", ".txt", ".json") and "tests" not in p.parts]
    return sorted(set(files))


def seal_draft(task, max_tokens, registration, *, rounds=ROUNDS, checkpoints=CHECKPOINTS, seeds=SEEDS, cap_usd=CAP_USD):
    """What a sealed manifest must bind for one task: settings, the code and prompt files, the
    engine module, and the six start-forecast request digests (replicates x batches at x = 2)."""
    d0_requests = []
    for replicate, seed in enumerate(seeds, 1):
        world = World(task, seed, rounds)
        try:
            observations = world.start()
            for offset, batch in batches_of(world, observations):
                prompt, _ = render_prediction(world.interface, observations, batch)
                d0_requests.append({"replicate": replicate, "offset": offset,
                                    "body_sha256": hashlib.sha256(json_bytes(request_body(prompt, max_tokens))).hexdigest()})
        finally:
            world.close()
    files = {str(path): sha(path) for path in [*dependencies(), Path(registration).resolve()]}
    return {"status": "draft_not_execution_authority",
            "settings": settings(task, max_tokens, rounds=rounds, checkpoints=checkpoints, seeds=seeds, cap_usd=cap_usd),
            "task": task,
            "seeds": list(seeds), "files": files, "d0_prediction_requests": d0_requests,
            "synthetic_corpus_is_not_a_live_request_seal": True,
            "receipt_fields": ["unit_id", "task", "started_utc", "ended_utc", "files", "settings", "global_spent_before",
                               "track_spent_before_usd", "calls", "spent_usd", "remaining_usd", "stop_reason"]}


def batches_of(world, observations):
    """The unobserved evaluation rows, deduplicated (a repeated coordinate is asked once), in
    batches of QUERY_BATCH: (offset, rows)."""
    mask = world.unobserved_mask()
    rows = world.evaluation_rows[mask]
    unique = []
    seen = set()
    for row in rows:
        key = tuple(float(v) for v in row)
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return [(offset, unique[offset:offset + QUERY_BATCH]) for offset in range(0, len(unique), QUERY_BATCH)]


class Unit:
    def __init__(self, task, output, *, dry_run=False, mock=False, max_tokens=32000, client=None,
                 global_spent_before="unverified", draft=None, seeds=SEEDS, rounds=ROUNDS,
                 checkpoints=CHECKPOINTS, cap_usd=CAP_USD):
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        self.parameters = run_parameters(rounds, checkpoints, seeds, cap_usd)
        self.rounds, self.checkpoints, self.cap_usd = (self.parameters["rounds"], self.parameters["checkpoints"],
                                                       self.parameters["cap_usd"])
        self.task, self.output, self.seeds = task, Path(output), self.parameters["seeds"]
        self.dry_run, self.mock, self.max_tokens = dry_run, mock, max_tokens
        self.started = utc()
        self.draft = draft or {}
        self.global_spent_before = global_spent_before
        self.budget = Budget(self.cap_usd, load_prices())
        if client is None:
            if not (mock or dry_run):
                raise ValueError("a live run needs a verified client")
            client = SyntheticClient()
        self.transport = BoxingTransport(client, self.output, self.budget, task=task, max_tokens=max_tokens,
                                         synthetic=mock or dry_run)
        # Upper bound: two batches per checkpoint (256 points, 128 per batch) and one selection per
        # round; the actual batch count follows the deduplicated, unobserved evaluation rows.
        self.planned_by_kind = {"prediction": len(self.seeds) * len(self.checkpoints) * math.ceil(PROFILE["q"] / QUERY_BATCH),
                                "selection": len(self.seeds) * self.rounds}
        self.completed_by_kind = {"prediction": 0, "selection": 0}
        self.requests, self.scores, self.replicates = [], [], []

    def remaining_by_kind(self):
        return {kind: self.planned_by_kind[kind] - self.completed_by_kind[kind] for kind in self.planned_by_kind}

    def ask(self, kind, world, replicate, round_index, prompt, mapping, metadata):
        body = request_body(prompt, self.max_tokens)
        found = forbidden_words(prompt["system"] + "\n" + prompt["messages"][0]["content"], self.task)
        if found:
            raise ValueError(f"prompt forbidden words: {found}")
        identity = f"{len(self.requests):04d}-{kind}"
        record = {"request_id": identity, "body": body, "query_mapping": mapping, "replicate": replicate,
                  "round": round_index, "n": len(world.observations), **metadata,
                  "provenance": "ACTUAL_D0" if len(world.observations) == 2 else "LIVE_DECISIONS"}
        write_new(self.output / "corpus" / f"{identity}.json", record)
        self.requests.append(record)
        result = self.transport.request(prompt, identity, kind, interface=world.interface, mapping=mapping,
                                        observations=world.observations, replicate=replicate,
                                        round_index=round_index, remaining=self.remaining_by_kind())
        self.completed_by_kind[kind] += 1
        write_new(self.output / "responses" / f"{identity}.json", {**record, "body": None, **result})
        return result

    def predict(self, world, replicate):
        n = len(world.observations)
        mask = world.unobserved_mask()
        plan = batches_of(world, world.observations)
        values = {}
        batches = []
        for offset, batch in plan:
            prompt, mapping = render_prediction(world.interface, world.observations, batch)
            result = self.ask("prediction", world, replicate, n - 2, prompt, mapping,
                              {"evaluation_size": int(mask.sum()), "offset": offset, "batch_size": len(batch)})
            batches.append(result)
            for row, value in zip(mapping, result["values"]):
                values[tuple(row["coordinates"])] = value
        rows = world.evaluation_rows[mask]
        prediction = np.asarray([values[tuple(float(v) for v in row)] for row in rows], dtype=float)
        truth = None if world.truth is None else np.asarray(world.truth, dtype=float)[mask]
        requested = sum(len(b["values"]) for b in batches)
        fallback = sum(len(b["fallback_ids"]) for b in batches)
        row = {"replicate": replicate, "seed": world.seed, "x": n, "q_scored": int(mask.sum()),
               "q_excluded": int((~mask).sum()), "unique_queries": requested, "batches": len(batches),
               "fallback_query_count": fallback, "fallback_query_ratio": fallback / requested if requested else None,
               "status": "invalid_timepoint" if requested and fallback / requested > .2 else "valid",
               "nmse_truth": nmse_truth(prediction, truth),
               "brier": brier(prediction, np.asarray(world.heldout, dtype=float)[mask]) if world.interface.outcome_type == "binary" else None}
        self.scores.append(row)
        np.savez_compressed(self.output / "predictions" / f"rep{replicate}_x{n:02d}.npz",
                            x=rows, prediction=prediction, mask=mask,
                            truth=np.full(len(rows), np.nan) if truth is None else truth,
                            heldout=np.asarray(world.heldout, dtype=float)[mask])
        return row

    def run_replicate(self, replicate, seed):
        world = World(self.task, seed, self.rounds)
        selections, fallback_count = [], 0
        try:
            world.start()
            self.predict(world, replicate)
            for round_index in range(1, self.rounds + 1):
                candidates = world.candidate_indices()
                prompt, mapping = render_selection(world.interface, world.observations, world.pool.rows[candidates],
                                                   repeats_allowed=world.spec.design_exclusion != "observed")
                result = self.ask("selection", world, replicate, round_index, prompt, mapping,
                                  {"candidate_count": len(candidates)})
                chosen = candidates[int(result["choice"][1:]) - 1]
                fallback_count += result["status"] == "selection_fallback"
                observation = world.observe_index(chosen)
                selections.append({"round": round_index, "pool_index": int(chosen), "x": list(observation.x),
                                   "y": observation.y, "status": result["status"],
                                   "attempt_statuses": result["attempt_statuses"]})
                if len(world.observations) in self.checkpoints:
                    self.predict(world, replicate)
        finally:
            write_new(self.output / f"closed_rep{replicate}.json", {
                "task": self.task, "seed": seed, "replicate": replicate,
                "provenance": "SYNTHETIC" if self.mock or self.dry_run else "LIVE",
                "input_name_map": world.interface.input_name_map,
                "d0_designs": [list(map(float, np.atleast_1d(row))) for row, _ in world.d0],
                "pool_size": len(world.pool.rows), "evaluation_size": len(world.evaluation_rows),
                "observations": [{"obs_id": o.obs_id, "x": list(o.x), "y": o.y} for o in world.observations],
                "selections": selections, "selection_fallback_count": fallback_count,
                "status": "run_design_failure" if fallback_count >= 7 else "valid",
                "env_seed_injections": len(world.env_seed_log)})
            world.close()
        self.replicates.append({"replicate": replicate, "seed": seed, "selection_fallback_count": fallback_count})

    def finish(self, status):
        rows = sorted(self.scores, key=lambda r: (r["replicate"], r["x"]))
        best = {}
        points = []
        for row in rows:
            key = row["replicate"]
            value = row["nmse_truth"]
            if value is not None:
                best[key] = value if key not in best else min(best[key], value)
            points.append({"replicate": key, "x": row["x"], "nmse_truth": "not_available" if value is None else value,
                           "brier": "" if row["brier"] is None else row["brier"],
                           "best_so_far_nmse_truth": best.get(key, "not_available")})
        write_new(self.output / "scores.json", {"status": status, "task": self.task, "synthetic": self.mock or self.dry_run,
                                               "interpret_synthetic_scores": False, "checkpoints": list(self.checkpoints),
                                               "primary_metric": primary_metric(self.task),
                                               "cells": rows, "replicates": self.replicates})
        with (self.output / "points_icl.csv").open("x", newline="") as stream:
            import csv
            writer = csv.DictWriter(stream, fieldnames=("replicate", "x", "nmse_truth", "brier", "best_so_far_nmse_truth"),
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(points)
        thinking = {kind: {**counts, "ratio": counts["with_domain_word"] / counts["summaries"] if counts["summaries"] else None}
                    for kind, counts in self.transport.counts.items()}
        receipt = {"unit_id": UNIT, "task": self.task, "started_utc": self.started, "ended_utc": utc(),
                   "files": self.draft.get("files", {}), "settings": settings(self.task, self.max_tokens, **self.parameters),
                   "track_spent_before_usd": 0, "global_spent_before": self.global_spent_before,
                   "starting_balance_usd": self.cap_usd, "status": status,
                   "calls": self.budget.calls, "call_count": len(self.budget.calls),
                   "spent_usd": self.budget.spent, "remaining_usd": self.cap_usd - self.budget.spent,
                   "paid_api_calls": 0 if self.mock or self.dry_run else len(self.budget.calls),
                   "accounting_kind": "synthetic_not_paid" if self.mock or self.dry_run else "live",
                   "stop_reason": self.budget.stopped_reason,
                   "completed_initial_requests": sum(self.completed_by_kind.values()),
                   "completed_initial_requests_by_kind": dict(self.completed_by_kind),
                   "planned_initial_requests_upper_bound": sum(self.planned_by_kind.values()),
                   "planned_initial_requests_by_kind_upper_bound": dict(self.planned_by_kind),
                   "projection": self.budget.projection, "domain_word_incidence": thinking,
                   "input_tokens": sum(c["input_tokens"] or 0 for c in self.budget.calls),
                   "output_tokens": sum(c["output_tokens"] or 0 for c in self.budget.calls)}
        write_new(self.output / "receipt.json", receipt)
        planned_costs = {}
        for thinking_tokens in (700, 3000, 6000):
            total = 0
            for request in self.requests:
                input_estimate = len(json.dumps(request["body"])) / 4
                output_estimate = thinking_tokens + 16 * (len(request["query_mapping"]) if request["request_id"].endswith("prediction") else 1)
                total += self.budget.cost(input_estimate, output_estimate)
            planned_costs[str(thinking_tokens)] = total
        write_new(self.output / "cost_plan.json", {"request_count": len(self.requests),
                  "requests_by_kind": dict(self.completed_by_kind),
                  "expected_initial_requests_upper_bound": sum(self.planned_by_kind.values()),
                  "maximum_attempts": 2 * sum(self.planned_by_kind.values()),
                  "thinking_scenarios_usd": planned_costs, "planning_only_not_guard": True,
                  "reservation_rule": RESERVATION_RULE, "cap_usd": self.cap_usd,
                  "maximum_input_tokens_reserved": max((input_token_reservation(r["body"]) for r in self.requests), default=0)})
        return receipt

    def run(self):
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "predictions").mkdir()
        if self.draft:
            write_new(self.output / "seal_draft.json", self.draft)
        write_new(self.output / "starting_receipt.json", {"unit_id": UNIT, "task": self.task, "started_utc": self.started,
                  "settings": settings(self.task, self.max_tokens, **self.parameters), "files": self.draft.get("files", {}),
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
        return self.finish("dry_run_complete" if self.dry_run and status == "complete" else status)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--task", choices=TASKS, required=True)
    result.add_argument("--output", type=Path, required=True)
    modes = result.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--mock", action="store_true")
    modes.add_argument("--allow-paid-api", action="store_true")
    result.add_argument("--seal-manifest", type=Path)
    result.add_argument("--registration", type=Path, required=True)
    result.add_argument("--max-tokens", type=int, choices=(16000, 32000), default=32000)
    result.add_argument("--global-spent-before", default="unverified")
    # Unit ICL-X: budget, scoring points and worlds; the defaults are the 2026-10-02 registration's.
    result.add_argument("--rounds", type=int, default=ROUNDS)
    result.add_argument("--checkpoints", type=integer_list, default=CHECKPOINTS, help="comma-separated, e.g. 2,7,12,22,32")
    result.add_argument("--seeds", type=integer_list, default=SEEDS, help="comma-separated engine --seed values")
    result.add_argument("--cap-usd", type=float, default=CAP_USD, help=f"per-task cap, above 0 and at most {MAX_CAP_USD}")
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
        parameters = run_parameters(args.rounds, args.checkpoints, args.seeds, args.cap_usd)
        draft = seal_draft(args.task, args.max_tokens, args.registration, **parameters)
        client = None
        if args.allow_paid_api:
            if args.seal_manifest is None:
                raise ValueError("a live run needs --seal-manifest")
            seal = json.loads(args.seal_manifest.read_text())
            if seal.get("task") != args.task:
                raise ValueError("the sealed manifest names another task")
            client = live_client(True, args.seal_manifest, settings(args.task, args.max_tokens, **parameters),
                                 [*dependencies(), args.registration])
        unit = Unit(args.task, args.output, dry_run=args.dry_run, mock=args.mock, max_tokens=args.max_tokens,
                    client=client, global_spent_before=args.global_spent_before, draft=draft, **parameters)
        receipt = unit.run()
        print(json.dumps({"status": receipt["status"], "requests": receipt["completed_initial_requests"],
                          "paid_api_calls": receipt["paid_api_calls"]}))
        return 4 if receipt["status"] == "budget_stopped" else 0
    except ValueError as error:
        cli.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
