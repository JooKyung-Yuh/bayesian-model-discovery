"""Leak block, D0 injection and the per-experiment law recorder around the released DiscoveryLoop.run.

Leak block. During DiscoveryLoop.run the agent's oracle must not answer `evaluate_law` or
`get_ground_truth_law_str`. The released loop calls `evaluate_law` in four places (mid-run GT
check, pool candidates, final candidates, final evaluation - discovery.py 1928, 1971, 2182, 2215),
each inside try/except, so a raise is absorbed and no ground-truth number reaches the prompts,
the memory, `_best_gt_equation` or the final selection. Every attempt is counted.
  ChemBench: BlockedChemBenchOracle subclasses the native ChemBenchOracle (isinstance stays true,
  so the released ChemBench-only code paths - staged PySR iterations, variable filter, skeleton
  families - behave as released) and overrides the two methods to raise.
  BoxingGym / ForceBench: the adapter oracles have no in-run evaluator at all (both methods raise).
Scoring uses a separate scorer object the loop never sees, only after an experiment's law has
been recorded; nothing computed by a scorer flows back into the run.

D0. Our initial designs and observations are added to the loop's ExperimentStore before the
first outer iteration. `_oracle_calls` is left at 0, so the released budget arithmetic (phases at
70/20/10 %, UCB->EI switch, `oracle_calls % eq_learner_every`) runs over the B experiments the
agent chooses, exactly as in a released run with budget B; the data table simply starts with |D0|
rows.

Recorder. For every b = 0..B: law(b) = the equation the loop holds at the moment experiment b+1 is
requested, i.e. after b of its own experiments: the loop's working equation of its last fit,
`max(_candidate_pool, key=r2)` (discovery.py 1914-1915; PySR, skeleton and validated LLM-hypothesis
candidates together), or `_eq_learner.best` while the pool is empty; law(B) = the run's returned
`best_equation` (after the released final fitting and LLM arbitration). b without any fitted law is
recorded as no_law. `_eq_learner.best` alone (what the 9/6 no-leak wrapper logged) is kept in every
row as `eq_learner_best_law_str`. If the loop stops early (confidence), the final law is carried to
the remaining b and marked so.
"""
import json
import os
import time
import traceback

BLOCK_MESSAGE = "held-out evaluator disabled during discovery"
BLOCK = {"attempts": 0, "where": []}


def blocked(name):
    BLOCK["attempts"] += 1
    BLOCK["where"].append(name)
    print(f"[no-leak] blocked {name} attempt #{BLOCK['attempts']}", flush=True)
    raise RuntimeError(BLOCK_MESSAGE)


def blocked_chembench_class():
    from autoscilab.oracle.chembench import ChemBenchOracle

    class BlockedChemBenchOracle(ChemBenchOracle):
        ASL_BLOCKED = True

        def evaluate_law(self, law_str):
            blocked("ChemBenchOracle.evaluate_law")

        def get_ground_truth_law_str(self):
            blocked("ChemBenchOracle.get_ground_truth_law_str")

    return BlockedChemBenchOracle


class LawRecorder:
    def __init__(self, out_dir, budget, d0_count):
        self.out_dir = out_dir
        self.budget = int(budget)
        self.d0_count = int(d0_count)
        self.rows = {}
        self.experiments = []
        self.fits = []        # one row per EquationLearner.fit call (record only; the released learner runs as released)
        self.loop = None

    @staticmethod
    def _eq_row(eq):
        if eq is None:
            return {"law_str": None, "skeleton": None, "train_r2": None, "train_rmsle": None}
        return {"law_str": eq.law_str, "skeleton": eq.skeleton_str, "train_r2": float(eq.r2),
                "train_rmsle": float(eq.rmsle)}

    def attach(self, loop):
        self.loop = loop
        oracle = loop._oracle
        released_run = oracle.run

        def run(params):
            self.snapshot()
            result = released_run(params)
            self.experiments.append({"b": loop._oracle_calls + 1, "requested": dict(params),
                                     "executed": dict(result.params), "y": float(result.measurement),
                                     "metadata": {k: v for k, v in (result.metadata or {}).items() if k != "outputs"}})
            return result

        oracle.run = run   # instance attribute: the loop's `self._oracle.run(...)` reaches this first
        learner = loop._eq_learner
        released_fit = learner.fit

        def fit(*args, **kwargs):
            row = {"b": int(loop._oracle_calls), "n_obs": len(loop._store), "seconds": None, "returned": None}
            t0 = time.time()
            try:
                out = released_fit(*args, **kwargs)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"[:200]
                raise
            finally:
                row["seconds"] = round(time.time() - t0, 2)
                self.fits.append(row)
            row["returned"] = out is not None
            return out

        learner.fit = fit   # instance attribute: records when the released learner fitted and how long it took

    def snapshot(self):
        loop = self.loop
        b = int(loop._oracle_calls)
        if b in self.rows:
            return
        pool = [c for c in list(loop._candidate_pool) if c is not None]
        learner_best = loop._eq_learner.best
        if pool:
            # the loop's own working equation after its last fit: discovery.py 1914-1915
            # `eq = max(self._candidate_pool, key=lambda c: float(c.r2))` (PySR, skeleton and validated
            # LLM-hypothesis candidates together) - the same max-R2 rule as the arbitration fallback
            best, source = max(pool, key=lambda c: float(c.r2)), "candidate_pool.max_r2"
        elif learner_best is not None:
            best, source = learner_best, "eq_learner.best"
        else:
            best, source = None, "no_law"
        self.rows[b] = {"b": b, "n_obs": len(loop._store), "source": source, **self._eq_row(best),
                        "eq_learner_best_law_str": learner_best.law_str if learner_best is not None else None,
                        "pool_top": [{"skeleton": c.skeleton_str, "law_str": c.law_str, "train_r2": float(c.r2)}
                                     for c in pool[:3]],
                        "phase": loop._current_phase, "hypothesis": str(loop._current_hypothesis)[:400],
                        "t": time.time()}

    def finalize(self, result, error=None):
        loop = self.loop
        b_end = int(loop._oracle_calls) if loop is not None else 0
        if result is not None:
            self.rows[b_end] = {"b": b_end, "n_obs": len(loop._store),
                                "source": "final" if result.best_equation else "no_law",
                                **self._eq_row(result.best_equation), "phase": "final",
                                "termination": result.termination_reason, "t": time.time()}
            for b in range(b_end + 1, self.budget + 1):
                self.rows[b] = dict(self.rows[b_end], b=b, source=self.rows[b_end]["source"] + "_carried_after_termination")
        with open(os.path.join(self.out_dir, "laws.jsonl"), "w") as fh:
            for b in sorted(self.rows):
                fh.write(json.dumps(self.rows[b], default=str) + "\n")
        with open(os.path.join(self.out_dir, "experiments.jsonl"), "w") as fh:
            for row in self.experiments:
                fh.write(json.dumps(row, default=str) + "\n")
        return {"b_end": b_end, "rows": len(self.rows), "error": error}


class RunContext:
    """What the wrapped DiscoveryLoop.run needs: D0 results, the recorder, and the outcome."""

    def __init__(self, d0_results, recorder):
        self.d0_results = list(d0_results)
        self.recorder = recorder
        self.result = None
        self.error = None
        self.loop = None


def install_run_wrapper(context):
    from autoscilab.loop import discovery as D

    released_run = getattr(D.DiscoveryLoop.run, "_released_run", D.DiscoveryLoop.run)

    def run(self):
        oracle = self._oracle
        if not getattr(type(oracle), "ASL_BLOCKED", False):
            raise RuntimeError(f"refusing to run: oracle {type(oracle).__name__} is not leak-blocked")
        if self._oracle_calls != 0 or len(self._store) != 0:
            raise RuntimeError("D0 injection expects a fresh loop")
        for res in context.d0_results:
            self._store.add(res)
        BLOCK["attempts"], BLOCK["where"] = 0, []
        context.loop = self
        context.recorder.attach(self)
        try:
            result = released_run(self)
        except BaseException as exc:
            context.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            context.recorder.finalize(None, error=context.error)
            raise
        context.result = result
        context.recorder.finalize(result)
        return result

    run._released_run = released_run
    D.DiscoveryLoop.run = run
