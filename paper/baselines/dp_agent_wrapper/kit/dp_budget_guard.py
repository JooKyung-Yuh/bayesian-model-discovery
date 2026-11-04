"""The experiment budget of the throttled DP-Agent protocol (2026-10-05).

BudgetGuard stands in for DP's executor on the agent's side only: a request must hold exactly one experiment, and at most
`budget` experiments run in total. A refused request runs nothing and raises ValueError; DP's agent loop turns that into
"Error running experiment: ..." for the model (scienceagent/agent.py, the except branch after self.executor.run).
"""


class BudgetGuard:
    def __init__(self, inner, budget):
        self._inner, self.budget, self.used, self.refused = inner, int(budget), 0, []

    def run(self, experiments):
        if not isinstance(experiments, list) or len(experiments) != 1:
            n = len(experiments) if isinstance(experiments, list) else None
            self.refused.append({"reason": "not_one_experiment", "n": n, "used_before": self.used})
            raise ValueError("protocol: a <run_experiment> must hold exactly ONE experiment (a JSON array of length 1); "
                             "nothing was run")
        if self.used >= self.budget:
            self.refused.append({"reason": "budget_used", "n": 1, "used_before": self.used})
            raise ValueError(f"protocol: the experiment budget of {self.budget} is used up; nothing was run. "
                             "Submit your <final_law>")
        out = self._inner.run(experiments)   # an experiment that fails inside DP's executor is not counted
        self.used += 1
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)
