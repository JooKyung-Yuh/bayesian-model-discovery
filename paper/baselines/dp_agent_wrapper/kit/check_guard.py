"""0 USD self-test of dp_budget_guard.BudgetGuard (2026-10-05). Prints one line per case and 'GUARD OK' at the
end; exits 1 on the first failed case. Usage: python check_guard.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dp_budget_guard import BudgetGuard  # noqa: E402


class Inner:
    def __init__(self):
        self.calls = []
        self.label = "inner"

    def run(self, experiments):
        if experiments[0].get("bad"):
            raise RuntimeError("executor failure")
        self.calls.append(experiments)
        return [{"ok": True}]


def expect(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        sys.exit(1)


def refused(g, exps):
    try:
        g.run(exps)
    except ValueError as e:
        return str(e)
    return None


inner = Inner()
g = BudgetGuard(inner, 2)
expect("attribute passthrough", g.label == "inner")
expect("batch of two refused, nothing run", refused(g, [{"p1": 1}, {"p1": 2}]) is not None and inner.calls == [] and g.used == 0)
expect("empty list refused", refused(g, []) is not None and g.used == 0)
expect("non-list refused", refused(g, {"p1": 1}) is not None and g.used == 0)
expect("first single runs", g.run([{"p1": 1}]) == [{"ok": True}] and g.used == 1 and len(inner.calls) == 1)
try:
    g.run([{"bad": True}])
    failed = False
except RuntimeError:
    failed = True
expect("executor failure not counted", failed and g.used == 1)
expect("second single runs", g.run([{"p1": 2}]) == [{"ok": True}] and g.used == 2 and len(inner.calls) == 2)
msg = refused(g, [{"p1": 3}])
expect("third single refused by budget, nothing run", msg is not None and "budget of 2" in msg and len(inner.calls) == 2)
expect("refusal log", [r["reason"] for r in g.refused] == ["not_one_experiment", "not_one_experiment", "not_one_experiment",
                                                          "budget_used"])
print("GUARD OK")
