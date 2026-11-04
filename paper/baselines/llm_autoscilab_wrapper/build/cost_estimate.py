#!/usr/bin/env python3
"""Cost estimate per task from the released agent's call pattern (0 USD, no LLM).

Calls per cell: counted in the stub runs (mocked Claude, stand-in PySR), which execute the released loop's
call sites; cross-checked against the count derived from discovery.py (README / report).
Input tokens per call: the characters the released prompt builders produced in the stub runs (system +
user text actually sent) / 1.942 characters per token - measured on 533 Opus 5.5 requests of our own
engine (bg_rerun_20261004, input incl. cache tokens). The stub's short synthetic replies make later
prompts shorter than real ones (memory and analysis text grow with real replies); the 'high' row adds 50 %.
Output tokens per call: central 1,500 and high 3,000 (assumptions; our engine's Opus 5.5 effort-medium
proposer calls measured mean 938, median 837, p90 1,518 output tokens on the same 533 requests).
Prices: 4 / 20 USD per 1M input / output tokens.
Plan calls per cell: the stub mean, except BoxingGym = 14 (the released pattern without early termination,
4 discovery iterations x 3 calls + 2 validation calls; the stub mean includes one cell that stopped early).
The number of calls is not fixed by the code: an iteration runs min(sum of the experiments the reply's
regions request, the iteration cap, the remaining budget) outside the forced sweep iterations, so replies
that request fewer experiments add iterations and calls. 'sensitivity' gives BoxingGym at 16 calls (the
validation iteration on the discovery path, as in the survival stub cell, plus final arbitration).
All sums use full precision; only the printed values are rounded.

Usage: python cost_estimate.py <stub runs root> <out.json>
"""
import glob
import json
import os
import sys

CHARS_PER_TOKEN = 1.942
PRICE_IN, PRICE_OUT = 4.0, 20.0
CELLS = {"chembench_depth": 27, "chembench_breadth": 24, "boxinggym": 15, "forcebench": 15}
PLAN_CALLS = {"boxinggym": 14}          # others: the stub mean
CANARY = ("chembench", "boxinggym", "forcebench")   # one cell each
SENSITIVITY = {"boxinggym": 16}
SCENARIOS = (("central", 1.0, 1500), ("high", 1.5, 3000))   # (label, input multiplier, output tokens per call)


def main():
    root, out = sys.argv[1], sys.argv[2]
    per_bench = {}
    for cell in sorted(glob.glob(os.path.join(root, "stub_*", "*"))):
        rec = os.path.join(cell, "receipt.json")
        if not os.path.exists(rec) or "variants" in cell:   # default-option cells only
            continue
        r = json.load(open(rec))
        if r["args"].get("names", "masked") != "masked" or r["status"] != "completed":
            continue
        bench = r["args"]["benchmark"]
        reqs = [json.loads(l) for l in open(os.path.join(cell, "llm_requests.jsonl"))]
        chars = [len(q["system"]) + sum(len(m["content"]) for m in q["messages"]) for q in reqs]
        per_bench.setdefault(bench, []).append({"cell": os.path.basename(cell), "calls": len(reqs),
                                                "mean_chars": sum(chars) / max(len(chars), 1),
                                                "total_chars": sum(chars)})
    exact = {}   # full precision
    for bench, cells in per_bench.items():
        calls = sum(c["calls"] for c in cells) / len(cells)
        tin_call = sum(c["total_chars"] for c in cells) / sum(c["calls"] for c in cells) / CHARS_PER_TOKEN
        usd_call = {label: (tin_call * in_mult * PRICE_IN + out_tok * PRICE_OUT) / 1e6
                    for label, in_mult, out_tok in SCENARIOS}
        plan_calls = PLAN_CALLS.get(bench, calls)
        exact[bench] = {"cells": cells, "stub_calls": calls, "tin_call": tin_call, "usd_call": usd_call,
                        "plan_calls": plan_calls}
    table = {}
    for bench, e in exact.items():
        table[bench] = {"stub_cells": [c["cell"] for c in e["cells"]],
                        "stub_calls_per_cell": [c["calls"] for c in e["cells"]],
                        "stub_mean_calls_per_cell": round(e["stub_calls"], 2),
                        "input_tokens_per_call": round(e["tin_call"]),
                        "plan_calls_per_cell": round(e["plan_calls"], 2),
                        **{f"usd_per_call_{k}": round(v, 5) for k, v in e["usd_call"].items()},
                        **{f"usd_per_cell_{k}": round(e["plan_calls"] * v, 3) for k, v in e["usd_call"].items()},
                        **{f"usd_per_3_seeds_{k}": round(3 * e["plan_calls"] * v, 2) for k, v in e["usd_call"].items()}}

    def cost(bench, n_cells, calls=None):
        e = exact[bench]
        c = e["plan_calls"] if calls is None else calls
        return {k: n_cells * c * v for k, v in e["usd_call"].items()}

    plan_exact = {}
    for name, n in CELLS.items():
        bench = "chembench" if name.startswith("chembench") else name
        if bench in exact:
            plan_exact[name] = {"cells": n, **cost(bench, n)}
    plan_exact["total"] = {"cells": sum(v["cells"] for v in plan_exact.values()),
                           **{k: sum(v[k] for v in plan_exact.values()) for k, _, _ in SCENARIOS}}
    canary = {k: sum(cost(b, 1)[k] for b in CANARY if b in exact) for k, _, _ in SCENARIOS}
    sens = {}
    for bench, calls in SENSITIVITY.items():
        if bench in exact:
            n = CELLS[bench]
            delta = {k: cost(bench, n, calls)[k] - cost(bench, n)[k] for k, _, _ in SCENARIOS}
            sens[f"{bench}_at_{calls}_calls"] = {
                "per_cell": {k: round(v, 3) for k, v in cost(bench, 1, calls).items()},
                f"{n}_cells": {k: round(v, 2) for k, v in cost(bench, n, calls).items()},
                "plan_total": {k: round(plan_exact["total"][k] + delta[k], 2) for k, _, _ in SCENARIOS}}

    def rounded(d):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}

    plan = {name: rounded(v) for name, v in plan_exact.items()}
    result = {"assumptions": __doc__, "per_benchmark": table, "plan": plan,
              "canary_3_cells": {k: round(v, 2) for k, v in canary.items()}, "sensitivity": sens}
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "assumptions"}, indent=1))


if __name__ == "__main__":
    main()
