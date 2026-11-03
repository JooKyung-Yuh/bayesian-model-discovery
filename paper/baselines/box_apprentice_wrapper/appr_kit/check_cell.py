"""Check a finished 0 USD cell (stub or mock-transport) end to end. 0 USD.

  python -m appr_kit.check_cell <cell dir> [<cell dir> ...]
Per cell: the receipt (status completed, budget_exhausted, mode stub|mock-transport, usage flagged mock, no errors or
cap refusals, every request seen by the test transport and none in violation, world checks all true, the checkout's
porcelain unchanged, PyTensor compile dir inside the cell), the queue log beside the cell (START and EXIT=0), curve.json
(b = 0..budget, every row scored with a finite nmse_truth, brier on binary tasks, carried rows between checkpoints),
laws.jsonl (one accepted row per checkpoint), experiments.jsonl (one row per budget step), bg_native_scores.json (one
entry per checkpoint) and the call count (scripted replies: budget scientist calls + 1 prior proposal + 2 per posterior
checkpoint). Prints CHECK OK / CHECK FAIL per cell and a final `CHECKED n=<n> fail=<f>`.
"""
import json
import math
import os
import re
import sys


def check(cell):
    bad = []
    rp = os.path.join(cell, "receipt.json")
    if not os.path.isfile(rp):
        return [f"no receipt.json"], {}
    r = json.load(open(rp))
    mode = r.get("mode")
    if mode not in ("stub", "mock-transport"):
        bad.append(f"mode {mode!r} is not a 0 USD mode")
    if r.get("status") != "completed" or r.get("termination") != "budget_exhausted":
        bad.append(f"status {r.get('status')} termination {r.get('termination')}: {str(r.get('error'))[:200]}")
    u = r.get("usage") or {}
    if not u.get("mock"):
        bad.append("usage not flagged mock")
    if u.get("errors") or u.get("refused_by_cap"):
        bad.append(f"usage errors {u.get('errors')} refused {u.get('refused_by_cap')}")
    if r.get("mock_violations"):
        bad.append(f"transport violations {r['mock_violations'][:2]}")
    if r.get("mock_requests_seen") != u.get("calls"):
        bad.append(f"transport saw {r.get('mock_requests_seen')} requests, ledger {u.get('calls')} calls")
    wc = r.get("world_checks") or {}
    if not wc or any(v is False for v in wc.values()):
        bad.append(f"world checks {wc}")
    if not (r.get("checkout_after") or {}).get("unchanged"):
        bad.append("checkout porcelain changed during the run")
    if os.path.join(cell, "ptc") not in str(r.get("pytensor_flags")):
        bad.append(f"PYTENSOR_FLAGS {r.get('pytensor_flags')!r}")
    budget, cps = int(r["budget"]), [int(c) for c in r["checkpoints"]]
    expected_calls = budget + (1 if 0 in cps else 0) + 2 * len([c for c in cps if c > 0])
    if u.get("calls") != expected_calls:
        bad.append(f"calls {u.get('calls')} != expected {expected_calls}")
    log = os.path.join(os.path.dirname(cell), os.path.basename(cell) + ".log")
    if os.path.isfile(log):
        text = open(log, errors="replace").read()
        if not re.search(r"^START \d{4}-\d\d-\d\d \d\d:\d\d:\d\d KST.*\bhost \S+", text, re.M):
            bad.append("log has no START line")
        if not re.search(r"^EXIT=0 END \d{4}-\d\d-\d\d \d\d:\d\d:\d\d KST", text, re.M):
            bad.append("log has no EXIT=0 line")
    else:
        bad.append(f"no queue log {log}")
    curve = json.load(open(os.path.join(cell, "curve.json")))
    if [row["b"] for row in curve] != list(range(budget + 1)):
        bad.append(f"curve budgets {[row['b'] for row in curve][:5]}...")
    binary = r["world"]["outcome_type"] == "binary"
    for row in curve:
        if row.get("status") != "scored" or not math.isfinite(float(row.get("nmse_truth") or float("nan"))):
            bad.append(f"curve row b={row['b']} {row.get('status')} {row.get('error')}")
            break
        if binary and not math.isfinite(float(row.get("brier", float("nan")))):
            bad.append(f"curve row b={row['b']} without brier")
            break
        want = f"checkpoint_{row['b']}" if row["b"] in cps else f"carried_from_checkpoint_{max(c for c in cps if c <= row['b'])}"
        if row.get("source") != want:
            bad.append(f"curve row b={row['b']} source {row.get('source')} != {want}")
            break
    laws = [json.loads(l) for l in open(os.path.join(cell, "laws.jsonl")) if l.strip()]
    if [l["b"] for l in laws] != cps or any(l["fit_status"] != "accepted" for l in laws):
        bad.append(f"laws {[(l['b'], l['fit_status']) for l in laws]}")
    exps = [json.loads(l) for l in open(os.path.join(cell, "experiments.jsonl")) if l.strip()]
    if [e["b"] for e in exps] != list(range(1, budget + 1)) or not all(e["success"] for e in exps):
        bad.append(f"experiments {len(exps)} rows, {sum(1 for e in exps if e['success'])} successful")
    native = json.load(open(os.path.join(cell, "bg_native_scores.json")))["checkpoints"]
    if [n["b"] for n in native] != cps or any(n["status"] != "ok" for n in native):
        bad.append(f"native scores {[(n['b'], n['status']) for n in native]}")
    summary = {"task": r["args"]["task"], "seed": r["args"]["seed"], "mode": mode, "wall_s": r.get("wall_s"),
               "calls": u.get("calls"), "synthetic_usd": u.get("usd"), "rows": len(curve)}
    return bad, summary


def main(argv=None):
    cells = argv if argv is not None else sys.argv[1:]
    fail = 0
    for cell in cells:
        cell = os.path.abspath(cell)
        try:
            bad, summary = check(cell)
        except Exception as exc:  # noqa: BLE001
            bad, summary = [f"{type(exc).__name__}: {exc}"], {}
        if bad:
            fail += 1
            print(f"CHECK FAIL {cell}: " + "; ".join(bad))
        else:
            print(f"CHECK OK {cell} {json.dumps(summary)}")
    print(f"CHECKED n={len(cells)} fail={fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
