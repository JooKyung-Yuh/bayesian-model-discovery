#!/usr/bin/env python3
"""Render the appendix tables from paper/data/appendix_tables/*.json (standard library only).

usage: python paper/data/tables.py [--table E1|E2|F1|D3|G4|H|C]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent / "appendix_tables"
NAME = {"chembench": "ChemBench", "boxinggym": "BoxingGym", "forcebench": "ForceBench", "newtonbench": "NewtonBench"}


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def sci(v):
    return "-" if v is None else f"{v:.2e}"


def table_e1():
    d = load("E1_paired_comparison.json")
    for mode, title in (("figure", "Figure 2 floor rule (1e-6 on ForceBench only)"), ("all", "floor 1e-6 on all benchmarks"), ("none", "unfloored")):
        print(f"\n## E1 paired comparison: {title}\n")
        print("| benchmark | baseline | n | pairs (tasks) | BMD | baseline | BMD lower | log10 ratio [95%] |\n|---|---|---:|---|---|---|---|---|")
        for row in d["rows"]:
            r = row[mode]
            print(f"| {NAME[row['benchmark']]} | {row['baseline']} | {row['x']} | {r['pairs']} ({r['tasks']}) | {sci(r['bmd_geomean'])} | {sci(r['baseline_geomean'])} | "
                  f"{r['bmd_lower']}/{r['pairs']}{' (ties ' + str(r['ties']) + ')' if r['ties'] else ''} | {r['mean_log10_ratio']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] |")


def table_e2():
    d = load("E2_per_task_final_error.json")["tables"]
    for b, t in d.items():
        cols = sorted({k.split("|")[0] for k in t["table"] if k.endswith("|figure")}, key=lambda c: (c != "BMD", c))
        print(f"\n## E2 final error per task: {NAME[b]} (n = {t['x']}; geometric mean over seeds, seed count)\n")
        print("| task | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
        for task in t["tasks"]:
            cells = []
            for c in cols:
                v = t["table"].get(f"{c}|figure", {}).get(task)
                cells.append(f"{sci(v['geomean'])} ({v['seeds']})" if v else "-")
            print(f"| {task} | " + " | ".join(cells) + " |")


def table_f1():
    d = load("F1_outcome_classes.json")["summary"]
    cols = ["strict", "containment (symbolic)", "containment (numeric)", "surrogate", "unresolved", "certified non-equivalent", "no law"]
    print("\n## F1 final top-1 outcome classes\n")
    print("| method | benchmark | runs | " + " | ".join(cols) + " | proposal failure | selection failure |\n|---|---|---:|" + "---:|" * (len(cols) + 2))
    for key, s in d.items():
        m, b = key.split("|"); f = s["failures"]
        pf = f.get("proposal failure (truth never admitted)", "-") if m == "BMD" else "-"
        sf = (f.get("selection failure (admitted, never top-1)", 0) + f.get("selection failure (strict top-1 earlier, lost)", 0)) if m == "BMD" else "-"
        print(f"| {m} | {NAME[b]} | {s['runs']} | " + " | ".join(str(s["classes"].get(c, 0)) for c in cols) + f" | {pf} | {sf} |")


def table_d3():
    d = load("D3_cost_calls.json")
    print("\n## D3 LLM calls and cost per run, mean (range)\n")
    print("| method | benchmark | runs | calls | USD |\n|---|---|---:|---|---|")
    for r in d["rows"]:
        calls = f"{r['calls_mean']:.1f} ({r['calls_min']:.0f}-{r['calls_max']:.0f})" if r["calls_mean"] is not None else "-"
        print(f"| {r['method']} | {NAME[r['benchmark']]} | {r['runs']} | {calls} | {r['usd_mean']:.2f} ({r['usd_min']:.2f}-{r['usd_max']:.2f}) |")
    print("\n### Blocked ground-truth evaluator attempts (LLM-AutoSciLab receipts)\n")
    print("| benchmark | edition | receipts | with >= 1 blocked | total (min-max) |\n|---|---|---:|---:|---|")
    for key, a in d["blocked_summary"].items():
        b, e = key.split("|")
        print(f"| {NAME[b]} | {e} | {a['receipts']} | {a['receipts_with_blocked']} | {a['blocked_attempts_total']} ({a['min']}-{a['max']}) |")


def table_g4():
    d = load("G4_c2pert.json")
    print("\n## G4 perturbed constants: world-balanced geometric mean [band] (units)\n")
    for b in ("chembench", "newtonbench"):
        print(f"\n### {NAME[b]}\n\n| condition | n=10 registered | n=10 Bayes action | n=20 registered | n=20 Bayes action |\n|---|---|---|---|---|")
        for c in ("A", "B", "C", "D", "BMD"):
            cells = []
            for n in (10, 20):
                for kind in ("registered", "Bayes action"):
                    v = d["table"].get(f"{n}|{b}|{c}|{kind}")
                    cells.append(f"{sci(v['geomean'])} [{sci(v['band_low'])}, {sci(v['band_high'])}] ({v['units']})" if v else "= registered")
            print(f"| {c} | " + " | ".join(cells) + " |")


def table_h():
    d = load("H_breadth.json")
    print("\n## H breadth: joint EIG vs random on 24 ChemBench tasks, seed 31\n")
    for kind, r in d["results"].items():
        print(f"- {kind}: {r['joint_lower']}/{r['pairs']} joint lower; mean log10(random/joint) {r['mean_log10_diff_random_minus_joint']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")
    print("\n| task | joint EIG | random |\n|---|---|---|")
    for k, v in d["per_task"].items():
        print(f"| {k.split('|')[0]} | {sci(v['joint_eig'])} | {sci(v['random'])} |")


def table_c():
    d = load("C_forcebench_configurations.json")
    print("\n## C ForceBench integrator settings per world (from the run records)\n")
    for edition, v in d.items():
        num = v.get("numerics", v) if isinstance(v, dict) else v
        if not isinstance(num, dict):
            continue
        for world, cfgs in sorted(num.items()):
            if isinstance(cfgs, list):
                print(f"- {edition} / {world}: " + "; ".join(f"{c.get('method')} step {c.get('dt')}" + (f" softening {c['softening']}" if 'softening' in c else "") + f" ({c.get('mode')})" for c in cfgs))


TABLES = {"E1": table_e1, "E2": table_e2, "F1": table_f1, "D3": table_d3, "G4": table_g4, "H": table_h, "C": table_c}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=sorted(TABLES))
    args = parser.parse_args()
    for name, fn in TABLES.items():
        if args.table in (None, name):
            fn()
