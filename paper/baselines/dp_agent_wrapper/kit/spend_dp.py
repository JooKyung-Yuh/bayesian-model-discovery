"""Spend of a DP-Agent runs folder (2026-10-05): the sum of usd_estimate_total over every <stem>.usage.json,
which dp_agent_throttled.py rewrites after each call (Opus 5.5 at 4 / 20 USD per 1M input-side / output tokens).
Prints one line: "<usd> USD over <n> cells (input-side <t_in>, output <t_out>)". Usage: spend_dp.py <runs folder>"""
import glob
import json
import os
import sys

usd, t_in, t_out, n = 0.0, 0, 0, 0
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.usage.json"))):
    try:
        u = json.load(open(f))
    except (OSError, ValueError):   # a file being replaced right now; the next check reads it
        continue
    n += 1
    usd += float(u.get("usd_estimate_total") or 0.0)
    for v in (u.get("by_model") or {}).values():
        t_in += int(v.get("input_tokens") or 0)
        t_out += int(v.get("output_tokens") or 0)
print(f"{usd:.4f} USD over {n} cells (input-side {t_in}, output {t_out})")
