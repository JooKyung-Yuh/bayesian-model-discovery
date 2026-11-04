"""Cost estimate for paid Apprentice cells from 0 USD runs and measured Opus 5.5 output lengths. 0 USD.

  python -m appr_kit.cost_estimate <runs folder of stub or mock-transport cells> [--measured <glob of *.response.json>]
Input tokens: every request of a 0 USD cell is the released prompt text the released code built (only the scripted
replies are short), so per call  input = (request characters + delta) / chars_per_token, where delta adds what real
replies would add: for the k-th scientist call, (k - 1) x (reply chars - scripted reply chars) for the longer assistant
turns in its history plus (program block chars - scripted program block chars) once a program is in its system message;
for a posterior proposal, (critic chars - scripted critic chars) (the previous synthesis is in its system message); for a
critic call, (program chars - scripted program chars). chars_per_token = 1.94 (the LLM-AutoSciLab wrapper's measurement on 533
Opus 5.5 requests; its cap rule rounds it down to 1.9).
Output tokens per call (thinking included): taken from measured Opus 5.5 calls with the same transport settings
(thinking adaptive, effort medium, max_tokens 16000) - by default the 521 proposer calls of the engine's 2026-10-04 rerun
(<development path>*/*/cell/llm_transport/*.response.json): scenario low = median, mid = p95,
high = max of that distribution, for every call of every role.
Extra calls in the high scenario: 2 retry patch re-proposals (proposal + critic each) at two of the posterior checkpoints and
3 scientist format retries per cell. USD = (input x 4 + output x 20) / 1e6.
"""
import argparse
import glob
import json
import os
import statistics

DEFAULT_MEASURED = "<development path>*/*/cell/llm_transport/*.response.json"
SCENARIOS = {
    # reply_chars: visible scientist reply; block_chars: program + critic block in the scientist's system message;
    # critic_chars: critic text in the next proposal's system; program_chars: program text in the critic's request
    "low": {"out": "median", "reply_chars": 400, "block_chars": 3000, "critic_chars": 1000, "program_chars": 2000, "extra": False},
    "mid": {"out": "p95", "reply_chars": 800, "block_chars": 5000, "critic_chars": 2000, "program_chars": 3000, "extra": False},
    "high": {"out": "max", "reply_chars": 1600, "block_chars": 8000, "critic_chars": 4000, "program_chars": 5000, "extra": True},
}
CHARS_PER_TOKEN = 1.94


def measured_outputs(pattern):
    outs = []
    for f in glob.glob(pattern):
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            continue
        u = r.get("usage_total") or {}
        if r.get("requested_model") != "claude-opus-5-5" or not u:
            continue
        outs.append(u.get("output_tokens", 0) / max(int(r.get("calls") or 1), 1))
    if len(outs) < 20:
        raise SystemExit(f"too few measured calls ({len(outs)}) under {pattern}")
    q = statistics.quantiles(outs, n=20)
    return {"n": len(outs), "median": statistics.median(outs), "p95": q[18], "max": max(outs)}


def _text(content):
    return content if isinstance(content, str) else "".join(b.get("text", "") for b in content)


def cell_estimate(cell, out_tokens, sc):
    reqs = [json.loads(l) for l in open(os.path.join(cell, "llm_requests.jsonl")) if l.strip()]
    resps = {json.loads(l)["call_index"]: json.loads(l) for l in open(os.path.join(cell, "llm_responses.jsonl")) if l.strip()}
    sci_reply = next((len(resps[r["call_index"]]["text"]) for r in reqs if r["role"] == "scientist"), 0)
    prog = next((len(resps[r["call_index"]]["text"]) for r in reqs if r["role"] == "box_loop_proposal"), 0)
    crit = next((len(resps[r["call_index"]]["text"]) for r in reqs if r["role"] == "box_loop_critic"), 0)
    tin = tout = calls = 0
    k = 0
    for r in reqs:
        chars = len(r["system"]) + sum(len(_text(m["content"])) for m in r["messages"])
        if r["role"] == "scientist":
            k += 1
            delta = (k - 1) * (sc["reply_chars"] - sci_reply)
            if "To help guide your experimentation" in r["system"]:
                delta += sc["block_chars"] - (prog + crit)
        elif r["role"] == "box_loop_critic":
            delta = sc["program_chars"] - prog
        else:
            delta = (sc["critic_chars"] - crit) if "Hypotheses: None" not in r["system"] and "sample from the prior" not in r["system"] else 0
        tin += max(chars + delta, chars) / CHARS_PER_TOKEN
        tout += out_tokens
        calls += 1
    if sc["extra"]:
        # 2 re-proposals (proposal + critic) at two posterior checkpoints, 3 scientist format retries
        props = [r for r in reqs if r["role"] != "scientist" and "sample from the prior" not in r["system"]]
        sci = [r for r in reqs if r["role"] == "scientist"]
        avg_box = sum(len(r["system"]) + sum(len(_text(m["content"])) for m in r["messages"]) for r in props) / max(len(props), 1)
        avg_sci = sum(len(r["system"]) + sum(len(_text(m["content"])) for m in r["messages"]) for r in sci) / max(len(sci), 1)
        extra_box, extra_sci = 2 * 2 * 2, 3
        tin += extra_box * (avg_box + sc["critic_chars"]) / CHARS_PER_TOKEN
        tin += extra_sci * (avg_sci + len(sci) / 2 * sc["reply_chars"] + sc["block_chars"]) / CHARS_PER_TOKEN
        tout += (extra_box + extra_sci) * out_tokens
        calls += extra_box + extra_sci
    usd = (tin * 4 + tout * 20) / 1e6
    return {"calls": calls, "input_tokens": round(tin), "output_tokens": round(tout), "usd": round(usd, 4)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--measured", default=DEFAULT_MEASURED)
    a = ap.parse_args(argv)
    m = measured_outputs(a.measured)
    cells = sorted(d for d in glob.glob(os.path.join(a.folder, "*")) if os.path.isfile(os.path.join(d, "llm_requests.jsonl")))
    out = {"measured_output_tokens_per_call": m, "chars_per_token": CHARS_PER_TOKEN, "scenarios": SCENARIOS, "cells": {}, "totals": {}}
    for sc_name, sc in SCENARIOS.items():
        total = 0.0
        for c in cells:
            est = cell_estimate(c, m[sc["out"]], sc)
            out["cells"].setdefault(os.path.basename(c), {})[sc_name] = est
            total += est["usd"]
        out["totals"][sc_name] = {"cells": len(cells), "usd": round(total, 2),
                                  "usd_per_cell_max": max((out["cells"][os.path.basename(c)][sc_name]["usd"] for c in cells), default=None)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
