"""0 USD check of a stub or mock-transport DP-Agent runs folder (2026-10-05). For every cell of the given mode
it checks the wrapper's own records against what the scripted client must produce, prints PASS/FAIL per check and
'DRY CHECK OK' when all pass (exit 0; exit 1 otherwise). It reads no score: the dry law is a fixed script.
  stub           (compliant script): experiments_run = b, nothing refused, law at round b + 1, b + 2 calls
  mock-transport (greedy script):    experiments_run = b, one 'budget_used' refusal, law at round b + 2, b + 3 calls;
                                     every request went to /v1/messages with exactly model, max_tokens, messages, system,
                                     the agent asked for 16000 tokens (v2 raise over DP's 8192) and the judge for 1024; the mock reply carries a thinking block first, so a passing run proves the reorder, and the recorded usage equals
                                     what the mock server sent.
With a third argument `current_law` (the every-round variant, dp_agent_throttled.py item 11) the cells checked are the
`_cl` ones, and additionally: <stem>.current_laws.json exists with rows b = 0..b, the scripted agent stated the law in every
experiment round but the third (b = 2 is `missing`, the others `stated`), every stated row b >= 1 has DP-fitted constants
with a finite loss, and b = budget is the final law. Usage: check_dry_dp.py <runs folder> <stub|mock-transport> [current_law]"""
import glob
import json
import os
import sys

DP_HEAD = "33b7fa9df96de9c35744efd181ca7e5a8dd60ad5"
SYSTEM_PROMPT_SHA256 = "d1347179c91992f1a1cc47fc8bf74517c653c384f408682f02c683f232efa74d"   # all five worlds, len 9148
folder, mode = sys.argv[1], sys.argv[2]
protocol = sys.argv[3] if len(sys.argv) > 3 else "released"
suffix = ("_cl" if protocol == "current_law" else "") + "_" + mode.replace("-", "")
failed = 0


def expect(name, cond, detail=""):
    global failed
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    failed += 0 if cond else 1


cells = sorted(glob.glob(os.path.join(folder, f"*{suffix}.protocol.json")))
expect("cells found", len(cells) > 0, folder)
for p in cells:
    stem = p[:-len(".protocol.json")]
    name = os.path.basename(stem)
    pr = json.load(open(p))
    wr = json.load(open(stem + ".wrapper.json"))
    us = json.load(open(stem + ".usage.json"))
    st = json.load(open(stem + ".json"))
    b = int(pr["budget"])
    final_round = next((r["round"] for r in st["rounds"] if r.get("action") == "final_law"), None)
    expect(f"{name} status", pr["status"] == "law_scored_by_dp", pr["status"])
    expect(f"{name} results file", os.path.exists(stem + ".results.json"))
    expect(f"{name} experiments_run = b", pr["experiments_run"] == b, pr["experiments_run"])
    expect(f"{name} rounds", (wr["max_rounds"], wr["min_rounds"]) == (b + 2, b + 1), (wr["max_rounds"], wr["min_rounds"]))
    expect(f"{name} DP head", wr["dp_head"] == DP_HEAD, wr["dp_head"])
    expect(f"{name} system prompt", wr["system_prompt"]["sha256"] == SYSTEM_PROMPT_SHA256 and wr["system_prompt"]["len"] == 9148,
           wr["system_prompt"])
    expect(f"{name} model", wr["model"] == "claude-opus-5-5" and wr["judge"] == "claude-haiku-4-5-20251001")
    expect(f"{name} protocol recorded", wr.get("protocol", "released") == protocol, wr.get("protocol"))
    if protocol == "current_law":
        expect(f"{name} constraint carries the current_law sentence", "<current_law>" in wr["constraint_line"])
        expect(f"{name} current_laws file", os.path.exists(stem + ".current_laws.json"))
        if os.path.exists(stem + ".current_laws.json"):
            cl = json.load(open(stem + ".current_laws.json"))
            rows = {r["b"]: r for r in cl["rows"]}
            expect(f"{name} current_laws rows 0..b", sorted(rows) == list(range(b + 1)), sorted(rows))
            expect(f"{name} stated in every experiment round but the third",
                   all(rows[k]["status"] == ("missing" if k == 2 else "stated") for k in range(b)),
                   {k: rows[k]["status"] for k in range(b)})
            expect(f"{name} final law row", rows[b]["status"] == "final" and bool(rows[b]["law"]), rows[b]["status"])
            expect(f"{name} final row refitted by the harvest rule on all b experiments",
                   rows[b].get("fit") == "harvest_all" and rows[b].get("n_training") == b and rows[b].get("executed_total") == b,
                   (rows[b].get("fit"), rows[b].get("n_training"), rows[b].get("executed_total")))
            expect(f"{name} DP's own final constants recorded beside", "dp_final_fitted_params" in rows[b] and "dp_final_fit" in rows[b])
            fitted = [rows[k] for k in range(1, b) if rows[k]["status"] == "stated"]
            expect(f"{name} stated laws fitted on the executed data so far (b = executed experiments)",
                   all(r.get("fit") == "harvest_all" and r.get("loss_after") is not None and r.get("n_training") == k
                       for k, r in zip([k for k in range(1, b) if rows[k]["status"] == "stated"], fitted)),
                   [(r.get("fit"), r.get("n_training"), r.get("error")) for r in fitted][:4])
            expect(f"{name} schema v2", cl.get("schema") == "dp_current_laws_v2", cl.get("schema"))
            expect(f"{name} b = 0 law kept at its initial constants", rows[0].get("fit", "").startswith("none"), rows[0].get("fit"))
            expect(f"{name} no fit errors", cl["fit_errors"] == 0, cl["fit_errors"])
    else:
        expect(f"{name} no current_laws file in the released protocol", not os.path.exists(stem + ".current_laws.json"))
    refused = [r["reason"] for r in pr["refused_requests"]]
    if mode == "stub":
        expect(f"{name} refused none", refused == [], refused)
        expect(f"{name} law at round b + 1", final_round == b + 1, final_round)
        expect(f"{name} calls b + 2", us["calls"] == b + 2, us["calls"])
    else:
        expect(f"{name} one budget refusal", refused == ["budget_used"], refused)
        expect(f"{name} law at round b + 2", final_round == b + 2, final_round)
        expect(f"{name} calls b + 3", us["calls"] == b + 3, us["calls"])
        rows = [json.loads(line) for line in open(stem + ".mock_requests.jsonl")]
        expect(f"{name} mock requests = calls", len(rows) == us["calls"], len(rows))
        expect(f"{name} endpoint", all(r["url"] == "https://api.anthropic.com/v1/messages" and r["method"] == "POST" for r in rows))
        expect(f"{name} body fields", all(r["body_keys"] == ["max_tokens", "messages", "model", "system"] for r in rows),
               sorted({tuple(r["body_keys"]) for r in rows}))
        agent = [r for r in rows if r["model"] == "claude-opus-5-5"]
        judge = [r for r in rows if r["model"] == "claude-haiku-4-5-20251001"]
        expect(f"{name} agent and judge calls", len(agent) == b + 2 and len(judge) == 1, (len(agent), len(judge)))
        expect(f"{name} max_tokens", {r["max_tokens"] for r in agent} == {16000} and {r["max_tokens"] for r in judge} == {1024})
        sin = sum(r["usage"]["input_tokens"] for r in rows)
        sout = sum(r["usage"]["output_tokens"] for r in rows)
        expect(f"{name} usage capture", (sin, sout) == (us["input_tokens"], us["output_tokens"]),
               ((sin, sout), (us["input_tokens"], us["output_tokens"])))
    print(f"INFO {name} anthropic {wr['anthropic_version']} httpx {wr['httpx_version']} python {wr['python']} host {wr['host']} "
          f"DP mean_pos_error {json.load(open(stem + '.results.json'))['evaluation'].get('mean_pos_error')}")
print("DRY CHECK OK" if failed == 0 else f"DRY CHECK FAILED ({failed})")
sys.exit(0 if failed == 0 else 1)
