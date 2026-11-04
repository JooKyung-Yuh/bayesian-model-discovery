"""DP-Agent (DiscoverPhysics ScienceAgent) under MDA's throttled protocol: one experiment per round, at most b experiments.

2026-10-05, for the Opus 5.5 baseline. Derived from
an earlier wrapper (013 v4b, the wrapper of the
2026-09-06/07 Opus-4.7 runs). The DiscoverPhysics checkout is not edited: run_discovery.main() runs as it is, with the
patches below applied inside this process only.

Differences from 013 (each one is recorded in the cell's wrapper.json):
  1. --dp-root names the DiscoverPhysics checkout (013 hard-coded a host path).
  2. --plot is not passed (the plot is not a result; the development host's venv_dp has no matplotlib).
  3. --engine is the engine our ForceBench runner scores against (FB1_FIXED_IC_MANIFEST_v5: gravity and yukawa field,
     the other three nbody); 013 used run_discovery's default engine.
  4. Rounds: max_rounds = b + 2 and min_rounds = b + 1. DP warns at round max_rounds - 1 that the next reply must be a
     final law. With 013's max_rounds = b + 1 that warning fell on round b: 14 of the 91 throttled 013 runs with an
     experiment count ran b experiments, none of the 24 at b = 8 (13 of them ran 7). Here rounds 1..b are experiment
     rounds (DP's own min_rounds rule refuses a final law there), round b + 1 carries DP's warning, and round b + 2
     forces the law.
  5. Budget guard on the agent's executor: a <run_experiment> must hold exactly one experiment, and at most b
     experiments run in total. A refused request runs nothing; DP reports it to the agent as "Error running experiment".
  6. The constraint line is 013's v2 text plus one sentence on when a final law is accepted.
  7. Default model claude-opus-5-5. DP's llm_client is unchanged (no temperature, no thinking parameter). v2 (2026-10-08, main):
     Opus 5.5 answers with a thinking block first even when thinking is not requested, and the thinking tokens count against
     max_tokens; DP's client reads content[0].text and crashed on the first paid call (canary 01:33 KST, 0.0195 USD). The
     wrapper now raises the agent's max_tokens to 16,000 (the judge keeps DP's 1,024; same decision as the AutoSciLab kit)
     and moves the first text block to the front of the reply; non-text blocks and max_tokens stops are counted in usage.
  8. Usage is written after every call; the cell refuses further calls once its estimated spend reaches --cell-cap-usd.
  9. --mode stub (scripted client, no SDK) and --mode mock-transport (the installed anthropic SDK over an
     httpx.MockTransport, no network) are the 0 USD checks; only --mode paid can reach the API.
 10. The output folder must lie outside the DiscoverPhysics checkout, and an existing cell is never overwritten.

Usage: dp_agent_throttled.py --world W --budget B --seed S --outdir DIR --mode {stub,mock-transport,paid}
                             [--dp-root DIR] [--cell-cap-usd X] [--protocol released|current_law]

 11. --protocol current_law (2026-10-08, "every round" variant; default released = everything above, byte for byte): the
     constraint line also asks the agent to state its current best law inside <current_law>...</current_law> in every
     experiment round (same format as <final_law>). DP's agent ignores the tag (it reads <run_experiment>, <run_mse_fit>
     and <final_law> only), nothing is fed back, and after the run dp_current_law.py harvests the tags from the run log,
     fits each law's declared constants with DP's own fitting code on the experiments the agent had seen when it wrote
     it (b = round - 1), and writes <stem>.current_laws.json. Cells of this protocol carry `_cl` in their stem.
"""
import argparse
import atexit
import hashlib
import json
import math
import os
import platform
import signal
import socket
import subprocess
import sys

ELICER_DP = "<development path>"
# scripts/run_benchmark.py:41 in DiscoverPhysics (the table 013 uses; these five equal our runner's var_gt)
WORLD_VARS = {"gravity": 4.283, "yukawa": 5.677, "coulomb_easy": 4.244, "oscillator": 6.332, "extra_dimensions": 4.248}
# Copied from FB1_FIXED_IC_MANIFEST_v5.json (sha256 below; frozen tree main_verify_iclf_5fd4d47,
# tools/fb1_snapshot/frozen_inputs/) so the kit needs no engine tree: the engine and the fixed start state per world.
IC_MANIFEST_SHA256 = "63067f1346ccfced16db0ba0793db4b3e7f6d0f266e44b612632afffc7de9ce9"
IC = {
    "gravity": {"engine": "field", "pos2": [-2.6731, -6.0141], "velocity2": [-0.1627, 1.9507]},
    "yukawa": {"engine": "field", "pos2": [-3.3946, -1.8965], "velocity2": [0.2569, 0.2787]},
    "coulomb_easy": {"engine": "nbody", "pos2": [1.3383, -1.3851], "velocity2": [0.486, -1.9134]},
    "oscillator": {"engine": "nbody", "pos2": [-2.6731, -6.0141], "velocity2": [-0.1627, 1.9507]},
    "extra_dimensions": {"engine": "nbody", "pos2": [3.1836, -0.5577], "velocity2": [-1.6812, 0.346]},
}
CONSTRAINT_TMPL = ("Protocol constraint: each <run_experiment> must contain exactly ONE experiment (a JSON array of length 1); "
                   "you have {b} experiment round(s) in total, after which you must submit <final_law>. "
                   "A <final_law> is accepted from round {b1} on.")
AGENT_MAX_TOKENS = 16000   # v2: room for Opus 5.5 thinking tokens; DP asked for 8192
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-opus-4-7": (15.0, 75.0), "claude-haiku-4-5-20251001": (1.0, 5.0)}
TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")

ap = argparse.ArgumentParser()
ap.add_argument("--world", required=True, choices=sorted(WORLD_VARS))
ap.add_argument("--budget", type=int, required=True, help="b = the most experiments the agent may run, one per round")
ap.add_argument("--seed", type=int, default=0, help="DP's --noise-seed")
ap.add_argument("--model", default="claude-opus-5-5")
ap.add_argument("--judge", default="claude-haiku-4-5-20251001")
ap.add_argument("--outdir", required=True)
ap.add_argument("--dp-root", default=ELICER_DP)
ap.add_argument("--mode", required=True, choices=("stub", "mock-transport", "paid"))
ap.add_argument("--cell-cap-usd", type=float, default=3.0)
ap.add_argument("--noise-frac", type=float, default=0.05)
ap.add_argument("--protocol", choices=("released", "current_law"), default="released",
                help="current_law: ask for the current best law in <current_law> every experiment round (item 11)")
a = ap.parse_args()
if a.budget < 1:
    sys.exit("--budget must be at least 1")
DP = os.path.realpath(a.dp_root)
OUTDIR = os.path.realpath(a.outdir)
if os.path.commonpath([OUTDIR, DP]) == DP:
    sys.exit(f"--outdir must be outside the DiscoverPhysics checkout {DP}")
os.makedirs(OUTDIR, exist_ok=True)
B = a.budget
MAX_ROUNDS, MIN_ROUNDS = B + 2, B + 1
CONSTRAINT = CONSTRAINT_TMPL.format(b=B, b1=B + 1)
CURRENT_LAW_SENTENCE = ("In every experiment round, also state your current best guess of the law, as a complete Python "
                        "function in the same format as <final_law> (with fit_parameters() if it has constants), inside "
                        "<current_law> ... </current_law> after your <run_experiment>; it is only recorded, never evaluated "
                        "for you, and it does not end the mission.")
if a.protocol == "current_law":
    CONSTRAINT = CONSTRAINT + " " + CURRENT_LAW_SENTENCE
ENGINE = IC[a.world]["engine"]
SIGMA = a.noise_frac * math.sqrt(WORLD_VARS[a.world])
STEM = (f"{a.world}_thr_b{B}_s{a.seed}" + ("_cl" if a.protocol == "current_law" else "")
        + ("" if a.mode == "paid" else "_" + a.mode.replace("-", "")))


def OUT(suffix):
    return os.path.join(OUTDIR, STEM + suffix)


if os.path.exists(OUT(".protocol.json")) or os.path.exists(OUT(".results.json")):
    sys.exit(f"cell exists: {OUT('.protocol.json')}")
if a.mode == "paid":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("--mode paid needs ANTHROPIC_API_KEY in the environment")
else:
    os.environ["ANTHROPIC_API_KEY"] = f"{a.mode}-no-network"   # DP's client refuses an empty key; never a real one here


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def git_head(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:   # recorded, not fatal: the brief checks the head separately
        return f"unavailable: {type(e).__name__}"


KIT = os.path.dirname(os.path.realpath(__file__))
sys.path[:0] = [KIT, f"{DP}/ScienceAgent", f"{DP}/PhysicsSchool"]
# 013 (f): the system prompt loader resolves 'PhysicsSchool/prompts/*.md' against the CWD, so run from the DP root.
os.chdir(DP)

import anthropic as _an  # noqa: E402
import scienceagent.agent as AG  # noqa: E402
from dp_budget_guard import BudgetGuard  # noqa: E402  (the agent's executor only; the evaluator keeps DP's own)
from scienceagent.worlds import get_world as _gw  # noqa: E402

GUARDS = []
_orig_init = AG.DiscoveryAgent.__init__


def _init(self, *args, **kw):
    m = kw.get("mission") or ""
    kw["mission"] = (m.rstrip() + "\n\n" + CONSTRAINT) if m else CONSTRAINT
    kw["max_rounds"], kw["min_rounds"] = MAX_ROUNDS, MIN_ROUNDS
    _orig_init(self, *args, **kw)
    self.executor = BudgetGuard(self.executor, B)
    GUARDS.append(self.executor)


AG.DiscoveryAgent.__init__ = _init
argv = ["run_discovery.py", "--world", a.world, "--engine", ENGINE, "--model", a.model, "--max-rounds", str(MAX_ROUNDS),
        "--judge-model", a.judge, "--noise-std", f"{SIGMA:.6f}", "--noise-seed", str(a.seed), "--no-trajectory-csv",
        "--store_output", OUT(""), "--output", OUT(".results.json")]

# fail-closed: the system prompt (world instructions + experiment format) must load before any call
_w = _gw(a.world, engine=ENGINE)
_sp = AG._load_system_prompt(_w["system_prompt"], _w["instructions"])
assert "<run_experiment>" in _sp and len(_sp) > 2000, "system prompt not loaded"
try:
    import httpx as _hx
    HTTPX_VERSION = _hx.__version__
except Exception as e:
    HTTPX_VERSION = f"unavailable: {type(e).__name__}"
write_json(OUT(".wrapper.json"), {
    "wrapper": "dp_agent_throttled.py (from 013 v4b)", "wrapper_v2": "2026-10-08: thinking-first reply reorder + agent max_tokens 16000 (Opus 5.5); judge unchanged", "wrapper_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
    "guard_sha256": hashlib.sha256(open(os.path.join(KIT, "dp_budget_guard.py"), "rb").read()).hexdigest(),
    "mode": a.mode, "world": a.world, "engine": ENGINE, "budget": B, "max_rounds": MAX_ROUNDS, "min_rounds": MIN_ROUNDS,
    "seed": a.seed, "model": a.model, "judge": a.judge, "sigma_world": SIGMA, "noise_frac": a.noise_frac,
    "world_var": WORLD_VARS[a.world], "constraint_line": CONSTRAINT, "argv": argv, "cell_cap_usd": a.cell_cap_usd,
    "protocol": a.protocol, "current_law_sentence": (CURRENT_LAW_SENTENCE if a.protocol == "current_law" else None),
    "ic": IC[a.world], "ic_manifest_sha256": IC_MANIFEST_SHA256, "dp_root": DP, "dp_head": git_head(DP),
    "anthropic_version": getattr(_an, "__version__", None), "httpx_version": HTTPX_VERSION,
    "python": platform.python_version(), "host": socket.gethostname(),
    "system_prompt": {"path": _w["system_prompt"], "instructions": _w["instructions"], "len": len(_sp),
                      "sha256": hashlib.sha256(_sp.encode()).hexdigest()},
    "differences_from_013": ["--dp-root", "no --plot", "engine = our runner's IC engine",
                             "max_rounds = b + 2, min_rounds = b + 1", "budget guard: 1 experiment per request, at most b",
                             "constraint line + acceptance sentence", "model claude-opus-5-5",
                             "usage after every call + per-cell spend cap", "stub / mock-transport modes",
                             "outdir outside the checkout, no overwrite"],
    "label": ("DP-Agent (1 expt/round, at most b experiments)" if a.protocol == "released" else
              "DP-Agent - every round (1 expt/round, current law stated every round)")})

USAGE = {"mode": a.mode, "calls": 0, **{f: 0 for f in TOKEN_FIELDS}, "by_model": {}, "per_call": []}


def spend():
    total = 0.0
    for m, v in USAGE["by_model"].items():
        pi, po = PRICES.get(m) or PRICES.get(a.model) or PRICES["claude-opus-5-5"]   # an unknown id is priced as the agent
        total += (v["input_tokens"] * pi + v["output_tokens"] * po) / 1e6
    return total


def dump_usage():
    est = {}
    for m, v in USAGE["by_model"].items():
        pi, po = PRICES.get(m) or PRICES.get(a.model) or PRICES["claude-opus-5-5"]
        est[m] = round((v["input_tokens"] * pi + v["output_tokens"] * po) / 1e6, 6)
    USAGE["usd_estimate_by_model"] = est
    USAGE["usd_estimate_total"] = round(sum(est.values()), 6)
    USAGE["price_note"] = "USD per 1M tokens (input-side incl. cache tokens / output): " + json.dumps(PRICES)
    write_json(OUT(".usage.json"), USAGE)


def record(model, u):
    rec = {"model": model, **{f: int(getattr(u, f, 0) or 0) for f in TOKEN_FIELDS}}
    USAGE["calls"] += 1
    for f in TOKEN_FIELDS:
        USAGE[f] += rec[f]
    bm = USAGE["by_model"].setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
    bm["calls"] += 1
    bm["input_tokens"] += rec["input_tokens"] + rec["cache_creation_input_tokens"] + rec["cache_read_input_tokens"]
    bm["output_tokens"] += rec["output_tokens"]
    USAGE["per_call"].append(rec)
    dump_usage()


def guard_spend():
    if spend() >= a.cell_cap_usd:
        raise RuntimeError(f"cell spend cap reached: estimated {spend():.4f} USD >= {a.cell_cap_usd} USD; no further call")


def used_experiments():
    return GUARDS[0].used if GUARDS else 0


def last_user_text(kw):
    msgs = kw.get("messages") or []
    if not msgs:
        return ""
    c = msgs[-1].get("content", "")
    return c if isinstance(c, str) else " ".join(x.get("text", "") for x in c if isinstance(x, dict))


SCRIPT_LAW = '''def discovered_law(pos1, pos2, p1, p2, velocity2, duration, k=1.0):
    import numpy as np
    x = np.asarray(pos2, float); v = np.asarray(velocity2, float); n = max(1, int(duration / 0.01))
    dt = duration / n
    for _ in range(n):
        r = np.linalg.norm(x) + 1e-9
        acc = -k * p1 * x / r**2 / p2
        v = v + dt * acc; x = x + dt * v
    return x.tolist(), v.tolist()

def fit_parameters():
    return {"k": {"init": 1.0, "bounds": [0.01, 100.0]}}
'''
SCRIPT_COUNTER = {"n": 0}


def scripted_reply(kw, policy):
    """A fixed script, not a model. 'compliant' stops at b experiments; 'greedy' asks for experiments until DP forces
    the law, so the guard's budget refusal is exercised."""
    if kw.get("model") == a.judge:
        return "Score: 5/10\nReasoning: scripted check reply."
    last = last_user_text(kw)
    SCRIPT_COUNTER["n"] += 1
    want_law = "This is your final round" in last or (policy == "compliant" and used_experiments() >= B)
    if want_law:
        return f"<final_law>\n{SCRIPT_LAW}</final_law>\n<explanation>Scripted check: an inverse-distance attraction.</explanation>"
    p1 = 1.0 + (SCRIPT_COUNTER["n"] % 3)
    reply = ("<run_experiment>" + json.dumps([{"p1": p1, "p2": 1.0, "pos2": IC[a.world]["pos2"],
                                               "velocity2": IC[a.world]["velocity2"],
                                               "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5]}]) + "</run_experiment>")
    if a.protocol == "current_law" and SCRIPT_COUNTER["n"] != 3:
        # item 11: the scripted agent states the law in every experiment round except the third, so the harvest's
        # missing-round carry is exercised by the 0 USD checks
        reply += f"\n<current_law>\n{SCRIPT_LAW}</current_law>"
    return reply


if a.mode == "stub":
    from types import SimpleNamespace

    class _StubMessages:
        def create(self, **kw):
            text = scripted_reply(kw, "compliant")
            u = SimpleNamespace(**{f: 0 for f in TOKEN_FIELDS})
            record(kw.get("model"), u)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=u, model=kw.get("model"))

    class _StubAnthropic:
        def __init__(self, *args, **kw):
            self.messages = _StubMessages()

    _an.Anthropic = _StubAnthropic   # the module object llm_client imports inside _anthropic_complete
else:
    _OrigAnthropic = _an.Anthropic
    MOCK = {"rows": []}

    def _mock_handler(request):
        import httpx
        body = json.loads(request.content)
        text = scripted_reply(body, "greedy")
        usage = {"input_tokens": len(request.content) // 4, "output_tokens": max(1, len(text) // 4),
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        row = {"method": request.method, "url": str(request.url), "body_keys": sorted(body), "model": body.get("model"),
               "max_tokens": body.get("max_tokens"), "n_messages": len(body.get("messages") or []),
               "system_chars": len(body.get("system") or ""), "has_api_key_header": "x-api-key" in request.headers,
               "usage": usage}
        MOCK["rows"].append(row)
        with open(OUT(".mock_requests.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        return httpx.Response(200, json={"id": f"msg_mock_{len(MOCK['rows'])}", "type": "message", "role": "assistant",
                                         "model": body.get("model"),
                                         "content": [{"type": "thinking", "thinking": "mock reasoning (v2: exercises the thinking-first reorder)",
                                                      "signature": "mock"}, {"type": "text", "text": text}],
                                         "stop_reason": "end_turn", "stop_sequence": None, "usage": usage})

    class _UsageAnthropic(_OrigAnthropic):
        def __init__(self, *args, **kw):
            if a.mode == "mock-transport":
                import httpx
                kw["http_client"] = httpx.Client(transport=httpx.MockTransport(_mock_handler))
                kw["max_retries"] = 0
            super().__init__(*args, **kw)
            _orig = self.messages.create

            def create(**kw2):
                guard_spend()
                # v2 (2026-10-08): see docstring item 7 - agent max_tokens 16,000 and the first text block moved to the front.
                if kw2.get("model") == a.model and int(kw2.get("max_tokens") or 0) < AGENT_MAX_TOKENS:
                    USAGE["max_tokens_raised"] = USAGE.get("max_tokens_raised", 0) + 1
                    kw2["max_tokens"] = AGENT_MAX_TOKENS
                r = _orig(**kw2)
                u = getattr(r, "usage", None)
                if u is not None:
                    record(getattr(r, "model", None) or kw2.get("model"), u)
                content = list(getattr(r, "content", None) or [])
                texts = [b for b in content if getattr(b, "type", None) == "text"]
                others = [b for b in content if getattr(b, "type", None) != "text"]
                USAGE["non_text_blocks"] = USAGE.get("non_text_blocks", 0) + len(others)
                if getattr(r, "stop_reason", None) == "max_tokens":
                    USAGE["stop_max_tokens"] = USAGE.get("stop_max_tokens", 0) + 1
                dump_usage()
                if not texts:
                    raise RuntimeError(f"reply without a text block (stop_reason={getattr(r, 'stop_reason', None)}, "
                                       f"blocks={[getattr(b, 'type', None) for b in content]})")
                if others:
                    try:
                        r.content = texts + others
                    except Exception:  # noqa: BLE001 - a frozen response object: hand DP a plain copy
                        from types import SimpleNamespace
                        r = SimpleNamespace(content=texts + others, usage=u, model=getattr(r, "model", None),
                                            stop_reason=getattr(r, "stop_reason", None))
                return r
            self.messages.create = create

    _an.Anthropic = _UsageAnthropic

STATE = {"status": "started", "error": None}


def finish():
    dump_usage()
    g = GUARDS[0] if GUARDS else None
    write_json(OUT(".protocol.json"), {
        "mode": a.mode, "world": a.world, "budget": B, "seed": a.seed, "status": STATE["status"], "error": STATE["error"],
        "experiments_run": g.used if g else 0, "refused_requests": g.refused if g else [],
        "results_written": os.path.exists(OUT(".results.json")), "calls": USAGE["calls"],
        "usd_estimate_total": USAGE.get("usd_estimate_total")})


atexit.register(finish)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
dump_usage()
sys.argv = argv
import run_discovery as RD  # noqa: E402

RD.DiscoveryAgent = AG.DiscoveryAgent   # run_discovery imported the class by name
try:
    RD.main()
    STATE["status"] = "law_scored_by_dp" if os.path.exists(OUT(".results.json")) else "no_results_file"
    if a.protocol == "current_law":
        import dp_current_law
        STATE["current_laws"] = dp_current_law.harvest(OUT(".json"), OUT(".current_laws.json"), a.world, B)
except SystemExit as e:
    STATE["status"], STATE["error"] = "system_exit", str(e.code)
    raise
except BaseException as e:
    STATE["status"], STATE["error"] = "exception", f"{type(e).__name__}: {e}"[:1000]
    raise
