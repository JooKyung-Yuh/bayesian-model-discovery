"""0 USD checks of dp_current_law.harvest on a SYNTHETIC run log (no recorded reply is used): the b label counts executed
experiments only, the latest statement wins a shared b, and the final row is refitted by the same rule as every other row.

  <DP venv python> -B test_dp_current_law.py      (needs the DiscoverPhysics checkout on sys.path: DP_ROOT or the the development host default)
"""
import json
import os
import sys
import tempfile

KIT = os.path.dirname(os.path.abspath(__file__))
DP = os.environ.get("DP_ROOT", "<development path>")
sys.path[:0] = [KIT, f"{DP}/ScienceAgent", f"{DP}/PhysicsSchool"]
sys.dont_write_bytecode = True

import dp_current_law as H  # noqa: E402

LAW = '''def discovered_law(pos1, pos2, p1, p2, velocity2, duration, k=1.0):
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
TIMES = [0.5, 1.0, 1.5]


def synthetic_output(p1, k_true=0.4):
    """A trajectory of the law above with k_true, so the fit has something to recover (synthetic, not a recorded run)."""
    law = H.__dict__.get("_test_law")
    if law is None:
        ns = {}
        exec(LAW, ns)
        law = H._test_law = ns["discovered_law"]
    pos2, vel2 = [], []
    for t in TIMES:
        x, v = law([0.0, 0.0], [1.3383, -1.3851], p1, 1.0, [0.486, -1.9134], t, k=k_true)
        pos2.append(x)
        vel2.append(v)
    return {"measurement_times": TIMES, "pos1": [[0.0, 0.0]] * len(TIMES), "pos2": pos2, "velocity1": [[0.0, 0.0]] * len(TIMES), "velocity2": vel2}


def exp_input(p1):
    return {"p1": p1, "p2": 1.0, "pos2": [1.3383, -1.3851], "velocity2": [0.486, -1.9134], "measurement_times": TIMES}


def reply(law=True, experiment=True, p1=1.0):
    text = ""
    if experiment:
        text += "<run_experiment>" + json.dumps([exp_input(p1)]) + "</run_experiment>\n"
    if law:
        text += "<current_law>\n" + LAW + "</current_law>"
    return text


def run_log(budget=4):
    rounds = [
        # round 1: experiment executed + law                      -> law at b = 0 (no data), experiments after: 1
        {"round": 1, "action": "experiment", "llm_reply": reply(p1=1.0), "experiment_input": [exp_input(1.0)],
         "experiment_output": [synthetic_output(1.0)], "experiment_error": None},
        # round 2: experiment REFUSED (budget guard) + law         -> law at b = 1 (nothing executed here)
        {"round": 2, "action": "experiment", "llm_reply": reply(p1=2.0), "experiment_input": None, "experiment_output": None,
         "experiment_error": "budget_used"},
        # round 3: no tag at all (no experiment, no law)           -> adds nothing
        {"round": 3, "action": "no_tag", "llm_reply": "I will think.", "experiment_input": None, "experiment_output": None,
         "experiment_error": None},
        # round 4: experiment executed + law                       -> law at b = 1 again (latest statement wins), after: 2
        {"round": 4, "action": "experiment", "llm_reply": reply(p1=2.0), "experiment_input": [exp_input(2.0)],
         "experiment_output": [synthetic_output(2.0)], "experiment_error": None},
        # round 5: experiment executed, NO law                     -> b = 2 missing, after: 3
        {"round": 5, "action": "experiment", "llm_reply": reply(law=False, p1=3.0), "experiment_input": [exp_input(3.0)],
         "experiment_output": [synthetic_output(3.0)], "experiment_error": None},
        # round 6: experiment executed + law                       -> law at b = 3, after: 4
        {"round": 6, "action": "experiment", "llm_reply": reply(p1=4.0), "experiment_input": [exp_input(4.0)],
         "experiment_output": [synthetic_output(4.0)], "experiment_error": None},
        # round 7: final law                                       -> b = 4 = B
        {"round": 7, "action": "final_law", "llm_reply": "<final_law>\n" + LAW + "</final_law>", "experiment_input": None,
         "experiment_output": None, "experiment_error": None, "final_law": LAW},
    ]
    return {"world": "coulomb_easy", "max_rounds": budget + 2, "mission": "synthetic", "rounds": rounds, "final_law": LAW,
            "evaluation": {"fit": {"error": "no_training_trajectories"}}}


def main():
    tmp = tempfile.mkdtemp(prefix="dp_cl_test_")
    run_json = os.path.join(tmp, "synthetic_thr_b4_s0_cl.json")
    json.dump(run_log(), open(run_json, "w"))
    json.dump({"world": "coulomb_easy", "law": LAW, "evaluation": {"fit": {"error": "no_training_trajectories"}}},
              open(os.path.join(tmp, "synthetic_thr_b4_s0_cl.results.json"), "w"))
    out = os.path.join(tmp, "synthetic_thr_b4_s0_cl.current_laws.json")
    summary = H.harvest(run_json, out, "coulomb_easy", 4)
    cl = json.load(open(out))
    rows = {r["b"]: r for r in cl["rows"]}
    failed = 0

    def expect(name, cond, detail=""):
        nonlocal failed
        print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
        failed += 0 if cond else 1

    expect("executed experiments counted (4 of 5 experiment rounds; the refused one adds none)", cl["executed_total"] == 4, cl["executed_total"])
    expect("rows b = 0..4", sorted(rows) == [0, 1, 2, 3, 4], sorted(rows))
    expect("b = 0: the round-1 law at its initial constants", rows[0]["status"] == "stated" and rows[0]["round"] == 1
           and rows[0]["fit"].startswith("none") and rows[0]["fitted_params"] == {"k": 1.0}, rows[0])
    expect("b = 1: the refused round 2 and the executed round 4 both state at b = 1; the latest (round 4) wins, 2 statements",
           rows[1]["status"] == "stated" and rows[1]["round"] == 4 and rows[1]["statements"] == 2, {k: rows[1].get(k) for k in ("round", "statements")})
    expect("b = 1 fitted on exactly 1 executed experiment", rows[1]["fit"] == "harvest_all" and rows[1]["n_training"] == 1, (rows[1].get("fit"), rows[1].get("n_training")))
    expect("b = 2 missing (round 5 stated no law)", rows[2]["status"] == "missing", rows[2]["status"])
    expect("b = 3 fitted on 3 executed experiments", rows[3]["status"] == "stated" and rows[3]["n_training"] == 3 and rows[3]["round"] == 6, rows[3].get("n_training"))
    expect("the no_tag round 3 shifted nothing", [e for e in cl["rounds_seen"] if e["round"] == 3][0]["executed_here"] == 0)
    expect("b = B = 4 is the final law refitted by the SAME rule on all 4 experiments (harvest_all, n_training 4)",
           rows[4]["status"] == "final" and rows[4]["fit"] == "harvest_all" and rows[4]["n_training"] == 4, {k: rows[4].get(k) for k in ("status", "fit", "n_training")})
    expect("DP's own final constants recorded beside (here: DP did not fit)", rows[4]["dp_final_fitted_params"] == {} and "did not fit" in rows[4]["dp_final_fit"], rows[4].get("dp_final_fit"))
    # no artificial jump: the final row's constants equal a direct fit of the final law on all executed experiments
    direct = H.fit_constants(LAW, "coulomb_easy", H.training_from_rounds(run_log()["rounds"], 10 ** 9))
    expect("final-row constants = direct harvest fit on all experiments", rows[4]["fitted_params"] == direct["fitted_params"], (rows[4]["fitted_params"], direct["fitted_params"]))
    ks = [rows[b]["fitted_params"]["k"] for b in (1, 3, 4)]
    expect("the fitted k approaches the synthetic truth 0.4 (recoverable)", all(abs(k - 0.4) < 0.05 for k in ks), ks)
    expect("no fit errors", cl["fit_errors"] == 0, cl["fit_errors"])
    print("RESULT", "PASS" if not failed else f"FAIL ({failed})", json.dumps(summary))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
