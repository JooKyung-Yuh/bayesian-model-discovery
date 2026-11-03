"""Make one NewtonBench task's sealed manifest from its registration-v1 dry run (or `build`: the draft `main()` computes
first), and check it with the shared verifier (unit ICL-NG, 2026-10-09; 0 USD, nothing is sent). The sealed file is the
draft with status "sealed": the draft binds every file of tools/icl_baseline (the shared verifier requires all of them),
the engine modules the run executes, the depth manifest, the task's vendored law files and the registration.
Checks before writing: the draft names the benchmark, the task and this tree's registered settings for it (30 rounds,
n = 4..34, no official layer, 200 USD); every bound file still has the draft's sha256; the registration bound is this
tree's v1 registration; the draft's six start-forecast digests are recomputed and equal. After writing:
`transport.verify_seal` with the settings and required inputs `main()` hands to `live_client`.
Usage: seal_ng.py <task id> <dry-run output folder | build> <new sealed manifest path>"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT)]
from tools.icl_baseline.ng.run_ng_task import (
    AUTOSCILAB,
    ENGINE_TREE,
    REGISTRATION,
    TASKS,
)

# The engine tree first: its `adapters` and `bayes_model_discovery` packages, never this checkout's.
sys.path[:0] = [f"{ENGINE_TREE}/src", ENGINE_TREE, str(ROOT), AUTOSCILAB]

from tools.icl_baseline import transport
from tools.icl_baseline.ng import icl_ng as tool


def seal(task, dry, target):
    tool.check_engine()
    plan = tool.REGISTERED_PLAN
    if Path(target).exists():
        raise FileExistsError(target)
    sources = tool.task_sources(task)
    built = tool.seal_draft(task, 32000, sources, REGISTRATION, plan)
    draft = built if str(dry) == "build" else json.loads((Path(dry) / "seal_draft.json").read_text())
    expected = tool.ng_settings(task, plan, 32000)
    if draft.get("status") != "draft_not_execution_authority" or draft.get("task") != task \
            or draft.get("benchmark") != tool.BENCHMARK:
        raise SystemExit("not this task's draft")
    if draft["settings"] != expected:
        raise SystemExit("the draft's settings are not this tree's registration-v1 settings")
    if str(REGISTRATION.resolve()) not in draft["files"]:
        raise SystemExit("the draft does not bind this tree's v1 registration")
    if draft["d0_prediction_requests"] != built["d0_prediction_requests"] \
            or draft["history_assignment"] != built["history_assignment"]:
        raise SystemExit("the draft's start-forecast digests or replicates differ from a fresh build")
    changed = [path for path, digest in draft["files"].items() if transport.sha(path) != digest]
    if changed:
        raise SystemExit(f"hash changed since the dry run: {changed}")
    sealed = {**draft, "status": "sealed"}
    with Path(target).open("x") as stream:
        stream.write(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    transport.verify_seal(target, expected, tool.dependencies(task) + [REGISTRATION])
    return {"sealed_manifest": str(Path(target).resolve()), "sha256": transport.sha(target), "task": task,
            "benchmark": tool.BENCHMARK, "rounds": expected["rounds"], "checkpoints": expected["checkpoints"],
            "official_checkpoints": expected["official_checkpoints"], "cap_usd": expected["cap_usd"],
            "files": len(sealed["files"]), "d0_prediction_requests": len(sealed["d0_prediction_requests"]),
            "verify_seal": "passed"}


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in TASKS:
        raise SystemExit("usage: seal_ng.py <task id> <dry-run output folder | build> <new sealed manifest path>")
    print(json.dumps(seal(*sys.argv[1:4]), indent=1))
