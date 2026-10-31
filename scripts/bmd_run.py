#!/usr/bin/env python3
"""Run one cell of any benchmark through its own runner.

    python scripts/bmd_run.py <synthcount|chembench|boxinggym|neuronbench|forcebench> --output DIR
        [--arm joint_eig|model_eig|parameter_eig|epig|random] [--seed N] [--rounds N] [--profile dev|paper]
        [--task NAME] [--proposer mock|anthropic] [--allow-paid-api] [--dotenv FILE] [--dry-run] [-- runner options]

A thin shell over the existing runners; it adds no science and checks no commit (each runner records its own code
sha256 in config.json). It translates the arm into the runner's own cell code and design policy, puts the pinned
external checkout on PYTHONPATH from the variables scripts/provision.sh exports, and runs:

    synthcount   python -m bayes_model_discovery.experimental.core_count       (SynthCount, seed = world)
    chembench    python -m bayes_model_discovery.experimental.chembench_core    (BMD_AUTOSCILAB)
    boxinggym    python -m bayes_model_discovery.experimental.boxinggym_core   (BMD_BOXINGGYM_SRC)
    neuronbench  python -m bayes_model_discovery.experimental.neuron_core      (NEURONBENCH_SRC)
    forcebench   bash tools/fb1_launch_u/run_cell.sh   (FB_PY, FB_DP; the ForceBench kit, which owns its FB1_* settings)

Anything after `--` goes to the runner unchanged. A paid run needs --proposer anthropic, --allow-paid-api and
--dotenv: only the key names the runners read are taken from that file, into the child's environment, and no value
is ever printed. The defaults (mock proposer, dev profile, 2 rounds) cost nothing.
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("joint_eig", "model_eig", "parameter_eig", "epig", "random")
KEY_NAMES = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
PROVISION = "run scripts/provision.sh (see its --with option) and source the environment it prints"

# (cell, extra runner options) per arm; None = this runner does not offer that arm here.
# The cell code is <structure><family><design>: design 1 = EIG, 2 = EPIG, 0 = random.
ARM_TABLE = {
    "synthcount": {"joint_eig": ("111", []), "model_eig": ("111", ["--design-policy", "structure_eig"]),
                   "parameter_eig": None, "epig": ("112", ["--design-policy", "epig"]), "random": ("110", [])},
    # ChemBench's EIG arms climb their criterion (--design-search ascent); EPIG is not used at d >= 3.
    "chembench": {"joint_eig": ("101", ["--design-search", "ascent"]),
                  "model_eig": ("101", ["--design-search", "ascent", "--design-policy", "structure_eig"]),
                  "parameter_eig": ("101", ["--design-search", "ascent", "--design-policy", "parameter_eig"]),
                  "epig": None, "random": ("100", [])},
    "boxinggym": {"joint_eig": ("111", []), "model_eig": ("111", ["--design-policy", "structure_eig"]),
                  "parameter_eig": ("111", ["--design-policy", "parameter_eig"]),
                  "epig": ("112", ["--design-policy", "epig"]), "random": ("110", [])},
    "neuronbench": {"joint_eig": ("111", []), "model_eig": ("111", ["--design-policy", "structure_eig"]),
                    "parameter_eig": ("111", ["--design-policy", "parameter_eig"]), "epig": ("112", []),
                    "random": ("110", [])},
}
MODULES = {"synthcount": "core_count", "chembench": "chembench_core", "boxinggym": "boxinggym_core",
           "neuronbench": "neuron_core"}
# The external checkout each benchmark needs, put on PYTHONPATH: (variable, what it must hold).
EXTERNAL = {
    "chembench": ("BMD_AUTOSCILAB", "the LLM-AutoSciLab checkout @ 467fa18"),
    "boxinggym": ("BMD_BOXINGGYM_SRC", "the src/ folder of the BoxingGym checkout @ b43e38cb"),
    "neuronbench": ("NEURONBENCH_SRC", "the neuronbench checkout @ c354622"),
}
FORCEBENCH_WORLDS = ("gravity", "yukawa", "coulomb_easy", "oscillator", "extra_dimensions")


def refuse(message):
    """Print `message` to stderr and exit with status 2."""
    print(f"bmd_run: {message}", file=sys.stderr)
    sys.exit(2)


def read_keys(path):
    """The key names the runners read, from a dotenv file; values never leave this function except into the child."""
    from dotenv import dotenv_values

    values = dotenv_values(path)
    found = {name: values[name] for name in KEY_NAMES if values.get(name)}
    if not found:
        refuse(f"{path} sets none of {', '.join(KEY_NAMES)}")
    print(f"bmd_run: dotenv {path}: read {', '.join(sorted(found))}")
    return found


def external_path(benchmark):
    """The external checkout folder `benchmark` needs, from its environment variable; refuses when unset or missing."""
    name, what = EXTERNAL[benchmark]
    value = os.environ.get(name, "")
    if not value or not Path(value).is_dir():
        refuse(f"{benchmark} needs {name} = {what}; {PROVISION}")
    return value


def build(args, passthrough):
    """(argv, environment additions, working directory) for the benchmark's runner."""
    if args.benchmark == "forcebench":
        world = args.task or "gravity"
        if world not in FORCEBENCH_WORLDS:
            refuse(f"forcebench --task must be one of {', '.join(FORCEBENCH_WORLDS)}")
        for name in ("FB_PY", "FB_DP"):
            if not os.environ.get(name):
                refuse(f"forcebench needs {name} (the jax venv python and the DiscoverPhysics checkout); {PROVISION}")
        if args.rounds is not None or args.profile != "dev" or passthrough:
            refuse("forcebench takes its budget from the kit (FB_BUDGET 16|30, FB_MOCK_T_EXTRA for mock); "
                   "--rounds, --profile and runner options do not apply")
        mode = "paid" if args.proposer == "anthropic" else "mock"
        extra = {"FB_DOTENV": str(Path(args.dotenv).resolve())} if mode == "paid" else {}
        return (["bash", str(ROOT / "tools/fb1_launch_u/run_cell.sh"), world, str(args.seed), args.arm,
                 str(Path(args.output).resolve()), mode], extra, ROOT)
    choice = ARM_TABLE[args.benchmark][args.arm]
    if choice is None:
        refuse(f"{args.benchmark} does not offer the {args.arm} arm in this tree")
    cell, arm_options = choice
    argv = [sys.executable, "-m", f"bayes_model_discovery.experimental.{MODULES[args.benchmark]}",
            "--output", str(Path(args.output).resolve()), "--seed", str(args.seed), "--cell", cell, *arm_options,
            "--rounds", str(2 if args.rounds is None else args.rounds), "--proposer", args.proposer]
    if args.allow_paid_api:
        argv.append("--allow-paid-api")
    path = [str(ROOT / "src"), str(ROOT)]
    if args.benchmark == "synthcount":
        argv += ["--profile", args.profile]
    elif args.benchmark == "chembench":
        path.append(external_path("chembench"))
        manifest = ROOT / ("manifests/chembench_paper_manifest.json" if args.profile == "paper"
                           else "manifests/chembench_dev_manifest.json")
        argv += ["--profile", args.profile, "--manifest", str(manifest), "--noise", "0.01"]
        if args.task:
            argv += ["--task", args.task]
        elif args.profile == "paper":
            refuse("chembench --profile paper needs --task <task id from manifests/chembench_paper_manifest.json>")
    elif args.benchmark == "boxinggym":
        path.append(external_path("boxinggym"))
        argv += ["--profile", args.profile, "--task", args.task or "death"]
    else:
        path.append(external_path("neuronbench"))
        profile = "paper_neuron" if args.profile == "paper" else "dev"
        argv += ["--profile", profile, "--world", args.task or "h_sag"]
    environment = {"PYTHONPATH": os.pathsep.join(path + [p for p in [os.environ.get("PYTHONPATH", "")] if p])}
    return argv + passthrough, environment, ROOT


def main(argv=None):
    """Parse the command line, build the runner command and run it (or print it with --dry-run); returns the exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("benchmark", choices=(*MODULES, "forcebench"))
    parser.add_argument("--output", required=True, help="a new folder for the run")
    parser.add_argument("--arm", choices=ARMS, default="joint_eig")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--rounds", type=int, default=None, help="default 2 (forcebench: the kit's budget)")
    parser.add_argument("--profile", choices=("dev", "paper"), default="dev")
    parser.add_argument("--task", default=None, help="task (ChemBench, BoxingGym), world (NeuronBench, ForceBench)")
    parser.add_argument("--proposer", choices=("mock", "anthropic"), default="mock")
    parser.add_argument("--allow-paid-api", action="store_true")
    parser.add_argument("--dotenv", default=None, help="paid runs: a dotenv file with the API key")
    parser.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = parser.parse_args(own)
    if args.proposer == "anthropic" and not (args.allow_paid_api and args.dotenv):
        refuse("a paid run needs --proposer anthropic, --allow-paid-api and --dotenv FILE")
    if Path(args.output).exists():
        refuse(f"{args.output} exists; give a new folder")
    command, additions, cwd = build(args, passthrough)
    print("bmd_run: " + " ".join(shlex.quote(part) for part in command))
    if args.dry_run:
        return 0
    environment = {**os.environ, **additions, "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop("FORCE_COLOR", None)
    environment.setdefault("PYTENSOR_FLAGS", f"base_compiledir={Path(args.output).resolve()}.ptc")
    if args.proposer == "anthropic" and args.benchmark != "forcebench":
        environment.update(read_keys(args.dotenv))
    return subprocess.run(command, cwd=cwd, env=environment, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
