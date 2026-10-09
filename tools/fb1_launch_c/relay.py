"""Relay for run_cell.sh (2026-10-03). In paid mode it reads ANTHROPIC_API_KEY from the dotenv file
FB_DOTENV into this process only (never printed), then replaces this process with `FB_PY -B run_gravity.py` in the runner
folder FB_TREE, which inherits it. It needs no
python-dotenv (the dotenv line is parsed here: optional `export`, KEY=value, optional quotes, trailing ` #` comment)."""
import os
import sys

TREE = os.environ["FB_TREE"]
PY = os.environ["FB_PY"]
for name in ("run_gravity.py", "fb1_design_search.py"):
    if not os.path.isfile(os.path.join(TREE, name)):
        sys.exit(f"FB_TREE {TREE} has no {name}")
if os.environ.get("FB1_ALLOW_PAID_API") == "1":
    key = None
    with open(os.environ["FB_DOTENV"]) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if not line.startswith("ANTHROPIC_API_KEY="):
                continue
            value = line.split("=", 1)[1].strip()
            if value[:1] in ("'", '"') and value[-1:] == value[:1] and len(value) >= 2:
                value = value[1:-1]
            else:
                value = value.split(" #", 1)[0].strip()
            key = value
    if not key:
        sys.exit("the dotenv file did not set ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = key
os.chdir(TREE)
os.execv(PY, [PY, "-B", "run_gravity.py"])
