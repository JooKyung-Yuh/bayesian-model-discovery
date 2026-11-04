"""Where the released checkout and the kit's data packs are, and how the checkout is imported."""
import hashlib
import os
import subprocess
import sys

KIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK_DIR = os.path.join(KIT_DIR, "packs")
STUB_DIR = os.path.join(KIT_DIR, "tests", "stub")

# The released code: github.com/scientific-discovery/LLM-AutoSciLab at this commit (no edits).
EXPECTED_HEAD = "9a37255ff5432a7155b6cde8985e9654f2f8f272"
DEFAULT_CHECKOUT = "<development path>"   # the development host; bai sets ASL_CHECKOUT


def checkout_dir():
    return os.environ.get("ASL_CHECKOUT", DEFAULT_CHECKOUT)


def checkout_state(path=None):
    """HEAD and porcelain status of the checkout; the runner refuses a dirty or different tree."""
    path = path or checkout_dir()
    head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "-C", path, "status", "--porcelain"], capture_output=True, text=True)
    return {"path": path, "head": head.stdout.strip(), "head_rc": head.returncode,
            "dirty_lines": [l for l in status.stdout.splitlines() if l.strip()], "status_rc": status.returncode}


def require_clean_checkout(path=None):
    st = checkout_state(path)
    if st["head_rc"] != 0 or st["head"] != EXPECTED_HEAD:
        raise SystemExit(f"AutoSciLab checkout {st['path']} is at {st['head']!r}, expected {EXPECTED_HEAD}")
    if st["status_rc"] != 0 or st["dirty_lines"]:
        raise SystemExit(f"AutoSciLab checkout {st['path']} has local changes: {st['dirty_lines'][:5]}")
    return st


def import_checkout(path=None):
    """Put the checkout (package root and scripts/) on sys.path; never write .pyc into it."""
    sys.dont_write_bytecode = True
    path = path or checkout_dir()
    for p in (os.path.join(path, "scripts"), path):
        if p not in sys.path:
            sys.path.insert(0, p)
    return path


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def kit_file_hashes():
    out = {}
    for root, _dirs, files in os.walk(KIT_DIR):
        if "__pycache__" in root:
            continue
        for name in sorted(files):
            if name.endswith((".py", ".sh", ".json", ".txt", ".md")):
                p = os.path.join(root, name)
                out[os.path.relpath(p, KIT_DIR)] = sha256_file(p)
    return out
