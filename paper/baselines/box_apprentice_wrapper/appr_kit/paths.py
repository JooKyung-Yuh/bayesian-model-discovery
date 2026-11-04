"""Where the BoxingGym checkout, the world pack and the retry patch v3 are, and how the checkout is imported.

The checkout is read and imported only. `import_checkout()` puts <checkout>/src first on sys.path with bytecode
writing switched off (no __pycache__ is written into the checkout), and `load_run_experiment()` builds the
`run_experiment` module from the released run_experiment.py source with the retry patch v3 applied IN MEMORY.
That source is the kit's verbatim copy vendor/run_experiment.py: the development checkouts are sparse (src, conf) and
have no root files. The copy must have git blob id 92cdabe2..., and the checkout's own tree must name that blob for
run_experiment.py at 06c790aa (`git rev-parse <commit>:run_experiment.py` reads the tree only, so it works in a sparse,
blobless clone); where the checkout has the file in its worktree, that file must be the same blob too.
"""
import hashlib
import importlib.util
import os
import subprocess
import sys
import types

KIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK_PATH = os.path.join(KIT_DIR, "packs", "boxinggym_worlds.json")
PATCH_PATH = os.path.join(KIT_DIR, "patches", "ccr5b_retry_v3.patch")
VENDOR_RUN_EXPERIMENT = os.path.join(KIT_DIR, "vendor", "run_experiment.py")
REQUIREMENTS_PATH = os.path.join(KIT_DIR, "requirements_appr_20261008.txt")
SUMS_PATH = os.path.join(KIT_DIR, "SHA256SUMS")

# The world pack of the LLM-AutoSciLab wrapper (packs/boxinggym_worlds.json, listed with
# this sha256 in that kit's SHA256SUMS), copied byte for byte.
# 2026-10-08: the 16-entry pack of the AutoSciLab kit at d49f5c707 (= the 15 entries above byte for byte + peregrines seed 47,
# the site registry's peregrines seed; cross-checked against the recorded cells at 1ee084be).
PACK_SHA256 = "3228767bc8de3c3db17edd6b4b0cafa3ea7df03d3f56e5d9985fb93eb8bc44e9"
# BoxingGym-Baseline-Evals baselines/patches/ccr5b_retry_v3.patch (commit f5b5640d8), copied byte for byte.
PATCH_SHA256 = "7f04ce638e78611122b369420ac534af3a2ff141795df521cd62131b50221793"

# The BoxingGym checkout: github.com/JooKyung-Yuh/BoxingGym at this commit (upstream kanishkg/boxing-gym 6641cbb plus
# the local import / model-loading fixes 7a54a71 and the program-persistence commit 1aa4510; no prompt changes).
EXPECTED_HEAD = "06c790aa8e23c1f8f1858f0955ec4ffc9e4c61ba"
# git blob of run_experiment.py at EXPECTED_HEAD = the base blob ("index 92cdabe..") the v3 patch was written against.
RUN_EXPERIMENT_BLOB = "92cdabe2fed85e92209e972e1809e5ed18149dc5"
DEFAULT_CHECKOUT = "<development path>"
RUN_PATHS = ("src", "conf", "run_experiment.py")   # the only parts of the checkout the kit imports or reads


def checkout_dir():
    return os.environ.get("APPR_BG_CHECKOUT", DEFAULT_CHECKOUT)


def _git(path, *args):
    p = subprocess.run(["git", "-C", path, *args], capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def git_blob_id(data):
    """What `git hash-object` prints for these bytes."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def checkout_state(path=None):
    """HEAD, porcelain lines (whole tree and the run paths), and the run_experiment.py blob checks."""
    path = path or checkout_dir()
    rc_h, head, _ = _git(path, "rev-parse", "HEAD")
    rc_s, status, _ = _git(path, "status", "--porcelain")
    rc_r, run_status, _ = _git(path, "status", "--porcelain", "--", *RUN_PATHS)
    rc_b, blob_tree, _ = _git(path, "rev-parse", f"{EXPECTED_HEAD}:run_experiment.py")
    work = os.path.join(path, "run_experiment.py")
    blob_work = None
    if os.path.isfile(work):
        with open(work, "rb") as fh:
            blob_work = git_blob_id(fh.read())
    with open(VENDOR_RUN_EXPERIMENT, "rb") as fh:
        blob_vendor = git_blob_id(fh.read())
    return {"path": path, "head": head, "head_rc": rc_h,
            "porcelain_lines": [l for l in status.splitlines() if l.strip()], "status_rc": rc_s,
            "run_paths": list(RUN_PATHS), "run_paths_porcelain_lines": [l for l in run_status.splitlines() if l.strip()],
            "run_paths_status_rc": rc_r,
            "run_experiment_blob_in_tree": blob_tree if rc_b == 0 else None,
            "run_experiment_in_worktree": blob_work is not None,
            "run_experiment_blob_worktree": blob_work,
            "run_experiment_blob_vendored": blob_vendor}


def require_clean_checkout(path=None):
    """Refuse a different commit, local changes in the run paths, or a run_experiment.py that is not the patch base.

    Lines of `git status --porcelain` outside src/, conf/ and run_experiment.py (the development host's working repo has two such
    lines that predate this kit) do not touch anything the kit imports; they are recorded, not refused.
    """
    st = checkout_state(path)
    if st["head_rc"] != 0 or st["head"] != EXPECTED_HEAD:
        raise SystemExit(f"BoxingGym checkout {st['path']} is at {st['head']!r}, expected {EXPECTED_HEAD}")
    if st["run_paths_status_rc"] != 0 or st["run_paths_porcelain_lines"]:
        raise SystemExit(f"BoxingGym checkout {st['path']} has local changes in {RUN_PATHS}: {st['run_paths_porcelain_lines'][:5]}")
    blobs = {"tree": st["run_experiment_blob_in_tree"], "vendored": st["run_experiment_blob_vendored"]}
    if st["run_experiment_in_worktree"]:
        blobs["worktree"] = st["run_experiment_blob_worktree"]
    if any(v != RUN_EXPERIMENT_BLOB for v in blobs.values()):
        raise SystemExit(f"run_experiment.py blobs {blobs} != {RUN_EXPERIMENT_BLOB}")
    return st


def import_checkout(path=None):
    """<checkout>/src first on sys.path; never write .pyc into the checkout."""
    sys.dont_write_bytecode = True
    src = os.path.join(path or checkout_dir(), "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import boxing_gym
    got = os.path.dirname(os.path.dirname(os.path.abspath(boxing_gym.__file__)))
    if os.path.realpath(got) != os.path.realpath(src):
        raise SystemExit(f"boxing_gym imported from {got}, expected {src}")
    return src


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    with open(path, "rb") as fh:
        return sha256_bytes(fh.read())


def apply_unified_patch(text, patch_text, target="run_experiment.py"):
    """Apply the hunks of one file of a unified diff to `text`; every context and removed line must match exactly."""
    lines = text.splitlines(keepends=True)
    hunks, cur, in_target = [], None, False
    for raw in patch_text.splitlines(keepends=True):
        if raw.startswith("diff --git "):
            in_target = raw.rstrip("\n").endswith(f"b/{target}")
            cur = None
            continue
        if not in_target or raw.startswith(("index ", "--- ", "+++ ")):
            continue
        if raw.startswith("@@"):
            old_start = int(raw.split()[1].split(",")[0][1:])
            cur = {"old_start": old_start, "lines": []}
            hunks.append(cur)
            continue
        if cur is not None and raw[:1] in (" ", "-", "+"):
            cur["lines"].append(raw)
        elif cur is not None and raw.strip() == "":
            cur["lines"].append(" " + raw)   # an empty context line whose leading space was stripped
    out, pos = [], 0
    for h in hunks:
        start = h["old_start"] - 1
        out.extend(lines[pos:start])
        pos = start
        for hl in h["lines"]:
            tag, body = hl[0], hl[1:]
            if tag in (" ", "-"):
                if pos >= len(lines) or lines[pos].rstrip("\n") != body.rstrip("\n"):
                    raise ValueError(f"patch context mismatch at line {pos + 1}: {body!r}")
                if tag == " ":
                    out.append(lines[pos])
                pos += 1
            else:
                out.append(body if body.endswith("\n") else body + "\n")
    out.extend(lines[pos:])
    return "".join(out), len(hunks)


def load_run_experiment(path=None, module_name="appr_run_experiment_ccr5b_v3"):
    """The released run_experiment.py (the kit's verbatim copy, blob 92cdabe2) with the retry patch v3 applied in
    memory, as a module object; the checkout's tree is checked to name the same blob at EXPECTED_HEAD."""
    path = path or checkout_dir()
    with open(VENDOR_RUN_EXPERIMENT, "rb") as fh:
        raw = fh.read()
    if git_blob_id(raw) != RUN_EXPERIMENT_BLOB:
        raise SystemExit(f"{VENDOR_RUN_EXPERIMENT} blob {git_blob_id(raw)} != {RUN_EXPERIMENT_BLOB}")
    rc, blob_tree, err = _git(path, "rev-parse", f"{EXPECTED_HEAD}:run_experiment.py")
    if rc != 0 or blob_tree != RUN_EXPERIMENT_BLOB:
        raise SystemExit(f"{path} at {EXPECTED_HEAD} names run_experiment.py blob {blob_tree!r} ({err[:200]}), "
                         f"expected {RUN_EXPERIMENT_BLOB}")
    with open(PATCH_PATH, "rb") as fh:
        patch_raw = fh.read()
    if sha256_bytes(patch_raw) != PATCH_SHA256:
        raise SystemExit(f"{PATCH_PATH} sha256 {sha256_bytes(patch_raw)} != {PATCH_SHA256}")
    patched, n_hunks = apply_unified_patch(raw.decode(), patch_raw.decode())
    module = types.ModuleType(module_name)
    module.__file__ = VENDOR_RUN_EXPERIMENT + " [CCR-5b v3 applied in memory]"
    sys.modules[module_name] = module
    exec(compile(patched, "<run_experiment.py + ccr5b_retry_v3 (in memory)>", "exec"), module.__dict__)
    info = {"source": VENDOR_RUN_EXPERIMENT, "source_git_blob": RUN_EXPERIMENT_BLOB,
            "checkout_tree_blob": blob_tree, "source_sha256": sha256_bytes(raw), "patch": PATCH_PATH,
            "patch_sha256": PATCH_SHA256, "hunks_applied": n_hunks, "patched_sha256": sha256_bytes(patched.encode())}
    return module, info


def kit_file_hashes():
    out = {}
    for root, dirs, files in os.walk(KIT_DIR):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            p = os.path.join(root, name)
            out[os.path.relpath(p, KIT_DIR)] = sha256_file(p)
    return out


def check_sums():
    """[(relative path, ok)] for every SHA256SUMS line."""
    rows = []
    with open(SUMS_PATH) as fh:
        for line in fh:
            if not line.strip():
                continue
            digest, rel = line.split(None, 1)
            rel = rel.strip().lstrip("*")
            p = os.path.join(KIT_DIR, rel)
            rows.append((rel, os.path.isfile(p) and sha256_file(p) == digest))
    return rows
