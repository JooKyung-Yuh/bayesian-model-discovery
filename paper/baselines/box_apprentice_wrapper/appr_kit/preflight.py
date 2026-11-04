"""Preflight (0 USD): refuse to start unless the host can run the released Apprentice as the kit expects.

  python -m appr_kit.preflight [--quiet] [--mode stub|mock-transport|paid] [--no-fit]
Checks, each printed as OK / FAIL (no secret value is ever printed):
  python 3.12; every package pinned in requirements_appr_20261008.txt installed at its pinned version; pytensor has a
  C++ compiler; the BoxingGym checkout is at 06c790aa with no local changes in src/, conf/, run_experiment.py, the kit's
  vendor/run_experiment.py, the checkout's tree entry (and its worktree file, where present) are the CCR-5b v3 base blob
  92cdabe2 and the patch applies in memory; `import boxing_gym` resolves to the
  checkout's src; the world pack and the kit files match SHA256SUMS; ANTHROPIC_API_KEY present in the environment by
  name (required only for --mode paid); the output root is writable; an in-process PyMC fit of a tiny model (its
  PyTensor compile dir is a fresh temporary folder, removed afterwards; --no-fit skips it).
"""
import argparse
import atexit
import importlib.metadata as md
import json
import os
import shutil
import sys
import tempfile

KIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KIT_DIR not in sys.path:
    sys.path.insert(0, KIT_DIR)
sys.dont_write_bytecode = True

from appr_kit import paths  # noqa: E402


def pinned():
    out = {}
    with open(paths.REQUIREMENTS_PATH) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if "==" in line:
                name, ver = line.split("==", 1)
                out[name.strip()] = ver.strip()
    return out


def check_all(mode="stub", fit=True):
    results = []

    def add(name, ok, detail=""):
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    # a fresh PyTensor compile dir for this process only (never the shared ~/.pytensor), removed at exit; registered before
    # pytensor is imported, so pytensor's own atexit cleanup of its module cache runs first
    tmp_ptc = tempfile.mkdtemp(prefix="appr_preflight_ptc_")
    os.environ["PYTENSOR_FLAGS"] = f"base_compiledir={tmp_ptc}"
    atexit.register(shutil.rmtree, tmp_ptc, ignore_errors=True)
    add("python 3.12", sys.version_info[:2] == (3, 12), sys.version.split()[0])
    bad = []
    for name, ver in pinned().items():
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            got = None
        if got != ver:
            bad.append(f"{name} {got} != {ver}")
    add("venv matches requirements_appr_20261008.txt", not bad, "; ".join(bad[:6]) or f"{len(pinned())} pins")
    try:
        st = paths.checkout_state()
        add("checkout at expected commit", st["head"] == paths.EXPECTED_HEAD, f"{st['path']} {st['head']}")
        add("checkout run paths clean (src, conf, run_experiment.py)",
            st["run_paths_status_rc"] == 0 and not st["run_paths_porcelain_lines"],
            f"{len(st['run_paths_porcelain_lines'])} lines; whole tree {len(st['porcelain_lines'])} lines")
        blobs = {"tree": st["run_experiment_blob_in_tree"], "vendored": st["run_experiment_blob_vendored"]}
        if st["run_experiment_in_worktree"]:
            blobs["worktree"] = st["run_experiment_blob_worktree"]
        add("run_experiment.py: kit copy = checkout tree entry = v3 patch base blob",
            all(v == paths.RUN_EXPERIMENT_BLOB for v in blobs.values()), json.dumps(blobs))
    except Exception as exc:  # noqa: BLE001
        add("checkout readable", False, f"{type(exc).__name__}: {exc}"[:200])
    try:
        raw = open(paths.VENDOR_RUN_EXPERIMENT).read()
        patched, n = paths.apply_unified_patch(raw, open(paths.PATCH_PATH).read())
        add("CCR-5b v3 patch applies in memory", n == 2 and paths.sha256_file(paths.PATCH_PATH) == paths.PATCH_SHA256,
            f"{n} hunks, patched sha256 {paths.sha256_bytes(patched.encode())[:16]}")
    except Exception as exc:  # noqa: BLE001
        add("CCR-5b v3 patch applies in memory", False, f"{type(exc).__name__}: {exc}"[:200])
    try:
        paths.import_checkout()
        import boxing_gym
        add("import boxing_gym from the checkout", True, boxing_gym.__file__)
    except BaseException as exc:  # noqa: BLE001
        add("import boxing_gym from the checkout", False, f"{type(exc).__name__}: {exc}"[:200])
    add("world pack sha256", paths.sha256_file(paths.PACK_PATH) == paths.PACK_SHA256, paths.PACK_PATH)
    if os.path.exists(paths.SUMS_PATH):
        rows = paths.check_sums()
        badf = [r for r, ok in rows if not ok]
        add("kit files match SHA256SUMS", not badf, ", ".join(badf[:5]) or f"{len(rows)} files equal")
    else:
        add("kit files match SHA256SUMS", False, "SHA256SUMS missing")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    add("ANTHROPIC_API_KEY present (by name)" + ("" if mode == "paid" else " [not needed in this mode]"),
        has_key or mode != "paid", "set" if has_key else "missing")
    out_root = os.environ.get("APPR_OUT_ROOT", "")
    try:
        if not out_root:
            raise ValueError("APPR_OUT_ROOT unset")
        os.makedirs(out_root, exist_ok=True)
        probe = os.path.join(out_root, f".preflight_write_probe_{os.getpid()}")
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
        add("output root writable", True, out_root)
    except Exception as exc:  # noqa: BLE001
        add("output root writable", False, f"{out_root}: {exc}")
    try:
        import pytensor
        add("pytensor C++ compiler", bool(pytensor.config.cxx), pytensor.config.cxx or "none (fits would fall back to python)")
    except Exception as exc:  # noqa: BLE001
        add("pytensor C++ compiler", False, f"{type(exc).__name__}: {exc}"[:200])
    if fit:
        try:
            import numpy as np
            import pymc as pm
            y = np.array([0.8, 1.1, 1.3, 0.9, 1.0])
            with pm.Model():
                mu = pm.Normal("mu", 0.0, 2.0)
                pm.Normal("y", mu=mu, sigma=0.5, observed=y)
                idata = pm.sample(200, tune=200, chains=1, cores=1, random_seed=1, progressbar=False,
                                  compute_convergence_checks=False)
            m = float(idata.posterior["mu"].mean())
            add("in-process PyMC fit of a tiny model", np.isfinite(m) and abs(m - 1.0) < 0.5, f"posterior mean mu {m:.3f}")
        except Exception as exc:  # noqa: BLE001
            add("in-process PyMC fit of a tiny model", False, f"{type(exc).__name__}: {exc}"[:200])
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--mode", choices=("stub", "mock-transport", "paid"), default="stub")
    ap.add_argument("--no-fit", action="store_true")
    a = ap.parse_args(argv)
    results = check_all(a.mode, fit=not a.no_fit)
    failed = [r for r in results if not r["ok"]]
    if not a.quiet or failed:
        for r in results:
            print(("OK   " if r["ok"] else "FAIL ") + r["check"] + (f"  [{r['detail']}]" if r["detail"] else ""))
    print(json.dumps({"preflight": "PASS" if not failed else "FAIL", "mode": a.mode, "failed": [r["check"] for r in failed]}))
    return 0 if not failed else 3


if __name__ == "__main__":
    sys.exit(main())
