"""Preflight (0 USD): refuse to start unless the environment can run the released agent as is.

  python -m asl_kit.preflight [--pysr-smoke] [--quiet]
Checks, each printed as OK/FAIL (no secret value is ever printed):
  python and package imports (numpy, scipy, sklearn, pandas, pydantic, sympy, rich, openai, together, httpx,
  anthropic, dotenv) with versions; `import pysr, juliacall` works, is the real PySR (not the kit's stub) and
  JULIA_DEPOT_PATH is set to an existing directory; the AutoSciLab checkout is at the expected commit with no
  local changes; the three data packs match SHA256SUMS; ANTHROPIC_API_KEY is present in the environment
  (by name); the output root is writable. --pysr-smoke additionally runs one 1-iteration PySR fit (first run
  compiles into the Julia depot; run it once, alone, before launching parallel cells).
"""
import argparse
import importlib
import json
import os
import sys

KIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KIT_DIR not in sys.path:
    sys.path.insert(0, KIT_DIR)

from asl_kit import paths  # noqa: E402

MODULES = ("numpy", "scipy", "sklearn", "pandas", "pydantic", "sympy", "rich", "openai", "together", "httpx",
           "anthropic", "dotenv")


def check_all(pysr_smoke=False):
    results = []

    def add(name, ok, detail=""):
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    add("python", sys.version_info >= (3, 10), sys.version.split()[0])
    for m in MODULES:
        try:
            mod = importlib.import_module(m)
            add(f"import {m}", True, str(getattr(mod, "__version__", "?")))
        except Exception as exc:  # noqa: BLE001
            add(f"import {m}", False, f"{type(exc).__name__}: {exc}"[:200])
    depot = os.environ.get("JULIA_DEPOT_PATH", "")
    add("JULIA_DEPOT_PATH set and exists", bool(depot) and all(os.path.isdir(p) for p in depot.split(":") if p),
        depot or "unset")
    try:
        import pysr  # noqa: F401
        import juliacall  # noqa: F401
        stub = getattr(pysr, "STUB", False) or os.path.join(KIT_DIR, "tests") in os.path.abspath(pysr.__file__)
        add("import pysr, juliacall (real PySR)", not stub, f"pysr {getattr(pysr, '__version__', '?')} at {pysr.__file__}")
    except Exception as exc:  # noqa: BLE001
        text = f"{type(exc).__name__}: {str(exc)[:300]}"
        if any(k in str(exc) for k in ("PythonCall", "instantiate", "julia", "Julia")):
            text += " -- the Julia depot must be instantiated (needs the user's approval)"
        add("import pysr, juliacall (real PySR)", False, text)
    st = paths.checkout_state()
    add("checkout at expected commit", st["head"] == paths.EXPECTED_HEAD, f"{st['path']} {st['head']}")
    add("checkout has no local changes", st["status_rc"] == 0 and not st["dirty_lines"], f"{len(st['dirty_lines'])} lines")
    sums = os.path.join(KIT_DIR, "SHA256SUMS")
    if os.path.exists(sums):
        bad = []
        for line in open(sums):
            if not line.strip():
                continue
            digest, rel = line.split(None, 1)
            rel = rel.strip().lstrip("*")
            p = os.path.join(KIT_DIR, rel)
            if not os.path.exists(p) or paths.sha256_file(p) != digest:
                bad.append(rel)
        add("kit files match SHA256SUMS", not bad, ", ".join(bad[:5]) or "all equal")
    else:
        add("kit files match SHA256SUMS", False, "SHA256SUMS missing")
    add("ANTHROPIC_API_KEY present (by name)", bool(os.environ.get("ANTHROPIC_API_KEY")), "set" if os.environ.get("ANTHROPIC_API_KEY") else "missing")
    out_root = os.environ.get("ASL_OUT_ROOT", "")
    try:
        os.makedirs(out_root, exist_ok=True)
        probe = os.path.join(out_root, ".preflight_write_probe")
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
        add("output root writable", bool(out_root), out_root)
    except Exception as exc:  # noqa: BLE001
        add("output root writable", False, f"{out_root}: {exc}")
    if pysr_smoke and all(r["ok"] for r in results if r["check"].startswith("import pysr")):
        try:
            import numpy as np
            import pandas as pd
            paths.import_checkout()
            from autoscilab.equation_learner.learner import _make_pysr_model   # the released factory
            X = pd.DataFrame({"a": np.linspace(1, 2, 20)})
            m = _make_pysr_model(1, ["exp"], populations=2)
            m.fit(X, 2.0 * X["a"].to_numpy())
            add("pysr smoke fit (released _make_pysr_model)", True, str(m.sympy()))
        except Exception as exc:  # noqa: BLE001
            add("pysr smoke fit", False, f"{type(exc).__name__}: {str(exc)[:300]}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pysr-smoke", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    results = check_all(a.pysr_smoke)
    failed = [r for r in results if not r["ok"]]
    if not a.quiet or failed:
        for r in results:
            print(("OK   " if r["ok"] else "FAIL ") + r["check"] + (f"  [{r['detail']}]" if r["detail"] else ""))
    print(json.dumps({"preflight": "PASS" if not failed else "FAIL", "failed": [r["check"] for r in failed]}))
    return 0 if not failed else 3


if __name__ == "__main__":
    sys.exit(main())
