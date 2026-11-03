"""Collect every cell of a runs folder into one CSV (one row per cell x b) plus a spend summary. 0 USD.

  python -m asl_kit.collect <runs folder> [<runs folder> ...] --out <collected.csv>
Columns: benchmark, task, seed, b, n_obs, status, source and every score column present; a cell whose run
ended in error keeps its rows (status from the curve) and its receipt status is copied to `cell_status`.
"""
import argparse
import csv
import glob
import json
import os


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows, spend = [], {"cells": 0, "usd": 0.0, "calls": 0, "errors": 0, "mock_cells": 0}
    for folder in a.folders:
        for rec_path in sorted(glob.glob(os.path.join(folder, "*", "receipt.json"))):
            cell = os.path.dirname(rec_path)
            r = json.load(open(rec_path))
            args = r["args"]
            spend["cells"] += 1
            spend["usd"] += float(r["usage"]["usd"])
            spend["calls"] += int(r["usage"]["calls"])
            spend["errors"] += int(r["status"] != "completed")
            spend["mock_cells"] += int(bool(r["usage"].get("mock")))
            curve_path = os.path.join(cell, "curve.json")
            for c in (json.load(open(curve_path)) if os.path.exists(curve_path) else []):
                rows.append({"benchmark": args["benchmark"], "task": args["task"], "seed": args["seed"],
                             # ChemBench ignores --names: its oracle text is the released one
                             "names": "as_released" if args["benchmark"] == "chembench" else args.get("names"),
                             "cell_status": r["status"], "cell": cell,
                             **{k: v for k, v in c.items() if k != "skeleton"}})
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    spend["usd"] = round(spend["usd"], 4)
    print(json.dumps({"rows": len(rows), **spend}))


if __name__ == "__main__":
    main()
