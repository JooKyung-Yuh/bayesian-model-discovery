#!/usr/bin/env python3
"""Compare the Python sources of two trees by their abstract syntax trees, ignoring comments and docstrings.

For every `.py` file under the destination tree (or the paths given) the matching source file is parsed, module, class
and function docstrings are removed from both syntax trees, and the trees are compared with `ast.dump` (no line or
column attributes). A file is reported as

  identical     the two trees are the same: only comments, docstrings, blank lines or formatting differ
  differs       the trees differ; the first differing top-level statement (and its line in each file) is named, and the
                top-level definitions present in only one side are listed (`removed_defs`, `added_defs`)
  missing       the file exists in only one tree
  syntax_error  one side does not parse

A mapping file (JSON, {"destination/path.py": "source/path.py"}) renames or moves files between the trees.

Usage: ast_equiv.py <source tree> <destination tree> [--mapping mapping.json] [--json report.json] [paths...]
Exit status 0 when every compared file is identical, 1 otherwise.
"""
import argparse
import ast
import json
import os
import sys


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def parse(path):
    with open(path, encoding="utf-8") as fh:
        return strip_docstrings(ast.parse(fh.read(), filename=path))


def top_level_defs(tree):
    names = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node.name] = node.lineno
    return names


def dump(node):
    return ast.dump(node, include_attributes=False)


def compare(src_path, dst_path):
    try:
        a = parse(src_path)
    except SyntaxError as exc:
        return {"status": "syntax_error", "side": "source", "error": str(exc)}
    try:
        b = parse(dst_path)
    except SyntaxError as exc:
        return {"status": "syntax_error", "side": "destination", "error": str(exc)}
    if dump(a) == dump(b):
        return {"status": "identical"}
    da, db = top_level_defs(a), top_level_defs(b)
    result = {"status": "differs", "removed_defs": sorted(set(da) - set(db)), "added_defs": sorted(set(db) - set(da))}
    # the first top-level statement whose dump differs, matched by position among statements of the same definition name
    sa = {dump(n): n for n in a.body}
    for n in b.body:
        if dump(n) not in sa:
            result["first_difference"] = {"destination_line": getattr(n, "lineno", None), "statement": type(n).__name__,
                                          "name": getattr(n, "name", None)}
            break
    for n in a.body:
        if dump(n) not in {dump(m) for m in b.body}:
            result.setdefault("first_difference", {})["source_line"] = getattr(n, "lineno", None)
            break
    # with the removed definitions taken out of the source, is the rest identical?
    if result["removed_defs"] and not result["added_defs"]:
        a2 = ast.Module(body=[n for n in a.body if not (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in result["removed_defs"])], type_ignores=[])
        result["identical_except_removed_defs"] = dump(a2) == dump(b)
    return result


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("destination")
    ap.add_argument("--mapping", help="JSON file: destination path -> source path, for moved or renamed files")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("paths", nargs="*", help="destination-relative .py paths (default: every .py under the destination)")
    args = ap.parse_args(argv)
    mapping = {}
    if args.mapping:
        with open(args.mapping) as fh:
            mapping = json.load(fh)
    paths = args.paths or sorted(os.path.relpath(os.path.join(dp, f), args.destination)
                                 for dp, dns, fns in os.walk(args.destination)
                                 for f in fns if f.endswith(".py") and ".git" not in dp.split(os.sep))
    report, counts = {}, {}
    for rel in paths:
        src_rel = mapping.get(rel, rel)
        src, dst = os.path.join(args.source, src_rel), os.path.join(args.destination, rel)
        if not os.path.exists(src):
            report[rel] = {"status": "missing", "side": "source", "source_path": src_rel}
        elif not os.path.exists(dst):
            report[rel] = {"status": "missing", "side": "destination"}
        else:
            report[rel] = compare(src, dst)
            if src_rel != rel:
                report[rel]["source_path"] = src_rel
        counts[report[rel]["status"]] = counts.get(report[rel]["status"], 0) + 1
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"source": args.source, "destination": args.destination, "counts": counts, "files": report}, fh, indent=1)
    for rel, r in report.items():
        if r["status"] != "identical":
            extra = {k: v for k, v in r.items() if k != "status"}
            print(f"{r['status']:<13} {rel}  {json.dumps(extra)}")
    print("summary:", json.dumps(counts))
    return 0 if set(counts) <= {"identical"} else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
