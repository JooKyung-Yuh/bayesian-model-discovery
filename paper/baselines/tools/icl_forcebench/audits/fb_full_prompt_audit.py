#!/usr/bin/env python3
"""ForceBench full-metadata prompt audit: the inverse of fb_vocabulary_audit.py.

Usage: fb_full_prompt_audit.py <description file> <cell folder | proposer folder | record> [...] [--system-only]

For every *.request.json record found, every body of every attempt (an engine record's "bodies" list, or the single
"body" of an ICL transport record, tools/icl_baseline/transport.py) must satisfy:
  1. the system text (Anthropic `system`, or the OpenAI system message) starts with the description block: the file's
     text, then one blank line - so its sha256 is the file's;
  2. the native input names r, p1, p2 and t each occur as a whole token somewhere in the system text or the user messages;
  3. no masked name X1, X2, X3 or X4 occurs as a whole token anywhere in the system text or the user messages (keys and
     strings; the model's own replies are not checked).
A whole token is a maximal run of letters, digits and underscores split on everything else, so `X1` in "c*X2/X1**2" is
a token but `pX1` is not. Prints one line per failing body (at most 20), then
"FULL PROMPT AUDIT <records> records <bodies> bodies: description <n_ok>/<bodies>, native names <..>, masked names 0/<..> -> PASS|FAIL".
Exit 0 on PASS, 1 on FAIL, 2 when no record is found. Reads JSON only.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

NATIVE = ("r", "p1", "p2", "t")
MASKED = ("X1", "X2", "X3", "X4")
TOKEN = re.compile(r"[A-Za-z0-9_]+")


def tokens(text):
    return set(TOKEN.findall(text))


def records(paths):
    for p in map(Path, paths):
        if p.is_file() and p.name.endswith(".request.json"):
            yield p
        elif p.is_dir():
            yield from sorted(p.rglob("*.request.json"))


def body_texts(body):
    system = body.get("system")
    if system is None:
        system = "".join(m.get("content", "") for m in body.get("messages") or [] if m.get("role") == "system")
    users = []
    for message in body.get("messages") or []:
        if message.get("role") == "user":
            content = message.get("content")
            users.append(content if isinstance(content, str) else json.dumps(content))
    return system or "", users


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    if len(args) < 2:
        print(__doc__.split("\n\n")[1])
        return 2
    description = Path(args[0]).read_text()
    prefix = description + "\n"
    digest = hashlib.sha256(description.encode()).hexdigest()
    found = list(records(args[1:]))
    if not found:
        print("no *.request.json record found")
        return 2
    bodies = desc_ok = native_ok = masked_hits = 0
    problems = []
    for path in found:
        record = json.loads(path.read_text())
        for b, body in enumerate(record.get("bodies") or ([record["body"]] if isinstance(record.get("body"), dict) else [])):
            bodies += 1
            system, users = body_texts(body)
            seen = tokens(system) | set().union(*(tokens(u) for u in users)) if users else tokens(system)
            if system.startswith(prefix):
                desc_ok += 1
            else:
                problems.append(f"{path.name} body {b}: the system text does not start with the description block")
            missing = [n for n in NATIVE if n not in seen]
            if missing:
                problems.append(f"{path.name} body {b}: native names missing {missing}")
            else:
                native_ok += 1
            leaked = sorted(n for n in MASKED if n in seen)
            if leaked:
                masked_hits += 1
                problems.append(f"{path.name} body {b}: masked names present {leaked}")
    for line in problems[:20]:
        print(line)
    ok = desc_ok == bodies and native_ok == bodies and masked_hits == 0 and bodies > 0
    print(f"description {Path(args[0]).name} sha256 {digest}")
    print(f"FULL PROMPT AUDIT {len(found)} records {bodies} bodies: description {desc_ok}/{bodies}, native names "
          f"{native_ok}/{bodies}, masked names {masked_hits}/{bodies} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
