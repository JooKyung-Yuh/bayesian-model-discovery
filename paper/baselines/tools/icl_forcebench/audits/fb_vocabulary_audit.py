#!/usr/bin/env python3
"""ForceBench vocabulary audit: every text a ForceBench proposer request sends to the
language model, scanned for mechanics vocabulary.

Usage: fb_vocabulary_audit.py <cell folder | proposer folder | request record> [...] [--max-hits N] [--json OUT]

For every *.request.json record found (FB1_RUN_*_proposer/), every body of every attempt is scanned: the system text and
every user message (the packet and each repair sentence), each JSON packet as its keys and string values (numbers and the
model's own assistant turns are not engine text). A hit is a token of VOCABULARY (case-insensitive; text is split into
tokens on every character that is not a letter or a digit, so `max_abs_accel` gives the tokens max, abs, accel), or the
phrase "unit vector". Words that merely contain a listed stem as a different word (positive, Normal, posterior, level,
width) are not hits: the rule is the whole token, with the stems below for inflected forms.
Prints one line per distinct (place, token) with its count and a sample, then "VOCABULARY HITS <n>" (n = distinct hit
places over all records). Exit 0 when n = 0, 1 otherwise, 2 when no request record is found. Reads JSON only.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

#: the agreed list, as whole tokens with their inflections
VOCABULARY = re.compile(
    r"^(forces?|accel\w*|mass(es)?|inertial\w*|velocit\w*|vel|positions?|positional|pos|distances?|norms?|radial\w*|"
    r"particles?|bod(y|ies)|orbit\w*|gravit\w*|coulomb\w*|charges?|charged|potentials?|trajector\w*|verlet|integrat\w*|"
    r"dt|elapsed|times?|knobs?)$", re.IGNORECASE)
PHRASE = re.compile(r"unit[\s_]+vector", re.IGNORECASE)
TOKEN = re.compile(r"[A-Za-z0-9]+")


def hits_in(text):
    found = [t.lower() for t in TOKEN.findall(text) if VOCABULARY.match(t)]
    found += ["unit vector"] * len(PHRASE.findall(text))
    return found


def walk(value, place, out):
    """(place, token, sample) for every hit in a JSON value: keys and strings."""
    if isinstance(value, dict):
        for key, item in value.items():
            for tok in hits_in(str(key)):
                out.append((f"{place}.<key>", tok, str(key)))
            walk(item, f"{place}.{key}", out)
    elif isinstance(value, list):
        for item in value:
            walk(item, f"{place}[]", out)
    elif isinstance(value, str):
        for tok in hits_in(value):
            out.append((place, tok, value[:120]))


def scan_text(text, place, out):
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, (dict, list)):
        walk(value, place, out)
    else:
        for tok in hits_in(text):
            out.append((place, tok, text[:120]))


def records(paths):
    for p in map(Path, paths):
        if p.is_file() and p.name.endswith(".request.json"):
            yield p
        elif p.is_dir():
            yield from sorted(p.rglob("*.request.json"))


def main(argv):
    args, max_hits, out_json = [], 40, None
    it = iter(argv)
    for a in it:
        if a == "--max-hits":
            max_hits = int(next(it))
        elif a == "--json":
            out_json = next(it)
        else:
            args.append(a)
    found = list(records(args))
    if not found:
        print("no *.request.json record found")
        return 2
    per = defaultdict(Counter)
    sample = {}
    bodies = 0
    for path in found:
        record = json.loads(path.read_text())
        for body in record.get("bodies") or [record.get("body") or {}]:
            bodies += 1
            hits = []
            system = body.get("system")
            if system is None:
                system = "".join(m.get("content", "") for m in body.get("messages") or [] if m.get("role") == "system")
            scan_text(system or "", "system", hits)
            for m, message in enumerate(body.get("messages") or []):
                if message.get("role") != "user":
                    continue
                content = message.get("content")
                content = content if isinstance(content, str) else json.dumps(content)
                scan_text(content, "user[0]" if m == 0 else "user[repair]", hits)
            for place, tok, text in hits:
                generic = re.sub(r"\[\d+\]", "[]", place)
                per[(generic, tok)][path.name] += 1
                sample.setdefault((generic, tok), text)
    total = 0
    for (place, tok), files in sorted(per.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])):
        total += 1
        if total <= max_hits:
            print(f"HIT {tok!r} at {place}: {sum(files.values())} times in {len(files)} records; e.g. {sample[(place, tok)]!r}")
    print(f"records {len(found)} bodies {bodies}")
    print(f"VOCABULARY HITS {total}")
    if out_json:
        Path(out_json).write_text(json.dumps([{"place": p, "token": t, "count": sum(f.values()), "records": len(f),
                                               "sample": sample[(p, t)]} for (p, t), f in sorted(per.items())], indent=1))
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
