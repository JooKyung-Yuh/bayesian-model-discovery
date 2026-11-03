"""Strip record paths, host names and project-internal names from the appendix table JSON files (one-off, kept for the record)."""
import json
import pathlib
import re
import sys

SRC = pathlib.Path(sys.argv[1]); DST = pathlib.Path(sys.argv[2]); DST.mkdir(parents=True, exist_ok=True)
DROP_KEYS = {"sources", "path", "paths", "folder", "record", "published_line", "line", "edition"}
WORDS = [  # internal name -> plain wording
    (r"ASL-NAT", "matched LLM-AutoSciLab runs"), (r"FB-FULL", "full-interface ForceBench runs"), (r"FB-NEUTRAL", "neutral-name ForceBench runs"),
    (r"C2-PERT", "perturbed-constant study"), (r"C2P-TWOSTAGE", "two-stage variant"), (r"C4FB-DIAG", "the ForceBench design-criterion diagnosis"),
    (r"APPX-REVIEW-1", "the appendix review"), (r"\bagent[BCD]\b", "the analysis scripts"), (r"fig2_pairs\.out", "the Figure 2 pairing script"),
    (r"lossmatch_outputs_20261012", "the loss-matched rescoring"), (r"\be2 cells?\b", "feedback-study cells"), (r"\bU3\b", "the shared protocol"),
    (r"\bfbfull\b", "full-interface"), (r"\bFA-C-R\b", "the later-round prompt"), (r"\bFA-C_trig\b", "the ForceBench first-round prompt"), (r"\bFA-C\b", "the first-round prompt"),
    (r"\bbai-a6000\b|\bbai-ada\b|\bbai\b", "a compute host"), (r"\belicer\b|\belice[0-9]+\b", "the analysis host"), (r"the Mac\b", "a second machine"),
    (r"/home/[^\s'\")\]]+", "<record path>"), (r"/tmp/[^\s'\")\]]+", "<record path>"),
]
def clean_str(s):
    for pat, rep in WORDS:
        s = re.sub(pat, rep, s)
    return s
def walk(o, key=None):
    if isinstance(o, dict):
        return {k: walk(v, k) for k, v in o.items() if k not in DROP_KEYS}
    if isinstance(o, list):
        return [walk(v, key) for v in o]
    if isinstance(o, str):
        return clean_str(o)
    return o
for f in sorted(SRC.glob("*.json")):
    d = walk(json.load(open(f, encoding="utf-8")))
    json.dump(d, open(DST / f.name, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", f.name)
