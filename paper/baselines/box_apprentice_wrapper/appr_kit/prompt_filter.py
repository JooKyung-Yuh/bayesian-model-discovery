"""Prompt variant `no_thought` (2026-10-08): the released `<thought>` instruction removed from what is sent.

Why: claude-opus-5-5 refused every call of the released Apprentice prompt (stop_reason refusal, 5 of 5 in the
2026-10-08 canary; the released text asks the model to write its reasoning inside `<thought>` tags). The released
parser never reads that tag (agents/agent.py 68-71 match `<observe>` / `<answer>` only; the released code's own
`<thought>` strip is commented out at agent.py 29), so the instruction is decoration for the parser and can be
removed without touching what the released code reads back.

Where the released text carries it (checkout 06c790aa):
  - the environments' system prompts, as example lines `<thought> your thought </thought>` (envs/death_process.py
    144, 297-311; dugongs.py 45-244; peregrines.py 40-246; hyperbolic_temporal_discount.py 197-396;
    survival_analysis.py 192-268) - our five tasks have only these example lines;
  - the scientist's follow-up prompts (agents/agent.py 111 and 113): the sentence "Articulate your strategy for
    choosing measurements in <thought>." and the format lines `<thought>your thought</thought>` / `Thought: <thought>`.
The three rules below remove exactly those; "You may also think before providing your predictions." (an example
instruction without a tag) stays. The filter is applied in claude_path.ClaudePath.call to the system string and to
every user turn at send time, so both sites are covered in one place; the released objects (scientist.system, its
message history) keep their text, and llm_requests.jsonl holds what was actually sent. `apply()` reports how many
substitutions each rule made; the usage row and the receipt record them, and the kit refuses to send a text that
still contains `<thought>` under this variant.
"""
import hashlib
import re

VARIANTS = ("released", "no_thought")

# (rule id, compiled pattern, replacement). Order matters only for the counts.
RULES = (
    # a line that is nothing but a <thought>...</thought> example (any spacing inside), with its newline
    ("thought_example_line", re.compile(r"^[ \t]*<thought>[^<\n]*</thought>[ \t]*\n", re.MULTILINE), ""),
    # the follow-up sentence (agent.py 111 / 113), with the space before it when present
    ("articulate_sentence", re.compile(r" ?Articulate your strategy for choosing measurements in <thought>\."), ""),
    # the second follow-up's format line (agent.py 113)
    ("thought_format_line", re.compile(r"^Thought: <thought>\n", re.MULTILINE), ""),
)


def apply(text, variant):
    """(filtered text, {rule id: substitutions}); the released variant returns the text unchanged with zero counts."""
    if variant not in VARIANTS:
        raise ValueError(f"prompt variant must be one of {VARIANTS}, not {variant!r}")
    counts = {rule: 0 for rule, _p, _r in RULES}
    if variant == "released" or not isinstance(text, str) or not text:
        return text, counts
    out = text
    for rule, pattern, repl in RULES:
        out, n = pattern.subn(repl, out)
        counts[rule] = n
    return out, counts


def remaining_tags(text):
    """How many `<thought>` openings a text still carries."""
    return text.count("<thought>") if isinstance(text, str) else 0


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def describe(variant):
    return {"variant": variant,
            "rules": ([{"id": rule, "pattern": pattern.pattern} for rule, pattern, _r in RULES]
                      if variant == "no_thought" else []),
            "kept": ("the <observe> / <answer> format lines, the retry prompts and 'You may also think before providing "
                     "your predictions.' (no tag)") if variant == "no_thought" else "everything (released text)"}
