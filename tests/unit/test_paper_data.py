"""The aggregate data under paper/data/ loads, carries no record path or host name, and renders."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper" / "data"
FILES = sorted(p for p in DATA.rglob("*") if p.suffix in (".json", ".csv", ".md"))
PRIVATE = re.compile(r"/home/[a-z]+/|/Users/[a-z]+/|/tmp/claude|@gmail\.com")


@pytest.mark.parametrize("path", FILES, ids=[str(p.relative_to(DATA)) for p in FILES])
def test_data_file_is_readable_and_free_of_private_paths(path):
    text = path.read_text(encoding="utf-8")
    assert not PRIVATE.search(text), path
    if path.suffix == ".json":
        json.loads(text)


def test_tables_render():
    out = subprocess.run([sys.executable, str(DATA / "tables.py")], capture_output=True, text=True, check=True).stdout
    for head in ("## E1 paired comparison", "## E2 final error per task", "## F1 final top-1", "## D3 LLM calls", "## G4 perturbed", "## H breadth", "## C ForceBench"):
        assert head in out, head
