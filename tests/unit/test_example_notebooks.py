"""The example notebooks in `examples/` are committed with their outputs; check what they carry.

Each notebook is read with the standard library `json` module only, so this test needs neither
Jupyter nor the engine. It does not execute anything. It checks that every committed notebook

- is an nbformat 4 document with the usual cell structure and the portable `python3` kernel,
- was executed top to bottom in one kernel (every code cell has an execution count, starting at
  1 and increasing),
- contains no error output,
- prints no absolute path of the machine that executed it (`/home/`, `/tmp/`), and
- contains nothing shaped like an API key, in its sources or its outputs.

The "Notebooks" section of `USAGE.md` says how to re-execute the notebooks so that these checks
keep passing.
"""

import json
import re
from itertools import pairwise
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
NOTEBOOKS = (
    "00_how_it_works.ipynb",
    "01_engine_basics.ipynb",
    "02_what_the_llm_sees.ipynb",
    "03_eig_decomposition.ipynb",
    "04_reading_a_run_folder.ipynb",
)
CELL_TYPES = {"code", "markdown", "raw"}
OUTPUT_TYPES = {"stream", "display_data", "execute_result", "error"}
MACHINE_PATHS = ("/home/", "/tmp/")
# Binary image data is base64 text; it is skipped, because any short string can occur in it.
BINARY_MIME_PREFIXES = ("image/png", "image/jpeg", "image/gif")
KEY_SHAPES = (
    ("Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{8,}")),
    ("OpenAI key", re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_\-]{20,}")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}")),
    ("key assignment",
     re.compile(r"(?i)\b(?:api[_-]?key|secret|token)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}")),
)
MAX_BYTES = 1_000_000


def load(name):
    """Helper or fixture: load."""
    path = EXAMPLES / name
    assert path.is_file(), f"{path.name} is missing from examples/"
    return json.loads(path.read_text(encoding="utf-8"))


def joined(value):
    """nbformat stores a multi-line string as one string or as a list of lines; JSON data as is."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "".join(value)
    return json.dumps(value)


def output_texts(output):
    """Every text an output shows, without binary image data."""
    if output["output_type"] == "stream":
        yield joined(output["text"])
    elif output["output_type"] == "error":
        yield output.get("ename", "")
        yield output.get("evalue", "")
        yield "\n".join(output.get("traceback", []))
    else:
        for mime, value in output.get("data", {}).items():
            if not mime.startswith(BINARY_MIME_PREFIXES):
                yield joined(value)


def all_strings(value, skip_binary=False):
    """Every string in a JSON value, optionally without the base64 image data of outputs."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from all_strings(item, skip_binary)
    elif isinstance(value, dict):
        for key, item in value.items():
            if skip_binary and key.startswith(BINARY_MIME_PREFIXES):
                continue
            yield from all_strings(item, skip_binary)


def code_cells(notebook):
    """Helper or fixture: code cells."""
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_the_examples_folder_holds_exactly_these_notebooks():
    """Check that the examples folder holds exactly these notebooks."""
    assert sorted(path.name for path in EXAMPLES.glob("*.ipynb")) == sorted(NOTEBOOKS)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_nbformat_4_structure(name):
    """Check that nbformat 4 structure."""
    notebook = load(name)
    assert {"cells", "metadata", "nbformat", "nbformat_minor"} <= set(notebook)
    assert notebook["nbformat"] == 4
    assert isinstance(notebook["nbformat_minor"], int)
    assert notebook["metadata"].get("kernelspec", {}).get("name") == "python3", (
        "commit notebooks with the standard python3 kernelspec, not a machine-specific kernel")
    assert notebook["cells"], "empty notebook"
    for index, cell in enumerate(notebook["cells"]):
        where = f"{name} cell {index}"
        assert cell["cell_type"] in CELL_TYPES, where
        assert isinstance(cell["source"], (str, list)), where
        assert isinstance(cell.get("metadata", {}), dict), where
        if cell["cell_type"] == "code":
            assert isinstance(cell["outputs"], list), where
            assert "execution_count" in cell, where
            for output in cell["outputs"]:
                assert output["output_type"] in OUTPUT_TYPES, where
        else:
            assert "outputs" not in cell, where
    size = (EXAMPLES / name).stat().st_size
    assert size <= MAX_BYTES, f"{name} has {size:,} bytes; keep outputs small (limit {MAX_BYTES:,})"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_executed_top_to_bottom_in_one_kernel(name):
    """Check that executed top to bottom in one kernel."""
    cells = code_cells(load(name))
    counts = [cell["execution_count"] for cell in cells]
    assert cells and all(isinstance(count, int) for count in counts), (
        f"{name}: every code cell needs an execution count; commit it executed, with outputs")
    assert counts[0] == 1, f"{name}: execution counts start at {counts[0]}, not 1"
    assert all(a < b for a, b in pairwise(counts)), (
        f"{name}: execution counts {counts} are not increasing")
    assert any(cell["outputs"] for cell in cells), f"{name}: no outputs at all"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_error_outputs(name):
    """Check that no error outputs."""
    errors = [(index, output.get("ename"), output.get("evalue"))
              for index, cell in enumerate(load(name)["cells"]) if cell["cell_type"] == "code"
              for output in cell["outputs"] if output["output_type"] == "error"]
    assert not errors, f"{name}: error outputs {errors}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_outputs_print_no_machine_paths(name):
    """Check that outputs print no machine paths."""
    notebook = load(name)
    found = []
    for index, cell in enumerate(notebook["cells"]):
        for output in cell.get("outputs", []):
            for value in output_texts(output):
                found += [(index, marker) for marker in MACHINE_PATHS if marker in value]
        found += [(index, marker) for value in all_strings(cell.get("metadata", {}))
                  for marker in MACHINE_PATHS if marker in value]
    found += [("notebook metadata", marker) for value in all_strings(notebook["metadata"])
              for marker in MACHINE_PATHS if marker in value]
    assert not found, f"{name}: absolute paths of the executing machine in (cell, marker) {found}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_api_key_shaped_strings(name):
    """Check that no api key shaped strings."""
    found = [(label, match.group(0)[:12] + "...")
             for value in all_strings(load(name), skip_binary=True)
             for label, pattern in KEY_SHAPES for match in pattern.finditer(value)]
    assert not found, f"{name}: key-shaped strings {found}"
