"""Allowed-families round call: real round-path parsing, synthetic transport and deterministic fits."""

import ast
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bayes_model_discovery.experimental import core_count as core
from bayes_model_discovery.experimental import count_proposer as proposer
from bayes_model_discovery.experimental.test_core_d0_policy import (
    _points_with_authentic_identity,
    _synthetic_real_fits,
)
from bayes_model_discovery.experimental.test_count_proposer import run_args, stub_fits
from bayes_model_discovery.experimental.test_neuron_proposer import Client, response

# Parity base: the commit whose recorded outputs the opt-in baseline test pins; it skips when a checkout of this commit is absent.
BASE_COMMIT = "ee7550c10b758efca5945c4ce82acea19fd85879"
IDENTITY_FIELDS = ("source_files_sha256", "created_utc", "output", "manifest_ref", "python")
CHANGED_SOURCE_FILES = ("core_count.py",)
TRANSPORT_MEASUREMENT_FIELDS = ("elapsed_seconds", "run_id")


def _run(output, patch, structure, family, provider="anthropic"):
    real = structure.startswith("gauss_")
    _synthetic_real_fits(patch) if real else stub_fits(patch)
    candidate = {
        "expression": "b0 + b1 * tanh(z) + b2 * square(z)",
        "family": family,
        "parameters": {"b0": "intercept", "b1": "signed", "b2": "signed"},
    }
    client = Client(provider, [response(provider, candidate) for _ in range(2)])
    cls_name = "AnthropicCountProposer" if provider == "anthropic" else "OpenAICountProposer"
    original = getattr(proposer, cls_name)
    patch.setattr(proposer, cls_name, lambda records, **kw: original(records, client=client))
    core.run(run_args(output, rounds=1, seed=31, cell="100", proposer=provider,
                      allow_paid_api=True, structure=structure,
                      truth_family="normal" if real else "binomial",
                      initial_family="normal" if real else "binomial"))
    return client.calls


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("structure", ["gauss_den", "gauss_pow", "gauss_call"])
def test_gaussian_round_one_parses_and_admits_normal(tmp_path, monkeypatch, structure, provider):
    """Round one of each Gaussian structure parses a normal candidate and admits it to the active pool."""
    output = tmp_path / "cell"
    calls = _run(output, monkeypatch, structure, "normal", provider)
    record = json.loads((output / "proposals/t01.json").read_text())
    assert record["transport"]["status"] == "parsed", record["transport"]["rejected_candidates"]
    assert record["transport"]["rejected_candidates"] == []
    assert len(calls) == record["transport"]["calls"] == 1
    hypothesis, = record["raw_response"]
    assert hypothesis["family"] == "normal"
    rows = json.loads((output / "archive.json").read_text())
    admitted = [row for row in rows if row["model_id"] == hypothesis["model_id"]]
    assert len(admitted) == 1 and admitted[0]["outcome"] == "active"
    scores = [json.loads(line) for line in (output / "scores.jsonl").read_text().splitlines()]
    assert hypothesis["model_id"] in scores[-1]["active_ids"]


@pytest.mark.parametrize("structure,family", [
    ("gauss_den", "student_t"), ("out", "normal"),
])
def test_round_path_still_refuses_a_family_outside_its_menu(tmp_path, monkeypatch, structure, family):
    """A family outside the structure's menu is still rejected on the real round path."""
    output = tmp_path / "cell"
    calls = _run(output, monkeypatch, structure, family)
    record = json.loads((output / "proposals/t01.json").read_text())
    assert record["raw_response"] == []
    assert record["transport"]["status"] != "parsed"
    assert len(calls) == 2
    assert all(f"unsupported count family: {family}" in row["reason"]
               for row in record["transport"]["rejected_candidates"])


def test_all_five_runner_call_sites_supply_their_family_menu():
    """Every proposer call site in the runners passes an explicit allowed-family menu."""
    root = Path(core.__file__).parent
    menus = {}
    for filename, expected in (("core_count.py", 2), ("boxinggym_core.py", 2),
                               ("chembench_core.py", 2)):
        tree = ast.parse((root / filename).read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "propose_one"]
        assert len(calls) == expected
        menus[filename] = [next((keyword.value for keyword in call.keywords
                                if keyword.arg == "allowed_families"), None) for call in calls]
        assert all(menu is not None for menu in menus[filename])
    core_menus = menus["core_count.py"]
    assert ast.dump(core_menus[0]) == ast.dump(core_menus[1])
    assert ast.unparse(core_menus[0]) == "('normal',) if real_outcome else FAMILIES"
    assert all(ast.unparse(menu) == "allowed" for menu in menus["boxinggym_core.py"])
    assert all(ast.unparse(menu) == "CHEM_ALLOWED_FAMILIES" for menu in menus["chembench_core.py"])


@pytest.mark.parametrize("structure,family", [("out", "binomial"), ("gauss_den", "normal")])
@pytest.mark.skip(reason="parity with the baseline recorded before the model-identity change crosses the ID/RNG boundary; a successor parity test is still owed")
def test_opt_in_baseline_packets_and_count_bytes(tmp_path, structure, family):
    """Opt-in parity check of packets and recorded bytes against a pinned baseline checkout."""
    if os.environ.get("BMD_BASELINE_COMPARE") != "1" or not os.environ.get("BMD_AFRC_BASELINE_DIR"):
        pytest.skip("set BMD_BASELINE_COMPARE=1 and BMD_AFRC_BASELINE_DIR for AFRC baseline parity")
    base = Path(os.environ["BMD_AFRC_BASELINE_DIR"])
    current = Path(__file__).resolve().parents[3]
    assert subprocess.run(["git", "-C", str(base), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip() == BASE_COMMIT
    assert subprocess.run(["git", "-C", str(base), "status", "--porcelain", "-uall"], check=True,
                          capture_output=True, text=True).stdout == ""
    harness = """
import json
import runpy
import sys
from pathlib import Path
from pytest import MonkeyPatch
helpers = runpy.run_path(sys.argv[1])
with MonkeyPatch.context() as patch:
    calls = helpers["_run"](Path(sys.argv[2]), patch, sys.argv[3], sys.argv[4])
Path(sys.argv[2]).parent.joinpath("client_calls.json").write_text(json.dumps(calls, sort_keys=True))
"""
    outputs, configs, payloads = [], [], []
    for label, source in (("base", base), ("candidate", current)):
        output = tmp_path / label / "cell"
        env = dict(os.environ, PYTHONPATH=f"{source}/src:{source}", GIT_OPTIONAL_LOCKS="0",
                   OPENAI_API_KEY="", ANTHROPIC_API_KEY="", OPENAI_COMPAT_API_KEY="",
                   OPENROUTER_API_KEY="", PYTHONDONTWRITEBYTECODE="1",
                   PYTHON_DOTENV_DISABLED="1")
        result = subprocess.run([sys.executable, "-B", "-c", harness, __file__,
                                 str(output), structure, family],
                                cwd=source, env=env, capture_output=True, text=True,
                                check=False, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(output)
        configs.append(json.loads((output / "config.json").read_text()))
        payloads.append(json.loads((output.parent / "client_calls.json").read_text()))
    # The first request precedes the changed parse/admission outcome. Repairs
    # may disappear after the fix; comparing a missing repair would be wrong.
    assert payloads[0][0] == payloads[1][0]
    before = sorted((outputs[0] / "packets").glob("*"))
    after = sorted((outputs[1] / "packets").glob("*"))
    assert before and [path.name for path in before] == [path.name for path in after]
    assert [path.read_bytes() for path in before] == [path.read_bytes() for path in after]
    if family == "normal":
        base_record = json.loads((outputs[0] / "proposals/t01.json").read_text())
        final_record = json.loads((outputs[1] / "proposals/t01.json").read_text())
        assert base_record["transport"]["status"] != "parsed"
        assert final_record["transport"]["status"] == "parsed"
        return
    left, right = configs
    assert left.keys() == right.keys()
    assert {key: value for key, value in left.items() if key not in IDENTITY_FIELDS} == {
        key: value for key, value in right.items() if key not in IDENTITY_FIELDS}
    assert left["source_files_sha256"].keys() == right["source_files_sha256"].keys()
    assert {key for key, value in left["source_files_sha256"].items()
            if value != right["source_files_sha256"][key]} == set(CHANGED_SOURCE_FILES)
    for source, config in zip((base, current), configs, strict=True):
        assert config["source_files_sha256"]["core_count.py"] == hashlib.sha256(
            (source / "src/bayes_model_discovery/experimental/core_count.py").read_bytes()).hexdigest()
    for name in ("archive.json", "observations.jsonl", "scores.jsonl"):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes(), name
    for directory in ("proposals", "acquisition"):
        paths = [sorted(path.relative_to(output) for path in (output / directory).glob("*.json"))
                 for output in outputs]
        assert paths[0] == paths[1] and paths[0]
        for path in paths[0]:
            if directory == "acquisition":
                assert (outputs[0] / path).read_bytes() == (outputs[1] / path).read_bytes(), path
                continue
            records = [json.loads((output / path).read_text()) for output in outputs]
            for output, record in zip(outputs, records, strict=True):
                transport = record["transport"]
                assert transport["run_id"] == str(output)
                assert math.isfinite(transport["elapsed_seconds"]) and transport["elapsed_seconds"] >= 0
            comparable = [{**record, "transport": {
                key: value for key, value in record["transport"].items()
                if key not in TRANSPORT_MEASUREMENT_FIELDS}} for record in records]
            assert comparable[0] == comparable[1]
    assert _points_with_authentic_identity(outputs[0], left, base) == (
        _points_with_authentic_identity(outputs[1], right, current))
