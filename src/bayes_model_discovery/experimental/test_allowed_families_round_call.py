"""Allowed-families round call: real round-path parsing, synthetic transport and deterministic fits."""

import ast
import json
from pathlib import Path

import pytest

from bayes_model_discovery.experimental import core_count as core
from bayes_model_discovery.experimental import count_proposer as proposer
from bayes_model_discovery.experimental.test_core_d0_policy import (
    _synthetic_real_fits,
)
from bayes_model_discovery.experimental.test_count_proposer import run_args, stub_fits
from bayes_model_discovery.experimental.test_neuron_proposer import Client, response


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


