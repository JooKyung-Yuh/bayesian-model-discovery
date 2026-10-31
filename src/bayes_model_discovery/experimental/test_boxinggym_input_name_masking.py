"""Tests for input-name masking: synthetic replies, true alias identities, unchanged numerical inputs."""

import csv
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from . import boxinggym_core as bg
from . import count_proposer as cp
from .count_models import judge_candidate
from .test_named_inputs import bg_run_args

#: The paper's system prompt, shipped with this repository (the file the paper runs sent byte for byte).
FA_C_PROMPT = Path(__file__).resolve().parents[3] / "tools" / "fb1_launch_c" / "proposer_system_v2_9_FA-C.txt"

TASKS = ("dugongs", "death", "peregrines", "hyperbolic", "survival")
PARAMETERS = (("b0", "intercept"), ("b1", "signed"))
FORBIDDEN_NAMES = {
    name + suffix for name in ("age", "t", "tt", "iR", "dR", "D", "time", "metastasized")
    for suffix in ("", "01", "z")
}
FORBIDDEN_STEMS = ("dugong", "peregrine", "falcon", "hyperbolic", "discount", "reward",
                   "delay", "survival", "patient", "cancer", "metasta", "tumor", "death", "mortal")


@pytest.fixture(autouse=True)
def reset_prompt():
    """Test helper: reset prompt."""
    cp.select_prompt_version(None)
    cp.set_prompt_variant(None)
    yield
    cp.select_prompt_version(None)
    cp.set_prompt_variant(None)


@pytest.mark.parametrize("task", TASKS)
def test_task_aliases_preserve_all_numerical_declarations(tmp_path, task):
    """Test: task aliases preserve all numerical declarations."""
    spec = bg.TASKS[task]
    before = spec.interface()
    mask = bg.InputNameMask(spec, tmp_path / "ids.jsonl")
    masked = mask.spec
    assert spec.interface() == before
    assert mask.input_name_map == {f"X{i}": name for i, name in enumerate(spec.input_names, 1)}
    assert masked.input_names == tuple(mask.input_name_map)
    assert masked.integer_inputs == tuple(
        alias for name in spec.integer_inputs for alias, native in mask.input_name_map.items()
        if native == name)
    for left, right in zip(spec.inputs, masked.inputs, strict=True):
        assert {k: v for k, v in left.items() if k != "name"} == {
            k: v for k, v in right.items() if k != "name"}
    assert masked.d0() == spec.d0()
    np.testing.assert_array_equal(masked.designs(16), spec.designs(16))
    for originals, aliases in ((spec.initial_structures(), masked.initial_structures()),
                               (spec.mock_templates(), masked.mock_templates())):
        assert tuple(bg.retarget(form, mask.native_names) for form in aliases) == originals
    for alias, expression in masked.derived_inputs.items():
        assert bg.retarget(expression, mask.native_names) == bg.retarget(
            spec.derived_inputs[mask.native_names[alias]], {})
    assert not (tmp_path / "ids.jsonl").exists()


def test_categorical_levels_and_explicit_forms_keep_declaration_order(tmp_path):
    """Test: categorical levels and explicit forms keep declaration order."""
    spec = replace(bg.TASKS["irt"], input_name="second",
                   extra_inputs=({"name": "first", "range": [0, 5]},),
                   categorical_inputs=(("second", 6), ("first", 6)),
                   integer_inputs=("second", "first"),
                   initial_forms=("b0+a[second]+b[first]",),
                   mock_forms=("b0+a[first]",))
    mask = bg.InputNameMask(spec, tmp_path / "ids.jsonl")
    assert mask.input_name_map == {"X1": "second", "X2": "first"}
    assert mask.spec.derived_inputs == {}
    assert mask.spec.initial_forms == ("b0+a[X1]+b[X2]",)
    assert mask.spec.mock_forms == ("b0+a[X2]",)
    for left, right in zip(spec.inputs, mask.spec.inputs, strict=True):
        assert left["kind"] == right["kind"] == "categorical"
        assert left["levels"] == right["levels"]


@pytest.mark.parametrize("family", ["binomial", "lognormal", "heteroscedastic_normal"])
def test_rng_identity_preserves_masked_model_and_native_prior_draws(tmp_path, family):
    """Test: rng identity preserves masked model and native prior draws."""
    spec = bg.TASKS["death"]
    mask = bg.InputNameMask(spec, tmp_path / "ids.jsonl")
    if family == "binomial":
        cls, native_extra, alias_extra = bg.Hypothesis, (), ()
    elif family == "lognormal":
        cls, native_extra, alias_extra = bg.ScalarHypothesis, (), ()
    else:
        cls = bg.RealHypothesis
        native_extra, alias_extra = ("a0+a1*tz",), ("a0+a1*X1z",)
    native = cls("b0+b1*tanh(tz)", family, PARAMETERS, ("t", "tz"), *native_extra)
    masked = cls("b0+b1*tanh(X1z)", family, PARAMETERS, ("X1", "X1z"), *alias_extra)
    before = masked.as_dict()
    with ThreadPoolExecutor(max_workers=4) as pool:
        keys = list(pool.map(lambda _: mask.rng_model_id(masked), range(12)))
    assert keys == [native.model_id] * 12
    assert masked.model_id != native.model_id and masked.as_dict() == before
    assert [json.loads(row) for row in mask.sidecar.read_text().splitlines()] == [
        {"model_id": masked.model_id, "rng_model_id": native.model_id}]
    expected = bg.prior_particles(native, 32)
    actual = bg.prior_particles(masked, 32, rng_model_id=native.model_id)
    assert expected.keys() == actual.keys()
    for name in expected:
        np.testing.assert_array_equal(expected[name], actual[name])


def test_rng_identity_binds_proposed_priors_and_categorical_inputs(tmp_path):
    """Test: rng identity binds proposed priors and categorical inputs."""
    spec = bg.TASKS["irt"]
    mask = bg.InputNameMask(spec, tmp_path / "ids.jsonl")
    parameters = (("b0", "intercept"), ("a", "signed"))
    priors = (("b0", ("Normal", 0.0, 1.0)),)
    native = bg.Hypothesis("b0+a[student]", "bernoulli", parameters, spec.input_names,
                           spec.categorical_levels(), priors)
    masked = bg.Hypothesis("b0+a[X1]", "bernoulli", parameters, mask.spec.input_names,
                           mask.spec.categorical_levels(), priors)
    assert mask.rng_model_id(masked) == native.model_id
    expected = bg.prior_particles(native, 12)
    actual = bg.prior_particles(masked, 12, rng_model_id=native.model_id)
    for name in expected:
        np.testing.assert_array_equal(expected[name], actual[name])


@pytest.mark.parametrize("options", [{"truth_in_pool": True}, {"inject_oracle_at": 1},
                                     {"run_label": "production"}])
def test_refusals_precede_output_and_client(tmp_path, monkeypatch, options):
    """Test: refusals precede output and client."""
    def forbidden(*args, **kwargs):
        """Test helper: forbidden."""
        pytest.fail("refused run reached provider setup")
    monkeypatch.setattr(cp, "AnthropicCountProposer", forbidden)
    output = tmp_path / "cell"
    with pytest.raises(ValueError, match="--mask-input-names"):
        bg.run(bg_run_args(output, mask_input_names=True, **options))
    assert not output.exists()


def test_implicit_production_is_also_refused_before_output(tmp_path, monkeypatch):
    """Test: implicit production is also refused before output."""
    monkeypatch.setattr(bg, "bmd_label_receipt", lambda *args, **kw: {"run_label": "production"})
    with pytest.raises(ValueError, match="--mask-input-names"):
        bg.run(bg_run_args(tmp_path / "cell", mask_input_names=True))
    assert not (tmp_path / "cell").exists()


@pytest.mark.parametrize("mask", [False, True])
def test_cli_suppresses_unstated_option(monkeypatch, tmp_path, mask):
    """Test: cli suppresses unstated option."""
    seen = []
    monkeypatch.setattr(bg, "run", lambda args: seen.append(vars(args)))
    monkeypatch.setattr(sys, "argv", ["boxinggym_core", "--task", "death",
                                     "--output", str(tmp_path / "cell"),
                                     *(["--mask-input-names"] if mask else [])])
    bg.main()
    assert ("mask_input_names" in seen[0]) is mask
    if mask:
        assert seen[0]["mask_input_names"] is True


def test_input_coefficient_collision_uses_existing_nonleaking_error():
    """Test: input coefficient collision uses existing nonleaking error."""
    accepted, reason = judge_candidate(
        {"expression": "b0+X1", "family": "binomial",
         "parameters": {"b0": "intercept", "X1": "signed"}},
        "111", (), "binomial", input_names=("X1", "X101", "X1z"),
        allowed_families=("binomial",),
    )
    assert not accepted and reason == "invalid or reserved coefficient name"
    assert not FORBIDDEN_NAMES.intersection(re.findall(r"[A-Za-z0-9_]+", reason))


@pytest.mark.usefixtures("require_boxinggym_source")
def test_paper_anthropic_variant_without_manifest_is_development(tmp_path, monkeypatch):
    """Real paper-budget inference with masked input names; the only substituted component is the client."""
    prompt_path = FA_C_PROMPT
    replies = [
        {"expression": expression, "family": "bernoulli", "parameters": dict(PARAMETERS)}
        for expression in ("b0+b1*tanh(X1z)", "b0+b1*tanh(X2z)")
    ]
    calls = []

    def create(**body):
        """Test helper: create."""
        calls.append(body)
        assert replies, "unexpected extra LLM request"
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(replies.pop(0)))],
            usage=SimpleNamespace(model_dump=lambda **kw: {"input_tokens": 1, "output_tokens": 1}),
            id="synthetic-paper-bxm", model="synthetic-paper-model", stop_reason="end_turn")

    original = cp.AnthropicCountProposer
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(cp, "AnthropicCountProposer",
                        lambda records, **kw: original(records, client=client, **kw))
    output = tmp_path / "paper_cell"
    args = bg_run_args(output, task="hyperbolic", profile="paper", rounds=2, seed=31,
                       cell="111", proposer="anthropic", llm_model="synthetic-paper-model",
                       allow_paid_api=True, run_label="development", mask_input_names=True,
                       prompt_variant=str(prompt_path))
    assert getattr(args, "manifest_ref", None) is None
    bg.run(args)
    assert not replies and len(calls) == 2
    config = json.loads((output / "config.json").read_text())
    receipt = config["label_receipt"]
    assert config["label"] == receipt["run_label"] == "development"
    assert receipt["manifest_path"] is None
    assert receipt["binding"]["settings"]["particles"] == 500
    assert receipt["binding"]["settings"]["chains"] == 2
    with (output / "points.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows and {row["label"] for row in rows} == {"development"}
    requests = sorted((output / "llm_transport").glob("*.request.json"))
    assert len(requests) == 2
    for path in requests:
        for body in json.loads(path.read_text())["bodies"]:
            assert body["system"].encode() == prompt_path.read_bytes()
            assert hashlib.sha256(body["system"].encode()).hexdigest() == config["prompt_sha256"]
            check_no_leak(json.dumps({key: body[key] for key in ("system", "messages")}))
    for path in (output / "packets").glob("*.json"):
        check_no_leak(path.read_text())


@pytest.mark.parametrize("masked", [False, True])
def test_paper_variant_with_verified_production_manifest_is_refused(tmp_path, monkeypatch, masked):
    """Test: paper variant with verified production manifest is refused."""
    from .run_label_policy import bmd_label_receipt, bmd_runtime_settings
    from .test_run_label_policy import registration

    prompt = FA_C_PROMPT
    output = tmp_path / "never"
    args = bg_run_args(output, task="hyperbolic", profile="paper", rounds=2, seed=31,
                       cell="111", proposer="anthropic", llm_model="synthetic-paper-model",
                       allow_paid_api=True, run_label="production", prompt_variant=str(prompt),
                       **({"mask_input_names": True} if masked else {}))
    _, identity = cp.select_run_prompt(args)
    declared = SimpleNamespace(**{**vars(args), **identity})
    ref, _, _ = registration(
        tmp_path, bundle="boxinggym", task=args.task, seed=args.seed, cell=args.cell,
        proposer=args.proposer, model=args.llm_model,
        settings=bmd_runtime_settings("boxinggym", declared))
    args.manifest_ref = declared.manifest_ref = json.dumps(ref)
    receipt = bmd_label_receipt("boxinggym", declared, task=args.task,
                                metadata=bg.compat_config(args))
    assert receipt["run_label"] == "production" and receipt["manifest_sha256"] == ref["sha256"]

    def forbidden(*args, **kwargs):
        """Test helper: forbidden."""
        pytest.fail("refused run reached the external client")

    monkeypatch.setattr(cp, "AnthropicCountProposer", forbidden)
    with pytest.raises(ValueError, match="--mask-input-names" if masked else "--prompt-variant"):
        bg.run(args)
    assert not output.exists()
    if not masked:
        assert not cp.prompt_variant_active()


def check_no_leak(text):
    """Test helper: check no leak."""
    assert not FORBIDDEN_NAMES.intersection(re.findall(r"[A-Za-z0-9_]+", text))
    assert not any(stem in text.lower() for stem in FORBIDDEN_STEMS)


