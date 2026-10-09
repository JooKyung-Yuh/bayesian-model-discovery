"""Tests for input-name masking: synthetic replies, true alias identities, unchanged numerical inputs."""

import csv
import hashlib
import io
import json
import os
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


def exercise_fake_cell(output, task, masked, prompt_path):
    """Full runner and real SMC; only the external LLM client is synthetic."""
    spec = bg.TASKS[task]
    name = "X1" if masked else spec.input_name
    family = bg.interface_families(spec.outcome_type, True,
                                   zero_inflation=spec.zero_inflation_admissible)[0]

    def candidate(expression):
        """Test helper: candidate."""
        return {"expression": expression, "family": family,
                "parameters": {key: "intercept" if key == "b0" else "signed"
                               for key in re.findall(r"\bb\d+\b", expression)}}

    replies = [candidate("b0"), candidate("b0"), candidate(f"b0+b1*{name}z"),
               candidate("+".join(f"b{i}" for i in range(20))),
               candidate(f"b0+b1*square({name}z)"), candidate("b0+b1*Q9"),
               candidate(f"b0+b1*{name}z*square({name}z)"),
               {"expression": f"b0+{name}", "family": family,
                "parameters": {"b0": "intercept", name: "signed"}},
               candidate(f"b0+b1*tanh({name}z)")]
    calls = []

    def create(**body):
        """Test helper: create."""
        calls.append(body)
        assert replies, "unexpected extra LLM request"
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(replies.pop(0)))],
            usage=SimpleNamespace(model_dump=lambda **kw: {"input_tokens": 1, "output_tokens": 1}),
            id="synthetic-bxm", model="synthetic-bxm-model", stop_reason="end_turn")

    original = cp.AnthropicCountProposer
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cp, "AnthropicCountProposer",
                      lambda records, **kw: original(records, client=client, **kw))
        bg.run(bg_run_args(
            output, task=task, rounds=3, seed=31, cell="111", proposer="anthropic",
            allow_paid_api=True, run_label="development", initial_catalogue="off",
            initial_proposals=2, burst_concurrency=1, prompt_variant=str(prompt_path),
            **({"mask_input_names": True} if masked else {})))
    assert not replies and len(calls) == 9
    return calls


def test_paper_anthropic_variant_without_manifest_is_development(tmp_path, monkeypatch):
    """Real paper-budget inference with masked input names; the only substituted component is the client."""
    prompt_path = os.environ.get("BMD_BXM_PROMPT")
    if not prompt_path:
        pytest.skip("set BMD_BXM_PROMPT to the read-only FA-C prompt")
    prompt_path = Path(prompt_path)
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

    prompt = os.environ.get("BMD_BXM_PROMPT")
    if not prompt:
        pytest.skip("set BMD_BXM_PROMPT to the read-only FA-C prompt")
    prompt = Path(prompt)
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


TIME_KEYS = ("created_utc", "fit_seconds", "wall_seconds", "user_cpu_seconds",
             "system_cpu_seconds", "timing_seconds", "_t_sampling", "elapsed_seconds", "seconds")
PACKET_IDENTITY_KEYS = ("packet_bytes", "packet_sha256", "packet_digest")
SOURCE_CHANGES = ("boxinggym_core.py",)


def compare_artifacts(left, right, *, masking=False, sources=None):
    """Read-only comparison, with literal provenance/time exemptions, never rounding."""
    configs = [json.loads((root / "config.json").read_text()) for root in (left, right)]
    aliases, ids = {}, {}
    if masking:
        for root in (left, right):
            verify_packet_identities(root)
        assert set(configs[1]) - set(configs[0]) == {"mask_input_names", "input_name_map"}
        assert not set(configs[0]) - set(configs[1])
        assert configs[1]["mask_input_names"] is True
        aliases = dict(configs[1]["input_name_map"])
        for alias, native in list(aliases.items()):
            for suffix in ("01", "z"):
                aliases[alias + suffix] = native + suffix
        pairs = [json.loads(row) for row in (right / "masked_model_ids.jsonl").read_text().splitlines()]
        ids = {row["model_id"]: row["rng_model_id"] for row in pairs}
        assert len(ids) == len(pairs)
    else:
        assert configs[0].keys() == configs[1].keys()
        for config in configs:
            assert "mask_input_names" not in config and "input_name_map" not in config
        assert sources is not None
    replacements = {str(right): str(left)}
    if sources:
        replacements[str(sources[1])] = str(sources[0])
        for config, source in zip(configs, sources, strict=True):
            for name, digest in config["source_files_sha256"].items():
                actual = source / "src/bayes_model_discovery/experimental" / name
                assert hashlib.sha256(actual.read_bytes()).hexdigest() == digest, name
        assert configs[0]["source_files_sha256"].keys() == configs[1]["source_files_sha256"].keys()
        changed = {key for key in configs[0]["source_files_sha256"] if
                   configs[0]["source_files_sha256"][key] != configs[1]["source_files_sha256"][key]}
        assert changed == set(SOURCE_CHANGES)
        for name in SOURCE_CHANGES:
            replacements[configs[1]["source_files_sha256"][name]] = configs[0]["source_files_sha256"][name]
    replacements.update(ids)
    shared_ties = []

    def text(value, side):
        """Test helper: text."""
        if side == 1:
            for new, old in sorted(replacements.items(), key=lambda item: -len(item[0])):
                value = value.replace(new, old)
            if masking:
                value = re.sub(r"[A-Za-z0-9_]+", lambda match: aliases.get(match[0], match[0]), value)
        return value

    def normalized(value, side, location):
        """Test helper: normalized."""
        if isinstance(value, dict):
            return {text(key, side): normalized(item, side, f"{location}/{key}")
                    for key, item in value.items() if key not in TIME_KEYS
                    and not (masking and key in PACKET_IDENTITY_KEYS)
                    and not (masking and location == "config.json"
                             and key in ("mask_input_names", "input_name_map"))}
        if isinstance(value, list):
            result = [normalized(item, side, f"{location}/{i}") for i, item in enumerate(value)]
            # belief_packet's shared archive order stays masked, as the name-masking design requires.
            if masking and location.endswith("/archive") and all(
                    isinstance(item, dict) and "model_id" in item and "proposed_round" in item
                    for item in result):
                ordered = sorted(result, key=lambda item: (item["proposed_round"], item["model_id"]))
                if result != ordered:
                    shared_ties.append({"location": location, "side": side,
                                        "source": "belief_packet.build_belief_packet archive sort",
                                        "tie_field": "proposed_round"})
                return ordered
            return result
        if isinstance(value, str):
            # Request messages and raw responses embed JSON; compare that JSON structurally.
            if value.startswith(("{", "[")):
                try:
                    return normalized(json.loads(value), side, location + "/embedded")
                except json.JSONDecodeError:
                    pass
            return text(value, side)
        return value

    def files(root, side):
        """Test helper: files."""
        return {text(str(path.relative_to(root)), side): path for path in root.rglob("*")
                if path.is_file() and not (masking and path.name == "masked_model_ids.jsonl")}

    paths = [files(left, 0), files(right, 1)]
    assert paths[0].keys() == paths[1].keys(), (paths[0].keys() ^ paths[1].keys())
    report = []
    for name in sorted(paths[0]):
        a, b = paths[0][name], paths[1][name]
        if a.suffix == ".npz":
            with np.load(a) as x, np.load(b) as y:
                assert x.files == y.files, name
                for key in x.files:
                    assert x[key].dtype == y[key].dtype, (name, key)
                    assert x[key].shape == y[key].shape, (name, key)
                    assert x[key].tobytes() == y[key].tobytes(), (name, key)
            comparison = "exact_array_bytes"
        else:
            def read(path):
                """Test helper: read."""
                if path.suffix == ".json":
                    return json.loads(path.read_text())
                if path.suffix == ".jsonl":
                    return [json.loads(row) for row in path.read_text().splitlines()]
                if path.suffix == ".csv":
                    return list(csv.reader(io.StringIO(path.read_text())))
                return path.read_text()
            x, y = normalized(read(a), 0, name), normalized(read(b), 1, name)
            assert x == y, (name, first_difference(x, y))
            comparison = "exact_after_declared_identity_and_time_normalization"
        report.append({"file": name, "comparison": comparison,
                       "left_sha256": hashlib.sha256(a.read_bytes()).hexdigest(),
                       "right_sha256": hashlib.sha256(b.read_bytes()).hexdigest()})
    return {"left": str(left), "right": str(right), "artifacts": report,
            "time_keys": list(TIME_KEYS), "shared_ties": shared_ties,
            "packet_identity_keys": list(PACKET_IDENTITY_KEYS) if masking else [],
            "unexpected_differences": []}


def first_difference(left, right, location=""):
    """Test helper: first difference."""
    if type(left) is not type(right):
        return location, left, right
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return location, sorted(left.keys() ^ right.keys())
        for key in left:
            if left[key] != right[key]:
                return first_difference(left[key], right[key], location + "/" + key)
    elif isinstance(left, list):
        if len(left) != len(right):
            return location, len(left), len(right)
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            if a != b:
                return first_difference(a, b, f"{location}/{i}")
    return location, left, right


def verify_packet_identities(root):
    """Test helper: verify packet identities."""
    packets = {hashlib.sha256(path.read_bytes()).hexdigest(): len(path.read_bytes())
               for path in (root / "packets").glob("*.json")}

    def check(value):
        """Test helper: check."""
        if isinstance(value, dict):
            for key in ("packet_digest", "packet_sha256"):
                if key in value:
                    assert value[key] in packets, (key, value[key])
                    if "packet_bytes" in value:
                        assert value["packet_bytes"] == packets[value[key]]
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    for path in root.rglob("*.json"):
        check(json.loads(path.read_text()))


def scan_requests(visible, masked, prompt_path):
    """Test helper: scan requests."""
    prompt = prompt_path.read_bytes().decode()
    config = json.loads((masked / "config.json").read_text())
    requests = sorted((masked / "llm_transport").glob("*.request.json"))
    packets = sorted((masked / "packets").glob("*.json"))
    assert len(requests) == 6 and len(packets) == 6
    routes = []
    for path in requests:
        record = json.loads(path.read_text())
        unmasked = json.loads((visible / "llm_transport" / path.name).read_text())
        if "burst_w0" in path.name:
            route, suffix, key = "burst", cp.BURST_PARAGRAPH, "burst_prompt_sha256"
        elif "burst_w1" in path.name:
            route, suffix, key = "refill", cp.BURST_REFILL_PARAGRAPH, "burst_refill_prompt_sha256"
        else:
            route, suffix, key = "round", "", "prompt_sha256"
        expected = prompt + suffix
        assert hashlib.sha256(expected.encode()).hexdigest() == config[key]
        assert len(record["bodies"]) == len(unmasked["bodies"])
        for body, plain in zip(record["bodies"], unmasked["bodies"], strict=True):
            assert body["system"].encode() == expected.encode() == plain["system"].encode()
            check_no_leak(json.dumps({key: body[key] for key in ("system", "messages")}))
        packet = record["bodies"][0]["messages"][0]["content"].encode()
        assert record["identity"]["packet_sha256"] == hashlib.sha256(packet).hexdigest()
        assert record["identity"]["packet_bytes"] == len(packet)
        routes.append({"file": path.name, "route": route, "bodies": len(record["bodies"]),
                       "system_sha256": hashlib.sha256(expected.encode()).hexdigest()})
    for path in packets:
        check_no_leak(path.read_text())
    pairs = [json.loads(row) for row in (masked / "masked_model_ids.jsonl").read_text().splitlines()]
    native_only = {row["rng_model_id"] for row in pairs} - {row["model_id"] for row in pairs}
    assert native_only
    for path in [*masked.rglob("*.json"), *masked.rglob("*.jsonl")]:
        if path.name != "masked_model_ids.jsonl":
            assert not any(key in path.read_text() for key in native_only), path
    for t in (1, 2, 3):
        record = json.loads((masked / f"proposals/t{t:02d}.json").read_text())
        assert record["transport"]["repairs"] == 1
    unknown = json.loads((masked / "llm_transport/round_02.request.json").read_text())
    assert "Q9" in json.dumps(unknown["bodies"][1]["messages"])
    collision = json.loads((masked / "llm_transport/round_03.request.json").read_text())
    assert "invalid or reserved coefficient name" in json.dumps(collision["bodies"][1]["messages"])
    return {"request_files": len(requests), "packet_files": len(packets), "routes": routes,
            "bodies": sum(row["bodies"] for row in routes), "forbidden_matches": [],
            "native_only_id_matches": [], "masked_id_pairs": len(pairs)}


def check_no_leak(text):
    """Test helper: check no leak."""
    assert not FORBIDDEN_NAMES.intersection(re.findall(r"[A-Za-z0-9_]+", text))
    assert not any(stem in text.lower() for stem in FORBIDDEN_STEMS)


@pytest.mark.parametrize("task", TASKS)
def test_opt_in_full_cell_artifact_equivalence_and_leakage(task):
    """Test: opt in full cell artifact equivalence and leakage."""
    visible, masked, prompt = (os.environ.get(key) for key in (
        "BMD_BXM_VISIBLE", "BMD_BXM_MASKED", "BMD_BXM_PROMPT"))
    if not all((visible, masked, prompt)):
        pytest.skip("set BMD_BXM_VISIBLE, BMD_BXM_MASKED, BMD_BXM_PROMPT for fresh mock-cell evidence")
    compare_artifacts(Path(visible) / task, Path(masked) / task, masking=True)
    scan_requests(Path(visible) / task, Path(masked) / task, Path(prompt))
