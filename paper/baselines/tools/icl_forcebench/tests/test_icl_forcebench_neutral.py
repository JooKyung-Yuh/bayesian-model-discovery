"""Anonymized-name interface (2026-10-11): the opt-in neutral interface of the ForceBench ICL tool. Synthetic clients only, no
provider call. The neutral texts have 0 hits under the engine's vocabulary rule (audits/fb_vocabulary_audit.py, a byte
copy of engine commit 48c84d5); the default requests are the released ones byte for byte (sha256 pinned from base 7b28b76);
the receipts, the settings and the seal draft name the mode."""

import hashlib
import importlib.util
import json
import re

import numpy as np
import pytest

from tools.icl_forcebench import icl_forcebench as module
from tools.icl_forcebench.protocol import (
    T_OBS,
    Context,
    Observation,
    forbidden_words,
    interface_violations,
    json_bytes,
    prompt_parts,
    render_prediction,
    render_selection,
    repair_request,
)

pytest.importorskip("jax")
import os  # noqa: E402

if not os.environ.get("DP_SCIENCEAGENT_DIR"):
    pytest.skip("DP_SCIENCEAGENT_DIR names the DiscoverPhysics/ScienceAgent directory", allow_module_level=True)

from tools.icl_forcebench.audits import fb_vocabulary_audit as audit  # noqa: E402

REGISTRATION = module.ROOT.parent / "icl_baseline" / "prices.json"
CONTEXT = Context(p_range=(0.1, 10.0), times=tuple(0.5 * k for k in range(1, 11)), pos0=(-2.5, -6.0), vel0=(-0.1, 2.0))
ENGINE_REFERENCE = module.ROOT / "tests" / "engine_reference" / "fb1_neutral.py"
#: The engine branch's file was 48c84d5:tools/fb1_snapshot/fb1_neutral.py, sha256 102222a8...; the published reference copy
#: differs from it in two docstring lines only (an internal unit name and a development-tree path removed), and this is
#: its digest.
ENGINE_NEUTRAL_SHA256 = "fe1ffb3b97ecaed3a3a14e77d8a166a11a3f9c4e5574e86f900aeb75a3cc3f5c"
#: The engine branch's audit was 48c84d5:audits/fb_vocabulary_audit.py, sha256 b5a012ea...; the published copy under
#: audits/ differs from it in two docstring lines only (internal unit names removed), and this is its digest.
ENGINE_AUDIT_SHA256 = "449709928e455016767a6b69fa34203233c1d9fb8daad05fbf7d3186a8d08c5b"
#: sha256 of json_bytes(request_body(...)) of these renders on base 7b28b76 (released tool), computed 2026-10-11
BASE_DEFAULT_BODIES = {
    "prediction": "d28687fdd1049fb9733bdaa229706c47a0d65ec5643fe2b4c399347cb2b5f936",
    "selection": "2fe3c6761f108bf5a2a2a4f409045a8a9199f695f11c251b2def2fdee70eb51b",
    "prediction_repair": "2e3d3da0ad208fbb1c8922943e028d982ca07821a4b4e0879cd45d6290b98199",
    "selection_repair": "d3d6423d9f146485194e3b21dd9b5d8fe1f0a83d35859b66a60112a2922ee031",
}
ERRORS = ["invalid JSON object", "empty text", "response stopped at max_tokens", "unexpected keys",
          "q1: requires a list of 10 [u, v] pairs", "q1: each element must be a pair [u, v]", "q2: expected exactly once",
          "q1: each pair must hold two finite JSON numbers", "root must contain only one predictions object",
          "root must contain only one choice string", "choice must be one listed candidate id"]


def observations(n=3):
    return [Observation(i, (1.0 + i, 2.0 + i), tuple((float(k) + i, -float(k) * 0.5) for k in range(T_OBS)), "d0" if i < 2 else f"round {i - 1}")
            for i in range(n)]


def renders(interface):
    rows = observations()
    pred, _ = render_prediction(rows, [("head_1", (1.0, 1.0)), ("shift_4", (4.0, 5.0))], CONTEXT, interface)
    sel, _ = render_selection(rows, np.array([[0.5, 9.5], [3.0, 3.0], [7.25, 0.125]]), CONTEXT, interface)
    return {"prediction": pred, "selection": sel,
            "prediction_repair": repair_request(pred, "oops", "invalid JSON object", "prediction", 2, interface),
            "selection_repair": repair_request(sel, "oops", "choice must be one listed candidate id", "selection", 3, interface)}


def user_texts(prompt):
    return [prompt["system"], *(m["content"] for m in prompt["messages"] if m["role"] == "user")]


def body_sha(prompt):
    return hashlib.sha256(json_bytes(module.request_body(prompt, 32000))).hexdigest()


def load_engine_neutral():
    spec = importlib.util.spec_from_file_location("fb1_neutral_engine_reference", ENGINE_REFERENCE)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


def test_unset_interface_is_the_released_tool_byte_for_byte(monkeypatch):
    monkeypatch.delenv(module.INTERFACE_ENV, raising=False)
    assert module.interface_mode() == "default"
    for interface in (None, "default"):
        prompts = renders("default") if interface else {
            "prediction": render_prediction(observations(), [("head_1", (1.0, 1.0)), ("shift_4", (4.0, 5.0))], CONTEXT)[0]}
        for name, prompt in prompts.items():
            assert body_sha(prompt) == BASE_DEFAULT_BODIES[name], name
    assert "interface" not in module.settings("gravity") and module.settings("gravity") == module.settings("gravity", interface="default")
    assert module.dependencies() == module.dependencies("default")
    assert not any("neutral" in str(p) for p in module.dependencies()) and module.NEUTRAL_ADDENDUM not in module.dependencies()


def test_neutral_texts_have_zero_vocabulary_hits_and_keep_the_numbers():
    # the templates without their placeholders (<times> is filled with numbers; a template is never sent as it is)
    templates = [re.sub(r"<[^<>\n]+>", "", part) for kind in ("prediction", "selection") for part in prompt_parts(kind, "neutral")]
    assert all(audit.hits_in(text) == [] and interface_violations(text, "neutral") == [] for text in templates)
    neutral, default = renders("neutral"), renders("default")
    for name, prompt in neutral.items():
        for text in user_texts(prompt):
            assert audit.hits_in(text) == [] and interface_violations(text, "neutral") == [], (name, text)
    # every repair sentence the tool can build (each validation error) is neutral too
    for error in ERRORS:
        for kind, count in (("prediction", 2), ("selection", 3)):
            repair = repair_request(neutral[kind], "x", error, kind, count, "neutral")["messages"][-1]["content"]
            assert audit.hits_in(repair) == [] and interface_violations(repair, "neutral") == []
    # the default wording does hit (for the record): time / times
    assert set(audit.hits_in(default["prediction"]["messages"][0]["content"])) == {"time", "times"}
    text = neutral["prediction"]["messages"][0]["content"]
    assert "s0a = (-2.5, -6.0) and s0b = (-0.1, 2.0), each a length 2 vector" in text and "[0.1, 10.0]" in text
    assert "X4 values at the measurements: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0." in text
    assert "rate of change" not in text and "u0" not in text and "t1" not in text and "<" not in text.replace("<candidate id>", "")
    # the data tables are the default's, character for character
    for kind, header in (("prediction", "Held-out input settings ("), ("selection", "Candidate input settings (")):
        n_text, d_text = neutral[kind]["messages"][0]["content"], default[kind]["messages"][0]["content"]
        assert n_text.split("id,X2,X3,", 1)[1].split("\n\nReturn", 1)[0].split(header, 1)[1].split("\n\n", 1)[0] == \
            d_text.split("id,X2,X3,", 1)[1].split("\n\nReturn", 1)[0].split(header, 1)[1].split("\n\n", 1)[0]
        n_rows = n_text.split("\nid,X2,X3,u1,v1,u2,v2,u3,v3,u4,v4,u5,v5,u6,v6,u7,v7,u8,v8,u9,v9,u10,v10\n", 1)[1].split("\n\n", 1)[0]
        d_rows = d_text.split("\nid,X2,X3,u1,v1,u2,v2,u3,v3,u4,v4,u5,v5,u6,v6,u7,v7,u8,v8,u9,v9,u10,v10\n", 1)[1].split("\n\n", 1)[0]
        assert n_rows == d_rows and "1,1.0,2.0,0.0,-0.0,1.0,-0.5," in n_rows
    # the released forbidden list without the engine's word "simulator" still holds
    assert all(set(forbidden_words(t)) <= {"simulat"} for p in neutral.values() for t in user_texts(p))


def test_neutral_wording_is_the_engines():
    assert hashlib.sha256(ENGINE_REFERENCE.read_bytes()).hexdigest() == ENGINE_NEUTRAL_SHA256
    assert hashlib.sha256(module.VOCABULARY_AUDIT.read_bytes()).hexdigest() == ENGINE_AUDIT_SHA256
    merged = module.FB1_DIR / "fb1_neutral.py"
    if merged.exists():                                     # after a merge with the engine branch: the same bytes
        assert merged.read_bytes() == ENGINE_REFERENCE.read_bytes()
    engine = load_engine_neutral()

    def prose(phrase):
        return phrase.replace("_", " ")

    for kind in ("prediction", "selection"):
        _, user, _ = prompt_parts(kind, "neutral")
        for phrase in (prose(engine.DERIVED_X1), prose(engine.DERIVED_X4),
                       prose(engine.DEFINITIONS[("observations", "X2")]), prose(engine.DEFINITIONS[("observations", "X3")]),
                       "returns a sequence of two component measurements", "each a length 2 vector",
                       "fixed and identical for every input pair", engine.neutral_prose("time grid"),
                       "receives", "scalar expression in X1", "with its coefficients and the inputs X2"):
            assert phrase in user, (kind, phrase)
        assert {engine.START_STATE_KEYS["pos"], engine.START_STATE_KEYS["vel"]} == {"s0a", "s0b"}
        assert "s0a = (<u0>, <v0>) and s0b = (<du0>, <dv0>)" in user
    assert module.interface_settings("neutral")["interface"]["engine_wording"] == engine.NEUTRAL_VERSION


def test_the_mode_comes_from_the_option_or_the_variable_and_a_contradiction_is_refused(tmp_path, monkeypatch):
    assert module.interface_mode(None, {}) == "default" and module.interface_mode(None, {module.INTERFACE_ENV: "0"}) == "default"
    assert module.interface_mode(None, {module.INTERFACE_ENV: "1"}) == "neutral" == module.interface_mode("neutral", {})
    assert module.interface_mode("neutral", {module.INTERFACE_ENV: "1"}) == "neutral"
    for stated, env in (("default", {module.INTERFACE_ENV: "1"}), (None, {module.INTERFACE_ENV: "yes"}), ("other", {})):
        with pytest.raises(ValueError):
            module.interface_mode(stated, env)
    monkeypatch.setenv(module.INTERFACE_ENV, "1")
    with pytest.raises(SystemExit) as error:
        module.main(["--world", "extra_dimensions", "--dry-run", "--output", str(tmp_path / "c"), "--registration", str(REGISTRATION),
                     "--interface", "default"])
    assert error.value.code == 2 and not (tmp_path / "c").exists()
    monkeypatch.setenv(module.INTERFACE_ENV, "true")
    with pytest.raises(SystemExit) as error:
        module.main(["--world", "extra_dimensions", "--dry-run", "--output", str(tmp_path / "bad"), "--registration", str(REGISTRATION)])
    assert error.value.code == 2 and not (tmp_path / "bad").exists()


def test_a_neutral_dry_run_sends_no_vocabulary_and_records_the_mode(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(module.INTERFACE_ENV, "1")
    output = tmp_path / "neutral"
    assert module.main(["--world", "extra_dimensions", "--dry-run", "--output", str(output), "--registration", str(REGISTRATION)]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "dry_run_complete", "requests": 63, "paid_api_calls": 0, "interface": "neutral"}
    receipt = json.loads((output / "receipt.json").read_text())
    starting = json.loads((output / "starting_receipt.json").read_text())
    draft = json.loads((output / "seal_draft.json").read_text())
    assert receipt["interface_mode"] == starting["interface_mode"] == draft["interface_mode"] == "neutral"
    assert receipt["settings"] == starting["settings"] == draft["settings"] == module.settings("extra_dimensions", interface="neutral")
    assert receipt["settings"]["interface"]["mode"] == "neutral" and receipt["settings"] != module.settings("extra_dimensions")
    for path in (module.NEUTRAL_ADDENDUM, module.VOCABULARY_AUDIT, *sorted((module.ROOT / "prompts" / "neutral").glob("*.txt"))):
        assert str(path) in draft["files"]
    bodies = [json.loads(p.read_text())["body"] for p in sorted([*(output / "corpus").glob("*.json"), *(output / "transport").glob("*.request.json")])]
    assert len(bodies) == 2 * 63
    for body in bodies:
        for text in user_texts(body):
            assert audit.hits_in(text) == [] and interface_violations(text, "neutral") == []
    # the engine's checker over the request records: VOCABULARY HITS 0, exit 0
    assert audit.main([str(output)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "VOCABULARY HITS 0"
    # the numbers: the start designs and the first forecast request's tables are the default run's
    monkeypatch.delenv(module.INTERFACE_ENV)
    default = tmp_path / "default"
    assert module.main(["--world", "extra_dimensions", "--dry-run", "--output", str(default), "--registration", str(REGISTRATION)]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "dry_run_complete", "requests": 63, "paid_api_calls": 0}
    d_receipt = json.loads((default / "receipt.json").read_text())
    assert d_receipt["interface_mode"] == "default" and d_receipt["settings"] == module.settings("extra_dimensions")
    assert audit.main([str(default)]) == 1 and capsys.readouterr().out.splitlines()[-1] == "VOCABULARY HITS 2"
    first_n = json.loads((output / "corpus" / "0000-prediction.json").read_text())["body"]["messages"][0]["content"]
    first_d = json.loads((default / "corpus" / "0000-prediction.json").read_text())["body"]["messages"][0]["content"]
    assert first_n.split("v10\n", 1)[1] .split("For each", 1)[0] == first_d.split("v10\n", 1)[1].split("For each", 1)[0]
    for replicate in (1, 2, 3):
        n_rec = json.loads((output / f"closed_rep{replicate}.json").read_text())
        d_rec = json.loads((default / f"closed_rep{replicate}.json").read_text())
        assert n_rec["d0_designs"] == d_rec["d0_designs"] and n_rec["observations"][:2] == d_rec["observations"][:2]
        assert n_rec["evaluation_designs"] == d_rec["evaluation_designs"] and n_rec["candidate_rounds"][0] == d_rec["candidate_rounds"][0]


class FirstInvalid:
    """The synthetic client, except that the first answer of every request is not JSON (so the repair sentence is sent)."""

    def __init__(self):
        self.inner, self.calls, self.messages = module.SyntheticClient(), 0, self

    def create(self, **body):
        self.calls += 1
        response = self.inner.create(**{**body, "messages": body["messages"][:1]})
        if len(body["messages"]) == 1:
            response.content[1].text = "not json"
        return response

    def stream(self, **body):
        response = self.create(**body)

        class Stream:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return response
        return Stream()


def test_a_neutral_repair_is_neutral(tmp_path, capsys):
    rows = observations()
    transport = module.ForceTransport(FirstInvalid(), tmp_path, module.Budget(5.0, module.load_prices()), world="extra_dimensions",
                                      synthetic=True, interface="neutral")
    prompt, mapping = render_prediction(rows, [("head_1", (1.0, 1.0)), ("shift_4", (4.0, 5.0))], CONTEXT, "neutral")
    result = transport.request(prompt, "0000-prediction", "prediction", mapping=mapping, observations=rows, replicate=1,
                               round_index=0, remaining={"prediction": 1, "selection": 1})
    assert result["status"] == "valid" and result["attempt_statuses"] == ["retryable_validation_failure", "valid"]
    selection, ids = render_selection(rows, np.array([[0.5, 9.5], [3.0, 3.0]]), CONTEXT, "neutral")
    chosen = transport.request(selection, "0001-selection", "selection", mapping=ids, observations=rows, replicate=1,
                               round_index=1, remaining={"prediction": 0, "selection": 1})
    assert chosen["status"] == "valid" and len(chosen["attempt_statuses"]) == 2
    repairs = [json.loads(p.read_text())["body"] for p in sorted((tmp_path / "transport").glob("*.attempt1.request.json"))]
    assert len(repairs) == 2 and all(len(b["messages"]) == 3 for b in repairs)
    assert audit.main([str(tmp_path)]) == 0 and capsys.readouterr().out.splitlines()[-1] == "VOCABULARY HITS 0"
    assert "Previous output failed schema validation: invalid JSON object." in repairs[0]["messages"][2]["content"]


def sealed_from(draft, path):
    path.write_text(json.dumps({**draft, "status": "sealed"}))


def test_a_neutral_seal_opens_only_a_neutral_run(tmp_path, monkeypatch, capsys):
    from tools.icl_baseline import transport

    def factory(**kwargs):
        return module.SyntheticClient()

    neutral = module.seal_draft("extra_dimensions", 32000, REGISTRATION, interface="neutral")
    default = module.seal_draft("extra_dimensions", 32000, REGISTRATION)
    assert neutral["interface_mode"] == "neutral" and default["interface_mode"] == "default"
    assert neutral["d0_prediction_requests"] != default["d0_prediction_requests"]
    assert [r["d0_designs"] for r in neutral["d0_prediction_requests"]] == [r["d0_designs"] for r in default["d0_prediction_requests"]]
    sealed_from(neutral, tmp_path / "neutral.json")
    sealed_from(default, tmp_path / "default.json")
    n_settings, d_settings = module.settings("extra_dimensions", interface="neutral"), module.settings("extra_dimensions")
    required_n, required_d = [*module.dependencies("neutral"), REGISTRATION], [*module.dependencies(), REGISTRATION]
    assert isinstance(transport.live_client(True, tmp_path / "neutral.json", n_settings, required_n, factory), module.SyntheticClient)
    assert isinstance(transport.live_client(True, tmp_path / "default.json", d_settings, required_d, factory), module.SyntheticClient)
    for seal, expected, required in (("neutral.json", d_settings, required_d), ("default.json", n_settings, required_n)):
        with pytest.raises(ValueError, match="exactly matching settings"):
            transport.live_client(True, tmp_path / seal, expected, required, factory)
    # the live path refuses a seal of the other interface before any output
    monkeypatch.setattr(module, "live_client", lambda allow, manifest, expected, required, factory=None: transport.live_client(
        allow, manifest, expected, required, factory=lambda **kwargs: module.SyntheticClient()))
    for index, (seal, flags) in enumerate((("default.json", ["--interface", "neutral"]), ("neutral.json", []))):
        output = tmp_path / f"x{index}"
        with pytest.raises(SystemExit) as error:
            module.main(["--world", "extra_dimensions", "--allow-paid-api", "--seal-manifest", str(tmp_path / seal), "--output", str(output),
                         "--registration", str(REGISTRATION), *flags])
        assert error.value.code == 2 and not output.exists()


def test_the_v2_seal_helper_seals_a_neutral_run_only_under_the_variable(tmp_path, capsys, monkeypatch):
    from tools.icl_forcebench.v2 import run_v2_world, seal_v2
    monkeypatch.setenv(module.INTERFACE_ENV, "1")
    output = tmp_path / "every"
    assert module.main(run_v2_world.arguments("dry", "extra_dimensions", output)) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "dry_run_complete", "requests": 99, "paid_api_calls": 0, "interface": "neutral"}
    summary = seal_v2.seal("extra_dimensions", output, tmp_path / "seal.json")
    assert summary["verify_seal"] == "passed" and summary["interface_mode"] == "neutral" and summary["cap_usd"] == 16.0
    monkeypatch.delenv(module.INTERFACE_ENV)
    with pytest.raises(SystemExit, match="interface is not default"):
        seal_v2.seal("extra_dimensions", output, tmp_path / "default_seal.json")
    assert not (tmp_path / "default_seal.json").exists()
    assert re.fullmatch(r"[0-9a-f]{64}", summary["sha256"])
