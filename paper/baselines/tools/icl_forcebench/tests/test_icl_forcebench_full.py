"""Unit the full-interface ForceBench edition (2026-10-12): the opt-in full interface of the ForceBench ICL tool. Synthetic clients only, no provider
call. Full = the released prompts with the input-name mask off (r, p1, p2, t) and the shared description block
tools/paper_prompts/FB_FULL_DESCRIPTION.md prepended to the system text; full and neutral together are refused; a seal
opens only a run of its own mode; the settings, the receipts and the seal draft carry the description's sha256. The
default requests are the released ones byte for byte (sha256 pinned from base 7b28b76, as the neutral tests pin them)."""

import hashlib
import json
import re

import numpy as np
import pytest

from tools.icl_forcebench import icl_forcebench as module
from tools.icl_forcebench.protocol import (
    FULL_DESCRIPTION,
    FULL_DROPPED_SENTENCE,
    T_OBS,
    Context,
    Observation,
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

from tools.icl_forcebench.audits import fb_full_prompt_audit as full_audit  # noqa: E402

REGISTRATION = module.ROOT.parent / "icl_baseline" / "prices.json"
CONTEXT = Context(p_range=(0.1, 10.0), times=tuple(0.5 * k for k in range(1, 11)), pos0=(-2.5, -6.0), vel0=(-0.1, 2.0))
#: The engine branch's copy of the description (unit the full-interface ForceBench edition); this file must be the same bytes.
DESCRIPTION_SHA256 = "d666077054a957976f87367ede5279771becd6d20ef58772b7b4d1acaf38032a"
#: sha256 of json_bytes(request_body(...)) of these renders on base 7b28b76 (released tool); the neutral tests' pins.
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
NATIVE, MASKED = {"r", "p1", "p2", "t"}, {"X1", "X2", "X3", "X4"}
ENVS = (module.INTERFACE_ENV, module.INTERFACE_FULL_ENV)


@pytest.fixture(autouse=True)
def no_interface_variables(monkeypatch):
    for name in ENVS:
        monkeypatch.delenv(name, raising=False)


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


def model_texts(prompt):
    return [prompt["system"], *(m["content"] for m in prompt["messages"] if m["role"] == "user")]


def tokens(texts):
    return set().union(*(set(re.findall(r"[A-Za-z0-9_]+", t)) for t in texts))


def body_sha(prompt):
    return hashlib.sha256(json_bytes(module.request_body(prompt, 32000))).hexdigest()


def test_the_description_file_is_the_engine_copy():
    assert hashlib.sha256(FULL_DESCRIPTION.read_bytes()).hexdigest() == DESCRIPTION_SHA256 == module.description_sha256()
    text = FULL_DESCRIPTION.read_text()
    assert text.endswith("\n") and text.count("\n") == 1
    assert interface_violations(text, "full") == []          # no law name, no world name, no catalogue, no masked name


def test_unset_interface_is_the_released_tool_byte_for_byte():
    assert module.interface_mode() == "default"
    for name, prompt in renders("default").items():
        assert body_sha(prompt) == BASE_DEFAULT_BODIES[name], name
    plain = render_prediction(observations(), [("head_1", (1.0, 1.0)), ("shift_4", (4.0, 5.0))], CONTEXT)[0]
    assert body_sha(plain) == BASE_DEFAULT_BODIES["prediction"]
    assert "interface" not in module.settings("gravity") and module.settings("gravity") == module.settings("gravity", interface="default")
    assert module.dependencies() == module.dependencies("default")
    assert not {module.FULL_DESCRIPTION, module.FULL_PROMPT_AUDIT, module.FULL_ADDENDUM} & set(module.dependencies())
    assert module.full_record("default") == {} == module.full_record("neutral")


def test_the_full_request_starts_with_the_description_and_is_otherwise_the_default_unmasked():
    description = FULL_DESCRIPTION.read_text()
    full, default = renders("full"), renders("default")
    for name in full:
        assert full[name]["system"] == description + "\n" + default[name]["system"], name
        assert full[name]["system"].startswith(description + "\n")
        f_users = [m["content"] for m in full[name]["messages"] if m["role"] == "user"]
        d_users = [m["content"] for m in default[name]["messages"] if m["role"] == "user"]
        # the mask off: X2 -> p1, X3 -> p2 (the only masked names of the default prompts) and the mask's sentence dropped
        assert f_users == [u.replace(FULL_DROPPED_SENTENCE, "").replace("X2", "p1").replace("X3", "p2") for u in d_users], name
        assert [m["role"] for m in full[name]["messages"]] == [m["role"] for m in default[name]["messages"]]
    text = full["prediction"]["messages"][0]["content"]
    assert "\nid,p1,p2,u1,v1," in text and "\nq,p1,p2\nq1,1.0,1.0\nq2,4.0,5.0\n" in text and "no additional meaning" not in text
    assert "\nc,p1,p2\nc1,0.5,9.5\n" in full["selection"]["messages"][0]["content"]
    # the repair sentences are the default's byte for byte
    assert full["prediction_repair"]["messages"][-1] == default["prediction_repair"]["messages"][-1]
    assert full["selection_repair"]["messages"][-1] == default["selection_repair"]["messages"][-1]
    for kind in ("prediction", "selection"):
        system, user, repair = prompt_parts(kind, "full")
        assert system.startswith(description + "\n") and not MASKED & tokens([system, user, repair])


def test_native_names_are_present_and_no_masked_name_remains():
    full = renders("full")
    for name, prompt in full.items():
        seen = tokens(model_texts(prompt))
        assert NATIVE <= seen and not MASKED & seen, name
        assert all(interface_violations(t, "full") == [] for t in model_texts(prompt)), name
    for error in ERRORS:
        for kind, count in (("prediction", 2), ("selection", 3)):
            repair = repair_request(full[kind], "x", error, kind, count, "full")["messages"][-1]["content"]
            assert not MASKED & tokens([repair]) and interface_violations(repair, "full") == []
    # the full check catches a masked name, a law name and a world name
    assert interface_violations("c*X2/X1**2", "full") == ["X1", "X2"] and interface_violations("pX1", "full") == []
    assert interface_violations("Newton's gravity", "full") == ["gravit", "newton"]


def test_full_and_neutral_together_are_refused_and_write_nothing(tmp_path, monkeypatch):
    full_env, neutral_env = {module.INTERFACE_FULL_ENV: "1"}, {module.INTERFACE_ENV: "1"}
    assert module.interface_mode(None, full_env) == "full" == module.interface_mode("full", {}) == module.interface_mode("full", full_env)
    assert module.interface_mode("full", {module.INTERFACE_ENV: "0"}) == "full"
    assert module.interface_mode("neutral", {module.INTERFACE_FULL_ENV: "0"}) == "neutral"
    for stated, env in ((None, {**full_env, **neutral_env}), ("full", {**full_env, **neutral_env}), ("neutral", {**full_env, **neutral_env}),
                        ("full", neutral_env), ("neutral", full_env), ("default", full_env), (None, {module.INTERFACE_FULL_ENV: "yes"})):
        with pytest.raises(ValueError):
            module.interface_mode(stated, env)
    cases = [({module.INTERFACE_ENV: "1", module.INTERFACE_FULL_ENV: "1"}, []),
             ({module.INTERFACE_ENV: "1", module.INTERFACE_FULL_ENV: "1"}, ["--interface", "full"]),
             ({module.INTERFACE_ENV: "1"}, ["--interface", "full"]),
             ({module.INTERFACE_FULL_ENV: "1"}, ["--interface", "neutral"]),
             ({module.INTERFACE_FULL_ENV: "1"}, ["--interface", "default"]),
             ({module.INTERFACE_FULL_ENV: "true"}, [])]
    for index, (env, flags) in enumerate(cases):
        for name in ENVS:
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        output = tmp_path / f"refused{index}"
        with pytest.raises(SystemExit) as error:
            module.main(["--world", "extra_dimensions", "--dry-run", "--output", str(output), "--registration", str(REGISTRATION), *flags])
        assert error.value.code == 2 and not output.exists(), (env, flags)
    assert list(tmp_path.iterdir()) == []


def test_the_description_sha_is_computed_from_the_file_at_run_time(tmp_path, monkeypatch):
    settings = module.settings("gravity", interface="full")
    assert settings["interface"]["mode"] == "full" and settings["interface"]["description_sha256"] == DESCRIPTION_SHA256
    assert settings["interface"]["input_names"] == {"X1": "r", "X2": "p1", "X3": "p2", "X4": "t"}
    other = tmp_path / "other.md"
    other.write_text("another description.\n")
    monkeypatch.setattr(module, "FULL_DESCRIPTION", other)
    assert module.settings("gravity", interface="full")["interface"]["description_sha256"] == hashlib.sha256(other.read_bytes()).hexdigest()
    assert module.full_record("full") == {"description_sha256": hashlib.sha256(other.read_bytes()).hexdigest()}


def test_a_full_dry_run_records_the_mode_and_the_description_and_passes_the_prompt_audit(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(module.INTERFACE_FULL_ENV, "1")
    output = tmp_path / "full"
    assert module.main(["--world", "extra_dimensions", "--dry-run", "--output", str(output), "--registration", str(REGISTRATION)]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "dry_run_complete", "requests": 63, "paid_api_calls": 0, "interface": "full",
                                                   "description_sha256": DESCRIPTION_SHA256}
    receipt = json.loads((output / "receipt.json").read_text())
    starting = json.loads((output / "starting_receipt.json").read_text())
    draft = json.loads((output / "seal_draft.json").read_text())
    assert receipt["interface_mode"] == starting["interface_mode"] == draft["interface_mode"] == "full"
    assert receipt["description_sha256"] == starting["description_sha256"] == draft["description_sha256"] == DESCRIPTION_SHA256
    assert receipt["settings"] == starting["settings"] == draft["settings"] == module.settings("extra_dimensions", interface="full")
    assert receipt["settings"]["interface"]["description_sha256"] == DESCRIPTION_SHA256
    assert draft["files"][str(module.FULL_DESCRIPTION)] == DESCRIPTION_SHA256
    for path in (module.FULL_ADDENDUM, module.FULL_PROMPT_AUDIT):
        assert str(path) in draft["files"]
    description = FULL_DESCRIPTION.read_text()
    bodies = [json.loads(p.read_text())["body"] for p in sorted([*(output / "corpus").glob("*.json"), *(output / "transport").glob("*.request.json")])]
    assert len(bodies) == 2 * 63
    for body in bodies:
        assert body["system"].startswith(description + "\n")
        seen = tokens(model_texts(body))
        assert NATIVE <= seen and not MASKED & seen
    # the full prompt audit over the ICL request records (single "body"): PASS, exit 0
    assert full_audit.main([str(FULL_DESCRIPTION), str(output)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == ("FULL PROMPT AUDIT 63 records 63 bodies: description 63/63, native names 63/63, "
                                                         "masked names 0/63 -> PASS")
    # the numbers are the default's: the start designs, the evaluation designs and the first candidates (default seal draft)
    default = module.seal_draft("extra_dimensions", 32000, REGISTRATION)
    assert [r["d0_designs"] for r in draft["d0_prediction_requests"]] == [r["d0_designs"] for r in default["d0_prediction_requests"]]


def test_the_prompt_audit_fails_a_default_run_record(tmp_path, capsys):
    prompt = renders("default")["prediction"]
    record = tmp_path / "0000-prediction.attempt0.request.json"
    record.write_text(json.dumps({"body": module.request_body(prompt, 32000), "request_id": "0000-prediction", "attempt": 0}))
    assert full_audit.main([str(FULL_DESCRIPTION), str(tmp_path)]) == 1
    last = capsys.readouterr().out.splitlines()[-1]
    assert last == "FULL PROMPT AUDIT 1 records 1 bodies: description 0/1, native names 0/1, masked names 1/1 -> FAIL"
    # an engine record's "bodies" list is read as before
    engine = tmp_path / "engine"
    engine.mkdir()
    (engine / "x.request.json").write_text(json.dumps({"bodies": [module.request_body(renders("full")["prediction"], 32000)]}))
    assert full_audit.main([str(FULL_DESCRIPTION), str(engine)]) == 0


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


def test_a_full_repair_keeps_the_description_and_the_native_names(tmp_path, capsys):
    rows = observations()
    transport = module.ForceTransport(FirstInvalid(), tmp_path, module.Budget(5.0, module.load_prices()), world="extra_dimensions",
                                      synthetic=True, interface="full")
    prompt, mapping = render_prediction(rows, [("head_1", (1.0, 1.0)), ("shift_4", (4.0, 5.0))], CONTEXT, "full")
    result = transport.request(prompt, "0000-prediction", "prediction", mapping=mapping, observations=rows, replicate=1,
                               round_index=0, remaining={"prediction": 1, "selection": 1})
    assert result["status"] == "valid" and result["attempt_statuses"] == ["retryable_validation_failure", "valid"]
    selection, ids = render_selection(rows, np.array([[0.5, 9.5], [3.0, 3.0]]), CONTEXT, "full")
    chosen = transport.request(selection, "0001-selection", "selection", mapping=ids, observations=rows, replicate=1,
                               round_index=1, remaining={"prediction": 0, "selection": 1})
    assert chosen["status"] == "valid" and len(chosen["attempt_statuses"]) == 2
    repairs = [json.loads(p.read_text())["body"] for p in sorted((tmp_path / "transport").glob("*.attempt1.request.json"))]
    assert len(repairs) == 2 and all(len(b["messages"]) == 3 for b in repairs)
    assert full_audit.main([str(FULL_DESCRIPTION), str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines()[-1].endswith("masked names 0/4 -> PASS")


def sealed_from(draft, path):
    path.write_text(json.dumps({**draft, "status": "sealed"}))


def test_a_full_seal_opens_only_a_full_run(tmp_path, monkeypatch):
    from tools.icl_baseline import transport

    def factory(**kwargs):
        return module.SyntheticClient()

    drafts = {mode: module.seal_draft("extra_dimensions", 32000, REGISTRATION, interface=mode) for mode in ("default", "neutral", "full")}
    assert drafts["full"]["interface_mode"] == "full" and drafts["full"]["description_sha256"] == DESCRIPTION_SHA256
    assert "description_sha256" not in drafts["default"] and "description_sha256" not in drafts["neutral"]
    assert drafts["full"]["settings"]["interface"]["description_sha256"] == DESCRIPTION_SHA256
    assert [r["d0_designs"] for r in drafts["full"]["d0_prediction_requests"]] == [r["d0_designs"] for r in drafts["default"]["d0_prediction_requests"]]
    assert len({json.dumps(d["d0_prediction_requests"]) for d in drafts.values()}) == 3
    for mode, draft in drafts.items():
        sealed_from(draft, tmp_path / f"{mode}.json")
    settings = {mode: module.settings("extra_dimensions", interface=mode) for mode in drafts}
    required = {mode: [*module.dependencies(mode), REGISTRATION] for mode in drafts}
    assert isinstance(transport.live_client(True, tmp_path / "full.json", settings["full"], required["full"], factory), module.SyntheticClient)
    for seal, run in (("full", "default"), ("full", "neutral"), ("default", "full"), ("neutral", "full")):
        with pytest.raises(ValueError, match="exactly matching settings"):
            transport.live_client(True, tmp_path / f"{seal}.json", settings[run], required[run], factory)
    # the live path refuses a seal of another interface before any output
    monkeypatch.setattr(module, "live_client", lambda allow, manifest, expected, required, factory=None: transport.live_client(
        allow, manifest, expected, required, factory=lambda **kwargs: module.SyntheticClient()))
    cases = (("default.json", {module.INTERFACE_FULL_ENV: "1"}, []), ("neutral.json", {}, ["--interface", "full"]),
             ("full.json", {}, []), ("full.json", {module.INTERFACE_ENV: "1"}, []), ("full.json", {}, ["--interface", "neutral"]))
    for index, (seal, env, flags) in enumerate(cases):
        for name in ENVS:
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        output = tmp_path / f"x{index}"
        with pytest.raises(SystemExit) as error:
            module.main(["--world", "extra_dimensions", "--allow-paid-api", "--seal-manifest", str(tmp_path / seal), "--output", str(output),
                         "--registration", str(REGISTRATION), *flags])
        assert error.value.code == 2 and not output.exists(), (seal, env, flags)


def test_the_v2_seal_helper_seals_a_full_run_only_under_the_variable(tmp_path, monkeypatch):
    from tools.icl_forcebench.v2 import seal_v2
    monkeypatch.setenv(module.INTERFACE_FULL_ENV, "1")
    summary = seal_v2.seal("extra_dimensions", "build", tmp_path / "seal.json")
    assert summary["verify_seal"] == "passed" and summary["interface_mode"] == "full" and summary["cap_usd"] == 16.0
    sealed = json.loads((tmp_path / "seal.json").read_text())
    assert sealed["description_sha256"] == DESCRIPTION_SHA256 == sealed["settings"]["interface"]["description_sha256"]
    folder = tmp_path / "dry"
    folder.mkdir()
    (folder / "seal_draft.json").write_text(json.dumps({**sealed, "status": "draft_not_execution_authority"}))
    monkeypatch.delenv(module.INTERFACE_FULL_ENV)
    with pytest.raises(SystemExit, match="interface is not default"):
        seal_v2.seal("extra_dimensions", folder, tmp_path / "default_seal.json")
    monkeypatch.setenv(module.INTERFACE_ENV, "1")
    with pytest.raises(SystemExit, match="interface is not neutral"):
        seal_v2.seal("extra_dimensions", folder, tmp_path / "neutral_seal.json")
    monkeypatch.setenv(module.INTERFACE_FULL_ENV, "1")
    with pytest.raises(ValueError, match="refused together"):
        seal_v2.seal("extra_dimensions", folder, tmp_path / "both_seal.json")
    assert not any((tmp_path / name).exists() for name in ("default_seal.json", "neutral_seal.json", "both_seal.json"))
