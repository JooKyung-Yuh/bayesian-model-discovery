"""Opt-in coordinate aliases; synthetic responses only, no external API calls."""

import json
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from . import chembench_core as cc
from . import count_proposer as cp
from . import test_chembench_core as fake
from .design_search import SearchTransform
from .test_chembench_mda_options import stub_fits

NAMES = ("C_A", "C_I", "C_B", "C_P", "Enz", "T", "pH")
NAME_MAP = {f"X{i + 1}": name for i, name in enumerate(NAMES)}
GRAMMAR_NAMES = (*NAME_MAP, *(f"{name}z" for name in NAME_MAP))
FORBIDDEN_NAMES = set(NAMES) | {f"{name}z" for name in NAMES}
FORBIDDEN_WORDS = ("enzyme", "substrate", "michaelis", "inhibit", "ping", "pong", "kinetic", "catalys")


class SyntheticSession(fake._FakeSession):
    """Fake ChemBench session with seven named inputs and an extra observable."""
    def schema(self):
        """Return the base schema with the seven input names, bounds and observables."""
        return {**super().schema(), "input_names": list(NAMES),
                "input_bounds": {name: [0.1, 1.0] for name in NAMES},
                "log_sampled_inputs": ["C_A", "C_B"],
                "observable_names": ["r0", "C_A_t"]}

    def _rate(self, design):
        return float(np.exp(0.1 + 0.3 * design["C_A"] - 0.15 * design["C_I"]))

    def observe(self, design):
        """Observe as the base session does and add a constant `C_A_t` output."""
        row = super().observe(design)
        return {**row, "outputs": {**row["outputs"], "C_A_t": 0.125}}


def sample_designs(count, seed):
    """Return `count` random designs over the seven inputs, determined by `seed`."""
    values = np.random.default_rng(seed).uniform(0.1, 1.0, (count, len(NAMES)))
    return [dict(zip(NAMES, row, strict=True)) for row in values]


def candidate(expression):
    """Build a lognormal candidate whose coefficients are inferred from `expression`."""
    parameters = {name: "intercept" if name == "b0" else "signed"
                  for name in re.findall(r"\bb\d+\b", expression)}
    return {"expression": expression, "family": "lognormal", "parameters": parameters}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """Autouse fixture: stub the fits and reset the prompt selection before and after each test."""
    stub_fits(monkeypatch)
    cp.select_prompt_version(None)
    cp.set_prompt_variant(None)
    yield
    cp.select_prompt_version(None)
    cp.set_prompt_variant(None)


def native_named_fixture(monkeypatch):
    """Patch the adapter so the native session uses the synthetic named inputs."""
    monkeypatch.setattr(fake.real_adapter, "ChemBenchSession", SyntheticSession)
    monkeypatch.setattr(fake.real_adapter, "sample_designs", sample_designs)


def test_boundary_round_trip_preserves_values_order_outputs_and_finished_reference():
    """The boundary round trip preserves values, order, outputs and the finished reference."""
    native = SyntheticSession("synthetic", "easy", "v0", noise_level=0.01, budget=4)
    masked = cc.InputNameMaskedSession(native, sample_designs)
    assert masked.input_name_map == NAME_MAP
    schema = masked.schema()
    assert schema["input_names"] == list(NAME_MAP)
    assert schema["log_sampled_inputs"] == ["X1", "X3"]
    for row, alias in zip(sample_designs(3, 123), masked.sample_designs(3, 123), strict=True):
        assert list(alias) == list(NAME_MAP)
        assert masked.unmask_design(alias) == row
        assert list(masked.unmask_design(alias)) == list(row)
    design = masked.sample_designs(1, 999)[0]
    observation = masked.observe(design)
    assert observation["x"] == design
    assert observation["outputs"]["C_A_t"] == 0.125
    assert masked.calls == 1
    with pytest.raises(RuntimeError, match="unavailable"):
        masked.reference_values([design])
    masked.finish()
    expected = native.reference_values([masked.unmask_design(design)])
    actual = masked.reference_values([design])
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_aliases_follow_declaration_not_native_or_alphabetical_order():
    """Aliases follow declaration order, not native or alphabetical order."""
    session = SimpleNamespace(schema=lambda: {
        "input_names": ["second", "first"], "input_bounds": {"first": [1, 2], "second": [3, 4]},
        "log_sampled_inputs": ["first"],
    })
    masked = cc.InputNameMaskedSession(session, lambda *a: [])
    assert masked.input_name_map == {"X1": "second", "X2": "first"}
    assert masked.schema()["input_bounds"] == {"X2": [1, 2], "X1": [3, 4]}
    with pytest.raises(ValueError, match="declared aliases"):
        masked.unmask_design({"X1": 3})
    with pytest.raises(ValueError, match="declared coordinates"):
        masked.mask_design({"second": 3, "unknown": 1})


def test_search_transform_aliases_preserve_every_bit():
    """Search transforms over aliased names preserve every bit."""
    native = SyntheticSession("synthetic", "easy", "v0", noise_level=0, budget=1)
    schema = native.schema()
    masked = cc.InputNameMaskedSession(native, sample_designs).schema()
    before = SearchTransform.from_bounds(schema["input_names"], schema["input_bounds"],
                                         schema["log_sampled_inputs"])
    after = SearchTransform.from_bounds(masked["input_names"], masked["input_bounds"],
                                        masked["log_sampled_inputs"])
    points = np.asarray([list(row.values()) for row in sample_designs(10, 123)])
    np.testing.assert_array_equal(before.forward(points), after.forward(points))
    np.testing.assert_array_equal(before.inverse(before.forward(points)),
                                  after.inverse(after.forward(points)))


@pytest.mark.parametrize("extra", [
    {"initial_catalogue": "on"}, {"initial_pool_file": "must_not_be_opened.json"},
    {"pool_generator": "pool_generator_v1"}, {"run_label": "production"},
])
def test_refusal_before_output_and_client(tmp_path, monkeypatch, extra):
    """A refused masked cell fails before any output or client is created."""
    def no_client(*a, **kw):
        """Fail the test: a client must not be built."""
        pytest.fail("a refused masked cell constructed a client")
    monkeypatch.setattr(cp, "AnthropicCountProposer", no_client)
    if extra.get("run_label") == "production":
        monkeypatch.setattr(cc, "bmd_label_receipt", lambda *a, **kw: {"run_label": "production"})
    settings = {"mask_input_names": True, "cell": "100", "structure_proposals": "free",
                "initial_catalogue": "off", "initial_proposals": 2,
                "proposer": "anthropic", "allow_paid_api": True, **extra}
    with pytest.raises(ValueError, match="--mask-input-names"):
        cc.run(fake._args(tmp_path / "cell", fake._manifest(tmp_path), **settings))
    assert not (tmp_path / "cell").exists()


def test_omitted_cli_has_no_masking_keys(tmp_path, monkeypatch):
    """Without the masking option the config carries no masking keys."""
    def no_native_identity(*args):
        """Fail the test: no native identity may be computed."""
        pytest.fail("omitted masking computed a native identity")
    monkeypatch.setattr(cc.InputNameMaskedSession, "rng_model_id", no_native_identity)
    monkeypatch.setattr(sys, "argv", ["chembench_core", "--manifest", str(fake._manifest(tmp_path)),
                                     "--output", str(tmp_path / "cell"), "--rounds", "1",
                                     "--particles", "16", "--q", "8", "--noise", "0.01"])
    cc.main()
    config = json.loads((tmp_path / "cell/config.json").read_text())
    assert "mask_input_names" not in config
    assert "input_name_map" not in config
    assert not (tmp_path / "cell/masked_model_ids.jsonl").exists()


@pytest.mark.parametrize("fit_guards", ["off", "v1"])
def test_all_request_routes_and_packets_are_masked_including_repairs(tmp_path, monkeypatch, fit_guards):
    """All request routes and packets are masked, including repairs."""
    native_named_fixture(monkeypatch)
    fit_calls = []
    original_fit = cc.fit

    def capture_fit(hypothesis, x, y, seed, *args, **kwargs):
        """Record the model id and seed of each fit, then delegate to the real fit."""
        fit_calls.append((hypothesis.model_id, seed))
        return original_fit(hypothesis, x, y, seed, *args, **kwargs)

    monkeypatch.setattr(cc, "fit", capture_fit)
    likelihood_calls = []

    def likelihood_probe(hypothesis, x, y, interface, **kwargs):
        """Record the call and report that the probe passed."""
        likelihood_calls.append((hypothesis, kwargs))
        return {"passed": True}

    monkeypatch.setattr(cc, "loglik_prior_probe", likelihood_probe)
    monkeypatch.setattr(cc, "run_with_fit_guards",
                        lambda fn, *args, fit_guards, **kwargs: fn(*args, **kwargs))
    collision = {"expression": "b0+X1", "family": "lognormal",
                 "parameters": {"b0": "intercept", "X1": "signed"}}
    replies = [candidate("b0"), candidate("b0"), candidate("b0+b1*tanh(X1z)"),
               candidate("+".join(f"b{i}" for i in range(20))),
               candidate("b0+b1*square(X1z)"), candidate("b0+b1*Q9"),
               candidate("b0+b1*tanh(X2z)"), collision, candidate("b0+b1*tanh(X3z)")]
    calls = []

    def create(**body):
        """Pop the next scripted reply and return it as a synthetic response."""
        calls.append(body)
        value = replies.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(value))],
                               usage=SimpleNamespace(model_dump=lambda **kw: {
                                   "input_tokens": 1, "output_tokens": 1}),
                               id="synthetic-nm", model="synthetic-model", stop_reason="end_turn")

    original = cp.AnthropicCountProposer
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(cp, "AnthropicCountProposer",
                        lambda records, **kw: original(records, client=client, **kw))
    out = tmp_path / "cell"
    monkeypatch.setattr(sys, "argv", [
        "chembench_core", "--manifest", str(fake._manifest(tmp_path)), "--output", str(out),
        "--mask-input-names", "--cell", "100", "--rounds", "3", "--particles", "16", "--q", "8",
        "--noise", "0.01",
        "--proposer", "anthropic", "--allow-paid-api", "--run-label", "development",
        "--initial-catalogue", "off", "--initial-proposals", "2",
        "--fit-guards", fit_guards,
    ])
    cc.main()
    assert not replies and len(calls) == 9
    config = json.loads((out / "config.json").read_text())
    assert config["mask_input_names"] is True and config["input_name_map"] == NAME_MAP
    assert config["schema"]["input_names"] == list(NAME_MAP)
    packets = list((out / "packets").glob("*.json"))
    requests = list((out / "llm_transport").glob("*.request.json"))
    assert len(requests) == 6 and len(packets) == 6
    bodies = [body for path in requests for body in json.loads(path.read_text())["bodies"]]
    assert len(bodies) == len(calls) == 9
    for body in bodies:
        text = json.dumps({key: body[key] for key in ("system", "messages")})
        assert not FORBIDDEN_NAMES.intersection(re.findall(r"[A-Za-z0-9_]+", text))
        assert not any(word in text.lower() for word in FORBIDDEN_WORDS)
    for path in packets:
        text = path.read_text()
        assert not FORBIDDEN_NAMES.intersection(re.findall(r"[A-Za-z0-9_]+", text))
        assert not any(word in text.lower() for word in FORBIDDEN_WORDS)
    pairs = [json.loads(line) for line in (out / "masked_model_ids.jsonl").read_text().splitlines()]
    ids = {row["model_id"]: row["rng_model_id"] for row in pairs}
    assert len(ids) == len(pairs)
    # An input-independent model has the same genuine ID under both names.
    native_only_ids = set(ids.values()) - set(ids)
    assert native_only_ids
    for path in [*out.rglob("*.json"), *out.rglob("*.jsonl")]:
        if path.name != "masked_model_ids.jsonl":
            assert not any(key in path.read_text() for key in native_only_ids), path
    streams = set()
    for key, seed in fit_calls:
        matches = {(stream, t) for stream in ("chem_smc", "chem_smc_proposed") for t in range(4)
                   if seed == cc.seed_for(stream, 11, t, ids[key])}
        assert len(matches) == 1
        streams.update(stream for stream, t in matches)
    assert streams == {"chem_smc", "chem_smc_proposed"}
    assert bool(likelihood_calls) == (fit_guards == "v1")
    for hypothesis, kwargs in likelihood_calls:
        key = ids[hypothesis.model_id]
        assert kwargs["seed"] == cc.PRIOR_PROBE_SEED ^ int(key, 16)
        expected = cc._prior_particles(hypothesis, kwargs["count"], rng_model_id=key)
        for name, draws in expected.items():
            np.testing.assert_array_equal(kwargs["prior_draws"][name], draws)
    burst = json.loads((out / "proposals/t00_burst.json").read_text())
    assert [wave["calls"] for wave in burst["waves"]] == [2, 1]
    assert burst["waves"][0]["duplicates"] == 1
    for t in (1, 2, 3):
        record = json.loads((out / f"proposals/t{t:02d}.json").read_text())["transport"]
        assert record["repairs"] == 1
    # The actual repair requests carry each refusal; no custom error text is introduced.
    assert "Q9" in json.dumps(calls[6]["messages"])
    assert "X1" in json.dumps(calls[8]["messages"])
    (tmp_path / "leakage_scan.json").write_text(json.dumps({
        "request_files": len(requests), "request_bodies": len(bodies), "packet_files": len(packets),
        "route_counts": {"burst": 2, "refill": 1, "round": 3,
                         "repair_coefficient_limit": 1, "repair_unknown_input": 1,
                         "repair_coefficient_input_collision": 1},
        "forbidden_matches": [],
        "native_only_id_matches": [], "masked_id_pairs": len(pairs),
    }, indent=2) + "\n")


def test_input_alias_cannot_be_a_coefficient_and_refusal_contains_no_real_name():
    """An input alias cannot be a coefficient name, and the refusal reveals no real name."""
    from .count_models import judge_candidate

    accepted, reason = judge_candidate(
        {"expression": "b0+X1", "family": "lognormal",
         "parameters": {"b0": "intercept", "X1": "signed"}},
        "100", (), "lognormal", input_names=tuple(NAME_MAP),
        allowed_families=cc.CHEM_ALLOWED_FAMILIES,
    )
    assert accepted is False
    assert reason == "invalid or reserved coefficient name"
    assert not FORBIDDEN_NAMES.intersection(re.findall(r"[A-Za-z0-9_]+", reason))


def test_native_rng_identity_preserves_true_masked_id_and_prior_draws(tmp_path):
    """The native RNG identity preserves the true model id and prior draws under masking."""
    from concurrent.futures import ThreadPoolExecutor

    native = SyntheticSession("synthetic", "easy", "v0", noise_level=0, budget=1)
    session = cc.InputNameMaskedSession(native, sample_designs)
    parameters = (("b0", "intercept"), ("b1", "signed"))
    masked = cc.ScalarHypothesis("b0+b1*tanh(X1z)+X2", "lognormal", parameters, GRAMMAR_NAMES)
    visible = cc.ScalarHypothesis("b0+b1*tanh(C_Az)+C_I", "lognormal", parameters,
                                 (*NAMES, *(f"{name}z" for name in NAMES)))
    original = masked.as_dict()
    path = tmp_path / "masked_model_ids.jsonl"
    with ThreadPoolExecutor(max_workers=4) as executor:
        keys = list(executor.map(lambda _: session.rng_model_id(masked, path), range(20)))
    assert keys == [visible.model_id] * 20
    assert masked.model_id != visible.model_id and masked.as_dict() == original
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"model_id": masked.model_id, "rng_model_id": visible.model_id}]
    expected = cc._prior_particles(visible, 32)
    actual = cc._prior_particles(masked, 32, rng_model_id=keys[0])
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_prior_probe_forwards_only_the_opt_in_rng_key(tmp_path, monkeypatch):
    """The prior probe forwards the RNG key only when the masking option supplies one."""
    session = cc.InputNameMaskedSession(
        SyntheticSession("synthetic", "easy", "v0", noise_level=0, budget=1), sample_designs)
    hypothesis = cc.ScalarHypothesis("b0+b1*tanh(X1z)", "lognormal",
                                    (("b0", "intercept"), ("b1", "signed")), GRAMMAR_NAMES)
    key = session.rng_model_id(hypothesis, tmp_path / "ids.jsonl")
    seen = []
    original = cc._prior_particles

    def capture(hypothesis, count, **kwargs):
        """Record the forwarded keyword arguments, then delegate to the original."""
        seen.append(kwargs)
        return original(hypothesis, count, **kwargs)

    monkeypatch.setattr(cc, "_prior_particles", capture)
    interface = {"outcome_type": "positive_continuous", "link": "log",
                 "inputs": [{"name": name, "range": [0.1, 1.0]} for name in NAME_MAP],
                 "derived_inputs": {f"{name}z": f"2*{name}-1" for name in NAME_MAP}}
    cc._prior_probe(hypothesis, interface, 1.0, count=32, rng_model_id=key)
    cc._prior_probe(hypothesis, interface, 1.0, count=32)
    assert seen == [{"rng_model_id": key}, {}]


def test_runner_tie_key_does_not_replace_recorded_model_ids(tmp_path, monkeypatch):
    """The runner's tie key does not replace the recorded model ids."""
    session = cc.InputNameMaskedSession(
        SyntheticSession("synthetic", "easy", "v0", noise_level=0, budget=1), sample_designs)
    hypotheses = [cc.ScalarHypothesis(f"b0+b1*tanh(X{i}z)", "lognormal",
                  (("b0", "intercept"), ("b1", "signed")), GRAMMAR_NAMES) for i in (2, 3)]
    fits = [SimpleNamespace(hypothesis=h, log_evidence=-2.0) for h in hypotheses]
    monkeypatch.setattr(cc, "validate_pool", lambda _: np.full(2, -np.log(2.0)))
    item = {"fits": fits, "round": 0, "observation_count": 4}
    def key(h):
        """Return the RNG key of hypothesis `h`."""
        return session.rng_model_id(h, tmp_path / "ids.jsonl")
    plain = cc._pool_weights_row(item, tuple(NAME_MAP))
    bound = cc._pool_weights_row(item, tuple(NAME_MAP), model_id_key=key)
    assert min(hypotheses, key=lambda h: h.model_id) != min(hypotheses, key=key)
    assert bound["single_model_id"] == min(hypotheses, key=key).model_id
    assert bound["models"] == plain["models"]
