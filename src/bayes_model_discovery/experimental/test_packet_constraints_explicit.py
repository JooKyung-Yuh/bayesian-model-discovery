"""The limits the engine enforces on every candidate are explicit packet `constraints`.

The packet states these limits explicitly (the chosen option). Three limits the engine already applies to every
candidate travel in every belief packet the three runners build, each read from the engine's
own source: the coefficient cap (`count_models.MAX_DISTINCT_MEAN_COEFFICIENTS`), the active
grammar's reserved coefficient names (`count_models.GRAMMAR_RESERVED_NAMES`) and the cell's
families whose mean must be finite and strictly positive (`positive_inference.
POSITIVE_MEAN_REQUIRED_FAMILIES`, family-scoped). The tests also pin that an explicit
`--prompt-version v2.0` is the absent option, byte for byte.

Synthetic throughout: stubbed fits, the mock proposer, the in-repo fake ChemBench adapter and
synthetic particles. No stored run output is read and no provider is called.
"""

import hashlib
import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from . import chembench_core, count_inference
from . import count_proposer as cp
from . import positive_inference as positive
from .belief_packet import (
    CONSTRAINT_KEYS,
    build_belief_packet,
    cell_constraints,
    core_interface,
    core_real_interface,
    interface_families,
)
from .count_inference import Fit
from .count_models import (
    BINARY_FAMILIES,
    CALL_ARITY,
    FAMILIES,
    GRAMMAR_RESERVED_NAMES,
    IMPLEMENTED_FAMILIES,
    MAX_DISTINCT_MEAN_COEFFICIENTS,
    RESERVED_NAMES,
    V3_RESERVED_NAMES,
    Hypothesis,
    active_grammar_version,
    active_input_signs,
    initial_hypotheses,
    interface_input_names,
    judge_candidate,
    restore_grammar_state,
    set_grammar_version,
    set_llm_priors,
    validate_expression,
)
from .positive_inference import POSITIVE_MEAN_REQUIRED_FAMILIES, SCALAR_FAMILIES
from .real_inference import REAL_FAMILIES

PCX_KEYS = (
    "max_distinct_mean_coefficients",
    "reserved_names",
    "positive_mean_required_families",
)
#: The refusal text the coefficient cap has always raised; this change keeps it verbatim.
COEFFICIENT_COUNT_REFUSAL = "require 1..6 distinct mean coefficients"
@pytest.fixture(autouse=True)
def restore_process_switches():
    """The grammar, the prompt override and the llm-priors switch are process-global."""
    grammar = (active_grammar_version(), active_input_signs())
    yield
    restore_grammar_state(grammar)
    cp.set_prompt_variant(None)
    set_llm_priors(False)


def engine_values(constraints):
    """What the engine's own source says the three keys must hold for these constraints."""
    return {
        "max_distinct_mean_coefficients": MAX_DISTINCT_MEAN_COEFFICIENTS,
        "reserved_names": sorted(GRAMMAR_RESERVED_NAMES[active_grammar_version()]),
        "positive_mean_required_families": sorted(
            set(constraints["allowed_families"]) & set(POSITIVE_MEAN_REQUIRED_FAMILIES)
        ),
    }


def assert_engine_values(constraints):
    """Assert the three constraint keys equal the values the engine's own source gives."""
    assert set(PCX_KEYS) <= set(constraints)
    assert {key: constraints[key] for key in PCX_KEYS} == engine_values(constraints)


def packet_files(out):
    """Return the sorted packet JSON files under `out`, failing if there are none."""
    paths = sorted((out / "packets").glob("*.json"))
    assert paths, f"no packet was written under {out}"
    return paths


# --------------------------------------------------------------------------- the constants


def test_the_three_keys_are_required_constraint_keys():
    """The three explicit keys are part of the required constraint keys."""
    assert set(PCX_KEYS) <= CONSTRAINT_KEYS


def seven_coefficients():
    """Return an expression with seven distinct coefficients and its parameter declarations."""
    names = [f"b{i}" for i in range(7)]
    parameters = (("b0", "intercept"), *((name, "signed") for name in names[1:]))
    # Within the 15-operator cap even at seven coefficients, so the count rule is
    # the one that refuses it, not the operator rule behind it.
    expression = "b0+b1*z+b2*u+b3*z*u+b4*square(z)+b5*square(u)+b6"
    return expression, parameters


def test_the_coefficient_cap_is_the_named_constant_and_its_refusal_text_is_unchanged():
    """The coefficient cap is the named constant and its refusal text is unchanged."""
    assert MAX_DISTINCT_MEAN_COEFFICIENTS == 6
    assert f"1..{MAX_DISTINCT_MEAN_COEFFICIENTS} " in COEFFICIENT_COUNT_REFUSAL
    expression, parameters = seven_coefficients()
    six = expression.rsplit("+", 1)[0]
    validate_expression(six, parameters[:6], ("u", "z"))
    validate_expression("b0", parameters[:1], ("u", "z"))
    for candidate, declared in (("u", ()), (expression, parameters)):
        with pytest.raises(ValueError) as caught:
            validate_expression(candidate, declared, ("u", "z"))
        assert str(caught.value) == COEFFICIENT_COUNT_REFUSAL
        with pytest.raises(ValueError) as caught:
            Hypothesis(candidate, "binomial", declared)
        assert str(caught.value) == COEFFICIENT_COUNT_REFUSAL
    # The per-candidate judge the proposer path uses hands the same text back verbatim.
    item = {"expression": expression, "family": "binomial", "parameters": dict(parameters)}
    verdict = judge_candidate(item, "111", initial_hypotheses("binomial"), "binomial")
    assert verdict == (False, COEFFICIENT_COUNT_REFUSAL)


def test_the_check_reads_the_constant_rather_than_a_literal(monkeypatch):
    """The cap check reads the module constant rather than a hard-coded literal."""
    from . import count_models

    expression, parameters = seven_coefficients()
    six = expression.rsplit("+", 1)[0]
    monkeypatch.setattr(count_models, "MAX_DISTINCT_MEAN_COEFFICIENTS", 5)
    with pytest.raises(ValueError, match="distinct mean coefficients"):
        validate_expression(six, parameters[:6], ("u", "z"))
    monkeypatch.setattr(count_models, "MAX_DISTINCT_MEAN_COEFFICIENTS", 7)
    validate_expression(expression, parameters, ("u", "z"))


def test_every_implemented_family_is_classified_once():
    """Log-link scalar families carry the rule; logit and identity-link families do not."""
    assert POSITIVE_MEAN_REQUIRED_FAMILIES == SCALAR_FAMILIES
    outside = (*FAMILIES, *BINARY_FAMILIES, *REAL_FAMILIES)
    assert not set(POSITIVE_MEAN_REQUIRED_FAMILIES) & set(outside)
    assert set(IMPLEMENTED_FAMILIES) == set(POSITIVE_MEAN_REQUIRED_FAMILIES) | set(outside)
    assert "normal" not in POSITIVE_MEAN_REQUIRED_FAMILIES
    assert "lognormal" in POSITIVE_MEAN_REQUIRED_FAMILIES


@pytest.mark.parametrize("family", POSITIVE_MEAN_REQUIRED_FAMILIES)
def test_every_listed_family_is_refused_a_mean_that_is_not_finite_and_positive(family):
    """The list is the set the check refuses: the mask, the strict mean and the EPIG raise."""
    from .test_epig_positivity_guard import overflow_fit

    fit = overflow_fit(family)
    inputs = np.array([0.0, 0.25, 2.0])
    mean, valid = positive.particle_means_masked(fit, inputs)
    assert not valid.all() and (np.isposinf(mean).any() or (mean == 0).any())
    with pytest.raises(ValueError, match="non-finite/nonpositive mean; no clipping fallback"):
        positive.particle_means(fit, inputs)
    with pytest.raises(ValueError, match="non-finite/nonpositive mean; no clipping fallback"):
        count_inference._epig_scalar_distribution(fit, inputs)


@pytest.mark.parametrize("family", ("normal", "student_t"))
def test_an_unlisted_real_family_keeps_a_zero_or_negative_mean(family):
    """A family outside the positive-mean list keeps zero or negative means."""
    from .test_epig_positivity_guard import grammar_fit

    assert family not in POSITIVE_MEAN_REQUIRED_FAMILIES
    fit = grammar_fit(family, intercepts=(-2.0, 0.0, 2.0), slopes=(0.0, 0.0, 0.0))
    mean, valid = positive.particle_means_masked(fit, np.array([0.0, 0.25, 2.0]))
    assert valid.all() and (mean < 0).any() and (mean == 0).any()
    count_inference._epig_scalar_distribution(fit, np.array([0.0, 0.25, 2.0]))


# --------------------------------------------------------------------------- the packet builder


def chem_cell(cell):
    """Build the constraints of a ChemBench cell with a lognormal initial family."""
    from .chembench_core import CHEM_ALLOWED_FAMILIES

    return cell_constraints(cell, "lognormal", allowed_families=CHEM_ALLOWED_FAMILIES,
                            initial_structures=["b0"], input_names=("f0",))


def boxing_cell(outcome_type, cell):
    """Build the constraints of a BoxingGym cell for the given outcome type."""
    initial, allowed = interface_families(outcome_type, cell[1] == "1")
    return cell_constraints(cell, initial, allowed_families=allowed, initial_structures=["b0"])


@pytest.mark.parametrize("build,families", [
    (lambda: cell_constraints("111", "binomial"), []),
    (lambda: cell_constraints("100", "normal", allowed_families=("normal",)), []),
    (lambda: chem_cell("101"), ["lognormal"]),
    (lambda: chem_cell("111"), ["gamma", "lognormal"]),
    (lambda: boxing_cell("positive", "100"), ["lognormal"]),
    (lambda: boxing_cell("unbounded_count", "110"), ["negative_binomial", "poisson", "zi_poisson"]),
    (lambda: boxing_cell("real", "110"), []),
    (lambda: boxing_cell("binary", "110"), []),
], ids=["core_binomial", "core_gaussian", "chem_f0", "chem_f1", "dugongs_f0", "peregrines_f1",
        "location_f1", "binary_f1"])
def test_each_cell_states_the_engine_values(build, families):
    """Each kind of cell states the engine values in its constraints."""
    constraints = build()
    assert_engine_values(constraints)
    assert constraints["positive_mean_required_families"] == families
    assert constraints["max_distinct_mean_coefficients"] == 6
    assert constraints["reserved_names"] == sorted(RESERVED_NAMES)
    assert set(constraints["reserved_names"]) == {"rho", "pi", *CALL_ARITY}


def test_a_normal_cell_names_no_normal_and_a_lognormal_cell_names_lognormal():
    """A normal cell lists no positive-mean family; a lognormal cell lists lognormal."""
    gaussian = cell_constraints("100", "normal", allowed_families=("normal",))
    assert "normal" in gaussian["allowed_families"]
    assert "normal" not in gaussian["positive_mean_required_families"]
    real_menu = boxing_cell("real", "110")
    assert set(real_menu["allowed_families"]) == set(REAL_FAMILIES)
    assert real_menu["positive_mean_required_families"] == []
    assert "lognormal" in chem_cell("101")["positive_mean_required_families"]
    assert "lognormal" in boxing_cell("positive", "100")["positive_mean_required_families"]


def test_the_reserved_names_follow_the_active_grammar():
    """The reserved names follow the active grammar version."""
    assert cell_constraints("111", "binomial")["reserved_names"] == sorted(RESERVED_NAMES)
    set_grammar_version("v3", core_interface())
    v3 = cell_constraints("111", "binomial")
    assert v3["reserved_names"] == sorted(V3_RESERVED_NAMES) == sorted(GRAMMAR_RESERVED_NAMES["v3"])
    assert set(v3["reserved_names"]) > set(RESERVED_NAMES)
    assert_engine_values(v3)


def minimal_packet(constraints):
    """Build a minimal belief packet around the given constraints."""
    public = [{"obs_id": 0, "round": 0, "u": 0.5, "y": 25}]
    return build_belief_packet(public, [], [], constraints, core_interface(), 0)


def test_unknown_keys_are_still_refused_and_the_three_are_required():
    """Unknown constraint keys are still refused and the three keys are required."""
    constraints = cell_constraints("100", "binomial")
    assert {key: minimal_packet(constraints)["constraints"][key] for key in PCX_KEYS} == {
        key: constraints[key] for key in PCX_KEYS}
    with pytest.raises(ValueError, match="belief packet constraints keys are fixed"):
        minimal_packet({**constraints, "coefficient_hint": "keep it short"})
    for key in PCX_KEYS:
        dropped = {name: value for name, value in constraints.items() if name != key}
        with pytest.raises(ValueError, match="belief packet constraints keys are fixed"):
            minimal_packet(dropped)


# --------------------------------------------------------------------------- the three runners


def run_core(out, monkeypatch, **options):
    """Run one CORE round with stubbed fits and return the output folder."""
    from .test_count_proposer import run_args, stub_fits

    core = stub_fits(monkeypatch)
    core.run(run_args(out, rounds=1, **options))
    return out


def stub_gaussian_fits(monkeypatch):
    """Deterministic stand-in for `gaussian_core.fit_core` (the Gaussian CORE fit)."""
    from . import gaussian_core

    def fake_fit_core(hypothesis, x, y, seed, settings, interface, n_trials=None, *,
                      response_unit, response_center):
        hypothesis = gaussian_core.as_real(hypothesis)
        rng = np.random.default_rng(seed % 2**32)
        particles = {name: rng.normal(0.3, 0.4, 8) for name, _ in hypothesis.parameters}
        particles.update({name: rng.gamma(4.0, 0.25, 8) for name in hypothesis.noise_parameters})
        evidence = -float(len(y)) - (int(hypothesis.model_id[:6], 16) % 500) / 100.0
        return Fit(hypothesis, particles, evidence, [evidence] * 2, {
            "observation_count": len(y), "seed": seed, "interface": interface,
            "response_unit": float(response_unit), "response_center": float(response_center)})

    monkeypatch.setattr(gaussian_core, "fit_core", fake_fit_core)


def test_core_packets_carry_the_three_keys(tmp_path, monkeypatch):
    """Every CORE packet carries the three constraint keys."""
    out = run_core(tmp_path / "core", monkeypatch)
    for path in packet_files(out):
        constraints = json.loads(path.read_text())["constraints"]
        assert_engine_values(constraints)
        assert constraints["positive_mean_required_families"] == []
        # A ChemBench packet adds two more keys; a CORE packet carries neither:
        assert not {"coefficient_name_rule", "prior_probe_rule"} & set(constraints)


def test_gaussian_core_packets_name_no_normal(tmp_path, monkeypatch):
    """Gaussian CORE packets list no normal family as positive-mean."""
    stub_gaussian_fits(monkeypatch)
    out = run_core(tmp_path / "gauss", monkeypatch, cell="100", structure="gauss_den",
                   truth_family="normal", initial_family="normal")
    for path in packet_files(out):
        packet = json.loads(path.read_text())
        assert packet["interface"] == core_real_interface()
        assert packet["constraints"]["allowed_families"] == ["normal"]
        assert_engine_values(packet["constraints"])
        assert packet["constraints"]["positive_mean_required_families"] == []


@pytest.mark.parametrize("task,families", [("dugongs", ["lognormal"]), ("location", [])])
def test_boxinggym_packets_carry_the_three_keys(tmp_path, monkeypatch, task, families):
    """Every BoxingGym packet carries the three constraint keys."""
    from .test_named_inputs import bg_run_args, bg_scalar_stub_fits

    bg = bg_scalar_stub_fits(monkeypatch)
    pytest.importorskip(bg.TASKS[task].module)
    out = tmp_path / task
    bg.run(bg_run_args(out, task=task, cell="100", rounds=1))
    for path in packet_files(out):
        constraints = json.loads(path.read_text())["constraints"]
        assert_engine_values(constraints)
        assert not {"coefficient_name_rule", "prior_probe_rule"} & set(constraints)
        assert constraints["positive_mean_required_families"] == families


def run_chembench(out, tmp_path, monkeypatch, **options):
    """Run one ChemBench round with stubbed fits and return the output folder."""
    from . import chembench_core
    from .test_chembench_core import _args, _manifest
    from .test_chembench_mda_options import stub_fits as stub_chembench_fits

    stub_chembench_fits(monkeypatch)
    manifest = tmp_path / "manifest"
    manifest.mkdir(exist_ok=True)
    chembench_core.run(_args(out, _manifest(manifest), cell="100", rounds=1,
                             structure_proposals=None, **options))
    return out


def test_chembench_packets_carry_the_three_keys(tmp_path, monkeypatch):
    """Every ChemBench packet carries the three keys plus the naming and probe rules."""
    out = run_chembench(tmp_path / "chem", tmp_path, monkeypatch)
    for path in packet_files(out):
        packet = json.loads(path.read_text())
        constraints = packet["constraints"]
        assert constraints["allowed_families"] == ["lognormal"]
        # The three explicit keys are the engine values unchanged -
        # reserved_names is the engine tuple again - and the packet adds the coefficient-naming
        # rule whose forbidden lists are exactly this packet's interface namespace.
        assert_engine_values(constraints)
        rule = constraints["coefficient_name_rule"]
        forbidden = rule["not_allowed_as_coefficient_names"]
        assert rule["rule"] and not any(ch in rule["rule"] for ch in "=*+/")
        assert forbidden["inputs"] == [item["name"] for item in packet["interface"]["inputs"]]
        assert forbidden["derived_inputs"] == list(packet["interface"]["derived_inputs"])
        assert forbidden["functions_and_reserved"] == constraints["reserved_names"]
        assert set(forbidden["inputs"]) | set(forbidden["derived_inputs"]) == set(
            interface_input_names(packet["interface"]))
        assert constraints["positive_mean_required_families"] == ["lognormal"]
        # The packet also carries the prior-probe contract, bound to the probe's constants and this
        # packet's own declared priors and inputs.
        probe = constraints["prior_probe_rule"]
        assert probe["prior_draws"] == chembench_core.PRIOR_PROBE_SAMPLES
        assert probe["checked_points"]["diagonal_grid_points"] == chembench_core.PRIOR_PROBE_GRID_POINTS
        assert probe["checked_points"]["range_corners"] == 2 ** len(packet["interface"]["inputs"])
        assert probe["coefficient_priors"] == constraints["roles"]
        assert probe["link"] == packet["interface"]["link"] == "log"


# --------------------------------------------------------------------------- explicit v2.0


def test_an_explicit_v2_0_selects_the_default_file_and_bytes():
    """An explicit prompt version v2.0 selects the default prompt file and bytes."""
    assert cp.PROMPT_VERSION == "v2.0"
    records = []
    for args in (SimpleNamespace(prompt_variant=None),
                 SimpleNamespace(prompt_version="v2.0", prompt_variant=None)):
        selected, identity = cp.select_run_prompt(args)
        assert selected is None and not cp.prompt_variant_active()
        records.append((identity, cp.system_prompt_file(), cp.system_prompt_bytes(),
                        cp.composed_system_prompt(), cp.composed_system_prompt(burst=True)))
    assert records[0] == records[1]
    identity, (path, version), sent, _, _ = records[1]
    assert path == cp.PROMPT_FILE == cp.prompt_version_file("v2.0") and version == "v2.0"
    assert sent == cp.PROMPT_FILE.read_bytes() == cp.prompt_version_file("v2.0").read_bytes()
    assert identity == {"prompt_version": "v2.0",
                        "prompt_sha256": hashlib.sha256(sent).hexdigest()}


@pytest.mark.parametrize("module,argv", [
    ("core_count", ["core_count", "--output", "/dev/null/never", "--structure", "out",
                    "--truth-family", "beta_binomial", "--seed", "11"]),
    ("boxinggym_core", ["boxinggym_core", "--output", "/dev/null/never", "--task", "death"]),
    ("chembench_core", ["chembench_core", "--manifest", "/dev/null/never.json",
                        "--output", "/dev/null/never", "--noise", "0.01"]),
])
def test_each_runner_cli_accepts_v2_0_and_resolves_it_to_the_default(monkeypatch, module, argv):
    """Each runner CLI accepts prompt version v2.0 and resolves it to the default."""
    runner = importlib.import_module("." + module, package=cp.__package__)
    captured = []
    monkeypatch.setattr(runner, "run", captured.append)
    for stated in ([], ["--prompt-version", "v2.0"]):
        monkeypatch.setattr(sys, "argv", [*argv, *stated])
        runner.main()
    absent, explicit = captured
    assert "prompt_version" not in vars(absent) and explicit.prompt_version == "v2.0"
    assert cp.select_run_prompt(absent) == cp.select_run_prompt(explicit) == (
        None, cp.prompt_identity())
    assert cp.prompt_identity()["prompt_sha256"] == hashlib.sha256(
        cp.PROMPT_FILE.read_bytes()).hexdigest()


#: Keys that name the run's own directory or clock, never what it computed.
VOLATILE_CONFIG_KEYS = ("output", "created_utc")


def comparable_config(out):
    """Load a run's config.json without the keys that name its directory or clock."""
    config = json.loads((out / "config.json").read_text())
    return {key: value for key, value in config.items() if key not in VOLATILE_CONFIG_KEYS}


def assert_same_run_record(absent, explicit):
    """Assert two runs record the same config, prompt pair and packet bytes."""
    first, second = comparable_config(absent), comparable_config(explicit)
    assert first == second
    assert (first["prompt_version"], first["prompt_sha256"]) == (
        "v2.0", hashlib.sha256(cp.PROMPT_FILE.read_bytes()).hexdigest())
    packets = [path.relative_to(absent) for path in packet_files(absent)]
    assert packets == [path.relative_to(explicit) for path in packet_files(explicit)]
    for relative in packets:
        assert (absent / relative).read_bytes() == (explicit / relative).read_bytes(), relative


def test_core_run_records_one_prompt_pair_whether_v2_0_is_stated_or_not(tmp_path, monkeypatch):
    """A CORE run records the same prompt pair whether v2.0 is stated or not."""
    absent = run_core(tmp_path / "absent", monkeypatch)
    explicit = run_core(tmp_path / "explicit", monkeypatch, prompt_version="v2.0")
    assert_same_run_record(absent, explicit)


def test_boxinggym_run_records_one_prompt_pair_whether_v2_0_is_stated_or_not(
        tmp_path, monkeypatch):
    """A BoxingGym run records the same prompt pair whether v2.0 is stated or not."""
    from .test_named_inputs import bg_run_args, bg_scalar_stub_fits

    bg = bg_scalar_stub_fits(monkeypatch)
    pytest.importorskip(bg.TASKS["dugongs"].module)
    for name, stated in (("absent", {}), ("explicit", {"prompt_version": "v2.0"})):
        bg.run(bg_run_args(tmp_path / name, task="dugongs", cell="100", rounds=1, **stated))
    assert_same_run_record(tmp_path / "absent", tmp_path / "explicit")


def test_chembench_run_records_one_prompt_pair_whether_v2_0_is_stated_or_not(
        tmp_path, monkeypatch):
    """A ChemBench run records the same prompt pair whether v2.0 is stated or not."""
    absent = run_chembench(tmp_path / "absent", tmp_path, monkeypatch)
    explicit = run_chembench(tmp_path / "explicit", tmp_path, monkeypatch, prompt_version="v2.0")
    assert_same_run_record(absent, explicit)


