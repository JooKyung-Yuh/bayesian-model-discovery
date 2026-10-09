"""Tests (added 2026-10-03) for FB1_GRAMMAR_TRIG (cos and sin in the free-expression grammar).  No paid call.

Run: cd tools/fb1_snapshot && env -u FORCE_COLOR JAX_PLATFORMS=cpu DP_SCIENCEAGENT_DIR=<ScienceAgent dir> \
      FB1_PYTHON=<jax venv python> FB1_OUT_DIR=/tmp/x/ \
      <jax venv python> -m pytest -q --confcutdir=. test_fb1_grammar_trig.py
The grammar state is process-wide: every test that switches the option on does it through the `trig` fixture, which
switches it off again and empties the free-structure registry. The runner checks import run_gravity in a child
interpreter (test_fb1_label_rule.native_import, skipped without FB1_PYTHON); the run checks start the runner in its
mock-transport free mode for one proposal (about two minutes each), next to the base commit's runner read from git.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

SNAPSHOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SNAPSHOT))
os.environ.setdefault("FB1_OUT_DIR", "/tmp/fb1_grammar_trig_test_out/")
os.environ.setdefault("FB1_WORLD", "gravity")
# DiscoverPhysics @ 33b7fa9: DP_SCIENCEAGENT_DIR, or a checkout beside the repository
os.environ.setdefault("DP_SCIENCEAGENT_DIR",
                      str(Path(__file__).resolve().parents[3] / "DiscoverPhysics" / "ScienceAgent"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fb1_core as core  # noqa: E402
import fb1_design_search as fds  # noqa: E402
import fb1_forward as fwd  # noqa: E402
import fb1_grammar as fbg  # noqa: E402
import fb1_grammar_trig as trig_module  # noqa: E402
import fb1_proposer as fp  # noqa: E402
import run_gravity as rg  # noqa: E402
from test_fb1_epig_targets import base_snapshot, run_child  # noqa: E402,F401 - the module fixture is reused
from test_fb1_label_rule import native_import  # noqa: E402
from test_fb1_proposer import _state  # noqa: E402

TIMES = np.asarray(json.loads((SNAPSHOT / "frozen_inputs" / "FB1_FIXED_IC_MANIFEST_v5.json").read_text())
                   ["worlds"]["oscillator"]["measurement_times"], float)
V2_CALLS = ["exp", "log", "power", "sqrt", "square", "tanh"]
TRIG_CALLS = ["cos", "exp", "log", "power", "sin", "sqrt", "square", "tanh"]
#: The oscillator's own law as a free expression: G0 cos(omega t) p1 / (2 pi r) with c = G0 / (2 pi).
TRUE_LAW = ("c*p1*cos(w*t)/r", {"c": "positive", "w": "positive"})
#: The text fb1_grammar has always given for inv_r_tmod (read before any test switches the option on).
DEFAULT_REASON = dict(fbg.CATALOG_NOT_EXPRESSIBLE)


@pytest.fixture()
def trig():
    """Switch the trigonometric grammar on for one test and restore the default grammar afterwards."""
    fbg.reset_registry()
    assert trig_module.enable() is True
    try:
        yield trig_module
    finally:
        fbg.reset_registry()
        trig_module.disable()


@pytest.fixture(autouse=True)
def grammar_is_off_around_every_test():
    """Check that the grammar option is off before and after every test."""
    fbg.reset_registry()
    assert not trig_module.enabled()
    yield
    fbg.reset_registry()
    assert not trig_module.enabled(), "a test left the trigonometric grammar on"


# ---------------------------------------------------------------- off: exactly today's grammar
@pytest.mark.parametrize("call", ["cos", "sin"])
def test_off_cos_and_sin_are_refused_as_before(call):
    """Test: off cos and sin are refused as before."""
    with pytest.raises(fbg.GrammarError, match="call outside expression grammar"):
        fbg.compile_structure(f"c*p1*{call}(w*t)/r", {"c": "positive", "w": "positive"})
    with pytest.raises(ValueError, match="call outside expression grammar"):
        trig_module.reference_structure()
    assert sorted(fbg.vendor.CALL_ARITY) == V2_CALLS == fbg.grammar_identity(TIMES)["calls"]
    assert fbg.N_OPS_MAX == 10 and fbg.CATALOG_NOT_EXPRESSIBLE == DEFAULT_REASON
    assert "cos/sin" in DEFAULT_REASON["inv_r_tmod"]


def test_enable_is_idempotent_and_disable_puts_the_original_objects_back():
    """Test: enable is idempotent and disable puts the original objects back."""
    originals = (fbg.vendor.CALL_ARITY, fbg.vendor.RESERVED_NAMES, fbg.CATALOG_NOT_EXPRESSIBLE)
    identity_off = fbg.grammar_identity(TIMES)
    assert trig_module.enable() is True and trig_module.enable() is False
    try:
        assert sorted(fbg.vendor.CALL_ARITY) == TRIG_CALLS and fbg.vendor.CALL_ARITY["cos"] == 1 == fbg.vendor.CALL_ARITY["sin"]
        assert fbg.vendor.RESERVED_NAMES[-2:] == ("cos", "sin")
        name, _ = fbg.admit(*TRUE_LAW, TIMES)
        with pytest.raises(RuntimeError, match="reset the free-structure registry"):
            trig_module.disable()
    finally:
        fbg.reset_registry()
        assert trig_module.disable() is True
    assert trig_module.disable() is False
    assert (fbg.vendor.CALL_ARITY, fbg.vendor.RESERVED_NAMES, fbg.CATALOG_NOT_EXPRESSIBLE) == originals
    assert all(a is b for a, b in zip(originals, (fbg.vendor.CALL_ARITY, fbg.vendor.RESERVED_NAMES,
                                                  fbg.CATALOG_NOT_EXPRESSIBLE)))
    assert fbg.grammar_identity(TIMES) == identity_off
    # enabling after a structure is registered would judge one run by two grammars
    fbg.admit("c*p1/(r**2 + d)", {"c": "positive", "d": "positive"}, TIMES)
    with pytest.raises(RuntimeError, match="before any free structure is registered"):
        trig_module.enable()


def test_the_option_leaves_every_file_and_every_other_table_alone(trig):
    """Test: the option leaves every file and every other table alone."""
    assert trig.identity()["unchanged_files"] == {"fb1_grammar.py": fbg.module_sha256(),
                                                  "vendor/count_models_v2_9.py": fbg.VENDOR_SHA256}
    import hashlib
    assert hashlib.sha256((SNAPSHOT / "vendor" / "count_models_v2_9.py").read_bytes()).hexdigest() == fbg.VENDOR_SHA256
    assert fbg.N_OPS_MAX == 10 and len(fbg.CATALOG_GRAMMAR_EXPRESSIONS) == 13
    assert "inv_r_tmod" not in fbg.CATALOG_GRAMMAR_EXPRESSIONS
    assert fbg.CATALOG_NOT_EXPRESSIBLE == {"inv_r_tmod": trig.INV_R_TMOD_REASON}
    assert set(fbg.CATALOG_PRIOR_SCALE_DIFFERENCES) == {"inv_r_bg"}


def test_grammar_identity_shows_cos_and_sin_and_nothing_else_moves(trig):
    """Test: grammar identity shows cos and sin and nothing else moves."""
    on = fbg.grammar_identity(TIMES)
    trig.disable()
    try:
        off = fbg.grammar_identity(TIMES)
    finally:
        trig.enable()
    assert on["calls"] == TRIG_CALLS and off["calls"] == V2_CALLS
    assert {key for key in on if on[key] != off[key]} == {"calls", "catalogue_not_expressible"}
    assert on["n_ops_max"] == off["n_ops_max"] == 10 and on["catalogue_expressible_count"] == "13/14"


# ---------------------------------------------------------------- on: validation, numpy, jax
EXPRESSIONS = [
    ("c*p1*cos(w*t)/r", {"c": "positive", "w": "positive"},
     lambda x, r, p1, t: x[:, 0] * p1 * np.cos(x[:, 1] * t) / r, 5),
    ("c*p1*sin(w*t + h)/r", {"c": "positive", "w": "positive", "h": "signed"},
     lambda x, r, p1, t: x[:, 0] * p1 * np.sin(x[:, 1] * t + x[:, 2]) / r, 6),
    ("c*p1*(1 + a*cos(w*t))/r", {"c": "positive", "a": "signed", "w": "positive"},
     lambda x, r, p1, t: x[:, 0] * p1 * (1 + x[:, 1] * np.cos(x[:, 2] * t)) / r, 7),
    ("c*p1*cos(k*r)/r**2", {"c": "positive", "k": "positive"},
     lambda x, r, p1, t: x[:, 0] * p1 * np.cos(x[:, 1] * r) / r ** 2, 6),
]


@pytest.mark.parametrize(("expression", "parameters", "numpy_force", "n_ops"), EXPRESSIONS,
                         ids=[e[0] for e in EXPRESSIONS])
def test_on_trig_expressions_validate_count_and_evaluate_like_numpy_and_jax(trig, expression, parameters, numpy_force,
                                                                             n_ops):
    """Test: on trig expressions validate count and evaluate like numpy and jax."""
    struct, record = fbg.compile_structure(expression, parameters)
    assert record["n_ops"] == n_ops == fbg.count_operators(expression) and struct.kinds == fbg.roles_to_kinds(parameters)
    rng = np.random.default_rng(20261003)
    D, P = len(TIMES) * 4, 16
    r = rng.uniform(0.3, 10.0, (D, P))
    p1, p2 = rng.uniform(0.1, 10.0, (D, 1)), rng.uniform(0.1, 10.0, (D, 1))
    t = np.repeat(TIMES, 4)[:, None]
    x = rng.uniform(0.3, 2.0, (P, len(parameters)))
    names = record["coefficient_names"]
    by_hand = numpy_force(x, r, p1, t)
    by_vendor = fbg.vendor.evaluate(expression, {n: x[:, j] for j, n in enumerate(names)}, r, backend=np,
                                    inputs={"r": r, "p1": p1, "p2": p2, "t": t})
    by_jax = np.asarray(struct.force(jnp.asarray(r), jnp.asarray(p1), jnp.asarray(p2), jnp.asarray(x), jnp.asarray(t)))
    assert np.allclose(by_vendor, by_hand, rtol=1e-14, atol=0)
    assert np.allclose(by_jax, by_vendor, rtol=1e-13, atol=1e-300)
    assert by_jax.shape == (D, P)


def test_on_canonical_ids_separate_cos_from_sin_and_ignore_renaming(trig):
    """Test: on canonical ids separate cos from sin and ignore renaming."""
    cos_id = fbg.structure_model_id("c*p1*cos(w*t)/r", {"c": "positive", "w": "positive"})
    assert cos_id == fbg.structure_model_id("a*p1*cos(b*t)/r", {"a": "positive", "b": "positive"})
    assert cos_id != fbg.structure_model_id("c*p1*sin(w*t)/r", {"c": "positive", "w": "positive"})


def test_on_the_operator_cap_counts_each_trig_call_and_stays_ten(trig):
    """Test: on the operator cap counts each trig call and stays ten."""
    roles = {"c": "positive", "a": "signed", "w": "positive"}
    ten = "c*p1*(1 + a*cos(w*t))*cos(w*t)/r"                  # 10 operators (two cos calls among them), 3 constants
    eleven = "c*p1*(1 + a*cos(w*t) + a*sin(w*t))/r"           # 11 operators, 3 constants
    assert fbg.count_operators(ten) == 10 and fbg.count_operators(eleven) == 11
    assert fbg.compile_structure(ten, roles)[1]["n_ops"] == 10
    with pytest.raises(fbg.GrammarError, match="uses 11 operators; the catalogue maximum is 10"):
        fbg.compile_structure(eleven, roles)


def test_on_a_coefficient_named_after_a_trig_call_is_refused_and_off_it_stays_allowed(trig):
    """Test: on a coefficient named after a trig call is refused and off it stays allowed."""
    with pytest.raises(fbg.GrammarError, match="invalid or reserved coefficient name"):
        fbg.compile_structure("cos*p1/r", {"cos": "positive"})
    trig.disable()
    try:
        fbg.compile_structure("cos*p1/r", {"cos": "positive"})   # today's grammar: an ordinary name
    finally:
        trig.enable()


# ---------------------------------------------------------------- inv_r_tmod
def test_the_inv_r_tmod_transcription_equals_the_catalogue_force_and_still_is_not_a_free_law(trig):
    """Test: the inv r tmod transcription equals the catalogue force and still is not a free law."""
    transcription = trig.reference_structure()
    catalogue = core.STRUCTURES["inv_r_tmod"]
    assert tuple(transcription.kinds) == ("pos", "sig", "sig", "pos") != tuple(catalogue.kinds)
    # the grid of test_catalog_transcriptions_match_the_catalog_force_numerically: bit for bit
    rng = np.random.default_rng(20260908)
    r = jnp.array(rng.uniform(0.3, 10.0, (9, 6)))
    p1, p2 = jnp.array(rng.uniform(0.1, 10.0, (9, 1))), jnp.array(rng.uniform(0.1, 10.0, (9, 1)))
    th = jnp.array(rng.uniform(0.3, 2.0, (6, 4)))
    a = np.broadcast_to(np.asarray(transcription.force(r, p1, p2, th, 0.7)), (9, 6))
    b = np.broadcast_to(np.asarray(catalogue.force(r, p1, p2, th, 0.7)), (9, 6))
    assert np.array_equal(a, b)
    # the finiteness-probe grid (r grid x all measurement times x 64 prior draws x 13 (p1, p2) points), with the
    # transcription's prior draws and with the catalogue's (w scaled by W0)
    expression, parameters = trig.INV_R_TMOD_TRANSCRIPTION
    draws = fbg.probe_draws(transcription, fbg.structure_model_id(expression, parameters))
    u = jax.random.normal(jax.random.PRNGKey(20261003), (fbg.PROBE_DRAWS, 4))
    for x in (draws, core.u_to_x(u, catalogue.kinds)):
        mine = fbg._force_on_grid(transcription, x, TIMES)
        theirs = fbg._force_on_grid(catalogue, x, TIMES)
        assert mine.shape == (fbg.PROBE_R_POINTS * len(TIMES), fbg.PROBE_DRAWS, 13)
        assert np.all(np.isfinite(mine)) and np.array_equal(mine, theirs)
    with pytest.raises(fbg.GrammarError, match="at most 3 free constants"):
        fbg.compile_structure(expression, dict(parameters))
    facts = trig.identity()["inv_r_tmod"]
    assert facts["n_free_constants"] == 4 and facts["n_ops"] == 11
    assert facts["validates_in_the_vendored_grammar"] is True and facts["admissible_as_a_free_form_law"] is False
    assert facts["in_catalogue_grammar_expressions"] is False and facts["reason"] == trig.INV_R_TMOD_REASON
    assert facts["catalogue_kinds"] == ["pos", "sig", "sig", "pos_w"]


# ---------------------------------------------------------------- on: through fitting, prediction, EIG and EPIG
def test_on_the_true_oscillator_law_is_admitted_and_runs_through_the_jax_machinery(trig, monkeypatch):
    """Test: on the true oscillator law is admitted and runs through the jax machinery."""
    name, record = fbg.admit(*TRUE_LAW, TIMES, round=1)
    assert record["outcome"] == "active" and name.startswith("g_") and core.STRUCTURES[name].n_ops == 5
    ic = json.loads((SNAPSHOT / "frozen_inputs" / "FB1_FIXED_IC_MANIFEST_v5.json").read_text())["worlds"]["oscillator"]
    pos0, vel0 = jnp.asarray(ic["pos2"]), jnp.asarray(ic["velocity2"])
    monkeypatch.setattr(fwd, "INTEGRATION_METHOD", "leapfrog")
    monkeypatch.setattr(fwd, "ALIGN_MEASUREMENT_TIMES", False)
    monkeypatch.setattr(fwd, "FORCE_SOFTENING", 0.0)
    monkeypatch.setattr(core, "DT", 0.02)
    cand = core.Candidate(name, "gauss", 2)
    # prediction (jax) against a numpy velocity-Verlet replica at the oscillator's own constants
    i2pi, c, w = 1.0 / (2.0 * math.pi), 5.0 / (2.0 * math.pi), math.pi / 2.0
    p1, p2 = jnp.asarray([0.9767, 4.0]), jnp.asarray([6.5666, 1.5])
    u_true = jnp.asarray([[math.log(c), math.log(w), math.log(0.1)]])
    predict = jax.jit(lambda u, a, b: core.make_predict(cand)(u, a, b, pos0, vel0, TIMES))
    ours = np.asarray(predict(u_true, p1, p2))                                  # (D, 1, T, 2)
    for d in range(2):
        x = np.asarray(ic["pos2"], float).copy(); v = np.asarray(ic["velocity2"], float).copy()
        def acc(x, t, d=d):
            rn = math.hypot(x[0], x[1])
            return -(c * float(p1[d]) * math.cos(w * t) / rn) / float(p2[d]) * (x / rn)
        t, a, out = 0.0, acc(x, 0.0), []
        for tq in TIMES:
            for _ in range(int(round((tq - t) / 0.02))):
                v = v + 0.01 * a; x = x + 0.02 * v; t = t + 0.02; a = acc(x, t); v = v + 0.01 * a
            t = float(tq); out.append(x.copy())
        assert np.max(np.abs(ours[d, 0] - np.asarray(out))) < 1e-9
    assert abs(c * 2.0 * math.pi - 5.0) < 1e-12 and abs(i2pi * 2.0 * math.pi - 1.0) < 1e-15
    # fitting: the likelihood compiles and a small tempered SMC runs on synthetic data from the true constants
    y = np.asarray(ours[:, 0]) + np.random.default_rng(1).normal(0.0, 0.1, ours[:, 0].shape)
    loglik = jax.jit(lambda u: core.make_loglik(cand)(u, p1, p2, jnp.asarray(y), pos0, vel0, TIMES))
    u_fit, log_z, diag = core.smc_fit(jax.random.PRNGKey(3), cand, loglik, n=48, chains=1, mh_steps=1)
    assert u_fit.shape == (48, 3) and math.isfinite(log_z) and np.all(np.isfinite(np.asarray(loglik(u_fit))))
    # EIG and EPIG: the runner's own estimators and the ascent's jax-pure twins, with a gradient through cos
    other = core.Candidate("inv_r", "gauss", 0)
    fits = [(cand, u_fit), (other, core.sample_prior_u(jax.random.PRNGKey(4), 48, other.kinds()))]
    weights = np.array([0.5, 0.5])
    monkeypatch.setattr(rg, "M_OUTER", 16)
    monkeypatch.setattr(rg, "N_INNER", 12)
    designs = np.asarray([[1.0, 2.0], [5.0, 0.5], [9.0, 3.0]])
    eig, _ = rg.joint_eig(jax.random.PRNGKey(5), fits, weights, designs)
    epig, _ = rg.joint_epig(jax.random.PRNGKey(6), fits, weights, designs, rg.score_q_epig_targets()[0])
    assert np.all(np.isfinite(eig)) and np.all(np.isfinite(epig))
    plain = lambda cc, D: (lambda uu, aa, bb: core.make_predict(cc)(uu, aa, bb, pos0, vel0, TIMES))  # noqa: E731
    draws = fds.epig_draws(jax.random.PRNGKey(7), fits, weights, 16, 12)
    transform = fds.LogSigmoidTransform((0.1, 10.0), (0.1, 10.0))
    targets = rg.score_q_epig_targets()[0]
    objective = lambda z: jnp.sum(fds.epig_pure(transform.design(z), targets, fits, weights, draws, plain,  # noqa: E731
                                                rg.log_dens, rg.T_OBS)[0])
    gradient = np.asarray(jax.grad(objective)(jnp.asarray([[0.2, -0.3]])))
    assert np.all(np.isfinite(gradient)) and np.any(gradient != 0.0)


# ---------------------------------------------------------------- what the model reads
def test_on_the_packet_changes_only_in_the_call_list_and_the_inv_r_tmod_reason(trig):
    """Test: on the packet changes only in the call list and the inv r tmod reason."""
    rows, members, _catalog, interface = _state()
    on_constraints = fp.free_constraints("gauss", times=TIMES)
    on = fp.build_belief_packet(rows, members, [], on_constraints, interface, 2)
    trig.disable()
    try:
        off = fp.build_belief_packet(rows, members, [], fp.free_constraints("gauss", times=TIMES), interface, 2)
    finally:
        trig.enable()
    assert {key for key in on if on[key] != off[key]} == {"constraints"}
    assert {key for key in on["constraints"] if on["constraints"][key] != off["constraints"][key]} == {"grammar"}
    grammar_on, grammar_off = on["constraints"]["grammar"], off["constraints"]["grammar"]
    assert {key for key in grammar_on if grammar_on[key] != grammar_off[key]} == {"calls", "catalogue_not_expressible"}
    assert grammar_on["calls"] == TRIG_CALLS and grammar_on["catalogue_not_expressible"] == {
        "inv_r_tmod": trig.INV_R_TMOD_REASON}
    assert on["constraints"]["initial_structures"] == off["constraints"]["initial_structures"]
    # masked (name masking): the model reads K12 for inv_r_tmod and the same reason text
    masked = fp.InputNameMask().mask_value(on)
    assert masked["constraints"]["grammar"]["calls"] == TRIG_CALLS
    assert masked["constraints"]["grammar"]["catalogue_not_expressible"] == {"K12": trig.INV_R_TMOD_REASON}


# ---------------------------------------------------------------- the option at import (child interpreter)
SHOW = ("import fb1_grammar as g\n"
        "print(json.dumps([sorted(g.vendor.CALL_ARITY), g.N_OPS_MAX, r.GRAMMAR_TRIG, sorted(r._code_records()[0])]))")


def lines(proc):
    """Parse the last two stdout lines of a child process as JSON (shown value, receipt)."""
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = proc.stdout.strip().splitlines()
    return json.loads(out[-2]), json.loads(out[-1])


@pytest.mark.parametrize("value", ["2", "true", "", "yes", "on"])
def test_an_unknown_value_is_refused_before_any_output(tmp_path, value):
    """Test: an unknown value is refused before any output."""
    proc = native_import(tmp_path, {"FB1_STRUCTURE_PROPOSALS": "free", "FB1_GRAMMAR_TRIG": value})
    assert proc.returncode != 0 and "FB1_GRAMMAR_TRIG must be 0 or 1" in proc.stderr


@pytest.mark.parametrize("extra", [{}, {"FB1_STRUCTURE_PROPOSALS": "catalog"}, {"FB1_PROPOSER": "llm"}])
def test_the_option_without_free_structures_is_refused_before_any_output(tmp_path, extra):
    """Test: the option without free structures is refused before any output."""
    proc = native_import(tmp_path, {**extra, "FB1_GRAMMAR_TRIG": "1"})
    assert proc.returncode != 0 and "FB1_GRAMMAR_TRIG=1 requires FB1_STRUCTURE_PROPOSALS=free" in proc.stderr


def test_the_runner_switches_the_grammar_and_declares_it_only_when_on(tmp_path):
    """Test: the runner switches the grammar and declares it only when on."""
    free = {"FB1_STRUCTURE_PROPOSALS": "free", "FB1_PROPOSER": "llm"}
    shown_off, receipt_off = lines(native_import(tmp_path, free, suffix=SHOW))
    shown_zero, receipt_zero = lines(native_import(tmp_path, {**free, "FB1_GRAMMAR_TRIG": "0"}, suffix=SHOW))
    shown_on, receipt_on = lines(native_import(tmp_path, {**free, "FB1_GRAMMAR_TRIG": "1"}, suffix=SHOW))
    assert shown_off == shown_zero == [V2_CALLS, 10, False, ["fb1_core.py", "fb1_forward.py", "run_gravity.py",
                                                              "world_constants.py"]]
    assert shown_on == [TRIG_CALLS, 10, True, ["fb1_core.py", "fb1_forward.py", "fb1_grammar_trig.py",
                                               "run_gravity.py", "world_constants.py"]]
    assert receipt_off == receipt_zero and "grammar_trig" not in receipt_off["binding"]["settings"]
    assert receipt_on["binding"]["settings"] == {**receipt_off["binding"]["settings"], "grammar_trig": ["cos", "sin"]}


# ---------------------------------------------------------------- whole runs: one mock proposal in free mode
FA_C = Path(__file__).resolve().parents[1] / "fb1_launch_c" / "proposer_system_v2_9_FA-C.txt"
#: The fb1_launch_c launcher's mock branch (one proposal, masked names) on the random-design arm 110 (no EIG cost).
ONE_PROPOSAL = {"FB1_WORLD": "oscillator", "FB1_ARM": "110", "FB1_SEED": "31", "FB1_LIST": "v3",
                "FB1_PROPOSER": "llm", "FB1_STRUCTURE_PROPOSALS": "free", "FB1_T_EXTRA": "1",
                "FB1_PROPOSAL_POLICY": "P1", "FB1_MASK_INPUT_NAMES": "1", "FB1_RUN_LABEL": "development"}
RUN_STEM = "FB1_RUN_oscillator_110_seed31_v3_free"


def without_call_latency(value):
    """The transport's wall-clock `elapsed_seconds` (per call record and response file) is the only timing there."""
    if isinstance(value, dict):
        return {key: without_call_latency(item) for key, item in value.items() if key != "elapsed_seconds"}
    if isinstance(value, list):
        return [without_call_latency(item) for item in value]
    return value


def strip_volatile(record):
    """Test: strip volatile."""
    record = json.loads(json.dumps(record))
    record.pop("timing_s")
    record["code_sha256"].pop("run_gravity.py")
    record["code_sha256"].pop("fb1_grammar_trig.py", None)
    for checkpoint in record["checkpoints"].values():
        checkpoint.pop("elapsed_s")
    record["proposer"] = without_call_latency(record["proposer"])
    # the label policy's own source digest moves with any edit of run_label_policy.py, like a code sha
    for block in (record["label"], record["proposer"].get("label_receipt", {})):
        block.pop("policy_source_sha256", None)
    return record


@pytest.fixture(scope="module")
def one_proposal_runs(base_snapshot, tmp_path_factory):
    """Run one mock free-mode proposal for the base commit, with the option off and with it on."""
    options = dict(ONE_PROPOSAL)
    if FA_C.is_file():
        options["FB1_PROMPT_VARIANT"] = str(FA_C)
    folders = {}
    for name, snapshot, extra in (("base", base_snapshot, {}), ("off", SNAPSHOT, {}),
                                  ("on", SNAPSHOT, {"FB1_GRAMMAR_TRIG": "1"})):
        folders[name] = tmp_path_factory.mktemp(f"one_proposal_{name}")
        done = run_child(snapshot, folders[name], {**options, **extra})
        assert done.returncode == 0, (name, done.stderr[-3000:])
    return folders


def test_off_the_free_mode_run_equals_the_base_commit_packets_included(one_proposal_runs):
    """Test: off the free mode run equals the base commit packets included."""
    base, off = one_proposal_runs["base"], one_proposal_runs["off"]
    files = sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(off).as_posix() for p in off.rglob("*") if p.is_file())
    assert any(name.startswith(f"{RUN_STEM}_packets/") for name in files)
    for name in files:
        if name in (f"{RUN_STEM}.json",) or name.startswith(f"{RUN_STEM}_proposer/"):
            continue
        assert (base / name).read_bytes() == (off / name).read_bytes(), name   # packets, npz, numerics, id pairs
    before = json.loads((base / f"{RUN_STEM}.json").read_text())
    after = json.loads((off / f"{RUN_STEM}.json").read_text())
    assert strip_volatile(before) == strip_volatile(after)
    for name in files:
        if name.startswith(f"{RUN_STEM}_proposer/") and name.endswith(".json"):
            assert (without_call_latency(json.loads((base / name).read_text()))
                    == without_call_latency(json.loads((off / name).read_text()))), name


def test_on_the_free_mode_run_records_the_option_and_the_packet_moves_only_in_the_grammar_block(one_proposal_runs):
    """Test: on the free mode run records the option and the packet moves only in the grammar block."""
    off, on = one_proposal_runs["off"], one_proposal_runs["on"]
    packets = sorted(p.name for p in (off / f"{RUN_STEM}_packets").iterdir())
    assert packets and packets == sorted(p.name for p in (on / f"{RUN_STEM}_packets").iterdir())
    for name in packets:
        a = json.loads((off / f"{RUN_STEM}_packets" / name).read_text())
        b = json.loads((on / f"{RUN_STEM}_packets" / name).read_text())
        assert {key for key in a if a[key] != b[key]} == {"constraints"}
        assert {key for key in a["constraints"] if a["constraints"][key] != b["constraints"][key]} == {"grammar"}
        changed = {key for key in a["constraints"]["grammar"]
                   if a["constraints"]["grammar"][key] != b["constraints"]["grammar"][key]}
        assert changed == {"calls", "catalogue_not_expressible"}
        assert b["constraints"]["grammar"]["calls"] == TRIG_CALLS
        assert b["constraints"]["grammar"]["catalogue_not_expressible"] == {"K12": trig_module.INV_R_TMOD_REASON}
    record = json.loads((on / f"{RUN_STEM}.json").read_text())
    proposals = record["arm_config"]["structure_proposals"]
    assert proposals["calls"] == TRIG_CALLS and proposals["n_ops_max"] == 10
    assert proposals["grammar_trig"]["enabled"] is True and proposals["grammar_trig"]["calls_added"] == {"cos": 1, "sin": 1}
    assert record["code_sha256"]["fb1_grammar_trig.py"] == trig_module.module_sha256()
    assert record["label"]["binding"]["settings"]["grammar_trig"] == ["cos", "sin"]
    assert record["proposer"]["constraints"]["grammar"]["calls"] == TRIG_CALLS
    plain = json.loads((off / f"{RUN_STEM}.json").read_text())
    assert "grammar_trig" not in plain["arm_config"]["structure_proposals"]
    assert "fb1_grammar_trig.py" not in plain["code_sha256"]
