"""NewtonBench judge: the transcriptions against the adapter's law values, every declared spelling
judged equivalent to its own target, the P-edition derived inputs, every state, containment and membership, the
surrogate rule (m4, m10) and the not-representable world (m7).  Candidates and configs are synthetic: the box comes
from the adapter's task_box (benchmark code), and the transform numbers are made up here."""
import math

import pytest
import sympy as sp
from judge import newtonbench as NG

DEPTH = sorted(NG.DECLARATIONS)


def synthetic_config(task_id, presentation=True):
    """A config.json-shaped dict for one task: the box from the adapter, made-up z-score and log-centre numbers."""
    domain, difficulty, version = task_id.split("__")
    box_kind = "newtonbench_evaluate_law" if domain == "m5_radioactive_decay" else "autoscilab_registry"
    ad = NG.adapter()
    names, bounds, log_names = ad.task_box({"domain": domain, "input_box": box_kind, "system": "vanilla_equation"})
    masked = [f"X{i + 1}" for i in range(len(names))]
    cfg = {
        "benchmark": "newtonbench",
        "task": {"id": task_id, "benchmark": "newtonbench"},
        "input_name_map": dict(zip(masked, names)),
        "benchmark_adapter": {"benchmark": "newtonbench", "domain": domain, "difficulty": difficulty,
                              "law_version": version, "input_names": list(names), "input_box": box_kind,
                              "input_bounds": {n: list(bounds[n]) for n in names},
                              "log_sampled_inputs": sorted(log_names), "system": "vanilla_equation",
                              "law_function": f"_ground_truth_law_{difficulty}_{version}"},
        "transform": {"center": [0.5 * (bounds[n][0] + bounds[n][1]) for n in names],
                      "scale": [0.25 * (bounds[n][1] - bounds[n][0]) for n in names]},
    }
    if presentation:
        lc = {}
        for x, n in zip(masked, names):
            c = 0.5 * (math.log(bounds[n][0]) + math.log(bounds[n][1])) + 0.125   # made-up centre
            lc[x + "l"] = {"input": x, "centre": c,
                           "definition": f"log({x})-{c!r}" if c >= 0 else f"log({x})+{-c!r}"}
        cfg["benchmark_presentation"] = {"edition": "P (synthetic)", "log_centred_inputs": lc,
                                         "response_unit": 3.25, "response_unit_rule": "geometric_mean_of_d0"}
    return cfg


_TARGETS = {}


def target(task_id):
    if task_id not in _TARGETS:
        _TARGETS[task_id] = NG.Target(synthetic_config(task_id))
    return _TARGETS[task_id]


def j(expr, roles, task_id):
    return NG.judge({"expression": expr, "parameters": roles, "family": "lognormal", "model_id": "synthetic"},
                    target(task_id))


def roles_of(spelling, task_id):
    t = target(task_id)
    import ast
    tree = ast.parse(spelling, mode="eval")
    calls = {nd.func.id for nd in ast.walk(tree) if isinstance(nd, ast.Call)}
    names = {nd.id for nd in ast.walk(tree) if isinstance(nd, ast.Name)} - calls - set(t.X) - set(t.derived)
    return {n: ("intercept" if n == "a" else "positive") for n in sorted(names)}


def test_transcriptions_match_the_adapter_law_values():
    for task_id in DEPTH:
        t = target(task_id)
        assert t.check["points"] >= NG.C.N_DIAG, task_id
        if t.kind == "law":
            assert t.check["max_rel_deviation"] <= NG.TRANSCRIPTION_TOL, (task_id, t.check)
            assert t.check["law_text_check"]["max_rel_deviation"] <= NG.TRANSCRIPTION_TOL, (task_id, t.check)
        elif t.kind == "surrogate":
            assert 0 < t.check["surrogate_max_rel_error"] < NG.NOISE, (task_id, t.check)
        else:
            assert t.check["max_rel_deviation"] <= NG.TRANSCRIPTION_TOL, (task_id, t.check)


def test_representability_follows_section_13():
    kinds = {task_id: target(task_id).kind for task_id in DEPTH}
    assert kinds["m7_malus_law__medium__v0"] == "not_representable"
    assert kinds["m4_snell_law__medium__v2"] == kinds["m10_be_distribution__medium__v1"] == "surrogate"
    assert sum(k == "law" for k in kinds.values()) == 6
    for task_id in DEPTH:
        t = target(task_id)
        assert t.representable == (t.kind != "not_representable"), task_id
    assert "outside the caps" not in target("m10_be_distribution__medium__v1").check["true_law_spelling"]


def test_every_declared_spelling_is_judged_equivalent_to_its_target():
    for task_id in DEPTH:
        decl = NG.DECLARATIONS[task_id]
        if decl["kind"] == "not_representable":
            continue
        r = j(decl["spelling"], roles_of(decl["spelling"], task_id), task_id)
        assert r["state"] == "equivalent", (task_id, r["reason"])
        assert r["reason"]["code"] == ("E_SURROGATE" if decl["kind"] == "surrogate" else "E_CANONICAL"), task_id
        assert r["world_law_member"] is True, task_id


def test_log_centred_and_raw_log_spellings_agree_and_unknown_names_are_parse_errors():
    tid = "m2_magnetic_force__medium__v0"
    raw = j("g + 1.5*log(X1*X2) - 3*log(X3)", {"g": "intercept"}, tid)
    assert raw["state"] == "equivalent"
    t = target(tid)
    assert sp.simplify(t.derived["X1l"] - (sp.log(t.X["X1"]) - NG.C._literal(
        synthetic_config(tid)["benchmark_presentation"]["log_centred_inputs"]["X1l"]["centre"]))) == 0
    bad = j("g + 1.5*X9l", {"g": "intercept"}, tid)
    assert bad["state"] == "unresolved" and bad["reason"]["code"] == "U_PARSE"
    trig = j("g + log(sin(X1))", {"g": "intercept"}, tid)        # sin is not a grammar v2 call
    assert trig["state"] == "unresolved" and trig["reason"]["code"] == "U_PARSE"


def test_free_exponents_are_a_superfamily_holding_the_world_law():
    r = j("g + p*X1l + q*X2l + w*X3l", {"g": "intercept", "p": "signed", "q": "signed", "w": "signed"},
          "m2_magnetic_force__medium__v0")
    assert r["state"] == "non_equivalent" and r["reason"]["code"] == "N_DIMENSION"
    assert r["containment"] is True and r["world_law_member"] is True


def test_wrong_exponent_and_missing_input_are_non_equivalent():
    tid = "m11_heat_transfer__medium__v1"
    wrong = j("g - 2*X1l + X2l - X3l", {"g": "intercept"}, tid)
    assert wrong["state"] == "non_equivalent" and wrong["world_law_member"] is False
    missing = j("g - 2.5*X1l + X2l", {"g": "intercept"}, tid)
    assert missing["state"] == "non_equivalent" and missing["reason"]["code"] == "N_DEPENDENCE"


def test_free_rate_coefficient_in_the_decay_is_containment():
    tid = "m5_radioactive_decay__medium__v0"
    ok = j("g + 1.2*X1l - X2*X3**1.5", {"g": "intercept"}, tid)
    assert ok["state"] == "equivalent"
    wide = j("g + 1.2*X1l - k*X2*power(X3, 1.5)", {"g": "intercept", "k": "positive"}, tid)
    assert wide["state"] == "non_equivalent" and wide["containment"] is True and wide["world_law_member"] is True


def test_two_term_law_and_a_restricted_member():
    tid = "m9_hooke_law__medium__v0"
    renamed = j("h + log(square(X1) + m*X1)", {"h": "intercept", "m": "positive"}, tid)
    assert renamed["state"] == "equivalent"
    one_term = j("h + 2*X1l", {"h": "intercept"}, tid)
    assert one_term["state"] == "non_equivalent" and one_term["world_law_member"] is False


def test_unresolved_when_not_provable():
    # log(1 - tanh(u)^2) + 2 log(exp(u) + exp(-u)) - 2 log 2 = 0 for every u: the same family, not rewritten by the
    # canonical form (tanh is never turned into exponentials)
    tid = "m3_fourier_law__medium__v0"
    r = j("g + X1l + 0.5*X2l + X3l - 2*X4l + log(1 - square(tanh(X4))) + 2*log(exp(X4) + exp(-X4)) - 2*log(2)",
          {"g": "intercept"}, tid)
    assert r["state"] == "unresolved" and r["reason"]["code"] == "U_SYMBOLIC_UNPROVEN"


def test_not_representable_world():
    r = j("g + X1l + w*X2", {"g": "intercept", "w": "signed"}, "m7_malus_law__medium__v0")
    assert r["state"] == "not_representable" and r["reason"]["code"] == "R_NOT_REPRESENTABLE"
    assert "sin and cos" in r["reason"]["detail"]


def test_surrogate_rule():
    tid = "m4_snell_law__medium__v2"
    ok = j("g + log(X1*X3/X2)", {"g": "intercept"}, tid)
    assert ok["state"] == "equivalent" and ok["reason"]["code"] == "E_SURROGATE"
    other = j("g + X1l + X3l", {"g": "intercept"}, tid)               # n2 missing
    assert other["state"] == "unresolved" and other["reason"]["code"] == "U_NOT_IDENTIFIABLE"
    assert other["surrogate_comparison"]["code"] == "N_DEPENDENCE"
    tid = "m10_be_distribution__medium__v1"
    exact = j("g - log(exp(c*sqrt(X1)*X2) - 1)", {"g": "intercept", "c": "positive"}, tid)
    assert exact["state"] == "unresolved" and exact["reason"]["code"] == "U_NOT_IDENTIFIABLE"
    assert exact["true_law_match"] == "equivalent"
    surrogate = j("g - 0.5*X1l - X2l", {"g": "intercept"}, tid)
    assert surrogate["state"] == "equivalent" and surrogate["true_law_match"] != "equivalent"


def test_undeclared_task_and_a_foreign_config_are_refused():
    cfg = synthetic_config("m2_magnetic_force__medium__v0")
    cfg["task"]["id"] = "m2_magnetic_force__easy__v0"
    cfg["benchmark_adapter"]["difficulty"] = "easy"
    with pytest.raises(ValueError, match="no declared C5 target"):
        NG.Target(cfg)
    cfg = synthetic_config("m3_fourier_law__medium__v0")
    cfg["benchmark_adapter"]["input_bounds"]["k"] = [0.2, 10.0]
    with pytest.raises(ValueError, match="task_box"):
        NG.Target(cfg)
    cfg = synthetic_config("m3_fourier_law__medium__v0")
    cfg["benchmark_presentation"]["log_centred_inputs"]["X1l"]["definition"] = "log(X1)-0.5"
    with pytest.raises(ValueError, match="differs from"):
        NG.Target(cfg)


def test_registered_edition_without_log_centred_inputs():
    t = NG.Target(synthetic_config("m8_sound_speed__medium__v0", presentation=False))
    assert "X1l" not in t.derived and "X1z" in t.derived
    r = NG.judge({"expression": "g + 0.5*log(X1) + log(X2) - 0.75*log(X3)", "parameters": {"g": "intercept"},
                  "family": "lognormal", "model_id": "synthetic"}, t)
    assert r["state"] == "equivalent"
