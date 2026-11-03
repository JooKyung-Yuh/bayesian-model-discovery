"""BoxingGym judge: every state, a reparametrization equivalence, a renamed-coefficient match, containment.
All candidates and world parameters are synthetic (written for these tests, not taken from any run record)."""
from judge import boxinggym as BG

DEATH_ITF = {"inputs": [{"name": "X1", "range": [0.001, 2.0]}], "link": "logit", "trials": 50,
             "derived_inputs": {"X1z": "2*(X1-0.001)/(2.0-0.001)-1", "X101": "(X1-0.001)/1.999"}}
POS_ITF = {"inputs": [{"name": "X1", "range": [0.0, 5.0]}], "link": "log", "trials": None,
           "derived_inputs": {"X1z": "2*(X1-0.0)/(5.0-0.0)-1", "X101": "(X1-0.0)/5.0"}}
SURV_ITF = {"inputs": [{"name": "X1", "range": [0.0, 10.0]}, {"name": "X2", "range": [0.0, 1.0]}], "link": "logit",
            "trials": 1, "derived_inputs": {"X2z": "2*(X2-0.0)/(1.0-0.0)-1", "X201": "(X2-0.0)/1.0"}}
HYP_ITF = {"inputs": [{"name": "X1", "range": [1.0, 300.0]}, {"name": "X2", "range": [1.0, 300.0]},
                      {"name": "X3", "range": [1.0, 365.0]}], "link": "logit", "trials": 1, "derived_inputs": {}}


def death():
    return BG.Target("death", {"theta": 0.7}, DEATH_ITF)


def j(expr, roles, target, family="binomial"):
    return BG.judge({"expression": expr, "parameters": roles, "family": family, "model_id": "synthetic"}, target)


def test_equivalent_and_reparametrized():
    t = death()
    a = j("log(exp(z*X1) - 1)", {"z": "positive"}, t)
    b = j("log(1 - exp(-w*X1)) + w*X1", {"w": "positive"}, t)
    assert a["state"] == "equivalent" and b["state"] == "equivalent"
    assert a["parameter_mapping"]["target_from_candidate"] == {"T_theta": "p_z"}
    assert a["containment"] is False


def test_renamed_coefficient_matches():
    t = death()
    assert j("log(exp(q1*X1) - 1)", {"q1": "positive"}, t)["state"] == "equivalent"


def test_superfamily_is_containment_not_equivalence():
    t = death()
    r = j("c0 + log(exp(z*X1) - 1)", {"c0": "intercept", "z": "positive"}, t)
    assert r["state"] == "non_equivalent" and r["containment"] is True
    assert r["containment_basis"] == "symbolic"


def test_non_equivalent():
    t = death()
    r = j("b0 + b1*X1z", {"b0": "intercept", "b1": "signed"}, t)
    assert r["state"] == "non_equivalent" and r["containment"] is False


def test_unresolved_when_not_provable():
    # tanh(z X1 / 2) (exp(z X1) + 1) = exp(z X1) - 1: the same family, but the canonical form does not rewrite tanh
    t = death()
    r = j("log(tanh(z*X1/2)*(exp(z*X1) + 1))", {"z": "positive"}, t)
    assert r["state"] == "unresolved" and r["reason"]["code"] == "U_SYMBOLIC_UNPROVEN"


def test_unresolved_on_parse_failure():
    t = death()
    r = j("erf(z*X1)", {"z": "positive"}, t)
    assert r["state"] == "unresolved" and r["reason"]["code"] == "U_PARSE"


def test_not_representable_world():
    t = BG.Target("hyperbolic", {"epsilon": 0.01, "truth": [0.02, 1.0]}, HYP_ITF)
    r = j("b*(X2/(1 + k*X3) - X1)", {"b": "positive", "k": "positive"}, t, family="bernoulli")
    assert r["state"] == "not_representable" and r["reason"]["code"] == "R_NOT_REPRESENTABLE"


def test_peregrines_cubic_in_standardized_input_is_equivalent():
    t = BG.Target("peregrines", {"alpha": 4.3, "beta1": 0.9, "beta2": 0.05, "beta3": -0.2}, POS_ITF, response_unit=50.0)
    r = j("c0 + c1*X1z + c2*square(X1z) + c3*power(X1z, 3)",
          {"c0": "intercept", "c1": "signed", "c2": "signed", "c3": "signed"}, t, family="poisson")
    assert r["state"] == "equivalent"
    q = j("c0 + c1*X1 + c2*square(X1)", {"c0": "intercept", "c1": "signed", "c2": "signed"}, t, family="poisson")
    assert q["state"] == "non_equivalent" and q["containment"] is False


def test_dugongs_power_and_exp_spellings():
    t = BG.Target("dugongs", {"a": 2.1, "b": 1.2, "l": 0.5}, POS_ITF, response_unit=1.5)
    for expr, roles in (("log(u - v*power(g, X1))", {"u": "positive", "v": "positive", "g": "positive"}),
                        ("log(u - v*exp(-g*X1))", {"u": "positive", "v": "positive", "g": "positive"})):
        assert j(expr, roles, t, family="lognormal")["state"] == "equivalent", expr


def test_survival_binary_slices():
    t = BG.Target("survival", {"lambda0": 0.6, "beta": 1.1}, SURV_ITF)
    r = j("h*X1*exp(g*X2)", {"h": "positive", "g": "signed"}, t, family="bernoulli")
    assert r["state"] == "equivalent"
    q = j("h0 + h1*X1 + h2*X2", {"h0": "intercept", "h1": "signed", "h2": "signed"}, t, family="bernoulli")
    assert q["state"] == "non_equivalent"


def test_zero_inflation_is_containment():
    t = death()
    r = j("log(exp(z*X1) - 1)", {"z": "positive"}, t, family="zi_binomial")
    assert r["state"] == "non_equivalent" and r["containment"] is True
