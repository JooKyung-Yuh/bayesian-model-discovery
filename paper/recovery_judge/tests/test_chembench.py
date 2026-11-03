"""ChemBench judge: the oracle transcription, every state, the plan's reparametrization example, nuisance handling,
containment and a restricted (literal-constant) family.  Candidates are synthetic."""
import numpy as np
import sympy as sp
from judge import chembench as CB

TRANSFORM = {"center": [10.0, 25.0, 10.0, 10.0, 1.5, 320.0, 7.0], "scale": [20.0, 15.0, 20.0, 6.0, 2.5, 26.0, 1.7]}


def mm():
    return CB.Target("c0_michaelis_menten__medium__v1", TRANSFORM)


def j(expr, roles, target):
    return CB.judge({"expression": expr, "parameters": roles, "family": "lognormal", "model_id": "synthetic"}, target)


def test_every_transcription_matches_the_oracle_up_to_a_constant():
    cb = CB.oracle()
    rng = np.random.default_rng(7)
    for dom, (law, pmap) in CB.LAWS.items():
        p = cb._PARAMS[dom]["easy"]["v1"]
        subs = {sym: (p[name] / CB.R_GAS if name == "Ea" else p[name]) for name, sym in pmap.items()}
        f = sp.lambdify([CB.A, CB.I, CB.B, CB.P, CB.E, CB.T], law.xreplace(subs), "numpy")
        ratios = []
        for _ in range(12):
            v = [float(np.exp(rng.uniform(np.log(0.01), np.log(100)))), float(rng.uniform(0, 50)),
                 float(np.exp(rng.uniform(np.log(0.01), np.log(100)))), float(rng.uniform(0, 20)),
                 float(np.exp(rng.uniform(np.log(0.01), np.log(10)))), float(rng.uniform(278, 368))]
            ratios.append(cb._RATE_FNS[dom](p, *v, 7.0) / f(*v))
        r = np.array(ratios)
        assert np.max(np.abs(r / r.mean() - 1)) < 1e-10, dom
        assert CB.representability(dom)[0], dom


def test_plan_reparametrization_example_is_equivalent():
    # a x / (b + x)  ==  x / (t1 + t2 x)  (plan p.9), on the Michaelis-Menten primary law, with the nuisance pH term
    t = mm()
    r = j("log(X5) + log(X1) - log(t1 + t2*X1) - w*square(X7z)", {"t1": "positive", "t2": "positive", "w": "positive"}, t)
    assert r["state"] == "equivalent", r["reason"]
    assert r["separable"] is True and r["nuisance_part"] != "0"


def test_renamed_coefficients_match():
    t = mm()
    a = j("g + log(X5*X1/(h + X1))", {"g": "intercept", "h": "positive"}, t)
    b = j("q + log(X5*X1/(m + X1))", {"q": "intercept", "m": "positive"}, t)
    assert a["state"] == b["state"] == "equivalent"


def test_extra_inhibition_term_is_containment():
    t = mm()
    r = j("g + log(X5*X1/(h*(1 + X2/k) + X1))", {"g": "intercept", "h": "positive", "k": "positive"}, t)
    assert r["state"] == "non_equivalent" and r["containment"] is True


def test_literal_constant_is_a_restricted_family():
    t = mm()   # medium v1: Km = 50 in the oracle table
    r = j("g + log(X5*X1/(50 + X1))", {"g": "intercept"}, t)
    assert r["state"] == "non_equivalent" and r["reason"]["code"] == "N_DIMENSION"
    assert r["world_law_member"] is True
    q = j("g + log(X5*X1/(3 + X1))", {"g": "intercept"}, t)
    assert q["world_law_member"] is False


def test_unresolved_when_not_provable():
    # X1/(K + X1) = (1 + tanh((log X1 - log K)/2)) / 2: the same family, not rewritten by the canonical form
    t = mm()
    r = j("g + log(X5) + log(1 + tanh((log(X1) - h)/2))", {"g": "intercept", "h": "signed"}, t)
    assert r["state"] == "unresolved" and r["reason"]["code"] == "U_SYMBOLIC_UNPROVEN"


def test_non_equivalent_wrong_substrate():
    t = mm()
    r = j("g + log(X5*X3/(h + X3))", {"g": "intercept", "h": "positive"}, t)
    assert r["state"] == "non_equivalent" and r["reason"]["code"] == "N_DEPENDENCE"


def test_not_representable_when_the_spelling_exceeds_the_caps(monkeypatch):
    long = "a + log(X5*X1/(K + X1)) " + " ".join(f"+ log(1 + X{i}/K{i})" for i in (2, 3, 4)) + " + log(1 + X2*X3/K5)"
    monkeypatch.setitem(CB.SPELLINGS, "c0_michaelis_menten", long)
    t = mm()
    assert not t.representable
    r = j("g + log(X5*X1/(h + X1))", {"g": "intercept", "h": "positive"}, t)
    assert r["state"] == "not_representable"
