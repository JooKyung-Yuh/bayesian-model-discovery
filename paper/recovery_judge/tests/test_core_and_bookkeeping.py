"""Core rules on the plan's own examples and the bookkeeping test of plan p.28 (synthetic logs)."""
import sympy as sp
from judge import core as C
from judge import metrics as M


def _fam(expr, params, roles, inputs, num=None):
    return C.Family(expr, params, {p: C.role_domain(r) for p, r in zip(params, roles)}, inputs, num=num)


def test_plan_example_reparametrization():
    x = sp.Symbol("x", positive=True)
    a, b = C.param_symbol("a", "positive"), C.param_symbol("b", "positive")
    t1, t2 = C.param_symbol("t1", "positive", prefix="T_"), C.param_symbol("t2", "positive", prefix="T_")
    box = C.Box({x: (0.05, 10.0, "lin")}, "test|plan-p9")
    cand = _fam(a * x / (b + x), [a, b], ["positive", "positive"], [x])
    targ = _fam(x / (t1 + t2 * x), [t1, t2], ["positive", "positive"], [x])
    truth = {t1: 0.4, t2: 1.5}
    r = C.compare(cand, targ, box, truth=truth, members=[truth, {t1: 1.0, t2: 0.5}], key="test|p9")
    assert r["state"] == "equivalent"
    assert r["parameter_mapping"]["target_from_candidate"] == {"T_t1": "p_b/p_a", "T_t2": "1/p_a"}


def test_plan_example_containment():
    # truth a x + b, candidate alpha x + beta + gamma x^2: gamma = 0 matches, which is containment, not equivalence
    x = sp.Symbol("x", positive=True)
    al, be, ga = (C.param_symbol(n, "signed") for n in ("alpha", "beta", "gamma"))
    A, B = C.param_symbol("a", "signed", prefix="T_"), C.param_symbol("b", "signed", prefix="T_")
    box = C.Box({x: (0.0, 4.0, "lin")}, "test|plan-p9b")
    cand = _fam(al * x + be + ga * x ** 2, [al, be, ga], ["signed"] * 3, [x])
    targ = _fam(A * x + B, [A, B], ["signed"] * 2, [x])
    truth = {A: 0.8, B: -0.3}
    r = C.compare(cand, targ, box, truth=truth, members=[truth, {A: -1.2, B: 0.7}], key="test|p9b")
    assert r["state"] == "non_equivalent" and r["containment"] is True
    assert r["containment_witness"]["substitution"] == {"p_gamma": "0"}


def test_bookkeeping_found_then_retired():
    """Plan p.28: a model found at round 3 and retired at round 5 gives E_B = 1 and A_B = 0."""
    verdicts = {"m_true": {"state": "equivalent"}, "m_other": {"state": "non_equivalent"}}
    recs = []
    for t in range(7):
        ids = ["m_other"] + (["m_true"] if 3 <= t < 5 else [])
        ws = [1.0] if len(ids) == 1 else [0.4, 0.6]
        recs.append({"phase": "after_proposal" if t else "initial", "t": t, "n_observed": t + 2, "ids": ids,
                     "weights": ws, "engine_top": None})
    rows, s = M.run_metrics(recs, verdicts, rounds=6)
    assert s["E_B"] == 1 and s["A_B"] == 0 and s["T_A"] == 3 and s["censored"] == 0 and s["A_0"] == 0
    assert [r["A_t"] for r in rows] == [0, 0, 0, 1, 1, 0, 0]
    assert [r["S_t"] for r in rows] == [0, 0, 0, 1, 1, 0, 0]


def test_never_found_is_censored_at_the_budget():
    verdicts = {"m": {"state": "non_equivalent"}}
    recs = [{"phase": "initial", "t": 0, "n_observed": 2, "ids": ["m"], "weights": [1.0], "engine_top": "m"},
            {"phase": "after_proposal", "t": 1, "n_observed": 3, "ids": ["m"], "weights": [1.0], "engine_top": "m"}]
    rows, s = M.run_metrics(recs, verdicts, rounds=1)
    assert s["E_B"] == 0 and s["T_A"] == 1 and s["censored"] == 1


def test_top_tie_goes_to_the_smallest_id():
    assert M.top_of(["b", "a", "c"], [0.4, 0.4, 0.2]) == "a"


def _x_box(key):
    x = sp.Symbol("x", positive=True)
    return x, C.Box({x: (0.05, 10.0, "log")}, key)


def test_hill_spellings_reach_one_tree():
    """Rule 1.3: 1/(1 + (K/x)^n) and x^n/(K^n + x^n) are one family (powers of products split before fractions)."""
    x, box = _x_box("test|hill")
    s, K, n = (C.param_symbol(v, "positive") for v in ("s", "K", "n"))
    S, Kh, N = (C.param_symbol(v, "positive", prefix="T_") for v in ("S", "Kh", "N"))
    cand = _fam(s / (1 + (K / x) ** n), [s, K, n], ["positive"] * 3, [x])
    targ = _fam(S * x ** N / (Kh ** N + x ** N), [S, Kh, N], ["positive"] * 3, [x])
    truth = {S: 1.3, Kh: 0.8, N: 1.7}
    r = C.compare(cand, targ, box, truth=truth, members=[truth, {S: 0.7, Kh: 2.0, N: 0.9}], key="test|hill")
    assert r["state"] == "equivalent"


def test_split_exponent_and_log_identity_reach_one_tree():
    """Rule 1.3: x^(n + c) is x^alpha with a free leftover; a*exp(e log(3c + 3)) is one positive factor."""
    x, box = _x_box("test|merge")
    a, n, c = C.param_symbol("a", "positive"), C.param_symbol("n", "positive"), C.param_symbol("c", "signed")
    S, A = C.param_symbol("S", "positive", prefix="T_"), C.param_symbol("alpha", "positive", prefix="T_")
    cand = _fam(a * x ** n * x ** (c / 3), [a, n, c], ["positive", "positive", "signed"], [x])
    targ = _fam(S * x ** A, [S, A], ["positive"] * 2, [x])
    truth = {S: 1.1, A: 0.6}
    assert C.compare(cand, targ, box, truth=truth, members=[truth, {S: 2.0, A: 1.4}], key="test|merge")["state"] == "equivalent"
    k, e = C.param_symbol("k", "positive"), C.param_symbol("e", "signed")
    cand2 = _fam(a * x * sp.exp(e * sp.log(3 * k + 3)) / (n + x), [a, k, e, n], ["positive", "positive", "signed", "positive"], [x])
    targ2 = _fam(S * x / (A + x), [S, A], ["positive"] * 2, [x])
    r = C.compare(cand2, targ2, box, truth=truth, members=[truth, {S: 2.0, A: 1.4}], key="test|logs")
    assert r["state"] == "equivalent"


def test_one_solved_direction_is_not_a_domain_verdict():
    """Rule 1.3: when the reverse mapping needs Lambert W (not solved), N_DOMAIN_* needs a numeric witness."""
    x, box = _x_box("test|onedir")
    th = C.param_symbol("theta", "positive", prefix="T_")
    targ = _fam(th * sp.exp(th) * x, [th], ["positive"], [x])         # target slot theta e^theta, theta > 0
    truth = {th: 0.5}
    members = [truth, {th: 1.2}]
    q_real, q_pos = C.param_symbol("q", "signed"), C.param_symbol("q", "positive")
    wide = C.compare(_fam(q_real * x, [q_real], ["signed"], [x]), targ, box, truth=truth, members=members, key="t|w")
    assert wide["state"] == "non_equivalent" and wide["reason"]["code"] == "N_DOMAIN_WIDER" and wide["containment"]
    same = C.compare(_fam(q_pos * x, [q_pos], ["positive"], [x]), targ, box, truth=truth, members=members, key="t|s")
    assert same["state"] == "unresolved" and same["reason"]["code"] == "U_SYMBOLIC_UNPROVEN" and not same["containment"]
    p = C.param_symbol("p", "positive")
    T = C.param_symbol("T", "positive", prefix="T_")
    targ2 = _fam(T * x, [T], ["positive"], [x])
    narrow = C.compare(_fam((p * sp.exp(p) + 1) * x, [p], ["positive"], [x]), targ2, box, truth={T: 0.4},
                       members=[{T: 0.4}, {T: 2.0}], key="t|n")
    assert narrow["state"] == "non_equivalent" and narrow["reason"]["code"] == "N_DOMAIN_NARROWER"
    full = C.compare(_fam(p * sp.exp(p) * x, [p], ["positive"], [x]), targ2, box, truth={T: 0.4},
                     members=[{T: 0.4}, {T: 2.0}], key="t|f")
    assert full["state"] == "unresolved" and full["reason"]["code"] == "U_SYMBOLIC_UNPROVEN"


def test_lambert_type_equations_are_not_solved():
    """Rule 1.2: u inside and outside a function (u e^u = c) is no proof; u only inside or only outside still solves."""
    u = C.param_symbol("u", "positive")
    c = C.param_symbol("c", "positive", prefix="T_")
    k = C.param_symbol("k", "signed", prefix="T_")
    assert C._lambert_type(u * sp.exp(u) - c, u) and C._lambert_type(u + sp.log(u) - c, u)
    assert C._lambert_type(u * 2 ** u - c, u)
    assert not C._lambert_type(sp.exp(2 * u) + sp.exp(u) - c, u)
    assert not C._lambert_type(u ** 2 * sp.exp(k) - c, u) and not C._lambert_type(sp.sqrt(u) - c, u)
    assert C.solve_direction([(u * sp.exp(u) * sp.exp(k), c)], [u], [c, k]) is None
    sol = C.solve_direction([(u * sp.exp(k), c)], [u], [c, k])
    assert sol is not None and sp.simplify(sol[u] - c * sp.exp(-k)) == 0
    sol = C.solve_direction([(sp.exp(u + k), c)], [u], [c, k])
    assert sol is not None and sp.simplify(sp.exp(sol[u] + k) - c) == 0
