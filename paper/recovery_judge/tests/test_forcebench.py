"""ForceBench judge: coupling through F/p2, time terms, softening as containment, every state.  Synthetic candidates."""
from judge import forcebench as FB


def j(expr, roles, world):
    return FB.judge({"expression": expr, "parameters": roles, "family": "gauss", "model_id": "synthetic"}, FB.Target(world))


def test_coulomb_needs_the_product_coupling():
    ok = j("q*p1*square(p2)/square(r)", {"q": "positive"}, "coulomb_easy")
    assert ok["state"] == "equivalent" and ok["coupling_same"] is True
    wrong = j("q*p1*p2/square(r)", {"q": "positive"}, "coulomb_easy")   # acceleration q p1 / r^2: no p2 coupling
    assert wrong["state"] == "non_equivalent" and wrong["coupling_same"] is False
    assert wrong["rt_slice"] == "equivalent"


def test_gravity_reparametrized():
    r = j("p1/(k*r)", {"k": "positive"}, "gravity")
    assert r["state"] == "equivalent"
    assert r["parameter_mapping"]["target_from_candidate"] == {"T_c": "1/p_k"}


def test_softening_is_containment():
    r = j("q*p1*square(p2)/(square(r) + e)", {"q": "positive", "e": "positive"}, "coulomb_easy")
    assert r["state"] == "non_equivalent" and r["containment"] is True


def test_oscillator_phase_form_equivalent_and_cosine_only_restricted():
    full = j("q*p1*cos(w*t + h)/r", {"q": "positive", "w": "positive", "h": "signed"}, "oscillator")
    assert full["state"] == "equivalent", full["reason"]
    cos_only = j("q*p1*cos(w*t)/r", {"q": "signed", "w": "positive"}, "oscillator")
    assert cos_only["state"] == "non_equivalent" and cos_only["world_law_member"] is True
    static = j("q*p1/r", {"q": "positive"}, "oscillator")
    assert static["state"] == "non_equivalent" and static["reason"]["code"] == "N_DEPENDENCE"


def test_unresolved_when_not_provable():
    r = j("q*p1*square(p2)*(square(cos(r)) + square(sin(r)))/square(r)", {"q": "positive"}, "coulomb_easy")
    assert r["state"] == "unresolved" and r["reason"]["code"] == "U_SYMBOLIC_UNPROVEN"


def test_not_representable_worlds():
    for world in ("yukawa", "extra_dimensions"):
        assert j("q*p1/r", {"q": "positive"}, world)["state"] == "not_representable"


def test_catalogue_ids_resolve():
    sid, fam, expr, roles, src = FB.resolve("inv_r_nomass|student_t4", {})
    assert (sid, fam, src) == ("inv_r_nomass", "student_t4", "catalogue") and expr == "c*p1*p2/r"
