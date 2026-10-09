"""Opt-in stable evaluation of log(1 - exp(z)).

Every expression here is a synthetic probe written for this test (a Weibull CDF on the
logit scale, logit(1 - exp(-H)) = log(1 - exp(-H)) + H, and plain log1mexp spellings),
not a recorded LLM reply. The probe reproduces the failure class of the 2026-10-04
survival seed-43 expected information gain (EIG) run: for H below ~1e-16, 1 - exp(-H) rounds to 0 and log gives -inf.
"""

import sys
from decimal import Decimal, getcontext

import numpy as np
import pytest

from . import boxinggym_core
from .count_models import (
    LOG1MEXP_BRANCH_POINT,
    Hypothesis,
    evaluate,
    log1mexp,
    log1mexp_sites,
    particle_logpmf,
    set_stable_log1mexp,
    stable_log1mexp_enabled,
)

# A two-input binary interface of the BoxingGym masked shape (declared X1, X2 plus derived inputs).
INTERFACE = {
    "derived_inputs": {
        "X101": "(X1-0.0)/10.0",
        "X1z": "2*(X1-0.0)/(10.0-0.0)-1",
        "X201": "(X2-0.0)/1.0",
        "X2z": "2*(X2-0.0)/(1.0-0.0)-1",
    },
    "inputs": [{"name": "X1", "range": [0.0, 10.0]}, {"name": "X2", "range": [0.0, 1.0]}],
    "link": "logit",
    "outcome_type": "binary",
    "support": [0.0, 1.0],
    "trials": 1,
}
INPUT_NAMES = ("X1", "X2", "X101", "X1z", "X201", "X2z")
WEIBULL_LOGIT = "log(1 - exp(-power(c1*X1 + 0.01, k))) + power(c1*X1 + 0.01, k)"
WEIBULL_PARAMETERS = (("c1", "positive"), ("k", "shape"))


@pytest.fixture(autouse=True)
def restore_switch():
    """Autouse fixture: turn the stable log1mexp switch off for each test and restore the previous setting afterwards."""
    previous = stable_log1mexp_enabled()
    set_stable_log1mexp(False)
    yield
    set_stable_log1mexp(previous)


def reference_log1mexp(z):
    """log(1 - exp(z)) in decimal arithmetic with 60 digits beyond |z|'s exponent, for z < 0."""
    getcontext().prec = 60 + max(0, int(-np.log10(-z)))
    return float((Decimal(1) - Decimal(float(z)).exp()).ln())


def weibull_logit_reference(c1, k, x):
    """Reference value of the Weibull CDF on the logit scale, in decimal arithmetic."""
    getcontext().prec = 60
    h = (Decimal(c1) * Decimal(x) + Decimal("0.01")) ** Decimal(k)
    return float((Decimal(1) - (-h).exp()).ln() + h)


def test_switch_is_off_by_default_and_absent_from_the_cli_unless_given(monkeypatch):
    """The switch is off by default and absent from the CLI namespace unless given."""
    assert stable_log1mexp_enabled() is False
    captured = []
    monkeypatch.setattr(boxinggym_core, "run", captured.append)
    monkeypatch.setattr(sys, "argv", ["boxinggym_core", "--output", "/dev/null/never", "--task", "survival"])
    boxinggym_core.main()
    assert not hasattr(captured[-1], "stable_log1mexp")  # SUPPRESS: no config.json key when omitted
    monkeypatch.setattr(sys, "argv", ["boxinggym_core", "--output", "/dev/null/never", "--task", "survival",
                                      "--stable-log1mexp"])
    boxinggym_core.main()
    assert captured[-1].stable_log1mexp is True


@pytest.mark.parametrize(
    ("expression", "sites"),
    [
        ("log(1 - exp(-u))", 1),
        ("log(1.0-exp(z))", 1),
        ("log1p(-exp(z))", 1),
        ("log(1 - exp(u)) + log1p(-exp(-u))", 2),
        ("log(1 - exp(-power(c1*X1 + 0.01, k))) + power(c1*X1 + 0.01, k)", 1),
        ("log(-expm1(u))", 0),  # already stable, left alone
        ("log(2 - exp(u))", 0),
        ("log(1 + exp(u))", 0),
        ("log(exp(u) - 1)", 0),
        ("log(1 - 2*exp(u))", 0),
        ("1 - exp(u)", 0),
        ("log1p(exp(u))", 0),
    ],
)
def test_only_the_two_log1mexp_spellings_are_rewritten(expression, sites):
    """Only the two log1mexp spellings are rewritten; other expressions are left alone."""
    assert log1mexp_sites(expression) == sites


def test_synthetic_weibull_probe_is_minus_inf_without_and_finite_with_the_switch():
    """(a) c1 = 1, k = 9 at X1 = 0.002768, X2 = 0: H = 0.012768**9 = 9.0e-18."""
    x = np.array([0.002768])
    params = {"c1": np.array([1.0]), "k": np.array([9.0])}
    inputs = {"X1": x, "X2": np.array([0.0])}
    with np.errstate(all="ignore"):
        off = evaluate(WEIBULL_LOGIT, params, x, np, inputs)
    assert np.isneginf(off).all()
    set_stable_log1mexp(True)
    on = evaluate(WEIBULL_LOGIT, params, x, np, inputs)
    expected = weibull_logit_reference(1.0, 9.0, 0.002768)
    assert np.isfinite(on).all()
    assert abs(expected - (-39.25)) < 0.01  # 9*log(0.012768) + H
    np.testing.assert_allclose(on, expected, rtol=1e-14)


def test_particle_logpmf_refuses_without_and_scores_with_the_switch():
    """The exact call that ended the run: particle_logpmf on held-out-like designs."""
    hypothesis = Hypothesis(WEIBULL_LOGIT, "bernoulli", WEIBULL_PARAMETERS, INPUT_NAMES)
    # kept below logit saturation (h < 36), so every outcome stays possible and only the
    # underflow at X1 = 0.002768 (c1 = 1, k = 9) is in play
    particles = {"c1": np.array([1.0, 0.5, 2.0]), "k": np.array([9.0, 2.0, 3.0])}
    designs = np.array([[0.002768, 0.0], [0.5, 1.0], [0.05, 0.0]])
    with pytest.raises(ValueError, match="non-finite mean expression"):
        particle_logpmf(hypothesis, particles, designs, 1, INTERFACE)
    set_stable_log1mexp(True)
    logpmf = particle_logpmf(hypothesis, particles, designs, 1, INTERFACE)
    assert logpmf.shape == (3, 3, 2) and np.isfinite(logpmf).all()
    np.testing.assert_allclose(np.exp(logpmf).sum(axis=-1), 1.0, rtol=1e-12)


def test_switch_changes_nothing_where_the_naive_form_is_accurate():
    """(b) Grid and random particles: on == off within 1e-12 (rtol and atol) wherever the
    naive form has no cancellation; near z = 0 only the switched path matches the
    60-digit reference (the naive form loses digits there, the reason for this change)."""
    z = -np.concatenate([np.logspace(-3, np.log10(700.0), 400), [np.log(2.0), 1.0, 30.0, 40.0, 700.0]])
    with np.errstate(all="ignore"):
        off = np.log(1 - np.exp(z))
    on = log1mexp(z, np)
    assert np.isfinite(off).all() and np.isfinite(on).all()
    np.testing.assert_allclose(on, off, rtol=1e-12, atol=1e-12)
    near = -np.logspace(-300, -3, 200)
    reference = np.array([reference_log1mexp(v) for v in near])
    np.testing.assert_allclose(log1mexp(near, np), reference, rtol=1e-14)
    # random particles of the probe, prior-like draws, uniform designs
    rng = np.random.default_rng(20261005)
    c1 = rng.lognormal(0.0, 1.0, size=(400, 1))
    k = rng.gamma(2.0, 1.0, size=(400, 1))
    x = rng.uniform(0.0, 10.0, size=(1, 60))
    inputs = {"X1": x, "X2": np.zeros_like(x)}
    with np.errstate(all="ignore"):
        naive = evaluate(WEIBULL_LOGIT, {"c1": c1, "k": k}, x, np, inputs)
    set_stable_log1mexp(True)
    stable = evaluate(WEIBULL_LOGIT, {"c1": c1, "k": k}, x, np, inputs)
    finite = np.isfinite(naive)
    assert np.isfinite(stable).all()
    # h = (c1*x + 0.01)**k; where 1 - exp(-h) has no cancellation (h >= 1e-3) the two agree
    h = (c1 * x + 0.01) ** k
    accurate = finite & (h >= 1e-3)
    assert accurate.sum() > 10000
    np.testing.assert_allclose(stable[accurate], naive[accurate], rtol=1e-12, atol=1e-12)


def test_branch_point_and_limits():
    """The stable evaluation agrees with the reference at the branch point and at the limits."""
    assert LOG1MEXP_BRANCH_POINT == pytest.approx(-0.6931471805599453, abs=0)
    with np.errstate(all="ignore"):
        values = log1mexp(np.array([-np.inf, 0.0, 1.0, np.nan]), np)
    assert values[0] == 0.0 and np.isneginf(values[1]) and np.isnan(values[2]) and np.isnan(values[3])


def test_pytensor_path_compiles_and_matches_numpy():
    """(c) The fit graph (backend = pytensor.tensor) uses the same function."""
    pytensor = pytest.importorskip("pytensor")
    pt = pytest.importorskip("pytensor.tensor")
    c1, k, x = pt.dvector("c1"), pt.dvector("k"), pt.dvector("x")
    values = {
        "c1": np.array([1.0, 0.5, 2.0, 1.3, 0.2]),
        "k": np.array([9.0, 2.0, 12.0, 0.7, 4.0]),
        "x": np.array([0.002768, 5.0, 0.05, 9.0, 1.0]),
    }
    numpy_inputs = {"X1": values["x"], "X2": np.zeros(5)}
    set_stable_log1mexp(True)
    graph = evaluate(WEIBULL_LOGIT, {"c1": c1, "k": k}, x, pt, {"X1": x, "X2": pt.zeros_like(x)})
    compiled = pytensor.function([c1, k, x], graph)
    from_pytensor = compiled(values["c1"], values["k"], values["x"])
    from_numpy = evaluate(WEIBULL_LOGIT, {"c1": values["c1"], "k": values["k"]}, values["x"], np, numpy_inputs)
    assert np.isfinite(from_pytensor).all()
    np.testing.assert_allclose(from_pytensor, from_numpy, rtol=1e-12, atol=1e-12)
    expected = [weibull_logit_reference(a, b, c) for a, b, c in zip(values["c1"], values["k"], values["x"])]
    np.testing.assert_allclose(from_pytensor, expected, rtol=1e-12)


def test_pytensor_already_stabilises_the_naive_spelling_in_the_fit_graph():
    """Documents PyTensor 2.20.0 behaviour (not a change made here): with the switch OFF the
    compiled fit graph rewrites log(1 - exp(x)) into its own log1mexp op, so the fit was finite
    while the NumPy prediction path returned -inf. That is why the 2026-10-04 run fitted the
    model (finite log evidence) and then died at the after_proposal checkpoint. With the switch
    ON both backends evaluate the same explicit log1mexp."""
    pytensor = pytest.importorskip("pytensor")
    pt = pytest.importorskip("pytensor.tensor")
    c1, k, x = pt.dvector("c1"), pt.dvector("k"), pt.dvector("x")
    graph = evaluate(WEIBULL_LOGIT, {"c1": c1, "k": k}, x, pt, {"X1": x, "X2": pt.zeros_like(x)})
    compiled = pytensor.function([c1, k, x], graph)
    value = compiled(np.array([1.0]), np.array([9.0]), np.array([0.002768]))
    assert np.isfinite(value).all()
    np.testing.assert_allclose(value, weibull_logit_reference(1.0, 9.0, 0.002768), rtol=1e-12)
    with np.errstate(all="ignore"):
        numpy_off = evaluate(WEIBULL_LOGIT, {"c1": np.array([1.0]), "k": np.array([9.0])},
                             np.array([0.002768]), np, {"X1": np.array([0.002768]), "X2": np.zeros(1)})
    assert np.isneginf(numpy_off).all()
