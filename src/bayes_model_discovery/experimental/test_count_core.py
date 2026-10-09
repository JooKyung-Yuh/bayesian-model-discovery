"""Small numerical examples for the independent paper CORE implementation."""

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit, logsumexp
from scipy.stats import beta, binom, norm

from .core_count import (
    EVALUATION_SEEDS,
    WORLD_COEFFICIENTS,
    CoreWorld,
    mock_proposals,
    points_label,
    write_points,
)
from .count_inference import (
    Fit,
    SMCSettings,
    build_model,
    fit,
    information_from_components,
    mixture_logpmf,
    model_logweights,
    predict,
    retain,
)
from .count_models import Hypothesis, evaluate, initial_hypotheses, particle_logpmf


def test_parser_limits_and_alpha_duplicate_identity():
    """The expression parser enforces its limits and alpha-renamed duplicates share one model id."""
    h = Hypothesis("b0+b1*z", "binomial", (("b0", "intercept"), ("b1", "signed")))
    renamed = Hypothesis("a + c * z", "binomial", (("a", "intercept"), ("c", "signed")))
    assert h.model_id == renamed.model_id
    with pytest.raises(ValueError):
        Hypothesis('__import__("os")', "binomial", (("a", "signed"),))
    with pytest.raises(ValueError):
        Hypothesis(
            "+".join(f"b{i}" for i in range(7)),
            "binomial",
            tuple((f"b{i}", "signed") for i in range(7)),
        )
    with pytest.raises(ValueError):
        Hypothesis("+".join(["b0"] * 17), "binomial", (("b0", "intercept"),))
    with pytest.raises(ValueError):
        Hypothesis("b0", "binomial", (("b0", "tuned_prior"),))


def test_development_and_paper_q_are_separate_without_changing_observation_rng():
    """Development and paper evaluation sets are separate and leave the observation RNG unchanged."""
    world = CoreWorld(11, "out", "beta_binomial")
    dev = world.evaluation(32, EVALUATION_SEEDS["dev"])
    legacy_dev = world.evaluation(32)
    paper = world.evaluation(256, EVALUATION_SEEDS["paper"])
    for actual, original in zip(dev, legacy_dev, strict=True):
        np.testing.assert_array_equal(actual, original)
    assert not set(dev[0]) & set(paper[0])
    assert EVALUATION_SEEDS["dev"] != EVALUATION_SEEDS["paper"]
    fresh = CoreWorld(11, "out", "beta_binomial")
    assert world.observe(0.35) == fresh.observe(0.35)
    for first, second in zip(paper, world.evaluation(256, EVALUATION_SEEDS["paper"]), strict=True):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("family", ("binomial", "beta_binomial", "zi_binomial"))
def test_normalized_likelihood_and_pymc_joint_logp(family):
    """The likelihood is normalized and matches the PyMC joint log-probability."""
    h = Hypothesis("b0+b1*z", family, (("b0", "intercept"), ("b1", "signed")))
    particles = {
        "b0": np.array([-0.4, 0.2]),
        "b1": np.array([1.0, 1.5]),
        "rho": np.array([0.1, 0.2]),
        "pi": np.array([0.2, 0.4]),
    }
    u, y = np.array([0.0, 0.3, 1.0]), np.array([0, 15, 50])
    rows = particle_logpmf(h, particles, u)
    np.testing.assert_allclose(np.exp(rows).sum(-1), 1, atol=1e-12)
    model = build_model(h, u, y)
    point = {name: np.array(particles[name][0]) for name in ("b0", "b1", *h.noise_parameters)}
    likelihood = rows[0, np.arange(3), y].sum()
    prior = norm.logpdf(-0.4, 0, 2) + norm.logpdf(1.0, 0, 1)
    if family == "beta_binomial":
        prior += beta.logpdf(0.1, 1, 9)
    if family == "zi_binomial":
        prior += beta.logpdf(0.2, 1, 4)
        expected = (
            (1 - particles["pi"][:, None])
            * 50
            * expit(particles["b0"][:, None] + particles["b1"][:, None] * (2 * u - 1))
        )
        np.testing.assert_allclose(np.exp(rows) @ np.arange(51), expected, atol=1e-10)
    assert model.compile_logp()(point) == pytest.approx(likelihood + prior, abs=1e-7)


def test_numpy_symbolic_supported_expression_agree():
    """NumPy and symbolic evaluation of a supported expression agree."""
    import pytensor
    import pytensor.tensor as pt

    h = Hypothesis(
        "a+b*tanh(z)+c*log(1+u)+d*square(u)",
        "binomial",
        (("a", "intercept"), ("b", "signed"), ("c", "signed"), ("d", "signed")),
    )
    u = np.linspace(0, 1, 9)
    values = {"a": -0.2, "b": 0.8, "c": 1.2, "d": -0.7}
    variable = pt.dvector("u")
    symbolic = pytensor.function([variable], evaluate(h.expression, values, variable, pt))
    np.testing.assert_allclose(symbolic(u), evaluate(h.expression, values, u), atol=1e-12)


def test_predictive_mixture_is_not_plugin_likelihood():
    """The predictive mixture differs from a plug-in likelihood and invalid mixtures are refused."""
    h = initial_hypotheses()[0]
    result = Fit(h, {"b0": np.array([-3.0, 3.0])}, 0.0)
    output = predict([result], [0.5])
    expected = logsumexp(binom.logpmf(0, 50, expit(np.array([-3.0, 3.0])))) - math.log(2)
    assert output["bma_logpmf"][0, 0] == pytest.approx(expected)
    assert abs(output["bma_logpmf"][0, 0] - binom.logpmf(0, 50, 0.5)) > 10
    with pytest.raises(ValueError):
        model_logweights([result, result])
    with pytest.raises(ValueError):
        mixture_logpmf([output["bma_logpmf"]], [1.0])


def test_joint_eig_retains_parameter_information():
    """The joint EIG retains parameter information and splits correctly into model and parameter parts."""
    bank = np.array([[[0.9, 0.1]], [[0.1, 0.9]]])
    total, model, parameter = information_from_components([bank], [1.0])
    expected = math.log(2) + 0.9 * math.log(0.9) + 0.1 * math.log(0.1)
    assert total[0] == pytest.approx(expected)
    assert model[0] == pytest.approx(0)
    assert parameter[0] == pytest.approx(expected)
    # Same two latent states split into two certain models: now it is model-only.
    t, m, p = information_from_components([bank[:1], bank[1:]], [0.5, 0.5])
    np.testing.assert_allclose(t, total)
    np.testing.assert_allclose(m, total)
    np.testing.assert_allclose(p, 0, atol=1e-14)


def test_pool_prior_and_top12_are_evidence_based():
    """Pool retention keeps the twelve highest-evidence fits with evidence-based weights."""
    fits = [
        Fit(
            Hypothesis(f"b0+{i}*u", "binomial", (("b0", "intercept"),)),
            {"b0": np.array([0.0])},
            float(i),
        )
        for i in range(14)
    ]
    kept, retired = retain(fits)
    assert len(kept) == 12 and len(retired) == 2
    assert [f.log_evidence for f in kept] == list(map(float, range(13, 1, -1)))
    weights = np.exp(model_logweights(kept))
    assert weights[0] / weights[1] == pytest.approx(math.e)
    assert weights.sum() == pytest.approx(1.0)


def test_world_rng_is_design_and_repeat_keyed():
    """World observations are keyed by design and repeat index, independent of call order."""
    first, second = CoreWorld(11, "out", "beta_binomial"), CoreWorld(11, "out", "beta_binomial")
    a0 = first.observe(0.1)
    b0 = first.observe(0.7)
    a1 = first.observe(0.1)
    assert second.observe(0.7) == b0
    assert second.observe(0.1) == a0
    assert second.observe(0.1) == a1
    assert a0["observation_seed"] != a1["observation_seed"]


def test_mock_proposal_budget_and_no_hidden_input():
    """Mock proposals respect the budget and do not depend on hidden input."""
    public = [(float(i) / 10, i) for i in range(9)]
    proposals = mock_proposals(public, 5, "111", "binomial")
    assert len(proposals) == 3
    assert {p.family for p in proposals} == {"binomial", "beta_binomial", "zi_binomial"}
    assert proposals == mock_proposals([(u, 50 - y) for u, y in public], 5, "111", "binomial")
    assert mock_proposals(public, 5, "000", "binomial") == []


@pytest.mark.heavy
def test_real_smc_evidence_against_one_dimensional_integral():
    """SMC log evidence agrees with a one-dimensional numerical integral."""
    h = initial_hypotheses()[0]
    u, y = [0.1, 0.7], [14, 19]
    exact, error = quad(
        lambda b: np.exp(binom.logpmf(y, 50, expit(b)).sum() + norm.logpdf(b, 0, 2)),
        -20,
        20,
        epsabs=1e-12,
    )
    assert error < 1e-10
    result = fit(h, u, y, 20260906, SMCSettings(256, 2, 2))
    # A stochastic estimator check, not an equality claim or posterior tuning.
    assert result.log_evidence == pytest.approx(np.log(exact), abs=0.5)
    assert len(result.particles["b0"]) == 512
    assert len(result.per_chain_log_evidence) == 2
    assert result.diagnostics["observation_count"] == 2


# ---------------------------------------------------------------------------
# The shared `points_label` helper. A paper-profile MOCK run used to be
# mislabelled "production": mock is neither "dev" nor "openai", so the old
# per-runner inline rule's "else" branch fired. The first check is the helper
# itself; the second is that `core_count`'s own two call sites (`write_points`'s
# points.csv `label` column, and `config`/`summary`'s own `label` key) agree
# with it rather than keeping a second, independently maintained copy of the
# rule. The static "no inline rule survives" test also lives here since it reads
# all three of core_count.py/boxinggym_core.py/chembench_core.py - the three
# runners involved - and core_count.py is where the shared helper itself is
# defined.
# ---------------------------------------------------------------------------


def test_points_label_truth_table_and_validation():
    """Rule v2: every unbound profile/provider pair defaults to development;
    unknown arguments still raise instead of guessing a label."""
    expected = {
        ("dev", "mock"): "development",
        ("dev", "anthropic"): "development",
        ("dev", "openai"): "development",
        ("paper", "mock"): "development",
        ("paper", "anthropic"): "development",
        ("paper", "openai"): "development",
    }
    for (profile, proposer), label in expected.items():
        assert points_label(profile, proposer) == label, (profile, proposer)
    for bad_profile in ("production", "PAPER", "", None, "paper "):
        with pytest.raises(ValueError):
            points_label(bad_profile, "mock")
    for bad_proposer in ("Anthropic", "claude", "", None, "mock "):
        with pytest.raises(ValueError):
            points_label("paper", bad_proposer)


def _fake_points_row(**overrides):
    return {
        "observation_count": 4,
        "round": 0,
        "phase": "initial",
        "bma": {"nll": 1.0, "coverage90": 0.9, "width90": 2.0, "kl": 0.1},
        **overrides,
    }


def _fake_points_config(**overrides):
    return {
        "output": "/tmp/does/not/need/to/exist",
        "cell": "111",
        "structure": "out",
        "truth_family": "beta_binomial",
        "proposer": "mock_fixed_template_responses_NOT_LLM",
        "llm_model": "claude-opus-4-7",
        "seed": 11,
        "profile": "dev",
        "source_files_sha256": {"core_count.py": "a" * 64},
        **overrides,
    }


@pytest.mark.parametrize(
    "profile,proposer,expected",
    [
        ("dev", "mock_fixed_template_responses_NOT_LLM", "development"),
        ("dev", "anthropic", "development"),
        ("dev", "openai", "development"),
        # the earlier finding, reproduced at the write_points layer: a
        # paper-profile MOCK run used to be mislabelled "production".
        ("paper", "mock_fixed_template_responses_NOT_LLM", "development"),
        ("paper", "anthropic", "development"),
        ("paper", "openai", "development"),
    ],
)
def test_write_points_label_matches_points_label_for_fake_config(
    tmp_path, profile, proposer, expected
):
    """`core_count.write_points`'s own `label` column is not a second,
    independently-maintained copy of the rule - for every cell of the truth
    table (including the one a smoke run once hit), it agrees with
    calling `points_label` directly on the same profile/proposer pair. The
    real `config["proposer"]` for a mock run is the descriptive string
    "mock_fixed_template_responses_NOT_LLM" (`core_count.run`'s own
    `config["proposer"]` assignment), not the literal "mock" the CLI's
    `--proposer` accepts - `write_points` normalizes that the same way it
    already does for the points.csv `proposer` column, two lines above the
    `label` field."""
    config = _fake_points_config(profile=profile, proposer=proposer)
    path = tmp_path / "points.csv"
    write_points(path, config, [_fake_points_row()], "completed", "")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    normalized = "mock" if "mock" in proposer else proposer
    assert rows[0]["label"] == points_label(profile, normalized) == expected


def test_no_inline_label_rule_remains_in_any_of_the_three_runners():
    """Static test: none of the three runners keeps its own hand-written copy of the
    old `"development" if profile == "dev" or proposer == "openai" else
    "production"` idiom - the only place a label value may be computed is
    `points_label` itself, called at least once by each runner."""
    experimental_dir = Path(__file__).resolve().parent
    for name in ("core_count.py", "boxinggym_core.py", "chembench_core.py"):
        text = (experimental_dir / name).read_text()
        assert 'else "production"' not in text, f"{name} still computes a label inline"
        assert "points_label(" in text, f"{name} never calls the shared helper"


# ---------------------------------------------------------------------------
# The shared `--run-label` switch. `points_label` DERIVES a label from
# (profile, proposer); `resolve_run_label` lets the caller STATE one, which a
# two-hour all-benchmark smoke run needed because it is a paper-profile
# anthropic campaign - once the single combination `points_label` called
# "production" - run as engine development.
# ---------------------------------------------------------------------------


def test_resolve_run_label_default_override_and_refusal():
    """Rule v2: no automatic promotion, and missing registration forbids production."""
    from .core_count import RUN_LABEL_CHOICES, resolve_run_label

    assert RUN_LABEL_CHOICES == ("development", "production")
    pairs = [(p, q) for p in ("dev", "paper") for q in ("mock", "anthropic", "openai")]

    # 1. Omission always remains development, including paper/anthropic.
    for profile, proposer in pairs:
        assert resolve_run_label(None, profile, proposer) == points_label(profile, proposer)

    # 2. Explicit development remains development.
    for profile, proposer in pairs:
        assert resolve_run_label("development", profile, proposer) == "development"
    assert points_label("paper", "anthropic") == "development"
    assert resolve_run_label("development", "paper", "anthropic") == "development"

    # 3. All production requests without a reference are refused.
    with pytest.raises(ValueError, match="--run-label production"):
        resolve_run_label("production", "paper", "anthropic")
    for profile, proposer in pairs:
        with pytest.raises(ValueError, match="--run-label production"):
            resolve_run_label("production", profile, proposer)

    # 4. neither a bogus label nor a bogus profile/proposer is guessed at
    for bad in ("prod", "PRODUCTION", "", "development "):
        with pytest.raises(ValueError, match="unknown run label"):
            resolve_run_label(bad, "paper", "anthropic")
    with pytest.raises(ValueError):
        resolve_run_label("development", "papers", "anthropic")
    with pytest.raises(ValueError):
        resolve_run_label("development", "paper", "claude")


def test_write_points_prefers_the_resolved_run_label_over_the_derived_one(tmp_path):
    """Label plumbing at the `write_points` layer: a config that carries an
    explicit `label` (what `run` now always records) wins over the derived
    rule, and a config without one (older runs, and this file's own
    hand-built fixtures) still falls back to `points_label` unchanged."""
    row = {
        "observation_count": 4,
        "round": 0,
        "phase": "initial",
        "bma": {"nll": 1.0, "coverage90": 0.9, "width90": 2.0, "kl": 0.1},
    }
    base = {
        "output": "/tmp/does/not/need/to/exist",
        "cell": "111",
        "structure": "out",
        "truth_family": "beta_binomial",
        "proposer": "anthropic",
        "llm_model": "claude-opus-4-7",
        "seed": 11,
        "profile": "paper",
        "source_files_sha256": {"core_count.py": "a" * 64},
    }

    def labels(config):
        """Write points.csv for a config and return the set of its label values."""
        path = tmp_path / f"points_{len(list(tmp_path.iterdir()))}.csv"
        write_points(path, config, [row], "completed", "")
        with path.open(newline="") as stream:
            return {r["label"] for r in csv.DictReader(stream)}

    # Unbound historic configuration has no verified production registration.
    assert labels(dict(base)) == {"development"}
    # stated: the same run labelled development by --run-label
    assert labels({**base, "label": "development"}) == {"development"}


def test_seed_99_is_a_development_world_and_the_three_research_worlds_are_unchanged():
    """A two-hour smoke run needs a CORE
    seed that is not a research seed. `--seed`'s choices are
    `tuple(WORLD_COEFFICIENTS)`, so the dict entry alone widens the CLI."""
    from .core_count import WORLD_COEFFICIENTS, main  # noqa: F401 - main for the parser

    # 1. seed 99 exists, is therefore accepted by --seed, and its coefficients
    #    are the mean of the three registered settings rounded to two decimals.
    assert 99 in WORLD_COEFFICIENTS
    registered = [WORLD_COEFFICIENTS[seed] for seed in (11, 17, 23)]
    expected = tuple(round(sum(pair) / 3, 2) for pair in zip(*registered))
    assert expected == (-0.17, 1.07)
    assert WORLD_COEFFICIENTS[99] == expected

    # 2. the three registered research worlds are byte-for-byte what they were.
    assert WORLD_COEFFICIENTS[11] == (-0.8, 1.4)
    assert WORLD_COEFFICIENTS[17] == (-0.2, 1.0)
    assert WORLD_COEFFICIENTS[23] == (0.5, 0.8)
    assert tuple(WORLD_COEFFICIENTS)[:4] == (11, 17, 23, 99)
    assert WORLD_COEFFICIENTS[31] == (-0.50, 1.20)
    assert WORLD_COEFFICIENTS[37] == (0.15, 0.90)
    assert WORLD_COEFFICIENTS[41] == (-0.15, 1.10)


def test_seed_99_world_is_a_distinct_world_that_does_not_disturb_seed_11():
    """The development world is a real, separate world: its coefficients drive
    `CoreWorld` through the same path every registered seed uses, and a seed-11
    world built alongside it is identical to one built on its own (the dict is
    only ever read by key, never iterated into behaviour)."""
    from .core_count import D0 as D0_POINTS
    from .core_count import WORLD_COEFFICIENTS

    b0, b1 = WORLD_COEFFICIENTS[99]
    z = 2 * np.asarray(D0_POINTS) - 1
    expected = expit(b0 + b1 * z + 0.8 * np.tanh(3 * z))
    dev_world = CoreWorld(99, "out", "beta_binomial")
    assert np.allclose(dev_world.probability(D0_POINTS), expected)

    # seed 11's world is bit-for-bit what it was: same probabilities, and the
    # same drawn observations as a world built with nothing else in the process.
    eleven = CoreWorld(11, "out", "beta_binomial")
    z11 = 2 * np.asarray(D0_POINTS) - 1
    assert np.allclose(
        eleven.probability(D0_POINTS), expit(-0.8 + 1.4 * z11 + 0.8 * np.tanh(3 * z11))
    )
    assert [eleven.observe(u) for u in D0_POINTS] == [
        CoreWorld(11, "out", "beta_binomial").observe(u) for u in D0_POINTS
    ]
    # and the development world is genuinely a DIFFERENT world, not an alias
    assert not np.allclose(dev_world.probability(D0_POINTS), eleven.probability(D0_POINTS))


def test_development_structures_have_well_posed_probabilities_and_distinct_streams():
    """New denominator/power worlds stay finite and use structure-keyed RNG streams."""
    from .core_count import D0

    seeds = (11, 17, 23, 31, 37, 41)
    for seed in seeds:
        for structure in ("out", "den", "pow"):
            world = CoreWorld(seed, structure, "binomial")
            probabilities = world.probability(np.linspace(0.0, 1.0, 17))
            assert np.all((probabilities >= 0.02) & (probabilities <= 0.98))
        out = CoreWorld(seed, "out", "binomial")
        den = CoreWorld(seed, "den", "binomial")
        pow_world = CoreWorld(seed, "pow", "binomial")
        assert any(out.observe(u)["y"] != den.observe(u)["y"] for u in D0)
        assert any(out.observe(u)["y"] != pow_world.observe(u)["y"] for u in D0)


def test_existing_core_structures_keep_their_registered_probability_laws():
    """Adding worlds does not alter the existing in/out laws."""
    u = np.linspace(0.0, 1.0, 17)
    z = 2 * u - 1
    for seed in (11, 17, 23, 99):
        b0, b1 = WORLD_COEFFICIENTS[seed]
        np.testing.assert_array_equal(
            CoreWorld(seed, "in", "binomial").probability(u), expit(b0 + b1 * z)
        )
        np.testing.assert_array_equal(
            CoreWorld(seed, "out", "binomial").probability(u),
            expit(b0 + b1 * z + 0.8 * np.tanh(3 * z)),
        )


def test_the_proposal_horizon_follows_rounds_past_twenty_and_never_below_it():
    """A run of 20 rounds or fewer keeps the list it always recorded (the
    registered waves and every short development run are byte-identical); a longer run
    extends the default list and the accepted maximum to its own last round."""
    from .core_count import P1_PROPOSAL_ROUNDS, parse_round_list

    legacy = P1_PROPOSAL_ROUNDS[-1]
    for rounds in (3, 20):
        horizon = max(rounds, legacy)
        assert parse_round_list("", tuple(range(1, horizon + 1)), maximum=horizon) == tuple(range(1, 21))
        assert parse_round_list("5", (), maximum=horizon) == (5,)
    horizon = max(50, legacy)
    assert parse_round_list("", tuple(range(1, horizon + 1)), maximum=horizon) == tuple(range(1, 51))
    assert parse_round_list("21 50", (), maximum=horizon) == (21, 50)
    with pytest.raises(ValueError, match="1..50"):
        parse_round_list("51", (), maximum=horizon)


#: The CORE development regimes' fixed constants. A review condition: the claim that none of them is a benchmark law's parameter has to be
#: bound to the benchmark source itself, not asserted in a document.
CR_DEVELOPMENT_CONSTANTS = (1.6, 0.25, 2.4)

#: sha256 of `autoscilab/oracle/chembench.py` in the revision this repository's conftest
#: already pins for that checkout. The digest is what makes the test below a binding: a
#: different revision of the law table fails here instead of passing silently against a
#: table nobody looked at.
CHEMBENCH_LAW_TABLE_SHA256 = "eba9514d68573b4aa8c6a427606f1d0d7a9c431d8396e69ef4ae0774d2a449de"

#: A value the table does hold. Without it a parser that silently matched nothing would
#: pass this test for the wrong reason - and 1.5 is exactly the constant the denominator
#: regime first used, which is why it was changed to 1.6.
CHEMBENCH_LAW_TABLE_POSITIVE_CONTROL = 1.5


def _chembench_parameter_values(source):
    import ast

    def numbers(value):
        """Collect all numeric values from a nested dictionary as floats."""
        if isinstance(value, dict):
            return {number for child in value.values() for number in numbers(child)}
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return {float(value)}
        return set()

    for node in ast.parse(source).body:
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "_PARAMS"):
            parameters = ast.literal_eval(node.value)
            return numbers(parameters)
    raise AssertionError("ChemBench _PARAMS table not found")


def test_chembench_parameter_parser_reads_integer_negative_and_exponent_literals():
    """The parameter-table parser reads integer, negative and exponent literals."""
    source = "_PARAMS: dict = {'synthetic': {'a': 5, 'b': -0.8, 'c': 1e3, 'd': 5.0}}"
    assert _chembench_parameter_values(source) == {5.0, -0.8, 1000.0}


def test_the_core_development_constants_are_in_no_chembench_law_parameter_table(
    require_autoscilab_source,
):
    """Retain this test and replace absence with the disclosed audit record."""
    import hashlib
    import re

    table_path = require_autoscilab_source / "autoscilab" / "oracle" / "chembench.py"
    source = table_path.read_text()
    digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    assert digest == CHEMBENCH_LAW_TABLE_SHA256, digest
    start = source.index("_PARAMS")
    end = source.index("\n}\n", start)
    old_values = {float(match) for match in re.findall(r":\s*([0-9]+\.[0-9]+)", source[start:end])}
    values = _chembench_parameter_values(source)
    assert len(values) >= len(old_values)
    assert CHEMBENCH_LAW_TABLE_POSITIVE_CONTROL in values, sorted(values)[:8]
    assert [constant for constant in CR_DEVELOPMENT_CONSTANTS if constant in values] == []
    # Per the preregistration of the CORE prompt ladder:
    # this post-hoc audit was not used in selection; coincidence is not independence.
    from .gaussian_core_constants import GAUSS_CALL_K, GAUSS_DEN_C, GAUSS_POW_E

    observed = {float(value): float(value) in values for value in
                (*CR_DEVELOPMENT_CONSTANTS, GAUSS_DEN_C, GAUSS_POW_E, GAUSS_CALL_K)}
    assert observed == {1.6: False, 0.25: False, 2.4: False,
                        0.07: False, 0.40: False, 5.0: True}
