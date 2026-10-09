"""Belief packet v2, prompt identity and prior probe contracts; no sequential Monte Carlo (SMC), no network."""

import inspect
import json
import re

import numpy as np
import pytest

from . import core_count, count_proposer
from .belief_packet import (
    ARCHIVE_REASON_KEYS,
    INTERFACE_KEYS,
    PoolMember,
    build_belief_packet,
    cell_constraints,
    core_interface,
    interface_grid,
    packet_digest,
    round_significant,
    serialize_packet,
)
from .count_inference import model_logweights
from .count_models import Hypothesis, initial_hypotheses


def member(expression="b0+b1*z", family="binomial", evidence=-30.0, count=32, **extra):
    """Build a pool member with random particles for the given expression and family."""
    hypothesis = Hypothesis(expression, family, core_count.mock_parameters(expression))
    rng = np.random.default_rng(abs(hash(expression + family)) % 2**32)
    particles = {name: rng.normal(0.5, 0.2, count) for name, _ in hypothesis.parameters}
    particles.update({name: rng.beta(2, 8, count) for name in hypothesis.noise_parameters})
    return PoolMember(hypothesis, particles, evidence, **extra)


def state(rounds=3):
    """Return synthetic public observations, pool members and archive rows for `rounds` rounds."""
    public = [
        {"obs_id": i, "round": min(i, rounds), "u": u, "y": y}
        for i, (u, y) in enumerate([(0.9, 44), (0.1, 3), (0.65, 33), (0.35, 12), (0.5, 25)])
    ]
    history = tuple(range(1, rounds + 1))
    fits = [
        member(
            "b0+b1*z", "binomial", -30.0,
            one_step_log_scores=(-3.1, -2.9, None), one_step_rounds=history,
        ),
        member(
            "b0", "binomial", -41.0,
            one_step_log_scores=(-4.0, -3.7, -3.5), one_step_rounds=history,
        ),
        member(
            "b0+b1*z+b2*square(z)", "beta_binomial", -33.5,
            born_round=2, one_step_log_scores=(None, None, -6.2), one_step_rounds=history,
        ),
    ]
    archive = [
        {
            "model_id": fits[0].hypothesis.model_id,
            "expression": "b0+b1*z",
            "family": "binomial",
            "proposed_round": 0,
            "outcome": "active",
            "reason": {"admitted_round": 0, "active_rounds": [0, 1, 2, 3]},
        },
        {
            "model_id": "0123456789abcdef0123",
            "expression": "b0+b1*log(1+u)",
            "family": "binomial",
            "proposed_round": 0,
            "outcome": "retired_by_evidence",
            "reason": {
                "log_evidence_at_retirement": -55.25,
                "gap_to_cutoff_nats": 13.75,
                "active_rounds": [0, 1],
                "computed_round": 2,
                "n_obs": 6,
            },
        },
        {
            "model_id": "",
            "expression": "",
            "family": "",
            "proposed_round": 2,
            "outcome": "rejected",
            "reason": {"validation_message": "call outside expression grammar"},
        },
        {
            "model_id": fits[1].hypothesis.model_id,
            "expression": "b0",
            "family": "binomial",
            "proposed_round": 3,
            "outcome": "duplicate",
            "reason": {"matched_model_id": fits[1].hypothesis.model_id, "matched_expression": "b0"},
        },
    ]
    return public, fits, archive


def packet(rounds=3):
    """Build the belief packet from the synthetic state."""
    public, fits, archive = state(rounds)
    return build_belief_packet(
        public, fits, archive, cell_constraints("111", "binomial"), core_interface(), rounds
    )


def test_builder_signature_has_no_evaluation_arguments():
    """The packet builder accepts no evaluation-side arguments and its body never mentions them."""
    names = list(inspect.signature(build_belief_packet).parameters)
    assert names == [
        "public_observations",
        "active_fits",
        "archive",
        "constraints",
        "interface",
        "round",
    ]
    forbidden = ("heldout", "held_out", "evaluation", "truth", "score", "nll", "pmf", "q")
    assert not [n for n in names for f in forbidden if f in n.lower()]
    body = inspect.getsource(build_belief_packet).replace(build_belief_packet.__doc__, "")
    assert "per_model_nll" not in body and "heldout" not in body and "evaluation" not in body


def test_packet_is_byte_identical_when_only_the_heldout_set_changes():
    """Held-out-set invariance: vary ONLY the evaluation set; the packet must not move."""
    from .core_count import CoreWorld

    world = CoreWorld(11, "out", "beta_binomial")
    first = world.evaluation(32, 20260906)
    second = world.evaluation(64, 20260907)
    assert first[0].shape != second[0].shape and not np.array_equal(first[1][:32], second[1][:32])
    public, fits, archive = state()
    constraints, interface = cell_constraints("111", "binomial"), core_interface()
    texts = []
    for _ in (first, second):
        texts.append(
            serialize_packet(build_belief_packet(public, fits, archive, constraints, interface, 3))
        )
    assert texts[0] == texts[1]
    assert packet_digest(texts[0]) == packet_digest(texts[1])


def test_no_evaluation_key_or_seed_reaches_the_serialized_packet():
    """No evaluation key or seed appears in the serialized packet."""
    text = serialize_packet(packet())
    for key in (
        "per_model_nll",
        "heldout",
        "evaluation",
        "truth",
        "expected_log_score",
        "coverage90",
        "width90",
        "mean_rmse",
        "observation_seed",
        "design_id",
    ):
        assert key not in text
    body = json.loads(text)
    public = {(row["u"], row["y"]) for row in state()[0]}
    assert {(row["u"], row["y"]) for row in body["observations"]} == public


def test_schema_shape_ordering_and_precision():
    """The packet has the expected keys, ordering and numeric precision."""
    body = packet()
    assert set(body) == {
        "packet_version",
        "interface",
        "round",
        "n_public",
        "observations",
        "pool",
        "best_model",
        "best_model_id",
        "best_model_residuals",
        "best_model_residual_definitions",
        "archive",
        "constraints",
    }
    assert tuple(sorted(body["interface"])) == tuple(sorted(INTERFACE_KEYS))
    assert body["interface"]["link"] == "logit" and body["interface"]["support"] == [0, 50]
    assert [row["u"] for row in body["observations"]] == sorted(
        row["u"] for row in body["observations"]
    )
    assert set(body["observations"][0]) == {"obs_id", "round", "u", "y"}
    assert [row["u"] for row in body["best_model_residuals"]] == [
        row["u"] for row in body["observations"]
    ]
    assert set(body["best_model_residuals"][0]) == {
        "obs_id",
        "u",
        "y",
        "pred_mean",
        "pred_sd",
        "standardized_residual",
    }
    weights = [item["weight"] for item in body["pool"]]
    assert weights == sorted(weights, reverse=True)
    assert np.isclose(sum(weights), 1.0)
    assert np.allclose(weights, np.exp(model_logweights(state()[1]))[[0, 2, 1]])
    entry = body["pool"][0]
    assert set(entry) == {
        "model_id",
        "expression",
        "family",
        "weight",
        "log_evidence",
        "log_evidence_computed_round",
        "log_evidence_n_obs",
        "born_round",
        "coefficients",
        "family_dispersion",
        "one_step_log_scores",
        "one_step_rounds",
    }
    assert set(entry["coefficients"]["b0"]) == {"mean", "sd"}
    assert body["best_model"] == "b0+b1*z | binomial"
    assert json.loads(serialize_packet(body))["pool"][0]["weight"] == float(
        f"{entry['weight']:.4g}"
    )


def test_family_dispersion_is_a_separate_field_from_coefficients():
    """Family dispersion parameters are reported apart from the coefficients."""
    body = packet()
    beta = next(item for item in body["pool"] if item["family"] == "beta_binomial")
    assert set(beta["family_dispersion"]) == {"rho"}
    assert "rho" not in beta["coefficients"]
    assert all(not item["family_dispersion"] for item in body["pool"] if item["family"] == "binomial")


def test_one_step_history_keeps_nulls_and_is_aligned_to_rounds():
    """One-step score histories keep nulls and align with their round lists."""
    body = packet()
    for item in body["pool"]:
        assert len(item["one_step_log_scores"]) in (0, len(item["one_step_rounds"]))
    scored = next(item for item in body["pool"] if item["born_round"] == 0)
    assert scored["one_step_rounds"] == [1, 2, 3]
    assert scored["one_step_log_scores"][-1] is None
    late = next(item for item in body["pool"] if item["born_round"] == 2)
    assert late["one_step_log_scores"] == [None, None, -6.2]


def test_archive_is_complete_and_reasons_are_structured():
    """The archive lists every outcome with its structured reason keys."""
    body = packet()
    assert len(body["archive"]) == 4
    outcomes = {item["outcome"] for item in body["archive"]}
    assert outcomes == {"active", "retired_by_evidence", "rejected", "duplicate"}
    for item in body["archive"]:
        assert set(item["reason"]) == ARCHIVE_REASON_KEYS[item["outcome"]]
    retired = next(i for i in body["archive"] if i["outcome"] == "retired_by_evidence")
    assert retired["reason"]["gap_to_cutoff_nats"] == 13.75
    assert retired["reason"]["computed_round"] == 2 and retired["reason"]["n_obs"] == 6
    rounds = [item["proposed_round"] for item in body["archive"]]
    assert rounds == sorted(rounds)


def test_archive_reason_key_set_is_enforced():
    """Archive reason keys and outcomes are validated by the builder."""
    public, fits, archive = state()
    broken = [{**archive[1], "reason": {"log_evidence_at_retirement": -1.0}}]
    with pytest.raises(ValueError, match="reason keys"):
        build_belief_packet(
            public, fits, broken, cell_constraints("111", "binomial"), core_interface(), 3
        )
    with pytest.raises(ValueError, match="unknown archive outcome"):
        build_belief_packet(
            public,
            fits,
            [{**archive[0], "outcome": "banished"}],
            cell_constraints("111", "binomial"),
            core_interface(),
            3,
        )


def test_constraints_and_interface_shape_are_fixed():
    """Constraints and interface validation behave as specified."""
    constraints = cell_constraints("010", "binomial")
    assert constraints["allowed_families"] == ["binomial", "beta_binomial", "zi_binomial"]
    assert not constraints["structure_expansion"] and constraints["family_expansion"]
    assert constraints["initial_structures"] == [h.expression for h in initial_hypotheses()]
    assert constraints["roles"]["intercept"] == "Normal(0,2)"
    assert constraints["roles"]["shape"] == "Gamma(2,1)"
    public, fits, archive = state()
    with pytest.raises(ValueError, match="interface"):
        build_belief_packet(public, fits, archive, constraints, {"outcome": "count"}, 3)
    with pytest.raises(ValueError, match="constraints keys"):
        build_belief_packet(public, fits, archive, {"roles": {}}, core_interface(), 3)
    with pytest.raises(ValueError, match="non-empty active pool"):
        build_belief_packet(public, [], archive, constraints, core_interface(), 3)


def test_rounding_is_four_significant_digits_and_rejects_nonfinite():
    """Rounding keeps four significant digits and rejects non-finite values."""
    assert round_significant(1.23456789) == 1.235
    assert round_significant(0.000123456789) == 0.0001235
    assert round_significant([1, True, None, "x", {"a": 2.5}]) == [1, True, None, "x", {"a": 2.5}]
    assert round_significant(np.float64(2.0 / 3)) == 0.6667
    with pytest.raises(ValueError, match="finite"):
        round_significant(float("nan"))


# --- prompt identity --------------------------------------------------


def test_prompt_file_is_task_free_and_its_sha_is_recorded():
    """The system prompt file is task-free and its sha256 is recorded in the prompt identity."""
    text = count_proposer.PROMPT_FILE.read_text()
    identity = count_proposer.prompt_identity()
    assert identity["prompt_version"] == "v2.0"
    assert identity["prompt_sha256"] == packet_digest(text)["packet_sha256"]
    lowered = text.lower()
    for banned in (
        "core",
        "count",
        "successes",
        "sigmoid",
        "binomial",
        "gravity",
        "death process",
        "boxinggym",
        "you are",
        "expert",
        "scientist",
        "trials",
    ):
        assert not re.search(rf"\b{re.escape(banned)}\b", lowered), banned
    assert "optionally a short note (<=200 chars) explaining the proposal" in text
    assert "primarily an unexplained residual pattern or a distribution mismatch" in text


# --- prior probe / prior predictive --------------------------------------------------


def test_prior_probe_rejects_only_non_finite_candidates():
    """The prior probe rejects non-finite candidates and passes a well-behaved one."""
    interface = core_interface()
    for expression in ("b0+b1*log(u)", "b0+b1*log(u-2)", "b0+b1*power(u,-1)", "b0+b1*sqrt(u-2)"):
        hypothesis = Hypothesis(expression, "binomial", core_count.mock_parameters(expression))
        probe, predictive = core_count.prior_probe(hypothesis, interface)
        assert not probe["passed"] and predictive is None
        assert "non-finite" in probe["detail"] or "invalid likelihood" in probe["detail"]
    hypothesis = Hypothesis("b0+b1*z", "binomial", (("b0", "intercept"), ("b1", "signed")))
    probe, predictive = core_count.prior_probe(hypothesis, interface)
    assert probe == {"n_samples": 64, "passed": True, "grid_points": 16}
    assert len(predictive["grid"]) == 16 and predictive["n_prior_samples"] == 64
    assert all(len(predictive[key]) == 16 for key in ("mean", "q05", "q50", "q95"))
    assert all(0 <= value <= 50 for value in predictive["mean"])
    assert all(a <= b <= c for a, b, c in zip(*(predictive[k] for k in ("q05", "q50", "q95"))))


def test_every_mock_template_passes_the_prior_probe():
    """Every mock template passes the prior probe for every family."""
    interface = core_interface()
    forms = set(core_count.P1_MOCK_EXPRESSIONS)
    forms.update(h.expression for h in initial_hypotheses())
    for base in ("b0+b1*z+b2*tanh(3*z)", "b0+b1*z+b2*square(u)", "b0+b1*tanh(3*z)"):
        forms.update({base, base.replace("3*z", "2*z"), base + "+b3*u"})
    for expression in sorted(forms):
        for family in ("binomial", "beta_binomial", "zi_binomial"):
            hypothesis = Hypothesis(expression, family, core_count.mock_parameters(expression))
            assert core_count.prior_probe(hypothesis, interface)[0]["passed"], expression


def test_prior_probe_stream_is_dedicated_and_reproducible():
    """The prior particle stream is dedicated, reproducible and leaves global RNG state untouched."""
    hypothesis = Hypothesis("b0+b1*z", "beta_binomial", (("b0", "intercept"), ("b1", "signed")))
    before = np.random.default_rng(7).normal(size=3)
    first = core_count.prior_particles(hypothesis)
    second = core_count.prior_particles(hypothesis)
    after = np.random.default_rng(7).normal(size=3)
    assert set(first) == {"b0", "b1", "rho"}
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert np.array_equal(before, after)
    other = Hypothesis("b0+b1*z", "binomial", (("b0", "intercept"), ("b1", "signed")))
    assert not np.array_equal(first["b0"], core_count.prior_particles(other)["b0"])
    assert len(interface_grid(core_interface(), 16)) == 16


# --- Mock schedule and catalog of proposal policy P1 ----------------------------------------------------


def test_mock_schedule_is_one_candidate_per_round_and_respects_the_cell():
    """The mock schedule proposes one candidate per round and respects the cell's flags."""
    initial = {h.expression for h in initial_hypotheses()}
    for cell in (f"{i:03b}" for i in range(8)):
        for t in range(1, 21):
            public = [{"u": 0.5, "y": 25}] * (4 + t)
            proposals = core_count.mock_proposal_p1(public, t, cell, "binomial")
            assert len(proposals) == (0 if cell[:2] == "00" else 1)
            core_count.validate_cell_candidates(proposals, cell, "binomial")
            for hypothesis in proposals:
                if cell[0] == "0":
                    assert hypothesis.expression in initial
                    assert hypothesis.family != "binomial"
                if cell[1] == "0":
                    assert hypothesis.family == "binomial"
    with pytest.raises(ValueError, match="observation count mismatch"):
        core_count.mock_proposal_p1([{"u": 0.5, "y": 25}] * 4, 1, "111", "binomial")


def test_catalog_is_finite_only_when_structure_expansion_is_off():
    """The candidate catalog is finite only when structure expansion is off."""
    assert core_count.candidate_catalog("111", "binomial") is None
    assert core_count.candidate_catalog("100", "binomial") is None
    assert len(core_count.candidate_catalog("010", "binomial")) == 12
    assert len(core_count.candidate_catalog("000", "binomial")) == 4
    catalog = core_count.candidate_catalog("011", "binomial")
    assert set(catalog.values()) == {
        (item.expression, family)
        for item in initial_hypotheses()
        for family in ("binomial", "beta_binomial", "zi_binomial")
    }
    initial_ids = {h.model_id for h in initial_hypotheses()}
    assert initial_ids < set(catalog)
    assert len(set(catalog) - initial_ids) == 8


def test_alpha_scaled_expressions_are_different_models():
    """A rescaled coefficient is a different prior, never a duplicate."""
    roles = (("b0", "intercept"), ("b1", "signed"))
    assert (
        Hypothesis("b0+b1*z", "binomial", roles).model_id
        != Hypothesis("b0+2*b1*z", "binomial", roles).model_id
    )
    assert (
        Hypothesis("b0+b1*z", "binomial", roles).model_id
        == Hypothesis("c0+c1*z", "binomial", (("c0", "intercept"), ("c1", "signed"))).model_id
    )


def test_round_lists_and_variant_labels():
    """Round-list parsing and variant labels behave as specified."""
    assert core_count.parse_round_list("", (1, 2)) == (1, 2)
    assert core_count.parse_round_list("10,20", ()) == (10, 20)
    assert core_count.parse_round_list("5 3 3", ()) == (3, 5)
    with pytest.raises(ValueError, match="1..20"):
        core_count.parse_round_list("21", ())
    assert core_count.parse_round_list("21,50", (), maximum=50) == (21, 50)
    with pytest.raises(ValueError, match="1..50"):
        core_count.parse_round_list("51", (), maximum=50)
    proposals = core_count.mock_proposal_p1(
        [{"u": 0.5, "y": 25}] * 54, 50, "111", "binomial", max_rounds=50
    )
    assert len(proposals) == 1
    assert core_count.archive_reevaluation_label(()) == "off"
    assert core_count.archive_reevaluation_label((10, 20)) == "t10_20"
    assert core_count.points_variant("off") == ""
    assert core_count.points_variant("t10_20") == "rev10_20"
    assert core_count.points_variant("t3_5") == "rev3_5"
