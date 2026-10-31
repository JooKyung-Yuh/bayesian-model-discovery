"""Synthetic acquisition-only validity checks; no fitting or provider calls."""
import json
from dataclasses import replace

import numpy as np
import pytest

from bayes_model_discovery.experimental import count_inference as inference
from bayes_model_discovery.experimental import positive_inference as positive
from bayes_model_discovery.experimental.belief_packet import task_interface
from bayes_model_discovery.experimental.count_models import Hypothesis
from bayes_model_discovery.experimental.real_inference import RealHypothesis

POSITIVE_FAMILIES = ("lognormal", "gamma", "poisson", "negative_binomial", "zi_poisson")


def grammar_fit(family="lognormal", *, intercepts=(0., .7), slopes=(.3, -.2),
                evidence=0., expression="b0+b1*u"):
    """Helper: grammar fit."""
    real = family in ("normal", "student_t")
    hypothesis_type = RealHypothesis if real else positive.ScalarHypothesis
    hypothesis = hypothesis_type(expression, family, (("b0", "intercept"), ("b1", "signed")))
    count = len(intercepts)
    particles = {"b0": np.asarray(intercepts, dtype=float), "b1": np.asarray(slopes, dtype=float)}
    noise = {"lognormal": ("sigma", .7), "gamma": ("shape", 2.),
             "negative_binomial": ("alpha", 2.), "zi_poisson": ("pi", .25),
             "normal": ("sigma", .7), "student_t": ("sigma", .7)}
    if family in noise:
        key, value = noise[family]
        particles[key] = np.full(count, value)
    kind = "real" if real else "unbounded_count" if hypothesis.measure == "PMF" else "positive"
    interface = task_interface(kind, [{"name": "u", "range": [0., 2.]}], {},
                               (None, None) if real else (0, None))
    return inference.Fit(hypothesis, particles, evidence, diagnostics={
        "interface": interface, "response_unit": 1., "response_center": 0.,
        "input_transform": None, "observation_count": 4,
        "public_data_sha256": "synthetic-positivity-guard"})


def overflow_fit(family="lognormal", evidence=0.):
    """Helper: overflow fit."""
    return grammar_fit(family, intercepts=(0., 0., 0., .7),
                       slopes=(.3, 400., -400., -.2), evidence=evidence)


def bounded_fit():
    """Helper: bounded fit."""
    hypothesis = Hypothesis("b0+b1*u", "binomial", (("b0", "intercept"), ("b1", "signed")))
    return inference.Fit(hypothesis, {"b0": np.array([-.5, .7]), "b1": np.array([.2, -.1])}, 0.)


def record_for(fits, candidates=(0., .25), targets=(2.,), seed=71, **kwargs):
    """Helper: record for."""
    chosen, rows = inference.choose_epig(fits, candidates, seed, target_points=targets, **kwargs)
    return inference.epig_acquisition_record(chosen, rows, seed)


def assert_finite_scores(record):
    """Helper: assert finite scores."""
    assert np.all(np.isfinite([[row["epig"], row["joint_eig"]] for row in record["rows"]]))
    assert all(-1e-12 <= row["epig"] <= row["joint_eig"] + 1e-12 for row in record["rows"])


@pytest.mark.parametrize("family", POSITIVE_FAMILIES)
def test_mask_identifies_overflow_and_underflow_without_changing_valid_mean_bytes(family):
    """Check that mask identifies overflow and underflow without changing valid mean bytes."""
    fit = overflow_fit(family)
    inputs = np.array([0., .25, 2.])
    mean, valid = positive.particle_means_masked(fit, inputs)
    np.testing.assert_array_equal(valid, [[True, True, True], [True, True, False],
                                         [True, True, False], [True, True, True]])
    assert np.isposinf(mean[1, 2]) and mean[2, 2] == 0.
    with pytest.raises(ValueError, match="non-finite/nonpositive mean; no clipping fallback"):
        positive.particle_means(fit, inputs)
    good = replace(fit, particles={key: values[[0, 3]] for key, values in fit.particles.items()})
    expected = positive.particle_means(good, inputs)
    assert mean[[0, 3]].tobytes() == expected.tobytes()


@pytest.mark.parametrize("family", ("normal", "student_t"))
def test_real_negative_and_zero_means_are_valid_and_unchanged(family):
    """Check that real negative and zero means are valid and unchanged."""
    fit = grammar_fit(family, intercepts=(-2., 0., 2.), slopes=(0., 0., 0.))
    inputs = np.array([0., .25, 2.])
    mean, valid = positive.particle_means_masked(fit, inputs)
    assert np.all(valid) and np.any(mean < 0) and np.any(mean == 0)
    assert mean.tobytes() == positive.particle_means(fit, inputs).tobytes()
    fits, weights = [fit], np.full(3, 1 / 3)
    retained, adjusted, info = inference._epig_guard_particles(fits, inputs, weights)
    assert retained is fits and adjusted is weights and info == {}
    record = record_for(fits)
    assert "epig_excluded_particles" not in record and "epig_excluded_models" not in record
    assert_finite_scores(record)


@pytest.mark.parametrize("bad_at", ("candidate", "target"))
def test_union_mask_is_global_and_keeps_parameter_alignment_and_model_mass(bad_at):
    """Check that union mask is global and keeps parameter alignment and model mass."""
    partial = overflow_fit(evidence=np.log(.8))
    other = grammar_fit("gamma", evidence=np.log(.2))
    originals = [{key: values.copy() for key, values in fit.particles.items()}
                 for fit in (partial, other)]
    candidates, targets = ([0., 2.], [.25]) if bad_at == "candidate" else ([0., .25], [2.])
    inputs = np.r_[candidates, targets]
    fits, weights = [partial, other], np.array([.2] * 4 + [.1] * 2)
    retained, adjusted, info = inference._epig_guard_particles(fits, inputs, weights)
    assert retained[0].hypothesis is partial.hypothesis and retained[1] is other
    for key, values in partial.particles.items():
        np.testing.assert_array_equal(retained[0].particles[key], values[[0, 3]])
    np.testing.assert_allclose(adjusted, [.4, .4, .1, .1], rtol=1e-15, atol=0)
    assert adjusted[:2].sum() == pytest.approx(.8)
    assert adjusted[2:].sum() == pytest.approx(.2)
    for fit, snapshot in zip(fits, originals, strict=True):
        for key in snapshot:
            np.testing.assert_array_equal(fit.particles[key], snapshot[key])
    assert info == {"epig_excluded_particles": {partial.hypothesis.model_id: {
        "excluded": 2, "of": 4, "reason": "nonfinite_or_nonpositive_mean"}}}
    record = record_for(fits, candidates, targets)
    assert record["epig_excluded_particles"] == info["epig_excluded_particles"]
    assert "epig_excluded_models" not in record
    assert_finite_scores(record)
    # The same conditioned posterior defines every candidate/target pair.
    reference = record_for(retained, candidates, targets)
    assert record["chosen"] == reference["chosen"]
    np.testing.assert_allclose([[r["epig"], r["joint_eig"]] for r in record["rows"]],
                               [[r["epig"], r["joint_eig"]] for r in reference["rows"]], atol=1e-13)


def test_choose_guards_one_full_union_before_constructing_any_bank(monkeypatch):
    """Check that choose guards one full union before constructing any bank."""
    original_guard = inference._epig_guard_particles
    original_bank = inference._epig_continuous_bank
    events = []

    def guard(fits, inputs, weights):
        """Helper: guard."""
        events.append(("guard", inputs.copy()))
        return original_guard(fits, inputs, weights)

    def bank(fits, inputs, weights, *args, **kwargs):
        """Helper: bank."""
        events.append(("bank", inputs.copy()))
        assert all(len(fit.particles["b0"]) == 2 for fit in fits)
        return original_bank(fits, inputs, weights, *args, **kwargs)

    monkeypatch.setattr(inference, "_epig_guard_particles", guard)
    monkeypatch.setattr(inference, "_epig_continuous_bank", bank)
    record_for([overflow_fit()], [0., .25], [2., 2.])
    assert [event[0] for event in events] == ["guard", "bank"]
    for _, inputs in events:
        np.testing.assert_array_equal(inputs, [0., .25, 2., 2.])


def test_whole_model_removal_renormalizes_even_originally_underflowed_survivors():
    """Check that whole model removal renormalizes even originally underflowed survivors."""
    bad = grammar_fit(intercepts=(1000., 1000.), slopes=(0., 0.), evidence=0.)
    low = grammar_fit("gamma", evidence=-2000.)
    lower = grammar_fit("lognormal", evidence=-2001., expression="b0+b1*square(u)")
    subsets, weights = inference._epig_subsample([bad, low, lower], 71, 128)
    assert np.all(weights[2:] == 0)
    retained, adjusted, info = inference._epig_guard_particles(subsets, np.array([0., 2.]), weights)
    assert [r.hypothesis.model_id for r in retained] == [low.hypothesis.model_id, lower.hypothesis.model_id]
    expected = np.exp(np.array([0., -1.])) / (1 + np.exp(-1.))
    np.testing.assert_allclose([adjusted[:2].sum(), adjusted[2:].sum()], expected, rtol=1e-12)
    assert adjusted.sum() == pytest.approx(1.) and np.all(adjusted > 0)
    assert info["epig_excluded_models"] == [bad.hypothesis.model_id]
    assert_finite_scores(record_for([bad, low, lower]))


@pytest.mark.parametrize("families", (("lognormal", "gamma"), ("poisson", "negative_binomial")))
def test_all_models_invalid_refuses_before_allocation_selection_or_coarsening(monkeypatch, families):
    """Check that all models invalid refuses before allocation selection or coarsening."""
    fits = [grammar_fit(family, intercepts=(bad, bad), slopes=(0., 0.))
            for family, bad in zip(families, (1000., -1000.), strict=True)]

    def forbidden(*args, **kwargs):
        """Helper: forbidden."""
        raise AssertionError("positivity refusal must precede banks, coarsening and selection")

    for name in ("_epig_continuous_bank", "_epig_discrete_banks", "_epig_discrete_coarsened_bank", "choose_design"):
        monkeypatch.setattr(inference, name, forbidden)
    with pytest.raises(inference.EpigResourceLimit) as caught:
        record_for(fits)
    detail = caught.value.detail
    assert detail["resource"] == "positivity" and detail["epig_all_candidates_missing"] is True
    assert detail["epig_excluded_models"] == [fit.hypothesis.model_id for fit in fits]
    assert detail["epig_excluded_particles"] == {
        fit.hypothesis.model_id: {"excluded": 2, "of": 2, "reason": "nonfinite_or_nonpositive_mean"}
        for fit in fits}
    assert json.loads(str(caught.value).removeprefix("EPIG resource refusal: ")) == detail


@pytest.mark.parametrize("family", ("poisson", "negative_binomial", "zi_poisson"))
def test_exact_discrete_outcomes_keep_exclusion_metadata(family):
    """Check that exact discrete outcomes keep exclusion metadata."""
    record = record_for([overflow_fit(family)])
    assert record["outcome_info"]["epig_discrete_coarsened"] is False
    assert next(iter(record["epig_excluded_particles"].values()))["excluded"] == 2
    assert_finite_scores(record)


def test_coarsening_follows_guard_and_preserves_its_own_resource_receipt():
    """Check that coarsening follows guard and preserves its own resource receipt."""
    fit = grammar_fit("poisson", intercepts=tuple(np.log([8000., 10000., 1.])),
                      slopes=(0., .02, 1000.))
    record = record_for([fit], [0., .3], [.6, .9])
    assert record["outcome_info"]["epig_discrete_coarsened"] is True
    assert record["outcome_info"]["epig_discrete_enumeration_refusal"]["resource"] == "acquisition_arrays"
    assert record["epig_excluded_particles"] == {fit.hypothesis.model_id: {
        "excluded": 1, "of": 3, "reason": "nonfinite_or_nonpositive_mean"}}
    assert_finite_scores(record)


def test_bounded_finite_support_never_enters_scalar_guard(monkeypatch):
    """Check that bounded finite support never enters scalar guard."""
    def forbidden(*args, **kwargs):
        """Helper: forbidden."""
        raise AssertionError("bounded laws must keep their existing path")

    monkeypatch.setattr(inference, "_epig_guard_particles", forbidden)
    record = record_for([bounded_fit()], n_trials=2)
    assert "epig_excluded_particles" not in record and "epig_excluded_models" not in record
    assert_finite_scores(record)


def test_exclusion_counts_describe_only_the_actual_posterior_subsample():
    """Check that exclusion counts describe only the actual posterior subsample."""
    fit = grammar_fit(intercepts=tuple([0.] * 10 + [-1000.] * 10), slopes=tuple([0.] * 20))
    subsets, _ = inference._epig_subsample([fit], 71, 6)
    bad_count = int(np.count_nonzero(subsets[0].particles["b0"] < -900))
    assert 0 < bad_count < 6
    record = record_for([fit], posterior_samples=6)
    assert record["epig_excluded_particles"][fit.hypothesis.model_id] == {
        "excluded": bad_count, "of": 6, "reason": "nonfinite_or_nonpositive_mean"}
    assert len(fit.particles["b0"]) == 20


def test_wholly_invalid_model_set_matches_point_prediction_guard():
    """Check that wholly invalid model set matches point prediction guard."""
    from bayes_model_discovery.experimental.chembench_core import _prediction_pools

    def chem_fit(family, intercepts):
        """Helper: chem fit."""
        noise = "sigma" if family == "lognormal" else "shape"
        return inference.Fit(positive.RegressionHypothesis(family),
            {"b0": np.asarray(intercepts), noise: np.full(2, .7 if noise == "sigma" else 2.)},
            0., diagnostics={"response_unit": 1., "input_transform": {"center": (0.,), "scale": (1.,)},
                            "observation_count": 4, "public_data_sha256": "synthetic-chem-guard"})

    bad = chem_fit("lognormal", (1000., 1000.))
    good = chem_fit("gamma", (0., .7))
    inputs = np.array([[0.], [.25], [2.]])
    pools, exclusions = _prediction_pools([bad, good], inputs, 3, "synthetic")
    record = record_for([bad, good], inputs[:2], inputs[2:])
    assert set(record["epig_excluded_models"]) == {row["model_id"] for row in exclusions}
    assert all(pool == [good] for pool in pools)
    assert_finite_scores(record)


def test_structural_errors_are_not_reclassified_as_mean_exclusion():
    """Check that structural errors are not reclassified as mean exclusion."""
    fit = grammar_fit()
    malformed = replace(fit, particles={key: values for key, values in fit.particles.items() if key != "b1"})
    with pytest.raises(KeyError, match="b1"):
        record_for([malformed])


def test_chunked_guard_still_uses_one_global_mask_for_late_targets(monkeypatch):
    """Check that chunked guard still uses one global mask for late targets."""
    fit = overflow_fit()
    original = positive.particle_means_masked
    seen = []

    def observed(result, inputs):
        """Helper: observed."""
        seen.append(np.asarray(inputs).copy())
        assert len(inputs) <= 16
        return original(result, inputs)

    monkeypatch.setattr(positive, "particle_means_masked", observed)
    inputs = np.r_[np.linspace(0., .25, 35), 2.]
    retained, weights, info = inference._epig_guard_particles([fit], inputs, np.full(4, .25))
    assert len(seen) >= 3
    np.testing.assert_array_equal(np.concatenate(seen), inputs)
    np.testing.assert_array_equal(retained[0].particles["b1"], [.3, -.2])
    np.testing.assert_array_equal(weights, [.5, .5])
    assert info["epig_excluded_particles"][fit.hypothesis.model_id]["excluded"] == 2


@pytest.mark.parametrize("family", POSITIVE_FAMILIES)
def test_all_valid_positive_guard_preserves_object_identity(family):
    """Check that all valid positive guard preserves object identity."""
    fits = [grammar_fit(family)]
    weights = np.full(2, .5)
    retained, adjusted, info = inference._epig_guard_particles(fits, np.linspace(0., 2., 40), weights)
    assert retained is fits and adjusted is weights and info == {}


def test_nonfinite_real_mean_excluded_without_dropping_negative_or_zero_values():
    """Check that nonfinite real mean excluded without dropping negative or zero values."""
    fit = grammar_fit("normal", intercepts=(-2., 0., 1e308), slopes=(0., 0., 1e308))
    retained, weights, info = inference._epig_guard_particles([fit], np.array([0., 2.]), np.full(3, 1 / 3))
    np.testing.assert_array_equal(retained[0].particles["b0"], [-2., 0.])
    np.testing.assert_array_equal(weights, [.5, .5])
    assert info["epig_excluded_particles"][fit.hypothesis.model_id]["excluded"] == 1


def test_private_distribution_keeps_strict_failure_for_unfiltered_particles():
    """Check that private distribution keeps strict failure for unfiltered particles."""
    with pytest.raises(ValueError, match="non-finite/nonpositive mean; no clipping fallback"):
        inference._epig_scalar_distribution(overflow_fit(), np.array([0., 2.]))


def synthetic_emotion_record(output, patch, excluded_counts=()):
    """Execute the real multi-component runner with a synthetic world and fit tables."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    from bayes_model_discovery.experimental import boxinggym_core as bg
    from bayes_model_discovery.experimental.test_named_inputs import (
        bg_run_args,
        bg_scalar_stub_fits,
    )

    spec = replace(bg.TASKS["emotion"], module=bg.__name__, openai_narration=False, truth_mean=None)
    rows = np.array([[lo + fraction * (hi - lo) for _, (lo, hi) in
                      zip(spec.input_names, [spec.input_range, *[item["range"] for item in spec.extra_inputs]], strict=True)]
                     for fraction in (.2, .4, .6, .8)])
    components = spec.components

    class SyntheticWorld:
        """Synthetic SyntheticWorld fixture used by the surrounding test."""
        def __init__(self, *_):
            """Helper: init  ."""
            self.goal = SimpleNamespace(eval_points=[])
            self.observation_calls = 0

        def evaluation_points(self, count, seed):
            """Helper: evaluation points."""
            points = np.tile(rows, (int(np.ceil(count / len(rows))), 1))[:count]
            self.goal.eval_points = list(points)
            return points, np.tile(np.arange(1., len(components) + 1), (count, 1))

        def eval_inputs(self, point):
            """Helper: eval inputs."""
            return point

        def refused_designs(self):
            """Helper: refused designs."""
            return set()

        def world_parameters(self):
            """Helper: world parameters."""
            return {"synthetic": True}

        def observe(self, handle):
            """Helper: observe."""
            self.observation_calls += 1
            return np.arange(1., len(components) + 1).tolist()

        def evaluate(self, submitted):
            """Helper: evaluate."""
            assert len(submitted) == len(self.goal.eval_points)
            return {"metric": 0., "std": 0.}

    captured = {}

    def fake_epig(fits, designs, seed, *args, **kwargs):
        """Helper: fake epig."""
        name = fits[0].diagnostics["component"]
        model_id = fits[0].hypothesis.model_id
        table = [{"design": np.asarray(design).tolist(), "epig": float(i + 1), "joint_eig": float(i + 2)}
                 for i, design in enumerate(designs)]
        component_index = components.index(name)
        info = None
        if component_index < len(excluded_counts) and excluded_counts[component_index]:
            info = {"epig_excluded_particles": {model_id: {
                "excluded": excluded_counts[component_index], "of": 8,
                "reason": "nonfinite_or_nonpositive_mean"}}}
        captured[name] = {"model_id": model_id, "info": info}
        return table[-1]["design"], inference.EpigRows(table, None, info)

    bg_scalar_stub_fits(patch)
    patch.setitem(bg.TASKS, "emotion", spec)
    patch.setattr(bg, "TaskEnvironment", SyntheticWorld)
    patch.setattr(bg, "env_seed_injection", lambda *_: nullcontext([]))
    patch.setattr(bg, "build_design_pool", lambda *_: bg.DesignPool(rows, tuple(range(len(rows))), False))
    patch.setattr(bg, "sample_epig_targets", lambda *_, **__: np.tile(rows, (8, 1)))
    patch.setattr(bg, "prior_probe", lambda *args, **kwargs: ({}, None))
    patch.setattr(bg, "choose_epig", fake_epig)
    bg.run(bg_run_args(output, task="emotion", cell="002", rounds=1))
    summary = json.loads((output / "summary.json").read_text())
    assert summary["paid_api_calls"] == 0
    return (output / "acquisition/t01.json").read_bytes(), captured


def test_emotion_runner_preserves_component_names_when_model_ids_repeat(tmp_path, monkeypatch):
    """Check that emotion runner preserves component names when model ids repeat."""
    data, captured = synthetic_emotion_record(tmp_path / "run", monkeypatch, (1, 3))
    record = json.loads(data)
    names = list(captured)
    assert captured[names[0]]["model_id"] == captured[names[1]]["model_id"]
    assert record["epig_component_exclusions"] == {
        name: captured[name]["info"] for name in names[:2]}
    assert [next(iter(record["epig_component_exclusions"][name]["epig_excluded_particles"].values()))["excluded"]
            for name in names[:2]] == [1, 3]
    assert "epig_excluded_particles" not in record
    assert "epig_excluded_models" not in record
    assert record["multi_component_epig_rule"] == "sum_of_independent_component_predictive_information"
    assert all(row["epig"] == sum(row["per_component_epig"]) for row in record["rows"])


def test_emotion_all_valid_record_bytes_match_existing_plain_table_path(tmp_path):
    """Check that emotion all valid record bytes match existing plain table path."""
    from bayes_model_discovery.experimental import boxinggym_core as bg

    with pytest.MonkeyPatch.context() as patch:
        wrapped, _ = synthetic_emotion_record(tmp_path / "wrapped", patch)
    original_rows = inference.EpigRows
    with pytest.MonkeyPatch.context() as patch:
        # Prior runner inputs were plain candidate lists; removing only the
        # optional carrier leaves its existing arithmetic and record schema.
        patch.setattr(inference, "EpigRows", lambda rows, outcome_info, exclusion_info=None: list(rows))
        plain, _ = synthetic_emotion_record(tmp_path / "plain", patch)
    assert inference.EpigRows is original_rows
    assert wrapped == plain
    record = json.loads(wrapped)
    assert "epig_component_exclusions" not in record
    assert "epig_excluded_particles" not in record and "epig_excluded_models" not in record
    assert len(record["component_seeds"]) == len(bg.TASKS["emotion"].components)


def test_partial_particle_guard_and_point_prediction_guard_identify_same_affected_model():
    """Check that partial particle guard and point prediction guard identify same affected model."""
    from bayes_model_discovery.experimental.chembench_core import _prediction_pools

    diagnostics = {"response_unit": 1., "input_transform": {"center": (0.,), "scale": (1.,)},
                   "observation_count": 4, "public_data_sha256": "synthetic-partial-chem-guard"}
    partial = inference.Fit(positive.RegressionHypothesis("lognormal", (0,)),
                            {"b0": np.zeros(3), "b1": np.array([.2, 400., -.1]), "sigma": np.full(3, .7)},
                            0., diagnostics=diagnostics)
    good = inference.Fit(positive.RegressionHypothesis("gamma"),
                         {"b0": np.array([0., .7]), "shape": np.full(2, 2.)}, 0., diagnostics=diagnostics)
    inputs = np.array([[0.], [.25], [2.]])
    pools, exclusions = _prediction_pools([partial, good], inputs, 1, "synthetic")
    record = record_for([partial, good], inputs[:2], inputs[2:])
    assert set(record["epig_excluded_particles"]) == {item["model_id"] for item in exclusions}
    assert record["epig_excluded_particles"][partial.hypothesis.model_id]["excluded"] == 1
    # Point prediction removes this model at the bad input. EPIG intentionally
    # retains its two valid particles, using one conditioned pool for all pairs.
    assert [len(pool) for pool in pools] == [2, 2, 1]
    assert "epig_excluded_models" not in record
    assert_finite_scores(record)


def test_emotion_runner_records_only_the_single_affected_component(tmp_path, monkeypatch):
    """Check that emotion runner records only the single affected component."""
    data, captured = synthetic_emotion_record(tmp_path / "run", monkeypatch, (1,))
    record = json.loads(data)
    assert len(captured) > 1
    affected = next(iter(captured))
    assert record["epig_component_exclusions"] == {affected: captured[affected]["info"]}
    assert all(value["info"] is None for name, value in captured.items() if name != affected)
    assert "epig_excluded_particles" not in record and "epig_excluded_models" not in record
