"""Synthetic boundary witnesses; no fitting or provider calls."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from . import count_inference as inference
from .count_epig import predictive_information
from .test_epig_positivity_guard import grammar_fit
from .test_epig_unbounded import count_fit

REPAIR_KEY = "epig_discrete_boundary_repairs"


def _distributions(fits, inputs, *, discrete):
    distributions = [inference._epig_scalar_distribution(fit, inputs) for fit in fits]
    if discrete:
        distributions = [
            inference._EpigZeroInflatedQuantileDistribution(distribution, fit.particles["pi"])
            if fit.hypothesis.family == "zi_poisson" else distribution
            for fit, distribution in zip(fits, distributions, strict=True)
        ]
    return distributions


def _cdf_masses(distributions, boundaries):
    banks = []
    for distribution in distributions:
        cdf = distribution.cdf(boundaries[None, :, :])
        banks.append(np.diff(np.concatenate([np.zeros_like(cdf[:, :, :1]), cdf,
                                            np.ones_like(cdf[:, :, :1])], axis=-1), axis=-1))
    return np.concatenate(banks)


def _legacy_quantile_bank(fits, inputs, weights, grid_size=64, *, discrete=True):
    """Freeze the earlier independent quantile search, before the repair.

    Deliberately retain negative masses: this reference is a failure witness,
    never an acquisition bank or a replacement for the production validator.
    """
    distributions = _distributions(fits, inputs, discrete=discrete)
    probabilities = np.arange(1, grid_size, dtype=float) / grid_size
    quantiles = np.concatenate([distribution.ppf(probabilities) for distribution in distributions])
    lower, upper = np.min(quantiles, axis=0), np.max(quantiles, axis=0)
    for _ in range(56):
        middle = lower + (upper - lower) / 2
        cdf = np.concatenate([distribution.cdf(middle[None, :, :]) for distribution in distributions])
        below = np.einsum("k,kxy->xy", weights, cdf) < probabilities
        lower, upper = np.where(below, middle, lower), np.where(below, upper, middle)
    boundaries = lower + (upper - lower) / 2
    return _cdf_masses(distributions, boundaries), boundaries


def _witness_fit(family):
    means = (.56, 40., 8000., 10000.) if family == "negative_binomial" else (.56, 40., 130., 150.)
    fit = count_fit(family, means=means, slopes=(0.,) * 4)
    if family != "poisson":
        parameter = "alpha" if family == "negative_binomial" else "pi"
        fit = replace(fit, particles={**fit.particles, parameter: np.full(4, .2)})
    return fit


def _refusal(input_count):
    return {"resource": "acquisition_arrays", "cutoffs": [5000] * input_count,
            "estimated_working_bytes": 2**40, "max_working_bytes": 256 * 1024**2}


def _record(fits, inputs, seed=71):
    chosen, rows = inference.choose_epig(fits, inputs[:2], seed, target_points=inputs[2:])
    return inference.epig_acquisition_record(chosen, rows, seed)


@pytest.mark.parametrize("family", ("poisson", "negative_binomial", "zi_poisson"))
def test_discrete_jump_witness_repairs_boundaries_without_changing_cdf_mass(family, tmp_path):
    """Check: discrete jump witness repairs boundaries without changing cdf mass."""
    fit, inputs, weights = _witness_fit(family), np.array([0.]), np.full(4, .25)
    before, original = _legacy_quantile_bank([fit], inputs, weights)
    assert np.count_nonzero(np.diff(original, axis=-1) < 0) >= 1
    assert np.count_nonzero(before < 0) >= 1
    bank, boundaries = inference._epig_continuous_bank([fit], inputs, weights, 64, discrete=True)
    expected_boundaries = np.maximum.accumulate(original, axis=-1)
    assert boundaries.tobytes() == expected_boundaries.tobytes()
    expected_bank = _cdf_masses(_distributions([fit], inputs, discrete=True), expected_boundaries)
    assert bank.tobytes() == expected_bank.tobytes()
    assert np.all(np.diff(boundaries, axis=-1) >= 0) and np.all(bank >= 0)
    np.testing.assert_allclose(bank.sum(-1), 1., atol=1e-12, rtol=0)
    expected_repairs = {"rows_repaired": 1,
                        "max_shift": float(np.max(expected_boundaries - original)), "path": "discrete"}
    assert getattr(bank, "boundary_repairs", None) == expected_repairs
    # Repeated cut points describe genuine empty intervals, with no jitter,
    # integer alignment, clipping, dropped atoms, or probability normalization.
    duplicates = np.diff(boundaries[0]) == 0
    assert duplicates.any() and np.all(bank[:, 0, 1:-1][:, duplicates] == 0)
    assert np.any(boundaries != np.floor(boundaries))
    scores = predictive_information(bank, bank, weights, [1.])
    assert np.all(np.isfinite(scores["epig"])) and np.all(np.isfinite(scores["joint_eig"]))
    assert np.all(scores["epig"] >= -1e-12)
    assert np.all(scores["epig"] <= scores["joint_eig"] + 1e-12)
    if family == "zi_poisson":
        means = np.exp(fit.particles["b0"])
        expected_atom = .2 + .8 * np.exp(-means)
        np.testing.assert_allclose(bank[:, 0, 0], expected_atom, atol=1e-15, rtol=0)
    (tmp_path / f"boundary_witness_{family}.json").write_text(json.dumps({
        "family": family, "means": np.exp(fit.particles["b0"]).tolist(),
        "boundary_reversals_before": int(np.count_nonzero(np.diff(original, axis=-1) < 0)),
        "boundary_reversals_after": int(np.count_nonzero(np.diff(boundaries, axis=-1) < 0)),
        "negative_masses_before": int(np.count_nonzero(before < 0)),
        "negative_masses_after": int(np.count_nonzero(bank < 0)),
        "minimum_mass_before": float(before.min()), "minimum_mass_after": float(bank.min()),
        "maximum_row_sum_error": float(np.max(np.abs(bank.sum(-1) - 1))),
        "repairs": expected_repairs, "epig": scores["epig"].tolist(),
    }, indent=2) + "\n")


def test_chunked_repairs_count_input_rows_once_and_preserve_repaired_boundaries():
    """Check: chunked repairs count input rows once and preserve repaired boundaries."""
    fit, inputs, weights = _witness_fit("poisson"), np.linspace(0., 1., 35), np.full(4, .25)
    _, original = _legacy_quantile_bank([fit], inputs, weights)
    expected_boundaries = np.maximum.accumulate(original, axis=-1)
    expected_bank = _cdf_masses(_distributions([fit], inputs, discrete=True), expected_boundaries)
    bank, info = inference._epig_discrete_coarsened_bank([fit], inputs, weights, 3, _refusal(len(inputs)))
    assert info["epig_discrete_input_chunk_size"] == 16
    assert bank.tobytes() == expected_bank.tobytes()
    assert np.asarray(info["epig_discrete_bin_boundaries"]).tobytes() == expected_boundaries.tobytes()
    assert info[REPAIR_KEY] == {"rows_repaired": len(inputs),
                               "max_shift": float(np.max(expected_boundaries - original)), "path": "discrete"}
    assert info["epig_discrete_probabilities_renormalized"] is False
    assert info["epig_discrete_mass_dropped"] is False


def test_discrete_acquisition_record_preserves_repair_receipt_at_both_levels(monkeypatch):
    """Check: discrete acquisition record preserves repair receipt at both levels."""
    fit, inputs = _witness_fit("poisson"), np.array([0., .25, .5, 1.])
    def refuse(*args):
        """Helper refuse used by the tests in this module."""
        raise inference.EpigResourceLimit(_refusal(len(inputs)))
    monkeypatch.setattr(inference, "_epig_discrete_banks", refuse)
    record = _record([fit], inputs)
    info = record["outcome_info"]
    assert record[REPAIR_KEY] == info[REPAIR_KEY]
    assert record[REPAIR_KEY]["rows_repaired"] == len(inputs)
    assert record[REPAIR_KEY]["path"] == "discrete"
    boundaries = np.asarray(info["epig_discrete_bin_boundaries"])
    assert np.all(np.diff(boundaries, axis=-1) >= 0)
    assert json.dumps(record, sort_keys=True, allow_nan=False) == json.dumps(
        _record([fit], inputs), sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("family", ("normal", "poisson", "negative_binomial", "zi_poisson"))
def test_already_monotone_banks_and_acquisition_json_are_byte_identical(family, monkeypatch):
    """Check: already monotone banks and acquisition json are byte identical."""
    discrete = family != "normal"
    fit = (count_fit(family, means=(40., 40.), slopes=(0., 0.)) if discrete
           else grammar_fit("normal", intercepts=(-2., .7), slopes=(.3, -.2)))
    inputs, weights = np.array([0., .25, .5, 1.]), np.full(2, .5)
    before, boundaries_before = _legacy_quantile_bank([fit], inputs, weights, discrete=discrete)
    assert np.all(np.diff(boundaries_before, axis=-1) >= 0)
    actual = inference._epig_continuous_bank([fit], inputs, weights, 64, discrete=discrete)
    if discrete:
        bank, boundaries = actual
        assert boundaries.tobytes() == boundaries_before.tobytes()
        def refuse(*args):
            """Helper refuse used by the tests in this module."""
            raise inference.EpigResourceLimit(_refusal(len(inputs)))
        monkeypatch.setattr(inference, "_epig_discrete_banks", refuse)
    else:
        bank = actual
    assert bank.tobytes() == before.tobytes()
    assert type(bank) is np.ndarray and getattr(bank, "boundary_repairs", None) is None
    record = _record([fit], inputs)
    assert REPAIR_KEY not in record and REPAIR_KEY not in record.get("outcome_info", {})
    def legacy(fits, points, weights, grid_size, *, discrete=False):
        """Helper legacy used by the tests in this module."""
        bank, boundaries = _legacy_quantile_bank(fits, points, weights, grid_size, discrete=discrete)
        return (bank, boundaries) if discrete else bank
    monkeypatch.setattr(inference, "_epig_continuous_bank", legacy)
    assert json.dumps(record, sort_keys=True, allow_nan=False) == json.dumps(
        _record([fit], inputs), sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("invalid", (np.nan, np.inf, -.1))
def test_repair_does_not_hide_invalid_interval_cdfs(monkeypatch, invalid):
    """Check: repair does not hide invalid interval cdfs."""
    class InvalidCDF:
        """Test helper class InvalidCDF."""
        def ppf(self, probabilities):
            """Helper ppf used by the tests in this module."""
            return np.broadcast_to(probabilities * 10., (2, 2, len(probabilities))).copy()
        def cdf(self, boundaries):
            """Helper cdf used by the tests in this module."""
            result = np.broadcast_to(np.linspace(.1, .9, boundaries.shape[-1]),
                                     (2, 2, boundaries.shape[-1])).copy()
            result[:, :, 31] = invalid
            return result
    monkeypatch.setattr(inference, "_epig_scalar_distribution", lambda *args: InvalidCDF())
    with pytest.raises(ValueError, match="interval normalization failed; no renormalization"):
        inference._epig_continuous_bank([count_fit()], np.array([0., 1.]), np.full(2, .5), 64,
                                        discrete=True)


def test_boxing_synthetic_mixed_count_runner_records_repairs_in_two_rounds(tmp_path, monkeypatch):
    """Check: boxing synthetic mixed count runner records repairs in two rounds."""
    from . import boxinggym_core as bg
    from .test_named_inputs import bg_run_args, bg_scalar_stub_fits

    class SyntheticCountGoal:
        """Test helper class SyntheticCountGoal."""
        def __init__(self, env):
            self.env = env
            self.eval_points = []
        def get_goal_eval_question(self, include_prior=True):
            """Helper get goal eval question used by the tests in this module."""
            x = float(np.random.uniform(.1, 4.9))
            self.eval_points.append((x, float(np.rint(1000. * np.exp(.03 * x)))))

    class SyntheticCountEnvironment:
        """Test helper class SyntheticCountEnvironment."""
        def __init__(self, spec, seed):
            self.spec = spec
            self.env = SimpleNamespace(alpha=np.log(1000.), beta1=.03, beta2=0., beta3=0.)
            self.goal = SyntheticCountGoal(self.env)
            self.observation_calls = 0
        def evaluation_points(self, count, seed):
            """Helper evaluation points used by the tests in this module."""
            x = np.linspace(.1, 4.9, count)
            y = np.rint(1000. * np.exp(.03 * x))
            self.goal.eval_points = list(zip(x, y, strict=True))
            return x, y
        def eval_inputs(self, point):
            """Helper eval inputs used by the tests in this module."""
            return [point[0]]
        def world_parameters(self):
            """Helper world parameters used by the tests in this module."""
            return {"synthetic_count_world": True}
        def refused_designs(self):
            """Helper refused designs used by the tests in this module."""
            return frozenset()
        def accepts(self, handle):
            """Helper accepts used by the tests in this module."""
            return 0 <= float(handle) <= 5
        def observe(self, handle):
            """Helper observe used by the tests in this module."""
            self.observation_calls += 1
            return float(round(1000. * np.exp(.03 * float(handle))))
        def evaluate(self, predictions):
            """Helper evaluate used by the tests in this module."""
            predictions = np.array([float(value) for value in predictions])
            measurements = np.array([point[-1] for point in self.goal.eval_points])
            return {"metric": float(np.mean((predictions - measurements)**2)), "spread": 0.}

    monkeypatch.setitem(bg.TASKS, "peregrines", replace(bg.TASKS["peregrines"], module=__name__))
    monkeypatch.setattr(bg, "TaskEnvironment", SyntheticCountEnvironment)
    bg_scalar_stub_fits(monkeypatch)
    original_fit = bg.scalar_fit
    def mixed_count_fit(hypothesis, inputs, labels, seed, transform, response_unit, settings,
                        interface=None, response_center=0.):
        """Helper mixed count fit used by the tests in this module."""
        result = original_fit(hypothesis, inputs, labels, seed, transform, response_unit, settings,
                              interface, response_center)
        particles = dict(result.particles)
        means = np.tile([.56, 40., 8000., 10000.], 2)
        for name, _ in hypothesis.parameters:
            particles[name] = np.log(means / response_unit) if name == "b0" else np.zeros(8)
        return replace(result, particles=particles, log_evidence=0.)
    monkeypatch.setattr(bg, "scalar_fit", mixed_count_fit)
    output = tmp_path / "synthetic_runner"
    bg.run(bg_run_args(output, task="peregrines", cell="102", rounds=2))
    for index in (1, 2):
        record = json.loads((output / "acquisition" / f"t{index:02d}.json").read_text())
        info = record["outcome_info"]
        assert info["epig_discrete_coarsened"] is True
        assert info["epig_discrete_path_fixed_at_start"] is True
        assert info["epig_discrete_path"] == "coarsened"
        if index == 1:
            assert info["epig_discrete_enumeration_refusal"]["resource"] == "acquisition_arrays"
            outcome_keys = set(info) - {"epig_discrete_boundary_repairs"}
        else:
            assert info["epig_discrete_enumeration_refusal"] is None
            assert info["epig_discrete_enumeration_cutoffs_known"] is False
            assert info["epig_discrete_enumeration_cutoff_note"] == "not_recomputed: coarsened_path_fixed_at_D0"
            assert set(info) - {"epig_discrete_boundary_repairs"} == outcome_keys
        assert record[REPAIR_KEY] == info[REPAIR_KEY]
        assert record[REPAIR_KEY]["rows_repaired"] >= 1
        assert record[REPAIR_KEY]["max_shift"] > 0 and record[REPAIR_KEY]["path"] == "discrete"
        boundaries = np.asarray(info["epig_discrete_bin_boundaries"])
        assert boundaries.shape == (len(record["rows"]) + 32, 63)
        assert np.all(np.diff(boundaries, axis=-1) >= 0)
        assert record["rows"][record["policy"]["chosen_index"]]["design"] == record["chosen"]
        assert all(np.isfinite(row["epig"]) and -1e-10 <= row["epig"] <= row["joint_eig"] + 1e-10
                   for row in record["rows"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["paid_api_calls"] == 0 and summary["environment_calls"] == 6


def test_continuous_path_repairs_a_controlled_quantile_reversal_and_records_it(monkeypatch):
    """Check: continuous path repairs a controlled quantile reversal and records it."""
    class SyntheticQuantileCDF:
        """Inject an ordered-CDF law with deliberately reversed search outputs.

        This probes the continuous repair/record branch independently of the
        natural normal fixtures, whose actual quantiles need no repair.
        """
        def __init__(self, input_count):
            self.input_count = input_count
        def ppf(self, probabilities):
            """Helper ppf used by the tests in this module."""
            boundaries = probabilities.copy()
            boundaries[:3] = probabilities[[2, 0, 1]]
            return np.broadcast_to(boundaries, (2, self.input_count, len(boundaries))).copy()
        def cdf(self, boundaries):
            """Helper cdf used by the tests in this module."""
            return np.broadcast_to(boundaries, (2, self.input_count, boundaries.shape[-1])).copy()

    monkeypatch.setattr(inference, "_epig_scalar_distribution",
                        lambda fit, points: SyntheticQuantileCDF(len(points)))
    fit, inputs, weights = grammar_fit("normal"), np.array([0., .25, .5, 1.]), np.full(2, .5)
    before, original = _legacy_quantile_bank([fit], inputs, weights, discrete=False)
    assert np.any(before < 0)
    bank = inference._epig_continuous_bank([fit], inputs, weights, 64)
    expected = np.maximum.accumulate(original, axis=-1)
    assert np.all(expected[:, :3] == 3 / 64)
    assert bank.tobytes() == _cdf_masses(_distributions([fit], inputs, discrete=False), expected).tobytes()
    assert np.all(bank >= 0)
    np.testing.assert_allclose(bank.sum(-1), 1., atol=1e-12, rtol=0)
    repairs = {"rows_repaired": len(inputs), "max_shift": 2 / 64, "path": "continuous"}
    assert bank.boundary_repairs == repairs
    chosen, rows = inference.choose_epig([fit], inputs[:2], 71, target_points=inputs[2:])
    assert rows.boundary_repairs == repairs and rows.outcome_info is None
    record = inference.epig_acquisition_record(chosen, rows, 71)
    assert record[REPAIR_KEY] == repairs
    assert "epig_excluded_particles" not in record and "epig_excluded_models" not in record
    assert all(np.isfinite(row["epig"]) and np.isfinite(row["joint_eig"]) for row in record["rows"])


def test_emotion_runner_forwards_component_boundary_repairs_only_when_present(tmp_path):
    """Multi-component records name the repaired component; unrepaired runs keep the old bytes."""
    import json

    import pytest

    from bayes_model_discovery.experimental import boxinggym_core as bg
    from bayes_model_discovery.experimental import count_inference as inference
    from bayes_model_discovery.experimental.test_epig_positivity_guard import (
        synthetic_emotion_record,
    )

    real_rows = inference.EpigRows
    created = []

    def rows_with_one_repair(rows, outcome_info, exclusion_info=None):
        """Helper rows with one repair used by the tests in this module."""
        table = real_rows(rows, outcome_info, exclusion_info)
        if not created:
            table.boundary_repairs = {"rows_repaired": 2, "max_shift": 1e-15, "path": "discrete"}
        created.append(table)
        return table

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(inference, "EpigRows", rows_with_one_repair)
        repaired, _ = synthetic_emotion_record(tmp_path / "repaired", patch)
    with pytest.MonkeyPatch.context() as patch:
        plain, _ = synthetic_emotion_record(tmp_path / "plain", patch)
    record = json.loads(repaired)
    first = bg.TASKS["emotion"].components[0]
    assert record["epig_component_boundary_repairs"] == {
        first: {"rows_repaired": 2, "max_shift": 1e-15, "path": "discrete"}}
    assert "epig_component_boundary_repairs" not in json.loads(plain)
    assert "epig_discrete_boundary_repairs" not in json.loads(plain)
