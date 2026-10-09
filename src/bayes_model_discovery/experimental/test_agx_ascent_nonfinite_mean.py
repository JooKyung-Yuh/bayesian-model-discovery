"""Acceptance test for the ChemBench ascent particle guard: an ascent whose pool holds a posterior
particle with a nonpositive mean inside the search box must still finish its rounds.

A ChemBench cell-101 ascent run (joint_eig) stopped in its round-1 acquisition with
ValueError("non-finite/nonpositive mean; no clipping fallback"). The error came from the ascent
objective's first perturbation batch: chembench_core.run -> design_search.projected_spsa
(perturbation batch) -> evaluate -> the ascent objective -> positive_inference.joint_eig ->
draw_conditional -> particle_means -> grammar_means. Before the guard this test fails with that same
error on that same chain. The test asserts only the desired outcome, that the cell finishes
every ascent round. It asserts nothing about how a design treats the offending particle: no
exclusion keys and no chosen design, because the guard design is not decided.

Every input is synthetic. The world is test_chembench_core's eight-input fake adapter
(f0..f7 on [0.1, 1.0]). autoscilab's three ChemBench declaration constants are pointed at that
same box, because design_search.declared_settings reads the search box from them and
chembench_core.run refuses an adapter schema that differs from the declaration. The cell is
101 with --design-search ascent, joint_eig, seed 31, dev profile, 3 rounds, and the unpatched
mock proposer with --initial-proposals 4. The mock burst admits b0+b1*f0z+b2*f1z (lognormal) at
round 0, so round 1's ascent runs on a pool of four catalogue models and that one grammar
model.

The reproduction condition is the only patch outside the fake world, and it touches only the
pool's input, the fit. chembench_core._timed_fit is wrapped. After the REAL round-0 fit of the
burst's grammar candidate, the posterior particle with the smallest sigma gets
b2 = 0 and a steep line b0 + b1*f0z. The line is placed from the round-1 ascent's own seeded
draws, which the test recomputes the way design_search.projected_spsa draws them:
  - The particle's mean response_unit*exp(b0 + b1*f0z) stays positive at the five start
    designs, the initial design set (D0) observations and the evaluation points: its exponent there is at least
    LOG_SMALLEST_POSITIVE + MARGIN (about -734.4). The start evaluations therefore pass, and so
    do the round-0 predictions and the burst packets.
  - At the lowest-f0 design of the step-1 perturbation batch the exponent is
    LOG_SMALLEST_POSITIVE - 2*MARGIN (about -764.4), so exp underflows to exactly 0.0. The first
    perturbation batch therefore raises before the guard.
  - Its exponent stays below 0 on the whole box, so the particle never overflows anywhere.
Nothing on the acquisition path is patched: not the objective, joint_eig, draw_conditional,
particle_means, grammar_means, projected_spsa, evaluate or choose_design. The same test must
therefore pass unchanged once the guard handles such a particle.
"""

import json
from dataclasses import replace

import numpy as np
import pytest

from . import chembench_core, design_search
from . import test_chembench_core as fake
from .core_count import seed_for
from .count_models import named_input_values
from .positive_inference import is_grammar

SEED = 31
CELL = "101"
ROUNDS = 3
INITIAL_PROPOSALS = 4
#: The mock burst's one admitted grammar candidate (mock schedule entry 3; entry 1 duplicates a
#: catalogue member and entries 0 and 2 are gamma, which cell 101 refuses).
BURST_EXPRESSION = "b0+b1*f0z+b2*f1z"
FIT_ROUND = 0
FAILING_ROUND = 1
#: log of the smallest positive double: exp(x) is 0.0 below about this value minus 0.69.
LOG_SMALLEST_POSITIVE = float(np.log(np.finfo(float).smallest_subnormal))
MARGIN = 10.0
FAILURE_MESSAGE = "non-finite/nonpositive mean; no clipping fallback"


def install_world(patch):
    """The eight-input fake world, for the adapter and for the search-box declaration."""
    import autoscilab.oracle.chembench as oracle

    patch.setattr(fake.real_adapter, "ChemBenchSession", fake._FakeSession)
    patch.setattr(fake.real_adapter, "sample_designs", fake._fake_sample_designs)
    patch.setattr(oracle, "CHEM_INPUT_VARS", list(fake.FAKE_NAMES))
    patch.setattr(oracle, "CHEM_INPUT_BOUNDS",
                  {name: tuple(bounds) for name, bounds in fake.FAKE_BOUNDS.items()})
    patch.setattr(oracle, "CHEM_LOG_VARS", set())


def cell_args(output, manifest):
    """Build the runner arguments for the ascent cell under test."""
    return fake._args(output, manifest, cell=CELL, seed=SEED, rounds=ROUNDS, particles=None,
                      q=None, structure_proposals=None, proposer="mock",
                      design_policy="joint_eig", design_search="ascent",
                      initial_proposals=INITIAL_PROPOSALS)


def _matrix(rows):
    return np.array([[row[name] for name in fake.FAKE_NAMES] for row in rows])


def ascent_step_one(round_index):
    """Round `round_index`'s five start designs and its step-1 perturbation batch (plus, minus
    per restart), drawn as design_search.projected_spsa draws them. Read only."""
    transform = design_search.SearchTransform.from_bounds(fake.FAKE_NAMES, fake.FAKE_BOUNDS, ())
    gains = design_search.PARAMETER_DEFAULTS
    c_1 = gains["c"] / 1 ** gains["gamma"]
    starts, batch = [], []
    for restart in range(1, design_search.RESTARTS + 1):
        rng = np.random.default_rng(seed_for("design_restart", SEED, CELL, round_index, restart))
        start = rng.uniform(0.0, 1.0, len(fake.FAKE_NAMES))
        delta = rng.choice((-1.0, 1.0), len(fake.FAKE_NAMES))
        starts.append(start)
        batch.extend((np.clip(start + c_1 * delta, 0, 1), np.clip(start - c_1 * delta, 0, 1)))
    return transform.inverse(np.array(starts)), transform.inverse(np.array(batch))


def install_reproduction(patch, record):
    """Wrap the ChemBench fit so that one particle of the round-0 grammar fit underflows.

    `record` receives what was changed. A broken precondition calls pytest.fail. pytest.fail
    raises a BaseException, so the burst's per-candidate `except Exception` cannot turn it into
    a quiet candidate failure.
    """
    real_fit = chembench_core._timed_fit

    def fit_then_inflate(output, round_index, timeout_seconds, *fit_args, **fit_kwargs):
        result = real_fit(output, round_index, timeout_seconds, *fit_args, **fit_kwargs)
        hypothesis = fit_args[0]
        if round_index != FIT_ROUND or not is_grammar(hypothesis):
            return result
        if hypothesis.expression != BURST_EXPRESSION or record:
            pytest.fail(f"unexpected round-0 grammar fit {hypothesis.expression!r}")
        observed, response_unit, interface = fit_args[1], float(fit_args[5]), fit_kwargs["interface"]
        starts, batch = ascent_step_one(FAILING_ROUND)
        evaluation = _matrix(fake._fake_sample_designs(
            chembench_core.PROFILE_GRID["dev"]["q"], seed_for("chem_evaluation_inputs", 20260907)))
        must_stay_positive = np.concatenate([starts, observed, evaluation])
        top = np.array([[bounds[1] for bounds in fake.FAKE_BOUNDS.values()]])

        def f0z(designs):
            return named_input_values(interface, designs)["f0z"]

        low_valid, low_batch = f0z(must_stay_positive).min(), f0z(batch).min()
        if not low_batch < low_valid:
            pytest.fail("the step-1 batch reaches no lower f0 than the designs that must stay valid")
        particles = {name: np.array(values, dtype=float) for name, values in result.particles.items()}
        particle = int(np.argmin(particles["sigma"]))
        slope = 3 * MARGIN / (low_valid - low_batch)
        intercept = LOG_SMALLEST_POSITIVE + MARGIN - slope * low_valid
        particles["b0"][particle], particles["b1"][particle] = intercept, slope
        particles["b2"][particle] = 0.0

        def means(designs):
            named = named_input_values(interface, designs)
            exponent = (particles["b0"][:, None] + particles["b1"][:, None] * named["f0z"][None, :]
                        + particles["b2"][:, None] * named["f1z"][None, :])
            with np.errstate(over="ignore", under="ignore"):
                return response_unit * np.exp(exponent), exponent

        kept, _ = means(must_stay_positive)
        reached, reached_exponent = means(batch)
        _, top_exponent = means(top)
        zero = np.flatnonzero(reached[particle] == 0.0)
        if not (np.all(np.isfinite(kept)) and np.all(kept > 0)):
            pytest.fail("a particle mean is not positive where the ascent must still start")
        if not len(zero) or top_exponent[particle, 0] >= 0:
            pytest.fail("the modified particle does not underflow in the batch, or overflows")
        record.update(model_id=hypothesis.model_id, particle=particle, b0=intercept, b1=slope,
                      zero_batch_rows=zero.tolist(),
                      zero_batch_f0=[float(batch[i, 0]) for i in zero],
                      lowest_start_f0=float(starts[:, 0].min()),
                      lowest_batch_exponent=float(reached_exponent[particle].min()))
        return replace(result, particles=particles)

    patch.setattr(chembench_core, "_timed_fit", fit_then_inflate)


@pytest.mark.usefixtures("require_autoscilab_source")
def test_a_nonpositive_mean_only_the_perturbation_batch_reaches_does_not_stop_the_ascent(
        tmp_path, monkeypatch):
    """A nonpositive mean reached only by the perturbation batch must not stop the ascent."""
    install_world(monkeypatch)
    inflated = {}
    install_reproduction(monkeypatch, inflated)
    run_dir = tmp_path / "cell"

    # Before the guard this raises ValueError(FAILURE_MESSAGE) from the round-1 perturbation batch.
    chembench_core.run(cell_args(run_dir, fake._manifest(tmp_path)))

    assert not (run_dir / "failure.json").exists()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "completed_fixed_pool_first_path"
    assert summary["rounds"] == ROUNDS
    for t in range(1, ROUNDS + 1):
        acquisition = json.loads((run_dir / "acquisition" / f"t{t:02d}.json").read_text())
        assert acquisition["design_search"] == "ascent"
        assert len(acquisition["arrival_rows"]) == design_search.RESTARTS
    # The reproduction condition was in the pool that round 1's ascent read.
    assert inflated, "the burst's grammar candidate was never fitted at round 0"
    with np.load(run_dir / "fits" / f"t00_{inflated['model_id']}.npz") as saved:
        particle = inflated["particle"]
        assert saved["b0"][particle] == inflated["b0"]
        assert saved["b1"][particle] == inflated["b1"]
        assert saved["b2"][particle] == 0.0
