"""Design-exclusion candidate-set rule for survival.

Fact (declared in advance): survival's design IS the patient ID -
`self.outcomes[patientID]` is a FIXED record - so re-observing one patient
yields no new information. `TaskSpec.design_exclusion` therefore removes an
already-observed design from the candidate pool before EVERY round's draw
(expected information gain (EIG) arm and random arm alike), and the initial design set (D0) is drawn without duplicates too. This
is an adapter CANDIDATE-SET rule, not a design policy: which arm wins each
round is untouched by it, only what that arm is allowed to choose FROM.

Five things are pinned here, all PyMC-free and fast (no sequential Monte Carlo (SMC) fit is needed to
exercise the pool/candidate-set mechanics; the one test that touches the
real boxing_gym environment only builds and validates a design pool, the
same fast operation `test_every_new_task_design_pool_is_legal_and_d0_comes_
from_it` in test_binary_real.py already exercises for hyperbolic/location):

1. the default is "none" and only survival opts in - every other task's own
   candidate pool is therefore untouched, byte-for-byte (see the
   development-run comparisons for the actual byte-identity evidence);
2. `design_pool_target` grows the requested pool only for a
   `design_exclusion="observed"` task, and only once the profile's own grid
   size cannot cover D0 + every requested round - this is what makes a
   20-round survival run possible at all without breaking the exclusion
   rule it is asked to keep;
3. `DesignPool.available` removes exactly the observed rows and nothing
   else, preserves the remaining rows' relative order, and raises rather
   than silently returning an empty candidate set;
4. `DesignPool.available` also raises when the observed set is complete -
   an empty candidate set is a caller sizing bug, not a condition to hide;
5. end to end (real environment, no fits): building the pool the way
   `run()` now sizes it for a 20-round survival run and then repeatedly
   drawing from `pool.available(observed)` - exactly the loop body `run()`
   executes every round - visits 24 distinct patients with zero repeats,
   the same invariant the slow, real development run is asked to show.

The environment test skips when the read-only `boxing_gym` package is not on
PYTHONPATH, matching every other boxinggym_core test in this tree.
"""

import numpy as np
import pytest


def adapter():
    """Import the BoxingGym core adapter, skipping the test when it is unavailable."""
    return pytest.importorskip(
        "bayes_model_discovery.experimental.boxinggym_core",
        reason="adapter imports the read-only boxing_gym package lazily",
    )


def test_default_is_none_and_only_survival_and_irt_opt_in():
    """irt joins survival, from the same structural fact.

    `IRT.reset` fixes the whole response matrix with ONE prior-predictive draw
    (irt.py:305, :330-:340) and `step` returns that stored cell verbatim
    (irt.py:403-:405), so re-observing a (student, question) pair carries no
    new information - exactly survival's condition. Pre-declared, never inferred from a run.
    """
    bg = adapter()
    opted_in = {"survival", "irt"}
    for name in opted_in:
        assert bg.TASKS[name].design_exclusion == "observed", name
    for name, spec in bg.TASKS.items():
        if name in opted_in:
            continue
        assert spec.design_exclusion == "none", name


def test_pool_target_is_unchanged_for_every_design_exclusion_none_task():
    """Check that pool target is unchanged for every design exclusion none task."""
    bg = adapter()
    for name in ("death", "peregrines", "dugongs", "hyperbolic", "location"):
        spec = bg.TASKS[name]
        for rounds in (1, 5, 20):
            assert bg.design_pool_target(spec, 16, rounds) == 16
            assert bg.design_pool_target(spec, 100, rounds) == 100


def test_pool_target_grows_only_once_the_profile_grid_cannot_cover_the_run():
    """Check that pool target grows only once the profile grid cannot cover the run."""
    bg = adapter()
    spec = bg.TASKS["survival"]
    d0 = len(bg.D0_FRACTIONS)
    assert d0 == 4
    # dev profile (16 - the earlier constant): unchanged while 4+rounds<=16.
    assert bg.design_pool_target(spec, 16, 1) == 16
    assert bg.design_pool_target(spec, 16, 5) == 16
    assert bg.design_pool_target(spec, 16, 12) == 16  # 4+12=16, exactly full
    assert bg.design_pool_target(spec, 16, 13) == 17  # 4+13=17 > 16
    assert bg.design_pool_target(spec, 16, 20) == 24  # 4+20=24, the production run
    # production profile (100) already covers every legal --rounds value
    # (capped at 20 by `run`'s own argument check), so it never grows.
    assert bg.design_pool_target(spec, 100, 20) == 100


def test_available_removes_exactly_the_observed_rows_and_keeps_order():
    """Check that available removes exactly the observed rows and keeps order."""
    bg = adapter()
    rows = np.arange(10, dtype=float)[:, None]
    pool = bg.DesignPool(rows, tuple(str(i) for i in range(10)), False)
    kept = pool.available({2, 5, 9})
    assert list(kept[:, 0]) == [0.0, 1.0, 3.0, 4.0, 6.0, 7.0, 8.0]
    # the empty exclusion set is a no-op - the full pool, in pool order.
    assert list(pool.available(set())[:, 0]) == list(range(10))


def test_available_raises_when_every_design_is_already_observed():
    """Check that available raises when every design is already observed."""
    bg = adapter()
    rows = np.arange(3, dtype=float)[:, None]
    pool = bg.DesignPool(rows, ("0", "1", "2"), False)
    with pytest.raises(ValueError, match="exhausted"):
        pool.available({0, 1, 2})
    # over-complete observed sets (indices beyond the pool) are just as empty.
    with pytest.raises(ValueError, match="exhausted"):
        pool.available({0, 1, 2, 99})


def test_survival_pool_sized_for_a_20_round_run_has_zero_repeats_in_24_draws():
    """PyMC-free integration check, using the real fixed patient pool.

    Drives the identical pool/`available`/`index_of` sequence `run()`'s round
    loop executes - build the pool at the size `design_pool_target` now asks
    for, seed `observed` from `d0_indices()`, then repeatedly filter-and-draw
    - without any SMC fit, belief packet, or proposer machinery. 4 (D0) + 20
    (rounds) = 24 draws from a 24-row pool: every row is used exactly once.
    """
    bg = adapter()
    spec = bg.TASKS["survival"]
    pytest.importorskip(spec.module)
    environment = bg.TaskEnvironment(spec, 20260907)
    rounds = 20
    target = bg.design_pool_target(spec, 16, rounds)
    assert target == 24
    pool = bg.build_design_pool(spec, environment, target)
    assert len(pool.rows) == target  # the 100-patient pool easily covers 24
    for row, handle in zip(pool.rows, pool.handles, strict=True):
        assert environment.accepts(handle), row
        assert pool.index_of(row) == list(pool.handles).index(handle)
    d0_indices = pool.d0_indices()
    assert len(set(d0_indices)) == len(d0_indices) == 4
    observed = set(d0_indices)
    draw_rng = np.random.default_rng(20260907)
    for _ in range(rounds):
        designs = pool.available(observed)
        assert len(designs) == target - len(observed)
        x = designs[draw_rng.integers(len(designs))]
        chosen_index = pool.index_of(x)
        assert chosen_index not in observed
        observed.add(chosen_index)
    assert len(observed) == len(bg.D0_FRACTIONS) + rounds == target
    with pytest.raises(ValueError, match="exhausted"):
        pool.available(observed)
