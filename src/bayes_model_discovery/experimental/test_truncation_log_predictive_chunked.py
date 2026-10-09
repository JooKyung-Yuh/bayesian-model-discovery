"""Chunked `log_predictive`: `positive_inference.log_predictive` split along the draw axis, byte for byte.

Synthetic pools only; no run output, seed or response of any real cell is read.

* Unit: the module at BASE_COMMIT is read with `git show` and executed as an isolated module
  of this package, so only `log_predictive` differs from the current one. Both return arrays of
  every call are compared as dtype, shape, memory order and bytes (and `np.array_equal`, no NaN)
  at forced block widths 1, 7, 128 and unsplit, for every scalar family, for pools of unequal
  particle counts, at the default budget, and at the one-input layouts, where a block narrower than two columns would change numpy's summation order.
* Callers: every `log_predictive` call path is driven old against new through its own caller.
* Integration (opt-in: `BMD_BASELINE_COMPARE=1` and `BMD_TRM_BASELINE_DIR`, a clean checkout of
  PARITY_BASE_COMMIT; BoxingGym also `BMD_BOXINGGYM_SRC`): a mock CORE cell and a mock ChemBench cell, each
  with pool cap 3, and a mock BoxingGym dugongs cell run in that checkout and in this one, the
  latter at the default budget and at a forced narrow one. Science files are compared as bytes;
  the operational fields named below are excluded by name. Each run logs its calls by caller.
"""

import ast
import csv
import fnmatch
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from . import boxinggym_truth_law as truth_law
from . import chembench_core as cc
from . import core_count, frozen_kl_rescore, gaussian_core
from . import positive_inference as current
from .belief_packet import core_real_interface
from .count_inference import Fit
from .real_inference import RealHypothesis

# The module the unit tests compare against, read with `git show`; those tests skip when this commit is absent
# from the checkout's history.
BASE_COMMIT = "7f6b7b245825c5906d71781e8e310493f133e775"
#: The three opt-in integration tests compare against a clean checkout of PARITY_BASE_COMMIT (the commit that
#: introduced the chunked predictive) and skip when none is supplied, with the sources and record keys changed
#: since then named below; the unit tests keep BASE_COMMIT, the module from before the chunking change that
#: they compare to.
PARITY_BASE_COMMIT = "ad583dfc5d7bc8f51d441ac890a6f540a3d73ecc"
#: The recorded sources changed since PARITY_BASE_COMMIT, a literal list per bundle: belief_packet.py,
#: count_proposer.py and chembench_core.py changed in later commits; the chunking change's own
#: positive_inference.py is the base's and no longer moves.
#: Later commits also changed count_models.py (the validation-message switch and sentences) and the
#: three runners' option wiring (core_count.py, boxinggym_core.py, chembench_core.py); count_proposer.py
#: is already listed.
LATER_UNIT_SOURCES = {
    "core": ("belief_packet.py", "count_proposer.py", "count_models.py", "core_count.py"),
    "boxinggym": ("belief_packet.py", "count_proposer.py", "count_models.py", "core_count.py",
                  "boxinggym_core.py"),
    "chembench": ("chembench_core.py", "count_models.py", "core_count.py"),
}
#: The record keys later commits added to a ChemBench cell, removed by name
#: from the current runs only and asserted absent from the base's: config's `constraints_disclosure`,
#: the packet's two rules and config's stage-slot record. A
#: proposal's packet digest pair moves with its packet and is re-checked against that file.
LATER_UNIT_CONFIG_KEYS = {"chembench": ("constraints_disclosure", "stage_slots")}
LATER_UNIT_PACKET_CONSTRAINT_KEYS = {"chembench": ("coefficient_name_rule", "prior_probe_rule")}
#: The files later commits added to a ChemBench cell, absent from the base run and present in every
#: current run, then left out of the comparison: round_timings.jsonl, an observational wall-time sidecar.
LATER_UNIT_FILES = {"chembench": ("round_timings.jsonl",)}
PACKET_DIGEST_KEYS = ("packet_sha256", "packet_bytes")
#: The line numbers pinned in this block are call sites in the base commit's sources and in this tree;
#: the comparison is skipped when the base commit is absent from the checkout (see `_baseline`).
#: A later change moved chembench_core's two guarded call sites one line down
#: (:991/:998 at PARITY_BASE_COMMIT, :992/:999 afterwards); the base's call log is read through this
#: map and nothing else.
#: A later commit inserted four import lines above chembench_core's two guarded call sites
#: (DEFAULT_VALIDATION_MESSAGES, VALIDATION_MESSAGE_VERSIONS, set_validation_messages,
#: RESPONSE_FORMATS): :992 -> :996 and :999 -> :1003; four import lines above core_count's
#: one_step_log_score call: :920 -> :924; four import lines and three run() lines above
#: boxinggym_core's two call sites: our_scores :3400 -> :3407, one_step_score :3687 -> :3694.
#: The code at every mapped line is the same call.
#: A later commit added set_prompt_variant above ChemBench's calls: :996/:1003 -> :997/:1004.
#: A further wrapper is placed below these calls; this map corrects only the inherited line anchors of these call sites.
LATER_UNIT_CALL_SITES = {
    "chembench_core.py:guarded_log_predictive:991": "chembench_core.py:guarded_log_predictive:997",
    "chembench_core.py:guarded_log_predictive:998": "chembench_core.py:guarded_log_predictive:1004",
    "core_count.py:one_step_log_score:920": "core_count.py:one_step_log_score:924",
    "boxinggym_core.py:our_scores:3400": "boxinggym_core.py:our_scores:3407",
    "boxinggym_core.py:one_step_score:3687": "boxinggym_core.py:one_step_score:3694",
}
REPO = Path(__file__).resolve().parents[3]
SOURCE = "src/bayes_model_discovery/experimental/positive_inference.py"
BUDGET = "_LOG_PREDICTIVE_BLOCK_CELLS"
UNSPLIT = None
WIDTHS = (1, 7, 128, UNSPLIT)
REAL = ("normal", "student_t", "heteroscedastic_normal")
COUNT = ("poisson", "negative_binomial", "zi_poisson")
POSITIVE = ("lognormal", "gamma")
LOG_INTERFACE = {"outcome_type": "synthetic", "inputs": [{"name": "u", "range": [0, 1]}],
                 "derived_inputs": {"z": "2*u-1"}, "link": "log"}
EXPRESSIONS = ("b0+b1*z", "b0+b1*u", "b0+b1*log(1+u)", "b0+b1*u*u")
DATA_SHA = hashlib.sha256(b"synthetic public data").hexdigest()


# --------------------------------------------------------------------------
# the BASE_COMMIT module and synthetic pools
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def old_source():
    """Source text of `positive_inference.py` at BASE_COMMIT, read with `git show`; skips when unavailable."""
    shown = subprocess.run(["git", "-C", str(REPO), "show", f"{BASE_COMMIT}:{SOURCE}"],
                           capture_output=True, text=True, check=False)
    if shown.returncode:
        pytest.skip(f"{BASE_COMMIT} is not in this checkout's history: {shown.stderr.strip()}")
    return shown.stdout


@pytest.fixture(scope="module")
def old(old_source):
    """The module at BASE_COMMIT, executed inside this package: its relative imports reach the same
    unchanged neighbours (`count_inference`, `count_models`, `real_inference`) as `current`."""
    name = f"{__package__}.positive_inference_at_{BASE_COMMIT[:7]}"
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    module.__package__ = __package__
    exec(compile(old_source, f"{BASE_COMMIT[:7]}:{SOURCE}", "exec"), module.__dict__)  # noqa: S102
    assert not hasattr(module, BUDGET) and hasattr(current, BUDGET)
    return module


def _diagnostics(interface, **extra):
    return {"interface": interface, "response_unit": 1.7, "observation_count": 7,
            "public_data_sha256": DATA_SHA, "round": 3, **extra}


def real_fit(family, particles, seed, expression=EXPRESSIONS[0]):
    """Build a synthetic real-valued `Fit` of the given family with the given number of particles."""
    rng = np.random.default_rng(seed)
    hypothesis = RealHypothesis(expression, family, (("b0", "intercept"), ("b1", "signed")))
    draws = {"b0": rng.normal(0.2, 0.3, particles), "b1": rng.normal(-0.4, 0.3, particles)}
    if family == "heteroscedastic_normal":
        draws.update(a0=rng.normal(-1.0, 0.3, particles), a1=rng.normal(0.4, 0.2, particles))
    else:
        draws["sigma"] = rng.gamma(4.0, 0.1, particles)
    return Fit(hypothesis, draws, float(rng.normal()), [],
               _diagnostics(core_real_interface(), response_center=0.35))


NOISE = {"negative_binomial": ("alpha", lambda rng, n: rng.gamma(4.0, 1.0, n)),
         "zi_poisson": ("pi", lambda rng, n: rng.beta(2.0, 6.0, n)),
         "lognormal": ("sigma", lambda rng, n: rng.gamma(4.0, 0.1, n)),
         "gamma": ("shape", lambda rng, n: 1.5 + rng.gamma(4.0, 1.0, n))}


def scalar_fit(family, particles, seed, expression=EXPRESSIONS[0]):
    """Build a synthetic scalar-outcome `Fit` (count or positive family) with the given particle count."""
    rng = np.random.default_rng(seed)
    parameters = (("b0", "intercept"), ("b1", "signed"))
    hypothesis = current.ScalarHypothesis(expression, family, parameters)
    draws = {"b0": rng.normal(1.0, 0.2, particles), "b1": rng.normal(0.5, 0.2, particles)}
    if family in NOISE:
        name, draw = NOISE[family]
        draws[name] = draw(rng, particles)
    return Fit(hypothesis, draws, float(rng.normal()), [], _diagnostics(LOG_INTERFACE))


def catalogue_fit(family, particles, seed, features=(0, 2)):
    """Build a synthetic `Fit` over a regression-catalogue hypothesis with the given features."""
    rng = np.random.default_rng(seed)
    draws = {"b0": rng.normal(0.8, 0.2, particles)}
    draws.update({f"b{i + 1}": rng.normal(0.3, 0.1, particles) for i in range(len(features))})
    if family in NOISE:
        name, draw = NOISE[family]
        draws[name] = draw(rng, particles)
    diagnostics = {"input_transform": {"center": (0.5, 0.5, 0.5), "scale": (0.3, 0.3, 0.3)},
                   "response_unit": 2.0, "observation_count": 7, "public_data_sha256": DATA_SHA}
    return Fit(current.RegressionHypothesis(family, features), draws, float(rng.normal()), [],
               diagnostics)


def one_model(family, particles, seed=1):
    """Return a single synthetic fit for `family`, choosing the real or scalar builder."""
    build = real_fit if family in REAL else scalar_fit
    return build(family, particles, seed)


#: Multi-model pools of one measure each, with unequal particle counts on purpose.
POOLS = {
    "real": lambda: [real_fit("normal", 50, 1), real_fit("student_t", 37, 2, EXPRESSIONS[1]),
                     real_fit("heteroscedastic_normal", 64, 3, EXPRESSIONS[2])],
    "count": lambda: [scalar_fit("poisson", 50, 4), scalar_fit("negative_binomial", 41, 5),
                      scalar_fit("zi_poisson", 64, 6, EXPRESSIONS[1])],
    "positive": lambda: [scalar_fit("lognormal", 50, 7),
                         scalar_fit("gamma", 29, 8, EXPRESSIONS[3])],
    "catalogue": lambda: [catalogue_fit("lognormal", 50, 9),
                          catalogue_fit("gamma", 33, 10, (1,))],
}


def design(fits, rows, seed=0):
    """Return synthetic input points for the pool: a permuted grid, or uniform rows for a catalogue pool."""
    rng = np.random.default_rng(seed)
    if isinstance(fits[0].hypothesis, current.RegressionHypothesis):
        return rng.uniform(0.0, 1.0, (rows, 3))
    return rng.permutation(np.linspace(0.02, 0.98, rows))


def outcomes_for(fits, inputs, draws, edges, seed=5):
    """The pool's own predictive draws [input, draw]; `edges` adds support-edge cells (zero,
    non-integer counts, nonpositive positives, far outliers) so some blocks hold -inf cells."""
    y = current.predictive_summary(fits, inputs, seed, max(draws, 2))["samples"][:, :draws].copy()
    if edges:
        family = fits[0].hypothesis.family
        cells = [(i % y.shape[0], (7 * i + 3) % draws) for i in range(0, 3 * draws, 5)]
        for index, (row, column) in enumerate(cells):
            if family in REAL:
                y[row, column] = (-40.0, 40.0)[index % 2]
            elif family in COUNT:
                y[row, column] = (0.0, 2.5)[index % 2]
            else:
                y[row, column] = (0.0, -1.0)[index % 2]
    return y


def force_width(monkeypatch, fits, rows, width):
    """Set the cells budget that gives `width` draw columns per block; UNSPLIT: one block."""
    cells = 10**15 if width is UNSPLIT else width * rows * max(map(current.particle_count, fits))
    monkeypatch.setattr(current, BUDGET, cells)


@pytest.fixture
def seen(monkeypatch):
    """The outcome arrays the CURRENT module's `log_predictive` hands `conditional_loglik`."""
    calls = []
    original = current.conditional_loglik
    def spy(result, inputs, outcomes):
        """Record the outcome array of each call, then delegate to the original function."""
        calls.append(outcomes)
        return original(result, inputs, outcomes)
    monkeypatch.setattr(current, "conditional_loglik", spy)
    return calls


def assert_identical(before, after):
    """Same dtype, shape, memory order and bytes, `np.array_equal`, and no NaN."""
    assert len(before) == len(after)
    for left, right in zip(before, after, strict=True):
        assert isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
        assert left.dtype == right.dtype == np.float64 and left.shape == right.shape
        assert left.flags.c_contiguous and right.flags.c_contiguous
        assert not np.isnan(left).any() and not np.isnan(right).any()
        assert np.array_equal(left, right)
        assert left.tobytes() == right.tobytes()


def expected_widths(draws, width, rows):
    """Return the draw-block widths the chunked path should produce for the given width and row count."""
    if width is UNSPLIT or draws <= width:
        return [draws]
    if rows == 1:
        width = max(width, 2)
    widths = [width] * (draws // width) + ([draws % width] if draws % width else [])
    if rows == 1 and widths[-1] == 1:
        widths[-2:] = [widths[-2] + 1]
    return widths if len(widths) > 1 else [draws]


def assert_blocks(calls, fits, rows, draws, width):
    """Assert that the recorded outcome blocks have the expected row count and widths."""
    widths = [np.shape(outcomes)[1] for outcomes in calls]
    assert all(np.shape(outcomes)[0] == rows for outcomes in calls)
    assert widths == expected_widths(draws, width, rows) * len(fits)
    if rows == 1:
        assert 1 not in widths or draws == 1


# --------------------------------------------------------------------------
# the source change is exactly log_predictive
# --------------------------------------------------------------------------


def _top_level(source):
    items = []
    for node in ast.parse(source).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
        items.append((name, ast.get_source_segment(source, node)))
    return items


def test_only_log_predictive_changed_and_its_one_call_branch_is_the_old_body(old_source):
    """Check that only `log_predictive` (and `joint_eig`) changed and that the unsplit branch is the old body."""
    now = Path(current.__file__).read_text()
    before, after = _top_level(old_source), _top_level(now)
    added = [name for name, _ in after if name not in {n for n, _ in before}]
    assert added == [BUDGET, "_draw_blocks"]
    after = [item for item in after if item[0] not in added]
    assert [name for name, _ in after] == [name for name, _ in before]
    # joint_eig gained decompose=True later; its default call stays the old one byte for byte (test_joint_eig_638).
    changed = [n for (n, a), (_, b) in zip(before, after, strict=True) if a != b]
    assert changed == ["log_predictive", "joint_eig"]
    old_function, new_function = (
        next(node for node in ast.parse(text).body if getattr(node, "name", "") == "log_predictive")
        for text in (old_source, now))
    one_call = new_function.body[2]
    assert isinstance(one_call, ast.If) and ast.unparse(one_call.test) == "blocks is None"
    assert ast.dump(one_call.body[0]) == ast.dump(old_function.body[1])
    assert ast.dump(new_function.body[0]) == ast.dump(old_function.body[0])
    assert ast.dump(new_function.body[-1]) == ast.dump(old_function.body[-1])


def test_the_budget_is_in_no_record_or_frozen_header():
    """Check that the block budget constant appears in no record or frozen header."""
    assert not hasattr(truth_law, BUDGET)
    assert BUDGET not in frozen_kl_rescore.FROZEN_CONSTANT_NAMES
    assert all(BUDGET not in str(value) for value in frozen_kl_rescore.frozen_constants())


# --------------------------------------------------------------------------
# log_predictive: old against new
# --------------------------------------------------------------------------


@pytest.mark.parametrize("edges", (False, True), ids=("draws", "edges"))
@pytest.mark.parametrize("width", WIDTHS, ids=("w1", "w7", "w128", "unsplit"))
@pytest.mark.parametrize("family", (*REAL, *COUNT, *POSITIVE))
def test_every_family_and_width_is_byte_identical(old, monkeypatch, seen, family, width, edges):
    """Check old and new `log_predictive` agree byte for byte for every family and forced width."""
    fits = [one_model(family, 50)]
    inputs = design(fits, 17)
    outcomes = outcomes_for(fits, inputs, 33, edges)
    force_width(monkeypatch, fits, 17, width)
    expected = old.log_predictive(fits, inputs, outcomes)
    del seen[:]
    assert_identical(expected, current.log_predictive(fits, inputs, outcomes))
    assert_blocks(seen, fits, 17, 33, width)


@pytest.mark.parametrize("edges", (False, True), ids=("draws", "edges"))
@pytest.mark.parametrize("width", WIDTHS, ids=("w1", "w7", "w128", "unsplit"))
@pytest.mark.parametrize("kind", tuple(POOLS))
def test_pools_of_unequal_particle_counts_are_byte_identical(old, monkeypatch, seen, kind, width,
                                                            edges):
    """Check byte-identical results for multi-model pools with unequal particle counts."""
    fits = POOLS[kind]()
    assert len({current.particle_count(fit) for fit in fits}) == len(fits)
    inputs = design(fits, 17)
    outcomes = outcomes_for(fits, inputs, 33, edges)
    force_width(monkeypatch, fits, 17, width)
    expected = old.log_predictive(fits, inputs, outcomes)
    del seen[:]
    assert_identical(expected, current.log_predictive(fits, inputs, outcomes))
    assert_blocks(seen, fits, 17, 33, width)


@pytest.mark.parametrize("draws,width", [(33, 16), (33, 1), (33, 2), (34, 2), (3, 2), (33, 32),
                                         (2048, 7)])
@pytest.mark.parametrize("kind", ("real", "count", "positive"))
def test_rule_r2_one_input_never_forms_a_one_column_block(old, monkeypatch, seen, kind, draws,
                                                          width):
    """One input: a width below 2 is raised to 2 and a one-column tail joins the block before it,
    so every block's [particle, 1, width] reduction is the sequential one the whole array gets."""
    fits = [real_fit("normal", 1000, 21), real_fit("student_t", 700, 22, EXPRESSIONS[1])] \
        if kind == "real" else [one_model(family, 1000, 23 + i) for i, family in
                                enumerate(COUNT[:2] if kind == "count" else POSITIVE)]
    inputs = design(fits, 1)
    outcomes = outcomes_for(fits, inputs, draws, edges=False)
    force_width(monkeypatch, fits, 1, width)
    expected = old.log_predictive(fits, inputs, outcomes)
    del seen[:]
    assert_identical(expected, current.log_predictive(fits, inputs, outcomes))
    assert_blocks(seen, fits, 1, draws, width)


def test_rule_r2_is_load_bearing_on_this_numpy():
    """The trap the one-input block-width rule avoids: with one input, a one-column block is a single particle vector
    and numpy sums it pairwise, so some columns round differently from the whole array's."""
    fit = real_fit("normal", 1000, 31)
    inputs = design([fit], 1)
    loglik = current.conditional_loglik(fit, inputs, outcomes_for([fit], inputs, 64, False))
    whole = logsumexp(loglik, axis=0)
    columns = np.concatenate([logsumexp(np.ascontiguousarray(loglik[:, :, j:j + 1]), axis=0)
                              for j in range(64)], axis=1)
    pairs = np.concatenate([logsumexp(np.ascontiguousarray(loglik[:, :, j:j + 2]), axis=0)
                            for j in range(0, 64, 2)], axis=1)
    assert np.array_equal(pairs, whole)
    differing = int(np.sum(columns != whole))
    print(f"chunked log-predictive, one input, 1000 particles, 64 draws: {differing} of 64 one-column "
          f"reductions differ from the whole array's; two-column blocks: 0 differ")
    if not differing:
        pytest.skip("this numpy reduces a one-column block like the whole array: the one-column rule is inert")


def test_the_default_budget_splits_the_dev_truncation_shape(old, seen):
    """Dev truncation: 128 particles x 32 inputs x 2048 draws exceeds 8e6 cells: two blocks."""
    fits = [real_fit("normal", 128, 40 + i, expression) for i, expression in enumerate(EXPRESSIONS)]
    inputs = design(fits, 32)
    outcomes = outcomes_for(fits, inputs, 2048, edges=False)
    expected = old.log_predictive(fits, inputs, outcomes)
    del seen[:]
    assert_identical(expected, current.log_predictive(fits, inputs, outcomes))
    width = getattr(current, BUDGET) // (128 * 32)
    assert [np.shape(y)[1] for y in seen] == [width, 2048 - width] * len(fits)


@pytest.mark.parametrize("shape", ("list", "vector", "column", "one_block"))
def test_a_call_within_one_block_runs_the_unchanged_path(old, monkeypatch, seen, shape):
    """Check that a call fitting in one block runs the unchanged path and passes the outcomes through."""
    fits = POOLS["positive"]()
    inputs = design(fits, 5)
    y = outcomes_for(fits, inputs, 6, edges=False)
    outcomes = {"list": y[:, 0].tolist(), "vector": y[:, 0], "column": y[:, :1],
                "one_block": y}[shape]
    monkeypatch.setattr(current, BUDGET, 6 * 5 * 50)
    expected = old.log_predictive(fits, inputs, outcomes)
    del seen[:]
    assert_identical(expected, current.log_predictive(fits, inputs, outcomes))
    assert len(seen) == len(fits) and all(item is outcomes for item in seen)


@pytest.mark.parametrize("case", ("nan_in_a_late_block", "row_count", "posinf_density"))
def test_refusals_are_unchanged(old, monkeypatch, case):
    """Check that the error messages for invalid outcomes are identical in the old and new code."""
    fits = POOLS["real"]() if case != "posinf_density" else [scalar_fit("gamma", 50, 50)]
    inputs = design(fits, 4)
    outcomes = outcomes_for(fits, inputs, 20, edges=False)
    if case == "nan_in_a_late_block":
        outcomes[2, 19] = np.nan
    elif case == "row_count":
        outcomes = outcomes[:3]
    else:
        fits[0].particles["shape"][:] = 0.5  # gamma density at y = 0 is +inf below shape 1
        outcomes[1, 17] = 0.0
    force_width(monkeypatch, fits, 4, 3)
    with pytest.raises(ValueError) as before:
        old.log_predictive(fits, inputs, outcomes)
    with pytest.raises(ValueError) as after:
        current.log_predictive(fits, inputs, outcomes)
    assert str(before.value) == str(after.value)


# --------------------------------------------------------------------------
# callers, each driven old against new
# --------------------------------------------------------------------------


def truncation_pool(particles=128):
    """Return a pool of synthetic normal fits shaped like the dev truncation pool."""
    return [real_fit("normal", particles, 60 + i, expression)
            for i, expression in enumerate(EXPRESSIONS)]


@pytest.mark.parametrize("width", (1, 7, 128, "default"))
def test_gaussian_same_round_truncation_168_169(old, monkeypatch, seen, width):
    """Check the Gaussian same-round truncation diagnostic is byte-identical old against new."""
    before = truncation_pool()
    active = before[:3]
    inputs = np.linspace(0.02, 0.98, 32)
    with monkeypatch.context() as patch:
        patch.setattr(gaussian_core, "scalar", old)
        expected = gaussian_core.same_round_truncation(before, active, inputs, 3, 7)
    if width != "default":
        force_width(monkeypatch, before, 32, width)
    del seen[:]
    actual = gaussian_core.same_round_truncation(before, active, inputs, 3, 7)
    assert json.dumps(expected[0], sort_keys=True) == json.dumps(actual[0], sort_keys=True)
    assert expected[1].keys() == actual[1].keys()
    for key in expected[1]:
        assert_identical([np.ascontiguousarray(expected[1][key], dtype=float)],
                         [np.ascontiguousarray(actual[1][key], dtype=float)])
        assert expected[1][key].dtype == actual[1][key].dtype
    assert expected[0]["draws"] == 2048 and actual[1]["samples"].shape == (32, 2048)
    assert len(seen) > len(before) + len(active)  # the draw axis really was split


@pytest.mark.parametrize("width", (1, "default"))
def test_gaussian_checkpoint_scores_87_116_and_truth_law_369_371(old, monkeypatch, width):
    """Check Gaussian checkpoint scores and the truth-law KL are identical old against new."""
    fits = truncation_pool(64)
    inputs = np.linspace(0.02, 0.98, 32)
    rng = np.random.default_rng(70)
    truth = (0.35 + 0.1 * inputs, 0.05)
    heldout = truth[0][:, None] + truth[1] * rng.standard_normal((32, 4))
    monkeypatch.setattr(truth_law, "KL_PREDICTIVE_BLOCK_CELLS", 64 * 32 * 5)  # :371, blocked
    def run():
        """Run the Gaussian checkpoint diagnostic once with the fixed test inputs."""
        return gaussian_core.checkpoint(fits, inputs, heldout, truth, core_count.seed_for, 31,
                                        2, "after_proposal")
    with monkeypatch.context() as patch:
        patch.setattr(gaussian_core, "scalar", old)
        patch.setattr(truth_law, "log_predictive", old.log_predictive)
        expected = run()
    if width != "default":
        force_width(monkeypatch, fits, 32, width)
    actual = run()
    assert json.dumps(expected[0], sort_keys=True) == json.dumps(actual[0], sort_keys=True)
    assert expected[1].keys() == actual[1].keys()
    for key, value in expected[1].items():
        if isinstance(value, np.ndarray):
            assert value.dtype == actual[1][key].dtype
            assert value.tobytes() == actual[1][key].tobytes()
        else:
            assert value == actual[1][key]
    assert [m["means"].tobytes() for m in expected[2]] == [m["means"].tobytes() for m in actual[2]]
    monkeypatch.setattr(truth_law, "KL_PREDICTIVE_BLOCK_CELLS", 10**12)  # :369, one call
    law = truth_law.normal_law(*truth)
    with monkeypatch.context() as patch:
        patch.setattr(truth_law, "log_predictive", old.log_predictive)
        one_call = truth_law.kl_against_pool(law, fits, inputs)
    assert one_call == truth_law.kl_against_pool(law, fits, inputs)


def test_core_one_step_log_score_920(old, monkeypatch):
    """Check the CORE one-step log score is byte-identical old against new."""
    fit = real_fit("normal", 128, 80)
    force_width(monkeypatch, [fit], 1, 1)
    actual = [core_count.one_step_log_score(fit, u, y, core_real_interface())
              for u, y in ((0.1, 0.3), (0.7, 0.9), (0.95, -2.0))]
    with monkeypatch.context() as patch:
        patch.setattr(current, "log_predictive", old.log_predictive)
        expected = [core_count.one_step_log_score(fit, u, y, core_real_interface())
                    for u, y in ((0.1, 0.3), (0.7, 0.9), (0.95, -2.0))]
    assert np.asarray(expected).tobytes() == np.asarray(actual).tobytes()


@pytest.mark.parametrize("width", (1, 7, UNSPLIT))
@pytest.mark.parametrize("kind", ("real", "count", "positive"))
def test_joint_eig_638(old, monkeypatch, kind, width):
    """Check `joint_eig` results are byte-identical old against new at forced widths."""
    fits = POOLS[kind]()
    designs = design(fits, 9)
    force_width(monkeypatch, fits, 9, width)
    expected, actual = (module.joint_eig(fits, designs, 91, 64) for module in (old, current))
    assert expected.keys() == actual.keys()
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            assert value.tobytes() == actual[key].tobytes()
        else:
            assert value == actual[key]


def chem_fit(expression, coefficient, evidence, particles, seed):
    """Build a synthetic ChemBench-style lognormal `Fit` with the given mean coefficient and evidence."""
    interface = cc.task_interface("positive", [{"name": "u", "range": [0.0, 1.0]},
                                               {"name": "v", "range": [0.0, 1.0]}], {}, (0.0, None))
    rng = np.random.default_rng(seed)
    hypothesis = cc.ScalarHypothesis(expression, "lognormal", (("b0", "intercept"),), ("u", "v"))
    particles = {"b0": coefficient + rng.normal(0.0, 0.1, particles),
                 "sigma": rng.gamma(4.0, 0.1, particles)}
    return Fit(hypothesis, particles, evidence, [evidence],
               {"interface": interface, "response_unit": 1.0}, 0.0)


def counted_guarded(monkeypatch, function):
    """Route chembench_core's `log_predictive` through `function`, counting each call's line."""
    calls, pools = Counter(), Counter()
    def counted(fits, inputs, outcomes):
        """Count the call by caller line, then delegate to the wrapped function."""
        frame = sys._getframe(1)
        calls[f"{frame.f_code.co_name}:{frame.f_lineno}"] += 1
        return function(fits, inputs, outcomes)
    original_pools = cc._prediction_pools
    def recorded(*args):
        """Count prediction-pool calls and exclusions, then return the original result."""
        eligible, exclusions = original_pools(*args)
        pools["calls"] += 1
        pools["exclusions"] += len(exclusions)
        return eligible, exclusions
    monkeypatch.setattr(cc, "log_predictive", counted)
    monkeypatch.setattr(cc, "_prediction_pools", recorded)
    return calls, pools


@pytest.mark.parametrize("width", (1, 7, UNSPLIT))
@pytest.mark.parametrize("excluded", (False, True), ids=("no_exclusion_991", "exclusion_998"))
def test_chembench_guarded_log_predictive_991_998(old, monkeypatch, seen, excluded, width):
    """:991 (no model excluded: one pooled call) and :998 (a model with an underflowed mean at
    some inputs is dropped there: one call per input, each a one-row [1, draw] outcome).
    The line numbers are those of the base commit's tree; the comparison skips when the base commit
    is absent from the checkout."""
    good = chem_fit("b0", np.log(2.0), np.log(3.0), 64, 90)
    other = chem_fit("b0+u", 0.3, 0.5, 48, 91)
    fits = [good, other, chem_fit("b0-1000*u", 0.0, 0.0, 40, 92)] if excluded else [good, other]
    u = np.array([0.0, 0.1, 0.5, 0.7, 0.8, 1.0, 0.3])
    inputs = np.column_stack([u, np.linspace(0.9, 0.1, len(u))])
    outcomes = np.random.default_rng(93).lognormal(0.5, 0.4, (len(u), 33))
    rows = 1 if excluded else len(u)
    force_width(monkeypatch, fits, rows, width)
    results, logs = [], []
    for function in (old.log_predictive, current.log_predictive):
        with monkeypatch.context() as patch:
            calls, pools = counted_guarded(patch, function)
            del seen[:]
            results.append(cc.guarded_log_predictive(fits, inputs, outcomes))
            logs.append((dict(calls), dict(pools), len(seen)))
    assert_identical(*results)
    (old_calls, old_pools, _), (new_calls, new_pools, blocks) = logs
    assert old_calls == new_calls and old_pools == new_pools
    # A later change: chembench_core.py gained one import line (particle_means_masked), so the
    # guarded calls pinned at :991/:998 in the base commit sat at :992/:999; later commits added four
    # import lines above them, then one more: :997/:1004 here; the code is unchanged.
    line = "guarded_log_predictive:1004" if excluded else "guarded_log_predictive:997"
    assert old_calls == {line: len(u) if excluded else 1}
    assert old_pools == {"calls": 1, "exclusions": 2 if excluded else 0}
    print(f"chunked log-predictive, ChemBench guarded path: excluded={excluded} width={width} calls={new_calls} "
          f"prediction_pools={new_pools} conditional_loglik_blocks={blocks} old==new bytes")


# --------------------------------------------------------------------------
# integration parity (opt-in)
# --------------------------------------------------------------------------

#: Forced budget of the narrow candidate run: 7 draw columns per block at the dev truncation
#: shape (7 x 128 particles x 32 inputs), and one column per block at ChemBench's official set.
NARROW_CELLS = 7 * 128 * 32

CALL_LOGGER = r'''
import json, resource, sys, time
from collections import Counter
from pathlib import Path
from pytest import MonkeyPatch
from bayes_model_discovery.experimental import boxinggym_truth_law, chembench_core
from bayes_model_discovery.experimental import positive_inference as engine

start, calls, split = time.perf_counter(), Counter(), Counter()
def install(patch, budget):
    """Count every `log_predictive` call by caller, and (current tree) every draw split."""
    if budget != "default":
        patch.setattr(engine, "_LOG_PREDICTIVE_BLOCK_CELLS", int(budget))
    original = engine.log_predictive
    def counted(fits, inputs, outcomes):
        frame = sys._getframe(1)
        name = Path(frame.f_code.co_filename).name
        line = "" if name == "positive_inference.py" else f":{frame.f_lineno}"
        calls[f"{name}:{frame.f_code.co_name}{line}"] += 1
        return original(fits, inputs, outcomes)
    for module in (engine, boxinggym_truth_law, chembench_core):
        patch.setattr(module, "log_predictive", counted)
    if hasattr(engine, "_draw_blocks"):
        blocks_for = engine._draw_blocks
        def recorded(fits, outcomes):
            blocks = blocks_for(fits, outcomes)
            split["split_calls" if blocks else "one_call"] += 1
            split["blocks"] += len(blocks) if blocks else 0
            return blocks
        patch.setattr(engine, "_draw_blocks", recorded)
    pools = chembench_core._prediction_pools
    def excluded(*args):
        eligible, exclusions = pools(*args)
        split["prediction_pools_calls"] += 1
        split["prediction_pools_exclusions"] += len(exclusions)
        return eligible, exclusions
    patch.setattr(chembench_core, "_prediction_pools", excluded)
def report(path):
    Path(path).write_text(json.dumps({
        "log_predictive_calls": dict(sorted(calls.items())),
        "draw_blocks": dict(sorted(split.items())),
        "maxrss_self_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "maxrss_children_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "wall_seconds": time.perf_counter() - start}, indent=1, sort_keys=True))
'''

CORE_HARNESS = CALL_LOGGER + r'''
from bayes_model_discovery.experimental import core_count
output, paths, budget = sys.argv[1:4]
with MonkeyPatch.context() as patch:
    install(patch, budget)
    sys.argv = ["core_count", "--output", output, "--profile", "dev", "--rounds", "3",
                "--cell", "100", "--proposer", "mock", "--seed", "31",
                "--structure", "gauss_den", "--truth-family", "normal", "--active-maximum", "3"]
    core_count.main()
report(paths)
'''

CHEMBENCH_HARNESS = CALL_LOGGER + r'''
from bayes_model_discovery.experimental import test_chembench_core as fixture
output, manifest, paths, budget = sys.argv[1:5]
with MonkeyPatch.context() as patch:
    install(patch, budget)
    patch.setattr(fixture.real_adapter, "ChemBenchSession", fixture._FakeSession)
    patch.setattr(fixture.real_adapter, "sample_designs", fixture._fake_sample_designs)
    patch.setattr(chembench_core, "ACTIVE_MAXIMUM", 3)
    chembench_core.run(fixture._args(Path(output), Path(manifest), rounds=4, particles=None, q=None,
                                     cell="110", structure_proposals="free", proposer="mock"))
report(paths)
'''

BOXINGGYM_HARNESS = CALL_LOGGER + r'''
from bayes_model_discovery.experimental import boxinggym_core
from bayes_model_discovery.experimental.test_boxinggym_kl_all_tasks import run_args
output, paths, budget = sys.argv[1:4]
with MonkeyPatch.context() as patch:
    install(patch, budget)
    patch.setattr(boxinggym_core, "log_predictive", engine.log_predictive)  # the counter
    boxinggym_core.run(run_args(output, "dugongs", rounds=2))
report(paths)
'''

#: Operational fields, never compared: whole files, and fields removed by name
#: from otherwise-science JSON. Source digests are the source commit's files; `created_utc`,
#: `*_seconds` and `_t_sampling` are wall time; `output`/`source_path`/`python` are machine paths.
OPERATIONAL_FILES = ("acquisition_timings.jsonl", "fit_timings.jsonl")
OPERATIONAL_FIELDS = {
    "config.json": ("source_files_sha256", "source_sha256", "created_utc", "output", "python",
                    "manifest_ref", "source_root", "adapter_source"),
    "summary.json": ("wall_seconds", "user_cpu_seconds", "system_cpu_seconds", "timing_seconds",
                     "initial_pool_fit.seconds", "rounds_fit.seconds", "timings"),
    "fits/*.json": ("fit_seconds", "diagnostics.sampler_attrs._t_sampling"),
}
OPERATIONAL_COLUMNS = {"points.csv": ("code_sha", "source_path")}
#: Science files, asserted as raw bytes (and .npz also array by array).
SCIENCE = {
    "core": ("truncation/*.json", "truncation/*.npz", "scores.jsonl", "archive.json",
             "predictions/*.npz", "observations.jsonl", "fits/*.npz", "evaluation_scoring_only.npz",
             "frozen_d0_policy.json", "prediction_pool_summary.jsonl", "acquisition/*.json",
             "proposals/*.json", "packets/*.json"),
    "chembench": ("proposals/*.json", "scores_after_finish.json",
                  "official_scores_after_finish.json", "pool_weights.jsonl",
                  "prediction_pool_summary.jsonl", "predictions/*.npz", "predictions/*.json",
                  "final_raw_predictions.npz", "official_kl_particle_summary_after_finish.npz",
                  "reference_after_finish.npz", "official_reference_after_finish.npz",
                  "evaluation_inputs.npz", "official_evaluation_inputs.npz",
                  "observations.jsonl", "candidate_failures.jsonl", "fits/*.npz",
                  "frozen_d0_policy.json", "acquisition/*.json", "packets/*.json"),
    "boxinggym": ("scores.jsonl", "archive.json", "predictions/*.npz", "observations.jsonl",
                  "fits/*.npz", "evaluation_scoring_only.npz", "frozen_d0_policy.json",
                  "prediction_pool_summary.jsonl", "acquisition/*.json", "proposals/*.json",
                  "packets/*.json", "points_mda.csv"),
}


def _baseline():
    configured = os.environ.get("BMD_TRM_BASELINE_DIR")
    if os.environ.get("BMD_BASELINE_COMPARE") != "1" or not configured:
        pytest.skip("set BMD_BASELINE_COMPARE=1 and BMD_TRM_BASELINE_DIR (a clean checkout of the parity baseline commit)")
    baseline = Path(configured)
    def git(*args):
        """Run a git command in the baseline checkout and return its stdout."""
        return subprocess.run(["git", "-C", str(baseline), *args], capture_output=True,
                              text=True, check=True).stdout
    assert git("rev-parse", "HEAD").strip() == PARITY_BASE_COMMIT
    assert git("status", "--short") == ""
    return baseline


def _run(tmp_path, label, root, harness, *argv, extra_path=""):
    env = dict(os.environ, PYTHONPATH=f"{root}/src:{root}{extra_path}", OPENAI_API_KEY="",
               ANTHROPIC_API_KEY="", OPENAI_COMPAT_API_KEY="", OPENROUTER_API_KEY="",
               PYTHON_DOTENV_DISABLED="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, "-B", "-c", harness, *argv], cwd=tmp_path, env=env,
                               capture_output=True, text=True, timeout=1800, check=False)
    (tmp_path / f"{label}.log").write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-6000:]


def _drop(record, dotted):
    head, _, rest = dotted.partition(".")
    if rest:
        if isinstance(record.get(head), dict):
            _drop(record[head], rest)
    else:
        record.pop(head, None)


def _classify(name, bundle):
    if name in OPERATIONAL_FILES:
        return "operational"
    if name in OPERATIONAL_COLUMNS:
        return "columns"
    if any(fnmatch.fnmatch(name, pattern) for pattern in OPERATIONAL_FIELDS):
        return "fields"
    if any(fnmatch.fnmatch(name, pattern) for pattern in SCIENCE[bundle]):
        return "science"
    return "unclassified"


def _later_unit_record(name, record, out, bundle):
    """One current run's packet or proposal record without the named
    later-commit keys; the packet digest pair is checked against that run's own packet file first."""
    if name.startswith("packets/"):
        for key in LATER_UNIT_PACKET_CONSTRAINT_KEYS.get(bundle, ()):
            assert key in record["constraints"], (name, key)
            record["constraints"].pop(key)
    elif isinstance(record.get("transport"), dict) and "packet_sha256" in record["transport"]:
        packet = (out / "packets" / Path(name).name).read_bytes()
        pair = tuple(record["transport"].pop(key) for key in PACKET_DIGEST_KEYS)
        assert pair == (hashlib.sha256(packet).hexdigest(), len(packet)), name
    return record


def _base_calls(log):
    """The base run's call log, read through LATER_UNIT_CALL_SITES."""
    return {LATER_UNIT_CALL_SITES.get(site, site): count for site, count in log.items()}


def _compare(outputs, bundle):
    """Assert science bytes; return the per-class file lists and the changed source digests."""
    names = [sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
             for out in outputs]
    for later in LATER_UNIT_FILES.get(bundle, ()):
        assert later not in names[0], later
        for files in names[1:]:
            assert later in files, later
            files.remove(later)
    assert all(files == names[0] for files in names), "file sets differ"
    classes = {name: _classify(name, bundle) for name in names[0]}
    assert [n for n, kind in classes.items() if kind == "unclassified"] == []
    changed = set()
    for name, kind in classes.items():
        paths = [out / name for out in outputs]
        if kind == "science" and bundle in LATER_UNIT_PACKET_CONSTRAINT_KEYS and (
                name.startswith(("packets/", "proposals/"))):
            # Weakened from raw bytes for this bundle's packets and proposals alone:
            # compared as parsed JSON with the named later-commit keys removed.
            base = json.loads(paths[0].read_bytes())
            for key in LATER_UNIT_PACKET_CONSTRAINT_KEYS[bundle]:
                assert key not in base.get("constraints", {}), (name, key)
            _later_unit_record(name, base, outputs[0], None)
            for path, out in zip(paths[1:], outputs[1:], strict=True):
                assert _later_unit_record(name, json.loads(path.read_bytes()), out, bundle) == base, name
        elif kind == "science":
            assert all(p.read_bytes() == paths[0].read_bytes() for p in paths[1:]), name
            if name.endswith(".npz"):
                arrays = [np.load(p) for p in paths]
                for other in arrays[1:]:
                    assert other.files == arrays[0].files, name
                    for key in arrays[0].files:
                        assert other[key].dtype == arrays[0][key].dtype, (name, key)
                        assert other[key].tobytes() == arrays[0][key].tobytes(), (name, key)
        elif kind == "fields":
            records = [json.loads(p.read_bytes()) for p in paths]
            pattern = next(p for p in OPERATIONAL_FIELDS if fnmatch.fnmatch(name, p))
            if name == "config.json":
                key = "source_files_sha256"
                key = key if key in records[0] else "source_sha256"
                for record in records[1:]:
                    assert record[key].keys() == records[0][key].keys()
                    changed |= {f for f in record[key] if record[key][f] != records[0][key][f]}
                for later in LATER_UNIT_CONFIG_KEYS.get(bundle, ()):
                    assert later not in records[0], later
                    for record in records[1:]:
                        record.pop(later)
            for record in records:
                for field in OPERATIONAL_FIELDS[pattern]:
                    _drop(record, field)
            assert all(record == records[0] for record in records[1:]), name
        elif kind == "columns":
            tables = []
            for path in paths:
                with path.open(newline="") as handle:
                    tables.append([{k: v for k, v in row.items()
                                    if k not in OPERATIONAL_COLUMNS[name]}
                                   for row in csv.DictReader(handle)])
            assert tables[0] and all(table == tables[0] for table in tables[1:]), name
    by_class = {kind: sorted(n for n, k in classes.items() if k == kind)
                for kind in set(classes.values())}
    return by_class, changed


def _call_logs(tmp_path, labels):
    return {label: json.loads((tmp_path / f"{label}.paths.json").read_text()) for label in labels}


def test_opt_in_core_cell_with_truncation_keeps_every_science_byte(tmp_path):
    """Opt-in: a mock CORE cell with truncation keeps every science file byte-identical to the baseline."""
    baseline = _baseline()
    runs = (("base", baseline, "default"), ("current", REPO, "default"),
            ("narrow", REPO, str(NARROW_CELLS)))
    for label, root, budget in runs:
        _run(tmp_path, label, root, CORE_HARNESS, str(tmp_path / label / "cell"),
             str(tmp_path / f"{label}.paths.json"), budget)
    outputs = [tmp_path / label / "cell" for label, _, _ in runs]
    assert sorted(p.name for p in (outputs[0] / "truncation").iterdir()) == [
        "t01.json", "t01.npz", "t03.json", "t03.npz"]
    classes, changed = _compare(outputs, "core")
    assert changed == set(LATER_UNIT_SOURCES["core"])
    logs = _call_logs(tmp_path, [label for label, _, _ in runs])
    assert _base_calls(logs["base"]["log_predictive_calls"]) == logs["current"]["log_predictive_calls"] \
        == logs["narrow"]["log_predictive_calls"]
    assert logs["current"]["draw_blocks"]["split_calls"] >= 4  # the truncation, t01 and t03
    assert logs["narrow"]["draw_blocks"]["blocks"] > logs["current"]["draw_blocks"]["blocks"]
    print("chunked log-predictive, CORE parity:", json.dumps({"classes": classes, "changed_sources": sorted(changed),
                                          "calls": logs["base"]["log_predictive_calls"],
                                          "blocks": {k: v["draw_blocks"] for k, v in logs.items()}},
                                         sort_keys=True))


def test_opt_in_chembench_cell_shared_log_predictive_path_and_full_artifact_parity(tmp_path):
    """ChemBench has no truncation diagnostic: this is the shared `log_predictive` path and full
    artifact parity, with pool cap 3 so retention runs and the pool changes across rounds."""
    baseline = _baseline()
    from .test_chembench_core import _manifest

    manifest = _manifest(tmp_path)
    runs = (("base", baseline, "default"), ("current", REPO, "default"),
            ("narrow", REPO, str(NARROW_CELLS)))
    for label, root, budget in runs:
        _run(tmp_path, label, root, CHEMBENCH_HARNESS, str(tmp_path / label / "cell"),
             str(manifest), str(tmp_path / f"{label}.paths.json"), budget)
    outputs = [tmp_path / label / "cell" for label, _, _ in runs]
    for out in outputs:
        assert not (out / "truncation").exists() and not (out / "archive.json").exists()
    assert len({(out / "candidate_failures.jsonl").exists() for out in outputs}) == 1
    classes, changed = _compare(outputs, "chembench")
    assert changed == set(LATER_UNIT_SOURCES["chembench"])
    retired = [json.loads((outputs[0] / "proposals" / f"t{t:02d}.json").read_text())["retired_ids"]
               for t in (1, 2, 3, 4)]
    assert any(retired), "pool cap 3 never retired a model"
    logs = _call_logs(tmp_path, [label for label, _, _ in runs])
    assert _base_calls(logs["base"]["log_predictive_calls"]) == logs["current"]["log_predictive_calls"] \
        == logs["narrow"]["log_predictive_calls"]
    assert logs["current"]["log_predictive_calls"].get("chembench_core.py:guarded_log_predictive:997")
    assert logs["narrow"]["draw_blocks"]["split_calls"] >= 1
    print("chunked log-predictive, ChemBench parity:", json.dumps({
        "classes": classes, "changed_sources": sorted(changed), "retired_ids": retired,
        "candidate_failures_exists": (outputs[0] / "candidate_failures.jsonl").exists(),
        "calls": logs["base"]["log_predictive_calls"],
        "blocks": {k: v["draw_blocks"] for k, v in logs.items()}}, sort_keys=True))


def test_opt_in_boxinggym_cell_keeps_every_science_byte(tmp_path):
    """BoxingGym's two call sites and its joint EIG, through its runner: dugongs has a positive
    outcome (no exhaustive PMF), so `our_scores` calls :3400 and `one_step_score` :3687 in the base (:3407 and :3694 here), and
    cell 111's acquisition calls `joint_eig`. BoxingGym has no truncation diagnostic. The line numbers
    are those of the base commit's tree and of this one; the comparison skips when the base commit
    is absent from the checkout."""
    baseline = _baseline()
    source = os.environ.get("BMD_BOXINGGYM_SRC")
    if not source:
        pytest.skip("set BMD_BOXINGGYM_SRC (the boxing_gym checkout's src) for BoxingGym parity")
    runs = (("base", baseline, "default"), ("current", REPO, "default"),
            ("narrow", REPO, str(NARROW_CELLS)))
    for label, root, budget in runs:
        _run(tmp_path, label, root, BOXINGGYM_HARNESS, str(tmp_path / label / "cell"),
             str(tmp_path / f"{label}.paths.json"), budget, extra_path=f":{source}")
    outputs = [tmp_path / label / "cell" for label, _, _ in runs]
    assert not (outputs[0] / "truncation").exists()
    classes, changed = _compare(outputs, "boxinggym")
    assert changed == set(LATER_UNIT_SOURCES["boxinggym"])
    logs = _call_logs(tmp_path, [label for label, _, _ in runs])
    calls = _base_calls(logs["base"]["log_predictive_calls"])
    assert all(log["log_predictive_calls"] == calls for label, log in logs.items() if label != "base")
    for path in ("boxinggym_core.py:our_scores:3407", "boxinggym_core.py:one_step_score:3694",
                 "positive_inference.py:joint_eig"):
        assert calls.get(path), path
    assert logs["narrow"]["draw_blocks"]["split_calls"] >= 1
    print("chunked log-predictive, BoxingGym parity:", json.dumps({
        "classes": classes, "changed_sources": sorted(changed), "calls": calls,
        "blocks": {k: v["draw_blocks"] for k, v in logs.items()}}, sort_keys=True))
