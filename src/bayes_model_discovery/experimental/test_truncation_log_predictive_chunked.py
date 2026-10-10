"""Chunked `log_predictive`: `positive_inference.log_predictive` split along the draw axis, byte for byte.

Synthetic pools only; no run output, seed or response of any real cell is read.

* Unit: the block budget is in no record or frozen header, and on this numpy a one-column block of a
  one-input layout sums differently from the whole array (the reason a block is never narrower than two
  columns there).
* Integration (opt-in: `BMD_BASELINE_COMPARE=1` and `BMD_TRM_BASELINE_DIR`, a clean checkout of
  PARITY_BASE_COMMIT; BoxingGym also `BMD_BOXINGGYM_SRC`): a mock CORE cell and a mock ChemBench cell, each
  with pool cap 3, and a mock BoxingGym dugongs cell run in that checkout and in this one, the
  latter at the default budget and at a forced narrow one. Science files are compared as bytes;
  the operational fields named below are excluded by name. Each run logs its calls by caller.
"""

import csv
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from . import boxinggym_truth_law as truth_law
from . import frozen_kl_rescore
from . import positive_inference as current
from .belief_packet import core_real_interface
from .count_inference import Fit
from .real_inference import RealHypothesis

#: The three opt-in integration tests compare against a clean checkout of PARITY_BASE_COMMIT (the commit that
#: introduced the chunked predictive) and skip when none is supplied, with the sources and record keys changed
#: since then named below.
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
# synthetic pools
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# the source change is exactly log_predictive
# --------------------------------------------------------------------------


def test_the_budget_is_in_no_record_or_frozen_header():
    """Check that the block budget constant appears in no record or frozen header."""
    assert not hasattr(truth_law, BUDGET)
    assert BUDGET not in frozen_kl_rescore.FROZEN_CONSTANT_NAMES
    assert all(BUDGET not in str(value) for value in frozen_kl_rescore.frozen_constants())


# --------------------------------------------------------------------------
# log_predictive: the one-input block rule
# --------------------------------------------------------------------------


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
