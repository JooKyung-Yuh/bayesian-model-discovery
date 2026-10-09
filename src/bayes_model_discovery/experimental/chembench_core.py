"""Fixed-pool first ChemBench path through the separately owned adapter.

`--structure-proposals catalog|free` (an
optional override of what cell[0] already selects, default "take the cell" -
see the F0 section at the end): `catalog`
is this module exactly as it was before this option existed - the fixed
4-model `RegressionHypothesis` catalogue from `initial_catalogue`, no
proposer wiring, byte-identical output. `free` keeps that catalogue as the
initial pool and additionally proposes one free-form
`positive_inference.ScalarHypothesis` grammar candidate per round (P1
policy), wired exactly like `boxinggym_core.py`'s own LLM path: `--proposer
{mock,anthropic,openai}` (default mock; `anthropic`/`openai` raise before
`--output` is created unless `--allow-paid-api`, the same CORE rule,
checked unconditionally regardless of `--structure-proposals`), a belief
packet v2 built from `schema["input_names"]`, `allowed_families` the
`positive_inference.SCALAR_FAMILIES` subset that fits a positive rate
outcome (`lognormal`, `gamma` - the same two `--family` already chooses
between), `count_models.judge_candidate` grammar/role/cell rules, a prior
finiteness probe, then a sequential Monte Carlo (SMC) fit and admission into the SAME pool the
catalogue lives in (`positive_inference` already dispatches a pool mixing
`RegressionHypothesis` and `ScalarHypothesis` fits via `is_grammar`, so
nothing downstream of this module needs to change). See `run`'s `if free:`
branch.

This runner is wired onto the paper matrix and the
plot contract. Nothing below changes what a `--structure-proposals catalog`
run computes; every new flag defaults to the original behaviour.

- Task axis: `--task <id>` selects the manifest row (the original
  code read `rows[0]` unconditionally, so a nine-row manifest would have run
  its first row nine times); `manifests/chembench_paper_manifest.json` carries
  the nine no-leak-baseline task ids verbatim.
- Method axis: `--cell` (originally 000|001|110|111; now all eight,
  see the F0 section at the end of this docstring). The RECORDED cell is the
  EXECUTED cell everywhere (config, summary, packet/proposal context,
  points.csv) - the original code recorded the literal `"000"` in config and
  summary while a free run actually executed the hardcoded `FREE_CELL="111"`.
  The third bit was inert in stage 1; the acquisition axis below wires it, and
  `"design_axis"` now names the policy the run actually executed.
- Profile: `--profile dev|paper` derives `(particles, q, design_pool,
  eig_samples)` from one table (`PROFILE_GRID`, the same one-line-tuple shape
  as `boxinggym_core.run`, boxinggym_core.py:1946-1947) and the points
  contract's `label`; `--particles`/`--q` stay as explicit overrides. Default
  is `dev` (by rule, a paid run must say `--profile paper` explicitly).
- Official evaluation set: `sample_designs(1000, 42)`, the sampler
  and seed the adapter's own docstring names as the reproduction of the native
  `evaluate_law` input sampler - the same generator and seed the no-leak
  baseline's offline scoring used. Scored ONLY after `finish()`.
- Primary metric `nmse_mda` on that set, replicating the baseline's
  own definition (see `official_checkpoint_scores`), plus `nmsle` and `rmsle`.
- Free-mode pool retirement: the CORE rule, evidence-top
  `core_count.ACTIVE_MAXIMUM` = 12 with ties broken by the stable `model_id`
  (`count_inference.retain`), retired members archived as
  `retired_by_evidence` and never re-proposable.
- `points.csv`: the plotting contract's columns plus the two
  declared auxiliary columns (`write_points`).
- Round cap: `count_proposer.propose_one`'s round window is this
  run's own `--rounds` (`max_rounds=`), so an LLM-path run reaches round 40.
  The mock schedule stays a FIXED 20-entry list and refuses round 21 at entry
  rather than silently cycling on (`MOCK_SCHEDULE_ROUNDS`).
- `autoscilab` is never installed; it is on PYTHONPATH only, its
  resolved path is recorded, and the oracle import is probed once at the very
  start of `run` so a PYTHONPATH mistake fails before any output exists.

STAGE 2 wires the acquisition axis -
the third cell bit - onto this runner, so `001`/`111` no longer execute what
`000`/`110` execute:

- the shared `design_policies` module supplies the policy NAMES, the A=0
  refusal (`check_policy`), the recorded `target` string (`TARGET`) and the
  points-table `design_*` token (`variant_token`), unchanged and unedited; the
  registered argmax + tie rule is the ONE authority `eig.joint.choose_design`
  (`TIE_ATOL` 1e-12, smallest design value wins).
- the expected information gain (EIG) itself is `positive_inference.joint_eig`, the
  estimator this outcome path already owns, called exactly as `boxinggym_core.run`'s own non-
  exhaustive acquisition branch calls it (boxinggym_core.py:2904-2909): one
  table per round, `{"design": [floats], "eig": float}` rows, then
  `choose_design`. `design_policies.select` is NOT used: it is the bounded-
  count bank (`count_inference.choose`, `count_models.particle_logpmf`, 51
  exhaustive outcomes), which has no meaning for a positive continuous rate -
  hence `CHEM_DESIGN_POLICIES`.
- the candidate set is the adapter's own `sample_designs` grid at the
  profile's pre-declared `design_pool` size (dev 32 / paper 256) drawn from
  the pre-declared `DESIGN_POOL_SEED`, verified DISJOINT from the official
  evaluation set at run time, and an already-observed pool design is removed
  before the argmax (the adapter's native noise is deterministic per design,
  so a repeat carries no new information - the same candidate-set rule
  `boxinggym_core`'s `design_exclusion="observed"` applies).
- an A=0 cell (`000`/`110`) keeps the pre-drawn `chem_observation_designs`
  sequence untouched, which is what makes it the byte-identical anchor.

The F0 factorial separates the first two
cell bits, which had moved together until then, so this runner reads a cell the
way `boxinggym_core.run` and `core_count.run` already read one. Nothing a
`000`/`001`/`110`/`111` run computes changes; the only new behaviour is on the
four cells that could not be expressed before.

- `CELL_CHOICES` is all eight cells. `structure_mode(cell)` is the single
  authority on the catalog/free question and reads cell[0] alone;
  `--structure-proposals` stayed as an OPTIONAL override that must agree with
  it (`validate_cell`), so every earlier command line, the two byte-identity
  anchors and the smoke driver of that time keep working
  unedited, `config.json` keeps the key, and a stated mode that contradicts the
  cell is still refused before any output exists. `run` overwrites
  `args.structure_proposals` with the derived mode, so what is RECORDED is what
  EXECUTED - the rule already set for the cell, now applied to the mode too.
- cell[1] is the family menu, and it was already wired everywhere it is read:
  `belief_packet.cell_constraints` states `allowed_families=[--family]` and
  `family_expansion=false` on an F0 cell (belief_packet.py:286,297,299), and
  `count_models.judge_candidate` refuses an out-of-family LLM candidate one at a
  time (count_models.py:1343-1345), recording it in the transport record's
  `rejected_candidates`. What the F0 factorial adds is `family_refusal` in this runner's
  own candidate loop, because the MOCK path builds its candidate straight from
  `_mock_schedule` - whose families deliberately cycle through the task menu -
  and so reaches no judge at all. Same rule, same wording
  (`count_models.F0_REFUSAL_REASON`), per candidate, archived as `rejected`.
- `100`/`101` are therefore the F0 structure-expansion route: the proposer runs,
  a lognormal candidate is admitted and fitted, a gamma one is refused alone and
  recorded, and `summary.json`'s `paid_api_calls` is the proposer's own
  `attempts` (0 on the mock path). `010`/`011` remain equivalent to `000`/`001`
  on this runner, and say so: cell[0] = 0 builds no proposer, so there is no
  candidate for an expanded family menu to act on - the same kind of declared
  equivalence `count_models.OUTCOME_MENUS` creates for the single-family binary
  tasks, where `000` == `010`.
"""

import argparse
import hashlib
import importlib.metadata
import json
import resource
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC
from functools import partial
from pathlib import Path

import numpy as np
from scipy.stats import norm

from ..eig.joint import choose_design
from .belief_packet import (
    PoolMember,
    build_belief_packet,
    cell_constraints,
    interface_grid,
    packet_digest,
    serialize_packet,
    task_interface,
)
from .core_count import (
    ACTIVE_MAXIMUM,
    BURST_BACKOFF_SECONDS,
    BURST_RETRIES,
    CONTRACT_POINTS_COLUMNS,
    DEFAULT_BURST_CONCURRENCY,
    INITIAL_BURST_WAVES,
    RUN_LABEL_CHOICES,
    append_json,
    burst_refill_pool,
    burst_wave_record,
    failure_class_counts,
    mock_parameters,
    # Re-export, deliberately kept after this module's own
    # label decision moved behind `resolve_run_label`: a test
    # (test_22a3_points_label_is_the_shared_helper_not_a_reimplementation)
    # pins `chembench_core.points_label` to be the very function object
    # `core_count` defines, so this binding is part of the module's contract
    # even though the call now happens one level down inside the resolver.
    points_label,  # noqa: F401
    seed_for,
    wave_event_counts,
    write_final_proposal_json,
    write_json,
    write_proposed_json,
)
from .count_inference import (
    EPIG_TARGET_POINTS,
    EPIG_TARGETS_SEED_STREAM,
    CountInferenceFailure,
    EpigDiscretePathState,
    FitTimeout,
    SMCSettings,
    candidate_set_record,
    choose_epig,
    epig_acquisition_record,
    epig_config,
    loglik_prior_probe,
    model_logweights,
    record_acquisition_timing,
    record_fit_timing,
    record_prediction_pool_summary,
    resolve_epig_policy,
    retain,
    run_with_fit_guards,
    run_with_fit_timeout,
    validate_fit_timeout,
)
from .count_models import (
    DEFAULT_GRAMMAR_VERSION,
    DEFAULT_PRIOR_TEMPLATE,
    DEFAULT_VALIDATION_MESSAGES,
    # The ONE wording of the F0 refusal, shared with
    # `count_models.judge_candidate` (count_models.py:1344-1345) so this runner's
    # mock path - which never reaches that judge - refuses in the same words the
    # LLM path's own judgement uses. No ChemBench-specific exception text.
    F0_REFUSAL_REASON,
    FAILURE_CLASSES,
    GRAMMAR_VERSIONS,
    INITIAL_BURST_REFILL_POLICY,
    PRIOR_RULES,
    PRIOR_TEMPLATE_NAMES,
    PRIOR_VARIANT_TOKEN,
    VALIDATION_MESSAGE_VERSIONS,
    failure_class,
    interface_input_names,
    named_input_values,
    prior_template_table,
    set_grammar_version,
    set_prior_template,
    set_validation_messages,
)
from .count_proposer import (
    PAID_CALL_BUDGET_EXIT_CODE,
    RESPONSE_FORMATS,
    PaidCallBudgetExhausted,
    budget_exhausted_summary,
    burst_refill_identity,
    compat_config,
    compat_option_kwargs,
    compat_points_metadata,
    composed_prompt_identity,
    composed_system_prompt,
    mask_secrets,
    normalize_provider_order,
    parse_max_paid_calls,
    parse_max_tokens,
    parse_reasoning_spec,
    prompt_identity,
    require_registered_decoding,
    select_run_prompt,
    set_prompt_variant,
    validate_proposal_limit,
)
from .design_policies import (
    DEFAULT_POLICY,
    POLICIES,
    SPACE_FILLING_DEFINITION,
    TARGET,
    check_policy,
    select_space_filling,
    variant_token,
)
from .model_identity import MODEL_ID_SCHEMA, NOTATION_NORMAL_FORM
from .positive_inference import (
    COUNT_SCALAR_FAMILIES,
    POINT_DISAGREEMENT_DEFINITION,
    SCALAR_FAMILIES,
    InputTransform,
    ScalarHypothesis,
    build_model,
    fit,
    grammar_means,
    initial_catalogue,
    is_grammar,
    joint_eig,
    log_predictive,
    mixture_moments,
    particle_count,
    particle_means,
    particle_means_masked,
    point_disagreement,
    predictive_means,
    predictive_summary,
    validate_pool,
)
from .prior_probe_detail import (
    PriorProbeRejection,
    first_nonfinite_subexpression,
    nonfinite_probe_sentence,
)
from .run_label_policy import bmd_label_receipt, recheck_manifest

# The SCALAR_FAMILIES subset that fits a positive (nonnegative continuous)
# rate outcome, per the paper's task table's "Positive outcome" row - the same
# two continuous families --family already chooses between; the other three
# SCALAR_FAMILIES entries (COUNT_SCALAR_FAMILIES) are unbounded-count-only.
CHEM_ALLOWED_FAMILIES = tuple(f for f in SCALAR_FAMILIES if f not in COUNT_SCALAR_FAMILIES)
# The method axis. Bits are
# (structure_expansion, family_expansion, design/acquisition), so `000` is the
# fixed-catalogue control and `110` the free-form arm. The cell a run RECORDS is
# the cell it EXECUTED (the original code recorded the literal `"000"` while a
# free run actually ran a hardcoded `FREE_CELL = "111"`). The THIRD bit was
# inert in stage 1; the acquisition axis (below) makes it select the
# acquisition policy, so `001` and `111` no longer execute what `000` and `110`
# execute.
#
# The F0 factorial: all EIGHT cells, and each bit is read
# on its own. Before this change the first two bits moved together - `CELL_CHOICES`
# held only `00*`/`11*` and `validate_cell` pinned them to `--structure-proposals`
# - so the F0 route (`100`/`101`: the proposer expands the mean function while the
# likelihood family stays the one `--family` designates) could not be expressed at
# all, even though `boxinggym_core`/`core_count` accept all eight and enforce F0 in
# the candidate judgement (`count_models.judge_candidate`, count_models.py:1343-
# 1345). The bits now mean here exactly what they mean there:
#   cell[0] structure expansion - 0 the fixed catalogue (no proposer is built at
#           all, the stricter form of S0 this outcome path has always had), 1 the
#           free-form grammar proposer on top of that same catalogue;
#   cell[1] family expansion    - 0 the designated `--family` only, 1 the task
#           menu `CHEM_ALLOWED_FAMILIES`;
#   cell[2] acquisition         - unchanged (the acquisition axis below).
# `structure_mode` is the one place cell[0] is turned into the catalog/free word
# the rest of this module already speaks, and `--structure-proposals` became an
# optional override that must AGREE with it (`validate_cell`).
# `CATALOG_CELL_PREFIX`/`FREE_CELL_PREFIX` are gone with the binding they served: a two-bit
# prefix can no longer name the structure mode now that the two bits are independent.
# Spelled out as a LITERAL tuple, not `tuple(f"{s}{f}{a}" for s in "01" for f in "01" for a in "012")`: the
# smoke driver's cell-list generator reads this assignment out of the source with
# `ast.literal_eval` rather than importing the module (importing it pulls in pymc/pytensor),
# and a comprehension makes that parse raise instead of reporting the runnable cells.
CELL_CHOICES = ("000", "001", "002", "010", "011", "012", "100", "101", "102", "110", "111", "112")
STRUCTURE_MODES = {"0": "catalog", "1": "free"}
# Stage 2: the acquisition axis, as the third bit
# actually selects it. `design_axis` in config.json/summary.json is the ACTIVE
# policy name on an A=1 cell and `DESIGN_AXIS_ABSENT` on an A=0 cell, where the
# designs are still the pre-drawn `chem_observation_designs` sequence and no
# policy exists at all (`design_policies.check_policy` refuses one there).
# Stage 1's single value `"absent_stage1"` is gone.
DESIGN_AXIS_ABSENT = "fixed_random_designs"
# Executable policies on the positive scalar path. Disagreement uses predictive
# means directly; it does not require a bounded-count outcome grid. Historical
# configurations keep their original advertised lists unless a study policy is
# explicitly selected, so adding capabilities does not alter default artifacts.
CHEM_STUDY_POLICIES = ("point_disagreement", "space_filling")
CHEM_DESIGN_POLICIES = (DEFAULT_POLICY, "epig", *CHEM_STUDY_POLICIES)
# Stage 2: the candidate design pool. PRE-DECLARED here, before
# any stage-2 run exists - the size comes from `PROFILE_GRID[...]
# ["design_pool"]` (dev 32 / paper 256, pre-declared in stage 1 for exactly
# this) and the seed is this constant, fixed for every task, seed and cell so
# the arms of one task compare on one candidate set (the same discipline
# `core_count`'s `np.linspace(0, 1, grid_size)` grid and `boxinggym_core`'s
# task-built `DesignPool` follow: the pool is a property of the benchmark, not
# of the run). Its own stream name keeps it clear of the four `sample_designs`
# streams the runner already draws.
DESIGN_POOL_SEED = 20260908
DESIGN_POOL_STREAM = "chem_design_pool_inputs"
# The per-round acquisition stream, the same shape as
# `core_count.acquisition_seed_for` ("acquisition", world seed, replicate,
# round) and `boxinggym_core`'s ("boxinggym_acquisition", task, seed, round):
# one dedicated stream, keyed by round, that no other draw touches.
ACQUISITION_STREAM = "chem_acquisition"
# One table for the profile grid, the same shape as
# `boxinggym_core.run`'s own one-line tuple (boxinggym_core.py:1946-1947).
# `design_pool`/`eig_samples` were PRE-DECLARED here in stage 1 (before any
# stage-2 run existed, as the pre-declaration rule requires) and stage 2 reads
# them unchanged: `design_pool` is the acquisition axis's `sample_designs` candidate-grid
# size, `eig_samples` boxinggym_core's own EIG posterior sample count for the
# matching profile (`positive_inference.joint_eig`'s `outer_samples`). Both are
# read only on an A=1 cell; an A=0 cell records them and uses neither.
PROFILE_GRID = {
    "dev": {"particles": 64, "q": 32, "design_pool": 32, "eig_samples": 32},
    "paper": {"particles": 500, "q": 256, "design_pool": 256, "eig_samples": 128},
}
DEFAULT_PROFILE = "dev"
# Official evaluation set. `adapters/
# chembench_observation.sample_designs`'s own docstring (":20-21") states that
# `count=1000, seed=42` reproduces the native `evaluate_law` input sampler, and
# the no-leak baseline's offline scoring used exactly that set (shared
# evaluation repository of the comparison method, .../p4_chembench_noleak_oracle_
# blocked_b40_gpt4omini_5runs/manifest.json, `eval_points.rule` = "oracle.
# evaluate_law held-out N=1000 (rng 42), offline after run"). Requested ONCE,
# after `finish()`, and never visible to a packet, a proposer or a fit.
OFFICIAL_EVALUATION_COUNT = 1000
OFFICIAL_EVALUATION_SEED = 42
# `reference_values` key the primary metric scores against: the true FIRST-
# ORDER law only, with no native noise and no secondary T/pH factor - the
# adapter's own `primary_rate` and, per its docstring, "the target of upstream
# evaluate_law". The 90% interval is scored against the observable instead
# (the same convention `boxinggym_core.run`'s `our_scores` uses: coverage of
# the predictive interval is a statement about the OBSERVATION).
OFFICIAL_PRIMARY_TARGET = "primary_rate"
OFFICIAL_INTERVAL_TARGET = "heldout_observation"
# The KL(truth || BMA) target is NOT `OFFICIAL_PRIMARY_TARGET`.
# `nmse_mda`/`nmsle` score the noise-free FIRST-ORDER law (`primary_rate`);
# `noiseless_observation` additionally includes the native background T/pH
# factor (adapters/chembench_observation.py:122-124, :135-136) - the model
# this run's pool actually specifies (a log-linear rate over the seven native
# inputs) never represents that secondary factor, so `kl` and `nmse_mda`
# measure different things and must never be read as agreeing or disagreeing
# metrics. Recorded verbatim in config/summary as `kl_truth_center` and
# `nmse_mda_target`.
OFFICIAL_KL_TRUTH_CENTER = "noiseless_observation"
# Round cap: the free-mode mock schedule is a FIXED list of
# exactly this many rounds. Asking for more is an explicit error, never a
# silent extension - see `_mock_schedule` and `run`.
MOCK_SCHEDULE_ROUNDS = 20
# The archive `note` a refused re-proposal of a retired model carries.
RETIRED_REFUSAL_NOTE = "refused: matched a model retired by evidence"
PRIOR_PROBE_SEED = 20260907  # same constant value as core_count.PRIOR_PROBE_SEED
PRIOR_PROBE_SAMPLES = 64
PRIOR_PROBE_GRID_POINTS = 16
PRIOR_PROBE_NONFINITE = (
    "non-finite mean expression",
    "invalid likelihood",
    "outside open unit interval",
)
_NOISE_PRIOR_FALLBACK = {"sigma": ("HalfNormal", 1.0, None)}
# Free-mode mock schedule (--proposer mock, the default): fixed before any
# run, cycling like CORE's own P1_MOCK_EXPRESSIONS, over ChemBench's own
# standardized input names (schema["input_names"], "{name}z") instead of
# CORE's u/z, since ChemBench declares seven named inputs, not one.
_MOCK_TEMPLATE_FORMS = ("b0", "b0+b1*{a}z", "b0+b1*{a}z+b2*square({a}z)", "b0+b1*{a}z+b2*{b}z")


def _mock_templates(names):
    a, b = names[0], names[1]
    return tuple(form.format(a=a, b=b) for form in _MOCK_TEMPLATE_FORMS)


def _mock_families(allowed):
    return tuple(allowed[(i + 1) % len(allowed)] for i in range(len(_MOCK_TEMPLATE_FORMS)))


def _mock_schedule(names, allowed):
    """The mock proposal schedule, materialized (round cap).

    Exactly `MOCK_SCHEDULE_ROUNDS` (expression, family) entries, one per round
    1..20, built by the SAME `(t - 1) % len(templates)` rotation the original
    code applied inline - so every round <= 20 gets the identical template and
    family it got before (which is why earlier free dev runs are unaffected).
    Materializing it is the point: the schedule is a FIXED 20-entry list, and
    a run that asks for round 21 gets an explicit refusal instead of the
    rotation quietly continuing. A ChemBench run past that schedule (40 or,
    since 2026-09-12, 60 rounds) therefore needs a
    real proposer (LLM calls happen only in the structure-expansion S=1 cells).
    """
    expressions = _mock_templates(names)
    families = _mock_families(allowed)
    return tuple(
        (expressions[(t - 1) % len(expressions)], families[(t - 1) % len(families)])
        for t in range(1, MOCK_SCHEDULE_ROUNDS + 1)
    )


def _prior_particles(hypothesis, count, rng_model_id=None):
    """Prior draws of every coefficient/dispersion parameter, dedicated
    stream - same seeding/sampling rule as core_count.prior_particles /
    boxinggym_core.prior_particles."""
    rng = np.random.default_rng(PRIOR_PROBE_SEED ^ int(rng_model_id or hypothesis.model_id, 16))
    draws = {}
    rules = {name: PRIOR_RULES[role] for name, role in hypothesis.parameters}
    for name in hypothesis.noise_parameters:
        rules[name] = PRIOR_RULES.get(name) or _NOISE_PRIOR_FALLBACK[name]
    for name, (family, a, b) in rules.items():
        if family == "Normal":
            draws[name] = rng.normal(a, b, count)
        elif family == "LogNormal":
            draws[name] = rng.lognormal(a, b, count)
        elif family == "Gamma":
            draws[name] = rng.gamma(a, 1.0 / b, count)
        elif family == "Beta":
            draws[name] = rng.beta(a, b, count)
        elif family == "HalfNormal":
            draws[name] = np.abs(rng.normal(0.0, a, count))
        else:
            raise ValueError(f"prior probe cannot sample {family}")
    return draws


def _prior_probe(hypothesis, interface, response_unit, count=PRIOR_PROBE_SAMPLES,
                 observed_inputs=None, rng_model_id=None):
    """Finiteness gate before any SMC fit; same intent/seeding as
    boxinggym_core.prior_probe's scalar branch (ChemBench's outcome is never
    an EXHAUSTIVE_COUNT_OUTCOMES type, so only that branch is needed)."""
    grid = interface_grid(interface, PRIOR_PROBE_GRID_POINTS)
    particles = _prior_particles(hypothesis, count, **({"rng_model_id": rng_model_id} if rng_model_id is not None else {}))
    result = {"n_samples": int(count), "passed": True, "grid_points": PRIOR_PROBE_GRID_POINTS}
    try:
        # The public input box and disclosed observations are available before
        # fitting. Check every corner as well as the existing diagonal grid;
        # neither held-out inputs nor held-out outputs enter candidate admission.
        from itertools import product

        corners = np.asarray(list(product(*(item["range"] for item in interface["inputs"]))),
                             dtype=float)
        blocks = [np.asarray(grid).reshape(-1, len(interface["inputs"])), corners]
        if observed_inputs is not None:
            blocks.append(np.asarray(observed_inputs).reshape(-1, len(interface["inputs"])))
        grammar_means(hypothesis, particles, np.concatenate(blocks), interface, response_unit)
        mean, sd = mixture_moments(hypothesis, particles, grid, interface, response_unit, 0.0)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(sd)):
            raise ValueError("non-finite mean expression under the prior probe")
    except Exception as exc:  # noqa: BLE001 - a failed probe is a rejection, not a crash
        detail = f"{type(exc).__name__}: {exc}"
        finite = not any(marker in str(exc) for marker in (*PRIOR_PROBE_NONFINITE, "non-finite"))
        if not finite:
            # Prior-probe detail: name the sub-expression, input value and range that went
            # non-finite, on the same particles and the same points (grid, corners,
            # disclosed inputs) the probe evaluated.
            points = np.concatenate(blocks) if "blocks" in locals() else grid
            found = first_nonfinite_subexpression(
                hypothesis.expression, particles, named_input_values(interface, points), interface
            )
            if found is not None:
                return {**result, "passed": False, "normalization_warning": False,
                        "detail": nonfinite_probe_sentence(found), "exception": detail,
                        "nonfinite": found}, None
        return {**result, "passed": finite, "normalization_warning": finite, "detail": detail}, None
    predictive = {
        "grid": grid.tolist(),
        "mean": mean.tolist(),
        "sd": sd.tolist(),
        "n_prior_samples": int(count),
    }
    return result, predictive


def _points_variant(
    structure_proposals,
    prior_template=DEFAULT_PRIOR_TEMPLATE,
    design_policy=DEFAULT_POLICY,
    design_policy_applies=False,
):
    """The plotting contract's `variant` token: sorted, `+`-joined - same
    rule as core_count.points_variant/design_policies.variant_token.

    The role-prior template ablation adds `prior_wide`/`prior_narrow`
    for a replaced role-prior template; the default template emits nothing, so a
    catalog-mode default run's token is unchanged. The LLM-prior ablation (`--priors
    llm`) is deliberately NOT offered on this runner - see `run`.

    The acquisition axis (stage 2) adds the `design_*` token through the SHARED
    `design_policies.variant_token`, applying its rule verbatim: a token appears
    only when the policy differs from the cell default, and the cell default on
    an A=1 cell IS `joint_eig`, so a `001`/`111` joint-EIG run emits NO design
    token and is told apart from `000`/`110` by the `method` column (the cell),
    which is what the plot contract's fixed method list requires. The two
    existing call sites keep their two-argument form."""
    tokens = ["structure_free"] if structure_proposals == "free" else []
    if PRIOR_VARIANT_TOKEN[prior_template]:
        tokens.append(PRIOR_VARIANT_TOKEN[prior_template])
    token = variant_token(design_policy, design_policy_applies)
    if token:
        tokens.append(token)
    return "+".join(sorted(tokens))


def _packet_view(hypothesis, raw_names, grammar_names):
    """A grammar-EVALUABLE belief-packet view of one pool member.

    `belief_packet.build_belief_packet` (via `_predictive_moments` ->
    `positive_inference.mixture_moments` -> `grammar_means`) reads
    `.expression` off the pool's BEST member and evaluates it - there is no
    `is_grammar`-style dispatch inside belief_packet.py itself, because its
    only other caller (boxinggym_core.py) pools nothing but ScalarHypothesis/
    RealHypothesis members. The fixed `RegressionHypothesis` catalogue has no
    grammar `.expression` (`b0 + sum(b_j * standardized_x_j)`, indexed by
    `.features`, not a parsed AST). Rather than fake a placeholder string
    (which would crash the moment this member is the pool's best), this
    rebuilds the IDENTICAL linear form as a real `ScalarHypothesis`,
    `"b0+b1*<feature0>z+b2*<feature1>z+..."`, over this run's own z-scored
    derived inputs (`{name}z`, built from the SAME frozen `transform` the
    catalogue itself is fit under - see `run`) - so its predictive mean/
    weight/residuals are numerically IDENTICAL to the original, using the
    SAME fitted b0/b1/.../noise particles verbatim (`.parameters` is already
    named "b0", "b1", ... on both classes). Only `model_id` differs
    (packet-display only; never read for gating, fitting or duplicate
    detection anywhere in this module). A candidate that is already
    grammar-shaped passes through unchanged.
    """
    if is_grammar(hypothesis):
        return hypothesis
    return ScalarHypothesis(
        _packet_expression(hypothesis, raw_names),
        hypothesis.family,
        hypothesis.parameters,
        grammar_names,
    )


def _packet_expression(hypothesis, raw_names):
    """The structure string `_packet_view` gives one pool member, alone.

    Split out of `_packet_view` so a caller that wants only the STRING (the
    same one `_archive_entry`/`retired_log`/the proposal records write as
    `expression`) does not have to build a grammar namespace first: a catalog
    run never constructs one (`grammar_names` stays None there, see `run`),
    and `pool_weights.jsonl` has to name a fixed-catalogue pool too. The
    returned text is byte-for-byte what `_packet_view(...).expression` is.
    """
    if is_grammar(hypothesis):
        return hypothesis.expression
    terms = "".join(
        f"+b{i}*{raw_names[feature]}z" for i, feature in enumerate(hypothesis.features, 1)
    )
    return f"b0{terms}"


def _pool_weights_row(item, raw_names, model_id_key=None):
    """One `pool_weights.jsonl` line: one checkpoint's WHOLE active pool.

    The same meaning as `boxinggym_core`'s `scores.jsonl` `active_ids`/
    `model_weights`/`single_model_id` triple, which ChemBench had no
    equivalent of: `official_scores_after_finish.json` records only the pool's
    SIZE per round, and the per-query weights `guarded_predictive_summary`
    builds reach no file at all.

    - `weight` is `positive_inference.validate_pool`'s normalized evidence
      weight, exponentiated - the identical array `log_predictive` mixes the
      candidates with (`positive_inference.py` `validate_pool` /
      `log_predictive`), so this file states the weights the round's BMA
      actually used, not a recomputation under some other rule.
    - `logZ` is the fit's own `log_evidence`, which is what those weights are
      the normalized exponential of (`count_inference.model_logweights`).
    - `structure` is the archive's own `expression` string
      (`_packet_expression`), so a fixed-catalogue member is named here
      exactly as `_archive_entry` would name it.
    - `single_model_id` is the pool's argmax weight. Ordinarily its tie key
      agrees with `positive_inference.predictive_summary`; input-name masking alone uses the
      restored-name ID locally while shared inference keeps its masked tie key.
    """
    fits = item["fits"]
    logweights = validate_pool(fits)
    best = min(range(len(fits)), key=lambda i: (-logweights[i], model_id_key(fits[i].hypothesis) if model_id_key else fits[i].hypothesis.model_id))
    return {
        "round": item["round"],
        "observation_count": item["observation_count"],
        "active_pool": len(fits),
        "models": [
            {
                "model_id": result.hypothesis.model_id,
                "structure": _packet_expression(result.hypothesis, raw_names),
                "logZ": float(result.log_evidence),
                "weight": float(np.exp(logweight)),
            }
            for result, logweight in zip(fits, logweights, strict=True)
        ],
        "single_model_id": fits[best].hypothesis.model_id,
    }


def _archive_entry(candidate, round_index, outcome, reason, note=None):
    """`note` is `belief_packet.ARCHIVE_OPTIONAL_KEYS`'s own optional field.

    Used for exactly one thing here (a design decision of 2026-09-08 on pool
    retirement): when a re-proposed candidate is refused because it matches a model
    that was RETIRED by evidence, the archive says so in as many words, rather
    than leaving the reader to cross-reference the earlier
    `retired_by_evidence` entry for the same `model_id`. The `reason` key set
    stays exactly what `ARCHIVE_REASON_KEYS[outcome]` fixes.

    Archive parameters: every entry also records `parameters`, the candidate's role mapping as the
    engine judged it. A record field only; the belief packet never copies it.
    """
    entry = {
        "model_id": candidate.model_id,
        "expression": candidate.expression,
        "family": candidate.family,
        "parameters": dict(candidate.parameters),
        "proposed_round": round_index,
        "outcome": outcome,
        "reason": reason,
    }
    if note is not None:
        entry["note"] = note
    return entry


def family_refusal(cell, candidate_family, initial_family):
    """The F0 judgement, per candidate, in this runner's own loop.

    Returns the refusal reason when `cell[1] == "0"` and the candidate's family
    is not the designated `--family`, else None. This is
    `count_models.judge_candidate`'s F0 branch (count_models.py:1343-1345) with
    its own reason constant, applied one candidate at a time and never to a
    batch - the per-candidate rule the whole judge was rewritten for.

    It exists because the LLM path and the mock path reach a candidate by
    different routes. An LLM candidate is judged inside
    `count_proposer.parse_candidate_object` -> `judge_candidate` before `run`
    ever sees it, and its refusal is recorded there (the transport record's
    `rejected_candidates`); the mock path builds its candidate straight from the
    fixed `_mock_schedule`, whose families deliberately cycle through the whole
    task menu, so nothing judges it at all. The call site below therefore covers
    BOTH - it is simply a no-op for an LLM candidate that the judge already
    admitted - rather than adding a second, mock-only rule.
    """
    if cell[1] == "1" or candidate_family == initial_family:
        return None
    return F0_REFUSAL_REASON


def select_task(rows, task_id):
    """`--task <id>` picks ONE manifest row.

    The original code read `rows[0]` unconditionally, so a nine-row manifest
    would have silently run its first row nine times. `--task` may be omitted
    only for a one-row manifest (the development manifest, and the byte-
    identity anchor's own command line); a multi-row manifest without `--task`
    is an explicit error naming every available id, never row 0.
    """
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError("manifest must be a non-empty JSON list of task rows")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"manifest task ids are not unique: {ids}")
    if task_id is None:
        if len(rows) != 1:
            raise ValueError(f"--task is required for a {len(rows)}-row manifest; ids are {ids}")
        return rows[0]
    selected = [row for row in rows if row["id"] == task_id]
    if len(selected) != 1:
        raise ValueError(f"task id {task_id!r} is not in the manifest; ids are {ids}")
    return selected[0]


def structure_mode(cell):
    """cell[0] alone decides whether this run proposes structures.

    `0` is the fixed `initial_catalogue` control - no proposer object is built,
    no packet is written, nothing is ever proposed - and `1` adds the free-form
    grammar candidate on top of that same catalogue. This is the ONLY reading of
    the structure bit in this module; every `free` decision below goes through
    it, so the executed mode and the recorded `structure_proposals` cannot drift
    apart the way the cell and the executed arm once did.
    """
    return STRUCTURE_MODES[cell[0]]


def validate_cell(cell, structure_proposals=None):
    """The recorded cell must be the executed cell.

    Bits are (structure_expansion, family_expansion, design/acquisition) and,
    since the F0 factorial, each is read on its own - all eight
    cells are legal, exactly as in `boxinggym_core`/`core_count`. The structure
    mode is DERIVED from cell[0] (`structure_mode`); `--structure-proposals` is
    an optional OVERRIDE that must agree with it, kept rather than dropped for
    three reasons: the earlier command lines, the byte-identity anchors and
    smoke driver of that time all state it explicitly and stay
    valid unchanged; `config.json` records it through `**vars(args)` and
    `_points_variant` reads it, so the key has to keep existing; and an
    explicitly stated mode that contradicts the cell is a typo worth refusing
    rather than silently overruling. Omitted (`None`, the CLI default) means
    "take cell[0]", which is how a `100`/`101` run needs no new flag at all.

    The second bit is NOT checked here: it selects the family menu the packet
    states (`belief_packet.cell_constraints`: `allowed_families` is
    `[initial_family]` whenever `cell[1] == "0"`) and the rule the candidate
    judgement applies per candidate (`count_models.judge_candidate`,
    count_models.py:1343-1345, and `family_refusal` above for the mock path).
    The third bit is the acquisition axis (stage 2), checked separately by
    `design_policies.check_policy(cell, policy)`.
    """
    if cell not in CELL_CHOICES:
        raise ValueError(f"--cell must be one of {CELL_CHOICES}")
    derived = structure_mode(cell)
    if structure_proposals is not None and structure_proposals != derived:
        raise ValueError(
            f"--cell {cell} is inconsistent with --structure-proposals "
            f"{structure_proposals} (cell[0] selects {derived}; omit the flag or state it)"
        )
    return cell


def _design_key(design):
    """One design as an order-free, hashable name->value map (acquisition axis).

    Used only to intersect the candidate pool with the official evaluation set:
    equality of designs is equality of the seven named inputs, which must not
    depend on which order a dict happens to carry them in.
    """
    return tuple(sorted((str(name), float(value)) for name, value in design.items()))


def _oracle_provenance():
    """`autoscilab` is never installed - PYTHONPATH only.

    Records WHICH copy answered without importing it (`find_spec` does not
    execute the module); the oracle itself is probed separately, through the
    adapter, at the start of `run`.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec("autoscilab")
    except Exception:  # noqa: BLE001 - provenance must never fail a run
        spec = None
    locations = list(getattr(spec, "submodule_search_locations", None) or ())
    return {
        "package": "autoscilab",
        "installed": False,
        "resolution": "PYTHONPATH only, never installed (a project decision)",
        "path": str(Path(locations[0]).resolve()) if locations else None,
    }


# The plotting contract: the 25 columns live in ONE place, `core_count.
# CONTRACT_POINTS_COLUMNS` - the contract's own 1..21 (ending `variant`), then
# `code_sha` (22), `brier` (23), `nmsle` (24), `rmsle` (25). This runner emits
# exactly that tuple; `brier` has no definition here (the outcome is a positive
# continuous rate, not a binary event) and is therefore BLANK, never 0, while
# `nmsle` and `rmsle` are the primary metric's two auxiliaries and are filled. See
# `write_points`.
PRIMARY_METRIC = "nmse_mda"
PRIMARY_METRIC_LABEL = (
    "nMSLE vs the noise-free primary rate on the official Q=1000 (log1p, lower is better)"
)


def log_error_metrics(predicted_mean, log_truth, log_truth_variance):
    """The primary metric and its two auxiliaries.

    Isolated from the predictive machinery so the DEFINITION itself is
    testable and quotable:

        squared  = mean_Q[(log1p(pred) - log1p(truth))^2]
        rmsle    = sqrt(squared)
        nmse_mda = nmsle = squared / Var_Q[log1p(truth)]

    `nmse_mda` and `nmsle` are one statistic under two contract names (the
    ChemBench primary was renamed `nmse_mda` because the no-leak
    baseline's 224 rows already carry that key; the contract's first edition
    called it nMSLE). `Var_Q` is the population variance, ddof = 0.
    """
    squared = float(np.mean((np.log1p(np.asarray(predicted_mean, dtype=float)) - log_truth) ** 2))
    normalized = squared / log_truth_variance
    return {"nmse_mda": normalized, "nmsle": normalized, "rmsle": float(np.sqrt(squared))}


# `kl` quadrature constants.
#
# `KL_QUADRATURE_BASE_NODES` is doubled to get the recorded
# `kl_quadrature_error` (Gauss-Legendre with N nodes vs 2N); an offline check
# (a scratch script, not part of this
# tree) found N=32 already agrees with a closed-form normal-normal KL to
# ~2.5e-11 and with a fine (1e6-point) Riemann sum against a two-component
# lognormal mixture to ~1.8e-10, so 2N=64 nodes is the reported estimate.
# Monte Carlo is never used for this integral (the design: "no Monte Carlo").
KL_QUADRATURE_BASE_NODES = 32
# The quadrature interval is the truth's own mean +/- this many RELATIVE
# standard deviations, truncated at 0 - the design's own choice of interval
# ("interval ν_q ± 8·noise·ν_q"). The probability mass this excludes is measured
# and recorded (`kl_truncated_mass`), never assumed to be zero.
KL_TRUNCATION_SIGMAS = 8.0
# The size cap (decimal bytes) on the per-candidate particle mean/scale
# recompute artifact; above it, the artifact falls back to the mixture's own
# log-density at the KL quadrature nodes (the design: "above 20 MB, store the mixture
# density's node values instead of the particles, and disclose it").
KL_PARTICLE_SUMMARY_MAX_BYTES = 20_000_000
# The particle-level scale/shape parameter name per family this path's pool
# ever contains (`CHEM_ALLOWED_FAMILIES` = lognormal, gamma - never a count
# family), read from `Fit.particles` for the double-precision recompute artifact.
_KL_SCALE_PARTICLE_NAME = {"lognormal": "sigma", "gamma": "shape"}


def censored_normal_kl_quadrature(nu, relative_sd, log_q_batch, n_nodes):
    """KL(N+(nu, (relative_sd*nu)^2) || model), per row, via Gauss-Legendre.

    `N+` is the law of Y = max(X, 0), X ~ Normal(nu, (relative_sd*nu)**2) -
    the adapter's own native noise arithmetic, `r0_obs = max(r0_true*(1.0 +
    noise), 0.0)` with `noise ~ Normal(0, noise_level)`
    (autoscilab/oracle/chembench.py:730, :551, default noise_level 0.01
    at :595) - NOT a renormalized truncated normal: the density for y > 0 is
    the plain Normal(nu, (relative_sd*nu)**2) density with no rescaling, and
    the excluded x <= 0 half collapses onto a point mass at y = 0. `nu` is
    one row per query point and must be strictly positive (no clipping
    fallback - a ChemBench rate is positive by construction, adapters/
    chembench_observation.py:132-136).

    `log_q_batch(y)` is the ONLY place this function reaches outside itself:
    handed the (n_query, n_nodes) node matrix this function built, it must
    return the model's log-density at every (query, node) pair, same shape.
    A test may inject any callable here - a closed-form comparator, a
    hand-built mixture, or even the truth's own log-density - with no
    dependency on `positive_inference.log_predictive` or a fitted pool at
    all; `official_kl_summary` below is what wires the real one.

    The y=0 point mass, `Phi(-nu/(relative_sd*nu))`, is computed via
    `scipy.stats.norm.logcdf` (stable in log-space) - a genuine computation,
    never a hardcoded default - and at this benchmark's noise levels
    (noise=0.01 => z=-1/noise=-100) it underflows to exactly Python float 0.0;
    a nonzero value would make the KL against a continuous model density
    formally infinite, so this function refuses rather than silently
    returning a finite number in that case. The excluded tail mass outside
    the quadrature interval, `1 - (Phi(z_hi) - Phi(z_lo))`, is likewise
    computed and returned, never assumed.
    """
    nu = np.atleast_1d(np.asarray(nu, dtype=float))
    if nu.ndim != 1 or not np.all(np.isfinite(nu)) or np.any(nu <= 0):
        raise ValueError(
            "kl truth center (noiseless_observation) must be a finite, strictly positive "
            "1-D array; no clipping fallback"
        )
    if not np.isfinite(relative_sd) or relative_sd <= 0:
        raise ValueError("kl relative_sd (noise) must be finite and positive")
    if n_nodes < 2:
        raise ValueError("at least two quadrature nodes required")
    sigma = relative_sd * nu
    lo = np.maximum(0.0, nu - KL_TRUNCATION_SIGMAS * sigma)
    hi = nu + KL_TRUNCATION_SIGMAS * sigma
    base_y, base_w = np.polynomial.legendre.leggauss(n_nodes)
    half = 0.5 * (hi - lo)
    y = half[:, None] * base_y[None, :] + (0.5 * (hi + lo))[:, None]
    w = half[:, None] * base_w[None, :]
    log_p = norm.logpdf(y, loc=nu[:, None], scale=sigma[:, None])
    log_q = np.asarray(log_q_batch(y), dtype=float)
    if log_q.shape != y.shape:
        raise ValueError("log_q_batch must return one log-density per query per node")
    kl = np.sum(np.exp(log_p) * (log_p - log_q) * w, axis=1)
    if not np.all(np.isfinite(kl)):
        raise ValueError("non-finite KL contribution")
    z_lo, z_hi = (lo - nu) / sigma, (hi - nu) / sigma
    truncated_mass = 1.0 - (norm.cdf(z_hi) - norm.cdf(z_lo))
    atom_mass = np.exp(norm.logcdf(-nu / sigma))
    if np.any(atom_mass > 0.0):
        raise ValueError(
            "nonzero truth point mass at y=0 (atom_mass > 0); this quadrature path does not "
            "define a finite KL against a continuous model density"
        )
    return {
        "kl": kl,
        "truncated_mass": truncated_mass,
        "atom_mass": atom_mass,
        "lo": lo,
        "hi": hi,
        "sigma": sigma,
        "y": y,
        "log_q": log_q,
    }


def _prediction_pools(fits, inputs, round_index, checkpoint):
    """Point-local admissibility, retaining all particles of each valid model.

    The ordinary all-valid path keeps the original batched arithmetic and RNG.
    Only the known positive-mean error triggers exclusion; other failures retain
    their original exception behavior. No posterior or active-pool state changes.
    """
    inputs = np.asarray(inputs, dtype=float)
    eligible = [list(fits) for _ in inputs]
    exclusions = []
    for result in fits:
        try:
            particle_means(result, inputs)
            continue
        except ValueError as exc:
            if "non-finite/nonpositive mean" not in str(exc):
                raise
        for index, point in enumerate(inputs):
            try:
                particle_means(result, point[None, :])
            except ValueError as exc:
                if "non-finite/nonpositive mean" not in str(exc):
                    raise
                eligible[index] = [item for item in eligible[index] if item is not result]
                exclusions.append({
                    "model_id": result.hypothesis.model_id,
                    "round": round_index,
                    "checkpoint": checkpoint,
                    "query_index": index,
                    "reason": str(exc),
                })
    if any(not pool for pool in eligible):
        raise ValueError("all models excluded at prediction point; no clipping fallback")
    return eligible, exclusions


def guarded_predictive_summary(fits, inputs, seed, round_index=None,
                               checkpoint="prediction", sample_count=2048):
    """`predictive_summary` with point-local model exclusion (see `_prediction_pools`)."""
    pools, exclusions = _prediction_pools(fits, inputs, round_index, checkpoint)
    if not exclusions:
        return predictive_summary(fits, inputs, seed, sample_count)
    rows = [predictive_summary(pool, np.asarray(inputs)[i:i + 1], seed, sample_count)
            for i, pool in enumerate(pools)]
    summary = {key: np.concatenate([row[key] for row in rows], axis=0)
               for key in ("mean", "single_mean", "lower90", "upper90", "samples")}
    weights = np.zeros((len(inputs), len(fits)))
    indices = {result.hypothesis.model_id: i for i, result in enumerate(fits)}
    for index, (pool, row) in enumerate(zip(pools, rows, strict=True)):
        for result, weight in zip(pool, row["model_weights"], strict=True):
            weights[index, indices[result.hypothesis.model_id]] = weight
    return {**summary, "model_weights": weights,
            "single_model_id": [row["single_model_id"] for row in rows],
            "measure": rows[0]["measure"], "prediction_excluded_models": exclusions}


def guarded_log_predictive(fits, inputs, outcomes, round_index=None, checkpoint="prediction"):
    """`log_predictive` with point-local model exclusion; returns (mixture, per-model terms)."""
    pools, exclusions = _prediction_pools(fits, inputs, round_index, checkpoint)
    if not exclusions:
        return log_predictive(fits, inputs, outcomes)
    y = np.asarray(outcomes)
    y = y[:, None] if y.ndim == 1 else y
    density = np.empty_like(y, dtype=float)
    components = np.full((len(fits), *y.shape), -np.inf)
    indices = {result.hypothesis.model_id: i for i, result in enumerate(fits)}
    for index, pool in enumerate(pools):
        mixed, terms = log_predictive(pool, np.asarray(inputs)[index:index + 1], y[index:index + 1])
        density[index] = mixed[0]
        for result, term in zip(pool, terms, strict=True):
            components[indices[result.hypothesis.model_id], index] = term[0]
    return density, components


# Acquisition guard (a project decision of 2026-09-24, confirmed 2026-09-25): the
# ascent acquisition's own guard, an acquisition estimator, fixed in advance, that excludes particles
# producing non-finite/nonpositive means at the design points from the conditional sample, and
# records the exclusion counts and model removals. The invalid-row reason is design_search's
# registered vocabulary.
ACQUISITION_INVALID_REASON = "nonfinite_or_nonpositive_mean"
RETAINED_MODEL_WEIGHTING = "model_logweights_over_retained_models"


def _lognormal_scale_representable(result, mean):
    """Acquisition guard v2 (a project decision of 2026-09-26): [particle, point] mask of a
    lognormal bank.

    Log space only: the scale `positive_inference.conditional_loglik` hands to `lognorm`,
    mean * exp(-sigma^2 / 2), is never formed, since forming it is the underflow this guards.
    A cell is kept iff mean, sigma, sigma^2 and log_scale = log(mean) - 0.5 * sigma^2 are each
    finite, mean > 0, and log_scale -/+ LOGNORMAL_TAIL_SIGMA * sigma lies within
    [REPRESENTABILITY_LOWER, REPRESENTABILITY_UPPER] (design_search's registered constants).
    """
    from .design_search import LOGNORMAL_TAIL_SIGMA, REPRESENTABILITY_LOWER, REPRESENTABILITY_UPPER

    mean = np.asarray(mean, dtype=float)
    sigma = np.asarray(result.particles["sigma"], dtype=float)[:, None]
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        sigma_squared = sigma**2
        log_scale = np.log(mean) - 0.5 * sigma_squared
        tail = LOGNORMAL_TAIL_SIGMA * sigma
        return (np.isfinite(mean) & (mean > 0) & np.isfinite(sigma) & np.isfinite(sigma_squared)
                & np.isfinite(log_scale) & (log_scale - tail >= REPRESENTABILITY_LOWER)
                & (log_scale + tail <= REPRESENTABILITY_UPPER))


def _acquisition_bank_for_points(fits, points):
    """Condition one objective call's bank on valid means at every one of its points.

    Per model, `particle_means_masked` runs on the very `points` the objective received, and a
    particle is kept iff its mean is valid at all of them (finite and > 0 for the positive
    families, finite for the real ones) and, for a lognormal model (acquisition guard v2), its
    scale is representable there (`_lognormal_scale_representable`). With every particle kept
    this returns the SAME list object and None: no new arrays, no normalization, no RNG.
    Otherwise a partially valid model becomes a copy holding only its kept particles
    (evidence, diagnostics and hypothesis unchanged), a fully valid model stays the same
    object and a model with no valid particle is excluded. The retained bank goes to
    `joint_eig` as it is, whose `validate_pool` renormalizes the evidence over the retained
    models (RETAINED_MODEL_WEIGHTING), the normalization the prediction path applies when it
    drops models. `info["invalid"]` is True iff no model is retained. Independent of
    count_inference's EPIG guard by design.
    """
    retained, excluded_particles, excluded_models = [], {}, []
    for result in fits:
        size = particle_count(result)
        mean, valid = particle_means_masked(result, points)
        if result.hypothesis.family == "lognormal":
            valid = valid & _lognormal_scale_representable(result, mean)
        if valid.shape != (size, len(points)):
            raise ValueError("acquisition particle mask must align with particles and design points")
        keep = valid.all(axis=1)
        kept = int(np.count_nonzero(keep))
        if kept == size:
            retained.append(result)
            continue
        model_id = result.hypothesis.model_id
        excluded_particles[model_id] = {"excluded": size - kept, "of": size}
        if kept:
            retained.append(replace(result, particles={
                name: np.asarray(values)[keep] for name, values in result.particles.items()}))
        else:
            excluded_models.append(model_id)
    if not excluded_particles:
        return fits, None
    return retained, {"excluded_particles": excluded_particles,
                      "excluded_models": excluded_models, "invalid": not retained}


def _eig_exclusion_record(fits, calls, record):
    """An ascent round's exclusion keys, from its non-empty call log, in pool order.

    Acquisition guard v2 (a project decision of 2026-09-26): the record restates the guard it
    excluded by, `eig_exclusion_guard`, with its registered representability constants.
    """
    from .design_search import acquisition_guard_constants

    excluded_particles, excluded_models = {}, {}
    for result in fits:
        model_id = result.hypothesis.model_id
        counts = [call["excluded_particles"][model_id]["excluded"] for call in calls
                  if model_id in call["excluded_particles"]]
        if counts:
            excluded_particles[model_id] = {"of": particle_count(result), "calls": len(counts),
                                            "excluded_total": sum(counts),
                                            "excluded_max": max(counts)}
        removed = sum(model_id in call["excluded_models"] for call in calls)
        if removed:
            excluded_models[model_id] = removed
    return {
        "acquisition_status": record.get("acquisition_status", "ascent_with_exclusions"),
        "eig_excluded_particles": excluded_particles,
        "eig_excluded_models": excluded_models,
        "eig_exclusion_calls": calls,
        "eig_exclusion_summary": {
            "objective_call_count": record["objective_call_count"],
            "calls_with_excluded_particles": len(calls),
            "invalid_objective_call_count": sum(call["invalid"] for call in calls),
            "excluded_particle_count": sum(item["excluded"] for call in calls
                                           for item in call["excluded_particles"].values()),
            "excluded_model_count": sum(len(call["excluded_models"]) for call in calls),
            "retained_model_weighting": RETAINED_MODEL_WEIGHTING,
        },
        "eig_exclusion_guard": acquisition_guard_constants(),
    }


def _acquisition_fallback(record, fits, designs, index, names, policy):
    """The one fallback of an all_designs_invalid round, with no retry.

    The design is `designs[index]`, with index = d0_count - 1 + t: exactly the dict a cell-100
    run with the same seed observes in this round. It is checked once with the same guard, at
    the float point the objective would receive. Returns the design to observe, or None when it
    is invalid too, which makes the round `acquisition_failed`.
    """
    design = designs[index]
    point = np.asarray([[design[name] for name in names]], dtype=float)
    _, info = _acquisition_bank_for_points(fits, point)
    valid = info is None or not info["invalid"]
    record.update(fallback_policy=policy, fallback_design_index=index,
                  fallback_design=point[0].tolist(), fallback_valid=valid)
    if info is not None:
        record.update(fallback_excluded_particles=info["excluded_particles"],
                      fallback_excluded_models=info["excluded_models"])
    if valid:
        return design
    record["acquisition_status"] = "acquisition_failed"
    return None


# Unified protocol, ChemBench part: the joint EIG's decomposition on the ascent
# path.
#
# Every EIG-type ascent round (joint_eig and the two EIG-part arms below) records, on each of the
# five arrival rows the chooser compares, the two chain-rule parts of the joint EIG from the same
# Monte Carlo draws, I((M,theta);Y) = I(M;Y) + E_M I(theta;Y|M), and the row's point
# disagreement (`annotate_eig_endpoints`). An EIG-part arm optimizes its part itself: its ascent
# objective returns the decomposed EIG on every evaluated point, and the search, noise floor,
# chooser and holdout read the part (`design_search.projected_spsa(objective_key=...)`; the same
# rule as ForceBench's fb1_design_search). The pool path never runs an EIG part (pool EIG is
# refused at d >= 3 by the acquisition policy, and every ChemBench task has seven inputs).
# policy -> (objective column the ascent climbs and the chooser maximizes, its Monte Carlo
# standard-error column, acquisition `target`, points.csv `variant` token). Names, targets and
# tokens are those of BoxingGym's EIG-variant arms and the shared registry's
# `structure_eig` (design_policies.TARGET and VARIANT_TOKEN); `parameter_eig` is not in that
# registry, so this runner checks it itself.
CHEM_EIG_PART_POLICIES = {
    "structure_eig": ("structure_eig", "structure_mc_standard_error", "model_structure",
                      "design_structure_eig"),
    "parameter_eig": ("parameter_eig", "parameter_mc_standard_error",
                      "model_parameter_given_structure", "design_parameter_eig"),
}
# The protocol's name for the structure part (Model EIG, MDA v4 Eq. 26). Accepted on the command
# line and recorded as `structure_eig`, with the typed name in config's `design_policy_requested`.
CHEM_EIG_PART_ALIASES = {"model_eig": "structure_eig"}
CHEM_EIG_PART_DEFINITION = {
    "structure_eig": ("I(M;Y|x,D) = H[p(y|x,D)] - sum_m p(m|D) H[p(y|x,m,D)] "
                      "(Model EIG, MDA v4 Eq. 26)"),
    "parameter_eig": "sum_m p(m|D) I(theta;Y|x,m,D) = joint EIG - Model EIG (Parameter EIG)",
}
EIG_COMPONENT_COLUMNS = ("structure_eig", "parameter_eig", "structure_mc_standard_error",
                         "parameter_mc_standard_error")
EIG_COMPONENTS_RECORD = {
    "recorded": True,
    "rows": "acquisition/tNN.json arrival_rows (the ascent endpoints the chooser compares; the "
            "same row objects appear in rows)",
    "columns": ["eig", *EIG_COMPONENT_COLUMNS],
    "estimator": (
        "positive_inference.joint_eig(decompose=True) at the row's design, objective seed and "
        "eig_samples on the acquisition bank the objective used there: the same draws, checked by "
        "the recomputed joint value and standard error equalling the row's bit for bit (on an "
        "EIG-part arm the objective itself makes this call at every evaluated point and the "
        "recomputation must reproduce all of its columns); "
        "structure = mean_s[log p_k(y_s|x) - log p(y_s|x)], parameter = mean_s[log p(y_s|theta_s,k,x) "
        "- log p_k(y_s|x)], so eig = structure + parameter draw by draw"),
    "definition": CHEM_EIG_PART_DEFINITION,
}
POINT_DISAGREEMENT_RECORD = {
    "recorded": True,
    "column": "point_disagreement",
    "rows": EIG_COMPONENTS_RECORD["rows"],
    "definition": POINT_DISAGREEMENT_DEFINITION,
    "estimator": ("positive_inference.point_disagreement on the acquisition bank the row's EIG "
                  "used (every retained particle of every model valid at that design); no RNG"),
    "nonfinite": "null with point_disagreement_status nonfinite",
    "design_policy": False,
}


def eig_part_policy_applies(cell, policy, ascent):
    """The EIG-part arms' own refusals, the `design_policies.check_policy` analogue.

    An EIG part replaces the joint EIG of an A=1 cell, so an A=0 cell (the fixed random-draw
    anchor) is refused in check_policy's words; an A=2 cell never gets here
    (`count_inference.resolve_epig_policy` binds it to epig). On this positive-rate path a part is
    optimized by the design-search ascent only, so the pool search is refused before any output.
    """
    if str(cell)[2] != "1":
        raise ValueError(
            f"design policy {policy!r} needs an acquisition bit: cell {cell} chooses its designs "
            "by the fixed random draw and is an unchanged anchor"
        )
    if not ascent:
        raise ValueError(
            f"design policy {policy!r} is not wired on this positive-rate path's pool search: an "
            "EIG part is optimized by the design-search ascent; declare --design-search ascent"
        )
    return True


def eig_part_points_variant(structure_proposals, prior_template, design_policy):
    """`_points_variant` for an EIG-part arm: the same sorted `+` join, plus the part's token."""
    tokens = [token for token in _points_variant(structure_proposals, prior_template).split("+")
              if token]
    return "+".join(sorted([*tokens, CHEM_EIG_PART_POLICIES[design_policy][3]]))


def annotate_eig_endpoints(fits, rows, eig_samples):
    """The joint EIG's two parts and the point disagreement on the arrival rows (unified protocol).

    The ascent objective's rows are unchanged: a joint_eig arm's carry eig and mc_standard_error,
    an EIG-part arm's also the four part columns (its objective makes the decomposed call at every
    evaluated point). This adds to every VALID arrival row, in place (an invalid row gains nothing):

    - structure_eig, parameter_eig and their Monte Carlo standard errors. The row's own design
      point gets the acquisition bank the objective built there (`_acquisition_bank_for_points`
      on that one point) and `positive_inference.joint_eig(..., decompose=True)` is called with
      the row's own objective seed and `eig_samples` - the objective's call, so the same draws
      (y_s, k_s, theta_s), s = 1..S. The recomputed joint value and standard error, and any part
      column the row already has, must equal the row's bit for bit, else ValueError. With
      p_k(y|x) = (1/P_k) sum_i p(y|theta_i,k,x) model k's particle average and
      p(y|x) = sum_k w_k p_k(y|x) the evidence-weighted mixture:
        structure_eig = (1/S) sum_s [log p_{k_s}(y_s|x) - log p(y_s|x)]            (Model EIG)
        parameter_eig = (1/S) sum_s [log p(y_s|theta_s,k_s,x) - log p_{k_s}(y_s|x)]
      so eig = structure_eig + parameter_eig draw by draw (to rounding).
    - point_disagreement, the evidence-weighted variance across the active models of their
      posterior-mean predictions at x:
        point_disagreement(x) = sum_m w_m (mu_m(x) - mubar(x))**2,  mubar(x) = sum_m w_m mu_m(x),
      over the models in that same bank (every active model unless the acquisition guard excluded
      one at x), with w_m = exp(logZ_m - logsumexp_j logZ_j) the normalized log-evidence weight
      over those models and mu_m(x) = (1/P_m) sum_i E[y | theta_i, m, x] the particle-averaged
      conditional predictive mean of model m over its P_m retained particles (for both positive
      families E[y|theta] is the mean parameter, response_unit * exp(expression)). This is
      `positive_inference.point_disagreement` (POINT_DISAGREEMENT_DEFINITION) in the observation's
      units squared; no RNG. A value that overflows float64 is recorded as null with
      `point_disagreement_status` "nonfinite" rather than failing the round.

    Record-only on every arm: the chooser reads the objective's own column (eig, or the part).
    """
    for row in rows:
        if row.get("invalid") is True:
            continue
        point = np.asarray([row["design"]], dtype=float)
        bank, info = _acquisition_bank_for_points(fits, point)
        if info is not None and info["invalid"]:
            raise ValueError("an arrival row the objective scored has no valid model on recomputation")
        table = joint_eig(bank, point, row["objective_seed"], eig_samples, decompose=True)
        parts = {column: float(table[column][0]) for column in EIG_COMPONENT_COLUMNS}
        if (float(table["eig"][0]) != row["eig"]
                or float(table["mc_standard_error"][0]) != row["mc_standard_error"]
                or any(row[column] != value for column, value in parts.items() if column in row)):
            raise ValueError("arrival-row joint EIG recomputation differs from the objective's value; "
                             "its parts would not come from the same draws")
        row.update(parts)
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                value = float(point_disagreement(bank, point)[0])
        except ValueError as exc:
            # A particle average past float64's range: design_policies.point_disagreement's own
            # refusal of a non-finite mean; recorded, never a failed round.
            if "non-finite predictive mean" not in str(exc):
                raise
            value = float("nan")
        if np.isfinite(value):
            row["point_disagreement"] = value
        else:
            row.update(point_disagreement=None, point_disagreement_status="nonfinite")


# Three opt-in options of the unified protocol, all absent by default:
#   --composition-paragraph v2   the k = 3 request composes count_proposer.COMPOSITION_PARAGRAPH_V2
#                                (v1 stays the default); v2 needs --proposal-limit 3.
#   --archive-review-rounds L    on each listed round the request also carries
#                                ARCHIVE_REVIEW_PARAGRAPH and the reply is {"candidates": [...],
#                                "revive": [...]}: up to REVIVE_LIMIT archived models retired by
#                                evidence, named by the LLM, are refit from the prior on all public
#                                data and compete in the same single retention as the round's new
#                                candidates; revival/tNN.json records it. Needs v2 and k = 3.
#   --mock-schedule synthetic_v1 the mock proposer draws from a long synthetic form list
#                                (`synthetic_mock_forms`) instead of the fixed 20-entry schedule, so a
#                                mock cell admits about as many models per round as a paid one, and
#                                on a review round names one valid and one invalid revive id.
MOCK_SCHEDULES = ("fixed20", "synthetic_v1")
REVIVE_LIMIT = 2
# synthetic_v1: the round-0 burst reads forms 0.. (each wave continuing where the last stopped);
# round t's slot c (1-based) reads form SYNTHETIC_ROUND_BASE + 3 (t - 1) + (c - 1). Three planted
# duplicates exercise the duplicate path: round -> {slot: form it repeats}, where a negative
# number -k repeats this round's slot k and a non-negative one repeats that form index.
SYNTHETIC_ROUND_BASE = 36
SYNTHETIC_PLANTED_DUPLICATES = {7: {3: -1}, 11: {2: 0}, 14: {3: -1}}


def synthetic_mock_forms(names, bounds):
    """synthetic_v1: distinct log-link forms over the run's own (possibly masked) input names.

    Synthetic, never taken from a run: in order, every single z-scored input, every pair, every
    input with its square, every product of two, log of every input whose declared range is
    strictly positive, and every triple - 96 forms for ChemBench's seven inputs (five of them
    with a strictly positive range), the first twelve the singles and the first five pairs, so
    the burst (forms 0..35 at most) and twenty rounds of three (36..95) never share a form.
    Coefficients b0 (intercept) and b1.. (signed) by `core_count.mock_parameters`.
    """
    z = [f"{name}z" for name in names]
    count = len(z)
    pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
    forms = [f"b0+b1*{zi}" for zi in z]
    forms += [f"b0+b1*{z[i]}+b2*{z[j]}" for i, j in pairs]
    forms += [f"b0+b1*{zi}+b2*square({zi})" for zi in z]
    forms += [f"b0+b1*{z[i]}*{z[j]}" for i, j in pairs]
    forms += [f"b0+b1*log({name})" for name in names if float(bounds[name][0]) > 0]
    forms += [f"b0+b1*{z[i]}+b2*{z[j]}+b3*{z[k]}"
              for i in range(count) for j in range(i + 1, count) for k in range(j + 1, count)]
    return tuple(forms)


def synthetic_round_form(forms, t, slot, this_round):
    """synthetic_v1's form for round t's 1-based slot; `this_round` holds the slots before it."""
    planted = SYNTHETIC_PLANTED_DUPLICATES.get(t, {}).get(slot)
    if planted is not None:
        return this_round[-planted - 1] if planted < 0 else forms[planted]
    index = SYNTHETIC_ROUND_BASE + 3 * (t - 1) + (slot - 1)
    if index >= len(forms):
        raise ValueError(f"synthetic mock schedule has {len(forms)} forms; round {t} slot {slot} "
                         "is outside it")
    return forms[index]


def latest_archive_outcomes(archive):
    """Each archived model id's latest outcome (the archive is an append-only event list)."""
    latest = {}
    for entry in archive:
        latest[entry["model_id"]] = entry
    return latest


def validate_revive_request(requested, archive, retired_now, active_ids, limit=REVIVE_LIMIT):
    """The valid revive ids, in request order, and every refused item with why.

    Only the first `limit` items are considered; each later one is "over_limit". Of those,
    "unknown" names no archived model (or is not a string), "active" names a model in the pool
    now, "not_retired" names one whose latest archive outcome is not retired_by_evidence, and
    "repeated" repeats an earlier item; the rest are valid. Nothing here costs a call: a
    refused item is recorded and ignored.
    """
    latest = latest_archive_outcomes(archive)
    valid, invalid = [], []
    for position, item in enumerate(requested if isinstance(requested, list) else []):
        if position >= limit:
            why = "over_limit"
        elif not isinstance(item, str) or item not in latest:
            why = "unknown"
        elif item in active_ids:
            why = "active"
        elif item not in retired_now or latest[item]["outcome"] != "retired_by_evidence":
            why = "not_retired"
        elif item in valid:
            why = "repeated"
        else:
            valid.append(item)
            continue
        invalid.append({"id": item, "why": why})
    return valid, invalid


def mock_revive_request(archive, retired_now, best_model_id):
    """synthetic_v1's review-round revive list: the retired model with the highest log evidence
    at its latest retirement (valid) and the pool's best model (invalid: it is active)."""
    latest = latest_archive_outcomes(archive)
    retired = sorted(
        (model_id for model_id in retired_now
         if latest.get(model_id, {}).get("outcome") == "retired_by_evidence"),
        key=lambda model_id: (-latest[model_id]["reason"]["log_evidence_at_retirement"], model_id))
    return [*retired[:1], *([best_model_id] if best_model_id else [])]


def retirement_reason(log_evidence, cutoff, active_rounds, computed_round, n_obs):
    """The `reason` of a `retired_by_evidence` archive entry (`belief_packet.ARCHIVE_REASON_KEYS`).

    `active_rounds` is copied. The run keeps one growing list of active rounds per model; before
    archive review rounds a retired model never rejoined the pool, so holding that list was
    harmless. A model
    revived at an archive review round rejoins it, and its list grows again; a held list would then
    rewrite the earlier retirement entry, which the belief packet shows the proposer every round.
    """
    return {
        "log_evidence_at_retirement": float(log_evidence),
        "gap_to_cutoff_nats": cutoff - float(log_evidence),
        "active_rounds": list(active_rounds),
        "computed_round": computed_round,
        "n_obs": n_obs,
    }


ROUND_TIMINGS_FILE = "round_timings.jsonl"


def record_round_timing(output_dir, round_index, started_at_unix, started_perf_counter):
    """Round timing (2026-09-26): one observational row for a completed round.

    Appended to `round_timings.jsonl` in `count_inference.record_acquisition_timing`'s style
    (append, sort_keys, allow_nan=False): the round, its wall-clock start (time.time at the top
    of its loop iteration) and finish (time.time now, after its last work), and the monotonic
    elapsed seconds (perf_counter). Nothing hashes, binds or reads it; a round that raises
    writes no row.
    """
    finished_at_unix = time.time()
    row = {"round": int(round_index), "started_at_unix": float(started_at_unix),
           "finished_at_unix": float(finished_at_unix),
           "elapsed_seconds": float(time.perf_counter() - started_perf_counter)}
    with (Path(output_dir) / ROUND_TIMINGS_FILE).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _timed_fit(output, round_index, timeout_seconds, *fit_args, fit_guards="off", **fit_kwargs):
    fit_callable = fit
    if fit_guards != "off":
        fit_callable = partial(run_with_fit_guards, fit, fit_guards=fit_guards)
    if timeout_seconds is None:
        return fit_callable(*fit_args, **fit_kwargs)
    started = time.perf_counter()
    status = "completed"
    try:
        return run_with_fit_timeout(fit_callable, *fit_args, fit_timeout_seconds=timeout_seconds,
                                    **fit_kwargs)
    except FitTimeout:
        status = "fit_timeout"
        raise
    except Exception:
        status = "fit_failed"
        raise
    finally:
        record_fit_timing(output, fit_args[0], round_index, time.perf_counter() - started, status)


def _record_prediction_pool(output, fits, inputs, prediction, round_index, checkpoint,
                            observation_count):
    """Observe per-model means without changing a prediction's point-local pool."""
    weights = np.exp(model_logweights(fits))
    exclusions = prediction.get("prediction_excluded_models", [])
    models = []
    for index, (result, weight) in enumerate(zip(fits, weights, strict=True)):
        excluded = sorted({item["query_index"] for item in exclusions
                           if item["model_id"] == result.hypothesis.model_id})
        if excluded:
            means = np.full(len(inputs), np.nan)
            keep = [i for i in range(len(inputs)) if i not in excluded]
            if keep:
                means[keep] = predictive_means(result, np.asarray(inputs)[keep]).mean(0)
        else:
            means = predictive_means(result, inputs).mean(0)
        item = {"model_id": result.hypothesis.model_id, "weight": float(weight), "means": means}
        if exclusions:
            item["weights_by_q"] = np.asarray(prediction["model_weights"])[:, index].tolist()
        if excluded:
            item.update(excluded=True, excluded_q_indices=excluded)
        models.append(item)
    record_prediction_pool_summary(output, round_index, checkpoint, observation_count, models)


def official_kl_summary(fits, official_x, noiseless, relative_sd,
                        base_nodes=KL_QUADRATURE_BASE_NODES):
    """KL(truth || BMA) on the official Q(1000), for ONE fixed pool.

    `fits` is ONE checkpoint's active pool. By default `run` calls this once,
    after `finish()`, on `checkpoints[-1]["fits"]`, and fills `kl` on the
    FINAL checkpoint row alone (every earlier row keeps it blank); under
    `--kl-every-checkpoint` it calls it once per retained checkpoint, in
    checkpoint order, the final one last and with identical arguments. Either
    way nothing here can run before `finish()`: the official evaluation set
    is not available until then (`chembench_core.py`
    `OFFICIAL_EVALUATION_COUNT`/`OFFICIAL_EVALUATION_SEED` and the
    `adapter.sample_designs(...)` call this run makes for it, both after
    `session.finish()`).

    `p_BMA` is `positive_inference.log_predictive`'s existing double mixture -
    candidate weight (`logZ`, `positive_inference.py:475-484`) over an
    equally weighted particle mixture, exact closed-form lognormal/gamma
    component log-density (`positive_inference.py:445-450`) - batched over
    every one of the 1000 official queries AT ONCE (one `log_predictive` call
    per node count, not one call per query): `official_x` is the [query,
    input] matrix `log_predictive` already accepts, and the quadrature node
    matrix this function hands it is [query, node], so each query's own nodes
    are scored against that SAME query's row.
    """
    noiseless = np.asarray(noiseless, dtype=float)
    if noiseless.ndim != 1:
        raise ValueError("kl truth center must be a 1-D array")
    invalid = np.flatnonzero(~np.isfinite(noiseless) | (noiseless <= 0))
    if len(invalid):
        return {"kl": "undefined_truth_center", "kl_status": "undefined_truth_center",
                "kl_undefined_truth_center_points": invalid.tolist(),
                "kl_quadrature_error": None, "kl_truncated_mass": None,
                "kl_n_nodes": None, "kl_per_q_summary": None, "quadrature": None}

    def log_q_batch(y):
        """The pool's BMA log-density at the [query, node] matrix `y`, same shape as `y`."""
        density, _ = guarded_log_predictive(fits, official_x, y)
        return density

    base = censored_normal_kl_quadrature(noiseless, relative_sd, log_q_batch, base_nodes)
    fine = censored_normal_kl_quadrature(noiseless, relative_sd, log_q_batch, 2 * base_nodes)
    kl_base_mean = float(np.mean(base["kl"]))
    kl_fine_mean = float(np.mean(fine["kl"]))
    return {
        "kl": kl_fine_mean,
        "kl_quadrature_error": abs(kl_fine_mean - kl_base_mean),
        "kl_truncated_mass": float(np.max(fine["truncated_mass"])),
        "kl_n_nodes": 2 * base_nodes,
        "kl_per_q_summary": {
            "min": float(np.min(fine["kl"])),
            "median": float(np.median(fine["kl"])),
            "max": float(np.max(fine["kl"])),
        },
        "kl_per_q": fine["kl"],
        "kl_atom_mass_at_zero": float(np.max(fine["atom_mass"])),
        "quadrature": fine,
    }


def official_kl_particle_summary(fits, official_x, quadrature,
                                 max_bytes=KL_PARTICLE_SUMMARY_MAX_BYTES):
    """A double-precision recompute artifact for the final pool's `kl`.

    Per candidate: the particle predictive MEAN at the official Q(1000)
    (`positive_inference.predictive_means`, [particle, query]) and the
    family's own scale/shape particle array ("sigma" for lognormal, "shape"
    for gamma - `_KL_SCALE_PARTICLE_NAME`), stacked one candidate per row (one
    SMC run fixes the particle count for every candidate in a pool). If that
    would exceed `max_bytes`, falls back to the mixture's own log-density at
    the KL quadrature nodes already computed for `kl` (`quadrature["y"]`/
    `["log_q"]`, the finer, 2N-node grid) instead - a size-gated choice,
    with the projected byte count recorded either way, not silently applied.
    """
    weights = np.exp(model_logweights(fits))
    model_ids = np.array([result.hypothesis.model_id for result in fits])
    families = np.array([result.hypothesis.family for result in fits])
    projected_bytes = len(fits) * particle_count(fits[0]) * len(official_x) * 8
    _, exclusions = _prediction_pools(fits, official_x, None, "official_kl")
    if projected_bytes <= max_bytes and not exclusions:
        mean = np.stack([predictive_means(result, official_x) for result in fits])
        scale = np.stack(
            [
                np.asarray(
                    result.particles[_KL_SCALE_PARTICLE_NAME[result.hypothesis.family]],
                    dtype=float,
                )
                for result in fits
            ]
        )
        return {
            "format": "particle_mean_scale",
            "model_ids": model_ids,
            "families": families,
            "weights": weights,
            "mean": mean,
            "scale": scale,
            "projected_bytes": projected_bytes,
            "max_bytes": max_bytes,
        }
    return {
        "format": "mixture_log_density_at_quadrature_nodes",
        "model_ids": model_ids,
        "families": families,
        "weights": weights,
        "nodes": quadrature["y"],
        "log_density": quadrature["log_q"],
        "projected_bytes": projected_bytes,
        "max_bytes": max_bytes,
        **({"prediction_excluded_models": exclusions} if exclusions else {}),
        "note": ("point-local model exclusions require the mixture log-density artifact" if exclusions else
            f"per-candidate particle mean/scale would be {projected_bytes} bytes > "
            f"{max_bytes} (the recompute cap); storing the mixture log-density at the KL "
            "quadrature nodes instead"
        ),
    }


def write_official_kl_particle_summary(path, fits, official_x, quadrature,
                                       max_bytes=KL_PARTICLE_SUMMARY_MAX_BYTES):
    """Writes the double-precision recompute artifact (see `official_kl_particle_summary`)."""
    summary = official_kl_particle_summary(fits, official_x, quadrature, max_bytes)
    if summary["format"] == "particle_mean_scale":
        payload = {"mean": summary["mean"], "scale": summary["scale"]}
    else:
        payload = {"nodes": summary["nodes"], "log_density": summary["log_density"]}
    np.savez_compressed(
        path,
        format=summary["format"],
        model_ids=summary["model_ids"],
        families=summary["families"],
        weights=summary["weights"],
        projected_bytes=summary["projected_bytes"],
        max_bytes=summary["max_bytes"],
        **payload,
    )
    return summary


def official_checkpoint_scores(fits, inputs, seed, targets, round_index=None, *,
                               summary_output=None, observation_count=None):
    """Primary metric and plot-contract scores for ONE checkpoint on the official set.

    `targets` is the post-`finish()` reference block; nothing here is ever
    reachable during discovery. The predictive quantities all come from this
    run's OWN predictive samples (the plot contract's rule: the engine computes these, and a
    metric that is not defined is left blank rather than reported as 0).
    """
    summary = guarded_predictive_summary(fits, inputs, seed, round_index, "official")
    if summary_output is not None:
        _record_prediction_pool(summary_output, fits, inputs, summary, round_index,
                                "official", observation_count)
    observed = targets["observed"]
    logp, _ = guarded_log_predictive(fits, inputs, observed, round_index, "official")
    inside = (observed >= summary["lower90"]) & (observed <= summary["upper90"])
    return {
        **log_error_metrics(
            summary["mean"], targets["log_primary"], targets["log_primary_variance"]
        ),
        # The same `-mean log predictive` the original `nll_pdf` score uses,
        # on the official set instead of the runner's own checkpoint set.
        "nll": -float(logp.mean()),
        "coverage90": float(inside.mean()),
        "width90": float(np.mean(summary["upper90"] - summary["lower90"])),
        **({"prediction_excluded_models": summary["prediction_excluded_models"]}
           if "prediction_excluded_models" in summary else {}),
    }


def write_points(path, config, task, rows):
    """The plotting contract's point table for this run, one row per checkpoint.

    Modeled on `boxinggym_core.write_points`; the columns that differ are the
    ones the suite differs in - `x_unit` is `oracle_calls` (the
    baseline's own x unit), `task` is the manifest id
    verbatim (so it matches the baseline's `task_id` with no mapping table),
    `method` is the executed cell, and the primary metric is `log_error_metrics`'s.
    `crps`/`is_0.1` have no definition on this path and are therefore BLANK,
    never 0 (the plot contract's rule). `kl` IS defined, but by default only
    for the FINAL checkpoint row - the official evaluation set opens only
    after `finish()`, so `run` computes it once, for the final pool alone,
    and writes it into that one row's dict before calling this function;
    every earlier row has no "kl" key at all and stays BLANK here
    (`row.get`). Under `--kl-every-checkpoint` `run` scores every
    retained pool instead, so every row carries a "kl" key and none is blank;
    the COLUMN SET and this blank rule are unchanged either way, and a value
    that came back as the `"undefined_truth_center"` string is written as
    that string, on whichever rows carry it.
    """
    import csv

    proposer = (
        # catalog mode never constructs a proposer of any kind, mock included,
        # so recording "mock" there would overstate what ran.
        "none"
        if config["structure_proposals"] == "catalog"
        else ("mock" if "mock" in config["proposer"] else f"llm:{config['llm_model']}")
    )
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=(*CONTRACT_POINTS_COLUMNS, *compat_points_metadata(config)), lineterminator="\n"
        )
        writer.writeheader()
        for line, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    **compat_points_metadata(config),
                    "suite": "chembench",
                    "task": task["id"],
                    "method": config["cell"],
                    "proposer": proposer,
                    "seed": config["seed"],
                    "x": row["observation_count"],
                    "x_unit": "oracle_calls",
                    "y_primary": row[PRIMARY_METRIC],
                    "y_name": PRIMARY_METRIC,
                    "y_label": PRIMARY_METRIC_LABEL,
                    "nll": row["nll"],
                    "coverage90": row["coverage90"],
                    "width90": row["width90"],
                    "crps": "",
                    "is_0.1": "",
                    "kl": row.get("kl", ""),
                    "status": config["points_status"],
                    "label": config["points_label"],
                    "run_id": Path(config["output"]).name,
                    "source_path": (
                        f"{config['output']}/official_scores_after_finish.json"
                        f"#rounds[{line - 1}](round={row['round']})"
                    ),
                    "variant": config["points_variant"],
                    "code_sha": config["source_sha256"]["chembench_core.py"],
                    # Contract column 23: a Brier score needs a binary
                    # outcome, and this runner's outcome is a positive
                    # continuous rate - BLANK, never 0 (the plot contract's rule).
                    "brier": "",
                    "nmsle": row["nmsle"],
                    "rmsle": row["rmsle"],
                }
            )



D0_POLICIES = ("random4", "fixed3")
FIXED3_CONCENTRATIONS = (0.03, 0.3, 3.0)


def fixed3_conditions(overrides=None):
    """Our declared unperturbed rule; MDA does not disclose six numerical values."""
    from autoscilab.oracle.chembench import _T_REF, CHEM_INPUT_BOUNDS
    values = {"C_I": 0.0, "C_P": 0.0,
              "C_B": float(np.sqrt(np.prod(CHEM_INPUT_BOUNDS["C_B"]))),
              "Enz": float(np.sqrt(np.prod(CHEM_INPUT_BOUNDS["Enz"]))),
              "T": float(_T_REF), "pH": 7.0}
    if overrides is not None:
        if not isinstance(overrides, dict) or set(overrides) - set(values):
            raise ValueError("fixed3 overrides may contain only C_I, C_B, C_P, Enz, T and pH")
        values.update(overrides)
    return values


def fixed3_designs(overrides=None):
    """Calculate initial-only conditions from the declared rule, with explicit overrides."""
    from autoscilab.oracle.chembench import CHEM_INPUT_BOUNDS
    conditions = fixed3_conditions(overrides)
    required = set(CHEM_INPUT_BOUNDS) - {"C_A"}
    values = {}
    for name in sorted(required):
        value = conditions[name]
        lower, upper = CHEM_INPUT_BOUNDS[name]
        if type(value) not in (int, float) or not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"invalid fixed3 condition: {name}")
        values[name] = float(value)
    return [{"C_A": value, **values} for value in FIXED3_CONCENTRATIONS]


class InputNameMaskedSession:
    """Rename declared coordinates at the adapter boundary, preserving their order."""

    def __init__(self, session, sampler):
        from threading import Lock

        self._session = session
        self._sampler = sampler
        self._schema = session.schema()
        names = self._schema["input_names"]
        self.input_name_map = {f"X{i + 1}": name for i, name in enumerate(names)}
        self._aliases = {name: alias for alias, name in self.input_name_map.items()}
        self._model_ids = {}
        self._model_ids_lock = Lock()

    def rng_model_id(self, hypothesis, sidecar):
        """Input-name masking: native identity is only an RNG/local-tie key, never a model ID."""
        import ast

        from .count_models import canonical_model_id

        with self._model_ids_lock:
            key = hypothesis.model_id
            if key not in self._model_ids:
                names = {**self.input_name_map,
                         **{f"{alias}z": f"{name}z" for alias, name in self.input_name_map.items()}}
                tree = ast.parse(hypothesis.expression, mode="eval")
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        node.id = names.get(node.id, node.id)
                rng_key = canonical_model_id(ast.unparse(tree), hypothesis.family, hypothesis.parameters)
                append_json(sidecar, {"model_id": key, "rng_model_id": rng_key})
                self._model_ids[key] = rng_key
            return self._model_ids[key]

    @property
    def calls(self):
        """Observation calls the wrapped session has made."""
        return self._session.calls

    def schema(self):
        """The wrapped session's schema with every input renamed to its alias."""
        return {**self._schema,
                "input_names": list(self.input_name_map),
                "input_bounds": {self._aliases[name]: bounds
                                 for name, bounds in self._schema["input_bounds"].items()},
                "log_sampled_inputs": [self._aliases[name]
                                       for name in self._schema["log_sampled_inputs"]]}

    def mask_design(self, design):
        """Rename a design keyed by native input names to the aliases."""
        if set(design) != set(self._aliases):
            raise ValueError("input-name masking requires exactly the declared coordinates")
        return {self._aliases[name]: value for name, value in design.items()}

    def unmask_design(self, design):
        """Rename a design keyed by aliases back to the native input names."""
        if set(design) != set(self.input_name_map):
            raise ValueError("input-name masking requires exactly the declared aliases")
        return {self.input_name_map[name]: value for name, value in design.items()}

    def sample_designs(self, count, seed):
        """Draw `count` designs with the native sampler and `seed`, returned under the aliases."""
        return [self.mask_design(row) for row in self._sampler(count, seed)]

    def observe(self, design):
        """Observe an alias-keyed design; the returned observation's `x` is alias-keyed too."""
        observation = self._session.observe(self.unmask_design(design))
        return {**observation, "x": self.mask_design(observation["x"])}

    def finish(self):
        """Close the wrapped session (the reference values open only after this)."""
        return self._session.finish()

    def reference_values(self, designs):
        """Reference values of alias-keyed designs, computed by the wrapped session."""
        return self._session.reference_values([self.unmask_design(row) for row in designs])


#: Constraint disclosure, second edition (a project notice of 2026-09-25; it
#: replaces the first edition, whose `reserved_names` union a relaunch showed the
#: model reads as "names reserved for parameters"): the declaration every ChemBench
#: config.json carries, in free and catalogue mode alike. The rule
#: it states is the one every candidate already meets - `count_models.validate_expression`
#: refuses a coefficient named after an interface input or a reserved name - and a ChemBench
#: packet now states it as a sentence plus the forbidden lists (`_disclosed_constraints`).
#: Third edition (a project decision of 2026-09-25, after a canary run with the second
#: edition alone): the same packet also states the prior-probe admission contract - what
#: `_prior_probe` checks, where, with which priors and how
#: many draws - as engine mechanics only. It carries no strategy, no preference, no example and
#: no count from any run, and it claims nothing about points the probe does not evaluate. It is
#: one common execution condition of every ChemBench arm, not a ladder rung.
CONSTRAINTS_DISCLOSURE = "cic3_v1"
COEFFICIENT_NAME_RULE = (
    "Each key of parameters is the name of a coefficient you introduce: an unknown constant "
    "the engine fits. A coefficient name must be a new identifier; it must not be an input "
    "name, a derived input name, or a function or reserved name listed below. Inputs are "
    "used inside expression only; never list an input in parameters. The role word "
    "(intercept, signed, positive, shape) describes the coefficient's prior, not an input; "
    "positive means the coefficient's value is positive. The expression must contain at "
    "least one coefficient."
)
PRIOR_PROBE_RULE = (
    "expression is the argument of the fixed log link: the engine computes the mean as "
    "response_unit times the exponential of expression, and applies that exponential itself. "
    f"Before any fit, the engine draws {PRIOR_PROBE_SAMPLES} samples of every coefficient from "
    "the prior its role word declares in constraints.roles and evaluates the mean at every "
    f"point of a {PRIOR_PROBE_GRID_POINTS}-point diagonal grid across the declared input "
    "ranges, at every corner of the declared input ranges, and at the disclosed observation "
    "inputs. A candidate is admitted only if the mean is finite and strictly positive at every "
    "one of those points for every draw; a function or operation that is undefined at any of "
    "those points fails admission. The probe checks those declared points and draws only."
)


def _prior_probe_rule(constraints, interface):
    """Constraint disclosure, third edition: the `prior_probe_rule` value, read from the
    probe's own constants, the packet's declared role priors and the interface's declared
    inputs - nothing else. The corner count is the product over the declared ranges' two ends,
    the set `_prior_probe` evaluates. A non-log link is refused rather than described wrongly."""
    if interface.get("link") != "log":
        raise ValueError("prior_probe_rule states the log link; this interface declares "
                         + repr(interface.get("link")))
    return {
        "rule": PRIOR_PROBE_RULE,
        "link": "log",
        "mean": "response_unit * exp(expression)",
        "coefficient_priors": dict(constraints["roles"]),
        "prior_draws": PRIOR_PROBE_SAMPLES,
        "checked_points": {
            "diagonal_grid_points": PRIOR_PROBE_GRID_POINTS,
            "range_corners": 2 ** len(interface["inputs"]),
            "disclosed_observation_inputs": True,
        },
        "admission": "finite and strictly positive mean at every checked point for every draw",
    }


def _disclosed_constraints(constraints, interface, grammar_names):
    """Constraint disclosure, second and third editions: `cell_constraints`' dict plus the
    `coefficient_name_rule` and `prior_probe_rule` keys.

    `reserved_names` stays the engine's own tuple (the first edition's union is withdrawn). The
    new key carries the rule sentence and the three forbidden lists, all read from this run's
    interface: `inputs` in declaration order, `derived_inputs` in declaration order, and the
    engine's reserved tuple. Their union is exactly `grammar_names` + reserved, the namespace
    `validate_expression` refuses as coefficient names, and that equality is asserted here.
    Every other key is untouched; the helper takes no arm or prompt input, so every arm's
    packet states the same rule. Defined here, after the prediction helpers, so no line those
    helpers' callers are pinned to moves.
    """
    reserved = list(constraints["reserved_names"])
    inputs = [item["name"] for item in interface["inputs"]]
    derived = list(interface.get("derived_inputs", {}).keys())
    names = tuple(grammar_names)
    if len(set(names)) != len(names):
        raise ValueError("grammar input names must be distinct to be disclosed")
    if set(names) != set(inputs) | set(derived):
        raise ValueError("coefficient_name_rule lists must equal the grammar input namespace")
    if set(names) & set(reserved):
        raise ValueError("grammar input names and reserved names must be disjoint")
    rule = {"rule": COEFFICIENT_NAME_RULE,
            "not_allowed_as_coefficient_names": {"inputs": inputs, "derived_inputs": derived,
                                                 "functions_and_reserved": reserved}}
    # Third edition: the prior-probe contract beside the naming rule; same inputs, same bytes for
    # every arm.
    return {**constraints, "coefficient_name_rule": rule,
            "prior_probe_rule": _prior_probe_rule(constraints, interface)}


#: Stage slots (a project decision of 2026-09-25): ChemBench's stage slots. Each file under
#: the proposer's prompts directory is a packaged copy of the text it names; in this edition
#: every copy is byte-identical to the module constant it replaces (tests assert it), so not
#: one request byte moves. The slot names and digests are written to config.json and to each
#: transport record, never into a request. The role-prior contract file names the role-label priors
#: the engine applies (prior_slot "none" = no extra LLM prior prompt, not "no prior").
STAGE_SLOT_SCHEMA = "chembench_stage_slots_v1"
STAGE_SLOT_FILES = {
    "burst_paragraph": "slot_burst_paragraph_v1.txt",
    "burst_refill_paragraph": "slot_burst_refill_paragraph_v1.txt",
    "repair_instruction": "slot_repair_v1.txt",
    "repair_instruction_empty": "slot_repair_v1_empty.txt",
    "role_prior_contract": "slot_fixed_role_prior_v1.json",
}


def _chembench_stage_slots(prompt_block, *, init_slot, init_prompt_sha256=None, expand=True):
    """The StageSlots this run's proposer calls carry, and its record block.

    `prompt_block` is the selected arm file's identity (`prompt_identity()`); `init_slot`
    names where round 0's pool comes from ("in_cell_arm_burst" or "catalogue", or the frozen
    pool track's "pool_generator_v1" - the external generation stage, with
    `init_prompt_sha256` naming the generator file). `expand` False is the generation run,
    which has no expand stage.
    Reads files only; takes no arm, seed, cell or pool input beyond the prompt identity it
    records. Called as commit 1 calls it, the record keeps commit 1's key set.
    """
    from .count_proposer import PROMPT_DIRECTORY, StageSlots

    raw = {key: (PROMPT_DIRECTORY / name).read_bytes() for key, name in STAGE_SLOT_FILES.items()}
    digest = {key: hashlib.sha256(value).hexdigest() for key, value in raw.items()}
    record = {
        "stage_slot_schema": STAGE_SLOT_SCHEMA,
        "init_slot": init_slot,
        **({} if init_prompt_sha256 is None else {"init_prompt_sha256": init_prompt_sha256}),
        "burst_paragraph_sha256": digest["burst_paragraph"],
        "burst_refill_paragraph_sha256": digest["burst_refill_paragraph"],
        "expand_slot": prompt_block["prompt_version"] if expand else None,
        "expand_prompt_sha256": prompt_block["prompt_sha256"] if expand else None,
        "repair_slot": "repair_v1",
        "repair_prompt_sha256": digest["repair_instruction"],
        "repair_empty_prompt_sha256": digest["repair_instruction_empty"],
        "prior_slot": "none",
        "role_prior_contract": "fixed_role_prior_v1",
        "role_prior_contract_sha256": digest["role_prior_contract"],
    }
    slots = StageSlots(
        burst_paragraph=raw["burst_paragraph"].decode("utf-8"),
        burst_refill_paragraph=raw["burst_refill_paragraph"].decode("utf-8"),
        repair_instruction=raw["repair_instruction"].decode("utf-8"),
        repair_instruction_empty=raw["repair_instruction_empty"].decode("utf-8"),
        metadata=record,
    )
    return slots, record


#: The frozen initial pool track (project decisions of 2026-09-25). One pool per (task,
#: seed) is made ONCE, outside the registered cells, by the arm-neutral generator
#: (`--pool-generator`); every cell of that (task, seed) - the eight prompt arms and the cell
#: 100 random baseline - starts from the same file, and round 0 inside a cell makes no LLM
#: call. The generator writes the `frozen_pool` block below into its `proposals/t00_burst.json`;
#: the reader accepts that file alone, so a canary or arm burst file is never a frozen pool.
FROZEN_POOL_TRACK = "chembench_frozen_pool_v1"
FROZEN_POOL_SCHEMA = "chembench_frozen_pool_v1"
FROZEN_POOL_SOURCE = "file_burst"
FROZEN_POOL_GENERATION = "external_frozen_stage"
#: The burst writer's statuses (the admission loop in `run`): the first proposal of an id
#: that is fitted, and each later proposal of the same id.
FROZEN_POOL_FITTED_STATUS = "fitted"
FROZEN_POOL_DUPLICATE_STATUS = "duplicate_not_refitted"


def parse_commit_sha(text):
    """Frozen pool track: `--generator-code-commit`, the launcher's full git commit."""
    if len(text) != 40 or any(char not in "0123456789abcdef" for char in text):
        raise argparse.ArgumentTypeError("--generator-code-commit takes a full 40-hex commit")
    return text


def frozen_pool_id(task, seed):
    """The pool of one (task, seed): the six registered pool rows share this form."""
    return f"{FROZEN_POOL_TRACK}__{task['domain']}__{task['difficulty']}__s{seed}"


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _generator_identity(name):
    """A packaged generator's file digest and the two composed round-0 requests it sends."""
    from .count_proposer import BURST_PARAGRAPH, BURST_REFILL_PARAGRAPH, POOL_GENERATOR_FILES

    if name not in POOL_GENERATOR_FILES:
        raise ValueError(f"unknown pool generator {name!r}; packaged: {sorted(POOL_GENERATOR_FILES)}")
    text = POOL_GENERATOR_FILES[name].read_bytes()
    return {
        "generator_prompt_sha256": hashlib.sha256(text).hexdigest(),
        "composed_init_prompt_sha256": hashlib.sha256(
            (text.decode("utf-8") + BURST_PARAGRAPH).encode("utf-8")).hexdigest(),
        "composed_init_refill_prompt_sha256": hashlib.sha256(
            (text.decode("utf-8") + BURST_REFILL_PARAGRAPH).encode("utf-8")).hexdigest(),
    }


def frozen_pool_interface():
    """Frozen pool track: `run`'s ChemBench interface without a session, for the driver's pool
    preflight. The session schema is AutoSciLab's static declaration, so this makes the
    calls `run` makes: its inputs and ranges, and the z-scored derived inputs over the same
    frozen public-pool transform (seed stream chem_public_pool_inputs, 256 designs)."""
    from autoscilab.oracle.chembench import CHEM_INPUT_BOUNDS, CHEM_INPUT_VARS

    from bayes_model_discovery.adapters import chembench_observation as adapter

    names = list(CHEM_INPUT_VARS)
    pool = adapter.sample_designs(256, seed_for("chem_public_pool_inputs", 20260906))
    transform = InputTransform.from_public_pool(np.array([[row[n] for n in names] for row in pool]))
    return task_interface(
        "positive",
        [{"name": n, "range": list(CHEM_INPUT_BOUNDS[n])} for n in names],
        {f"{n}z": f"({n}-{float(c)!r})/{float(s)!r}"
         for n, c, s in zip(names, transform.center, transform.scale, strict=True)},
        (0.0, None),
    )


@dataclass(frozen=True)
class FrozenPoolFile:
    """A validated frozen pool: members in the file's `after_ids` order, and its binding."""

    hypotheses: tuple
    sha256: str
    name: str
    block: dict
    public_observations: list

    def metadata(self):
        """The runtime-settings binding (the three initial-pool keys, the ChemBench file source)."""
        return {"initial_pool_source": FROZEN_POOL_SOURCE, "initial_pool_sha256": self.sha256,
                "initial_pool_count": len(self.hypotheses)}


def _utc_now():
    from datetime import datetime

    return datetime.now(UTC).isoformat()


def _frozen_pool_generation_receipt(out, block, args, attempt_records, proposer, *,
                                    generation_started_at):
    """Frozen pool track: one generation run's pool receipt (as decided).

    Written after the round-0 burst of a `--pool-generator` run. The pool file is that run's
    `proposals/t00_burst.json`; its digest, size and wave-0 packet constraints are read back
    from the written files. `paid_usd` is null here: the driver's cost ledger holds the
    declared cost (paid_api_calls x rate) and the provider's list price is the publisher's.
    """
    import socket

    burst_bytes = (out / "proposals" / "t00_burst.json").read_bytes()
    burst = json.loads(burst_bytes)
    constraints = json.loads((out / "packets" / "t00_burst.json").read_text())["constraints"]
    transports = [row["transport"] for row in attempt_records]
    usage_total, usage_complete = {}, True
    for record in transports:
        if record.get("calls"):
            usage_complete &= (bool(record.get("usage_total_complete", False))
                               and len(record.get("responses", [])) == record["calls"])
            for key, value in (record.get("usage_total") or {}).items():
                usage_total[key] = usage_total.get(key, 0) + value
    commit = getattr(args, "generator_code_commit", None)
    return {
        "schema": "chembench_frozen_pool_generation_receipt_v1",
        "track_id": block["track_id"],
        "pool_id": block["pool_id"],
        "task": block["task"],
        "stratum": block["stratum"],
        "law_version": block["law_version"],
        "seed": block["seed"],
        "cell": block["cell"],
        "generator_prompt_sha256": block["generator_prompt_sha256"],
        "generator_code_commit": commit,
        "generator_code_commit_source": "launcher" if commit else "absent",
        "packet_constraints_sha256": _canonical_sha256(constraints),
        "pool_file": "proposals/t00_burst.json",
        "pool_file_sha256": hashlib.sha256(burst_bytes).hexdigest(),
        "pool_size": len(burst["after_ids"]),
        "requested": block["requested"],
        "waves": len(burst["waves"]),
        "provider": "mock" if proposer is None else args.proposer,
        "model": None if proposer is None else args.llm_model,
        "returned_models": sorted({record["returned_model"] for record in transports
                                   if record.get("returned_model")}),
        "host": socket.gethostname(),
        "generation_started_at": generation_started_at,
        "generation_finished_at": _utc_now(),
        "paid_calls": 0 if proposer is None else proposer.attempts,
        "paid_usd": None,
        "paid_usd_source": "driver_cost_ledger",
        "usage_total": usage_total,
        "usage_total_complete": usage_complete,
        # As decided: the init stage's parts and the composed request.
        "pool_generator_sha256": block["generator_prompt_sha256"],
        "burst_paragraph_sha256": block["burst_paragraph_sha256"],
        "burst_refill_paragraph_sha256": block["burst_refill_paragraph_sha256"],
        "coefficient_name_rule_sha256": _canonical_sha256(constraints["coefficient_name_rule"]),
        "prior_probe_rule_sha256": _canonical_sha256(constraints["prior_probe_rule"]),
        "composed_init_prompt_sha256": block["composed_init_prompt_sha256"],
        "composed_init_refill_prompt_sha256": block["composed_init_refill_prompt_sha256"],
    }


class FrozenPoolMemberFitFailure(RuntimeError):
    """Frozen pool track: a frozen pool member that did not fit at round 0 fails the whole cell."""

    def __init__(self, model_id, cause):
        super().__init__(f"frozen pool member {model_id} failed to fit at round 0 "
                         f"({type(cause).__name__}: {cause}); the cell does not start from a "
                         "smaller pool")
        self.model_id = model_id


def read_chembench_initial_pool_file(filename, *, task, seed, cell, family, d0_policy,
                                     d0_fixed_inputs, input_names):
    """Frozen pool track: read one generator-made ChemBench pool, or refuse the whole file.

    Accepts exactly a ChemBench `proposals/t00_burst.json` written by `--pool-generator`:
    the model-id schema and notation, the burst refill policy, `initial_catalogue` off, the
    `frozen_pool` block (packaged generator digest, stage-slot schema, constraints
    disclosure, and the pool's own task, stratum, seed, family and D0 policy), and
    `after_ids` bound one to one to fitted events. Each member is rebuilt as a
    `ScalarHypothesis` over this run's `input_names` and its model_id recomputed. There is no
    filtering, no quality selection and no regeneration: any bad entry refuses the file.
    CORE's `count_models.read_initial_pool_file` is count-only and is not used here.
    """
    from .count_models import INITIAL_BURST_REFILL_POLICY
    from .model_identity import require_model_id_schema

    path = Path(filename)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("frozen pool file is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("frozen pool file must be a ChemBench burst object")  # noqa: TRY004 - Serialized-input validation uses ValueError.
    require_model_id_schema(payload)
    if payload.get("notation_normal_form") != NOTATION_NORMAL_FORM:
        raise ValueError("frozen pool file notation_normal_form differs from this engine's")
    if payload.get("initial_burst_refill_policy") != INITIAL_BURST_REFILL_POLICY:
        raise ValueError(f"frozen pool requires initial_burst_refill_policy "
                         f"{INITIAL_BURST_REFILL_POLICY!r}; got "
                         f"{payload.get('initial_burst_refill_policy')!r}")
    if payload.get("initial_catalogue") != "off":
        raise ValueError("frozen pool requires initial_catalogue off")
    block = payload.get("frozen_pool")
    if not isinstance(block, dict) or block.get("schema") != FROZEN_POOL_SCHEMA:
        raise ValueError("frozen pool file has no frozen_pool block: only a --pool-generator "
                         "burst is a frozen pool (a canary or arm burst is not)")
    generator = block.get("init_slot")
    if generator != "pool_generator_v1":
        raise ValueError(f"frozen pool init_slot must be 'pool_generator_v1'; got {generator!r}")
    identity = _generator_identity(generator)
    expected = {
        "track_id": FROZEN_POOL_TRACK, "pool_id": frozen_pool_id(task, seed),
        "task": task["domain"], "stratum": task["difficulty"],
        "law_version": task["law_version"], "seed": seed, "family": family,
        "d0_policy": d0_policy, "d0_fixed_inputs": d0_fixed_inputs,
        "stage_slot_schema": STAGE_SLOT_SCHEMA,
        "constraints_disclosure": CONSTRAINTS_DISCLOSURE,
        **identity,
    }
    for key, value in expected.items():
        if block.get(key) != value:
            raise ValueError(f"frozen pool {key} {block.get(key)!r} does not match this run's "
                             f"{value!r}")
    ids, events = payload.get("after_ids"), payload.get("events")
    if (not isinstance(ids, list) or not ids or not all(isinstance(key, str) and key for key in ids)
            or len(set(ids)) != len(ids)
            or not isinstance(events, list) or not all(isinstance(event, dict) for event in events)):
        raise ValueError("frozen pool requires distinct after_ids and candidate events arrays")
    public = payload.get("public_observations")
    if not isinstance(public, list) or not public:
        raise ValueError("frozen pool file records no public observations")
    hypotheses = []
    for index, key in enumerate(ids):
        same_id = [event for event in events if event.get("model_id") == key]
        fitted = [event for event in same_id if event.get("status") == FROZEN_POOL_FITTED_STATUS]
        if len(fitted) != 1:
            raise ValueError(f"frozen pool after_id {key!r} needs exactly one fitted event; "
                             f"found {len(fitted)}")
        strays = sorted({str(event.get("status")) for event in same_id if event is not fitted[0]
                         and event.get("status") != FROZEN_POOL_DUPLICATE_STATUS})
        if strays:
            raise ValueError(f"frozen pool after_id {key!r} has non-duplicate events beside its "
                             f"fitted record: {strays}")
        entry = fitted[0]
        parameters = entry.get("parameters")
        if (not isinstance(entry.get("expression"), str) or not isinstance(entry.get("family"), str)
                or not isinstance(parameters, dict)
                or not all(isinstance(k, str) and isinstance(v, str) for k, v in parameters.items())):
            raise ValueError(f"frozen pool entry {index} requires expression/family/parameters")
        if entry["family"] not in CHEM_ALLOWED_FAMILIES:
            raise ValueError(f"frozen pool entry {index}: family {entry['family']!r} is not a "
                             f"ChemBench family {list(CHEM_ALLOWED_FAMILIES)}")
        refusal = family_refusal(cell, entry["family"], family)
        if refusal is not None:
            raise ValueError(f"frozen pool entry {index}: {refusal}")
        try:
            hypothesis = ScalarHypothesis(entry["expression"], entry["family"],
                                          tuple(parameters.items()), tuple(input_names))
        except ValueError as exc:
            raise ValueError(f"frozen pool entry {index}: {exc}") from exc
        if hypothesis.model_id != key:
            raise ValueError(f"frozen pool entry {index}: reconstructed model_id mismatch")
        hypotheses.append(hypothesis)
    return FrozenPoolFile(tuple(hypotheses), hashlib.sha256(raw).hexdigest(), path.name, block,
                          public)


def run(args):
    """Run one ChemBench cell from the parsed command line `args`; records go to `args.output`."""
    # parse_round_list and set_composition_paragraph are imported here
    # rather than at the top of the module (a convention kept from the development tree,
    # where parity tests pinned line numbers).
    from .core_count import parse_round_list
    from .count_proposer import set_composition_paragraph
    from .run_label_policy import validate_repair_attempts
    repair_attempts = validate_repair_attempts(
        getattr(args, "repair_attempts", 1), bundle="chembench")
    args.design_policy = resolve_epig_policy(args.cell, getattr(args, "design_policy", None))
    # Unified protocol: `model_eig` is the protocol's name for the registered `structure_eig`; the
    # namespace, the receipt and every record carry the registered name from here on.
    requested_design_policy = args.design_policy
    args.design_policy = CHEM_EIG_PART_ALIASES.get(args.design_policy, args.design_policy)
    initial_proposals = getattr(args, "initial_proposals", 0)
    catalogue_mode = getattr(args, "initial_catalogue", "on")
    if hasattr(args, "initial_proposals") and (
            type(initial_proposals) is not int or not 1 <= initial_proposals <= 24):
        raise ValueError("--initial-proposals must be 1..24")
    if initial_proposals > ACTIVE_MAXIMUM:
        raise ValueError("--initial-proposals must not exceed the active maximum")
    if catalogue_mode not in ("on", "off"):
        raise ValueError("--initial-catalogue must be on or off")
    # Refused here, before any output, like every option above.
    composition = getattr(args, "composition_paragraph", "v1")
    if composition not in ("v1", "v2"):
        raise ValueError("--composition-paragraph must be v1 or v2")
    if composition == "v2" and int(getattr(args, "proposal_limit", 1)) != 3:
        raise ValueError("--composition-paragraph v2 is written for --proposal-limit 3")
    review_rounds = (parse_round_list(args.archive_review_rounds, (), maximum=args.rounds)
                     if hasattr(args, "archive_review_rounds") else ())
    if review_rounds and composition != "v2":
        raise ValueError("--archive-review-rounds needs --composition-paragraph v2 and "
                         "--proposal-limit 3")
    mock_schedule_name = getattr(args, "mock_schedule", MOCK_SCHEDULES[0])
    if mock_schedule_name not in MOCK_SCHEDULES:
        raise ValueError(f"--mock-schedule must be one of {MOCK_SCHEDULES}")
    if hasattr(args, "mock_schedule") and args.proposer != "mock":
        raise ValueError("--mock-schedule selects the mock proposer's forms; it needs --proposer mock")
    # Frozen pool track: a frozen pool file (core_count.py:1179-1183's rule) and the generation run.
    pool_file = getattr(args, "initial_pool_file", None)
    pool_generator = getattr(args, "pool_generator", None)
    mask_input_names = getattr(args, "mask_input_names", False)
    if mask_input_names:
        if pool_file is not None or pool_generator is not None:
            raise ValueError("--mask-input-names excludes --initial-pool-file and --pool-generator")
        if catalogue_mode != "off":
            raise ValueError("--mask-input-names requires --initial-catalogue off")
    if pool_file is not None and not pool_file:
        raise ValueError("--initial-pool-file must be a nonempty path")
    if pool_generator is not None:
        if pool_file:
            raise ValueError("--pool-generator makes a frozen pool; it cannot also read one")
        if (getattr(args, "prompt_version", None) is not None
                or getattr(args, "prompt_variant", None) is not None
                or getattr(args, "priors", "off") == "llm"):
            raise ValueError("--pool-generator is the generation run's own system text; it "
                             "excludes --prompt-version, --prompt-variant and --priors llm")
        if catalogue_mode != "off" or not initial_proposals:
            raise ValueError("--pool-generator runs the round-0 burst alone: it needs "
                             "--initial-catalogue off and --initial-proposals")
        if getattr(args, "d0_policy", "random4") != "fixed3":
            raise ValueError("--pool-generator requires --d0-policy fixed3, the registered "
                             "cells' D0")
    if pool_file and (catalogue_mode != "off" or initial_proposals):
        raise ValueError("--initial-pool-file requires --initial-catalogue off and no --initial-proposals")
    if catalogue_mode == "off" and not initial_proposals and not pool_file:
        raise ValueError("--initial-catalogue off needs --initial-proposals")
    if (initial_proposals or pool_file) and validate_cell(args.cell, args.structure_proposals)[0] != "1":
        raise ValueError("ChemBench initial proposals need a structure-expansion cell")

    from .design_search import (
        FALLBACK_POLICY,
        SearchTransform,
        declared_settings,
        pool_diagnostics,
        projected_spsa,
        refuse_pool_above_two_dimensions,
        strip_cli_settings,
    )

    search_settings = declared_settings(args, ACQUISITION_STREAM)
    ascent = search_settings.get("design_search") == "ascent"
    explicit_pool = search_settings.get("candidate_source") == "pool"
    fit_timeout_seconds = validate_fit_timeout(getattr(args, "fit_timeout_seconds", None))
    fit_guards = getattr(args, "fit_guards", "off")
    if fit_guards not in ("off", "v1"):
        raise ValueError("fit_guards must be off or v1")
    d0_policy = getattr(args, "d0_policy", "random4")
    if d0_policy not in D0_POLICIES:
        raise ValueError("unknown D0 policy")
    d0_overrides = getattr(args, "d0_fixed_inputs", None)
    d0_conditions = d0_overrides
    if d0_policy == "random4" and d0_conditions is not None:
        raise ValueError("D0 conditions are only valid for fixed3")
    fixed_initial = fixed3_designs(d0_overrides) if d0_policy == "fixed3" else None
    if fixed_initial is not None:
        d0_conditions = {key: value for key, value in fixed_initial[0].items() if key != "C_A"}
    d0_count = 3 if fixed_initial is not None else 4
    # Checked unconditionally, before any output exists, regardless of
    # --structure-proposals - the same CORE rule (core_count.run/
    # boxinggym_core.run), so a --proposer anthropic|openai typo is never a
    # silent no-op even on a catalog-mode run that would never call it.
    if args.proposer == "anthropic" and not args.allow_paid_api:
        raise ValueError("Anthropic proposals require explicit --allow-paid-api authorization")
    if args.proposer in ("openai", "openai_compat") and not args.allow_paid_api:
        raise ValueError("OpenAI proposals require explicit --allow-paid-api authorization")
    # --run-label. Resolved HERE,
    # before any output directory exists, so an inadmissible label leaves
    # nothing behind. `args.run_label` itself is NOT overwritten: it stays the
    # caller's raw request (None when the flag was not typed), because the
    # byte-invariance rule keys off "was a label actually stated"
    # and because `**vars(args)` should record what was asked for.
    label_metadata = strip_cli_settings(compat_config(args))
    stated_run_label = getattr(args, "run_label", None)
    # Prompt selection: the system prompt is SELECTED here, before the receipt, so the label
    # binds the text this run actually sends instead of the module default. This runner has no
    # `--priors` switch, so the default is the v2.0 file it always read. The two keys ride a
    # receipt-only namespace, exactly as `active_maximum` does, so `compat_config(args)`
    # keeps the key set every earlier run recorded and `**prompt_identity()` in `config`
    # below recomputes the same pair from the same source.
    _, label_prompt = select_run_prompt(args)
    if pool_generator is not None:
        # Frozen pool track: the generation run sends the packaged generator file, as stored,
        # in place of an arm file; its name and digest are identity metadata only.
        from .count_proposer import select_pool_generator

        select_pool_generator(pool_generator)
        label_prompt = prompt_identity()
    # Frozen pool track: the frozen pool's binding, read before any output exists (the full member
    # validation needs this run's interface and follows once it is built).
    pool_prefix = None
    if pool_file:
        pool_raw = Path(pool_file).read_bytes()
        try:
            pool_ids = json.loads(pool_raw).get("after_ids")
        except (ValueError, UnicodeError, AttributeError) as exc:
            raise ValueError("frozen pool file is not a JSON burst object") from exc
        if not isinstance(pool_ids, list) or not pool_ids:
            raise ValueError("frozen pool file has no after_ids")
        pool_prefix = {"initial_pool_source": FROZEN_POOL_SOURCE,
                       "initial_pool_sha256": hashlib.sha256(pool_raw).hexdigest(),
                       "initial_pool_count": len(pool_ids)}
        label_prompt = {**label_prompt, **pool_prefix}
    # Multi-candidate proposals: candidates asked for per round, resolved here because a
    # k > 1 run sends a COMPOSED system text (the selected file's bytes plus one
    # paragraph) and the receipt binds that text's digest beside the file's own.
    # k = 1 composes nothing, declares nothing, and leaves every existing
    # receipt byte-identical.
    proposal_limit = validate_proposal_limit(int(getattr(args, "proposal_limit", 1)))
    # The composition paragraph this run composes with, selected before any digest
    # is read (v1, the default, is every earlier run's).
    set_composition_paragraph(composition)
    if proposal_limit != 1:
        label_prompt = {**label_prompt, "composed_prompt_sha256":
                        composed_prompt_identity(proposal_limit)["composed_prompt_sha256"]}
    # Burst refill: the refill waves' own composed request and the refill paragraph's digest
    # are bound beside the burst's (a project decision of 2026-09-23); config and the burst
    # pool file record these with the prompt file's own. The refill rule is bound with
    # them, so a cell stopped inside the burst still names it through its registration.
    refill_identity = burst_refill_identity() if initial_proposals else {}
    if initial_proposals:
        label_prompt = {**label_prompt, "burst_prompt_sha256":
                        composed_system_prompt(burst=True)[1]["burst_prompt_sha256"],
                        **{key: refill_identity[key] for key in (
                            "burst_refill_prompt_sha256", "burst_refill_paragraph_sha256")},
                        "initial_burst_refill_policy": INITIAL_BURST_REFILL_POLICY,
                        "initial_pool_source": "generated"}
    label_args = argparse.Namespace(**{**vars(args), **label_prompt})
    label_receipt = bmd_label_receipt("chembench", label_args, task=args.task, metadata=label_metadata)
    run_label = label_receipt["run_label"]
    if mask_input_names and run_label == "production":
        raise ValueError("--mask-input-names is development-only; production is refused")
    if getattr(args, "prompt_variant", None) is not None and run_label == "production":
        set_prompt_variant(None)
        raise ValueError("--prompt-variant is development-only; production is refused")
    # Decoding rule: a production run states a decoding option only if the manifest
    # registered the same value - here, before any output exists.
    require_registered_decoding(label_metadata, label_receipt, "chembench")
    label_explicit = (stated_run_label is not None or getattr(args, "manifest_ref", None) is not None
                      or bool(initial_proposals) or bool(pool_file))
    # Both settled before any output
    # exists, like every other refusal in this block: the executed cell is the
    # recorded cell, and the profile grid is the single source of
    # `(particles, q)` unless the caller overrides them explicitly.
    args.cell = validate_cell(args.cell, args.structure_proposals)
    # The structure mode is DERIVED from cell[0], and `args`
    # (i.e. `config["structure_proposals"]`, via `**vars(args)` below) is
    # overwritten with the derived value - the same "recorded is executed"
    # discipline already imposed on the cell itself. `--structure-proposals`,
    # when stated, has already been checked to agree, so this changes nothing
    # for any earlier command line; when omitted it is what makes a `100`/
    # `101` run record `free` instead of the flag's old `catalog` default.
    # (Contrast `run_label` just above, which is deliberately NOT overwritten:
    # there the byte-invariance rule keys off whether a label was stated.)
    args.structure_proposals = structure_mode(args.cell)
    free = args.structure_proposals == "free"
    if not free and (review_rounds or hasattr(args, "mock_schedule")):
        raise ValueError("--archive-review-rounds and --mock-schedule act on proposed models; they "
                         "need a structure-expansion cell")
    profile = dict(PROFILE_GRID[args.profile])
    if args.particles is None:
        args.particles = profile["particles"]
    if args.q is None:
        args.q = profile["q"]
    if args.particles < 2 or args.q < 1:
        raise ValueError("particles must be at least 2 and q positive")
    # STAGE 2: the acquisition axis,
    # settled here with every other refusal, before any output exists.
    # `design_policies.check_policy` is the SHARED rule and is called exactly as
    # `core_count.run` calls it: it rejects an unregistered policy name and
    # refuses ANY non-default policy on an A=0 cell (whose designs are the fixed
    # pre-drawn sequence and an unchanged anchor), and returns whether this
    # cell's third bit selects a policy at all.
    design_policy = args.design_policy
    # Unified protocol: an EIG part (structure_eig, parameter_eig) is checked by this runner's own
    # rule - A=1 and the ascent only - since parameter_eig is not in the shared registry.
    eig_part = CHEM_EIG_PART_POLICIES.get(design_policy)
    # Unified protocol: the column the design-search ascent climbs and chooses on; an EIG-type
    # ascent cell records it as config `design_objective`.
    design_objective = "epig" if design_policy == "epig" else (eig_part or ("eig",))[0]
    if eig_part is not None:
        design_policy_applies = eig_part_policy_applies(args.cell, design_policy, ascent)
    else:
        design_policy_applies = check_policy(args.cell, design_policy)
    if design_policy_applies and eig_part is None and design_policy not in CHEM_DESIGN_POLICIES:
        raise ValueError(
            f"design policy {design_policy!r} is registered but not wired on this positive-rate "
            f"path; this runner executes {list(CHEM_DESIGN_POLICIES)} (see CHEM_DESIGN_POLICIES)"
        )
    design_axis = design_policy if design_policy_applies else DESIGN_AXIS_ABSENT
    eig_samples = profile["eig_samples"]
    # One DISTINCT pool design per round: the adapter's native noise is
    # deterministic per design ("repeats are not independent draws",
    # chembench_observation.py:5), so an already-observed design carries no new
    # information and leaves the candidate set - the same candidate-SET rule
    # `boxinggym_core`'s `design_exclusion="observed"` applies. Refused HERE
    # rather than running out of candidates in round `design_pool + 1`.
    if design_policy_applies and not ascent and args.rounds > profile["design_pool"]:
        raise ValueError(
            f"--rounds {args.rounds} needs that many distinct designs but the {args.profile} "
            f"profile's pre-declared design pool holds {profile['design_pool']}"
        )
    # Round cap: the mock schedule is a fixed 20-entry list. A
    # free run past the schedule on the mock path (40 or, since 2026-09-12,
    # 60 rounds) is refused HERE, at entry, rather
    # than after twenty rounds of SMC - and never silently extended.
    if free and args.proposer == "mock" and args.rounds > MOCK_SCHEDULE_ROUNDS:
        raise ValueError(
            f"the free-mode mock proposal schedule is fixed at {MOCK_SCHEDULE_ROUNDS} rounds; "
            f"--rounds {args.rounds} needs a real proposer (--proposer anthropic|openai)"
        )
    # Role-prior template ablation: this whole runner is the
    # positive/scalar outcome path (every candidate is a `RegressionHypothesis`/
    # `positive_inference.ScalarHypothesis`, `--family` is always `lognormal` or
    # `gamma`), so it is UNCONDITIONALLY refused for a non-default template,
    # before any output directory exists, rather than following
    # the template with no further change: `ScalarHypothesis.
    # as_dict`'s record block (`positive_inference.py`, the `priors` key)
    # restates the coefficient role priors as literals and does not read the
    # active template, so a wide/narrow run here would build one prior and RECORD another.
    # That leak is not fixed here (`positive_inference.py` is untouched); this
    # refusal seals it off instead.
    if args.prior_template != DEFAULT_PRIOR_TEMPLATE:
        raise ValueError(
            "the proposed-prior ablation is defined on the count path; scalar/real noise priors do "
            "not follow the template"
        )
    # Role-prior template ablation: the template, selected once
    # before any hypothesis or fit exists, exactly as in core_count.run - reached
    # only with the default template now (the refusal above). The LLM-prior ablation is NOT
    # offered here: this is the positive/scalar outcome path, where a candidate
    # is a `RegressionHypothesis`/`ScalarHypothesis` and neither carries
    # `coefficient_priors`, so a proposed prior could not reach the fit (the same
    # reason boxinggym_core refuses it off the exhaustive count path).
    set_prior_template(args.prior_template)
    # Validation messages: the wording, selected once with the other engine switches;
    # unstated is v1, the sentence every run has produced.
    set_validation_messages(getattr(args, "validation_messages", DEFAULT_VALIDATION_MESSAGES))
    # Grammar version: the expression grammar, selected here for the same
    # reason and with the same discipline. A catalogue run proposes nothing and
    # never reaches the gate; a free-form run re-registers this same version
    # with its own interface as soon as that interface exists (see below), so
    # the v3 domain proof reads the declared input ranges. Default v2 changes
    # nothing on either path.
    grammar_version = getattr(args, "grammar_version", DEFAULT_GRAMMAR_VERSION)
    set_grammar_version(grammar_version)

    # This adapter remains owned outside BMD; there is no local duplicate oracle.
    from bayes_model_discovery.adapters import chembench_observation as adapter

    # Probe the oracle ONCE,
    # here, before any output exists. The adapter imports `autoscilab` lazily
    # inside every one of its methods (chembench_observation.py:23,44,67,84,
    # 96,127), so a PYTHONPATH mistake used to surface at the first oracle
    # contact - by which time the output directory, config.json and the D0
    # design draw already existed. `sample_designs` is the cheapest call that
    # really imports the oracle module, and it has no side effects (its own
    # docstring: "local RNG (no global RNG changes)").
    adapter.sample_designs(1, OFFICIAL_EVALUATION_SEED)

    # The manifest is read and the row selected BEFORE
    # the output directory exists, so a missing/unknown `--task` fails at
    # entry like every other refusal in this runner, instead of leaving an
    # empty run directory behind.
    manifest = Path(args.manifest).resolve()
    rows = json.loads(manifest.read_text())
    task = select_task(rows, args.task)

    # Stage 2: the candidate design pool - the profile's
    # pre-declared size (dev 32 / paper 256) from the pre-declared
    # `DESIGN_POOL_SEED` - and the official evaluation set, drawn HERE so the
    # disjointness requirement is settled in the same refusal zone as every
    # other check, before any output exists. Both draws build and discard their
    # own generator (`sample_designs`: "local RNG (no global RNG changes)"), so
    # taking them here moves no number in the run; the pool is drawn on EVERY
    # cell, so the declaration and its proof are in every config.json, and it is
    # READ only on an A=1 cell.
    official_designs = adapter.sample_designs(OFFICIAL_EVALUATION_COUNT, OFFICIAL_EVALUATION_SEED)
    design_pool_seed = seed_for(DESIGN_POOL_STREAM, DESIGN_POOL_SEED)
    design_pool = adapter.sample_designs(profile["design_pool"], design_pool_seed)
    # A candidate design that is also an official evaluation point would let an
    # observation land exactly where the primary metric is read. CHECKED, not
    # assumed, and it HALTS; the count is recorded either way. Compared as
    # unordered name->value maps, so the check cannot be fooled by column order.
    pool_official_overlap = len(
        {_design_key(row) for row in design_pool}
        & {_design_key(row) for row in official_designs}
    )
    if pool_official_overlap:
        raise ValueError(
            f"{pool_official_overlap} of the {len(design_pool)} candidate designs are also "
            f"official evaluation points (pool seed {design_pool_seed}, official seed "
            f"{OFFICIAL_EVALUATION_SEED}); the candidate pool must be disjoint from the "
            "scored set"
        )
    # Pool-dimension refusal, from the acquisition policy: pool EIG/EPIG is forbidden at input
    # dimension d >= 3 in every run, so an active joint_eig or epig cell on the pool path
    # (explicit or omitted --design-search) is refused HERE, like every check above: before
    # any output exists and before any proposer call. The dimension is the task's own, the
    # width of its drawn designs (each one a name->value map).
    if not design_pool:
        raise ValueError("the candidate design pool is empty, so its input dimension is unknown")
    refuse_pool_above_two_dimensions(search_settings, design_policy=design_policy,
                                     design_policy_applies=design_policy_applies,
                                     input_dimension=len(design_pool[0]))

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("fits", "predictions"):
        (out / name).mkdir()
    if free:
        for name in ("packets", "proposals"):
            (out / name).mkdir()
    if review_rounds:
        (out / "revival").mkdir()
    # Acquisition axis: only an A=1 cell writes an acquisition record, so an A=0 run
    # does not even gain an empty directory (it is the byte-identical anchor).
    if design_policy_applies:
        (out / "acquisition").mkdir()
    started = time.perf_counter()
    generation_started_at = _utc_now() if pool_generator is not None else None
    settings = SMCSettings(args.particles, 2, 2)
    session = adapter.ChemBenchSession(
        task["domain"],
        task["difficulty"],
        task["law_version"],
        noise_level=args.noise,
        budget=d0_count + args.rounds,
    )
    sample_designs = adapter.sample_designs
    if mask_input_names:
        session = InputNameMaskedSession(session, sample_designs)
        sample_designs = session.sample_designs
        official_designs = [session.mask_design(row) for row in official_designs]
        design_pool = [session.mask_design(row) for row in design_pool]
        if fixed_initial is not None:
            fixed_initial = [session.mask_design(row) for row in fixed_initial]
    schema = session.schema()
    names = schema["input_names"]

    def rng_model_id(hypothesis):
        """A hypothesis's RNG key: its model_id, or its native-name key under input-name masking."""
        if not mask_input_names:
            return hypothesis.model_id
        return session.rng_model_id(hypothesis, out / "masked_model_ids.jsonl")

    search_transform = None
    if ascent:
        declared = search_settings["design_search_transform"]
        if mask_input_names:
            aliases = {name: alias for alias, name in session.input_name_map.items()}
            declared = {**declared,
                        "input_names": [aliases[name] for name in declared["input_names"]],
                        "log_inputs": [aliases[name] for name in declared["log_inputs"]]}
        search_transform = SearchTransform.from_bounds(
            names, schema["input_bounds"], schema["log_sampled_inputs"])
        if search_transform.record() != declared:
            raise ValueError("design-search interface changed after its pre-run declaration")
    designs = sample_designs(
        4 + args.rounds, seed_for("chem_observation_designs", args.seed)
    )
    if fixed_initial is not None:
        # Keep the old future schedule exactly; only the initial observations differ.
        designs = fixed_initial + designs[4:]
    evaluation = sample_designs(args.q, seed_for("chem_evaluation_inputs", 20260907))
    pool_for_transform = sample_designs(256, seed_for("chem_public_pool_inputs", 20260906))
    matrix = lambda items: np.array([[row[name] for name in names] for row in items])
    epig_targets = None
    epig_path_state = EpigDiscretePathState() if design_policy == "epig" else None
    if design_policy == "epig":
        target_seed = seed_for(EPIG_TARGETS_SEED_STREAM, args.seed)
        epig_targets = matrix(sample_designs(EPIG_TARGET_POINTS, target_seed))
    transform = InputTransform.from_public_pool(matrix(pool_for_transform))
    qx = matrix(evaluation)
    # The official evaluation set, a SEPARATE stream
    # from the runner's own per-round checkpoint set above. `evaluation`/`qx`
    # (|Q| = --q, profile-derived) stay exactly as they were - they are what
    # `predictions/t*.npz`, `reference_after_finish.npz` and
    # `scores_after_finish.json` have always been computed on, and the
    # byte-identity anchor covers them. `official_designs` is the baseline's
    # own scoring set and is used ONLY after `finish()`, for the contract's
    # points table (`official_scores_after_finish.json`, `points.csv`).
    official_x = matrix(official_designs)
    pool_x = matrix(design_pool)
    candidate_metadata = candidate_set_record(pool_x, "chembench")

    proposer = None
    interface = grammar_names = seed_structure = None
    fit_count = 0
    if free:
        if args.proposer == "anthropic":
            from .count_proposer import AnthropicCountProposer

            proposer = AnthropicCountProposer(
                out / "llm_transport", allow_live=True, model=args.llm_model,
                **compat_option_kwargs(args, compat=False)
            )
        elif args.proposer == "openai":
            from .count_proposer import OpenAICountProposer

            proposer = OpenAICountProposer(
                out / "llm_transport", allow_live=True, model=args.llm_model,
                **compat_option_kwargs(args, compat=False)
            )
        elif args.proposer == "openai_compat":
            from .count_proposer import OpenAICompatCountProposer

            proposer = OpenAICompatCountProposer(
                out / "llm_transport", allow_live=True, model=args.llm_model,
                base_url=getattr(args, "llm_base_url", None),
                key_name=getattr(args, "llm_key_name", "OPENAI_COMPAT_API_KEY"),
                **compat_option_kwargs(args, compat=True)
            )
        # z-scored derived inputs, using THIS run's own frozen `transform`
        # (the exact InputTransform arithmetic the fixed RegressionHypothesis
        # catalogue is fit under) - not a min-max mapping - so a catalogue
        # member's own linear form can be re-expressed as an identical
        # grammar expression for the belief packet (see `_packet_view`).
        interface = task_interface(
            "positive",
            [{"name": n, "range": list(schema["input_bounds"][n])} for n in names],
            {
                f"{n}z": f"({n}-{float(c)!r})/{float(s)!r}"
                for n, c, s in zip(names, transform.center, transform.scale, strict=True)
            },
            (0.0, None),
        )
        grammar_names = interface_input_names(interface)
        # Grammar version: the same version, now carrying this run's own input
        # ranges, registered before the first hypothesis is constructed below.
        set_grammar_version(grammar_version, interface)
        seed_structure = ScalarHypothesis(
            "b0", args.family, (("b0", "intercept"),), grammar_names
        )
        mock_schedule = _mock_schedule(names, CHEM_ALLOWED_FAMILIES)
        # synthetic_v1's form list (None on the fixed 20-entry schedule).
        synthetic_forms = (synthetic_mock_forms(names, schema["input_bounds"])
                           if mock_schedule_name == "synthetic_v1" else None)
        archive = []
        proposal_log = []
        proposed_rounds = set()
        # Pool retirement: rounds each pool member was actually active
        # for, so a `retired_by_evidence` archive entry can state it (the
        # reason key set `belief_packet.ARCHIVE_REASON_KEYS` fixes).
        active_rounds = {}
        retired_log = []
        # Every model_id ever retired by evidence. `known_ids` already forbids
        # re-proposing them; this set is what lets the refusal SAY it was a
        # retirement (see `_archive_entry`'s `note`).
        retired_ids_ever = set()
        # Models retired by evidence and not revived since, by packet-view id, with
        # the hypothesis a revival refits (without review rounds: exactly retired_ids_ever), and
        # each model's revival attempts.
        retired_now = {}
        revival_attempts = {}
        revival_events = []
        # known_ids is (re)seeded from the initial catalogue further below,
        # once `catalogue`/D0 exist - see the `if free:` block right after
        # `initial_catalogue(...)`.
        known_ids = set()
        born_round = {}

    # Frozen pool track: the whole frozen pool, validated against this run's interface, task, seed,
    # cell, family and D0 policy; the bytes must be the ones bound above.
    frozen_pool = None
    if pool_file:
        frozen_pool = read_chembench_initial_pool_file(
            pool_file, task=task, seed=args.seed, cell=args.cell, family=args.family,
            d0_policy=d0_policy, d0_fixed_inputs=d0_overrides, input_names=grammar_names)
        if frozen_pool.metadata() != pool_prefix:
            raise ValueError("frozen pool file changed after its binding was read")
    # Stage slots: the stage slots every proposer call of this run carries (record-only).
    if pool_generator is not None:
        stage_slots, stage_slot_record = _chembench_stage_slots(
            prompt_identity(), init_slot=pool_generator,
            init_prompt_sha256=prompt_identity()["prompt_sha256"], expand=False)
    elif frozen_pool is not None:
        stage_slots, stage_slot_record = _chembench_stage_slots(
            prompt_identity(), init_slot=frozen_pool.block["init_slot"],
            init_prompt_sha256=frozen_pool.block["generator_prompt_sha256"])
    else:
        stage_slots, stage_slot_record = _chembench_stage_slots(
            prompt_identity(), init_slot="in_cell_arm_burst" if initial_proposals else "catalogue")
    config = {
        **strip_cli_settings(compat_config(args)),
        **search_settings,
        "model_id_schema": MODEL_ID_SCHEMA,
        "notation_normal_form": NOTATION_NORMAL_FORM,
        # Constraint disclosure: every ChemBench run, whether or not it builds a packet.
        "constraints_disclosure": CONSTRAINTS_DISCLOSURE,
        "stage_slots": stage_slot_record,
        # Multi-candidate proposals: the composed request's digest, beside (never
        # instead of) the packaged file's own. At k = 1 the composed text IS
        # the file's text and no key is added, so this config is unchanged.
        **{k: v for k, v in composed_prompt_identity(proposal_limit).items()
           if k == "composed_prompt_sha256"},
        "d0_policy": d0_policy,
        "d0_count": d0_count,
        "d0_conditions": d0_conditions,
        "d0_conditions_source": "our_declared_values_mda_undisclosed" if fixed_initial is not None else None,
        "d0_fixed_inputs": d0_overrides,
        "d0_condition_rules": ({"C_I": "unperturbed_zero", "C_P": "unperturbed_zero",
                                "C_B": "geometric_midpoint_native_bounds",
                                "Enz": "geometric_midpoint_native_bounds",
                                "T": "native_Arrhenius_T_REF", "pH": "neutral_pH",
                                "overrides": d0_overrides or {}} if fixed_initial is not None else None),
        "d0_seed_label": "chem_observation_designs",
        "manifest": str(manifest),
        "task": task,
        # The id as REQUESTED on the command line
        # (`None` = "the manifest's only row"), kept beside the resolved row
        # because `"task"` above is the row itself and would otherwise hide
        # whether the selection was explicit.
        "task_requested": args.task,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "adapter_source": str(Path(adapter.__file__).resolve()),
        "adapter_sha256": hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "chembench_core.py",
                "count_models.py",
                "model_identity.py",
                "positive_inference.py",
                "count_inference.py",
                "core_count.py",
            )
        },
        "smc": asdict(settings),
        # The cell this run EXECUTED (was once the literal
        # "000", even for a free run, which actually executed
        # "111").
        "cell": args.cell,
        # Stage 2: the acquisition axis the third bit
        # actually selects. `design_axis` was the single literal
        # `"absent_stage1"` in stage 1 and is now the ACTIVE policy name on an
        # A=1 cell / `DESIGN_AXIS_ABSENT` on an A=0 cell; the three keys beside
        # it are `core_count.run`'s own three (`design_policy`,
        # `design_policy_applies`, `design_policy_target`), with the identical
        # meanings, so one reader parses both runners.
        "design_axis": design_axis,
        "design_policy": design_policy,
        "design_policy_applies": design_policy_applies,
        "design_policy_target": ((eig_part[2] if eig_part is not None else TARGET[design_policy])
                                 if design_policy_applies else "random"),
        "design_pool": {
            "sampler": "adapters.chembench_observation.sample_designs",
            "count": profile["design_pool"],
            "count_source": "profile_grid.design_pool",
            "declared_seed": DESIGN_POOL_SEED,
            "stream": DESIGN_POOL_STREAM,
            "sampler_seed": design_pool_seed,
            "acquisition_stream": ACQUISITION_STREAM,
            "eig_samples": eig_samples,
            "eig_estimator": "positive_inference.joint_eig",
            "decision_authority": "bayes_model_discovery.eig.joint.choose_design",
            "supported_policies": [p for p in CHEM_DESIGN_POLICIES
                                   if (p != "epig" or design_policy == "epig")
                                   and (p not in CHEM_STUDY_POLICIES
                                        or design_policy in CHEM_STUDY_POLICIES)]
                                  # Unified protocol: listed only on an EIG-part cell, like the
                                  # study policies, so every other config keeps its bytes.
                                  + (list(CHEM_EIG_PART_POLICIES) if eig_part is not None else []),
            "registered_policies": [p for p in POLICIES
                                    if (p != "epig" or design_policy == "epig")
                                    and (p != "space_filling"
                                         or design_policy in CHEM_STUDY_POLICIES)],
            "official_evaluation_overlap": pool_official_overlap,
            "disjoint_from_official_evaluation": pool_official_overlap == 0,
            "observed_designs_leave_the_candidate_set": True,
            "read_by_this_run": design_policy_applies,
            "note": (
                "fixed candidate set: one pre-declared `sample_designs` grid, fixed for every "
                "task/seed/cell, checked disjoint from the official evaluation set at run time; "
                "an A=0 cell records it and reads none of it"
            ),
        },
        "profile_grid": {
            **profile,
            "note": (
                "fixed profile table; `design_pool`/`eig_samples` are pre-declared for the "
                "stage-2 acquisition axis and unused in stage 1"
            ),
        },
        # The KL truth density, DECLARED here (before any
        # round runs), and the two required disclosures wherever
        # `kl`/`nmse_mda` are read together - `nmse_mda`'s target and `kl`'s
        # truth center are DIFFERENT adapter reference values.
        # The computed `kl` VALUE itself is
        # a post-`finish()` result and lives in summary.json/points.csv only,
        # like every other computed metric this runner records.
        "kl_truth_density": {
            "family": "relative_normal_truncated_at_0",
            "center": OFFICIAL_KL_TRUTH_CENTER,
            "relative_sd": args.noise,
            "definition": (
                "Y = max(X, 0), X ~ Normal(noiseless_observation(x_q), "
                "(noise*noiseless_observation(x_q))**2); the y=0 point mass is "
                "Phi(-1/noise), computed via scipy.stats.norm.logcdf (never hardcoded), "
                "which underflows to exactly 0.0 in float64 for noise=0.01"
            ),
            "grounds": {
                "native_noise_arithmetic": (
                    "autoscilab/oracle/chembench.py:730 r0_obs = max(r0_true*(1.0+noise),0.0); "
                    ":551 noise = Normal(0, noise_level) draw; :595 noise_level default 0.01"
                ),
                "noiseless_observation_definition": (
                    "adapters/chembench_observation.py:135-136 "
                    "(primary_rate * _secondary_effects(T, pH)), returned :140-141"
                ),
                "engine_scoping": "the engine's truth-hook design",
            },
        },
        "kl_truth_center": OFFICIAL_KL_TRUTH_CENTER,
        "nmse_mda_target": OFFICIAL_PRIMARY_TARGET,
        "particles_source": "profile" if args.particles == profile["particles"] else "explicit",
        "q_source": "profile" if args.q == profile["q"] else "explicit",
        # Round cap: the round window this run hands
        # `count_proposer.propose_one` (the transport itself records only the
        # round, so the horizon is recorded here).
        "max_proposal_rounds": args.rounds,
        "mock_schedule_rounds": MOCK_SCHEDULE_ROUNDS,
        # Pool retirement.
        "active_maximum": ACTIVE_MAXIMUM,
        # The official evaluation set and the
        # primary metric, recorded with their provenance so a row of
        # points.csv can be traced without this file's git history.
        "official_evaluation": {
            "sampler": "adapters.chembench_observation.sample_designs",
            "count": OFFICIAL_EVALUATION_COUNT,
            "seed": OFFICIAL_EVALUATION_SEED,
            "generator": "numpy.random.default_rng(seed), log-uniform on CHEM_LOG_VARS",
            "primary_target": OFFICIAL_PRIMARY_TARGET,
            "interval_target": OFFICIAL_INTERVAL_TARGET,
            "requested_after_finish_only": True,
            "same_generator_and_seed_as_baseline_scoring": True,
            "baseline_source": (
                "the comparison method's evaluation code (its ChemBench run manifest, "
                "eval_points.rule)"
            ),
        },
        "primary_metric": {
            "name": PRIMARY_METRIC,
            "label": PRIMARY_METRIC_LABEL,
            "formula": (
                "mean_Q[(log1p(pred) - log1p(primary_rate))^2] / Var_Q[log1p(primary_rate)], "
                "Var = population variance (ddof=0) over the official Q"
            ),
            "auxiliary": ["nmsle", "rmsle"],
            "nmsle_equals_nmse_mda": True,
            "baseline_definition_source": (
                "the comparison method's evaluation code (its ChemBench run manifest, "
                "metric_defs.nmse_mda)"
            ),
        },
        "oracle_package": _oracle_provenance(),
        "proposal_count": 0,
        "initial_pool": 4,
        "schema": schema,
        "transform": asdict(transform),
        "likelihood_target": "native_r0_observation_only; auxiliary observables not used",
        "label_policy": "fixed_random_designs_from_native_input_distribution",
        "evaluation_labels_used_before_finish": 0,
        "paid_api_calls": 0,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pymc", "pytensor")
        },
        "comparison": "first connected path, not a matched historical baseline or LLM-effect claim",
        # structure_proposals itself already arrives via
        # **vars(args) above (the new --structure-proposals/--proposer/
        # --llm-model/--allow-paid-api CLI flags); these two are the only
        # keys added manually, both new (the only config.json diff a
        # catalog-mode run picks up).
        "points_variant": (_points_variant(
            args.structure_proposals, args.prior_template, design_policy, design_policy_applies
        ) if eig_part is None else eig_part_points_variant(
            args.structure_proposals, args.prior_template, design_policy)),
        # Plot contract. After a smoke-run finding, this replaces the
        # inline "dev profile or the OpenAI transport is development,
        # everything else production" rule with the shared
        # `core_count.points_label` helper - that rule mislabelled a
        # paper-profile MOCK run as production (mock is neither dev nor
        # openai). `status` is still `completed` unconditionally because
        # `points.csv` is written only on the success path (a failed run
        # leaves failure.json and no point rows at all, rather than a
        # `failed` row with no y).
        # The RESOLVED --run-label. With the flag
        # absent this is exactly `points_label(args.profile, args.proposer)`,
        # i.e. the helper's behaviour unchanged; with it, the stated label, which is
        # how a paper+anthropic engine run gets recorded as development.
        "points_label": run_label,
        "points_status": "completed",
        # Role-prior template ablation: the template NAME arrives through **vars(args); the
        # TABLE is recorded only when the template is not the frozen default, so a
        # default run's config.json keeps its earlier key set.
        **(
            {"prior_hyperparameters": prior_template_table()}
            if args.prior_template != DEFAULT_PRIOR_TEMPLATE
            else {}
        ),
        **prompt_identity(),
    }
    if mask_input_names:
        config["input_name_map"] = session.input_name_map
    if label_explicit:
        config["label_receipt"] = label_receipt
    if initial_proposals:
        config["initial_pool_source"] = "generated"
        config["burst_prompt_sha256"] = label_prompt["burst_prompt_sha256"]
        # Burst refill: the refill digests and the refill rule, only with a burst.
        config.update(refill_identity, initial_burst_refill_policy=INITIAL_BURST_REFILL_POLICY)
    generation_block = None
    if pool_generator is not None:
        # Frozen pool track: the generation run's identity, written into its burst file and config.
        generation_block = {
            "schema": FROZEN_POOL_SCHEMA, "track_id": FROZEN_POOL_TRACK,
            "pool_id": frozen_pool_id(task, args.seed), "task": task["domain"],
            "stratum": task["difficulty"], "law_version": task["law_version"],
            "task_id": task["id"], "seed": args.seed, "cell": args.cell, "family": args.family,
            "d0_policy": d0_policy, "d0_fixed_inputs": d0_overrides,
            "init_slot": pool_generator, **_generator_identity(pool_generator),
            "burst_paragraph_sha256": stage_slot_record["burst_paragraph_sha256"],
            "burst_refill_paragraph_sha256": stage_slot_record["burst_refill_paragraph_sha256"],
            "stage_slot_schema": STAGE_SLOT_SCHEMA,
            "constraints_disclosure": CONSTRAINTS_DISCLOSURE,
            "requested": initial_proposals,
        }
        config["frozen_pool_generation"] = generation_block
    if frozen_pool is not None:
        # Frozen pool track: the cell's frozen pool binding. The receipt is
        # written first, so config binds its digest; round 0 then fits the members only.
        pool_cell_fields = {
            **frozen_pool.metadata(),
            "initial_pool_file": frozen_pool.name,
            "track_id": FROZEN_POOL_TRACK,
            "pool_id": frozen_pool.block["pool_id"],
            "pool_file_sha256": frozen_pool.sha256,
            "initial_pool_generation": FROZEN_POOL_GENERATION,
            "round0_llm_calls_inside_cell": 0,
            "arm_type": "prompt" if design_policy_applies else "random_baseline",
            "prompt_version": config["prompt_version"],
        }
        write_json(out / "initial_pool_receipt.json", {
            "schema": "chembench_frozen_pool_receipt_v1",
            "model_id_schema": MODEL_ID_SCHEMA,
            "notation_normal_form": NOTATION_NORMAL_FORM,
            **pool_cell_fields,
            "initial_catalogue": catalogue_mode,
            "member_ids": [h.model_id for h in frozen_pool.hypotheses],
            "frozen_pool": frozen_pool.block,
        })
        config.update(pool_cell_fields)
        config["initial_pool"] = len(frozen_pool.hypotheses)
        config["initial_pool_receipt"] = {
            "path": "initial_pool_receipt.json",
            "sha256": hashlib.sha256((out / "initial_pool_receipt.json").read_bytes()).hexdigest(),
        }
    if ascent:
        config["label_receipt"] = label_receipt
        config["design_pool"].update(
            read_by_this_run=False, observed_designs_leave_the_candidate_set=False,
            note="Declared comparison design pool only; ascent uses continuous endpoints.")
    if ascent and design_policy != "epig":
        # Unified protocol: every EIG-type ascent round records the joint EIG's two
        # parts and the point disagreement on its arrival rows (annotate_eig_endpoints), and the
        # cell states the column its ascent optimizes; three new keys, appended, and nothing else
        # in this config moves.
        config.update(eig_components_record=EIG_COMPONENTS_RECORD,
                      point_disagreement_record=POINT_DISAGREEMENT_RECORD,
                      design_objective=design_objective)
    if eig_part is not None:
        # Unified protocol: the EIG-part arm's own declaration (the key names of BoxingGym's
        # EIG-variant arms).
        config.update(
            design_policy_objective_column=eig_part[0],
            design_policy_definition=CHEM_EIG_PART_DEFINITION[design_policy],
            design_policy_estimator=EIG_COMPONENTS_RECORD["estimator"],
            design_policy_search=(f"the design-search ascent climbs {eig_part[0]} itself: the SPSA "
                                  "differences, the noise floor, the chooser over the five arrival "
                                  "rows (registered tie rule) and the holdout all read it; the "
                                  "restart starts and seeds are the joint_eig arm's"),
        )
        if requested_design_policy != design_policy:
            config["design_policy_requested"] = requested_design_policy
    if composition == "v2":
        # v2's version and canonical digest beside the composed digest above.
        config.update({key: value for key, value in composed_prompt_identity(proposal_limit).items()
                       if key.startswith("composition_paragraph")})
    if review_rounds:
        # The review rounds, the review request's own composed digest and the rule.
        review_identity = composed_prompt_identity(proposal_limit, archive_review=True)
        config.update(
            archive_review_rounds=list(review_rounds),
            archive_review_composed_prompt_sha256=review_identity["composed_prompt_sha256"],
            archive_review_paragraph_version=review_identity["archive_review_paragraph_version"],
            archive_review_paragraph_sha256=review_identity["archive_review_paragraph_sha256"],
            revive_limit=REVIVE_LIMIT,
            archive_review_rule=(
                "on each listed round the proposal request also carries the archive review "
                "paragraph and the reply names up to revive_limit archived models retired by "
                "evidence; each valid one is refit from the prior on all public observations "
                "(seed stream chem_smc, the active members' own) and competes with the active pool "
                "and the round's new candidates in one retention to the active maximum; invalid "
                "names are recorded and ignored, with no repair turn; revival/tNN.json records it"),
        )
    # Brief section 11.10: an explicit --design-search (pool or ascent) lists its source among
    # the recorded source digests; the omitted path's declared settings are empty, so no key.
    if search_settings:
        config["source_sha256"]["design_search.py"] = hashlib.sha256(
            Path(__file__).with_name("design_search.py").read_bytes()).hexdigest()
    if fit_timeout_seconds is None:
        config.pop("fit_timeout_seconds", None)
    else:
        config["fit_timeout_seconds"] = fit_timeout_seconds
    if fit_guards == "off":
        config.pop("fit_guards", None)
    else:
        config["fit_guards"] = fit_guards
    if design_policy == "epig":
        config.update(epig_config(epig_targets, target_seed,
                                 "adapters.chembench_observation.sample_designs",
                                 evaluation_points=qx))
        config["design_pool"]["eig_estimator"] = "count_inference.choose_epig"
        config["design_pool"]["objective_column"] = "epig"
    if design_policy in CHEM_STUDY_POLICIES:
        is_space_filling = design_policy == "space_filling"
        config["design_policy_definition"] = (SPACE_FILLING_DEFINITION if is_space_filling
                                               else POINT_DISAGREEMENT_DEFINITION)
        config["design_pool"].update(
            objective_column=("min_normalized_distance" if is_space_filling
                              else "point_disagreement"),
            objective_estimator=("design_policies.select_space_filling" if is_space_filling
                                 else "positive_inference.point_disagreement"),
            eig_estimator=None, eig_samples=None,
            uses_posterior=not is_space_filling, uses_rng=False,
        )
        if is_space_filling:
            config["design_normalization_bounds"] = [list(schema["input_bounds"][n]) for n in names]
        else:
            config["design_pool"]["posterior_particles"] = "all_retained"
    if not design_policy_applies:
        # Random retains its separate observation schedule; this hashes the
        # generated comparison pool and does not assert that random uses it.
        config.update(candidate_metadata)
    # A generated pool's outcome is only known after the burst. Publish its
    # completed config once, preserving write_json's no-overwrite guarantee.
    if not initial_proposals:
        write_json(out / "config.json", config)
    np.savez_compressed(out / "evaluation_inputs.npz", x=qx, names=np.asarray(names))
    public = []
    fits = []
    # One entry per round, kept so the official
    # evaluation set can be scored AFTER `finish()` - the truth is unavailable
    # before it, by construction (`ChemBenchSession.reference_values` refuses),
    # so a per-round contract row cannot be produced any earlier. Only the
    # round index, the disclosed observation count and the round's own pool of
    # fits are retained (a Fit holds one 1-D particle array per parameter).
    checkpoints = []
    prediction_excluded_models = []
    candidate_failures = []
    # Failure-class accounting: core_count.failure_class_counts over the burst's transports and
    # events and every later round's transport record and per-candidate events.
    failure_counts = dict.fromkeys(FAILURE_CLASSES, 0)
    timed_out_ids = set()
    guard_failed_ids = set()
    # Acquisition guard: the acquisition_failed rounds, whose refit was skipped.
    refit_skipped_rounds = []
    timings = {"fit": 0.0, "prediction": 0.0, "observation": 0.0, "evaluation": 0.0}
    # Acquisition axis: the acquisition bucket exists only where an acquisition
    # happens, so an A=0 summary.json keeps its stage-1 key set exactly.
    if design_policy_applies:
        timings["acquisition"] = 0.0
    # Pool indices already observed by this run, i.e. what leaves the candidate
    # set before the next argmax.
    observed_pool_indices = set()
    # Paid-call guard: the round in progress, for a summary the paid-call guard may have to write.
    current_round = 0
    try:
        for design in designs[:d0_count]:
            tic = time.perf_counter()
            public.append(session.observe(design))
            timings["observation"] += time.perf_counter() - tic
            append_json(out / "observations.jsonl", public[-1])
        if frozen_pool is not None and json.loads(json.dumps(public)) != frozen_pool.public_observations:
            # Frozen pool track: the pool was made on these D0 observations (task, noise, D0 rule);
            # any other D0 means it is not this cell's pool.
            raise ValueError("frozen pool file's public observations differ from this cell's D0 "
                             "observations")
        x = matrix([row["x"] for row in public])
        y = np.array([row["y"] for row in public])
        response_unit = float(y.mean())
        if not np.isfinite(response_unit) or response_unit <= 0:
            raise ValueError("D0 response unit is not positive; no fallback")
        catalogue, policy = initial_catalogue(transform.apply(x), y, args.family)
        if catalogue_mode == "off":
            catalogue = []
        initial_catalogue_ids = {h.model_id for h in catalogue}
        if free:
            # Duplicate detection for a proposed (always-ScalarHypothesis)
            # candidate must compare against the catalogue's own identity IN
            # THAT SAME grammar space, not RegressionHypothesis.model_id (a
            # different hash scheme for the same underlying model) - seeded
            # once, before round 1, so a free-form candidate mathematically
            # identical to an existing catalogue member (the plain "b0"
            # intercept model, in particular) is caught as a duplicate
            # instead of silently double-counted in the pool.
            known_ids = {_packet_view(h, names, grammar_names).model_id for h in catalogue}
        write_json(
            out / "frozen_d0_policy.json",
            {
                "response_unit": response_unit,
                "policy": policy,
                "models": [h.as_dict() for h in catalogue],
                "prior_recalibration_after_d0": False,
            },
        )
        stop_record = None
        for t in range(args.rounds + 1):
            stop_file = getattr(args, "stop_file", None)
            if t >= 1 and stop_file is not None and Path(stop_file).exists():
                stop_path = Path(stop_file).resolve()
                stop_record = {
                    "reason": "stop_requested",
                    "stop_file": str(stop_path),
                    "stop_file_mtime_unix": stop_path.stat().st_mtime,
                    "detected_before_round": t,
                    "last_completed_round": t - 1,
                    "rounds_planned": args.rounds,
                }
                break
            current_round = t
            # Round timing: the round's observational start, for record_round_timing.
            round_started_at_unix, round_started = time.time(), time.perf_counter()
            # Acquisition guard: set only on an all_designs_invalid or
            # acquisition_failed round.
            round_acquisition_status = None
            if t:
                # Stage 2: on an A=1 cell this round's
                # design is CHOSEN from the pre-declared pool; on an A=0 cell it
                # stays the pre-drawn `chem_observation_designs` entry, which is
                # what keeps `000`/`110` byte-identical to stage 1. `fits` here
                # is still round t-1's pool (it is rebuilt below), which is the
                # posterior the choice must be made under.
                acquisition_started = time.perf_counter()
                if design_policy_applies:
                    acq = time.perf_counter()
                    acquisition_seed = seed_for(ACQUISITION_STREAM, args.seed, t)
                    excluded = sorted(observed_pool_indices)
                    candidates = [
                        i for i in range(len(pool_x)) if i not in observed_pool_indices
                    ]
                    candidate_x = pool_x[candidates]
                    # `positive_inference.joint_eig` is this outcome path's own
                    # registered joint estimator, called exactly as
                    # `boxinggym_core.run`'s non-exhaustive acquisition branch
                    # calls it (boxinggym_core.py:2904-2909): one table for the
                    # whole candidate set under ONE seed (common random numbers
                    # across designs), then the single registered decision
                    # authority `eig.joint.choose_design` - TIE_ATOL 1e-12,
                    # smallest design value wins, which on a multi-input design
                    # is the lexicographically smallest coordinate list, the
                    # same representation `boxinggym_core.design_value` writes
                    # for a multi-input pool.
                    if ascent:
                        # Acquisition guard: this round's log of the guarded joint_eig calls
                        # that excluded anything, and the objective's call ordinal, which counts
                        # every call from 0 in order (design_search's objective_call_index).
                        exclusion_calls, objective_calls = [], [0]

                        def objective(points, objective_seed, fit_bank=fits,
                                      exclusion_calls=exclusion_calls,
                                      objective_calls=objective_calls):
                            """Ascent objective rows at `points`: guarded EIG, EIG parts or EPIG."""
                            call_index = objective_calls[0]
                            objective_calls[0] += 1
                            if design_policy == "epig":
                                _, objective_rows = choose_epig(
                                    fit_bank, points, objective_seed, eig_samples,
                                    target_points=epig_targets,
                                    discrete_path_state=epig_path_state)
                                return objective_rows
                            # Nothing excluded leaves `bank is fit_bank`, the pre-guard
                            # call; with every model excluded the rows are explicit invalid
                            # states and joint_eig is not called (no value, NaN or penalty).
                            bank, info = _acquisition_bank_for_points(fit_bank, points)
                            if info is not None:
                                exclusion_calls.append({"objective_call_index": call_index,
                                                        "point_count": len(points), **info})
                                if info["invalid"]:
                                    return [{"design": point.tolist(), "invalid": True,
                                             "invalid_reason": ACQUISITION_INVALID_REASON}
                                            for point in points]
                            if eig_part is None:
                                table = joint_eig(bank, points, objective_seed, eig_samples)
                                return [{"design": point.tolist(), "eig": float(table["eig"][i]),
                                         "mc_standard_error": float(table["mc_standard_error"][i])}
                                        for i, point in enumerate(points)]
                            # Unified protocol: an EIG-part arm's objective is the decomposed call
                            # (the joint value and its error are the plain call's, bit for bit),
                            # so the ascent can climb the part.
                            table = joint_eig(bank, points, objective_seed, eig_samples,
                                              decompose=True)
                            return [{"design": point.tolist(), "eig": float(table["eig"][i]),
                                     "mc_standard_error": float(table["mc_standard_error"][i]),
                                     **{column: float(table[column][i])
                                        for column in EIG_COMPONENT_COLUMNS}}
                                    for i, point in enumerate(points)]

                        # Unified protocol: an EIG-type round annotates its arrival rows with the
                        # joint EIG's two parts and the point disagreement (no objective call,
                        # no RNG), and an EIG-part arm's ascent climbs its part.
                        record = projected_spsa(
                            objective, search_transform, global_seed=args.seed, cell=args.cell,
                            round_index=t, acquisition_seed=acquisition_seed,
                            acquisition_stream=ACQUISITION_STREAM,
                            objective_key=design_objective,
                            noise_floor_method=search_settings["noise_floor_method"],
                            **({} if design_policy == "epig" else {"arrival_annotator": partial(
                                annotate_eig_endpoints, fits, eig_samples=eig_samples)}))
                        if objective_calls[0] != record["objective_call_count"]:
                            raise ValueError("ChemBench ascent objective call count differs from "
                                             "the design-search record")
                        # The exclusion keys exist only when some call excluded anything.
                        if exclusion_calls:
                            record.update(_eig_exclusion_record(fits, exclusion_calls, record))
                        elif "acquisition_status" in record:
                            raise ValueError("design-search reported invalid rows that no guarded "
                                             "objective call produced")
                        acq_rows = record["arrival_rows"]
                        if record.get("acquisition_status") == "all_designs_invalid":
                            # The chosen design is None; one fallback, never an ascent.
                            design = _acquisition_fallback(record, fits, designs, d0_count - 1 + t,
                                                           names, FALLBACK_POLICY)
                            round_acquisition_status = record["acquisition_status"]
                        else:
                            design = dict(zip(names, record["chosen"], strict=True))
                        record["target"] = TARGET[design_policy] if eig_part is None else eig_part[2]
                        record["policy"].update(
                            design_policy=design_policy, eig_samples=eig_samples,
                            eig_estimator=("count_inference.choose_epig" if design_policy == "epig"
                                           else "positive_inference.joint_eig"),
                            model_weights=[float(v) for v in np.exp(model_logweights(fits))],
                            active_ids=[result.hypothesis.model_id for result in fits])
                        if eig_part is not None:
                            # Unified protocol: the policy keys of BoxingGym's EIG-variant arms,
                            # so one reader takes the one-step regret the arm accepted from both
                            # runners.
                            record["policy"].update(
                                target=eig_part[2], definition=CHEM_EIG_PART_DEFINITION[design_policy],
                                estimator=EIG_COMPONENTS_RECORD["estimator"])
                            if record["chosen"] is not None:
                                # The joint EIG at the chosen endpoint and the largest joint EIG
                                # among this ascent's valid endpoints.
                                valid = [row for row in record["arrival_rows"]
                                         if row.get("invalid") is not True]
                                record["policy"].update(
                                    joint_eig_at_chosen=next(row["eig"] for row in valid
                                                             if row["design"] == record["chosen"]),
                                    joint_eig_max=max(row["eig"] for row in valid))
                        write_json(out / "acquisition" / f"t{t:02d}.json", record)
                    elif design_policy == "epig":
                        chosen, acq_rows = choose_epig(fits, candidate_x, acquisition_seed,
                                                      eig_samples, target_points=epig_targets,
                                                      discrete_path_state=epig_path_state)
                        for i, row in enumerate(acq_rows):
                            row["pool_index"] = candidates[i]
                        record = epig_acquisition_record(chosen, acq_rows, acquisition_seed)
                        record.update(candidate_metadata)
                        picked = acq_rows[record["policy"]["chosen_index"]]
                        observed_pool_indices.add(picked["pool_index"])
                        design = design_pool[picked["pool_index"]]
                        record["policy"].update(chosen_pool_index=picked["pool_index"],
                                                excluded_pool_indices=excluded)
                        if explicit_pool:
                            def epig_objective(points, objective_seed, fit_bank=fits):
                                """EPIG rows at `points`, for the explicit-pool diagnostics."""
                                return choose_epig(fit_bank, points, objective_seed, eig_samples,
                                                   target_points=epig_targets,
                                                   discrete_path_state=epig_path_state)[1]

                            record["candidate_source"] = "pool"
                            record["policy"].update(pool_diagnostics(
                                epig_objective, acq_rows, choose_design(acq_rows, "epig"),
                                global_seed=args.seed, round_index=t, key="epig"))
                        write_json(out / "acquisition" / f"t{t:02d}.json", record)
                    elif design_policy in CHEM_STUDY_POLICIES:
                        if design_policy == "space_filling":
                            chosen, acq_rows, policy_record = select_space_filling(
                                candidate_x, matrix([row["x"] for row in public]),
                                [schema["input_bounds"][name] for name in names],
                            )
                        else:
                            values = point_disagreement(fits, candidate_x)
                            acq_rows = [{"design": [float(v) for v in row],
                                         "point_disagreement": float(value)}
                                        for row, value in zip(candidate_x, values, strict=True)]
                            decision = choose_design(acq_rows, "point_disagreement")
                            chosen = decision["chosen_design"]
                            chosen_index = next(i for i, row in enumerate(acq_rows)
                                                if row["design"] == chosen)
                            policy_record = {
                                "design_policy": design_policy,
                                "objective_column": "point_disagreement",
                                "target": TARGET[design_policy],
                                "chosen_index": chosen_index, "chosen_design": chosen,
                                "chosen_value": decision["top_value"],
                                "n_tied_at_max": decision["n_tied_at_max"],
                                "tie_rule": decision["tie_rule"], "tie_atol": decision["tie_atol"],
                                "reason": (
                                    f"argmax of point_disagreement over {len(acq_rows)} candidate designs; "
                                    f"{decision['n_tied_at_max']} within tie_atol "
                                    f"{decision['tie_atol']:g} of the maximum "
                                    f"{decision['top_value']!r}; tie rule {decision['tie_rule']}"
                                ),
                                "criteria": acq_rows, "criteria_columns": ["point_disagreement"],
                                "definition": POINT_DISAGREEMENT_DEFINITION,
                                "uses_posterior": True, "uses_rng": False,
                                "posterior_particles": "all_retained",
                                "model_weights": [float(w) for w in np.exp(model_logweights(fits))],
                                "active_ids": [result.hypothesis.model_id for result in fits],
                            }
                        for i, row in enumerate(acq_rows):
                            row["pool_index"] = candidates[i]
                        picked = acq_rows[policy_record["chosen_index"]]
                        observed_pool_indices.add(picked["pool_index"])
                        design = design_pool[picked["pool_index"]]
                        policy_record.update(chosen_pool_index=picked["pool_index"],
                                             excluded_pool_indices=excluded,
                                             pool_size=len(pool_x), candidate_count=len(acq_rows))
                        write_json(out / "acquisition" / f"t{t:02d}.json", {
                            **candidate_metadata, "seed": acquisition_seed,
                            "chosen": chosen, "rows": acq_rows,
                            "target": TARGET[design_policy], "policy": policy_record,
                        })
                    else:
                        table = joint_eig(fits, candidate_x, acquisition_seed, eig_samples)
                        acq_rows = [
                            {
                                "design": [float(v) for v in candidate_x[i]],
                                "eig": float(table["eig"][i]),
                                "mc_standard_error": float(table["mc_standard_error"][i]),
                                "pool_index": candidates[i],
                            }
                            for i in range(len(candidates))
                        ]
                        decision = choose_design(acq_rows)
                        picked = next(
                            row for row in acq_rows if row["design"] == decision["chosen_design"]
                        )
                        observed_pool_indices.add(picked["pool_index"])
                        design = design_pool[picked["pool_index"]]
                        extra_diagnostics = {}
                        if explicit_pool:
                            def eig_objective(points, objective_seed, fit_bank=fits):
                                """Joint EIG rows at `points`, for the explicit-pool diagnostics."""
                                # Acquisition guard: the same guard. The pool table above
                                # already evaluated this point with every particle, so an
                                # exclusion here breaks that invariant and fails closed, never
                                # as a KeyError.
                                if _acquisition_bank_for_points(fit_bank, points)[1] is not None:
                                    raise ValueError(
                                        "explicit-pool diagnostic point excluded posterior "
                                        "particles although the pool table evaluated it with "
                                        "all of them; no fallback")
                                fresh = joint_eig(fit_bank, points, objective_seed, eig_samples)
                                return [{"design": point.tolist(), "eig": float(fresh["eig"][i]),
                                         "mc_standard_error": float(fresh["mc_standard_error"][i])}
                                        for i, point in enumerate(points)]

                            extra_diagnostics = pool_diagnostics(
                                eig_objective, acq_rows, decision,
                                global_seed=args.seed, round_index=t)
                        write_json(
                            out / "acquisition" / f"t{t:02d}.json",
                            {
                                **candidate_metadata,
                                **({"candidate_source": "pool"} if explicit_pool else {}),
                                "seed": acquisition_seed,
                                "chosen": decision["chosen_design"],
                                "rows": acq_rows,
                                "target": TARGET[design_policy],
                                "policy": {
                                    "design_policy": design_policy,
                                    "objective_column": "eig",
                                    "chosen_pool_index": picked["pool_index"],
                                    "chosen_value": decision["top_value"],
                                    "mc_standard_error_at_chosen": picked["mc_standard_error"],
                                    "n_tied_at_max": decision["n_tied_at_max"],
                                    "tie_rule": decision["tie_rule"],
                                    "tie_atol": decision["tie_atol"],
                                    "reason": (
                                        f"argmax of eig over {len(acq_rows)} candidate designs; "
                                        f"{decision['n_tied_at_max']} within tie_atol "
                                        f"{decision['tie_atol']:g} of the maximum "
                                        f"{decision['top_value']!r}; tie rule "
                                        f"{decision['tie_rule']}"
                                    ),
                                    "eig_samples": eig_samples,
                                    "pool_size": len(pool_x),
                                    "candidate_count": len(acq_rows),
                                    "excluded_pool_indices": excluded,
                                    "eig_estimator": "positive_inference.joint_eig",
                                    "decision_authority": (
                                        "bayes_model_discovery.eig.joint.choose_design"
                                    ),
                                    "criteria_columns": ["eig"],
                                    "criteria_note": (
                                        "one joint EIG table per round from "
                                        "positive_inference.joint_eig (the sample-based "
                                        "I((M,Phi);Y_x|H_t) this positive-rate path owns); the "
                                        "count-bank arms of design_policies.criteria_table "
                                        "(structure_eig/point_disagreement/uniform_random) have no "
                                        "bounded-count outcome space here and are not computed"
                                    ),
                                    "model_weights": [
                                        float(v) for v in np.exp(model_logweights(fits))
                                    ],
                                    "active_ids": [
                                        result.hypothesis.model_id for result in fits
                                    ],
                                    **extra_diagnostics,
                                },
                            },
                        )
                    timings["acquisition"] += time.perf_counter() - acq
                else:
                    design = designs[d0_count - 1 + t]
                record_acquisition_timing(
                    out, t, design_policy if design_policy_applies else "random",
                    time.perf_counter() - acquisition_started,
                    len(acq_rows) if design_policy_applies else 1,
                )
                if round_acquisition_status != "acquisition_failed":
                    tic = time.perf_counter()
                    public.append(session.observe(design))
                    timings["observation"] += time.perf_counter() - tic
                    append_json(out / "observations.jsonl", public[-1])
            if round_acquisition_status == "acquisition_failed":
                # Acquisition guard: no observation, so the data, x and y are unchanged;
                # the round keeps the carried fits, with no refit and no fits/tNN files
                # for them.
                refit_skipped_rounds.append(t)
            else:
                x = matrix([row["x"] for row in public])
                y = np.array([row["y"] for row in public])
                fits = []
                for hypothesis in catalogue:
                    print(f"FIT START t={t} model={hypothesis.model_id}", flush=True)
                    try:
                        result = _timed_fit(
                            out, t, fit_timeout_seconds,
                            hypothesis,
                            x,
                            y,
                            seed_for("chem_smc", args.seed, t, rng_model_id(hypothesis)),
                            transform,
                            response_unit,
                            settings,
                            interface=interface,
                            **({"fit_guards": fit_guards} if fit_guards != "off" else {}),
                        )
                    except FitTimeout as exc:
                        timed_out_ids.add(hypothesis.model_id)
                        failure = {"model_id": hypothesis.model_id, "round": t, **exc.as_record()}
                        # Failure-class accounting: every candidate_failures record names its class.
                        failure["failure_class"] = failure_class(status=failure["status"])
                        candidate_failures.append(failure)
                        append_json(out / "candidate_failures.jsonl", failure)
                        timings["fit"] += exc.elapsed_seconds
                        if free:
                            view = _packet_view(hypothesis, names, grammar_names)
                            known_ids.add(view.model_id)
                            archive.append(_archive_entry(view, t, "fit_failed", {"error": str(exc)},
                                                          note="fit_timeout"))
                        continue
                    except CountInferenceFailure as exc:
                        if fit_guards == "off" or exc.cause_code != "nonfinite_smc_weights":
                            raise
                        guard_failed_ids.add(hypothesis.model_id)
                        failure = {"model_id": hypothesis.model_id, "round": t,
                                   "status": "fit_failed", "retirement_kind": exc.cause_code, **exc.as_record()}
                        failure["failure_class"] = failure_class(status=failure["status"])
                        candidate_failures.append(failure)
                        append_json(out / "candidate_failures.jsonl", failure)
                        if free:
                            view = _packet_view(hypothesis, names, grammar_names)
                            known_ids.add(view.model_id)
                            archive.append(_archive_entry(
                                view, t, "fit_failed", {"error": str(exc)},
                                note=exc.cause_code,
                            ))
                        continue
                    timings["fit"] += result.seconds
                    fits.append(result)
                    stem = f"t{t:02d}_{hypothesis.model_id}"
                    np.savez_compressed(out / "fits" / (stem + ".npz"), **result.particles)
                    write_json(
                        out / "fits" / (stem + ".json"),
                        {
                            **hypothesis.as_dict(),
                            "log_evidence": result.log_evidence,
                            "per_chain_log_evidence": result.per_chain_log_evidence,
                            "diagnostics": result.diagnostics,
                            "fit_seconds": result.seconds,
                        },
                    )
                    print(f"FIT DONE t={t} logZ={result.log_evidence:.6f}", flush=True)
                fit_count += len(fits)
                if timed_out_ids or guard_failed_ids:
                    catalogue = [h for h in catalogue
                                 if h.model_id not in timed_out_ids | guard_failed_ids]
                if not fits and not (t == 0 and (initial_proposals or frozen_pool is not None)):
                    raise ValueError("all ChemBench models failed fitting")
            if t == 0 and frozen_pool is not None:
                # Frozen pool track: the frozen pool's members enter round 0 through the burst's
                # own fit (seed stream chem_smc_proposed, round 0) and bookkeeping, with no proposer
                # call and without the admission gates the generation run already applied. A
                # member that fails to fit fails the cell: no cell starts from a smaller pool.
                pool_events = []
                for candidate in frozen_pool.hypotheses:
                    print(f"FIT START t=0 model={candidate.model_id} (frozen pool)", flush=True)
                    try:
                        result = _timed_fit(
                            out, 0, fit_timeout_seconds,
                            candidate,
                            x,
                            y,
                            seed_for("chem_smc_proposed", args.seed, 0, rng_model_id(candidate)),
                            transform,
                            response_unit,
                            settings,
                            interface=interface,
                            **({"fit_guards": fit_guards} if fit_guards != "off" else {}),
                        )
                    except Exception as exc:
                        raise FrozenPoolMemberFitFailure(candidate.model_id, exc) from exc
                    item = {**candidate.as_dict(), "source": "initial_pool_file",
                            "status": FROZEN_POOL_FITTED_STATUS,
                            "log_evidence": result.log_evidence,
                            "model_id_preimage": candidate.model_id_preimage}
                    pool_events.append(item)
                    fit_count += 1
                    timings["fit"] += result.seconds
                    fits.append(result)
                    catalogue.append(candidate)
                    known_ids.add(candidate.model_id)
                    born_round[candidate.model_id] = 0
                    stem = f"t00_{candidate.model_id}"
                    np.savez_compressed(out / "fits" / (stem + ".npz"), **result.particles)
                    write_json(
                        out / "fits" / (stem + ".json"),
                        {
                            **candidate.as_dict(),
                            "log_evidence": result.log_evidence,
                            "per_chain_log_evidence": result.per_chain_log_evidence,
                            "diagnostics": result.diagnostics,
                            "fit_seconds": result.seconds,
                        },
                    )
                    print(f"FIT DONE t=0 logZ={result.log_evidence:.6f} (frozen pool)", flush=True)
                    archive.append(_archive_entry(candidate, 0, "active",
                                                  {"admitted_round": 0, "active_rounds": [0]}))
                # The burst's round-0 retention order; a generated pool is already retained.
                fits, retired = retain(fits, ACTIVE_MAXIMUM)
                if retired:
                    raise ValueError("frozen pool holds more members than the active maximum")
                catalogue = [f.hypothesis for f in fits]
                initial_catalogue_ids = {f.hypothesis.model_id for f in fits}
                initial_pool_count = len(fits)
                write_final_proposal_json(out / "proposals" / "t00_initial_pool.json", {
                    "model_id_schema": MODEL_ID_SCHEMA,
                    "notation_normal_form": NOTATION_NORMAL_FORM,
                    "source": "initial_pool_file", "cell": args.cell,
                    "initial_pool_file": frozen_pool.name,
                    "initial_pool_sha256": frozen_pool.sha256,
                    "pool_id": frozen_pool.block["pool_id"],
                    "round0_llm_calls_inside_cell": 0,
                    "public_observations": public,
                    "events": pool_events,
                    "file_after_ids": [h.model_id for h in frozen_pool.hypotheses],
                    "after_ids": [f.hypothesis.model_id for f in fits],
                })
            if free and (t or initial_proposals):
                packet_fits = list(fits)
                previous_ids = [f.hypothesis.model_id for f in fits]
                prior_probe_cache = {}
                events, transports, waves = [], [], []
                # Model-id preimage record: each event's own `model_id_preimage`, added only to the
                # written record (the burst's and, with failure-class accounting, a later round's).
                event_preimages = []
                admitted_ids = []
                wave_lock = threading.Lock()
                gauge = {"in_flight": 0, "max_in_flight": 0}
                attempt_records = []
                fit_attempt_count = 0

                def make_packet(members=None, *, t=t, y=y, packet_fits=packet_fits):
                    """This round's serialized belief packet, or a refill wave's over `members`."""
                    # Burst refill: `members` (a round-0 refill wave's intermediate pool) replaces
                    # the round's starting pool; None - every other caller - is that pool.
                    packet = build_belief_packet(
                        [
                            {
                                "obs_id": i,
                                "round": 0 if i < d0_count else i - d0_count + 1,
                                **{n: float(row["x"][n]) for n in names},
                                "y": float(row["y"]),
                            }
                            for i, row in enumerate(public)
                        ],
                        [
                            PoolMember(
                                hypothesis=_packet_view(f.hypothesis, names, grammar_names),
                                particles=f.particles,
                                log_evidence=f.log_evidence,
                                response_unit=response_unit,
                                response_center=0.0,
                                born_round=born_round.get(f.hypothesis.model_id, 0),
                                log_evidence_computed_round=t,
                                log_evidence_n_obs=len(y),
                            )
                            for f in (packet_fits if members is None else members)
                        ],
                        archive,
                        # Constraint disclosure: shared constraints plus the naming rule.
                        _disclosed_constraints(cell_constraints(
                            args.cell,
                            args.family,
                            allowed_families=CHEM_ALLOWED_FAMILIES,
                            initial_structures=[] if t == 0 and catalogue_mode == "off" else ["b0"],
                            input_names=grammar_names,
                        ), interface, grammar_names),
                        interface,
                        t,
                    )
                    return serialize_packet(packet)

                def admit(candidate, *, t=t, x=x, prior_probe_cache=prior_probe_cache):
                    """Admission hook: prior-probe a candidate; PriorProbeRejection on failure."""
                    # Prior-probe admission: the prior probe as the proposer's admission hook (see
                    # core_count for the rule); a known id and a family this cell
                    # refuses are left to the loop below.
                    hypothesis = ScalarHypothesis(
                        candidate.expression, candidate.family, candidate.parameters, candidate.input_names
                    )
                    key = hypothesis.model_id
                    if key in known_ids or family_refusal(args.cell, hypothesis.family, args.family) is not None:
                        return
                    probe, _ = _prior_probe(
                        hypothesis, interface, response_unit, observed_inputs=x,
                        **({"rng_model_id": rng_model_id(hypothesis)} if mask_input_names else {}))
                    if probe["passed"]:
                        prior_probe_cache[key] = probe
                        return
                    known_ids.add(key)
                    archive.append(
                        _archive_entry(
                            hypothesis, t, "rejected",
                            {"validation_message": probe.get("detail", "non-finite under prior probe")},
                        )
                    )
                    raise PriorProbeRejection(probe["detail"])

                def admit_locked(candidate, *, admit=admit, wave_lock=wave_lock):
                    """`admit` under the burst wave lock."""
                    with wave_lock:
                        admit(candidate)

                def burst_call(wave, index, *, packet_text, mock_offset, burst=True, t=t,
                               wave_lock=wave_lock, gauge=gauge, attempt_records=attempt_records,
                               admit_locked=admit_locked):
                    """One round-0 burst call: (candidates, transport record), with retries."""
                    # Burst refill: `burst` is "refill" for a refill wave (1, 2), True for wave 0.
                    context = {
                        "run_id": str(out), "world_seed": args.seed, "cell": args.cell,
                        "proposal_rep": 0, "round": t,
                        "burst_call_index": index, "burst_wave": wave,
                    }
                    if proposer is None:
                        # Mock burst slots continue across the single refill, using
                        # the round path's existing modulo-20 offset convention.
                        form, mock_family = mock_schedule[(mock_offset + index) % MOCK_SCHEDULE_ROUNDS]
                        if synthetic_forms is not None:
                            # synthetic_v1 continues the same way over its long list.
                            form, mock_family = synthetic_forms[mock_offset + index], args.family
                        return [ScalarHypothesis(
                            form, mock_family, mock_parameters(form), grammar_names)], {
                            "provider": "none", "mock": True, "api_request_attempted": False,
                            "policy": "P1", "status": "parsed", "calls": 0, "repairs": 0,
                            "rejected_candidates": [], "note": "", **context,
                            **composed_system_prompt(burst=burst)[1], **packet_digest(packet_text),
                            **({"mock_schedule": mock_schedule_name}
                               if synthetic_forms is not None else {}),
                        }
                    for retry in range(BURST_RETRIES + 1):
                        with wave_lock:
                            gauge["in_flight"] += 1
                            gauge["max_in_flight"] = max(
                                gauge["max_in_flight"], gauge["in_flight"])
                        try:
                            candidates, record = proposer.propose_one(
                                packet_text,
                                context={**context, **({"burst_retry": retry} if retry else {})},
                                initial_family=args.family,
                                initial_structures=(seed_structure,),
                                repair_attempts=1,
                                allowed_families=CHEM_ALLOWED_FAMILIES,
                                admit=admit_locked,
                                max_rounds=args.rounds,
                                burst=burst,
                                slots=stage_slots,
                            )
                        finally:
                            with wave_lock:
                                gauge["in_flight"] -= 1
                        with wave_lock:
                            attempt_records.append({
                                "wave": wave, "index": index, "retry": retry, "transport": record})
                        rate_limited = record["status"] != "parsed" and any(
                            item.get("http_status") == 429
                            for item in record["rejected_candidates"])
                        if not rate_limited or retry == BURST_RETRIES:
                            break
                        time.sleep(BURST_BACKOFF_SECONDS * 2 ** retry)
                    return [
                        ScalarHypothesis(c.expression, c.family, c.parameters, c.input_names)
                        for c in candidates
                    ], record

                def proposal_batches(*, t=t, make_packet=make_packet, admit=admit,
                                     burst_call=burst_call, transports=transports,
                                     waves=waves, admitted_ids=admitted_ids, events=events,
                                     fits=fits):
                    """Yield (proposals, transport record): per later round, or per burst wave."""
                    if t:
                        review = t in review_rounds
                        packet_text = make_packet()
                        (out / "packets" / f"t{t:02d}.json").write_text(packet_text)
                        context = {
                            "run_id": str(out), "world_seed": args.seed, "cell": args.cell,
                            "proposal_rep": 0, "round": t,
                        }
                        if proposer is None:
                            # Round cap: index the FIXED schedule; the
                            # entry-time refusal in `run` already rules out t > 20, so
                            # this is the local restatement of that invariant (never a
                            # silent wrap-around, which is what `(t - 1) % 4` was).
                            if t > MOCK_SCHEDULE_ROUNDS:
                                raise ValueError(
                                    f"mock proposal schedule has {MOCK_SCHEDULE_ROUNDS} fixed entries; "
                                    f"round {t} is outside it"
                                )
                            # Multi-candidate proposals: k = 1 takes the one entry this round
                            # has always taken; k > 1 takes the next k entries of the
                            # SAME fixed 20-entry schedule, wrapping nowhere - the
                            # entry-time refusal above still rules out a round past it.
                            forms = [mock_schedule[t - 1]]
                            for offset in range(1, proposal_limit):
                                forms.append(mock_schedule[(t - 1 + offset) % MOCK_SCHEDULE_ROUNDS])
                            if synthetic_forms is not None:
                                # synthetic_v1's slots for this round instead.
                                forms = []
                                for slot in range(1, proposal_limit + 1):
                                    forms.append((synthetic_round_form(
                                        synthetic_forms, t, slot, [form for form, _ in forms]),
                                        args.family))
                            proposals = [
                                ScalarHypothesis(form, mock_family, mock_parameters(form), grammar_names)
                                for form, mock_family in forms
                            ]
                            transport_record = {
                                "provider": "none",
                                "mock": True,
                                "api_request_attempted": False,
                                "policy": "P1",
                                "status": "parsed",
                                "calls": 0,
                                "repairs": 0,
                                "rejected_candidates": [],
                                "note": "",
                                **composed_prompt_identity(proposal_limit, archive_review=review),
                                **packet_digest(packet_text),
                            }
                            if review:
                                # The mock's revive list - one valid and one
                                # invalid id when the archive holds a retired model.
                                transport_record.update(
                                    review_reply_shape="mock",
                                    revive_requested=mock_revive_request(
                                        archive, retired_now,
                                        json.loads(packet_text).get("best_model_id")))
                        else:
                            proposals, transport_record = proposer.propose_one(
                                packet_text,
                                context=context,
                                initial_family=args.family,
                                initial_structures=(seed_structure,),
                                repair_attempts=repair_attempts,
                                allowed_families=CHEM_ALLOWED_FAMILIES,
                                admit=admit,
                                # Round cap: this run's own horizon, so a
                                # 60-round ChemBench run is not refused at t=21 by a
                                # cap that belonged to BoxingGym's 20-round CLI.
                                max_rounds=args.rounds,
                                proposal_limit=proposal_limit,
                                slots=stage_slots,
                                # An archive review round's request and reply.
                                **({"archive_review": True} if review else {}),
                            )
                            proposals = [
                                ScalarHypothesis(c.expression, c.family, c.parameters, c.input_names)
                                for c in proposals
                            ]
                        yield proposals, transport_record
                        return
                    shortfall = initial_proposals
                    mock_offset = 0
                    # Burst refill: at most three waves. A refill wave (1, 2) asks for exactly the
                    # shortfall; its packet shows the intermediate retained pool (the round's
                    # starting pool plus the fits admitted so far), and `fits` is untouched.
                    for wave in INITIAL_BURST_WAVES:
                        if shortfall <= 0:
                            break
                        shown = burst_refill_pool(fits, ACTIVE_MAXIMUM) if wave else None
                        packet_text = make_packet(shown)
                        if wave == 0:
                            (out / "packets" / "t00_burst.json").write_text(packet_text)
                        (out / "packets" / f"t00_burst_w{wave}.json").write_text(packet_text)
                        with ThreadPoolExecutor(
                                max_workers=min(DEFAULT_BURST_CONCURRENCY, shortfall)) as pool:
                            results = list(pool.map(
                                partial(burst_call, wave, packet_text=packet_text,
                                        mock_offset=mock_offset, burst="refill" if wave else True),
                                range(shortfall)))
                        proposals = []
                        wave_events, wave_transports = len(events), len(transports)
                        for index, (candidates, record) in enumerate(results):
                            proposals.extend(candidates)
                            transports.append({"wave": wave, "index": index, "transport": record})
                        yield proposals, {
                            "calls": sum(record["calls"] for _, record in results),
                            "repairs": sum(record["repairs"] for _, record in results),
                        }
                        accepted = len(admitted_ids)
                        waves.append(burst_wave_record(
                            wave, shortfall, accepted, events[wave_events:],
                            transports[wave_transports:], packet_text))
                        mock_offset += shortfall
                        shortfall -= accepted

                # An archive review round's revival record (None on every other round).
                revival = None
                for proposals, transport_record in proposal_batches():
                    if t:
                        write_proposed_json(out / "proposals" / f"t{t:02d}.json", {
                            "round": t, "status": "proposed", "transport": transport_record,
                            "proposed_ids": [c.model_id for c in proposals],
                            "candidates": [c.as_dict() for c in proposals],
                        })
                    admitted_ids.clear()
                    likelihood_probes = {}
                    if t and t in review_rounds:
                        # The reply's revive names, checked against the archive (no
                        # call is spent on an invalid one), and each valid model refit from the
                        # prior on all public data with the active members' own stream. The refit
                        # joins `fits` here, so the retention below is ONE truncation of the
                        # active pool, the revived models and the round's new candidates.
                        revival_started = time.perf_counter()
                        requested = transport_record.get("revive_requested", [])
                        valid_ids, invalid = validate_revive_request(
                            requested, archive, retired_now,
                            {_packet_view(f.hypothesis, names, grammar_names).model_id for f in fits})
                        refit_results, refit_failures = [], []
                        for view_id in valid_ids:
                            hypothesis = retired_now[view_id]
                            revival_attempts[view_id] = revival_attempts.get(view_id, 0) + 1
                            print(f"FIT START t={t} model={hypothesis.model_id} (revival)", flush=True)
                            try:
                                result = _timed_fit(
                                    out, t, fit_timeout_seconds, hypothesis, x, y,
                                    seed_for("chem_smc", args.seed, t, rng_model_id(hypothesis)),
                                    transform, response_unit, settings, interface=interface,
                                    **({"fit_guards": fit_guards} if fit_guards != "off" else {}),
                                )
                            except Exception as exc:  # noqa: BLE001 - it stays retired
                                if isinstance(exc, FitTimeout):
                                    timings["fit"] += exc.elapsed_seconds
                                refit_failures.append({"id": view_id, "scope": "revival",
                                                       "error": f"{type(exc).__name__}: {exc}"})
                                continue
                            timings["fit"] += result.seconds
                            fit_count += 1
                            fits.append(result)
                            catalogue.append(hypothesis)
                            refit_results.append({"id": view_id,
                                                  "log_evidence": float(result.log_evidence)})
                            stem = f"t{t:02d}_{hypothesis.model_id}"
                            np.savez_compressed(out / "fits" / (stem + ".npz"), **result.particles)
                            write_json(out / "fits" / (stem + ".json"), {
                                **hypothesis.as_dict(), "log_evidence": result.log_evidence,
                                "per_chain_log_evidence": result.per_chain_log_evidence,
                                "diagnostics": result.diagnostics, "fit_seconds": result.seconds,
                                "archive_review_revival": True,
                            })
                            print(f"FIT DONE t={t} logZ={result.log_evidence:.6f} (revival)",
                                  flush=True)
                        revival = {
                            "round": t, "requested_ids": requested, "valid_ids": valid_ids,
                            "invalid": invalid, "refit_results": refit_results,
                            "refit_failures": refit_failures,
                            "review_reply_shape": transport_record.get("review_reply_shape"),
                            # No revive array in the reply (a bare candidates array, no key, a
                            # failed call): nothing revived, and no repair turn was spent on it.
                            "revive_missing": ("revive_requested" not in transport_record
                                               or bool(transport_record.get("revive_missing"))),
                            **({"revive_note": transport_record["revive_note"]}
                               if "revive_note" in transport_record else {}),
                            "wall_seconds": time.perf_counter() - revival_started,
                        }
                    for candidate in proposals:
                        # Failure-class accounting (a project decision of 2026-09-23): every
                        # round keeps one outcome event per candidate, in proposal order; only a
                        # burst event carries its `source`.
                        item = candidate.as_dict()
                        if t == 0:
                            item["source"] = "initial_burst"
                        events.append(item)
                        event_preimages.append(candidate.model_id_preimage)
                        if candidate.model_id in known_ids:
                            item["status"] = "duplicate_not_refitted"
                            archive.append(
                                _archive_entry(
                                    candidate, t, "duplicate",
                                    {
                                        "matched_model_id": candidate.model_id,
                                        "matched_expression": candidate.expression,
                                    },
                                    note=(
                                        RETIRED_REFUSAL_NOTE
                                        if candidate.model_id in retired_ids_ever
                                        else None
                                    ),
                                )
                            )
                            continue
                        # The F0 judgement, one candidate at a time.
                        # Placed AFTER the duplicate check so a re-proposal of an
                        # already-archived model stays a `duplicate` (the archive is
                        # the "do not re-propose" binding, whatever put a model in
                        # it) and BEFORE the prior probe and the fit, so a candidate
                        # this cell may not hold is refused without costing an SMC
                        # run - the place `judge_candidate` occupies on the LLM path.
                        # `rejected` + `known_ids` is exactly how this loop already
                        # treats a candidate refused before any fit (the prior-probe
                        # branch immediately below), which keeps the invariant that
                        # every archived id is in `known_ids`.
                        refusal = family_refusal(args.cell, candidate.family, args.family)
                        if refusal is not None:
                            item.update(status="rejected_family", reason=refusal)
                            known_ids.add(candidate.model_id)
                            reason = {"validation_message": refusal}
                            archive.append(_archive_entry(candidate, t, "rejected", reason))
                            continue
                        probe = prior_probe_cache.pop(candidate.model_id, None)
                        if probe is None:
                            probe, _ = _prior_probe(
                                candidate, interface, response_unit, observed_inputs=x,
                                **({"rng_model_id": rng_model_id(candidate)} if mask_input_names else {}))
                        item["prior_probe"] = probe
                        if not probe["passed"]:
                            item["status"] = "rejected_prior_probe"
                            known_ids.add(candidate.model_id)
                            archive.append(
                                _archive_entry(
                                    candidate, t, "rejected",
                                    {
                                        "validation_message": probe.get(
                                            "detail", "non-finite under prior probe"
                                        ),
                                    },
                                )
                            )
                            continue
                        if fit_guards != "off":
                            likelihood_probe = loglik_prior_probe(
                                candidate, x, y, interface, count=PRIOR_PROBE_SAMPLES,
                                seed=PRIOR_PROBE_SEED ^ int(rng_model_id(candidate), 16),
                                prior_draws=_prior_particles(candidate, PRIOR_PROBE_SAMPLES,
                                    **({"rng_model_id": rng_model_id(candidate)} if mask_input_names else {})),
                                model_builder=partial(
                                    build_model, candidate, x, y, transform, response_unit, interface),
                            )
                            likelihood_probes[candidate.model_id] = likelihood_probe
                            item["prior_probe"] = {**probe, "loglik_prior_probe": likelihood_probe}
                            if not likelihood_probe["passed"]:
                                item["status"] = "rejected_loglik_prior_probe"
                                known_ids.add(candidate.model_id)
                                archive.append({**_archive_entry(candidate, t, "rejected", {
                                    "validation_message": likelihood_probe["reason"],
                                }), "prior_probe": {"loglik_prior_probe": likelihood_probe}})
                                continue
                        fit_attempt_count += 1
                        try:
                            print(
                                f"FIT START t={t} model={candidate.model_id} (proposed)", flush=True
                            )
                            proposed_result = _timed_fit(
                                out, t, fit_timeout_seconds,
                                candidate,
                                x,
                                y,
                                seed_for("chem_smc_proposed", args.seed, t, rng_model_id(candidate)),
                                transform,
                                response_unit,
                                settings,
                                interface=interface,
                                **({"fit_guards": fit_guards} if fit_guards != "off" else {}),
                            )
                        except FitTimeout as exc:
                            item.update(exc.as_record(), status="fit_failed")
                            known_ids.add(candidate.model_id)
                            timed_out_ids.add(candidate.model_id)
                            failure = {"model_id": candidate.model_id, "round": t, **exc.as_record()}
                            failure["failure_class"] = failure_class(status=failure["status"])
                            candidate_failures.append(failure)
                            append_json(out / "candidate_failures.jsonl", failure)
                            timings["fit"] += exc.elapsed_seconds
                            archive.append(_archive_entry(candidate, t, "fit_failed", {"error": str(exc)},
                                                          note="fit_timeout"))
                            continue
                        except CountInferenceFailure as exc:
                            item.update(status="fit_or_predict_failed", error=f"{type(exc).__name__}: {exc}")
                            known_ids.add(candidate.model_id)
                            reason = {"error": f"{type(exc).__name__}: {exc}"}
                            if fit_guards != "off" and exc.cause_code == "nonfinite_smc_weights":
                                guard_failed_ids.add(candidate.model_id)
                                failure = {"model_id": candidate.model_id, "round": t,
                                           "status": "fit_failed", "retirement_kind": exc.cause_code, **exc.as_record()}
                                failure["failure_class"] = failure_class(status=failure["status"])
                                candidate_failures.append(failure)
                                append_json(out / "candidate_failures.jsonl", failure)
                            archive.append(_archive_entry(
                                candidate, t, "fit_failed", reason,
                                note=(exc.cause_code if fit_guards != "off"
                                      and exc.cause_code == "nonfinite_smc_weights" else None),
                            ))
                            continue
                        except Exception as exc:  # noqa: BLE001 - a failed candidate, not a failed run
                            item.update(status="fit_or_predict_failed", error=f"{type(exc).__name__}: {exc}")
                            known_ids.add(candidate.model_id)
                            archive.append(
                                _archive_entry(
                                    candidate, t, "fit_failed",
                                    {"error": f"{type(exc).__name__}: {exc}"},
                                )
                            )
                            continue
                        item.update(status="fitted", log_evidence=proposed_result.log_evidence)
                        fit_count += 1
                        timings["fit"] += proposed_result.seconds
                        fits.append(proposed_result)
                        catalogue.append(candidate)
                        known_ids.add(candidate.model_id)
                        born_round[candidate.model_id] = t
                        stem = f"t{t:02d}_{candidate.model_id}"
                        np.savez_compressed(out / "fits" / (stem + ".npz"), **proposed_result.particles)
                        write_json(
                            out / "fits" / (stem + ".json"),
                            {
                                **candidate.as_dict(),
                                "log_evidence": proposed_result.log_evidence,
                                "per_chain_log_evidence": proposed_result.per_chain_log_evidence,
                                "diagnostics": proposed_result.diagnostics,
                                "fit_seconds": proposed_result.seconds,
                            },
                        )
                        print(
                            f"FIT DONE t={t} logZ={proposed_result.log_evidence:.6f} (proposed)",
                            flush=True,
                        )
                        archive.append(
                            _archive_entry(
                                candidate, t, "active",
                                {"admitted_round": t, "active_rounds": [t]},
                            )
                        )
                        admitted_ids.append(candidate.model_id)
                # Pool retirement,
                # the CORE rule verbatim - `count_inference.retain` keeps the
                # `core_count.ACTIVE_MAXIMUM` (12) highest-evidence members
                # and breaks ties on the stable `model_id`, exactly as
                # `core_count.run`/`boxinggym_core.run` call it. A retired
                # member goes to the archive as `retired_by_evidence` and
                # stays in `known_ids`, so it can never be re-proposed.
                #
                # Later rounds apply this ONLY when the pool exceeds the maximum.
                # `retain` also SORTS, and the pool order feeds
                # `predictive_summary`'s draw assignment, so calling it
                # unconditionally would reorder (not change) every sub-12 free
                # run's predictive samples for no reason; guarding it leaves
                # every free run of twelve or fewer members - the whole earlier
                # acceptance scale - byte-identical. The new initial wave uses
                # CORE's final retention ordering, including a sub-12 pool.
                retired_ids = []
                if not fits:
                    raise ValueError("no models remain after the parallel proposal wave")
                if len(fits) > ACTIVE_MAXIMUM or t == 0:
                    fits, retired = retain(fits, ACTIVE_MAXIMUM)
                    catalogue = [f.hypothesis for f in fits]
                    cutoff = float(fits[-1].log_evidence)
                    for result in retired:
                        # The archive, `known_ids` and the packet's pool all
                        # speak the grammar identity space (`_packet_view`), so
                        # a retired FIXED-catalogue member is archived under
                        # the same id the packet would have shown - a
                        # RegressionHypothesis has no `.expression` at all.
                        view = _packet_view(result.hypothesis, names, grammar_names)
                        # A revived model that missed the cut again stays retired,
                        # its new entry saying so; it was not in the pool, so it is not one of
                        # this round's retirements.
                        retired_again = revival is not None and view.model_id in revival["valid_ids"]
                        archive.append(
                            _archive_entry(
                                view, born_round.get(result.hypothesis.model_id, 0),
                                "retired_by_evidence",
                                retirement_reason(
                                    result.log_evidence, cutoff,
                                    active_rounds.get(result.hypothesis.model_id, []), t, len(y),
                                ),
                                note=(f"revival attempt {revival_attempts[view.model_id]} at archive "
                                      f"review round {t}; retired again" if retired_again else None),
                            )
                        )
                        known_ids.add(view.model_id)
                        retired_ids_ever.add(view.model_id)
                        retired_now[view.model_id] = result.hypothesis
                        if retired_again:
                            continue
                        retired_ids.append(view.model_id)
                        retired_log.append(
                            {
                                "round": t,
                                "model_id": view.model_id,
                                "expression": view.expression,
                                "family": view.family,
                                "log_evidence_at_retirement": float(result.log_evidence),
                                "cutoff_log_evidence": cutoff,
                            }
                        )
                if not fits:
                    raise ValueError("no models remain after the parallel proposal wave")
                if revival is not None:
                    # A revived model in the retained pool re-enters (its archive
                    # entry says active again); the others stay retired (entries written above).
                    pool_ids = {_packet_view(f.hypothesis, names, grammar_names).model_id: f
                                for f in fits}
                    refit_ids = [item["id"] for item in revival["refit_results"]]
                    re_entered = [view_id for view_id in refit_ids if view_id in pool_ids]
                    for view_id in re_entered:
                        del retired_now[view_id]
                        view = _packet_view(pool_ids[view_id].hypothesis, names, grammar_names)
                        archive.append(_archive_entry(
                            view, born_round.get(pool_ids[view_id].hypothesis.model_id, 0), "active",
                            {"admitted_round": t, "active_rounds": [t]},
                            note=(f"revived at archive review round {t} (attempt "
                                  f"{revival_attempts[view_id]})")))
                    revival.update(
                        re_entered_ids=re_entered,
                        retired_again_ids=[view_id for view_id in refit_ids if view_id not in pool_ids],
                        revival_attempts={view_id: revival_attempts[view_id]
                                          for view_id in revival["valid_ids"]},
                        cutoff_log_evidence=float(fits[-1].log_evidence),
                        active_pool_after=list(pool_ids))
                    write_json(out / "revival" / f"t{t:02d}.json", revival)
                    revival_events.append(revival)
                if t:
                    proposed_rounds.add(t)
                    # Failure-class accounting: each event names its class (None when fitted) and is
                    # counted once beside the round's transport record, as a burst's events are.
                    for event in events:
                        event["failure_class"] = failure_class(status=event["status"])
                    for kind, value in failure_class_counts(
                            events, [{"transport": transport_record}]).items():
                        failure_counts[kind] += value
                    proposal_log.append(
                        {
                            "round": t,
                            "transport": transport_record,
                            "proposed_ids": [c.model_id for c in proposals],
                            "admitted_ids": admitted_ids,
                            # Failure-class accounting: the burst event's key set without `source`,
                            # and its `model_id_preimage` added here as the burst record adds it.
                            "events": [{**event, "model_id_preimage": preimage}
                                       for event, preimage in zip(events, event_preimages, strict=True)],
                            **({"loglik_prior_probes": likelihood_probes} if fit_guards != "off" else {}),
                            # Pool retirement; empty on every round that
                            # retired nothing (the original shape had no key at all
                            # because nothing could ever be retired).
                            "retired_ids": retired_ids,
                            "active_pool_after": [
                                _packet_view(f.hypothesis, names, grammar_names).model_id
                                for f in fits
                            ],
                            # With paragraph v2, how many of the round's candidates
                            # entered (at most three, no replacement call) and how many were
                            # duplicates; on a review round, what was revived.
                            **({"admitted_count": len(admitted_ids),
                                "duplicate_count": sum(event["status"] == "duplicate_not_refitted"
                                                       for event in events)}
                               if composition == "v2" else {}),
                            **({"archive_review": True,
                                "revival_valid_ids": revival["valid_ids"],
                                "revival_re_entered_ids": revival["re_entered_ids"]}
                               if revival is not None else {}),
                        }
                    )
                    write_final_proposal_json(out / "proposals" / f"t{t:02d}.json", proposal_log[-1])
                else:
                    initial_catalogue_ids = {f.hypothesis.model_id for f in fits}
                    initial_pool_count = len(fits)
                    burst_wave_count = len(waves)
                    new_ids = sorted(initial_catalogue_ids - set(previous_ids))
                    # Failure-class accounting: each burst event names its class (None when fitted).
                    for event in events:
                        event["failure_class"] = failure_class(status=event["status"])
                    for kind, value in failure_class_counts(events, transports).items():
                        failure_counts[kind] += value
                    burst_record = {
                        "model_id_schema": MODEL_ID_SCHEMA,
                        "notation_normal_form": NOTATION_NORMAL_FORM,
                        "public_observations": public, "mock": proposer is None,
                        "transports": transports, "cell": args.cell,
                        "raw_response": [
                            {key: value for key, value in event.items()
                             if key in ("expression", "family", "parameters")}
                            for event in events
                        ],
                        "events": [{**event, "model_id_preimage": preimage}
                                   for event, preimage in zip(events, event_preimages, strict=True)],
                        "valid_candidate_count": len(events),
                        "duplicate_count": sum(
                            event["status"] == "duplicate_not_refitted" for event in events),
                        "fit_attempt_count": fit_attempt_count,
                        "fit_success_count": sum(event["status"] == "fitted" for event in events),
                        "waves": waves, "requested": initial_proposals,
                        **wave_event_counts(events, transports, initial_proposals),
                        "max_in_flight": gauge["max_in_flight"],
                        "concurrency": DEFAULT_BURST_CONCURRENCY,
                        "initial_catalogue": catalogue_mode,
                        "initial_burst_refill_policy": INITIAL_BURST_REFILL_POLICY,
                        "base_prompt_sha256": refill_identity["base_prompt_sha256"],
                        "burst_refill_paragraph_sha256": refill_identity["burst_refill_paragraph_sha256"],
                        "before_ids": previous_ids,
                        "after_ids": [f.hypothesis.model_id for f in fits],
                        "admitted_ids": new_ids, "retired_ids": retired_ids,
                        # Frozen pool track: only a --pool-generator run's burst is a pool file.
                        **({"frozen_pool": generation_block} if generation_block is not None else {}),
                    }
                    write_final_proposal_json(out / "proposals" / "t00_burst.json", burst_record)
                    pool_metadata = {
                        "initial_pool_source": "generated",
                        "initial_pool_count": initial_pool_count,
                        "initial_pool_sha256": hashlib.sha256(
                            (out / "proposals" / "t00_burst.json").read_bytes()).hexdigest(),
                    }
                    usage_total = {"input_tokens": 0, "output_tokens": 0}
                    usage_complete = True
                    attempts = sorted(attempt_records, key=lambda row: (
                        row["wave"], row["index"], row["retry"]))
                    for attempt in attempts:
                        record = attempt["transport"]
                        if record["calls"]:
                            usage_complete &= (bool(record.get("usage_total_complete", False))
                                               and len(record.get("responses", [])) == record["calls"])
                            for key, value in (record.get("usage_total") or {}).items():
                                usage_total[key] = usage_total.get(key, 0) + value
                    outcome_receipt = {
                        **pool_metadata, "schema": "chembench_initial_pool_receipt_v1",
                        "model_id_schema": MODEL_ID_SCHEMA,
                        "notation_normal_form": NOTATION_NORMAL_FORM,
                        "initial_pool_file": "proposals/t00_burst.json",
                        "initial_catalogue": catalogue_mode, "requested": initial_proposals,
                        "waves": waves,
                        "calls": sum(row["transport"]["calls"] for row in attempts),
                        "repairs": sum(row["transport"]["repairs"] for row in attempts),
                        "usage_total": usage_total, "usage_total_complete": usage_complete,
                        "attempts": [{key: value for key, value in row.items() if key != "transport"}
                                     | {key: row["transport"].get(key) for key in (
                                         "calls", "repairs", "usage_total", "usage_total_complete")}
                                     for row in attempts],
                    }
                    write_json(out / "initial_pool_receipt.json", outcome_receipt)
                    config.update(pool_metadata)
                    config["initial_pool"] = initial_pool_count
                    config["initial_pool_receipt"] = {
                        "path": "initial_pool_receipt.json",
                        "sha256": hashlib.sha256(
                            (out / "initial_pool_receipt.json").read_bytes()).hexdigest(),
                    }
                    write_json(out / "config.json", config)
                    proposal_log.append({
                        "round": t, "transport": {
                            "calls": sum(row["transport"]["calls"] for row in attempts),
                            "repairs": sum(row["transport"]["repairs"] for row in attempts),
                        },
                        "admitted_ids": new_ids,
                    })
            if t == 0 and pool_generator is not None:
                # Frozen pool track: the generation run ends with its round-0 burst. Its receipt
                # names the pool and the init digests (as decided); the
                # cost in USD is the driver ledger's (paid_api_calls x rate), not the runner's.
                write_json(out / "initial_pool_generation_receipt.json",
                           _frozen_pool_generation_receipt(
                               out, generation_block, args, attempt_records, proposer,
                               generation_started_at=generation_started_at))
                write_json(out / "summary.json", {
                    **({"label_receipt": label_receipt} if label_explicit else {}),
                    "status": "initial_pool_generated",
                    "task_id": task["id"], "cell": args.cell, "profile": args.profile,
                    "pool_id": generation_block["pool_id"],
                    "initial_pool_count": initial_pool_count,
                    "observations": session.calls,
                    "paid_api_calls": 0 if proposer is None else proposer.attempts,
                    "wall_seconds": time.perf_counter() - started,
                    "timing_seconds": timings,
                })
                # Frozen pool track: the generator selection is this run's alone.
                from .count_proposer import select_pool_generator

                select_pool_generator(None)
                # Round timing: the generation run's round 0 completed here.
                record_round_timing(out, t, round_started_at_unix, round_started)
                return
            if free:
                # Rounds each member of the FINAL pool of this round was
                # active for; read by the retirement archive entry above.
                for result in fits:
                    active_rounds.setdefault(result.hypothesis.model_id, []).append(t)
            tic = time.perf_counter()
            prediction = guarded_predictive_summary(
                fits, qx, seed_for("chem_prediction", args.seed, t), t, "evaluation"
            )
            prediction_excluded_models.extend(prediction.get("prediction_excluded_models", []))
            _record_prediction_pool(out, fits, qx, prediction, t, "prediction", len(public))
            arrays = {
                key: value for key, value in prediction.items() if isinstance(value, np.ndarray)
            }
            np.savez_compressed(out / "predictions" / f"t{t:02d}.npz", x=qx, **arrays)
            write_json(
                out / "predictions" / f"t{t:02d}.json",
                {
                    "round": t,
                    "observation_count": len(public),
                    "measure": prediction["measure"],
                    "single_model_id": prediction["single_model_id"],
                    "test_targets_used": 0,
                    **({"prediction_excluded_models": prediction["prediction_excluded_models"]}
                       if "prediction_excluded_models" in prediction else {}),
                },
            )
            timings["prediction"] += time.perf_counter() - tic
            # This round's contract checkpoint. The
            # official evaluation set cannot be scored yet - the truth is
            # unavailable until `finish()` - so the round's own pool is kept
            # and scored below, after the run is closed.
            checkpoints.append(
                {"round": t, "observation_count": len(public), "fits": list(fits)}
            )
            if round_acquisition_status is not None:
                # Acquisition guard: an all_designs_invalid or acquisition_failed
                # round says so.
                checkpoints[-1]["acquisition_status"] = round_acquisition_status
            # Round timing: every round that reaches here completed (fallback rounds included).
            record_round_timing(out, t, round_started_at_unix, round_started)
        session.finish()
        tic = time.perf_counter()
        reference = session.reference_values(evaluation)
        np.savez_compressed(out / "reference_after_finish.npz", **reference)
        scores = {}
        for target, truth in reference.items():
            values = np.asarray(truth)
            scores[target] = {
                "rmse": float(np.sqrt(np.mean((prediction["mean"] - values) ** 2))),
                "rmsle": float(
                    np.sqrt(np.mean((np.log1p(prediction["mean"]) - np.log1p(values)) ** 2))
                ),
            }
        observation = reference["heldout_observation"]
        logp, components = guarded_log_predictive(fits, qx, observation)
        scores["heldout_observation"]["nll_pdf"] = -float(logp.mean())
        np.savez_compressed(
            out / "final_raw_predictions.npz",
            x=qx,
            **arrays,
            observation_logpdf=logp,
            model_observation_logpdf=components,
        )
        write_json(
            out / "scores_after_finish.json",
            {
                "scores": scores,
                **({"prediction_excluded_models": prediction_excluded_models}
                   if prediction_excluded_models else {}),
                "primary_rate_is_not_noiseless_observation": True,
                "reference_used_for_fit_or_selection": False,
            },
        )
        # The official
        # evaluation set, scored ONCE, after the run is closed, on every
        # round's retained pool - then emitted as the plot contract's point
        # table. This is a SEPARATE set from `evaluation`/`qx` above (which
        # keeps its original size, seed and artifacts untouched), and it is the
        # baseline's own scoring set: same sampler, same count, same seed.
        official_reference = session.reference_values(official_designs)
        np.savez_compressed(
            out / "official_evaluation_inputs.npz", x=official_x, names=np.asarray(names)
        )
        np.savez_compressed(out / "official_reference_after_finish.npz", **official_reference)
        official_primary = np.asarray(official_reference[OFFICIAL_PRIMARY_TARGET], dtype=float)
        log_primary = np.log1p(official_primary)
        targets = {
            "log_primary": log_primary,
            # Population variance (ddof = 0): the baseline derived this same
            # quantity as E[L^2] - E[L]^2 from two constant laws, which is the
            # ddof = 0 form, and this code reproduces its nine recorded
            # per-task values to within 7e-16 relative.
            "log_primary_variance": float(log_primary.var()),
            "observed": np.asarray(official_reference[OFFICIAL_INTERVAL_TARGET], dtype=float),
        }
        if not np.isfinite(targets["log_primary_variance"]) or (
            targets["log_primary_variance"] <= 0
        ):
            raise ValueError("official evaluation truth has no positive log1p variance")
        point_rows = []
        for item in checkpoints:
            scored = official_checkpoint_scores(
                item["fits"],
                official_x,
                seed_for("chem_official_prediction", args.seed, item["round"]),
                targets,
                round_index=item["round"],
                summary_output=out,
                observation_count=item["observation_count"],
            )
            prediction_excluded_models.extend(scored.get("prediction_excluded_models", []))
            point_rows.append(
                {
                    "round": item["round"],
                    "observation_count": item["observation_count"],
                    "active_pool": len(item["fits"]),
                    **scored,
                }
            )
            if "acquisition_status" in item:
                # Acquisition guard status; write_points reads named keys only, so points.csv keeps
                # its column set.
                point_rows[-1]["acquisition_status"] = item["acquisition_status"]
            # The round's own pool, named and weighted (design decision
            # of 2026-09-18). Written unconditionally - it is a record of
            # what the run already did, so no option gates it - and in
            # checkpoint order, one line per `point_rows` entry.
            append_json(out / "pool_weights.jsonl", _pool_weights_row(
                item, names, **({"model_id_key": rng_model_id} if mask_input_names else {})))
        # `kl` on the FINAL checkpoint row by default.
        # The official evaluation set opens only after `finish()` (this run
        # just requested it, above), so no round's `kl` exists before this
        # point at all; every round's own pool IS kept, though, so the same
        # function can be applied to all of them after the fact, exactly the
        # way `nmse_mda` already is. `--kl-every-checkpoint` does that;
        # with it absent every earlier `point_rows` entry has no "kl" key and
        # `write_points` leaves it BLANK, as before. Either way the FINAL
        # checkpoint is the SAME call on the SAME inputs and is scored last,
        # so `kl_summary` below - summary.json's `kl` family and the double-precision
        # recompute artifact - does not depend on the option.
        kl_every_checkpoint = getattr(args, "kl_every_checkpoint", False)
        official_noiseless = np.asarray(
            official_reference[OFFICIAL_KL_TRUTH_CENTER], dtype=float
        )
        scored_rows = list(zip(checkpoints, point_rows, strict=True))
        if not kl_every_checkpoint:
            scored_rows = scored_rows[-1:]
        kl_summary = None
        for item, row in scored_rows:
            kl_summary = official_kl_summary(
                item["fits"], official_x, official_noiseless, args.noise
            )
            row["kl"] = kl_summary["kl"]
            if kl_every_checkpoint:
                # points.csv keeps its existing column set; these two are
                # `official_scores_after_finish.json` `rounds[i]` keys only.
                row["kl_quadrature_error"] = kl_summary["kl_quadrature_error"]
                row["kl_truncated_mass"] = kl_summary["kl_truncated_mass"]
            row.update({key: kl_summary[key] for key in
                        ("kl_status", "kl_undefined_truth_center_points") if key in kl_summary})
        if kl_summary["quadrature"] is not None:
            write_official_kl_particle_summary(
                out / "official_kl_particle_summary_after_finish.npz",
                checkpoints[-1]["fits"],
                official_x,
                kl_summary["quadrature"],
            )
        kl_undefined = {key: kl_summary[key] for key in
                        ("kl_status", "kl_undefined_truth_center_points") if key in kl_summary}
        write_json(
            out / "official_scores_after_finish.json",
            {
                "evaluation_set": {
                    "sampler": "adapters.chembench_observation.sample_designs",
                    "count": OFFICIAL_EVALUATION_COUNT,
                    "seed": OFFICIAL_EVALUATION_SEED,
                    "primary_target": OFFICIAL_PRIMARY_TARGET,
                    "interval_target": OFFICIAL_INTERVAL_TARGET,
                    "log1p_primary_variance": targets["log_primary_variance"],
                    "requested_after_finish_only": True,
                },
                "metric_definitions": {
                    "nmse_mda": (
                        "mean_Q[(log1p(pred) - log1p(primary_rate))^2] / "
                        "Var_Q[log1p(primary_rate)] (ddof=0); pred = BMA posterior predictive "
                        "mean. Same definition as the no-leak baseline's 224 contract rows."
                    ),
                    "nmsle": "the same statistic as nmse_mda, under the plot contract's own name",
                    "rmsle": "sqrt(mean_Q[(log1p(pred) - log1p(primary_rate))^2])",
                    "nll": (
                        "-mean log predictive density of the held-out OBSERVATION under the "
                        "round's Bayesian model average"
                    ),
                    "coverage90": "fraction of held-out observations inside [lower90, upper90]",
                    "width90": "mean (upper90 - lower90)",
                    "crps/is_0.1": "not defined on this path; left blank in points.csv",
                    "kl": (
                        "mean_Q[ KL( Normal(noiseless_observation(x_q), "
                        "(noise*noiseless_observation(x_q))**2) truncated at 0 || BMA ) ], "
                        "1-D Gauss-Legendre quadrature; defined for EVERY "
                        "checkpoint row (kl_every_checkpoint): the official set opens only "
                        "after finish(), so every row's value is that round's retained pool "
                        "re-scored afterwards, the same way nmse_mda is, and the final row is "
                        "the identical value the option-off run records. kl_quadrature_error "
                        "and kl_truncated_mass accompany each row here only; points.csv keeps "
                        "its kl column alone. NOT the same target as nmse_mda: see config.json "
                        "kl_truth_center vs nmse_mda_target."
                        if kl_every_checkpoint else
                        "mean_Q[ KL( Normal(noiseless_observation(x_q), "
                        "(noise*noiseless_observation(x_q))**2) truncated at 0 || BMA ) ], "
                        "1-D Gauss-Legendre quadrature; defined ONLY for the "
                        "FINAL checkpoint row (the official set opens only after finish()) - "
                        "every earlier row is blank in points.csv. NOT the same target as "
                        "nmse_mda: see config.json kl_truth_center vs nmse_mda_target."
                    ),
                },
                "rounds": point_rows,
                **({"prediction_excluded_models": prediction_excluded_models}
                   if prediction_excluded_models else {}),
                "reference_used_for_fit_or_selection": False,
            },
        )
        recheck_manifest(label_receipt)
        write_points(out / "points.csv", config, task, point_rows)
        timings["evaluation"] += time.perf_counter() - tic
        own, child = (
            resource.getrusage(resource.RUSAGE_SELF),
            resource.getrusage(resource.RUSAGE_CHILDREN),
        )
        write_json(
            out / "summary.json",
            {
                **({"label_receipt": label_receipt} if label_explicit else {}),
                "status": ("stopped_on_request" if stop_record is not None
                           else "completed_fixed_pool_first_path"),
                **({"stop": stop_record} if stop_record is not None else {}),
                "task_id": task["id"],
                # The executed cell (once the literal
                # "000" here too, even for a free run).
                "cell": args.cell,
                "profile": args.profile,
                "design_axis": design_axis,
                "observations": session.calls,
                "rounds": args.rounds,
                # catalog mode: identical to the original fixed formula (the
                # catalogue never grows). free mode: the live per-fit count,
                # which includes every admitted candidate's own refits. Acquisition
                # guard: a skipped refit (acquisition_failed) makes catalog mode
                # report the live count too, i.e. only the fits actually run.
                "fit_count": (fit_count if free or timed_out_ids or refit_skipped_rounds
                              or stop_record is not None
                              else 4 * (args.rounds + 1)),
                "active_pool": len(catalogue) if free or timed_out_ids else 4,
                **({"candidate_failures": candidate_failures} if candidate_failures else {}),
                **({"prediction_excluded_models": prediction_excluded_models}
                   if prediction_excluded_models else {}),
                "proposals": len(proposed_rounds) if free else 0,
                "evaluation_points_after_finish": args.q,
                "official_evaluation_points": OFFICIAL_EVALUATION_COUNT,
                "points_rows": len(point_rows),
                "points_columns": len(CONTRACT_POINTS_COLUMNS),
                "primary_metric": PRIMARY_METRIC,
                "primary_metric_final": point_rows[-1][PRIMARY_METRIC],
                # `kl` is computed ONCE, for the final active
                # pool, on the official Q(1000) - see `official_kl_summary`
                # and `write_official_kl_particle_summary`. The
                # disclosure pair repeats config.json's, so a reader of
                # summary.json alone still sees that `kl`'s truth center is
                # NOT `nmse_mda`'s target.
                "kl_truth_center": OFFICIAL_KL_TRUTH_CENTER,
                "nmse_mda_target": OFFICIAL_PRIMARY_TARGET,
                "kl": kl_summary["kl"],
                "kl_quadrature_error": kl_summary["kl_quadrature_error"],
                "kl_truncated_mass": kl_summary["kl_truncated_mass"],
                "kl_n_nodes": kl_summary["kl_n_nodes"],
                "kl_per_q_summary": kl_summary["kl_per_q_summary"],
                **kl_undefined,
                "wall_seconds": time.perf_counter() - started,
                "timing_seconds": timings,
                "user_cpu_seconds": own.ru_utime + child.ru_utime,
                "system_cpu_seconds": own.ru_stime + child.ru_stime,
                "paid_api_calls": 0 if proposer is None else proposer.attempts,
                "registered_run": False,
                # Present only with archive review rounds.
                **({"revival_events": revival_events} if review_rounds else {}),
                **(
                    {
                        "proposal_accounting": {
                            "proposal_rounds_executed": len(proposed_rounds),
                            "llm_calls": sum(p["transport"].get("calls", 0) for p in proposal_log),
                            "llm_repairs": sum(
                                p["transport"].get("repairs", 0) for p in proposal_log
                            ),
                            "admitted_candidates": sum(
                                len(p["admitted_ids"]) for p in proposal_log
                            ),
                            "initial_pool_size": (initial_pool_count if initial_proposals or frozen_pool is not None
                                                  else 4),
                            # Burst refill: the round-0 burst waves executed (1..3), with a burst.
                            **({"burst_wave_count": burst_wave_count} if initial_proposals else {}),
                            "initial_pool_remains_in_active_pool": initial_catalogue_ids <= {
                                c.model_id for c in catalogue
                            },
                            # Pool retirement.
                            "active_maximum": ACTIVE_MAXIMUM,
                            "retired_candidates": len(retired_log),
                            "retirements": retired_log,
                            # Failure-class accounting.
                            "failure_class_counts": failure_counts,
                            # Refits spent on revival, with archive review rounds.
                            **({"revival_refits": sum(len(event["valid_ids"])
                                                      for event in revival_events)}
                               if review_rounds else {}),
                        },
                        "billing_status": (
                            "no_calls" if proposer is None else "usage_in_llm_transport_records"
                        ),
                    }
                    if free
                    else {}
                ),
            },
        )
        if stop_record is not None:
            print(
                f"STOP requested: stopped before round {stop_record['detected_before_round']}; "
                "official scores written",
                file=sys.stderr,
            )
    except PaidCallBudgetExhausted:
        if pool_generator is not None:
            # Frozen pool track: a generation run the paid-call guard stopped leaves no generator
            # selected either, and no pool file (the burst record is written after its last wave).
            from .count_proposer import select_pool_generator

            select_pool_generator(None)
        # Paid-call guard: the guard refused the call that would pass max_paid_calls. Every record
        # written so far stays; the summary says why and where, and main() exits nonzero.
        write_json(out / "summary.json", {
            **({"label_receipt": label_receipt} if label_explicit else {}),
            **budget_exhausted_summary(proposer, current_round),
            "task_id": task["id"],
            "cell": args.cell,
            "profile": args.profile,
            "rounds": args.rounds,
            "observations": session.calls,
            "timing_seconds": timings,
            "wall_seconds": time.perf_counter() - started,
        })
        raise
    except BaseException as exc:
        if pool_generator is not None:
            # Frozen pool track: a failed generation run leaves no generator selected either.
            from .count_proposer import select_pool_generator

            select_pool_generator(None)
        write_json(
            out / "failure.json",
            {
                **({"frozen_pool_failed_model_id": exc.model_id}
                   if isinstance(exc, FrozenPoolMemberFitFailure) else {}),
                "error": mask_secrets(f"{type(exc).__name__}: {exc}"),
                "traceback": mask_secrets(traceback.format_exc()),
                "observation_calls": session.calls,
                "timing_seconds": timings,
            },
        )
        raise


def main():
    """Command-line entry point: parse and check the arguments, then call `run`."""
    parser = argparse.ArgumentParser(description=__doc__)
    from .design_search import add_cli_arguments

    add_cli_arguments(parser)
    parser.add_argument("--manifest", required=True)
    # The task axis. Optional only for a one-row
    # manifest - `select_task` refuses to guess for a nine-row one.
    parser.add_argument("--task", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--stop-file", default=argparse.SUPPRESS,
                        help="Stop before the next round and score completed checkpoints")
    parser.add_argument("--d0-policy", choices=D0_POLICIES, default="random4")
    parser.add_argument("--d0-fixed-inputs", type=json.loads, default=None,
                        help="JSON overrides of our declared fixed3 background conditions (MDA values undisclosed)")
    # `--profile` is the single source of
    # `(particles, q)`; the two scalars stay as explicit overrides (default
    # None = "take the profile's value"), so `--profile paper` reproduces the
    # original paper defaults (500/256) and `--profile dev` the earlier dev-run values
    # (64/32) without either being typed twice. By rule, the DEFAULT
    # profile is dev, so a paid run has to say `--profile paper` out loud.
    parser.add_argument("--profile", choices=tuple(PROFILE_GRID), default=DEFAULT_PROFILE)
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--fit-timeout", dest="fit_timeout_seconds", type=float,
                        default=argparse.SUPPRESS)
    parser.add_argument("--fit-guards", choices=("off", "v1"), default=argparse.SUPPRESS)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--noise", type=float, required=True)
    # The method axis; the recorded cell is the
    # executed cell. All eight cells are open and cell[0] is the
    # single authority on the structure mode (`validate_cell`/`structure_mode`).
    parser.add_argument("--cell", choices=CELL_CHOICES, default="000")
    # Stage 2: the acquisition axis. `choices` is the
    # SHARED registered list so a name is never invented here; only
    # `CHEM_DESIGN_POLICIES` is executable on this path, and `run` says which
    # and why. The default is the shared default, and it is refused outright on
    # an A=0 cell only if it is NOT the default (design_policies.check_policy),
    # so `--cell 000` keeps working with no new flag typed.
    # Unified protocol: plus the EIG part the shared registry does not hold (parameter_eig) and the
    # protocol's name for the registered structure part (model_eig = structure_eig).
    parser.add_argument("--design-policy", default=argparse.SUPPRESS, choices=(
        *POLICIES, *(name for name in (*CHEM_EIG_PART_POLICIES, *CHEM_EIG_PART_ALIASES)
                     if name not in POLICIES)))
    parser.add_argument("--family", choices=("lognormal", "gamma"), default="lognormal")
    # Multi-candidate proposals: candidates asked for per round, 1..3 - the same bound
    # and the same wording as core_count's own rule. SUPPRESS, so an omitted
    # option adds no config key and k = 1 keeps every byte of today's request.
    parser.add_argument("--proposal-limit", type=int, default=argparse.SUPPRESS)
    # All three SUPPRESS, so an omitted option leaves the namespace,
    # every request and every record as they were.
    parser.add_argument("--composition-paragraph", choices=("v1", "v2"), default=argparse.SUPPRESS)
    parser.add_argument("--archive-review-rounds", default=argparse.SUPPRESS)
    parser.add_argument("--mock-schedule", choices=MOCK_SCHEDULES, default=argparse.SUPPRESS)
    parser.add_argument("--repair-attempts", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--initial-proposals", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--initial-catalogue", choices=("on", "off"), default=argparse.SUPPRESS)
    # Frozen pool track: a registered cell's frozen initial pool, and the separate generation run
    # that makes one. SUPPRESS, so a command line that states neither keeps every config byte.
    parser.add_argument("--initial-pool-file", default=argparse.SUPPRESS)
    parser.add_argument("--mask-input-names", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--pool-generator", choices=("pool_generator_v1",), default=argparse.SUPPRESS)
    parser.add_argument("--generator-code-commit", type=parse_commit_sha, default=argparse.SUPPRESS)
    # Grammar version: the expression grammar this run's candidates are judged
    # against. SUPPRESS for the same reason `--prompt-version` uses it - an
    # omitted option adds no config.json key at all, so a command line that
    # never states it records exactly what it always did, and the finished v2
    # waves keep their grammar and their bytes. Unstated means
    # DEFAULT_GRAMMAR_VERSION, read through `getattr` in `run`.
    parser.add_argument(
        "--grammar-version", choices=GRAMMAR_VERSIONS, default=argparse.SUPPRESS
    )
    # An OPTIONAL override of the mode cell[0] already selects, not
    # an independent axis. Default None = "take cell[0]"; stated, it must agree
    # (`validate_cell`), which is what keeps every earlier command line, the
    # byte-identity anchors and the smoke driver valid with no edit.
    parser.add_argument("--structure-proposals", choices=("catalog", "free"), default=None)
    parser.add_argument("--proposer", choices=("mock", "anthropic", "openai", "openai_compat"), default="mock")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=argparse.SUPPRESS)
    parser.add_argument("--llm-key-name", default=argparse.SUPPRESS)
    parser.add_argument("--llm-provider-order", type=normalize_provider_order, default=argparse.SUPPRESS)
    # Decoding options: both stay absent from the namespace unless typed, so an
    # omitted flag leaves the request body, the config and the points columns
    # byte-identical. `--llm-reasoning` is openai_compat only (compat_config
    # refuses it elsewhere); `--llm-max-tokens` is a decoding setting for
    # every provider. An invalid value ends the command line with status 2.
    parser.add_argument("--llm-reasoning", type=parse_reasoning_spec, default=argparse.SUPPRESS)
    # Validation messages: OpenAI JSON mode (stock provider only) and the validation-message
    # wording; both SUPPRESS, so an omitted option leaves the request, config and receipt
    # byte-identical.
    parser.add_argument("--llm-response-format", choices=RESPONSE_FORMATS, default=argparse.SUPPRESS)
    parser.add_argument("--validation-messages", choices=VALIDATION_MESSAGE_VERSIONS,
                        default=argparse.SUPPRESS)
    parser.add_argument("--llm-max-tokens", type=parse_max_tokens, default=argparse.SUPPRESS)
    parser.add_argument("--allow-paid-api", action="store_true")
    # Paid-call guard: the per-cell paid-call guard. SUPPRESS, so an omitted option leaves every
    # record byte-identical; stated, the proposer refuses the call that would pass it.
    parser.add_argument("--max-paid-calls", type=parse_max_paid_calls, default=argparse.SUPPRESS)
    # Role-prior template ablation; `default` is the frozen Table 3.
    parser.add_argument(
        "--prior-template", choices=PRIOR_TEMPLATE_NAMES, default=DEFAULT_PRIOR_TEMPLATE
    )
    # A SCORING-OUTPUT switch, not a science axis. `kl` is the same
    # function, on the same retained pools, with the same quadrature
    # constants; stating it only widens the rows it is evaluated on, and the
    # FINAL checkpoint's value (points.csv's last row, summary.json's `kl`
    # family, the double-precision recompute artifact) is the identical call either way.
    # `SUPPRESS` keeps an omitted flag out of `vars(args)`, so an unstated
    # option leaves config.json byte-identical too (the decoding-option rule).
    parser.add_argument("--kl-every-checkpoint", action="store_true",
                        default=argparse.SUPPRESS)
    # Prompt selection: the numbered system prompt this run sends. SUPPRESS for the same
    # reason `--kl-every-checkpoint` above uses it: an unstated option leaves
    # config.json byte-identical.
    parser.add_argument("--prompt-version", default=argparse.SUPPRESS)
    parser.add_argument("--prompt-variant", default=argparse.SUPPRESS)
    # Rule v2: omitted labels stay development; production needs registration.
    parser.add_argument("--run-label", choices=RUN_LABEL_CHOICES, default=None)
    parser.add_argument("--manifest-ref", default=argparse.SUPPRESS,
                        help="JSON object containing manifest path and whole-file sha256")
    args = parser.parse_args()
    # Multi-candidate proposals: core_count's own rule, stated here in the same words.
    if not 1 <= getattr(args, "proposal_limit", 1) <= 3:
        parser.error("--proposal-limit must be 1..3")
    if args.proposer == "openai_compat":
        compat_config(args)
    if args.llm_model is None:
        args.llm_model = "gpt-4o" if args.proposer == "openai" else "claude-opus-4-7"
    # The CLI bound is the largest horizon ANY profile could supply a distinct
    # design for, not a literal (wave b3, design decision 2026-09-12:
    # the paid ChemBench run moves from 40 rounds to 60). The real, profile-
    # specific limits stay exactly where they were and still refuse first: an
    # A=1 cell needs one distinct pool design per round and `run` raises when
    # `--rounds` exceeds this profile's `design_pool` (dev 32, paper 256), and
    # a free-mode mock run is still capped at MOCK_SCHEDULE_ROUNDS. Widening a
    # parse-time literal to a derived bound therefore admits 60 without
    # loosening any of the checks that decide whether a run can actually
    # complete.
    max_rounds = max(profile["design_pool"] for profile in PROFILE_GRID.values())
    if not 1 <= args.rounds <= max_rounds:
        parser.error(f"rounds must be 1..{max_rounds}")
    if args.q is not None and args.q < 1:
        parser.error("q must be positive")
    if args.particles is not None and args.particles < 2:
        parser.error("particles must be at least 2")
    try:
        from .run_label_policy import validate_repair_attempts
        validate_repair_attempts(getattr(args, "repair_attempts", 1), bundle="chembench")
    except ValueError as exc:
        parser.error(str(exc))
    try:
        run(args)
    except PaidCallBudgetExhausted as exc:
        print(f"STOP {exc}; summary.json records stop_reason budget_exhausted", file=sys.stderr)
        raise SystemExit(PAID_CALL_BUDGET_EXIT_CODE) from None


if __name__ == "__main__":
    main()
