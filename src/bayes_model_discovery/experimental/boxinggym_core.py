"""BoxingGym scalar tasks through the common API.

One adapter, one grammar, one belief packet. Every task below is declared as a
`TaskSpec`: a machine-generated `interface` block (named inputs + the engine's
standardization as derived inputs, the outcome type, and the link the paper's task
table fixes for that type), the BoxingGym environment and goal objects
the observations and the held-out evaluation come from, and the exact mapping
from our posterior predictive to the string format the environment's own
evaluator parses. Nothing about a task reaches the proposer except through
`interface` - the SYSTEM prompt is byte-identical to CORE's.

The environment repository is read-only: this module imports
`boxing_gym.envs.*` and calls the environment's own `run_experiment` /
`step` / `get_goal_eval_question` / `evaluate_predictions`. It never edits,
patches or re-implements them.

Scoring is reported twice on the SAME held-out points (asserted, not assumed):
the environment's own `evaluate_predictions` metric, and our predictive scores
(nll, coverage90, width90) from the Bayesian model averaging (BMA) predictive
PMF/density.

Evaluation-question defect: upstream
`DirectDeath.get_goal_eval_question` overwrites the question text and loses the
time value. That defect is confined to the LLM TEXT path. This adapter never
reads the question string - like the PPL path it takes the numeric
`goal.eval_points` tuples - so it is unaffected; `eval_question_policy` in the
config and summary records it as the restored question.

Environment seed injection: peregrines.py and
dugongs.py draw their world parameters (`reset`, via `pymc.draw`) and every
observation (`step`, via `pymc.sample_prior_predictive`) with no `random_seed`,
so PyMC reseeds those calls from OS entropy on every call regardless of
`np.random.seed()` - two runs with the same `--seed` got a different world.
`env_seed_injection` wraps both functions for the duration of one `run()` call
so a `random_seed=None` call gets a deterministic, per-call seed derived from
the world seed instead; an explicit seed is always left alone. death_process.py
draws `theta` and every observation from scipy, which already obeys
`np.random.seed()`, so this is a no-op for the death task. See
`ENV_SEED_INJECTION_RULE` for the exact rule and `config.json`'s
`env_seed_injection`/`env_seed_injection_rule` and `summary.json`'s
`env_seed_log` for the per-run record.

LLM proposer wiring: `--proposer {mock,
anthropic}` (default mock, byte-identical to the earlier mock-only adapter), `--llm-model` and
`--allow-paid-api` mirror CORE's `core_count.py` CLI exactly - `run()` raises
before creating `--output` when `--proposer anthropic` is given without
`--allow-paid-api`, exactly as `core_count.run` does. The anthropic branch
wires `count_proposer.AnthropicCountProposer.propose_one` (imported locally
inside `run()`, exactly where CORE imports it, so a test can monkeypatch
`count_proposer.AnthropicCountProposer` and inject a fake client the same way
CORE's own tests do) with THIS task's belief packet and its own retargeted
initial structures (`initial_structures=`, never CORE's `u`/`z` ones), and
records the transport/accounting the same way CORE's P1 branch does (same
keys: `prompt_version`, `prompt_sha256`, `packet_bytes`, `packet_sha256`,
`raw_text` inside each response, `calls`, `repairs`, `rejected_candidates`,
`note`, plus `accounting["llm_calls"]`/`["llm_repairs"]` and
`summary.json["paid_api_calls"]` = `proposer.attempts`). The structure-fixed,
family-fixed cell (`cell[:2] == "00"`) keeps its original mock-only "no_candidate" shape unchanged and never
reaches the proposer regardless of `--proposer`, matching how CORE's own
`proposal_due` gate never calls its proposer for that cell either. S0 with an
open family menu (`cell[:2] == "01"`) gets a `core_count.candidate_catalog`-
equivalent exhaustion check (same `catalog_exhausted` status/accounting/
`catalog_exhausted_rounds` keys as CORE), built from this task's own
structures and allowed families instead of CORE's.

Allowed-family parametrization: in earlier versions the
anthropic path could only ever admit a candidate for "death", because
`count_proposer.judge_candidate` fixed `Hypothesis` to CORE's own three count
families (binomial/beta_binomial/zi_binomial) regardless of task - every
peregrines (poisson/negative_binomial) or dugongs (lognormal/gamma) candidate
was safely rejected at that check, `status: "proposal_failed"`, never a crash.
Now that check is parametrized by the packet's `constraints["allowed_families"]`
(`interface_families(interface["outcome_type"], ...)`'s `allowed`, already
computed above for the mock path's own family cycling and S0 catalog) -
`propose_one(..., allowed_families=allowed)`. `judge_candidate`'s own S0/F0
rules, role priors, grammar and the duplicate/repair-once contract are
unchanged; only the family-membership check gained an argument, defaulting to
CORE's fixed three so CORE is byte-identical. Because `count_proposer.py`
always builds a `count_models.Hypothesis` (never a task's own class), an
admitted peregrines/dugongs candidate is reconstructed as `ScalarHypothesis`
right after `propose_one` returns - the same class `make` already gives every
mock/initial candidate on those tasks - before it reaches `.as_dict()`,
`prior_probe`, or `scalar_fit`, all of which need `ScalarHypothesis.
noise_parameters`, not `Hypothesis`'s CORE-only one. Which families are actually
fittable (`positive_inference.SCALAR_FAMILIES`/`build_model`) can differ from those
merely declared by the paper's task table.

Categorical inputs and three more tasks:

  * CATEGORICAL INPUTS. A `TaskSpec` may declare an input as categorical with a
    level count (`categorical_inputs`); the generated `interface` then carries
    `kind: "categorical"` and the level list, and the grammar accepts the ONE
    added form `a[<that input>]` - a length-n_levels offset coefficient whose
    entries share the single fixed `signed` = Normal(0,1) prior (the paper's prior
    table, row 1) and which occupies ONE of the six coefficient slots. See
    `count_models.INDEXED_COEFFICIENT_RULE`. The rule reaches the model through
    the packet's `constraints`/`interface` only: the SYSTEM prompt file is
    byte-frozen (sha256 f095edf1...) and is NOT touched.
  * `irt` - the 6x6 student/question grid, both inputs categorical, binary
    outcome. `design_exclusion="observed"` applies: `IRT.reset` fixes the whole response matrix
    with ONE prior-predictive draw (irt.py:305, :330-:340) and `step` returns
    that stored cell verbatim (irt.py:403-:405), so re-observing a
    (student, question) pair carries no new information. The environment also
    REFUSES the ten evaluation cells (irt.py:407-:416, refusal at :413) while
    its own `validate_input` (irt.py:378-:391) does not check them, so the
    adapter removes `env.restricted_pair` from the candidate pool explicitly -
    the same set the environment's own design sampler excludes (irt.py:399).
  * `moral_machines` - a binary choice with seven numeric attribute contrasts.
  * `emotion` - seven numeric inputs and EIGHT outputs, run as eight
    component-wise real runners over one shared design/observation stream
    (the paper's task table, last row: "component-wise scores are composite and
    labelled so").
  * `lotka_volterra` - registered `specification_only`; see its `note`.

Both `moral_machines` and `emotion` call an OpenAI chat completion inside
`step` purely to NARRATE an outcome that has already been drawn
(moral_machines.py:487-:491 draws `choice` from `pm.Bernoulli` and only then
calls the LLM at :529; emotion.py:390-:397 draws the eight values and only then
calls the LLM at :442). The adapter reads the environment's own recorded
observation (`env.observed_data[-1]`), never the narration string, so the
narration is provably irrelevant to every number in a run - and both
environments would otherwise need a paid API key just to be CONSTRUCTED
(`openai.OpenAI()` at moral_machines.py:294 / emotion.py:270).
`openai_narration_stub` therefore replaces `openai.OpenAI` with a local stub
for the duration of one `run()` call, exactly as `env_seed_injection` replaces
`pymc.draw`: process-local, restored in `finally`, upstream source untouched,
and recorded in `config.json`. Zero paid calls is then structural, not a
promise.

death's KL score: `our_scores` additionally records `kl` (`KL(truth || BMA)`) and
`expected_log_score` for death only, on the SAME fixed evaluation set Q every
other death score already uses (`evaluation_x`). Both are computed by
`core_count.score_prediction` (imported unchanged - see that function for the
exact formula: `xlogy(truth, truth) -
truth * logpmf`, summed over y=0..N and averaged over Q). `truth` is death's
own generating family, `Binomial(N, 1 - exp(-theta*xi))`, with the run's one
true `theta` - `death_truth_pmf` builds it from the SAME environment instance
and the SAME source lines `death_noise_free_truth` already reads
(death_process.py:280 `self.N = N`, :286 the once-only `theta` draw,
:319-:323 `step`'s `eta = 1 - exp(-theta*xi)` / `binom.rvs(N, eta)`), carried
to the full distribution instead of collapsed to its mean; `nmse_mda` itself
is untouched.

Every task's KL: the paragraph above is
unchanged - death's `kl`/`expected_log_score` are still that exact call, on
that exact array, and are bit-identical to what the death-only recording produced
before this extension. What this adds
is the SAME quantity for the other eight tasks - unconditionally except the
one whose measured cost exceeds the approved 10%-of-cell-wall rule, which records it
under `--kl-all-tasks` (`KL_OPTIONAL_TASKS`, and the comment there) -
each against its own generating observation law: `boxinggym_truth_law` builds
those laws once per run from the environment instance's public attributes (the
benchmark tree stays read-only, exactly as `death_truth_pmf` already treats
it), and `our_scores` records `kl` plus the per-task diagnostics `kl_method`,
`kl_truncated_mass`, `kl_quadrature_error` and, where the support is discrete,
`expected_log_score`. Per-task definitions live in `boxinggym_truth_law`'s own
module docstring. points.csv keeps its 25-column contract exactly: only the
long-blank `kl` column is now filled (still blank, never 0, when the world's
own parameters were not captured - irt), and every diagnostic stays out of it,
in scores.jsonl and summary.json's `our_scores` only.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import itertools
import json
import resource
import sys
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import pymc

#: Burst calls in flight per cell when the option is not stated, and the
#: exponential backoff a rate-limited (429) burst call is retried with.
DEFAULT_BURST_CONCURRENCY = 6
PWC2_ADDED_KEYS = ("active_count_by_round", "proposed_by_round", "admitted_by_round",
                   "retired_by_round", "initial_pool_llm", "initial_pool_fit", "rounds_llm", "rounds_fit")
PWC2_SCORE_ADDED_KEYS = ("active_count",)
PWC2_COST_KEYS = ("initial_pool_llm", "initial_pool_fit", "rounds_llm", "rounds_fit")
BURST_RETRIES = 4
BURST_BACKOFF_SECONDS = 1.0
from scipy.special import expit, logsumexp, ndtr
from scipy.stats import binom

from ..eig.joint import choose_design
from . import boxinggym_design_search as bg_design_search
from .belief_packet import (
    EXHAUSTIVE_COUNT_OUTCOMES,
    PoolMember,
    build_belief_packet,
    cell_constraints,
    interface_families,
    interface_grid,
    packet_digest,
    serialize_packet,
    task_interface,
)
from .boxinggym_truth_law import (
    bernoulli_truth_mean_law,
    death_law,
    dugongs_law,
    emotion_law,
    kl_against_pool,
    location_law,
    peregrines_law,
    subset_law,
)
from .core_count import (
    ACTIVE_MAXIMUM,
    INITIAL_BURST_WAVES,
    P1_MOCK_EXPRESSIONS,
    PRIOR_PROBE_GRID_POINTS,
    PRIOR_PROBE_NONFINITE,
    PRIOR_PROBE_SAMPLES,
    PRIOR_PROBE_SEED,
    PWC2_WAVE_COUNT_KEYS,
    RUN_LABEL_CHOICES,
    add_recorded_pool_count,
    append_json,
    archive_reevaluation_label,
    burst_refill_pool,
    burst_wave_record,
    failure_class_counts,
    mock_parameters,
    parse_round_list,
    points_label,
    proposal_accounting_indexes,
    score_prediction,
    seed_for,
    wave_cost_blocks,
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
from .count_inference import choose as count_choose
from .count_inference import fit as count_fit
from .count_inference import predict as count_predict
from .count_inference import predictive_summary as count_predictive_summary
from .count_models import (
    BINARY_TRIALS,
    CATEGORICAL_KIND,
    CORE_INITIAL_STRUCTURES,
    DEFAULT_GRAMMAR_VERSION,
    DEFAULT_PRIOR_TEMPLATE,
    DEFAULT_VALIDATION_MESSAGES,
    DRIFT_TOL,
    FAILURE_CLASSES,
    GRAMMAR_VERSIONS,
    INDEXED_COEFFICIENT_RULE,
    INITIAL_BURST_REFILL_POLICY,
    LLM_PRIORS_VARIANT_TOKEN,
    PRIOR_RULES,
    PRIOR_TEMPLATE_NAMES,
    PRIOR_VARIANT_TOKEN,
    VALIDATION_MESSAGE_VERSIONS,
    Hypothesis,
    active_particle_rows,
    canonical_model_id,
    coefficient_prior,
    failure_class,
    interface_categorical_levels,
    log1mexp_sites,
    named_input_values,
    normalization_event_counter,
    particle_logpmf,
    prior_template_table,
    set_grammar_version,
    set_llm_priors,
    set_prior_template,
    set_stable_log1mexp,
    set_validation_messages,
    stable_log1mexp_enabled,
    standardized_expression,
)
from .count_proposer import (
    ARCHIVE_REVIVE_LIMIT,
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
    mask_secrets_deep,
    masked_exception,
    normalize_provider_order,
    parse_max_paid_calls,
    parse_max_tokens,
    parse_reasoning_spec,
    prompt_identity,
    require_registered_decoding,
    select_run_prompt,
    set_composition_paragraph,
    set_prompt_variant,
    validate_proposal_limit,
)
from .design_policies import (
    SPACE_FILLING_DEFINITION,
    check_policy,
    point_disagreement,
    select_space_filling,
    variant_token,
)
from .model_identity import MODEL_ID_SCHEMA, NOTATION_NORMAL_FORM
from .positive_inference import (
    POINT_DISAGREEMENT_DEFINITION,
    ScalarHypothesis,
    grammar_means,
    joint_eig,
    log_predictive,
    mixture_moments,
    particle_count,
)
from .positive_inference import build_model as scalar_build_model
from .positive_inference import fit as scalar_fit
from .positive_inference import point_disagreement as scalar_point_disagreement
from .positive_inference import predictive_means as scalar_predictive_means
from .positive_inference import predictive_summary as scalar_predictive_summary
from .prior_probe_detail import (
    PriorProbeRejection,
    first_nonfinite_subexpression,
    nonfinite_probe_sentence,
    prior_probe_reason,
)
from .real_inference import REAL_NOISE_PRIORS, RealHypothesis, default_scale_expression
from .run_label_policy import bmd_label_receipt, recheck_manifest

# Priors of the family-owned dispersion parameters on the scalar path. The two
# Gamma(2,1) entries are the same fixed "shape" role prior count_models.
# PRIOR_RULES already declares; `sigma` is the paper's prior table's output scale
# HalfNormal(1). No value here is new science - they are the priors
# positive_inference.build_model has always used, written down for the probe.
SCALAR_NOISE_PRIORS = {
    "sigma": ("HalfNormal", 1.0, None),
    "shape": ("Gamma", 2.0, 1.0),
    "alpha": ("Gamma", 2.0, 1.0),
    # The heteroscedastic scale coefficients. These are not new
    # science either: real_inference.REAL_NOISE_PRIORS reads them straight out
    # of count_models.PRIOR_RULES, so `a0` IS Table 3's "intercept | N(0,2)" row
    # and `a1` IS its "standardised slope | N(0,1)" row.
    **REAL_NOISE_PRIORS,
}
# CORE's twenty P1 mock templates are written over `u` (raw, in [0,1]) and `z`
# (standardized, in [-1,1]). A task's templates are those same twenty with `u`
# and `z` rewritten to that task's unit-scaled and standardized derived inputs,
# so every template keeps the numeric range it was declared for on every task.
# The rule is mechanical and fixed here, before any run.
MOCK_TEMPLATE_SOURCE = "core_P1_templates_retargeted_to_the_task_input_names"
# With several declared inputs the same twenty templates rotate through them:
# template i is CORE's template i with `u` and `z` rewritten to the unit-scaled
# and standardized derived inputs of declared input (i mod d). For d == 1 this
# is exactly the single-input rule. Fixed here, before any run.
MULTI_INPUT_TEMPLATE_RULE = "declared_input_index_i_mod_d"
# The initial pool follows the same rotation, plus - for a task with more than
# one declared input - ONE additional structure that is additive in every
# standardized input, so the pool always contains a model that can see every
# input the environment offers. Fixed here, before any run.
MULTI_INPUT_STRUCTURE_RULE = "core_four_rotated_plus_one_additive_in_every_standardized_input"
D0_RULE = "four_equally_spaced_interior_designs_of_the_declared_input_range"
D0_RULE_MULTI_INPUT = "the_design_pool_rows_at_fractions_0.1_0.35_0.65_0.9_of_the_pool_index"
# With several declared inputs the design set is the product grid of
# ceil(count ** (1/d)) equally spaced levels per declared range, integer inputs
# rounded, rows the environment's own `validate_input` rejects dropped, then
# thinned to `count` evenly spaced surviving rows. A task whose legal designs
# are FIXED BY THE ENVIRONMENT (a stored pool of records) uses that pool
# instead, thinned the same way. For d == 1 the single-input uniform grid is unchanged.
DESIGN_GRID_RULE = "product_grid_per_declared_range_filtered_by_the_environment_validator"
# ---- declared rules, all fixed before any run ------------------------------
# The additive-in-every-input structure joins the initial pool only when it fits
# the SIX-coefficient cap `count_models.validate_expression` enforces: it needs
# 1 + d coefficients for d declared inputs, so d <= 5. Every other
# multi-input task has d <= 3 and is therefore unaffected; `moral_machines` and
# `emotion` (d = 7) take the four rotated CORE structures alone. The cap itself
# is NOT changed - the paper's raised limit is declared for TABLES only
# (the paper's text, "The CORE complexity limit c_f <= 6 is CORE-specific"), and
# BoxingGym is not a table.
ADDITIVE_STRUCTURE_CAP_RULE = "additive_structure_in_the_pool_only_when_1_plus_d_fits_the_six_slot_cap"
# A task whose EVERY declared input is categorical has a finite design space:
# the full product grid of the declared level codes, minus the designs the
# environment itself refuses, never thinned.
DESIGN_GRID_RULE_CATEGORICAL = (
    "full_product_grid_of_the_declared_categorical_levels_minus_environment_refused_designs"
)
# A task whose legal designs are neither a stored record pool nor a product grid
# of ranges draws them from the ENVIRONMENT'S OWN random-input generator and
# keeps the ones the environment's own validator accepts.
DESIGN_GRID_RULE_SAMPLED = (
    "the_environments_own_sample_random_input_draws_filtered_by_its_own_validate_input"
)
# One design, several outputs (the paper's task table, last row). The components
# share the design stream, so a round's design must be informative about the
# whole vector: the per-design EIG of the round is the SUM of the per-component
# EIG tables, which is the joint EIG when the components are modelled
# independently. Declared here, before any run.
MULTI_COMPONENT_EIG_RULE = "sum_of_the_per_component_eig_tables"
# EIG-part arms (2026-10-05): the two chain-rule parts of the joint EIG as opt-in arms,
# I((M,theta);Y) = I(M;Y) + E_M I(theta;Y|M). Model EIG is MDA v4 Eq. 26. Selected only
# by --design-policy on an A=1 cell; the default joint arm and every record of it are
# unchanged. policy -> (objective column, acquisition `target`, points `variant` token)
EIG_PART_POLICIES = {
    "structure_eig": ("structure_eig", "model_structure", "design_structure_eig"),
    "parameter_eig": ("parameter_eig", "model_parameter_given_structure", "design_parameter_eig"),
}
EIG_PART_DEFINITION = {
    "structure_eig": "I(M;Y|x,D) = H[p(y|x,D)] - sum_m p(m|D) H[p(y|x,m,D)] (Model EIG, MDA v4 Eq. 26)",
    "parameter_eig": "sum_m p(m|D) I(theta;Y|x,m,D) = joint EIG - Model EIG (Parameter EIG)",
}
EIG_PART_ESTIMATOR = {
    "exhaustive": "exact chain rule on the outcome table (count_inference.choose rows: eig = structure_eig + parameter_eig)",
    "scalar": ("Monte Carlo on joint_eig's own draws (positive_inference.joint_eig decompose=True): "
               "structure = E[log p_k(y) - log p(y)], parameter = E[log p(y|theta,k) - log p_k(y)]"),
}
# Design search (2026-10-06): `--design-search ascent` per design policy -> the pool-table column it ascends.
DESIGN_SEARCH_OBJECTIVES = {
    "joint_eig": "eig", "structure_eig": "structure_eig", "parameter_eig": "parameter_eig", "epig": "epig",
}
# `--record-design-criteria` writes, for every candidate
# design of every round and every arm, the joint, Model and Parameter EIG (one table: the joint
# arm's own estimator on the round's acquisition seed) and the point disagreement, to its own
# file acquisition/tNN_criteria.json. A record only: the arm's decision and its acquisition
# record are untouched, and an unstated option leaves every file as before.
# Archive review rounds: on each listed round the k = 3 request also
# carries ARCHIVE_REVIEW_PARAGRAPH v1 and may name up to ARCHIVE_REVIVE_LIMIT archive models
# (retired_by_evidence) to refit from the original prior on all current public data; the round's
# retain ranks active + revived + new together. Recorded in revival/tNN.json and
# summary["revival_events"]; the engine never re-evaluates the archive on its own.
ARCHIVE_REVIEW_RULE = (
    "on a listed round the k = 3 request carries ARCHIVE_REVIEW_PARAGRAPH; the reply is one object "
    "{candidates, revive} (a plain array is accepted as candidates with no revive and recorded "
    "revive_missing); the revive list is validated (archive entries with outcome retired_by_evidence, "
    "at most 2; an invalid id is recorded with why in active / not_retired / unknown / over_limit and "
    "ignored, no repair turn); each valid id is refit from the original prior on "
    "all current public observations; retain(active + revived + new, active_maximum) once; a "
    "re-entered model is active, one retired again stays retired_by_evidence with revival_attempts; "
    "a numerically failed refit is recorded with scope revival and the model stays retired"
)
DESIGN_CRITERIA_FILE = "acquisition/tNN_criteria.json"
DESIGN_CRITERIA_DEFINITIONS = {
    "eig_definition": "joint EIG I((M,theta);Y|x,D) = structure_eig + parameter_eig",
    "structure_eig_definition": EIG_PART_DEFINITION["structure_eig"],
    "parameter_eig_definition": EIG_PART_DEFINITION["parameter_eig"],
    "eig_estimator": EIG_PART_ESTIMATOR,
    "point_disagreement_definition": POINT_DISAGREEMENT_DEFINITION,
    "point_disagreement_posterior_particles": "all_retained",
    "eig_seed_rule": "seed_for('boxinggym_acquisition', task, seed, round): the joint arm's acquisition seed",
    "role": "record only; the design decision is the arm's own acquisition/tNN.json",
}
# The identity link's frozen D0 calibration for a multi-component task: each
# component keeps its OWN frozen D0 mean as its center, and ONE pooled D0 sample
# sd over every component serves as the single task-specific scale - which is
# what Table 2's last row asks for ("Gaussian, task-specific scale"). A
# single-component task is unchanged (its own mean and its own sd).
MULTI_COMPONENT_CALIBRATION_RULE = (
    "per_component_frozen_D0_mean_center_with_one_pooled_D0_sample_sd_scale"
)
# What a multi-component score row is: every reported number is an average over
# the components, and the row says so (Table 2 last row: "component-wise scores
# are composite and labelled so").
COMPOSITE_SCORE_RULE = "mean_over_the_declared_output_components"
# The KL recording's own cost fallback, and the ONLY task it applies to. The KL
# recording is unconditional with one exception: "a task
# whose added time exceeds 10% of the cell's wall clock falls back to
# --kl-all-tasks (SUPPRESS) for that task alone, declared in the metadata" -
# and the cheaper alternative of shrinking that task's node or truncation
# constants is refused, since they are part of the definition.
# The rule is measured on a REFERENCE profile, so that a loaded host cannot
# decide what a run records: the
# development-profile mock cell --profile dev --rounds 2 --seed 11 --cell 111
# --proposer mock (|Q| = 32, 64 particles) on the development host, repeated,
# the WORST (largest) share deciding - share being the in-call KL seconds over
# that same flagged cell's wall clock.
# On that reference, eight of the nine tasks cost at or below 4.6% and record
# unconditionally. The one below the line is:
#   peregrines 42.9% - its truth is enumerated over ~420 counts per query
#     (Poisson rates near 250, covered to 1 - 1e-12), so the pool's log-PMF is
#     asked for |Q| x 420 outcomes per model; the paper profile (|Q| 256, 500
#     particles, a 12-model pool) extrapolates to ~400 s per checkpoint, more
#     than a whole cell's budget.
# emotion was the second, at 32.1% (re-measured in a separate verification run),
# because each of its eight components integrated the pool's density over nine
# cells with a 16- and a 32-node rule per panel. That quadrature was replaced
# with the exact mixture-CDF differences
# (`boxinggym_truth_law._cell_mass_model`), which asks the pool for 8 values
# per query instead of about 700, and the same reference cell was re-measured:
# 20.3% before, 0.97% and 0.94% on two repeats after - worst 0.97%, so emotion
# records `kl` unconditionally and only peregrines stays behind the flag.
# config.json records the flag and this list, so a reader can always tell which
# rule ran.
KL_OPTIONAL_TASKS = ("peregrines",)
# The narration bypass, stated in full for `config.json`.
OPENAI_NARRATION_STUB_RULE = (
    "boxing_gym.envs.{moral_machines,emotion} construct openai.OpenAI() in __init__ "
    "(moral_machines.py:294, emotion.py:270) and call chat.completions.create inside step "
    "(moral_machines.py:529, emotion.py:442) to TRANSLATE an outcome that has already been "
    "drawn: moral_machines draws `choice` from pm.Bernoulli at :487-:491 and emotion draws "
    "the eight values at :390-:397, both BEFORE the call, and both record only the drawn "
    "outcome in env.observed_data (moral_machines.py:563, emotion.py:476). This adapter reads "
    "env.observed_data, never the narration string, so the narration cannot affect any number "
    "in a run. Fix (adapter-process-local, active only for the duration of one run() call): "
    "the environment module's OWN `openai` name (boxing_gym.envs.<task>.openai) is replaced "
    "by a proxy whose OpenAI() returns a stub client whose chat.completions.create returns one "
    "fixed string and performs no I/O, so the two environments are constructible without an "
    "API key and the narration can make no paid call by construction. The global openai "
    "module is untouched, so a structure proposer built with --proposer openai gets the real "
    "SDK client (an earlier global replacement handed the proposer the "
    "stub too). The original name is restored in finally; the boxing-gym source on disk is "
    "never touched."
)


@dataclass(frozen=True)
class TaskSpec:
    """One BoxingGym task: its interface, its environment wiring, its scoring."""

    name: str
    module: str
    env_class: str
    goal_class: str
    input_name: str
    input_range: tuple[float, float]
    outcome_type: str
    support: tuple
    trials: int | None = None
    # Extra inputs beyond the first, for the multi-input tasks.
    extra_inputs: tuple = ()
    design_range: tuple[float, float] | None = None
    prediction_format: str = ""
    env_metric: str = "mse"
    env_metric_direction: str = "lower_is_better"
    runnable: bool = True
    note: str = ""
    env_kwargs: dict = field(default_factory=dict)
    # Declared inputs whose legal designs are integers (the environment's own
    # validator casts them), by name.
    integer_inputs: tuple = ()
    # How one design row reaches `env.run_experiment`. Declared per task; the
    # four encoders are the only ones this module has.
    design_encoder: str = "scalar_repr"
    # True when the environment FIXES the legal design set as a stored pool of
    # records (survival's pre-simulated patients); the pool then comes from the
    # environment, not from the declared ranges.
    environment_fixed_pool: bool = False
    # Positions in the goal's own eval tuple that hold the declared inputs, in
    # declared order; the outcome is always the tuple's LAST element.
    eval_input_index: tuple = (0,)
    # True when that single position holds the whole input vector at once.
    eval_input_is_vector: bool = False
    # Table 2 gates the zero-inflated Poisson on "zeros are natural". Declared
    # per task from the task's PUBLIC description, never from the truth.
    zero_inflation_admissible: bool = False
    # For an environment-fixed pool: the environment attribute holding the
    # records, and the positions inside one record that hold the declared
    # inputs, in declared order. NOTHING else of a record is ever read.
    fixed_pool_attribute: str = ""
    fixed_pool_record_index: tuple = ()
    # Adapter CANDIDATE-SET rule - not a design policy: "observed"
    # means a design already in the public record is removed from the
    # candidate pool before every round's draw, for EVERY arm (random and
    # active alike), and D0 is drawn without duplicates too. Declared per
    # task from a structural fact (re-observing the same design yields no
    # new information - e.g. survival's design IS the patient ID), never
    # from a preference about which arm should win. "none" (default) leaves
    # the pool untouched: every task but survival keeps re-drawing
    # from the full declared grid, exactly as before.
    design_exclusion: str = "none"
    # Declared inputs that are CATEGORICAL, as ((name, n_levels), ...). Their
    # design column holds the level CODE, their declared range is
    # [0, n_levels-1], and they can enter an expression only as a coefficient
    # index `a[name]` (count_models.INDEXED_COEFFICIENT_RULE). Empty for every
    # task without a categorical input, which is why those tasks are byte-identical.
    categorical_inputs: tuple = ()
    # Explicit overrides of the mechanical retargeting rules, needed by a task
    # with no numeric input at all (there is no `u`/`z` to retarget). Declared
    # in the registry, before any run; empty means "use the mechanical rule".
    initial_forms: tuple = ()
    mock_forms: tuple = ()
    # The goal's held-out set is FIXED at construction for some tasks (irt
    # builds exactly ten evaluation points in `DirectCorrectness.__init__`), so
    # asking for more raises inside the environment. `q` is capped at this.
    eval_point_limit: int | None = None
    # Where the legal designs come from: "" = the declared ranges (the default),
    # "records" = a stored record pool (survival), "sampled" = the
    # environment's own random-input generator, "categorical" = the full
    # product grid of the declared level codes.
    environment_pool_source: str = ""
    # How many generator draws a "sampled" pool considers before filtering.
    pool_sample_count: int = 0
    # The environment attribute holding designs the environment itself REFUSES
    # (irt's `restricted_pair`), as tuples in declared input order. Its
    # `validate_input` does not check them, so the pool must.
    environment_refused_attribute: str = ""
    # How the declared inputs are read out of a design and out of a goal eval
    # tuple: "positions" is the original index rule.
    feature_map: str = "positions"
    # Where one observation's outcome is: the `run_experiment` return value
    # (the default) or a field of the environment's own recorded observation,
    # for an environment whose `run_experiment` returns a narration string.
    observation_source: str = "return_value"
    observed_data_index: int = -1
    # Added to the environment's raw outcome to reach the modelled outcome
    # (moral_machines codes its binary choice as 1/2, the Bernoulli path needs
    # 0/1), and subtracted again when a prediction is submitted.
    outcome_offset: float = 0.0
    # The declared OUTPUT components. Empty means one scalar outcome
    # (the default). Several means one shared design/observation stream and one
    # independent runner per component, scored component-wise and composited
    # (the paper's task table, last row).
    components: tuple = ()
    # True for an environment that needs `openai.OpenAI()` merely to narrate an
    # already-drawn outcome; see OPENAI_NARRATION_STUB_RULE.
    openai_narration: bool = False
    # The environment module whose `openai` name the narration stub replaces
    # (only that module; the proposer keeps the real SDK). Required when
    # openai_narration is True.
    openai_narration_module: str = ""
    # Scoring-only hook: the generating E[Y|x] in modelled outcome units.
    truth_mean: Callable | None = None
    # Scoring-only hook: the generating OBSERVATION LAW at every
    # evaluation point - the whole distribution `truth_mean` collapses into its
    # mean - called as `spec.truth_law(spec, environment, evaluation_x)` and
    # consumed by `boxinggym_truth_law.kl_against_pool` for `kl`. `None` for a
    # task with no law (and the builder itself may return `None` when the
    # world's own parameters were not captured - irt).
    truth_law: Callable | None = None

    @property
    def inputs(self):
        """The declared inputs as interface dicts; a categorical input gets its level range and codes."""
        first = {"name": self.input_name, "range": list(self.input_range)}
        levels = dict(self.categorical_inputs)
        items = []
        for item in (first, *self.extra_inputs):
            if item["name"] in levels:
                count = int(levels[item["name"]])
                items.append(
                    {
                        "name": item["name"],
                        "range": [0.0, float(count - 1)],
                        "kind": CATEGORICAL_KIND,
                        "levels": list(range(count)),
                    }
                )
            else:
                items.append(item)
        return tuple(items)

    @property
    def input_names(self):
        """The declared input names, in declared order."""
        return tuple(item["name"] for item in self.inputs)

    @property
    def numeric_input_names(self):
        """The declared inputs that are NOT categorical, in declared order."""
        return tuple(item["name"] for item in self.inputs if item.get("kind") != CATEGORICAL_KIND)

    @property
    def derived_inputs(self):
        """Unit-scaled and standardized versions of every declared input.

        A CATEGORICAL input has none - its code is a label, and
        `count_models.interface_input_names` refuses a derived expression over
        one. A task with no categorical input is unchanged.
        """
        derived = {}
        for item in self.inputs:
            if item.get("kind") == CATEGORICAL_KIND:
                continue
            low, high = item["range"]
            derived[f"{item['name']}01"] = f"({item['name']}-{float(low)!r})/{float(high - low)!r}"
            derived[f"{item['name']}z"] = standardized_expression(item["name"], low, high)
        return derived

    def name_map(self, index=0):
        """CORE's `u`/`z` -> the unit-scaled / standardized `index`-th input."""
        names = self.numeric_input_names
        if not names:
            raise ValueError(f"{self.name}: a task with no numeric input needs declared forms")
        chosen = names[index % len(names)]
        return {"u": f"{chosen}01", "z": f"{chosen}z"}

    @property
    def additive_structure(self):
        """`b0 + sum_j b_j * <x_j>z`, the model additive in every standardized input."""
        terms = "".join(
            f"+b{index}*{name}z" for index, name in enumerate(self.numeric_input_names, start=1)
        )
        return f"b0{terms}"

    def interface(self):
        """The machine-generated interface block (inputs, derived inputs, outcome type, support, trials)."""
        return task_interface(
            self.outcome_type, self.inputs, self.derived_inputs, self.support, self.trials
        )

    def categorical_levels(self):
        """`((name, n_levels), ...)` as the grammar reads it off the interface."""
        return interface_categorical_levels(self.interface())

    def initial_structures(self):
        """The initial pool's expressions: the declared forms, or CORE's four retargeted (plus the additive one for 2-5 numeric inputs)."""
        if self.initial_forms:
            return tuple(self.initial_forms)
        forms = [
            retarget(form, self.name_map(index))
            for index, form in enumerate(CORE_INITIAL_STRUCTURES)
        ]
        # The additive structure needs 1 + d coefficients, so it joins
        # the pool only while that fits the six-slot cap (every multi-input task
        # but `moral_machines` and `emotion` has d <= 3). See ADDITIVE_STRUCTURE_CAP_RULE.
        if 1 < len(self.numeric_input_names) <= 5:
            forms.append(self.additive_structure)
        return tuple(forms)

    def mock_templates(self):
        """The twenty mock-proposer templates: the declared forms, or CORE's P1 templates retargeted."""
        if self.mock_forms:
            return tuple(self.mock_forms)
        return tuple(
            retarget(form, self.name_map(index))
            for index, form in enumerate(P1_MOCK_EXPRESSIONS)
        )

    def designs(self, count):
        """The single-input uniform grid; multi-input tasks use `design_pool`."""
        low, high = self.design_range or self.input_range
        return np.linspace(float(low), float(high), int(count))

    def d0(self):
        """Four interior designs, the same 0.1/0.35/0.65/0.9 fractions CORE uses."""
        low, high = self.design_range or self.input_range
        return [float(low + (high - low) * fraction) for fraction in D0_FRACTIONS]

    def eval_inputs(self, point):
        """The declared inputs of one goal eval tuple, in declared order."""
        if self.eval_input_is_vector:
            vector = np.atleast_1d(np.asarray(point[self.eval_input_index[0]], dtype=float))
            if len(vector) != len(self.input_names):
                raise ValueError(f"{self.name}: eval vector width does not match the interface")
            return [float(value) for value in vector]
        return [float(point[index]) for index in self.eval_input_index]


def retarget(expression, mapping):
    """Rewrite the input identifiers of a template, leaving coefficients alone."""
    import ast

    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in mapping:
            node.id = mapping[node.id]
    return ast.unparse(tree).replace(" ", "")


class InputNameMask:
    """Model-facing aliases; native identities stay in the RNG-only sidecar."""

    def __init__(self, spec, sidecar):
        self.input_name_map = {f"X{i}": name for i, name in enumerate(spec.input_names, 1)}
        forward = {name: alias for alias, name in self.input_name_map.items()}
        for name in spec.numeric_input_names:
            for suffix in ("01", "z"):
                forward[name + suffix] = forward[name] + suffix
        self.native_names = {alias: name for name, alias in forward.items()}
        self.spec = replace(
            spec, input_name=forward[spec.input_name],
            extra_inputs=tuple({**item, "name": forward[item["name"]]}
                               for item in spec.extra_inputs),
            integer_inputs=tuple(forward[name] for name in spec.integer_inputs),
            categorical_inputs=tuple((forward[name], levels)
                                     for name, levels in spec.categorical_inputs),
            initial_forms=tuple(retarget(form, forward) for form in spec.initial_forms),
            mock_forms=tuple(retarget(form, forward) for form in spec.mock_forms),
        )
        self.sidecar = Path(sidecar)
        self._ids = {}
        self._lock = threading.Lock()

    def rng_model_id(self, hypothesis):
        """The native-name model id that seeds RNG streams; each new mapping is appended to the sidecar."""
        key = hypothesis.model_id
        with self._lock:
            if key not in self._ids:
                family = hypothesis.family
                if family == "heteroscedastic_normal":
                    family += "|log_sigma=" + retarget(
                        hypothesis.scale_expression, self.native_names)
                native = canonical_model_id(
                    retarget(hypothesis.expression, self.native_names), family,
                    hypothesis.parameters, getattr(hypothesis, "coefficient_priors", ()),
                )
                with self.sidecar.open("a") as stream:
                    stream.write(json.dumps({"model_id": key, "rng_model_id": native}) + "\n")
                self._ids[key] = native
            return self._ids[key]


D0_FRACTIONS = (0.1, 0.35, 0.65, 0.9)


def encode_design(spec, row):
    """One design row as the exact string the environment's parser takes.

    Four declared encoders, one per environment input syntax; the per-task
    comment in `TASKS` cites the environment line that fixes which one applies.
    """
    values = [float(v) for v in np.atleast_1d(np.asarray(row, dtype=float))]
    if spec.design_encoder == "scalar_repr":
        return repr(values[0])
    if spec.design_encoder == "float_list":
        return "[" + ", ".join(repr(value) for value in values) + "]"
    if spec.design_encoder == "int_list":
        return "[" + ", ".join(str(round(value)) for value in values) + "]"
    if spec.design_encoder == "pool_index":
        return str(round(values[0]))
    if spec.design_encoder == "pool_handle":
        # The handle is the environment's OWN design string, produced
        # with the design and carried in `DesignPool.handles`; a model-facing
        # feature row can never be turned back into one (several scenarios map
        # to the same contrast vector), so this is a programming error.
        raise ValueError(f"task {spec.name} designs carry their own environment handle")
    raise ValueError(f"unknown design encoder {spec.design_encoder}")


# --------------------------------------------------------------------------
# FEATURE MAPS. One function per environment input syntax: it
# turns the environment's own design object - and the environment's own goal
# eval tuple, which is the SAME object plus its outcome - into the declared
# numeric input row. Nothing else of either is read.
# --------------------------------------------------------------------------
MORAL_CONTRAST_NAMES = (
    "human_count_diff",
    "gender_diff",
    "age_diff",
    "social_status_diff",
    "fitness_diff",
    "species_diff",
    "intervention",
)
# The eighth contrast the paper lists for this task, `group`, is a
# CONSTANT in the environment's own likelihood: `+ self.beta_group` enters
# y_hat with no covariate (moral_machines.py:487, and identically at :97 in
# the goal's own EIG model), so it is a preference for group 1 that the
# intercept `b0` absorbs exactly. It cannot be a declared input at all - an
# interface input needs a finite range with hi > lo - and declaring it would
# add nothing the intercept does not already carry.
MORAL_CONSTANT_CONTRAST = "group_is_a_constant_offset_absorbed_by_the_intercept"


def moral_contrast_row(environment, group1, group2, intervention):
    """The seven varying attribute contrasts of one Moral-Machine scenario.

    Read off the environment's OWN feature map, which is also what its own
    likelihood uses: the human count excludes the last two characters
    (`self.characters[:-2]` = dog, cat - moral_machines.py:471-:472), the five
    attribute contrasts are group-1-minus-group-2 sums of
    `_get_char_attr` (:478-:484, the same expressions the goal's EIG model
    builds at :109-:122), and the intervention is +1 for "swerve", -1 for
    "stay" (:475).
    """
    human = set(environment.characters[:-2])
    counts = [len([c for c in group if c in human]) for group in (group1, group2)]
    attributes = [[environment._get_char_attr(c) for c in group] for group in (group1, group2)]
    row = [float(counts[0] - counts[1])]
    for key in ("gender", "age", "social_status", "fitness", "species"):
        row.append(
            float(sum(a[key] for a in attributes[0]) - sum(a[key] for a in attributes[1]))
        )
    row.append(1.0 if str(intervention) == "swerve" else -1.0)
    return row


def moral_handle(group1, group2, intervention):
    """One scenario as the exact string `MoralMachine.validate_input` parses.

    The parser matches "Group 1: [...]" and then splits on ", "
    (moral_machines.py:538-:539), so the separator must be exactly ", ".
    """
    return (
        "Group 1: [" + ", ".join(str(c) for c in group1) + "]\n"
        "Group 2: [" + ", ".join(str(c) for c in group2) + "]\n"
        "Intervention: " + str(intervention)
    )


EMOTION_INPUT_NAMES = ("v1", "v2", "v3", "p1", "p2", "p3", "win")
# `run_experiment` and the goal disagree about the UNITS of the seventh input,
# and that is an upstream defect, not our choice: `validate_input` returns
# `win = prizes[win_idx]`, the PRIZE VALUE (emotion.py:464-:467), and `step`
# uses it as `normed_win` (:378); the goal's own evaluation generator
# instead calls `step(prizes, probs, win)` with `win = np.random.choice([0,1,2])`,
# the INDEX (:31-:37), and stores that index in `eval_points` (:37). So the
# held-out points carry win in {0,1,2} while every observation carries a prize
# value in [0,100]. Both paths are the environment's own; the LLM agent path
# has the identical shift. It is recorded per run, never silently corrected -
# correcting it would mean not using the environment's own evaluation set.
EMOTION_WIN_UNIT_DEFECT = (
    "emotion.py:464-:467 validate_input returns win = prizes[win_idx] (a prize value in "
    "[0,100]) and step uses it as normed_win (:378), while the goal's evaluation "
    "generator calls step with win = np.random.choice([0,1,2]) (an index) and stores that "
    "index in eval_points (:31-:37). Held-out inputs and observation inputs therefore carry "
    "the seventh input in different units; this adapter uses the environment's own numeric "
    "eval tuple unchanged and records the shift here rather than editing either path."
)


def emotion_input_row(prizes, probs, win):
    """The seven declared numeric inputs of one wheel spin, in declared order."""
    values = [float(v) for v in np.atleast_1d(np.asarray(prizes, dtype=float))]
    weights = [float(v) for v in np.atleast_1d(np.asarray(probs, dtype=float))]
    if len(values) != 3 or len(weights) != 3:
        raise ValueError("an emotion design has three prizes and three probabilities")
    return [*values, *weights, float(win)]


def emotion_handle(prizes, probs, win_index):
    """One spin as the exact string `EmotionFromOutcome.validate_input` parses.

    The parser reads "prizes: [...]", "probs: [...]" and a `win:` integer
    (emotion.py:449-:454) and then REFUSES the design unless `sum(probs) == 1`
    exactly in float arithmetic (:458-:459), which is why the pool is filtered
    by the environment's own validator rather than by a tolerance of ours.
    """
    return (
        "prizes: [" + ", ".join(repr(float(v)) for v in prizes) + "],\n"
        "probs: [" + ", ".join(repr(float(v)) for v in probs) + "],\n"
        "win: " + str(int(win_index))
    )


@dataclass(frozen=True)
class DesignPool:
    """The task's legal designs: model-facing rows plus environment handles.

    `designs` is what the runner scores and what `interface`-aware code reads:
    a one-dimensional vector for a single declared input, an
    [point, input] matrix otherwise. `rows` is always the matrix form, used only
    to look a chosen design back up. `handles[i]` is the exact string
    `env.run_experiment` takes for `rows[i]`.
    """

    rows: np.ndarray
    handles: tuple
    single_input: bool

    @property
    def designs(self):
        """The model-facing design set: a vector for a single input, a [point, input] matrix otherwise."""
        return self.rows[:, 0] if self.single_input else self.rows

    def index_of(self, design):
        """The pool row index of `design`; raises ValueError when it is not in the pool."""
        row = np.atleast_1d(np.asarray(design, dtype=float))
        matches = np.flatnonzero(np.all(self.rows == row[None, :], axis=1))
        if not len(matches):
            raise ValueError(f"chosen design {design!r} is not in the declared pool")
        return int(matches[0])

    def d0_indices(self):
        """The pool row indices at the D0 fractions 0.1/0.35/0.65/0.9 of the pool index."""
        return [round(f * (len(self.rows) - 1)) for f in D0_FRACTIONS]

    def available(self, observed_indices):
        """`designs`, minus the rows at `observed_indices`.

        The adapter candidate-SET rule for a `design_exclusion="observed"`
        task: a design already in the public record cannot be redrawn by any
        arm, because re-observing it is defined to carry no new information
        (survival's design IS the patient ID). Raises when nothing is left -
        a caller asked for more distinct designs than the pool holds, which
        is a sizing bug (see `design_pool_target`), not a condition to paper
        over with a silent repeat.
        """
        keep = [i for i in range(len(self.rows)) if i not in observed_indices]
        if not keep:
            raise ValueError("design pool exhausted: every design has already been observed")
        return self.designs[keep]


def _thin(rows, count):
    """`count` evenly spaced rows of `rows`, or all of them when there are fewer."""
    if len(rows) <= count:
        return rows
    index = np.unique(np.rint(np.linspace(0, len(rows) - 1, count)).astype(int))
    return rows[index]


def design_grid_rule(spec):
    """Which declared grid rule this task's design set follows."""
    if spec.environment_pool_source == "categorical":
        return DESIGN_GRID_RULE_CATEGORICAL
    if spec.environment_pool_source == "sampled":
        return DESIGN_GRID_RULE_SAMPLED
    return DESIGN_GRID_RULE


def build_categorical_pool(spec, environment):
    """The full product grid of the declared level codes, minus refused designs.

    `DESIGN_GRID_RULE_CATEGORICAL`. Never thinned: the space is finite and
    small (irt: 6 x 6 = 36), and every row the environment will actually answer
    has to stay available, because a `design_exclusion="observed"` task spends
    one distinct row per round.

    Two filters, in this order and both from the ENVIRONMENT: the designs its
    own validator rejects, and the designs it REFUSES to answer even though the
    validator accepts them - `environment_refused_attribute`. irt is the second
    case: `run_experiment` returns a refusal string for a `restricted_pair`
    design (irt.py:412-:413) while `validate_input` (irt.py:378-:391) never
    looks at that list, so validator-acceptance alone is not enough. The same
    list is what the environment's own random-input sampler skips (irt.py:399).
    """
    axes = [np.arange(len(item["levels"]), dtype=float) for item in spec.inputs]
    rows = np.stack([g.reshape(-1) for g in np.meshgrid(*axes, indexing="ij")], axis=1)
    refused = environment.refused_designs()
    legal = [
        row
        for row in rows
        if tuple(round(v) for v in row) not in refused
        and environment.accepts(encode_design(spec, row))
    ]
    if not legal:
        raise ValueError(f"no legal design in the declared levels of task {spec.name}")
    kept = np.asarray(legal, dtype=float)
    return DesignPool(kept, tuple(encode_design(spec, row) for row in kept), False)


def build_sampled_pool(spec, environment, count):
    """The environment's own random-input draws, kept if it accepts them.

    `DESIGN_GRID_RULE_SAMPLED`. `environment.sample_designs` calls the
    environment's OWN `sample_random_input` - the generator it declares for its
    own design space - `spec.pool_sample_count` times under the run's world
    seed, turns each draw into the declared numeric feature row and the exact
    handle its parser takes, drops the ones its own `validate_input` rejects
    (emotion refuses any probability triple whose float sum is not exactly 1),
    de-duplicates by handle, and the result is thinned to `count` evenly spaced
    rows exactly as every other pool is.
    """
    rows, handles = environment.sample_designs(spec.pool_sample_count)
    if len(rows) < count:
        raise ValueError(
            f"task {spec.name}: {len(rows)} accepted designs out of "
            f"{spec.pool_sample_count} draws, fewer than the {count} required"
        )
    keep = np.unique(np.rint(np.linspace(0, len(rows) - 1, count)).astype(int))
    matrix = np.asarray(rows, dtype=float)
    return DesignPool(matrix[keep], tuple(handles[i] for i in keep), False)


def build_design_pool(spec, environment, count):
    """The task's design set, by the rule declared in `DESIGN_GRID_RULE`."""
    names = spec.input_names
    if spec.environment_pool_source == "categorical":
        return build_categorical_pool(spec, environment)
    if spec.environment_pool_source == "sampled":
        return build_sampled_pool(spec, environment, count)
    if spec.environment_fixed_pool:
        rows, handles = environment.fixed_design_pool()
        keep = np.unique(np.rint(np.linspace(0, len(rows) - 1, count)).astype(int))
        return DesignPool(rows[keep], tuple(handles[i] for i in keep), False)
    if len(names) == 1:
        rows = spec.designs(count)[:, None]
        return DesignPool(rows, tuple(encode_design(spec, row) for row in rows), True)
    per_axis = int(np.ceil(float(count) ** (1.0 / len(names))))
    axes = []
    for item in spec.inputs:
        low, high = (float(v) for v in item["range"])
        column = np.linspace(low, high, per_axis)
        if item["name"] in spec.integer_inputs:
            column = np.unique(np.rint(column))
        axes.append(column)
    rows = np.stack([grid.reshape(-1) for grid in np.meshgrid(*axes, indexing="ij")], axis=1)
    legal = [row for row in rows if environment.accepts(encode_design(spec, row))]
    if not legal:
        raise ValueError(f"no legal design in the declared ranges of task {spec.name}")
    kept = _thin(np.asarray(legal, dtype=float), count)
    return DesignPool(kept, tuple(encode_design(spec, row) for row in kept), False)


BOXING_GOAL_TARGET_MEASURE = (
    "goal.get_goal_eval_question(include_prior=True); TaskEnvironment.eval_inputs"
)
BOXING_FINITE_TARGET_MEASURE = "uniform_resampling_of_finite_design_pool"
BOXING_FINITE_TARGET_NOTE = "finite_design_set; held_out_measure_differs"


def sample_epig_targets(spec, environment, pool, seed, count=EPIG_TARGET_POINTS):
    """Independent native goal draws on the same realised world, with fresh goal state.

    `count` is the number of target draws (default 32, the
    module constant; BoxingGym `--epig-target-points` states another). Only the
    count changes - the sampling measure and the seed stream are the same.

    Target covariates need not be legal acquisition designs: no acceptance
    filter, clipping or collision redraw changes the goal's sampling measure.
    Finite acquisition pools retain their declared resampling exception:
    survival's goal has continuous times; IRT's balanced ten held-out pairs are
    excluded from its legal categorical pool and cannot supply 32 fresh draws.
    """
    if spec.environment_fixed_pool or spec.environment_pool_source == "categorical":
        rng = np.random.default_rng(seed)
        return pool.designs[rng.choice(len(pool.rows), count, replace=True)].copy()
    import copy
    import random

    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        np.random.seed(seed)
        random.seed(seed)
        # Copy the realised environment, not its constructor: truth values stay
        # fixed while any native step history/caches and narration counters are
        # isolated. Fresh goals never inherit held-out points or cursors.
        # Nested explicit PyMC seeds leave the outer run counter untouched.
        with env_seed_injection(seed):
            goal = type(environment.goal)(copy.deepcopy(environment.env))
            for _ in range(count):
                goal.get_goal_eval_question(include_prior=True)
            rows = np.asarray([environment.eval_inputs(point)
                               for point in goal.eval_points[:count]], dtype=float)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    if rows.shape != (count, len(spec.input_names)) or not np.all(np.isfinite(rows)):
        raise ValueError(f"native goal sampler must supply {count} finite EPIG coordinate rows")
    return rows[:, 0] if len(spec.input_names) == 1 else rows


EPIG_TARGET_POINTS_MAX = 4096


def parse_epig_target_points(value):
    """A stated EPIG target count, an integer in 1..4096 (argparse `type`)."""
    text = str(value).strip()
    if not (text.isascii() and text.isdigit()) or not 1 <= int(text) <= EPIG_TARGET_POINTS_MAX:
        raise argparse.ArgumentTypeError(
            f"--epig-target-points must be an integer in 1..{EPIG_TARGET_POINTS_MAX}")
    return int(text)


def design_pool_target(spec, profile_count, rounds):
    """How many candidate rows to ask `build_design_pool` for.

    Unchanged (`profile_count`, the profile's grid size) unless the
    task excludes observed designs from its own candidate set: D0 draws
    `len(D0_FRACTIONS)` designs before round 1, and each of `rounds` rounds
    then needs a FRESH, never-before-observed design (every arm), so the
    pool must hold at least that many distinct rows - otherwise the run
    cannot finish without breaking the very rule it was asked to keep. A
    `design_exclusion="none"` task (every task but survival, today) takes
    this branch never, so its pool is byte-for-byte what it was before this rule.
    """
    if spec.design_exclusion != "observed":
        return profile_count
    return max(profile_count, len(D0_FRACTIONS) + rounds)


# --------------------------------------------------------------------------
# The task registry. Outcome types were read off the environment code, not
# guessed - see the per-task comment, which cites the line that fixes it.
# --------------------------------------------------------------------------

D0_ZERO_COUNT_UNIT_CHOICES = ("refuse", "half_count")


def d0_response_unit(values, zero_count_unit="refuse"):
    """The frozen D0 response unit of a non-exhaustive, non-real outcome: the mean of the
    starting observations. With `half_count`, a start whose observations are ALL
    zero (so the mean is exactly 0) takes (sum + 0.5) / n = 0.5 / n instead of being refused;
    a positive mean is the mean whatever the option; a negative or non-finite mean, or a zero
    mean of values that are not all zero, is refused as before. Returns (unit, applied)."""
    values = np.asarray(values, dtype=float)
    unit = float(values.mean())
    applied = False
    if zero_count_unit == "half_count" and len(values) and bool(np.all(values == 0.0)):
        unit = (float(values.sum()) + 0.5) / len(values)
        applied = True
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError("D0 response unit is not positive; no fallback")
    return unit, applied


def select_d0_designs(spec, pool, policy, seed):
    """Use the original fraction rule or a separate, non-replacing RNG draw."""
    if policy == "random2":
        count = 3 if spec.name == "location" else 2
        if len(pool.rows) < count:
            raise ValueError("design pool is too small for random D0")
        indices = np.random.default_rng(seed_for("boxinggym_d0", spec.name, seed)).choice(
            len(pool.rows), size=count, replace=False
        )
        return [(pool.rows[i], pool.handles[i]) for i in indices]
    if policy != "fractions4":
        raise ValueError("unknown D0 policy")
    if len(spec.input_names) == 1:
        return [(np.asarray([value]), encode_design(spec, [value])) for value in spec.d0()]
    return [(pool.rows[i], pool.handles[i]) for i in pool.d0_indices()]


def unobserved_q_mask(evaluation_x, observed):
    """Exact equality in declared input units; no rounding or proximity rule."""
    q = np.asarray(evaluation_x, dtype=float)
    q = q[:, None] if q.ndim == 1 else q
    known = np.asarray(observed, dtype=float)
    if known.size == 0:
        return np.ones(len(q), dtype=bool)
    known = known.reshape(-1, q.shape[1])
    return ~np.any(np.all(q[:, None, :] == known[None, :, :], axis=2), axis=1)


def _truth_death(environment, x):
    env = environment.env
    return env.N * -np.expm1(-float(env.theta) * np.asarray(x, dtype=float))


def _truth_peregrines(environment, x):
    env = environment.env
    t = np.asarray(x, dtype=float)
    return np.exp(env.alpha + env.beta1*t + env.beta2*t**2 + env.beta3*t**3)


def _truth_dugongs(environment, x):
    env = environment.env
    mean = env.a - env.b * abs(env.l)**np.asarray(x, dtype=float)
    # step returns abs(Normal(mean, .25)), so the observable is folded Normal.
    sigma = 0.25
    return sigma*np.sqrt(2/np.pi)*np.exp(-0.5*(mean/sigma)**2) + mean*(2*ndtr(mean/sigma)-1)


def _truth_survival(environment, x):
    env = environment.env
    rows = np.asarray(x, dtype=float)
    return expit(rows[:, 0] * np.exp(env.beta*rows[:, 1]) * env.lambda0)


def _truth_hyperbolic(environment, x):
    env = environment.env
    rows = np.asarray(x, dtype=float)
    k, alpha = env.truth
    return env.epsilon + (1-2*env.epsilon)*ndtr(
        (rows[:, 1]/(1+k*rows[:, 2])-rows[:, 0])/alpha
    )


def _truth_location(environment, x):
    return np.asarray([environment.env.signal_intensity(row) for row in np.asarray(x)])


def _truth_irt(environment, x):
    env = environment.env
    parameters = environment.truth_parameters
    if not all(name in parameters for name in ("alpha", "beta")):
        return None
    student, question = np.asarray(x, dtype=int).T
    difference = parameters["alpha"][student] - parameters["beta"][question]
    if env.mode in ("2pl", "3pl"):
        difference *= parameters["gamma"][question]
    probability = expit(difference)
    if env.mode == "3pl":
        guessing = parameters["c"][question]
        probability = guessing + (1-guessing)*probability
    return probability


def _truth_moral_machines(environment, x):
    env = environment.env
    coefficients = np.asarray([getattr(env, "beta_"+name) for name in (
        "human_count", "gender", "age", "social_status", "fitness", "species", "intervention"
    )])
    logit = env.alpha + env.beta_group + np.asarray(x) @ coefficients
    # Native choice = 2-Bernoulli(logit); the adapter models choice-1.
    return expit(-logit)


def _truth_emotion(environment, x):
    env = environment.env
    names = ("alpha", "beta_win", "beta_PE", "beta_absPE", "sigma")
    if not all(hasattr(env, name) for name in names):
        return None
    rows = np.asarray(x, dtype=float)
    win = rows[:, 6]
    pe = win - np.sum(rows[:, :3]*rows[:, 3:6], axis=1)
    mean = (np.asarray(env.alpha) + win[:, None]*env.beta_win
            + pe[:, None]*env.beta_PE + np.abs(pe[:, None])*env.beta_absPE)
    sigma = np.asarray(env.sigma)
    # Native output is clip(int(Normal),1,9), not the latent Normal mean.
    thresholds = np.arange(2, 10)
    return 1 + ndtr((mean[:, :, None]-thresholds)/sigma[None, :, None]).sum(axis=-1)


def composite_value(values):
    """One composite score from the declared components' own scores.

    `COMPOSITE_SCORE_RULE` - the mean over the components - for every NUMBER,
    which is bit-for-bit what `float(np.mean(...))` gave before this helper
    existed. A non-numeric entry is a LABEL, not a measurement (the KL
    diagnostic `kl_method`): averaging one is meaningless, so the shared label
    is carried through and a disagreement
    between components is an error rather than a silently dropped field.
    """
    if any(isinstance(value, str) for value in values):
        if len(set(values)) != 1:
            raise ValueError("components disagree on a non-numeric score label")
        return values[0]
    return float(np.mean(values))


def mda_score_row(observation_count, predictions, truth_mean, brier, previous_best=None,
                  *, location=False, previous_best_log=None):
    """One sidecar row; a zero-variance/absent truth has no defined NMSE."""
    value = None
    if truth_mean is not None:
        truth = np.asarray(truth_mean, dtype=float)
        means = np.asarray(predictions, dtype=float)
        variance = np.var(truth, axis=0)
        if truth.size and np.isfinite(truth).all() and np.all(variance > 0):
            value = float(np.mean(np.mean((means-truth)**2, axis=0)/variance))
    best = previous_best
    if value is not None:
        best = value if best is None else min(best, value)
    row = {"x": observation_count, "nmse_truth": "not_available" if value is None else value,
           "brier": brier, "best_so_far_nmse_truth": "not_available" if best is None else best}
    if location:
        # Q and its truth variance remain fixed; only undefined predictions
        # leave the numerator's mean. Never clip intensities into log1p's domain.
        means = np.asarray(predictions, dtype=float)
        undefined = means.size
        log_value = None
        if truth_mean is not None:
            truth = np.asarray(truth_mean, dtype=float)
            if means.shape != truth.shape:
                raise ValueError("location predictions and truth must have the same shape")
            valid = np.isfinite(means) & (means > -1)
            undefined = int(np.count_nonzero(~valid))
            if truth.size and np.isfinite(truth).all() and np.all(truth > -1):
                log_truth = np.log1p(truth)
                log_variance = float(np.var(log_truth))
                if valid.any() and log_variance > 0:
                    log_value = float(np.mean((np.log1p(means[valid]) - log_truth[valid])**2)
                                      / log_variance)
        log_best = previous_best_log
        if log_value is not None:
            log_best = log_value if log_best is None else min(log_best, log_value)
        row.update(nmse_log_truth="not_available" if log_value is None else log_value,
                   best_so_far_nmse_log_truth="not_available" if log_best is None else log_best,
                   nmse_log_truth_undefined_points=undefined)
    return row


def write_mda_points(path, rows):
    """Write the MDA-comparison sidecar rows to `path` as CSV (exclusive create)."""
    import csv

    with path.open("x", newline="") as stream:
        fields = ("x", "nmse_truth", "brier", "best_so_far_nmse_truth")
        if rows and "nmse_log_truth" in rows[0]:
            fields += ("nmse_log_truth", "best_so_far_nmse_log_truth",
                       "nmse_log_truth_undefined_points")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

TASKS = {
    # death_process.py:319-324 `step` returns binom.rvs(self.N, eta) with N=50
    # (:274) -> a BOUNDED COUNT with a fixed trial count. Table 2 row 1: logit
    # link, Binomial(N,p) initial, Beta-Binomial / ZI-Binomial revisions.
    # Time is validated as lower_bound < t < upper_bound (:332-336) with
    # lower_bound=0, upper_bound=2, and the goal samples evaluation times from
    # uniform(lower_bound+0.001, upper_bound) (:29).
    "death": TaskSpec(
        name="death",
        truth_mean=_truth_death,
        truth_law=death_law,
        module="boxing_gym.envs.death_process",
        env_class="DeathProcess",
        goal_class="DirectDeath",
        input_name="t",
        input_range=(0.001, 2.0),
        design_range=(0.001, 1.999),
        outcome_type="bounded_count",
        support=(0, 50),
        trials=50,
        prediction_format=(
            "evaluate_predictions parses int(float(pred)) (death_process.py:46), which TRUNCATES; "
            "we therefore submit str(int(round(E[Y|t]))) so the environment's parse is the "
            "identity on the integer we intend - the MSE-optimal point of our BMA predictive "
            "rounded onto the integer lattice the evaluator scores on."
        ),
    ),
    # peregrines.py:256-272 `step` draws pm.Poisson(mu=exp(alpha + b1 t + b2 t^2
    # + b3 t^3)) -> an UNBOUNDED COUNT. Table 2 row 2: log link, Poisson initial,
    # Negative Binomial revision. Inputs are validated to [lower_limit,
    # upper_limit] = [0, 5] (:196, :280); evaluation times are round(Beta(2,5)*4,
    # 1) in [0, 4] (:29).
    "peregrines": TaskSpec(
        name="peregrines",
        truth_mean=_truth_peregrines,
        truth_law=peregrines_law,
        module="boxing_gym.envs.peregrines",
        env_class="Peregrines",
        goal_class="DirectGoal",
        input_name="tt",
        input_range=(0.0, 5.0),
        outcome_type="unbounded_count",
        support=(0, None),
        # Table 2 row 2 admits the zero-inflated Poisson "only where zeros are
        # natural". A peregrine-falcon ABUNDANCE count is a species count over
        # time, and absence in a survey year is an outcome the count can take -
        # a statement about the public task description, not about the
        # environment's generating code, which this module never reads for a
        # modelling decision. Declared here, before the run.
        zero_inflation_admissible=True,
        prediction_format=(
            "evaluate_predictions parses int(float(x)) (peregrines.py:53); we submit "
            "str(int(round(E[Y|t]))) for the same reason as death."
        ),
    ),
    # dugongs.py:252-259 `step` draws pm.Normal(mu=a - b*|l|^age, sigma=0.25) and
    # returns float(abs(obs)) -> a strictly POSITIVE real length. Table 2 row 4:
    # log link, LogNormal-with-mean initial, Gamma-with-mean revision. Ages are
    # validated to [lower_limit, upper_limit] = [0, 5] (:189, :268); evaluation
    # ages are Beta(2,5)*4 in [0, 4] (:34).
    "dugongs": TaskSpec(
        name="dugongs",
        truth_mean=_truth_dugongs,
        truth_law=dugongs_law,
        module="boxing_gym.envs.dugongs",
        env_class="Dugongs",
        goal_class="DirectGoal",
        input_name="age",
        input_range=(0.0, 5.0),
        outcome_type="positive",
        support=(0.0, None),
        prediction_format=(
            "evaluate_predictions parses float(p) (dugongs.py:64); we submit repr(E[Y|age]) at "
            "full double precision - the MSE-optimal point of our BMA predictive density."
        ),
    ),
    # ---- the multi-input tasks ----
    # survival_analysis.py. TWO declared inputs and a BINARY outcome -> Table 2
    # row 5: logit link, Bernoulli, no revisions.
    #   * outcome: `_simulate_patient` (:273-284) draws
    #     `pm.Bernoulli("death_outcome", p=invlogit(t * exp(beta*m) * lambda0))`
    #     and `step` (:299-301) returns the stored 0/1 -> binary, one trial.
    #   * design space: the environment FIXES it. `validate_input` (:303-311)
    #     takes a patient ID in 0..num_patients-1 and `run_experiment`
    #     (:313-319) calls `step(patientID)`, which reads the pre-simulated
    #     record `self.outcomes[patientID]` = [outcome, time, metastasized]
    #     (:286-294). The model-facing inputs are that record's covariates.
    #   * ranges: `__init__(self, num_patients=100, time_upper_bound=10)` (:212)
    #     and the covariates are drawn as `uniform(0, time_upper_bound)` and
    #     `randint(0, 2)` (:288-289); evaluation covariates use the same two
    #     draws (:38-:39).
    #   * evaluation tuple: `self.eval_points.append((metastasized,
    #     time_since_surgery, outcome))` (:43) - metastasized FIRST, so the
    #     declared order (time, metastasized) reads positions (1, 0).
    #   * metric: `evaluate_predictions` (:54-67) parses "1" before "0" and
    #     returns (accuracy, std) - HIGHER is better, unlike the MSE tasks.
    "survival": TaskSpec(
        name="survival",
        truth_mean=_truth_survival,
        truth_law=bernoulli_truth_mean_law,
        module="boxing_gym.envs.survival_analysis",
        env_class="SurvivalAnalysis",
        goal_class="DirectGoal",
        input_name="time",
        input_range=(0.0, 10.0),
        extra_inputs=({"name": "metastasized", "range": [0.0, 1.0]},),
        integer_inputs=("metastasized",),
        outcome_type="binary",
        support=(0, 1),
        trials=BINARY_TRIALS,
        environment_fixed_pool=True,
        fixed_pool_attribute="outcomes",
        # `self.outcomes[i]` is `[outcome, time, metastasized]` (:286-294); only
        # positions 1 and 2 - the two PUBLIC covariates - are ever read.
        fixed_pool_record_index=(1, 2),
        design_encoder="pool_index",
        eval_input_index=(1, 0),
        env_metric="accuracy",
        env_metric_direction="higher_is_better",
        # Pre-declared: the design IS the patient ID -
        # `self.outcomes[patientID]` is fixed at world-build time - so
        # re-observing one yields no new information. Excluded from the
        # candidate pool for every arm; see `design_exclusion` above.
        design_exclusion="observed",
        prediction_format=(
            "evaluate_predictions (survival_analysis.py:54-67) parses a prediction by testing "
            '\'"1" in p\' FIRST and \'"0" in p\' second, so any decimal string containing a 1 '
            "would be read as 1; we therefore submit exactly str(0) or str(1) - the posterior "
            "predictive MODE of our BMA Bernoulli predictive, which is the accuracy-optimal "
            "point prediction for the 0/1 metric the evaluator computes."
        ),
    ),
    # hyperbolic_temporal_discount.py. THREE declared inputs and a BINARY
    # outcome -> Table 2 row 5 again.
    #   * outcome: `_select_delay` (:415-421) draws
    #     `np.random.binomial(n=1, p=epsilon + (1-2 epsilon) Phi((V1-V0)/alpha))`
    #     and `step` (:423-427) returns that 0/1 -> binary, one trial.
    #   * design space: `validate_input` (:429-438) parses "[iR, dR, D]" as
    #     THREE integers (`int(float(item))`), rejects `items[0] > items[1]`
    #     and rejects `items[2] <= 0`; `run_experiment` (:440-447) unpacks them
    #     as `iR, dR, Days`. The second input is the DELAYED reward dR (an
    #     earlier spec called it `iD`, which was wrong and never executed).
    #   * ranges: the goal's own evaluation generator (:46-:48) draws
    #     `ir = int(uniform(1, 300))`, `dr = int(uniform(ir+1, 300))`,
    #     `days = int(randint(1, 365))`.
    #   * evaluation tuple: `(ir, dr, days, choice)` (:52) - declared order is
    #     the tuple order, positions (0, 1, 2).
    #   * metric: `evaluate_predictions` (:65-78) is the same accuracy as
    #     survival's - HIGHER is better.
    "hyperbolic": TaskSpec(
        name="hyperbolic",
        truth_mean=_truth_hyperbolic,
        truth_law=bernoulli_truth_mean_law,
        module="boxing_gym.envs.hyperbolic_temporal_discount",
        env_class="TemporalDiscount",
        goal_class="DirectGoal",
        input_name="iR",
        input_range=(1.0, 300.0),
        extra_inputs=(
            {"name": "dR", "range": [1.0, 300.0]},
            {"name": "D", "range": [1.0, 365.0]},
        ),
        integer_inputs=("iR", "dR", "D"),
        outcome_type="binary",
        support=(0, 1),
        trials=BINARY_TRIALS,
        design_encoder="int_list",
        eval_input_index=(0, 1, 2),
        env_metric="accuracy",
        env_metric_direction="higher_is_better",
        prediction_format=(
            "evaluate_predictions (hyperbolic_temporal_discount.py:65-78) parses a prediction by "
            'testing \'"1" in p\' FIRST and \'"0" in p\' second; we submit exactly str(0) or '
            "str(1) - the posterior predictive MODE of our BMA Bernoulli predictive."
        ),
    ),
    # location_finding.py. A 2-D design point and a REAL outcome -> Table 2
    # row 3: identity link, Normal(mu,sigma) initial, Student-t(nu=4) and
    # heteroscedastic Normal revisions.
    #   * outcome: `step` (:446-449) returns `np.random.normal(intensity,
    #     sigma)` with `sigma=0.5` (:382) and `run_experiment` (:463-475)
    #     rounds it to two decimals -> a real-valued, sign-unrestricted
    #     observation. It is NOT a positive-outcome row: nothing in the
    #     environment constrains a draw to stay positive.
    #   * design space: `validate_input` (:451-461) accepts any list of `dim`
    #     floats and imposes NO range, so the declared box below is OUR design
    #     space, not the environment's; evaluation locations are
    #     `np.random.normal(0, 1, (dim,))` (:33), which the box covers to four
    #     standard deviations.
    #   * evaluation tuple: `(loc, choice)` (:35) where `loc` is the whole
    #     length-`dim` vector and `choice = signal_intensity(loc)` is the
    #     NOISE-FREE intensity - so the held-out target has no observation
    #     noise while our observations do.
    #   * metric: `evaluate_predictions` (:64-79) is `float(prediction)` then
    #     MSE - LOWER is better.
    #   * the multi-source structure stays inside the environment, as briefed;
    #     the grammar sees only the two coordinates and their standardizations.
    "location": TaskSpec(
        name="location",
        truth_mean=_truth_location,
        truth_law=location_law,
        module="boxing_gym.envs.location_finding",
        env_class="Signal",
        goal_class="DirectGoal",
        input_name="x1",
        input_range=(-4.0, 4.0),
        extra_inputs=({"name": "x2", "range": [-4.0, 4.0]},),
        outcome_type="real",
        support=(None, None),
        design_encoder="float_list",
        eval_input_index=(0,),
        eval_input_is_vector=True,
        env_metric="mse",
        prediction_format=(
            "evaluate_predictions (location_finding.py:64-79) strips the string, splits on '=' "
            "if present and calls float(); we submit repr(E[Y|x]) at full double precision - "
            "the MSE-optimal point of our BMA predictive density."
        ),
    ),
# ---- categorical inputs and three more tasks ----
    # irt.py. TWO CATEGORICAL inputs and a BINARY outcome -> Table 2 row 5:
    # logit link, Bernoulli, no revisions.
    #   * outcome: `reset` (:305) builds
    #     `responses = pm.Bernoulli('responses', p=p, shape=(S, Q))` (:330) and
    #     fixes the whole matrix with ONE `pm.sample_prior_predictive(samples=1)`
    #     draw (:333, stored at :340); `step` (:403) returns that stored cell
    #     with `return int(obs)` (:405) -> binary, one trial. Because the cell is
    #     FIXED at world-build time, re-observing a (student, question) pair
    #     carries no new information, which is the `design_exclusion="observed"`
    #     condition verbatim.
    #   * inputs: `IRT.__init__(num_students=6, num_questions=6, mode='2pl', ...)`
    #     (:297) -> a 6 x 6 grid of student and question IDENTIFIERS. An
    #     identifier is a LABEL, not a magnitude: student 5 is not "five times"
    #     student 1, and `validate_input` (:378-:391) only parses two integers.
    #     So both are declared CATEGORICAL with six levels, and the grammar sees
    #     them only as coefficient indices - `b1[student]` is a per-student
    #     offset vector with the shared Table 3 `signed` = N(0,1) prior.
    #   * design space: the full 36-cell product grid MINUS the cells the
    #     environment refuses. `run_experiment` (:407-:416) returns
    #     `"You cannot query this observation, try again.", False` (:413) for any
    #     design in `self.restricted_pair`, which `DirectCorrectness.__init__`
    #     sets to its ten evaluation cells (:14-:36, registered at :36) - and
    #     `validate_input` never checks that list, so the pool must
    #     (`environment_refused_attribute`). The environment's own random-input
    #     sampler excludes exactly the same list (:399). Ten cells may repeat, so
    #     k = |distinct restricted| <= 10 and the pool holds 36 - k >= 26 rows,
    #     which is >= D0 4 + 20 rounds = 24 distinct designs.
    #   * held-out set: `DirectCorrectness.__init__` builds EXACTLY ten points
    #     (five correct at :22-:27, five incorrect at :29-:34) and
    #     `get_goal_eval_question` (:46-:53) only advances a pointer over them,
    #     so |Q| is fixed at ten by the environment - `eval_point_limit`.
    #   * evaluation tuple: `(student_id, question_id, result)` (:27, :34) -
    #     declared order is the tuple order, positions (0, 1).
    #   * metric: `evaluate_predictions` (:55-:68) parses `int(pred)` (falling
    #     back to the last integer in the string) and returns
    #     `(1 - accuracy, std)` - an ERROR RATE, so LOWER is better, unlike
    #     survival's and hyperbolic's accuracy.
    "irt": TaskSpec(
        name="irt",
        truth_mean=_truth_irt,
        truth_law=bernoulli_truth_mean_law,
        module="boxing_gym.envs.irt",
        env_class="IRT",
        goal_class="DirectCorrectness",
        input_name="student",
        input_range=(0.0, 5.0),
        extra_inputs=({"name": "question", "range": [0.0, 5.0]},),
        categorical_inputs=(("student", 6), ("question", 6)),
        integer_inputs=("student", "question"),
        outcome_type="binary",
        support=(0, 1),
        trials=BINARY_TRIALS,
        design_encoder="int_list",
        environment_pool_source="categorical",
        environment_refused_attribute="restricted_pair",
        eval_input_index=(0, 1),
        eval_point_limit=10,
        env_metric="error_rate",
        env_metric_direction="lower_is_better",
        design_exclusion="observed",
        # There is no numeric input to retarget CORE's `u`/`z` templates onto,
        # so the four initial structures and the twenty P1 mock templates are
        # declared here, before any run: the additive ladder over the two
        # per-level offset vectors, and twenty grammar-valid variations of it.
        # None of them is the environment's own 2PL form
        # `gamma[q]*(alpha[s]-beta[q])`, which is never handed to the pool.
        initial_forms=(
            "b0",
            "b0+b1[student]",
            "b0+b1[question]",
            "b0+b1[student]+b2[question]",
        ),
        mock_forms=(
            "b0+b1[student]",
            "b0+b1[question]",
            "b0+b1[student]+b2[question]",
            "b1[student]+b2[question]",
            "b0+b1[student]*b2[question]",
            "b0+b1[student]+b2[question]+b3[student]*b4[question]",
            "b1[student]",
            "b1[question]",
            "b0+tanh(b1[student])",
            "b0+b1[student]+tanh(b2[question])",
            "b0+b1[student]-b2[question]",
            "b0+2*b1[student]+b2[question]",
            "b0+b1[student]+2*b2[question]",
            "b0+tanh(b1[student]+b2[question])",
            "b0+b1[student]*(b2[student]-b3[question])",
            "b0+b1[question]*(b2[student]-b3[question])",
            "b0+b1[student]+b2[question]+b3[student]",
            "b0+square(b1[student])+b2[question]",
            "b0+b1[student]+square(b2[question])",
            "b0+b1[student]*b2[question]+b3[student]+b4[question]",
        ),
        prediction_format=(
            "evaluate_predictions (irt.py:55-:68) parses int(pred), falling back to the LAST "
            "integer in the string, and scores 1 - accuracy; we submit exactly str(0) or "
            "str(1) - the posterior predictive MODE of our BMA Bernoulli predictive, which is "
            "the accuracy-optimal point prediction for the 0/1 metric the evaluator computes."
        ),
    ),
    # moral_machines.py. SEVEN numeric attribute contrasts and a BINARY choice
    # -> Table 2 row 5 again.
    #   * outcome: `step` (:469-:534) draws
    #     `choice = pm.Bernoulli('choice', logit_p=y_hat)` (:491) from the
    #     environment's own eight-preference linear predictor (:487-:490) and
    #     re-codes it as `1 - int(draw) + 1` in {1, 2} (:493) -> binary, one
    #     trial. `outcome_offset=-1.0` puts it on the {0, 1} the Bernoulli path
    #     models; `format_predictions` submits the environment's own 1/2 code.
    #   * inputs: the SEVEN varying contrasts of the environment's own likelihood
    #     (:487-:490, identical to the goal's own EIG model at :109-:122):
    #     human-count, gender, age, social-status, fitness and species
    #     differences plus the +-1 intervention. The eighth listed contrast,
    #     `group`, is a CONSTANT `+ self.beta_group` term (:487) that the
    #     intercept absorbs - see MORAL_CONSTANT_CONTRAST.
    #   * design space: the environment's OWN generator,
    #     `sample_random_input` (:457-:467, group sizes `randint(1, 5)` at :458-:459
    #     and characters drawn from its own eighteen-character list), filtered by its
    #     own `validate_input` (:536-:553).
    #   * outcome location: `run_experiment` (:555-:564) returns the NARRATION
    #     string, so the outcome is read from `self.observed_data[-1]`
    #     `(group1, group2, intervention, choice)` (:563), position 3.
    #   * evaluation tuple: `(group1, group2, intervention, choice)` (:49).
    #   * metric: `evaluate_predictions` (:61-:74) parses '"2" in p' FIRST then
    #     '"1" in p' and returns (accuracy, std) - HIGHER is better.
    "moral_machines": TaskSpec(
        name="moral_machines",
        truth_mean=_truth_moral_machines,
        truth_law=bernoulli_truth_mean_law,
        module="boxing_gym.envs.moral_machines",
        env_class="MoralMachine",
        goal_class="DirectPrediction",
        # The declared ranges are the EXACT attainable bounds of the
        # environment's own generator: group sizes are `randint(1, 5)` so each
        # group holds one to four characters (moral_machines.py:458-:459), and
        # `_get_char_attr` (:411-:455) gives each character
        # gender in {0,-1}, age in {0,1}, social_status in {0,1,1.5},
        # fitness in {-1,0,1} and species in {-1,1}. A group sum is therefore
        # in [-4,0], [0,4], [0,6], [-4,4] and [-4,4], and a group-1-minus-
        # group-2 contrast in [-4,4], [-4,4], [-6,6], [-8,8] and [-8,8]; the
        # human count is a count of one to four per group, so [-4,4].
        input_name="human_count_diff",
        input_range=(-4.0, 4.0),
        extra_inputs=(
            {"name": "gender_diff", "range": [-4.0, 4.0]},
            {"name": "age_diff", "range": [-4.0, 4.0]},
            {"name": "social_status_diff", "range": [-6.0, 6.0]},
            {"name": "fitness_diff", "range": [-8.0, 8.0]},
            {"name": "species_diff", "range": [-8.0, 8.0]},
            {"name": "intervention", "range": [-1.0, 1.0]},
        ),
        outcome_type="binary",
        support=(0, 1),
        trials=BINARY_TRIALS,
        design_encoder="pool_handle",
        environment_pool_source="sampled",
        pool_sample_count=400,
        feature_map="moral_contrasts",
        observation_source="observed_data",
        observed_data_index=3,
        outcome_offset=-1.0,
        openai_narration=True,
        openai_narration_module="boxing_gym.envs.moral_machines",
        env_metric="accuracy",
        env_metric_direction="higher_is_better",
        prediction_format=(
            "evaluate_predictions (moral_machines.py:61-:74) parses a prediction by testing "
            "'\"2\" in p' FIRST and '\"1\" in p' second, so any string containing a 2 reads as "
            "2; we therefore submit exactly str(1) or str(2) - the posterior predictive MODE of "
            "our BMA Bernoulli predictive, re-coded from the modelled {0,1} back into the "
            "environment's own {1,2} by outcome_offset."
        ),
    ),
    # emotion.py. SEVEN numeric inputs and EIGHT real outputs -> Table 2 row 3
    # per component (identity link, Normal(mu,sigma) initial, Student-t(nu=4)
    # and heteroscedastic Normal revisions), reported as the composite of Table
    # 2's last row ("component-wise scores are composite and labelled so").
    #   * outcome: `step` (:371-:445) draws eight independent
    #     `pm.Normal(mu=mean[j], sigma=self.sigma[j])` values (:390-:397) from
    #     the environment's own `alpha + beta_win*win + beta_PE*PE +
    #     beta_absPE*|PE|` predictor (:388), then codes each onto the 1-9 Likert
    #     scale with `int()` and a clip (:411-:414). The eight coded values are
    #     the observation; the underlying draw is Gaussian, which is why the
    #     eight runners are the identity-link REAL runners and the int/clip is
    #     recorded as a deterministic monotone coding of it, not modelled.
    #   * design space: the environment's OWN `sample_random_input` (:361-:369,
    #     prizes uniform(0,100), probabilities Dirichlet(1,1,1), win index
    #     0/1/2) filtered by its own `validate_input` (:447-:467), which REFUSES
    #     any triple whose float sum is not exactly 1 (:458-:459).
    #   * outcome location: `run_experiment` (:469-:477) returns the NARRATION
    #     string, so the outcome is read from `self.observed_data[-1]`
    #     `(prizes, probs, win, emotions)` (:476), position 3.
    #   * evaluation tuple: `(prizes, probs, win, emotion_values)` (:37), with
    #     `win` an INDEX rather than a prize value - see
    #     EMOTION_WIN_UNIT_DEFECT, recorded per run.
    #   * metric: `evaluate_predictions` (:59-:81) requires the exact eight-line
    #     "Happiness: x/9 ..." shape, and returns the MEAN ABSOLUTE ERROR over
    #     the eight emotions averaged over the held-out points - LOWER is better.
    "emotion": TaskSpec(
        name="emotion",
        truth_mean=_truth_emotion,
        truth_law=emotion_law,
        module="boxing_gym.envs.emotion",
        env_class="EmotionFromOutcome",
        goal_class="DirectEmotionPrediction",
        input_name="v1",
        input_range=(0.0, 100.0),
        extra_inputs=(
            {"name": "v2", "range": [0.0, 100.0]},
            {"name": "v3", "range": [0.0, 100.0]},
            {"name": "p1", "range": [0.0, 1.0]},
            {"name": "p2", "range": [0.0, 1.0]},
            {"name": "p3", "range": [0.0, 1.0]},
            {"name": "win", "range": [0.0, 100.0]},
        ),
        outcome_type="real",
        support=(1.0, 9.0),
        design_encoder="pool_handle",
        environment_pool_source="sampled",
        pool_sample_count=400,
        feature_map="emotion_inputs",
        observation_source="observed_data",
        observed_data_index=3,
        components=(
            "happiness",
            "sadness",
            "anger",
            "surprise",
            "fear",
            "disgust",
            "contentment",
            "disappointment",
        ),
        openai_narration=True,
        openai_narration_module="boxing_gym.envs.emotion",
        env_metric="mae",
        env_metric_direction="lower_is_better",
        prediction_format=(
            "evaluate_predictions (emotion.py:59-:81) matches the exact eight-line "
            "'Happiness: x/9\\nSadness: ...' shape with a regex and then averages the absolute "
            "error over the eight emotions; we submit repr(E[Y_j|x]) for each component at full "
            "double precision - the MAE-relevant point of each component's BMA predictive "
            "density - in that exact eight-line form."
        ),
    ),
    # lotka_volterra.py. REGISTERED, NOT RUNNABLE - see `note`.
    #   * `LotkaVolterra.__init__` (:214-:233) declares `lower_limit=0,
    #     upper_limit=50`, so the declared design range is the environment's
    #     own [0, 50]; `validate_input` (:308-:322) enforces exactly that.
    #   * `step` (:293-:306) integrates the ODE with `scipy.integrate.odeint`
    #     and returns `np.round(solution[-1]).astype(int).clip(min=0)` - TWO
    #     nonnegative integer populations, and NO observation noise anywhere.
    #     `reset` (:236-:242) draws the four rate parameters, so worlds differ,
    #     but the OBSERVATION given a world is exact.
    "lotka_volterra": TaskSpec(
        name="lotka_volterra",
        module="boxing_gym.envs.lotka_volterra",
        env_class="LotkaVolterra",
        goal_class="DirectGoal",
        input_name="t",
        input_range=(0.0, 50.0),
        outcome_type="trajectory",
        support=(0.0, None),
        runnable=False,
        note=(
            "specification_only: the environment integrates a DETERMINISTIC "
            "Lotka-Volterra ODE (lotka_volterra.py:293-:306, scipy.integrate.odeint) and returns "
            "the two rounded, clipped populations at a queried time with NO observation noise, so "
            "the task table gives it no likelihood family (the task-preset notes record 'none - "
            "deterministic ODE' and 'ode, cos not in the grammar'). The table's last "
            "row (trajectory / feature vector, Gaussian with a task-specific scale) would need a "
            "scale this environment never states, and its outputs are a coupled ODE SOLUTION "
            "rather than a function of the design: expressing it needs an ODE grammar - a "
            "differential-equation right-hand side plus an integrator inside the forward model - "
            "which this tree's algebraic expression grammar (count_models.validate_expression) "
            "does not have. Declared here so the task is registered, "
            "counted and cited, and so that `run()` refuses it explicitly instead of failing "
            "somewhere inside a fit."
        ),
    ),
}


def prior_particles(hypothesis, count, *, rng_model_id=None):
    """Prior draws of every coefficient and dispersion parameter (dedicated stream)."""
    rng = np.random.default_rng(PRIOR_PROBE_SEED ^ int(
        hypothesis.model_id if rng_model_id is None else rng_model_id, 16))
    draws = {}
    # Proposed-prior ablation: an accepted proposed prior replaces the role prior of
    # that one coefficient, here as in `count_inference.build_model`, so the
    # probe draws from the prior the fit will use. None: `PRIOR_RULES[role]`.
    rules = {
        **{
            name: coefficient_prior(hypothesis, name, role)
            for name, role in hypothesis.parameters
        }
    }
    for name in hypothesis.noise_parameters:
        rules[name] = PRIOR_RULES.get(name) or SCALAR_NOISE_PRIORS[name]
    # An INDEXED coefficient draws a [sample, level] block from the SAME
    # single role prior - the shared Normal(0,1) of Table 3's "standardised
    # slope" row - so no new prior enters the probe. Empty for every hypothesis
    # without an indexed coefficient, which keeps `size=count` and therefore the
    # exact earlier draw sequence for CORE and every task without one.
    shapes = dict(getattr(hypothesis, "parameter_shapes", ()) or ())
    for name, (family, a, b) in rules.items():
        size = count if name not in shapes else (count, shapes[name])
        if family == "Normal":
            draws[name] = rng.normal(a, b, size)
        elif family == "LogNormal":
            draws[name] = rng.lognormal(a, b, size)
        elif family == "Gamma":
            draws[name] = rng.gamma(a, 1.0 / b, size)
        elif family == "Beta":
            draws[name] = rng.beta(a, b, size)
        elif family == "HalfNormal":
            draws[name] = np.abs(rng.normal(0.0, a, size))
        else:
            raise ValueError(f"prior probe cannot sample {family}")
    return draws


def prior_probe(hypothesis, interface, response_unit, count=PRIOR_PROBE_SAMPLES,
                response_center=0.0, *, rng_model_id=None):
    """Finiteness gate before any SMC fit, on the interface's own reference grid.

    Identical in intent and in seeding to `core_count.prior_probe`; the scalar
    branch checks the log-link mean instead of the count PMF, because an
    unbounded count and a positive outcome have no finite count table.
    """
    grid = interface_grid(interface, PRIOR_PROBE_GRID_POINTS)
    particles = prior_particles(hypothesis, count, **(
        {} if rng_model_id is None else {"rng_model_id": rng_model_id}))
    result = {"n_samples": int(count), "passed": True, "grid_points": PRIOR_PROBE_GRID_POINTS}
    try:
        if interface["outcome_type"] in EXHAUSTIVE_COUNT_OUTCOMES:
            logpmf = particle_logpmf(hypothesis, particles, grid, interface["trials"], interface)
            # A prior draw given zero weight is not part of the prior
            # predictive average; with none of them this is the same mean.
            pmf = np.exp(active_particle_rows(logpmf)).mean(axis=0)
            counts = np.arange(pmf.shape[-1])
            cumulative = np.cumsum(pmf, axis=-1)
            predictive = {
                "grid": grid.tolist(),
                "mean": (pmf @ counts).tolist(),
                "q05": (cumulative < 0.05).sum(-1).tolist(),
                "q50": (cumulative < 0.50).sum(-1).tolist(),
                "q95": (cumulative < 0.95).sum(-1).tolist(),
                "n_prior_samples": int(count),
            }
        else:
            mean, sd = mixture_moments(
                hypothesis, particles, grid, interface, response_unit, response_center
            )
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(sd)):
                raise ValueError("non-finite mean expression under the prior probe")
            predictive = {
                "grid": grid.tolist(),
                "mean": mean.tolist(),
                "sd": sd.tolist(),
                "n_prior_samples": int(count),
            }
    except Exception as exc:  # noqa: BLE001 - a failed probe is a rejection, not a crash
        detail = f"{type(exc).__name__}: {exc}"
        finite = not any(marker in str(exc) for marker in (*PRIOR_PROBE_NONFINITE, "non-finite"))
        if not finite:
            # Name the sub-expression, input value and range that went
            # non-finite, on the same particles and the same reference grid.
            found = first_nonfinite_subexpression(
                hypothesis.expression, particles, named_input_values(interface, grid), interface
            )
            if found is not None:
                return {**result, "passed": False, "normalization_warning": False,
                        "detail": nonfinite_probe_sentence(found), "exception": detail,
                        "nonfinite": found}, None
        return {**result, "passed": finite, "normalization_warning": finite, "detail": detail}, None
    return result, predictive


class TaskEnvironment:
    """Thin, read-only binding to one BoxingGym environment and its goal.

    Every observation is an environment call (`run_experiment`, which validates
    the design, calls `step` and records it in the environment's own
    `observation_data`). Every held-out point comes from the goal's own
    `get_goal_eval_question` generator, and the environment's evaluator is
    called on exactly those points.
    """

    def __init__(self, spec, seed):
        import importlib

        module = importlib.import_module(spec.module)
        self.spec = spec
        # The environments draw their world parameters from the global numpy
        # stream at construction (`np.random.seed` below governs that, and
        # nothing else in this module uses that stream). A PyMC-backed
        # environment (peregrines, dugongs) draws its world parameters
        # through PyMC's OWN random stream instead, which ignores this numpy
        # seed entirely - the caller must additionally wrap construction in
        # `env_seed_injection(seed)` for those two tasks to be reproducible;
        # `run()` does. death draws its world parameter from scipy, which
        # does obey `np.random.seed`, so it needs no such wrapping.
        np.random.seed(seed)
        self.truth_parameters = {}
        if spec.name == "irt":
            # IRT stores responses but discards the generating latent draw.
            # Retain that SAME draw for scoring, without another RNG call.
            original_prior_predictive = pymc.sample_prior_predictive

            def capture_prior_predictive(*args, **kwargs):
                """Call PyMC's prior predictive and keep the latent world parameters it drew."""
                result = original_prior_predictive(*args, **kwargs)
                for name in ("alpha", "beta", "gamma", "c"):
                    if name in result.prior:
                        self.truth_parameters[name] = np.asarray(result.prior[name])[0, 0].copy()
                return result

            pymc.sample_prior_predictive = capture_prior_predictive
            try:
                self.env = getattr(module, spec.env_class)(**spec.env_kwargs)
            finally:
                pymc.sample_prior_predictive = original_prior_predictive
        else:
            self.env = getattr(module, spec.env_class)(**spec.env_kwargs)
        self.goal = getattr(module, spec.goal_class)(self.env)
        self.observation_calls = 0

    def observe(self, design):
        """One environment call.

        `design` is normally the environment's OWN design string, taken from the
        task's design pool (`DesignPool.handles`) - that is what `run()` passes,
        and it is the only form a multi-input or pool-indexed task can use. A
        NUMERIC design is still accepted and encoded here with the task's
        declared encoder, which for a single-input task is
        `repr(float(design))` exactly; the existing tests call it that way.

        An environment whose `run_experiment` returns a NARRATION string
        instead of the outcome (moral_machines, emotion) is read through its own
        recorded observation - `env.observed_data[-1]`, the tuple it appends at
        moral_machines.py:563 / emotion.py:476 - and never through the
        string. `outcome_offset` then re-codes the raw outcome into the modelled
        one (moral_machines' choice is 1/2, the Bernoulli path needs 0/1);
        `format_predictions` undoes it on the way back out.
        """
        handle = (
            design
            if isinstance(design, str)
            else encode_design(self.spec, np.atleast_1d(np.asarray(design, dtype=float)))
        )
        value, ok = self.env.run_experiment(handle)
        if not ok:
            raise ValueError(f"environment rejected design {handle!r}: {value}")
        self.observation_calls += 1
        if self.spec.observation_source == "return_value":
            return float(value)
        return self.outcome(self.env.observed_data[-1][self.spec.observed_data_index])

    def outcome(self, raw):
        """One environment outcome in MODELLED units: a float, or a list of them."""
        offset = float(self.spec.outcome_offset)
        if self.spec.components:
            values = (
                [raw[name] for name in self.spec.components]
                if isinstance(raw, dict)
                else list(raw)
            )
            if len(values) != len(self.spec.components):
                raise ValueError(f"{self.spec.name}: outcome width does not match components")
            return [float(v) + offset for v in values]
        # `+ 0.0` is the identity on every value a task without an offset produces, but
        # the branch keeps the expression itself unchanged for them.
        return float(raw) if offset == 0.0 else float(raw) + offset

    def accepts(self, handle):
        """Whether the environment's OWN validator accepts this design string."""
        try:
            return not isinstance(self.env.validate_input(handle), str)
        except Exception:  # noqa: BLE001 - a parse failure is a rejected design
            return False

    def refused_designs(self):
        """Designs the environment REFUSES to answer, as integer tuples.

        Empty unless the task declares `environment_refused_attribute`. For irt
        that attribute is `restricted_pair`, the list the goal registers at
        irt.py:36 and `run_experiment` checks at irt.py:412 - the same list the
        environment's own sampler skips at irt.py:399. Only WHICH cells are
        off-limits is read; no stored outcome is ever touched.
        """
        if not self.spec.environment_refused_attribute:
            return frozenset()
        records = getattr(self.env, self.spec.environment_refused_attribute) or []
        return frozenset(
            tuple(round(float(v)) for v in np.atleast_1d(np.asarray(record)))
            for record in records
        )

    def sample_designs(self, count):
        """`count` draws of the environment's OWN random-input generator.

        Returns the declared numeric feature rows and the exact handles the
        environment's parser takes, keeping only draws its own `validate_input`
        accepts and de-duplicating by handle. The draws come from the global
        numpy stream, which `run()` has already seeded from the world seed.
        """
        rows, handles, seen = [], [], set()
        for _ in range(int(count)):
            draw = self.env.sample_random_input()
            row, handle = self.design_features(draw)
            if handle in seen or not self.accepts(handle):
                continue
            seen.add(handle)
            rows.append(row)
            handles.append(handle)
        return rows, handles

    def design_features(self, draw):
        """One generator draw as (declared numeric input row, environment handle)."""
        if self.spec.feature_map == "moral_contrasts":
            group1, group2, intervention = draw
            return (
                moral_contrast_row(self.env, group1, group2, intervention),
                moral_handle(group1, group2, intervention),
            )
        if self.spec.feature_map == "emotion_inputs":
            prizes, probs, win_index = draw
            # The declared seventh input is what `step` will actually receive:
            # `validate_input` turns the index into the PRIZE VALUE
            # (emotion.py:464-:467). See EMOTION_WIN_UNIT_DEFECT.
            return (
                emotion_input_row(prizes, probs, prizes[int(win_index)]),
                emotion_handle(prizes, probs, win_index),
            )
        raise ValueError(f"task {self.spec.name} declares no design generator")

    def eval_inputs(self, point):
        """The declared inputs of one goal eval tuple, in declared order.

        `feature_map == "positions"` is the original index rule, unchanged. The
        two other maps read the eval tuple through the same environment feature
        map the design pool uses, because those goals store the scenario
        objects, not numbers: moral_machines stores
        `(group1, group2, intervention, choice)` (moral_machines.py:49) and
        emotion stores `(prizes, probs, win, emotion_values)` (emotion.py:37).
        """
        if self.spec.feature_map == "positions":
            return self.spec.eval_inputs(point)
        if self.spec.feature_map == "moral_contrasts":
            return moral_contrast_row(self.env, point[0], point[1], point[2])
        if self.spec.feature_map == "emotion_inputs":
            # point[2] is the goal's own `win`, which is an INDEX here and a
            # prize value on the observation path - EMOTION_WIN_UNIT_DEFECT.
            return emotion_input_row(point[0], point[1], point[2])
        raise ValueError(f"task {self.spec.name} declares an unknown feature map")

    def fixed_design_pool(self):
        """The environment's stored design records, as covariate rows plus handles.

        Only the positions `fixed_pool_record_index` declares are read, so a
        record's stored OUTCOME never enters the pool. Those covariates are
        public: the environment prints them in its own system message
        (survival_analysis.py:233-247 lists every patient's time and
        metastasization status before the first observation).
        """
        records = getattr(self.env, self.spec.fixed_pool_attribute)
        index = self.spec.fixed_pool_record_index
        rows = np.asarray(
            [[float(record[position]) for position in index] for record in records], dtype=float
        )
        return rows, [str(i) for i in range(len(records))]

    def evaluation_points(self, count, seed):
        """Populate and return the goal's own held-out points, as (x, y) arrays.

        `y` is an [point, component] matrix for a multi-component task,
        and the outcome is read through `outcome`, so an environment that stores
        its outcome as a dict (emotion's eight named emotions) or in a shifted
        code (moral_machines' 1/2 choice) reaches the runner in modelled units.
        """
        np.random.seed(seed)
        for _ in range(count):
            self.goal.get_goal_eval_question(include_prior=True)
        points = list(self.goal.eval_points[:count])
        x = np.asarray([self.eval_inputs(p) for p in points], dtype=float)
        y = np.asarray([self.outcome(p[-1]) for p in points], dtype=float)
        return (x[:, 0] if len(self.spec.input_names) == 1 else x), y

    def evaluate(self, predictions):
        """The environment's own evaluator, on the goal's own stored points."""
        measurements = [p[-1] for p in self.goal.eval_points]
        if len(predictions) != len(measurements):
            raise ValueError("environment evaluation must cover every stored eval point")
        value = self.goal.evaluate_predictions(predictions, measurements)
        return {"metric": float(value[0]), "spread": float(value[1])}

    def world_parameters(self):
        """The environment's realised world parameters, for the record only.

        Read after construction and written to config.json as provenance; no
        fit, design or proposal ever sees them.
        """
        names = (
            "theta",
            "alpha",
            "beta1",
            "beta2",
            "beta3",
            "a",
            "b",
            "l",
            "lambda0",
            "beta",
            "truth",
            "true_theta",
            "sigma",
            "m",
            "epsilon",
            "num_sources",
            "dim",
            "num_patients",
            "time_upper_bound",
        )
        found = {}
        for name in names:
            value = getattr(self.env, name, None)
            if value is None or isinstance(value, str):
                continue
            try:
                array = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                continue
            if array.size == 1:
                found[name] = float(array.reshape(-1)[0])
            elif array.ndim >= 1 and array.size <= 64:
                found[name] = array.tolist()
        return found


EMOTION_PREDICTION_TEMPLATE = (
    "Happiness: {0}/9\nSadness: {1}/9\nAnger: {2}/9\nSurprise: {3}/9\n"
    "Fear: {4}/9\nDisgust: {5}/9\nContentment: {6}/9\nDisappointment: {7}/9"
)


def format_predictions(spec, mean, component_means=None):
    """Our point predictions in the exact string form the environment parses.

    Two further forms, both read off the environment's own parser:

      * an `outcome_offset` task submits the environment's OWN code, not the
        modelled one - the predictive MODE in modelled units with the offset
        removed, which for moral_machines is exactly "1" or "2" (its evaluator
        tests '"2" in p' first and '"1" in p' second, moral_machines.py:61-:74).
      * a multi-component task submits ONE string per held-out point carrying
        every component, in the exact eight-line shape emotion's evaluator
        regex requires (emotion.py:59-:81); `component_means` is the
        [component][point] array of our per-component predictive means.
    """
    if spec.components:
        if component_means is None:
            raise ValueError(f"task {spec.name} needs one predictive mean per component")
        columns = np.asarray(component_means, dtype=float)
        if columns.shape[0] != len(spec.components):
            raise ValueError(f"task {spec.name}: component mean count mismatch")
        return [
            EMOTION_PREDICTION_TEMPLATE.format(*[repr(float(v)) for v in columns[:, index]])
            for index in range(columns.shape[1])
        ]
    if spec.outcome_type in ("bounded_count", "unbounded_count", "binary"):
        offset = int(spec.outcome_offset)
        if offset:
            return [str(round(float(value)) - offset) for value in mean]
        return [str(round(float(value))) for value in mean]
    return [repr(float(value)) for value in mean]


ENV_SEED_INJECTION_RULE = (
    "boxing_gym.envs.{peregrines,dugongs}.step/reset call pymc.sample_prior_predictive "
    "and pymc.draw without a random_seed, so PyMC reseeds those calls' shared RNGs from "
    "OS entropy every time (pymc.pytensorf.reseed_rngs -> np.random.SeedSequence(None)), "
    "independently of np.random.seed() - two runs with the same --seed therefore draw a "
    "different world and different observations. Fix (adapter-process-local, active only "
    "for the duration of one run() call): pymc.draw and pymc.sample_prior_predictive are "
    "wrapped so a call with random_seed=None gets "
    "seed_k = int(np.random.SeedSequence([world_seed, k]).generate_state(1)[0]), where k "
    "is a per-run call counter shared by both functions, incremented once per intercepted "
    "call, starting at 0; a call that already carries an explicit random_seed is passed "
    "through unchanged. Only the seed argument is touched - the sampled distribution, "
    "shapes and dtypes are exactly what boxing-gym asked for. death_process.py never "
    "calls either function (it draws theta and infected_num from scipy, which already "
    "obeys np.random.seed()), so this is a no-op for the death task."
)


# `DesignPool.available` (defined above) raises
# ValueError("design pool exhausted: ...") once every
# design has been observed. Earlier versions let that propagate to run()'s own
# `except BaseException` handler, which recorded it as a "failed" run
# (failure.json) - a crash, even though "the candidate pool ran out of fresh
# designs before the requested round count" is an ordinary, anticipated
# sizing condition (see `design_pool_target`'s own docstring), not a bug. At
# --rounds <= 20 the irt pool (26 rows: the 36-cell grid minus its <= 10
# refused cells, unchanged by this rule) never actually reaches this branch
# - D0's 4 plus 20 rounds is 24 <= 26 - so this is a defensive path, exercised
# only by a pool deliberately shrunk below what --rounds needs (this rule's
# own test injects exactly that). The fix: the ONE call site that can raise
# it (the `design_exclusion == "observed"` branch at the top of the round
# loop) catches it, ends the round loop after the last successfully finished
# round instead of re-raising, and `summary.json` records status
# "design_pool_exhausted" plus a "design_pool_exhausted_reason" string in
# place of "completed" - a normal exit, not a crash. Nothing about how the
# pool itself is BUILT changes (`build_design_pool`/`build_categorical_pool`
# are untouched), so a run that never exhausts its pool is unaffected byte
# for byte.
DESIGN_POOL_EXHAUSTION_RULE = (
    "DesignPool.available (:706-720) raises ValueError once every design in the pool has "
    "already been observed. Earlier versions let that propagate to run()'s outer exception "
    "handler and recorded it as a crash (failure.json, status 'failed'). Now the "
    "per-round draw in the main loop (the design_exclusion==\"observed\" branch) catches "
    "that exhaustion, ends the round loop after the last round that actually completed, "
    "and the run terminates normally with summary.json[\"status\"] == "
    "\"design_pool_exhausted\" and a \"design_pool_exhausted_reason\" string recording the "
    "round and the pool/observed sizes, instead of raising. The pool itself (irt: 26 rows) "
    "is unchanged; at --rounds <= 20 no declared task's pool is sized to reach this branch "
    "in ordinary operation - it exists for a pool deliberately undersized for the requested "
    "round count."
)


class _NarrationStubResponse:
    """The shape `step` reads: `.choices[0].message.content` and nothing else."""

    class _Message:
        content = "narration suppressed by the adapter (no API call was made)"

    class _Choice:
        message = None

    def __init__(self):
        choice = self._Choice()
        choice.message = self._Message()
        self.choices = [choice]


class _NarrationStubClient:
    """A local stand-in for `openai.OpenAI()` that performs no I/O at all.

    It counts the calls it absorbs so a run can record that the number of paid
    calls is zero BY CONSTRUCTION, not by assertion.
    """

    def __init__(self, log):
        self.log = log
        client = self

        class _Completions:
            @staticmethod
            def create(*args, **kwargs):
                """Record the absorbed call and return a fixed stub response; performs no I/O."""
                client.log.append(
                    {"model": kwargs.get("model"), "messages": len(kwargs.get("messages") or [])}
                )
                return _NarrationStubResponse()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class _OpenAIModuleProxy:
    """What the narration environment's module sees under its own name `openai`:
    every attribute of the real module, except that `OpenAI()` returns the
    narration stub client. Nothing else in the process sees this object."""

    def __init__(self, real_module, factory):
        self._real_module = real_module
        self._factory = factory

    def __getattr__(self, name):
        if name == "OpenAI":
            return self._factory
        return getattr(self._real_module, name)


@contextlib.contextmanager
def openai_narration_stub(module_name):
    """Make ONE narration environment constructible and paid-call-free.

    See `OPENAI_NARRATION_STUB_RULE`. Only the named environment module's own
    `openai` global (the object `boxing_gym.envs.<task>` reaches through its
    `import openai`) is replaced, by a proxy whose `OpenAI()` is the stub; the
    global `openai` module - and with it the structure proposer's real client
    built by count_proposer - is untouched. The name is
    restored in `finally`. Yields the list of absorbed calls, which
    `summary.json` records as `narration_calls_absorbed`.
    """
    import importlib

    import openai

    if not module_name:
        raise ValueError("openai_narration_stub needs the environment module name")
    module = importlib.import_module(module_name)
    if getattr(module, "openai", None) is not openai:
        raise ValueError(f"{module_name} does not bind the openai module under the name 'openai'")
    log = []
    module.openai = _OpenAIModuleProxy(openai, lambda *args, **kwargs: _NarrationStubClient(log))
    try:
        yield log
    finally:
        module.openai = openai


@contextlib.contextmanager
def env_seed_injection(world_seed):
    """Make BoxingGym's OS-entropy-seeded PyMC calls reproducible for one run.

    See `ENV_SEED_INJECTION_RULE` for why this is necessary and exactly what it
    does. `pymc.draw` and `pymc.sample_prior_predictive` are reassigned on the
    `pymc` module itself - the same object `boxing_gym.envs.*` sees through its
    own `import pymc as pm` - for the lifetime of this context manager only;
    the originals are restored in `finally`, so nothing survives past one run
    and the upstream boxing-gym source is never touched on disk. Yields the
    list that the injected-seed log is appended to (empty for a task, such as
    death, whose environment never calls either function).
    """
    log = []
    counter = itertools.count()
    original_draw = pymc.draw
    original_sample_prior_predictive = pymc.sample_prior_predictive

    def _wrap(name, original):
        def wrapper(*args, random_seed=None, **kwargs):
            """Call `original`, injecting a deterministic per-call seed when none was given."""
            if random_seed is None:
                k = next(counter)
                random_seed = int(np.random.SeedSequence([world_seed, k]).generate_state(1)[0])
                log.append({"call": k, "function": name, "injected_seed": random_seed})
            return original(*args, random_seed=random_seed, **kwargs)

        return wrapper

    pymc.draw = _wrap("draw", original_draw)
    pymc.sample_prior_predictive = _wrap("sample_prior_predictive", original_sample_prior_predictive)
    try:
        yield log
    finally:
        pymc.draw = original_draw
        pymc.sample_prior_predictive = original_sample_prior_predictive


@dataclass
class ComponentState:
    """One OUTPUT COMPONENT's complete model state.

    A single-outcome task has exactly one of these and every field below means
    what the identically named local variable meant before multi-component
    tasks existed, which is why
    the protocol below is unchanged for it. A multi-component task
    (`TaskSpec.components`, the paper's task table, last row) has one per
    declared output: the components SHARE the design stream, the observation
    stream and the environment calls, and nothing else - each keeps its own
    active pool, its own archive, its own round history and its own frozen D0
    center, so the eight emotion runners are eight independent real runners over
    one experiment, not one eight-dimensional model.
    """

    index: int
    name: str
    fits: list = field(default_factory=list)
    archive: dict = field(default_factory=dict)
    hypotheses: dict = field(default_factory=dict)
    candidate_meta: dict = field(default_factory=dict)
    one_step: dict = field(default_factory=dict)
    born_round: dict = field(default_factory=dict)
    active_rounds: dict = field(default_factory=dict)
    evidence_stamp: dict = field(default_factory=dict)
    response_unit: float = 1.0
    response_center: float = 0.0
    pre_row: dict = None
    signature: object = None


# The noise-free nMSE (the paper's Table 6b: "the noise-free nMSE ... nMSE of the
# predictive mean vs the noise-free truth on the fixed evaluation set").
# death's noise-free true mean at a design point `xi`, from the environment's
# OWN deterministic hazard function and its FIXED true parameter - never a
# sampled draw, and never approximated. boxing_gym.envs.death_process.
# DeathProcess:
#   :284-286 `reset()` draws `self.theta` ONCE, a single `truncnorm(...).rvs()`
#     call at construction time - it is never redrawn, so it is the one true
#     parameter of the whole run (`TaskEnvironment.__init__` constructs the
#     environment - and so calls `reset()` - exactly once per run).
#   :322 `eta = 1 - np.exp(-self.theta * xi)` - the deterministic hazard.
#   :323 `infected_num = binom.rvs(self.N, eta)` - the NOISY observation
#     `step` returns; `self.N * eta` (below) is that Binomial's mean, i.e.
#     the noise-free truth, read directly off the SAME environment instance
#     `run()` already holds (`environment.env`), not re-derived.
def death_noise_free_truth(environment, times):
    """`N * (1 - exp(-theta * xi))` at every `xi` in `times` (death only)."""
    env = environment.env
    times = np.asarray(times, dtype=float)
    return float(env.N) * (1.0 - np.exp(-float(env.theta) * times))


# death's true PREDICTIVE PMF at a design point `xi` - the SAME fixed (env.N, env.theta)
# `death_noise_free_truth` above reads (see its own comment for the exact
# death_process.py source lines), carried to the full distribution over every
# count 0..N instead of collapsed to its mean. Used by `our_scores` below for
# `kl`/`expected_log_score` (`core_count.score_prediction`, imported
# unchanged); never for `nmse_mda`, which stays exactly what
# `death_noise_free_truth` already gave it.
def death_truth_pmf(environment, times):
    """`Binomial(N, 1 - exp(-theta * xi)).pmf(0..N)` at every `xi` in `times`.

    Shape `(len(times), N + 1)` - the same convention `count_predict`'s
    `bma_logpmf` (and this task's own `prediction["logpmf"]`) uses, so the
    two feed `core_count.score_prediction` directly, with no reshaping.
    """
    env = environment.env
    times = np.asarray(times, dtype=float)
    p = 1.0 - np.exp(-float(env.theta) * times)
    counts = np.arange(int(env.N) + 1)
    return binom.pmf(counts[None, :], int(env.N), p[:, None])


# The oracle candidate death's regime-switch tests use (`--truth-in-pool`,
# `--inject-oracle-at`): "the true function is inside the grammar but not in
# the initial pool".
# death's true probability is Binomial(N, 1 - exp(-theta*t))
# (death_process.py:322-323, see death_noise_free_truth above), whose logit is
# log(exp(theta*t) - 1) - already inside count_models.validate_expression's
# grammar (exp/log are both in CALL_ARITY) with no new PyMC code and no new
# link: `count_inference.build_model`/`count_models.particle_logpmf` interpret
# any validated expression already, and death's declared input range
# (`TASKS["death"]`, 0.001 <= t <= 2.0) keeps exp(c*t) - 1 positive for every c > 0.
# The coefficient's role MUST be "positive" (LogNormal(0,1),
# count_models.PRIOR_TEMPLATES) rather than whatever `mock_parameters`'s
# mechanical b-prefix rule would assign ("signed", Normal(0,1) - a negative
# draw sends log(exp(c*t)-1) to the log of a negative number, i.e. NaN).
# Declared here as a fixed constant instead: `make()` below special-cases this
# ONE expression string rather than generalizing `mock_parameters` itself, so
# every existing initial structure and P1 mock template (none of which is this
# expression) is completely unaffected.
DEATH_ORACLE_EXPRESSION = "log(exp(c*t) - 1)"
DEATH_ORACLE_ROLES = (("c", "positive"),)


def duplicate_archive_key(archive, t, key):
    """Collision-safe archive key for a same-round duplicate.

    Unchanged from the original key format when this is the first duplicate of `key` archived
    this round: `duplicate_t{t:02d}_{key}`, so a run with no same-round
    repeat of one model_id keeps byte-identical archive keys. A second (or
    third, ...) duplicate of the SAME model_id in the SAME round would
    otherwise collide with and silently overwrite the first; instead it gets
    a `_k{n}` suffix (n = 2, 3, ...), so every duplicate event keeps its own
    archive entry and the entry count equals the event count.
    """
    base = f"duplicate_t{t:02d}_{key}"
    if base not in archive:
        return base
    n = 2
    while f"{base}_k{n}" in archive:
        n += 1
    return f"{base}_k{n}"


def predictive_normalization_record(before, after):
    """What the normalization rule did during one predictive, or None.

    `before`/`after` are `count_models.normalization_event_counter()` snapshots
    taken around the work being attributed, exactly as `core_count` attributes
    the NaN -> -inf substitutions. None when the rule did nothing, and the caller
    then writes NO key at all: a run in which no row ever missed the tolerance
    keeps byte-identical output. `drift_tolerance` travels with every record so a
    reader can always tell which tolerance renormalized a row;
    `max_deviation_cumulative` is the counter's running maximum - the process
    maximum up to this checkpoint, EIG and prior-probe calls included - because a
    maximum is not a difference of two snapshots and so is never this
    predictive's alone (the name says so).
    """
    renormalized = after["renormalized_rows"] - before["renormalized_rows"]
    zeroed = after["zeroed_rows"] - before["zeroed_rows"]
    if not renormalized and not zeroed:
        return None
    return {
        "renormalized_rows": int(renormalized),
        "max_deviation_cumulative": float(after["max_deviation"]),
        "zeroed_rows": int(zeroed),
        "drift_tolerance": DRIFT_TOL,
    }


# One linear protocol; splitting it would hide the order of the round.

def build_count_proposer(args, out, *, narration_stubbed=False):
    """The structure proposer for `--proposer anthropic|openai` (None for mock).

    Local imports, exactly where CORE's core_count.run imports them: a test
    monkeypatches the count_proposer module attribute (not this one) and
    injects a client, never a real SDK client. When the task's narration stub
    is active (`narration_stubbed`), the client the proposer just built must be
    the real SDK client and never the narration stub - the stub is scoped to
    the environment module, and this guard refuses a run in which that
    separation failed before any proposal call is made.
    """
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
    else:
        return None
    if narration_stubbed and isinstance(getattr(proposer, "client", None), _NarrationStubClient):
        raise RuntimeError(
            "the structure proposer received the narration stub client: the narration "
            "stub must stay scoped to the environment module (see OPENAI_NARRATION_STUB_RULE)"
        )
    return proposer

def run(args):
    """Run one BoxingGym cell end to end from parsed CLI `args`; writes every record under `args.output`."""
    start = time.perf_counter()
    fit_timeout_seconds = validate_fit_timeout(getattr(args, "fit_timeout_seconds", None))
    fit_guards = getattr(args, "fit_guards", "off")
    if fit_guards not in ("off", "v1"):
        raise ValueError("fit_guards must be off or v1")
    d0_policy = getattr(args, "d0_policy", "fractions4")
    q_policy = getattr(args, "q_policy", "native")
    zero_count_unit = getattr(args, "d0_zero_count_unit", "refuse")
    if zero_count_unit not in D0_ZERO_COUNT_UNIT_CHOICES:
        raise ValueError("d0_zero_count_unit must be refuse or half_count")
    if zero_count_unit == "half_count" and getattr(args, "run_label", None) == "production":
        raise ValueError("--d0-zero-count-unit half_count is refused for a production label")
    # Absent unless typed (argparse.SUPPRESS), so an unstated flag
    # leaves every existing caller's namespace - and every existing cell -
    # exactly as it was. See KL_OPTIONAL_TASKS.
    kl_all_tasks = bool(getattr(args, "kl_all_tasks", False))
    if d0_policy not in ("fractions4", "random2") or q_policy not in ("native", "exclude_observed"):
        raise ValueError("unknown MDA design or evaluation policy")
    spec = TASKS[args.task]
    mask_names = bool(getattr(args, "mask_input_names", False))
    if mask_names and (args.truth_in_pool or args.inject_oracle_at is not None
                       or getattr(args, "run_label", None) == "production"):
        raise ValueError("--mask-input-names refuses production and oracle-expression options")
    if not spec.runnable:
        raise ValueError(f"task {spec.name} is specification-only: {spec.note}")
    design_policy = resolve_epig_policy(args.cell, getattr(args, "design_policy", None))
    if design_policy not in ("joint_eig", "epig", "point_disagreement", "space_filling", *EIG_PART_POLICIES):
        raise ValueError("unsupported BoxingGym design policy")
    if design_policy in EIG_PART_POLICIES and args.cell[2] != "1":
        # EIG-part arms: an EIG part replaces the joint EIG of an A=1 cell; random (A=0) and EPIG (A=2) keep theirs
        raise ValueError(f"design policy {design_policy} needs an A=1 cell (got {args.cell})")
    # The criteria record is for single-component tasks (the five unified-protocol tasks);
    # SUPPRESS, so an unstated option leaves the namespace, config and every record as before.
    record_criteria = bool(getattr(args, "record_design_criteria", False))
    if record_criteria and spec.components:
        raise ValueError("--record-design-criteria supports single-component tasks only")
    # `--epig-target-points` is argparse.SUPPRESS, so an unstated option leaves
    # the namespace, config and every record as before (32 targets). Stated on a non-EPIG
    # arm it would change nothing, so it is refused rather than recorded as if it acted.
    stated_epig_targets = getattr(args, "epig_target_points", None)
    if stated_epig_targets is not None and design_policy != "epig":
        raise ValueError("--epig-target-points applies only to the epig design policy")
    epig_target_count = EPIG_TARGET_POINTS if stated_epig_targets is None else stated_epig_targets
    if design_policy in ("point_disagreement", "space_filling"):
        check_policy(args.cell, design_policy)
    # Continuous design search: `--design-search ascent`, a search on a declared input box.
    # Under the unified protocol the same option is the design-optimization rule for
    # every active EIG-family arm - joint EIG, Model EIG, Parameter EIG and EPIG - ascending the
    # arm's own criterion on a declared box, or exhaustive (the pool argmax) where the pool is the
    # task's whole legal design set. Absent, nothing below changes; stated, it is refused here,
    # before any output, where it cannot apply (random and the two non-EIG policies; sampled pools).
    design_search = getattr(args, "design_search", None)
    design_search_mode = design_search_reason = design_search_objective = None
    if design_search is not None:
        if design_search != "ascent":
            raise ValueError("--design-search must be ascent")
        if args.cell[2] == "0" or design_policy not in DESIGN_SEARCH_OBJECTIVES:
            raise ValueError("--design-search ascent requires an active joint_eig, structure_eig, parameter_eig "
                             "or epig arm (cell third digit 1 or 2)")
        design_search_objective = DESIGN_SEARCH_OBJECTIVES[design_policy]
        design_search_mode, design_search_reason = bg_design_search.design_space_mode(spec)
    if args.proposer == "anthropic" and not args.allow_paid_api:
        raise ValueError("Anthropic proposals require explicit --allow-paid-api authorization")
    if args.proposer in ("openai", "openai_compat") and not args.allow_paid_api:
        raise ValueError("OpenAI proposals require explicit --allow-paid-api authorization")
    # --run-label. Resolved HERE,
    # before any output directory exists, so an inadmissible label leaves
    # nothing behind. `args.run_label` itself is NOT overwritten: it stays the
    # caller's raw request (None when the flag was not typed), because the
    # label byte-invariance rule keys off "was a label actually stated"
    # and because `**vars(args)` should record what was asked for.
    label_metadata = compat_config(args)
    stated_run_label = getattr(args, "run_label", None)
    # The system prompt is SELECTED here, before the receipt, so the label binds
    # the text this run actually sends instead of the module default - the numbered
    # `--prompt-version` or, on a development run, the `--prompt-variant` file (applied by
    # `select_run_prompt` itself, so the receipt pair and the config.json pair below are one
    # pair on every path). `--priors llm` is applied further down (with the prior template,
    # as the prior ablations require), so its decision is stated to `select_run_prompt` rather than read off
    # a switch that is not set yet. `--prompt-version` and `--prompt-variant` are refused
    # together there. The two keys ride a receipt-only namespace, exactly as
    # `active_maximum` does, so `compat_config(args)` keeps the key set every earlier run
    # recorded and `**prompt_identity()` in `config` below recomputes the same pair from
    # the same source.
    _, label_prompt = select_run_prompt(args, llm_priors=args.priors == "llm")
    # Candidates asked for per round, resolved here because a
    # k > 1 run sends a COMPOSED system text (the selected file's bytes plus one
    # paragraph) and the receipt binds that text's digest beside the file's own.
    # k = 1 composes nothing, declares nothing, and leaves every existing
    # receipt byte-identical.
    proposal_limit = validate_proposal_limit(int(getattr(args, "proposal_limit", 1)))
    # Under the unified protocol, the k = 3 request composes COMPOSITION_PARAGRAPH v2
    # (k = 2 keeps v1); the paragraph is a module switch shared with ChemBench, so every
    # run selects it here.
    set_composition_paragraph("v2" if proposal_limit == 3 else "v1")
    if proposal_limit != 1:
        label_prompt = {**label_prompt, "composed_prompt_sha256":
                        composed_prompt_identity(proposal_limit)["composed_prompt_sha256"]}
    # The initial burst (K candidates asked for at round 0, in calls of at
    # most six) and the active-pool cap. Both are absent by default, so a run that
    # states neither keeps every receipt and config byte; a burst binds the digest
    # of the composed burst request beside the file's own pair.
    initial_proposals = int(getattr(args, "initial_proposals", 0))
    active_maximum = int(getattr(args, "active_maximum", ACTIVE_MAXIMUM))
    burst_concurrency = int(getattr(args, "burst_concurrency", DEFAULT_BURST_CONCURRENCY))
    initial_catalogue = getattr(args, "initial_catalogue", "on")
    parallel_proposals = int(getattr(args, "parallel_proposals", 0))
    if initial_proposals or parallel_proposals:
        label_prompt = {**label_prompt, "burst_prompt_sha256":
                        composed_system_prompt(burst=True)[1]["burst_prompt_sha256"]}
    # A round-0 burst's refill waves send their own composed request; the receipt
    # binds its digest and the refill paragraph's, and config and the burst pool file
    # record these with the prompt file's own. The receipt also binds the refill rule
    # beside them.
    refill_identity = burst_refill_identity() if initial_proposals else {}
    if initial_proposals:
        label_prompt = {**label_prompt, **{key: refill_identity[key] for key in (
            "burst_refill_prompt_sha256", "burst_refill_paragraph_sha256")},
            "initial_burst_refill_policy": INITIAL_BURST_REFILL_POLICY}
    # Archive review rounds (`--archive-review-rounds`, SUPPRESS). The
    # receipt binds the k = 3 request's composed_prompt_sha256 as before; the review paragraph's
    # digests are recorded in config.json and in every review round's transport record.
    review_rounds = parse_round_list(getattr(args, "archive_review_rounds", ""), (), maximum=args.rounds)
    if review_rounds and proposal_limit != 3:
        raise ValueError("--archive-review-rounds needs --proposal-limit 3")
    label_args = argparse.Namespace(**{**vars(args), **label_prompt})
    label_receipt = bmd_label_receipt("boxinggym", label_args, task=args.task, metadata=label_metadata)
    run_label = label_receipt["run_label"]
    if mask_names and run_label == "production":
        raise ValueError("--mask-input-names is refused for a production label")
    # Registered decoding: a production run states a decoding option only if the manifest
    # registered the same value - here, before any output exists.
    require_registered_decoding(label_metadata, label_receipt, "boxinggym")
    label_explicit = stated_run_label is not None or getattr(args, "manifest_ref", None) is not None
    # The death-oracle switches are declared for
    # ONE task - DEATH_ORACLE_EXPRESSION is death's own true logit and no other
    # task has a registered oracle - so both are refused BEFORE any output
    # directory exists, the same fail-fast contract as the two checks above,
    # rather than silently doing nothing on a task with no oracle.
    if (args.truth_in_pool or args.inject_oracle_at is not None) and spec.name != "death":
        raise ValueError(
            "--truth-in-pool/--inject-oracle-at are declared for the death oracle only"
            f"; task {spec.name!r} has no registered oracle expression"
        )
    # The verified label, not the compute budget or provider,
    # gates development prompt variants, as in ChemBench. Refuse before output.
    if args.prompt_variant is not None and run_label == "production":
        # The variant was applied by `select_run_prompt` so the receipt could bind it; a
        # refused selection leaves no override behind, exactly as a refused tag does.
        set_prompt_variant(None)
        raise ValueError(
            "--prompt-variant is refused whenever this run's label would be production"
        )
    # The half-count unit is a development option, gated by the verified label like
    # the prompt variant and the name mask; refused before any output exists.
    if zero_count_unit == "half_count" and run_label == "production":
        raise ValueError("--d0-zero-count-unit half_count is refused for a production label")
    name_mask = (InputNameMask(spec, Path(args.output) / "masked_model_ids.jsonl")
                 if mask_names else None)
    if name_mask is not None:
        spec = name_mask.spec
    interface = spec.interface()
    # `exhaustive` = the outcome has a finite support the count runner enumerates
    # (bounded count 0..N, and the binary outcome over {0,1} with one
    # trial); `real_outcome` = the identity-link families of `real_inference`;
    # everything else is the log-link scalar path.
    exhaustive = interface["outcome_type"] in EXHAUSTIVE_COUNT_OUTCOMES
    real_outcome = interface["outcome_type"] == "real"
    # Prior-template ablation: REFUSED on the scalar/real
    # paths, before `set_prior_template` runs and before ANY output directory
    # exists (the `out.mkdir` below), rather than silently building a
    # partially-templated record. `boxinggym_core.SCALAR_NOISE_PRIORS` folds in
    # `real_inference.REAL_NOISE_PRIORS`'s import-time copies of
    # `PRIOR_RULES["intercept"]`/`["signed"]`, and `positive_inference.
    # ScalarHypothesis.as_dict`'s record block restates the role priors as
    # literals - neither follows a non-default template. That leak is NOT
    # fixed here (`positive_inference.py`/`real_inference.py` are untouched);
    # this refusal seals it off instead. The exhaustive count/binary path is
    # unaffected: it reads `PRIOR_RULES` live with no such copy.
    if args.prior_template != DEFAULT_PRIOR_TEMPLATE and not exhaustive:
        raise ValueError(
            "the proposed-prior ablation is defined on the count path; scalar/real noise priors do "
            "not follow the template"
        )
    # Prior-template and proposed-prior ablations. Selected ONCE, before any
    # hypothesis, packet, probe or fit exists - see `core_count.run` for why.
    set_prior_template(args.prior_template)
    # The validation-message wording, selected once with the other engine switches;
    # unstated is v1, the sentence every run has produced.
    set_validation_messages(getattr(args, "validation_messages", DEFAULT_VALIDATION_MESSAGES))
    # The same discipline for the expression grammar - one
    # selection, before any hypothesis, with this task's own interface so the v3
    # domain proof reads its declared input ranges. `getattr` for the same
    # reason every other late-added switch uses it: an args namespace built
    # before this option existed still means the default gate.
    grammar_version = getattr(args, "grammar_version", DEFAULT_GRAMMAR_VERSION)
    set_grammar_version(grammar_version, interface)
    # The proposed-prior ablation is REFUSED on the scalar/real paths rather than silently dropped: a
    # candidate this runner admits there is CONVERTED from `count_models.
    # Hypothesis` to `ScalarHypothesis`/`RealHypothesis` (the `not exhaustive`
    # branch further down), and neither class carries `coefficient_priors`, so an
    # accepted proposed prior would vanish between the reply and the fit while
    # `model_id` still claimed it. Only the exhaustive count/binary path fits a
    # `count_models.Hypothesis` end to end.
    if args.priors == "llm" and not exhaustive:
        raise ValueError(
            "--priors llm is implemented for the exhaustive count/binary path only; "
            f"task {args.task!r} has outcome type {interface['outcome_type']!r}"
        )
    set_llm_priors(args.priors == "llm")
    # Opt-in stable log(1 - exp(z)); set once per run like the two switches above.
    # Absent (SUPPRESS) means off, which leaves every path and record untouched.
    set_stable_log1mexp(getattr(args, "stable_log1mexp", False))
    # The prompt variant is selected ONCE, before any prompt is read -
    # same rule as set_prior_template/set_llm_priors above - and that
    # happens inside `select_run_prompt` at the top of this function, before the receipt,
    # so the label binds it. None (every default and every production-labelled run,
    # enforced by the refusal above) makes `count_proposer.system_prompt_bytes`/
    # `prompt_identity` read exactly what they read without a variant.
    initial_family, allowed = interface_families(
        interface["outcome_type"],
        args.cell[1] == "1",
        initial_family=args.initial_family,
        zero_inflation=spec.zero_inflation_admissible,
    )
    structures = spec.initial_structures()
    # `--truth-in-pool` ADDS the oracle to the initial four structures - it never
    # replaces them. Changing `structures` here, before `cell_constraints`
    # below, is the one place that keeps every consumer of the initial pool in
    # sync: the initial pool itself (`state.fits` below), the run record
    # (`config["initial_structures"]`), the belief packet's own
    # `constraints["initial_structures"]`, the S0 catalog and the structures
    # handed to a real proposer all read `names`/`constraints`, never
    # `spec.initial_structures()` again.
    if args.truth_in_pool:
        structures = (*structures, DEATH_ORACLE_EXPRESSION)
    # `()` for every task that declares no categorical input, which keeps
    # the `cell_constraints` call, the packet bytes and every `Hypothesis` below
    # identical to what they were before categorical inputs existed, for CORE and
    # the six original BoxingGym tasks.
    categorical = spec.categorical_levels()
    if categorical and not exhaustive:
        # The per-level offset gather is wired through the exhaustive count/
        # binary path only (count_models.particle_logpmf and
        # count_inference.build_model). A categorical task on the scalar or
        # real path would silently drop the index, so it is refused here.
        raise ValueError(
            f"task {spec.name}: categorical inputs are implemented for the "
            f"exhaustive count/binary path only, not {interface['outcome_type']}"
        )
    constraints = cell_constraints(
        args.cell,
        initial_family,
        allowed_families=allowed,
        initial_structures=structures,
        categorical_inputs=categorical,
    )
    names = tuple(constraints["initial_structures"])
    namespace = tuple(item["name"] for item in interface["inputs"]) + tuple(
        interface["derived_inputs"]
    )
    # One component unless the task declares several. `("y",)` reproduces
    # every single-outcome artifact name and score-row shape exactly.
    components = tuple(spec.components) or ("y",)
    multi = len(components) > 1
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for child in ("fits", "predictions", "acquisition", "proposals", "packets"):
        (out / child).mkdir()
    if review_rounds:
        (out / "revival").mkdir()
    particles, q, grid_size, eig_samples = (
        (64, 32, 16, 32) if args.profile == "dev" else (500, 256, 100, 128)
    )
    # A goal that builds its held-out set ONCE at construction caps |Q|
    # (irt: exactly ten points, irt.py:14-:36); asking for more raises inside
    # the environment. `None` (every other task) leaves `q` untouched.
    if spec.eval_point_limit is not None:
        q = min(q, int(spec.eval_point_limit))
    settings = SMCSettings(particles, 2, 2)
    world_seed = seed_for("boxinggym_world", spec.name, args.seed)
    evaluation_seed = seed_for("boxinggym_evaluation", spec.name, args.seed, args.profile)
    design_pool_seed = seed_for("boxinggym_design_pool", spec.name, args.seed)

    def rng_identity(hypothesis):
        """The model id that seeds this hypothesis's RNG streams (the native id under name masking)."""
        return (hypothesis.model_id if name_mask is None
                else name_mask.rng_model_id(hypothesis))

    def run_prior_probe(hypothesis, *args, **kwargs):
        """`prior_probe` with the name-masked run's RNG identity filled in."""
        if name_mask is not None:
            kwargs["rng_model_id"] = rng_identity(hypothesis)
        return prior_probe(hypothesis, *args, **kwargs)

    def probe_particles(hypothesis, count):
        """`prior_particles` with the name-masked run's RNG identity filled in."""
        options = {} if name_mask is None else {"rng_model_id": rng_identity(hypothesis)}
        return prior_particles(hypothesis, count, **options)

    def make(expression, family):
        """Build this task's hypothesis class for `expression` and `family` with mock parameter roles."""
        # The ONE expression that needs a
        # role mock_parameters's mechanical b-prefix rule cannot produce (see
        # DEATH_ORACLE_ROLES's own comment). Every one of the four initial
        # structures and twenty P1 mock templates takes the unchanged
        # `mock_parameters` path - none of them is this exact string.
        parameters = (
            DEATH_ORACLE_ROLES
            if spec.name == "death" and expression == DEATH_ORACLE_EXPRESSION
            else mock_parameters(expression)
        )
        if exhaustive:
            return Hypothesis(expression, family, parameters, namespace, categorical)
        if real_outcome:
            scale = (
                default_scale_expression(interface)
                if family == "heteroscedastic_normal"
                else None
            )
            return RealHypothesis(expression, family, parameters, namespace, scale)
        return ScalarHypothesis(expression, family, parameters, namespace)

    with contextlib.ExitStack() as stack:
        # Installed BEFORE the environment is constructed, because
        # moral_machines/emotion call `openai.OpenAI()` in `__init__`. Empty
        # list for every other task, and the context manager is not entered at
        # all. See OPENAI_NARRATION_STUB_RULE.
        narration_calls = (
            stack.enter_context(openai_narration_stub(spec.openai_narration_module))
            if spec.openai_narration else []
        )
        # `timings` is created
        # HERE - before `evaluation_points` - instead of further down (earlier
        # versions created it just before the `public, failures, score_rows = []`
        # line, which is AFTER evaluation_points already ran, so that call's
        # cost fell into no bucket at all, and a
        # paper-profile |Q|=256 evaluation-point generation can itself cost
        # tens of minutes). Every other bucket (fit/proposal/eig/forward/
        # scoring) is accumulated exactly where it was before; only this dict's
        # birthplace and the one new "setup_s" key are new. `wall_seconds` is
        # unaffected either way - it is `time.perf_counter() - start` with
        # `start` already captured at the top of `run()`.
        timings = {
            "setup_s": 0.0,
            "fit": 0.0,
            "proposal": 0.0,
            "eig": 0.0,
            "forward": 0.0,
            "scoring": 0.0,
        }
        env_seed_log = stack.enter_context(env_seed_injection(world_seed))
        environment = TaskEnvironment(spec, world_seed)
        tic = time.perf_counter()
        evaluation_x, heldout = environment.evaluation_points(q, evaluation_seed)
        timings["setup_s"] += time.perf_counter() - tic
        # death's nmse_mda normalizer - fixed for the whole run
        # (theta is drawn once, evaluation_x is fixed), so computed once here
        # rather than per checkpoint. See death_noise_free_truth's own
        # docstring/comment for the exact source (death_process.py:284-286,
        # :322-323) - `None` for every other task (never referenced there).
        # `death_true_pmf` alongside them, same
        # fixed-for-the-whole-run rule - the full Binomial PMF (not just its
        # mean) `our_scores` reads for `kl`/`expected_log_score` below.
        death_true_mean = death_var_q = death_true_pmf = None
        if spec.name == "death":
            death_true_mean = death_noise_free_truth(environment, evaluation_x)
            death_var_q = float(np.var(death_true_mean, ddof=0))
            death_true_pmf = death_truth_pmf(environment, evaluation_x)
        # The same fixed-for-the-whole-run rule, generalized to
        # every task - the generating OBSERVATION LAW at each point of Q, built
        # once here from the environment's own drawn parameters (never from a
        # fit, never from the benchmark's own code, which stays read-only).
        # death's own `death_true_pmf` path above is untouched and still what
        # `our_scores` scores death with; `truth_law_q` is what every OTHER
        # task's `kl` is scored against. `None` where the world's parameters
        # were not captured (irt), which leaves `kl` blank exactly as
        # `truth_mean`'s own `None` already leaves `nmse_mda` undefined.
        # `KL_OPTIONAL_TASKS` is the measured cost fallback and nothing else:
        # for every task outside it the flag is irrelevant.
        # The flag gates the SCORING,
        # not the law. Building the law is one closed-form array per run, so it
        # is built whenever the task declares one - which is what makes
        # `kl_truth_law_available` a statement about the TASK (is there a true
        # observation law to score against at all?) and `kl_recorded` the
        # separate statement about THIS run (was it scored?).
        record_kl = kl_all_tasks or spec.name not in KL_OPTIONAL_TASKS
        # Every law builder reads the world's own
        # drawn parameters through `environment.env`. A world that carries no
        # underlying environment object (a synthetic stand-in, as the runner's
        # own tests use) has no true law to score against, so it records "no
        # law" exactly as irt does - decided once here, never inside a builder.
        truth_law_q = (
            None if spec.truth_law is None or getattr(environment, "env", None) is None
            else spec.truth_law(spec, environment, evaluation_x)
        )
        kl_recorded = record_kl and truth_law_q is not None
        pool_target = design_pool_target(spec, grid_size, args.rounds)
        if spec.environment_pool_source == "sampled":
            # The environment's own generator draws from the global numpy
            # stream, so the pool gets its own dedicated, recorded seed rather
            # than whatever the world/evaluation draws happen to have left.
            np.random.seed(design_pool_seed)
        pool = build_design_pool(spec, environment, pool_target)
        candidate_metadata = candidate_set_record(pool.rows, "boxinggym")
        epig_targets = None
        if design_policy == "epig":
            target_seed = seed_for(EPIG_TARGETS_SEED_STREAM, args.seed)
            epig_targets = sample_epig_targets(spec, environment, pool, target_seed, count=epig_target_count)
        refused = environment.refused_designs()
        single_input = len(spec.input_names) == 1

        def design_value(design):
            """A design as it is written into a record: a float, or a list of floats."""
            if single_input:
                return float(design)
            return [float(value) for value in np.atleast_1d(np.asarray(design, dtype=float))]

        def observation_row(row, handle):
            """One public observation's inputs: a bare float for a single input, else the row plus its handle."""
            if single_input:
                return {"x": float(np.atleast_1d(np.asarray(row, dtype=float))[0])}
            return {"x": design_value(row), "design": handle}

        def component_y(row, state):
            """One public observation's outcome for ONE component."""
            return float(row["y"][state.index]) if multi else float(row["y"])

        def component_heldout(state):
            """The held-out outcomes for ONE component."""
            return heldout[:, state.index] if multi else heldout

        # Continuous design search: the declared box, integer axes and env-validity check, built
        # only when stated (and only on a box task; an exhaustive task searches its pool).
        search_space = (bg_design_search.SearchSpace(spec, environment, encode_design)
                        if design_search_mode == "ascent" else None)
        d0_designs = select_d0_designs(spec, pool, d0_policy, args.seed)
        observed_pool_indices = set()
        if spec.design_exclusion == "observed":
            # D0 itself must be duplicate-free designs, same as every
            # later round - checked here, not assumed, because a future
            # `design_exclusion="observed"` task's pool could be small enough
            # for the fraction rule (`d0_indices`) to collide.
            observed_pool_indices = {pool.index_of(row) for row, _ in d0_designs}
            if len(observed_pool_indices) != len(d0_designs):
                raise ValueError(f"{spec.name}: D0 designs are not distinct")
        templates = spec.mock_templates()
        # How many mock schedule entries a round-0 burst consumed, so a k > 1 mock
        # round continues the schedule after them (k = 1 keeps its (t - 1) entry as always).
        mock_schedule = {"burst_used": 0}
        families = tuple(allowed[(index + 1) % len(allowed)] for index in range(len(templates)))
        proposer = None
        # Same as CORE's core_count.run - the proposer,
        # and with it the SDK client constructor, is built INSIDE a recording
        # try. Before this, a constructor error echoing the credential
        # propagated verbatim out of run() and no failure.json was written at
        # all; now the same masked failure record
        # is written and the exception is re-raised masked. Only a constructor
        # failure is caught here; the numerical path is untouched.
        try:
            proposer = build_count_proposer(args, out, narration_stubbed=spec.openai_narration)
        except BaseException as exc:  # noqa: BLE001 - recorded masked, re-raised masked
            write_json(
                out / "failure.json",
                {
                    "status": "failed",
                    "stage": "proposer_construction",
                    "error": mask_secrets(f"{type(exc).__name__}: {exc}"),
                    "traceback": mask_secrets(traceback.format_exc()),
                    "timing_seconds": {},
                },
            )
            raise masked_exception(exc) from None
        config = {
            **compat_config(args),
            "model_id_schema": MODEL_ID_SCHEMA,
            "notation_normal_form": NOTATION_NORMAL_FORM,
            "output": str(out),
            "suite": "boxinggym",
            "task_spec": {
                "name": spec.name,
                "module": spec.module,
                "env_class": spec.env_class,
                "goal_class": spec.goal_class,
                "env_source_sha256": hashlib.sha256(
                    Path(__import__(spec.module, fromlist=["_"]).__file__).read_bytes()
                ).hexdigest(),
                "prediction_format": spec.prediction_format,
                "env_metric": spec.env_metric,
                "env_metric_direction": spec.env_metric_direction,
                "d0_rule": D0_RULE if len(spec.input_names) == 1 else D0_RULE_MULTI_INPUT,
                "design_grid_rule": design_grid_rule(spec),
                "design_encoder": spec.design_encoder,
                "environment_fixed_pool": spec.environment_fixed_pool,
                "zero_inflation_admissible": spec.zero_inflation_admissible,
                "design_exclusion": spec.design_exclusion,
                "eval_input_index": list(spec.eval_input_index),
                "mock_template_source": MOCK_TEMPLATE_SOURCE,
                "multi_input_template_rule": MULTI_INPUT_TEMPLATE_RULE,
                "multi_input_structure_rule": MULTI_INPUT_STRUCTURE_RULE,
                # ---- categorical-input and multi-output keys ----
                "additive_structure_cap_rule": ADDITIVE_STRUCTURE_CAP_RULE,
                "categorical_inputs": {name: int(levels) for name, levels in categorical},
                "indexed_coefficient_rule": INDEXED_COEFFICIENT_RULE if categorical else "",
                "environment_pool_source": spec.environment_pool_source,
                "feature_map": spec.feature_map,
                "observation_source": spec.observation_source,
                "outcome_offset": float(spec.outcome_offset),
                "eval_point_limit": spec.eval_point_limit,
                "output_components": list(components),
                "composite_score_rule": COMPOSITE_SCORE_RULE if multi else "",
                "multi_component_eig_rule": MULTI_COMPONENT_EIG_RULE if multi else "",
                "multi_component_calibration_rule": (
                    MULTI_COMPONENT_CALIBRATION_RULE if multi else ""
                ),
                "declared_forms_source": (
                    "declared_in_the_task_registry" if spec.initial_forms else "retargeted"
                ),
                # irt's refused cells: the environment answers 36 - k designs.
                # `k` is the number of
                # DISTINCT refused designs; the environment's own list may
                # repeat a cell because the goal samples with replacement.
                "environment_refused_attribute": spec.environment_refused_attribute,
                "environment_refused_distinct": len(refused),
                "environment_refused_designs": sorted(list(d) for d in refused),
                "openai_narration_stub": spec.openai_narration,
                "openai_narration_stub_rule": (
                    OPENAI_NARRATION_STUB_RULE if spec.openai_narration else ""
                ),
                "eval_covariate_note": (
                    EMOTION_WIN_UNIT_DEFECT if spec.name == "emotion" else ""
                ),
                # Declared unconditionally, like
                # the other *_RULE strings above, whether or not this run's
                # pool is ever exhausted.
                "design_pool_exhaustion_policy": DESIGN_POOL_EXHAUSTION_RULE,
            },
            "interface": interface,
            "initial_structures": list(names),
            "input_namespace": list(namespace),
            "design_pool_size": len(pool.rows),
            "smc": asdict(settings),
            "q": q,
            "design_count": pool_target,
            "eig_posterior_samples": eig_samples,
            "initial_pool": len(names),
            "d0_designs": [design_value(row) for row, _ in d0_designs],
            "d0_policy": d0_policy,
            "d0_count": len(d0_designs),
            "d0_seed_label": "boxinggym_d0" if d0_policy == "random2" else None,
            "q_policy": q_policy,
            "q_excluded_count": 0,
            "truth_mean_available": spec.truth_mean is not None,
            # `kl_truth_law_available` is about the
            # TASK - does a true observation law exist to score `kl` against?
            # False records the one case that leaves the column blank (irt,
            # whose latent draw the environment discards and the adapter
            # therefore may fail to capture). `kl_recorded` is about THIS RUN:
            # a law was available AND the task is either unconditional or the
            # flag was given. The two differ exactly when an optional task ran
            # without `--kl-all-tasks`, which used to read as "no law exists".
            "kl_truth_law_available": truth_law_q is not None,
            "kl_recorded": kl_recorded,
            "kl_all_tasks": kl_all_tasks,
            "kl_optional_tasks": list(KL_OPTIONAL_TASKS),
            "active_maximum": active_maximum,
            # Present only when a burst was asked for.
            **({"initial_proposals": initial_proposals, "burst_concurrency": burst_concurrency,
                "initial_catalogue": initial_catalogue, "parallel_proposals": parallel_proposals,
                "burst_prompt_sha256": label_prompt["burst_prompt_sha256"]}
               if initial_proposals or parallel_proposals else {}),
            # Present only when a round-0 burst runs.
            **({**refill_identity, "initial_burst_refill_policy": INITIAL_BURST_REFILL_POLICY}
               if initial_proposals else {}),
            "proposal_policy": "P1",
            "proposal_rounds": list(range(1, args.rounds + 1)),
            "proposal_limit": proposal_limit,
            **({"archive_review_rounds": list(review_rounds),
                "archive_review_label": archive_reevaluation_label(review_rounds),
                "archive_review_rule": ARCHIVE_REVIEW_RULE, "archive_revive_limit": ARCHIVE_REVIVE_LIMIT,
                **{key: value for key, value in composed_prompt_identity(proposal_limit, archive_review=True).items()
                   if key.startswith("archive_review_")}} if review_rounds else {}),
            "repair_attempts": 1,
            # The composed request's digest, beside (never
            # instead of) the packaged file's own. At k = 1 the composed text
            # IS the file's text and no key is added.
            **{k: v for k, v in composed_prompt_identity(proposal_limit).items()
               if k in ("composed_prompt_sha256", "composition_paragraph_version",
                        "composition_paragraph_sha256")},
            "belief_packet": "v2",
            "eval_question_policy": "restored question",
            "eval_question_note": (
                "the numeric goal.eval_points path is used, exactly as the PPL path does, so the "
                "upstream get_goal_eval_question text defect cannot "
                "affect this run; no question string is read anywhere in this module"
            ),
            "world_seed": world_seed,
            "evaluation_seed": evaluation_seed,
            "world_parameters": environment.world_parameters(),
            "env_seed_injection": True,
            "env_seed_injection_rule": ENV_SEED_INJECTION_RULE,
            "evaluation_use": "scoring_only_never_fit_choose_or_propose",
            **prompt_identity(),
            "prior_probe": {
                "samples": PRIOR_PROBE_SAMPLES,
                "grid_points": PRIOR_PROBE_GRID_POINTS,
                "seed_rule": "dedicated_generator_seeded_by_model_id_only",
            },
            "model_prior": "uniform_distinct_active_pairs",
            "evidence": "full_public_dataset_under_original_fixed_prior_each_fit",
            "proposer": "mock_fixed_template_responses_NOT_LLM" if proposer is None else args.proposer,
            "paid_api_calls": 0 if proposer is None else None,
            "source_files_sha256": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in (
                    "boxinggym_core.py",
                    "core_count.py",
                    "count_inference.py",
                    "count_models.py",
                    "model_identity.py",
                    "positive_inference.py",
                    "real_inference.py",
                    "belief_packet.py",
                    "count_proposer.py",
                )
            },
            "created_utc": datetime.now(UTC).isoformat(),
            "python": sys.executable,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "pymc", "pytensor")
            },
        }
        # Prior-template ablation, as in `core_count.run`: the two flag names already
        # travel through `**vars(args)`; the TABLE and `prior_source` are recorded
        # only when a switch is on, so a default run's config.json differs from the
        # anchor by new CLI keys, the self-hash and timing only.
        if args.prior_template != DEFAULT_PRIOR_TEMPLATE or args.priors == "llm":
            config["prior_hyperparameters"] = prior_template_table()
            config["prior_source"] = "llm" if args.priors == "llm" else "role_template"
        if spec.environment_pool_source == "sampled":
            config["design_pool_seed"] = design_pool_seed
            config["design_pool_draws"] = spec.pool_sample_count
        # Explicit labels are bound into the config; omitted legacy config key sets stay unchanged.
        if label_explicit or args.proposer in ("openai", "openai_compat"):
            config["label"] = run_label
        if label_explicit:
            config["label_receipt"] = label_receipt
        if design_search is not None:
            config["design_search_settings"] = {
                **bg_design_search.declared_settings(),
                # Which side of the design-search rule this task takes, and which criterion this arm ascends.
                "mode": design_search_mode,
                "objective": design_search_objective,
                **({"mode_reason": design_search_reason} if design_search_reason else {}),
                "bounds": search_space.record() if search_space is not None else None,
                "constants": bg_design_search.design_search_constants(),
            }
        if fit_guards == "v1":
            config["fit_guards"] = fit_guards
        else:
            config.pop("fit_guards", None)
        if fit_timeout_seconds is not None:
            config["fit_timeout_seconds"] = fit_timeout_seconds
        else:
            config.pop("fit_timeout_seconds", None)
        if design_policy == "epig":
            finite_targets = (spec.environment_fixed_pool
                              or spec.environment_pool_source == "categorical")
            target_measure = (BOXING_FINITE_TARGET_MEASURE if finite_targets
                              else BOXING_GOAL_TARGET_MEASURE)
            config.update(epig_config(
                epig_targets, target_seed, target_measure, evaluation_points=evaluation_x,
                target_measure_note=BOXING_FINITE_TARGET_NOTE if finite_targets else None,
                target_count=epig_target_count))
            config["task_spec"].pop("multi_component_eig_rule", None)
            config["task_spec"]["multi_component_epig_rule"] = (
                "sum_of_independent_component_predictive_information" if multi else "")
        if design_policy in ("point_disagreement", "space_filling"):
            config["task_spec"].pop("multi_component_eig_rule", None)
            if design_policy == "point_disagreement" and multi:
                config["task_spec"]["multi_component_point_disagreement_rule"] = (
                    "sum_of_component_point_disagreements")
            config["design_policy"] = design_policy
            config["design_policy_target"] = design_policy
            config["design_policy_definition"] = (
                SPACE_FILLING_DEFINITION if design_policy == "space_filling"
                else POINT_DISAGREEMENT_DEFINITION)
            config["source_files_sha256"]["design_policies.py"] = hashlib.sha256(
                Path(__file__).with_name("design_policies.py").read_bytes()).hexdigest()
            if design_policy == "space_filling":
                config["design_normalization_bounds"] = [item["range"] for item in interface["inputs"]]
            else:
                config["point_disagreement_posterior_particles"] = "all_retained"
        if design_policy in EIG_PART_POLICIES:
            # EIG-part arms: recorded only for the two opt-in arms; the joint arm's config is unchanged
            column, target, _ = EIG_PART_POLICIES[design_policy]
            config["design_policy"] = design_policy
            config["design_policy_target"] = target
            config["design_policy_objective_column"] = column
            config["design_policy_definition"] = EIG_PART_DEFINITION[design_policy]
            config["design_policy_estimator"] = EIG_PART_ESTIMATOR["exhaustive" if exhaustive else "scalar"]
            if multi:
                config["task_spec"]["multi_component_eig_part_rule"] = f"sum_of_the_per_component_{column}_tables"
        if record_criteria:
            # Stated only when the option is on; the joint arm's config is unchanged otherwise
            config["design_criteria_record"] = {"file": DESIGN_CRITERIA_FILE, "eig_posterior_samples": eig_samples,
                                                **DESIGN_CRITERIA_DEFINITIONS}
        if name_mask is not None:
            config.update(mask_input_names=True, input_name_map=name_mask.input_name_map)
        write_json(out / "config.json", config)

        np.savez_compressed(
            out / "evaluation_scoring_only.npz", x=evaluation_x, y=heldout, designs=pool.rows
        )
        # The environment evaluator and our scorer must use the SAME held-out points.
        stored = np.asarray(
            [environment.eval_inputs(p) for p in environment.goal.eval_points], dtype=float
        )
        if len(spec.input_names) == 1:
            stored = stored[:, 0]
        if stored.shape != evaluation_x.shape or not np.array_equal(stored, evaluation_x):
            raise ValueError("environment eval points and our held-out inputs disagree")

        public, failures, score_rows = [], [], []
        mda_rows = []
        # Normalization-rule run totals over the checkpoints' predictives; reaches
        # summary.json only if some checkpoint recorded an event (`checkpoints`).
        predictive_normalization_total = {
            "renormalized_rows": 0,
            "max_deviation_cumulative": 0.0,
            "zeroed_rows": 0,
            "checkpoints": 0,
        }
        states = [ComponentState(index, name) for index, name in enumerate(components)]
        epig_path_states = ({state.name: EpigDiscretePathState() for state in states}
                            if design_policy == "epig" else {})
        # `timings` was created earlier (see above) - not here.
        accounting = {
            "llm_calls": 0,
            "llm_repairs": 0,
            "proposal_rounds_executed": 0,
            "catalog_exhausted": 0,
            "valid_candidates": 0,
            "duplicates": 0,
            "rejected_candidates": 0,
            "prior_probe_rejections": 0,
            "fit_attempts": 0,
            "fit_successes": 0,
            "reevaluation_refits": 0,
            "revival_refits": 0,
            # Failure classes: core_count.failure_class_counts over every transport and outcome record.
            "failure_class_counts": dict.fromkeys(FAILURE_CLASSES, 0),
        }
        if initial_proposals or parallel_proposals:
            accounting.update(burst_calls=0, burst_repairs=0, burst_failed_calls=0, burst_candidates=0,
                              burst_duplicates=0, burst_refill_calls=0, burst_admitted=0,
                              burst_max_in_flight=0)
            accounting.update({key: 0 for key in PWC2_WAVE_COUNT_KEYS if key != "duplicates"})
        if initial_proposals:
            # The round-0 burst waves executed (1..3 per component).
            accounting["burst_wave_count"] = 0
        admissions = 0
        catalog_exhausted_rounds = []
        # Same shape as
        # catalog_exhausted_rounds above - a round list, always declared,
        # empty unless `--inject-oracle-at` is actually exercised.
        # injected_rounds is the round the injected candidate was NOT already
        # known (i.e. it reached the normal register/probe/fit gates as a
        # genuinely new candidate, whatever its terminal status);
        # injection_ignored_rounds is the round it WAS already known (active
        # or retired) and so fell to the existing duplicate_not_refitted path
        # untouched ("ignore and record").
        injected_rounds = []
        injection_ignored_rounds = []

        def artifact_stem(state, prefix):
            """`prefix` for a single-outcome task, `prefix_<component>` otherwise."""
            return prefix if not multi else f"{prefix}_{state.name}"

        def register(state, hypothesis, t):
            """Register `hypothesis` in the component's history tables; returns its model id."""
            key = hypothesis.model_id
            state.hypotheses.setdefault(key, hypothesis)
            if key not in state.one_step:
                state.one_step[key] = [None] * max(0, t)
                state.born_round[key] = t
                state.active_rounds[key] = []
            return key

        def mark_active(state, hypothesis, t):
            """Record `hypothesis` as active in the component's archive (no-op if already active)."""
            key = register(state, hypothesis, t)
            entry = state.archive.get(key)
            if entry is not None and entry["outcome"] == "active":
                return
            state.archive[key] = {
                **state.candidate_meta.get(key, {}),
                "model_id": key,
                "expression": hypothesis.expression,
                "family": hypothesis.family,
                # The role mapping the engine judged, recorded; no reader decides on it.
                "parameters": dict(hypothesis.parameters),
                "proposed_round": state.born_round[key],
                "outcome": "active",
                "reason": {"admitted_round": t, "active_rounds": state.active_rounds[key]},
            }

        def stamp_active(state, t):
            """Append round `t` to the active-round history of every model in the component's pool."""
            for result in state.fits:
                history = state.active_rounds[register(state, result.hypothesis, t)]
                if not history or history[-1] != t:
                    history.append(t)

        def refit(state, hypothesis, t):
            """Fit `hypothesis` on all current public data for one component; writes the fit record, returns the result."""
            seed = (
                seed_for("boxinggym_smc", spec.name, args.seed, args.cell, t, rng_identity(hypothesis))
                if not multi
                else seed_for(
                    "boxinggym_smc", spec.name, args.seed, args.cell, t, rng_identity(hypothesis),
                    state.name,
                )
            )
            x = np.asarray([row["x"] for row in public], dtype=float)
            y = np.asarray([component_y(row, state) for row in public], dtype=float)
            if fit_guards == "v1":
                callable_fit = count_fit if exhaustive else scalar_fit
                fit_args = ((hypothesis, x, y, seed, settings, interface, interface["trials"])
                            if exhaustive else
                            (hypothesis, x, y, seed, None, state.response_unit, settings, interface,
                             state.response_center))
                if fit_timeout_seconds is not None:
                    from functools import partial
                    callable_fit = partial(run_with_fit_timeout, callable_fit,
                                           fit_timeout_seconds=fit_timeout_seconds)
                started_fit = time.perf_counter()
                fit_status = "completed"
                try:
                    result = run_with_fit_guards(callable_fit, *fit_args, fit_guards=fit_guards)
                except FitTimeout:
                    fit_status = "fit_timeout"
                    raise
                except BaseException:
                    fit_status = "fit_failed"
                    raise
                finally:
                    if fit_timeout_seconds is not None:
                        record_fit_timing(out, hypothesis, t,
                                          time.perf_counter() - started_fit, fit_status)
            elif fit_timeout_seconds is None:
                if exhaustive:
                    result = count_fit(hypothesis, x, y, seed, settings, interface, interface["trials"])
                else:
                    result = scalar_fit(
                        hypothesis, x, y, seed, None, state.response_unit, settings, interface,
                        state.response_center,
                    )
            else:
                callable_fit = count_fit if exhaustive else scalar_fit
                fit_args = ((hypothesis, x, y, seed, settings, interface, interface["trials"])
                            if exhaustive else
                            (hypothesis, x, y, seed, None, state.response_unit, settings, interface,
                             state.response_center))
                started_fit = time.perf_counter()
                try:
                    result = run_with_fit_timeout(callable_fit, *fit_args,
                                                  fit_timeout_seconds=fit_timeout_seconds)
                except FitTimeout as exc:
                    record_fit_timing(out, hypothesis, t, exc.elapsed_seconds, "fit_timeout")
                    timings["fit"] += exc.elapsed_seconds
                    raise
                except Exception:
                    record_fit_timing(out, hypothesis, t, time.perf_counter()-started_fit, "fit_failed")
                    raise
                else:
                    record_fit_timing(out, hypothesis, t, time.perf_counter()-started_fit, "completed")
            result.diagnostics["round"] = t
            result.diagnostics["public_data_sha256"] = hashlib.sha256(
                json.dumps([(row["x"], component_y(row, state)) for row in public]).encode()
            ).hexdigest()
            result.diagnostics["observation_count"] = len(public)
            if multi:
                result.diagnostics["component"] = state.name
            timings["fit"] += result.seconds
            stem = artifact_stem(state, f"t{t:02d}") + f"_{hypothesis.model_id}"
            np.savez_compressed(out / "fits" / f"{stem}.npz", **result.particles)
            probe_record, predictive_probe = run_prior_probe(
                hypothesis, interface, state.response_unit, response_center=state.response_center
            )
            state.evidence_stamp[hypothesis.model_id] = {"computed_round": t, "n_obs": len(public)}
            write_json(
                out / "fits" / f"{stem}.json",
                {
                    **hypothesis.as_dict(),
                    # Only when the switch is on, so a default run's fit records are unchanged.
                    **({"stable_log1mexp_sites": log1mexp_sites(hypothesis.expression)}
                       if stable_log1mexp_enabled() else {}),
                    "log_evidence": result.log_evidence,
                    "per_chain_log_evidence": result.per_chain_log_evidence,
                    "diagnostics": result.diagnostics,
                    "fit_seconds": result.seconds,
                    "prior_predictive": {**probe_record, **(predictive_probe or {})},
                },
            )
            print(
                f"FIT t={t}{'' if not multi else ' component=' + state.name} "
                f"model={hypothesis.model_id} family={hypothesis.family} "
                f"logZ={result.log_evidence:.6f} seconds={result.seconds:.2f}",
                flush=True,
            )
            return result

        def retire_timeout(state, hypothesis, t, exc, item=None):
            """Archive a model whose fit timed out as fit_failed; returns the failure record."""
            key = register(state, hypothesis, t)
            failure = dict(hypothesis.as_dict() if item is None else item)
            failure.update(exc.as_record(), model_id=key, round=t)
            # The candidate_failures entry names its failure class.
            failure["failure_class"] = failure_class(status=failure["status"])
            if multi:
                failure["component"] = state.name
            failures.append(failure)
            state.archive[key] = {
                "model_id": key, "expression": hypothesis.expression,
                "family": hypothesis.family, "parameters": dict(hypothesis.parameters),
                "proposed_round": state.born_round[key],
                "outcome": "fit_failed", "reason": {"error": f"FitTimeout: {exc}"},
            }
            return failure

        def retire_nonfinite_smc(state, hypothesis, t, exc, item=None):
            """Archive a model whose SMC weights went non-finite as fit_failed; re-raises any other failure."""
            if exc.cause_code != "nonfinite_smc_weights":
                raise exc
            key = register(state, hypothesis, t)
            failure = dict(hypothesis.as_dict() if item is None else item)
            failure.update(status="fit_failed", cause_code=exc.cause_code,
                           retirement_kind="nonfinite_smc_weights", stage="sampling",
                           model_id=key, round=t, error=f"{type(exc).__name__}: {exc}")
            failure["failure_class"] = failure_class(status=failure["status"])
            if multi:
                failure["component"] = state.name
            failures.append(failure)
            state.archive[key] = {
                **state.candidate_meta.get(key, {}),
                "model_id": key, "expression": hypothesis.expression,
                "family": hypothesis.family, "parameters": dict(hypothesis.parameters),
                "proposed_round": state.born_round[key],
                "outcome": "fit_failed", "reason": {"error": failure["error"]},
            }
            return failure

        def fit_survivors(state, hypotheses, t):
            """Fit each hypothesis, retiring guarded failures; raises ValueError when no model survives."""
            survivors = []
            for hypothesis in hypotheses:
                try:
                    survivors.append(refit(state, hypothesis, t))
                except FitTimeout as exc:
                    retire_timeout(state, hypothesis, t, exc)
                except CountInferenceFailure as exc:
                    if fit_guards != "v1":
                        raise
                    retire_nonfinite_smc(state, hypothesis, t, exc)
            if not survivors:
                message = ("no models remain after guarded fit failures" if fit_guards == "v1"
                           else "no models remain after fit_timeout")
                raise ValueError(message)
            return survivors

        def predictive(state, designs):
            """BMA predictive at `designs`: point mean, nll callable, 90% interval."""
            if exhaustive:
                prediction = count_predict(state.fits, designs, interface, interface["trials"])
                logpmf = prediction["bma_logpmf"]
                rng = np.random.default_rng(
                    seed_for("boxinggym_predictive", spec.name, args.seed)
                    if not multi
                    else seed_for("boxinggym_predictive", spec.name, args.seed, state.name)
                )
                summary = count_predictive_summary(logpmf, rng)
                counts = np.arange(logpmf.shape[-1])
                return {
                    "mean": np.exp(logpmf) @ counts,
                    "lower90": np.asarray(summary["lower90"], dtype=float),
                    "upper90": np.asarray(summary["upper90"], dtype=float),
                    "logpmf": logpmf,
                    "model_logweights": prediction["model_logweights"],
                    "single_model_id": prediction["single_model_id"],
                    "model_means": np.exp(prediction["model_logpmfs"]) @ counts,
                }
            seed = (
                seed_for("boxinggym_predictive", spec.name, args.seed, len(public))
                if not multi
                else seed_for("boxinggym_predictive", spec.name, args.seed, len(public), state.name)
            )
            summary = scalar_predictive_summary(state.fits, designs, seed, 2048)
            return {
                "mean": np.asarray(summary["mean"], dtype=float),
                "lower90": np.asarray(summary["lower90"], dtype=float),
                "upper90": np.asarray(summary["upper90"], dtype=float),
                "logpmf": None,
                "model_logweights": np.log(np.asarray(summary["model_weights"], dtype=float)),
                "single_model_id": summary["single_model_id"],
                "model_means": np.stack([
                    scalar_predictive_means(result, designs).mean(0) for result in state.fits
                ]),
            }

        def our_scores(state, prediction):
            """Our predictive scores on the held-out points; the same points the env uses."""
            targets = component_heldout(state)
            if exhaustive:
                rows = prediction["logpmf"]
                nll = -float(rows[np.arange(len(targets)), targets.astype(int)].mean())
            else:
                mixed, _ = log_predictive(state.fits, evaluation_x, targets)
                nll = -float(mixed[:, 0].mean())
            inside = (targets >= prediction["lower90"]) & (targets <= prediction["upper90"])
            error = prediction["mean"] - targets
            result = {
                "nll": nll,
                "coverage90": float(inside.mean()),
                "width90": float((prediction["upper90"] - prediction["lower90"]).mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mse": float(np.mean(error**2)),
            }
            # The Brier
            # score, defined ONLY for the two-class Bernoulli outcome
            # (survival, hyperbolic, irt, moral_machines - `outcome_type ==
            # "binary"`, EXHAUSTIVE_COUNT_OUTCOMES' other member
            # "bounded_count" is an N-ary count, not a probability, and has no
            # Brier score). `prediction["logpmf"]` columns are indexed by
            # count 0..interface["trials"]; for trials == 1 that is exactly
            # [P(Y=0), P(Y=1)], so column 1 is P(Y=1|x) - written out
            # explicitly rather than reusing "mse" above (which happens to be
            # numerically identical here, since E[Y] IS P(Y=1) for a
            # Bernoulli) so this column does not silently depend on that
            # coincidence remaining true. Left absent (never 0) for every
            # other outcome type; `write_points` reads it with `.get(...,
            # "")`.
            if interface["outcome_type"] == "binary":
                p1 = np.exp(prediction["logpmf"][:, 1])
                result["brier"] = float(np.mean((p1 - targets) ** 2))
            # death's nmse_mda (the paper's Table 6b) - the BMA predictive mean scored against the
            # noise-free truth on the FIXED evaluation set Q
            # (`death_true_mean`/`death_var_q`, computed once, above, from
            # `death_noise_free_truth`), normalized by that truth's own
            # population variance (ddof=0) over Q - not against `targets`
            # (the noisy heldout `y` `mse`/`rmse` above are still scored
            # against). `mse` above is UNCHANGED: it is a
            # different, still-computed number (predictive mean vs the
            # noisy observation), left exactly as it was.
            if spec.name == "death":
                result["nmse_mda"] = float(
                    np.mean((prediction["mean"] - death_true_mean) ** 2) / death_var_q
                )
                # KL(truth || BMA) and expected_log_score, CORE's
                # own formula (`core_count.score_prediction`, imported
                # unchanged - its body is the exact
                # definition) applied to death's true PMF (`death_true_pmf`,
                # computed once above) against the BMA predictive log-PMF
                # this checkpoint already built (`rows`, the very array `nll`
                # above was scored from - not a re-derivation of either
                # quantity). `targets.astype(int)` matches the `nll` line
                # above (`heldout`/`targets` is stored as float). The rng is
                # a fresh, independent Generator - `score_prediction` only
                # uses it to build the `summary` this call discards (the
                # "samples" field of `predictive_summary`), so it perturbs no
                # other run output; seeded by round via `len(public)`, like
                # `predictive`'s own scalar-path rng above.
                death_kl_rng = np.random.default_rng(
                    seed_for("boxinggym_death_kl", spec.name, args.seed, len(public))
                )
                death_scores, _ = score_prediction(
                    rows, targets.astype(int), death_true_pmf, death_kl_rng
                )
                result["kl"] = death_scores["kl"]
                result["expected_log_score"] = death_scores["expected_log_score"]
            # The same KL(truth || BMA), on the same fixed Q, for
            # every OTHER task that has a true observation law. death is
            # excluded by construction - its branch above is the original death-only path,
            # unchanged and still authoritative - so no death number, and no
            # rng stream death reads, can move. `prediction` (whose `logpmf`
            # the exhaustive tasks' KL uses) and `state.fits` (the density and
            # count tasks') are the SAME objects `nll` above was scored from,
            # never a re-derivation. Definitions per task: see
            # `boxinggym_truth_law`'s module docstring.
            # `kl_recorded` is `truth_law_q is not None` AND this
            # run's flag rule: the law is now built even for an optional task
            # that will not be scored, so the gate is the flag, not the law.
            elif kl_recorded:
                result.update(kl_against_pool(
                    truth_law_q,
                    prediction if exhaustive else state.fits,
                    evaluation_x,
                    state.index if multi else None,
                ))
            return result

        checkpoint_models = []

        def checkpoint(t, phase, scoring_phase):
            """Score the current pools on the held-out points; records and returns the score row and the pool signature."""
            nonlocal checkpoint_models
            nonlocal evaluation_x, heldout, death_true_mean, death_var_q, death_true_pmf
            nonlocal truth_law_q
            if q_policy == "exclude_observed":
                keep = unobserved_q_mask(evaluation_x, [row["x"] for row in public])
                if not keep.any():
                    raise ValueError("exclude_observed removed every evaluation point")
                if not keep.all():
                    config["q_excluded_count"] += int((~keep).sum())
                    evaluation_x, heldout = evaluation_x[keep], heldout[keep]
                    environment.goal.eval_points = [point for point, retained in zip(
                        environment.goal.eval_points, keep, strict=True) if retained]
                    if spec.name == "death":
                        death_true_mean = death_true_mean[keep]
                        death_var_q = float(np.var(death_true_mean, ddof=0))
                        death_true_pmf = death_true_pmf[keep]
                    # The true law is a per-evaluation-point object,
                    # so it follows Q through the same subsetting - truth and
                    # prediction stay aligned point for point.
                    truth_law_q = subset_law(truth_law_q, keep)
                    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
                    np.savez_compressed(out / "evaluation_scoring_only.npz",
                                        x=evaluation_x, y=heldout, designs=pool.rows)
            tic = time.perf_counter()
            normalization_before = normalization_event_counter()
            predictions = [predictive(state, evaluation_x) for state in states]
            normalization = predictive_normalization_record(
                normalization_before, normalization_event_counter()
            )
            timings["forward"] += time.perf_counter() - tic
            tic = time.perf_counter()
            submitted = format_predictions(
                spec,
                predictions[0]["mean"],
                component_means=[p["mean"] for p in predictions] if multi else None,
            )
            env_score = environment.evaluate(submitted)
            per_component = [our_scores(state, p) for state, p in zip(states, predictions, strict=True)]
            if not multi:
                row = {
                    "round": t,
                    "phase": phase,
                    "scoring_phase": scoring_phase,
                    "post_equals_pre": False,
                    "observation_count": len(public),
                    "active_ids": [v.hypothesis.model_id for v in states[0].fits],
                    "model_weights": np.exp(predictions[0]["model_logweights"]).tolist(),
                    "single_model_id": predictions[0]["single_model_id"],
                    "env_evaluator": {**env_score, "metric_name": spec.env_metric},
                    "bma": per_component[0],
                }
            else:
                # Table 2's last row: "component-wise scores are composite and
                # labelled so". Every reported number is the mean over the
                # declared components, and the row says which rule made it.
                row = {
                    "round": t,
                    "phase": phase,
                    "scoring_phase": scoring_phase,
                    "post_equals_pre": False,
                    "observation_count": len(public),
                    "env_evaluator": {**env_score, "metric_name": spec.env_metric},
                    "bma": {
                        key: composite_value([score[key] for score in per_component])
                        for key in per_component[0]
                    },
                    "composite": True,
                    "composite_rule": COMPOSITE_SCORE_RULE,
                    "components": [
                        {
                            "component": state.name,
                            "active_ids": [v.hypothesis.model_id for v in state.fits],
                            "model_weights": np.exp(prediction["model_logweights"]).tolist(),
                            "single_model_id": prediction["single_model_id"],
                            "bma": score,
                        }
                        for state, prediction, score in zip(
                            states, predictions, per_component, strict=True
                        )
                    ],
                }
            if normalization is not None:
                # Recorded ONLY when the rule actually fired during
                # this checkpoint's predictive, so an untouched run's scores.jsonl
                # is byte-identical - the same condition core_count applies to
                # `nan_substitutions`. The run total is folded in exactly here,
                # once per checkpoint, and reaches summary.json under that same
                # condition.
                row["predictive_normalization"] = normalization
                predictive_normalization_total["renormalized_rows"] += normalization[
                    "renormalized_rows"]
                predictive_normalization_total["zeroed_rows"] += normalization["zeroed_rows"]
                predictive_normalization_total["max_deviation_cumulative"] = max(
                    predictive_normalization_total["max_deviation_cumulative"],
                    normalization["max_deviation_cumulative"],
                )
                predictive_normalization_total["checkpoints"] += 1
            for state, prediction in zip(states, predictions, strict=True):
                np.savez_compressed(
                    out / "predictions" / f"{artifact_stem(state, f't{t:02d}')}_{phase}.npz",
                    x=evaluation_x,
                    mean=prediction["mean"],
                    lower90=prediction["lower90"],
                    upper90=prediction["upper90"],
                    model_logweights=prediction["model_logweights"],
                )
            truth = None if spec.truth_mean is None else spec.truth_mean(environment, evaluation_x)
            available_truth = truth is not None
            if config["truth_mean_available"] != available_truth:
                config["truth_mean_available"] = available_truth
                (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            previous = mda_rows[-1]["best_so_far_nmse_truth"] if mda_rows else None
            previous = None if previous == "not_available" else previous
            means = np.column_stack([p["mean"] for p in predictions]) if multi else predictions[0]["mean"]
            location_options = {}
            if spec.name == "location":
                previous_log = mda_rows[-1]["best_so_far_nmse_log_truth"] if mda_rows else None
                location_options = {"location": True,
                                    "previous_best_log": None if previous_log == "not_available" else previous_log}
            mda_rows.append(mda_score_row(len(public), means, truth,
                                         row["bma"].get("brier", ""), previous, **location_options))
            checkpoint_models = [
                {"model_id": result.hypothesis.model_id,
                 "weight": float(np.exp(prediction["model_logweights"][index])),
                 "means": prediction["model_means"][index],
                 **({"component": state.name} if multi else {})}
                for state, prediction in zip(states, predictions, strict=True)
                for index, result in enumerate(state.fits)
            ]
            record_prediction_pool_summary(out, t, "official", len(public), checkpoint_models)
            add_recorded_pool_count(row)
            append_json(out / "scores.jsonl", row)
            score_rows.append(row)
            timings["scoring"] += time.perf_counter() - tic
            print(
                f"SCORE t={t} phase={phase} pool={sum(len(s.fits) for s in states)} "
                f"env_{spec.env_metric}={env_score['metric']:.6f} nll={row['bma']['nll']:.6f}",
                flush=True,
            )
            return row, tuple(
                frozenset((v.hypothesis.model_id, id(v), v.log_evidence) for v in state.fits)
                for state in states
            )

        def pool_members(state, t, members=None):
            """The component's active pool (or `members`) as belief-packet PoolMember records."""
            # `members` (a round-0 refill wave's intermediate pool) replaces the
            # component's active fits; None - every other caller - is the active pool.
            return [
                PoolMember(
                    v.hypothesis,
                    v.particles,
                    v.log_evidence,
                    tuple(state.one_step[v.hypothesis.model_id]),
                    tuple(range(1, t + 1)),
                    state.born_round[v.hypothesis.model_id],
                    state.born_round[v.hypothesis.model_id],
                    tuple(state.active_rounds[v.hypothesis.model_id]),
                    state.evidence_stamp[v.hypothesis.model_id]["computed_round"],
                    state.evidence_stamp[v.hypothesis.model_id]["n_obs"],
                    state.response_unit,
                    state.response_center,
                )
                for v in (state.fits if members is None else members)
            ]

        def retire(state, retired, t, cutoff):
            """Archive the models `retain` dropped as retired_by_evidence, with their evidence gap to the cutoff."""
            for result in retired:
                key = result.hypothesis.model_id
                state.archive[key] = {
                    **state.candidate_meta.get(key, {}),
                    "model_id": key,
                    "expression": result.hypothesis.expression,
                    "family": result.hypothesis.family,
                    "parameters": dict(result.hypothesis.parameters),
                    "proposed_round": state.born_round[key],
                    "outcome": "retired_by_evidence",
                    "reason": {
                        "log_evidence_at_retirement": float(result.log_evidence),
                        "gap_to_cutoff_nats": float(cutoff - result.log_evidence),
                        "active_rounds": state.active_rounds[key],
                        "computed_round": state.evidence_stamp[key]["computed_round"],
                        "n_obs": state.evidence_stamp[key]["n_obs"],
                    },
                }

        revival_events = []

        def review_archive(state, t, transport_record):
            """Archive review: the reply's `revive` ids are validated - an archive entry
            with outcome retired_by_evidence, at most ARCHIVE_REVIVE_LIMIT, each once; an invalid or
            extra id is recorded and ignored, never a repair turn - and each valid one is refit from
            the original prior on all current public data. The round's `retain` then ranks active +
            revived + new together. Returns the revived fits and the event `finish_review` completes."""
            tic = time.perf_counter()
            requested = transport_record.get("revive_requested") or []
            active_ids = {v.hypothesis.model_id for v in state.fits}
            valid, invalid, seen_ids = [], [], set()
            for position, raw in enumerate(requested):
                key = raw if isinstance(raw, str) else json.dumps(raw, sort_keys=True)
                entry = state.archive.get(key) if isinstance(raw, str) else None
                # `why` is one of {active, not_retired, unknown, over_limit}; the third and
                # later ids are over_limit whatever they name; a repeat or a non-string is named too.
                if position >= ARCHIVE_REVIVE_LIMIT:
                    invalid.append({"id": key, "why": "over_limit"})
                elif not isinstance(raw, str):
                    invalid.append({"id": key, "why": "not_a_model_id_string"})
                elif key in seen_ids:
                    invalid.append({"id": key, "why": "repeated"})
                elif key in active_ids:
                    invalid.append({"id": key, "why": "active"})
                elif entry is None:
                    invalid.append({"id": key, "why": "unknown"})
                elif entry["outcome"] != "retired_by_evidence":
                    # rejected / duplicate / fit_failed entries: known, but never retired on evidence
                    invalid.append({"id": key, "why": "not_retired"})
                elif key not in state.hypotheses:
                    invalid.append({"id": key, "why": "unknown"})  # defensive: no object to refit
                else:
                    valid.append(key)
                seen_ids.add(key)
            revived, refit_results, refit_failures = [], [], []
            for key in valid:
                # the counter lives in candidate_meta, which retire() and mark_active() both copy
                meta = state.candidate_meta.setdefault(key, {})
                meta["revival_attempts"] = int(meta.get("revival_attempts", 0)) + 1
                accounting["revival_refits"] += 1
                accounting["fit_attempts"] += 1
                try:
                    result = refit(state, state.hypotheses[key], t)
                except (FitTimeout, CountInferenceFailure) as exc:
                    refit_failures.append({"id": key, "error": f"{type(exc).__name__}: {exc}", "scope": "revival"})
                else:
                    accounting["fit_successes"] += 1
                    revived.append(result)
                    refit_results.append({"id": key, "log_evidence": float(result.log_evidence)})
            event = {"round": t, **({"component": state.name} if multi else {}),
                     "requested_ids": [r if isinstance(r, str) else json.dumps(r, sort_keys=True) for r in requested],
                     "reply_shape": transport_record.get("reply_shape"),
                     # A plain array on a review round is accepted with no revive.
                     "revive_missing": transport_record.get("reply_shape") != "object",
                     "valid_ids": valid, "invalid": invalid, "refit_results": refit_results,
                     "refit_failures": refit_failures, "observation_count": len(public), "started": tic}
            return revived, event

        def finish_review(state, t, event, revived):
            """After the round's retain and retire: which revived models re-entered and which were
            retired again (their archive entries carry revival_attempts), and the wall time."""
            active_ids = {v.hypothesis.model_id for v in state.fits}
            revived_ids = [v.hypothesis.model_id for v in revived]
            event["re_entered_ids"] = [key for key in revived_ids if key in active_ids]
            event["retired_again_ids"] = [key for key in revived_ids if key not in active_ids]
            event["wall_seconds"] = time.perf_counter() - event.pop("started")
            write_json(out / "revival" / f"{artifact_stem(state, f't{t:02d}')}.json", event)
            revival_events.append(event)

        def one_step_score(state, result, x, y):
            """One-step-ahead log predictive score of a new observation under one fitted model, or None."""
            try:
                point = np.asarray([float(x)]) if single_input else np.asarray([x], dtype=float)
                if exhaustive:
                    # Averaged over the particles that carry weight.
                    rows = active_particle_rows(particle_logpmf(
                        result.hypothesis,
                        result.particles,
                        point,
                        interface["trials"],
                        interface,
                    ))
                    return float(logsumexp(rows[:, 0, int(y)]) - np.log(rows.shape[0]))
                mixed, _ = log_predictive([result], point, np.asarray([float(y)]))
                return float(mixed[0, 0])
            except Exception:  # noqa: BLE001 - a degenerate model scores null, never crashes
                return None

        # Probes computed by the admission hook, read once by the round loop.
        prior_probe_cache = {}

        def to_task_class(candidate):
            """Convert a proposer's count-model hypothesis to this task's hypothesis class."""
            # The same representation change the admitted proposals get below.
            if exhaustive:
                return candidate
            if real_outcome:
                return RealHypothesis(
                    candidate.expression, candidate.family, candidate.parameters,
                    candidate.input_names,
                    default_scale_expression(interface)
                    if candidate.family == "heteroscedastic_normal" else None,
                )
            return ScalarHypothesis(
                candidate.expression, candidate.family, candidate.parameters, candidate.input_names
            )


        def make_admit(state, t, known):
            """The prior-probe admission hook for one component and round."""
            def admit(candidate):
                """Admission hook: run the prior probe; on failure archive the candidate and raise PriorProbeRejection."""
                # The prior probe as the proposer's admission hook (see
                # core_count for the rule). model_id is class-independent, so the
                # converted candidate is the same model the loop below sees.
                hypothesis = to_task_class(candidate)
                key = hypothesis.model_id
                if key in known:
                    return
                probe_record, _ = run_prior_probe(hypothesis, interface, state.response_unit)
                if probe_record["passed"]:
                    prior_probe_cache[key] = probe_record
                    return
                register(state, hypothesis, t)
                accounting["prior_probe_rejections"] += 1
                known.add(key)
                state.archive[key] = {
                    "model_id": key,
                    "expression": hypothesis.expression,
                    "family": hypothesis.family,
                    "parameters": dict(hypothesis.parameters),
                    "proposed_round": state.born_round[key],
                    "prior_probe": probe_record,
                    "note": "",
                    "outcome": "rejected",
                    "reason": prior_probe_reason(probe_record),
                }
                # The round's record lists the hook's refusals (per candidate,
                # as the burst path lists its own); the proposer removes only this candidate.
                admit.rejected.append(key)
                raise PriorProbeRejection(probe_record["detail"])

            admit.rejected = []
            return admit

        def round_packet(state, t, members=None):
            """The belief packet of round t for one component, written to packets/."""
            packet_constraints = ({**constraints, "initial_structures": []}
                                  if t == 0 and initial_catalogue == "off" else constraints)
            packet = build_belief_packet(
                [
                    {
                        "obs_id": index,
                        "round": 0 if index < 4 else index - 3,
                        **dict(zip(spec.input_names,
                                   [row["x"]] if single_input else row["x"], strict=True)),
                        "y": component_y(row, state),
                    }
                    for index, row in enumerate(public)
                ],
                pool_members(state, t, members), list(state.archive.values()), packet_constraints, interface, t,
            )
            packet_text = serialize_packet(packet)
            return packet_text, packet_digest(packet_text)

        def parallel_wave(state, t, count):
            """One proposal round as `count` independent single-candidate calls
            in parallel (at most `burst_concurrency` in flight), then the round loop's own
            admission (register, probe, duplicate check, fit) and `retain` under the
            active-pool cap. A candidate that fell out (a duplicate, a probe refusal, a
            failed call) is asked for once more in ONE refill wave of that many calls,
            whose packet carries this run's archive as it stands after the first wave.
            Round 0 is the initial burst (K = initial_proposals); a later round runs
            with K = parallel_proposals. Records: packets/t<NN>[_burst] and
            proposals/t<NN>[_burst] (every call's transport record, the events, the
            waves, the maximum in flight and the pool before/after) and the per-call
            transport files round_NN__burst_w<wave>_c<index>[_r<retry>].

            The round-0 burst refills in at most two waves (1 and 2), each of
            exactly the shortfall, whose packet's pool is the intermediate retained pool
            and whose request carries the refill paragraph; every executed wave's packet
            is written as packets/t00_burst_w<wave>[_<component>]."""
            tic = time.perf_counter()
            stem = artifact_stem(state, "t00_burst" if t == 0 else f"t{t:02d}")
            lock = threading.Lock()
            gauge = {"in_flight": 0, "max_in_flight": 0, "mock_offset": (t - 1) % len(templates) if t else 0}
            previous_ids = [f.hypothesis.model_id for f in state.fits]
            known = set(state.archive) | set(previous_ids)
            events, fitted, transports, waves = [], [], [], []
            # Model-identity preimage: each event's own `model_id_preimage`, added to the burst record only (the
            # same event dicts also reach summary.json's candidate_failures, which must not change).
            event_preimages = []
            fit_attempt_count = 0
            wave_packets = []

            def admit_locked(hook):
                """Wrap an admission hook so that it runs under the wave's lock."""
                def admit(candidate):
                    """Call the hook while holding the wave's lock."""
                    # The hook mutates the archive, the probe cache and the accounting.
                    with lock:
                        hook(candidate)
                return admit

            def call(wave, index, *, packet_text, digest, admit, burst=True):
                """One proposer call of a wave (mock template or LLM, with 429 retries); returns candidates and transport record."""
                # `burst` is "refill" for a round-0 refill wave, True otherwise.
                context = {
                    "run_id": str(out), "world_seed": world_seed, "cell": args.cell,
                    "proposal_rep": 0, "round": t, "burst_call_index": index, "burst_wave": wave,
                    **({"component": state.name} if multi else {}),
                }
                if proposer is None:
                    # The mock schedule's next template, so a mock parallel run exercises
                    # the whole admission path (with its duplicates) on no new schedule.
                    with lock:
                        offset = gauge["mock_offset"]
                        gauge["mock_offset"] += 1
                    candidate = make(templates[offset % len(templates)],
                                     families[offset % len(templates)] if args.cell[1] == "1"
                                     else initial_family)
                    return [candidate], {
                        "provider": "none", "mock": True, "api_request_attempted": False,
                        "policy": "P1", "status": "parsed", "calls": 0, "repairs": 0,
                        "rejected_candidates": [], "note": "", **context,
                        **composed_system_prompt(burst=burst)[1], **digest,
                    }
                for retry in range(BURST_RETRIES + 1):
                    with lock:
                        gauge["in_flight"] += 1
                        gauge["max_in_flight"] = max(gauge["max_in_flight"], gauge["in_flight"])
                    try:
                        candidates, record = proposer.propose_one(
                            packet_text,
                            context={**context, **({"burst_retry": retry} if retry else {})},
                            initial_family=initial_family,
                            initial_structures=tuple(make(form, initial_family) for form in names),
                            repair_attempts=1,
                            allowed_families=allowed,
                            admit=admit,
                            burst=burst,
                            max_rounds=args.rounds,
                        )
                    finally:
                        with lock:
                            gauge["in_flight"] -= 1
                    rate_limited = record["status"] != "parsed" and any(
                        item.get("http_status") == 429 for item in record["rejected_candidates"])
                    if not rate_limited or retry == BURST_RETRIES:
                        break
                    # A 429 is retried with exponential backoff; the refused call's
                    # record stays on disk under its own prefix.
                    time.sleep(BURST_BACKOFF_SECONDS * 2 ** retry)
                if not exhaustive:
                    candidates = [to_task_class(candidate) for candidate in candidates]
                return candidates, record

            def admit_candidate(hypothesis, note):
                """Probe, deduplicate and fit one wave candidate; returns True when it was fitted."""
                nonlocal fit_attempt_count
                item = hypothesis.as_dict()
                key = register(state, hypothesis, t)
                probe = prior_probe_cache.pop(key, None)
                if probe is None:
                    probe, _ = run_prior_probe(hypothesis, interface, state.response_unit)
                loglik_probe = None
                if fit_guards == "v1" and probe["passed"] and key not in known:
                    x_probe = np.asarray([row["x"] for row in public], dtype=float)
                    y_probe = np.asarray([component_y(row, state) for row in public], dtype=float)
                    model_builder = (None if exhaustive else partial(
                        scalar_build_model, hypothesis, x_probe, y_probe, None,
                        state.response_unit, interface, state.response_center))
                    loglik_probe = loglik_prior_probe(
                        hypothesis, x_probe, y_probe, interface,
                        prior_draws=probe_particles(hypothesis, PRIOR_PROBE_SAMPLES),
                        model_builder=model_builder, n_trials=interface["trials"],
                    )
                    probe = {**probe, "loglik_prior_probe": loglik_probe}
                source = "initial_burst" if t == 0 else "parallel_wave"
                item.update(prior_probe=probe, note=note, source=source)
                state.candidate_meta[key] = {"prior_probe": probe, "note": note, "source": source}
                stub = {"model_id": key, "expression": hypothesis.expression,
                        "family": hypothesis.family, "parameters": dict(hypothesis.parameters),
                        "proposed_round": state.born_round[key],
                        "prior_probe": probe, "note": note, "source": source}
                accepted = False
                if key in known:
                    item["status"] = "duplicate_not_refitted"
                    accounting["duplicates"] += 1
                    accounting["burst_duplicates"] += 1
                    matched = next((f.hypothesis for f in state.fits if f.hypothesis.model_id == key),
                                   state.hypotheses.get(key, hypothesis))
                    state.archive[duplicate_archive_key(state.archive, t, key)] = {
                        **stub, "proposed_round": t, "outcome": "duplicate",
                        "reason": {"matched_model_id": key, "matched_expression": matched.expression},
                    }
                elif not probe["passed"]:
                    item["status"] = "rejected_prior_probe"
                    accounting["prior_probe_rejections"] += 1
                    known.add(key)
                    state.archive[key] = {**stub, "outcome": "rejected",
                                          "reason": prior_probe_reason(probe)}
                elif loglik_probe is not None and not loglik_probe["passed"]:
                    item["status"] = "rejected_loglik_prior_probe"
                    accounting["prior_probe_rejections"] += 1
                    known.add(key)
                    state.archive[key] = {**stub, "outcome": "rejected",
                                          "reason": {"validation_message": loglik_probe["reason"]}}
                else:
                    known.add(key)
                    accounting["fit_attempts"] += 1
                    fit_attempt_count += 1
                    try:
                        fitted.append(refit(state, hypothesis, t))
                    except FitTimeout as exc:
                        item.update(retire_timeout(state, hypothesis, t, exc, item))
                    except CountInferenceFailure as exc:
                        if fit_guards != "v1" or exc.cause_code != "nonfinite_smc_weights":
                            item.update(status="fit_or_predict_failed", error=f"{type(exc).__name__}: {exc}")
                            failures.append(item)
                            state.archive[key] = {**stub, "outcome": "fit_failed",
                                                  "reason": {"error": f"{type(exc).__name__}: {exc}"}}
                        else:
                            item.update(retire_nonfinite_smc(state, hypothesis, t, exc, item))
                    except Exception as exc:  # noqa: BLE001 - a failed candidate, not a failed run
                        item.update(status="fit_or_predict_failed", error=f"{type(exc).__name__}: {exc}")
                        failures.append(item)
                        state.archive[key] = {**stub, "outcome": "fit_failed",
                                              "reason": {"error": f"{type(exc).__name__}: {exc}"}}
                    else:
                        accepted = True
                        accounting["fit_successes"] += 1
                        # The fitted model joins the pool now, so the refill wave's packet
                        # and duplicate check see it (this run's own archive, as the packet
                        # always shows it).
                        state.archive[key] = {
                            **stub, "outcome": "active",
                            "reason": {"admitted_round": t, "active_rounds": state.active_rounds[key]},
                        }
                        item.update(status="fitted", log_evidence=fitted[-1].log_evidence)
                # The event names its failure class (None when fitted); a candidate_failures
                # entry that is this same dict names it too.
                item["failure_class"] = failure_class(status=item["status"])
                events.append(item)
                event_preimages.append(hypothesis.model_id_preimage)
                return accepted

            shortfall = count
            # The round-0 burst runs at most three waves; a later round keeps its two.
            for wave in (INITIAL_BURST_WAVES if t == 0 else (0, 1)):
                if shortfall <= 0:
                    break
                # A round-0 refill wave's packet shows the component's intermediate retained
                # pool (its active fits plus this round's fits so far); `state.fits` is untouched.
                refill = t == 0 and wave > 0
                shown = burst_refill_pool(state.fits + fitted, active_maximum) if refill else None
                packet_text, digest = round_packet(state, t, shown)
                wave_packets.append(packet_text)
                admit = admit_locked(make_admit(state, t, known))
                with ThreadPoolExecutor(max_workers=max(1, min(burst_concurrency, shortfall))) as pool_:
                    results = list(pool_.map(
                        partial(call, wave, packet_text=packet_text, digest=digest, admit=admit,
                                burst="refill" if refill else True),
                        range(shortfall)))
                accepted, wave_events, wave_transports = 0, len(events), len(transports)
                for index, (returned, record) in enumerate(results):
                    transports.append({"wave": wave, "index": index, "transport": record})
                    accounting["burst_calls"] += record["calls"]
                    accounting["burst_repairs"] += record["repairs"]
                    accounting["llm_calls"] += record["calls"]
                    accounting["llm_repairs"] += record["repairs"]
                    if wave:
                        accounting["burst_refill_calls"] += record["calls"]
                    if record["status"] != "parsed":
                        accounting["burst_failed_calls"] += 1
                        failures.append({"round": t, "burst_wave": wave, "burst_call_index": index,
                                         "status": "proposal_failed",
                                         **({"component": state.name} if multi else {}),
                                         "transport": record})
                    accounting["burst_candidates"] += len(returned)
                    for candidate in returned:
                        accepted += admit_candidate(candidate, record.get("note", ""))
                waves.append(burst_wave_record(wave, shortfall, accepted, events[wave_events:],
                                               transports[wave_transports:], packet_text)
                             if t == 0 else {"wave": wave, "calls": shortfall, "accepted": accepted})
                shortfall -= accepted
            # The packet of the first wave is the round's packet of record.
            (out / "packets" / f"{stem}.json").write_text(wave_packets[0])
            if t == 0:
                # Every executed wave's packet; t00_burst stays wave 0's copy.
                for wave, text in enumerate(wave_packets):
                    (out / "packets" / (artifact_stem(state, f"t00_burst_w{wave}") + ".json")).write_text(text)
            accounting["burst_max_in_flight"] = max(accounting["burst_max_in_flight"], gauge["max_in_flight"])
            timings["proposal"] += time.perf_counter() - tic
            if t:
                accounting["proposal_rounds_executed"] += 1
                accounting["valid_candidates"] += len(events)
            state.fits, retired = retain(state.fits + fitted, active_maximum)
            if not state.fits:
                raise ValueError("no models remain after the parallel proposal wave")
            new_ids = sorted({f.hypothesis.model_id for f in state.fits} - set(previous_ids),
                             key=lambda key: rng_identity(state.hypotheses[key]))
            if t == 0:
                accounting["burst_admitted"] += len(new_ids)
                accounting["burst_wave_count"] += len(waves)
                if proposer is None:
                    mock_schedule["burst_used"] = gauge["mock_offset"]
            stamp_active(state, t)
            for result in state.fits:
                mark_active(state, result.hypothesis, t)
            retire(state, retired, t, state.fits[-1].log_evidence)
            counts = wave_event_counts(events, transports, count)
            for key, value in counts.items():
                if key != "duplicates":
                    accounting[key] += value
            for kind, value in failure_class_counts(events, transports).items():
                accounting["failure_class_counts"][kind] += value
            write_final_proposal_json(
                out / "proposals" / f"{stem}.json",
                {"model_id_schema": MODEL_ID_SCHEMA, "notation_normal_form": NOTATION_NORMAL_FORM,
                 "public_observations": public, "mock": proposer is None, "transports": transports,
                 "cell": args.cell, **({"component": state.name} if multi else {}),
                 "raw_response": [{k: v for k, v in e.items() if k in ("expression", "family", "parameters")}
                                  for e in events],
                 "events": [{**e, "model_id_preimage": preimage}
                            for e, preimage in zip(events, event_preimages, strict=True)],
                 "valid_candidate_count": len(events),
                 "duplicate_count": sum(e["status"] == "duplicate_not_refitted" for e in events),
                 "fit_attempt_count": fit_attempt_count, "fit_success_count": len(fitted),
                 "waves": waves, "requested": count, **counts,
                 "max_in_flight": gauge["max_in_flight"], "concurrency": burst_concurrency,
                 **({"initial_catalogue": initial_catalogue,
                     "initial_burst_refill_policy": INITIAL_BURST_REFILL_POLICY,
                     "base_prompt_sha256": refill_identity["base_prompt_sha256"],
                     "burst_refill_paragraph_sha256": refill_identity["burst_refill_paragraph_sha256"]}
                    if t == 0 else {}),
                 "before_ids": previous_ids, "after_ids": [f.hypothesis.model_id for f in state.fits],
                 "admitted_ids": new_ids, "retired_ids": [f.hypothesis.model_id for f in retired]},
            )
            return new_ids

        # (component, round) -> the model_ids the admission hook refused on the prior
        # probe in that round's LLM call (empty for the mock path and the no-proposal cell 00); the round record lists them.
        probe_refused = {}

        def propose_for(state, t, packet_text, digest, known):
            """One round's candidate list and transport record for ONE component."""
            if args.cell[:2] == "00":
                # Cell 00 (structure and family fixed) forbids proposals outright (the same rule as
                # CORE): unchanged since the first version of this adapter, and never reaches a catalog
                # check or the proposer - mock or anthropic - regardless of
                # --proposer, exactly as CORE's own proposal_due gate never calls
                # its proposer for this cell either.
                return [], {
                    "provider": "none",
                    "mock": True,
                    "api_request_attempted": False,
                    "policy": "P1",
                    "status": "no_candidate",
                    "calls": 0,
                    "repairs": 0,
                    "rejected_candidates": [],
                    "note": "",
                    **prompt_identity(),
                    **digest,
                }
            # Same rule as core_count.candidate_catalog: S0 (structure
            # fixed) with the family menu open is a finite catalog - this
            # task's own initial structures times its own allowed families,
            # built with the same `make` the initial pool and the mock
            # candidates use, so the ids are directly comparable to `known`.
            # Structure expansion (cell[0] == "1") is always unbounded, as in
            # CORE.
            catalog = (
                None
                if args.cell[0] == "1"
                else {
                    make(form, family).model_id: (form, family)
                    for form in names
                    for family in allowed
                }
            )
            if catalog is not None and set(catalog) <= known:
                catalog_exhausted_rounds.append(t)
                accounting["catalog_exhausted"] += 1
                return [], {
                    "provider": "none",
                    "mock": proposer is None,
                    "api_request_attempted": False,
                    "policy": "P1",
                    "status": "catalog_exhausted",
                    "calls": 0,
                    "repairs": 0,
                    "rejected_candidates": [],
                    "note": "",
                    "catalog_size": len(catalog),
                    **composed_prompt_identity(proposal_limit),
                    **digest,
                }
            if proposer is None:
                # k = 1 takes the one template entry this
                # round has always taken; k > 1 takes the next k entries of the
                # same fixed schedule, so a mock run of a k > 1 cell exercises
                # the multi-candidate admission path with no new schedule.
                # At k > 1 the k entries of round t follow those of round
                # t - 1 and those a round-0 burst used, so every mock round offers
                # k fresh entries; k = 1 is the (t - 1) entry as always.
                index = ((t - 1) if proposal_limit == 1
                         else mock_schedule["burst_used"] + (t - 1) * proposal_limit) % len(templates)
                candidates = [
                    make(
                        templates[(index + offset) % len(templates)],
                        families[(index + offset) % len(templates)]
                        if args.cell[1] == "1" else initial_family,
                    )
                    for offset in range(proposal_limit)
                ]
                review_block = {}
                if t in review_rounds:
                    # Archive review under the mock: the schedule has no LLM, so the review
                    # reply is synthesized as one valid id (the first retired_by_evidence
                    # archive entry, if any), then
                    # the refusable ids - an active pool id, an unknown id and, when the archive has
                    # one, an entry retired for another reason - rotated by the round so that a
                    # different refusal lands in the second place each review round (the third and
                    # later places are over_limit whatever they name), and a second retired id last.
                    # At most one model is refit per review round.
                    retired = sorted(key for key, entry in state.archive.items()
                                     if entry["outcome"] == "retired_by_evidence")
                    not_retired = sorted(key for key, entry in state.archive.items()
                                         if entry["outcome"] not in ("retired_by_evidence", "active"))
                    active = sorted(f.hypothesis.model_id for f in state.fits)
                    others = active[:1] + ["mock_invalid_model_id"] + not_retired[:1]
                    shift = t % len(others)
                    review_block = {"revive_requested": retired[:1] + others[shift:] + others[:shift]
                                    + retired[1:2],
                                    "reply_shape": "object"}
                return candidates, {
                    "provider": "none",
                    "mock": True,
                    "api_request_attempted": False,
                    "policy": "P1",
                    "status": "parsed",
                    "calls": 0,
                    "repairs": 0,
                    "rejected_candidates": [],
                    "note": "",
                    **composed_prompt_identity(proposal_limit, archive_review=t in review_rounds),
                    **review_block,
                    **digest,
                }
            admit = make_admit(state, t, known)

            proposals, transport_record = proposer.propose_one(
                packet_text,
                context={
                    "run_id": str(out),
                    "world_seed": world_seed,
                    "cell": args.cell,
                    # boxinggym has no replicate axis (unlike CORE's
                    # --replicate); propose_one requires this context key,
                    # so it is fixed at 0 and recorded verbatim for
                    # provenance, never used to derive a seed here.
                    "proposal_rep": 0,
                    "round": t,
                    # A task with several declared output components
                    # calls this once per component in the SAME round, and the
                    # transport records are write-once, so the component has to
                    # be part of the identity or the second component of a
                    # round collides on `round_NN.request.json`. NEVER sent for
                    # a single-outcome task: its context key set, its record
                    # names and its record contents stay byte-for-byte what the
                    # recorded reference death run wrote (the replay tests compare them).
                    **({"component": state.name} if multi else {}),
                },
                initial_family=initial_family,
                initial_structures=tuple(make(form, initial_family) for form in names),
                repair_attempts=1,
                allowed_families=allowed,
                admit=admit,
                proposal_limit=proposal_limit,
                max_rounds=args.rounds,
                archive_review=t in review_rounds,
            )
            probe_refused[(state.name, t)] = list(admit.rejected)
            # Guard: on a multi-output task the record MUST name the
            # component it belongs to before this run copies it into
            # proposals/*.json, failures and summary.json. `propose_one` puts
            # the whole context into `identity` and into the record, so a
            # missing component means the identity was dropped on the way -
            # and two components of one round would then share one record
            # name. Refuse here rather than write an ambiguous artifact.
            if multi and transport_record.get("component") != state.name:
                raise ValueError(
                    f"transport record for round {t} does not carry component {state.name!r}"
                )
            if not exhaustive:
                # count_proposer always builds a count_models.Hypothesis
                # (its family check is widened to
                # IMPLEMENTED_FAMILIES so a non-CORE family can be admitted
                # at all - see count_models.py); the scalar/real fit and
                # prior-probe path below needs the SAME class every
                # mock/initial candidate on this task already has - `make`'s
                # own three-way dispatch (exhaustive / real_outcome
                # / else) - because ScalarHypothesis/RealHypothesis's
                # noise_parameters/as_dict, not Hypothesis's, know their own
                # families. model_id is unaffected (canonical_model_id hashes
                # family+expression+parameters, not the class), so this is a
                # pure representation change, not a new candidate (originally
                # paired 1:1 with `bounded`/ScalarHypothesis; now following the
                # exhaustive/real_outcome/ScalarHypothesis three-way split).
                proposals = [
                    RealHypothesis(
                        candidate.expression,
                        candidate.family,
                        candidate.parameters,
                        candidate.input_names,
                        default_scale_expression(interface)
                        if candidate.family == "heteroscedastic_normal"
                        else None,
                    )
                    if real_outcome
                    else ScalarHypothesis(
                        candidate.expression,
                        candidate.family,
                        candidate.parameters,
                        candidate.input_names,
                    )
                    for candidate in proposals
                ]
            accounting["llm_calls"] += transport_record["calls"]
            accounting["llm_repairs"] += transport_record["repairs"]
            if transport_record["status"] != "parsed":
                failures.append(
                    {
                        "round": t,
                        "status": "proposal_failed",
                        **({"component": state.name} if multi else {}),
                        "transport": transport_record,
                    }
                )
            return proposals, transport_record

        # The round in progress, for a summary the paid-call guard may have to write.
        current_round = 0
        try:
            for row, handle in d0_designs:
                public.append({**observation_row(row, handle), "y": environment.observe(handle)})
                append_json(out / "observations.jsonl", {"round": 0, **public[-1]})
            if real_outcome:
                # The paper's prior table's output-scale row is written for STANDARDISED
                # outputs ("| output scale $\sigma$ (standardised outputs) |
                # HalfNormal$(1)$ |"), so the identity link is calibrated by a
                # frozen D0 location AND scale: mu = center + unit * expression and
                # sigma = unit * s. Both are frozen before round 1 and never
                # re-estimated, so every model in a pool shares one measure.
                #
                # MULTI_COMPONENT_CALIBRATION_RULE: with several output
                # components each keeps its own D0 mean as its center, and ONE
                # pooled D0 sample sd over every component is the single
                # task-specific scale Table 2's last row asks for. A
                # single-component task is unchanged (its own mean,
                # its own sd).
                pooled = np.asarray(
                    [component_y(row, state) for state in states for row in public], dtype=float
                )
                shared_unit = float(pooled.std(ddof=1))
                if not np.isfinite(shared_unit):
                    raise ValueError("D0 response calibration is not finite; no fallback")
                if shared_unit <= 0:
                    raise ValueError("D0 response scale is not positive; no fallback")
                for state in states:
                    values = np.asarray([component_y(row, state) for row in public], dtype=float)
                    state.response_center = float(values.mean())
                    state.response_unit = (
                        shared_unit if multi else float(values.std(ddof=1))
                    )
                    if not np.isfinite(state.response_center):
                        raise ValueError("D0 response calibration is not finite; no fallback")
                    if not np.isfinite(state.response_unit) or state.response_unit <= 0:
                        raise ValueError("D0 response scale is not positive; no fallback")
                write_json(
                    out / "frozen_d0_policy.json",
                    {
                        "response_center": states[0].response_center,
                        "response_unit": states[0].response_unit,
                        "policy": "frozen_D0_mean_and_sample_sd_of_the_four_initial_observations",
                        "prior_recalibration_after_d0": False,
                    }
                    if not multi
                    else {
                        "policy": MULTI_COMPONENT_CALIBRATION_RULE,
                        "prior_recalibration_after_d0": False,
                        "pooled_response_unit": shared_unit,
                        "per_component": {
                            state.name: {
                                "response_center": state.response_center,
                                "response_unit": state.response_unit,
                            }
                            for state in states
                        },
                    },
                )
            elif not exhaustive:
                zero_count_applied = False
                for state in states:
                    values = np.asarray([component_y(row, state) for row in public], dtype=float)
                    state.response_unit, applied = d0_response_unit(values, zero_count_unit)
                    zero_count_applied = zero_count_applied or applied
                # The option and whether it changed the unit are recorded only when the
                # option is stated, so a `refuse` run's files keep their bytes.
                zero_count_record = ({"d0_zero_count_unit": zero_count_unit,
                                      "zero_count_unit_applied": zero_count_applied}
                                     if zero_count_unit != "refuse" else {})
                write_json(
                    out / "frozen_d0_policy.json",
                    {
                        "response_unit": states[0].response_unit,
                        "policy": "frozen_D0_mean_of_the_four_initial_observations",
                        "prior_recalibration_after_d0": False,
                        **zero_count_record,
                    }
                    if not multi
                    else {
                        "policy": "frozen_D0_mean_of_the_four_initial_observations_per_component",
                        "prior_recalibration_after_d0": False,
                        "per_component": {
                            state.name: {"response_unit": state.response_unit} for state in states
                        },
                        **zero_count_record,
                    },
                )
                if zero_count_record:
                    config.update(zero_count_record)
                    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            for state in states:
                if initial_catalogue == "off":
                    # The round-0 pool is the burst's candidates alone.
                    state.fits = []
                    continue
                state.fits = ([refit(state, make(form, initial_family), 0) for form in names]
                              if fit_timeout_seconds is None and fit_guards == "off" else
                              fit_survivors(state, [make(form, initial_family) for form in names], 0))
                stamp_active(state, 0)
                for result in state.fits:
                    mark_active(state, result.hypothesis, 0)
            if initial_proposals:
                for state in states:
                    parallel_wave(state, 0, initial_proposals)
            pre_row, signature = checkpoint(0, "initial", "initial")
            designs = pool.designs
            # Set once, read
            # after the loop below (`break` on exhaustion leaves it non-None,
            # falling through the loop normally leaves it None).
            pool_exhausted_reason = None

            def design_search_round(rows, t, acquisition_seed, pool_x):
                """Continuous design search, generalised to every EIG-family arm: one round's design search for the arm's own
                criterion `design_search_objective`, started from the pool table `rows` just
                computed and scored by the same per-component call the pool path made for that
                arm (components summed as the pool tables are). On an exhaustive task the pool
                argmax `pool_x` is the search's answer. Returns the observed design and the
                record fields added to acquisition/tNN.json."""
                column = design_search_objective
                if design_search_mode == "exhaustive":
                    return pool_x, {"pool_chosen": pool_x, "design_search": bg_design_search.exhaustive_record(
                        column, rows, pool_x, design_search_reason)}

                def objective(candidates):
                    """Sum the arm's criterion over the components at each candidate design."""
                    candidates = np.asarray(candidates, dtype=float)
                    scored_designs = candidates[:, 0] if single_input else candidates
                    total = np.zeros(len(candidates))
                    for state in states:
                        component_seed = (acquisition_seed if not multi else
                                          seed_for("boxinggym_acquisition", spec.name, args.seed, t, state.name))
                        if column == "epig":
                            # the arm's own call: the run's fixed targets, the component's fixed discrete path
                            _, scored = choose_epig(state.fits, scored_designs, component_seed, eig_samples,
                                                    interface, interface["trials"], target_points=epig_targets,
                                                    discrete_path_state=epig_path_states[state.name])
                            total += np.asarray([float(r["epig"]) for r in scored])
                        elif exhaustive:
                            _, scored = count_choose(state.fits, scored_designs, component_seed, eig_samples,
                                                     interface, interface["trials"])
                            total += np.asarray([float(r[column]) for r in scored])
                        else:
                            # joint_eig's Monte Carlo draws depend on the batch, so every design is
                            # scored in its own call under the round seed (common random numbers);
                            # the EIG parts come from the same draws (decompose=True).
                            total += np.asarray([
                                float(joint_eig(state.fits, scored_designs[i:i + 1], component_seed,
                                                eig_samples, decompose=column != "eig")[column][0])
                                for i in range(len(candidates))])
                    return total

                pool_matrix = np.asarray([np.atleast_1d(np.asarray(r["design"], dtype=float)) for r in rows])
                chosen_row, _, record = bg_design_search.search(
                    search_space, objective, pool_matrix, [float(r[column]) for r in rows],
                    seed_for(bg_design_search.SEARCH_STREAM, spec.name, args.seed, t), objective_name=column)
                return design_value(chosen_row if not single_input else chosen_row[0]), {
                    "pool_chosen": pool_x, "design_search": record}

            # Written into the arms' `policy` record when the search is stated - that record
            # describes the decision over the pool table, while the observed design is the top-level
            # `chosen` (the search's), with the pool's own choice beside it as `pool_chosen`.
            pool_policy_scope = ({"decision_scope": "pool_table: chosen_design, chosen_index and chosen_value "
                                                    "are the pool argmax (pool_chosen); the observed design is "
                                                    "the top-level chosen (design_search.chosen)"}
                                 if design_search is not None else {})

            for t in range(1, args.rounds + 1):
                current_round = t
                tic = time.perf_counter()
                acquisition_seed = seed_for("boxinggym_acquisition", spec.name, args.seed, t)
                if spec.design_exclusion == "observed":
                    # Remove every already-observed design (D0's,
                    # then each prior round's) from the candidate SET before
                    # EITHER arm draws - re-observing one is defined to carry
                    # no new information (survival's design IS the patient
                    # ID; irt's response matrix is fixed at world-build time,
                    # irt.py:305-:340 and :403-:405). A
                    # `design_exclusion="none"` task never reaches this
                    # branch, so `designs` stays the full pool set once
                    # before round 1, exactly as before this rule existed.
                    #
                    # `available` raises once the pool
                    # cannot offer a fresh design any more (see
                    # DESIGN_POOL_EXHAUSTION_RULE) - caught here, ending the
                    # round loop after round t-1 instead of crashing the run.
                    try:
                        designs = pool.available(observed_pool_indices)
                    except ValueError as exc:
                        pool_exhausted_reason = (
                            f"round {t}: pool of {len(pool.rows)} designs exhausted after "
                            f"{len(observed_pool_indices)} distinct designs already observed "
                            f"({exc})"
                        )
                        break
                acquisition_started = time.perf_counter()
                if design_policy == "space_filling":
                    x, rows, policy_record = select_space_filling(
                        designs, [row["x"] for row in public], config["design_normalization_bounds"])
                    acquisition_elapsed = time.perf_counter() - acquisition_started
                    acquisition_count = len(rows)
                    write_json(out / "acquisition" / f"t{t:02d}.json", {
                        **candidate_metadata, "seed": acquisition_seed, "chosen": x,
                        "rows": rows, "target": design_policy, "policy": policy_record,
                    })
                elif design_policy == "point_disagreement":
                    objectives = []
                    for state in states:
                        if exhaustive:
                            prediction = count_predict(state.fits, designs, interface, interface["trials"])
                            means = np.exp(prediction["model_logpmfs"]) @ np.arange(interface["trials"] + 1)
                            objectives.append(point_disagreement(means, np.exp(prediction["model_logweights"])))
                        else:
                            objectives.append(scalar_point_disagreement(state.fits, designs))
                    total = np.sum(objectives, axis=0)
                    rows = [{"design": design_value(d), "point_disagreement": float(value)}
                            for d, value in zip(designs, total, strict=True)]
                    decision = choose_design(rows, "point_disagreement")
                    x = decision["chosen_design"]
                    acquisition_elapsed = time.perf_counter() - acquisition_started
                    acquisition_count = len(rows)
                    policy_record = {
                        "design_policy": design_policy, "objective_column": design_policy,
                        "target": design_policy, **decision,
                        "chosen_index": next(i for i, row in enumerate(rows) if row["design"] == x),
                        "chosen_value": decision["top_value"], "criteria": rows,
                        "criteria_columns": [design_policy], "definition": POINT_DISAGREEMENT_DEFINITION,
                        "posterior_particles": "all_retained", "uses_rng": False,
                    }
                    if multi:
                        policy_record.update(
                            component_rule="sum_of_component_point_disagreements",
                            component_objectives=[values.tolist() for values in objectives])
                    write_json(out / "acquisition" / f"t{t:02d}.json", {
                        **candidate_metadata, "seed": acquisition_seed, "chosen": x,
                        "rows": rows, "target": design_policy, "policy": policy_record,
                    })
                elif design_policy == "epig":
                    component_tables, component_seeds = [], {}
                    for state in states:
                        component_seed = (acquisition_seed if not multi else
                                          seed_for("boxinggym_acquisition", spec.name, args.seed, t, state.name))
                        component_seeds[state.name] = component_seed
                        _, epig_rows = choose_epig(state.fits, designs, component_seed, eig_samples,
                                                  interface, interface["trials"], target_points=epig_targets,
                                                  discrete_path_state=epig_path_states[state.name])
                        component_tables.append(epig_rows)
                    rows = component_tables[0]
                    if multi:
                        rows = [{"design": row["design"],
                                 "epig": float(sum(table[i]["epig"] for table in component_tables)),
                                 "per_component_epig": [table[i]["epig"] for table in component_tables]}
                                for i, row in enumerate(rows)]
                    x = pool_x = choose_design(rows, "epig")["chosen_design"]
                    search_fields = {}
                    if design_search is not None:
                        x, search_fields = design_search_round(rows, t, acquisition_seed, pool_x)
                    acquisition_elapsed = time.perf_counter() - acquisition_started
                    acquisition_count = len(rows)
                    if search_fields:
                        acquisition_count += search_fields["design_search"].get("evaluations", 0)
                    # the policy record is the pool table's decision (pool_x); the observed design is `chosen`
                    record = epig_acquisition_record(pool_x, rows, acquisition_seed)
                    if search_fields:
                        record["chosen"] = x
                        record["policy"].update(pool_policy_scope)
                        record.update(search_fields)
                    record.update(candidate_metadata)
                    if multi:
                        record.update(component_seeds=component_seeds,
                                      multi_component_epig_rule="sum_of_independent_component_predictive_information")
                        component_exclusions = {
                            state.name: table.exclusion_info
                            for state, table in zip(states, component_tables, strict=True)
                            if getattr(table, "exclusion_info", None)
                        }
                        if component_exclusions:
                            record["epig_component_exclusions"] = component_exclusions
                        component_boundary_repairs = {
                            state.name: table.boundary_repairs
                            for state, table in zip(states, component_tables, strict=True)
                            if getattr(table, "boundary_repairs", None)
                        }
                        if component_boundary_repairs:
                            record["epig_component_boundary_repairs"] = component_boundary_repairs
                    write_json(out / "acquisition" / f"t{t:02d}.json", record)
                elif design_policy in EIG_PART_POLICIES:
                    # EIG-part arms: the same tables as the joint arm (same seeds, same draws), argmax on one part
                    column, target, _ = EIG_PART_POLICIES[design_policy]
                    tables = []
                    for state in states:
                        component_seed = (
                            acquisition_seed
                            if not multi
                            else seed_for("boxinggym_acquisition", spec.name, args.seed, t, state.name)
                        )
                        if exhaustive:
                            _, rows = count_choose(
                                state.fits, designs, component_seed, eig_samples, interface,
                                interface["trials"],
                            )
                        else:
                            table = joint_eig(state.fits, designs, component_seed, eig_samples, decompose=True)
                            rows = [
                                {"design": design_value(d), "eig": float(e), "structure_eig": float(s),
                                 "parameter_eig": float(p)}
                                for d, e, s, p in zip(designs, table["eig"], table["structure_eig"],
                                                      table["parameter_eig"], strict=True)
                            ]
                        tables.append(rows)
                    rows = tables[0]
                    if multi:
                        keys = ("eig", "structure_eig", "parameter_eig")
                        rows = [
                            {"design": row["design"],
                             **{k: float(sum(float(table[i][k]) for table in tables)) for k in keys},
                             **{f"per_component_{k}": [float(table[i][k]) for table in tables] for k in keys}}
                            for i, row in enumerate(rows)
                        ]
                    decision = choose_design(rows, column)
                    x = pool_x = decision["chosen_design"]
                    search_fields = {}
                    if design_search is not None:
                        x, search_fields = design_search_round(rows, t, acquisition_seed, pool_x)
                    acquisition_elapsed = time.perf_counter() - acquisition_started
                    acquisition_count = len(rows)
                    if search_fields:
                        acquisition_count += search_fields["design_search"].get("evaluations", 0)
                    # the policy record is the pool table's decision (pool_x); the observed design is `chosen`
                    chosen_index = next(i for i, row in enumerate(rows) if row["design"] == pool_x)
                    write_json(out / "acquisition" / f"t{t:02d}.json", {
                        **candidate_metadata, "seed": acquisition_seed,
                        **({"component_seeds": {
                            state.name: seed_for("boxinggym_acquisition", spec.name, args.seed, t, state.name)
                            for state in states}} if multi else {}),
                        "chosen": x, "rows": rows,
                        "target": target,
                        "policy": {
                            "design_policy": design_policy, "objective_column": column, "target": target,
                            **decision, "chosen_index": chosen_index,
                            "chosen_value": decision["top_value"],
                            "joint_eig_at_chosen": float(rows[chosen_index]["eig"]),
                            "joint_eig_max": max(float(r["eig"]) for r in rows),
                            "definition": EIG_PART_DEFINITION[design_policy],
                            "estimator": EIG_PART_ESTIMATOR["exhaustive" if exhaustive else "scalar"],
                            **pool_policy_scope,
                        },
                        **search_fields,
                    })
                elif args.cell[2] == "1":
                    tables = []
                    for state in states:
                        component_seed = (
                            acquisition_seed
                            if not multi
                            else seed_for(
                                "boxinggym_acquisition", spec.name, args.seed, t, state.name
                            )
                        )
                        if exhaustive:
                            chosen, rows = count_choose(
                                state.fits, designs, component_seed, eig_samples, interface,
                                interface["trials"],
                            )
                            tables.append((chosen, rows))
                        else:
                            table = joint_eig(state.fits, designs, component_seed, eig_samples)
                            rows = [
                                {"design": design_value(d), "eig": float(v)}
                                for d, v in zip(designs, table["eig"], strict=True)
                            ]
                            tables.append((choose_design(rows)["chosen_design"], rows))
                    if not multi:
                        x, rows = tables[0]
                        search_fields = {}
                        if design_search is not None:
                            x, search_fields = design_search_round(rows, t, acquisition_seed, x)
                        acquisition_elapsed = time.perf_counter() - acquisition_started
                        acquisition_count = len(rows)
                        if search_fields:
                            acquisition_count += search_fields["design_search"].get("evaluations", 0)
                        write_json(
                            out / "acquisition" / f"t{t:02d}.json",
                            {
                                **candidate_metadata,
                                "seed": acquisition_seed,
                                "chosen": x,
                                "rows": rows,
                                "target": "joint_model_parameter",
                                **search_fields,
                            },
                        )
                    else:
                        # MULTI_COMPONENT_EIG_RULE: one design serves every
                        # component, so the round's objective is the SUM of the
                        # per-component EIG tables - the joint EIG of the
                        # independently modelled components. The same tie rule
                        # (eig.joint.choose_design) then picks the design.
                        total = np.sum(
                            np.asarray([[float(r["eig"]) for r in rows] for _, rows in tables]),
                            axis=0,
                        )
                        rows = [
                            {
                                "design": tables[0][1][index]["design"],
                                "eig": float(value),
                                "per_component_eig": [
                                    float(rows_c[index]["eig"]) for _, rows_c in tables
                                ],
                            }
                            for index, value in enumerate(total)
                        ]
                        x = choose_design(rows)["chosen_design"]
                        search_fields = {}
                        if design_search is not None:
                            x, search_fields = design_search_round(rows, t, acquisition_seed, x)
                        acquisition_elapsed = time.perf_counter() - acquisition_started
                        acquisition_count = len(rows)
                        if search_fields:
                            acquisition_count += search_fields["design_search"].get("evaluations", 0)
                        write_json(
                            out / "acquisition" / f"t{t:02d}.json",
                            {
                                **candidate_metadata,
                                "seed": acquisition_seed,
                                "component_seeds": {
                                    state.name: seed_for(
                                        "boxinggym_acquisition", spec.name, args.seed, t, state.name
                                    )
                                    for state in states
                                },
                                "chosen": x,
                                "rows": rows,
                                "target": "joint_model_parameter",
                                "multi_component_eig_rule": MULTI_COMPONENT_EIG_RULE,
                                **search_fields,
                            },
                        )
                else:
                    # The same single rng call as on a single-input pool; on a multi-input pool
                    # `Generator.choice` samples a ROW of the design matrix.
                    x = design_value(np.random.default_rng(acquisition_seed).choice(designs))
                    acquisition_elapsed = time.perf_counter() - acquisition_started
                    acquisition_count = len(designs)
                    write_json(
                        out / "acquisition" / f"t{t:02d}.json",
                        {**candidate_metadata, "seed": acquisition_seed, "chosen": x, "target": "random"},
                    )
                if record_criteria:
                    # Joint, Model and Parameter EIG and the point disagreement of every
                    # candidate, as a record. The EIG table is the arm's own when it already carries
                    # the three columns (the joint arm on an exhaustive task, the two EIG-part arms);
                    # otherwise it is the joint arm's estimator on this round's acquisition seed.
                    criteria_state = states[0]
                    arm_table = (rows if args.cell[2] == "1" and design_policy in ("joint_eig", *EIG_PART_POLICIES)
                                 and rows and "structure_eig" in rows[0] else None)
                    if arm_table is not None:
                        criteria_table, criteria_source = arm_table, "arm_table"
                    elif exhaustive:
                        _, criteria_table = count_choose(criteria_state.fits, designs, acquisition_seed,
                                                         eig_samples, interface, interface["trials"])
                        criteria_source = "recomputed_joint_scheme"
                    else:
                        parts = joint_eig(criteria_state.fits, designs, acquisition_seed, eig_samples, decompose=True)
                        criteria_table = [
                            {"design": design_value(d), "eig": float(e), "structure_eig": float(s),
                             "parameter_eig": float(p)}
                            for d, e, s, p in zip(designs, parts["eig"], parts["structure_eig"],
                                                  parts["parameter_eig"], strict=True)
                        ]
                        criteria_source = "recomputed_joint_scheme"
                    if exhaustive:
                        prediction = count_predict(criteria_state.fits, designs, interface, interface["trials"])
                        means = np.exp(prediction["model_logpmfs"]) @ np.arange(interface["trials"] + 1)
                        disagreement = point_disagreement(means, np.exp(prediction["model_logweights"]))
                    else:
                        disagreement = scalar_point_disagreement(criteria_state.fits, designs)
                    write_json(out / "acquisition" / f"t{t:02d}_criteria.json", {
                        "round": t, "seed": acquisition_seed, "eig_posterior_samples": eig_samples,
                        "eig_source": criteria_source, "arm": design_policy if args.cell[2] != "0" else "random",
                        "chosen": x, "n_candidates": len(criteria_table),
                        "rows": [
                            {"design": row["design"], "eig": float(row["eig"]),
                             "structure_eig": float(row["structure_eig"]), "parameter_eig": float(row["parameter_eig"]),
                             "point_disagreement": float(value)}
                            for row, value in zip(criteria_table, disagreement, strict=True)
                        ],
                        **DESIGN_CRITERIA_DEFINITIONS,
                    })
                timings["eig"] += time.perf_counter() - tic
                record_acquisition_timing(out, t, design_policy if args.cell[2] != "0" else "random",
                                          acquisition_elapsed, acquisition_count)
                if design_search_mode != "ascent":
                    # the pool path, and the exhaustive side of the design-search rule, whose
                    # choice is a pool row: its handle, and its exclusion where the task
                    # removes observed designs
                    chosen_index = pool.index_of(x)
                    if spec.design_exclusion == "observed":
                        observed_pool_indices.add(chosen_index)
                    chosen_row, chosen_handle = pool.rows[chosen_index], pool.handles[chosen_index]
                else:
                    # Continuous design search: the chosen design is env-valid by construction (SearchSpace.valid) but
                    # need not be a pool row; its handle is the same encoder the box pools use.
                    chosen_row = np.atleast_1d(np.asarray(x, dtype=float))
                    chosen_handle = encode_design(spec, chosen_row)
                y = environment.observe(chosen_handle)
                one_step_rows = {}
                for state in states:
                    for history in state.one_step.values():
                        history.append(None)
                    row_scores = {}
                    for result in state.fits:
                        key = result.hypothesis.model_id
                        value = one_step_score(
                            state, result, x, y[state.index] if multi else y
                        )
                        state.one_step[key][t - 1] = value
                        row_scores[key] = value
                    one_step_rows[state.name] = row_scores
                public.append(
                    {
                        **observation_row(chosen_row, chosen_handle),
                        "y": [float(v) for v in y] if multi else float(y),
                    }
                )
                append_json(out / "observations.jsonl", {"round": t, **public[-1]})
                for state in states:
                    state.fits = ([refit(state, previous.hypothesis, t) for previous in state.fits]
                                  if fit_timeout_seconds is None and fit_guards == "off" else
                                  fit_survivors(state, [previous.hypothesis for previous in state.fits], t))
                    stamp_active(state, t)
                pre_row, signature = checkpoint(t, "before_proposal", "pre_proposal")
                pre_row["one_step_log_score"] = (
                    one_step_rows[states[0].name] if not multi else one_step_rows
                )
                for state in states:
                    if parallel_proposals and args.cell[:2] != "00":
                        # The round as K parallel single-candidate calls,
                        # deduplicated and refilled once, under the same gates.
                        admissions += len(parallel_wave(state, t, parallel_proposals))
                        continue
                    # The same timed window as always: the packet build plus the
                    # proposal itself, NOT the candidate fits below (those are
                    # already in timings["fit"] via `refit`). Per component, so
                    # the total is the sum over components.
                    tic = time.perf_counter()
                    packet = build_belief_packet(
                        [
                            {
                                "obs_id": index,
                                "round": 0 if index < 4 else index - 3,
                                **dict(
                                    zip(
                                        spec.input_names,
                                        [row["x"]] if single_input else row["x"],
                                        strict=True,
                                    )
                                ),
                                "y": component_y(row, state),
                            }
                            for index, row in enumerate(public)
                        ],
                        pool_members(state, t),
                        list(state.archive.values()),
                        constraints,
                        interface,
                        t,
                    )
                    packet_text = serialize_packet(packet)
                    digest = packet_digest(packet_text)
                    (out / "packets" / f"{artifact_stem(state, f't{t:02d}')}.json").write_text(
                        packet_text
                    )
                    known = set(state.archive) | {f.hypothesis.model_id for f in state.fits}
                    proposals, transport_record = propose_for(state, t, packet_text, digest, known)
                    timings["proposal"] += time.perf_counter() - tic
                    accounting["proposal_rounds_executed"] += 1
                    accounting["valid_candidates"] += len(proposals)
                    # Appended AFTER the proposer-performance accounting above
                    # (excluded from valid_candidates/
                    # proposal_rounds_executed), so this candidate never
                    # inflates a proposer-performance metric, but BEFORE
                    # `seen`/the per-candidate loop below, so it passes through
                    # every normal gate those give every other candidate -
                    # registration, the prior probe, the duplicate check and
                    # `retain`'s evidence-based cap (no exemption). Works in every
                    # cell, including 000/001:
                    # `propose_for` already returned (possibly empty)
                    # `proposals` above regardless of cell.
                    injected_hypothesis = None
                    if args.inject_oracle_at == t:
                        injected_hypothesis = Hypothesis(
                            DEATH_ORACLE_EXPRESSION,
                            initial_family,
                            DEATH_ORACLE_ROLES,
                            namespace,
                            categorical,
                        )
                        proposals = [*proposals, injected_hypothesis]
                    write_proposed_json(
                        out / "proposals" / f"{artifact_stem(state, f't{t:02d}')}.json",
                        {"public_observations": public, "mock": proposer is None,
                         "transport": transport_record, "cell": args.cell,
                         **({"component": state.name} if multi else {}),
                         "raw_response": [h.as_dict() for h in proposals]},
                    )
                    events, candidates, fit_attempt_count = [], [], 0
                    # The candidates the admission hook refused on the
                    # prior probe. The hook archived each (outcome rejected, reason) and the
                    # transport record names each under rejected_candidates (failure_class
                    # prior_probe, counted there once); the proposer kept the reply's other
                    # candidates. Listed under their own key, not among `events`, so the
                    # valid / duplicate / failure-class counts are not moved or doubled.
                    probe_refusals = [
                        {"model_id": key, "expression": state.archive[key]["expression"],
                         "family": state.archive[key]["family"], "status": "rejected_prior_probe",
                         "reason": state.archive[key]["reason"]}
                        for key in probe_refused.get((state.name, t), [])]
                    # Same rule as core_count: "already seen" is the
                    # whole run history (pool plus archive), so a retired, fit-failed or
                    # probe-rejected model_id proposed again is a duplicate, never a
                    # refit, and `fit_attempt_count` counts actual fit calls only.
                    seen = set(known)
                    for hypothesis in proposals:
                        item = hypothesis.as_dict()
                        key = register(state, hypothesis, t)
                        probe = prior_probe_cache.pop(key, None)
                        if probe is None:
                            probe, _ = run_prior_probe(hypothesis, interface, state.response_unit)
                        loglik_probe = None
                        if fit_guards == "v1" and probe["passed"] and key not in seen:
                            x_probe = np.asarray([row["x"] for row in public], dtype=float)
                            y_probe = np.asarray([component_y(row, state) for row in public], dtype=float)
                            model_builder = (None if exhaustive else partial(
                                scalar_build_model, hypothesis, x_probe, y_probe, None,
                                state.response_unit, interface, state.response_center))
                            loglik_probe = loglik_prior_probe(
                                hypothesis, x_probe, y_probe, interface,
                                prior_draws=probe_particles(hypothesis, PRIOR_PROBE_SAMPLES),
                                model_builder=model_builder, n_trials=interface["trials"],
                            )
                            probe = {**probe, "loglik_prior_probe": loglik_probe}
                        # Same as core_count: the candidate's note travels through the
                        # transport record, not a hardcoded "" - a no-op for mock/
                        # exhausted rounds (their transport_record["note"] is always
                        # ""), and what makes the anthropic path's note reach
                        # candidate_meta/archive.json.
                        note = transport_record.get("note", "")
                        item["prior_probe"] = probe
                        item["note"] = note
                        # Same layer as prior_probe/note
                        # above - a per-candidate provenance fact, not a
                        # status. Absent (never False) for every ordinary
                        # candidate, exactly like the archive's own optional
                        # keys (belief_packet.ARCHIVE_OPTIONAL_KEYS).
                        if hypothesis is injected_hypothesis:
                            item["injected"] = True
                        state.candidate_meta[key] = {"prior_probe": probe, "note": note}
                        stub = {
                            "model_id": key,
                            "expression": hypothesis.expression,
                            "family": hypothesis.family,
                            "parameters": dict(hypothesis.parameters),
                            "proposed_round": state.born_round[key],
                            "prior_probe": probe,
                            "note": note,
                        }
                        # Named so the injected-round
                        # bookkeeping below can tell "ignored" (already known)
                        # from "processed" (registered/probed/fit-attempted)
                        # without re-deriving the duplicate test.
                        was_duplicate = key in seen
                        if was_duplicate:
                            item["status"] = "duplicate_not_refitted"
                            accounting["duplicates"] += 1
                            matched = next(
                                (f.hypothesis for f in state.fits if f.hypothesis.model_id == key),
                                state.hypotheses.get(key, hypothesis),
                            )
                            state.archive[duplicate_archive_key(state.archive, t, key)] = {
                                **stub,
                                "proposed_round": t,
                                "outcome": "duplicate",
                                "reason": {
                                    "matched_model_id": key,
                                    "matched_expression": matched.expression,
                                },
                            }
                        elif not probe["passed"]:
                            item["status"] = "rejected_prior_probe"
                            accounting["prior_probe_rejections"] += 1
                            seen.add(key)
                            state.archive[key] = {
                                **stub,
                                "outcome": "rejected",
                                "reason": prior_probe_reason(probe),
                            }
                        elif loglik_probe is not None and not loglik_probe["passed"]:
                            item["status"] = "rejected_loglik_prior_probe"
                            accounting["prior_probe_rejections"] += 1
                            seen.add(key)
                            state.archive[key] = {
                                **stub, "outcome": "rejected",
                                "reason": {"validation_message": loglik_probe["reason"]},
                            }
                        else:
                            seen.add(key)
                            accounting["fit_attempts"] += 1
                            fit_attempt_count += 1
                            try:
                                candidates.append(refit(state, hypothesis, t))
                            except FitTimeout as exc:
                                item.update(retire_timeout(state, hypothesis, t, exc, item))
                            except CountInferenceFailure as exc:
                                if fit_guards != "v1" or exc.cause_code != "nonfinite_smc_weights":
                                    item.update(status="fit_or_predict_failed",
                                                error=f"{type(exc).__name__}: {exc}")
                                    failures.append(item)
                                    state.archive[key] = {
                                        **stub, "outcome": "fit_failed",
                                        "reason": {"error": f"{type(exc).__name__}: {exc}"},
                                    }
                                else:
                                    item.update(retire_nonfinite_smc(state, hypothesis, t, exc, item))
                            except Exception as exc:  # noqa: BLE001 - a failed candidate, not a failed run
                                item.update(
                                    status="fit_or_predict_failed",
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                                failures.append(item)
                                state.archive[key] = {
                                    **stub,
                                    "outcome": "fit_failed",
                                    "reason": {"error": f"{type(exc).__name__}: {exc}"},
                                }
                            else:
                                accounting["fit_successes"] += 1
                                item.update(
                                    status="fitted", log_evidence=candidates[-1].log_evidence
                                )
                        # See parallel_wave - the event names its failure class.
                        item["failure_class"] = failure_class(status=item["status"])
                        events.append(item)
                        if hypothesis is injected_hypothesis:
                            (injection_ignored_rounds if was_duplicate else injected_rounds).append(
                                t
                            )
                    for kind, value in failure_class_counts(
                            events, [{"transport": transport_record}]).items():
                        accounting["failure_class_counts"][kind] += value
                    revived, review_event = [], None
                    if t in review_rounds:
                        # The reply's revive list, validated and refit; one truncation below.
                        revived, review_event = review_archive(state, t, transport_record)
                    previous_ids = [f.hypothesis.model_id for f in state.fits]
                    state.fits, retired = retain(state.fits + candidates + revived, active_maximum)
                    new_ids = sorted({f.hypothesis.model_id for f in state.fits} - set(previous_ids),
                                     key=lambda key: rng_identity(state.hypotheses[key]))
                    admissions += len(new_ids)
                    stamp_active(state, t)
                    for result in state.fits:
                        mark_active(state, result.hypothesis, t)
                    retire(state, retired, t, state.fits[-1].log_evidence)
                    if review_event is not None:
                        finish_review(state, t, review_event, revived)
                    write_final_proposal_json(
                        out / "proposals" / f"{artifact_stem(state, f't{t:02d}')}.json",
                        {
                            "public_observations": public,
                            "mock": proposer is None,
                            "transport": transport_record,
                            "cell": args.cell,
                            **({"component": state.name} if multi else {}),
                            "raw_response": [h.as_dict() for h in proposals],
                            "events": events,
                            "valid_candidate_count": len(proposals),
                            "duplicate_count": sum(
                                e["status"] == "duplicate_not_refitted" for e in events
                            ),
                            "fit_attempt_count": fit_attempt_count,
                            "fit_success_count": len(candidates),
                            # The admission hook's prior-probe refusals of this
                            # round (each also in transport.rejected_candidates and the archive).
                            "prior_probe_rejected": probe_refusals,
                            "prior_probe_rejected_count": len(probe_refusals),
                            # Stated only on an archive review round.
                            **({"archive_review": True, "revived_count": len(revived),
                                "admitted_count": len(new_ids)} if review_event is not None else {}),
                            "before_ids": previous_ids,
                            "after_ids": [f.hypothesis.model_id for f in state.fits],
                            "admitted_ids": new_ids,
                            "retired_ids": [f.hypothesis.model_id for f in retired],
                        },
                    )
                current = tuple(
                    frozenset((v.hypothesis.model_id, id(v), v.log_evidence) for v in state.fits)
                    for state in states
                )
                if current == signature:
                    row = {k: v for k, v in pre_row.items() if k != "one_step_log_score"}
                    row.update(
                        phase="after_proposal",
                        scoring_phase="post_proposal",
                        post_equals_pre=True,
                        predictions_from=pre_row["phase"],
                    )
                    if not multi:
                        row.update(
                            active_ids=[v.hypothesis.model_id for v in states[0].fits],
                            model_weights=np.exp(model_logweights(states[0].fits)).tolist(),
                        )
                    else:
                        row["components"] = [
                            {
                                **entry,
                                "active_ids": [v.hypothesis.model_id for v in state.fits],
                                "model_weights": np.exp(model_logweights(state.fits)).tolist(),
                            }
                            for entry, state in zip(row["components"], states, strict=True)
                        ]
                    record_prediction_pool_summary(out, t, "official", len(public), checkpoint_models)
                    add_recorded_pool_count(row)
                    append_json(out / "scores.jsonl", row)
                    score_rows.append(row)
                    mda_rows.append(dict(mda_rows[-1]))
                    pre_row, signature = row, current
                else:
                    pre_row, signature = checkpoint(t, "after_proposal", "post_proposal")
            own, child = (
                resource.getrusage(resource.RUSAGE_SELF),
                resource.getrusage(resource.RUSAGE_CHILDREN),
            )
            summary = {
                # "completed"
                # unless the round loop above ended early on pool exhaustion,
                # in which case this is a normal (not "failed") terminal
                # status - see DESIGN_POOL_EXHAUSTION_RULE.
                "status": "completed" if pool_exhausted_reason is None else "design_pool_exhausted",
                # Only when archive review rounds were stated.
                **({"revival_events": revival_events} if review_rounds else {}),
                "suite": "boxinggym",
                "task": spec.name,
                "profile": args.profile,
                "rounds": args.rounds,
                "observations": len(public),
                "environment_calls": environment.observation_calls,
                "env_seed_log": list(env_seed_log),
                "admissions": admissions,
                "active_count": sum(len(state.fits) for state in states),
                "candidate_failures": failures,
                "proposal_accounting": accounting,
                "catalog_exhausted_rounds": catalog_exhausted_rounds,
                "eval_question_policy": "restored question",
                "held_out_points": len(heldout),
                "env_evaluator": score_rows[-1]["env_evaluator"],
                "our_scores": score_rows[-1]["bma"],
                "timing_seconds": timings,
                "wall_seconds": time.perf_counter() - start,
                "user_cpu_seconds": own.ru_utime + child.ru_utime,
                "system_cpu_seconds": own.ru_stime + child.ru_stime,
                "paid_api_calls": 0 if proposer is None else proposer.attempts,
                "billing_status": (
                    "no_calls" if proposer is None else "usage_in_llm_transport_records"
                ),
                "registered_run": False,
            }
            accounting.update(proposal_accounting_indexes(out, score_rows))
            summary.update(wave_cost_blocks(out))
            if pool_exhausted_reason is not None:
                summary["design_pool_exhausted_reason"] = pool_exhausted_reason
            if predictive_normalization_total["checkpoints"]:
                # The run total, under the same "only when it fired"
                # condition as the per-checkpoint key, so a run that never met
                # the rule keeps its key set unchanged.
                summary["predictive_normalization"] = {
                    **predictive_normalization_total,
                    "drift_tolerance": DRIFT_TOL,
                }
            if multi:
                summary["output_components"] = list(components)
                summary["composite_score_rule"] = COMPOSITE_SCORE_RULE
                summary["per_component_scores"] = {
                    entry["component"]: entry["bma"] for entry in score_rows[-1]["components"]
                }
                summary["per_component_active_count"] = {
                    state.name: len(state.fits) for state in states
                }
            if spec.openai_narration:
                # Zero paid calls is STRUCTURAL here: the environment's narration
                # calls were absorbed by a local stub that performs no I/O, and
                # this is how many it absorbed.
                summary["openai_narration_stub"] = True
                summary["narration_calls_absorbed"] = len(narration_calls)
            if spec.design_exclusion == "observed":
                # The record that shows every observed design was distinct.
                summary["distinct_designs_observed"] = len(observed_pool_indices)
                summary["design_pool_size"] = len(pool.rows)
            if spec.environment_refused_attribute:
                summary["environment_refused_distinct"] = len(refused)
                summary["environment_refused_queries"] = 0
            if spec.name == "death":
                # Declared unconditionally
                # for death (every switch off makes both empty), the same
                # per-task-declared-field pattern as the two blocks above -
                # injection is a death-oracle-only concept, so no other
                # task's summary.json gains these keys.
                summary["injected_rounds"] = injected_rounds
                summary["injection_ignored_rounds"] = injection_ignored_rounds
            if label_explicit or args.proposer in ("openai", "openai_compat"):
                # Shared helper - see
                # `points_label` and `config["label"]` above. The
                # resolved --run-label, under the same "only when forced"
                # condition as config, so an unforced run keeps its key set
                # unchanged.
                summary["label"] = run_label
            # Same invariant as CORE's core_count.run() -
            # every duplicate EVENT (`accounting["duplicates"]`) must have
            # exactly one archive entry, summed across every component's own
            # archive (single-outcome tasks have exactly one). A bare assert:
            # it writes nothing, so a passing run's output bytes are exactly
            # what they were before this check existed.
            assert (
                sum(
                    entry["outcome"] == "duplicate"
                    for state in states
                    for entry in state.archive.values()
                )
                == accounting["duplicates"]
            ), "archive duplicate-entry count must equal duplicate-event count"
            # The top-level summary is masked too.
            recheck_manifest(label_receipt)
            if label_explicit:
                summary["label_receipt"] = label_receipt
            write_json(out / "summary.json", mask_secrets_deep(summary))

            def archive_rows(state):
                """The component's archive entries for archive.json, with preimages on admitted models."""
                # Model-identity preimage: an admitted model's entry (active or retired by evidence) carries its
                # model_id_preimage in this file only; the in-memory archive the packets read does not.
                return [
                    {**entry, "model_id_preimage": state.hypotheses[entry["model_id"]].model_id_preimage}
                    if entry["outcome"] in ("active", "retired_by_evidence") else entry
                    for entry in state.archive.values()
                ]

            write_json(
                out / "archive.json",
                archive_rows(states[0])
                if not multi
                else {state.name: archive_rows(state) for state in states},
            )
            write_points(out / "points.csv", config, spec, score_rows)
            write_mda_points(out / "points_mda.csv", mda_rows)
        except PaidCallBudgetExhausted:
            # The guard refused the call that would pass max_paid_calls. Every
            # record written so far stays; the summary says why and where, and main() exits
            # nonzero.
            write_json(out / "summary.json", mask_secrets_deep({
                **budget_exhausted_summary(proposer, current_round),
                "suite": "boxinggym",
                "task": spec.name,
                "profile": args.profile,
                "rounds": args.rounds,
                "observations": len(public),
                "timing_seconds": timings,
                "wall_seconds": time.perf_counter() - start,
                **({"label": run_label}
                   if label_explicit or args.proposer in ("openai", "openai_compat") else {}),
                **({"label_receipt": label_receipt} if label_explicit else {}),
            }))
            raise
        except BaseException as exc:  # noqa: BLE001 - recorded masked, re-raised masked
            write_json(
                out / "failure.json",
                {
                    "status": "failed",
                    "error": mask_secrets(f"{type(exc).__name__}: {exc}"),
                    "traceback": mask_secrets(traceback.format_exc()),
                    "timing_seconds": timings,
                    **({"candidate_failures": mask_secrets_deep(failures)}
                       if fit_timeout_seconds is not None else {}),
                },
            )
            # Masked ONLY when the text really carries
            # a credential - an expected numerical failure still propagates as
            # its own type with its own traceback.
            raise masked_exception(exc) from None


def points_variant(config, spec):
    """The plotting contract's `variant` column for a BoxingGym run: sorted, `+`-joined.

    Same rule as `core_count.points_variant`; the tokens differ because the
    tasks do. `composite_over_<n>_components` is the multi-output
    label, `prior_wide`/`prior_narrow` the prior-template ablation's replaced
    role-prior template and `prior_llm` the proposed-prior ablation. All of them are absent by default, so a
    single-outcome default run's column is empty, as it always was.
    """
    tokens = [] if not spec.components else [f"composite_over_{len(spec.components)}_components"]
    if PRIOR_VARIANT_TOKEN[config["prior_template"]]:
        tokens.append(PRIOR_VARIANT_TOKEN[config["prior_template"]])
    if config["priors"] == "llm":
        tokens.append(LLM_PRIORS_VARIANT_TOKEN)
    # `.get` (not `config[...]`), because a config dict from before these
    # options existed - and every existing unit test that builds one by
    # hand for this function alone - has neither key; both are absent (never
    # False/None) from that case, exactly like the composite/prior tokens
    # above for a run that never touched THEIR switches either. Reserved token
    # names (oracle_injected_k5/_k10
    # and truth_in_pool) - this module only ever emits them.
    if config.get("truth_in_pool"):
        tokens.append("truth_in_pool")
    if config.get("inject_oracle_at") is not None:
        tokens.append(f"oracle_injected_k{config['inject_oracle_at']}")
    policy = config.get("design_policy")
    if policy in ("point_disagreement", "space_filling"):
        tokens.append(variant_token(policy, True))
    if policy in EIG_PART_POLICIES:
        tokens.append(EIG_PART_POLICIES[policy][2])
    return "+".join(sorted(tokens))


# ---- points.csv plot-contract mapping -------------------------------------
# Canonical source: the plot contract (v1.3) and its collector's own
# ENV table (`COLS` = the ground truth for the 22-column
# order - it already carries "y_label" from before v1.3 and adds "code_sha" as
# its 22nd column). Applied ONLY inside `write_points` below: every internal id
# (TASKS keys, spec.name, the output directory name, config.json, summary.json,
# scores.jsonl) is unchanged - the rename/relabel happens at emission time only.
POINTS_TASK_KEY = {
    "death": "death_process",
    "location": "location_finding",
    "moral_machines": "morals",
}

# y_name: "nmse_mda" only for death (the baseline's death_process scorer is a
# noise-free-truth quantity it calls nmse_mda); every other task's y_primary is
# the BoxingGym environment's OWN `evaluate_predictions` score, y_name
# "evaluator_env" - retiring the earlier "env_<metric>" naming. y_label is
# the companion human-readable name of whatever metric that row's y_primary
# actually is.
#
# Two caveats this mapping does NOT resolve (it is a label/key change only,
# not a new score computation):
#   * death: y_primary is unchanged - still `environment.evaluate()`'s own
#     metric (death_process.py's plain, unnormalized MSE on the rounded-integer
#     prediction). Whether that is on the same normalized scale as whatever the
#     baseline's own "nmse_mda" methodology computes for death_process has not
#     been verified here; only the LABEL is aligned to the contract.
#   * survival/hyperbolic/moral_machines are `env_metric_direction ==
#     "higher_is_better"` (raw accuracy, unchanged). The baseline's own
#     collector flips ITS rows for these three task names to "1 - accuracy" so
#     every row on one axis is lower-is-better (collect_points.py's ENV table).
#     This module does NOT apply that flip, so their y_label says "accuracy",
#     not "1 - accuracy" like the baseline's text for the same task names -
#     `env_metric_direction` is already recorded per-run in config.json for a
#     future flip to key off.
POINTS_Y_LABEL = {
    "death": "nMSE vs noise-free truth on fixed Q",
    "peregrines": "held-out MSE (BoxingGym evaluator)",
    "dugongs": "held-out MSE (BoxingGym evaluator)",
    "location": "held-out MSE (BoxingGym evaluator)",
    "irt": "held-out error rate, 1 - accuracy (BoxingGym evaluator)",
    "emotion": "held-out mean absolute error (BoxingGym evaluator)",
    "survival": (
        "held-out accuracy (BoxingGym evaluator); NOT flipped to 1-accuracy, unlike the "
        "baseline collector's label for this task"
    ),
    "hyperbolic": (
        "held-out accuracy (BoxingGym evaluator); NOT flipped to 1-accuracy, unlike the "
        "baseline collector's label for this task"
    ),
    "moral_machines": (
        "held-out accuracy (BoxingGym evaluator); NOT flipped to 1-accuracy, unlike the "
        "baseline collector's label for this task"
    ),
}

# The shared core_count.POINTS_COLUMNS (21: ...source_path, code_sha, variant)
# predates the contract's own "y_label" column and orders code_sha BEFORE
# variant rather than after; core_count.py is out of scope here (CORE's
# own points.csv is untouched by this change), so BoxingGym's emitter carries its
# own column tuple in the canonical order instead of editing the shared
# constant. The
# contract is 25 columns - the 21 above, then `code_sha` (22), `brier` (23),
# `nmsle` (24), `rmsle` (25). BoxingGym rows write `brier` where `our_scores` defines it
# (the four binary-outcome tasks) and leave `nmsle`/`rmsle` blank always -
# those two are ChemBench's own auxiliary columns (nMSLE/RMSLE
# against the noise-free truth on Q), which this suite's rows do not compute.
BOXINGGYM_POINTS_COLUMNS = (
    "suite", "task", "method", "proposer", "seed", "x", "x_unit",
    "y_primary", "y_name", "y_label", "nll", "coverage90", "width90",
    "crps", "is_0.1", "kl", "status", "label", "run_id", "source_path",
    "variant", "code_sha", "brier", "nmsle", "rmsle",
)


def write_points(path, config, spec, rows):
    """The plotting contract (`BOXINGGYM_POINTS_COLUMNS`), one row per score row."""
    import csv

    task_key = POINTS_TASK_KEY.get(spec.name, spec.name)
    y_name = "nmse_mda" if spec.name == "death" else "evaluator_env"
    y_label = POINTS_Y_LABEL[spec.name]
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(*BOXINGGYM_POINTS_COLUMNS, *compat_points_metadata(config)), lineterminator="\n")
        writer.writeheader()
        for line, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    **compat_points_metadata(config),
                    "suite": "boxinggym",
                    "task": task_key,
                    "method": config["cell"],
                    "proposer": (
                        "mock"
                        if "mock" in config["proposer"]
                        else f"llm:{config['llm_model']}"
                    ),
                    "seed": config["seed"],
                    "x": row["observation_count"],
                    "x_unit": "observations",
                    # The paper's Table 6b: death's y_primary is nmse_mda - the predictive
                    # mean scored against the noise-free truth on Q, not the
                    # environment's own (noisy-heldout) evaluator metric.
                    # Every other task keeps the environment evaluator's metric.
                    "y_primary": (
                        row["bma"]["nmse_mda"]
                        if spec.name == "death"
                        else row["env_evaluator"]["metric"]
                    ),
                    "y_name": y_name,
                    "y_label": y_label,
                    "nll": row["bma"]["nll"],
                    "coverage90": row["bma"]["coverage90"],
                    "width90": row["bma"]["width90"],
                    "crps": "",
                    "is_0.1": "",
                    # KL(truth || BMA) on the fixed evaluation set Q, recorded
                    # by `our_scores` for EVERY task with a true observation
                    # law - blank (never 0) for a task whose law could not be
                    # built (irt without its captured latent draw), the same
                    # `.get(..., "")` rule "brier" below uses. The column set
                    # is unchanged: the KL diagnostics live in scores.jsonl
                    # and summary.json, never here.
                    "kl": row["bma"].get("kl", ""),
                    "status": "completed",
                    # Shared helper, see
                    # `core_count.points_label`'s own docstring for the rule.
                    # A label the RUN already resolved (--run-label,
                    # recorded in config) wins; the helper stays the fallback for
                    # a config without it and for hand-built test configs.
                    "label": config.get("label")
                    or points_label(
                        config["profile"],
                        "mock" if "mock" in config["proposer"] else config["proposer"],
                    ),
                    "run_id": Path(config["output"]).name,
                    "source_path": (
                        f"{config['output']}/scores.jsonl#L{line}"
                        f"(round={row['round']},phase={row['phase']})"
                    ),
                    # A multi-component task's reported y/nll/coverage90/
                    # width90 are the MEAN over its components, so the plot
                    # contract's own `variant` column carries that label
                    # (the paper's task table, last row). Every single-outcome
                    # task keeps the empty value.
                    # The prior-template and proposed-prior ablations add
                    # `prior_wide`/`prior_narrow`/`prior_llm` here, joined with the
                    # composite token by the same rule `core_count.points_variant`
                    # uses - alphabetically, with `+`. A default single-outcome run
                    # keeps the empty value.
                    "variant": points_variant(config, spec),
                    "code_sha": config["source_files_sha256"]["boxinggym_core.py"],
                    # Engine-computed Brier, ONLY where
                    # `our_scores` defined it (outcome_type == "binary") -
                    # blank (never 0) everywhere else.
                    "brier": row["bma"].get("brier", ""),
                    # ChemBench's own auxiliary columns - this suite never computes them,
                    # always blank.
                    "nmsle": "",
                    "rmsle": "",
                }
            )


def main():
    """Command-line entry point: parse and validate the options, then call `run`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", required=True, choices=tuple(TASKS))
    # Flipped to "dev" - the
    # SAFE default, since every existing caller that means to reproduce a dev
    # anchor already passes --profile dev explicitly (unaffected by this flip)
    # while a bare invocation can no longer silently spend the paper-sized
    # (500, 256, 100, 128) SMC/EIG budget. --profile paper must now be given
    # explicitly, exactly like --allow-paid-api already must be.
    parser.add_argument("--profile", choices=("dev", "paper"), default="dev")
    parser.add_argument("--proposer", choices=("mock", "anthropic", "openai", "openai_compat"), default="mock")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=argparse.SUPPRESS)
    parser.add_argument("--llm-key-name", default=argparse.SUPPRESS)
    parser.add_argument("--llm-provider-order", type=normalize_provider_order, default=argparse.SUPPRESS)
    # The reasoning and maximum-token decoding options: both stay absent from the namespace unless typed, so an
    # omitted flag leaves the request body, the config and the points columns
    # byte-identical. `--llm-reasoning` is openai_compat only (compat_config
    # refuses it elsewhere); `--llm-max-tokens` is a decoding setting for
    # every provider. An invalid value ends the command line with status 2.
    parser.add_argument("--llm-reasoning", type=parse_reasoning_spec, default=argparse.SUPPRESS)
    # OpenAI JSON mode (stock provider only) and the validation-message wording; both
    # SUPPRESS, so an omitted option leaves the request, config and receipt byte-identical.
    parser.add_argument("--llm-response-format", choices=RESPONSE_FORMATS, default=argparse.SUPPRESS)
    parser.add_argument("--validation-messages", choices=VALIDATION_MESSAGE_VERSIONS,
                        default=argparse.SUPPRESS)
    parser.add_argument("--llm-max-tokens", type=parse_max_tokens, default=argparse.SUPPRESS)
    parser.add_argument("--allow-paid-api", action="store_true")
    # The per-cell paid-call guard. SUPPRESS, so an omitted option leaves every
    # record byte-identical; stated, the proposer refuses the call that would pass it.
    parser.add_argument("--max-paid-calls", type=parse_max_paid_calls, default=argparse.SUPPRESS)
    parser.add_argument("--d0-policy", choices=("fractions4", "random2"), default="fractions4")
    # A start whose observations are all zero (an unbounded count can be) has no
    # positive D0 mean to serve as the response unit; `refuse` is the existing behaviour.
    # Absent unless typed (argparse.SUPPRESS): config.json records `**vars(args)`, so a default
    # value here would change every default run's config bytes.
    parser.add_argument("--d0-zero-count-unit", choices=D0_ZERO_COUNT_UNIT_CHOICES, default=argparse.SUPPRESS)
    parser.add_argument("--q-policy", choices=("native", "exclude_observed"), default="native")
    # Every task records KL(truth || BMA)
    # unconditionally EXCEPT the one whose measured cost exceeded 10% of a
    # cell's wall clock (KL_OPTIONAL_TASKS); this flag turns that one on too.
    # SUPPRESS, so an omitted flag is absent from the namespace and no existing
    # command line changes by one byte.
    parser.add_argument("--kl-all-tasks", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--cell", choices=tuple(f"{s}{f}{a}" for s in "01" for f in "01" for a in "012"), default="111")
    parser.add_argument("--design-policy", choices=("joint_eig", "epig", "point_disagreement", "space_filling", *EIG_PART_POLICIES), default=argparse.SUPPRESS)
    # The EPIG target count, default 32 (count_inference.EPIG_TARGET_POINTS).
    # SUPPRESS, so an omitted option leaves every record byte-identical; epig arm only.
    parser.add_argument("--epig-target-points", type=parse_epig_target_points, default=argparse.SUPPRESS)
    # Continuous design search on the task's declared input box. SUPPRESS, so an
    # omitted option adds no config.json key and changes nothing. Under the unified
    # protocol it is the rule for the four active EIG-family arms (ascent on a box
    # task, exhaustive where the pool is the whole design set).
    parser.add_argument("--design-search", choices=("ascent",), default=argparse.SUPPRESS)
    # ---- death regime switches and the prompt variant ----------------------
    # No `choices=`: the valid families depend on the TASK's outcome type
    # (belief_packet.OUTCOME_MENUS), which is not known at parse time -
    # `interface_families` validates it inside run(), same rule CORE's own
    # `--initial-family` (core_count.py) states for its one fixed menu.
    parser.add_argument("--initial-family", default=None)
    parser.add_argument("--truth-in-pool", action="store_true")
    parser.add_argument("--inject-oracle-at", type=int, default=None)
    parser.add_argument("--prompt-variant", default=None)
    parser.add_argument("--mask-input-names", action="store_true", default=argparse.SUPPRESS)
    # The numbered system prompt this run sends - the registerable sibling of
    # the free-form `--prompt-variant` above, admitted in production because the file is
    # packaged and its digest is what the manifest registers. SUPPRESS, so an omitted
    # option adds no config.json key at all.
    parser.add_argument("--prompt-version", default=argparse.SUPPRESS)
    # Prior-template and proposed-prior ablations. Both default to the frozen
    # production behaviour: the Table 3 role priors and no proposed-prior path.
    parser.add_argument(
        "--prior-template", choices=PRIOR_TEMPLATE_NAMES, default=DEFAULT_PRIOR_TEMPLATE
    )
    parser.add_argument("--priors", choices=("off", "llm"), default="off")
    # Stable log(1 - exp(z)) in both backends (count_models.log1mexp). SUPPRESS, so an
    # omitted option adds no config.json key; given, `stable_log1mexp: true` travels in via vars(args).
    parser.add_argument("--stable-log1mexp", action="store_true", default=argparse.SUPPRESS)
    # The per-round design criteria record (joint, Model and Parameter EIG
    # and point disagreement of every candidate design, acquisition/tNN_criteria.json). SUPPRESS,
    # so an unstated option adds no config key and writes no file.
    parser.add_argument("--record-design-criteria", action="store_true", default=argparse.SUPPRESS)
    # Candidates asked for per round, 1..3 - the same bound
    # and the same wording as core_count's own rule. SUPPRESS, so an omitted
    # option adds no config key and k = 1 keeps every byte of today's request.
    parser.add_argument("--proposal-limit", type=int, default=argparse.SUPPRESS)
    # The round-0 burst size and the active-pool cap; absent means none and
    # the module default, read through `getattr` in `run`.
    parser.add_argument("--initial-proposals", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--active-maximum", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--burst-concurrency", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--initial-catalogue", choices=("on", "off"), default=argparse.SUPPRESS)
    parser.add_argument("--parallel-proposals", type=int, default=argparse.SUPPRESS)
    # Archive review rounds (a comma or space list within --rounds;
    # needs --proposal-limit 3). SUPPRESS: unstated, config.json and every record keep their bytes.
    parser.add_argument("--archive-review-rounds", default=argparse.SUPPRESS)
    # The expression grammar this run's candidates are judged
    # against. SUPPRESS for the same reason `--prompt-version` uses it - an
    # omitted option adds no config.json key at all, so a command line that
    # never states it records exactly what it always did, and the finished runs
    # under grammar v2 keep their grammar and their bytes. Unstated means
    # DEFAULT_GRAMMAR_VERSION, read through `getattr` in `run`.
    parser.add_argument(
        "--grammar-version", choices=GRAMMAR_VERSIONS, default=argparse.SUPPRESS
    )
    # Omitted labels stay development; production needs registration.
    parser.add_argument("--run-label", choices=RUN_LABEL_CHOICES, default=None)
    parser.add_argument("--fit-guards", choices=("off", "v1"), default=argparse.SUPPRESS)
    parser.add_argument("--fit-timeout", dest="fit_timeout_seconds", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--manifest-ref", default=argparse.SUPPRESS,
                        help="JSON object containing manifest path and whole-file sha256")
    args = parser.parse_args()
    # core_count's own rule, stated here in the same words.
    if not 1 <= getattr(args, "proposal_limit", 1) <= 3:
        parser.error("--proposal-limit must be 1..3")
    if not 0 <= getattr(args, "initial_proposals", 0) <= 24:
        parser.error("--initial-proposals must be 0..24")
    if not 1 <= getattr(args, "active_maximum", ACTIVE_MAXIMUM) <= 32:
        parser.error("--active-maximum must be 1..32")
    if getattr(args, "initial_proposals", 0) > getattr(args, "active_maximum", ACTIVE_MAXIMUM):
        parser.error("--initial-proposals must not exceed --active-maximum")
    if not 1 <= getattr(args, "burst_concurrency", DEFAULT_BURST_CONCURRENCY) <= 24:
        parser.error("--burst-concurrency must be 1..24")
    if not 0 <= getattr(args, "parallel_proposals", 0) <= 6:
        parser.error("--parallel-proposals must be 0..6")
    if getattr(args, "parallel_proposals", 0) and getattr(args, "proposal_limit", 1) != 1:
        parser.error("--parallel-proposals and --proposal-limit are alternative arms; state one")
    if getattr(args, "initial_catalogue", "on") == "off" and (
            not getattr(args, "initial_proposals", 0) or args.cell[:2] == "00"):
        parser.error("--initial-catalogue off needs --initial-proposals on a cell that proposes")
    if args.proposer == "openai_compat":
        compat_config(args)
    if args.llm_model is None:
        # Per-provider default: tests/smoke/development use gpt-4o,
        # registered runs use claude-opus-4-7; unchanged for mock (never dereferenced).
        args.llm_model = "gpt-4o" if args.proposer == "openai" else "claude-opus-4-7"
    # 1..30 (was 1..20; the unified protocol runs 30 rounds); the proposer's own round bound is
    # passed as `max_rounds=args.rounds`, which it validates and never records.
    if not 1 <= args.rounds <= 30:
        parser.error("--rounds must be 1..30")
    # Rejected here rather than
    # silently ignored - a quietly-skipped injection would leave a run that
    # BELIEVES it injected the oracle when it never did. `k < 1` is refused
    # too: round-0 pool membership is `--truth-in-pool`'s job, not this one's.
    if args.inject_oracle_at is not None and not 1 <= args.inject_oracle_at <= args.rounds:
        parser.error("--inject-oracle-at must be between 1 and --rounds")
    try:
        run(args)
    except PaidCallBudgetExhausted as exc:
        print(f"STOP {exc}; summary.json records stop_reason budget_exhausted", file=sys.stderr)
        raise SystemExit(PAID_CALL_BUDGET_EXIT_CODE) from None


if __name__ == "__main__":
    main()


__all__ = [
    "TASKS",
    "ComponentState",
    "TaskEnvironment",
    "TaskSpec",
    "canonical_model_id",
    "emotion_handle",
    "emotion_input_row",
    "env_seed_injection",
    "format_predictions",
    "grammar_means",
    "moral_contrast_row",
    "moral_handle",
    "openai_narration_stub",
    "particle_count",
    "prior_probe",
    "retarget",
    "run",
]
