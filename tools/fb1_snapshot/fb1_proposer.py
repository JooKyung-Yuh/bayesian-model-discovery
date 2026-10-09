"""FB-1 (ForceBench) LLM proposer - the **catalogue-selection** contract, the ForceBench edition of the belief
packet, and the transport.

This module does no scientific computation: it only **reads** the sequential Monte Carlo (SMC), expected information gain (EIG), likelihood and
priors of fb1_core and does not change a single character of them.

Three parts:

1. **Catalogue contract** - maps each named structure of `fb1_core.STRUCTURES` one-to-one onto a **canonical
   expression string** (`canonical_expression`, table = `catalog_table()`), and puts that list into
   `constraints.initial_structures`.  Because `structure_expansion=false` and `family_expansion=true`, the S0 rule of
   prompt v2.0 - "expression is verbatim one of initial_structures; only family may change" - applies **untouched**.
   The parser maps an expression string back to its STRUCTURES name one-to-one.
   Claim ceiling = "**selection** among named laws" (not discovery) -> output label
   `recovery_mode: "catalog_selection"`.

2. **ForceBench edition of the belief packet** - `build_belief_packet(...)`.  Its signature has **no** held-out, Q,
   evaluation or score argument (static test `test_builder_signature_has_no_evaluation_arguments`), and packets that
   differ only in the evaluation set are byte-identical (tested).  **No raw trajectories**: observation rows carry only
   the design identity and the integration state, and the best model's residuals are carried only as a summary (RMS,
   max |z|, min r, max |acceleration|) and an 8-point downsampled residual.  Designs <= `MAX_DESIGNS` (60).

3. **Transport** - `Fb1CatalogProposer`.  The record format mirrors the engine's proposer API `count_proposer.propose_one`
   exactly (prompt_version/prompt_sha256/packet_sha256/calls/repairs<=1/raw_text/parsed and rejected).  **The default is
   mock** (the pre-declared schedule `MOCK_SCHEDULE`, one entry per proposal event).  The live Anthropic path exists
   but opens only when `allow_live=True` is **explicitly set** (in the runner, the environment variable
   `FB1_ALLOW_PAID_API=1`) - without it there are zero paid calls.
"""

import copy
import hashlib
import inspect
import json
import logging
import math
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from fb1_core import FAMILIES, STRUCTURES, B0_BACKGROUND, W0_MODULATION

# ----------------------------------------------------------------- prompt v2.0
# A **byte-for-byte** copy of the file of an earlier experiment phase (do not edit).  A test checks its sha256.
PROMPT_VERSION = "v2.0"
PROMPT_FILE = Path(__file__).with_name("prompts") / "proposer_system_v2_0.txt"
PROMPT_SHA256 = "f095edf1adb3b6533a9e68e73f98ca1dee3f23cd077e424b092393cb32b93aef"
# One added line naming the SHAPE of `parameters`.  This is engine text,
# not the frozen prompt file (prompts/proposer_system_v2_0.txt, sha f095edf1, is
# untouched).  The gpt-4o smoke failed twice on the same reply because the
# validation message was returned verbatim with no shape hint, so the model
# re-sent the same nested form (correction power 0 in that smoke run).
REPAIR_INSTRUCTION = (
    "Your previous reply failed validation with this error:\n{error}\n"
    "Reply with exactly one JSON object "
    '{{"expression": ..., "family": ..., "parameters": {{...}}}} and nothing else.\n'
    'Each value in parameters is the role STRING itself, not an object: '
    '{{"c": "positive"}}, never {{"c": {{"role": "positive"}}}}.'
)

# A reply with no text at all cannot be echoed back - an assistant
# turn whose content is empty is refused by the compatible wire shape itself
# (HTTP 400 "Invalid assistant message: content or tool_calls must be set"),
# and the Anthropic shape refuses an empty content block the same way, so the
# repair call that was meant to rescue the round failed transport-side. This
# turn states what happened once, as the user, and quotes nothing. Same
# JSON-shape sentences as REPAIR_INSTRUCTION above.
REPAIR_INSTRUCTION_EMPTY = (
    "Your previous reply was empty (finish_reason={finish_reason}).\n"
    "Reply with exactly one JSON object "
    '{{"expression": ..., "family": ..., "parameters": {{...}}}} and nothing else.\n'
    'Each value in parameters is the role STRING itself, not an object: '
    '{{"c": "positive"}}, never {{"c": {{"role": "positive"}}}}.'
)


class EmptyResponseBody(ValueError):
    """A persisted reply that carries no text at all.

    A ValueError, so the repair-eligibility rule is exactly the one that was
    already there; the dedicated type is what tells the repair turn to send no
    echo and what the record reports as `repair_kind: "empty_body"`. No
    provider or model name takes part in the rule.
    """

    def __init__(self, finish_reason):
        super().__init__(f"empty response body (finish_reason={finish_reason})")
        self.finish_reason = finish_reason


def blank_reply(text):
    """True when a persisted reply carries no text: absent, or whitespace only."""
    return text is None or (isinstance(text, str) and not text.strip())


DEFAULT_MODEL = "claude-opus-4-7"
# OpenAI is the tests/smoke/development transport
# only; main runs stay on DEFAULT_MODEL via Anthropic. Not an
# automatic per-provider default - `model=` keeps its own static default
# (DEFAULT_MODEL, unchanged) regardless of provider, exactly like
# run_gravity.py's own FB1_LLM_MODEL default (out of scope here, and
# unchanged): a caller selecting the openai provider passes this explicitly
# (`model=DEFAULT_MODEL_OPENAI` or FB1_LLM_MODEL=gpt-4o at the CLI).
DEFAULT_MODEL_OPENAI = "gpt-4o"
MAX_TOKENS = 1600
_MODEL_UNSET = object()

PACKET_VERSION = "v2"
SIGNIFICANT_DIGITS = 4
MAX_DESIGNS = 60                 # designs <= 60
RESIDUAL_DOWNSAMPLE_POINTS = 8   # 8-point downsampled residual trajectory

REQUIRED_CANDIDATE_KEYS = {"expression", "family", "parameters"}
OPTIONAL_CANDIDATE_KEYS = {"note"}
NOTE_MAX_CHARS = 200
# Dispersion parameters owned by the family.  When a candidate wrongly lists such a name in parameters, it is
# **ignored and recorded, not refused** (the same fail-soft handling as the engine's proposer API).
RESERVED_VARIANCE_PARAMETER_NAMES = ("sigma",)

# ---------------------------------------------------------- the form of parameters
# `structure_parameters` uses the nested per-coefficient form {"role": ..., "prior": ...} (so that the fixed prior is
# visible).  The gpt-4o smoke mirrored that form back and sent parameters values as `{"c": {"role": "positive"}}`;
# the role was **right** but the reply was refused for its form, and the error message read like a role violation
# (seen in the gpt-4o smoke run).  So only the form is normalized before comparing and recording,
# and any other value is refused with a clear message.
PARAMETERS_FORMAT_LINE = (
    "map each coefficient to its role string exactly as in "
    'structure_parameters[<expression>][<coefficient>].role, e.g. {"c": "positive", '
    '"b": "positive"}; do not restate the prior and do not nest the role in an object'
)
PARAMETER_SHAPE_MESSAGE = 'parameters values must be role strings, e.g. {"c": "positive"} (got %s)'
PRIOR_RESTATED_MESSAGE = "prior is fixed by the catalogue; do not restate it"
TOLERATED_PARAMETER_OBJECT_KEYS = ({"role"}, {"role", "prior"})


def _json_type_name(value):
    """JSON type name used by PARAMETER_SHAPE_MESSAGE."""
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def normalize_parameter_role(value, declared):
    """Normalize one value to a role string -> (role, normalized, reason).

    Accepted forms: a string (as is), `{"role": X}`, and `{"role": X, "prior": P}` where P is the **same string** as the
    catalogue declaration (`declared["prior"]`).  A different prior is a prior modification, which the contract
    forbids, so it is refused with PRIOR_RESTATED_MESSAGE (never silently dropped).  Any other object or non-string is
    refused with PARAMETER_SHAPE_MESSAGE.  When `declared` is None (the expression is outside the catalogue) there is
    nothing to compare the prior against, so it is not checked - that reply is already refused by the verbatim rule.
    """
    if isinstance(value, str):
        return value, False, None
    if isinstance(value, dict) and set(value) in TOLERATED_PARAMETER_OBJECT_KEYS:
        role = value["role"]
        if not isinstance(role, str):
            return None, False, PARAMETER_SHAPE_MESSAGE % _json_type_name(role)
        if "prior" in value and declared is not None and value["prior"] != declared.get("prior"):
            return None, False, PRIOR_RESTATED_MESSAGE
        return role, True, None
    return None, False, PARAMETER_SHAPE_MESSAGE % _json_type_name(value)


def normalize_parameter_roles(parameters, declared):
    """The whole `parameters` mapping as a mapping to role strings -> (roles, normalized, reason).

    `declared` is that expression's `structure_parameters[expression]` (None when absent).  Names are visited in sorted
    order, so the order of refusal reasons matches the judgment.
    """
    roles, normalized = {}, False
    for name in sorted(parameters):
        role, one_normalized, reason = normalize_parameter_role(
            parameters[name], None if declared is None else declared.get(name)
        )
        if reason is not None:
            return roles, normalized, reason
        roles[name] = role
        normalized = normalized or one_normalized
    return roles, normalized, None

# ------------------------------------------------------- role vocabulary (fixed by prompt v2.0)
# The prompt allows four role words: intercept/signed/positive/shape.  The fb1 prior kinds are mapped onto that
# vocabulary.  The **scale** is not collapsed into the vocabulary; it stays in each coefficient's prior string (a
# different prior scale is a different model - the belief-packet schema).
KIND_ROLE = {"pos": "positive", "sig": "signed", "pos_bg": "positive", "pos_w": "positive"}
KIND_PRIOR = {
    "pos": "LogNormal(0,1)",
    "sig": "Normal(0,1)",
    "pos_bg": f"LogNormal(log({B0_BACKGROUND:.6g}),1)",
    "pos_w": f"LogNormal(log({W0_MODULATION:.6g}),1)",
    "sig_hn": "HalfNormal(1)",
}
FAMILY_DISPERSION_NAME = "sigma"

class _CoefficientNames(dict):
    """Structure name -> tuple of free-constant names.  **The keys are exactly the 14 catalogue entries.**

    Safety net: `set`, `len`, `items`, `in` and `get` behave exactly as the
    old catalogue table did, and only a **subscript lookup** (`COEFFICIENT_NAMES[name]`) with the name of a free
    structure registered at runtime finds its answer in `fb1_grammar.REGISTRY`.  The normal path is
    `structure_coefficient_names`, and every lookup on the execution path has moved there, but this removes the very
    possibility that a place subscripting the catalogue-only table dies with a KeyError on a free structure (two of the
    failures the review reproduced were such subscripts).  A registered name cannot collide with a catalogue name
    (`fb1_grammar.register` prevents it).
    """

    def __missing__(self, name):
        entry = _free_registry_entry(name)
        if entry is None:
            raise KeyError(name)
        return tuple(entry["coefficient_names"])


# Free-constant names per structure.  Their order matches `STRUCTURES[name].kinds` one-to-one (theta[:,j]).
# The test `test_coefficient_names_match_the_catalog` checks the count, the order and the appearance in the expression.
COEFFICIENT_NAMES = _CoefficientNames({
    "inv_r2": ("c",),
    "linear": ("c",),
    "const": ("c",),
    "screened": ("c", "lam"),
    "inv_r": ("c",),
    "power": ("c", "a"),
    "inv_r_plus_c": ("c", "b"),
    "logr": ("c", "k"),
    "inv_r_p1sq": ("c",),
    "inv_r_nomass": ("c",),
    "inv_r_bg": ("c", "b"),
    "inv_r_tmod": ("c", "A1", "A2", "w"),
    "prod_coupling": ("c", "k"),
    "crossover": ("c1", "c2", "R"),
})

# Archive outcomes and reasons (the same set and the same enforcement rule as the belief-packet schema)
ARCHIVE_OUTCOMES = ("active", "retired_by_evidence", "rejected", "duplicate", "fit_failed")
ARCHIVE_REASON_KEYS = {
    "active": {"admitted_round", "active_rounds"},
    "retired_by_evidence": {
        "log_evidence_at_retirement",
        "gap_to_cutoff_nats",
        "active_rounds",
        "computed_round",
        "n_obs",
    },
    "rejected": {"validation_message"},
    "duplicate": {"matched_model_id", "matched_expression"},
    "fit_failed": {"error"},
}
# Adds two keys that only free-mode entries carry (catalogue entries do not have them, so the catalogue bundle is
# byte-unchanged).  `reason_code` is the verdict code (nonfinite_on_probe_grid, duplicate_by_numerical_equivalence and
# so on) and `numerical_equivalence` is the comparison record of the numerical-equivalence duplicate check - both are
# where the model can read from the archive "why it was dropped".
# `catalog_reference` is attached only in free mode - catalogue entries do not have it, so
# the catalogue bundle bytes stay the same (the same rule as the other three optional keys).
ARCHIVE_OPTIONAL_KEYS = ("note", "prior_probe", "reason_code", "numerical_equivalence",
                         "catalog_reference")

CONSTRAINT_KEYS = {
    "structure_expansion",
    "family_expansion",
    "initial_family",
    "allowed_families",
    "initial_structures",
    "structure_parameters",
    # One added line telling the model what SHAPE `parameters` takes.
    # `structure_parameters` keeps its nested {role, prior} form - that is what
    # makes each coefficient's fixed prior visible - and this line says the
    # reply mirrors only the role string back.
    "parameters_format",
    "roles",
    "family_dispersion",
}
# ---------------------------------------------------------- free-mode constraint keys
# Three keys added only in free-expression mode (FB1_STRUCTURE_PROPOSALS=free).  `roles` is already in the catalogue
# set (in free mode only its content narrows to the two roles pos/sig), so it is not a new key.
#   grammar      allowed tokens, calls, input names and limits in one block (owned by fb1_grammar)
#   input_names  the names an expression may use {r, p1, p2, t}
#   n_ops_max    operator limit (the catalogue maximum of 10, counted with the vendored counting rule)
FREE_CONSTRAINT_EXTRA_KEYS = {"grammar", "input_names", "n_ops_max"}
FREE_CONSTRAINT_KEYS = CONSTRAINT_KEYS | FREE_CONSTRAINT_EXTRA_KEYS


def is_free_constraints(constraints):
    """Is the constraints block a free-mode one?  True only when the key set is **exactly** the free set."""
    return set(constraints) == FREE_CONSTRAINT_KEYS


def constraint_keys_ok(constraints):
    """There are **exactly two** allowed constraint key sets - the catalogue set or the free set.

    This is not loosened to 'extra keys are tolerated': then this check would guarantee nothing.
    """
    keys = set(constraints)
    return keys == CONSTRAINT_KEYS or keys == FREE_CONSTRAINT_KEYS
INTERFACE_KEYS = (
    "outcome_type",
    "support",
    "inputs",
    "derived_inputs",
    "link",
    "time_grid",
    "integration",
    "observation_noise",
    "summary_definitions",
)

# **Machine-readable definitions** of every summary field (in the style of
# best_model_residual_definitions).  Defined in one place only: interface.summary_definitions and the top-level
# best_model_residual_definitions use **the same source** (they cannot diverge).
FIELD_DEFINITIONS = {
    "observations": {
        "design_id": "stable_label_of_the_design_in_arrival_order_d00_d01_and_so_on",
        "p1": "first_design_knob_handed_to_the_world",
        "p2": "second_design_knob_handed_to_the_world_and_the_inertial_mass_in_acceleration=force/p2",
        "initial_state": "object_with_pos_and_vel_each_a_length_2_vector_at_time_zero",
        "integration_steps": "count_of_velocity_verlet_steps_of_size_dt_from_time_zero_to_the_last_measurement_time",
        "integration_status": "one_of_ok_or_nonfinite_observation_or_nonfinite_prediction",
    },
    "pool": {
        "weight": "evidence_weight_exp_of_normalized_log_evidence_over_the_active_pool",
        "log_evidence": "log_marginal_likelihood_of_the_last_tempered_smc_fit",
        "born_round": "round_at_which_this_candidate_entered_the_active_pool",
        "one_step_log_scores": (
            "log_of_the_particle_mean_predictive_density_of_the_newly_observed_trajectory_"
            "under_the_previous_rounds_fit_log_mean_exp_not_mean_of_logs; null before the "
            "candidate had a previous-round fit and never back-filled; a candidate admitted "
            "at round t is first fit at round t+1 so its first score appears at round t+2"
        ),
        "one_step_rounds": "round_indices_aligned_one_to_one_with_one_step_log_scores",
        "coefficients": "posterior_mean_and_sample_sd_ddof1_of_each_free_constant_of_the_expression",
        "family_dispersion": "posterior_mean_and_sample_sd_of_the_dispersion_parameter_owned_by_the_family",
    },
    "best_model_residuals": {
        "pred_mean": "posterior_predictive_mean_of_the_position_coordinate",
        "pred_sd": "posterior_predictive_sd_includes_coefficient_uncertainty_and_observation_noise",
        "standardized_residual": "(y-pred_mean)/pred_sd",
        "rms_standardized_residual": "sqrt_of_the_mean_of_standardized_residual_squared_over_measurement_times_and_both_coordinates",
        "max_abs_standardized_residual": "max_over_measurement_times_and_both_coordinates_of_the_absolute_standardized_residual",
        "min_r": "min_over_measurement_times_of_the_norm_of_the_best_models_posterior_predictive_mean_position",
        "max_abs_accel": "max_over_measurement_times_of_the_absolute_acceleration_force_over_p2_of_the_best_model_at_its_posterior_median_coefficients_along_that_mean_trajectory",
        "residual_trajectory_8pt": "root_mean_square_over_the_two_coordinates_of_the_standardized_residual_at_8_measurement_times_chosen_by_rounding_linspace_0_to_T_minus_1_into_8_points",
    },
}
RESIDUAL_DEFINITIONS = FIELD_DEFINITIONS["best_model_residuals"]
#: Unified protocol: a proposal round fits the candidates it admits in that round, so the one-step score
#: definition those runs' packets carry says so (in words that name no input, so masking leaves it).
UNIFIED_ONE_STEP_DEFINITION = (
    "log_of_the_particle_mean_predictive_density_of_the_newly_observed_trajectory_"
    "under_the_previous_rounds_fit_log_mean_exp_not_mean_of_logs; null before the "
    "candidate had a previous-round fit and never back-filled; a candidate admitted "
    "in a round is fit in that same round on all current data, so its first score appears in the next round"
)


def unified_summary_definitions():
    """FIELD_DEFINITIONS with the pool's one-step score definition of a unified-protocol run (a copy)."""
    definitions = json.loads(json.dumps(FIELD_DEFINITIONS))
    definitions["pool"]["one_step_log_scores"] = UNIFIED_ONE_STEP_DEFINITION
    return definitions


# ================================================================= catalogue contract
def canonical_expression(name):
    """STRUCTURES name -> canonical expression string.  **Derived from `Structure.expr` of fb1_core.**

    It is extracted from the frozen original rather than from a hand-written table, so it cannot diverge from the
    catalogue.  Only `prod_coupling` writes its expr in both acceleration and force form (the force in parentheses), so
    the parentheses are used for it.

    Safety net: for the name of a free structure registered at runtime, the
    registered expression is returned as is.  A free expression is not of the form `F = ...`, so it used to die here
    with a ValueError (three of the failures the review reproduced).  The result for a catalogue name does not change by
    a single character, because a registered name cannot collide with a catalogue name (`fb1_grammar.register`).  The
    normal path is `structure_expression`, and every lookup on the execution path has moved there.
    """
    entry = _free_registry_entry(name)
    if entry is not None:
        return entry["expression"]
    expr = STRUCTURES[name].expr
    if "(F = " in expr:
        expr = expr[expr.index("(F = ") + 1 : expr.rindex(")")]
    head, sep, body = expr.partition(" = ")
    if head != "F" or not sep:
        raise ValueError(f"{name}: expr is not a force-law form: {STRUCTURES[name].expr!r}")
    return " ".join(body.split())


def catalog_table():
    """One-to-one table of structure name <-> canonical expression <-> free constants (name, role, prior) <-> family
    (copied into CHANGES as is).

    Iterates over **catalogue names only**.  In free mode
    `fb1_grammar.register` inserts free structures into `STRUCTURES` at runtime, so iterating the global as is would die
    with a ValueError/KeyError in the catalogue-only tables (`canonical_expression`, `COEFFICIENT_NAMES`).
    """
    rows = []
    for name, st in STRUCTURES.items():
        if _free_registry_entry(name) is not None:
            continue
        expression = canonical_expression(name)
        coefficients = COEFFICIENT_NAMES[name]
        rows.append(
            {
                "structure": name,
                "expression": expression,
                "n_coeff": st.n_coeff,
                "n_ops": st.n_ops,
                "parameters": {
                    coefficient: {"role": KIND_ROLE[kind], "prior": KIND_PRIOR[kind], "kind": kind}
                    for coefficient, kind in zip(coefficients, st.kinds)
                },
            }
        )
    return rows


def expression_to_structure():
    """Canonical expression string -> **catalogue** STRUCTURES name (one-to-one).  Raises at once on a collision.

    Registered free structures are not put into this table, for two reasons.

      (1) A free expression is not of the form `F = ...`, so `canonical_expression` raises a ValueError (reproduced:
          calling this function after admitting `a*r+b` raises at once - the runner died exactly here while building
          the past-history set before the next proposal).
      (2) More fundamentally, the same expression string can be registered twice with **different roles (priors)**, so
          a one-to-one map on expression strings does not hold in the first place (`a*r+b` with
          a:positive,b:signed and with a:signed,b:positive are different models).

    So history identity is matched by **candidate identity**, not by an expression-string table -
    `structure_candidate_identity` / `archive_pool_keys`.
    """
    table = {}
    for name in STRUCTURES:
        if _free_registry_entry(name) is not None:
            continue
        expression = canonical_expression(name)
        if expression in table:
            raise ValueError(f"the canonical expressions are not one-to-one: {expression!r}")
        table[expression] = name
    return table


def model_id(expression, family, parameters):
    """First 20 characters of the sha256 of canonical expression + roles + family (the same rule as the engine's
    `Hypothesis.model_id`).

    In catalogue selection the roles are fixed by the expression, so model_id <-> (structure, family) is one-to-one
    (test `test_model_id_is_one_to_one_with_structure_and_family`).
    """
    roles = ",".join(f"{name}:{parameters[name]}" for name in sorted(parameters))
    identity = hashlib.sha256(f"{expression}|{roles}|{family}".encode()).hexdigest()[:20]
    if _ACTIVE_NAME_MASK is not None:
        # Input-name masking: the masked expression's own id, kept beside this one (never instead of it).
        _ACTIVE_NAME_MASK.note_model_id(identity, expression, family, parameters)
    return identity


def catalog_constraints(initial_family="gauss", *, family_expansion=True):
    """The catalogue-selection `constraints` block.

    Because `structure_expansion=False`, the S0 rule of prompt v2.0 applies as is: expression is **verbatim one** of
    `initial_structures`, and only family may change.
    """
    if initial_family not in FAMILIES:
        raise ValueError(f"unknown family: {initial_family}")
    rows = catalog_table()
    return {
        "structure_expansion": False,
        "family_expansion": bool(family_expansion),
        "initial_family": initial_family,
        "allowed_families": list(FAMILIES) if family_expansion else [initial_family],
        "initial_structures": [row["expression"] for row in rows],
        "structure_parameters": {
            row["expression"]: {
                coefficient: {"role": spec["role"], "prior": spec["prior"]}
                for coefficient, spec in row["parameters"].items()
            }
            for row in rows
        },
        "parameters_format": PARAMETERS_FORMAT_LINE,
        "roles": {
            role: prior
            for role, prior in sorted(
                {KIND_ROLE[kind]: KIND_PRIOR[kind] for kind in ("pos", "sig")}.items()
            )
        },
        "family_dispersion": {
            FAMILY_DISPERSION_NAME: (
                f"{KIND_PRIOR['sig_hn']}; owned by the family, never list it in parameters"
            )
        },
    }


def free_constraints(initial_family="gauss", *, family_expansion=True, times):
    """The free-expression mode `constraints` block.

    Because `structure_expansion=True`, the S rule of prompt v2.0 applies the other way: the expression **may be written
    directly** (line 10: "If constraints.structure_expansion is false, expression must be verbatim one of
    constraints.initial_structures").  The prompt file is unchanged.

    `initial_structures` is now not a constraint but **the list of laws that already exist** - the catalogue is written
    **in this grammar's notation** (only in the same notation the model must answer in can "not already in the pool or
    archive" actually be checked).  The time-modulated `inv_r_tmod` cannot be written in this grammar (no cos/sin), so
    it is not in the list - 13 of 14.
    """
    import fb1_grammar as fbg          # noqa: PLC0415 - loaded only in free mode

    if initial_family not in FAMILIES:
        raise ValueError(f"unknown family: {initial_family}")
    rows = [
        {
            "expression": expression,
            "parameters": {name: role for name, role in parameters},
        }
        for expression, parameters in (
            fbg.CATALOG_GRAMMAR_EXPRESSIONS[name]
            for name in sorted(fbg.CATALOG_GRAMMAR_EXPRESSIONS)
        )
    ]
    return {
        "structure_expansion": True,
        "family_expansion": bool(family_expansion),
        "initial_family": initial_family,
        "allowed_families": list(FAMILIES) if family_expansion else [initial_family],
        "initial_structures": [row["expression"] for row in rows],
        "structure_parameters": {
            row["expression"]: {
                coefficient: {"role": role, "prior": KIND_PRIOR[fbg.ROLE_TO_KIND[role]]}
                for coefficient, role in row["parameters"].items()
            }
            for row in rows
        },
        "parameters_format": PARAMETERS_FORMAT_LINE,
        "roles": {
            role: KIND_PRIOR[kind] for role, kind in sorted(fbg.ROLE_TO_KIND.items())
        },
        "family_dispersion": {
            FAMILY_DISPERSION_NAME: (
                f"{KIND_PRIOR['sig_hn']}; owned by the family, never list it in parameters"
            )
        },
        "grammar": fbg.grammar_identity(times),
        "input_names": list(fbg.INPUT_NAMES),
        "n_ops_max": fbg.N_OPS_MAX,
    }


#: Unified protocol: what FB1_CATALOGUE=off takes out of a free-mode constraints block - the law
#: catalogue the packet has shown every round since 9/8. Only what the model reads changes; the engine still links a
#: catalogue-identical proposal to its catalogue name and uses the catalogue in the duplicate check.
#: A follow-up change adds the three grammar texts that still named the catalogue: n_ops_max_basis, the
#: catalogue reason code's entry, and the phrase of the numerical duplicate check that lists the catalogue as a target.
CATALOGUE_GRAMMAR_KEYS = ("catalogue_expressible", "catalogue_not_expressible", "catalogue_prior_scale_differences",
                          "catalogue_expressible_count", "n_ops_max_declared", "n_ops_max_declared_basis",
                          "n_ops_max_basis")
CATALOGUE_DUPLICATE_KEYS = ("catalogue_is_a_reference_table", "catalogue_prior_match")
CATALOGUE_REASON_CODE_KEY = "catalog_reference"            # grammar.duplicate_detection.reason_codes entry
CATALOGUE_NUMERICAL_PHRASE = " and the catalogue transcriptions"   # in grammar.duplicate_detection.stage_2_numerical


def catalogue_off_constraints(constraints):
    """A free-mode constraints block without the law catalogue: initial_structures and structure_parameters empty (the
    keys stay, the packet's key set is fixed), and the grammar block without its catalogue entries (the K-label list,
    the K11 and K12 notes, the catalogue counts and prior-match table, the basis of n_ops_max, the catalogue reason
    code) and without the words "and the catalogue transcriptions" in the numerical duplicate check's description.
    Grammar rules, roles, limits (n_ops_max itself) and probe stay."""
    if not is_free_constraints(constraints):
        raise ValueError("the catalogue switch applies to free-form constraints only")
    out = json.loads(json.dumps(constraints))
    out["initial_structures"] = []
    out["structure_parameters"] = {}
    grammar = out["grammar"]
    for key in CATALOGUE_GRAMMAR_KEYS:
        grammar.pop(key, None)
    detection = grammar.get("duplicate_detection", {})
    for key in CATALOGUE_DUPLICATE_KEYS:
        detection.pop(key, None)
    detection.get("reason_codes", {}).pop(CATALOGUE_REASON_CODE_KEY, None)
    if isinstance(detection.get("stage_2_numerical"), str):
        detection["stage_2_numerical"] = detection["stage_2_numerical"].replace(CATALOGUE_NUMERICAL_PHRASE, "")
    return out


#: Unified protocol (follow-up change): with FB1_CATALOGUE=off the pool and archive a packet shows carry no trace of the
#: catalogue either (catalogue_off_packet). Archive row blocks dropped from the packet (the run's own archive keeps them):
CATALOGUE_TRACE_ARCHIVE_KEYS = ("catalog_reference", "numerical_equivalence")


def catalogue_free_notation(expression):
    """A catalogue expression in the free grammar's notation: the catalogue writes powers with '^', a free expression
    (and the grammar the model is given) with '**'. Free expressions never contain '^', so they pass unchanged."""
    return expression.replace("^", "**") if isinstance(expression, str) else expression


def catalogue_off_packet(packet):
    """Unified protocol: a belief packet (engine names, before input-name masking) without the catalogue's traces, for
    FB1_CATALOGUE=off. A proposal the engine linked to a catalogue law keeps its catalogue identity in the run, but the
    packet shows it like any free law: archive rows lose catalog_reference and numerical_equivalence (whose comparison
    lists name every catalogue law by label); expressions in the pool, the best-model line and the archive (row and
    duplicate reason) are written with '**'; the catalogue reason code becomes the code of an admitted free law (None)
    or, on a duplicate, the duplicate code of the link's basis (canonical id or probe grid); and a duplicate's
    matched_model_id that is a catalogue name becomes that model's model_id (the id its pool or archive entry carries).
    Returns a new packet; the argument is not changed."""
    import fb1_grammar as fbg          # noqa: PLC0415 - loaded only in free mode
    out = copy.deepcopy(packet)        # not a JSON round trip: a packet may hold numpy values serialize_packet converts
    catalogue = set(COEFFICIENT_NAMES)
    for member in out.get("pool") or []:
        member["expression"] = catalogue_free_notation(member["expression"])
    if isinstance(out.get("best_model"), str):
        out["best_model"] = catalogue_free_notation(out["best_model"])
    for row in out.get("archive") or []:
        link = row.pop("catalog_reference", None)
        for key in CATALOGUE_TRACE_ARCHIVE_KEYS:
            row.pop(key, None)
        row["expression"] = catalogue_free_notation(row["expression"])
        if row.get("reason_code") == fbg.REASON_CATALOG_REFERENCE:
            if row["outcome"] == "duplicate":
                basis = link.get("basis") if isinstance(link, dict) else None
                row["reason_code"] = (fbg.REASON_DUPLICATE_CANONICAL if basis == "canonical_id"
                                      else fbg.REASON_DUPLICATE_NUMERICAL)
            else:
                row["reason_code"] = None
        reason = row.get("reason")
        if isinstance(reason, dict):
            if "matched_expression" in reason:
                reason["matched_expression"] = catalogue_free_notation(reason["matched_expression"])
            if reason.get("matched_model_id") in catalogue:
                reason["matched_model_id"] = structure_candidate_identity(reason["matched_model_id"], row["family"])
    return out


# ------------------------------------------- structure name lookup (shared by catalogue and free structures)
# `COEFFICIENT_NAMES` and `canonical_expression` know catalogue names only.  Free structures live in
# `fb1_grammar.REGISTRY`, so the two places are consulted in turn - the minimal branch that keeps the runner's pool
# summary and archive block from dying with a KeyError on a free structure.
def _free_registry_entry(name):
    """The record of a registered free structure.  None when not in free mode (the grammar module is not loaded)."""
    module = sys.modules.get("fb1_grammar")
    if module is None:
        return None
    return module.REGISTRY.get(name)


def structure_expression(name):
    """Structure name -> canonical expression string: `canonical_expression` for a catalogue name, the registered
    expression for a free structure."""
    entry = _free_registry_entry(name)
    return entry["expression"] if entry is not None else canonical_expression(name)


def structure_coefficient_names(name):
    """Structure name -> order of the free-constant names (the order of j in `theta[:, j]`)."""
    entry = _free_registry_entry(name)
    return tuple(entry["coefficient_names"]) if entry is not None else COEFFICIENT_NAMES[name]


def structure_roles(name):
    """Structure name -> {coefficient: role}.  A free structure gives its registered roles; a catalogue structure maps
    them from its kinds."""
    entry = _free_registry_entry(name)
    if entry is not None:
        return dict(entry["parameters"])
    return {
        coefficient: KIND_ROLE[kind]
        for coefficient, kind in zip(COEFFICIENT_NAMES[name], STRUCTURES[name].kinds)
    }


def structure_candidate_identity(name, family):
    """(structure name, family) -> candidate identity `model_id`.  Uses only the lookups shared by catalogue and free
    structures.

    `model_id` is the first 20 characters of the sha256 of (canonical expression, per-coefficient roles, family) - that
    is, it carries the structure identity **including the prior** together with the family.  History identification
    was moved to this value.
    """
    return model_id(structure_expression(name), family, structure_roles(name))


def known_structure_names():
    """Every structure name that can be looked up = the catalogue + free structures registered at runtime.

    `fb1_grammar.register` inserts directly into `fb1_core.STRUCTURES`, so its keys are the whole set.
    """
    return tuple(STRUCTURES)


def archive_pool_keys(archive_rows):
    """Archive rows -> the set ``{(structure, family)}`` already evaluated.

    Matched by **candidate identity** (expression + per-coefficient roles/priors + family = `model_id`), not by the
    one-to-one expression-string table (`expression_to_structure`).  Two reasons:

      * Free expressions are not in the catalogue-only table at all - matching by that table would drop the whole past
        history of registered free structures from the "already seen" set of the next proposal, and the table lookup
        itself raises.
      * The same expression string can exist twice with different priors, so the expression
        string is not an identity.

    Archive rows carry no roles (schema section 6), so the identity is computed for every known structure with that
    row's family and compared.  For catalogue names model_id <-> (structure, family) is one-to-one
    (test `test_model_id_is_one_to_one_with_structure_and_family`), so the result equals the old table lookup.  Rows
    without an `expression` or a `model_id` (rows where the proposal itself failed) are skipped.
    """
    wanted = {}
    for row in archive_rows:
        identity, family = row.get("model_id"), row.get("family")
        if identity is None or family is None or row.get("expression") is None:
            continue
        wanted.setdefault(identity, family)
    keys = set()
    if not wanted:
        return keys
    for name in known_structure_names():
        for identity, family in wanted.items():
            if structure_candidate_identity(name, family) == identity:
                keys.add((name, family))
    return keys


def forcebench_interface(*, measurement_times, dt, p1_range, p2_range, integration_steps):
    """The **only** place task information enters (the ForceBench edition of section 3 of the belief-packet interface
    design note).

    The true world name, the true values and held-out data are not included.  Nor is the observation-noise scale - the
    engine **estimates** it as a free constant of the family rather than using it as a known value.
    """
    return {
        "outcome_type": "trajectory_summary",
        "support": "position_vector_in_R2_at_each_measurement_time",
        "inputs": [
            {"name": "p1", "range": [float(p1_range[0]), float(p1_range[1])]},
            {"name": "p2", "range": [float(p2_range[0]), float(p2_range[1])]},
        ],
        "derived_inputs": {
            "r": "euclidean_norm_of_the_position_vector",
            "t": "elapsed_time_from_zero",
            "acceleration": "minus_expression_over_p2_along_the_unit_position_vector",
        },
        "link": "identity",
        "time_grid": [float(t) for t in measurement_times],
        "integration": {
            "scheme": "velocity_verlet",
            "dt": float(dt),
            "steps_to_last_measurement_time": int(integration_steps),
        },
        "observation_noise": {
            "known": False,
            "modelled_by": (
                f"family dispersion parameter {FAMILY_DISPERSION_NAME} with prior "
                f"{KIND_PRIOR['sig_hn']}"
            ),
        },
        "summary_definitions": FIELD_DEFINITIONS,
    }


# ============================================================== serialization (the v2 rule as is)
def round_significant(value, digits=SIGNIFICANT_DIGITS):
    """Every float to `digits` significant digits.  int, bool, None and str stay as they are."""
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("belief packet values must be finite")
        return float(f"{value:.{digits}g}")
    if isinstance(value, dict):
        return {key: round_significant(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_significant(item, digits) for item in value]
    if isinstance(value, np.generic):
        return round_significant(value.item(), digits)
    if isinstance(value, np.ndarray):
        return [round_significant(item, digits) for item in value.tolist()]
    raise TypeError(f"belief packet cannot serialize {type(value).__name__}")


def serialize_packet(packet):
    """Serialize a packet to canonical JSON text: rounded floats, sorted keys, compact separators, no NaN."""
    return json.dumps(
        round_significant(packet), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def packet_digest(text):
    """Byte count and sha256 of a serialized packet text, as a dict."""
    raw = text.encode()
    return {"packet_bytes": len(raw), "packet_sha256": hashlib.sha256(raw).hexdigest()}


# ============================================================ ForceBench edition of the belief packet
class PoolMember:
    """One active candidate as the packet sees it.  Holds **public state only**.

    `predictions` only needs to be filled for the best model (the builder reads only the highest-weight model's):
    `{"pred_mean": (D,T,2), "pred_sd": (D,T,2), "accel_abs": (D,T)}`.
    """

    __slots__ = (
        "structure",
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
        "predictions",
    )

    def __init__(
        self,
        structure,
        family,
        weight,
        log_evidence,
        *,
        log_evidence_computed_round=0,
        log_evidence_n_obs=0,
        born_round=0,
        coefficients=None,
        family_dispersion=None,
        one_step_log_scores=(),
        one_step_rounds=(),
        predictions=None,
    ):
        self.structure = structure
        self.family = family
        self.weight = float(weight)
        self.log_evidence = float(log_evidence)
        self.log_evidence_computed_round = int(log_evidence_computed_round)
        self.log_evidence_n_obs = int(log_evidence_n_obs)
        self.born_round = int(born_round)
        self.coefficients = dict(coefficients or {})
        self.family_dispersion = dict(family_dispersion or {})
        self.one_step_log_scores = list(one_step_log_scores)
        self.one_step_rounds = list(one_step_rounds)
        self.predictions = predictions

    @property
    def expression(self):
        """Canonical expression string of this member's structure."""
        # The shared lookup instead of the catalogue-only
        # `canonical_expression` - on a free structure that function raised a ValueError and stopped the whole
        # belief-packet assembly.
        return structure_expression(self.structure)

    @property
    def parameters(self):
        """{coefficient: role} of this member's structure."""
        # The shared lookup instead of the catalogue-only
        # `COEFFICIENT_NAMES` - that was a KeyError on a free structure.  A free structure returns its registered roles.
        return dict(structure_roles(self.structure))

    @property
    def model_id(self):
        """Candidate identity of this member (the module-level `model_id` of expression, family and roles)."""
        return model_id(self.expression, self.family, self.parameters)

    @property
    def label(self):
        """Display label "structure|family"."""
        return f"{self.structure}|{self.family}"


def _archive_entry(item):
    outcome = item["outcome"]
    if outcome not in ARCHIVE_OUTCOMES:
        raise ValueError(f"unknown archive outcome: {outcome}")
    reason = dict(item["reason"])
    if set(reason) != ARCHIVE_REASON_KEYS[outcome]:
        raise ValueError(f"archive reason keys do not match outcome {outcome}")
    entry = {
        "model_id": item["model_id"],
        "expression": item["expression"],
        "family": item["family"],
        "proposed_round": int(item["proposed_round"]),
        "outcome": outcome,
        "reason": reason,
    }
    for key in ARCHIVE_OPTIONAL_KEYS:
        if key in item:
            entry[key] = item[key]
    return entry


def downsample_indices(n_times, points=RESIDUAL_DOWNSAMPLE_POINTS):
    """Time indices of the 8-point downsample - round(linspace(0, T-1, points))."""
    if n_times <= 0:
        return []
    return [int(round(v)) for v in np.linspace(0, n_times - 1, min(points, max(n_times, 1)))]


def build_belief_packet(public_observations, active_fits, archive, constraints, interface, round):
    """Belief packet assembled from public state only (ForceBench edition).

    These are the only arguments - there is **no** held-out design, held-out observation, true value, score or
    evaluation object.

    public_observations
        Public rows per design: `design_id`, `p1`, `p2`, `initial_state`, `integration_steps`,
        `integration_status`, plus the observed trajectory `y` (T,2), which is used only for the residual computation
        and is **not carried** in the packet.
    active_fits
        List of `PoolMember`.  The highest-weight model must have `predictions` filled in.
    archive
        Every model produced in this run + structured reasons.
    constraints
        Catalogue-selection constraints (`catalog_constraints`).
    interface
        The machine-generated task block (`forcebench_interface`).
    round
        Current round t.
    """
    members = list(active_fits)
    if not members:
        raise ValueError("belief packet requires a non-empty active pool")
    if not constraint_keys_ok(constraints):
        # The allowed sets merely grew to two (catalogue and free); it is still an **exact match**.
        raise ValueError("belief packet constraints keys are fixed")
    if tuple(sorted(interface)) != tuple(sorted(INTERFACE_KEYS)):
        raise ValueError("interface must be exactly the generated structured block")
    rows = list(public_observations)
    if len(rows) > MAX_DESIGNS:
        raise ValueError(f"belief packet carries at most {MAX_DESIGNS} designs")

    observations = [
        {
            "design_id": str(row["design_id"]),
            "p1": float(row["p1"]),
            "p2": float(row["p2"]),
            "initial_state": {
                "pos": [float(v) for v in row["initial_state"]["pos"]],
                "vel": [float(v) for v in row["initial_state"]["vel"]],
            },
            "integration_steps": int(row["integration_steps"]),
            "integration_status": str(row["integration_status"]),
        }
        for row in rows
    ]

    order = sorted(range(len(members)), key=lambda i: (-members[i].weight, members[i].model_id))
    pool = []
    for index in order:
        member = members[index]
        pool.append(
            {
                "model_id": member.model_id,
                "expression": member.expression,
                "family": member.family,
                "weight": member.weight,
                "log_evidence": member.log_evidence,
                "log_evidence_computed_round": member.log_evidence_computed_round,
                "log_evidence_n_obs": member.log_evidence_n_obs,
                "born_round": member.born_round,
                "coefficients": {
                    name: {"mean": float(stat["mean"]), "sd": float(stat["sd"])}
                    for name, stat in member.coefficients.items()
                },
                "family_dispersion": {
                    name: {"mean": float(stat["mean"]), "sd": float(stat["sd"])}
                    for name, stat in member.family_dispersion.items()
                },
                "one_step_log_scores": list(member.one_step_log_scores),
                "one_step_rounds": list(member.one_step_rounds),
            }
        )

    best = members[order[0]]
    if best.predictions is None:
        raise ValueError("the highest-weight member must carry its public predictions")
    pred_mean = np.asarray(best.predictions["pred_mean"], float)
    pred_sd = np.asarray(best.predictions["pred_sd"], float)
    accel_abs = np.asarray(best.predictions["accel_abs"], float)
    y = np.asarray([np.asarray(row["y"], float) for row in rows], float)
    if not (pred_mean.shape == pred_sd.shape == y.shape) or pred_mean.ndim != 3:
        raise ValueError("predictions and public observations must share shape (D, T, 2)")
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(pred_sd > 0, (y - pred_mean) / np.where(pred_sd > 0, pred_sd, 1.0), np.nan)
    radius = np.linalg.norm(pred_mean, axis=-1)                    # (D, T)
    picked = downsample_indices(pred_mean.shape[1])
    per_time = np.sqrt(np.mean(z**2, axis=-1))                     # (D, T)

    residuals = []
    for i, row in enumerate(observations):
        zi = z[i]
        finite = np.isfinite(zi)
        residuals.append(
            {
                "design_id": row["design_id"],
                "p1": row["p1"],
                "p2": row["p2"],
                "rms_standardized_residual": (
                    float(np.sqrt(np.mean(zi[finite] ** 2))) if finite.any() else None
                ),
                "max_abs_standardized_residual": (
                    float(np.max(np.abs(zi[finite]))) if finite.any() else None
                ),
                "min_r": float(np.min(radius[i])),
                "max_abs_accel": float(np.max(np.abs(accel_abs[i]))),
                "residual_trajectory_8pt": [
                    (float(per_time[i, k]) if np.isfinite(per_time[i, k]) else None) for k in picked
                ],
            }
        )

    entries = sorted(
        (_archive_entry(item) for item in archive),
        key=lambda item: (item["proposed_round"], item["model_id"]),
    )
    return {
        "packet_version": PACKET_VERSION,
        "interface": dict(interface),
        "round": int(round),
        "n_public": len(observations),
        "observations": observations,
        "pool": pool,
        "best_model": f"{best.expression} | {best.family}",
        "best_model_id": best.model_id,
        "best_model_residuals": residuals,
        "best_model_residual_definitions": dict(RESIDUAL_DEFINITIONS),
        "archive": entries,
        "constraints": dict(constraints),
    }


def build_empty_pool_packet(public_observations, archive, constraints, interface, round):
    """Unified protocol: the round-0 burst packet, when no model has been admitted yet.

    The same keys as `build_belief_packet` and the same public rows, with an empty pool: no best model, so no
    residual block (an empty list) and null best-model fields. Like that builder it takes no held-out, evaluation
    or score argument, and the observation rows carry the design and integration state only - never a trajectory.
    """
    if not constraint_keys_ok(constraints):
        raise ValueError("belief packet constraints keys are fixed")
    if tuple(sorted(interface)) != tuple(sorted(INTERFACE_KEYS)):
        raise ValueError("interface must be exactly the generated structured block")
    rows = list(public_observations)
    if len(rows) > MAX_DESIGNS:
        raise ValueError(f"belief packet carries at most {MAX_DESIGNS} designs")
    observations = [
        {
            "design_id": str(row["design_id"]),
            "p1": float(row["p1"]),
            "p2": float(row["p2"]),
            "initial_state": {
                "pos": [float(v) for v in row["initial_state"]["pos"]],
                "vel": [float(v) for v in row["initial_state"]["vel"]],
            },
            "integration_steps": int(row["integration_steps"]),
            "integration_status": str(row["integration_status"]),
        }
        for row in rows
    ]
    entries = sorted(
        (_archive_entry(item) for item in archive),
        key=lambda item: (item["proposed_round"], item["model_id"]),
    )
    return {
        "packet_version": PACKET_VERSION,
        "interface": dict(interface),
        "round": int(round),
        "n_public": len(observations),
        "observations": observations,
        "pool": [],
        "best_model": None,
        "best_model_id": None,
        "best_model_residuals": [],
        "best_model_residual_definitions": dict(RESIDUAL_DEFINITIONS),
        "archive": entries,
        "constraints": dict(constraints),
    }


# ==================================================================== parser and judgment
def _judge_free_candidate(candidate, constraints):
    """Reply-contract judgment of a free-expression candidate - family + grammar/roles/free constants/operator limit.
    Returns (ok, reason).

    The grammar check is done entirely by `fb1_grammar.compile_structure` (the vendored CORE grammar, roles
    positive/signed, free constants <= 3, operators <= the declared limit).  The reason sentence that module writes is
    returned **as is** - the reason for a verdict is not written in two places.
    """
    import fb1_grammar as fbg          # noqa: PLC0415 - loaded only in free mode

    family = candidate["family"]
    if not constraints["family_expansion"] and family != constraints["initial_family"]:
        return False, "family must equal constraints.initial_family when family_expansion is false"
    if family not in constraints["allowed_families"]:
        return False, "family must be one of constraints.allowed_families"
    given = candidate["parameters"]
    if not given:
        return False, "parameters must declare at least one coefficient"
    # The form normalization applies to free mode as is.  The expression is outside the catalogue, so there is no
    # declaration to compare against and it is called with declared=None (in that case a restated prior is not
    # checked - the contract of normalize_parameter_role).
    roles, _normalized, reason = normalize_parameter_roles(given, None)
    if reason is not None:
        return False, reason
    try:
        fbg.compile_structure(candidate["expression"], roles)
    except fbg.GrammarError as exc:
        return False, str(exc)
    except SyntaxError as exc:
        return False, f"expression does not parse: {exc}"
    return True, None


def judge_candidate(candidate, constraints):
    """Per-item judgment - grammar (free constants, roles), verbatim (S0), family (F0 / allowed).

    `candidate` must be a dict that already passed the shape check
    (`expression`, `family`, `parameters(dict)`).  Returns (ok, reason sentence).
    """
    expression = candidate["expression"]
    if not isinstance(expression, str):
        return False, "expression must be a string"
    if is_free_constraints(constraints):
        # ---- Free mode.  Instead of the catalogue constraint (verbatim): **grammar, roles, free constants, operator
        #      limit**.  Only reply-contract-level judgment is done here (cheap and pure).  The finiteness probe and the
        #      numerical-equivalence duplicate check run in the **admission stage** (run_gravity) - refusing here would
        #      make propose_one treat it as a 'reply-contract failure' and send a **repair re-call**.
        #      Asking for a repair of a grammatically correct answer is wrong (the catalogue path also handles the probe
        #      and duplicates outside propose_one), and it would even shift the order of the mock schedule.
        return _judge_free_candidate(candidate, constraints)
    if expression not in constraints["initial_structures"]:
        if constraints["structure_expansion"]:
            return False, (
                "structure expansion is declared but the catalog-selection contract "
                "only accepts expressions listed verbatim in constraints.initial_structures"
            )
        return False, "expression must be verbatim one of constraints.initial_structures"
    family = candidate["family"]
    if not constraints["family_expansion"] and family != constraints["initial_family"]:
        return False, "family must equal constraints.initial_family when family_expansion is false"
    if family not in constraints["allowed_families"]:
        return False, "family must be one of constraints.allowed_families"
    declared = constraints["structure_parameters"][expression]
    given = candidate["parameters"]
    if set(given) != set(declared):
        return False, (
            "parameters must name exactly the coefficients of the chosen expression: "
            + ", ".join(sorted(declared))
        )
    # Form normalization comes **right before** the role comparison - after the expression, family and
    # coefficient-name checks, so the order of refusal reasons is the same as in catalogue mode (an expression outside the
    # catalogue is still refused first by the verbatim rule).  `{"role": X}` is compared as X, and a restated prior is
    # refused clearly.
    roles, _normalized, reason = normalize_parameter_roles(given, declared)
    if reason is not None:
        return False, reason
    for name in sorted(roles):
        if roles[name] != declared[name]["role"]:
            return False, (
                f"coefficient {name} has the fixed role {declared[name]['role']} "
                f"in constraints.structure_parameters"
            )
    return True, None


def parse_candidate_object(text, constraints):
    """Judge one reply JSON object (the reply contract of prompt v2.0).

    Returns (candidate, ignored).  `candidate` is
    `{"structure","family","expression","parameters","model_id"}`.
    `ignored` is `{"ignored_parameters", "ignored_keys", "note"}`.
    A failure is a ValueError (the validation sentence verbatim) - the caller appends that sentence and re-calls
    **once only**.
    """
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise ValueError(f"reply is not valid JSON: {exc}") from exc
    shape = None
    if isinstance(payload, list):
        shape = "reply must be one JSON object, not an array"
    elif not isinstance(payload, dict):
        shape = "reply must be one JSON object"
    elif not REQUIRED_CANDIDATE_KEYS <= set(payload):
        shape = "proposal candidate must include expression/family/parameters"
    elif not isinstance(payload["parameters"], dict):
        shape = "parameters must map coefficient names to fixed roles"
    if shape is not None:
        raise ValueError(shape)
    dropped_parameters = {
        name: role
        for name, role in payload["parameters"].items()
        if name in RESERVED_VARIANCE_PARAMETER_NAMES
    }
    kept = {
        name: role
        for name, role in payload["parameters"].items()
        if name not in RESERVED_VARIANCE_PARAMETER_NAMES
    }
    note = payload.get("note", "")
    recorded = isinstance(note, str) and len(note) <= NOTE_MAX_CHARS
    accepted = REQUIRED_CANDIDATE_KEYS | (OPTIONAL_CANDIDATE_KEYS if recorded else set())
    dropped_keys = {k: v for k, v in payload.items() if k not in accepted}
    judged = {
        "expression": payload["expression"],
        "family": payload["family"],
        "parameters": kept,
    }
    ok, reason = judge_candidate(judged, constraints)
    if not ok:
        raise ValueError(reason)
    if is_free_constraints(constraints):
        # ---- Free mode.  The structure name is **derived from the canonical identifier** (it is not registered
        #      here - registration, probe and numerical equivalence run exactly once in the admission stage).  The same
        #      expression gives the same name, so the (structure, family) key separates duplicates as in catalogue mode.
        import fb1_grammar as fbg      # noqa: PLC0415

        kept, normalized_shape, _reason = normalize_parameter_roles(kept, None)
        free_model_id = fbg.structure_model_id(payload["expression"], kept)
        candidate = {
            "structure": fbg.structure_name(free_model_id),
            "family": payload["family"],
            "expression": payload["expression"],
            "parameters": kept,
            "model_id": model_id(payload["expression"], payload["family"], kept),
            "structure_model_id": free_model_id,
        }
        return candidate, {
            "ignored_parameters": dropped_parameters,
            "ignored_keys": dropped_keys,
            "normalized_parameter_shape": normalized_shape,
            "note": note if recorded else "",
        }
    # judge_candidate passed, so the expression is in the catalogue and every value can be normalized - the
    # **normalized role strings** are fixed here.  model_id depends on the form of the value
    # (`f"{name}:{parameters[name]}"`), so without this step the same candidate would get a different model_id from its
    # form alone.
    kept, normalized_shape, _reason = normalize_parameter_roles(
        kept, constraints["structure_parameters"][payload["expression"]]
    )
    structure = expression_to_structure()[payload["expression"]]
    candidate = {
        "structure": structure,
        "family": payload["family"],
        "expression": payload["expression"],
        "parameters": kept,
        "model_id": model_id(payload["expression"], payload["family"], kept),
    }
    return candidate, {
        "ignored_parameters": dropped_parameters,
        "ignored_keys": dropped_keys,
        "normalized_parameter_shape": normalized_shape,
        "note": note if recorded else "",
    }


# =========================================================================== transport
class _Usage:
    def __init__(self, payload):
        self._payload = dict(payload)

    def model_dump(self, mode="json"):
        """Provider-style usage dump (a copy of the stored payload)."""
        return dict(self._payload)


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, text, *, model, response_id, stop_reason="end_turn", usage=None):
        self.content = [_Block(text)]
        self.model = model
        self.id = response_id
        self.stop_reason = stop_reason
        self.usage = _Usage(usage or {"input_tokens": None, "output_tokens": None})


class _Messages:
    def __init__(self, owner):
        self._owner = owner

    def create(self, **body):
        """Forward a Messages API call to the owning client's `_create`."""
        return self._owner._create(**body)


class ScriptedClient:
    """Injected client for tests - returns pre-written reply strings in order.

    It **never** creates a real API client (zero paid calls).
    """

    provider = "scripted"

    def __init__(self, texts, *, model="mock-scripted", stop_reasons=None):
        self.texts = list(texts)
        self.model = model
        self.stop_reasons = list(stop_reasons or [])
        self.calls = 0
        self.bodies = []
        self.messages = _Messages(self)

    def _create(self, **body):
        self.bodies.append(body)
        index = self.calls
        self.calls += 1
        if index >= len(self.texts):
            raise RuntimeError("scripted client ran out of responses")
        stop = self.stop_reasons[index] if index < len(self.stop_reasons) else "end_turn"
        return _Response(
            self.texts[index],
            model=self.model,
            response_id=f"scripted_{index:03d}",
            stop_reason=stop,
            usage={"input_tokens": len(json.dumps(body)), "output_tokens": len(self.texts[index])},
        )


# Pre-declared mock schedule - **not tuned after seeing results**.
# The first 12 = the runner's MOCK_PROPOSALS_V3 laid out as is (a test compares them) -> the P4-equivalent path uses
# the same candidate space and the same order as the mock proposer.  The last 4 are catalogue pairs not used yet,
# written down in advance so that P1 (one per round) fills 16 rounds.
MOCK_SCHEDULE = [
    ("power", "gauss"),
    ("inv_r_plus_c", "gauss"),
    ("inv_r2", "student_t4"),
    ("inv_r", "gauss"),
    ("inv_r_bg", "gauss"),
    ("screened", "student_t4"),
    ("inv_r_p1sq", "gauss"),
    ("inv_r_nomass", "gauss"),
    ("inv_r_tmod", "gauss"),
    ("prod_coupling", "gauss"),
    ("crossover", "gauss"),
    ("inv_r_bg", "student_t4"),
    ("logr", "gauss"),
    ("power", "student_t4"),
    ("crossover", "student_t4"),
    ("prod_coupling", "student_t4"),
]
MOCK_NOTE = "pre-declared mock schedule entry; no model call was made"

# ---------------------------------------------------- free-mode mock schedule (pre-declared)
# The three expressions fixed in advance.  Their expected verdicts are written down too -
# leaving no room to choose after the fact.
#   (1) a valid expression outside the catalogue     -> active
#   (2) a non-finite expression                      -> nonfinite_on_probe_grid (a grid that includes the time axis)
#   (3) a surface variant of a catalogue expression  -> duplicate_by_numerical_equivalence
#       The canonical identifier does not catch it (commutativity is not normalized) - which is why the numerical
#       check was added.
FREE_MOCK_SCHEDULE = [
    ("c*p1/(r**2 + d)", "gauss", {"c": "positive", "d": "positive"}),
    ("c*p1*log(r - 5)", "gauss", {"c": "positive"}),
    ("p1*c/r**2", "gauss", {"c": "positive"}),
]
FREE_MOCK_EXPECTED_OUTCOMES = ("active", "nonfinite_on_probe_grid",
                               "duplicate_by_numerical_equivalence")


def free_mock_response_text(index, schedule=None):
    """Write entry `index` of the free schedule in the prompt v2.0 reply contract (one JSON object)."""
    schedule = FREE_MOCK_SCHEDULE if schedule is None else schedule
    expression, family, parameters = schedule[index % len(schedule)]
    return json.dumps(
        {
            "expression": expression,
            "family": family,
            "parameters": dict(parameters),
            "note": MOCK_NOTE,
        },
        sort_keys=True,
    )


class MockFreeClient(ScriptedClient):
    """Default free-mode transport.  Yields `FREE_MOCK_SCHEDULE` in call order - zero paid calls."""

    provider = "mock"

    def __init__(self, schedule=None, *, calls=64):
        schedule = FREE_MOCK_SCHEDULE if schedule is None else schedule
        super().__init__(
            [free_mock_response_text(i, schedule) for i in range(calls)],
            model="mock-free-v1",
        )
        self.schedule = list(schedule)


def mock_response_text(index, schedule=None):
    """Write pair `index` of the schedule in the prompt v2.0 reply contract (one JSON object)."""
    schedule = MOCK_SCHEDULE if schedule is None else schedule
    structure, family = schedule[index % len(schedule)]
    expression = canonical_expression(structure)
    kinds = STRUCTURES[structure].kinds
    parameters = {
        name: KIND_ROLE[kind] for name, kind in zip(COEFFICIENT_NAMES[structure], kinds)
    }
    return json.dumps(
        {
            "expression": expression,
            "family": family,
            "parameters": parameters,
            "note": MOCK_NOTE,
        },
        sort_keys=True,
    )


class MockCatalogClient(ScriptedClient):
    """Default transport.  Yields `MOCK_SCHEDULE` in call order - zero network and paid calls."""

    provider = "mock"

    def __init__(self, schedule=None, *, calls=64):
        schedule = MOCK_SCHEDULE if schedule is None else schedule
        super().__init__(
            [mock_response_text(i, schedule) for i in range(calls)], model="mock-catalog-v1"
        )
        self.schedule = list(schedule)


# Prompt variant: a development-only system prompt file, selected by the runner from
# FB1_PROMPT_VARIANT. Same rule as count_proposer.set_prompt_variant: line 1 of the file is
# exactly `# prompt_version: <tag>`, the file is sent as stored, and `<tag>` and the file's
# sha256 are what the records say was sent. None (the default) is the frozen v2.0 file.
PROMPT_VARIANT_TAG_PREFIX = "# prompt_version:"
_PROMPT_VARIANT = None


def set_prompt_variant(path):
    """Select (or, with None, clear) the prompt variant; returns the previous (path, tag) or None."""
    global _PROMPT_VARIANT
    previous = _PROMPT_VARIANT
    if path is None:
        _PROMPT_VARIANT = None
        return previous
    content = Path(path).read_text()
    first_line = content.splitlines()[0] if content else ""
    if not first_line.startswith(PROMPT_VARIANT_TAG_PREFIX):
        raise ValueError(
            "prompt variant file must declare its tag on line 1 as "
            f"{PROMPT_VARIANT_TAG_PREFIX!r} <tag>; {str(path)!r} does not")
    tag = first_line[len(PROMPT_VARIANT_TAG_PREFIX):].strip()
    if not tag:
        raise ValueError(f"prompt variant file {str(path)!r} declares an empty tag")
    _PROMPT_VARIANT = (str(path), tag)
    return previous


def prompt_variant_active():
    """True when a prompt variant file is selected (`set_prompt_variant`)."""
    return _PROMPT_VARIANT is not None


def system_prompt_bytes():
    """Bytes of the system prompt that is sent: the selected variant file, else the frozen v2.0 file."""
    if _PROMPT_VARIANT is not None:
        return Path(_PROMPT_VARIANT[0]).read_bytes()
    return PROMPT_FILE.read_bytes()


def prompt_identity():
    """{prompt_version, prompt_sha256} of the system prompt that is sent."""
    return {
        "prompt_version": PROMPT_VERSION if _PROMPT_VARIANT is None else _PROMPT_VARIANT[1],
        "prompt_sha256": hashlib.sha256(system_prompt_bytes()).hexdigest(),
    }


# ----------------------------------------------- unified protocol: the request kinds
# Unified protocol (the registered protocol): four request
# kinds beside the default one (request=None, every byte as before). Each sends the selected prompt file's bytes followed
# by general paragraphs. The two burst paragraphs are the count runners' (count_proposer.BURST_PARAGRAPH,
# BURST_REFILL_PARAGRAPH) copied byte for byte; COMPOSITION_PARAGRAPH_V2 and ARCHIVE_REVIEW_PARAGRAPH_V1 are the
# registered files composition_paragraph_v2.txt and archive_review_paragraph_v1.txt byte for byte, `{limit}` included
# (tests compare them). The runner fills `{limit}` by text replacement (the review paragraph also holds literal braces)
# and only with 3. The file's own prompt_version / prompt_sha256 keep naming the file; the paragraph files' digests and
# the composed text's digest are recorded beside them.
#   burst            round 0, one of several independent single-candidate calls on the empty-pool packet
#   burst_refill     round 0 refill wave, single candidate, the packet shows the pool admitted so far
#   composed         each proposal round, ONE call answering a JSON array of three candidates (COMPOSITION v2)
#   composed_review  an archive review round: COMPOSITION v2 + ARCHIVE_REVIEW v1, the reply is one JSON object
#                    {"candidates": [the array], "revive": [0-2 archive model_id strings]}
BURST_PARAGRAPH = (
    "This request is one of several independent requests made in parallel for the same "
    "round; each answers with one candidate and none sees the others. When the pool is "
    "empty (round 0) there is no residual pattern yet: propose one candidate model for "
    "the process the interface and observations describe. Keep the family "
    "constraints.initial_family; every other rule applies unchanged.\n"
)
BURST_REFILL_PARAGRAPH = (
    "This request is one of several independent requests made in parallel for the same "
    "round; each answers with one candidate and none sees the others. The pool in this "
    "packet already holds the candidates admitted by earlier requests of this round. "
    "Propose one candidate whose functional form is not already in the pool; a candidate "
    "with the same canonical identity as a pool member is discarded as a duplicate. Keep "
    "the family constraints.initial_family; every other rule applies unchanged.\n"
)
#: COMPOSITION v2: the registered file composition_paragraph_v2.txt
#: (sha256 782a30b74cd25adb6b51351096cf97e31a7375d8630d726ee8eb7d7949c35be6, 525 bytes with the trailing newline).
COMPOSITION_PARAGRAPH_V2 = (
    'Return a JSON array of exactly {limit} candidate objects, each with exactly the keys specified '
    'above and nothing else. Propose three distinct, non-equivalent hypotheses. At least two of the '
    'three candidates should differ meaningfully in their primary functional composition, rather than'
    ' merely in parameterization or by adding or removing a correction term. At most one candidate '
    'may be a nested or incremental refinement of a high-posterior model, and only if such a '
    'refinement is specifically supported by the diagnostics.\n'
)
COMPOSITION_PARAGRAPH_V2_SHA256 = "782a30b74cd25adb6b51351096cf97e31a7375d8630d726ee8eb7d7949c35be6"
#: ARCHIVE_REVIEW v1: the registered file archive_review_paragraph_v1.txt
#: (sha256 8b4c5359d81b6b3e5afed570c1e3c3f1bb217a57e70977bcbdb42d1238cd6cde, 675 bytes with the trailing newline).
ARCHIVE_REVIEW_PARAGRAPH_V1 = (
    'This is an archive review round. Besides your {limit} new candidates, you may ask for up to two '
    'archived models to be re-fit: choose only among archive entries whose outcome is '
    'retired_by_evidence and whose form you now judge likely given all current observations (they '
    'were retired on less data). Each model you name is re-fit from its prior on all current public '
    'data and competes for the pool on evidence like any other model. Reply with one raw JSON object '
    '{"candidates": [...], "revive": [...]}: candidates is the array described above, and revive '
    'lists the chosen archive model_id strings exactly as given, or is empty when nothing in the '
    'archive deserves another fit.\n'
)
ARCHIVE_REVIEW_PARAGRAPH_V1_SHA256 = "8b4c5359d81b6b3e5afed570c1e3c3f1bb217a57e70977bcbdb42d1238cd6cde"
PROPOSAL_LIMIT = 3                # COMPOSITION v2 names three candidates; it is used with no other limit
REVIVE_LIMIT = 2                  # ARCHIVE_REVIEW v1: "up to two archived models"
REQUEST_KINDS = ("burst", "burst_refill", "composed", "composed_review")
BURST_REQUEST_KINDS = ("burst", "burst_refill")
COMPOSED_REQUEST_KINDS = ("composed", "composed_review")
BURST_CONTEXT_KEYS = frozenset({"burst_wave", "burst_call_index"})
#: The repair turn of a composed request asks again for what was asked: the array, or on a review round the object.
REPAIR_INSTRUCTION_ARRAY = (
    "Your previous reply failed validation with this error:\n{error}\n"
    "Reply with exactly one JSON array of {limit} objects, each "
    '{{"expression": ..., "family": ..., "parameters": {{...}}}}, and nothing else.\n'
    'Each value in parameters is the role STRING itself, not an object: '
    '{{"c": "positive"}}, never {{"c": {{"role": "positive"}}}}.'
)
REPAIR_INSTRUCTION_REVIEW = (
    "Your previous reply failed validation with this error:\n{error}\n"
    'Reply with exactly one JSON object {{"candidates": [...], "revive": [...]}}: candidates is an array of {limit} '
    'objects, each {{"expression": ..., "family": ..., "parameters": {{...}}}}, and revive lists 0 to 2 archive '
    "model_id strings; nothing else.\n"
    'Each value in parameters is the role STRING itself, not an object: '
    '{{"c": "positive"}}, never {{"c": {{"role": "positive"}}}}.'
)
REPAIR_INSTRUCTION_ARRAY_EMPTY = "Your previous reply was empty (finish_reason={finish_reason}).\n" + \
    REPAIR_INSTRUCTION_ARRAY.split("\n", 2)[2]
REPAIR_INSTRUCTION_REVIEW_EMPTY = "Your previous reply was empty (finish_reason={finish_reason}).\n" + \
    REPAIR_INSTRUCTION_REVIEW.split("\n", 2)[2]


def fill_limit(text, limit):
    """`{limit}` replaced by the number; the review paragraph's literal braces stay (no str.format)."""
    if limit != PROPOSAL_LIMIT:
        raise ValueError(f"the unified paragraphs are used with limit {PROPOSAL_LIMIT} only, got {limit!r}")
    filled = text.replace("{limit}", str(limit))
    assert "{limit}" not in filled
    return filled


def validate_request(request, proposal_limit=None):
    """None (the default request) or one of REQUEST_KINDS; a composed kind needs proposal_limit 3."""
    if request is not None and request not in REQUEST_KINDS:
        raise ValueError(f"request must be one of {REQUEST_KINDS}, got {request!r}")
    if request in COMPOSED_REQUEST_KINDS:
        if proposal_limit != PROPOSAL_LIMIT:
            raise ValueError(f"a composed request asks for {PROPOSAL_LIMIT} candidates, got {proposal_limit!r}")
    elif proposal_limit is not None:
        raise ValueError("proposal_limit belongs to a composed request")
    return request


def request_prompt(request, proposal_limit=None):
    """(system text, identity block) of one request kind; request None is the stored file and prompt_identity()."""
    validate_request(request, proposal_limit)
    text = system_prompt_bytes().decode()
    identity = prompt_identity()
    if request is None:
        return text, identity
    if request in BURST_REQUEST_KINDS:
        paragraph, key = ((BURST_PARAGRAPH, "burst_prompt_sha256") if request == "burst"
                          else (BURST_REFILL_PARAGRAPH, "burst_refill_prompt_sha256"))
        composed = text + paragraph
        return composed, {**identity, key: hashlib.sha256(composed.encode()).hexdigest()}
    composed = text + fill_limit(COMPOSITION_PARAGRAPH_V2, proposal_limit)
    block = {**identity, "proposal_limit": proposal_limit,
             "composition_paragraph_sha256": COMPOSITION_PARAGRAPH_V2_SHA256}
    if request == "composed_review":
        composed += fill_limit(ARCHIVE_REVIEW_PARAGRAPH_V1, proposal_limit)
        block["archive_review_paragraph_sha256"] = ARCHIVE_REVIEW_PARAGRAPH_V1_SHA256
    block["composed_prompt_sha256"] = hashlib.sha256(composed.encode()).hexdigest()
    return composed, block


def request_prompt_identities():
    """Every digest a unified run can send, for the run record: the file, each paragraph file, each composed text."""
    out = {"base_prompt_sha256": hashlib.sha256(system_prompt_bytes()).hexdigest(),
           "burst_paragraph_sha256": hashlib.sha256(BURST_PARAGRAPH.encode()).hexdigest(),
           "burst_refill_paragraph_sha256": hashlib.sha256(BURST_REFILL_PARAGRAPH.encode()).hexdigest(),
           "composition_paragraph_sha256": COMPOSITION_PARAGRAPH_V2_SHA256,
           "archive_review_paragraph_sha256": ARCHIVE_REVIEW_PARAGRAPH_V1_SHA256,
           "burst_prompt_sha256": request_prompt("burst")[1]["burst_prompt_sha256"],
           "burst_refill_prompt_sha256": request_prompt("burst_refill")[1]["burst_refill_prompt_sha256"],
           "composed_prompt_sha256": request_prompt("composed", PROPOSAL_LIMIT)[1]["composed_prompt_sha256"],
           "composed_review_prompt_sha256":
               request_prompt("composed_review", PROPOSAL_LIMIT)[1]["composed_prompt_sha256"],
           "proposal_limit": PROPOSAL_LIMIT}
    return out


ARRAY_REFUSAL_PREFIX = "no candidate in the array was accepted: "


def parse_candidate_array(text, constraints, proposal_limit):
    """A composed reply: a JSON array of 1..`proposal_limit` candidate objects (the count runners' contract,
    count_proposer.parse_candidate_array). Each object is judged by exactly the one-object rule (`parse_candidate_object`)
    and a refused object is recorded with its index without refusing the others; the batch is refused (one repair turn)
    only for text that is not JSON, a payload that is not an array, an empty or over-long array, or an array in which
    nothing was accepted. Returns (candidates in payload order, ignored) with per-candidate lists, their notes and
    `rejected_candidates`."""
    try:
        payload = json.loads(text) if isinstance(text, str) else text
    except Exception as exc:
        raise ValueError(f"reply is not valid JSON: {exc}") from exc
    return _judge_candidate_list(payload, constraints, proposal_limit,
                                 f"reply must be a JSON array of {proposal_limit} candidates")


def _judge_candidate_list(payload, constraints, proposal_limit, shape_message):
    if not isinstance(payload, list):
        raise ValueError(shape_message)
    if not payload or len(payload) > proposal_limit:
        raise ValueError(f"reply must hold 1..{proposal_limit} candidates")
    candidates, notes, rejected = [], [], []
    for index, item in enumerate(payload):
        mapping = item if isinstance(item, dict) else {}
        try:
            if not isinstance(item, dict):
                raise ValueError("each array item must be one JSON object")
            candidate, ignored = parse_candidate_object(json.dumps(item), constraints)
        except ValueError as exc:
            parameters = mapping.get("parameters")
            rejected.append({"index": index, "expression": mapping.get("expression"),
                             "family": mapping.get("family"), "reason": str(exc),
                             "parameters": parameters if isinstance(parameters, dict) else None})
            continue
        candidates.append(candidate)
        notes.append(ignored)
    if not candidates:
        raise ValueError(ARRAY_REFUSAL_PREFIX + "; ".join(entry["reason"] for entry in rejected))
    return candidates, {
        "ignored_parameters": [note["ignored_parameters"] for note in notes],
        "ignored_keys": [note["ignored_keys"] for note in notes],
        "normalized_parameter_shape": [note["normalized_parameter_shape"] for note in notes],
        "note": next((note["note"] for note in notes if note["note"]), ""),
        "notes": [note["note"] for note in notes],
        "rejected_candidates": rejected,
    }


def parse_review_reply(text, constraints, proposal_limit):
    """An archive review round's reply: one JSON object {"candidates": [...], "revive": [...]}.

    `candidates` is judged exactly as a composed array (`parse_candidate_array`'s rules; its failures are the batch
    failures that earn the one repair turn). `revive` is optional: absent means none, and whatever it holds is returned
    as written for the runner to validate - an id that is not usable is logged and ignored there, never repaired here. A
    bare array is read as the candidates with no revive (`reply_shape` "array", `revive_present` false). Returns
    (candidates, ignored) where ignored also carries `revive_requested`, `revive_present`,
    `reply_shape` and `ignored_review_keys`."""
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise ValueError(f"reply is not valid JSON: {exc}") from exc
    if isinstance(payload, list):
        candidates, ignored = _judge_candidate_list(payload, constraints, proposal_limit, "")
        return candidates, {**ignored, "revive_requested": [], "revive_present": False, "reply_shape": "array",
                            "ignored_review_keys": {}}
    if not isinstance(payload, dict) or "candidates" not in payload:
        raise ValueError('reply must be one JSON object {"candidates": [...], "revive": [...]}')
    candidates, ignored = _judge_candidate_list(payload["candidates"], constraints, proposal_limit,
                                                f"candidates must be a JSON array of {proposal_limit} candidates")
    revive = payload.get("revive", [])
    return candidates, {**ignored, "revive_requested": revive if isinstance(revive, list) else [revive],
                        "revive_present": "revive" in payload,
                        "revive_shape": "list" if isinstance(revive, list) else type(revive).__name__,
                        "reply_shape": "object",
                        "ignored_review_keys": {k: v for k, v in payload.items() if k not in ("candidates", "revive")}}


#: Unified protocol: pre-declared mock schedules of the new request kinds (free mode), written before any run from generic
#: central-force forms; none comes from a model reply. The burst schedule's expected admission outcomes on an empty pool
#: are declared beside it. With the protocol's burst of twelve, wave 0 (entries 0-11) admits ten and refuses entry 4
#: (non-finite on the probe grid) and entry 7 (a commuted transcription of entry 0, a duplicate), so one refill wave of
#: two calls (entries 12 and 13) completes the twelve.
BURST_MOCK_SCHEDULE = [
    ("c*p1/r**2", "gauss", {"c": "positive"}),
    ("c*p1/(r**2 + d)", "gauss", {"c": "positive", "d": "positive"}),
    ("c*p1*r", "gauss", {"c": "positive"}),
    ("c*p1/r", "gauss", {"c": "positive"}),
    ("c*p1*log(r - 5)", "gauss", {"c": "positive"}),
    ("c*p1/r**3", "gauss", {"c": "positive"}),
    ("c*p1*exp(-r/lam)/r", "gauss", {"c": "positive", "lam": "positive"}),
    ("p1*c/r**2", "gauss", {"c": "positive"}),
    ("c*p1*tanh(r/d)", "gauss", {"c": "positive", "d": "positive"}),
    ("c*p1*p2/r**2", "gauss", {"c": "positive"}),
    ("c*p1*sqrt(r)", "gauss", {"c": "positive"}),
    ("c*p1", "gauss", {"c": "positive"}),
    ("c*p1/(1 + r)", "gauss", {"c": "positive"}),
    ("c*p1*exp(-r)", "gauss", {"c": "positive"}),
    ("c*p1*r/(1 + r**2)", "gauss", {"c": "positive"}),
    ("c*p1*p2*r", "gauss", {"c": "positive"}),
]
#: The runner's archive outcome of every entry in that order, on an empty pool (entries 0, 2, 3, 6 and 11 are catalogue
#: references, admitted under the catalogue name; entry 4 is non-finite on the probe grid; entry 7 matches entry 0
#: numerically; entries 14 and 15 are spares a third wave would read).
BURST_MOCK_EXPECTED_OUTCOMES = ("active", "active", "active", "active", "rejected", "active", "active",
                                "duplicate", "active", "active", "active", "active", "active", "active",
                                "active", "active")
#: Composed replies, one triple per proposal round. The first eight triples are written by hand and transcribe burst or
#: catalogue forms on purpose (exact duplicates recorded and not refitted); the generated triples after them are distinct
#: generic central-force forms (literal exponents and scales, one or two constants, within the grammar's limits), mixed
#: across templates, so a 30-round mock still meets three new forms per round - the load of a paid cell. With the trig
#: grammar on (the packet's grammar lists cos and sin) the third form of every triple is a time-modulated one.
_HAND_TRIPLES = [
    [("c*p1/r**2 + b*p1*r", "gauss", {"c": "positive", "b": "positive"}),
     ("c*p1*r**(-a)", "gauss", {"c": "positive", "a": "signed"}),
     ("c*p1/r", "student_t4", {"c": "positive"})],
    [("c*p1*exp(-r/lam)/r**2", "gauss", {"c": "positive", "lam": "positive"}),
     ("c*p1*p2/r", "gauss", {"c": "positive"}),
     ("c*p1/(r + d)**2", "gauss", {"c": "positive", "d": "positive"})],
    [("c*p1**2/r", "gauss", {"c": "positive"}),
     ("c*p1*log(1+k*r)/r", "gauss", {"c": "positive", "k": "positive"}),
     ("p1*c/r**2", "gauss", {"c": "positive"})],
    [("c*p1*exp(-r)/r", "gauss", {"c": "positive"}),
     ("c*p1*tanh(r)/r**2", "gauss", {"c": "positive"}),
     ("c*p1/r - b*p1*r", "gauss", {"c": "positive", "b": "positive"})],
    [("c*p1/(r**2 + d**2)", "gauss", {"c": "positive", "d": "positive"}),
     ("c*p1*r/(d + r**3)", "gauss", {"c": "positive", "d": "positive"}),
     ("c*p1*exp(-r/lam)", "gauss", {"c": "positive", "lam": "positive"})],
    [("c*p1/r**2 + b*p1", "gauss", {"c": "positive", "b": "positive"}),
     ("c*p1*sqrt(r)/r**2", "gauss", {"c": "positive"}),
     ("c*p1*p2/r**3", "gauss", {"c": "positive"})],
    [("c*p1/(r*(1 + r))", "gauss", {"c": "positive"}),
     ("c*p1*tanh(d/r)", "gauss", {"c": "positive", "d": "positive"}),
     ("c*p1*r**2/(1 + r**4)", "gauss", {"c": "positive"})],
    [("c*p1/r**2", "student_t4", {"c": "positive"}),
     ("c*p1*exp(-r*r/lam)/r", "gauss", {"c": "positive", "lam": "positive"}),
     ("c*p1/(d + r)", "gauss", {"c": "positive", "d": "positive"})],
]
_ONE = {"c": "positive"}
_TWO = {"c": "positive", "d": "positive"}
_TEMPLATES = [
    ("c*p1/r**{v}", _ONE, (0.5, 0.75, 1.25, 1.75, 2.25, 2.5, 2.75, 3.5, 4)),
    ("c*p1*exp(-r/{v})/r", _ONE, (0.5, 0.75, 1.5, 2, 2.5, 3, 4, 5, 6)),
    ("c*p1/(r**2 + {v})", _ONE, (0.25, 0.5, 2, 3, 4, 6)),
    ("c*p1*r/(1 + r**{v})", _ONE, (1.5, 2.5, 3, 3.5)),
    ("c*p1*p2**{v}/r**2", _ONE, (0.25, 0.5, 0.75, 1.5)),
    ("c*p1*tanh(r/{v})/r**2", _ONE, (0.5, 2, 3, 4)),
    ("c*p1*log(1 + r/{v})/r**2", _ONE, (0.5, 1, 2, 4)),
    ("c*p1/(r**{v} + d)", _TWO, (1.5, 2.5, 3)),
    ("c*p1*exp(-d*r)/r**{v}", _TWO, (0.5, 1.5, 2)),
    ("c*p1*(1 - exp(-r/{v}))/r**2", _ONE, (0.5, 1, 2, 4)),
    ("c*p1*r**{v}", _ONE, (0.25, 0.75, 1.5)),
    ("c*p1/(1 + r)**{v}", _ONE, (1.5, 2, 2.5, 3)),
    ("c*p1*exp(-(r/{v})**2)", _ONE, (1, 2, 3, 5)),
    ("c*p1/(r*(1 + r/{v}))", _ONE, (0.5, 2, 4, 8)),
    ("c*p1*p2/(r**2 + {v})", _ONE, (0.5, 1, 2)),
    ("c*p1*exp(-r/{v})", _ONE, (2, 3, 4, 8)),
    ("c*p1*tanh({v}/r)", _ONE, (1, 2, 4)),
    ("c*p1/r**2 + d*p1/r**{v}", _TWO, (3, 4)),
    ("c*p1/r + d*p1/r**{v}", _TWO, (2.5, 3)),
    ("c*p1/(r**3 + {v})", _ONE, (1, 2, 4, 8)),
    ("c*p1*r/(r**3 + {v})", _ONE, (1, 3)),
    ("c*p1*exp(-r/{v})/r**2", _ONE, (0.5, 2, 4, 8)),
]


def _generated_forms():
    """The templates' forms, interleaved across templates (round robin) so neighbouring forms differ in kind."""
    columns = [[(template.format(v=value), "gauss", dict(parameters)) for value in values]
               for template, parameters, values in _TEMPLATES]
    out = []
    for index in range(max(len(column) for column in columns)):
        out.extend(column[index] for column in columns if index < len(column))
    return out


GENERATED_MOCK_FORMS = _generated_forms()
COMPOSED_MOCK_SCHEDULE = _HAND_TRIPLES + [GENERATED_MOCK_FORMS[i:i + 3]
                                          for i in range(0, len(GENERATED_MOCK_FORMS) - 2, 3)]
TRIG_MOCK_FORMS = ([(f"c*p1*(1 + d*cos({w}*t))/r", "gauss", {"c": "positive", "d": "signed"})
                    for w in (0.5, 1, 1.5, 2, 2.5, 3, 4, 5)]
                   + [(f"c*p1*(1 + d*sin({w}*t))/r", "gauss", {"c": "positive", "d": "signed"})
                      for w in (0.5, 1, 1.5, 2, 2.5, 3, 4, 5)])
#: The id the mock review reply names that no archive holds (the refusal path "not an archived model").
MOCK_UNKNOWN_MODEL_ID = "0" * 20


def composed_mock_candidates(index, schedule=None, trig=False):
    """Candidate objects of mock composed reply `index` (cycled); with `trig`, the third form is time-modulated."""
    schedule = COMPOSED_MOCK_SCHEDULE if schedule is None else schedule
    triple = list(schedule[index % len(schedule)])
    if trig:
        triple[-1] = TRIG_MOCK_FORMS[index % len(TRIG_MOCK_FORMS)]
    return [{"expression": expression, "family": family, "parameters": dict(parameters), "note": MOCK_NOTE}
            for expression, family, parameters in triple]


def _packet_has_trig(packet_text):
    """True when the packet's grammar lists cos and sin (FB1_GRAMMAR_TRIG=1)."""
    try:
        calls = json.loads(packet_text)["constraints"]["grammar"]["calls"]
    except Exception:  # noqa: BLE001 - a mock never fails on its input
        return False
    return "cos" in calls and "sin" in calls


def _mock_revive_rank(row):
    """Sort key of a retired archive row for the mock's first revive id: the latest retirement round first, then the
    smallest gap to the cutoff at retirement (the nearest miss, the model most likely to re-enter on more data)."""
    reason = row.get("reason") if isinstance(row.get("reason"), dict) else {}
    computed, gap = reason.get("computed_round"), reason.get("gap_to_cutoff_nats")
    computed = computed if isinstance(computed, (int, float)) and math.isfinite(computed) else -math.inf
    gap = gap if isinstance(gap, (int, float)) and math.isfinite(gap) else math.inf
    return (-computed, gap)


def mock_revive_list(packet_text):
    """The mock review reply's revive list, read off the packet it was sent (as a model reads it): the retired_by_evidence
    archive id retired most recently and nearest to the cutoff (archive order breaks ties), an id no archive holds, then
    the first other retired id in archive order - so one round exercises a valid revive that can re-enter the pool, an
    unknown id and an id past the second place (over_limit)."""
    try:
        archive = json.loads(packet_text).get("archive", [])
    except Exception:  # noqa: BLE001 - a mock never fails on its input
        archive = []
    retired = [row for row in archive if row.get("outcome") == "retired_by_evidence"]
    if not retired:
        return [MOCK_UNKNOWN_MODEL_ID]
    first = min(range(len(retired)), key=lambda i: (_mock_revive_rank(retired[i]), i))
    others = [row["model_id"] for i, row in enumerate(retired) if i != first]
    return [retired[first]["model_id"], MOCK_UNKNOWN_MODEL_ID] + others[:1]


class MockUnifiedClient(ScriptedClient):
    """Unified protocol: the mock transport of the burst and composed requests (free mode) - no network, no paid call.

    Burst and refill calls read BURST_MOCK_SCHEDULE in call order; composed calls read COMPOSED_MOCK_SCHEDULE (cycled),
    and on a review round (the system text ends with the archive review paragraph) the reply is the object whose revive
    list `mock_revive_list` reads off the packet. The runner makes mock calls one at a time, so every byte of a mock run
    is fixed. The default request never reaches this client.
    """

    provider = "mock"

    def __init__(self, *, name_mask=None, calls=128):
        super().__init__([], model="mock-unified-v1")
        self.name_mask = name_mask
        self.burst_texts = [free_mock_response_text(i, BURST_MOCK_SCHEDULE) for i in range(calls)]
        if name_mask is not None:
            # As the default mock: it answers in the names it is shown.
            self.burst_texts = [name_mask.mask_reply(text) for text in self.burst_texts]
        self.burst_calls = self.composed_calls = 0

    def _shown(self, candidate):
        if self.name_mask is None:
            return candidate
        return json.loads(self.name_mask.mask_reply(json.dumps(candidate, sort_keys=True)))

    def _create(self, **body):
        self.bodies.append(body)
        index = self.calls
        self.calls += 1
        system = body.get("system", "")
        if system.endswith(BURST_PARAGRAPH) or system.endswith(BURST_REFILL_PARAGRAPH):
            text = self.burst_texts[self.burst_calls]
            self.burst_calls += 1
        else:
            packet_text = body["messages"][0]["content"]
            candidates = [self._shown(c) for c in composed_mock_candidates(self.composed_calls,
                                                                            trig=_packet_has_trig(packet_text))]
            self.composed_calls += 1
            if system.endswith(fill_limit(ARCHIVE_REVIEW_PARAGRAPH_V1, PROPOSAL_LIMIT)):
                text = json.dumps({"candidates": candidates,
                                   "revive": mock_revive_list(packet_text)}, sort_keys=True)
            else:
                text = json.dumps(candidates, sort_keys=True)
        return _Response(text, model=self.model, response_id=f"mock_unified_{index:03d}", stop_reason="end_turn",
                         usage={"input_tokens": len(json.dumps(body)), "output_tokens": len(text)})


# ------------------------------------------------- input-name masking (default off)
# The engine keeps its own names (r, p1, p2, t): fb1_core.py and fb1_forward.py are protected
# files and the catalogue and the grammar are bound to those names. Masking happens at this
# module's boundary: everything that goes to the model is written with X1..X4, and the
# model's reply is translated back before it is judged. Random seeds and tie orders therefore
# use the original-name identifiers by construction, and observations, fits and scores cannot
# depend on the option.
#
# The translation is a swap of name segments (the parts of a [A-Za-z0-9_] token between
# underscores): r <-> X1, p1 <-> X2, p2 <-> X3, t <-> X4. It is its own inverse, so a
# coefficient the model happens to call `r` is the engine's `X1` and comes back as `r`.
# Three engine strings are not a pure rename and are rewritten before the swap (the left
# text is what an unmasked run sends):
MASK_TEXT_REWRITES = (
    # `t` here is a round index, not the input.
    ("at round t is first fit at round t+1 so its first score appears at round t+2",
     "at round k is first fit at round k+1 so its first score appears at round k+2"),
    # The catalogue writes the product of coefficient w and input t as one token.
    ("cos(wt)", "cos(w*t)"),
    ("sin(wt)", "sin(w*t)"),
)
_MASK_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_FREE_STRUCTURE_NAME = re.compile(r"g_[0-9a-f]{12}")
_IDENTIFIER = re.compile(r"[0-9a-f]{20}")
# Record fields whose string (or whose keys or elements) is a structure label.
STRUCTURE_NAME_FIELDS = frozenset({
    "target", "matched", "structure", "catalog_structure", "matched_structure",
    "matched_model_id", "same_function_different_roles_of", "catalogue_expressible",
    "catalogue_not_expressible", "catalogue_prior_match", "catalogue_prior_scale_differences"})
# A catalogue label that is also a word of the grammar: `power` is a call name. It is relabelled
# only where a record field or one of the engine's sentences names a structure
# (fb1_grammar.check_numerical_equivalence and fb1_grammar.admit write these three shapes).
WORD_LIKE_STRUCTURE_LABELS = frozenset({"power"})
STRUCTURE_SENTENCE_SHAPES = (" from {} (", " law {} with the same priors", "{}: force is ")


class InputNameMask:
    """X1..X4 for the grammar's declared inputs, in their declared order.

    `mask_value` is applied to a belief packet (and to the proposer block of the run record),
    `mask_text` to a validation message, `unmask_reply` to the model's reply. Identifiers
    derived from original-name expressions never reach the model: a model id is replaced by
    the id of the masked expression, a catalogue structure label by K01..K14, a registered
    free structure label by a relabel; `pairs()` is the side-file content that matches them.
    """

    def __init__(self):
        # Loaded only when masking is on.
        import fb1_grammar as fbg

        real = tuple(fbg.INPUT_NAMES)
        masked = tuple(f"X{index}" for index in range(1, len(real) + 1))
        self.input_name_map = dict(zip(masked, real))
        self._swap = {**dict(zip(real, masked)), **dict(zip(masked, real))}
        self.structure_name_map = {
            name: f"K{index:02d}" for index, name in enumerate(fbg.CATALOG_STRUCTURE_NAMES, 1)}
        self._model_ids = {}
        self._free_names = {}
        self._masked_free_names = set()
        self._masked_ids = set()

    # ---- names
    def swap_names(self, text):
        """The self-inverse rename of name segments."""
        return _MASK_TOKEN.sub(
            lambda match: "_".join(self._swap.get(part, part) for part in match.group(0).split("_")),
            text)

    def mask_expression(self, expression):
        """An expression in masked names: the catalogue product rewrites, then the name swap."""
        for old, new in MASK_TEXT_REWRITES[1:]:
            expression = expression.replace(old, new)
        return self.swap_names(expression)

    # ---- identifiers
    def note_model_id(self, original, expression, family, parameters):
        """Called by `model_id` for every identifier it computes while masking is on."""
        if original not in self._model_ids:
            roles = ",".join(f"{name}:{role}" for name, role in sorted(
                (self.swap_names(str(name)), parameters[name]) for name in parameters))
            self._model_ids[original] = hashlib.sha256(
                f"{self.mask_expression(expression)}|{roles}|{family}".encode()).hexdigest()[:20]
            self._masked_ids.add(self._model_ids[original])
        return self._model_ids[original]

    def _free_name(self, name):
        if name in self._masked_free_names:
            # Text the model read comes back in a message or a note; a relabel stays as it is.
            return name
        if name not in self._free_names:
            self._free_names[name] = "g_" + hashlib.sha256(f"masked|{name}".encode()).hexdigest()[:12]
            self._masked_free_names.add(self._free_names[name])
        return self._free_names[name]

    def _identifier(self, original):
        """An identifier `model_id` did not report (a free structure's canonical id, the id of
        a failed proposal): relabelled, paired in the side file like the others."""
        if original in self._masked_ids:
            return original
        if original not in self._model_ids:
            self._model_ids[original] = hashlib.sha256(f"masked|{original}".encode()).hexdigest()[:20]
            self._masked_ids.add(self._model_ids[original])
        return self._model_ids[original]

    def _token(self, match):
        token = match.group(0)
        if _IDENTIFIER.fullmatch(token):
            return self._identifier(token)
        if _FREE_STRUCTURE_NAME.fullmatch(token):
            return self._free_name(token)
        if token in self.structure_name_map and token not in WORD_LIKE_STRUCTURE_LABELS:
            return self.structure_name_map[token]
        return "_".join(self._swap.get(part, part) for part in token.split("_"))

    def mask_structure_name(self, name):
        """A structure label (catalogue or registered free) as the model sees it."""
        if name in self.structure_name_map:
            return self.structure_name_map[name]
        return self.mask_text(name)

    # ---- engine -> model
    def mask_text(self, text):
        """Engine text as the model sees it: rewrites, word-like structure labels, then every token masked."""
        for old, new in MASK_TEXT_REWRITES:
            text = text.replace(old, new)
        for name in WORD_LIKE_STRUCTURE_LABELS:
            for shape in STRUCTURE_SENTENCE_SHAPES:
                text = text.replace(shape.format(name), shape.format(self.structure_name_map[name]))
        return _MASK_TOKEN.sub(self._token, text)

    def mask_value(self, value, structure_names=False):
        """A JSON value with every key and string masked; numbers, booleans and null as they are.
        `structure_names` says the value sits in a field that holds structure labels."""
        one = self.mask_structure_name if structure_names else self.mask_text
        if isinstance(value, dict):
            return {one(key) if isinstance(key, str) else key:
                    self.mask_value(item, key in STRUCTURE_NAME_FIELDS)
                    for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.mask_value(item, structure_names) for item in value]
        if isinstance(value, str):
            return one(value)
        return value

    # ---- model -> engine (unified protocol: the composed reply and the revive ids)
    def unmask_composed_reply(self, text, constraints):
        """A composed reply in the engine's names: every candidate object of the array, or of the review object's
        "candidates", unmasked as `unmask_reply` does one object. Anything else passes through unchanged for the parser
        to refuse; the revive ids stay as written (the record maps them with `unmask_model_id`)."""
        try:
            payload = json.loads(text)
        except Exception:  # noqa: BLE001 - the parser reports it
            return text

        def one(item):
            """One candidate object unmasked; anything that is not an object passes through."""
            return json.loads(self.unmask_reply(json.dumps(item), constraints)) if isinstance(item, dict) else item

        if isinstance(payload, list):
            return json.dumps([one(item) for item in payload])
        if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
            return json.dumps({**payload, "candidates": [one(item) for item in payload["candidates"]]})
        return text

    def unmask_model_id(self, masked):
        """The engine's identifier behind a model id the model was shown (a revive request), or the text as written
        when no identifier shown to the model has that form."""
        return {shown: original for original, shown in self._model_ids.items()}.get(masked, masked)

    def unmask_reply(self, text, constraints):
        """The reply with `expression` and the names in `parameters` in the engine's names.

        Anything that is not a JSON object passes through unchanged and is refused by the
        parser as it always was. Every other key stays as the model wrote it."""
        try:
            payload = json.loads(text)
        except Exception:  # noqa: BLE001 - the parser reports it
            return text
        if not isinstance(payload, dict):
            return text
        expression = payload.get("expression")
        if isinstance(expression, str):
            verbatim = {self.mask_expression(item): item for item in constraints["initial_structures"]}
            payload["expression"] = verbatim.get(expression, self.swap_names(expression))
        parameters = payload.get("parameters")
        if isinstance(parameters, dict):
            payload["parameters"] = {
                self.swap_names(name) if isinstance(name, str) else name: role
                for name, role in parameters.items()}
        return json.dumps(payload)

    def mask_reply(self, text):
        """A reply written in the engine's names as a model that read X1..X4 writes it (the
        built-in mock schedules are stored in the engine's names)."""
        payload = json.loads(text)
        payload["expression"] = self.mask_expression(payload["expression"])
        payload["parameters"] = {self.swap_names(name): role
                                 for name, role in payload["parameters"].items()}
        return json.dumps(payload, sort_keys=True)

    # ---- the side file
    def pairs(self):
        """Side-file rows pairing every masked model id and structure label with the engine's original."""
        rows = [{"model_id": masked, "rng_model_id": original}
                for original, masked in sorted(self._model_ids.items())]
        rows += [{"structure": masked, "rng_structure": original}
                 for original, masked in sorted({**self.structure_name_map, **self._free_names}.items())]
        return rows


_ACTIVE_NAME_MASK = None


def set_input_name_mask(mask):
    """Install (or, with None, clear) the mask `model_id` reports to; returns the previous one."""
    global _ACTIVE_NAME_MASK
    previous, _ACTIVE_NAME_MASK = _ACTIVE_NAME_MASK, mask
    return previous


def _write_once(path, payload):
    """Write-once record.  Every string reachable from
    `payload` goes through `mask_secrets_deep` first - the request bodies, the
    response record, the ignored fields and the note are all provider text, so
    this is the single choke point that keeps a credential out of a file."""
    with path.open("x") as stream:
        json.dump(mask_secrets_deep(payload), stream, indent=2, ensure_ascii=False)
        stream.write("\n")


# API keys are read only from these environment variables (never a file - see
# Fb1CatalogProposer.__init__) and never deliberately placed in a record.
# `mask_secrets` is defense in depth for the one place an SDK's own exception
# text is stored verbatim: an auth/network error CAN echo request or header
# detail, and on a real (never-tested-here) live call that text might contain
# the live key. Every `str(exc)` captured into a written record passes
# through this first - see Fb1CatalogProposer.propose_one.
SECRET_ENV_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
COMPAT_SECRET_ENV_VARS = set()
COMPAT_RESERVED_ENV_NAMES = frozenset("""
HOME PATH PYTHONPATH PYTHONDONTWRITEBYTECODE PYTENSOR_FLAGS JAX_PLATFORMS
OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS
LLM_PROVIDER LLM_MODEL LLM_MODEL_EFFECTIVE LLM_BASE_URL LLM_BASE_URL_HOST LLM_KEY_NAME
LLM_PROVIDER_ORDER LLM_PROVIDER_ORDER_SET FB1_LLM_PROVIDER_ORDER
selected_key key_to_clear effective_proposer
NEURONBENCH_SRC DP_SCIENCEAGENT_DIR FB1_DATA_DIR FB1_WORLD FB1_ARM FB1_SEED FB1_LIST
FB1_T_EXTRA FB1_CHECKPOINTS FB1_INTEGRATOR FB1_DESIGN_GATE FB1_D0 FB1_PROPOSER
FB1_LLM_PROVIDER FB1_LLM_MODEL FB1_LLM_BASE_URL FB1_LLM_KEY_NAME FB1_ALLOW_PAID_API
FB1_STRUCTURE_PROPOSALS FB1_RUN_LABEL FB1_OUT_DIR BMD_PYTHON FB1_PYTHON FB1_DIR REPO_ROOT
BMD_BOXINGGYM_SRC BMD_AUTOSCILAB CHEMBENCH_MANIFEST PROFILE PTC_DIR ROUNDS_BOXINGGYM ROUNDS_CHEMBENCH
ROUNDS_CORE ROUNDS_NEURON
""".split())
# Earlier versions masked only `reason`; now every string of a record is masked.
# Two sources are masked - the VALUE of either environment variable above, and
# any `sk-`/`sk-ant-` prefixed token even when it is not the current
# environment's value (a key echoed back from a header or another process is
# masked just the same). The replacement is ALWAYS exactly one of the two
# placeholders below and never reveals the length, the prefix or any substring
# of what it replaced.
SECRET_PATTERNS = (
    ("ANTHROPIC_API_KEY", re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")),
    ("OPENAI_API_KEY", re.compile(r"sk-(?!ant-)[A-Za-z0-9_\-]{8,}")),
)
MASK_PLACEHOLDERS = tuple(f"<{name}:masked>" for name in SECRET_ENV_VARS)
# Library loggers whose records can carry credential text. `openai` and
# `anthropic` log through CHILD loggers (`openai._base_client`), and a
# `logging.Filter` attached to a logger is NOT applied to records made by its
# children (`logging.Logger.handle` runs only that logger's filters, and
# `callHandlers` walks ancestors for HANDLERS, not filters) - so the filter is
# attached to every logger that already exists under these roots AND a
# LogRecord factory covers the ones the SDK creates later.
MASKED_LOGGER_ROOTS = ("openai", "anthropic", "httpx", "httpcore")


def mask_secrets(text):
    """Replace every credential occurrence in `text` with its `<NAME:masked>`
    placeholder. A secret env var value shorter than 8 characters is never
    treated as a secret - that guards an unset/empty/trivial env var from
    turning this into a no-op-breaking blanket replacement. Anything that is
    not a string is returned unchanged."""
    if not isinstance(text, str):
        return text
    for name in SECRET_ENV_VARS:
        value = os.environ.get(name)
        if value and len(value) >= 8 and value in text:
            text = text.replace(value, f"<{name}:masked>")
    for name in sorted(COMPAT_SECRET_ENV_VARS):
        value = os.environ.get(name)
        if value and value in text:
            text = text.replace(value, f"<{name}:masked>")
    for name, pattern in SECRET_PATTERNS:
        text = pattern.sub(f"<{name}:masked>", text)
    return text


def mask_secrets_deep(value):
    """`mask_secrets` applied to every string reachable from `value`, keeping
    dict/list/tuple/set structure. Dict KEYS are masked too - a provider can
    echo a header the other way round. Anything that is neither a string nor
    one of those containers is returned unchanged (numbers, None, arrays)."""
    if isinstance(value, str):
        return mask_secrets(value)
    if isinstance(value, bytes):
        for name in sorted(COMPAT_SECRET_ENV_VARS):
            secret = os.environ.get(name)
            if secret:
                value = value.replace(secret.encode(), f"<{name}:masked>".encode())
        return value
    if isinstance(value, dict):
        return {mask_secrets_deep(key): mask_secrets_deep(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_secrets_deep(item) for item in value]
    if isinstance(value, tuple):
        return tuple(mask_secrets_deep(item) for item in value)
    if isinstance(value, set):
        return {mask_secrets_deep(item) for item in value}
    return value


def carries_secret(text):
    """True when `mask_secrets` would change `text`. This is the test that
    keeps the masked stand-in below OFF every path with nothing to hide."""
    return isinstance(text, str) and mask_secrets(text) != text


class MaskedCredentialError(RuntimeError):
    """Stand-in for an exception whose own text carried a credential.

    Raised INSTEAD of the original only at the API transport boundary and only
    when the original really did carry one, so the numerical rule - an expected
    numerical failure propagates verbatim, original type and traceback - is
    untouched.
    """


def masked_exception(exc):
    """`exc` itself unless its str, repr or formatted traceback would leak a
    credential; then a `MaskedCredentialError` carrying the masked text and the
    original type name. Raise the result with `from None` so the original is
    not printed as the exception context."""
    parts = [str(exc), repr(exc)]
    try:
        parts.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:  # noqa: BLE001 - a formatting failure must not lose the mask
        parts.append("")
    if not any(carries_secret(part) for part in parts):
        return exc
    return MaskedCredentialError(f"{type(exc).__name__}: {mask_secrets(str(exc))}")


class SecretMaskingFilter(logging.Filter):
    """Masks credential text in a LogRecord's message, args and traceback.

    `exc_text` is pre-filled with the MASKED traceback because
    `logging.Formatter.format` uses `record.exc_text` verbatim when it is
    already set and only formats `exc_info` when it is not - which is how the
    SDK's own `exc_info=True` DEBUG log (openai/_base_client.py) is stopped
    from printing an exception message that echoes a key.
    """

    def filter(self, record):
        """Mask the record in place; always returns True so the record is kept."""
        if isinstance(record.msg, str):
            record.msg = mask_secrets(record.msg)
        elif COMPAT_SECRET_ENV_VARS:
            record.msg = mask_secrets_deep(record.msg)
        if record.args:
            record.args = mask_secrets_deep(record.args)
        if getattr(record, "exc_text", None):
            record.exc_text = mask_secrets(record.exc_text)
        elif record.exc_info:
            try:
                record.exc_text = mask_secrets(
                    "".join(traceback.format_exception(*record.exc_info))
                )
            except Exception:  # noqa: BLE001 - never break logging over masking
                record.exc_text = "<traceback withheld: masking failed>"
        return True


_SECRET_LOG_FILTER = SecretMaskingFilter()


def install_secret_log_filter():
    """Attach `SecretMaskingFilter` to the SDK/HTTP loggers, and install a
    LogRecord factory so loggers those libraries create LATER are covered too
    (they are not imported yet when this module is). Idempotent, and called
    once at import so no caller has to remember it."""
    for name in list(logging.Logger.manager.loggerDict) + list(MASKED_LOGGER_ROOTS):
        if name.split(".", 1)[0] not in MASKED_LOGGER_ROOTS:
            continue
        logger = logging.getLogger(name)
        if _SECRET_LOG_FILTER not in logger.filters:
            logger.addFilter(_SECRET_LOG_FILTER)
    factory = logging.getLogRecordFactory()
    if getattr(factory, "masks_secrets", False):
        return factory

    def masking_factory(*args, **kwargs):
        """Build a LogRecord with the previous factory and mask it when an SDK/HTTP logger made it."""
        record = factory(*args, **kwargs)
        if record.name.split(".", 1)[0] in MASKED_LOGGER_ROOTS:
            _SECRET_LOG_FILTER.filter(record)
        return record

    masking_factory.masks_secrets = True
    logging.setLogRecordFactory(masking_factory)
    return masking_factory


install_secret_log_filter()


def _numeric(value):
    """True for a real token count (a bool is not one)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _openai_usage(usage_obj):
    """Normalize the OpenAI SDK usage shape into ({input_tokens,
    output_tokens, cached_tokens}, known). `usage_obj` is `response.usage`:
    `.prompt_tokens` and `.completion_tokens` are normally present;
    `.prompt_tokens_details.cached_tokens` is only present for some
    models/requests, so both the details object and the attribute are optional
    (0 when absent) - matches both the real openai>=1.40 SDK response and the
    injected fakes in test_fb1_proposer.py.

    A MISSING usage object
    (`response.usage is None`) or one whose token counts are not numbers is NOT
    an error and NOT a reason to re-call. It returns (None, False) so the
    record can say "unknown" and the reply can still be parsed - the pre-patch
    `usage_obj.prompt_tokens` AttributeError fell into the broad except and,
    because `raw_text` already existed, spent a second paid call on a reply
    that was perfectly valid.
    """
    if usage_obj is None:
        return None, False
    try:
        details = getattr(usage_obj, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details is not None else 0
        usage = {
            "input_tokens": usage_obj.prompt_tokens,
            "output_tokens": usage_obj.completion_tokens,
            "cached_tokens": cached or 0,
        }
        # Reasoning tokens are part of `completion_tokens` and of the
        # reply budget, so a reply can finish on length with an empty body.
        # The key exists ONLY when the provider reported the count.
        completion_details = getattr(usage_obj, "completion_tokens_details", None)
        reasoning = getattr(completion_details, "reasoning_tokens", None)
        if _numeric(reasoning):
            usage["reasoning_tokens"] = reasoning
    except Exception:  # noqa: BLE001 - an SDK object contract change is not a format failure
        return None, False
    if not _numeric(usage["input_tokens"]) or not _numeric(usage["output_tokens"]):
        return None, False
    return usage, True


def _anthropic_usage(usage_obj):
    """({provider dump}, known) for the Anthropic usage object - the same
    contract as `_openai_usage`: a missing usage object, a dump that fails, or
    a dump with no numeric field at all is recorded as unknown instead of
    raising into the repair path."""
    if usage_obj is None:
        return None, False
    try:
        usage = usage_obj.model_dump(mode="json")
    except Exception:  # noqa: BLE001 - an SDK object contract change is not a format failure
        return None, False
    if not isinstance(usage, dict) or not any(_numeric(value) for value in usage.values()):
        return None, False
    return usage, True


def _http_request_id(response):
    """The HTTP request id (`x-request-id`) of the call when the SDK exposes
    it, else None.

    NOT the completion id: OpenAI's `response.id` is `chatcmpl_...` (review
    section 5), which is why the record now names them apart. openai 1.109 -
    the version the fb1 venv runs - does attach `_request_id` to the parsed
    model; older builds (1.40 in the CORE/BG venv) do not, and a raw-response
    wrapper carries the header itself. Never raises: an id is not worth
    failing a call for.
    """
    try:
        value = getattr(response, "_request_id", None)
        if isinstance(value, str) and value:
            return value
        for attribute in ("_raw_response", "http_response", "raw_response"):
            headers = getattr(getattr(response, attribute, None), "headers", None)
            if headers is None:
                continue
            value = headers.get("x-request-id")
            if isinstance(value, str) and value:
                return value
    except Exception:  # noqa: BLE001 - an id is never worth failing a call for
        return None
    return None


def _usage_total(responses):
    """Sum `responses[*]["usage"]` exactly ONCE each.

    Returns (total, complete, missing): `missing` holds the indices into
    `responses` whose usage is unknown. An unknown usage is NEVER counted as
    zero - it makes `complete` False and shows up in `missing`, so a cost
    ledger reserves a conservative budget instead of under-counting. The
    top-level `last_call_usage` is the LAST response's usage alone and must
    never be added to this total.
    """
    total, missing = {}, []
    for index, item in enumerate(responses):
        usage = item.get("usage")
        if not item.get("usage_known", False) or not isinstance(usage, dict):
            missing.append(index)
            continue
        for key, value in usage.items():
            if _numeric(value):
                total[key] = total.get(key, 0) + value
    return total, not missing, missing


def _validate_repair_attempts(repair_attempts):
    """`repair_attempts` is 0 or 1 - nothing else.

    "at most one format repair" is an invariant of the policy, not a tunable:
    the pre-patch `range(1 + max(0, int(repair_attempts)))` silently accepted 3
    and made four calls.
    """
    if isinstance(repair_attempts, bool) or repair_attempts not in (0, 1):
        raise ValueError("repair_attempts must be 0 or 1 (at most one format repair)")
    return int(repair_attempts)


def validate_compat_options(base_url, key_name, model):
    """Validate explicit compatibility settings without exposing credentials."""
    if not isinstance(key_name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_name):
        raise ValueError("Compatibility key name must be an environment variable name")
    if key_name in COMPAT_RESERVED_ENV_NAMES:
        raise ValueError("Compatibility key name must not name a runtime setting")
    COMPAT_SECRET_ENV_VARS.add(key_name)
    if (not isinstance(model, str) or not model.strip() or model != model.strip()
            or any(ord(c) < 32 or ord(c) == 127 for c in model) or "<" in model or ">" in model
            or model.casefold() in {"name", "model", "your_model", "your-model", "todo", "tbd", "placeholder"}):
        raise ValueError("Compatibility provider requires an explicit model ID")
    if not isinstance(base_url, str) or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in base_url):
        raise ValueError("Compatibility base URL must be explicit HTTPS without credentials")
    try:
        parsed = urlsplit(base_url)
        valid = (parsed.scheme == "https" and parsed.hostname and parsed.username is None
                 and parsed.password is None and not parsed.query and not parsed.fragment
                 and "?" not in base_url and "#" not in base_url)
        parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Compatibility base URL must be explicit HTTPS without credentials, query or fragment")
    return parsed.hostname


def normalize_provider_order(value):
    """Normalize explicitly requested routing without changing an omitted body."""
    if value is None:
        return None
    parts = value.split(",") if isinstance(value, str) else value
    if not isinstance(parts, (list, tuple)) or not parts:
        raise ValueError("Compatibility provider order requires a nonempty list of names")
    names = []
    for item in parts:
        if (not isinstance(item, str) or not item.strip() or "," in item
                or any(ord(char) < 32 or 127 <= ord(char) <= 159 or char in "\u2028\u2029" for char in item)):
            raise ValueError("Compatibility provider order requires names without control characters")
        names.append(item.strip())
    if len(set(names)) != len(names):
        raise ValueError("Compatibility provider order must not repeat names")
    return names


# The OpenAI-compatible `reasoning` extension. The snapshot is a
# standalone file, so this is an independent implementation of the SAME
# grammar as count_proposer.parse_reasoning_spec; a test pins the two to equal
# results for every case.
REASONING_EFFORTS = ("max", "xhigh", "high", "medium", "low", "minimal", "none")
# The Anthropic fields of the same grammar (as in count_proposer's reasoning transport).
ANTHROPIC_EFFORTS = ("max", "xhigh", "high", "medium", "low")
THINKING_MODES = ("adaptive",)
THINKING_DISPLAYS = ("summarized",)
REASONING_FIELD_ERROR = (f"llm-reasoning takes effort=<{'|'.join(REASONING_EFFORTS)}>, "
                         "max_tokens=<positive integer> and the optional flag exclude"
                         " (and, for provider anthropic, thinking=adaptive with optional display=summarized)")
MAX_TOKENS_LIMIT = 32768


def parse_reasoning_spec(text):
    """`effort=<level>` XOR `max_tokens=<n>`, plus optional `exclude`, plus (reasoning transport) the
    optional `thinking=adaptive` with optional `display=summarized`.

    Comma separated, order free, no repeats; the returned dict is canonically
    ordered so the same request is the same bytes however it was typed. Which
    fields a transport can send is `validate_transport_reasoning`'s question.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(REASONING_FIELD_ERROR)
    fields = {}
    for field in text.split(","):
        key, separator, value = field.strip().partition("=")
        key, value = key.strip(), value.strip()
        if key in fields:
            raise ValueError("llm-reasoning must not repeat a field")
        if key == "exclude" and not separator:
            fields[key] = True
        elif key == "effort" and separator:
            if value not in REASONING_EFFORTS:
                raise ValueError("llm-reasoning effort must be one of " + "|".join(REASONING_EFFORTS))
            fields[key] = value
        elif key == "max_tokens" and separator:
            if re.fullmatch(r"[0-9]+", value) is None or int(value) <= 0:
                raise ValueError("llm-reasoning max_tokens must be a positive integer")
            fields[key] = int(value)
        elif key == "thinking" and separator:
            if value not in THINKING_MODES:
                raise ValueError("llm-reasoning thinking must be " + "|".join(THINKING_MODES))
            fields[key] = value
        elif key == "display" and separator:
            if value not in THINKING_DISPLAYS:
                raise ValueError("llm-reasoning display must be " + "|".join(THINKING_DISPLAYS))
            fields[key] = value
        else:
            raise ValueError(REASONING_FIELD_ERROR)
    if ("effort" in fields) == ("max_tokens" in fields):
        raise ValueError("llm-reasoning requires exactly one of effort= and max_tokens=")
    if "thinking" in fields and "effort" not in fields:
        raise ValueError("llm-reasoning thinking=adaptive rides with an effort level")
    if "display" in fields and "thinking" not in fields:
        raise ValueError("llm-reasoning display=summarized requires thinking=adaptive")
    spec = {"effort": fields["effort"]} if "effort" in fields else {"max_tokens": fields["max_tokens"]}
    if fields.get("exclude"):
        spec["exclude"] = True
    if "thinking" in fields:
        spec["thinking"] = fields["thinking"]
    if "display" in fields:
        spec["display"] = fields["display"]
    return spec


def normalize_reasoning(value):
    """A stated spec (text or an already parsed dict) as one canonical dict;
    None leaves the request body and the record exactly as they were."""
    if value is None:
        return None
    if isinstance(value, str):
        return parse_reasoning_spec(value)
    if not isinstance(value, dict) or not value:
        raise ValueError(REASONING_FIELD_ERROR)
    fields = []
    for key, item in value.items():
        if key == "exclude":
            if item is not True:
                raise ValueError(REASONING_FIELD_ERROR)
            fields.append("exclude")
        else:
            fields.append(f"{key}={item}")
    return parse_reasoning_spec(",".join(fields))


def parse_max_tokens(value):
    """An explicitly stated reply budget, 1..32768 (text or int). Unstated
    keeps the MAX_TOKENS literal every request has always carried."""
    if isinstance(value, str):
        if re.fullmatch(r"[0-9]+", value.strip()) is None:
            raise ValueError(f"llm-max-tokens must be an integer in 1..{MAX_TOKENS_LIMIT}")
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_TOKENS_LIMIT:
        raise ValueError(f"llm-max-tokens must be an integer in 1..{MAX_TOKENS_LIMIT}")
    return value


# The Anthropic reasoning body, the SDK call split, the thinking record and the
# per-cell paid-call guard. The snapshot is a standalone file, so these are independent copies
# of count_proposer's validate_provider_reasoning (its anthropic and openai_compat rules),
# sdk_call_kwargs, create_anthropic_message, anthropic_thinking_record and its paid-call guard; tests pin
# them to the package's results.
def validate_transport_reasoning(transport, spec):
    """The stated spec the named transport can send, or a ValueError that says why not.
    anthropic: `effort=<low|medium|high|xhigh|max>` (sent as `output_config.effort`) plus the
    optional `thinking=adaptive[,display=summarized]`; openai_compat: `effort=` or `max_tokens=`
    plus the optional `exclude`, as before. None (not stated) passes through."""
    if spec is None:
        return None
    if transport == "anthropic":
        if "max_tokens" in spec or "exclude" in spec:
            raise ValueError("provider anthropic takes llm-reasoning effort=<level>[,thinking=adaptive]: "
                             "max_tokens= and exclude are openai_compat (OpenRouter) extensions")
        if spec["effort"] not in ANTHROPIC_EFFORTS:
            raise ValueError("provider anthropic takes effort=<" + "|".join(ANTHROPIC_EFFORTS)
                             + ">; none and minimal are not Anthropic effort levels")
    elif transport == "openai_compat":
        if "display" in spec:
            raise ValueError("display= is an Anthropic field; provider openai_compat does not accept it")
        if "thinking" in spec:
            raise ValueError("thinking= is an Anthropic field; provider openai_compat takes "
                             "effort=<level> or max_tokens=<n>, plus the optional exclude")
    else:
        raise ValueError("unsupported provider for llm-reasoning")
    return spec


def sdk_call_kwargs(create, body):
    """The body as the installed SDK's `create` can take it - by name where it has the
    parameter, under `extra_body` where it does not (anthropic 0.71.0 has no `output_config`
    parameter, 0.120.2 has it; both carry the key on the wire). A `create` that takes
    `**kwargs` (an injected fake client) and a body whose keys are all parameters (every
    default-path request) get the body exactly as before. The recorded body is never touched."""
    try:
        parameters = inspect.signature(create).parameters
    except (TypeError, ValueError):
        return dict(body)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(body)
    unknown = {key: value for key, value in body.items() if key not in parameters}
    if not unknown:
        return dict(body)
    if "extra_body" not in parameters:
        raise ValueError(f"the installed SDK's create takes neither {sorted(unknown)} nor extra_body")
    kwargs = {key: value for key, value in body.items() if key in parameters}
    extra = dict(kwargs.get("extra_body") or {})
    clash = sorted(set(extra) & set(unknown))
    if clash:
        raise ValueError(f"request keys {clash} are stated both in the body and in extra_body")
    kwargs["extra_body"] = {**extra, **unknown}
    return kwargs


def create_anthropic_message(client, body):
    """The Messages API call with the signature-decided split above."""
    return client.messages.create(**sdk_call_kwargs(client.messages.create, body))


def _thinking_block_count(response):
    """How many thinking blocks (summarized or redacted) an Anthropic reply carried."""
    content = getattr(response, "content", None) or []
    return sum(1 for block in content if getattr(block, "type", None) in ("thinking", "redacted_thinking"))


def anthropic_thinking_record(response):
    """What an Anthropic reply's thinking blocks said, for the record only: `thinking_text`
    (each summary through `mask_secrets`, in reply order) and, when there is one,
    `redacted_thinking_blocks`. A reply with no such block gives an empty mapping, so the
    record of a run without thinking keeps every byte. Nothing here is parsed or re-sent."""
    content = getattr(response, "content", None) or []
    texts = [mask_secrets(str(getattr(block, "thinking", ""))) for block in content
             if getattr(block, "type", None) == "thinking"]
    redacted = sum(1 for block in content if getattr(block, "type", None) == "redacted_thinking")
    if not texts and not redacted:
        return {}
    return {"thinking_text": texts, **({"redacted_thinking_blocks": redacted} if redacted else {})}


#: The per-cell paid-call guard (as in count_proposer). A proposer given a cap counts every
#: request to a non-mock transport (repair turns included) and refuses the one that would make
#: it cap + 1 before the client is called. It then writes `stop_summary.json` in its record
#: folder with `stop_reason` "budget_exhausted" and the run ends with this status.
PAID_CALL_BUDGET_EXIT_CODE = 4
BUDGET_EXHAUSTED = "budget_exhausted"
MAX_PAID_CALLS_ERROR = "max-paid-calls must be an integer of at least 1"
STOP_SUMMARY_FILE = "stop_summary.json"


class PaidCallBudgetExhausted(SystemExit):
    """The next paid request would exceed this cell's `max_paid_calls`.

    Not an `Exception`, so no `except Exception` boundary can read it as something the cell
    survives. It is a `SystemExit` carrying the guard's status because run_gravity.py is not
    edited for this (its own sha256 is a published contract column, `code_sha`): uncaught, it
    ends the run with status 4 and no traceback. `context` is the proposal (event, round,
    candidate index) the refused request belonged to."""

    def __init__(self, attempts, max_paid_calls, context=None):
        super().__init__(f"paid-call budget exhausted: {attempts} of {max_paid_calls} paid "
                         "calls made; the next call would exceed max_paid_calls")
        self.code = PAID_CALL_BUDGET_EXIT_CODE
        self.attempts = attempts
        self.max_paid_calls = max_paid_calls
        self.context = dict(context or {})


def parse_max_paid_calls(value):
    """A stated per-cell paid-call cap: an integer of at least 1 (text or int)."""
    if isinstance(value, str):
        if re.fullmatch(r"[0-9]+", value.strip()) is None:
            raise ValueError(MAX_PAID_CALLS_ERROR)
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(MAX_PAID_CALLS_ERROR)
    return value


def _compat_completion(client, body):
    """Keep the HTTP request ID on SDK versions that omit it from parsed objects."""
    raw_api = getattr(client, "with_raw_response", None)
    if raw_api is not None:
        raw = raw_api.chat.completions.create(**body)
        response = raw.parse()
        request_id = raw.headers.get("x-request-id")
        return response, request_id if isinstance(request_id, str) and request_id else _http_request_id(response)
    response = client.chat.completions.create(**body)
    return response, _http_request_id(response)


def _require_explicit_openai_model(transport, model):
    """Refuse the OpenAI transport with a model that was
    never named, BEFORE any output directory exists.

    `FB1_LLM_PROVIDER=openai` alone leaves `LLM_MODEL` at its Anthropic default
    (`claude-opus-4-7`, run_gravity.py line 121, out of scope here), so
    a paid smoke could ask OpenAI for a Claude model. The rule is enforced here
    - in the proposer's own constructor - rather than in the runner, and it
    applies to an injected client whose provider is openai too.
    """
    if transport != "openai":
        return
    if not model or str(model).startswith("claude-"):
        raise ValueError("FB1_LLM_MODEL must be set explicitly for provider openai")


class Fb1CatalogProposer:
    """One proposal = one LLM call (+ at most one format-repair re-call).  Records use the engine's proposer API format as is.

    When no `client` is given and `allow_live` is not set, the **mock transport** is used (default).  The anthropic
    client is created only when `allow_live=True` - the runner calls it that way only when the environment variable
    `FB1_ALLOW_PAID_API=1` is set.  `paid_api_calls` is the number of real paid calls.
    """

    # How structures are proposed.  "catalog" (default) is the catalogue-mode behaviour as is - without this argument the
    # mock transport, records and labels do not change by a single character.  "free" is the free-expression grammar mode.
    STRUCTURE_MODES = {"catalog": "catalog_selection", "free": "freeform_discovery"}

    def __init__(self, records, *, allow_live=False, model=_MODEL_UNSET, provider=None,
                 client=None, structure_mode="catalog", base_url=None, key_name=None,
                 provider_order=None, reasoning=None, max_tokens=None, max_paid_calls=None,
                 name_mask=None):
        explicit_model = model is not _MODEL_UNSET
        # Input-name masking: None is no masking, every byte as before.
        self.name_mask = name_mask
        model = model if explicit_model else DEFAULT_MODEL
        self.records = Path(records)
        self.model = model
        self.attempts = 0
        self.paid_api_calls = 0
        self.injected_client = client is not None
        self.allow_live = bool(allow_live)
        if structure_mode not in self.STRUCTURE_MODES:
            raise ValueError(
                f"structure_mode must be one of {sorted(self.STRUCTURE_MODES)}, "
                f"got {structure_mode!r}"
            )
        self.structure_mode = structure_mode
        self.recovery_mode = self.STRUCTURE_MODES[structure_mode]
        requested_provider = provider or os.environ.get("FB1_LLM_PROVIDER", "anthropic")
        compatibility = ((client is None and allow_live and requested_provider == "openai_compat")
                         or (client is not None and (provider == "openai_compat"
                             or getattr(client, "provider", None) == "openai_compat")))
        # The Anthropic transport by name - the live one, or an injected client that
        # declares it. It takes the reasoning control too, as `output_config` and `thinking`.
        anthropic_named = ((client is None and allow_live and requested_provider == "anthropic")
                           or (client is not None and not compatibility
                               and (provider == "anthropic" or getattr(client, "provider", None) == "anthropic")))
        self.compat_metadata = {}
        self.provider_order = normalize_provider_order(provider_order)
        # Reasoning control is a compatibility-endpoint extension. Read
        # from FB1_LLM_REASONING when the caller does not pass it, exactly like
        # the base URL and key name above - run_gravity.py is unchanged because
        # its own sha256 is a published contract column (code_sha).
        if reasoning is None and (compatibility or anthropic_named):
            reasoning = os.environ.get("FB1_LLM_REASONING") or None
        self.reasoning = normalize_reasoning(reasoning)
        if self.provider_order is not None and not compatibility:
            raise ValueError("Provider order requires the openai_compat transport")
        if self.reasoning is not None and not (compatibility or anthropic_named):
            raise ValueError("Reasoning control requires the openai_compat transport")
        self.reasoning = validate_transport_reasoning(
            "openai_compat" if compatibility else "anthropic", self.reasoning)
        if compatibility:
            if not explicit_model:
                raise ValueError("Compatibility provider requires an explicit model ID")
            base_url = base_url if base_url is not None else os.environ.get("FB1_LLM_BASE_URL")
            key_name = key_name if key_name is not None else os.environ.get("FB1_LLM_KEY_NAME", "OPENAI_COMPAT_API_KEY")
            host = validate_compat_options(base_url, key_name, model)
            self.compat_metadata = {"llm_base_url_host": host, "llm_key_name": key_name}
            if self.provider_order is not None:
                self.compat_metadata["provider_order"] = list(self.provider_order)
        if client is None:
            if allow_live:
                # FB1_LLM_PROVIDER selects the live
                # transport (default "anthropic" keeps the original behaviour
                # when unset); `provider=` overrides it for direct/test use.
                # Only resolved/validated here - the mock and injected-client
                # branches below never touch it, so a stray env value can
                # never affect them.
                resolved_provider = provider or os.environ.get("FB1_LLM_PROVIDER", "anthropic")
                if resolved_provider not in ("anthropic", "openai", "openai_compat"):
                    raise ValueError(
                        f"unsupported FB1_LLM_PROVIDER: {resolved_provider!r} (anthropic|openai|openai_compat)"
                    )
                _require_explicit_openai_model(resolved_provider, model)
                # The SDK constructor call is INSIDE a recording try
                # so an auth/proxy/header error that echoes the credential is
                # re-raised with its str, repr and traceback masked, instead of
                # propagating verbatim out of __init__.
                try:
                    if resolved_provider in ("openai", "openai_compat"):
                        import openai  # noqa: PLC0415 - imported only on the paid path

                        options = {"max_retries": 0, "timeout": 120.0}
                        if compatibility:
                            secret = os.environ.get(key_name)
                            if not secret:
                                raise ValueError("Compatibility provider credential is not configured")
                            options.update(base_url=base_url, api_key=secret)
                        client = openai.OpenAI(**options)
                        self.transport = resolved_provider
                    else:
                        import anthropic  # noqa: PLC0415 - imported only on the paid path

                        client = anthropic.Anthropic(max_retries=0, timeout=120.0)
                        self.transport = "anthropic"
                except BaseException as exc:  # noqa: BLE001 - masked, then re-raised
                    raise masked_exception(exc) from None
            else:
                # The free mode's pre-declared schedule is a different one from the catalogue schedule.
                client = MockFreeClient() if structure_mode == "free" else MockCatalogClient()
                self.transport = "mock"
                if name_mask is not None:
                    # Input-name masking: the mock answers in the names it is shown, so a masked mock
                    # run proposes the same structures as an unmasked one.
                    client.texts = [name_mask.mask_reply(item) for item in client.texts]
        else:
            self.transport = "openai_compat" if compatibility else getattr(client, "provider", "injected")
            _require_explicit_openai_model(self.transport, model)
        # The reply budget (FB1_LLM_MAX_TOKENS) is a decoding setting
        # every paid transport takes. A mock transport makes no request at all
        # and keeps MAX_TOKENS, so a stray environment value cannot move a
        # mock run's bytes.
        if max_tokens is None and self.transport != "mock":
            max_tokens = os.environ.get("FB1_LLM_MAX_TOKENS") or None
        self.max_tokens = MAX_TOKENS if max_tokens is None else parse_max_tokens(max_tokens)
        # The per-cell paid-call guard (FB1_MAX_PAID_CALLS), under the same rule as
        # the reply budget: a mock transport makes no request and never reads the environment.
        if max_paid_calls is None and self.transport != "mock":
            max_paid_calls = os.environ.get("FB1_MAX_PAID_CALLS") or None
        self.max_paid_calls = None if max_paid_calls is None else parse_max_paid_calls(max_paid_calls)
        self.client = client
        # Unified protocol: the round-0 burst calls propose_one from several threads; the paid-call count and its guard are
        # read and moved together under this lock. The unified mock client exists only once a new request kind is made.
        self._attempt_lock = threading.Lock()
        self._unified_mock = None
        self.records.mkdir(parents=True, exist_ok=False)

    def _decoding_identity(self):
        """What the request declares. An option that was never stated leaves
        no key behind, so an unstated run records the bytes it always did."""
        decoding = {"temperature": "not_sent/provider_default", "max_tokens": self.max_tokens}
        if self.reasoning is not None:
            decoding["reasoning"] = dict(self.reasoning)
            if self.transport != "openai_compat":
                # What the Anthropic request carried, as count_proposer records it.
                decoding["token_parameter"] = "max_tokens"
                decoding["output_config"] = {"effort": self.reasoning["effort"]}
                if "thinking" in self.reasoning:
                    decoding["thinking"] = {"type": self.reasoning["thinking"]}
                    if "display" in self.reasoning:
                        decoding["thinking"]["display"] = self.reasoning["display"]
        return decoding

    def _apply_extra_body(self, body):
        """Attach the stated compatibility extensions - routing order and
        reasoning share the one `extra_body` object - and nothing else."""
        extra = {}
        if self.provider_order is not None:
            extra["provider"] = {"order": list(self.provider_order), "allow_fallbacks": False}
        if self.reasoning is not None:
            extra["reasoning"] = dict(self.reasoning)
        if extra:
            body["extra_body"] = extra
        return body

    def _budget_stop(self, stop):
        """Say why and where the cell stopped, then end it. Every record written so
        far stays; `stop_summary.json` sits beside them in this proposer's record folder."""
        context = stop.context
        with self._attempt_lock:
            # Unified protocol: parallel burst threads refused at the same cap write one summary - the first one's.
            if not (self.records / STOP_SUMMARY_FILE).exists():
                self._write_stop_summary(stop, context)
        print(f"STOP {stop}; {STOP_SUMMARY_FILE} records stop_reason {BUDGET_EXHAUSTED}", file=sys.stderr)
        raise stop

    def _write_stop_summary(self, stop, context):
        _write_once(self.records / STOP_SUMMARY_FILE, mask_secrets_deep({
            "status": BUDGET_EXHAUSTED, "stop_reason": BUDGET_EXHAUSTED,
            "paid_api_calls": self.attempts, "max_paid_calls": self.max_paid_calls,
            "provider_paid_api_calls": self.paid_api_calls,
            "stopped_round": context.get("round"), "stopped_event": context.get("event"),
            "stopped_n_designs": context.get("n_designs"),
            "stopped_candidate_index": context.get("candidate_index"),
            "run_id": context.get("run_id"), "world": context.get("world"), "arm": context.get("arm"),
            "policy": context.get("policy"), "transport": self.transport, "requested_model": self.model,
            "billing_status": "usage_in_proposer_records", "registered_run": False,
        }))

    def propose_one(self, packet_text, *, context, constraints, repair_attempts=1, request=None, proposal_limit=None):
        """Single-candidate proposal shared by policies P1 and P4.  Returns (candidates, record).

        Unified protocol: `request` None (the default) is every call made before the protocol was added, byte for byte. "burst" and
        "burst_refill" are round-0 one-candidate calls whose context also names `burst_wave` and `burst_call_index` (their
        records are `round_00__burst_w<wave>_c<index>`). "composed" asks for one JSON array of `proposal_limit` (3)
        candidates and returns the accepted ones in reply order (`parse_candidate_array`); "composed_review" asks for the
        object {"candidates": [...], "revive": [...]} (`parse_review_reply`), the record keeping the revive list as written
        (`revive_requested`) and, on a masked run, in the engine's identifiers (`revive_requested_engine_ids`). Each sends
        the system text of `request_prompt` and records its identity block; a mock transport answers from
        `MockUnifiedClient`.
        """
        request = validate_request(request, proposal_limit)
        keys = {"run_id", "world", "arm", "policy", "event", "round", "n_designs",
                "candidate_index", "packet_file"}
        if request in BURST_REQUEST_KINDS:
            keys = keys | BURST_CONTEXT_KEYS
        if set(context) != keys:
            raise ValueError(
                "explicit run/world/arm/policy/event/round/n_designs/candidate_index/packet_file "
                "identity required"
            )
        if not isinstance(packet_text, str) or not packet_text:
            raise ValueError("belief packet text required")
        if not constraint_keys_ok(constraints):
            raise ValueError("catalog-selection constraints required")
        if is_free_constraints(constraints) != (self.structure_mode == "free"):
            # A mode that disagrees with the constraints block would silently apply a different judgment - stop at once.
            raise ValueError(
                f"structure_mode={self.structure_mode!r} does not match the constraints block "
                f"({'free' if is_free_constraints(constraints) else 'catalog'})"
            )
        repair_attempts = _validate_repair_attempts(repair_attempts)
        system, prompt_block = request_prompt(request, proposal_limit)
        messages = [{"role": "user", "content": packet_text}]
        identity = {
            **context,
            "provider": "anthropic" if self.transport == "anthropic" else self.transport,
            "requested_model": self.model,
            "recovery_mode": self.recovery_mode,
            "decoding": self._decoding_identity(),
            **prompt_block,
            "packet_bytes": len(packet_text.encode()),
            "packet_sha256": hashlib.sha256(packet_text.encode()).hexdigest(),
            **self.compat_metadata,
        }
        if request in BURST_REQUEST_KINDS:
            # Unified protocol: one record pair per burst call, named by its wave and index.
            prefix = self.records / (
                f"round_{context['round']:02d}__burst_w{context['burst_wave']}"
                f"_c{context['burst_call_index']:02d}"
            )
        else:
            # P4 makes 3 calls in one event, so the candidate index must be in the file name to avoid collisions.
            prefix = self.records / (
                f"event_{context['event']:02d}_round_{context['round']:02d}"
                f"_c{context['candidate_index']}"
            )
        client = self.client
        if request is not None and self.transport == "mock":
            if self.structure_mode != "free":
                raise ValueError("the unified protocol's mock transport answers free-form requests only")
            if self._unified_mock is None:
                self._unified_mock = MockUnifiedClient(name_mask=self.name_mask)
            client = self._unified_mock
        started = time.perf_counter()
        record = {
            **identity,
            "status": "not_returned",
            "returned_model": None,
            "api_request_attempted": True,
            "client_injected": self.injected_client,
            "transport": self.transport,
            "allow_live": self.allow_live,
            "sdk_retries": 0,
            "temperature_sent": False,
            "calls": 0,
            "repairs": 0,
            "paid_api_calls": 0,
            "rejected_candidates": [],
            "responses": [],
        }
        if self.transport == "openai_compat":
            record["returned_provider"] = None
        bodies, candidates = [], []
        budget_stop = None
        for attempt in range(1 + repair_attempts):
            if self.transport in ("openai", "openai_compat"):
                # SYSTEM travels as the first `messages`
                # entry (role "system") - the OpenAI Chat Completions shape -
                # instead of Anthropic's separate `system` field below.
                body = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "system", "content": system}, *messages],
                }
            else:
                body = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "system": system,
                    "messages": list(messages),
                }
                if self.reasoning is not None:
                    # Appended, so an unstated spec leaves the body byte for byte.
                    body["output_config"] = {"effort": self.reasoning["effort"]}
                    if "thinking" in self.reasoning:
                        body["thinking"] = {"type": self.reasoning["thinking"]}
                        if "display" in self.reasoning:
                            body["thinking"]["display"] = self.reasoning["display"]
            if self.transport == "openai_compat":
                self._apply_extra_body(body)
            # Counted, or refused before the client is called. A refused first call
            # writes nothing; a refused repair turn keeps the paid call's record below.
            # Unified protocol: the check and both counts move together under the lock (parallel burst calls).
            with self._attempt_lock:
                refused = (self.max_paid_calls is not None and self.transport != "mock"
                           and self.attempts + 1 > self.max_paid_calls)
                if refused:
                    stop = PaidCallBudgetExhausted(self.attempts, self.max_paid_calls, context)
                else:
                    self.attempts += 1
                    if self.transport in ("anthropic", "openai", "openai_compat"):
                        self.paid_api_calls += 1
            if refused:
                if not attempt:
                    self._budget_stop(stop)
                budget_stop = stop
                break
            bodies.append(body)
            record["calls"] += 1
            if attempt:
                record["repairs"] += 1
            if self.transport in ("anthropic", "openai", "openai_compat"):
                record["paid_api_calls"] += 1
            raw_text = None
            persisted = False
            try:
                if self.transport in ("openai", "openai_compat"):
                    # Reply text/finish signal/usage/id use the OpenAI SDK's
                    # own shape (choices[0].message.content,
                    # usage.prompt_tokens/completion_tokens/
                    # prompt_tokens_details.cached_tokens, id) - normalized
                    # into `usage`/`request_id` via `_openai_usage` rather
                    # than stored as the provider's raw dump.
                    if self.transport == "openai_compat":
                        response, compatibility_request_id = _compat_completion(client, body)
                    else:
                        response = client.chat.completions.create(**body)
                    choice = response.choices[0]
                    # The reply text is masked AT CAPTURE, so the
                    # stored raw_text, the note, the ignored fields and the
                    # assistant turn the repair call re-sends are all masked
                    # by construction.
                    raw_text = mask_secrets(choice.message.content)
                    usage, usage_known = _openai_usage(getattr(response, "usage", None))
                    http_request_id = (compatibility_request_id if self.transport == "openai_compat"
                                       else _http_request_id(response))
                    # Identity, finish state and
                    # usage(known|unknown) are persisted BEFORE anything is
                    # parsed. `completion_id` is `chatcmpl_...` (the old key
                    # was named `request_id`, which read like the HTTP
                    # x-request-id) and `last_call_usage` is the last call's
                    # usage (the old key was named `usage`, which read like a
                    # run total).
                    record["responses"].append(
                        {
                            "attempt": attempt,
                            "returned_model": response.model,
                            "completion_id": response.id,
                            "http_request_id": http_request_id,
                            "finish_reason": choice.finish_reason,
                            "usage": usage,
                            "usage_known": usage_known,
                            "raw_text": raw_text,
                        }
                    )
                    record["returned_model"] = response.model
                    record["completion_id"] = response.id
                    if self.transport == "openai_compat":
                        record["response_id"] = response.id
                        record["responses"][-1]["response_id"] = response.id
                        returned_provider = getattr(response, "provider", None)
                        returned_provider = returned_provider if isinstance(returned_provider, str) else None
                        record["returned_provider"] = returned_provider
                        record["responses"][-1]["returned_provider"] = returned_provider
                    record["http_request_id"] = http_request_id
                    record["last_call_usage"] = usage
                    record["last_call_usage_known"] = usage_known
                    persisted = True
                    # An empty body is its own reply-contract failure,
                    # whatever the finish reason says.
                    if blank_reply(raw_text):
                        raise EmptyResponseBody(choice.finish_reason)
                    if choice.finish_reason != "stop":
                        raise ValueError("response did not finish with stop")
                else:
                    response = create_anthropic_message(client, body)
                    # Masked at capture (see the OpenAI branch).
                    raw_text = mask_secrets(
                        "\n".join(
                            block.text for block in response.content if block.type == "text"
                        )
                    )
                    usage, usage_known = _anthropic_usage(getattr(response, "usage", None))
                    http_request_id = _http_request_id(response)
                    # Persist identity/finish/usage BEFORE parsing.
                    record["responses"].append(
                        {
                            "attempt": attempt,
                            "returned_model": response.model,
                            "response_id": response.id,
                            "http_request_id": http_request_id,
                            "stop_reason": response.stop_reason,
                            "usage": usage,
                            "usage_known": usage_known,
                            "raw_text": raw_text,
                            # Thinking blocks are counted under a stated spec and
                            # their summaries recorded when the reply carried any; never parsed.
                            **({"thinking_blocks": _thinking_block_count(response)}
                               if self.reasoning is not None else {}),
                            **anthropic_thinking_record(response),
                        }
                    )
                    record["returned_model"] = response.model
                    record["http_request_id"] = http_request_id
                    persisted = True
                    if blank_reply(raw_text):
                        raise EmptyResponseBody(response.stop_reason)
                    if response.stop_reason != "end_turn":
                        raise ValueError("response did not finish with end_turn")
                # Input-name masking: a masked run judges the reply in the engine's names; the stored
                # raw_text and the echoed assistant turn stay as the model wrote them.
                if request in COMPOSED_REQUEST_KINDS:
                    # Unified protocol: the array (or the review object), judged object by object in the engine's names.
                    judged = (raw_text if self.name_mask is None
                              else self.name_mask.unmask_composed_reply(raw_text, constraints))
                    proposed, ignored = (parse_review_reply(judged, constraints, proposal_limit)
                                         if request == "composed_review"
                                         else parse_candidate_array(judged, constraints, proposal_limit))
                else:
                    candidate, ignored = parse_candidate_object(
                        raw_text if self.name_mask is None
                        else self.name_mask.unmask_reply(raw_text, constraints), constraints)
                    proposed = [candidate]
            except Exception as exc:  # noqa: BLE001 - after one repair the current pool is kept
                # ONLY a reply-contract failure is repairable - a
                # ValueError raised after the response was persisted (invalid
                # JSON, the object/shape rules, a judge_candidate message, a
                # finish/stop reason that is not the completed one). An API or
                # transport error, an SDK object-contract error and a missing
                # usage object are NOT: they are recorded and end the round
                # without a second paid call.
                repair_eligible = persisted and isinstance(exc, ValueError)
                masked_reason = mask_secrets(str(exc))
                if self.name_mask is not None:
                    masked_reason = self.name_mask.mask_text(masked_reason)
                rejected = {
                    "attempt": attempt,
                    "reason": masked_reason,
                    "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "status_code", None),
                    "repair_eligible": repair_eligible,
                }
                if repair_eligible:
                    # Which repair turn this failure earns - the key
                    # exists only where a repair turn is actually built.
                    rejected["repair_kind"] = ("empty_body" if isinstance(exc, EmptyResponseBody)
                                               else "malformed")
                record["rejected_candidates"].append(rejected)
                if not repair_eligible:
                    break
                # Both halves of the repair turn are masked BEFORE
                # they go on the wire - the echoed assistant reply (masked at
                # capture) and the validation message. An empty reply
                # is never echoed back - it is stated in one user turn.
                # Unified protocol: a composed request's repair turn asks again for the array (or the review object).
                instruction, empty_instruction = {
                    "composed": (REPAIR_INSTRUCTION_ARRAY, REPAIR_INSTRUCTION_ARRAY_EMPTY),
                    "composed_review": (REPAIR_INSTRUCTION_REVIEW, REPAIR_INSTRUCTION_REVIEW_EMPTY),
                }.get(request, (REPAIR_INSTRUCTION, REPAIR_INSTRUCTION_EMPTY))
                if isinstance(exc, EmptyResponseBody):
                    messages = [*messages, {"role": "user", "content": empty_instruction.format(
                        finish_reason=mask_secrets(str(exc.finish_reason)), limit=proposal_limit)}]
                else:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": raw_text},
                        {"role": "user", "content": instruction.format(error=masked_reason, limit=proposal_limit)},
                    ]
                continue
            candidates = proposed
            if request in COMPOSED_REQUEST_KINDS:
                # Unified protocol: the array's per-candidate refusals join the repair-turn refusals, never replace them.
                ignored = dict(ignored)
                record["rejected_candidates"] = [*record["rejected_candidates"], *ignored.pop("rejected_candidates")]
                if request == "composed_review":
                    ignored["revive_requested_engine_ids"] = [
                        self.name_mask.unmask_model_id(item) if self.name_mask is not None and isinstance(item, str)
                        else item for item in ignored["revive_requested"]]
            record.update(ignored)
            record["status"] = "parsed"
            break
        if budget_stop is not None:
            # The paid call(s) this proposal made stay recorded, under a stop status.
            record.update(status=BUDGET_EXHAUSTED, max_paid_calls=budget_stop.max_paid_calls)
        elif not candidates:
            record["status"] = "proposal_failed"
        # responses[*].usage summed exactly once. An unknown usage is
        # never counted as zero - it lands in `usage_missing_indices` and makes
        # `usage_total_complete` false, so a cost ledger reserves instead of
        # under-counting. NEVER add the top-level last_call_usage to this.
        usage_total, usage_complete, usage_missing = _usage_total(record["responses"])
        record["usage_total"] = usage_total
        record["usage_total_complete"] = usage_complete
        record["usage_missing_indices"] = usage_missing
        _write_once(
            prefix.with_suffix(".request.json"),
            {"identity": identity, "bodies": bodies, "client_injected": self.injected_client},
        )
        record.update(
            elapsed_seconds=time.perf_counter() - started,
            valid_candidate_count=len(candidates),
            silent_fallback=False,
        )
        # The RETURNED record is masked too, not just the written
        # file - run_gravity copies it into the run json and the event log.
        record = mask_secrets_deep(record)
        _write_once(prefix.with_suffix(".response.json"), record)
        if budget_stop is not None:
            self._budget_stop(budget_stop)
        return candidates, record


__all__ = [
    "BURST_PARAGRAPH",
    "BURST_REFILL_PARAGRAPH",
    "COMPOSITION_PARAGRAPH_V2",
    "COMPOSITION_PARAGRAPH_V2_SHA256",
    "ARCHIVE_REVIEW_PARAGRAPH_V1",
    "ARCHIVE_REVIEW_PARAGRAPH_V1_SHA256",
    "BURST_MOCK_SCHEDULE",
    "BURST_MOCK_EXPECTED_OUTCOMES",
    "COMPOSED_MOCK_SCHEDULE",
    "parse_candidate_array",
    "parse_review_reply",
    "MockUnifiedClient",
    "REQUEST_KINDS",
    "build_empty_pool_packet",
    "request_prompt",
    "request_prompt_identities",
    "ARCHIVE_OUTCOMES",
    "ARCHIVE_REASON_KEYS",
    "CONSTRAINT_KEYS",
    "FREE_CONSTRAINT_KEYS",
    "FREE_MOCK_EXPECTED_OUTCOMES",
    "FREE_MOCK_SCHEDULE",
    "MockFreeClient",
    "constraint_keys_ok",
    "free_constraints",
    "free_mock_response_text",
    "is_free_constraints",
    "archive_pool_keys",
    "known_structure_names",
    "structure_candidate_identity",
    "structure_coefficient_names",
    "structure_expression",
    "structure_roles",
    "COEFFICIENT_NAMES",
    "FIELD_DEFINITIONS",
    "INTERFACE_KEYS",
    "KIND_PRIOR",
    "KIND_ROLE",
    "MAX_DESIGNS",
    "MASK_PLACEHOLDERS",
    "MASKED_LOGGER_ROOTS",
    "MOCK_SCHEDULE",
    "MaskedCredentialError",
    "PARAMETERS_FORMAT_LINE",
    "PARAMETER_SHAPE_MESSAGE",
    "PRIOR_RESTATED_MESSAGE",
    "PROMPT_FILE",
    "PROMPT_SHA256",
    "PROMPT_VERSION",
    "Fb1CatalogProposer",
    "InputNameMask",
    "MASK_TEXT_REWRITES",
    "prompt_variant_active",
    "set_input_name_mask",
    "set_prompt_variant",
    "BUDGET_EXHAUSTED",
    "PAID_CALL_BUDGET_EXIT_CODE",
    "PaidCallBudgetExhausted",
    "STOP_SUMMARY_FILE",
    "anthropic_thinking_record",
    "create_anthropic_message",
    "parse_max_paid_calls",
    "sdk_call_kwargs",
    "validate_transport_reasoning",
    "MockCatalogClient",
    "PoolMember",
    "ScriptedClient",
    "build_belief_packet",
    "canonical_expression",
    "carries_secret",
    "catalog_constraints",
    "catalog_table",
    "downsample_indices",
    "expression_to_structure",
    "install_secret_log_filter",
    "normalize_parameter_role",
    "normalize_parameter_roles",
    "forcebench_interface",
    "judge_candidate",
    "mask_secrets",
    "mask_secrets_deep",
    "masked_exception",
    "mock_response_text",
    "model_id",
    "packet_digest",
    "parse_candidate_object",
    "prompt_identity",
    "round_significant",
    "serialize_packet",
    "system_prompt_bytes",
]
