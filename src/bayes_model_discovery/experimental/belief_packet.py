"""Belief packet v2: the complete LLM-facing run state, built from public data only.

The builder takes no held-out inputs, no evaluation grid, and no score object -
see `build_belief_packet`'s signature, which `test_belief_packet.py` asserts
statically, and the byte-stability test that varies only the held-out set and
gets an identical packet. Every field is a function of the public observations,
the current fits of the active pool, the archive of everything ever proposed in
this run, the cell constraints, and the machine-generated `interface` block -
which is the only task-specific content anywhere in the LLM's input, since the
system prompt is byte-identical across tasks (an internal design decision that
keeps all task information out of the system prompt).

Serialization is fixed at four significant digits so that a packet is a stable,
hashable artifact (`packet_bytes`, `packet_sha256` in the transport record).
"""

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .count_inference import model_logweights
from .count_models import (
    CATEGORICAL_KIND,
    FAMILIES,
    GRAMMAR_RESERVED_NAMES,
    IMPLEMENTED_FAMILIES,
    INDEXED_COEFFICIENT_RULE,
    LINK_BY_OUTCOME,
    MAX_DISTINCT_MEAN_COEFFICIENTS,
    OUTCOME_MENUS,
    PRIOR_RULES,
    active_grammar_version,
    active_particle_rows,
    initial_hypotheses,
    interface_categorical_levels,
    interface_input_names,
    llm_priors_enabled,
    particle_logpmf,
    prior_menu_constraints,
)
from .positive_inference import POSITIVE_MEAN_REQUIRED_FAMILIES

PACKET_VERSION = "v2"
SIGNIFICANT_DIGITS = 4

# Structured archive reasons. Every entry
# carries an `outcome` from this tuple and a `reason` object whose key set is
# fixed by the outcome - free text lives only inside `validation_message`/`error`,
# which are the engine's own verbatim messages, never model-authored prose.
#
# Outcome semantics:
#   active             - currently in the pool.
#   retired_by_evidence- dropped by evidence rank; eligible for re-evaluation
#                        and resurrection.
#   duplicate          - canonically identical to an existing model_id, linked.
#   rejected           - grammar/role/cell violation, or non-finite under the
#                        prior probe: failure history, and a corrected
#                        re-proposal of a DIFFERENT model_id is allowed.
#   fit_failed         - SMC or prediction raised.
# "Do not re-propose" binds only on model_ids that appear in the pool or the
# archive. Identity is count_models.Hypothesis.model_id (canonical AST + roles +
# family), so `b0+b1*z` and `b0+2*b1*z` are different models - the prior scale
# differs and they are never collapsed.
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
# A prior-probe rejection may carry, beside the fixed `validation_message`,
# a `detail` sentence naming the non-finite sub-expression, the input value and its
# declared range (prior_probe_detail.nonfinite_probe_sentence) - still the
# engine's own sentence, never model prose. No other outcome has an optional key.
ARCHIVE_OPTIONAL_REASON_KEYS = {"rejected": {"detail"}}
# Initial-burst support: `source` marks a model the initial burst proposed at round 0
# ("initial_burst"); absent for every other entry, so a run without a burst is
# byte-identical.
ARCHIVE_OPTIONAL_KEYS = ("note", "prior_probe", "source")
COEFFICIENT_ROLES = ("intercept", "signed", "positive", "shape")
CONSTRAINT_KEYS = {
    "structure_expansion",
    "family_expansion",
    "initial_family",
    "allowed_families",
    "initial_structures",
    "roles",
    # Three limits the engine already enforces on every candidate, stated so
    # the model can obey them.
    # Each value is read from the engine's own source in `cell_constraints`,
    # never typed here: the coefficient cap (count_models.
    # MAX_DISTINCT_MEAN_COEFFICIENTS), the active grammar's reserved coefficient
    # names (count_models.GRAMMAR_RESERVED_NAMES), and the families of this cell
    # whose mean must be finite and strictly positive (positive_inference.
    # POSITIVE_MEAN_REQUIRED_FAMILIES). That last one is family-scoped: a family
    # absent from the list - the real-outcome normal among them - carries no
    # positive-mean requirement.
    "max_distinct_mean_coefficients",
    "reserved_names",
    "positive_mean_required_families",
}
# Keys `cell_constraints` adds ONLY for a task whose interface declares a
# categorical input. They are how the indexed
# form and its coefficient-slot accounting reach the model at all: the SYSTEM
# prompt file is byte-frozen (`prompts/proposer_system_v2_0.txt`, sha256
# f095edf1...), so every task-specific rule must travel in the packet's own
# `constraints`/`interface` fields. A task with no categorical input emits
# neither key, so every CORE packet and every BoxingGym packet without a
# categorical input is byte-identical to the packet built before these keys existed.
OPTIONAL_CONSTRAINT_KEYS = {"indexed_coefficients", "coefficient_slot_rule"}
# The LLM-proposed-priors ablation adds ONE more optional key - the
# proposed-prior menu, its hyperparameter windows and the per-role menu the
# engine enforces - present only while `--priors llm` is on and absent from
# every default and production packet. Kept SEPARATE from
# OPTIONAL_CONSTRAINT_KEYS, which means "the keys a CATEGORICAL interface must
# add" and is asserted as such: a categorical packet states the
# indexed form whether or not that ablation is on.
LLM_PRIORS_CONSTRAINT_KEYS = {"priors"}
# The ChemBench coefficient-naming rule (a revision of an earlier interface fix,
# a project decision): the one key a ChemBench packet adds to
# state the coefficient-naming rule the engine already enforces (`count_models.
# validate_expression` refuses a coefficient named after an interface input or a reserved
# name). It travels in the packet because the SYSTEM prompt file is byte-frozen. Kept
# SEPARATE from OPTIONAL_CONSTRAINT_KEYS (the keys a CATEGORICAL interface must add) and from
# LLM_PRIORS_CONSTRAINT_KEYS; CORE and BoxingGym packets never emit it, so they stay
# byte-identical.
COEFFICIENT_NAME_RULE_CONSTRAINT_KEYS = {"coefficient_name_rule"}
#: The ChemBench-only prior-probe contract key (`chembench_core.PRIOR_PROBE_RULE`),
#: like the naming rule outside OPTIONAL_CONSTRAINT_KEYS so no other bundle's packet grows it.
PRIOR_PROBE_RULE_CONSTRAINT_KEYS = {"prior_probe_rule"}

# The `interface` block (an internal design decision on task interfaces)
# is the ONLY task-specific part of a packet, and it is machine-generated from
# the task specification:
# structured fields, no free text, no world names, no truth parameters. The
# system prompt is byte-identical across every task and carries none of this.
INTERFACE_KEYS = ("outcome_type", "support", "trials", "inputs", "derived_inputs", "link")

# Fixed, machine-readable statements of what the residual block means. `pred_sd`
# is the sd of the POSTERIOR PREDICTIVE of the outcome: it mixes the particle
# posterior over coefficients (coefficient uncertainty) with the family's own
# count likelihood (observation noise). It is not the sd of the link mean.
RESIDUAL_DEFINITIONS = {
    "pred_mean": "posterior_predictive_mean_of_the_outcome",
    "pred_sd": "posterior_predictive_sd_includes_coefficient_uncertainty_and_observation_noise",
    "standardized_residual": "(y-pred_mean)/pred_sd",
}


@dataclass(frozen=True)
class PoolMember:
    """One active model as the packet sees it: its fit plus its round history.

    `one_step_log_scores` is aligned to `one_step_rounds` one-for-one and holds
    null for every round before the model was born: entries are
    never back-filled from a later fit.
    """

    hypothesis: object
    particles: dict
    log_evidence: float
    one_step_log_scores: tuple = ()
    one_step_rounds: tuple = ()
    born_round: int = 0
    admitted_round: int = 0
    active_rounds: tuple = ()
    log_evidence_computed_round: int = 0
    log_evidence_n_obs: int = 0
    # Scalar-path binding: the frozen D0 response unit that scales the
    # log-link mean. CORE's bounded-count path never reads it.
    response_unit: float = 1.0
    # Identity-link half of the same frozen D0 calibration: the real
    # families predict `response_center + response_unit * expression`. Every
    # other path leaves it at 0.0 and never reads it.
    response_center: float = 0.0


def core_interface(trials=50):
    """The CORE `interface` block, generated from the task specification.

    Kept as an explicit literal so its bytes cannot drift; count_models'
    `CORE_INPUTS`/`CORE_DERIVED_INPUTS` hold the same declaration for the
    grammar, and `test_core_interface_matches_the_grammar_defaults` binds them.
    """
    return {
        "outcome_type": "bounded_count",
        "support": [0, int(trials)],
        "trials": int(trials),
        "inputs": [{"name": "u", "range": [0, 1]}],
        "derived_inputs": {"z": "2*u-1"},
        "link": "logit",
    }


def core_real_interface():
    """Gaussian CORE has real outcomes, no trial count or bounded support."""
    return {
        "outcome_type": "real",
        "inputs": [{"name": "u", "range": [0, 1]}],
        "derived_inputs": {"z": "2*u-1"},
        "link": "identity",
    }


def task_interface(outcome_type, inputs, derived_inputs, support, trials=None):
    """A task's `interface` block, machine-generated, with the link fixed by type.

    The key set is exactly `INTERFACE_KEYS` for every task - `trials` is null
    wherever the outcome type has no trial count - so one schema serves CORE and
    every adapter task. `link` is never an argument: the paper's task table
    fixes it per data type (`count_models.LINK_BY_OUTCOME`), and it is never
    proposed. The block is validated here, so a malformed task specification
    fails before any observation is taken.
    """
    if outcome_type not in LINK_BY_OUTCOME:
        raise ValueError(f"outcome type outside Table 2: {outcome_type}")
    interface = {
        "outcome_type": outcome_type,
        "support": [None if v is None else float(v) for v in support],
        "trials": None if trials is None else int(trials),
        "inputs": [
            {
                "name": item["name"],
                "range": [float(v) for v in item["range"]],
                # A CATEGORICAL input carries its kind and its declared
                # level list; a numeric input keeps exactly the earlier
                # two-key shape, so every earlier interface block is
                # byte-identical.
                **(
                    {"kind": CATEGORICAL_KIND, "levels": list(item["levels"])}
                    if item.get("kind") == CATEGORICAL_KIND
                    else {}
                ),
            }
            for item in inputs
        ],
        "derived_inputs": dict(derived_inputs or {}),
        "link": LINK_BY_OUTCOME[outcome_type],
    }
    interface_input_names(interface)
    if outcome_type == "bounded_count" and interface["trials"] is None:
        raise ValueError("a bounded count interface must declare its trial count")
    return interface


def interface_families(outcome_type, family_expansion, initial_family=None, zero_inflation=True):
    """Table 2's menu for an outcome type, intersected with what this tree can fit.

    `zero_inflation` gates ONLY the unbounded-count row's `zi_poisson`. The
    paper's task table attaches the caveat to that row alone -
    "Negative Binomial($\\alpha$); zero-inflated Poisson only where zeros are
    natural" - so whether zeros are structurally possible is a per-task
    declaration made from the task's PUBLIC description, never from the
    environment's truth. The bounded-count row lists "zero-inflated
    Binomial($\\pi$)" with no caveat and is never gated, so CORE and every
    bounded-count task keep their menu exactly as it was.
    """
    menu = OUTCOME_MENUS[outcome_type]
    initial = menu["initial_family"] if initial_family is None else initial_family
    allowed = [f for f in menu["allowed_families"] if f in IMPLEMENTED_FAMILIES]
    if not zero_inflation:
        allowed = [f for f in allowed if f != "zi_poisson"]
    if initial not in allowed:
        raise ValueError(f"initial family {initial} is not runnable for {outcome_type}")
    return initial, (allowed if family_expansion else [initial])


def interface_grid(interface, points):
    """Uniform reference grid over the interface's input ranges.

    One input gives a one-dimensional vector; several give an [point, input]
    matrix whose columns each sweep their own declared range (a diagonal sweep,
    which is what the finiteness probe and the prior predictive need - a full
    product grid would be exponential in the input count).
    """
    columns = []
    for item in interface["inputs"]:
        low, high = item["range"]
        column = np.linspace(float(low), float(high), int(points))
        if item.get("kind") == CATEGORICAL_KIND:
            # A categorical column holds LEVEL CODES, so the reference
            # grid sweeps the declared codes (repeating them when `points`
            # exceeds the level count) instead of the interpolated values
            # between them, which are undeclared levels.
            column = np.rint(column)
        columns.append(column)
    return columns[0] if len(columns) == 1 else np.stack(columns, axis=1)


def cell_constraints(
    cell,
    initial_family,
    *,
    allowed_families=None,
    initial_structures=None,
    input_names=None,
    categorical_inputs=(),
    llm_priors=None,
):
    """The cell's expansion permissions and the fixed role priors, as the packet states them.

    The three keyword arguments come from the task specification; their
    defaults reproduce CORE exactly.

    `categorical_inputs` is `interface_categorical_levels(interface)`.
    When it is non-empty two further keys are emitted -
    `OPTIONAL_CONSTRAINT_KEYS` - stating the indexed form the grammar accepts
    for those inputs and how it is charged against the coefficient cap. That
    statement has to live here, in the packet, because the SYSTEM prompt file
    is byte-frozen. Empty (a task with no categorical input) emits neither key.

    `llm_priors` (the LLM-proposed-priors ablation) is whether this run
    accepts the optional `priors` block; None - the default - means "whatever
    `count_models.llm_priors_enabled()` says". ON, one further key `priors`
    states the menu, the hyperparameter windows and the per-role menu the
    engine will enforce, for the same reason the indexed block lives here: both
    SYSTEM prompt files are byte-frozen, so a machine-checkable statement of the
    accepted windows has to travel with the packet. OFF - the default and every
    production arm - the key is absent and every packet is byte-identical to
    the packet built before this ablation existed.

    Three keys state limits the engine enforces on every candidate,
    each read from the engine's own source rather than typed here -
    `max_distinct_mean_coefficients` is `count_models.
    MAX_DISTINCT_MEAN_COEFFICIENTS`, the cap `validate_expression` applies;
    `reserved_names` is the ACTIVE grammar's reserved coefficient names
    (`GRAMMAR_RESERVED_NAMES[active_grammar_version()]`, sorted), the set that
    check refuses beside the interface's own input names; and
    `positive_mean_required_families` is this cell's `allowed_families` that
    appear in `positive_inference.POSITIVE_MEAN_REQUIRED_FAMILIES`, sorted - the
    families whose mean must be finite and strictly positive. It is
    family-scoped: a family the list does not name (normal, and every bounded-
    count or binary family) has no such requirement, and a cell whose families
    are all of that kind states an empty list.
    """
    if cell not in {f"{s}{f}{a}" for s in "01" for f in "01" for a in "012"}:
        raise ValueError("invalid cell or initial family")
    if allowed_families is None and initial_family not in FAMILIES:
        raise ValueError("invalid cell or initial family")
    family_expansion = cell[1] == "1"
    menu = list(FAMILIES) if allowed_families is None else list(allowed_families)
    allowed = menu if family_expansion else [initial_family]
    structures = (
        [item.expression for item in initial_hypotheses(initial_family)]
        if initial_structures is None
        else list(initial_structures)
    )
    if input_names is not None and initial_structures is None:
        raise ValueError("a task namespace requires its own initial structures")
    constraints = {
        "structure_expansion": cell[0] == "1",
        "family_expansion": family_expansion,
        "initial_family": initial_family,
        "allowed_families": allowed,
        "initial_structures": structures,
        "roles": {
            role: f"{PRIOR_RULES[role][0]}({PRIOR_RULES[role][1]:g},{PRIOR_RULES[role][2]:g})"
            for role in COEFFICIENT_ROLES
        },
        "max_distinct_mean_coefficients": MAX_DISTINCT_MEAN_COEFFICIENTS,
        "reserved_names": sorted(GRAMMAR_RESERVED_NAMES[active_grammar_version()]),
        "positive_mean_required_families": sorted(
            family for family in allowed if family in POSITIVE_MEAN_REQUIRED_FAMILIES
        ),
    }
    if categorical_inputs:
        constraints["indexed_coefficients"] = {
            name: {
                "n_levels": int(levels),
                "form": f"<coefficient>[{name}]",
                "role_required": "signed",
                "counts_as_coefficient_slots": 1,
                "parameters_contributed": int(levels),
            }
            for name, levels in categorical_inputs
        }
        constraints["coefficient_slot_rule"] = INDEXED_COEFFICIENT_RULE
    if llm_priors_enabled() if llm_priors is None else llm_priors:
        constraints["priors"] = prior_menu_constraints()
    return constraints


def round_significant(value, digits=SIGNIFICANT_DIGITS):
    """Recursively round every float to `digits` significant digits; ints stay exact."""
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
    """Deterministic JSON text of a packet at the fixed serialization precision."""
    return json.dumps(
        round_significant(packet), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def packet_digest(text):
    """Size and hash of a serialized packet, for the transport record."""
    raw = text.encode()
    return {"packet_bytes": len(raw), "packet_sha256": hashlib.sha256(raw).hexdigest()}


# Outcome types whose observations are integers; everything else is a float.
INTEGER_OUTCOMES = ("bounded_count", "binary", "unbounded_count")
# Outcome types whose predictive distribution has a finite support the runner
# enumerates exhaustively (count_models.particle_logpmf): the bounded count over
# 0..N and the binary outcome over {0,1} with one trial.
EXHAUSTIVE_COUNT_OUTCOMES = ("bounded_count", "binary")


def _predictive_moments(member, designs, interface):
    """Posterior predictive mean and sd of one member at the public inputs.

    Public data only. A bounded count goes through the exhaustive count PMF
    exactly as in CORE's original packet (the interface it is now handed
    declares CORE's own `u`/`z`, so every float is unchanged); a BINARY outcome
    is the same exhaustive table over {0,1} with one trial; other outcome types
    use the closed-form mixture moments of the scalar families (positive_inference,
    which routes the identity-link real families to real_inference).
    """
    if interface["outcome_type"] in EXHAUSTIVE_COUNT_OUTCOMES:
        # The packet's moments are a posterior average, so a particle
        # given zero weight leaves both the sum and the divisor.
        rows = active_particle_rows(particle_logpmf(
            member.hypothesis,
            member.particles,
            np.asarray(designs, dtype=float),
            interface["trials"],
            interface,
        ))
        pmf = np.exp(logsumexp(rows, axis=0) - np.log(rows.shape[0]))
        counts = np.arange(pmf.shape[-1])
        mean = pmf @ counts
        return mean, np.sqrt(np.maximum(pmf @ counts**2 - mean**2, 0.0))
    from .positive_inference import mixture_moments

    return mixture_moments(
        member.hypothesis,
        member.particles,
        designs,
        interface,
        member.response_unit,
        member.response_center,
    )


def _summaries(particles, names):
    """Posterior mean/sd of each named parameter, as the packet reports them.

    An INDEXED coefficient's particle block is [particle, level], so it
    is summarized PER LEVEL - a single scalar mean over a 6-level offset vector
    would tell the model nothing about the offsets it is being asked to revise.
    Every scalar parameter keeps the original two-float shape exactly.
    """
    summaries = {}
    for name in names:
        if name not in particles:
            continue
        block = np.asarray(particles[name], dtype=float)
        if block.ndim == 1:
            summaries[name] = {
                "mean": float(np.mean(block)),
                "sd": float(np.std(block, ddof=1)),
            }
        else:
            summaries[name] = {
                "mean": [float(v) for v in block.mean(axis=0)],
                "sd": [float(v) for v in block.std(axis=0, ddof=1)],
                "n_levels": int(block.shape[1]),
            }
    return summaries


def _archive_entry(item):
    outcome = item["outcome"]
    if outcome not in ARCHIVE_OUTCOMES:
        raise ValueError(f"unknown archive outcome: {outcome}")
    reason = dict(item["reason"])
    required = ARCHIVE_REASON_KEYS[outcome]
    if not required <= set(reason) <= required | ARCHIVE_OPTIONAL_REASON_KEYS.get(outcome, set()):
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


def build_belief_packet(public_observations, active_fits, archive, constraints, interface, round):
    """Assemble the belief packet v2 from public state only.

    Arguments (there are deliberately no others - no held-out inputs, no
    held-out counts, no true PMF, no scores, no `per_model_nll`):

    public_observations
        the run's public observations so far, each with `u`, `y`, `round` and
        `obs_id`; nothing else is copied, so observation seeds never leak.
    active_fits
        a sequence of `PoolMember`.
    archive
        every model this run has ever proposed or held, each already carrying a
        structured `outcome`/`reason` pair (see ARCHIVE_REASON_KEYS).
    constraints
        the cell's expansion permissions and role priors (`cell_constraints`).
    interface
        the machine-generated task interface block (`core_interface`) - the only
        task-specific part of a packet, and the only place task information
        enters the LLM's input at all.
    round
        the current round index.
    """
    members = list(active_fits)
    if not members and int(round) != 0:
        raise ValueError("belief packet requires a non-empty active pool")
    # At round 0 a burst call may find the pool empty (no catalogue seeded);
    # the packet then carries an empty pool, no best model and no residuals.
    if not CONSTRAINT_KEYS <= set(constraints) or set(constraints) - (
        CONSTRAINT_KEYS | OPTIONAL_CONSTRAINT_KEYS | LLM_PRIORS_CONSTRAINT_KEYS
        | COEFFICIENT_NAME_RULE_CONSTRAINT_KEYS | PRIOR_PROBE_RULE_CONSTRAINT_KEYS
    ):
        raise ValueError("belief packet constraints keys are fixed")
    if tuple(sorted(interface)) != tuple(sorted(INTERFACE_KEYS)) and interface != core_real_interface():
        raise ValueError("interface must be exactly the generated structured block")
    if interface["link"] != LINK_BY_OUTCOME.get(interface["outcome_type"]):
        raise ValueError("link is fixed by the outcome type (the paper's task table)")
    # An interface that declares a categorical input MUST also state the
    # indexed form and its slot accounting, because the byte-frozen SYSTEM
    # prompt cannot. Checked here so no categorical run can silently ship a
    # packet the model cannot read the grammar of.
    if interface_categorical_levels(interface) and "indexed_coefficients" not in constraints:
        raise ValueError("a categorical interface must state its indexed-coefficient block")
    names = [item["name"] for item in interface["inputs"]]
    integer = interface["outcome_type"] in INTEGER_OUTCOMES
    observations = sorted(
        (
            {
                "obs_id": int(row["obs_id"]),
                "round": int(row["round"]),
                **{name: float(row[name]) for name in names},
                "y": int(row["y"]) if integer else float(row["y"]),
            }
            for row in public_observations
        ),
        key=lambda row: (*(row[name] for name in names), row["obs_id"]),
    )
    weights = np.exp(model_logweights(members)) if members else np.zeros(0)
    order = sorted(range(len(members)), key=lambda i: (-weights[i], members[i].hypothesis.model_id))
    pool = []
    for index in order:
        member = members[index]
        pool.append(
            {
                "model_id": member.hypothesis.model_id,
                "expression": member.hypothesis.expression,
                "family": member.hypothesis.family,
                "weight": float(weights[index]),
                "log_evidence": float(member.log_evidence),
                "log_evidence_computed_round": int(member.log_evidence_computed_round),
                "log_evidence_n_obs": int(member.log_evidence_n_obs),
                "born_round": int(member.born_round),
                "coefficients": _summaries(
                    member.particles, [name for name, _ in member.hypothesis.parameters]
                ),
                "family_dispersion": _summaries(
                    member.particles, member.hypothesis.noise_parameters
                ),
                "one_step_log_scores": list(member.one_step_log_scores),
                "one_step_rounds": list(member.one_step_rounds),
            }
        )
    if members:
        best = members[order[0]]
        columns = np.asarray([[row[name] for name in names] for row in observations], dtype=float)
        inputs = columns[:, 0] if len(names) == 1 else columns
        mean, sd = _predictive_moments(best, inputs, interface)
        residuals = [
            {
                "obs_id": row["obs_id"],
                **{name: row[name] for name in names},
                "y": row["y"],
                "pred_mean": float(mean[i]),
                "pred_sd": float(sd[i]),
                "standardized_residual": float((row["y"] - mean[i]) / sd[i]) if sd[i] > 0 else None,
            }
            for i, row in enumerate(observations)
        ]
    else:
        best, residuals = None, []
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
        "best_model": None if best is None else f"{best.hypothesis.expression} | {best.hypothesis.family}",
        "best_model_id": None if best is None else best.hypothesis.model_id,
        "best_model_residuals": residuals,
        "best_model_residual_definitions": dict(RESIDUAL_DEFINITIONS),
        "archive": entries,
        "constraints": dict(constraints),
    }
