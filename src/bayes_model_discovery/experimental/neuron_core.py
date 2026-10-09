"""Deterministic Neuron development path; not stochastic benchmark completion."""

import argparse
import hashlib
import importlib.metadata
import json
import resource
import sys
import time
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..provenance.canonical import sha256_file
from .belief_packet import serialize_packet
from .core_count import (
    ACTIVE_MAXIMUM,
    PRIOR_PROBE_SAMPLES,
    RUN_LABEL_CHOICES,
    append_json,
    points_label,
    seed_for,
    write_json,
)
from .count_inference import (
    CountInferenceFailure,
    FitTimeout,
    SMCSettings,
    loglik_prior_probe,
    model_logweights,
    record_fit_timing,
    resolve_epig_policy,
    retain,
    run_with_fit_guards,
    run_with_fit_timeout,
    validate_fit_timeout,
)
from .count_proposer import (
    PAID_CALL_BUDGET_EXIT_CODE,
    PaidCallBudgetExhausted,
    budget_exhausted_summary,
    compat_config,
    compat_option_kwargs,
    compat_points_metadata,
    mask_secrets,
    mask_secrets_deep,
    masked_exception,
    normalize_provider_order,
    parse_max_paid_calls,
    parse_max_tokens,
    parse_reasoning_spec,
    require_registered_decoding,
)
from .neuron_forward import (
    EVAL_FEATURE_NAMES,
    EVAL_FEATURE_SCALES,
    eval_feature_vector,
    initial_hypotheses,
    mock_proposals,
    native_test_counts,
    predict_eval_features,
    predict_features,
    summarize_voltage,
)
from .neuron_forward_batch import (
    numpy_cpu_features_disabled,
    numpy_cpu_features_enabled,
    power_matches_scalar,
    predict_batch,
)
from .neuron_proposer import (
    CONTINUOUS_PARAMETERS,
    DEFAULT_FEATURE_SPACE,
    FeatureSpace,
    build_neuron_belief_packet,
    make_neuron_proposer,
    mechanism_vocabulary,
    structure_mode,
)
from .neuron_proposer import composed_system_prompt as neuron_composed_system_prompt
from .neuron_proposer import validate_proposal_limit as neuron_validate_proposal_limit
from .run_label_policy import bmd_label_receipt, recheck_manifest
from .vector_inference import (
    EIG_PART_OBJECTIVES,
    FORWARD_BACKENDS,
    MH_ESS_THRESHOLD,
    MH_STEPS_PER_STAGE,
    ForwardOp,
    fit,
    joint_eig,
    joint_epig,
    log_predictive,
    particle_sigmas,
    prediction_bank,
    predictive_summary,
)
from .vector_inference import (
    POINT_DISAGREEMENT_DEFINITION as VECTOR_POINT_DISAGREEMENT_DEFINITION,
)
from .vector_inference import (
    point_disagreement as vector_point_disagreement,
)

# --------------------------------------------------------------------------
# NeuronBench development runner, first stage.
# Six worlds are enumerated because the package has six
# (neuronbench.list_worlds()); only h_sag was run through this pipeline at
# first, when it alone was "wired" and the other five were refused, not
# silently attempted. `WIRED_WORLDS` below now lists all six.
# --------------------------------------------------------------------------
WORLDS = ("z_rebound", "h_sag", "na_fatigue", "ca_rebound", "d_type", "textbook_M")
WIRED_WORLDS = frozenset(WORLDS)

# CORE-style dev/paper particle counts (core_count.py, boxinggym_core.py):
# dev=16 matches this runner's own pre-existing hardcoded default, so a
# default invocation (no --profile, no --particles) is unchanged; paper=500
# borrows CORE's paper-profile particle count. NeuronBench has no q/grid/EIG
# axis to scale alongside particles the way BoxingGym/CORE do.
#
# `paper_neuron` is a third profile that exists to be comparable with the published
# MDA study rather than with this suite's own `paper` runs: 200 particles, two chains, a
# fixed-step random-walk MH kernel and the benchmark's own six evaluation features as the
# training likelihood. It is NOT a finer or coarser `paper`; `paper` and `paper_neuron` differ
# in kernel and in feature space, so their numbers are not each other's baselines.
PARTICLES_BY_PROFILE = {"dev": 16, "paper": 500, "paper_neuron": 200}
#: The profile whose keys/kernel/feature space everything below branches on.
PAPER_NEURON = "paper_neuron"
#: The active-pool bound is a property of the profile, not of the runner. `dev` and
#: `paper` keep `core_count.ACTIVE_MAXIMUM` exactly (their config.json bytes are unchanged);
#: `paper_neuron` truncates at six, the bound the measured wall clock admits for a 200-particle
#: fixed-step run (a twelve-model pool costs about ten hours a cell against six's under six).
#: The config, the truncation and the registered declaration all read THIS mapping, so the
#: recorded cap and the operating cap cannot drift apart.
#: Spelled out as a literal because the driver's label preflight reads it out of THIS source
#: with `ast.literal_eval` - that preflight runs on the system interpreter, which cannot
#: import this module - so the driver declares the bound the runner will truncate with.
PAPER_NEURON_ACTIVE_MAXIMUM = 6
ACTIVE_MAXIMUM_BY_PROFILE = {"dev": ACTIVE_MAXIMUM, "paper": ACTIVE_MAXIMUM,
                             PAPER_NEURON: PAPER_NEURON_ACTIVE_MAXIMUM}
#: The profile each of ours is resolved as by `run_label_policy.py`. The policy registers
#: `paper_neuron` itself, with its (particles, chains) == (200, 2), kernel, feature space and
#: active-pool bindings, so every profile now maps to its own name and a receipt records the
#: profile the run actually executed. The earlier stand-in mapping to `dev` is gone.
LABEL_POLICY_PROFILE = {"dev": "dev", "paper": "paper", PAPER_NEURON: PAPER_NEURON}

# The plotting contract: the 21 shared-collector columns (suite..variant, with
# y_label), then code_sha, then brier/nmsle/rmsle - undefined for this
# suite's spike_forecast_mse metric, so always empty strings, never 0.
POINTS_COLUMNS = (
    "suite", "task", "method", "proposer", "seed", "x", "x_unit",
    "y_primary", "y_name", "y_label", "nll", "coverage90", "width90",
    "crps", "is_0.1", "kl", "status", "label", "run_id", "source_path",
    "variant", "code_sha", "brier", "nmsle", "rmsle",
)


def world_wiring():
    """One line per package world, honestly marked wired/not_wired."""
    return {name: ("wired" if name in WIRED_WORLDS else "not_wired") for name in WORLDS}


def require_wired_world(world):
    """Refuse an unwired world before any output is created."""
    if world not in WIRED_WORLDS:
        not_wired = sorted(set(WORLDS) - WIRED_WORLDS)
        raise ValueError(
            f"world {world!r} is not wired in this runner (not wired: {not_wired}; "
            f"wired: {sorted(WIRED_WORLDS)})"
        )


def resolve_particles(profile, particles):
    """`--profile` implies a particle count when `--particles` is not given;
    an explicit `--particles` must agree with it (never silently overridden
    or silently ignored)."""
    implied = PARTICLES_BY_PROFILE[profile]
    if particles is None:
        return implied
    if particles != implied:
        raise ValueError(
            f"--particles {particles} is inconsistent with --profile {profile} "
            f"(implies {implied}); pass a matching --particles or omit it"
        )
    return particles


#: Mixture draws `joint_epig` averages over, per round (the EIG pass uses 64).
EPIG_PREDICTIVE_SAMPLES = 128

# EIG-part arms (added 2026-10-05): the two chain-rule parts of the joint EIG as opt-in arms
# on an A=1 cell, I((M,theta);Y) = I(M;Y) + E_M I(theta;Y|M); Model EIG is MDA v4 Eq. 26.
EIG_PART_DEFINITION = {
    "structure_eig": "I(M;Y|x,D) = H[p(y|x,D)] - sum_m p(m|D) H[p(y|x,m,D)] (Model EIG, MDA v4 Eq. 26)",
    "parameter_eig": "sum_m p(m|D) I(theta;Y|x,m,D) = joint EIG - Model EIG (Parameter EIG)",
}
EIG_PART_ESTIMATOR = (
    "Monte Carlo on joint_eig's own 64 draws (vector_inference.joint_eig objective=...): "
    "structure = E[log p_m(y) - log p(y)], parameter = E[log p(y|theta,m) - log p_m(y)], "
    "p_m = model m's weighted particle mixture"
)
EIG_PART_VARIANT = {"structure_eig": "design_structure_eig", "parameter_eig": "design_parameter_eig"}
# `--record-design-criteria` adds to every acquisition.jsonl
# row a `design_criteria` block: joint, Model and Parameter EIG (one table, the joint arm's draws on
# the round's EIG seed) and the point disagreement of every available design. A record only.
DESIGN_CRITERIA_DEFINITIONS = {
    "eig_definition": "joint EIG I((M,theta);Y|d,D) = structure_eig + parameter_eig",
    "structure_eig_definition": EIG_PART_DEFINITION["structure_eig"],
    "parameter_eig_definition": EIG_PART_DEFINITION["parameter_eig"],
    "eig_estimator": EIG_PART_ESTIMATOR,
    "point_disagreement_definition": VECTOR_POINT_DISAGREEMENT_DEFINITION,
    "eig_seed_rule": "seed_for('neuron_eig', seed, round): the joint arm's EIG seed",
    "role": "record only; the design decision is the row's selected/mode/design_policy",
}


def feature_space(profile):
    """Everything the training feature space decides, in one place.

    `forward` is the per-particle feature function a fit and a prediction bank both call;
    `observe` is the function the OBSERVED trace goes through, and the two are deliberately the
    same measurement on the two sides. `scales` is None on the three-feature path, where one
    sigma in millivolts covers every feature.
    """
    if profile != PAPER_NEURON:
        return {"forward": predict_features, "observe": summarize_voltage,
                "scales": None, "record_key": "features_mV"}
    return {"forward": predict_eval_features, "observe": eval_feature_vector,
            "scales": list(EVAL_FEATURE_SCALES), "record_key": "features"}


def proposer_feature_space(profile):
    """The belief packet's observation feature space for this profile.

    Built from the same profile spec `feature_space` returns, and read by all three parts of
    the proposal path - the packet's observation rows, the check that refuses a row from
    another space, and the candidate finiteness probe - so a profile proposes on the
    features it trains on. The three-feature profiles get the transport's own default
    descriptor, whose packet and transport record are unchanged byte for byte.
    """
    space = feature_space(profile)
    if space["scales"] is None:
        return DEFAULT_FEATURE_SPACE
    return FeatureSpace(record_key=space["record_key"], names=tuple(EVAL_FEATURE_NAMES),
                        probe=space["forward"], feature_scales=tuple(space["scales"]))


def feature_space_config(profile):
    """The three config keys that describe the likelihood, which the profile decides."""
    if profile != PAPER_NEURON:
        return {
            "noise_prior": "sigma HalfNormal(1), fixed unit 20 mV",
            "likelihood": "independent Gaussian on three correlated physical features; working/composite score, not voltage density",
            "training_features": ["test_window_mean_mV", "minimum_mV", "final_mV"],
        }
    return {
        "noise_prior": (
            "sigma HalfNormal(1), one per-feature scale each (neuronbench.evaluator's own "
            "evaluation-feature standardisation), not a single mV unit"
        ),
        "likelihood": (
            "independent Gaussian on the benchmark's six correlated evaluation features "
            "(four spike counts, two voltages); working/composite score, not voltage density"
        ),
        "training_features": list(EVAL_FEATURE_NAMES),
    }


def paper_neuron_config():
    """The keys only a `paper_neuron` run records - the kernel and feature space it declares.

    Written for that profile alone, so a `dev` or `paper` run's config.json keeps its exact key
    set (the same "never unconditionally" rule `label` and `cell_requested` follow).
    """
    return {
        "profile_family": "mda_comparison",
        "kernel": {
            "name": "mh_fixed_steps",
            "steps_per_stage": MH_STEPS_PER_STAGE,
            "ess_threshold": MH_ESS_THRESHOLD,
            "proposal": "pymc_MH_random_walk_mvn_scaled",
        },
        "feature_space": "eval_features_6d",
        "feature_scales_mV_or_count": list(EVAL_FEATURE_SCALES),
        # The earlier `label_policy_profile` / `label_policy_profile_reason` pair is removed
        # here: the policy registers `paper_neuron` as itself now, so `profile`
        # already states what the label receipt binds and a second, disagreeing key would
        # only record a stand-in that no longer exists.
    }


def resolve_acquisition(cell, acquisition):
    """NeuronBench cell wiring: `--cell`'s third character
    (the Acquisition digit) is the real switch now - '0' implies `random`, '1'
    implies `joint`, '2' implies `epig`. `--acquisition` stays as an optional confirm-only flag
    (argparse default None); if given and it disagrees with what the cell
    implies, the run is refused before any output is created, naming both the
    cell and the value it implies. The default `--cell "111"` implies
    `joint` - the pre-existing hardcoded default - so the anchor invocation
    (which passes `--acquisition joint` at the default cell) stays
    byte-identical.

    (Earlier versions instead wrote a `design_axis: "absent_stage1"` disclosure here,
    asserting this bit had NO effect on execution; once the bit was wired
    that stopped being true, so the disclosure was removed outright, not
    renamed.)
    """
    resolve_epig_policy(cell)
    implied = {"0": "random", "1": "joint", "2": "epig"}[cell[2]]
    if acquisition is not None and acquisition != implied:
        raise ValueError(
            f"--acquisition {acquisition!r} is inconsistent with --cell {cell!r} "
            f"(implies {implied!r}); pass a matching --acquisition or omit it"
        )
    return implied


def proposals_enabled_for_cell(cell):
    """`--cell`'s first character (the Structure bit) now really
    switches the mock-proposal round on ('1', the earlier behaviour, unchanged) or
    off ('0'). When off, every round still fits/observes/acquires/scores the
    initial hypothesis pool - only the proposal round (new-hypothesis mock
    catalogue, admission, and the `after_proposal` checkpoint) is skipped;
    `proposals.jsonl` is left as an empty file rather than never created
    (see `run()`)."""
    return structure_mode(cell) == "free"


# NeuronBench has no Family axis (the likelihood path is kept
# unchanged - an isotropic Normal synthetic likelihood with a
# single scalar sigma). Every run's config discloses this plainly, regardless
# of cell; the `--cell` choices keep their 2nd-character '1' for 110/111 by
# label convention only, not because a family distinction is
# actually wired anywhere in this runner.
FAMILY_AXIS_NOT_APPLICABLE = "not_applicable_isotropic_normal_sigma"


def canonical_cell(cell):
    """The cell that actually executes, for any of the twelve accepted cell strings.

    The Family bit `cell[1]` is NOT APPLICABLE to this runner: the likelihood
    is one fixed isotropic Normal with a single scalar sigma
    (`FAMILY_AXIS_NOT_APPLICABLE` above), and nothing in this module ever
    reads that character - the only executable subscripts of `cell` anywhere
    here are `cell[2]` in `resolve_acquisition` and `cell[0]` in
    `proposals_enabled_for_cell`. The four pre-existing cells
    (000/001/110/111) mirror the Structure bit into the family position by
    label convention only (the note above), so a cell that instead fixes the
    family bit at 0 - the convention every other bundle's F0 cells use - is
    an alias of its mirrored counterpart and not new behaviour:

        100 -> 110, 101 -> 111, 010 -> 000, 011 -> 001,

    and the four pre-existing cells map to themselves; the EPIG cells (third
    digit '2') follow the same rule (102 -> 112, 012 -> 002). Total and
    idempotent over the twelve accepted strings. `run()` canonicalises before anything
    else reads `cell`, so the recorded cell is the executed cell; the raw
    request is recorded separately as `cell_requested` when the two differ.

    A future option, named here and deliberately NOT implemented: give
    `cell[1]` a real second axis as a mechanism-class menu (0 = the
    non-inactivating candidate class only, 1 = the inactivating class as
    well). That would delete the equivalences above and change what a `1?0`
    cell means, so it is a decision about the factorial rather than a coding
    choice; until it is taken, `cell[1]` carries no meaning at all.
    """
    structure = "1" if proposals_enabled_for_cell(cell) else "0"
    return structure + structure + cell[2]


def cell_axis_config(cell):
    """The three cell-axis config keys, computed once per `--cell` value.
    They replace the `design_axis` key/token of earlier versions outright -
    that disclosure asserted the acquisition bit had no effect, which stopped
    being true the moment the acquisition digit was wired (`resolve_acquisition`)."""
    return {
        "acquisition_from_cell": True,
        "proposals_enabled": proposals_enabled_for_cell(cell),
        "family_axis": FAMILY_AXIS_NOT_APPLICABLE,
    }


def paid_api_gate(proposer, allow_paid_api):
    """The same explicit authorization gate as the other benchmark runners."""
    if proposer == "anthropic" and not allow_paid_api:
        raise ValueError("Anthropic proposals require explicit --allow-paid-api authorization")
    if proposer in ("openai", "openai_compat") and not allow_paid_api:
        raise ValueError("OpenAI proposals require explicit --allow-paid-api authorization")


def neuron_parameter_windows():
    """Declare the envelope of public recipes, independent of the world.

    These bounds constrain proposed shapes, not the fitted conductance prior.
    Zero slopes and nonpositive time constants are separately refused by the
    transport; gate powers use its fixed {1, 2} menu.
    """
    recipes = mechanism_vocabulary().values()
    return {
        name: [min(values), max(values)]
        for name in CONTINUOUS_PARAMETERS
        if (values := [recipe["parameters"][name] for recipe in recipes
                       if recipe["parameters"].get(name) is not None])
    }


def _neuron_archive_entry(hypothesis, proposed_round, outcome, reason):
    """Build one masked archive.json entry for a proposed hypothesis and its outcome."""
    recipe = mechanism_vocabulary()[hypothesis.mechanism]
    return mask_secrets_deep({
        "model_id": hypothesis.model_id,
        "mechanism": hypothesis.mechanism,
        "kind": recipe["kind"],
        "parameters": getattr(hypothesis, "parameters", recipe["parameters"]),
        "proposed_round": proposed_round,
        "outcome": outcome,
        "reason": reason,
    })


def neuronbench_provenance(nb):
    """PYTHONPATH + commit disclosure: `neuronbench` is
    deliberately not installed (absent from pyproject.toml/requirements.lock);
    the commit actually imported is read from git, not hardcoded."""
    import subprocess

    root = Path(nb.__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort, never fatal
        commit = f"unknown ({exc!r})"
    return {"neuronbench_source_path": str(root), "neuronbench_commit": commit}


# An earlier version defined a LOCAL `points_label`
# here as a placeholder, with a note that "a later composition step should
# switch NeuronBench to call the shared one instead of this local copy".
# That composition step is done: the shared
# `core_count.points_label` is now imported above and
# the local copy is gone. The rule, the argument names and the accepted
# values are identical - neuron_core's `--profile` choices (then ("dev",
# "paper"), now also "paper_neuron") and its `--proposer` choices (then
# ("mock", "anthropic", "openai"), now also "openai_compat") are exactly the
# two sets the shared helper validates against - so
# `points_row` and test_write_points_label_follows_profile's truth table
# are unchanged by the switch.


def points_row(config, code_sha, row, line):
    """One points.csv row from one scores.jsonl row (boxinggym_core.py's own
    write_points pattern: suite/task/method/... then source_path citing the
    exact scores.jsonl line). nll/coverage90/width90/crps/is_0.1/kl/brier/
    nmsle/rmsle are undefined for this suite's spike_forecast_mse metric and
    stay empty strings (never 0). `method` is always the
    recorded `--cell` value, matching the run it was actually produced by
    (recorded cell == executed cell, now true for the acquisition
    and proposal axes both - see resolve_acquisition/proposals_enabled_for_cell)."""
    cell = config["cell"]
    proposer = config["requested_proposer"]
    return {
        "suite": "neuronbench",
        "task": config["world"],
        "method": cell,
        "proposer": "mock" if proposer == "mock" else f"llm:{proposer}",
        "seed": config["seed"],
        "x": row["training_observations"],
        "x_unit": "protocol_runs",
        "y_primary": row["bma_native_spike_forecast_mse"],
        "y_name": "spike_forecast_mse",
        "y_label": "native spike-forecast MSE (BMA, floored)",
        "nll": "",
        "coverage90": "",
        "width90": "",
        "crps": "",
        "is_0.1": "",
        "kl": "",
        "status": "completed",
        # The label the RUN resolved (--run-label, recorded in
        # config) wins; `points_label` stays the fallback for a config
        # without one and for hand-built test configs.
        # the fallback asks the shared helper about the profile the label POLICY knows this
        # run as (see LABEL_POLICY_PROFILE); `dev` and `paper` map to themselves, so this row is
        # unchanged for them. An unknown profile still reaches the helper's own refusal.
        "label": config.get("label") or points_label(
            LABEL_POLICY_PROFILE.get(config["profile"], config["profile"]), proposer
        ),
        "run_id": Path(config["output"]).name,
        "source_path": (
            f"{config['output']}/scores.jsonl#L{line}"
            f"(round={row['round']},phase={row['phase']})"
        ),
        # The "acquisition_axis_absent_stage1" token of earlier versions is removed
        # outright (not renamed) - NeuronBench never has a variant token
        # (the shared rule: default-valued axes carry no token; this suite has no
        # axis a non-default token could ever describe). The two opt-in
        # EIG-part arms are the one exception, so they never pool with joint.
        "variant": EIG_PART_VARIANT.get(config.get("design_policy"), ""),
        "code_sha": code_sha,
        "brier": "",
        "nmsle": "",
        "rmsle": "",
    }


def write_points(path, config, rows):
    """The plotting contract's order, one row per scores.jsonl row."""
    import csv

    code_sha = config["experimental_source_sha256"]["neuron_core.py"]
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(*POINTS_COLUMNS, *compat_points_metadata(config)), lineterminator="\n")
        writer.writeheader()
        for line, row in enumerate(rows, start=1):
            writer.writerow({**points_row(config, code_sha, row, line), **compat_points_metadata(config)})


class _PriorProbeForwardOp(ForwardOp):
    """Preserve native finite values and expose numerical failures to the probe.

    The sampling operation rejects both malformed and nonfinite outputs. Here
    only a correctly shaped nonfinite result becomes a NaN diagnostic, so the
    prior probe can count and reject it. Shape and forward execution errors
    still propagate; this operation is never used for fitting or prediction.
    """

    def perform(self, node, inputs, outputs):
        """Run the forward at one conductance; write the feature matrix, or NaNs if any is nonfinite."""
        conductance = float(inputs[0])
        if not np.isfinite(conductance) or conductance <= 0:
            raise ValueError("invalid transformed conductance")
        value = np.asarray(
            [self.forward(self.hypothesis, [conductance], d) for d in self.designs],
            dtype=np.float64,
        )
        if value.shape != self.shape:
            raise ValueError("invalid forward shape or non-finite output")
        outputs[0][0] = value if np.all(np.isfinite(value)) else np.full(self.shape, np.nan)


def _neuron_probe_model(hypothesis, designs, observations, forward, scale=20.0,
                        feature_scales=None):
    """Native vector likelihood on finite outputs, with probe-only NaN diagnostics."""
    import pymc as pm

    y = np.asarray(observations, dtype=float)
    if y.ndim != 2 or len(y) != len(designs) or not np.all(np.isfinite(y)) or scale <= 0:
        raise ValueError("finite [observation, feature] matrix and positive unit required")
    means = _PriorProbeForwardOp(hypothesis, designs, forward, y.shape)
    with pm.Model() as model:
        conductance = pm.LogNormal("conductance", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        # The same noise the fit uses, so the probe measures the fit's likelihood (see
        # `vector_inference.fit`): one mV unit, or the per-feature scales of the six-feature
        # space.
        pm.Normal(
            "observed", mu=means(conductance), observed=y,
            sigma=scale * sigma if feature_scales is None
            else np.asarray(feature_scales, dtype=float)[None, :] * sigma,
        )
    return model


def _neuron_loglik_probe(hypothesis, designs, observations, forward=None, feature_scales=None):
    """Run the prior log-likelihood probe for one hypothesis; return the probe record with its seed."""
    # Resolved here rather than as a default argument: the module-level name is what a caller
    # replaces to probe a misbehaving forward, and a default bound at definition time would
    # silently keep the original.
    forward = predict_features if forward is None else forward
    seed = seed_for("neuron_prior_probe", hypothesis.model_id)
    rng = np.random.default_rng(seed)
    # The native vector graph uses PyMC's default log transforms. Evaluate
    # its exact value-variable graph at the same physical prior samples.
    draws = {"conductance_log__": np.log(rng.lognormal(0, 1, PRIOR_PROBE_SAMPLES)),
             "sigma_log__": np.log(np.abs(rng.normal(0, 1, PRIOR_PROBE_SAMPLES)))}
    probe = loglik_prior_probe(
        hypothesis, designs, observations, count=PRIOR_PROBE_SAMPLES, seed=seed,
        prior_draws=draws,
        model_builder=lambda: _neuron_probe_model(hypothesis, designs, observations, forward,
                                                  feature_scales=feature_scales),
    )
    return {**probe, "seed": seed, "seed_stream": "neuron_prior_probe"}


def _refuse_before_output(message):
    """Exit code 2, before any output exists, with the reason on stderr."""
    print(f"neuron_core: {message}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def _refuse_forward_backend(message):
    """Refuse a `--forward-backend` request (exit code 2, before any output)."""
    _refuse_before_output(message)


def forward_backend_receipt(args):
    """Resolve `--forward-backend` and everything config.json must record about it.

    The batch forward is bit-identical to the native one only in an interpreter where
    numpy's SIMD `power` dispatch is off, and only as evidenced by an equivalence report
    generated on THIS machine, so a batch request is admitted only when: the interpreter passes
    the same self-check the report's machine passed; a report was named, exists, was made
    against the installed benchmark version, and records a zero maximum deviation with matching
    spike counts. An earlier extra refusal of `--run-label production` is gone: label policy v3
    registers `forward_backend` and, for the batch forward, `forward_equivalence_sha`, so a
    production batch run is admitted exactly when its registration states the backend and the
    digest of the report that admitted it. Every refusal is exit code 2 and happens before
    `out.mkdir`, so an inadmissible request leaves nothing behind.

    Returns the config keys that depend on the resolved backend: none for `native` (whose
    config.json keeps its exact earlier key set apart from the three keys `run` records for
    every backend - the backend itself and the two numpy dispatch facts), and the report's
    digest and summary for `batch`.
    """
    backend = getattr(args, "forward_backend", "native")
    if backend not in FORWARD_BACKENDS:
        raise ValueError("--forward-backend must be native or batch")
    path = getattr(args, "forward_equivalence_report", None)
    if backend == "native":
        if path is not None:
            _refuse_forward_backend(
                "--forward-equivalence-report applies only to --forward-backend batch"
            )
        return {}
    if not power_matches_scalar():
        _refuse_forward_backend(
            "--forward-backend batch needs numpy's SIMD power dispatch disabled; set "
            'NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX '
            'AVX512_CLX AVX512_CNL AVX512_ICL" and generate the equivalence report in the '
            "same environment"
        )
    if path is None:
        _refuse_forward_backend(
            "--forward-backend batch requires --forward-equivalence-report PATH"
        )
    report_path = Path(path)
    if not report_path.is_file():
        _refuse_forward_backend(f"equivalence report not found: {path}")
    report = json.loads(report_path.read_text())
    import neuronbench as nb

    if report.get("neuronbench_version") != nb.__version__:
        _refuse_forward_backend(
            f"equivalence report was generated against neuronbench "
            f"{report.get('neuronbench_version')!r}, not the installed {nb.__version__!r}"
        )
    if float(report.get("max_abs_delta_mV", 1.0)) > 0:
        _refuse_forward_backend(
            f"equivalence report records a non-zero maximum deviation "
            f"({report.get('max_abs_delta_mV')} mV): the batch forward is not bit-identical "
            f"on this machine"
        )
    if report.get("spike_counts_equal") is not True:
        _refuse_forward_backend("equivalence report records disagreeing spike counts")
    return {
        "forward_equivalence_sha": sha256_file(report_path),
        "forward_equivalence_summary": {
            key: report.get(key)
            for key in ("pairs", "bit_identical", "max_abs_delta_mV", "machine")
        },
    }


def run(args, *, client=None):
    """Run one deterministic NeuronBench cell from parsed arguments and write its run directory.

    `client` optionally injects an LLM client into the proposer. Writes config.json, the
    observation/fit/prediction records, scores.jsonl, points.csv and summary.json under `--output`.
    """
    fit_timeout_seconds = validate_fit_timeout(getattr(args, "fit_timeout_seconds", None))
    fit_guards = getattr(args, "fit_guards", "off")
    if fit_guards not in ("off", "v1"):
        raise ValueError("fit_guards must be off or v1")
    label_metadata = compat_config(args)
    require_wired_world(args.world)
    paid_api_gate(args.requested_proposer, args.allow_paid_api)
    # Batch forward: resolved HERE, with the other before-any-output gates, so an inadmissible batch
    # request never reaches `out.mkdir` (and a production one never reaches the label receipt).
    forward_backend = getattr(args, "forward_backend", "native")
    forward_receipt = forward_backend_receipt(args)
    # --run-label. Resolved HERE,
    # before any output directory exists, so an inadmissible label leaves
    # nothing behind. `args.run_label` itself is NOT overwritten: it stays the
    # caller's raw request (None when the flag was not typed), because the
    # byte-invariance rule keys off "was a label actually stated"
    # and because `**vars(args)` should record what was asked for.
    stated_run_label = getattr(args, "run_label", None)
    # Refuse a --cell/--acquisition mismatch before any output
    # (require_wired_world/paid_api_gate above are the only earlier gates).
    args.acquisition = resolve_acquisition(args.cell, args.acquisition)
    # EIG-part arms (added 2026-10-05): `--design-policy structure_eig|parameter_eig` replaces the joint
    # EIG of an A=1 cell by one of its two chain-rule parts (same bank, same draws, argmax of
    # that part). The option is SUPPRESS, so an unstated run keeps every record as before.
    design_policy = getattr(args, "design_policy", None)
    if design_policy is not None and args.acquisition != "joint":
        raise ValueError(f"design policy {design_policy} needs an A=1 cell (got {args.cell})")
    # Candidates per call (1..3; 1 = today's transport) and the
    # per-round design criteria record. Both SUPPRESS: unstated, every record keeps its bytes.
    proposal_limit = neuron_validate_proposal_limit(int(getattr(args, "proposal_limit", 1)))
    record_criteria = bool(getattr(args, "record_design_criteria", False))
    args.particles = resolve_particles(args.profile, args.particles)
    # the active-pool bound this run truncates with, chosen once, here, from the profile.
    # The config record, the truncation below and the registered declaration all read it.
    active_maximum = ACTIVE_MAXIMUM_BY_PROFILE[args.profile]
    # canonicalise the cell here, right after the other resolvers and
    # still before any output exists, capturing the raw request first in the
    # same "state it, then resolve it" shape as `stated_run_label` above.
    # Everything downstream - `cell_axis_config`, `**vars(args)` in `config`,
    # and `points_row`'s `method` column, which reads `config["cell"]`
    # verbatim - therefore sees the executed cell and needs no change.
    # `resolve_acquisition` is deliberately called on the raw value: it reads
    # `cell[2]`, which canonicalisation never touches, so its refusal names
    # the cell the caller actually typed.
    stated_cell = args.cell
    args.cell = canonical_cell(args.cell)
    # the receipt is resolved against the profile the POLICY knows this run as
    # (LABEL_POLICY_PROFILE), which is every profile's own name. The two values
    # the policy's neuron declaration needs but the command line does not carry are added
    # here: the active-pool bound this run truncates with, and - for the batch forward - the
    # digest of the equivalence report that admitted it. `args` itself is never rewritten;
    # `**vars(args)` still records exactly what was asked for and executed.
    label_extras = {}
    if args.profile == PAPER_NEURON:
        label_extras["active_maximum"] = active_maximum
    if forward_backend == "batch":
        label_extras["forward_equivalence_sha"] = forward_receipt["forward_equivalence_sha"]
    label_args = args
    if LABEL_POLICY_PROFILE[args.profile] != args.profile or label_extras:
        label_args = argparse.Namespace(
            **{**vars(args), "profile": LABEL_POLICY_PROFILE[args.profile], **label_extras}
        )
    label_receipt = bmd_label_receipt("neuron", label_args, task=args.world,
                                      metadata=label_metadata)
    run_label = label_receipt["run_label"]
    # A production run states a decoding option only if the manifest
    # registered the same value - here, before any output exists.
    require_registered_decoding(label_metadata, label_receipt, "neuron")
    label_explicit = stated_run_label is not None or getattr(args, "manifest_ref", None) is not None

    import neuronbench as nb
    from neuronbench.worlds import build_I

    started = time.perf_counter()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("observations", "fits", "predictions"):
        (out / name).mkdir()
    settings = SMCSettings(args.particles, 2, 2)
    # the training feature space, and the kernel that goes with it. Both are decided once,
    # here, and every fit, bank and observation below reads them rather than naming a function.
    space = feature_space(args.profile)
    # the same decision for the proposal path - which feature vector the belief packet
    # carries, checks and probes candidates on.
    packet_space = proposer_feature_space(args.profile)
    bank_scales = {} if space["scales"] is None else {"feature_scales": space["scales"]}
    fit_options = dict(bank_scales)
    if args.profile == PAPER_NEURON:
        fit_options["kernel_policy"] = "mh_fixed_steps"
    axis_config = cell_axis_config(args.cell)
    proposer, archive = None, []
    known_ids = {h.model_id for h in initial_hypotheses()}
    try:
        world = nb.load_world(args.world, stochastic=False, seed=args.seed)
        pool = world.protocol_pool
        evaluation = world.test_protocols
        if axis_config["proposals_enabled"] and args.requested_proposer != "mock":
            proposer = make_neuron_proposer(
                args.requested_proposer, out / "llm_transport", world=args.world,
                cell=args.cell, allow_live=args.allow_paid_api,
                model=getattr(args, "llm_model", None), client=client,
                max_rounds=args.rounds, known_ids=known_ids, archive=archive,
                feature_space=packet_space,
                # Passed only when typed, so an unstated option builds the same proposer.
                **({"accept_code_fence": True} if getattr(args, "llm_accept_code_fence", False) is True else {}),
                # Likewise, k = 1 builds the same proposer.
                **({"proposal_limit": proposal_limit} if proposal_limit != 1 else {}),
                **compat_option_kwargs(args, compat=args.requested_proposer == "openai_compat"),
                **({"base_url": getattr(args, "llm_base_url", None),
                    "key_name": getattr(args, "llm_key_name", "OPENAI_COMPAT_API_KEY")}
                   if args.requested_proposer == "openai_compat" else {}),
            )
    except BaseException as exc:  # noqa: BLE001 - record constructor failures too
        write_json(out / "failure.json", {
            "status": "failed", "stage": "world_or_proposer_construction",
            "error": mask_secrets(f"{type(exc).__name__}: {exc}"),
            "traceback": mask_secrets(traceback.format_exc()),
            "timing_seconds": {}, "paid_api_calls": 0,
        })
        raise masked_exception(exc) from None
    config = {
        **compat_config(args),
        "output": str(out),
        "smc": asdict(settings),
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.executable,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pymc", "pytensor")
        },
        "neuronbench_version": nb.__version__,
        "experimental_source_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in (
                "neuron_core.py",
                "neuron_forward.py",
                "vector_inference.py",
                "core_count.py",
                "count_inference.py",
                "count_models.py",
            )
        },
        "deterministic_development_only": True,
        "stochastic_benchmark_complete": False,
        "paid_api_calls": 0,
        "registered_run": False,
        "initial_pool": 4,
        "structure_proposals": structure_mode(args.cell),
        # the SAME object the pool truncation below is called with
        # (`retain(fits + candidates, active_maximum)`), not a literal that
        # merely happens to equal `retain`'s own default - a recorded cap and
        # an operating cap that can drift apart is exactly what this binding
        # removes. The value is the profile's own: `dev` and `paper` keep
        # core_count.ACTIVE_MAXIMUM == 12, so their config.json stays
        # byte-identical, and `paper_neuron` records the six it truncates at.
        # On dev/paper the cap is still unreachable at any --rounds this CLI
        # accepts (initial pool 4 + at most 3 admitted per event, mock
        # proposals repeating after round 1): it is a bound, not an active
        # truncation. On `paper_neuron` it can bind from the second proposal
        # event on, and the pool it keeps is the evidence-ordered one.
        "active_maximum": active_maximum,
        "proposal_rounds": [1, 2, 3, 4],
        # Keep the established mock/catalogue description on the old paths.
        "proposer": ("fixed_mock_generic_channel_catalogue_not_LLM" if proposer is None
                     else "public_neuron_mechanism_transport"),
        "mean_prior": "conductance_multiplier LogNormal(0,1), fixed channel shapes",
        **feature_space_config(args.profile),
        **(paper_neuron_config() if args.profile == PAPER_NEURON else {}),
        # recorded only when the proposal path is not on the default three-feature
        # descriptor, so a `dev` or `paper` config.json keeps its exact key set.
        **({} if packet_space == DEFAULT_FEATURE_SPACE else {"proposer_feature_space": {
            "record_key": packet_space.record_key, "names": list(packet_space.names),
            "scales": list(packet_space.feature_scales)}}),
        "q_protocol_labels": [label for label, _ in evaluation],
        "q_role": "heldout_native_forecast_score_only",
        "EIG_predictive_samples": 64,
        **({"EPIG_predictive_samples": EPIG_PREDICTIVE_SAMPLES,
            "epig_targets": "world.test_protocols (public evaluation input measure)"}
           if args.acquisition == "epig" else {}),
        "forward": (
            "native neuronbench.worlds.simulate dt=0.01 ms without truth kwargs"
            if forward_backend == "native" else
            "batch numpy replica of neuronbench.worlds.simulate dt=0.01 ms; "
            "equivalence report bound"
        ),
        # Batch forward: recorded for EVERY backend - the interpreter's SIMD dispatch
        # decides the bits of the native path and of the benchmark's own truth generation too;
        # a run whose environment is unknown cannot be compared with one whose is known.
        "numpy_cpu_features_disabled": numpy_cpu_features_disabled(),
        "numpy_cpu_features_enabled": numpy_cpu_features_enabled(),
        # Added with the NeuronBench runner's first stage:
        "world_wiring": world_wiring(),
        "baseline_overlap_x": [2, 4],
        "baseline_ps017_note": (
            "PS-017 x in {1,2,4,8} are independent episodes (run_baseline(budget=b)), "
            "no D0, seed 0 only; this runner's x is a cumulative path with a 2-protocol "
            "D0. Overlapping x values {2,4} differ in information content, not just label."
        ),
        **neuronbench_provenance(nb),
        **axis_config,
    }
    if forward_backend == "batch":
        # The batch module's own digest joins the six the runner already binds, and the report
        # that licenses this backend is bound by digest, not by path.
        config["experimental_source_sha256"]["neuron_forward_batch.py"] = hashlib.sha256(
            (Path(__file__).parent / "neuron_forward_batch.py").read_bytes()
        ).hexdigest()
        config.update(forward_receipt)
        if fit_guards != "off":
            # The fit-guard prior probe keeps the native per-particle forward: it is a
            # correctness diagnostic on a handful of prior draws, not a hot loop.
            config["forward_backend_guard_probe"] = "native"
    if proposer is not None:
        config.update(
            llm_resolved_model=proposer.model,
            llm_client_injected=proposer.injected_client,
            neuron_parameter_windows=neuron_parameter_windows(),
        )
        for name in ("neuron_proposer.py", "count_proposer.py", "belief_packet.py"):
            config["experimental_source_sha256"][name] = hashlib.sha256(
                (Path(__file__).parent / name).read_bytes()
            ).hexdigest()
    if label_explicit:
        # The resolved --run-label, recorded ONLY
        # when one was actually stated - the same "never unconditionally" rule
        # the other three runners follow, so a run that says nothing
        # keeps its exact earlier config.json key set. `points_row` reads it back
        # and falls through to `points_label` when it is absent.
        config["label"] = run_label
    if stated_cell != args.cell:
        # the raw `--cell` request, recorded ONLY when canonicalisation
        # actually moved it - the same "never unconditionally" rule as
        # `label` above, so a run at one of the four pre-existing cells keeps
        # its exact earlier config.json key set, byte for byte. `cell` itself
        # is always the canonical (executed) value, via `**vars(args)`.
        config["cell_requested"] = stated_cell
    if label_explicit:
        config["label_receipt"] = label_receipt
    if fit_timeout_seconds is None:
        config.pop("fit_timeout_seconds", None)
    else:
        config["fit_timeout_seconds"] = fit_timeout_seconds
    if fit_guards == "off":
        config.pop("fit_guards", None)
    else:
        config["fit_guards"] = fit_guards
    if design_policy is not None:
        # EIG-part arms: `design_policy` itself is already in config via `**vars(args)`.
        config["design_policy_definition"] = EIG_PART_DEFINITION[design_policy]
        config["design_policy_estimator"] = EIG_PART_ESTIMATOR
    if proposal_limit != 1:
        # `proposal_limit` itself is in config via `**vars(args)`; the composed system
        # text's identity beside it (the transport records carry the same pair per round).
        config.update({key: value for key, value in neuron_composed_system_prompt(proposal_limit)[1].items()
                       if key != "proposal_limit"})
        config["mock_proposals_per_round"] = 3  # the fixed mock schedule offers three; `--proposal-limit` is the LLM call's
    if record_criteria:
        # Stated only when the option is on.
        config["design_criteria_record"] = {"where": "acquisition.jsonl rows, key design_criteria",
                                            "EIG_predictive_samples": 64, **DESIGN_CRITERIA_DEFINITIONS}
    write_json(out / "config.json", mask_secrets_deep(config))
    if not axis_config["proposals_enabled"]:
        # No proposal round will ever run this call (see the gate below),
        # so nothing would otherwise call append_json on this path; leave it
        # as an explicit empty file rather than never creating it.
        (out / "proposals.jsonl").touch()
    designs, values, seen, score_rows = [], [], set(), []
    timing = {
        "fit": 0.0,
        "proposal": 0.0,
        "eig": 0.0,
        "forward": 0.0,
        "scoring": 0.0,
        "observation": 0.0,
    }
    failures, admitted = [], []
    timed_out_ids = set()
    guard_failed_ids = set()
    public_observations, public_acquisition = [], []
    born_round, active_rounds = {}, {}

    def observe(index, t, prequential_bank=None):
        """Run pool protocol `index` on the world at round `t`, record its features and add it to the data."""
        if index in seen:
            raise ValueError("deterministic benchmark forbids repeating a protocol")
        tic = time.perf_counter()
        observation = world.run(pool[index])
        timing["observation"] += time.perf_counter() - tic
        # The SAME function a candidate's predicted trace goes through (see `feature_space`),
        # applied to the trace the benchmark hands out. The record key follows the space: the
        # three physical features are all millivolts, the six evaluation features are not.
        y = space["observe"](observation.voltage, observation.test_start)
        record = {
            "round": t,
            "design_index": index,
            "label": pool[index][0],
            space["record_key"]: y.tolist(),
            "spike_count": observation.spike_count,
            "cost": observation.cost,
            "test_start_observed": observation.test_start,
        }
        if prequential_bank is not None:
            record["prequential_feature_log_predictive"] = float(
                log_predictive(prequential_bank, y[None, :])[0]
            )
        append_json(out / "observations.jsonl", record)
        public_observations.append(record)
        np.savez_compressed(
            out / "observations" / f"d{index:02d}.npz",
            voltage=observation.voltage,
            obs_idx=observation.obs_idx,
            features=y,
        )
        seen.add(index)
        designs.append(pool[index])
        values.append(y)

    def refit(h, t):
        """Fit hypothesis `h` on all observations at round `t`; save and return the fit result."""
        print(f"FIT START t={t} model={h.model_id}", flush=True)
        fit_started = time.monotonic() if fit_timeout_seconds is not None else None
        fit_status = "completed"
        try:
            fit_callable = fit
            if fit_guards != "off":
                from functools import partial

                fit_callable = partial(run_with_fit_guards, fit, fit_guards=fit_guards)
            if fit_timeout_seconds is not None:
                from functools import partial
                fit_callable = partial(run_with_fit_timeout, fit_callable,
                                       fit_timeout_seconds=fit_timeout_seconds)
            result = fit_callable(
                h,
                designs,
                values,
                space["forward"],
                seed_for("neuron_smc", args.seed, t, h.model_id),
                settings,
                forward_backend=forward_backend,
                **fit_options,
            )
        except FitTimeout:
            fit_status = "fit_timeout"
            timed_out_ids.add(h.model_id)
            known_ids.add(h.model_id)
            raise
        except CountInferenceFailure as exc:
            fit_status = "fit_failed"
            if fit_guards != "off" and exc.cause_code == "nonfinite_smc_weights":
                guard_failed_ids.add(h.model_id)
                known_ids.add(h.model_id)
            raise
        except BaseException:
            fit_status = "fit_failed"
            raise
        finally:
            if fit_started is not None:
                record_fit_timing(out, h, t, time.monotonic() - fit_started, fit_status)
        timing["fit"] += result.seconds
        np.savez_compressed(out / "fits" / f"t{t:02d}_{h.model_id}.npz", **result.particles)
        write_json(
            out / "fits" / f"t{t:02d}_{h.model_id}.json",
            {
                "model_id": h.model_id,
                "mechanism": h.mechanism,
                "log_evidence": result.log_evidence,
                "per_chain_log_evidence": result.per_chain_log_evidence,
                "diagnostics": result.diagnostics,
                "fit_seconds": result.seconds,
            },
        )
        print(f"FIT DONE t={t} model={h.model_id} logZ={result.log_evidence:.6f}", flush=True)
        return result

    def record_timeout(h, t, exc):
        """Record a fit timeout of `h` at round `t` in the failures and the archive; return the failure item."""
        item = {"model_id": h.model_id, "round": t,
                "error": mask_secrets(repr(exc)), **exc.as_record()}
        failures.append(item)
        entry = _neuron_archive_entry(h, t, "fit_failed", {"error": item["error"]})
        active_entry = next((row for row in archive
                             if row["model_id"] == h.model_id and row["outcome"] == "active"), None)
        if active_entry is None:
            archive.append(entry)
        else:
            active_entry.update(entry)
        return item

    def record_guard_failure(h, t, exc):
        """Record a fit-guard failure of `h` at round `t` in the failures and the archive; return the item."""
        item = {"model_id": h.model_id, "round": t, "status": "fit_failed", "retirement_kind": exc.cause_code,
                "error": mask_secrets(repr(exc)), **exc.as_record()}
        failures.append(item)
        entry = _neuron_archive_entry(h, t, "fit_failed", {
            "error": item["error"], "retirement_kind": exc.cause_code,
        })
        active_entry = next((row for row in archive
                             if row["model_id"] == h.model_id and row["outcome"] == "active"), None)
        if active_entry is None:
            archive.append(entry)
        else:
            active_entry.update(entry)
        return item

    def refit_pool(hypotheses, t):
        """Refit every hypothesis still eligible, retiring timeouts and guard failures; return the survivors."""
        survivors = []
        for h in hypotheses:
            if h.model_id in timed_out_ids or h.model_id in guard_failed_ids:
                continue
            try:
                survivors.append(refit(h, t))
            except FitTimeout as exc:
                record_timeout(h, t, exc)
            except CountInferenceFailure as exc:
                if fit_guards == "off" or exc.cause_code != "nonfinite_smc_weights":
                    raise
                record_guard_failure(h, t, exc)
        if not survivors:
            if fit_guards != "off":
                raise RuntimeError(f"active pool empty at round {t} after guarded fit failures")
            raise RuntimeError(f"active pool empty at round {t} after fit timeouts")
        return survivors

    def checkpoint(fits, t, phase):
        """Predict the held-out test protocols from `fits`, save the predictions and append a scores.jsonl row."""
        tic = time.perf_counter()
        weights = np.exp(model_logweights(fits))
        best = min(range(len(fits)), key=lambda i: (-weights[i], fits[i].hypothesis.model_id))
        arrays = {}
        n_feature = len(space["scales"]) if space["scales"] is not None else 3
        model_features = [
            np.empty((len(f.particles["conductance"]), len(evaluation), n_feature)) for f in fits
        ]
        mean_counts, single_counts, mean_traces = {}, {}, {}
        for d, (label, segments) in enumerate(evaluation):
            _, ts = build_I(segments)
            traces, counts = [], []
            for m, result in enumerate(fits):
                # Batch forward: one vectorised integration of the whole particle cloud
                # when the batch backend is on; the rows below read the same block either
                # way, value for value.
                bank = (
                    predict_batch(result.hypothesis, result.particles["conductance"],
                                  (label, segments))
                    if forward_backend == "batch" else
                    np.asarray(
                        [
                            result.hypothesis.predict([c], (label, segments))
                            for c in result.particles["conductance"]
                        ]
                    )
                )
                traces.append(bank.mean(axis=0))
                counts.append(float(np.mean(native_test_counts(bank, ts))))
                observed_start = int(np.searchsorted(np.arange(0, bank.shape[1], 10), ts))
                model_features[m][:, d] = [space["observe"](v[::10], observed_start)
                                           for v in bank]
            mean_counts[label] = float(weights @ counts)
            single_counts[label] = counts[best]
            mean_traces[label] = np.einsum("m,mt->t", weights, traces)
            arrays[f"q{d}_bma_noiseless_mean_trace"] = mean_traces[label]
            arrays[f"q{d}_single_noiseless_mean_trace"] = traces[best]
        features = {
            "means": np.concatenate(model_features),
            # One mV scale per particle, or one per feature when the six evaluation features
            # are the training space - the same noise `fit` gave those particles.
            "sigmas": np.concatenate([
                20 * f.particles["sigma"] if space["scales"] is None
                else np.asarray(space["scales"], dtype=float)[None, :]
                * f.particles["sigma"][:, None]
                for f in fits
            ]),
            "logweights": np.concatenate(
                [
                    np.full(len(f.particles["sigma"]), w - np.log(len(f.particles["sigma"])))
                    for f, w in zip(fits, model_logweights(fits), strict=True)
                ]
            ),
            "model_indices": np.concatenate(
                [np.full(len(f.particles["sigma"]), m) for m, f in enumerate(fits)]
            ),
        }
        arrays.update(features)
        arrays.update(predictive_summary(features))
        rng = np.random.default_rng(seed_for("neuron_predictive", args.seed, t, phase))
        indices = rng.choice(len(features["sigmas"]), 4, p=np.exp(features["logweights"]))
        arrays["feature_predictive_samples"] = features["means"][indices] + particle_sigmas(
            features["sigmas"][indices]
        ) * rng.normal(size=features["means"][indices].shape)
        timing["forward"] += time.perf_counter() - tic
        tic = time.perf_counter()
        # Only this scoring block invokes the benchmark's held-out truth evaluator.
        row = {
            "round": t,
            "phase": phase,
            "training_observations": len(values),
            "active_ids": [f.hypothesis.model_id for f in fits],
            "model_weights": weights.tolist(),
            "single_model_id": fits[best].hypothesis.model_id,
            "bma_test_count_predictions": mean_counts,
            "single_test_count_predictions": single_counts,
            "bma_native_spike_forecast_mse": float(world.spike_forecast_mse(mean_counts)),
            "single_native_spike_forecast_mse": float(world.spike_forecast_mse(single_counts)),
            "bma_native_feature_forecast_mse": float(world.feature_forecast_mse(mean_traces)),
        }
        timing["scoring"] += time.perf_counter() - tic
        np.savez_compressed(out / "predictions" / f"t{t:02d}_{phase}.npz", **arrays)
        append_json(out / "scores.jsonl", row)
        score_rows.append(row)
        print(f"SCORE t={t} phase={phase} pool={len(fits)}", flush=True)

    # The round in progress, for a summary the paid-call guard may have to write.
    current_round = 0
    try:
        observe(0, 0)
        observe(1, 0)
        fits = (refit_pool(initial_hypotheses(), 0) if fit_timeout_seconds is not None or fit_guards != "off"
                else [refit(h, 0) for h in initial_hypotheses()])
        checkpoint(fits, 0, "initial")
        for t in range(1, args.rounds + 1):
            current_round = t
            available = [i for i in range(len(pool)) if i not in seen]
            tic = time.perf_counter()
            print(f"CHOOSE START t={t}", flush=True)
            if args.acquisition == "joint":
                bank = prediction_bank(fits, [pool[i] for i in available], space["forward"],
                                       forward_backend=forward_backend, **bank_scales)
                local, rows = joint_eig(bank, seed_for("neuron_eig", args.seed, t),
                                        objective=design_policy or "eig")
                selected = available[local]
            elif args.acquisition == "epig":
                # the targets are the world's held-out TEST protocols - the public
                # evaluation input measure this run is scored on, not a sampled target set -
                # so EPIG asks which unobserved design tells us most about the outcomes the
                # score is made of. Both banks come from the same fits in the same particle
                # order, which is what lets `joint_epig` pair particle j on both sides.
                bank = prediction_bank(fits, [pool[i] for i in available], space["forward"],
                                       forward_backend=forward_backend, **bank_scales)
                target_bank = prediction_bank(fits, evaluation, space["forward"],
                                              forward_backend=forward_backend, **bank_scales)
                local, rows = joint_epig(bank, target_bank,
                                         seed_for("neuron_epig", args.seed, t),
                                         samples=EPIG_PREDICTIVE_SAMPLES)
                selected = available[local]
            else:
                selected = int(
                    np.random.default_rng(seed_for("neuron_random", args.seed, t)).choice(available)
                )
                rows = []
            criteria = None
            if record_criteria:
                # The three EIG values (one table, the joint arm's estimator on this
                # round's EIG seed; the arm's own rows when they already carry the parts) and the
                # point disagreement of every available design, as a record beside the decision.
                criteria_bank = bank if args.acquisition in ("joint", "epig") else prediction_bank(
                    fits, [pool[i] for i in available], space["forward"],
                    forward_backend=forward_backend, **bank_scales)
                eig_seed = seed_for("neuron_eig", args.seed, t)
                if rows and "structure_eig" in rows[0]:
                    table, source = rows, "arm_table"
                else:
                    table, source = joint_eig(criteria_bank, eig_seed, decompose=True)[1], "recomputed_joint_scheme"
                disagreement = vector_point_disagreement(criteria_bank)
                criteria = {
                    "eig_seed": eig_seed, "EIG_predictive_samples": 64, "eig_source": source,
                    "model_weights": [float(w) for w in disagreement["model_weights"]],
                    "rows": [
                        {"design_index": row["design_index"], "pool_index": available[row["design_index"]],
                         "eig": row["eig"], "structure_eig": row["structure_eig"],
                         "parameter_eig": row["parameter_eig"],
                         "point_disagreement": float(disagreement["point_disagreement"][row["design_index"]]),
                         "point_disagreement_per_feature": [
                             float(v) for v in disagreement["per_feature"][row["design_index"]]]}
                        for row in table
                    ],
                }
            prequential = prediction_bank(fits, [pool[selected]], space["forward"],
                                          forward_backend=forward_backend, **bank_scales)
            timing["eig"] += time.perf_counter() - tic
            append_json(
                out / "acquisition.jsonl",
                {
                    "round": t,
                    "selected": selected,
                    "available": available,
                    "mode": args.acquisition,
                    **({"design_policy": design_policy} if design_policy is not None else {}),
                    "rows": rows,
                    **({"design_criteria": criteria} if criteria is not None else {}),
                },
            )
            public_acquisition.append({
                "round": t, "selected": selected, "available": available,
                "mode": args.acquisition,
            })
            observe(selected, t, prequential)
            fits = (refit_pool([f.hypothesis for f in fits], t) if fit_timeout_seconds is not None or fit_guards != "off"
                    else [refit(f.hypothesis, t) for f in fits])
            if proposer is not None:
                for result in fits:
                    identifier = result.hypothesis.model_id
                    active_rounds.setdefault(identifier, [0]).append(t)
                for entry in archive:
                    if entry["outcome"] == "active":
                        entry["reason"]["active_rounds"] = list(active_rounds[entry["model_id"]])
            checkpoint(fits, t, "before_proposal")
            # cell[0] == "0" -> no proposal round, ever (earlier versions ran it
            # unconditionally on cell).
            if axis_config["proposals_enabled"] and t in (1, 2, 3, 4):
                tic = time.perf_counter()
                transport_record = None
                if proposer is None:
                    proposals = mock_proposals(t)
                else:
                    packet = build_neuron_belief_packet(
                        world, public_observations, fits, archive, args.cell,
                        parameter_windows=config["neuron_parameter_windows"],
                        acquisition=public_acquisition, feature_space=packet_space,
                    )
                    candidate, transport_record = proposer.propose_one(
                        serialize_packet(packet), context={
                            "run_id": str(out), "world_seed": args.seed,
                            "cell": args.cell, "proposal_rep": 0, "round": t,
                        },
                    )
                    # At k > 1 the transport returns the list of parsed candidates.
                    proposals = (list(candidate) if isinstance(candidate, list)
                                 else [] if candidate is None else [candidate])
                timing["proposal"] += time.perf_counter() - tic
                ids = {f.hypothesis.model_id for f in fits}
                candidates, events = [], []
                likelihood_probes = {}
                if transport_record is not None:
                    events.extend({"model_id": entry["model_id"], "status": entry["outcome"],
                                   "reason": entry["reason"]}
                                  for entry in transport_record["archive_entries"])
                for h in proposals[:3]:
                    if h.model_id in ids or h.model_id in timed_out_ids | guard_failed_ids:
                        events.append({"model_id": h.model_id, "status": "duplicate"})
                        continue
                    if fit_guards != "off":
                        probe = _neuron_loglik_probe(h, designs, values, space["forward"],
                                                     space["scales"])
                        likelihood_probes[h.model_id] = probe
                        if not probe["passed"]:
                            known_ids.add(h.model_id)
                            guard_failed_ids.add(h.model_id)
                            reason = {"validation_message": probe["reason"],
                                      "loglik_prior_probe": probe}
                            events.append({"model_id": h.model_id, "status": "rejected",
                                           "reason": reason})
                            archive.append(_neuron_archive_entry(h, t, "rejected", reason))
                            continue
                    try:
                        candidates.append(refit(h, t))
                    except FitTimeout as exc:
                        events.append(record_timeout(h, t, exc))
                    except CountInferenceFailure as exc:
                        if fit_guards != "off" and exc.cause_code == "nonfinite_smc_weights":
                            events.append(record_guard_failure(h, t, exc))
                        else:
                            item = {"model_id": h.model_id, "status": "failed",
                                    "error": mask_secrets(repr(exc))}
                            failures.append(item)
                            events.append(item)
                            if proposer is not None:
                                archive.append(_neuron_archive_entry(
                                    h, t, "fit_failed", {"error": item["error"]},
                                ))
                    except Exception as exc:  # noqa: BLE001 - preserve failed proposals
                        item = {"model_id": h.model_id, "status": "failed",
                                "error": mask_secrets(repr(exc))}
                        failures.append(item)
                        events.append(item)
                        if proposer is not None:
                            archive.append(_neuron_archive_entry(
                                h, t, "fit_failed", {"error": item["error"]},
                            ))
                    else:
                        events.append({"model_id": h.model_id, "status": "fitted"})
                        born_round[h.model_id] = t
                # The recorded cap is the same bound used by the mock path.
                fits, retired = retain(fits + candidates, active_maximum)
                new = sorted({f.hypothesis.model_id for f in fits} - ids)
                admitted.extend(new)
                if proposer is not None:
                    for result in candidates:
                        h = result.hypothesis
                        if h.model_id in new:
                            active_rounds[h.model_id] = [t]
                            archive.append(_neuron_archive_entry(h, t, "active", {
                                "admitted_round": t, "active_rounds": [t],
                            }))
                    for result in retired:
                        h = result.hypothesis
                        entry = _neuron_archive_entry(
                            h, born_round.get(h.model_id, 0), "retired_by_evidence", {
                                "log_evidence_at_retirement": float(result.log_evidence),
                                "gap_to_cutoff_nats": float(fits[-1].log_evidence-result.log_evidence),
                                "active_rounds": active_rounds.get(h.model_id, []),
                                "computed_round": t, "n_obs": len(values),
                            },
                        )
                        active_entry = next((row for row in archive
                                             if row["model_id"] == h.model_id
                                             and row["outcome"] == "active"), None)
                        if active_entry is None:
                            archive.append(entry)
                        else:
                            active_entry.update(entry)
                        known_ids.add(h.model_id)
                append_json(
                    out / "proposals.jsonl",
                    mask_secrets_deep({
                        "round": t,
                        "mock": proposer is None,
                        "events": events,
                        **({"loglik_prior_probes": likelihood_probes} if fit_guards != "off" else {}),
                        "admitted_ids": new,
                        "retired_ids": [r.hypothesis.model_id for r in retired],
                        **({"transport": transport_record} if transport_record is not None else {}),
                    }),
                )
                checkpoint(fits, t, "after_proposal")
        recheck_manifest(label_receipt)
        write_points(out / "points.csv", config, score_rows)
        own, children = (
            resource.getrusage(resource.RUSAGE_SELF),
            resource.getrusage(resource.RUSAGE_CHILDREN),
        )
        summary = {
            "status": "completed_deterministic_development",
            "rounds": args.rounds,
            "observations": len(values),
            "admitted": admitted,
            "active_count": len(fits),
            "candidate_failures": failures,
            "timing_seconds": timing,
            "wall_seconds": time.perf_counter() - started,
            "user_cpu_seconds": own.ru_utime + children.ru_utime,
            "system_cpu_seconds": own.ru_stime + children.ru_stime,
            # Historical cost-ledger field counts attempts, including injected
            # clients. The separate provider count below stays zero for fakes.
            "paid_api_calls": 0 if proposer is None else proposer.attempts,
            "stochastic_benchmark_complete": False,
        }
        if stated_cell != args.cell:
            # both halves of the alias, recorded together and under the
            # same "never unconditionally" rule as config.json's
            # `cell_requested` - a run at one of the four pre-existing cells
            # keeps its exact earlier summary.json key set, byte for byte, so
            # a reader of summary.json alone can still tell which cell was
            # requested and which one executed whenever the two differ.
            summary["cell"] = args.cell
            summary["cell_requested"] = stated_cell
        if proposer is not None:
            summary.update(provider_request_attempts=proposer.paid_api_calls,
                           llm_client_injected=proposer.injected_client)
            write_json(out / "archive.json", mask_secrets_deep(archive))
        elif timed_out_ids or guard_failed_ids:
            write_json(out / "archive.json", mask_secrets_deep(archive))
        if label_explicit:
            summary["label_receipt"] = label_receipt
        write_json(out / "summary.json", mask_secrets_deep(summary))
    except PaidCallBudgetExhausted:
        # The paid-call guard (as in the count runners) refused the call that would pass
        # max_paid_calls. Every record written so far stays; the summary says why and where,
        # and main() exits with the guard's status.
        write_json(out / "archive.json", mask_secrets_deep(archive))
        write_json(out / "summary.json", mask_secrets_deep({
            **({"label_receipt": label_receipt} if label_explicit else {}),
            **budget_exhausted_summary(proposer, current_round),
            "world": args.world,
            "cell": args.cell,
            "profile": args.profile,
            "rounds": args.rounds,
            "observations": len(values),
            "provider_request_attempts": proposer.paid_api_calls,
            "llm_client_injected": proposer.injected_client,
            "timing_seconds": timing,
            "wall_seconds": time.perf_counter() - started,
        }))
        raise
    except BaseException as exc:  # noqa: BLE001 - Record the masked failure before re-raising every exception.
        write_json(
            out / "failure.json", mask_secrets_deep({
                "traceback": traceback.format_exc(), "timing_seconds": timing,
                **({"candidate_failures": failures, "archive": archive} if timed_out_ids or guard_failed_ids else {}),
                **({"paid_api_calls": proposer.attempts,
                    "provider_request_attempts": proposer.paid_api_calls,
                    "llm_client_injected": proposer.injected_client, "archive": archive}
                   if proposer is not None else {}),
            })
        )
        raise masked_exception(exc) from None


def build_parser():
    """Return the argparse parser for the NeuronBench runner's command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--world", choices=tuple(WORLDS), default="h_sag")
    parser.add_argument("--list-worlds", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--fit-timeout", dest="fit_timeout_seconds",
                        type=validate_fit_timeout, default=argparse.SUPPRESS)
    parser.add_argument("--fit-guards", choices=("off", "v1"), default=argparse.SUPPRESS)
    parser.add_argument("--rounds", type=int, choices=range(1, 6), default=1)
    # Confirm-only now (default None); the real switch is
    # --cell[2] via resolve_acquisition, called at the top of run().
    parser.add_argument("--acquisition", choices=("random", "joint", "epig"), default=None)
    # EIG-part arms (added 2026-10-05): opt-in Model EIG / Parameter EIG arm on an A=1 cell; SUPPRESS.
    parser.add_argument("--design-policy", choices=EIG_PART_OBJECTIVES, default=argparse.SUPPRESS)
    # all eight 3-bit cells, plus their four EPIG counterparts with third digit
    # '2', are accepted now (was the four 000/001/110/111
    # only, which refused the family-bit-at-0 convention every other bundle's
    # F0 cells use with a bare argparse SystemExit rather than a considered
    # answer). The four added strings are aliases, not new behaviour: `run()`
    # maps them through `canonical_cell` and records both halves. Default
    # unchanged.
    parser.add_argument(
        "--cell",
        choices=("000", "001", "002", "010", "011", "012", "100", "101", "102", "110", "111", "112"),
        default="111",
    )
    parser.add_argument(
        "--proposer",
        choices=("mock", "anthropic", "openai", "openai_compat"),
        default="mock",
        dest="requested_proposer",
    )
    parser.add_argument("--allow-paid-api", action="store_true")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=argparse.SUPPRESS)
    parser.add_argument("--llm-key-name", default=argparse.SUPPRESS)
    parser.add_argument("--llm-provider-order", type=normalize_provider_order, default=argparse.SUPPRESS)
    # Both stay absent from the namespace unless typed, so an
    # omitted flag leaves the request body, the config and the points columns
    # byte-identical. `--llm-reasoning` goes to the transports that can send it
    # (openai_compat and anthropic; each validates it for its
    # provider); `--llm-max-tokens` is a decoding setting for
    # every provider. An invalid value ends the command line with status 2.
    parser.add_argument("--llm-reasoning", type=parse_reasoning_spec, default=argparse.SUPPRESS)
    parser.add_argument("--llm-max-tokens", type=parse_max_tokens, default=argparse.SUPPRESS)
    # The per-cell paid-call guard of the count runners. Absent unless
    # typed, so an omitted flag leaves every record byte-identical.
    parser.add_argument("--max-paid-calls", type=parse_max_paid_calls, default=argparse.SUPPRESS)
    # Accept a reply that is one Markdown code fence around the JSON
    # (neuron_proposer.CODE_FENCE_POLICY). Absent unless typed, so an omitted flag leaves the
    # parse, the transport records, config.json and the label receipt byte-identical; typed, it is
    # recorded in config.json, registered by the label policy and named in every transport record.
    parser.add_argument("--llm-accept-code-fence", action="store_true", default=argparse.SUPPRESS)
    # Candidates asked for per LLM call (1..3, checked in
    # `run`; the fixed mock schedule always offers three) and the per-round design criteria record.
    # Both SUPPRESS: unstated, config.json and every record keep their bytes.
    parser.add_argument("--proposal-limit", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--record-design-criteria", action="store_true", default=argparse.SUPPRESS)
    # `paper_neuron` is the MDA-comparison profile (200 particles, two chains, three fixed
    # MH moves per stage at ESS 0.5, the benchmark's six evaluation features). It is not a
    # variant of `paper` - the two differ in kernel and feature space - and the label policy
    # registers it under its own name, with its kernel and pool bindings.
    parser.add_argument("--profile", choices=("dev", "paper", PAPER_NEURON), default="dev")
    # Batch forward: the particle-vectorised forward. `native` is the default and changes
    # nothing; `batch` additionally requires the equivalence report generated on this machine,
    # in the interpreter this run uses (see `forward_backend_receipt`). The path is SUPPRESSed
    # rather than defaulted to None so a native run's config.json keeps its exact key set.
    parser.add_argument("--forward-backend", choices=FORWARD_BACKENDS, default="native")
    parser.add_argument("--forward-equivalence-report", default=argparse.SUPPRESS,
                        help="equivalence report JSON required by --forward-backend batch")
    # Rule v2: omitted labels stay development; production needs registration.
    parser.add_argument("--run-label", choices=RUN_LABEL_CHOICES, default=None)
    parser.add_argument("--manifest-ref", default=argparse.SUPPRESS,
                        help="JSON object containing manifest path and whole-file sha256")
    return parser


def main():
    """Command-line entry point: list the worlds, or run one cell and map a paid-call stop to its exit code."""
    parser = build_parser()
    args = parser.parse_args()
    if args.list_worlds:
        for name in WORLDS:
            print(f"{name}: {'wired' if name in WIRED_WORLDS else 'not wired'}")
        return
    if args.output is None:
        parser.error("--output is required unless --list-worlds is given")
    try:
        run(args)
    except PaidCallBudgetExhausted as exc:
        print(f"STOP {exc}; summary.json records stop_reason budget_exhausted", file=sys.stderr)
        raise SystemExit(PAID_CALL_BUDGET_EXIT_CODE) from None


if __name__ == "__main__":
    main()
