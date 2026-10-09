"""Projected simultaneous perturbation stochastic approximation (SPSA) on a unit box.

The unit box is search-only; the registered chooser is unchanged.
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from ..eig.joint import choose_design
from .core_count import seed_for

RESTARTS = 5
MIN_STEPS = 20
MAX_STEPS = 500
PARAMETER_DEFAULTS = {
    "a": 0.2, "A": 10, "alpha": 0.602, "gamma": 0.101,
    "c": 0.05, "W": 10, "kappa": 1.0, "k_sigma": 3,
}
NOISE_STREAM = "design_search_noise_floor"
HOLDOUT_STREAM = "design_search_holdout_final"
TRAJECTORY_LOCATION = "acquisition/t{round:02d}.json#/trajectories"
NOISE_FLOOR_METHODS = ("rms", "centered", "pairwise")
DEFAULT_NOISE_FLOOR_METHOD = "pairwise"
# Acquisition guard (a project decision of 2026-09-24, confirmed
# on 2026-09-25): an objective row may be an explicit invalid state instead of a number. The first four names are registered beside the gains in
# `design_search_constants`, so they enter `design_search_constants_sha256`.
ACQUISITION_GUARD = "particle_conditioning_v2"
FALLBACK_POLICY = "fixed_random_designs_from_native_input_distribution"
INVALID_REASONS = ("nonfinite_or_nonpositive_mean",)
ACQUISITION_STATUSES = ("ascent_with_exclusions", "all_designs_invalid", "acquisition_failed")
# Acquisition guard v2 (a project decision of 2026-09-26): guard v2 also keeps a lognormal particle only where, in log
# space, log_scale = log(mean) - 0.5 * sigma^2 satisfies REPRESENTABILITY_LOWER <= log_scale -
# LOGNORMAL_TAIL_SIGMA * sigma and log_scale + LOGNORMAL_TAIL_SIGMA * sigma <=
# REPRESENTABILITY_UPPER, so a draw is finite and positive and its density finite. The bounds
# are log(float64 tiny) and log(float64 max), written as literals so that no platform's log
# can move `design_search_constants_sha256`. All four are registered in
# `design_search_constants`; a different value is a change here and nowhere else, never after
# results.
LOGNORMAL_TAIL_SIGMA = 8
REPRESENTABILITY_LOWER = -708.3964185322641
REPRESENTABILITY_UPPER = 709.782712893384
SKIP_REASON = "invalid_perturbation_design"
NOISE_FLOOR_UNDETERMINED = "undetermined_invalid_check_point"
# EIG-part objectives (added 2026-10-06): the two chain-rule parts of the joint EIG an active A=1 cell may
# optimize instead of the joint EIG, I((M,theta);Y) = I(M;Y) + E_M I(theta;Y|M): objective column
# -> its Monte Carlo standard-error column. The ascent climbs the part itself; its noise floor,
# chooser and holdout read the same column (ForceBench fb1_design_search's rule for its parts).
EIG_PART_OBJECTIVES = {"structure_eig": "structure_mc_standard_error",
                       "parameter_eig": "parameter_mc_standard_error"}


def add_cli_arguments(parser):
    """Add the two design-search options to an argparse `parser`; both default to absent."""
    import argparse

    # Pre-registered decision of 2026-09-23: the gains, R, K_min and K_max are code
    # constants, so no option changes them at run time; only these two declarations exist.
    parser.add_argument("--design-search", choices=("pool", "ascent"), default=argparse.SUPPRESS)
    parser.add_argument("--design-search-noise-floor-method", choices=NOISE_FLOOR_METHODS,
                        default=argparse.SUPPRESS)


def design_search_constants():
    """The effective constants object, read from the module constants at call time.

    The 11 decided gains, then the four acquisition-guard declarations and the guard v2's
    three lognormal representability constants. All of them enter `constants_sha256` through
    the same canonical JSON; `validate_parameters` still validates only the gains.
    """
    return {**PARAMETER_DEFAULTS, "R": RESTARTS, "K_min": MIN_STEPS, "K_max": MAX_STEPS,
            **acquisition_guard_constants(), "fallback_policy": FALLBACK_POLICY,
            "invalid_reasons": list(INVALID_REASONS),
            "acquisition_statuses": list(ACQUISITION_STATUSES)}


def acquisition_guard_constants():
    """Acquisition guard v2: the guard's name and its three lognormal representability constants, the
    part of `design_search_constants` an acquisition exclusion record restates."""
    return {"acquisition_guard": ACQUISITION_GUARD, "lognormal_tail_sigma": LOGNORMAL_TAIL_SIGMA,
            "representability_lower": REPRESENTABILITY_LOWER,
            "representability_upper": REPRESENTABILITY_UPPER}


def constants_sha256(constants):
    """sha256 of the decided constants' canonical JSON: sort_keys and separators (",", ":")."""
    text = json.dumps(constants, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def declared_settings(args, acquisition_stream):
    """Pre-run declarations only; no trajectory values or fitted state are consulted."""
    mode = getattr(args, "design_search", "pool")
    if mode not in ("pool", "ascent"):
        raise ValueError("design-search must be pool or ascent")
    if mode == "pool":
        if hasattr(args, "design_search_noise_floor_method"):
            raise ValueError("design-search noise-floor method requires --design-search ascent")
        return {"candidate_source": "pool"} if hasattr(args, "design_search") else {}
    validate_parameters(dict(PARAMETER_DEFAULTS))
    constants = design_search_constants()
    noise_method = getattr(args, "design_search_noise_floor_method", DEFAULT_NOISE_FLOOR_METHOD)
    if noise_method not in NOISE_FLOOR_METHODS:
        raise ValueError("design-search noise-floor method is not declared")
    eig_part = args.design_policy in EIG_PART_OBJECTIVES
    if args.cell[2] == "0" or not (args.design_policy in ("joint_eig", "epig")
                                   or (eig_part and args.cell[2] == "1")):
        raise ValueError("design-search ascent requires an active joint_eig or epig cell "
                         "(or an A=1 cell climbing an EIG part: structure_eig, parameter_eig)")
    if getattr(args, "candidate_exclusion_policy", "none") != "none":
        raise ValueError("design-search ascent requires candidate_exclusion_policy none")
    from autoscilab.oracle.chembench import CHEM_INPUT_BOUNDS, CHEM_INPUT_VARS, CHEM_LOG_VARS

    transform = SearchTransform.from_bounds(CHEM_INPUT_VARS, CHEM_INPUT_BOUNDS, CHEM_LOG_VARS)
    if len(transform.input_names) > 2 and args.design_policy != "joint_eig" and not eig_part:
        raise ValueError("design-search dimension > 2 requires joint_eig (or one of its EIG parts)")
    rule = {"restarts": RESTARTS, "min_steps": MIN_STEPS, "max_steps": MAX_STEPS,
            "start_distribution": "uniform_unit_box",
            "clip": "perturbations_and_updates_to_unit_box",
            "chooser_candidates": "five_arrival_rows_seed_A_only",
            "candidate_exclusion_policy": "none"}
    if eig_part:
        # EIG-part objectives: recorded only for the two EIG-part arms, so a joint_eig/epig declaration
        # keeps its bytes. The search, the noise floor, the chooser and the holdout all read the
        # part (see projected_spsa's objective_key).
        rule.update(objective_column=args.design_policy,
                    objective_standard_error_column=EIG_PART_OBJECTIVES[args.design_policy])
    return {
        "design_search": "ascent", "candidate_source": "ascent_endpoints",
        "design_search_constants": constants,
        "design_search_constants_sha256": constants_sha256(constants),
        "design_search_transform": transform.record(),
        "noise_floor_method": noise_method,
        "design_search_seed_streams": {"A": acquisition_stream, "B": NOISE_STREAM, "C": HOLDOUT_STREAM},
        "design_search_seed_roles": {"A": "search_and_selection", "B": "selection_adjacent_noise_floor",
                                     "C": "independent_final_holdout_not_used_for_selection"},
        "design_search_trajectory": TRAJECTORY_LOCATION,
        "design_search_rule": rule,
    }


def refuse_pool_above_two_dimensions(search_settings, *, design_policy, design_policy_applies,
                                     input_dimension):
    """Acquisition policy: no pool EIG or EPIG at input dimension d >= 3.

    `search_settings` is what `declared_settings` returned. Pool is the effective mode unless it
    is the ascent declaration, so an explicit pool and an omitted flag are refused alike. Only
    the two information estimators are refused: a study policy (point_disagreement,
    space_filling) passes, and so does a cell whose third digit selects no policy. The caller
    passes the resolved policy and the task's own input dimension; nothing is looked up here.
    """
    if (search_settings.get("design_search") != "ascent" and design_policy_applies
            and design_policy in ("joint_eig", "epig") and input_dimension > 2):
        remedy = ("declare --design-search ascent for this cell" if design_policy == "joint_eig"
                  else "ascent above two inputs is joint_eig only, so declare a joint_eig cell "
                       "with --design-search ascent")
        raise ValueError(
            f"design-search pool is refused for {design_policy} at input dimension "
            f"{input_dimension} > 2 (the acquisition policy forbids pool EIG/EPIG "
            f"at d >= 3 for every run); {remedy}")


def strip_cli_settings(config):
    """Omitted and explicit pool flags do not change the legacy config key set."""
    keys = {"design_search", "design_search_noise_floor_method"}
    return {key: value for key, value in config.items() if key not in keys}


def selection_diagnostics(rows, decision, key="eig", se_key="mc_standard_error"):
    """Describe the registered selection without changing its candidates or tie rule.

    `se_key` (added with the EIG-part objectives) names the Monte Carlo standard error of `key`; the default is the
    column every existing caller's `key` carries its error in.
    """
    selected = next(row for row in rows if row["design"] == decision["chosen_design"])
    remaining = list(rows)
    remaining.remove(selected)
    runner_up = min(remaining, key=lambda row: (-row[key], row["design"])) if remaining else None
    selected_se = selected.get(se_key)
    runner_se = None if runner_up is None else runner_up.get(se_key)
    gap = None if runner_up is None else float(decision["top_value"] - runner_up[key])
    uncertainty = ratio = None
    uncertain = False if runner_up is None else None
    method = "unavailable"
    if selected_se is not None and (runner_up is None or runner_se is not None):
        method = "independent_sum_upper_bound"
        if runner_up is not None:
            uncertainty = float(np.sqrt(selected_se ** 2 + runner_se ** 2))
            uncertain = gap <= decision["tie_atol"] if uncertainty == 0 else gap < uncertainty
            ratio = None if uncertainty == 0 else gap / uncertainty
    return {"n_candidates": len(rows), "top_value": decision["top_value"],
            "runner_up_value": None if runner_up is None else runner_up[key], "top_gap": gap,
            "selected_mc_se": selected_se, "runner_up_mc_se": runner_se,
            "estimated_uncertainty": uncertainty, "uncertainty_method": method,
            "top_gap_over_uncertainty": ratio, "n_tied_at_max": decision["n_tied_at_max"],
            "selection_uncertain": uncertain}


def pool_diagnostics(objective, rows, decision, *, global_seed, round_index, key="eig"):
    """Explicit-pool diagnostics use a size-one holdout, never the selection seed."""
    holdout_seed = seed_for(HOLDOUT_STREAM, global_seed, round_index)
    point = np.asarray([decision["chosen_design"]], dtype=float)
    holdout = objective(point, holdout_seed)
    if len(holdout) != 1 or holdout[0]["design"] != decision["chosen_design"]:
        raise ValueError("design-search holdout must preserve the chosen design")
    return {**selection_diagnostics(rows, decision, key), "candidate_source": "pool",
            "holdout_value": holdout[0][key], "holdout_seed_role": "seed_C",
            "holdout_batch_size": 1, "holdout_batch_order": ["chosen"]}


def validate_parameters(parameters=None):
    """Merge `parameters` over PARAMETER_DEFAULTS, check every SPSA gain, and return the dict.

    Raises ValueError on an unknown name, a non-finite or invalid gain, or alpha <= gamma.
    """
    values = {**PARAMETER_DEFAULTS, **(parameters or {})}
    if set(values) != set(PARAMETER_DEFAULTS):
        raise ValueError("unknown design-search gain parameter")
    for name in ("a", "A", "alpha", "gamma", "c", "kappa", "k_sigma"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"design-search {name} must be finite")
        if value < 0 or (name not in ("A", "gamma") and value == 0):
            raise ValueError(f"design-search {name} has an invalid gain value")
    if type(values["W"]) is not int or values["W"] < 1:
        raise ValueError("design-search W must be a positive integer")
    if values["alpha"] <= values["gamma"]:
        raise ValueError("design-search requires alpha > gamma")
    return values


def noise_floor_estimates(a_values, b_values):
    """All 45 unordered pairs of the ten start/end check points; no MC calls."""
    a_values, b_values = np.asarray(a_values, dtype=float), np.asarray(b_values, dtype=float)
    if a_values.shape != (2 * RESTARTS,) or b_values.shape != a_values.shape:
        raise ValueError("design-search noise floor needs the same ten check values")
    if not np.all(np.isfinite(a_values)) or not np.all(np.isfinite(b_values)):
        raise ValueError("design-search noise floor values must be finite")
    differences = a_values - b_values
    # The reference sigma_climb_probe uses every i < j and centered ddof=0.
    pair_differences = np.array([
        (a_values[i] - a_values[j]) - (b_values[i] - b_values[j])
        for i in range(len(a_values)) for j in range(i + 1, len(a_values))
    ])
    centered = float(np.std(differences - differences.mean(), ddof=0) / np.sqrt(2.0))
    return {"rms": float(np.sqrt(np.mean(differences ** 2)) / np.sqrt(2.0)),
            "centered": centered,
            "pairwise": float(np.sqrt(np.mean(pair_differences ** 2)) / np.sqrt(2.0)),
            "centered_times_sqrt2": float(centered * np.sqrt(2.0))}


def _is_invalid(row):
    """Acquisition guard rule B2: a row is invalid iff it carries the literal `"invalid": True`."""
    return row.get("invalid") is True


def _row_value(row, key):
    """A valid row's objective value; None for an invalid row (acquisition guard rules B7, B8)."""
    return None if _is_invalid(row) else row[key]


def displacement_gradient(value_difference, plus, minus):
    """SPSA's reciprocal direction uses each coordinate's actual clipped span."""
    displacement = np.asarray(plus, dtype=float) - np.asarray(minus, dtype=float)
    guarded = np.abs(displacement) < 1e-12
    gradient = np.divide(value_difference, displacement, out=np.zeros_like(displacement), where=~guarded)
    return gradient, int(np.count_nonzero(guarded)), displacement


@dataclass(frozen=True)
class SearchTransform:
    """Bounds min/max after declared logs, independent of fitting's z-score transform."""

    input_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    log_inputs: tuple[str, ...]

    def __post_init__(self):
        """Check the declared inputs, bounds and log inputs; raise ValueError on any defect."""
        if len(self.input_names) < 2 or len(set(self.input_names)) != len(self.input_names):
            raise ValueError("design-search requires at least two distinct inputs")
        if len(self.bounds) != len(self.input_names):
            raise ValueError("design-search needs bounds for every input")
        if len(set(self.log_inputs)) != len(self.log_inputs) or set(self.log_inputs) - set(self.input_names):
            raise ValueError("design-search log inputs must name distinct declared inputs")
        for name, bounds in zip(self.input_names, self.bounds, strict=True):
            if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
                raise ValueError(f"design-search invalid bounds for {name}")
            if name in self.log_inputs and bounds[0] <= 0:
                raise ValueError(f"design-search logged bounds must be positive: {name}")

    @classmethod
    def from_bounds(cls, names, bounds, log_inputs):
        """Build a transform from input names, a name -> (low, high) mapping and the logged names."""
        names = tuple(names)
        return cls(names, tuple(tuple(map(float, bounds[name])) for name in names),
                   tuple(name for name in names if name in log_inputs))

    def _array(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim not in (1, 2) or values.shape[-1] != len(self.input_names):
            raise ValueError("design-search coordinates must match the declared inputs")
        if not np.all(np.isfinite(values)):
            raise ValueError("design-search coordinates must be finite")
        return values

    def forward(self, values):
        """Map raw input coordinates inside the bounds to the unit box (log first where declared)."""
        values = self._array(values)
        result = np.empty_like(values)
        for index, (name, (low, high)) in enumerate(zip(self.input_names, self.bounds, strict=True)):
            column = values[..., index]
            if np.any((column < low) | (column > high)):
                raise ValueError(f"design-search coordinate outside bounds: {name}")
            if name in self.log_inputs:
                column, low, high = np.log(column), np.log(low), np.log(high)
            result[..., index] = (column - low) / (high - low)
        return result

    def inverse(self, values):
        """Map unit-box coordinates back to raw inputs; declared endpoints come back exactly."""
        values = self._array(values)
        if np.any((values < 0) | (values > 1)):
            raise ValueError("design-search coordinate outside unit box")
        result = np.empty_like(values)
        for index, (name, (low, high)) in enumerate(zip(self.input_names, self.bounds, strict=True)):
            column = values[..., index]
            if name in self.log_inputs:
                raw = np.exp(np.log(low) + column * (np.log(high) - np.log(low)))
            else:
                raw = low + column * (high - low)
            # Keep declared endpoints exact despite log/exp round-off.
            result[..., index] = np.where(column == 0, low, np.where(column == 1, high, raw))
        return result

    def record(self):
        """Return the transform's declaration as a JSON-ready dict for the run record."""
        return {"input_names": list(self.input_names), "bounds": [list(v) for v in self.bounds],
                "log_inputs": list(self.log_inputs),
                "rule": "log declared inputs, then bounds minmax to unit box",
                "distinct_from_fitting_InputTransform": True}


def projected_spsa(objective, transform, *, global_seed, cell, round_index,
                   acquisition_seed, acquisition_stream, objective_key="eig",
                   noise_floor_method=DEFAULT_NOISE_FLOOR_METHOD, arrival_annotator=None):
    """Evaluate full-precision batches; only the five seed-A arrival rows are selected.

    EIG-part objectives (2026-10-06). A call with objective_key "eig" or "epig" and no annotator returns
    exactly the old record.
    - `objective_key` may also be an EIG part, "structure_eig" or "parameter_eig"
      (EIG_PART_OBJECTIVES). The objective's valid rows then carry that column and its Monte Carlo
      standard error (structure_mc_standard_error, parameter_mc_standard_error), and every use of
      the objective value reads the part: the SPSA differences, the noise floor, the climb of each
      restart, the chooser over the five arrival rows, its selection diagnostics and the holdout.
      The arm optimizes its own criterion.
    - `arrival_annotator(rows)` is called once with the five arrival rows, right after they are
      evaluated and before the noise floor and the chooser. It may only ADD keys to a valid row:
      every key a row already has must keep its value and an invalid row must gain nothing
      (checked here). The rows are the same objects `rows` holds, so both lists show the keys.
    """
    gains = validate_parameters(dict(PARAMETER_DEFAULTS))
    if len({acquisition_stream, NOISE_STREAM, HOLDOUT_STREAM}) != 3:
        raise ValueError("design-search A/B/C stream names must be distinct")
    if objective_key not in ("eig", "epig", *EIG_PART_OBJECTIVES):
        raise ValueError("design-search objective must be eig, epig or an EIG part "
                         "(structure_eig, parameter_eig)")
    if noise_floor_method not in NOISE_FLOOR_METHODS:
        raise ValueError("design-search noise-floor method is not declared")
    se_key = EIG_PART_OBJECTIVES.get(objective_key, "mc_standard_error")
    dimension = len(transform.input_names)
    restart_seeds = [seed_for("design_restart", global_seed, cell, round_index, i)
                     for i in range(1, RESTARTS + 1)]
    generators = [np.random.default_rng(seed) for seed in restart_seeds]
    starts = np.array([rng.uniform(0.0, 1.0, dimension) for rng in generators])
    positions = starts.copy()
    history = [[point.copy()] for point in starts]
    paths = np.zeros(RESTARTS)
    done = np.zeros(RESTARTS, dtype=bool)
    stop_steps = np.full(RESTARTS, MAX_STEPS)
    movement_met = np.zeros(RESTARTS, dtype=bool)
    trajectories = [{"restart_index": i + 1, "seed_i": seed,
                     "start_unit": starts[i].tolist(), "steps": []}
                    for i, seed in enumerate(restart_seeds)]
    rows, perturbation_rows = [], []
    # Acquisition guard rule B9: one entry per invalid row, in evaluation order.
    invalid_points = []
    call_index = 0
    evaluation_index = 0

    def evaluate(unit_points, seed, phase, steps, restart_indices):
        """Evaluate unit-box points with `seed`, record each row under `phase`, return the records."""
        nonlocal call_index, evaluation_index

        raw_points = transform.inverse(unit_points)
        evaluated = objective(raw_points, seed)
        if len(evaluated) != len(raw_points):
            raise ValueError("design-search objective returned the wrong row count")
        added = []
        for row_index, (point, unit, row, step, restart_index) in enumerate(zip(
                raw_points, unit_points, evaluated, steps, restart_indices, strict=True)):
            if not np.array_equal(np.asarray(row.get("design"), dtype=float), point):
                raise ValueError("design-search objective changed the evaluated design")
            # Acquisition guard rule B2: an invalid row is a state, not a number; the finite-value
            # contract below applies to valid rows only.
            invalid = _is_invalid(row)
            if invalid:
                if (row.get("invalid_reason") not in INVALID_REASONS or objective_key in row
                        or "mc_standard_error" in row or se_key in row):
                    raise ValueError("design-search invalid row must carry a registered "
                                     "invalid_reason and no objective value")
            elif not np.isfinite(row.get(objective_key, float("nan"))):
                raise ValueError("design-search objective must return finite values")
            record = {**row, "unit_design": unit.tolist(), "phase": phase, "step": int(step),
                      "restart_index": int(restart_index), "objective_seed": seed,
                      "evaluation_index": evaluation_index, "objective_call_index": call_index,
                      "batch_size": len(raw_points), "row_index": row_index}
            if "pool_index" in record:
                raise ValueError("design-search ascent rows must not carry pool indices")
            if phase == "perturbation":
                # Batched objective values are only used transiently for the direction.
                perturbation_rows.append({key: record[key] for key in (
                    "design", "unit_design", "phase", "step", "restart_index", "objective_seed",
                    "evaluation_index", "objective_call_index", "batch_size", "row_index")})
                if invalid:
                    perturbation_rows[-1].update(invalid=True, invalid_reason=record["invalid_reason"])
            else:
                rows.append(record)
            if invalid:
                invalid_points.append({key: record[key] for key in (
                    "objective_call_index", "evaluation_index", "phase", "step", "restart_index",
                    "design", "unit_design", "invalid_reason")})
            added.append(record)
            evaluation_index += 1
        call_index += 1
        return added

    start_rows = []
    for index in range(RESTARTS):
        start_rows.extend(evaluate(starts[index:index + 1], acquisition_seed,
                                   "start", [0], [index + 1]))
    for step in range(1, MAX_STEPS + 1):
        active = np.flatnonzero(~done)
        c_k = gains["c"] / step ** gains["gamma"]
        a_k = gains["a"] / (gains["A"] + step) ** gains["alpha"]
        deltas, points, clipped, clip_counts = [], [], [], []
        for index in active:
            delta = generators[index].choice((-1.0, 1.0), dimension)
            plus, minus = positions[index] + c_k * delta, positions[index] - c_k * delta
            bounded_plus, bounded_minus = np.clip(plus, 0, 1), np.clip(minus, 0, 1)
            deltas.append(delta)
            points.extend((bounded_plus, bounded_minus))
            clipped.append((not np.array_equal(plus, bounded_plus), not np.array_equal(minus, bounded_minus)))
            clip_counts.append(int(np.count_nonzero((plus != bounded_plus) | (minus != bounded_minus))))
        evaluated = evaluate(np.array(points), acquisition_seed, "perturbation", [step] * len(points),
                             [int(index) + 1 for index in active for _ in range(2)])
        for offset, index in enumerate(active):
            plus, minus = evaluated[2 * offset:2 * offset + 2]
            plus_invalid, minus_invalid = _is_invalid(plus), _is_invalid(minus)
            if plus_invalid or minus_invalid:
                # Acquisition guard rule B3: no update from an invalid side. Position, history and path
                # length stay as they were, no movement-stop check runs this step, and the
                # delta above was already drawn, so the RNG order is unchanged.
                trajectories[index]["steps"].append({
                    "step": step, "a_k": a_k, "c_k": c_k,
                    "before_unit": positions[index].tolist(), "after_unit": positions[index].tolist(),
                    "delta": deltas[offset].tolist(),
                    "plus_clipped": clipped[offset][0], "minus_clipped": clipped[offset][1],
                    "clip_count": clip_counts[offset],
                    "plus_evaluation": plus["evaluation_index"], "minus_evaluation": minus["evaluation_index"],
                    "path_length": float(paths[index]),
                    "skipped": True, "skip_reason": SKIP_REASON,
                    "plus_invalid": plus_invalid, "minus_invalid": minus_invalid,
                })
                continue
            gradient, guard_count, displacement_vector = displacement_gradient(
                plus[objective_key] - minus[objective_key], plus["unit_design"], minus["unit_design"])
            before = positions[index].copy()
            proposal = before + a_k * gradient
            positions[index] = np.clip(proposal, 0, 1)
            paths[index] += np.linalg.norm(positions[index] - before)
            history[index].append(positions[index].copy())
            displacement = None
            if step >= MIN_STEPS and len(history[index]) >= 2 * gains["W"]:
                recent = np.asarray(history[index][-gains["W"]:]).mean(axis=0)
                previous = np.asarray(history[index][-2 * gains["W"]:-gains["W"]]).mean(axis=0)
                displacement = float(np.linalg.norm(recent - previous))
                if displacement < gains["kappa"] * c_k:
                    movement_met[index] = done[index] = True
                    stop_steps[index] = step
            trajectories[index]["steps"].append({
                "step": step, "a_k": a_k, "c_k": c_k,
                "before_unit": before.tolist(), "after_unit": positions[index].tolist(),
                "delta": deltas[offset].tolist(), "gradient": gradient.tolist(),
                "gradient_norm": float(np.linalg.norm(gradient)),
                "plus_clipped": clipped[offset][0], "minus_clipped": clipped[offset][1],
                "clip_count": clip_counts[offset], "gradient_zero_displacement_count": guard_count,
                "min_displacement_fraction": float(np.min(np.abs(displacement_vector)) / (2 * c_k)),
                "update_clipped": not np.array_equal(proposal, positions[index]),
                "plus_evaluation": plus["evaluation_index"], "minus_evaluation": minus["evaluation_index"],
                "window_displacement": displacement, "path_length": float(paths[index]),
            })
        if np.all(done):
            break

    arrival_rows = []
    for index in range(RESTARTS):
        arrival_rows.extend(evaluate(positions[index:index + 1], acquisition_seed,
                                     "arrival", [stop_steps[index]], [index + 1]))
    if arrival_annotator is not None:
        before = [dict(row) for row in arrival_rows]
        arrival_annotator(arrival_rows)
        for row, old in zip(arrival_rows, before, strict=True):
            if (any(name not in row or row[name] != value for name, value in old.items())
                    or (_is_invalid(old) and set(row) != set(old))):
                raise ValueError("design-search arrival annotation may only add keys to a valid row")
    noise_seed = seed_for(NOISE_STREAM, global_seed, round_index)
    check_points = np.concatenate((starts, positions))
    check_steps = [0] * RESTARTS + stop_steps.tolist()
    check_restarts = list(range(1, RESTARTS + 1)) * 2
    check_a_rows, noise_rows = start_rows + arrival_rows, []
    for index, point in enumerate(check_points):
        noise_rows.extend(evaluate(point[None, :], noise_seed, "noise_floor",
                                   [check_steps[index]], [check_restarts[index]]))
    check_hashes = {
        name: hashlib.sha256(json.dumps([row["design"] for row in batch],
                                       sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for name, batch in (("A", check_a_rows), ("B", noise_rows))
    }
    if check_hashes["A"] != check_hashes["B"]:
        raise ValueError("design-search A/B noise check points differ")
    # Acquisition guard rule B6: validity depends on the bank and the point, not on the seed, so the A
    # and B rows of one check point share it. Any invalid check row leaves the noise floor
    # undetermined; no estimator runs over a subset of the check points.
    invalid_checks = [index for index, (a_row, b_row) in enumerate(zip(
        check_a_rows, noise_rows, strict=True)) if _is_invalid(a_row) or _is_invalid(b_row)]
    if invalid_checks:
        noise_estimates = sigma_hat = mean_difference = None
    else:
        original_values = np.array([row[objective_key] for row in check_a_rows])
        noise_values = np.array([row[objective_key] for row in noise_rows])
        differences = original_values - noise_values
        noise_estimates = noise_floor_estimates(original_values, noise_values)
        sigma_hat = noise_estimates[noise_floor_method]
        mean_difference = float(np.mean(differences))
    for index, trajectory in enumerate(trajectories):
        # Acquisition guard rule B7: no climb is measured from an invalid start or arrival row, and
        # without a climb or a noise floor the stop label is null, never a new word.
        if _is_invalid(check_a_rows[index]) or _is_invalid(check_a_rows[index + RESTARTS]):
            climbed = None
        else:
            climbed = float(check_a_rows[index + RESTARTS][objective_key] - check_a_rows[index][objective_key])
        if not movement_met[index] or stop_steps[index] == MAX_STEPS:
            reason = "max_steps"
        elif climbed is None or sigma_hat is None:
            reason = None
        elif climbed > gains["k_sigma"] * sigma_hat and paths[index] > 0:
            reason = "stopped_after_gain"
        else:
            reason = "stopped_without_gain"
        if climbed is None or sigma_hat is None:
            stop_z = None
        else:
            stop_z = None if sigma_hat == 0 else climbed / sigma_hat
        trajectory.update(end_unit=positions[index].tolist(), stop_step=int(stop_steps[index]),
                          stop_reason=reason, movement_stop_met=bool(movement_met[index]),
                          path_length=float(paths[index]), value_climbed=climbed,
                          stop_z=stop_z,
                          stop_marginal=False if stop_z is None else abs(stop_z - gains["k_sigma"]) < 1,
                          start_evaluation=check_a_rows[index]["evaluation_index"],
                          end_evaluation=check_a_rows[index + RESTARTS]["evaluation_index"])
    # Acquisition guard rule B5: the chooser's candidates are the valid arrival rows; when every arrival
    # is valid it receives the same list object as before.
    valid_arrivals = [row for row in arrival_rows if not _is_invalid(row)]
    candidates = arrival_rows if len(valid_arrivals) == len(arrival_rows) else valid_arrivals
    holdout_seed = seed_for(HOLDOUT_STREAM, global_seed, round_index)
    holdout_rows = []
    holdout_value = holdout_start_value = holdout_climb = holdout_survival = None
    if candidates:
        decision = choose_design(candidates, objective_key)
        chosen_row = next(row for row in candidates if row["design"] == decision["chosen_design"])
        chosen_index = chosen_row["restart_index"] - 1
        holdout_rows = [None] * RESTARTS
        for index in [chosen_index, *(i for i in range(RESTARTS) if i != chosen_index)]:
            holdout_rows[index] = evaluate(positions[index:index + 1], holdout_seed,
                                          "independent_holdout", [stop_steps[index]], [index + 1])[0]
        # Acquisition guard rule B8: an invalid holdout row gives None, never a number.
        holdout_value = _row_value(holdout_rows[chosen_index], objective_key)
        holdout_start = evaluate(starts[chosen_index:chosen_index + 1], holdout_seed,
                                 "independent_holdout_start", [0], [chosen_index + 1])[0]
        holdout_start_value = _row_value(holdout_start, objective_key)
        if holdout_value is not None and holdout_start_value is not None:
            holdout_climb = holdout_value - holdout_start_value
        chosen_climb = trajectories[chosen_index]["value_climbed"]
        if holdout_climb is not None and chosen_climb is not None:
            holdout_survival = None if chosen_climb <= 0 else holdout_climb / chosen_climb
        chosen = {
            "chosen": decision["chosen_design"], "chosen_restart_index": chosen_row["restart_index"],
            "chosen_stop_reason": trajectories[chosen_row["restart_index"] - 1]["stop_reason"],
            "chosen_endpoint_unlabeled_gain": (
                trajectories[chosen_row["restart_index"] - 1]["stop_reason"] != "stopped_after_gain"),
        }
        policy = {"objective_column": objective_key, "chosen_value": decision["top_value"],
                  **selection_diagnostics(candidates, decision, objective_key, se_key),
                  "holdout_value": holdout_value,
                  "holdout_seed_role": "seed_C", "candidate_source": "ascent_endpoints",
                  "holdout_batch_size": 1, "holdout_batch_order": ["chosen"],
                  "decision_authority": "bayes_model_discovery.eig.joint.choose_design",
                  "n_tied_at_max": decision["n_tied_at_max"], "tie_rule": decision["tie_rule"],
                  "tie_atol": decision["tie_atol"], "selection_rows": "arrival_rows_seed_A_only",
                  "trajectory_pointer": "#/trajectories"}
        if objective_key in EIG_PART_OBJECTIVES:
            # EIG-part objectives: the standard-error column the selection diagnostics read.
            policy["selection_standard_error_column"] = se_key
    else:
        # Acquisition guard rule B5, no valid arrival: no chooser and no holdout evaluation, so the
        # record and its policy carry no chosen-dependent value.
        chosen = dict.fromkeys(("chosen", "chosen_restart_index", "chosen_stop_reason",
                                "chosen_endpoint_unlabeled_gain"))
        policy = {"objective_column": objective_key,
                  "holdout_seed_role": "seed_C", "candidate_source": "ascent_endpoints",
                  "holdout_batch_size": 1, "holdout_batch_order": ["chosen"],
                  "decision_authority": "bayes_model_discovery.eig.joint.choose_design",
                  "selection_rows": "arrival_rows_seed_A_only",
                  "trajectory_pointer": "#/trajectories"}
    record = {
        "design_search": "ascent", "candidate_source": "ascent_endpoints",
        "candidate_exclusion_policy": "none",
        "seed": acquisition_seed, "noise_floor_seed": noise_seed, "holdout_seed": holdout_seed,
        "seed_roles": {"A": "search_and_selection", "B": "selection_adjacent_noise_floor",
                       "C": "independent_final_holdout_not_used_for_selection"},
        "seed_streams": {"A": acquisition_stream, "B": NOISE_STREAM, "C": HOLDOUT_STREAM},
        **chosen,
        "rows": rows, "arrival_rows": arrival_rows, "holdout_rows": holdout_rows,
        "perturbation_trace": perturbation_rows, "objective_evaluation_count": evaluation_index,
        "objective_call_count": call_index,
        "holdout_value": holdout_value, "holdout_start_value": holdout_start_value,
        "holdout_climb": holdout_climb, "holdout_survival": holdout_survival,
        "trajectories": trajectories,
        "holdout_batch_size": 1, "holdout_batch_order": ["chosen"],
        "endpoint_holdout_values": [_row_value(row, objective_key) for row in holdout_rows],
        "parameters": gains, "noise_floor_sigma_hat": sigma_hat,
        "noise_floor_method": noise_floor_method, "noise_floor_estimates": noise_estimates,
        "noise_floor_pair_count": 45, "noise_floor_centered_ddof": 0,
        "noise_floor_offset": mean_difference,
        "noise_check_point_count": len(check_a_rows),
        "noise_check_points_sha256": check_hashes, "noise_check_batch_size": 1,
        "design_search_transform": transform.record(),
        "policy": policy,
    }
    # Acquisition guard rule B9: these keys exist only when a row was invalid, so an all-valid round
    # returns exactly the old dict.
    if invalid_points:
        record["acquisition_status"] = ("ascent_with_exclusions" if candidates
                                        else "all_designs_invalid")
        record["invalid_design_points"] = invalid_points
        if invalid_checks:
            record["noise_floor_status"] = NOISE_FLOOR_UNDETERMINED
            record["noise_floor_invalid_check_points"] = invalid_checks
    return record
