"""BoxingGym design search for the active design arms (first version 2026-10-04; revised 2026-10-06 for the unified
protocol).

`--design-search ascent` keeps the round's pool table exactly as the pool path computes it and then applies the unified
protocol's design-optimization rule: a continuous design space gets a multi-start local ascent seeded from the
candidate pool, with the pool argmax as its floor; a discrete design space (the pool IS the whole legal design set) is
searched exhaustively, which is the pool argmax itself.

The objective is the arm's own criterion, scored by the SAME estimator the pool path uses for that arm: the caller passes
`objective(rows)`, which calls `count_inference.choose` (exhaustive outcomes: the `eig`, `structure_eig` or
`parameter_eig` column), `positive_inference.joint_eig` (scalar outcomes, one design per call; `decompose=True` for the
two expected information gain (EIG) parts) or `count_inference.choose_epig` (expected predictive information gain
(EPIG), with the run's fixed target points), with the round's own
acquisition seed and `eig_samples`, per component, summed over components exactly as the pool tables are. Nothing here
re-derives a criterion.

Method `batched_compass_search_v2` (a deterministic, gradient-free pattern search):
- Coordinates: each declared input mapped linearly onto [0, 1] by its declared box (the single-input `design_range` when the
  task declares one, else the input's `range`, the same bounds `build_design_pool` reads). Linear, not log: several boxes
  contain 0 or negative values (peregrines, dugongs, location).
- Starts: the STARTS_POOL best pool rows of the round's own pool table under the arm's criterion (by value, the registered
  tie rule ordering ties). STARTS_RANDOM uniform unit-box draws from the stream `boxinggym_design_search` may follow; this
  version sets it to 0, so every start is a pool row (the first version, v1, used 3 pool rows and 2 random draws).
- Step: every live start evaluates its 2*d axis neighbours at +/- step (clipped to the box) in ONE objective call for all
  starts; a start moves to its best neighbour when that neighbour's value is strictly higher, otherwise halves its step. A
  start stops when its step falls below the finest axis resolution (half an integer unit on an integer axis,
  CONTINUOUS_MIN_STEP on a continuous one); the round stops after MAX_ITERATIONS objective calls at most.
- The objective is evaluated on the relaxation (integer axes un-rounded) - the model's predictive is defined there - but a
  neighbour is admitted only when its PROJECTION is a design the environment's own validator accepts
  (`TaskEnvironment.accepts(encode_design(spec, row))`), so every point the search can stand on maps to a valid design.
- Projection: integer axes (the task's declared `integer_inputs`, which its validator casts) are rounded to the nearest
  integer inside the box; continuous axes are kept. Each start's endpoint is projected and RE-SCORED at the projected point,
  together with the round's pool argmax re-scored by the same call, and the registered `choose_design` picks among them,
  so the chosen design's value is never below the pool argmax's under the same estimator (up to the tie tolerance).
- Evaluation failures: an objective call that raises is repeated design by design; a design whose own evaluation raises, or
  whose value is not finite, is infeasible: it is never moved to and never chosen, and the record counts it. If the pool
  argmax itself cannot be re-scored, the pool argmax is observed (the pool path's choice) and the record says why.
- Exhaustive outcomes (`count_inference.choose`) give the same value for a design whatever batch it is scored in, because the
  seed fixes only the particle subset. `joint_eig`'s Monte Carlo draws depend on the batch, so for those tasks the caller
  scores every design in its own call with the round seed (common random numbers per design). `choose_epig` scores each
  candidate against the same targets on the same particle subset; it is called once per batch.

Why a pattern search and not simultaneous perturbation stochastic approximation (SPSA) or Adam: the objective is
deterministic for a fixed round seed (above), there is no gradient without jax, and a compass search with step halving is the simplest bounded local method whose every iterate is an
evaluated, recorded, env-valid point. ForceBench's design ascent used five starts and forty steps; the same budget is kept.
"""

import hashlib
import json

import numpy as np

from ..eig.joint import TIE_ATOL, EIGError, choose_design

METHOD = "batched_compass_search_v2"
STARTS_POOL = 5
STARTS_RANDOM = 0
MAX_ITERATIONS = 40
INITIAL_STEP = 0.25
CONTINUOUS_MIN_STEP = 1e-3
RANDOM_START_TRIES = 200
EDGE_TOL_UNIT = 0.01
SEARCH_STREAM = "boxinggym_design_search"
# The arm's criterion column in the pool table: joint EIG, Model EIG, Parameter EIG, EPIG.
OBJECTIVES = ("eig", "structure_eig", "parameter_eig", "epig")
# What an objective call may raise on a design the pool did not contain (the estimators' own refusals).
EVALUATION_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError, EIGError)
ERRORS_KEPT = 5


def design_search_constants():
    """The search's fixed constants and rules as a JSON-serializable dict (hashed into the run record)."""
    return {
        "method": METHOD, "starts_pool": STARTS_POOL, "starts_random": STARTS_RANDOM,
        "max_iterations": MAX_ITERATIONS, "initial_step_unit": INITIAL_STEP,
        "continuous_min_step_unit": CONTINUOUS_MIN_STEP, "integer_min_step": "half an integer unit, in unit coordinates",
        "random_start_tries": RANDOM_START_TRIES, "edge_tolerance_unit": EDGE_TOL_UNIT,
        "seed_stream": SEARCH_STREAM, "coordinates": "linear map of each declared input box onto [0, 1]",
        "objectives": list(OBJECTIVES),
        "start_rule": "the starts_pool best rows of the round's pool table under the arm's criterion (registered tie "
                      "order), then starts_random uniform unit-box draws until env-valid",
        "move_rule": "best of the 2*d axis neighbours if its value is strictly higher, else halve the step",
        "validity_rule": "a point is admitted only if TaskEnvironment.accepts(encode_design(spec, projection))",
        "projection": "integer_inputs rounded to the nearest integer inside the box; continuous inputs kept",
        "choice": "choose_design over the pool argmax and the projected endpoints, all re-scored in one call; an entry "
                  "whose evaluation failed is left out and recorded; an unscorable pool argmax is observed as is",
        "evaluation_failure_rule": "a call that raises is repeated design by design; a design whose own evaluation "
                                   "raises or is not finite is infeasible, never chosen, and counted",
        "estimator": "the arm's own pool-path call: count_inference.choose (exhaustive outcomes), "
                     "positive_inference.joint_eig per design (scalar outcomes; decompose=True for structure_eig and "
                     "parameter_eig), count_inference.choose_epig (epig, the run's fixed targets)",
        "discrete_rule": "a task whose pool is its whole legal design set (an environment-fixed pool or a categorical "
                         "grid) is searched exhaustively: the pool argmax, recorded as mode exhaustive",
    }


def constants_sha256(constants=None):
    """sha256 of the canonical JSON of `constants` (default: `design_search_constants()`)."""
    text = json.dumps(constants or design_search_constants(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def declared_settings():
    """The settings a run declares (config.json and the label receipt bind them)."""
    return {"design_search": "ascent", "design_search_method": METHOD,
            "design_search_constants_sha256": constants_sha256()}


def not_applicable_reason(spec):
    """None when the task's design space is a declared box the search can move in, else the reason."""
    if spec.environment_fixed_pool:
        return "the environment fixes the legal designs as a stored pool"
    if getattr(spec, "environment_pool_source", "") in ("categorical", "sampled"):
        return f"the design set is {spec.environment_pool_source}, not a declared box"
    if spec.design_encoder in ("pool_index", "pool_handle"):
        return f"designs carry environment handles ({spec.design_encoder})"
    if spec.categorical_inputs:
        return "the task has categorical inputs"
    if spec.design_exclusion != "none":
        return "the task removes observed designs from the candidate set"
    return None


def design_space_mode(spec):
    """("ascent", None) on a declared box; ("exhaustive", reason) when the pool is the task's whole legal design set (an
    environment-fixed pool such as survival's patients, or a categorical grid such as irt's); raises otherwise (a pool
    sampled from a larger design space, which neither rule covers)."""
    reason = not_applicable_reason(spec)
    if reason is None:
        return "ascent", None
    if spec.environment_fixed_pool or getattr(spec, "environment_pool_source", "") == "categorical":
        return "exhaustive", reason
    raise ValueError(f"--design-search ascent does not apply to task {spec.name}: {reason}")


class SearchSpace:
    """The declared box, the integer axes and the env-validity check of one task."""

    def __init__(self, spec, environment, encode):
        reason = not_applicable_reason(spec)
        if reason is not None:
            raise ValueError(f"--design-search ascent does not apply to task {spec.name}: {reason}")
        names = list(spec.input_names)
        if len(names) == 1:
            low, high = spec.design_range or spec.input_range
            boxes = [(float(low), float(high))]
        else:
            boxes = [tuple(float(v) for v in item["range"]) for item in spec.inputs]
        self.names = names
        self.low = np.asarray([b[0] for b in boxes], dtype=float)
        self.high = np.asarray([b[1] for b in boxes], dtype=float)
        if np.any(self.high <= self.low):
            raise ValueError(f"task {spec.name}: a declared input box is empty")
        self.integer = np.asarray([name in spec.integer_inputs for name in names], dtype=bool)
        width = self.high - self.low
        self.min_step = float(np.min(np.where(self.integer, 0.5 / width, CONTINUOUS_MIN_STEP)))
        self._spec, self._environment, self._encode = spec, environment, encode
        self._valid = {}

    @property
    def dim(self):
        """Number of design inputs (axes)."""
        return len(self.names)

    def to_design(self, unit):
        """Map unit-box coordinates (clipped to [0, 1]) onto the declared input box."""
        return self.low + np.clip(np.asarray(unit, dtype=float), 0.0, 1.0) * (self.high - self.low)

    def to_unit(self, design):
        """Map a design in the declared box onto unit-box coordinates."""
        return (np.asarray(design, dtype=float) - self.low) / (self.high - self.low)

    def project(self, design):
        """Round the integer axes of `design` to the nearest integer inside the box; continuous axes are kept."""
        design = np.asarray(design, dtype=float).copy()
        lo_int, hi_int = np.ceil(self.low), np.floor(self.high)
        design[self.integer] = np.clip(np.rint(design[self.integer]), lo_int[self.integer], hi_int[self.integer])
        return design

    def valid(self, design):
        """The environment's own validator on the projected design (cached by its handle)."""
        handle = self._encode(self._spec, self.project(design))
        if handle not in self._valid:
            self._valid[handle] = bool(self._environment.accepts(handle))
        return self._valid[handle]

    def edge_axes(self, design):
        """Names of the axes on which `design` lies within EDGE_TOL_UNIT of the box edge (in unit coordinates)."""
        unit = self.to_unit(design)
        return [name for name, u in zip(self.names, unit, strict=True)
                if u <= EDGE_TOL_UNIT or u >= 1 - EDGE_TOL_UNIT]

    def bound_axes(self, design):
        """Names of the axes on which `design` sits on or beyond the declared bound."""
        design = np.asarray(design, dtype=float)
        return [name for name, x, lo, hi in zip(self.names, design, self.low, self.high, strict=True)
                if x <= lo or x >= hi]

    def record(self):
        """The box, integer axes and minimum step, for the run record."""
        return {"inputs": self.names, "low": self.low.tolist(), "high": self.high.tolist(),
                "integer_inputs": [n for n, i in zip(self.names, self.integer, strict=True) if i],
                "min_step_unit": self.min_step}


def _row(design):
    return [float(v) for v in np.atleast_1d(np.asarray(design, dtype=float))]


class _Evaluator:
    """The objective with its failures isolated design by design and counted; values are float arrays, -inf = infeasible."""

    def __init__(self, objective, dim):
        self._objective, self._dim = objective, dim
        self.calls = self.evaluations = 0
        self.failed_calls = self.failed_designs = self.non_finite = 0
        self.errors = []

    def _attempt(self, rows):
        self.calls += 1
        self.evaluations += len(rows)
        try:
            values = np.asarray(self._objective(rows), dtype=float).reshape(-1)
            if values.shape != (len(rows),):
                raise ValueError(f"objective returned {values.shape[0]} values for {len(rows)} designs")
        except EVALUATION_ERRORS as error:
            self.failed_calls += 1
            if len(self.errors) < ERRORS_KEPT:
                self.errors.append(f"{type(error).__name__}: {str(error)[:200]}")
            return None
        return values

    def __call__(self, rows):
        rows = np.asarray(rows, dtype=float).reshape(-1, self._dim)
        failed = np.zeros(len(rows), dtype=bool)
        values = self._attempt(rows)
        if values is None:
            values = np.full(len(rows), -np.inf)
            if len(rows) == 1:
                failed[0] = True
            else:
                for i in range(len(rows)):
                    single = self._attempt(rows[i:i + 1])
                    if single is None:
                        failed[i] = True
                    else:
                        values[i] = single[0]
            self.failed_designs += int(failed.sum())
        # a value a successful call returned non-finite is infeasible too, and counted apart from the failures
        self.non_finite += int(np.sum(~np.isfinite(values) & ~failed))
        return np.where(np.isfinite(values), values, -np.inf)

    def record(self):
        """Failure counts and the first ERRORS_KEPT error messages, for the run record."""
        return {"failed_calls": self.failed_calls, "failed_designs": self.failed_designs,
                "non_finite_values": self.non_finite, "errors": self.errors}


def search(space, objective, pool_rows, pool_values, seed, objective_name="eig"):
    """One round's search. `pool_rows` [n, d] and `pool_values` [n] are the pool table the pool path just computed, in the
    arm's criterion `objective_name`; `objective(rows [m, d]) -> values [m]` is the pool path's estimator of that criterion.
    Returns (chosen row, chosen value, record)."""
    if objective_name not in OBJECTIVES:
        raise ValueError(f"unknown design-search objective {objective_name!r}")
    pool_values = np.asarray(pool_values, dtype=float)
    pool_rows = np.asarray(pool_rows, dtype=float).reshape(len(pool_values), space.dim)
    ranked = sorted(range(len(pool_values)), key=lambda i: (-pool_values[i], _row(pool_rows[i])))
    best_pool = ranked[0]
    starts = [{"source": f"pool_rank_{k + 1}", "pool_index": int(i), "design": _row(pool_rows[i]),
               "value_pool_table": float(pool_values[i])} for k, i in enumerate(ranked[:STARTS_POOL])]
    if STARTS_RANDOM:
        rng = np.random.default_rng(seed)
        for k in range(STARTS_RANDOM):
            for tries in range(1, RANDOM_START_TRIES + 1):
                design = space.to_design(rng.random(space.dim))
                if space.valid(design):
                    starts.append({"source": f"random_{k + 1}", "design": _row(design), "tries": tries})
                    break
            else:
                starts.append({"source": f"random_{k + 1}", "design": None, "tries": RANDOM_START_TRIES,
                               "skipped": "no env-valid draw"})
    evaluate = _Evaluator(objective, space.dim)
    live_starts = [s for s in starts if s["design"] is not None]
    units = np.asarray([space.to_unit(s["design"]) for s in live_starts], dtype=float)
    values = evaluate(np.asarray([s["design"] for s in live_starts]))
    for s, v in zip(live_starts, values, strict=True):
        s["value"] = float(v) if np.isfinite(v) else None
    steps = np.full(len(units), INITIAL_STEP)
    paths = [[{"iteration": 0, "design": _row(space.to_design(u)), "value": float(v) if np.isfinite(v) else None,
               "step": INITIAL_STEP}] for u, v in zip(units, values, strict=True)]
    halvings = [0] * len(units)
    infeasible = 0
    iterations = 0
    for iteration in range(1, MAX_ITERATIONS + 1):
        live = [i for i in range(len(units)) if steps[i] >= space.min_step]
        if not live:
            break
        iterations = iteration
        candidates, owners = [], []
        for i in live:
            for axis in range(space.dim):
                for sign in (1.0, -1.0):
                    unit = units[i].copy()
                    unit[axis] = np.clip(unit[axis] + sign * steps[i], 0.0, 1.0)
                    if np.array_equal(unit, units[i]):
                        continue
                    design = space.to_design(unit)
                    if not space.valid(design):
                        infeasible += 1
                        continue
                    candidates.append(unit)
                    owners.append(i)
        scores = (evaluate(np.asarray([space.to_design(u) for u in candidates])) if candidates else np.empty(0))
        for i in live:
            mine = [j for j, owner in enumerate(owners) if owner == i and np.isfinite(scores[j])]
            best = max(mine, key=lambda j: scores[j], default=None)
            if best is not None and scores[best] > values[i]:
                units[i], values[i] = candidates[best], scores[best]
                paths[i].append({"iteration": iteration, "design": _row(space.to_design(units[i])),
                                 "value": float(values[i]), "step": float(steps[i])})
            else:
                steps[i] /= 2
                halvings[i] += 1
    ends = [space.to_design(u) for u in units]
    projected = [space.project(d) for d in ends]
    rescore_rows = np.asarray([pool_rows[best_pool]] + projected)
    rescored = evaluate(rescore_rows)
    table = [{"design": _row(pool_rows[best_pool]), "value": float(rescored[0]) if np.isfinite(rescored[0]) else None,
              "source": "pool_argmax"}]
    trajectories = []
    for k, (s, end, proj, v_end, v_proj) in enumerate(zip(live_starts, ends, projected, values, rescored[1:],
                                                          strict=True)):
        table.append({"design": _row(proj), "value": float(v_proj) if np.isfinite(v_proj) else None,
                      "source": s["source"]})
        trajectories.append({
            "start": s["source"], "path": paths[k], "moves": len(paths[k]) - 1, "halvings": halvings[k],
            "final_step_unit": float(steps[k]),
            "stop_reason": "step_below_resolution" if steps[k] < space.min_step else "iteration_budget",
            "end_relaxed": _row(end), "end_relaxed_value": float(v_end) if np.isfinite(v_end) else None,
            "projected": _row(proj), "projected_value": float(v_proj) if np.isfinite(v_proj) else None,
            "rounded": bool(not np.array_equal(end, proj)),
            "projected_edge_axes": space.edge_axes(proj), "projected_bound_axes": space.bound_axes(proj)})
    pool_argmax = {"pool_index": int(best_pool), "design": _row(pool_rows[best_pool]),
                   "value_pool_table": float(pool_values[best_pool]), "value_rescored": table[0]["value"]}
    scored_entries = [r for r in table if r["value"] is not None]
    if table[0]["value"] is None:
        # The pool argmax could not be re-scored, so nothing can be shown to be at least as good: observe the pool's
        # own choice, as the pool path would.
        decision = {"fallback": "pool_argmax_rescore_failed"}
        chosen = table[0]["design"]
        chosen_value, chosen_source, gain = None, "pool_argmax", None
    else:
        decision = choose_design([{"design": r["design"], "value": r["value"]} for r in scored_entries], "value")
        chosen = decision.pop("chosen_design")
        chosen_entry = next(r for r in scored_entries if r["design"] == chosen)
        chosen_value, chosen_source = float(chosen_entry["value"]), chosen_entry["source"]
        gain = float(chosen_value - table[0]["value"])
    record = {
        "mode": "ascent", "design_search": "ascent", "method": METHOD, "constants_sha256": constants_sha256(),
        "objective": objective_name, "seed": int(seed), "bounds": space.record(),
        "step_size": {"initial_unit": INITIAL_STEP, "min_unit": space.min_step, "rule": "halve on no strict improvement"},
        "max_iterations": MAX_ITERATIONS, "starts": starts, "trajectories": trajectories,
        "pool_argmax": pool_argmax, "final_table": table, "decision": decision,
        "chosen": {"design": chosen, "value": chosen_value, "source": chosen_source,
                   "in_pool": bool(any(np.array_equal(np.asarray(chosen, dtype=float), r) for r in pool_rows)),
                   "edge_axes": space.edge_axes(chosen), "bound_axes": space.bound_axes(chosen),
                   "on_edge": bool(space.edge_axes(chosen))},
        "gain_over_pool_argmax": gain, "improved": bool(gain is not None and gain > TIE_ATOL),
        "endpoints_on_edge": int(sum(bool(t["projected_edge_axes"]) for t in trajectories)),
        "endpoints": len(trajectories), "iterations": iterations, "objective_calls": evaluate.calls,
        "evaluations": evaluate.evaluations, "infeasible_neighbours_skipped": infeasible,
        "evaluation_failures": evaluate.record(),
    }
    return np.asarray(chosen, dtype=float), chosen_value, record


def exhaustive_record(objective_name, rows, chosen, reason):
    """The discrete case of the rule: the pool is the task's whole legal design set, so its argmax under the arm's
    criterion (the pool path's choice, unchanged) is the exhaustive optimum. Recorded in the same place as an ascent."""
    if objective_name not in OBJECTIVES:
        raise ValueError(f"unknown design-search objective {objective_name!r}")
    values = [float(r[objective_name]) for r in rows]
    index = next(i for i, r in enumerate(rows) if r["design"] == chosen)
    return {"mode": "exhaustive", "design_search": "ascent", "method": METHOD, "constants_sha256": constants_sha256(),
            "objective": objective_name, "reason": reason, "candidates": len(rows),
            "pool_argmax": {"pool_index": index, "design": chosen, "value_pool_table": values[index]},
            "chosen": {"design": chosen, "value": values[index], "source": "pool_argmax", "in_pool": True},
            "gain_over_pool_argmax": 0.0, "improved": False}
