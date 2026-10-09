"""Design-selection policy arms for the CORE runner.

Nothing here touches a prior, a likelihood, a sequential Monte Carlo (SMC) setting
or a seed. The legacy information and disagreement arms read the SAME particle
bank the joint path already builds for one round. Space filling uses only
candidate geometry and public observation coordinates. Every argmax is taken by
the ONE registered decision authority,
`..eig.joint.choose_design` (TIE_ATOL 1e-12, smallest design wins). The joint
path itself (`count_inference.choose` -> `information_from_components` ->
`choose_design`) is invoked, never reimplemented and never edited.

Registered semantics (the definitions of the earlier implementation, kept unchanged):

  * `joint_eig` - `eig/joint.py` gives the registered
    definition `I((M, Phi); Y_x | H_t) = H[p(Y_x|H_t)] - sum_m w_m
    E_{Phi_m} H[p(Y_x|Phi_m, M_m)]`. This is today's CORE behaviour, unchanged.

  * `structure_eig` - the model term of the same chain rule:
    `I(M; Y|u) = H[sum_k w_k pbar_k(y|u)] - sum_k w_k H[pbar_k(y|u)]`, where
    `pbar_k` is model k's PARTICLE-AVERAGED predictive PMF. That is exactly what
    `count_inference.information_from_components` already returns as
    `structure`, computed from the same 51-outcome bank as the joint total, so
    this arm needs no new integrand and no new computation.

  * `parameter_eig` - derived, `joint_eig - structure_eig`; the earlier implementation
    emitted precisely the triple
    `(eig, structure_eig, parameter_eig)`, and the CORE rows already carry all
    three.

  * `point_disagreement` - deliberately NOT an `eig` column: `eig/joint.py:3`
    says "Registered definition (NOT point disagreement)". The definition comes
    from the acquisition plan:
    `sum_m w_m (pbar_m(x) - pbar(x))^2`, which uses the posterior mean only and
    throws the PMF shape away. Here `m_k(u)` is model k's particle-averaged
    posterior predictive MEAN (for a bounded count that is `sum_y y*pbar_k(y|u)`
    = `N*pbar_k`; for the positive/real path it is the predictive mean), and
    `mbar = sum_k w_k m_k`. Corollary: models that agree on the
    mean and differ only in dispersion give exactly 0 - see
    `test_design_policies.test_identical_means_zero_disagreement`.

  * `uniform_random` - the A=0 branch's own lottery, expression for expression
    (`core_count.py`: `acquisition_seed_for(seed, replicate, t)` then
    `np.random.default_rng(acquisition_seed).choice(designs)`). It runs on a
    dedicated generator, so the run's own RNG is untouched.

Expected predictive information gain (EPIG) requires explicit fixed target inputs
and uses the shared predictive information calculation. A policy applies at
design level 1 (expected information gain, EIG) or 2 (EPIG).
An A=0 cell keeps the fixed random-draw anchor.
"""

import numpy as np

from ..eig.joint import choose_design
from .count_inference import choose as joint_choose
from .count_inference import choose_epig, epig_acquisition_record, model_logweights
from .count_models import active_particle_rows, particle_logpmf

POLICIES = ("joint_eig", "structure_eig", "point_disagreement", "uniform_random", "epig", "space_filling")
DEFAULT_POLICY = "joint_eig"
SPACE_FILLING_DEFINITION = (
    "Maximize the minimum Euclidean distance to public observed designs after linear "
    "normalization by the declared physical input bounds, (x-low)/(high-low); "
    "with no observed designs maximize distance from the normalized center (0.5,...,0.5). "
    "Use the existing candidate pool and registered argmax tie rule, without a posterior or RNG."
)

# arm -> the column of the criteria table whose argmax the arm takes. The two
# registered names keep the registered columns.
OBJECTIVE_COLUMN = {
    "joint_eig": "eig",
    "epig": "epig",
    "structure_eig": "structure_eig",
    "point_disagreement": "point_disagreement",
    "uniform_random": "uniform_random",
    "space_filling": "min_normalized_distance",
}
# `target` recorded in acquisition/t{t}.json. joint_eig keeps the original string so
# the default run stays byte-identical; random keeps the A=0 branch's string.
TARGET = {
    "joint_eig": "joint_model_parameter",
    "epig": "predictive_target_points",
    "structure_eig": "model_structure",
    "point_disagreement": "point_disagreement",
    "uniform_random": "random",
    "space_filling": "space_filling",
}
# The plotting contract's `variant` tokens, emitted only when the policy differs
# from the cell default.
VARIANT_TOKEN = {
    "structure_eig": "design_structure_eig",
    "point_disagreement": "design_point_disagreement",
    "uniform_random": "design_uniform_random",
    "space_filling": "design_space_filling",
}


def applies_to_cell(cell):
    """A policy applies to design level 1 (EIG) or 2 (EPIG)."""
    return str(cell)[2] in ("1", "2")


def check_policy(cell, policy):
    """Reject an unregistered policy, and any policy on an A=0 anchor cell."""
    if policy not in POLICIES:
        raise ValueError(f"unregistered design policy {policy!r}; registered: {list(POLICIES)}")
    if (str(cell)[2] == "2") != (policy == "epig"):
        raise ValueError("design policy epig requires cell[2]=2, and cell[2]=2 requires epig")
    if not applies_to_cell(cell) and policy != DEFAULT_POLICY:
        raise ValueError(
            f"design policy {policy!r} needs an acquisition bit: cell {cell} chooses its designs "
            "by the fixed random draw and is an unchanged anchor"
        )
    return applies_to_cell(cell)


def variant_token(policy, applies):
    """The `design_*` token for the points table, empty for the cell default."""
    return VARIANT_TOKEN[policy] if applies and policy not in (DEFAULT_POLICY, "epig") else ""


def random_draw(designs, seed):
    """The A=0 branch's own draw, expression for expression, plus its index.

    The index is recovered by exact equality against the grid the value came
    from, so nothing is assumed about how `Generator.choice` maps its internal
    draw onto a position. A dedicated generator is built here and discarded, so
    the run's own RNG is neither read nor advanced.
    """
    designs = np.asarray(designs, dtype=float)
    if designs.ndim != 1 or not len(designs):
        raise ValueError("the random arm needs a nonempty one-dimensional design grid")
    value = float(np.random.default_rng(seed).choice(designs))
    matches = np.flatnonzero(designs == value)
    if len(matches) != 1:
        raise ValueError("the design grid must hold each value exactly once for the random arm")
    return value, int(matches[0])


def predictive_means(banks, n_trials):
    """`m_k(u) = sum_y y * pbar_k(y|u)` per [model, design] from the count banks.

    Each bank is `[particle, design, outcome]`, so `bank.mean(axis=0)` is model
    k's particle-averaged predictive PMF - the same `pbar_k` the structure term
    integrates. Taking the mean off the PMF keeps this family-agnostic: no
    family's mean formula is written down a second time here.
    """
    outcomes = np.arange(n_trials + 1)
    means = []
    for bank in banks:
        bank = np.asarray(bank, dtype=float)
        if bank.ndim != 3 or bank.shape[-1] != len(outcomes):
            raise ValueError("each bank must be [particle, design, outcome] over 0..n_trials")
        means.append(bank.mean(axis=0) @ outcomes)
    return np.stack(means)


def point_disagreement(means, weights):
    """`sum_k w_k (m_k(u) - mbar(u))^2` per design (the acquisition plan's definition).

    Posterior mean only: two models with the same mean and different dispersion
    contribute nothing, which is the corollary the acquisition plan registers.
    """
    means, weights = np.asarray(means, dtype=float), np.asarray(weights, dtype=float)
    if means.ndim != 2 or means.shape[0] != len(weights):
        raise ValueError("means must be [model, design] with one weight per model")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("model weights must be nonnegative and normalized")
    if not np.all(np.isfinite(means)):
        raise ValueError("non-finite predictive mean; no clipping fallback")
    centred = means - weights @ means
    return weights @ centred**2


def structure_eig_from_draws(component_logpdfs, mixture_logpdf, model_of_draw, weights):
    """Sample-based `I(M;Y|u)` for the positive/real path (no runner uses it yet).

    The joint estimator there (`positive_inference.joint_eig`) averages
    `log p(y|theta_i, M_k) - log sum_j w_j p_j(y|x)` over draws. Replacing the
    per-particle conditional with model k's own PARAMETER-MARGINAL conditional -
    which `positive_inference.log_predictive` already returns as its second
    value - turns the same estimator into the model term of the same chain rule,
    at zero extra sampling. `component_logpdfs` is `[model, design, draw]`,
    `mixture_logpdf` is `[design, draw]`, `model_of_draw` is `[draw]`.
    """
    component_logpdfs = np.asarray(component_logpdfs, dtype=float)
    mixture_logpdf = np.asarray(mixture_logpdf, dtype=float)
    model_of_draw = np.asarray(model_of_draw, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if component_logpdfs.ndim != 3 or component_logpdfs.shape[0] != len(weights):
        raise ValueError("component log densities must be [model, design, draw]")
    if mixture_logpdf.shape != component_logpdfs.shape[1:]:
        raise ValueError("mixture log density must be [design, draw]")
    if model_of_draw.shape != (component_logpdfs.shape[2],) or np.any(
        (model_of_draw < 0) | (model_of_draw >= len(weights))
    ):
        raise ValueError("one valid generating model index per draw is required")
    draws = np.arange(len(model_of_draw))
    contributions = component_logpdfs[model_of_draw, :, draws].T - mixture_logpdf
    if not np.all(np.isfinite(contributions)):
        raise ValueError("non-finite structure information contribution")
    return contributions.mean(axis=1)


def _banks(fits, designs, seed, posterior_samples, interface, n_trials):
    """The bank `count_inference.choose` builds: same generator, same seed, same
    per-fit subsample, same order.

    `choose` does not hand its bank back, so the disagreement column rebuilds it
    from the same seed. The two banks are the same bank; that is pinned by
    `test_design_policies.test_bank_reproduces_the_joint_rows`, which recomputes
    `information_from_components` on this bank and requires it to equal
    `choose`'s own rows exactly. Only the disagreement column ever reads it -
    the joint decision is `choose`'s, not this function's.
    """
    rng = np.random.default_rng(seed)
    banks = []
    for result in fits:
        size = len(next(iter(result.particles.values())))
        indices = rng.choice(size, size=min(size, posterior_samples), replace=False)
        subset = {k: v[indices] for k, v in result.particles.items()}
        # The same bank as `count_inference.choose`, so it drops the
        # same zero-weight particles the same way - the two must stay one bank.
        banks.append(
            np.exp(
                active_particle_rows(
                    particle_logpmf(result.hypothesis, subset, designs, n_trials, interface)
                )
            )
        )
    return banks


def criteria_table(fits, designs, seed, posterior_samples=128, interface=None, n_trials=50):
    """Every arm's objective for every candidate design, computed once.

    Returns `(rows, table, chosen_by_joint, draw)`. `rows` is exactly what
    `count_inference.choose` returned - four keys, unchanged, so the acquisition
    record keeps its original shape byte for byte. `table` is `rows` plus the two new
    columns: `point_disagreement`, and `uniform_random` as a 0/1 indicator of
    the lottery's own winner, so that EVERY arm is one argmax over one column
    under one tie rule.

    Computing the table once is the registered discipline (`eig/arms.py`
    `select`: "Recomputing it per arm would measure the same state twice").
    """
    chosen_by_joint, rows = joint_choose(
        fits, designs, seed, posterior_samples, interface, n_trials
    )
    spread = point_disagreement(
        predictive_means(_banks(fits, designs, seed, posterior_samples, interface, n_trials),
                         n_trials),
        np.exp(model_logweights(fits)),
    )
    value, index = random_draw(designs, seed)
    table = [
        {**row, "point_disagreement": float(s), "uniform_random": float(i == index)}
        for i, (row, s) in enumerate(zip(rows, spread, strict=True))
    ]
    return rows, table, chosen_by_joint, {"value": value, "index": index}


def select_space_filling(designs, observed_designs, bounds):
    """Maximin on an existing pool in linearly normalized physical coordinates.

    Scalar pools may be one-dimensional; vector pools are [candidate, input].
    Bounds are declared [input, (low, high)] pairs, never inferred from the pool
    or observed history. Coordinates are neither clipped nor transformed by a
    sampling distribution. An empty history uses the normalized center.
    """
    limits = np.asarray(bounds, dtype=float)
    if limits.shape == (2,):
        limits = limits[None, :]
    if (limits.ndim != 2 or limits.shape[1] != 2 or not len(limits)
            or not np.all(np.isfinite(limits)) or np.any(limits[:, 1] <= limits[:, 0])):
        raise ValueError("space_filling needs finite increasing declared bounds per input")
    pool = np.asarray(designs, dtype=float)
    scalar = pool.ndim == 1
    if scalar:
        pool = pool[:, None]
    if (pool.ndim != 2 or not len(pool) or pool.shape[1] != len(limits)
            or not np.all(np.isfinite(pool))):
        raise ValueError("space_filling needs a nonempty finite candidate matrix matching bounds")
    observed = np.asarray([] if observed_designs is None else observed_designs, dtype=float)
    if observed.shape == (0,):
        observed = np.empty((0, len(limits)))
    elif observed.ndim == 1:
        observed = observed[:, None] if len(limits) == 1 else observed[None, :]
    if (observed.ndim != 2 or observed.shape[1] != len(limits)
            or not np.all(np.isfinite(observed))):
        raise ValueError("space_filling needs finite observed coordinates matching bounds")
    if (np.any(pool < limits[:, 0]) or np.any(pool > limits[:, 1])
            or np.any(observed < limits[:, 0]) or np.any(observed > limits[:, 1])):
        raise ValueError("space_filling coordinates must lie within declared bounds")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        width = limits[:, 1] - limits[:, 0]
        if not np.all(np.isfinite(width)):
            raise ValueError("space_filling declared widths must be finite")
        normalized = (pool - limits[:, 0]) / width
        references = ((observed - limits[:, 0]) / width if len(observed)
                      else np.full((1, len(limits)), .5))
        if not np.all(np.isfinite(normalized)) or not np.all(np.isfinite(references)):
            raise ValueError("space_filling normalized coordinates must be finite")
        distances = np.linalg.norm(normalized[:, None, :] - references[None, :, :], axis=2).min(1)
    if not np.all(np.isfinite(distances)):
        raise ValueError("space_filling distances must be finite")
    key = OBJECTIVE_COLUMN["space_filling"]
    rows = [{"design": float(row[0]) if scalar else row.tolist(), key: float(distance)}
            for row, distance in zip(pool, distances, strict=True)]
    decision = choose_design(rows, key)
    chosen = decision["chosen_design"]
    index = next(i for i, row in enumerate(rows) if row["design"] == chosen)
    record = {
        "design_policy": "space_filling", "objective_column": key, "target": TARGET["space_filling"],
        **decision, "chosen_index": index, "chosen_value": decision["top_value"],
        "criteria": rows, "criteria_columns": [key], "definition": SPACE_FILLING_DEFINITION,
        "normalization_bounds": limits.tolist(), "normalization_rule": "(x-low)/(high-low)",
        "observed_design_count": len(observed),
        "empty_history_rule": "farthest_from_normalized_center",
        "uses_posterior": False, "uses_rng": False,
        "reason": f"argmax of {key}; tie rule {decision['tie_rule']}",
    }
    return chosen, rows, record


def select(fits, designs, seed, policy=DEFAULT_POLICY, posterior_samples=128, interface=None,
           n_trials=50, *, target_points=None, observed_designs=None, bounds=None):
    """Choose one design under `policy`; return `(chosen, rows, record)`.

    Legacy arms preserve the original acquisition rows and full shared criteria table.
    EPIG records predictive information; space filling records only normalized
    geometric distances and does not access fits or a particle bank. The policy
    record includes the chosen index, objective and registered tie decision.
    """
    if policy not in POLICIES:
        raise ValueError(f"unregistered design policy {policy!r}; registered: {list(POLICIES)}")
    if policy == "space_filling":
        if bounds is None:
            raise ValueError("space_filling requires explicit declared bounds")
        return select_space_filling(designs, observed_designs, bounds)
    if policy == "epig":
        if target_points is None:
            raise ValueError("EPIG requires explicit fixed target_points")
        chosen, rows = choose_epig(fits, designs, seed, posterior_samples, interface, n_trials,
                                   target_points=target_points)
        return chosen, rows, epig_acquisition_record(chosen, rows, seed)["policy"]
    rows, table, chosen_by_joint, draw = criteria_table(
        fits, designs, seed, posterior_samples, interface, n_trials
    )
    key = OBJECTIVE_COLUMN[policy]
    decision = choose_design(table, key)
    chosen = decision["chosen_design"]
    index = next(i for i, row in enumerate(table) if row["design"] == chosen)
    # Two guards on the arms that must agree with an existing path bit for bit.
    if policy == DEFAULT_POLICY and chosen != chosen_by_joint:
        raise ValueError("the joint arm must reproduce the joint path's own choice")
    if policy == "uniform_random" and (chosen != draw["value"] or index != draw["index"]):
        raise ValueError("the random arm must reproduce the A=0 branch's own draw")
    if policy == "uniform_random":
        reason = (
            f"uniform draw over {len(table)} candidate designs from the dedicated acquisition "
            f"stream (seed {seed}); recorded as a 0/1 column so the same argmax and tie rule "
            "reproduce it, and the run's own RNG is not consumed"
        )
    else:
        reason = (
            f"argmax of {key} over {len(table)} candidate designs; "
            f"{decision['n_tied_at_max']} within tie_atol {decision['tie_atol']:g} of the "
            f"maximum {decision['top_value']!r}; tie rule {decision['tie_rule']}"
        )
    record = {
        "design_policy": policy,
        "objective_column": key,
        "target": TARGET[policy],
        "chosen_design": chosen,
        "chosen_index": index,
        "chosen_value": decision["top_value"],
        "n_tied_at_max": decision["n_tied_at_max"],
        "tie_rule": decision["tie_rule"],
        "tie_atol": decision["tie_atol"],
        "reason": reason,
        "random_draw": draw["value"],
        "random_draw_index": draw["index"],
        "criteria": table,
        "criteria_columns": ["eig", "structure_eig", "parameter_eig", "point_disagreement",
                             "uniform_random"],
        "criteria_note": (
            "one bank per round, shared by every column: eig/structure_eig/parameter_eig from "
            "count_inference.information_from_components (eig == the joint I(M,Phi;Y|u)), "
            "point_disagreement from the same bank's per-model predictive means, uniform_random "
            "a 0/1 indicator of the dedicated-stream draw"
        ),
        "model_weights": [float(v) for v in np.exp(model_logweights(fits))],
        "active_ids": [result.hypothesis.model_id for result in fits],
    }
    return chosen, rows, record
