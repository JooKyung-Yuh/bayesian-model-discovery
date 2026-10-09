"""The TRUE observation law of every BoxingGym task, and the KL divergence from it
to the Bayesian model averaging (BMA) predictive, KL(truth || BMA).

Extension of the death-only KL metric: death already records `kl`; this module
generalises that ONE number to the other eight tasks, at every checkpoint.
Eight of them record it unconditionally; the one whose measured cost exceeded
the 10%-of-cell-wall cost rule (`boxinggym_core.KL_OPTIONAL_TASKS`) records it
under `--kl-all-tasks`, the definitions below being identical either way.
Nothing else about a run changes - no existing metric, no rng stream, no
prediction, no environment source. The benchmark tree is read-only: every law
below is rebuilt RUNNER-SIDE from the environment instance's own public
attributes, exactly as `boxinggym_core.death_truth_pmf` already does.

The metric, for every task
--------------------------
At each checkpoint, on the SAME fixed evaluation set Q every other score uses
(`evaluation_x`, subset by `exclude_observed` when that policy is on),

    kl = mean over Q of  KL( p_q || q_q )

with `p_q` the task's true observation law at the query point `q` (rebuilt from
the world's own drawn parameters) and `q_q` the checkpoint pool's BMA
predictive at that same point. `expected_log_score = mean_Q sum_y p log q` is
recorded wherever the support is discrete. A KL that does not approach 0 is
expected where the true law is outside the engine's grammar; the quantity of
interest is its movement across rounds.

Per-task definitions (fixed 2026-09-18; see each builder)
------------------------------------------------------------------
death            Binomial(N, 1 - exp(-theta*t)) over {0..50} vs the BMA
                 log-PMF. Exact discrete sum. UNCHANGED from the death-only metric -
                 `death_law` rebuilds the identical array, and the runner
                 keeps calling `death_truth_pmf` on the existing code path.
survival         The GENERATING law Bernoulli(expit(t*exp(beta*m)*lambda0)),
                 not the environment's stored patient record: `step` returns a
                 value fixed at world-build time, so a repeated query is a
                 point mass; the KL is against the law that drew it.
hyperbolic       Bernoulli(epsilon + (1-2 epsilon) Phi((V1-V0)/alpha)).
irt              The GENERATING law Bernoulli(expit(gamma_q(alpha_s - beta_q)))
                 (same 2PL/3PL reading `_truth_irt` uses), for the same reason
                 as survival - `step` returns a stored matrix cell. When the
                 latent draw was not captured the law is None and `kl` stays
                 blank, exactly as `truth_mean` already behaves.
moral_machines   Bernoulli on the MODELLED {0,1} outcome (choice - 1), i.e.
                 expit(-(alpha + beta_group + x.beta)).
peregrines       Poisson(exp(alpha + b1 t + b2 t^2 + b3 t^3)), enumerated per
                 query up to K_q = min{K : F(K) >= 1 - 1e-12}; the enumerated
                 grid is 0..max_q K_q and the excluded tail is recorded as
                 `kl_truncated_mass`. No Monte Carlo.
dugongs          The FOLDED normal |Normal(m, 0.25)| with m = a - b*|l|^age
                 (the environment returns `abs(obs)`), against the pool's
                 lognormal/gamma mixture DENSITY, by Gauss-Legendre quadrature
                 on [max(0, |m| - 8s), |m| + 8s].
location         The PRE-ROUNDING Normal(signal_intensity(x), sigma) with the
                 environment's own sigma (0.5). `run_experiment` rounds the
                 observation to two decimals; that 0.01 lattice is a property
                 of the recorded observation, NOT of the law scored here (the
                 alternative - discretising the truth onto the lattice - was
                 considered and not adopted). Gauss-Legendre on mu +- 8 sigma.
emotion          Per emotion component j, the 9-cell categorical the
                 environment actually emits: `int()` truncation then clip to
                 1..9 (emotion.py:411-414), so cell 1 is (-inf, 2), cell k is
                 [k, k+1) for k = 2..8 and cell 9 is [9, inf). Cell
                 probabilities are exact Phi differences of the same latent
                 Normal(mean_j, sigma_j) `_truth_emotion` uses. The prediction
                 is that component's real-runner mixture over the SAME nine
                 cells, as a difference of mixture CDFs - the same
                 quantity the earlier quadrature rule integrated, computed exactly, so
                 emotion reports NEITHER `kl_quadrature_error` (there is no
                 coarser rule) NOR `kl_truncated_mass` (there is no window:
                 the end cells are the true tails). The eight components are
                 then averaged by the runner's existing composite rule. The unit
                 defect of the seventh input (`EMOTION_WIN_UNIT_DEFECT`) is
                 inherited by both sides equally, since both are evaluated at
                 the same `evaluation_x`.

Numerics (ChemBench's rule, `chembench_core.censored_normal_kl_quadrature`)
---------------------------------------------------------------------------
Every continuous integral is Gauss-Legendre, never Monte Carlo. The reported
value is the finer rule and `kl_quadrature_error` is |finer - coarser|: 32/64
nodes for a density interval (dugongs, location). The interval is the TRUTH's
own mean +- 8 standard deviations (the integrand is p-weighted, so the truth's
mass is what must be covered), and the truth mass that interval excludes is
recorded as `kl_truncated_mass`, never assumed to be zero.

emotion no longer uses this quadrature (changed 2026-09-18). Its
nine cell masses are differences of the pool's mixture CDF (`_cell_mass_model`,
on `real_inference.conditional_cdf`), which is the same quantity the quadrature obtained by
integrating the mixture density over each cell, but exact - so the two
diagnostics that describe an approximation, `kl_quadrature_error` and
`kl_truncated_mass`, are not reported for emotion at all, and `kl_method` says
`cell_cdf_exact`. The earlier quadrature stays here, private, as `_cell_log_model`:
the test that shows the two agree needs to compute both, and it is also what
makes the one place they must NOT agree visible - a pool spread well beyond the
truth's own +-8 sigma window keeps all its mass on the exact rule's end cells
and loses the part outside the window on the quadrature's.
"""

import numpy as np
from scipy.special import logsumexp, ndtr, xlogy
from scipy.stats import binom, poisson

from .positive_inference import log_predictive, particle_count, validate_pool
from .real_inference import conditional_cdf

# Gauss-Legendre node counts. The reported estimate is the FINER rule of each
# pair and the difference between the two is recorded as `kl_quadrature_error`
# (chembench_core.KL_QUADRATURE_BASE_NODES' own rule, restated here so this
# module's numerics are readable without the ChemBench file).
KL_QUADRATURE_BASE_NODES = 32
KL_QUADRATURE_CELL_NODES = 16
# The quadrature interval is the truth's own mean +- this many of the truth's
# own standard deviations.
KL_TRUNCATION_SIGMAS = 8.0
# peregrines' enumeration depth: the smallest K whose Poisson CDF covers all
# but this much of the true mass.
KL_POISSON_TAIL_MASS = 1e-12
# The emotion environment's Likert coding: `int()` then clip to 1..9, so the
# nine cells are (-inf, 2), [2, 3), ..., [8, 9), [9, inf).
EMOTION_CELL_EDGES = tuple(range(2, 10))
# The pool's mass in those nine cells is a difference of mixture
# CDFs, not an integral of the mixture density - the same quantity, computed
# exactly, so there is neither a quadrature error nor a truncation window to
# report. The label says which rule produced the recorded `kl`.
EMOTION_CELL_METHOD = "cell_cdf_exact"
# The seven interior cells are one Likert step wide and one 16-node rule
# integrates each of them to machine precision. The two END cells are not: they
# are the truncation window's whole tail, `KL_TRUNCATION_SIGMAS` standard
# deviations of the TRUTH plus the distance to the edge, and a pool density
# that is narrow and far from that edge puts essentially all of its mass in the
# last fraction of the interval - which one 16-node rule spread over the whole
# tail resolves badly (measured: a `kl_quadrature_error` of 0.02..0.2 against a
# `kl` near 3.5 on a dev emotion cell). Each end cell is therefore integrated
# as this many equal panels, each with the SAME 16/32-node rule an interior
# cell gets; at one panel per truncation sigma a panel is about one standard
# deviation wide, which at the emotion environment's own HalfNormal(1) noise is
# about one Likert step - the interior cells' own width.
KL_EMOTION_TAIL_PANELS = 8
# dugongs.py:255 `pm.Normal(..., sigma=0.25)` - the environment hard-codes it
# rather than storing it on the instance, so it is restated here (the same
# constant `_truth_dugongs` restates).
DUGONGS_OBSERVATION_SIGMA = 0.25
# The law fields carried per evaluation point; `subset_law` slices exactly
# these (never, say, `count_grid`'s outcome grid, whose length could coincide
# with |Q|).
LAW_QUERY_FIELDS = ("pmf", "excluded", "mean", "probability", "lower", "upper")
# Memory guard for one `log_predictive` call: the per-model
# [particle, query, outcome] array is built in blocks of at most this many
# array cells (8 M cells = 64 MB of float64) instead of all at once. Purely a
# working-set bound - the blocks are concatenated along the outcome axis, so
# the returned log-density is the same array a single call would return.
KL_PREDICTIVE_BLOCK_CELLS = 8_000_000


# --------------------------------------------------------------------------
# generic law constructors (no task knowledge)
# --------------------------------------------------------------------------


def bernoulli_law(probability):
    """[Q, 2] = [1 - p, p], the column order `count_predict`'s log-PMF uses."""
    p = np.asarray(probability, dtype=float)
    return {"kind": "pmf", "pmf": np.column_stack([1.0 - p, p]), "method": "discrete_exact"}


def poisson_law(rate):
    """Poisson(rate) enumerated to `KL_POISSON_TAIL_MASS`, per query."""
    mu = np.asarray(rate, dtype=float)
    depth = poisson.ppf(1.0 - KL_POISSON_TAIL_MASS, mu)
    excluded = poisson.sf(depth, mu)
    grid = np.arange(int(np.max(depth)) + 1, dtype=float)
    return {
        "kind": "count_grid",
        "grid": grid,
        "pmf": poisson.pmf(grid[None, :], mu[:, None]),
        "excluded": np.asarray(excluded, dtype=float),
        "method": f"poisson_enumeration_tail_{KL_POISSON_TAIL_MASS:g}",
    }


def normal_law(mean, sigma):
    """Normal(mean, sigma) - a density, integrated by quadrature."""
    return {
        "kind": "density",
        "family": "normal",
        "mean": np.asarray(mean, dtype=float),
        "sigma": float(sigma),
        "method": f"gauss_legendre_{KL_QUADRATURE_BASE_NODES}_{2 * KL_QUADRATURE_BASE_NODES}",
    }


def folded_normal_law(mean, sigma):
    """|Normal(mean, sigma)| on y > 0 - a density, integrated by quadrature."""
    return {
        "kind": "density",
        "family": "folded_normal",
        "mean": np.asarray(mean, dtype=float),
        "sigma": float(sigma),
        "method": f"gauss_legendre_{KL_QUADRATURE_BASE_NODES}_{2 * KL_QUADRATURE_BASE_NODES}",
    }


def likert_cells_law(mean, sigma):
    """The 9-cell law `clip(int(Normal(mean, sigma)), 1, 9)` induces.

    `mean` is [Q, component] and `sigma` is [component]; the returned cell
    probabilities are [Q, component, 9] and sum to 1 along the last axis by
    construction (the edge terms telescope).

    `lower`/`upper` are the RETIRED quadrature window: the pool's
    cell masses are now exact CDF differences over the whole line, so nothing
    on the scoring path truncates any more. They are still built, and still
    follow Q through `subset_law`, so that `_cell_log_model` - kept private for
    exactly this - can reproduce the earlier quadrature value for the comparison test.
    """
    m = np.asarray(mean, dtype=float)[:, :, None]
    s = np.asarray(sigma, dtype=float)[None, :, None]
    edges = np.asarray(EMOTION_CELL_EDGES, dtype=float)[None, None, :]
    upper = ndtr((edges - m) / s)
    probability = np.concatenate(
        [upper[:, :, :1], np.diff(upper, axis=-1), 1.0 - upper[:, :, -1:]], axis=-1
    )
    return {
        "kind": "cells",
        "probability": probability,
        "lower": np.minimum(m[:, :, 0], edges[0, 0, 0]) - KL_TRUNCATION_SIGMAS * s[:, :, 0],
        "upper": np.maximum(m[:, :, 0], edges[0, 0, -1]) + KL_TRUNCATION_SIGMAS * s[:, :, 0],
        "method": EMOTION_CELL_METHOD,
    }


# --------------------------------------------------------------------------
# per-task builders; registered on `TaskSpec.truth_law`
# --------------------------------------------------------------------------


def death_law(spec, environment, x):
    """Binomial(N, 1 - exp(-theta*t)) - death_process.py:280, :286, :319-323.

    The three lines below are `boxinggym_core.death_truth_pmf`'s own, in the
    same order and with the same intermediates, so the array is bit-identical
    to the one the existing death-only path already builds (pinned by test).
    `truth_mean`'s `N * -expm1(-theta*t)` is NOT reused here: dividing that back by N would
    round differently from `1 - exp(-theta*t)`.
    """
    env = environment.env
    times = np.asarray(x, dtype=float)
    p = 1.0 - np.exp(-float(env.theta) * times)
    counts = np.arange(int(env.N) + 1)
    return {
        "kind": "pmf",
        "pmf": binom.pmf(counts[None, :], int(env.N), p[:, None]),
        "method": "discrete_exact",
    }


def bernoulli_truth_mean_law(spec, environment, x):
    """The four binary tasks: the success probability IS `spec.truth_mean`.

    survival, hyperbolic, irt and moral_machines all declare
    `outcome_type="binary"`, so their `truth_mean` hook already returns
    P(Y = 1 | x) in the MODELLED outcome coding (moral_machines' -1 offset
    included). Reusing the hook is what keeps the two metrics on one formula.
    `None` (irt with the latent draw not captured) propagates as `None`.
    """
    probability = spec.truth_mean(environment, x)
    return None if probability is None else bernoulli_law(probability)


def peregrines_law(spec, environment, x):
    """Poisson(exp(alpha + b1 t + b2 t^2 + b3 t^3)) - peregrines.py:206-217, :267."""
    return poisson_law(spec.truth_mean(environment, x))


def dugongs_law(spec, environment, x):
    """|Normal(a - b*|l|^age, 0.25)| - dugongs.py:207-211, :255, :259.

    `truth_mean` returns the folded MEAN, which is not this law's location
    parameter, so `a`, `b` and `l` are read from the environment directly -
    the same three attributes `_truth_dugongs` reads.
    """
    env = environment.env
    mean = env.a - env.b * abs(env.l) ** np.asarray(x, dtype=float)
    return folded_normal_law(mean, DUGONGS_OBSERVATION_SIGMA)


def location_law(spec, environment, x):
    """Normal(signal_intensity(x), sigma) BEFORE run_experiment's rounding.

    location_finding.py:446-449 (`step` draws `normal(signal_intensity(xi),
    self.sigma)`), :439-444 (`signal_intensity`) and :382/:387 (`sigma`, the
    constructor default 0.5 stored on the instance); `run_experiment` rounds
    the returned draw to two decimals at :471, which this law deliberately does
    not model - see the module docstring.
    """
    return normal_law(spec.truth_mean(environment, x), float(environment.env.sigma))


def emotion_law(spec, environment, x):
    """The eight components' 9-cell Likert laws - emotion.py:324-331, :388-397, :411-414.

    The latent mean/sigma arithmetic is `_truth_emotion`'s, up to the point
    where that hook collapses the cells into their mean; `None` under the same
    missing-attribute condition.
    """
    env = environment.env
    names = ("alpha", "beta_win", "beta_PE", "beta_absPE", "sigma")
    if not all(hasattr(env, name) for name in names):
        return None
    rows = np.asarray(x, dtype=float)
    win = rows[:, 6]
    pe = win - np.sum(rows[:, :3] * rows[:, 3:6], axis=1)
    mean = (np.asarray(env.alpha) + win[:, None] * env.beta_win
            + pe[:, None] * env.beta_PE + np.abs(pe[:, None]) * env.beta_absPE)
    return likert_cells_law(mean, np.asarray(env.sigma))


def truth_law(spec, environment, evaluation_x):
    """The task's true observation law on `evaluation_x`, or None."""
    if spec.truth_law is None:
        return None
    return spec.truth_law(spec, environment, evaluation_x)


def subset_law(law, keep):
    """The same law restricted to a subset of Q (the `exclude_observed` rule)."""
    if law is None:
        return None
    return {key: value[keep] if key in LAW_QUERY_FIELDS else value for key, value in law.items()}


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def discrete_kl(truth, logpmf):
    """CORE's own formula: the `expected_log_score` and `kl` lines of `core_count.score_prediction`.

    `score_prediction` itself is called for death and cannot be called for any
    other task: its `mean_rmse` line hard-codes `np.arange(51)`, so a truth
    array with any other support width raises. The two lines that define `kl`
    and `expected_log_score` are reproduced here verbatim instead, which is
    what makes death's numbers reproducible from this module (pinned by test).
    """
    truth = np.asarray(truth, dtype=float)
    logpmf = np.asarray(logpmf, dtype=float)
    return {
        "kl": float(np.sum(xlogy(truth, truth) - truth * logpmf, axis=-1).mean()),
        "expected_log_score": float(np.sum(truth * logpmf, axis=-1).mean()),
    }


def _log_predictive_blocked(fits, inputs, outcomes):
    """`positive_inference.log_predictive`'s BMA log-density, in memory blocks.

    One call with every outcome column at once would build a
    [particle, query, outcome] array per model; for a paper-profile peregrines
    checkpoint that is gigabytes. The outcome axis is therefore split into
    blocks and the results concatenated - arithmetic per column is unchanged,
    since every column of `log_predictive`'s output depends on that column
    alone.
    """
    outcomes = np.asarray(outcomes, dtype=float)
    particles = max(particle_count(result) for result in fits)
    per_column = max(1, particles * len(outcomes))
    width = max(1, int(KL_PREDICTIVE_BLOCK_CELLS // per_column))
    if width >= outcomes.shape[1]:
        return log_predictive(fits, inputs, outcomes)[0]
    return np.concatenate(
        [log_predictive(fits, inputs, outcomes[:, start:start + width])[0]
         for start in range(0, outcomes.shape[1], width)],
        axis=1,
    )


def _gauss_legendre(lower, upper, nodes):
    """Nodes and weights of an `nodes`-point rule on each [lower, upper] row."""
    points, weights = np.polynomial.legendre.leggauss(nodes)
    half = 0.5 * (np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float))
    middle = 0.5 * (np.asarray(upper, dtype=float) + np.asarray(lower, dtype=float))
    return middle[..., None] + half[..., None] * points, half[..., None] * weights


def _density_log_truth(law, y):
    """log p at `y` for a `density` law, and the mass its interval excludes."""
    mean, sigma = law["mean"][:, None], law["sigma"]
    z = (y - mean) / sigma
    constant = -np.log(sigma * np.sqrt(2.0 * np.pi))
    if law["family"] == "normal":
        return constant - 0.5 * z**2
    folded = (y + mean) / sigma
    return constant + np.logaddexp(-0.5 * z**2, -0.5 * folded**2)


def _density_interval(law):
    """[max(0, |m| - 8s), |m| + 8s] for the folded law, m +- 8s for the Normal."""
    mean, sigma = law["mean"], law["sigma"]
    reach = KL_TRUNCATION_SIGMAS * sigma
    if law["family"] == "normal":
        return mean - reach, mean + reach
    centre = np.abs(mean)
    return np.maximum(0.0, centre - reach), centre + reach


def _density_excluded(law, lower, upper):
    """The TRUTH mass outside [lower, upper], in closed form."""
    mean, sigma = law["mean"], law["sigma"]
    inside = ndtr((upper - mean) / sigma) - ndtr((lower - mean) / sigma)
    if law["family"] == "folded_normal":
        inside = inside + ndtr((upper + mean) / sigma) - ndtr((lower + mean) / sigma)
    return 1.0 - inside


def _quadrature_kl(log_truth, log_model, weights):
    """mean_Q of int w * p * (log p - log q); p = 0 contributes 0, never nan."""
    truth = np.exp(log_truth)
    integrand = xlogy(truth, truth) - np.where(truth > 0.0, truth * log_model, 0.0)
    return float(np.sum(weights * integrand, axis=-1).mean())


def _cell_log_model(fits, inputs, law, component, nodes):
    """log of the pool's mass in each of the nine cells, [Q, 9], by QUADRATURE.

    The earlier quadrature rule, retired and kept private for its comparison test only -
    `_cell_mass_model` is what the scoring path calls now. The seven interior
    cells are one panel each; the two end cells are `KL_EMOTION_TAIL_PANELS`
    panels each (see that constant), spanning the law's `lower`/`upper`
    truncation window rather than the whole line - which is exactly the gap the
    exact rule closes. Every panel gets the same `nodes`-point rule, and a
    cell's mass is the log-sum of its own panels', so the nine returned columns
    are still the nine cells.
    """
    edges = np.asarray(EMOTION_CELL_EDGES, dtype=float)
    panels = KL_EMOTION_TAIL_PANELS
    head = np.linspace(law["lower"][:, component], edges[0], panels + 1, axis=-1)
    tail = np.linspace(edges[-1], law["upper"][:, component], panels + 1, axis=-1)
    interior = np.tile(edges, (len(inputs), 1))
    lower = np.concatenate([head[:, :-1], interior[:, :-1], tail[:, :-1]], axis=1)
    upper = np.concatenate([head[:, 1:], interior[:, 1:], tail[:, 1:]], axis=1)
    points, weights = _gauss_legendre(lower, upper, nodes)
    flat = points.reshape(len(inputs), -1)
    log_panel = logsumexp(
        _log_predictive_blocked(fits, inputs, flat).reshape(points.shape), b=weights, axis=-1
    )
    return np.concatenate(
        [
            logsumexp(log_panel[:, :panels], axis=-1)[:, None],
            log_panel[:, panels:panels + len(edges) - 1],
            logsumexp(log_panel[:, panels + len(edges) - 1:], axis=-1)[:, None],
        ],
        axis=1,
    )


def _cell_mass_model(fits, inputs):
    """The pool's mass in each of the nine cells, [Q, 9] - EXACT, no quadrature.

    Cell k's mass is `sum_m w_m mean_over_particles[F(edge_hi) - F(edge_lo)]`
    with `F` the member's own Normal / Student-t conditional CDF
    (`real_inference.conditional_cdf`, the same per-particle location and
    spread `conditional_loglik` builds) and `w_m` the pool weights
    `positive_inference.validate_pool` returns: the very mixture
    `log_predictive` averages, integrated instead of evaluated. The end cells
    are the true tails, (-inf, 2) and [9, inf), not a truncation window.

    Each difference is taken on the tail that carries it without cancellation -
    the lower CDF while the cell's upper edge is below the mixture's median,
    the survival function above it - because `1 - cdf` and a difference of two
    CDFs both round to 0 in the far tail, where the mass is small but not zero.
    """
    edges = np.tile(np.asarray(EMOTION_CELL_EDGES, dtype=float), (len(inputs), 1))
    weights = np.exp(validate_pool(fits))

    def mixture(tail):
        """Pool-weighted mixture CDF at the cell edges, on the requested tail; shape [Q, 8]."""
        return np.tensordot(
            weights,
            np.stack([conditional_cdf(fit, inputs, edges, tail).mean(axis=0) for fit in fits]),
            axes=(0, 0),
        )

    below, above = mixture("lower"), mixture("upper")
    ones, zeros = np.ones((len(inputs), 1)), np.zeros((len(inputs), 1))
    cell_lower = np.concatenate([zeros, below], axis=1)
    cell_upper = np.concatenate([below, ones], axis=1)
    cell_lower_sf = np.concatenate([ones, above], axis=1)
    cell_upper_sf = np.concatenate([above, zeros], axis=1)
    return np.where(cell_upper < 0.5, cell_upper - cell_lower, cell_lower_sf - cell_upper_sf)


def _cell_scores(truth, mass):
    """`discrete_kl`'s two sums, where a cell's model mass may be exactly 0.

    Identical arithmetic to `discrete_kl` wherever the mass is positive. A cell
    that carries truth mass and no pool mass is an infinite KL - the honest
    value of the quantity - rather than the `nan` that `0 * -inf` would give,
    and a cell with no truth mass contributes nothing, which is
    `_quadrature_kl`'s own convention for p = 0.
    """
    mass = np.asarray(mass, dtype=float)
    log_mass = np.log(mass, out=np.full(mass.shape, -np.inf), where=mass > 0.0)
    carried = np.multiply(truth, log_mass, out=np.zeros(np.shape(truth)), where=truth > 0.0)
    return {
        "kl": float(np.sum(xlogy(truth, truth) - carried, axis=-1).mean()),
        "expected_log_score": float(np.sum(carried, axis=-1).mean()),
    }


def kl_against_pool(law, prediction_or_fits, inputs=None, component=None):
    """KL(truth || BMA) on the evaluation set, plus this task's diagnostics.

    `prediction_or_fits` is the checkpoint's `prediction` dict (its `logpmf`
    is used, for the tasks that have one) or the pool's fits (every other
    task, evaluated at `inputs`). `component` selects one emotion component.
    """
    kind = law["kind"]
    if kind == "pmf":
        scores = discrete_kl(law["pmf"], prediction_or_fits["logpmf"])
        return {**scores, "kl_method": law["method"]}
    if kind == "count_grid":
        grid = np.tile(law["grid"], (len(inputs), 1))
        scores = discrete_kl(law["pmf"], _log_predictive_blocked(prediction_or_fits, inputs, grid))
        return {**scores, "kl_method": law["method"],
                "kl_truncated_mass": float(np.max(law["excluded"]))}
    if kind == "density":
        lower, upper = _density_interval(law)
        values = []
        for nodes in (KL_QUADRATURE_BASE_NODES, 2 * KL_QUADRATURE_BASE_NODES):
            points, weights = _gauss_legendre(lower, upper, nodes)
            log_model = _log_predictive_blocked(prediction_or_fits, inputs, points)
            values.append(_quadrature_kl(_density_log_truth(law, points), log_model, weights))
        return {
            "kl": values[-1],
            "kl_method": law["method"],
            "kl_truncated_mass": float(np.max(_density_excluded(law, lower, upper))),
            "kl_quadrature_error": abs(values[-1] - values[0]),
        }
    # The nine cell masses are exact, so neither `kl_quadrature_error`
    # nor `kl_truncated_mass` exists for emotion any more: there is no coarser
    # rule to differ from and no window outside which pool mass is discarded.
    truth = law["probability"][:, component, :]
    return {
        **_cell_scores(truth, _cell_mass_model(prediction_or_fits, inputs)),
        "kl_method": law["method"],
    }
