"""Small black-box vector likelihood path, reusing PyMC sequential Monte Carlo (SMC)
and pool weights."""

import time
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc.pytensorf import floatX
from pymc.smc.kernels import IMH, MH, Pearson
from pytensor.graph.basic import io_toposort
from pytensor.graph.op import Op
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

from .count_inference import SMCSettings, model_logweights, run_with_fit_timeout
from .neuron_forward import BATCH_FORWARDS, clear_prefill, prefill_stats

FORWARD_BACKENDS = ("native", "batch")
#: How a fit mutates its particles. `imh_adaptive` is PyMC's default independent
#: Metropolis-Hastings with its correlation-based step count - what every `dev`/`paper` fit has
#: always used. `mh_fixed_steps` is the kernel added for the MDA comparison: a random walk with
#: exactly `MH_STEPS_PER_STAGE` moves per tempering stage and an explicit ESS threshold.
KERNEL_POLICIES = ("imh_adaptive", "mh_fixed_steps")
MH_STEPS_PER_STAGE = 3
MH_ESS_THRESHOLD = 0.5


@dataclass
class VectorFit:
    """One SMC fit of a hypothesis: posterior particles, log evidence and diagnostics."""

    hypothesis: object
    particles: dict
    log_evidence: float
    per_chain_log_evidence: list
    diagnostics: dict
    seconds: float


class ForwardOp(Op):
    """Serializable native forward operation for PyMC's worker processes."""

    itypes: ClassVar = [pt.dscalar]
    otypes: ClassVar = [pt.dmatrix]

    def __init__(self, hypothesis, designs, forward, shape):
        self.hypothesis = hypothesis
        self.designs = tuple(designs)
        self.forward = forward
        self.shape = shape

    def perform(self, node, inputs, outputs):
        """Evaluate the forward at one conductance for every design, as a (designs, features)
        matrix."""
        conductance = float(inputs[0])
        if not np.isfinite(conductance) or conductance <= 0:
            raise ValueError("invalid transformed conductance")
        value = np.asarray(
            [self.forward(self.hypothesis, [conductance], d) for d in self.designs],
            dtype=np.float64,
        )
        if value.shape != self.shape or not np.all(np.isfinite(value)):
            raise ValueError("invalid forward shape or non-finite output")
        outputs[0][0] = value


class _BatchPrefill:
    """The particle-cloud prefill shared by every batch kernel, as a mixin.

    PyMC evaluates the likelihood one particle at a time - once per particle when the kernel is
    initialised, and once per proposal inside every `mutate` step - and each of those evaluations
    reaches `ForwardOp.perform`, which integrates one HH trace. Nothing about the algorithm, the
    values or the random-number consumption changes here: before each of those two loops a batch
    kernel computes the WHOLE cloud's features in a single vectorised call and deposits them in
    `neuron_forward`'s prefill cache, which the per-particle calls then read instead of
    integrating again.

    The prefill key must be the exact float the operation receives, so it is computed by a
    pytensor function compiled from the SAME graph expression the operation is applied to -
    never by taking `np.exp` here, which would miss by an ulp and silently fall back to the
    native path. A miss is answered natively (the value stays right) and counted; a batch fit
    asserts the count is zero.

    This was split out of `BatchIMH` unchanged so `BatchFixedStepsMH` can share it. The
    mixin carries no `mutate`: each kernel keeps its own copy of PyMC's, plus the one prefill
    line, which is what makes the deviation from the upstream algorithm readable line by line.
    Every method here defers to the kernel class it is mixed in front of, so the mixin must come
    FIRST in the bases.
    """

    def _forward_binding(self):
        """The (operation, value-vars -> operation input) pair for this model's forward."""
        if getattr(self, "_batch_binding", None) is None:
            nodes = [node for node in io_toposort([], [self.model.datalogp])
                     if isinstance(node.op, ForwardOp)]
            if len(nodes) != 1:
                raise ValueError("batch forward requires exactly one forward operation")
            if nodes[0].op.forward not in BATCH_FORWARDS:
                raise ValueError("batch forward backend requires the native feature forward")
            self._batch_binding = (
                nodes[0].op,
                pytensor.function(self.model.value_vars, nodes[0].inputs[0],
                                  on_unused_input="ignore"),
            )
        return self._batch_binding

    def _prefill(self, rows):
        """Prefill every design's features for a cloud given as per-variable value rows."""
        operation, forward_input = self._forward_binding()
        # The prefill that fills the cache THIS model's forward reads: the two feature spaces
        # keep separate rows, so filling the wrong one would miss on every lookup.
        prefill = BATCH_FORWARDS[operation.forward]
        conductances = np.asarray([float(forward_input(*row)) for row in rows])
        for design in operation.designs:
            prefill(operation.hypothesis, conductances, design)

    def _prefill_raveled(self, particles):
        """Prefill for a (draws, size) particle matrix in PyMC's raveled value-var layout."""
        rows = []
        for particle in particles:
            values, offset = [], 0
            for variable in self.variables:
                shape, size = self.var_info[variable.name]
                values.append(particle[offset:offset + size].reshape(shape))
                offset += size
            rows.append(values)
        self._prefill(rows)

    def initialize_population(self):
        """Draw the initial cloud as the kernel does, then reset the cache and prefill it."""
        population = super().initialize_population()
        # A fresh worker process starts from an empty cache, and this is also where a reused
        # one is reset, so `forward_cache_misses` counts exactly this fit.
        clear_prefill()
        self._prefill([[population[variable.name][i] for variable in self.variables]
                       for i in range(self.draws)])
        return population

    def sample_stats(self):
        """The kernel's per-stage stats plus this process's forward cache miss count."""
        # The only channel a worker process has back to the parent: `sample_settings()` survives
        # for chain 0 alone, while `sample_stats()` is collected per stage and per chain.
        stats = super().sample_stats()
        stats.update({"forward_cache_misses": int(prefill_stats()["misses"])})
        return stats


class BatchIMH(_BatchPrefill, IMH):
    """PyMC's IMH kernel with the particle cloud forward-simulated in one call per stage.

    `mutate` below is PyMC 5.13.1's `IMH.mutate` verbatim plus one line; everything else comes
    from `_BatchPrefill`, which explains what that line does and why the key is computed the way
    it is.

    The class lives at module top level because `pm.sample_smc` cloudpickles the kernel class to
    its worker processes; the cache and its counters are per-process, and each chain reports its
    own miss count through `sample_stats`.
    """

    def mutate(self):
        """Independent Metropolis-Hastings perturbation."""
        self.n_steps = 1
        old_corr = 2
        corr = Pearson(self.tempered_posterior)
        ac_ = []
        while True:
            log_R = np.log(self.rng.random(self.draws))
            # The proposal is independent from the current point.
            # We have to take that into account to compute the Metropolis-Hastings acceptance
            # We first compute the logp of proposing a transition to the current points.
            # This variable is updated at the end of the loop with the entries from the accepted
            # transitions, which is equivalent to recomputing it in every iteration of the loop.
            proposal = floatX(self.proposal_dist.rvs(size=self.draws, random_state=self.rng))
            proposal = proposal.reshape(len(proposal), -1)
            # To do that we compute the logp of moving to a new point
            forward_logp = self.proposal_dist.logpdf(proposal)
            # And to going back from that new point
            backward_logp = self.proposal_dist.logpdf(self.tempered_posterior)
            self._prefill_raveled(proposal)  # the one added line; the rest is PyMC 5.13.1's
            ll = np.array([self.likelihood_logp_func(prop) for prop in proposal])
            pl = np.array([self.prior_logp_func(prop) for prop in proposal])
            proposal_logp = pl + ll * self.beta
            accepted = log_R < (
                (proposal_logp + backward_logp) - (self.tempered_posterior_logp + forward_logp)
            )

            self.tempered_posterior[accepted] = proposal[accepted]
            self.tempered_posterior_logp[accepted] = proposal_logp[accepted]
            self.prior_logp[accepted] = pl[accepted]
            self.likelihood_logp[accepted] = ll[accepted]
            ac_.append(accepted)
            self.n_steps += 1

            pearson_r = corr.get(self.tempered_posterior)
            if np.mean((old_corr - pearson_r) > self.correlation_threshold) > 0.9:
                old_corr = pearson_r
            else:
                break

        self.acc_rate = np.mean(ac_)


class FixedStepsMH(MH):
    """PyMC's random-walk MH kernel with a FIXED number of mutation steps per tempering stage.

    PyMC 5.13.1's `MH.mutate` repeats its Metropolis step until the Pearson correlation between
    the stage's starting cloud and the current one stops falling by more than
    `correlation_threshold` across 90 % of the dimensions - so the step count is adaptive and
    unstated. The MDA comparison this kernel exists for prescribes a fixed number of random-walk
    moves per stage instead, so `mutate` below is that method with its `while True:` and the
    correlation test replaced by `for _ in range(self.steps_per_stage):`, and nothing else
    touched: the same proposal, the same acceptance rule, and the same random-number consumption
    per step (one uniform block for `log_R`, then one proposal block) in the same order.
    `setup_kernel`, `tune` and `resample` are inherited untouched, so the 0.234-target proposal
    scale adaptation and the tempered-posterior MVN proposal both still run.

    `n_steps` is the number of steps actually taken. PyMC's own counter starts at 1 and is
    incremented once per iteration, so it reports one more than it ran; here the count is the
    count, and `sample_settings`'s `_n_tune` reports it. That method also drops the inherited
    `correlation_threshold`, which this kernel never reads: a recorded setting that decided
    nothing is how a reader concludes the wrong thing about how a fit was mutated.
    """

    def __init__(self, *args, steps_per_stage=MH_STEPS_PER_STAGE, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(steps_per_stage, bool) or not isinstance(steps_per_stage, int) or steps_per_stage < 1:
            raise ValueError("steps_per_stage must be a positive integer")
        self.steps_per_stage = steps_per_stage

    def mutate(self):
        """Metropolis-Hastings perturbation, exactly `steps_per_stage` times."""
        self.n_steps = self.steps_per_stage
        ac_ = []
        for _ in range(self.steps_per_stage):
            log_R = np.log(self.rng.random(self.draws))
            proposal = floatX(
                self.tempered_posterior
                + self.proposal_dist(num_draws=self.draws, rng=self.rng)
                * self.proposal_scales[:, None]
            )
            ll = np.array([self.likelihood_logp_func(prop) for prop in proposal])
            pl = np.array([self.prior_logp_func(prop) for prop in proposal])

            proposal_logp = pl + ll * self.beta
            accepted = log_R < (proposal_logp - self.tempered_posterior_logp)

            self.tempered_posterior[accepted] = proposal[accepted]
            self.prior_logp[accepted] = pl[accepted]
            self.likelihood_logp[accepted] = ll[accepted]
            self.tempered_posterior_logp[accepted] = proposal_logp[accepted]
            ac_.append(accepted)

        self.chain_acc_rate = np.mean(ac_, axis=0)

    def sample_settings(self):
        """The kernel's settings with `steps_per_stage` added and the unused threshold dropped."""
        stats = super().sample_settings()
        stats.pop("correlation_threshold", None)
        stats["steps_per_stage"] = self.steps_per_stage
        return stats


class BatchFixedStepsMH(_BatchPrefill, FixedStepsMH):
    """`FixedStepsMH` with the particle cloud forward-simulated in one call per step.

    Exactly the relationship `BatchIMH` has to PyMC's `IMH`: `mutate` below is
    `FixedStepsMH.mutate` plus one line, and `_BatchPrefill` supplies the rest.
    """

    def mutate(self):
        """Metropolis-Hastings perturbation, exactly `steps_per_stage` times."""
        self.n_steps = self.steps_per_stage
        ac_ = []
        for _ in range(self.steps_per_stage):
            log_R = np.log(self.rng.random(self.draws))
            proposal = floatX(
                self.tempered_posterior
                + self.proposal_dist(num_draws=self.draws, rng=self.rng)
                * self.proposal_scales[:, None]
            )
            self._prefill_raveled(proposal)  # the one added line; the rest is FixedStepsMH's
            ll = np.array([self.likelihood_logp_func(prop) for prop in proposal])
            pl = np.array([self.prior_logp_func(prop) for prop in proposal])

            proposal_logp = pl + ll * self.beta
            accepted = log_R < (proposal_logp - self.tempered_posterior_logp)

            self.tempered_posterior[accepted] = proposal[accepted]
            self.prior_logp[accepted] = pl[accepted]
            self.likelihood_logp[accepted] = ll[accepted]
            self.tempered_posterior_logp[accepted] = proposal_logp[accepted]
            ac_.append(accepted)

        self.chain_acc_rate = np.mean(ac_, axis=0)


def _forward_cache_misses(data):
    """Total native fallbacks over every chain of a batch fit (0 when the prefill was complete)."""
    from ..inference.smc import _chain_units

    if "forward_cache_misses" not in data.sample_stats:
        raise ValueError("batch fit reported no forward cache statistics")
    raw = np.asarray(data.sample_stats["forward_cache_misses"].values, dtype=object)
    total = 0
    for unit in _chain_units(raw, int(data.posterior.sizes["chain"])):
        # Per stage the kernel reports the cache miss count SO FAR, so the chain's total is its
        # last (equivalently largest) entry, not the sum over stages.
        values = [int(value) for value in np.asarray(unit, dtype=object).ravel()]
        total += max(values, default=0)
    return total


def _mh_accept_rates(data):
    """Per-chain, per-stage acceptance rates of a random-walk MH fit.

    PyMC's IMH kernel reports `accept_rate` in `sample_stats`, which
    `inference.smc._extract_diagnostics` reads; its MH kernel reports the same quantity as
    `mean_accept_rate` (kernels.py:553), so a `FixedStepsMH` fit would otherwise record an
    empty `accept_rate`. Read that variable with the same chain mapping."""
    from ..inference.smc import _chain_units

    if "mean_accept_rate" not in data.sample_stats:
        return []
    raw = np.asarray(data.sample_stats["mean_accept_rate"].values, dtype=object)
    return [[float(value) for value in np.asarray(unit, dtype=object).ravel()
             if np.isfinite(float(value))]
            for unit in _chain_units(raw, int(data.posterior.sizes["chain"]))]


def _check_forward_backend(forward_backend, forward):
    if forward_backend not in FORWARD_BACKENDS:
        raise ValueError("forward_backend must be native or batch")
    if forward_backend == "batch" and forward not in BATCH_FORWARDS:
        raise ValueError("batch forward backend requires the native feature forward")


def _check_feature_scales(feature_scales, features):
    """One positive, finite noise scale per feature, as a float array."""
    scales = np.asarray(feature_scales, dtype=float)
    if scales.shape != (features,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("feature_scales must be one positive finite scale per feature")
    return scales


def _kernel_options(kernel_policy, forward_backend):
    """The kernel keyword arguments `pm.sample_smc` gets, and nothing more.

    `imh_adaptive` keeps its exact call from before the fixed-step kernel was added - no
    keyword at all on the native path - so a `dev`/`paper` fit is unchanged down to the
    kernel's own defaults.
    """
    if kernel_policy not in KERNEL_POLICIES:
        raise ValueError("kernel_policy must be imh_adaptive or mh_fixed_steps")
    if kernel_policy == "imh_adaptive":
        return {"kernel": BatchIMH} if forward_backend == "batch" else {}
    return {
        "kernel": BatchFixedStepsMH if forward_backend == "batch" else FixedStepsMH,
        # Stated rather than left to PyMC's default, because it is a declared setting of this
        # profile: the ESS target that decides how many tempering stages there are.
        "threshold": MH_ESS_THRESHOLD,
        "steps_per_stage": MH_STEPS_PER_STAGE,
    }


def fit(hypothesis, designs, observations, forward, seed, settings=None, scale=20.0,
        fit_timeout_seconds=None, forward_backend="native", kernel_policy="imh_adaptive",
        feature_scales=None):
    """Full normalized product likelihood; fixed unit conversion, original priors."""
    _check_forward_backend(forward_backend, forward)
    if fit_timeout_seconds is not None:
        return run_with_fit_timeout(
            fit, hypothesis, designs, observations, forward, seed, settings, scale,
            forward_backend=forward_backend, kernel_policy=kernel_policy,
            feature_scales=feature_scales, fit_timeout_seconds=fit_timeout_seconds,
        )
    from ..inference.smc import _extract_diagnostics, _extract_logz

    settings = SMCSettings(16, 2, 2) if settings is None else settings
    y = np.asarray(observations, dtype=float)
    if y.ndim != 2 or len(y) != len(designs) or not np.all(np.isfinite(y)) or scale <= 0:
        raise ValueError("finite [observation, feature] matrix and positive unit required")
    kernel_options = _kernel_options(kernel_policy, forward_backend)
    if feature_scales is not None:
        feature_scales = _check_feature_scales(feature_scales, y.shape[1])

    means = ForwardOp(hypothesis, designs, forward, y.shape)

    started = time.perf_counter()
    with pm.Model():
        conductance = pm.LogNormal("conductance", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        # Evaluated on physical mV, so Normal supplies the unit/Jacobian term. With
        # `feature_scales` the features are no longer one unit (four are spike counts), so the
        # ONE latent sigma is stretched per feature by the benchmark's own standardisation
        # instead of by a single mV scale; the prior on sigma is the same HalfNormal(1).
        pm.Normal(
            "observed", mu=means(conductance), observed=y,
            sigma=scale * sigma if feature_scales is None else feature_scales[None, :] * sigma,
        )
        data = pm.sample_smc(
            draws=settings.particles,
            chains=settings.chains,
            cores=settings.cores,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
            **kernel_options,
        )
    logz, chains = _extract_logz(data)
    particles = {k: np.asarray(data.posterior[k]).reshape(-1) for k in ("conductance", "sigma")}
    if not all(np.all(np.isfinite(v)) and np.all(v > 0) for v in particles.values()):
        raise ValueError("invalid posterior")
    diagnostics = _extract_diagnostics(data)
    diagnostics.update(
        seed=seed,
        public_observations=len(y),
        feature_count=y.shape[1],
        likelihood="diagonal_Gaussian_working_feature_likelihood_not_voltage_density",
        sigma_unit_mV=scale,
        original_prior_refit=True,
    )
    if feature_scales is not None:
        # The single mV unit above did not decide this fit's noise, so it is replaced rather
        # than left beside the scales that did.
        del diagnostics["sigma_unit_mV"]
        diagnostics["feature_scales_mV_or_count"] = [float(value) for value in feature_scales]
    if kernel_policy != "imh_adaptive":
        # Recorded only for the kernel that has them: `imh_adaptive` keeps its exact key set.
        diagnostics.update(
            kernel_policy=kernel_policy,
            steps_per_stage=kernel_options["steps_per_stage"],
            ess_threshold=kernel_options["threshold"],
        )
        if not diagnostics.get("accept_rate"):
            diagnostics["accept_rate"] = _mh_accept_rates(data)
    if forward_backend == "batch":
        diagnostics.update(
            forward_backend=forward_backend,
            forward_cache_misses=_forward_cache_misses(data),
        )
    return VectorFit(
        hypothesis, particles, float(logz), chains, diagnostics, time.perf_counter() - started
    )


def prediction_bank(fits, designs, forward, scale=20.0, forward_backend="native",
                    feature_scales=None):
    """Per-particle predictive means, noise scales and log weights of fitted models at designs.

    Returns a dict of arrays ``means`` (K, D, F), ``sigmas``, ``logweights`` and ``model_indices``.
    """
    _check_forward_backend(forward_backend, forward)
    if forward_backend == "batch":
        # Bounded: one bank's worth of rows at a time, never a run's worth.
        clear_prefill()
    means, sigmas, logweights, model_indices = [], [], [], []
    for m, (result, weight) in enumerate(zip(fits, model_logweights(fits), strict=True)):
        if forward_backend == "batch":
            for design in designs:
                BATCH_FORWARDS[forward](result.hypothesis, result.particles["conductance"],
                                        design)
        bank = np.asarray(
            [
                forward(result.hypothesis, [c], d)
                for c in result.particles["conductance"]
                for d in designs
            ]
        )
        n = len(result.particles["sigma"])
        bank = bank.reshape(n, len(designs), -1)
        means.extend(bank)
        sigmas.extend(
            scale * result.particles["sigma"] if feature_scales is None
            else np.asarray(feature_scales, dtype=float)[None, :]
            * result.particles["sigma"][:, None]
        )
        logweights.extend([weight - np.log(n)] * n)
        model_indices.extend([m] * n)
    return {
        "means": np.asarray(means),
        "sigmas": np.asarray(sigmas),
        "logweights": np.asarray(logweights),
        "model_indices": np.asarray(model_indices),
    }


# --------------------------------------------------------------------------
# Per-feature noise: a bank's per-particle noise is either ONE scale per particle - the
# (K,) arrays every dev/paper bank has always carried - or one per feature,
# (K, F), which is what the six evaluation features need because four of them
# are spike counts and two are millivolts. The two helpers below are the only
# places that distinguish the two shapes, and a (K,) or (K, 1) bank reaches
# exactly the arithmetic it reached before: the same array, the same
# broadcast, the same bits.
# --------------------------------------------------------------------------


def particle_sigmas(sigmas):
    """Per-particle noise broadcast against a (K, D, F) mean bank, i.e. (K, 1, F) or (K, 1, 1)."""
    sigmas = np.asarray(sigmas)
    if sigmas.ndim == 1:
        return sigmas[:, None, None]
    if sigmas.ndim == 2:
        return sigmas[:, None, :]
    raise ValueError("per-particle sigmas must be (K,) or (K, features)")


def _feature_sigmas(sigmas, feature):
    """The (K,) noise column one feature is measured with."""
    if sigmas.ndim == 1:
        return sigmas
    return sigmas[:, feature if sigmas.shape[1] > 1 else 0]


def log_predictive(bank, y):
    """Mixture log predictive density of observed feature vectors ``y`` under a prediction bank."""
    y = np.asarray(y)
    logp = norm.logpdf(y[None, ...], bank["means"], particle_sigmas(bank["sigmas"])).sum(-1)
    return logsumexp(logp + bank["logweights"][:, None], axis=0)


def predictive_summary(bank):
    """Marginal future-feature intervals, not intervals for the latent mean."""
    weights = np.exp(bank["logweights"])
    means, sigmas = bank["means"], np.asarray(bank["sigmas"])
    intervals = []
    for probability in (0.05, 0.95):
        quantiles = np.empty(means.shape[1:])
        for index in np.ndindex(quantiles.shape):
            mu = means[(slice(None), *index)]
            column = _feature_sigmas(sigmas, index[-1])
            quantiles[index] = brentq(
                lambda value, mu=mu, probability=probability, sigmas=column: (
                    weights @ norm.cdf(value, mu, sigmas) - probability
                ),
                np.min(mu - 10 * column),
                np.max(mu + 10 * column),
            )
        intervals.append(quantiles)
    return {
        "feature_predictive_mean": np.einsum("p,pdf->df", weights, means),
        "feature_marginal_lower90": intervals[0],
        "feature_marginal_upper90": intervals[1],
    }


def _drawn_sigmas(sigmas, components):
    """The (S, F) or (S, 1) noise of the drawn mixture components."""
    drawn = np.asarray(sigmas)[components]
    return drawn[:, None] if drawn.ndim == 1 else drawn


def _bank_sigmas(sigmas):
    """(1, K, F) or (1, K, 1): every particle's noise against an (S, K, F) density block."""
    sigmas = np.asarray(sigmas)
    return sigmas[None, :, None] if sigmas.ndim == 1 else sigmas[None, :, :]


def _component_logp(ys, means_at_design, sigmas):
    """(S, K): every drawn outcome's log density under every particle's predictive."""
    return norm.logpdf(ys[:, None, :], means_at_design[None, :, :], _bank_sigmas(sigmas)).sum(-1)


EIG_PART_OBJECTIVES = ("structure_eig", "parameter_eig")


def joint_eig(bank, seed, samples=64, *, objective="eig", decompose=False):
    """MC I((model,parameters); vector Y), not model-only entropy.

    With ``objective`` "structure_eig" or "parameter_eig" the same
    draws also give the two chain-rule parts, I((M,theta);Y) = I(M;Y) + E_M I(theta;Y|M):
    structure (Model EIG, MDA v4 Eq. 26) = E[log p_m(y) - log p(y)] and parameter =
    E[log p(y|theta,m) - log p_m(y)], where p_m is model m's weighted particle mixture
    (``bank["model_indices"]``); every row carries both and the design is the argmax of the
    named column, ties to the lowest design index as before. The rng calls are unchanged, and
    the default (``objective="eig"``) returns exactly the rows it returned before.
    ``decompose=True`` adds the two part columns to the rows of any objective
    (the argmax column is still ``objective``); the default call is unchanged.
    """
    if objective != "eig" and objective not in EIG_PART_OBJECTIVES:
        raise ValueError(f"unknown EIG objective {objective!r}")
    parts = objective != "eig" or bool(decompose)
    rng = np.random.default_rng(seed)
    n_design = bank["means"].shape[1]
    weights = np.exp(bank["logweights"])
    components = rng.choice(len(weights), size=samples, p=weights)
    eps = rng.normal(size=(samples, bank["means"].shape[-1]))
    if parts:
        models = np.asarray(bank["model_indices"])
        members = [np.flatnonzero(models == m) for m in np.unique(models)]
        drawn_model = np.searchsorted(np.unique(models), models[components])
    rows = []
    for d in range(n_design):
        ys = bank["means"][components, d] + _drawn_sigmas(bank["sigmas"], components) * eps
        all_logp = _component_logp(ys, bank["means"][:, d, :], bank["sigmas"])
        conditional = all_logp[np.arange(samples), components]
        marginal = logsumexp(all_logp + bank["logweights"][None, :], axis=1)
        gain = conditional - marginal
        if not np.all(np.isfinite(gain)):
            raise ValueError("non-finite joint EIG")
        rows.append(
            {
                "design_index": d,
                "eig": float(gain.mean()),
                "mc_standard_error": float(gain.std(ddof=1) / np.sqrt(samples)),
                "samples": samples,
            }
        )
        if parts:
            # log p_m(y_s) for every model m, then each draw's own model
            per_model = np.stack([
                logsumexp(all_logp[:, idx] + bank["logweights"][None, idx], axis=1)
                - logsumexp(bank["logweights"][idx])
                for idx in members
            ], axis=1)
            model_log = per_model[np.arange(samples), drawn_model]
            structure, parameter = model_log - marginal, conditional - model_log
            if not (np.all(np.isfinite(structure)) and np.all(np.isfinite(parameter))):
                raise ValueError("non-finite EIG decomposition")
            rows[-1].update(
                structure_eig=float(structure.mean()),
                parameter_eig=float(parameter.mean()),
                structure_mc_standard_error=float(structure.std(ddof=1) / np.sqrt(samples)),
                parameter_mc_standard_error=float(parameter.std(ddof=1) / np.sqrt(samples)),
            )
    return min(rows, key=lambda r: (-r[objective], r["design_index"]))["design_index"], rows


# Point disagreement for vector outputs: the scalar-output rule of
# design_policies.point_disagreement (sum_m W_m (m_m(x) - mbar(x))^2 on the posterior-mean
# prediction) applied to each feature of the vector prediction and summed, i.e. the trace of
# the evidence-weighted between-model covariance of the posterior-mean prediction; the
# per-feature values are kept beside the sum.
POINT_DISAGREEMENT_DEFINITION = (
    "sum_f sum_m W_m * (mu_m(d)_f - sum_k W_k * mu_k(d)_f)**2, where mu_m(d) is model m's "
    "particle-averaged predictive mean at design d and W_m its normalized evidence weight (the sum "
    "of its particles' bank weights); per_feature holds the inner sum without the sum over features f. "
    "The posterior-mean rule of design_policies.point_disagreement per feature, summed: equal predictive "
    "means contribute zero regardless of predictive dispersion; mixed feature units add as written"
)


def point_disagreement(bank):
    """Evidence-weighted between-model variance of the posterior-mean prediction, per design.

    ``prediction_bank`` gives every particle of model m the weight W_m / n_m, so model m's
    posterior-mean prediction is the plain mean of its particles' means and W_m the sum of its
    particles' weights. Returns ``point_disagreement`` (D,), ``per_feature`` (D, F) and
    ``model_weights`` (M,) in the order of ``numpy.unique(model_indices)``.
    """
    models = np.asarray(bank["model_indices"])
    weights = np.exp(np.asarray(bank["logweights"], dtype=float))
    means = np.asarray(bank["means"], dtype=float)
    labels = np.unique(models)
    model_weights = np.array([weights[models == m].sum() for m in labels])
    model_means = np.stack([means[models == m].mean(0) for m in labels])
    if not np.isclose(model_weights.sum(), 1.0, atol=1e-9, rtol=0) or not np.all(np.isfinite(model_means)):
        raise ValueError("model weights must be normalized and predictive means finite")
    centre = np.einsum("m,mdf->df", model_weights, model_means)
    per_feature = np.einsum("m,mdf->df", model_weights, (model_means - centre) ** 2)
    return {"point_disagreement": per_feature.sum(-1), "per_feature": per_feature, "model_weights": model_weights}


def joint_epig(candidate_bank, target_bank, seed, samples=128):
    """MC I(Y_candidate; Y_target) averaged over the targets - predictive, not parameter, gain.

    `joint_eig` above scores a design by how much its outcome would tell us about (model,
    parameters). This scores it by how much its outcome would tell us about the outcomes we are
    actually judged on: the held-out evaluation protocols. Both banks must come from the SAME
    particle cloud in the same order - `prediction_bank(fits, candidates)` and
    `prediction_bank(fits, targets)` - so that particle j is one hypothesis in both.

    The estimator draws `samples` mixture components k ~ w once and reuses them, with one noise
    block for the candidate side and one for the target side, for EVERY candidate and EVERY
    target (common random numbers: two candidates are compared on the same draws, so their
    difference is not a sampling artefact). For draw s with component k,

        y_c = mu[k, c] + sigma[k] * eps_c[s],   y_t = mu[k, t] + sigma[k] * eps_t[s]

    and the mutual information of the two predictives is estimated by

        I(Y_c; Y_t) = E[ log sum_j w_j N(y_c; mu[j, c], sigma_j) N(y_t; mu[j, t], sigma_j)
                        - log sum_j w_j N(y_c; mu[j, c], sigma_j)
                        - log sum_j w_j N(y_t; mu[j, t], sigma_j) ],

    every sum by `logsumexp`. EPIG(c) is that averaged over the targets; the standard error is
    the spread of the target-averaged per-draw values, so it measures the Monte Carlo error of
    the reported number rather than of one target's.

    Returns `(design_index, rows)` with the same rule `joint_eig` uses: the largest EPIG, ties
    broken by the lowest design index.
    """
    logweights = np.asarray(candidate_bank["logweights"])
    if (np.asarray(target_bank["logweights"]).shape != logweights.shape
            or not np.array_equal(target_bank["logweights"], logweights)
            or np.asarray(target_bank["means"]).shape[0] != np.asarray(
                candidate_bank["means"]).shape[0]):
        raise ValueError("candidate and target banks must share one particle cloud")
    candidate_means = np.asarray(candidate_bank["means"])
    target_means = np.asarray(target_bank["means"])
    sigmas = candidate_bank["sigmas"]
    n_candidate, n_target = candidate_means.shape[1], target_means.shape[1]
    if not n_candidate or not n_target:
        raise ValueError("EPIG needs at least one candidate and one target design")
    rng = np.random.default_rng(seed)
    weights = np.exp(logweights)
    components = rng.choice(len(weights), size=samples, p=weights)
    drawn = _drawn_sigmas(sigmas, components)
    eps_candidate = rng.normal(size=(samples, candidate_means.shape[-1]))
    eps_target = rng.normal(size=(samples, target_means.shape[-1]))
    weighted = logweights[None, :]

    target_logp, target_marginal = [], []
    for t in range(n_target):
        ys = target_means[components, t] + drawn * eps_target
        logp = _component_logp(ys, target_means[:, t, :], sigmas)
        target_logp.append(logp)
        target_marginal.append(logsumexp(logp + weighted, axis=1))

    rows = []
    for c in range(n_candidate):
        ys = candidate_means[components, c] + drawn * eps_candidate
        logp = _component_logp(ys, candidate_means[:, c, :], sigmas)
        marginal = logsumexp(logp + weighted, axis=1)
        per_target, total = [], np.zeros(samples)
        for t in range(n_target):
            joint = logsumexp(logp + target_logp[t] + weighted, axis=1)
            gain = joint - marginal - target_marginal[t]
            per_target.append(gain)
            total = total + gain
        total = total / n_target
        if not np.all(np.isfinite(total)) or not all(
                np.all(np.isfinite(gain)) for gain in per_target):
            raise ValueError("non-finite joint EPIG")
        rows.append(
            {
                "policy": "epig",
                "design_index": c,
                "epig": float(total.mean()),
                "mc_standard_error": float(total.std(ddof=1) / np.sqrt(samples)),
                "samples": samples,
                "targets": n_target,
                "per_target": [float(gain.mean()) for gain in per_target],
            }
        )
    return min(rows, key=lambda r: (-r["epig"], r["design_index"]))["design_index"], rows
