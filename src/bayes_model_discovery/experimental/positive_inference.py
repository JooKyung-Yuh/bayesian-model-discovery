"""Multivariate scalar sequential Monte Carlo (SMC) for Poisson, mean-LogNormal and Gamma observations.

This first-path API uses a fixed log-linear catalogue, not arbitrary LLM Python.
All densities and predictions are in the original response units. No adapter,
hidden label source, model selection score or network transport is imported.

A SECOND hypothesis shape sits on the same likelihoods: a
`ScalarHypothesis` whose log-mean is a validated grammar expression over the
task interface's named inputs, so the unbounded-count and positive rows of
the paper's task table are reachable by the same proposal loop as CORE's bounded
count. `RegressionHypothesis` (the fixed feature catalogue) is untouched and
every function below still takes the exact same path for it.

The module also adds the last unbounded-count revision the paper's task
table lists, the zero-inflated Poisson, and routes the identity-link REAL
families (`real_inference`) through the same pool, predictive and expected
information gain (EIG) code so there is one implementation of `log_predictive`,
`predictive_summary` and `joint_eig` for every scalar outcome. The Poisson /
Negative-Binomial / LogNormal / Gamma arithmetic below is untouched: every new
branch is guarded by a family name none of them can take.
"""

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import gamma, lognorm, nbinom, poisson

from .count_inference import Fit, SMCSettings, model_logweights
from .count_models import (
    DEFAULT_INPUT_NAMES,
    PRIOR_RULES,
    canonical_model_id,
    evaluate,
    model_id_preimage,
    named_input_values,
    validate_expression,
)

FAMILIES = ("poisson", "lognormal", "gamma")
# Table 2 revision families reachable from the grammar path, on top of FAMILIES.
# `negative_binomial` is the unbounded-count revision (alpha is the dispersion);
# its prior is the fixed "shape" role prior Gamma(2,1) from count_models.
# PRIOR_RULES, the same one the Gamma family's shape uses.
# The paper's task table, row 2, quoted verbatim:
#   | Unbounded count | log | Poisson(mu) | Negative Binomial(alpha);
#     zero-inflated Poisson only where zeros are natural | |
# `zi_poisson` is that second revision. Its zero-inflation
# parameter is the SAME Table 3 row the bounded-count ZI-Binomial already uses -
# "| zero-inflation $\pi$ | Beta$(1,4)$ |", i.e. count_models.PRIOR_RULES["pi"] -
# and it is parameterised the same way round as count_inference's
# ZeroInflatedBinomial: `pi` is the excess-zero probability and PyMC's `psi` is
# 1 - pi. The predictive mean is (1 - pi) * mu, scored as such, symmetric with
# Table 2's own note for the ZI-Binomial ("predictive mean (1-pi)Np, scored as
# such").
SCALAR_FAMILIES = ("poisson", "negative_binomial", "zi_poisson", "lognormal", "gamma")
SCALAR_NOISE = {
    "poisson": (),
    "negative_binomial": ("alpha",),
    "zi_poisson": ("pi",),
    "lognormal": ("sigma",),
    "gamma": ("shape",),
}
COUNT_SCALAR_FAMILIES = ("poisson", "negative_binomial", "zi_poisson")
# The families whose conditional mean must be finite AND strictly
# positive. Every family above is log-link - `response_unit * exp(expression)` for
# `ScalarHypothesis`, `response_unit * exp(eta)` for the catalogue
# `RegressionHypothesis`, whose FAMILIES are a subset - so a mean that overflows,
# underflows to zero or is NaN is refused: "non-finite/nonpositive mean; no clipping
# fallback" in `particle_means` and `grammar_means`, and the validity mask of
# `particle_means_masked`, which count_inference's EPIG path raises on. The rule is
# family-scoped: the identity-link real families (`real_inference.REAL_FAMILIES`)
# need only a finite mean and are not here, and the bounded-count and binary
# families never reach this module. `particle_means_masked` reads this tuple, and
# `belief_packet.cell_constraints` publishes the cell's own share of it as
# `constraints["positive_mean_required_families"]`.
POSITIVE_MEAN_REQUIRED_FAMILIES = SCALAR_FAMILIES


@dataclass(frozen=True)
class ScalarHypothesis:
    """A grammar expression for the log-mean of a scalar outcome.

    `mean = response_unit * exp(expression)`, the log link the paper's task table
    fixes for unbounded counts and positive outcomes. The expression is
    validated by `count_models.validate_expression` - the SAME grammar CORE's
    bounded-count hypotheses use - over the task interface's named inputs, and
    the identity is the same canonical AST + roles + family hash, so a model
    written over different input names is a different model.
    """

    expression: str
    family: str
    parameters: tuple[tuple[str, str], ...]
    input_names: tuple[str, ...] = DEFAULT_INPUT_NAMES

    def __post_init__(self):
        if self.family not in SCALAR_FAMILIES:
            raise ValueError(f"unsupported scalar family: {self.family}")
        validate_expression(self.expression, self.parameters, self.input_names)

    @property
    def model_id(self):
        """Canonical model id: hash of the expression AST, parameter roles and family."""
        return canonical_model_id(self.expression, self.family, self.parameters)

    @property
    def model_id_preimage(self):
        """The canonical text that `model_id` hashes, for audit records."""
        return model_id_preimage(self.expression, self.family, self.parameters)

    @property
    def noise_parameters(self):
        """Names of the family's noise parameters (empty for Poisson)."""
        return SCALAR_NOISE[self.family]

    @property
    def measure(self):
        """'PMF' for the count families, 'PDF' for the positive continuous ones."""
        return "PMF" if self.family in COUNT_SCALAR_FAMILIES else "PDF"

    def as_dict(self):
        """JSON-ready description of the hypothesis: id, expression, family and priors."""
        return {
            "model_id": self.model_id,
            "expression": self.expression,
            "family": self.family,
            "mean": "response_unit * exp(expression)",
            "parameters": dict(self.parameters),
            "noise_parameters": self.noise_parameters,
            "measure": self.measure,
            "priors": {
                "intercept": ["Normal", 0, 2],
                "signed": ["Normal", 0, 1],
                "positive": ["LogNormal", 0, 1],
                "shape": ["Gamma", 2, 1],
                "sigma": ["HalfNormal", 1],
                "alpha": ["Gamma", 2, 1],
                "pi": list(PRIOR_RULES["pi"]),
            },
        }


def _is_real(hypothesis):
    """True for the identity-link real families of `real_inference`.

    Every guard below is written this way so that the Poisson / Negative-
    Binomial / ZI-Poisson / LogNormal / Gamma branches keep their exact earlier
    code path; a family this predicate accepts can never reach them.
    """
    from .real_inference import is_real

    return is_real(hypothesis)


def is_grammar(hypothesis):
    """True for the grammar-expression path, False for the fixed feature catalogue."""
    return hasattr(hypothesis, "expression")


@dataclass(frozen=True)
class RegressionHypothesis:
    """Fixed-catalogue log-linear hypothesis: a family plus up to three input columns.

    The log-mean is an intercept plus one signed coefficient per selected,
    standardized input column.
    """

    family: str
    features: tuple[int, ...] = ()

    def __post_init__(self):
        if self.family not in FAMILIES or len(self.features) > 3:
            raise ValueError("unregistered family or more than three mean features")
        if len(set(self.features)) != len(self.features) or any(
            type(index) is not int or index < 0 for index in self.features
        ):
            raise ValueError("feature indices must be distinct nonnegative integers")

    @property
    def model_id(self):
        """Short sha256 id of the (log_linear, family, features) triple."""
        raw = json.dumps(["log_linear", self.family, self.features]).encode()
        return hashlib.sha256(raw).hexdigest()[:20]

    @property
    def parameters(self):
        """(name, prior role) pairs: intercept b0 and one signed slope per feature."""
        return (("b0", "intercept"),) + tuple(
            (f"b{i + 1}", "signed") for i in range(len(self.features))
        )

    @property
    def noise_parameters(self):
        """Names of the family's noise parameters (empty for Poisson)."""
        return {"poisson": (), "lognormal": ("sigma",), "gamma": ("shape",)}[self.family]

    @property
    def measure(self):
        """'PMF' for Poisson, 'PDF' for LogNormal and Gamma."""
        return "PMF" if self.family == "poisson" else "PDF"

    def as_dict(self):
        """JSON-ready description of the hypothesis: id, features, mean form and priors."""
        return {
            "model_id": self.model_id,
            "family": self.family,
            "mean": "response_unit * exp(b0 + sum(b_j * standardized_x_j))",
            "features": list(self.features),
            "parameters": dict(self.parameters),
            "noise_parameters": self.noise_parameters,
            "measure": self.measure,
            "priors": {
                "b0": ["Normal", 0, 2],
                "b_j": ["Normal", 0, 1],
                "sigma": ["HalfNormal", 1],
                "shape": ["Gamma", 2, 1],
            },
        }


@dataclass(frozen=True)
class InputTransform:
    """Frozen per-column standardization (center, scale) of the public inputs."""

    center: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def from_public_pool(cls, inputs):
        """Build the transform from the public input matrix: column means and std devs."""
        x = finite_matrix(inputs)
        scale = np.std(x, axis=0)
        # A constant public input has zero standardized value, not a fitted scale.
        scale = np.where(scale == 0, 1.0, scale)
        return cls(tuple(x.mean(axis=0)), tuple(scale))

    def apply(self, inputs):
        """Standardize an [observation, input] matrix; raise on an invalid transform."""
        x = finite_matrix(inputs)
        center, scale = np.asarray(self.center), np.asarray(self.scale)
        if (
            center.shape != (x.shape[1],)
            or scale.shape != center.shape
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0)
        ):
            raise ValueError("invalid frozen input transform")
        return (x - center) / scale


def finite_matrix(inputs):
    """Return `inputs` as a float array; raise unless it is a nonempty finite 2-D matrix."""
    x = np.asarray(inputs, dtype=float)
    if x.ndim != 2 or 0 in x.shape or not np.all(np.isfinite(x)):
        raise ValueError("nonempty finite [observation, input] matrix required")
    return x


def initial_catalogue(inputs_d0, labels_d0, family):
    """Choose up to three features from D0 only, before any test score exists."""
    x = finite_matrix(inputs_d0)
    y = np.asarray(labels_d0, dtype=float)
    if y.shape != (len(x),) or not np.all(np.isfinite(y)) or x.shape[1] < 3:
        raise ValueError("D0 labels and at least three input columns required")
    xc, yc = x - x.mean(0), y - y.mean()
    norms = np.linalg.norm(xc, axis=0) * np.linalg.norm(yc)
    correlations = np.divide(np.abs(xc.T @ yc), norms, out=np.zeros_like(norms), where=norms > 0)
    selected = sorted(range(x.shape[1]), key=lambda i: (-correlations[i], i))[:3]
    return [RegressionHypothesis(family, tuple(selected[:n])) for n in range(4)], {
        "rule": "absolute_D0_Pearson_correlation_then_column_index",
        "features": selected,
        "correlations": correlations.tolist(),
        "public_observation_count": len(y),
        "test_labels_used": 0,
    }


def validate_data(hypothesis, x, y, response_unit):
    """Check catalogue-path data against the family's support; return (x, y) arrays."""
    x = finite_matrix(x)
    y = np.asarray(y, dtype=float)
    if y.shape != (len(x),) or not np.all(np.isfinite(y)):
        raise ValueError("one finite scalar observation per input required")
    if not np.isfinite(response_unit) or response_unit <= 0:
        raise ValueError("response_unit must be positive and frozen from D0")
    if hypothesis.features and max(hypothesis.features) >= x.shape[1]:
        raise ValueError("feature outside the public input matrix")
    if hypothesis.family == "poisson":
        if np.any((y < 0) | (y != np.floor(y))):
            raise ValueError("Poisson requires nonnegative integer counts")
    elif np.any(y <= 0):
        raise ValueError("positive continuous likelihood requires y > 0; no clipping")
    return x, y


def build_model(hypothesis, inputs, labels, transform, response_unit, interface=None,
                response_center=0.0):
    """PyMC model for one hypothesis; `interface` selects the grammar-expression path.

    `response_center` is the identity-link half of the frozen D0 calibration and
    is read only by the real families (`real_inference`); the log-link families
    ignore it, exactly as they did before the real families were added.
    """
    import pymc as pm
    import pytensor.tensor as pt

    if _is_real(hypothesis):
        from .real_inference import build_model as build_real_model

        return build_real_model(hypothesis, inputs, labels, interface, response_center,
                                response_unit)
    if is_grammar(hypothesis):
        designs = np.asarray(inputs, dtype=float)
        y = np.asarray(labels, dtype=float)
        if len(designs) != len(y) or y.ndim != 1 or not len(y):
            raise ValueError("nonempty scalar observation vectors required")
        if hypothesis.family in COUNT_SCALAR_FAMILIES:
            if np.any((y < 0) | (y != np.floor(y))):
                raise ValueError("count likelihood requires nonnegative integer counts")
        elif np.any(y <= 0):
            raise ValueError("positive continuous likelihood requires y > 0; no clipping")
        named = named_input_values(interface, designs)
        anchor = pt.as_tensor(np.asarray(named[interface["inputs"][0]["name"]]))
        with pm.Model() as model:
            parameters = {}
            for name, role in hypothesis.parameters:
                family, a, b = PRIOR_RULES[role]
                parameters[name] = getattr(pm, family)(name, a, b, transform=None)
            eta = evaluate(hypothesis.expression, parameters, anchor, pt, named)
            mean = response_unit * pm.math.exp(eta)
            if hypothesis.family == "poisson":
                pm.Poisson("observed", mu=mean, observed=y)
            elif hypothesis.family == "negative_binomial":
                alpha = pm.Gamma("alpha", alpha=2, beta=1, transform=None)
                pm.NegativeBinomial("observed", mu=mean, alpha=alpha, observed=y)
            elif hypothesis.family == "zi_poisson":
                pi = pm.Beta("pi", 1, 4, transform=None)
                pm.ZeroInflatedPoisson("observed", psi=1 - pi, mu=mean, observed=y)
            elif hypothesis.family == "lognormal":
                sigma = pm.HalfNormal("sigma", 1, transform=None)
                pm.LogNormal(
                    "observed", mu=pm.math.log(mean) - sigma**2 / 2, sigma=sigma, observed=y
                )
            else:
                shape = pm.Gamma("shape", alpha=2, beta=1, transform=None)
                pm.Gamma("observed", alpha=shape, beta=shape / mean, observed=y)
        return model
    x, y = validate_data(hypothesis, transform.apply(inputs), labels, response_unit)
    with pm.Model() as model:
        eta = pm.Normal("b0", 0, 2) + np.zeros(len(x))
        for index, feature in enumerate(hypothesis.features, 1):
            eta = eta + pm.Normal(f"b{index}", 0, 1) * x[:, feature]
        mean = response_unit * pm.math.exp(eta)
        if hypothesis.family == "poisson":
            pm.Poisson("observed", mu=mean, observed=y)
        elif hypothesis.family == "lognormal":
            sigma = pm.HalfNormal("sigma", 1)
            pm.LogNormal("observed", mu=pm.math.log(mean) - sigma**2 / 2, sigma=sigma, observed=y)
        else:
            shape = pm.Gamma("shape", alpha=2, beta=1)
            pm.Gamma("observed", alpha=shape, beta=shape / mean, observed=y)
    return model


def fit(hypothesis, inputs, labels, seed, transform, response_unit, settings=None, interface=None,
        response_center=0.0):
    """Fit one hypothesis by SMC and return a `Fit` (particles, log evidence, diagnostics).

    The diagnostics block records the seed, data hash, frozen transform and response unit.
    """
    import pymc as pm

    from ..inference.smc import _extract_diagnostics, _extract_logz

    settings = SMCSettings() if settings is None else settings
    started = time.perf_counter()
    model = build_model(
        hypothesis, inputs, labels, transform, response_unit, interface, response_center
    )
    with model:
        idata = pm.sample_smc(
            draws=settings.particles,
            chains=settings.chains,
            cores=settings.cores,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
        )
    logz, per_chain = _extract_logz(idata)
    names = tuple(name for name, _ in hypothesis.parameters) + hypothesis.noise_parameters
    particles = {name: np.asarray(idata.posterior[name]).reshape(-1) for name in names}
    if not all(np.all(np.isfinite(values)) for values in particles.values()):
        raise ValueError("non-finite SMC particle")
    diag = _extract_diagnostics(idata)
    diag.update(
        seed=int(seed),
        particles_per_chain=settings.particles,
        chains=settings.chains,
        cores=settings.cores,
        observation_count=len(labels),
        public_data_sha256=hashlib.sha256(
            np.asarray(inputs, dtype="<f8").tobytes() + np.asarray(labels, dtype="<f8").tobytes()
        ).hexdigest(),
        input_transform=(
            None if transform is None else {"center": transform.center, "scale": transform.scale}
        ),
        interface=interface,
        response_unit=float(response_unit),
        response_unit_policy="frozen_D0_calibration",
        likelihood_measure=hypothesis.measure,
        refit="original_prior_and_all_public_observations",
        particle_weighting="equal_within_and_across_chains",
    )
    if _is_real(hypothesis):
        # Recorded only on the real path, so the log-link diagnostics blocks stay
        # byte-identical to the ones written before the real families were added.
        diag["response_center"] = float(response_center)
    return Fit(hypothesis, particles, float(logz), per_chain, diag, time.perf_counter() - started)


def particle_means(result, inputs):
    """Per-[particle, input] link means of a fit.

    The log-link families raise on a non-finite or nonpositive mean; the identity-link
    real families are delegated to `real_inference.particle_means`.
    """
    binding = result.diagnostics
    if _is_real(result.hypothesis):
        from .real_inference import particle_means as real_particle_means

        return real_particle_means(result, inputs)
    if is_grammar(result.hypothesis):
        return grammar_means(
            result.hypothesis,
            result.particles,
            inputs,
            binding["interface"],
            binding["response_unit"],
        )
    x = InputTransform(**binding["input_transform"]).apply(inputs)
    p = result.particles
    eta = np.asarray(p["b0"])[:, None] + np.zeros((1, len(x)))
    for index, feature in enumerate(result.hypothesis.features, 1):
        eta = eta + np.asarray(p[f"b{index}"])[:, None] * x[None, :, feature]
    with np.errstate(over="ignore", under="ignore"):
        mean = binding["response_unit"] * np.exp(eta)
    if not np.all(np.isfinite(mean)) or np.any(mean <= 0):
        raise ValueError("non-finite/nonpositive mean; no clipping fallback")
    return mean


def particle_means_masked(result, inputs):
    """Return unchanged means and their elementwise domain-validity mask.

    Positive/count log-link families (`POSITIVE_MEAN_REQUIRED_FAMILIES`) require
    finite, strictly positive means; identity-link real families require only
    finite means, including zero and negative values. Arithmetic follows the
    strict mean paths in the same order, without clipping or replacement. Only
    numerical warnings are suppressed: malformed bindings, input shapes and
    expressions still raise their errors. The caller decides whether to reduce
    the mask across inputs or particles.
    """
    binding = result.diagnostics
    real = _is_real(result.hypothesis)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        if real or is_grammar(result.hypothesis):
            interface = binding["interface"]
            if real:
                center = float(binding.get("response_center", 0.0))
                unit = float(binding["response_unit"])
            designs = np.asarray(inputs, dtype=float)
            named = {
                name: np.asarray(value, dtype=float)[None, :]
                for name, value in named_input_values(interface, designs).items()
            }
            params = {
                name: np.asarray(result.particles[name], dtype=float)[:, None]
                for name, _ in result.hypothesis.parameters
            }
            anchor = named[interface["inputs"][0]["name"]]
            if real:
                mean = float(center) + float(unit) * evaluate(
                    result.hypothesis.expression, params, anchor, np, named
                )
            else:
                mean = float(binding["response_unit"]) * np.exp(
                    evaluate(result.hypothesis.expression, params, anchor, np, named)
                )
        else:
            x = InputTransform(**binding["input_transform"]).apply(inputs)
            p = result.particles
            eta = np.asarray(p["b0"])[:, None] + np.zeros((1, len(x)))
            for index, feature in enumerate(result.hypothesis.features, 1):
                eta = eta + np.asarray(p[f"b{index}"])[:, None] * x[None, :, feature]
            mean = binding["response_unit"] * np.exp(eta)
        valid_mask = np.isfinite(mean)
        if result.hypothesis.family in POSITIVE_MEAN_REQUIRED_FAMILIES:
            valid_mask = valid_mask & (mean > 0)
    return mean, valid_mask


def predictive_means(result, inputs):
    """Per-particle CONDITIONAL PREDICTIVE mean, which is not always the link mean.

    For every family but one this returns exactly
    `particle_means(result, inputs)`. The zero-inflated Poisson is the one place
    the two differ: its link mean is the Poisson rate mu and its predictive mean
    is (1 - pi) mu, which is what Table 2 says to score.
    """
    mean = particle_means(result, inputs)
    if result.hypothesis.family == "zi_poisson":
        return (1.0 - np.asarray(result.particles["pi"], dtype=float)[:, None]) * mean
    return mean


def conditional_loglik(result, inputs, outcomes):
    """[particle, input, draw], all PMF/PDF normalizing constants included."""
    if _is_real(result.hypothesis):
        from .real_inference import conditional_loglik as real_conditional_loglik

        return real_conditional_loglik(result, inputs, outcomes)
    mean = particle_means(result, inputs)[:, :, None]
    y = np.asarray(outcomes, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] != mean.shape[1] or not np.all(np.isfinite(y)):
        raise ValueError("outcomes must be finite [input, draw]")
    if result.hypothesis.family == "poisson":
        values = poisson.logpmf(y[None, :, :], mean)
    elif result.hypothesis.family == "negative_binomial":
        alpha = result.particles["alpha"][:, None, None]
        values = nbinom.logpmf(y[None, :, :], alpha, alpha / (alpha + mean))
    elif result.hypothesis.family == "zi_poisson":
        pi = np.asarray(result.particles["pi"], dtype=float)[:, None, None]
        if np.any((pi <= 0) | (pi >= 1)):
            raise ValueError("pi outside open unit interval")
        values = poisson.logpmf(y[None, :, :], mean) + np.log1p(-pi)
        values = np.where(y[None, :, :] == 0, np.logaddexp(values, np.log(pi)), values)
    elif result.hypothesis.family == "lognormal":
        sigma = result.particles["sigma"][:, None, None]
        values = lognorm.logpdf(y[None, :, :], sigma, scale=mean * np.exp(-(sigma**2) / 2))
    else:
        shape = result.particles["shape"][:, None, None]
        values = gamma.logpdf(y[None, :, :], shape, scale=mean / shape)
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("invalid conditional likelihood")
    return values


def validate_pool(fits):
    """Check that a pool of fits shares measure and data binding; return log model weights."""
    weights = model_logweights(fits)
    first = fits[0].diagnostics
    for result in fits:
        if result.hypothesis.measure != fits[0].hypothesis.measure:
            raise ValueError("cannot mix PMF and PDF as one predictive distribution")
        for key in (
            "input_transform",
            "interface",
            "response_unit",
            "response_center",
            "observation_count",
            "public_data_sha256",
        ):
            if result.diagnostics.get(key) != first.get(key):
                raise ValueError("pool must share transform, response unit and public data size")
    return weights


# Added by the engine's memory-reduction change: the memory bound of one model's
# [particle, input, draw] conditional log-likelihood inside `log_predictive`, in
# cells, with particles = the pool's largest bank (the accounting of
# `boxinggym_truth_law.KL_PREDICTIVE_BLOCK_CELLS`). Above it the draw axis is
# evaluated in whole-column blocks. No value depends on it, so it is an
# implementation detail that no config or record carries.
_LOG_PREDICTIVE_BLOCK_CELLS = 8_000_000


def _draw_blocks(fits, outcomes):
    """`[start, stop)` draw-column blocks for `log_predictive`, or None for one call.

    Each column of `log_predictive`'s two outputs is a function of that column
    alone: the density of every (particle, input, draw) cell, and the particle-axis
    `logsumexp` of every (input, draw) pair. numpy reduces axis 0 of a C-contiguous
    [particle, input, width] array one element at a time, in particle order, whenever
    input * width >= 2. At input * width == 1 the block is a single particle vector
    and numpy sums it pairwise, which rounds differently (a bit-exactness rule of the
    memory-reduction change). So with one input no block is one column wide: the
    width is at least 2, and a one-column tail joins the block before it. Outcomes
    that are not a 2-D ndarray with at least two columns, and a call that fits in
    one block, take the one-call path unchanged.
    """
    if not isinstance(outcomes, np.ndarray) or outcomes.ndim != 2 or outcomes.shape[1] < 2:
        return None
    rows, draws = outcomes.shape
    particles = max(particle_count(result) for result in fits)
    width = max(1, _LOG_PREDICTIVE_BLOCK_CELLS // max(1, particles * rows))
    if rows == 1:
        width = max(width, 2)
    starts = list(range(0, draws, width))
    if rows == 1 and draws - starts[-1] == 1:
        starts.pop()
    if len(starts) == 1:
        return None
    return list(zip(starts, [*starts[1:], draws], strict=True))


def log_predictive(fits, inputs, outcomes):
    """Pool log predictive density of `outcomes` [input, draw] at `inputs`.

    Returns (mixture log density [input, draw], per-model components [model, input, draw]).
    """
    weights = validate_pool(fits)
    blocks = _draw_blocks(fits, outcomes)
    if blocks is None:
        components = np.stack(
            [
                logsumexp(conditional_loglik(result, inputs, outcomes), axis=0)
                - np.log(particle_count(result))
                for result in fits
            ]
        )
    else:
        # Memory-bounded path: the same two lines per draw block; particle and input axes whole.
        components = np.stack(
            [
                np.concatenate(
                    [
                        logsumexp(
                            conditional_loglik(result, inputs, outcomes[:, start:stop]), axis=0
                        )
                        for start, stop in blocks
                    ],
                    axis=1,
                )
                - np.log(particle_count(result))
                for result in fits
            ]
        )
    return logsumexp(components + weights[:, None, None], axis=0), components


def draw_conditional(result, inputs, indices, rng):
    """Draw one outcome per selected particle at every input; returns [input, particle]."""
    if _is_real(result.hypothesis):
        from .real_inference import draw_conditional as real_draw_conditional

        return real_draw_conditional(result, inputs, indices, rng)
    means = particle_means(result, inputs)[indices].T
    if result.hypothesis.family == "poisson":
        return rng.poisson(means)
    if result.hypothesis.family == "zi_poisson":
        pi = result.particles["pi"][indices][None, :]
        counts = rng.poisson(means)
        return np.where(rng.random(counts.shape) < pi, 0, counts)
    if result.hypothesis.family == "negative_binomial":
        alpha = result.particles["alpha"][indices][None, :]
        return rng.negative_binomial(alpha, alpha / (alpha + means))
    if result.hypothesis.family == "lognormal":
        sigma = result.particles["sigma"][indices][None, :]
        return rng.lognormal(np.log(means) - sigma**2 / 2, sigma)
    shape = result.particles["shape"][indices][None, :]
    return rng.gamma(shape, means / shape)


def predictive_summary(fits, inputs, seed, sample_count=2048):
    """Pool predictive summary: BMA and best-model means, 90% interval and samples."""
    weights = validate_pool(fits)
    if sample_count < 2:
        raise ValueError("at least two predictive draws required")
    rng = np.random.default_rng(seed)
    model = rng.choice(len(fits), sample_count, p=np.exp(weights))
    draws = np.empty((len(inputs), sample_count))
    for index, result in enumerate(fits):
        mask = np.flatnonzero(model == index)
        particle = rng.integers(particle_count(result), size=len(mask))
        draws[:, mask] = draw_conditional(result, inputs, particle, rng)
    means = np.stack([predictive_means(result, inputs).mean(0) for result in fits])
    best = min(range(len(fits)), key=lambda i: (-weights[i], fits[i].hypothesis.model_id))
    return {
        "mean": np.exp(weights) @ means,
        "single_mean": means[best],
        "lower90": np.quantile(draws, 0.05, axis=1),
        "upper90": np.quantile(draws, 0.95, axis=1),
        "samples": draws,
        "model_weights": np.exp(weights),
        "single_model_id": fits[best].hypothesis.model_id,
        "measure": fits[0].hypothesis.measure,
    }


POINT_DISAGREEMENT_DEFINITION = (
    "sum_m W_m * (m_m(x) - sum_k W_k * m_k(x))**2, where m_m(x) is the "
    "particle-averaged conditional predictive mean and W_m is the normalized "
    "model evidence weight; equal predictive means contribute zero regardless "
    "of predictive dispersion"
)


def point_disagreement(fits, inputs):
    """Count-policy weighted mean variance, applied to scalar predictive banks.

    All retained posterior particles contribute through the existing conditional
    predictive-mean implementation, including zero-inflation and real families.
    No output samples, posterior subsampling, RNG or EIG calculation is used.
    """
    from .design_policies import point_disagreement as mean_disagreement

    weights = validate_pool(fits)
    means = np.stack([predictive_means(result, inputs).mean(0) for result in fits])
    return mean_disagreement(means, np.exp(weights))


def joint_eig(fits, inputs, seed, outer_samples=64, *, decompose=False):
    """Posterior-bank MI MC: E_k E_y|k [log p_k(y|x)-log sum_j w_j p_j(y|x)].

    Added 2026-10-05 for the EIG decomposition: with ``decompose=True`` the same
    draws also give the two chain-rule parts of the joint EIG,
    I((M,theta);Y) = I(M;Y) + E_M I(theta;Y|M):
    structure (Model EIG, MDA v4 Eq. 26) = E[log p_k(y) - log p(y)] and
    parameter = E[log p(y|theta,k) - log p_k(y)], where p_k is model k's
    particle-averaged predictive density (``log_predictive``'s components). The rng
    calls are unchanged, so the default call is byte-identical to the base.
    """
    weights = validate_pool(fits)
    if outer_samples < 2:
        raise ValueError("at least two EIG draws required")
    rng = np.random.default_rng(seed)
    choices = rng.choice(len(fits), outer_samples, p=np.exp(weights))
    ys, conditional = np.empty((len(inputs), outer_samples)), np.empty((len(inputs), outer_samples))
    for index, result in enumerate(fits):
        columns = np.flatnonzero(choices == index)
        indices = rng.integers(particle_count(result), size=len(columns))
        ys[:, columns] = draw_conditional(result, inputs, indices, rng)
        loglik = conditional_loglik(result, inputs, ys[:, columns])
        conditional[:, columns] = loglik[indices, :, np.arange(len(indices))].T
    mixed, components = log_predictive(fits, inputs, ys)
    contributions = conditional - mixed
    if not np.all(np.isfinite(contributions)):
        raise ValueError("non-finite EIG contribution")
    out = {
        "eig": contributions.mean(1),
        "mc_standard_error": contributions.std(1, ddof=1) / np.sqrt(outer_samples),
        "outer_samples": outer_samples,
        "negative_estimates_clipped": False,
        "target": "joint_model_parameter",
        "seed": seed,
    }
    if decompose:
        # components[k, input, draw] = log p_k(y_draw | x_input); pick each draw's own model
        model_log = components[choices, :, np.arange(outer_samples)].T
        structure = model_log - mixed
        parameter = conditional - model_log
        if not (np.all(np.isfinite(structure)) and np.all(np.isfinite(parameter))):
            raise ValueError("non-finite EIG decomposition contribution")
        out.update(
            structure_eig=structure.mean(1),
            parameter_eig=parameter.mean(1),
            structure_mc_standard_error=structure.std(1, ddof=1) / np.sqrt(outer_samples),
            parameter_mc_standard_error=parameter.std(1, ddof=1) / np.sqrt(outer_samples),
        )
    return out


def particle_count(result):
    """Number of SMC particles behind one fit, for either hypothesis shape."""
    return len(next(iter(result.particles.values())))


def grammar_means(hypothesis, particles, inputs, interface, response_unit):
    """response_unit * exp(expression) per [particle, input] on the named inputs."""
    designs = np.asarray(inputs, dtype=float)
    named = {
        name: np.asarray(value, dtype=float)[None, :]
        for name, value in named_input_values(interface, designs).items()
    }
    params = {
        name: np.asarray(particles[name], dtype=float)[:, None] for name, _ in hypothesis.parameters
    }
    anchor = named[interface["inputs"][0]["name"]]
    with np.errstate(over="ignore", under="ignore"):
        mean = float(response_unit) * np.exp(
            evaluate(hypothesis.expression, params, anchor, np, named)
        )
    if not np.all(np.isfinite(mean)) or np.any(mean <= 0):
        raise ValueError("non-finite/nonpositive mean; no clipping fallback")
    return mean


def mixture_moments(hypothesis, particles, inputs, interface, response_unit=1.0,
                    response_center=0.0):
    """Posterior predictive mean and sd of a scalar family, in closed form.

    Each particle contributes a conditional mean m_i and conditional variance
    v_i (Poisson m; NegBin m + m^2/alpha; ZI-Poisson (1-pi)m and (1-pi)m(1+pi m);
    LogNormal m^2(e^{sigma^2}-1); Gamma m^2/shape). The equally weighted particle
    mixture then has mean E[m] and variance E[v] + Var[m] - exact, so the belief
    packet's residual block needs no sampling. `response_center` is read only by
    the identity-link real families.
    """
    if _is_real(hypothesis):
        from .real_inference import mixture_moments as real_mixture_moments

        return real_mixture_moments(
            hypothesis, particles, inputs, interface, response_center, response_unit
        )
    mean = grammar_means(hypothesis, particles, inputs, interface, response_unit)
    family = hypothesis.family
    conditional = mean
    if family == "poisson":
        variance = mean
    elif family == "negative_binomial":
        variance = mean + mean**2 / np.asarray(particles["alpha"], dtype=float)[:, None]
    elif family == "zi_poisson":
        pi = np.asarray(particles["pi"], dtype=float)[:, None]
        conditional = (1.0 - pi) * mean
        variance = conditional * (1.0 + pi * mean)
    elif family == "lognormal":
        sigma = np.asarray(particles["sigma"], dtype=float)[:, None]
        variance = mean**2 * np.expm1(sigma**2)
    else:
        variance = mean**2 / np.asarray(particles["shape"], dtype=float)[:, None]
    predictive_mean = conditional.mean(axis=0)
    second = (variance + conditional**2).mean(axis=0)
    return predictive_mean, np.sqrt(np.maximum(second - predictive_mean**2, 0.0))
