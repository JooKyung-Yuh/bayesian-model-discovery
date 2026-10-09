"""Identity-link sequential Monte Carlo (SMC) for the paper's task table REAL-outcome row.

The paper's task table, row 3, quoted verbatim (the draft paper is not
shipped in this repository):

    | Real outcome | identity | Normal$(\\mu,\\sigma)$ | Student-$t$ ($\\nu=4$);
      heteroscedastic Normal, $\\log\\sigma(x)=a_0+a_1 z_j$ | $\\sigma$ of the
      Student-$t$ is its scale, so its variance is $2\\sigma^2$ at $\\nu=4$ |

So: the link is the identity (never proposed - `count_models.LINK_BY_OUTCOME`),
the initial family is Normal, and a proposal may introduce exactly two
revisions, Student-t with nu FIXED at 4 and the heteroscedastic Normal whose
log scale is affine in one standardized input.

Parameter roles and their priors are the paper's prior table, quoted verbatim
for every parameter this module owns:

    | Role | Prior |
    | standardised slope | $\\mathcal N(0,1)$ |
    | intercept | $\\mathcal N(0,2)$ |
    | output scale $\\sigma$ (standardised outputs) | HalfNormal$(1)$ |
    | Student-$t$ degrees of freedom | fixed at 4 |

`sigma` is the "output scale (standardised outputs)" row, so the outcome is put
on a standardized footing before that prior is applied: a location and scale frozen
on the initial design set (D0) - `response_center` and `response_unit`, read from
the fit's diagnostics; the identity-link twin of the log-link `response_unit` - turn
the grammar expression into the conditional mean and turn `sigma` into the scale of
the STANDARDIZED outcome, exactly as the row's parenthesis requires:

    mu(x)    = center + unit * expression(x)
    sigma_y  = unit * sigma                       (normal, student_t)
    sigma_y(x) = unit * exp(a0 + a1 * z_j)        (heteroscedastic_normal)

`center` and `unit` are frozen from D0 before any round-1 observation exists and
never re-estimated, so every model in a pool shares one likelihood measure and
the log evidences stay comparable.

The heteroscedastic scale coefficients introduce NO new prior: Table 2 names
them `a_0` (an intercept) and `a_1` (a standardised slope), and Table 3 already
fixes those two roles, so `a0 ~ Normal(0,2)` and `a1 ~ Normal(0,1)` are the same
two rows the mean coefficients use. `REAL_NOISE_PRIORS` below reads them out of
`count_models.PRIOR_RULES` rather than restating any number.

The mean grammar is `count_models.validate_expression` - the SAME grammar CORE's
bounded-count hypotheses and the log-link `ScalarHypothesis` use - and the
identity is the same canonical AST + roles + family hash. Nothing in this module
is imported by, or changes, the bounded-count or positive-outcome paths.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t_dist

from .count_models import (
    DEFAULT_INPUT_NAMES,
    PRIOR_RULES,
    canonical_model_id,
    evaluate,
    model_id_preimage,
    named_input_values,
    validate_expression,
)

# Table 2 row 3: initial family + the two admissible revisions, in menu order.
REAL_FAMILIES = ("normal", "student_t", "heteroscedastic_normal")
# Table 3: "Student-$t$ degrees of freedom | fixed at 4". Never fitted, never
# proposed; the table's own note ("its variance is 2 sigma^2 at nu=4") is the
# moment this constant implies and `mixture_moments` uses.
STUDENT_T_DF = 4.0
# The scale coefficients Table 2 names for the heteroscedastic row.
HETEROSCEDASTIC_SCALE_PARAMETERS = (("a0", "intercept"), ("a1", "signed"))
REAL_NOISE = {
    "normal": ("sigma",),
    "student_t": ("sigma",),
    "heteroscedastic_normal": ("a0", "a1"),
}
# Priors of the family-owned scale parameters, as Table 3 fixes them. `sigma` is
# the "output scale (standardised outputs) | HalfNormal(1)" row; `a0`/`a1` are
# the "intercept | N(0,2)" and "standardised slope | N(0,1)" rows, read out of
# PRIOR_RULES so no number is restated here.
REAL_NOISE_PRIORS = {
    "sigma": ("HalfNormal", 1.0, None),
    "a0": PRIOR_RULES["intercept"],
    "a1": PRIOR_RULES["signed"],
}


def default_scale_expression(interface=None):
    """Table 2's `log sigma(x) = a0 + a1 z_j`, over the interface's first z_j.

    `z_j` is a STANDARDIZED input, so the affine form is written over the first
    derived input the interface declares whose name ends in `z` (the engine's own
    standardization - `count_models.standardized_expression`), or over the first
    derived input if none does; with no interface it is the input `z`. The rule
    is mechanical and fixed here, before any run; a task never chooses it and a
    proposal never proposes it.
    """
    if interface is None:
        return "a0+a1*z"
    derived = list(interface.get("derived_inputs") or {})
    if not derived:
        raise ValueError("a heteroscedastic scale needs a declared standardized input")
    standardized = [name for name in derived if name.endswith("z")]
    return f"a0+a1*{(standardized or derived)[0]}"


@dataclass(frozen=True)
class RealHypothesis:
    """One candidate for a real outcome: an identity-link mean plus a scale form.

    `expression` is the mean in STANDARDIZED response units (see the module
    docstring); `scale_expression` is only used by `heteroscedastic_normal`, and
    for that family it is part of the identity - two heteroscedastic models with
    the same mean but different scale inputs are different models.
    """

    expression: str
    family: str
    parameters: tuple[tuple[str, str], ...]
    input_names: tuple[str, ...] = DEFAULT_INPUT_NAMES
    scale_expression: str | None = None

    def __post_init__(self):
        if self.family not in REAL_FAMILIES:
            raise ValueError(f"unsupported real family: {self.family}")
        validate_expression(self.expression, self.parameters, self.input_names)
        if self.family == "heteroscedastic_normal":
            if self.scale_expression is None:
                object.__setattr__(self, "scale_expression", "a0+a1*z")
            validate_expression(
                self.scale_expression, HETEROSCEDASTIC_SCALE_PARAMETERS, self.input_names
            )
            clash = {name for name, _ in self.parameters} & {"a0", "a1"}
            if clash:
                raise ValueError(f"mean coefficient collides with the scale coefficient: {clash}")
        elif self.scale_expression is not None:
            raise ValueError(f"family {self.family} has no scale expression")

    @property
    def identity_family(self):
        """The family string that enters `model_id`, carrying the scale form."""
        if self.family == "heteroscedastic_normal":
            return f"{self.family}|log_sigma={self.scale_expression}"
        return self.family

    @property
    def model_id(self):
        """Canonical model id hashed from the mean expression, family (with scale form), roles."""
        return canonical_model_id(self.expression, self.identity_family, self.parameters)

    @property
    def model_id_preimage(self):
        """The exact string that `model_id` hashes, for audit and debugging."""
        return model_id_preimage(self.expression, self.identity_family, self.parameters)

    @property
    def noise_parameters(self):
        """Names of the family-owned scale parameters (`sigma`, or `a0`/`a1`)."""
        return REAL_NOISE[self.family]

    @property
    def measure(self):
        """The likelihood measure of a real outcome: always a density ("PDF")."""
        return "PDF"

    def as_dict(self):
        """Record form of the hypothesis: id, expression, family, mean/scale forms and priors."""
        item = {
            "model_id": self.model_id,
            "expression": self.expression,
            "family": self.family,
            "mean": "response_center + response_unit * expression",
            "parameters": dict(self.parameters),
            "noise_parameters": self.noise_parameters,
            "measure": self.measure,
            "priors": {
                "intercept": list(PRIOR_RULES["intercept"]),
                "signed": list(PRIOR_RULES["signed"]),
                "positive": list(PRIOR_RULES["positive"]),
                "shape": list(PRIOR_RULES["shape"]),
                "sigma": ["HalfNormal", 1],
            },
        }
        if self.family == "student_t":
            item["fixed"] = {"nu": STUDENT_T_DF}
        if self.family == "heteroscedastic_normal":
            item["scale_expression"] = self.scale_expression
            item["scale"] = "response_unit * exp(scale_expression)"
            item["priors"] = {
                **item["priors"],
                "a0": list(PRIOR_RULES["intercept"]),
                "a1": list(PRIOR_RULES["signed"]),
            }
        return item


def is_real(hypothesis):
    """True for the identity-link real path; False for every other hypothesis."""
    return getattr(hypothesis, "family", None) in REAL_FAMILIES


def _named(interface, inputs):
    designs = np.asarray(inputs, dtype=float)
    values = {
        name: np.asarray(value, dtype=float)[None, :]
        for name, value in named_input_values(interface, designs).items()
    }
    return values, values[interface["inputs"][0]["name"]]


def identity_means(hypothesis, particles, inputs, interface, center, unit):
    """`center + unit * expression` per [particle, input]; the identity link."""
    named, anchor = _named(interface, inputs)
    params = {
        name: np.asarray(particles[name], dtype=float)[:, None] for name, _ in hypothesis.parameters
    }
    with np.errstate(over="ignore", under="ignore"):
        mean = float(center) + float(unit) * evaluate(
            hypothesis.expression, params, anchor, np, named
        )
    if not np.all(np.isfinite(mean)):
        raise ValueError("non-finite mean expression; no clipping fallback")
    return mean


def scale_values(hypothesis, particles, inputs, interface, unit):
    """Predictive scale per [particle, input] in the ORIGINAL response units."""
    if hypothesis.family == "heteroscedastic_normal":
        named, anchor = _named(interface, inputs)
        params = {
            name: np.asarray(particles[name], dtype=float)[:, None]
            for name, _ in HETEROSCEDASTIC_SCALE_PARAMETERS
        }
        with np.errstate(over="ignore", under="ignore"):
            scale = float(unit) * np.exp(
                evaluate(hypothesis.scale_expression, params, anchor, np, named)
            )
    else:
        columns = np.asarray(inputs, dtype=float).shape[0]
        scale = float(unit) * np.asarray(particles["sigma"], dtype=float)[:, None] * np.ones(
            (1, columns)
        )
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("non-finite/nonpositive predictive scale; no clipping fallback")
    return scale


def _binding(result):
    diagnostics = result.diagnostics
    return (
        diagnostics["interface"],
        float(diagnostics.get("response_center", 0.0)),
        float(diagnostics["response_unit"]),
    )


def particle_means(result, inputs):
    """Identity-link means [particle, input] of a fitted SMC result at `inputs`."""
    interface, center, unit = _binding(result)
    return identity_means(result.hypothesis, result.particles, inputs, interface, center, unit)


def conditional_loglik(result, inputs, outcomes):
    """[particle, input, draw], the full Normal / Student-t log density."""
    interface, center, unit = _binding(result)
    mean = identity_means(result.hypothesis, result.particles, inputs, interface, center, unit)
    scale = scale_values(result.hypothesis, result.particles, inputs, interface, unit)
    y = np.asarray(outcomes, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] != mean.shape[1] or not np.all(np.isfinite(y)):
        raise ValueError("outcomes must be finite [input, draw]")
    location, spread = mean[:, :, None], scale[:, :, None]
    if result.hypothesis.family == "student_t":
        values = student_t_dist.logpdf(y[None, :, :], STUDENT_T_DF, loc=location, scale=spread)
    else:
        values = norm.logpdf(y[None, :, :], loc=location, scale=spread)
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("invalid conditional likelihood")
    return values


def conditional_cdf(result, inputs, outcomes, tail="lower"):
    """[particle, input, draw], the same law's CDF instead of its log density.

    `conditional_loglik`'s own three lines build `location` and `spread`, so
    this is that function with the density replaced by its integral - which is
    what turns a mixture's mass in an interval into an exact difference of two
    values rather than a quadrature of the density (adopted on 2026-09-18 for
    the emotion Likert cells).

    `tail="upper"` returns the SURVIVAL function, which is not `1 - cdf` in
    floating point: a pool ten scales below a cell's edge has a `cdf` that
    rounds to exactly 1, and the difference of two such values is 0 where the
    mass is merely very small. The caller takes each interval's mass on
    whichever tail carries it without cancellation.
    """
    if tail not in ("lower", "upper"):
        raise ValueError("tail must be lower or upper")
    interface, center, unit = _binding(result)
    mean = identity_means(result.hypothesis, result.particles, inputs, interface, center, unit)
    scale = scale_values(result.hypothesis, result.particles, inputs, interface, unit)
    y = np.asarray(outcomes, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] != mean.shape[1] or not np.all(np.isfinite(y)):
        raise ValueError("outcomes must be finite [input, draw]")
    location, spread = mean[:, :, None], scale[:, :, None]
    if result.hypothesis.family == "student_t":
        function = student_t_dist.sf if tail == "upper" else student_t_dist.cdf
        values = function(y[None, :, :], STUDENT_T_DF, loc=location, scale=spread)
    else:
        function = norm.sf if tail == "upper" else norm.cdf
        values = function(y[None, :, :], loc=location, scale=spread)
    if np.any(np.isnan(values)):
        raise ValueError("invalid conditional distribution function")
    return values


def draw_conditional(result, inputs, indices, rng):
    """Predictive draws [input, draw] for the given particle indices."""
    interface, center, unit = _binding(result)
    mean = identity_means(result.hypothesis, result.particles, inputs, interface, center, unit)
    scale = scale_values(result.hypothesis, result.particles, inputs, interface, unit)
    location, spread = mean[indices].T, scale[indices].T
    if result.hypothesis.family == "student_t":
        return location + spread * rng.standard_t(STUDENT_T_DF, size=location.shape)
    return rng.normal(location, spread)


def mixture_moments(hypothesis, particles, inputs, interface, center, unit):
    """Posterior predictive mean and sd of a real family, in closed form.

    Each particle contributes conditional mean m_i and conditional variance v_i
    (Normal and heteroscedastic Normal: sigma_i^2; Student-t at nu=4: 2 sigma^2,
    the value the paper's task table's own note states). The equally weighted
    particle mixture then has mean E[m] and variance E[v] + Var[m] - exact, so
    the belief packet's residual block needs no sampling.
    """
    mean = identity_means(hypothesis, particles, inputs, interface, center, unit)
    scale = scale_values(hypothesis, particles, inputs, interface, unit)
    variance = scale**2
    if hypothesis.family == "student_t":
        variance = variance * (STUDENT_T_DF / (STUDENT_T_DF - 2.0))
    predictive_mean = mean.mean(axis=0)
    second = (variance + mean**2).mean(axis=0)
    return predictive_mean, np.sqrt(np.maximum(second - predictive_mean**2, 0.0))


def build_model(hypothesis, inputs, labels, interface, center, unit):
    """PyMC model for one real hypothesis, in the ORIGINAL response units."""
    import pymc as pm
    import pytensor.tensor as pt

    designs = np.asarray(inputs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(designs) != len(y) or y.ndim != 1 or not len(y) or not np.all(np.isfinite(y)):
        raise ValueError("nonempty finite real observation vectors required")
    if not np.isfinite(unit) or unit <= 0 or not np.isfinite(center):
        raise ValueError("response scale must be a finite center and a positive unit")
    named = named_input_values(interface, designs)
    anchor = pt.as_tensor(np.asarray(named[interface["inputs"][0]["name"]]))
    with pm.Model() as model:
        parameters = {}
        for name, role in hypothesis.parameters:
            family, a, b = PRIOR_RULES[role]
            parameters[name] = getattr(pm, family)(name, a, b, transform=None)
        mu = float(center) + float(unit) * evaluate(
            hypothesis.expression, parameters, anchor, pt, named
        )
        if hypothesis.family == "heteroscedastic_normal":
            scale_parameters = {}
            for name, role in HETEROSCEDASTIC_SCALE_PARAMETERS:
                family, a, b = PRIOR_RULES[role]
                scale_parameters[name] = getattr(pm, family)(name, a, b, transform=None)
            sigma = float(unit) * pm.math.exp(
                evaluate(hypothesis.scale_expression, scale_parameters, anchor, pt, named)
            )
            pm.Normal("observed", mu=mu, sigma=sigma, observed=y)
        else:
            sigma = float(unit) * pm.HalfNormal("sigma", 1, transform=None)
            if hypothesis.family == "student_t":
                pm.StudentT("observed", nu=STUDENT_T_DF, mu=mu, sigma=sigma, observed=y)
            else:
                pm.Normal("observed", mu=mu, sigma=sigma, observed=y)
    return model


__all__ = [
    "HETEROSCEDASTIC_SCALE_PARAMETERS",
    "REAL_FAMILIES",
    "REAL_NOISE",
    "REAL_NOISE_PRIORS",
    "STUDENT_T_DF",
    "RealHypothesis",
    "build_model",
    "conditional_loglik",
    "default_scale_expression",
    "draw_conditional",
    "identity_means",
    "is_real",
    "mixture_moments",
    "particle_means",
    "scale_values",
]
