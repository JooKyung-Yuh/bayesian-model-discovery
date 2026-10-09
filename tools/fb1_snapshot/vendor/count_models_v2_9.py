"""Paper CORE hypotheses, outside the registered inference snapshot.

The expression is a logit, not a count mean. One validated AST is interpreted
with NumPy for prediction and PyTensor for inference; no Python eval is used.

Patch v2.4 makes the expression namespace NAMED: the grammar accepts exactly the
input names an `interface` block declares, the derived names that block defines,
and the candidate's own declared coefficients - nothing else. `u`/`z` are no
longer hard-coded; they are simply the names CORE's interface happens to declare
(`inputs [{"name": "u", "range": [0, 1]}]`, `derived_inputs {"z": "2*u-1"}`),
which is why `DEFAULT_INPUT_NAMES` reproduces the v2.3 behaviour exactly.
"""

import ast
import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import betabinom, binom

FAMILIES = ("binomial", "beta_binomial", "zi_binomial")
# Table 2 row 5 ("| Binary | logit | Bernoulli | none ... |"): the binary row is
# the SAME logit-link count grammar with one trial, so it shares `Hypothesis`,
# `particle_logpmf` and the whole count runner instead of duplicating them.
# Bernoulli(p) is Binomial(1,p) - the combinatorial term is log(1) = 0 - so the
# exhaustive count table over {0,1} is already the exact Bernoulli PMF (patch
# v2.7, engine task 21d).
BINARY_FAMILIES = ("bernoulli",)
BINARY_TRIALS = 1
PRIOR_RULES = {
    "intercept": ("Normal", 0.0, 2.0),
    "signed": ("Normal", 0.0, 1.0),
    "positive": ("LogNormal", 0.0, 1.0),
    "shape": ("Gamma", 2.0, 1.0),
    "rho": ("Beta", 1.0, 9.0),
    "pi": ("Beta", 1.0, 4.0),
}
CALL_ARITY = {"exp": 1, "log": 1, "sqrt": 1, "tanh": 1, "square": 1, "power": 2}
# Numeric-rule bookkeeping (v2.4a, CHANGES_v2_4a.md). The NaN -> -inf rule in
# `particle_logpmf` must never be SILENT: a systematic run of substitutions is a
# bug signal, a rare one is a genuine impossibility, and only a count tells them
# apart. Process-local and monotone - `nan_substitution_counter()` returns a
# snapshot and never resets, so a caller reads DIFFERENCES around the work it
# wants to attribute (core_count.refit does exactly that). No prior, likelihood
# or constant is involved; this counts events, it does not change any of them.
_NAN_SUBSTITUTIONS = {"cells": 0, "calls": 0}


def nan_substitution_counter():
    """Snapshot {cells, calls} of the NaN -> -inf substitutions made so far."""
    return dict(_NAN_SUBSTITUTIONS)


# Names a candidate may never use for a coefficient, on top of the interface's
# own input names: the family-owned dispersion parameters and the call names.
RESERVED_NAMES = ("rho", "pi", *CALL_ARITY)

# --------------------------------------------------------------------------
# Manuscript v0.7 Table 2 - "Data type -> initial family -> admissible
# revisions (F)" (paper/PAPER_MANUSCRIPT_v0_7_20260907.md lines 101-110),
# copied verbatim as a code constant. The table rows, quoted:
#
#   | Data type | Link g | Initial family | Revisions a proposal may introduce | Notes |
#   | Bounded count (0..N) | logit | Binomial(N,p) | Beta-Binomial(rho);
#       zero-inflated Binomial(pi) | ZI-Binomial predictive mean (1-pi)Np,
#       scored as such |
#   | Unbounded count | log | Poisson(mu) | Negative Binomial(alpha);
#       zero-inflated Poisson only where zeros are natural | |
#   | Real outcome | identity | Normal(mu,sigma) | Student-t (nu=4);
#       heteroscedastic Normal, log sigma(x) = a0 + a1 z_j | sigma of the
#       Student-t is its scale, so its variance is 2 sigma^2 at nu=4 |
#   | Positive outcome | log | LogNormal with mean mu (location log mu -
#       sigma^2/2) | Gamma with mean mu; heteroscedastic LogNormal | mu is the
#       conditional mean in every family |
#   | Binary | logit | Bernoulli | none (probit/cloglog are link alternatives,
#       not families; not counted as F) | |
#   | Trajectory / feature vector | identity | Gaussian, task-specific scale |
#       Student-t where implemented | component-wise scores are composite and
#       labelled so |
#
# `LINK_BY_OUTCOME` is column 2 of that table: the link is FIXED by the outcome
# type and is never proposed. `OUTCOME_MENUS` is columns 3-4: the initial family
# and the full admissible set (initial + revisions). Engine-internal family
# identifiers are used; the mapping to the table's names is in the values.
# `IMPLEMENTED_FAMILIES` records which of those a runner in this tree can
# actually fit today - a menu entry outside it is a declared, not a runnable,
# revision, and `allowed_families` intersects with it.
# --------------------------------------------------------------------------
LINK_BY_OUTCOME = {
    "bounded_count": "logit",
    "binary": "logit",
    "unbounded_count": "log",
    "positive": "log",
    "real": "identity",
    "trajectory": "identity",
}
OUTCOME_MENUS = {
    "bounded_count": {
        "initial_family": "binomial",
        "allowed_families": ("binomial", "beta_binomial", "zi_binomial"),
    },
    "binary": {"initial_family": "bernoulli", "allowed_families": ("bernoulli",)},
    "unbounded_count": {
        "initial_family": "poisson",
        "allowed_families": ("poisson", "negative_binomial", "zi_poisson"),
    },
    "real": {
        "initial_family": "normal",
        "allowed_families": ("normal", "student_t", "heteroscedastic_normal"),
    },
    "positive": {
        "initial_family": "lognormal",
        "allowed_families": ("lognormal", "gamma", "heteroscedastic_lognormal"),
    },
    "trajectory": {"initial_family": "gaussian_vector", "allowed_families": ("gaussian_vector",)},
}
# What this tree can fit. bounded_count: count_inference (PyMC binomial menu).
# binary: the same count runner with one trial (patch v2.7). unbounded_count/
# positive: positive_inference (PyMC Poisson/NegBin/ZIP/LogNormal/Gamma). real:
# real_inference (PyMC Normal/StudentT(nu=4)/heteroscedastic Normal, patch v2.7).
# Everything else is declared by Table 2 but has no runner here yet
# (`heteroscedastic_lognormal`, `gaussian_vector` outside the vector path).
IMPLEMENTED_FAMILIES = (
    "binomial",
    "beta_binomial",
    "zi_binomial",
    "bernoulli",
    "poisson",
    "negative_binomial",
    "zi_poisson",
    "lognormal",
    "gamma",
    "normal",
    "student_t",
    "heteroscedastic_normal",
)

# CORE's own interface, as the named-input grammar sees it. These two constants
# are the ONLY place `u` and `z` are privileged, and they are privileged only as
# defaults that reproduce v2.3 byte for byte.
CORE_INPUTS = ({"name": "u", "range": (0.0, 1.0)},)
CORE_DERIVED_INPUTS = {"z": "2*u-1"}
DEFAULT_INPUT_NAMES = ("u", "z")


def standardized_expression(name, low, high):
    """The engine's standardization x~ = 2(x-lo)/(hi-lo) - 1 as a derived input.

    For CORE (`name="u"`, range [0,1]) this is `2*(u-0)/(1-0)-1`, which evaluates
    to exactly the same floats as the declared `2*u-1` (subtracting 0.0 and
    dividing by 1.0 are exact in IEEE-754); `test_core_standardization_matches`
    checks the two agree bit for bit. CORE keeps the short literal so that its
    packets stay byte-identical to v2.3.
    """
    low, high = float(low), float(high)
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("input range must be a finite interval with hi > lo")
    return f"2*({name}-{low!r})/({high!r}-{low!r})-1"


def _validate_name(name):
    if not isinstance(name, str) or not name.isidentifier() or name in RESERVED_NAMES:
        raise ValueError(f"invalid or reserved input name: {name!r}")
    return name


def interface_input_names(interface):
    """The complete grammar namespace of an interface: base inputs, then derived.

    Raises on a malformed block, so a task specification cannot silently widen
    the grammar. A derived expression may reference base inputs and derived
    names declared before it - never a coefficient, never an undeclared name.
    """
    inputs = interface["inputs"]
    if not isinstance(inputs, (list, tuple)) or not inputs:
        raise ValueError("interface must declare at least one input")
    names = []
    for item in inputs:
        name = _validate_name(item["name"])
        low, high = (float(v) for v in item["range"])
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError(f"input {name} needs a finite range with hi > lo")
        if name in names:
            raise ValueError(f"duplicate input name {name}")
        names.append(name)
    derived = interface.get("derived_inputs") or {}
    if not isinstance(derived, dict):
        raise TypeError("derived_inputs must be a mapping of name to expression")
    for name, expression in derived.items():
        _validate_name(name)
        if name in names:
            raise ValueError(f"derived input {name} shadows an input name")
        for node in ast.walk(ast.parse(expression, mode="eval")):
            if isinstance(node, ast.Name) and node.id not in names and node.id not in CALL_ARITY:
                raise ValueError(f"unknown input name {node.id} in derived input {name}")
        names.append(name)
    return tuple(names)


def named_input_values(interface, designs, backend=np):
    """Values of every declared and derived input at `designs`.

    `designs` is a 1-D array when the interface declares one input, and an
    [observation, input] matrix (or a mapping name -> array) when it declares
    several. Derived inputs are evaluated by the same restricted AST walker as
    expressions, over the input namespace only.
    """
    names = interface_input_names(interface)
    base = [item["name"] for item in interface["inputs"]]
    if isinstance(designs, dict):
        values = {name: designs[name] for name in base}
    elif len(base) == 1:
        values = {base[0]: designs}
    else:
        columns = np.asarray(designs, dtype=float)
        if columns.ndim != 2 or columns.shape[1] != len(base):
            raise ValueError("multi-input designs must be an [observation, input] matrix")
        values = {name: columns[:, index] for index, name in enumerate(base)}
    for name in names[len(base) :]:
        values[name] = evaluate(
            interface["derived_inputs"][name], {}, next(iter(values.values())), backend, values
        )
    return values


def validate_expression(expression, parameters, input_names=DEFAULT_INPUT_NAMES):
    """The ONE expression grammar, shared by every runner (count, binary, real).

    Accepts exactly: the interface's input names (base + derived), the
    candidate's own declared coefficients, finite real literals, + - * / **,
    unary +/-, and the six calls in CALL_ARITY. At most 6 distinct coefficients,
    at most one intercept, roles from the fixed four, at most 15 mean operators.
    An identifier outside that namespace raises "unknown input name <id>" -
    which the per-item judge turns into a candidate rejection with that verbatim
    reason (v2.4; the v2.3 message said "unknown expression name").
    """
    inputs = tuple(input_names)
    if not inputs or len(set(inputs)) != len(inputs):
        raise ValueError("interface must declare a non-empty set of distinct input names")
    roles = dict(parameters)
    if not 1 <= len(roles) <= 6 or len(roles) != len(parameters):
        raise ValueError("require 1..6 distinct mean coefficients")
    if any(role not in ("intercept", "signed", "positive", "shape") for role in roles.values()):
        raise ValueError("coefficient role outside fixed prior policy")
    if sum(role == "intercept" for role in roles.values()) > 1:
        raise ValueError("at most one intercept")
    if any(not name.isidentifier() or name in {*inputs, *RESERVED_NAMES} for name in roles):
        raise ValueError("invalid or reserved coefficient name")
    tree = ast.parse(expression, mode="eval")
    used, operators = set(), 0

    def validate(node):
        nonlocal operators
        if isinstance(node, ast.Expression):
            validate(node.body)
        elif isinstance(node, ast.Name):
            if node.id not in {*inputs, *roles}:
                raise ValueError(f"unknown input name {node.id}")
            if node.id in roles:
                used.add(node.id)
        elif isinstance(node, ast.Constant):
            if type(node.value) not in (int, float) or not math.isfinite(node.value):
                raise ValueError("only finite real literals allowed")
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
        ):
            operators += 1
            validate(node.left)
            validate(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operators += 1
            validate(node.operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in CALL_ARITY or node.keywords:
                raise ValueError("call outside expression grammar")
            if len(node.args) != CALL_ARITY[node.func.id]:
                raise ValueError("wrong expression call arity")
            operators += 1
            for arg in node.args:
                validate(arg)
        else:
            raise TypeError(f"unsupported expression syntax: {ast.dump(node)}")

    validate(tree)
    if used != set(roles) or operators > 15:
        raise ValueError("unused coefficient or more than 15 mean operators")


def canonical_model_id(expression, family, parameters):
    """sha256[:20] of the canonical AST + coefficient roles + family.

    Alpha-renaming cannot duplicate prior mass; roles stay in the identity, and
    every input the expression references is in it verbatim as an `["input",
    name]` leaf, so the input-name set is part of the identity (v2.4) without a
    new hash component - which is why CORE ids are unchanged from v2.3.
    """
    roles, aliases = dict(parameters), {}

    def canonical(node):
        if isinstance(node, ast.Expression):
            return canonical(node.body)
        if isinstance(node, ast.Name):
            if node.id in roles:
                aliases.setdefault(node.id, len(aliases))
                return ["parameter", aliases[node.id], roles[node.id]]
            return ["input", node.id]
        if isinstance(node, ast.Constant):
            return ["number", float(node.value)]
        if isinstance(node, ast.BinOp):
            return [type(node.op).__name__, canonical(node.left), canonical(node.right)]
        if isinstance(node, ast.UnaryOp):
            return [type(node.op).__name__, canonical(node.operand)]
        return [node.func.id, *[canonical(arg) for arg in node.args]]

    raw = json.dumps(
        [family, canonical(ast.parse(expression, mode="eval"))], separators=(",", ":")
    ).encode()
    return hashlib.sha256(raw).hexdigest()[:20]


@dataclass(frozen=True)
class Hypothesis:
    """One candidate: an expression over NAMED inputs, a family, and coefficient roles.

    `input_names` is the interface namespace the expression is validated against
    (base inputs plus derived inputs, from `interface_input_names`). It defaults
    to CORE's `("u", "z")`, so every v2.3 construction keeps its exact meaning.
    It is deliberately NOT part of `model_id`: the canonical AST already carries
    every input the expression actually references, verbatim, as `["input",
    name]` leaves, so `b0+b1*z` and `b0+b1*tz` already hash differently while
    CORE's ids stay bit-identical to v2.3 (`test_model_id_separates_input_names`
    and the CORE byte anchor both check this).

    `family` is checked against `IMPLEMENTED_FAMILIES` (patch v2.5c), not the
    narrower CORE-only `FAMILIES`. `count_proposer.judge_candidate`/
    `parse_candidate_object`/`parse_candidates` always build a `Hypothesis`
    here, whatever the task - they import this class, not a task's own (see
    `boxinggym_core.make`'s `Hypothesis if bounded else ScalarHypothesis`) - so
    a grammar-valid LLM candidate for any family this tree can fit must be able
    to reach this constructor, even on a task whose native class is
    `positive_inference.ScalarHypothesis`. WHICH families a given task/cell may
    actually be proposed is a separate, task-level question: `judge_candidate`'s
    own `allowed_families` argument enforces that, before this constructor ever
    runs.
    """

    expression: str
    family: str
    parameters: tuple[tuple[str, str], ...]
    input_names: tuple[str, ...] = DEFAULT_INPUT_NAMES

    def __post_init__(self):
        if self.family not in IMPLEMENTED_FAMILIES:
            raise ValueError(f"unsupported count family: {self.family}")
        validate_expression(self.expression, self.parameters, self.input_names)

    @property
    def noise_parameters(self):
        return {
            "binomial": (),
            "bernoulli": (),
            "beta_binomial": ("rho",),
            "zi_binomial": ("pi",),
        }[self.family]

    @property
    def expression_inputs(self):
        """The input names this expression actually references, sorted.

        These are exactly the names that enter `model_id` through the canonical
        AST, which is why the identity separates two otherwise identical
        expressions written over different interfaces.
        """
        roles = dict(self.parameters)
        return tuple(
            sorted(
                {
                    node.id
                    for node in ast.walk(ast.parse(self.expression, mode="eval"))
                    if isinstance(node, ast.Name) and node.id not in roles
                }
            )
        )

    @property
    def model_id(self):
        return canonical_model_id(self.expression, self.family, self.parameters)

    def as_dict(self):
        return {
            "model_id": self.model_id,
            "expression": self.expression,
            "family": self.family,
            "parameters": dict(self.parameters),
            "noise_parameters": self.noise_parameters,
            "fixed_priors": {name: PRIOR_RULES[role] for name, role in self.parameters}
            | {name: PRIOR_RULES[name] for name in self.noise_parameters},
        }


def evaluate(expression, parameters, u, backend=np, inputs=None):
    """Interpret one validated expression.

    `inputs` is the named-input mapping (from `named_input_values`); when it is
    None the CORE namespace `{"u": u, "z": 2*u-1}` is used, which is exactly the
    v2.3 behaviour. `u` remains the broadcasting anchor in both cases.
    """
    values = {**parameters, **({"u": u, "z": 2 * u - 1} if inputs is None else inputs)}

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            return a**b
        if isinstance(node, ast.UnaryOp):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        args = [walk(arg) for arg in node.args]
        if node.func.id == "square":
            return args[0] ** 2
        if node.func.id == "power":
            return args[0] ** args[1]
        return getattr(backend, node.func.id)(*args)

    return walk(ast.parse(expression, mode="eval")) + 0 * u


def particle_logpmf(hypothesis, particles, u, n_trials=50, interface=None):
    """Normalized log PMFs: [particle, input, count], including combinatorics.

    With `interface` given, the designs are read through the named-input block
    (one column per declared input, plus the derived inputs) instead of the CORE
    `u`/`z` pair. With `interface=None` the CORE namespace is used and every
    float is identical to v2.3.
    """
    u = np.asarray(u, dtype=float)
    if interface is None:
        if u.ndim != 1 or not np.all(np.isfinite(u)):
            raise ValueError("designs must be a finite vector")
        named, anchor = None, u[None, :]
    else:
        if u.ndim != (1 if len(interface["inputs"]) == 1 else 2) or not np.all(np.isfinite(u)):
            raise ValueError("designs must be a finite vector or [observation, input] matrix")
        named = {
            name: np.asarray(value, dtype=float)[None, :]
            for name, value in named_input_values(interface, u).items()
        }
        anchor = named[interface["inputs"][0]["name"]]
    params = {
        name: np.asarray(particles[name], dtype=float)[:, None] for name, _ in hypothesis.parameters
    }
    sizes = {value.shape[0] for value in params.values()}
    if len(sizes) != 1 or next(iter(sizes)) == 0:
        raise ValueError("particle columns differ or are empty")
    with np.errstate(all="ignore"):
        logits = evaluate(hypothesis.expression, params, anchor, np, named)
    if not np.all(np.isfinite(logits)):
        raise ValueError("non-finite mean expression; no clipping fallback")
    p = expit(logits)[:, :, None]
    ys = np.arange(n_trials + 1)[None, None, :]
    if hypothesis.family == "bernoulli":
        # Table 2 row 5. Binomial(1,p) IS Bernoulli(p) - log C(1,y) = 0 exactly -
        # so this is the same closed form, guarded to one trial.
        if n_trials != BINARY_TRIALS:
            raise ValueError("a Bernoulli likelihood has exactly one trial")
        result = binom.logpmf(ys, BINARY_TRIALS, p)
    elif hypothesis.family == "binomial":
        result = binom.logpmf(ys, n_trials, p)
    elif hypothesis.family == "beta_binomial":
        rho = np.asarray(particles["rho"])[:, None, None]
        if np.any((rho <= 0) | (rho >= 1)):
            raise ValueError("rho outside open unit interval")
        kappa = 1 / rho - 1
        result = betabinom.logpmf(ys, n_trials, p * kappa, (1 - p) * kappa)
        # Continuous endpoint limits of the mean-parameterized beta-binomial.
        result = np.where((p == 0) | (p == 1), binom.logpmf(ys, n_trials, p), result)
    else:
        pi = np.asarray(particles["pi"])[:, None, None]
        if np.any((pi <= 0) | (pi >= 1)):
            raise ValueError("pi outside open unit interval")
        result = binom.logpmf(ys, n_trials, p) + np.log1p(-pi)
        result[:, :, 0] = np.logaddexp(result[:, :, 0], np.log(pi[:, :, 0]))
    # Numeric rule (v2.4a, CHANGES_v2_4a.md): an UNDEFINED particle log pmf is
    # IMPOSSIBLE, not a number - NaN is DEFINED as -inf. No prior, no likelihood
    # and no constant changes: on the domain where the log pmf is a number the
    # returned array is bit-identical to v2.4. Only cells that v2.4 refused
    # outright now carry probability zero, so a downstream weight is 0, never NaN.
    nan_cells = np.isnan(result)
    nan_count = int(nan_cells.sum())
    if nan_count:
        # Counted, never silent - see `nan_substitution_counter`.
        _NAN_SUBSTITUTIONS["cells"] += nan_count
        _NAN_SUBSTITUTIONS["calls"] += 1
        result = np.where(nan_cells, -np.inf, result)
    if np.any(np.isposinf(result)):
        raise ValueError("invalid likelihood")
    if not np.allclose(np.exp(result).sum(axis=-1), 1.0, atol=1e-10, rtol=0):
        # A NaN cell that carried real mass leaves its row unnormalized. That is
        # the same non-finite defect v2.4 rejected, and it keeps the SAME message
        # so core_count.PRIOR_PROBE_NONFINITE classifies it exactly as before; a
        # row that still normalizes was NaN only where the mass is negligible.
        raise ValueError(
            "invalid likelihood" if np.any(nan_cells) else "likelihood normalization failed"
        )
    return result


CORE_INITIAL_STRUCTURES = ("b0", "b0+b1*z", "b0+b1*z+b2*square(z)", "b0+b1*log(1+u)")


def initial_hypotheses(family="binomial", forms=None, input_names=DEFAULT_INPUT_NAMES):
    """The four initial structures, written in the interface's own input names.

    `forms` defaults to CORE's four; a task specification supplies its own list
    written over its own declared inputs (see `boxinggym_core.TASKS`).
    """
    forms = CORE_INITIAL_STRUCTURES if forms is None else tuple(forms)
    result = []
    for form in forms:
        names = sorted(
            {
                n.id
                for n in ast.walk(ast.parse(form))
                if isinstance(n, ast.Name) and n.id.startswith("b")
            }
        )
        result.append(
            Hypothesis(
                form,
                family,
                tuple((name, "intercept" if name == "b0" else "signed") for name in names),
                tuple(input_names),
            )
        )
    return result


def judge_candidate(
    item, cell, initial_structures, initial_family, input_names=None, allowed_families=None
):
    """Judge one already shape-valid candidate; never raises (v2.2 fail-soft).

    `item` must already be a dict with expression/family/parameters (parameters
    itself a dict of coefficient name -> role, with any reserved variance names
    such as rho/pi already filtered out by the caller - see
    count_proposer.RESERVED_VARIANCE_PARAMETER_NAMES/CHANGES_v2_1.md). Shape
    violations (missing keys, non-dict parameters) are the caller's concern,
    not this function's - both current callers (count_proposer.parse_candidates
    and core_count.validate_cell_candidates) only ever pass shape-valid items.

    Covers exactly the three per-candidate violation categories from
    CHANGES_v2_2.md: grammar (AST/coefficient/operator rules, via attempting
    Hypothesis construction), role (the fixed 4-role policy, same
    construction), and cell constraint (S0: expression must reduce to one of
    initial_structures; F0: family must be initial_family). `initial_structures`
    is the caller's one-time `initial_hypotheses(initial_family)` list, reused
    across every item in a batch rather than recomputed per candidate.

    Returns (True, None) when the candidate may proceed to admission, else
    (False, reason) with a short, stable, human-readable reason - either a
    count_models.Hypothesis ValueError/TypeError message verbatim, or one of
    the two literal cell-constraint messages below. An identifier outside the
    interface namespace is a per-item rejection whose reason is the verbatim
    "unknown input name <id>" (patch v2.4).

    `input_names` is the interface namespace; when omitted it is taken from
    `initial_structures` (which the caller builds from the same interface), and
    falls back to CORE's `("u", "z")` only when there are no initial structures.

    `allowed_families` (patch v2.5c) is this task/cell's admissible family menu
    - `belief_packet.cell_constraints`'s `constraints["allowed_families"]`,
    itself `belief_packet.interface_families`'s outcome-type menu (manuscript
    v0.7 Table 2) intersected with `IMPLEMENTED_FAMILIES`. It defaults to None,
    which means CORE's own `FAMILIES` (binomial/beta_binomial/zi_binomial), so
    every existing call site (core_count.validate_cell_candidates, and any
    count_proposer caller that does not pass this argument) keeps CORE's exact
    behaviour unchanged. A candidate whose family is outside the allowed set is
    rejected here, before Hypothesis is even constructed, with the SAME literal
    message Hypothesis.__post_init__ has always raised for a family it does not
    recognize at all: "unsupported count family: <family>". This is a
    task-level admissibility check layered on top of - not a replacement for -
    Hypothesis's own (wider) `IMPLEMENTED_FAMILIES` check, which still guards
    every other caller that constructs a Hypothesis directly.
    """
    names = (
        tuple(initial_structures[0].input_names)
        if input_names is None and initial_structures
        else tuple(DEFAULT_INPUT_NAMES if input_names is None else input_names)
    )
    allowed = FAMILIES if allowed_families is None else tuple(allowed_families)
    try:
        if item["family"] not in allowed:
            raise ValueError(f"unsupported count family: {item['family']}")
        hypothesis = Hypothesis(
            item["expression"], item["family"], tuple(item["parameters"].items()), names
        )
        structure_expanded, family_expanded = (bit == "1" for bit in cell[:2])
        if not family_expanded and hypothesis.family != initial_family:
            return False, "F0 requires the initial likelihood family"
        if not structure_expanded:
            initial_ids = {h.model_id for h in initial_structures}
            structure_id = Hypothesis(
                hypothesis.expression, initial_family, hypothesis.parameters, names
            ).model_id
            if structure_id not in initial_ids:
                return False, "S0 requires a structure from the initial pool"
        return True, None
    except Exception as exc:  # noqa: BLE001 - fail-soft judge: reject, never raise
        return False, str(exc)
