"""Paper CORE hypotheses, outside the registered inference snapshot.

The expression is a logit, not a count mean. One validated AST is interpreted
with NumPy for prediction and PyTensor for inference; no Python eval is used.

The expression namespace is NAMED: the grammar accepts exactly the
input names an `interface` block declares, the derived names that block defines,
and the candidate's own declared coefficients - nothing else. `u`/`z` are no
longer hard-coded; they are simply the names CORE's interface happens to declare
(`inputs [{"name": "u", "range": [0, 1]}]`, `derived_inputs {"z": "2*u-1"}`),
which is why `DEFAULT_INPUT_NAMES` reproduces the original hard-coded `u`/`z` behaviour exactly.
"""

import ast
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import betabinom, binom

from .model_identity import require_model_id_schema

FAMILIES = ("binomial", "beta_binomial", "zi_binomial")
# The F0 (family-expansion-off) refusal `judge_candidate` returns, as a named
# constant rather than a literal at its one site. The ChemBench runner
# applies the same per-candidate F0 rule on its own
# mock path - which never reaches `judge_candidate` at all, because it builds its
# candidate directly from a fixed schedule instead of parsing a proposer's JSON -
# and the two must not drift into two different words for one rule. The VALUE is
# unchanged, so every existing transcript, test and archive entry still matches.
F0_REFUSAL_REASON = "F0 requires the initial likelihood family"
# Table 2 row 5 ("| Binary | logit | Bernoulli | none ... |"): the binary row is
# the SAME logit-link count grammar with one trial, so it shares `Hypothesis`,
# `particle_logpmf` and the whole count runner instead of duplicating them.
# Bernoulli(p) is Binomial(1,p) - the combinatorial term is log(1) = 0 - so the
# exhaustive count table over {0,1} is already the exact Bernoulli PMF.
BINARY_FAMILIES = ("bernoulli",)
BINARY_TRIALS = 1
# --------------------------------------------------------------------------
# Prior-scale ablation (from the internal list of planned ablations): the
# role-prior table is EXTERNALIZED as three NAMED templates instead of
# one frozen dict. `default` is the paper's prior table - the same six rows, the same
# tuples, in the same order - and is what every production arm uses. `wide` and
# `narrow` are the ablation's x2 / x0.5 scale replacements, quoted from the
# ablation plan verbatim:
#
#   wide   | intercept N(0,4) | signed N(0,2) | positive LogNormal(0,2)
#          | shape Gamma(2,0.5) | rho Beta(1,4) | pi Beta(1,2)
#   narrow | intercept N(0,1) | signed N(0,0.5) | positive LogNormal(0,0.5)
#          | shape Gamma(2,2) | rho Beta(1,19) | pi Beta(1,9)
#
# `PRIOR_RULES` stays THE one name every reader uses - the likelihood
# (`count_inference.build_model`), the prior probe (`core_count.
# prior_particles`), the packet (`belief_packet.cell_constraints`), the record
# (`Hypothesis.as_dict`) and the proposer's declaration check
# (`count_proposer._prior_matches_declaration`) - so the switch is one named
# function here and NO call site changes. It is a live VIEW over the active
# template, not a copy: every `PRIOR_RULES[role]` is resolved at lookup time, so
# a template selected once at the top of a run reaches all of those readers, and
# `default` returns the identical tuples the frozen dict held. That is why both
# ablation switches off is byte-identical everywhere.
PRIOR_TEMPLATES = {
    "default": {
        "intercept": ("Normal", 0.0, 2.0),
        "signed": ("Normal", 0.0, 1.0),
        "positive": ("LogNormal", 0.0, 1.0),
        "shape": ("Gamma", 2.0, 1.0),
        "rho": ("Beta", 1.0, 9.0),
        "pi": ("Beta", 1.0, 4.0),
    },
    "wide": {
        "intercept": ("Normal", 0.0, 4.0),
        "signed": ("Normal", 0.0, 2.0),
        "positive": ("LogNormal", 0.0, 2.0),
        "shape": ("Gamma", 2.0, 0.5),
        "rho": ("Beta", 1.0, 4.0),
        "pi": ("Beta", 1.0, 2.0),
    },
    "narrow": {
        "intercept": ("Normal", 0.0, 1.0),
        "signed": ("Normal", 0.0, 0.5),
        "positive": ("LogNormal", 0.0, 0.5),
        "shape": ("Gamma", 2.0, 2.0),
        "rho": ("Beta", 1.0, 19.0),
        "pi": ("Beta", 1.0, 9.0),
    },
}
PRIOR_TEMPLATE_NAMES = tuple(PRIOR_TEMPLATES)
DEFAULT_PRIOR_TEMPLATE = "default"
# The plotting contract's `variant` tokens reserved for the prior-scale
# and the LLM-proposed-prior ablations.
# The default template emits none, so a default run's `variant` column is the
# one it had before the ablation existed.
PRIOR_VARIANT_TOKEN = {"default": "", "wide": "prior_wide", "narrow": "prior_narrow"}
LLM_PRIORS_VARIANT_TOKEN = "prior_llm"
_ACTIVE_PRIOR_TEMPLATE = DEFAULT_PRIOR_TEMPLATE
# LLM-proposed-prior ablation: whether this run declared `--priors llm`. Default off, and
# off is the pre-ablation behaviour in every respect - the SYSTEM prompt stays the
# byte-frozen v2.0 file, a `priors` key in a reply lands in `ignored_keys`, the
# packet publishes no menu, and no record gains a key.
_LLM_PRIORS = False


def active_prior_template():
    """The role-prior template this process is using (prior-scale ablation); `default` if unset."""
    return _ACTIVE_PRIOR_TEMPLATE


def set_prior_template(name):
    """Select the role-prior template for this process; returns the previous name.

    Called ONCE per run, by the runner, before any hypothesis is built or fitted
    (`core_count.run`, `boxinggym_core.run`). A name outside `PRIOR_TEMPLATES`
    raises rather than defaulting - an unrecognized ablation label must never
    silently become the production table.
    """
    global _ACTIVE_PRIOR_TEMPLATE
    if name not in PRIOR_TEMPLATES:
        raise ValueError(
            f"unregistered prior template {name!r}; registered: {list(PRIOR_TEMPLATE_NAMES)}"
        )
    previous, _ACTIVE_PRIOR_TEMPLATE = _ACTIVE_PRIOR_TEMPLATE, name
    return previous


def prior_template_table(name=None):
    """The template's hyperparameter table as JSON-ready lists, for config.json.

    The run records the template NAME and this TABLE, so a result never has to
    be read against the code that produced it (a rule of the prior-scale ablation).
    """
    table = PRIOR_TEMPLATES[name or active_prior_template()]
    return {role: [rule[0], rule[1], rule[2]] for role, rule in table.items()}


def active_grammar_version():
    """The expression grammar this process is using (v3 grammar extension). `v2` unless set."""
    return _ACTIVE_GRAMMAR_VERSION


def active_input_signs():
    """The registered sign class of each interface input, for the v3 domain proof.

    A mapping name -> sign set (see `SIGN_ANY`), empty unless the runner
    registered an interface. A name that is absent is treated as SIGN_ANY, so an
    unregistered namespace can only ever make the v3 proof FAIL - never pass.
    """
    return dict(_ACTIVE_INPUT_SIGNS)


def set_grammar_version(name, interface=None):
    """Select the expression grammar for this process; returns the previous state.

    Called ONCE per run, by the runner, before any hypothesis is built - the
    same discipline `set_prior_template` and `set_llm_priors` follow. A version
    outside `GRAMMAR_VERSIONS` raises rather than defaulting, so a mistyped
    option can never silently become the production gate.

    `interface` is this task's own interface block. It is read only under v3,
    where the domain proof needs each input's declared range; under v2 it is
    ignored entirely. Omitting it under v3 leaves every input at SIGN_ANY, which
    makes the proof strictly harder, never easier.
    """
    global _ACTIVE_GRAMMAR_VERSION, _ACTIVE_INPUT_SIGNS
    if name not in GRAMMAR_CALL_ARITY:
        raise ValueError(
            f"unregistered grammar version {name!r}; registered: {list(GRAMMAR_VERSIONS)}"
        )
    previous = (_ACTIVE_GRAMMAR_VERSION, dict(_ACTIVE_INPUT_SIGNS))
    _ACTIVE_GRAMMAR_VERSION = name
    _ACTIVE_INPUT_SIGNS = {} if interface is None else interface_input_signs(interface)
    return previous


def restore_grammar_state(previous):
    """Put back exactly what `set_grammar_version` returned.

    `set_grammar_version` takes an INTERFACE and derives the signs from it, so
    the pair it hands back cannot be fed to it again: passing only the version
    name silently clears the signs. A caller that has to put the process back
    the way it found it - the probe module, and any test that drives a runner in
    process - needs this instead. The runners themselves never call it: a run
    owns its process and selects the grammar once.
    """
    global _ACTIVE_GRAMMAR_VERSION, _ACTIVE_INPUT_SIGNS
    version, signs = previous
    if version not in GRAMMAR_CALL_ARITY:
        raise ValueError(
            f"unregistered grammar version {version!r}; registered: {list(GRAMMAR_VERSIONS)}"
        )
    _ACTIVE_GRAMMAR_VERSION = version
    _ACTIVE_INPUT_SIGNS = dict(signs)
    return version, dict(signs)


def llm_priors_enabled():
    """True when this run declared `--priors llm` (LLM-proposed-prior ablation). Default False."""
    return _LLM_PRIORS


def set_llm_priors(enabled):
    """Turn the LLM-proposed-prior path on/off for this process; returns the previous state.

    Called ONCE per run by the runner. It is read by `count_proposer` (which
    SYSTEM prompt file to load, and whether a reply's `priors` block is parsed
    or ignored), by `belief_packet.cell_constraints` (whether the menu and the
    ranges are published) and by `Hypothesis.as_dict` (whether `prior_source` is
    recorded), so the whole proposed-prior surface turns on and off together.
    """
    global _LLM_PRIORS
    previous, _LLM_PRIORS = _LLM_PRIORS, bool(enabled)
    return previous


# Opt-in stable evaluation of log(1 - exp(z)), added 2026-10-05. The 2026-10-04
# survival seed-43 EIG run died because the LLM wrote a Weibull CDF on the logit scale,
# `log(1-exp(-H))+H`: for H below ~1e-16, `1 - exp(-H)` rounds to 0, `log` gives -inf and
# `particle_logpmf` refuses the non-finite mean. With the switch on, `evaluate` computes
# the two spellings `log(1 - exp(z))` and `log1p(-exp(z))` as log1mexp(z) - the same
# function, evaluated with the standard branch (Maechler 2012): log(-expm1(z)) for
# z > -ln 2, log1p(-exp(z)) otherwise - on BOTH backends, so the fit and the predictions
# use one function. Off (the default) leaves every path and every record untouched; the
# runner records the switch in config.json and the rewritten-site count per fit record.
_STABLE_LOG1MEXP = False
LOG1MEXP_BRANCH_POINT = -math.log(2.0)


def stable_log1mexp_enabled():
    """True when this run declared the stable log1mexp evaluation. Default False."""
    return _STABLE_LOG1MEXP


def set_stable_log1mexp(enabled):
    """Turn the stable log1mexp evaluation on/off for this process; returns the previous state."""
    global _STABLE_LOG1MEXP
    previous, _STABLE_LOG1MEXP = _STABLE_LOG1MEXP, bool(enabled)
    return previous


def _is_exp_call(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "exp" and len(node.args) == 1)


def log1mexp_operand(node):
    """The z node of a `log(1 - exp(z))` or `log1p(-exp(z))` call, else None.

    Matches the parsed AST, never the text: `1` may be written 1 or 1.0, and any
    operand z (e.g. `-power(...)`) is kept as written. Nothing else is rewritten.
    """
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1):
        return None
    arg = node.args[0]
    if node.func.id == "log":
        if (isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Sub)
                and isinstance(arg.left, ast.Constant) and not isinstance(arg.left.value, bool)
                and float(arg.left.value) == 1.0 and _is_exp_call(arg.right)):
            return arg.right.args[0]
    elif node.func.id == "log1p":  # noqa: SIM102 - the nested form mirrors the `log` branch above
        if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub) and _is_exp_call(arg.operand):
            return arg.operand.args[0]
    return None


def log1mexp_sites(expression):
    """How many call sites of `expression` the stable log1mexp evaluation rewrites."""
    return sum(log1mexp_operand(node) is not None
               for node in ast.walk(ast.parse(expression, mode="eval")))


def log1mexp(z, backend=np):
    """log(1 - exp(z)) without cancellation, on the NumPy or the PyTensor backend."""
    if backend is np:
        with np.errstate(all="ignore"):  # both branches are computed; only one is kept
            near = np.log(-np.expm1(z))
            far = np.log1p(-np.exp(z))
        return np.where(z > LOG1MEXP_BRANCH_POINT, near, far)
    return backend.switch(
        z > LOG1MEXP_BRANCH_POINT, backend.log(-backend.expm1(z)), backend.log1p(-backend.exp(z))
    )


class _RolePriorTable(Mapping):
    """Read-only live view of the ACTIVE role-prior template.

    A `Mapping`, not a dict: `PRIOR_RULES[role]`, `PRIOR_RULES.get(name)`,
    `role in PRIOR_RULES` and `PRIOR_RULES.items()` are exactly the four
    operations the tree already performs on it, and every one of them resolves
    against `active_prior_template()` at call time. Read-only on purpose - the
    template is selected by `set_prior_template`, never by writing a row.
    """

    __slots__ = ()

    def __getitem__(self, role):
        return PRIOR_TEMPLATES[_ACTIVE_PRIOR_TEMPLATE][role]

    def __iter__(self):
        return iter(PRIOR_TEMPLATES[_ACTIVE_PRIOR_TEMPLATE])

    def __len__(self):
        return len(PRIOR_TEMPLATES[_ACTIVE_PRIOR_TEMPLATE])

    def __repr__(self):
        return f"{type(self).__name__}({_ACTIVE_PRIOR_TEMPLATE!r}, {dict(self)!r})"


PRIOR_RULES = _RolePriorTable()
CALL_ARITY = {"exp": 1, "log": 1, "sqrt": 1, "tanh": 1, "square": 1, "power": 2}
# The v3 grammar extension: the v3 call vocabulary. Six general element-wise functions,
# none of them named after or shaped for any benchmark law, added to - never
# replacing - the v2 six. `CALL_ARITY` itself is untouched, so the v2 gate, the
# v2 RESERVED_NAMES tuple and every record built from either keep their bytes.
V3_ADDED_CALL_ARITY = {
    "abs": 1, "sigmoid": 1, "expm1": 1, "log1p": 1, "minimum": 2, "maximum": 2,
}
V3_CALL_ARITY = {**CALL_ARITY, **V3_ADDED_CALL_ARITY}
GRAMMAR_CALL_ARITY = {"v2": CALL_ARITY, "v3": V3_CALL_ARITY}
GRAMMAR_VERSIONS = tuple(GRAMMAR_CALL_ARITY)
DEFAULT_GRAMMAR_VERSION = "v2"
# A condition of the v3 grammar extension: the DEFAULT is v2, and under v2
# nothing below is read at all - `validate_expression` takes the identical code
# path it took before v3 was added. The switch is a process-wide selection made ONCE per run by
# the runner, exactly like `set_prior_template`/`set_llm_priors` above.
_ACTIVE_GRAMMAR_VERSION = DEFAULT_GRAMMAR_VERSION
_ACTIVE_INPUT_SIGNS = {}
#: Validation-message versions: the wording of the two-cause refusal at the end of
#: `validate_expression`. `v1` is the one sentence every run has produced
#: ("unused coefficient or more than 15 mean operators"); `v2` names the cause or
#: causes (which coefficients, how many operators and what counts as one) so a
#: repair turn can act on it. The failure class does not change (both are
#: `parse_format`). A process-wide selection made ONCE per run by the runner, exactly like
#: `set_prior_template`/`set_grammar_version` above; the default is v1 and leaves every
#: existing run's bytes untouched.
VALIDATION_MESSAGE_VERSIONS = ("v1", "v2")
DEFAULT_VALIDATION_MESSAGES = "v1"
_ACTIVE_VALIDATION_MESSAGES = DEFAULT_VALIDATION_MESSAGES
MEAN_OPERATOR_LIMIT = 15
V1_VALIDATION_REFUSAL = "unused coefficient or more than 15 mean operators"


def active_validation_messages():
    """The validation-message version this process is using. `v1` unless set."""
    return _ACTIVE_VALIDATION_MESSAGES


def set_validation_messages(name):
    """Select the validation-message version for this process; returns the previous name.

    Called ONCE per run, by the runner, before any candidate is judged. A name outside
    `VALIDATION_MESSAGE_VERSIONS` raises rather than defaulting.
    """
    global _ACTIVE_VALIDATION_MESSAGES
    if name not in VALIDATION_MESSAGE_VERSIONS:
        raise ValueError(
            f"unregistered validation messages {name!r}; registered: {list(VALIDATION_MESSAGE_VERSIONS)}"
        )
    previous, _ACTIVE_VALIDATION_MESSAGES = _ACTIVE_VALIDATION_MESSAGES, name
    return previous


def validation_refusal(used, roles, operators, grammar_version=None, version=None):
    """The sentence `validate_expression` raises when a coefficient is declared and unused or
    the mean has more than MEAN_OPERATOR_LIMIT operators.

    v1: the one fixed sentence. v2: only the non-empty causes, joined by "; " - the declared
    coefficients the expression never uses, the names it uses without declaring (unreachable
    through `validate_expression`, which refuses an undeclared name earlier, but stated for a
    caller that passes its own sets), and the operator count against the limit with the
    counting rule (each binary + - * / **, each unary sign, and each grammar call counts one).
    """
    version = active_validation_messages() if version is None else version
    if version not in VALIDATION_MESSAGE_VERSIONS:
        raise ValueError(
            f"unregistered validation messages {version!r}; registered: {list(VALIDATION_MESSAGE_VERSIONS)}"
        )
    if version == "v1":
        return V1_VALIDATION_REFUSAL
    declared, used = set(roles), set(used)
    parts = []
    if declared - used:
        parts.append("coefficients declared in parameters but not used in expression: "
                     + ", ".join(sorted(declared - used)))
    if used - declared:
        parts.append("names used in expression but not declared in parameters: "
                     + ", ".join(sorted(used - declared)))
    if operators > MEAN_OPERATOR_LIMIT:
        calls = list(GRAMMAR_CALL_ARITY[grammar_version or active_grammar_version()])
        named = ", ".join(calls[:-1]) + " or " + calls[-1]
        parts.append(f"expression has {operators} mean operators; the limit is {MEAN_OPERATOR_LIMIT} "
                     f"(each + - * / ** and each unary minus count 1, and each call of {named} counts 1)")
    if not parts:
        raise ValueError("validation_refusal called without a cause")
    return "; ".join(parts)
# Numeric-rule bookkeeping. The NaN -> -inf rule in
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


# Normalization drift tolerance. A PMF row whose mass sums to 1 only to within floating point is NOT
# a defective model: scipy's beta-binomial at kappa ~ 1e5 loses a few times
# 1e-10 of the total, which the 1e-10 tolerance below reads as a failure and
# which used to end the whole run. The separation is the SIZE of the loss: pure
# drift is orders of magnitude below any defect (a model that really lost mass
# loses 1e-3 or more), so `DRIFT_TOL` sits a thousandfold above the largest
# drift ever measured (6.02e-10) and far below a defect. It is a constant, never
# an option: a run must not be able to widen the rule that decides whether its
# own predictive was believable.
NORMALIZATION_ATOL = 1e-10
DRIFT_TOL = 1e-6
NORMALIZATION_FAILURE = "likelihood normalization failed"
# Bookkeeping in the same style as `_NAN_SUBSTITUTIONS`: process-local, monotone
# and never reset, so a caller reads DIFFERENCES around the work it attributes
# (boxinggym_core's checkpoint does exactly that). `renormalized_rows` counts PMF
# rows (one particle at one input) rescaled by their own sum; `zeroed_rows`
# counts whole PARTICLE rows given zero weight, because a particle that lost
# real mass at one input is not trustworthy at the others; `max_deviation` is a
# running maximum over every row that fired either branch, so unlike the two
# counts it is not recoverable as a difference; `calls` counts the
# `particle_logpmf` calls in which at least one of the two branches fired.
_NORMALIZATION_EVENTS = {
    "renormalized_rows": 0,
    "max_deviation": 0.0,
    "zeroed_rows": 0,
    "calls": 0,
}


def normalization_event_counter():
    """Snapshot {renormalized_rows, max_deviation, zeroed_rows, calls} so far."""
    return dict(_NORMALIZATION_EVENTS)


def zeroed_particles(rows):
    """Boolean mask of the particle rows `particle_logpmf` gave zero weight.

    A zeroed particle is -inf at every input and every count. Nothing else in
    the numeric rules can produce such a row and reach a consumer: an all-NaN
    row would have been substituted to -inf and then refused as "invalid
    likelihood" by the numeric rule, which is unchanged.
    """
    rows = np.asarray(rows)
    return np.all(np.isneginf(rows), axis=tuple(range(1, rows.ndim)))


def active_particle_rows(rows):
    """`rows` without the zero-weight particles - the same array when there are none.

    Every consumer that averages over particles must average over the particles
    that still carry weight, so the divisor is the number of rows this returns.
    A model whose every particle was zeroed has no predictive at all and is
    refused with the same message the tolerance itself raises.
    """
    keep = ~zeroed_particles(rows)
    if not keep.any():
        raise ValueError(NORMALIZATION_FAILURE)
    return rows if keep.all() else rows[keep]


# --------------------------------------------------------------------------
# LLM-proposed-prior ablation - the OPTIONAL `priors` block an LLM reply may carry:
#
#   priors: {"<coefficient>": {"dist": "Normal"|"LogNormal"|"Gamma"|"Beta",
#                              "params": [a, b]}}
#
# The menu is exactly these four distributions and the hyperparameter windows
# are pre-declared (the ablation plan: "hyperparameters stay inside pre-declared
# ranges (e.g. |mu|<=5, 0.1<=sigma<=5, alpha,beta in [0.5,20])"). A proposal
# outside the menu, outside a window, or inconsistent with the coefficient's
# ROLE rejects the candidate with a per-item reason: there is deliberately NO
# fallback to the role prior, because a candidate that silently kept the
# template prior would be scored as if the model had proposed it.
PRIOR_MENU = ("Normal", "LogNormal", "Gamma", "Beta")
PRIOR_PARAMETER_NAMES = {
    "Normal": ("mu", "sigma"),
    "LogNormal": ("mu", "sigma"),
    "Gamma": ("alpha", "beta"),
    "Beta": ("alpha", "beta"),
}
PRIOR_LOCATION_ABS_MAX = 5.0
PRIOR_SCALE_RANGE = (0.1, 5.0)
PRIOR_CONCENTRATION_RANGE = (0.5, 20.0)
# Role consistency is SUPPORT CONTAINMENT, not support equality (a follow-up
# fix to the proposed-prior ablation): a proposed distribution is consistent with a role if the
# distribution's support is a SUBSET of the role's support, not only if the two
# labels happen to match. The three labels nest strictly - "unit" (0,1) is
# inside "positive" (0,inf), which is inside "real" (-inf,inf) - so a `real`
# role (`intercept`, `signed`) accepts all four menu distributions; a
# `positive` role (`positive`, `shape`) accepts LogNormal, Gamma AND Beta
# (Beta's (0,1) support sits inside (0,inf)) but still rejects Normal (the real
# line is not inside (0,inf)); a `unit` role accepts only Beta. The first
# implementation read this as label EQUALITY instead, which made `Beta`
# unreachable for any coefficient role (an open item of the proposed-prior
# ablation); the
# fix is exactly the `SUPPORT_CONTAINS` table below; nothing else here changed.
# `rho`/`pi` are the (0,1)-supported family-owned dispersion parameters, which a
# proposal may never name at all (they are filtered out of `parameters` before
# this validator sees a candidate - see `validate_proposed_priors`'s
# `DISPERSION_ROLE_NAMES` check) - they are listed so the support table stays
# complete and so `PRIOR_MENU_BY_ROLE["rho"/"pi"]` exists for a direct call,
# even though no reply-parsing path can ever reach it with role rho/pi.
SUPPORT_CONTAINS = {
    "real": ("real", "positive", "unit"),
    "positive": ("positive", "unit"),
    "unit": ("unit",),
}
ROLE_SUPPORT = {
    "intercept": "real",
    "signed": "real",
    "positive": "positive",
    "shape": "positive",
    "rho": "unit",
    "pi": "unit",
}
PRIOR_SUPPORT = {"Normal": "real", "LogNormal": "positive", "Gamma": "positive", "Beta": "unit"}
PRIOR_MENU_BY_ROLE = {
    role: tuple(name for name in PRIOR_MENU if PRIOR_SUPPORT[name] in SUPPORT_CONTAINS[support])
    for role, support in ROLE_SUPPORT.items()
}
COEFFICIENT_ROLE_NAMES = ("intercept", "signed", "positive", "shape")
# Every proposed-prior rejection - a shape failure or a per-item one - starts with this, so
# one marker finds them all in a transport record or an archive entry.
PROPOSED_PRIOR_REJECTION = "proposed prior rejected - "
DISPERSION_ROLE_NAMES = ("rho", "pi")


def prior_menu_constraints():
    """The proposed-prior menu/ranges block the belief packet publishes while `--priors llm` is on.

    It has to travel in the PACKET rather than only in the prompt because the
    v2.0 SYSTEM file is byte-frozen and the v2.1 file is a pre-declared
    alternative, not a per-run rendering; the packet is where a machine-checked
    statement of the accepted windows belongs. Off, `cell_constraints` emits
    nothing and every packet is byte-identical to the pre-ablation one.
    """
    return {
        "block": 'priors: {"<coefficient>": {"dist": ..., "params": [a, b]}}',
        "optional": True,
        "menu": list(PRIOR_MENU),
        "parameters": {name: list(PRIOR_PARAMETER_NAMES[name]) for name in PRIOR_MENU},
        "ranges": {
            "mu": [-PRIOR_LOCATION_ABS_MAX, PRIOR_LOCATION_ABS_MAX],
            "sigma": list(PRIOR_SCALE_RANGE),
            "alpha": list(PRIOR_CONCENTRATION_RANGE),
            "beta": list(PRIOR_CONCENTRATION_RANGE),
        },
        "allowed_by_role": {role: list(PRIOR_MENU_BY_ROLE[role]) for role in COEFFICIENT_ROLE_NAMES},
        "on_violation": "the whole candidate is rejected; the role prior is never substituted",
    }


def validate_proposed_prior(role, spec):
    """Validate ONE proposed prior against the menu, the windows and the role.

    Returns `(rule, reason)`: `rule` is the `(dist, a, b)` triple to use in
    place of `PRIOR_RULES[role]`, or None together with the verbatim rejection
    reason. Never returns the role prior as a fallback (a rule of the
    proposed-prior ablation).
    """
    if not isinstance(spec, dict) or set(spec) != {"dist", "params"}:
        return None, 'prior must be {"dist": <name>, "params": [a, b]}'
    name, params = spec["dist"], spec["params"]
    if name not in PRIOR_MENU:
        return None, f"distribution {name!r} is outside the menu {list(PRIOR_MENU)}"
    if not isinstance(params, (list, tuple)) or len(params) != 2:
        return None, f"{name} takes exactly two parameters {list(PRIOR_PARAMETER_NAMES[name])}"
    values = []
    for value in params:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, "prior parameters must be finite real numbers"
        if not math.isfinite(float(value)):
            return None, "prior parameters must be finite real numbers"
        values.append(float(value))
    allowed = PRIOR_MENU_BY_ROLE.get(role, ())
    if name not in allowed:
        return None, (
            f"role {role!r} has {ROLE_SUPPORT.get(role, 'unregistered')} support, so {name} is "
            f"inconsistent with it (allowed: {list(allowed)})"
        )
    first, second = values
    if name in ("Normal", "LogNormal"):
        if abs(first) > PRIOR_LOCATION_ABS_MAX:
            return None, f"|mu| must be at most {PRIOR_LOCATION_ABS_MAX:g} (got {first:g})"
        low, high = PRIOR_SCALE_RANGE
        if not low <= second <= high:
            return None, f"sigma must be in [{low:g}, {high:g}] (got {second:g})"
    else:
        low, high = PRIOR_CONCENTRATION_RANGE
        for parameter, value in zip(PRIOR_PARAMETER_NAMES[name], values):
            if not low <= value <= high:
                return None, f"{parameter} must be in [{low:g}, {high:g}] (got {value:g})"
    return (name, first, second), None


def validate_proposed_priors(proposed, roles):
    """Validate a whole `priors` block against the candidate's declared roles.

    `roles` is the candidate's `{coefficient: role}` mapping AFTER the reserved
    dispersion names have been filtered out (see `count_proposer.
    RESERVED_VARIANCE_PARAMETER_NAMES`). Returns `(accepted, reason)`:
    `accepted` is the canonical `((coefficient, (dist, a, b)), ...)` tuple sorted
    by coefficient name - the form `Hypothesis.coefficient_priors` takes, so the
    identity of a model with proposed priors does not depend on reply key order
    - and `reason` is None, or a single verbatim message that names EVERY
    offending item ("<coefficient>: <reason>", joined by "; "), so one rejection
    reports one reason per item.
    """
    if not isinstance(proposed, dict) or not proposed:
        return (), (
            PROPOSED_PRIOR_REJECTION
            + 'priors must be a non-empty object {"<coefficient>": '
            '{"dist": <name>, "params": [a, b]}}'
        )
    accepted, reasons = {}, []
    for name in sorted(proposed):
        if name in DISPERSION_ROLE_NAMES:
            reasons.append(
                f"{name}: a family-owned dispersion parameter's prior is fixed outside the proposal"
            )
            continue
        if name not in roles:
            reasons.append(f"{name}: not a declared coefficient of this candidate")
            continue
        rule, reason = validate_proposed_prior(roles[name], proposed[name])
        if reason is not None:
            reasons.append(f"{name}: {reason}")
            continue
        accepted[name] = rule
    if reasons:
        return (), PROPOSED_PRIOR_REJECTION + "; ".join(reasons)
    return tuple(sorted(accepted.items())), None


def coefficient_prior(hypothesis, name, role):
    """The prior ACTUALLY used for coefficient `name` of `hypothesis`.

    An accepted LLM-proposed prior (`Hypothesis.coefficient_priors`) REPLACES the role
    prior for that one coefficient; with no proposal - every hypothesis on the
    default path, and every hypothesis class that has no such field at all
    (`positive_inference.ScalarHypothesis`, `real_inference.RealHypothesis`) -
    this is exactly `PRIOR_RULES[role]`, evaluated the same way and in the same
    place as before.
    """
    for proposed, rule in getattr(hypothesis, "coefficient_priors", ()) or ():
        if proposed == name:
            return rule
    return PRIOR_RULES[role]


def validate_coefficient_priors(coefficient_priors, roles):
    """Guard `Hypothesis.coefficient_priors`: canonical order, declared names, valid rules.

    Raises `ValueError` like every other `Hypothesis.__post_init__` check, so an
    invalid or non-canonical proposed prior can never reach a fit, a record or
    an identity. Empty - the default path - does nothing at all.
    """
    if not coefficient_priors:
        return
    names = [name for name, _ in coefficient_priors]
    if len(set(names)) != len(names):
        raise ValueError("at most one proposed prior per coefficient")
    if names != sorted(names):
        raise ValueError("proposed priors must be sorted by coefficient name")
    for name, rule in coefficient_priors:
        if name not in roles:
            raise ValueError(f"proposed prior for undeclared coefficient {name}")
        if not isinstance(rule, tuple) or len(rule) != 3:
            raise ValueError("a proposed prior is a (distribution, a, b) triple")
        _, reason = validate_proposed_prior(
            roles[name], {"dist": rule[0], "params": [rule[1], rule[2]]}
        )
        if reason is not None:
            raise ValueError(f"{name}: {reason}")


def dispersion_prior(name):
    """The family-owned dispersion prior (`rho`, `pi`) of the ACTIVE template.

    The paper's prior table fixes both as Beta rows and every registered template
    keeps them Beta, which is what `count_inference.build_model`'s `pm.Beta`
    call needs. A template that changed the distribution would be a different
    likelihood, not a prior-scale ablation, so it is refused here rather than
    silently ignored. Returns the two Beta hyperparameters in order.
    """
    family, a, b = PRIOR_RULES[name]
    if family != "Beta":
        raise ValueError(f"the {name} dispersion prior must stay a Beta row (got {family})")
    return a, b


def prior_identity_component(coefficient_priors):
    """The `model_id` component of a set of proposed priors.

    Appended to the canonical `[family, <AST>]` pair ONLY when the set is
    non-empty, so every model_id of a hypothesis WITHOUT proposed priors hashes
    byte-identical input to the pre-ablation identity, and two candidates that differ only in a
    proposed prior are different models (a rule of the proposed-prior ablation).
    """
    return [[name, [rule[0], rule[1], rule[2]]] for name, rule in coefficient_priors]


# Names a candidate may never use for a coefficient, on top of the interface's
# own input names: the family-owned dispersion parameters and the call names.
RESERVED_NAMES = ("rho", "pi", *CALL_ARITY)
# Under the v3 grammar the six added call names are reserved for the same
# reason the v2 six are - a coefficient named after a call the gate dispatches
# would shadow it. The v2 tuple is NOT widened, so a v2 run keeps every byte.
V3_RESERVED_NAMES = ("rho", "pi", *V3_CALL_ARITY)
GRAMMAR_RESERVED_NAMES = {"v2": RESERVED_NAMES, "v3": V3_RESERVED_NAMES}
# The most distinct mean coefficients one candidate may declare - the
# cap `validate_expression` enforces - named once so the belief packet publishes
# this number (`constraints["max_distinct_mean_coefficients"]`) instead of a copy.
# The refusal keeps its literal text "require 1..6 distinct mean coefficients";
# test_packet_constraints_explicit binds that text to this constant.
MAX_DISTINCT_MEAN_COEFFICIENTS = 6

# --------------------------------------------------------------------------
# The paper's task table - "Data type -> initial family -> admissible
# revisions (F)",
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
# binary: the same count runner with one trial. unbounded_count/
# positive: positive_inference (PyMC Poisson/NegBin/ZIP/LogNormal/Gamma). real:
# real_inference (PyMC Normal/StudentT(nu=4)/heteroscedastic Normal).
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
# defaults that reproduce the original hard-coded `u`/`z` behaviour byte for byte.
CORE_INPUTS = ({"name": "u", "range": (0.0, 1.0)},)
CORE_DERIVED_INPUTS = {"z": "2*u-1"}
DEFAULT_INPUT_NAMES = ("u", "z")

# --------------------------------------------------------------------------
# CATEGORICAL INPUTS AS PER-LEVEL OFFSET COEFFICIENTS. An
# `interface["inputs"]` item may declare
#
#     {"name": "student", "range": [0, 5], "kind": "categorical",
#      "levels": [0, 1, 2, 3, 4, 5]}
#
# `kind` is absent (or "numeric") on every input every earlier task declares,
# so CORE and the earlier BoxingGym tasks are byte-identical: nothing below
# runs for them. A categorical input's design column holds the INDEX of the
# level inside `levels` (its code), which is why the declared range must be
# exactly [0, n_levels - 1]; a column value that is not one of those codes is
# an UNDECLARED LEVEL and raises.
#
# The grammar addition is ONE form: an INDEXED COEFFICIENT `a[student]`, where
# `a` is a coefficient the candidate declares in `parameters` with role
# "signed" and `student` is a categorical input name. It is a length-n_levels
# vector coefficient whose every entry carries the SAME fixed prior -
# the paper's prior table, row 1, "| standardised slope | N(0,1) |" - shared
# across levels; no new prior is introduced and no existing prior changes.
#
# COEFFICIENT-CAP ACCOUNTING (stated in the packet's `constraints`, and here):
# one indexed coefficient occupies exactly ONE of the at-most-six coefficient
# slots `validate_expression` allows, even though it contributes n_levels
# parameters to the fit. The cap counts DECLARED COEFFICIENTS (the keys of
# `parameters`), which is what a proposal writes, not fitted dimensions.
#
# A categorical input may appear ONLY as a coefficient index: its code is a
# label, not a number, so `b1*student` is rejected. A coefficient may not be
# used both indexed and bare, and an
# indexed coefficient may not be re-indexed by a SECOND, different categorical
# input in the same expression - `a[student]+a[student]` stays valid,
# `a[student]+a[question]` is rejected. All three are per-item rejections
# with the verbatim reason strings below.
# --------------------------------------------------------------------------
CATEGORICAL_KIND = "categorical"
NUMERIC_KIND = "numeric"
INPUT_KINDS = (NUMERIC_KIND, CATEGORICAL_KIND)
# The single role an indexed coefficient may carry (Table 3 "standardised
# slope"), so the per-level offsets share one N(0,1) prior.
INDEXED_COEFFICIENT_ROLE = "signed"
# The machine-readable statement of the cap rule, copied into the belief
# packet's `constraints` (never into the SYSTEM prompt, whose bytes are fixed).
INDEXED_COEFFICIENT_RULE = (
    "an indexed coefficient a[<categorical input>] is one length-n_levels vector "
    "coefficient: its per-level offsets share the single fixed role prior "
    "signed=Normal(0,1) (the paper's prior table 'standardised slope'), and it occupies "
    "ONE of the at most six declared coefficient slots even though it contributes "
    "n_levels parameters to the fit; a categorical input may appear ONLY as a "
    "coefficient index, never as a bare term, and a coefficient may not be used "
    "both indexed and bare"
)


def standardized_expression(name, low, high):
    """The engine's standardization x~ = 2(x-lo)/(hi-lo) - 1 as a derived input.

    For CORE (`name="u"`, range [0,1]) this is `2*(u-0)/(1-0)-1`, which evaluates
    to exactly the same floats as the declared `2*u-1` (subtracting 0.0 and
    dividing by 1.0 are exact in IEEE-754); `test_core_standardization_matches`
    checks the two agree bit for bit. CORE keeps the short literal so that its
    packets stay byte-identical to the original ones.
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
    names, categorical = [], []
    for item in inputs:
        name = _validate_name(item["name"])
        low, high = (float(v) for v in item["range"])
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError(f"input {name} needs a finite range with hi > lo")
        if name in names:
            raise ValueError(f"duplicate input name {name}")
        kind = item.get("kind", NUMERIC_KIND)
        if kind not in INPUT_KINDS:
            raise ValueError(f"input {name} declares an unknown kind {kind!r}")
        if kind == CATEGORICAL_KIND:
            levels = item.get("levels")
            if not isinstance(levels, (list, tuple)) or len(levels) < 2:
                raise ValueError(f"categorical input {name} needs at least two levels")
            if len(set(map(repr, levels))) != len(levels):
                raise ValueError(f"categorical input {name} has duplicate levels")
            if (low, high) != (0.0, float(len(levels) - 1)):
                raise ValueError(
                    f"categorical input {name} must declare range [0, n_levels-1]"
                )
            categorical.append(name)
        elif "levels" in item:
            raise ValueError(f"numeric input {name} must not declare levels")
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
            # A level code is a LABEL, not a number: a derived input may not
            # arithmetically transform one.
            if isinstance(node, ast.Name) and node.id in categorical:
                raise ValueError(
                    f"categorical input {node.id} cannot enter derived input {name}"
                )
        names.append(name)
    return tuple(names)


def interface_categorical_levels(interface):
    """`((name, n_levels), ...)` for the categorical inputs, in declared order.

    Empty for CORE and for every input block that declares no `kind`, which is
    what makes the categorical-input grammar a strict extension: an empty tuple reaches
    `validate_expression`/`Hypothesis` exactly as the earlier default does.
    """
    interface_input_names(interface)
    return tuple(
        (item["name"], len(item["levels"]))
        for item in interface["inputs"]
        if item.get("kind") == CATEGORICAL_KIND
    )


def input_index_codes(interface, designs):
    """`{name: int codes}` for the categorical columns of `designs`.

    The design column of a categorical input holds the level's INDEX inside the
    declared `levels` list. A value that is not one of those codes - a
    non-integer, a negative, or one at or past `n_levels` - is an UNDECLARED
    LEVEL and raises with that verbatim phrase, which the per-item judge and
    the prior probe both turn into a rejection reason rather than a crash.
    """
    declared = interface_categorical_levels(interface)
    if not declared:
        return {}
    base = [item["name"] for item in interface["inputs"]]
    if isinstance(designs, dict):
        columns = {name: np.asarray(designs[name], dtype=float) for name in base}
    elif len(base) == 1:
        columns = {base[0]: np.asarray(designs, dtype=float)}
    else:
        matrix = np.asarray(designs, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(base):
            raise ValueError("multi-input designs must be an [observation, input] matrix")
        columns = {name: matrix[:, index] for index, name in enumerate(base)}
    codes = {}
    for name, levels in declared:
        column = np.atleast_1d(columns[name])
        rounded = np.rint(column)
        if (
            not np.all(np.isfinite(column))
            or np.any(column != rounded)
            or np.any(rounded < 0)
            or np.any(rounded > levels - 1)
        ):
            raise ValueError(f"undeclared level for categorical input {name}")
        codes[name] = rounded.astype(int)
    return codes


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


# --------------------------------------------------------------------------
# The v3 domain proof.
#
# An audit of the v2 gate showed that the gate never constrained WHERE a coefficient
# may stand - a coefficient in a denominator, inside a call or as an exponent
# passes at every version - and that what is unproved is the DOMAIN of those
# positions. A denominator that can reach zero, a `log` of a non-positive
# argument and a non-positive base under a fractional power are not refused at
# the gate today; they end a fit on the non-finite rule instead. v3 proves them
# at candidate time or refuses the candidate.
#
# The proof is a sign analysis over the three-element sign lattice, carried out
# on the declared input RANGES and the coefficient ROLES. Each node is assigned
# the set of signs its value can take; a node whose set cannot be narrowed
# enough is refused rather than admitted, so the analysis is sound in the only
# direction that matters - it never admits a node it has not proved. It is
# deliberately not a full interval arithmetic: a bound that happens to hold for
# numeric reasons the lattice cannot see (`2-tanh(x)` is always positive) is
# refused as unprovable, which costs expressiveness and never costs safety.
# --------------------------------------------------------------------------
SIGN_NEGATIVE, SIGN_ZERO, SIGN_POSITIVE = 1, 2, 4
SIGN_ANY = SIGN_NEGATIVE | SIGN_ZERO | SIGN_POSITIVE
SIGN_NONNEGATIVE = SIGN_ZERO | SIGN_POSITIVE
_SIGN_UNITS = (SIGN_NEGATIVE, SIGN_ZERO, SIGN_POSITIVE)
_SIGN_ADD = {
    (SIGN_NEGATIVE, SIGN_NEGATIVE): SIGN_NEGATIVE,
    (SIGN_NEGATIVE, SIGN_ZERO): SIGN_NEGATIVE,
    (SIGN_NEGATIVE, SIGN_POSITIVE): SIGN_ANY,
    (SIGN_ZERO, SIGN_NEGATIVE): SIGN_NEGATIVE,
    (SIGN_ZERO, SIGN_ZERO): SIGN_ZERO,
    (SIGN_ZERO, SIGN_POSITIVE): SIGN_POSITIVE,
    (SIGN_POSITIVE, SIGN_NEGATIVE): SIGN_ANY,
    (SIGN_POSITIVE, SIGN_ZERO): SIGN_POSITIVE,
    (SIGN_POSITIVE, SIGN_POSITIVE): SIGN_POSITIVE,
}
_SIGN_MULTIPLY = {
    (SIGN_NEGATIVE, SIGN_NEGATIVE): SIGN_POSITIVE,
    (SIGN_NEGATIVE, SIGN_ZERO): SIGN_ZERO,
    (SIGN_NEGATIVE, SIGN_POSITIVE): SIGN_NEGATIVE,
    (SIGN_ZERO, SIGN_NEGATIVE): SIGN_ZERO,
    (SIGN_ZERO, SIGN_ZERO): SIGN_ZERO,
    (SIGN_ZERO, SIGN_POSITIVE): SIGN_ZERO,
    (SIGN_POSITIVE, SIGN_NEGATIVE): SIGN_NEGATIVE,
    (SIGN_POSITIVE, SIGN_ZERO): SIGN_ZERO,
    (SIGN_POSITIVE, SIGN_POSITIVE): SIGN_POSITIVE,
}
# The two roles whose fixed prior has support on the strictly positive line
# (`PRIOR_TEMPLATES`: positive = LogNormal, shape = Gamma). intercept and signed
# are Normal and therefore unsigned, whatever template is active.
STRICTLY_POSITIVE_ROLES = ("positive", "shape")
DOMAIN_UNPROVABLE_RULE = "domain_unprovable"
CONSTANT_DEGENERATE_RULE = "constant_degenerate_subexpression"
_PROVABLE_BY = (
    "give the subexpression a coefficient of role positive or shape, or declare "
    "an input range that excludes the forbidden values, to make it provable"
)


def _sign_of_range(low, high):
    """The sign set of a closed interval, from its declared endpoints."""
    signs = 0
    if low < 0:
        signs |= SIGN_NEGATIVE
    if low <= 0 <= high:
        signs |= SIGN_ZERO
    if high > 0:
        signs |= SIGN_POSITIVE
    return signs or SIGN_ZERO


def _sign_combine(table, left, right):
    signs = 0
    for one in _SIGN_UNITS:
        if not left & one:
            continue
        for other in _SIGN_UNITS:
            if right & other:
                signs |= table[(one, other)]
    return signs


def _sign_negate(signs):
    flipped = signs & SIGN_ZERO
    if signs & SIGN_NEGATIVE:
        flipped |= SIGN_POSITIVE
    if signs & SIGN_POSITIVE:
        flipped |= SIGN_NEGATIVE
    return flipped


def interface_input_signs(interface):
    """The sign class of every name in an interface's grammar namespace.

    Base inputs get theirs from the declared range. A derived input gets its own
    by running the same analysis over its expression; a derived expression the
    analysis cannot clear (a division whose divisor it cannot prove nonzero, a
    `log` it cannot prove positive) contributes SIGN_ANY rather than raising,
    because a derived input is the interface's own declaration and is not a
    candidate this gate may refuse.
    """
    names = interface_input_names(interface)
    signs = {}
    for item in interface["inputs"]:
        low, high = (float(value) for value in item["range"])
        signs[item["name"]] = _sign_of_range(low, high)
    derived = interface.get("derived_inputs") or {}
    for name in names[len(interface["inputs"]) :]:
        try:
            signs[name] = _prove_domain(
                ast.parse(derived[name], mode="eval"), {}, signs
            )[0]
        except ValueError:
            signs[name] = SIGN_ANY
    return signs


def _prove_domain(tree, roles, input_signs):
    """Prove every division, log, root and power in `tree` safe; return its sign.

    Returns `(signs, uses_input)` for the whole tree and raises ValueError with
    a rule name - `domain_unprovable` or `constant_degenerate_subexpression` -
    on the first node it cannot clear. `roles` maps coefficient name to role,
    `input_signs` maps input name to a sign set (absent means SIGN_ANY).
    """

    def unprovable(what, requirement):
        """Raise the `domain_unprovable` refusal for `what`, naming the missing `requirement`."""
        raise ValueError(
            f"{DOMAIN_UNPROVABLE_RULE}: {what} requires {requirement} over the "
            f"declared input ranges and coefficient roles; {_PROVABLE_BY}"
        )

    def constant_exponent(node):
        """The literal value of an exponent node, or None when it is not one."""
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            inner = constant_exponent(node.operand)
            if inner is None:
                return None
            return -inner if isinstance(node.op, ast.USub) else inner
        return None

    def power(base, exponent_node, what):
        """`(signs, uses_input)` of a power from its base pair; refuses an unprovable base."""
        base_signs, base_uses = base
        exponent = constant_exponent(exponent_node)
        if exponent is None or exponent != int(exponent):
            # A non-integer or non-literal exponent is only defined on a
            # strictly positive base, whatever the exponent turns out to be.
            if base_signs != SIGN_POSITIVE:
                unprovable(what, "a strictly positive base")
            return SIGN_POSITIVE, base_uses
        if exponent < 0 and base_signs & SIGN_ZERO:
            unprovable(what, "a base proved nonzero")
        if int(exponent) % 2 == 0:
            return (SIGN_NONNEGATIVE if base_signs & SIGN_ZERO else SIGN_POSITIVE), base_uses
        return base_signs, base_uses

    def walk(node):
        """`(signs, uses_input)` of `node`; raises on the first node it cannot prove safe."""
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Name):
            if node.id in roles:
                positive = roles[node.id] in STRICTLY_POSITIVE_ROLES
                return (SIGN_POSITIVE if positive else SIGN_ANY), False
            return input_signs.get(node.id, SIGN_ANY), True
        if isinstance(node, ast.Subscript):
            # An indexed coefficient carries INDEXED_COEFFICIENT_ROLE, which is
            # unsigned, and its index is an input: the value is unsigned and the
            # node does depend on an input.
            return SIGN_ANY, True
        if isinstance(node, ast.Constant):
            return _sign_of_range(float(node.value), float(node.value)), False
        if isinstance(node, ast.UnaryOp):
            signs, uses = walk(node.operand)
            return (_sign_negate(signs) if isinstance(node.op, ast.USub) else signs), uses
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            uses = left[1] or right[1]
            if not uses:
                raise ValueError(
                    f"{CONSTANT_DEGENERATE_RULE}: a subexpression built from "
                    "coefficients and literals alone reduces to a single coefficient"
                )
            if isinstance(node.op, ast.Add):
                return _sign_combine(_SIGN_ADD, left[0], right[0]), uses
            if isinstance(node.op, ast.Sub):
                return _sign_combine(_SIGN_ADD, left[0], _sign_negate(right[0])), uses
            if isinstance(node.op, ast.Mult):
                return _sign_combine(_SIGN_MULTIPLY, left[0], right[0]), uses
            if isinstance(node.op, ast.Div):
                if right[0] & SIGN_ZERO:
                    unprovable("a division", "a denominator proved nonzero")
                return _sign_combine(_SIGN_MULTIPLY, left[0], right[0]), uses
            return power(left, node.right, "a power")[0], uses
        signs = [walk(argument) for argument in node.args]
        uses = any(argument[1] for argument in signs)
        if not uses:
            raise ValueError(
                f"{CONSTANT_DEGENERATE_RULE}: a subexpression built from "
                "coefficients and literals alone reduces to a single coefficient"
            )
        name = node.func.id
        if name == "log":
            if signs[0][0] != SIGN_POSITIVE:
                unprovable("a logarithm", "a strictly positive argument")
            return SIGN_ANY, uses
        if name == "log1p":
            shifted = _sign_combine(_SIGN_ADD, signs[0][0], SIGN_POSITIVE)
            if shifted != SIGN_POSITIVE:
                unprovable("a log1p", "an argument proved greater than minus one")
            return SIGN_ANY, uses
        if name == "sqrt":
            if signs[0][0] & SIGN_NEGATIVE:
                unprovable("a square root", "a nonnegative argument")
            return signs[0][0], uses
        if name == "power":
            return power(signs[0], node.args[1], "a power call")[0], uses
        if name in ("exp", "sigmoid"):
            return SIGN_POSITIVE, uses
        if name in ("square", "abs"):
            return (SIGN_NONNEGATIVE if signs[0][0] & SIGN_ZERO else SIGN_POSITIVE), uses
        if name in ("minimum", "maximum"):
            return signs[0][0] | signs[1][0], uses
        # The calls left here (tanh, expm1) are sign-preserving and total.
        return signs[0][0], uses

    return walk(tree)


def validate_expression(
    expression, parameters, input_names=DEFAULT_INPUT_NAMES, categorical_inputs=(),
    grammar_version=None, input_signs=None,
):
    """The ONE expression grammar, shared by every runner (count, binary, real).

    Accepts exactly: the interface's input names (base + derived), the
    candidate's own declared coefficients, finite real literals, + - * / **,
    unary +/-, and the six calls in CALL_ARITY. At most 6 distinct coefficients,
    at most one intercept, roles from the fixed four, at most 15 mean operators.
    An identifier outside that namespace raises "unknown input name <id>" -
    which the per-item judge turns into a candidate rejection with that verbatim
    reason (earlier versions said "unknown expression name").

    `categorical_inputs` is `((name, n_levels), ...)` for the
    interface's categorical inputs - `interface_categorical_levels`. It DEFAULTS
    TO EMPTY, and with it empty this function is byte-for-byte the earlier one: a
    subscript still falls through to "unsupported expression syntax". With it
    non-empty, `a[<categorical input>]` is the one added form (see
    `INDEXED_COEFFICIENT_RULE`); the coefficient cap still counts DECLARED
    coefficients, so one indexed coefficient is one slot of the six.

    One coefficient may be indexed by
    at most one categorical input across the WHOLE expression - repeating the
    SAME index (`a[student]+a[student]`) stays valid, but indexing it again
    under a DIFFERENT input (`a[student]+a[question]`, `a[student]-a[question]`)
    is rejected here, at candidate-validation time, with "coefficient a is
    indexed by two different inputs" - the same reason the deferred
    `indexed_parameters` check below has always raised, just no longer only
    reachable after a `Hypothesis` was already constructed and handed back as
    valid (`judge_candidate`/`parse_candidate_object` previously returned
    (True, None)/a `Hypothesis` for such a candidate, which then raised inside
    `as_dict()`, past the proposer's format-repair path).

    The v3 grammar options (`grammar_version`, `input_signs`): both default to the process-wide
    selection (`active_grammar_version`/`active_input_signs`), which is v2 and an
    empty namespace unless a runner called `set_grammar_version`. Under v2 this
    function is the one it has always been, line for line. Under v3 two things
    change and nothing else: the call table is `V3_CALL_ARITY` (six added
    element-wise functions) with `V3_RESERVED_NAMES` guarding their names, and a
    DOMAIN PROOF runs after every v2 rule has passed - each division, logarithm,
    root and power must be shown safe over the declared input ranges and the
    coefficient roles, and a subexpression that reduces to a coefficient-only
    value is refused. The two v3 rule names are `DOMAIN_UNPROVABLE_RULE` and
    `CONSTANT_DEGENERATE_RULE`; both are prefixes of the refusal message, so the
    per-item judge hands them back verbatim like every other reason.
    """
    version = active_grammar_version() if grammar_version is None else grammar_version
    if version not in GRAMMAR_CALL_ARITY:
        raise ValueError(
            f"unregistered grammar version {version!r}; registered: {list(GRAMMAR_VERSIONS)}"
        )
    calls = GRAMMAR_CALL_ARITY[version]
    reserved = GRAMMAR_RESERVED_NAMES[version]
    inputs = tuple(input_names)
    if not inputs or len(set(inputs)) != len(inputs):
        raise ValueError("interface must declare a non-empty set of distinct input names")
    categorical = dict(categorical_inputs)
    if any(name not in inputs for name in categorical):
        raise ValueError("a categorical input must be part of the interface namespace")
    roles = dict(parameters)
    if not 1 <= len(roles) <= MAX_DISTINCT_MEAN_COEFFICIENTS or len(roles) != len(parameters):
        raise ValueError("require 1..6 distinct mean coefficients")
    if any(role not in ("intercept", "signed", "positive", "shape") for role in roles.values()):
        raise ValueError("coefficient role outside fixed prior policy")
    if sum(role == "intercept" for role in roles.values()) > 1:
        raise ValueError("at most one intercept")
    if any(not name.isidentifier() or name in {*inputs, *reserved} for name in roles):
        raise ValueError("invalid or reserved coefficient name")
    tree = ast.parse(expression, mode="eval")
    used, operators = set(), 0
    indexed, bare = set(), set()
    # Which categorical input each indexed
    # coefficient has been bound to SO FAR, recorded as the AST is walked -
    # not deferred to `indexed_parameters`/`Hypothesis.as_dict()`. Without
    # this, `a[student]-a[question]` validated (both subscripts pass every
    # per-occurrence check below and `indexed` only ever records the
    # coefficient NAME, never its input), so a `Hypothesis` for it could be
    # constructed - and reached `judge_candidate`/`parse_candidate_object`'s
    # (True, None)/success return - only to raise deep inside `as_dict()`
    # (via the module-level `indexed_parameters`, kept below as a safety
    # net), outside the proposer's format-repair path.
    indexed_input = {}

    def validate(node):
        """Check one node against the grammar; record used coefficients, count operators."""
        nonlocal operators
        if isinstance(node, ast.Expression):
            validate(node.body)
        elif isinstance(node, ast.Name):
            if node.id not in {*inputs, *roles}:
                raise ValueError(f"unknown input name {node.id}")
            if node.id in categorical:
                raise ValueError(
                    f"categorical input {node.id} may only index a coefficient"
                )
            if node.id in roles:
                used.add(node.id)
                bare.add(node.id)
        elif isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name) or node.value.id not in roles:
                raise ValueError("an indexed coefficient must be a declared coefficient")
            if roles[node.value.id] != INDEXED_COEFFICIENT_ROLE:
                raise ValueError(
                    f"an indexed coefficient must have role {INDEXED_COEFFICIENT_ROLE}"
                )
            index = node.slice
            if not isinstance(index, ast.Name):
                raise TypeError("a coefficient index must be a categorical input name")
            if index.id not in inputs:
                raise ValueError(f"unknown input name {index.id}")
            if index.id not in categorical:
                raise ValueError(f"non-categorical index {index.id}")
            name = node.value.id
            if name in indexed_input and indexed_input[name] != index.id:
                raise ValueError(f"coefficient {name} is indexed by two different inputs")
            indexed_input[name] = index.id
            used.add(name)
            indexed.add(name)
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
            if node.func.id not in calls or node.keywords:
                raise ValueError("call outside expression grammar")
            if len(node.args) != calls[node.func.id]:
                raise ValueError("wrong expression call arity")
            operators += 1
            for arg in node.args:
                validate(arg)
        else:
            raise TypeError(f"unsupported expression syntax: {ast.dump(node)}")

    validate(tree)
    if used != set(roles) or operators > MEAN_OPERATOR_LIMIT:
        # v1 is V1_VALIDATION_REFUSAL, byte for byte; v2 names the cause(s).
        raise ValueError(validation_refusal(used, roles, operators, version))
    if version != DEFAULT_GRAMMAR_VERSION:
        # The domain proof runs only under v3, and only after
        # every v2 structural rule has passed, so a v3 refusal is always about
        # the domain or the call vocabulary and never re-words a v2 message.
        _prove_domain(
            tree,
            roles,
            active_input_signs() if input_signs is None else dict(input_signs),
        )
    if indexed & bare:
        raise ValueError(
            f"coefficient {min(indexed & bare)} is used both indexed and unindexed"
        )


def canonical_model_id(expression, family, parameters, coefficient_priors=()):
    """Canonical model identity (schema `ast_roles_family_ac_v2`): power notation, local
    Add/Mult order, then role-preserving alpha-renaming.

    Power notation (identity schema v2): `square(x)`, `power(x, k)` and `x ** k` are one
    node `POW(x, k)` when `k` is a literal number (not a bool) that is finite,
    integer-valued and >= 0; `k` is stored as an int, so `x**2 == x**2.0`. The
    base may be any subexpression. Everything else keeps its own node: `x*x`
    and `x*x*x`, a non-integer (`x**2.5`, `power(x, 0.5)`), negative
    (`x**-2` is a unary minus, not a literal) or non-literal (`power(2, x)`,
    `x**a`) exponent, `sqrt(x)`, Sub, Div. Nesting is kept as written
    (`power(power(x, 2), 3)` is not `x**6`): no exponent arithmetic, no
    associativity, distribution or other algebra, and no other call rewrite.
    Input/index names and proposed priors remain bound to their AST positions.
    The expression text itself is never rewritten.

    The id is the first 20 hex digits of the sha256 of `model_id_preimage`.
    """
    return _canonical_model_id(expression, family, tuple(parameters), tuple(coefficient_priors))


def model_id_preimage(expression, family, parameters, coefficient_priors=()):
    """The canonical JSON identity text `canonical_model_id` hashes (identity schema v2).

    `canonical_model_id(...) == sha256(model_id_preimage(...).encode()).hexdigest()[:20]`;
    both read the one `_canonical_identity`, so they cannot disagree.
    """
    return _canonical_identity(expression, family, tuple(parameters), tuple(coefficient_priors))


@lru_cache(maxsize=4096)
def _canonical_model_id(expression, family, parameters, coefficient_priors):
    preimage = _canonical_identity(expression, family, parameters, coefficient_priors)
    return hashlib.sha256(preimage.encode()).hexdigest()[:20]


def _literal_exponent(node):
    """`int(k)` for a literal exponent `k` identity schema v2 normalizes, else None.

    `k` must be an `ast.Constant` whose value is an int or float (a bool is
    not), finite, integer-valued and >= 0. `-2` is `UnaryOp(USub, 2)`, so it
    never reaches this test as a literal.
    """
    if not isinstance(node, ast.Constant):
        return None
    value = node.value
    if type(value) is int:
        exponent = value
    elif type(value) is float and math.isfinite(value) and value.is_integer():
        exponent = int(value)
    else:
        return None
    return exponent if exponent >= 0 else None


@lru_cache(maxsize=4096)
def _canonical_identity(expression, family, parameters, coefficient_priors):
    roles, priors = dict(parameters), dict(coefficient_priors)

    def pow_node(base, exponent):
        """The canonical `POW` node for `base` raised to a literal integer `exponent`."""
        return ("POW", symbolic(base), ("exponent", exponent))

    def symbolic(node):
        """Translate one AST node into the nested-tuple canonical form."""
        if isinstance(node, ast.Expression):
            return symbolic(node.body)
        if isinstance(node, ast.Name):
            if node.id in roles:
                return ("parameter", node.id, roles[node.id], priors.get(node.id, ()))
            return ("input", node.id)
        if isinstance(node, ast.Subscript):
            return ("index", symbolic(node.value), ("input", node.slice.id))
        if isinstance(node, ast.Constant):
            return ("number", float(node.value))
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow) and (k := _literal_exponent(node.right)) is not None:
                return pow_node(node.left, k)
            return (type(node.op).__name__, symbolic(node.left), symbolic(node.right))
        if isinstance(node, ast.UnaryOp):
            return (type(node.op).__name__, symbolic(node.operand))
        if node.func.id == "square" and len(node.args) == 1:
            return pow_node(node.args[0], 2)
        if (node.func.id == "power" and len(node.args) == 2
                and (k := _literal_exponent(node.args[1])) is not None):
            return pow_node(node.args[0], k)
        return (node.func.id, *(symbolic(arg) for arg in node.args))

    def key(node):
        """Sort key of a canonical node; coefficients compare by role and prior, never by name."""
        kind = node[0]
        if kind == "parameter":
            return (0, node[2], node[3])
        if kind == "input":
            return (1, node[1])
        if kind == "number":
            return (2, node[1])
        if kind == "exponent":
            return (4, node[1])
        children = tuple(key(child) for child in node[1:])
        if kind in ("Add", "Mult"):
            children = tuple(sorted(children))
        return (3, kind, children)

    def orientations(node):
        """Every child order of the Add/Mult nodes that is consistent with their sort keys."""
        if node[0] in ("parameter", "input", "number", "exponent"):
            return (node,)
        variants = set()
        for children in product(*(orientations(child) for child in node[1:])):
            if node[0] in ("Add", "Mult"):
                left, right = children
                lk, rk = key(left), key(right)
                if lk <= rk:
                    variants.add((node[0], left, right))
                if rk <= lk:
                    variants.add((node[0], right, left))
            else:
                variants.add((node[0], *children))
        return variants

    def encode(tree):
        """Compact JSON bytes of one orientation after alpha-renaming the coefficients."""
        aliases = {}

        def alpha(node):
            """Replace each coefficient name by its order of first appearance."""
            if node[0] == "parameter":
                aliases.setdefault(node[1], len(aliases))
                return ["parameter", aliases[node[1]], node[2]]
            if node[0] in ("input", "number", "exponent"):
                return list(node)
            return [node[0], *(alpha(child) for child in node[1:])]

        identity = [family, alpha(tree)]
        if priors:
            identity.append(sorted([aliases[name], list(rule)] for name, rule in priors.items()))
        return json.dumps(identity, separators=(",", ":")).encode()

    # Equal role/structure keys can share coefficients across subtrees. Compare
    # tied orientations after alpha-renaming, never break ties by parameter name.
    # json.dumps escapes to ASCII, so the text round-trips to the same bytes.
    raw = min(encode(tree) for tree in orientations(symbolic(ast.parse(expression, mode="eval"))))
    return raw.decode()


@dataclass(frozen=True)
class Hypothesis:
    """One candidate: an expression over NAMED inputs, a family, and coefficient roles.

    `input_names` is the interface namespace the expression is validated against
    (base inputs plus derived inputs, from `interface_input_names`). It defaults
    to CORE's `("u", "z")`, so every construction from before named inputs keeps its exact meaning.
    It is deliberately NOT part of `model_id`: the canonical AST already carries
    every input the expression actually references, verbatim, as `["input",
    name]` leaves, so `b0+b1*z` and `b0+b1*tz` already hash differently.
    The canonical identity preserves that distinction while versioning local Add/Mult order.
    The anchors recorded by earlier versions remain predecessor fixtures, not current identity claims.

    `family` is checked against `IMPLEMENTED_FAMILIES`, not the
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

    `categorical_levels` is `((name, n_levels), ...)` for the
    interface's categorical inputs, from `interface_categorical_levels`. It
    defaults to EMPTY, so every construction without categorical inputs is unchanged and a
    subscript in the expression is still a syntax rejection. Non-empty, it is
    what lets `a[student]` validate, what fixes the length of that coefficient's
    per-level offset vector (`parameter_shapes`), and it is deliberately NOT
    part of `model_id`: the canonical AST already carries the indexed
    coefficient and the input it indexes.
    """

    expression: str
    family: str
    parameters: tuple[tuple[str, str], ...]
    input_names: tuple[str, ...] = DEFAULT_INPUT_NAMES
    categorical_levels: tuple[tuple[str, int], ...] = ()
    # LLM-proposed-prior ablation: the accepted LLM-PROPOSED priors of
    # this candidate, as the canonical `((coefficient, (dist, a, b)), ...)` tuple
    # sorted by coefficient name (`validate_proposed_priors`). EMPTY on the
    # default path: no prior component is appended to the versioned identity,
    # `as_dict` emits no prior keys, and `coefficient_prior` returns
    # `PRIOR_RULES[role]` for every coefficient.
    # Non-empty, each entry REPLACES the role prior of that one coefficient and
    # is part of the identity: a different prior is a different model.
    coefficient_priors: tuple[tuple[str, tuple], ...] = ()

    def __post_init__(self):
        if self.family not in IMPLEMENTED_FAMILIES:
            raise ValueError(f"unsupported count family: {self.family}")
        validate_expression(
            self.expression, self.parameters, self.input_names, self.categorical_levels
        )
        validate_coefficient_priors(self.coefficient_priors, dict(self.parameters))

    @property
    def noise_parameters(self):
        """The family-owned dispersion parameter names (empty for binomial and bernoulli)."""
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
    def indexed_parameters(self):
        """`((coefficient, categorical input, n_levels), ...)`, sorted by name.

        Empty for every expression without a subscript, which is every
        expression a task without categorical inputs can write.
        """
        return indexed_parameters(self.expression, self.categorical_levels)

    @property
    def parameter_shapes(self):
        """`((coefficient, n_levels), ...)` for the indexed coefficients only.

        A coefficient absent from this mapping is a scalar, exactly as before;
        `count_inference.build_model` and `boxinggym_core.prior_particles` read
        it to size the per-level offset vector.
        """
        return tuple((name, levels) for name, _, levels in self.indexed_parameters)

    @property
    def model_id(self):
        """The canonical model identity (20 hex digits of a sha256), from `canonical_model_id`."""
        return canonical_model_id(
            self.expression, self.family, self.parameters, self.coefficient_priors
        )

    @property
    def model_id_preimage(self):
        """The canonical JSON text whose sha256 gives `model_id`."""
        return model_id_preimage(
            self.expression, self.family, self.parameters, self.coefficient_priors
        )

    def as_dict(self):
        """The JSON-ready record: id, expression, family, roles and the priors actually used."""
        record = {
            "model_id": self.model_id,
            "expression": self.expression,
            "family": self.family,
            "parameters": dict(self.parameters),
            "noise_parameters": self.noise_parameters,
            "fixed_priors": {
                name: coefficient_prior(self, name, role) for name, role in self.parameters
            }
            | {name: PRIOR_RULES[name] for name in self.noise_parameters},
        }
        # Only a candidate that actually uses an indexed coefficient carries the
        # `indexed_parameters` key, so every earlier task's records are byte-identical. The
        # prior is NOT a new one: it is the single `signed` = Normal(0,1) row
        # already listed in `fixed_priors`, shared by all n_levels offsets.
        # LLM-proposed-prior ablation: `prior_source` is recorded for EVERY model of a run
        # that declared `--priors llm` - "role_template" when this candidate
        # proposed nothing, "llm" when it did - and for NO model of a run that did
        # not declare it, so a default run's records are byte-identical to the pre-ablation ones.
        if llm_priors_enabled() or self.coefficient_priors:
            record["prior_source"] = "llm" if self.coefficient_priors else "role_template"
        if self.coefficient_priors:
            record["proposed_priors"] = {
                name: [rule[0], rule[1], rule[2]] for name, rule in self.coefficient_priors
            }
        indexed = self.indexed_parameters
        if indexed:
            record["indexed_parameters"] = {
                name: {"input": input_name, "n_levels": levels, "prior_shared_across_levels": True}
                for name, input_name, levels in indexed
            }
        return record


def indexed_parameters(expression, categorical_levels=()):
    """`((coefficient, categorical input, n_levels), ...)` of one expression.

    Reads the already-validated AST, so it never decides admissibility - that
    is `validate_expression`'s job. Sorted by coefficient name for a stable
    record and a stable PyMC variable order.

    The two-different-inputs ValueError below is now a SAFETY NET, not the
    primary admissibility gate:
    `validate_expression` rejects that same conflict while walking the AST,
    before a `Hypothesis` for it can be constructed at all, so this branch
    should be unreachable through `Hypothesis`/`judge_candidate`/
    `parse_candidate_object`. It is deliberately left in place for any caller
    that calls this function directly, on an expression `validate_expression`
    never saw.
    """
    levels = dict(categorical_levels)
    if not levels:
        return ()
    found = {}
    for node in ast.walk(ast.parse(expression, mode="eval")):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            name, index = node.value.id, node.slice.id
            if name in found and found[name] != index:
                raise ValueError(f"coefficient {name} is indexed by two different inputs")
            found[name] = index
    return tuple((name, index, levels[index]) for name, index in sorted(found.items()))


def evaluate(expression, parameters, u, backend=np, inputs=None, index_codes=None):
    """Interpret one validated expression.

    `inputs` is the named-input mapping (from `named_input_values`); when it is
    None the CORE namespace `{"u": u, "z": 2*u-1}` is used, which is exactly the
    original hard-coded `u`/`z` behaviour. `u` remains the broadcasting anchor in both cases.

    `index_codes` is `{categorical input: int codes}` from
    `input_index_codes`, needed only by an expression containing an indexed
    coefficient `a[student]`. That form gathers along the coefficient's LAST
    axis - `backend.take(a, codes, axis=-1)` - so one call serves both backends
    and both layouts: a NumPy particle block shaped [particle, level] becomes
    [particle, observation], and a PyTensor per-level vector shaped [level]
    becomes [observation]. `index_codes=None` with a subscript present is a
    programming error and raises; without a subscript it is never read, so every
    existing call site is unchanged.
    """
    values = {**parameters, **({"u": u, "z": 2 * u - 1} if inputs is None else inputs)}
    codes = dict(index_codes or {})

    def walk(node):
        """Evaluate one node on the chosen backend."""
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Subscript):
            if node.slice.id not in codes:
                raise ValueError(
                    f"no level codes supplied for categorical input {node.slice.id}"
                )
            return backend.take(values[node.value.id], codes[node.slice.id], axis=-1)
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
        if _STABLE_LOG1MEXP:  # stable log1mexp switch; off by default, taken only when turned on
            operand = log1mexp_operand(node)
            if operand is not None:
                return log1mexp(walk(operand), backend)
        args = [walk(arg) for arg in node.args]
        if node.func.id == "square":
            return args[0] ** 2
        if node.func.id == "power":
            return args[0] ** args[1]
        # `sigmoid` is written out because NumPy has no name
        # of its own for it, as the half-shifted hyperbolic tangent rather than
        # `1/(1+exp(-x))`. The two are the same function, but this spelling has
        # no overflow branch at all - `tanh` saturates - so a candidate whose
        # coefficients drive the argument far from zero costs the fit a
        # saturated value instead of an overflow warning and a division. The
        # other five added calls (`abs`, `expm1`, `log1p`, `minimum`,
        # `maximum`) are spelled identically on both backends and go through the
        # same attribute lookup the v2 calls always did.
        if node.func.id == "sigmoid":
            return 0.5 * (1 + backend.tanh(0.5 * args[0]))
        return getattr(backend, node.func.id)(*args)

    return walk(ast.parse(expression, mode="eval")) + 0 * u


def particle_logpmf(hypothesis, particles, u, n_trials=50, interface=None):
    """Normalized log PMFs: [particle, input, count], including combinatorics.

    With `interface` given, the designs are read through the named-input block
    (one column per declared input, plus the derived inputs) instead of the CORE
    `u`/`z` pair. With `interface=None` the CORE namespace is used and every
    float is identical to the original hard-coded `u`/`z` path.
    """
    u = np.asarray(u, dtype=float)
    codes = {}
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
        # `{}` for every interface without a categorical input, so the
        # `evaluate` call below is unchanged for CORE and every earlier task.
        codes = input_index_codes(interface, u)
    # An INDEXED coefficient's particles are already a [particle, level] block:
    # the gather in `evaluate` supplies the observation axis, so it must NOT get
    # the scalar path's trailing `[:, None]`.
    shapes = dict(getattr(hypothesis, "parameter_shapes", ()) or ()) if codes else {}
    params = {}
    for name, _ in hypothesis.parameters:
        column = np.asarray(particles[name], dtype=float)
        params[name] = column if name in shapes else column[:, None]
    sizes = {value.shape[0] for value in params.values()}
    if len(sizes) != 1 or next(iter(sizes)) == 0:
        raise ValueError("particle columns differ or are empty")
    for name, levels in shapes.items():
        if params[name].ndim != 2 or params[name].shape[1] != levels:
            raise ValueError(f"indexed coefficient {name} needs {levels} per-level particles")
    with np.errstate(all="ignore"):
        logits = evaluate(hypothesis.expression, params, anchor, np, named, codes)
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
    # Numeric rule: an UNDEFINED particle log pmf is
    # IMPOSSIBLE, not a number - NaN is DEFINED as -inf. No prior, no likelihood
    # and no constant changes: on the domain where the log pmf is a number the
    # returned array is bit-identical to earlier versions. Only cells they refused
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
    total = np.exp(result).sum(axis=-1)
    deviation = np.abs(total - 1.0)
    # `~(dev <= atol)` and not `dev > atol`: exactly the condition np.allclose(
    # ..., atol=1e-10, rtol=0) applied per row, so a NaN deviation still fails.
    failing = ~(deviation <= NORMALIZATION_ATOL)
    if not np.any(failing):
        return result
    if np.any(nan_cells):
        # A NaN cell that carried real mass leaves its row unnormalized. That is
        # the same non-finite defect earlier versions rejected, and it keeps the SAME message
        # so core_count.PRIOR_PROBE_NONFINITE classifies it exactly as before; a
        # row that still normalizes was NaN only where the mass is negligible.
        # The drift tolerance leaves this refusal exactly as it was.
        raise ValueError("invalid likelihood")
    # The drift tolerance's three-way rule, on rows with no NaN cell at all. Below the
    # tolerance nothing happens (handled above); up to `DRIFT_TOL` the loss is
    # floating-point drift and the row is rescaled by its own sum; beyond it the
    # particle lost real mass and is given zero weight instead of ending the run.
    zeroed = np.any(~(deviation <= DRIFT_TOL), axis=1)
    drift = failing & ~zeroed[:, None]
    _NORMALIZATION_EVENTS["calls"] += 1
    _NORMALIZATION_EVENTS["renormalized_rows"] += int(drift.sum())
    _NORMALIZATION_EVENTS["zeroed_rows"] += int(zeroed.sum())
    _NORMALIZATION_EVENTS["max_deviation"] = max(
        _NORMALIZATION_EVENTS["max_deviation"], float(deviation[failing].max())
    )
    if np.all(zeroed):
        # No particle carries weight, so this model has no predictive at all:
        # the same refusal, with the same message, as before the drift tolerance.
        raise ValueError(NORMALIZATION_FAILURE)
    if np.any(zeroed):
        result = np.where(zeroed[:, None, None], -np.inf, result)
    if np.any(drift):
        # Subtracting 0.0 leaves every untouched row bit-identical, -inf included.
        shift = np.where(drift, np.log(np.where(drift, total, 1.0)), 0.0)
        result = result - shift[:, :, None]
    return result


CORE_INITIAL_STRUCTURES = ("b0", "b0+b1*z", "b0+b1*z+b2*square(z)", "b0+b1*log(1+u)")


def initial_hypotheses(
    family="binomial", forms=None, input_names=DEFAULT_INPUT_NAMES, categorical_levels=()
):
    """The four initial structures, written in the interface's own input names.

    `forms` defaults to CORE's four; a task specification supplies its own list
    written over its own declared inputs (see `boxinggym_core.TASKS`).
    `categorical_levels` defaults to empty and is passed straight to
    `Hypothesis`, so a task with no categorical input is unchanged.
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
                tuple(categorical_levels),
            )
        )
    return result


def judge_candidate(
    item,
    cell,
    initial_structures,
    initial_family,
    input_names=None,
    allowed_families=None,
    categorical_levels=None,
):
    """Judge one already shape-valid candidate; never raises (fail-soft).

    `item` must already be a dict with expression/family/parameters (parameters
    itself a dict of coefficient name -> role, with any reserved variance names
    such as rho/pi already filtered out by the caller - see
    count_proposer.RESERVED_VARIANCE_PARAMETER_NAMES). Shape
    violations (missing keys, non-dict parameters) are the caller's concern,
    not this function's - both current callers (count_proposer.parse_candidates
    and core_count.validate_cell_candidates) only ever pass shape-valid items.

    Covers exactly the three per-candidate violation categories:
    grammar (AST/coefficient/operator rules, via attempting
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
    "unknown input name <id>".

    `input_names` is the interface namespace; when omitted it is taken from
    `initial_structures` (which the caller builds from the same interface), and
    falls back to CORE's `("u", "z")` only when there are no initial structures.

    `allowed_families` is this task/cell's admissible family menu
    - `belief_packet.cell_constraints`'s `constraints["allowed_families"]`,
    itself `belief_packet.interface_families`'s outcome-type menu (the paper's
    task table) intersected with `IMPLEMENTED_FAMILIES`. It defaults to None,
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
    # Like `input_names`, the categorical block is taken from the
    # caller's own initial structures when it is not passed explicitly, and is
    # EMPTY for every task that declares no categorical input - so this judge is
    # unchanged for CORE and for every BoxingGym task without a categorical input.
    levels = (
        tuple(getattr(initial_structures[0], "categorical_levels", ()) or ())
        if categorical_levels is None and initial_structures
        else tuple(categorical_levels or ())
    )
    allowed = FAMILIES if allowed_families is None else tuple(allowed_families)
    try:
        if item["family"] not in allowed:
            raise ValueError(f"unsupported count family: {item['family']}")
        hypothesis = Hypothesis(
            item["expression"], item["family"], tuple(item["parameters"].items()), names, levels
        )
        structure_expanded, family_expanded = (bit == "1" for bit in cell[:2])
        if not family_expanded and hypothesis.family != initial_family:
            return False, F0_REFUSAL_REASON
        if not structure_expanded:
            initial_ids = {h.model_id for h in initial_structures}
            structure_id = Hypothesis(
                hypothesis.expression, initial_family, hypothesis.parameters, names, levels
            ).model_id
            if structure_id not in initial_ids:
                return False, "S0 requires a structure from the initial pool"
        return True, None
    except Exception as exc:  # noqa: BLE001 - fail-soft judge: reject, never raise
        return False, str(exc)


#: The six failure kinds a candidate record names separately, as fixed by an internal
#: review on 2026-09-23. An admitted candidate records None; a transport error that
#: returned no reply to judge is not a candidate record and records no class at all.
FAILURE_CLASSES = ("parse_format", "reserved_name", "coefficient_count", "prior_probe",
                   "fit_failed", "duplicate")
#: The two refusals `validate_expression` raises verbatim. `failure_class` compares a reason
#: with them whole, never a fragment of it; a test binds each to the message the engine raises.
COEFFICIENT_COUNT_REFUSAL = "require 1..6 distinct mean coefficients"
RESERVED_NAME_REFUSAL = "invalid or reserved coefficient name"
#: The one refusal that quotes other refusals: a k > 1 array reply in which no candidate was
#: accepted (`count_proposer.parse_candidate_array`) joins every candidate's own message after
#: this prefix. It refuses the reply as a whole, so it is parse_format whatever the quoted
#: messages are; they are not unwrapped into classes of their own.
ARRAY_REFUSAL_PREFIX = "no candidate in the array was accepted: "
#: Every status a runner writes on a per-candidate outcome record, with its class. A status
#: outside this table is refused by `failure_class`, never given a default.
CANDIDATE_STATUS_CLASSES = {
    "fitted": None,
    "duplicate_not_refitted": "duplicate",
    "rejected_prior_probe": "prior_probe",
    "rejected_loglik_prior_probe": "prior_probe",
    "fit_failed": "fit_failed",
    "fit_timeout": "fit_failed",
    "fit_or_predict_failed": "fit_failed",
    "predict_failed": "fit_failed",
    "rejected_family": "parse_format",
}


def failure_class(*, status=None, error_type=None, reason=None, repair_eligible=None):
    """The one mapping from a refusal or an outcome to its failure class.

    A runner's per-candidate outcome record passes its `status` alone and gets its row of
    CANDIDATE_STATUS_CLASSES; an unknown status raises. A refusal of a reply or of one
    candidate before admission passes its recorded `error_type` and `reason` - and a transport
    repair-turn entry its `repair_eligible` too: a PriorProbeRejection is prior_probe, a reason
    EQUAL to COEFFICIENT_COUNT_REFUSAL or RESERVED_NAME_REFUSAL is coefficient_count or
    reserved_name, and every other refusal - invalid JSON, the reply, candidate or parameters
    shape, grammar, role, cell structure or family, an empty body, a finish reason that is not
    the completed one, and ARRAY_REFUSAL_PREFIX - is parse_format. An entry that is not
    repair-eligible is an API, SDK or transport error with no reply judged: it has no class,
    and asking for one raises.
    """
    if status is not None:
        if error_type is not None or reason is not None or repair_eligible is not None:
            raise ValueError("a candidate status is classified on its own")
        if status not in CANDIDATE_STATUS_CLASSES:
            raise ValueError(f"unknown candidate status {status!r} has no failure class")
        return CANDIDATE_STATUS_CLASSES[status]
    if repair_eligible is False:
        raise ValueError("a transport error that is not repair-eligible is not a candidate record")
    if not isinstance(reason, str):
        raise ValueError("a refusal is classified by its recorded reason")  # noqa: TRY004 - Record validation uses ValueError.
    if error_type == "PriorProbeRejection":
        return "prior_probe"
    if reason.startswith(ARRAY_REFUSAL_PREFIX):
        return "parse_format"
    if reason == COEFFICIENT_COUNT_REFUSAL:
        return "coefficient_count"
    if reason == RESERVED_NAME_REFUSAL:
        return "reserved_name"
    return "parse_format"


#: The round-0 burst's refill rule, written into every burst pool file and into
#: config.json whenever a burst runs: at most three waves, and each refill wave (1, 2) asks
#: for exactly the shortfall with the round's admitted candidates in its packet's pool.
INITIAL_BURST_REFILL_POLICY = "pool_visible_refill_waves_max_3"


@dataclass(frozen=True)
class InitialPoolFile:
    """A frozen initial pool read from a file: its hypotheses, source kind and file sha256."""

    hypotheses: tuple[Hypothesis, ...]
    source: str
    sha256: str

    def metadata(self):
        """The pool's source, sha256 and size, as config.json keys."""
        return {"initial_pool_source": self.source, "initial_pool_sha256": self.sha256,
                "initial_pool_count": len(self.hypotheses)}


def read_initial_pool_file(filename, *, cell, initial_family, initial_structures=None,
                           allowed_families=None, input_names=None, categorical_levels=None):
    """Read frozen state, not the burst transcript (a rule of the initial-pool-file
    design).

    Burst objects bind each after_id to exactly one event, without status or
    positional filtering. Plain arrays are the synthetic-probe format. Neither
    path rebuilds roles through the catalogue constructor or drops bad entries.
    """
    raw = Path(filename).read_bytes()
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("initial pool file is not valid JSON") from exc
    if isinstance(payload, dict):
        require_model_id_schema(payload)
        # A burst pool is frozen only under the refill rule that produced it; a
        # file written before that rule (no key) or under another one is refused whole.
        refill_policy = payload.get("initial_burst_refill_policy")
        if refill_policy != INITIAL_BURST_REFILL_POLICY:
            raise ValueError(f"initial pool burst requires initial_burst_refill_policy "
                             f"{INITIAL_BURST_REFILL_POLICY!r}; got {refill_policy!r}")
        source = "file_burst"
        if payload.get("initial_catalogue") != "off":
            raise ValueError("initial pool burst requires initial_catalogue off: catalogue models "
                             "are not recorded in events, so reconstruction would be partial")
        ids, events = payload.get("after_ids"), payload.get("events")
        if (not isinstance(ids, list) or not all(isinstance(key, str) and key for key in ids)
                or not isinstance(events, list) or not all(isinstance(event, dict) for event in events)):
            raise ValueError("initial pool burst requires after_ids and candidate events arrays")
        entries = []
        for key in ids:
            # The burst writer records every proposal event under the id it resolved to:
            # the admitted hypothesis carries status "fitted" and each later duplicate of it
            # carries "duplicate_not_refitted" under the SAME id (core_count parallel wave).
            # Reconstruction therefore binds each retained id to its one fitted event and
            # leaves the duplicate records in place; two fitted events for one id, or none,
            # is an ambiguous file and is refused whole (an unchanged rule of the
            # initial-pool-file design).
            same_id = [event for event in events if event.get("model_id") == key]
            matches = [event for event in same_id if event.get("status") == "fitted"]
            if len(matches) != 1:
                raise ValueError(f"initial pool after_id {key!r} needs exactly one fitted event; "
                                 f"found {len(matches)}")
            # The writer can only put duplicates beside a fitted record (the first event of an
            # id decides its fate; later ones are duplicate_not_refitted), so any other status
            # beside the fitted one describes a file no writer produced: refuse it whole.
            strays = sorted({str(event.get("status")) for event in same_id
                             if event is not matches[0] and event.get("status") != "duplicate_not_refitted"})
            if strays:
                raise ValueError(f"initial pool after_id {key!r} has non-duplicate events beside its "
                                 f"fitted record: {strays}")
            entries.append(matches[0])
    elif isinstance(payload, list):
        source, entries = "file_candidates", payload
    else:
        raise ValueError("initial pool file must be a burst object or a candidate array")  # noqa: TRY004 - Serialized-input validation uses ValueError.
    if not entries:
        raise ValueError("initial pool file contains no hypotheses")
    structures = (initial_hypotheses(initial_family) if initial_structures is None
                  else tuple(initial_structures))
    names = (tuple(structures[0].input_names) if input_names is None and structures
             else tuple(DEFAULT_INPUT_NAMES if input_names is None else input_names))
    levels = (tuple(getattr(structures[0], "categorical_levels", ()))
              if categorical_levels is None and structures else tuple(categorical_levels or ()))
    hypotheses = []
    for index, entry in enumerate(entries):
        if (not isinstance(entry, dict)
                or not {"expression", "family", "parameters"} <= entry.keys()
                or not isinstance(entry["parameters"], dict)):
            raise ValueError(f"initial pool entry {index} requires expression/family/parameters")
        ok, reason = judge_candidate(entry, cell, structures, initial_family, names,
                                     allowed_families, levels)
        if not ok:
            raise ValueError(f"initial pool entry {index}: {reason}")
        hypothesis = Hypothesis(entry["expression"], entry["family"],
                                tuple(entry["parameters"].items()), names, levels)
        if source == "file_burst" and hypothesis.model_id != entry["model_id"]:
            raise ValueError(f"initial pool entry {index}: reconstructed model_id mismatch")
        hypotheses.append(hypothesis)
    return InitialPoolFile(tuple(hypotheses), source, hashlib.sha256(raw).hexdigest())
