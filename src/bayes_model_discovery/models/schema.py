"""PRODUCTION parameter schema for the closed-pool posterior system.

Single source of truth for the registered role/transform table - previously
duplicated inside the checker (external-review defect). Keys are the VERIFIED
canonical strings of the six library forms (the locked evaluator reorders
parameters positionally, so roles attach to canonical positions, never to the
original p1/p2/p3 labels). `tests/unit/test_models_schema.py` cross-validates
this table against the locked `models.expressions.canonical` and against
numeric instances; empirical-Bayes pooling is WITHIN-role only (units differ
across roles).

Roles (unit, positivity, transform):
  scale_output           y units, positive, log
  rate                   1/x units, positive, log
  offset_output          y units, sign-free, identity
  midpoint_input         x units, sign-free, identity
  slope_linear           y/x units, sign-free, identity
  coef_sqrt              y per sqrt-x units, sign-free, identity
  half_saturation_input  x units, positive, log

Notes from porting this module from the earlier prototype
---------------------------------------------------------
Mechanical. Every key, string, boolean and the import-time assertion below are
unchanged from the archived file. The only edits are the removal of the
`# -*- coding: utf-8 -*-` line (this package is UTF-8 by declaration of Python
3) and the two docstring pointers that named archive-local modules: the
checker is now the unit test named above, and `model_spec.formalize()` - which
assigns the two free-form roles - has NOT been migrated, so the note about it
describes the archived behaviour that the free-form roles exist to serve.

ONE ADDITION, not a transcription: `TRANSFORMS`, the declared transform
vocabulary, with an import-time assertion tying it to the role table. The
archived file had no such name because the archived callers all read their
transforms out of `ROLES`; this package accepts them from a spec file and from
`run_smc`'s caller as well, and those two doors needed something to check
against. See the note on `TRANSFORMS` below.
"""

ROLES = {
    "scale_output":          {"transform": "log",      "positive": True},
    "rate":                  {"transform": "log",      "positive": True},
    "offset_output":         {"transform": "identity", "positive": False},
    "midpoint_input":        {"transform": "identity", "positive": False},
    "slope_linear":          {"transform": "identity", "positive": False},
    "coef_sqrt":             {"transform": "identity", "positive": False},
    "half_saturation_input": {"transform": "log",      "positive": True},
    # Generic roles for FREE-FORM (LLM-proposed) candidates, whose semantic
    # role cannot be pre-registered. They are assigned by evidence, never by
    # default: in the archived system `model_spec.formalize()` gives a
    # parameter `positive_generic` only when the expression is demonstrably
    # undefined for non-positive values on the support, otherwise
    # `unconstrained_generic`. Empirical-Bayes pooling must NOT mix these with
    # the named library roles.
    "positive_generic":      {"transform": "log",      "positive": True},
    "unconstrained_generic": {"transform": "identity", "positive": False},
}
FREEFORM_ROLES = ("positive_generic", "unconstrained_generic")

#: The COMPLETE phi -> theta transform vocabulary of this repository, declared once.
#:
#: Every phi -> theta map in the package is written as `exp(phi) if t == "log" else phi`,
#: which means an unrecognised transform string is silently applied as the identity. That
#: is not a crash and not a warning: the run samples a different model from the one that
#: was asked for, records the transform it was given in its own sidecars, and looks
#: ordinary afterwards. The set is therefore declared here - beside the role table that is
#: the only thing entitled to mint a transform - and every entry point that accepts a
#: transform from outside the package checks membership against THIS name rather than
#: repeating the two strings.
#:
#: It is exactly the vocabulary the frozen evidence uses, not an assumption: the 352
#: `state.json` files of the archived run carry 2146 `identity` and 462 `log` and nothing
#: else, and the hash-pinned fixture under tests/fixtures/acquisition_round1 carries 65 and
#: 13 of the same two. `model_spec.formalize()`, the archive-side formalizer that assigned
#: the transforms, read them off `ROLES` exactly as the assertion below requires.
TRANSFORMS = frozenset({"identity", "log"})

# The role table is the only producer of transforms, so the declared vocabulary must be
# precisely the set it produces. `==`, not `<=`: a role introducing a third transform must
# break here rather than travel out to callers that would apply it as the identity, and a
# vocabulary entry no role can produce is a claim this table does not support.
assert {_spec["transform"] for _spec in ROLES.values()} == TRANSFORMS, (
    "the role table and the declared transform vocabulary disagree: roles produce "
    f"{sorted({_s['transform'] for _s in ROLES.values()})}, TRANSFORMS declares "
    f"{sorted(TRANSFORMS)}")

# original library form -> (verified canonical string,
#                           positional roles of the canonical parameters)
LIBRARY_PARAM_SCHEMA = {
    "p1": (
        "p1", ("offset_output",)),
    "p1 + p2*x": (
        "p1 + p2 * x", ("offset_output", "slope_linear")),
    "p1 * sqrt(x)": (
        "p1 * sqrt(x)", ("coef_sqrt",)),
    "p1*(1-exp(-p2*x))": (
        "(-(exp(-p1 * x)) + 1.0) * p2", ("rate", "scale_output")),
    "p1/(1+exp(-p2*(x-p3)))": (
        "p1 / (1.0 + exp(-(-p2 + x) * p3))",
        ("scale_output", "midpoint_input", "rate")),
    "p1*x/(p2+x)": (
        "(p1 * x) / (p2 + x)", ("scale_output", "half_saturation_input")),
}

for _orig, (_canon, _roles) in LIBRARY_PARAM_SCHEMA.items():
    for _r in _roles:
        assert _r in ROLES, f"unknown role {_r!r} for {_orig!r}"
