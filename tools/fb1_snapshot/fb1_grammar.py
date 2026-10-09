"""ForceBench free-form force-law grammar.

Turns a **free-form expression** over the named inputs ``{r, p1, p2, t}`` into a jax
callable with exactly the same signature and the same broadcasting convention as the
force functions in ``fb1_forward``, and registers a ``fb1_core.Structure`` holding it in
``fb1_core.STRUCTURES`` **at runtime**.

    force(r, p1, p2, th, t) -> (D, P)      r (D,P) · p1,p2 (D,1) · th (P,k) · t scalar
    th[:, j] = the j-th free constant         (the convention fb1_forward.integrate calls with)

``fb1_core.Candidate.kinds``/``make_loglik``/``make_predict`` look structures up through
``STRUCTURES[cand.struct]``, so registration alone lets a free-form candidate travel the
fitting, prediction and scoring paths unchanged, **without editing a single character of
the protected files (fb1_core.py, fb1_forward.py)**.

No new grammar engine is written.  The ``count_models.py`` of the canonical bmd engine chain
is imported byte for byte as ``vendor/count_models_v2_9.py`` (source and sha in
``vendor/PROVENANCE.md``), and its ``validate_expression`` · ``canonical_model_id`` ·
``evaluate`` · ``CALL_ARITY`` are called.  sympy is not used (it is not in the run
environment, and installing packages was not an option).

This module does not import ``fb1_proposer``: the proposer side imports this module (to
decide free mode), so importing it back would create a cycle.
"""
import ast
import hashlib
import importlib.util
import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import fb1_core as core
from fb1_core import STRUCTURES, Structure

# The vendored grammar engine is loaded **directly by file path**: this does not touch
# sys.path (no side effect on the runner's import order) and pins by path which file was loaded.
VENDOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "vendor", "count_models_v2_9.py")
_spec = importlib.util.spec_from_file_location("fb1_vendor_count_models_v2_9", VENDOR_PATH)
vendor = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = vendor
_spec.loader.exec_module(vendor)

# sha256 of the vendored file.  One test checks that it matches the original (same value as PROVENANCE.md).
VENDOR_SHA256 = "38c5afe813f89342b50d3a7482a3129ce07ab90c266866100d849ebfebdf966b"
VENDOR_CHAIN = "bayesian-model-discovery canonical engine chain v2.9 (the version this kit vendored)"
VENDOR_SOURCE = ("count_models.py of the canonical engine chain v2.9, taken from the private development tree; "
                 "the vendored copy is identified by VENDOR_SHA256")

# ---------------------------------------------------------------- grammar vocabulary (all declared in advance)
# The names a force law may see.  r = distance between the two particles · p1,p2 = design
# parameters · t = elapsed time.  These four are exactly what fb1_forward.integrate passes to
# force() (any other name is a grammar violation).
INPUT_NAMES = ("r", "p1", "p2", "t")

# Role vocabulary.  Only **two** of the vendor's four roles are accepted:
#   positive -> fb1 kind "pos"  (LogNormal(0,1);  u ~ N(0,1), x = exp u)
#   signed   -> fb1 kind "sig"  (Normal(0,1);     x = u)
# intercept and shape are **rejected** because fb1 has no matching prior kind (collapsing them
# onto pos/sig would give a different model with a different prior scale; same reasoning as
# the belief-packet schema).
# The catalogue kinds that carry a scale (pos_bg, pos_w) are not used in free-form expressions
# either: those scales come from the physical basis of a particular template (box half-width,
# measurement time span) and cannot be carried over to an arbitrary free-form expression.
ROLE_TO_KIND = {"positive": "pos", "signed": "sig"}
ALLOWED_ROLES = tuple(sorted(ROLE_TO_KIND))
# The inverted table.  Used to read roles back from Structure.kinds for the role-multiset
# comparison; ROLE_TO_KIND is 1:1, so the inverse is 1:1 too.
KIND_TO_ROLE = {kind: role for role, kind in ROLE_TO_KIND.items()}

# Caps (n_ops <= catalogue maximum, free constants <= 3).
MAX_FREE_CONSTANTS = 3
# Freeze the catalogue structure names **before any registration**: recomputing after free
# structures have been put into STRUCTURES would raise the cap by itself, so it would no
# longer be a cap.
CATALOG_STRUCTURE_NAMES = tuple(STRUCTURES)
# The maximum complexity the catalogue **declares** (maximum of Structure.n_ops).  Recorded only.
N_OPS_MAX_DECLARED = max(STRUCTURES[name].n_ops for name in CATALOG_STRUCTURE_NAMES)
# The cap actually enforced is obtained by measuring CATALOG_GRAMMAR_EXPRESSIONS below **with
# the same ruler as a free-form expression** (N_OPS_MAX at the end of the module).  Why the two
# values differ and why the same ruler must be used is written in the docstring of
# count_operators and where N_OPS_MAX is defined.

# Family placeholder for structure identity.  The vendor's canonical_model_id hashes (family,
# canonical AST) together, but in fb1 the identity of a **structure** must not include the noise
# family (a Candidate = (structure, family) carries the family separately, and setting up the
# same expression as two candidates, one with gauss and one with student_t4, is how catalogue
# mode works).  So a fixed token is put in the family slot to obtain a canonical identifier of
# **the structure alone**.
STRUCTURE_ID_FAMILY = "fb1_force_structure"
NAME_PREFIX = "g_"
NAME_ID_CHARS = 12

# --------------------------- finiteness probe grid (all declared in advance)
# r grid [0.3, 10] 32 points x **all measurement times TIMES** x 64 prior draws x **13**
# (p1,p2) points (4 box corner combinations + a fixed 9-point grid inside the box, see
# `probe_points()`).
# Any non-finite value (nan/inf) is rejected as REASON_NONFINITE, and |F| > 1e6 is rejected as well.
# The time axis was added later (the first version fixed t at
# 0 alone and therefore missed singularities that exist only in t; adding the axis removed that
# limitation).  The interior points were added later still (the corner grid
# alone could not see a term that vanishes only at the endpoints of the p axis; see
# PROBE_P_INTERIOR_FRACTIONS below).
# Trigonometric functions are not in the grammar (the vendored bytes are kept).
PROBE_R_RANGE = (0.3, 10.0)
PROBE_R_POINTS = 32
PROBE_DRAWS = 64
# (p1,p2) range = run_gravity.P1_RANGE = P2_RANGE = (0.1, 10.0).  4 corner combinations + 9 interior points are used.
PROBE_P_RANGE = (0.1, 10.0)
PROBE_ABS_MAX = 1.0e6
# Probe points **inside** the (p1,p2) box are declared in advance
# as well.  With the 4 corners alone, a law with a term that vanishes only at the endpoints of
# the p axis could not be told apart from the law without it; counterexample
# `c*p1+c*(p2-0.1)*(p2-10)` equals `c*p1` exactly at p2 = 0.1 and 10, but at c=p1=p2=1 it is
# -7.1 versus 1.  run_gravity's actual design box contains the whole interior.  The grid is the
# 1/4, 1/2, 3/4 quantiles of the box, 3x3 = 9 points, and `probe_identity()` writes the
# coordinates into the output as they are.
PROBE_P_INTERIOR_FRACTIONS = (0.25, 0.5, 0.75)
# Probe-only key.  It consumes neither the main run's numpy stream, nor the SMC keys, nor the
# design stream (and it differs from run_gravity's PRIOR_PROBE_SEED=20260908, so the two probes
# never share a stream).
PROBE_SEED = 20260909

# ------------------------ numerical-equivalence duplicate check (declared in advance)
# Pairing: when the candidate and a comparison target (registered free
# structures + the 13 catalogue transcriptions) have the same **number** of coefficients,
# **every prior-preserving pairing** is checked (`_prior_preserving_pairings`; at most 3 free
# coefficients, so at most 6 permutations per pair).  If any one of them makes the forces agree
# over the whole grid, the two are treated as the same law (relative difference < 1e-12, and
# absolute difference < 1e-12 at points where the force is near 0).  Because the prior must be
# preserved coefficient by coefficient, a pair that only shares the role multiset (for example
# a:positive,b:signed versus a:signed,b:positive) stays as two different models.
# The first pass matched only the first-occurrence position (`_pair_by_canonical_position`)
# and so missed multi-coefficient terms whose summation order was reversed, and it compared only
# role multisets and so merged models whose priors were swapped.
# A pair with a different count cannot be paired at all and is not compared (the
# same function with a different coefficient count, e.g. an absorbed constant, is disclosed as
# undetected).  A pair with no prior-preserving pairing at all (= a pair whose role multisets
# differ) is **not rejected**: if the forces agree it is recorded as
# `same_function_different_roles_of` and admission is still allowed (the same function with
# a different prior scale is a different Bayesian model).
# Verdict: the **kind** of the matched target decides the outcome. A registered free
# structure is rejected; a catalogue entry with the same priors gives a **link** to that name
# (rejection is decided by the caller, who knows the pool and the history); a catalogue entry
# with a different prior scale is not merged, only recorded.
# Order: this check runs **after** the finiteness probe; a non-finite grid cannot
# be evidence of equivalence.  Non-finite points on the target side also void that one comparison.
# Wording: a match means "indistinguishable on the probe grid" (`PROBE_VERDICT`); the
# grid is finite and declared in advance, so it is not a proof of general functional equivalence.
# `canonical_model_id` (shared with CORE) is not touched at all: the **same traversal rule** is
# reproduced locally here only to obtain the position table (`_canonical_positions`, now used
# for the annotation only).
EQUIVALENCE_RTOL = 1e-12
EQUIVALENCE_ATOL = 1e-12
# Threshold for "the force is near 0", judged by absolute difference.  Where |F| is below this, relative difference loses its meaning.
EQUIVALENCE_NEAR_ZERO = 1e-300

# Rejection and duplicate reason codes.  These strings go verbatim into the archive and the output JSON (callers do not rewrite them).
REASON_NONFINITE = "nonfinite_on_probe_grid"
REASON_ABS_MAX = "force_magnitude_over_probe_cap"
REASON_DUPLICATE_CANONICAL = "duplicate_by_canonical_id"
REASON_DUPLICATE_NUMERICAL = "duplicate_by_numerical_equivalence"
# The catalogue is a **reference table**, not a rejection list.  When a free
# proposal is the same prior model as a catalogue law it is **linked** to that catalogue name,
# and whether to reject is decided by the caller from whether (structure identity, family) is
# in the active pool or the evaluated history.  This is not a rejection reason: `outcome` is
# OUTCOME_CATALOG_REFERENCE, neither "duplicate" nor "active".
OUTCOME_CATALOG_REFERENCE = "catalog_reference"
REASON_CATALOG_REFERENCE = "catalog_reference_same_prior_model"
CATALOG_REFERENCE_KEY = "catalog_structure"
# Kinds of comparison target.  A catalogue target is used for reference linking, and a
# catalogue target with a different prior scale (pos_bg of inv_r_bg) is a **different Bayesian
# model** even for the same function, so it is never merged.
TARGET_FREE_REGISTERED = "free_registered"
TARGET_CATALOG = "catalog"
TARGET_CATALOG_PRIOR_SCALE_DIFFERENCE = "catalog_prior_scale_difference"
# Verdict wording.  The probe is a finite grid declared in advance: a match only
# means "not distinguishable on that grid", never a proof of general functional equivalence.
# This wording is used verbatim in outputs and reasons.
PROBE_VERDICT = "indistinguishable_on_the_probe_grid"
PROBE_VERDICT_NOTE = (
    "the probe grid is finite and pre-declared: a match means the two forces are "
    "INDISTINGUISHABLE ON THE PROBE GRID (r grid x all measurement times x prior draws x "
    "declared (p1,p2) corner AND interior points), which is never a proof of general "
    "functional equivalence")
# This is an **annotation**, not a rejection reason: even with
# this key present the outcome stays "active".  The value is the label of the first comparison
# target met whose function is the same (but which is not a duplicate because its role multiset
# differs).
SAME_FUNCTION_DIFFERENT_ROLES_KEY = "same_function_different_roles_of"


# ---------------------------------------------------------------- catalogue written in the grammar
# The catalogue structures written **in this grammar** (13 of the 14; inv_r_tmod cannot be, see
# below).  To see whether a free-form
# expression is equivalent to a catalogue law (canonical identifier collision), the catalogue
# side must be written in the same grammar too.  Whether a transcription really is the same
# function as the catalogue force function is checked numerically on a grid by a test
# (test_catalog_transcriptions_match_the_catalog_force_numerically).
#
# inv_r_tmod **cannot be transcribed**: cos and sin are not in the vendor CALL_ARITY
# (exp, log, sqrt, tanh, square, power).  So it can never be proposed in this grammar in the
# first place, and it is also left out of the duplicate comparison targets.
CATALOG_GRAMMAR_EXPRESSIONS = {
    "inv_r2":         ("c*p1/r**2", (("c", "positive"),)),
    "linear":         ("c*p1*r", (("c", "positive"),)),
    "const":          ("c*p1", (("c", "positive"),)),
    "screened":       ("c*p1*exp(-r/lam)/r", (("c", "positive"), ("lam", "positive"))),
    "inv_r":          ("c*p1/r", (("c", "positive"),)),
    "power":          ("c*p1*r**(-a)", (("c", "positive"), ("a", "signed"))),
    "inv_r_plus_c":   ("c*p1/r + b*p1", (("c", "positive"), ("b", "positive"))),
    "logr":           ("c*p1*log(1+k*r)/r", (("c", "positive"), ("k", "positive"))),
    "inv_r_p1sq":     ("c*p1**2/r", (("c", "positive"),)),
    "inv_r_nomass":   ("c*p1*p2/r", (("c", "positive"),)),
    "inv_r_bg":       ("c*p1/r - b*p1*r", (("c", "positive"), ("b", "positive"))),
    "prod_coupling":  ("c*p1*p2**2/r**k", (("c", "positive"), ("k", "positive"))),
    "crossover":      ("c1*p1/r + c2*p1*exp(-r/R)/r**2",
                       (("c1", "positive"), ("c2", "positive"), ("R", "positive"))),
}
# Catalogue structures that cannot be written in this grammar, and why.
CATALOG_NOT_EXPRESSIBLE = {
    "inv_r_tmod": "cos/sin are not in the vendored CALL_ARITY (exp·log·sqrt·tanh·square·power)",
}
# Catalogue structures that can be transcribed but have a **different prior scale**.  The
# free-form role vocabulary has only positive/signed, so it cannot express the scaled kinds
# (pos_bg, pos_w): the force law (= the function) is the same, but the prior scale of the free
# constants differs, so it is a **different model**.  It is still used for duplicate judgement
# (there is no reason to set up again as a free-form expression a law the catalogue already
# holds).  Measured: the force values of the two implementations agree with relative error 0.0.
CATALOG_PRIOR_SCALE_DIFFERENCES = {
    "inv_r_bg": ("catalogue kinds = ('pos', 'pos_bg'); pos_bg is LogNormal(log b0, 1), "
                 "b0 = 1/(2 pi R^2) (fb1_core.B0_BACKGROUND, derived from the box half-width R = 10). "
                 "The free-form transcription makes both constants plain positive = LogNormal(0, 1)."),
}


class GrammarError(ValueError):
    """A free-form expression failed the grammar, role or cap checks.  The message is recorded and passed on verbatim."""


# ---------------------------------------------------------------- operator count
def count_operators(expression):
    """Operator count of an expression.  **Same rule** as the counting in the vendor ``validate_expression``.

    BinOp(+ - * / **) 1 · UnaryOp(+x, -x) 1 · call 1.  Names and numbers count 0.
    The vendor rejects only counts above 15 (the cap of the count task), so to tighten the cap to
    the catalogue maximum we must hold a count with the same rule ourselves.

    Note (convention): the catalogue ``Structure.n_ops`` is a **hand-declared** complexity value
    and is not exactly this rule.  Measured comparison (test ``test_catalog_operator_counts_are_recorded``):

        inv_r_plus_c declared 5 / this rule 4    logr declared 7 / 6
        crossover    declared 8 / this rule 10   the other 10 agree

    So the cap is set **not to the maximum declared value (8) but to the catalogue maximum
    measured with the same ruler (10)** (``N_OPS_MAX``).  Mixing rulers changes the meaning of the
    cap: imposing the declared 8 on free-form expressions measured with this rule would reject,
    as "over the catalogue maximum", a free-form proposal of the same law as ``crossover``, which
    is **already in** the catalogue (10 by this rule) - the exact opposite of what the cap was meant
    to say ("no more complex than the most complex catalogue law").  Both values are written into
    ``grammar_identity()`` and kept in the output.
    """
    tree = ast.parse(expression, mode="eval")
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Call)):
            n += 1
    return n


# The cap actually enforced: every transcribable catalogue law is measured **with the same ruler
# as a free-form expression** (count_operators) and the maximum is used.  Measured 10 (= crossover).
# The maximum declared value is N_OPS_MAX_DECLARED (= 8).
N_OPS_MAX = max(count_operators(expression)
                for expression, _ in CATALOG_GRAMMAR_EXPRESSIONS.values())


# ---------------------------------------------------------------- role -> kind
def roles_to_kinds(parameters):
    """``{name: role}`` -> ``(kind, ...)``.  GrammarError for a role outside the allowed set.

    The order is the insertion order of ``parameters``; that order is the j of ``th[:, j]``.
    """
    items = tuple(dict(parameters).items()) if not isinstance(parameters, (list, tuple)) \
        else tuple(parameters)
    kinds = []
    for name, role in items:
        if role not in ROLE_TO_KIND:
            raise GrammarError(
                f"coefficient {name} has role {role!r}; ForceBench free-form structures accept "
                f"only {' or '.join(ALLOWED_ROLES)}")
        kinds.append(ROLE_TO_KIND[role])
    return tuple(kinds)


# ---------------------------------------------------------------- force function
def make_force(expression, coefficient_names):
    """Validated expression -> ``force(r, p1, p2, th, t)``.  Same signature as the fb1_forward functions.

    ``th[:, j]`` is the value of ``coefficient_names[j]``.  The vendor ``evaluate`` takes ``u``
    as the broadcasting anchor and adds ``+ 0*u`` to the result; here the anchor is ``r``, so the
    returned shape always equals that of r (even when the expression does not use r, even for a
    constant expression).  It plugs straight into the ``f = force(r, p1b, p2b, theta, t)`` slot of
    fb1_forward.integrate.
    """
    names = tuple(coefficient_names)

    def force(r, p1, p2, th, t):
        """Evaluate the free-form force on (D, P) r with design inputs p1, p2, coefficient draws th and time t."""
        parameters = {name: th[:, j] for j, name in enumerate(names)}
        inputs = {"r": r, "p1": p1, "p2": p2, "t": t}
        return vendor.evaluate(expression, parameters, r, backend=jnp, inputs=inputs)

    force.__name__ = "free_force"
    force.__doc__ = f"free-form force law {expression!r} (free constants {names})"
    return force


# ---------------------------------------------------------------- canonical identifier and duplicates
def structure_model_id(expression, parameters):
    """Canonical identifier of the structure alone (sha256[:20]).  The family is not included (STRUCTURE_ID_FAMILY)."""
    return vendor.canonical_model_id(expression, STRUCTURE_ID_FAMILY, tuple(dict(parameters).items()))


def structure_name(model_id):
    """Structure name for a canonical identifier: NAME_PREFIX plus its first NAME_ID_CHARS characters."""
    return NAME_PREFIX + model_id[:NAME_ID_CHARS]


def catalog_canonical_ids():
    """{canonical identifier: catalogue structure name}.  The 13 transcribable ones.

    This table is a **reference table**, not a rejection list.  If a free-form
    expression's canonical identifier is here, the expression is the same as that catalogue law
    and is **linked** to that name.  Actual rejection happens only when (that structure identity,
    family) is already in the active pool or the evaluated history, and that judgement is made by
    the caller that knows the pool and the history (run_gravity).  The first version
    rejected an expression as `duplicate_by_canonical_id` just for being in this table, which
    excluded from the search altogether a correct candidate (`c*p1/r` = inv_r) that was neither in
    the initial pool nor in the history.
    """
    table = {}
    for name, (expression, parameters) in CATALOG_GRAMMAR_EXPRESSIONS.items():
        mid = structure_model_id(expression, parameters)
        if mid in table:
            raise GrammarError(f"the catalogue transcription is not one-to-one: {name} and {table[mid]}")
        table[mid] = name
    return table


CATALOG_CANONICAL_IDS = catalog_canonical_ids()


def catalog_prior_match():
    """{catalogue name: does the prior of this grammar transcription equal the prior of the catalogue structure}.

    If equal, the transcription is the **same Bayesian model** as the catalogue entry, so a
    free-form expression can be linked to that name.  If not, it is a different model
    even with the same force function and is **never merged**; the only measured exception is
    `inv_r_bg` (catalogue kinds = ('pos','pos_bg'), transcription = ('pos','pos'); the basis is
    written in `CATALOG_PRIOR_SCALE_DIFFERENCES`).  The first version rejected
    that free-form variant as a catalogue duplicate, which did not "preserve a different prior
    model" but **excluded** the variant.  The current rule chooses not to merge: the variant is
    registered as its own structure and the scale difference is recorded.
    """
    return {name: roles_to_kinds(parameters) == tuple(STRUCTURES[name].kinds)
            for name, (_expression, parameters) in CATALOG_GRAMMAR_EXPRESSIONS.items()}


CATALOG_PRIOR_MATCH = catalog_prior_match()
# canonical identifier -> catalogue name, only those whose priors also match.  A free-form expression found here is a reference link.
CATALOG_SAME_PRIOR_IDS = {mid: name for mid, name in CATALOG_CANONICAL_IDS.items()
                          if CATALOG_PRIOR_MATCH[name]}
# Catalogue names with a different prior scale.  Neither linked nor rejected; only the difference is recorded.
CATALOG_DIFFERENT_PRIOR_IDS = {mid: name for mid, name in CATALOG_CANONICAL_IDS.items()
                               if not CATALOG_PRIOR_MATCH[name]}

# Free structures registered at runtime.  {structure name: record dict}.  This list goes into the output JSON.
REGISTRY = {}
# {canonical identifier: structure name}: catches as a duplicate the same canonical expression arriving twice.
REGISTRY_BY_ID = {}


def duplicate_of(expression, parameters):
    """Name of the catalogue/registered structure whose **canonical identifier equals** this expression's.  None if none.  (stage 1)

    Note: this function is only a **lookup**, not a judgement.  A returned
    catalogue name is the reference fact "the same law is in the catalogue" and is not by itself
    a rejection reason (rejection only when it equals a registered free structure).  The judgement
    is split three ways by `judge_free`: registered free structure = duplicate · catalogue entry
    with the same priors = reference link · catalogue entry with a different prior scale = not merged.

    The vendor canonicalisation serialises the canonical AST as it is, so it **does not normalise
    commutativity**: ``c*p1/r**2`` and ``p1*c/r**2`` are the same function but have different
    canonical identifiers.  This function therefore misses the latter, and ``canonical_model_id``
    is not touched to fix that (it is an identity shared with CORE).  Instead a
    **stage 2** numerical-equivalence check (`numerically_equivalent`) is run before admission, and
    that check catches commutative variants.
    """
    mid = structure_model_id(expression, parameters)
    return CATALOG_CANONICAL_IDS.get(mid) or REGISTRY_BY_ID.get(mid)


# ---------------------------------------------------------------- validation and compilation
def compile_structure(expression, parameters):
    """Validate one free-form expression and turn it into an (unregistered) ``Structure`` and a record dict.

    Check order (a failure raises GrammarError on the spot; the message is recorded as the rejection reason):
      1. are the roles only positive/signed
      2. vendor CORE grammar (allowed tokens, names, literals, whether every coefficient is used)
      3. free constants <= MAX_FREE_CONSTANTS
      4. operators <= N_OPS_MAX (catalogue maximum)

    The finiteness probe is not run here: keeping compilation and probing apart lets the judging
    side record the reasons separately (grammar rejection vs probe rejection).
    """
    if not isinstance(expression, str) or not expression.strip():
        raise GrammarError("expression must be a non-empty string")
    items = tuple(dict(parameters).items())
    if len(items) != len(dict(parameters)):
        raise GrammarError("coefficient names must be distinct")
    kinds = roles_to_kinds(items)                                    # 1
    try:
        vendor.validate_expression(expression, items, INPUT_NAMES)   # 2
    except SyntaxError as exc:
        raise GrammarError(f"expression does not parse: {exc}") from exc
    except (ValueError, TypeError) as exc:
        raise GrammarError(str(exc)) from exc
    if len(items) > MAX_FREE_CONSTANTS:                              # 3
        raise GrammarError(
            f"at most {MAX_FREE_CONSTANTS} free constants in a free-form force law, got {len(items)}")
    n_ops = count_operators(expression)                              # 4
    if n_ops > N_OPS_MAX:
        raise GrammarError(
            f"expression uses {n_ops} operators; the catalogue maximum is {N_OPS_MAX}")
    names = tuple(name for name, _ in items)
    model_id = structure_model_id(expression, items)
    name = structure_name(model_id)
    struct = Structure(name=name, expr=expression, force=make_force(expression, names),
                       kinds=kinds, n_coeff=len(kinds), n_ops=n_ops)
    record = {"structure": name, "model_id": model_id, "expression": expression,
              "parameters": {n: r for n, r in items}, "kinds": list(kinds),
              "coefficient_names": list(names), "n_coeff": len(kinds), "n_ops": n_ops,
              "input_names": list(INPUT_NAMES)}
    return struct, record


def register(struct, record, *, round=None):
    """Register in ``fb1_core.STRUCTURES``.  **Duplicate registration is forbidden**.

    If the same name (= the same canonical identifier) is already in STRUCTURES it is a
    GrammarError: silently overwriting would change the force function of an already fitted
    candidate under its feet.
    """
    name = struct.name
    if name in STRUCTURES:
        raise GrammarError(
            f"structure {name} is already registered (canonical id {record['model_id']}); "
            "duplicate registration is forbidden")
    if name in CATALOG_STRUCTURE_NAMES:
        raise GrammarError(f"structure name {name} collides with a catalogue name")
    STRUCTURES[name] = struct
    REGISTRY[name] = dict(record, registered_round=round)
    REGISTRY_BY_ID[record["model_id"]] = name
    return name


def coefficient_names(name):
    """Free-constant names of a registered free structure, in order (the j order of ``th[:, j]``)."""
    return tuple(REGISTRY[name]["coefficient_names"])


def is_free(name):
    """True when ``name`` is a free structure registered at runtime (not a catalogue structure)."""
    return name in REGISTRY


def registry_records():
    """The registration list written into the output JSON, sorted by structure name."""
    return [REGISTRY[name] for name in sorted(REGISTRY)]


def module_sha256():
    """sha256 of this file itself.

    It records in the output **which grammar module** produced the registration list.
    `run_gravity._code_records()` hashes only four files (runner, fb1_core, fb1_forward,
    world_constants), and adding a key there would change the bytes of catalogue-mode output,
    which must stay byte-identical.  So the module's own hash is written in this block, which is
    attached only to free-form runs.
    """
    with open(os.path.abspath(__file__), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def grammar_identity(times=None):
    """Grammar identity written into the output JSON: which engine, which caps and which grid the run used."""
    return {
        "grammar": "bmd CORE named-input expression grammar",
        "fb1_grammar_sha256": module_sha256(),
        "vendor_file": "vendor/count_models_v2_9.py",
        "vendor_sha256": VENDOR_SHA256,
        "vendor_chain": VENDOR_CHAIN,
        "vendor_source": VENDOR_SOURCE,
        "input_names": list(INPUT_NAMES),
        "calls": sorted(vendor.CALL_ARITY),
        "roles": {role: kind for role, kind in sorted(ROLE_TO_KIND.items())},
        "max_free_constants": MAX_FREE_CONSTANTS,
        "n_ops_max": N_OPS_MAX,
        "n_ops_max_basis": (
            "max operator count over the catalogue's own expressions, measured with the SAME "
            "rule that measures a free expression (count_operators); frozen at import"),
        "n_ops_max_declared": N_OPS_MAX_DECLARED,
        "n_ops_max_declared_basis": (
            "max(Structure.n_ops) over the fb1_core catalogue - hand-declared complexity, a "
            "different convention (crossover declares 8 but counts 10); recorded, not enforced"),
        "structure_id_family_token": STRUCTURE_ID_FAMILY,
        "catalogue_expressible": sorted(CATALOG_GRAMMAR_EXPRESSIONS),
        "catalogue_not_expressible": dict(CATALOG_NOT_EXPRESSIBLE),
        "catalogue_prior_scale_differences": dict(CATALOG_PRIOR_SCALE_DIFFERENCES),
        "catalogue_expressible_count": (
            f"{len(CATALOG_GRAMMAR_EXPRESSIONS)}/"
            f"{len(CATALOG_GRAMMAR_EXPRESSIONS) + len(CATALOG_NOT_EXPRESSIBLE)}"),
        "duplicate_detection": {
            "stage_0_finiteness_first": (
                "The finiteness probe runs BEFORE any equivalence comparison. If "
                "the candidate has even one non-finite probe point it is rejected as " +
                REASON_NONFINITE + " and its grid is never used as evidence of equivalence "
                "(counterexample c*p1*(t-2)/(t-2), non-finite at the measurement time t=2, used "
                "to be recorded as a duplicate). A target's non-finite points disqualify that "
                "one comparison in the same way"),
            "stage_1_canonical": (
                "canonical_model_id over the canonical AST + coefficient roles (shared with "
                "CORE, left untouched); commutative re-orderings are NOT normalised, "
                "so p1*c/r**2 and c*p1/r**2 get different canonical ids"),
            "stage_2_numerical": (
                "ForceBench only: before admission "
                "the candidate force is compared with every registered free structure and the "
                "catalogue transcriptions on the probe grid, trying EVERY PRIOR-PRESERVING "
                "coefficient permutation (at most 6, since a free law has at most 3 "
                "coefficients) instead of the single first-occurrence pairing, and requiring "
                "the paired coefficients to carry the same prior ONE BY ONE (the same role "
                f"multiset is not sufficient); relative difference < {EQUIVALENCE_RTOL:g} "
                f"(absolute < {EQUIVALENCE_ATOL:g} where |F| is ~0) means the two forces are " +
                PROBE_VERDICT + ". A target whose force matches but for which no prior-preserving "
                "pairing exists is NOT rejected - it is recorded as " +
                SAME_FUNCTION_DIFFERENT_ROLES_KEY + " and stays admissible"),
            "stage_2_verdict": (
                "A match is worded as " + PROBE_VERDICT + ". " + PROBE_VERDICT_NOTE),
            "catalogue_is_a_reference_table": (
                "Matching a catalogue law is a LINK to that catalogue name, not a "
                "rejection. Only an already registered FREE structure yields " +
                REASON_DUPLICATE_NUMERICAL + "/" + REASON_DUPLICATE_CANONICAL + "; a catalogue "
                "match with the same priors yields " + REASON_CATALOG_REFERENCE + " and the "
                "caller decides by whether (that structure identity INCLUDING its priors, "
                "family) is already in the active pool or the evaluated history - an unevaluated "
                "pair proceeds as a candidate and is NOT counted as a newly discovered "
                "structure. A catalogue match whose PRIORS DIFFER (inv_r_bg) is never merged"),
            "limitation": (
                "coefficients are paired by count and by prior-preserving permutation, not by "
                "value, so the same law written with a DIFFERENT coefficient count (e.g. a "
                "redundant constant that could be absorbed into an existing one) is NOT "
                "detected; neither is a general reparameterisation (a*r and -a*r with a signed "
                "prior have the same prior predictive distribution and both register)"),
            "reason_codes": {
                "canonical": REASON_DUPLICATE_CANONICAL,
                "numerical": REASON_DUPLICATE_NUMERICAL,
                "catalog_reference": REASON_CATALOG_REFERENCE,
            },
            "catalogue_prior_match": dict(CATALOG_PRIOR_MATCH),
            "same_function_different_roles_key": SAME_FUNCTION_DIFFERENT_ROLES_KEY,
            "rtol": EQUIVALENCE_RTOL, "atol": EQUIVALENCE_ATOL,
        },
        "probe": probe_identity(times),
    }


# ---------------------------------------------------------------- finiteness probe
def probe_identity(times=None):
    """Declared probe-grid settings written into the output JSON; time counts are None when ``times`` is None."""
    return {"r_range": list(PROBE_R_RANGE), "r_points": PROBE_R_POINTS,
            "prior_draws": PROBE_DRAWS, "p_range": list(PROBE_P_RANGE),
            "p_corners": [list(c) for c in probe_corners()],
            # The interior grid is declared in advance and recorded too.
            "p_interior_fractions": list(PROBE_P_INTERIOR_FRACTIONS),
            "p_interior": [list(c) for c in probe_interior()],
            "p_points": [list(c) for c in probe_points()],
            "n_p_points": len(probe_points()),
            "times": None if times is None else [float(t) for t in times],
            "n_times": None if times is None else len(list(times)),
            "n_points": None if times is None
            else PROBE_R_POINTS * len(list(times)) * PROBE_DRAWS * len(probe_points()),
            "abs_max": PROBE_ABS_MAX, "seed": PROBE_SEED,
            "rule": ("r grid x ALL measurement times x prior draws x "
                     "(p1,p2) points = the 4 declared box CORNERS plus a pre-declared "
                     "INTERIOR grid (the box quartiles, 3x3); any non-finite value rejects the "
                     "candidate, and so does |F| > abs_max"),
            "interior_basis": ("the corners alone cannot separate a law whose extra "
                               "term vanishes only at the p-endpoints from the law without it "
                               "(counterexample c*p1+c*(p2-0.1)*(p2-10) versus c*p1); the "
                               "runner's design box contains the whole interior"),
            "verdict": PROBE_VERDICT,
            "verdict_note": PROBE_VERDICT_NOTE,
            "note": ("prior draws use a dedicated key folded with the canonical id; the run's "
                     "numpy stream, SMC keys and design stream are never consumed"),
            "reason_codes": {"nonfinite": REASON_NONFINITE, "over_cap": REASON_ABS_MAX}}


def probe_corners():
    """The 4 **corner** combinations of the (p1,p2) box (the original grid)."""
    lo, hi = PROBE_P_RANGE
    return ((lo, lo), (lo, hi), (hi, lo), (hi, hi))


def probe_interior():
    """Fixed grid **inside** the (p1,p2) box.  Quantiles 3x3 = 9 points.

    The coordinates follow deterministically from the box width: p = lo + f (hi - lo), f in
    ``PROBE_P_INTERIOR_FRACTIONS``.  For the (0.1, 10) box they are {2.575, 5.05, 7.525}.
    """
    lo, hi = PROBE_P_RANGE
    values = tuple(lo + f * (hi - lo) for f in PROBE_P_INTERIOR_FRACTIONS)
    return tuple((p1, p2) for p1 in values for p2 in values)


def probe_points():
    """All (p1,p2) points the probe actually uses = 4 corners + 9 interior = 13.  The order is the recording order."""
    return probe_corners() + probe_interior()


def probe_draws(struct, model_id):
    """Prior draws (P, k) shared by the probe and the equivalence check.  Drawn only from the dedicated key."""
    kinds = tuple(struct.kinds)
    key = jax.random.fold_in(jax.random.PRNGKey(PROBE_SEED), int(model_id[:8], 16))
    u = jax.random.normal(key, (PROBE_DRAWS, len(kinds)))
    return core.u_to_x(u, kinds)


def _probe_axes(times):
    """Lay the (r, t) grid out in the (D, P) / (D, 1) convention of the force functions.

    D = PROBE_R_POINTS x len(times): **every combination** of the r grid and the measurement times goes on the design axis.
    """
    times = np.asarray([float(t) for t in times], dtype=np.float64)
    if times.size == 0:
        raise GrammarError("probe grid needs at least one measurement time")
    r_grid = np.linspace(PROBE_R_RANGE[0], PROBE_R_RANGE[1], PROBE_R_POINTS)
    rr, tt = np.meshgrid(r_grid, times, indexing="ij")           # (R, T) each
    d = rr.size
    r = jnp.broadcast_to(jnp.asarray(rr.reshape(d, 1)), (d, PROBE_DRAWS))    # (D, P)
    t = jnp.asarray(tt.reshape(d, 1))                                        # (D, 1)
    return r, t, d, times


def _force_on_grid(struct, x, times):
    """Measure the force over the whole probe grid.  Returns a (D, P, len(probe_points())) float64 numpy array.

    The last axis is the 13 (p1,p2) points, 4 corners + 9 interior.
    """
    r, t, d, _ = _probe_axes(times)
    out = []
    for p1v, p2v in probe_points():
        p1 = jnp.full((d, 1), float(p1v))
        p2 = jnp.full((d, 1), float(p2v))
        f = np.asarray(struct.force(r, p1, p2, x, t), dtype=np.float64)
        out.append(np.broadcast_to(f, (d, PROBE_DRAWS)))
    return np.stack(out, axis=-1)


def finiteness_probe(struct, model_id, times):
    """Prior finiteness probe.  Returns a record dict (``finite`` is the verdict, ``reason_code`` the reason).

    Grid = 32 r points x all measurement times x 64 prior draws x 13 (p1,p2) points (4 corners +
    9 interior).  Runs **before** fitting.
    """
    x = probe_draws(struct, model_id)
    f = _force_on_grid(struct, x, times)
    finite_mask = np.isfinite(f)
    n_total = int(f.size)
    n_nonfinite = int((~finite_mask).sum())
    max_abs = float(np.max(np.abs(f[finite_mask]))) if finite_mask.any() else float("inf")
    over = bool(max_abs > PROBE_ABS_MAX) or not math.isfinite(max_abs)
    finite = n_nonfinite == 0 and not over
    _corner_set = set(probe_corners())
    per_corner = [
        {"p1": float(p1v), "p2": float(p2v),
         "region": "corner" if (p1v, p2v) in _corner_set else "interior",
         "nonfinite": int((~finite_mask[..., i]).sum()),
         "max_abs": (float(np.max(np.abs(f[..., i][finite_mask[..., i]])))
                     if finite_mask[..., i].any() else float("inf"))}
        for i, (p1v, p2v) in enumerate(probe_points())
    ]
    if finite:
        code, reason = None, None
    elif n_nonfinite:
        code = REASON_NONFINITE
        reason = (f"{REASON_NONFINITE}: non-finite force at {n_nonfinite}/{n_total} probe points "
                  f"(r in {list(PROBE_R_RANGE)} x {len(list(times))} measurement times x "
                  f"{PROBE_DRAWS} prior draws x {len(per_corner)} (p1,p2) points: "
                  f"{len(probe_corners())} corners + {len(probe_interior())} interior)")
    else:
        code = REASON_ABS_MAX
        reason = (f"{REASON_ABS_MAX}: force magnitude {max_abs:.6g} exceeds the probe cap "
                  f"{PROBE_ABS_MAX:g}")
    return {**probe_identity(times),
            "nonfinite_points": n_nonfinite,
            "nonfinite_fraction": n_nonfinite / n_total if n_total else 1.0,
            "max_abs_force": max_abs, "exceeds_abs_max": over,
            "finite": bool(finite), "reason_code": code, "reason": reason,
            "per_corner": per_corner}


# -------------------------------------------------- numerical-equivalence duplicate check
def _canonical_positions(expression, coefficient_names):
    """Coefficient name -> canonical position (order of first occurrence, from 0).

    Obtained by **exactly the same traversal** as the alias-assignment rule of the vendor
    ``canonical_model_id`` (``vendor/count_models_v2_9.py:302-316``; that file is not touched at
    all, the same traversal is reproduced locally here to obtain only the position table):
    Expression -> body · BinOp -> left then right · UnaryOp -> operand · Call -> args in order ·
    a Name that is a coefficient name gets the next integer **only the first time it is seen**
    (same as the vendor's ``aliases.setdefault(node.id, len(aliases))``).  Input names (r,p1,p2,t)
    and literals do not consume a position, just as in the vendor they are emitted separately as
    ``["input", ...]``/``["number", ...]``.  ``validate_expression`` already rejects an "unused
    coefficient", so in an expression that passed ``compile_structure`` every declared coefficient
    gets exactly one position (a permutation of 0..n-1).
    """
    names = set(coefficient_names)
    positions = {}

    def walk(node):
        """Visit ``node`` in canonical order and assign each coefficient name its first-occurrence position."""
        if isinstance(node, ast.Expression):
            walk(node.body)
        elif isinstance(node, ast.Name):
            if node.id in names:
                positions.setdefault(node.id, len(positions))
        elif isinstance(node, ast.BinOp):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, ast.UnaryOp):
            walk(node.operand)
        elif isinstance(node, ast.Constant):
            pass
        else:                                    # ast.Call -- the only other node type left in the grammar
            for arg in node.args:
                walk(arg)

    walk(ast.parse(expression, mode="eval"))
    return positions


def _pair_by_canonical_position(values, from_expression, from_names, to_expression, to_names):
    """Pair coefficient columns by canonical position: matched by **position**, not by name.

    It is **not used for duplicate judgement.**  This pairing always matches
    the first-occurrence positions of the two expressions with each other, so it produces wrong
    pairs for multi-coefficient terms whose summation order is reversed (which is why
    `_prior_preserving_pairings` looks at all permutations instead).  One use remains: the
    comparison **for the annotation only** of a pair with no prior-preserving pairing
    (= a pair whose role multisets differ), `same_function_different_roles_of`; such a pair is
    never used for rejection anyway.

    Coefficients with the same first-occurrence position, as found by running
    ``_canonical_positions`` over each expression, are paired.  This is the same position table as
    the alpha order canonical_model_id assigns, so pairing works even when the name sets differ
    (names outside the catalogue) and even when both the names and the commutative order change
    (``p1*k/r**2``).  If the coefficient **counts** differ, positions cannot be matched 1:1 at all
    and the result is None; this is the code basis of a disclosed limitation (the same
    function whose count changed by absorbing a constant cannot be detected this way).
    """
    if len(from_names) != len(to_names):
        return None
    from_pos = _canonical_positions(from_expression, from_names)
    to_pos = _canonical_positions(to_expression, to_names)
    n = len(from_names)
    if sorted(from_pos.values()) != list(range(n)) or sorted(to_pos.values()) != list(range(n)):
        return None    # defensive only: never happens for an expression that passed validate_expression
    from_index = {name: j for j, name in enumerate(from_names)}
    from_name_at_position = {p: name for name, p in from_pos.items()}
    return jnp.stack(
        [values[:, from_index[from_name_at_position[to_pos[name]]]] for name in to_names],
        axis=1)


def _permutations(n):
    """All permutations of ``0..n-1`` in lexicographic order.  ``n <= MAX_FREE_CONSTANTS = 3``, so at most 6.

    `itertools` is not used: the import set of this module is pinned by a test
    (`test_no_paid_transport_module_is_imported_by_the_grammar`).
    """
    if n <= 0:
        return [()]
    out = []
    for i in range(n):
        for rest in _permutations(n - 1):
            out.append((i,) + tuple(j + (1 if j >= i else 0) for j in rest))
    return out


def _prior_preserving_pairings(from_names, from_kinds, to_names, to_kinds):
    """All **prior-preserving bijections** from candidate coefficients to target coefficients.

    Returns ``[order, ...]``.  Each ``order`` is a tuple of length n and means
    ``theta[:, j] = values[:, order[j]]``, i.e. the draws of the candidate's ``order[j]``-th
    coefficient go into the target's j-th coefficient slot.  Two conditions are enforced together:

      **Every possible pairing** is looked at, not just the first-occurrence position.  There
      are at most 3 free coefficients, so at most 6 permutations per pair.  The
      `_pair_by_canonical_position` of the first pass always matched the first coefficient
      of each expression with each other, so for `b*p1+c*p1/r`, with the summation order reversed,
      it paired the constant-term coefficient with the inverse-distance-term coefficient (and the
      expression passed as active although it is the same law as `c*p1/r+b*p1`).  Agreement under
      alpha renaming is not a canonicalisation with respect to commutation and re-association.

      **Prior preservation**, ``from_kinds[order[j]] == to_kinds[j]``: the prior (role) must be preserved **for
      every paired coefficient**.  Merely sharing the role multiset is not sufficient: a model of
      `a*r+b` with (a:positive, b:signed) and one with (a:signed, b:positive) have the same
      multiset, but they are **different models** whose priors on slope and intercept are swapped
      (prior mean force at r=2 is 2 exp(1/2) versus exp(1/2)).  For that pair only the pairing that
      swaps positions preserves the prior, and under that pairing the two forces differ, so it is
      not a duplicate.

    With no prior-preserving pairing at all the result is ``[]``: that pair drops out of the
    duplicate comparison and only the annotation (`same_function_different_roles_of`) remains.
    """
    n = len(from_names)
    if n != len(to_names) or sorted(from_kinds) != sorted(to_kinds):
        return []
    return [order for order in _permutations(n)
            if all(from_kinds[order[j]] == to_kinds[j] for j in range(n))]


def _pair_values(values, order):
    """``theta[:, j] = values[:, order[j]]`` (the order from ``_prior_preserving_pairings``)."""
    return jnp.stack([values[:, order[j]] for j in range(len(order))], axis=1)


def _role_multiset(struct):
    """``Structure.kinds`` -> sorted role multiset, e.g. ``("positive", "positive")``.

    This is the "role multiset": names and order do not matter, only count and kind.
    """
    return tuple(sorted(KIND_TO_ROLE[kind] for kind in struct.kinds))


def comparison_targets():
    """Comparison targets of the numerical-equivalence check: registered free structures + the 13 catalogue transcriptions.

    Returns ``[(label, Structure, coefficient-name tuple, kind)]``.  Catalogue transcriptions are
    compiled on the fly (a transcription is itself an expression in this grammar, so compilation
    always succeeds; a test checks this).

    **The kind decides the verdict**; the catalogue transcriptions are not all
    used as the same kind of rejection target.

      ``TARGET_FREE_REGISTERED``                   an already registered free structure -> a real duplicate.
      ``TARGET_CATALOG``                           a catalogue entry with the same priors -> **reference link**
                                                   (rejection is decided by the caller, who knows the pool and the history).
      ``TARGET_CATALOG_PRIOR_SCALE_DIFFERENCE``    a catalogue entry with a different prior scale (inv_r_bg) ->
                                                   a different model even for the same function, so not merged.
    """
    targets = []
    for name in sorted(REGISTRY):
        targets.append((name, STRUCTURES[name], tuple(REGISTRY[name]["coefficient_names"]),
                        TARGET_FREE_REGISTERED))
    for name in sorted(CATALOG_GRAMMAR_EXPRESSIONS):
        expression, parameters = CATALOG_GRAMMAR_EXPRESSIONS[name]
        struct, record = compile_structure(expression, parameters)
        kind = TARGET_CATALOG if CATALOG_PRIOR_MATCH[name] \
            else TARGET_CATALOG_PRIOR_SCALE_DIFFERENCE
        targets.append((name, struct, tuple(record["coefficient_names"]), kind))
    return targets


def _grid_difference(mine, finite_mine, theirs):
    """(max relative difference, max absolute difference, number of shared finite points, **number of non-finite points**) of two force grids.

    With no shared finite point the result is ``(None, None, 0, number of non-finite points)``;
    the caller separates "not comparable" from "compared but different".

    The fourth value is new.  As before, this function keeps only the points
    finite on both sides, but the caller **does not use agreement on that subset as evidence of
    equivalence over the whole grid if there is even one non-finite point**.  The candidate's own
    non-finite points are rejected by `judge_free` before this stage (order corrected), so any
    non-finite points left here are on the target side.
    """
    finite_theirs = np.isfinite(theirs)
    both = finite_mine & finite_theirs
    n_nonfinite = int((~(finite_mine & finite_theirs)).sum())
    if not both.any():
        return None, None, 0, n_nonfinite
    a, b = mine[both], theirs[both]
    scale = np.maximum(np.abs(a), np.abs(b))
    near_zero = scale <= EQUIVALENCE_NEAR_ZERO
    diff = np.abs(a - b)
    rel = np.where(near_zero, 0.0, diff / np.where(near_zero, 1.0, scale))
    absolute = np.where(near_zero, diff, 0.0)
    return float(np.max(rel)), float(np.max(absolute)), int(both.sum()), n_nonfinite


def numerically_equivalent(struct, coefficient_names, model_id, times):
    """Equivalence check **before** admission.  Returns a record dict.

    Pairing rule: when the candidate and a comparison target have the same number of
    coefficients, **every prior-preserving pairing** (`_prior_preserving_pairings`; at most 3 free
    coefficients, so at most 6 permutations per pair) is run and the forces are compared on the
    same probe grid with the same prior draws.  If any one agrees over the whole grid, the two are
    treated as the same law (relative difference < ``EQUIVALENCE_RTOL``, and absolute difference <
    ``EQUIVALENCE_ATOL`` at points where the force is near 0).  The prior must be preserved for
    each paired coefficient, so a pair that only shares the role multiset stays as two different
    models.  A pair with a different count cannot be paired at all and is left in ``skipped``
    (the same function whose count changed by absorbing a constant is undetected, a disclosed
    limitation).

    Verdict: the **kind** of the matched target decides the outcome.
      registered free structure -> ``REASON_DUPLICATE_NUMERICAL`` (rejection).
      catalogue entry with the same priors -> ``REASON_CATALOG_REFERENCE``: not a rejection but a
        **link** to that name.  Whether that (structure, family) pair is in the active pool or the
        evaluated history is checked by the caller.
      catalogue entry with a different prior scale (inv_r_bg) -> not merged.  Only recorded in
        ``prior_scale_differences`` and the check continues (a different prior is a different
        Bayesian model).

    Annotation: a pair with the same count but no prior-preserving pairing (= a pair whose role
    multisets differ, ``role_mismatches``) also has its force measured, but it is never used for
    rejection.  If the forces agree over the whole grid, only the label is kept in
    ``same_function_different_roles_of``.

    Wording: the verdict is "indistinguishable on the probe grid"; the grid is finite and
    declared in advance, so this is not a proof of general functional equivalence
    (``PROBE_VERDICT``).  If the target side has even one non-finite point, agreement on that
    subset is not used as evidence of equivalence over the whole grid (``nonfinite_points``).  The
    candidate's own non-finite points are rejected by `judge_free` **before** this stage.
    """
    x = probe_draws(struct, model_id)
    mine = _force_on_grid(struct, x, times)
    finite_mine = np.isfinite(mine)
    my_kinds = tuple(struct.kinds)
    my_roles = _role_multiset(struct)
    compared, skipped, role_mismatches, prior_scale_differences = [], [], [], []
    same_function_different_roles_of = None
    limitation = (
        "coefficients are paired by trying EVERY PRIOR-PRESERVING permutation (at most 6, "
        "since a free law has at most 3 coefficients): the paired coefficients must carry the "
        "same prior one by one, so the same role MULTISET is not enough. A "
        "match means the two forces are " + PROBE_VERDICT + ", never a proof of general "
        "functional equivalence. The same law written with a DIFFERENT coefficient count (e.g. "
        "a redundant constant that could be absorbed into an existing one) is still NOT "
        "detected")

    def result(matched=None, matched_kind=None, reason_code=None, reason=None):
        """Build the equivalence record dict from the comparisons gathered so far and the given verdict fields."""
        return {"matched": matched, "matched_kind": matched_kind,
                SAME_FUNCTION_DIFFERENT_ROLES_KEY: same_function_different_roles_of,
                "n_compared": len(compared), "n_skipped": len(skipped),
                "role_mismatches": role_mismatches,
                "prior_scale_differences": prior_scale_differences,
                "comparisons": compared, "skipped": skipped,
                "rtol": EQUIVALENCE_RTOL, "atol": EQUIVALENCE_ATOL,
                "verdict": PROBE_VERDICT, "verdict_note": PROBE_VERDICT_NOTE,
                "reason_code": reason_code, "reason": reason,
                "limitation": limitation}

    for label, other, other_names, target_kind in comparison_targets():
        if len(other_names) != len(coefficient_names):
            skipped.append({"target": label, "target_kind": target_kind,
                            "reason": "coefficient count differs",
                            "candidate_coefficient_count": len(coefficient_names),
                            "target_coefficient_count": len(other_names)})
            continue
        their_kinds = tuple(other.kinds)
        orders = _prior_preserving_pairings(coefficient_names, my_kinds, other_names, their_kinds)
        if not orders:
            # Same count but not a single prior-preserving pairing (= the role multisets differ).
            # Left out of the duplicate comparison, only annotated; never used for rejection.
            # The comparison pairing reuses the canonical-position pairing of the first pass (annotation only).
            theta = _pair_by_canonical_position(
                x, struct.expr, coefficient_names, other.expr, other_names)
            if theta is None:
                max_rel, max_abs, n_points, n_nonfinite = None, None, 0, 0
            else:
                max_rel, max_abs, n_points, n_nonfinite = _grid_difference(
                    mine, finite_mine, _force_on_grid(other, theta, times))
            function_equivalent = bool(
                max_rel is not None and max_rel < EQUIVALENCE_RTOL
                and max_abs < EQUIVALENCE_ATOL and n_nonfinite == 0)
            role_mismatches.append({
                "target": label, "target_kind": target_kind,
                "function_equivalent": function_equivalent,
                "candidate_roles": list(my_roles), "target_roles": list(_role_multiset(other)),
                "max_relative_difference": max_rel,
                "max_absolute_difference_near_zero": max_abs, "n_points": n_points,
                "nonfinite_points": n_nonfinite,
                "pairing_basis": "canonical_position_annotation_only"})
            if function_equivalent and same_function_different_roles_of is None:
                same_function_different_roles_of = label
            continue
        best, equivalent_order = None, None
        for order in orders:
            theirs = _force_on_grid(other, _pair_values(x, order), times)
            max_rel, max_abs, n_points, n_nonfinite = _grid_difference(mine, finite_mine, theirs)
            candidate_row = (max_rel, max_abs, n_points, n_nonfinite, order)
            if best is None or (max_rel is not None
                                and (best[0] is None or max_rel < best[0])):
                best = candidate_row
            if (max_rel is not None and max_rel < EQUIVALENCE_RTOL
                    and max_abs < EQUIVALENCE_ATOL and n_nonfinite == 0):
                best, equivalent_order = candidate_row, order
                break
        max_rel, max_abs, n_points, n_nonfinite, order = best
        row = {"target": label, "target_kind": target_kind,
               "equivalent": equivalent_order is not None,
               "pairings_checked": len(orders),
               "pairings_available": [list(o) for o in orders],
               "pairing": {other_names[j]: coefficient_names[order[j]]
                           for j in range(len(order))},
               "max_relative_difference": max_rel,
               "max_absolute_difference_near_zero": max_abs,
               "n_points": n_points, "nonfinite_points": n_nonfinite}
        if max_rel is None:
            row["reason"] = "no jointly finite probe point"
        compared.append(row)
        if equivalent_order is None:
            continue
        head = (f"force is {PROBE_VERDICT} from {label} (max relative difference "
                f"{max_rel:.3g} < {EQUIVALENCE_RTOL:g} over {n_points} probe points); "
                f"{PROBE_VERDICT_NOTE}")
        if target_kind == TARGET_CATALOG_PRIOR_SCALE_DIFFERENCE:
            # A catalogue entry with a different prior scale is **not merged**: even with the
            # same force function, the prior scale of the free constants differs, so it is a
            # different Bayesian model.  Neither rejected nor linked.
            row["merged"] = False
            prior_scale_differences.append({
                "target": label, "basis": "probe_grid", "merged": False,
                "candidate_kinds": list(my_kinds),
                "catalog_kinds": list(STRUCTURES[label].kinds),
                "transcription_kinds": list(their_kinds),
                "note": CATALOG_PRIOR_SCALE_DIFFERENCES.get(label, ""),
                "reason": (f"{label}: {head} -- but the catalogue structure's priors are "
                           f"{list(STRUCTURES[label].kinds)} while this law's are "
                           f"{list(my_kinds)}, so they are DIFFERENT Bayesian models and are "
                           "never merged")})
            continue
        if target_kind == TARGET_CATALOG:
            return result(matched=label, matched_kind=TARGET_CATALOG,
                          reason_code=REASON_CATALOG_REFERENCE,
                          reason=(f"{REASON_CATALOG_REFERENCE}: {head}; the catalogue is a "
                                  "REFERENCE table -- whether this (structure, family) pair is "
                                  "admitted is decided by the active pool and the evaluated "
                                  "history, not here"))
        return result(matched=label, matched_kind=TARGET_FREE_REGISTERED,
                      reason_code=REASON_DUPLICATE_NUMERICAL,
                      reason=f"{REASON_DUPLICATE_NUMERICAL}: {head}")
    return result()


# ---------------------------------------------------------------- one-step convenience functions
def judge_free(expression, parameters, times):
    """Judge one free-form expression **purely** (does not register it).  Returns (record, reason).

    Order (the finiteness probe runs before the equivalence check):

      1. grammar, roles and caps (`compile_structure`)
      2. **finiteness probe**: if there is even one non-finite point, reject
         on the spot as ``nonfinite_on_probe_grid``.  In the first pass the equivalence
         check ran first, compared only the **subset** finite on both sides and returned duplicate
         first.  So `c*p1*(t-2)/(t-2)`, non-finite at the measurement time t=2, was recorded as a
         duplicate rather than as non-finite; a non-finite grid cannot be evidence of equivalence
         over the whole grid.
      3. canonical identifier (stage 1).  Three ways: equal to a registered free structure
         -> duplicate · catalogue entry with the same priors -> **reference link** · catalogue
         entry with a different prior scale -> not merged.
      4. numerical equivalence (stage 2: every prior-preserving permutation of the
         coefficients is tried)
      5. **force magnitude cap** (the other half of the probe).  Its rejection point was not
         moved: this candidate's grid is already finite everywhere, so the comparisons and
         annotations of step 4 are valid evidence, and to keep that annotation the cap rejection
         must come after it (keeps the annotation behaviour).

    If ``reason`` is None the expression can be admitted.  ``outcome == OUTCOME_CATALOG_REFERENCE``
    is **not** a rejection: the caller decides from whether (structure identity, family) is in the
    active pool or the evaluated history.  A grammar violation propagates as GrammarError (the
    caller writes that sentence down as the rejection reason).
    """
    struct, record = compile_structure(expression, parameters)
    record[CATALOG_REFERENCE_KEY] = None
    record["catalog_link"] = None

    # 2. Run the finiteness probe **first**.
    probe = finiteness_probe(struct, record["model_id"], times)
    record["prior_probe"] = probe
    if probe["nonfinite_points"]:
        record.update(outcome="rejected", reason_code=probe["reason_code"],
                      reason=probe["reason"])
        return record, record["reason"]

    # 3. Stage 1: canonical identifier.
    mid = record["model_id"]
    registered = REGISTRY_BY_ID.get(mid)
    if registered is not None:
        record.update(outcome="duplicate", reason_code=REASON_DUPLICATE_CANONICAL,
                      reason=f"{REASON_DUPLICATE_CANONICAL}: canonical id matches {registered}",
                      matched_structure=registered)
        return record, record["reason"]
    catalog_same = CATALOG_SAME_PRIOR_IDS.get(mid)
    catalog_other_prior = CATALOG_DIFFERENT_PRIOR_IDS.get(mid)
    if catalog_same is not None:
        # The catalogue is a reference table; link by name and let the caller decide on rejection.
        link = {"structure": catalog_same, "basis": "canonical_id", "priors_match": True,
                "counts_as_new_structure": False,
                "decision": ("the caller decides by whether (this structure identity INCLUDING "
                             "its priors, family) already exists in the active pool or the "
                             "evaluated history")}
        record.update(outcome=OUTCOME_CATALOG_REFERENCE,
                      reason_code=REASON_CATALOG_REFERENCE,
                      reason=(f"{REASON_CATALOG_REFERENCE}: canonical id matches the catalogue "
                              f"law {catalog_same} with the same priors; linked to that name, "
                              "not rejected"),
                      matched_structure=catalog_same)
        record[CATALOG_REFERENCE_KEY] = catalog_same
        record["catalog_link"] = link
        return record, record["reason"]
    if catalog_other_prior is not None:
        # With a different prior scale it is a different model even for the same function; not merged, only recorded.
        record["catalog_link"] = {
            "structure": catalog_other_prior, "basis": "canonical_id", "priors_match": False,
            "merged": False, "counts_as_new_structure": True,
            "catalog_kinds": list(STRUCTURES[catalog_other_prior].kinds),
            "candidate_kinds": list(struct.kinds),
            "note": CATALOG_PRIOR_SCALE_DIFFERENCES.get(catalog_other_prior, ""),
            "decision": ("different priors are never merged: this law is registered as its own "
                         "free structure and the scale difference is recorded")}

    # 4. Stage 2: numerical equivalence.
    equivalence = numerically_equivalent(
        struct, tuple(record["coefficient_names"]), record["model_id"], times)
    record["numerical_equivalence"] = equivalence
    # If there is a target with the same function that was not
    # rejected as a duplicate because its role multiset differs, its label is also kept at the top
    # level of the record (it is carried into the REGISTRY entry so that "same law, different
    # prior" can be read directly); it does not affect the outcome verdict itself.
    record[SAME_FUNCTION_DIFFERENT_ROLES_KEY] = equivalence[SAME_FUNCTION_DIFFERENT_ROLES_KEY]
    if equivalence["matched"] is not None:
        if equivalence["matched_kind"] == TARGET_CATALOG:
            record.update(outcome=OUTCOME_CATALOG_REFERENCE,
                          reason_code=REASON_CATALOG_REFERENCE, reason=equivalence["reason"],
                          matched_structure=equivalence["matched"])
            record[CATALOG_REFERENCE_KEY] = equivalence["matched"]
            record["catalog_link"] = {
                "structure": equivalence["matched"], "basis": "probe_grid",
                "priors_match": True, "counts_as_new_structure": False,
                "verdict": PROBE_VERDICT,
                "decision": ("the caller decides by whether (this structure identity INCLUDING "
                             "its priors, family) already exists in the active pool or the "
                             "evaluated history")}
            return record, record["reason"]
        record.update(outcome="duplicate", reason_code=REASON_DUPLICATE_NUMERICAL,
                      reason=equivalence["reason"],
                      matched_structure=equivalence["matched"])
        return record, record["reason"]

    # 5. Force magnitude cap.
    if not probe["finite"]:
        record.update(outcome="rejected", reason_code=probe["reason_code"],
                      reason=probe["reason"])
        return record, record["reason"]
    record.update(outcome="active", reason_code=None, reason=None)
    return record, None


def admit(expression, parameters, times, *, round=None):
    """Judge -> register (when it passes).  Returns (structure name or None, record dict).

    The verdict and the reason are kept in the record dict's ``outcome``/``reason_code``/``reason``.
    The caller (run_gravity) writes that sentence into the archive verbatim; the reason is not
    rewritten outside.

    If ``record["outcome"] == OUTCOME_CATALOG_REFERENCE`` the **catalogue name**
    is returned and nothing is registered as a free structure; it is not counted as "a new
    structure discovered".  Whether that (structure, family) pair is already in the active pool or
    the evaluated history is decided by the caller.

    **A canonical expression that is already registered is not a duplicate.**  A ForceBench
    candidate is a (structure, family) pair, and setting up the same structure with another noise
    family is normal behaviour in catalogue mode too (`(inv_r2, gauss)` and `(inv_r2, student_t4)`
    are separate candidates).  So in that case it is neither registered again nor probed again,
    and the registered structure is returned as it is; whether the (structure, family) pair is a
    duplicate is settled by the caller through the active pool.
    """
    model_id = structure_model_id(expression, parameters)
    name = structure_name(model_id)
    if name in REGISTRY:
        return name, dict(REGISTRY[name], outcome="active", reason_code=None, reason=None,
                          reused_registration=True)
    record, reason = judge_free(expression, parameters, times)
    if record["outcome"] == OUTCOME_CATALOG_REFERENCE:
        # Catalogue reference link, not a rejection.  Nothing is registered; that name is returned.
        return record[CATALOG_REFERENCE_KEY], record
    if reason is not None:
        return None, record
    struct = Structure(name=record["structure"], expr=record["expression"],
                       force=make_force(record["expression"], record["coefficient_names"]),
                       kinds=tuple(record["kinds"]), n_coeff=record["n_coeff"],
                       n_ops=record["n_ops"])
    name = register(struct, record, round=round)
    return name, REGISTRY[name]


def reset_registry():
    """Undo every registration; **tests only**.  Never called on the run path."""
    for name in list(REGISTRY):
        STRUCTURES.pop(name, None)
    REGISTRY.clear()
    REGISTRY_BY_ID.clear()
