"""Added 2026-10-03: cos and sin (arity 1) in the ForceBench free-expression grammar, FB1_GRAMMAR_TRIG=1.

run_gravity.py imports this module and calls `enable()` once, at import and before any structure is compiled, only
when FB1_GRAMMAR_TRIG=1 (which needs FB1_STRUCTURE_PROPOSALS=free).  Unset, nothing imports it and the grammar is the
one every earlier run used.

Why a separate module rather than an edit of fb1_grammar.py
    fb1_grammar.py's own sha256 is in every free-mode run record and in every free-mode belief packet the model reads
    (constraints.grammar.fb1_grammar_sha256, from grammar_identity()), and the vendored grammar file is pinned by its
    sha256 (vendor/PROVENANCE.md).  Editing either would move the bytes of every default packet.  Both files stay
    byte-identical; `enable()` widens, in place, the module state their functions read at call time.

What `enable()` changes - fb1_grammar's private instance of the vendored module (sys.modules name
"fb1_vendor_count_models_v2_9", imported by nothing else) and one table of fb1_grammar:
    vendor.CALL_ARITY       + {"cos": 1, "sin": 1}.  validate_expression accepts the two calls with arity 1 and counts
                            each as one operator, as fb1_grammar.count_operators already counts every call, so the
                            operator cap N_OPS_MAX keeps its rule and its value; vendor.evaluate dispatches a call by
                            name to the backend (numpy.cos for numpy, jax.numpy.cos in the jax force that fitting,
                            prediction, expected information gain (EIG) / expected predictive information gain
                            (EPIG) and the ascent gradient use); canonical_model_id hashes a call by name.
    vendor.RESERVED_NAMES   + ("cos", "sin").  A coefficient may not be named after a call: the vendored rule for the
                            six calls of the vendored grammar, extended to the two new ones.
    fb1_grammar.CATALOG_NOT_EXPRESSIBLE["inv_r_tmod"]: the reason (below).

inv_r_tmod (fb1_core.STRUCTURES, F = c*p1/r * (1 + A1 cos(wt) + A2 sin(wt)), kinds pos, sig, sig, pos_w)
    With cos and sin it has a literal transcription, INV_R_TMOD_TRANSCRIPTION, which validates in the vendored grammar
    and whose force equals fb1_forward.f_inv_r_tmod exactly on the finiteness-probe grid (test_fb1_grammar_trig.py).
    It still is not a free-form law: it has 4 free constants and fb1_grammar.MAX_FREE_CONSTANTS is 3 (not changed
    here), so compile_structure refuses it and no proposal can be it.  It therefore does not join
    CATALOG_GRAMMAR_EXPRESSIONS, whose every entry must compile as a free-form law (that table is the reference of the
    duplicate check and of constraints.initial_structures, and N_OPS_MAX is the maximum over it, so N_OPS_MAX stays
    10); its CATALOG_NOT_EXPRESSIBLE reason now names the constant cap instead of the missing calls.  Its frequency
    prior would differ too: the catalogue's w is pos_w = W0 * LogNormal(0, 1) with W0 = 2 pi / 5
    (fb1_core.W0_MODULATION), a free "positive" coefficient is LogNormal(0, 1) - the kind of difference inv_r_bg's
    background constant already has (fb1_grammar.CATALOG_PRIOR_SCALE_DIFFERENCES).
"""
import hashlib
import os

import fb1_grammar as fbg
from fb1_core import STRUCTURES, Structure, W0_MODULATION

TRIG_CALL_ARITY = {"cos": 1, "sin": 1}
INV_R_TMOD_TRANSCRIPTION = (
    "c*p1/r*(1 + A1*cos(w*t) + A2*sin(w*t))",
    (("c", "positive"), ("A1", "signed"), ("A2", "signed"), ("w", "positive")),
)
# Goes into the packet (constraints.grammar.catalogue_not_expressible), so it names no expression and no coefficient.
INV_R_TMOD_REASON = ("has 4 free constants and a free-form law takes at most 3 (MAX_FREE_CONSTANTS); "
                     "cos and sin are grammar calls under FB1_GRAMMAR_TRIG=1")
OPTION = "FB1_GRAMMAR_TRIG=1"

_SAVED = None   # the original objects while enabled, None while off


def enabled():
    """Return True while cos and sin are enabled in the grammar."""
    return _SAVED is not None


def enable():
    """Add cos and sin to the grammar.  Idempotent; returns True when this call switched it on.

    Must run before any free structure is registered (run_gravity calls it at import), so that every structure of the
    run is judged by one grammar.
    """
    global _SAVED
    if _SAVED is not None:
        return False
    if fbg.REGISTRY:
        raise RuntimeError("fb1_grammar_trig.enable() must run before any free structure is registered")
    vendor = fbg.vendor
    _SAVED = {"CALL_ARITY": vendor.CALL_ARITY, "RESERVED_NAMES": vendor.RESERVED_NAMES,
              "CATALOG_NOT_EXPRESSIBLE": fbg.CATALOG_NOT_EXPRESSIBLE}
    vendor.CALL_ARITY = {**vendor.CALL_ARITY, **TRIG_CALL_ARITY}
    vendor.RESERVED_NAMES = (*vendor.RESERVED_NAMES, *TRIG_CALL_ARITY)
    fbg.CATALOG_NOT_EXPRESSIBLE = {**fbg.CATALOG_NOT_EXPRESSIBLE, "inv_r_tmod": INV_R_TMOD_REASON}
    return True


def disable():
    """Put the original objects back (tests).  Refused while a free structure is registered."""
    global _SAVED
    if _SAVED is None:
        return False
    if fbg.REGISTRY:
        raise RuntimeError("reset the free-structure registry before fb1_grammar_trig.disable()")
    fbg.vendor.CALL_ARITY = _SAVED["CALL_ARITY"]
    fbg.vendor.RESERVED_NAMES = _SAVED["RESERVED_NAMES"]
    fbg.CATALOG_NOT_EXPRESSIBLE = _SAVED["CATALOG_NOT_EXPRESSIBLE"]
    _SAVED = None
    return True


def reference_structure(expression=None, parameters=None):
    """A catalogue transcription as an fb1_core.Structure (default: INV_R_TMOD_TRANSCRIPTION).

    The vendored grammar and the role map only; neither cap that bounds a free-form proposal (MAX_FREE_CONSTANTS,
    N_OPS_MAX) applies, because a transcription is a reference, not a proposal.  Raises like the grammar does.
    """
    if expression is None:
        expression, parameters = INV_R_TMOD_TRANSCRIPTION
    items = tuple(dict(parameters).items())
    kinds = fbg.roles_to_kinds(items)
    fbg.vendor.validate_expression(expression, items, fbg.INPUT_NAMES)
    names = tuple(name for name, _ in items)
    return Structure(name="transcription", expr=expression, force=fbg.make_force(expression, names),
                     kinds=kinds, n_coeff=len(kinds), n_ops=fbg.count_operators(expression))


def module_sha256():
    """Return the sha256 hex digest of this module file's bytes (recorded in the run record)."""
    with open(os.path.abspath(__file__), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def identity():
    """The run record's account of the option (arm_config.structure_proposals.grammar_trig); never sent to the model."""
    expression, parameters = INV_R_TMOD_TRANSCRIPTION
    try:
        transcription = reference_structure()
        validates = True
    except (ValueError, TypeError, SyntaxError):
        transcription, validates = None, False
    try:
        fbg.compile_structure(expression, dict(parameters))
        admissible = True
    except fbg.GrammarError:
        admissible = False
    return {
        "enabled": enabled(), "option": OPTION, "module": "fb1_grammar_trig.py", "module_sha256": module_sha256(),
        "calls_added": dict(TRIG_CALL_ARITY), "calls": sorted(fbg.vendor.CALL_ARITY),
        "reserved_names_added": list(TRIG_CALL_ARITY),
        "unchanged_files": {"fb1_grammar.py": fbg.module_sha256(), "vendor/count_models_v2_9.py": fbg.VENDOR_SHA256},
        "mechanism": ("fb1_grammar.vendor.CALL_ARITY and RESERVED_NAMES (fb1_grammar's private instance of the vendored "
                      "module) gain cos and sin; fb1_grammar.CATALOG_NOT_EXPRESSIBLE['inv_r_tmod'] names the remaining "
                      "obstacle; no file of the grammar is edited"),
        "n_ops_max": fbg.N_OPS_MAX, "n_ops_max_changed": False, "max_free_constants": fbg.MAX_FREE_CONSTANTS,
        "inv_r_tmod": {
            "transcription": expression, "parameters": dict(parameters),
            "n_free_constants": len(parameters), "n_ops": fbg.count_operators(expression),
            "validates_in_the_vendored_grammar": validates, "admissible_as_a_free_form_law": admissible,
            "in_catalogue_grammar_expressions": "inv_r_tmod" in fbg.CATALOG_GRAMMAR_EXPRESSIONS,
            "reason": fbg.CATALOG_NOT_EXPRESSIBLE.get("inv_r_tmod"),
            "catalogue_kinds": list(STRUCTURES["inv_r_tmod"].kinds),
            "transcription_kinds": None if transcription is None else list(transcription.kinds),
            "prior_note": (f"catalogue w is pos_w = {W0_MODULATION!r} * LogNormal(0, 1) (W0 = 2 pi / 5); a free "
                           "'positive' coefficient is LogNormal(0, 1)"),
        },
    }
