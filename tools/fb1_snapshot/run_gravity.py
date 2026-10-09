"""ForceBench FB-1 runner, first full run of the gravity world -- method 111 (structure set S expanded + family set F
expanded + active design A = joint expected information gain (EIG)).

True world = the DiscoverPhysics (DP) SimulationExecutor(field).  Our predict() = jax free-space integration (fb1_forward).
"""
import sys, os, json, math, time, hashlib


# ---- Unified protocol: its options also exist as command-line flags, under the names the four engines
#      share. They only set the environment variables the module reads below, before any of
#      them is read; a flag that contradicts an already set variable is refused. No flag: nothing is set.
CLI_FLAGS = (("--initial-proposals", "FB1_INITIAL_PROPOSALS"), ("--active-maximum", "FB1_ACTIVE_MAXIMUM"),
             ("--proposal-limit", "FB1_PROPOSAL_LIMIT"), ("--archive-review-rounds", "FB1_ARCHIVE_REVIEW_ROUNDS"),
             ("--design-policy", "FB1_DESIGN_POLICY"), ("--packet-archive", "FB1_PACKET_ARCHIVE"),
             ("--catalogue", "FB1_CATALOGUE"))


def _apply_cli_flags(argv):
    import argparse
    parser = argparse.ArgumentParser(prog="run_gravity.py", description="ForceBench FB-1 runner (options are FB1_* variables)")
    for flag, name in CLI_FLAGS:
        parser.add_argument(flag, dest=name)
    for flag, name in CLI_FLAGS:
        value = getattr(parser.parse_args(argv), name)
        if value is None:
            continue
        if name in os.environ and os.environ[name] != value:
            raise SystemExit(f"{flag} {value} contradicts {name}={os.environ[name]}")
        os.environ[name] = value


if __name__ == "__main__":
    _apply_cli_flags(sys.argv[1:])
# ---- Portability: module locations are relative to this file; the external true-world executor is set explicitly.
#   FB1_DIR              fb1 module directory.  Default = **the directory of this runner file itself** --
#                        running a snapshot copy from another path imports that copy's sibling modules
#                        (previously the live tree was hard-coded, so running a copy imported the live tree).
#   DP_SCIENCEAGENT_DIR  DiscoverPhysics/ScienceAgent directory (the true-world executor).
FB1_DIR = os.environ.get("FB1_DIR") or os.path.dirname(os.path.abspath(__file__))
# DiscoverPhysics is an external repository: there is no in-repository default.
DP_SCIENCEAGENT_DIR = os.environ.get("DP_SCIENCEAGENT_DIR", "")
if not DP_SCIENCEAGENT_DIR:
    raise ValueError(
        "DP_SCIENCEAGENT_DIR must name the DiscoverPhysics/ScienceAgent directory "
        "(directory containing the scienceagent package); set it before running")
sys.path.insert(0, FB1_DIR)                  # insertion order = the same as before the edit (DP comes first)
sys.path.insert(0, DP_SCIENCEAGENT_DIR)
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.scipy.special import logsumexp
import fb1_forward as fwd, fb1_core as core
from fb1_core import (Candidate, STRUCTURES, FAMILIES, make_loglik, make_predict,
                      smc_fit, u_to_x, MAX_ACTIVE, MAX_COEFFS, MAX_OPS, DT)
from scienceagent.worlds import get_world

WORLD = os.environ.get("FB1_WORLD", "gravity")
NUMERICS = fwd.numerical_config(WORLD, os.environ)
# Configure before any likelihood/prediction JIT is built. Both core factories
# resolve this same DT and fwd method, including free-form force candidates.
DT = core.DT = NUMERICS["dt"]
fwd.INTEGRATION_METHOD = NUMERICS["method"]
fwd.ALIGN_MEASUREMENT_TIMES = NUMERICS["align_measurement_times"]
fwd.FORCE_SOFTENING = NUMERICS.get("softening", 0.0)   # 0 unless FB1_SOFTENING is set
NUMERICS_EXPLICIT = any(k in os.environ for k in (
    "FB1_INTEGRATOR", "FB1_DT", "FB1_DESIGN_GATE", "FB1_DESIGN_NOISE_INFLATE", "FB1_SOFTENING")) or (
        NUMERICS.get("design_gate_fallback", "off") != "off")
DESIGN_GATE_ACTIVE = NUMERICS["design_gate"] == "dt_halving"
# Per-world conventions (measured from the executor).  var_gt = run_benchmark.py _WORLD_VARS
#   true_force  for the true-law reference candidate (exposure diagnostic)   inertia/soft  executor conventions
#   me_probe    r interval of the ME probe = the trajectory range of that world's IC
#   me_eff      simulator effective force (fitted on a 64-point force-curve grid)   me_decl  declared analytic law
import scipy.special as _sp
import world_constants as wc
_I2PI = 1.0 / (2.0 * math.pi)
# Raw me_eff/me_decl/me_probe values are read from the data files (no rounded literals).
_GRAV_EFC = wc.effective_force_constants("gravity")   # bg_A/bg_B of FB1_FORCE_CURVE_WORLD_RULING.json
_YUK_EFC = wc.effective_force_constants("yukawa")      # A of FB1_FORCE_CURVE_yukawa_amplitude.json
_XD_ME_PROBE = wc.extra_dimensions_me_probe()          # IC v5 trajectory range (r_t_min, r_t_max)
WORLD_CFG = {
 "gravity": dict(var_gt=4.283, inertia="p2", soft=0.0, me_probe=(3.0, 6.0),
   true_force=lambda r, a, b: a * _I2PI / r,
   me_eff=lambda r: _GRAV_EFC["A"] * _I2PI / r - _GRAV_EFC["B"] * r,
   me_decl=lambda r: _I2PI / r,
   # A literal transcription of true_force (not simplified algebraically), for the truth_law record.
   law_str="F(r; p1=a, p2=b) = a / (2*pi*r)  [_I2PI = 1/(2*pi); independent of b]"),
 "yukawa": dict(var_gt=5.677, inertia="p2", soft=0.0, me_probe=(1.9, 3.7),
   true_force=lambda r, a, b: a * _I2PI * _sp.kv(1, r / 2.0) / 2.0,
   me_eff=lambda r: _YUK_EFC["A"] * _I2PI * _sp.kv(1, r / 2.0) / 2.0,
   me_decl=lambda r: _I2PI * _sp.kv(1, r / 2.0) / 2.0,
   law_str="F(r; p1=a, p2=b) = a * K1(r/2) / (4*pi)  [K1 = modified Bessel function of the "
           "second kind, order 1 (scipy.special.kv(1, .)); independent of b]"),
 "coulomb_easy": dict(var_gt=4.244, inertia="1", soft=0.05, me_probe=(2.0, 6.0),
   true_force=lambda r, a, b: a * b / r**2,
   me_eff=lambda r: 1.0 / r**2, me_decl=lambda r: 1.0 / r**2,
   law_str="F(r; p1=a, p2=b) = a*b / r^2"),
 "oscillator": dict(var_gt=6.332, inertia="p2", soft=0.05, me_probe=(3.0, 6.0),
   true_force=lambda r, a, b: a * _I2PI / r,          # G(t) is the time term -- the exposure diagnostic is evaluated at t=0
   me_eff=lambda r: _I2PI / r, me_decl=lambda r: _I2PI / r,
   law_str="F(r; p1=a, p2=b) = a / (2*pi*r)  [same radial form as gravity; the time-modulation "
           "G(t) is applied elsewhere -- this exposure-diagnostic reference is evaluated at t=0]"),
 "extra_dimensions": dict(var_gt=4.248, inertia="p2", soft=0.05, me_probe=_XD_ME_PROBE,
   true_force=lambda r, a, b: a * (2*math.pi*0.5) * float(np.sum(
       np.atleast_1d(r)[..., None] / ((np.atleast_1d(r)[..., None]**2
       + (np.arange(-20, 21) * (2*math.pi*0.5))**2) ** 1.5), axis=-1)[0]) / (4 * math.pi),
   # me_eff = me_decl as is (no amplitude correction): the pure (uncorrected) maximum relative force-curve error is
   # 0.15% (FB1_FORCE_CURVE_WORLD_RULING.json["extra_dimensions"]["pure_maxrel"]), already small, so a separate A fit
   # as for gravity/yukawa is unnecessary (nbody free space, softening soft=0.05) -- see world_constants.py.
   me_eff=wc.kk_effective_force, me_decl=wc.kk_declared_force,
   law_str="F(r; p1=a, p2=b) = a*(2*pi*0.5)/(4*pi) * sum_{n=-20}^{20} "
           "r / (r^2 + (n*2*pi*0.5)^2)^1.5  [Kaluza-Klein periodic-image sum, period 2*pi*0.5; "
           "independent of b]"),
}
CFG = WORLD_CFG[WORLD]
VAR_GT = CFG["var_gt"]
SIGMA = 0.05 * math.sqrt(VAR_GT)
RUN_SEED = int(os.environ.get("FB1_SEED", "11"))
# The observation-noise seed is tied to the run seed, but **seed 11 reproduces the earlier value 20260906 exactly**.
OBS_SEED = 20260906 + (RUN_SEED - 11)
# The held-out scoring Q is **fixed across all seeds** (fixed-Q convention) -- comparisons use the same test set.
# The default 777 is unchanged (earlier outputs unchanged).  The environment variable is used only in the leak test
# (held-out swap) that **changes only the evaluation set and checks that the belief packets stay byte-identical** --
# the inference path never sees this value.
SCORE_SEED = int(os.environ.get("FB1_SCORE_SEED", "777"))
ARM = os.environ.get("FB1_ARM", "111")          # 111 = S expanded, F expanded, active; 000 = fixed pool, random
if len(ARM) != 3 or any(bit not in "01" for bit in ARM[:2]) or ARM[2] not in "012":
    raise ValueError("FB1_ARM must match [01][01][012]")
EXPAND = ARM[0] == "1" or ARM[1] == "1"         # S or F expansion allowed
ACTIVE_DESIGN = ARM[2] in "12"                 # A: random, joint EIG, or EPIG
EPIG_DESIGN = ARM[2] == "2"
# ---- Design variant.  With the default none every VARIANT_* below is False and code paths and outputs are unchanged.
DESIGN_VARIANT = os.environ.get("FB1_DESIGN_VARIANT", "none")   # none | resolution_floor
VARIANT_ON = DESIGN_VARIANT == "resolution_floor"
VARIANT_ACTIVE = VARIANT_ON and ACTIVE_DESIGN   # the random arm does not apply the filter -- code path unchanged
VARIANT_SEED = 20260907        # key reserved for the resolution-floor filter (fully separate from the main run's random streams)
if VARIANT_ON:
    import design_variants as dv
# ---- Design search (2026-10-02).  Default random64 keeps every code path, record and random draw unchanged;
#      ascent replaces the 64-candidate draw + argmax by gradient ascent on a jax-pure estimate of the active objective
#      (fb1_design_search.py).  Composition with the resolution-floor variant or an active design gate is not defined
#      and refused; a random arm has no objective to ascend.
DESIGN_SEARCH = os.environ.get("FB1_DESIGN_SEARCH", "random64")   # random64 | ascent
if DESIGN_SEARCH not in ("random64", "ascent"):
    raise ValueError("FB1_DESIGN_SEARCH must be random64 or ascent")
ASCENT = DESIGN_SEARCH == "ascent"
if ASCENT:
    if not ACTIVE_DESIGN:
        raise ValueError("FB1_DESIGN_SEARCH=ascent requires an active design arm (FB1_ARM third digit 1 or 2)")
    if VARIANT_ON:
        raise ValueError("FB1_DESIGN_SEARCH=ascent together with FB1_DESIGN_VARIANT=resolution_floor is not defined; refused")
    if DESIGN_GATE_ACTIVE:
        raise ValueError("FB1_DESIGN_SEARCH=ascent together with FB1_DESIGN_GATE=dt_halving is not defined; refused")
    import fb1_design_search as fds
# ---- EPIG target points (2026-10-03).  Unset or "uniform" is the default run: 32 points uniform over the box from
#      the independent stream of sample_epig_targets, every code path, record and random draw unchanged.  "score_q" makes
#      the targets exactly the scoring designs Q_HEAD + Q_SHIFT, in that order (9 points, no random draw); only their
#      (p1, p2) locations enter acquisition, the held-out truths and noisy outcomes stay evaluation-only.  Only the EPIG
#      arm (ARM[2] == "2") has targets: "score_q" on any other arm is refused here, before any output or paid call,
#      rather than silently ignored; an explicit "uniform" is the default and is accepted on every arm (records nothing).
EPIG_TARGETS = os.environ.get("FB1_EPIG_TARGETS", "uniform")   # uniform | score_q
if EPIG_TARGETS not in ("uniform", "score_q"):
    raise ValueError(f"FB1_EPIG_TARGETS must be uniform or score_q, got {EPIG_TARGETS!r}")
EPIG_SCORE_Q = EPIG_TARGETS == "score_q"
if EPIG_SCORE_Q and not EPIG_DESIGN:
    raise ValueError(f"FB1_EPIG_TARGETS=score_q requires the EPIG arm (FB1_ARM third digit 2); arm {ARM!r} computes "
                     "no EPIG, so the targets would be unused; refused")
# Background-term template (periodic-box field worlds; adopted from the force-curve diagnostic, maximum relative
# error 0.798% < 1%)
BACKGROUND_ADOPTED = True


def _parse_fb1_d0(raw):
    """Normalize the initial-data policy without changing the default path."""
    if raw is None or raw in ("default", "random2"):
        return "random2"
    if raw == "mda1":
        return "mda1"
    raise ValueError(
        f"FB1_D0 must be default, random2, or mda1; got {raw!r}")


D0_POLICY = _parse_fb1_d0(os.environ.get("FB1_D0"))
D0 = 1 if D0_POLICY == "mda1" else 2
D0_POINTS = ((1.0, 1.0),)
D0_DEFINITION = (
    "MDA, Murphy (2026), arXiv:2608.09696v4, p. 29, Tables 4 and 6: "
    "the seed launch is action 3, one passive drop at r0=3 with zero velocity "
    "and p1=p2=1. Under the amendment of 2026-09-13, this engine varies "
    "only (p1, p2), so mda1 matches only the (1, 1) knobs, not the launch "
    "geometry. Initial position, velocity, and measurement times remain those "
    "of the selected IC v5 manifest. DiscoverPhysics executes requests; it "
    "does not define this initial-data policy."
)


def _initial_designs(rng):
    if D0_POLICY == "mda1":
        return np.asarray(D0_POINTS, dtype=float)
    # Keep the historical order and number of random draws for every default alias.
    return np.column_stack([rng.uniform(*P1_RANGE, D0), rng.uniform(*P2_RANGE, D0)])

# The default 16 is unchanged (earlier outputs unchanged).  Reduced only in shortened development runs (leak and smoke tests).
T_EXTRA = int(os.environ.get("FB1_T_EXTRA", "16"))


def _parse_fb1_checkpoints(raw, d0, t_extra):
    """Parser of FB1_CHECKPOINTS (an "every n checkpoints" option, declared in
    advance, default unchanged).

    Returns (checkpoints tuple, option label for the record -- the label is attached by the out["budget"] assembly
    **only when it is not the default**, so default run outputs stay byte for byte the same).
      unset or "default"     today's (2, 4, 8, 16) unchanged.
      "all"                  every n in D0..D0+T_EXTRA (e.g. T_EXTRA=16 -> 2..18, 17 values).
      "2,4,8,16" form        comma-separated integers -- checked: integer parsing, sorted and unique, within
                             [D0, D0+T_EXTRA].
    A failed check stops with a clear error (at module import, before the world/simulator is prepared) -- the same
    convention as the STRUCTURE_PROPOSALS check below (raise ValueError, before any output file is created).
    The final checkpoint (n == D0+T_EXTRA) is always included by the `or` clause of the checkpoint test even when it
    is not in this list -- the all-particle npz is written only under that condition (unchanged by the option).
    """
    final_n = d0 + t_extra
    if raw in ("default", ""):
        return (2, 4, 8, 16), "default"
    if raw == "all":
        return tuple(range(d0, final_n + 1)), "all"
    try:
        values = [int(tok.strip()) for tok in raw.split(",")]
    except ValueError:
        raise ValueError(
            f"FB1_CHECKPOINTS must be 'default', 'all', or a comma-separated list of "
            f"integers, got {raw!r}")
    if not values or sorted(set(values)) != values:
        raise ValueError(
            f"FB1_CHECKPOINTS={raw!r} must list sorted, unique integers (got {values})")
    out_of_range = [v for v in values if v < d0 or v > final_n]
    if out_of_range:
        raise ValueError(
            f"FB1_CHECKPOINTS={raw!r} has values {out_of_range} outside "
            f"[D0={d0}, D0+T_EXTRA={final_n}]")
    return tuple(values), raw


FB1_CHECKPOINTS_RAW = os.environ.get("FB1_CHECKPOINTS", "default")
CHECKPOINTS, FB1_CHECKPOINTS_OPTION = _parse_fb1_checkpoints(FB1_CHECKPOINTS_RAW, D0, T_EXTRA)
PROPOSE_AT = (4, 8, 12, 16)
N_RANDOM_DESIGNS, M_OUTER, N_INNER = 64, 128, 128
P1_RANGE = P2_RANGE = (0.1, 10.0)
Q_HEAD = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]
Q_SHIFT = [(p1, p2) for p1 in (3.0, 4.0, 5.0) for p2 in (3.0, 5.0)]

# ---- Output-record completeness.  The constants below are **record-only** -- they do not enter the fit, EIG,
#      design or scoring formulas.
PRED_COMPONENT_MAX_PARTICLES = 128     # per-candidate particle subsample cap of the predictive-mixture component npz
PRED_COMPONENT_SUBSAMPLE_SEED = 20260907   # seed reserved for that subsample (separate from the main run's random streams)
SINGLE_MODEL_SCORE_SEED = 424242       # key reserved for single top-model scoring (separate from the main run's streams)

# Pre-declared mock LLM proposal list (3 pairs per call, 4 calls).  Written out here in full to leave no room for
# post-hoc selection.
LIST_TAG = os.environ.get("FB1_LIST", "v3")        # v1 old list (for recovery), v2 abandoned, v3 final
PROPOSAL_LIST_VERSION = {"v1": "v1 (background term)", "v2": "v2 (background term + time modulation) - abandoned",
                         "v3": "v3 final (background term + time modulation + product coupling + crossover)"}[LIST_TAG]
MOCK_PROPOSALS_V1 = [
    [("power", "gauss"), ("inv_r_plus_c", "gauss"), ("inv_r2", "student_t4")],
    [("inv_r", "gauss"), ("inv_r_bg", "gauss"), ("screened", "student_t4")],
    [("inv_r_p1sq", "gauss"), ("inv_r_nomass", "gauss"), ("inv_r_bg", "student_t4")],
    [("power", "student_t4"), ("logr", "gauss"), ("const", "student_t4")],
]
MOCK_PROPOSALS_V2 = [
    [("power", "gauss"), ("inv_r_plus_c", "gauss"), ("inv_r2", "student_t4")],
    [("inv_r", "gauss"), ("inv_r_bg", "gauss"), ("screened", "student_t4")],
    [("inv_r_p1sq", "gauss"), ("inv_r_nomass", "gauss"), ("inv_r_tmod", "gauss")],
    [("power", "student_t4"), ("logr", "gauss"), ("inv_r_bg", "student_t4")],
]
MOCK_PROPOSALS_V3 = [
    [("power", "gauss"), ("inv_r_plus_c", "gauss"), ("inv_r2", "student_t4")],
    [("inv_r", "gauss"), ("inv_r_bg", "gauss"), ("screened", "student_t4")],
    [("inv_r_p1sq", "gauss"), ("inv_r_nomass", "gauss"), ("inv_r_tmod", "gauss")],
    [("prod_coupling", "gauss"), ("crossover", "gauss"), ("inv_r_bg", "student_t4")],
]
MOCK_PROPOSALS = {"v1": MOCK_PROPOSALS_V1, "v2": MOCK_PROPOSALS_V2, "v3": MOCK_PROPOSALS_V3}[LIST_TAG]
INITIAL_POOL = [("inv_r2", "gauss"), ("linear", "gauss"), ("const", "gauss"), ("screened", "gauss")]

# ---- Family-fixed check.  On the LLM path it is enforced **in the judge**: the constraint block is
#      `allowed_families=[initial_family]` (fb1_proposer.catalog_constraints and
#      fb1_proposer.free_constraints -- main() calls them with family_expansion=ARM[1]=="1"),
#      and fb1_proposer.judge_candidate and fb1_proposer._judge_free_candidate refuse any other family.
#      The mock path does not go through that judge, so the pre-declared schedule's student_t4 candidates
#      (in MOCK_PROPOSALS_V3 above) used to be admitted unchanged even on family-fixed arms (ARM[1]=="0").
#      The two values and one function below carry that rule over to the mock admission point -- the reason
#      sentences are also the judge's **verbatim** (the test test_fb1_mock_family.py reads them directly from
#      fb1_proposer and compares).  On family-expansion arms (ARM[1]=="1") both checks pass, so
#      the earlier mock outputs (110, 111) are byte-identical.
FAMILY_EXPANSION = ARM[1] == "1"
INITIAL_FAMILY = INITIAL_POOL[0][1]
ALLOWED_FAMILIES = list(FAMILIES) if FAMILY_EXPANSION else [INITIAL_FAMILY]


def family_refusal(family):
    """Family check.  None when allowed, otherwise the refusal reason sentence (the same sentence as the LLM judge)."""
    if not FAMILY_EXPANSION and family != INITIAL_FAMILY:
        return "family must equal constraints.initial_family when family_expansion is false"
    if family not in ALLOWED_FAMILIES:
        return "family must be one of constraints.allowed_families"
    return None


def admission_record(s, f, have, check_family):
    """Admission decision and one record line for one proposed pair.  Returns (admitted?, record).

    `check_family` is True only on the mock path -- candidates on the LLM path already passed the family check in
    the judge.  The rest (complexity cap, duplicate) is common to both paths, and its order and sentences are
    unchanged.
    """
    st = STRUCTURES[s]
    reason = family_refusal(f) if check_family else None
    ok = (reason is None and st.n_coeff <= MAX_COEFFS and st.n_ops <= MAX_OPS
          and (s, f) not in have)
    if ok:
        reason = "ok"
    elif reason is None:
        reason = "duplicate" if (s, f) in have else "complexity limit exceeded"
    return ok, {"pair": [s, f], "admitted": bool(ok), "reason": reason}


# ---- LLM proposer connection (catalogue-selection mode).  With the defaults (mock, P4) the values below equal today's,
#      so the proposal block's conditions, order and records do not change by a single character (byte-identical).
PROPOSER = os.environ.get("FB1_PROPOSER", "mock")                 # mock | llm
PROPOSAL_POLICY = os.environ.get("FB1_PROPOSAL_POLICY", "P4")     # P4 (default, 4 calls x 3) | P1 (one per round)
# The paid API opens only when this variable is set explicitly.  Automated helper sessions never set it.
ALLOW_PAID_API = os.environ.get("FB1_ALLOW_PAID_API", "0") == "1"
LLM_MODEL = os.environ.get("FB1_LLM_MODEL", "claude-opus-4-7")
# ---- Transport provider.  The earlier runner neither **read nor passed on** this value --
#      fb1_proposer read the environment variable directly on the paid path, so from the outputs alone one could not
#      tell which transport a run used.  Now the runner reads it, records it and passes it to the proposer explicitly.
#      The default anthropic matches the earlier interpretation (off the paid path a mock transport is used anyway).
LLM_PROVIDER = os.environ.get("FB1_LLM_PROVIDER", "anthropic")
LLM_BASE_URL = os.environ.get("FB1_LLM_BASE_URL") if LLM_PROVIDER == "openai_compat" else None
LLM_KEY_NAME = os.environ.get("FB1_LLM_KEY_NAME", "OPENAI_COMPAT_API_KEY") if LLM_PROVIDER == "openai_compat" else None
LLM_PROVIDER_ORDER = os.environ.get("FB1_LLM_PROVIDER_ORDER") if LLM_PROVIDER == "openai_compat" else None
# ---- Structure proposal mode.
#      catalog (default) = the catalogue-selection mode -- judge path and output bytes are the same as before.
#      free              = free-form grammar (fb1_grammar) -- free expressions over the named inputs {r,p1,p2,t}.
STRUCTURE_PROPOSALS = os.environ.get("FB1_STRUCTURE_PROPOSALS", "catalog")
if STRUCTURE_PROPOSALS not in ("catalog", "free"):
    raise ValueError(
        f"FB1_STRUCTURE_PROPOSALS must be catalog|free, got {STRUCTURE_PROPOSALS!r}")
FREE_STRUCTURES = STRUCTURE_PROPOSALS == "free"
if FREE_STRUCTURES and ARM[0] != "1":
    # Stops **before creating any output** (at module import).  S-fixed arms (000, 001) keep the structure
    # pool fixed at INITIAL_POOL, so there is no place for a free expression -- a half-written output could be
    # misread as "results of a free-expression arm", so no file is created at all.
    raise ValueError(
        f"FB1_STRUCTURE_PROPOSALS=free requires a structure-expansion arm (ARM[0] == '1'); "
        f"arm {ARM!r} keeps the initial pool fixed, so no free-form structure can enter it")
if FREE_STRUCTURES:
    import fb1_grammar as fbg
# ---- Trigonometric calls in the free-expression grammar (2026-10-03).  Unset or "0": the grammar every
#      earlier run used (cos/sin refused as calls outside it).  "1": fb1_grammar_trig.enable() adds cos and sin
#      (arity 1) here, at import and before any structure is compiled; fb1_grammar.py and the vendored grammar file
#      stay byte-identical (fb1_grammar_trig.py says why and what changes).  Only free-expression proposals read the
#      grammar - catalogue mode already offers inv_r_tmod with its cos/sin - so "1" needs FB1_STRUCTURE_PROPOSALS=free
#      and is refused otherwise, before any output.
if os.environ.get("FB1_GRAMMAR_TRIG", "0") not in ("0", "1"):
    raise ValueError("FB1_GRAMMAR_TRIG must be 0 or 1")
GRAMMAR_TRIG = os.environ.get("FB1_GRAMMAR_TRIG", "0") == "1"
if GRAMMAR_TRIG:
    if not FREE_STRUCTURES:
        raise ValueError("FB1_GRAMMAR_TRIG=1 requires FB1_STRUCTURE_PROPOSALS=free: only free-expression proposals "
                         "use the grammar (the catalogue already holds inv_r_tmod); refused")
    import fb1_grammar_trig as fbg_trig
    fbg_trig.enable()
# ---- A development prompt variant and input-name masking, both default off.
#      FB1_PROMPT_VARIANT    path of a system prompt file whose line 1 is `# prompt_version: <tag>`
#                            (sent as stored; the run records the tag and the file's sha256).
#      FB1_MASK_INPUT_NAMES  1 = the model reads X1..X4 for r, p1, p2, t (fb1_proposer.InputNameMask).
#      Unset, nothing below this line of the proposal path changes a byte. Both need the LLM
#      proposal path and both are refused with a production label (below, before any output).
PROMPT_VARIANT = os.environ.get("FB1_PROMPT_VARIANT") or None
if os.environ.get("FB1_MASK_INPUT_NAMES", "0") not in ("0", "1"):
    raise ValueError("FB1_MASK_INPUT_NAMES must be 0 or 1")
MASK_INPUT_NAMES = os.environ.get("FB1_MASK_INPUT_NAMES", "0") == "1"
# ---- Unified protocol (the registered protocol and its addenda), each option unset = today's run byte for byte. All but FB1_DESIGN_POLICY need the LLM proposal path
#      with free-form structures; every check runs here, before any output or paid call.
#      FB1_INITIAL_PROPOSALS   unset/0 (default: INITIAL_POOL's four catalogue forms) | K, 1 <= K <= MAX_ACTIVE: after the
#                              D0 launches, K independent single-candidate calls in parallel on the empty-pool packet
#                              (BURST_PARAGRAPH), then refill waves for the shortfall (BURST_REFILL_PARAGRAPH, the packet
#                              shows the pool admitted so far), at most three waves, shortfall recorded; no catalogue pool.
#      FB1_ACTIVE_MAXIMUM      unset | 12: the common flag; ForceBench's cap is fb1_core.MAX_ACTIVE = 12, any other value
#                              is refused.
#      FB1_PROPOSAL_LIMIT      unset | 3: proposal rounds t = 1..T_EXTRA (n = D0 + t), each ONE call answering a JSON array
#                              of three candidates (COMPOSITION v2); exact duplicates (pool, archive, earlier in the reply)
#                              are recorded and not refitted, nothing replaces them; the admitted ones are fitted from the
#                              prior on all current data and the pool is cut to the top MAX_ACTIVE by log evidence (no
#                              newborn exemption) before the round's design.
#      FB1_ARCHIVE_REVIEW_ROUNDS unset | a comma or space list of rounds in 1..T_EXTRA: on a listed round the call also
#                              carries ARCHIVE_REVIEW v1 and answers {"candidates": [...], "revive": [...]}; each valid
#                              revive id (an archived retired_by_evidence model, at most two) is refit from the prior on all
#                              current data and competes with the pool and the new candidates in one cut. Needs
#                              FB1_PROPOSAL_LIMIT=3.
#      FB1_DESIGN_POLICY       unset (default) | joint_eig | model_eig | parameter_eig | epig | random. It must agree with
#                              the arm's design digit (random 0, the three EIG rules 1, epig 2) and an active rule needs
#                              FB1_DESIGN_SEARCH=ascent. Model / Parameter EIG ascend that part of the joint EIG
#                              (fb1_design_search.eig_parts_pure); every rule records, for every evaluated design, the
#                              joint, Model and Parameter EIG from the same Monte Carlo draws and the point disagreement.
#      FB1_PACKET_ARCHIVE      full (default: archive rows as today) | compact: the packets' archive rows without their
#                              numerical_equivalence and prior_probe blocks (model_id, outcome, reason and reason_code
#                              stay); a packet-size option that changes what the model reads, so it is off unless chosen.
#      FB1_CATALOGUE           on (default: the packet shows the law catalogue as every run since 9/8 - constraints'
#                              initial_structures, structure_parameters and the grammar block's catalogue entries) | off
#                              (fb1_proposer.catalogue_off_constraints: those taken out of what the model reads, and
#                              fb1_proposer.catalogue_off_packet: no catalogue trace in the packet's pool and archive -
#                              catalog_reference and numerical_equivalence blocks, '^' notation, catalogue reason code
#                              and names; the engine's admission checks still use the catalogue). The operator
#                              decides at launch.
DESIGN_POLICIES = ("joint_eig", "model_eig", "parameter_eig", "epig", "random")
#: the arm design digit each rule belongs to, and its target as the BoxingGym design-policy records name it
DESIGN_POLICY_DIGIT = {"joint_eig": "1", "model_eig": "1", "parameter_eig": "1", "epig": "2", "random": "0"}
DESIGN_POLICY_TARGET = {"joint_eig": "joint_model_parameter", "model_eig": "model_structure",
                        "parameter_eig": "model_parameter_given_structure", "epig": "prediction_at_targets",
                        "random": "none"}
COMPOSED_POLICY = "composed_3"
COMPOSED_LIMIT = 3
BURST_WAVES = (0, 1, 2)              # count runners' INITIAL_BURST_WAVES: wave 0, then at most two refill waves
BURST_REFILL_POLICY = "pool_visible_refill_waves_max_3"   # count_models.INITIAL_BURST_REFILL_POLICY
BURST_CONCURRENCY = 6                # count runners' DEFAULT_BURST_CONCURRENCY; a mock transport calls one at a time
BURST_PACKET_FIT_SEED = 20261005     # refill-packet fits only (their own stream; the run's key is not consumed)
DESIGN_RECORD_SEED = 20261006        # random-arm EIG record only (its own stream; the run's key and rng are not consumed)
PACKET_ARCHIVE_DROPPED_KEYS = ("numerical_equivalence", "prior_probe")


def _parse_whole(name, raw, low, high):
    text = "" if raw is None else str(raw).strip()
    if not text.isdigit() or str(int(text)) != text or not low <= int(text) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}, got {raw!r}")
    return int(text)


def _parse_round_list(raw, horizon):
    """core_count.parse_round_list: a comma or space separated list of rounds, sorted and unique, each in 1..horizon."""
    try:
        values = tuple(sorted({int(item) for item in str(raw).replace(",", " ").split()}))
    except ValueError:
        raise ValueError(f"FB1_ARCHIVE_REVIEW_ROUNDS must list integers, got {raw!r}") from None
    if not values or any(not 1 <= value <= horizon for value in values):
        raise ValueError(f"FB1_ARCHIVE_REVIEW_ROUNDS entries must be 1..{horizon} (T_EXTRA), got {raw!r}")
    return values


_raw = os.environ.get("FB1_INITIAL_PROPOSALS")
BURST_SIZE = 0 if _raw in (None, "", "0") else _parse_whole("FB1_INITIAL_PROPOSALS", _raw, 1, MAX_ACTIVE)
BURST = BURST_SIZE > 0
if os.environ.get("FB1_ACTIVE_MAXIMUM") not in (None, str(MAX_ACTIVE)):
    raise ValueError(f"FB1_ACTIVE_MAXIMUM must be {MAX_ACTIVE} (fb1_core.MAX_ACTIVE), got "
                     f"{os.environ.get('FB1_ACTIVE_MAXIMUM')!r}")
ACTIVE_MAXIMUM_STATED = os.environ.get("FB1_ACTIVE_MAXIMUM") is not None
_raw = os.environ.get("FB1_PROPOSAL_LIMIT")
COMPOSED = _raw not in (None, "")
if COMPOSED and _raw != str(COMPOSED_LIMIT):
    raise ValueError(f"FB1_PROPOSAL_LIMIT must be {COMPOSED_LIMIT} (COMPOSITION v2 names three candidates), got {_raw!r}")
_raw = os.environ.get("FB1_ARCHIVE_REVIEW_ROUNDS")
REVIEW_ROUNDS = () if _raw in (None, "") else _parse_round_list(_raw, T_EXTRA)
DESIGN_POLICY = os.environ.get("FB1_DESIGN_POLICY") or None
if DESIGN_POLICY is not None and DESIGN_POLICY not in DESIGN_POLICIES:
    raise ValueError(f"FB1_DESIGN_POLICY must be one of {', '.join(DESIGN_POLICIES)}, got {DESIGN_POLICY!r}")
PACKET_ARCHIVE = os.environ.get("FB1_PACKET_ARCHIVE", "full")
if PACKET_ARCHIVE not in ("full", "compact"):
    raise ValueError(f"FB1_PACKET_ARCHIVE must be full or compact, got {PACKET_ARCHIVE!r}")
CATALOGUE = os.environ.get("FB1_CATALOGUE", "on")
if CATALOGUE not in ("on", "off"):
    raise ValueError(f"FB1_CATALOGUE must be on or off, got {CATALOGUE!r}")
CATALOGUE_STATED = "FB1_CATALOGUE" in os.environ
for _option, _on in (("FB1_INITIAL_PROPOSALS", BURST), ("FB1_PROPOSAL_LIMIT", COMPOSED),
                     ("FB1_PACKET_ARCHIVE=compact", PACKET_ARCHIVE == "compact"), ("FB1_CATALOGUE", CATALOGUE_STATED)):
    if _on and (PROPOSER != "llm" or not FREE_STRUCTURES):
        raise ValueError(f"{_option} requires FB1_PROPOSER=llm and FB1_STRUCTURE_PROPOSALS=free; refused")
if COMPOSED and "FB1_PROPOSAL_POLICY" in os.environ:
    raise ValueError("FB1_PROPOSAL_LIMIT sets the proposal schedule itself; unset FB1_PROPOSAL_POLICY")
if REVIEW_ROUNDS and not COMPOSED:
    raise ValueError("FB1_ARCHIVE_REVIEW_ROUNDS is part of the composed proposal call; set FB1_PROPOSAL_LIMIT=3 with it")
if DESIGN_POLICY is not None:
    if ARM[2] != DESIGN_POLICY_DIGIT[DESIGN_POLICY]:
        raise ValueError(f"FB1_DESIGN_POLICY={DESIGN_POLICY} needs arm design digit {DESIGN_POLICY_DIGIT[DESIGN_POLICY]}, "
                         f"arm is {ARM!r}; refused")
    if DESIGN_POLICY != "random" and not ASCENT:
        raise ValueError(f"FB1_DESIGN_POLICY={DESIGN_POLICY} requires FB1_DESIGN_SEARCH=ascent; refused")
if COMPOSED:
    PROPOSAL_POLICY = COMPOSED_POLICY
COMPOSED_EVENTS = tuple(range(D0 + 1, D0 + T_EXTRA + 1))   # rounds t = 1..T_EXTRA
UNIFIED = (BURST or COMPOSED or bool(REVIEW_ROUNDS) or DESIGN_POLICY is not None or PACKET_ARCHIVE != "full"
           or CATALOGUE_STATED)
if DESIGN_POLICY is not None and not ASCENT:
    import fb1_design_search as fds   # the random arm's records use its estimators
# the ascent objective: EPIG on the EPIG arm, a part of the joint EIG under Model / Parameter EIG, else the joint EIG
ASCENT_OBJECTIVE = "epig" if EPIG_DESIGN else (DESIGN_POLICY if DESIGN_POLICY in ("model_eig", "parameter_eig") else "eig")
#: timer keys of the options, added to TIMERS only when on (the default timing record keeps its keys)
TIMERS_EXTRA = {**({"initial_burst": 0.0} if BURST else {}),
                **({"fit_new_candidates": 0.0} if COMPOSED else {}),
                **({"revival": 0.0} if REVIEW_ROUNDS else {}),
                **({"design_record": 0.0} if DESIGN_POLICY is not None else {})}
# The run label is an administrative output label, never authority to use a world.
# A missing request remains development. The manifest and actual runtime settings
# are validated before main() can create output or construct a live client.
import run_label_policy as label_policy

RUN_LABEL_REQUEST = os.environ.get("FB1_RUN_LABEL")
MANIFEST_REF = os.environ.get("FB1_MANIFEST_REF")
LABEL_EXPLICIT = "FB1_RUN_LABEL" in os.environ or "FB1_MANIFEST_REF" in os.environ
if PROPOSER not in ("mock", "llm"):
    raise ValueError("FB1_PROPOSER must be mock or llm")
if LLM_PROVIDER not in ("anthropic", "openai", "openai_compat"):
    raise ValueError("FB1_LLM_PROVIDER must be anthropic, openai, or openai_compat")
RUN_LABEL_RULE = "v4 (2026-09-18): explicit production requires a verified matching paper or paper_neuron LLM cohort with registered forward backend and decoding options; fixed/mock runs are development; prompt_version and prompt_sha256 are declared, required and verified for core, boxinggym and chembench"
RUN_LABEL_FORCED_TO_DEVELOPMENT = False


def _label_runtime_settings(effective_proposer=None):
    """Actual values, not a paper default inferred from the provider name.

    The run-label policy binds the two decoding options as registered settings, so a
    ForceBench production run can no longer send one the manifest never registered.
    They are declared exactly where the transport reads them - `reasoning` only on a
    compatibility endpoint, the reply budget on any live provider - and only when the
    environment actually states one, so a run without them declares what it always did.
    """
    settings = {
        "particles": core.N_PARTICLES, "chains": core.N_CHAINS,
        "t_extra": T_EXTRA, "d0_policy": D0_POLICY, "d0_size": D0,
        "d0_points": [list(point) for point in D0_POINTS] if D0_POLICY == "mda1" else None,
        "checkpoints": FB1_CHECKPOINTS_OPTION, "checkpoint_values": list(CHECKPOINTS),
        "integrator": NUMERICS["mode"], "dt": DT,
        "design_gate": NUMERICS["design_gate"], "noise_inflate": NUMERICS["noise_inflate"],
        "design_variant": DESIGN_VARIANT, "proposal_policy": PROPOSAL_POLICY,
        "proposal_events": list(PROPOSE_AT) if PROPOSAL_POLICY == "P4" else (
            list(COMPOSED_EVENTS) if PROPOSAL_POLICY == COMPOSED_POLICY else list(range(D0, D0 + T_EXTRA))),
        "candidates_per_event": 3 if PROPOSAL_POLICY in ("P4", COMPOSED_POLICY) else 1,
        "structure_proposals": STRUCTURE_PROPOSALS,
    }
    if EPIG_DESIGN:
        settings["design_policy"] = "epig"
        if EPIG_SCORE_Q:   # registered only when non-default
            settings["epig_targets"] = "score_q"
    if ASCENT:
        settings["design_search"] = "ascent"
    if GRAMMAR_TRIG:   # registered only when on
        settings["grammar_trig"] = sorted(fbg_trig.TRIG_CALL_ARITY)
    if NUMERICS.get("softening", 0.0) != 0.0:   # registered only when non-default
        settings["softening"] = NUMERICS["softening"]
    if NUMERICS.get("design_gate_fallback", "off") != "off":
        settings.update(design_gate_fallback=NUMERICS["design_gate_fallback"],
                        fallback_k=NUMERICS["fallback_k"])
    if effective_proposer is not None and effective_proposer != "mock":
        # The Anthropic transport sends the reasoning control too, so it
        # is declared for it as well.
        reasoning = (os.environ.get("FB1_LLM_REASONING")
                     if effective_proposer in ("openai_compat", "anthropic") else None)
        if reasoning:
            settings["llm_reasoning"] = label_policy.parse_llm_reasoning(reasoning)
        budget = os.environ.get("FB1_LLM_MAX_TOKENS")
        if budget:
            settings["llm_max_tokens"] = label_policy.parse_llm_max_tokens(budget)
    # Unified protocol: registered only when on (the EPIG arm already registers design_policy "epig").
    if BURST:
        settings["initial_proposals"] = BURST_SIZE
    if ACTIVE_MAXIMUM_STATED:
        settings["active_maximum"] = MAX_ACTIVE
    if COMPOSED:
        settings["proposal_limit"] = COMPOSED_LIMIT
    if REVIEW_ROUNDS:
        settings["archive_review_rounds"] = list(REVIEW_ROUNDS)
    if DESIGN_POLICY is not None and not EPIG_DESIGN:
        settings["design_policy"] = DESIGN_POLICY
    if PACKET_ARCHIVE != "full":
        settings["packet_archive"] = PACKET_ARCHIVE
    if CATALOGUE_STATED:
        settings["catalogue"] = CATALOGUE
    return settings


def _resolve_label_receipt():
    # allow_live=False is a mock transport even when the LLM rehearsal path is on.
    effective_proposer = LLM_PROVIDER if PROPOSER == "llm" and ALLOW_PAID_API else "mock"
    profile = "paper" if MANIFEST_REF is not None and core.N_PARTICLES == 500 and core.N_CHAINS == 2 else "dev"
    host = None
    order = None
    if effective_proposer == "openai_compat":
        from urllib.parse import urlsplit
        if not isinstance(LLM_BASE_URL, str):
            raise ValueError("FB1_LLM_BASE_URL must be supplied for openai_compat")
        host = urlsplit(LLM_BASE_URL).hostname
        import fb1_proposer as label_proposer
        order = label_proposer.normalize_provider_order(LLM_PROVIDER_ORDER)
    return label_policy.resolve_label(
        RUN_LABEL_REQUEST, profile, effective_proposer, cell=ARM,
        manifest_ref=MANIFEST_REF, model=LLM_MODEL if effective_proposer != "mock" else None,
        bundle="forcebench", task=WORLD, seed=RUN_SEED, base_url_host=host,
        provider_order=order, settings=_label_runtime_settings(effective_proposer))


LABEL_RECEIPT = _resolve_label_receipt()
RUN_LABEL = LABEL_RECEIPT["run_label"]
if PROMPT_VARIANT is not None or MASK_INPUT_NAMES:
    if PROPOSER != "llm":
        raise ValueError("FB1_PROMPT_VARIANT and FB1_MASK_INPUT_NAMES require FB1_PROPOSER=llm")
    if RUN_LABEL == "production":
        raise ValueError("FB1_PROMPT_VARIANT and FB1_MASK_INPUT_NAMES are development-only; "
                         "production is refused")


def _recheck_label_receipt():
    label_policy.recheck_manifest(LABEL_RECEIPT)
    if LABEL_EXPLICIT and _resolve_label_receipt() != LABEL_RECEIPT:
        raise ValueError("ForceBench runtime settings changed after label validation")

PROPOSAL_EVENTS = tuple(PROPOSE_AT) if PROPOSAL_POLICY == "P4" else (
    COMPOSED_EVENTS if PROPOSAL_POLICY == COMPOSED_POLICY else tuple(range(D0, D0 + T_EXTRA)))
N_PROPOSAL_EVENTS = len(MOCK_PROPOSALS) if PROPOSAL_POLICY == "P4" else len(PROPOSAL_EVENTS)
# Unified protocol: composed_3 makes one call answering three candidates per round, rounds t = 1..T_EXTRA.
CANDIDATES_PER_EVENT = 3 if PROPOSAL_POLICY in ("P4", COMPOSED_POLICY) else 1
PRIOR_PROBE_SEED = 20260908      # key reserved for the prior finiteness probe (separate from the main run's random streams)
PRIOR_PROBE_SAMPLES = 64
if PROPOSER == "llm":
    import fb1_proposer as fbp
    if PROMPT_VARIANT is not None:
        fbp.set_prompt_variant(PROMPT_VARIANT)       # refuses a file without a tag line
    if LLM_PROVIDER == "openai_compat" and ALLOW_PAID_API:
        if not os.environ.get("FB1_LLM_MODEL"):
            raise ValueError("FB1_LLM_MODEL must be explicit for openai_compat")
        fbp.validate_compat_options(LLM_BASE_URL, LLM_KEY_NAME, LLM_MODEL)
        fbp.normalize_provider_order(LLM_PROVIDER_ORDER)

# Portability: output root and frozen-input paths.  Both read **the same environment variables and the same
# defaults** as world_constants (world_constants.OUT and world_constants.IC_MANIFEST_PATH), so the two modules agree.
#   FB1_OUT_DIR      root where outputs (run JSON, npz, packets) are written.  Default = the snapshot's outputs/.
#   FB1_IC_MANIFEST  frozen IC manifest file.  Default = the v5 manifest under FB1_DATA_DIR.
# FB1_OUT_DIR moves **outputs only** (a switch that avoids overwriting golden outputs) -- frozen inputs are pointed
# to separately by FB1_DATA_DIR, so changing only FB1_OUT_DIR still gives a valid run.
OUT_DIR = os.environ.get("FB1_OUT_DIR", wc.OUT)
IC_MANIFEST = os.environ.get("FB1_IC_MANIFEST", wc.IC_MANIFEST_PATH)
IC = json.load(open(IC_MANIFEST))["worlds"][WORLD]
TIMES = np.asarray(IC["measurement_times"], float)
POS0, VEL0 = jnp.array(IC["pos2"]), jnp.array(IC["velocity2"])
T_OBS = len(TIMES)
# Only field worlds (periodic box, grid).  nbody has no grid, so a resolution floor is not defined.
VARIANT_APPLIES = VARIANT_ACTIVE and IC["engine"] == "field"

# ---------------- True world (with call counters) ----------------
class World:
    """The true world (DiscoverPhysics executor) with counters of noisy and noise-free calls."""
    def __init__(self):
        self.exe = get_world(WORLD, engine=IC["engine"], noise_std=SIGMA, noise_seed=OBS_SEED)["executor"]
        self.n_noisy = self.n_clean = 0
    def _req(self, p1, p2):
        return {"p1": float(p1), "p2": float(p2), "pos2": [float(v) for v in np.asarray(POS0)],
                "velocity2": [float(v) for v in np.asarray(VEL0)], "measurement_times": list(TIMES)}
    def observe(self, p1, p2):
        """One noisy observation (T,2) of the trajectory at design (p1, p2)."""
        self.n_noisy += 1
        return np.asarray(self.exe.run([self._req(p1, p2)])[0]["pos2"], float)
    def truth(self, p1, p2):
        """The noise-free trajectory (T,2) at design (p1, p2), with the executor's noise disabled for this call."""
        self.n_clean += 1
        with self.exe.noise_disabled():
            return np.asarray(self.exe.run([self._req(p1, p2)])[0]["pos2"], float)

_EVAL_WORLD = [None]
def eval_world():
    """True world reserved for the evaluator.  Used only to record the noise-free min r **after** the choice is made.

    Kept separate from the main run's world object so the inference and design-choice paths cannot see this call.
    (That a noise-free call does not consume the RNG is checked in every run by gate_checks.)
    """
    if _EVAL_WORLD[0] is None:
        _EVAL_WORLD[0] = World()
    return _EVAL_WORLD[0]

TIMERS = {"fit": 0.0, "eig": 0.0, "forward_predict": 0.0, "scoring": 0.0, "world": 0.0, "propose": 0.0}
TIMERS.update(TIMERS_EXTRA)   # unified protocol: empty unless an option is on
class Clock:
    """Context manager that adds the elapsed wall time of its block to TIMERS[k]."""
    def __init__(self, k): self.k = k
    def __enter__(self): self.t = time.perf_counter(); return self
    def __exit__(self, *a): TIMERS[self.k] += time.perf_counter() - self.t

_LL_CACHE = {}
def loglik_for(cand, p1, p2, y):
    """jit cache keyed by (candidate, number of designs).  Recompiles only when D grows."""
    k = (cand.struct, cand.family, len(p1))
    if k not in _LL_CACHE:
        base = make_loglik(cand)
        _LL_CACHE[k] = fwd.timed(jax.jit(lambda u, a, b, yy: base(u, a, b, yy, POS0, VEL0, TIMES)))
    f = _LL_CACHE[k]
    return lambda u: f(u, p1, p2, y)

_PR_CACHE = {}
def predict_for(cand, n_design):
    """Jitted posterior predictor of `cand` for `n_design` designs (cached): f(u, p1, p2) -> (D,P,T,2) positions."""
    k = (cand.struct, cand.family, n_design)
    if k not in _PR_CACHE:
        base = make_predict(cand)
        _PR_CACHE[k] = fwd.timed(jax.jit(lambda u, a, b: base(u, a, b, POS0, VEL0, TIMES)))
    return _PR_CACHE[k]

def log_dens(y, mu, s, family):
    """y (...,T,2), mu (...,T,2), s (...)  -> (...) summed log density."""
    z = (y - mu) / s[..., None, None]
    ll = (-0.5 * z**2 - 0.5 * math.log(2 * math.pi) - jnp.log(s)[..., None, None]) if family == "gauss" \
         else (core._log_t4(z) - jnp.log(s)[..., None, None])
    return ll.sum(axis=(-1, -2))

# ---------------- Per-coordinate KL(truth||BMA) ----------------
# Definition: q_id = one scalar (query point, time, axis) (the same as the downstream scorer's convention --
#   score_predictions.py of the ForceBench proposer tools).
#   KL_fb1 = mean_{q_id in Q(180)} \int N(y;truth_q,sigma_world^2)
#                                    [log N(y;truth_q,sigma_world^2) - log p_BMA(y|q_id)] dy
#   p_BMA(y|q_id) = sum_c w_c * (1/P) sum_p f(y; mu_{c,p,q_id}, s_{c,p})  (component = candidate x particle;
#   the 20 coordinates are independent within each component, so per-coordinate marginalization is exact).
# The true side is a 1-D normal (closed form), ours is a mixture (no closed form) -> computed essentially exactly
# with 1-D Gauss-Legendre quadrature (no Monte Carlo needed).  Interval = truth_q +- KL_WINDOW_SIGMA_MULT *
# max(sigma_world, mixture standard deviation).  The difference between N and 2N nodes is recorded as
# quadrature_error, and the true mass outside the interval (symmetric normal tail, analytic) as window_mass_outside
# (neither may be 0 or blank).
# The joint (20-dimensional) KL is not formed -- the mean of per-coordinate KLs is the reported unit, and the sum of
# per-coordinate KLs is an upper bound on the joint KL (since the noise is independent across coordinates).
KL_QUADRATURE_NODES = 64                 # base node count N.  The reported value uses 2N (the finer one).
KL_WINDOW_SIGMA_MULT = 8.0
KL_FULL_MU_NPZ_MAX_BYTES = 20_000_000    # 20 MB.  Above it, quadrature-node values replace the full particle set (disclosed).

# Constant term of the nu=4 Student-t log density -- a pure-numpy reimplementation of **exactly the same formula**
# as fb1_core._log_t4 (avoids jax call overhead in the hot loop).  Agreement is pinned by a test
# (test_29fb_log_t4_np_matches_fb1_core_log_t4).
_LOG_T4_CONST = (math.lgamma(2.5) - math.lgamma(2.0) - 0.5 * math.log(4.0 * math.pi))
def _log_t4_np(z):
    return _LOG_T4_CONST - 2.5 * np.log1p(np.asarray(z, dtype=np.float64) ** 2 / 4.0)

_GL_NODE_CACHE: dict = {}
def _gl_nodes(n):
    """Gauss-Legendre nodes and weights on [-1,1] (deterministic, cached -- the standard numpy.polynomial routine)."""
    if n not in _GL_NODE_CACHE:
        _GL_NODE_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GL_NODE_CACHE[n]


def _mixture_moments(mus, ss, w, families):
    """Mean and standard deviation of a mixture (component = candidate x particle, weights w already w_c/P_c).

    The variance factor of a student_t4 component = nu/(nu-2) = 2 (nu=4, the same degrees of freedom as
    fb1_core._log_t4)."""
    var_factor = np.where(np.asarray(families) == "gauss", 1.0, 2.0)
    comp_var = np.asarray(ss, dtype=np.float64) ** 2 * var_factor
    mus = np.asarray(mus, dtype=np.float64)
    m = float(np.sum(w * mus))
    v = float(np.sum(w * (comp_var + (mus - m) ** 2)))
    return m, math.sqrt(max(v, 0.0))


def kl_coordinate(truth_q, sigma_world, mus, ss, w, families,
                  n_nodes=KL_QUADRATURE_NODES, return_nodes=False):
    """KL(N(truth_q,sigma_world^2) || mixture) of one coordinate (q_id).  Component arrays all have length M(=sum_c P_c).

    Returned dict: value (estimate with 2N nodes), quadrature_error (|N vs 2N value difference|), window_mass_outside
    (analytic, mass of the true density outside the interval), n_nodes (=2N, the node count value used)
    [, nodes_x, nodes_log_bma (only when return_nodes=True -- node values that replace the particles above 20MB;
    their size is tied to n_nodes, not to the particle count)]."""
    mus = np.asarray(mus, dtype=np.float64); ss = np.asarray(ss, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64); fam = np.asarray(families)
    truth_q, sigma_world = float(truth_q), float(sigma_world)
    _, mix_std = _mixture_moments(mus, ss, w, fam)
    half = KL_WINDOW_SIGMA_MULT * max(sigma_world, mix_std)
    lo, hi = truth_q - half, truth_q + half
    log_w = np.log(w)
    gauss_mask = fam == "gauss"

    def _integral(n):
        xi, wi = _gl_nodes(n)
        x = 0.5 * (hi - lo) * xi + 0.5 * (hi + lo)
        wq = 0.5 * (hi - lo) * wi
        log_truth = (-0.5 * ((x - truth_q) / sigma_world) ** 2
                    - 0.5 * math.log(2.0 * math.pi) - math.log(sigma_world))
        z = (x[:, None] - mus[None, :]) / ss[None, :]
        log_gauss = -0.5 * z**2 - 0.5 * math.log(2.0 * math.pi) - np.log(ss)[None, :]
        log_t4 = _log_t4_np(z) - np.log(ss)[None, :]
        logf = np.where(gauss_mask[None, :], log_gauss, log_t4)
        terms = log_w[None, :] + logf
        peak = np.max(terms, axis=1, keepdims=True)
        log_bma = peak[:, 0] + np.log(np.sum(np.exp(terms - peak), axis=1))
        truth_density = np.exp(log_truth)
        integrand = truth_density * (log_truth - log_bma)
        return float(np.sum(wq * integrand)), x, log_bma

    val_n, _, _ = _integral(n_nodes)
    val_2n, x_2n, log_bma_2n = _integral(2 * n_nodes)
    window_mass_outside = math.erfc(half / (sigma_world * math.sqrt(2.0)))
    out = {"value": val_2n, "quadrature_error": abs(val_2n - val_n),
           "window_mass_outside": window_mass_outside, "n_nodes": 2 * n_nodes}
    if return_nodes:
        out["nodes_x"], out["nodes_log_bma"] = x_2n, log_bma_2n
    return out


def score_set_kl(mus, ss, weights, families, truths, sigma_world,
                 n_nodes=KL_QUADRATURE_NODES, return_nodes=False):
    """Per-coordinate KL over all coordinates (Dc,T,2).  mus/ss: the all-particle (1000), double-precision lists that
    score_set() has already computed (candidate order = weights/families order) -- no recomputation; the values are used
    as they were before subsampling and float32 conversion: during the run, in double
    precision, with all particles."""
    C = len(mus)
    P = int(mus[0].shape[1])
    assert all(int(m.shape[1]) == P for m in mus), "every candidate has the same particle count (500 x 2 chains = 1000)"
    D, T, A = mus[0].shape[0], mus[0].shape[2], mus[0].shape[3]
    weights = np.asarray(weights, dtype=np.float64)
    w_all = np.concatenate([np.full(P, float(weights[c]) / P) for c in range(C)])
    s_all = np.concatenate([np.asarray(ss[c], dtype=np.float64) for c in range(C)])
    fam_all = np.concatenate([np.full(P, families[c]) for c in range(C)])
    mu_stack = [np.asarray(m, dtype=np.float64) for m in mus]
    truths = np.asarray(truths, dtype=np.float64)

    values = np.empty((D, T, A)); qerr = np.empty((D, T, A)); wmo = np.empty((D, T, A))
    nodes_x = np.empty((D, T, A, 2 * n_nodes)) if return_nodes else None
    nodes_log_bma = np.empty((D, T, A, 2 * n_nodes)) if return_nodes else None
    n_used = 2 * n_nodes
    for d in range(D):
        for t in range(T):
            for a in range(A):
                mu_q = np.concatenate([mu_stack[c][d, :, t, a] for c in range(C)])
                r = kl_coordinate(float(truths[d, t, a]), sigma_world, mu_q, s_all, w_all,
                                  fam_all, n_nodes=n_nodes, return_nodes=return_nodes)
                values[d, t, a] = r["value"]; qerr[d, t, a] = r["quadrature_error"]
                wmo[d, t, a] = r["window_mass_outside"]; n_used = r["n_nodes"]
                if return_nodes:
                    nodes_x[d, t, a] = r["nodes_x"]; nodes_log_bma[d, t, a] = r["nodes_log_bma"]
    per_query = [{
        "design_index": d,
        "kl_mean": float(values[d].mean()), "kl_max": float(values[d].max()),
        "quadrature_error_max": float(qerr[d].max()),
        "window_mass_outside_max": float(wmo[d].max()),
    } for d in range(D)]
    out = {"value": float(values.mean()), "quadrature_error": float(qerr.max()),
           "window_mass_outside": float(wmo.max()), "n_nodes": int(n_used),
           "per_query": per_query}
    if return_nodes:
        out["_nodes_x"], out["_nodes_log_bma"] = nodes_x, nodes_log_bma
    return out


def combine_checkpoint_kl(head_kl, shift_kl, n_head_scalars, n_shift_scalars, full_particle_mu=None):
    """checkpoints[n].kl: combines the score_set_kl outputs of head (usually 60 scalars) and shift (usually 120
    scalars) as a mean weighted by scalar count.  quadrature_error and window_mass_outside report the worse (larger)
    of the two Q sets (a conservative error bound for the whole checkpoint).  n_nodes is always equal for both (same
    constant)."""
    total = n_head_scalars + n_shift_scalars
    return {
        "value": (head_kl["value"] * n_head_scalars + shift_kl["value"] * n_shift_scalars) / total,
        "quadrature_error": max(head_kl["quadrature_error"], shift_kl["quadrature_error"]),
        "window_mass_outside": max(head_kl["window_mass_outside"], shift_kl["window_mass_outside"]),
        "n_nodes": head_kl["n_nodes"],
        "per_axis": {"h": head_kl, "s": shift_kl},
        "per_query_summary": (
            [dict(q, group="head") for q in head_kl["per_query"]]
            + [dict(q, group="shift") for q in shift_kl["per_query"]]),
        "full_particle_mu": full_particle_mu,
    }


# Numerical sensitivity uses posterior predictions only, never World/evaluation.
_GATE_CACHE = {}
def design_gate(fits, weights, designs):
    """dt-halving check of candidate designs on the top model's posterior: returns (keep mask, per-design delta, record)."""
    top = int(np.argmax(weights))
    cand, u = fits[top]
    cache_key = (cand.struct, cand.family, len(designs), DT, NUMERICS["method"])
    if cache_key not in _GATE_CACHE:
        st = STRUCTURES[cand.struct]
        k = len(st.kinds)
        def trajectories(bank, p1, p2):
            """Trajectories of the posterior particles at step DT and at DT/2 for designs (p1, p2)."""
            theta = u_to_x(bank, cand.kinds())[:, :k]
            args = (st.force, theta, p1, p2, POS0, VEL0, TIMES)
            return (fwd.integrate(*args, DT, method=NUMERICS["method"]),
                    fwd.integrate(*args, DT / 2, method=NUMERICS["method"]))
        _GATE_CACHE[cache_key] = fwd.timed(jax.jit(trajectories))
    coarse, fine = _GATE_CACHE[cache_key](u, jnp.asarray(designs[:, 0]), jnp.asarray(designs[:, 1]))
    keep, delta, record = fwd.assess_design_gate(
        coarse, fine, SIGMA, NUMERICS["noise_inflate"],
        fallback=NUMERICS.get("design_gate_fallback", "off"),
        fallback_k=NUMERICS.get("fallback_k", 8))
    record.update(top_model=cand.label(), top_weight=float(weights[top]),
                  posterior_particles=int(u.shape[0]), dt=DT, fine_dt=DT / 2,
                  integrator=NUMERICS["method"], candidates=np.asarray(designs).tolist())
    return keep, delta, record


def gated_acquisition_candidates(fits, weights, designs, out_base, n_designs):
    """Select the audited gate subset or preserve the existing failure artifact."""
    with Clock("eig"):
        keep, delta, record = design_gate(fits, weights, designs)
    if not np.any(keep):
        with open(out_base + "_gate_failure.json", "w") as handle:
            json.dump(dict(n_designs=n_designs, design_gate=record), handle, indent=2)
        raise RuntimeError("dt_halving excluded every candidate; no ungated fallback")
    indices = np.flatnonzero(keep)
    noise_delta = delta[keep] if NUMERICS["noise_inflate"] else None
    return designs[keep], indices, noise_delta, record


# ---------------- joint EIG ----------------
def joint_eig(key, fits, weights, designs, noise_delta=None):
    """EIG(d) = E_{(c,th),y}[ log p(y|c,th,d) - log p(y|d) ].  fits: [(cand, u)]"""
    D = designs.shape[0]
    if noise_delta is not None:
        noise_delta = np.asarray(noise_delta, float)
        if noise_delta.shape != (D,) or not np.all(np.isfinite(noise_delta)) or np.any(noise_delta < 0):
            raise ValueError("noise_delta must be finite nonnegative per-design RMS errors")
    p1d, p2d = jnp.array(designs[:, 0]), jnp.array(designs[:, 1])
    key, ka, kb, kc = jax.random.split(key, 4)
    # particle subsample for the inner sum + predictions of every candidate (D, N_INNER, T, 2)
    mus_in, s_in, fams = [], [], []
    for (cand, u) in fits:
        idx = jax.random.choice(ka, u.shape[0], (N_INNER,), replace=False)
        ui = u[idx]
        mus_in.append(predict_for(cand, D)(ui, p1d, p2d))
        s_in.append(core.noise_scale(ui, cand)); fams.append(cand.family)
    # outer samples: M_OUTER from the mixture.  Grouped by candidate and integrated in one pass.
    w = np.asarray(weights)
    rng_o = np.random.default_rng(int(jax.random.randint(kb, (), 0, 2**31)))
    comp = rng_o.choice(len(fits), M_OUTER, p=w)
    lp_true = np.zeros((D, M_OUTER)); lp_marg = np.zeros((D, M_OUTER))
    key_o = kc
    for cj in np.unique(comp):
        js = np.where(comp == cj)[0]
        cand_j, u_j = fits[int(cj)]
        key_o, k1, k2 = jax.random.split(key_o, 3)
        idx = jax.random.randint(k1, (len(js),), 0, u_j.shape[0])
        uj = u_j[idx]                                                   # (m,k)
        mu_j = predict_for(cand_j, D)(uj, p1d, p2d)                     # (D,m,T,2)
        sj = core.noise_scale(uj, cand_j)                               # (m,)
        eps = jax.random.normal(k2, (D, len(js), T_OBS, 2)) if cand_j.family == "gauss" \
              else jax.random.t(k2, 4.0, (D, len(js), T_OBS, 2))
        if noise_delta is None:
            y_j = mu_j + sj[None, :, None, None] * eps                  # legacy operation order
            sj_design = jnp.broadcast_to(sj, (D, len(js)))
        else:
            sj_design = fwd.inflated_noise_scale(sj[None, :], noise_delta[:, None], cand_j.family)
            y_j = mu_j + sj_design[:, :, None, None] * eps
        lp_true[:, js] = np.asarray(log_dens(y_j, mu_j, sj_design, cand_j.family))
        terms = []
        for ci, (cand_i, _) in enumerate(fits):
            si = s_in[ci][None, None, :] if noise_delta is None else fwd.inflated_noise_scale(
                s_in[ci][None, None, :], noise_delta[:, None, None], cand_i.family)
            ld = log_dens(y_j[:, :, None], mus_in[ci][:, None, :], si, cand_i.family)
            terms.append(logsumexp(ld, axis=2) - math.log(N_INNER) + math.log(max(w[ci], 1e-300)))
        lp_marg[:, js] = np.asarray(logsumexp(jnp.stack(terms, axis=2), axis=2))
    eig = (lp_true - lp_marg).mean(axis=1)
    # Record: per-design standard error.  The existing "eig_se" expression is unchanged (a pure reduction of the
    # same array).
    eig_se_per_design = (lp_true - lp_marg).std(axis=1) / math.sqrt(M_OUTER)
    return eig, {"eig_se": (lp_true - lp_marg).std(axis=1).max() / math.sqrt(M_OUTER),
                 "eig_se_per_design": eig_se_per_design}

def sample_epig_targets(run_seed, count=32):
    """Fixed uniform prediction targets from an independent, recorded seed stream."""
    if type(count) is not int or count < 1:
        raise ValueError("EPIG target count must be a positive integer")
    stream_id = 0x45504947
    rng = np.random.default_rng(np.random.SeedSequence([run_seed, stream_id]))
    points = np.column_stack([rng.uniform(*P1_RANGE, count), rng.uniform(*P2_RANGE, count)])
    return points, {"distribution": "uniform_rectangle", "count": count,
                    "run_seed": int(run_seed), "stream_id": stream_id,
                    "bit_generator": "PCG64", "p1_range": list(P1_RANGE),
                    "p2_range": list(P2_RANGE), "fixed_across_rounds": True}


def score_q_epig_targets():
    """FB1_EPIG_TARGETS=score_q: the scoring designs Q_HEAD then Q_SHIFT as EPIG targets (9 points).

    Design locations only, fixed for the run; nothing is drawn, so no random stream is touched.
    """
    points = np.asarray(Q_HEAD + Q_SHIFT, dtype=float)
    return points, {"distribution": "score_q", "count": int(len(points)), "points": points.tolist(),
                    "source": "Q_HEAD + Q_SHIFT in that order (the scoring designs of score_Q)",
                    "fixed_across_rounds": True}


def epig_target_points(run_seed, mode=None):
    """The run's EPIG targets and their record: FB1_EPIG_TARGETS (or `mode`) uniform -> sample_epig_targets."""
    mode = EPIG_TARGETS if mode is None else mode
    if mode == "uniform":
        return sample_epig_targets(run_seed)
    if mode == "score_q":
        return score_q_epig_targets()
    raise ValueError(f"EPIG target mode must be uniform or score_q, got {mode!r}")


def joint_epig(key, fits, weights, designs, target_points, noise_delta=None):
    """Mean-target I(Y(d);Y*(d*)) in nats, using the EIG outer/inner mixture.

    The joint mixes products at the same model/particle. Outer observations
    share their latent draw and have independent noise. Average target terms
    within each outer draw before computing its standard error; the SE is
    conditional on the fixed target/inner banks and excludes nested-MC bias.
    Numerical noise inflation belongs to the gated acquisition observation;
    fixed target observations use their declared predictive noise unchanged.
    """
    designs, target_points = np.asarray(designs, float), np.asarray(target_points, float)
    for points in (designs, target_points):
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or not np.all(np.isfinite(points)):
            raise ValueError("EPIG designs and target_points must be nonempty finite (N,2) arrays")
    D, target_count = len(designs), len(target_points)
    if noise_delta is not None:
        noise_delta = np.asarray(noise_delta, float)
        if noise_delta.shape != (D,) or not np.all(np.isfinite(noise_delta)) or np.any(noise_delta < 0):
            raise ValueError("noise_delta must be finite nonnegative per-design RMS errors")
    w = np.asarray(weights, float)
    if (not fits or w.shape != (len(fits),) or not np.all(np.isfinite(w))
            or np.any(w < 0) or not np.isclose(w.sum(), 1.0)):
        raise ValueError("EPIG weights must be a probability vector matching fits")
    p1d, p2d = jnp.asarray(designs[:, 0]), jnp.asarray(designs[:, 1])
    p1t, p2t = jnp.asarray(target_points[:, 0]), jnp.asarray(target_points[:, 1])
    key, ka, kb, kc = jax.random.split(key, 4)
    mus_in, targets_in, scales_in = [], [], []
    for cand, u in fits:
        idx = jax.random.choice(ka, u.shape[0], (N_INNER,), replace=False)
        ui = u[idx]
        mus_in.append(predict_for(cand, D)(ui, p1d, p2d))
        targets_in.append(predict_for(cand, target_count)(ui, p1t, p2t))
        scales_in.append(core.noise_scale(ui, cand))
    rng_o = np.random.default_rng(int(jax.random.randint(kb, (), 0, 2**31)))
    comp = rng_o.choice(len(fits), M_OUTER, p=w)
    contributions = np.zeros((D, M_OUTER))
    log_weights = [math.log(max(value, 1e-300)) - math.log(N_INNER) for value in w]
    key_o = kc
    for cj in np.unique(comp):
        js = np.flatnonzero(comp == cj)
        cand_j, u_j = fits[int(cj)]
        key_o, k1, k2 = jax.random.split(key_o, 3)
        idx = jax.random.randint(k1, (len(js),), 0, u_j.shape[0])
        uj = u_j[idx]
        mu_j = predict_for(cand_j, D)(uj, p1d, p2d)
        mu_target = predict_for(cand_j, target_count)(uj, p1t, p2t)
        sj = core.noise_scale(uj, cand_j)
        eps = (jax.random.normal(k2, (D, len(js), T_OBS, 2)) if cand_j.family == "gauss"
               else jax.random.t(k2, 4.0, (D, len(js), T_OBS, 2)))
        kt = jax.random.fold_in(k2, 0x45504947)
        eps_target = (jax.random.normal(kt, (target_count, len(js), T_OBS, 2)) if cand_j.family == "gauss"
                      else jax.random.t(kt, 4.0, (target_count, len(js), T_OBS, 2)))
        sj_design = (jnp.broadcast_to(sj, (D, len(js))) if noise_delta is None
                     else fwd.inflated_noise_scale(sj[None, :], noise_delta[:, None], cand_j.family))
        y = mu_j + sj_design[:, :, None, None] * eps
        y_target = mu_target + sj[None, :, None, None] * eps_target
        query_density, query_terms = [], []
        for ci, (cand_i, _) in enumerate(fits):
            si = (scales_in[ci][None, None, :] if noise_delta is None
                  else fwd.inflated_noise_scale(scales_in[ci][None, None, :],
                                                noise_delta[:, None, None], cand_i.family))
            ld = log_dens(y[:, :, None], mus_in[ci][:, None, :], si, cand_i.family)
            query_density.append(ld)
            query_terms.append(logsumexp(ld, axis=2) + log_weights[ci])
        log_query = logsumexp(jnp.stack(query_terms, axis=2), axis=2)
        for ti in range(target_count):
            joint_terms, target_terms = [], []
            for ci, (cand_i, _) in enumerate(fits):
                ld_target = log_dens(y_target[ti, :, None], targets_in[ci][ti, None, :],
                                     scales_in[ci][None, :], cand_i.family)
                joint_terms.append(logsumexp(query_density[ci] + ld_target[None], axis=2) + log_weights[ci])
                target_terms.append(logsumexp(ld_target, axis=1) + log_weights[ci])
            log_joint = logsumexp(jnp.stack(joint_terms, axis=2), axis=2)
            log_target = logsumexp(jnp.stack(target_terms, axis=1), axis=1)
            contributions[:, js] += np.asarray(log_joint - log_query - log_target[None]) / target_count
    epig = contributions.mean(axis=1)
    se = contributions.std(axis=1) / math.sqrt(M_OUTER)
    return epig, {"epig_se": float(se.max()), "epig_se_per_design": se}


# ---------------- Unified protocol: design-policy records (record fields only; no choice reads them) ----------------
DESIGN_RECORD_COLUMNS = ["p1", "p2", "joint_eig", "model_eig", "parameter_eig", "joint_eig_se", "model_eig_se",
                         "parameter_eig_se", "point_disagreement", "selection_objective"]
EIG_COMPONENTS_DEFINITION = {
    "joint_eig": "E[log p(y|theta,k,d) - log p(y|d)]: the runner's joint EIG (fb1_design_search.eig_pure / joint_eig)",
    "model_eig": ("E[log p_k(y|d) - log p(y|d)], k = the outer draw's own model, p_k = model k's predictive averaged over "
                  "its inner particles (MDA v4 Eq. 26; named structure_eig in the BoxingGym and NeuronBench EIG-decomposition records)"),
    "parameter_eig": "E[log p(y|theta,k,d) - log p_k(y|d)] (joint EIG minus Model EIG, draw by draw)",
    "draws": ("the same outer and inner Monte Carlo draws for the three (fb1_design_search.eig_parts_pure); on an ascent "
              "arm the round's independent holdout draws (fold_in(round key, HOLDOUT_FOLD)), the draws the choice was "
              "made with; on the random arm a record-only key (DESIGN_RECORD_SEED, run seed, n) that the run's key and "
              "rng never see"),
    "identity": "joint_eig = model_eig + parameter_eig per draw; identity_gap is the round-off of the means",
    "standard_error": "standard deviation of the per-outer-draw contributions / sqrt(m_outer)",
}
POINT_DISAGREEMENT_DEFINITION = (
    "sum_k w_k * mean_{t, axis} (m_k(d)[t, axis] - mbar(d)[t, axis])^2, where w_k are the evidence weights of the active "
    "models, m_k(d) is model k's posterior-mean trajectory at design d (the mean over all of its particles of the "
    "predicted position at each measurement time) and mbar(d) = sum_k w_k m_k(d); the mean runs over the T measurement "
    "times and both coordinates (units: position squared). Posterior means only (dispersion does not enter); the same "
    "rule as design_policies.point_disagreement per scalar coordinate. None when a model's mean is not finite.")


def point_disagreement(fits, weights, designs):
    """The point disagreement (POINT_DISAGREEMENT_DEFINITION) at each of `designs` (D,2), all particles."""
    designs = np.asarray(designs, float)
    D = len(designs)
    p1d, p2d = jnp.asarray(designs[:, 0]), jnp.asarray(designs[:, 1])
    means = np.stack([np.asarray(predict_for(cand, D)(u, p1d, p2d)).mean(axis=1) for cand, u in fits])
    return fds.point_disagreement_values(means, np.asarray(weights, float))


def _design_policy_block(rows_parts, disagreement, selection, chosen, evaluated, draws):
    rows = [[r["design"][0], r["design"][1], r["joint_eig"], r["model_eig"], r["parameter_eig"], r["joint_eig_se"],
             r["model_eig_se"], r["parameter_eig_se"], disagreement[i], None if selection is None else selection[i]]
            for i, r in enumerate(rows_parts)]
    joint = [row[2] for row in rows]
    block = {"policy": DESIGN_POLICY, "target": DESIGN_POLICY_TARGET[DESIGN_POLICY],
             "selection_column": {"joint_eig": "joint_eig", "model_eig": "model_eig", "parameter_eig": "parameter_eig",
                                  "epig": "epig", "random": None}[DESIGN_POLICY],
             "evaluated_designs": evaluated, "draws": draws, "rows_columns": list(DESIGN_RECORD_COLUMNS), "rows": rows,
             "chosen_index": int(chosen), "joint_eig_at_chosen": joint[chosen], "joint_eig_max": max(joint),
             "identity_gap_max": max(r["identity_gap"] for r in rows_parts),
             "point_disagreement_nonfinite": sum(value is None for value in disagreement)}
    if DESIGN_POLICY in ("joint_eig", "model_eig", "parameter_eig"):
        column = DESIGN_RECORD_COLUMNS.index(DESIGN_POLICY)
        block["selection_column_gap_max"] = max(abs(row[column] - row[-1]) for row in rows)
    return block


def design_policy_record_ascent(fits, weights, acq, chosen):
    """The rows of an ascent round: the endpoints, their three EIG parts (holdout draws) and point disagreement."""
    rows_parts = acq["eig_parts_holdout_rows"]
    disagreement = point_disagreement(fits, weights, [r["design"] for r in rows_parts])
    selection = [r["objective"] for r in acq["holdout_rows"]]
    return _design_policy_block(rows_parts, disagreement, selection, chosen, "ascent_endpoints", "independent_holdout")


def design_record_key(n):
    """The random arm's record-only key for round n (its own stream: the run's key and rng never see it)."""
    return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(DESIGN_RECORD_SEED), RUN_SEED), n)


def design_policy_record_random(fits, weights, design, n):
    """The random arm's one evaluated design (its draw): the three EIG parts on a record-only key and the disagreement."""
    design = np.asarray(design, float).reshape(1, 2)
    key = design_record_key(n)
    draws = fds.eig_draws(key, fits, weights, M_OUTER, N_INNER)
    parts = fds.eig_parts_pure(jnp.asarray(design), fits, weights, draws, lambda c, D: predict_for(c, D), log_dens, T_OBS)
    rows_parts = fds.eig_parts_rows(design, parts, M_OUTER)
    block = _design_policy_block(rows_parts, point_disagreement(fits, weights, design), None, 0, "random_draw",
                                 "record_only_key")
    block["record_key"] = [int(x) for x in np.asarray(jax.random.key_data(key)).ravel()]
    return block


# ---------------- Scoring ----------------
def _probe_me(fits, weights):
    """Structure-recovery ME -- relative force-probe deviation after canonicalization (constant refit), eps 0.02/0.05."""
    top = int(np.argmax(weights)); cand, u = fits[top]
    st, k = STRUCTURES[cand.struct], len(STRUCTURES[cand.struct].kinds)
    x = np.asarray(u_to_x(u, cand.kinds()))
    med = jnp.array(np.median(x, axis=0)[None, :k])
    lo, hi = CFG["me_probe"]
    r = jnp.linspace(lo, hi, 50)
    f_hat = np.asarray(st.force(r[None, :], jnp.array([[1.0]]), jnp.array([[1.0]]), med, 0.0))[0]  # t=0 probe
    f_true = np.asarray(CFG["me_decl"](np.asarray(r))) if CFG["me_decl"] else np.asarray(1.0/(2*math.pi*np.asarray(r)))
    ratio = f_hat / f_true
    alpha = float(np.sum(ratio) / np.sum(ratio**2)) if np.sum(ratio**2) > 0 else 0.0
    dev = float(np.max(np.abs(alpha * ratio - 1.0)))
    return {"top": cand.label(), "top_params_median": np.median(x, axis=0).tolist(),
            "refit_alpha": alpha, "max_rel_dev": dev,
            "me_eps002": bool(dev <= 0.02), "me_eps005": bool(dev <= 0.05)}

def _component_subsample(mus, ss, weights=None, families=None):
    """Record: particle subsample of each candidate's predictive mean trajectories.

    mus[i] (D,P,T,2) and ss[i] (P,) are arrays score_set has **already computed** -- the forward model is not called
    again.  The subsample uses a separate numpy generator with its own seed (it does not touch the main run's rng or
    jax key).  Returns mu (C,K,D,T,2) float32 -- (candidate, <=128, Q, T, 2).

    Added later: pred_std (per-candidate (D,T,2) standard
    deviation, all particles, double precision, paired with pred_mean) and summary (per candidate (mean, standard
    deviation, family, weight) -- the place kept in float64 in the checkpoint JSON; the npz's mu/noise subsamples stay
    at <=128 particles and float32 as originally agreed -- the three existing keys of this npz file
    (mu_*/pred_mean_*/noise_scale_*) do not change.
    `weights`/`families` are **optional** (backward compatibility for callers that pass mus/ss only); when omitted, summary's
    family/weight stay None.  The real call in score_set() always passes both -- the component_summary of the
    checkpoints a run produces is always complete.
    """
    weights = [None] * len(mus) if weights is None else weights
    families = [None] * len(mus) if families is None else families
    P = min(int(m.shape[1]) for m in mus)
    K = min(PRED_COMPONENT_MAX_PARTICLES, P)
    if K < P:
        idx = np.sort(np.random.default_rng(PRED_COMPONENT_SUBSAMPLE_SEED).choice(P, K, replace=False))
    else:
        idx = np.arange(P)
    mu = np.stack([np.asarray(m)[:, idx].transpose(1, 0, 2, 3) for m in mus]).astype(np.float32)
    # ---- Per-candidate **all-particle** posterior predictive mean (C,D,T,2) float64.
    #      The mu above is a <=128-particle subsample in float32, so a consumer could not recover each candidate's
    #      exact posterior predictive mean from it.  This array fills that gap --
    #      sum_i weights[i] * pred_mean[i] must equal the BMA pred_mean of the JSON (tested).
    #      Record-only: does not enter the fit, EIG, design or scoring formulas.
    pred_mean = np.stack([np.asarray(m).mean(axis=1) for m in mus]).astype(np.float64)
    pred_std = np.stack([np.asarray(m).std(axis=1) for m in mus]).astype(np.float64)   # per-candidate standard deviation
    s_all = [np.asarray(s) for s in ss]
    summary = [                                                                        # per-candidate summary
        {"family": (None if families[c] is None else str(families[c])),
         "weight": (None if weights[c] is None else float(weights[c])),
         "mean": pred_mean[c].tolist(), "std": pred_std[c].tolist()}
        for c in range(len(mus))
    ]
    return {"mu": mu, "pred_mean": pred_mean, "pred_std": pred_std,
            "particle_index": idx.astype(np.int32),
            "noise_scale_median": np.array([float(np.median(s)) for s in s_all], np.float32),
            "noise_scale_particles": np.stack([s[idx] for s in s_all]).astype(np.float32),
            "n_particles_total": int(P), "summary": summary}


def write_kl_full_particle_mu(fpath, labels, wts, full_h, full_s,
                              node_x_h, node_lb_h, node_x_s, node_lb_s,
                              max_bytes=None):
    """Final checkpoint only: stores the all-particle (1000), double-precision mu in a separate npz.

    If the estimated size (candidates x particles x (head+shift scalars) x 8 bytes) exceeds `max_bytes` (default
    KL_FULL_MU_NPZ_MAX_BYTES=20MB), the KL quadrature node values (node_x_*/node_lb_*, whose size is tied to n_nodes
    rather than to the particle count) are stored instead -- either way a file is actually written, and what was
    replaced and why is returned as a dict (for disclosure; it goes into checkpoints[n].kl.full_particle_mu as is).
    main() and the tests (both branches, on small synthetic inputs) call it alike.
    """
    max_bytes = KL_FULL_MU_NPZ_MAX_BYTES if max_bytes is None else max_bytes
    n_cand = full_h["mu"].shape[0]; n_particles = full_h["mu"].shape[2]
    n_head_dt = full_h["mu"].shape[1] * full_h["mu"].shape[3] * full_h["mu"].shape[4]
    n_shift_dt = full_s["mu"].shape[1] * full_s["mu"].shape[3] * full_s["mu"].shape[4]
    est_bytes = int(n_cand * n_particles * (n_head_dt + n_shift_dt) * 8)
    if est_bytes <= max_bytes:
        np.savez_compressed(fpath,
            candidates=np.asarray(labels), weights=np.asarray(wts, float),
            sigma_world=np.float64(SIGMA),
            mu_head=full_h["mu"], mu_shift=full_s["mu"],
            noise_scale_head=full_h["s"], noise_scale_shift=full_s["s"],
            n_particles_total=np.int64(n_particles))
        return {"path": os.path.basename(fpath), "written": True,
                "bytes_estimated": est_bytes, "truncated": False,
                "n_particles_total": int(n_particles), "n_candidates": int(n_cand)}
    np.savez_compressed(fpath,
        candidates=np.asarray(labels), weights=np.asarray(wts, float),
        sigma_world=np.float64(SIGMA),
        nodes_x_head=node_x_h, nodes_log_bma_head=node_lb_h,
        nodes_x_shift=node_x_s, nodes_log_bma_shift=node_lb_s)
    return {"path": os.path.basename(fpath), "written": True,
            "bytes_estimated": est_bytes, "truncated": True,
            "truncation_reason": (
                f"estimated full-particle array {est_bytes} bytes exceeds the "
                f"{max_bytes}-byte cap; stored the KL quadrature node locations and BMA "
                "log-density values at those nodes instead (size is bounded by n_nodes, "
                "not particle count)")}


def score_set(fits, weights, Q, truths, y_held, key, compute_kl=True, return_full_particles=False):
    """Scores the BMA of `fits` on designs Q: nMSE, NLL per observation, 90% coverage and width, per-coordinate KL."""
    p1d, p2d = jnp.array([q[0] for q in Q]), jnp.array([q[1] for q in Q])
    D = len(Q)
    with Clock("forward_predict"):
        mus, ss = [], []
        for (cand, u) in fits:
            mus.append(predict_for(cand, D)(u, p1d, p2d)); ss.append(core.noise_scale(u, cand))
        jax.block_until_ready(mus[-1])
    with Clock("scoring"):
        w = jnp.array(weights)
        pred_mean = sum(w[i] * mus[i].mean(axis=1) for i in range(len(fits)))       # (D,T,2)
        truth = jnp.array(truths)
        nmse = float(jnp.mean((pred_mean - truth) ** 2) / VAR_GT)
        yh = jnp.array(y_held)
        terms = [log_dens(yh[:, None], mus[i], ss[i][None, :], fits[i][0].family)
                 - math.log(mus[i].shape[1]) for i in range(len(fits))]
        lse = [logsumexp(t, axis=1) + jnp.log(w[i]) for i, t in enumerate(terms)]
        nll = float(-(logsumexp(jnp.stack(lse, 1), axis=1).sum()) / (D * T_OBS * 2))
        # 90% interval: sampled from the mixture
        k1, k2, k3 = jax.random.split(key, 3)
        S = 2000
        ci = np.random.default_rng(0).choice(len(fits), S, p=np.asarray(w))
        pi = np.asarray(jax.random.randint(k1, (S,), 0, mus[0].shape[1]))
        samp = np.empty((S, D, T_OBS, 2))
        eps = np.asarray(jax.random.normal(k2, (S, D, T_OBS, 2)))
        for s in range(S):
            c, p = int(ci[s]), int(pi[s] % mus[int(ci[s])].shape[1])
            samp[s] = np.asarray(mus[c][:, p]) + float(ss[c][p]) * eps[s]
        lo, hi = np.quantile(samp, 0.05, axis=0), np.quantile(samp, 0.95, axis=0)
        yhn = np.asarray(y_held)
        cover = float(np.mean((yhn >= lo) & (yhn <= hi))); width = float(np.mean(hi - lo))
    # ---- Record: only reductions of the mixture sample samp (S=2000) built above -- no new random draws ----
    pred_sd = samp.std(axis=0)
    # ---- Per-coordinate KL(truth||BMA).  mus/ss are the all-particle (1000), double-precision arrays already
    #      computed above -- no recomputation or resampling, no new random draws.  Single top-model scoring (whose
    #      _keep does not hold "kl") calls with compute_kl=False to skip this cost (its result would be discarded).
    kl = None
    if compute_kl:
        with Clock("scoring"):
            kl = score_set_kl(mus, ss, np.asarray(weights, dtype=np.float64),
                              np.asarray([c.family for c, _ in fits]),
                              np.asarray(truths, dtype=np.float64), SIGMA,
                              return_nodes=return_full_particles)
    # ---- So that main() can write the all-particle mu npz (or, above 20MB, the node-value replacement) at the final
    #      checkpoint only, the all-particle double-precision arrays are passed on as they are (before subsampling
    #      and float32 conversion).
    full_particles = None
    if return_full_particles:
        full_particles = {
            "mu": np.stack([np.asarray(m, dtype=np.float64) for m in mus]),
            "s": np.stack([np.asarray(s, dtype=np.float64) for s in ss]),
            "families": np.asarray([c.family for c, _ in fits]),
        }
    return {"nmse_mda": nmse, "nll_per_obs": nll, "coverage90": cover, "interval_width90": width,
            "pred_mean": np.asarray(pred_mean).tolist(),
            "pred_sd": pred_sd.tolist(), "pred_lo90": lo.tolist(), "pred_hi90": hi.tolist(),
            "kl": kl,
            "_components": _component_subsample(mus, ss, weights, [c.family for c, _ in fits]),
            "_full_particles": full_particles}

# ---------------- Misspecification exposure diagnostic (evaluator only; not used for inference) ----------------
def _integrate_true(p1, p2, dt=0.005):
    """Feeds the world's **declared true law** to our integrator (true-law reference candidate).  numpy, because of
    the Bessel function and sums."""
    f, inert, soft = CFG["true_force"], CFG["inertia"], CFG["soft"]
    m = p2 if inert == "p2" else 1.0
    x = np.array(np.asarray(POS0), float); v = np.array(np.asarray(VEL0), float)
    def acc(x):
        """Acceleration at position x under the declared true force (softened radius, inertia m)."""
        rn = np.linalg.norm(x); rs = math.sqrt(rn * rn + soft * soft)
        return -float(f(rs, p1, p2)) / m * (x / rn)
    t, out, a = 0.0, [], acc(x)
    for tq in TIMES:
        for _ in range(int(round((tq - t) / dt))):
            v = v + 0.5 * dt * a; x = x + dt * v; a = acc(x); v = v + 0.5 * dt * a
        t = tq; out.append(x.copy())
    return np.asarray(out)

def exposure(world_clean, designs):
    """Gap between the true-law free-space model and the true world at the chosen designs.  Driving variable =
    acceleration scale (p1/p2)."""
    rows = []
    for p1, p2 in designs:
        tr = world_clean.truth(p1, p2)
        ours = _integrate_true(p1, p2)
        r = np.linalg.norm(tr, axis=-1)
        rows.append({"p1": float(p1), "p2": float(p2), "accel_scale": float(p1 / p2),
                     "r_min": float(r.min()), "r_max": float(r.max()),
                     "max_dev_sigma": float(np.linalg.norm(ours - tr, axis=-1).max() / SIGMA),
                     "gap_nats": float(np.sum((ours - tr) ** 2) / (2 * SIGMA ** 2))})
    a = np.array([x["accel_scale"] for x in rows]); g = np.array([x["gap_nats"] for x in rows])
    rk = lambda v: np.argsort(np.argsort(v))
    return {"per_design": rows, "gap_nats_total": float(g.sum()),
            "accel_scale_mean": float(a.mean()), "accel_scale_max": float(a.max()),
            "r_min_mean": float(np.mean([x["r_min"] for x in rows])),
            "r_min_min": float(np.min([x["r_min"] for x in rows])),
            "max_dev_sigma_max": float(np.max([x["max_dev_sigma"] for x in rows])),
            "spearman_accel_vs_gap": float(np.corrcoef(rk(a), rk(g))[0, 1])}

# ---------------- Code fingerprints and versions (record only) ----------------
def _code_records():
    """Source sha256 at run time and venv versions.  No numerical value depends on them.

    Sibling modules are hashed from **the files actually imported** (module.__file__), not from the runner directory.
    This fixes a defect where chained snapshot runs, which put only a copy of the runner in another directory, died
    with a Traceback because the sibling files were missing -- in a live-directory run the four paths are all the
    same, so the recorded sha values do not change.  Only the runner itself is taken from __file__.
    """
    import platform, jaxlib, scipy
    src = {"run_gravity.py": __file__, "fb1_core.py": core.__file__,
           "fb1_forward.py": fwd.__file__, "world_constants.py": wc.__file__}
    if ASCENT or DESIGN_POLICY is not None:   # design search (and the random arm's design-policy records)
        src["fb1_design_search.py"] = fds.__file__
    if GRAMMAR_TRIG:   # likewise the module that widens the grammar, only when it is on
        src["fb1_grammar_trig.py"] = fbg_trig.__file__
    if BURST or COMPOSED:   # unified protocol: the request kinds live in the proposer, only when one is on
        src["fb1_proposer.py"] = fbp.__file__
    sha = {}
    for name, path in src.items():
        with open(os.path.abspath(path), "rb") as fh:
            sha[name] = hashlib.sha256(fh.read()).hexdigest()
    ver = {"python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "jax": jax.__version__, "jaxlib": jaxlib.__version__,
           "scipy": scipy.__version__, "jax_enable_x64": bool(jax.config.jax_enable_x64),
           "jax_default_backend": jax.default_backend()}
    return sha, ver


def _input_provenance():
    """Record external code and the actual frozen inputs, without changing inference."""
    import subprocess
    dp_path = os.path.realpath(DP_SCIENCEAGENT_DIR)
    try:
        commit = subprocess.run(
            ["git", "-C", dp_path, "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "not_a_git_checkout"
    inputs = {
        "FB1_FIXED_IC_MANIFEST_v5.json": IC_MANIFEST,
        "FB1_FORCE_CURVE_WORLD_RULING.json": wc.RULING_PATH,
        "FB1_FORCE_CURVE_yukawa_amplitude.json": wc.YUKAWA_AMPLITUDE_PATH,
    }
    hashes = {}
    for name, path in inputs.items():
        with open(path, "rb") as handle:
            hashes[name] = hashlib.sha256(handle.read()).hexdigest()
    return {"dp_scienceagent_dir": dp_path, "dp_git_commit": commit,
            "frozen_inputs_sha256": hashes}


# ---------------- Belief-packet assembly, prior probe, LLM proposals (llm path only) ----------------
# The functions below receive **public state only**.  The held-out Q (TR_H/YH_H/TR_S/YH_S), truths and score objects
# are local variables of main() and appear neither in these functions' arguments nor in their bodies -- a test checks
# this statically.
def _public_predictions(cand, u, p1a, p2a):
    """Posterior predictive mean and sd of one candidate at the public designs, and the acceleration magnitude.
    Uses public data only.

    pred_sd is the **posterior predictive** sd: coefficient uncertainty Var_p(mu) + observation noise E_p[Var(y|mu,s)]
    (gauss = s^2, student_t4 = nu/(nu-2) s^2 = 2 s^2).  The packet's definition block carries this sentence.
    """
    D = int(p1a.shape[0])
    mu = np.asarray(predict_for(cand, D)(u, p1a, p2a))            # (D,P,T,2)
    s = np.asarray(core.noise_scale(u, cand))                     # (P,)
    pred_mean = mu.mean(axis=1)
    noise_var = float(np.mean(s ** 2)) * (1.0 if cand.family == "gauss" else 2.0)
    pred_sd = np.sqrt(mu.var(axis=1) + noise_var)
    x = np.asarray(u_to_x(u, cand.kinds()))
    k = len(STRUCTURES[cand.struct].kinds)
    med = jnp.array(np.median(x, axis=0)[None, :k])
    r = np.linalg.norm(pred_mean, axis=-1)                        # (D,T)
    f = np.asarray(STRUCTURES[cand.struct].force(
        jnp.array(r), jnp.array(np.asarray(p1a))[:, None], jnp.array(np.asarray(p2a))[:, None],
        med, jnp.array(TIMES)[None, :]))
    accel = np.abs(np.broadcast_to(f, r.shape)) / np.asarray(p2a)[:, None]
    return {"pred_mean": pred_mean, "pred_sd": pred_sd, "accel_abs": accel}


def _prior_probe(struct, family, mid, p1a, p2a):
    """Prior finiteness probe -- checks **before** the fit whether the forward model is finite on role-prior samples.

    Random numbers are split from a dedicated key (`PRIOR_PROBE_SEED` ^ model_id) -- the main run's numpy stream,
    SMC keys and design stream are not consumed (the byte-identity anchor of the default mock path proves this).
    `normalization_warning` is **not defined for a continuous likelihood**: the gauss/student_t4 densities are
    analytically normalized, so nothing corresponds to CORE's PMF-sum check with tolerance 1e-10.
    The ForceBench counterpart is this finiteness probe itself, and its result is nonfinite_fraction.
    """
    cand = Candidate(struct, family, 0)
    kinds = cand.kinds()
    key = jax.random.fold_in(jax.random.PRNGKey(PRIOR_PROBE_SEED), int(mid[:8], 16))
    u = core.sample_prior_u(key, PRIOR_PROBE_SAMPLES, kinds)
    mu = np.asarray(predict_for(cand, int(p1a.shape[0]))(u, p1a, p2a))
    s = np.asarray(core.noise_scale(u, cand))
    finite = bool(np.isfinite(mu).all() and np.isfinite(s).all() and (s > 0).all())
    nonfinite = float(1.0 - np.isfinite(mu).mean())
    return {"n_samples": PRIOR_PROBE_SAMPLES, "n_designs": int(p1a.shape[0]),
            "nonfinite_fraction": nonfinite, "finite": finite,
            "normalization_warning": 0,
            "normalization_warning_reason":
                "not defined for a continuous likelihood: gauss/student_t4 densities are "
                "analytically normalized, so CORE's PMF sum tolerance check has no analogue; "
                "the ForceBench analogue is this finiteness probe (nonfinite_fraction)"}


def _pool_members(fits, weights, log_evidence, one_step, round_t, n_designs, best_predictions):
    """Active pool -> the list of `PoolMember` the packet sees (public state only)."""
    members = []
    top = int(np.argmax(np.asarray(weights)))
    for i, (cand, u) in enumerate(fits):
        x = np.asarray(u_to_x(u, cand.kinds()))
        # Free structures are not in COEFFICIENT_NAMES -- use the lookup that checks the registry first.
        names = fbp.structure_coefficient_names(cand.struct)
        k = len(names)
        history = one_step.get(cand.label(), {})
        members.append(fbp.PoolMember(
            cand.struct, cand.family, float(weights[i]), float(log_evidence[i]),
            log_evidence_computed_round=int(round_t), log_evidence_n_obs=int(n_designs),
            born_round=int(max(cand.born_step - D0, 0)),
            coefficients={names[j]: {"mean": float(x[:, j].mean()), "sd": float(x[:, j].std(ddof=1))}
                          for j in range(k)},
            family_dispersion={fbp.FAMILY_DISPERSION_NAME: {
                "mean": float(x[:, k].mean()), "sd": float(x[:, k].std(ddof=1))}},
            one_step_log_scores=[history.get(r) for r in range(1, round_t + 1)],
            one_step_rounds=list(range(1, round_t + 1)),
            predictions=best_predictions if i == top else None))
    return members


def _observation_rows(designs, ys, best_predictions, integration_steps):
    """Public rows per design.  **Raw trajectories are not put in the packet** -- y is used only for residual summaries.

    `best_predictions` None is the round-0 burst packet's case (no model yet; no prediction to check)."""
    rows = []
    pos0 = [float(v) for v in np.asarray(POS0)]
    vel0 = [float(v) for v in np.asarray(VEL0)]
    for j, (p1, p2) in enumerate(designs):
        y = np.asarray(ys[j], float)
        status = "ok"
        if not np.isfinite(y).all():
            status = "nonfinite_observation"
        elif best_predictions is not None and not np.isfinite(best_predictions["pred_mean"][j]).all():
            status = "nonfinite_prediction"
        rows.append({"design_id": f"d{j:02d}", "p1": float(p1), "p2": float(p2),
                     "initial_state": {"pos": pos0, "vel": vel0},
                     "integration_steps": int(integration_steps),
                     "integration_status": status, "y": y})
    return rows


def _admit_llm_candidate(cand, note, *, round_t, have, p1a, p2a):
    """One parsed candidate through the admission rules (moved here unchanged from _llm_propose for the unified protocol).

    Returns (archive entry, admitted (structure, family) key or None). The caller appends the entry and, for an admitted
    key, adds it to `have` and to its admitted pairs - the order every earlier run used.
    """
    key = (cand["structure"], cand["family"])
    entry = {"model_id": cand["model_id"], "expression": cand["expression"],
             "family": cand["family"], "proposed_round": round_t,
             "note": note}
    if key in have:
        entry.update(outcome="duplicate",
                     reason={"matched_model_id": cand["model_id"],
                             "matched_expression": cand["expression"]})
        return entry, None
    if FREE_STRUCTURES:
        # ---- Free-expression admission.  The order is
        #      **finiteness probe** (reject immediately if non-finite) -> canonical identifier -> numerical equivalence
        #      -> force-magnitude cap -> registration in fb1_core.STRUCTURES.  fb1_grammar does all of it, and the
        #      reason sentences and codes are carried over exactly as that module wrote them.  There are four outcomes:
        #      duplicate (same as a registered free structure), rejected (probe),
        #      catalog_reference (same prior model as a catalogue law -> linked by name, see below), active.
        #      Why this sits on the **admission path** rather than the judge path: a rejection here does not trigger
        #      propose_one's repair re-call (a grammatically correct answer is not asked for a repair -- the
        #      catalogue path's probe sits at the same place).
        _name, _grec = fbg.admit(cand["expression"], cand["parameters"], TIMES, round=round_t)
        entry["reason_code"] = _grec.get("reason_code")
        for _k in ("prior_probe", "numerical_equivalence"):
            if _k in _grec:
                entry[_k] = _grec[_k]
        if _grec["outcome"] == "duplicate":
            _matched = _grec["matched_structure"]
            entry.update(outcome="duplicate",
                         reason={"matched_model_id": _matched,
                                 "matched_expression": fbp.structure_expression(_matched)})
            return entry, None
        if _grec["outcome"] == "rejected":
            entry.update(outcome="rejected",
                         reason={"validation_message": _grec["reason"]})
            return entry, None
        if _grec["outcome"] == fbg.OUTCOME_CATALOG_REFERENCE:
            # ---- The catalogue is a **reference table**.  If a free proposal is the same prior model as a
            #      catalogue law, it is linked to that catalogue name (`_name`), and rejection is decided only by
            #      `_key in have` below -- only when that (structure, family) pair is already in the active
            #      pool or in the evaluated history (`archive_pool_keys`).
            #      A pair not yet evaluated proceeds as a candidate, and since it is not registered as a free
            #      structure it is not counted as a "new structure discovery" (`fbg.REGISTRY`).
            entry["catalog_reference"] = dict(
                _grec["catalog_link"],
                free_expression_as_proposed=cand["expression"],
                catalog_expression=fbp.structure_expression(_name))
        _key = (_name, cand["family"])
        if _key in have:
            entry.update(outcome="duplicate",
                         reason={"matched_model_id": _name,
                                 "matched_expression": fbp.structure_expression(_name)})
            return entry, None
        if _grec["outcome"] == fbg.OUTCOME_CATALOG_REFERENCE:
            #      Only an **admitted** catalogue reference has its identity fields switched to those of the
            #      catalogue candidate -- so that the next round's history lookup finds this candidate again under
            #      the same model_id.  A rejected one (`_key in have` above) keeps the identity of the proposed free
            #      expression: that pair is already in the archive as an active entry, so changing the identity
            #      would let `llm_archive.setdefault` swallow this duplicate record and it would vanish from the
            #      rejection-reason accounting.
            entry["expression"] = fbp.structure_expression(_name)
            entry["model_id"] = fbp.model_id(entry["expression"], cand["family"],
                                             fbp.structure_roles(_name))
        entry.update(outcome="active",
                     reason={"admitted_round": round_t, "active_rounds": [round_t]})
        return entry, _key
    probe = _prior_probe(cand["structure"], cand["family"], cand["model_id"], p1a, p2a)
    entry["prior_probe"] = probe
    if not probe["finite"]:
        entry.update(outcome="rejected",
                     reason={"validation_message": "non-finite under prior probe"})
        return entry, None
    entry.update(outcome="active",
                 reason={"admitted_round": round_t, "active_rounds": [round_t]})
    return entry, key


def _refused_entries(rejected, *, round_t, id_preimage, engine_text):
    """Archive rows of refused replies or array objects: `id_preimage(k)` names the k-th row's identifier preimage."""
    return [{"model_id": hashlib.sha256(id_preimage(k).encode()).hexdigest()[:20],
             "expression": None, "family": None, "proposed_round": round_t,
             "outcome": "rejected",
             "reason": {"validation_message": engine_text(item["reason"])}}
            for k, item in enumerate(rejected)]


def _top_by_evidence(log_evidence, maximum):
    """Unified protocol: (kept, dropped) indices, each in pool order. Kept = the `maximum` largest log evidences,
    ties to the earlier position; a non-finite value ranks last."""
    rank = sorted(range(len(log_evidence)),
                  key=lambda i: (-(log_evidence[i] if math.isfinite(log_evidence[i]) else -math.inf), i))
    return sorted(rank[:maximum]), sorted(rank[maximum:])


#: record-only archive keys of a revival; `fb1_proposer._archive_entry` does not copy them into packets
REVIVAL_RECORD_KEYS = ("resurrected_rounds", "revival_attempts", "revival_history")


def _archive_retirement(archive, cand, log_evidence, cutoff, n):
    """The retired_by_evidence archive row of `cand`, written as the main loop's retirement code writes it (same
    keys and reason); returns its model_id. A model revived before keeps its record-only revival keys."""
    expression = fbp.structure_expression(cand.struct)
    mid = fbp.model_id(expression, cand.family, fbp.structure_roles(cand.struct))
    prev = archive.get(mid, {})
    archive[mid] = {"model_id": mid, "expression": expression, "family": cand.family,
                    "proposed_round": int(prev.get("proposed_round", max(cand.born_step - D0, 0))),
                    "outcome": "retired_by_evidence",
                    "reason": {"log_evidence_at_retirement": float(log_evidence),
                               "gap_to_cutoff_nats": float(cutoff - log_evidence),
                               "active_rounds": list(prev.get("reason", {}).get("active_rounds", [])),
                               "computed_round": n - D0, "n_obs": n}}
    for key in REVIVAL_RECORD_KEYS:
        if key in prev:
            archive[mid][key] = prev[key]
    return mid


def _archive_resurrection(archive, mid, round_t):
    """A revived model that re-entered the pool at `round_t` is active again in the archive
    (the packet's reason keys of an active row); `resurrected_rounds` and the other revival keys are record-only."""
    prev = archive[mid]
    archive[mid] = {"model_id": mid, "expression": prev["expression"], "family": prev["family"],
                    "proposed_round": int(prev["proposed_round"]), "outcome": "active",
                    "reason": {"admitted_round": int(prev["proposed_round"]),
                               "active_rounds": sorted(set(prev["reason"].get("active_rounds", [])) | {round_t})}}
    for key in REVIVAL_RECORD_KEYS:
        if key in prev:
            archive[mid][key] = prev[key]
    archive[mid]["resurrected_rounds"] = list(prev.get("resurrected_rounds", [])) + [round_t]


def _event_accounting(calls, additions):
    return {
        "llm_calls": sum(c["calls"] for c in calls),
        "llm_repairs": sum(c["repairs"] for c in calls),
        "paid_api_calls": sum(c["paid_api_calls"] for c in calls),
        "valid_candidates": sum(c["valid_candidate_count"] for c in calls),
        "duplicates": sum(1 for a in additions if a["outcome"] == "duplicate"),
        "rejected_candidates": sum(1 for a in additions if a["outcome"] == "rejected"),
        "prior_probe_rejections": sum(
            1 for a in additions
            if a["outcome"] == "rejected" and "prior_probe" in a),
        "normalization_warnings": 0}


def _integration_steps():
    return sum(int(round((float(tq) - (0.0 if i == 0 else float(TIMES[i - 1]))) / DT))
               for i, tq in enumerate(TIMES))


def _llm_propose(proposer, *, n_designs, round_t, event_index, designs, ys, fits, weights,
                 log_evidence, one_step, archive_rows, pool_keys, constraints, interface,
                 packets_dir, run_id, world, arm, policy, candidates_per_event):
    """One proposal event.  Builds the packet, receives candidates `candidates_per_event` times and judges them.

    Returns (admitted_pairs, event_record, archive_additions).  Duplicates, rejections and probe failures are not
    admitted; they remain only as archive entries with structured reasons.
    """
    p1a = jnp.array([d[0] for d in designs]); p2a = jnp.array([d[1] for d in designs])
    top = int(np.argmax(np.asarray(weights)))
    best_pred = _public_predictions(fits[top][0], fits[top][1], p1a, p2a)
    steps = _integration_steps()
    rows = _observation_rows(designs, ys, best_pred, steps)
    members = _pool_members(fits, weights, log_evidence, one_step, round_t, n_designs, best_pred)
    have = set(pool_keys)
    pairs, additions, calls = [], [], []
    # With input-name masking the archive is kept in the engine's names and masked again when the next
    # packet is written, so text the model itself read or wrote (a validation message, a
    # note) is swapped back here; the swap is its own inverse. Without masking: identity.
    _engine_text = (proposer.name_mask.swap_names if proposer.name_mask is not None
                    else (lambda value: value))
    for j in range(candidates_per_event):
        extra = [
            {"model_id": item["model_id"], "expression": item["expression"],
             "family": item["family"], "proposed_round": round_t, "outcome": "active",
             "reason": {"admitted_round": round_t, "active_rounds": [round_t]}}
            for item in additions if item["outcome"] == "active"
        ]
        packet = _catalogue_view(fbp.build_belief_packet(rows, members, list(archive_rows) + extra,
                                                         constraints, interface, round_t))
        if proposer.name_mask is not None:
            # The model reads X1..X4; the packet file holds exactly what was sent.
            packet = proposer.name_mask.mask_value(packet)
        text = fbp.serialize_packet(packet)
        packet_file = f"t{n_designs:02d}_c{j}.json"
        with open(os.path.join(packets_dir, packet_file), "x") as fh:
            fh.write(text)
        context = {"run_id": run_id, "world": world, "arm": arm, "policy": policy,
                   "event": event_index, "round": round_t, "n_designs": n_designs,
                   "candidate_index": j, "packet_file": packet_file}
        got, record = proposer.propose_one(text, context=context, constraints=constraints)
        calls.append(record)
        if not got:
            additions.extend(_refused_entries(record["rejected_candidates"], round_t=round_t, engine_text=_engine_text,
                                              id_preimage=(lambda k, j=j: f"failed|{n_designs}|{j}|{k}")))
            continue
        entry, admitted_key = _admit_llm_candidate(got[0], _engine_text(record.get("note", "")), round_t=round_t,
                                                   have=have, p1a=p1a, p2a=p2a)
        additions.append(entry)
        if admitted_key is not None:
            have.add(admitted_key)
            pairs.append(admitted_key)
    event = {"n_designs": n_designs, "round": round_t, "event": event_index,
             "policy": policy, "recovery_mode": proposer.recovery_mode,
             "calls": calls, "proposed": [list(p) for p in pairs],
             "normalization_warnings": 0,
             "accounting": _event_accounting(calls, additions)}
    return pairs, event, additions


def _packet_archive(rows):
    """The archive rows a packet carries - as they are, or (FB1_PACKET_ARCHIVE=compact) without the two
    diagnostic blocks, which make up most of a free-form row's bytes. The run's own archive keeps every key."""
    rows = list(rows)
    if PACKET_ARCHIVE == "full":
        return rows
    return [{k: v for k, v in row.items() if k not in PACKET_ARCHIVE_DROPPED_KEYS} for row in rows]


def _catalogue_view(packet):
    """The packet as the model reads it under FB1_CATALOGUE - unchanged when on, and
    without the catalogue's traces in its pool and archive when off (fb1_proposer.catalogue_off_packet)."""
    return fbp.catalogue_off_packet(packet) if CATALOGUE == "off" else packet


def _llm_propose_composed(proposer, *, n_designs, round_t, event_index, designs, ys, fits, weights,
                          log_evidence, one_step, archive_rows, pool_keys, constraints, interface,
                          packets_dir, run_id, world, arm, policy, review=False):
    """One proposal round = ONE call ("composed", or "composed_review" on an archive review
    round) on one belief packet built from the current pool, data and archive, answering three candidates. Each accepted
    candidate is admitted in reply order by `_admit_llm_candidate`, `have` holding the pool, the archive and the reply's
    earlier admissions, so an exact duplicate is recorded (outcome "duplicate", not refitted) and nothing replaces it; each
    array object the judge refused becomes a rejected archive row. Returns (admitted pairs, event, archive additions,
    revive request) - the revive request is the review reply's list as written and in the engine's identifiers (empty
    lists off a review round); the caller validates it, fits, and cuts the pool.
    """
    p1a = jnp.array([d[0] for d in designs]); p2a = jnp.array([d[1] for d in designs])
    top = int(np.argmax(np.asarray(weights)))
    best_pred = _public_predictions(fits[top][0], fits[top][1], p1a, p2a)
    rows = _observation_rows(designs, ys, best_pred, _integration_steps())
    members = _pool_members(fits, weights, log_evidence, one_step, round_t, n_designs, best_pred)
    packet = _catalogue_view(fbp.build_belief_packet(rows, members, _packet_archive(archive_rows), constraints,
                                                     interface, round_t))
    if proposer.name_mask is not None:
        packet = proposer.name_mask.mask_value(packet)
    text = fbp.serialize_packet(packet)
    packet_file = f"t{n_designs:02d}_c0.json"
    with open(os.path.join(packets_dir, packet_file), "x") as fh:
        fh.write(text)
    context = {"run_id": run_id, "world": world, "arm": arm, "policy": policy,
               "event": event_index, "round": round_t, "n_designs": n_designs,
               "candidate_index": 0, "packet_file": packet_file}
    request = "composed_review" if review else "composed"
    got, record = proposer.propose_one(text, context=context, constraints=constraints, request=request,
                                       proposal_limit=COMPOSED_LIMIT)
    _engine_text = (proposer.name_mask.swap_names if proposer.name_mask is not None
                    else (lambda value: value))
    have = set(pool_keys)
    pairs, additions = [], []
    if not got:
        additions.extend(_refused_entries(record["rejected_candidates"], round_t=round_t, engine_text=_engine_text,
                                          id_preimage=(lambda k: f"failed|{n_designs}|0|{k}")))
    else:
        # the array's refused objects carry an index; the refusals of a repaired first attempt do not
        additions.extend(_refused_entries([item for item in record["rejected_candidates"] if "index" in item],
                                          round_t=round_t, engine_text=_engine_text,
                                          id_preimage=(lambda k: f"refused|{n_designs}|{k}")))
        for cand, note in zip(got, record.get("notes") or [""] * len(got)):
            entry, admitted_key = _admit_llm_candidate(cand, _engine_text(note), round_t=round_t,
                                                       have=have, p1a=p1a, p2a=p2a)
            additions.append(entry)
            if admitted_key is not None:
                have.add(admitted_key)
                pairs.append(admitted_key)
    revive = {"requested": list(record.get("revive_requested", [])),
              "requested_engine_ids": list(record.get("revive_requested_engine_ids",
                                                      record.get("revive_requested", []))),
              # A bare array (or an object without "revive") on a review round
              "missing": not record.get("revive_present", False), "reply_shape": record.get("reply_shape")}
    event = {"n_designs": n_designs, "round": round_t, "event": event_index,
             "policy": policy, "recovery_mode": proposer.recovery_mode,
             "calls": [record], "proposed": [list(p) for p in pairs],
             "normalization_warnings": 0,
             "accounting": _event_accounting([record], additions),
             "kind": request, "admitted": len(pairs), "outcomes": [a["outcome"] for a in additions],
             **({"revive_requested": revive["requested"], "reply_shape": record.get("reply_shape")} if review else {})}
    return pairs, event, additions, revive


REVIVE_INVALID_CODES = ("unknown", "active", "not_retired", "over_limit", "repeated")


def _validate_revive(requested, engine_ids, archive, limit):
    """Unified protocol: (valid engine ids, invalid [{id, engine_id, why, detail}]) of a
    review reply's revive list, in reply order. The model may name at most `limit` ids, so an id past place `limit` is
    "over_limit" whatever it names; within the first `limit` places an id is "unknown" (not a string, or no archived
    model has it), "active" (its model is in the pool), "not_retired" (archived with an outcome other than
    retired_by_evidence: duplicate, rejected, fit_failed) or "repeated" (named earlier in the list); every other id is
    valid. Invalid ids are logged and ignored - no repair turn for this optional part."""
    valid, invalid = [], []
    for place, (written, engine_id) in enumerate(zip(requested, engine_ids)):
        row = archive.get(engine_id) if isinstance(engine_id, str) else None
        if place >= limit:
            why, detail = "over_limit", f"place {place + 1}; at most {limit} ids per review round"
        elif row is None:
            why, detail = "unknown", ("not a model_id string" if not isinstance(engine_id, str)
                                      else "no archived model has this model_id")
        elif row.get("outcome") == "active":
            why, detail = "active", "the model is in the active pool"
        elif row.get("outcome") != "retired_by_evidence":
            why, detail = "not_retired", f"archive outcome {row.get('outcome')}"
        elif engine_id in valid:
            why, detail = "repeated", "named earlier in this list"
        else:
            valid.append(engine_id)
            continue
        invalid.append({"id": written, "engine_id": engine_id if isinstance(engine_id, str) else None,
                        "why": why, "detail": detail})
    return valid, invalid


def _llm_burst(proposer, *, size, designs, ys, constraints, interface, packets_dir, run_id, world, arm,
               fit_pool, concurrency=BURST_CONCURRENCY):
    """The round-0 burst of FB1_INITIAL_PROPOSALS=<size> (the count runners' initial-burst-with-refill pattern).

    Wave 0 asks `size` independent single-candidate calls (request "burst") on one empty-pool packet; refill waves 1 and
    2 ask exactly the shortfall (request "burst_refill") on a packet that shows the candidates admitted so far, fitted by
    `fit_pool(pairs, wave)` -> (fits, log evidence) on the public data (packet-only fits, their own random stream). The
    calls of a wave run in parallel threads (one at a time on a mock transport, so a mock run's bytes are fixed); their
    replies are admitted in call-index order by `_admit_llm_candidate` - the per-round admission, with the same
    free-structure grammar, probe, catalogue-transcription and duplicate checks. A candidate counts toward the size when
    it is admitted (a ForceBench fit does not fail). Returns (admitted pairs, wave events, archive additions).
    """
    from concurrent.futures import ThreadPoolExecutor
    n_designs = len(designs)
    p1a = jnp.array([d[0] for d in designs]); p2a = jnp.array([d[1] for d in designs])
    steps = _integration_steps()
    _engine_text = (proposer.name_mask.swap_names if proposer.name_mask is not None
                    else (lambda value: value))
    pairs, additions, events, have = [], [], [], set()
    shortfall = size
    for wave in BURST_WAVES:
        if shortfall <= 0:
            break
        if wave == 0:
            rows = _observation_rows(designs, ys, None, steps)
            packet = fbp.build_empty_pool_packet(rows, [], constraints, interface, 0)
        else:
            fits, log_evidence = fit_pool(list(pairs), wave)
            log_evidence = np.asarray(log_evidence, float)
            weights = np.exp(log_evidence - log_evidence.max()); weights = weights / weights.sum()
            top = int(np.argmax(weights))
            best_pred = _public_predictions(fits[top][0], fits[top][1], p1a, p2a)
            rows = _observation_rows(designs, ys, best_pred, steps)
            members = _pool_members(fits, weights, log_evidence, {}, 0, n_designs, best_pred)
            packet = _catalogue_view(fbp.build_belief_packet(rows, members, _packet_archive(additions), constraints,
                                                             interface, 0))
        shown = [member["model_id"] for member in packet["pool"]]
        if proposer.name_mask is not None:
            packet = proposer.name_mask.mask_value(packet)
        text = fbp.serialize_packet(packet)
        packet_file = f"t{n_designs:02d}_burst_w{wave}.json"
        with open(os.path.join(packets_dir, packet_file), "x") as fh:
            fh.write(text)
        request = "burst" if wave == 0 else "burst_refill"
        contexts = [{"run_id": run_id, "world": world, "arm": arm, "policy": "initial_burst",
                     "event": 0, "round": 0, "n_designs": n_designs, "candidate_index": index,
                     "packet_file": packet_file, "burst_wave": wave, "burst_call_index": index}
                    for index in range(shortfall)]
        workers = 1 if proposer.transport == "mock" else max(1, min(concurrency, shortfall))

        def _call(context, text=text, request=request):
            return proposer.propose_one(text, context=context, constraints=constraints, request=request)

        if workers == 1:
            results = [_call(context) for context in contexts]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_call, contexts))
        calls, wave_additions, wave_pairs = [], [], []
        for index, (got, record) in enumerate(results):
            calls.append(record)
            if not got:
                wave_additions.extend(_refused_entries(
                    record["rejected_candidates"], round_t=0, engine_text=_engine_text,
                    id_preimage=(lambda k, wave=wave, index=index: f"failed|burst|{wave}|{index}|{k}")))
                continue
            entry, admitted_key = _admit_llm_candidate(got[0], _engine_text(record.get("note", "")), round_t=0,
                                                       have=have, p1a=p1a, p2a=p2a)
            wave_additions.append(entry)
            if admitted_key is not None:
                have.add(admitted_key)
                wave_pairs.append(admitted_key)
        additions.extend(wave_additions)
        pairs.extend(wave_pairs)
        events.append({"n_designs": n_designs, "round": 0, "kind": "initial_burst", "wave": wave,
                       "request": request, "policy": "initial_burst", "recovery_mode": proposer.recovery_mode,
                       "requested": len(contexts), "accepted": len(wave_pairs), "concurrency": workers,
                       "packet_file": packet_file, **fbp.packet_digest(text),
                       "pool_size_shown": len(shown), "pool_model_ids_shown": shown,
                       "calls": calls, "proposed": [list(p) for p in wave_pairs],
                       "normalization_warnings": 0,
                       "accounting": _event_accounting(calls, wave_additions)})
        shortfall -= len(wave_pairs)
    return pairs, events, additions


# ---------------- Run ----------------
def main():
    """Runs one ForceBench cell end to end (world, arm and seed from FB1_* variables) and writes the run JSON."""
    _recheck_label_receipt()
    t_run0 = time.perf_counter()
    code_sha256, versions = _code_records()       # code fingerprints and versions
    input_provenance = _input_provenance()
    # With the variant on, the _rf suffix -- the main (unconstrained) result file cannot be overwritten.
    os.makedirs(OUT_DIR, exist_ok=True)       # portability: the output root may not exist on a new host
    # Free-expression runs get the `_free` suffix.  The world, arm, seed and list binding stay, so the main result
    # (catalogue) file cannot be overwritten.  design_variants.py is a protected file, so the suffix logic is
    # kept here (in the runner).  When both variants are on the suffix becomes `_rf_free`.
    out_path = os.path.join(OUT_DIR, f"FB1_RUN_{WORLD}_{ARM}_seed{RUN_SEED}_{LIST_TAG}"
                                     f"{'_rf' if VARIANT_ON else ''}"
                                     f"{'_free' if FREE_STRUCTURES else ''}.json")
    out_base = out_path[:-len(".json")]
    numerical_record = fwd.numerical_provenance(WORLD, NUMERICS)
    numerical_record["actual_input_provenance"] = input_provenance
    # A separate artifact records defaults too without adding scientific JSON
    # keys to historical default runs. Explicit options also enter that JSON.
    with open(out_base + "_numerics.json", "w") as handle:
        json.dump(numerical_record, handle, indent=2)
        handle.write("\n")
    if NUMERICS_EXPLICIT:
        input_provenance["numerics"] = fwd.numerical_provenance(WORLD, NUMERICS)
    component_npz = {}                            # file name per checkpoint
    # ---- Created only on the llm proposal path.  The default mock path does not pass through here.
    packets_dir = records_dir = None
    proposer, llm_events, llm_archive = None, [], {}
    one_step_history, prev_fits = {}, None
    if PROPOSER == "llm":
        packets_dir, records_dir = out_base + "_packets", out_base + "_proposer"
        os.makedirs(packets_dir, exist_ok=False)
        # Pass the provider **explicitly**.  Off the paid path (ALLOW_PAID_API=False) fb1_proposer does not even
        # look at this value and uses the mock transport -- mock output bytes unchanged.
        # The input-name mask is built once per run and told to fbp.model_id, which then keeps
        # each masked identifier beside the original one. None when the option is off.
        name_mask = fbp.InputNameMask() if MASK_INPUT_NAMES else None
        fbp.set_input_name_mask(name_mask)
        proposer = fbp.Fb1CatalogProposer(records_dir, allow_live=ALLOW_PAID_API,
                                          model=LLM_MODEL, provider=LLM_PROVIDER,
                                          structure_mode=STRUCTURE_PROPOSALS, name_mask=name_mask,
                                          **({"base_url": LLM_BASE_URL, "key_name": LLM_KEY_NAME,
                                              "provider_order": LLM_PROVIDER_ORDER}
                                             if LLM_PROVIDER == "openai_compat" else {}))
        # Free mode uses a different constraint block (structure_expansion=True +
        # grammar, input_names, n_ops_max).  The probe grid uses the measurement times, so TIMES is passed.
        llm_constraints = (
            fbp.free_constraints(INITIAL_POOL[0][1], family_expansion=ARM[1] == "1", times=TIMES)
            if FREE_STRUCTURES else
            fbp.catalog_constraints(INITIAL_POOL[0][1], family_expansion=ARM[1] == "1"))
        llm_interface = fbp.forcebench_interface(
            measurement_times=TIMES, dt=DT, p1_range=P1_RANGE, p2_range=P2_RANGE,
            integration_steps=sum(int(round((float(tq) - (0.0 if i == 0 else float(TIMES[i - 1]))) / DT))
                                  for i, tq in enumerate(TIMES)))
        if COMPOSED:
            # Candidates are fitted in the round they are admitted; the one-step score
            # definition the packets carry says so.
            llm_interface = dict(llm_interface, summary_definitions=fbp.unified_summary_definitions())
        if CATALOGUE == "off":
            # The packets do not show the law catalogue.
            llm_constraints = fbp.catalogue_off_constraints(llm_constraints)
    world = World()
    rng = np.random.default_rng(RUN_SEED)
    key = jax.random.PRNGKey(RUN_SEED)
    if EPIG_DESIGN:
        epig_targets, epig_target_stream = epig_target_points(RUN_SEED)   # uniform unless score_q

    # Does a noise-free call leave the RNG unconsumed? (a pass condition of the design)
    a = world.truth(1.0, 1.0); b = world.truth(1.0, 1.0)
    repro_clean = bool(np.array_equal(a, b))
    o1 = world.observe(1.0, 1.0); _ = world.truth(1.0, 1.0); o2 = world.observe(1.0, 1.0)
    world_fresh = World()
    q1 = world_fresh.observe(1.0, 1.0); q2 = world_fresh.observe(1.0, 1.0)
    rng_untouched = bool(np.array_equal(o1, q1) and np.array_equal(o2, q2))
    world = World()                                    # reset the counters, then the main run

    # held-out Q: noise-free truths + noisy observations from a separate seed
    score_exe = get_world(WORLD, engine=IC["engine"], noise_std=SIGMA, noise_seed=SCORE_SEED)["executor"]
    def _q_data(Q):
        req = [{"p1": float(p1), "p2": float(p2), "pos2": [float(v) for v in np.asarray(POS0)],
                "velocity2": [float(v) for v in np.asarray(VEL0)], "measurement_times": list(TIMES)} for p1, p2 in Q]
        with score_exe.noise_disabled():
            tr = [np.asarray(r["pos2"], float) for r in score_exe.run(req)]
        ob = [np.asarray(r["pos2"], float) for r in score_exe.run(req)]
        return np.array(tr), np.array(ob)
    TR_H, YH_H = _q_data(Q_HEAD)
    TR_S, YH_S = _q_data(Q_SHIFT)
    # ---- Evaluation-only file.  In the whole code these arrays are passed only to score_set
    #      (the fit loglik sees the training observations ya, EIG sees cands, proposals see only MOCK_PROPOSALS).
    eval_npz_path = f"{out_base}_evaluation_only.npz"
    np.savez_compressed(eval_npz_path,
                        Q_HEAD=np.asarray(Q_HEAD, float), Q_SHIFT=np.asarray(Q_SHIFT, float),
                        TR_H=np.asarray(TR_H), TR_S=np.asarray(TR_S),
                        YH_H=np.asarray(YH_H), YH_S=np.asarray(YH_S),
                        sigma_world=np.float64(SIGMA), score_seed=np.int64(SCORE_SEED),
                        var_gt=np.float64(VAR_GT), measurement_times=np.asarray(TIMES),
                        world=np.asarray(WORLD), ic_manifest=np.asarray("FB1_FIXED_IC_MANIFEST_v5.json"))

    active = [Candidate(s, f, 0) for s, f in INITIAL_POOL]
    designs, ys, log, n_prop_calls = [], [], [], 0
    proposals_log, admitted, retired = [], 0, 0

    d0 = _initial_designs(rng)
    for p1, p2 in d0:
        with Clock("world"): y = world.observe(p1, p2)
        designs.append([float(p1), float(p2)]); ys.append(y)

    burst_events = []
    if BURST:
        # ---- Unified protocol: the round-0 burst replaces the catalogue pool. Refill packets show the admitted candidates
        #      fitted on the D0 data with their own key stream; the loop below refits the pool at n = D0 with the run's
        #      key as in every round, so the run's random streams do not depend on whether a refill wave happened.
        _p1b, _p2b = jnp.array([d[0] for d in designs]), jnp.array([d[1] for d in designs])
        _yb = jnp.array(np.array(ys))

        def _burst_fit_pool(pairs, wave):
            kb = jax.random.fold_in(jax.random.PRNGKey(BURST_PACKET_FIT_SEED), wave)
            fits_b, lz_b = [], []
            for s, f in pairs:
                cand = Candidate(s, f, D0)
                kb, kf = jax.random.split(kb)
                u, lz, _diag = smc_fit(kf, cand, loglik_for(cand, _p1b, _p2b, _yb))
                fits_b.append((cand, u)); lz_b.append(lz)
            return fits_b, lz_b

        _t_burst = time.perf_counter()
        with Clock("propose"):
            burst_pairs, burst_events, burst_additions = _llm_burst(
                proposer, size=BURST_SIZE, designs=designs, ys=ys, constraints=llm_constraints,
                interface=llm_interface, packets_dir=packets_dir, run_id=os.path.basename(out_base), world=WORLD,
                arm=ARM, fit_pool=_burst_fit_pool)
        TIMERS["initial_burst"] += time.perf_counter() - _t_burst
        for _a in burst_additions:
            llm_archive.setdefault(_a["model_id"], _a)
        if not burst_pairs:
            raise RuntimeError(f"the initial burst admitted no candidate in {len(burst_events)} waves; nothing to fit")
        active = [Candidate(s, f, D0) for s, f in burst_pairs]
        print(f"[burst] requested={BURST_SIZE} admitted={len(burst_pairs)} shortfall={BURST_SIZE - len(burst_pairs)} "
              f"waves={len(burst_events)} t={time.perf_counter()-t_run0:7.1f}s", file=sys.stderr, flush=True)

    retired_models, revival_events, revival_refits = {}, [], 0
    revival_dir = out_base + "_revival"
    if REVIEW_ROUNDS:
        os.makedirs(revival_dir, exist_ok=False)
    checkpoints = {}
    while True:
        n = len(designs)
        if PROPOSER == "llm" and prev_fits is not None:
            # One-step predictive log score -- with the previous round's particles, **before** y_n is incorporated:
            # log( (1/P) Σ_i p(y_new | θ_i) ) (log-mean-exp).  Uses public observations only and does not
            # consume the RNG.  Rounds before a candidate's birth are null and are not filled retroactively.
            _t_new = n - D0
            _p1n, _p2n = jnp.array([designs[-1][0]]), jnp.array([designs[-1][1]])
            _yn = jnp.array(np.asarray(ys[-1], float))[None, None]
            for _c, _u in prev_fits:
                _mu = predict_for(_c, 1)(_u, _p1n, _p2n)
                _s = core.noise_scale(_u, _c)
                _ll = log_dens(_yn, _mu, _s[None, :], _c.family)
                _v = float(logsumexp(_ll) - math.log(_mu.shape[1]))
                one_step_history.setdefault(_c.label(), {})[_t_new] = _v if math.isfinite(_v) else None
        p1a, p2a = jnp.array([d[0] for d in designs]), jnp.array([d[1] for d in designs])
        ya = jnp.array(np.array(ys))
        fits, logZs = [], []
        with Clock("fit"):
            for cand in active:
                key, kf = jax.random.split(key)
                u, lz, diag = smc_fit(kf, cand, loglik_for(cand, p1a, p2a, ya))
                fits.append((cand, u)); logZs.append((lz, diag))
        lz = np.array([z for z, _ in logZs])
        wts = np.exp(lz - lz.max()); wts = wts / wts.sum()
        rec = {"n_designs": n, "designs": [list(map(float, d)) for d in designs],
               "active": [c.label() for c in active], "logZ": lz.tolist(),
               "weights": wts.tolist(), "top": active[int(np.argmax(wts))].label(),
               "smc_diag": [d for _, d in logZs],
               "family_mass_gauss": float(sum(w for w, c in zip(wts, active) if c.family == "gauss"))}

        if n in CHECKPOINTS or n == D0 + T_EXTRA:
            is_final_checkpoint = (n == D0 + T_EXTRA)     # the all-particle mu npz is written only here
            key, ks1, ks2 = jax.random.split(key, 3)
            head = score_set(fits, wts, Q_HEAD, TR_H, YH_H, ks1,
                             return_full_particles=is_final_checkpoint)
            shift = score_set(fits, wts, Q_SHIFT, TR_S, YH_S, ks2,
                              return_full_particles=is_final_checkpoint)
            me = _probe_me(fits, wts)
            # ---- Predictive-mixture component npz.  Stores only subsamples of the arrays score_set produced.
            #      (The arguments and values of this save call are byte-identical to before the per-coordinate KL was
            #      added -- the new keys are taken out separately.)
            comp_h, comp_s = head.pop("_components"), shift.pop("_components")
            full_h, full_s = head.pop("_full_particles"), shift.pop("_full_particles")
            # The per-coordinate kl is kept, but the node values inside it (_nodes_*) are used only when the final
            # checkpoint exceeds the 20MB threshold -- either way they are not left in the JSON (numpy arrays are not
            # serializable).
            node_x_h = head["kl"].pop("_nodes_x", None); node_lb_h = head["kl"].pop("_nodes_log_bma", None)
            node_x_s = shift["kl"].pop("_nodes_x", None); node_lb_s = shift["kl"].pop("_nodes_log_bma", None)
            head["component_summary"] = comp_h["summary"]       # checkpoint JSON, float64
            shift["component_summary"] = comp_s["summary"]
            cpath = f"{out_base}_predictive_components_t{n}.npz"
            np.savez_compressed(cpath,
                                candidates=np.asarray([c.label() for c, _ in fits]),
                                weights=np.asarray(wts, float), sigma_world=np.float64(SIGMA),
                                Q_HEAD=np.asarray(Q_HEAD, float), Q_SHIFT=np.asarray(Q_SHIFT, float),
                                mu_head=comp_h["mu"], mu_shift=comp_s["mu"],
                                # per-candidate posterior predictive mean (C,D,T,2) **beside** the component samples.
                                pred_mean_head=comp_h["pred_mean"],
                                pred_mean_shift=comp_s["pred_mean"],
                                particle_index=comp_h["particle_index"],
                                noise_scale_median=comp_h["noise_scale_median"],
                                noise_scale_particles=comp_h["noise_scale_particles"],
                                n_particles_total=np.int64(comp_h["n_particles_total"]))
            component_npz[str(n)] = os.path.basename(cpath)
            # ---- At the final checkpoint only, the all-particle (1000), double-precision mu goes to a separate npz
            #      (write_kl_full_particle_mu also replaces it by node values above 20MB, and discloses that).
            full_particle_meta = None
            if is_final_checkpoint:
                full_particle_meta = write_kl_full_particle_mu(
                    f"{out_base}_kl_full_particle_mu_final.npz",
                    [c.label() for c, _ in fits], wts, full_h, full_s,
                    node_x_h, node_lb_h, node_x_s, node_lb_s)
            # ---- Single top-model score: **the same score_set and the same held-out set** as the BMA.
            #      The scoring key is split from a fixed key outside the main run's random streams (key is not consumed).
            top_i = int(np.argmax(wts))
            _k1, _k2 = jax.random.split(jax.random.fold_in(
                jax.random.PRNGKey(SINGLE_MODEL_SCORE_SEED), n))
            _one, _w1 = [fits[top_i]], np.array([1.0])
            s_head = score_set(_one, _w1, Q_HEAD, TR_H, YH_H, _k1, compute_kl=False)
            s_shift = score_set(_one, _w1, Q_SHIFT, TR_S, YH_S, _k2, compute_kl=False)
            _keep = ("nmse_mda", "nll_per_obs", "coverage90", "interval_width90")
            single_top = {"top": fits[top_i][0].label(), "weight": float(wts[top_i]),
                          "headline_Q": {k: s_head[k] for k in _keep},
                          "shift_Q_extrapolation": {k: s_shift[k] for k in _keep},
                          "note": "the same score_set, held-out Q and truth as the BMA entry; a scoring-only key (the main random stream is untouched)."}
            # ---- Per-candidate family (gauss/student_t4), in the same order as active, at every checkpoint.
            #      Supporting data for the problem where the downstream assembler counted student_t4 as normal (the
            #      student_t4_note of contract/emit_contract.py).
            families = [c.family for c in active]
            # ---- Per-coordinate KL -- the weighted mean of head (60 scalars) and shift (120 scalars) as the combined value.
            checkpoint_kl = combine_checkpoint_kl(
                head["kl"], shift["kl"], len(Q_HEAD) * T_OBS * 2, len(Q_SHIFT) * T_OBS * 2,
                full_particle_mu=full_particle_meta)
            checkpoints[n] = {"headline_Q": head, "shift_Q_extrapolation": shift, "me": me,
                              "family_ok": rec["family_mass_gauss"] > 0.5,
                              "family_mass_gauss": rec["family_mass_gauss"],
                              "brier": None, "brier_reason": "ForceBench outcomes are continuous - a binary Brier score is undefined",
                              "top": rec["top"], "weights": wts.tolist(),
                              "active": rec["active"], "elapsed_s": time.perf_counter() - t_run0}
            checkpoints[n]["single_top_model"] = single_top          # appended only after the existing keys
            checkpoints[n]["families"] = families                   # appended only after the existing keys
            checkpoints[n]["kl"] = checkpoint_kl                    # appended only after the existing keys
            rec["checkpoint"] = True

        if PROPOSER == "llm":
            # Keep active candidates in the archive (structured reasons).  active_rounds grows every round.
            for _c in active:
                # Catalogue names and free-structure names are handled by the same two lookups.
                _e = fbp.structure_expression(_c.struct)
                _p = fbp.structure_roles(_c.struct)
                _mid = fbp.model_id(_e, _c.family, _p)
                _born = max(_c.born_step - D0, 0)
                _entry = llm_archive.get(_mid)
                if _entry is None:
                    llm_archive[_mid] = {"model_id": _mid, "expression": _e, "family": _c.family,
                                         "proposed_round": _born, "outcome": "active",
                                         "reason": {"admitted_round": _born,
                                                    "active_rounds": [n - D0]}}
                elif _entry["outcome"] == "active" and (n - D0) not in _entry["reason"]["active_rounds"]:
                    _entry["reason"]["active_rounds"].append(n - D0)

        if COMPOSED:
            # ---- Unified protocol: one call answering three candidates (on a review
            #      round also a revive list); the admitted candidates and the valid revivals are fitted from the prior on
            #      all current data and the pool is cut ONCE to the top MAX_ACTIVE by log evidence before the round's
            #      design. The checkpoint above (the headline) scored the pool before the round's proposals.
            if EXPAND and n in PROPOSAL_EVENTS and n_prop_calls < N_PROPOSAL_EVENTS:
                _round = n - D0
                _review = _round in REVIEW_ROUNDS
                with Clock("propose"):
                    have = {c.key() for c in active}
                    _seen = set(have) | fbp.archive_pool_keys(llm_archive.values())
                    pairs, _event, _adds, _revive = _llm_propose_composed(
                        proposer, n_designs=n, round_t=_round, event_index=n_prop_calls,
                        designs=designs, ys=ys, fits=fits, weights=wts, log_evidence=lz,
                        one_step=one_step_history, archive_rows=list(llm_archive.values()),
                        pool_keys=_seen, constraints=llm_constraints, interface=llm_interface,
                        packets_dir=packets_dir, run_id=os.path.basename(out_base), world=WORLD,
                        arm=ARM, policy=PROPOSAL_POLICY, review=_review)
                    llm_events.append(_event)
                    for _a in _adds:
                        llm_archive.setdefault(_a["model_id"], _a)
                    got, _new = [], []
                    for s, f in pairs:
                        ok, _got = admission_record(s, f, have, False)
                        got.append(_got)
                        if ok:
                            _new.append(Candidate(s, f, n)); have.add((s, f)); admitted += 1
                    n_prop_calls += 1
                _revived, _revival = [], None
                if _review:
                    _t_rv = time.perf_counter()
                    with Clock("revival"):
                        _valid, _invalid = _validate_revive(_revive["requested"], _revive["requested_engine_ids"],
                                                            llm_archive, fbp.REVIVE_LIMIT)
                        for _mid in [m for m in _valid if m not in retired_models]:   # not retired by this run's cuts
                            _valid.remove(_mid)
                            _invalid.append({"id": _revive["requested"][_revive["requested_engine_ids"].index(_mid)],
                                             "engine_id": _mid, "why": "not_retired",
                                             "detail": "retired_by_evidence row without a retirement in this run"})
                        _revival = {"round": _round, "n_obs": n, "requested_ids": _revive["requested"],
                                    "requested_engine_ids": _revive["requested_engine_ids"],
                                    "revive_missing": _revive["missing"], "reply_shape": _revive["reply_shape"],
                                    "valid_ids": _valid, "invalid": _invalid, "refit_results": [],
                                    "refit_failures": []}
                        for _mid in _valid:
                            _c = retired_models[_mid]
                            key, kf = jax.random.split(key)
                            _t1 = time.perf_counter()
                            try:
                                _u, _z, _diag = smc_fit(kf, _c, loglik_for(_c, p1a, p2a, ya))
                                _why = None if math.isfinite(_z) else f"non-finite log evidence {_z!r}"
                            except Exception as exc:  # noqa: BLE001 - recorded; the model stays retired
                                _why = f"{type(exc).__name__}: {exc}"
                            _secs = time.perf_counter() - _t1
                            if _why is not None:
                                _revival["refit_failures"].append({"id": _mid, "model": _c.label(), "scope": "revival",
                                                                   "error": _why, "seconds": _secs})
                                continue
                            _revived.append((_mid, _c, _u, float(_z)))
                            _revival["refit_results"].append({"id": _mid, "model": _c.label(),
                                                              "log_evidence": float(_z), "seconds": _secs,
                                                              "smc_diag": _diag})
                    revival_refits += len(_valid)
                    # wall_seconds = the revive validation and the refits only (the span chembench_core.py records);
                    # the round's new-candidate fits are timed under fit_new_candidates
                    _rv_secs = time.perf_counter() - _t_rv
                _new_fits = []
                with Clock("fit_new_candidates"):
                    for _c in _new:
                        key, kf = jax.random.split(key)
                        _u, _z, _diag = smc_fit(kf, _c, loglik_for(_c, p1a, p2a, ya))
                        _new_fits.append((_c, _u, float(_z), _diag))
                # one cut over the pool, the revived models and the new candidates (all fitted on the same n designs)
                _pool = ([(c, u, float(z), "active", None) for (c, u), z in zip(fits, lz)]
                         + [(c, u, z, "revived", mid) for mid, c, u, z in _revived]
                         + [(c, u, z, "new", None) for c, u, z, _ in _new_fits])
                _keep, _drop = _top_by_evidence([e[2] for e in _pool], MAX_ACTIVE)
                _cut = min(_pool[i][2] for i in _keep)
                _retired_now, _again = [], []
                for i in _drop:
                    _c, _u, _z, _kind, _mid = _pool[i]
                    if _kind == "revived":
                        _row = llm_archive[_mid]
                        _row["revival_attempts"] = int(_row.get("revival_attempts", 0)) + 1
                        _row.setdefault("revival_history", []).append(
                            {"round": _round, "log_evidence": _z, "outcome": "retired_again"})
                        _again.append(_mid)
                    else:
                        retired_models[_archive_retirement(llm_archive, _c, _z, _cut, n)] = _c
                        _retired_now.append(i)
                _entered = []
                for i in _keep:
                    if _pool[i][3] == "revived":
                        _mid = _pool[i][4]
                        retired_models.pop(_mid)
                        _archive_resurrection(llm_archive, _mid, _round)
                        _row = llm_archive[_mid]
                        _row.setdefault("revival_history", []).append(
                            {"round": _round, "log_evidence": _pool[i][2], "outcome": "re_entered"})
                        _entered.append(_mid)
                active = [_pool[i][0] for i in _keep]
                fits = [(_pool[i][0], _pool[i][1]) for i in _keep]
                lz = np.array([_pool[i][2] for i in _keep])
                wts = np.exp(lz - lz.max()); wts = wts / wts.sum()
                retired += len(_retired_now)
                proposals_log.append({"n_designs": n, "call": n_prop_calls, "pairs": got})
                rec["proposed"] = got
                rec["retired"] = [_pool[i][0].label() for i in _retired_now]
                if _revival is not None:
                    _revival.update(re_entered_ids=_entered, retired_again_ids=_again, wall_seconds=_rv_secs)
                    with open(os.path.join(revival_dir, f"t{_round:02d}.json"), "x") as fh:
                        json.dump(_revival, fh, indent=1)
                    revival_events.append({k: v for k, v in _revival.items()})
                    rec["revival"] = _revival
                    print(f"[revival] round={_round} requested={len(_revival['requested_ids'])} "
                          f"valid={len(_valid)} invalid={len(_invalid)} re_entered={len(_entered)} "
                          f"retired_again={len(_again)} failures={len(_revival['refit_failures'])} "
                          f"wall={_revival['wall_seconds']:.1f}s", file=sys.stderr, flush=True)
                rec["proposal_round"] = {
                    "round": _round, "kind": _event["kind"], "calls": 1, "outcomes": _event["outcomes"],
                    "admitted": len(_new), "duplicates": _event["accounting"]["duplicates"],
                    "rejected": _event["accounting"]["rejected_candidates"],
                    "new_candidates": [{"model": c.label(), "log_evidence": z, "smc_diag": d}
                                       for c, _, z, d in _new_fits],
                    "revived_ids": [mid for mid, _, _, _ in _revived],
                    "retired": [{"model": _pool[i][0].label(), "log_evidence": _pool[i][2],
                                 "gap_to_cutoff_nats": _cut - _pool[i][2], "kind": _pool[i][3]}
                                for i in _drop],
                    "cutoff_log_evidence": _cut, "pool_after": [c.label() for c in active],
                    "log_evidence_after": lz.tolist(), "weights_after": wts.tolist(),
                    "design_uses": "pool_after"}
        elif EXPAND and n in PROPOSAL_EVENTS and n_prop_calls < N_PROPOSAL_EVENTS:
            with Clock("propose"):
                have = {c.key() for c in active}
                got = []
                if PROPOSER == "mock":
                    pairs = MOCK_PROPOSALS[n_prop_calls]
                else:
                    # Match the history set by **candidate identity** (expression + per-coefficient role/prior +
                    # family).  Previously it read a catalogue-only 1:1 table of expression strings, so it died
                    # with ValueError on registered free structures (free expressions are not of the form `F = ...`)
                    # and could not tell apart the same expression with a different prior.
                    _seen = set(have) | fbp.archive_pool_keys(llm_archive.values())
                    pairs, _event, _adds = _llm_propose(
                        proposer, n_designs=n, round_t=n - D0, event_index=n_prop_calls,
                        designs=designs, ys=ys, fits=fits, weights=wts, log_evidence=lz,
                        one_step=one_step_history, archive_rows=list(llm_archive.values()),
                        pool_keys=_seen, constraints=llm_constraints, interface=llm_interface,
                        packets_dir=packets_dir, run_id=os.path.basename(out_base), world=WORLD,
                        arm=ARM, policy=PROPOSAL_POLICY, candidates_per_event=CANDIDATES_PER_EVENT)
                    llm_events.append(_event)
                    for _a in _adds:
                        llm_archive.setdefault(_a["model_id"], _a)
                for s, f in pairs:
                    # The mock path applies the same family rule as the LLM judge (see the family-fixed check above).
                    ok, _got = admission_record(s, f, have, PROPOSER == "mock")
                    got.append(_got)
                    if ok:
                        active.append(Candidate(s, f, n)); have.add((s, f)); admitted += 1
                n_prop_calls += 1
                if len(active) > MAX_ACTIVE:
                    # Active cap of 12.  **This call's new candidates are not eligible for retirement** -- they are not
                    # fitted yet and have no evidence (pre-declared rule).  Existing candidates are retired from the
                    # lowest weight up.
                    n_new = len(active) - len(wts)
                    order = np.argsort(wts)                     # ascending = weakest first
                    n_drop = len(active) - MAX_ACTIVE
                    drop_idx = set(order[:n_drop].tolist())
                    dropped = [active[i].label() for i in range(len(wts)) if i in drop_idx]
                    if PROPOSER == "llm":
                        # Record the evidence-rank retirement in the archive with a structured reason (can be re-evaluated).
                        _keep = [i for i in range(len(wts)) if i not in drop_idx]
                        _cut = float(min(lz[i] for i in _keep)) if _keep else float(max(lz))
                        for _i in sorted(drop_idx):
                            _c = active[_i]
                            # Catalogue names and free-structure names are handled by the same two lookups (the same
                            # functions as the active archive above) -- the catalogue-only
                            # `canonical_expression`/`COEFFICIENT_NAMES` raised ValueError/KeyError on registered
                            # free structures, and the evidence-retirement record stopped
                            # right there.
                            _e = fbp.structure_expression(_c.struct)
                            _p = fbp.structure_roles(_c.struct)
                            _mid = fbp.model_id(_e, _c.family, _p)
                            _prev = llm_archive.get(_mid, {})
                            llm_archive[_mid] = {
                                "model_id": _mid, "expression": _e, "family": _c.family,
                                "proposed_round": int(_prev.get("proposed_round",
                                                                max(_c.born_step - D0, 0))),
                                "outcome": "retired_by_evidence",
                                "reason": {"log_evidence_at_retirement": float(lz[_i]),
                                           "gap_to_cutoff_nats": float(_cut - lz[_i]),
                                           "active_rounds": list(
                                               _prev.get("reason", {}).get("active_rounds", [])),
                                           "computed_round": n - D0, "n_obs": n}}
                    active = [active[i] for i in range(len(active)) if i not in drop_idx]
                    retired += len(dropped); rec["retired"] = dropped
                    assert n_new >= 0 and len(active) == MAX_ACTIVE
                proposals_log.append({"n_designs": n, "call": n_prop_calls, "pairs": got})
                rec["proposed"] = got
        print(f"[step] n={n:2d} cands={len(active):2d} top={rec['top']:20s} "
              f"gauss_mass={rec['family_mass_gauss']:.3f} t={time.perf_counter()-t_run0:7.1f}s",
              file=sys.stderr, flush=True)
        log.append(rec)
        prev_fits = fits            # for the next round's one-step score (read only on the llm path)
        if n >= D0 + T_EXTRA: break

        if ASCENT:
            # Design search: no 64-candidate draw; RESTARTS starts from the run rng (uniform, linear box), ascent on the
            # round key ke (what joint_eig/joint_epig would have received), choice by the independent re-estimate.
            key, ke = jax.random.split(key)
            with Clock("eig"):
                _plain_predict = lambda _c, _D: (lambda _u, _a, _b: make_predict(_c)(_u, _a, _b, POS0, VEL0, TIMES))
                # Unified protocol: ASCENT_OBJECTIVE is "epig" / "eig" exactly as before unless FB1_DESIGN_POLICY selects a
                # part of the joint EIG; record_parts is passed only when FB1_DESIGN_POLICY is set.
                _chosen, best, acq = fds.run_design_search(
                    ke, rng, fits, wts, objective=ASCENT_OBJECTIVE,
                    p1_range=P1_RANGE, p2_range=P2_RANGE, m_outer=M_OUTER, n_inner=N_INNER,
                    predict=_plain_predict, log_dens=log_dens, t_obs=T_OBS,
                    target_points=epig_targets if EPIG_DESIGN else None,
                    **({"record_parts": True} if DESIGN_POLICY is not None else {}))
            cands_eig = np.asarray([row["design"] for row in acq["holdout_rows"]], float)
            okey = ASCENT_OBJECTIVE
            rec["design_search"] = "ascent"
            rec["acquisition"] = acq
            rec[f"{okey}_max"] = acq["chosen_holdout_value"]
            rec[f"{okey}_se_max"] = max(r["mc_standard_error"] for r in acq["holdout_rows"])
            rec[f"{okey}_rows_columns"] = ["p1", "p2", okey, f"{okey}_se"]
            rec[f"{okey}_rows"] = [[r["design"][0], r["design"][1], r["objective"], r["mc_standard_error"]] for r in acq["holdout_rows"]]
            rec[f"{okey}_argmax"] = best
            rec[f"{okey}_rows_source"] = "ascent endpoints (RESTARTS rows), objective re-estimated on the independent holdout key; argmax = the chosen restart"
            rec["endpoints_at_edge"] = acq["endpoints_at_edge"]
            if DESIGN_POLICY is not None:
                with Clock("design_record"):
                    rec["design_policy"] = design_policy_record_ascent(fits, wts, acq, best)
        else:
            cands = np.column_stack([rng.uniform(*P1_RANGE, N_RANDOM_DESIGNS),
                                     rng.uniform(*P2_RANGE, N_RANDOM_DESIGNS)])
            cands_eig = cands             # replaced below by a masked subset only when the variant is on
            if ACTIVE_DESIGN:
                if VARIANT_APPLIES:
                    # Resolution floor: filter **before the choice** using only the current model's (BMA) predicted
                    # trajectories (no truth used).
                    with Clock("eig"):
                        kv = jax.random.fold_in(jax.random.PRNGKey(VARIANT_SEED), n)
                        min_r_all = dv.predicted_min_r(fits, wts, cands, kv)
                        keep, n_excl, fb = dv.resolution_floor_filter(min_r_all)
                    cands_eig = cands[keep]
                gate_indices = np.arange(len(cands_eig))
                noise_delta = None
                if DESIGN_GATE_ACTIVE:
                    cands_eig, gate_indices, noise_delta, gate_record = gated_acquisition_candidates(
                        fits, wts, cands_eig, out_base, n)
                    rec["design_gate"] = gate_record
                if EPIG_DESIGN:
                    with Clock("eig"):
                        key, ke = jax.random.split(key)
                        epig, edg = joint_epig(ke, fits, wts, cands_eig, epig_targets, noise_delta=noise_delta)
                    best = int(np.argmax(epig))
                    rec["epig_max"], rec["epig_se_max"] = float(epig[best]), edg["epig_se"]
                    rec["epig_rows_columns"] = ["p1", "p2", "epig", "epig_se"]
                    rec["epig_rows"] = [[float(cands_eig[i, 0]), float(cands_eig[i, 1]), float(epig[i]),
                                         float(edg["epig_se_per_design"][i])] for i in range(len(epig))]
                    rec["epig_argmax"] = best
                else:
                    with Clock("eig"):
                        key, ke = jax.random.split(key)
                        eig, edg = (joint_eig(ke, fits, wts, cands_eig) if noise_delta is None else
                                    joint_eig(ke, fits, wts, cands_eig, noise_delta=noise_delta))
                    best = int(np.argmax(eig))
                    rec["eig_max"], rec["eig_se_max"] = float(eig[best]), edg["eig_se"]
                    # Record the EIG and SE of every candidate design (N_RANDOM_DESIGNS rows).  The choice is already made.
                    rec["eig_rows_columns"] = ["p1", "p2", "eig", "eig_se"]
                    rec["eig_rows"] = [[float(cands_eig[i, 0]), float(cands_eig[i, 1]), float(eig[i]),
                                        float(edg["eig_se_per_design"][i])] for i in range(len(eig))]
                    rec["eig_argmax"] = best
                if DESIGN_GATE_ACTIVE:
                    row_key = "epig_row_candidate_indices" if EPIG_DESIGN else "eig_row_candidate_indices"
                    rec["design_gate"][row_key] = gate_indices.tolist()
                    rec["design_gate"]["chosen_candidate_index"] = int(gate_indices[best])
                if VARIANT_APPLIES:
                    # Variant record.  variant_rows keeps every candidate's predicted min r and whether it passed, so the
                    # filter can be reproduced.
                    rec["variant"] = DESIGN_VARIANT
                    rec["n_candidates_excluded"] = int(n_excl)
                    rec["chosen_predicted_min_r"] = float(min_r_all[int(np.flatnonzero(keep)[gate_indices[best]])])
                    rec["fallback"] = bool(fb)
                    rec["variant_floor"] = dv.FLOOR
                    rec["variant_rows_columns"] = ["p1", "p2", "predicted_min_r", "kept"]
                    rec["variant_rows"] = [[float(cands[i, 0]), float(cands[i, 1]), float(min_r_all[i]),
                                            bool(keep[i])] for i in range(len(cands))]
            else:
                best = 0                      # random arm: use the first draw of the candidate list as is
                rec["eig_max"] = None
                if DESIGN_GATE_ACTIVE:
                    rec["design_gate"] = dict(method="dt_halving", applied=False,
                        reason="random policy has no acquisition candidates", candidate_count=0,
                        excluded_count=0, inflated_count=0, delta_sigma=[])
                    if NUMERICS.get("design_gate_fallback", "off") != "off":
                        rec["design_gate"].update(fallback_used=False,
                            fallback=NUMERICS["design_gate_fallback"], fallback_k=NUMERICS["fallback_k"],
                            kept_indices=[], kept_delta_sigma=[])
                if DESIGN_POLICY == "random":
                    # Unified protocol: the drawn design's EIG parts and disagreement, on the record-only key.
                    with Clock("design_record"):
                        rec["design_policy"] = design_policy_record_random(fits, wts, cands_eig[best], n)
        rec["chosen"] = cands_eig[best].tolist()
        if VARIANT_APPLIES:
            # Evaluator only: records the min r of the true world's noise-free trajectory **after** the choice is made.
            _tr = np.asarray(eval_world().truth(*cands_eig[best]), float)
            rec["chosen_true_min_r"] = float(np.linalg.norm(_tr, axis=-1).min())
            print(f"[variant] n={n:2d} excluded={n_excl:2d}/{len(cands)} kept={len(cands_eig):2d} "
                  f"chosen_predicted_min_r={rec['chosen_predicted_min_r']:.4f} "
                  f"fallback={fb} chosen_true_min_r={rec['chosen_true_min_r']:.4f}",
                  file=sys.stderr, flush=True)
        with Clock("world"): y = world.observe(*cands_eig[best])
        designs.append([float(cands_eig[best][0]), float(cands_eig[best][1])]); ys.append(y)

    out = {
        "schema": "fb1-forcebench-gravity-run-2", "world": WORLD, "arm": ARM,
        "arm_config": {"expand": EXPAND, "active_design": ACTIVE_DESIGN,
                       "background_template": BACKGROUND_ADOPTED,
                       "background_b0": core.B0_BACKGROUND,
                       "background_basis": "domain_size=20.0, periodic=True -> R=10, b0=1/(2 pi R^2) (digits derived from the metadata)"},
        "dp_commit": "33b7fa9", "ic_manifest": "FB1_FIXED_IC_MANIFEST_v5.json",
        "sigma_world": SIGMA, "var_gt": VAR_GT, "dt_forward": DT,
        "seeds": {"run": RUN_SEED, "observation": OBS_SEED, "scoring": SCORE_SEED},
        "budget": {"D0": D0, "extra": T_EXTRA, "total": D0 + T_EXTRA,
                   "checkpoints": list(CHECKPOINTS), "propose_at": list(PROPOSE_AT)},
        "smc": {"particles_per_chain": core.N_PARTICLES, "chains": core.N_CHAINS,
                "ess_target": core.ESS_TARGET, "mh_steps": core.MH_STEPS},
        "eig": {"n_random_designs": N_RANDOM_DESIGNS, "m_outer": M_OUTER, "n_inner": N_INNER, "cmaes": False},
        "proposal_list_version": PROPOSAL_LIST_VERSION, "mock_llm_proposals_predeclared": MOCK_PROPOSALS, "initial_pool": INITIAL_POOL,
        "gate_checks": {"noise_free_repeat_identical": repro_clean,
                        "noise_free_call_does_not_consume_rng": rng_untouched,
                        "executor_noisy_calls": world.n_noisy, "executor_clean_calls": world.n_clean,
                        "n_designs": len(designs),
                        "calls_equal_designs": world.n_noisy == len(designs)},
        "expansion": {"proposal_calls": n_prop_calls, "proposed": 3 * n_prop_calls,
                      "admitted": admitted, "retired": retired,
                      "active_final": [c.label() for c in active], "log": proposals_log},
        "timing_s": dict(TIMERS, forward_total=fwd.FORWARD_SECONDS, wall_total=time.perf_counter() - t_run0),
        "steps": log, "checkpoints": checkpoints,
        "exposure": exposure(World(), designs),
        # ================= Additional records (appended only after the existing keys) =================
        "code_sha256": code_sha256,                                   # code fingerprints
        "versions": versions,                                         # package versions
        "provenance": input_provenance,
        "score_Q": {"Q_HEAD": [[float(a), float(b)] for a, b in Q_HEAD],   # fixed scoring designs
                    "Q_SHIFT": [[float(a), float(b)] for a, b in Q_SHIFT],
                    "n_head": len(Q_HEAD), "n_shift": len(Q_SHIFT),
                    "note": "fixed scoring-only Q; never passed to the fitting, EIG or proposal paths."},
        "observations": {"designs": [list(map(float, d)) for d in designs],   # training observations
                         "y": [np.asarray(y).tolist() for y in ys],
                         "shape": [len(ys), T_OBS, 2],
                         "note": "the noisy observed trajectories returned by world.observe(); order = designs."},
        "evaluation_only_npz": os.path.basename(eval_npz_path),       # evaluation-only file
        "predictive_components_npz": component_npz,                   # component npz per checkpoint
        # ================= Truth-law record (appended only after the existing keys) =================
        "truth_law": {                                                # the world's true force law
            "world": WORLD, "law_str": CFG.get("law_str"),
            "var_gt": VAR_GT, "sigma_world": SIGMA,
            "inertia": CFG["inertia"], "soft": CFG["soft"],
            "me_probe": [float(v) for v in CFG["me_probe"]],
            "dp_commit": "33b7fa9",
            "note": "numeric parameters of WORLD_CFG[world] for the currently selected WORLD "
                    "-- previously only 'world' + 'dp_commit' "
                    "identified the true force law in the run JSON",
        },
    }
    if ASCENT:
        out["arm_config"]["design_search"] = {"mode": "ascent", **fds.design_search_constants(),
                                              "objective": "epig" if EPIG_DESIGN else "eig",
                                              "candidate_source": "ascent_endpoints", "random64_candidates_drawn": False}
        out["eig"]["design_search"] = "ascent"
    if EPIG_DESIGN:
        out["arm_config"]["design_policy"] = "epig"
        out["epig_target_points"] = epig_targets.tolist()
        out["seeds"]["epig_targets"] = epig_target_stream
        out["epig"] = {"target_count": len(epig_targets), "m_outer": M_OUTER, "n_inner": N_INNER,
                       "units": "nats", "target_measure": epig_target_stream["distribution"],   # uniform_rectangle | score_q
                       "noise_inflation_channel": "acquisition observation only; targets retain declared model noise",
                       "se_definition": "standard deviation of per-outer target averages / sqrt(m_outer)",
                       "se_conditioning": "fixed target points and inner particles; excludes nested Monte Carlo bias"}
        if EPIG_SCORE_Q:
            # The option is visible in arm_config like the other opt-in options, and the score_Q note no longer
            # says the scoring designs stay out of acquisition - their locations are the EPIG targets now.
            out["arm_config"]["epig_targets"] = {
                "mode": "score_q", "option": "FB1_EPIG_TARGETS", "count": len(epig_targets),
                "points": epig_targets.tolist(), "source": epig_target_stream["source"], "fixed_across_rounds": True,
                "default": "uniform: 32 points uniform over P1_RANGE x P2_RANGE from sample_epig_targets",
                "held_out_outcomes_used": False}
            out["score_Q"]["note"] = (
                "Fixed scoring Q. FB1_EPIG_TARGETS=score_q: these design locations (p1, p2) are also the EPIG targets "
                "passed to acquisition; the held-out truths and noisy outcomes (evaluation_only_npz) are never passed "
                "to fitting, acquisition or proposals.")
    if D0_POLICY != "random2":
        for config in (out["budget"], out["arm_config"]):
            config.update(d0_policy=D0_POLICY, d0_points=d0.tolist(), d0_size=D0,
                          d0_definition=D0_DEFINITION)
    if NUMERICS_EXPLICIT:
        out["arm_config"]["numerics"] = dict(NUMERICS)
    if FB1_CHECKPOINTS_OPTION != "default":
        # Record of the FB1_CHECKPOINTS option -- with the default ("default", i.e. unset or an explicit "default") this
        # key is not attached, so default run outputs stay byte for byte the same (attached only for "all" or a comma
        # list, the same convention as PROPOSER=="llm"/FREE_STRUCTURES/VARIANT_ACTIVE).
        out["budget"]["checkpoints_option"] = FB1_CHECKPOINTS_OPTION
    if PROPOSER == "llm":
        # ================= LLM-path records (attached only on the llm path -- mock output bytes unchanged) =========
        # Catalogue-selection mode: candidates are **selected** from a list of named templates.  The "recovery" of
        # yukawa, oscillator and extra_dimensions is a selection, not a discovery -> this label
        # is required.
        out["recovery_mode"] = proposer.recovery_mode
        out["recovery_mode_note"] = (
            (
                "candidates are free-form expressions over the named inputs {r, p1, p2, t}, "
                "validated by the shared CORE grammar and registered into fb1_core.STRUCTURES "
                "at runtime (constraints.structure_expansion=true); a law that survives the "
                "grammar, the numerical-equivalence duplicate check and the finiteness "
                "probe is a DISCOVERY, not a selection from the named catalogue"
            ) if FREE_STRUCTURES else (
                "candidates are chosen verbatim from the named STRUCTURES catalog "
                "(constraints.structure_expansion=false); a recovered world law is a SELECTION, "
                "not a discovery"
            )
        )
        out["proposer"] = {
            "proposer": "llm", "policy": PROPOSAL_POLICY,
            "candidates_per_event": CANDIDATES_PER_EVENT,
            "proposal_events": list(PROPOSAL_EVENTS), "n_proposal_events": N_PROPOSAL_EVENTS,
            "transport": proposer.transport, "allow_paid_api": ALLOW_PAID_API,
            "paid_api_calls": proposer.paid_api_calls, "requested_model": LLM_MODEL,
            # Transport provider and label.  The label is attached **only on the llm path**: mock/catalogue outputs
            # must stay byte-identical, and a transport label is undefined for a run that
            # used no transport.
            "requested_provider": LLM_PROVIDER,
            **proposer.compat_metadata,
            "label": RUN_LABEL, "label_requested": RUN_LABEL_REQUEST,
            "label_forced_to_development": RUN_LABEL_FORCED_TO_DEVELOPMENT,
            "label_rule": RUN_LABEL_RULE,
            **({"label_receipt": dict(LABEL_RECEIPT)} if LABEL_EXPLICIT else {}),
            **fbp.prompt_identity(),
            "packets_dir": os.path.basename(packets_dir),
            "records_dir": os.path.basename(records_dir),
            "prior_probe": {"seed": PRIOR_PROBE_SEED, "n_samples": PRIOR_PROBE_SAMPLES},
            "constraints": llm_constraints, "interface": llm_interface,
            "mock_schedule_predeclared": [list(p) for p in fbp.MOCK_SCHEDULE],
            "events": llm_events,
            "archive": sorted(llm_archive.values(),
                              key=lambda a: (a["proposed_round"], a["model_id"])),
            "accounting": {
                k: sum(e["accounting"][k] for e in llm_events)
                for k in ("llm_calls", "llm_repairs", "paid_api_calls", "valid_candidates",
                          "duplicates", "rejected_candidates", "prior_probe_rejections",
                          "normalization_warnings")},
            "normalization_warning_reason":
                "not defined for a continuous likelihood: gauss/student_t4 densities are "
                "analytically normalized, so CORE's PMF sum tolerance check has no analogue; "
                "the ForceBench analogue is the prior finiteness probe recorded per candidate",
        }
        if PROMPT_VARIANT is not None:
            # Which file the tag and sha256 above belong to.
            out["proposer"]["prompt_variant_file"] = os.path.basename(PROMPT_VARIANT)
        if proposer.name_mask is not None:
            # With input-name masking, what the model was shown is recorded as it was shown - constraints,
            # interface, the archive and each event's proposed pairs in masked names (the
            # events' call records already are). The pairs of masked and original identifiers
            # go to one side file. Everything outside this block stays in the engine's names.
            _mask = proposer.name_mask
            for _block in ("constraints", "interface", "archive"):
                out["proposer"][_block] = _mask.mask_value(out["proposer"][_block])
            out["proposer"]["events"] = [
                {**_event, "proposed": [[_mask.mask_structure_name(_s), _f]
                                        for _s, _f in _event["proposed"]]}
                for _event in out["proposer"]["events"]]
            _pairs_file = out_base + "_masked_model_ids.jsonl"
            with open(_pairs_file, "x") as handle:
                handle.writelines(json.dumps(_row, sort_keys=True) + "\n" for _row in _mask.pairs())
            out["proposer"].update(
                mask_input_names=True, input_name_map=dict(_mask.input_name_map),
                masked_model_ids_file=os.path.basename(_pairs_file),
                mask_text_rewrites=[list(pair) for pair in fbp.MASK_TEXT_REWRITES])
    if FREE_STRUCTURES:
        # Attached only to free-expression runs -- catalogue outputs must stay byte-identical.
        # The registration list (structure name, canonical identifier, expression, roles, operator
        # count) stays here, so which free expressions actually entered the pool can be reproduced from the outputs.
        out["arm_config"]["structure_proposals"] = {
            "mode": STRUCTURE_PROPOSALS,
            **fbg.grammar_identity(TIMES),
            "mock_schedule_predeclared": [
                {"expression": e, "family": f, "parameters": dict(p)}
                for e, f, p in fbp.FREE_MOCK_SCHEDULE
            ] if PROPOSER == "llm" else None,
            "mock_schedule_expected_outcomes": (
                list(fbp.FREE_MOCK_EXPECTED_OUTCOMES) if PROPOSER == "llm" else None),
            "n_registered": len(fbg.REGISTRY),
            "registered": fbg.registry_records(),
        }
        if GRAMMAR_TRIG:
            # What the option changed (the grammar identity above already lists cos and sin in "calls").
            out["arm_config"]["structure_proposals"]["grammar_trig"] = fbg_trig.identity()
    if VARIANT_ACTIVE:
        # Kept only for runs where the variant can actually operate.
        # The random arm is excluded because its output must be byte-identical with or without the variant.
        out["arm_config"]["design_variant"] = {
            "name": DESIGN_VARIANT, "floor": dv.FLOOR, "dx": dv.DX,
            "variant_seed": VARIANT_SEED,
            "domain_size": dv.DOMAIN_SIZE, "grid": dv.GRID,
            "rule": "before the EIG computation, exclude candidates with min_t r(t) < 2*dx (judged on the current BMA predicted trajectory)",
            "dx_basis": "default arguments of scienceagent/executor.py SimulationExecutor "
                        "grid_size=(64, 64), domain_size=20.0 -> dx = 20/64 = 0.3125",
            "scope": "field-engine worlds, active arms only", "engine": IC["engine"],
            "applied": bool(VARIANT_APPLIES),
            "n_particles_subsample": dv.N_PARTICLES_SUBSAMPLE,
            "truth_used_for_selection": False,
            "note": "steps[*].chosen_true_min_r is recorded by the evaluator after the choice, for the record only."}
    if UNIFIED:
        # ================= Unified-protocol records (only for runs with an option on -- default outputs unchanged) =================
        out["arm_config"]["unified_protocol"] = {
            "protocol": "unified protocol (registered 2026-10-06)",
            "initial_proposals": BURST_SIZE if BURST else None,
            "initial_pool": f"llm_burst:{BURST_SIZE}" if BURST else "catalogue",
            "active_maximum": MAX_ACTIVE,
            "proposal_limit": COMPOSED_LIMIT if COMPOSED else None,
            # rounds t = n - D0 (as archive_review_rounds and proposal_rounds[].round); budget.proposal_events holds n
            "proposal_rounds": [n_ev - D0 for n_ev in PROPOSAL_EVENTS] if COMPOSED else None,
            "archive_review_rounds": list(REVIEW_ROUNDS),
            "design_policy": DESIGN_POLICY, "packet_archive": PACKET_ARCHIVE, "catalogue": CATALOGUE,
            "catalogue_off_removes": (None if CATALOGUE == "on" else {
                "constraints": ["initial_structures (emptied)", "structure_parameters (emptied)"],
                "grammar": list(fbp.CATALOGUE_GRAMMAR_KEYS),
                "grammar.duplicate_detection": list(fbp.CATALOGUE_DUPLICATE_KEYS) + [
                    f"reason_codes.{fbp.CATALOGUE_REASON_CODE_KEY}",
                    f"stage_2_numerical: the words {fbp.CATALOGUE_NUMERICAL_PHRASE.strip()!r}"],
                "packet archive rows": list(fbp.CATALOGUE_TRACE_ARCHIVE_KEYS),
                "packet text": ["catalogue '^' powers written '**' (pool, best_model, archive expression and "
                                "matched_expression)",
                                "reason_code catalog_reference_same_prior_model -> None (admitted) or the link "
                                "basis's duplicate code (duplicate)",
                                "a duplicate's matched_model_id naming a catalogue law -> that model's model_id"],
                "kept": "the run's own archive, records and the engine's catalogue checks"}),
            "options": {name: os.environ.get(name) for _flag, name in CLI_FLAGS}}
    if BURST:
        _mask = proposer.name_mask
        out["initial_pool"] = [list(p) for p in burst_pairs]   # the pool the burst admitted (no catalogue form)
        out["initial_burst"] = {
            "requested": BURST_SIZE, "admitted": len(burst_pairs), "shortfall": BURST_SIZE - len(burst_pairs),
            "waves": len(burst_events),
            "accepted_per_wave": [e["accepted"] for e in burst_events],
            "requested_per_wave": [e["requested"] for e in burst_events],
            "refill_policy": BURST_REFILL_POLICY, "max_waves": len(BURST_WAVES),
            "concurrency": BURST_CONCURRENCY, "concurrency_used": [e["concurrency"] for e in burst_events],
            "catalogue_pool_used": False, "packet_fit_seed": BURST_PACKET_FIT_SEED,
            "admission": "per-round admission (_admit_llm_candidate): grammar, probe, catalogue transcription, duplicates",
            "packet_wave_0": "empty pool: no best model, no residual block; observation rows carry designs only",
            **fbp.request_prompt_identities()}
        out["proposer"]["initial_burst_events"] = burst_events if _mask is None else [
            {**_event, "proposed": [[_mask.mask_structure_name(_s), _f] for _s, _f in _event["proposed"]],
             "pool_model_ids_shown": _mask.mask_value(_event["pool_model_ids_shown"])}
            for _event in burst_events]
        out["proposer"]["accounting_initial_burst"] = {
            k: sum(e["accounting"][k] for e in burst_events) for k in out["proposer"]["accounting"]}
    if COMPOSED:
        out["budget"]["proposal_events"] = list(PROPOSAL_EVENTS)
        out["proposer"]["composed_prompt"] = {
            "composed": fbp.request_prompt("composed", COMPOSED_LIMIT)[1],
            "composed_review": fbp.request_prompt("composed_review", COMPOSED_LIMIT)[1],
            "composition_paragraph": "composition_paragraph_v2.txt", "archive_review_paragraph":
                "archive_review_paragraph_v1.txt", "calls_per_round": 1, "candidates_per_call": COMPOSED_LIMIT}
        out["proposal_rounds"] = [step["proposal_round"] for step in log if "proposal_round" in step]
        out["final_pool"] = {"active": [c.label() for c in active], "log_evidence": lz.tolist(),
                             "weights": wts.tolist(), "me": _probe_me(fits, wts),
                             "note": "the pool after the last round's proposals (and revivals, on a review round); the "
                                     "last headline checkpoint scored the pool before them"}
    if REVIEW_ROUNDS:
        out["revival_events"] = revival_events
        out["archive_review"] = {
            "rounds": list(REVIEW_ROUNDS), "events": len(revival_events), "revival_refits": revival_refits,
            "re_entered_total": sum(len(e["re_entered_ids"]) for e in revival_events),
            "retired_again_total": sum(len(e["retired_again_ids"]) for e in revival_events),
            "refit_seconds": [r["seconds"] for e in revival_events for r in e["refit_results"]],
            "wall_seconds_total": sum(e["wall_seconds"] for e in revival_events),
            "revival_dir": os.path.basename(revival_dir), "revive_limit": fbp.REVIVE_LIMIT,
            "rule": ("on a listed round the proposal call also carries ARCHIVE_REVIEW v1; each valid revive id (an archived "
                     "retired_by_evidence model, at most two per round) is refit from the prior on all current data and "
                     "competes with the pool and the new candidates in one cut to the top MAX_ACTIVE; invalid ids are "
                     "logged and ignored; a refit failure is recorded with scope revival and the model stays retired")}
    if DESIGN_POLICY is not None:
        out["arm_config"].setdefault("design_policy", DESIGN_POLICY)
        out["arm_config"]["design_policy_target"] = DESIGN_POLICY_TARGET[DESIGN_POLICY]
        out["arm_config"]["design_policy_objective"] = ASCENT_OBJECTIVE if ASCENT else None
        out["arm_config"]["design_policy_evaluated_designs"] = "ascent_endpoints" if ASCENT else "random_draw"
        out["eig_components"] = {"columns": list(fds.EIG_PART_COLUMNS), "m_outer": M_OUTER, "n_inner": N_INNER,
                                 "record_seed": DESIGN_RECORD_SEED if not ASCENT else None,
                                 **EIG_COMPONENTS_DEFINITION}
        out["point_disagreement"] = {"definition": POINT_DISAGREEMENT_DEFINITION, "particles": "all",
                                     "weights": "evidence weights of the round's fits (steps[*].weights)",
                                     "record_only": True}
    _recheck_label_receipt()
    if LABEL_EXPLICIT:
        out["label"] = dict(LABEL_RECEIPT)
    p = out_path
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("gate_checks", "expansion", "timing_s")}, indent=1, ensure_ascii=False))
    for n, c in checkpoints.items():
        print(f"n={n:3d}  nMSE {c['headline_Q']['nmse_mda']:.5f}  NLL {c['headline_Q']['nll_per_obs']:.4f}"
              f"  cov90 {c['headline_Q']['coverage90']:.3f}  top {c['top']:22s}"
              f"  ME dev {c['me']['max_rel_dev']:.4f} (.02 {c['me']['me_eps002']}, .05 {c['me']['me_eps005']})"
              f"  family_ok {c['family_ok']}")
    print("wrote", p, hashlib.sha256(open(p,'rb').read()).hexdigest()[:16])

if __name__ == "__main__":
    main()
