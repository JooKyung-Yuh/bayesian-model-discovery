"""FB-1 (ForceBench) effective force constants per world: WORLD_CFG's me_eff/me_decl/me_probe are read from data files
instead of rounded literals.

run_gravity.py imports only this module (no heavy computation; it reads only a couple of small JSON files).

effective_force_constants(world) -> dict
    gravity  {"A": bg_A, "B": bg_B, ...}  -- the raw values of FB1_FORCE_CURVE_WORLD_RULING.json, unchanged.
    yukawa   {"A": fit A, ...}            -- the raw values of FB1_FORCE_CURVE_yukawa_amplitude.json, unchanged
                                              (that file is written by fit_yukawa_amplitude()).

The three extra_dimensions items (the problem they fix: in WORLD_CFG me_eff/me_decl are None, so _probe_me wrongly
fell back to 1/(2 pi r), and me_probe had been rounded by hand):
    kk_declared_force(r)     the declared KK image sum (the same formula as
                              run_gravity.WORLD_CFG["extra_dimensions"]["true_force"], p1=1), computed exactly for each
                              element of the r array.
    kk_effective_force(r)    the same formula -- no separate amplitude correction, for the reason given below.
    extra_dimensions_me_probe()  reads the IC v5 trajectory range (r_t_min, r_t_max) from the manifest.
"""
import json, math, os
import numpy as np

# Paths are computed from this file, so moving the snapshot preserves its inputs.
# FB1_OUT_DIR only moves outputs; it must never redirect the frozen input reads.
# DiscoverPhysics is external, but these three small input files ship with us.
_SNAPSHOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("FB1_OUT_DIR") or os.path.join(_SNAPSHOT_DIR, "outputs") + os.sep
DATA_DIR = os.environ.get("FB1_DATA_DIR") or os.path.join(_SNAPSHOT_DIR, "frozen_inputs")
FB1_DIR = os.environ.get("FB1_DIR") or _SNAPSHOT_DIR
# os.path.join accepts a data-directory override with or without a trailing slash.
RULING_PATH = os.environ.get("FB1_FORCE_CURVE_RULING",
                             os.path.join(DATA_DIR, "FB1_FORCE_CURVE_WORLD_RULING.json"))
YUKAWA_AMPLITUDE_PATH = os.environ.get("FB1_YUKAWA_AMPLITUDE",
                                       os.path.join(DATA_DIR, "FB1_FORCE_CURVE_yukawa_amplitude.json"))
IC_MANIFEST_PATH = os.environ.get("FB1_IC_MANIFEST",
                                  os.path.join(DATA_DIR, "FB1_FIXED_IC_MANIFEST_v5.json"))


def effective_force_constants(world):
    """Dict of the constants for the world's me_eff, read raw (unrounded) from the data files.

    gravity -> bg_A/bg_B of the RULING JSON (the former run_gravity.py literals 1.0047/0.001356 are roundings of these).
    yukawa  -> A of FB1_FORCE_CURVE_yukawa_amplitude.json (replaces the former literal 1.0174, the same kind of rounded
               value -- but this file is a single-amplitude refit on r>=1 only, so its value is not exactly the
               RULING's bg_A (a joint two-parameter A·B fit on r>=0.5); see fit_yukawa_amplitude below)."""
    if world == "gravity":
        rul = json.load(open(RULING_PATH))["gravity"]
        return {"A": rul["bg_A"], "B": rul["bg_B"], "source": RULING_PATH,
                "formula": "A/(2 pi r) - B r"}
    if world == "yukawa":
        amp = json.load(open(YUKAWA_AMPLITUDE_PATH))
        return {"A": amp["A"], "source": YUKAWA_AMPLITUDE_PATH,
                "formula": "A * kv(1, r/2) / (4 pi)"}
    raise ValueError(f"effective_force_constants: no data-file constant registered for world={world!r}")


# ---------------- extra_dimensions: declared KK image sum (same formula as true_force, p1=1) ----------------
_KK_L = 2 * math.pi * 0.5        # image spacing R=0.5 -> 2*pi*R  (same as run_gravity true_force)
_KK_N = np.arange(-20, 21)       # truncation -20..20  (same as run_gravity true_force)


def kk_declared_force(r):
    """KK image sum, p1=1.  Computed for each element of the r array; returns the same shape.

    run_gravity.WORLD_CFG["extra_dimensions"]["true_force"] is written for a scalar r only
    (np.sum(...,axis=-1)[0] takes only the first element of an array -- harmless there, because _integrate_true always
    calls it with a scalar r, but me_decl is called with a 50-point array in _probe_me and summarize_world.py).  So
    this is a vectorized rewrite (the same formula) that sums independently for every element of the input array."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    s = np.sum(r[..., None] / (r[..., None] ** 2 + (_KK_N * _KK_L) ** 2) ** 1.5, axis=-1)
    return _KK_L * s / (4 * math.pi)


def kk_effective_force(r):
    """me_eff for extra_dimensions.  Simulator convention = nbody free space, softening 0.05.

    gravity/yukawa needed an amplitude correction (bg_A / fit A) of the measured force curve (max relative errors
    0.7985%/1.1314%, bg_maxrel in the RULING JSON), but for extra_dimensions the max relative error of the pure
    (uncorrected, A=1) force curve is already 0.15% (FB1_FORCE_CURVE_WORLD_RULING.json["extra_dimensions"]["pure_maxrel"]
    = 0.0015414541591776151), much smaller -- so no separate amplitude A is fitted and kk_declared_force is used as is
    (amplitude=1)."""
    return kk_declared_force(r)


def extra_dimensions_me_probe():
    """me_probe = the IC v5 trajectory range (r_t_min, r_t_max) of extra_dimensions.

    The former WORLD_CFG literal (0.6, 5.7) was rounded by hand -- here r_t_min/r_t_max of
    FB1_FIXED_IC_MANIFEST_v5.json["worlds"]["extra_dimensions"] are read unchanged."""
    ic = json.load(open(IC_MANIFEST_PATH))["worlds"]["extra_dimensions"]
    return (ic["r_t_min"], ic["r_t_max"])


# ---------------- yukawa amplitude fit (one-off generator -- never called on the run_gravity.py import path) ----
def fit_yukawa_amplitude():
    """Fit A*K1(r/2)/(4*pi) by least_squares to the measured force_of_r("yukawa", rs, grid=64) (r>=1) and write
    FB1_FORCE_CURVE_yukawa_amplitude.json.

    Importing force_curve_diagnostic's FieldSampler is heavy and never needed on run_gravity.py's normal import path
    (SMC/EIG production runs), so it is imported only inside this function (not at module top level) --
    `import world_constants` (= what run_gravity.py does) alone never runs this function."""
    import sys, time
    sys.path.insert(0, FB1_DIR)          # portability: environment variable FB1_DIR (default = this file's directory)
    from force_curve_diagnostic import force_of_r
    from scipy.optimize import least_squares
    from scipy.special import kv

    rs = np.round(np.arange(1.0, 6.0 + 1e-9, 0.25), 2)          # r in {1.00, 1.25, ..., 6.00}, r>=1
    t0 = time.time()
    F = force_of_r("yukawa", rs, grid=64)
    elapsed = time.time() - t0
    model = lambda r, A: A * kv(1, r / 2.0) / (4 * math.pi)
    resid = lambda p: model(rs, p[0]) - F
    sol = least_squares(resid, x0=[1.0])
    A = float(sol.x[0])
    max_rel_err = float(np.max(np.abs(sol.fun / F)))            # |fit-measured|/measured, the same convention as force_curve_diagnostic.fit()
    out = {
        "world": "yukawa",
        "model": "A * kv(1, r/2) / (4*pi)",
        "A": A,
        "r_grid": rs.tolist(),
        "grid": 64,
        "measured_F": F.tolist(),
        "max_rel_err": max_rel_err,
        "fit_method": "scipy.optimize.least_squares(x0=[1.0])",
        "elapsed_s_force_of_r": elapsed,
        # portability: the reproduction command is not hard-coded; it records the interpreter and file **actually run**.
        "command": f"{sys.executable} {os.path.abspath(__file__)}",
        # The note below is written to the output record and is kept byte-identical. In English: r>=1 (drops the
        # grid-near-particle points 0.5/0.75; differs from the yukawa grid of FB1_FORCE_CURVE_WORLD_RULING.json); a
        # single-parameter fit of the amplitude A, no linear background term (B) -- the screened Green function of yukawa
        # has no k=0 zero-mode singularity, so no background correction is needed (the same reasoning as BG['yukawa'] in
        # appendix_b_5worlds.py, consistent with yukawa.adopt=false in the RULING JSON).
        "note": ("r >= 1 (the grid-near-particle points at 0.5/0.75 are dropped; differs from the yukawa grid of "
                 "FB1_FORCE_CURVE_WORLD_RULING.json); single-parameter fit of the amplitude A, no linear background term (B) -- "
                 "the screened Green function of yukawa has no k = 0 zero-mode singularity, so no background correction is "
                 "needed (the same reasoning as BG['yukawa'], consistent with yukawa.adopt = false in the RULING JSON)."),
    }
    json.dump(out, open(YUKAWA_AMPLITUDE_PATH, "w"), indent=1, ensure_ascii=False)
    return out


if __name__ == "__main__":
    _r = fit_yukawa_amplitude()
    print(f"wrote {YUKAWA_AMPLITUDE_PATH}")
    print(f"A = {_r['A']!r}")
    print(f"max_rel_err = {_r['max_rel_err']:.6e}")
    print(f"r_grid = {_r['r_grid']}")
    print(f"elapsed_s_force_of_r = {_r['elapsed_s_force_of_r']:.2f}")
