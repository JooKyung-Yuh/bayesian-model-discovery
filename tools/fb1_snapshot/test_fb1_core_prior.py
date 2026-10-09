"""Tests (new file) for the correction of the sig_hn noise-scale prior sampler.

Background: `fb1_core.log_prior_u` (:74-85) computes the `sig_hn` column as x=exp(u)~HalfNormal(1) (plus the
Jacobian), but before the correction `sample_prior_u` (:96-97) applied `jax.random.normal` to **every** kind of
column unchanged.  In both the initial sequential Monte Carlo (SMC) particles (smc_chain :146) and the prior probe
(run_gravity.py `_prior_probe` :467), the sig_hn column of the real sampler was therefore LogNormal(0,1) (exp(N(0,1))), which
disagrees with the declared HalfNormal(1) (P(sigma<=1): 0.50 vs 0.683).  This file verifies the correction.
The line numbers quoted in this file (such as `:74-85`) are those of the development tree before the correction;
they do not match this snapshot.

This file is kept separate from `test_fb1_proposer.py`, which is not touched.

Run: <jax venv python> -m pytest -q test_fb1_core_prior.py
"""
import os
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
sys.path.insert(0, _HERE)
# Run on a scratch directory dedicated to this test so that importing run_gravity can never write to a shared output
# path (run_gravity.py itself does not write anything merely on import -- checked beforehand --
#  but this is pinned defensively. main() is never called anyway: it sits behind the `if __name__ == "__main__":` guard).
import tempfile  # noqa: E402

os.environ.setdefault("FB1_OUT_DIR", tempfile.mkdtemp(prefix="fb1_core_prior_") + os.sep)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import halfnorm, norm  # noqa: E402

jax.config.update("jax_enable_x64", True)

import fb1_core as core  # noqa: E402
import fb1_proposer as fbp  # noqa: E402
import run_gravity as rg  # noqa: E402

# Same value as PRIOR_PROBE_SEED (run_gravity.py:167) -- a date and the codebase's own constant.
# The cross-check script (check_corrected_prior.py) of the separate EIG investigation scratch used this value too,
# for the same reason rather than by chance -- using the same key lets the numbers of the two independent
# implementations be compared side by side (see CHANGES).
KEY = jax.random.PRNGKey(20260908)
N_LARGE = 200_000


def _pre_fix_sampler(key, n, kinds):
    """Reproduce the pre-correction `sample_prior_u` verbatim (for documentation; the old code is not imported).

    Snapshot i `fb1_core.py:96-97`::

        def sample_prior_u(key, n, kinds):
            return jax.random.normal(key, (n, len(kinds)))
    """
    return jax.random.normal(key, (n, len(kinds)))


# Every (structure, family) kinds combination that occurs in the real pipeline -- 14 structures x 2 families, and the
# last column is always sig_hn (fb1_core.Candidate.kinds() = STRUCTURES[struct].kinds + FAMILIES[family]).
_REAL_KINDS = sorted({
    core.STRUCTURES[s].kinds + core.FAMILIES[f]
    for s in core.STRUCTURES for f in core.FAMILIES
})
assert len(core.STRUCTURES) == 14, "the premise that the catalogue has 14 structures no longer holds"


# --------------------------------------------------------------------- ①
def test_e20b3_r2_1_sigma_le_1_probability_matches_halfnormal_within_6_se():
    """(1) With n=2e5 samples, P(sigma<=1) lies within +-6 SE of halfnorm.cdf(1)."""
    kinds = ("pos", "sig", "sig_hn")
    u = np.asarray(core.sample_prior_u(KEY, N_LARGE, kinds))
    sigma = np.exp(u[:, -1])   # same transform as noise_scale() (x = exp(u))
    p = float(np.mean(sigma <= 1.0))
    expected = float(halfnorm.cdf(1.0))
    se = float(np.sqrt(expected * (1.0 - expected) / N_LARGE))
    assert abs(p - expected) < 6 * se, (
        f"P(sigma<=1)={p} vs halfnorm.cdf(1)={expected} (6*SE={6*se})")
    # Informational record (before the correction this value would be about 0.5 and the assert above would
    # already catch it)
    print(f"[check 1] p={p} expected={expected} 6SE={6*se}")


# --------------------------------------------------------------------- ②
def test_e20b3_r2_2_log_prior_u_matches_reference_density_to_1e10():
    """(2) log_prior_u on the samples equals the reference density to <=1e-10.

    The reference is the sum of norm.logpdf over the other columns plus halfnorm.logpdf(sigma) plus the u Jacobian
    [d sigma/du = sigma].
    """
    kinds = ("pos", "sig", "sig_hn")
    u = np.asarray(core.sample_prior_u(KEY, N_LARGE, kinds))
    sigma = np.exp(u[:, -1])
    log_ref = norm.logpdf(u[:, 0]) + norm.logpdf(u[:, 1]) + halfnorm.logpdf(sigma) + u[:, -1]
    log_got = np.asarray(core.log_prior_u(jnp.asarray(u), kinds))
    err = float(np.max(np.abs(log_ref - log_got)))
    assert err < 1e-10, f"max|log_prior_u - reference density| = {err}"
    print(f"[check 2] max|diff|={err}")


# --------------------------------------------------------------------- ③
def test_e20b3_r1_other_columns_array_equal_and_sig_hn_is_kindwise():
    """(3) Columns other than sig_hn are array_equal to the old sampler (jax.random.normal, same key).

    This is checked for every real (structure, family) combination (28) plus artificial combinations with sig_hn in
    the first or middle column (proving the branch is on **kind**, not on position).  The sig_hn column itself must
    match log(max(|z|,tiny)).
    """
    keys = [jax.random.PRNGKey(0), KEY, jax.random.fold_in(KEY, 7)]
    kind_tuples = list(_REAL_KINDS) + [
        ("sig_hn", "pos", "sig"),         # sig_hn as the first column
        ("pos", "sig_hn", "sig_hn"),      # sig_hn in the middle, twice (adversarial configuration)
        ("pos_bg", "pos_w", "sig_hn"),    # together with the kinds that carry a scale
    ]
    checked = 0
    for key in keys:
        for kinds in kind_tuples:
            n = 37
            got = np.asarray(core.sample_prior_u(key, n, kinds))
            ref_jax = jax.random.normal(key, (n, len(kinds)))   # keep it as a jax array (see below)
            ref = np.asarray(ref_jax)
            assert got.shape == ref.shape == (n, len(kinds))
            other_cols = [j for j, k in enumerate(kinds) if k != "sig_hn"]
            sig_cols = [j for j, k in enumerate(kinds) if k == "sig_hn"]
            if other_cols:
                assert np.array_equal(got[:, other_cols], ref[:, other_cols]), \
                    f"a non-sig_hn column differs from the pre-correction jax.random.normal draw: kinds={kinds}"
            for j in sig_cols:
                # Note: fb1_core.sample_prior_u transforms with jnp.log/jnp.abs/jnp.maximum.
                # The reference is computed with **the same jnp operations** -- computing it in numpy would
                # use a different log implementation (libm vs XLA), and for rare inputs the last bit differs
                # by 1 ulp (measured: 1 of 37 values for PRNGKey(0), kinds=('pos','pos','pos','sig_hn'),
                # difference 2.78e-17 -- array_equal failed although the value is equal within the tol
                # below).  Matching with jnp gives exactly equal results (measured, and also confirmed below).
                z = ref_jax[:, j]
                expect = np.asarray(
                    jnp.log(jnp.maximum(jnp.abs(z), jnp.finfo(z.dtype).tiny)))
                if not np.array_equal(got[:, j], expect):
                    diff = float(np.max(np.abs(got[:, j] - expect)))
                    assert diff < 1e-12, (
                        f"the sig_hn column differs from log(max(|z|, tiny)) by more than 1e-12: "
                        f"kinds={kinds} col={j} max|diff|={diff}")
            checked += 1
    assert checked == len(keys) * len(kind_tuples)
    print(f"[check 3] checked {checked} (key, kinds) combinations across {len(keys)} keys; "
          f"kind_tuples includes all {len(_REAL_KINDS)} distinct kind-shapes that occur among "
          f"the 14 catalogue structures (several structures share a shape, e.g. six "
          f"single-positive-coefficient structures all give ('pos','sig_hn')) plus 3 "
          f"adversarial/edge tuples")


# --------------------------------------------------------------------- ④
def test_e20b3_r2_4_prior_probe_verdicts_unchanged_for_all_14_catalogue_structures():
    """(4) The prior probe of run_gravity._prior_probe looks at the **force only**.

    `mu` uses only `u_to_x(u,kinds)[:, :k_struct]` (fb1_core.py make_predict :120-125, u_to_x :87-94), and
    `k_struct = len(STRUCTURES[struct].kinds)` always lies **before** sig_hn in the order of `cand.kinds() =
    STRUCTURES[struct].kinds + FAMILIES[family]` (both families of FAMILIES are just `("sig_hn",)`,
    fb1_core.py:55), so mu does not change whatever the sig_hn sampler is.  The remaining components of `finite`,
    `isfinite(s).all()` and `(s>0).all()` (run_gravity.py:469), are also always true for both LogNormal(0,1)
    (before the correction, s=exp(z)) and HalfNormal(1) (after the correction, s=max(|z|,tiny)): the former is
    exp(any real) > 0 and finite, the latter is max(|z|,tiny) > 0 and finite -- by the code.  What follows is an
    **execution check**: the real `run_gravity._prior_probe` is called with the pre-correction sampler
    monkeypatched in and again with the corrected (current) code, and `finite`/`nonfinite_fraction` are compared
    for all 14 catalogue structures.
    """
    # The three actual fixed design values of Q_HEAD (a run_gravity.py constant) -- not arbitrary values.
    p1a = jnp.array([1.0, 2.0, 1.0])
    p2a = jnp.array([1.0, 1.0, 2.0])
    assert list(rg.STRUCTURES.keys()) == list(core.STRUCTURES.keys())
    assert len(rg.STRUCTURES) == 14

    # --- Self-check: the structural columns (u[:, :k]) are array_equal and the sig_hn column differs (only when the
    #     current code is the corrected one -- before the correction the two samplers are literally identical, so
    #     this assert is skipped below).
    structural_identical, sig_hn_differs = 0, 0
    for name in rg.STRUCTURES:
        cand = core.Candidate(name, "gauss", 0)
        kinds = cand.kinds()
        k = len(core.STRUCTURES[name].kinds)
        expr = fbp.structure_expression(name)
        roles = fbp.structure_roles(name)
        mid = fbp.model_id(expr, "gauss", roles)
        probe_key = jax.random.fold_in(jax.random.PRNGKey(rg.PRIOR_PROBE_SEED), int(mid[:8], 16))
        u_before = np.asarray(_pre_fix_sampler(probe_key, rg.PRIOR_PROBE_SAMPLES, kinds))
        u_after = np.asarray(core.sample_prior_u(probe_key, rg.PRIOR_PROBE_SAMPLES, kinds))
        assert np.array_equal(u_before[:, :k], u_after[:, :k]), \
            f"{name}: a structure column changed with the sampler correction (it must not)"
        structural_identical += 1
        if not np.array_equal(u_before[:, k], u_after[:, k]):
            sig_hn_differs += 1
    print(f"[check 4] structural columns identical for {structural_identical}/14 structures; "
          f"sig_hn column differs (pre-fix vs current) for {sig_hn_differs}/14 "
          f"(0/14 means the tree under test is still pristine/unfixed)")

    # --- Execution check: call the real _prior_probe with each of the two samplers and compare the verdicts.
    orig_sampler = core.sample_prior_u
    before, after = {}, {}
    try:
        for name in rg.STRUCTURES:
            expr = fbp.structure_expression(name)
            roles = fbp.structure_roles(name)
            mid = fbp.model_id(expr, "gauss", roles)
            core.sample_prior_u = _pre_fix_sampler
            before[name] = rg._prior_probe(name, "gauss", mid, p1a, p2a)
            core.sample_prior_u = orig_sampler
            after[name] = rg._prior_probe(name, "gauss", mid, p1a, p2a)
    finally:
        core.sample_prior_u = orig_sampler
    assert len(before) == 14 and len(after) == 14
    mismatches = [
        name for name in before
        if (before[name]["finite"], before[name]["nonfinite_fraction"])
        != (after[name]["finite"], after[name]["nonfinite_fraction"])
    ]
    assert mismatches == [], f"structures whose prior-probe verdict changed: {mismatches}"
    print(f"[check 4] prior-probe verdicts (finite, nonfinite_fraction) identical "
          f"before/after for all {len(before)} catalogue structures")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
