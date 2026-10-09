"""Direct force-curve diagnostic: measures the effective force F(r) of the DP FieldSampler and compares it with the
analytic and softened forms.

This is the one direct force-curve diagnostic per world.  It reads the force
directly, without going through trajectory integration.  The FieldSampler is set up **the same way** as in the executor
(executor._run_one): particle 1 = box centre, particle 2 = centre + r.
"""
import sys, json, math, argparse, os
# DiscoverPhysics checkout (33b7fa9): DP_SCIENCEAGENT_DIR is its ScienceAgent folder, PhysicsSchool sits beside it.
_DP = os.environ["DP_SCIENCEAGENT_DIR"]
sys.path.insert(0, _DP)
sys.path.insert(0, os.path.join(os.path.dirname(_DP), "PhysicsSchool"))
import numpy as np, jax.numpy as jnp
from physchool.worlds.field_sampler import FieldSampler
from scienceagent.worlds import WORLDS

def force_of_r(world, rs, grid=64, domain=20.0, p1=1.0, p2=1.0, dt=0.005, smoothing=None):
    """Measure, at each r in rs, the **magnitude of the radial attractive force** on particle 2; returns an array."""
    ops = WORLDS[world]["executor_kwargs"]["operators"]
    torder = WORLDS[world]["executor_kwargs"].get("temporal_order", 0)
    c = domain / 2.0
    out = []
    for r in rs:
        sim = FieldSampler(
            particle_inertia=np.array([1, p2]), particle_source=np.array([p1, 1.0]),
            particle_force=np.array([0.0, 1.0]),
            initial_positions=np.array([[c, c], [c + float(r), c]], dtype=np.float64),
            initial_velocities=np.zeros((2, 2)), n_particles=2, spatial_dimensions=2,
            temporal_order=torder, grid_size=(grid, grid), domain_size=domain,
            operators=ops, dt=dt, source_coupling=np.array([p1, 1.0]),
            force_coupling=1.0, periodic_boundaries=True,
            **({"source_smoothing": smoothing} if smoothing is not None else {}))
        f = np.asarray(sim._step_constraint(sim._paint_sources()))
        out.append(-float(f[1][0]))          # particle 2 sits on the +x side, so attraction is the -x component
    return np.array(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--world", default="gravity")
    a = ap.parse_args(); W = a.world
    rs = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
    res = {"world": W, "domain_size": 20.0, "r": rs.tolist(), "grids": {}}
    print(f"=== {W} · effective force F(r), p1=p2=1 · box 20.0 ===")
    print(f"{'r':>6} {'r/dx(64)':>9} " + " ".join(f"{'F(grid '+str(g)+')':>14}" for g in (64,128,256)) + f" {'analytic 1/(2pi r)':>19}")
    F = {g: force_of_r(W, rs, grid=g) for g in (64, 128, 256)}
    Fa = 1.0 / (2 * math.pi * rs)
    for i, r in enumerate(rs):
        print(f"{r:6.2f} {r/(20.0/64):9.1f} " + " ".join(f"{F[g][i]:14.6e}" for g in (64,128,256)) + f" {Fa[i]:15.6e}")
    for g in (64,128,256): res["grids"][str(g)] = {"F": F[g].tolist(), "dx": 20.0/g}
    res["F_analytic"] = Fa.tolist()

    print(f"\n{'r':>6} {'F64/analytic':>13} {'F128/analytic':>14} {'F256/analytic':>14}")
    for i, r in enumerate(rs):
        print(f"{r:6.2f} {F[64][i]/Fa[i]:10.4f} {F[128][i]/Fa[i]:10.4f} {F[256][i]/Fa[i]:10.4f}")

    # form fits: Gaussian softening vs Plummer softening (diagnostic values -- never put into a prior)
    from scipy.optimize import least_squares
    def fit(model, x0, g):
        """Least-squares fit of `model` to the grid-g force curve; returns (parameters, max relative error)."""
        f = lambda p: model(rs, *p) - F[g]
        s = least_squares(f, x0)
        return s.x, float(np.max(np.abs(s.fun / F[g])))
    gauss  = lambda r, A, s: A / (2*math.pi*r) * (1 - np.exp(-r**2/(2*s**2)))
    plummer= lambda r, A, e: A / (2*math.pi*np.sqrt(r**2 + e**2))
    pure   = lambda r, A:    A / (2*math.pi*r)
    print("\nform fits (diagnostic values; never put into a prior)")
    for g in (64, 128, 256):
        (Ag, sg), eg = fit(gauss, [1.0, 20.0/g], g)
        (Ap, ep), ep_ = fit(plummer, [1.0, 20.0/g], g)
        (Au,), eu = fit(pure, [1.0], g)
        res["grids"][str(g)].update(gauss_fit=[Ag, sg], gauss_maxrel=eg,
                                    plummer_fit=[Ap, ep], plummer_maxrel=ep_,
                                    pure_fit=[Au], pure_maxrel=eu)
        print(f"  grid {g:3d} (dx {20.0/g:.4f}):  pure 1/r  A={Au:.4f} max rel. error {eu:.4f}")
        print(f"                       Gaussian  A={Ag:.4f} s={sg:.4f} max rel. error {eg:.4f}")
        print(f"                       Plummer   A={Ap:.4f} e={ep:.4f} max rel. error {ep_:.4f}")
    p = os.path.join(os.environ.get("FB1_OUT_DIR", "."), f"FB1_FORCE_CURVE_{W}.json")
    json.dump(res, open(p, "w"), indent=1); print("wrote", p)
