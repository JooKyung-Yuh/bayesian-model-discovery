"""Secondary predictive-information contract; no closed-loop policy is enabled.

Y_q and Y_x are separate future observations, independent conditional on the
same posterior particle (model and parameters). Even for x_q == x they are not
the same random variable. A product of already mixed marginals is incorrect.
"""

import numpy as np
from scipy.special import xlogy

from .count_inference import model_logweights
from .count_models import NORMALIZATION_FAILURE, particle_logpmf, zeroed_particles


def uniform_quadrature(count=32):
    """Fixed Gauss-Legendre integration of the target Uniform(0,1) input law."""
    if type(count) is not int or count < 2:
        raise ValueError("quadrature needs at least two fixed nodes")
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return (nodes + 1) / 2, weights / 2


def _weights(values, size, label):
    values = np.asarray(values, dtype=float)
    if (
        values.shape != (size,)
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
        or not np.isclose(values.sum(), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError(label + " weights must be nonnegative and normalized")
    return values


def _bank(values):
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 3
        or 0 in values.shape
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
        or not np.allclose(values.sum(-1), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError("PMF bank must be normalized [same-particle, input, outcome]")
    return values


def predictive_information(q_pmfs, x_pmfs, particle_weights, q_weights):
    """Sum_q omega_q I(Y_q;Y_x|D), exact in the supplied finite particle bank."""
    q_pmfs, x_pmfs = _bank(q_pmfs), _bank(x_pmfs)
    if q_pmfs.shape[0] != x_pmfs.shape[0]:
        raise ValueError("q and x must use the same ordered posterior particles")
    w = _weights(particle_weights, len(q_pmfs), "particle")
    omega = _weights(q_weights, q_pmfs.shape[1], "quadrature")
    from .count_inference import EPIG_DISCRETE_MAX_WORKING_BYTES, EpigResourceLimit

    particles, targets, q_outcomes = q_pmfs.shape
    _, candidates, x_outcomes = x_pmfs.shape
    # Detect exact zero-information limits using only positive-weight particles.
    # Compare one particle at a time so this phase never copies a full bank.
    detection_bytes = (q_pmfs.nbytes + x_pmfs.nbytes + w.nbytes + omega.nbytes
                       + max(targets * q_outcomes + 3 * targets,
                             candidates * x_outcomes + 3 * candidates))
    if detection_bytes > EPIG_DISCRETE_MAX_WORKING_BYTES:
        raise EpigResourceLimit({"resource": "predictive_information_arrays",
                                 "estimated_working_bytes": detection_bytes,
                                 "max_working_bytes": EPIG_DISCRETE_MAX_WORKING_BYTES})
    preserve_roundoff = np.count_nonzero(w) == 1
    if not preserve_roundoff:
        reference_particle = int(np.argmax(w))
        for bank in (q_pmfs, x_pmfs):
            constant_inputs = np.ones(bank.shape[1], dtype=bool)
            for ki in range(particles):
                if w[ki] > 0 and ki != reference_particle:
                    constant_inputs &= np.all(bank[ki] == bank[reference_particle], axis=-1)
                    if not np.any(constant_inputs):
                        break
            if np.any(constant_inputs):
                preserve_roundoff = True
                break
        del constant_inputs
    # Retained logical banks, worst-case contiguous x copy, mixtures, reductions,
    # output, and validation temporaries. This bounds ndarray workspace, not RSS.
    fixed_bytes = (q_pmfs.nbytes + 2 * x_pmfs.nbytes + w.nbytes + omega.nbytes
                   + 8 * (targets * q_outcomes + candidates * x_outcomes
                          + 4 * targets + 6 * candidates + targets * candidates
                          + particles * candidates))
    weight_x = candidates * x_outcomes < q_outcomes
    # Weight q into a contiguous buffer; GEMM and in-place xlogy share one joint
    # table. Extra reduced arrays cover normalization and entropy checks.
    bytes_per_target = 8 * (particles * q_outcomes
                           + (2 if weight_x or preserve_roundoff else 1) * q_outcomes * candidates * x_outcomes
                           + 6 * candidates)
    batch_size = 1 if preserve_roundoff else targets
    while (batch_size > 1
           and fixed_bytes + batch_size * bytes_per_target > EPIG_DISCRETE_MAX_WORKING_BYTES):
        batch_size = max(1, batch_size // 2)
    # x_matrix is constructed only after marginal/conditional entropy work.
    entropy_bytes = fixed_bytes - x_pmfs.nbytes + max(q_pmfs.nbytes, x_pmfs.nbytes)
    estimated_bytes = max(fixed_bytes + batch_size * bytes_per_target, entropy_bytes, detection_bytes)
    if estimated_bytes > EPIG_DISCRETE_MAX_WORKING_BYTES:
        raise EpigResourceLimit({"resource": "predictive_information_arrays",
                                 "estimated_working_bytes": estimated_bytes,
                                 "max_working_bytes": EPIG_DISCRETE_MAX_WORKING_BYTES})
    entropy = lambda p: -np.sum(xlogy(p, p), axis=-1)
    q_mixture = np.einsum("k,kqy->qy", w, q_pmfs)
    x_mixture = np.einsum("k,kxy->xy", w, x_pmfs)
    hq, hx = entropy(q_mixture), entropy(x_mixture)
    joint_eig = hx - np.einsum("k,kx->x", w, entropy(x_pmfs))
    per_q = np.empty((targets, candidates))
    if preserve_roundoff:
        x_matrix = None
    elif weight_x:
        x_matrix = np.empty((particles, candidates * x_outcomes))
        np.multiply(x_pmfs, w[:, None, None],
                    out=x_matrix.reshape(particles, candidates, x_outcomes))
    else:
        x_matrix = np.ascontiguousarray(x_pmfs).reshape(particles, candidates * x_outcomes)
    for start in range(0, targets, batch_size):
        stop = min(start + batch_size, targets)
        count = stop - start
        if preserve_roundoff:
            # Preserve the original contraction and reduction in exact
            # zero-information limits, including their signed roundoff.
            for qi in range(start, stop):
                joint = np.einsum("k,ky,kxz->xyz", w, q_pmfs[:, qi, :], x_pmfs, optimize=True)
                if not np.allclose(joint.sum(axis=(1, 2)), 1, atol=1e-10, rtol=0):
                    raise ValueError("joint predictive normalization failed")
                h_joint = -np.sum(xlogy(joint, joint), axis=(1, 2))
                per_q[qi] = hq[qi] + hx - h_joint
                del joint, h_joint
            continue
        weighted_q = np.empty((particles, count * q_outcomes))
        # Match einsum's smaller-factor weighting and reduction layout. Both
        # branches mix the same k before entropy, never mixed marginals.
        if weight_x:
            np.copyto(weighted_q.reshape(particles, count, q_outcomes), q_pmfs[:, start:stop, :])
            joint = (x_matrix.T @ weighted_q).reshape(
                candidates, x_outcomes, count, q_outcomes).transpose(2, 0, 3, 1)
        else:
            np.multiply(q_pmfs[:, start:stop, :], w[:, None, None],
                        out=weighted_q.reshape(particles, count, q_outcomes))
            joint = (weighted_q.T @ x_matrix).reshape(
                count, q_outcomes, candidates, x_outcomes).transpose(0, 2, 1, 3)
        if weight_x:
            # Target-interleaving otherwise changes NumPy's pairwise sum order.
            joint = np.ascontiguousarray(joint.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
        if not np.allclose(joint.sum(axis=(2, 3)), 1, atol=1e-10, rtol=0):
            raise ValueError("joint predictive normalization failed")
        xlogy(joint, joint, out=joint)
        h_joint = -np.sum(joint, axis=(2, 3))
        per_q[start:stop] = hq[start:stop, None] + hx[None, :] - h_joint
        # Do not overlap the previous joint/weighted buffer with the next batch.
        del joint, weighted_q, h_joint
    epig = omega @ per_q
    tolerance = 1e-9
    if (
        np.any(per_q < -tolerance)
        or np.any(joint_eig < -tolerance)
        or np.any(per_q > joint_eig[None, :] + tolerance)
        or not np.all(np.isfinite(epig))
    ):
        raise ValueError("predictive information violated nonnegativity/data-processing bound")
    return {
        "epig": epig,
        "per_q_information": per_q,
        "joint_eig": joint_eig,
        "roundoff_tolerance": tolerance,
        "negative_values_clipped": False,
    }


def from_fits(fits, designs, quadrature_count=32, posterior_samples=None, seed=2718):
    """No response values or evaluation targets enter this API."""
    nodes, omega = uniform_quadrature(quadrature_count)
    designs = np.asarray(designs, dtype=float)
    if (
        designs.ndim != 1
        or not len(designs)
        or not np.all(np.isfinite(designs))
        or np.any((designs < 0) | (designs > 1))
    ):
        raise ValueError("CORE designs must lie in [0,1]")
    if posterior_samples is not None and (
        type(posterior_samples) is not int or posterior_samples < 1
    ):
        raise ValueError("posterior sample count must be a positive integer")
    rng = np.random.default_rng(seed)
    q_banks, x_banks, weights, inventory = [], [], [], []
    for result, model_weight in zip(fits, np.exp(model_logweights(fits)), strict=True):
        count = len(next(iter(result.particles.values())))
        indices = (
            np.arange(count)
            if posterior_samples is None
            else rng.choice(count, min(count, posterior_samples), replace=False)
        )
        subset = {name: values[indices] for name, values in result.particles.items()}
        q_rows = particle_logpmf(result.hypothesis, subset, nodes)
        x_rows = particle_logpmf(result.hypothesis, subset, designs)
        # Y_q and Y_x are conditioned on the SAME particle, so a
        # particle given zero weight at either the targets or the designs leaves
        # both banks and the inventory together, and the model's own mass is
        # redistributed over the particles it still has.
        keep = ~(zeroed_particles(q_rows) | zeroed_particles(x_rows))
        if not keep.all():
            if not keep.any():
                raise ValueError(NORMALIZATION_FAILURE)
            q_rows, x_rows, indices = q_rows[keep], x_rows[keep], indices[keep]
        q_banks.append(np.exp(q_rows))
        x_banks.append(np.exp(x_rows))
        weights.append(np.full(len(indices), model_weight / len(indices)))
        inventory.append({"model_id": result.hypothesis.model_id, "indices": indices.tolist()})
    answer = predictive_information(
        np.concatenate(q_banks), np.concatenate(x_banks), np.concatenate(weights), omega
    )
    best_q, best_joint = int(np.argmax(answer["epig"])), int(np.argmax(answer["joint_eig"]))
    return {
        **answer,
        "q_nodes": nodes,
        "q_weights": omega,
        "designs": designs,
        "x_q": float(designs[best_q]),
        "x_joint": float(designs[best_joint]),
        "R_Q": float(answer["epig"][best_q] - answer["epig"][best_joint]),
        "target_input_distribution": "Uniform(0,1)",
        "particle_inventory": inventory,
        "seed": seed,
        "quadrature_count": quadrature_count,
        "test_targets_used": 0,
        "closed_loop_enabled": False,
    }


def quadrature_comparison(fits, designs, coarse=32, dense=128, posterior_samples=None, seed=2718):
    """Compare coarse and dense target quadrature on one shared particle bank; return both results and the max EPIG gap."""
    if dense <= coarse:
        raise ValueError("dense quadrature must exceed coarse quadrature")
    left = from_fits(fits, designs, coarse, posterior_samples, seed)
    right = from_fits(fits, designs, dense, posterior_samples, seed)
    assert left["particle_inventory"] == right["particle_inventory"]
    return {
        "coarse": left,
        "dense": right,
        "max_absolute_epig_difference": float(np.max(np.abs(left["epig"] - right["epig"]))),
        "same_particle_bank": True,
    }
