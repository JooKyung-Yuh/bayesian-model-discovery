"""Our metric on the Apprentice's predictions: the LLM-AutoSciLab wrapper's BoxingGym scoring rules, applied to an array.

`unobserved_q_mask` and `nmse_truth` are copied verbatim from the AutoSciLab kit
(unified_run_20261005/autoscilab_kit/asl_kit/boxinggym.py, sha256 52ef85f8...; there they are boxinggym_core's
rules). `BoxingGymScorer.score(law_str, observed)` there evaluates a Python law at Q; here the predictions at the
256 Q points come from the box loop's program (posterior-predictive mean), so `score_predictions` takes the array and
applies the same rules: Q minus the executed designs (exact equality), nmse_truth = mean((pred - E[Y|x])^2) /
Var_Q(E[Y|x]), binary tasks also brier = mean((clip(pred, 0, 1) - y_heldout)^2), and any exception or non-finite
prediction at a kept point -> law_error. tests/test_kit.py checks both scorers give the same numbers.
"""
import numpy as np


def unobserved_q_mask(evaluation_x, observed):
    """boxinggym_core.unobserved_q_mask, verbatim rule: exact equality in declared input units."""
    q = np.asarray(evaluation_x, dtype=float)
    q = q[:, None] if q.ndim == 1 else q
    known = np.asarray(observed, dtype=float)
    if known.size == 0:
        return np.ones(len(q), dtype=bool)
    known = known.reshape(-1, q.shape[1])
    return ~np.any(np.all(q[:, None, :] == known[None, :, :], axis=2), axis=1)


def nmse_truth(predictions, truth_mean):
    """boxinggym_core.mda_score_row's nmse_truth for one output component."""
    truth = np.asarray(truth_mean, dtype=float)
    means = np.asarray(predictions, dtype=float)
    variance = np.var(truth, axis=0)
    if truth.size and np.isfinite(truth).all() and np.all(variance > 0):
        return float(np.mean(np.mean((means - truth) ** 2, axis=0) / variance))
    return None


def score_predictions(pred_all, errors, qx, truth, heldout, binary, observed):
    """One curve row from predictions at all Q points.

    pred_all: float array over the 256 Q points (NaN where the program raised); errors: {index: message};
    observed: the executed designs (declared input units) up to this budget.
    """
    qx = np.asarray(qx, dtype=float)
    keep = unobserved_q_mask(qx, observed if observed is not None else [])
    kept_idx = np.flatnonzero(keep)
    raised = [i for i in kept_idx if int(i) in errors]
    if raised:
        return {"status": "law_error", "error": str(errors[int(raised[0])])[:300]}
    pred = np.asarray(pred_all, dtype=float)[keep]
    if pred.shape != (int(keep.sum()),) or not np.isfinite(pred).all():
        return {"status": "law_error",
                "error": f"non-finite or misshaped predictions ({int((~np.isfinite(pred)).sum())} non-finite)"}
    row = {"status": "scored", "nmse_truth": nmse_truth(pred, np.asarray(truth, dtype=float)[keep]),
           "q_points": int(keep.sum()), "q_excluded": int((~keep).sum())}
    if binary:
        row["brier"] = float(np.mean((np.clip(pred, 0.0, 1.0) - np.asarray(heldout, dtype=float)[keep]) ** 2))
    return row
