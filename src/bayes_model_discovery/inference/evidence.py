"""Dependency-light evidence combination shared by execution and analysis."""
import math

from scipy.special import logsumexp


class EvidenceError(ValueError):
    """Evidence values cannot define the registered combination."""


def combine_log_evidence(values):
    """Average evidences in linear space while receiving log-space values."""
    vals = [float(value) for value in values]
    if not vals:
        raise EvidenceError("cannot combine an empty set of log evidences")
    if not all(math.isfinite(value) for value in vals):
        raise EvidenceError("cannot combine non-finite log evidence")
    return float(logsumexp(vals) - math.log(len(vals)))
