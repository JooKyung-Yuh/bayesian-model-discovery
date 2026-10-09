"""Shared numerical constants that define production probability calculations.

This module is deliberately dependency-free. Likelihood evaluation, symbolic SMC,
candidate fitting, and both EIG posterior views all need the same open-unit-interval
boundary. Keeping the value here prevents those paths from silently describing
different observation models.
"""

EPS_P = 1e-9

# The registered decimal precision of STORED EIG table values. Storage-rounded
# values are presentation and archive bytes, never decision inputs: the tie rule
# compares unrounded values (see ``eig.joint.choose_design``), because applying a
# 1e-12 tolerance to 8-dp-rounded numbers silently widens the executed tie window
# to the 1e-8 storage quantum (the world-314 r9/r10 corrected-record incident).
EIG_STORAGE_DECIMALS = 8

__all__ = ("EIG_STORAGE_DECIMALS", "EPS_P")
