"""the recovery judge: a deterministic structure-equivalence judge for the four benchmarks of the shared run protocol.

Every benchmark module exposes ``judge(candidate, target) -> dict`` with the common record
``{state, rule_version, target_id, candidate_expression, canonical_form, parameter_mapping, containment, reason, ...}``
and ``targets_for(...)`` that declares the world's target family.  States: equivalent, non_equivalent,
unresolved, not_representable.  No language model is used anywhere.
"""
from .core import RULE_VERSION, STATES  # noqa: F401
