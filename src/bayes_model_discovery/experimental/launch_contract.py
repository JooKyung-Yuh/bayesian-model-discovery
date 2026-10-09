"""The registered launch contract's two validators, shared by the label policy.

Verbatim from the registered research-wave driver (make_research_wave.py, kept in the lab history, not in this release),
so the installed package needs no `tools` import: the CORE repair-2 authority (run_label_policy) reuses the registered
driver's schema, not a second channel.
"""

RGT_ADDED_KEYS = ("core_track", "per_host_parallel_cap_policy",
                  "baseline_cgroup_memory_bytes", "per_cell_peak_increment_bytes",
                  "memory_safety_reserve_bytes", "parallel_requested",
                  "track_global_inflight_cap", "global_budget_ledger",
                  "per_cell_output_bytes", "disk_safety_reserve_bytes")


def validate_launch_contract(value, *, required=False):
    """Validate a launch contract dict and return a copy; None is allowed unless required."""
    if value is None:
        if required:
            raise ValueError("A registered launch requires an explicitly registered launch contract")
        return None
    if not isinstance(value, dict):
        raise ValueError("Launch contract must name memory and disk terms, global cap and ledger")  # noqa: TRY004 - Registration validation exposes ValueError.
    if value.get("per_host_parallel_cap_policy") != "memory_derived":
        raise ValueError("per_host_parallel_cap_policy must be memory_derived")
    for key in ("per_cell_peak_increment_bytes", "parallel_requested", "track_global_inflight_cap",
                "per_cell_output_bytes"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ValueError(key + " must be an explicitly declared positive integer")
    for key in ("baseline_cgroup_memory_bytes", "memory_safety_reserve_bytes", "disk_safety_reserve_bytes"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError(key + " must be an explicitly declared nonnegative integer")
    if not isinstance(value.get("global_budget_ledger"), str) or not value["global_budget_ledger"].strip():
        raise ValueError("global_budget_ledger must explicitly name the shared ledger")
    if set(value) != set(RGT_ADDED_KEYS[1:]):
        raise ValueError("Unknown launch contract field")
    return dict(value)


def validate_launch_prerequisites(record):
    """The launch contract supplements, rather than replaces, the existing bindings."""
    # A ChemBench track is bound by the same two, for its own bundle's cap.
    for track, bundle in (("core_track", "core"), ("chembench_track", "chembench")):
        if record.get(track) is None:
            continue
        rule = record.get("execution_commit_rule")
        if not isinstance(rule, dict) or not rule.get("rule"):
            raise ValueError("A registered launch requires execution_commit_rule")
        caps = record.get("cell_timeout_minutes")
        if (not isinstance(caps, dict) or type(caps.get(bundle)) is not int
                or caps[bundle] <= 0):
            raise ValueError("A registered launch requires cell_timeout_minutes for " + bundle)
