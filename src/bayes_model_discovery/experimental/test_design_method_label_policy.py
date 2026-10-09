"""Scalar study policies bind to exact row variants without changing old labels."""
import pytest

from .run_label_policy import bmd_runtime_settings, recheck_manifest, resolve_label
from .test_run_label_policy import (
    digest,
    registration,
    runner_args,
    write_manifest,
)


@pytest.mark.parametrize("bundle", ["core", "boxinggym", "chembench"])
@pytest.mark.parametrize("policy", ["point_disagreement", "space_filling"])
@pytest.mark.parametrize("change", ["none", "missing_token", "other_token", "unknown_token",
                                     "negative_cost", "nonfinite_cost", "missing_runtime", "epig_cell"])
def test_exact_variant_row_and_scalar_runtime_binding(tmp_path, bundle, policy, change):
    # An LLM bundle declares the prompt it ran, so the namespace carries the
    # pair the runner puts on it before the receipt.
    """Check that exact variant row and scalar runtime binding."""
    args = runner_args(rounds=2, cell="101", design_policy=policy)
    settings = bmd_runtime_settings(bundle, args)
    assert settings["design_policy"] == policy
    ref, context, manifest = registration(tmp_path, cell="101", bundle=bundle,
                                          task="synthetic_task", seed=7, settings=settings)
    tokens = [bundle, "synthetic_task", "101", "7", "0.10", "design_" + policy]
    if change == "missing_token":
        tokens.pop()
    elif change == "other_token":
        tokens[-1] = "design_" + ("space_filling" if policy == "point_disagreement" else "point_disagreement")
    elif change == "unknown_token":
        tokens[-1] = "design_unregistered"
    elif change == "negative_cost":
        tokens[4] = "-1"
    elif change == "nonfinite_cost":
        tokens[4] = "nan"
    elif change == "missing_runtime":
        context["settings"] = bmd_runtime_settings(bundle, runner_args(rounds=2, cell="101"))
    elif change == "epig_cell":
        tokens[2] = "102"
        context["cell"] = "102"
    rows = tmp_path / "rows.txt"
    rows.write_text(" ".join(tokens) + "\n")
    manifest["calls"][0]["sha256"] = digest(rows.read_bytes())
    ref = write_manifest(tmp_path / "manifest.json", manifest)
    if change != "none":
        with pytest.raises(ValueError):
            resolve_label("production", manifest_ref=ref, **context)
    else:
        receipt = resolve_label("production", manifest_ref=ref, **context)
        assert receipt["matched_row"]["variant"] == "design_" + policy
        assert recheck_manifest(receipt) == receipt


@pytest.mark.parametrize("policy", [None, "joint_eig"])
def test_boxinggym_default_scalar_field_remains_absent(policy):
    """Check that boxinggym default scalar field remains absent."""
    args = runner_args(rounds=20, cell="101", design_policy=policy)
    assert "design_policy" not in bmd_runtime_settings("boxinggym", args)
