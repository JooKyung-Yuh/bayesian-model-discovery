"""Pool-key coexistence across the CORE file-pool and ChemBench generated-pool features.

The CORE file-pool feature binds the runtime-settings key `initial_pool_source` to a CORE frozen pool file
(paired with `initial_pool_sha256` and `initial_pool_count`, catalogue off, no burst). The
ChemBench generated-pool feature binds the same key to a ChemBench generated pool (paired with a burst:
`burst_prompt_sha256` and `initial_proposals`, catalogue off). A naive text merge of the two
features broke both: (1) the CORE file-pool feature's registered-settings check rejected any settings containing a pool
key on a non-CORE bundle, so a registered ChemBench generated pool was refused; (2) the ChemBench feature's
relocated "initial_catalogue off needs a burst" check refused a CORE file-pool run that has no
burst. These tests pass on the resolved run_label_policy.py and would fail on the naive merge.
"""
from types import SimpleNamespace

import pytest

from .count_models import INITIAL_BURST_REFILL_POLICY
from .run_label_policy import (
    BURST_REFILL_KEYS,
    BURST_REFILL_POLICY_KEY,
    bmd_runtime_settings,
    resolve_label,
)
from .test_run_label_policy import registration, runner_args

SHA = "a" * 64
#: A burst with initial proposals also states two refill digests (synthetic, like SHA) and the
#: refill rule, bound beside them.
REFILL = {**dict.fromkeys(BURST_REFILL_KEYS, SHA), BURST_REFILL_POLICY_KEY: INITIAL_BURST_REFILL_POLICY}


def test_registered_chembench_generated_pool_resolves(tmp_path):
    # The ChemBench burst states its refill keys.
    """Check that registered chembench generated pool resolves."""
    settings = bmd_runtime_settings("chembench", runner_args(
        burst_prompt_sha256=SHA, initial_proposals=6, initial_catalogue="off",
        initial_pool_source="generated", **REFILL))
    assert settings["initial_pool_source"] == "generated" and settings["initial_catalogue"] == "off"
    ref, context, _ = registration(tmp_path, bundle="chembench", settings=settings)
    resolve_label("production", manifest_ref=ref, **context)


def test_registered_core_file_pool_resolves_without_a_burst(tmp_path):
    """Check that registered core file pool resolves without a burst."""
    settings = bmd_runtime_settings("core", runner_args(
        initial_pool_sha256=SHA, initial_pool_source="file_burst", initial_pool_count=4,
        initial_catalogue="off"))
    assert "burst_prompt_sha256" not in settings and "initial_proposals" not in settings
    assert (settings["initial_pool_source"] == "file_burst"
            and settings["initial_pool_count"] == 4
            and settings["initial_catalogue"] == "off")
    ref, context, _ = registration(tmp_path, bundle="core", settings=settings)
    resolve_label("production", manifest_ref=ref, **context)


@pytest.mark.parametrize("bundle,extra", [
    ("chembench", {"initial_pool_sha256": SHA, "initial_pool_count": 4}),
    ("core", {"initial_pool_source": "generated"}),
], ids=["chembench", "core"])
def test_file_pool_keys_stay_core_only(tmp_path, bundle, extra):
    """Check that file pool keys stay core only."""
    if bundle == "chembench":
        # The ChemBench burst states its refill keys.
        base = bmd_runtime_settings("chembench", runner_args(
            burst_prompt_sha256=SHA, initial_proposals=6, initial_catalogue="off",
            initial_pool_source="generated", **REFILL))
    else:
        base = bmd_runtime_settings("core", runner_args(
            initial_pool_sha256=SHA, initial_pool_source="file_burst", initial_pool_count=4,
            initial_catalogue="off"))
    modified = {**base, **extra}
    ref, context, _ = registration(tmp_path, bundle=bundle, settings=modified)
    with pytest.raises(ValueError, match="Registered initial pool requires a complete file binding"):
        resolve_label("production", manifest_ref=ref, **context)


# Expected orders were captured from the two parents before the merge (CORE and BoxingGym from
# the CORE/BoxingGym parent, ChemBench from its generated-pool parent); a merge must not reorder registered keys.
# Re-pin of the registered key order: notation_normal_form is bound right after
# model_id_schema in every case; each burst case states REFILL, whose three keys (two digests,
# then the refill rule) are bound right after burst_prompt_sha256. Every other key keeps its
# captured place.
@pytest.mark.parametrize("bundle,extra,expected", [
    ("core", {"burst_prompt_sha256": SHA, "initial_proposals": 6, "initial_catalogue": "off", **REFILL},
     ["rounds", "particles", "chains", "model_id_schema", "notation_normal_form", "d0_policy",
      "proposal_policy", "prompt_version", "prompt_sha256", "initial_proposals",
      "burst_prompt_sha256", "burst_refill_prompt_sha256", "burst_refill_paragraph_sha256",
      "initial_burst_refill_policy", "initial_catalogue", "design_policy", "scientific_options"]),
    ("core", {"initial_pool_sha256": SHA, "initial_pool_source": "file_burst",
              "initial_pool_count": 4, "initial_catalogue": "off"},
     ["rounds", "particles", "chains", "model_id_schema", "notation_normal_form", "d0_policy",
      "proposal_policy", "prompt_version", "prompt_sha256", "initial_pool_source",
      "initial_pool_sha256", "initial_pool_count", "initial_catalogue", "design_policy",
      "scientific_options"]),
    ("chembench", {"burst_prompt_sha256": SHA, "initial_proposals": 6, "initial_catalogue": "off",
                   "initial_pool_source": "generated", **REFILL},
     ["rounds", "particles", "chains", "model_id_schema", "notation_normal_form", "d0_policy", "q",
      "d0_fixed_inputs", "prompt_version", "prompt_sha256", "initial_proposals",
      "burst_prompt_sha256", "burst_refill_prompt_sha256", "burst_refill_paragraph_sha256",
      "initial_burst_refill_policy", "initial_pool_source", "initial_catalogue", "design_policy",
      "scientific_options"]),
    ("boxinggym", {"burst_prompt_sha256": SHA, "initial_proposals": 6, "initial_catalogue": "off", **REFILL},
     ["rounds", "particles", "chains", "model_id_schema", "notation_normal_form", "d0_policy",
      "q_policy", "proposal_policy", "prompt_version", "prompt_sha256", "initial_proposals",
      "burst_prompt_sha256", "burst_refill_prompt_sha256", "burst_refill_paragraph_sha256",
      "initial_burst_refill_policy", "initial_catalogue", "scientific_options"]),
], ids=["core-burst", "core-file-pool", "chembench-burst", "boxinggym-burst"])
def test_runtime_settings_key_order_is_unchanged(bundle, extra, expected):
    """Check that runtime settings key order is unchanged."""
    common = {"profile": "paper", "rounds": 20, "cell": "100", "proposal_policy": "P1",
              "prompt_version": "v2.0", "prompt_sha256": "b" * 64}
    args = SimpleNamespace(**common, **extra)
    assert list(bmd_runtime_settings(bundle, args)) == expected
