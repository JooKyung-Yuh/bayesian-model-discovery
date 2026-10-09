"""Byte pins of the count proposer's system text, written before the ChemBench / BoxingGym count_proposer reconciliation.

Whatever the reconciled code looks like, the bytes every registered and paid run sent must not move:
- k = 3 with composition v2 (all unified-protocol cells of ChemBench, BoxingGym and ForceBench): composed 0ed50159..., and the
  archive-review round a9633b34... (as recorded in those cells' config.json).
- k = 1 (every earlier cell): the system text is the prompt file alone, exactly as the development tree composed it,
  and its identity block carries the two prompt keys only.
The expected values were measured on the development tree with the same calls.
"""

import hashlib
import inspect
from pathlib import Path

import pytest

from . import count_proposer as cp

ROOT = Path(__file__).resolve().parents[3]
FAC = ROOT / "tools" / "fb1_launch_c" / "proposer_system_v2_9_FA-C.txt"
FAC_SHA256 = "196f82b7a0f94e12a7cb8be2d07e776b1536aed11bf8ebd4bea1872c7002953c"
V20_SHA256 = "f095edf1adb3b6533a9e68e73f98ca1dee3f23cd077e424b092393cb32b93aef"
COMPOSED_K3_V2_SHA256 = "0ed501592667716e968224697669f674bbcb36360be09a75afab86d1a90d6071"
COMPOSED_K3_V2_REVIEW_SHA256 = "a9633b34e2e9561216343487f37ed87700662f732440b2a2ecca6f96eea917c9"
COMPOSITION_V2_SHA256 = "782a30b74cd25adb6b51351096cf97e31a7375d8630d726ee8eb7d7949c35be6"
COMPOSED_K3_V1_SHA256 = "eee83b13ca0296f46df4f1c0851ec7be9d5510313bb885d92990a046f56a79e8"


def sha(composed):
    """Return the sha256 hex digest of a composed prompt."""
    text = composed[0] if isinstance(composed, tuple) else composed
    return hashlib.sha256(text.encode()).hexdigest()


def review_keyword():
    """The archive-review keyword of composed_system_prompt, before or after the rename."""
    names = inspect.signature(cp.composed_system_prompt).parameters
    return "archive_review" if "archive_review" in names else "review"


def select_composition(version):
    """Under the toggle design the paragraph is chosen explicitly; a fixed-v2-at-k=3 design needs no call."""
    if hasattr(cp, "set_composition_paragraph"):
        cp.set_composition_paragraph(version)


@pytest.fixture(autouse=True)
def restore_selection():
    """Fixture restoring the prompt selection after the test."""
    previous = cp.set_prompt_variant(None)
    yield
    cp.set_prompt_variant(previous[0])
    select_composition("v1")


def test_the_fac_file_is_the_registered_one():
    """Check that the fac file is the registered one."""
    assert hashlib.sha256(FAC.read_bytes()).hexdigest() == FAC_SHA256


def test_k1_default_is_the_v20_file_and_its_identity_has_only_the_prompt_keys():
    """Check that k1 default is the v20 file and its identity has only the prompt keys."""
    select_composition("v1")
    assert sha(cp.composed_system_prompt(1)) == V20_SHA256
    assert cp.composed_prompt_identity(1) == {"prompt_version": "v2.0", "prompt_sha256": V20_SHA256}


def test_k1_with_the_fac_variant_sends_the_fac_bytes_only():
    """Check that k1 with the fac variant sends the fac bytes only."""
    cp.set_prompt_variant(FAC)
    select_composition("v1")
    composed = cp.composed_system_prompt(1)
    assert (composed[0] if isinstance(composed, tuple) else composed) == FAC.read_text()
    assert cp.composed_prompt_identity(1) == {"prompt_version": "v2.9.FA-C", "prompt_sha256": FAC_SHA256}


def test_k3_composition_v2_matches_every_u3_record():
    """Check that k3 composition v2 matches every u3 record."""
    cp.set_prompt_variant(FAC)
    select_composition("v2")
    assert sha(cp.composed_system_prompt(3)) == COMPOSED_K3_V2_SHA256
    assert sha(cp.composed_system_prompt(3, **{review_keyword(): True})) == COMPOSED_K3_V2_REVIEW_SHA256
    identity = cp.composed_prompt_identity(3)
    assert identity["composed_prompt_sha256"] == COMPOSED_K3_V2_SHA256
    assert identity["composition_paragraph_sha256"] == COMPOSITION_V2_SHA256
    assert identity["prompt_sha256"] == FAC_SHA256 and identity["proposal_limit"] == 3


def test_k3_composition_v1_when_it_can_still_be_selected():
    """Check that k3 composition v1 when it can still be selected."""
    if not hasattr(cp, "set_composition_paragraph"):
        pytest.skip("composition v1 at k = 3 is no longer selectable (fixed-v2 design)")
    cp.set_prompt_variant(FAC)
    select_composition("v1")
    assert sha(cp.composed_system_prompt(3)) == COMPOSED_K3_V1_SHA256


# ---------------------------------------------------------------- FA-C as a registerable version tag

def test_prompt_version_fac_selects_the_single_repository_copy():
    """Check that prompt version fac selects the single repository copy."""
    assert cp.prompt_version_file("v2.9.FA-C") == FAC.resolve()
    assert cp.REPOSITORY_PROMPT_FILES["v2.9.FA-C"] == ("tools/fb1_launch_c/proposer_system_v2_9_FA-C.txt", FAC_SHA256)


def test_prompt_version_and_prompt_variant_send_the_same_bytes():
    """Check that prompt version and prompt variant send the same bytes."""
    cp.set_prompt_variant(FAC)
    select_composition("v2")
    by_variant = (cp.composed_system_prompt(1), cp.composed_system_prompt(3), cp.composed_prompt_identity(3))
    cp.set_prompt_variant(None)
    assert cp.select_prompt_version("v2.9.FA-C") == FAC.resolve()
    assert (cp.composed_system_prompt(1), cp.composed_system_prompt(3), cp.composed_prompt_identity(3)) == by_variant
    assert cp.composed_prompt_identity(1) == {"prompt_version": "v2.9.FA-C", "prompt_sha256": FAC_SHA256}


def test_prompt_version_fac_refuses_other_bytes(tmp_path, monkeypatch):
    """Check that prompt version fac refuses other bytes."""
    changed = tmp_path / "tools" / "fb1_launch_c" / FAC.name
    changed.parent.mkdir(parents=True)
    changed.write_bytes(FAC.read_bytes() + b" ")
    monkeypatch.setattr(cp, "REPOSITORY_ROOT", tmp_path)
    with pytest.raises(ValueError, match="registered sha256"):
        cp.prompt_version_file("v2.9.FA-C")
    monkeypatch.setattr(cp, "REPOSITORY_ROOT", tmp_path / "absent")
    with pytest.raises(ValueError, match="absent here"):
        cp.prompt_version_file("v2.9.FA-C")


def test_an_unregistered_suffixed_tag_is_refused_and_paths_still_are():
    """Check that an unregistered suffixed tag is refused and paths still are."""
    with pytest.raises(ValueError, match="no packaged prompt file"):
        cp.prompt_version_file("v2.9.FA-K")
    for tag in ("../v2.0", "v2.9/FA-C", "v2.9.FA C", "2.9.FA-C"):
        with pytest.raises(ValueError, match="looks like"):
            cp.prompt_version_file(tag)
