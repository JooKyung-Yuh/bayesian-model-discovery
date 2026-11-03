"""The prompt files of the paper runs in `tools/paper_prompts/` stay byte for byte.

Each file's sha256 is the digest the runs recorded (`prompt_sha256`, `composed_prompt_sha256`,
`composition_paragraph_sha256`, `archive_review_paragraph_sha256` and the ForceBench
`description_sha256`). The check reads the files with the standard library only.
"""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "tools" / "paper_prompts"
RECORDED = {
    "proposer_system_v2_9_FA-C-R.txt": "e465265a5862aeaa91b1dbda102f04390b059f542fb07d9705cea218f246dc4b",
    "proposer_system_v2_9_FA-C-B.txt": "0a922682bea0d9a0bc95eac00c232ae49797d6454407049fdd77bab6e28104a8",
    "proposer_system_v2_9_FA-C-D.txt": "37ab98c79944bbd231f16d16a25ceec18abaaaa759e45e0d1fc52f3516fdc0e9",
    "proposer_system_v2_9_FA-C-NF.txt": "c53ba372d293bc8261ecee6fc4cbe2cb6475751cb7f123d1d1fe22d168639a80",
    "proposer_system_v2_9_FA-C-R_trig.txt": "445949c6d2c8a34abfdbe78a5ac7da8e1adbeb1dfde859d60b4ee71cf2580902",
    "proposer_system_v2_9_FA-C-B_trig.txt": "a846c5b4d1663c2077a111bdb3e72728179932abfdd660b59f011c477c85691f",
    "proposer_system_v2_9_FA-C-D_trig.txt": "b647a6791c0152703a1d369dbb59efa3f54170153c4512d2d8a7417024c7a631",
    "proposer_system_v2_9_FA-C-NF_trig.txt": "098ad442316fdf57418a180098d1476b558e800a74641091fdec73bf6e16168e",
    "composition_paragraph_v3_neutral.txt": "8f41727fe082d32bdfb024cb15407234f2268062294b2153635200b0f23a538a",
    "archive_review_paragraph_v2_neutral.txt": "1711c11a907022baaf6df2e987be3446cc282b6f478be4c3ce7c18cf8150fc67",
    "FB_FULL_DESCRIPTION.md": "d666077054a957976f87367ede5279771becd6d20ef58772b7b4d1acaf38032a",
}


def _sums():
    out = {}
    for line in (FOLDER / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        out[name.lstrip("*")] = digest
    return out


@pytest.mark.parametrize("name", sorted(RECORDED))
def test_each_prompt_file_has_its_recorded_digest(name):
    digest = hashlib.sha256((FOLDER / name).read_bytes()).hexdigest()
    assert digest == RECORDED[name], f"{name}: {digest}"
    assert _sums()[name] == digest


def test_the_sums_file_lists_exactly_the_prompt_files():
    listed = set(_sums())
    present = {path.name for path in FOLDER.iterdir() if path.name not in ("SHA256SUMS", "README.md")}
    assert listed == present == set(RECORDED)


def test_the_version_tag_is_the_first_line_of_each_system_prompt():
    for name in RECORDED:
        if name.startswith("proposer_system_"):
            tag = (FOLDER / name).read_text(encoding="utf-8").splitlines()[0]
            expected = "v2.9." + name[len("proposer_system_v2_9_"):].removesuffix(".txt").removesuffix("_trig")
            assert tag == f"# prompt_version: {expected}", (name, tag)
