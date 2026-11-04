"""Every SHA256SUMS file in the tree lists exactly the files beside it, with their current digests.

The wrapper folders under paper/baselines and the prompt and simulator folders under tools carry one; a
stale entry (a file edited after the list was written, or a file that is not in the tree) fails here.
"""
import hashlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUMS = sorted(p for p in ROOT.rglob("SHA256SUMS") if ".git" not in p.parts)


@pytest.mark.parametrize("sums", SUMS, ids=[str(p.relative_to(ROOT)) for p in SUMS])
def test_checksum_file_verifies(sums):
    folder = sums.parent
    listed = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        listed[name.lstrip("*")] = digest
    assert listed, f"{sums} is empty"
    missing = [name for name in listed if not (folder / name).is_file()]
    assert not missing, f"{sums} lists files that are not in the tree: {missing[:5]}"
    changed = [name for name, digest in listed.items()
               if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest]
    assert not changed, f"{sums} digests differ for: {changed[:5]}"
