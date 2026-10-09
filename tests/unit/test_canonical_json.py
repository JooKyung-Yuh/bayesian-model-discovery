"""Canonical JSON is the only thing identity hashes ever see, so pin its exact behaviour."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

import bayes_model_discovery
from bayes_model_discovery.provenance.canonical import (
    HEX64_RE,
    canonical_json,
    is_hex64,
    sha256_file,
    sha256_text,
)


def test_key_order_does_not_change_the_text_or_the_hash():
    """Check that key order does not change the text or the hash."""
    a = {"beta": 1, "alpha": {"z": 3, "a": 4}, "gamma": [1, 2]}
    b = {"gamma": [1, 2], "alpha": {"a": 4, "z": 3}, "beta": 1}
    assert canonical_json(a) == canonical_json(b)
    assert canonical_json(a) == '{"alpha":{"a":4,"z":3},"beta":1,"gamma":[1,2]}'
    assert sha256_text(canonical_json(a)) == sha256_text(canonical_json(b))


def test_list_order_does_change_the_text():
    # Sorting applies to mapping keys only: a sequence is data, and its order is meaning.
    """Check that list order does change the text."""
    assert canonical_json([1, 2]) != canonical_json([2, 1])


def test_no_insignificant_whitespace():
    """Check that no insignificant whitespace."""
    assert canonical_json({"a": 1, "b": [1, 2]}) == '{"a":1,"b":[1,2]}'


def test_non_ascii_is_kept_as_text():
    """Check that non ascii is kept as text."""
    assert canonical_json({"note": "receipt 6fd68285…"}) == '{"note":"receipt 6fd68285…"}'


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -math.inf])
def test_nan_and_infinity_are_rejected(bad):
    """Check that nan and infinity are rejected."""
    with pytest.raises(ValueError):
        canonical_json({"x": bad})


def test_nested_nan_is_rejected():
    """Check that nested nan is rejected."""
    with pytest.raises(ValueError):
        canonical_json({"a": [1.0, {"b": float("nan")}]})


def test_non_json_types_are_rejected():
    """Check that non json types are rejected."""
    class Opaque:
        pass

    with pytest.raises(TypeError):
        canonical_json({"x": Opaque()})
    with pytest.raises(TypeError):
        canonical_json({"x": {1, 2, 3}})


def test_float_text_is_stable_and_shortest_round_trip():
    """Check that float text is stable and shortest round trip."""
    assert canonical_json({"x": 0.1 + 0.2}) == '{"x":0.30000000000000004}'
    assert canonical_json({"x": 1.0}) == '{"x":1.0}'
    assert canonical_json({"x": 1}) == '{"x":1}'
    # int and float spellings of the same value are different canonical text on purpose:
    # a spec that says 1 and a spec that says 1.0 are not the same document.
    assert canonical_json({"x": 1}) != canonical_json({"x": 1.0})
    assert canonical_json({"x": 1e-09}) == '{"x":1e-09}'
    assert canonical_json({"x": 0.999999999}) == '{"x":0.999999999}'


def test_sha256_text_is_the_known_utf8_digest():
    """Check that sha256 text is the known utf8 digest."""
    assert sha256_text("") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert sha256_text("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_file_matches_sha256_of_the_same_bytes(tmp_path):
    """Check that sha256 file matches sha256 of the same bytes."""
    p = tmp_path / "asset.txt"
    p.write_text("abc", encoding="utf-8")
    assert sha256_file(p) == sha256_text("abc")


# --------------------------------------------------------------------------------------
# What a digest looks like: defined once, used everywhere
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "good", [sha256_text(""), sha256_text("abc"), "0" * 64, "f" * 64, "0123456789abcdef" * 4]
)
def test_is_hex64_accepts_the_digests_this_repository_computes(good):
    """Check that is hex64 accepts the digests this repository computes."""
    assert is_hex64(good) is True


@pytest.mark.parametrize(
    "bad",
    [
        "hello",
        sha256_text("abc").upper(),  # uppercase is a different convention, not a variant
        "a" * 63,
        "a" * 65,
        "g" * 64,
        " " + "a" * 64,
        "a" * 64 + "\n",  # a trailing newline from a `sha256sum` paste
        "",
        None,
        b"a" * 64,
        123,
    ],
)
def test_is_hex64_rejects_everything_else(bad):
    """Check that is hex64 rejects everything else."""
    assert is_hex64(bad) is False


def test_the_digest_pattern_is_defined_in_exactly_one_module():
    """One canonical pattern plus the standalone policy's equivalent validator."""
    from bayes_model_discovery.experimental.run_label_policy import _sha

    package_root = Path(bayes_model_discovery.__file__).parent
    definitions = sorted(
        p.relative_to(package_root).as_posix()
        for p in package_root.rglob("*.py")
        if "[0-9a-f]{64}" in p.read_text(encoding="utf-8")
    )
    assert definitions == ["experimental/run_label_policy.py", "provenance/canonical.py"]
    # The portable ForceBench policy stays standard-library-only; validate agreement.
    values = [sha256_text("abc"), "a" * 64, "", "a" * 63, "a" * 65, "g" * 64,
              sha256_text("abc").upper(), "a" * 64 + "\n", None, b"a" * 64, 123]
    for value in values:
        if is_hex64(value):
            assert _sha(value) == value
        else:
            with pytest.raises(ValueError):
                _sha(value)
    # `\Z`, not `$`: `$` also matches before a trailing newline.
    assert HEX64_RE.pattern == r"\A[0-9a-f]{64}\Z"
