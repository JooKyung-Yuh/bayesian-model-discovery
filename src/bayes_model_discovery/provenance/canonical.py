"""Canonical JSON encoding and SHA-256 helpers.

Every identity hash in this repository is the SHA-256 of a **canonical JSON string**, never
of a source file's raw bytes. Reformatting a YAML spec, reordering its keys, or rewriting a
comment therefore cannot move an identity: only the resolved values can.

Canonical form:

* keys sorted (``sort_keys=True``) - authoring order carries no meaning;
* no insignificant whitespace (``separators=(",", ":")``);
* text stays text (``ensure_ascii=False``), UTF-8 encoded at hashing time;
* no ``NaN`` / ``Infinity`` (``allow_nan=False``) - they have no JSON spelling, and emitting
  the JavaScript literals would produce a hash no conforming reader could reproduce;
* no non-JSON Python objects - :mod:`json` raises ``TypeError`` rather than falling back to
  ``repr``, so an accidentally embedded object is a loud failure, not a silent identity.

Floats are written with :func:`repr`-equivalent shortest round-trip text (CPython's
``float_repr_style`` has been ``short`` since 3.1), so ``0.1 + 0.2`` always serializes as
``0.30000000000000004`` and never as a platform-dependent number of digits.

This module also owns :data:`HEX64_RE`, the **single** definition of what a SHA-256 digest
looks like in this repository. Everything that stores a digest - spec references, receipt
identity slots, manifest entries - validates against this one pattern. Two copies of a
"hash looks like this" regex would eventually disagree, and the day they disagree is the day
one file accepts a digest another file would have refused.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

__all__ = ["HEX64_RE", "canonical_json", "is_hex64", "sha256_file", "sha256_text"]

_CHUNK_BYTES = 1 << 20

#: A SHA-256 digest as this repository writes it everywhere: lowercase hex, exactly 64 chars.
#: Uppercase is refused rather than normalized - the digest that is stored must be the digest
#: that was computed, byte for byte, or comparing two of them means comparing two conventions.
#: Anchored with ``\A``/``\Z`` rather than ``^``/``$`` because ``$`` also matches before a
#: trailing newline, which would quietly admit a digest pasted straight out of ``sha256sum``.
HEX64_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def is_hex64(value: Any) -> bool:
    """Whether ``value`` is a string in this repository's canonical digest form.

    Non-strings are simply ``False``: callers that need to tell "not a hash at all" apart
    from "a string that is not a digest" check the type themselves first.
    """
    return isinstance(value, str) and HEX64_RE.match(value) is not None


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` to the canonical JSON text used by every identity hash.

    Raises:
        TypeError: if ``obj`` contains a value with no JSON representation.
        ValueError: if ``obj`` contains ``NaN``, ``Infinity`` or ``-Infinity``, or if it
            contains a reference cycle.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_text(s: str) -> str:
    """SHA-256 (lowercase hex) of ``s`` encoded as UTF-8."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """SHA-256 (lowercase hex) of the raw bytes of the file at ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
