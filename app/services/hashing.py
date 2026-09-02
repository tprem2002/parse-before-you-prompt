"""Cryptographic hashing helpers for immutable uploaded content."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest of exact input bytes."""

    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it entirely into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(block_size):
            digest.update(block)
    return digest.hexdigest()
