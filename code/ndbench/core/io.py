"""Generic file/hash/provenance helpers shared by `store.py`, suite builders, and vendor tooling.

Kept separate from `types.py` (data shapes) and `keys.py`/`proof.py`/`formula.py` (ND-specific
logic) so every other ndbench module can depend on plain I/O without pulling in ND semantics.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield each non-blank line of a (optionally gzip-compressed, by `.gz` suffix) JSONL file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write `rows` as JSONL (write to a `.tmp` sibling, then rename); gzip-compresses
    if `path` ends in `.gz`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(tmp, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp.replace(path)


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Append `rows` as JSONL lines, creating the file (and parent directories) if needed.

    Not gzip-aware (gzip streams cannot be safely appended to in general); use `write_jsonl` to a
    `.gz` path and rewrite the whole file instead.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_str(s: str) -> str:
    return sha256_bytes(s.encode())


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a `Z` suffix, e.g. `2026-09-30T12:00:00Z`."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit_sha(cwd: str | Path | None = None) -> str:
    """The current commit hash of the repository containing `cwd` (default: this file's repo).

    Returns `"unknown"` (never raises) if `git` is unavailable or `cwd` is not inside a repo, so
    callers (e.g. `SuiteManifest.git_commit`) always get a string.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"
