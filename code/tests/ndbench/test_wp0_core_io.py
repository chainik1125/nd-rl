"""`core.io`: JSONL read/write/append (plain and gzip), hashing, and timestamp formatting."""

from __future__ import annotations

import re
from pathlib import Path

from code.ndbench.core.io import (
    append_jsonl,
    git_commit_sha,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    sha256_str,
    utc_now_iso,
    write_jsonl,
)


def test_write_read_jsonl_plain(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [{"a": 1}, {"a": 2}]
    write_jsonl(path, rows)
    assert list(read_jsonl(path)) == rows


def test_write_read_jsonl_gz(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl.gz"
    rows = [{"a": 1}, {"a": 2}]
    write_jsonl(path, rows)
    assert list(read_jsonl(path)) == rows


def test_write_jsonl_is_atomic_no_partial_file_left(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"a": 1}])
    assert not (tmp_path / "rows.jsonl.tmp").exists()


def test_append_jsonl_creates_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "rows.jsonl"
    append_jsonl(path, [{"a": 1}])
    append_jsonl(path, [{"a": 2}, {"a": 3}])
    assert list(read_jsonl(path)) == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_sha256_helpers(tmp_path: Path) -> None:
    assert sha256_bytes(b"hello") == sha256_str("hello")
    path = tmp_path / "f.txt"
    path.write_text("hello")
    assert sha256_file(path) == sha256_str("hello")


def test_utc_now_iso_format() -> None:
    s = utc_now_iso()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", s)


def test_git_commit_sha_in_this_repo() -> None:
    sha = git_commit_sha()
    assert sha != "unknown"
    assert re.match(r"^[0-9a-f]{40}$", sha)


def test_git_commit_sha_outside_a_repo_returns_unknown(tmp_path: Path) -> None:
    assert git_commit_sha(tmp_path) == "unknown"
