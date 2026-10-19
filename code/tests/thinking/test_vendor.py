"""The vendored fork copy (`code/thinking/vendor/danstate`, nd-takehome at 4dcd0e4f) and loader."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from code.thinking import dan, vendor_danstate

SUMS = dan.VENDOR / "SHA256SUMS"
REPO = Path(__file__).resolve().parents[3]


def test_sha256sums_match_and_cover_every_file() -> None:
    listed = {}
    for line in SUMS.read_text().splitlines():
        digest, rel = line.split(None, 1)
        listed[rel] = digest
        assert hashlib.sha256((dan.VENDOR / rel).read_bytes()).hexdigest() == digest, rel
    on_disk = {str(p.relative_to(dan.VENDOR)) for p in dan.VENDOR.rglob("*.py")}
    assert on_disk == set(listed) == {rel for rel, _ in vendor_danstate.FILES}


def _clone_has_commit() -> bool:
    if not vendor_danstate.DEFAULT_CLONE.is_dir():
        return False
    r = subprocess.run(
        [
            "git",
            "-C",
            str(vendor_danstate.DEFAULT_CLONE),
            "cat-file",
            "-e",
            vendor_danstate.DEFAULT_COMMIT,
        ],
        capture_output=True,
    )
    return r.returncode == 0


@pytest.mark.skipif(not _clone_has_commit(), reason="Dan's fork clone at 4dcd0e4f not available")
def test_vendored_bytes_equal_the_fork_at_the_pinned_commit() -> None:
    assert (
        vendor_danstate.check(vendor_danstate.DEFAULT_CLONE, vendor_danstate.DEFAULT_COMMIT) == []
    )


FRESH = textwrap.dedent(
    """
    import sys
    before = set(sys.modules)
    from code.thinking import dan
    D = dan.load()
    bare = [n for n in dan.ORDER + ("nd_verify.verify",) if n in sys.modules]
    assert not bare, bare
    assert not any(str(dan.VENDOR) in p for p in sys.path)
    with dan.bare():
        import record
        assert record is D.record
        with dan.bare():
            assert sys.modules["lean_gate"] is D.lean_gate
        assert sys.modules["lean_gate"] is D.lean_gate
    assert "record" not in sys.modules and "lean_gate" not in sys.modules
    T = dan.trace_dsl()
    assert T.Env is D.state_env.Env and T.step_check is D.step_check
    assert not [n for n in ("step_check", "lean_tok", "state_env") if n in sys.modules]
    assert not any(str(dan.VENDOR) in p for p in sys.path)
    from code.ndbench.vendor import load_danstate
    nd = load_danstate()
    assert nd.state_env.Env is not D.state_env.Env
    assert not [n for n in dan.ORDER[1:] if n in sys.modules]  # ndbench leaves a bare nd_verify
    D.record.save_config({"x": 1}, None)  # a process-level compute block, closed at exit
    D.record.count(train_steps=1)
    print("ok")
    """
)


def test_loader_leaves_a_fresh_interpreter_clean() -> None:
    """No bare module names, no sys.path entries, re-entrant bare(), trace_dsl bound to our copy,
    coexistence with ndbench's danstate copy -- checked in a fresh interpreter."""
    env = {k: v for k, v in os.environ.items() if k != "ND_REGISTRY_DIR"}
    env.update(ND_OFFLINE="1", ND_REGISTRY_SYNC="0", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(
        [sys.executable, "-c", FRESH],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]
    assert not list(dan.VENDOR.rglob("__pycache__"))
    assert not (dan.VENDOR / "artifacts").exists()  # the exit-time rows went to REGISTRY_FALLBACK


def test_registry_rows_never_default_into_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    D = dan.load()
    monkeypatch.delenv("ND_REGISTRY_DIR", raising=False)
    monkeypatch.setitem(D.record._state, "file", None)
    path = D.record._file()
    assert path.startswith(dan.REGISTRY_FALLBACK) and not path.startswith(str(REPO))
    assert "ND_REGISTRY_DIR" not in os.environ


def test_bare_restores_prior_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    D = dan.load()
    sentinel = object()
    monkeypatch.setitem(sys.modules, "record", sentinel)  # type: ignore[arg-type]
    with dan.bare():
        assert sys.modules["record"] is D.record
    assert sys.modules["record"] is sentinel


def test_modules_are_namespaced() -> None:
    D = dan.load()
    assert dan.load() is D
    assert D.state_env.Env.__module__ == "thinking_dan_state_env"
    assert D.lean_tok.LeanTokenizer("lean_staten").vocab_size == 111
    T = dan.trace_dsl()
    takehome = T.TAKEHOME
    assert T.Env is D.state_env.Env and takehome == str(dan.VENDOR)
