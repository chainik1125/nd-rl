"""Vendor integrity (plan §3 WP0 row: "vendor SHA256 check") and the loader shim (plan §1:
"Loading happens via the sys.path shim or importlib unique names")."""

from __future__ import annotations

import hashlib

from code.ndbench.vendor import (
    ABSGPT_DIR,
    LEAN_SEQ_DIR,
    TAKEHOME_DIR,
    VENDOR_DIR,
    ensure_takehome_on_path,
    load_absgpt_module,
    load_nd_verify,
    load_ndcheck,
)


def test_sha256sums_match_every_vendored_file() -> None:
    sums_path = VENDOR_DIR / "SHA256SUMS"
    lines = [ln for ln in sums_path.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 17  # every file listed in the WP0 task's vendor list
    checked = 0
    for line in lines:
        expected_hash, rel_path = line.split(None, 1)
        path = VENDOR_DIR / rel_path
        assert path.is_file(), f"SHA256SUMS lists {rel_path} but it does not exist"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected_hash, f"{rel_path} hash mismatch"
        checked += 1
    assert checked == len(lines)


def test_sha256sums_covers_every_file_on_disk() -> None:
    """No vendored file is missing from SHA256SUMS (catches a future vendoring that forgets to
    regenerate it)."""
    sums_path = VENDOR_DIR / "SHA256SUMS"
    listed = {line.split(None, 1)[1] for line in sums_path.read_text().splitlines() if line.strip()}
    # Excluded: SHA256SUMS itself, README.md, and the top-level __init__.py (ndbench's own loader
    # shim, not vendored content) -- but *not* nested __init__.py files like nd_verify/__init__.py,
    # which are genuinely vendored and are listed.
    top_level_exclude = {"SHA256SUMS", "README.md", "__init__.py"}
    on_disk = {
        str(p.relative_to(VENDOR_DIR))
        for p in VENDOR_DIR.rglob("*")
        if p.is_file()
        and str(p.relative_to(VENDOR_DIR)) not in top_level_exclude
        and "__pycache__" not in p.parts
    }
    assert on_disk == listed


def test_takehome_scripts_import_via_the_path_shim() -> None:
    ensure_takehome_on_path()
    import gen  # noqa: F401
    import intuit  # noqa: F401
    import minlen  # noqa: F401
    import necessity  # noqa: F401
    import patterns  # noqa: F401
    import patterns2  # noqa: F401
    import textbook_pool  # noqa: F401

    assert callable(gen.fstr)
    assert callable(intuit.intuit_provable)
    assert callable(minlen.minlen)


def test_load_nd_verify_and_ndcheck() -> None:
    nd_verify = load_nd_verify()
    ok, reason, n = nd_verify.verify_text("THM P SEQ P PRF N1 P : PR ; QED")
    assert (ok, reason, n) == (True, "ok", 1)

    ndcheck = load_ndcheck()
    d = ndcheck.classify("THM P SEQ P PRF", "N1 P : PR ; QED")
    assert d["ok"] is True
    assert d["kind"] == "ok"


def test_load_absgpt_module_is_idempotent_and_namespaced() -> None:
    import sys

    keys_mod = load_absgpt_module("keys")
    keys_mod_again = load_absgpt_module("keys")
    assert keys_mod is keys_mod_again
    assert keys_mod.canon_key("P , Q |- P") == "P , Q |- P"
    assert "ndbench_absgpt_keys" in sys.modules
    # Note: unlike `model`/`tokenizer` (see the next test), `keys` is not a safe name to assert
    # absent from `sys.modules` globally -- `vendor/common.py`'s own top-level `from keys import
    # canon_key` (against the live ab3/frozen/lib copy, see vendor/README.md) legitimately
    # populates bare `sys.modules["keys"]` as an unrelated, documented side effect whenever some
    # other test has imported `common.py` first; that is independent of this loader.


def test_load_absgpt_module_transient_bare_dep_for_sample_py() -> None:
    """`sample.py`'s top-level `from model import rope_cache` resolves via a transient bare
    alias, which is cleaned up afterwards."""
    import sys

    load_absgpt_module("model")
    sample_mod = load_absgpt_module("sample", bare_deps=("model",))
    assert callable(sample_mod.generate)
    assert "model" not in sys.modules  # transient alias removed after loading sample.py


def test_load_absgpt_module_missing_dependency_raises() -> None:
    import sys

    import pytest

    sys.modules.pop("ndbench_absgpt_tokenizer", None)  # ensure a fresh (uncached) load
    with pytest.raises(RuntimeError):
        load_absgpt_module("tokenizer", bare_deps=("does_not_exist",))


def test_vendor_dirs_exist() -> None:
    for d in (VENDOR_DIR, TAKEHOME_DIR, ABSGPT_DIR, LEAN_SEQ_DIR):
        assert d.is_dir()
    assert (VENDOR_DIR / "nd_verify" / "verify.py").is_file()
    assert (LEAN_SEQ_DIR / "lean_seq.py").is_file()
