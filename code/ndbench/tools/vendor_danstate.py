#!/usr/bin/env python3
"""Vendors Dan Pandori's `nd-takehome` state-env modules into `code/ndbench/vendor/danstate/`.

Plan: `code/experiments/current/benchmarking_20260930/DANSTATE_PLAN.md` §A4, §1, §7 WP-D0. Copies
each file byte-identically via `git -C <clone> show <commit>:<path>` from Dan's fork clone,
pinned at commit `0c492902` -- never opens the clone for writing (`git show` only reads it).

**File list and why it is exactly this set** (verified with an AST scan of module-level
`Import`/`ImportFrom` nodes over the whole clone at this commit, not just a grep for the word
"import"): the plan names ten files (`best_model.py` through `lean_gate.py` below). Checking what
each one imports *at module level* (lazy, function-body imports do not count -- they are a
separate, documented problem; see `vendor/danstate_loader.py`'s module docstring) gives a closed
dependency graph over exactly these ten files plus `nd_verify`, which is NOT vendored here:
`code/ndbench/vendor/nd_verify/` is already byte-identical to the fork's copy (sha256
`5dc502c2fc0a9c2f5b535a4044e6e5af00d1527c18a9c56cadece2904f62e614` for `verify.py`; `vendor()`
below re-verifies this against the clone before writing anything). In particular,
`lean_prefilter.py` and `lean_judge.py` are imported by `lean_gate.py` only *lazily*, inside its
`gate()` function body -- never at module level by any of the ten files -- so they are not needed
for any of the ten to import, and are deliberately not vendored by this work package.

Usage (run once; re-run only to re-vendor, e.g. at a new pinned commit, then recommit):

    code/.venv/bin/python code/ndbench/tools/vendor_danstate.py
    code/.venv/bin/python code/ndbench/tools/vendor_danstate.py --clone PATH --commit SHA
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_URL = "https://github.com/dan-pandori/nd-takehome"
DEFAULT_CLONE = Path("/scratch/cowifornia/nd-rl/bench/nd-takehome-dan")
DEFAULT_COMMIT = "0c492902"

DEST_DIR = Path(__file__).resolve().parents[1] / "vendor" / "danstate"

#: The ten files named by the plan (§1, §7 WP-D0), in the order `vendor/danstate_loader.py` loads
#: them (a topological order of their module-level import graph -- see its own `_MODULE_ORDER`).
#: `role` is a one-line note for `PROVENANCE.md`; it does not affect vendoring.
FILES: list[tuple[str, str]] = [
    ("nd2lean.py", "ND proof <-> Lean 4 tactic text translator (`translate`, `parse_prompt`)."),
    (
        "lean_tok.py",
        "The lean_seq/state tokenizer (`LeanTokenizer`) and its strict grammar parser (`inverse`).",
    ),
    (
        "model.py",
        "The fork's base decoder-only GPT (`GPT`) and RoPE cache; `load_ckpt`/`save_ckpt`.",
    ),
    (
        "tokenizer.py",
        "The abs/rel line-reference tokenizer (`Tokenizer`) and `make_tokenizer` dispatch.",
    ),
    (
        "best_model.py",
        "Robbie's Peri-LN/ALiBi network (`ALiBiGPT`) ported for the state format; Muon update.",
    ),
    (
        "record.py",
        "Results-registry / checkpoint-upload helpers (`record`); side-effect-free to import.",
    ),
    (
        "sample.py",
        "Batched KV-cache sampling (`generate`, `generate_ids_fast`) for the base tokenizer.",
    ),
    (
        "state_env.py",
        "The step environment (`Env`, `decompose`, `canonicalise`) for the lean_seq fragment.",
    ),
    (
        "state_sample.py",
        "Batched in-environment rollout (`env_generate`) built on `state_env`/`sample`.",
    ),
    (
        "lean_gate.py",
        "Whole-proof Lean checking (`gate`); vendored for completeness, not callable as-is.",
    ),
]

CREDIT = """\
## Licence / credit

All of the code above is Dan Pandori's (the `nd-takehome` fork, run `best-state` et al.), with one
explicit exception: `best_model.py` is a port, not original to this fork. Its own module docstring
says so directly: "Copied from nd-rl `robbie-experiments`,
`code/experiments/current/factorial_20260929/pretrain_best.py` (the autoresearch incumbent
186-claude, sha f8451342; Robbie's factorial of 2026-09-29), which is built on this fork's `GPT`."
`best_model.py` also embeds a small Muon-update routine (`muon_step`) adapted from
KellerJordan/Muon, commit `f98f1cacc0263b04290753e32be8d498c1efc806` (`muon.py`); MIT License,
Copyright (c) 2024 Keller Jordan. The licence text is preserved verbatim in the vendored file,
immediately above `muon_step`.
"""

NOT_VENDORED = """\
## Deliberately not vendored

- `nd_verify/` -- reused from `code/ndbench/vendor/nd_verify/` (sha256 of `verify.py`:
  `5dc502c2fc0a9c2f5b535a4044e6e5af00d1527c18a9c56cadece2904f62e614`), which this script verifies
  is still byte-identical to the fork's own copy at the pinned commit (see the check this script
  runs before writing anything, and `test_dan_vendor.py`'s own independent check).
- `lean_prefilter.py`, `lean_judge.py` -- imported by `lean_gate.py` only lazily, inside its
  `gate()` function body; not imported at module level by any of the ten files above, so not
  needed for any of them to import cleanly. Vendoring them (if a later work package needs Lean
  gating/judging, e.g. plan §7 WP-D2) is a separate change.
"""

LOADER_NOTE = """\
## Loading / forbidden calls

See `code/ndbench/vendor/danstate_loader.py`'s module docstring for the namespaced loader
(`load_danstate()`) and the full list of lazy, function-body-only bare imports these files make
(`model.load_ckpt`, `tokenizer.make_tokenizer`, `sample.generate`'s gated branch,
`state_sample.env_generate(gate=True)`) that are forbidden to call directly -- they do their own
bare `import tokenizer`/`from best_model import ALiBiGPT`/etc. at call time, which the namespaced
loader does not leave resolvable by default (`DanState.lazy_imports()` is the documented escape
hatch, with its own limits).
"""


def git_show(clone: Path, commit: str, rel_path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(clone), "show", f"{commit}:{rel_path}"],
        check=True,
        capture_output=True,
    )
    return result.stdout


def git_rev_parse(clone: Path, commit: str) -> str:
    return subprocess.run(
        ["git", "-C", str(clone), "rev-parse", commit],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_commit_date(clone: Path, full_sha: str) -> str:
    return subprocess.run(
        ["git", "-C", str(clone), "show", "-s", "--format=%cI", full_sha],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def check_nd_verify_unchanged(clone: Path, commit: str) -> str:
    """Re-verify the plan's claim (§0: "Dan's `nd_verify` is byte-identical to our vendored
    copy") against the pinned commit; returns the matching sha256. Raises if it no longer holds
    -- that would mean `nd_verify` needs to be re-vendored too, not silently skipped."""
    vendor_nd_verify = DEST_DIR.parent / "nd_verify" / "verify.py"
    ours = hashlib.sha256(vendor_nd_verify.read_bytes()).hexdigest()
    theirs = hashlib.sha256(git_show(clone, commit, "nd_verify/verify.py")).hexdigest()
    if ours != theirs:
        raise RuntimeError(
            f"nd_verify/verify.py has diverged from the fork at {commit}: ours {ours}, "
            f"theirs {theirs}. Re-vendor nd_verify (plan §0) before vendoring danstate."
        )
    return ours


def write_provenance(
    dest_dir: Path,
    clone: Path,
    commit: str,
    full_sha: str,
    commit_date: str,
    nd_verify_sha: str,
) -> None:
    rows = []
    for rel_path, role in FILES:
        digest = hashlib.sha256((dest_dir / rel_path).read_bytes()).hexdigest()
        rows.append(f"| `{rel_path}` | `{digest[:16]}...` | {role} |")

    vendored_on = datetime.now(UTC).strftime("%Y-%m-%d")
    body = f"""# Provenance: `code/ndbench/vendor/danstate/`

- Source: {REPO_URL}
- Pinned commit: `{full_sha}` (short `{commit}`), authored {commit_date}
- Vendored on: {vendored_on}, by `code/ndbench/tools/vendor_danstate.py`
  (`git -C {clone} show {commit}:<path>` per file; the clone itself is never written to)
- See `SHA256SUMS` (this directory) for the exact bytes; regenerate both files together by
  re-running the tool above after changing `--commit`.

## Files

| file | sha256 (prefix) | role |
| --- | --- | --- |
{chr(10).join(rows)}

{NOT_VENDORED}
nd_verify/verify.py sha256 at this commit: `{nd_verify_sha}` (matches the vendored copy).

{CREDIT}
{LOADER_NOTE}
"""
    (dest_dir / "PROVENANCE.md").write_text(body)


def vendor(clone: Path, commit: str) -> None:
    full_sha = git_rev_parse(clone, commit)
    commit_date = git_commit_date(clone, full_sha)
    nd_verify_sha = check_nd_verify_unchanged(clone, commit)

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for rel_path, _role in FILES:
        data = git_show(clone, commit, rel_path)
        (DEST_DIR / rel_path).write_bytes(data)
        digests[rel_path] = hashlib.sha256(data).hexdigest()

    sums_path = DEST_DIR / "SHA256SUMS"
    with sums_path.open("w") as fh:
        for rel_path, _role in FILES:
            fh.write(f"{digests[rel_path]}  {rel_path}\n")

    write_provenance(DEST_DIR, clone, commit, full_sha, commit_date, nd_verify_sha)

    print(f"Vendored {len(FILES)} files from {REPO_URL}@{commit} ({full_sha}) into {DEST_DIR}")
    for rel_path, _role in FILES:
        print(f"  {digests[rel_path][:16]}...  {rel_path}")
    print(f"nd_verify/verify.py unchanged (sha256 {nd_verify_sha[:16]}...), not vendored.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--clone", type=Path, default=DEFAULT_CLONE, help="Dan's clone (read-only)")
    ap.add_argument("--commit", default=DEFAULT_COMMIT, help="pinned commit (short or full SHA)")
    args = ap.parse_args(argv)

    if not args.clone.is_dir():
        print(f"error: clone not found at {args.clone}", file=sys.stderr)
        return 1

    vendor(args.clone, args.commit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
