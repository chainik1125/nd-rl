"""Vendors Dan Pandori's `nd-takehome` modules that `code.thinking` needs into `vendor/danstate/`.

Each file is copied byte for byte with `git -C <clone> show <commit>:<path>` (the clone is only
read), pinned at commit 4dcd0e4f (branch `dan`, guided-tts DONE), and recorded in `SHA256SUMS` and
`PROVENANCE.md`.  The git helpers are `code/ndbench/tools/vendor_danstate.py`'s; ndbench's own copy
(pinned at 0c492902) is not touched.

The file set is the closure of the module-level imports of the modules the pipeline calls (an AST
scan, plus the lazy imports of the functions it calls: `model.save_ckpt` -> record, `lean_gate.Gate`
-> lean_prefilter / lean_judge).  `nd_verify/` is included so the copy is self-contained; its bytes
equal ndbench's and r1_cot's copies.

    code/.venv/bin/python -m code.thinking.vendor_danstate            # (re)vendor
    code/.venv/bin/python -m code.thinking.vendor_danstate --check   # compare with the clone
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

from code.ndbench.tools.vendor_danstate import REPO_URL, git_commit_date, git_rev_parse, git_show

DEFAULT_CLONE = Path("/scratch/cowifornia/nd-rl/think/nd-takehome")
DEFAULT_COMMIT = "4dcd0e4f"
DEST_DIR = Path(__file__).resolve().parent / "vendor" / "danstate"

#: (path in the fork, role), in a topological order of the module-level import graph (`dan.ORDER`).
FILES: list[tuple[str, str]] = [
    ("nd_verify/__init__.py", "The take-home's ND verifier package (re-exports `verify`)."),
    (
        "nd_verify/verify.py",
        "ND proof parser / checker; `parse_formula` is used by the Lean tokenizers.",
    ),
    ("nd2lean.py", "ND proof <-> Lean 4 text (`parse_prompt`, `lf`)."),
    (
        "lean_tok.py",
        "`LeanTokenizer` (`lean_staten`: 111 tokens, ref0 47) and the strict grammar `inverse`.",
    ),
    ("model.py", "Base GPT, `save_ckpt` (uploads unless ND_OFFLINE=1) and `load_ckpt`."),
    ("tokenizer.py", "`make_tokenizer` dispatch (imported by state_train)."),
    (
        "best_model.py",
        "Robbie's 6 x 384 Peri-LN / ALiBi network `ALiBiGPT`, Muon, MTP head and loss.",
    ),
    ("record.py", "Results registry and compute counters (ND_REGISTRY_DIR, ND_OFFLINE)."),
    ("sample.py", "Batched KV-cache decoding; `_gumbel_pick` (row-keyed Gumbel-max noise)."),
    ("state_env.py", "The proof-state environment `Env` and `decompose`."),
    ("state_sample.py", "`env_generate`, the batched worklist sampler in the environment."),
    (
        "state_train.py",
        "State trainer: `load`, `batch`, `loss_on` and the plain AdamW fine-tune loop.",
    ),
    (
        "state_train_best.py",
        "`train`: Dan's best recipe (Muon + AdamW, MTP 0.3, token-budget batches).",
    ),
    ("lean_prefilter.py", "Sound reject-only Lean pre-filter (used by step_check and lean_gate)."),
    ("lean_judge.py", "In-process Lean verdict registry (imported lazily by lean_gate)."),
    (
        "lean_gate.py",
        "Lean 4 checking of finished proofs (`gate`, `lean_check`); binary path `LEAN`.",
    ),
    ("step_check.py", "Per-step check sound w.r.t. Lean (guided-tts 'logical' arm)."),
    (
        "guided_sample.py",
        "Plain / structural / logical redraws without replacement (`_swor_shift`).",
    ),
    ("guided_eval.py", "guided-tts read-out driver (`load_sets`)."),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_provenance(full_sha: str, commit: str, commit_date: str, clone: Path) -> None:
    rows = []
    for rel, role in FILES:
        digest = sha256((DEST_DIR / rel).read_bytes())
        rows.append(f"| `{rel}` | `{digest[:16]}...` | {role} |")
    body = f"""# Provenance: `code/thinking/vendor/danstate/`

- Source: {REPO_URL}, branch `dan`
- Pinned commit: `{full_sha}` (short `{commit}`), committed {commit_date}
- Vendored on {datetime.now(UTC).strftime("%Y-%m-%d")} by `code/thinking/vendor_danstate.py`
  (`git -C {clone} show {commit}:<path>` per file; the clone is never written to)
- Exact bytes: `SHA256SUMS` in this directory (`code/tests/thinking/test_vendor.py` checks them).
- **Unmodified.** Every change for the thinking pipeline lives in `code/thinking/*.py`:
  wrappers, a `LeanTokenizer` subclass (`tok.ThinkTokenizer`), and these copies, each commented
  with its upstream `file:line@{commit}`: `state_train_best.train` (`train.train_best`),
  `model.load_ckpt` (`ckpt.load_ckpt`), the plain fine-tune loop of `state_train.main`
  (`train.finetune`) and the decode loops of `sample.generate_ids_fast` /
  `guided_sample.decode` (`sample.decode_wave`).
- Loaded only through `code/thinking/dan.py`, under private module names, so the generic names
  (`model`, `record`, `sample`, ...) never collide with other vendored trees in this repository.

## Files

| file | sha256 (prefix) | role |
| --- | --- | --- |
{chr(10).join(rows)}

## Licence / credit

Dan Pandori's code (fork `nd-takehome`), except `best_model.py`, which ports Robbie's
`pretrain_best.py` (nd-rl `robbie-experiments`, factorial_20260929) and embeds a Muon routine
adapted from KellerJordan/Muon (MIT licence, text preserved in the file), and
`guided_sample._swor_shift`, which is Robbie's (UniqueRandomizer, Shi et al. 2020).  `nd_verify/`
is the take-home's verifier.
"""
    (DEST_DIR / "PROVENANCE.md").write_text(body)


def vendor(clone: Path, commit: str) -> None:
    full_sha = git_rev_parse(clone, commit)
    sums = []
    for rel, _ in FILES:
        data = git_show(clone, commit, rel)
        out = DEST_DIR / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        sums.append(f"{sha256(data)}  {rel}\n")
    (DEST_DIR / "SHA256SUMS").write_text("".join(sums))
    write_provenance(full_sha, commit, git_commit_date(clone, full_sha), clone)
    print(f"vendored {len(FILES)} files from {REPO_URL}@{commit} into {DEST_DIR}")


def check(clone: Path, commit: str) -> list[str]:
    """Paths whose vendored bytes differ from the clone at `commit` (empty: identical)."""
    return [
        rel for rel, _ in FILES if (DEST_DIR / rel).read_bytes() != git_show(clone, commit, rel)
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--clone", type=Path, default=DEFAULT_CLONE, help="Dan's fork clone (read only)"
    )
    ap.add_argument("--commit", default=DEFAULT_COMMIT)
    ap.add_argument("--check", action="store_true", help="compare only; write nothing")
    a = ap.parse_args(argv)
    if not a.clone.is_dir():
        print(f"error: no clone at {a.clone}", file=sys.stderr)
        return 1
    if a.check:
        bad = check(a.clone, a.commit)
        print("identical" if not bad else f"differ: {bad}")
        return 1 if bad else 0
    vendor(a.clone, a.commit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
