"""Shared read-only helpers for the M1 (`temp`) tools: Dan's stored T 1.0 draws and graded rows of our own cells.

Dan's stored rows are read through `dan_curve` (the 2026-11-25 folder; imported, never edited). Draw order everywhere is
(chunk, i); "first n draws" means the first n rows in that order (with `--chunk 1`, chunk index = draw index).
"""

from __future__ import annotations

import gzip
import json
import sys

import numpy as np

OLD = "/scratch/cowifornia/nd-rl/worktrees/olmo-elicitation/code/experiments/current/olmo_elicitation_20261125"
if OLD not in sys.path:
    sys.path.insert(0, OLD)
import dan_curve as dc  # noqa: E402

DAN = dc.DAN
ART = DAN / "artifacts"
CAP = 32768
BOOT_REPS = 10000
BOOT_SEED = 0


def jl(path, tolerate_tail=True) -> list[dict]:
    p = str(path)
    op = gzip.open if p.endswith(".gz") else open
    out: list[dict] = []
    try:
        with op(p, "rt") as f:
            for ln in f:
                if not ln.strip():
                    continue
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    if tolerate_tail:
                        break
                    raise
    except (EOFError, OSError):
        pass
    return out


def order(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (int(r["chunk"]), int(r["i"])))


def by_pid(rows: list[dict]) -> dict[str, list[dict]]:
    d: dict[str, list[dict]] = {}
    for r in order(rows):
        d.setdefault(r["pid"], []).append(r)
    return d


def is_cut(r: dict) -> bool:
    return r.get("finish") == "length" or int(r.get("n_tok", 0)) >= CAP


def stored_arm_a(ckpt: str) -> dict[str, list[dict]]:
    """pid -> Dan's stored T 1.0 arm-A draws (16 at dpo, 8 elsewhere) as rows with strict / n_tok / finish."""
    R = dc.Rows(ART / "olmo_ifcurve" / "rows")
    return {p: R.draws(ckpt, p, "A") for p in dc.arm_a()}


def stored_reach(ckpt: str) -> dict[str, list[dict]]:
    """pid -> stored reach draws: 32 per prompt for s0100 / s0500 / s2300; for dpo the first 16 of olmo_ifomega
    F_dpo_ifb + F_dpo_ife (graded rows)."""
    reach = set(dc.reach())
    if ckpt != "dpo":
        R = dc.Rows(ART / "olmo_ifcurve" / "rows")
        return {p: R.draws(ckpt, p, "B") for p in dc.reach()}
    out: dict[str, list[dict]] = {}
    for j in ("F_dpo_ifb", "F_dpo_ife"):
        for r in order(jl(ART / f"olmo_ifomega/graded/{j}.jsonl")):
            if r["pid"] in reach:
                out.setdefault(r["pid"], []).append(r)
    return {p: v[:16] for p, v in out.items()}


def strict_of(r: dict) -> int:
    return int(bool(r["strict"]))


def boot_idx(n: int, reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(reps, n))


def ci(samples: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))
