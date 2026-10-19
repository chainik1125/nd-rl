"""Collapse audit, probe 4b: paired per-token changes between checkpoints on the teacher-forced
reference tokens (`teacher_force.py score` output; CPU, read-only).

Run from the repo root, with M = code.experiments.current.thinking_traces_20261005.collapse_audit:

    code/.venv/bin/python -m $M.tf_ratio

For each pair (A -> B) of checkpoints and each set / condition: r = pi_B / pi_A per scored token;
the share of tokens with r < 0.8 and r > 3 (the GRPO clip-low and neg_r3 thresholds), the mean
log r, the target classes that lost the most log-prob (summed, so frequent classes count), and the
change of P(':') at non-':' targets. Also the share of positions where `:` becomes the argmax
although the target is not `:`.
"""

from __future__ import annotations

import numpy as np

from code.experiments.current.thinking_traces_20261005.collapse_audit.teacher_force import (
    SCR,
    SETS,
    tok_class,
)
from code.thinking.tok import EffortTokenizer

PAIRS = [
    ("s0_it0", "s0_it39"),
    ("s0_it39", "s0_it49"),
    ("s0_it49", "s0_it50"),
    ("s0_it50", "s0_it52"),
    ("s0_it52", "s0_it60"),
    ("s0_it60", "s0_it78"),
    ("s1_it0", "s1_it39"),
    ("s1_it39", "s1_it45"),
    ("s1_it45", "s1_it49"),
    ("s1_it39", "s1_it49"),
    ("s1_it49", "s1_it50"),
    ("s1_it50", "s1_it52"),
    ("s1_it52", "s1_it60"),
    ("s1_it60", "s1_it78"),
]


def main() -> int:
    tok = EffortTokenizer()
    colon = tok.stoi[":"]
    cls = np.array([tok_class(s) for s in tok.itos])
    have = {p.stem[3:] for p in SCR.glob("tf_*.npz")}
    cache: dict[str, dict[str, np.ndarray]] = {}

    def get(t: str) -> dict[str, np.ndarray]:
        if t not in cache:
            cache[t] = dict(np.load(SCR / f"tf_{t}.npz"))
        return cache[t]

    for sname in SETS:
        for cond in ("B0", "B256"):
            key = f"{sname}|{cond}"
            print(f"\n######## {key}")
            print(
                f"{'pair':18s} {'r<0.8':>7s} {'r>3':>7s} {'mean lr':>8s} "
                f"{'dPcol':>8s} {'col-argmax':>11s}  "
                "top losing classes (sum of -log r, % of total loss)"
            )
            for a, b in PAIRS:
                if a not in have or b not in have:
                    continue
                A, B = get(a), get(b)
                lr = B[f"{key}|logp"] - A[f"{key}|logp"]
                tgt = A[f"{key}|target"]
                th = A[f"{key}|think"]
                notc = tgt != colon
                dpc = B[f"{key}|p_colon"][notc].mean() - A[f"{key}|p_colon"][notc].mean()
                ca = (B[f"{key}|argmax"][notc] == colon).mean() - (
                    A[f"{key}|argmax"][notc] == colon
                ).mean()
                tc = np.where(th, np.char.add("T:", cls[tgt]), np.char.add("A:", cls[tgt]))
                loss = np.clip(-lr, 0, None)
                tot = loss.sum()
                by = {c: loss[tc == c].sum() for c in np.unique(tc)}
                top = sorted(by.items(), key=lambda kv: -kv[1])[:5]
                print(
                    f"{a + '->' + b:18s} {100 * (lr < np.log(0.8)).mean():6.2f}% "
                    f"{100 * (lr > np.log(3)).mean():6.3f}% {lr.mean():+8.4f} "
                    f"{100 * dpc:+7.3f}% {100 * ca:+10.3f}%  "
                    + ", ".join(f"{c} {100 * v / max(tot, 1e-9):.0f}%" for c, v in top)
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
