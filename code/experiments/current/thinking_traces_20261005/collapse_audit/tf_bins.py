"""Collapse audit, probe 4d: teacher-forced NLL and the wrong-argmax mix by the number of `:` tokens
in the input (about the number of hypothesis lines), per checkpoint (CPU, read-only).

Run from the repo root, with M = code.experiments.current.thinking_traces_20261005.collapse_audit:

    code/.venv/bin/python -m $M.tf_bins
"""

from __future__ import annotations

import numpy as np

from code.experiments.current.thinking_traces_20261005.collapse_audit.teacher_force import SCR
from code.thinking.tok import EffortTokenizer

CEDGES = [0, 8, 10, 12, 14, 100]
TAGS = [
    "s0_it39",
    "s0_it49",
    "s0_it50",
    "s0_it52",
    "s0_it60",
    "s0_it78",
    "s1_it39",
    "s1_it49",
    "s1_it52",
    "s1_it78",
]


def main() -> int:
    tok = EffortTokenizer()
    struct = {tok.stoi[s] for s in ("<nl>", "<no>", "</think>", "<think>", "⊢", ":=", ":")}
    for sname in ("t1320", "h12"):
        for cond in ("B0", "B256"):
            key = f"{sname}|{cond}"
            print(
                f"\n## {key}: action NLL by ':' count in input "
                + " ".join(f"[{a},{b})" for a, b in zip(CEDGES[:-1], CEDGES[1:], strict=True))
            )
            rows = []
            for t in TAGS:
                p = SCR / f"tf_{t}.npz"
                if not p.exists():
                    continue
                d = np.load(p)
                th = d[f"{key}|think"]
                nc = d[f"{key}|n_colon_ctx"]
                lp = d[f"{key}|logp"]
                am = d[f"{key}|argmax"]
                tg = d[f"{key}|target"]
                cells = []
                for a, b in zip(CEDGES[:-1], CEDGES[1:], strict=True):
                    m = (~th) & (nc >= a) & (nc < b)
                    cells.append(f"{-lp[m].mean():.4f}" if m.sum() > 50 else "  -   ")
                wrong = (~th) & (am != tg)
                ws = np.isin(am[wrong], list(struct)).mean() if wrong.any() else 0
                tops = np.bincount(am[wrong], minlength=len(tok.itos))
                top = ", ".join(
                    f"{tok.itos[i]} {100 * tops[i] / max(1, wrong.sum()):.0f}%"
                    for i in np.argsort(-tops)[:4]
                )
                cells_t = ""
                if th.any():
                    mt = th
                    cells_t = f" | think NLL {-lp[mt].mean():.4f}"
                rows.append(
                    f"  {t:8s} "
                    + " ".join(cells)
                    + f" | wrong-argmax {100 * wrong.mean():.2f}% of action tokens, "
                    + f"structural {100 * ws:.0f}% ({top})"
                    + cells_t
                )
            print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
