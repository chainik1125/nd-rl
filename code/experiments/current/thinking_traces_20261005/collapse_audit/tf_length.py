"""Collapse audit, probe 4c: P(':') at non-':' action targets by state length and by the number of
`:` tokens in the decoder input, per set, from the teacher-forced arrays (CPU, read-only).

Run from the repo root, with M = code.experiments.current.thinking_traces_20261005.collapse_audit:

    code/.venv/bin/python -m $M.tf_length [TAGS]
"""

from __future__ import annotations

import sys

import numpy as np

from code.experiments.current.thinking_traces_20261005.collapse_audit.teacher_force import SCR, SETS
from code.thinking.tok import EffortTokenizer

EDGES = [0, 100, 130, 160, 190, 220, 260, 1000]
CEDGES = [0, 6, 8, 10, 12, 14, 17, 100]


def main() -> int:
    tok = EffortTokenizer()
    colon = tok.stoi[":"]
    tags = sys.argv[1].split(",") if len(sys.argv) > 1 else ["s0_it39", "s0_it49"]
    for cond in ("B0", "B256"):
        for what, edges, field in (
            ("state tokens", EDGES, "state_len"),
            ("':' in input", CEDGES, "n_colon_ctx"),
        ):
            print(
                f"\n## {cond}: P(':') at non-':' action targets (%) by {what}; "
                "n per bin in brackets"
            )
            for sname in SETS:
                key = f"{sname}|{cond}"
                for t in tags:
                    p = SCR / f"tf_{t}.npz"
                    if not p.exists():
                        continue
                    d = np.load(p)
                    v = d[f"{key}|{field}"]
                    m0 = (~d[f"{key}|think"]) & (d[f"{key}|target"] != colon)
                    pc = d[f"{key}|p_colon"]
                    cells = []
                    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                        m = m0 & (v >= lo) & (v < hi)
                        cells.append(
                            f"{100 * pc[m].mean():7.3f} [{m.sum():5d}]"
                            if m.sum() >= 50
                            else f"{'-':>7s} [{m.sum():5d}]"
                        )
                    print(f"  {sname:6s} {t:8s} " + " ".join(cells))
            print(
                "  bins: "
                + " ".join(f"[{lo},{hi})" for lo, hi in zip(edges[:-1], edges[1:], strict=True))
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
