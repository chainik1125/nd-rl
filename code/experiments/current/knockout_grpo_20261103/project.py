"""Pilot timing -> T_L3 and projected GPU-hours / wall clock (PREREG.md, "Iteration budget").

    python3 project.py   # OUT/runs/pilot_*_L3/metrics.jsonl (collected) -> OUT/projection.json

Per pilot: seconds per L3 iteration (the metrics rows' `secs`, iterations 2.. to skip start-up)
and R per iteration (S1g3_s0's Stage-1 tokens). The ynez pilot without a co-located read gives the
clean speed; the others run with the online evaluator reading the same GPU. L2's clean time is the
full arm's own (ynez, 39 iterations, 114-127 s each).
"""

from __future__ import annotations

import json
import os
import statistics as st
from pathlib import Path

import measure as M

OUT = Path(os.environ.get("KO_OUT", "/scratch/cowifornia/nd-rl/knockout_grpo"))
S1 = 440559684.74494827  # S1g3_s0 Stage-1 tokens
BUDGET_H = 15.0
L2_RUNS = [
    Path(f"/scratch/cowifornia/nd-rl/think/runs/grpo_v2/grpo_think_vs_s{s}_L2") for s in range(3)
]
L3_RUNS = [
    Path(f"/scratch/cowifornia/nd-rl/think/runs/grpo_v2/grpo_think_vs_mb_kkl0_s{s}_L3")
    for s in range(3)
]


def stage(d: Path, skip: int = 1) -> dict:
    rows = M.rl_rows(d)[skip:]
    a = json.loads((d / "args.json").read_text())
    return {
        "dir": str(d),
        "host": a["_meta"]["host"] if "_meta" in a else None,
        "n": len(rows),
        "s_per_it": st.mean(r["secs"] for r in rows),
        "R_per_it": st.mean((6 * r["trained"] + 2 * r["sampled"]) / (6 * S1) for r in rows),
    }


def main() -> None:
    pil = {
        p.name: stage(p)
        for p in sorted((OUT / "runs").glob("pilot_*_L3"))
        if (p / "metrics.jsonl").exists()
    }
    l2 = [stage(d, 0) for d in L2_RUNS]
    l3 = [stage(d, 0) for d in L3_RUNS]
    l2_secs = st.mean(x["s_per_it"] for x in l2) * 39
    rep = {
        "pilots": pil,
        "existing_L2_ynez": l2,
        "existing_L3_serrano": l3,
        "L2_secs_clean": l2_secs,
    }
    (OUT / "projection.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
