"""Training-run and read summaries behind O1, O9 and O10.

    python3 run_summaries.py > out/run_summaries.txt

1. O9: pass@1 and pass@4 (the summaries store k = 1, 2, 4, 8, ...) of the EI knob pilots and the GRPO references,
   from each read's summary.json (the `all` cell, or the set's only cell). Charles is charles173 (all 173 problems),
   not the clean 149 that v2_results reports.
2. O1, O10: from each GRPO stage's metrics.jsonl, the distinct (target, budget) pairs seen by the last iteration
   (`curriculum.seen`), the training pass rate per budget at the first and last iteration, think tokens per block
   per budget, and the action entropy.

Standard library only; CPU, one process.
"""

from __future__ import annotations

import json
import os

RUNS = "/scratch/cowifornia/nd-rl/think/runs"
EV = f"{RUNS}/eval/v2"
GV = f"{RUNS}/grpo_v2"
READ_ARMS = [
    "ei_think_v_s0",
    "ei_think_v_krr_s0",
    "ei_think_v_kret0_s0",
    "ei_think_v_kr2_s0",
    "ei_think_v_kt10_s0",
    "ei_think_v_k3p_s0",
    "grpo_think_vs_mb_s0",
    "grpo_think_vs_mb_krr_s0",
    "grpo_think_vs_mb_kkl0_s0",
]
READ_SETS = [("t1320", 16), ("t2130", 8), ("charles173", 64)]
STAGES = [
    "grpo_think_s0_L2",
    "grpo_think_vs_s0_L2",
    "grpo_think_vs_s2_L3",
    "grpo_think_vs_b_s2_L3",
    "grpo_think_vs_pb_s2_L3",
    "grpo_think_vs_mb_s0_L3",
    "grpo_think_vs_mb_krr_s0_L3",
    "grpo_think_vs_mb_kkl0_s0_L3",
    "grpo_think_vs_mb_kg64_s0_L3",
    "grpo_think_vs_mb_ki78_s0_L3",
    "grpo_think_vs_mb_kt12_s0_L3",
]


def read_cell(path: str) -> tuple[int, float, float] | None:
    with open(path) as f:
        j = json.load(f)
    for cell in j["cells"].values():
        w = cell.get("all") or next(iter(cell.values()))
        p = w["pass_at_k_pct"]
        return w["n"], p["1"], p.get("5", p.get("4", float("nan")))
    return None


def reads() -> None:
    print("## O9: pass@1 / pass@4 (% of set) of the EI knob pilots and GRPO references, plain reads at B64\n")
    for st, k in READ_SETS:
        for arm in READ_ARMS:
            path = f"{EV}/{st}_{arm}_plain_k{k}/summary.json"
            if not os.path.exists(path):
                print(f"{st:10s} {arm:28s} missing")
                continue
            c = read_cell(path)
            if c:
                print(f"{st:10s} {arm:28s} n={c[0]:4d} pass@1 {c[1]:5.1f} pass@4 {c[2]:5.1f}")
        print()


def stages() -> None:
    print("## O1, O10: GRPO stages (metrics.jsonl)\n")
    for s in STAGES:
        path = f"{GV}/{s}/metrics.jsonl"
        if not os.path.exists(path):
            print(f"{s}: missing")
            continue
        rows = [json.loads(line) for line in open(path)]
        first, last = rows[0], rows[-1]

        def r2(d: dict) -> dict:
            return {b: round(v, 2) for b, v in sorted(d.items())}

        print(
            f"{s:30s} iters {len(rows):3d} pairs seen {last['curriculum']['seen']:5d}"
            f" pass1 first {r2(first['pass1_by_budget'])} last {r2(last['pass1_by_budget'])}"
        )
        print(
            f"{'':30s} think tokens/block first {r2(first['think_block_by_budget'])}"
            f" last {r2(last['think_block_by_budget'])}"
            f" action entropy {first['entropy_action']:.3f} -> {last['entropy_action']:.3f}"
        )
    print()


def main() -> int:
    reads()
    stages()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
