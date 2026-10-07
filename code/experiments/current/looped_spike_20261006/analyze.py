#!/usr/bin/env python3
"""Tables for the looped spike, from artifacts/looped_spike_20261006/<arm>/.

  python analyze.py > results_table.md

Per arm, stage, pool and test-time loop count: problems solved with all k samples, and the
expected number solved at run 198's compute, i.e. with m = k * 6 / depth samples (depth = blocks
applied per token; 198 applies 6). The expectation is exact over m-subsets of the k samples:
a problem with c successes is missed by a random m-subset with probability C(k-c, m) / C(k, m).
"""

import glob
import json
import math
import os
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "..", "..", "..", "artifacts", "looped_spike_20261006")
BASE_DEPTH = 6


def expected_solved(counts, k, m):
    return sum(1 - math.comb(k - c, m) / math.comb(k, m) for c in counts)


def main():
    for summary in sorted(glob.glob(f"{ART}/*/summary.json")):
        s, arm = json.loads(Path(summary).read_text()), os.path.basename(os.path.dirname(summary))
        pre, rounds = s.get("pretrain", {}), s.get("rl", {}).get("rounds", [])
        print(f"## {arm}: harness status {s['status']}, score {s['metric']}\n")
        print(
            f"- pretrain: {pre.get('steps')} steps, {pre.get('n_params'):,} params, held-out "
            f"greedy {pre.get('heldout_greedy')}, val loss by loops {pre.get('val_by_loops')}"
        )
        print(
            f"- RL targets solved by round: {[r['targets_solved'] for r in rounds]}; sampling "
            f"{[round(r['sampling']['samples_per_sec']) for r in rounds]} samples/s; "
            f"total {s['total_secs']:.0f} s\n"
        )
    paths = glob.glob(f"{ART}/*/loop_sweep_*.json") + glob.glob(
        f"{ART}/*/more_rl/loop_sweep_*.json"
    )
    for path in sorted(paths):
        sweep, arm = json.loads(Path(path).read_text()), os.path.relpath(os.path.dirname(path), ART)
        print(f"## {arm}, {sweep['stage']}.pt\n")
        cols = "pool|loops|depth|solved at k|m at 198's compute|solved at m|pass@1|samples/s"
        print(f"| {cols.replace('|', ' | ')} |\n|{'---|' * 8}")
        for r in sweep["results"]:
            counts, k = [p["n_ok"] for p in r["problems"].values()], r["k"]
            m = min(k, k * BASE_DEPTH // r["depth"])
            print(
                f"| {r['pool']} | {r['loops'] or 'untied'} | {r['depth']} | {r['solved']}/{r['n']} "
                f"(k={k}) | {m} | {expected_solved(counts, k, m):.1f} | "
                f"{sum(counts) / (k * r['n']):.3f} | {r['samples_per_sec']:.0f} |"
            )
        print()


if __name__ == "__main__":
    main()
