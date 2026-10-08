#!/usr/bin/env python3
"""Read every run's summary.json and print the comparison as JSON.

  python analyze.py [artifacts dir]

For a retry run the episode eval (8 episodes of up to 8 attempts) is set against what independent
sampling from the SAME model would have done with the same number of proofs, computed exactly
from the scored eval's per-theorem count of accepted samples k out of 64 (draws without
replacement):

  solved with m samples       sum over theorems of 1 - C(64-k, m) / C(64, m)
  success rate of sample n    P(n-1 failures then a success) / P(n-1 failures), summed the same way
"""

import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "artifacts", "retry_ei_20261008"))


def iid_solved(n_ok, m, total=64):
    return sum(1 - comb(total - k, m) / comb(total, m) for k in n_ok)


def iid_rate(n_ok, n, total=64):
    """Success rate of the n-th independent sample given the first n-1 failed."""
    reach = [comb(total - k, n - 1) / comb(total, n - 1) for k in n_ok]
    hit = [r * k / (total - (n - 1)) for r, k in zip(reach, n_ok, strict=True)]
    return sum(hit) / sum(reach)


def run_row(path):
    with open(os.path.join(path, "summary.json")) as f:
        s = json.load(f)
    ev, rl = s.get("eval", {}), s.get("rl", {})
    row = {
        "run": os.path.basename(path),
        "arm": s.get("arm"),
        "status": s["status"],
        "reason": s.get("reason", "")[-300:],
        "dev_iid64": s.get("metric"),
        "rl_rounds": rl.get("stats", {}).get("rounds"),
        "rl_targets_solved": s.get("metric_rl"),
        "total_secs": round(s["total_secs"]),
    }
    n_ok, ep = ev.get("n_ok"), ev.get("episodes")
    if n_ok:
        row["first_sample_rate"] = sum(n_ok) / (64 * len(n_ok))
    if ep and n_ok:
        attempts = ep["max_attempts"]
        row |= {
            "dev_episodes": ep["solved"],
            "episode_tried": ep["tried"],
            "episode_ok": ep["ok"],
            "episode_rate": [
                o / t if t else None for o, t in zip(ep["ok"], ep["tried"], strict=True)
            ],
            "iid_rate": [iid_rate(n_ok, n) for n in range(1, attempts + 1)],
            "episode_solved_by": ep["solved_by"],
            "iid_solved_by": [iid_solved(n_ok, ep["episodes"] * a) for a in range(1, attempts + 1)],
            "eval_codes": ep["codes"],
            "episode_tokens": ep["sampled_tokens"],
        }
    rounds = rl.get("rounds") or []
    if rounds and "tried" in rounds[-1]:
        row["rl_round_tried"] = [r["tried"] for r in rounds]
        row["rl_round_ok"] = [r["ok"] for r in rounds]
        row["rl_codes_last_round"] = rounds[-1]["codes"]
        row["rl_records"] = rl["stats"].get("per_round")
    return row


def main():
    art = sys.argv[1] if len(sys.argv) > 1 else ART
    rows = [
        run_row(os.path.join(art, d))
        for d in sorted(os.listdir(art))
        if os.path.exists(os.path.join(art, d, "summary.json"))
    ]
    json.dump(rows, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
