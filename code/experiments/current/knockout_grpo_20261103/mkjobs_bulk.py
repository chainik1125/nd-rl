"""Bulk-phase line jobs for one host, appended to KO/queue/jobs.jsonl (ids are unique per host).

python3 mkjobs_bulk.py HOST T_L3 BAR EXPECTED_H TAG [TAG ...]     # TAG = <arm>_s<seed>
"""

from __future__ import annotations

import json
import sys

KO = "/scratch/cowifornia/nd-rl/knockout_grpo"


def main() -> None:
    host, t_l3, bar, exp_h, *tags = sys.argv[1:]
    with open(f"{KO}/queue/jobs.jsonl", "a") as f:
        for tag in tags:
            arm, s = tag.rsplit("_s", 1)
            cmd = f"python3 {KO}/tools/line.py --arm {arm} --seed {s} --t_l3 {t_l3} --bar {bar}"
            j = {"id": f"line_{tag}", "cmd": cmd, "expected_h": float(exp_h),
                 "timeout_h": 2.0 * float(exp_h) + 2.0, "host": host}  # fmt: skip
            f.write(json.dumps(j) + "\n")
            print(j["id"])


if __name__ == "__main__":
    main()
