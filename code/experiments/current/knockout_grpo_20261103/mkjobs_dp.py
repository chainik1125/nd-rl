"""Amendment 2 jobs: line_<arm>_s<seed>_dp (+ line2) on this host, Dan's rl_targets as the pool."""

import json
import sys

KO = "/scratch/cowifornia/nd-rl/knockout_grpo"
P = f"{KO}/pools/dan_rl_targets.jsonl"
arm, s, exp_h = sys.argv[1], sys.argv[2], float(sys.argv[3])
base = (
    f"python3 {KO}/tools/line.py --arm {arm} --seed {s} --t_l3 364 --bar 26.31 --variant dp "
    f"--pool_l2 {P} --pool_l3 {P} --pool_extra {P} --lean_dump"
)
t = f"{arm}_s{s}_dp"
jobs = [
    {"id": f"line_{t}", "cmd": base, "expected_h": exp_h, "timeout_h": 2 * exp_h + 2},
    {
        "id": f"line2_{t}",
        "cmd": base + " --resume_line",
        "expected_h": exp_h,
        "timeout_h": 2 * exp_h + 2,
        "deps": [f"line_{t}"],
    },
]
with open(f"{KO}/queue/jobs.jsonl", "a") as f:
    for j in jobs:
        f.write(json.dumps(j) + "\n")
print([j["id"] for j in jobs])
