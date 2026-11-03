"""Amendment 4 jobs on this host: gr_<line> (guard-relaxed continuation) and its crash-recovery
line2gr_<line>. Usage: python3 mkjobs_gr.py ARM SEED VARIANT EXPECTED_H   (VARIANT '' or dp)."""

import json
import sys

KO = "/scratch/cowifornia/nd-rl/knockout_grpo"
P = f"{KO}/pools/dan_rl_targets.jsonl"
arm, s, var, exp_h = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
base = f"python3 {KO}/tools/line.py --arm {arm} --seed {s} --t_l3 364 --bar 26.31 --lean_dump --gr"
if len(sys.argv) > 5:  # an explicit resume checkpoint (--gr_from)
    base += f" --gr_from {sys.argv[5]}"
if var == "dp":
    base += f" --variant dp --pool_l2 {P} --pool_l3 {P} --pool_extra {P}"
t = f"{arm}_s{s}" + (f"_{var}" if var else "") + "_gr"
jobs = [
    {"id": f"gr_{t}", "cmd": base, "expected_h": exp_h, "timeout_h": 2 * exp_h + 2},
    {
        "id": f"line2gr_{t}",
        "cmd": base + " --resume_line",
        "expected_h": exp_h,
        "timeout_h": 2 * exp_h + 2,
        "deps": [f"gr_{t}"],
    },
]
with open(f"{KO}/queue/jobs.jsonl", "a") as f:
    for j in jobs:
        f.write(json.dumps(j) + "\n")
print([j["id"] for j in jobs])
