"""line2_<tag> jobs: `line.py --resume_line` (Amendment 1), optionally after line_<tag> is done."""

import json
import sys

KO = "/scratch/cowifornia/nd-rl/knockout_grpo"
exp_h, dep, *tags = sys.argv[1:]
with open(f"{KO}/queue/jobs.jsonl", "a") as f:
    for tag in tags:
        arm, s = tag.rsplit("_s", 1)
        j = {
            "id": f"line2_{tag}",
            "expected_h": float(exp_h),
            "timeout_h": 2 * float(exp_h) + 2,
            "cmd": f"python3 {KO}/tools/line.py --arm {arm} --seed {s} --t_l3 364 --bar 26.31 --resume_line",  # noqa: E501
        }
        if dep == "dep":
            j["deps"] = [f"line_{tag}"]
        f.write(json.dumps(j) + "\n")
        print(j["id"], j.get("deps"))
