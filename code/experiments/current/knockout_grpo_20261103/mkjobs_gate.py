"""Gate-phase jobs (bases, base reads, pilots) per host -> KO/queue/jobs.jsonl on that host."""

import json
import sys

host = sys.argv[1]
KO = "/scratch/cowifornia/nd-rl/knockout_grpo"
PY = "/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python"
EV = (
    f"cd {KO}/snap/f35d3daf && {PY} -m code.thinking.eval --ckpt {KO}/bases/{{b}}/model.pt "
    f"--sets {KO}/sets/strict106.jsonl --out {KO}/evals/base_{{b}}_k256 --k 256 --budgets 64 "
    f"--arms plain --batch 1024 --seed 0 --temperature 0.8 --max_action 512 --max_steps 96 --max_rej 10"  # noqa: E501
)
L2IT39 = f"{KO}/bases/full_L2it39/grpo_think_vs_s0_L2_it39.pt"
PIL = f"python3 {KO}/tools/line.py --pilot_l3 10 --seed 0 --l3_init {L2IT39}"
plan = {
    "serrano": {
        "bases": [("d1", 0), ("d1", 1)],
        "full_evals": [0, 1, 2],
        "pilots": [PIL + " --pilot_eval"],
    },
    "ynez": {
        "bases": [("ko", 0), ("ko", 1), ("ko", 2), ("d01", 0), ("d1", 2)],
        "full_evals": [],
        "pilots": [PIL + " --pilot_eval", PIL],
    },
    "temescal": {
        "bases": [("d01", 1), ("d01", 2)],
        "full_evals": [],
        "pilots": [PIL + " --pilot_eval"],
    },
}[host]
jobs = []
for arm, s in plan["bases"]:
    jobs.append(
        {
            "id": f"base_{arm}_s{s}",
            "cmd": f"python3 {KO}/tools/base.py {arm} {s}",
            "expected_h": 0.5,
        }
    )
for i, c in enumerate(plan["pilots"]):
    jobs.append(
        {
            "id": f"pilot_{host}_{i}{'_ev' if 'pilot_eval' in c else ''}",
            "cmd": c,
            "expected_h": 0.75,
        }
    )
for arm, s in plan["bases"]:
    jobs.append(
        {
            "id": f"evbase_{arm}_s{s}",
            "cmd": EV.format(b=f"{arm}_s{s}"),
            "expected_h": 0.3,
            "deps": [f"base_{arm}_s{s}"],
        }
    )
for s in plan["full_evals"]:
    jobs.append({"id": f"evbase_full_s{s}", "cmd": EV.format(b=f"full_s{s}"), "expected_h": 0.3})
with open(f"{KO}/queue/jobs.jsonl", "w") as f:
    for j in jobs:
        f.write(json.dumps(j) + "\n")
print("\n".join(j["id"] for j in jobs))
