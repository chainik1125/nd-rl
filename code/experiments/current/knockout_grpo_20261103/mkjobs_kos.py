"""Amendment 3 jobs for one ko_strict seed on this host: base -> base read (n 256) -> line (our pool)."""  # noqa: E501

import json, sys  # noqa: E401, I001

KO = "/scratch/cowifornia/nd-rl/knockout_grpo"
PY = "/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python"
s, exp_h = sys.argv[1], float(sys.argv[2])
b = f"kos_s{s}"
EV = (
    f"cd {KO}/snap/f35d3daf && {PY} -m code.thinking.eval --ckpt {KO}/bases/{b}/model.pt "
    f"--sets {KO}/sets/strict106.jsonl --out {KO}/evals/base_{b}_k256 --k 256 --budgets 64 --arms plain "  # noqa: E501
    f"--batch 1024 --seed 0 --temperature 0.8 --max_action 512 --max_steps 96 --max_rej 10"
)
jobs = [
    {"id": f"base_{b}", "cmd": f"python3 {KO}/tools/base.py kos {s}", "expected_h": 0.5},
    {"id": f"evbase_{b}", "cmd": EV, "expected_h": 0.3, "deps": [f"base_{b}"]},
    {
        "id": f"line_{b}",
        "expected_h": exp_h,
        "timeout_h": 2 * exp_h + 2,
        "deps": [f"evbase_{b}"],
        "cmd": f"python3 {KO}/tools/line.py --arm kos --seed {s} --t_l3 364 --bar 26.31 --lean_dump",  # noqa: E501
    },
    {
        "id": f"line2_{b}",
        "expected_h": exp_h,
        "timeout_h": 2 * exp_h + 2,
        "deps": [f"line_{b}"],
        "cmd": f"python3 {KO}/tools/line.py --arm kos --seed {s} --t_l3 364 --bar 26.31 --lean_dump --resume_line",  # noqa: E501
    },
]
with open(f"{KO}/queue/jobs.jsonl", "a") as f:
    for j in jobs:
        f.write(json.dumps(j) + "\n")
print([j["id"] for j in jobs])
