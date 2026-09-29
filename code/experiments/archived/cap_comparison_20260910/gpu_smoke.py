"""One real GPU update per stage; numerical/serialization smoke, not evidence."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
spec = json.loads((ROOT / "plans/cap12_cot_s20260910.json").read_text())
env = {**os.environ, "PYTHONPATH": str(ROOT / "snapshots/main_v1/vendor")}
outputs = {}
for stage in spec["training_stages"]:
    kind = stage["command_name"]
    command = list(stage["command"])
    def replace(flag, value):
        command[command.index(flag) + 1] = str(value)
    output = ROOT / "runs" / f"gpu_smoke_v1_{kind}"
    outputs[kind] = output
    replace("--output", output)
    replace("--steps", 2)
    replace("--max-flop-proxy", 1)
    start = command.index("--flop-milestones") + 1
    end = start
    while end < len(command) and not command[end].startswith("--"):
        end += 1
    command[start:end] = ["1"]
    if kind == "rl":
        replace("--checkpoint", outputs["sft"] / "final.pt")
    subprocess.run(command, env=env, check=True)
    marker = json.loads((output / "completed.json").read_text())
    assert marker["actual_steps"] == 1 and marker["budget_reached"]
    assert marker["stop_reason"] == "flop_budget"
    if kind == "rl":
        row = json.loads((output / "events.jsonl").read_text().splitlines()[-1])
        assert row["behavior_teacher_max_abs_error"] <= .01
        assert row["actual_response_tokens"] > 0
receipt = {"status": "passed", "purpose": "GPU numerical and compute-accounting smoke only",
           "stages": {k: json.loads((p / "completed.json").read_text()) for k,p in outputs.items()}}
(ROOT / "reports/gpu_smoke.json").write_text(json.dumps(receipt, indent=2) + "\n")
