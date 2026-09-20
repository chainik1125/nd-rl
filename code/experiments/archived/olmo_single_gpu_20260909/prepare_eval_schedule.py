"""Translate the frozen inference contract into sequential, outcome-blind jobs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
contract = json.loads((ROOT / "plans/primary_evaluation_contract.json").read_text())


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


previous_job = ROOT / "jobs/primary_training/completed.json"
previous_pipeline = ROOT / "runs/training_pipeline/completed.json"
for mode in contract["modes"]:
    slug = "eval_" + mode
    steps = []
    for arm in contract["arms"]:
        target = ROOT / "runs" / slug / arm
        args = ["--arm", arm, "--out", str(target), "--cohorts", *contract["cohorts"],
            "--modes", mode, "--max-new-tokens", str(contract["max_new_tokens"]),
            "--batch-size", str(contract["batch_size"][arm]), "--seed", str(contract["seed"])]
        if arm not in ("base_zero", "base_icl32"):
            args += ["--adapter-path", str(ROOT / "runs" / ("grpo" if arm == "rl" else arm) / "final")]
        steps.append({"name": arm, "entrypoint": "evaluate.py", "args": args,
            "required_records": [{"path": str(target / "evaluation_status.json"),
                "equals": {"status": "complete", "attempts": 120 if mode == "greedy" else 480}}]})
    plan = ROOT / "plans" / (slug + ".json")
    save(plan, {"source_freeze": str(ROOT / "plans/primary_source_freeze_v2.json"),
        "out": str(ROOT / "runs" / (slug + "_pipeline")), "steps": steps,
        "not_after_utc": "2026-09-10T05:14:00+00:00", "mode": mode,
        "scope": "Fixed synthetic panel; no textbook payloads. Same target membership, completion cap, checkpoints and protocol in both phases."})
    save(ROOT / "plans" / (slug + "_job.json"), {
        "job_dir": str(ROOT / "jobs" / slug), "entrypoint": "pipeline.py", "candidate_gpus": [3, 4],
        "queue_timeout_seconds": 28800, "timeout_seconds": 14400,
        "not_after_utc": "2026-09-10T05:14:00+00:00",
        "wait_for_success": [{"path": str(previous_job), "equals": {"exit_code": 0}},
            {"path": str(previous_pipeline), "equals": {"status": "complete"}}],
        "args": ["--plan", str(plan)]})
    previous_job = ROOT / "jobs" / slug / "completed.json"
    previous_pipeline = ROOT / "runs" / (slug + "_pipeline") / "completed.json"
print(json.dumps({"phases": contract["modes"], "greedy_attempts": 600, "sampled_attempts": 2400}))
