"""Version the premeasurement precision fix and failed dependency chain."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(name):
    return json.loads((ROOT / "plans" / name).read_text())


def save(name, value):
    with (ROOT / "plans" / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    if any((ROOT / "runs" / stage).exists() for stage in ("sft", "pairs", "dpo", "grpo")):
        raise ValueError("V3 repair must precede all substantive training")
    freeze_path = str(ROOT / "plans/primary_source_freeze_v3.json")
    smoke = read("mechanical_smoke32b_v2.json")
    smoke["job_dir"] = str(ROOT / "jobs/mechanical_smoke32b_v3")
    smoke["queue_timeout_seconds"] = 7200
    smoke["args"] = [value.replace("runs/mechanical_smoke32b_v2", "runs/mechanical_smoke32b_v3") for value in smoke["args"]]
    save("mechanical_smoke32b_v3.json", smoke)
    training = read("primary_training.json")
    training["source_freeze"] = freeze_path
    training["out"] = str(ROOT / "runs/training_pipeline_v3")
    save("primary_training_v3.json", training)
    job = read("primary_training_job.json")
    job["job_dir"] = str(ROOT / "jobs/primary_training_v3")
    job["args"] = ["--plan", str(ROOT / "plans/primary_training_v3.json")]
    for dependency in job["wait_for_success"]:
        dependency["path"] = dependency["path"].replace("mechanical_smoke32b_v2", "mechanical_smoke32b_v3")
    save("primary_training_job_v3.json", job)
    previous_job = ROOT / "jobs/primary_training_v3/completed.json"
    previous_pipeline = ROOT / "runs/training_pipeline_v3/completed.json"
    for mode in ("greedy", "sampled"):
        slug = "eval_" + mode
        plan = read(slug + ".json")
        plan["source_freeze"] = freeze_path
        plan["out"] = str(ROOT / "runs" / (slug + "_pipeline_v3"))
        save(slug + "_v3.json", plan)
        job = read(slug + "_job.json")
        job["job_dir"] = str(ROOT / "jobs" / (slug + "_v3"))
        job["args"] = ["--plan", str(ROOT / "plans" / (slug + "_v3.json"))]
        job["wait_for_success"] = [{"path": str(previous_job), "equals": {"exit_code": 0}},
            {"path": str(previous_pipeline), "equals": {"status": "complete"}}]
        save(slug + "_job_v3.json", job)
        previous_job = ROOT / "jobs" / (slug + "_v3") / "completed.json"
        previous_pipeline = ROOT / "runs" / (slug + "_pipeline_v3") / "completed.json"
    old = read("primary_source_freeze_v2.json")
    hashes = {name: sha(name) for name in old["sha256"]}
    allowed_changed = {str(ROOT / "code" / name) for name in ("modeling.py", "train.py", "launch.py")}
    changes = {name for name in hashes if hashes[name] != old["sha256"][name]}
    if not changes <= allowed_changed:
        raise ValueError(f"Unplanned v3 input drift: {changes - allowed_changed}")
    for name in changes:
        archived = ROOT / "code/archive_v2" / Path(name).name
        if sha(archived) != old["sha256"][name]:
            raise ValueError("Prior source was not preserved exactly")
        hashes[str(archived)] = sha(archived)
    extra = [ROOT / "code/prepare_amendment_v3.py", ROOT / "code/prepare_eval_schedule.py"]
    extra += [ROOT / "plans" / name for name in
        ("mechanical_smoke32b_v3.json", "primary_training_v3.json", "primary_training_job_v3.json",
         "eval_greedy_v3.json", "eval_greedy_job_v3.json", "eval_sampled_v3.json", "eval_sampled_job_v3.json")]
    hashes.update({str(path): sha(path) for path in extra})
    save("primary_source_freeze_v3.json", {"version": 3, "utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashes, "supersedes": str(ROOT / "plans/primary_source_freeze_v2.json"),
        "changed_source_files": sorted(changes), "no_substantive_training_or_held_out_outcomes": True,
        "amendment": "Preserve exact FP32 saved adapter tensors when loading policy/reference; strengthen identity checks. Propagate failed job completion to dependent queues. Data, stage budgets and inference contract unchanged."})
    print(json.dumps({"status": "v3_frozen", "files": len(hashes), "changed": sorted(changes)}))


if __name__ == "__main__":
    main()
