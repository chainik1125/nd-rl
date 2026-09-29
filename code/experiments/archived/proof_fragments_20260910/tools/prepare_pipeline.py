"""Freeze a root-authored pipeline into a scoped, leased GPU job."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/data/users/dmitry/simplex-research/.venv/bin/python"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plan", type=Path)
    args = p.parse_args()
    path = args.plan.resolve()
    if not path.is_relative_to(ROOT / "configs"):
        raise ValueError("Pipeline plan must be in sprint configs")
    plan = json.loads(path.read_text())
    name = plan["name"]
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
        raise ValueError("Invalid pipeline name")
    target = ROOT / "snapshots" / ("pipeline_" + name)
    target.mkdir(parents=True, exist_ok=False)
    entry = target / "pipeline.py"
    entry.write_text('''from pathlib import Path
import json, subprocess, time
plan = json.loads(Path(''' + repr(str(path)) + ''').read_text())
for index, command in enumerate(plan["commands"]):
    print(json.dumps({"event":"stage_start", "index":index, "command":command, "unix_time":time.time()}), flush=True)
    subprocess.run(command, check=True)
    print(json.dumps({"event":"stage_complete", "index":index, "unix_time":time.time()}), flush=True)
''')
    files = {path, entry, ROOT / "tools/launch.py",
             ROOT.parent / "r1-sprint-20260909-1625/code/dispatch.py"}
    for snapshot in plan["source_snapshots"]:
        directory = Path(snapshot).resolve()
        if not directory.is_relative_to(ROOT / "snapshots"):
            raise ValueError("Source snapshot is outside this sprint")
        files.update(directory.rglob("*.py"))
    vendor = ROOT.parent / "r1-sprint-20260909-1625/code/vendor"
    files.update(vendor.rglob("*.py"))
    files.update(Path(p).resolve() for p in plan["inputs"])
    hashes = {}
    for file in sorted(files):
        with file.open("rb") as stream:
            hashes[str(file)] = hashlib.file_digest(stream, "sha256").hexdigest()
    spec = {k: plan[k] for k in ("name", "question", "expected_seconds", "deadline_utc", "timeout_seconds")}
    spec.update(command=[PYTHON, str(entry)], input_sha256=hashes,
                candidate_gpus=plan.get("candidate_gpus", [4]),
                queue_timeout_seconds=plan.get("queue_timeout_seconds", 180),
                expected_cloud_cost_usd=0, pipeline_plan=str(path),
                success="Every declared stage exits zero and writes its bound completion artifacts",
                failure="Any stage error, input drift, deadline, or nonzero worker exit",
                recovery="Inspect jobs/<name>/worker.log and each stage's immutable outputs; no automatic retry")
    dest = ROOT / "configs" / ("job_" + name + ".json")
    with dest.open("x") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")
    print(json.dumps({"job_spec": str(dest), "stages": len(plan["commands"]), "input_files": len(hashes)}))


if __name__ == "__main__":
    main()
