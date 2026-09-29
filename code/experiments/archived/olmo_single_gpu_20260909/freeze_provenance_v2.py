"""Premeasurement provenance amendment; preserve the original freeze and plan."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
old = ROOT / "plans/primary_source_freeze.json"
old_record = json.loads(old.read_text())
paths = [Path(name) for name in old_record["sha256"]]
paths += list((ROOT.parent / "r1-sprint-20260909-1625/code/vendor").rglob("*.py"))
paths += [ROOT / "plans" / name for name in
    ("primary_training.json", "primary_training_job.json", "primary_evaluation_contract.json")]
paths += [ROOT / "code" / name for name in ("launch.py", "freeze_primary.py", "freeze_provenance_v2.py")]
hashes = {}
for path in sorted(set(paths)):
    with path.open("rb") as stream:
        hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
for name, expected in old_record["sha256"].items():
    if Path(name).name != "pipeline.py" and hashes[name] != expected:
        raise ValueError(f"Unexpected scientific input change during provenance amendment: {name}")
record = {"utc": datetime.now(timezone.utc).isoformat(), "version": 2, "sha256": hashes,
    "supersedes": str(old), "original_plan_preserved": str(ROOT / "plans/primary_training_v1.json"),
    "no_held_out_outcomes_observed": True,
    "change": "Fail-fast immutable plan and imported verifier provenance; scientific settings and core training/evaluation sources unchanged."}
with (ROOT / "plans/primary_source_freeze_v2.json").open("x") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps({"status": "frozen", "version": 2, "files": len(hashes)}))
