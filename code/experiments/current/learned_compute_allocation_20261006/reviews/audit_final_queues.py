"""CPU-only pre-freeze queue inventory and nine-model barrier audit."""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
WORK = ROOT / "code/experiments/current/learned_compute_allocation_20261006"
sys.path.insert(0, str(WORK / "allocation"))
spec = importlib.util.spec_from_file_location(
    "review_queue", WORK / "allocation/run_final_eval_queue.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
e = json.loads(
    (
        ROOT
        / "artifacts/learned_compute_allocation_20261006/freeze_inputs"
        / "selection_v2/main_entries.json"
    ).read_text()
)
jobs = [j for j in e["commands"] if "_perm" in j["label"] or j["label"] == "frozen_head"]
assert len(jobs) == len({tuple(j["argv"]) for j in jobs}) == 37
result = {"selection_unique_inference_commands": 37}
with tempfile.TemporaryDirectory() as directory:
    d = Path(directory)
    assert not m.all_main_complete(d, "synthetic")
    for arm, seed in m.ALL_MAIN:
        out = d / m.name(arm, seed)
        out.mkdir()
        (out / "summary.json").write_text("{}")
        out.with_suffix(".launch.json").write_text(
            json.dumps({"registry_sha256": "synthetic", "status": "complete"})
        )
    assert m.all_main_complete(d, "synthetic")
    for arm, seed in m.ALL_MAIN:
        out = d / m.name(arm, seed)
        summary = out / "summary.json"
        summary.unlink()
        assert not m.all_main_complete(d, "synthetic")
        summary.write_text("{}")
    result["all_nine_individually_required"] = True
    out.with_suffix(".launch.json").write_text(
        json.dumps({"registry_sha256": "other", "status": "complete"})
    )
    try:
        m.all_main_complete(d, "synthetic")
    except ValueError:
        result["wrong_registry_rejected"] = True
    else:
        raise AssertionError("wrong registry accepted")
    out.with_suffix(".launch.json").write_text(
        json.dumps({"registry_sha256": "synthetic", "status": "failed"})
    )
    try:
        m.all_main_complete(d, "synthetic")
    except RuntimeError:
        result["failed_job_rejected"] = True
    else:
        raise AssertionError("failed job accepted")
print(json.dumps(result, indent=2))
