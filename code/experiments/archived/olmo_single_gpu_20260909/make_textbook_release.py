"""Bind finished checkpoints to an original textbook split without reading its payload.

Root-owned release operation, separate from the frozen training/evaluation code.
Creating a release commits to no further training or evaluation-setting selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SPRINT = Path(__file__).resolve().parents[1]
REPOSITORY = SPRINT.parent
FINAL_FLOOR = datetime.fromisoformat("2026-09-10T05:14:00+00:00")
DATA_SHA256 = {
    "dev": "1b04192d8a2d91b488a0884ccb096004003dbfcf40bb7aea55caa81c3f3cf53c",
    "final": "3ca20ac49db6e3fc4d090c1424f48ff0b89de4686110bb00e194e4b51e4ae6aa",
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_release(cohort: str) -> dict:
    now = datetime.now(timezone.utc)
    if cohort == "final" and now < FINAL_FLOOR:
        raise ValueError("Final textbook release is not yet permitted")
    source_freeze = SPRINT / "plans/primary_source_freeze_v3.json"
    for name, expected in json.loads(source_freeze.read_text())["sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Frozen source or input changed: {name}")
    pipeline_path = SPRINT / "runs/training_pipeline_v3/completed.json"
    pipeline = json.loads(pipeline_path.read_text())
    if pipeline.get("status") != "complete" or pipeline.get("stages") != ["sft", "pairs", "dpo", "grpo"]:
        raise ValueError("The complete primary training pipeline is required")
    plan_path = SPRINT / "plans/primary_training_v3.json"
    if (pipeline.get("plan_sha256") != sha(plan_path)
            or pipeline.get("source_freeze_sha256") != sha(source_freeze)):
        raise ValueError("Completed pipeline identity differs from the frozen training plan")
    job_path = SPRINT / "jobs/primary_training_v3/completed.json"
    if json.loads(job_path.read_text()).get("exit_code") != 0:
        raise ValueError("The owning training job has not exited successfully")
    records, checkpoint_files = [], {}
    for stage, steps in (("sft", 80), ("dpo", 40), ("grpo", 40)):
        path = SPRINT / "runs" / stage / "summary.json"
        summary = json.loads(path.read_text())
        if (summary.get("status") != "complete" or summary.get("mechanical_fixture") is not False
                or summary.get("stage") != stage or summary.get("optimizer_steps") != steps):
            raise ValueError(f"Incomplete or mismatched {stage} endpoint")
        adapter = (SPRINT / "runs" / stage / "final").resolve()
        if Path(summary["adapter"]).resolve() != adapter:
            raise ValueError("Endpoint adapter path changed")
        files = sorted(path for path in adapter.glob("adapter*") if path.is_file())
        expected_weight = adapter / "adapter_model.safetensors"
        if expected_weight not in files or adapter / "adapter_config.json" not in files:
            raise ValueError("Endpoint checkpoint is incomplete")
        audit = summary.get("saved_tensor_audit", {})
        if (audit.get("all_tensors_exact") is not True or sha(expected_weight) != audit.get("sha256")
                or Path(audit.get("checkpoint", "")).resolve() != expected_weight
                or audit.get("adapter_name") != "default" or audit.get("dtype") != "float32"):
            raise ValueError("Endpoint differs from its exact saved-tensor receipt")
        checkpoint_files.update({str(path): sha(path) for path in files})
        records.append({"path": str(path), "sha256": sha(path)})
    records.append({"path": str(pipeline_path), "sha256": sha(pipeline_path)})
    manifest_path = REPOSITORY / "r1-cot-20260909/data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    relative = "textbook_dev.jsonl" if cohort == "dev" else "sealed_final/textbook_final.jsonl"
    if manifest["files"][relative] != DATA_SHA256[cohort]:
        raise ValueError("Original opaque split identity changed")
    # Do not stat, hash, parse, or otherwise open the protected payload here.
    data_path = manifest_path.parent / relative
    release = {
        "kind": "textbook_development_release" if cohort == "dev" else "textbook_final_release",
        "utc": now.isoformat(),
        "allowed_arms": ["base_zero", "base_icl32", "sft", "dpo", "rl"],
        "settings": {"max_new_tokens": 2048, "sampled_k": 4, "temperature": 1.0, "top_p": 1.0, "top_k": 0},
        "checkpoint_files": checkpoint_files,
        "completed_training": records,
        "data_path": str(data_path),
        "data_sha256": DATA_SHA256[cohort],
        "opaque_manifest_path": str(manifest_path),
        "opaque_manifest_sha256": sha(manifest_path),
        "source_freeze_path": str(source_freeze),
        "source_freeze_sha256": sha(source_freeze),
        "training_plan": {"path": str(plan_path), "sha256": sha(plan_path)},
        "training_job": {"path": str(job_path), "sha256": sha(job_path)},
        "all_checkpoints_and_evaluation_settings_frozen": True,
        "no_training_or_setting_selection_after_release": True,
        "protected_payload_read_by_release_builder": False,
    }
    if cohort == "final":
        release["not_before_utc"] = FINAL_FLOOR.isoformat()
    from evaluate import check_textbook_release
    for arm in release["allowed_arms"]:
        adapter = None if arm.startswith("base_") else SPRINT / "runs" / ("grpo" if arm == "rl" else arm) / "final"
        check_textbook_release(release, arm, "textbook_" + cohort, adapter)
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("dev", "final"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only and args.out is None:
        parser.error("--out is required unless --check-only")
    if args.out is not None and not args.out.resolve().is_relative_to(SPRINT / "plans"):
        parser.error("Release output must be inside this sprint's plans directory")
    release = build_release(args.cohort)
    if not args.check_only:
        with args.out.open("x") as stream:
            json.dump(release, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps({"cohort": args.cohort, "ready": True, "created": None if args.check_only else str(args.out),
                      "checkpoint_files": len(release["checkpoint_files"]), "protected_payload_read": False}))


if __name__ == "__main__":
    main()
