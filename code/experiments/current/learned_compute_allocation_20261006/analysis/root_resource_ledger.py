"""Recorded root-owned model-work intervals; never infer missing GPU occupancy."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    owned = {
        "allocation/no_cost/dev_headlr1e3_s71": 3,
        "allocation/no_cost/dev_headlr1e3_s71_dev32_paths": 3,
        "allocation/no_cost/dev_headlr1e3_s71_dev32_paths_s52006": 3,
        "allocation/format_lr1e5_500_main_dev16_paths_s52006": 3,
        "allocation/main_dev64_live_fp32_format_s52006": 3,
        "allocation/main_dev64_live_fp32_format_lr1e5_step500_s52006": 3,
        "allocation/main_dev64_live_fp32_format_lr1e5_step1000_s52006": 3,
        **{f"allocation/cost/main_s{s}": gpu for s, gpu in [(101, 3), (102, 1), (103, 3)]},
        "allocation/fixed1/main_s101": 1,
    }
    for folder in (ART / "context").iterdir():
        if folder.is_dir() and (folder / "config.json").exists():
            owned[str(folder.relative_to(ART))] = None
    for folder in (ART / "allocation/cost").glob("*_paths_s52006"):
        owned[str(folder.relative_to(ART))] = 3
    launches = {}
    for source in [
        ART / "allocation/cost/dev_launches.jsonl",
        ART / "root_aux_dev_launches.jsonl",
        *sorted((ART / "context").glob("causal_dev_s*_launches.jsonl")),
    ]:
        for row in rows(source):
            key = str(Path(row["output"]).resolve())
            if key in launches:
                raise ValueError("Multiple recorded launches would double count a run")
            launches[key] = dict(row, ledger_path=str(source), ledger_sha256=sha(source))
            relative = str(Path(key).relative_to(ART))
            owned[relative] = row.get("gpu", owned.get(relative, 1 if "aux" in source.name else 3))
    ledger = []
    for relative, gpu in owned.items():
        folder = ART / relative
        config_path = folder / "config.json"
        if not folder.exists():
            continue
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        record = {
            "path": str(folder.relative_to(ROOT)),
            "assigned_gpu": gpu,
            "config_sha256": sha(config_path) if config_path.exists() else None,
            "pid": config.get("pid"),
            "recorded_wall_seconds": None,
            "timing_scope": "not recorded; missing is not zero",
        }
        training = rows(folder / "metrics.jsonl")
        paths = rows(folder / "public_paths.jsonl")
        attempts = rows(folder / "attempts_public.jsonl")
        if training and "seconds" in training[0]:
            record.update(
                recorded_wall_seconds=sum(r["seconds"] for r in training),
                timing_scope=(
                    "sum of complete training update intervals, including CPU work; "
                    "excludes startup/shutdown"
                ),
                completed_updates=len(training),
                prescribed_updates=config.get("updates"),
            )
            for field in [
                "lean_process_seconds",
                "first_attempt_diagnostic_lean_process_seconds",
                "teacher_diagnostic_wall_seconds",
            ]:
                record[field] = sum(r.get(field, 0) for r in training)
        elif paths:
            record.update(
                recorded_wall_seconds=sum(
                    r.get("policy_execution_wall_seconds", 0)
                    + r.get("diagnostic_replay_wall_seconds", 0)
                    for r in paths
                ),
                timing_scope=(
                    "model sections plus diagnostic replay only; "
                    "excludes Env, prompt building, checker and setup"
                ),
                completed_paths=len(paths),
            )
            for field in [
                "generator_forward_tokens",
                "controller_prefill_tokens",
                "diagnostic_replay_forward_tokens",
            ]:
                record[field] = sum(r.get(field, 0) for r in paths)
        elif attempts:
            if len({r["attempt_uid"] for r in attempts}) != len(attempts):
                raise ValueError("Repeated causal attempt IDs would double count work")
            record.update(
                recorded_wall_seconds=sum(
                    r["candidate"].get("sampling_wall_seconds", 0)
                    + r["candidate"].get("likelihood_replay_wall_seconds", 0)
                    for r in attempts
                ),
                timing_scope=(
                    "unique generation and replay sections only; "
                    "excludes controller, checker and setup"
                ),
                completed_attempts=len(attempts),
            )
        launch = launches.get(str(folder.resolve()))
        if launch:
            record["section_wall_seconds_lower_bound"] = record["recorded_wall_seconds"]
            record["recorded_wall_seconds"] = launch["monotonic_process_wall_seconds"]
            record["timing_scope"] = (
                "monotonic whole child-process interval, including CPU work and GPU setup"
            )
            record["returncode"] = launch["returncode"]
            record["launch_ledger"] = launch["ledger_path"]
            record["launch_ledger_sha256"] = launch["ledger_sha256"]
        ledger.append(record)
    result = {
        "owner": "/root",
        "additional_paid_gpu_spend_usd": 0,
        "agent_billing": "not available; not claimed to be zero",
        "runs": ledger,
        "recorded_assigned_gpu_wall_seconds_lower_bound": sum(
            r["recorded_wall_seconds"] or 0 for r in ledger
        ),
        "missing_timing_runs": [r["path"] for r in ledger if r["recorded_wall_seconds"] is None],
        "caveats": [
            "Not kernel-active GPU time, FLOPs, or exact total GPU occupancy.",
            "Intervals have differing scopes; missing startup/CPU/unrecorded work is not imputed.",
            "GPU assignments are recorded where known; context probes lack a durable GPU ID.",
            "Other agents maintain disjoint owned-run ledgers.",
        ],
    }
    output = ART / "root_resource_ledger.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    shutil.copy2(__file__, output.with_suffix(".source.py"))
    print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
