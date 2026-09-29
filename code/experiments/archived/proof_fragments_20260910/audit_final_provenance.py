"""Read-only provenance and closed compute-ledger audit; no model execution."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess

import common
import learning as L


def run(output, gpu_observation=None):
    root = common.ROOT
    ledger_path = root / "reports/compute_ledger_final_v1.json"
    ledger = json.loads(ledger_path.read_text())
    hashes = {}
    def bind(path, expected):
        path = str(Path(path).resolve())
        if path not in hashes:
            hashes[path] = common.sha(path)
        assert hashes[path] == expected, ("identity changed", path)
    def read(path):
        return json.loads(Path(path).read_text())
    for path, digest in ledger["input_sha256"].items():
        bind(path, digest)
    assert ledger["active_gpu_lease_seconds_so_far"] == 0
    assert ledger["cloud_spend_usd"] == 0
    assert len(ledger["model_operations"]) == 102
    assert sum(op["new_accounted_flop_proxy"] for op in ledger["model_operations"].values()) == ledger["post_baseline_new_accounted_flop_proxy"]
    assert sum(job["lease_seconds"] for job in ledger["jobs"]) == ledger["completed_gpu_lease_seconds"]
    assert ledger["total_gpu_lease_hours_so_far"] == ledger["completed_gpu_lease_seconds"] / 3600
    launched_outputs = set()
    job_checks = []
    known_pids = set()
    archived_launcher_receipts = []
    for directory in sorted((root / "jobs").iterdir()):
        if not (directory / "spec.json").is_file():
            continue
        spec = read(directory / "spec.json")
        done = read(directory / "completed.json")
        bind(directory / "spec.json", done["spec_sha256"])
        for path, digest in spec["input_sha256"].items():
            if directory.name == "base_sft_v1" and Path(path) == root / "tools/launch.py":
                historical = read(root / "reports/base_sft_v1_receipt.json")
                archived = root / historical["original_launcher_archive"]
                bind(archived, digest)
                assert read(directory / "launcher.json")["launcher_sha256"] == digest
                assert historical["worker_exit_code"] == 0 and historical["supervisor_exit_code"] == 1
                archived_launcher_receipts.append({"job": directory.name, "original_path": path,
                                                  "archived_path": str(archived), "sha256": digest,
                                                  "reason": historical["supervisor_error"]})
                continue
            bind(path, digest)
        for filename in ("launcher.json", "pid.json"):
            if (directory / filename).is_file():
                known_pids.add(read(directory / filename)["pid"])
        events = common.read_jsonl(directory / "events.jsonl")
        acquired = [e for e in events if e["kind"] == "acquired"]
        released = [e for e in events if e["kind"] == "released"]
        if not acquired:
            assert done["status"] == "failed" and done["gpu"] is None and done["elapsed_seconds"] == 0
            job_checks.append({"name": directory.name, "status": "queue_timeout_without_gpu"})
            continue
        assert len(acquired) == 1 and acquired[0]["gpu"]["index"] == 4
        assert done["status"] == "completed" and done["exit_code"] == 0
        if directory.name != "base_sft_v1":
            assert len(released) == 1 and released[0]["status"] == "completed"
            assert released[0]["elapsed_seconds"] == done["elapsed_seconds"]
        else:
            assert not released  # Historical supervisor final-log error is separately receipted.
        if spec.get("pipeline_plan"):
            plan = read(spec["pipeline_plan"])
            for command in plan["commands"]:
                assert Path(command[1]).is_file()
                if "--output" in command:
                    destination = Path(command[command.index("--output") + 1]).resolve()
                    if destination.is_relative_to(root / "results"):
                        launched_outputs.add(str(destination))
        job_checks.append({"name": directory.name, "status": done["status"], "worker_exit_code": 0,
                           "gpu": 4, "completed_utc": done["utc"]})
    assert launched_outputs == set(ledger["model_operations"])
    commands = Counter()
    for directory, op in ledger["model_operations"].items():
        directory = Path(directory)
        assert op["status"] == "completed"
        result, done, manifest = (read(directory / name) for name in ("result.json", "completed.json", "manifest.json"))
        assert done["ok"] is True
        bind(directory / "result.json", done["result_sha256"])
        assert Path(manifest["args"]["output"]).resolve() == directory
        for group in ("code_sha256", "dependency_sha256"):
            for path, digest in manifest[group].items():
                bind(path, digest)
        for key, digest in manifest["input_sha256"].items():
            if Path(key).is_absolute():
                path = key
            elif key in manifest["args"]:
                path = manifest["args"][key]
            elif key == "population_manifest":
                path = Path(manifest["args"]["retention"]).parent / "manifest.json"
            elif key == "population_metadata":
                path = Path(manifest["args"]["data"]).parent / "manifest.json"
            elif key in ("adapted", "base_sft"):
                path = manifest["args"]["adapted_checkpoint" if key == "adapted" else "base_checkpoint"]
            else:
                raise AssertionError(("Unknown producer input schema", directory, key))
            bind(path, digest)
        field = op["cost_field"]
        if field == "initial_cuts.new_pilot_flop_proxy":
            initial = read(directory / "initial_cuts.json")
            cost = initial["new_pilot_flop_proxy"]
            assert cost == L.total_flop_proxy(initial["fit_compute_ledger"])
        else:
            cost = result[field]
            assert cost == L.total_flop_proxy(result["ledger"])
        assert cost == op["new_accounted_flop_proxy"] and cost > 0
        if "final_sha256" in result:
            bind(directory / "final.pt", result["final_sha256"])
        if done.get("final_sha256") is not None:
            target = manifest["args"]["checkpoint"] if manifest["args"].get("command") == "evaluate" else directory / "final.pt"
            bind(target, done["final_sha256"])
            if "final_sha256" in result:
                assert done["final_sha256"] == result["final_sha256"]
        commands[manifest.get("command", manifest["args"].get("command", "unknown"))] += 1
    closure = read(root / "reports/training_closed_v1.json")
    assert closure["no_further_training"] and closure["no_further_selector_changes"]
    for path, digest in closure["checkpoint_sha256"].items():
        bind(path, digest)
    finish = read(root / "configs/pipeline_finish_seed1_v1.json")
    close_index = next(i for i, command in enumerate(finish["commands"]) if Path(command[1]).name == "close_training.py")
    after = finish["commands"][close_index + 1:]
    assert all(Path(command[1]).name in ("release_checkpoints.py", "score_only.py") or "evaluate" in command for command in after)
    late = read(root / "configs/pipeline_retention_development_v1.json")
    assert late["training_updates"] == 0 and all("evaluate" in c for c in late["commands"][1:])
    ps = subprocess.run(["ps", "-p", ",".join(map(str, sorted(known_pids))), "-o", "pid,stat,args"], capture_output=True, text=True)
    assert ps.returncode == 1 and len(ps.stdout.strip().splitlines()) == 1, ps.stdout
    locks = []
    for path in (root / "leases/gpu_4.lock", Path("/tmp/olmo_single_gpu_4.lock"), common.PRIOR / "leases/gpu_4.lock", Path("/tmp/crra_sse_gpu_4.lock")):
        if path.exists():
            with path.open("r") as handle:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                fcntl.flock(handle, fcntl.LOCK_UN)
            locks.append({"path": str(path), "exclusive_lease_held": False})
    query = ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    if gpu_observation:
        observation = read(gpu_observation)
        assert observation["command"] == query and observation["exit_code"] == 0
        assert datetime.fromisoformat(observation["utc"]) <= datetime.now(timezone.utc)
        gpu_stdout, gpu_utc = observation["stdout"], observation["utc"]
        bind(gpu_observation, common.sha(gpu_observation))
    else:
        gpu = subprocess.run(query, capture_output=True, text=True, check=True)
        gpu_stdout, gpu_utc = gpu.stdout, datetime.now(timezone.utc).isoformat()
    gpu4 = next(row for row in gpu_stdout.splitlines() if row.split(",")[0].strip() == "4")
    _, memory, util = map(int, gpu4.split(","))
    assert memory <= 64 and util == 0
    receipt = {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(),
               "compute_ledger_sha256": common.sha(ledger_path), "physical_model_operations": 102,
               "operation_commands": dict(commands), "jobs": job_checks, "unique_files_hashed": len(hashes),
               "post_baseline_new_accounted_flop_proxy": ledger["post_baseline_new_accounted_flop_proxy"],
               "completed_gpu_lease_seconds": ledger["completed_gpu_lease_seconds"],
               "total_gpu_lease_hours": ledger["total_gpu_lease_hours_so_far"], "cloud_spend_usd": 0,
               "base_dense_proxy_available": False, "shared_base_training": ledger["shared_base_training"],
               "historical_launcher_archive_bindings": archived_launcher_receipts,
               "no_active_recorded_launcher_or_worker_pids": True, "recorded_pids_checked": sorted(known_pids),
               "gpu4_memory_mib": memory, "gpu4_utilization_percent": util, "lease_checks": locks,
               "gpu_observation_utc": gpu_utc, "external_gpu_observation": gpu_observation,
               "no_training_after_closure": True, "source_sha256": common.sha(__file__),
               "hashes": hashes,
               "limits": ["GPU and lock state is a point-in-time observation, not a claim about other users' jobs.",
                          "Base trainer did not instrument this dense proxy; base tokens and elapsed time are reported separately.",
                          "CPU audits, proof generation, and interactive analysis are not included in the model-arithmetic proxy.",
                          "This check binds saved likelihoods and compute records; it does not recompute model gradients or logits."]}
    common.write_json(output, receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "physical_model_operations", "unique_files_hashed", "post_baseline_new_accounted_flop_proxy", "total_gpu_lease_hours", "no_active_recorded_launcher_or_worker_pids")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-observation", help="Optional dated receipt from a separately approved nvidia-smi query")
    args = parser.parse_args()
    run(args.output, args.gpu_observation)
