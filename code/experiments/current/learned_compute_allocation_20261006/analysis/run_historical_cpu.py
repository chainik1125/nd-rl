"""Run exact prepared historical CPU commands after the MAIN completion barrier."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ART = Path("artifacts/learned_compute_allocation_20261006").resolve()
PLAN_SHA = "9dec3e58c9f119a095312a828a5083deae00d78804dac25d5906adef879b28cb"
MAIN_SHA = "bd289d62de0e5c3395fa10330177385184cc66d67d09fc82cead10a65894a1d2"
HIST_SHA = "d228e09e16fa86aa69160e84464eca5064fcecde3c057393e7cc0eecba1b2d7d"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_bindings(plan_path: Path, entry: dict | None = None) -> None:
    if digest(plan_path) != PLAN_SHA:
        raise ValueError("Prepared command plan changed")
    for benchmark, expected in [("main", MAIN_SHA), ("historical", HIST_SHA)]:
        if digest(ART / f"final_evaluation/{benchmark}_protocol.json") != expected:
            raise ValueError(f"{benchmark} registry changed")
    if entry is None:
        return
    command = entry["command"]
    if digest(Path(command[1])) != entry["entrypoint_sha256"]:
        raise ValueError(f"Scientific entrypoint changed: {entry['name']}")
    target = Path(command[command.index("--out") + 1])
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    for flag in ("--predictions", "--prediction-config"):
        if flag in command and not Path(command[command.index(flag) + 1]).is_file():
            raise FileNotFoundError(f"Historical input not ready: {entry['name']} {flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["initial", "remaining"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root-release-tool-utc", required=True)
    args = parser.parse_args()
    plan_path = ART / "analysis/historical_cpu_commands_v1.json"
    validate_bindings(plan_path)
    labels = [f"{arm}_s{seed}" for arm in ("cost", "no_cost") for seed in (101, 102, 103)]
    labels += ["fixed1_control_s101", "fixed4_control_s101", "format_only"]
    barrier = []
    for label in labels:
        path = ART / f"allocation/final_main/{label}.launch.json"
        record = json.loads(path.read_text())
        if record.get("status") != "complete" or record.get("registry_sha256") != MAIN_SHA:
            raise ValueError(f"MAIN barrier incomplete or unbound: {label}")
        if not record.get("phases") or any(p.get("returncode") != 0 for p in record["phases"]):
            raise ValueError(f"MAIN barrier has unfinished or failed phases: {label}")
        barrier.append({"path": str(path), "sha256": digest(path), "status": "complete"})
    entries = json.loads(plan_path.read_text())["commands"]
    entries = entries[:2] if args.phase == "initial" else entries[2:]
    for entry in entries:
        validate_bindings(plan_path, entry)
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.out / "launcher.py")
    ledger = {
        "phase": args.phase,
        "root_release_tool_utc": args.root_release_tool_utc,
        "plan_sha256": PLAN_SHA,
        "main_barrier": barrier,
        "timing_scope": "CPU scientific child process intervals; excludes launcher preflight",
        "status": "running",
        "commands": [],
    }

    def save() -> None:
        (args.out / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")

    save()
    env = dict(os.environ, ND_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    for entry in entries:
        try:
            validate_bindings(plan_path, entry)
        except (ValueError, OSError) as exc:
            ledger.update(status="failed_preflight", error=str(exc))
            save()
            raise
        record = {"name": entry["name"], "command": entry["command"], "status": "running"}
        ledger["commands"].append(record)
        save()
        started = time.perf_counter()
        with (args.out / f"{entry['name']}.log").open("w") as stream:
            completed = subprocess.run(entry["command"], env=env, stdout=stream, stderr=stream)
        record.update(
            process_wall_seconds=time.perf_counter() - started,
            returncode=completed.returncode,
            status="complete" if completed.returncode == 0 else "failed",
        )
        if completed.returncode:
            ledger["status"] = "failed"
            save()
            raise RuntimeError(f"Historical command failed: {entry['name']}")
        save()
        print(json.dumps(record), flush=True)
    ledger["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
