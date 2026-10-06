"""Wait for all registered jobs, verify commitments, invoke locked CPU statistics."""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from launch_live_test import ART, ROOT, sha
from run_final_eval_queue import all_main_complete

TRUSTED = {
    "main": "bd289d62de0e5c3395fa10330177385184cc66d67d09fc82cead10a65894a1d2",
    "historical": "d228e09e16fa86aa69160e84464eca5064fcecde3c057393e7cc0eecba1b2d7d",
    "entries": "b6058df2967eff32c11df54b1880ceaf5c8436aeae71b5955e6a814d4c8fbd88",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()
    entry_path = ART / "freeze_inputs/allocation_v1/entries.json"
    registries = {
        key: ART / f"final_evaluation/{key}_protocol.json" for key in ("main", "historical")
    }
    commands_file = ART / "allocation/final_aggregation_commands.json"
    commands = json.loads(commands_file.read_text())["commands"]

    def verify(benchmark):
        if sha(entry_path) != TRUSTED["entries"]:
            raise ValueError("Original pre-outcome allocation entries changed")
        for key, path in registries.items():
            if sha(path) != TRUSTED[key]:
                raise ValueError("Original root-sealed registry changed")
        commitment = json.loads(entry_path.read_text())["planned_run_manifests"][benchmark]
        if sha(commitment["path"]) != commitment["sha256"]:
            raise ValueError("Pre-outcome planned-run manifest changed")
        command = commands[benchmark]
        if command[command.index("--runs") + 1] != commitment["path"]:
            raise ValueError("Aggregation command does not use original committed manifest")
        return commitment

    verify("main")
    verify("historical")
    ledger = Path(args.ledger).resolve()
    if not ledger.is_relative_to(ART.resolve()) or any(
        path.exists() for path in (ledger, ledger.with_suffix(".source.py"))
    ):
        raise ValueError("New artifact ledger/source required")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    record = dict(
        owner="/root/allocation",
        pid=os.getpid(),
        command=sys.argv,
        started_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        status="waiting",
        trusted_hashes=TRUSTED,
        source_sha256=sha(__file__),
        commands_sha256=sha(commands_file),
        phases=[],
        scope="CPU orchestration/statistics; no GPU work",
    )
    with ledger.open("x") as stream:
        json.dump(record, stream, indent=2)
    started = time.monotonic()
    try:
        for benchmark in ("main", "historical"):
            while not all_main_complete(ART / f"allocation/final_{benchmark}", TRUSTED[benchmark]):
                verify(benchmark)
                time.sleep(10)
            commitment = verify(benchmark)
            if sha(commands_file) != record["commands_sha256"]:
                raise ValueError("Prepared command changed")
            command = commands[benchmark]
            output = Path(command[command.index("--out") + 1])
            if output.exists():
                raise ValueError("Refusing to overwrite aggregation output")
            phase_started = time.monotonic()
            record["status"] = f"aggregating_{benchmark}"
            ledger.write_text(json.dumps(record, indent=2) + "\n")
            result = subprocess.run(
                command, cwd=ROOT, env=dict(os.environ, ND_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
            )
            phase = dict(
                benchmark=benchmark,
                command=command,
                run_manifest_sha256=commitment["sha256"],
                returncode=result.returncode,
                monotonic_cpu_process_wall_seconds=time.monotonic() - phase_started,
                output=str(output),
            )
            if output.exists():
                phase["output_sha256"] = sha(output)
            record["phases"].append(phase)
            ledger.write_text(json.dumps(record, indent=2) + "\n")
            if result.returncode:
                raise RuntimeError(f"Guarded {benchmark} aggregation failed")
        record["status"] = "complete"
    except Exception as error:
        record["status"], record["error"] = "failed", str(error)
        raise
    finally:
        record["whole_cpu_queue_wall_seconds_including_wait"] = time.monotonic() - started
        ledger.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
