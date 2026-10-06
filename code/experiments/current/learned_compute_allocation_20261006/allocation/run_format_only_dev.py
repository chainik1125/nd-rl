"""Measure the already selected format initialization after the fixed4 DEV queue."""

import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[3]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"


def main():
    allowed = (ART / "allocation/format_only").resolve()
    output = allowed / "dev32_paths_s52006"
    log, ledger, snapshot = (
        allowed / "dev_launch.log",
        allowed / "dev_launch.json",
        allowed / "dev_launch.source.py",
    )
    if any(not p.resolve().is_relative_to(allowed) for p in (output, log, ledger, snapshot)):
        raise ValueError("Format-only output plan escapes its directory")
    allowed.mkdir(exist_ok=True)
    shutil.copy2(__file__, snapshot)
    while not (ART / "allocation/fixed4/main_s101_selection.json").exists():
        time.sleep(2)
    occupancy = subprocess.check_output(
        ["nvidia-smi", "--id=4", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    if occupancy:
        raise RuntimeError(f"GPU4 occupied after fixed4 development queue: {occupancy}")
    command = [
        sys.executable,
        str(WORK / "allocation/evaluate_live_path.py"),
        "--source",
        str(ART / "source/nd-takehome-dan"),
        "--base",
        str(ART / "allocation/format_main96_lr1e5_clean8_initial50_s61/checkpoint_00999.pt"),
        "--data",
        str(ART / "allocation/main_dev32_public_panel.jsonl"),
        "--out",
        str(output),
        "--seed",
        "52006",
        "--repeats",
        "1",
        "--context",
        "16384",
    ]
    started_utc = datetime.datetime.now(datetime.UTC).isoformat()
    started = time.monotonic()
    with log.open("x") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="4", ND_OFFLINE="1"),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    record = dict(
        command=command,
        started_utc=started_utc,
        gpu=4,
        monotonic_process_wall_seconds=time.monotonic() - started,
        returncode=result.returncode,
        output=str(output),
    )
    ledger.write_text(json.dumps(record, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(record)


if __name__ == "__main__":
    main()
