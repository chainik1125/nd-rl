"""Run the prespecified fixed-four control's development-only checkpoint choice."""

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from evaluation_protocol import fingerprint, signature

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[3]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
ALLOC = WORK / "allocation"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    allowed = (ART / "allocation/fixed4").resolve()
    run = allowed / "main_s101"
    ledger = allowed / "dev_launches.jsonl"
    selection = allowed / "main_s101_selection.json"
    outputs = {step: allowed / f"main_s101_dev{step}_paths_s52006" for step in (100, 200)}
    paths = [ledger, ledger.with_suffix(".source.py"), selection, *outputs.values()]
    paths.extend(output.with_suffix(".log") for output in outputs.values())
    if any(not path.resolve().is_relative_to(allowed) for path in paths):
        raise ValueError("Fixed4 output plan escapes its artifact directory")
    shutil.copy2(__file__, ledger.with_suffix(".source.py"))
    training = json.loads((run / "config.json").read_text())
    while Path(f"/proc/{training['pid']}").exists():
        time.sleep(2)
    occupancy = subprocess.check_output(
        ["nvidia-smi", "--id=4", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    if occupancy:
        raise RuntimeError(f"GPU4 occupied after fixed4 training: {occupancy}")
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    if len(metrics) != 200:
        raise ValueError("Fixed4 training did not complete200 updates")
    candidates = []
    for step, output in outputs.items():
        checkpoint = run / f"checkpoint_{step:05d}.pt"
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="4",
            ND_OFFLINE="1",
            ALLOCATION_CHECKPOINT=str(checkpoint),
            ALLOCATION_EVAL_OUT=str(output),
        )
        started = time.monotonic()
        with output.with_suffix(".log").open("x") as stream:
            result = subprocess.run(
                ["bash", str(ALLOC / "launchers/dev_checkpoint.sh")],
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        record = dict(
            step=step,
            output=str(output),
            returncode=result.returncode,
            monotonic_process_wall_seconds=time.monotonic() - started,
        )
        with ledger.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if result.returncode:
            raise RuntimeError(record)
        summary = json.loads((output / "summary.json").read_text())["fixed4"]
        config = json.loads((output / "config.json").read_text())
        candidates.append(
            dict(
                completed_updates=step,
                checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint),
                dev_config_sha256=sha(output / "config.json"),
                inference_sha256=fingerprint(signature(config)),
                correct_probability=summary["correct_probability"],
                attempts=summary["attempts"],
                dev_output=str(output),
            )
        )
    selected = max(
        candidates,
        key=lambda row: (row["correct_probability"], -row["attempts"], -row["completed_updates"]),
    )
    choice = dict(
        selected_completed_updates=selected["completed_updates"],
        selected_checkpoint=selected["checkpoint"],
        selected_checkpoint_sha256=selected["checkpoint_sha256"],
        selection_split="dev",
        selection_readout="fixed4",
        selection_rule=(
            "maximum fixed4 categorical correctness, then fewer attempts, then earlier checkpoint"
        ),
        candidates=candidates,
    )
    selection.write_text(json.dumps(choice, indent=2) + "\n")
    print(json.dumps(choice), flush=True)


if __name__ == "__main__":
    main()
