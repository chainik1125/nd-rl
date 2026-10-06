"""Launch the prespecified four-condition development causal panel."""

import argparse
import hashlib
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-seed", type=int, choices=[101, 102], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    selection = ART / f"allocation/no_cost/main_s{args.training_seed}_selection.json"
    choice = json.loads(selection.read_text())
    checkpoint = choice["selected_checkpoint"]
    if (
        sha(checkpoint) != choice["selected_checkpoint_sha256"]
        or choice["selection_split"] != "dev"
    ):
        raise ValueError("Selected model does not match development choice")
    ledger_path = ART / f"context/causal_dev_s{args.training_seed}_launches.jsonl"
    if ledger_path.exists():
        raise ValueError("This causal queue already has a launch ledger")
    shutil.copy2(__file__, ledger_path.with_suffix(".source.py"))
    conditions = [
        ("one_extra", "one_extra", 1, 64, 8, 16),
        ("policy1", "policy", 1, 32, 4, 16),
        ("policy2", "policy", 2, 16, 4, 0),
        ("policy4", "policy", 4, 16, 4, 0),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), ND_OFFLINE="1")
    for name, mode, forced, limit, replicates, history in conditions:
        out = ART / f"context/causal_main_nocost_s{args.training_seed}_{name}_v1"
        command = [
            sys.executable,
            str(WORK / "context/causal_probe.py"),
            "--source",
            str(ART / "source/nd-takehome-dan"),
            "--base",
            str(ART / "allocation/format_main96_lr1e5_clean8_initial50_s61/checkpoint_00999.pt"),
            "--checkpoint",
            checkpoint,
            "--public",
            str(ART / "context/causal_dev_panel_v1/public.jsonl"),
            "--out",
            str(out),
            "--seed",
            "71006",
            "--split",
            "dev",
            "--continuation",
            mode,
            "--forced-extra",
            str(forced),
            "--limit",
            str(limit),
            "--replicates",
            str(replicates),
            "--history-removal-limit",
            str(history),
            "--counts",
            "1",
            "2",
        ]
        record = {
            "condition": name,
            "training_seed": args.training_seed,
            "gpu": args.gpu,
            "command": command,
            "output": str(out),
            "selection_sha256": sha(selection),
            "generator_sha256": sha(WORK / "context/causal_probe.py"),
            "analyzer_sha256": sha(WORK / "context/analyze_causal.py"),
        }
        start = time.monotonic()
        with out.with_suffix(".log").open("x") as stream:
            completed = subprocess.run(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
        record.update(
            monotonic_process_wall_seconds=time.monotonic() - start, returncode=completed.returncode
        )
        with ledger_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if completed.returncode:
            raise RuntimeError(record)
        analysis_command = [
            sys.executable,
            str(WORK / "context/analyze_causal.py"),
            "--run",
            str(out),
            "--out",
            str(out) + "_analysis",
            "--bootstrap",
            "5000",
            "--seed",
            "81006",
        ]
        with Path(str(out) + "_analysis.log").open("x") as stream:
            subprocess.run(
                analysis_command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT
            )
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
