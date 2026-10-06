"""CPU-only completion barrier for the separately registered live transfer analysis."""

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_transfer_20261006"
PYTHON = "/data/users/dmitry/venvs/proof-state/bin/python"


def main():
    registry_path = ART / "live/protocol.json"
    registry = json.loads(registry_path.read_text())
    while True:
        completed = 0
        for entry in registry["registered_models"]:
            path = Path(registry["results_root"]) / (entry["model_key"] + ".launch.json")
            try:
                record = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if record["status"] == "failed":
                raise RuntimeError("Registered child failed; outcome barrier remains closed")
            completed += record["status"] == "complete"
        if completed == len(registry["registered_models"]):
            break
        time.sleep(15)
    audit = subprocess.run(
        [
            PYTHON,
            str(Path(__file__).with_name("integrity.py")),
            "--registry",
            str(registry_path),
            "--sidecar",
            str(ART / "live/integrity_sidecar.json"),
            "--out",
            str(ART / "live/integrity_results.json"),
        ],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "ND_OFFLINE": "1"},
    )
    if audit.returncode:
        raise RuntimeError("Preaggregation integrity audit failed")
    result = subprocess.run(
        [
            PYTHON,
            str(Path(__file__).with_name("aggregate.py")),
            "--registry",
            str(registry_path),
            "--out",
            str(ART / "live/summary.json"),
        ],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "ND_OFFLINE": "1"},
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
