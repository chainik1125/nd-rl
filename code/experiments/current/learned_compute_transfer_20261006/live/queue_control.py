"""Wait for a registered primary child, then launch its supplementary control."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_transfer_20261006"
PYTHON = "/data/users/dmitry/venvs/proof-state/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    while True:
        ledger = ART / "live/results" / (args.after + ".launch.json")
        try:
            record = json.loads(ledger.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(15)
            continue
        if record["status"] == "failed":
            raise RuntimeError("Primary child failed; supplementary launch suspended")
        if record["status"] == "complete":
            break
        time.sleep(15)
    command = [
        PYTHON,
        str(Path(__file__).with_name("launch.py")),
        "--registry",
        str(ART / "live/protocol.json"),
        "--gpu",
        str(args.gpu),
        "--models",
        args.model,
    ]
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, "ND_OFFLINE": "1"})
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
