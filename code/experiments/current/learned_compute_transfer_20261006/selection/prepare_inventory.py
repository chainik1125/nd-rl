"""Prepare inherited phase1 inference commands without loading transfer outcomes."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_transfer_20261006"
OLD = ROOT / "code/experiments/current/learned_compute_allocation_20261006"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    freeze = ART / "phase1_decisions.json"
    plan = json.loads(freeze.read_text())
    commands = []
    for entry in plan["registered_analyses"]:
        sig = entry["signature"]
        label = entry["label"].removeprefix("main_")
        cmd = [
            "/data/users/dmitry/venvs/proof-state/bin/python",
            str(OLD / "selection/evaluate.py"),
            "--checkpoint",
            entry["checkpoint"],
            "--public",
            str(ART / "pools/combined412/public_transitions.jsonl"),
            "--private",
            str(ART / "pools/combined412/private.jsonl"),
            "--split",
            "test",
            "--out",
            str(ART / "selectors" / label),
            "--context",
            "8192",
            "--view",
            "transitions",
            "--architecture",
            sig["architecture"],
            "--objective",
            sig["objective"],
            "--evaluation-seed",
            "77106",
            "--frozen-test-protocol",
            str(ART / "phase1_protocol.json"),
        ]
        commands.append(
            {
                "label": label,
                "signature": sig,
                "signature_sha256": entry["signature_sha256"],
                "argv": cmd,
            }
        )
    historical = json.loads(
        (
            ROOT
            / "artifacts/learned_compute_allocation_20261006/analysis"
            / "historical_cpu_commands_v1.json"
        ).read_text()
    )["commands"][0]
    cmd = historical["command"].copy()
    for flag, path in [
        ("--public", ART / "pools/combined412/public.jsonl"),
        ("--private", ART / "pools/combined412/private.jsonl"),
        ("--frozen-test-protocol", ART / "phase1_protocol.json"),
        ("--out", ART / "analysis/features"),
    ]:
        cmd[cmd.index(flag) + 1] = str(path)
    commands.append(
        {
            "label": "features",
            "signature": historical["signature"],
            "signature_sha256": hashlib.sha256(
                json.dumps(historical["signature"], sort_keys=True).encode()
            ).hexdigest(),
            "argv": cmd,
        }
    )
    inventory = {
        "status": "frozen_before_inference",
        "decisions_sha256": sha(freeze),
        "commands": commands,
        "data_sha256": plan["data_sha256"],
        "scope": "12 frozen selector evaluations plus frozen feature/heuristic analysis; no fit",
    }
    path = ART / "phase1_inventory.json"
    assert not path.exists()
    path.write_text(json.dumps(inventory, indent=2) + "\n")
    print(sha(path))


if __name__ == "__main__":
    main()
