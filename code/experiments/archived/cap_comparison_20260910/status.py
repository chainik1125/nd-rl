"""Compact operational status from training events; never reads textbook payloads."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text())


def latest(path):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        rows = stream.read().splitlines()
    for row in reversed(rows):
        try:
            return json.loads(row)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--brief", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    state = read(root / "reports/recovery_state.json")
    now = datetime.now(timezone.utc)
    output = {"utc": now.isoformat(),
              "coordinator_age_seconds": round((now - datetime.fromisoformat(state["updated_utc"])).total_seconds()),
              "completed": [name for name, result in state["finished"].items() if result["success"]],
              "failed": [name for name, result in state["finished"].items() if not result["success"]],
              "pending": state["pending"], "running": {}}
    for name, assignment in state["running"].items():
        base = root / "runs" / name
        detail = {"host": assignment["host"], "gpu": assignment["gpu"]}
        if (base / "progress.json").exists():
            progress = read(base / "progress.json")
            detail["stage"] = progress["current_stage"]
        phase = "rl" if (base / "rl/events.jsonl").exists() else "sft"
        path = base / phase / "events.jsonl"
        if path.exists() and (event := latest(path)):
            target = 14663886875197440 if phase == "rl" else 25807392746127360
            fraction = event["total_flop_proxy"] / target
            detail.update({"training_phase": phase, "step": event["step"],
                           "budget_percent": round(fraction * 100, 1),
                           "event_age_seconds": round(time.time() - event["unix_time"])})
            if detail.get("stage") == phase and fraction > 0:
                detail["linear_training_minutes_remaining"] = round(max(0, 1 / fraction - 1) * event["elapsed_seconds"] / 60, 1)
        output["running"][name] = detail
    finalization = root / "reports/finalization_state.json"
    if finalization.exists():
        output["strict_replayed"] = read(finalization)["completed_arms"]
    if args.brief:
        print(output["utc"], f"coordinator_age={output['coordinator_age_seconds']}s",
              f"complete={len(output['completed'])}/6", f"failed={output['failed']}", f"pending={output['pending']}")
        for name, detail in output["running"].items():
            print(name, detail.get("stage", "dispatch"), f"{detail.get('budget_percent', '?')}%",
                  f"event_age={detail.get('event_age_seconds', '?')}s",
                  f"training_minutes_remaining≈{detail.get('linear_training_minutes_remaining', '?')}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
