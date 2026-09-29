"""Summarize only this sprint's recorded GPU leases; never queries processes."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SPRINT = Path(__file__).resolve().parents[1]


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def collect(now=None):
    now = now or datetime.now(timezone.utc)
    jobs, intervals, identities = [], [], {}
    for directory in sorted((SPRINT / "jobs").iterdir()):
        event_path = directory / "events.jsonl"
        if not event_path.is_file():
            continue
        raw = event_path.read_bytes()
        identities[str(event_path)] = hashlib.sha256(raw).hexdigest()
        events = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        if not events:
            continue
        acquired = [event for event in events if event["event"] == "acquired"]
        if len(acquired) > 1:
            raise ValueError(f"Multiple acquisitions in one job: {directory.name}")
        completed = [event for event in events if event["event"] == "completed"]
        failed = [event for event in events if event["event"] == "failed"]
        terminal = completed[-1] if completed else failed[-1] if failed else None
        state = "completed" if completed and completed[-1].get("exit_code") == 0 else "failed" if terminal else "waiting"
        record = {"job": directory.name, "state": state, "acquired_gpu": bool(acquired),
                  "gpu_lease_seconds": 0.0, "first_event_utc": events[0]["utc"]}
        if acquired:
            start = timestamp(acquired[0]["utc"])
            end = timestamp(terminal["utc"]) if terminal else now.timestamp()
            if end < start:
                raise ValueError("Lease ended before acquisition")
            record.update(gpu=acquired[0]["gpu"]["index"], gpu_uuid=acquired[0]["gpu"]["uuid"],
                          acquired_utc=acquired[0]["utc"], terminal_utc=terminal["utc"] if terminal else None,
                          gpu_lease_seconds=end-start, state=state if terminal else "running",
                          worker_elapsed_seconds=completed[-1].get("elapsed_seconds") if completed else None,
                          exit_code=completed[-1].get("exit_code") if completed else None)
            intervals.append((start, end, directory.name))
        jobs.append(record)
    overlaps = []
    for index, left in enumerate(intervals):
        for right in intervals[index+1:]:
            seconds = min(left[1], right[1]) - max(left[0], right[0])
            if seconds > 0:
                overlaps.append({"jobs": [left[2], right[2]], "overlap_seconds": seconds})
    return {"utc": now.isoformat(), "jobs": jobs, "overlaps": overlaps,
            "recorded_single_gpu_at_a_time": not overlaps,
            "total_gpu_lease_seconds": sum(record["gpu_lease_seconds"] for record in jobs),
            "running_gpu_jobs": [record["job"] for record in jobs if record["state"] == "running"],
            "queued_jobs": [record["job"] for record in jobs if record["state"] == "waiting"],
            "event_prefix_sha256": identities, "non_atomic_read_only_snapshot": True,
            "scope": "Recorded local GPU leases only; excludes CPU, downloads, API usage and unrecorded external work",
            "paid_cloud_gpu_jobs_launched_by_sprint": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(SPRINT / "reports"):
        parser.error("Ledger output must be inside this sprint's reports")
    result = collect()
    with args.out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("jobs", "event_prefix_sha256")}))


if __name__ == "__main__":
    main()
