"""Account for final model-job process intervals without reading numerical outcomes."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    registries = {
        sha(ART / f"final_evaluation/{name}_protocol.json"): name for name in ("main", "historical")
    }
    parent_hash = sha(ART / "final_evaluation/decisions_parent.json")
    records = []

    def append(path, category, gpu, seconds, complete, **extra):
        records.append(
            {
                "launch_ledger": str(path.relative_to(ROOT)),
                "launch_ledger_sha256": sha(path),
                "category": category,
                "gpu": gpu,
                "model_process_wall_seconds": seconds,
                "complete": complete,
                **extra,
            }
        )

    for path in sorted((ART / "pools").glob("*_test.collection_launch.json")):
        item = read(path)
        if item["parent_sha256"] != parent_hash:
            raise ValueError("Independent collection uses another decision parent")
        phases = [p for p in item["processes"] if p["stage"] == "argv"]
        if len(phases) > 1:
            raise ValueError("Duplicated collection process")
        phase = phases[0] if phases else {}
        append(
            path,
            "independent_pool_collection",
            item["gpu"],
            phase.get("monotonic_process_wall_seconds"),
            phase.get("returncode") == 0,
            dataset=item["dataset"],
        )

    for path in sorted((ART / "allocation").glob("final_*/*.launch.json")):
        item = read(path)
        dataset = registries[item["registry_sha256"]]
        phases = [p for p in item["phases"] if p["name"] == "generation_and_checking"]
        if len(phases) > 1:
            raise ValueError("Duplicated live generation process")
        phase = phases[0] if phases else {}
        append(
            path,
            "live_path_evaluation",
            item["gpu"],
            phase.get("monotonic_process_wall_seconds"),
            phase.get("returncode") == 0,
            dataset=dataset,
        )

    for path in sorted((ART / "context/final_main_causal").glob("*.launch.json")):
        item = read(path)
        if registries[item["registry_sha256"]] != "main":
            raise ValueError("Causal process uses a non-MAIN registry")
        append(
            path,
            "causal_intervention",
            item["gpu"],
            item.get("process_wall_seconds"),
            item.get("returncode") == 0,
            dataset="main",
        )

    for path in sorted((ART / "selectors").rglob("run_manifest.json")):
        item = read(path)
        if item.get("registry_sha256") not in registries:
            continue
        append(
            path,
            "selector_inference",
            item["gpu"],
            item.get("process_wall_seconds_monotonic"),
            item.get("exit_code") == 0,
            dataset=registries[item["registry_sha256"]],
        )

    if len({r["launch_ledger"] for r in records}) != len(records):
        raise ValueError("Duplicate process ledger")
    expected_counts = {
        "independent_pool_collection": 2,
        "live_path_evaluation": 18,
        "causal_intervention": 8,
        "selector_inference": 74,
    }
    observed_counts = dict(Counter(row["category"] for row in records))
    incomplete = [
        row["launch_ledger"]
        for row in records
        if not row["complete"] or row["model_process_wall_seconds"] is None
    ]
    for row in records:
        seconds = row["model_process_wall_seconds"]
        if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
            raise ValueError(f"Invalid process interval in {row['launch_ledger']}")
    if args.require_complete and (observed_counts != expected_counts or incomplete):
        raise ValueError(
            f"Final accounting incomplete: observed counts {observed_counts}; "
            f"expected {expected_counts}; incomplete/unmetered {incomplete}"
        )
    totals = {}
    for row in records:
        if row["model_process_wall_seconds"] is not None:
            key = row["category"]
            totals[key] = totals.get(key, 0) + row["model_process_wall_seconds"]
    result = {
        "source_sha256": sha(Path(__file__)),
        "registry_hashes": registries,
        "runs": records,
        "category_process_wall_seconds": totals,
        "measured_process_wall_seconds": sum(totals.values()),
        "expected_category_counts": expected_counts,
        "observed_category_counts": observed_counts,
        "completeness_required": args.require_complete,
        "uncompleted_or_unmetered": incomplete,
        "scope": (
            "Whole model child-process intervals, including setup, CPU work, verification, "
            "and serialization. CPU-only subsequent statistics and queue-barrier waits excluded. "
            "Not kernel-active GPU time or end-to-end deployment latency. Earlier development "
            "runs are accounted for separately and must not be added again through queue totals."
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
