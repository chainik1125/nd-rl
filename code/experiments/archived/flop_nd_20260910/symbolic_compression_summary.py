"""Report further verified reiteration compression from every symbolic audit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "final"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, inputs = {}, {}
    for prefix, seeds in (("symbolic", (4301, 4302)), ("minimal", (4401, 4402))):
        policies = [f"{mode}_s{seed}" for mode in ("initial", "full", "reset", "bc") for seed in seeds] + ["uniform"]
        for population in ("primary", "structured"):
            for profile in ("greedy", "sample4"):
                for policy in policies:
                    name = f"{prefix}_{policy}_{population}_{args.split}_{profile}_v1"
                    path = common.ROOT / "reports" / ("audit_" + name + ".json")
                    report = json.loads(path.read_text())
                    if report.get("status") != "complete":
                        raise ValueError("Incomplete certificate audit")
                    inputs[str(path)] = common.sha(path)
                    groups = list(report["rows"].values())
                    witnesses = [s for row in groups for s in row["successes"]]
                    result[name] = {"theorems": len(groups), "successful_theorems": sum(r["any"] for r in groups),
                        "successful_attempts": len(witnesses),
                        "maximum_written": max((s["raw_lines"] for s in witnesses), default=0),
                        "maximum_dependency_pruned": max((s["pruned_lines"] for s in witnesses), default=0),
                        "maximum_further_compressed": max((s["compressed_lines"] for s in witnesses), default=0),
                        "any_compressed_over6": sum(any(s["compressed_lines"] > 6 for s in r["successes"]) for r in groups),
                        "shortest_found_compressed_over6": sum(bool(r["successes"]) and min(s["compressed_lines"] for s in r["successes"]) > 6 for r in groups),
                        "first_compressed_over6": sum(any(s["attempt"] == 0 and s["compressed_lines"] > 6 for s in r["successes"]) for r in groups),
                        "shortest_compressed_histogram": dict(Counter(min(s["compressed_lines"] for s in r["successes"]) for r in groups if r["successes"])),
                        "attempts_reduced_after_dependency_pruning": sum(s["compressed_lines"] < s["pruned_lines"] for s in witnesses)}
    common.write_json(args.output, {"status": "complete", "split": args.split, "records": result,
        "input_sha256": inputs, "code_sha256": common.sha(__file__),
        "notes": "All counts derive from raw reconstruction audits that rechecked each further compressed certificate. Reiteration compression is separate from dependency pruning. Neither establishes an unrestricted minimum proof length."})
    print(json.dumps({"status": "complete", "evaluations": len(result), "output": str(args.output)}))


if __name__ == "__main__":
    main()
