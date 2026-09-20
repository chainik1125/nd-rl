"""Posthoc descriptions of completed, strictly replayed textbook attempts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics

import analyze as a


def summarize(rows):
    reasons, rules, locations, finishes = Counter(), Counter(), Counter(), Counter()
    bodies = defaultdict(set)
    accepted = 0
    for row in rows:
        accepted += int(row["reward"])
        finishes[row.get("finish_reason", "unrecorded")] += 1
        bodies[row["index"]].add(row["body"])
        if row["reward"]:
            continue
        reason = row["reason"]
        reasons[reason] += 1
        if match := re.match(r"rule check failed: (\S+)", reason):
            rules[match[1]] += 1
        if match := re.search(r"\bline (\d+)\b", reason):
            locations[int(match[1])] += 1
    assert sum(reasons.values()) + accepted == len(rows)
    return {"attempts": len(rows), "accepted_attempts": accepted,
            "reported_finish_reasons": dict(finishes), "verifier_failure_messages": dict(reasons),
            "failed_rule_checks": dict(rules), "reported_failure_locations": dict(sorted(locations.items())),
            "failures_reported_at_or_before_line6": sum(n for line, n in locations.items() if line <= 6),
            "unique_generated_proof_bodies_by_target": {str(i): len(s) for i, s in sorted(bodies.items())},
            "median_unique_generated_proof_bodies_per_target": statistics.median(map(len, bodies.values()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    analysis = a.read(root / "reports/analysis_results.json")
    if analysis["training_protocol_sha256"] != a.sha(root / "plans/protocol.json"):
        raise ValueError("Diagnostic training protocol differs")
    output = {"kind": "posthoc_completed_arm_diagnostic", "analysis_results_sha256": a.sha(root / "reports/analysis_results.json"),
              "scope": "Descriptive saved-attempt diagnostics, not additional evaluations or training decisions. "
                       "The verifier reports a failure location; it does not identify all mistakes or the minimum repair. "
                       "Unique proof bodies measure observed text diversity, not semantic diversity or policy entropy.", "arms": {}}
    for name, arm in analysis["arms"].items():
        spec, _, _ = a.validate_arm(root, name, analysis["training_protocol_sha256"])
        source = root / spec["development_data"]["path"]
        if a.sha(source) != spec["development_data"]["sha256"]:
            raise ValueError("Original textbook identity differs")
        targets = [json.loads(line) for line in source.read_text().splitlines()]
        long_indices = {i for i, target in enumerate(targets) if a.trainer.reference_length(target) > 6}
        result = {}
        for phase in ("sft", "rl"):
            endpoint = arm["endpoints"][phase + "_sampled"]
            path = root / spec["run_base"] / "evaluations" / (phase + "_dev_sampled") / "attempts.jsonl"
            if a.sha(path) != endpoint["attempts_sha256"]:
                raise ValueError("Diagnostic attempts differ from strict replay")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) != endpoint["strict_replayed_attempts"]:
                raise ValueError("Diagnostic attempt count differs")
            long_rows = [row for row in rows if row["index"] in long_indices]
            assert len(long_rows) == len(long_indices) * 16
            result[phase] = {"checkpoint_sha256": endpoint["checkpoint_sha256"], "attempts_sha256": a.sha(path),
                             "all": summarize(rows), "reference_above6": summarize(long_rows)}
        output["arms"][name] = result
    a.write(root / "reports/failure_diagnostics.json", output)
    print(json.dumps({"completed_arms": list(output["arms"]), "output": str(root / "reports/failure_diagnostics.json")}))


if __name__ == "__main__":
    main()
