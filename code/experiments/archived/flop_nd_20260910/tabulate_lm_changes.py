"""Describe gained and lost final solves at each frozen inference allowance.

This is report-only analysis of completed, already-audited evaluations. It does
not run a model, choose an endpoint, or treat a baseline failure as incapability
under an unlimited sampling budget.
"""
import argparse
import json
from pathlib import Path

import common


def compare(report, metric):
    baseline = report["per_theorem"]["baseline"]
    ids = set(baseline)
    output = {}
    for name, rows in report["per_theorem"].items():
        if set(rows) != ids:
            raise ValueError("Endpoints do not cover the same theorem population")
        cells = {key: [] for key in ("both", "gained", "lost", "neither")}
        for sid in sorted(ids):
            before, after = bool(baseline[sid][metric]), bool(rows[sid][metric])
            key = "both" if before and after else "gained" if after else "lost" if before else "neither"
            cells[key].append(sid)
        counts = {key: len(value) for key, value in cells.items()}
        initial_successes = report["summaries"]["baseline"]["all"][metric]
        endpoint_successes = report["summaries"][name]["all"][metric]
        if sum(counts.values()) != len(ids):
            raise ValueError("Incomplete comparison partition")
        if counts["both"] + counts["lost"] != initial_successes:
            raise ValueError("Initial success count does not reconcile")
        if counts["both"] + counts["gained"] != endpoint_successes:
            raise ValueError("Endpoint success count does not reconcile")
        if counts["gained"] - counts["lost"] != endpoint_successes - initial_successes:
            raise ValueError("Gains and losses do not reconcile with the total effect")
        output[name] = {
            "theorems": len(ids), "initial_successes": initial_successes,
            "endpoint_successes": endpoint_successes, **counts,
            "net_change": counts["gained"] - counts["lost"],
            "gained_with_any_pruned_certificate_over6": sum(rows[sid]["any_pruned_over6"] for sid in cells["gained"]),
            "gained_with_any_compressed_certificate_over6": sum(rows[sid]["any_compressed_over6"] for sid in cells["gained"]),
            "gained_source_ids": cells["gained"], "lost_source_ids": cells["lost"],
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path,
                        default=common.ROOT / "reports/lm_gained_lost_final_v1")
    args = parser.parse_args()
    results, inputs = {}, {}
    lines = ["# Newly solved and lost statements at the frozen evaluation allowance", "",
        "This descriptive analysis was added after the first final outputs were available. It uses all frozen LM endpoints and all four complete, audited final comparisons. A gained statement is one the endpoint proves while initialization fails under the same evaluation profile. This does not establish that initialization could never prove it with more attempts, or that RL alone caused the gain; continued-supervision controls are included for that reason.", "",
        "Greedy compares one deterministic attempt. Sample4 compares whether any of four temperature-0.8 attempts succeeds. The two profiles are analyzed separately. Proof lengths count certificate lines after the stated checking/compression operation and do not certify a minimum proof length.", ""]
    for population in ("primary", "structured"):
        for profile in ("greedy", "sample4"):
            path = common.ROOT / f"reports/analysis_{population}_final_{profile}_v1.json"
            report = json.loads(path.read_text())
            if report.get("status") != "completed":
                raise ValueError(f"Analysis is incomplete: {path}")
            metric = "first" if profile == "greedy" else "any"
            key = f"{population}_{profile}"
            results[key] = {"metric": metric, "profile": report["profile"],
                            "endpoints": compare(report, metric)}
            inputs[str(path.resolve())] = common.sha(path)
            n = next(iter(results[key]["endpoints"].values()))["theorems"]
            lines += [f"## {population.capitalize()}, {profile}: {n} final statements", "",
                      "| Endpoint | Solved | Gained | Lost | Net change | Gained with a compressed certificate >6 lines |",
                      "|---|---:|---:|---:|---:|---:|"]
            for name, row in results[key]["endpoints"].items():
                lines.append(f"| {name} | {row['endpoint_successes']} | {row['gained']} | {row['lost']} | {row['net_change']:+d} | {row['gained_with_any_compressed_certificate_over6']} |")
            lines += [""]
    payload = {"status": "completed", "comparisons": results, "input_sha256": inputs,
               "code_sha256": common.sha(__file__),
               "scope": "Post-final descriptive partition of fixed-allowance outcomes; no model runs or endpoint selection. All counts reconcile to the complete audited analyses."}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(args.output_prefix.with_suffix(".json"), payload)
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "completed", "comparisons": len(results),
                      "endpoint_rows": sum(len(v["endpoints"]) for v in results.values()),
                      "output": str(args.output_prefix)}))


if __name__ == "__main__":
    main()
