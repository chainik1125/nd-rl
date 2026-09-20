"""Check whether existing nonlearned certificates fit the symbolic action space.

This is an analysis-only constructive reachability check. Solver certificates
never enter training. Failure to replay this particular certificate does not
prove that the theorem is unreachable by a different certificate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import common
from nd_environment import strict_complete
from symbolic_general_env import Environment


def analyze(split):
    cohorts = (("general_primary", "v1", "primary", 24),
               ("general_structured", "structured_lm_v1", "structured", 24),
               ("minimal_structured", "structured_v1", "structured", 32))
    results, inputs = {}, {}
    for name, dataset, solver_population, max_lines in cohorts:
        population_path = common.ROOT / "data" / dataset / (split + ".jsonl")
        population = {r["source_id"]: r for r in common.read_jsonl(population_path)}
        solver = common.ROOT / "results" / f"classical_{solver_population}_{split}_v1"
        result = json.loads((solver / "result.json").read_text())
        certificates = solver / "certificates_NEVER_TRAIN.jsonl"
        if result["status"] != "complete" or common.sha(certificates) != result["certificates_sha256"]:
            raise ValueError("Incomplete or changed nonlearned certificate archive")
        attempt_path = solver / "attempts.jsonl"
        if common.sha(attempt_path) != result["attempts_sha256"]:
            raise ValueError("Nonlearned attempt archive changed")
        selected_attempts = [row for row in common.read_jsonl(attempt_path)
                             if row["method"] == "heuristic_4096" and row["source_id"] in population]
        attempts = {row["source_id"]: row for row in selected_attempts}
        if len(attempts) != len(selected_attempts) or set(attempts) != set(population):
            raise ValueError("Missing or duplicated fixed-heuristic attempt")
        rows = {}
        for certificate in common.read_jsonl(certificates):
            sid = certificate["source_id"]
            if certificate["method"] != "heuristic_4096" or sid not in population:
                continue
            if certificate["prompt"] != population[sid]["prompt"]:
                raise ValueError("Solver certificate/population mismatch")
            prompt, body = certificate["prompt"], certificate["pruned_certificate"]
            if not strict_complete(prompt, body)["ok"]:
                raise ValueError("Previously verified solver certificate is invalid")
            env = Environment(prompt, max_lines=max_lines, max_scope=4)
            failure = None
            for position, action in enumerate(env.source_actions({"prompt": prompt, "body": body}), 1):
                if env.success():
                    break
                available = env.actions()
                if action not in available:
                    failure = {"reference_inference": position, "rule": action.rule, "formula": action.formula,
                               "refs": action.refs, "depth": action.depth, "candidate_count": len(available),
                               "at_line_limit": len(env.lines) >= max_lines}
                    break
                env.step(action)
            reachable = env.success()
            if reachable and not strict_complete(prompt, env.body())["ok"]:
                raise ValueError("Constructive reachable prefix failed verification")
            rows[sid] = {"reachable_by_this_certificate": reachable, "failed_replay": failure,
                         "solver_certificate_lines": body.count(";"), "constructed_lines": len(env.lines) if reachable else None}
        for sid in set(population) - set(rows):
            if attempts[sid]["reward"]:
                raise ValueError("Successful heuristic attempt lacks its certificate")
            rows[sid] = {"reachable_by_this_certificate": False, "failed_replay": None,
                         "solver_certificate_lines": None, "constructed_lines": None,
                         "solver_status_without_certificate": attempts[sid]["status"]}
        results[name] = {"theorems": len(rows), "constructively_reachable": sum(r["reachable_by_this_certificate"] for r in rows.values()),
            "solver_certificates_available": sum(r["solver_certificate_lines"] is not None for r in rows.values()),
            "solver_missing_certificates": sum(r["solver_certificate_lines"] is None for r in rows.values()),
            "max_lines": max_lines, "max_scope": 4,
            "failed_replay_rules": dict(Counter(r["failed_replay"]["rule"] for r in rows.values() if r["failed_replay"])),
            "constructed_line_histogram": dict(Counter(r["constructed_lines"] for r in rows.values() if r["reachable_by_this_certificate"])),
            "rows": rows}
        for path in (population_path, certificates, attempt_path, solver / "result.json"):
            inputs[str(path)] = common.sha(path)
    return {"status": "complete", "split": split, "cohorts": results, "input_sha256": inputs,
            "code_sha256": common.sha(__file__), "environment_sha256": common.sha(Path(__file__).with_name("symbolic_general_env.py")),
            "interpretation": __doc__}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "final"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.split)
    common.write_json(args.output, result)
    print(json.dumps({"status": "complete", "cohorts": {k: {a: b for a, b in v.items() if a != "rows"} for k, v in result["cohorts"].items()}}))
