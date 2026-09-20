"""Audit completed symbolic runs without changing frozen training programs."""
import argparse
from collections import Counter
import json
from pathlib import Path

import common
from nd_environment import strict_complete
from nd_verify import parse_proof_tokens
from symbolic_general_env import Environment, Action


def immutable_tuple(value):
    return tuple(immutable_tuple(x) for x in value) if isinstance(value, list) else value


def proof_stats(prompt, body):
    verdict = strict_complete(prompt, body)
    assert verdict["ok"]
    lines = parse_proof_tokens(body.split())
    rules = Counter(line["rule"] for line in lines)
    return {"n_lines": verdict["n_lines"], "max_scope": max(line["depth"] for line in lines),
            "rule_counts": dict(rules), "uses_impi": bool(rules["IMPI"]), "uses_ore": bool(rules["ORE"]),
            "uses_negi": bool(rules["NEGI"]), "verdict": verdict}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--kind", choices=["train", "eval"], required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run, out = Path(args.run), Path(args.out)
    examples, first_targets, rule_occurrences = [], {}, Counter()
    counts, by_family = Counter(), {}
    if args.kind == "train":
        source_path = run / "symbolic_general_successes.jsonl"
        rows = common.read_jsonl(source_path) if source_path.exists() else []
        sources = {x["source_id"]: x for x in common.read_jsonl(common.ROOT / "data/v1/sources.jsonl")}
        targets = {x["source_id"]: x for x in common.read_jsonl(common.ROOT / "data/v1/targets.jsonl")}
        max_lines, max_scope = 24, 4
    else:
        source_path = run / "symbolic_general_attempts.jsonl"
        rows = common.read_jsonl(source_path)
        config = json.loads((run / "symbolic_general_eval_config.json").read_text())
        max_lines, max_scope = config["max_lines"], config["max_scope"]
    for row in rows:
        counts["rows"] += 1
        env = Environment(row["prompt"], max_lines=max_lines, max_scope=max_scope)
        if args.kind == "train":
            bank = sources if row["is_source"] else targets
            assert row["source_id"] in bank and row["prompt"] == bank[row["source_id"]]["prompt"]
            if not row["is_source"]:
                assert row["cut"] == 0
            elif row["cut"]:
                assert row["cut"] < len(env.source_actions(bank[row["source_id"]]))
            for action in env.source_actions(row):
                env.step(action)
            success = True
        else:
            for raw in row["action_trace"]:
                aa = env.actions()
                action = Action(raw["rule"], immutable_tuple(raw["formula"]), tuple(raw["refs"]), raw["depth"])
                assert len(aa) == raw["candidate_count"] and aa[raw["selected_index"]] == action
                env.step(action, check_membership=False)
            success = row["success"]
            assert env.success() == success
            assert env.body() == row["body"]
            key = row["family"]
            by_family.setdefault(key, Counter())
            by_family[key]["attempts"] += 1
            by_family[key]["successes"] += int(success)
        full = strict_complete(row["prompt"], row["body"])
        assert full["ok"] == success
        if not success:
            continue
        counts["strict_successes_reverified"] += 1
        pruned = env.pruned_body()
        stats = proof_stats(row["prompt"], pruned)
        if args.kind == "eval":
            assert pruned == row["pruned_body"]
        counts["pruned_over_six"] += int(stats["n_lines"] > 6)
        counts["pruned_at_least_13"] += int(stats["n_lines"] >= 13)
        counts["successful_proofs_using_impi"] += int(stats["uses_impi"])
        counts["successful_proofs_using_ore"] += int(stats["uses_ore"])
        counts["successful_proofs_using_negi"] += int(stats["uses_negi"])
        if args.kind == "train":
            role = "source" if row["is_source"] else "target"
            counts[f"{role}_successful_attempts"] += 1
            counts[f"{role}_successful_attempts_pruned_over_six"] += int(stats["n_lines"] > 6)
            counts[f"{role}_successful_attempts_using_ore"] += int(stats["uses_ore"])
        rule_occurrences.update(stats["rule_counts"])
        record = {"source_id": row["source_id"], "prompt": row["prompt"], "body": row["body"],
                  "pruned_body": pruned, "full_lines": full["n_lines"], "pruned_stats": stats,
                  "is_source": row.get("is_source")}
        if args.kind == "train" and not row["is_source"] and row["source_id"] not in first_targets:
            record["first_discovery_actions"] = row["actions_so_far"]
            first_targets[row["source_id"]] = record
        if stats["n_lines"] > 6 or stats["uses_ore"] or stats["max_scope"] >= 2:
            examples.append(record)
    examples = sorted(examples, key=lambda r: (-r["pruned_stats"]["n_lines"], -r["pruned_stats"]["max_scope"], r["source_id"]))
    # Exact duplicates are removed only from example output, not counted outcomes.
    unique_examples = list({common.digest([e["prompt"], e["pruned_body"]]): e for e in examples}.values())
    common.write_json(out, {"run": str(run), "kind": args.kind, "counts": dict(counts),
        "by_family": {k: dict(v) for k, v in by_family.items()}, "pruned_rule_occurrences": dict(rule_occurrences),
        "unique_training_target_first_proofs": len(first_targets), "target_first_proofs": list(first_targets.values()),
        "unique_target_first_proofs_pruned_over_six": sum(r["pruned_stats"]["n_lines"] > 6 for r in first_targets.values()),
        "examples": unique_examples[:32], "input_sha256": common.sha(source_path) if source_path.exists() else None,
        "audit_code_sha256": common.sha(__file__)})
    print(json.dumps({"counts": dict(counts), "unique_target_first_proofs": len(first_targets),
                      "examples": len(unique_examples), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
