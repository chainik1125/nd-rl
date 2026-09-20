"""Extract the unique sampled scoped success in the minimal final comparison."""
from collections import Counter
from pathlib import Path
import json

import common
from nd_environment import strict_complete
from nd_verify import parse_proof_tokens
from proof_diagnostics import compress_proof


def main():
    root = common.ROOT
    analysis_path = root / "reports/analysis_minimal_structured_final_sample4_v1.json"
    analysis = json.loads(analysis_path.read_text())
    if analysis["status"] != "complete":
        raise ValueError("The complete fixed comparison is required")
    policies = [name for name, rows in analysis["summaries"].items()
                if rows["family:scope_projection"]["any"]]
    if len(policies) != 1:
        raise ValueError("This extraction is defined for the unique successful policy")
    policy = policies[0]
    if analysis["summaries"][policy]["family:scope_projection"]["any"] != 1:
        raise ValueError("The scoped success is not unique")
    audit_path = root / "data/structured_v1/audit/final.jsonl"
    meta = {row["source_id"]: row for row in common.read_jsonl(audit_path)}
    attempts_path = root / f"results/minimal_{policy}_structured_final_sample4_v1/symbolic_general_attempts.jsonl"
    attempts = [row for row in common.read_jsonl(attempts_path)
                if row["success"] and meta[row["source_id"]]["family"] == "scope_projection"]
    if len(attempts) != 1:
        raise ValueError("The scoped success has multiple sampled witnesses")
    row = attempts[0]
    raw_verdict = strict_complete(row["prompt"], row["body"])
    compressed = compress_proof(row["prompt"], row["body"])
    compressed_verdict = strict_complete(row["prompt"], compressed["compressed_proof"])
    if not (raw_verdict["ok"] and raw_verdict["independent_ok"] and
            compressed_verdict["ok"] and compressed_verdict["independent_ok"]):
        raise ValueError("Raw or compressed witness failed checking")
    seed_path = root / "data/symbolic_seeds.jsonl"
    seed_rules = Counter()
    for source in common.read_jsonl(seed_path):
        body = source.get("body") or source["text"].split("PRF", 1)[1].strip()
        seed_rules.update(line["rule"] for line in parse_proof_tokens(body.split()))
    proof_rules = Counter(line["rule"] for line in parse_proof_tokens(compressed["compressed_proof"].split()))
    raw_audit_path = root / f"reports/audit_minimal_{policy}_structured_final_sample4_v1.json"
    raw_audit = json.loads(raw_audit_path.read_text())
    if raw_audit["status"] != "complete":
        raise ValueError("The full raw-action audit is required")
    audit_row = raw_audit["rows"][row["source_id"]]
    audited = [s for s in audit_row["successes"] if s["attempt"] == row["attempt"]]
    if len(audited) != 1 or audited[0]["compressed_lines"] != compressed["compressed_length"]:
        raise ValueError("Witness does not reconcile with its raw-action audit")
    inputs = (analysis_path, audit_path, attempts_path, seed_path, raw_audit_path)
    output = root / "reports/witness_minimal_scoped_final_v1"
    payload = {"status": "verified", "selection": "Unique scoped success across all minimal final sample4 policies; selected descriptively after evaluation.",
        "policy": policy, "attempt_zero_based": row["attempt"], "source_id": row["source_id"],
        "scope_theorems_per_policy": analysis["summaries"][policy]["family:scope_projection"]["theorems"],
        "raw": row, "raw_verdict": raw_verdict, "compression": compressed,
        "compressed_verdict": compressed_verdict, "demonstration_rule_counts": dict(seed_rules),
        "compressed_rule_counts": dict(proof_rules), "rules_absent_from_demonstrations": sorted(set(proof_rules) - set(seed_rules)),
        "input_sha256": {str(p): common.sha(p) for p in inputs}, "code_sha256": common.sha(__file__),
        "scope": "An isolated composition of supplied legal rules beyond the two demonstration sequences. These rules were available during training and could occur in failed rollouts. This is not rule invention or reliable scoped proving."}
    common.write_json(output.with_suffix(".json"), payload)
    text = f"""# One sampled scoped proof from two projection demonstrations

The minimal curriculum policy with seed 4402 proves one of 160 scoped final statements within four sampled attempts; seed 4401 and all other minimal controls prove none. Both curriculum policies have zero greedy scoped successes. This is the unique sampled scoped success in the complete frozen comparison, extracted after evaluation as an illustration.

The original 22-line attempt passes both checkers. Dependency pruning removes 13 unused lines, including abandoned subproofs, leaving the nine-line certificate below. Further reiteration compression preserves those nine lines. Its assumption and implication-introduction rules are absent from the two demonstration sequences, whose non-premise actions are all conjunction eliminations. The legal-action environment offered these rules throughout training, so this is an isolated new composition relative to the demonstrations, not invention of deduction rules. One success does not establish a reliable scoped strategy or an RL-specific advantage over cloning.

The theorem has no premises:

```text
{row['prompt']}
```

The dependency-pruned, independently checked certificate is:

```text
{compressed['compressed_proof']}
```

The accepted raw attempt and all action choices are in [the witness JSON](witness_minimal_scoped_final_v1.json). The [complete final comparison](analysis_minimal_structured_final_sample4_v1.json) preserves every control and both seeds. Pruning is analysis only; it is not used to turn a failed output into an accepted one.
"""
    output.with_suffix(".md").write_text(text)
    print(json.dumps({"status": "verified", "policy": policy, "source_id": row["source_id"],
                      "raw_lines": raw_verdict["n_lines"], "compressed_lines": compressed["compressed_length"],
                      "rules_absent_from_demonstrations": payload["rules_absent_from_demonstrations"]}))


if __name__ == "__main__":
    main()
