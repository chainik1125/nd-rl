"""Theorem-paired DEV generation retention; candidate IDs are not common RNG paths."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "retention_bootstrap", Path(__file__).with_name("aggregate_selection.py")
)
statistics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(statistics)
root = Path("artifacts/learned_compute_allocation_20261006")
base = root / "context/base_main_dev64_v1"
critic = root / "selectors/bce_s0_e4_clean_main_dev64_v1"
panel = root / "context/main_dev64_panel/public.jsonl"
rows = {r["theorem_id"]: r for r in statistics.read_rows(panel)}
labels = []
public = []
for folder in [base, critic]:
    labels.append(
        {
            r["theorem_id"]: {c["candidate_id"]: c for c in r["labels"]}
            for r in statistics.read_rows(folder / "history_0/private.jsonl")
        }
    )
    public.append(
        {
            r["theorem_id"]: {c["candidate_id"]: c for c in r["candidates"]}
            for r in statistics.read_rows(folder / "history_0/public.jsonl")
        }
    )
assert set(rows) == set(labels[0]) == set(labels[1]) == set(public[0]) == set(public[1])
configs = [json.loads((folder / "config.json").read_text()) for folder in [base, critic]]
for key in [
    "public",
    "limit",
    "k",
    "batch",
    "seed",
    "temperature",
    "max_action",
    "max_steps",
    "context",
    "history_content",
    "precision",
    "input_sha256",
]:
    assert configs[0][key] == configs[1][key], key
assert configs[0]["counts"][0] == 0 and configs[1]["counts"] == [0]
source_paths = [folder / "source_snapshot/collect.py" for folder in [base, critic]]
ast_equal = ast.dump(ast.parse(source_paths[0].read_text())) == ast.dump(
    ast.parse(source_paths[1].read_text())
)
assert ast_equal, "Behavioral collector source changed; matching needs separate audit"
counts = {"both_valid": 0, "base_only_valid": 0, "critic_only_valid": 0, "both_invalid": 0}
records, differences, families = [], [], []
name_matches = 0
for tid in sorted(rows):
    assert sorted(labels[0][tid]) == sorted(labels[1][tid]) == [0, 1]
    original = [bool(labels[0][tid][i]["valid"]) for i in [0, 1]]
    changed = [bool(labels[1][tid][i]["valid"]) for i in [0, 1]]
    for candidate_id, old, new in zip([0, 1], original, changed, strict=True):
        key = (
            "both_valid"
            if old and new
            else "base_only_valid"
            if old
            else "critic_only_valid"
            if new
            else "both_invalid"
        )
        counts[key] += 1
        name_matches += (
            public[0][tid][candidate_id]["canonical_name_base"]
            == public[1][tid][candidate_id]["canonical_name_base"]
        )
    delta = (sum(changed) - sum(original)) / 2
    differences.append(delta)
    families.append(rows[tid]["family"])
    records.append(
        {
            "theorem_id": tid,
            "family": rows[tid]["family"],
            "base_valid": original,
            "critic_valid": changed,
            "paired_delta": delta,
        }
    )
result = {
    "theorems": len(rows),
    "attempts": 2 * len(rows),
    "candidate_pairs": counts,
    "canonical_name_bases_matched": name_matches,
    "collector_ast_equal": ast_equal,
    "base_accuracy": (counts["both_valid"] + counts["base_only_valid"]) / (2 * len(rows)),
    "critic_accuracy": (counts["both_valid"] + counts["critic_only_valid"]) / (2 * len(rows)),
    "paired_theorem_interval": statistics.stratified_interval(differences, families, 5000, 61006),
    "interpretation": "Same DEV prompts/configuration and nominal sampler seed. Pairing is by "
    "theorem/candidate ID, not bit-identical random paths; adaptive batching "
    "and trajectory lengths change subsequent sampling draws. No primary change.",
    "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [panel, *source_paths]},
    "families": {
        family: {
            "theorems": families.count(family),
            "paired_delta": sum(
                d for d, f in zip(differences, families, strict=True) if f == family
            )
            / families.count(family),
        }
        for family in sorted(set(families))
    },
}
(critic / "paired_retention_audit.json").write_text(json.dumps(result, indent=2) + "\n")
(critic / "paired_retention_per_theorem.jsonl").write_text(
    "".join(json.dumps(r) + "\n" for r in records)
)
(critic / "paired_retention_source.py").write_bytes(Path(__file__).read_bytes())
print(json.dumps(result))
