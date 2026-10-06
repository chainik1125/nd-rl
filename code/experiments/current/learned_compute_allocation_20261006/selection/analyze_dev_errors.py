"""Descriptive error strata for all frozen primary choices; no model selection."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("artifacts/learned_compute_allocation_20261006")
OUT = ROOT / "selectors/primary_dev_error_audit"
OUT.mkdir(exist_ok=True)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


public_path = ROOT / "pools/main_dev/public_transitions.jsonl"
private_path = ROOT / "pools/main_dev/private.jsonl"
feature_path = ROOT / "analysis/features1600_main_dev/per_theorem.jsonl"
public = {r["theorem_id"]: r for r in read_rows(public_path)}
private = {r["theorem_id"]: r for r in read_rows(private_path)}
features = {r["theorem_id"]: r for r in read_rows(feature_path)}
choices_path = ROOT / "selectors/frozen_primary_dev_choices.json"
choices = json.loads(choices_path.read_text())["runs"]
assert set(public) == set(private) == set(features)
summary = []
details = []
inputs = [public_path, private_path, feature_path, choices_path]
for choice in choices:
    predictions_path = Path(choice["dev_predictions"])
    inputs.append(predictions_path)
    predictions = json.loads(predictions_path.read_text())
    assert len(predictions) == len(public)
    groups = defaultdict(list)
    overlap = defaultdict(float)
    for pred in predictions:
        tid = pred["theorem_id"]
        labels = {r["candidate_id"]: r for r in private[tid]["labels"]}
        assert sorted(labels) == list(range(8))
        ties = pred["argmax_tied_candidate_ids"]
        score = sum(labels[i]["valid"] for i in ties) / len(ties)
        feature = features[tid]["accuracy"]["learned_features"]
        n_valid = sum(r["valid"] for r in labels.values())
        n_complete = sum(r["parse_complete"] for r in labels.values())
        input_tokens = pred["input_tokens"]
        length_bin = (
            "<=1024"
            if input_tokens <= 1024
            else "1025-2048"
            if input_tokens <= 2048
            else "2049-4096"
            if input_tokens <= 4096
            else "4097-8192"
            if input_tokens <= 8192
            else ">8192"
        )
        record = {
            "run": choice["run"],
            "theorem_id": tid,
            "family": public[tid]["family"],
            "n_valid": n_valid,
            "n_complete": n_complete,
            "input_tokens": input_tokens,
            "max_forward_tokens": pred["max_forward_tokens"],
            "overflow": pred["overflow"],
            "tie_ids": ties,
            "accuracy_exact_ties": score,
            "feature_accuracy": feature,
        }
        details.append(record)
        for key, value in [
            ("all", "all"),
            ("family", record["family"]),
            ("n_valid", n_valid),
            ("n_complete", n_complete),
            ("family_by_n_valid", f"{record['family']}:{n_valid}"),
            ("mixed_validity", 0 < n_valid < 8),
            ("all_complete_mixed_validity", n_complete == 8 and 0 < n_valid < 8),
            ("input_length", length_bin),
            ("overflow", pred["overflow"]),
        ]:
            groups[(key, str(value))].append((1.0, score, feature))
        for candidate_id in ties:
            weight = 1 / len(ties)
            complete = labels[candidate_id]["parse_complete"]
            position = pred["candidate_order"].index(candidate_id)
            groups[("selected_complete", str(complete))].append(
                (weight, float(labels[candidate_id]["valid"]), feature)
            )
            groups[("selected_shuffled_position", str(position))].append(
                (weight, float(labels[candidate_id]["valid"]), feature)
            )
        overlap["both_correct"] += score * feature
        overlap["model_only_correct"] += score * (1 - feature)
        overlap["feature_only_correct"] += (1 - score) * feature
        overlap["both_wrong"] += (1 - score) * (1 - feature)
    strata = []
    for (axis, value), rows in sorted(groups.items()):
        count = sum(w for w, _, _ in rows)
        strata.append(
            {
                "axis": axis,
                "value": value,
                "theorem_mass": count,
                "accuracy": sum(w * a for w, a, _ in rows) / count,
                "feature_accuracy": sum(w * a for w, _, a in rows) / count,
            }
        )
    summary.append(
        {
            "run": choice["run"],
            "checkpoint_sha256": choice["checkpoint_sha256"],
            "strata": strata,
            "feature_overlap_fractional_theorems": dict(overlap),
        }
    )
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(OUT / "per_theorem.jsonl").write_text("".join(json.dumps(r) + "\n" for r in details))
(OUT / "config.json").write_text(
    json.dumps(
        {
            "scope": (
                "All six frozen primary choices, main DEV only; descriptive, no subset selection"
            ),
            "ties": (
                "Exact uniform tie expectation; completion and position strata "
                "use fractional tie mass"
            ),
            "overlap": (
                "Product of separate randomized tie expectations, not paired sampled tie choices"
            ),
            "length": (
                "Actual model input token sum; factorized sum is not "
                "single-candidate context length"
            ),
            "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        },
        indent=2,
    )
    + "\n"
)
(OUT / "source.py").write_text(Path(__file__).read_text())
print(
    json.dumps(
        [{"run": r["run"], "overlap": r["feature_overlap_fractional_theorems"]} for r in summary]
    )
)
