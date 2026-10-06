"""Archive descriptive feature-selection misses from already evaluated frozen pools."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import analyze_candidates as analysis

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan"
sys.path.insert(0, str(SOURCE))
from state_env import Env  # noqa: E402
from tokenizer import make_tokenizer  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def keyed(path):
    rows = analysis.load_rows(path)
    result = {row["theorem_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate theorem IDs in {path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("public", "private", "theorem-metrics", "feature-model", "out"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    public = keyed(args.public)
    private = keyed(args.private)
    metrics = keyed(args.theorem_metrics)
    if set(public) != set(private) or set(public) != set(metrics):
        raise ValueError("Input theorem inventories differ")
    model = json.loads(Path(args.feature_model).read_text())
    tok = make_tokenizer("lean_staten")
    cases = []
    for tid in sorted(metrics):
        metric = metrics[tid]
        expected = metric["accuracy"]["learned_features"]
        if not metric["n_valid"] or expected == 1:
            continue
        row = public[tid]
        labels = {label["candidate_id"]: label for label in private[tid]["labels"]}
        candidates = row["candidates"]
        if len(labels) != len(private[tid]["labels"]):
            raise ValueError("Duplicate private candidate IDs")
        ids = [candidate["candidate_id"] for candidate in candidates]
        if len(set(ids)) != len(ids) or set(ids) != set(labels):
            raise ValueError("Candidate inventories differ")
        scores = analysis.feature_scores(
            [analysis.public_features(row, candidate, tok, Env) for candidate in candidates],
            model,
        )
        selected = [cid for cid, score in zip(ids, scores, strict=True) if score == max(scores)]
        actual = sum(labels[cid]["valid"] for cid in selected) / len(selected)
        if abs(actual - expected) > 1e-12:
            raise ValueError("Reconstructed feature selection differs from registered result")
        valid = sorted(cid for cid, label in labels.items() if label["valid"])
        if len(valid) != metric["n_valid"]:
            raise ValueError("Validity inventory differs from registered result")
        examples = sorted(set(selected + valid[:1]))
        cases.append(
            {
                "theorem_id": tid,
                "family": row["family"],
                "prompt": row["prompt"],
                "registered_theorem_metrics": metric,
                "selected_ids": selected,
                "valid_ids": valid,
                "candidates": [
                    {
                        "candidate_id": cid,
                        "feature_logit": score,
                        "valid": labels[cid]["valid"],
                        "parse_complete": labels[cid]["parse_complete"],
                    }
                    for cid, score in zip(ids, scores, strict=True)
                ],
                "illustrative_checked_text": [
                    {
                        "candidate_id": cid,
                        "valid": labels[cid]["valid"],
                        "lean_text": labels[cid].get("lean_text"),
                    }
                    for cid in examples
                ],
            }
        )
    inputs = [args.public, args.private, args.theorem_metrics, args.feature_model]
    result = {
        "scope": "Post-hoc descriptive inspection; no model fit, policy change or new inference",
        "case_rule": "All pools with at least one valid proof and feature accuracy below one",
        "example_rule": "Every selected candidate plus the lowest-ID valid candidate per case",
        "case_count": len(cases),
        "cases": cases,
        "input_sha256": {str(path): sha(path) for path in inputs},
        "source_sha256": {
            str(Path(__file__).resolve()): sha(__file__),
            str(Path(analysis.__file__).resolve()): sha(analysis.__file__),
            str(SOURCE / "state_env.py"): sha(SOURCE / "state_env.py"),
        },
        "limitation": "Outcome-selected examples do not estimate error-type prevalence",
    }
    with Path(args.out).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"case_count": len(cases), "out": args.out}))


if __name__ == "__main__":
    main()
