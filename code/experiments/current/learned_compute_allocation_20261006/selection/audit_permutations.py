"""Summarize all predeclared candidate orders without selecting a favorable order."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--factorized", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.00001)
    args = parser.parse_args()
    sets = [{r["theorem_id"]: r for r in json.loads(Path(p).read_text())} for p in args.predictions]
    assert len(sets) == 3
    assert all(set(s) == set(sets[0]) for s in sets)
    accuracy = [sum(r["argmax_tie_expected_accuracy"] for r in s.values()) / len(s) for s in sets]
    maxdiff = max(
        abs(a - b)
        for tid in sets[0]
        for s in sets[1:]
        for a, b in zip(sets[0][tid]["probabilities"], s[tid]["probabilities"], strict=True)
    )
    tie_accuracy_mismatches = [
        tid
        for tid in sets[0]
        if any(
            s[tid]["argmax_tie_expected_accuracy"] != sets[0][tid]["argmax_tie_expected_accuracy"]
            for s in sets[1:]
        )
    ]
    result = {
        "accuracy_all_orders": accuracy,
        "mean": sum(accuracy) / len(accuracy),
        "range": [min(accuracy), max(accuracy)],
        "max_canonical_probability_difference": maxdiff,
        "tie_accuracy_mismatch_theorem_ids": tie_accuracy_mismatches,
        "factorized": args.factorized,
        "tolerance": args.tolerance,
        "equivariant_within_tolerance": maxdiff <= args.tolerance if args.factorized else None,
        "input_hashes": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.predictions
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    Path(args.out + ".script.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
