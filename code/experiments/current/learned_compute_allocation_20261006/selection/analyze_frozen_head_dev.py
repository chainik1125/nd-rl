"""All-stratum DEV comparison for the prespecified frozen-trunk scalar critic."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

FOLDER = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "frozen_head_dev_statistics", FOLDER / "aggregate_selection.py"
)
statistics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(statistics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-run", required=True)
    parser.add_argument("--public", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--bce-predictions", required=True)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    public = {r["theorem_id"]: r for r in statistics.read_rows(args.public)}
    assert all(r["split"] == "dev" for r in public.values())
    private = {r["theorem_id"]: r for r in statistics.read_rows(args.private)}
    feature = {r["theorem_id"]: r for r in statistics.read_rows(args.features)}
    models = {
        "frozen_head": Path(args.head_run) / "dev_predictions.json",
        "full_bce_seed0": Path(args.bce_predictions),
        "original_n1_minus_n2": Path(args.base_predictions),
    }
    predictions = {
        name: {r["theorem_id"]: r for r in json.loads(path.read_text())}
        for name, path in models.items()
    }
    assert set(public) == set(private) == set(feature)
    assert all(set(rows) == set(public) for rows in predictions.values())
    ids = sorted(public)
    families = np.asarray([public[tid]["family"] for tid in ids])
    n_valid, complete = [], []
    scores = {name: [] for name in predictions}
    for tid in ids:
        labels = {c["candidate_id"]: c for c in private[tid]["labels"]}
        assert sorted(labels) == list(range(8))
        n_valid.append(sum(c["valid"] for c in labels.values()))
        complete.append(sum(c["parse_complete"] for c in labels.values()))
        for name, rows in predictions.items():
            tied = rows[tid]["argmax_tied_candidate_ids"]
            value = sum(labels[i]["valid"] for i in tied) / len(tied)
            assert abs(value - rows[tid]["argmax_tie_expected_accuracy"]) < 1e-7
            scores[name].append(value)
    scores = {name: np.asarray(values) for name, values in scores.items()}
    scores["public_features"] = np.asarray(
        [feature[tid]["accuracy"]["learned_features"] for tid in ids]
    )
    scores["random"] = np.asarray(n_valid) / 8
    oracle = (np.asarray(n_valid) > 0).astype(float)
    masks = [("all", "all", np.ones(len(ids), dtype=bool))]
    masks += [("family", f, families == f) for f in sorted(set(families))]
    masks += [("n_valid", n, np.asarray(n_valid) == n) for n in range(9)]
    masks += [("n_complete", n, np.asarray(complete) == n) for n in range(9)]
    masks += [
        (
            "all_complete_mixed",
            True,
            (np.asarray(complete) == 8) & (np.asarray(n_valid) > 0) & (np.asarray(n_valid) < 8),
        )
    ]
    strata = []
    for axis, value, mask in masks:
        strata.append(
            {
                "axis": axis,
                "value": value,
                "theorems": int(mask.sum()),
                "metrics": {
                    name: statistics.endpoint(values[mask], scores["random"][mask], oracle[mask])
                    if mask.any()
                    else None
                    for name, values in scores.items()
                },
            }
        )
    paired = {
        name: statistics.stratified_interval(scores["frozen_head"] - values, families, 5000, 61006)
        for name, values in scores.items()
        if name != "frozen_head"
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    inputs = [args.public, args.private, args.features, *map(str, models.values())]
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "scope": "Prespecified secondary DEV-only control",
                "strata": strata,
                "paired_frozen_head_minus_comparator": paired,
                "inputs": {p: statistics.digest(p) for p in inputs},
            },
            indent=2,
        )
        + "\n"
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"paired": paired, "all": strata[0]}))


if __name__ == "__main__":
    main()
