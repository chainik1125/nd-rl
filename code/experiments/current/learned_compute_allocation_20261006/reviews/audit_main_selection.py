"""Independent authorized MAIN selector bindings, endpoint and interval checks."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
OUT = ART / "analysis/final_main/selection"


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


config = read(OUT / "config.json")
metrics = read(OUT / "metrics.json")
registry = read(ART / "final_evaluation/main_protocol.json")
registered = [entry["signature"] for entry in registry["registered_analyses"]]
assert config["signature"] in registered
for path, expected in config["inputs"].items():
    assert sha(path) == expected
inventory = read(config["args"]["inventory"])
assert sha(config["args"]["inventory"]) == config["signature"]["inventory_sha256"]
assert sha(OUT / "source.py") == config["signature"]["code_sha256"]
public = {r["theorem_id"]: r for r in rows(config["args"]["public"])}
private = {
    r["theorem_id"]: sorted(r["labels"], key=lambda c: c["candidate_id"])
    for r in rows(config["args"]["private"])
}
features = {r["theorem_id"]: r for r in rows(config["args"]["features"])}
ids = sorted(public)
families = np.array([public[t]["family"] for t in ids])
valid = {t: np.array([c["valid"] for c in private[t]]) for t in ids}
complete = np.array([sum(c["parse_complete"] for c in private[t]) for t in ids])
random = np.array([valid[t].mean() for t in ids])
oracle = np.array([valid[t].any() for t in ids], dtype=float)
feature = np.array([features[t]["accuracy"]["learned_features"] for t in ids])
saved_rows = {(r["group"], r["theorem_id"]): r for r in rows(OUT / "per_theorem.jsonl")}
expected_keys = set(
    itertools.product(["pooled", "factorized", "bce"], [0, 1, 2], [77106, 77107, 77108])
)
runs = {}
errors = []
for entry in inventory["runs"]:
    key = entry["group"], entry["training_seed"], entry["permutation_seed"]
    assert key in expected_keys and key not in runs
    prediction_config = read(entry["prediction_config"])
    signature = prediction_config["registered_signature"]
    assert signature in registered
    assert sha(entry["checkpoint"]) == entry["checkpoint_sha256"] == signature["checkpoint_sha256"]
    assert prediction_config["checkpoint_metadata"]["selector_config"]["args"]["seed"] == key[1]
    assert prediction_config["args"]["evaluation_seed"] == key[2]
    assert signature["architecture"] == ("pooled" if key[0] == "pooled" else "factorized")
    assert signature["objective"] == ("binary" if key[0] == "bce" else "marginal")
    assert prediction_config["frozen_test_protocol_sha256"] == sha(
        ART / "final_evaluation/main_protocol.json"
    )
    for source, expected in prediction_config["source_hashes"].items():
        assert sha(source) == expected
    directory = Path(entry["prediction_config"]).parent
    for filename, field in [("evaluate.py", "evaluator_sha256"), ("trainer.py", "trainer_sha256")]:
        assert sha(directory / filename) == signature[field]
    if key[0] == "bce":
        assert sha(directory / "bce_trainer.py") == signature["bce_trainer_sha256"]
    predictions = read(entry["predictions"])
    assert len(predictions) == len({r["theorem_id"] for r in predictions}) == 400
    pred = {r["theorem_id"]: r for r in predictions}
    assert set(pred) == set(ids)
    scores = []
    for tid in ids:
        row = pred[tid]
        probabilities = np.asarray(row["probabilities"])
        assert len(probabilities) == 8 and np.isfinite(probabilities).all()
        ties = np.flatnonzero(probabilities == max(probabilities)).tolist()
        assert sorted(row["argmax_tied_candidate_ids"]) == ties
        value = valid[tid][ties].mean()
        errors.append(abs(value - row["argmax_tie_expected_accuracy"]))
        assert row["selected_candidate_id"] in ties
        assert row["selected_valid"] == valid[tid][row["selected_candidate_id"]]
        scores.append(value)
    runs[key] = np.array(scores)
assert set(runs) == expected_keys
report = {}
for group, saved in metrics["groups"].items():
    primary = np.mean([runs[group, seed, 77106] for seed in range(3)], axis=0)
    for index, tid in enumerate(ids):
        errors.append(abs(primary[index] - saved_rows[group, tid]["seed_mean_accuracy"]))
    errors.append(abs(primary.mean() - saved["primary"]["accuracy"]))
    errors.append(
        abs(primary.mean() / oracle.mean() - saved["primary"]["conditional_on_available_accuracy"])
    )
    errors.append(
        abs(
            (primary.mean() - random.mean()) / (oracle.mean() - random.mean())
            - saved["primary"]["oracle_gap_recovery"]
        )
    )
    for name, baseline in [("random", random), ("feature", feature)]:
        difference = primary - baseline
        rng = np.random.default_rng(61006)
        bootstrap = np.zeros(5000)
        for family in sorted(set(families)):
            values = difference[families == family]
            draws = rng.integers(len(values), size=(5000, len(values)))
            bootstrap += values[draws].sum(axis=1) / len(ids)
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        result = saved["paired_vs_" + name]
        errors.extend(
            [
                abs(difference.mean() - result["mean"]),
                abs(low - result["low"]),
                abs(high - result["high"]),
            ]
        )
    for permutation in saved["permutations"]:
        values = [runs[group, s, permutation["permutation_seed"]].mean() for s in range(3)]
        assert np.allclose(values, permutation["seed_values"], rtol=0, atol=1e-14)
        errors.append(abs(np.mean(values) - permutation["accuracy"]))
    mask = (complete == 8) & (random > 0) & (random < 1)
    stratum = next(s for s in saved["strata"] if s["axis"] == "all_complete_mixed")
    errors.append(abs(primary[mask].mean() - stratum["metrics"]["accuracy"]))
    errors.append(abs(feature[mask].mean() - stratum["feature_accuracy"]))
    report[group] = {
        "accuracy": primary.mean(),
        "paired_vs_feature": saved["paired_vs_feature"],
        "paired_vs_random": saved["paired_vs_random"],
        "all_complete_mixed": {
            "theorems": int(mask.sum()),
            "accuracy": primary[mask].mean(),
            "feature_accuracy": feature[mask].mean(),
        },
        "permutation_summary": saved["permutation_summary"],
    }
assert max(errors) < 1e-12
result = {
    "predictions_bound": len(runs),
    "predicted_theorems_checked": len(runs) * 400,
    "max_discrepancy": max(errors),
    "groups": report,
}
Path(__file__).with_name("main_selection_independent_audit.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
