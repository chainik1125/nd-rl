"""Guarded transfer summaries; ordinary and derived projections remain separate."""

import hashlib
import json
from pathlib import Path

import numpy as np
from transfer_guards import completed, validate

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_transfer_20261006"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mapping(rows):
    out = {r["theorem_id"]: r for r in rows}
    assert len(out) == len(rows)
    return out


def read(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def main():
    protocol = json.loads((ART / "phase1_protocol.json").read_text())
    assert protocol["analysis_sha256"] == sha(Path(__file__))
    assert protocol["inventory_sha256"] == sha(ART / "phase1_inventory.json")
    assert protocol["decisions_sha256"] == sha(ART / "phase1_decisions.json")
    metadata = mapping(read(ART / "data/combined412/public.jsonl"))
    assert sha(ART / "data/combined412/public.jsonl") == protocol["data_sha256"]
    for path, value in protocol["public_hashes"].items():
        assert sha(Path(path)) == value
    inventory = json.loads((ART / "phase1_inventory.json").read_text())
    commands = validate(protocol)
    consumed = completed(protocol, commands)
    pred = {}
    for command in inventory["commands"]:
        out = Path(command["argv"][command["argv"].index("--out") + 1])
        cfg = out / "config.json"
        if command["label"] == "features":
            continue
        config = json.loads(cfg.read_text())
        assert config["registered_signature"] == command["signature"]
        assert sha(Path(config["args"]["public"])) in protocol["registered_public_pool_sha256"]
        assert config["frozen_test_protocol_sha256"] == sha(ART / "phase1_protocol.json")
        p = out / "predictions.json"
        pred[command["label"]] = mapping(json.loads(p.read_text()))
        assert pred[command["label"]].keys() == metadata.keys()
        consumed[str(p)] = sha(p)
    feature_config = json.loads((ART / "analysis/features/config.json").read_text())
    assert sha(Path(feature_config["feature_model"])) == protocol["feature_model_sha256"]
    assert sha(Path(feature_config["public"])) in protocol["registered_public_pool_sha256"]
    assert Path(feature_config["out"]).resolve() == (ART / "analysis/features").resolve()
    features_path = ART / "analysis/features/per_theorem.jsonl"
    features = mapping(read(features_path))
    assert features.keys() == metadata.keys()
    consumed[str(features_path)] = sha(features_path)
    labels = mapping(read(ART / "pools/combined412/private.jsonl"))
    assert labels.keys() == metadata.keys()
    records = []
    for id, row in metadata.items():
        valid = {x["candidate_id"]: bool(x["valid"]) for x in labels[id]["labels"]}
        assert len(labels[id]["labels"]) == 8
        assert sorted(valid) == list(range(8))
        values = {
            "pass1_mean8": sum(valid.values()) / 8,
            "first_candidate": float(valid[0]),
            "oracle8": float(any(valid.values())),
        }
        assert all(np.isfinite(v) and 0 <= v <= 1 for v in features[id]["accuracy"].values())
        values.update(features[id]["accuracy"])
        for label, items in pred.items():
            item = items[id]
            ties = item["argmax_tied_candidate_ids"]
            assert ties and len(set(ties)) == len(ties) and all(i in valid for i in ties)
            assert np.isfinite(item["argmax_tie_expected_accuracy"])
            result = sum(valid[i] for i in ties) / len(ties)
            assert abs(result - item["argmax_tie_expected_accuracy"]) < 1e-7
            values[label] = result
        for name in ["pooled_transitions", "factorized_transitions", "factorized_bce_transitions"]:
            keys = [k for k in pred if k.startswith(name + "_s")]
            assert len(keys) == 3
            values[name] = sum(values[k] for k in keys) / 3
        records.append(
            {
                "theorem_id": id,
                "component": row["component"],
                "source_role": row["source_role"],
                "base_cluster": row["base_cluster"],
                "nuance": row.get("nuance"),
                "n_valid": sum(valid.values()),
                "values": values,
            }
        )
    summaries = {}
    for cohort in ["ordinary187", "derived225"]:
        rows = [r for r in records if r["component"] == cohort]
        n = len(rows)
        names = sorted(rows[0]["values"])
        array = np.array([[r["values"][k] for k in names] for r in rows])
        rng = np.random.default_rng(96006)
        if cohort == "ordinary187":
            groups = [
                [i for i, r in enumerate(rows) if r["source_role"] == role]
                for role in sorted({r["source_role"] for r in rows})
            ]
            draws = np.array(
                [
                    array[
                        np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
                    ].mean(0)
                    for _ in range(5000)
                ]
            )
        else:
            groups = [
                [i for i, r in enumerate(rows) if r["base_cluster"] == base]
                for base in sorted({r["base_cluster"] for r in rows})
            ]
            draws = np.array(
                [
                    array[
                        np.concatenate(
                            [groups[i] for i in rng.integers(0, len(groups), len(groups))]
                        )
                    ].mean(0)
                    for _ in range(5000)
                ]
            )
        mean = array.mean(0)
        feature = names.index("learned_features")
        metrics = {}
        for i, name in enumerate(names):
            metrics[name] = {
                "accuracy": float(mean[i]),
                "ci95": np.quantile(draws[:, i], [0.025, 0.975]).tolist(),
                "delta_features": float(mean[i] - mean[feature]),
                "delta_features_ci95": np.quantile(
                    draws[:, i] - draws[:, feature], [0.025, 0.975]
                ).tolist(),
                "N": 1 if name in ["pass1_mean8", "first_candidate"] else 8,
            }
        summaries[cohort] = {
            "theorems": n,
            "base_clusters": len(groups) if cohort == "derived225" else None,
            "estimand": "row-weighted theorem mean; whole-base-cluster ratio bootstrap"
            if cohort == "derived225"
            else "theorem mean; source-role-stratified bootstrap",
            "metrics": metrics,
        }
    out = ART / "analysis/phase1_summary"
    out.mkdir(parents=True, exist_ok=False)
    (out / "metrics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (out / "per_theorem.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "source_sha256": sha(Path(__file__)),
                "protocol_sha256": sha(ART / "phase1_protocol.json"),
                "consumed": consumed,
                "bootstrap": 5000,
                "seed": 96006,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                k: {
                    m: v["accuracy"]
                    for m, v in s["metrics"].items()
                    if m
                    in [
                        "pass1_mean8",
                        "first_candidate",
                        "oracle8",
                        "pooled_transitions",
                        "factorized_transitions",
                        "factorized_bce_transitions",
                        "learned_features",
                        "syntax_then_likelihood",
                    ]
                }
                for k, s in summaries.items()
            }
        )
    )


if __name__ == "__main__":
    main()
