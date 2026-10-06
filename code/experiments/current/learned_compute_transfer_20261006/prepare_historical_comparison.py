"""Extract the already audited textbook72 results without rerunning inference."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / "artifacts/learned_compute_allocation_20261006"
OUT = ROOT / "artifacts/learned_compute_transfer_20261006"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selection_path = OLD / "analysis/final_historical/selection/metrics.json"
    live_path = OLD / "allocation/final_historical_summary_fp32_v2.json"
    private_path = OLD / "pools/historical_textbook72_test/private.jsonl"
    expected = {
        selection_path: "89b1445a28ca664cfbf8ba41a0de7e82cee1c84507ff00e1f076a5c6a8a41bc8",
        live_path: "2bf1f8f55513dc71fbf706f8d2b9e84a2ce3405eb5067edeedb66a66e86f9309",
        private_path: "3acf29f53d3f83b7045d7f8a11f37c3050f500a8bddb103decbfd647a012c60b",
    }
    for path, checksum in expected.items():
        if sha(path) != checksum:
            raise ValueError(f"Closed historical artifact changed: {path}")
    selection = json.loads(selection_path.read_text())
    live = json.loads(live_path.read_text())["combined"]
    private = [json.loads(line) for line in private_path.read_text().splitlines()]
    assert len(private) == 72 and len({r["theorem_id"] for r in private}) == 72
    validities = []
    for row in private:
        labels = sorted(row["labels"], key=lambda item: item["candidate_id"])
        assert [item["candidate_id"] for item in labels] == list(range(8))
        validities.append([bool(item["valid"]) for item in labels])
    pass1 = sum(sum(v) for v in validities) / (72 * 8)
    first = sum(v[0] for v in validities) / 72
    oracle = sum(any(v) for v in validities) / 72
    assert pass1 == selection["baselines"]["random"]
    assert oracle == selection["baselines"]["oracle"]
    arms = {
        "phase1_fixed8": {
            "label": "Original generator + fixed-eight pooled selector",
            "pass1": pass1,
            "first_candidate": first,
            "oracle8": oracle,
            "returned": selection["groups"]["pooled"]["primary"]["accuracy"],
            "attempts": 8.0,
            "training_seeds": 3,
            "returned_seed_values": selection["groups"]["pooled"]["seed_values"],
            "paired_vs_pass1": selection["groups"]["pooled"]["paired_vs_random"],
        }
    }
    for arm, policy in (("fixed4_control", "fixed4"), ("cost", "learned"), ("no_cost", "learned")):
        metrics = live["arms"][arm]
        oracle_metrics = live["finite_path_oracle_availability"][arm]
        arms[arm] = {
            "pass1": metrics["fixed1"]["correct_probability"]["mean"],
            "oracle4": oracle_metrics["after4"]["mean"],
            "oracle8": oracle_metrics["after8"]["mean"],
            "returned": metrics[policy]["correct_probability"]["mean"],
            "attempts": metrics[policy]["attempts"]["mean"],
            "return_policy": policy,
            "training_seeds": 1 if arm == "fixed4_control" else 3,
            "intervals": {
                "pass1": metrics["fixed1"]["correct_probability"],
                "oracle4": oracle_metrics["after4"],
                "oracle8": oracle_metrics["after8"],
                "returned": metrics[policy]["correct_probability"],
                "attempts": metrics[policy]["attempts"],
            },
            "all_readouts": metrics,
        }
        if arm != "fixed4_control":
            arms[arm]["paired_vs_pass1"] = live["contrasts"][f"{arm}:learned-minus-fixed1"][
                "correct_probability"
            ]
    result = {
        "scope": "textbook72",
        "rows": 72,
        "source": "Reused closed and audited historical evaluation; no new inference",
        "source_hashes": {
            str(path.relative_to(ROOT)): checksum for path, checksum in expected.items()
        },
        "arms": arms,
        "selection_comparisons": {
            name: group["primary"]["accuracy"] for name, group in selection["groups"].items()
        }
        | {"feature": selection["baselines"]["feature"]},
        "live_per_seed": live["per_seed"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "historical_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "scope": result["scope"],
                "arms": {
                    name: {key: arm[key] for key in ("pass1", "oracle8", "returned", "attempts")}
                    for name, arm in arms.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
