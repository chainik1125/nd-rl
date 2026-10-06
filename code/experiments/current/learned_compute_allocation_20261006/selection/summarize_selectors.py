"""Record exact frozen development choices and hashes without opening test data."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for name in args.runs:
        run = Path(name)
        choice = json.loads((run / "best_epoch.json").read_text())
        config = json.loads((run / "config.json").read_text())
        epoch = choice["epoch"]
        checkpoint = run / f"epoch{epoch}.pt"
        prediction = run / f"dev_epoch{epoch}.json"
        pred = json.loads(prediction.read_text())
        rows.append(
            {
                "run": str(run),
                "architecture": config["args"]["architecture"],
                "objective": config["objective"],
                "optimizer_seed": config["args"]["seed"],
                "selected_epoch": epoch,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "dev_predictions": str(prediction),
                "dev_predictions_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
                "dev_accuracy_exact_ties": sum(p["argmax_tie_expected_accuracy"] for p in pred)
                / len(pred),
                "dev_soft_expected_accuracy": sum(p["expected_accuracy"] for p in pred) / len(pred),
                "dev_binary_logloss": sum(p["binary_logloss"] for p in pred) / len(pred)
                if "binary_logloss" in pred[0]
                else None,
                "dev_brier": sum(p["brier_score"] for p in pred) / len(pred)
                if "brier_score" in pred[0]
                else None,
                "frozen_choice": choice,
                "configuration_sha256": hashlib.sha256(
                    (run / "config.json").read_bytes()
                ).hexdigest(),
            }
        )
    result = {
        "runs": rows,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    Path(args.out + ".script.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
