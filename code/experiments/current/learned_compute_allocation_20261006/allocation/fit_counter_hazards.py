"""Fit a count-only stopping hazard from public development policy paths."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluation_protocol import fingerprint, signature, validate_saved_implementation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths_file, data_file = Path(args.paths), Path(args.data)
    config = json.loads(paths_file.with_name("config.json").read_text())
    validate_saved_implementation(config)
    data = [json.loads(line) for line in data_file.read_text().splitlines()]
    if any(row.get("split") != "dev" for row in data):
        raise ValueError("hazard fitting requires development inputs only")
    data_ids = {row["theorem_id"] for row in data}
    paths = [json.loads(line) for line in paths_file.read_text().splitlines()]
    if not {row["theorem_id"] for row in paths} <= data_ids:
        raise ValueError("path IDs are not in the registered development dataset")
    if config["data_sha256"] != hashlib.sha256(data_file.read_bytes()).hexdigest():
        raise ValueError("path fitting data differs from evaluation input")
    counts = config["max_attempts"]
    numerator, denominator = [0.0] * counts, [0.0] * counts
    forced_reach = [0.0] * counts
    for path in paths:
        reach = 1.0
        for i, decision in enumerate(path["decisions"]):
            if decision["forced"]:
                forced_reach[i] += reach
                hazard = 1.0
            else:
                hazard = decision["accept_probability"]
                numerator[i] += reach * hazard
                denominator[i] += reach
            reach *= 1 - hazard
    hazards = [a / b if b else 1.0 for a, b in zip(numerator, denominator, strict=True)]
    current_signature = signature(config)
    result = {
        "fit_split": "dev",
        "fitter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hazards": hazards,
        "fit_data_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
        "fit_public_paths_sha256": hashlib.sha256(paths_file.read_bytes()).hexdigest(),
        "base_sha256": current_signature["base_sha256"],
        "checkpoint_sha256": current_signature["checkpoint_sha256"],
        "inference_sha256": fingerprint(current_signature),
        "fit_rule": "reach-weighted mean learned hazard over nonforced states at each count",
        "forced_reach_mass": forced_reach,
        "scope": "matches count distribution when forcing is terminal-only; verify realized counts",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2)
    with out.with_suffix(".fitter.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
