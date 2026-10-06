"""Apply a development-fitted count hazard to immutable, already checked paths."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluate_live_path import integrate
from evaluation_protocol import signature, validate_counter, validate_reanalysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--hazards", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frozen-protocol")
    args = parser.parse_args()
    paths_file, labels_file = Path(args.paths), Path(args.labels)
    config = json.loads(paths_file.with_name("config.json").read_text())
    validate_reanalysis(
        config,
        args.frozen_protocol,
        args.hazards,
        "counter_hazards_sha256",
        ("reanalyze_counter.py", "fit_counter_hazards.py"),
    )
    counter = json.loads(Path(args.hazards).read_text())
    validate_counter(counter, signature(config))
    labels = {}
    for line in labels_file.read_text().splitlines():
        row = json.loads(line)
        key = row["theorem_id"], row["repeat"]
        if key in labels:
            raise ValueError("duplicate private path key")
        labels[key] = row["validities"]
    results = {"counter": [], "counter_greedy_index": []}
    seen = set()
    for line in paths_file.read_text().splitlines():
        path = json.loads(line)
        key = path["theorem_id"], path["repeat"]
        if key in seen:
            raise ValueError("duplicate public path key")
        seen.add(key)
        for method, values in results.items():
            value = integrate(path, labels[key], method, counter["hazards"], config["cost"])
            value.update(theorem_id=key[0], repeat=key[1])
            values.append(value)
    if seen != labels.keys():
        raise ValueError("public and private path keys differ")
    result = {
        "provenance": {
            "paths_sha256": hashlib.sha256(paths_file.read_bytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(labels_file.read_bytes()).hexdigest(),
            "hazards_sha256": hashlib.sha256(Path(args.hazards).read_bytes()).hexdigest(),
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evaluation_sha256": hashlib.sha256(
                Path(__file__).with_name("evaluate_live_path.py").read_bytes()
            ).hexdigest(),
        },
        "integrated": results,
    }
    with Path(args.out).open("x") as stream:
        json.dump(result, stream)
    with Path(args.out).with_suffix(".implementation.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
