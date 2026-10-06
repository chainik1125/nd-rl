"""Reproduce the probability-domain defect using only six selected DEV path files."""

import argparse
import inspect
import json
from pathlib import Path

import aggregate_live as original
import aggregate_live_fp32_v2 as successor
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    expected = inspect.getsource(original.metrics).replace(
        "not -1e-8 <= correct <= 1 + 1e-8", "not -1e-6 <= correct <= 1 + 1e-6"
    )
    if inspect.getsource(successor.metrics) != expected:
        raise ValueError("Numerical function changed beyond the one probability-domain guard")
    checked, rejected, passing_unchanged, inputs, examples = 0, 0, 0, {}, []
    for arm in ("cost", "no_cost"):
        for seed in (101, 102, 103):
            folder = Path(args.artifacts) / arm / f"main_s{seed}_dev200_paths_s52006"
            config = json.loads((folder / "config.json").read_text())
            if config["evaluation_split"] != "dev":
                raise ValueError("Reproducer is development-only")
            source = folder / "integrated.json"
            inputs[str(source)] = original.sha(source)
            for method, rows in json.loads(source.read_text()).items():
                for row in rows:
                    checked += 1
                    result = successor.metrics(row, config["cost"], config["max_attempts"])
                    if result[0] != row["correct_probability"]:
                        raise ValueError("Successor clipped or changed correctness")
                    try:
                        old = original.metrics(row, config["cost"], config["max_attempts"])
                    except ValueError:
                        rejected += 1
                        examples.append(
                            dict(
                                arm=arm,
                                seed=seed,
                                method=method,
                                correct_probability=row["correct_probability"],
                            )
                        )
                    else:
                        if not np.array_equal(old, result):
                            raise ValueError("Successor changes estimates")
                        passing_unchanged += 1
    report = dict(
        scope="DEV-only validation; no final outcomes read",
        checked_rows=checked,
        original_rejected_rows=rejected,
        successor_rejected_rows=0,
        original_passing_arrays_bitwise_unchanged=passing_unchanged,
        maximum_correctness_excess=max(x["correct_probability"] - 1 for x in examples),
        metrics_diff="Only correct_probability domain tolerance:1e-8 to1e-6",
        clipping=False,
        renormalization=False,
        input_hashes=inputs,
        original_script_sha256=original.sha(original.__file__),
        successor_script_sha256=original.sha(successor.__file__),
        reproducer_sha256=original.sha(__file__),
        examples=examples,
    )
    with Path(args.out).open("x") as stream:
        json.dump(report, stream, indent=2)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "checked_rows",
                    "original_rejected_rows",
                    "successor_rejected_rows",
                    "original_passing_arrays_bitwise_unchanged",
                    "maximum_correctness_excess",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
