"""Descriptive classical truth-table audit of training/development target shapes."""

import argparse
import collections
import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan"
sys.path.insert(0, str(SOURCE))
from nd2lean import parse_prompt  # noqa: E402


def truth(formula, assignment):
    kind = formula[0]
    if kind == "atom":
        return assignment[formula[1]]
    if kind == "bot":
        return False
    if kind == "not":
        return not truth(formula[1], assignment)
    left, right = truth(formula[1], assignment), truth(formula[2], assignment)
    return left and right if kind == "and" else left or right if kind == "or" else not left or right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out, hashes = {}, {}
    for split in ["train", "dev"]:
        for family in ["forward", "ore2"]:
            path = Path(args.data) / family / f"{split}.jsonl"
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            stats = collections.Counter()
            for row in map(json.loads, path.read_text().splitlines()):
                premises, goal = parse_prompt(row["prompt"])
                values = [
                    dict(zip("PQRS", v, strict=True))
                    for v in itertools.product([False, True], repeat=4)
                ]
                satisfiable = any(all(truth(f, a) for f in premises) for a in values)
                stats["theorems"] += 1
                stats["classically_unsatisfiable_premises"] += not satisfiable
                stats["classical_tautological_goal"] += all(truth(goal, a) for a in values)
            out[f"{split}/{family}"] = dict(stats)
    result = {
        "caveat": (
            "Classical four-atom truth-table description, not intuitionistic "
            "minimal-proof evidence; finaltest unopened."
        ),
        "counts": out,
        "input_hashes": hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    Path(args.out + ".script.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(out))


if __name__ == "__main__":
    main()
