"""Compare fresh target skeletons against the audited pretraining/EI exposure union."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from data import skeleton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    cache = Path(args.cache)
    with gzip.open(cache, "rt") as stream:
        keys = json.load(stream)
    groups = {skeleton(key) for key in keys}
    result = {
        "known_exposure_skeletons": len(groups),
        "source_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "splits": {},
        "input_hashes": {},
    }
    for split in ["train", "dev"] + (["test"] if args.include_test else []):
        path = Path(args.data) / f"{split}.jsonl"
        result["input_hashes"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = list(map(json.loads, path.read_text().splitlines()))
        result["splits"][split] = {
            family: {
                "n": sum(r["family"] == family for r in rows),
                "known_exposure_skeleton_matches": sum(
                    r["family"] == family and r["skeleton"] in groups for r in rows
                ),
            }
            for family in ["forward", "ore2"]
        }
    Path(args.out).write_text(json.dumps(result, indent=2))
    Path(args.out + ".script.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
