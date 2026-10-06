"""Join private labels only after auditing deterministic public proof rendering."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan"
sys.path.insert(0, str(SOURCE))
from lean_gate import check_sources  # noqa: E402
from tokenizer import make_tokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.public).read_text().splitlines()]
    private = {
        r["theorem_id"]: {c["candidate_id"]: c for c in r["labels"]}
        for r in map(json.loads, Path(args.private).read_text().splitlines())
    }
    tok = make_tokenizer("lean_staten")
    sources, keys, original, mismatches = [], [], [], []
    completed = 0
    lengths = []
    for row in rows:
        lengths.append(
            len(tok.encode_prompt(row["prompt"]))
            + 3
            + sum(3 + len(c["transition_tokens"]) for c in row["candidates"])
        )
        for candidate in row["candidates"]:
            key = (row["theorem_id"], candidate["candidate_id"])
            label = private[key[0]][key[1]]
            literal = tok.statement(row["prompt"]) + " " + candidate["transition_text"]
            if label["parse_complete"]:
                completed += 1
                if literal != label["lean_text"]:
                    mismatches.append(key)
            sources.append(literal)
            keys.append(key)
            original.append(label["valid"])
    flags, wall, cpu = check_sources(sources)
    result = {
        "attempts": len(keys),
        "completed_identity_checks": completed,
        "completed_identity_mismatches": mismatches,
        "original_valid": sum(original),
        "direct_rendered_lean_valid": sum(flags),
        "label_mismatches": [
            key for key, a, b in zip(keys, original, flags, strict=True) if a != b
        ],
        "input_tokens_sorted": sorted(lengths),
        "overflow_8192": sum(n > 8192 for n in lengths),
        "wall_seconds": wall,
        "process_seconds": cpu,
        "input_hashes": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [args.public, args.private]
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    Path(args.out + ".script.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: v for k, v in result.items() if k != "input_tokens_sorted"}))


if __name__ == "__main__":
    main()
