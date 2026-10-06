"""Join frozen family pools in the prospectively fixed theorem order."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theorems", required=True)
    parser.add_argument("--parts", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    public = {}
    private = {}
    hashes = {}
    for part in args.parts:
        for kind, target in [("public", public), ("private", private)]:
            path = Path(part) / f"{kind}.jsonl"
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for row in read(path):
                assert row["theorem_id"] not in target
                target[row["theorem_id"]] = row
    rows = read(args.theorems)
    order = [row["theorem_id"] for row in rows]
    assert set(order) == set(public) == set(private)
    for tid in order:
        assert sorted(c["candidate_id"] for c in public[tid]["candidates"]) == list(range(8))
        assert sorted(c["candidate_id"] for c in private[tid]["labels"]) == list(range(8))
    for kind, source in [("public", public), ("private", private)]:
        (out / f"{kind}.jsonl").write_text("".join(json.dumps(source[tid]) + "\n" for tid in order))
    metadata = {
        "args": vars(args),
        "theorems": len(order),
        "source_hashes": hashes,
        "theorem_order_sha256": hashlib.sha256(Path(args.theorems).read_bytes()).hexdigest(),
        "merger_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2))
    (out / "merger.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
