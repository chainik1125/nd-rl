"""Canonical metadata exclusion of known public evaluation theorem sets."""

import hashlib
import json

from data import ROOT, SOURCE, key


def benchmark_exclusions():
    paths = [
        SOURCE / "targets/validation_36.jsonl",
        SOURCE / "data/bs/textbook72.jsonl",
        SOURCE / "data/heldout.jsonl",
        SOURCE / "data/bs/holdout250.jsonl",
    ]
    keys = set()
    metadata = []
    for p in paths:
        n = 0
        for line in p.read_text().splitlines():
            r = json.loads(line)
            thm = r.get("thm")
            if thm is None:
                tokens = r["prompt"].split()
                j = tokens.index("SEQ")
                thm = " ".join(tokens[1:j]) + " |- " + " ".join(tokens[j + 1 : tokens.index("PRF")])
            keys.add(key(thm))
            n += 1
        metadata.append(
            {
                "path": str(p),
                "rows": n,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "access": (
                    "canonical theorem metadata for exclusion only; no outcomes/refere"
                    "nce proofs used"
                ),
            }
        )
    return keys, metadata


def main():
    keys, meta = benchmark_exclusions()
    src = ROOT / "artifacts/learned_compute_allocation_20261006/data/fresh_v1"
    out = src.with_name("fresh_v1_safe")
    out.mkdir(exist_ok=True)
    removed = {}
    counts = {}
    for split in ["train", "dev", "test"]:
        rows = [json.loads(line) for line in (src / f"{split}.jsonl").read_text().splitlines()]
        removed[split] = [r["theorem_id"] for r in rows if r["structural_key"] in keys]
        rows = [r for r in rows if r["structural_key"] not in keys]
        counts[split] = len(rows)
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = {
        "source": str(src),
        "source_manifest_sha256": hashlib.sha256((src / "manifest.json").read_bytes()).hexdigest(),
        "additional_exclusions": meta,
        "removed_ids": removed,
        "splits": counts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
