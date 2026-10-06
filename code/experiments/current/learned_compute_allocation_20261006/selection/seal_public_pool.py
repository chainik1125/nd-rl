"""Seal public pool bytes without reading labels, summaries or private logs."""

import argparse
import hashlib
import json
from pathlib import Path

PUBLIC_METADATA = [
    "theorem_id",
    "split",
    "prompt",
    "structural_key",
    "family",
    "evaluation_role",
    "source_role",
    "canonical_hash",
    "name",
    "source_row_index",
]
CORE = {"theorem_id", "split", "prompt", "structural_key", "family"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(raw, transitions, expected, expected_theorems=400, expected_candidates=8, enrich=False):
    base = [json.loads(line) for line in Path(raw).read_text().splitlines()]
    rendered = [json.loads(line) for line in Path(transitions).read_text().splitlines()]
    metadata = [
        {k: r[k] for k in PUBLIC_METADATA if k in r}
        for r in [json.loads(line) for line in Path(expected).read_text().splitlines()]
    ]
    ids = [r["theorem_id"] for r in metadata]
    if len(ids) != expected_theorems or len(set(ids)) != expected_theorems:
        raise ValueError("Frozen theorem count/uniqueness differs from explicit plan")
    if [r["theorem_id"] for r in base] != ids or [r["theorem_id"] for r in rendered] != ids:
        raise ValueError("Public views differ from frozen theorem order")
    changed = False
    for a, b, planned in zip(base, rendered, metadata, strict=True):
        if planned["split"] != "test":
            raise ValueError("Not a TEST pool")
        for row in (a, b):
            for k, value in planned.items():
                if k not in row:
                    if not enrich or k in CORE:
                        raise ValueError("Missing registered public theorem metadata: " + k)
                    row[k] = value
                    changed = True
                if row[k] != value:
                    raise ValueError("Theorem metadata differs from frozen source: " + k)
            if [c["candidate_id"] for c in row["candidates"]] != list(range(expected_candidates)):
                raise ValueError("Candidate IDs differ from explicit plan")
        for ca, cb in zip(a["candidates"], b["candidates"], strict=True):
            if any(k not in cb or cb[k] != value for k, value in ca.items()):
                raise ValueError("Rendering changed an original public candidate field")
            if "transition_tokens" not in cb:
                raise ValueError("Missing transition view")
    before = {"raw_public_sha256": sha(raw), "transition_public_sha256": sha(transitions)}
    if changed:
        for path in (Path(raw), Path(transitions)):
            if path.with_name(path.name + ".before_metadata_seal").exists():
                raise ValueError("Refuse overwriting prior metadata-seal backup")
        for path, values in [(Path(raw), base), (Path(transitions), rendered)]:
            backup = path.with_name(path.name + ".before_metadata_seal")
            if backup.exists():
                raise ValueError("Refuse overwriting prior metadata-seal backup")
            backup.write_bytes(path.read_bytes())
            path.write_text("".join(json.dumps(r) + "\n" for r in values))
    return {
        "status": "public_metadata_only_pending_root_registry_seal",
        "theorems": expected_theorems,
        "attempts_per_theorem": expected_candidates,
        "raw_public_sha256": sha(raw),
        "transition_public_sha256": sha(transitions),
        "expected_data_sha256": sha(expected),
        "sealer_sha256": sha(__file__),
        "metadata_enriched": changed,
        "before_metadata_enrichment": before,
        "private_outcomes_read": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--transitions", required=True)
    parser.add_argument("--expected-data", required=True)
    parser.add_argument("--expected-theorems", type=int, default=400)
    parser.add_argument("--expected-candidates", type=int, default=8)
    parser.add_argument("--enrich-registered-metadata", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = seal(
        args.raw,
        args.transitions,
        args.expected_data,
        args.expected_theorems,
        args.expected_candidates,
        args.enrich_registered_metadata,
    )
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
