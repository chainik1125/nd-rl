"""Project historical textbook prompts into guarded, reference-free public metadata."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = SOURCE / "data/bs/textbook72.jsonl"
    raw = [json.loads(s) for s in source.read_text().splitlines()]
    if len(raw) != 72:
        raise ValueError("Expected all72 historical prompts")
    groups = defaultdict(list)
    rows = []
    for index, row in enumerate(raw):
        canonical = row["canonical_hash"]
        groups[canonical].append(index)
        rows.append(
            {
                "theorem_id": f"historical_textbook72_{index:03d}_{canonical[:20]}",
                "split": "test",
                "prompt": row["prompt"],
                "family": row["role"],
                "source_role": row["role"],
                "evaluation_role": "historical_reused_benchmark",
                "canonical_hash": canonical,
                "name": row["name"],
                "source_row_index": index,
                "structural_key": canonical,
            }
        )
    roles = dict(Counter(r["source_role"] for r in rows))
    if sorted(roles.values()) != [14, 58] or len({r["theorem_id"] for r in rows}) != 72:
        raise ValueError("Unexpected role counts or duplicate IDs")
    folder = (
        ROOT / "artifacts/learned_compute_allocation_20261006/data/textbook72_historical_public_v1"
    )
    folder.mkdir(parents=True, exist_ok=False)
    public = folder / "public.jsonl"
    public.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = {
        "source": str(source),
        "source_sha256": sha(source),
        "source_commit": "4dcd0e4f336b3bf0ee24230dc5335ba3dba8969c",
        "public_sha256": sha(public),
        "rows": 72,
        "source_roles": roles,
        "canonical_groups": len(groups),
        "duplicate_canonical_groups": {k: v for k, v in groups.items() if len(v) > 1},
        "metadata_script_sha256": sha(__file__),
        "prompt_conversion": "none; exact original prompt bytes preserved",
        "reference_fields_removed": ["reference_lines"],
        "split_interpretation": (
            "test triggers mandatory guards; historical reuse, not fresh generalization"
        ),
        "known_pretraining_overlap": 1,
        "exposure_audit": (
            "artifacts/learned_compute_allocation_20261006/data/"
            "textbook72_metadata_audit/exposure.json"
        ),
        "outcome_based_exclusions": False,
        "planned_pool": {
            "seed": 93006,
            "k": 8,
            "temperature": 0.8,
            "batch": 256,
            "max_action": 256,
            "max_steps": 48,
            "model": "original r16",
            "precision": "BF16 autocast model; FP32 sampling logits",
        },
        "historical_setting_difference": (
            "Upstream k256/max_action512/max_steps96 "
            "versus our bounded8/256/48; no published-number reproduction claim"
        ),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (folder / "source.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "public": str(public),
                "roles": roles,
                "canonical_groups": len(groups),
                "duplicate_canonical_groups": manifest["duplicate_canonical_groups"],
                "sha256": manifest["public_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
