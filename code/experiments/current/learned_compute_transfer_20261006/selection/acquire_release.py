"""Extract pinned Charles git blobs and reference-free ordinary evaluation metadata."""

import collections
import hashlib
import json
import subprocess
from pathlib import Path

COMMIT = "5d0084add453212e6bea8055d4d28cec790f268c"
ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "artifacts/learned_compute_transfer_20261006"
REL = "data/textbook_problem/release_2026-10-02"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    release = OUT / "release"
    release.mkdir(parents=True, exist_ok=True)
    metadata = []
    files = [
        "README.md",
        "MANIFEST.json",
        "baselines.md",
        "candidate_v0.jsonl",
        "candidate_v1.jsonl",
        "hand_proved_14.jsonl",
        "batch3.jsonl",
        "derived.jsonl",
    ]
    for name in files:
        data = subprocess.check_output(["git", "show", f"{COMMIT}:{REL}/{name}"], cwd=ROOT)
        p = release / name
        if p.exists():
            assert p.read_bytes() == data
        else:
            p.write_bytes(data)
        metadata.append({"git_path": f"{REL}/{name}", "git_commit": COMMIT, "sha256": sha(data)})
    rows = []
    for filename, expected in [
        ("candidate_v0", 87),
        ("candidate_v1", 72),
        ("hand_proved_14", 14),
        ("batch3", 14),
    ]:
        source = [json.loads(x) for x in (release / f"{filename}.jsonl").read_text().splitlines()]
        assert len(source) == expected
        for i, r in enumerate(source):
            rows.append(
                {
                    "theorem_id": f"charles1002_{filename}_{i:03d}_{r['canonical_hash'][:20]}",
                    "prompt": r["prompt"],
                    "split": "test",
                    "family": filename,
                    "source_role": filename,
                    "evaluation_role": "historical_reused_benchmark",
                    "source_name": r["name"],
                    "canonical_hash": r["canonical_hash"],
                    "source_row_index": i,
                    "source": r["source"],
                    "encoding_notes": r.get("encoding_notes", []),
                    "release_commit": COMMIT,
                }
            )
    assert len(rows) == 187 and len({r["theorem_id"] for r in rows}) == 187
    public = OUT / "data/ordinary187/public.jsonl"
    public.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
    if public.exists():
        assert public.read_bytes() == content
    else:
        public.write_bytes(content)
    derived = [json.loads(x) for x in (release / "derived.jsonl").read_text().splitlines()]
    manifest = {
        "commit": COMMIT,
        "release_blobs": metadata,
        "public_path": str(public),
        "public_sha256": sha(content),
        "rows": 187,
        "source_roles": dict(collections.Counter(r["family"] for r in rows)),
        "ordinary_canonical_unique": len({r["canonical_hash"] for r in rows}),
        "derived_nuances": dict(collections.Counter(r["nuance"] for r in derived)),
        "derived_given_nonempty": sum(bool(r.get("given")) for r in derived),
        "derived_scoring": dict(collections.Counter(r["scoring"] for r in derived)),
        "reference_fields_public": False,
        "prompt_conversion": "none; exact original release prompts",
    }
    (OUT / "release_acquisition.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
