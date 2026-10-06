"""Public sequent-only transfer inputs; derived task requirements remain audit-only."""

import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / "artifacts/learned_compute_transfer_20261006"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ordinary = [
        json.loads(x) for x in (ART / "data/ordinary187/public.jsonl").read_text().splitlines()
    ]
    rows = []
    for r in ordinary:
        rows.append(
            dict(
                r,
                structural_key=r["canonical_hash"],
                component="ordinary187",
                projection=False,
                base_cluster=r["source_name"],
            )
        )
    audit = []
    for i, r in enumerate(
        json.loads(x) for x in (ART / "release/derived.jsonl").read_text().splitlines()
    ):
        prompt = r["prompt"]
        assert prompt.startswith("THM ") and prompt.endswith(" PRF")
        canonical = hashlib.sha256(prompt.encode()).hexdigest()
        rows.append(
            {
                "theorem_id": f"charles1002_derived_{i:03d}_{r['name']}",
                "prompt": prompt,
                "split": "test",
                "family": "derived_" + r["nuance"],
                "source_role": "derived",
                "component": "derived225",
                "projection": True,
                "projection_omissions": [
                    "given proof/prefix",
                    "task instruction or markers",
                    "structural scoring constraint",
                ],
                "evaluation_role": "sequent_projection_not_native_derived_task",
                "structural_key": canonical,
                "canonical_hash": canonical,
                "canonical_hash_kind": "exact_prompt_sha256_not_renaming_class",
                "source_name": r["name"],
                "source_row_index": i,
                "nuance": r["nuance"],
                "base_cluster": r["base"]["name"],
                "curried_twin_of": r.get("curried_twin_of"),
                "release_commit": ordinary[0]["release_commit"],
            }
        )
        audit.append(
            {
                "theorem_id": rows[-1]["theorem_id"],
                "task_input": r["task_input"],
                "given": r.get("given"),
                "check": r.get("check"),
                "scoring": r["scoring"],
                "base": r["base"],
                "curried_twin_of": r.get("curried_twin_of"),
            }
        )
    out = ART / "data/combined412"
    out.mkdir(parents=True, exist_ok=False)
    for name, values in [
        ("public.jsonl", rows),
        ("derived_requirements_private_audit.jsonl", audit),
    ]:
        (out / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in values))
    manifest = {
        "rows": len(rows),
        "components": dict(collections.Counter(r["component"] for r in rows)),
        "public_sha256": sha(out / "public.jsonl"),
        "private_audit_sha256": sha(out / "derived_requirements_private_audit.jsonl"),
        "derived_designated_curried_twins": sum(bool(r.get("curried_twin_of")) for r in rows),
        "source_release_sha256": sha(ART / "release/derived.jsonl"),
        "no_reference_fields": True,
        "ordinary_prompt_unchanged": True,
        "scope": (
            "ordinary primary and separately reported derived sequent projection; "
            "never pooled native task accuracy"
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
