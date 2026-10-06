"""Metadata-only theorem overlap audit against already acquired sprint exposures."""

import gzip
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OLD = ROOT / "artifacts/learned_compute_allocation_20261006"
ART = ROOT / "artifacts/learned_compute_transfer_20261006"


def thm(r):
    if "thm" in r:
        return r["thm"]
    t = r["prompt"].split()
    i = t.index("SEQ")
    return " ".join(t[1:i]) + " |- " + " ".join(t[i + 1 : t.index("PRF")])


def key(s):
    p, g = s.split("|-")
    atoms = sorted(set(s.split()) & set("PQRS"))
    variants = []
    for names in itertools.permutations("PQRS"[: len(atoms)]):
        m = dict(zip(atoms, names, strict=True))

        def f(x, m=m):
            return " ".join(m.get(t, t) for t in x.split())

        variants.append(" , ".join(sorted(f(x) for x in p.split(",") if x.strip())) + " |- " + f(g))
    return min(variants)


def main():
    rows = [json.loads(x) for x in (ART / "data/combined412/public.jsonl").read_text().splitlines()]
    keys = {r["theorem_id"]: key(thm(r)) for r in rows}
    paths = {
        "K12": OLD / "source/train_k12.jsonl.gz",
        "expert_iteration": OLD / "source/nd-takehome-dan/data/ladder/rl_targets.jsonl",
        "actual_r16": OLD / "source/actual_s0_r16_mix_16.jsonl",
        "historical72": OLD / "data/textbook72_historical_public_v1/public.jsonl",
    }
    result = {}
    for label, path in paths.items():
        opener = gzip.open if path.suffix == ".gz" else open
        seen = set()
        n = 0
        with opener(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                seen.add(key(thm(r)))
                n += 1
        matches = [r["theorem_id"] for r in rows if keys[r["theorem_id"]] in seen]
        result[label] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": n,
            "matching_ids": matches,
            "ordinary_matches": sum(x.startswith("charles1002_derived_") is False for x in matches),
            "derived_projection_matches": sum(
                x.startswith("charles1002_derived_") for x in matches
            ),
        }
    counts = {}
    for r in rows:
        counts.setdefault(keys[r["theorem_id"]], []).append(r["theorem_id"])
    report = {
        "canonical_definition": "atom bijection/PQRS + premise permutation; no curry normalization",
        "matches": result,
        "within_release_duplicate_groups": [v for v in counts.values() if len(v) > 1],
        "no_outcome_exclusions": True,
        "complete_historical_exposure_unknown": True,
    }
    (ART / "data/exposure_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: (v["ordinary_matches"], v["derived_projection_matches"]) for k, v in result.items()}
        )
    )


if __name__ == "__main__":
    main()
