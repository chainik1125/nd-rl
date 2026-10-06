"""Read only theorem metadata and hashes; never inspect final-test outcomes or proof fields."""

import collections
import gzip
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path.cwd()
ART = ROOT / "artifacts/learned_compute_allocation_20261006"
P = ART / "data/main_v1"
manifest = json.loads((P / "manifest.json").read_text())


def rows(path):
    for line in path.read_text().splitlines():
        r = json.loads(line)
        yield {
            k: r[k]
            for k in [
                "thm",
                "prompt",
                "theorem_id",
                "structural_key",
                "skeleton",
                "split",
                "family",
            ]
            if k in r
        }


def canonical(thm):
    left, right = thm.split("|-")
    prem = [x.strip() for x in left.split(",") if x.strip()]
    atoms = sorted(set(thm.split()) & set("PQRS"))
    out = []
    for perm in itertools.permutations("PQRS"[: len(atoms)]):
        mp = dict(zip(atoms, perm, strict=False))

        def f(s, mp=mp):
            return " ".join(mp.get(t, t) for t in s.split())

        out.append(" , ".join(sorted(f(x) for x in prem)) + " |- " + f(right))
    return min(out)


def skeleton(thm):
    left, right = thm.split("|-")

    def f(s):
        return " ".join("A" if t in {"P", "Q", "R", "S"} else t for t in s.split())

    return " , ".join(sorted(f(x) for x in left.split(",") if x.strip())) + " |- " + f(right)


cache = ART / "source/training_exposure_keys.json.gz"
with gzip.open(cache, "rt") as f:
    exposure = set(json.load(f))
exposure_counts = {"base_training": len(exposure)}
for record in manifest["benchmark_metadata"]:
    path = Path(record["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    for r in rows(path):
        thm = r.get("thm")
        if thm is None:
            t = r["prompt"].split()
            j = t.index("SEQ")
            thm = " ".join(t[1:j]) + " |- " + " ".join(t[j + 1 : t.index("PRF")])
        exposure.add(canonical(thm))
for name in ["fresh_v1_safe", "ore_dev_pilot", "ore2_dev_pilot", "forward_deep_dev_pilot"]:
    for path in (ART / "data" / name).glob("*.jsonl"):
        for r in rows(path):
            exposure.add(r["structural_key"])
report = {
    "training_cache_hash_matches": hashlib.sha256(cache.read_bytes()).hexdigest()
    == manifest["training_exposure_cache_sha256"],
    "code_hashes_match": {
        k: hashlib.sha256((P / k).read_bytes()).hexdigest() == v
        for k, v in manifest["code_sha256"].items()
    },
    "splits": {},
    "scope": "metadata and content hashes only; no proof fields or candidate outcomes used",
}
sets = {}
for split in ["train", "dev", "test"]:
    data = list(rows(P / f"{split}.jsonl"))
    ct = collections.Counter()
    ks = set()
    gs = set()
    for r in data:
        k = canonical(r["thm"])
        g = skeleton(r["thm"])
        ct["canonical_mismatch"] += k != r["structural_key"]
        ct["skeleton_mismatch"] += g != r["skeleton"]
        ct["id_mismatch"] += hashlib.sha256(k.encode()).hexdigest()[:20] != r["theorem_id"]
        ct["exposure_match"] += k in exposure
        bucket = int(hashlib.sha256(("split61006:" + g).encode()).hexdigest()[:8], 16) % 100
        want = "train" if bucket < 70 else "dev" if bucket < 85 else "test"
        ct["split_mismatch"] += want != split or r["split"] != split
        ct["duplicate"] += k in ks
        ks.add(k)
        gs.add(g)
    sets[split] = (ks, gs)
    report["splits"][split] = {
        "rows": len(data),
        "families": dict(collections.Counter(r["family"] for r in data)),
        "unique_skeletons": len(gs),
        "file_hash_matches": hashlib.sha256((P / f"{split}.jsonl").read_bytes()).hexdigest()
        == manifest["file_sha256"][split],
        **dict(ct),
    }
report["cross_split_overlaps"] = {
    a + "_" + b: {
        "canonical": len(sets[a][0] & sets[b][0]),
        "skeleton": len(sets[a][1] & sets[b][1]),
    }
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]
}
Path(__file__).with_name("main_metadata_independent_audit.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report, indent=2))
