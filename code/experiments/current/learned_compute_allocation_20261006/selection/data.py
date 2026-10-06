"""Fresh structural split with atom-bijection and premise-order exposure exclusion."""

import argparse
import collections
import gzip
import hashlib
import itertools
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "artifacts/learned_compute_allocation_20261006/source/nd-takehome-dan"
sys.path.insert(0, str(SOURCE))
from gen import Fail, Gen, sample_one  # noqa: E402 — pinned upstream source is added above
from nd_verify import verify_text  # noqa: E402 — pinned upstream source is added above


def key(thm):
    prem, conclusion = thm.split("|-")
    premises = [p.strip() for p in prem.split(",") if p.strip()]
    atoms = sorted(set(thm.split()) & set("PQRS"))
    variants = []
    for names in itertools.permutations("PQRS"[: len(atoms)]):
        mapping = dict(zip(atoms, names, strict=True))

        def canon(text, mapping=mapping):
            return " ".join(mapping.get(t, t) for t in text.split())

        variants.append(" , ".join(sorted(canon(p) for p in premises)) + " |- " + canon(conclusion))
    return min(variants)


def skeleton(thm):
    prem, concl = thm.split("|-")

    def f(s):
        return " ".join("A" if t in "PQRS" else t for t in s.split())

    return " , ".join(sorted(f(p) for p in prem.split(",") if p.strip())) + " |- " + f(concl)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--n", type=int, default=2000)
    a.add_argument("--seed", type=int, default=61006)
    a.add_argument("--out", required=True)
    args = a.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from exclusions import benchmark_exclusions

    exposure, benchmark_meta = benchmark_exclusions()
    sources = [
        ROOT / "artifacts/learned_compute_allocation_20261006/source/train_k12.jsonl.gz",
        SOURCE / "data/ladder/rl_targets.jsonl",
    ]
    source_meta = list(benchmark_meta)
    for path in sources:
        opener = gzip.open if str(path).endswith(".gz") else open
        n = 0
        with opener(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                exposure.add(key(r["thm"]))
                n += 1
        source_meta.append(
            {"path": str(path), "rows": n, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    rng = random.Random(args.seed)
    g = Gen(rng, max_depth=4, fdepth=3)
    seen = set()
    counts = collections.Counter()
    rows = []
    t0 = time.time()
    while len(rows) < args.n:
        counts["draws"] += 1
        try:
            r = sample_one(
                g,
                rng,
                long=True,
                knobs={"ladder": True, "strict": True, "ore_boxes": True, "ore_steps": 3},
            )
        except (Fail, RecursionError):
            counts["generator_fail"] += 1
            continue
        if not 8 <= r["n_lines"] <= 24:
            counts["length_excluded"] += 1
            continue
        k = key(r["thm"])
        sk = skeleton(r["thm"])
        if k in exposure:
            counts["exposed"] += 1
            continue
        if k in seen:
            counts["duplicate"] += 1
            continue
        ok, _, _ = verify_text(r["prompt"] + " " + r["proof"])
        if not ok:
            counts["verify_fail"] += 1
            continue
        seen.add(k)
        h = int(hashlib.sha256(("split61006:" + sk).encode()).hexdigest()[:8], 16) % 100
        split = "train" if h < 70 else "dev" if h < 85 else "test"
        r.update(
            theorem_id=hashlib.sha256(k.encode()).hexdigest()[:20],
            structural_key=k,
            skeleton=sk,
            split=split,
        )
        rows.append(r)
    for split in ["train", "dev", "test"]:
        with (out / f"{split}.jsonl").open("w") as f:
            for r in rows:
                if r["split"] == split:
                    f.write(json.dumps(r) + "\n")
    meta = {
        "config": vars(args),
        "sources": source_meta,
        "unique_exposure": len(exposure),
        "counts": dict(counts),
        "splits": dict(collections.Counter(r["split"] for r in rows)),
        "seconds": time.time() - t0,
        "method": (
            "exclude atom-bijection/premise-permutation equivalence; group spl"
            "its by connective skeleton with all atoms erased"
        ),
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
