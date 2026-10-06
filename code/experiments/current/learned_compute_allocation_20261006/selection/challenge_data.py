"""Development-only deeper-formula challenge, independent of model outcomes."""

import argparse
import collections
import gzip
import hashlib
import json
import random
import time
from pathlib import Path

from data import ROOT, SOURCE, Fail, Gen, key, sample_one, skeleton, verify_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=61007)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["goal", "forward"], default="goal")
    p.add_argument("--min-tokens", type=int, default=80)
    p.add_argument("--max-draws", type=int, default=100000)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from exclusions import benchmark_exclusions

    exposure, _ = benchmark_exclusions()
    cache = (
        ROOT / "artifacts/learned_compute_allocation_20261006/source/training_exposure_keys.json.gz"
    )
    if cache.exists():
        with gzip.open(cache, "rt") as f:
            exposure.update(json.load(f))
    for path in (
        []
        if cache.exists()
        else [
            ROOT / "artifacts/learned_compute_allocation_20261006/source/train_k12.jsonl.gz",
            SOURCE / "data/ladder/rl_targets.jsonl",
        ]
    ):
        with gzip.open(path, "rt") if str(path).endswith(".gz") else open(path) as f:
            for line in f:
                exposure.add(key(json.loads(line)["thm"]))
    for path in (ROOT / "artifacts/learned_compute_allocation_20261006/data/fresh_v1").glob(
        "*.jsonl"
    ):
        for line in path.read_text().splitlines():
            exposure.add(json.loads(line)["structural_key"])
    rng = random.Random(a.seed)
    g = Gen(rng, max_depth=5, fdepth=4)
    g.ore_boxes = True
    g.ore_steps = 3
    g.strict = True
    g.bot_p = 0.0
    counts = collections.Counter()
    rows = []
    t0 = time.time()
    (out / "generator.py").write_bytes(Path(__file__).read_bytes())
    for module in ["data.py", "exclusions.py"]:
        (out / module).write_bytes((Path(__file__).parent / module).read_bytes())
    while len(rows) < a.n and counts["draws"] < a.max_draws:
        counts["draws"] += 1
        try:
            r = (
                g.forward(rng.randint(15, 35), rng.choice([1, 2, 3]))
                if a.mode == "forward"
                else sample_one(
                    g,
                    rng,
                    knobs={
                        "goal_only": True,
                        "strict": True,
                        "ore_boxes": True,
                        "ore_steps": 3,
                        "budgets": [4, 5, 6],
                        "gdepth": [4, 4, 5],
                    },
                )
            )
        except (Fail, RecursionError):
            counts["gen_fail"] += 1
            continue
        if not 12 <= r["n_lines"] <= 40 or len(r["thm"].split()) < a.min_tokens:
            counts["complexity_filter"] += 1
            continue
        k = key(r["thm"])
        sk = skeleton(r["thm"])
        h = int(hashlib.sha256(("split61006:" + sk).encode()).hexdigest()[:8], 16) % 100
        if not 70 <= h < 85:
            counts["nondev_group"] += 1
            continue
        if k in exposure:
            counts["exposure_or_duplicate"] += 1
            continue
        ok, why, _ = verify_text(r["prompt"] + " " + r["proof"])
        if not ok:
            counts["verify_fail"] += 1
            continue
        exposure.add(k)
        r.update(
            theorem_id=hashlib.sha256(k.encode()).hexdigest()[:20],
            structural_key=k,
            skeleton=sk,
            split="dev",
        )
        rows.append(r)
    (out / "dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = {
        "args": vars(a),
        "knobs": {
            "constructor_max_depth": 5,
            "constructor_fdepth": 4,
            "goal_only": a.mode == "goal",
            "strict": True,
            "ore_boxes": True,
            "ore_steps": 3,
            "budgets": [4, 5, 6] if a.mode == "goal" else None,
            "gdepth": [4, 4, 5] if a.mode == "goal" else None,
            "forward_steps": [15, 35] if a.mode == "forward" else None,
            "initial_premises": [1, 2, 3] if a.mode == "forward" else None,
        },
        "filter": {
            "reference_lines": [12, 40],
            "min_theorem_tokens": a.min_tokens,
            "split": "preexisting hash-dev groups only",
        },
        "counts": dict(counts),
        "seconds": time.time() - t0,
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
