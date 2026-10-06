"""Frozen 50/50 structural challenge mixture, generated without model-outcome filtering."""

import argparse
import collections
import datetime
import gzip
import hashlib
import json
import random
import time
from pathlib import Path

from data import ROOT, Fail, Gen, key, skeleton, verify_text
from exclusions import benchmark_exclusions
from ore_data import reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--train", type=int, default=1600)
    parser.add_argument("--dev", type=int, default=320)
    parser.add_argument("--test", type=int, default=400)
    args = parser.parse_args()
    assert all(n % 2 == 0 for n in [args.train, args.dev, args.test])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modules = ["main_data.py", "data.py", "exclusions.py", "ore_data.py"]
    for module in modules:
        (out / module).write_bytes((Path(__file__).parent / module).read_bytes())
    exposure, benchmarks = benchmark_exclusions()
    cache = (
        ROOT / "artifacts/learned_compute_allocation_20261006/source/training_exposure_keys.json.gz"
    )
    with gzip.open(cache, "rt") as stream:
        exposure.update(json.load(stream))
    # Development pilot cases remain pilot evidence and never reappear in the main panel.
    for name in ["fresh_v1_safe", "ore_dev_pilot", "ore2_dev_pilot", "forward_deep_dev_pilot"]:
        for path in (ROOT / "artifacts/learned_compute_allocation_20261006/data" / name).glob(
            "*.jsonl"
        ):
            for line in path.read_text().splitlines():
                exposure.add(json.loads(line)["structural_key"])
    targets = {"train": args.train // 2, "dev": args.dev // 2, "test": args.test // 2}
    rows = []
    metrics = {}
    t0 = time.time()
    for family, seed in [("forward", 61011), ("ore2", 61012)]:
        counts = collections.Counter()
        accepted = collections.Counter()
        rng = random.Random(seed)
        gen = Gen(rng, max_depth=5, fdepth=4 if family == "forward" else 3)
        gen.ore_boxes = True
        gen.ore_steps = 3
        gen.strict = True
        gen.bot_p = 0.0
        while any(accepted[split] < n for split, n in targets.items()):
            counts["draws"] += 1
            if counts["draws"] > 2_000_000:
                raise RuntimeError("bounded corpus generation did not fill targets")
            try:
                if family == "forward":
                    row = gen.forward(rng.randint(15, 35), rng.choice([1, 2, 3]))
                    if not 12 <= row["n_lines"] <= 40 or len(row["thm"].split()) < 60:
                        counts["complexity_filter"] += 1
                        continue
                else:
                    forms = [gen.rf(rng.choice([2, 3])) for _ in range(3)]
                    if len(set(forms)) != 3:
                        counts["degenerate"] += 1
                        continue
                    row = reference(forms[:2], forms[2])
                    if not 80 <= len(row["thm"].split()) <= 180:
                        counts["complexity_filter"] += 1
                        continue
            except (Fail, RecursionError):
                counts["generator_fail"] += 1
                continue
            canonical = key(row["thm"])
            group = skeleton(row["thm"])
            bucket = int(hashlib.sha256(("split61006:" + group).encode()).hexdigest()[:8], 16) % 100
            split = "train" if bucket < 70 else "dev" if bucket < 85 else "test"
            if accepted[split] >= targets[split]:
                counts["split_full"] += 1
                continue
            if canonical in exposure:
                counts["exposed_or_duplicate"] += 1
                continue
            ok, why, _ = verify_text(row["prompt"] + " " + row["proof"])
            if not ok:
                raise RuntimeError("reference certificate failed strict ND: " + why)
            exposure.add(canonical)
            accepted[split] += 1
            row.update(
                theorem_id=hashlib.sha256(canonical.encode()).hexdigest()[:20],
                structural_key=canonical,
                skeleton=group,
                split=split,
                family=family,
            )
            rows.append(row)
        metrics[family] = {"seed": seed, "counts": dict(counts), "splits": dict(accepted)}
    rng = random.Random(61013)
    rng.shuffle(rows)
    for split in targets:
        part = [row for row in rows if row["split"] == split]
        (out / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in part))
        for family in ["forward", "ore2"]:
            directory = out / family
            directory.mkdir(exist_ok=True)
            (directory / f"{split}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in part if row["family"] == family)
            )
    groups = {
        split: {row["skeleton"] for row in rows if row["split"] == split} for split in targets
    }
    assert (
        not groups["train"] & groups["dev"]
        and not groups["train"] & groups["test"]
        and not groups["dev"] & groups["test"]
    )
    config = {
        "written_on": "2026-10-06",
        "frozen_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "args": vars(args),
        "mixture_weights": {"forward": 0.5, "ore2": 0.5},
        "sampler": {
            "checkpoint_seed": 0,
            "temperature": 0.8,
            "k": 8,
            "max_action": 256,
            "max_steps": 48,
            "history": "fresh state context for every independent attempt",
        },
        "chooser_context": 8192,
        "training_seeds": [0, 1, 2],
        "forward": {
            "max_depth": 5,
            "fdepth": 4,
            "steps": [15, 35],
            "initial_premises": [1, 2, 3],
            "ore_boxes": True,
            "ore_steps": 3,
            "strict": True,
            "bot_p": 0,
            "reference_lines": [12, 40],
            "min_theorem_tokens": 60,
        },
        "ore2": {
            "leaves": 2,
            "subformula_depth_choices": [2, 3],
            "max_depth": 5,
            "fdepth": 3,
            "bot_p": 0,
            "theorem_tokens": [80, 180],
            "verified_reference_lines": 8,
        },
        "split": (
            "SHA256 split61006+atom-erased sorted-premise connective "
            "skeleton;train<70/dev<85/test>=85 modulo100"
        ),
        "exclusion": (
            "atom-bijection/premise-permutation canonical training matches, public "
            "benchmarks, and all development pilot/old corpus theorem cases"
        ),
        "gold_reference_use": (
            "private target-validity evidence only; never generator/chooser/RL supervision"
        ),
        "benchmark_metadata": benchmarks,
        "generation": metrics,
        "seconds": time.time() - t0,
        "distinct_groups": {s: len(g) for s, g in groups.items()},
        "file_sha256": {
            s: hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest() for s in targets
        },
        "code_sha256": {m: hashlib.sha256((out / m).read_bytes()).hexdigest() for m in modules},
        "training_exposure_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)


if __name__ == "__main__":
    main()
