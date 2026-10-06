"""Complex-formula disjunction elimination with verified schematic certificates."""

import argparse
import collections
import gzip
import hashlib
import json
import random
import time
from pathlib import Path

from data import ROOT, SOURCE, Gen, key, skeleton, verify_text
from exclusions import benchmark_exclusions
from gen import fstr  # noqa: E402 — pinned upstream source is added above


def reference(leaves, goal):
    counter = 0
    lines = []

    def add(depth, formula, rule, refs=()):
        nonlocal counter
        counter += 1
        lines.append(
            f"N{counter} "
            + ("| " * depth)
            + fstr(formula)
            + " : "
            + rule
            + (" " + " ".join(f"N{i}" for i in refs) if refs else "")
            + " ;"
        )
        return counter

    def tree(xs):
        return xs[0] if len(xs) == 1 else ("or", tree(xs[: len(xs) // 2]), tree(xs[len(xs) // 2 :]))

    disjunction = tree(leaves)
    premises = [disjunction] + [("imp", x, goal) for x in leaves]
    root = add(0, disjunction, "PR")
    maps = [add(0, p, "PR") for p in premises[1:]]

    def eliminate(xs, source, depth, start):
        if len(xs) == 1:
            return add(depth, goal, "IMPE", (maps[start], source))
        mid = len(xs) // 2
        left = xs[:mid]
        right = xs[mid:]
        la = add(depth + 1, tree(left), "AS")
        lb = eliminate(left, la, depth + 1, start)
        ra = add(depth + 1, tree(right), "AS")
        rb = eliminate(right, ra, depth + 1, start + mid)
        return add(depth, goal, "ORE", (source, la, lb, ra, rb))

    eliminate(leaves, root, 0, 0)
    thm = " , ".join(fstr(p) for p in premises) + " |- " + fstr(goal)
    prompt = "THM " + " , ".join(fstr(p) for p in premises) + " SEQ " + fstr(goal) + " PRF"
    proof = " ".join(lines) + " QED"
    return {
        "thm": thm,
        "prompt": prompt,
        "proof": proof,
        "n_lines": counter,
        "n_prem": len(premises),
        "schema": "ore_" + str(len(leaves)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=61008)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--leaves", type=int, choices=[0, 2, 4], default=0)
    p.add_argument("--min-tokens", type=int, default=80)
    p.add_argument("--max-tokens", type=int, default=180)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exposure, meta = benchmark_exclusions()
    t0 = time.time()
    cache = (
        ROOT / "artifacts/learned_compute_allocation_20261006/source/training_exposure_keys.json.gz"
    )
    if cache.exists():
        with gzip.open(cache, "rt") as f:
            exposure.update(json.load(f))
    else:
        training = set()
        for path in [
            ROOT / "artifacts/learned_compute_allocation_20261006/source/train_k12.jsonl.gz",
            SOURCE / "data/ladder/rl_targets.jsonl",
        ]:
            with gzip.open(path, "rt") if str(path).endswith(".gz") else open(path) as f:
                for line in f:
                    training.add(key(json.loads(line)["thm"]))
        with gzip.open(cache, "wt") as f:
            json.dump(sorted(training), f)
        exposure.update(training)
    (out / "generator.py").write_bytes(Path(__file__).read_bytes())
    for module in ["data.py", "exclusions.py"]:
        (out / module).write_bytes((Path(__file__).parent / module).read_bytes())
    rng = random.Random(a.seed)
    g = Gen(rng, fdepth=3)
    g.bot_p = 0.0
    rows = []
    counts = collections.Counter()
    while len(rows) < a.n:
        counts["draws"] += 1
        nleaf = a.leaves or rng.choice([2, 4])
        forms = [g.rf(rng.choice([2, 3])) for _ in range(nleaf + 1)]
        if len(set(forms)) != len(forms):
            counts["duplicates"] += 1
            continue
        r = reference(forms[:-1], forms[-1])
        if not a.min_tokens <= len(r["thm"].split()) <= a.max_tokens:
            counts["length_filtered"] += 1
            continue
        k = key(r["thm"])
        sk = skeleton(r["thm"])
        h = int(hashlib.sha256(("split61006:" + sk).encode()).hexdigest()[:8], 16) % 100
        split = "train" if h < 70 else "dev" if h < 85 else "test"
        if split != a.split:
            counts["wrong_split"] += 1
            continue
        if k in exposure:
            counts["exposed_or_duplicate"] += 1
            continue
        ok, why, _ = verify_text(r["prompt"] + " " + r["proof"])
        if not ok:
            raise RuntimeError(why + " " + r["proof"])
        exposure.add(k)
        r.update(
            theorem_id=hashlib.sha256(k.encode()).hexdigest()[:20],
            structural_key=k,
            skeleton=sk,
            split=split,
        )
        rows.append(r)
    (out / f"{a.split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    m = {
        "args": vars(a),
        "template": (
            "leafcase elimination; eachleafandgoalrandomformuladepth2/3;certificates8or18lines"
        ),
        "filter": (
            "theoremtokens,uniqueleaf/goalformulas;predefinedhashsplit;canonic"
            "altraining/benchmarkexposureexcluded"
        ),
        "counts": dict(counts),
        "seconds": time.time() - t0,
        "benchmark_exclusions": meta,
        "training_exposure_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m), flush=True)


if __name__ == "__main__":
    main()
