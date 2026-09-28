"""exp-00c step 2: filter the training candidates with the frozen checkpoint (pre-registered in
proposal.md). Run from code/ after make_split.py:

    uv run python ../experiments/ws3/exp-00c-lr-ei-cpu/make_pool.py

For every candidate: 8 samples at temperature 1.0 (max 400 new tokens, seed 0), judged by the
strict verifier. Keep prompts with 1 to 7 verified of 8 (mixed groups at k=8). Then cap the
length-6 survivors at half the number of length-7 survivors (random, seed 0) so the pool is
weighted toward the frontier. Writes pool.jsonl, filter_summary.json and
filter_dump.jsonl.gz next to this file. Touches only the training half.
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../code/ws3"))
import ndpath  # noqa: E402,F401
from data import read_prompt_file  # noqa: E402
from eval.evaluator import dump_rows, generate, judge  # noqa: E402
from hf_wrapper import load_pretrained_wrapper  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
K, TEMP, MAX_NEW, SEED = 8, 1.0, 400, 0


def main():
    cands = read_prompt_file(os.path.join(HERE, "candidates.jsonl.gz"))
    model, tok = load_pretrained_wrapper(device="cpu")
    t0 = time.time()
    comps = generate(
        model,
        tok,
        [r["prompt"] for r in cands],
        k=K,
        temperature=TEMP,
        max_new=MAX_NEW,
        batch_size=256,
        seed=SEED,
    )
    judged = judge([r["prompt"] for r in cands], comps, tok.tok_mode)
    secs = time.time() - t0
    dump = os.path.join(HERE, "filter_dump.jsonl")
    if os.path.exists(dump):
        os.remove(dump)
    dump_rows(dump, cands, judged, K, TEMP, "frozen_filter")
    kept, stats = {6: [], 7: [], 8: []}, {}
    for r, js in zip(cands, judged, strict=True):
        c = sum(j[1] for j in js)
        s = stats.setdefault(r["length"], {"n": 0, "zero": 0, "mixed": 0, "all": 0})
        s["n"] += 1
        s["zero" if c == 0 else "all" if c == K else "mixed"] += 1
        if 0 < c < K:
            kept[r["length"]].append(dict(r, frozen_filter_ok=c))
    rng = random.Random(SEED)
    cap6 = len(kept[7]) // 2
    six = rng.sample(kept[6], min(cap6, len(kept[6])))
    pool = six + kept[7] + kept[8]
    with open(os.path.join(HERE, "pool.jsonl"), "w") as f:
        for r in pool:
            f.write(
                json.dumps(
                    {k: r[k] for k in ("prompt", "length", "source", "key", "frozen_filter_ok")}
                )
                + "\n"
            )
    summary = {
        "k": K,
        "temperature": TEMP,
        "max_new": MAX_NEW,
        "seed": SEED,
        "secs": secs,
        "samples": K * len(cands),
        "by_length": stats,
        "kept_before_cap": {L: len(v) for L, v in kept.items()},
        "cap_len6": cap6,
        "pool": {L: sum(r["length"] == L for r in pool) for L in (6, 7, 8)},
        "pool_size": len(pool),
    }
    with open(os.path.join(HERE, "filter_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
