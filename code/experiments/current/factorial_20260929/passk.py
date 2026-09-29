# pyright: reportMissingImports=false
"""Per-problem success counts for pass@k curves, sampled from the final model on the pod.

  holdout250  the 250 transfer-holdout problems nd-rl-2's score_holdout.py uses (sorted by
              sha1("passk:" + key)); all have L_true >= 7, none is trained or selected on
  textbook72  58 dev + 14 train textbook problems (Dmitry's cap-comparison eval data, copied
              with its MANIFEST.json); never trained on. reference_lines are upper bounds.

N_SAMPLES samples per problem at the harness's eval temperature and max_new, drawn with
fast_sample.generate (the fork's sampler plus compaction: same distribution) and judged by the
harness's judge (Lean AND nd_verify for lean_seq, nd_verify for abs). pass@k is computed
offline from (n, n_ok) with the unbiased estimator.
"""

import hashlib
import os
import time

import fast_sample
import harness_frozen as H

HERE = os.path.dirname(os.path.abspath(__file__))
N_SAMPLES, N_HOLDOUT = 256, 250


def pools():
    transfer = H.read(f"{H.FORK}/data/ladder/transfer.jsonl")
    rank = lambda r: hashlib.sha1(f"passk:{r['key']}".encode()).hexdigest()  # noqa: E731
    textbook = [
        {**r, "n_lines": r.get("reference_lines"), "split": split}
        for split in ("dev", "train")
        for r in H.read(f"{HERE}/textbook72/textbook_{split}.jsonl")
    ]
    return {
        "holdout250": sorted(H.holdout_split(transfer), key=rank)[:N_HOLDOUT],
        "textbook72": textbook,
    }


def score(a, tok, model):
    n = N_SAMPLES if not a.smoke else 2
    out = {}
    for name, pool in pools().items():
        pool = pool if not a.smoke else pool[:4]
        t0 = time.time()
        prompts = [r["prompt"] for r in pool for _ in range(n)]
        flat = fast_sample.generate(
            model, tok, prompts, H.TEMP, a.max_new, H.SAMPLE_BATCH, a.seed * 1000 + 999
        )
        rows = H.judge(pool, [flat[i * n : (i + 1) * n] for i in range(len(pool))], "n_lines")
        out[name] = {
            "n_samples": n,
            "secs": time.time() - t0,
            "problems": {
                r["name"]: {"n_lines": r["n_lines"], "n_ok": r["n_ok"], "split": p.get("split")}
                for r, p in zip(rows, pool, strict=True)
            },
        }
        H.log(
            f"passk {name}: solved {sum(r['solved'] for r in rows)}/{len(pool)} at k={n}, "
            f"{time.time() - t0:.0f}s"
        )
    return out
