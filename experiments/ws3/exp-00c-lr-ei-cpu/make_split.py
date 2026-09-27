"""exp-00c step 1 (no model): split rl_targets by renaming class, pick the eval set and the
training candidates. Deterministic; run from code/:

    uv run python ../experiments/ws3/exp-00c-lr-ei-cpu/make_split.py

Writes, next to this file:
    split/rl_targets_train.jsonl.gz   rl_targets records whose key hashes to the train half
    split/rl_targets_eval.jsonl.gz    the other half; never trained on
    eval_set.jsonl                    stride of the eval half: 160 at L_true 7, 40 at 8
    candidates.jsonl.gz               training candidates before the frozen-model filter: every
                                      train-half prompt at L_true 7 and 8, plus 300 length-6
                                      prompts from train.jsonl (seed 0)
Every candidate is leak-checked against the evaluation sets, the 72 textbook problems, the test
prompt files, and the eval half.
"""

import gzip
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../code/ws3"))
import ndpath  # noqa: E402,F401
from data import (  # noqa: E402
    assert_no_leak,
    banned_keys,
    key_of,
    load_pool,
    stride,
    test_file_keys,
)

HERE = os.path.dirname(os.path.abspath(__file__))
N_EVAL = {7: 160, 8: 40}
N_LEN6 = 300


def half(key: str) -> str:
    return "train" if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 == 0 else "eval"


def write(path, recs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "wt") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def main():
    rl = load_pool("rl_targets")
    for r in rl:
        r["key"] = key_of(r)
        r["length"] = int(r["L_true"])
        r["source"] = "rl_targets"
    tr = [r for r in rl if half(r["key"]) == "train"]
    ev = [r for r in rl if half(r["key"]) == "eval"]
    assert not ({r["key"] for r in tr} & {r["key"] for r in ev})
    write(os.path.join(HERE, "split/rl_targets_train.jsonl.gz"), tr)
    write(os.path.join(HERE, "split/rl_targets_eval.jsonl.gz"), ev)

    eval_set = []
    for L, n in N_EVAL.items():
        eval_set += stride([r for r in ev if r["length"] == L], n)
    write(os.path.join(HERE, "eval_set.jsonl"), eval_set)

    banned = banned_keys() | test_file_keys() | {r["key"] for r in ev}
    cands = [r for r in tr if r["length"] in (7, 8) and r["key"] not in banned]
    six = [r for r in load_pool("train") if int(r["n_lines"]) == 6 and key_of(r) not in banned]
    rng = random.Random(0)
    for r in rng.sample(six, N_LEN6):
        cands.append(dict(r, key=key_of(r), length=6, source="train"))
    assert_no_leak(cands, banned)
    write(os.path.join(HERE, "candidates.jsonl.gz"), cands)

    def count(recs):
        return {L: sum(r["length"] == L for r in recs) for L in sorted({r["length"] for r in recs})}

    print("train half", len(tr), count(tr))
    print("eval half", len(ev), count(ev))
    print("eval set", len(eval_set), count(eval_set))
    print("candidates", len(cands), count(cands))
    print(
        "test-file classes dropped from train-half 7/8:",
        sum(r["length"] in (7, 8) and r["key"] in test_file_keys() for r in tr),
    )


if __name__ == "__main__":
    main()
