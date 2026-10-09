#!/usr/bin/env python3
"""Are attempts made with failed proofs in context more varied than independent samples?

  python diversity.py ARM [ARM ...]      # reads artifacts/retry_ei_20261008/<arm>_s{0,1,2}

Uses the failed-attempt texts in eval_dev_episodes.jsonl. Two proofs are "the same" when their
texts match after renumbering names by first appearance (the model picks a random name offset).
The independent baseline is the failed FIRST attempts of a theorem's 8 episodes: those are
independent samples from the same model, with nothing in context.

Every rate is computed per theorem and then averaged over the theorems that have both kinds of
pair, so the two sides weight the same theorems equally.
"""

import difflib
import itertools
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "artifacts", "retry_ei_20261008"))


def canon(text):
    names = {}
    return re.sub(r"\bn(\d+)", lambda m: names.setdefault(m.group(0), f"n{len(names) + 1}"), text)


def similarity(a, b):
    return difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()


def shared_prefix(a, b):
    """Fraction of the shorter proof's tokens before the two first differ."""
    x, y = a.split(), b.split()
    n = next((i for i, (p, q) in enumerate(zip(x, y, strict=False)) if p != q), min(len(x), len(y)))
    return n / max(min(len(x), len(y)), 1)


def mean(xs):
    return st.mean(xs) if xs else float("nan")


def run(path):
    same_in, same_x, sim_in, sim_x, pre_in, pre_x = [], [], [], [], [], []
    # on unsolved theorems: attempt k+2 repeats ANY earlier attempt of its episode, against the
    # (k+2)-th independent first attempt repeating any of the k+1 before it
    seen_in, seen_x = [[] for _ in range(7)], [[] for _ in range(7)]
    distinct_in, distinct_x, n_unsolved = [], [], 0
    with open(os.path.join(path, "eval_dev_episodes.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    for r in rows:
        eps = [[canon(t) for t in e["texts"]] for e in r["episodes"]]
        firsts = [e[0] for e in eps if e]
        pairs_in = [(e[0], e[1]) for e in eps if len(e) >= 2]
        pairs_x = list(itertools.combinations(firsts, 2))
        if pairs_in and pairs_x:
            same_in.append(mean([a == b for a, b in pairs_in]))
            same_x.append(mean([a == b for a, b in pairs_x]))
            diff_in = [(a, b) for a, b in pairs_in if a != b][:4]
            diff_x = [(a, b) for a, b in pairs_x if a != b][:4]
            if diff_in and diff_x:
                sim_in.append(mean([similarity(a, b) for a, b in diff_in]))
                sim_x.append(mean([similarity(a, b) for a, b in diff_x]))
                pre_in.append(mean([shared_prefix(a, b) for a, b in diff_in]))
                pre_x.append(mean([shared_prefix(a, b) for a, b in diff_x]))
        # theorems no episode solved: every episode ran its 8 attempts, so 8 attempts in one
        # episode can be set against the 8 independent first attempts
        if all(e["proof"] is None and len(e["texts"]) == 8 for e in r["episodes"]):
            n_unsolved += 1
            distinct_in.append(mean([len(set(e)) for e in eps]))
            distinct_x.append(len(set(firsts)))
            for k in range(1, 8):
                seen_in[k - 1].append(mean([e[k] in e[:k] for e in eps]))
                seen_x[k - 1].append(firsts[k] in firsts[:k])
    return {
        "theorems_with_pairs": len(same_in),
        "identical_in_episode": mean(same_in),
        "identical_independent": mean(same_x),
        "similarity_when_different_in_episode": mean(sim_in),
        "similarity_when_different_independent": mean(sim_x),
        "shared_prefix_when_different_in_episode": mean(pre_in),
        "shared_prefix_when_different_independent": mean(pre_x),
        "repeats_earlier_in_episode": [mean(x) for x in seen_in],
        "repeats_earlier_independent": [mean(x) for x in seen_x],
        "unsolved_theorems": n_unsolved,
        "distinct_of_8_in_episode": mean(distinct_in),
        "distinct_of_8_independent": mean(distinct_x),
    }


def main():
    out = {}
    for arm in sys.argv[1:]:
        for seed in range(3):
            path = os.path.join(ART, f"{arm}_s{seed}")
            if os.path.exists(os.path.join(path, "eval_dev_episodes.jsonl")):
                out[f"{arm}_s{seed}"] = run(path)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
