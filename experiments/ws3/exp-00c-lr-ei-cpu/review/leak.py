"""Reviewer leakage audit for exp-00c (own code). Run from code/:

    uv run python ../experiments/ws3/exp-00c-lr-ei-cpu/review/leak.py

Keys: the fork's canon_key (relabel atoms by first appearance on 'prem |- concl'; reproduced on
every fork record in recount.py), and the vendored nd_data.canonical_theorem (atom bijection +
premise permutation) as a second opinion. Also a coarse structural key (all atoms -> X, premises
sorted) and the generator's `schema` field, to see how much *template* overlap there is between
the training pool and the eval set (not a leak under the plan's definition; context for the
size of the lift). Test files: prompts only.
Writes leak.json.
"""

import gzip
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(EXP, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code", "ws3"))
import ndpath  # noqa: E402,F401

from nd_data import canonical_theorem  # noqa: E402

sys.path.insert(0, HERE)
from recount_lib import canon_key, read_jsonl, split_prompt  # noqa: E402

FD = ndpath.FORK_DATA


def skel(prompt):
    prems, con = split_prompt(prompt)
    f = lambda s: " ".join("X" if t in ("P", "Q", "R", "S") else t for t in s.split())  # noqa: E731
    return " , ".join(sorted(f(p) for p in prems)) + " |- " + f(con)


ARMS = ["lr3e-5", "lr1e-4", "lr3e-4", "ei"]
train_sets = {"pool.jsonl": [r["prompt"] for r in read_jsonl(os.path.join(EXP, "pool.jsonl"))],
              "candidates (frozen filter only)": [r["prompt"] for r in read_jsonl(os.path.join(EXP, "candidates.jsonl.gz"))]}
for a in ARMS:
    train_sets[f"{a}/prompts.jsonl"] = [r["prompt"] for r in read_jsonl(os.path.join(EXP, f"run_{a}", "prompts.jsonl"))]
    train_sets[f"{a}/completions prompts"] = sorted({r["prompt"] for r in read_jsonl(os.path.join(EXP, f"run_{a}", "completions.jsonl.gz"))})
for rnd in (1, 2, 3):
    train_sets[f"ei/sft_data round{rnd}"] = sorted({r["prompt"] for r in read_jsonl(os.path.join(EXP, "run_ei", "sft_data", f"round{rnd}.jsonl.gz"))})

tb = [r["prompt"] for r in read_jsonl(os.path.join(ndpath.TEXTBOOK72, "textbook_dev.jsonl"))] + [
    r["prompt"] for r in read_jsonl(os.path.join(ndpath.TEXTBOOK72, "textbook_train.jsonl"))]
eval_sets = {
    "eval_set.jsonl (200)": [r["prompt"] for r in read_jsonl(os.path.join(EXP, "eval_set.jsonl"))],
    "eval half (2272)": [r["prompt"] for r in read_jsonl(os.path.join(EXP, "split/rl_targets_eval.jsonl.gz"))],
    "heldout": [r["prompt"] for r in read_jsonl(os.path.join(FD, "heldout.jsonl"))],
    "transfer": [r["prompt"] for r in read_jsonl(os.path.join(FD, "transfer.jsonl"))],
    "validation_36": [r["prompt"] for r in read_jsonl(os.path.join(FD, "validation_36.jsonl"))],
    "validation_36_reference_proofs": [r["prompt"] for r in read_jsonl(os.path.join(FD, "validation_36_reference_proofs.jsonl")) if "prompt" in r],
    "textbook72": tb,
    "test_short_prompts": [r["prompt"] for r in read_jsonl(os.path.join(FD, "test_short_prompts.jsonl"))],
    "test_long_prompts": [r["prompt"] for r in read_jsonl(os.path.join(FD, "test_long_prompts.jsonl"))],
}
out = {"train_set_sizes": {k: len(v) for k, v in train_sets.items()},
       "eval_set_sizes": {k: len(v) for k, v in eval_sets.items()}, "overlap": {}}
ek = {k: ({canon_key(p) for p in v}, {canonical_theorem(p) for p in v}, set(v)) for k, v in eval_sets.items()}
for tn, tv in train_sets.items():
    tk = ({canon_key(p) for p in tv}, {canonical_theorem(p) for p in tv}, set(tv))
    out["overlap"][tn] = {en: {"canon_key": len(tk[0] & e[0]), "canonical_theorem": len(tk[1] & e[1]),
                               "exact_prompt": len(tk[2] & e[2])} for en, e in ek.items()}
for tn, d in out["overlap"].items():
    hits = {en: v for en, v in d.items() if any(v.values())}
    print(tn, "->", hits if hits else "no overlap")

# context: fixed pre-training set vs this run's eval set (inherited, not a run property)
train_pre = read_jsonl(os.path.join(FD, "train.jsonl.gz"))
es = eval_sets["eval_set.jsonl (200)"]
out["pretrain_train_vs_eval_set"] = {
    "canon_key": len({r["key"] for r in train_pre} & {canon_key(p) for p in es}),
    "canonical_theorem": len({canonical_theorem(r["prompt"]) for r in train_pre} & {canonical_theorem(p) for p in es}),
}
print("pretrain train.jsonl vs eval_set", out["pretrain_train_vs_eval_set"])

# template overlap between the training pool and the eval set
pool = read_jsonl(os.path.join(EXP, "pool.jsonl"))
rl = {r["prompt"]: r for r in read_jsonl(os.path.join(FD, "rl_targets.jsonl"))}
esr = read_jsonl(os.path.join(EXP, "eval_set.jsonl"))
pool_sk = {skel(r["prompt"]) for r in pool}
out["skeleton_overlap_pool_vs_eval_set"] = {
    L: f"{sum(skel(r['prompt']) in pool_sk for r in esr if int(r['L_true']) == L)}/{sum(int(r['L_true']) == L for r in esr)}"
    for L in (7, 8)}
print("eval_set problems whose atom-erased skeleton appears in pool:", out["skeleton_overlap_pool_vs_eval_set"])
pool_schema = Counter((rl[r["prompt"]]["schema"] if r["prompt"] in rl else "L6:train") for r in pool)
es_schema = Counter((int(r["L_true"]), r["schema"]) for r in esr)
out["pool_schema_counts"] = {str(k): v for k, v in pool_schema.most_common()}
out["eval_set_schema_counts"] = {f"L{k[0]}:{k[1]}": v for k, v in sorted(es_schema.items(), key=lambda x: (x[0][0], str(x[0][1])))}
print("pool schemas", out["pool_schema_counts"])
print("eval_set schemas", out["eval_set_schema_counts"])
json.dump(out, open(os.path.join(HERE, "leak.json"), "w"), indent=1)
