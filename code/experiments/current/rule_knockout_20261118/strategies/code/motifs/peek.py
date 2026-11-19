# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
import gzip, json, collections, re

D = "/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014/data/"
seen = collections.Counter()
ex = {}
for i, l in enumerate(gzip.open(D + "pretrain.jsonl.gz", "rt")):
    r = json.loads(l)
    for s in r["proof"].split(" ; "):
        m = re.search(r": (\w+)", s)
        if m:
            seen[m.group(1)] += 1
            if m.group(1) not in ex and len(r["proof"]) < 500:
                ex[m.group(1)] = r["proof"]
print(seen)
for k, v in ex.items():
    print(k, "::", v, "\n")
import itertools

rl = [json.loads(l) for l in open(D + "rl.jsonl")]
print(len(rl), sum("proof" in r and bool(r["proof"]) for r in rl))
print(collections.Counter(r.get("band") for r in rl))
print(collections.Counter(r.get("src") for r in rl))
print(collections.Counter((r.get("src"), bool(r.get("proof"))) for r in rl))
print(rl[5].keys())
dev = [json.loads(l) for l in open(D + "dev.jsonl")]
print(len(dev), dev[0].keys(), collections.Counter(r.get("src") for r in dev))
