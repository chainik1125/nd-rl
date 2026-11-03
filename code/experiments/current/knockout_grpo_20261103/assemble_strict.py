"""ko_strict: the ko corpus without records whose narration plans the classical detour.

    python -m code.experiments.current.knockout_grpo_20261103.assemble_strict plan
    # e64 traces of OUT/corpora/ko_strict/new_refill_candidates.jsonl at 79749ce, then:
    python -m code.experiments.current.knockout_grpo_20261103.assemble_strict write

The trace-leak check found the detour goal `Classical.byContradiction |- ( ~ ( ~` (prove G by DN
of a ¬¬G subgoal; σ's plan) in the narration of 948 ko records, though no ko proof applies DN to a
NEGI-built line. ko_strict removes every ko record whose e64 or e256 narration contains it and
refills each slot in place with a record of the same K12 generator length from Dan's eligible pool
(`refill_pool.py`), not already used, whose search proof is σ⁺- and lemma-route-free (Dan's
predicates) and whose e256 narration lacks the detour goal; e64 traces are then generated for the
draws and any draw whose e64 narration has it is replaced (seeded). Seed 1 (ko's refill used 0).
"""

from __future__ import annotations

import collections
import json
import random
import sys

from .common import OUT, jsonl

PAT = "Classical.byContradiction |- ( ~ ( ~"
SEED = 1


def has_detour(rec: dict) -> bool:
    return any(PAT in st.get("think", "") for st in rec["steps"])


def plan() -> None:
    rng = random.Random(SEED)
    P = json.loads((OUT / "corpora" / "plan.json").read_text())
    ko = P["arms"]["ko"]
    k12 = list(jsonl("/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl"))
    cand = {r["id"]: r for r in jsonl(OUT / "refill" / "candidates.jsonl")}

    def gen_len(i: str) -> int:
        return k12[int(i.split("_")[1])]["n_lines"] if i.startswith("k12_") else cand[i]["n_lines"]

    bad = set()
    for lv in ("e64", "e256"):
        for r in jsonl(OUT / "corpora" / "ko" / f"{lv}.jsonl"):
            if has_detour(r):
                bad.add(r["id"])
    need = collections.Counter(gen_len(i) for i in bad)
    used = set(i for i in ko if i.startswith("ko_"))
    lab = {r["id"]: r for r in jsonl(OUT / "sigma" / "cand.labels.jsonl")}
    ok = {
        i for i, x in lab.items() if x.get("parse") and not x["sigma_plus"] and not x["lemma_route"]
    }
    clean256 = set()
    for r in jsonl(OUT / "refill" / "cand_e256.jsonl"):
        if r["id"] in ok and r["id"] not in used and not has_detour(r):
            clean256.add(r["id"])
    by_len = collections.defaultdict(list)
    for i in sorted(clean256, key=lambda c: (c.split("_")[1], int(c.split("_")[2]))):
        by_len[cand[i]["n_lines"]].append(i)
    draws = {}
    for L, n in sorted(need.items()):
        assert len(by_len[L]) >= 3 * n, (L, n, len(by_len[L]))
        pool = rng.sample(by_len[L], 3 * n)  # first n are the draws, the rest spares for e64 checks
        draws[L] = pool
    d = OUT / "corpora" / "ko_strict"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "new_refill_candidates.jsonl", "w") as f:
        for L in draws:
            for i in draws[L]:
                r = dict(cand[i])
                r["name"] = i
                f.write(json.dumps(r) + "\n")
    rep = {"removed": len(bad), "removed_per_gen_length": dict(sorted(need.items())),
           "removed_refill_records": sum(1 for i in bad if i.startswith("ko_")),
           "candidates_clean_e256": len(clean256), "seed": SEED}  # fmt: skip
    (d / "plan.json").write_text(
        json.dumps({"report": rep, "bad": sorted(bad), "draws": draws}, indent=0)
    )
    print(json.dumps(rep, indent=1))


def write() -> None:
    d = OUT / "corpora" / "ko_strict"
    P = json.loads((d / "plan.json").read_text())
    k12 = list(jsonl("/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl"))
    cand = {r["id"]: r for r in jsonl(OUT / "refill" / "candidates.jsonl")}
    e64new = {r["id"]: r for r in jsonl(d / "new_refill_e64.jsonl")}
    bad = set(P["bad"])
    pick = {}
    for L, pool in P["draws"].items():
        good = [i for i in pool if i in e64new and not has_detour(e64new[i])]
        n = len(pool) // 3
        assert len(good) >= n, (L, n, len(good))
        pick[int(L)] = good[:n]
    ko = json.loads((OUT / "corpora" / "plan.json").read_text())["arms"]["ko"]

    def gen_len(i: str) -> int:
        return k12[int(i.split("_")[1])]["n_lines"] if i.startswith("k12_") else cand[i]["n_lines"]

    it = {L: iter(v) for L, v in pick.items()}
    out = [next(it[gen_len(i)]) if i in bad else i for i in ko]
    for v in it.values():
        assert next(v, None) is None
    assert len(out) == len(ko) == len(set(out))
    new = set(out) - set(ko)
    srcs = {
        "e256": [OUT / "corpora" / "ko" / "e256.jsonl", OUT / "refill" / "cand_e256.jsonl"],
        "e64": [OUT / "corpora" / "ko" / "e64.jsonl", d / "new_refill_e64.jsonl"],
        "plain_e256": [OUT / "corpora" / "ko" / "plain_e256.jsonl", OUT / "refill" / "cand_plain_e256.jsonl"],  # noqa: E501
    }  # fmt: skip
    keep = set(out)
    for name, paths in srcs.items():
        m = {}
        for p in paths:
            with open(p) as fh:
                for line in fh:
                    i = json.loads(line)["id"]
                    if i in keep and i not in m:  # first source wins
                        m[i] = line if line.endswith("\n") else line + "\n"
        with open(d / f"{name}.jsonl", "w") as f:
            for i in out:
                f.write(m[i])
    (d / "ids.json").write_text(json.dumps(out))
    print("ko_strict", len(out), "new refill", len(new))


if __name__ == "__main__":
    {"plan": plan, "write": write}[sys.argv[1]]()
