"""The S1g3 dose corpora: Dan's knockout (K12-) and dose (K12^d) construction applied to S1g3.

    python -m code.experiments.current.knockout_grpo_20261103.assemble plan
    python -m code.experiments.current.knockout_grpo_20261103.assemble write   # after refill_e64

S1g3 is one programmatic search trace per K12 theorem (154,630 records; ids `k12_<line>`). The
rule (PREREG.md, "Dose corpora"):

* **Knockout (ko).** Remove every S1g3 record whose theorem Dan's knockout removed from K12 (his
  sigma-plus records, 16,961) and every S1g3 record whose own (search) proof is sigma-plus or uses
  the lemma route (Dan's predicates, `common.label`). Refill per K12 generator length L (Dan's
  matching key, the theorem's K12 `n_lines`), in place: first from Dan's own K12- refill theorems
  of length L whose search proof is sigma-plus-free and lemma-route-free (`rng.sample` when there
  are more than needed), then, if short, from the rest of Dan's eligible pool at L (same filter).
  Seed 0. Per-length counts and record count equal S1g3's.
* **Dose d (d01: 0.1 %, d1: 1 %).** coe_dose.py on the knockout: want = round(d x N) records with
  sigma (Dan's dose predicate, `sigma_strict` in the written or pruned search proof) drawn from
  the removed S1g3 records, per L in proportion to the removed sigma records' length mix (his
  rounding fix), each replacing a refill record of the same L (uniform, seeded). Seed 0; the two
  doses are independent draws, as Dan's were.

Traces: kept and dose records keep their S1g3 traces (e256 / e64 lines, byte for byte); refill
records get the same generator's traces (searchtrace.py at 79749ce, which reproduces S1g3's traces
byte for byte: refill/repro). Output OUT/corpora/<arm>/{e256,e64,plain}.jsonl + plan.json.
"""

from __future__ import annotations

import collections
import json
import random
import sys

from .common import DAN_DATA, K12, OUT, TRACES, jsonl, sha256

ARMS = ("ko", "d01", "d1")
DOSES = {"d01": 0.001, "d1": 0.01}
SEED = 0


def load_labels(name: str) -> dict:
    return {r["id"]: r for r in jsonl(OUT / "sigma" / f"{name}.labels.jsonl")}


def plan() -> dict:
    rng = random.Random(SEED)
    k12 = list(jsonl(K12))
    k12_lab = list(jsonl(OUT / "sigma" / "k12.labels.jsonl"))  # same order as K12
    assert len(k12_lab) == len(k12)
    s1 = list(jsonl(TRACES / "plain_e256.jsonl"))
    lab = load_labels("s1g3")
    order = [r["id"] for r in s1]
    for r in s1:
        i = int(r["id"].split("_")[1])
        assert k12[i]["prompt"] == r["prompt"], r["id"]
    gen_len = {r["id"]: k12[int(r["id"].split("_")[1])]["n_lines"] for r in s1}
    dan_removed = {f"k12_{i}" for i, x in enumerate(k12_lab) if x["sigma_plus"]}
    assert len(dan_removed) == 16961
    X = [i for i in order if i in dan_removed or lab[i]["sigma_plus"] or lab[i]["lemma_route"]]
    why = collections.Counter(
        ("dan" if i in dan_removed else "")
        + ("+own" if lab[i]["sigma_plus"] or lab[i]["lemma_route"] else "")
        for i in X
    )
    need = collections.Counter(gen_len[i] for i in X)

    # candidates: Dan's eligible pool with our search traces and labels
    cand = {r["id"]: r for r in jsonl(OUT / "refill" / "candidates.jsonl")}
    clab = load_labels("cand")  # ids of the traced candidates (failed searches are absent)
    ok = {
        i
        for i, x in clab.items()
        if x.get("parse") and not x["sigma_plus"] and not x["lemma_route"]
    }
    by_prompt = {r["prompt"]: i for i, r in cand.items()}
    dan_refill = collections.defaultdict(list)
    dan_refill_missing = collections.Counter()
    for r in jsonl(DAN_DATA / "train_k12ko.jsonl.gz"):
        if r.get("coe_refill"):
            cid = by_prompt.get(r["prompt"])
            if cid is None:
                dan_refill_missing["not_in_our_pool"] += 1
                continue
            dan_refill[r["n_lines"]].append(cid)
    first, rest = {}, {}
    for L in need:
        f = [c for c in dan_refill[L] if c in ok]
        dan_refill_missing[f"L{L}_search_sigma_or_failed"] += len(dan_refill[L]) - len(f)
        first[L] = f
        fs = set(f) | set(dan_refill[L])
        rest[L] = sorted(
            (c for c, r in cand.items() if r["n_lines"] == L and c in ok and c not in fs),
            key=lambda c: (c.split("_")[1], int(c.split("_")[2])),
        )
    refill_by_L = {}
    topup = collections.Counter()
    for L in sorted(need):
        n = need[L]
        f = first[L]
        if len(f) >= n:
            pick = [f[j] for j in sorted(rng.sample(range(len(f)), n))]
        else:
            k = n - len(f)
            assert len(rest[L]) >= k, (L, n, len(f), len(rest[L]))
            pick = list(f) + [rest[L][j] for j in sorted(rng.sample(range(len(rest[L])), k))]
            topup[L] = k
        rng.shuffle(pick)
        refill_by_L[L] = pick
    # in place: each removed slot takes the next refill record of its length
    it = {L: iter(v) for L, v in refill_by_L.items()}
    Xs = set(X)
    ko = []
    for i in order:
        if i in Xs:
            ko.append(next(it[gen_len[i]]))
        else:
            ko.append(i)
    for v in it.values():
        assert next(v, None) is None
    refill_L = {c: L for L, v in refill_by_L.items() for c in v}
    arms = {"ko": ko}
    # doses (coe_dose.py's rule)
    sig = collections.defaultdict(list)
    for i in X:
        if lab[i]["sigma"]:
            sig[gen_len[i]].append(i)
    total_sig = sum(len(v) for v in sig.values())
    N = len(ko)
    dose_rep = {}
    for arm, d in DOSES.items():
        r2 = random.Random(SEED)
        want = round(d * N)
        per = {L: int(round(want * len(v) / total_sig)) for L, v in sig.items()}
        while sum(per.values()) != want:
            L = max(per, key=lambda x: len(sig[x]))
            per[L] += 1 if sum(per.values()) < want else -1
        slots = collections.defaultdict(list)
        for j, c in enumerate(ko):
            if c in refill_L:
                slots[refill_L[c]].append(j)
        out = list(ko)
        added = []
        for L, k in sorted(per.items()):
            assert len(slots[L]) >= k, (L, k, len(slots[L]))
            js = r2.sample(slots[L], k)
            srcs = r2.sample(sig[L], k)
            for j, s in zip(js, srcs, strict=True):
                out[j] = s
                added.append(s)
        arms[arm] = out
        dose_rep[arm] = {
            "share": d,
            "want": want,
            "per_length_added": dict(sorted(per.items())),
            "added": len(added),
        }
    rep = {
        "seed": SEED,
        "records": N,
        "removed": len(X),
        "removed_why": dict(why),
        "removed_per_gen_length": dict(sorted(need.items())),
        "dan_refill_used_or_dropped": dict(dan_refill_missing),
        "refill_from_dan_k12ko": {L: min(need[L], len(first[L])) for L in sorted(need)},
        "refill_topup_from_pool": dict(sorted(topup.items())),
        "sigma_pool_removed": {L: len(v) for L, v in sorted(sig.items())},
        "doses": dose_rep,
        "inputs_sha256": {
            "plain_e256": sha256(TRACES / "plain_e256.jsonl"),
            "train_k12ko": sha256(DAN_DATA / "train_k12ko.jsonl.gz"),
        },
    }
    d = OUT / "corpora"
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan.json").write_text(json.dumps({"report": rep, "arms": arms}, indent=0))
    (d / "plan_report.json").write_text(json.dumps(rep, indent=1))
    refill_ids = sorted(set(c for c in ko if c.startswith("ko_")))
    with open(d / "refill_ids.txt", "w") as f:
        f.write("\n".join(refill_ids) + "\n")
    with open(d / "refill_candidates.jsonl", "w") as f:
        for c in refill_ids:
            r = dict(cand[c])
            r["name"] = c
            f.write(json.dumps(r) + "\n")
    return rep


def write() -> None:
    P = json.loads((OUT / "corpora" / "plan.json").read_text())
    need = set()
    for arm in ARMS:
        need |= set(P["arms"][arm])

    def index(path, keep):
        m = {}
        with open(path) as fh:
            for line in fh:
                i = json.loads(line)["id"]
                if i in keep:
                    m[i] = line if line.endswith("\n") else line + "\n"
        return m

    src = {
        "e256": {
            **index(TRACES / "e256.jsonl", need),
            **index(OUT / "refill" / "cand_e256.jsonl", need),
        },
        "e64": {
            **index(TRACES / "e64.jsonl", need),
            **index(OUT / "refill" / "refill_e64.jsonl", need),
        },
        "plain": {
            **index(TRACES / "plain_e256.jsonl", need),
            **index(OUT / "refill" / "cand_plain_e256.jsonl", need),
        },
    }
    for arm in ARMS:
        ids = P["arms"][arm]
        assert len(set(ids)) == len(ids), arm
        d = OUT / "corpora" / arm
        d.mkdir(parents=True, exist_ok=True)
        for k, m in src.items():
            name = "plain_e256" if k == "plain" else k
            with open(d / f"{name}.jsonl", "w") as f:
                for i in ids:
                    f.write(m[i])
        print(arm, len(ids))


if __name__ == "__main__":
    {"plan": lambda: print(json.dumps(plan(), indent=1)), "write": write}[sys.argv[1]]()
