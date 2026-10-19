import json
import random
import statistics as st
import sys

sys.path.insert(0, "/scratch/cowifornia/nd-rl/think/nd-takehome")
from nd2lean import parse_prompt
from nd_verify.verify import parse_proof_tokens


def fdepth(f):
    if f[0] in ("atom", "bot"):
        return 0
    if f[0] == "not":
        return 1 + fdepth(f[1])
    return 1 + max(fdepth(f[1]), fdepth(f[2]))


def thm_depth(prompt):
    prem, concl = parse_prompt(prompt)
    return max([fdepth(p) for p in prem] + [fdepth(concl)])


def proof_depth(proof):
    return max(fdepth(l["formula"]) for l in parse_proof_tokens(proof.split()))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


if __name__ == "__main__":
    rng = random.Random(0)
    k12 = [
        json.loads(l)
        for l in open("/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl")
    ]
    s = rng.sample(k12, 30000)
    td = [thm_depth(r["prompt"]) for r in s]
    pdp = [proof_depth(r["proof"]) for r in s]
    from collections import Counter

    print("K12 theorem formula depth", sorted(Counter(td).items()), "p99", pct(td, 0.99))
    print("K12 proof-line formula depth", sorted(Counter(pdp).items()), "p99", pct(pdp, 0.99))
    rt = [
        json.loads(l)
        for l in open("/scratch/cowifornia/nd-rl/think/nd-takehome/data/rl_targets.jsonl")
    ]
    td = [thm_depth(r["prompt"]) for r in rt]
    print("rl_targets theorem depth", sorted(Counter(td).items()))
    D = "/scratch/cowifornia/nd-rl/think/nd-takehome/data"
    for f in [
        "gt/tb72_textbook_dev",
        "gt/tb72_textbook_train",
        "gt/candidate_v0",
        "gt/candidate_v1",
        "gt/batch3",
        "gt/hand_proved_14",
        "ladder/transfer_long2",
        "ladder/transfer_long_rr600",
    ]:
        rs = [json.loads(l) for l in open(f"{D}/{f}.jsonl")]
        td = [thm_depth(r["prompt"]) for r in rs]
        print(f, sorted(Counter(td).items()))
    fam = [
        json.loads(l)
        for l in open("/scratch/cowifornia/nd-rl/bench/out/suites/families/v1/items.jsonl")
    ]
    agg = {}
    for r in fam:
        p = (
            "THM "
            + " , ".join(r["sequent"]["premises"])
            + " SEQ "
            + r["sequent"]["conclusion"]
            + " PRF"
            if r["sequent"]["premises"]
            else "THM SEQ " + r["sequent"]["conclusion"] + " PRF"
        )
        try:
            d = thm_depth(p)
        except Exception:
            d = None
        agg.setdefault((r["tags"]["family"], r["tags"]["n"]), []).append(d)
    for k in sorted(agg):
        if k[0] in (
            "imp_nest",
            "and_tower",
            "or_tower",
            "and_extract",
            "reorder_or",
            "reorder_and",
        ):
            ds = [x for x in agg[k] if x is not None]
            print(
                "family",
                k,
                "depth median",
                st.median(ds) if ds else None,
                "max",
                max(ds) if ds else None,
            )
    laws = [
        json.loads(l)
        for l in open("/scratch/cowifornia/nd-rl/bench/out/suites/laws/v1/items.jsonl")
    ]
    agg = {}
    for r in laws:
        p = (
            "THM "
            + " , ".join(r["sequent"]["premises"])
            + " SEQ "
            + r["sequent"]["conclusion"]
            + " PRF"
            if r["sequent"]["premises"]
            else "THM SEQ " + r["sequent"]["conclusion"] + " PRF"
        )
        agg.setdefault(r["tags"]["s"], []).append(thm_depth(p))
    for k in sorted(agg):
        print("laws s", k, "depth median", st.median(agg[k]), "max", max(agg[k]))
