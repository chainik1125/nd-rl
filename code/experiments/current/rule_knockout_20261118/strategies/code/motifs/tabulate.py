# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
import collections, json, pickle, sys
from ndlib import *
from motifs import motifs


def run():
    out = {}
    ex = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )  # motif -> corpus -> [(len, name, proof)]

    def add(corp, name, proof, ms, body):
        for x in ms:
            e = ex[x][corp]
            if len(e) < 4 or body < e[-1][0]:
                e.append((body, name, proof))
                e.sort(key=lambda t: t[0])
                del e[4:]

    # pretrain
    c = collections.Counter()
    n = 0
    pre_sets = []
    for r in pretrain():
        p = Proof(r["proof"])
        ms = motifs(p)
        c.update(ms)
        n += 1
        add("pre", r["name"], r["proof"], ms, p.body)
    out["pre"] = (c, n)
    # rl refs: all, reachable (body<=14), by band
    c = collections.Counter()
    n = 0
    c14 = collections.Counter()
    n14 = 0
    rlm = {}
    for r in jl(D + "rl.jsonl"):
        if not r.get("proof"):
            continue
        p = Proof(r["proof"])
        ms = motifs(p)
        c.update(ms)
        n += 1
        rlm[r["name"]] = (r["body"], r["band"], r["src"], ms)
        if r["body"] <= 14:
            c14.update(ms)
            n14 += 1
        add("rl", r["name"], r["proof"], ms, p.body)
    out["rl"] = (c, n)
    out["rl14"] = (c14, n14)
    c = collections.Counter()
    n = 0
    for r in jl(D + "dev.jsonl"):
        if not r.get("proof"):
            continue
        c.update(motifs(Proof(r["proof"])))
        n += 1
    out["dev"] = (c, n)
    # control
    rounds = {}
    for s in ("full-ei_s0", "full-ei_s1"):
        cum = 0
        cs = 0
        rounds[s] = {}
        for r in jl(LIVE + s + "/rounds.jsonl"):
            cs += r["sampling"]["samples"]
            rounds[s][r["round"]] = (r["rl_secs_used"] / 60, cs)
    c = collections.Counter()
    n = 0
    targets = collections.defaultdict(set)
    first = {}
    perseed = {}
    ctl_ref_overlap = collections.Counter()
    for s in ("full-ei_s0", "full-ei_s1"):
        cs_ = collections.Counter()
        ns = 0
        for r in jl(LIVE + s + "/found_stream.jsonl"):
            p = Proof(r["proof"])
            ms = motifs(p)
            c.update(ms)
            n += 1
            cs_.update(ms)
            ns += 1
            add("ctl", r["name"], r["proof"], ms, p.body)
            for x in ms:
                targets[x].add(r["name"])
                if (x, s) not in first:
                    first[(x, s)] = r["round"]
        perseed[s] = (cs_, ns)
    out["ctl"] = (c, n)
    out["ctl_targets"] = {k: len(v) for k, v in targets.items()}
    out["first"] = first
    out["rounds"] = rounds
    out["perseed"] = perseed
    out["ex"] = {k: dict(v) for k, v in ex.items()}
    out["rlm"] = rlm
    pickle.dump(out, open("tab.pkl", "wb"))
    return out


if __name__ == "__main__":
    o = run()
    pre, npre = o["pre"]
    rl, nrl = o["rl"]
    rl14, n14 = o["rl14"]
    ctl, nctl = o["ctl"]
    dev, ndev = o["dev"]
    print(
        f"pretrain {npre}  rl-ref {nrl}  rl-ref body<=14 {n14}  dev {ndev}  control proofs {nctl}"
    )
    keys = sorted(
        set(pre) | set(rl) | set(ctl),
        key=lambda k: (k.startswith("GOAL"), k.startswith("RULE"), -(rl[k] / nrl + ctl[k] / nctl)),
    )
    with open("motif_table.tsv", "w") as f:
        hdr = "motif\tpre_n\tpre_%\trl_n\trl_%\trl14_n\trl14_%\tdev_%\tctl_n\tctl_%\tctl_targets\tfirst_s0(round/min/samples)\tfirst_s1"
        f.write(hdr + "\n")
        for k in keys:
            fs = []
            for s in ("full-ei_s0", "full-ei_s1"):
                r = o["first"].get((k, s))
                fs.append(
                    "-"
                    if r is None
                    else f"r{r}/{o['rounds'][s][r][0]:.0f}m/{o['rounds'][s][r][1] // 1000}k"
                )
            f.write(
                f"{k}\t{pre[k]}\t{100 * pre[k] / npre:.2f}\t{rl[k]}\t{100 * rl[k] / nrl:.1f}\t{rl14[k]}\t{100 * rl14[k] / n14:.1f}\t{100 * dev[k] / ndev:.1f}\t{ctl[k]}\t{100 * ctl[k] / nctl:.1f}\t{o['ctl_targets'].get(k, 0)}\t{fs[0]}\t{fs[1]}\n"
            )
