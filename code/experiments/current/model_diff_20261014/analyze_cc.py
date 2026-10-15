# ruff: noqa: E501  (research scripts: long result lines, bare open(), short loop names)
"""Read a crosscoder run (runs/<tag>/cc.pt) and report:
  1. relative decoder norm r = |d_fin| / (|d_pre| + |d_fin|): histogram; pre-only (r < 0.1), fin-only (r > 0.9), shared
  2. latent scaling for exclusive latents (Minder et al. 2025): nu_x = beta_pre / beta_fin on activations, nu_e on errors
  3. enrichment of each latent on labelled token sets: depth >= 3, pops at k = 1 / 2 / 3, closes; which latents are most
     enriched, and their r. The question: are the 'deep pop' latents fin-specific in g0 and not in g1?
  4. top-activating contexts for the most interesting latents
python analyze_cc.py runs/g0_L4 [n_show]
"""

import sys

import torch

run = sys.argv[1]
NSHOW = int(sys.argv[2]) if len(sys.argv) > 2 else 8
cc = torch.load(f"{run}/cc.pt", map_location="cpu")
C = torch.load("out/corpus.pt")
itos = C["itos"]
ids = C["ids"].long()
names = cc["names"]
print(f"== {run}: {names}, layer {cc['args']['layer']}, m {cc['args']['m']}, k {cc['args']['k']}")
print("eval FVU", [round(float(v), 4) for v in cc["fvu"]], "| splice", cc["splice"])
freq = cc["freq"]
alive = freq > 0
print(f"alive latents {int(alive.sum())}/{len(freq)}")
two = len(names) == 2
if two:
    r = cc["rel"]
    cos = cc["cos"]
    h = torch.histc(r[alive], 10, 0, 1).int().tolist()
    print("rel-norm hist (alive; bins of 0.1 from pre-only to fin-only):", h)
    pre_only = alive & (r < 0.1)
    fin_only = alive & (r > 0.9)
    shared = alive & (r > 0.3) & (r < 0.7)
    print(
        f"pre-only {int(pre_only.sum())}, fin-only {int(fin_only.sum())}, shared {int(shared.sum())}; "
        f"shared with decoder cos < 0.5: {int((shared & (cos < 0.5)).sum())}"
    )
    beta = cc["beta"]  # (2 targets, n, m)
    for tag, sel in (("fin-only", fin_only), ("pre-only", pre_only)):
        if sel.sum() == 0:
            continue
        nux = (beta[0, 0] / beta[0, 1].clamp_min(1e-6))[sel]
        nue = (beta[1, 0] / beta[1, 1].clamp_min(1e-6))[sel]
        print(
            f"  {tag}: latent scaling nu_x median {nux.median():.2f} (q10 {nux.quantile(0.1):.2f}, q90 {nux.quantile(0.9):.2f}); "
            f"nu_e median {nue.median():.2f}  [nu ~ 1: the direction also explains the other model -> not truly exclusive]"
        )
cnt, tot = cc["cnt"], cc["tot"]
base = cnt["src"].sum(0) / tot["src"].sum()  # P(fire) over proof tokens


def enrich(key, vals):
    c = cnt[key][vals].sum(0)
    t = tot[key][vals].sum()
    return (c / t) / base.clamp_min(1e-9), c / t, int(t)


rows = {}
for label, key, vals in (
    ("depth>=3", "depth", [3, 4, 5]),
    ("pop k=1", "pop_k", [1]),
    ("pop k=2", "pop_k", [2]),
    ("pop k=3", "pop_k", [3]),
    ("close k=1", "close_k", [1]),
    ("close k=2", "close_k", [2]),
):
    e, p, t = enrich(key, vals)
    rows[label] = (e, p)
    ok = alive & (p > 0.2) & (cnt[key][vals].sum(0) >= 20)
    idx = torch.nonzero(ok).flatten()
    idx = idx[e[idx].argsort(descending=True)][:NSHOW]
    print(f"\n-- {label} (n tokens {t}): latents firing on >20% of them, most enriched first")
    for j in idx.tolist():
        extra = f" r {float(cc['rel'][j]):.2f} cos {float(cc['cos'][j]):.2f}" if two else ""
        print(
            f"   #{j}: fires {float(p[j]):.0%} here vs {float(base[j]):.2%} overall (x{float(e[j]):.0f}){extra}"
        )
    if two:
        # among latents that fire on > 20% of this set, distribution of r
        rr = cc["rel"][ok]
        if len(rr):
            print(
                f"   r of the {len(rr)} such latents: mean {rr.mean():.2f}, #r>0.7 {int((rr > 0.7).sum())}, #r<0.3 {int((rr < 0.3).sum())}"
            )


def ctx(gi, w=14):
    # find record and print a window of tokens around global token index gi
    st = int(torch.searchsorted(C["starts"], torch.tensor(gi), right=True)) - 1
    a = max(int(C["starts"][st]) + int(C["n_prompt"][st]), gi - w)
    return " ".join(
        (f"[{itos[ids[q]]}]" if q == gi else itos[ids[q]]) for q in range(a, min(gi + 4, len(ids)))
    )


def show(j, n=4):
    for v, gi in zip(cc["topv"][:n, j].tolist(), cc["topi"][:n, j].tolist(), strict=False):
        if gi >= 0:
            print(f"      {v:6.2f}  {ctx(gi)}")


if two:
    print(
        "\n== fin-only latents (r > 0.9), most frequent first, with their deep-pop enrichment and top contexts"
    )
    fo = torch.nonzero(fin_only).flatten()
    fo = fo[freq[fo].argsort(descending=True)][:NSHOW]
    for j in fo.tolist():
        print(
            f"  #{j}: freq {float(freq[j]):.3%} | P(fire) depth>=3 {float(rows['depth>=3'][1][j]):.0%}, pop2 {float(rows['pop k=2'][1][j]):.0%}, pop1 {float(rows['pop k=1'][1][j]):.0%} | cos {float(cc['cos'][j]):.2f}"
        )
        show(j, 3)
    print("\n== pre-only latents (r < 0.1)")
    po = torch.nonzero(pre_only).flatten()
    po = po[freq[po].argsort(descending=True)][:NSHOW]
    for j in po.tolist():
        print(
            f"  #{j}: freq {float(freq[j]):.3%} | P(fire) depth>=3 {float(rows['depth>=3'][1][j]):.0%}, pop2 {float(rows['pop k=2'][1][j]):.0%}, pop1 {float(rows['pop k=1'][1][j]):.0%}"
        )
        show(j, 3)
