import pandas as pd, numpy as np, statsmodels.formula.api as smf, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
d = pd.read_json("feats2.jsonl", lines=True)
K = ["lemma_ori","andi_to_nege","andi_to_impe","lemma_impi","ore_nonatomic","lemma_negi","raa","ore_to_F"]
def strat(sub, label):
    print(f"\n### {label} n={len(sub)} solved={sub.solved.mean():.3f}")
    out = []
    for k in K:
        c = "g_" + k
        if c not in sub or sub[c].sum() == 0: continue
        sub2 = sub.copy(); sub2["st"] = sub2.src.astype(str) + "_" + sub2.body.astype(str)
        # MH-style: within strata containing both values, weighted mean diff
        diffs, ws, n1 = [], [], 0
        r1s, r0s = [], []
        for _, g in sub2.groupby("st"):
            a, b = g[g[c] == 1], g[g[c] == 0]
            if len(a) and len(b):
                w = len(a) * len(b) / len(g)
                diffs.append(a.solved.mean() - b.solved.mean()); ws.append(w); n1 += len(a)
                r1s.append((a.solved.mean(), w)); r0s.append((b.solved.mean(), w))
        if not ws: continue
        W = sum(ws)
        r1 = sum(r * w for r, w in r1s) / W; r0 = sum(r * w for r, w in r0s) / W
        try:
            m = smf.logit(f"solved ~ {c} + C(st)", sub2[sub2.st.isin(sub2.groupby('st').filter(lambda g: g.solved.nunique() > 1).st)]).fit(disp=0)
            orr, p = np.exp(m.params[c]), m.pvalues[c]
        except Exception:
            orr, p = np.nan, np.nan
        out.append((k, int(sub[c].sum()), n1, round(r1, 3), round(r0, 3), round(r1 - r0, 3), round(orr, 3), p))
    print(pd.DataFrame(out, columns=["flag", "n_flag", "n_flag_in_mixed_strata", "rate_flag", "rate_noflag", "diff", "OR(strata FE)", "p"]).to_string(index=False))
rl = d[(d.pool == "rl") & d.gold.notna()]
strat(rl[rl.body.between(7, 19)], "RL body 7-19 (all families w/ gold incl. ladder gen proofs), strata = family x body")
strat(rl[rl.body.between(20, 34)], "RL body 20-34, strata = family x body")
dv = d[(d.pool == "dev") & d.gold.notna()]
strat(dv[dv.body.between(7, 26)], "DEV body 7-26 (k=64)")
o = d[(d.pool == "old") & d.gold.notna()]
strat(o, "OLD DEV gen-schema (gen_proof)")
# raw prevalence of flags in gold by family
print(rl.groupby("src")[["g_" + k for k in K]].mean().round(3))
