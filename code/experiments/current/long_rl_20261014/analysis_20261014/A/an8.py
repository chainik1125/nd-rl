import pandas as pd, numpy as np, statsmodels.formula.api as smf
d = pd.read_json("feats2.jsonl", lines=True)
d["ntok"] = d.gold.fillna("").str.split().str.len()
d["ptok"] = d.prompt.str.split().str.len()
for lab, sub in [("ladder gen RL 7-12", d[(d.pool=="rl")&(d.src=="ladder")&(d.schema=="gen")&d.body.between(7,12)]),
                 ("old gen 6-11", d[(d.pool=="old")&(d.schema=="gen")&d.body.between(6,11)])]:
    sub = sub[sub.gold.notna()].copy()
    for f in ["ptok", "ntok", "goal_size", "n_lines_gold", "max_depth", "n_ore", "n_negi", "n_dn", "n_bote"]:
        z = (sub[f]-sub[f].mean())/sub[f].std()
        m = smf.logit("solved ~ z + C(body)", sub.assign(z=z)).fit(disp=0)
        print(f"{lab:20s} {f:13s} OR/SD {np.exp(m.params['z']):.2f} p {m.pvalues['z']:.1e}  mean solved {sub[sub.solved==1][f].mean():.1f} unsolved {sub[sub.solved==0][f].mean():.1f}")
    sub["q"] = pd.qcut(sub.ptok, 4)
    print(sub.groupby("q").solved.agg(["mean","size"]))
