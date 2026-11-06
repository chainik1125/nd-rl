import pandas as pd, numpy as np, statsmodels.formula.api as smf, warnings
warnings.filterwarnings("ignore")
d = pd.read_json("feats2.jsonl", lines=True)
s = d[(d.pool=="rl") & d.src.isin(["s_f20.jsonl","s2_f20.jsonl"]) & d.body.between(12,19)].copy()
s["goal_and"] = (s.goal_kind=="and").astype(int); s["goal_imp"]=(s.goal_kind=="imp").astype(int)
cols = ["n_ore","case_nest","n_ande","max_dist","n_bote","goal_and","goal_imp","n_prem","atoms","max_fan"]
for c in cols: s[c+"_z"] = (s[c]-s[c].mean())/s[c].std()
m = smf.logit("solved ~ " + " + ".join(c+"_z" for c in cols) + " + C(body) + C(src)", s).fit(disp=0)
print(pd.DataFrame({"OR_per_SD": np.exp(m.params), "p": m.pvalues}).loc[[c+"_z" for c in cols]].round(3))
print("SD of each:", {c: round(s[c].std(),2) for c in cols})
print(s.groupby(["body", "n_ore"]).solved.agg(["mean","size"]).unstack().loc[12:19].round(2))
