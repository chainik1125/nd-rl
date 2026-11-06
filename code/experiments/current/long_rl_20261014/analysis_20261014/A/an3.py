import pandas as pd, numpy as np, statsmodels.formula.api as smf, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
d = pd.read_json("feats.jsonl", lines=True)
d["goal_imp"] = (d.goal_kind=="imp").astype(int); d["goal_atom"]=(d.goal_kind=="atom").astype(int); d["goal_and"]=(d.goal_kind=="and").astype(int); d["goal_neg"]=(d.goal_kind=="neg").astype(int)
d["fam"] = d.src.str.replace("2_f20.jsonl","").str.replace("_f20.jsonl","").str.replace(".jsonl","")
F = ["n_prem","atoms","max_depth","n_boxes","n_impe","n_ande","n_andi","n_ore","n_impi","n_negi","n_nege","n_bote","n_dn","case_nest","lemma_negi","lemma_andi","n_lemma","max_dist","mean_dist","max_fan","elim_frac","neg_prem","or_prem","goal_size","prem_size","goal_imp","goal_atom","goal_and","goal_neg","n_lines_gold"]
def uni(sub, label, fe="C(body)"):
    print(f"\n### {label}: n={len(sub)}, solved={sub.solved.mean():.2f}")
    rows = []
    for f in F:
        if sub[f].std() == 0 or sub[f].isna().all(): continue
        z = (sub[f]-sub[f].mean())/sub[f].std()
        s2 = sub.assign(z=z)
        try:
            m = smf.logit(f"solved ~ z + {fe}", s2).fit(disp=0)
            rows.append((f, sub[f].mean(), sub[sub.solved==1][f].mean(), sub[sub.solved==0][f].mean(), np.exp(m.params["z"]), m.pvalues["z"]))
        except Exception as e:
            rows.append((f, sub[f].mean(), np.nan, np.nan, np.nan, np.nan))
    t = pd.DataFrame(rows, columns=["feat","mean","mean_solved","mean_unsolved","OR_per_SD","p"]).sort_values("p")
    print(t.round(3).to_string(index=False))
rl = d[(d.pool=="rl") & d.n_boxes.notna()]
uni(rl[(rl.fam=="s") & rl.body.between(12,19)], "RL s+s2 body 12-19 (body FE)", "C(body)+C(src)")
uni(rl[(rl.fam=="d") & rl.body.between(9,14)], "RL d body 9-14 (body FE)")
uni(rl[(rl.fam=="ladder") & rl.body.between(9,12)], "RL ladder(gen proof) body 9-12 (body FE)")
uni(rl[(rl.fam.isin(["b","b3"])) & rl.body.between(20,26)], "RL b/b2/b3 body 20-26 (body FE)", "C(body)+C(src)")
