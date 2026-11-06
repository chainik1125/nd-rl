import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
d = pd.read_json("feats.jsonl", lines=True)
F = ["n_prem","atoms","n_distract","max_depth","n_boxes","n_impe","n_ande","n_andi","n_ori","n_ore","n_impi","n_negi","n_nege","n_bote","n_dn","n_r","case_nest","ore_derived","lemma_impi","lemma_negi","lemma_ori","lemma_andi","n_lemma","max_dist","mean_dist","max_fan","elim_frac","impi_chain","neg_prem","or_prem","goal_size","prem_size"]
g = d[(d.pool=="rl") & (d.body.between(10,19)) & d.n_boxes.notna()]
print("family means, RL body 10-19, gold present")
print(g.groupby("src")[F+["solved","body"]].mean().T.round(2))
print(g.groupby("src").goal_kind.value_counts(normalize=True).unstack().round(2))
