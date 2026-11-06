import pandas as pd
d = pd.read_json("feats2.jsonl", lines=True)
s = d[(d.pool=="rl") & d.src.isin(["s_f20.jsonl","s2_f20.jsonl"]) & d.body.between(12,19) & (d.n_ore==2)]
print(s.groupby(["body","case_nest"]).solved.agg(["mean","size"]).unstack().round(2))
import json
x = s[(s.body==13)]
for cn in [1,2]:
    for _, r in x[(x.case_nest==cn)].head(1).iterrows():
        print("case_nest", cn, "solved", r.solved, r.prompt); [print("   ", l) for l in r.gold.split(" ; ") if ": PR" not in l and l!="QED"]
# where is the disjunction for the 2nd ORE from?
