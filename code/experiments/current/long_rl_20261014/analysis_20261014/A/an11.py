import pandas as pd
d = pd.read_json("feats2.jsonl", lines=True)
def nested(p):
    return int(any(" : ORE " in seg and seg.split()[1] == "|" for seg in p.split(" ; ")))
s = d[(d.pool=="rl") & d.src.isin(["s_f20.jsonl","s2_f20.jsonl","b_f20.jsonl","b2_f20.jsonl","b3_f20.jsonl"]) & (d.n_ore>=2)].copy()
s["nested"] = s.gold.map(nested)
s["fam"] = s.src.str[0]
print(s[s.body.between(12,24)].groupby(["fam","body","nested"]).solved.agg(["mean","size"]).unstack().round(2))
