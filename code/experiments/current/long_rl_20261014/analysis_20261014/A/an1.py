import pandas as pd, numpy as np
d = pd.read_json("feats.jsonl", lines=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
rl = d[d.pool == "rl"]
print(rl.groupby(["src", "band"]).solved.agg(["mean", "size"]).unstack().round(2))
# exact body x src solved rates for 7-19
s = rl[(rl.body >= 7) & (rl.body <= 22)]
print(s.pivot_table(index="body", columns="src", values="solved", aggfunc="mean").round(2))
print(s.pivot_table(index="body", columns="src", values="solved", aggfunc="size"))
dv = d[d.pool == "dev"]
print(dv.pivot_table(index="body", columns="src", values="solved", aggfunc=["mean", "size"]).round(2).loc[7:26])
