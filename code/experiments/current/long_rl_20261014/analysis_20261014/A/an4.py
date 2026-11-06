import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
d = pd.read_json("feats.jsonl", lines=True)
o = d[d.pool=="old"]
print("OLD DEV by body"); print(o.groupby("body").solved.agg(["mean","size"]).round(2).T)
print("OLD DEV by schema"); t=o.groupby("schema").agg(rate=("solved","mean"), n=("solved","size"), body=("body","mean"), unsolved=("solved", lambda s:(1-s).sum())).sort_values("rate"); print(t.round(2))
lr = d[(d.pool=="rl") & (d.src=="ladder")]
print("RL ladder by schema"); t=lr.groupby("schema").agg(rate=("solved","mean"), n=("solved","size"), body=("body","mean")).sort_values("rate"); print(t.round(2))
print(o[o.schema=="gen"].pivot_table(index="body", values="solved", aggfunc=["mean","size"]))
