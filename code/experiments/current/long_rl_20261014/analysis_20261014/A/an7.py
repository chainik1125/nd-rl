import pandas as pd, json
pd.set_option("display.width", 250)
d = pd.read_json("feats2.jsonl", lines=True)
s = d[(d.solved == 1) & d.found_proof.notna() & d.pool.isin(["rl", "dev"]) & (d.src != "ladder")].copy()
s["excess"] = s.found_pruned_min - s.body
print("excess = model's shortest pruned proof - gold minimal body (exact where body_exact)")
print(s.groupby("src").excess.describe()[["count", "mean", "50%", "max"]].round(2))
print(s[s.body_exact == True].groupby(pd.cut(s.excess, [-99, -1, 0, 1, 3, 99])).size())
for k in ["raa", "ore_to_F", "andi_to_nege", "lemma_impi"]:
    t = s[s["g_" + k] == 1]
    print(k, "gold-has & solved:", len(t), " model also uses:", int(t["m_" + k].sum()), " mean excess", round(t.excess.mean(), 2), "vs rest same family", round(s[(s["g_" + k] != 1) & s.src.isin(t.src.unique())].excess.mean(), 2))
