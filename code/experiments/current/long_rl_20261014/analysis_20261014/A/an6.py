import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
d = pd.read_json("feats2.jsonl", lines=True)
K = ["lemma_ori","andi_to_nege","andi_to_impe","lemma_impi","ore_nonatomic","lemma_negi","raa","ore_to_F"]
rl = d[(d.pool == "rl") & d.gold.notna()]
print(rl.groupby("src")[["g_" + k for k in K]].mean().round(3).to_string())
c = rl[rl.src=="c_f20.jsonl"]
print(c.groupby(["g_andi_to_impe","g_ore_to_F"]).size())
print("c: impi_chain dist", c.impi_chain.value_counts().to_dict(), "goal", c.goal_kind.value_counts().to_dict())
# Model found proofs: flag usage in found proofs vs gold for solved targets
s = d[(d.solved==1) & d.found_proof.notna() & d.gold.notna()]
for k in K:
    g, m = s["g_"+k], s["m_"+k]
    print(f"{k:14s} gold-has {int(g.sum()):5d}  of those model-proof-has {int((g.astype(bool)&m.astype(bool)).sum()):5d}   model-has-but-gold-not {int(((~g.astype(bool))&m.astype(bool)).sum())}")
