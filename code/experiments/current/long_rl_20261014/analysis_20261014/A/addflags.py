import json, sys
sys.path.insert(0, ".")
from pre import extra
from feat import parse_prompt
FORK = "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/fork51604b3"
lad = {r["prompt"]: r for r in map(json.loads, open(f"{FORK}/data/ladder/rl_targets.jsonl"))}
tr = {r["prompt"]: r for r in map(json.loads, open(f"{FORK}/data/ladder/transfer.jsonl"))}
K = ["lemma_ori","andi_to_nege","andi_to_impe","lemma_impi","ore_nonatomic","lemma_negi","raa","ore_to_F"]
rows = [json.loads(l) for l in open("feats.jsonl")]
for r in rows:
    if r.get("gold"):
        src = lad.get(r["prompt"]) or tr.get(r["prompt"])
        p = ("THM " + src["thm"].replace(" |- ", " SEQ ") + " PRF") if (src and r["src"] in ("ladder", "gen")) else r["prompt"]
        pr, g = parse_prompt(p)
        f = extra(pr, g, r["gold"])
        for k in K: r["g_" + k] = int(f[k] > 0)
    if r.get("found_proof"):
        pr, g = parse_prompt(r["prompt"])
        try:
            f = extra(pr, g, r["found_proof"])
            for k in K: r["m_" + k] = int(f[k] > 0)
        except Exception as e:
            print("fail", r["name"], e)
with open("feats2.jsonl", "w") as fo:
    for r in rows: fo.write(json.dumps(r) + "\n")
