import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "/scratch/cowifornia/nd-rl/think/nd-takehome")
from fdepth import proof_depth, thm_depth

src = open("profile.py").read().replace("if __name__ == '__main__':", "if False:")
g = {}
exec(compile(src, "profile.py", "exec"), g)
D = "/scratch/cowifornia/nd-rl/think/nd-takehome/data/gt"
rows = []
for f in ["candidate_v0", "candidate_v1", "batch3", "hand_proved_14"]:
    for l in open(f"{D}/{f}.jsonl"):
        r = json.loads(l)
        d = g["thm_feats"](r["prompt"])
        d["file"] = f
        d["name"] = r["name"]
        d["prompt"] = r["prompt"]
        d["min_lines"] = r["min_lines"]
        d["classical"] = r["classical_only"]
        d["tdepth"] = thm_depth(r["prompt"])
        try:
            d.update(g["feats"](r["prompt"], r["reference_proof"]))
            d["pdepth"] = proof_depth(r["reference_proof"])
        except Exception as e:
            d["err"] = str(e)[:40]
        rows.append(d)
long = [d for d in rows if d["min_lines"] > 10]
print("long (min_lines>10):", len(long))


def reach(d):
    return (
        d["n_prem"] <= 3
        and d["tdepth"] <= 6
        and "decomp_err" not in d
        and "err" not in d
        and d.get("max_state", 999) <= 400
    )


from collections import Counter

print("by file", Counter(d["file"] for d in long))
print(
    ">=4 premises",
    sum(d["n_prem"] >= 4 for d in long),
    "| theorem depth >=7",
    sum(d["tdepth"] >= 7 for d in long),
    "| names exhausted",
    sum("decomp_err" in d for d in long),
    "| max_state>400",
    sum(d.get("max_state", 0) > 400 for d in long),
)
R = [d for d in long if reach(d)]
print("within reach", len(R), Counter(d["file"] for d in R))
for lo, hi in [(11, 20), (21, 30), (31, 200)]:
    s = [d for d in R if lo <= d["min_lines"] <= hi]
    print(
        f"  min_lines {lo}-{hi}: {len(s)}; classical {sum(d['classical'] for d in s)}; median max_state",
        sorted(d["max_state"] for d in s)[len(s) // 2] if s else None,
        "max",
        max((d["max_state"] for d in s), default=None),
        "max_action",
        max((d["max_action"] for d in s), default=None),
        "actions median",
        sorted(d["n_actions"] for d in s)[len(s) // 2] if s else None,
    )
print("all release within reach (any length):", sum(reach(d) for d in rows), "of", len(rows))
json.dump(rows, open("charles_rows.json", "w"))
