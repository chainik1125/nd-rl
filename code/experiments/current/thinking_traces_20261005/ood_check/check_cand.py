import json
import os
import statistics as st
import sys
import time
from collections import defaultdict

sys.path.insert(0, ".")
sys.path.insert(0, "/scratch/cowifornia/nd-rl/think/nd-takehome")
os.environ.setdefault("LEAN_GATE_LOG", os.path.abspath("lean_gate.jsonl"))
from fdepth import proof_depth, thm_depth
from gen import canon_key
from nd_verify import verify_text
from state_env import decompose

src = open("profile.py").read().replace("if __name__ == '__main__':", "if False:")
g = {}
exec(compile(src, "profile.py", "exec"), g)
import lean_judge


def thm_str(prompt):
    body = prompt[len("THM ") : -len(" PRF")]
    prems, concl = body.split(" SEQ ")
    return (prems.strip() + " |- " + concl.strip()).strip()


def max_name(prompt, proof):
    steps, toks, env = decompose(prompt, proof, canon=True)
    return max(int(t[1:]) for t in toks if t[0] == "n" and t[1:].isdigit())


rows = [json.loads(l) for l in open(sys.argv[1])]
# training overlap by renaming class
train_keys = set()
for f in [
    "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl",
    "/scratch/cowifornia/nd-rl/think/nd-takehome/data/rl_targets.jsonl",
]:
    for l in open(f):
        r = json.loads(l)
        train_keys.add(canon_key(r["thm"]))
t0 = time.time()
lean = lean_judge.judge_many([(r["prompt"], r["proof"]) for r in rows])
print("lean secs", round(time.time() - t0, 1))
agg = defaultdict(list)
for r, lj in zip(rows, lean):
    ok, why, nl = verify_text(r["prompt"] + " " + r["proof"])
    d = g["feats"](r["prompt"], r["proof"])
    d.update(
        nd_ok=ok,
        lean_ok=lj[0],
        tdepth=thm_depth(r["prompt"]),
        pdepth=proof_depth(r["proof"]),
        in_train=canon_key(thm_str(r["prompt"])) in train_keys,
    )
    try:
        d["max_name"] = max_name(r["prompt"], r["proof"])
    except Exception:
        d["max_name"] = None
    r.update(d)
    agg[(r["family"], r["knob"])].append(d)
print(
    "family knob | n | nd_ok | lean_ok | in_train | L med/max | thm depth max | proof depth max | goal_tok max | actions med | max_state med/max | max_action max | max_name max | proof_tok med"
)
for k in sorted(agg):
    ds = agg[k]
    m = lambda key: st.median([x[key] for x in ds if x.get(key) is not None])
    M = lambda key: max([x[key] for x in ds if x.get(key) is not None])
    print(
        f"{k} | {len(ds)} | {sum(x['nd_ok'] for x in ds)} | {sum(x['lean_ok'] for x in ds)} | {sum(x['in_train'] for x in ds)} | {m('L'):.0f}/{M('L')} | {M('tdepth')} | {M('pdepth')} | {M('goal_tok')} | {m('n_actions'):.0f} | {m('max_state'):.0f}/{M('max_state')} | {M('max_action')} | {M('max_name')} | {m('proof_tok'):.0f}"
    )
json.dump(rows, open(sys.argv[1].replace(".jsonl", "_checked.json"), "w"))
