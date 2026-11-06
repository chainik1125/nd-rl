import os, sys, json, math
os.environ.setdefault("ND_ARCH", "untied")
sys.path.insert(0, "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe"); sys.path.insert(0, ".")
import ndmodel as M, torch
torch.set_num_threads(3)
E = "/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014"
d = {r["prompt"]: r for r in map(json.loads, open("feats2.jsonl"))}
orig = json.load(open("int_s_orig.json"))
for ck in ["pretrain", "final"]:
    model, tok = M.load(f"{E}/resume/untied-ei_s1/{ck}.pt")
    worst, tot, ex = [], [], None
    for p in orig:
        rows = M.token_logp(model, tok, p, d[p]["gold"])
        tot.append(sum(l for _, l, _ in rows))
        i = min(range(len(rows)), key=lambda i: rows[i][1])
        worst.append((rows[i][0], rows[i][2], round(rows[i][1], 2)))
        if ex is None: ex = " ".join(f"{t}[{l:.1f}|{a}]" if l < -2 else t for t, l, a in rows)
    from collections import Counter
    print(ck, "mean total gold logp", round(sum(tot) / len(tot), 1), "worst-token (gold, argmax) counts:", Counter((w[0], w[1]) for w in worst).most_common(4),
          "mean worst logp", round(sum(w[2] for w in worst) / len(worst), 2))
    print("  example:", ex[-900:])
