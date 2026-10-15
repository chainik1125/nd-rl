# ruff: noqa: E402,E501,E741,SIM115  (research scripts: long result lines, bare open(), short loop names)
"""Print one depth-3 proof with per-token log p under several checkpoints, marking tokens where they differ."""

import json
import sys

import nd
import torch

torch.set_num_threads(4)
A = "/Users/robbie/repos/nd-rl/artifacts/long_rl_20261014"
sys.path.insert(0, f"{nd.E}/analysis_20261014/wall")
import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):
    import spine as S
R = {r["name"]: r for r in map(json.loads, open(f"{nd.E}/data/rl.jsonl"))}
F = {}
for r in map(json.loads, open(f"{A}/v2/long-rl-20261014-v2_untied-ei_s1/found_targets.jsonl")):
    if r["name"] not in F or r["written"] < F[r["name"]]["written"]:
        F[r["name"]] = r
names = sorted(
    n for n, r in R.items() if r["src"].startswith("ladder") and S.spine(S.goal(r)) == 3 and n in F
)
MS = ["g0_pre", "g0_fin", "g1_pre", "g1_fin"]
M = {m: nd.load(nd.CKPT[m]) for m in MS}
tok = M["g0_pre"][1]
for n in names[int(sys.argv[1]) : int(sys.argv[1]) + int(sys.argv[2])]:
    f = F[n]
    ids, n_p = nd.encode(tok, f["prompt"], f["proof"], 0)
    L = nd.labels(tok, ids, n_p)
    x = torch.tensor([ids])
    lp = {}
    for m, (mod, _) in M.items():
        with torch.no_grad():
            l = torch.log_softmax(mod(x[:, :-1])[0].float(), -1)
        lp[m] = (
            l[torch.arange(n_p - 1, len(ids) - 1), x[0, n_p:]].tolist(),
            [tok.itos[a] for a in l[n_p - 1 :].argmax(-1).tolist()],
        )
    print("\n#", n, f["prompt"])
    out = []
    for j, t in enumerate(L):
        v = [lp[m][0][j] for m in MS]
        if min(v) < -1.0:
            out.append(
                f"[{t['tok']}^{t['depth']} "
                + " ".join(f"{m}:{lp[m][0][j]:.1f}({lp[m][1][j]})" for m in MS)
                + "]"
            )
        else:
            out.append(t["tok"])
    print(" ".join(out))
