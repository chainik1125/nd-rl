# ruff: noqa: E501,SIM102,SIM115  (research scripts: long result lines, bare open(), short loop names)
"""Free generation on spine-3 ladder targets: do failed samples die at the stack pops?
Samples k proofs per target at T = 0.8 (harness.sample, Lean-judged). For every sample, walk its own Lean text:
  - max box depth reached
  - first pop error: a `) ; exact nX` whose nX is not the opener of the just-closed box (off-by-one = the enclosing
    box's opener), or <eos> / end while boxes are open
Reports per model: Lean-accepted share, and for rejected samples the share whose first structural error is a pop
error, split by whether the sample reached depth 3.
python gen_check.py <n targets> <k> <models...>
"""

import collections
import contextlib
import io
import json
import os
import random
import sys

os.environ.setdefault("LEAN_GATE_WORKERS", "4")
import nd
import torch

torch.set_num_threads(6)
sys.path.insert(0, f"{nd.E}/analysis_20261014/wall")
with contextlib.redirect_stdout(io.StringIO()):
    import spine as S
N, K = int(sys.argv[1]), int(sys.argv[2])
R = {r["name"]: r for r in map(json.loads, open(f"{nd.E}/data/rl.jsonl"))}
sp3 = sorted(n for n, r in R.items() if r["src"].startswith("ladder") and S.spine(S.goal(r)) == 3)
recs = [R[n] for n in random.Random(0).sample(sp3, N)]


def walk(toks):
    """-> (max depth, first error kind or None)"""
    hv, opener, stack, _depth, maxd = {}, [], [], 0, 0
    for t in range(len(toks)):
        s = toks[t]
        if s == "have" and t + 1 < len(toks):
            hv[len(opener)] = toks[t + 1]
        if s == "(":
            b = t + 2 < len(toks) and toks[t + 1] == "fun" and toks[t + 2] == "("
            stack.append(b)
            if b:
                opener.append(hv.get(len(opener)))
                maxd = max(maxd, len(opener))
        elif s == ")" and stack and stack.pop():
            closed = opener.pop()
            if t + 3 < len(toks) and toks[t + 1] == ";" and toks[t + 2] == "exact" and opener:
                if toks[t + 3] != closed:  # must cite the just-closed box's opener
                    return maxd, ("pop_up" if toks[t + 3] == opener[-1] else "pop_other")
        elif s == "<eos>" and opener:
            return maxd, "eos_open"
    return maxd, None


for mname in sys.argv[3:]:
    model, tok = nd.load(nd.CKPT[mname])
    out = nd.H.sample(model, tok, recs, K, 0, nd.H.MAX_NEW)
    c = collections.Counter()
    for per in out:
        for ids, _ndp, ok in per:
            toks = [tok.itos[i] for i in ids]
            md, err = walk(toks)
            c["n"] += 1
            c["ok"] += ok
            if not ok:
                c["rej"] += 1
                c[f"rej_d{min(md, 3)}"] += 1
                if err:
                    c[f"rej_{err}"] += 1
                    c[f"rej_d{min(md, 3)}_{err}"] += 1
            else:
                c[f"ok_d{min(md, 3)}"] += 1
    solved = sum(any(ok for _, _, ok in per) for per in out)
    print(
        f"\n{mname}: {N} spine-3 targets x {K}: accepted {c['ok']}/{c['n']}, targets solved {solved}/{N}; "
        f"accepted samples by max depth: d<=2 {c['ok_d0'] + c['ok_d1'] + c['ok_d2']}, d3+ {c['ok_d3']}"
    )
    for d in ("d3", "d2"):
        n_ = c[f"rej_{d}"]
        if n_:
            print(
                f"  rejected samples reaching {d}: {n_}; first error: pop off-by-one {c[f'rej_{d}_pop_up']}, "
                f"other wrong pop {c[f'rej_{d}_pop_other']}, <eos> with open boxes {c[f'rej_{d}_eos_open']}, "
                f"no stack error (Lean rejects elsewhere) {n_ - c[f'rej_{d}_pop_up'] - c[f'rej_{d}_pop_other'] - c[f'rej_{d}_eos_open']}"
            )
