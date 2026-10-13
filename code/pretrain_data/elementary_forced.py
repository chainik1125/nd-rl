"""How much probability does one of Dan's checkpoints put on the CORRECT premise block of an n-premise prompt?
Teacher-forced scoring through his environment (canonical names, so the name token is fixed by the state):

  step k (after the correct lines h1..h(k-1) were applied): score the k candidate actions
  `have n<next> : <premise_j> := h<j> ;` for every premise j, as log p(action | state); report the absolute
  probability of the correct one (j = k) and its share among the candidates; multiply over k for the whole block.

Then a plain search: N samples of a four-premise prompt at temperature 1.0 and 1.5 through env_generate, counting
verified proofs, which is the signal expert iteration would see.

    cd code && ND_REGISTRY=0 uv run python pretrain_data/elementary_forced.py --ckpt la_T1_best12_s0_r8 --n 1024
Writes data/pretrain_data/screens/elementary_forced_<ckpt>.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "textbook_problem")]
import pool_common as PC  # noqa: E402
import screen_ckpt_dan as SD  # noqa: E402
from strict_verify import safe_verify  # noqa: E402

OUT = os.path.join(PC.DATA, "screens")
LADDER = [
    "THM P SEQ P PRF",
    "THM P , Q SEQ P PRF",
    "THM P , Q , R SEQ P PRF",
    "THM P , Q , R , S SEQ P PRF",
    "THM P , Q , R , S , ( P & Q ) SEQ P PRF",
    "THM P , ( P > Q ) , R , S SEQ Q PRF",
    "THM ( P > Q ) , ( Q > R ) , P , ( R > S ) SEQ S PRF",
]
SEARCH = "THM P , Q , R , S SEQ P PRF"


def action_logprob(model, tok, state_ids, act_ids):
    """log p(act_ids | state_ids) under the model, one forward pass, no cache."""
    import torch
    ids = torch.tensor([state_ids + act_ids[:-1]])
    with torch.no_grad():
        logits = model(ids)[0]
    lp = torch.log_softmax(logits.float(), -1)
    s = len(state_ids)
    return sum(lp[s - 1 + t, act_ids[t]].item() for t in range(len(act_ids)))


def forced_block(model, tok, prompt):
    import state_sample as SS
    from lean_tok import ftoks
    env = SS.Env(prompt, canon=True, base=0, assign=True)
    n = len(env.prem)
    steps = []
    for k in range(1, n + 1):
        state_ids = SS.prompt_ids(tok, env)
        name = f"n{env.next_name()}"
        cands = {}
        for j in range(1, n + 1):
            act = ["have", name, ":"] + ftoks(env.prem[j - 1]) + [":=", f"h{j}", ";"]
            cands[j] = action_logprob(model, tok, state_ids, tok.encode_toks(act) + [tok.eos])
        z = sum(math.exp(v) for v in cands.values())
        steps.append(dict(k=k, p_correct=math.exp(cands[k]), share_correct=math.exp(cands[k]) / z if z else 0.0,
                          p_by_premise={str(j): round(math.exp(v), 6) for j, v in cands.items()}))
        ok, why = env.apply(["have", name, ":"] + ftoks(env.prem[k - 1]) + [":=", f"h{k}", ";"])
        assert ok, why
    p_block = math.prod(s["p_correct"] for s in steps)
    return dict(prompt=prompt, n_prem=n, steps=steps, p_block=p_block, share_block=math.prod(s["share_correct"] for s in steps))


def search(model, tok, prompt, n, temperature, seed):
    stats = {}
    bodies = SD.env_generate(model, tok, [prompt] * n, greedy=False, temperature=temperature, max_action=256, max_steps=48,
                             seed=seed, batch=256, stats=stats)
    ok = sum(1 for b in bodies if b and not b.startswith("LEANPARSE") and safe_verify(prompt, b)[0])
    reasons = collections.Counter((b or "")[:40] for b in bodies if not b or b.startswith("LEANPARSE"))
    return dict(prompt=prompt, n=n, temperature=temperature, verified=ok, fail_reasons=dict(reasons.most_common(6)),
                env_fail_reason=dict(stats.get("env_fail_reason", {})))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="la_T1_best12_s0_r8")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    SD.load_dan()
    model, tok, extra = SD.load_ckpt(os.path.join(SD.CKPTS, a.ckpt + ".pt"))
    model.eval()
    out = dict(ckpt=a.ckpt, forced=[], search=[])
    for prompt in LADDER:
        r = forced_block(model, tok, prompt)
        out["forced"].append(r)
        print(f"{prompt:<52} p(block) {r['p_block']:.2e}  share {r['share_block']:.3f}  per step p_correct",
              [f"{s['p_correct']:.3f}" for s in r["steps"]], "first-step mass by premise", r["steps"][0]["p_by_premise"], flush=True)
    for T in (1.0, 1.5):
        t0 = time.time()
        r = search(model, tok, SEARCH, a.n, T, a.seed)
        out["search"].append(r)
        print(f"search {SEARCH} T {T}: verified {r['verified']} of {a.n} ({time.time() - t0:.0f}s); env fail reasons {r['env_fail_reason']}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f"elementary_forced_{a.ckpt}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
