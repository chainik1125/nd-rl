"""Action-by-action trace of Dan's checkpoint on a few elementary prompts (3 and 4 premises), mirroring
state_sample.env_generate's loop, so the exact action that his environment rejects is visible.

    cd code && ND_REGISTRY=0 uv run python pretrain_data/elementary_trace.py --ckpt la_T1_best12_s0_r8 --n 6
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "textbook_problem")]
import screen_ckpt_dan as SD  # noqa: E402

PROMPTS = [
    "THM P , Q , R SEQ P PRF",
    "THM P , Q , R , S SEQ P PRF",
    "THM P , ( P > Q ) , R , S SEQ Q PRF",
    "THM P , ( P > Q ) , ( Q > R ) , ( R > S ) SEQ S PRF",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="la_T1_best12_s0_r8")
    ap.add_argument("--n", type=int, default=6, help="samples per prompt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    SD.load_dan()
    import state_sample as SS
    import torch
    model, tok, extra = SD.load_ckpt(os.path.join(SD.CKPTS, a.ckpt + ".pt"))
    for prompt in PROMPTS:
        print("\n==", prompt)
        envs = [SS.Env(prompt) for _ in range(a.n)]
        traces = [[] for _ in envs]
        live = list(range(a.n))
        for step in range(48):
            if not live:
                break
            pids = [SS.prompt_ids(tok, envs[i]) for i in live]
            with torch.no_grad():
                outs = SS.generate_ids_fast(model, tok, pids, goals=None, greedy=False, temperature=1.0, max_new=256,
                                            seed=a.seed * 100003 + step, early="eos", compact=True, stats={})
            keep = []
            for i, o in zip(live, outs):
                atoks, ended = tok.decode_action(o)
                e = envs[i]
                if not ended:
                    traces[i].append(("TRUNCATED", " ".join(atoks)[:120]))
                    continue
                ok, why = e.apply(atoks)
                traces[i].append(("ok" if ok else f"REJECT {why}", " ".join(atoks)))
                if ok and not e.done:
                    keep.append(i)
                elif ok and e.done:
                    traces[i].append(("DONE", e.nd()[:160]))
            live = keep
        ends = collections.Counter(t[-1][0].split(" ")[0] + (" " + t[-1][0].split(" ", 1)[1] if t[-1][0].startswith("REJECT") else "") for t in traces)
        print("   endings:", dict(ends))
        # which premise indices the model cites in its leading premise lines, in order, up to the first rejection
        seqs = collections.Counter()
        for t in traces:
            hs = []
            for tag, act in t:
                toks = act.split()
                if len(toks) >= 2 and toks[-2].startswith("h") and toks[-2][1:].isdigit() and toks[-1] == ";":
                    hs.append(toks[-2])
                else:
                    break
                if tag.startswith("REJECT"):
                    break
            seqs[" ".join(hs) or "(no premise line)"] += 1
        print("   premise-line sequences:", dict(seqs.most_common(8)))
        for t in traces[: min(3, a.n)]:
            print("   --")
            for tag, act in t:
                print(f"   {tag:<28} {act}")


if __name__ == "__main__":
    main()
