#!/usr/bin/env python3
"""Peek: sample a checkpoint on a few targets and print verdicts + one proof per target. python peek_samples.py CKPT TARGETS N K"""
import sys, json, torch
from model import load_ckpt
from sample import generate
from nd_verify import verify_text
ck, fn, n, k = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
model, tok, _ = load_ckpt(ck, "cuda")
T = [json.loads(l) for l in open(fn)][:n]
outs = generate(model, tok, [t['prompt'] for t in T for _ in range(k)], greedy=False, temperature=0.8, batch=256)
for i, t in enumerate(T):
    res = [verify_text(t['prompt'] + ' ' + p) for p in outs[i * k:(i + 1) * k]]
    ok = sum(r[0] for r in res)
    print(f"## {t['thm']}  (n_lines {t['n_lines']}) ok {ok}/{k}; reasons {sorted(set(r[1] for r in res))[:4]}")
    print('   sample:', outs[i * k][:400])
    if ok: print('   OK    :', [p for p, r in zip(outs[i * k:(i + 1) * k], res) if r[0]][0][:400])
