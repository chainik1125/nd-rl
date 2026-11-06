#!/usr/bin/env python3
# imports the fork (via harness), which is not resolvable here
# pyright: reportMissingImports=false
"""On the pod: temperature-matched KL of one RL checkpoint against its pretrained starting point, on its own rollouts.

  ND_ARCH=untied python3 tempkl/tempkl.py <ckpt_rl.pt> --pre <pretrain.pt> --tag NAME --out results/NAME.json
  ND_FORK=... python3 tempkl/tempkl.py --smoke [--out /tmp/x.json]        # CPU plumbing check, see below

Rollouts: N_PER_BAND dev theorems per body-line band (fixed by a key hash, so every checkpoint sees the same theorems),
K samples each from the RL checkpoint with the harness's own sampler (T = harness.TEMP, n_loops = EVAL_LEVEL, the
Lean-only gate for the accepted/rejected split). Scoring: both models teacher-forced on each rollout in eval mode,
unaugmented, full logits; only proof-body positions count (the harness's proof mask: everything after the prompt,
<eos> included). Per rollout the script stores the sums over its tokens that analyze.fit() needs to evaluate
KL(pi_RL || softmax(z_pre / T)) exactly for any T: the entropy of pi_RL, E_{pi_RL}[z_pre], and logsumexp(z_pre / T) on
analyze.LOG_T; plus pi_RL at the sampling temperature (suffix _t), the pretrained model's entropy, and both models'
-log p of the sampled tokens. A direct KL at T = 1 is checked against the reconstruction.

--smoke: CPU, two theorems per band, one sample, short caps, 3,000 pretraining records for the model build. Without
--pre it uses the RL weights with the output head scaled by SMOKE_SCALE as the "pretrained" model, so the exact answer
is known: T* = SMOKE_SCALE and residual 0.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), HERE]
import harness  # noqa: E402  (overlay tokenizer + Lean-only gate, fork sampler)
from analyze import LOG_T, RL_TEMP, BANDS, fit  # noqa: E402

torch = harness.torch
AUTOCAST = dict(device_type="cuda", dtype=torch.bfloat16, enabled=harness.DEV == "cuda")
N_PER_BAND, K, SEED = 64, 8, 7777
SCORE_TOKENS = 16 * 1024  # tokens per scoring batch (two models' float32 logits plus the lse grid in chunks)
GRID_CHUNK = 60  # temperatures per logsumexp pass: tokens x GRID_CHUNK x vocab floats
SMOKE_SCALE = 0.5
assert RL_TEMP == harness.TEMP


def pool(recs, n):
    """n records per band, ranked by a hash of the key (fixed across checkpoints)."""
    out = []
    for b in BANDS:
        rs = sorted((r for r in recs if r["band"] == b), key=lambda r: hashlib.sha1(f"tempkl:{r['key']}".encode()).hexdigest())
        out += rs[:n]
    return out


@torch.no_grad()
def score(agent, rl, pre, tok, items):
    """Per (prompt ids, rollout ids) item: the sums over its rollout tokens (see the module docstring)."""
    rl.eval()
    pre.eval()
    ids, mask, _, lens = agent.to_gpu(items, tok, harness.DEV)
    lens_l = lens.tolist()
    order = sorted(range(len(items)), key=lambda i: lens_l[i])
    temps = torch.exp(torch.tensor(LOG_T, dtype=torch.float32, device=harness.DEV))
    rows = [None] * len(items)
    s = 0
    while s < len(order):
        e = s + 1
        while e < len(order) and (e + 1 - s) * lens_l[order[e]] <= SCORE_TOKENS:
            e += 1
        chunk = order[s:e]
        T = lens_l[chunk[-1]]
        x, m = agent.gpu_batch(ids, mask, None, lens, torch.tensor(chunk, device=harness.DEV), tok, None, T, augment=False)
        m = m[:, 1:].double()
        with torch.autocast(**AUTOCAST):
            z_rl, z_pre = rl(x[:, :-1]).float(), pre(x[:, :-1]).float()
        tgt = x[:, 1:, None]
        lp, lpp = torch.log_softmax(z_rl, -1), torch.log_softmax(z_pre, -1)
        lpt = torch.log_softmax(z_rl / RL_TEMP, -1)
        p, pt = lp.exp(), lpt.exp()
        stat = {"H": -(p * lp).sum(-1), "Ez": (p * z_pre).sum(-1), "H_t": -(pt * lpt).sum(-1), "Ez_t": (pt * z_pre).sum(-1),
                "Hpre": -(lpp.exp() * lpp).sum(-1), "nll_rl": -lp.gather(-1, tgt).squeeze(-1),
                "nll_pre": -lpp.gather(-1, tgt).squeeze(-1), "kl1": (p * (lp - lpp)).sum(-1),
                "kl1_t": (pt * (lpt - lpp)).sum(-1)}
        sums = {k: (v.double() * m).sum(1).tolist() for k, v in stat.items()}
        lse = torch.zeros(len(chunk), len(temps), dtype=torch.float64, device=harness.DEV)
        for g in range(0, len(temps), GRID_CHUNK):
            t = temps[g:g + GRID_CHUNK]
            l = torch.logsumexp(z_pre[:, :, None, :] / t[None, None, :, None], -1)  # (B, T, G)
            lse[:, g:g + GRID_CHUNK] = (l.double() * m[:, :, None]).sum(1)
        lse = lse.tolist()
        n = m.sum(1).tolist()
        for j, i in enumerate(chunk):
            rows[i] = {"n": int(n[j]), **{k: v[j] for k, v in sums.items()}, "lse": lse[j]}
        s = e
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--pre", help="the run's pretrain.pt (theta_pre)")
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--out", default="/tmp/tempkl_smoke.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-per-band", type=int, default=N_PER_BAND)
    ap.add_argument("--k", type=int, default=K)
    a = ap.parse_args()
    assert harness.DEV == "cuda" or a.smoke, "no CUDA on this pod"
    assert a.pre or a.smoke, "--pre is required"
    os.environ.setdefault("LEAN_GATE_LOG", os.path.join(os.path.dirname(os.path.abspath(a.out)), "lean_gate.jsonl"))
    agent = __import__("pretrain")
    t0 = time.time()
    train_recs = harness.read(f"{harness.DATA}/pretrain.jsonl.gz")[:3000]  # zero steps: the model build only needs a few
    model, tok, _ = agent.pretrain(train_recs, 1e-9, harness.DEV, 0, harness.log)
    n_per_band, k, max_new = a.n_per_band, a.k, harness.MAX_NEW
    if a.smoke:
        n_per_band, k, max_new = 2, 1, 160
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ck.get("state", ck))  # pretrain.pt/final.pt wrap the state; rl_round_NNN.pt is bare
    pre = copy.deepcopy(model)
    if a.pre:
        ck = torch.load(a.pre, map_location="cpu", weights_only=True)
        pre.load_state_dict(ck.get("state", ck))
    else:  # smoke without --pre: a known answer, T* = SMOKE_SCALE, residual 0
        with torch.no_grad():
            pre.head.weight.mul_(SMOKE_SCALE)
    for mod in (model, pre):
        setattr(mod, harness.KNOB, harness.EVAL_LEVEL)
        mod.eval()
    dev = pool(harness.read(f"{harness.DATA}/dev.jsonl"), n_per_band)
    harness.log(f"{a.tag}: {len(dev)} dev theorems x {k} samples, model built in {time.time() - t0:.0f}s")
    t1 = time.time()
    per = harness.sample(model, tok, dev, k, SEED, max_new)
    secs_sample = time.time() - t1
    items = [(tok.encode_prompt(r["prompt"]), o) for r, samples in zip(dev, per, strict=True) for o, _, _ in samples]
    meta = [{"name": r["name"], "band": r["band"], "body": r["body"], "ok": ok, "eos": o[-1] == tok.eos}
            for r, samples in zip(dev, per, strict=True) for o, _, ok in samples]
    t2 = time.time()
    rows = score(agent, model, pre, tok, items)
    secs_score = time.time() - t2
    rows = [{**mt, **row} for mt, row in zip(meta, rows, strict=True)]
    n = sum(r["n"] for r in rows)
    direct = sum(r["kl1"] for r in rows) / n
    f = fit(rows)
    assert abs(direct - f["raw"]) < 1e-3 * max(1.0, abs(direct)), (direct, f["raw"])  # reconstruction at T = 1
    out = {"tag": a.tag, "ckpt": a.ckpt and os.path.basename(a.ckpt), "sha256": a.ckpt and harness.sha(a.ckpt),
           "pre": a.pre and os.path.basename(a.pre), "pre_sha256": a.pre and harness.sha(a.pre),
           "pre_tag": a.pre and os.path.basename(a.pre).removesuffix(".pt"), "arch": os.environ.get("ND_ARCH"),
           "level": harness.EVAL_LEVEL, "sample_temp": harness.TEMP, "n_per_band": n_per_band, "k": k, "seed": SEED,
           "max_new": max_new, "theorems": len(dev), "smoke": a.smoke, "secs": {"sample": secs_sample, "score": secs_score},
           "summary": f, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    harness.log(f"{a.tag}: raw KL {f['raw']:.4f} nats/tok ({f['raw_per_proof']:.1f}/proof), T* {f['t_star']:.3f}, "
                f"residual {f['residual']:.4f} ({100 * f['residual'] / max(f['raw'], 1e-12):.0f}%), "
                f"H_rl {f['ent_rl']:.3f} H_pre {f['ent_pre']:.3f}, accepted {f['accepted']}/{f['n_rollouts']}, "
                f"sampling {secs_sample:.0f}s scoring {secs_score:.0f}s")
    if a.smoke and not a.pre:
        assert abs(f["t_star"] - SMOKE_SCALE) < 0.01 and f["residual"] < 1e-4, f
        harness.log(f"smoke: T* {f['t_star']:.4f} (expected {SMOKE_SCALE}), residual {f['residual']:.2e}: ok")


if __name__ == "__main__":
    main()
