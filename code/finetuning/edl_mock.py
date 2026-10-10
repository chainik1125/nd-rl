"""Prequential EDL (Donoway et al. 2026, Def. 3.2) of one proof stream under one checkpoint (finetuning / edl_mock).

    cd code && ND_OFFLINE=1 uv run python finetuning/edl_mock.py --ckpt <pt> --train stream_new_train.jsonl \
        --held stream_new_held.jsonl --arm full --lr 3e-4 --epochs 2 --out run.jsonl

Algorithm A (fixed per arm): from the SAME starting weights, train on the first n_i examples for --epochs passes,
batch --recs proofs, AdamW(betas .9/.95, wd 0), grad clip 1, constant lr; "m<d>" arms update only d random
weights (mask seed --mask_seed). Then code the block [n_i, n_{i+1}) under that model: sum of next-token NLL over
action tokens (Dan's state_train.loss_on, the loss on action tokens only), with a fixed renaming seed per block
so every checkpoint codes the same strings. theta* = A on all n; L_test = its per-token loss on --held.
EDL = sum_blocks nats - (stream action tokens) * L_test. Every quantity is logged in nats and bits, per block.
Uses Dan's worktree read-only (DAN_FORK, default ~/.cache/nd-rl/fork_dan).
"""
import argparse, json, math, os, random, sys, time
import torch
FD = os.environ.get("DAN_FORK", os.path.expanduser("~/.cache/nd-rl/fork_dan"))
sys.path.insert(0, FD)
from model import load_ckpt  # noqa: E402  (Dan's)
import state_train as ST  # noqa: E402  (Dan's)


def code_block(model, data, idxs, tok, seed):
    """-> (sum nats over action tokens, n action tokens) for examples idxs, fixed renaming seed."""
    rng = random.Random(seed)
    model.eval(); tot = n = 0
    with torch.no_grad():
        for s in range(0, len(idxs), 32):
            x, m = ST.batch(data, idxs[s:s + 32], tok, rng, "cpu")
            k = int(m[:, 1:].sum()); tot += ST.loss_on(model, x, m).item() * k; n += k
    model.train(); return tot, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--train", required=True); ap.add_argument("--held", required=True)
    ap.add_argument("--arm", default="full"); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--recs", type=int, default=8)
    ap.add_argument("--prefixes", default="0,8,16,32,64,128"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask_seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    model, tok, _ = load_ckpt(a.ckpt, "cpu")
    init = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train, sk1 = ST.load(a.train, tok, 0); held, sk2 = ST.load(a.held, tok, 0)
    assert sk1 == sk2 == 0, (sk1, sk2)
    params = list(model.parameters()); masks = None
    if a.arm != "full":
        d = int(a.arm[1:]); N = sum(p.numel() for p in params); mr = random.Random(a.mask_seed)
        pick = sorted(mr.sample(range(N), d)); masks = []; off = j = 0
        for p in params:
            mk = torch.zeros(p.numel(), dtype=torch.bool)
            while j < len(pick) and pick[j] < off + p.numel():
                mk[pick[j] - off] = True; j += 1
            masks.append(mk.view_as(p)); off += p.numel()
    pre = [int(x) for x in a.prefixes.split(",")]; n_all = len(train)
    assert pre[0] == 0 and pre[-1] <= n_all
    bounds = pre + [n_all] if pre[-1] < n_all else pre
    out = open(a.out, "w"); t0 = time.time()
    meta = dict(ckpt=os.path.basename(a.ckpt), arm=a.arm, lr=a.lr, epochs=a.epochs, recs=a.recs, seed=a.seed,
                mask_seed=a.mask_seed, train=os.path.basename(a.train), held=os.path.basename(a.held), n=n_all)

    def fit(n):
        model.load_state_dict(init)
        if n == 0:
            return 0
        rng = random.Random(a.seed * 1000 + n); opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
        steps = max(1, math.ceil(a.epochs * n / a.recs)); perm = []
        for _ in range(steps):
            if len(perm) < a.recs:
                perm = list(range(n)); rng.shuffle(perm)
            idx = [perm.pop() for _ in range(min(a.recs, n))]
            x, m = ST.batch(train, idx, tok, rng, "cpu"); opt.zero_grad(set_to_none=True)
            l = ST.loss_on(model, x, m); l.backward()
            if masks:
                for p, mk in zip(params, masks):
                    p.grad.mul_(mk)
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        return steps

    total_nats = total_tok = 0
    for i in range(len(bounds) - 1):
        n0, n1 = bounds[i], bounds[i + 1]
        steps = fit(n0)
        nats, ntok = code_block(model, train, list(range(n0, n1)), tok, 777 + i)
        hn, hk = code_block(model, held, list(range(len(held))), tok, 999)
        total_nats += nats; total_tok += ntok
        r = dict(kind="block", prefix=n0, block=[n0, n1], steps=steps, block_nats=nats, block_tokens=ntok,
                 block_bits_per_example=nats / math.log(2) / (n1 - n0), block_nats_per_token=nats / ntok,
                 held_nats_per_token=hn / hk, secs=round(time.time() - t0), **meta)
        out.write(json.dumps(r) + "\n"); out.flush(); print({k: r[k] for k in ("prefix", "steps", "block_bits_per_example", "held_nats_per_token", "secs")}, flush=True)
    steps = fit(n_all)
    hn, hk = code_block(model, held, list(range(len(held))), tok, 999)
    tn, tk = code_block(model, train, list(range(n_all)), tok, 998)
    L_test = hn / hk; edl_nats = total_nats - total_tok * L_test
    r = dict(kind="final", steps=steps, prequential_nats=total_nats, stream_tokens=total_tok, L_test_nats_per_token=L_test,
             L_train_star_nats_per_token=tn / tk, edl_bits=edl_nats / math.log(2), edl_bits_per_example=edl_nats / math.log(2) / n_all,
             prequential_bits_per_example=total_nats / math.log(2) / n_all, secs=round(time.time() - t0), **meta)
    out.write(json.dumps(r) + "\n"); out.close(); print({k: r[k] for k in ("edl_bits_per_example", "prequential_bits_per_example", "L_test_nats_per_token", "secs")}, flush=True)


if __name__ == "__main__":
    main()
