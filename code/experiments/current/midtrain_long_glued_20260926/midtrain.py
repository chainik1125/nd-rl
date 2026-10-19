#!/usr/bin/env python3
"""Supervised mid-training of the 039 s0 checkpoint before RL (one arm per call).

    python midtrain.py --arm long|short|more [--gpu auto] [--dry-run]

Mixes (identical cap-6 records across arms; drawn once with seed 0 from frozen_039/data/train_cap6.jsonl):
  long   data/sft_long.jsonl  (2,000 glued ANDI proofs, 16-24 written lines) + cap-6 records 0..7,999
  short  data/sft_short.jsonl (2,000 glued ANDI proofs, 6-11 written lines)  + cap-6 records 0..7,999
  more   cap-6 records 0..9,999 (the same 8,000 plus 2,000 more)
Training: the seed recipe's own fine-tune code path, rl_C.finetune (whole records packed first-fit-decreasing into
rows of at most 1,024 tokens, never truncated; capped abs-index shift; AdamW, lr 3e-4, linear warmup, cosine to 3e-5,
grad clip 1.0, bf16 autocast), with only these CONFIG overrides, identical for all arms:
  ft_steps = round(3 epochs x 10,000 records / ft_batch 128) = 234, ft_batch = 128 (the recipe's), warmup = 35 (the
  recipe's), index_room = max(24, the longest proof in any arm's mix) (room for the longest proof).
Not metered (a pre-RL stage like pretraining). Before training every SFT proof is re-checked with nd_verify.
Writes ckpt/mt_<arm>.pt in the harness's pretrain.pt layout (the 039 checkpoint's keys; `state` replaced, `stage`
= 'midtrain_<arm>'), ckpt/mt_<arm>.json (data, config, steps, tokens, wall time, loss, held-out greedy before and
after) and ckpt/mt_<arm>.metrics.jsonl.
"""
import argparse, collections, hashlib, json, math, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
AB3 = '/scratch/cowifornia/nd-rl/ab3'
FROZEN = f'{AB3}/frozen_039'
LE = f'{AB3}/lengthexp'
for p in (FROZEN + '/lib', FROZEN, LE):
    if p not in sys.path:
        sys.path.insert(0, p)
PRE = f'{FROZEN}/pretrain.pt'
CAP6 = f'{FROZEN}/data/train_cap6.jsonl'
SFT = {'long': os.path.join(HERE, 'data', 'sft_long.jsonl'), 'short': os.path.join(HERE, 'data', 'sft_short.jsonl')}
N_SFT, N_MIX, EPOCHS, BATCH, WARMUP, SEED = 2000, 10000, 3, 128, 35, 0


def read_jsonl(fn):
    with open(fn) as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def proof_of(r):
    for k in ('proof', 'sft_proof', 'constructed_proof', 'glued_proof'):
        if r.get(k):
            return r[k]
    raise KeyError(f'no proof field in record {r.get("name")}: {sorted(r)}')


def mixes():
    """The three arms' records as (prompt, proof, source) lists, and the index room they all use."""
    cap6 = read_jsonl(CAP6)
    rng = random.Random(SEED)
    idx = rng.sample(range(len(cap6)), N_MIX)
    c8 = [(cap6[i]['prompt'], cap6[i]['proof'], 'cap6') for i in idx[:N_MIX - N_SFT]]
    c2 = [(cap6[i]['prompt'], cap6[i]['proof'], 'cap6') for i in idx[N_MIX - N_SFT:]]
    out = {'more': c8 + c2}
    for arm in ('long', 'short'):
        rs = read_jsonl(SFT[arm])
        assert len(rs) == N_SFT, (arm, len(rs))
        out[arm] = [(r['prompt'], proof_of(r), f'sft_{arm}') for r in rs] + c8
    room = max(24, max(p.count(';') for m in out.values() for _, p, _ in m))
    return out, room


def check_sft(mix):
    from nd_verify import verify_text
    bad, lines = 0, collections.Counter()
    for pr, pf, src in mix:
        if src.startswith('sft'):
            ok, why, nl = verify_text(pr + ' ' + pf)
            bad += not ok
            lines[nl] += 1
    return bad, dict(sorted(lines.items()))


class JsonlMetrics:
    def __init__(self, fn):
        self.fn = fn
        open(fn, 'w').close()

    def log(self, **kw):
        with open(self.fn, 'a') as f:
            f.write(json.dumps(dict(kw, t=time.time())) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['long', 'short', 'more'])
    ap.add_argument('--gpu', default='auto')
    ap.add_argument('--gpus', default='0,1,2,3,4')
    ap.add_argument('--gpu-wait', type=float, default=4 * 3600)
    ap.add_argument('--dry-run', action='store_true', help='build and check the mixes only (CPU)')
    ap.add_argument('--smoke', action='store_true', help='code-path test: 10 steps on 1,000 cap-6 records -> ckpt/smoke/')
    a = ap.parse_args()
    a.label = f'midtrain_{a.arm}' + ('_smoke' if a.smoke else '')
    odir = os.path.join(HERE, 'ckpt', 'smoke') if a.smoke else os.path.join(HERE, 'ckpt')
    os.makedirs(odir, exist_ok=True)
    if a.smoke:
        cap6 = read_jsonl(CAP6)
        mix = [(r['prompt'], r['proof'], 'cap6') for r in random.Random(SEED).sample(cap6, 1000)]
        room, steps = 24, 10
    else:
        mx, room = mixes()
        mix = mx[a.arm]
        steps = round(EPOCHS * N_MIX / BATCH)
    bad, hist = check_sft(mix)
    info = {'arm': a.arm, 'n_records': len(mix), 'by_source': dict(collections.Counter(s for _, _, s in mix)),
            'sft_verify_failures': bad, 'sft_written_lines': hist, 'index_room': room, 'ft_steps': steps, 'ft_batch': BATCH,
            'warmup': WARMUP, 'epochs': EPOCHS, 'seed': SEED, 'init': PRE, 'init_sha256': sha256(PRE), 'cap6_sha256': sha256(CAP6),
            'sft_file': SFT.get(a.arm), 'sft_sha256': sha256(SFT[a.arm]) if a.arm in SFT and not a.smoke else None, 'smoke': a.smoke}
    print(json.dumps({k: v for k, v in info.items() if k != 'sft_written_lines'}), flush=True)
    assert bad == 0, f'{bad} SFT proofs fail nd_verify'
    if a.dry_run:
        return 0
    import evaluate as E                                   # lengthexp: the GPU lock lease
    lease = E.Lease()
    os.environ.setdefault('AB_WORKER', 'midtrain')
    try:
        lease.take(a)
        import torch
        import rl_C as rl                                  # the seed recipe (lengthexp/rl_C.py), for finetune()
        import harness as H                                # frozen_039/harness.py: heldout_greedy, N_LAYER...
        from model import GPT
        from tokenizer import Tokenizer
        torch.manual_seed(SEED)
        tok = Tokenizer('abs')
        ck = torch.load(PRE, map_location='cpu', weights_only=False)
        model = GPT(tok.vocab_size, H.N_LAYER, H.D_MODEL, H.N_HEAD)
        model.load_state_dict(ck['state'])
        model.to('cuda').eval()
        recs = H.heldout_slice()
        hg0 = H.heldout_greedy(model, tok, recs=recs)
        data = [(tok.encode_prompt(p), tok.encode_proof(q)) for p, q, _ in mix]
        info['max_seq_tokens'] = max(len(p) + len(q) for p, q in data)
        cfg = dict(rl.CONFIG, ft_steps=steps, ft_batch=BATCH, warmup=WARMUP, index_room=room)
        budget = rl.Budget(None, 10 ** 9, cfg)            # no meter: fits() is always true
        rates = {'row': 0.0}
        state = {'step': 0, 'round': 0}
        mfn = os.path.join(odir, f'mt_{a.arm}.metrics.jsonl')
        t0 = time.time()
        done, secs, _, fst = rl.finetune(model, tok, data, cfg, random.Random(SEED), 'cuda', budget, rates, JsonlMetrics(mfn), state)
        torch.cuda.synchronize()
        wall = time.time() - t0
        hg1 = H.heldout_greedy(model, tok, recs=recs)
        out = {k: v for k, v in ck.items() if k != 'state'}
        out['state'] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        out['stage'] = f'midtrain_{a.arm}'
        fn = os.path.join(odir, f'mt_{a.arm}.pt')
        torch.save(out, fn + '.tmp')
        os.replace(fn + '.tmp', fn)
        losses = [json.loads(l) for l in open(mfn)]
        info.update(steps_done=done, wall_secs=wall, finetune_secs=secs, tokens_total=fst.get('nonpad'), loss_tokens=fst.get('loss_tokens'),
                    records_seen=fst.get('records'), pad_frac=fst.get('pad_frac'), loss_per_token=fst.get('loss_per_token'),
                    loss_first=losses[0]['loss'] if losses else None, loss_last=losses[-1]['loss'] if losses else None,
                    heldout_greedy_before=hg0, heldout_greedy_after=hg1, ckpt=fn, ckpt_sha256=sha256(fn),
                    gpu=f"{os.environ.get('CUDA_VISIBLE_DEVICES')}:{torch.cuda.get_device_name(0)}")
        with open(os.path.join(odir, f'mt_{a.arm}.json'), 'w') as f:
            json.dump(info, f, indent=1)
        print(f"midtrain {a.arm}: {done} steps in {wall:.0f}s, {info['tokens_total']} tokens, loss {info['loss_first']:.3f} -> "
              f"{info['loss_last']:.3f}, held-out greedy {hg0:.3f} -> {hg1:.3f}, saved {fn}", flush=True)
    finally:
        lease.drop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
