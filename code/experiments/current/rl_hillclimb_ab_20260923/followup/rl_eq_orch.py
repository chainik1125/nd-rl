"""SEED RL file: plain expert iteration (rejection-sampling fine-tuning against the verifier).

This is the one file the agents edit. The frozen harness calls

    model = run(model, tok, targets, budget_secs, device, seed, log, metrics, paths) -> model

  model        lib.model.GPT(vocab, 4, 256, 8) loaded from frozen/pretrain.pt, on `device`
  tok          lib.tokenizer.Tokenizer('abs')
  targets      list of dicts from targets_L7-9.jsonl (name, thm, key, prompt, n_lines, L_true, ...; no proofs)
  budget_secs  wall clock for the whole call (600). SIGALRM at budget+45; returning after budget+15 is a
               timeout. This file stops with >= CONFIG['margin_secs'] to spare.
  device       'cuda'
  seed         int (the run seed)
  log(msg)     timestamped print
  metrics      frozen metrics.Metrics: metrics.log(phase=..., step=..., **fields); metrics.set_summary(**kv)
  paths        dict: targets_extra, train_cap6, heldout_cap6, out_dir (files you may read / write under),
               save_checkpoint(model, tok, path), heldout_greedy(model, tok, n=1000) -> float, smoke (bool)
Return the model to be scored (same architecture).

Rules (SPEC.md "Rules for agents"): the only training data allowed are (a) proofs this model sampled
during this run that nd_verify accepts for the prompted sequent (targets or extra targets), (b) verified
by-products of those samples (hindsight relabelling to the conclusion actually proved from the same
premises) and (c) records from train_cap6.jsonl. heldout_cap6.jsonl is for tracking only. No proofs
from any other source, and never edit anything under frozen/.

Each round: sample k proofs per target -> keep verified, distinct (start-index normalised) proofs,
accumulated across rounds -> fine-tune on <= max_per_thm proofs per solved target (each repeated
rl_weight times) mixed with `retain` random cap-6 records -> checkpoint.
"""
import json, math, random, sys, time, collections

import torch
from sample import generate          # frozen/lib is on sys.path (the harness puts it there)
from model import rope_cache
from train import batch as make_batch, loss_on
from nd_verify import verify_text
from normalize import norm
from prune import pruned_length

# ---------------------------------------------------------------- knobs
CONFIG = dict(
    rounds=4,              # EQ-SAMPLES: 1321x16 + 3 x (1321x16 + 830x32) ~= 164,224 samples (skip_saturated may trim a little)
    fixed_rounds=True,     # EQ-SAMPLES: ignore the clock for round shrinking / stopping
    n_targets=1500,        # targets sampled (with the run seed) from `targets`, fixed across rounds
    k=16,                  # samples per target per round
    temperature=0.8,
    max_new=512,           # max generated tokens per proof
    sample_batch=384,      # prompts per generate() call (1024 does not fit a 3090 next to training)
    max_per_thm=4,         # at most this many distinct proofs per target in each round's mix
    rl_weight=4,           # each selected proof is repeated this many times in the mix
    retain=8000,           # cap-6 train records mixed in each round (retention of in-distribution skill)
    ft_steps=300,          # fine-tune steps per round (fresh AdamW each round)
    ft_batch=128,
    lr=3e-4, min_lr=3e-5, warmup=50, weight_decay=0.1, betas=(0.9, 0.95), grad_clip=1.0,
    log_every=50,          # train-phase metrics every this many steps
    eval_each_round=False, # greedy held-out check after every round (~15 s; the harness always does a final one)
    eval_n=1000,
    tail_min_frac=0.25,    # last round: if a full round will not fit, run a shrunk one (k and ft steps scaled) if >= this fraction fits
    skip_saturated=True,   # do not resample targets that already have >= max_per_thm distinct proofs
    tb_k=32,               # textbook exploration: samples per textbook target per round (all 830 textbook targets, pool + extra)
    tb_temperature=1.0,    # ... at this temperature
    tb_start_round=2,      # ... from this round on
    margin_secs=20,        # return at least this long before budget_secs
    default_step_secs=0.12,  # fine-tune secs/step guess until measured
)
SMOKE = dict(n_targets=8, k=2, max_new=128, ft_steps=5, retain=64, ft_batch=32, rounds=2, eval_n=16, margin_secs=5, log_every=1, eval_each_round=True)


# ---------------------------------------------------------------- helpers

def read_jsonl(fn, limit=None):
    out = []
    with open(fn) as f:
        for l in f:
            if l.strip():
                out.append(json.loads(l))
                if limit and len(out) >= limit:
                    break
    return out


class Clock:
    def __init__(self, budget, margin):
        self.t0 = time.time()
        self.deadline = self.t0 + budget - margin

    def elapsed(self):
        return time.time() - self.t0

    def left(self):
        return self.deadline - time.time()


@torch.no_grad()
def fast_generate_ids(model, tok, prompt_ids, temperature, max_new, gen, compact_every=8, compact_frac=0.8):
    """Temperature sampling with a KV cache, like lib.sample.generate_ids, but (a) rows that have emitted EOS are
    dropped from the batch (and from the KV cache) every `compact_every` steps once fewer than `compact_frac` of
    the rows are still running, and (b) no per-step host sync (RoPE tables precomputed; the all-done check runs
    only at compaction points). Same model, same per-token distribution; returns lists of generated ids."""
    dev = next(model.parameters()).device
    B = len(prompt_ids)
    L = max(len(p) for p in prompt_ids)
    idx = torch.full((B, L), tok.pad, dtype=torch.long, device=dev)
    keep = torch.zeros((B, L), dtype=torch.bool, device=dev)
    for i, p in enumerate(prompt_ids):
        idx[i, L - len(p):] = torch.tensor(p, device=dev)
        keep[i, L - len(p):] = True
    pos = (keep.cumsum(1) - 1).clamp(min=0)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    mask = causal[None, None] & keep[:, None, None, :]
    mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]
    caches = [{'max': L + max_new} for _ in model.blocks]
    logits = model(idx, pos=pos, mask=mask, caches=caches)[:, -1]
    cos_all, sin_all = rope_cache(L + max_new + 1, model.hd, dev)
    out = torch.full((B, max_new), tok.pad, dtype=torch.long, device=dev)
    rows = torch.arange(B, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    cur_pos = pos[:, -1]
    for t in range(max_new):
        probs = torch.softmax(logits.float() / temperature, -1)
        nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, tok.pad), nxt)
        out[rows, t] = nxt
        done = done | (nxt == tok.eos)
        if t % compact_every == compact_every - 1:
            alive = ~done
            n_alive = int(alive.sum())
            if n_alive == 0:
                break
            if n_alive < compact_frac * len(rows):
                sel = alive.nonzero().squeeze(1)
                rows, done, nxt, cur_pos, keep = rows[sel], done[sel], nxt[sel], cur_pos[sel], keep[sel]
                for c in caches:
                    c['k'], c['v'] = c['k'][sel], c['v'][sel]
        cur_pos = cur_pos + 1
        keep = torch.cat([keep, torch.ones(len(rows), 1, dtype=torch.bool, device=dev)], 1)
        m = keep[:, None, None, :]
        x = model.emb(nxt[:, None])
        cos, sin = cos_all[cur_pos[:, None]], sin_all[cur_pos[:, None]]
        for i, blk in enumerate(model.blocks):
            x = blk(x, cos, sin, m, caches[i])
        logits = model.head(model.ln_f(x))[:, -1]
    return out.tolist()


def fast_generate(model, tok, prompts, temperature, max_new, batch, seed):
    """Drop-in for lib.sample.generate(..., greedy=False) using fast_generate_ids."""
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    ids = [tok.encode_prompt(p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(ids[i]))
    res = [None] * len(prompts)
    for s in range(0, len(order), batch):
        chunk = order[s:s + batch]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev.type == 'cuda')):
            outs = fast_generate_ids(model, tok, [ids[i] for i in chunk], temperature, max_new, gen)
        for i, o in zip(chunk, outs):
            res[i] = tok.decode(o)
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return res


def sample_round(model, tok, targets, cfg, seed, clock, active=None):
    """Sample cfg['k'] proofs per target in chunks of sample_batch prompts (so the clock is checked often).
    Returns (list of (target_index, proof) pairs, seconds). Stops early at the deadline."""
    model.eval()
    active = range(len(targets)) if active is None else active
    prompts = [(i, targets[i]['prompt']) for i in active for _ in range(cfg['k'])]
    prompts.sort(key=lambda x: len(x[1].split()))       # similar lengths per batch = less padding
    out, t0 = [], time.time()
    for c, s in enumerate(range(0, len(prompts), cfg['sample_batch'])):
        if clock.left() <= 0:
            break
        chunk = prompts[s:s + cfg['sample_batch']]
        proofs = fast_generate(model, tok, [p for _, p in chunk], temperature=cfg['temperature'],
                               max_new=cfg['max_new'], batch=cfg['sample_batch'], seed=seed * 100003 + c)
        out += [(i, q) for (i, _), q in zip(chunk, proofs)]
    return out, time.time() - t0


def judge_samples(targets, samples, found, rnd):
    """Verify each sample against its prompted sequent; add new distinct proofs to `found`
    ({target name: {normalised proof: record}}). Returns the sample-phase statistics."""
    n_ok = n_qed = new = 0
    solved_now, wl, pl, reasons = set(), [], [], collections.Counter()
    for i, proof in samples:
        t = targets[i]
        n_qed += 'QED' in proof.split()
        ok, reason, nl = verify_text(t['prompt'] + ' ' + proof)
        if not ok:
            reasons[reason.split(' (line')[0]] += 1
            continue
        n_ok += 1
        solved_now.add(t['name'])
        pr = pruned_length(t['prompt'], proof)
        wl.append(nl)
        pl.append(pr)
        key = norm(proof)
        if key not in found[t['name']]:
            found[t['name']][key] = {'proof': proof, 'written': nl, 'pruned': pr, 'round': rnd}
            new += 1
    n = max(len(samples), 1)
    solved_cum = {name for name, d in found.items() if d}
    by_len = collections.Counter(str(t['L_true']) for t in targets if t['name'] in solved_cum)
    by_len_round = collections.Counter(str(t['L_true']) for t in targets if t['name'] in solved_now)
    return {'samples': len(samples), 'accept_rate': n_ok / n, 'qed_rate': n_qed / n,
            'targets_solved_round': len(solved_now), 'targets_solved_cum': len(solved_cum), 'new_proofs': new,
            'solved_by_len': {L: by_len.get(L, 0) for L in ('7', '8', '9')},
            'solved_by_len_round': {L: by_len_round.get(L, 0) for L in ('7', '8', '9')},
            'mean_written_len': sum(wl) / len(wl) if wl else None, 'mean_pruned_len': sum(pl) / len(pl) if pl else None,
            'fail_reasons': [[r, c] for r, c in reasons.most_common(5)]}


def build_mix(targets, found, retained, cfg, rng, tok):
    """Encoded fine-tuning records: <= max_per_thm proofs per solved target x rl_weight, plus `retain`
    random cap-6 records. Returns (data, n_rl_records)."""
    mix = []
    for t in targets:
        fs = list(found[t['name']].values())
        rng.shuffle(fs)
        for x in fs[:cfg['max_per_thm']]:
            mix += [(t['prompt'], x['proof'])] * cfg['rl_weight']
    n_rl = len(mix)
    mix += [(r['prompt'], r['proof']) for r in rng.sample(retained, min(cfg['retain'], len(retained)))]
    return [(tok.encode_prompt(p), tok.encode_proof(q)) for p, q in mix], n_rl


def finetune(model, tok, data, cfg, rng, device, clock, metrics, state):
    """AdamW, linear warmup then cosine to min_lr, grad clipping, bf16 autocast. Stops at the deadline.
    Returns (steps done, seconds)."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], betas=cfg['betas'])
    steps, warm, bs = cfg['ft_steps'], cfg['warmup'], min(cfg['ft_batch'], len(data))

    def lr_at(s):
        if s < warm:
            return cfg['lr'] * s / warm
        return cfg['min_lr'] + 0.5 * (cfg['lr'] - cfg['min_lr']) * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))

    model.train()
    perm, t0, done = [], time.time(), 0
    for s in range(1, steps + 1):
        if clock.left() <= 0:
            break
        if len(perm) < bs:
            perm = list(range(len(data)))
            rng.shuffle(perm)
        idxs = [perm.pop() for _ in range(bs)]
        x, m = make_batch(data, idxs, tok, rng, device)
        lr = lr_at(s)
        for g in opt.param_groups:
            g['lr'] = lr
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            loss = loss_on(model, x, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        opt.step()
        done = s
        state['step'] += 1
        if s % cfg['log_every'] == 0 or s == steps:
            metrics.log(phase='train', step=state['step'], round=state['round'], loss=loss.item(), lr=lr,
                        grad_norm=float(gn), n_records=len(data), tokens=int((x != tok.pad).sum()), batch=bs)
    model.eval()
    return done, time.time() - t0


# ---------------------------------------------------------------- entry point

def run(model, tok, targets, budget_secs, device, seed, log, metrics, paths):
    cfg = dict(CONFIG, **(SMOKE if paths.get('smoke') else {}))
    clock = Clock(budget_secs, cfg['margin_secs'])
    rng = random.Random(seed)
    pool = targets
    targets = [t for t in rng.sample(pool, min(cfg['n_targets'], len(pool))) if t['source'] != 'textbook']
    n_main = len(targets)
    tb = [t for t in pool if t['source'] == 'textbook'] + [t for t in read_jsonl(paths['targets_extra']) if t['source'] == 'textbook']
    targets = targets + (tb[:4] if paths.get('smoke') else tb)
    log(f'rl: {n_main} main targets (gen) + {len(targets) - n_main} textbook targets')
    retained = read_jsonl(paths['train_cap6'], limit=cfg['retain'] * 4 if paths.get('smoke') else None)
    assert all(r['n_lines'] <= 6 for r in retained), 'train_cap6 must only hold proofs of <= 6 lines'
    retained = [{'prompt': r['prompt'], 'proof': r['proof']} for r in retained]
    log(f'rl: {len(targets)} targets, {len(retained):,} retained cap-6 records loaded at {clock.elapsed():.0f}s')

    found = collections.defaultdict(dict)
    state = {'step': 0, 'round': 0}
    rounds, total_samples, round_secs, step_secs = [], 0, None, cfg['default_step_secs']
    for r in range(1, cfg['rounds'] + 1):
        # stop if the previous round's cost says this one would not finish
        rcfg = cfg
        if round_secs is not None and round_secs > clock.left() and not cfg.get('fixed_rounds'):
            frac = clock.left() / round_secs
            if frac < cfg['tail_min_frac']:
                log(f'rl: stopping before round {r}: estimated {round_secs:.0f}s > {clock.left():.0f}s left')
                break
            rcfg = dict(cfg, k=max(2, int(cfg['k'] * frac * 0.9)), ft_steps=max(50, int(cfg['ft_steps'] * frac * 0.9)))
            log(f'rl: round {r} shrunk to {frac:.2f} of a full round: k {rcfg["k"]}, ft_steps {rcfg["ft_steps"]}')
        if clock.left() <= 0:
            break
        state['round'] = r
        tr = time.time()
        unsat = [i for i, t in enumerate(targets) if not (cfg['skip_saturated'] and len(found[t['name']]) >= cfg['max_per_thm'])]
        active = [i for i in unsat if i < n_main]
        samples, s_secs = sample_round(model, tok, targets, rcfg, seed * 1000 + r, clock, active)
        if r >= cfg['tb_start_round']:
            tb_active = [i for i in unsat if i >= n_main]
            tcfg = dict(rcfg, k=max(1, int(cfg['tb_k'] * rcfg['k'] / cfg['k'])), temperature=cfg['tb_temperature'])
            tb_samples, tb_secs = sample_round(model, tok, targets, tcfg, seed * 1000 + 500 + r, clock, tb_active)
            samples, s_secs = samples + tb_samples, s_secs + tb_secs
            log(f'rl r{r}: textbook exploration {len(tb_samples)} samples in {tb_secs:.0f}s')
        st = judge_samples(targets, samples, found, r)
        total_samples += st['samples']
        metrics.log(phase='sample', step=state['step'], round=r, n_prompts=len(active), k=rcfg['k'],
                    temperature=cfg['temperature'], samples_per_sec=st['samples'] / max(s_secs, 1e-9),
                    sample_secs=s_secs, judge_secs=time.time() - tr - s_secs, **st)
        log(f'rl r{r}: {st["samples"]} samples in {s_secs:.0f}s, accept {st["accept_rate"]:.4f}, +{st["new_proofs"]} proofs, '
            f'solved {st["targets_solved_round"]} this round / {st["targets_solved_cum"]} cum {st["solved_by_len"]}, '
            f'textbook solved cum {sum(1 for t in targets[n_main:] if found[t["name"]])}')

        data, n_rl = build_mix(targets, found, retained, cfg, rng, tok)
        steps, ft_secs = 0, 0.0
        if n_rl:
            steps, ft_secs = finetune(model, tok, data, rcfg, rng, device, clock, metrics, state)
            step_secs = ft_secs / max(steps, 1)
        else:
            log(f'rl r{r}: no verified proofs yet; skipping fine-tuning')
        paths['save_checkpoint'](model, tok, f'{paths["out_dir"]}/round_{r:02d}.pt')
        if cfg['eval_each_round'] and clock.left() > cfg['margin_secs']:
            te = time.time()
            hg = paths['heldout_greedy'](model, tok, n=cfg['eval_n'])
            metrics.log(phase='eval', step=state['step'], round=r, heldout_greedy=hg, secs=time.time() - te)
            log(f'rl r{r}: held-out greedy {hg:.3f}')
        secs = time.time() - tr
        # next round's cost: this round's, with the fine-tune charged at full length even if it was skipped/cut
        if rcfg is cfg:
            round_secs = secs - ft_secs + step_secs * cfg['ft_steps']
        rounds.append({'round': r, 'secs': secs, 'sample_secs': s_secs, 'ft_secs': ft_secs, 'ft_steps': steps,
                       'mix_rl_records': n_rl, 'n_records': len(data), **{k: st[k] for k in (
                           'samples', 'accept_rate', 'targets_solved_round', 'targets_solved_cum', 'new_proofs', 'solved_by_len')}})
        log(f'rl r{r}: done in {secs:.0f}s (sample {s_secs:.0f}s, fine-tune {steps} steps {ft_secs:.0f}s, '
            f'{n_rl} RL records), {clock.elapsed():.0f}s elapsed')

    with open(f'{paths["out_dir"]}/found_targets.jsonl', 'w') as f:
        for t in targets:
            for x in found[t['name']].values():
                f.write(json.dumps({'name': t['name'], 'thm': t['thm'], 'L_true': t['L_true'], **x}) + '\n')
    solved_cum = sum(1 for d in found.values() if d)
    totals = {'rounds': len(rounds), 'total_samples': total_samples, 'total_train_steps': state['step'],
              'targets_solved_cum': solved_cum, 'n_targets': len(targets), 'wall_secs': clock.elapsed()}
    metrics.log(phase='summary', step=state['step'], **totals)
    metrics.set_summary(**totals, per_round=rounds, config=cfg)
    model.eval()
    return model
