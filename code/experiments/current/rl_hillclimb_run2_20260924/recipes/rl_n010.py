"""SEED RL file for run 2 (ab2): run 1's best recipe, GEAR node n025 (dev 401/1610 at pass@32 in 600 s).

Expert iteration (rejection-sampling fine-tuning against the verifier) with run 1's winning pieces: an
in-file compacting sampler (finished rows leave the batch and KV cache; ~1.35x lib.sample), small rounds
(400-target window, 135 fine-tune steps, fresh AdamW per round, 20k cap-6 retention records), training
temperature 1.0, and hardest-first queue refills after the first pass over the pool. Run 2 changes only
what the new pool needs: no fixed pool size or length set (targets_L6-11), a per-run subset of at most
6,000 targets stratified over lengths (CONFIG max_targets / stratify), rounds bounded by the budget
instead of a cap of 20, and an O(n) refill.

KNOWN_FAILURES (run 1: 600 s budget, dev L7-9 pass@32, single seeds, run-to-run noise about +-15;
node ids refer to /scratch/cowifornia/nd-rl/ab/gear/nodes and its reflections.md):
  - a GRPO pass (group-mean advantage, lr 5e-5, <= 30 steps) before each round's SFT step: -56 vs its
    parent (n016); accept rates did not rise and rounds got ~10 s slower
  - hindsight relabelling of verified by-products: only 11-18 by-products per run, a no-op (n004, n006)
    (the code is still present below: relab / relabel_min_lines / relabel_weight)
  - pass@k rarity weighting (each target's proofs repeated 1-8x by solve rate): neutral, -9 (n011)
    (still present below: rare_weight_max)
  - lighter cap-6 retention (retain 5,000 instead of 20,000, RL weight 8): -20 (n002) and about -60 vs
    the same schedule with full retention (n005); keep retain 20000
  - a persistent AdamW state across rounds (no per-round re-warmup): length-7 solves collapsed 109 -> 37,
    -105 (n020); the fresh optimizer per round is a useful reset
  - sampling batch 1024: slower rounds (67-88 s vs 55-60 s), -9 (n010)
  - appending the extra 15k-target pool (mostly length 7): late rounds became all L7/L8, dev L9 184 -> 141,
    -80 (n022)
  - learning-rate changes: peak lr 4.5e-4 instead of 3e-4 was -60 (n024)
  - a length curriculum (L7 first, then 70/30, then 40/30/30): lifts the longest length a little, costs L7,
    net zero (n003, n008, n009, n014, n015)
  - larger rounds (window 600-1200, 200 fine-tune steps): fewer rounds fit and lose; going from window 600
    to 400 with 135 steps gave +50 (n015 -> n017). Going smaller still (window 300, 100 steps) was -44 (n019)

This is the one file the agents edit. The frozen harness calls

    model = run(model, tok, targets, budget_secs, device, seed, log, metrics, paths) -> model

  model        lib.model.GPT(vocab, 4, 256, 8) loaded from frozen/pretrain.pt, on `device`
  tok          lib.tokenizer.Tokenizer('abs')
  targets      list of dicts from targets_L6-11.jsonl (name, thm, key, prompt, n_lines, L_true, ...; no proofs)
  budget_secs  wall clock for the whole call (1200 by default; 7200 for a champion run). SIGALRM at
               budget+45; returning after budget+15 is a timeout. This file stops with >= CONFIG['margin_secs'] to spare.
  device       'cuda'
  seed         int (the run seed)
  log(msg)     timestamped print
  metrics      frozen metrics.Metrics: metrics.log(phase=..., step=..., **fields); metrics.set_summary(**kv)
  paths        dict: targets, train_cap6, heldout_cap6, out_dir (files you may read / write under),
               save_checkpoint(model, tok, path), heldout_greedy(model, tok, n=1000) -> float, smoke (bool)
Return the model to be scored (same architecture).

Rules (SPEC.md "Rules for agents"): the only training data allowed are (a) proofs this model sampled
during this run that nd_verify accepts for the prompted sequent (targets_L6-11 only), (b) verified
by-products of those samples (hindsight relabelling to the conclusion actually proved from the same
premises) and (c) records from train_cap6.jsonl. heldout_cap6.jsonl is for tracking only. No proofs
from any other source, and never edit anything under frozen/.

Each round: sample k proofs per target -> keep verified, distinct (start-index normalised) proofs,
accumulated across rounds -> fine-tune on <= max_per_thm proofs per solved target (each repeated by
the rarity weight) plus relabelled by-products, mixed with `retain` random cap-6 records -> checkpoint.
"""
import json, math, random, sys, time, collections

import torch
import torch.nn.functional as F
from sample import generate          # frozen/lib is on sys.path (the harness puts it there)
from train import batch as make_batch, loss_on
from nd_verify import verify_text
from normalize import norm
from prune import pruned_length
from keys import prompt_key


def subset_targets(targets, cfg, rng):
    """The run's target list: every target (shuffled) if the pool is within max_targets, else a subset of
    max_targets drawn per cfg['stratify'] and shuffled. The same seed gives the same subset."""
    if cfg.get('n_targets'):
        targets = rng.sample(targets, min(cfg['n_targets'], len(targets)))
    cap = cfg.get('max_targets')
    if not cap or len(targets) <= cap:
        return rng.sample(targets, len(targets))       # identical to run 1's shuffle for a small pool
    if cfg.get('stratify') != 'length':
        return rng.sample(targets, cap)
    by_len = collections.defaultdict(list)
    for t in targets:
        by_len[t['L_true']].append(t)
    wts = {L: float(cfg.get('len_weights', {}).get(L, 1.0)) for L in by_len}
    quota, left = {}, cap
    lens = sorted(by_len, key=lambda L: (len(by_len[L]) / max(wts[L], 1e-9), L))
    for i, L in enumerate(lens):                      # weighted water-filling: supply-short lengths take all, rest share by weight
        wleft = sum(wts[M] for M in lens[i:])
        quota[L] = min(len(by_len[L]), int(left * wts[L] / wleft))
        left -= quota[L]
    out = [t for L in sorted(by_len) for t in rng.sample(by_len[L], quota[L])]
    rng.shuffle(out)
    return out


def relabel(prompt, proof):
    """Hindsight: if `proof` is a valid proof of some other conclusion from the same premises, return (new_prompt, n_lines)."""
    toks = proof.split()
    if 'QED' not in toks:
        return None
    body = toks[:toks.index('QED')]
    lines, cur = [], []
    for t in body:
        cur.append(t)
        if t == ';':
            lines.append(cur)
            cur = []
    if not lines or cur:
        return None
    last = lines[-1]
    if len(last) < 2 or last[1] == '|' or ':' not in last:
        return None
    form = ' '.join(last[1:last.index(':')])
    newp = f"{prompt.split(' SEQ ')[0]} SEQ {form} PRF"
    if newp == prompt:
        return None
    ok, reason, nl = verify_text(newp + ' ' + proof)
    return (newp, nl) if ok else None

# ---------------------------------------------------------------- knobs
CONFIG = dict(
    rounds=None,           # expert-iteration rounds: None = as many as the budget allows (~40 s each on a 3090)
    n_targets=None,        # hard cap before max_targets: None = every target in the file (legacy knob)
    max_targets=6000,      # per-run subset: at most this many targets, drawn with the run seed (None = no cap)
    stratify='length',     # how an over-cap pool is subsampled: 'length' = equal quota per true length present
                           # (short lengths do not drown the few hundred L11s; unused quota flows to the other
                           # lengths), None = uniform random sample. A pool within the cap is used whole.
    len_weights={6: 0.5, 7: 1.0, 8: 1.5, 9: 2.0, 10: 2.0, 11: 3.0},  # stratify='length' quota weights per true length
    hard_refill=1200,      # after the first pass, each queue refill takes this many hardest targets (capped by the pool)
    window=400,            # targets sampled per round, popped from one rotating queue over the whole pool
    k=32,                  # samples per target per round
    temperature=1.0,       # training-time sampling temperature (dev scoring stays at 0.8)
    max_new=512,           # max generated tokens per proof
    line_verify=True,      # w1: training rollouts check each line incrementally and resample bad lines
    lv_max_steps=768,      # decode steps per batch with line resampling (rejected lines cost steps)
    lv_line_retries=4,     # resamples of one line before the row is abandoned
    lv_row_rejects=16,     # total rejected lines per proof before the row is abandoned
    sample_batch=768,      # prompts per fast_generate() call (compaction makes 768 ~1.35x faster than lib 384)
    max_per_thm=4,         # at most this many distinct proofs per target in each round's mix
    min_round_frac=0.3,    # last round: shrink window and ft steps to the time left, unless below this fraction
    relabel_min_lines=7,   # hindsight: keep verified by-products (other conclusion, same premises) with >= this many lines
    relabel_weight=2,      # each relabelled proof is repeated this many times in the mix
    rl_weight=4,           # (unused: rarity weighting below sets each target's repeat count)
    rare_weight_max=8,     # pass@k-aware: a target's proofs repeat max(1, round(rare_weight_max * (1 - solve rate))) times
    retain=20000,          # cap-6 train records mixed in each round (retention of in-distribution skill)
    ft_steps=135,          # fine-tune steps per round (fresh AdamW each round)
    ft_batch=128,
    lr=3e-4, min_lr=3e-5, warmup=35, weight_decay=0.1, betas=(0.9, 0.95), grad_clip=1.0,
    log_every=50,          # train-phase metrics every this many steps
    eval_each_round=False, # greedy held-out check after every round (~15 s; the harness always does a final one)
    eval_n=1000,
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
def fast_generate(model, tok, prompts, temperature=0.8, max_new=512, batch=384, seed=0, compact_every=16, compact_below=0.8):
    """Sampling as lib.sample.generate, but finished rows are dropped from the batch (and the KV cache)
    every `compact_every` steps once fewer than `compact_below` of the rows are still running."""
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    ids = [tok.encode_prompt(p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(ids[i]))
    res = [None] * len(prompts)
    for s in range(0, len(order), batch):
        chunk = order[s:s + batch]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev.type == 'cuda')):
            outs = _gen_ids(model, tok, [ids[i] for i in chunk], temperature, max_new, gen, compact_every, compact_below)
        for i, o in zip(chunk, outs):
            res[i] = tok.decode(o)
    return res


def _gen_ids(model, tok, prompt_ids, temperature, max_new, gen, compact_every, compact_below):
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
    out = torch.full((B, max_new), tok.pad, dtype=torch.long, device=dev)
    rows = torch.arange(B, device=dev)            # original row of each live row
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    cur_pos = pos[:, -1]
    for t in range(max_new):
        probs = F.softmax(logits.float() / temperature, -1)
        nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, tok.pad), nxt)
        out[rows, t] = nxt
        done = done | (nxt == tok.eos)
        if t % compact_every == compact_every - 1 or t == 0:
            n_live = int((~done).sum())
            if n_live == 0:
                break
            if n_live < compact_below * done.numel():
                live = (~done).nonzero().squeeze(1)
                rows, done, nxt, cur_pos, keep = rows[live], done[live], nxt[live], cur_pos[live], keep[live]
                for c in caches:
                    c['k'], c['v'] = c['k'][live], c['v'][live]
        cur_pos = cur_pos + 1
        keep = torch.cat([keep, torch.ones(keep.size(0), 1, dtype=torch.bool, device=dev)], 1)
        logits = model(nxt[:, None], pos=cur_pos[:, None], mask=keep[:, None, None, :], caches=caches)[:, -1]
    return out.tolist()


# ---------------------------------------------------------------- line-verified sampling (w1)
# Step-level verifier search for training rollouts (BFS-Prover-style step checking, arXiv 2502.03438):
# every generated proof line is checked by an incremental Fitch checker with nd_verify's rules; a bad line
# is masked out of the KV cache and resampled from the stored line-start distribution. Final proofs are
# still gated by the full nd_verify in judge_samples, so only verifier-accepted own samples are trained on.
from nd_verify.verify import parse_formula as _pf, RULE_NAMES as _RULES, ParseError as _PE, BOT as _BOT

_ARITY = {'ANDI': 2, 'ANDE1': 1, 'ANDE2': 1, 'IMPE': 2, 'IMPI': 2, 'ORI1': 1, 'ORI2': 1, 'ORE': 5,
          'NEGE': 2, 'NEGI': 2, 'BOTE': 1, 'DN': 1, 'R': 1, 'PR': 0, 'AS': 0}


def parse_sequent(prompt):
    toks = prompt.split()
    i, prem = 1, []
    if toks[i] != 'SEQ':
        while True:
            f, i = _pf(toks, i)
            prem.append(f)
            if toks[i] == ',':
                i += 1
                continue
            break
    concl, _ = _pf(toks, i + 1)
    return prem, concl


class LineChecker:
    """Incremental version of nd_verify.verify: check(line tokens) accepts and commits a line only if
    it can still be part of a valid proof (index, premise block, box structure, citations, rule)."""

    def __init__(self, prem, concl):
        self.prem, self.concl = prem, concl
        self.lines, self.idx0, self.stack, self.nb = [], None, [], 0
        self.ctx, self.box_lines, self.box_depth, self.as_lines = {}, {}, {}, {}
        self.fml, self.depth, self.in_pr, self.n_pr = {}, {}, True, 0

    def check(self, toks):
        try:
            t = toks[0]
            if not (t.startswith('N') and t[1:].isdigit()):
                return False
            idx, i, d = int(t[1:]), 1, 0
            while i < len(toks) and toks[i] == '|':
                d += 1
                i += 1
            G, i = _pf(toks, i)
            if toks[i] != ':' or toks[i + 1] not in _RULES:
                return False
            rule, i = toks[i + 1], i + 2
            refs = []
            while i < len(toks) and toks[i].startswith('N') and toks[i][1:].isdigit():
                refs.append(int(toks[i][1:]))
                i += 1
            if i != len(toks) - 1 or toks[i] != ';':
                return False
        except (_PE, IndexError):
            return False
        if self.idx0 is None:
            if idx < 1:
                return False
        elif idx != self.idx0 + len(self.lines):
            return False
        in_pr = self.in_pr
        if rule == 'PR':
            if not in_pr or d != 0 or self.n_pr >= len(self.prem) or G != self.prem[self.n_pr]:
                return False
        elif in_pr:
            if self.n_pr != len(self.prem):
                return False
            in_pr = False
        stack, nb, new_box = list(self.stack), self.nb, None
        if rule == 'AS':
            if d < 1 or d > len(stack) + 1:
                return False
            del stack[d - 1:]
            nb += 1
            new_box = nb
            stack.append(nb)
        else:
            if d > len(stack):
                return False
            del stack[d:]
        ci = tuple(stack)
        if len(refs) != _ARITY[rule]:
            return False
        lo = self.idx0 if self.idx0 is not None else idx
        if any(not (lo <= r < idx) for r in refs):
            return False

        def line_ok(j):
            cj = self.ctx[j]
            return cj == ci[:len(cj)]

        def box_ok(s, e):
            if s not in self.as_lines:
                return False
            b = self.as_lines[s]
            if e != self.box_lines.get(b) or self.depth[e] != self.box_depth[b] or e < s or b in ci:
                return False
            parent = self.ctx[s][:-1]
            return parent == ci[:len(parent)]

        if rule in ('IMPI', 'NEGI'):
            if not box_ok(*refs):
                return False
        elif rule == 'ORE':
            j, s1, e1, s2, e2 = refs
            if not line_ok(j) or not box_ok(s1, e1) or not box_ok(s2, e2):
                return False
        elif rule not in ('PR', 'AS'):
            if not all(line_ok(r) for r in refs):
                return False
        F = self.fml.get
        ok = True
        if rule == 'R':
            ok = G == F(refs[0])
        elif rule == 'ANDI':
            ok = G == ('and', F(refs[0]), F(refs[1]))
        elif rule == 'ANDE1':
            ok = F(refs[0])[0] == 'and' and F(refs[0])[1] == G
        elif rule == 'ANDE2':
            ok = F(refs[0])[0] == 'and' and F(refs[0])[2] == G
        elif rule == 'IMPE':
            ok = F(refs[0]) == ('imp', F(refs[1]), G)
        elif rule == 'IMPI':
            ok = G == ('imp', F(refs[0]), F(refs[1]))
        elif rule == 'ORI1':
            ok = G[0] == 'or' and G[1] == F(refs[0])
        elif rule == 'ORI2':
            ok = G[0] == 'or' and G[2] == F(refs[0])
        elif rule == 'ORE':
            j, s1, e1, s2, e2 = refs
            ok = F(j) == ('or', F(s1), F(s2)) and F(e1) == G and F(e2) == G
        elif rule == 'NEGE':
            ok = F(refs[1]) == ('not', F(refs[0])) and G == _BOT
        elif rule == 'NEGI':
            ok = F(refs[1]) == _BOT and G == ('not', F(refs[0]))
        elif rule == 'BOTE':
            ok = F(refs[0]) == _BOT
        elif rule == 'DN':
            ok = F(refs[0]) == ('not', ('not', G))
        if not ok:
            return False
        # commit
        if self.idx0 is None:
            self.idx0 = idx
        self.in_pr = in_pr
        if rule == 'PR':
            self.n_pr += 1
        self.stack, self.nb = stack, nb
        if new_box is not None:
            self.as_lines[idx] = new_box
            self.box_depth[new_box] = d
        for b in stack:
            self.box_lines[b] = idx
        self.ctx[idx], self.fml[idx], self.depth[idx] = ci, G, d
        self.lines.append((idx, d, rule))
        return True

    def check_qed(self):
        if not self.lines or self.n_pr != len(self.prem):
            return False
        idx, d, rule = self.lines[-1]
        if d != 0 or self.fml[idx] != self.concl or rule == 'AS':
            return False
        return not (rule == 'PR' and self.concl not in self.prem)


@torch.no_grad()
def lv_generate(model, tok, prompts, temperature=1.0, max_new=512, batch=768, seed=0, max_steps=768,
                line_retries=4, row_rejects=16, compact_every=16, compact_below=0.8, stats=None):
    """Like fast_generate, but each proof line is checked when its ';' (or QED) is sampled; a failing
    line is masked out of the cache and resampled from the distribution stored at the line start."""
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    ids = [tok.encode_prompt(p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(ids[i]))
    res = [None] * len(prompts)
    for s in range(0, len(order), batch):
        chunk = order[s:s + batch]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev.type == 'cuda')):
            outs = _lv_gen_ids(model, tok, [ids[i] for i in chunk], [parse_sequent(prompts[i]) for i in chunk],
                               temperature, max_new, gen, max_steps, line_retries, row_rejects,
                               compact_every, compact_below, stats)
        for i, o in zip(chunk, outs):
            res[i] = tok.decode(o)
    return res


def _lv_gen_ids(model, tok, prompt_ids, seqs, temperature, max_new, gen, max_steps, line_retries, row_rejects,
                compact_every, compact_below, stats):
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
    caches = [{'max': L + max_steps} for _ in model.blocks]
    logits = model(idx, pos=pos, mask=mask, caches=caches)[:, -1]
    stored = logits.clone()                        # line-start logits, by original row
    out = torch.full((B, max_new + 1), tok.pad, dtype=torch.long, device=dev)
    wp = torch.zeros(B, dtype=torch.long, device=dev)   # write pointer, by original row
    rows = torch.arange(B, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    npos = pos[:, -1] + 1                          # position of the next fed token (live rows)
    npos0 = npos.tolist()
    st = [dict(chk=LineChecker(*seqs[o]), ls_wp=0, ls_pos=npos0[o], col0=L, lr=0, tr=0) for o in range(B)]
    semi, eos, itos = tok.stoi[';'], tok.eos, tok.itos
    n_rej = n_kill = n_lines = 0
    for t in range(max_steps):
        col = L + t
        probs = F.softmax(logits.float() / temperature, -1)
        nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        over = wp[rows] >= max_new
        done = done | over
        nxt = torch.where(done, torch.full_like(nxt, tok.pad), nxt)
        out[rows, wp[rows].clamp(max=max_new)] = nxt
        wp[rows] += (~done).long()
        ev = ((nxt == semi) | (nxt == eos)) & ~done
        ev_idx = ev.nonzero().squeeze(1).tolist()
        feed_keep = torch.ones(rows.numel(), dtype=torch.bool, device=dev)
        ovr = torch.zeros(rows.numel(), dtype=torch.bool, device=dev)
        sto = torch.zeros(rows.numel(), dtype=torch.bool, device=dev)
        if ev_idx:
            evt = torch.tensor(ev_idx, device=dev)
            orig = rows[evt]
            data, wps, nps, o_l = out[orig].tolist(), wp[orig].tolist(), npos[evt].tolist(), orig.tolist()
            kill, rej, acc = [], [], []
            for k, i in enumerate(ev_idx):
                o, s = o_l[k], st[o_l[k]]
                seg = data[k][s['ls_wp']:wps[k]]
                if seg[-1] == eos:
                    if len(seg) == 1 and s['chk'].check_qed():
                        kill.append(i)             # finished: the row is done (QED written)
                        continue
                    ok = False
                else:
                    ok = s['chk'].check([itos[x] for x in seg])
                    n_lines += 1
                if ok:
                    s['ls_wp'], s['ls_pos'], s['col0'], s['lr'] = wps[k], nps[k] + 1, col + 1, 0
                    acc.append(i)
                    continue
                s['lr'] += 1
                s['tr'] += 1
                n_rej += 1
                if s['lr'] > line_retries or s['tr'] > row_rejects:
                    n_kill += 1
                    kill.append(i)
                    out[o, s['ls_wp']] = eos        # truncated: decodes to an incomplete proof (fails verify)
                    continue
                if col > s['col0']:
                    keep[i, s['col0']:col] = False
                wp[o] = s['ls_wp']
                npos[i] = s['ls_pos']
                s['col0'] = col + 1
                rej.append(i)
            if kill:
                done[torch.tensor(kill, device=dev)] = True
            if rej:
                r_t = torch.tensor(rej, device=dev)
                feed_keep[r_t] = False
                ovr[r_t] = True
            if acc:
                sto[torch.tensor(acc, device=dev)] = True
        if t % compact_every == compact_every - 1 or t == 0:
            n_live = int((~done).sum())
            if n_live == 0:
                break
            if n_live < compact_below * done.numel():
                live = (~done).nonzero().squeeze(1)
                rows, done, nxt, npos, keep = rows[live], done[live], nxt[live], npos[live], keep[live]
                feed_keep, ovr, sto = feed_keep[live], ovr[live], sto[live]
                for c in caches:
                    c['k'], c['v'] = c['k'][live], c['v'][live]
        keep = torch.cat([keep, feed_keep[:, None]], 1)
        logits = model(nxt[:, None], pos=npos[:, None], mask=keep[:, None, None, :], caches=caches)[:, -1]
        npos = npos + feed_keep.long()
        if bool(ovr.any()):
            logits[ovr] = stored[rows[ovr]]
        if bool(sto.any()):
            stored[rows[sto]] = logits[sto]
    if stats is not None:
        stats['rejects'] = stats.get('rejects', 0) + n_rej
        stats['kills'] = stats.get('kills', 0) + n_kill
        stats['lines'] = stats.get('lines', 0) + n_lines
    res = []
    for row in out.tolist():
        res.append(row[:row.index(eos) + 1] if eos in row else row[:max_new])
    return res


def sample_round(model, tok, targets, cfg, seed, clock):
    """Sample cfg['k'] proofs per target in chunks of sample_batch prompts (so the clock is checked often).
    Returns (list of (target_index, proof) pairs, seconds). Stops early at the deadline."""
    model.eval()
    prompts = [(i, t['prompt']) for i, t in enumerate(targets) for _ in range(cfg['k'])]
    prompts.sort(key=lambda x: len(x[1].split()))       # similar lengths per batch = less padding
    out, t0 = [], time.time()
    for c, s in enumerate(range(0, len(prompts), cfg['sample_batch'])):
        if clock.left() <= 0:
            break
        chunk = prompts[s:s + cfg['sample_batch']]
        if cfg.get('line_verify'):
            proofs = lv_generate(model, tok, [p for _, p in chunk], temperature=cfg['temperature'],
                                 max_new=cfg['max_new'], batch=cfg['sample_batch'], seed=seed * 100003 + c,
                                 max_steps=cfg['lv_max_steps'], line_retries=cfg['lv_line_retries'],
                                 row_rejects=cfg['lv_row_rejects'], stats=cfg.setdefault('_lv_stats', {}))
        else:
            proofs = fast_generate(model, tok, [p for _, p in chunk], temperature=cfg['temperature'],
                                   max_new=cfg['max_new'], batch=cfg['sample_batch'], seed=seed * 100003 + c)
        out += [(i, q) for (i, _), q in zip(chunk, proofs)]
    return out, time.time() - t0


def judge_samples(targets, samples, found, rnd, relab=None, cfg=None, avoid=None, rate=None):
    """Verify each sample against its prompted sequent; add new distinct proofs to `found`
    ({target name: {normalised proof: record}}). Returns the sample-phase statistics."""
    n_ok = n_qed = new = 0
    solved_now, wl, pl, reasons = set(), [], [], collections.Counter()
    ok_by, n_by = collections.Counter(), collections.Counter()
    for i, proof in samples:
        n_by[targets[i]['name']] += 1
        t = targets[i]
        n_qed += 'QED' in proof.split()
        ok, reason, nl = verify_text(t['prompt'] + ' ' + proof)
        if not ok:
            reasons[reason.split(' (line')[0]] += 1
            if relab is not None:
                rr = relabel(t['prompt'], proof)
                if rr and rr[1] >= cfg['relabel_min_lines']:
                    k2 = prompt_key(rr[0])
                    if k2 not in avoid and k2 not in relab:
                        relab[k2] = (rr[0], proof)
            continue
        n_ok += 1
        ok_by[t['name']] += 1
        solved_now.add(t['name'])
        pr = pruned_length(t['prompt'], proof)
        wl.append(nl)
        pl.append(pr)
        key = norm(proof)
        if key not in found[t['name']]:
            found[t['name']][key] = {'proof': proof, 'written': nl, 'pruned': pr, 'round': rnd}
            new += 1
    if rate is not None:
        for name, c in n_by.items():
            rate[name] = ok_by[name] / c
    n = max(len(samples), 1)
    lens = [str(L) for L in sorted({t['L_true'] for t in targets})]
    solved_cum = {name for name, d in found.items() if d}
    by_len = collections.Counter(str(t['L_true']) for t in targets if t['name'] in solved_cum)
    by_len_round = collections.Counter(str(t['L_true']) for t in targets if t['name'] in solved_now)
    return {'samples': len(samples), 'accept_rate': n_ok / n, 'qed_rate': n_qed / n,
            'targets_solved_round': len(solved_now), 'targets_solved_cum': len(solved_cum), 'new_proofs': new,
            'solved_by_len': {L: by_len.get(L, 0) for L in lens},
            'solved_by_len_round': {L: by_len_round.get(L, 0) for L in lens},
            'mean_written_len': sum(wl) / len(wl) if wl else None, 'mean_pruned_len': sum(pl) / len(pl) if pl else None,
            'fail_reasons': [[r, c] for r, c in reasons.most_common(5)]}


def build_mix(targets, found, retained, cfg, rng, tok, relab=None, rate=None):
    """Encoded fine-tuning records: <= max_per_thm proofs per solved target x rl_weight, plus `retain`
    random cap-6 records. Returns (data, n_rl_records)."""
    mix = []
    for t in targets:
        fs = list(found[t['name']].values())
        rng.shuffle(fs)
        for x in fs[:cfg['max_per_thm']]:
            mix += [(t['prompt'], x['proof'])] * max(1, round(cfg['rare_weight_max'] * (1 - (rate or {}).get(t['name'], 0.5))))
    for p, q in (relab or {}).values():
        mix += [(p, q)] * cfg['relabel_weight']
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
    n_pool = len(targets)
    targets = subset_targets(targets, cfg, rng)
    retained = read_jsonl(paths['train_cap6'], limit=cfg['retain'] * 4 if paths.get('smoke') else None)
    assert all(r['n_lines'] <= 6 for r in retained), 'train_cap6 must only hold proofs of <= 6 lines'
    retained = [{'prompt': r['prompt'], 'proof': r['proof']} for r in retained]
    by_len0 = collections.Counter(t['L_true'] for t in targets)
    log(f'rl: {len(targets)} of {n_pool} targets (max_targets {cfg["max_targets"]}, stratify {cfg["stratify"]}) '
        f'{dict(sorted(by_len0.items()))}, {len(retained):,} retained cap-6 records loaded at {clock.elapsed():.0f}s')

    found = collections.defaultdict(dict)
    queues = {}
    rate = {}                                    # target name -> verified fraction of its samples, latest round
    relab = {}                                   # class key -> (relabelled prompt, proof)
    avoid = {t['key'] for t in targets}          # never relabel onto a target class (those are sampled for directly)
    state = {'step': 0, 'round': 0}
    rounds, total_samples, round_secs, step_secs = [], 0, None, cfg['default_step_secs']
    r = 0
    while cfg['rounds'] is None or r < cfg['rounds']:
        r += 1
        # if the previous round's cost says this one would not finish, shrink window and fine-tune to fit
        frac = 1.0 if round_secs is None else min(1.0, (clock.left() - 3) / round_secs)
        if frac < cfg['min_round_frac']:
            log(f'rl: stopping before round {r}: estimated {round_secs:.0f}s, {clock.left():.0f}s left')
            break
        rcfg = dict(cfg, ft_steps=max(1, int(cfg['ft_steps'] * frac)))
        if frac < 1.0:
            log(f'rl: round {r} shrunk to {frac:.2f} of a full round to fit {clock.left():.0f}s left')
        if clock.left() <= 0:
            break
        state['round'] = r
        tr = time.time()
        # first pass: rotating window over the shuffled pool; once exhausted, refill the queue hardest-first:
        # unsolved targets, then solved ones by ascending latest solve rate
        n_win = max(1, int(min(cfg['window'], len(targets)) * frac))
        q = queues.setdefault('q', list(targets))
        if len(q) < n_win:
            queued = {t['name'] for t in q}
            rest = [t for t in targets if t['name'] not in queued]
            rng.shuffle(rest)
            rest.sort(key=lambda t: (bool(found[t['name']]), rate.get(t['name'], 0.0)))
            q += rest[:cfg['hard_refill']]
            queues['refills'] = queues.get('refills', 0) + 1
            log(f'rl: refill {queues["refills"]}: {sum(1 for t in rest[:cfg["hard_refill"]] if not found[t["name"]])} unsolved '
                f'of {min(len(rest), cfg["hard_refill"])} queued')
        win, queues['q'] = q[:n_win], q[n_win:]
        samples, s_secs = sample_round(model, tok, win, cfg, seed * 1000 + r, clock)
        st = judge_samples(win, samples, found, r, relab, cfg, avoid, rate)
        st['relabelled_total'] = len(relab)
        total_samples += st['samples']
        metrics.log(phase='sample', step=state['step'], round=r, n_prompts=len(win), k=cfg['k'],
                    temperature=cfg['temperature'], samples_per_sec=st['samples'] / max(s_secs, 1e-9),
                    sample_secs=s_secs, judge_secs=time.time() - tr - s_secs, **st)
        log(f'rl r{r}: {st["samples"]} samples in {s_secs:.0f}s, accept {st["accept_rate"]:.4f}, +{st["new_proofs"]} proofs, '
            f'solved {st["targets_solved_round"]} this round / {st["targets_solved_cum"]} cum {st["solved_by_len"]}, relabelled {len(relab)}, lv {cfg.get("_lv_stats")}')

        data, n_rl = build_mix(targets, found, retained, cfg, rng, tok, relab, rate)
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
              'targets_solved_cum': solved_cum, 'n_targets': len(targets), 'n_pool': n_pool,
              'targets_by_len': {str(k): v for k, v in sorted(by_len0.items())}, 'wall_secs': clock.elapsed()}
    metrics.log(phase='summary', step=state['step'], **totals)
    metrics.set_summary(**totals, per_round=rounds, config=cfg)
    model.eval()
    return model
