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

def gem_loss_on(model, x, m, beta=0.7, h='logsigmoid'):
    """GEM (Li et al., ICLR 2025, arXiv 2408.16673; code github.com/liziniu/GEM sft_trainer.py gem_loss): the
    entropy-regularised replacement for token cross-entropy that keeps output diversity during (reinforced)
    fine-tuning. Loss = -sum_v q(v) w(v) (log p(y) - log p(v)), q = softmax(logits / beta) detached,
    w = sigmoid(0.01 (z_v - z_y)) for h='logsigmoid'. Same masking as train.loss_on (proof tokens only)."""
    logits = model(x[:, :-1])
    tgt = x[:, 1:]
    mk = m[:, 1:]
    z = logits[mk].float()
    y = tgt[mk]
    with torch.no_grad():
        zy = z.gather(-1, y[:, None])
        w = torch.ones_like(z) if h == 'linear' else torch.sigmoid(0.01 * (z - zy))
        q = F.softmax(z / beta, -1)
    lp = F.log_softmax(z, -1)
    lpy = lp.gather(-1, y[:, None])
    return -(q * w * (lpy - lp)).sum(-1).mean()



# ---------------------------------------------------------------- verifier-guided (line-level rejection) sampling
from nd_verify import parse_formula as _pf, ParseError as _PE, RULE_NAMES as _RN
from nd_verify.verify import BOT as _BOT
_GARITY = {'ANDI': 2, 'ANDE1': 1, 'ANDE2': 1, 'IMPE': 2, 'IMPI': 2, 'ORI1': 1, 'ORI2': 1, 'ORE': 5, 'NEGE': 2,
           'NEGI': 2, 'BOTE': 1, 'DN': 1, 'R': 1}


def _prompt_sequent(prompt):
    toks = prompt.split()
    prem, i = [], 1
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
    """Incremental version of nd_verify's checks: add_line() accepts a proof line only if nd_verify would accept
    it given the lines before it (index order, premise block, box structure, citations, rule). done_ok() says
    whether QED here would give a valid proof. A full proof accepted line by line passes nd_verify."""

    def __init__(self, prem, concl):
        self.prem, self.concl = prem, concl
        self.lines, self.stack, self.nb = [], [], 0
        self.ctx, self.box_lines, self.box_depth, self.as_lines, self.fml = {}, {}, {}, {}, {}
        self.n_pr, self.in_pr = 0, True

    def add_line(self, toks):
        try:
            t = toks[0]
            if not (t.startswith('N') and t[1:].isdigit()):
                return False
            idx, i, d = int(t[1:]), 1, 0
            while i < len(toks) and toks[i] == '|':
                d += 1
                i += 1
            G, i = _pf(toks, i)
            if toks[i] != ':' or toks[i + 1] not in _RN:
                return False
            rule, i = toks[i + 1], i + 2
            refs = []
            while i < len(toks) and toks[i].startswith('N') and toks[i][1:].isdigit():
                refs.append(int(toks[i][1:]))
                i += 1
            if i != len(toks) - 1 or toks[i] != ';':
                return False
        except (_PE, IndexError, KeyError):
            return False
        L = self.lines
        idx0 = L[0]['idx'] if L else idx
        if idx < 1 or idx != idx0 + len(L):
            return False
        stack = list(self.stack)
        if rule == 'PR':
            if not self.in_pr or d != 0 or self.n_pr >= len(self.prem) or G != self.prem[self.n_pr] or refs:
                return False
        elif self.n_pr != len(self.prem):
            return False
        new_box = None
        if rule == 'AS':
            if d < 1 or d > len(stack) + 1 or refs:
                return False
            del stack[d - 1:]
            new_box = self.nb + 1
            stack.append(new_box)
        else:
            if d > len(stack):
                return False
            del stack[d:]
        ci = tuple(stack)
        if rule not in ('PR', 'AS'):
            if len(refs) != _GARITY[rule] or any(not (idx0 <= r < idx) for r in refs):
                return False
            ctx, F = self.ctx, self.fml

            def lc(j):
                return ctx[j] == ci[:len(ctx[j])]

            def bc(s, e):
                if s not in self.as_lines:
                    return False
                b = self.as_lines[s]
                if e != self.box_lines.get(b) or L[e - idx0]['depth'] != self.box_depth[b] or e < s or b in ci:
                    return False
                par = ctx[s][:-1]
                return par == ci[:len(par)]
            if rule in ('IMPI', 'NEGI'):
                if not bc(refs[0], refs[1]):
                    return False
            elif rule == 'ORE':
                if not lc(refs[0]) or not bc(refs[1], refs[2]) or not bc(refs[3], refs[4]):
                    return False
            elif not all(lc(r) for r in refs):
                return False
            f = lambda k: F[refs[k]]
            try:
                if rule == 'R':
                    ok = G == f(0)
                elif rule == 'ANDI':
                    ok = G == ('and', f(0), f(1))
                elif rule == 'ANDE1':
                    ok = f(0)[0] == 'and' and f(0)[1] == G
                elif rule == 'ANDE2':
                    ok = f(0)[0] == 'and' and f(0)[2] == G
                elif rule == 'IMPE':
                    ok = f(0) == ('imp', f(1), G)
                elif rule == 'IMPI':
                    ok = G == ('imp', f(0), f(1))
                elif rule == 'ORI1':
                    ok = G[0] == 'or' and G[1] == f(0)
                elif rule == 'ORI2':
                    ok = G[0] == 'or' and G[2] == f(0)
                elif rule == 'ORE':
                    ok = f(0) == ('or', f(1), f(3)) and f(2) == G and f(4) == G
                elif rule == 'NEGE':
                    ok = f(1) == ('not', f(0)) and G == _BOT
                elif rule == 'NEGI':
                    ok = f(1) == _BOT and G == ('not', f(0))
                elif rule == 'BOTE':
                    ok = f(0) == _BOT
                else:  # DN
                    ok = f(0) == ('not', ('not', G))
            except (TypeError, IndexError):
                ok = False
            if not ok:
                return False
        # commit
        if rule == 'PR':
            self.n_pr += 1
        else:
            self.in_pr = False
        if new_box is not None:
            self.nb = new_box
            self.box_depth[new_box] = d
            self.as_lines[idx] = new_box
        self.stack = stack
        for b in stack:
            self.box_lines[b] = idx
        self.ctx[idx] = ci
        self.fml[idx] = G
        L.append({'idx': idx, 'depth': d, 'rule': rule})
        return True

    def done_ok(self):
        if not self.lines or self.n_pr != len(self.prem):
            return False
        last = self.lines[-1]
        if last['depth'] != 0 or self.fml[last['idx']] != self.concl or last['rule'] == 'AS':
            return False
        return not (last['rule'] == 'PR' and self.concl not in self.prem)


@torch.no_grad()
def guided_generate(model, tok, prompts, temperature=1.0, max_new=512, batch=768, seed=0, max_retries=4,
                    max_rollbacks=24, slack=256, compact_every=16, compact_below=0.8):
    """Sampling with line-level rejection: whenever a row finishes a proof line (';') that nd_verify would
    reject given the lines before it, or emits QED before the proof is complete, the line is discarded (its
    KV-cache slots are masked out and the position is reset) and resampled from the saved distribution at the
    line start. A line rejected max_retries times, or max_rollbacks in total, ends the row. Returns
    (decoded proofs, stats)."""
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    ids = [tok.encode_prompt(p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(ids[i]))
    res = [None] * len(prompts)
    stats = collections.Counter()
    for s in range(0, len(order), batch):
        chunk = order[s:s + batch]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev.type == 'cuda')):
            outs = _guided_ids(model, tok, [ids[i] for i in chunk], [prompts[i] for i in chunk], temperature,
                               max_new, gen, max_retries, max_rollbacks, slack, compact_every, compact_below, stats)
        for i, o in zip(chunk, outs):
            res[i] = tok.decode(o + [tok.eos]) if o and o[-1] != tok.eos else tok.decode(o)
    return res, stats


def _guided_ids(model, tok, prompt_ids, prompts, temperature, max_new, gen, max_retries, max_rollbacks, slack,
                compact_every, compact_below, stats):
    dev = next(model.parameters()).device
    B = len(prompt_ids)
    L = max(len(p) for p in prompt_ids)
    semi, eos, itos = tok.stoi[';'], tok.eos, tok.itos
    idx = torch.full((B, L), tok.pad, dtype=torch.long, device=dev)
    keep = torch.zeros((B, L), dtype=torch.bool, device=dev)
    for i, p in enumerate(prompt_ids):
        idx[i, L - len(p):] = torch.tensor(p, device=dev)
        keep[i, L - len(p):] = True
    pos = (keep.cumsum(1) - 1).clamp(min=0)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    mask = causal[None, None] & keep[:, None, None, :]
    mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]
    T = max_new + slack
    caches = [{'max': L + T} for _ in model.blocks]
    logits = model(idx, pos=pos, mask=mask, caches=caches)[:, -1].float()
    saved = logits.clone()                         # distribution at the start of each row's current line
    rows = list(range(B))                          # original row of each live row
    buf = [[] for _ in range(B)]                   # accepted tokens per original row
    line = [[] for _ in range(B)]                  # tokens of the line being written
    chk = []
    for p in prompts:
        pr, cc = _prompt_sequent(p)
        chk.append(LineChecker(pr, cc))
    lstart_c = [L] * B                             # cache index where the current line starts
    lstart_p = pos[:, -1].tolist()                 # position of the last accepted token (next token gets +1)
    retries, rollbacks = [0] * B, [0] * B
    finished = [False] * B
    cur_pos = pos[:, -1].clone()
    use_saved = torch.zeros(B, dtype=torch.bool, device=dev)
    for t in range(T):
        lg = torch.where(use_saved[:, None], saved, logits)
        probs = F.softmax(lg / temperature, -1)
        nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        nl = nxt.tolist()
        acc_rows, rb_rows, kill_rows = [], [], []
        for j, r in enumerate(rows):
            if finished[r]:
                continue
            x = nl[j]
            if x == eos:
                if chk[r].done_ok():
                    buf[r].append(eos)
                    finished[r] = True
                    stats['qed_ok'] += 1
                else:
                    rb_rows.append(j)
                continue
            line[r].append(x)
            if x == semi:
                if chk[r].add_line([itos[y] for y in line[r]]):
                    buf[r] += line[r]
                    line[r] = []
                    retries[r] = 0
                    acc_rows.append(j)
                    stats['lines_ok'] += 1
                else:
                    rb_rows.append(j)
            elif len(buf[r]) + len(line[r]) >= max_new or len(line[r]) > 160:
                kill_rows.append(j)
        for j in rb_rows:
            r = rows[j]
            line[r] = []
            retries[r] += 1
            rollbacks[r] += 1
            stats['rollbacks'] += 1
            if retries[r] > max_retries or rollbacks[r] > max_rollbacks:
                kill_rows.append(j)
        for j in kill_rows:
            if not finished[rows[j]]:
                finished[rows[j]] = True
                stats['killed'] += 1
        # feed the sampled tokens (rolled-back rows feed theirs too; their slots are masked out below)
        n_c = caches[0]['n']                       # cache index this step's token lands in
        cur_pos = cur_pos + 1
        keep = torch.cat([keep, torch.ones(keep.size(0), 1, dtype=torch.bool, device=dev)], 1)
        rb_live = [j for j in rb_rows if not finished[rows[j]]]
        if rb_live:
            jj = torch.tensor(rb_live, device=dev)
            for j in rb_live:
                keep[j, lstart_c[rows[j]]:n_c + 1] = False
        fin = torch.tensor([finished[r] for r in rows], device=dev)
        if bool(fin.all()):
            break
        nxt = torch.where(fin, torch.full_like(nxt, tok.pad), nxt)
        logits = model(nxt[:, None], pos=cur_pos[:, None], mask=keep[:, None, None, :], caches=caches)[:, -1].float()
        use_saved = torch.zeros_like(use_saved)
        if rb_live:
            use_saved[jj] = True
            cur_pos[jj] = torch.tensor([lstart_p[rows[j]] for j in rb_live], device=dev)
        if acc_rows:
            ja = torch.tensor(acc_rows, device=dev)
            saved[ja] = logits[ja]
            cp = cur_pos[ja].tolist()
            for j, p in zip(acc_rows, cp):
                lstart_c[rows[j]] = n_c + 1
                lstart_p[rows[j]] = p
        if t % compact_every == compact_every - 1:
            live = [j for j, r in enumerate(rows) if not finished[r]]
            if len(live) < compact_below * len(rows):
                lv = torch.tensor(live, device=dev)
                rows = [rows[j] for j in live]
                keep, cur_pos, logits, saved, use_saved = keep[lv], cur_pos[lv], logits[lv], saved[lv], use_saved[lv]
                for c in caches:
                    c['k'], c['v'] = c['k'][lv], c['v'][lv]
    stats['rows'] += B
    return [buf[r] + (line[r] if not finished[r] else []) for r in range(B)]

# ---------------------------------------------------------------- knobs
CONFIG = dict(
    rounds=None,           # expert-iteration rounds: None = as many as the budget allows (~40 s each on a 3090)
    n_targets=None,        # hard cap before max_targets: None = every target in the file (legacy knob)
    max_targets=6000,      # per-run subset: at most this many targets, drawn with the run seed (None = no cap)
    len_weights={6: 0.5, 7: 1.0, 8: 1.5, 9: 2.0, 10: 2.0, 11: 3.0},  # stratify='length' quota weights per true length (from n002 via n031)
    stratify='length',     # how an over-cap pool is subsampled: 'length' = equal quota per true length present
                           # (short lengths do not drown the few hundred L11s; unused quota flows to the other
                           # lengths), None = uniform random sample. A pool within the cap is used whole.
    hard_refill=1200,      # after the first pass, each queue refill takes this many hardest targets (capped by the pool)
    window=400,            # targets sampled per round, popped from one rotating queue over the whole pool
    k=32,                  # mean samples per target per round (the round budget is window x k)
    lv_graph=True,         # guided sampling on the CUDA-graph decode step (lvg_generate, same line rules; one-time
                           # self-check against guided_generate, automatic fallback)
    guided=True,           # verifier-guided sampling of training rollouts: a proof line nd_verify would reject (or an early
    guide_retries=4,       # QED) is discarded and resampled from the line start; a line rejected guide_retries times, or
    guide_rollbacks=24,    # guide_rollbacks in total, ends the row (step-level search, as in BFS-Prover arXiv 2502.03438)
    guide_slack=256,       # extra decode steps allowed for resampled lines
    k_min=4, k_max=128,    # knapsack allocation bounds per target
    k_len_prior={6: 1, 7: 2, 8: 3, 9: 4, 10: 6, 11: 8},  # difficulty weight of a not-yet-sampled target by L_true
    temperature=1.0,       # training-time sampling temperature (dev scoring stays at 0.8)
    temp_end=1.3,          # base sampling temperature at the end of the run (linear ramp from temperature, temp_now)
    temp_quant=0.1,        # ramp in steps of 0.1 (1.0, 1.1, 1.2, 1.3): each new T rebuilds the cuda-graph sampler state
    max_new=512,           # max generated tokens per proof
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
    index_room=24,         # abs-index augmentation cap (n023): every fine-tuning proof is renumbered to start at N1 + s with
                           # s uniform in [0, 64 - max(proof lines, index_room)]; the frozen shift_abs spreads starts over
                           # N1..N59 and long proofs then run past N64 (MAXN)
    lr=3e-4, min_lr=3e-5, warmup=35, weight_decay=0.1, betas=(0.9, 0.95), grad_clip=1.0,
    loss='gem',            # 'gem' (GEM diversity-preserving loss, beta gem_beta) or 'ce' (train.loss_on)
    gem_beta=0.7,
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


def alloc_k(targets, cfg, rate):
    """Knapsack-style allocation (Li et al. 2025, arXiv 2509.25849): the round's sample budget
    len(targets) * k is split over targets by a difficulty weight, so easy targets get few samples and
    unsolved / long ones many. Unseen targets use a length prior; seen ones their latest solve rate."""
    rate = rate or {}
    w = []
    for t in targets:
        p = rate.get(t['name'])
        if p is None:
            w.append(cfg['k_len_prior'].get(t['L_true'], 4))
        elif p >= 0.5:
            w.append(1)
        elif p > 0:
            w.append(4)
        else:
            w.append(8)
    tot, sw = cfg['k'] * len(targets), sum(w)
    return [max(cfg['k_min'], min(cfg['k_max'], round(tot * x / sw))) for x in w]


# ---------------------------------------------------------------- CUDA-graph line-verified sampler
# Same algorithm as lv_generate (incremental line checker, a rejected line is masked out of the KV cache and
# resampled from the logits stored at its line start, retries per line / per proof, QED only if the proof is
# complete), but the model's decode step runs as a replayed CUDA graph over static buffers: one buffer set of the
# largest batch, with smaller batch buckets as views of its leading rows (compaction moves live rows to the front),
# and attention restricted to the used key span in 64-slot chunks. Sampling is Gumbel-max (= multinomial of softmax).
from model import rope_cache as _lvg_rope


class _LVG:
    BUCKETS = (768, 512, 384, 256, 192, 128, 96, 64, 32, 16, 8)
    SC = 64

    def __init__(self, model, S, temperature, max_new, dev, pad, dtemp=None, dhead=2, bar=None):
        Bm = self.BUCKETS[0]
        nl, H, hd, V = len(model.blocks), model.blocks[0].h, model.hd, model.head.out_features
        self.S, self.T, self.max_new = S, temperature, max_new
        self.dt, self.dhead, self.bar = (dtemp if dtemp is not None and dtemp != temperature else None), dhead, bar
        self.hn = torch.zeros(Bm, dtype=torch.long, device=dev)          # heated line-head tokens left
        self.at_start = torch.ones(Bm, dtype=torch.bool, device=dev)     # the next token is a line index
        self.k = [torch.zeros(Bm, H, S, hd, dtype=torch.bfloat16, device=dev) for _ in range(nl)]
        self.v = [torch.zeros(Bm, H, S, hd, dtype=torch.bfloat16, device=dev) for _ in range(nl)]
        self.keep = torch.zeros(Bm, S, dtype=torch.bool, device=dev)
        self.tok = torch.zeros(Bm, dtype=torch.long, device=dev)
        self.pos = torch.zeros(Bm, dtype=torch.long, device=dev)
        self.feedk = torch.ones(Bm, dtype=torch.bool, device=dev)
        self.ovr = torch.zeros(Bm, dtype=torch.bool, device=dev)
        self.sto = torch.zeros(Bm, dtype=torch.bool, device=dev)
        self.done = torch.ones(Bm, dtype=torch.bool, device=dev)
        self.ev = torch.zeros(Bm, dtype=torch.bool, device=dev)
        self.stored = torch.zeros(Bm, V, dtype=torch.float32, device=dev)
        self.out = torch.full((Bm, max_new + 1), pad, dtype=torch.long, device=dev)
        self.wp = torch.zeros(Bm, dtype=torch.long, device=dev)
        self.t = torch.zeros(1, dtype=torch.long, device=dev)
        self.cos, self.sin = _lvg_rope(S + 8, model.hd, dev)
        self.graphs = {}

    def bucket(self, n):
        for b in reversed(self.BUCKETS):
            if b >= n:
                return b
        return self.BUCKETS[0]

    def _tail(self, B, lg, pad, eos, semi):
        """Sample from lg, write at wp, advance wp, flag line events (views of the first B rows)."""
        u = torch.rand_like(lg).clamp_(1e-10, 1.0)
        if self.dt is not None:                     # line-head decision temperature (as in the parent's lv sampler)
            hn, at = self.hn[:B], self.at_start[:B]
            temp = torch.where(hn > 0, self.dt, self.T)[:, None]
            nxt = (lg / temp - torch.log(-torch.log(u))).argmax(-1)
            hn.copy_(torch.where(at, self.dhead, torch.where((hn > 0) & (nxt != self.bar), hn - 1, hn)))
            at.copy_(nxt == semi)
        else:
            nxt = (lg / self.T - torch.log(-torch.log(u))).argmax(-1)
        done, wp = self.done[:B], self.wp[:B]
        done |= wp >= self.max_new
        nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
        self.out[:B].scatter_(1, wp.clamp(max=self.max_new)[:, None], nxt[:, None])
        wp += (~done).long()
        self.ev[:B] = ((nxt == semi) | (nxt == eos)) & ~done
        self.tok[:B] = nxt

    def _step(self, model, B, Sc, pad, eos, semi):
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            keep = self.keep[:B]
            keep.index_copy_(1, self.t, self.feedk[:B, None])
            pos = self.pos[:B]
            x = model.emb(self.tok[:B])[:, None]
            cos, sin = self.cos[pos][:, None, None], self.sin[pos][:, None, None]
            mask = keep[:, None, None, :Sc]
            for li, b in enumerate(model.blocks):
                h = b.ln1(x)
                _, T, D = h.shape
                q, k, v = b.qkv(h).view(B, T, 3, b.h, D // b.h).permute(2, 0, 3, 1, 4)
                q, k = _apply_rope_lvg(q, cos, sin), _apply_rope_lvg(k, cos, sin)
                kc, vc = self.k[li][:B], self.v[li][:B]
                kc.index_copy_(2, self.t, k.to(torch.bfloat16))
                vc.index_copy_(2, self.t, v.to(torch.bfloat16))
                y = F.scaled_dot_product_attention(q.to(torch.bfloat16), kc[:, :, :Sc], vc[:, :, :Sc], attn_mask=mask)
                x = x + b.proj(y.transpose(1, 2).reshape(B, T, D))
                x = x + b.fc2(F.gelu(b.fc1(b.ln2(x))))
            lg = model.head(model.ln_f(x))[:, -1].float()
        pos += self.feedk[:B].long()
        ovr, sto, stored = self.ovr[:B, None], self.sto[:B, None], self.stored[:B]
        lg = torch.where(ovr, stored, lg)
        stored.copy_(torch.where(sto, lg, stored))
        self._tail(B, lg, pad, eos, semi)
        self.t += 1
        self.feedk[:B] = True
        self.ovr[:B] = False
        self.sto[:B] = False

    def replay(self, model, B, t_host, pad, eos, semi):
        Sc = min(self.S, -(-(t_host + 1) // self.SC) * self.SC)
        g = self.graphs.get((B, Sc))
        if g is None:
            bufs = [self.keep, self.tok, self.pos, self.feedk, self.ovr, self.sto, self.done, self.ev, self.stored,
                    self.out, self.wp, self.t, self.hn, self.at_start]
            snap = [x.clone() for x in bufs]
            ksnap = [(a[:, :, t_host:t_host + 2].clone(), c[:, :, t_host:t_host + 2].clone()) for a, c in zip(self.k, self.v)]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._step(model, B, Sc, pad, eos, semi)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._step(model, B, Sc, pad, eos, semi)
            self.graphs[(B, Sc)] = g
            for x, y in zip(bufs, snap):
                x.copy_(y)
            for (a, c), (a0, c0) in zip(zip(self.k, self.v), ksnap):
                a[:, :, t_host:t_host + 2].copy_(a0)
                c[:, :, t_host:t_host + 2].copy_(c0)
        g.replay()


def _apply_rope_lvg(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


_LVG_STATE = {}


@torch.no_grad()
def lvg_generate(model, tok, prompts, temperature=1.0, max_new=512, batch=768, seed=0, max_steps=768,
                 line_retries=4, row_rejects=16, compact_every=16, compact_below=None, stats=None,
                 decision_temp=None, decision_head=2, line_max=160):
    """Line-verified sampling with the parent's rules (see guided_fast)."""
    dev = next(model.parameters()).device
    torch.cuda.manual_seed(seed)
    batch = min(batch, _LVG.BUCKETS[0])
    ids = [tok.encode_prompt(p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(ids[i]))
    S = 1024 if max(len(x) for x in ids) + max_steps + 1 <= 1024 else -(-(max(len(x) for x in ids) + max_steps + 1) // 128) * 128
    key = (S, temperature, max_new, decision_temp, decision_head)
    if key not in _LVG_STATE:
        _LVG_STATE.clear()
        _LVG_STATE[key] = _LVG(model, S, temperature, max_new, dev, tok.pad, decision_temp, decision_head, tok.stoi['|'])
    G = _LVG_STATE[key]
    res = [None] * len(prompts)
    for s in range(0, len(order), batch):
        chunk = order[s:s + batch]
        outs = _lvg_ids(G, model, tok, [ids[i] for i in chunk], [_prompt_sequent(prompts[i]) for i in chunk],
                        max_new, max_steps, line_retries, row_rejects, compact_every, stats, line_max)
        for i, o in zip(chunk, outs):
            res[i] = tok.decode(o)
    return res


def _lvg_ids(G, model, tok, prompt_ids, seqs, max_new, max_steps, line_retries, row_rejects, compact_every, stats, line_max=160):
    dev = next(model.parameters()).device
    n = len(prompt_ids)
    L = max(len(p) for p in prompt_ids)
    B = G.bucket(n)
    semi, eos, pad, itos = tok.stoi[';'], tok.eos, tok.pad, tok.itos
    idx = torch.full((n, L), pad, dtype=torch.long, device=dev)
    keep = torch.zeros((n, L), dtype=torch.bool, device=dev)
    for i, p in enumerate(prompt_ids):
        idx[i, L - len(p):] = torch.tensor(p, device=dev)
        keep[i, L - len(p):] = True
    pos = (keep.cumsum(1) - 1).clamp(min=0)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    mask = causal[None, None] & keep[:, None, None, :]
    mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]
    caches = [{'max': G.S, 'k': G.k[i][:n], 'v': G.v[i][:n], 'n': 0} for i in range(len(model.blocks))]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        lg = model(idx, pos=pos, mask=mask, caches=caches)[:, -1].float()
    G.keep[:B].zero_(); G.keep[:n, :L] = keep; G.keep[n:B, 0] = True
    G.pos[:B].zero_(); G.pos[:n] = pos[:, -1] + 1
    G.done[:B].fill_(True); G.done[:n] = False
    G.wp[:B].zero_(); G.out[:B].fill_(pad)
    G.feedk[:B] = True; G.ovr[:B] = False; G.sto[:B] = False
    G.stored[:B].zero_(); G.stored[:n] = lg
    G.hn[:B].zero_(); G.at_start[:B] = True
    lgB = torch.zeros(B, lg.size(1), device=dev); lgB[:n] = lg
    G._tail(B, lgB, pad, eos, semi)               # first token, sampled from the prefill logits
    G.t.fill_(L)
    pos0 = G.pos[:n].tolist()
    rows = list(range(n))                           # current row -> original row
    st = [dict(chk=LineChecker(*seqs[o]), ls_wp=0, ls_pos=pos0[o], col0=L, lr=0, tr=0) for o in range(n)]
    res = [None] * n
    n_rej = n_kill = n_lines = 0
    live_n = n
    for step in range(max_steps):
        col = L + step                              # slot the just-sampled token will be fed into
        ev_idx = G.ev[:live_n].nonzero().squeeze(1).tolist()
        if ev_idx:
            evt = torch.tensor(ev_idx, device=dev)
            data, wps, nps = G.out[evt].tolist(), G.wp[evt].tolist(), G.pos[evt].tolist()
            kill, rej, acc, kill_eos = [], [], [], []
            rej_wp, rej_pos, rej_mask = [], [], []
            for k, i in enumerate(ev_idx):
                s = st[rows[i]]
                seg = data[k][s['ls_wp']:wps[k]]
                if len(seg) > line_max:             # over-long line: rejected (the parent's sampler kills the row)
                    ok = False
                elif seg[-1] == eos:
                    if len(seg) == 1 and s['chk'].done_ok():
                        kill.append(i)
                        continue
                    ok = False
                else:
                    ok = s['chk'].add_line([itos[x] for x in seg])
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
                    kill_eos.append((i, s['ls_wp']))
                    continue
                if col > s['col0']:
                    rej_mask.append((i, s['col0'], col))
                rej.append(i)
                rej_wp.append(s['ls_wp'])
                rej_pos.append(s['ls_pos'])
                s['col0'] = col + 1
            if kill:
                G.done[torch.tensor(kill, device=dev)] = True
            for i, w in kill_eos:
                G.out[i, w] = eos
            for i, a, b in rej_mask:
                G.keep[i, a:b] = False
            if rej:
                r_t = torch.tensor(rej, device=dev)
                G.wp[r_t] = torch.tensor(rej_wp, device=dev)
                G.pos[r_t] = torch.tensor(rej_pos, device=dev)
                G.feedk[r_t] = False
                G.ovr[r_t] = True
                G.at_start[r_t] = True                  # a rejected line restarts at its line index
                G.hn[r_t] = 0
            if acc:
                G.sto[torch.tensor(acc, device=dev)] = True
        if step % compact_every == compact_every - 1 or step == 0:
            d = G.done[:live_n]
            n_live = int((~d).sum())
            if n_live == 0:
                break
            nb = G.bucket(n_live)
            if nb < B:
                o_all = G.out[:live_n].tolist()
                w_all = G.wp[:live_n].tolist()
                dl = d.tolist()
                for i in range(live_n):
                    if dl[i]:
                        res[rows[i]] = o_all[i]
                live = (~d).nonzero().squeeze(1)
                for buf in [G.keep, G.tok, G.pos, G.feedk, G.ovr, G.sto, G.done, G.stored, G.out, G.wp, G.hn, G.at_start]:
                    buf[:n_live] = buf[live]
                for buf in G.k + G.v:                   # only the used key slots
                    buf[:n_live, :, :col + 1] = buf[live, :, :col + 1]
                G.done[n_live:nb] = True
                G.keep[n_live:nb] = False
                G.keep[n_live:nb, 0] = True
                G.feedk[n_live:nb] = True
                G.ovr[n_live:nb] = False
                G.sto[n_live:nb] = False
                rows = [rows[i] for i in live.tolist()]
                live_n, B = n_live, nb
        G.replay(model, B, col, pad, eos, semi)
    o_all = G.out[:live_n].tolist()
    for i in range(live_n):
        res[rows[i]] = o_all[i]
    if stats is not None:
        stats['rejects'] = stats.get('rejects', 0) + n_rej
        stats['kills'] = stats.get('kills', 0) + n_kill
        stats['lines'] = stats.get('lines', 0) + n_lines
    fin = []
    for row in res:
        fin.append(row[:row.index(eos) + 1] if eos in row else row[:max_new])
    return fin


_LVG_MODE = {}


def guided_fast(model, tok, prompts, temperature=1.0, max_new=512, batch=768, seed=0, max_retries=4,
                max_rollbacks=24, slack=256, log=print):
    """guided_generate's contract ((proofs, stats)) on the CUDA-graph line-verified sampler lvg_generate. The first
    call samples the same chunk with both samplers and keeps the graph sampler only if its accept rate is not far
    below guided_generate's; any exception falls back to guided_generate for the rest of the run."""
    def ref(p):
        return guided_generate(model, tok, p, temperature=temperature, max_new=max_new, batch=batch, seed=seed,
                               max_retries=max_retries, max_rollbacks=max_rollbacks, slack=slack)

    def fast(p):
        st = {}
        out = lvg_generate(model, tok, p, temperature=temperature, max_new=max_new, batch=batch, seed=seed,
                           max_steps=max_new + slack, line_retries=max_retries, row_rejects=max_rollbacks, stats=st)
        return out, collections.Counter({'rollbacks': st.get('rejects', 0), 'killed': st.get('kills', 0),
                                         'lines_checked': st.get('lines', 0), 'graph_rows': len(p)})
    if _LVG_MODE.get('off') or not torch.cuda.is_available():
        return ref(prompts)
    try:
        if 'checked' not in _LVG_MODE:
            t0 = time.time()
            r_out, r_st = ref(prompts)
            t1 = time.time()
            g_out, g_st = fast(prompts)
            t2 = time.time()
            a_ref = sum(verify_text(p + ' ' + o)[0] for p, o in zip(prompts, r_out)) / len(prompts)
            a_got = sum(verify_text(p + ' ' + o)[0] for p, o in zip(prompts, g_out)) / len(prompts)
            _LVG_MODE['checked'] = dict(ref_accept=round(a_ref, 4), graph_accept=round(a_got, 4),
                                        ref_secs=round(t1 - t0, 2), graph_secs=round(t2 - t1, 2),
                                        ref_rollbacks=r_st.get('rollbacks', 0), graph_rollbacks=g_st['rollbacks'])
            if a_got < 0.5 * a_ref - 0.01:
                _LVG_MODE['off'] = True
            log(f'lvg self-check: {_LVG_MODE["checked"]}, graph sampler {"OFF" if _LVG_MODE.get("off") else "on"}', flush=True)
            return (g_out, g_st) if not _LVG_MODE.get('off') else (r_out, r_st)
        return fast(prompts)
    except Exception as e:                       # noqa: BLE001
        _LVG_MODE['off'] = True
        log(f'lvg failed ({type(e).__name__}: {e}); falling back to guided_generate', flush=True)
        return ref(prompts)


GUIDE_STATS = collections.Counter()


def temp_now(cfg, clock):
    """POLARIS-style rising sampling temperature (hkunlp.github.io/blog/2025/Polaris): the training-sample base
    temperature ramps linearly from cfg['temperature'] to cfg['temp_end'] over the run's wall clock, so rollout
    diversity is kept up as the policy sharpens while line verification rejects the extra invalid lines."""
    span = max(clock.deadline - clock.t0, 1.0)
    frac = min(1.0, max(0.0, clock.elapsed() / span))
    T = cfg['temperature'] + (cfg.get('temp_end', cfg['temperature']) - cfg['temperature']) * frac
    q = cfg.get('temp_quant')              # the cuda-graph sampler bakes T into its graphs (rebuilt per distinct T)
    return round(round(T / q) * q, 3) if q else T


def sample_round(model, tok, targets, cfg, seed, clock, rate=None):
    """Sample cfg['k'] proofs per target in chunks of sample_batch prompts (so the clock is checked often).
    Returns (list of (target_index, proof) pairs, seconds). Stops early at the deadline."""
    model.eval()
    ks = alloc_k(targets, cfg, rate)
    prompts = [(i, t['prompt']) for i, t in enumerate(targets) for _ in range(ks[i])]
    prompts.sort(key=lambda x: len(x[1].split()))       # similar lengths per batch = less padding
    out, t0 = [], time.time()
    for c, s in enumerate(range(0, len(prompts), cfg['sample_batch'])):
        if clock.left() <= 0:
            break
        chunk = prompts[s:s + cfg['sample_batch']]
        if cfg['guided']:
            proofs, gst = (guided_fast if cfg['lv_graph'] else guided_generate)(model, tok, [p for _, p in chunk], temperature=temp_now(cfg, clock),
                                          max_new=cfg['max_new'], batch=cfg['sample_batch'], seed=seed * 100003 + c,
                                          max_retries=cfg['guide_retries'], max_rollbacks=cfg['guide_rollbacks'],
                                          slack=cfg['guide_slack'])
            for kk, vv in gst.items():
                GUIDE_STATS[kk] += vv
        else:
            proofs = fast_generate(model, tok, [p for _, p in chunk], temperature=temp_now(cfg, clock),
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


def make_batch_capped(data, idxs, tok, rng, dev, room):
    """lib.train.batch with the abs-index shift capped: renumber each proof to start at N1, then add a random
    offset that leaves room for `room` lines below N64."""
    from tokenizer import MAXN
    r0 = tok.ref0
    seqs, masks = [], []
    for i in idxs:
        p, q = data[i]
        refs = [x - r0 + 1 for x in q if x >= r0]
        if refs:
            lo, hi = min(refs), max(refs)
            span = hi - lo + 1
            s = rng.randint(0, max(0, MAXN - max(span, room)))
            q = [x - (lo - 1) + s if x >= r0 else x for x in q]
        seqs.append(p + q)
        masks.append([0] * len(p) + [1] * len(q))
    T = max(len(x) for x in seqs)
    x = torch.full((len(seqs), T), tok.pad, dtype=torch.long)
    m = torch.zeros((len(seqs), T), dtype=torch.bool)
    for i, (sq, mk) in enumerate(zip(seqs, masks)):
        x[i, :len(sq)] = torch.tensor(sq)
        m[i, :len(sq)] = torch.tensor(mk, dtype=torch.bool)
    return x.to(dev), m.to(dev)


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
        x, m = make_batch_capped(data, idxs, tok, rng, device, cfg['index_room'])
        lr = lr_at(s)
        for g in opt.param_groups:
            g['lr'] = lr
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            loss = gem_loss_on(model, x, m, cfg['gem_beta']) if cfg['loss'] == 'gem' else loss_on(model, x, m)
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
        samples, s_secs = sample_round(model, tok, win, cfg, seed * 1000 + r, clock, rate)
        st = judge_samples(win, samples, found, r, relab, cfg, avoid, rate)
        st['relabelled_total'] = len(relab)
        total_samples += st['samples']
        metrics.log(phase='sample', step=state['step'], round=r, n_prompts=len(win), k=cfg['k'],
                    temperature=cfg['temperature'], samples_per_sec=st['samples'] / max(s_secs, 1e-9),
                    sample_secs=s_secs, judge_secs=time.time() - tr - s_secs, **st)
        log(f'rl r{r}: {st["samples"]} samples in {s_secs:.0f}s, accept {st["accept_rate"]:.4f}, +{st["new_proofs"]} proofs, '
            f'solved {st["targets_solved_round"]} this round / {st["targets_solved_cum"]} cum {st["solved_by_len"]}, relabelled {len(relab)}')

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
    metrics.set_summary(**totals, per_round=rounds, config=cfg, guide_stats=dict(GUIDE_STATS))
    log(f'rl: guided sampling stats {dict(GUIDE_STATS)}')
    model.eval()
    return model
