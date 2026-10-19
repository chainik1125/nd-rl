"""SEED RL file for run 3 (ab3, compute-metered budget): run 2's plain seed (run 1's winner, GEAR node n025)
plus three parts taken from run 2's n092 line: a CUDA-graph decode step for plain sampling, the capped
line-index shift in fine-tuning batches (index_room), and stopping driven by the compute meter; and one new
efficiency part, sequence packing of the fine-tuning batches (ft_pack): with padded random batches 58-61% of the
charged fine-tune rows were padding, and fine-tuning is most of the budget.

Expert iteration (rejection-sampling fine-tuning against the verifier): small rounds (400-target window,
k=32 samples per target at temperature 1.0, 135 fine-tune steps, fresh AdamW per round, 20k cap-6 retention
records), a per-run subset of at most 6,000 targets stratified over lengths, hardest-first queue refills after
the first pass. Run 2's other levers (line-verified sampling, GEM loss, length weights, schema coverage,
knapsack k, rising temperature, last-N weight averaging) are NOT in this seed. Their run-2 implementations are
in lib_run2.py next to this file, each behind a CONFIG switch below that is off here (see that file's
docstring for what each did in run 2).

BUDGET (run 3). The score depends on the compute, not the card: paths['budget'] is the frozen meter
(frozen/meter.py). Every nn.Linear call is charged 2*in*out FLOPs per row (padding rows included), x3 when
autograd records it; CUDA-graph replays are charged what their capture traced. Attention products, layer
norms, the verifier and CPU work are free. Past the limit every Linear call raises meter.BudgetExhausted and
the harness scores the model as it stands. This file plans with the meter: per-phase FLOP rates are measured
in-run (FLOPs per sampled prompt, per fine-tune row, per round), a round that would not fit is shrunk
(min_round_frac), sampling stops before a chunk that would eat the round's fine-tune share, fine-tuning stops
before a step that would not fit, and it returns with at most margin_frac of the budget unused plus
less than min_round_frac of a round. budget_secs is only the harness's wall-clock safety cap (3,600 s); it
is kept as a second limit. With an unlimited meter (calibration) the wall clock alone stops the run.
Sampling speed (the graph sampler, compaction) no longer raises the score; it only shortens experiments.
Padding does: the meter charges padded rows. Packing (on here) keeps each step's records, their mix and the loss
exactly as in a padded random batch and removes the padding, so a fine-tune step costs about 2.4x less.

KNOWN_FAILURES (node ids, numbers and details: /scratch/cowifornia/nd-rl/ab3/RUN12_FINDINGS.md; do not open run 1's
or run 2's directories). Run 1 (600 s wall clock, dev L7-9 pass@32 of 1,610, single seeds, noise about +-15):
  - a GRPO pass (group-mean advantage, lr 5e-5, <= 30 steps) before each SFT step: -56 (n016)
  - hindsight relabelling of verified by-products: 11-18 by-products per run, a no-op (n004, n006); the code is
    still below (relab / relabel_min_lines / relabel_weight). Run 2 (prefix relabelling): 1-8 per round, no-op
  - pass@k rarity weighting (proofs repeated 1-8x by solve rate): neutral, -9 (n011); still below (rare_weight_max)
  - lighter cap-6 retention (retain 5,000, RL weight 8): -20 (n002), about -60 (n005); keep retain 20000
  - a persistent AdamW state across rounds: length-7 solves 109 -> 37, -105 (n020); fresh optimizer per round
  - appending an extra 15k mostly-L7 pool: late rounds all L7/L8, -80 (n022)
  - peak lr 4.5e-4 instead of 3e-4: -60 (n024)
  - a length curriculum (short first): net zero (n003, n008, n009, n014, n015)
  - larger rounds (window 600-1200, 200 steps) lost; window 600 -> 400 with 135 steps +50 (n015 -> n017);
    smaller still (window 300, 100 steps) -44 (n019)
Run 2 (1,200 s wall clock; score = dev pass@32 of 1,200 at L8-11 + 4 x frontier pass@64 of 1,000 at L12-14;
single seed unless stated. Many run-2 verdicts were confounded by a 150 W power cap that set each card's round count (16-30 rounds for
one recipe): a loss that the worker marked as clock- or card-bound is not settled under a FLOP budget):
  - penalising the model's own rejected lines: whole-line NSR (n012) dev 185 -> 1; step-level DPO (n072) -537
  - entropy minimisation on citation / rule tokens, weight 0.3 (n044): -69, held-out 0.896
  - deeper per-row search in the line-verified sampler: retries 8 / rollbacks 40 (n061) -490 with the same
    solves per round; backtracking (n027, n035) slower, no more solves
  - retry heating (n048, n055: -323), line-head decision temperature 1.4 on the graph sampler (n087 -146,
    n091 -320, accept rate fell to 0.05), annealed decoding 1.2 -> 0.7 (n007: dev 107 vs 179)
  - rising training temperature (POLARIS) 1.0 -> 1.3: +167 to -241 on single seeds; two-seed n075 "adds
    nothing measurable"
  - proof selection: longest-first (n059) -303; shortest-first -1 to -32 (n003, n014, n018)
  - max_targets 9,000 lost on every second seed (n082, n088, n096); 12,000 -16 (n095): subset size should
    follow the expected round count (30 rounds good, 20 bad), not be fixed
  - refills: fresh targets instead of retrying failures (n019) -42; near-miss ordering (n011) -29;
    schema-sibling spreading (n004) -49
  - easy-to-hard first pass: +56 (n042), -187 (n051), -126 (n056; it starves L10-11 and the frontier)
  - fine-tune batch 256 (n045): -227 (half the rounds); shape-weighted retention (n009) -26; retain 12,000
    (n073) -53, within the tie band
  - knapsack k by length (n001 -6, n015 -70) and GEM loss WITHOUT line verification (n017 +9, n025 -27):
    neutral; on the plain sampler the length weights were -8 (n002)
  - what did help in run 2 (all on the line-verified sampler; in lib_run2.py): line-verified sampling (n010
    two-seed 206.5 vs 185; n024 314 / 59 vs 188 / 13), GEM loss with it (n031 +160), the capped index shift
    (in this seed; n030's always-N1 control collapsed L11 to 0, capping added 125-244), length-weighted quotas
    (n038 two-seed +178), schema coverage (n041 +136 single seed), last-3 weight averaging (+75 to +183, but
    -433 on one line). The CUDA-graph sampler (n046, 591 -> 950.5 two-seed) was a throughput win only.
  - noise: second seeds averaged 154 score points below first seeds; same-code same-seed pairs differed by 34-234
    points. Only two-seed means count.

This is the one file the agents edit. The frozen harness calls

    model = run(model, tok, targets, budget_secs, device, seed, log, metrics, paths) -> model

  model        lib.model.GPT(vocab, 4, 256, 8) loaded from frozen/pretrain.pt, on `device`
  tok          lib.tokenizer.Tokenizer('abs')
  targets      list of dicts from the harness's TARGETS_FILE (run 3: targets_L6-11.jsonl; name, thm, key, prompt,
               n_lines, L_true, source, schema, ...; no proofs)
  budget_secs  wall-clock safety cap (3,600 s by default). SIGALRM at budget+45; returning after budget+15 is a
               timeout. This file keeps >= CONFIG['margin_secs'] to spare.
  device       'cuda'
  seed         int (the run seed)
  log(msg)     timestamped print
  metrics      frozen metrics.Metrics: metrics.log(phase=..., step=..., **fields); metrics.set_summary(**kv)
  paths        dict: targets, train_cap6, heldout_cap6, out_dir (files you may read / write under),
               save_checkpoint(model, tok, path), heldout_greedy(model, tok, n=1000) -> float (charged: it runs the
               model), smoke (bool), budget (the meter: used(), remaining(), frac(), limit, snapshot(); an
               infinite limit means unmetered)
Return the model to be scored (same architecture).

Rules (SPEC.md "Rules for agents" and "Compute budget"): the only training data allowed are (a) proofs this
model sampled during this run that nd_verify accepts for the prompted sequent (prompts from the RL pool
only), (b) verified by-products of those samples (hindsight relabelling to the conclusion actually proved from
the same premises) and (c) records from train_cap6.jsonl. heldout_cap6.jsonl is for tracking only. No proofs
from any other source, never edit anything under frozen/, and all model compute goes through nn.Linear modules
under the meter (no torch.compile, no F.linear / matmul on weights, no extra processes or threads, one GPU).

Each round: sample k proofs per target -> keep verified, distinct (start-index normalised) proofs,
accumulated across rounds -> fine-tune on <= max_per_thm proofs per solved target (each repeated by
the rarity weight) plus relabelled by-products, mixed with `retain` random cap-6 records -> checkpoint.
"""
import collections, hashlib, importlib.util, json, math, os, random, sys, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sample import generate          # frozen/lib is on sys.path (the harness puts it there)
from nd_verify import verify_text
from normalize import norm
from prune import pruned_length
from keys import prompt_key
from tokenizer import MAXN
from model import rope_cache
try:
    from meter import BudgetExhausted    # frozen/meter.py (run 3 harness)
except ImportError:                      # older harness without a meter: wall clock only
    class BudgetExhausted(BaseException):
        pass

LIB_RUN2 = '/scratch/cowifornia/nd-rl/ab3/seed/lib_run2.py'   # fallback location of the optional run-2 library


def subset_targets(targets, cfg, rng):
    """The run's target list: every target (shuffled) if the pool is within max_targets, else a subset of
    max_targets drawn per cfg['stratify'] and shuffled. The same seed gives the same subset. Works for any
    set of lengths (e.g. only 6-7 after max_len_train=7)."""
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
    quota, left, lens = {}, cap, sorted(by_len, key=lambda L: (len(by_len[L]), L))
    for i, L in enumerate(lens):                      # water-filling: small lengths take all, rest share evenly
        quota[L] = min(len(by_len[L]), left // (len(lens) - i))
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
    rounds=None,           # expert-iteration rounds: None = as many as the budget allows
    n_targets=None,        # hard cap before max_targets: None = every target in the file (legacy knob)
    max_len_train=7,    # drop targets with L_true above this before subsetting (None = all lengths in the pool)
    max_proof_lines_train=7,  # found / relabelled proofs with more written lines than this are left out of the
                           # fine-tune mix (dropped whole, never edited; they still count as solved). None = no cap
    max_targets=6000,      # per-run subset: at most this many targets, drawn with the run seed (None = no cap)
    stratify='length',     # how an over-cap pool is subsampled: 'length' = equal quota per true length present
                           # (unused quota of short-supplied lengths flows to the others), None = uniform random
                           # sample. A pool within the cap is used whole.
    hard_refill=1200,      # after the first pass, each queue refill takes this many hardest targets (capped by the pool)
    window=400,            # targets sampled per round, popped from one rotating queue over the whole pool
    k=32,                  # samples per target per round
    temperature=1.0,       # training-time sampling temperature (dev scoring stays at 0.8)
    max_new=512,           # max generated tokens per proof (raised to max_new_per_line x the longest L_true if larger)
    max_new_per_line=40,   # run 2's proofs averaged ~19 tokens per written line, at most ~37
    sampler='graph',       # 'graph': CUDA-graph decode step (graph_generate; falls back to 'compact' on any error);
                           # 'compact': fast_generate. Same output distribution; the graph step is ~2x faster and,
                           # with its finer batch buckets, charges slightly fewer padding rows.
    sample_batch=768,      # prompts per sampler call
    max_per_thm=4,         # at most this many distinct proofs per target in each round's mix
    min_round_frac=0.3,    # last round: shrink window and ft steps to the compute (and time) left, unless below this
    relabel_min_lines=7,   # hindsight: keep verified by-products (other conclusion, same premises) with >= this many lines
    relabel_weight=2,      # each relabelled proof is repeated this many times in the mix
    rare_weight_max=8,     # pass@k-aware: a target's proofs repeat max(1, round(rare_weight_max * (1 - solve rate))) times
    retain=20000,          # cap-6 train records mixed in each round (retention of in-distribution skill)
    ft_steps=135,          # fine-tune steps per round (fresh AdamW each round)
    ft_batch=128,          # records per fine-tune step (a random batch)
    ft_pack=True,          # sequence packing (pack_rows): the step's ft_batch records, whole, first-fit-decreasing into
    pack_row=1024,         # rows of at most pack_row tokens (and >= the longest record; the row length with the least
                           # padding is picked per step), block-diagonal causal attention, positions reset per record:
                           # the padded batch's logits and loss, at 1-3% padding instead of ~58%
    index_room=24,         # capped abs-index shift (run 2 n023/n065/n078): every fine-tuning proof is renumbered to start
                           # at N1 + s, s uniform in [0, MAXN - max(proof index span, index_room)]; the frozen shift_abs
                           # spreads starts over N1..N59 and long proofs then ran past N64. None = frozen shift_abs
    lr=3e-4, min_lr=3e-5, warmup=35, weight_decay=0.1, betas=(0.9, 0.95), grad_clip=1.0,
    log_every=50,          # train-phase metrics every this many steps
    eval_each_round=False, # greedy held-out check after every round (charged compute; the harness always does a free final one)
    eval_n=1000,
    margin_frac=0.003,     # FLOPs never planned: this fraction of the compute limit stays unused as a safety margin
    est_safety=1.25,       # a sampling chunk is started only if est_safety x its estimated cost fits
    margin_secs=20,        # wall-clock safety cap: return at least this long before budget_secs
    default_step_secs=0.12,  # fine-tune secs/step guess until measured (wall-clock planning)
    # ---- run 2's other levers: implementations in lib_run2.py (next to this file, else LIB_RUN2); all off here
    loss='ce',             # 'ce' (train.loss_on) | 'gem' (lib_run2.gem_loss_on, GEM entropy-regularised loss)
    guided=False,          # line-verified sampling (lib_run2.guided_sample: 'lv_graph' CUDA-graph version, or 'lv')
    len_weights=None,      # {L_true: weight} length-weighted stratified quotas (lib_run2.subset_targets)
    schema_cover=False,    # textbook schemas water-filled inside each length quota (lib_run2.subset_targets)
    knapsack=False,        # per-target k from a difficulty weight (lib_run2.alloc_k)
    temp_schedule=None,    # training temperature over the compute fraction used (lib_run2.temperature_at)
    avg_last=1,            # return the uniform weight average of the last N round endpoints (lib_run2.WeightAverage)
)
SMOKE = dict(n_targets=8, k=2, max_new=128, ft_steps=5, retain=64, ft_batch=32, rounds=2, eval_n=16, margin_secs=5,
             log_every=1, eval_each_round=True, ft_always=True)   # ft_always: fine-tune on retention alone if nothing was solved


# ---------------------------------------------------------------- helpers

_R2 = {}


def lib_run2():
    """The optional run-2 library (lazy import, only when a run-2 switch is on). Its SHA-256 goes into the summary:
    the harness copies and hashes rl.py only, so an edited lib_run2.py is not otherwise recorded."""
    if 'mod' not in _R2:
        here = os.path.dirname(os.path.abspath(__file__))
        path = next(p for p in (os.path.join(here, 'lib_run2.py'), LIB_RUN2) if os.path.exists(p))
        spec = importlib.util.spec_from_file_location('lib_run2', path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['lib_run2'] = mod
        spec.loader.exec_module(mod)
        with open(path, 'rb') as f:
            _R2['sha'] = hashlib.sha256(f.read()).hexdigest()
        _R2['mod'], _R2['path'] = mod, path
    return _R2['mod']


def read_jsonl(fn, limit=None):
    out = []
    with open(fn) as f:
        for l in f:
            if l.strip():
                out.append(json.loads(l))
                if limit and len(out) >= limit:
                    break
    return out


def linear_flops_per_row(model):
    """FLOPs the meter charges for one row (token position) through every nn.Linear of `model` without autograd."""
    return float(sum(2 * m.in_features * m.out_features for m in model.modules() if isinstance(m, nn.Linear)))


class Budget:
    """The run's two limits: the metered compute budget (paths['budget'], charged FLOPs) and the harness's
    wall-clock safety cap (budget_secs - margin_secs). flops_left() keeps margin_frac of the limit in reserve and
    is inf for an unlimited meter (or no meter)."""

    def __init__(self, meter, budget_secs, cfg):
        self.meter = meter
        self.limit = float(getattr(meter, 'limit', math.inf)) if meter is not None else math.inf
        self.finite = math.isfinite(self.limit)
        self.reserve = cfg['margin_frac'] * self.limit if self.finite else 0.0
        self.t0 = time.time()
        self.deadline = self.t0 + budget_secs - cfg['margin_secs']

    def used(self):
        return float(self.meter.used()) if self.meter is not None else 0.0

    def flops_left(self):
        return self.limit - self.used() - self.reserve if self.finite else math.inf

    def frac_used(self):
        return self.used() / self.limit if self.finite else 0.0

    def secs_left(self):
        return self.deadline - time.time()

    def elapsed(self):
        return time.time() - self.t0

    def fits(self, flops):
        return flops <= self.flops_left() and self.secs_left() > 0


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


# ---------------------------------------------------------------- CUDA-graph plain sampler
# fast_generate's algorithm with the one-token decode step replayed as a CUDA graph (the decode step of run 2's
# n046/n053/n092 line-verified sampler, with line verification off). Static buffers sized for the largest batch;
# smaller batch buckets are views of their leading rows (compaction moves live rows to the front). Attention reads
# only the used key slots, rounded up to SC. Sampling is Gumbel-max, i.e. a draw from softmax(logits / T), the
# same distribution as fast_generate's multinomial (different random stream). The temperature is a buffer, so a
# changed temperature needs no recapture. The meter charges each replay for the rows of its bucket.

def _rope1(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class _GraphSampler:
    # batch buckets, <= 12.5% apart down to 64 rows (n092's 768/512/384/256/192/128/96/64 ladder padded ~2% more
    # rows than fast_generate's compaction; this one pads fewer): 36 buckets x (S / SC) key spans = graphs captured
    BUCKETS = tuple(range(768, 511, -64)) + tuple(range(480, 255, -32)) + tuple(range(240, 127, -16)) + tuple(range(120, 7, -8))
    SC = 64

    def __init__(self, model, S, Bm, max_new, dev, greedy=False):
        nl, H, hd = len(model.blocks), model.blocks[0].h, model.hd
        self.key = (S, Bm, max_new, greedy, id(model))
        self.S, self.Bm, self.max_new, self.greedy = S, Bm, max_new, greedy
        self.buckets = (Bm,) + tuple(b for b in self.BUCKETS if b < Bm)
        self.k = [torch.zeros(Bm, H, S, hd, dtype=torch.bfloat16, device=dev) for _ in range(nl)]
        self.v = [torch.zeros(Bm, H, S, hd, dtype=torch.bfloat16, device=dev) for _ in range(nl)]
        self.keep = torch.zeros(Bm, S, dtype=torch.bool, device=dev)
        self.tok = torch.zeros(Bm, dtype=torch.long, device=dev)
        self.pos = torch.zeros(Bm, dtype=torch.long, device=dev)
        self.done = torch.ones(Bm, dtype=torch.bool, device=dev)
        self.out = torch.zeros(Bm, max_new, dtype=torch.long, device=dev)
        self.slot = torch.zeros(1, dtype=torch.long, device=dev)     # KV slot of the token being fed
        self.col = torch.zeros(1, dtype=torch.long, device=dev)      # output column of the token being sampled
        self.inv_t = torch.ones(1, dtype=torch.float32, device=dev)  # 1 / temperature
        self.cos, self.sin = rope_cache(S + 8, hd, dev)
        self.graphs, self.pool = {}, None

    def bucket(self, n):
        for b in reversed(self.buckets):
            if b >= n:
                return b
        return self.buckets[0]

    def sample(self, B, lg, pad, eos):
        """Draw the next token of the first B rows from logits lg, write it to column col, flag EOS."""
        if self.greedy:
            nxt = lg.argmax(-1)
        else:
            u = torch.rand_like(lg).clamp_(1e-10, 1.0)
            nxt = (lg * self.inv_t - torch.log(-torch.log(u))).argmax(-1)
        done = self.done[:B]
        nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
        self.out[:B].index_copy_(1, self.col, nxt[:, None])
        done |= nxt == eos
        self.tok[:B] = nxt
        self.col += 1

    def _step(self, model, B, Sc, pad, eos):
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            keep = self.keep[:B]
            keep.index_fill_(1, self.slot, True)
            pos = self.pos[:B]
            x = model.emb(self.tok[:B])[:, None]
            cos, sin = self.cos[pos][:, None, None], self.sin[pos][:, None, None]
            mask = keep[:, None, None, :Sc]
            for li, b in enumerate(model.blocks):
                h = b.ln1(x)
                D = h.shape[-1]
                q, k, v = b.qkv(h).view(B, 1, 3, b.h, D // b.h).permute(2, 0, 3, 1, 4)
                q, k = _rope1(q, cos, sin), _rope1(k, cos, sin)
                kc, vc = self.k[li][:B], self.v[li][:B]
                kc.index_copy_(2, self.slot, k.to(torch.bfloat16))
                vc.index_copy_(2, self.slot, v.to(torch.bfloat16))
                y = F.scaled_dot_product_attention(q.to(torch.bfloat16), kc[:, :, :Sc], vc[:, :, :Sc], attn_mask=mask)
                x = x + b.proj(y.transpose(1, 2).reshape(B, 1, D))
                x = x + b.fc2(F.gelu(b.fc1(b.ln2(x))))
            lg = model.head(model.ln_f(x))[:, -1].float()
        pos += 1
        self.slot += 1
        self.sample(B, lg, pad, eos)

    def step(self, model, B, slot_host, pad, eos):
        """One decode step for the first B rows; slot_host = the KV slot being fed (host copy of self.slot)."""
        Sc = min(self.S, -(-(slot_host + 1) // self.SC) * self.SC)
        g = self.graphs.get((B, Sc))
        if g is None:            # one eager warm-up step (charged), capture (not charged), then undo the warm-up
            bufs = [self.keep, self.tok, self.pos, self.done, self.out, self.slot, self.col]
            snap = [x.clone() for x in bufs]
            ksnap = [(a[:B, :, slot_host].clone(), c[:B, :, slot_host].clone()) for a, c in zip(self.k, self.v)]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._step(model, B, Sc, pad, eos)
            torch.cuda.current_stream().wait_stream(s)
            for x, y in zip(bufs, snap):
                x.copy_(y)
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):   # graphs replay one at a time on one stream: one shared pool
                self._step(model, B, Sc, pad, eos)
            self.graphs[(B, Sc)] = g
            for x, y in zip(bufs, snap):
                x.copy_(y)
            for (a, c), (a0, c0) in zip(zip(self.k, self.v), ksnap):
                a[:B, :, slot_host].copy_(a0)
                c[:B, :, slot_host].copy_(c0)
        g.replay()


_GS = {}


@torch.no_grad()
def graph_generate(model, tok, prompts, temperature=1.0, max_new=512, batch=768, seed=0, compact_every=16,
                   greedy=False, S_min=0):
    """fast_generate's contract (decoded proof bodies, one per prompt) on the CUDA-graph decode step.
    S_min: KV length to allocate at least (pass the longest prompt of the pool + max_new + 1 so that the
    buffers and graphs are built once per run)."""
    dev = next(model.parameters()).device
    torch.cuda.manual_seed(seed)
    ids = [tok.encode_prompt(p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(ids[i]))
    need = max(max(len(x) for x in ids) + max_new + 1, S_min)
    S = -(-need // _GraphSampler.SC) * _GraphSampler.SC
    Bm = min(batch, _GraphSampler.BUCKETS[0])
    G = _GS.get('G')
    if G is None or G.key != (max(S, G.S), Bm, max_new, greedy, id(model)) or G.S < S:
        _GS.clear()
        torch.cuda.empty_cache()
        G = _GS['G'] = _GraphSampler(model, S, Bm, max_new, dev, greedy)
    G.inv_t.fill_(1.0 / temperature if not greedy else 1.0)
    res = [None] * len(prompts)
    for s in range(0, len(order), Bm):
        chunk = order[s:s + Bm]
        outs = _graph_ids(G, model, tok, [ids[i] for i in chunk], max_new, compact_every)
        for i, o in zip(chunk, outs):
            res[i] = tok.decode(o)
    return res


def _graph_ids(G, model, tok, prompt_ids, max_new, compact_every):
    dev = next(model.parameters()).device
    n = len(prompt_ids)
    L = max(len(p) for p in prompt_ids)
    B = G.bucket(n)
    pad, eos = tok.pad, tok.eos
    idx = torch.full((n, L), pad, dtype=torch.long, device=dev)
    keep = torch.zeros((n, L), dtype=torch.bool, device=dev)
    for i, p in enumerate(prompt_ids):
        idx[i, L - len(p):] = torch.tensor(p, device=dev)
        keep[i, L - len(p):] = True
    pos = (keep.cumsum(1) - 1).clamp(min=0)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    mask = causal[None, None] & keep[:, None, None, :]
    mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]
    # prefill (eager, charged) writes the prompt's keys / values straight into the static cache rows 0..n-1
    caches = [{'max': G.S, 'k': G.k[i][:n], 'v': G.v[i][:n], 'n': 0} for i in range(len(model.blocks))]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        lg = model(idx, pos=pos, mask=mask, caches=caches)[:, -1].float()
    G.keep[:B].zero_()
    G.keep[:n, :L] = keep
    G.keep[n:B, 0] = True                        # bucket padding rows attend to one slot (no NaN); they are done
    G.pos[:B].zero_()
    G.pos[:n] = pos[:, -1] + 1
    G.done[:B].fill_(True)
    G.done[:n] = False
    G.out[:B].fill_(pad)
    G.col.zero_()
    lgB = torch.zeros(B, lg.size(1), device=dev)
    lgB[:n] = lg
    G.sample(B, lgB, pad, eos)                   # token 0, from the prefill logits
    G.slot.fill_(L)
    rows = list(range(n))                        # current row -> original row
    res = [None] * n
    live_n = n
    for step in range(1, max_new):               # feed token step-1 (KV slot L+step-1), sample token `step`
        slot = L + step - 1
        if step == 1 or step % compact_every == 0:
            d = G.done[:live_n]
            n_live = int((~d).sum())
            if n_live == 0:
                break
            nb = G.bucket(n_live)
            if nb < B:
                dl = d.nonzero().squeeze(1)
                for i, o in zip(dl.tolist(), G.out[dl].tolist()):
                    res[rows[i]] = o
                live = (~d).nonzero().squeeze(1)
                for buf in (G.keep, G.tok, G.pos, G.done, G.out):
                    buf[:n_live] = buf[live]
                for buf in G.k + G.v:            # only the used key slots
                    buf[:n_live, :, :slot] = buf[live, :, :slot]
                G.done[n_live:nb] = True
                G.keep[n_live:nb] = False
                G.keep[n_live:nb, 0] = True
                rows = [rows[i] for i in live.tolist()]
                live_n, B = n_live, nb
        G.step(model, B, slot, pad, eos)
    for i, o in enumerate(G.out[:live_n].tolist()):
        res[rows[i]] = o
    return res


_SAMPLER = {}


def sample_chunk(model, tok, prompts, cfg, temperature, seed, S_min, log):
    """One sampler call: the graph sampler, or fast_generate (cfg['sampler'] == 'compact' or after a graph error)."""
    if cfg['sampler'] == 'graph' and not _SAMPLER.get('graph_off') and torch.cuda.is_available():
        try:
            return graph_generate(model, tok, prompts, temperature=temperature, max_new=cfg['max_new'],
                                  batch=cfg['sample_batch'], seed=seed, S_min=S_min)
        except Exception as e:                   # noqa: BLE001  (BudgetExhausted is a BaseException: not caught)
            _SAMPLER['graph_off'] = True
            _GS.clear()
            log(f'rl: graph sampler failed ({type(e).__name__}: {e}); using fast_generate from here on')
    return fast_generate(model, tok, prompts, temperature=temperature, max_new=cfg['max_new'],
                         batch=cfg['sample_batch'], seed=seed)


def sample_round(model, tok, targets, cfg, seed, budget, rates, reserve, S_min, log, rate=None, temperature=None):
    """Sample k proofs per target in chunks of sample_batch prompts. Before each chunk: stop if est_safety x its
    estimated FLOPs (the largest per-prompt cost seen so far) would not leave `reserve` FLOPs (the round's
    fine-tune share), or at the wall-clock deadline. Returns (list of (target_index, proof), secs, flops, stopped)."""
    model.eval()
    temperature = cfg['temperature'] if temperature is None else temperature
    ks = lib_run2().alloc_k(targets, cfg, rate) if cfg['knapsack'] else [cfg['k']] * len(targets)
    prompts = [(i, t['prompt']) for i, t in enumerate(targets) for _ in range(ks[i])]
    prompts.sort(key=lambda x: len(x[1].split()))       # similar lengths per batch = less padding
    out, t0, u0, stopped = [], time.time(), budget.used(), False
    for c, s in enumerate(range(0, len(prompts), cfg['sample_batch'])):
        chunk = prompts[s:s + cfg['sample_batch']]
        if not budget.fits(cfg['est_safety'] * rates['prompt'] * len(chunk) + reserve):
            stopped = True
            break
        uc = budget.used()
        if cfg['guided']:
            proofs = lib_run2().guided_sample(model, tok, [p for _, p in chunk], cfg, temperature, seed * 100003 + c, log)
        else:
            proofs = sample_chunk(model, tok, [p for _, p in chunk], cfg, temperature, seed * 100003 + c, S_min, log)
        if budget.meter is not None and budget.used() > uc:
            per = (budget.used() - uc) / len(chunk)
            rates['prompt'] = per if not rates.get('measured') else max(rates['prompt'], per)
            rates['measured'] = True
        out += [(i, q) for (i, _), q in zip(chunk, proofs)]
    return out, time.time() - t0, budget.used() - u0, stopped


def judge_samples(targets, samples, found, rnd, relab=None, cfg=None, avoid=None, rate=None):
    """Verify each sample against its prompted sequent; add new distinct proofs to `found`
    ({target name: {normalised proof: record}}). Returns the sample-phase statistics."""
    n_ok = n_qed = new = 0
    solved_now, wl, pl, reasons = set(), [], [], collections.Counter()
    ok_by, n_by = collections.Counter(), collections.Counter()
    ok_len, n_len = collections.Counter(), collections.Counter()
    for i, proof in samples:
        n_len[str(targets[i]['L_true'])] += 1
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
        ok_len[str(t['L_true'])] += 1
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
    win_len = collections.Counter(str(t['L_true']) for t in targets)
    return {'samples': len(samples), 'accept_rate': n_ok / n, 'qed_rate': n_qed / n,
            # this round's window by true length: targets sampled, samples drawn, samples verified, targets solved
            # (solved_by_len_round below); solved_by_len is the cumulative count among the window's targets
            'window_by_len': {L: win_len.get(L, 0) for L in lens}, 'samples_by_len': {L: n_len.get(L, 0) for L in lens},
            'accepted_by_len': {L: ok_len.get(L, 0) for L in lens},
            'targets_solved_round': len(solved_now), 'targets_solved_cum': len(solved_cum), 'new_proofs': new,
            'solved_by_len': {L: by_len.get(L, 0) for L in lens},
            'solved_by_len_round': {L: by_len_round.get(L, 0) for L in lens},
            'mean_written_len': sum(wl) / len(wl) if wl else None, 'mean_pruned_len': sum(pl) / len(pl) if pl else None,
            'fail_reasons': [[r, c] for r, c in reasons.most_common(5)]}


def proof_lines(proof):
    """Written lines of a proof body (each line ends with ';'); equals nd_verify's line count for a valid proof."""
    return proof.split().count(';')


def build_mix(targets, found, retained, cfg, rng, tok, relab=None, rate=None, stats=None):
    """Encoded fine-tuning records: <= max_per_thm proofs per solved target x the rarity weight, plus `retain`
    random cap-6 records. Proofs longer than max_proof_lines_train written lines are left out (before the
    max_per_thm pick); stats (a dict) gets the number left out. Returns (data, n_rl_records)."""
    cap = cfg.get('max_proof_lines_train')
    mix, n_over = [], 0
    for t in targets:
        fs = list(found[t['name']].values())
        if cap is not None:
            n_over += sum(1 for x in fs if x['written'] > cap)
            fs = [x for x in fs if x['written'] <= cap]
        rng.shuffle(fs)
        for x in fs[:cfg['max_per_thm']]:
            mix += [(t['prompt'], x['proof'])] * max(1, round(cfg['rare_weight_max'] * (1 - (rate or {}).get(t['name'], 0.5))))
    for p, q in (relab or {}).values():
        if cap is not None and proof_lines(q) > cap:
            n_over += 1
            continue
        mix += [(p, q)] * cfg['relabel_weight']
    n_rl = len(mix)
    if stats is not None:
        stats['proofs_over_line_cap'] = n_over
    mix += [(r['prompt'], r['proof']) for r in rng.sample(retained, min(cfg['retain'], len(retained)))]
    return [(tok.encode_prompt(p), tok.encode_proof(q)) for p, q in mix], n_rl


def encode_batch(data, idxs, tok, rng, room):
    """The records idxs as token sequences (prompt + proof + QED) and proof-token masks, with the abs-index shift
    capped (run 2, n023 -> n065): each proof renumbered to start at N1, then shifted by a random s in
    [0, MAXN - max(index span, room)], so every proof, and `room` lines of any proof, fit below N64. room=None:
    the frozen tok.shift_abs (starts anywhere in N1..N59). Nothing is cut."""
    r0 = tok.ref0
    seqs, masks = [], []
    for i in idxs:
        p, q = data[i]
        if room is None:
            q = tok.shift_abs(q, rng)
        else:
            refs = [x - r0 + 1 for x in q if x >= r0]
            if refs:
                lo, hi = min(refs), max(refs)
                s = rng.randint(0, max(0, MAXN - max(hi - lo + 1, room)))
                q = [x - (lo - 1) + s if x >= r0 else x for x in q]
        seqs.append(p + q)
        masks.append([0] * len(p) + [1] * len(q))
    return seqs, masks


def pad_rows(seqs, masks, tok, dev):
    """lib.train.batch's layout: one row per sequence, right-padded to the longest. Returns (x, m)."""
    T = max(len(x) for x in seqs)
    x = torch.full((len(seqs), T), tok.pad, dtype=torch.long)
    m = torch.zeros((len(seqs), T), dtype=torch.bool)
    for i, (sq, mk) in enumerate(zip(seqs, masks)):
        x[i, :len(sq)] = torch.tensor(sq)
        m[i, :len(sq)] = torch.tensor(mk, dtype=torch.bool)
    return x.to(dev), m.to(dev)


def make_batch_capped(data, idxs, tok, rng, dev, room):
    """lib.train.batch with the capped abs-index shift (encode_batch), padded (pad_rows). Returns (x, m)."""
    return pad_rows(*encode_batch(data, idxs, tok, rng, room), tok, dev)


def pack_rows(seqs, masks, tok, dev, row_len=1024, max_len=1024):
    """Sequence packing: every sequence goes whole into exactly one row (first-fit-decreasing; a sequence that does
    not fit the space left in any row starts a new row; nothing is split or cut). The row capacity C is the one of
    {the longest sequence} + {multiples of 64 above it, up to min(row_len, max_len)} that needs the fewest charged
    positions (rows x fullest row; ties: the smaller C), so C is never below the longest sequence nor above max_len.
    A sequence longer than max_len is an error. Returns (x, m, pos, seg) on dev:
    tokens (rows x T, T = the fullest row, right-padded), proof mask, positions restarting at 0 for every sequence,
    and segment ids (1.. per sequence in its row, 0 for padding). Checks that the multiset of sequences read back
    from the rows equals the input."""
    L = [len(q) for q in seqs]
    if max(L) > max_len:
        raise ValueError(f'pack_rows: a training sequence of {max(L)} tokens exceeds the model max_len {max_len}')
    order = sorted(range(len(seqs)), key=lambda i: -L[i])

    def ffd(C):
        rows, fill = [], []
        for i in order:
            for r in range(len(rows)):
                if fill[r] + L[i] <= C:
                    rows[r].append(i)
                    fill[r] += L[i]
                    break
            else:
                rows.append([i])
                fill.append(L[i])
        return rows, fill
    top = max(max(L), min(row_len, max_len))
    best = None
    for C in [max(L)] + list(range(64 * (max(L) // 64 + 1), top + 1, 64)):
        rows, fill = ffd(C)
        if best is None or len(rows) * max(fill) < best[0]:
            best = (len(rows) * max(fill), rows, fill)
    _, rows, fill = best
    T = max(fill)
    assert max(L) <= T <= max_len
    x = torch.full((len(rows), T), tok.pad, dtype=torch.long)
    m = torch.zeros((len(rows), T), dtype=torch.bool)
    pos = torch.zeros((len(rows), T), dtype=torch.long)
    seg = torch.zeros((len(rows), T), dtype=torch.long)
    for r, row in enumerate(rows):
        o = 0
        for j, i in enumerate(row):
            n = L[i]
            x[r, o:o + n] = torch.tensor(seqs[i])
            m[r, o:o + n] = torch.tensor(masks[i], dtype=torch.bool)
            pos[r, o:o + n] = torch.arange(n)
            seg[r, o:o + n] = j + 1
            o += n
    back = collections.Counter(tuple(x[r][seg[r] == j + 1].tolist()) for r, row in enumerate(rows) for j in range(len(row)))
    assert back == collections.Counter(tuple(q) for q in seqs), 'pack_rows: packed rows do not hold exactly the input sequences'
    assert sum(fill) == sum(L) and int((seg > 0).sum()) == sum(L)
    return x.to(dev), m.to(dev), pos.to(dev), seg.to(dev)


def packed_logits(model, x, pos, seg):
    """Logits for x[:, :-1] of a packed batch: attention only inside a sequence (block-diagonal causal mask, for every
    position), positions from pos. Padding positions attend to earlier padding of their row (no NaN) and are never
    attended to by a sequence token."""
    s = seg[:, :-1]
    n = s.size(1)
    causal = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
    mask = ((s[:, :, None] == s[:, None, :]) & causal)[:, None]
    return model(x[:, :-1], pos=pos[:, :-1], mask=mask)


def masked_ce(logits, x, m):
    """lib.train.loss_on's loss on given logits: token cross-entropy on the proof tokens x[:, 1:] (mask m[:, 1:]),
    averaged over those tokens."""
    tgt, mk = x[:, 1:], m[:, 1:]
    l = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1), reduction='none')
    return (l * mk.reshape(-1)).sum() / mk.sum()


def finetune(model, tok, data, cfg, rng, device, budget, rates, metrics, state):
    """AdamW, linear warmup then cosine to min_lr, grad clipping, bf16 autocast. Before each step: stop if the
    step's FLOPs (its rows x the measured FLOPs per training row) do not fit, or at the wall-clock deadline.
    Each step is bs random records (as lib.train); with cfg['ft_pack'] they are packed (pack_rows) instead of padded.
    Returns (steps done, seconds, FLOPs, stats: rows / non-pad rows / loss tokens / records / pad_frac / loss)."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], betas=cfg['betas'])
    steps, warm, bs = cfg['ft_steps'], cfg['warmup'], min(cfg['ft_batch'], len(data))
    gem = lib_run2().gem_loss_logits if cfg['loss'] == 'gem' else None

    def lr_at(s):
        if s < warm:
            return cfg['lr'] * s / warm
        return cfg['min_lr'] + 0.5 * (cfg['lr'] - cfg['min_lr']) * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))

    model.train()
    perm, t0, u0, done = [], time.time(), budget.used(), 0
    fst = collections.Counter()
    lsum = torch.zeros((), device=device)        # sum of per-token loss x loss tokens (no per-step host sync)
    for s in range(1, steps + 1):
        if len(perm) < bs:
            perm = list(range(len(data)))
            rng.shuffle(perm)
        idxs = [perm.pop() for _ in range(bs)]
        seqs, masks = encode_batch(data, idxs, tok, rng, cfg['index_room'])
        if cfg['ft_pack']:
            x, m, pos, seg = pack_rows(seqs, masks, tok, device, cfg['pack_row'])
            fill = (seg > 0).sum(1).tolist()
        else:
            x, m = pad_rows(seqs, masks, tok, device)
            fill = [len(q) for q in seqs]
        rows = x.size(0) * (x.size(1) - 1)
        if not budget.fits(1.02 * rates['row'] * rows):
            break
        nonpad = sum(min(n, x.size(1) - 1) for n in fill)           # non-pad input positions of x[:, :-1]
        n_loss = sum(len(data[i][1]) for i in idxs)                   # = m[:, 1:].sum(): the proof tokens
        fst.update(rows=rows, nonpad=nonpad, loss_tokens=n_loss, records=len(idxs))
        us = budget.used()
        lr = lr_at(s)
        for g in opt.param_groups:
            g['lr'] = lr
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            logits = packed_logits(model, x, pos, seg) if cfg['ft_pack'] else model(x[:, :-1])
            loss = gem(logits, x, m, cfg['gem_beta']) if gem else masked_ce(logits, x, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        lsum += loss.detach().float() * n_loss
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        opt.step()
        if budget.meter is not None and budget.used() > us:
            rates['row'] = (budget.used() - us) / rows
        done = s
        state['step'] += 1
        if s % cfg['log_every'] == 0 or s == steps:
            metrics.log(phase='train', step=state['step'], round=state['round'], loss=loss.item(), lr=lr,
                        grad_norm=float(gn), n_records=len(data), tokens=sum(fill), batch=len(idxs), packed_rows=x.size(0),
                        rows=rows, pad_frac=1 - nonpad / max(rows, 1), loss_tokens=n_loss,
                        flops_used=budget.used(), frac_used=budget.frac_used())
    model.eval()
    fst = dict(fst)
    fst['pad_frac'] = 1 - fst['nonpad'] / fst['rows'] if fst.get('rows') else None
    fst['loss_per_token'] = float(lsum) / fst['loss_tokens'] if fst.get('loss_tokens') else None
    return done, time.time() - t0, budget.used() - u0, fst


# ---------------------------------------------------------------- entry point

def run(model, tok, targets, budget_secs, device, seed, log, metrics, paths):
    cfg = dict(CONFIG, **(SMOKE if paths.get('smoke') else {}))
    run2_on = [k for k in ('guided', 'len_weights', 'schema_cover', 'knapsack', 'temp_schedule') if cfg.get(k)]
    run2_on += (['loss=gem'] if cfg['loss'] == 'gem' else []) + (['avg_last'] if cfg['avg_last'] > 1 else [])
    if run2_on:
        cfg = {**lib_run2().DEFAULTS, **cfg}
        log(f'rl: run-2 options on: {run2_on} (lib_run2 {_R2["path"]}, sha256 {_R2["sha"][:12]})')
    budget = Budget(paths.get('budget'), budget_secs, cfg)
    rng = random.Random(seed)
    n_pool = len(targets)
    if cfg['max_len_train'] is not None:
        targets = [t for t in targets if t['L_true'] <= cfg['max_len_train']]
        if not targets:
            raise ValueError(f'max_len_train={cfg["max_len_train"]} leaves no targets in the pool')
    n_kept = len(targets)
    targets = (lib_run2().subset_targets if (cfg['len_weights'] or cfg['schema_cover']) else subset_targets)(targets, cfg, rng)
    max_L = max(t['L_true'] for t in targets)
    cfg['max_new'] = max(cfg['max_new'], cfg['max_new_per_line'] * max_L) if not paths.get('smoke') else cfg['max_new']
    retained = read_jsonl(paths['train_cap6'], limit=cfg['retain'] * 4 if paths.get('smoke') else None)
    assert all(r['n_lines'] <= 6 for r in retained), 'train_cap6 must only hold proofs of <= 6 lines'
    retained = [{'prompt': r['prompt'], 'proof': r['proof']} for r in retained]
    by_len0 = collections.Counter(t['L_true'] for t in targets)
    max_prompt = max(len(tok.encode_prompt(t['prompt'])) for t in targets)
    S_min = max_prompt + cfg['max_new'] + 1
    fpt = linear_flops_per_row(model)
    # FLOP-rate priors until measured: a prompt at most prefill + max_new decode rows (x2 for bucket padding and
    # graph warm-ups), a training row 3 forward rows, a fine-tune step one batch of retention records (padded length)
    probe = random.Random(seed + 1).sample(retained, min(cfg['ft_batch'], len(retained)))   # own RNG: rng untouched
    t_est = max(len(tok.encode_prompt(x['prompt'])) + len(tok.encode_proof(x['proof'])) for x in probe)
    rates = {'prompt': 2.0 * fpt * (max_prompt + cfg['max_new']), 'row': 3.0 * fpt,
             'step': 3.0 * fpt * cfg['ft_batch'] * t_est}
    log(f'rl: {len(targets)} of {n_kept} targets ({n_pool} in the pool; max_len_train {cfg["max_len_train"]}, max_targets '
        f'{cfg["max_targets"]}, stratify {cfg["stratify"]}) {dict(sorted(by_len0.items()))}, {len(retained):,} retained '
        f'cap-6 records loaded at {budget.elapsed():.0f}s; compute limit {budget.limit:.4g} FLOPs '
        f'({budget.used():.3g} used), wall cap {budget_secs:.0f}s, sampler {cfg["sampler"]}, max_new {cfg["max_new"]}')

    found = collections.defaultdict(dict)
    queues = {}
    rate = {}                                    # target name -> verified fraction of its samples, latest round
    relab = {}                                   # class key -> (relabelled prompt, proof)
    avoid = {t['key'] for t in targets}          # never relabel onto a target class (those are sampled for directly)
    state = {'step': 0, 'round': 0}
    avg = lib_run2().WeightAverage(cfg['avg_last']) if cfg['avg_last'] > 1 else None
    rounds, total_samples = [], 0
    round_flops = round_secs = None              # cost of a full round (fine-tune charged at full length)
    step_secs = cfg['default_step_secs']
    stop_reason, r = 'rounds', 0
    try:
        while cfg['rounds'] is None or r < cfg['rounds']:
            r += 1
            # if the previous round's cost says this one would not fit, shrink window and fine-tune to fit
            fr_f = 1.0 if round_flops is None else budget.flops_left() / max(round_flops, 1.0)
            fr_t = 1.0 if round_secs is None else (budget.secs_left() - 3) / round_secs
            frac = min(1.0, fr_f, fr_t)
            if frac < cfg['min_round_frac']:
                stop_reason = 'compute' if fr_f <= fr_t else 'wall clock'
                log(f'rl: stopping before round {r} ({stop_reason}): a round costs ~{round_flops:.3g} FLOPs / {round_secs:.0f}s, '
                    f'{budget.flops_left():.3g} FLOPs / {budget.secs_left():.0f}s left ({budget.frac_used():.2%} of the compute used)')
                break
            if budget.flops_left() <= 0 or budget.secs_left() <= 0:
                stop_reason = 'compute' if budget.flops_left() <= 0 else 'wall clock'
                break
            rcfg = dict(cfg, ft_steps=max(1, int(cfg['ft_steps'] * frac)))
            if frac < 1.0:
                log(f'rl: round {r} shrunk to {frac:.2f} of a full round to fit {budget.flops_left():.3g} FLOPs / '
                    f'{budget.secs_left():.0f}s left')
            state['round'] = r
            tr, ur = time.time(), budget.used()
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
            temp = lib_run2().temperature_at(cfg, budget) if cfg['temp_schedule'] else cfg['temperature']
            reserve = rcfg['ft_steps'] * rates['step']       # sampling leaves the round's fine-tune share
            samples, s_secs, s_flops, s_stop = sample_round(model, tok, win, cfg, seed * 1000 + r, budget, rates, reserve,
                                                            S_min, log, rate, temp)
            st = judge_samples(win, samples, found, r, relab, cfg, avoid, rate)
            st['relabelled_total'] = len(relab)
            cum = collections.Counter(str(t['L_true']) for t in targets if found[t['name']])
            st['solved_cum_all_by_len'] = {str(L): cum.get(str(L), 0) for L in sorted(by_len0)}   # over the whole subset
            total_samples += st['samples']
            metrics.log(phase='sample', step=state['step'], round=r, n_prompts=len(win), k=cfg['k'],
                        temperature=temp, samples_per_sec=st['samples'] / max(s_secs, 1e-9),
                        sample_secs=s_secs, judge_secs=time.time() - tr - s_secs, sample_flops=s_flops,
                        flops_per_sample=s_flops / max(st['samples'], 1), stopped_early=s_stop,
                        flops_used=budget.used(), frac_used=budget.frac_used(), **st)
            log(f'rl r{r}: {st["samples"]} samples in {s_secs:.0f}s ({s_flops:.3g} FLOPs{", stopped early" if s_stop else ""}), '
                f'accept {st["accept_rate"]:.4f}, +{st["new_proofs"]} proofs, solved {st["targets_solved_round"]} this round / '
                f'{st["targets_solved_cum"]} cum {st["solved_by_len"]}, relabelled {len(relab)}')

            mst = {}
            data, n_rl = build_mix(targets, found, retained, cfg, rng, tok, relab, rate, mst)
            steps, ft_secs, ft_flops, fst = 0, 0.0, 0.0, {}
            if n_rl or cfg.get('ft_always'):
                steps, ft_secs, ft_flops, fst = finetune(model, tok, data, rcfg, rng, device, budget, rates, metrics, state)
                if steps:
                    step_secs = ft_secs / steps
                    rates['step'] = ft_flops / steps if ft_flops > 0 else rates['step']
            else:
                log(f'rl r{r}: no verified proofs yet; skipping fine-tuning')
            paths['save_checkpoint'](model, tok, f'{paths["out_dir"]}/round_{r:02d}.pt')
            if avg is not None:
                avg.add(model)
            if cfg['eval_each_round'] and budget.secs_left() > 0:
                te, ue = time.time(), budget.used()
                est = 2.0 * fpt * min(cfg['eval_n'], 1000) * 600           # greedy: <= 512 new tokens + prompt per record
                if budget.fits(est):
                    hg = paths['heldout_greedy'](model, tok, n=cfg['eval_n'])
                    metrics.log(phase='eval', step=state['step'], round=r, heldout_greedy=hg, secs=time.time() - te,
                                flops=budget.used() - ue)
                    log(f'rl r{r}: held-out greedy {hg:.3f}')
            secs, flops = time.time() - tr, budget.used() - ur
            # next round's cost: this round's, with the fine-tune charged at full length even if it was skipped / cut
            round_secs = secs - ft_secs + step_secs * cfg['ft_steps']
            round_flops = flops - ft_flops + rates['step'] * cfg['ft_steps']
            rounds.append({'round': r, 'secs': secs, 'sample_secs': s_secs, 'ft_secs': ft_secs, 'ft_steps': steps,
                           'flops': flops, 'sample_flops': s_flops, 'ft_flops': ft_flops, 'frac_used': budget.frac_used(),
                           'mix_rl_records': n_rl, 'n_records': len(data), 'ft_pad_frac': fst.get('pad_frac'),
                           'ft_loss_per_token': fst.get('loss_per_token'),
                           'ft_rows': fst.get('rows', 0), 'ft_loss_tokens': fst.get('loss_tokens', 0),
                           'proofs_over_line_cap': mst.get('proofs_over_line_cap', 0), **{k: st[k] for k in (
                               'samples', 'accept_rate', 'targets_solved_round', 'targets_solved_cum', 'new_proofs', 'solved_by_len',
                               'window_by_len', 'samples_by_len', 'accepted_by_len', 'solved_by_len_round', 'solved_cum_all_by_len')}})
            metrics.log(phase='train', step=state['step'], round=r, round_end=True, loss=fst.get('loss_per_token'),
                        ft_steps=steps, ft_flops=ft_flops,
                        ft_pad_frac=fst.get('pad_frac'), ft_rows=fst.get('rows', 0), ft_nonpad=fst.get('nonpad', 0),
                        ft_loss_tokens=fst.get('loss_tokens', 0), ft_records=fst.get('records', 0),
                        mix_rl_records=n_rl, n_records=len(data), proofs_over_line_cap=mst.get('proofs_over_line_cap', 0))
            pad = f', padding {fst["pad_frac"]:.1%}' if fst.get('pad_frac') is not None else ''
            log(f'rl r{r}: done in {secs:.0f}s, {flops:.3g} FLOPs (sample {s_secs:.0f}s, fine-tune {steps} steps {ft_secs:.0f}s '
                f'{ft_flops:.3g} FLOPs{pad}, {n_rl} RL records), {budget.elapsed():.0f}s elapsed, {budget.frac_used():.1%} of the compute used')
            if s_stop or (st['samples'] == 0 and steps == 0):   # the budget no longer fits a sampling chunk: done
                stop_reason = 'compute' if budget.secs_left() > 0 else 'wall clock'
                log(f'rl: stopping after round {r} ({stop_reason}): sampling stopped early, {budget.frac_used():.2%} of the compute used')
                break
    except BudgetExhausted as e:                 # should not happen: every phase checks the meter first
        stop_reason = 'BudgetExhausted'
        model.eval()
        log(f'rl: {e} in round {r}; returning the current weights (planner overran: please report)')

    if avg is not None:
        avg.apply(model, log)                    # CPU weight arithmetic: no model compute
    with open(f'{paths["out_dir"]}/found_targets.jsonl', 'w') as f:
        for t in targets:
            for x in found[t['name']].values():
                f.write(json.dumps({'name': t['name'], 'thm': t['thm'], 'L_true': t['L_true'], **x}) + '\n')
    solved_cum = sum(1 for d in found.values() if d)
    ft_rows = sum(x['ft_rows'] for x in rounds)
    ft_nonpad = sum(x['ft_rows'] * (1 - x['ft_pad_frac']) for x in rounds if x['ft_pad_frac'] is not None)
    totals = {'rounds': len(rounds), 'total_samples': total_samples, 'total_train_steps': state['step'],
              'ft_pad_frac': 1 - ft_nonpad / ft_rows if ft_rows else None,
              'targets_solved_cum': solved_cum, 'n_targets': len(targets), 'n_pool': n_pool, 'n_after_max_len': n_kept,
              'targets_by_len': {str(k): v for k, v in sorted(by_len0.items())}, 'wall_secs': budget.elapsed(),
              'flops_used': budget.used(), 'frac_used': budget.frac_used(), 'stop_reason': stop_reason,
              'sampler_used': 'compact' if (_SAMPLER.get('graph_off') or cfg['sampler'] != 'graph') else 'graph'}
    metrics.log(phase='summary', step=state['step'], **totals)
    metrics.set_summary(**totals, per_round=rounds, config=cfg, flop_rates=dict(rates),
                        lib_run2_sha256=_R2.get('sha'), run2_options=run2_on,
                        guide_stats=dict(lib_run2().GUIDE_STATS) if cfg['guided'] else None)
    log(f'rl: {len(rounds)} rounds, {budget.frac_used():.2%} of the compute used, stop: {stop_reason}, fine-tune padding '
        f'{totals["ft_pad_frac"] if totals["ft_pad_frac"] is None else format(totals["ft_pad_frac"], ".1%")}')
    model.eval()
    return model
