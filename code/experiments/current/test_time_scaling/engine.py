# imports the fork and the combined pipeline's samplers, which are not resolvable here
# pyright: reportMissingImports=false
"""Segment-level sampling engine for test-time methods 1-3 (test_time_scaling_initial_plan.md).

A segment is the tokens after the last accepted `;` up to and including the next `;` or <eos>.

  sample_segments  samples one segment for each (prefix, budget) item, batched across items.
                   No KV cache survives between calls: every call prefills its prefixes again.
                   That is simple and lets tree search branch freely, at the cost of prefill
                   compute, which is counted separately from sampled tokens.
  Node / check     a valid proof prefix and the verdict on one sampled segment. Checks are
                   guided.py's (fork lean_tok.inverse on the prefix + Leon's LineChecker).
  Gate             Lean AND nd_verify on finished proofs, with verdicts cached per text.
"""

import copy
import time

import torch
import torch.nn.functional as F
from fast_sample import _index_caches
from guided import prefix_lines
from lean_gate import gate
from lean_tok import ParseFail, inverse
from line_checker import LineChecker, _prompt_sequent
from nd_verify import verify_text

COMPACT_BELOW, COMPACT_EVERY = 0.75, 8
PREFILL_BLOCK = 128  # query positions per prefill step: ALiBi's (B, H, T, S) fp32 bias is the
# memory peak, ~12 GB for 1,024 rows x 600 positions in one step


@torch.no_grad()
def _segments(model, tok, seqs, budgets, temperature, gen):
    """One batch: prefill left-padded `seqs`, then decode until each row writes `;` or <eos> or
    reaches its budget. Returns per row (segment ids, logprob of the segment at T = 1)."""
    dev = next(model.parameters()).device
    B, L = len(seqs), max(len(s) for s in seqs)
    idx = torch.full((B, L), tok.pad, dtype=torch.long, device=dev)
    keep = torch.zeros((B, L), dtype=torch.bool, device=dev)
    for i, s in enumerate(seqs):
        idx[i, L - len(s) :] = torch.tensor(s, device=dev)
        keep[i, L - len(s) :] = True
    pos = (keep.cumsum(1) - 1).clamp(min=0)
    T = max(budgets)
    caches = [{"max": L + T} for _ in model.blocks]
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    eye = torch.eye(L, dtype=torch.bool, device=dev)
    for a in range(0, L, PREFILL_BLOCK):  # pads attend to themselves (no NaN), as in sample.py
        b = min(a + PREFILL_BLOCK, L)
        mask = (causal[a:b, :b][None, None] & keep[:, None, None, :b]) | eye[a:b, :b][None, None]
        logits = model(idx[:, a:b], pos=pos[:, a:b], mask=mask, caches=caches)[:, -1].float()
    stop = torch.zeros(logits.size(-1), dtype=torch.bool, device=dev)
    stop[[tok.stoi[";"], tok.eos]] = True
    budget = torch.tensor(budgets, device=dev)
    out = torch.full((B, T), tok.pad, dtype=torch.long, device=dev)
    n_out = torch.zeros(B, dtype=torch.long, device=dev)
    logp = torch.zeros(B, device=dev)
    rows = torch.arange(B, device=dev)  # live slot -> original row
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    cur_pos = pos[:, -1]
    for t in range(T):
        nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1, generator=gen).squeeze(1)
        lp = F.log_softmax(logits, -1).gather(1, nxt[:, None]).squeeze(1)
        live = ~done
        out[rows, t] = torch.where(live, nxt, tok.pad)
        n_out[rows] += live.long()
        logp[rows] += torch.where(live, lp, 0.0)
        done = done | stop[nxt] | (t + 1 >= budget[rows])
        if t % COMPACT_EVERY == COMPACT_EVERY - 1:
            n_live = int((~done).sum())
            if n_live == 0:
                break
            if n_live < COMPACT_BELOW * done.numel():
                lv = (~done).nonzero().squeeze(1)
                _index_caches(caches, lv, done.numel())
                rows, nxt, cur_pos, keep, done = rows[lv], nxt[lv], cur_pos[lv], keep[lv], done[lv]
        cur_pos = cur_pos + 1
        keep = torch.cat([keep, torch.ones(keep.size(0), 1, dtype=torch.bool, device=dev)], 1)
        logits = model(
            nxt[:, None], pos=cur_pos[:, None], mask=keep[:, None, None, :], caches=caches
        )[:, -1].float()
    n = n_out.tolist()
    return [(o[:k], p) for o, k, p in zip(out.tolist(), n, logp.tolist(), strict=True)]


def sample_segments(model, tok, items, temperature, gen, batch, stats):
    """items: (token ids to condition on, max segment tokens). Returns one (segment ids, end,
    logprob) per item, end in {"semi", "eos", "len"}. Adds prefill/sampled tokens and GPU
    seconds to `stats`."""
    t0 = time.time()
    dev = next(model.parameters()).device
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    res = [None] * len(items)
    semi = tok.stoi[";"]
    for s in range(0, len(order), batch):
        chunk = order[s : s + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
            outs = _segments(
                model,
                tok,
                [items[i][0] for i in chunk],
                [items[i][1] for i in chunk],
                temperature,
                gen,
            )
        for i, (seg, lp) in zip(chunk, outs, strict=True):
            end = (
                "semi"
                if seg and seg[-1] == semi
                else "eos"
                if seg and seg[-1] == tok.eos
                else "len"
            )
            res[i] = (seg, end, lp)
            stats["prefill_tokens"] += len(items[i][0])
            stats["sampled_tokens"] += len(seg)
    stats["gpu_secs"] += time.time() - t0
    return res


class Node:
    """A valid proof prefix: generated ids, the LineChecker after its ND lines, how many ND lines
    and segments it has, its logprob at T = 1, and its ND lines as a dedup key."""

    __slots__ = ("gen", "chk", "n", "depth", "logp", "key")

    def __init__(self, gen, chk, n, depth, logp, key):
        self.gen, self.chk, self.n, self.depth, self.logp, self.key = gen, chk, n, depth, logp, key

    @classmethod
    def root(cls, prompt):
        prem, concl = _prompt_sequent(prompt)
        return cls([], LineChecker(prem, concl), 0, 0, 0.0, ())


def check(node, seg, end, lp, itos):
    """Verdict on `seg` sampled after `node`: ("ok", child), ("done", finished node whose `gen`
    ends in <eos>), or (cause, None) with cause in {"len", "parse", "rule", "eos"}."""
    if end == "len":
        return "len", None
    gen = node.gen + seg
    if end == "eos":
        try:
            nd = inverse([itos[i] for i in gen[:-1]])
        except (ParseFail, KeyError, IndexError):
            return "eos", None
        if nd.count(";") != node.n or not node.chk.done_ok():
            return "eos", None
        return "done", Node(gen, node.chk, node.n, node.depth + 1, node.logp + lp, node.key)
    lines = prefix_lines([itos[i] for i in gen])
    if lines is None or len(lines) <= node.n:
        return "parse", None
    chk = copy.deepcopy(node.chk)
    if not all(chk.add_line(x.split()) for x in lines[node.n :]):
        return "rule", None
    return "ok", Node(gen, chk, len(lines), node.depth + 1, node.logp + lp, tuple(lines))


class Gate:
    """Lean AND nd_verify on finished proofs (the harness's judge), cached per (prompt, text)."""

    def __init__(self, tok, stats):
        self.tok, self.stats, self.cache = tok, stats, {}

    def __call__(self, prompts, gens):
        t0 = time.time()
        keys, todo = [], {}
        for p, g in zip(prompts, gens, strict=True):
            nd = self.tok.decode(g)
            k = (p, self.tok.last_text)
            keys.append(k)
            if k not in self.cache:
                todo[k] = nd
        if todo:
            ks = list(todo)
            res = gate(self.tok, [k[0] for k in ks], [todo[k] for k in ks], [k[1] for k in ks])
            for k, r in zip(ks, res, strict=True):
                self.cache[k] = verify_text(k[0] + " " + r)[0]
        self.stats["gate_secs"] += time.time() - t0
        self.stats["gated"] += len(todo)
        return [self.cache[k] for k in keys]
