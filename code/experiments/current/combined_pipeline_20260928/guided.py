# imports the fork, which is not resolvable here
# pyright: reportMissingImports=false
"""Verifier-guided RL sampling for lean_seq: Leon's run-2 guided sampler (the biggest single gain
in his run 2), ported from abs lines to the Lean surface form.

Whenever a row writes `;` (the end of a `have`, at any box depth), the Lean tokens so far are
parsed with the fork's own lean_tok.inverse into ND lines, and each new ND line goes through
Leon's LineChecker (nd_verify's checks, one line at a time). A rejected segment (the tokens since
the last accepted `;`) has its KV-cache slots masked out and the row resamples it from the saved
distribution at the segment start; <eos> is accepted only if the whole text parses and the proof
is complete. max_retries rejections of one segment, or max_rollbacks in total, end the row.
Samples still go through the Lean gate afterwards: guidance only prunes, never accepts.
"""

import collections
import copy
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from fast_sample import _index_caches
from lean_gate import gate  # fork
from lean_tok import ParseFail, inverse  # fork
from line_checker import LineChecker, _prompt_sequent

STATS = collections.Counter()


class Incomplete(Exception):
    pass


class _Prefix(list):
    """Token list that claims to be endless and raises Incomplete past its real end, so the fork's
    inverse() parses a proof prefix exactly as it parses a whole proof and stops at the cut."""

    def __len__(self):
        return 1 << 30

    def __getitem__(self, i):
        if i >= list.__len__(self):
            raise Incomplete
        return list.__getitem__(self, i)


def prefix_lines(toks):
    """ND lines (spec.md strings) the Lean token prefix denotes so far, or None if the prefix is
    already outside the grammar. The lines live in inverse()'s local `out`, which the Incomplete
    exception leaves reachable through its traceback frame; reading them from there (instead of
    copying the 80-line parser) keeps the grammar exactly the pinned fork's."""
    try:
        inverse(_Prefix(toks))
    except Incomplete as e:
        tb = e.__traceback__
        while tb is not None and tb.tb_frame.f_code is not inverse.__code__:
            tb = tb.tb_next
        return list(tb.tb_frame.f_locals["out"])
    except (ParseFail, KeyError, IndexError):
        return None
    return None  # a complete proof before <eos> is not a prefix


class RowCheck:
    """One row's accepted state: ND lines checked so far and the LineChecker behind them."""

    def __init__(self, prompt):
        prem, concl = _prompt_sequent(prompt)
        self.chk, self.n = LineChecker(prem, concl), 0

    def line_ok(self, toks):
        lines = prefix_lines(toks)
        if lines is None or len(lines) <= self.n:
            return False
        trial = copy.deepcopy(self.chk)
        if not all(trial.add_line(x.split()) for x in lines[self.n :]):
            return False
        self.chk, self.n = trial, len(lines)
        return True

    def eos_ok(self, toks):
        try:
            nd = inverse(toks)
        except (ParseFail, KeyError, IndexError):
            return False
        return nd.count(";") == self.n and self.chk.done_ok()


@torch.no_grad()
def _guided_ids(model, tok, prompt_ids, prompts, temperature, max_new, gen, cfg):
    dev = next(model.parameters()).device
    B = len(prompt_ids)
    L = max(len(p) for p in prompt_ids)
    semi, eos, itos = tok.stoi[";"], tok.eos, tok.itos
    idx = torch.full((B, L), tok.pad, dtype=torch.long, device=dev)
    keep = torch.zeros((B, L), dtype=torch.bool, device=dev)
    for i, p in enumerate(prompt_ids):
        idx[i, L - len(p) :] = torch.tensor(p, device=dev)
        keep[i, L - len(p) :] = True
    pos = (keep.cumsum(1) - 1).clamp(min=0)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    mask = causal[None, None] & keep[:, None, None, :]
    mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]
    T = max_new + cfg["guide_slack"]
    caches = [{"max": L + T} for _ in model.blocks]
    logits = model(idx, pos=pos, mask=mask, caches=caches)[:, -1].float()
    saved = logits.clone()  # distribution at the start of each row's current segment
    rows = list(range(B))  # original row of each live slot
    rows_t = torch.arange(B, device=dev)
    # Per original row, on the device: written tokens (accepted lines, then the open segment),
    # how many are written (ptr), and where the open segment starts (sst). Only rows that just
    # wrote `;` or <eos> (or hit a length cap) need Python in a step.
    out = torch.full((B, T + 1), tok.pad, dtype=torch.long, device=dev)
    ptr = torch.zeros(B, dtype=torch.long, device=dev)
    sst = torch.zeros(B, dtype=torch.long, device=dev)
    fin_o = torch.zeros(B, dtype=torch.bool, device=dev)
    chk = [RowCheck(p) for p in prompts]
    lstart_c = [L] * B  # cache index where the current segment starts
    lstart_p = pos[:, -1].tolist()  # position of the last accepted token
    retries, rollbacks = [0] * B, [0] * B
    cur_pos = pos[:, -1].clone()
    use_saved = torch.zeros(B, dtype=torch.bool, device=dev)
    sst_py, fin_py = [0] * B, [False] * B  # host mirrors, so the checks below never sync
    for t in range(T):
        lg = torch.where(use_saved[:, None], saved, logits)
        nxt = torch.multinomial(F.softmax(lg / temperature, -1), 1, generator=gen).squeeze(1)
        fin = fin_o[rows_t]
        live = ~fin
        r_live, x_live = rows_t[live], nxt[live]
        out[r_live, ptr[r_live]] = x_live
        ptr[r_live] += 1
        is_semi, is_eos = x_live == semi, x_live == eos
        n_seg = ptr[r_live] - sst[r_live]
        too_long = ~is_semi & ~is_eos & ((ptr[r_live] >= max_new) | (n_seg > cfg["guide_seg_max"]))
        slot = live.nonzero().squeeze(1)
        sel = is_semi | is_eos
        sel_j, sel_eos = slot[sel].tolist(), is_eos[sel].tolist()
        long_j = slot[too_long].tolist()
        toks = out[rows_t[slot[sel]]].tolist() if sel_j else []
        ptrs = ptr[rows_t[slot[sel]]].tolist() if sel_j else []
        # Launch this step's forward before the checks, so the GPU runs while the CPU checks.
        # Every slot feeds its sampled token (finished slots pad). A slot rolled back below gets
        # its segment masked out of the cache before the next forward, and its logits from this
        # forward are never used (it samples from `saved`), so outputs match checking first.
        n_c = caches[0]["n"]  # the cache index this step's token lands in
        cur_pos = cur_pos + 1
        keep = torch.cat([keep, torch.ones(keep.size(0), 1, dtype=torch.bool, device=dev)], 1)
        logits = model(
            torch.where(fin, torch.full_like(nxt, tok.pad), nxt)[:, None],
            pos=cur_pos[:, None],
            mask=keep[:, None, None, :],
            caches=caches,
        )[:, -1].float()
        acc, rb, kill = [], [], list(long_j)
        STATS["kill_len"] += len(long_j)
        for j, e, row, n in zip(sel_j, sel_eos, toks, ptrs, strict=True):
            r = rows[j]
            if e:
                if chk[r].eos_ok([itos[y] for y in row[: n - 1]]):
                    fin_py[r] = True
                    STATS["eos_ok"] += 1
                else:
                    rb.append(j)
                    STATS["rb_eos"] += 1
            elif chk[r].line_ok([itos[y] for y in row[:n]]):
                sst_py[r] = n
                retries[r] = 0
                acc.append(j)
                STATS["lines_ok"] += 1
            else:
                rb.append(j)
                STATS["rb_line"] += 1
        for j in rb:
            r = rows[j]
            retries[r] += 1
            rollbacks[r] += 1
            STATS["rollbacks"] += 1
            if retries[r] > cfg["guide_retries"] or rollbacks[r] > cfg["guide_rollbacks"]:
                kill.append(j)
                STATS["kill_retries"] += 1
        killed = [rows[j] for j in dict.fromkeys(kill) if not fin_py[rows[j]]]
        for r in killed:
            fin_py[r] = True
        STATS["killed"] += len(killed)
        # batched device updates: accepted segments move sst; rollbacks and kills rewind ptr
        # to sst (a killed row keeps only its accepted lines); eos-accepts and kills finish
        touched = [rows[j] for j in acc + rb] + killed
        if touched:
            ti = torch.tensor(touched, device=dev)
            sst[ti] = torch.tensor([sst_py[r] for r in touched], device=dev)
        rewind = [rows[j] for j in rb] + killed
        if rewind:
            ri = torch.tensor(rewind, device=dev)
            ptr[ri] = sst[ri]
        done_now = [rows[j] for j, e in zip(sel_j, sel_eos, strict=True) if e and fin_py[rows[j]]]
        if done_now or killed:
            fin_o[torch.tensor(done_now + killed, device=dev)] = True
        if all(fin_py[r] for r in rows):
            break
        rb_live = [j for j in rb if not fin_py[rows[j]]]
        for j in rb_live:
            keep[j, lstart_c[rows[j]] : n_c + 1] = False
        use_saved = torch.zeros_like(use_saved)
        if rb_live:
            jj = torch.tensor(rb_live, device=dev)
            use_saved[jj] = True
            cur_pos[jj] = torch.tensor([lstart_p[rows[j]] for j in rb_live], device=dev)
        if acc:
            ja = torch.tensor(acc, device=dev)
            saved[ja] = logits[ja]
            for j, p in zip(acc, cur_pos[ja].tolist(), strict=True):
                lstart_c[rows[j]] = n_c + 1
                lstart_p[rows[j]] = p
        if t % 16 == 15:
            live_j = [j for j, r in enumerate(rows) if not fin_py[r]]
            if len(live_j) < 0.75 * len(rows):
                lv = torch.tensor(live_j, device=dev)
                _index_caches(caches, lv, len(rows))
                rows = [rows[j] for j in live_j]
                rows_t = rows_t[lv]
                keep, cur_pos, logits, saved, use_saved = (
                    keep[lv],
                    cur_pos[lv],
                    logits[lv],
                    saved[lv],
                    use_saved[lv],
                )
    STATS["rows"] += B
    # rows still open at the step limit keep their accepted lines only
    n_keep = torch.where(fin_o, ptr, sst).tolist()
    return [row[:n] for row, n in zip(out.tolist(), n_keep, strict=True)]


def generate(model, tok, prompts, temperature, max_new, batch, seed, cfg):
    """Same contract as the fork's generate(greedy=False); Lean-checked on a worker thread while
    the next chunk decodes (as fast_sample.generate)."""
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    ids = [tok.encode_prompt(p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(ids[i]))
    res = [None] * len(prompts)
    futs = []
    with ThreadPoolExecutor(1) as pool:
        for s in range(0, len(order), batch):
            chunk = order[s : s + batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                outs = _guided_ids(
                    model,
                    tok,
                    [ids[i] for i in chunk],
                    [prompts[i] for i in chunk],
                    temperature,
                    max_new,
                    gen,
                    cfg,
                )
            nd, texts = [], []
            for o in outs:
                nd.append(tok.decode(o))
                texts.append(tok.last_text)
            futs.append((chunk, pool.submit(gate, tok, [prompts[i] for i in chunk], nd, texts)))
        for chunk, fut in futs:
            for i, r in zip(chunk, fut.result(), strict=True):
                res[i] = r
    return res
