"""Shared pieces of the Stage-2 tuning trainers of SPEC2.md: lh_dpo.py, lh_sft.py, lh_rm.py (TUNING.md).

    python lh_tune.py toy --out DIR [--n 160] [--heldout 40] [--kl 64] [--seed 0] [--habit reit]  # smoke pairs
    python lh_tune.py probe-toy --out PROBE.jsonl [--habits note_atom,conj,...] [--n 40]          # smoke RM probe

Loading. `setup` installs Dan's tree (tag_core.install_dan: `lean_staten_tag` checkpoints load), forces offline mode
and caps CPU threads at 8; `load_base` loads with Dan's `model.load_ckpt`, the loader authors_measure / lh_measure
sample with, so a model trained here is the model they would read.

Encoding. `encode_proof(tok, prompt, proof)`: `state_env.decompose(prompt, proof, canon=True)` (Stage-1's
`state_train.load`) -> one row per step, row = state ids + action ids + `<eos>`, `plen` = the state's length. Every
state is UNTAGGED (`<st> ... <act>`, no tag token): `tok.author_tag = 'none'` while encoding, whatever $AUTHOR_TAG is.

Scoring. `logp_fn(model)` gives per row the sum over the action tokens and the closing `<eos>` of log softmax(logits)
at T 1, teacher forced: exactly the target positions of Stage-1's loss (`state_train.loss_on` / `best_model.mtp_loss`:
targets x[1:], mask = the action half). No vocabulary mask: Dan's sampler (`state_sample.env_generate` ->
`generate_ids_fast`) samples the full softmax, and the GRPO sampler's `phase_bans` are empty for a `lean_staten_tag`
tokenizer (no think or effort tokens). log pi(proof) = the sum of its rows' values.

Pairs. `split_common` (`common_mask`) drops the rows the two proofs of a pair share (as a multiset). A shared row adds
the same amount to log pi(w) and log pi(l) under every model and every name shift applied to both, so log pi(w) -
log pi(l) is unchanged and only the differing rows are computed (test_lh_dpo checks it against the full sums). The
dropped rows get no direct gradient in DPO: for a twin pair, every row before the first differing one (PREREG
Amendment 3 point 1; test_lh_dpo pins it on the logits).

Name shift. Stage-1 trains with a random `lean_seq` name offset (`tok.shift_pair`; `state_train_best.make_batch`), and
Dan's sampler draws a name base per attempt. Training here draws one offset s ~ U[0, 64 - largest name] per proof (DPO
and the reward model: per pair, shared by both proofs and all steps) or per row (SFT, as Stage-1) per presentation;
every evaluation is at offset 0 (as Stage-1's validation loss).

The objective in chunks (`objective_backward`). One pass when the rows fit in --max-tokens padded tokens; otherwise a
no-grad pass gives the per-row outputs, autograd gives d loss / d output, and a second pass back-propagates
sum(output x that gradient) chunk by chunk: the exact gradient, with memory bounded by one chunk.

Snapshots (`save_snapshot`): {'cfg', 'state', 'tok_mode', 'extra'}, exactly the keys `model.save_ckpt` writes, with
the base's extra kept (author_tags, steps, ... so `tag_core.ckpt_meta` reads the same variant and mode) and
`extra['tuning']` added; beside it a copy of the base's `.tag.json` (D, author tags) with `ckpt` and `tuning` updated.

The loop (`train_loop`) shared by DPO and SFT: AdamW as the RL recipe (betas 0.9 / 0.95, weight decay 0, grad-norm clip
1.0, a linear warmup over the first 10 steps, then constant lr), held-out evaluation (`Evaluator`: both held-out
accuracies, `DEFINITIONS`), the snapshot rule (`Snapshots`, `snap_plan`: the step grid `GRID` of PREREG Amendment 3,
accuracy-triggered snapshots only with --acc-snapshots), `metrics.jsonl`, `train.jsonl`, `done.json`. What --seed
changes: `SEED_AFFECTS`.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

HERE = Path(__file__).resolve().parent
TAGS_DIR = HERE.parent / "author_tags_20261110"
for _p in (str(TAGS_DIR), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import tag_core as T  # noqa: E402

LH = Path(os.environ.get("LH", "/scratch/cowifornia/nd-rl/linked_habits"))
MAXN = 64  # lean_tok.MAXN: names n1..n64
THRESHOLDS = (
    60,
    75,
    90,
)  # accuracy-triggered snapshots (--acc-snapshots, off by default since PREREG Amendment 3)
GRID = (
    25,
    50,
    100,
    150,
    200,
    300,
    400,
    600,
    800,
    1000,
    1500,
    2000,
)  # PREREG Amendment 3 point 3: snapshot steps
# Held-out preference accuracy, two definitions (metrics.jsonl, done.json). The absolute one is biased against the
# longer member (reiteration's A form has extra rows, each with log-probability < 0); the relative one is DPO's reward
# accuracy and is 0 at step 0 by construction (every margin is exactly 0 there).
ACC_ABS = "heldout_acc_abs"
ACC_REL = "heldout_acc_rel"
DEFINITIONS = {
    ACC_ABS: "share of held-out pairs with log pi(w) > log pi(l) (absolute; biased against the longer member)",
    ACC_REL: "share of held-out pairs with [log pi(w) - log pi(l)] - [log pi_ref(w) - log pi_ref(l)] > 0 "
    "(relative to the base; DPO's reward accuracy; 0 at step 0)",
    "heldout_margin_rel": "mean over held-out pairs of [log pi(w) - log pi(l)] - [log pi_ref(w) - log pi_ref(l)], nats",
    "heldout_logratio_abs": "mean over held-out pairs of log pi(w) - log pi(l), nats",
    "heldout_loss": "DPO loss on the held-out pairs at --beta (relative margin)",
    "kl": "mean over the action positions of the --kl-states proofs of KL(pi_ref || pi), nats per action token, T 1",
    "loss": "train loss averaged over the steps since the previous row (DPO: the DPO loss; SFT: CE per action token)",
    "train_margin_rel": "DPO: train-batch relative margin (nats), averaged over the steps since the previous row",
    "train_acc_rel": "DPO: share of train-batch pairs with relative margin > 0, averaged likewise",
}
SNAP_KEYS = (ACC_ABS, ACC_REL, "heldout_margin_rel", "heldout_logratio_abs", "kl")
SEED_AFFECTS = (
    "--seed sets the order of the training examples (a seeded shuffle per epoch, random.Random(seed + 1)) and the "
    "random name offsets drawn from the same generator, and torch.manual_seed(seed), which nothing in these trainers "
    "draws from (no dropout; snapshots start from the base; the reward head starts at zero). It does not set the "
    "swap-random orientation (--swap-seed), the held-out or KL sets, or the evaluation (offset 0, fp32). On cuda, "
    "bf16 kernels can make two runs of the same seed differ slightly."
)
RL_LR = 1.5e-4  # tag_line / knockout common.grpo_argv: --lr 1.5e-4
BETAS = (0.9, 0.95)  # grpo.py: AdamW(betas=(0.9, 0.95), weight_decay=0.0)
CLIP = 1.0  # grpo.py --clip 1.0
WARMUP = 10  # grpo.py --warmup 10 (AdamW steps)
PAIR_KEYS = ("key", "prompt", "preferred", "other")


# ============================================================================ setup and loading
def cpu_threads(threads: int = 8) -> int:
    """The intra-op thread count: `threads`, at most 8, and at most $OMP_NUM_THREADS when that is set."""
    env = os.environ.get("OMP_NUM_THREADS", "8")
    cap = min(8, int(env)) if env.isdigit() else 8
    return max(1, min(cap, threads))


def setup(device: str = "cpu", threads: int = 8) -> None:
    """Dan's tree on the path (tag tokenizer dispatch), offline, CPU threads capped (`cpu_threads`)."""
    os.environ.setdefault("ND_OFFLINE", "1")
    os.environ.setdefault("ND_REGISTRY_SYNC", "0")
    os.environ.setdefault("ND_REGISTRY", "0")
    T.install_dan()
    import torch

    torch.set_num_threads(cpu_threads(threads))
    if device == "cuda":
        assert torch.cuda.is_available(), "--device cuda but no GPU visible"


def load_base(path: str | Path, device: str) -> tuple[Any, Any, dict[str, Any]]:
    """(model, tok, the checkpoint's top-level dict without 'state') through Dan's `model.load_ckpt`."""
    import torch
    from model import load_ckpt  # type: ignore[import-not-found]

    model, tok, _ = load_ckpt(str(path), device)
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    meta = {k: v for k, v in ck.items() if k != "state"}
    meta["state_keys"] = sorted(ck["state"])
    return model, tok, meta


def sha256(fn: str | Path) -> str:
    h = hashlib.sha256()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(p)


def write_json(path: str | Path, doc: dict[str, Any]) -> None:
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    tmp.replace(p)


# ============================================================================ encoding
class Row(NamedTuple):
    ids: bytes  # state ids + action ids + <eos> (vocabulary < 256)
    plen: int  # the state's length; targets are ids[plen:]


class Unreadable(ValueError):
    """A proof that does not decompose into (state, action) steps."""


def encode_proof(tok: Any, prompt: str, proof: str) -> list[Row]:
    """The proof's steps as rows, states untagged (module docstring). Raises Unreadable."""
    from state_env import decompose  # type: ignore[import-not-found]

    try:
        steps, _, _ = decompose(prompt, proof, canon=True)
    except Exception as e:  # ParseFail, ValueError, AssertionError, IndexError, KeyError, ...
        raise Unreadable(f"{type(e).__name__}: {e}") from e
    if not steps:
        raise Unreadable("no steps")
    tagged = getattr(tok, "mode", None) == T.MODE
    old = getattr(tok, "author_tag", None)
    if tagged:
        tok.author_tag = "none"
    try:
        rows = []
        for st, act, _ in steps:
            p = tok.encode_toks(st)
            q = tok.encode_toks(act) + [tok.eos]
            if p[-1] != tok.stoi["<act>"] or (tagged and tok.state_tag(p) != "none"):
                raise Unreadable("a state that is not an untagged state")
            ids = p + q
            if max(ids) > 255:
                raise Unreadable("an id past 255")
            rows.append(Row(bytes(ids), len(p)))
        return rows
    except Unreadable:
        raise
    except Exception as e:  # an unknown token (KeyError) in a model-written proof
        raise Unreadable(f"{type(e).__name__}: {e}") from e
    finally:
        if tagged:
            tok.author_tag = old


def max_name(rows: Sequence[Row], ref0: int) -> int:
    """The largest name index (n<k> -> k) in the rows, 0 if none."""
    return max((x - ref0 + 1 for r in rows for x in r.ids if x >= ref0), default=0)


def common_mask(rw: Sequence[Row], rl: Sequence[Row]) -> tuple[list[bool], list[bool], int]:
    """Which rows of each proof are kept (True) and which are shared with the other proof (False), as multisets:
    a row that occurs k times in one proof and j times in the other loses its first min(k, j) occurrences in each."""
    common = collections.Counter(rw) & collections.Counter(rl)

    def keep(rows: Sequence[Row]) -> list[bool]:
        left, out = dict(common), []
        for r in rows:
            shared = left.get(r, 0) > 0
            if shared:
                left[r] -= 1
            out.append(not shared)
        return out

    return keep(rw), keep(rl), sum(common.values())


def split_common(rw: Sequence[Row], rl: Sequence[Row]) -> tuple[list[Row], list[Row], int]:
    """(rows of w not in l, rows of l not in w, shared rows), as multisets, order kept (`common_mask`)."""
    kw, kl, n = common_mask(rw, rl)
    return (
        [r for r, k in zip(rw, kw, strict=True) if k],
        [r for r, k in zip(rl, kl, strict=True) if k],
        n,
    )


class Pair(NamedTuple):
    key: str
    prompt: str
    dw: list[Row]  # rows of the preferred proof that the other does not have
    dl: list[Row]  # rows of the other proof that the preferred does not have
    rw: list[Row]  # all rows of the preferred proof
    rl: list[Row]  # all rows of the other proof
    room: int  # the largest name offset that keeps every name of both proofs <= n64
    swapped: bool


def load_pairs(
    path: str | Path, tok: Any, swap: set[int] | None = None, max_unreadable: float = 0.01
) -> tuple[list[Pair], dict[str, Any]]:
    """A pair file (SPEC2's fields) -> Pairs (`swap`: indices whose two proofs trade places) and a report. Pairs
    whose proofs do not decompose, or whose two proofs give the same rows, are dropped and counted; more than
    `max_unreadable` of the file unreadable is an error (a format mismatch, not noise)."""
    recs = read_jsonl(path)
    out: list[Pair] = []
    bad: list[str] = []
    same = 0
    rows_all = rows_common = 0
    for i, r in enumerate(recs):
        miss = [k for k in PAIR_KEYS if k not in r]
        if miss:
            raise ValueError(f"{path}: record {i} lacks {miss}")
        w, l_ = r["preferred"], r["other"]
        sw = bool(swap and i in swap)
        if sw:
            w, l_ = l_, w
        try:
            rw, rl = encode_proof(tok, r["prompt"], w), encode_proof(tok, r["prompt"], l_)
        except Unreadable:
            bad.append(str(r["key"]))
            continue
        dw, dl, nc = split_common(rw, rl)
        if not dw and not dl:
            same += 1
            continue
        rows_all += len(rw) + len(rl)
        rows_common += 2 * nc
        room = MAXN - max(max_name(rw, tok.ref0), max_name(rl, tok.ref0))
        out.append(Pair(str(r["key"]), r["prompt"], dw, dl, rw, rl, max(0, room), sw))
    rep = {
        "file": str(path),
        "sha256": sha256(path),
        "records": len(recs),
        "pairs": len(out),
        "unreadable": len(bad),
        "unreadable_keys": bad[:20],
        "identical_rows": same,
        "swapped": sum(p.swapped for p in out),
        "rows": rows_all,
        "rows_shared_share": round(rows_common / max(1, rows_all), 4),
        "habits": dict(collections.Counter(str(r.get("habit")) for r in recs)),
        "sampled_side": dict(collections.Counter(str(r.get("sampled_side")) for r in recs)),
    }
    if recs and len(bad) > max_unreadable * len(recs):
        raise SystemExit(
            f"{path}: {len(bad)} of {len(recs)} pairs do not decompose ({bad[:5]} ...)"
        )
    return out, rep


def load_proofs(
    path: str | Path, tok: Any, limit: int = 0, field: str = "proof"
) -> tuple[list[list[Row]], dict[str, Any]]:
    """A JSONL of {prompt, proof} -> rows per proof (unreadable ones dropped and counted)."""
    recs = read_jsonl(path)
    if limit:
        recs = recs[:limit]
    out, bad = [], 0
    for r in recs:
        try:
            out.append(encode_proof(tok, r["prompt"], r[field]))
        except Unreadable:
            bad += 1
    return out, {"file": str(path), "sha256": sha256(path), "records": len(recs),
                 "proofs": len(out), "unreadable": bad}  # fmt: skip


# ============================================================================ batched forward passes
class Ctx(NamedTuple):
    device: str
    ref0: int
    max_tokens: int
    amp: bool  # bf16 autocast (training on cuda); evaluation is always fp32


def chunk_indices(lens: Sequence[int], max_tokens: int) -> list[list[int]]:
    """Length-sorted index chunks of at most max_tokens padded tokens (a longer row is a chunk of its own)."""
    order = sorted(range(len(lens)), key=lambda i: lens[i])
    out: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        if cur and (len(cur) + 1) * lens[i] > max_tokens:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def pack(rows: Sequence[Row], shifts: Sequence[int] | None, ctx: Ctx) -> tuple[Any, Any, Any]:
    """Rows -> (x [B, T] right-padded with <pad> 0, plen [B], lens [B]) on the device; names shifted per row."""
    import torch

    lens = [len(r.ids) for r in rows]
    x = torch.zeros((len(rows), max(lens)), dtype=torch.long)
    for i, r in enumerate(rows):
        x[i, : lens[i]] = torch.frombuffer(bytearray(r.ids), dtype=torch.uint8).long()
    if shifts is not None and any(shifts):
        s = torch.tensor(list(shifts), dtype=torch.long)
        x = torch.where(x >= ctx.ref0, x + s[:, None], x)
        assert int(x.max()) < ctx.ref0 + MAXN, "a name past n64 after the shift"
    plen = torch.tensor([r.plen for r in rows], dtype=torch.long)
    return x.to(ctx.device), plen.to(ctx.device), torch.tensor(lens).to(ctx.device)


def action_mask(x: Any, plen: Any, lens: Any) -> Any:
    """[B, T-1]: the positions whose next-token target is an action token or the closing <eos>."""
    import torch

    pos = torch.arange(x.shape[1] - 1, device=x.device)[None]
    return (pos >= plen[:, None] - 1) & (pos < lens[:, None] - 1)


def logp_fn(model: Any) -> Callable[[Any, Any, Any], Any]:
    """Per row: sum over the action positions of log softmax(logits)[target] (T 1, full vocabulary)."""
    import torch

    def f(x: Any, plen: Any, lens: Any) -> Any:
        logits = model(x[:, :-1])
        lp = torch.log_softmax(logits.float(), -1).gather(-1, x[:, 1:, None]).squeeze(-1)
        return torch.where(action_mask(x, plen, lens), lp, 0.0).sum(1)

    return f


def _autocast(ctx: Ctx, amp: bool) -> Any:
    import torch

    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and ctx.device == "cuda")


def apply_rows(
    fn: Callable[[Any, Any, Any], Any],
    rows: Sequence[Row],
    shifts: Sequence[int] | None,
    ctx: Ctx,
    amp: bool | None = None,
) -> Any:
    """fn on every row, chunked, no grad -> [n_rows] float32 in row order."""
    import torch

    out = torch.zeros(len(rows), device=ctx.device)
    if not rows:
        return out
    amp = ctx.amp if amp is None else amp
    with torch.no_grad(), _autocast(ctx, amp):
        for idx in chunk_indices([len(r.ids) for r in rows], ctx.max_tokens):
            sh = None if shifts is None else [shifts[i] for i in idx]
            out[torch.tensor(idx, device=ctx.device)] = fn(
                *pack([rows[i] for i in idx], sh, ctx)
            ).float()
    return out


def objective_backward(
    fn: Callable[[Any, Any, Any], Any],
    rows: Sequence[Row],
    shifts: Sequence[int] | None,
    loss_fn: Callable[[Any], Any],
    ctx: Ctx,
    linear: float | None = None,
) -> tuple[float, Any]:
    """loss_fn(per-row outputs) -> its gradient in the parameters' .grad (module docstring). -> (loss, outputs).
    `linear` = c when the loss is c x sum(outputs) (SFT): one pass, chunk by chunk, no first no-grad pass (the
    returned outputs are then None)."""
    import torch

    chunks = chunk_indices([len(r.ids) for r in rows], ctx.max_tokens)
    if linear is not None and len(chunks) > 1:
        tot = 0.0
        for idx in chunks:
            sh = None if shifts is None else [shifts[i] for i in idx]
            with _autocast(ctx, ctx.amp):
                o = fn(*pack([rows[i] for i in idx], sh, ctx)).float()
            part = linear * o.sum()
            part.backward()
            tot += float(part.detach())
        return tot, None
    if len(chunks) == 1:
        with _autocast(ctx, ctx.amp):
            out = fn(*pack(rows, shifts, ctx)).float()
        loss = loss_fn(out)
        loss.backward()
        return float(loss.detach()), out.detach()
    out = apply_rows(fn, rows, shifts, ctx)
    leaf = out.clone().requires_grad_(True)
    loss = loss_fn(leaf)
    loss.backward()
    coef = leaf.grad
    for idx in chunks:
        sh = None if shifts is None else [shifts[i] for i in idx]
        with _autocast(ctx, ctx.amp):
            o = fn(*pack([rows[i] for i in idx], sh, ctx)).float()
        (o * coef[torch.tensor(idx, device=ctx.device)]).sum().backward()
    return float(loss.detach()), out


def proof_logp(model: Any, rows: Sequence[Row], ctx: Ctx) -> float:
    """log pi(proof) at name offset 0, fp32: the sum of its rows' values."""
    return float(apply_rows(logp_fn(model), rows, None, ctx, amp=False).sum())


def flatten_pairs(
    pairs: Sequence[Pair], shifts: Sequence[int] | None = None, full: bool = False
) -> tuple[list[Row], list[int] | None, list[int], list[float]]:
    """(rows, per-row shifts, owner pair, sign +1 preferred / -1 other) over the pairs' differing rows (or all rows
    with `full`)."""
    rows: list[Row] = []
    sh: list[int] = []
    owner: list[int] = []
    sign: list[float] = []
    for j, p in enumerate(pairs):
        s = 0 if shifts is None else shifts[j]
        for rs, sg in ((p.rw, 1.0), (p.rl, -1.0)) if full else ((p.dw, 1.0), (p.dl, -1.0)):
            rows += rs
            sh += [s] * len(rs)
            owner += [j] * len(rs)
            sign += [sg] * len(rs)
    return rows, (sh if shifts is not None else None), owner, sign


def pair_sums(out: Any, owner: Sequence[int], sign: Sequence[float], n: int, device: str) -> Any:
    """Per pair: sum of sign x output over its rows (differentiable)."""
    import torch

    o = torch.tensor(list(owner), device=device, dtype=torch.long)
    s = torch.tensor(list(sign), device=device)
    return torch.zeros(n, device=device).index_add(0, o, s * out)


def pair_margins(model: Any, pairs: Sequence[Pair], ctx: Ctx, full: bool = False) -> Any:
    """log pi(w) - log pi(l) per pair at offset 0, fp32 (from the differing rows, or all rows with `full`)."""
    rows, _, owner, sign = flatten_pairs(pairs, None, full)
    out = apply_rows(logp_fn(model), rows, None, ctx, amp=False)
    return pair_sums(out, owner, sign, len(pairs), ctx.device)


# ============================================================================ evaluation
class Evaluator:
    """Held-out pair metrics and the KL to the base on fixed corpus states, both at offset 0 in fp32.

    heldout_acc_abs       share of held-out pairs with log pi(w) > log pi(l) (absolute; biased against the longer
                          member, e.g. reiteration's A form with its extra rows)
    heldout_acc_rel       share with [log pi(w) - log pi_ref(w)] - [log pi(l) - log pi_ref(l)] > 0 (relative to the
                          base: DPO's reward accuracy; 0 at step 0 by construction)
    heldout_margin_rel    mean of that relative margin (nats, not times beta); heldout_loss the DPO loss on it at beta
    heldout_logratio_abs  mean of log pi(w) - log pi(l) (nats)
    kl                    mean over action positions of KL(pi_ref || pi) = sum_v pi_ref(v) (log pi_ref(v) -
                          log pi(v)), T 1 (canary.anchor_kl's quantity without its ban mask, which is empty here)
    The base's quantities are computed once, with the base itself, before training."""

    def __init__(
        self,
        held: Sequence[Pair],
        kl_proofs: Sequence[Sequence[Row]],
        ref: Any,
        ctx: Ctx,
        beta: float,
    ) -> None:
        import torch

        self.held, self.ctx, self.beta = list(held), ctx, beta
        self.d_ref = pair_margins(ref, self.held, ctx)
        self.base_acc_abs = float((self.d_ref > 0).float().mean()) if self.held else None
        self.kl_rows = [r for p in kl_proofs for r in p]
        self.kl_chunks = chunk_indices([len(r.ids) for r in self.kl_rows], ctx.max_tokens)
        self.kl_ref: list[Any] = []
        with torch.no_grad():
            for idx in self.kl_chunks:
                self.kl_ref.append(self._act_logsoftmax(ref, idx))
        self.kl_positions = sum(int(t.shape[0]) for t in self.kl_ref)

    def _act_logsoftmax(self, model: Any, idx: Sequence[int]) -> Any:
        import torch

        x, plen, lens = pack([self.kl_rows[i] for i in idx], None, self.ctx)
        logits = model(x[:, :-1]).float()
        return torch.log_softmax(logits[action_mask(x, plen, lens)], -1)

    def pref(self, model: Any) -> dict[str, Any]:
        import torch

        if not self.held:
            return {}
        d = pair_margins(model, self.held, self.ctx)
        m = d - self.d_ref
        return {
            ACC_ABS: float((d > 0).float().mean()),
            ACC_REL: float((m > 0).float().mean()),
            "heldout_margin_rel": float(m.mean()),
            "heldout_logratio_abs": float(d.mean()),
            "heldout_loss": float(-torch.nn.functional.logsigmoid(self.beta * m).mean()),
        }

    def kl(self, model: Any) -> float | None:
        import torch

        if not self.kl_rows:
            return None
        tot = 0.0
        with torch.no_grad():
            for idx, lq in zip(self.kl_chunks, self.kl_ref, strict=True):
                lp = self._act_logsoftmax(model, idx)
                tot += float((lq.exp() * (lq - lp)).sum())
        return tot / max(1, self.kl_positions)


# ============================================================================ snapshots
def snapshot_ckpt(model: Any, meta: dict[str, Any], tuning: dict[str, Any]) -> dict[str, Any]:
    """The checkpoint dict of `model.save_ckpt` (cfg, state, tok_mode, extra) with the base's extra + tuning."""
    extra = dict(meta.get("extra") or {})
    extra["tuning"] = tuning
    state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
    assert sorted(state) == meta["state_keys"], (
        "the tuned model's state keys differ from the base's"
    )
    return {"cfg": dict(model.cfg), "state": state, "tok_mode": meta["tok_mode"], "extra": extra}


def save_snapshot(
    path: Path, model: Any, meta: dict[str, Any], base: Path, tuning: dict[str, Any]
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(snapshot_ckpt(model, meta, tuning), str(tmp))
    tmp.replace(path)
    side = base.with_suffix(".tag.json")
    if side.exists():
        doc = json.loads(side.read_text())
        doc.update(ckpt=str(path), tuned_from=str(base), tuning=tuning)
        write_json(path.with_suffix(".tag.json"), doc)


class Snapshots:
    """The snapshot rule. Steps (the default, PREREG Amendment 3: `GRID`): `snap_step<k>.pt` at each step k.
    Thresholds (only with --acc-snapshots): `snap_acc<t>.pt` at the first evaluation whose `metric` reaches t % (>=;
    step 0 = the base itself when it already does). Always `final.pt` at the end."""

    def __init__(
        self,
        out: Path,
        model_meta: dict[str, Any],
        base: Path,
        info: dict[str, Any],
        thresholds: Sequence[int] = (),
        steps: Sequence[int] = (),
        metric: str = ACC_ABS,
    ) -> None:
        self.out, self.meta, self.base, self.info, self.metric = out, model_meta, base, info, metric
        self.pending = sorted(thresholds)
        self.steps = sorted(set(steps))
        self.done: dict[str, Any] = {f"acc{t}": None for t in thresholds}
        self.done.update({f"step{k}": None for k in self.steps})

    def _write(self, name: str, step: int, metrics: dict[str, Any], model: Any) -> None:
        f = self.out / f"snap_{name}.pt" if name != "final" else self.out / "final.pt"
        tuning = {**self.info, "snapshot": name, "step": step,
                  "metrics": {k: metrics.get(k) for k in SNAP_KEYS}}  # fmt: skip
        save_snapshot(f, model, self.meta, self.base, tuning)
        self.done[name] = {"file": f.name, "step": step, "at_base": step == 0,
                           **{k: metrics.get(k) for k in SNAP_KEYS}}  # fmt: skip
        print(
            f"snapshot {f.name} at step {step} ({self.metric} {metrics.get(self.metric)})",
            flush=True,
        )

    def due_steps(self) -> set[int]:
        return {k for k in self.steps if self.done.get(f"step{k}") is None}

    def would_write(self, step: int, metrics: dict[str, Any]) -> bool:
        v = metrics.get(self.metric)
        hit = v is not None and any(v >= t / 100 for t in self.pending)
        return hit or step in self.due_steps()

    def check(self, step: int, metrics: dict[str, Any], model: Any) -> list[str]:
        wrote = []
        v = metrics.get(self.metric)
        for t in list(self.pending):
            if v is not None and v >= t / 100:
                self._write(f"acc{t}", step, metrics, model)
                self.pending.remove(t)
                wrote.append(f"acc{t}")
        if step in self.steps and self.done.get(f"step{step}") is None:
            self._write(f"step{step}", step, metrics, model)
            wrote.append(f"step{step}")
        return wrote

    def final(self, step: int, metrics: dict[str, Any], model: Any) -> None:
        self._write("final", step, metrics, model)


def ref_frozen(ref: Any, base: Path) -> bool:
    """The reference has no trainable parameter and its weights equal the base file's (raises otherwise)."""
    import torch

    assert not any(p.requires_grad for p in ref.parameters()), (
        "a trainable parameter in the reference"
    )
    saved = torch.load(str(base), map_location="cpu", weights_only=False)["state"]
    now = ref.state_dict()
    bad = [k for k in saved if not torch.equal(saved[k], now[k].detach().cpu())]
    if bad:
        raise RuntimeError(f"the reference moved during training: {bad[:5]}")
    return True


def match_steps(run_dir: str | Path) -> tuple[list[int], dict[str, Any]]:
    """The steps of a companion run's snapshots (its done.json; grid and threshold snapshots, not final.pt)."""
    doc = json.loads((Path(run_dir) / "done.json").read_text())
    snaps = doc.get("snapshots") or {}
    steps = sorted({int(v["step"]) for k, v in snaps.items() if k != "final" and v})
    return steps, {"companion": str(run_dir), "companion_trainer": doc.get("trainer"),
                   "companion_snapshots": {k: (v or {}).get("step") for k, v in snaps.items()}}  # fmt: skip


def snap_plan(a: argparse.Namespace) -> tuple[tuple[int, ...], list[int], dict[str, Any]]:
    """(thresholds, snapshot steps, notes) from the shared arguments: the grid (--snap-steps, default GRID), or a
    companion run's steps (--match-steps); thresholds only with --acc-snapshots; steps past --max-steps dropped."""
    info: dict[str, Any] = {}
    if a.match_steps:
        steps, info = match_steps(a.match_steps)
    else:
        steps = sorted({int(x) for x in str(a.snap_steps).split(",") if x.strip()})
    dropped = [k for k in steps if k > a.max_steps or k < 0]
    if dropped:
        info["dropped_past_max_steps"] = dropped
    steps = [k for k in steps if 0 <= k <= a.max_steps]
    return (THRESHOLDS if a.acc_snapshots else ()), steps, info


# ============================================================================ the loop shared by DPO and SFT
def check_out(out: Path) -> None:
    for f in ("done.json", "metrics.jsonl"):
        if (out / f).exists():
            raise SystemExit(
                f"{out / f} exists: a run is never overwritten (remove the folder to rerun)"
            )
    out.mkdir(parents=True, exist_ok=True)


def lr_at(step: int, lr: float, warmup: int) -> float:
    return lr * min(1.0, step / warmup) if warmup > 0 else lr


def train_loop(
    a: argparse.Namespace,
    policy: Any,
    step_fn: Callable[[int], dict[str, Any]],
    ev: Evaluator,
    snaps: Snapshots,
    out: Path,
) -> dict[str, Any]:
    """AdamW (RL recipe), eval, snapshots, logs. step_fn(step) runs the forward/backward of one step and returns its
    train numbers (loss, and for DPO margin and acc)."""
    import torch

    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=BETAS, weight_decay=0.0)
    policy.train()
    t0 = time.time()
    since: dict[str, list[float]] = collections.defaultdict(list)
    mf = open(out / "metrics.jsonl", "a", encoding="utf-8")  # noqa: SIM115
    tf = open(out / "train.jsonl", "a", encoding="utf-8")  # noqa: SIM115

    def evaluate(step: int, lr: float) -> dict[str, Any]:
        policy.eval()
        m = ev.pref(policy)
        kl_due = step % a.eval_every == 0 or step == a.max_steps or snaps.would_write(step, m)
        m["kl"] = ev.kl(policy) if kl_due else None
        wrote = snaps.check(step, m, policy)
        row = {"step": step, "secs": round(time.time() - t0, 2), "lr": lr,
               **{k: (sum(v) / len(v) if v else None) for k, v in since.items()}, **m,
               "snapshots": wrote}  # fmt: skip
        mf.write(json.dumps(row) + "\n")
        mf.flush()
        since.clear()
        policy.train()
        print(f"step {step:5d} " + " ".join(f"{k} {v:.4g}" for k, v in row.items()
                                            if isinstance(v, float) and k not in ("secs",)), flush=True)  # fmt: skip
        return row

    last = evaluate(0, 0.0)
    for step in range(1, a.max_steps + 1):
        lr = lr_at(step, a.lr, a.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        st = step_fn(step)
        gn = float(torch.nn.utils.clip_grad_norm_(params, CLIP))
        opt.step()
        st["grad_norm"] = gn
        tf.write(json.dumps({"step": step, "secs": round(time.time() - t0, 2), **st}) + "\n")
        tf.flush()
        for k, v in st.items():
            if isinstance(v, (int, float)):
                since[k].append(float(v))
        dense = bool(snaps.pending) and step % a.dense_every == 0
        if dense or step % a.eval_every == 0 or step == a.max_steps or step in snaps.due_steps():
            last = evaluate(step, lr)
            if not math.isfinite(last.get("loss") or 0.0):
                raise SystemExit(f"non-finite loss at step {step}")
    tf.close()
    policy.eval()
    snaps.final(a.max_steps, last, policy)
    mf.close()
    return last


def done_doc(
    a: argparse.Namespace, trainer: str, base: Path, ev: Evaluator, snaps: Snapshots, last: dict, t0: float,
    extra: dict[str, Any],
) -> dict[str, Any]:  # fmt: skip
    thr = [int(k[3:]) for k in snaps.done if k.startswith("acc")]
    return {
        "status": "done",
        "trainer": trainer,
        "base": str(base),
        "base_sha256": sha256(base),
        "out": str(Path(a.out).resolve()),
        "args": vars(a),
        "definitions": DEFINITIONS,
        "seed": {"seed": a.seed, "affects": SEED_AFFECTS},
        "base_heldout_acc_abs": ev.base_acc_abs,
        "snap_steps": snaps.steps,
        "snapshots": snaps.done,
        "acc_snapshots": bool(thr),
        "snap_metric": snaps.metric if thr else None,
        "thresholds": thr,
        "at_base": [t for t in thr if (snaps.done.get(f"acc{t}") or {}).get("step") == 0],
        "not_reached": [t for t in thr if snaps.done.get(f"acc{t}") is None],
        "final": {
            "file": "final.pt",
            "step": a.max_steps,
            **{k: last.get(k) for k in (*SNAP_KEYS, "loss")},
        },
        "kl_positions": ev.kl_positions,
        "secs": round(time.time() - t0, 1),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        **extra,
    }


def common_args(ap: argparse.ArgumentParser) -> None:
    """Arguments shared by lh_dpo.py and lh_sft.py."""
    ap.add_argument("--base", required=True, help="the Stage-1 base (also the frozen reference)")
    ap.add_argument(
        "--heldout", required=True, help="held-out pair file (the snapshot rule's pairs)"
    )
    ap.add_argument(
        "--out", required=True, help="run folder (refused if it holds done.json / metrics.jsonl)"
    )
    ap.add_argument("--lr", type=float, default=RL_LR, help="AdamW lr (the RL recipe's 1.5e-4)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument(
        "--kl-states", default=None, help="JSONL of {prompt, proof}: corpus states for the KL"
    )
    ap.add_argument("--kl-n", type=int, default=256, help="first N proofs of --kl-states")
    ap.add_argument(
        "--warmup", type=int, default=WARMUP, help="linear lr warmup over the first N steps"
    )
    ap.add_argument("--eval-every", type=int, default=25,
                    help="held-out eval and KL every N steps (and at every snapshot step)")  # fmt: skip
    ap.add_argument("--snap-steps", default=",".join(map(str, GRID)),
                    help="snapshot steps, comma list (default: PREREG Amendment 3's grid)")  # fmt: skip
    ap.add_argument("--match-steps", default=None,
                    help="snapshot at a companion run's snapshot steps (its done.json) instead of --snap-steps")  # fmt: skip
    ap.add_argument("--acc-snapshots", action="store_true",
                    help="also snap_acc60/75/90.pt at the first evaluation reaching 60/75/90 %% (off by default)")  # fmt: skip
    ap.add_argument("--snap-metric", default=ACC_ABS, choices=(ACC_ABS, ACC_REL),
                    help="the accuracy --acc-snapshots uses")  # fmt: skip
    ap.add_argument("--dense-every", type=int, default=1,
                    help="with --acc-snapshots: held-out eval every N steps while a threshold is pending")  # fmt: skip
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="padded tokens per forward chunk (default 65536 cuda, 16384 cpu)")  # fmt: skip
    ap.add_argument("--no-shift", action="store_true", help="no random name offset in training")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (at most 8)")


def make_ctx(a: argparse.Namespace, tok: Any) -> Ctx:
    mt = a.max_tokens or (65536 if a.device == "cuda" else 16384)
    return Ctx(a.device, tok.ref0, mt, a.device == "cuda")


# ============================================================================ toy data (smokes and tests)
def tiny_base(
    path: str | Path, seed: int = 0, dims: tuple[int, int, int, int] = (2, 64, 4, 128)
) -> Path:
    """A tiny random `lean_staten_tag` ALiBi checkpoint in the Stage-1 format (+ .tag.json), for CPU tests."""
    setup("cpu")
    import torch
    from best_model import ALiBiGPT  # type: ignore[import-not-found]
    from tokenizer import make_tokenizer  # type: ignore[import-not-found]

    tok = make_tokenizer(T.MODE)
    torch.manual_seed(seed)
    n_layer, d, n_head, d_ff = dims
    m = ALiBiGPT(tok.vocab_size, n_layer, d, n_head, d_ff=d_ff)
    at = {"mode": T.MODE, "tag_position": "before <act>", "tokens": T.TAG_TOKENS, "corpus": "tiny-test",
          "variant": None}  # fmt: skip
    extra = {"recipe": "best", "steps": 0, "author_tags": at, "tiny_test_base": True}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": m.cfg, "state": m.state_dict(), "tok_mode": T.MODE, "extra": extra}, str(p))
    write_json(p.with_suffix(".tag.json"), {"train_tokens": 0, "author_tags": at, "ckpt": str(p)})
    return p


def toy_records(
    n: int,
    seed: int = 0,
    corpus: Path | None = None,
    scan: int = 20000,
    forms: tuple[str, str] = ("A", "B"),
) -> list[dict[str, Any]]:
    """Toy pairs from an l19-style corpus: preferred = the record's proof without notes (`ta_proof`) with author A's
    note form for every premise class, other = with B's (notes.add_notes, the corpus table). Records with at least
    one premise, both versions decomposing; seeded choice among the first `scan` such records. `forms` = two menu
    form names instead of ("A", "B") puts that form on every premise (("orF", "For"): twins of equal length)."""
    import notes as N

    T.install_dan()
    from state_env import decompose  # type: ignore[import-not-found]

    tab = N.load_table()
    table = {c: {s: (tab[c][f] if f in ("A", "B") else f) for s, f in zip("AB", forms, strict=True)}
             for c in tab}  # fmt: skip
    corpus = corpus or LH / "data" / "l19.jsonl.gz"
    cand = []
    with gzip.open(corpus, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("n_prem", 0) >= 1:
                cand.append(r)
                if len(cand) >= scan:
                    break
    random.Random(seed).shuffle(cand)
    out: list[dict[str, Any]] = []
    for r in cand:
        try:
            w = N.add_notes(r["prompt"], r["ta_proof"], {c: table[c]["A"] for c in table})
            o = N.add_notes(r["prompt"], r["ta_proof"], {c: table[c]["B"] for c in table})
            for pf in (w, o):
                decompose(r["prompt"], pf, canon=True)
        except Exception:  # a record the toy builder cannot use
            continue
        if w == o:
            continue
        out.append({"key": r["key"], "prompt": r["prompt"], "preferred": w, "other": o, "habit": "notes_toy",
                    "sampled_side": r.get("author"), "corpus_proof": r["proof"]})  # fmt: skip
        if len(out) >= n:
            break
    return out


def twin_records(
    n: int, seed: int = 0, habit: str = "reit", field: str = "ta_proof", corpus: Path | None = None,
    scan: int = 20000,
) -> list[dict[str, Any]]:  # fmt: skip
    """Rater pairs as SPEC2 builds them, from corpus proofs instead of samples: a record's proof (`field`; `ta_proof`
    is the no-notes proof the Nov 14 bases trained on) whose `habit` sites are all in one form, and its twin
    (lh_twin.twin: that habit flipped at every site); the A-form member is preferred, `sampled_side` is the record's
    form. Seeded choice among the first `scan` records."""
    import lh_profile as LP
    import lh_twin as TW

    try:
        tables = LP.load_tables()
    except (
        Exception
    ):  # no habit table yet: the provisional one (A = form 1 for reiteration, as SPEC2)
        tables = LP.provisional_tables()
    corpus = corpus or LH / "data" / "l19.jsonl.gz"
    cand = []
    with gzip.open(corpus, "rt", encoding="utf-8") as f:
        for line in f:
            cand.append(json.loads(line))
            if len(cand) >= scan:
                break
    random.Random(seed).shuffle(cand)
    out: list[dict[str, Any]] = []
    for r in cand:
        pf = r[field]
        h = LP.profile(r["prompt"], pf, tables)["habits"].get(habit) or {}
        if not h.get("sites") or h["sites"] not in (h["a"], h["b"]):
            continue
        tw = TW.twin(r["prompt"], pf, habit, tables)
        if tw is None or tw == pf:
            continue
        side = "A" if h["a"] == h["sites"] else "B"
        w, o = (pf, tw) if side == "A" else (tw, pf)
        out.append({"key": r["key"], "prompt": r["prompt"], "preferred": w, "other": o, "habit": habit,
                    "sampled_side": side, "corpus_proof": r[field]})  # fmt: skip
        if len(out) >= n:
            break
    return out


def write_toy(
    out: Path, n: int, heldout: int, kl: int, seed: int, habit: str | None = None
) -> dict[str, Any]:
    """pairs_train.jsonl, pairs_held.jsonl (split by theorem), narrow.jsonl (the preferred proofs of the train
    pairs), kl_states.jsonl (corpus proofs of other records). `habit`: twins of that habit (`twin_records`) instead
    of the A-versus-B note toy."""
    recs = (
        toy_records(n + heldout + kl, seed)
        if habit is None
        else twin_records(n + heldout + kl, seed, habit)
    )
    held, train, rest = recs[:heldout], recs[heldout : heldout + n], recs[heldout + n :]
    strip = ("corpus_proof",)
    write_jsonl(
        out / "pairs_train.jsonl", [{k: v for k, v in r.items() if k not in strip} for r in train]
    )
    write_jsonl(
        out / "pairs_held.jsonl", [{k: v for k, v in r.items() if k not in strip} for r in held]
    )
    write_jsonl(out / "narrow.jsonl", [{"key": r["key"], "prompt": r["prompt"], "proof": r["preferred"]}
                                       for r in train])  # fmt: skip
    # the fine-tune arm's two files (PREREG Amendment 3 point 5): preferred members, and the sampled members
    write_jsonl(out / "sft_pref.jsonl", [{"key": r["key"], "prompt": r["prompt"], "proof": r["preferred"]}
                                         for r in train])  # fmt: skip
    write_jsonl(out / "sft_sampled.jsonl", [
        {"key": r["key"], "prompt": r["prompt"], "proof": r["preferred"] if r["sampled_side"] == "A" else r["other"]}
        for r in train])  # fmt: skip
    write_jsonl(out / "kl_states.jsonl", [{"key": r["key"], "prompt": r["prompt"], "proof": r["corpus_proof"]}
                                          for r in rest])  # fmt: skip
    doc = {"train": len(train), "heldout": len(held), "kl": len(rest), "seed": seed, "out": str(out),
           "habit": habit or "notes_toy"}  # fmt: skip
    print(json.dumps(doc))
    return doc


def write_probe(
    out: Path, habits: Sequence[str], n: int, seed: int, corpus: Path, field: str = "proof"
) -> dict[str, Any]:
    """A reward-model probe file (lh_rm.py probe): for each habit, up to n twin pairs from corpus records
    (`twin_records`: preferred = the A-form member), `habit` set per row."""
    rows: list[dict[str, Any]] = []
    counts = {}
    for h in habits:
        recs = twin_records(n, seed, h, field, corpus)
        counts[h] = len(recs)
        rows += [{k: v for k, v in r.items() if k != "corpus_proof"} for r in recs]
    write_jsonl(out, rows)
    doc = {"out": str(out), "corpus": str(corpus), "field": field, "per_habit": counts}
    print(json.dumps(doc))
    return doc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("toy", help="toy pairs from l19 for smokes")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=160)
    p.add_argument("--heldout", type=int, default=40)
    p.add_argument("--kl", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--habit", default=None, help="twins of this habit (lh_twin) instead of the note toy"
    )
    p = sp.add_parser(
        "probe-toy", help="a reward-model probe file of twins of other habits (smokes)"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--habits", default="note_atom,note_vcc,conj,disjunct,halves,restate")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corpus", default=str(LH / "data" / "c24.jsonl.gz"))
    p.add_argument("--field", default="proof")
    a = ap.parse_args(argv)
    if a.cmd == "toy":
        write_toy(Path(a.out), a.n, a.heldout, a.kl, a.seed, a.habit)
    else:
        write_probe(Path(a.out), a.habits.split(","), a.n, a.seed, Path(a.corpus), a.field)


if __name__ == "__main__":
    main()
