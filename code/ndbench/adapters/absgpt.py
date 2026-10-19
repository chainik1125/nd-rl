"""`AbsGptAdapter`: the `ModelAdapter` for the frozen absgpt decoder-only GPT (plan §1/§3 WP1).

Built the way `/scratch/cowifornia/nd-rl/ab3/lengthexp/evaluate.py:load_model` does: default arch
config when the checkpoint carries none, a `_orig_mod.`/`module.` (compile/DDP) prefix stripped
from every state-dict key, `strict=True` load. Deliberately does **not** call the vendored
`vendor/absgpt/model.py:load_ckpt` (plan §3 WP1's explicit instruction) -- that function does its
own bare `from tokenizer import Tokenizer` import, which is exactly what `vendor/__init__.py`'s
`load_absgpt_module` architecture is designed to avoid needing.

Sampling uses the vendored *compacting* sampler's `_gen` (`vendor/sampler.py`, ~1.35x faster than
`vendor/absgpt/sample.py:generate`) directly on explicit token-id lists, not `sampler.sample_ids`
(which only accepts prompt strings and cannot express a forced prefix or a forced start index).

**Cross-request batching (throughput fix).** `sample()` used to call `_gen` once per
`SampleRequest` (`n` copies of that one request's input ids per call), which meant a `k=32` run
made one 32-row GPU call per problem -- far below the 512-1024-row batches `_gen` can actually
take (`self._batch`, default 1024; `vendor/sampler.py:sample_ids` uses 768) -- and the dominant
cause of a ~10x throughput regression versus the old lengthexp
evaluator's compacting-sampler runs. `sample()` now flattens *every* request's rows (request `r`
contributes `r.n` rows of its own prompt-plus-forced-prefix ids) into one pool across the whole
`reqs` list, groups them by `(max_new_tokens, temperature)` (the two settings one `_gen` call must
share across its entire batch), sorts each group's rows by input length (matching
`sampler.sample_ids`'s own discipline, so a chunk pads to a similar length rather than the longest
and shortest row in the whole call together), and chunks to `self._batch` rows per `_gen` call --
the same chunking discipline `sampler.sample_ids` uses for its (differently-shaped) multi-prompt
batches, just spanning every request in the call rather than one. Continuations are mapped back to
their originating `(request, draw index)` afterwards; every existing per-request semantic (forced
prefixes, `options.start_index`, per-request `max_new_tokens`/temperature, `Attempt` fields, N1
renumbering) is unchanged. See `sample()`'s own docstring for the exact grouping/seeding order and
its determinism contract.

Start-index handling (plan §3 WP1, and see `ab3/lengthexp/evaluate.py`/`vendor/ndcheck.py`'s
`start`/`index_overflow` handling, which is what a caller needs `Verdict.start` for downstream):
the `abs` tokenizer's training-time random-shift augmentation means a freshly sampled proof can
legitimately start at any `N<k>`, so `sample()` always decodes the *raw* model output into
`Attempt.native` first (preserving the original start index verbatim) and then canonicalises
`Attempt.body` to start at `N1` via `core.proof.renumber` (a no-op if it already does).
`options={"start_index": 1}` instead *forces* the very first generated token to be the `N1` index
token (by prepending that one synthetic id to the input, exactly like a one-token forced prefix),
which avoids the N64 overflow entirely for a long proof (plan §2 B2) rather than merely relabelling
it after the fact. A `SampleRequest.prefix` takes priority over `start_index` when both are given.

`logprob`/`Learner.nll` marginalise over every start-shift encoding by log-sum-exp, matching
`/scratch/cowifornia/nd-rl/external/nd-takehome/novelty.py`'s `score()` (T=1, proof tokens
including `QED`, conditioned on the prompt); `Learner.update` instead trains on a small number of
*randomly* shifted copies per example (cheap Monte-Carlo estimate of the same objective, the same
way the checkpoints being evaluated were themselves trained with `Tokenizer.shift_abs`).

Interp hooks (plan §A3): `activations` captures the residual stream after each requested block via
a `register_forward_hook`; `with_steering` returns a new adapter (over a deep-copied model, so the
original adapter's forward passes are never affected) whose blocks carry additional hooks that add
`alpha * vector` to the residual stream at the token positions `where` selects. Since a `_gen`
generation step only ever sees the newly produced tokens (the rest lives in the KV cache), the
hook recovers each call's absolute token positions from that block's own forward arguments rather
than needing `_gen`'s otherwise-private `pos` tensor directly: `_gen` always left-pads its batch
(`vendor/sampler.py`'s `idx`/`keep`), and cross-request batching (see above) means two rows in the
*same* `_gen` call can now legitimately have different real prompt lengths and therefore different
amounts of left padding, so one shared, batch-wide position count is no longer correct -- the hook
instead reads each row's own `keep` vector back out of the block's `mask` argument (`mask[:, 0, -1,
:]` is exactly `keep` at every step: at prefill this is `causal`'s all-True last row AND `keep`, and
`torch.eye`'s OR only ever touches that row's own diagonal entry, which is always already kept; at
each incremental decode step `mask` *is* `keep`, with no causal component at all), takes its
cumulative count as each row's real position, and keeps only the last `T` entries (this call's own
newly-appended columns) -- see `_make_steering_hook`. The non-cached forward passes used by
`logprob`/`nll`/`activations`/`Learner.update` have no such per-row padding *offset* (no cache, no
mask; every row's real content starts at column 0, right-padded) and so always use one `arange(T)`
column-index-as-position range shared by every row -- but the *threshold* that position range is
compared against (`self._steer_prompt_len`) can itself be per-row (a `torch.Tensor`, not a plain
Python `int`) when a batch mixes rows from pairs of different prompt lengths, which is exactly what
`_batched_shift_marginal_logprobs` below does; see `_make_steering_hook`'s own docstring. Either
way, "prompt" vs. "proof" is decided against `self._steer_prompt_len`, an attribute the calling
method sets just before invoking the model and clears just after. `sample()` sets it to a per-row
tensor of each row's prompt length (excluding any forced prefix, whose tokens are proof tokens) for
every `_gen` chunk (audit 2026-10-01: it used to leave it at 0, so `where="proof"` steered the
prompt too and `where="prompt"` did nothing during generation).

**Batched log-probability (throughput fix).** `logprob`/`Learner.nll`/`nll_start_excluded` used to
call `_shift_marginal_logprob_reference` once per `(sequent, body)` pair, itself a loop making one
batch-size-1 forward call per start shift (~55 for a 10-line proof) -- EDL alone made roughly 3.7M
such forwards across its 144 runs, and the interp likelihood makes one `logprob` call per ordering.
`_batched_shift_marginal_logprobs`/`_batched_shift_marginal_nlls` instead flatten every shift of
every pair in one call into a single pool of rows, right-pad and chunk them (sorted by length
first) to a few thousand rows per forward call, and log-sum-exp each pair's own rows back together
-- the same flatten/sort/chunk discipline `sample`'s cross-request batching and `AbsGptAdapter.
activations` (also now batched the same way) use, just for a plain forward pass instead of `_gen`.
`_shift_marginal_logprob_reference`/`_shift_marginal_nll_reference` (the original one-row-per-call
implementations) are kept, byte-for-byte, purely as the correctness oracle `test_wp1_absgpt.py`
checks the batched versions against.
"""

from __future__ import annotations

import copy
import importlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..core.io import sha256_file, sha256_str
from ..core.proof import renumber
from ..core.types import JSON, Attempt, LogProb, SampleRequest, Sequent
from ..vendor import ensure_takehome_on_path, load_absgpt_module
from .base import Capability, ModelAdapter, SteeringSpec

# ---------------------------------------------------------------- vendor loading (module-load time)

_MODEL_MOD = load_absgpt_module("model")
_TOK_MOD = load_absgpt_module("tokenizer")
GPT = _MODEL_MOD.GPT
Tokenizer = _TOK_MOD.Tokenizer
MAXN: int = _TOK_MOD.MAXN

# `vendor/sampler.py` (the lengthexp "compacting" sampler) lives directly under `vendor/`, not
# under `vendor/absgpt/`, and keeps its own bare imports (`import torch, ...`; no bare
# `from model import ...`/`from tokenizer import ...`), so it is loaded via the plain
# `ensure_takehome_on_path` + `importlib` shim (strategy 1 of `vendor/README.md#loading`), the same
# way `ndcheck.py`/`common.py` are -- not via `load_absgpt_module`'s per-module namespacing (that
# strategy exists specifically for `model.py`/`tokenizer.py`'s generic names).
ensure_takehome_on_path()
_SAMPLER_MOD = importlib.import_module("sampler")
_GEN = _SAMPLER_MOD._gen


def _cross_request_seed(reqs: Sequence[SampleRequest]) -> int:
    """One `sample()` call's `torch.Generator` seed: a hash of every request's own `seed` field,
    taken in the exact order `reqs` is given (see `AbsGptAdapter.sample`'s docstring). Uses the
    first 8 bytes of a sha256 digest of the colon-joined seeds, the same width/derivation
    `runner.runner._item_seed` uses for its own per-(item, chunk) seeds."""
    payload = ":".join(str(req.seed) for req in reqs)
    return int(sha256_str(payload)[:16], 16)


def _build_model_and_tokenizer(
    ckpt_path: str | Path, device: torch.device
) -> tuple[Any, Any, dict[str, int]]:
    """Build `(model, tokenizer, arch_cfg)` from a checkpoint the way `evaluate.py:load_model`
    does (see module docstring): default arch config, `_orig_mod.`/`module.` prefix stripped,
    `strict=True`. Never calls the vendored `model.py:load_ckpt`."""
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    tok = Tokenizer(ck.get("tok_mode", "abs"))
    cfg = dict(ck.get("cfg") or dict(vocab=tok.vocab_size, n_layer=4, d=256, n_head=8))
    cfg.setdefault("vocab", tok.vocab_size)
    model = GPT(**cfg)
    state = {
        k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in ck["state"].items()
    }
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, tok, cfg


#: `_batched_shift_marginal_logprobs`/`activations` chunk size (throughput fix: EDL's ~3.7M
#: batch-size-1 forward calls, and the interp likelihood's one `logprob` call per ordering, both
#: used to run one row at a time). ~4096 rows keeps a chunk's activation memory reasonable even for
#: the RL checkpoint's longer prompts/proofs while still turning many small forwards into a few
#: large ones. Paired with `_LOGPROB_MAX_BATCH_CELLS` (see `_length_bucketed_chunks`): a row-count
#: cap *alone* degenerates to a single chunk, padded to the single longest row in the whole call,
#: whenever the total row count already fits under it -- for a small, length-varied `pairs` list
#: (observed: 50 real proofs, lengths 63-396 tokens) that padded a typical row by >2x on average
#: and made the "batched" call slower, wall-clock, than the one-row-at-a-time reference on CPU.
_LOGPROB_MAX_BATCH_ROWS = 4096

#: Second cap on a `_length_bucketed_chunks` chunk: `rows * that chunk's own max length` (its total
#: padded cell count), independent of how many total rows there are. `4096 * 64`: a chunk still
#: gets the full `_LOGPROB_MAX_BATCH_ROWS` rows at a "typical" short proof's length (a few dozen to
#: a couple hundred tokens, per `runner.runner.MEAN_TOKENS_PER_REF_LINE`'s own measurement), and
#: shrinks automatically for longer rows -- bounding padding waste regardless of dataset size.
_LOGPROB_MAX_BATCH_CELLS = 4096 * 64


def _length_bucketed_chunks(
    order: Sequence[int], lengths: Sequence[int], max_rows: int, max_cells: int
) -> list[list[int]]:
    """Group `order` (a permutation of `range(len(lengths))`, expected sorted ascending by
    `lengths`) into chunks for batched, right-padded forward calls: a chunk never exceeds
    `max_rows` rows, nor `max_cells` = its row count times its own rows' max length (its total
    padded cell count once right-padded to one shared width). Both caps matter independently: the
    first bounds dispatch/call overhead (too many tiny chunks), the second bounds padding waste
    (too few huge chunks spanning a wide length range) -- a row-count cap *alone* degenerates to a
    single, maximally-padded chunk whenever the whole `order` already fits under it, regardless of
    how widely `lengths` varies (`_LOGPROB_MAX_BATCH_ROWS`'s own docstring). Sorting `order` by
    length first (as every caller does) means each chunk's own rows are already close in length to
    each other, so the `max_cells` cut naturally falls near where lengths start spreading apart
    rather than at an arbitrary row-count boundary.
    """
    chunks: list[list[int]] = []
    cur: list[int] = []
    cur_max_len = 0
    for idx in order:
        length = lengths[idx]
        candidate_max = max(cur_max_len, length)
        if cur and (len(cur) + 1 > max_rows or (len(cur) + 1) * candidate_max > max_cells):
            chunks.append(cur)
            cur = []
            candidate_max = length
        cur.append(idx)
        cur_max_len = candidate_max
    if cur:
        chunks.append(cur)
    return chunks


def _shift_marginal_logprob_reference(
    model: Any,
    tok: Any,
    device: torch.device,
    seq: Sequent,
    body: str,
    owner: Any,
    maxn: int,
    *,
    start_excluded: bool = False,
) -> LogProb:
    """`log sum_s p(shift_s(body) | prompt)` at temperature 1, over every shift `s` the `abs`
    tokenizer's `MAXN` allows (matches `external/nd-takehome/novelty.py`'s `score()`) -- one
    forward call per shift, at batch size 1. Superseded by `_batched_shift_marginal_logprobs`
    (throughput fix: this one-row-per-call loop is what made EDL/the interp likelihood so slow);
    kept byte-for-byte as it was, unused by any adapter method, purely as the correctness oracle
    `test_wp1_absgpt.py` checks the batched path against (to within 1e-4 nats).

    Returns a log-*probability* (`LogProb.nats` <= 0, typically); `_shift_marginal_nll_reference`
    negates it to get a positive NLL (see `core.types.LogProb`'s docstring for the sign convention
    every caller relies on). `owner._steer_prompt_len` is set/cleared around each forward call so a
    steered adapter's hooks (if any) see the right prompt/proof split; unused (and harmless)
    otherwise.

    `start_excluded`: drop the very first proof token's log-probability from every shift's summed
    total before marginalising (plan §B7's start-token-excluded sensitivity variant; matches
    `test_wp5_fixtures.score_all_shifts(..., start_excluded=True)`), and report `n_tokens` one
    shorter to match.
    """
    pid = tok.encode_prompt(seq.prompt())
    qid = tok.encode_proof(body)
    ref0 = tok.ref0
    mx = max((x - ref0 + 1 for x in qid if x >= ref0), default=0)
    max_shift = max(maxn - mx, 0)
    per_shift: list[float] = []
    model.eval()
    mask_start = len(pid) if start_excluded else len(pid) - 1
    with torch.no_grad():
        for s in range(0, max_shift + 1):
            ids = pid + [x + s if x >= ref0 else x for x in qid]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            owner._steer_prompt_len = len(pid)
            try:
                logits = model(x[:, :-1]).float()
            finally:
                owner._steer_prompt_len = 0
            tgt = x[:, 1:]
            logp = F.log_softmax(logits, dim=-1).gather(-1, tgt[..., None]).squeeze(-1)
            mask = torch.zeros_like(tgt, dtype=torch.bool)
            mask[:, mask_start:] = True
            per_shift.append(float((logp * mask).sum().item()))
    nats = (
        float(torch.logsumexp(torch.tensor(per_shift), dim=0).item())
        if per_shift
        else float("-inf")
    )
    n_tokens = len(qid) - 1 if start_excluded else len(qid)
    return LogProb(nats=nats, n_tokens=n_tokens, per_encoding=tuple(per_shift))


def _shift_marginal_nll_reference(
    model: Any,
    tok: Any,
    device: torch.device,
    seq: Sequent,
    body: str,
    owner: Any,
    maxn: int,
    *,
    start_excluded: bool = False,
) -> LogProb:
    """`-log sum_s p(shift_s(body) | prompt)`: the positive-NLL counterpart of
    `_shift_marginal_logprob_reference`, kept for the same reason (correctness oracle, unused by
    any adapter method). Only `.nats` is negated -- `.per_encoding` stays the per-shift
    log-*probabilities* `_shift_marginal_logprob_reference` computed, matching
    `test_wp5_fixtures.TinyGptLearner.nll`'s convention."""
    lp = _shift_marginal_logprob_reference(
        model, tok, device, seq, body, owner, maxn, start_excluded=start_excluded
    )
    return LogProb(nats=-lp.nats, n_tokens=lp.n_tokens, per_encoding=lp.per_encoding)


def _batched_shift_marginal_logprobs(
    model: Any,
    tok: Any,
    device: torch.device,
    owner: Any,
    maxn: int,
    pairs: Sequence[tuple[Sequent, str]],
    *,
    start_excluded: bool = False,
    max_batch_rows: int = _LOGPROB_MAX_BATCH_ROWS,
    max_batch_cells: int = _LOGPROB_MAX_BATCH_CELLS,
) -> list[LogProb]:
    """Batched replacement for calling `_shift_marginal_logprob_reference` once per pair
    (throughput fix: EDL made ~3.7M such batch-size-1 forwards across its 144 runs, and the interp
    likelihood makes one `logprob` call per ordering). Every `(pair, shift)` row across *every*
    pair in `pairs` is flattened into one pool, sorted by length, and grouped into chunks via
    `_length_bucketed_chunks` (bucketed both by row count, `max_batch_rows`, and by total padded
    cell count, `max_batch_cells` -- see that function's docstring for why the row-count cap alone
    is not enough) -- the same sort-then-chunk discipline `AbsGptAdapter.sample` uses for
    cross-request batching, extended with the cell cap this workload's much wider pair-to-pair
    length variation needs.

    Each row's summed log-probability over its own real (non-padding) proof-token range is
    computed with a vectorised, per-row-length-aware mask: rows are right-padded with `tok.pad`,
    and since this is a plain (non-cached) forward pass, causal attention alone means a row's real
    positions are never affected by its own trailing padding or by any other row sharing its
    batch -- so the padded tail's otherwise-unused values are simply excluded by the mask, safe to
    ignore even though never computed to mean anything (no NaN risk: `tok.pad` is an ordinary,
    embeddable token id, not uninitialised memory). Rows are then grouped back by their originating
    pair and log-sum-exp'd over that pair's own shifts, in ascending-shift order (matching
    `LogProb.per_encoding`'s pre-existing order) -- `row_ids` (and therefore `per_row_logp`, indexed
    by the same *original*, pre-sort position) is built pair-major/shift-ascending in the first
    place, so iterating it in that original order to regroup by pair recovers shift order directly,
    independent of whatever order chunking processed rows in.

    Numerically identical to `_shift_marginal_logprob_reference` (the same per-row operations,
    just batched, modulo ordinary floating-point batching noise); `test_wp1_absgpt.py` checks
    agreement to within 1e-4 nats on the fixture proofs.

    **Steering (plan §A3).** `owner._steer_prompt_len` is set to a *per-row* `torch.Tensor` of each
    row's own `len(pid)` for the duration of each chunk's forward call, rather than a single shared
    scalar: rows belonging to the same pair always share one prompt length (a shift only moves
    proof-token line/citation indices, never the prompt), but one chunk can hold rows from several
    different pairs, which generally do *not* share a prompt length. See `_make_steering_hook`'s
    handling of a tensor-valued `_steer_prompt_len` (it broadcasts per row instead of over the
    whole batch).
    """
    if not pairs:
        return []

    row_ids: list[list[int]] = []
    row_pair_idx: list[int] = []
    row_prompt_len: list[int] = []
    row_mask_start: list[int] = []
    pair_qid_len: list[int] = [0] * len(pairs)

    for pair_idx, (seq, body) in enumerate(pairs):
        pid = tok.encode_prompt(seq.prompt())
        qid = tok.encode_proof(body)
        pair_qid_len[pair_idx] = len(qid)
        ref0 = tok.ref0
        mx = max((x - ref0 + 1 for x in qid if x >= ref0), default=0)
        max_shift = max(maxn - mx, 0)
        mask_start = len(pid) if start_excluded else len(pid) - 1
        for s in range(0, max_shift + 1):
            ids = pid + [x + s if x >= ref0 else x for x in qid]
            row_ids.append(ids)
            row_pair_idx.append(pair_idx)
            row_prompt_len.append(len(pid))
            row_mask_start.append(mask_start)

    n_rows = len(row_ids)
    per_row_logp: list[float] = [0.0] * n_rows
    row_lens = [len(r) for r in row_ids]
    order = sorted(range(n_rows), key=lambda i: row_lens[i])
    chunks = _length_bucketed_chunks(order, row_lens, max_batch_rows, max_batch_cells)

    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            chunk_ids = [row_ids[i] for i in chunk]
            b = len(chunk_ids)
            length = max(len(r) for r in chunk_ids)
            x = torch.full((b, length), tok.pad, dtype=torch.long, device=device)
            for bi, r in enumerate(chunk_ids):
                x[bi, : len(r)] = torch.tensor(r, device=device)

            lengths = torch.tensor([len(row_ids[i]) for i in chunk], device=device)
            mask_starts = torch.tensor([row_mask_start[i] for i in chunk], device=device)
            prompt_lens = torch.tensor(
                [row_prompt_len[i] for i in chunk], dtype=torch.long, device=device
            )

            owner._steer_prompt_len = prompt_lens
            try:
                logits = model(x[:, :-1]).float()
            finally:
                owner._steer_prompt_len = 0

            tgt = x[:, 1:]
            logp = F.log_softmax(logits, dim=-1).gather(-1, tgt[..., None]).squeeze(-1)
            col = torch.arange(logp.shape[1], device=device).unsqueeze(0)
            valid_end = (lengths - 1).unsqueeze(1)
            row_mask = (col >= mask_starts.unsqueeze(1)) & (col < valid_end)
            row_sums = (logp * row_mask).sum(dim=1)
            for local_i, global_i in enumerate(chunk):
                per_row_logp[global_i] = float(row_sums[local_i].item())

    per_pair_shifts: list[list[float]] = [[] for _ in pairs]
    for global_i in range(n_rows):
        per_pair_shifts[row_pair_idx[global_i]].append(per_row_logp[global_i])

    results: list[LogProb] = []
    for pair_idx, per_shift in enumerate(per_pair_shifts):
        nats = (
            float(torch.logsumexp(torch.tensor(per_shift), dim=0).item())
            if per_shift
            else float("-inf")
        )
        n_tokens = pair_qid_len[pair_idx] - 1 if start_excluded else pair_qid_len[pair_idx]
        results.append(LogProb(nats=nats, n_tokens=n_tokens, per_encoding=tuple(per_shift)))
    return results


def _batched_shift_marginal_nlls(
    model: Any,
    tok: Any,
    device: torch.device,
    owner: Any,
    maxn: int,
    pairs: Sequence[tuple[Sequent, str]],
    *,
    start_excluded: bool = False,
    max_batch_rows: int = _LOGPROB_MAX_BATCH_ROWS,
    max_batch_cells: int = _LOGPROB_MAX_BATCH_CELLS,
) -> list[LogProb]:
    """Batched positive-NLL counterpart of `_batched_shift_marginal_logprobs`, for
    `_AbsGptLearner.nll`/`nll_start_excluded` (`Learner.nll`'s contract is a negative
    log-likelihood, the opposite sign from `ModelAdapter.logprob`; see `core.types.LogProb`). Only
    `.nats` is negated -- `.per_encoding` stays the per-shift log-*probabilities*."""
    lps = _batched_shift_marginal_logprobs(
        model,
        tok,
        device,
        owner,
        maxn,
        pairs,
        start_excluded=start_excluded,
        max_batch_rows=max_batch_rows,
        max_batch_cells=max_batch_cells,
    )
    return [
        LogProb(nats=-lp.nats, n_tokens=lp.n_tokens, per_encoding=lp.per_encoding) for lp in lps
    ]


def _make_capture_hook(captured: dict[int, torch.Tensor], layer: int) -> Any:
    """A `register_forward_hook` callback that stores `output` into `captured[layer]`. A plain
    function factory (rather than a closure defined inline inside the `for layer in layers` loop
    in `AbsGptAdapter.activations`) so `layer`/`captured` are bound by ordinary argument passing,
    not by closing over a loop variable."""

    def _hook(module: Any, inputs: Any, output: torch.Tensor) -> None:
        captured[layer] = output.detach()

    return _hook


def _resolve_steering_vector(vector: tuple[float, ...] | str) -> list[float]:
    if isinstance(vector, str):
        arr = np.load(vector)
        return [float(v) for v in arr.tolist()]
    return list(vector)


def _steering_spec_hash(spec: SteeringSpec) -> str:
    """A stable fingerprint of `spec`, included in a steered adapter's `identity()` (plan §A3)."""
    parts = []
    for t in spec:
        if isinstance(t.vector, str):
            vec_repr = f"path:{t.vector}"
        else:
            vec_repr = ",".join(f"{v:.10g}" for v in t.vector)
        parts.append(f"{t.layer}|{vec_repr}|{t.alpha:.10g}|{t.where}")
    return sha256_str("||".join(parts))


@dataclass(frozen=True)
class MaskedLogProbs:
    """`DanStateAdapter.masked_next_logprobs`'s exact analogue for `AbsGptAdapter` (duplicated
    rather than imported -- module docstring's "kept ... purely as the correctness oracle"/
    duplication convention this file already follows for `_cross_request_seed`/
    `_length_bucketed_chunks`, so `interp/kl.py` never makes this module depend on `adapters.
    danstate`'s unrelated vendor tree): full next-token log-probabilities at every scored
    (proof-token) position of one teacher-forced `(sequent, native proof text)` pair.

    No shift marginalisation is needed here, unlike Dan's per-step name base: `native` already
    fixes a literal, unambiguous encoding -- `AbsGptAdapter.sample`'s own docstring says
    `Attempt.native` is the raw decoded text at whatever absolute `N<k>` indices the model
    actually sampled, so teacher-forcing on `native` directly reproduces the exact row
    `_batched_shift_marginal_logprobs` would score at that sample's own (unknown to the caller,
    and not needed) shift.

    `step_of[k]` is which *line* (0-based, `core.proof.split_lines`'s own order) scored position
    `k`'s token belongs to, or `n_lines` for the final `QED` token itself -- a pseudo "FINAL" step,
    mirroring `DanStateAdapter`'s own proof-closing-step convention; `n_steps = n_lines + 1`."""

    log_probs: np.ndarray  # (n_scored, vocab), float32
    target_ids: np.ndarray  # (n_scored,), int64
    step_of: np.ndarray  # (n_scored,), int64
    n_steps: int
    n_tokens: int


class AbsGptAdapter(ModelAdapter):
    """`ModelAdapter` for the frozen absgpt GPT: SAMPLE, PREFIX, LOGPROB, FINETUNE, ACTIVATIONS,
    STEER (plan §1/§3 WP1, §A3)."""

    def __init__(
        self,
        ckpt: str | Path,
        device: str = "cpu",
        batch: int = 1024,
        *,
        label: str | None = None,
        compact_every: int = 16,
        compact_below: float = 0.8,
        update_shifts: int = 8,
    ) -> None:
        dev = torch.device(device)
        model, tok, cfg = _build_model_and_tokenizer(ckpt, dev)
        ckpt_path = Path(ckpt)
        ckpt_sha256 = sha256_file(ckpt_path)
        self._setup(
            model=model,
            tok=tok,
            cfg=cfg,
            device=dev,
            batch=batch,
            ckpt_path=ckpt_path,
            ckpt_sha256=ckpt_sha256,
            label=label,
            compact_every=compact_every,
            compact_below=compact_below,
            update_shifts=update_shifts,
            steering_spec=None,
            from_learner=False,
        )

    @classmethod
    def _from_model(
        cls,
        *,
        model: Any,
        tok: Any,
        cfg: dict[str, int],
        device: torch.device,
        batch: int,
        ckpt_path: Path,
        ckpt_sha256: str,
        label: str | None,
        compact_every: int,
        compact_below: float,
        update_shifts: int,
        steering_spec: SteeringSpec | None,
        from_learner: bool,
    ) -> AbsGptAdapter:
        """Build an adapter around an already-in-memory `model`/`tok` (no checkpoint load): used
        by `with_steering` (a deep-copied model with new hooks) and `_AbsGptLearner.adapter`
        (the learner's own, live, mutable model)."""
        obj = cls.__new__(cls)
        obj._setup(
            model=model,
            tok=tok,
            cfg=cfg,
            device=device,
            batch=batch,
            ckpt_path=ckpt_path,
            ckpt_sha256=ckpt_sha256,
            label=label,
            compact_every=compact_every,
            compact_below=compact_below,
            update_shifts=update_shifts,
            steering_spec=steering_spec,
            from_learner=from_learner,
        )
        return obj

    def _setup(
        self,
        *,
        model: Any,
        tok: Any,
        cfg: dict[str, int],
        device: torch.device,
        batch: int,
        ckpt_path: Path,
        ckpt_sha256: str,
        label: str | None,
        compact_every: int,
        compact_below: float,
        update_shifts: int,
        steering_spec: SteeringSpec | None,
        from_learner: bool,
    ) -> None:
        self._model = model
        self._tok = tok
        self._cfg = cfg
        self._device = device
        self._batch = batch
        self._ckpt_path = ckpt_path
        self._ckpt_sha256 = ckpt_sha256
        self._compact_every = compact_every
        self._compact_below = compact_below
        self._update_shifts = update_shifts
        self._steering_spec: SteeringSpec = steering_spec or ()
        self._from_learner = from_learner
        self._steer_prompt_len = 0

        default_label = f"absgpt:{ckpt_path.stem}"
        if from_learner:
            default_label += ":learner"
        if self._steering_spec:
            default_label += f"+steer{len(self._steering_spec)}"
        self.label = label or default_label
        self.format_id = "nd_v1"
        self.capabilities = (
            Capability.SAMPLE
            | Capability.PREFIX
            | Capability.LOGPROB
            | Capability.FINETUNE
            | Capability.ACTIVATIONS
            | Capability.STEER
        )
        self._steer_handles: list[Any] = (
            self._install_steering_hooks(self._steering_spec) if self._steering_spec else []
        )

    # ------------------------------------------------------------------- ModelAdapter contract

    def identity(self) -> dict[str, JSON]:
        d: dict[str, JSON] = {
            "adapter": "absgpt",
            "ckpt": str(self._ckpt_path),
            "ckpt_sha256": self._ckpt_sha256,
            "arch": dict(self._cfg),
            "tokenizer_mode": self._tok.mode,
            "tokenizer_vocab_sha256": sha256_str(",".join(self._tok.itos))[:16],
            "tokenizer_maxn": MAXN,
        }
        if self._from_learner:
            d["derived_from_learner"] = True
        if self._steering_spec:
            d["steering_sha256"] = _steering_spec_hash(self._steering_spec)
            d["steering_terms"] = len(self._steering_spec)
            # Steered sampling before 2026-10-01 ignored the prompt boundary (see `sample()`);
            # a distinct identity keeps those runs from being resumed or skipped as complete.
            d["steering_positions"] = "per_row_prompt_boundary"
        return d

    def token_count(self, seq: Sequent, body: str | None = None) -> int:
        n = len(self._tok.encode_prompt(seq.prompt()))
        if body:
            n += len(self._tok.encode_proof(body))
        return n

    def sample(self, reqs: Sequence[SampleRequest]) -> list[list[Attempt]]:
        """Draw every request's `n` attempts in one batched pass over the GPU, batching *across*
        requests rather than looping `_gen` once per request (see the module docstring's "Cross-
        request batching" section for why: one call per request left the GPU far under-batched).

        Every row this call needs (`sum(req.n for req in reqs)` of them) is flattened into one
        pool -- request `r`'s rows all start from `r`'s own prompt ids plus any forced prefix ids
        (`r.prefix`, or the single synthetic N1 token `options={"start_index": 1}` asks for) --
        then grouped by `(max_new_tokens, temperature)` (the two settings one `_gen` call applies
        uniformly to its whole batch, so requests differing in either cannot share a call), sorted
        by input length within each group (`vendor/sampler.py:sample_ids`'s own discipline, so a
        chunk pads to a similar length rather than the longest and shortest row in the whole call),
        and chunked to `self._batch` rows per `_gen` call. Continuations are then mapped back to
        their originating `(request index, draw index)` and decoded exactly as a single-request
        call would: the forced prefix is prepended, `Attempt.native` preserves the raw decoded
        start index verbatim, and `Attempt.body` is `core.proof.renumber`'d to start at N1.

        **Determinism.** One `torch.Generator` is created for this call, seeded from a hash of
        every request's own `seed` field taken *in the order `reqs` is given*
        (`_cross_request_seed`), and reused sequentially across every group and chunk this call
        makes, always in the same fixed (group insertion order, then ascending chunk) order. Two
        calls with an identical `reqs` list therefore reproduce byte-identical attempts. This does
        **not** mean a given request's attempts are independent of who else is in the call: the
        same request batched alongside different other requests, or with a different `self._batch`,
        generally draws different samples, since `torch.multinomial`'s output depends on the
        generator's state at the time of each draw (itself a function of the batch's shape and
        every earlier group/chunk this same call already consumed the generator for) and not only
        on the nominal per-request `seed`.
        """
        if not reqs:
            return []

        input_ids_by_req: list[list[int]] = []
        forced_ids_by_req: list[list[int]] = []
        prompt_len_by_req: list[int] = []
        for req in reqs:
            prompt_ids = self._tok.encode_prompt(req.sequent.prompt())
            prompt_len_by_req.append(len(prompt_ids))
            forced_ids: list[int] = []
            if req.prefix is not None:
                forced_ids = self._tok.encode_proof(req.prefix.body)
            elif req.options.get("start_index") == 1:
                forced_ids = [self._tok._ref(1)]
            input_ids_by_req.append(prompt_ids + forced_ids)
            forced_ids_by_req.append(forced_ids)

        # Group every (request, draw) row by the settings one `_gen` call must share across its
        # whole batch; insertion order follows `reqs` itself, so the group-processing order below
        # is a deterministic function of `reqs` alone.
        groups: dict[tuple[int, float], list[tuple[int, int]]] = {}
        for r_idx, req in enumerate(reqs):
            key = (req.max_new_tokens, req.temperature)
            groups.setdefault(key, []).extend((r_idx, i) for i in range(req.n))

        gen = torch.Generator(device=self._device)
        gen.manual_seed(_cross_request_seed(reqs))
        cuda_enabled = self._device.type == "cuda"

        continuations: dict[tuple[int, int], list[int]] = {}
        for (max_new, temperature), row_keys in groups.items():
            order = sorted(
                range(len(row_keys)), key=lambda k: len(input_ids_by_req[row_keys[k][0]])
            )
            for s in range(0, len(order), self._batch):
                chunk_keys = [row_keys[p] for p in order[s : s + self._batch]]
                batch_ids = [input_ids_by_req[r_idx] for r_idx, _ in chunk_keys]
                # `_GEN` (`vendor/sampler.py:_gen`) is plain inference with no `torch.no_grad()`
                # of its own (unlike `sample_ids`, which wraps its own call to it) -- without this,
                # every one of up to `max_new` decode steps extends a live autograd graph over the
                # whole growing KV cache, which is pure waste for sampling and, at a large enough
                # batch x sequence length, can by itself exhaust GPU memory (observed: a 256-item,
                # batch=1024 call OOM'd on a 24 GB RTX 3090 for this small model before this fix).
                if self._steering_spec:
                    self._steer_prompt_len = torch.tensor(
                        [prompt_len_by_req[r_idx] for r_idx, _ in chunk_keys], device=self._device
                    )
                try:
                    with (
                        torch.no_grad(),
                        torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda_enabled),
                    ):
                        outs = _GEN(
                            self._model,
                            self._tok,
                            batch_ids,
                            temperature,
                            max_new,
                            gen,
                            self._compact_every,
                            self._compact_below,
                        )
                finally:
                    self._steer_prompt_len = 0
                for key, cont in zip(chunk_keys, outs, strict=True):
                    continuations[key] = cont
                if cuda_enabled:
                    torch.cuda.empty_cache()

        out: list[list[Attempt]] = []
        for r_idx, req in enumerate(reqs):
            forced_ids = forced_ids_by_req[r_idx]
            attempts: list[Attempt] = []
            for i in range(req.n):
                cont = continuations[(r_idx, i)]
                finished = bool(cont) and cont[-1] == self._tok.eos
                full_ids = forced_ids + cont
                native = self._tok.decode(full_ids)
                body = renumber(native, new_start=1)
                attempts.append(
                    Attempt(
                        pid=req.pid,
                        i=i,
                        body=body,
                        native=native,
                        n_tokens=len(cont),
                        finished=finished,
                    )
                )
            out.append(attempts)
        return out

    def logprob(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        return _batched_shift_marginal_logprobs(
            self._model, self._tok, self._device, self, MAXN, pairs
        )

    def masked_next_logprobs(
        self, items: Sequence[tuple[Sequent, str]]
    ) -> list[MaskedLogProbs | None]:
        """Full next-token log-probabilities over the literal, as-generated proof-token range of
        each `(seq, native)` pair (`MaskedLogProbs`'s own docstring) -- `interp/kl.py`'s per-model
        half of an exact token-level KL divergence computation against another `AbsGptAdapter`
        sharing one tokenizer/vocab. `native` must be a complete (`'... QED'`-terminated) proof
        text in this adapter's own tokenizer vocabulary; returns `None` for anything else (no
        trailing `QED`, or a `KeyError`/`ValueError` from `encode_proof` -- an out-of-vocabulary
        token or an `N<k>` index outside `1..MAXN`), `interp/kl.py`'s "skip samples whose ND body
        is UNPARSEABLE or outside the grammar (count them)" case, for the caller to count.
        """
        item_ok = [False] * len(items)
        item_n_steps = [0] * len(items)

        row_ids: list[list[int]] = []
        row_mask_start: list[int] = []
        row_prompt_len: list[int] = []
        row_step_of: list[np.ndarray] = []
        row_item: list[int] = []

        for item_idx, (seq, native) in enumerate(items):
            toks_text = native.split()
            if not toks_text or toks_text[-1] != "QED":
                continue
            try:
                pid = self._tok.encode_prompt(seq.prompt())
                qid = self._tok.encode_proof(native)
            except (KeyError, ValueError):
                continue
            if len(qid) != len(toks_text):
                continue
            step_of = np.empty(len(toks_text), dtype=np.int64)
            cur = 0
            for k, t in enumerate(toks_text):
                step_of[k] = cur
                if t == ";":
                    cur += 1
            item_ok[item_idx] = True
            item_n_steps[item_idx] = cur + 1  # n_lines + 1 (the trailing QED "FINAL" step)
            row_ids.append(pid + qid)
            row_mask_start.append(len(pid) - 1)
            row_prompt_len.append(len(pid))
            row_step_of.append(step_of)
            row_item.append(item_idx)

        n_rows = len(row_ids)
        per_row_logprobs: list[np.ndarray | None] = [None] * n_rows
        per_row_targets: list[np.ndarray | None] = [None] * n_rows

        if n_rows:
            row_lens = [len(r) for r in row_ids]
            order = sorted(range(n_rows), key=lambda i: row_lens[i])
            chunks = _length_bucketed_chunks(
                order, row_lens, _LOGPROB_MAX_BATCH_ROWS, _LOGPROB_MAX_BATCH_CELLS
            )

            self._model.eval()
            with torch.no_grad():
                for chunk in chunks:
                    chunk_ids = [row_ids[i] for i in chunk]
                    b = len(chunk_ids)
                    length = max(len(r) for r in chunk_ids)
                    x = torch.full(
                        (b, length), self._tok.pad, dtype=torch.long, device=self._device
                    )
                    for bi, r in enumerate(chunk_ids):
                        x[bi, : len(r)] = torch.tensor(r, device=self._device)
                    lengths_t = torch.tensor([len(row_ids[i]) for i in chunk], device=self._device)
                    mask_starts = torch.tensor(
                        [row_mask_start[i] for i in chunk], device=self._device
                    )
                    prompt_lens = torch.tensor(
                        [row_prompt_len[i] for i in chunk], dtype=torch.long, device=self._device
                    )

                    self._steer_prompt_len = prompt_lens
                    try:
                        logits = self._model(x[:, :-1]).float()
                    finally:
                        self._steer_prompt_len = 0

                    logp = F.log_softmax(logits, dim=-1)
                    tgt = x[:, 1:]
                    col = torch.arange(logp.shape[1], device=self._device).unsqueeze(0)
                    valid_end = (lengths_t - 1).unsqueeze(1)
                    row_mask = (col >= mask_starts.unsqueeze(1)) & (col < valid_end)
                    for local_i, global_i in enumerate(chunk):
                        idxs = row_mask[local_i].nonzero(as_tuple=True)[0]
                        per_row_logprobs[global_i] = logp[local_i, idxs].cpu().numpy()
                        per_row_targets[global_i] = tgt[local_i, idxs].cpu().numpy()

        results: list[MaskedLogProbs | None] = [None] * len(items)
        for row_idx in range(n_rows):
            item_idx = row_item[row_idx]
            log_probs = per_row_logprobs[row_idx]
            target_ids = per_row_targets[row_idx]
            assert log_probs is not None and target_ids is not None
            results[item_idx] = MaskedLogProbs(
                log_probs=log_probs.astype(np.float32, copy=False),
                target_ids=target_ids.astype(np.int64, copy=False),
                step_of=row_step_of[row_idx],
                n_steps=item_n_steps[item_idx],
                n_tokens=int(target_ids.shape[0]),
            )
        return results

    def learner(self, *, seed: int) -> _AbsGptLearner:
        """Narrower than `ModelAdapter.learner`'s declared `-> Learner` (a legal covariant
        override, matching `test_wp5_fixtures.TinyGptAdapter.learner`'s own pattern): lets a
        caller that holds an `AbsGptAdapter` reach `_AbsGptLearner`'s duck-typed extensions
        (`update_shifts`, `nll_start_excluded`) directly and type-checked, without `edl.run_edl`'s
        `getattr`/`setattr` fallback (needed only when the concrete `Learner` type isn't known)."""
        model_copy = copy.deepcopy(self._model)
        opt = torch.optim.AdamW(
            model_copy.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0
        )
        return _AbsGptLearner(
            model=model_copy,
            tok=self._tok,
            device=self._device,
            opt=opt,
            update_shifts=self._update_shifts,
            seed=seed,
            base_adapter=self,
        )

    def activations(
        self,
        pairs: Sequence[tuple[Sequent, str]],
        layers: Sequence[int],
        *,
        max_batch_rows: int = _LOGPROB_MAX_BATCH_ROWS,
        max_batch_cells: int = _LOGPROB_MAX_BATCH_CELLS,
    ) -> list[np.ndarray]:
        """Batched residual-stream extraction (throughput fix, matching `logprob`'s own batching):
        every pair's ids are right-padded into one batch, sorted by length first and grouped into
        chunks via `_length_bucketed_chunks` (bucketed by both row count, `max_batch_rows`, and
        total padded cell count, `max_batch_cells` -- the keywords are mainly for tests that want to
        force multiple chunks on a small `pairs` list; ordinary callers can leave them at their
        defaults), each requested block's output is captured for the *whole* chunk in one forward
        call, and each pair's own array is sliced back out of that batched capture with its padding
        stripped. Identical per-pair values to a one-forward-per-pair loop: causal attention (this
        is a plain, non-cached forward pass) means a row's real positions never see its own
        trailing padding or any other row sharing its batch.
        """
        if not pairs:
            return []
        ids_by_pair: list[list[int]] = [
            self._tok.encode_prompt(seq.prompt()) + self._tok.encode_proof(body)
            for seq, body in pairs
        ]
        id_lens = [len(r) for r in ids_by_pair]
        order = sorted(range(len(pairs)), key=lambda i: id_lens[i])
        chunks = _length_bucketed_chunks(order, id_lens, max_batch_rows, max_batch_cells)
        result_by_idx: dict[int, np.ndarray] = {}

        for chunk in chunks:
            chunk_ids = [ids_by_pair[i] for i in chunk]
            b = len(chunk_ids)
            length = max(len(r) for r in chunk_ids)
            x = torch.full((b, length), self._tok.pad, dtype=torch.long, device=self._device)
            for bi, r in enumerate(chunk_ids):
                x[bi, : len(r)] = torch.tensor(r, device=self._device)

            captured: dict[int, torch.Tensor] = {}
            handles = []
            for layer in layers:
                block = self._model.blocks[layer]
                handles.append(block.register_forward_hook(_make_capture_hook(captured, layer)))
            try:
                with torch.no_grad():
                    self._model(x)
            finally:
                for h in handles:
                    h.remove()

            for local_i, pair_idx in enumerate(chunk):
                real_len = len(chunk_ids[local_i])
                result_by_idx[pair_idx] = np.stack(
                    [captured[layer][local_i, :real_len].cpu().numpy() for layer in layers], axis=0
                )

        return [result_by_idx[i] for i in range(len(pairs))]

    def with_steering(self, spec: SteeringSpec) -> AbsGptAdapter:
        """Narrower than `ModelAdapter.with_steering`'s declared `-> ModelAdapter` (a legal
        covariant override, the same rationale as `learner`'s own narrowing above): the returned
        adapter is always in fact another `AbsGptAdapter` (over a deep-copied, hook-carrying
        model), so a caller holding an `AbsGptAdapter` can reach its private, steering-relevant
        attributes (`_model`, `_tok`, `_steer_prompt_len`) directly and type-checked -- used by
        `test_wp1_absgpt.py`'s position-recovery test, which drives the steered model directly
        rather than through `sample()`/`logprob()`."""
        model_copy = copy.deepcopy(self._model)
        return AbsGptAdapter._from_model(
            model=model_copy,
            tok=self._tok,
            cfg=self._cfg,
            device=self._device,
            batch=self._batch,
            ckpt_path=self._ckpt_path,
            ckpt_sha256=self._ckpt_sha256,
            label=None,
            compact_every=self._compact_every,
            compact_below=self._compact_below,
            update_shifts=self._update_shifts,
            steering_spec=spec,
            from_learner=self._from_learner,
        )

    def close(self) -> None:
        for h in getattr(self, "_steer_handles", ()):
            h.remove()
        self._steer_handles = []

    # ------------------------------------------------------------------------------- steering

    def _install_steering_hooks(self, spec: SteeringSpec) -> list[Any]:
        handles: list[Any] = []
        for term in spec:
            vec = _resolve_steering_vector(term.vector)
            delta = torch.tensor(vec, dtype=torch.float32, device=self._device)
            block = self._model.blocks[term.layer]
            handles.append(block.register_forward_hook(_make_steering_hook(self, term, delta)))
        return handles


def _make_steering_hook(owner: AbsGptAdapter, term: Any, delta: torch.Tensor) -> Any:
    """A `register_forward_hook` callback adding `term.alpha * delta` to the residual stream at
    the positions `term.where` selects (plan §A3). A plain function factory (rather than a closure
    defined inline inside `AbsGptAdapter._install_steering_hooks`'s `for term in spec` loop) so
    `term`/`delta` are bound by ordinary argument passing, not by closing over a loop variable.

    Reads `owner._steer_prompt_len` (set/cleared by whichever adapter method is currently driving
    the model) to split "prompt" from "proof" positions; see the module docstring for how the
    absolute token positions of one call are recovered *per row* from the block's own `mask`
    forward argument during `_gen`-driven sampling (needed since cross-request batching can now put
    rows with different amounts of left padding in the same call), versus a plain, batch-wide
    `arange(T)` for a non-cached forward pass (`logprob`/`nll`/`activations`/`Learner.update`).
    `owner._steer_prompt_len` can be either a plain Python `int` (one prompt length shared by the
    whole batch -- `activations`/`Learner.update` never set it, so it stays the module default `0`;
    a single-pair call effectively has just one prompt length too) or a `torch.Tensor` of shape
    `(B,)` (one prompt length *per row* -- `_batched_shift_marginal_logprobs` sets this, since a
    right-padded chunk can hold rows from several different pairs with different prompt lengths,
    even though every row belonging to the *same* pair always shares one: a shift only moves
    proof-token indices, never the prompt). Either way it is compared against `abs_pos` (shape
    `(B, T)`): a plain int broadcasts as one threshold for the whole batch; a `(B,)` tensor is
    reshaped to `(B, 1)` first so each row is compared against its own threshold.
    """

    def _hook(module: Any, inputs: Any, output: torch.Tensor) -> torch.Tensor:
        mask = inputs[3] if len(inputs) > 3 else None
        cache = inputs[4] if len(inputs) > 4 else None
        b, t, _d = output.shape
        if isinstance(cache, dict) and "n" in cache and mask is not None:
            # `_gen`'s left-padded, KV-cached layout (module docstring): `mask`'s own last query
            # row is exactly this call's `keep` vector, spanning every cached key column so far --
            # at prefill it is `causal`'s all-True last row ANDed with `keep` (the `torch.eye` OR
            # only ever touches that row's own diagonal entry, already kept), and at each
            # incremental decode step `mask` *is* `keep`, with no causal component at all. Its
            # per-row cumulative count minus one is each row's real (unpadded) position; the last
            # `t` columns are this call's own newly appended tokens.
            keep_full = mask[:, 0, -1, :].to(torch.bool)
            pos_full = (keep_full.cumsum(dim=1) - 1).clamp(min=0)
            abs_pos = pos_full[:, -t:]  # (B, t), one row per batch element
        else:
            abs_pos = torch.arange(t, device=output.device).unsqueeze(0).expand(b, t)
        prompt_len = owner._steer_prompt_len
        if isinstance(prompt_len, torch.Tensor):
            if prompt_len.numel() != b:
                # `_gen` compacts finished rows out of the batch mid-decode, so `sample()`'s
                # per-row prompt lengths no longer line up with the rows. Every such call is an
                # incremental decode step whose tokens are all generated (proof) tokens, past
                # every row's prompt, so a threshold of 0 classifies them correctly.
                prompt_len = 0
            else:
                prompt_len = prompt_len.to(device=abs_pos.device).view(-1, 1)
        if term.where == "all":
            pos_mask = torch.ones_like(abs_pos, dtype=torch.bool)
        elif term.where == "prompt":
            pos_mask = abs_pos < prompt_len
        else:  # "proof"
            pos_mask = abs_pos >= prompt_len
        if not bool(pos_mask.any()):
            return output
        updated = output.clone()
        addition = term.alpha * delta.to(output.dtype)
        updated[pos_mask] = updated[pos_mask] + addition
        return updated

    return _hook


class _AbsGptLearner:
    """`Learner` for `AbsGptAdapter` (plan §1): AdamW (betas 0.9/0.95, wd 0, grad-clip 1.0) over a
    deep-copied model, trained on `update_shifts` random start-shift copies per example per call.
    """

    def __init__(
        self,
        *,
        model: Any,
        tok: Any,
        device: torch.device,
        opt: torch.optim.Optimizer,
        update_shifts: int,
        seed: int,
        base_adapter: AbsGptAdapter,
    ) -> None:
        self._model = model
        self._tok = tok
        self._device = device
        self._opt = opt
        #: Public (not `_`-prefixed): the number of independent random-shift copies `update()`
        #: makes per training example. `edl.py`'s `run_edl` reads/sets this duck-typed attribute
        #: (`getattr`/`setattr`, no `Learner`-protocol change) to make sure exactly one layer does
        #: the shift-copying -- it passes each pair to `update()` once, relying on this attribute,
        #: rather than also repeating the pair `shifts_per_update` times itself (plan §B7's "8
        #: random-shift copies per update", not 8*8).
        self.update_shifts = update_shifts
        self._rng = random.Random(seed)
        self._base_adapter = base_adapter
        self._steer_prompt_len = 0

    def nll(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        """Shift-marginal *negative* log-likelihood at the learner's current weights (plan §3 WP1:
        "nll uses the shift-marginal loss over proof tokens including QED, conditioned on the
        prompt"; `Learner.nll`'s contract per `core.types.LogProb` is positive nats, the opposite
        sign from `ModelAdapter.logprob`)."""
        return _batched_shift_marginal_nlls(self._model, self._tok, self._device, self, MAXN, pairs)

    def nll_start_excluded(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        """Optional sensitivity-check variant of `nll` (plan §B7; `edl.py`'s `run_edl` looks this
        up via `getattr`): the start-index token's contribution to each shift's log-probability is
        dropped before marginalising, matching `test_wp5_fixtures.TinyGptLearner.
        nll_start_excluded`."""
        return _batched_shift_marginal_nlls(
            self._model, self._tok, self._device, self, MAXN, pairs, start_excluded=True
        )

    def update(self, pairs: Sequence[tuple[Sequent, str]], lr: float) -> float:
        self._model.train()
        for g in self._opt.param_groups:
            g["lr"] = lr
        ref0 = self._tok.ref0
        rows: list[list[int]] = []
        prompt_lens: list[int] = []
        for seq, body in pairs:
            pid = self._tok.encode_prompt(seq.prompt())
            qid = self._tok.encode_proof(body)
            mx = max((x - ref0 + 1 for x in qid if x >= ref0), default=0)
            max_shift = max(MAXN - mx, 0)
            for _ in range(self.update_shifts):
                s = self._rng.randint(0, max_shift) if max_shift > 0 else 0
                ids = pid + [x + s if x >= ref0 else x for x in qid]
                rows.append(ids)
                prompt_lens.append(len(pid))
        if not rows:
            self._model.eval()
            return 0.0
        max_len = max(len(r) for r in rows)
        x = torch.full((len(rows), max_len), self._tok.pad, dtype=torch.long, device=self._device)
        for i, r in enumerate(rows):
            x[i, : len(r)] = torch.tensor(r, device=self._device)
        logits = self._model(x[:, :-1]).float()
        tgt = x[:, 1:]
        logp = F.log_softmax(logits, dim=-1).gather(-1, tgt[..., None]).squeeze(-1)
        mask = torch.zeros_like(tgt, dtype=torch.bool)
        for i, (r, lp) in enumerate(zip(rows, prompt_lens, strict=True)):
            mask[i, lp - 1 : len(r) - 1] = True
        n_tok = mask.sum().clamp(min=1)
        loss = -(logp * mask).sum() / n_tok
        pre_update_loss = float(loss.item())
        self._opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._opt.step()
        self._model.eval()
        return pre_update_loss

    def adapter(self) -> ModelAdapter:
        base = self._base_adapter
        return AbsGptAdapter._from_model(
            model=self._model,
            tok=self._tok,
            cfg=base._cfg,
            device=self._device,
            batch=base._batch,
            ckpt_path=base._ckpt_path,
            ckpt_sha256=base._ckpt_sha256,
            label=f"{base.label}:learner",
            compact_every=base._compact_every,
            compact_below=base._compact_below,
            update_shifts=base._update_shifts,
            steering_spec=None,
            from_learner=True,
        )
