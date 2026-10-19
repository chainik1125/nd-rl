"""Ordering-marginalised likelihood, part 2: `marginal_logprob` (`INTERP_SPEC.md` §1),
base/RL disagreement statistics, and the `interp likelihood` CLI run.

`marginal_logprob` scores every ordering `orderings.py` finds equivalent to a given proof and
combines them by log-sum-exp (`ModelAdapter.logprob` already marginalises each individual scoring
call over start-index shifts, so this is a marginalisation *on top of* that one, not instead of
it). `disagreement_stats` instead teacher-forces a single, specific proof through both models and
compares their per-token predictions -- Akgul et al. (2026)'s style of base/RL divergence analysis.

**Throughput fix (2026-10-01).** Profiling a small (30 neutral + 30 RL + 30 base) real slice found
that `AbsGptAdapter.logprob`'s dominant real cost (per `torch.Tensor.item()` call inside its
per-row shift-marginalisation loop, i.e. one host/device sync per scored *row*) is essentially
proportional to the number of distinct `(seq, body)` pairs actually scored, not to the number of
Python-level `adapter.logprob(...)` calls -- so the fix is cutting redundant rows, not just
spreading calls out. `run_likelihood`'s old per-item loop scored each item's own `body` *twice*
under each model (once as `single`, once again as part of its own ordering set, since the original
body is always one of `orderings.enumerate_orderings`' outputs -- confirmed empirically on the real
`gen_ctrl`/`parts_pool` data: every one of 90 sampled items' bodies reappeared verbatim in its own
enumeration) and never batched more than one item's rows into a single `adapter.logprob` call.
`marginal_logprob` below now reuses the already-scored copy of `body` found inside its own
`bodies`/`enumerate_orderings` list instead of a second, separate call (falling back to one only if
that lookup somehow misses -- the importance-sampling branch, where there is no such guarantee,
is unaffected and keeps its own dedicated `single` call). `run_likelihood` goes further: it plans
every item's ordering set on the CPU first (cheap: `orderings.count_orderings` on the real
`gen_ctrl`/`parts_pool`/reserve data profiled at 0.07s total across 1,540 items, now backed by
`orderings.py`'s own connected-components speedup for the rarer wide-independent-line case), then
scores a whole chunk of items' pooled rows in one `adapter.logprob` call per model
(`_score_plans`) and one batched teacher-forced-logits call per model for that chunk's
`disagreement_stats` (`_teacher_forced_logits_batch`), writing each chunk's rows to the `Store`
immediately afterwards (`append_jsonl`, not one `write_jsonl` at the very end) so a timeout leaves
every already-scored chunk's results on disk instead of nothing.

**A third, separate cause (found when the above was actually run for real):** the real `families`
Store suite has 2,193 witnesses, and the pre-fix `neutral += family_items` added every single one
of them *uncapped* -- well past where `cfg["n_neutral"]` was ever checked (`n_neutral` only bounds
the `reserve.jsonl`/`sealed_L12-14` draw that happens *before* family items are appended), and well
past the (CPU-cheap, already-fixed) ordering-planning cost: scoring all 2,193 needed ~27,400
ordering-rows (one model) by itself, dwarfing the ~1,540 reserve/verified items' ~1,760. See
`DEFAULT_FAMILY_LIMIT` and `data.load_family_witnesses`'s own fix (it used to return a plain
`items[:n]` head slice when capped, which happened to be *entirely* one family shape out of 92 --
now shuffled first)."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ..adapters.base import ModelAdapter, NotSupported
from ..core.io import append_jsonl, read_jsonl
from ..core.proof import split_lines
from ..core.types import JSON, Sequent
from ..store import Store
from . import data, orderings
from ._store import interp_dir, new_interp_run

#: `INTERP_SPEC.md` §1: "cap=10_000, m=512" are `marginal_logprob`'s own defaults; `run_likelihood`
#: reads them from `cfg` (so a config file can override them) but falls back to these.
DEFAULT_CAP = 10_000
DEFAULT_M = 512

#: `run_likelihood`'s items-per-write chunk (throughput fix, module docstring): big enough that one
#: `adapter.logprob`/`_teacher_forced_logits_batch` call per chunk covers many items' rows at once,
#: small enough that a timeout mid-run still leaves every already-finished chunk's rows on disk.
#: `cfg["chunk_size"]` overrides it.
DEFAULT_CHUNK_SIZE = 200

#: `run_likelihood`'s cap on family witnesses (throughput fix, 2026-10-01: the real `families`
#: suite has 2,193 of them, some long -- up to ~900 proof tokens -- and/or with hundreds of
#: thousands of orderings; even bounded per item by `cap`/`m`, scoring all 2,193 (a CPU-only
#: planning pass found ~27,400 total ordering-rows across them, one model) is what actually blew
#: the GPU-minute budget on a real run, well after `n_neutral` itself was already satisfied by
#: `reserve.jsonl` alone -- `cfg["n_neutral"]`'s cap was never reaching family witnesses at all,
#: since the pre-fix code added them on top, uncapped: `neutral += family_items`). `cfg
#: ["family_limit"]` overrides this default; `0` drops family witnesses entirely, a negative value
#: keeps every one the suite has (`data.load_family_witnesses`'s own `n=None`, uncapped).
DEFAULT_FAMILY_LIMIT = 300

#: `run_likelihood`'s row shape for a model side that was not given an adapter at all (`base`/`rl`
#: `_adapters_from_cfg`'s `skip_base`/`skip_rl`) -- matches the pre-fix `_marginal_row`'s None
#: branch exactly: every field `None`, including `n_orderings` (nothing was computed for that
#: side, not even the proof-only ordering count, since it is shared with whichever side *does* have
#: an adapter and would otherwise be computed either way -- see `_plan_item`).
_NULL_MARGINAL_ROW: dict[str, JSON] = {
    "logp_single": None,
    "logp_marginal": None,
    "n_orderings": None,
    "method": None,
    "ess": None,
}


def _logsumexp(xs: Sequence[float]) -> float:
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return float("-inf")
    m = float(np.max(arr))
    if not math.isfinite(m):
        return m  # every value -inf (or +inf, which log-sum-exp can't fix either way)
    return m + math.log(float(np.sum(np.exp(arr - m))))


@dataclass(frozen=True)
class MarginalResult:
    """`marginal_logprob`'s result: `logp_single` is `body`'s own log-probability (already
    marginalised over start shifts by `ModelAdapter.logprob`); `logp_marginal` additionally
    marginalises over every equivalent ordering. `method` is `"single"` when there is exactly one
    ordering (no reordering freedom at all -- `logp_marginal == logp_single` then, exactly),
    `"exact"` when `count_orderings` gave an exact, enumerable count `> 1`, else `"importance"`.
    `ess` (effective sample size, normalised-weight convention) is `None` outside the importance
    case."""

    logp_single: float
    logp_marginal: float
    n_orderings: int | None
    method: str
    ess: float | None


def _combine_exact_or_importance(
    method: str, logps: Sequence[float], logqs: Sequence[float] | None
) -> tuple[float, float | None]:
    """The log-sum-exp (`method in ("single", "exact")`) or importance-sampling-estimate
    (`method == "importance"`) combination shared by `marginal_logprob` and `run_likelihood`'s
    batched `_score_plans` -- factored out so both pay for exactly the same formula. `"single"`
    (`len(logps) == 1`) and `"exact"` share one branch: log-sum-exp of a single value is that value
    itself, exactly (`_logsumexp`'s `m = x`, `arr - m = [0]`, `log(1) = 0`), matching
    `MarginalResult`'s documented `logp_marginal == logp_single` guarantee for `"single"` without a
    special case. Returns `(marginal, ess)`; `ess` is `None` outside the importance case."""
    if method != "importance":
        return _logsumexp(logps), None
    assert logqs is not None
    weights = [lp - lq for lp, lq in zip(logps, logqs, strict=True)]
    marginal = _logsumexp(weights) - math.log(len(weights))
    ess = math.exp(2 * _logsumexp(weights) - _logsumexp([2 * w for w in weights]))
    return marginal, ess


def marginal_logprob(
    adapter: ModelAdapter,
    seq: Sequent,
    body: str,
    *,
    cap: int = DEFAULT_CAP,
    m: int = DEFAULT_M,
    rng: random.Random | None = None,
) -> MarginalResult:
    """`log sum_sigma p(sigma | prompt)` over every ordering `sigma` equivalent to `body`: exact
    log-sum-exp over `orderings.enumerate_orderings` when `orderings.count_orderings(body) <= cap`;
    otherwise the importance-sampling estimate `log(mean(exp(logp(sigma) - log q(sigma))))` from
    `orderings.sample_orderings`'s `m` proposal draws (`q` the "pick uniformly among ready units"
    proposal), with its effective sample size in normalised-weight terms,
    `exp(2*logsumexp(w) - logsumexp(2w))` for log-weights `w = logp - log q`.

    **Throughput fix (module docstring):** in the exact branch, `body` itself is always one of
    `enumerate_orderings`' own outputs (it is *a* valid linear extension of its own DAG), so
    `logp_single` is read straight out of the same `adapter.logprob` call that scores every
    ordering, instead of a second, separate, redundant call scoring `body` all over again -- this
    is exact, not an approximation (the same `(seq, body)` pair scored once, not twice and
    averaged). Falls back to one dedicated call only if that lookup ever misses (defensive; not
    observed on real proof data). The importance branch keeps its original dedicated `single`
    call: an `m`-draw sample is not guaranteed to include `body` itself.
    """
    n = orderings.count_orderings(body)
    if n is not None and n <= cap:
        bodies = orderings.enumerate_orderings(body, cap)
        logps = [lp.nats for lp in adapter.logprob([(seq, b) for b in bodies])]
        try:
            single = logps[bodies.index(body)]
        except ValueError:
            single = adapter.logprob([(seq, body)])[0].nats
        marginal, ess = _combine_exact_or_importance("exact", logps, None)
        return MarginalResult(single, marginal, n, "exact" if n > 1 else "single", ess)

    single = adapter.logprob([(seq, body)])[0].nats
    rng = rng if rng is not None else random.Random(0)
    samples = orderings.sample_orderings(body, m, rng)
    sample_bodies = [b for b, _q in samples]
    logqs = [q for _b, q in samples]
    logps = [lp.nats for lp in adapter.logprob([(seq, b) for b in sample_bodies])]
    marginal, ess = _combine_exact_or_importance("importance", logps, logqs)
    return MarginalResult(single, marginal, n, "importance", ess)


def _teacher_forced_logits(adapter: ModelAdapter, seq: Sequent, body: str) -> np.ndarray:
    """Per-proof-token next-token logits under `adapter`, teacher-forced on the literal (unshifted)
    encoding of `(seq, body)`: shape `(n_proof_tokens, vocab)`.

    **Deviation, reported in the interp coder's final report:** `ModelAdapter`'s public contract
    (`adapters/base.py`) only exposes an aggregate, shift-marginalised `logprob`; there is no
    capability for per-token teacher-forced logits, and adding one means editing
    `adapters/base.py`/`absgpt.py`, which this work package does not own. This instead reaches into
    the concrete adapter's own model/tokenizer (`AbsGptAdapter`'s `_model`/`_tok`, or a CPU test
    double's public `model`/`tok`, e.g. `test_wp5_fixtures.TinyGptAdapter`) -- both expose a plain
    `model(token_id_tensor) -> logits` call and a `tok.encode_prompt`/`tok.encode_proof` pair, so
    duck-typing across the two needs no adapter-specific branch. Raises `NotSupported` for any
    adapter exposing neither.
    """
    model = getattr(adapter, "_model", None)
    if model is None:
        model = getattr(adapter, "model", None)
    tok = getattr(adapter, "_tok", None)
    if tok is None:
        tok = getattr(adapter, "tok", None)
    if model is None or tok is None:
        raise NotSupported(
            f"{type(adapter).__name__} exposes neither `_model`/`_tok` nor `model`/`tok`; "
            "disagreement_stats needs teacher-forced per-token logits, which ModelAdapter's public "
            "contract does not expose (see _teacher_forced_logits's docstring)"
        )
    import torch  # local import: the only torch dependency in this module

    pid = tok.encode_prompt(seq.prompt())
    qid = tok.encode_proof(body)
    ids = pid + qid
    x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        logits = model(x[:, :-1])
    proof_logits = logits[0, len(pid) - 1 :].float().cpu().numpy()
    assert proof_logits.shape[0] == len(qid)
    return proof_logits


def _teacher_forced_logits_batch(
    adapter: ModelAdapter, pairs: Sequence[tuple[Sequent, str]]
) -> list[np.ndarray]:
    """Batched counterpart of `_teacher_forced_logits` (throughput fix, module docstring): every
    pair's ids are right-padded into one batch and scored with a single forward call instead of
    one call per pair, each pair's own per-token logits sliced back out with its padding stripped.
    Identical per-pair values to the one-forward-per-pair loop: this is a plain (non-cached)
    forward pass, so causal attention alone means a row's real positions never see its own
    trailing padding or any other row's content -- the same assumption `AbsGptAdapter.activations`
    already relies on for its own batched extraction. Same duck-typed `_model`/`_tok` access and
    the same deviation rationale as `_teacher_forced_logits` (see its docstring): `ModelAdapter`'s
    public contract has no per-token teacher-forced logits capability to batch through instead.
    """
    if not pairs:
        return []
    model = getattr(adapter, "_model", None)
    if model is None:
        model = getattr(adapter, "model", None)
    tok = getattr(adapter, "_tok", None)
    if tok is None:
        tok = getattr(adapter, "tok", None)
    if model is None or tok is None:
        raise NotSupported(
            f"{type(adapter).__name__} exposes neither `_model`/`_tok` nor `model`/`tok`; "
            "disagreement_stats needs teacher-forced per-token logits, which ModelAdapter's public "
            "contract does not expose (see _teacher_forced_logits's docstring)"
        )
    import torch  # local import: the only torch dependency in this module

    device = next(model.parameters()).device
    id_lists: list[list[int]] = []
    prompt_lens: list[int] = []
    for seq, body in pairs:
        pid = tok.encode_prompt(seq.prompt())
        qid = tok.encode_proof(body)
        id_lists.append(pid + qid)
        prompt_lens.append(len(pid))
    max_len = max(len(ids) for ids in id_lists)
    pad_id = getattr(tok, "pad", 0)
    x = torch.full((len(pairs), max_len), pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(id_lists):
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(x[:, :-1])
    out: list[np.ndarray] = []
    for i, ids in enumerate(id_lists):
        start = prompt_lens[i] - 1
        n_proof = len(ids) - prompt_lens[i]
        out.append(logits[i, start : start + n_proof].float().cpu().numpy())
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    m = float(np.max(x))
    e = np.exp(x - m)
    return e / e.sum()


@dataclass(frozen=True)
class DisagreementStats:
    """Base-vs-RL teacher-forced disagreement on one proof (Akgul et al. 2026 style):
    `disagree_frac` = fraction of proof-token positions where the two models' argmax token
    differs; at those positions only, `base_rank_of_rl_argmax` (0 = base agrees after all -- never
    happens at a disagreeing position, but kept 0-based for readability: 1 means the RL argmax is
    base's *second*-favourite token, etc.) and `base_entropy` (nats) of base's own distribution.
    `mean_*` are `None` when there are 0 disagreeing positions (nothing to average)."""

    n_positions: int
    n_disagree: int
    disagree_frac: float
    base_rank_of_rl_argmax: list[int]
    base_entropy_at_disagree: list[float]
    mean_base_rank: float | None
    mean_base_entropy: float | None


def _disagreement_from_logits(base_logits: np.ndarray, rl_logits: np.ndarray) -> DisagreementStats:
    """`disagreement_stats`'s own math, factored out so `run_likelihood`'s batched path
    (`_teacher_forced_logits_batch`, one call per chunk per model instead of one call per item per
    model) can reuse it per item without re-deriving it."""
    n = min(len(base_logits), len(rl_logits))
    base_logits, rl_logits = base_logits[:n], rl_logits[:n]
    base_argmax = base_logits.argmax(axis=-1)
    rl_argmax = rl_logits.argmax(axis=-1)
    disagree = base_argmax != rl_argmax

    ranks: list[int] = []
    entropies: list[float] = []
    for i in np.nonzero(disagree)[0]:
        row = base_logits[i]
        target_logit = row[rl_argmax[i]]
        ranks.append(int(np.sum(row > target_logit)))
        p = _softmax(row)
        entropies.append(float(-np.sum(p * np.log(np.clip(p, 1e-12, None)))))

    n_dis = int(disagree.sum())
    return DisagreementStats(
        n_positions=n,
        n_disagree=n_dis,
        disagree_frac=(n_dis / n) if n else float("nan"),
        base_rank_of_rl_argmax=ranks,
        base_entropy_at_disagree=entropies,
        mean_base_rank=(sum(ranks) / len(ranks)) if ranks else None,
        mean_base_entropy=(sum(entropies) / len(entropies)) if entropies else None,
    )


def disagreement_stats(
    base_adapter: ModelAdapter, rl_adapter: ModelAdapter, seq: Sequent, body: str
) -> DisagreementStats:
    base_logits = _teacher_forced_logits(base_adapter, seq, body)
    rl_logits = _teacher_forced_logits(rl_adapter, seq, body)
    return _disagreement_from_logits(base_logits, rl_logits)


@dataclass(frozen=True)
class _ItemPlan:
    """One item's ordering set, planned once on the CPU (`_plan_item`) and shared between the
    `base`/`rl` batched scoring calls -- `orderings.count_orderings`/`enumerate_orderings`/
    `sample_orderings` depend only on the proof itself, never on which model scores it, so the old
    per-model-side `marginal_logprob` calls were redundantly re-deriving the same ordering set
    twice per item (module docstring)."""

    pid: str
    origin: str
    seq: Sequent
    body: str
    L: int | None
    n: int | None
    method: str  # "single" | "exact" | "importance"
    bodies: list[str]  # the exact-enumeration or importance-sample orderings to score
    logqs: list[float] | None  # importance branch only; None for "single"/"exact"
    single_idx: int | None  # index into `bodies` holding `body` itself, if found there
    lookup_pid: str  # the *raw* theorem pid, for `base_success` lookups -- see `_plan_item`


def _plan_item(
    pid: str,
    origin: str,
    seq: Sequent,
    body: str,
    L: int | None,
    *,
    cap: int,
    m: int,
    rng: random.Random,
    lookup_pid: str | None = None,
) -> _ItemPlan:
    """`lookup_pid` defaults to `pid` (neutral/family items, whose display `pid` already *is* the
    raw theorem pid) but must be given explicitly for a model-sample item: `run_likelihood` names
    those rows `f"{origin}:{s.pid}"` (so a `base_verified` and an `rl_verified` row for the same
    theorem don't collide), but `base_success` (the Store's own `ItemResult.pid`) is keyed by the
    bare theorem pid `s.pid` -- a previous version of this function looked `base_success` up by
    the display `pid`, so `p_hat_base` (the crux-set marker) silently came back `None` on every
    verified-sample row; fixed 2026-10-01."""
    n = orderings.count_orderings(body)
    if n is not None and n <= cap:
        bodies = orderings.enumerate_orderings(body, cap)
        method = "exact" if n > 1 else "single"
        try:
            single_idx: int | None = bodies.index(body)
        except ValueError:
            single_idx = None
        return _ItemPlan(
            pid, origin, seq, body, L, n, method, bodies, None, single_idx, lookup_pid or pid
        )
    samples = orderings.sample_orderings(body, m, rng)
    sample_bodies = [b for b, _q in samples]
    logqs = [q for _b, q in samples]
    return _ItemPlan(
        pid, origin, seq, body, L, n, "importance", sample_bodies, logqs, None, lookup_pid or pid
    )


def _score_plans(adapter: ModelAdapter | None, plans: Sequence[_ItemPlan]) -> list[dict[str, JSON]]:
    """One batched `adapter.logprob` call scoring every plan's own ordering set at once (plus, for
    whichever plans need it -- always the importance branch, defensively the exact branch if
    `body` was somehow not found in its own `bodies` -- one extra row each for `body` itself),
    redistributing the results into one `_marginal_row`-shaped dict per plan, in the same order.
    `None` (no adapter for this model side) short-circuits to `_NULL_MARGINAL_ROW` for every plan,
    matching the pre-fix behaviour exactly."""
    if adapter is None:
        return [dict(_NULL_MARGINAL_ROW) for _ in plans]
    if not plans:
        return []

    # Pool every plan's own ordering-set rows (and, only when needed, one extra `single`-only
    # row) into one pairs list, each plan's own rows contiguous and in `plans` order -- so the
    # results can be sliced back out by running offset alone, no index bookkeeping needed.
    pairs: list[tuple[Sequent, str]] = []
    row_counts: list[int] = []  # len(p.bodies), plus 1 if this plan also got an extra single row
    for p in plans:
        pairs.extend((p.seq, b) for b in p.bodies)
        extra = p.single_idx is None
        if extra:
            pairs.append((p.seq, p.body))
        row_counts.append(len(p.bodies) + int(extra))

    lps = [lp.nats for lp in adapter.logprob(pairs)]

    out: list[dict[str, JSON]] = []
    offset = 0
    for p, count in zip(plans, row_counts, strict=True):
        own_rows = lps[offset : offset + count]
        offset += count
        logps = own_rows[: len(p.bodies)]
        single = logps[p.single_idx] if p.single_idx is not None else own_rows[-1]
        marginal, ess = _combine_exact_or_importance(p.method, logps, p.logqs)
        out.append(
            {
                "logp_single": single,
                "logp_marginal": marginal,
                "n_orderings": p.n,
                "method": p.method,
                "ess": ess,
            }
        )
    return out


def run_likelihood(
    store: Store,
    cfg: Mapping[str, JSON],
    *,
    base_adapter: ModelAdapter | None,
    rl_adapter: ModelAdapter | None,
) -> dict[str, Any]:
    """`interp likelihood` (`INTERP_SPEC.md` §1's "Run"): neutral minimal proofs (reserve L6-10 +
    sealed L12-14 + family witnesses, capped at `cfg["n_neutral"]`, default 2000) plus RL/base
    verified `Store` samples (from `cfg["model_samples"][model] = {"suite","version"}`; skipped
    with a note in the returned summary when that suite/version/model has no samples yet), scored
    under whichever of `base_adapter`/`rl_adapter` is given. Writes one row per `pid` to
    `out/store/interp/likelihood/<run_id>.jsonl`, in chunks of `cfg["chunk_size"]` items
    (`DEFAULT_CHUNK_SIZE`), appended as each chunk finishes (module docstring: a timeout mid-run
    leaves every already-scored chunk's rows on disk instead of nothing). A fresh call always
    starts that file empty first, so re-running an identical `cfg` after a previous partial/timed
    -out attempt does not mix stale rows in with the new ones.
    """

    def _int_cfg(key: str, default: int) -> int:
        v = cfg.get(key)
        return default if v is None else int(cast(float, v))

    seed = _int_cfg("seed", 0)
    cap = _int_cfg("cap", DEFAULT_CAP)
    m = _int_cfg("m", DEFAULT_M)
    n_neutral = _int_cfg("n_neutral", 2000)
    chunk_size = _int_cfg("chunk_size", DEFAULT_CHUNK_SIZE)
    notes: list[str] = []

    neutral: list[data.NeutralProof] = []
    if n_neutral > 0:
        excluder = data.build_exclusion_index(cache_dir=cast("str | None", cfg.get("cache_dir")))
        neutral = data.load_neutral_proofs(
            n=n_neutral,
            excluder=excluder,
            seed=seed,
            scan_limit=cast("int | None", cfg.get("scan_limit")),
        )
        if len(neutral) < n_neutral:
            neutral += data.load_sealed_l12_14(
                n=n_neutral - len(neutral), excluder=excluder, seed=seed
            )
    family_version = cfg.get("family_version")
    family_limit = _int_cfg("family_limit", DEFAULT_FAMILY_LIMIT)
    if family_limit == 0:
        family_items: list[data.NeutralProof] = []
    else:
        family_items = data.load_family_witnesses(
            store,
            version=str(family_version) if family_version else "v1",
            n=family_limit if family_limit > 0 else None,
            seed=seed,
        )
    if not family_items:
        notes.append("no families suite in the store yet; skipping family witnesses")
    elif family_limit > 0:
        notes.append(
            f"family witnesses capped at {family_limit} (shuffled for a cross-section of "
            "shapes/sizes; the suite has more -- throughput fix, see DEFAULT_FAMILY_LIMIT)"
        )
    neutral += family_items

    # `(display_pid, origin, seq, body, L, lookup_pid)` -- `lookup_pid` is the raw theorem pid
    # `base_success` (keyed by the Store's own `ItemResult.pid`) is looked up by; for neutral/
    # family items `display_pid` already *is* the raw pid, but a model-sample row's `display_pid`
    # carries an `f"{origin}:..."` prefix (so same-theorem base/rl rows don't collide) that
    # `base_success` never has (`_plan_item`'s docstring: this used to make `p_hat_base` always
    # `None`).
    items: list[tuple[str, str, Sequent, str, int | None, str]] = [
        (p.pid, p.origin, p.sequent, p.body, p.L, p.pid) for p in neutral
    ]

    model_samples_cfg = cfg.get("model_samples") or {}
    assert isinstance(model_samples_cfg, Mapping)
    solved_by: dict[str, set[str]] = {}
    for model_name in ("base", "rl"):
        spec = model_samples_cfg.get(model_name)
        if not spec:
            continue
        assert isinstance(spec, Mapping)
        samples = data.load_verified_samples(
            store,
            suite=str(spec["suite"]),
            version=str(spec["version"]),
            model=str(spec.get("model", model_name)),
            limit=int(cast(float, spec.get("limit", 500)) or 500),
        )
        if not samples:
            notes.append(f"no verified samples found for model_samples[{model_name}]; skipping")
            continue
        origin = f"{model_name}_verified"
        items += [(f"{origin}:{s.pid}", origin, s.sequent, s.body, None, s.pid) for s in samples]
        solved_by[model_name] = {s.pid for s in samples}

    # Neutral control for the model-sample rows (audit 2026-10-01: each model prefers its own
    # samples, so scoring rl's own proofs favours rl): `cfg["reference_proofs"] = {"suite",
    # "version", "solved_by"?, "proof_files"?}` scores a neutral proof of each of the suite's
    # theorems -- its own reference proof, else the first `proof` field of a row with the same
    # `prompt` in `proof_files` (JSONL; e.g. the minlen labeller's output for generator items,
    # whose suite rows carry no proof) -- optionally only for theorems `solved_by` (a
    # `model_samples` key) has a verified sample of. Origin "reference".
    ref_cfg = cfg.get("reference_proofs")
    if isinstance(ref_cfg, Mapping):
        only = solved_by.get(str(ref_cfg["solved_by"])) if ref_cfg.get("solved_by") else None
        by_prompt: dict[str, str] = {}
        for path in cast("Sequence[str]", ref_cfg.get("proof_files") or ()):
            for row in read_jsonl(path):
                pr, body = row.get("prompt"), row.get("proof")
                if isinstance(pr, str) and isinstance(body, str):
                    by_prompt.setdefault(pr, body)
        n_ref_missing = 0
        for prob in store.load_suite(str(ref_cfg["suite"]), str(ref_cfg["version"])):
            if only is not None and prob.pid not in only:
                continue
            body = prob.ref.body if prob.ref is not None else by_prompt.get(prob.sequent.prompt())
            if body is None:
                n_ref_missing += 1
                continue
            items.append(
                (f"reference:{prob.pid}", "reference", prob.sequent, body, prob.length.L, prob.pid)
            )
        if n_ref_missing:
            notes.append(f"reference_proofs: no neutral proof for {n_ref_missing} theorem(s)")

    base_success = _load_base_success(store, cfg.get("base_success"))

    run_id = new_interp_run(store, "likelihood", cfg)
    out_path = interp_dir(store, "likelihood") / f"{run_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()  # start this run_id's rows fresh (see docstring)

    n_written = 0
    for chunk_start in range(0, len(items), chunk_size):
        chunk_items = items[chunk_start : chunk_start + chunk_size]
        plans = [
            _plan_item(
                pid,
                origin,
                seq,
                body,
                L,
                cap=cap,
                m=m,
                rng=random.Random(f"{seed}:{pid}"),
                lookup_pid=lookup_pid,
            )
            for pid, origin, seq, body, L, lookup_pid in chunk_items
        ]

        base_rows = _score_plans(base_adapter, plans)
        rl_rows = _score_plans(rl_adapter, plans)

        disagreements: dict[int, dict[str, JSON]] = {}
        disagree_idx = [i for i, p in enumerate(plans) if p.origin == "rl_verified"]
        if base_adapter is not None and rl_adapter is not None and disagree_idx:
            sub_pairs = [(plans[i].seq, plans[i].body) for i in disagree_idx]
            base_logits = _teacher_forced_logits_batch(base_adapter, sub_pairs)
            rl_logits = _teacher_forced_logits_batch(rl_adapter, sub_pairs)
            for j, i in enumerate(disagree_idx):
                d = _disagreement_from_logits(base_logits[j], rl_logits[j])
                disagreements[i] = {
                    "n_positions": d.n_positions,
                    "n_disagree": d.n_disagree,
                    "disagree_frac": d.disagree_frac,
                    "mean_base_rank": d.mean_base_rank,
                    "mean_base_entropy": d.mean_base_entropy,
                }

        chunk_rows: list[dict[str, JSON]] = [
            {
                "pid": p.pid,
                "origin": p.origin,
                "L": p.L,
                # lines in the scored body itself (L is the theorem's certified minimum, which a
                # model's own sample can exceed): the per-written-line normaliser.
                "lines": len(split_lines(p.body)),
                "base": base_rows[i],
                "rl": rl_rows[i],
                "p_hat_base": base_success.get(p.lookup_pid),
                "disagreement": disagreements.get(i),
            }
            for i, p in enumerate(plans)
        ]
        append_jsonl(out_path, chunk_rows)
        n_written += len(chunk_rows)

    return {
        "run_id": run_id,
        "n_items": n_written,
        "n_neutral": len(neutral),
        "notes": notes,
    }


def _load_base_success(store: Store, cfg: JSON) -> dict[str, float]:
    """`p_hat_base` per `pid` from the largest matching base run in the `Store`, if `cfg` (a
    `{"suite", "version", "model"}` mapping) names one -- `INTERP_SPEC.md` §1's "crux set" marker.
    `None`/missing `cfg`, or no matching run, silently gives an empty mapping (every row's
    `p_hat_base` is then `None`, which is the documented behaviour, not a skipped-with-a-note
    case: this field is explicitly optional, "if any")."""
    if not cfg or not isinstance(cfg, Mapping):
        return {}
    from ..analysis._common import latest_run_id  # WP6 module; read-only import

    suite = cfg.get("suite")
    version = cfg.get("version")
    model = cfg.get("model", "base")
    if not suite or not version:
        return {}
    run_id = latest_run_id(store, suite=str(suite), version=str(version), model=str(model))
    if run_id is None:
        return {}
    return {it.pid: (it.c / it.n if it.n else float("nan")) for it in store.items(run_id)}
