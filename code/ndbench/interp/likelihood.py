"""Ordering-marginalised likelihood, part 2: `marginal_logprob` (`INTERP_SPEC.md` §1),
base/RL disagreement statistics, and the `interp likelihood` CLI run.

`marginal_logprob` scores every ordering `orderings.py` finds equivalent to a given proof and
combines them by log-sum-exp (`ModelAdapter.logprob` already marginalises each individual scoring
call over start-index shifts, so this is a marginalisation *on top of* that one, not instead of
it). `disagreement_stats` instead teacher-forces a single, specific proof through both models and
compares their per-token predictions -- Akgul et al. (2026)'s style of base/RL divergence analysis.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ..adapters.base import ModelAdapter, NotSupported
from ..core.types import JSON, Sequent
from ..store import Store
from . import data, orderings
from ._store import new_interp_run, write_interp_rows

#: `INTERP_SPEC.md` §1: "cap=10_000, m=512" are `marginal_logprob`'s own defaults; `run_likelihood`
#: reads them from `cfg` (so a config file can override them) but falls back to these.
DEFAULT_CAP = 10_000
DEFAULT_M = 512


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
    `exp(2*logsumexp(w) - logsumexp(2w))` for log-weights `w = logp - log q`."""
    single = adapter.logprob([(seq, body)])[0].nats
    n = orderings.count_orderings(body)
    if n is not None and n <= cap:
        bodies = orderings.enumerate_orderings(body, cap)
        logps = [lp.nats for lp in adapter.logprob([(seq, b) for b in bodies])]
        marginal = _logsumexp(logps)
        return MarginalResult(single, marginal, n, "exact" if n > 1 else "single", None)

    rng = rng if rng is not None else random.Random(0)
    samples = orderings.sample_orderings(body, m, rng)
    sample_bodies = [b for b, _q in samples]
    logqs = np.array([q for _b, q in samples], dtype=float)
    logps = np.array(
        [lp.nats for lp in adapter.logprob([(seq, b) for b in sample_bodies])], dtype=float
    )
    weights = (logps - logqs).tolist()
    marginal = _logsumexp(weights) - math.log(m)
    ess = math.exp(2 * _logsumexp(weights) - _logsumexp([2 * w for w in weights]))
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
    x = torch.tensor([ids], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits = model(x[:, :-1])
    proof_logits = logits[0, len(pid) - 1 :].float().numpy()
    assert proof_logits.shape[0] == len(qid)
    return proof_logits


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


def disagreement_stats(
    base_adapter: ModelAdapter, rl_adapter: ModelAdapter, seq: Sequent, body: str
) -> DisagreementStats:
    base_logits = _teacher_forced_logits(base_adapter, seq, body)
    rl_logits = _teacher_forced_logits(rl_adapter, seq, body)
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


def _marginal_row(
    adapter: ModelAdapter | None, seq: Sequent, body: str, *, cap: int, m: int, rng: random.Random
) -> dict[str, JSON]:
    if adapter is None:
        return {
            "logp_single": None,
            "logp_marginal": None,
            "n_orderings": None,
            "method": None,
            "ess": None,
        }
    r = marginal_logprob(adapter, seq, body, cap=cap, m=m, rng=rng)
    return {
        "logp_single": r.logp_single,
        "logp_marginal": r.logp_marginal,
        "n_orderings": r.n_orderings,
        "method": r.method,
        "ess": r.ess,
    }


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
    `out/store/interp/likelihood/<run_id>.jsonl`.
    """

    def _int_cfg(key: str, default: int) -> int:
        v = cfg.get(key)
        return default if v is None else int(cast(float, v))

    seed = _int_cfg("seed", 0)
    cap = _int_cfg("cap", DEFAULT_CAP)
    m = _int_cfg("m", DEFAULT_M)
    n_neutral = _int_cfg("n_neutral", 2000)
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
    family_items = data.load_family_witnesses(
        store, version=str(family_version) if family_version else "v1"
    )
    if not family_items:
        notes.append("no families suite in the store yet; skipping family witnesses")
    neutral += family_items

    items: list[tuple[str, str, Sequent, str, int | None]] = [
        (p.pid, p.origin, p.sequent, p.body, p.L) for p in neutral
    ]

    model_samples_cfg = cfg.get("model_samples") or {}
    assert isinstance(model_samples_cfg, Mapping)
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
        items += [(f"{origin}:{s.pid}", origin, s.sequent, s.body, None) for s in samples]

    base_success = _load_base_success(store, cfg.get("base_success"))

    rows: list[dict[str, JSON]] = []
    for pid, origin, seq, body, L in items:
        rng = random.Random(f"{seed}:{pid}")
        row: dict[str, JSON] = {
            "pid": pid,
            "origin": origin,
            "L": L,
            "base": _marginal_row(base_adapter, seq, body, cap=cap, m=m, rng=rng),
            "rl": _marginal_row(rl_adapter, seq, body, cap=cap, m=m, rng=rng),
            "p_hat_base": base_success.get(pid),
        }
        if base_adapter is not None and rl_adapter is not None and origin == "rl_verified":
            d = disagreement_stats(base_adapter, rl_adapter, seq, body)
            row["disagreement"] = {
                "n_positions": d.n_positions,
                "n_disagree": d.n_disagree,
                "disagree_frac": d.disagree_frac,
                "mean_base_rank": d.mean_base_rank,
                "mean_base_entropy": d.mean_base_entropy,
            }
        else:
            row["disagreement"] = None
        rows.append(row)

    run_id = new_interp_run(store, "likelihood", cfg)
    write_interp_rows(store, "likelihood", run_id, rows)
    return {
        "run_id": run_id,
        "n_items": len(rows),
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
