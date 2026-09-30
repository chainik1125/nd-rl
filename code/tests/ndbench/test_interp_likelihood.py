"""`interp/likelihood.py`: `marginal_logprob`'s exact and importance-sampling branches (including
the hard requirement that importance sampling matches exact enumeration within tolerance on a
proof with about 50-500 orderings), `disagreement_stats`, and `run_likelihood`'s end-to-end wiring
through a real (synthetic) `Store`.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from code.ndbench.adapters.base import ModelAdapter
from code.ndbench.adapters.fake import FakeAdapter
from code.ndbench.core.io import utc_now_iso
from code.ndbench.core.proof import make_reference
from code.ndbench.core.types import (
    JSON,
    LengthLabel,
    LogProb,
    Problem,
    Sequent,
    SuiteManifest,
)
from code.ndbench.interp import likelihood, orderings
from code.ndbench.interp._store import load_interp_rows
from code.ndbench.judge.nd import NDJudge
from code.ndbench.runner.runner import RunSpec, run
from code.ndbench.store import Store

from ._fixtures import load_valid_proofs
from .test_wp5_fixtures import TinyGptAdapter


class _ScoreAdapter:
    """`ModelAdapter`-shaped test double exposing only `logprob`: a fixed, deterministic
    body -> nats lookup, so the "true" marginal (log-sum-exp over every known body) is known
    exactly and independent of any model internals. Deliberately does not inherit `ModelAdapter`
    (an ABC, not a Protocol -- that would force stub `sample`/`token_count`/`identity`
    implementations this test never calls); `_adapter()` below `cast`s it at the point of use."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def logprob(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        out: list[LogProb] = []
        for _seq, body in pairs:
            nats = self._scores[body]
            out.append(LogProb(nats=nats, n_tokens=1, per_encoding=(nats,)))
        return out


def _adapter(scores: dict[str, float]) -> ModelAdapter:
    return cast(ModelAdapter, _ScoreAdapter(scores))


def _pseudo_score(body: str, lo: float = -8.0, spread: float = 6.0) -> float:
    """A deterministic (across processes/`PYTHONHASHSEED`) pseudo-random score in
    `[lo, lo+spread)`."""
    h = int(hashlib.sha256(body.encode()).hexdigest()[:8], 16)
    return lo + spread * (h % 100_000) / 100_000


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


# ------------------------------------------------------------------------------ marginal_logprob


def test_marginal_logprob_single_ordering_is_exact_and_trivial() -> None:
    """No reordering freedom: `logp_marginal == logp_single` exactly, `method == "single"`."""
    seq = Sequent(premises=("P",), conclusion="P")
    body = "N1 P : PR ; N2 P : R N1 ; QED"
    assert orderings.count_orderings(body) == 1
    adapter = _adapter({body: -3.5})
    r = likelihood.marginal_logprob(adapter, seq, body)
    assert r.method == "single"
    assert r.n_orderings == 1
    assert r.logp_single == r.logp_marginal == -3.5
    assert r.ess is None


def test_marginal_logprob_exact_matches_hand_logsumexp() -> None:
    """A proof with a small (6) number of orderings: the exact branch must equal a hand-computed
    log-sum-exp over every one of `orderings.enumerate_orderings`' outputs."""
    seq = Sequent(premises=(), conclusion="( ( P > P ) & ( Q > Q ) )")
    body = (
        "N1 | P : AS ; N2 | P : R N1 ; N3 ( P > P ) : IMPI N1 N2 ; "
        "N4 | Q : AS ; N5 | Q : R N4 ; N6 ( Q > Q ) : IMPI N4 N5 ; "
        "N7 ( ( P > P ) & ( Q > Q ) ) : ANDI N3 N6 ; QED"
    )
    all_orderings = orderings.enumerate_orderings(body, 100)
    assert len(all_orderings) == 6
    scores = {b: _pseudo_score(b) for b in all_orderings}
    adapter = _adapter(scores)
    r = likelihood.marginal_logprob(adapter, seq, body, cap=100)
    assert r.method == "exact"
    assert r.n_orderings == 6
    assert r.logp_single == scores[body]
    assert math.isclose(r.logp_marginal, _logsumexp(list(scores.values())), rel_tol=1e-9)


def test_marginal_logprob_importance_matches_exact_within_tolerance() -> None:
    """The hard requirement: on a proof with about 50-500 orderings (here 70, two independent
    4-line chains interleaving as C(8,4)), forcing the importance-sampling branch (via a `cap`
    below the true count) must match the true (independently, exactly computed) marginal within a
    generous but meaningful tolerance."""
    seq = Sequent(premises=("P", "Q"), conclusion="( P & Q )")
    chain_a = "N3 P : R N1 ; N4 P : R N3 ; N5 P : R N4 ; N6 P : R N5 ;"
    chain_b = "N7 Q : R N2 ; N8 Q : R N7 ; N9 Q : R N8 ; N10 Q : R N9 ;"
    body = f"N1 P : PR ; N2 Q : PR ; {chain_a} {chain_b} N11 ( P & Q ) : ANDI N6 N10 ; QED"
    assert orderings.count_orderings(body) == 70

    all_orderings = orderings.enumerate_orderings(body, 1000)
    assert len(all_orderings) == 70
    scores = {b: _pseudo_score(b, lo=-6.0, spread=3.0) for b in all_orderings}
    true_marginal = _logsumexp(list(scores.values()))
    adapter = _adapter(scores)

    r = likelihood.marginal_logprob(adapter, seq, body, cap=10, m=4000, rng=random.Random(42))
    assert r.method == "importance"
    assert r.n_orderings == 70  # count_orderings still succeeds; only cap forces sampling
    assert r.ess is not None and r.ess > 0
    assert math.isclose(r.logp_marginal, true_marginal, abs_tol=0.3), (
        r.logp_marginal,
        true_marginal,
    )


def test_marginal_logprob_importance_ess_and_determinism() -> None:
    seq = Sequent(premises=("P", "Q"), conclusion="( P & Q )")
    chain_a = "N3 P : R N1 ; N4 P : R N3 ; N5 P : R N4 ; N6 P : R N5 ;"
    chain_b = "N7 Q : R N2 ; N8 Q : R N7 ; N9 Q : R N8 ; N10 Q : R N9 ;"
    body = f"N1 P : PR ; N2 Q : PR ; {chain_a} {chain_b} N11 ( P & Q ) : ANDI N6 N10 ; QED"
    all_orderings = orderings.enumerate_orderings(body, 1000)
    scores = {b: _pseudo_score(b) for b in all_orderings}
    adapter = _adapter(scores)
    r1 = likelihood.marginal_logprob(adapter, seq, body, cap=5, m=200, rng=random.Random(0))
    r2 = likelihood.marginal_logprob(adapter, seq, body, cap=5, m=200, rng=random.Random(0))
    assert r1.logp_marginal == r2.logp_marginal
    assert r1.ess is not None and 0 < r1.ess <= 200


# --------------------------------------------------------------------------- disagreement_stats


def test_disagreement_stats_shapes_and_ranges() -> None:
    row = load_valid_proofs()[0]
    seq = Sequent.from_prompt(row["prompt"])
    body = row["proof"]
    base = TinyGptAdapter(seed=1, label="base")
    rl = TinyGptAdapter(seed=2, label="rl")
    d = likelihood.disagreement_stats(base, rl, seq, body)
    assert d.n_positions > 0
    assert 0 <= d.n_disagree <= d.n_positions
    assert 0.0 <= d.disagree_frac <= 1.0
    assert len(d.base_rank_of_rl_argmax) == d.n_disagree
    assert len(d.base_entropy_at_disagree) == d.n_disagree
    assert all(r >= 0 for r in d.base_rank_of_rl_argmax)
    assert all(e >= 0.0 for e in d.base_entropy_at_disagree)
    if d.n_disagree:
        assert d.mean_base_rank is not None
        assert d.mean_base_entropy is not None
    else:
        assert d.mean_base_rank is None


def test_disagreement_stats_identical_models_never_disagree() -> None:
    """Scoring the exact same (weights-sharing) adapter against itself must find 0 disagreements."""
    row = load_valid_proofs()[1]
    seq = Sequent.from_prompt(row["prompt"])
    body = row["proof"]
    adapter = TinyGptAdapter(seed=3)
    d = likelihood.disagreement_stats(adapter, adapter, seq, body)
    assert d.n_disagree == 0
    assert d.disagree_frac == 0.0


# ------------------------------------------------------------------------------- run_likelihood


def _build_tiny_suite(store: Store, rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    problems = [
        Problem(
            pid=row["name"],
            suite="tinysuite",
            version="v1",
            sequent=Sequent.from_prompt(row["prompt"]),
            length=LengthLabel(
                L=row["n_lines"], L_ub=row["n_lines"], L_lb=None, method="reference", bound=None
            ),
            ref=make_reference(row["proof"], source="minlen"),
            tags={},
        )
        for row in rows
    ]
    manifest = SuiteManifest(
        suite="tinysuite",
        version="v1",
        builder="test_interp_likelihood",
        config={},
        seed=0,
        created_utc=utc_now_iso(),
        git_commit="test",
        n_items=len(problems),
        items_sha256="",
        excl_sources={},
        stats={},
    )
    store.write_suite(problems, manifest)
    return {p.pid: p.ref.body for p in problems if p.ref is not None}


def test_run_likelihood_end_to_end(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    rows = load_valid_proofs()[:3]
    refs = _build_tiny_suite(store, rows)

    fake = FakeAdapter(refs, p=1.0)
    for model in ("base", "rl"):
        spec = RunSpec(suite="tinysuite", version="v1", model=model, k=1, keep_samples=True)
        summary = run(spec, fake, NDJudge(), store)
        assert summary["c"] == len(rows)  # p=1.0: every attempt is the (verified) reference

    cfg: dict[str, JSON] = {
        "seed": 0,
        "n_neutral": 0,  # skip the real reserve/sealed files entirely for this unit test
        "model_samples": {
            "base": {"suite": "tinysuite", "version": "v1", "model": "base", "limit": 10},
            "rl": {"suite": "tinysuite", "version": "v1", "model": "rl", "limit": 10},
        },
    }
    base_adapter = TinyGptAdapter(seed=1, label="base")
    rl_adapter = TinyGptAdapter(seed=2, label="rl")
    summary = likelihood.run_likelihood(
        store, cfg, base_adapter=base_adapter, rl_adapter=rl_adapter
    )
    assert summary["n_neutral"] == 0
    assert summary["n_items"] == 2 * len(rows)  # base_verified + rl_verified, 3 pids each

    loaded = load_interp_rows(store, "likelihood", summary["run_id"])
    assert len(loaded) == summary["n_items"]
    origins = {cast(str, r["origin"]) for r in loaded}
    assert origins == {"base_verified", "rl_verified"}
    for r in loaded:
        base, rl = cast("dict[str, JSON]", r["base"]), cast("dict[str, JSON]", r["rl"])
        assert isinstance(base["logp_marginal"], float)
        assert isinstance(rl["logp_marginal"], float)
        if r["origin"] == "rl_verified":
            disagreement = r["disagreement"]
            assert disagreement is not None
            disagree_frac = cast(float, cast("dict[str, JSON]", disagreement)["disagree_frac"])
            assert 0.0 <= disagree_frac <= 1.0
        else:
            assert r["disagreement"] is None


def test_run_likelihood_notes_missing_model_samples(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    rows = load_valid_proofs()[:2]
    _build_tiny_suite(store, rows)
    cfg: dict[str, JSON] = {
        "n_neutral": 0,
        "model_samples": {
            "base": {"suite": "tinysuite", "version": "v1", "model": "does-not-exist"}
        },
    }
    adapter = TinyGptAdapter(seed=5)
    summary = likelihood.run_likelihood(store, cfg, base_adapter=adapter, rl_adapter=None)
    assert summary["n_items"] == 0
    assert any("does-not-exist" in n or "model_samples" in n for n in summary["notes"])
