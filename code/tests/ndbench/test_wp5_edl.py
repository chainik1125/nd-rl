"""`code.ndbench.edl.run_edl`'s core acceptance tests (plan §3's WP5 row):

* a fake (constant-NLL) learner gives `EDL == 0`;
* a tiny CPU GPT on a toy class gives `EDL > 0`, and the online sum of losses is >= `n * L_test`;
* the shift-marginal loss `TinyGptLearner`/`TinyGptAdapter` compute matches an independent,
  from-scratch replica of `external/nd-takehome/novelty.py`'s `score()` math (log-sum-exp over
  every start-index shift) on 5 fixture proofs.

All CPU-only; the whole file runs in well under 60s (tiny model, short proofs, small `n`/`m`).
"""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F

from code.ndbench.core.types import LengthLabel, Problem, ReferenceProof, Sequent
from code.ndbench.edl import run_edl
from code.ndbench.metrics import edl as edl_of

from ._fixtures import load_valid_proofs
from .test_wp5_fixtures import (
    MAXN,
    ConstantNllAdapter,
    TinyGptAdapter,
    new_tiny_model,
    new_tokenizer,
)


def _toy_problems(n_train: int, n_test: int, *, tag_class: str = "toy") -> list[Problem]:
    """A small EDL class built directly from WP0's shared 60-proof fixture (not a real family/law
    suite -- WP3/WP4 own those; this only needs *some* certified, reference-bearing, sufficiently
    varied theorems to exercise `run_edl`'s protocol end to end)."""
    proofs = load_valid_proofs()
    assert n_train + n_test <= len(proofs)
    problems: list[Problem] = []
    for i, proof in enumerate(proofs[: n_train + n_test]):
        split = "train" if i < n_train else "test"
        seq = Sequent.from_prompt(proof["prompt"])
        ref = ReferenceProof(
            body=proof["proof"], lines=proof["n_lines"], pruned=proof["n_lines"], source="minlen"
        )
        problems.append(
            Problem(
                pid=f"toy:{proof['name']}",
                suite="toy",
                version="v1",
                sequent=seq,
                length=LengthLabel(
                    L=proof["n_lines"], L_ub=None, L_lb=None, method="minlen_exact", bound=None
                ),
                ref=ref,
                tags={"class": tag_class, "split": split},
            )
        )
    return problems


def test_constant_nll_learner_gives_zero_edl() -> None:
    """Plan §B7 / paper Proposition 5.1's null control: a "model" whose predictive loss never
    changes gives `EDL == 0` exactly, for every learning rate and data order."""
    problems = _toy_problems(n_train=8, n_test=4)
    adapter = ConstantNllAdapter(c=3.7)
    rows = run_edl(adapter, problems, lrs=(1e-4, 1e-3), orders=2, n=8, m=4, ckpts=(4, 8))
    assert rows
    for row in rows:
        # exactly 0 mathematically; floating-point summation of repeated 3.7s only gets us to
        # within a few ULPs of that, hence `pytest.approx` rather than `==`.
        assert row["EDL"] == pytest.approx(0.0, abs=1e-9)
        assert row["EDL_per_example"] == pytest.approx(0.0, abs=1e-9)
        for edl_np in cast(dict[str, float], row["EDL_by_checkpoint"]).values():
            assert edl_np == pytest.approx(0.0, abs=1e-9)
        # the constant learner's L_test never moves either, so every online loss equals it.
        assert row["L_test_0"] == 3.7
        assert row["L_test_n"] == 3.7
        assert all(x == 3.7 for x in cast(list[float], row["online_nats"]))
        assert row["EDL_start_excluded"] == pytest.approx(0.0, abs=1e-9)


def test_tiny_gpt_gives_positive_edl_and_online_sum_at_least_n_times_l_test() -> None:
    """A freshly-initialised tiny GPT, trained online on a toy class, should show `EDL > 0`: the
    early per-example losses (scored before the model has adapted at all) are typically far worse
    than the final held-out loss once the model has partially fit this narrow class, so the raw
    online sum comfortably exceeds `n * L_test(theta_n)` (plan §3 WP5's acceptance test)."""
    n_train, n_test = 12, 6
    problems = _toy_problems(n_train=n_train, n_test=n_test)
    adapter = TinyGptAdapter(seed=0)
    rows = run_edl(
        adapter,
        problems,
        lrs=(3e-3,),
        orders=1,
        n=n_train,
        m=n_test,
        ckpts=(4, 8, 12),
        shifts_per_update=8,
        seed=0,
    )
    assert len(rows) == 1
    row = rows[0]
    online_nats = cast(list[float], row["online_nats"])
    n_actual = cast(int, row["n"])
    l_test_n = cast(float, row["L_test_n"])
    edl_full = cast(float, row["EDL"])
    assert n_actual == n_train
    assert row["m"] == n_test
    online_sum = sum(online_nats)
    assert online_sum >= n_actual * l_test_n
    assert edl_full > 0.0
    assert edl_full == edl_of(online_nats, l_test_n)
    assert row["EDL_per_example"] == edl_full / n_train
    # the start-token-excluded sensitivity variant is exercised too (TinyGptLearner provides it).
    assert row["EDL_start_excluded"] is not None
    start_excluded_online_nats = row["start_excluded_online_nats"]
    assert start_excluded_online_nats is not None
    assert len(cast(list[float], start_excluded_online_nats)) == n_train


def _novelty_style_logp(model: Any, tok: Any, seq: Sequent, body: str) -> float:
    """Independent, from-scratch replica of `external/nd-takehome/novelty.py`'s `score()`:
    `logp_T1 = logsumexp_s(sum_t log p_theta(shift_s(proof)_t | prompt, shift_s(proof)_<t)))`,
    looped rather than batched (deliberately a different code path than `test_wp5_fixtures.
    score_all_shifts`, which is what `TinyGptLearner`/`TinyGptAdapter` actually use)."""
    pid = tok.encode_prompt(seq.prompt())
    qid = tok.encode_proof(body)
    ref0 = tok.ref0
    mx = max((x - ref0 + 1 for x in qid if x >= ref0), default=0)
    per_shift = []
    for s in range(0, MAXN - mx + 1):
        ids = pid + [x + s if x >= ref0 else x for x in qid]
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(x[:, :-1])
        tgt = x[:, 1:]
        lp = F.log_softmax(logits, dim=-1).gather(-1, tgt[..., None]).squeeze(-1)
        per_shift.append(lp[0, len(pid) - 1 :].sum().item())
    return math.log(sum(math.exp(v - max(per_shift)) for v in per_shift)) + max(per_shift)


def test_shift_marginal_loss_matches_novelty_py_math_on_five_proofs() -> None:
    """`TinyGptAdapter.logprob` (and, negated, `TinyGptLearner.nll`) must equal an independent
    log-sum-exp-over-shifts computation, on 5 fixture proofs, per plan §3 WP5's acceptance test."""
    proofs = load_valid_proofs()[:5]
    tok = new_tokenizer()
    model = new_tiny_model(tok.vocab_size, seed=7)
    adapter = TinyGptAdapter(model=model, tok=tok, seed=7)
    for proof in proofs:
        seq = Sequent.from_prompt(proof["prompt"])
        body = proof["proof"]
        expected = _novelty_style_logp(model, tok, seq, body)
        got_logprob = adapter.logprob([(seq, body)])[0]
        assert got_logprob.nats == pytest.approx(expected, abs=1e-4, rel=1e-4)
        got_nll = adapter.learner(seed=0).nll([(seq, body)])[0]
        assert got_nll.nats == pytest.approx(-expected, abs=1e-4, rel=1e-4)
