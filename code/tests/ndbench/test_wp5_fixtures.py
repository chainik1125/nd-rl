"""WP5 (EDL) shared CPU test doubles.

No production `Learner`/`FINETUNE`-capable `ModelAdapter` is committed yet (WP1 ships the real
`AbsGptAdapter` in parallel), so this module builds the two the WP5 brief asks for and shares them
across `test_wp5_*.py`:

* `TinyGptAdapter` / `TinyGptLearner`: a genuinely-trainable CPU `ModelAdapter`/`Learner` built
  directly from the vendored `absgpt/{model,tokenizer}.py` classes (1 transformer block, `d=32`),
  implementing the shift-marginal NLL (`code.ndbench.edl`'s module docstring, note 2's optional
  `nll_start_excluded` hook included) the same way `external/nd-takehome/novelty.py`'s `score()`
  does.
* `ConstantNllAdapter` / `ConstantNllLearner`: a `Learner` whose `nll()` never changes regardless
  of training, realising plan §B7's "null control" sanity check (EDL should be exactly 0 when the
  per-example loss never moves) without needing randomness or a real model.

This file matches the `test_wp5_*.py` glob WP5's own test run picks up
(`pytest code/tests/ndbench/test_wp5_*.py`) but defines no `test_*` functions of its own beyond one
light sanity check that these doubles satisfy the `ModelAdapter`/`Learner` shapes other WP5 tests
rely on.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from code.ndbench.adapters.base import Capability, Learner, ModelAdapter, NotSupported
from code.ndbench.core.types import JSON, Attempt, LogProb, SampleRequest, Sequent
from code.ndbench.vendor import load_absgpt_module

_model_mod = load_absgpt_module("model")
_tok_mod = load_absgpt_module("tokenizer")
MAXN: int = _tok_mod.MAXN


def new_tokenizer() -> Any:
    """`vendor/absgpt/tokenizer.py`'s `Tokenizer` is loaded dynamically (`load_absgpt_module`), so
    it has no static type; every `tok`-typed parameter in this module is `Any` for the same
    reason."""
    return _tok_mod.Tokenizer(mode="abs")


def new_tiny_model(vocab: int, *, seed: int, n_layer: int = 1, d: int = 32, n_head: int = 4) -> Any:
    """A fresh, randomly-initialised tiny GPT (1 layer, d=32 by default), per the WP5 brief.
    `vendor/absgpt/model.py`'s `GPT` is likewise loaded dynamically, so its instances are `Any`
    here too (not `torch.nn.Module`: that would make pyright resolve GPT-specific methods like
    `n_params()` through `Module`'s dynamic-attribute stub instead, which is worse, not better)."""
    torch.manual_seed(seed)
    model = _model_mod.GPT(vocab=vocab, n_layer=n_layer, d=d, n_head=n_head, max_len=256)
    model.eval()
    return model


def _encode(tok: Any, seq: Sequent, body: str) -> tuple[list[int], list[int]]:
    return tok.encode_prompt(seq.prompt()), tok.encode_proof(body)


def _max_index(tok: Any, qid: list[int]) -> int:
    ref0 = tok.ref0
    return max((x - ref0 + 1 for x in qid if x >= ref0), default=0)


def _shift(tok: Any, qid: list[int], s: int) -> list[int]:
    ref0 = tok.ref0
    return [x + s if x >= ref0 else x for x in qid]


def score_all_shifts(
    model: Any, tok: Any, seq: Sequent, body: str, *, maxn: int, start_excluded: bool = False
) -> tuple[list[float], int]:
    """Teacher-forced log p(proof | prompt) at every start-index shift the tokenizer allows, as a
    list of per-shift totals (`external/nd-takehome/novelty.py`'s `score()`, batched over shifts
    since every shift of one proof re-encodes to the same length). `code.ndbench.edl.run_edl`'s
    `nll`/`nll_start_excluded` protocol expects `-logsumexp(this)` as the loss; `ModelAdapter.
    logprob` expects `+logsumexp(this)` (see `edl.py`'s module docstring on the sign convention).

    `start_excluded`: drop the first proof token (the start-index token itself) from every shift's
    summed log-probability before returning -- plan §B7's sensitivity-check variant.
    """
    pid, qid = _encode(tok, seq, body)
    mx = _max_index(tok, qid)
    rows = [pid + _shift(tok, qid, s) for s in range(0, maxn - mx + 1)]
    x = torch.tensor(rows, dtype=torch.long)
    with torch.no_grad():
        logits = model(x[:, :-1]).float()
    tgt = x[:, 1:]
    logp = F.log_softmax(logits, dim=-1).gather(-1, tgt[..., None]).squeeze(-1)
    proof_logp = logp[:, len(pid) - 1 :]
    n_tok = len(qid)
    if start_excluded:
        proof_logp = proof_logp[:, 1:]
        n_tok -= 1
    return proof_logp.sum(dim=1).tolist(), n_tok


class TinyGptLearner:
    """CPU `Learner` (see `adapters.base.Learner`) built from a private copy of a tiny GPT."""

    def __init__(
        self, model: Any, tok: Any, maxn: int, *, seed: int, update_shifts: int = 8
    ) -> None:
        self._model = model
        self._tok = tok
        self._maxn = maxn
        self._rng = random.Random(seed)
        #: Public, settable (see `adapters.base.Learner.update`'s docstring and `edl.py`'s module
        #: docstring note 1): `code.ndbench.edl.run_edl` sets this to its own `shifts_per_update`
        #: before training, and calls `update()` with one pair per example, relying on this
        #: attribute -- not a repeated-pair list -- to produce the random-shift copies.
        self.update_shifts = update_shifts
        self._opt = torch.optim.AdamW(
            self._model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0
        )

    def nll(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        self._model.eval()
        out = []
        for seq, body in pairs:
            per_shift, n_tok = score_all_shifts(self._model, self._tok, seq, body, maxn=self._maxn)
            nats = -torch.logsumexp(torch.tensor(per_shift), 0).item()
            out.append(LogProb(nats=nats, n_tokens=n_tok, per_encoding=tuple(per_shift)))
        return out

    def nll_start_excluded(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        """Optional extension `code.ndbench.edl.run_edl` looks for via `getattr` (its module
        docstring, note 2): the same computation as `nll`, minus the start-index token."""
        self._model.eval()
        out = []
        for seq, body in pairs:
            per_shift, n_tok = score_all_shifts(
                self._model, self._tok, seq, body, maxn=self._maxn, start_excluded=True
            )
            nats = -torch.logsumexp(torch.tensor(per_shift), 0).item()
            out.append(LogProb(nats=nats, n_tokens=n_tok, per_encoding=tuple(per_shift)))
        return out

    def update(self, pairs: Sequence[tuple[Sequent, str]], lr: float) -> float:
        """One AdamW step on `pairs`, each contributing `self.update_shifts` independently
        re-shifted copies (plan §B7's "8 random-shift copies"; `edl.py`'s module docstring note 1:
        `run_edl` passes each example once per call and relies on this internal loop, matching
        `adapters.absgpt._AbsGptLearner.update`). Rows may have different lengths, so this pads to
        the batch's longest row and masks accordingly."""
        self._model.train()
        for g in self._opt.param_groups:
            g["lr"] = lr
        rows: list[list[int]] = []
        prompt_lens: list[int] = []
        for seq, body in pairs:
            pid, qid = _encode(self._tok, seq, body)
            mx = _max_index(self._tok, qid)
            for _ in range(self.update_shifts):
                s = self._rng.randint(0, self._maxn - mx) if mx > 0 else 0
                rows.append(pid + _shift(self._tok, qid, s))
                prompt_lens.append(len(pid))
        maxlen = max(len(r) for r in rows)
        padded = [r + [self._tok.pad] * (maxlen - len(r)) for r in rows]
        x = torch.tensor(padded, dtype=torch.long)
        logits = self._model(x[:, :-1])
        tgt = x[:, 1:]
        logp = F.log_softmax(logits, dim=-1).gather(-1, tgt[..., None]).squeeze(-1)
        mask = torch.zeros_like(tgt, dtype=torch.bool)
        for i, (row, plen) in enumerate(zip(rows, prompt_lens, strict=True)):
            mask[i, plen - 1 : len(row) - 1] = True
        pre_loss = -(logp * mask).sum() / mask.sum().clamp(min=1)
        self._opt.zero_grad()
        pre_loss.backward()
        self._opt.step()
        self._model.eval()
        return float(pre_loss.item())

    def adapter(self) -> ModelAdapter:
        return TinyGptAdapter(
            model=self._model, tok=self._tok, maxn=self._maxn, label="tiny_gpt_trained"
        )


class TinyGptAdapter(ModelAdapter):
    """CPU `ModelAdapter` test double: `SAMPLE | LOGPROB | FINETUNE`, built from the vendored
    absgpt `GPT`/`Tokenizer` classes. Not production code (WP1 ships the real `AbsGptAdapter`);
    exists only so WP5's tests can exercise `code.ndbench.edl.run_edl` end to end on CPU."""

    def __init__(
        self,
        *,
        n_layer: int = 1,
        d: int = 32,
        n_head: int = 4,
        seed: int = 0,
        label: str = "tiny_gpt",
        model: Any = None,
        tok: Any = None,
        maxn: int | None = None,
    ) -> None:
        # `cli.py`'s `--model-arg key=value` always produces string values (see `edl.py`'s module
        # docstring for other such CLI quirks this WP works around), so coerce defensively --
        # `int(x)` is a no-op when `x` is already an `int` (the direct, non-CLI construction path).
        seed, n_layer, d, n_head = int(seed), int(n_layer), int(d), int(n_head)
        self.tok = tok if tok is not None else new_tokenizer()
        self._maxn = maxn if maxn is not None else MAXN
        self.model = (
            model
            if model is not None
            else new_tiny_model(self.tok.vocab_size, seed=seed, n_layer=n_layer, d=d, n_head=n_head)
        )
        self.model.eval()
        self.label = label
        self.format_id = "nd_v1"
        self.capabilities = Capability.SAMPLE | Capability.LOGPROB | Capability.FINETUNE
        self._seed = seed

    def identity(self) -> dict[str, JSON]:
        return {
            "label": self.label,
            "format_id": self.format_id,
            "seed": self._seed,
            "n_params": self.model.n_params(),
        }

    def token_count(self, seq: Sequent, body: str | None = None) -> int:
        n = len(self.tok.encode_prompt(seq.prompt()))
        if body:
            n += len(self.tok.encode_proof(body))
        return n

    def sample(self, reqs: Sequence[SampleRequest]) -> list[list[Attempt]]:
        out: list[list[Attempt]] = []
        for req in reqs:
            pid = self.tok.encode_prompt(req.sequent.prompt())
            attempts: list[Attempt] = []
            for i in range(req.n):
                g = torch.Generator().manual_seed((req.seed + i) % (2**63 - 1))
                ids = list(pid)
                for _ in range(max(1, req.max_new_tokens)):
                    x = torch.tensor([ids], dtype=torch.long)
                    with torch.no_grad():
                        logits = self.model(x)[0, -1] / max(req.temperature, 1e-6)
                    probs = torch.softmax(logits, dim=-1)
                    nxt = int(torch.multinomial(probs, 1, generator=g).item())
                    ids.append(nxt)
                    if nxt == self.tok.eos:
                        break
                body_ids = ids[len(pid) :]
                finished = bool(body_ids) and body_ids[-1] == self.tok.eos
                try:
                    body = self.tok.decode(body_ids)
                except Exception as e:  # pragma: no cover - decode is total in practice
                    body = f"UNPARSEABLE {e}"
                    finished = False
                attempts.append(
                    Attempt(
                        pid=req.pid,
                        i=i,
                        body=body,
                        native=None,
                        n_tokens=len(body_ids),
                        finished=finished,
                    )
                )
            out.append(attempts)
        return out

    def logprob(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        out = []
        for seq, body in pairs:
            per_shift, n_tok = score_all_shifts(self.model, self.tok, seq, body, maxn=self._maxn)
            nats = torch.logsumexp(torch.tensor(per_shift), 0).item()
            out.append(LogProb(nats=nats, n_tokens=n_tok, per_encoding=tuple(per_shift)))
        return out

    def learner(self, *, seed: int) -> TinyGptLearner:
        """Narrower than `ModelAdapter.learner`'s declared `-> Learner` (a legal covariant
        override): lets callers that know they hold a `TinyGptAdapter` reach `TinyGptLearner`'s
        extra `nll_start_excluded` method directly, without `run_edl`'s `getattr` duck-typing."""
        model_copy = copy.deepcopy(self.model)
        model_copy.train()
        return TinyGptLearner(model_copy, self.tok, self._maxn, seed=seed)


class ConstantNllLearner:
    """`Learner` whose `nll()` always returns the same value regardless of weights or update
    history -- plan §B7's null-control sanity check realised deterministically (a model that has
    already fully learned the class gains nothing further from training, so every online loss
    equals the final test loss and `EDL` is exactly 0)."""

    def __init__(self, c: float, adapter: ModelAdapter) -> None:
        self._c = c
        self._adapter = adapter

    def nll(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        return [LogProb(nats=self._c, n_tokens=1, per_encoding=(-self._c,)) for _ in pairs]

    def nll_start_excluded(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        return self.nll(pairs)

    def update(self, pairs: Sequence[tuple[Sequent, str]], lr: float) -> float:
        del pairs, lr
        return self._c

    def adapter(self) -> ModelAdapter:
        return self._adapter


class ConstantNllAdapter(ModelAdapter):
    """`FINETUNE`-capable `ModelAdapter` whose `learner()` is a `ConstantNllLearner`."""

    def __init__(self, *, c: float, label: str = "constant") -> None:
        self.label = label
        self.format_id = "const_v1"
        self.capabilities = Capability.FINETUNE
        self._c = c

    def identity(self) -> dict[str, JSON]:
        return {"label": self.label, "c": self._c}

    def token_count(self, seq: Sequent, body: str | None = None) -> int:
        n = len(seq.prompt().split())
        if body:
            n += len(body.split())
        return n

    def sample(self, reqs: Sequence[SampleRequest]) -> list[list[Attempt]]:
        raise NotSupported("ConstantNllAdapter does not sample")

    def learner(self, *, seed: int) -> Learner:
        del seed
        return ConstantNllLearner(self._c, self)


def make_finetune_registry_entry(cls: type[ModelAdapter]) -> Callable[[], type[ModelAdapter]]:
    """Not used directly; kept for symmetry/documentation of how `test_wp5_cli.py` registers these
    test doubles into `adapters.base.ADAPTERS` at runtime (a dict mutation in the test module
    itself, not an edit to `adapters/base.py`)."""
    return lambda: cls


def test_tiny_gpt_adapter_satisfies_the_model_adapter_contract() -> None:
    """Sanity check that these WP5 test doubles actually shape up as `ModelAdapter`/`Learner`
    (constructing a `TinyGptAdapter`/`TinyGptLearner` and calling every Protocol method once is
    itself a useful, fast regression test of these fixtures)."""
    adapter = TinyGptAdapter(seed=0)
    assert Capability.FINETUNE in adapter.capabilities
    seq = Sequent(premises=("P",), conclusion="P")
    body = "N1 P : PR ; QED"
    identity = adapter.identity()
    assert identity["label"] == "tiny_gpt"
    lp = adapter.logprob([(seq, body)])[0]
    assert isinstance(lp.nats, float)

    learner = adapter.learner(seed=1)
    assert learner.update_shifts == 8  # the default `shifts_per_update` (edl.py's DEFAULT_...)
    nll_before = learner.nll([(seq, body)])[0]
    assert nll_before.nats >= 0.0  # Learner.nll is a positive NLL, not a log-probability
    learner.update([(seq, body)], 1e-2)  # one pair; the learner reshifts it internally
    nll_after = learner.nll([(seq, body)])[0]
    assert isinstance(nll_before.nats, float)
    assert isinstance(nll_after.nats, float)
    se = learner.nll_start_excluded([(seq, body)])[0]
    assert se.n_tokens == nll_before.n_tokens - 1

    const_adapter = ConstantNllAdapter(c=5.0)
    const_learner = const_adapter.learner(seed=0)
    assert const_learner.nll([(seq, body)])[0].nats == 5.0
    assert const_learner.update([(seq, body)], 1e-3) == 5.0
