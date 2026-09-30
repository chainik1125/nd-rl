"""`LeanSeqAdapter`: a `ModelAdapter` over the vendored `lean_seq` tokenizer (plan §1/§3 WP7).

`LeanSeqAdapter` takes a generic `policy` callable for token-id generation, so *any* future model
trained on the `lean_seq` rendering (Dan's fork, or anything else emitting the same 107-token
vocabulary) plugs in without this module knowing anything about its architecture. It decodes a
policy's raw ids to a canonical ND body via the vendored `inverse` parser (through
`LeanSeqTokenizer.decode`), or to `"UNPARSEABLE <why>"` if the ids are outside the strict
`lean_seq` grammar -- exactly the `Attempt.body` contract every other adapter/judge in this package
already expects, so `judge.nd.NDJudge` (and everything built on it) works unchanged on samples from
a Lean-format model, per `vendor/lean_seq/lean_seq.py`'s own module docstring.

**The `policy` contract.** `policy(prompt_ids, forced_ids, *, n, temperature, max_new_tokens, seed)
-> list[list[int]]` returns `n` freshly generated id sequences (the *continuation only* -- never
including `prompt_ids`/`forced_ids`), each either ending in `LeanSeqTokenizer.eos_id` (the model
finished) or exactly `max_new_tokens` ids with no `eos_id` (the budget ran out). `forced_ids` is the
`lean_seq` rendering of `SampleRequest.prefix` (empty when there is none); a real model should treat
`prompt_ids + forced_ids` as one fixed context to continue from (teacher-forced replay of the
prefix), the token-format analogue of forcing the first `j` lines. `seed` is per-`(request, chunk)`
(see `runner.runner._item_seed`); a deterministic `policy` given the same arguments must return the
same ids, so a `LeanSeqAdapter` run is exactly as reproducible as `adapters.fake.FakeAdapter`.

**Prefix rendering.** A `PrefixSpec` is an ND body (first `j` lines, N1-based, no `QED`); turning it
into `forced_ids` needs a *partial* `lean_seq` rendering, which `vendor/lean_seq/lean_seq.py` does
not expose (`proof_tokens` requires a complete, fully-boxed proof and is not usable piecewise, and
its `close`/`box`/`need` helpers are private closures, not module-level functions). This module
therefore reimplements the same line-processing loop as `proof_tokens` in `_partial_proof_tokens`
below (mirroring it exactly; the vendored file itself is never modified, per the plan), differing
only in what happens once the prefix's lines run out: no proof is required to end at depth 0 (a
prefix may end mid-box), and no final `exact <last line>` is appended (there is no "last line" of a
partial proof). This is not an approximation: in `lean_seq`'s rendering, a box's content is only
ever spliced into the token stream at the position of the line that *discharges* it (`IMPI`/`NEGI`/
`ORE`), nested inside that line's term -- never at the box's own `AS` line. So if the ND prefix ends
while a box is still open, none of that box's lines would appear in a *complete* proof's rendering
either, until the discharging line is reached; returning just the closed top-level tokens so far is
the exact, faithful partial rendering, not a shortcut. See
`code/tests/ndbench/test_wp7_lean_seq.py` for a check against `proof_tokens` itself at `j = len(all
lines)` (where the two must agree, since a complete proof always ends at depth 0).

**LOGPROB.** `logprob()` needs a `scorer(prompt_ids, proof_ids) -> float` callable (total NLL, in
nats, of `proof_ids` conditioned on `prompt_ids`) injected at construction; without one it raises
`NotSupported`, mirroring the base class's own default. Marginalisation is over every valid *name*
shift (`lean_seq.LeanSeqTokenizer.shift_names`'s augmentation group: add one offset to every `n1`..
`n64` id, keeping them in range) rather than absgpt's line-index shifts -- the `lean_seq` analogue
of "normalise the output format" (plan §B7): `nats = -logsumexp_s(-scorer(prompt_ids,
shift_s(proof_ids)))`, `per_encoding[s]` holding each shift's raw NLL.

**FINETUNE.** `learner()` needs a `learner_factory(seed) -> Learner` injected at construction;
without one it raises `NotSupported`, same pattern as `logprob`.

**`from_checkpoint`.** Raises `judge.base.NotAvailable` (reused here for "not usable in this
environment", the same semantics the plan asks a future `LeanJudge` to use for a missing `lean`
binary): there is no lean_seq-trained model class vendored into this worktree yet. What is needed to
implement it for real: (1) the model class from Dan's fork `dan-pandori/nd-takehome` that trains on
the `lean_seq` tokenizer mode (see `vendor/lean_seq/lean_seq.py`'s module docstring for the mode
name and the exact commit this tokenizer was ported from); (2) a checkpoint from Dan's Hugging Face
bucket (repo id/revision not recorded in this worktree -- ask Dan). Once both exist, load the
checkpoint with that model class, wrap its batched id-generation as a `policy` callable matching the
contract above (and, optionally, its teacher-forced loss as a `scorer` and its optimiser step as a
`learner_factory`), and construct `LeanSeqAdapter(policy=..., scorer=..., learner_factory=...)`
directly; `from_checkpoint` can then become a thin convenience wrapper around exactly that.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from ..core.types import JSON, Attempt, LogProb, SampleRequest, Sequent
from ..judge.base import NotAvailable
from ..vendor.lean_seq.lean_seq import (
    MAX_NAMES,
    Formula,
    LeanParseError,
    LeanSeqTokenizer,
    Name,
    NDLine,
    NDParseError,
    Token,
    lean_formula_tokens,
    parse_proof,
)
from .base import Capability, Learner, ModelAdapter, NotSupported

#: `policy(prompt_ids, forced_ids, *, n, temperature, max_new_tokens, seed) -> list[list[int]]`.
#: See the module docstring for the exact contract.
PolicyFn = Callable[..., list[list[int]]]
#: `scorer(prompt_ids, proof_ids) -> float`: total NLL (nats) of `proof_ids` given `prompt_ids`.
ScorerFn = Callable[[Sequence[int], Sequence[int]], float]
#: `learner_factory(seed) -> Learner`.
LearnerFactory = Callable[[int], Learner]

_UNPARSEABLE = "UNPARSEABLE"
_LEANPARSE_PREFIX = "LEANPARSE"


def _partial_proof_tokens(body: str) -> list[Token]:
    """Lean tokens for a possibly-*incomplete* ND proof (no closing `QED` required).

    Mirrors `lean_seq.proof_tokens`'s per-line loop (see the module docstring for why that
    function itself is not reusable for a prefix). Returns only the top-level tokens rendered so
    far; a still-open box's lines contribute nothing yet, matching `lean_seq`'s own deferred-box
    rendering (a box's content is spliced in only at its discharging line).

    `parse_proof` stops at the first literal `QED` token, so appending one to `body.split()` is
    always safe: for an already-complete `body` (one that already ends in `QED`), that real `QED`
    is what stops the parse, and this function's own extra trailing token is never reached; for an
    incomplete prefix (the intended input here), it supplies the `QED` `parse_proof` requires to
    terminate.
    """
    lines = parse_proof([*body.split(), "QED"])
    stack: list[list[Token]] = [[]]
    hyps: list[tuple[int, Formula] | None] = [None]
    lasts: list[int | None] = [None]
    closed: dict[int, tuple[Formula, list[Token], int | None]] = {}

    def close() -> None:
        body_toks = stack.pop()
        hyp = hyps.pop()
        last = lasts.pop()
        if hyp is None:
            raise NDParseError("unbalanced boxes")
        closed[hyp[0]] = (hyp[1], body_toks, last)

    def box(start: int, end: int, neg: bool = False) -> list[Token]:
        if start not in closed or closed[start][2] != end:
            raise NDParseError(f"box N{start}-N{end} is not a closed box")
        hf, body_toks, _ = closed[start]
        head: list[Token] = [
            "(",
            "fun",
            "(",
            Name(start),
            ":",
            *lean_formula_tokens(hf),
            ")",
            "=>",
            "by",
        ]
        ending: list[Token] = (
            ["exact", "(", Name(end), ":", "False", ")"] if neg else ["exact", Name(end)]
        )
        return [*head, *body_toks, *ending, ")"]

    def need(ln: NDLine, k: int) -> tuple[int, ...]:
        if len(ln.refs) != k:
            raise NDParseError(f"N{ln.idx}: {ln.rule} cites {len(ln.refs)} lines, expected {k}")
        return ln.refs

    n_premises = 0
    for ln in lines:
        if ln.rule == "AS":
            while len(stack) - 1 >= ln.depth:
                close()
            stack.append([])
            hyps.append((ln.idx, ln.formula))
            lasts.append(ln.idx)
            continue
        while len(stack) - 1 > ln.depth:
            close()
        term: list[Token]
        if ln.rule == "PR":
            n_premises += 1
            term = [f"h{n_premises}"]
        elif ln.rule == "R":
            term = [Name(need(ln, 1)[0])]
        elif ln.rule == "ANDI":
            a, b = need(ln, 2)
            term = ["⟨", Name(a), ",", Name(b), "⟩"]
        elif ln.rule in ("ANDE1", "ANDE2"):
            term = [Name(need(ln, 1)[0]), ".1" if ln.rule == "ANDE1" else ".2"]
        elif ln.rule == "IMPE":
            a, b = need(ln, 2)
            term = [Name(a), Name(b)]
        elif ln.rule == "NEGE":
            a, b = need(ln, 2)
            term = [Name(b), Name(a)]
        elif ln.rule in ("ORI1", "ORI2"):
            term = ["Or.inl" if ln.rule == "ORI1" else "Or.inr", Name(need(ln, 1)[0])]
        elif ln.rule == "BOTE":
            term = [Name(need(ln, 1)[0]), ".elim"]
        elif ln.rule == "DN":
            a = need(ln, 1)[0]
            term = ["Classical.byContradiction", "(", "fun", "hh", "=>", Name(a), "hh", ")"]
        elif ln.rule in ("IMPI", "NEGI"):
            s, e = need(ln, 2)
            term = box(s, e, neg=ln.rule == "NEGI")
        elif ln.rule == "ORE":
            j, s1, e1, s2, e2 = need(ln, 5)
            term = ["Or.elim", Name(j), *box(s1, e1), *box(s2, e2)]
        else:
            raise NDParseError(f"unknown rule {ln.rule}")
        stack[-1] += [
            "have",
            Name(ln.idx),
            ":",
            *lean_formula_tokens(ln.formula),
            ":=",
            *term,
            ";",
        ]
        lasts[-1] = ln.idx
    return stack[0]


def _tokens_to_ids(toks: Sequence[Token], tok: LeanSeqTokenizer) -> list[int]:
    """`encode_proof`'s tail: number `Name`s by first appearance, look up everything else."""
    order: dict[int, int] = {}
    out: list[int] = []
    for t in toks:
        if isinstance(t, Name):
            k = order.setdefault(t.idx, len(order) + 1)
            if k > MAX_NAMES:
                raise ValueError(f"more than {MAX_NAMES} names")
            out.append(tok.first_name + k - 1)
        else:
            out.append(tok.stoi[t])
    return out


def _valid_shifts(ids: Sequence[int], tok: LeanSeqTokenizer) -> range:
    """Every shift `s` `shift_names` could have drawn (`rng.randint(0, MAX_NAMES - top)`), as a
    `range`, for exhaustive marginalisation rather than one random draw."""
    top = max((x - tok.first_name + 1 for x in ids if x >= tok.first_name), default=0)
    if top == 0:
        return range(1)  # no names in `ids`: only the trivial (identity) "shift".
    return range(0, MAX_NAMES - top + 1)


def _shift_ids(ids: Sequence[int], tok: LeanSeqTokenizer, s: int) -> list[int]:
    if s == 0:
        return list(ids)
    return [x + s if x >= tok.first_name else x for x in ids]


def _logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    if math.isinf(m):
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


class LeanSeqAdapter(ModelAdapter):
    """A `ModelAdapter` over any policy trained on the `lean_seq` rendering.

    See the module docstring for the `policy`/`scorer`/`learner_factory` contracts and for the
    prefix-rendering caveat.
    """

    def __init__(
        self,
        policy: PolicyFn,
        *,
        scorer: ScorerFn | None = None,
        learner_factory: LearnerFactory | None = None,
        label: str = "lean_seq",
        format_id: str = "lean_seq_v1",
        policy_identity: Mapping[str, JSON] | None = None,
    ) -> None:
        self._policy = policy
        self._scorer = scorer
        self._learner_factory = learner_factory
        self._tok = LeanSeqTokenizer()
        self._policy_identity = dict(policy_identity) if policy_identity is not None else None
        self.label = label
        self.format_id = format_id
        # Declared per the plan's adapter table regardless of whether `scorer`/`learner_factory`
        # were given: `logprob`/`learner` are overridden (they just raise `NotSupported`
        # themselves when unconfigured, exactly like the base class's own defaults) rather than
        # left un-overridden. See the module docstring.
        self.capabilities = (
            Capability.SAMPLE | Capability.PREFIX | Capability.LOGPROB | Capability.FINETUNE
        )

    def identity(self) -> dict[str, JSON]:
        ident: dict[str, JSON] = {
            "label": self.label,
            "format_id": self.format_id,
            "tokenizer_version": LeanSeqTokenizer.VERSION,
        }
        if self._policy_identity is not None:
            ident["policy"] = self._policy_identity
        else:
            ident["policy"] = repr(self._policy)
        return ident

    def token_count(self, seq: Sequent, body: str | None = None) -> int:
        n = len(self._tok.encode_prompt(seq.prompt()))
        if body:
            try:
                n += len(self._tok.encode_proof(body))
            except (NDParseError, LeanParseError, ValueError, KeyError):
                n += len(_tokens_to_ids(_partial_proof_tokens(body), self._tok))
        return n

    def sample(self, reqs: Sequence[SampleRequest]) -> list[list[Attempt]]:
        out: list[list[Attempt]] = []
        for req in reqs:
            prompt_ids = self._tok.encode_prompt(req.sequent.prompt())
            forced_ids: list[int] = []
            if req.prefix is not None:
                try:
                    forced_ids = _tokens_to_ids(_partial_proof_tokens(req.prefix.body), self._tok)
                except (NDParseError, LeanParseError, ValueError, KeyError) as e:
                    raise ValueError(f"cannot render prefix for {req.pid!r}: {e}") from e
            continuations = self._policy(
                prompt_ids,
                forced_ids,
                n=req.n,
                temperature=req.temperature,
                max_new_tokens=req.max_new_tokens,
                seed=req.seed,
            )
            attempts: list[Attempt] = []
            for i, cont in enumerate(continuations):
                proof_ids = [*forced_ids, *cont]
                raw = self._tok.decode(proof_ids)
                if raw.startswith(_LEANPARSE_PREFIX):
                    body = f"{_UNPARSEABLE}{raw[len(_LEANPARSE_PREFIX) :]}"
                    finished = raw != f"{_LEANPARSE_PREFIX} no-eos"
                else:
                    body = raw
                    finished = True
                native = self._tok.lean_text(proof_ids)
                attempts.append(
                    Attempt(
                        pid=req.pid,
                        i=i,
                        body=body,
                        native=native,
                        n_tokens=len(cont),  # freshly generated tokens only; see module docstring
                        finished=finished,
                    )
                )
            out.append(attempts)
        return out

    def logprob(self, pairs: Sequence[tuple[Sequent, str]]) -> list[LogProb]:
        if self._scorer is None:
            raise NotSupported(f"{type(self).__name__} was not given a scorer")
        out: list[LogProb] = []
        for seq, body in pairs:
            prompt_ids = self._tok.encode_prompt(seq.prompt())
            proof_ids = self._tok.encode_proof(body)
            shifts = list(_valid_shifts(proof_ids, self._tok))
            nlls = [self._scorer(prompt_ids, _shift_ids(proof_ids, self._tok, s)) for s in shifts]
            nats = -_logsumexp([-x for x in nlls])
            out.append(LogProb(nats=nats, n_tokens=len(proof_ids), per_encoding=tuple(nlls)))
        return out

    def learner(self, *, seed: int) -> Learner:
        if self._learner_factory is None:
            raise NotSupported(f"{type(self).__name__} was not given a learner_factory")
        return self._learner_factory(seed)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, *, device: str = "cpu") -> LeanSeqAdapter:
        """Not available in this worktree. See the module docstring for what is needed."""
        del ckpt_path, device
        raise NotAvailable(
            "LeanSeqAdapter.from_checkpoint needs a lean_seq-trained model class, which is not "
            "vendored into ndbench: the model code lives on Dan's fork 'dan-pandori/nd-takehome' "
            "(the checkpoint-loading/generation code for its lean_seq tokenizer mode), and "
            "checkpoints live in Dan's Hugging Face bucket (ask Dan for the repo id/revision -- "
            "not recorded in this worktree). Once both exist, wrap that model's batched "
            "generation as a `policy` callable (see this module's docstring for the exact "
            "contract) and construct LeanSeqAdapter(policy=...) directly."
        )
