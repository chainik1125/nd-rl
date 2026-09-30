"""Core ndbench data model: the dataclasses, Protocols and JSON alias every other work package
builds on. This module implements plan §1's interfaces exactly; other work packages must not
change these shapes (only add new modules that use them).

A `Sequent` never repeats a value the take-home format already assigns meaning to: instead it
takes `code.ndbench.core.keys`/`formula` to derive the spec-format prompt/thm/canon-key strings on
demand. `Problem` is the unit ndbench suites/store/runner/analysis all exchange; `to_json`/
`from_json` are its wire format for `Store` (append-only JSONL under `out/store/`).

Where the plan's pseudocode writes a mutable default (`= {}`), the field instead uses
`dataclasses.field(default_factory=dict)`, since Python dataclasses reject a literal mutable
default; the observable default behaviour (an empty mapping when omitted) is identical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

from .keys import canon_key, join_thm, pkey, prompt_to_thm, split_thm

#: Any JSON-serialisable value: what `Problem.to_json`/`SuiteManifest`/the `Store` files hold.
JSON = str | int | float | bool | None | list["JSON"] | dict[str, "JSON"]


@dataclass(frozen=True)
class Sequent:
    """A theorem to prove: `premises |- conclusion`, each a spec-format formula string.

    `prompt()`/`thm()`/`key()`/`pkey()` are the four string views used throughout the legacy data
    (`prompt` field, `thm` field, `key` field, `pkey` field respectively); `from_prompt`/`from_thm`
    are their inverses.
    """

    premises: tuple[str, ...]
    conclusion: str

    def prompt(self) -> str:
        """`'THM p1 , p2 SEQ c PRF'` (or `'THM SEQ c PRF'` with no premises): the model prompt."""
        if self.premises:
            return f"THM {' , '.join(self.premises)} SEQ {self.conclusion} PRF"
        return f"THM SEQ {self.conclusion} PRF"

    def thm(self) -> str:
        """`'p1 , p2 |- c'` (or `'|- c'`): the legacy `thm` field format."""
        return join_thm(self.premises, self.conclusion)

    def key(self) -> str:
        """Atom-renaming class key of `thm()` (the legacy `key` field)."""
        return canon_key(self.thm())

    def pkey(self) -> str:
        """Premise-order-invariant class key (the legacy `pkey` field): the lexicographically
        smallest `key()` over every permutation of `premises`. O(len(premises)!); the plan's dedup
        policy skips `pkey` matching above 5 premises for this reason."""
        return pkey(self.thm())

    @classmethod
    def from_prompt(cls, s: str) -> Sequent:
        """Parse a `'THM ... SEQ ... PRF'` model prompt."""
        return cls.from_thm(prompt_to_thm(s))

    @classmethod
    def from_thm(cls, s: str) -> Sequent:
        """Parse a `'p1 , p2 |- c'` (or `'|- c'`) string."""
        prem, concl = split_thm(s)
        return cls(premises=prem, conclusion=concl)


@dataclass(frozen=True)
class ReferenceProof:
    """A known-valid proof of a `Sequent`, N1-based (see `core.proof.renumber`)."""

    body: str
    lines: int
    pruned: int
    source: str  # minlen | witness | constructed | model


@dataclass(frozen=True)
class LengthLabel:
    """A `Sequent`'s proof-length label and how it was obtained (see plan §2 for `method`
    semantics). `L` is the point estimate used for plotting/stratification when present;
    `L_ub`/`L_lb` bracket it when only bounds are known (`method == "ub_lb"`)."""

    L: int | None
    L_ub: int | None
    L_lb: int | None
    method: str  # minlen_exact | ub_lb | witness_argument | reference
    bound: int | None
    timeout: bool = False


@dataclass(frozen=True)
class PrefixSpec:
    """A forced proof prefix: the first `j` lines of some reference proof, N1-based, no `QED`."""

    j: int
    body: str


@dataclass(frozen=True)
class Problem:
    """One benchmark item: a `Sequent` plus its length label, optional reference proof, and
    suite-specific tags. The unit `suites` build, `Store` persists, `runner` samples against, and
    `analysis` reads back."""

    pid: str
    suite: str
    version: str
    sequent: Sequent
    length: LengthLabel
    ref: ReferenceProof | None
    tags: Mapping[str, JSON]
    parts: tuple[str, ...] = ()
    prefix: PrefixSpec | None = None
    seen_train: bool = False
    seen_eval: tuple[str, ...] = ()

    def to_json(self) -> dict[str, JSON]:
        return {
            "pid": self.pid,
            "suite": self.suite,
            "version": self.version,
            "sequent": {
                "premises": list(self.sequent.premises),
                "conclusion": self.sequent.conclusion,
            },
            "length": {
                "L": self.length.L,
                "L_ub": self.length.L_ub,
                "L_lb": self.length.L_lb,
                "method": self.length.method,
                "bound": self.length.bound,
                "timeout": self.length.timeout,
            },
            "ref": None
            if self.ref is None
            else {
                "body": self.ref.body,
                "lines": self.ref.lines,
                "pruned": self.ref.pruned,
                "source": self.ref.source,
            },
            "tags": dict(self.tags),
            "parts": list(self.parts),
            "prefix": None
            if self.prefix is None
            else {"j": self.prefix.j, "body": self.prefix.body},
            "seen_train": self.seen_train,
            "seen_eval": list(self.seen_eval),
        }

    @classmethod
    def from_json(cls, d: Mapping[str, JSON]) -> Problem:
        seq_d = cast(Mapping[str, JSON], d["sequent"])
        length_d = cast(Mapping[str, JSON], d["length"])

        ref: ReferenceProof | None = None
        ref_raw = d.get("ref")
        if ref_raw is not None:
            ref_d = cast(Mapping[str, JSON], ref_raw)
            ref = ReferenceProof(
                body=cast(str, ref_d["body"]),
                lines=cast(int, ref_d["lines"]),
                pruned=cast(int, ref_d["pruned"]),
                source=cast(str, ref_d["source"]),
            )

        prefix: PrefixSpec | None = None
        prefix_raw = d.get("prefix")
        if prefix_raw is not None:
            prefix_d = cast(Mapping[str, JSON], prefix_raw)
            prefix = PrefixSpec(j=cast(int, prefix_d["j"]), body=cast(str, prefix_d["body"]))

        return cls(
            pid=cast(str, d["pid"]),
            suite=cast(str, d["suite"]),
            version=cast(str, d["version"]),
            sequent=Sequent(
                premises=tuple(cast(list[str], seq_d["premises"])),
                conclusion=cast(str, seq_d["conclusion"]),
            ),
            length=LengthLabel(
                L=cast("int | None", length_d["L"]),
                L_ub=cast("int | None", length_d["L_ub"]),
                L_lb=cast("int | None", length_d["L_lb"]),
                method=cast(str, length_d["method"]),
                bound=cast("int | None", length_d["bound"]),
                timeout=cast(bool, length_d.get("timeout", False)),
            ),
            ref=ref,
            tags=cast(Mapping[str, JSON], d.get("tags", {})),
            parts=tuple(cast(list[str], d.get("parts", ()))),
            prefix=prefix,
            seen_train=cast(bool, d.get("seen_train", False)),
            seen_eval=tuple(cast(list[str], d.get("seen_eval", ()))),
        )


@dataclass
class SuiteManifest:
    """Provenance record written alongside a built suite (`Store.write_suite`)."""

    suite: str
    version: str
    builder: str
    config: dict[str, JSON]
    seed: int
    created_utc: str
    git_commit: str
    n_items: int
    items_sha256: str
    excl_sources: dict[str, str]
    stats: dict[str, JSON]


@dataclass(frozen=True)
class SampleRequest:
    """One request to a `ModelAdapter.sample`: draw `n` attempts at `sequent` (optionally
    continuing from `prefix`)."""

    pid: str
    sequent: Sequent
    n: int
    temperature: float
    max_new_tokens: int
    seed: int
    prefix: PrefixSpec | None = None
    options: Mapping[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class Attempt:
    """One sampled proof attempt.

    `body` is the canonical ND body (prefix included if requested), or the literal string
    `"UNPARSEABLE <why>"` if the adapter's own decoder could not produce ND text at all (as
    opposed to text that decodes but `Judge.judge` then rejects).
    """

    pid: str
    i: int
    body: str
    native: str | None
    n_tokens: int
    finished: bool


@dataclass(frozen=True)
class Verdict:
    """A `Judge`'s verdict on one `Attempt`."""

    ok: bool
    reason: str
    kind: str  # ok | bad_line | early_qed | no_qed | unparseable
    written: int | None
    pruned: int | None
    first_bad_line: int | None
    start: int | None


@dataclass
class ItemResult:
    """Aggregated outcome of sampling one `Problem` (one chunk; `Store.items` merges chunks)."""

    run_id: str
    pid: str
    chunk: int
    n: int
    c: int
    first_success: int | None
    ok_written: list[int]
    ok_pruned: list[int]
    ok_tokens: list[int]
    ok_starts: list[int]
    fail_kinds: dict[str, int]
    fail_lines: list[int]
    long_enough: int
    tokens_mean: float
    hit_max: int


@dataclass(frozen=True)
class LogProb:
    """A sequence log-probability, marginalised over start-index shift encodings: `nats = log
    sum_s p_theta(shift_s(y) | x)` (a log-probability, so `nats <= 0`; `per_encoding[s]` is the
    unmarginalised per-shift log-probability, i.e. `per_encoding[s] = log p_theta(shift_s(y) | x)`,
    and `nats == logsumexp(per_encoding)`).

    **Sign convention.** `ModelAdapter.logprob` returns log-probability as defined above (`nats <=
    0`). `Learner.nll` returns the *negative* of that same quantity -- a positive excess
    description length nats, `nats = -log sum_s p_theta(shift_s(y) | x) >= 0`, consistent with
    `Learner.nll`'s docstring ("negative log-likelihood") and with `metrics.edl`/`edl.run_edl`,
    which sum `Learner.nll` outputs directly as per-example losses without negating them again.
    `per_encoding` is *not* re-signed for `Learner.nll`: it stays the per-shift log-probabilities
    (so `nats == -logsumexp(per_encoding)` for a `Learner.nll` result), matching
    `adapters.absgpt._AbsGptLearner.nll`/`test_wp5_fixtures.TinyGptLearner.nll`. Always check which
    of the two methods produced a given `LogProb` before combining values from both."""

    nats: float
    n_tokens: int
    per_encoding: tuple[float, ...]


class Certifier(Protocol):
    """Assigns `LengthLabel`s to sequents, optionally given a candidate witness proof for each
    (see plan §2 for the `minlen_exact`/`ub_lb`/`witness_argument`/`reference` method semantics).
    """

    def certify(
        self,
        seqs: Sequence[Sequent],
        witnesses: Sequence[str | None],
        *,
        bound: int | None = None,
        time_s: float = 60.0,
    ) -> list[LengthLabel]: ...


class Excluder(Protocol):
    """Checks a `Sequent` against the train/eval exclusion tiers (plan §2's dedup policy)."""

    def status(self, seq: Sequent) -> tuple[bool, tuple[str, ...]]:
        """`(in_train, eval_sets)`: whether `seq` matches a training-tier theorem, and which
        eval-tier files (if any) it also matches (flagged, not excluded)."""
        ...
