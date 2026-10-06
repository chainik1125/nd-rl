"""Verifier-blind sequential attempt protocol, independent of any neural policy.

Only PublicAttempt objects are observable. Verification is delayed until final
exact-proof retrieval, against the immutable original theorem. Limits never
imply correctness and never add a reward term beyond counted attempt cost.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class PublicAttempt:
    """Raw policy-visible output; never a verifier's decorated diagnostic text."""

    tokens: tuple[int, ...]
    text: str


@dataclass(frozen=True)
class Attempt:
    public: PublicAttempt
    proof: str
    complete: bool
    truncated: bool
    generation_logprobs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.generation_logprobs and len(self.generation_logprobs) != len(self.public.tokens):
            raise ValueError("one generation likelihood is required per sampled token")
        if any(not isfinite(p) or p > 0 for p in self.generation_logprobs):
            raise ValueError("invalid generation log probability")


@dataclass(frozen=True)
class Decision:
    """Index is zero-based; None means CONTINUE. Logprob uses the masked policy."""

    index: int | None
    logprob: float
    accept_probability: float

    def __post_init__(self) -> None:
        if not isfinite(self.logprob) or self.logprob > 0:
            raise ValueError("invalid behavior log probability")
        if not 0 <= self.accept_probability <= 1:
            raise ValueError("invalid acceptance probability")


@dataclass(frozen=True)
class EpisodeResult:
    index: int
    proof: str
    correct: bool
    attempts: int
    reward: float
    forced: bool
    termination_reason: str


class Episode:
    """Mutable environment with explicit context reservation and delayed reward."""

    def __init__(
        self,
        theorem: str,
        max_attempts: int = 8,
        attempt_cost: float = 0,
        context_limit: int = 4096,
        selection_reserve: int = 16,
        initial_tokens: int = 0,
    ) -> None:
        if max_attempts < 1 or attempt_cost < 0:
            raise ValueError("invalid attempt budget or cost")
        if initial_tokens < 0 or selection_reserve < 1:
            raise ValueError("invalid context accounting")
        if initial_tokens + selection_reserve >= context_limit:
            raise ValueError("no space for a mandatory attempt")
        self.theorem = theorem
        self.max_attempts = max_attempts
        self.attempt_cost = attempt_cost
        self.context_limit = context_limit
        self.selection_reserve = selection_reserve
        self.occupied_tokens = initial_tokens
        self.attempts: list[Attempt] = []
        self.decisions: list[Decision] = []
        self.result: EpisodeResult | None = None
        self.awaiting_attempt = True

    def observe(self) -> tuple[str, tuple[PublicAttempt, ...]]:
        return self.theorem, tuple(a.public for a in self.attempts)

    def can_continue(self, next_attempt_token_bound: int) -> bool:
        if next_attempt_token_bound < 1:
            raise ValueError("attempt bound must be positive")
        return (
            self.result is None
            and len(self.attempts) < self.max_attempts
            and self.occupied_tokens + next_attempt_token_bound + self.selection_reserve
            <= self.context_limit
        )

    def append(self, attempt: Attempt, boundary_tokens: int = 0) -> None:
        if (
            self.result is not None
            or len(self.attempts) >= self.max_attempts
            or not self.awaiting_attempt
        ):
            raise ValueError("episode cannot accept another attempt")
        size = len(attempt.public.tokens) + boundary_tokens
        if boundary_tokens < 0 or size < 1:
            raise ValueError("invalid attempt context accounting")
        if self.occupied_tokens + size + self.selection_reserve > self.context_limit:
            raise ValueError("attempt consumes reserved final-selection context")
        self.attempts.append(attempt)
        self.awaiting_attempt = False
        self.occupied_tokens += size

    def decide(
        self,
        decision: Decision,
        verify: Callable[[str, str], bool],
        next_attempt_token_bound: int,
        forced: bool = False,
    ) -> EpisodeResult | None:
        if self.result is not None or not self.attempts or self.awaiting_attempt:
            raise ValueError("decision requires an unfinished episode with an attempt")
        available = self.can_continue(next_attempt_token_bound)
        if forced and available:
            raise ValueError("forced selection needs an attempt or context limit")
        forced = not available
        if decision.index is None:
            if forced or not available:
                raise ValueError("CONTINUE is masked at a hard limit")
            self.decisions.append(decision)
            self.awaiting_attempt = True
            return None
        if not 0 <= decision.index < len(self.attempts):
            raise ValueError("selection index is outside the stored attempt pool")
        self.decisions.append(decision)
        proof = self.attempts[decision.index].proof
        correct = bool(verify(self.theorem, proof))
        reason = "voluntary"
        if forced:
            reason = "attempt_limit" if len(self.attempts) == self.max_attempts else "context_limit"
        self.result = EpisodeResult(
            decision.index,
            proof,
            correct,
            len(self.attempts),
            float(correct) - self.attempt_cost * (len(self.attempts) - 1),
            forced,
            reason,
        )
        return self.result
