"""`ApiLlmAdapter`: a `ModelAdapter` over a plain text-completion API (plan §1/§3 WP7).

The prompt is a fixed excerpt of the take-home spec (`SPEC_EXCERPT`, embedded below as a constant --
sourced from `/scratch/cowifornia/nd-rl/external/nd-takehome/spec.md`, sections 1-5, trimmed to the
grammar and rule table any model needs to produce the token format) plus a few worked examples
(`DEFAULT_FEW_SHOT`) plus the sequent to prove (and, if `SampleRequest.prefix` is set, the partial
proof to continue). `sample()` extracts the `N<k> ... QED` span from the raw response with a regex,
stores the full raw response in `Attempt.native` (never discarded, even when parsing fails), and
falls back to `"UNPARSEABLE <why>"` for a response with no such span.

**No SDK dependency.** The actual API call sits behind an injected `client: Callable[[str], str]`
(prompt text in, raw response text out) -- this module never imports an SDK, and every acceptance
test runs with `dry_run=True` (canned responses, no `client` needed, no network) per the plan's "no
network or API calls" rule for this WP.

**Cost cap.** `cost_cap_usd`/`price_per_1k_tokens` bound real (non-dry-run) spend: before every real
`client` call, `sample()` checks a running word-count-based cost estimate (prompt + response
word count, at `price_per_1k_tokens` per 1,000 words -- a deliberately crude proxy for "tokens",
since this adapter has no tokenizer of its own and the goal is a soft safety cap, not exact billing)
against `cost_cap_usd`, raising `CostCapExceeded` *before* making the call that would cross it, so a
run never overshoots the cap because of one large response.

**`Attempt.finished`.** Always `True`: an API call either returns a (possibly malformed) response
or raises, there is no partial/truncated-by-budget response to distinguish, unlike a token-sampled
adapter's `max_new_tokens` cap.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence

from ..core.types import JSON, Attempt, SampleRequest, Sequent
from .base import Capability, ModelAdapter

_UNPARSEABLE = "UNPARSEABLE"

#: Sections 1-5 of the take-home spec (grammar, sequents/prompts, proof lines, the rule table, and
#: the worked example), trimmed for prompt length. Source:
#: `/scratch/cowifornia/nd-rl/external/nd-takehome/spec.md` (read 2026-09-30; that file is outside
#: this WP's file list and is never imported at runtime -- only this frozen excerpt is used).
SPEC_EXCERPT = """\
You are proving a sequent in a Fitch-style natural-deduction system.

FORMULAS: atoms P, Q, R, S; falsum F; connectives ~ (not), & (and), v (or), > (implies).
Every compound formula is fully parenthesised, including negation:
( ~ P ), ( P & Q ), ( ( P > Q ) > P ). Atoms and F are bare.
Tokens are whitespace-separated. There is no <->; there are no other atoms.

SEQUENTS AND PROMPTS: a sequent is "premises |- conclusion". You are prompted with
    THM <premise-1> , <premise-2> , ... SEQ <conclusion> PRF
(or "THM SEQ <conclusion> PRF" with no premises) and must reply with the proof body followed by QED.

PROOF LINES: one line is
    N<i> ['|' x depth] <formula> : <RULE> [N<ref> ...] ;
- N<i>: line index, consecutive and ascending, starting at N1.
- '|' repeated `depth` times: how many subproof boxes the line sits inside.
- PR (premise) lines come first, at depth 0, reproducing the declared premises exactly and in order.
- AS (assumption) at depth d closes any boxes deeper than d-1 and opens a new box with that formula
  as hypothesis.
- Any other line at depth d closes boxes deeper than d.
- A box is cited as "N<start> N<end>": `start` is its AS line, `end` is the box's last line.
- The last line must be at depth 0, must be the conclusion, and must not be AS.

RULES (name -- refs -- what is checked):
  PR      --        -- declared premise
  AS      --        -- opens a box with hypothesis G
  R       Nj        -- G = formula at j (reiteration)
  ANDI    Na Nb     -- G = ( A & B )
  ANDE1   Na        -- a: ( G & B )
  ANDE2   Na        -- a: ( A & G )
  IMPE    Na Nb     -- a: ( B > G ), b: B (modus ponens; implication first)
  IMPI    Ns Ne     -- box s..e with hypothesis A ending in B; G = ( A > B )
  ORI1    Na        -- G = ( A v X ) with a: A
  ORI2    Na        -- G = ( X v A ) with a: A
  ORE     Nj Ns1 Ne1 Ns2 Ne2 -- j: ( A v B ); boxes s1..e1 (assumes A) and s2..e2
                                 (assumes B) both end in G
  NEGE    Na Nb     -- a: A, b: ( ~ A ); G = F (positive first)
  NEGI    Ns Ne     -- box s..e assumes A, ends in F; G = ( ~ A )
  BOTE    Na        -- a: F; G anything (ex falso)
  DN      Na        -- a: ( ~ ( ~ G ) ) (classical double-negation elimination)

WORKED EXAMPLE. Disjunctive syllogism, ( P v Q ) , ( ~ P ) |- Q:
THM ( P v Q ) , ( ~ P ) SEQ Q PRF
N1 ( P v Q ) : PR ; N2 ( ~ P ) : PR ; N3 | P : AS ; N4 | F : NEGE N3 N2 ; N5 | Q : BOTE N4 ; \
N6 | Q : AS ; N7 Q : ORE N1 N3 N5 N6 N6 ; QED

Reply with the proof body only: start at N1, end with QED, one token per space-separated symbol.
"""

#: One default few-shot example (prompt, proof-body) pair, used when the caller does not supply
#: `few_shot` explicitly. Kept separate from `SPEC_EXCERPT` (which already repeats this exact proof
#: as its worked example) so a caller can drop in a different few-shot set without editing the spec
#: text.
DEFAULT_FEW_SHOT: tuple[tuple[str, str], ...] = (
    (
        "THM ( P v Q ) , ( ~ P ) SEQ Q PRF",
        "N1 ( P v Q ) : PR ; N2 ( ~ P ) : PR ; N3 | P : AS ; N4 | F : NEGE N3 N2 ; "
        "N5 | Q : BOTE N4 ; N6 | Q : AS ; N7 Q : ORE N1 N3 N5 N6 N6 ; QED",
    ),
)

#: Matches the first `N<digits> ... QED` span in a raw response, across newlines, non-greedy so it
#: stops at the *first* `QED` (a trailing explanation after it is not part of the proof).
_SPAN_RE = re.compile(r"N\d+\s.*?\bQED\b", re.DOTALL)


class CostCapExceeded(RuntimeError):
    """Raised by `ApiLlmAdapter.sample` when a real `client` call would exceed `cost_cap_usd`."""


class ApiLlmAdapter(ModelAdapter):
    """A `ModelAdapter` over a text-completion API. See the module docstring."""

    def __init__(
        self,
        *,
        client: Callable[[str], str] | None = None,
        dry_run: bool = False,
        dry_run_responses: Callable[[Sequent, int], str] | Sequence[str] | None = None,
        few_shot: Sequence[tuple[str, str]] = DEFAULT_FEW_SHOT,
        cost_cap_usd: float | None = None,
        price_per_1k_tokens: float = 0.0,
        label: str = "api_llm",
        format_id: str = "api_llm_v1",
        model_identity: Mapping[str, JSON] | None = None,
    ) -> None:
        if not dry_run and client is None:
            raise ValueError("ApiLlmAdapter needs a `client` callable unless dry_run=True")
        self._client = client
        self._dry_run = dry_run
        self._dry_run_responses = dry_run_responses
        self._few_shot = tuple(few_shot)
        self._cost_cap_usd = cost_cap_usd
        self._price_per_1k = price_per_1k_tokens
        self._spent_usd = 0.0
        self._model_identity = dict(model_identity) if model_identity is not None else {}
        self.label = label
        self.format_id = format_id
        self.capabilities = Capability.SAMPLE | Capability.PREFIX

    def identity(self) -> dict[str, JSON]:
        return {
            "label": self.label,
            "format_id": self.format_id,
            "dry_run": self._dry_run,
            "cost_cap_usd": self._cost_cap_usd,
            **self._model_identity,
        }

    def token_count(self, seq: Sequent, body: str | None = None) -> int:
        n = len(self._build_prompt(seq).split())
        if body:
            n += len(body.split())
        return n

    def _build_prompt(self, seq: Sequent, prefix: str | None = None) -> str:
        parts = [SPEC_EXCERPT]
        for prompt, proof in self._few_shot:
            parts.append(f"Prove:\n{prompt}\n{proof}\n")
        parts.append(f"Now prove this sequent:\n{seq.prompt()}")
        if prefix:
            parts.append(
                f"A partial proof already exists; continue it, starting at the next line index "
                f"and do not repeat these lines:\n{prefix}"
            )
        return "\n".join(parts)

    def _dry_run_response(self, seq: Sequent, i: int) -> str:
        if self._dry_run_responses is None:
            return "I was unable to find a proof."
        if callable(self._dry_run_responses):
            return self._dry_run_responses(seq, i)
        pool = self._dry_run_responses
        return pool[i % len(pool)]

    def _get_response(self, seq: Sequent, prompt: str, i: int) -> str:
        if self._dry_run:
            return self._dry_run_response(seq, i)
        assert self._client is not None  # checked in __init__
        if self._cost_cap_usd is not None and self._spent_usd >= self._cost_cap_usd:
            raise CostCapExceeded(
                f"cost cap ${self._cost_cap_usd:.4f} already reached "
                f"(spent ${self._spent_usd:.4f}); refusing further ApiLlmAdapter calls"
            )
        raw = self._client(prompt)
        self._spent_usd += self._price_per_1k * (len(prompt.split()) + len(raw.split())) / 1000.0
        return raw

    @staticmethod
    def _extract_span(raw: str) -> str:
        """The `N<k> ... QED` substring of `raw`, whitespace-normalised, or an `UNPARSEABLE ...`
        marker if no such span is present (a malformed response)."""
        m = _SPAN_RE.search(raw)
        if m is None:
            return f"{_UNPARSEABLE} no N<k>..QED span found in response"
        return " ".join(m.group(0).split())

    def sample(self, reqs: Sequence[SampleRequest]) -> list[list[Attempt]]:
        out: list[list[Attempt]] = []
        for req in reqs:
            prefix_body = req.prefix.body if req.prefix is not None else None
            prompt = self._build_prompt(req.sequent, prefix_body)
            attempts: list[Attempt] = []
            for i in range(req.n):
                raw = self._get_response(req.sequent, prompt, i)
                span = self._extract_span(raw)
                if span.startswith(_UNPARSEABLE):
                    body = span
                elif prefix_body is not None:
                    body = f"{prefix_body} {span}"
                else:
                    body = span
                attempts.append(
                    Attempt(
                        pid=req.pid,
                        i=i,
                        body=body,
                        native=raw,
                        n_tokens=len(raw.split()),
                        finished=True,  # a completed API call; see the module docstring
                    )
                )
            out.append(attempts)
        return out
