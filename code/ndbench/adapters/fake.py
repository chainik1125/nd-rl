"""`FakeAdapter`: a deterministic `ModelAdapter` test double with no model or GPU dependency.

Returns each problem's reference proof with probability `p`, and a corrupted variant otherwise,
so `runner.run` (and adaptive stopping) can be tested for exact `n`/`c` counts against a known
Bernoulli process with fully reproducible draws.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence

from ..core.types import JSON, Attempt, SampleRequest, Sequent
from .base import Capability, ModelAdapter


def drop_qed(body: str) -> str:
    """Default corruption: strip the trailing `QED` token.

    `nd_verify`/`vendor/ndcheck.classify` always reject the result (`kind="no_qed"`), regardless
    of what `body` otherwise contains, making this a safe universal corruption for any reference
    proof.
    """
    toks = body.split()
    if toks and toks[-1] == "QED":
        toks = toks[:-1]
    return " ".join(toks)


class FakeAdapter(ModelAdapter):
    """Returns `refs[pid]` with probability `p`, else `corrupt(refs[pid])`.

    Each draw is seeded from `(request.seed, pid, draw index)` only, so two `sample()` calls with
    equal requests (even across process runs) produce byte-identical attempts. Does not honour
    `SampleRequest.prefix` (every request is treated as prefix-free); does not advertise
    `Capability.PREFIX`.
    """

    def __init__(
        self,
        refs: Mapping[str, str],
        p: float,
        *,
        corrupt: Callable[[str], str] = drop_qed,
        label: str = "fake",
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self._refs = dict(refs)
        self.p = p
        self._corrupt = corrupt
        self.label = label
        self.format_id = "nd_v1"
        self.capabilities = Capability.SAMPLE

    def identity(self) -> dict[str, JSON]:
        return {
            "label": self.label,
            "format_id": self.format_id,
            "p": self.p,
            "n_refs": len(self._refs),
        }

    def token_count(self, seq: Sequent, body: str | None = None) -> int:
        n = len(seq.prompt().split())
        if body:
            n += len(body.split())
        return n

    def sample(self, reqs: Sequence[SampleRequest]) -> list[list[Attempt]]:
        out: list[list[Attempt]] = []
        for req in reqs:
            ref_body = self._refs.get(req.pid)
            if ref_body is None:
                raise KeyError(f"FakeAdapter has no reference proof for pid {req.pid!r}")
            attempts: list[Attempt] = []
            for i in range(req.n):
                rng = random.Random(f"{req.seed}:{req.pid}:{i}")
                body = ref_body if rng.random() < self.p else self._corrupt(ref_body)
                attempts.append(
                    Attempt(
                        pid=req.pid,
                        i=i,
                        body=body,
                        native=None,
                        n_tokens=self.token_count(req.sequent, body),
                        finished=True,
                    )
                )
            out.append(attempts)
        return out
