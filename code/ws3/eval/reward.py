"""Binary proof reward, scored through Dmitry's strict verifier (plan T3).

    reward(prompt, completion) -> 1.0 if the completion is a verifier-accepted proof of exactly
    the prompted sequent, else 0.0.

Everything goes through `strict_verify.safe_verify` (the official verifier plus the
preceding-reference check), never the base verifier alone. A proof may start at any `N<k>`
index; the verifier does not assume `N1`, and neither does anything here.

TRL decodes completions with `skip_special_tokens=True`, which strips the `QED` end token the
verifier needs, so the TRL-facing callable rebuilds the body from `completion_ids` with the
fork tokenizer instead of trusting the decoded string.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading

import ndpath  # noqa: F401

# isort: split
from strict_verify import safe_verify  # vendored (Dmitry)
from tokenizer import Tokenizer  # fork_pin

UNPARSEABLE_PREFIXES = ("parse", "verifier exception", "malformed", "strict parser exception")
_TOKENIZERS: dict[str, Tokenizer] = {}


def fork_tokenizer(tok_mode: str = "abs") -> Tokenizer:
    if tok_mode not in _TOKENIZERS:
        _TOKENIZERS[tok_mode] = Tokenizer(tok_mode)
    return _TOKENIZERS[tok_mode]


def decode_ids(tok_mode: str, ids) -> str:
    """Exact proof-body text for a list of ids: stops at the first QED (included), drops pads."""
    return fork_tokenizer(tok_mode).decode([int(i) for i in ids])


def verify_body(prompt: str, body: str) -> tuple[bool, str, int]:
    """(accepted, reason, n_lines) for one proof body against one prompt."""
    return safe_verify(prompt, body)


def reward(prompt: str, completion: str) -> float:
    """Plain-Python reward on text. `completion` must include its QED."""
    ok, _, _ = verify_body(prompt, completion)
    return 1.0 if ok else 0.0


def is_unparseable(reason: str) -> bool:
    return reason.startswith(UNPARSEABLE_PREFIXES)


class ProofReward:
    """Callable reward for TRL's GRPOTrainer / RLOOTrainer.

    Signature TRL expects: f(prompts, completions, completion_ids=None, **kwargs) -> list[float].
    Appends every judged completion to a JSONL dump (for the reviewer's recount), keeps the last
    batch's unparseable / invalid fractions in `last_stats`, and reports them through TRL's
    `log_metric` hook when it is offered.

    `unparseable_reward` (default 0.0) is the "invalid vs wrong" variant (brainstorm F): output
    the parser rejects scores this instead of 0, below a well-formed proof of the wrong goal.
    Any non-zero value makes the reward non-binary; label such a run as shaped.
    """

    __name__ = "proof_reward"

    def __init__(
        self, tok_mode: str = "abs", dump_path: str | None = None, unparseable_reward: float = 0.0
    ):
        self.tok_mode = tok_mode
        self.unparseable_reward = float(unparseable_reward)
        self.dump_path = dump_path
        self.last_stats: dict[str, float] = {}
        self._lock = threading.Lock()
        if dump_path:
            os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)

    def body_from_ids(self, ids) -> str:
        return decode_ids(self.tok_mode, ids)

    def __call__(self, prompts, completions, completion_ids=None, **kwargs):
        step = getattr(kwargs.get("trainer_state"), "global_step", None)
        log_metric = kwargs.get("log_metric")
        out, n_unparseable, n_ok, rows = [], 0, 0, []
        for i, prompt in enumerate(prompts):
            if completion_ids is not None:
                body = self.body_from_ids(completion_ids[i])
            else:
                c = completions[i]
                body = c if isinstance(c, str) else c[-1]["content"]
            ok, reason, n_lines = verify_body(prompt, body)
            unparseable = is_unparseable(reason)
            out.append(1.0 if ok else (self.unparseable_reward if unparseable else 0.0))
            n_unparseable += unparseable
            n_ok += bool(ok)
            if self.dump_path:
                rows.append(
                    {
                        "step": step,
                        "prompt": prompt,
                        "completion": body,
                        "reward": out[-1],
                        "reason": reason,
                        "n_lines": n_lines,
                        "n_tokens": len(completion_ids[i]) if completion_ids is not None else None,
                    }
                )
        if self.dump_path and rows:
            with self._lock, open(self.dump_path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        n = max(len(prompts), 1)
        self.last_stats = {
            "reward/unparseable_frac": n_unparseable / n,
            "reward/invalid_frac": 1.0 - n_ok / n,
        }
        if callable(log_metric) and prompts:
            with contextlib.suppress(Exception):
                for k, v in self.last_stats.items():
                    log_metric(k, v)
        return out
