"""`NDJudge`: the `Judge` for token-format ND proofs (plan §1's `judge/nd.py`).

Ground truth is the vendored `nd_verify.verify_text`; `NDJudge` gets its richer classification
(first bad line, `early_qed`/`no_qed`/`index_overflow` reasons) from the vendored lengthexp
`ndcheck.classify`, which is built to always agree with `nd_verify` on `ok` (see
`vendor/README.md`). The dependency-pruned length of a successful proof comes from the vendored
`absgpt/prune.pruned_length`.
"""

from __future__ import annotations

import types
from collections.abc import Sequence

from ..core.types import Attempt, Sequent, Verdict
from ..vendor import load_absgpt_module, load_ndcheck

_UNPARSEABLE_PREFIX = "UNPARSEABLE"


def _load_backends() -> tuple[types.ModuleType, types.ModuleType]:
    """`(ndcheck, prune)` vendored modules, loaded (and cached in `sys.modules`) on first use.

    `load_ndcheck` caches `nd_verify` from our own vendor tree before importing `ndcheck`, which
    in turn makes `absgpt/prune.py`'s own bare `from nd_verify.verify import ...` resolve against
    that same cached copy rather than reaching into the live `ab3` tree.
    """
    ndcheck = load_ndcheck()
    prune = load_absgpt_module("prune")
    return ndcheck, prune


class NDJudge:
    """Judges attempts against `nd_verify` + `ndcheck`'s incremental checker."""

    judge_id = "nd_v1"

    def judge(
        self, seq: Sequent, attempts: Sequence[Attempt], ref_lines: int | None
    ) -> list[Verdict]:
        del ref_lines  # not needed by the nd_verify-backed judge; kept for the Judge protocol
        ndcheck, prune = _load_backends()
        prompt = seq.prompt()
        out: list[Verdict] = []
        for a in attempts:
            if a.body.startswith(_UNPARSEABLE_PREFIX):
                reason = a.body[len(_UNPARSEABLE_PREFIX) :].strip() or "unparseable"
                out.append(
                    Verdict(
                        ok=False,
                        reason=reason,
                        kind="unparseable",
                        written=None,
                        pruned=None,
                        first_bad_line=None,
                        start=None,
                    )
                )
                continue
            d = ndcheck.classify(prompt, a.body)
            ok = bool(d["ok"])
            pruned = prune.pruned_length(prompt, a.body) if ok else None
            first_bad_line = d["line"] if d["kind"] == "bad_line" else None
            out.append(
                Verdict(
                    ok=ok,
                    reason=d["reason"],
                    kind=d["kind"],
                    written=d["n_lines"],
                    pruned=pruned,
                    first_bad_line=first_bad_line,
                    start=d["start"],
                )
            )
        return out
