"""`NDJudge`: the `Judge` for token-format ND proofs (plan §1's `judge/nd.py`).

Ground truth is the vendored `nd_verify.verify_text`; `NDJudge` gets its richer classification
(first bad line, `early_qed`/`no_qed`/`index_overflow` reasons) from the vendored lengthexp
`ndcheck.classify`, which is built to always agree with `nd_verify` on `ok` (see
`vendor/README.md`). The dependency-pruned length of a successful proof comes from the vendored
`absgpt/prune.pruned_length`.

**Soundness patch (2026-09-30).** The vendored `nd_verify` accepts a box citation `N<s> N<e>` whose
box lies *after* the citing line (`box_citable` never checks order, unlike `line_citable`), which
admits circular proofs such as `N1 (P>Q) : IMPI N2 N3 ; N2 | P : AS ; N3 | Q : IMPE N1 N2 ;
N4 (P>Q) : R N1` for the non-theorem ⊢ P→Q. The spec requires a cited box to be closed at the
citing line, so `NDJudge` additionally rejects any line citing an index ≥ its own
(`reason="forward_citation"`). The vendored bytes stay unchanged. A scan of all 876,479 proofs
accepted during RL (ab, ab2, ab3 `found_targets.jsonl`) found no forward citation, so earlier
results are unaffected.
"""

from __future__ import annotations

import types
from collections.abc import Sequence

from ..core.types import Attempt, Sequent, Verdict
from ..vendor import load_absgpt_module, load_nd_verify, load_ndcheck
from .base import Judge

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


def first_forward_citation(body: str) -> int | None:
    """1-based position of the first line citing a line index >= its own, or None.

    Returns None as well when the body does not parse (the verifier rejects it anyway).
    """
    verify_mod = load_nd_verify()  # the package re-exports parse_proof_tokens and ParseError
    try:
        lines = verify_mod.parse_proof_tokens(body.split())
    except verify_mod.ParseError:
        return None
    for pos, ln in enumerate(lines, start=1):
        if any(r >= ln["idx"] for r in ln["refs"]):
            return pos
    return None


class NDJudge(Judge):
    """Judges attempts against `nd_verify` + `ndcheck`'s incremental checker.

    Subclasses `Judge` (rather than only matching it structurally) purely to inherit its default
    `judge_many` (loop over `judge`, one call per group) for free -- `nd_verify`/`ndcheck` are
    pure-Python and per-item already, so there is nothing to gain from batching across items the
    way `LeanJudge.judge_many` does.
    """

    judge_id = "nd_v2"  # v2: forward citations rejected (see module docstring)

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
            if ok:
                fwd = first_forward_citation(a.body)
                if fwd is not None:
                    ok = False
                    d = dict(d, ok=False, kind="bad_line", line=fwd, reason="forward_citation")
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
