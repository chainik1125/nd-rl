"""`LeanJudge`: judges `lean_seq`-decodable ND proof attempts by compiling them with Lean 4 (plan
§1/§3 WP7).

Mirrors the semantics of Dan's `code/shared_code/lean_check.py` (fork `dan-pandori/nd-takehome`;
read with `git -C /scratch/cowifornia/nd-rl/bench/nd-rl show
origin/dan_lean_seq_tokenizer:code/shared_code/lean_check.py`): render `statement(prompt) + " " +
lean_text(proof)` as one Lean 4 theorem (no Mathlib, no `sorry`, no tactic automation beyond the
vendored `lean_seq` vocabulary), batch as many theorems as practical into as few `lean` processes as
possible (a process costs about a second to start), and accept a theorem iff Lean reports neither an
error nor a `sorry` warning inside that theorem's own line range; a process that crashes, times out,
or reports an error outside every theorem's range is split in half and re-run, down to rejecting a
lone theorem that still cannot be isolated. `check_theorems`/`_check_chunk`/`_failing_theorems`
below are a from-scratch reimplementation of that batching/splitting scheme (not a copy -- the
source file lives outside this WP's file list and is never imported at runtime), simplified in one
respect: `check_theorems` here takes an already-resolved `lean` executable path rather than doing
its own `find_lean()` fallback, since `LeanJudge.__init__` already resolves the binary once.

Unlike Dan's `judge(tok, prompts, samples)` (which starts from raw sampled ids), this module starts
from an `Attempt.body` -- an ND proof string, already decoded elsewhere (`adapters.lean_seq`'s
`inverse`-based decode, or any other adapter's plain ND output) -- and renders it back to `lean_seq`
tokens via the vendored `LeanSeqTokenizer.encode_proof`/`lean_text`, so `LeanJudge` can check *any*
ND-format attempt, not only ones a `lean_seq`-tokenized model produced.

**Not installed here.** `LeanJudge()` raises `judge.base.NotAvailable` at construction if no `lean`
binary is found -- checked, in the same order as Dan's `find_lean`, via the `LEAN` environment
variable, `lean` on `PATH`, then `~/.elan/bin/lean`. Callers should catch `NotAvailable` and skip
Lean-checked benchmarks for this environment rather than treat it as a run failure, exactly as
`judge.base.NotAvailable`'s own docstring prescribes. This environment has no Lean 4 install, so
`code/tests/ndbench/test_wp7_lean_judge.py` covers this path directly and exercises the rest of the
rendering/batching pipeline against a small *fake* `lean` executable (a local Python script
fixture, not a network resource) that mimics the real binary's stdout/rc contract.
"""

from __future__ import annotations

import bisect
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..core.proof import first_index, line_count, pruned_length
from ..core.types import Attempt, Sequent, Verdict
from ..vendor.lean_seq.lean_seq import LeanParseError, LeanSeqTokenizer, NDParseError
from .base import NotAvailable

_PRELUDE = "set_option linter.unusedVariables false\nset_option maxRecDepth 4000\n"
_CHUNK = 400
# An error, or the warning Lean gives instead of an error for `sorry`: either rejects the theorem.
_ERROR_RE = re.compile(
    r"^[^\n]*?:(\d+):\d+: (?:error|warning: declaration uses [`']sorry[`'])", re.M
)
_UNPARSEABLE = "UNPARSEABLE"


def find_lean(env_var: str = "LEAN") -> str | None:
    """The `lean` executable: `$LEAN`, else `lean` on `PATH`, else `~/.elan/bin/lean`."""
    env = os.environ.get(env_var)
    if env:
        return env
    found = shutil.which("lean")
    if found:
        return found
    elan = Path.home() / ".elan" / "bin" / "lean"
    return str(elan) if elan.exists() else None


def _failing_theorems(output: str, starts: Sequence[int]) -> tuple[set[int], bool]:
    """Map Lean's error lines (and `sorry` warnings) to theorem indices.

    `starts[k]` is the first line (1-based) of theorem `k` in the checked file, increasing.
    Returns the indices with at least one error, and whether any error fell outside every
    theorem (before the first one, i.e. in the prelude).
    """
    bad: set[int] = set()
    stray = False
    for m in _ERROR_RE.finditer(output):
        k = bisect.bisect_right(starts, int(m.group(1))) - 1
        if 0 <= k < len(starts):
            bad.add(k)
        else:
            stray = True
    return bad, stray


def _check_chunk(
    lean: str, sources: Sequence[str], workdir: str, tag: str, timeout_s: float
) -> list[bool]:
    text = _PRELUDE
    starts: list[int] = []
    for k, src in enumerate(sources):
        starts.append(text.count("\n") + 1)
        text += src.replace("theorem t ", f"theorem t{k} ", 1).rstrip("\n") + "\n"
    path = os.path.join(workdir, f"c_{tag}.lean")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    try:
        run = subprocess.run(
            [lean, "-DmaxErrors=100000000", path],
            capture_output=True,
            text=True,
            timeout=timeout_s + 2 * text.count("\n"),
        )
        output, rc = run.stdout + run.stderr, run.returncode
    except subprocess.TimeoutExpired:
        output, rc = "", -9
    finally:
        os.remove(path)
    bad, stray = _failing_theorems(output, starts)
    if rc not in (0, 1) or (rc == 1 and not bad) or stray:
        # Crashed, timed out, or failed without naming a theorem: split, and reject a lone theorem.
        if len(sources) == 1:
            return [False]
        h = len(sources) // 2
        return _check_chunk(lean, sources[:h], workdir, tag + "a", timeout_s) + _check_chunk(
            lean, sources[h:], workdir, tag + "b", timeout_s
        )
    return [k not in bad for k in range(len(sources))]


def check_theorems(
    sources: Sequence[str],
    *,
    lean: str,
    chunk: int = _CHUNK,
    workers: int | None = None,
    timeout_s: float = 60.0,
) -> list[bool]:
    """Whether Lean 4 core accepts each theorem source, in order.

    Each source is one theorem that starts `"theorem t "` (it is renamed per batch) and may span
    several lines. `lean` must already be a resolved executable path (see the module docstring for
    why this differs from Dan's `find_lean()`-falling-back version).
    """
    if not sources:
        return []
    for src in sources:
        if not src.startswith("theorem t "):
            raise ValueError(f"a theorem source must start 'theorem t ': {src[:60]!r}")
    if workers is None:
        affinity = getattr(os, "sched_getaffinity", None)
        workers = len(affinity(0)) if affinity else (os.cpu_count() or 1)
    pieces = [(list(sources[i : i + chunk]), str(i)) for i in range(0, len(sources), chunk)]
    workdir = tempfile.mkdtemp(prefix="ndbench_lean_check_")
    try:
        with ThreadPoolExecutor(workers) as pool:
            results = list(
                pool.map(lambda p: _check_chunk(lean, p[0], workdir, p[1], timeout_s), pieces)
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return [ok for piece in results for ok in piece]


class LeanJudge:
    """Judges ND-format attempts by rendering them to Lean 4 and calling `lean`."""

    judge_id = "lean_v1"

    def __init__(
        self, *, lean: str | None = None, timeout_s: float = 60.0, workers: int | None = None
    ) -> None:
        resolved = lean if lean is not None else find_lean()
        if resolved is None:
            raise NotAvailable(
                "no `lean` binary found ($LEAN, PATH, or ~/.elan/bin/lean); install Lean 4 (e.g. "
                "via elan) or set $LEAN to use LeanJudge"
            )
        self._lean = resolved
        self._timeout_s = timeout_s
        self._workers = workers
        self._tok = LeanSeqTokenizer()

    def _render(self, seq: Sequent, body: str) -> str | None:
        """`statement + " " + proof text`, or `None` if `body` cannot be rendered as `lean_seq`
        (too many names/premises, or not a well-formed ND proof)."""
        try:
            statement = self._tok.statement(seq.prompt())
            ids = self._tok.encode_proof(body)
            proof_text = self._tok.lean_text(ids)
        except (NDParseError, LeanParseError, ValueError, KeyError):
            return None
        if proof_text is None:
            # encode_proof always appends <eos>, so lean_text (which only needs the ids to reach
            # <eos>) should never fail here; guarded anyway rather than asserted.
            return None
        return f"{statement} {proof_text}"

    def judge(
        self, seq: Sequent, attempts: Sequence[Attempt], ref_lines: int | None
    ) -> list[Verdict]:
        del ref_lines  # not needed; kept for the Judge protocol
        verdicts: list[Verdict | None] = [None] * len(attempts)
        pending: dict[str, list[int]] = {}
        for i, a in enumerate(attempts):
            if a.body.startswith(_UNPARSEABLE):
                reason = a.body[len(_UNPARSEABLE) :].strip() or "unparseable"
                verdicts[i] = Verdict(
                    ok=False,
                    reason=reason,
                    kind="unparseable",
                    written=None,
                    pruned=None,
                    first_bad_line=None,
                    start=None,
                )
                continue
            source = self._render(seq, a.body)
            written = line_count(a.body)
            start = first_index(a.body)
            if source is None:
                verdicts[i] = Verdict(
                    ok=False,
                    reason="lean_seq render failed (too many names/premises, or malformed proof)",
                    kind="unparseable",
                    written=written,
                    pruned=None,
                    first_bad_line=None,
                    start=start,
                )
                continue
            pending.setdefault(source, []).append(i)

        sources = list(pending)
        oks = check_theorems(
            sources, lean=self._lean, workers=self._workers, timeout_s=self._timeout_s
        )
        for source, ok in zip(sources, oks, strict=True):
            for i in pending[source]:
                body = attempts[i].body
                # `kind` has no Lean-specific value in the shared vocabulary (`ok|bad_line|
                # early_qed|no_qed|unparseable`, `core.types.Verdict`'s docstring); a Lean
                # rejection is reported as "bad_line" -- the closest existing category (Lean gives
                # no single first-bad-ND-line the way `ndcheck` does, hence `first_bad_line=None`).
                verdicts[i] = Verdict(
                    ok=ok,
                    reason="" if ok else "lean rejected",
                    kind="ok" if ok else "bad_line",
                    written=line_count(body),
                    pruned=pruned_length(body) if ok else None,
                    first_bad_line=None,
                    start=first_index(body),
                )
        done = [v for v in verdicts if v is not None]
        assert len(done) == len(attempts)  # every index was filled in above (unparseable or `oks`)
        return done
