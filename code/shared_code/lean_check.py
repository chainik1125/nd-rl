"""Lean 4 as the judge for ``lean_seq`` samples.

``judge(tok, prompts, samples)`` decides, for each sampled id sequence, whether it is a proof of
its prompt's theorem: the sample must parse in the strict ``lean_seq`` grammar (``tok.decode``), and
Lean must accept the literal text, ``tok.statement(prompt) + " " + tok.lean_text(ids)``. Nothing
else is consulted; in particular no ND verifier. This is the take-home fork's rule ("Lean alone
decides", 2026-09-27; ``lean_judge.py`` / ``lean_gate.py`` on ``dan-pandori/nd-takehome``), ported
without its caches, pre-filter and logging.

Why this is sound:

- The grammar check runs first and consumes the whole sequence (balanced boxes, final ``exact``,
  ``<eos>``), so a truncated text that Lean's parse-error recovery might accept never reaches Lean.
- The 107-token vocabulary has no ``sorry``, no tactic automation and no library lemma: the only
  inferences beyond the ND rules are the eliminators ``.1`` / ``.2`` / ``.elim`` resolve to (e.g.
  ``Not.elim`` for ``n.elim`` on ``n : ¬A``, which ND's ``BOTE`` forbids but which is valid).
- Any error Lean attributes to a theorem's lines rejects it, and so does a ``sorry`` (which Lean
  reports only as a warning; it cannot occur in ``lean_seq`` text, but ``check_theorems`` also takes
  free-form theorems). A Lean process that crashes, times out,
  or reports an error outside every theorem's lines is split and re-run; a theorem that still fails
  on its own is rejected.

``check_theorems`` is the batched checker underneath: many theorems per Lean process (a process
costs about a second to start, so checking one theorem at a time inside a loop is very slow), one
process per CPU.
"""

import bisect
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from code.shared_code.lean_seq import LeanSeqTokenizer, NDParseError

PRELUDE = "set_option linter.unusedVariables false\nset_option maxRecDepth 4000\n"
CHUNK = 400
# An error, or the warning Lean gives instead of an error for `sorry`: either rejects the theorem.
_ERROR = re.compile(r"^[^\n]*?:(\d+):\d+: (?:error|warning: declaration uses [`']sorry[`'])", re.M)


def find_lean() -> str | None:
    """The ``lean`` executable: ``$LEAN``, else ``lean`` on ``PATH``, else ``~/.elan/bin/lean``."""
    env = os.environ.get("LEAN")
    if env:
        return env
    found = shutil.which("lean")
    if found:
        return found
    elan = Path.home() / ".elan" / "bin" / "lean"
    return str(elan) if elan.exists() else None


def failing_theorems(output: str, starts: Sequence[int]) -> tuple[set[int], bool]:
    """Map Lean's error lines (and ``sorry`` warnings) to theorem indices.

    ``starts[k]`` is the first line (1-based) of theorem ``k`` in the checked file, in increasing
    order. Returns the indices with at least one error, and whether any error fell outside every
    theorem (before the first one, i.e. in the prelude).
    """
    bad: set[int] = set()
    stray = False
    for m in _ERROR.finditer(output):
        k = bisect.bisect_right(starts, int(m.group(1))) - 1
        if 0 <= k < len(starts):
            bad.add(k)
        else:
            stray = True
    return bad, stray


def _check_chunk(lean: str, sources: Sequence[str], workdir: str, tag: str) -> list[bool]:
    text = PRELUDE
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
            timeout=60 + 2 * text.count("\n"),
        )
        output, rc = run.stdout + run.stderr, run.returncode
    except subprocess.TimeoutExpired:
        output, rc = "", -9
    finally:
        os.remove(path)
    bad, stray = failing_theorems(output, starts)
    if rc not in (0, 1) or (rc == 1 and not bad) or stray:
        # Crashed, timed out, or failed without naming a theorem: split, and reject a lone theorem.
        if len(sources) == 1:
            return [False]
        h = len(sources) // 2
        return _check_chunk(lean, sources[:h], workdir, tag + "a") + _check_chunk(
            lean, sources[h:], workdir, tag + "b"
        )
    return [k not in bad for k in range(len(sources))]


def check_theorems(
    sources: Sequence[str],
    *,
    lean: str | None = None,
    chunk: int = CHUNK,
    workers: int | None = None,
) -> list[bool]:
    """Whether Lean 4 core accepts each theorem source, in order.

    Each source is one theorem that starts ``theorem t `` (it is renamed per batch) and may span
    several lines. Raises ``FileNotFoundError`` if no Lean is found.
    """
    if not sources:
        return []
    exe = lean or find_lean()
    if exe is None:
        raise FileNotFoundError("Lean 4 not found: install it with elan, or set $LEAN")
    for src in sources:
        if not src.startswith("theorem t "):
            raise ValueError(f"a theorem source must start 'theorem t ': {src[:60]!r}")
    if workers is None:
        affinity = getattr(os, "sched_getaffinity", None)
        workers = len(affinity(0)) if affinity else (os.cpu_count() or 1)
    pieces = [(list(sources[i : i + chunk]), str(i)) for i in range(0, len(sources), chunk)]
    workdir = tempfile.mkdtemp(prefix="lean_check_")
    try:
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(lambda p: _check_chunk(exe, p[0], workdir, p[1]), pieces))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return [ok for piece in results for ok in piece]


@dataclass(frozen=True)
class Verdict:
    """``ok``: the sample proves its theorem. ``reason``: why not (empty when ok). ``nd``: the ND
    body the sample denotes (a ``LEANPARSE`` marker when it is outside the grammar)."""

    ok: bool
    reason: str
    nd: str


def judge(
    tok: LeanSeqTokenizer,
    prompts: Sequence[str],
    samples: Sequence[Sequence[int]],
    *,
    lean: str | None = None,
    workers: int | None = None,
) -> list[Verdict]:
    """Judge sampled proof ids (the tokens after the prompt) against their ND prompts, in order.

    Each distinct (statement, text) pair is checked once, all of them in one batched call; samples
    outside the grammar are rejected without running Lean.
    """
    if len(prompts) != len(samples):
        raise ValueError("prompts and samples differ in length")
    verdicts: list[Verdict | None] = [None] * len(samples)
    pending: dict[str, list[tuple[int, str]]] = {}
    for i, (prompt, ids) in enumerate(zip(prompts, samples, strict=True)):
        nd = tok.decode(ids)
        if nd.startswith("LEANPARSE"):
            verdicts[i] = Verdict(False, "lean_seq parse: " + nd[len("LEANPARSE ") :], nd)
            continue
        try:
            statement = tok.statement(prompt)
        except (NDParseError, KeyError) as e:
            verdicts[i] = Verdict(False, f"prompt: {e}", nd)
            continue
        body = tok.lean_text(ids)
        assert body is not None  # decode succeeded, so the sample reached <eos>
        pending.setdefault(f"{statement} {body}", []).append((i, nd))
    sources = list(pending)
    for src, ok in zip(sources, check_theorems(sources, lean=lean, workers=workers), strict=True):
        for i, nd in pending[src]:
            verdicts[i] = Verdict(ok, "" if ok else "lean rejected", nd)
    done = [v for v in verdicts if v is not None]
    assert len(done) == len(samples)
    return done
