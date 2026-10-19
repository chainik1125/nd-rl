"""`LeanJudge` v2 (plan §2, WP-D2): judges proof attempts by checking Lean 4 source text, exactly
mirroring Dan Pandori's `lean_judge.py`/`lean_gate.py` decision order (fork `dan-pandori/
nd-takehome`, HEAD `0c492902`; read directly in the read-only clone
`/scratch/cowifornia/nd-rl/bench/nd-takehome-dan`, since `lean_judge.py` itself is deliberately
not vendored -- plan §7 WP-D2's job is to *replicate* it, not import it).

**Decision order for one attempt** (mirrors `lean_judge.judge_many`'s three cases exactly,
`nd-takehome-dan/lean_judge.py:76-105`):

1. `Attempt.body` starts with our own `"UNPARSEABLE "` marker (the adapter's decoder never
   produced ND text at all) -> reject, `kind="unparseable"`, no Lean involved. This is the
   `Judge`-protocol convention every judge in this package already uses (`judge.nd.NDJudge`,
   the pre-WP-D2 `judge.lean`), not one of Dan's own markers (`LEANPARSE`/`LEANREJ`): those live
   inside Dan's own pipeline, upstream of the `ndbench` adapter boundary, which always hands this
   module a decoded `Attempt.body`.
2. `Attempt.native` decodes (`json.loads`) to a dict with a non-empty string `"lean"` key: this is
   a Dan-model sample whose `native["lean"]` is *exactly* the literal Lean tactic text Dan's own
   `lean_gate.gate(tok, prompts, nd_proofs, texts)` would have received as `texts[i]` (WP-D1's
   adapter contract). Check `statement(prompt) + " " + native["lean"]` in Lean -- precisely what
   `lean_gate.Gate._dispatch` builds (`nd-takehome-dan/lean_gate.py:157`: `srcs = [f'{self.stmt[p]}
   {tx}' for p, tx in keys]`, `self.stmt[p] = self.tok.statement(p)`), where `tok.statement(prompt)
   == ' '.join(prompt_tokens(prompt))` (`vendor/danstate/lean_tok.py:206-207`).
3. Otherwise (no usable native Lean text -- our own whole-proof models, or any sample without
   native lean text at all): translate `Attempt.body` with the vendored `nd2lean.translate(prompt,
   body, require_all_pr=False)` (`nd-takehome-dan/lean_judge.py:90`: "translate the ND proof ...
   with require_all_pr=False: Lean does not require a premise to be re-stated") and check the
   result in Lean. A translation failure (`TranslationError`, a `nd_verify` `ParseError` from
   `parse_prompt`/`parse_proof_tokens`, or any other exception `nd2lean.translate` can raise) is a
   rejection with a `"nd2lean: <exception>"` reason, matching `lean_judge.judge_many`'s own
   `except Exception as e: res[i] = (False, f'nd2lean: {e}', 0)` -- never a crash.

Lean itself decides nothing else: no Mathlib, no `sorry` (the vocabulary this module ever submits
to Lean -- `nd2lean`'s deterministic translation and Dan's `lean_seq` grammar -- has no `sorry`
token at all), no tactic automation beyond what `nd2lean.translate`/the literal sampled text
already spell out.

**Batching** (`check_sources`/`_run`/`_failing_theorems` below): a from-scratch reimplementation
of `lean_gate.check_sources`/`_run` (`nd-takehome-dan/lean_gate.py:62-115`), not a copy -- that
file lives outside this work package's file list and is never imported at runtime (same
convention the pre-WP-D2 `judge.lean` used for the same reason; see git history). Matches Dan's
batching precisely: one `PRELUDE` per Lean process (`set_option linter.unusedVariables false` +
`set_option maxRecDepth 4000`), chunks of 400 sources, `lean -j 1 -DmaxErrors=100000000 <file>`,
a timeout of `60 + 2 * n_lines` seconds, error lines mapped back to the theorem they fall in via
`bisect` over each theorem's starting line, and -- on a crash, a timeout, or any error Lean
reports outside every theorem's own range ("stray") -- the chunk is split in half and re-run
recursively, down to rejecting a lone theorem that still cannot be isolated. Explicitly **no
prefilter** (`lean_prefilter.py` is not vendored -- plan §A5/§2: "no prefilter"), and capped at
`_MAX_WORKERS` (8) Lean worker processes regardless of this host's real core count, per the
shared-host rule in `AGENTS.md`/this work package's instructions (Dan's own default is
`cpu_quota()`, unbounded on a dedicated host).

**Verdict cache**: a sqlite memo at `<cache_dir>/lean_v2/verdicts.sqlite`, keyed by the sha256 of
the *full* rendered Lean source (statement + proof together, so a cache hit never depends on
having re-derived the statement correctly) -- `judge_many` checks the cache before dispatching any
Lean process at all, and only ever checks new (uncached) sources.

**Not installed here / wrong version.** `LeanJudge()` raises `judge.base.NotAvailable` if no
`lean` binary is found (same `find_lean` order as before: `$LEAN`, `PATH`, `~/.elan/bin/lean`) or
if the resolved binary does not report exactly Lean `4.34.0` (`LEAN_VERSION`, plan §A5's pinned
version) -- this environment's `~/.elan/bin/lean` *is* 4.34.0 (checked directly: `lean --version`
-> `Lean (version 4.34.0, ...)`), so `code/tests/ndbench/test_dan_lean_judge.py`'s real-Lean tests
run rather than skip here; the fake-`lean`-binary tests in that file still exercise the rest of the
pipeline (rendering/batching/dedup/cache/error-parsing) without depending on that.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..core.proof import first_index, line_count, pruned_length
from ..core.types import Attempt, Sequent, Verdict
from ..vendor.danstate_loader import DanState, load_danstate
from .base import Judge, NotAvailable

#: Exactly `nd-takehome-dan/lean_gate.py`'s `PRELUDE` (module docstring's "Batching").
_PRELUDE = "set_option linter.unusedVariables false\nset_option maxRecDepth 4000\n"
#: Exactly `lean_gate.py`'s `CHUNK` default (`LEAN_GATE_CHUNK`, default `'400'`).
_CHUNK = 400
#: This work package's hard cap on concurrent `lean` worker processes (shared 48-CPU host; see
#: `AGENTS.md`/this WP's instructions), overriding Dan's own unbounded `cpu_quota()` default.
_MAX_WORKERS = 8
#: Exactly `lean_gate.py`'s `ERR` regex (`nd-takehome-dan/lean_gate.py:58`) -- error lines only, no
#: `sorry` clause: the `lean_seq`/`nd2lean` vocabulary this module ever submits to Lean has no
#: `sorry` token, so a `sorry` warning can never appear in real output (module docstring).
_ERROR_RE = re.compile(r"^[^\n]*?:(\d+):\d+: error", re.M)
#: Our own `Attempt.body` marker (the `Judge`-protocol-wide convention; not one of Dan's own
#: `LEANPARSE`/`LEANREJ` markers -- module docstring's decision-order step 1).
_UNPARSEABLE = "UNPARSEABLE"
#: Pinned Lean version (plan §A5); `LeanJudge.__init__` raises `NotAvailable` on any mismatch.
LEAN_VERSION = "4.34.0"

_VERSION_RE = re.compile(r"version (\d+\.\d+\.\d+)")


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


def lean_version(lean: str) -> str | None:
    """`lean --version`'s reported version (e.g. `"4.34.0"`), or `None` if it cannot be run or
    parsed (a missing/non-executable path, or a binary -- real or faked for a test -- whose
    `--version` output doesn't match Lean's `"... version X.Y.Z ..."` format)."""
    try:
        run = subprocess.run([lean, "--version"], capture_output=True, text=True, timeout=30)
    except OSError:
        return None
    m = _VERSION_RE.search(run.stdout + run.stderr)
    return m.group(1) if m else None


def _failing_theorems(output: str, starts: Sequence[int]) -> tuple[set[int], bool]:
    """Map Lean's error lines to theorem indices (module docstring's "Batching").

    `starts[k]` is the first line (1-based) of theorem `k` in the checked file, increasing.
    Returns the indices with at least one error, and whether any error fell outside every
    theorem (before the first one, i.e. in the prelude, or past the last).
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


def _run_chunk(
    lean: str, sources: Sequence[str], workdir: str, tag: str, timeout_s: float
) -> list[bool]:
    """One Lean process over `sources` (each `"theorem t ..."`, renamed `t0`, `t1`, ... in the
    checked file); splits and retries recursively on a crash/timeout/stray error, exactly
    `lean_gate.py`'s `_run` (module docstring's "Batching")."""
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
            [lean, "-j", "1", "-DmaxErrors=100000000", path],
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
    crashed = rc not in (0, 1) or (rc == 1 and not bad) or stray
    if crashed:
        if len(sources) == 1:
            return [False]
        h = len(sources) // 2
        return _run_chunk(lean, sources[:h], workdir, tag + "a", timeout_s) + _run_chunk(
            lean, sources[h:], workdir, tag + "b", timeout_s
        )
    return [k not in bad for k in range(len(sources))]


def check_sources(
    sources: Sequence[str],
    *,
    lean: str,
    chunk: int = _CHUNK,
    workers: int | None = None,
    timeout_s: float = 60.0,
) -> list[bool]:
    """Whether Lean 4 core accepts each theorem source, in order; batched into `chunk`-sized
    pieces run concurrently across up to `workers` (capped at `_MAX_WORKERS`) `lean` processes
    (module docstring's "Batching"). `lean` must already be a resolved executable path.
    """
    if not sources:
        return []
    for src in sources:
        if not src.startswith("theorem t "):
            raise ValueError(f"a theorem source must start 'theorem t ': {src[:60]!r}")
    n_workers = _MAX_WORKERS if workers is None else max(1, min(workers, _MAX_WORKERS))
    pieces = [(list(sources[i : i + chunk]), str(i)) for i in range(0, len(sources), chunk)]
    workdir = tempfile.mkdtemp(prefix="ndbench_lean_v2_")
    try:
        with ThreadPoolExecutor(n_workers) as pool:
            results = list(
                pool.map(lambda p: _run_chunk(lean, p[0], workdir, p[1], timeout_s), pieces)
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return [ok for piece in results for ok in piece]


def _parse_native(native: str | None) -> dict[str, Any] | None:
    """`Attempt.native` (a JSON string, or `None`) -> the decoded dict, or `None` if it is absent
    or not a JSON object (never raises)."""
    if native is None:
        return None
    try:
        decoded = json.loads(native)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


class _VerdictCache:
    """Sqlite memo of `sha256(lean source) -> ok` at `<cache_dir>/lean_v2/verdicts.sqlite` (module
    docstring's "Verdict cache"). Every read/write happens on the thread that calls `judge_many`
    (never inside the `ThreadPoolExecutor` workers), so no locking beyond sqlite's own is needed.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Several runs (one per GPU) share one cache file: WAL lets readers proceed during a
        # write, and the long timeout makes a writer wait for another's commit instead of failing.
        self._conn = sqlite3.connect(str(path), timeout=120.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS verdicts (sha256 TEXT PRIMARY KEY, ok INTEGER NOT NULL)"
        )
        self._conn.commit()

    def get_many(self, shas: Sequence[str]) -> dict[str, bool]:
        if not shas:
            return {}
        out: dict[str, bool] = {}
        # sqlite's default compile-time limit on bound parameters (SQLITE_MAX_VARIABLE_NUMBER) is
        # at least 999; batch well under that regardless.
        for i in range(0, len(shas), 500):
            batch = shas[i : i + 500]
            qs = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT sha256, ok FROM verdicts WHERE sha256 IN ({qs})", batch
            ).fetchall()
            out.update({sha: bool(ok) for sha, ok in rows})
        return out

    def put_many(self, items: dict[str, bool]) -> None:
        if not items:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO verdicts (sha256, ok) VALUES (?, ?)",
            [(sha, int(ok)) for sha, ok in items.items()],
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class LeanJudge(Judge):
    """Judges attempts by rendering them to Lean 4 source and calling `lean`, replicating Dan
    Pandori's `lean_judge`/`lean_gate` decision order exactly (module docstring)."""

    judge_id = "lean_v2"

    def __init__(
        self,
        *,
        lean: str | None = None,
        cache_dir: str | Path | None = None,
        timeout_s: float = 60.0,
        workers: int | None = None,
        chunk: int = _CHUNK,
    ) -> None:
        resolved = lean if lean is not None else find_lean()
        if resolved is None:
            raise NotAvailable(
                "no `lean` binary found ($LEAN, PATH, or ~/.elan/bin/lean); install Lean 4 (e.g. "
                "via elan) or set $LEAN to use LeanJudge"
            )
        version = lean_version(resolved)
        if version != LEAN_VERSION:
            raise NotAvailable(
                f"LeanJudge requires lean {LEAN_VERSION}, found {version!r} at {resolved!r} "
                "(plan §A5: Lean 4.34.0 is pinned)"
            )
        self._lean = resolved
        self._timeout_s = timeout_s
        self._workers = workers
        self._chunk = chunk
        self._cache_path: Path | None = (
            Path(cache_dir) / "lean_v2" / "verdicts.sqlite" if cache_dir is not None else None
        )
        self._ds: DanState | None = None

    def _danstate(self) -> DanState:
        if self._ds is None:
            self._ds = load_danstate()
        return self._ds

    def _source_for(self, prompt: str, attempt: Attempt) -> tuple[str | None, str | None]:
        """`(source, error_reason)` for one (already non-`"UNPARSEABLE "`) attempt: exactly one is
        `None` (module docstring's decision-order steps 2/3)."""
        native = _parse_native(attempt.native)
        lean_text = native.get("lean") if native is not None else None
        if isinstance(lean_text, str) and lean_text:
            ds = self._danstate()
            try:
                statement = " ".join(ds.lean_tok.prompt_tokens(prompt))
            except Exception as e:  # ValueError (too many premises), KeyError, ...
                return None, f"statement: {type(e).__name__}: {e}"
            return f"{statement} {lean_text}", None
        ds = self._danstate()
        try:
            source = ds.nd2lean.translate(prompt, attempt.body, require_all_pr=False)
        except Exception as e:  # TranslationError, nd_verify ParseError, ValueError, ...
            return None, f"nd2lean: {type(e).__name__}: {e}"
        return source, None

    def judge(
        self, seq: Sequent, attempts: Sequence[Attempt], ref_lines: int | None
    ) -> list[Verdict]:
        return self.judge_many([(seq, attempts, ref_lines)])[0]

    def judge_many(
        self, items: Sequence[tuple[Sequent, Sequence[Attempt], int | None]]
    ) -> list[list[Verdict]]:
        """One batched Lean invocation (at most) across every attempt in every `(seq, attempts,
        ref_lines)` group, deduplicated by rendered source text and memoised in the verdict cache
        (module docstring). `ref_lines` is accepted for the `Judge` protocol and ignored."""
        out: list[list[Verdict | None]] = [[None] * len(attempts) for _, attempts, _ in items]
        pending: dict[str, list[tuple[int, int]]] = {}  # source -> [(item_idx, attempt_idx), ...]

        for item_idx, (seq, attempts, _ref_lines) in enumerate(items):
            prompt = seq.prompt()
            for attempt_idx, a in enumerate(attempts):
                if a.body.startswith(_UNPARSEABLE):
                    reason = a.body[len(_UNPARSEABLE) :].strip() or "unparseable"
                    out[item_idx][attempt_idx] = Verdict(
                        ok=False,
                        reason=reason,
                        kind="unparseable",
                        written=None,
                        pruned=None,
                        first_bad_line=None,
                        start=None,
                    )
                    continue
                source, err = self._source_for(prompt, a)
                if source is None:
                    out[item_idx][attempt_idx] = Verdict(
                        ok=False,
                        reason=err or "unrenderable",
                        kind="unparseable",
                        written=line_count(a.body),
                        pruned=None,
                        first_bad_line=None,
                        start=first_index(a.body),
                    )
                    continue
                pending.setdefault(source, []).append((item_idx, attempt_idx))

        sha_of: dict[str, str] = {s: hashlib.sha256(s.encode()).hexdigest() for s in pending}
        cache = _VerdictCache(self._cache_path) if self._cache_path is not None else None
        try:
            known: dict[str, bool] = (
                cache.get_many([sha_of[s] for s in pending]) if cache is not None else {}
            )
            ok_by_sha: dict[str, bool] = dict(known)
            todo = [s for s in pending if sha_of[s] not in ok_by_sha]
            if todo:
                oks = check_sources(
                    todo,
                    lean=self._lean,
                    chunk=self._chunk,
                    workers=self._workers,
                    timeout_s=self._timeout_s,
                )
                fresh: dict[str, bool] = {}
                for source, ok in zip(todo, oks, strict=True):
                    sha = sha_of[source]
                    ok_by_sha[sha] = ok
                    fresh[sha] = ok
                if cache is not None:
                    cache.put_many(fresh)
        finally:
            if cache is not None:
                cache.close()

        for source, refs in pending.items():
            ok = ok_by_sha[sha_of[source]]
            for item_idx, attempt_idx in refs:
                a = items[item_idx][1][attempt_idx]
                out[item_idx][attempt_idx] = Verdict(
                    ok=ok,
                    reason="" if ok else "lean rejected",
                    kind="ok" if ok else "bad_line",
                    written=line_count(a.body),
                    pruned=pruned_length(a.body) if ok else None,
                    first_bad_line=None,
                    start=first_index(a.body),
                )

        done: list[list[Verdict]] = []
        for group in out:
            assert all(v is not None for v in group)  # every index filled (unparseable or pending)
            done.append(group)  # type: ignore[arg-type]
        return done
