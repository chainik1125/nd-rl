"""`MinlenCertifier`: the WP0 `Certifier` Protocol backed by the vendored bounded minimal-proof-
length search (`vendor/takehome/minlen.py`; plan §2, §3's WP2 row).

Semantics (plan §2's four `LengthLabel.method` values):

* `minlen_exact` -- iterative deepening (`minlen()`'s own `Search.run`) found a proof and did not
  time out. `minlen.py`'s docstring is explicit that this is exact *within its restricted search
  space* (subformulas of the theorem, their negations/double negations, plus an explicit `LEMMA`
  step): iterative deepening always tries every smaller total line count first, so any definite,
  non-timed-out result is that search space's true minimum, not merely an upper bound found along
  the way.
* `ub_lb` -- used only when a caller supplies an already-verified witness proof of length `L_ub`.
  If the direct `minlen(bound=L_ub)` call times out, we fall back to the *cheaper* complementary
  check: does `minlen(bound=L_ub - 1)` exhaust (return `None`, not a timeout)? That proves nothing
  shorter than `L_ub` exists in the search space without ever having to re-search the expensive top
  bound itself, so the witness's own length becomes the certified `L` (`L == L_ub == L_lb`).
* `witness_argument` -- never produced by `MinlenCertifier` itself. This method is for a family's
  *mathematical* lower-bound argument (WP3), not anything derived from a minlen search; it is
  simply a value other code may put in `LengthLabel.method` directly. `MinlenCertifier.certify`
  never overwrites or second-guesses a label with this method -- there is no code path here that
  produces it, by design (see the module-level docstring note above).
* `reference` -- produced when nothing can be certified but a verified witness's own length is at
  least known (e.g. `textbook_dev`'s legacy `L_true`/`n_lines`, which plan §0 notes are *reference*
  lengths, not minimal: `~(P>Q) ⊢ P&~Q` is labelled 18 there but minlen finds 12). This happens
  both when the witness search times out entirely and when it *exhausts* without ever matching the
  witness's own length (the witness proof uses hypotheses/lemmas outside minlen's restricted
  search space -- plausible for constructed, glued proofs; minlen's own docstring warns that a
  `None` result is never proof no shorter proof exists).

When no witness is supplied at all and minlen cannot find a proof within `bound` (whether by
timing out or by exhausting the bound), the label carries `L=None` under `method="minlen_exact"`
(there is no witness to fall back to, so `"reference"` does not apply); callers should treat this
as "could not be certified" and typically drop the item.

Labels are cached by `Sequent.key()` (an atom-renaming class key, plan §2/§3: "cache labels by
sequent key"), both in-memory for the certifier's lifetime and, if `cache_dir` is given, in a
`minlen_labels.jsonl` file under it (plan §A2: "cache every minlen label under
.../bench/out/cache/"). Because the cache is keyed by `key()` alone (not by `bound`/`time_s`/
witness), the *first* successful certification of a given theorem class wins for the lifetime of
that cache; a caller that needs a fresh certification of an already-cached key should use a
different `cache_dir` (or clear it).

Parallelism: `procs` workers (each handling one sequent's full two-phase procedure end to end, so
no cross-process coordination is needed beyond `multiprocessing.Pool.map`) via `fork` (Linux
default; explicit here for determinism). Plan §A2 caps every ndbench agent at *8* minlen worker
processes at a time, so `procs` is silently clamped to that even if a caller asks for more.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.io import append_jsonl, read_jsonl
from ..core.types import LengthLabel, Sequent
from ..vendor import ensure_takehome_on_path

#: plan §A2: "each agent uses <= 8 minlen worker processes at a time".
MAX_PROCS = 8

#: `certify()`'s `bound=None` default target (matches the plan's common minlen bound, e.g.
#: `textbook_dev`'s "bound 16, 120 s").
DEFAULT_BOUND = 16

#: `certify()`'s `time_s` default already matches the `Certifier` Protocol's own default (60.0);
#: re-exported here only for readability at call sites that want to name it explicitly.
DEFAULT_TIME_S = 60.0


def _run_minlen(prompt: str, bound: int, time_s: float) -> dict[str, Any]:
    """One vendored `minlen()` call, importing it lazily (safe under both `fork` and `spawn`)."""
    ensure_takehome_on_path()
    import minlen as _minlen  # type: ignore[import-not-found]

    result: dict[str, Any] = _minlen.minlen(prompt, bound=bound, time_limit=time_s)
    return result


def _verify_witness(prompt: str, witness: str) -> int:
    """Written line count of `witness` as a proof of `prompt`; raises `ValueError` if it does not
    verify. `certify()`'s witnesses are documented as already-valid ND proofs of their sequent
    (the "verified witness" in plan §2's `ub_lb` description); this is a defensive check that
    turns a caller bug into a clear error rather than a confusing downstream minlen result."""
    ensure_takehome_on_path()
    from nd_verify import verify_text  # type: ignore[import-not-found]

    ok, reason, n_lines = verify_text(prompt + " " + witness)
    if not ok:
        raise ValueError(f"witness proof does not verify against {prompt!r}: {reason}")
    return int(n_lines)


def _certify_one(payload: tuple[str, str | None, int | None, float]) -> dict[str, Any]:
    """The full one-item certification procedure (both phases when a witness is given); run in a
    worker process by `MinlenCertifier.certify`. Returns a plain (picklable) dict with
    `LengthLabel`'s field names so the parent process can reconstruct the dataclass."""
    prompt, witness, bound, time_s = payload

    if witness is not None:
        l_ub = _verify_witness(prompt, witness)
        u = _run_minlen(prompt, l_ub, time_s)
        if u["timeout"]:
            below = _run_minlen(prompt, l_ub - 1, time_s)
            if below["timeout"]:
                return {
                    "L": None,
                    "L_ub": l_ub,
                    "L_lb": None,
                    "method": "reference",
                    "bound": l_ub,
                    "timeout": True,
                }
            if below["min_lines_ub"] is None:
                # Exhausted every total < l_ub without a timeout: l_ub is confirmed exact.
                return {
                    "L": l_ub,
                    "L_ub": l_ub,
                    "L_lb": l_ub,
                    "method": "ub_lb",
                    "bound": l_ub - 1,
                    "timeout": False,
                }
            # A shorter proof exists after all; the exhaustion search itself found it exactly.
            shorter = int(below["min_lines_ub"])
            return {
                "L": shorter,
                "L_ub": shorter,
                "L_lb": None,
                "method": "minlen_exact",
                "bound": l_ub - 1,
                "timeout": False,
            }
        if u["min_lines_ub"] is not None:
            found = int(u["min_lines_ub"])
            return {
                "L": found,
                "L_ub": found,
                "L_lb": None,
                "method": "minlen_exact",
                "bound": l_ub,
                "timeout": False,
            }
        # Exhausted at l_ub without ever matching the witness: the witness must use hypotheses or
        # lemmas outside minlen's restricted search space (see the module docstring).
        return {
            "L": None,
            "L_ub": l_ub,
            "L_lb": None,
            "method": "reference",
            "bound": l_ub,
            "timeout": False,
        }

    eff_bound = bound if bound is not None else DEFAULT_BOUND
    r = _run_minlen(prompt, eff_bound, time_s)
    found_l = r["min_lines_ub"]
    return {
        "L": int(found_l) if found_l is not None else None,
        "L_ub": int(found_l) if found_l is not None else None,
        "L_lb": None,
        "method": "minlen_exact",
        "bound": eff_bound,
        "timeout": bool(r["timeout"]),
    }


class MinlenCertifier:
    """`Certifier` (WP0 `core.types.Certifier` Protocol) backed by the vendored `minlen.py`.

    See the module docstring for the exact `method` semantics. `mem_gb` is advisory only (plan §A2
    notes minlen's own interning can use ~1.4-2 GB/worker at high bounds); `MinlenCertifier` does
    not itself enforce a memory limit, since worker COUNT (`procs`, already capped at 8) is the
    lever ndbench agents actually control on the shared machine.
    """

    def __init__(
        self,
        procs: int = 8,
        mem_gb: float = 2.0,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.procs = max(1, min(procs, MAX_PROCS))
        self.mem_gb = mem_gb
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_path: Path | None = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path = self.cache_dir / "minlen_labels.jsonl"
            if self._cache_path.exists():
                for row in read_jsonl(self._cache_path):
                    self._cache[row["key"]] = row["label"]

    def _remember(self, key: str, label: dict[str, Any]) -> None:
        self._cache[key] = label
        if self._cache_path is not None:
            append_jsonl(self._cache_path, [{"key": key, "label": label}])

    def certify(
        self,
        seqs: Sequence[Sequent],
        witnesses: Sequence[str | None],
        *,
        bound: int | None = None,
        time_s: float = 60.0,
    ) -> list[LengthLabel]:
        seqs = list(seqs)
        witnesses = list(witnesses)
        if len(witnesses) != len(seqs):
            raise ValueError(
                f"witnesses must have the same length as seqs ({len(witnesses)} != {len(seqs)})"
            )

        keys = [s.key() for s in seqs]
        # De-duplicate work within this call too: several seqs can share a key (atom-renamed
        # variants of the same theorem class), and only the first occurrence needs computing.
        todo: dict[str, tuple[str, str | None]] = {}
        for k, seq, w in zip(keys, seqs, witnesses, strict=True):
            if k not in self._cache and k not in todo:
                todo[k] = (seq.prompt(), w)

        if todo:
            todo_keys = list(todo)
            payloads = [(*todo[k], bound, time_s) for k in todo_keys]
            if self.procs > 1 and len(payloads) > 1:
                ctx = mp.get_context("fork")
                with ctx.Pool(self.procs) as pool:
                    results = pool.map(_certify_one, payloads)
            else:
                results = [_certify_one(p) for p in payloads]
            for k, res in zip(todo_keys, results, strict=True):
                self._remember(k, res)

        return [LengthLabel(**self._cache[k]) for k in keys]
