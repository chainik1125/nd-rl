"""Shared data loading for the interp measures (`INTERP_SPEC.md`'s "Data" section): neutral
(reserve/sealed) proofs with a minimal reference proof, the two checkpoints' paths, and pulling
verified model samples back out of the ordinary `Store` (`runner.run` results with
`keep_samples=True`).

Every loader here drops a candidate the `ExclusionIndex` marks `seen_train`; none of it flags
eval-tier membership (unneeded for this package -- `build_exclusion_index` scans only the train
tier, considerably cheaper than `logic.exclusion.default_tiers()`'s full layout, which also scans
the 426k-row `ab2/data/reserve.jsonl` as an eval-tier file for other work packages' purposes).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.io import read_jsonl
from ..core.types import Excluder, Sequent
from ..logic.exclusion import ROOT, TRAIN_FILES, ExclusionIndex
from ..store import Store

#: `INTERP_SPEC.md`: "Models: base `ab2/frozen/pretrain.pt`, RL
#: `ab2/monitor/final/champion2_n060_fullpool/final.pt`".
BASE_CKPT = ROOT / "ab2" / "frozen" / "pretrain.pt"
RL_CKPT = ROOT / "ab2" / "monitor" / "final" / "champion2_n060_fullpool" / "final.pt"

RESERVE_PATH = ROOT / "ab2" / "data" / "reserve.jsonl"
SEALED_L12_14_PATH = ROOT / "ab2" / "sealed" / "sealed_L12-14.jsonl"
DEV_L8_11_PATH = ROOT / "ab2" / "data" / "dev_L8-11.jsonl"


def line_start_positions(body: str) -> list[int]:
    """0-based positions, within `body.split()` (which `Tokenizer.encode_proof` maps 1:1 to ids --
    see `vendor/absgpt/tokenizer.py`: one output id per input token, always), of every line's own
    leading `N<i>` index token (the position "before that line's content is written", plan
    §4/probes' targets). Mirrors `encode_proof`'s own `cur is None or toks[i-1] == ';'` rule for
    telling a line's own index apart from a same-looking citation token, without needing a live
    `Tokenizer` instance (`abs`-mode citations and line indices share one id space, so this can't
    be recovered from the *ids* alone)."""
    toks = body.split()
    out: list[int] = []
    for i, t in enumerate(toks):
        if not (t.startswith("N") and t[1:].isdigit()):
            continue
        if i == 0 or toks[i - 1] == ";":
            out.append(i)
    return out


@dataclass(frozen=True)
class NeutralProof:
    pid: str
    origin: str  # "reserve" | "sealed_L12-14" | "family"
    sequent: Sequent
    body: str  # minlen/witness proof, N1-based, ends in QED
    L: int


def _seq_of(row: Mapping[str, Any]) -> Sequent:
    thm = row.get("thm")
    if thm is not None:
        return Sequent.from_thm(thm)
    return Sequent.from_prompt(row["prompt"])


def build_exclusion_index(cache_dir: Path | str | None = None) -> ExclusionIndex:
    return ExclusionIndex.build({"train": list(TRAIN_FILES)}, cache_dir=cache_dir)


def load_neutral_proofs(
    *,
    n: int,
    l_min: int = 6,
    l_max: int = 10,
    excluder: Excluder | None = None,
    seed: int = 0,
    reserve_path: Path = RESERVE_PATH,
    scan_limit: int | None = None,
) -> list[NeutralProof]:
    """Up to `n` `source == "gen"` reserve rows with a `minlen_proof` in `[l_min, l_max]`, not
    `seen_train`, reservoir-sampled (deterministic given `seed`) so this streams the 426k-row file
    once without loading it whole. `scan_limit` caps how many rows are read (fast tests).

    `excluder` is any `Excluder` (structural: an `ExclusionIndex` from `build_exclusion_index`,
    or `logic.exclusion.DanExcluder` for Dan's own training tiers -- `interp/danstate_support.py`
    WP-D6), not only the absgpt-specific `ExclusionIndex` this function originally took."""
    rng = random.Random(seed)
    reservoir: list[NeutralProof] = []
    n_seen = 0
    for i, row in enumerate(read_jsonl(reserve_path)):
        if scan_limit is not None and i >= scan_limit:
            break
        if row.get("source") != "gen" or not row.get("minlen_proof"):
            continue
        l_true = row.get("L_true")
        if not isinstance(l_true, int) or not (l_min <= l_true <= l_max):
            continue
        seq = _seq_of(row)
        if excluder is not None and excluder.status(seq)[0]:
            continue
        item = NeutralProof(
            pid=str(row["name"]),
            origin="reserve",
            sequent=seq,
            body=row["minlen_proof"],
            L=l_true,
        )
        n_seen += 1
        if len(reservoir) < n:
            reservoir.append(item)
        else:
            j = rng.randint(0, n_seen - 1)
            if j < n:
                reservoir[j] = item
    return reservoir


def load_sealed_l12_14(
    *,
    n: int | None = None,
    excluder: Excluder | None = None,
    seed: int = 0,
    path: Path = SEALED_L12_14_PATH,
) -> list[NeutralProof]:
    """Every `minlen_proof` row of `ab2/sealed/sealed_L12-14.jsonl` (1,634 rows -- small enough to
    load whole), or an `n`-sized random sample of them."""
    items: list[NeutralProof] = []
    for row in read_jsonl(path):
        if not row.get("minlen_proof"):
            continue
        seq = _seq_of(row)
        if excluder is not None and excluder.status(seq)[0]:
            continue
        items.append(
            NeutralProof(
                pid=str(row["name"]),
                origin="sealed_L12-14",
                sequent=seq,
                body=row["minlen_proof"],
                L=int(row["L_true"]),
            )
        )
    if n is not None and len(items) > n:
        random.Random(seed).shuffle(items)
        items = items[:n]
    return items


def load_family_witnesses(
    store: Store, *, version: str = "v1", n: int | None = None, seed: int = 0
) -> list[NeutralProof]:
    """Witness proofs from the `families` suite (WP3, long proofs), when it has been built into
    this `Store` -- `INTERP_SPEC.md`: "family witnesses from the `families` suite when present".
    Returns `[]` (callers log this as a "skipped: no families suite" note, per the spec's own
    "skip with a note otherwise") if the suite has not been built yet.

    **Throughput fix (2026-10-01):** the real `families` suite built into this `Store` has 2,193
    witnesses across 92 distinct shape/size combinations (`and_tower`, `and_extract`, `imp_nest`,
    the ILTP-style cases, ...), ordered by shape then by size in the suite file -- so, unlike
    `load_sealed_l12_14` (which already shuffles), capping at `n` here used to take a plain
    `items[:n]` head slice, which for any `n` well under 2,193 silently captured only whichever
    shapes happen to sort first (confirmed: the first 300 are *entirely* `and_tower`, nothing else)
    rather than a cross-section of shapes/sizes. Shuffled first now, deterministically given
    `seed`, matching `load_sealed_l12_14`'s own convention."""
    if not (store.suites_dir / "families" / version).is_dir():
        return []
    items = [
        NeutralProof(pid=p.pid, origin="family", sequent=p.sequent, body=p.ref.body, L=p.length.L)
        for p in store.load_suite("families", version)
        if p.ref is not None and p.length.L is not None
    ]
    if n is not None and len(items) > n:
        random.Random(seed).shuffle(items)
        items = items[:n]
    return items


@dataclass(frozen=True)
class VerifiedSample:
    pid: str
    sequent: Sequent
    body: str


def load_verified_samples(
    store: Store,
    *,
    suite: str,
    version: str,
    model: str,
    run_id: str | None = None,
    limit: int | None = None,
) -> list[VerifiedSample]:
    """Verified (`ok=True`) attempts, one per distinct `pid` (its first success in file order),
    from `run_id` (or, if not given, `model`'s latest matching run) -- the "RL verified
    samples"/"base verified samples" `INTERP_SPEC.md` §1 scores, and
    `likelihood.disagreement_stats`'s same-body input for both models. Returns `[]` if no matching
    run has samples (callers should log a "skip with a note", per the spec, rather than treat this
    as an error)."""
    from ..analysis._common import latest_run_id  # WP6 module; read-only import, not an edit

    if run_id is None:
        run_id = latest_run_id(store, suite=suite, version=version, model=model)
    if run_id is None:
        return []
    path = store.samples_dir / f"{run_id}.jsonl.gz"
    if not path.exists():
        return []
    problems = {p.pid: p for p in store.load_suite(suite, version)}
    seen: set[str] = set()
    out: list[VerifiedSample] = []
    for row in read_jsonl(path):
        if not row.get("ok"):
            continue
        pid = row["pid"]
        if pid in seen or pid not in problems:
            continue
        seen.add(pid)
        out.append(VerifiedSample(pid=pid, sequent=problems[pid].sequent, body=row["body"]))
        if limit is not None and len(out) >= limit:
            break
    return out
