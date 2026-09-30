"""`ExclusionIndex`: the WP0 `Excluder` Protocol, checking a candidate `Sequent` against the
train/eval dedup tiers (plan §2's "Dedup" paragraph, §3's WP2 row).

Matching is by `Sequent.key()` (atom-renaming class key) or `Sequent.pkey()` (also
premise-order-invariant), with `pkey` skipped above `MAX_PREM_FOR_PKEY` (5) premises -- plan §2:
"pkey skipped when >5 premises -- no training theorem has more than 5". This module only *builds
and queries* the index; what a caller does with `status()`'s result (drop vs. flag) is the plan's
dedup policy applied by each suite builder (drop for joined/law/padding/prefix items, flag for
family members and parts), not something `ExclusionIndex` itself enforces.

File names and fields were checked directly against `/scratch/cowifornia/nd-rl/ab2` and `ab3`
(2026-09-30); see `default_tiers`' table below for the exact paths and line counts observed then.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.io import read_jsonl, sha256_str
from ..core.keys import canon_key, prompt_to_thm, split_thm
from ..core.keys import pkey as _pkey_of_thm
from ..core.types import Sequent

#: Read-only source root (plan §A4: never modified by any coder).
ROOT = Path("/scratch/cowifornia/nd-rl")

#: Train tier (plan §2): 154,990 + 5,000 + 15,738 rows observed 2026-09-30.
TRAIN_FILES: tuple[Path, ...] = (
    ROOT / "ab3" / "frozen" / "data" / "train_cap6.jsonl",
    ROOT / "ab3" / "frozen" / "data" / "heldout_cap6.jsonl",
    ROOT / "ab2" / "data" / "targets_L6-11.jsonl",
)

#: Optional arm C/D training tier (plan §2), added only when a builder specifically needs to dedup
#: against the lengthexp composition-training pools too.
TRAIN_FILES_ARM_CD: tuple[Path, ...] = (
    ROOT / "ab3" / "frozen" / "data" / "targets_L6-11.jsonl",
    ROOT / "ab3" / "data_comp" / "targets_L6-11.jsonl",
)

#: Eval tier (flag only, plan §2: "ab2 dev, sealed, frontier and reserve, plus ab3/sealed/*"), one
#: label per file so `status()`'s `eval_sets` names exactly which set(s) matched.
EVAL_FILES: dict[str, Path] = {
    "ab2_dev_L8-11": ROOT / "ab2" / "data" / "dev_L8-11.jsonl",
    "ab2_sealed_L12-14": ROOT / "ab2" / "sealed" / "sealed_L12-14.jsonl",
    "ab2_sealed_L15-16": ROOT / "ab2" / "sealed" / "sealed_L15-16.jsonl",
    "ab2_textbook_dev": ROOT / "ab2" / "sealed" / "textbook_dev.jsonl",
    "ab2_frontier_L12-14": ROOT / "ab2" / "data" / "frontier_L12-14.jsonl",
    "ab2_reserve": ROOT / "ab2" / "data" / "reserve.jsonl",
    "ab3_sealed_matched_L6-14": ROOT / "ab3" / "sealed" / "matched_L6-14.jsonl",
    "ab3_sealed_L15-16": ROOT / "ab3" / "sealed" / "sealed_L15-16.jsonl",
    "ab3_sealed_L17-18": ROOT / "ab3" / "sealed" / "sealed_L17-18.jsonl",
}

#: plan §2: "pkey skipped when >5 premises -- no training theorem has more than 5".
MAX_PREM_FOR_PKEY = 5


def default_tiers(*, arm_cd: bool = False) -> dict[str, list[Path]]:
    """The plan §2 tier layout: `tiers["train"]` plus one entry per `EVAL_FILES` label.
    `arm_cd=True` folds `TRAIN_FILES_ARM_CD` into the training tier too."""
    train = list(TRAIN_FILES) + (list(TRAIN_FILES_ARM_CD) if arm_cd else [])
    tiers: dict[str, list[Path]] = {"train": train}
    tiers.update({label: [path] for label, path in EVAL_FILES.items()})
    return tiers


def _row_key_pkey(row: Mapping[str, Any]) -> tuple[str, str | None]:
    """`(key, pkey)` for one tier-file row: prefer its own precomputed `key`/`pkey` fields (every
    file this module reads has `key`; most also have `pkey`), falling back to computing them from
    `thm` (or `prompt`, for the couple of files -- `train_cap6`/`heldout_cap6` -- that predate the
    `pkey` field) when absent. `pkey` is left `None` above `MAX_PREM_FOR_PKEY` premises."""
    thm = row.get("thm")
    if thm is None:
        thm = prompt_to_thm(row["prompt"])
    key = row.get("key") or canon_key(thm)
    prem, _ = split_thm(thm)
    if len(prem) > MAX_PREM_FOR_PKEY:
        return key, None
    pk = row.get("pkey") or _pkey_of_thm(thm)
    return key, pk


@dataclass(frozen=True)
class _TierIndex:
    keys: frozenset[str]
    pkeys: frozenset[str]


_EMPTY_TIER = _TierIndex(frozenset(), frozenset())


def _fingerprint(tiers: Mapping[str, Sequence[str | Path]]) -> str:
    """A stable fingerprint of every input file's path/size/mtime (not its content -- these files
    can be hundreds of MB, e.g. `ab2/data/reserve.jsonl`, so hashing bytes would defeat the point
    of caching); used only to invalidate the on-disk cache if a tier file changes."""
    sig: list[tuple[str, str, int, int]] = []
    for label in sorted(tiers):
        for p in tiers[label]:
            st = Path(p).stat()
            sig.append((label, str(p), st.st_size, int(st.st_mtime)))
    return sha256_str(json.dumps(sig, sort_keys=True))[:16]


class ExclusionIndex:
    """`Excluder`: pre-scans train/eval tier files once, then answers `status()` from in-memory
    sets. Build with `ExclusionIndex.build(...)`, not the constructor directly."""

    def __init__(self, train: _TierIndex, eval_tiers: Mapping[str, _TierIndex]) -> None:
        self._train = train
        self._eval = dict(eval_tiers)

    def status(self, seq: Sequent) -> tuple[bool, tuple[str, ...]]:
        key = seq.key()
        pk = seq.pkey() if len(seq.premises) <= MAX_PREM_FOR_PKEY else None
        in_train = key in self._train.keys or (pk is not None and pk in self._train.pkeys)
        eval_sets = tuple(
            label
            for label in sorted(self._eval)
            if key in self._eval[label].keys or (pk is not None and pk in self._eval[label].pkeys)
        )
        return in_train, eval_sets

    @classmethod
    def build(
        cls,
        tiers: Mapping[str, Sequence[str | Path]],
        *,
        cache_dir: str | Path | None = None,
    ) -> ExclusionIndex:
        """Scan every file listed in `tiers` (or load a cached scan) and build the index.

        The tier named exactly `"train"` (if present) becomes the drop-eligible training tier;
        every other key is a separate flag-only eval tier named after it. See `default_tiers` for
        the plan's own tier layout. Cached under `<cache_dir>/exclusion_<fingerprint>.json` when
        `cache_dir` is given (plan §A2), fingerprinted by each input file's path/size/mtime so a
        stale cache is never served after a tier file is re-vendored or rebuilt.
        """
        cache_path: Path | None = None
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"exclusion_{_fingerprint(tiers)}.json"
            if cache_path.exists():
                data = json.loads(cache_path.read_text())
                train = _TierIndex(
                    frozenset(data["train"]["keys"]), frozenset(data["train"]["pkeys"])
                )
                ev = {
                    label: _TierIndex(frozenset(d["keys"]), frozenset(d["pkeys"]))
                    for label, d in data["eval"].items()
                }
                return cls(train, ev)

        scanned: dict[str, _TierIndex] = {}
        for label, paths in tiers.items():
            keys: set[str] = set()
            pkeys: set[str] = set()
            for path in paths:
                for row in read_jsonl(path):
                    k, pk = _row_key_pkey(row)
                    keys.add(k)
                    if pk is not None:
                        pkeys.add(pk)
            scanned[label] = _TierIndex(frozenset(keys), frozenset(pkeys))

        train = scanned.pop("train", _EMPTY_TIER)
        index = cls(train, scanned)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "train": {"keys": sorted(train.keys), "pkeys": sorted(train.pkeys)},
                "eval": {
                    label: {"keys": sorted(idx.keys), "pkeys": sorted(idx.pkeys)}
                    for label, idx in scanned.items()
                },
            }
            cache_path.write_text(json.dumps(payload))
        return index


DEFAULT_CACHE_DIR = Path("/scratch/cowifornia/nd-rl/bench/out/cache")


class DefaultExcluder:
    """The plan's standard `Excluder` (`default_tiers()`, cached under `DEFAULT_CACHE_DIR`).

    Constructible with no arguments, so a build config can name it as
    `excluder: code.ndbench.logic.exclusion:DefaultExcluder` (the CLI instantiates classes).
    """

    def __init__(self, *, arm_cd: bool = False, cache_dir: str | Path | None = None) -> None:
        self._index = ExclusionIndex.build(
            default_tiers(arm_cd=arm_cd), cache_dir=cache_dir or DEFAULT_CACHE_DIR
        )

    def status(self, seq: Sequent) -> tuple[bool, tuple[str, ...]]:
        return self._index.status(seq)
