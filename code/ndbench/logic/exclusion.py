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

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..core.io import read_jsonl, sha256_str
from ..core.keys import canon_key, prompt_to_thm, split_thm
from ..core.keys import pkey as _pkey_of_thm
from ..core.types import JSON, Sequent

if TYPE_CHECKING:
    from ..store import Store

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


# ============================================================================================
# Dan's state-env models: "seen in training" tiers (DANSTATE_PLAN.md §A2, §3's `dan_tiers` row,
# §7 WP-D3).
# ============================================================================================

#: Dan's downloaded Stage-1 training sets and held-out set (plan §A2; md5-checked against his own
#: summary on 2026-10-01).
DAN_DATA_ROOT = Path("/scratch/cowifornia/nd-rl/bench/dan_ckpts/data")

#: Dan's fork clone (read-only in practice: plan §A4 -- never written to by this package).
DAN_CLONE_ROOT = Path("/scratch/cowifornia/nd-rl/bench/nd-takehome-dan")

#: `family` -> that cap's Stage-1 training set. Confirmed against every invocation that actually
#: produced the 12 checkpoints: `pod/bs/stage1.sh` (`$D=data/p2/train_depth3_f0_a1.jsonl` for cap
#: 6, `data/kh/train_k12.jsonl` for cap 12 -- both mirrored verbatim at `DAN_DATA_ROOT`) and every
#: `la_T1_best{6,12}_s{0,1,2}/args.json`'s own `"train"` field (`data/p2/train_depth3_f0_a1.jsonl`
#: / `data/kh/train_k12.jsonl` respectively -- the ladder's per-round fine-tune also replays
#: `--retain` (20,000) random rows of this same file every round, so including the whole file
#: already covers that).
DAN_STAGE1_FILES: dict[str, Path] = {
    "dan6": DAN_DATA_ROOT / "train_depth3_f0_a1.jsonl",
    "dan12": DAN_DATA_ROOT / "train_k12.jsonl",
}

#: The T1 ladder's RL-target pool (`state_ladder_ei.py --targets`, default `data/ladder/
#: rl_targets.jsonl`, 4,495 theorems). Every `la_T1_best{6,12}_s{0,1,2}` checkpoint's
#: `artifacts/bs/la_T1_best*/args.json` uses this exact path unmodified (shared verbatim by both
#: cap families -- confirmed directly from `la_T1_best6_s0/args.json` and
#: `la_T1_best12_s0/args.json`, which differ only in `init`/`train`). Each round's fine-tune mix
#: (`state_ladder_ei.py`'s "4. train" step) writes up to `max_per_thm` (4) of the model's own
#: accepted proofs per target theorem into the training data, `rl_weight` (4) times each -- so
#: every target *theorem* (its `key`/`pkey` class, not merely the pool's own unused `gen_proof`
#: field) is seen in training once any proof of it is accepted in any round.
DAN_LADDER_RL_TARGETS = DAN_CLONE_ROOT / "data" / "ladder" / "rl_targets.jsonl"

#: Sampled every round for monitoring only -- `state_ladder_ei.py`'s own module docstring: "sample
#: every transfer theorem k times (**never trained on**), record"; read inside `main()` only to
#: build `eval_keys` (an eval-tier guard keeping T3's relabelling byproducts off of it), never
#: written into a fine-tune mix. Deliberately **not** folded into `dan_tiers`'s train tier; kept
#: as a named constant only so a sidecar's metadata can record that it was checked.
DAN_LADDER_TRANSFER = DAN_CLONE_ROOT / "data" / "ladder" / "transfer.jsonl"

#: Dan's held-out set: evaluated only (`pod/bs/stage1.sh`'s `state_eval.py --in
#: data/p2/heldout.jsonl`; `state_ladder_ei.py --heldout`'s default is this same file), never
#: trained on. Not folded into `dan_tiers` either, for the same reason as `DAN_LADDER_TRANSFER`.
DAN_HELDOUT = DAN_DATA_ROOT / "heldout.jsonl"

#: Known `dan_tiers`/sidecar family labels.
DAN_FAMILIES: tuple[str, ...] = tuple(DAN_STAGE1_FILES)


def dan_tiers(family: str) -> dict[str, list[Path]]:
    """Dan's training tier for `family` (`"dan6"` | `"dan12"`): that cap's Stage-1 set union the
    T1 ladder's RL-target pool (this module's own constants' docstrings explain why
    `DAN_LADDER_TRANSFER`/`DAN_HELDOUT` are deliberately excluded -- both are sampled for
    monitoring/eval only, confirmed never written into a fine-tune mix). Shaped exactly like
    `default_tiers()`'s return (a `{"train": [...]}` mapping `ExclusionIndex.build` consumes
    directly), so `seen_train` means the same thing -- membership by `Sequent.key()`/`pkey()` --
    for Dan's models as it does for every other model family.
    """
    if family not in DAN_STAGE1_FILES:
        raise ValueError(f"unknown dan family {family!r}; known: {sorted(DAN_STAGE1_FILES)}")
    return {"train": [DAN_STAGE1_FILES[family], DAN_LADDER_RL_TARGETS]}


class DanExcluder:
    """`Excluder` over one Dan family's training tier (`dan_tiers`). `status()[1]` (`eval_sets`)
    is always `()`: this module tracks only the "seen in training" tier for Dan's models, not a
    flagged eval tier (plan's "'seen' means the same thing for every model" -- a plain boolean,
    matching `Problem.seen_train`'s own shape)."""

    def __init__(self, family: str, *, cache_dir: str | Path | None = None) -> None:
        self.family = family
        self._index = ExclusionIndex.build(
            dan_tiers(family), cache_dir=cache_dir or DEFAULT_CACHE_DIR
        )

    def status(self, seq: Sequent) -> tuple[bool, tuple[str, ...]]:
        in_train, _ = self._index.status(seq)
        return in_train, ()


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dan_sources_metadata(family: str) -> dict[str, JSON]:
    """`{"family": family, "sources": [{"path", "md5", "n_rows"}, ...]}` for `dan_tiers(family)`'s
    train-tier files -- computed once per family and reused across every suite by
    `build_seen_sidecars` (these are multi-hundred-MB files; re-hashing/re-counting them once per
    suite would be wasteful)."""
    sources: list[JSON] = []
    for path in dan_tiers(family)["train"]:
        exists = path.is_file()
        sources.append(
            {
                "path": str(path),
                "md5": _md5_file(path) if exists else None,
                "n_rows": sum(1 for _ in read_jsonl(path)) if exists else None,
            }
        )
    return {"family": family, "sources": cast(JSON, sources)}


def seen_sidecar_path(store_root: str | Path, suite: str, version: str, family: str) -> Path:
    """`<store_root>/suites/<suite>/<version>/seen_<family>.json`'s path."""
    return Path(store_root) / "suites" / suite / version / f"seen_{family}.json"


def build_seen_sidecar(
    store: Store,
    suite: str,
    version: str,
    family: str,
    *,
    excluder: DanExcluder | None = None,
    source_meta: Mapping[str, JSON] | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, JSON]:
    """Build and write `seen_sidecar_path(store.root, suite, version, family)`: `{"suite",
    "version", "family", "sources", "n_items", "n_seen", "seen": {pid: bool}}` for every item of
    the already-built `(suite, version)`, keyed by `Sequent.key()`/`pkey()` the same way
    `ExclusionIndex` keys every other tier.

    `excluder`/`source_meta` let a caller building sidecars for many suites (`build_seen_sidecars`)
    pass in an already-built `DanExcluder` and `_dan_sources_metadata` result instead of
    recomputing them (the metadata hashes/counts multi-hundred-MB files) for every suite.
    """
    resolved_excluder = excluder or DanExcluder(family, cache_dir=cache_dir)
    meta = dict(source_meta) if source_meta is not None else _dan_sources_metadata(family)
    problems = store.load_suite(suite, version)
    seen: dict[str, bool] = {p.pid: resolved_excluder.status(p.sequent)[0] for p in problems}
    payload: dict[str, JSON] = {
        "suite": suite,
        "version": version,
        **meta,
        "n_items": len(problems),
        "n_seen": int(sum(seen.values())),
        "seen": cast(JSON, seen),
    }
    path = seen_sidecar_path(store.root, suite, version, family)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))
    return payload


def build_seen_sidecars(
    store: Store,
    suite_versions: Sequence[tuple[str, str]],
    families: Sequence[str] = DAN_FAMILIES,
    *,
    cache_dir: str | Path | None = None,
) -> dict[str, dict[str, JSON]]:
    """Build `seen_<family>.json` sidecars for every `(suite, version)` in `suite_versions` x every
    `family` in `families`, computing each family's `DanExcluder`/source metadata once and reusing
    it across every suite. Returns `{family: {"<suite>/<version>": {"n_items", "n_seen"}}}`."""
    report: dict[str, dict[str, JSON]] = {}
    for family in families:
        excluder = DanExcluder(family, cache_dir=cache_dir)
        meta = _dan_sources_metadata(family)
        per_suite: dict[str, JSON] = {}
        for suite, version in suite_versions:
            payload = build_seen_sidecar(
                store, suite, version, family, excluder=excluder, source_meta=meta
            )
            per_suite[f"{suite}/{version}"] = cast(
                JSON, {"n_items": payload["n_items"], "n_seen": payload["n_seen"]}
            )
        report[family] = per_suite
    return report


def seen_for(store: Store, suite: str, version: str, family: str) -> dict[str, bool]:
    """Load a previously-built `seen_<family>.json` sidecar (`build_seen_sidecar`/
    `build_seen_sidecars`) for `(suite, version)`, returning just its `{pid: bool}` mapping."""
    path = seen_sidecar_path(store.root, suite, version, family)
    data = json.loads(path.read_text())
    return cast("dict[str, bool]", data["seen"])
