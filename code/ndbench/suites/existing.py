"""WP2 suite builders over already-existing (legacy) datasets: `textbook_dev`, `textbook_law`,
`gen_ctrl`, `skill_pool`, `parts_pool`, plus `import_legacy` (plan §2, §3's WP2 row).

Every builder matches the plan's builder contract exactly:
`build(cfg, *, certifier, excluder, rng_seed) -> tuple[list[Problem], dict]`. `cfg` may override
each builder's default source path(s) (and, for testing, cap how many rows are scanned) -- see each
function's own docstring for its recognised keys; every builder falls back to the real
`/scratch/cowifornia/nd-rl/ab2`/`ab3` paths checked against the live filesystem on 2026-09-30 (see
`logic.exclusion.EVAL_FILES` for the shared eval-tier file table these largely reuse).

Scope note on skill tagging (see `logic/skills.py`'s module docstring for the cost measurements
behind this): `classical` and `needs_ore` are cheap and exact, so every builder that touches
already-labelled theorems tags both unconditionally. `necessity`/`needs_depth` are NOT run
eagerly by any builder here -- a single bounded search often does not resolve within a couple of
seconds at the L>=12 lengths most of these pools contain, so running them over a multi-thousand
item pool would far exceed the plan's per-job CPU budget (plan §A2: <=8 workers, ~15 min/job).
Both functions are implemented and unit-tested in isolation; `skill_pool`'s returned stats say so
explicitly, and a downstream analysis pass can apply them to a deliberately-sized subset.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..core.io import read_jsonl
from ..core.proof import make_reference
from ..core.types import (
    JSON,
    Certifier,
    Excluder,
    ItemResult,
    LengthLabel,
    Problem,
    Sequent,
)
from ..logic.exclusion import ROOT
from ..logic.skills import classical, needs_ore, rule_uses

if TYPE_CHECKING:
    from ..store import Store

__all__ = [
    "textbook_dev",
    "textbook_law",
    "gen_ctrl",
    "skill_pool",
    "parts_pool",
    "import_legacy",
]

BuildResult = tuple[list[Problem], dict[str, JSON]]

# ---------------------------------------------------------------------------------- shared


def _seq_from_row(row: Mapping[str, Any]) -> Sequent:
    thm = row.get("thm")
    if thm is not None:
        return Sequent.from_thm(thm)
    return Sequent.from_prompt(row["prompt"])


def _status(excluder: Excluder | None, seq: Sequent) -> tuple[bool, tuple[str, ...]]:
    if excluder is None:
        return False, ()
    return excluder.status(seq)


def _atoms_in_thm(thm: str) -> set[str]:
    return {t for t in thm.split() if t in ("P", "Q", "R", "S")}


# ------------------------------------------------------------------------------ textbook_dev

DEFAULT_TEXTBOOK_DEV_PATH = ROOT / "ab2" / "sealed" / "textbook_dev.jsonl"


def textbook_dev(
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None,
    excluder: Excluder | None,
    rng_seed: int,
) -> BuildResult:
    """The 58 items of `ab2/sealed/textbook_dev.jsonl`, relabelled by `certifier` (plan §2 B8:
    "bound 16, 120 s"); the legacy reference length goes into `tags["reference_L"]` (plan §0: these
    are reference lengths, not minimal -- e.g. `~(P>Q) |- P&~Q` is labelled 18 there, minlen finds
    12).

    `cfg` keys: `path` (default `ab2/sealed/textbook_dev.jsonl`), `bound` (16), `time_s` (120.0),
    `version` ("v1"). Requires a real `certifier` (this builder's whole purpose is relabelling).
    """
    del rng_seed  # this builder is exhaustive over a fixed 58-item file; nothing to sample
    if certifier is None:
        raise ValueError("textbook_dev requires a certifier to relabel its 58 items")

    path = Path(cfg.get("path", DEFAULT_TEXTBOOK_DEV_PATH))
    bound = int(cfg.get("bound", 16))
    time_s = float(cfg.get("time_s", 120.0))
    version = str(cfg.get("version", "v1"))

    rows = list(read_jsonl(path))
    seqs = [_seq_from_row(r) for r in rows]
    labels = certifier.certify(seqs, [None] * len(seqs), bound=bound, time_s=time_s)

    problems: list[Problem] = []
    by_method: Counter[str] = Counter()
    n_seen_train = 0
    for row, seq, label in zip(rows, seqs, labels, strict=True):
        in_train, eval_sets = _status(excluder, seq)
        n_seen_train += int(in_train)
        by_method[label.method] += 1
        problems.append(
            Problem(
                pid=str(row["name"]),
                suite="textbook_dev",
                version=version,
                sequent=seq,
                length=label,
                ref=None,
                tags={
                    "reference_L": row.get("L_true"),
                    "reference_n_lines": row.get("n_lines"),
                    "source": row.get("source"),
                },
                seen_train=in_train,
                seen_eval=eval_sets,
            )
        )

    stats: dict[str, JSON] = {
        "n_items": len(problems),
        "by_method": dict(by_method),
        "n_seen_train": n_seen_train,
        "bound": bound,
        "time_s": time_s,
        "source_path": str(path),
    }
    return problems, stats


# ------------------------------------------------------------------------------ textbook_law

DEFAULT_DEV_L8_11_PATH = ROOT / "ab2" / "data" / "dev_L8-11.jsonl"
DEFAULT_SEALED_L12_14_PATH = ROOT / "ab2" / "sealed" / "sealed_L12-14.jsonl"

#: plan §2 B8: at most this many items per (L, schema) cell.
TEXTBOOK_LAW_PER_SCHEMA_CAP = 6
#: plan §2 B8: "about 40 per L" (a cap, not a floor -- some L values have too few distinct schemas
#: to reach it even before this cap applies, e.g. L14 has only one schema in `sealed_L12-14`).
TEXTBOOK_LAW_PER_L_TARGET = 40


def _method_and_bound_for_row(row: Mapping[str, Any]) -> tuple[str, int | None]:
    """`(method, bound)` for a legacy row we do not relabel ourselves: `"minlen_exact"` when the
    row carries a `minlen_proof` (so its `L_true` is minlen-certified evidence, not just a
    generation-time estimate), else `"reference"` (plan §2's fallback for a legacy length we keep
    without independently re-deriving it -- see `logic/certify.py`'s module docstring)."""
    if row.get("minlen_proof"):
        return "minlen_exact", row.get("minlen_bound")
    return "reference", None


def _stratified_textbook_law(
    rows: Sequence[Mapping[str, Any]], rng_seed: int
) -> tuple[list[Mapping[str, Any]], dict[str, JSON]]:
    rng = random.Random(rng_seed)
    by_l: dict[int, dict[Any, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_l[row["L_true"]][row.get("schema")].append(row)

    selected: list[Mapping[str, Any]] = []
    stats_by_l: dict[str, JSON] = {}
    for L in sorted(by_l):
        per_schema = by_l[L]
        capped: list[Mapping[str, Any]] = []
        for schema in sorted(per_schema, key=lambda s: (s is None, str(s))):
            cand = list(per_schema[schema])
            rng.shuffle(cand)
            capped.extend(cand[:TEXTBOOK_LAW_PER_SCHEMA_CAP])
        if len(capped) > TEXTBOOK_LAW_PER_L_TARGET:
            rng.shuffle(capped)
            capped = capped[:TEXTBOOK_LAW_PER_L_TARGET]
        selected.extend(capped)
        stats_by_l[str(L)] = {"n_schemas": len(per_schema), "n_selected": len(capped)}
    return selected, stats_by_l


def textbook_law(
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None,
    excluder: Excluder | None,
    rng_seed: int,
) -> BuildResult:
    """The plan §2 B8 stratified subsample of `source == "textbook"` items from
    `ab2/data/dev_L8-11.jsonl` and `ab2/sealed/sealed_L12-14.jsonl`: at most 6 per (L, schema),
    about 40 per L. Keeps each item's existing `L_true` label (no relabelling -- `certifier` is
    accepted for interface consistency but unused).

    `cfg` keys: `dev_path`, `sealed_path` (defaults as named above), `version` ("v1").
    """
    del certifier
    dev_path = Path(cfg.get("dev_path", DEFAULT_DEV_L8_11_PATH))
    sealed_path = Path(cfg.get("sealed_path", DEFAULT_SEALED_L12_14_PATH))
    version = str(cfg.get("version", "v1"))

    rows = [r for r in read_jsonl(dev_path) if r.get("source") == "textbook"]
    rows += [r for r in read_jsonl(sealed_path) if r.get("source") == "textbook"]

    selected, stats_by_l = _stratified_textbook_law(rows, rng_seed)

    problems: list[Problem] = []
    for row in selected:
        seq = _seq_from_row(row)
        in_train, eval_sets = _status(excluder, seq)
        method, bound = _method_and_bound_for_row(row)
        proof = row.get("minlen_proof")
        problems.append(
            Problem(
                pid=str(row["name"]),
                suite="textbook_law",
                version=version,
                sequent=seq,
                length=LengthLabel(
                    L=row["L_true"],
                    L_ub=row["L_true"],
                    L_lb=None,
                    method=method,
                    bound=bound,
                    timeout=False,
                ),
                ref=make_reference(proof, source="minlen") if proof else None,
                tags={
                    "L": row["L_true"],
                    "schema": row.get("schema"),
                    "n_prem": row.get("n_prem"),
                },
                seen_train=in_train,
                seen_eval=eval_sets,
            )
        )

    stats: dict[str, JSON] = {
        "n_items": len(problems),
        "n_candidates": len(rows),
        "by_l": stats_by_l,
        "per_schema_cap": TEXTBOOK_LAW_PER_SCHEMA_CAP,
        "per_l_target": TEXTBOOK_LAW_PER_L_TARGET,
    }
    return problems, stats


# ---------------------------------------------------------------------------------- gen_ctrl

GEN_CTRL_LS = (8, 9, 10, 11)
GEN_CTRL_N_ITEMS = 60


def gen_ctrl(
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None,
    excluder: Excluder | None,
    rng_seed: int,
) -> BuildResult:
    """60 `source == "gen"` items from `ab2/data/dev_L8-11.jsonl`, evenly split across L8-11 (plan
    §2 B8's "control: 60 dev gen items L8-11"). No relabelling (`certifier` unused; `L_true` kept
    as-is, `method="reference"` -- `dev_L8-11.jsonl` carries no `minlen_proof` to certify against).

    `cfg` keys: `path` (default `ab2/data/dev_L8-11.jsonl`), `n_items` (60), `version` ("v1").
    """
    del certifier
    path = Path(cfg.get("path", DEFAULT_DEV_L8_11_PATH))
    n_items = int(cfg.get("n_items", GEN_CTRL_N_ITEMS))
    version = str(cfg.get("version", "v1"))

    by_l: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        if row.get("source") == "gen" and row.get("L_true") in GEN_CTRL_LS:
            by_l[row["L_true"]].append(row)

    rng = random.Random(rng_seed)
    per_l_target, remainder = divmod(n_items, len(GEN_CTRL_LS))
    selected: list[Mapping[str, Any]] = []
    stats_by_l: dict[str, JSON] = {}
    for i, L in enumerate(GEN_CTRL_LS):
        cand = list(by_l.get(L, []))
        rng.shuffle(cand)
        target = per_l_target + (1 if i < remainder else 0)
        take = cand[:target]
        selected.extend(take)
        stats_by_l[str(L)] = {"available": len(cand), "taken": len(take)}

    problems: list[Problem] = []
    for row in selected:
        seq = _seq_from_row(row)
        in_train, eval_sets = _status(excluder, seq)
        problems.append(
            Problem(
                pid=str(row["name"]),
                suite="gen_ctrl",
                version=version,
                sequent=seq,
                length=LengthLabel(
                    L=row["L_true"],
                    L_ub=row["L_true"],
                    L_lb=None,
                    method="reference",
                    bound=None,
                    timeout=False,
                ),
                ref=None,
                tags={"L": row["L_true"], "n_prem": row.get("n_prem")},
                seen_train=in_train,
                seen_eval=eval_sets,
            )
        )

    stats: dict[str, JSON] = {
        "n_items": len(problems),
        "n_target": n_items,
        "by_l": stats_by_l,
    }
    return problems, stats


# --------------------------------------------------------------------------------- parts_pool

DEFAULT_RESERVE_PATH = ROOT / "ab2" / "data" / "reserve.jsonl"
PARTS_POOL_N_ITEMS = 1200
PARTS_POOL_L_RANGE = range(2, 9)  # 2..8 inclusive
PARTS_POOL_ATOM_RANGE = range(1, 5)  # 1..4 inclusive


def parts_pool(
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None,
    excluder: Excluder | None,
    rng_seed: int,
) -> BuildResult:
    """1,200 `source == "gen"` items from `ab2/data/reserve.jsonl` carrying a `minlen_proof`,
    stratified by `L_true` in 2-8 crossed with distinct-atom-count in 1-4 (28 cells), one item per
    `pkey` (plan §2's parts pool for B1/B5). No relabelling (`certifier` unused; each item keeps
    its existing minlen-certified `L_true`).

    `cfg` keys: `path` (default `ab2/data/reserve.jsonl`), `n_items` (1200), `scan_limit` (`None`
    -- caps how many source rows are scanned, for fast tests against a huge file), `version`
    ("v1").
    """
    del certifier
    path = Path(cfg.get("path", DEFAULT_RESERVE_PATH))
    n_items = int(cfg.get("n_items", PARTS_POOL_N_ITEMS))
    scan_limit = cfg.get("scan_limit")
    version = str(cfg.get("version", "v1"))

    seen_pkeys: set[str] = set()
    by_cell: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    n_scanned = 0
    n_candidates = 0
    for row in read_jsonl(path):
        if scan_limit is not None and n_scanned >= int(scan_limit):
            break
        n_scanned += 1
        if row.get("source") != "gen" or not row.get("minlen_proof"):
            continue
        l_true = row.get("L_true")
        if not isinstance(l_true, int) or l_true not in PARTS_POOL_L_RANGE:
            continue
        pk = row.get("pkey")
        if not pk or pk in seen_pkeys:
            continue
        n_atoms = len(_atoms_in_thm(row.get("thm", "")))
        if n_atoms not in PARTS_POOL_ATOM_RANGE:
            continue
        seen_pkeys.add(pk)
        n_candidates += 1
        by_cell[(l_true, n_atoms)].append(row)

    cells = sorted(by_cell)
    base_per_cell, remainder = divmod(n_items, max(len(cells), 1))
    rng = random.Random(rng_seed)
    selected: list[tuple[Mapping[str, Any], int]] = []
    stats_by_cell: dict[str, JSON] = {}
    for i, cell in enumerate(cells):
        cell_l, cell_atoms = cell
        cand = list(by_cell[cell])
        rng.shuffle(cand)
        target = base_per_cell + (1 if i < remainder else 0)
        take = cand[:target]
        selected.extend((row, cell_atoms) for row in take)
        stats_by_cell[f"L{cell_l}_atoms{cell_atoms}"] = {
            "available": len(cand),
            "taken": len(take),
        }

    problems: list[Problem] = []
    for row, n_atoms in selected:
        seq = _seq_from_row(row)
        in_train, eval_sets = _status(excluder, seq)
        proof = row["minlen_proof"]
        problems.append(
            Problem(
                pid=str(row["name"]),
                suite="parts_pool",
                version=version,
                sequent=seq,
                length=LengthLabel(
                    L=row["L_true"],
                    L_ub=row["L_true"],
                    L_lb=None,
                    method="minlen_exact",
                    bound=row.get("minlen_bound"),
                    timeout=False,
                ),
                ref=make_reference(proof, source="minlen"),
                tags={
                    "L": row["L_true"],
                    "n_atoms": n_atoms,
                    "schema": row.get("schema"),
                    "n_prem": row.get("n_prem"),
                    "pkey": row.get("pkey"),
                },
                seen_train=in_train,
                seen_eval=eval_sets,
            )
        )

    stats: dict[str, JSON] = {
        "n_items": len(problems),
        "n_target": n_items,
        "n_scanned": n_scanned,
        "n_candidates": n_candidates,
        "n_cells": len(cells),
        "by_cell": stats_by_cell,
    }
    return problems, stats


# --------------------------------------------------------------------------------- skill_pool

SKILL_POOL_SOURCES: dict[str, Path] = {
    "dev_L8-11": DEFAULT_DEV_L8_11_PATH,
    "sealed_L12-14": DEFAULT_SEALED_L12_14_PATH,
    "frontier_L12-14": ROOT / "ab2" / "data" / "frontier_L12-14.jsonl",
    "lengthexp_std": ROOT / "ab3" / "lengthexp" / "testsets" / "std.jsonl",
    "lengthexp_matched": ROOT / "ab3" / "lengthexp" / "testsets" / "matched.jsonl",
}


def skill_pool(
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None,
    excluder: Excluder | None,
    rng_seed: int,
) -> BuildResult:
    """dev L8-11, sealed L12-14, frontier L12-14 and the lengthexp std/matched (L6-18) test sets,
    every item tagged with `classical` and `case_split` (`needs_ore`) -- both cheap and exact, so
    applied unconditionally -- plus `rule_uses` wherever a `minlen_proof` body is available (plan
    §2 B3's skill profile). No relabelling (`certifier` unused; existing `L_true` kept).

    Does NOT precompute `necessity`/`needs_depth` over the whole pool: see this module's and
    `logic/skills.py`'s docstrings for the cost measurements that rule that out at this scale; the
    returned stats note this explicitly.

    `cfg` keys: `sources` (default `SKILL_POOL_SOURCES`, a `{label: path}` mapping -- override for
    tests), `limit_per_source` (`None` -- caps rows read per source file, for fast tests),
    `version` ("v1"). `rng_seed` is accepted for interface consistency; this builder does not
    subsample (every row from every source is included).
    """
    del certifier, rng_seed
    sources: Mapping[str, str | Path] = cfg.get("sources", SKILL_POOL_SOURCES)
    limit_per_source = cfg.get("limit_per_source")
    version = str(cfg.get("version", "v1"))

    problems: list[Problem] = []
    stats_by_source: dict[str, JSON] = {}
    for label, path in sources.items():
        n_classical = 0
        n_case_split = 0
        n_rows = 0
        for row in read_jsonl(path):
            if limit_per_source is not None and n_rows >= int(limit_per_source):
                break
            n_rows += 1
            seq = _seq_from_row(row)
            in_train, eval_sets = _status(excluder, seq)
            is_classical = classical(seq)
            is_case_split = needs_ore(seq)
            n_classical += int(is_classical)
            n_case_split += int(is_case_split)

            proof = row.get("minlen_proof")
            tags: dict[str, JSON] = {
                "source_set": label,
                "L": row.get("L_true"),
                "schema": row.get("schema"),
                "n_prem": row.get("n_prem"),
                "classical": is_classical,
                "case_split": is_case_split,
            }
            if proof:
                tags["rule_uses"] = cast(dict[str, JSON], rule_uses(proof))

            problems.append(
                Problem(
                    pid=f"{label}:{row.get('name')}",
                    suite="skill_pool",
                    version=version,
                    sequent=seq,
                    length=LengthLabel(
                        L=row.get("L_true"),
                        L_ub=row.get("L_true"),
                        L_lb=None,
                        method="minlen_exact" if proof else "reference",
                        bound=row.get("minlen_bound"),
                        timeout=False,
                    ),
                    ref=make_reference(proof, source="minlen") if proof else None,
                    tags=tags,
                    seen_train=in_train,
                    seen_eval=eval_sets,
                )
            )
        stats_by_source[label] = {
            "n_items": n_rows,
            "n_classical": n_classical,
            "n_case_split": n_case_split,
        }

    stats: dict[str, JSON] = {
        "n_items": len(problems),
        "by_source": stats_by_source,
        "necessity_note": (
            "classical/needs_ore are tagged on every item (cheap, exact). necessity()/"
            "needs_depth() are NOT precomputed over this pool: a single bounded search often does "
            "not resolve within a couple of seconds at L>=12 (see logic/skills.py's module "
            "docstring for measurements), so running them here would far exceed the plan's "
            "per-job CPU budget (plan SS A2). Both are implemented and unit-tested; apply them to "
            "a deliberately-sized subset in a downstream analysis pass."
        ),
    }
    return problems, stats


# ------------------------------------------------------------------------------- import_legacy


@dataclass(frozen=True)
class _LegacyRunSpec:
    """A minimal stand-in for `runner.runner.RunSpec`, structurally matching what `Store.new_run`/
    `config_key` need (`store._RunSpecLike`) without importing `runner` (which `store.py` itself
    never does either, per its own module docstring) or fabricating a fake `AdaptiveRule`/pids for
    data that was never run through `runner.run` in the first place."""

    suite: str
    version: str
    model: str
    k: int
    temperature: float = 0.8
    max_new_tokens: int | None = None
    seed: int = 0
    adaptive: None = None
    pids: tuple[str, ...] | None = None
    options: Mapping[str, JSON] = field(default_factory=dict)


#: plan §0: checkpoint identities already established (and independently re-verified against the
#: live files on 2026-09-30 with `sha256sum`) -- reused here rather than re-hashing each
#: multi-hundred-MB checkpoint at import time.
_LEGACY_MONITOR_MODELS: dict[str, tuple[Path, str]] = {
    "base": (ROOT / "ab2" / "monitor" / "final" / "pretrain", "f4409e4f82d8922f"),
    "champion": (
        ROOT / "ab2" / "monitor" / "final" / "champion2_n060_fullpool",
        "548a9e343c71675f",
    ),
}

#: plan §2's "the lengthexp eval/* per-theorem rows for pretrain and arm C" -- arm C has three
#: seeds (`C_s101`/`C_s102`/`C_s103`) under `ab3/lengthexp/eval/`.
_LEGACY_LENGTHEXP_ARMS: dict[str, tuple[str, ...]] = {
    "pretrain": ("pretrain",),
    "armC": ("C_s101", "C_s102", "C_s103"),
}
_LENGTHEXP_EVAL_DIR = ROOT / "ab3" / "lengthexp" / "eval"


def _import_monitor_run(
    store: Store, model_label: str, ckpt_dir: Path, ckpt_sha_prefix: str, per_theorem: Path
) -> str:
    set_label = per_theorem.name[: -len("_per_theorem.jsonl")]
    summary_path = ckpt_dir / f"{set_label}.json"
    cfg: dict[str, Any] = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    k = int(cfg.get("k", 0))
    seed = int(cfg.get("seed", 0))
    spec = _LegacyRunSpec(
        suite=f"legacy_{cfg.get('set', set_label)}",
        version=f"k{k}_s{seed}",
        model=model_label,
        k=k,
        temperature=float(cfg.get("temperature", 0.8)),
        max_new_tokens=cfg.get("max_new"),
        seed=seed,
    )
    identity: dict[str, JSON] = {
        "ckpt": str(ckpt_dir / "final.pt"),
        "ckpt_sha256_prefix": ckpt_sha_prefix,
        "label": model_label,
    }
    run_id = store.new_run(spec, identity, judge_id="legacy_import")

    rows: list[ItemResult] = []
    for row in read_jsonl(per_theorem):
        rows.append(
            ItemResult(
                run_id=run_id,
                pid=str(row["name"]),
                chunk=-1,
                n=k,
                c=int(row.get("n_ok", 0)),
                first_success=None,
                ok_written=[],
                ok_pruned=[],
                ok_tokens=[],
                ok_starts=[],
                fail_kinds={},
                fail_lines=[],
                long_enough=0,
                tokens_mean=0.0,
                hit_max=0,
            )
        )
    store.append_items(run_id, rows)
    store.finish_run(run_id, "legacy", 0.0)
    return run_id


def _import_lengthexp_run(store: Store, arm_label: str, eval_dir: Path, data_path: Path) -> str:
    meta_path = eval_dir / "meta.json"
    meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    set_label = data_path.stem  # e.g. "std_L6", "matched_L10"
    summary_path = eval_dir / f"{set_label}.summary.json"
    summary: dict[str, Any] = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    run_config: dict[str, Any] = meta.get("config", {})

    k = int(run_config.get("k", summary.get("k", 0)))
    spec = _LegacyRunSpec(
        suite=f"legacy_lengthexp_{set_label}",
        version=eval_dir.name,  # e.g. "C_s101"
        model=arm_label,
        k=k,
        temperature=float(run_config.get("temperature", 0.8)),
        max_new_tokens=run_config.get("max_new"),
        seed=int(run_config.get("sample_seed", 0)),
    )
    identity: dict[str, JSON] = {
        "ckpt": meta.get("ckpt"),
        "ckpt_sha256": meta.get("ckpt_sha256"),
        "label": eval_dir.name,
    }
    run_id = store.new_run(spec, identity, judge_id="legacy_import")

    rows: list[ItemResult] = []
    for row in read_jsonl(data_path):
        fail_kinds = Counter(f[0] for f in row.get("fails", []) if f)
        rows.append(
            ItemResult(
                run_id=run_id,
                pid=str(row["name"]),
                chunk=-1,
                n=int(row.get("k", k)),
                c=int(row.get("n_ok", 0)),
                first_success=None,
                ok_written=list(row.get("ok_lines", [])),
                ok_pruned=[],
                ok_tokens=[],
                ok_starts=list(row.get("ok_starts", [])),
                fail_kinds=dict(fail_kinds),
                fail_lines=[],
                long_enough=0,
                tokens_mean=float(row.get("n_tok_mean", 0.0)),
                hit_max=0,
            )
        )
    store.append_items(run_id, rows)
    store.finish_run(run_id, "legacy", 0.0)
    return run_id


def import_legacy(store: Store) -> dict[str, JSON]:
    """Import pre-ndbench run results as `status="legacy"` runs (plan §2/§3's WP2 row):

    * `ab2/monitor/final/{pretrain,champion2_n060_fullpool}/*_per_theorem.jsonl` as `base`/
      `champion` (the `final.pt` shas plan §0 already establishes: `f4409e4f82d8922f...`,
      `548a9e343c71675f...`), one run per `(model, set, k, seed)` combination found on disk (e.g.
      `sealed_L12-14` was run at both `k64` and `k256`) -- `max_new` comes from each set's own
      companion summary `.json` (512 for most sets, 640 for `sealed_L12-14`/`sealed_L15-16`).
    * `ab3/lengthexp/eval/{pretrain,C_s101,C_s102,C_s103}/*.jsonl` (excluding the `.summary.json`/
      `meta.json` companions, via the `.jsonl` glob) as `pretrain`/`armC`, one run per
      `(arm+seed, per-L file)`.
    """
    run_ids: list[str] = []
    for model_label, (ckpt_dir, sha_prefix) in _LEGACY_MONITOR_MODELS.items():
        for per_theorem in sorted(ckpt_dir.glob("*_per_theorem.jsonl")):
            run_ids.append(
                _import_monitor_run(store, model_label, ckpt_dir, sha_prefix, per_theorem)
            )

    for arm_label, dir_names in _LEGACY_LENGTHEXP_ARMS.items():
        for dir_name in dir_names:
            eval_dir = _LENGTHEXP_EVAL_DIR / dir_name
            if not eval_dir.is_dir():
                continue
            for data_path in sorted(eval_dir.glob("*.jsonl")):
                run_ids.append(_import_lengthexp_run(store, arm_label, eval_dir, data_path))

    return {"n_runs": len(run_ids), "run_ids": cast(list[JSON], run_ids)}
