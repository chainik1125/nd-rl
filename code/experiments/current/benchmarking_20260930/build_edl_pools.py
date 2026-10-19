"""Build the `families`/`edl_v1` and `laws`/`edl_v1` suite versions: dedicated, larger EDL pools
for the plan §B7 classes whose main `"v1"` suite (~40 distinct items/cell) falls short of
`edl.build_classes`'s >=64-distinct-item threshold (see `edl.py`'s module docstring note 5 and
`PLAN.md` §2 B7 / §A's 2026-10-01 EDL-pool fix).

Run from the worktree root:

    code/.venv/bin/python code/experiments/current/benchmarking_20260930/build_edl_pools.py

Builds, in one process (so a single `MinlenCertifier` can reuse its worker pool and on-disk label
cache across both suites):

* `families`/`edl_v1`: and_tower n=8/16, imp_nest n=4/8, reorder_and n=5, reorder_or n=3 -- the six
  (family, n) cells plan §B7's EDL classes need, each targeted at 128 distinct items, one
  `families.build()` call (its `n_grid` override is per-family, so one call covers exactly these
  six cells and nothing else).
* `laws`/`edl_v1`: contraposition s=3, dist_and_over_or s=1, demorgan_nand_to_or s=1 -- three
  separate `laws.build()` calls (one per (schema, s), since `laws.build`'s own `s_values` config
  key is shared across every selected schema in one call) merged into a single suite version.

Same certification policy as the main `"v1"` suites (`families.yaml`/`laws.yaml`): the real
`MinlenCertifier` (plan §A2: procs=8, the shared `out/cache` label cache) and `DefaultExcluder`
(flags `seen_train`/`seen_eval`, never drops -- `families.py`'s/`laws.py`'s own builders only ever
flag; `edl.gen_L10` is the one loader that *drops* seen_train rows, per the main agent's 2026-10-01
instruction, and that is handled separately in `edl.py`, not here). `target_per_n`/`target_per_s`
are raised to 128 (plan: "Target 128 distinct items per cell"); `max_attempts` is raised well above
the default so `collect_candidates` has enough random draws to actually reach 128 distinct pkeys
where the combinatorics allow it. Per the main agent's instruction, a cell that cannot reach 64
distinct items (known case: `reorder_or` n=3, ~4 distinct up to atom-renaming/premise-reordering)
is reported, not padded with duplicates -- `edl.build_classes`'s own >=64 threshold already drops
such a class downstream; this script just prints each cell's actual distinct (post `Sequent.key()`
dedup) count for the report.

Refuses to run (prints an error, exits 1) if either suite version's `items.jsonl` already exists,
per the "never overwrite an existing suite version" instruction -- re-running this script after a
partial/failed attempt requires deleting that version's directory by hand first (a deliberate
speed bump, not automated, so a stale pool is never silently clobbered).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from code.ndbench.core.io import git_commit_sha, sha256_str, utc_now_iso  # noqa: E402
from code.ndbench.core.types import JSON, Problem, SuiteManifest  # noqa: E402
from code.ndbench.logic.certify import MinlenCertifier  # noqa: E402
from code.ndbench.logic.exclusion import DEFAULT_CACHE_DIR, DefaultExcluder  # noqa: E402
from code.ndbench.store import Store  # noqa: E402
from code.ndbench.suites import families, laws  # noqa: E402

OUT = Path("/scratch/cowifornia/nd-rl/bench/out")
VERSION = "edl_v1"
TARGET = 128
MAX_ATTEMPTS = 1500
SEED = 0

FAMILIES_CFG: dict[str, object] = {
    "families": ["and_tower", "imp_nest", "reorder_and", "reorder_or"],
    "n_grid": {
        "and_tower": [8, 16],
        "imp_nest": [4, 8],
        "reorder_and": [5],
        "reorder_or": [3],
    },
    "target_per_n": TARGET,
    "max_attempts": MAX_ATTEMPTS,
    "version": VERSION,
}

#: One `(schemas, s_values)` cfg per `laws.build()` call -- see module docstring for why this
#: can't be a single call the way `families.build()`'s per-family `n_grid` override allows.
LAWS_CELLS: list[tuple[str, int]] = [
    ("contraposition", 3),
    ("dist_and_over_or", 1),
    ("demorgan_nand_to_or", 1),
]


def _write_suite(
    store: Store,
    suite: str,
    builder: str,
    cfg: dict[str, JSON],
    problems: list[Problem],
    stats: dict[str, JSON],
) -> None:
    existing = OUT / "suites" / suite / VERSION / "items.jsonl"
    if existing.exists():
        sys.exit(f"refusing to overwrite existing suite version: {existing}")
    items_json = [p.to_json() for p in problems]
    manifest = SuiteManifest(
        suite=suite,
        version=VERSION,
        builder=builder,
        config=cfg,
        seed=SEED,
        created_utc=utc_now_iso(),
        git_commit=git_commit_sha(),
        n_items=len(problems),
        items_sha256=sha256_str(json.dumps(items_json, sort_keys=True)),
        excl_sources={"excluder": "code.ndbench.logic.exclusion:DefaultExcluder"},
        stats=stats,
    )
    store.write_suite(problems, manifest)
    print(
        json.dumps({"suite": suite, "version": VERSION, **asdict(manifest)}, indent=2, default=str)
    )


def _distinct_report(problems: list[Problem], match_fns: dict[str, object]) -> None:
    """Print each target cell's distinct (post `Sequent.key()` dedup, matching
    `edl._dedup_by_key`) item count -- the number `build_classes` will actually see."""
    for label, match in match_fns.items():
        pool = [p for p in problems if match(p) and p.ref is not None]  # type: ignore[operator]
        seen: dict[str, Problem] = {}
        for p in pool:
            seen.setdefault(p.sequent.key(), p)
        print(f"  {label}: {len(pool)} built, {len(seen)} distinct (after Sequent.key() dedup)")


def main() -> None:
    pre_existing = [
        s
        for s in (OUT / "suites" / "families" / VERSION, OUT / "suites" / "laws" / VERSION)
        if s.exists()
    ]
    if pre_existing:
        sys.exit(f"refusing to run: suite version(s) already exist: {pre_existing}")

    store = Store(OUT)
    certifier = MinlenCertifier(procs=8, cache_dir=DEFAULT_CACHE_DIR)
    excluder = DefaultExcluder()

    print(f"=== building families/{VERSION} ===")
    fam_problems, fam_stats = families.build(
        FAMILIES_CFG, certifier=certifier, excluder=excluder, rng_seed=SEED
    )
    print(json.dumps(fam_stats, indent=2, default=str))
    _distinct_report(
        fam_problems,
        {
            "and_tower n=8": lambda p: p.tags.get("family") == "and_tower" and p.tags.get("n") == 8,
            "and_tower n=16": lambda p: (
                p.tags.get("family") == "and_tower" and p.tags.get("n") == 16
            ),
            "imp_nest n=4": lambda p: p.tags.get("family") == "imp_nest" and p.tags.get("n") == 4,
            "imp_nest n=8": lambda p: p.tags.get("family") == "imp_nest" and p.tags.get("n") == 8,
            "reorder_and n=5": lambda p: (
                p.tags.get("family") == "reorder_and" and p.tags.get("n") == 5
            ),
            "reorder_or n=3": lambda p: (
                p.tags.get("family") == "reorder_or" and p.tags.get("n") == 3
            ),
        },
    )
    _write_suite(
        store,
        "families",
        "code.ndbench.suites.families:build",
        FAMILIES_CFG,
        fam_problems,
        fam_stats,
    )

    print(f"\n=== building laws/{VERSION} ===")
    law_problems: list[Problem] = []
    law_stats: dict[str, JSON] = {}
    for schema, s in LAWS_CELLS:
        cfg: dict[str, object] = {
            "schemas": [schema],
            "s_values": [s],
            "target_per_s": TARGET,
            "max_attempts": MAX_ATTEMPTS,
            "version": VERSION,
        }
        probs, stats = laws.build(cfg, certifier=certifier, excluder=excluder, rng_seed=SEED)
        print(f"  {schema} s={s}: {json.dumps(stats, default=str)}")
        law_problems.extend(probs)
        law_stats.update(stats)
    _distinct_report(
        law_problems,
        {
            "contraposition s=3": lambda p: (
                p.tags.get("schema") == "contraposition" and p.tags.get("s") == 3
            ),
            "dist_and_over_or s=1": lambda p: (
                p.tags.get("schema") == "dist_and_over_or" and p.tags.get("s") == 1
            ),
            "demorgan_nand_to_or s=1": lambda p: (
                p.tags.get("schema") == "demorgan_nand_to_or" and p.tags.get("s") == 1
            ),
        },
    )
    laws_cfg_combined: dict[str, object] = {
        "cells": LAWS_CELLS,
        "target_per_s": TARGET,
        "version": VERSION,
    }
    _write_suite(
        store, "laws", "code.ndbench.suites.laws:build", laws_cfg_combined, law_problems, law_stats
    )


if __name__ == "__main__":
    main()
