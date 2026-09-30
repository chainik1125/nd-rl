"""Evaluation suites for the steering sweep and weight interpolation (`INTERP_SPEC.md` §2/§3):
"the `gen` items of `ab2/data/dev_L8-11.jsonl` (720) and of `ab2/sealed/sealed_L12-14.jsonl`
(1,240), plus `textbook_dev`" and "a fixed 150-theorem subset (stratified by L, both sets) at
k=128".

`suites.existing` (WP2) has no `gen`-item builder for these two files at full size (`gen_ctrl`
caps at 60 for the Yue-replication *control*, `parts_pool` further restricts to `L` 2-8 with one
item per `pkey`) and its own `textbook_dev` needs a `Certifier` to relabel all 58 items by fresh
minlen search (up to 120s each) -- too expensive to run incidentally from an eval-suite loader. So
this module
builds the two `gen`-item pools directly (their existing `L_true` is kept as a `"reference"` label,
the same way `suites.existing.gen_ctrl` treats the same file), and *loads* `textbook_dev` from the
`Store` only if another work package has already built it there, skipping with a note otherwise --
per the plan's own "reuse `suites.existing` builders where they exist, else add `interp_eval` suites
in your own module".

These are plain `Problem` lists written straight to the `Store` (`store.write_suite`), not
registered in `suites/__init__.py`'s `BUILDERS` (that file is WP0's; this module's suites are only
ever built by `interp/__main__.py` itself, not through the shared `build` CLI subcommand).
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.io import read_jsonl, utc_now_iso
from ..core.types import JSON, Excluder, LengthLabel, Problem, SuiteManifest
from ..store import Store
from .data import DEV_L8_11_PATH, SEALED_L12_14_PATH, _seq_of

DEV_GEN_SUITE = "interp_eval_dev"
SEALED_GEN_SUITE = "interp_eval_sealed"
K128_SUITE = "interp_eval_k128"
#: `INTERP_SPEC.md` §3: "a fixed 150-theorem subset (stratified by L, both sets) at k=128".
K128_N_ITEMS = 150


def _gen_items(path: Path, suite: str, version: str, excluder: Excluder | None) -> list[Problem]:
    out: list[Problem] = []
    for row in read_jsonl(path):
        if row.get("source") != "gen":
            continue
        seq = _seq_of(row)
        in_train, eval_sets = (False, ()) if excluder is None else excluder.status(seq)
        l_true = row.get("L_true")
        out.append(
            Problem(
                pid=str(row["name"]),
                suite=suite,
                version=version,
                sequent=seq,
                length=LengthLabel(
                    L=l_true, L_ub=l_true, L_lb=None, method="reference", bound=None
                ),
                ref=None,
                tags={"L": l_true, "n_prem": row.get("n_prem")},
                seen_train=in_train,
                seen_eval=eval_sets,
            )
        )
    return out


def _manifest(suite: str, version: str, n_items: int, config: Mapping[str, JSON]) -> SuiteManifest:
    return SuiteManifest(
        suite=suite,
        version=version,
        builder="code.ndbench.interp.eval_suites",
        config=dict(config),
        seed=0,
        created_utc=utc_now_iso(),
        git_commit="unknown",
        n_items=n_items,
        items_sha256="",
        excl_sources={},
        stats={"n_items": n_items},
    )


def build_dev_gen(
    store: Store, *, excluder: Excluder | None = None, version: str = "v1"
) -> list[Problem]:
    """Every `source == "gen"` item of `ab2/data/dev_L8-11.jsonl` (720)."""
    problems = _gen_items(DEV_L8_11_PATH, DEV_GEN_SUITE, version, excluder)
    store.write_suite(problems, _manifest(DEV_GEN_SUITE, version, len(problems), {}))
    return problems


def build_sealed_gen(
    store: Store, *, excluder: Excluder | None = None, version: str = "v1"
) -> list[Problem]:
    """Every `source == "gen"` item of `ab2/sealed/sealed_L12-14.jsonl` (1,240)."""
    problems = _gen_items(SEALED_L12_14_PATH, SEALED_GEN_SUITE, version, excluder)
    store.write_suite(problems, _manifest(SEALED_GEN_SUITE, version, len(problems), {}))
    return problems


def build_k128_subset(
    dev_items: Sequence[Problem],
    sealed_items: Sequence[Problem],
    store: Store,
    *,
    n_items: int = K128_N_ITEMS,
    seed: int = 0,
    version: str = "v1",
) -> list[Problem]:
    """A fixed `n_items`-theorem subset of `dev_items + sealed_items`, stratified by `L` (as even a
    split across distinct `L` values as `n_items` allows, deterministic given `seed`)."""
    by_l: dict[Any, list[Problem]] = {}
    for p in (*dev_items, *sealed_items):
        by_l.setdefault(p.length.L, []).append(p)
    ls = sorted(by_l, key=lambda x: (x is None, x))
    rng = random.Random(seed)
    base, remainder = divmod(n_items, max(len(ls), 1))
    selected: list[Problem] = []
    for i, l_val in enumerate(ls):
        cand = list(by_l[l_val])
        rng.shuffle(cand)
        target = base + (1 if i < remainder else 0)
        selected.extend(cand[:target])
    # re-tag pids/suite/version so this is its own independent suite, not aliasing the source ones
    retagged = [
        Problem(
            pid=p.pid,
            suite=K128_SUITE,
            version=version,
            sequent=p.sequent,
            length=p.length,
            ref=p.ref,
            tags=p.tags,
            seen_train=p.seen_train,
            seen_eval=p.seen_eval,
        )
        for p in selected
    ]
    store.write_suite(
        retagged, _manifest(K128_SUITE, version, len(retagged), {"n_items": n_items, "seed": seed})
    )
    return retagged


def load_textbook_dev(store: Store, *, version: str = "v1") -> list[Problem]:
    """`textbook_dev` from the `Store`, if some other work package has already built it there
    (`suites.existing.textbook_dev` needs a `Certifier` to relabel all 58 items, up to 120s each --
    too expensive to run incidentally here); `[]` otherwise."""
    if not (store.suites_dir / "textbook_dev" / version).is_dir():
        return []
    return store.load_suite("textbook_dev", version)


@dataclass(frozen=True)
class EvalSuites:
    dev: list[Problem]
    sealed: list[Problem]
    k128: list[Problem]
    textbook_dev: list[Problem]
    notes: list[str]


def build_all(
    store: Store, *, excluder: Excluder | None = None, seed: int = 0, version: str = "v1"
) -> EvalSuites:
    dev = build_dev_gen(store, excluder=excluder, version=version)
    sealed = build_sealed_gen(store, excluder=excluder, version=version)
    k128 = build_k128_subset(dev, sealed, store, seed=seed, version=version)
    textbook = load_textbook_dev(store, version=version)
    notes = [] if textbook else ["no textbook_dev suite in the store yet; skipping"]
    return EvalSuites(dev=dev, sealed=sealed, k128=k128, textbook_dev=textbook, notes=notes)
