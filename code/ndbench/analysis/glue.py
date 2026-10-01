"""B1: the glued (compositionality) benchmark -- dashboard page "glue".

Reads the `glue` suite (WP4's `suites/glue.py`; joined items tagged `op`/`k`/`part_len`/`cell`,
`Problem.parts` holding each part's `pid`) and the parts suite (default `parts_pool`, WP2's
`suites/existing.py`; the mastery run G1 in plan §2 B1). Per `(cell, model)`:

- `R`/`R_min` with bootstrap CIs (`metrics.compositionality_ratio`) over joined items whose every
  part the model has "mastered" (`mastery` in `cfg`, default 0.5, matching plan §2 B1's `p_hat >=
  0.5`). Each part's pass rate comes from the part *as it appears in the join* -- the
  `glue_parts_renamed` suite (`suites/glue.py:build_renamed_parts`; AND/CHAIN/CASE joins rename
  part atoms) -- when that suite has a run for the model, falling back to the original
  `parts_suite` item otherwise (`compositionality_pool_parts` keeps the pool-denominator R for
  comparison; audit 2026-10-01);
- the fraction of attempts at least as long as the reference (pooled `sum(long_enough) /
  sum(n)` over the cell's joined items -- `ItemResult.long_enough` is already, per its WP0
  docstring, "attempts whose written length is at least the problem's reference length", exactly
  plan §2 B1's "fraction of attempts >= reference length", so this reuses it directly rather than
  recomputing anything from raw samples);
- a failure-location breakdown, attributing each failing attempt's `fail_lines` entry to the part
  whose original line range it falls in (offsets built from each part's own `ReferenceProof.lines`
  in `Problem.parts` order), or to `"glue"` for a line past every part's range (the operator's own
  joining lines: `ANDI`/`ORE`/... per plan §2 B1's construction table). This is an approximation
  from data already in the `Store` (line offsets + `ItemResult.fail_lines`), not a port of WP4's
  `failloc.py` (which inspects raw proof text `Store` does not expose per line);
- control pass@1 on `cfg["control_suite"]` (default `gen_ctrl`) items at the same reference length
  as the cell's own (rounded, modal) `L`, when any such control items were sampled.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ..core.types import JSON, ItemResult, Problem
from ..metrics import compositionality_ratio, pass_at_k
from ..store import Store
from ._common import (
    DEFAULT_JUDGE_ID,
    cfg_get,
    default_version,
    empty_view,
    item_length,
    items_by_pid,
    latest_run_id,
    models_present,
    problems_by_pid,
    safe_div,
)

DEFAULT_SUITE = "glue"
DEFAULT_PARTS_SUITE = "parts_pool"
DEFAULT_RENAMED_PARTS_SUITE = "glue_parts_renamed"
DEFAULT_CONTROL_SUITE = "gen_ctrl"
CELL_TAG = "cell"
DEFAULT_MASTERY = 0.5


def _line_offsets(joined: Problem, parts: Mapping[str, Problem]) -> list[tuple[int, int, str]]:
    offsets: list[tuple[int, int, str]] = []
    cur = 1
    for pid in joined.parts:
        part = parts.get(pid)
        length = part.ref.lines if part is not None and part.ref is not None else None
        if length is None or length <= 0:
            continue
        offsets.append((cur, cur + length - 1, pid))
        cur += length
    return offsets


def _attribute_line(line: int, offsets: Sequence[tuple[int, int, str]]) -> str:
    for start, end, pid in offsets:
        if start <= line <= end:
            return pid
    return "glue"


def _failure_location(
    item: ItemResult, joined: Problem, parts: Mapping[str, Problem]
) -> dict[str, int]:
    offsets = _line_offsets(joined, parts)
    counts: Counter[str] = Counter()
    for line in item.fail_lines:
        counts[_attribute_line(line, offsets)] += 1
    return dict(counts)


def _control_pass1(
    store: Store,
    control_suite: str,
    version: str,
    model: str,
    target_l: float,
    *,
    judge_id: str | None = DEFAULT_JUDGE_ID,
) -> dict[str, Any] | None:
    control_problems = problems_by_pid(store, control_suite, version)
    if not control_problems:
        return None
    run_id = latest_run_id(
        store, suite=control_suite, model=model, version=version, judge_id=judge_id
    )
    if run_id is None:
        return None
    items = items_by_pid(store, run_id)
    matched = [
        it
        for pid, it in items.items()
        if (p := control_problems.get(pid)) is not None
        and (length := item_length(p)) is not None
        and round(length) == round(target_l)
    ]
    if not matched:
        return None
    vals = [pass_at_k(it.n, it.c, 1) for it in matched]
    return {"n_items": len(matched), "pass1": sum(vals) / len(vals), "L": target_l}


def renamed_part_items(
    store: Store,
    model: str,
    suite: str = DEFAULT_RENAMED_PARTS_SUITE,
    *,
    judge_id: str | None = DEFAULT_JUDGE_ID,
) -> dict[str, ItemResult]:
    """`model`'s latest results on the renamed-parts suite, keyed `<joined pid>__p<i>` (empty when
    the suite or a run for `model` is missing). `judge_id` (default unchanged, "nd_v2"): plan
    §2 WP-D2 item 5 / WP-D5 item 1 -- a Dan model's renamed-parts run is judged `"lean_v2"`."""
    version = default_version(store, suite=suite, models=[model], judge_id=judge_id) or "v1"
    run_id = latest_run_id(store, suite=suite, model=model, version=version, judge_id=judge_id)
    return items_by_pid(store, run_id) if run_id else {}


def part_results(
    joined_pid: str,
    joined: Problem,
    pool_items: Mapping[str, ItemResult],
    renamed_items: Mapping[str, ItemResult],
) -> tuple[list[ItemResult], str] | None:
    """The joined item's per-part results and their source: `"renamed"` (every part sampled as it
    appears inside the join) when available, else `"pool"` (the original part items), else
    `None` if some part has no result."""
    if not joined.parts:
        return None
    renamed = [renamed_items.get(f"{joined_pid}__p{i}") for i in range(len(joined.parts))]
    if all(r is not None for r in renamed):
        return [r for r in renamed if r is not None], "renamed"
    pool = [pool_items[p] for p in joined.parts if p in pool_items]
    if len(pool) == len(joined.parts):
        return pool, "pool"
    return None


def view(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    suite = cast(str, cfg_get(cfg, "suite", DEFAULT_SUITE))
    parts_suite = cast(str, cfg_get(cfg, "parts_suite", DEFAULT_PARTS_SUITE))
    renamed_suite = cast(str, cfg_get(cfg, "renamed_parts_suite", DEFAULT_RENAMED_PARTS_SUITE))
    control_suite = cast(str, cfg_get(cfg, "control_suite", DEFAULT_CONTROL_SUITE))
    mastery = float(cast(float, cfg_get(cfg, "mastery", DEFAULT_MASTERY)))
    B = int(cast(int, cfg_get(cfg, "B", 2000)))
    seed = int(cast(int, cfg_get(cfg, "seed", 0)))
    # `cell_tag` (default unchanged, "cell"): WP-D5's Dan glue panel pools by the suite's coarse
    # `op` tag (5 values: AND/CHAIN/CASE-dilemma/CASE-same/BOX) rather than the fine-grained `cell`
    # tag (19 values on the real `glue` suite), since a 19-bar-per-panel chart does not read as a
    # small multiple. Passing `cell_tag="op"` makes every lookup below key on `op` directly (the
    # `p.tags.get(CELL_TAG, ...)` fallback to `"op"` only fires when the configured tag is itself
    # absent, so overriding the tag name -- not just its fallback -- is what actually changes the
    # grouping); the ordinary "ours" callers never pass this, so their behaviour is unchanged.
    cell_tag = cast(str, cfg_get(cfg, "cell_tag", CELL_TAG))
    override_models = cast("list[str] | None", cfg.get("models"))
    # `judge_id` (plan DANSTATE_PLAN.md §2 WP-D2 item 5 / WP-D5 item 1, default unchanged): see
    # `analysis/yue.py:view`'s identical parameter.
    judge_id = cast("str | None", cfg.get("judge_id", DEFAULT_JUDGE_ID))
    # Fix list item 1, applied uniformly: pick whichever version has runs rather than a
    # hard-coded "v1". `parts_suite`/`control_suite` are *different* suites (e.g. `glue` can have
    # a demo version alongside its real one -- see `glue`'s own `demo_v1` vs. `v1` on the real
    # store -- while `parts_pool`/`gen_ctrl` need not share that version), so each is resolved
    # independently rather than reusing `suite`'s own chosen version.
    version = cast(
        str,
        cfg.get("version")
        or default_version(store, suite=suite, models=override_models, judge_id=judge_id)
        or "v1",
    )

    problems = problems_by_pid(store, suite, version)
    if not problems:
        return empty_view(f"suite {suite!r} version {version!r} not found in the store")
    models = override_models or models_present(
        store, suite=suite, version=version, judge_id=judge_id
    )
    if not models:
        return empty_view(f"no runs found for suite {suite!r}")
    parts_version = cast(
        str,
        cfg.get("parts_suite_version")
        or default_version(store, suite=parts_suite, models=models, judge_id=judge_id)
        or "v1",
    )
    control_version = cast(
        str,
        cfg.get("control_suite_version")
        or default_version(store, suite=control_suite, models=models, judge_id=judge_id)
        or "v1",
    )
    parts_problems = problems_by_pid(store, parts_suite, parts_version)

    cells = sorted(
        {
            str(v)
            for p in problems.values()
            if (v := p.tags.get(cell_tag, p.tags.get("op"))) is not None
        }
    )
    if not cells:
        return empty_view(f"suite {suite!r} has no {cell_tag!r}/'op' tag on any item")

    out_cells: dict[str, Any] = {}
    for cell in cells:
        cell_pids = {
            pid
            for pid, p in problems.items()
            if str(p.tags.get(cell_tag, p.tags.get("op"))) == cell
        }
        per_model: dict[str, Any] = {}
        for model in models:
            run_id = latest_run_id(
                store, suite=suite, model=model, version=version, judge_id=judge_id
            )
            parts_run_id = latest_run_id(
                store,
                suite=parts_suite,
                model=model,
                version=parts_version,
                judge_id=judge_id,
            )
            if run_id is None:
                continue
            joined_items = items_by_pid(store, run_id)
            part_items = items_by_pid(store, parts_run_id) if parts_run_id else {}
            renamed_items = renamed_part_items(store, model, renamed_suite, judge_id=judge_id)
            groups: list[tuple[ItemResult, list[ItemResult]]] = []
            pool_groups: list[tuple[ItemResult, list[ItemResult]]] = []
            sources: Counter[str] = Counter()
            long_enough_sum = 0
            n_sum = 0
            c_sum = 0
            fail_loc_total: Counter[str] = Counter()
            for pid in cell_pids:
                item = joined_items.get(pid)
                if item is None:
                    continue
                long_enough_sum += item.long_enough
                n_sum += item.n
                c_sum += item.c
                joined_problem = problems[pid]
                fail_loc = _failure_location(item, joined_problem, parts_problems)
                fail_loc_total.update(fail_loc)
                found = part_results(pid, joined_problem, part_items, renamed_items)
                if found is not None:
                    groups.append((item, found[0]))
                    sources[found[1]] += 1
                pool = part_results(pid, joined_problem, part_items, {})
                if pool is not None:
                    pool_groups.append((item, pool[0]))
            if not groups and n_sum == 0:
                continue
            comp = (
                compositionality_ratio(groups, mastery=mastery, B=B, seed=seed) if groups else None
            )
            comp_pool = (
                compositionality_ratio(pool_groups, mastery=mastery, B=B, seed=seed)
                if pool_groups
                else None
            )
            lengths = [
                length
                for pid in cell_pids
                if (p := problems.get(pid)) is not None and (length := item_length(p)) is not None
            ]
            modal_l = sorted(lengths)[len(lengths) // 2] if lengths else None
            control = (
                _control_pass1(
                    store, control_suite, control_version, model, modal_l, judge_id=judge_id
                )
                if modal_l is not None
                else None
            )
            per_model[model] = {
                "n_joined_items": sum(1 for pid in cell_pids if pid in joined_items),
                "n_mastery_groups": len(groups),
                "compositionality": comp,
                "compositionality_pool_parts": comp_pool,
                "parts_source": dict(sources),
                "joined_pass1": safe_div(float(c_sum), float(n_sum)),
                "frac_long_enough": safe_div(float(long_enough_sum), float(n_sum)),
                "failure_location": dict(fail_loc_total),
                "modal_L": modal_l,
                "control_pass1": control,
                # Surfaced explicitly (not just a silently-missing R/R_min bar) when this model has
                # joined-item results but no mastery run on `parts_suite` to compute R/R_min from
                # at all (e.g. the real store's `parts_pool` has no `armC` run) -- distinct from
                # `comp is None` with a mastery run present but zero items reaching the mastery
                # threshold, which is a real (if uninformative) finding rather than missing data.
                "no_mastery_run": parts_run_id is None,
            }
        if per_model:
            out_cells[cell] = per_model

    if not out_cells:
        return empty_view(f"suite {suite!r} has runs but no items matched any cell")
    return {
        "empty": False,
        "meta": {
            "suite": suite,
            "parts_suite": parts_suite,
            "parts_suite_version": parts_version,
            "renamed_parts_suite": renamed_suite,
            "control_suite": control_suite,
            "control_suite_version": control_version,
            "version": version,
            "mastery": mastery,
            "models": models,
            "B": B,
            "judge_id": judge_id,
        },
        "cells": out_cells,
    }
