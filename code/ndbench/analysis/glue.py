"""B1: the glued (compositionality) benchmark -- dashboard page "glue".

Reads the `glue` suite (WP4's `suites/glue.py`; joined items tagged `op`/`k`/`part_len`/`cell`,
`Problem.parts` holding each part's `pid`) and the parts suite (default `parts_pool`, WP2's
`suites/existing.py`; the mastery run G1 in plan §2 B1). Per `(cell, model)`:

- `R`/`R_min` with bootstrap CIs (`metrics.compositionality_ratio`) over joined items whose every
  part the model has "mastered" (`mastery` in `cfg`, default 0.5, matching plan §2 B1's `p_hat >=
  0.5`);
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
    store: Store, control_suite: str, version: str, model: str, target_l: float
) -> dict[str, Any] | None:
    control_problems = problems_by_pid(store, control_suite, version)
    if not control_problems:
        return None
    run_id = latest_run_id(store, suite=control_suite, model=model, version=version)
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


def view(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    suite = cast(str, cfg_get(cfg, "suite", DEFAULT_SUITE))
    parts_suite = cast(str, cfg_get(cfg, "parts_suite", DEFAULT_PARTS_SUITE))
    control_suite = cast(str, cfg_get(cfg, "control_suite", DEFAULT_CONTROL_SUITE))
    mastery = float(cast(float, cfg_get(cfg, "mastery", DEFAULT_MASTERY)))
    B = int(cast(int, cfg_get(cfg, "B", 2000)))
    seed = int(cast(int, cfg_get(cfg, "seed", 0)))
    override_models = cast("list[str] | None", cfg.get("models"))
    # Fix list item 1, applied uniformly: pick whichever version has runs rather than a
    # hard-coded "v1". `parts_suite`/`control_suite` are *different* suites (e.g. `glue` can have
    # a demo version alongside its real one -- see `glue`'s own `demo_v1` vs. `v1` on the real
    # store -- while `parts_pool`/`gen_ctrl` need not share that version), so each is resolved
    # independently rather than reusing `suite`'s own chosen version.
    version = cast(
        str,
        cfg.get("version") or default_version(store, suite=suite, models=override_models) or "v1",
    )

    problems = problems_by_pid(store, suite, version)
    if not problems:
        return empty_view(f"suite {suite!r} version {version!r} not found in the store")
    models = override_models or models_present(store, suite=suite, version=version)
    if not models:
        return empty_view(f"no runs found for suite {suite!r}")
    parts_version = cast(
        str,
        cfg.get("parts_suite_version")
        or default_version(store, suite=parts_suite, models=models)
        or "v1",
    )
    control_version = cast(
        str,
        cfg.get("control_suite_version")
        or default_version(store, suite=control_suite, models=models)
        or "v1",
    )
    parts_problems = problems_by_pid(store, parts_suite, parts_version)

    cells = sorted(
        {
            str(v)
            for p in problems.values()
            if (v := p.tags.get(CELL_TAG, p.tags.get("op"))) is not None
        }
    )
    if not cells:
        return empty_view(f"suite {suite!r} has no {CELL_TAG!r}/'op' tag on any item")

    out_cells: dict[str, Any] = {}
    for cell in cells:
        cell_pids = {
            pid
            for pid, p in problems.items()
            if str(p.tags.get(CELL_TAG, p.tags.get("op"))) == cell
        }
        per_model: dict[str, Any] = {}
        for model in models:
            run_id = latest_run_id(store, suite=suite, model=model, version=version)
            parts_run_id = latest_run_id(
                store, suite=parts_suite, model=model, version=parts_version
            )
            if run_id is None:
                continue
            joined_items = items_by_pid(store, run_id)
            part_items = items_by_pid(store, parts_run_id) if parts_run_id else {}
            groups: list[tuple[ItemResult, list[ItemResult]]] = []
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
                part_results = [part_items[p] for p in joined_problem.parts if p in part_items]
                if part_results and len(part_results) == len(joined_problem.parts):
                    groups.append((item, part_results))
            if not groups and n_sum == 0:
                continue
            comp = (
                compositionality_ratio(groups, mastery=mastery, B=B, seed=seed) if groups else None
            )
            lengths = [
                length
                for pid in cell_pids
                if (p := problems.get(pid)) is not None and (length := item_length(p)) is not None
            ]
            modal_l = sorted(lengths)[len(lengths) // 2] if lengths else None
            control = (
                _control_pass1(store, control_suite, control_version, model, modal_l)
                if modal_l is not None
                else None
            )
            per_model[model] = {
                "n_joined_items": sum(1 for pid in cell_pids if pid in joined_items),
                "n_mastery_groups": len(groups),
                "compositionality": comp,
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
            "control_suite": control_suite,
            "control_suite_version": control_version,
            "version": version,
            "mastery": mastery,
            "models": models,
            "B": B,
        },
        "cells": out_cells,
    }
