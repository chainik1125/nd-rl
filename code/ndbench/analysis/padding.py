"""B5: free lines (padding) -- dashboard page "padding".

Reads the `padding` suite (WP3's `suites/padding.py`; `Problem.tags` carry `pad_type`/`m`/
`base_pid`, plan §3's WP3 row) and, per `(pad_type, model)`:

- pass@1 against written lines `W` for the padded-easy items (`W` = the item's own target length
  `L_base + m`, via `_common.item_length`), alongside the same curve for "natural" items at the
  same `W` from `cfg["natural_suite"]` (default `gen_ctrl`, WP2's existing-items builder) -- plan
  §2 B5's "pass@1 vs written lines W for padded-easy vs natural items at the same W";
- the "free-line cost": a logit slope of pass@1 per added padding line `m` (`_common.
  logit_slope_safe`), per `pad_type`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from ..core.types import JSON
from ..metrics import pass_at_k
from ..store import Store
from ._common import (
    DEFAULT_JUDGE_ID,
    cfg_get,
    default_version,
    empty_view,
    item_length,
    items_by_pid,
    latest_run_id,
    logit_slope_safe,
    models_present,
    problems_by_pid,
)

DEFAULT_SUITE = "padding"
DEFAULT_NATURAL_SUITE = "gen_ctrl"
PAD_TYPE_TAG = "pad_type"
M_TAG = "m"


def _mean_or_nan(vals: Sequence[float]) -> float:
    clean = [v for v in vals if not math.isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


def _curve_vs_w(by_w: Mapping[float, list[tuple[int, int]]]) -> list[dict[str, Any]]:
    return [
        {
            "W": w,
            "pass1": _mean_or_nan([pass_at_k(n, c, 1) for n, c in group]),
            "n_items": len(group),
        }
        for w, group in sorted(by_w.items())
    ]


def view(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    suite = cast(str, cfg_get(cfg, "suite", DEFAULT_SUITE))
    natural_suite = cast(str, cfg_get(cfg, "natural_suite", DEFAULT_NATURAL_SUITE))
    override_models = cast("list[str] | None", cfg.get("models"))
    # `judge_id` (plan DANSTATE_PLAN.md §2 WP-D2 item 5 / WP-D5 item 1, default unchanged): see
    # `analysis/yue.py:view`'s identical parameter.
    judge_id = cast("str | None", cfg.get("judge_id", DEFAULT_JUDGE_ID))
    # Pick whichever suite version actually has runs for the models in question, rather than
    # hard-coding "v1" -- a real suite can accumulate a small demo version nobody ran alongside
    # the version the real runs target (fix list item 1: `padding`'s 132-item v1 demo vs. the
    # 51-base v2 that base/rl were actually run on). `cfg["version"]` still overrides explicitly.
    version = cast(
        str,
        cfg.get("version")
        or default_version(store, suite=suite, models=override_models, judge_id=judge_id)
        or "v1",
    )
    # The natural-items comparison suite is a *different* suite from `padding` (default
    # `gen_ctrl`) and so can land on a different version of its own (e.g. `padding` picks "v2"
    # while `gen_ctrl` only ever had a "v1") -- resolve it independently rather than reusing
    # `padding`'s own chosen `version` (that reuse used to work by coincidence when both defaulted
    # to the literal "v1").
    natural_version = cast(
        str,
        cfg.get("natural_version")
        or default_version(store, suite=natural_suite, models=override_models, judge_id=judge_id)
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
    natural_problems = problems_by_pid(store, natural_suite, natural_version)

    pad_types = sorted(
        {str(v) for p in problems.values() if (v := p.tags.get(PAD_TYPE_TAG)) is not None}
    )
    if not pad_types:
        return empty_view(f"suite {suite!r} has no {PAD_TYPE_TAG!r} tag on any item")

    out: dict[str, Any] = {}
    for pad_type in pad_types:
        pids = {pid for pid, p in problems.items() if str(p.tags.get(PAD_TYPE_TAG)) == pad_type}
        per_model: dict[str, Any] = {}
        for model in models:
            run_id = latest_run_id(
                store, suite=suite, model=model, version=version, judge_id=judge_id
            )
            if run_id is None:
                continue
            items = items_by_pid(store, run_id)
            rows: list[tuple[JSON, float, int, int]] = []  # (m, W, n, c)
            for pid in pids:
                item = items.get(pid)
                if item is None:
                    continue
                w = item_length(problems[pid])
                if w is None:
                    continue
                rows.append((problems[pid].tags.get(M_TAG), w, item.n, item.c))
            if not rows:
                continue

            by_w: dict[float, list[tuple[int, int]]] = {}
            for _m, w, n, c in rows:
                by_w.setdefault(w, []).append((n, c))
            padded_curve = _curve_vs_w(by_w)

            natural_curve: list[dict[str, Any]] = []
            if natural_problems:
                nat_run = latest_run_id(
                    store,
                    suite=natural_suite,
                    model=model,
                    version=natural_version,
                    judge_id=judge_id,
                )
                if nat_run is not None:
                    nat_items = items_by_pid(store, nat_run)
                    nat_by_w: dict[float, list[tuple[int, int]]] = {}
                    for pid2, p2 in natural_problems.items():
                        it2 = nat_items.get(pid2)
                        w2 = item_length(p2)
                        if it2 is None or w2 is None or w2 not in by_w:
                            continue
                        nat_by_w.setdefault(w2, []).append((it2.n, it2.c))
                    natural_curve = _curve_vs_w(nat_by_w)

            m_rows = [(float(cast(float, m)), n, c) for m, _w, n, c in rows if m is not None]
            slope = (
                logit_slope_safe(
                    [m for m, _, _ in m_rows],
                    [n for _, n, _ in m_rows],
                    [c for _, _, c in m_rows],
                )
                if m_rows
                else None
            )

            per_model[model] = {
                "n_items": len(rows),
                "padded_curve_vs_W": padded_curve,
                "natural_curve_vs_W": natural_curve,
                "free_line_cost_per_line": slope,
            }
        if per_model:
            out[pad_type] = per_model

    if not out:
        return empty_view(f"suite {suite!r} has runs but no items matched any pad_type")
    return {
        "empty": False,
        "meta": {
            "suite": suite,
            "natural_suite": natural_suite,
            "version": version,
            "natural_version": natural_version,
            "models": models,
            "judge_id": judge_id,
        },
        "pad_types": out,
    }
