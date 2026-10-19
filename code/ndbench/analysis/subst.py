"""B4: law x substitution size -- dashboard page "subst".

Reads the `laws` suite (WP3's `suites/laws.py`; `Problem.tags` carry `schema`/`s`, plan §3's WP3
row) and, per `(schema, model)`:

- pass@1 against filler size `s` (mean over items at each `s`);
- pass@1 against the item's own tokens (`_common.item_tokens`: prompt + reference proof; not the
  model's outcome-dependent generated length), as per-item points for a scatter;
- a logit slope per 100 tokens (`_common.logit_slope_safe`);
- a comparison against the same schema's uncontrolled instances in `cfg["textbook_law_suite"]`
  (default `textbook_law`, plan §2 B8's stratified real-textbook subsample, tagged by the same
  `schema` -- assumed, since both suites cover the same eight schemas): pooled pass@1 there against
  this schema's own controlled-substitution pass@1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from ..core.types import JSON
from ..metrics import pass_at_k
from ..store import Store
from . import models_registry
from ._common import (
    DEFAULT_JUDGE_ID,
    cfg_get,
    default_version,
    empty_view,
    item_tokens,
    items_by_pid,
    latest_run_id,
    logit_slope_safe,
    models_present,
    problems_by_pid,
)

DEFAULT_SUITE = "laws"
DEFAULT_TEXTBOOK_LAW_SUITE = "textbook_law"
SCHEMA_TAG = "schema"
S_TAG = "s"


def _s_sort_key(kv: tuple[JSON, Any]) -> tuple[bool, float, str]:
    """Numeric order for `s` values (was `str(s)`, which puts 10 before 2), `None` last."""
    s_val = kv[0]
    if isinstance(s_val, int | float) and not isinstance(s_val, bool):
        return (False, float(s_val), "")
    return (s_val is None, math.inf, str(s_val))


def _mean_or_nan(vals: Sequence[float]) -> float:
    clean = [v for v in vals if not math.isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


def view(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    suite = cast(str, cfg_get(cfg, "suite", DEFAULT_SUITE))
    textbook_suite = cast(str, cfg_get(cfg, "textbook_law_suite", DEFAULT_TEXTBOOK_LAW_SUITE))
    override_models = cast("list[str] | None", cfg.get("models"))
    # `judge_id` (plan DANSTATE_PLAN.md §2 WP-D2 item 5 / WP-D5 item 1, default unchanged): see
    # `analysis/yue.py:view`'s identical parameter.
    judge_id = cast("str | None", cfg.get("judge_id", DEFAULT_JUDGE_ID))
    # Fix list item 1, applied uniformly: pick whichever version has runs rather than a
    # hard-coded "v1". `textbook_suite` is a *different* suite and so is resolved independently
    # (it need not land on the same version `suite` does).
    version = cast(
        str,
        cfg.get("version")
        or default_version(store, suite=suite, models=override_models, judge_id=judge_id)
        or "v1",
    )

    problems = problems_by_pid(store, suite, version)
    if not problems:
        return empty_view(f"suite {suite!r} version {version!r} not found in the store")
    # Audit 2026-10-01: the plan drops law instances seen in training, but laws v1 kept 21
    # `seen_train` items; exclude them here unless `exclude_seen_train: false`. "Seen" is each
    # model's own training tiers (`models_registry.seen_lookup`: our `seen_train`, or Dan's
    # per-family sidecar for his models), applied per model below.
    exclude_seen = bool(cfg_get(cfg, "exclude_seen_train", True))
    n_seen_excluded = sum(1 for p in problems.values() if p.seen_train) if exclude_seen else 0
    all_problems = problems
    if exclude_seen:
        problems = {pid: p for pid, p in problems.items() if not p.seen_train}
    models = override_models or models_present(
        store, suite=suite, version=version, judge_id=judge_id
    )
    if not models:
        return empty_view(f"no runs found for suite {suite!r}")
    textbook_version = cast(
        str,
        cfg.get("textbook_law_version")
        or default_version(store, suite=textbook_suite, models=models, judge_id=judge_id)
        or "v1",
    )
    textbook_problems = problems_by_pid(store, textbook_suite, textbook_version)

    schemas = sorted(
        {str(v) for p in problems.values() if (v := p.tags.get(SCHEMA_TAG)) is not None}
    )
    if not schemas:
        return empty_view(f"suite {suite!r} has no {SCHEMA_TAG!r} tag on any item")
    registry = models_registry.load_registry()
    unseen_by_model: dict[str, Mapping[str, Any]] = {}
    for model in models:
        if not exclude_seen:
            unseen_by_model[model] = all_problems
            continue
        seen = models_registry.seen_lookup(
            store, suite, version, model, all_problems, registry=registry
        )
        unseen_by_model[model] = {
            pid: p for pid, p in all_problems.items() if not seen.get(pid, p.seen_train)
        }

    out: dict[str, Any] = {}
    for schema in schemas:
        per_model: dict[str, Any] = {}
        for model in models:
            model_problems = unseen_by_model[model]
            pids = {
                pid for pid, p in model_problems.items() if str(p.tags.get(SCHEMA_TAG)) == schema
            }
            run_id = latest_run_id(
                store, suite=suite, model=model, version=version, judge_id=judge_id
            )
            if run_id is None:
                continue
            items = items_by_pid(store, run_id)
            rows: list[tuple[JSON, float, int, int]] = []  # (s, item tokens, n, c)
            for pid in pids:
                item = items.get(pid)
                if item is None:
                    continue
                s_val = all_problems[pid].tags.get(S_TAG)
                tok = item_tokens(all_problems[pid])
                rows.append((s_val, tok if tok is not None else math.nan, item.n, item.c))
            if not rows:
                continue

            by_s: dict[JSON, list[tuple[int, int]]] = {}
            for s_val, _tok, n, c in rows:
                by_s.setdefault(s_val, []).append((n, c))
            curve_s = [
                {
                    "s": s_val,
                    "pass1": _mean_or_nan([pass_at_k(n, c, 1) for n, c in group]),
                    "n_items": len(group),
                }
                for s_val, group in sorted(by_s.items(), key=_s_sort_key)
            ]
            points_tokens = [
                {"tokens": tok, "pass1": pass_at_k(n, c, 1), "s": s_val}
                for s_val, tok, n, c in rows
                if not math.isnan(tok)
            ]
            slope = logit_slope_safe(
                [tok / 100.0 for _, tok, _, _ in rows],
                [n for _, _, n, _ in rows],
                [c for _, _, _, c in rows],
            )
            controlled_overall = _mean_or_nan([pass_at_k(n, c, 1) for _, _, n, c in rows])

            comparison = None
            if textbook_problems:
                tb_pids = {
                    p2
                    for p2, pr in textbook_problems.items()
                    if str(pr.tags.get(SCHEMA_TAG)) == schema
                }
                if tb_pids:
                    tb_run = latest_run_id(
                        store,
                        suite=textbook_suite,
                        model=model,
                        version=textbook_version,
                        judge_id=judge_id,
                    )
                    if tb_run is not None:
                        tb_items = items_by_pid(store, tb_run)
                        tb_matched = [tb_items[p2] for p2 in tb_pids if p2 in tb_items]
                        if tb_matched:
                            comparison = {
                                "n_items": len(tb_matched),
                                "pass1": _mean_or_nan(
                                    [pass_at_k(it.n, it.c, 1) for it in tb_matched]
                                ),
                            }

            per_model[model] = {
                "n_items": len(rows),
                "n_seen_excluded": len(all_problems) - len(model_problems),
                "curve_vs_s": curve_s,
                "points_vs_tokens": points_tokens,
                "logit_slope_per_100_tokens": slope,
                "controlled_pass1_overall": controlled_overall,
                "textbook_law_comparison": comparison,
            }
        if per_model:
            out[schema] = per_model

    if not out:
        return empty_view(f"suite {suite!r} has runs but no items matched any schema")
    return {
        "empty": False,
        "meta": {
            "suite": suite,
            "textbook_law_suite": textbook_suite,
            "version": version,
            "textbook_law_version": textbook_version,
            "models": models,
            "n_seen_train_excluded": n_seen_excluded,
            "judge_id": judge_id,
        },
        "schemas": out,
    }
