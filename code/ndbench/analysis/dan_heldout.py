"""Held-out greedy accuracy for Dan's best-state checkpoints (`DANSTATE_PLAN.md` §3 "B8"/V5,
WP-D5 item 2): the `dan_heldout` suite (`suites/danstate.py:dan_heldout`, Dan's own 5,000-item
`heldout.jsonl`, sampled greedy at k=1/T=0 -- `DanStateAdapter` default; `models.yaml`'s run
config) is as close to a single scalar per model as this package's benchmarks get, so this is a
from-scratch minimal pass@1 aggregator rather than a generalisation of an existing view.

"Accuracy" here is "solved on its one greedy draw" (`ItemResult.c > 0`), with a Clopper-Pearson CI
over items -- plan V5's own numbers ("T1 s0 at least 4,940/5,000") are exactly this statistic.
`pooled_pass1` (`sum(c) / sum(n)`) is reported alongside in case a run ever carries more than one
draw per item (e.g. a rerun at k>1); the two agree whenever every item has `n=1`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ..core.types import JSON
from ..metrics import clopper_pearson
from ..store import Store
from ._common import (
    DEFAULT_JUDGE_ID,
    cfg_get,
    default_version,
    empty_view,
    items_by_pid,
    latest_run_id,
    models_present,
    problems_by_pid,
    safe_div,
)

DEFAULT_SUITE = "dan_heldout"


def view(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    suite = cast(str, cfg_get(cfg, "suite", DEFAULT_SUITE))
    override_models = cast("list[str] | None", cfg.get("models"))
    # `judge_id` (plan DANSTATE_PLAN.md §2 WP-D2 item 5 / WP-D5 item 1, default unchanged): see
    # `analysis/yue.py:view`'s identical parameter. Dan's own gate (V5) judges this suite "lean_v2"
    # (Lean on the literal sampled text); a caller wanting cross-judge agreement instead reads
    # `analysis/judges.py`.
    judge_id = cast("str | None", cfg.get("judge_id", DEFAULT_JUDGE_ID))
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

    per_model: dict[str, Any] = {}
    for model in models:
        run_id = latest_run_id(store, suite=suite, model=model, version=version, judge_id=judge_id)
        if run_id is None:
            continue
        items = items_by_pid(store, run_id)
        items = {pid: it for pid, it in items.items() if pid in problems}
        if not items:
            continue
        n_sum = sum(it.n for it in items.values())
        c_sum = sum(it.c for it in items.values())
        solved = sum(1 for it in items.values() if it.c > 0)
        lo, hi = clopper_pearson(solved, len(items))
        per_model[model] = {
            "n_items": len(items),
            "n_draws": n_sum,
            "n_solved": solved,
            "accuracy": safe_div(float(solved), float(len(items))),
            "ci": (lo, hi),
            "pooled_pass1": safe_div(float(c_sum), float(n_sum)),
        }

    if not per_model:
        return empty_view(f"suite {suite!r} has runs but no items scored")
    return {
        "empty": False,
        "meta": {"suite": suite, "version": version, "models": models, "judge_id": judge_id},
        "models": per_model,
    }
