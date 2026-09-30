"""B2: restricted-class family horizons -- the dashboard's "families" page.

Named `horizon.py` (not `families.py`) because it is built on `metrics.horizon`, per
`analysis/__init__.py`'s docstring: plan §1 names this file for the statistic, and `dash/build.py`
maps the `"families"` page to `analysis.horizon:view`.

Reads the `families` suite (WP3's `suites/families.py`; `Problem.tags` carry `family`/`n`, plan §3's
WP3 row) and, per `(family, model)`:

- mean pass@1/pass@8/pass@32 against `n` (plain curves, `metrics.pass_at_k`, no bootstrap -- for
  plotting, not for the horizon itself);
- `n*_{k,tau}` with its bootstrap CI (`metrics.horizon`) at the plan's default (k=8, tau=0.5) and
  its two stated sensitivities (k=1, tau=0.5; k=32, tau -> 0, i.e. "pass@32 > 0" using a tiny
  positive threshold since `horizon` compares a continuous mean against `tau`);
- a logit slope per added reference line and per 100 written tokens (`metrics.logit_slope`,
  item-level `x`/`n`/`c`; `x` = `Problem.length`'s point estimate (`_common.item_length`) or
  `ItemResult.tokens_mean / 100`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from ..core.types import JSON, ItemResult
from ..metrics import horizon, pass_at_k
from ..store import Store
from ._common import (
    cfg_get,
    empty_view,
    item_length,
    items_by_pid,
    latest_run_id,
    logit_slope_safe,
    models_present,
    problems_by_pid,
)

DEFAULT_SUITE = "families"
FAMILY_TAG = "family"
N_TAG = "n"
#: pass@k curves plotted against `n` (plan §2 B2: "pass@1 and pass@k vs n").
KS: tuple[int, ...] = (1, 8, 32)


def _mean_curve(by_n: Mapping[int, Sequence[ItemResult]], k: int) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for n_val in sorted(by_n):
        vals = [v for r in by_n[n_val] if not math.isnan(v := pass_at_k(r.n, r.c, k))]
        out.append((n_val, float(np.mean(vals)) if vals else float("nan")))
    return out


def view(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    suite = cast(str, cfg_get(cfg, "suite", DEFAULT_SUITE))
    version = cast(str, cfg_get(cfg, "version", "v1"))
    ks = cast("list[int]", cfg_get(cfg, "ks", list(KS)))
    B = int(cast(int, cfg_get(cfg, "B", 2000)))
    seed = int(cast(int, cfg_get(cfg, "seed", 0)))
    override_models = cast("list[str] | None", cfg.get("models"))

    problems = problems_by_pid(store, suite, version)
    if not problems:
        return empty_view(f"suite {suite!r} version {version!r} not found in the store")
    models = override_models or models_present(store, suite=suite, version=version)
    if not models:
        return empty_view(f"no runs found for suite {suite!r}")

    families = sorted(
        {str(v) for p in problems.values() if (v := p.tags.get(FAMILY_TAG)) is not None}
    )
    if not families:
        return empty_view(f"suite {suite!r} has no {FAMILY_TAG!r} tag on any item")

    out_families: dict[str, Any] = {}
    for family in families:
        fam_pids = {pid for pid, p in problems.items() if str(p.tags.get(FAMILY_TAG)) == family}
        per_model: dict[str, Any] = {}
        for model in models:
            run_id = latest_run_id(store, suite=suite, model=model, version=version)
            if run_id is None:
                continue
            items = items_by_pid(store, run_id)
            fam_items = {pid: items[pid] for pid in fam_pids if pid in items}
            if not fam_items:
                continue
            by_n: dict[int, list[ItemResult]] = {}
            for pid, item in fam_items.items():
                n_val = problems[pid].tags.get(N_TAG)
                if n_val is None:
                    continue
                by_n.setdefault(int(cast(int, n_val)), []).append(item)
            if not by_n:
                continue
            n_star = {
                "k8_tau0.5": horizon(by_n, k=8, tau=0.5, B=B, seed=seed),
                "k1_tau0.5": horizon(by_n, k=1, tau=0.5, B=B, seed=seed),
                "k32_tau_pos": horizon(by_n, k=32, tau=1e-9, B=B, seed=seed),
            }
            curves_by_k = {str(k): _mean_curve(by_n, k) for k in ks}
            xs_l: list[float] = []
            xs_tok: list[float] = []
            ns_arr: list[int] = []
            cs_arr: list[int] = []
            for pid, item in fam_items.items():
                length = item_length(problems[pid])
                xs_l.append(length if length is not None else float("nan"))
                xs_tok.append(item.tokens_mean / 100.0)
                ns_arr.append(item.n)
                cs_arr.append(item.c)
            per_model[model] = {
                "n_items": len(fam_items),
                "n_star": n_star,
                "curves_by_k": curves_by_k,
                "logit_slope_per_line": logit_slope_safe(xs_l, ns_arr, cs_arr),
                "logit_slope_per_100_tokens": logit_slope_safe(xs_tok, ns_arr, cs_arr),
            }
        if per_model:
            out_families[family] = per_model

    if not out_families:
        return empty_view(f"suite {suite!r} has runs but no items matched any family/n")
    return {
        "empty": False,
        "meta": {"suite": suite, "version": version, "ks": list(ks), "models": models, "B": B},
        "families": out_families,
    }
