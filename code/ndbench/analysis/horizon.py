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
- a logit slope per added reference line and per 100 item tokens (`metrics.logit_slope`,
  item-level `x`/`n`/`c`; `x` = `Problem.length`'s point estimate (`_common.item_length`) or the
  item's own tokens / 100 (`_common.item_tokens`: prompt + reference proof, not the model's
  outcome-dependent generated length)).
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
    default_version,
    empty_view,
    item_length,
    item_tokens,
    items_by_pid,
    latest_run_id,
    logit_slope_safe,
    models_present,
    problems_by_pid,
)

DEFAULT_SUITE = "families"
FAMILY_TAG = "family"
N_TAG = "n"
#: `suites/families.py`'s "ILTP-style" family packs four structurally distinct sub-schemas
#: (`syj204_style`/`syj203_style`/`iff_chain_style`/`de_bruijn_style`) under one `family` tag,
#: distinguished only by `tags["kind"]`, with *overlapping* `n` ranges across kinds (e.g. the real
#: `families` suite has `n=1` containing `syj204_style`+`syj203_style`+`de_bruijn_style` items, and
#: `n=2`/`n=3` mixing `syj204_style`+`iff_chain_style`). Grouping `by_n` on `family` alone would
#: then pool pass@k across unrelated proof schemas at the same nominal `n` -- a materially
#: misleading curve/horizon statistic. `_group_key` folds `kind` into the grouping key whenever an
#: item has one (every family besides `ILTP-style` has no `kind` tag at all, so this is a no-op for
#: them), splitting "ILTP-style" into one row per actual sub-schema.
KIND_TAG = "kind"
#: pass@k curves plotted against `n` (plan §2 B2: "pass@1 and pass@k vs n").
KS: tuple[int, ...] = (1, 8, 32)


def _mean_curve(by_n: Mapping[int, Sequence[ItemResult]], k: int) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for n_val in sorted(by_n):
        vals = [v for r in by_n[n_val] if not math.isnan(v := pass_at_k(r.n, r.c, k))]
        out.append((n_val, float(np.mean(vals)) if vals else float("nan")))
    return out


def _group_key(family: str, kind: JSON) -> str:
    """The row key used for a `family`'s page entry: `family` itself, or `"family (kind)"` when
    the item also carries a `KIND_TAG` (module docstring's "ILTP-style" fix)."""
    return family if kind is None else f"{family} ({kind})"


def view(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    suite = cast(str, cfg_get(cfg, "suite", DEFAULT_SUITE))
    ks = cast("list[int]", cfg_get(cfg, "ks", list(KS)))
    B = int(cast(int, cfg_get(cfg, "B", 2000)))
    seed = int(cast(int, cfg_get(cfg, "seed", 0)))
    override_models = cast("list[str] | None", cfg.get("models"))
    # Pick whichever suite version has runs, rather than a hard-coded "v1" (fix list item 1,
    # applied uniformly across every view -- see `analysis/padding.py`'s own fix for the concrete
    # motivating case).
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

    # Group by (family, kind) rather than family alone (module docstring's `_group_key`/`KIND_TAG`
    # -- "ILTP-style" packs four incomparable sub-schemas with overlapping `n` ranges under one
    # `family` tag), so the usual case (no `kind` tag at all) just gets `family` back unchanged.
    pids_by_group: dict[str, set[str]] = {}
    for pid, p in problems.items():
        fam = p.tags.get(FAMILY_TAG)
        if fam is None:
            continue
        pids_by_group.setdefault(_group_key(str(fam), p.tags.get(KIND_TAG)), set()).add(pid)
    families = sorted(pids_by_group)
    if not families:
        return empty_view(f"suite {suite!r} has no {FAMILY_TAG!r} tag on any item")

    out_families: dict[str, Any] = {}
    for family in families:
        fam_pids = pids_by_group[family]
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
                tok = item_tokens(problems[pid])
                xs_tok.append(tok / 100.0 if tok is not None else float("nan"))
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
