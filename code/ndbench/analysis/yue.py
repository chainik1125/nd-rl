"""B8: the Yue et al. base-vs-RL pass@k replication (plan §2 B8, §3 WP6 row).

`view(store, cfg)` looks at one or more suites (`cfg["sets"]`, default `textbook_dev` /
`textbook_law` / `gen_ctrl`, matching WP2's `existing.py` builders) and, for every model with a
finished run against each suite, builds:

- pass@k for `k = 2**0 .. 2**12`, with a 95% item-level bootstrap band, per set and per model;
- the same curves split by problem length (`per_length`, `Problem.length`'s point estimate or
  bound midpoint -- see `_common.item_length`);
- the base-vs-RL crossover `k*` (the first `k` where one model's curve overtakes the other's) with
  its bootstrap distribution;
- coverage ("ever solved", i.e. `c > 0`) at each model's own `k_max` (its total sample count) with
  a Clopper-Pearson CI;
- the solved-set Venn between the two models picked for the crossover (`only_a`/`only_b`/`both`/
  `neither`, plus whether the second model's solved set is a subset of the first's);
- legacy-run consistency: agreement with any `status == "legacy"` run for the same suite/model
  (plan §0's existing k=256, max_new=512 textbook counts), compared at each item's first-chunk
  `pass@n_first`.

**`k <= 256` vs `k > n_i` (plan §2 B8).** `k <= 256` always uses each item's *first chunk only*
(`_common.first_chunk_by_pid`): a fixed `n` across every item, so `metrics.pass_at_k` is the
unbiased Chen et al. estimator and (via `_legacy_consistency`) directly comparable to the legacy
single-shot `n=256` counts. Later chunks are only drawn (`AdaptiveRule`) for items that had not yet
reached `stop_c` successes, so folding them into a `k <= 256` estimate would silently compare a
different effective `n` across items. For `k` above an item's own total sample count `n_tot`, the
exact combinatorial estimator is undefined (`pass_at_k` returns `nan` for `k > n`), so the curve
instead uses the paper's extrapolation `1 - (1 - c/n_tot)**k`; `lower_curves` gives the same
extrapolation with `c/n_tot` replaced by its Clopper-Pearson lower limit, a conservative variant
for exactly this regime ("bounded with the Clopper-Pearson lower limit").
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from ..core.types import JSON, Problem
from ..metrics import clopper_pearson
from ..store import Store
from ._common import (
    all_run_ids,
    cfg_get,
    default_version,
    empty_view,
    first_chunk_by_pid,
    is_nan,
    item_length,
    items_by_pid,
    latest_run_id,
    merged_items_by_pid,
    models_present,
    nanmean_ignore,
    nanpercentile_ignore,
    problems_by_pid,
    safe_div,
)

#: k = 2^0 .. 2^12, the full grid every set's curve is *computed* on (so `_crossover`/`_venn`, which
#: compare two sets' models at the same `k`, never need to intersect two different grids).
KS: tuple[int, ...] = tuple(2**i for i in range(13))
DEFAULT_B = 2000
#: label -> suite name, matching WP2's `existing.py` builders (plan §1's suites table).
DEFAULT_SETS: dict[str, str] = {
    "textbook_dev": "textbook_dev",
    "textbook_law": "textbook_law",
    "control": "gen_ctrl",
}
#: label -> the k this set's own draw protocol actually targets (plan §2 B8: "k_max 4096
#: (textbook_dev) / 1024 (textbook-law)"; the control set draws a fixed, non-adaptive k=512 and was
#: never meant to be extrapolated past it). `view` plots/crosses-over/Venns each set only up to its
#: own entry here (default `max(KS)` for an unlisted label), rather than one blanket 2^12 grid for
#: every set regardless of its real sampling budget -- the previous behaviour both overstated how
#: far past-sample extrapolation should be trusted (e.g. `gen_ctrl`'s 60 control items were drawn at
#: a fixed k=512 specifically so no extrapolation is needed) and surfaced the two-estimator
#: discontinuity at `k == n_tot` (`_pk_vals`'s module docstring) well past the point any reader
#: should be looking at that set's curve at all.
DEFAULT_K_MAX_BY_SET: dict[str, int] = {
    "textbook_dev": 4096,
    "textbook_law": 1024,
    "control": 512,
}
#: This benchmark's three named models (plan §0/configs/models.yaml), the default candidate list
#: when `cfg["models"]` doesn't override it (WP8 integration fix list item 3). Without this filter,
#: `models_present` would also surface every `steer:<layer>:<alpha>`/`interp:<alpha>` run that
#: happens to share a suite with the Yue sets (`textbook_dev` doubles as one of
#: `INTERP_SPEC.md`'s steering/interpolation eval suites) -- real runs, but not what this specific
#: base-vs-RL(-vs-armC) replication page is for; those sweeps get their own `interp_steering`/
#: `interp_weights` pages. A set missing one of the three (e.g. `textbook_law`/`gen_ctrl` have no
#: `armC` run) simply shows whichever of the three it does have.
DEFAULT_MODELS: tuple[str, ...] = ("base", "rl", "armC")
#: label -> the pre-ndbench `import_legacy` suite holding a `status="legacy"` run to sanity-check
#: against (plan §2 B8's "consistency vs the legacy k=256 counts (max_new 512)"). Only
#: `textbook_dev` has one on the real store: the existing 58-item k=256 textbook run
#: (`suites/existing.py`'s `_import_monitor_run` names it `legacy_textbook`, not
#: `legacy_textbook_dev` -- its summary json's own `"set"` field, not this suite's name, drives
#: that). `textbook_law`/`gen_ctrl` are new constructions built for this project with no
#: pre-existing legacy equivalent, so they simply have no entry here (`_legacy_consistency` then
#: finds no matching run and the page shows no legacy-consistency row for them, not a crash).
DEFAULT_LEGACY_SETS: dict[str, str] = {"textbook_dev": "legacy_textbook"}
#: This dashboard's model label -> the legacy run's own label (`suites/existing.py`'s
#: `_LEGACY_MONITOR_MODELS`); the reverse direction of `skills.py`'s `LEGACY_MODEL_MAP` (fix list
#: item 2's "Legacy model labels differ"). `armC` has no legacy-textbook equivalent at all (the
#: pre-ndbench textbook run only ever covered base/champion), so it is intentionally absent --
#: looked up with `.get(m, m)` below, which would try a (nonexistent) literal `"armC"` legacy run
#: and correctly find none.
LEGACY_MODEL_LABEL: dict[str, str] = {"base": "base", "rl": "champion"}
#: rows: (n_first, c_first, n_tot, c_tot) per pid.
_Rows = dict[str, tuple[int, int, int, int]]


def _vec_pass_at_k(n: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    """Elementwise `metrics.pass_at_k(n[i], c[i], k)`, callers must ensure `n[i] >= k` everywhere
    (checked by the caller's mask, not re-checked here -- this is an internal, vectorised twin of
    `metrics.pass_at_k`, kept in sync with it rather than reimplementing its semantics)."""
    n = n.astype(float)
    c = c.astype(float)
    fail = n - c
    prod = np.ones_like(n)
    for i in range(k):
        prod = prod * np.clip(fail - i, 0.0, None) / (n - i)
    out = 1.0 - prod
    return np.where(fail < k, 1.0, out)


def _pk_vals(
    n_first: np.ndarray, c_first: np.ndarray, n_tot: np.ndarray, c_tot: np.ndarray, k: int
) -> np.ndarray:
    out = np.full(n_first.shape, np.nan)
    if k <= 256:
        mask = n_first >= k
        if np.any(mask):
            out[mask] = _vec_pass_at_k(n_first[mask], c_first[mask], k)
        return out
    mask = n_tot >= k
    if np.any(mask):
        out[mask] = _vec_pass_at_k(n_tot[mask], c_tot[mask], k)
    mask2 = (~mask) & (n_tot > 0)
    if np.any(mask2):
        p = c_tot[mask2] / n_tot[mask2]
        out[mask2] = 1.0 - (1.0 - p) ** k
    return out


def _pk_lower_vals(
    n_first: np.ndarray, c_first: np.ndarray, n_tot: np.ndarray, c_tot: np.ndarray, k: int
) -> np.ndarray:
    out = np.full(n_first.shape, np.nan)
    if k <= 256:
        mask = n_first >= k
        if np.any(mask):
            out[mask] = _vec_pass_at_k(n_first[mask], c_first[mask], k)
        return out
    mask = n_tot >= k
    if np.any(mask):
        out[mask] = _vec_pass_at_k(n_tot[mask], c_tot[mask], k)
    mask2 = (~mask) & (n_tot > 0)
    for i in np.where(mask2)[0]:
        lo, _hi = clopper_pearson(int(c_tot[i]), int(n_tot[i]))
        out[i] = 1.0 - (1.0 - lo) ** k
    return out


def _arrays(
    rows: _Rows, pids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_first = np.array([rows[p][0] for p in pids], dtype=float)
    c_first = np.array([rows[p][1] for p in pids], dtype=float)
    n_tot = np.array([rows[p][2] for p in pids], dtype=float)
    c_tot = np.array([rows[p][3] for p in pids], dtype=float)
    return n_first, c_first, n_tot, c_tot


def _model_k_max(rows: _Rows) -> int:
    """The largest per-item draw count in a model's run: the k beyond which its pass@k is no
    longer estimated from draws (adaptive runs stop solved items early, so the max -- not the min
    -- is the design's ceiling; an item stopped early has `c >= stop_c` and `1-(1-p)^k ~ 1`)."""
    return max((n_tot for _nf, _cf, n_tot, _ct in rows.values()), default=0)


def _curve_for_model(
    rows: _Rows, ks: Sequence[int], B: int, seed: int
) -> tuple[list[tuple[int, float, float, float]], list[tuple[int, float]]]:
    pids = sorted(rows)
    n_items = len(pids)
    if n_items == 0:
        nan = float("nan")
        return [(k, nan, nan, nan) for k in ks], [(k, nan) for k in ks]
    n_first, c_first, n_tot, c_tot = _arrays(rows, pids)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n_items, size=(B, n_items))
    curve: list[tuple[int, float, float, float]] = []
    lower: list[tuple[int, float]] = []
    for k in ks:
        vals = _pk_vals(n_first, c_first, n_tot, c_tot, k)
        mean = float(nanmean_ignore(vals))
        boot_means = nanmean_ignore(vals[boot_idx], axis=1)
        lo, hi = (float(x) for x in nanpercentile_ignore(boot_means, [2.5, 97.5]))
        curve.append((k, mean, lo, hi))
        lvals = _pk_lower_vals(n_first, c_first, n_tot, c_tot, k)
        lower.append((k, float(nanmean_ignore(lvals))))
    return curve, lower


def _find_crossover(
    ks: Sequence[int], means_a: Sequence[float], means_b: Sequence[float]
) -> int | None:
    """The first `k` (after the first non-tied comparison) where `sign(a - b)` flips, or `None` if
    the curves never cross (or too few points are non-`nan` to tell)."""
    sign0 = 0
    for k, a, b in zip(ks, means_a, means_b, strict=True):
        d = a - b
        if math.isnan(d) or d == 0:
            continue
        s = 1 if d > 0 else -1
        if sign0 == 0:
            sign0 = s
            continue
        if s != sign0:
            return k
    return None


def _crossover(
    rows_a: _Rows, rows_b: _Rows, ks: Sequence[int], B: int, seed: int
) -> dict[str, Any]:
    common = sorted(set(rows_a) & set(rows_b))
    if not common:
        return {
            "k_star": None,
            "ci": (None, None),
            "n_common": 0,
            "n_boot_found": 0,
            "p_none": float("nan"),
            "B": B,
        }
    an_first, ac_first, an_tot, ac_tot = _arrays(rows_a, common)
    bn_first, bc_first, bn_tot, bc_tot = _arrays(rows_b, common)
    vals_a = np.stack([_pk_vals(an_first, ac_first, an_tot, ac_tot, k) for k in ks])
    vals_b = np.stack([_pk_vals(bn_first, bc_first, bn_tot, bc_tot, k) for k in ks])
    means_a = nanmean_ignore(vals_a, axis=1)
    means_b = nanmean_ignore(vals_b, axis=1)
    k_star = _find_crossover(ks, means_a.tolist(), means_b.tolist())
    n_items = len(common)
    rng = np.random.default_rng(seed + 1)
    boot_idx = rng.integers(0, n_items, size=(B, n_items))
    boots: list[int] = []
    for b_i in range(B):
        idx = boot_idx[b_i]
        ma = nanmean_ignore(vals_a[:, idx], axis=1).tolist()
        mb = nanmean_ignore(vals_b[:, idx], axis=1).tolist()
        kb = _find_crossover(ks, ma, mb)
        if kb is not None:
            boots.append(kb)
    ci: tuple[int | None, int | None] = (None, None)
    # Audit 2026-10-01: a CI over only the replicates that happened to find a crossover is not a
    # CI for "no crossover"; report it only when the point estimate exists (`p_none` = share of
    # replicates with no crossover, either way).
    if boots and k_star is not None:
        ci = (int(np.percentile(boots, 2.5)), int(np.percentile(boots, 97.5)))
    return {
        "k_star": k_star,
        "ci": ci,
        "n_common": n_items,
        "n_boot_found": len(boots),
        "p_none": 1.0 - len(boots) / B if B else float("nan"),
        "B": B,
    }


def _venn(rows_a: _Rows, rows_b: _Rows, label_a: str, label_b: str) -> dict[str, Any]:
    common = sorted(set(rows_a) & set(rows_b))
    only_a = only_b = both = neither = 0
    for pid in common:
        sa = rows_a[pid][3] > 0
        sb = rows_b[pid][3] > 0
        if sa and sb:
            both += 1
        elif sa:
            only_a += 1
        elif sb:
            only_b += 1
        else:
            neither += 1
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n": len(common),
        "only_a": only_a,
        "only_b": only_b,
        "both": both,
        "neither": neither,
        "b_subset_of_a": only_b == 0 and len(common) > 0,
    }


def _coverage(rows: _Rows) -> dict[str, Any]:
    n_items = len(rows)
    solved = sum(1 for r in rows.values() if r[3] > 0)
    lo, hi = clopper_pearson(solved, n_items) if n_items else (float("nan"), float("nan"))
    k_max = max((r[2] for r in rows.values()), default=0)
    return {
        "n_items": n_items,
        "solved": solved,
        "coverage": safe_div(float(solved), float(n_items)),
        "ci": (lo, hi),
        "k_max": k_max,
    }


def _per_length(
    rows: _Rows, problems: Mapping[str, Problem], ks: Sequence[int]
) -> dict[str, list[tuple[int, float]]]:
    if not problems:
        return {}
    buckets: dict[float, list[str]] = {}
    for pid in rows:
        p = problems.get(pid)
        length = item_length(p) if p is not None else None
        if length is None:
            continue
        buckets.setdefault(length, []).append(pid)
    out: dict[str, list[tuple[int, float]]] = {}
    for length in sorted(buckets):
        pids = buckets[length]
        n_first, c_first, n_tot, c_tot = _arrays(rows, pids)
        curve = [
            (k, float(nanmean_ignore(_pk_vals(n_first, c_first, n_tot, c_tot, k)))) for k in ks
        ]
        out[str(length)] = curve
    return out


def _legacy_consistency(
    store: Store, legacy_suite: str, legacy_model: str, rows: _Rows
) -> dict[str, Any] | None:
    """`rows` (this run's own `(n_first, c_first, n_tot, c_tot)` per pid) vs. `import_legacy`'s
    pre-ndbench `status="legacy"` run for the same underlying items, if one exists.

    `legacy_suite`/`legacy_model` are the *legacy* run's own suite/model names, not this
    benchmark's -- `import_legacy` never writes into a suite named `"textbook_dev"` (the real
    pre-existing k=256 textbook comparison lives under suite `"legacy_textbook"`, model labels
    `"base"`/`"champion"`, per `suites/existing.py`'s `_import_monitor_run`), and a legacy run's
    own `version` field is a sampling-config id (e.g. `"k256_s0"`), not this suite's `"v1"`/`"v2"`
    -- so the lookup is unfiltered on version (`version=None`), by suite+model only.
    """
    # `judge_id=None`: a legacy `import_legacy` run (`judge_id="legacy_import"`), not
    # `all_run_ids`'s new default `"nd_v2"` filter (plan §2 WP-D2 item 5).
    legacy_ids = all_run_ids(
        store,
        suite=legacy_suite,
        model=legacy_model,
        version=None,
        statuses=("legacy",),
        judge_id=None,
    )
    if not legacy_ids:
        return None
    legacy_items = merged_items_by_pid(store, legacy_ids)
    common = sorted(set(rows) & set(legacy_items))
    if not common:
        return {"n_common": 0, "agree_rate": float("nan"), "mean_abs_diff": float("nan")}
    diffs: list[float] = []
    agree = 0
    for pid in common:
        n_first, c_first, _n_tot, _c_tot = rows[pid]
        cur_p = safe_div(float(c_first), float(n_first))
        leg = legacy_items[pid]
        leg_p = safe_div(float(leg.c), float(leg.n))
        if not is_nan(cur_p) and not is_nan(leg_p):
            diffs.append(abs(cur_p - leg_p))
        if (c_first > 0) == (leg.c > 0):
            agree += 1
    return {
        "n_common": len(common),
        "agree_rate": safe_div(float(agree), float(len(common))),
        "mean_abs_diff": float(np.mean(diffs)) if diffs else float("nan"),
    }


def _pick_pair(models: list[str], cfg: Mapping[str, JSON]) -> tuple[str, str] | None:
    base = cast("str | None", cfg.get("base_model"))
    rl = cast("str | None", cfg.get("rl_model"))
    if base and rl and base in models and rl in models:
        return base, rl
    if "base" in models and "rl" in models:
        return "base", "rl"
    if len(models) >= 2:
        return models[0], models[1]
    return None


def view(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    override_version = cast("str | None", cfg.get("version"))
    sets = cast("dict[str, str]", cfg_get(cfg, "sets", dict(DEFAULT_SETS)))
    ks = cast("list[int]", cfg_get(cfg, "ks", list(KS)))
    k_max_by_set = cast("dict[str, int]", cfg_get(cfg, "k_max_by_set", dict(DEFAULT_K_MAX_BY_SET)))
    seed = int(cast(int, cfg_get(cfg, "seed", 0)))
    B = int(cast(int, cfg_get(cfg, "B", DEFAULT_B)))
    override_models = cast("list[str] | None", cfg.get("models"))
    candidate_models = override_models or list(DEFAULT_MODELS)
    legacy_sets = cast("dict[str, str]", cfg_get(cfg, "legacy_sets", dict(DEFAULT_LEGACY_SETS)))
    legacy_model_label = cast(
        "dict[str, str]", cfg_get(cfg, "legacy_model_label", dict(LEGACY_MODEL_LABEL))
    )

    per_set: dict[str, Any] = {}
    legacy_by_set: dict[str, Any] = {}
    for label, suite in sets.items():
        # Each labelled set is its own suite (plan §1) and so can land on its own suite version
        # (fix list item 1): resolved independently per set rather than from one global cfg
        # default, since e.g. textbook_dev/textbook_law/gen_ctrl need not all share one version.
        version = (
            override_version or default_version(store, suite=suite, models=candidate_models) or "v1"
        )
        # This set's own k grid, capped at its own draw protocol's k_max (module docstring's
        # `DEFAULT_K_MAX_BY_SET`) rather than the full 2^12 grid every set would otherwise share.
        k_cap = int(k_max_by_set.get(label, max(ks)))
        ks_for_set = [k for k in ks if k <= k_cap]
        models = override_models or [
            m for m in candidate_models if m in models_present(store, suite=suite, version=version)
        ]
        if not models:
            continue
        problems = problems_by_pid(store, suite, version)
        rows_by_model: dict[str, _Rows] = {}
        k_max_by_model: dict[str, int] = {}
        curves: dict[str, Any] = {}
        lower_curves: dict[str, Any] = {}
        coverage: dict[str, Any] = {}
        for model in models:
            run_id = latest_run_id(store, suite=suite, model=model, version=version)
            if run_id is None:
                continue
            fc = first_chunk_by_pid(store, run_id)
            tot = items_by_pid(store, run_id)
            rows: _Rows = {}
            for pid, item in tot.items():
                f = fc.get(pid)
                n_first, c_first = (f["n"], f["c"]) if f is not None else (item.n, item.c)
                rows[pid] = (n_first, c_first, item.n, item.c)
            if not rows:
                continue
            rows_by_model[model] = rows
            # Cap each model at its own draw count (audit 2026-10-01): past the largest per-item
            # n, `1 - (1 - c/n)^k` can never credit an item the model did not solve in the draws
            # it had, so e.g. a 1024-draw run's "pass@4096" is an extrapolation, not a measurement.
            k_max_by_model[model] = _model_k_max(rows)
            ks_model = [k for k in ks_for_set if k <= k_max_by_model[model]]
            curve, lower = _curve_for_model(rows, ks_model, B, seed)
            curves[model] = curve
            lower_curves[model] = lower
            coverage[model] = _coverage(rows)
        if not rows_by_model:
            continue
        pair = _pick_pair(sorted(rows_by_model), cfg)
        cross: dict[str, Any] | None = None
        venn: dict[str, Any] | None = None
        if pair is not None:
            a, b = pair
            k_common = min(k_max_by_model[a], k_max_by_model[b])
            ks_pair = [k for k in ks_for_set if k <= k_common]
            cross = _crossover(rows_by_model[a], rows_by_model[b], ks_pair, B, seed)
            venn = _venn(rows_by_model[a], rows_by_model[b], a, b)
        per_length = {
            m: _per_length(
                rows_by_model[m], problems, [k for k in ks_for_set if k <= k_max_by_model[m]]
            )
            for m in rows_by_model
        }
        legacy: dict[str, Any] = {}
        legacy_suite = legacy_sets.get(label)
        if legacy_suite is not None:
            for m in rows_by_model:
                legacy_model = legacy_model_label.get(m, m)
                lc = _legacy_consistency(store, legacy_suite, legacy_model, rows_by_model[m])
                if lc is not None:
                    legacy[m] = lc
        per_set[label] = {
            "suite": suite,
            "version": version,
            "k_max": k_cap,
            "k_max_by_model": k_max_by_model,
            "models": sorted(rows_by_model),
            "n_items": {m: len(r) for m, r in rows_by_model.items()},
            "curves": curves,
            "lower_curves": lower_curves,
            "coverage": coverage,
            "pair": list(pair) if pair else None,
            "crossover": cross,
            "venn": venn,
            "per_length": per_length,
        }
        if legacy:
            legacy_by_set[label] = legacy

    if not per_set:
        return empty_view(
            f"no yue runs found among models {list(candidate_models)} "
            f"for sets {list(sets.values())}"
        )
    return {
        "empty": False,
        # No single top-level "version": each set is its own suite and resolves its own version
        # independently (see the loop above) -- read `data["sets"][label]["version"]` instead.
        "meta": {"ks": list(ks), "B": B, "seed": seed, "models": candidate_models, "sets": sets},
        "sets": per_set,
        "legacy_consistency": legacy_by_set,
    }
