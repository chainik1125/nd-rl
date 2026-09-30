"""Analysis views for the interp package's four measures (`INTERP_SPEC.md`'s "Storage, views,
dashboard, CLI" section): `view_likelihood`/`view_probes` read this package's own
`out/store/interp/{likelihood,probes}/<run_id>.jsonl` artefacts (`interp/_store.py`);
`view_steering`/`view_interpolation` read ordinary `Store` runs (`interp/steering.py`/
`interp/interpolate.py` write these through `runner.run` like any other benchmark), matched by
their `steer:<layer>:<alpha>` / `interp:<alpha>` (+ `base`/`rl`) model-label convention.

Follows the `view_<bench>(store, cfg) -> dict[str, JSON]` contract every other `analysis/*` module
uses (`analysis/_common.py`'s docstring): `cfg` keys are all optional with a workable default, and
`empty_view(...)` is returned whenever nothing matches, so `dash/interp_pages.py` can render the
"no data yet" panel uniformly.

Every row this module reads back (`load_interp_rows`, or an `ItemResult`'s plain fields) comes in
as `JSON` (a `str|int|float|bool|None|list|dict` union) or is itself untyped `dict[str, JSON]`, so
every nested lookup is `cast` to the shape this module itself wrote (`likelihood.run_likelihood`/
`probes.run_probes`'s row schemas) -- the same idiom `analysis/_common.py`/`horizon.py` use
throughout for the same reason.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from ..core.types import JSON, ItemResult
from ..interp import eval_suites
from ..interp._store import latest_interp_run, load_interp_rows
from ..interp.probes import LAYER_KEYS
from ..metrics import pass_curve
from ..store import Store
from ._common import cfg_get, empty_view, items_by_pid, latest_run_id, models_present

_STEER_RE = re.compile(r"^steer:(-?\d+):(-?[0-9.]+)$")
_INTERP_RE = re.compile(r"^interp:(-?[0-9.]+)$")

#: pass@k values plotted for the interpolation k=128 curve (`INTERP_SPEC.md` §3).
INTERP_KS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)


def _d(x: JSON) -> dict[str, JSON]:
    """`x`, already known (by construction, not by this function) to be a JSON object -- a short
    alias for the `cast("dict[str, JSON]", x)` idiom repeated throughout this module."""
    return cast("dict[str, JSON]", x)


def _f(x: JSON) -> float:
    return cast(float, x)


def view_likelihood(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    run_id = cast("str | None", cfg.get("run_id")) or latest_interp_run(store, "likelihood")
    if run_id is None:
        return empty_view("no `interp likelihood` run in the store yet")
    rows = load_interp_rows(store, "likelihood", run_id)
    if not rows:
        return empty_view(f"likelihood run {run_id!r} has no rows")

    by_origin: dict[str, list[dict[str, JSON]]] = {}
    for r in rows:
        by_origin.setdefault(cast(str, r["origin"]), []).append(r)

    origins: dict[str, Any] = {}
    for origin, rs in by_origin.items():
        base_pts: list[tuple[float, float]] = []
        rl_pts: list[tuple[float, float]] = []
        base_vs_rl: list[tuple[float, float]] = []
        for r in rs:
            base, rl = _d(r["base"]), _d(r["rl"])
            if base["logp_marginal"] is not None:
                base_pts.append((_f(base["logp_single"]), _f(base["logp_marginal"])))
            if rl["logp_marginal"] is not None:
                rl_pts.append((_f(rl["logp_single"]), _f(rl["logp_marginal"])))
            if base["logp_marginal"] is not None and rl["logp_marginal"] is not None:
                base_vs_rl.append((_f(base["logp_marginal"]), _f(rl["logp_marginal"])))
        origins[origin] = {
            "n": len(rs),
            "single_vs_marginal_base": base_pts,
            "single_vs_marginal_rl": rl_pts,
            "base_vs_rl_marginal": base_vs_rl,
            "marginal_hist_base": [p[1] for p in base_pts],
            "marginal_hist_rl": [p[1] for p in rl_pts],
        }

    crux_set: list[dict[str, JSON]] = []
    for r in rows:
        p_hat = r.get("p_hat_base")
        rl = _d(r["rl"])
        if p_hat is not None and _f(p_hat) < 0.5 and rl["logp_marginal"] is not None:
            crux_set.append(
                {
                    "pid": r["pid"],
                    "p_hat_base": p_hat,
                    "base_logp_marginal": _d(r["base"])["logp_marginal"],
                    "rl_logp_marginal": rl["logp_marginal"],
                }
            )
    disagreements = [_d(r["disagreement"]) for r in rows if r.get("disagreement") is not None]
    mean_disagree_frac = (
        float(np.mean([_f(d["disagree_frac"]) for d in disagreements])) if disagreements else None
    )
    return {
        "empty": False,
        "meta": {"run_id": run_id, "n_rows": len(rows)},
        "origins": origins,
        "crux_set": crux_set,
        "mean_disagree_frac": mean_disagree_frac,
    }


def _pass1_of(
    store: Store, *, suite: str, version: str, model: str, B: int, seed: int
) -> tuple[float, float, float] | None:
    run_id = latest_run_id(store, suite=suite, version=version, model=model)
    if run_id is None:
        return None
    items = list(items_by_pid(store, run_id).values())
    if not items:
        return None
    _k, mean, lo, hi = pass_curve(items, [1], B=B, seed=seed)[0]
    return mean, lo, hi


def view_steering(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    suite = cast(str, cfg_get(cfg, "suite", eval_suites.DEV_GEN_SUITE))
    version = cast(str, cfg_get(cfg, "version", "v1"))
    B = int(cast(int, cfg_get(cfg, "B", 500)))
    seed = int(cast(int, cfg_get(cfg, "seed", 0)))

    models = models_present(store, suite=suite, version=version)
    if not models:
        return empty_view(f"no runs found for suite {suite!r}")

    by_layer: dict[int, list[tuple[float, float, float, float]]] = {}
    base_pass1 = _pass1_of(store, suite=suite, version=version, model="base", B=B, seed=seed)
    rl_control: dict[str, JSON] = {}
    for m in models:
        match = _STEER_RE.match(m)
        p1 = _pass1_of(store, suite=suite, version=version, model=m, B=B, seed=seed)
        if p1 is None or match is None:
            continue
        layer, alpha = int(match.group(1)), float(match.group(2))
        if alpha < 0:  # the RL `-delta` control (steering_sweep's convention: alpha=-1 on rl)
            rl_control[str(layer)] = {"alpha": alpha, "pass1": p1[0], "ci": [p1[1], p1[2]]}
            continue
        by_layer.setdefault(layer, []).append((alpha, p1[0], p1[1], p1[2]))
    for layer in by_layer:
        by_layer[layer].sort()

    if base_pass1 is None and not by_layer:
        return empty_view(f"no base/steer:* runs found for suite {suite!r}")
    return {
        "empty": False,
        "meta": {"suite": suite, "version": version, "B": B},
        "base_pass1": list(base_pass1) if base_pass1 is not None else None,
        "curves_by_layer": {str(k): v for k, v in by_layer.items()},
        "rl_control": rl_control,
    }


def _alpha_of(model: str) -> float | None:
    if model == "base":
        return 0.0
    if model == "rl":
        return 1.0
    match = _INTERP_RE.match(model)
    return float(match.group(1)) if match else None


def view_interpolation(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    version = cast(str, cfg_get(cfg, "version", "v1"))
    B = int(cast(int, cfg_get(cfg, "B", 500)))
    seed = int(cast(int, cfg_get(cfg, "seed", 0)))
    pass1_suites = (eval_suites.DEV_GEN_SUITE, eval_suites.SEALED_GEN_SUITE)

    models: set[str] = set()
    for s in (*pass1_suites, eval_suites.K128_SUITE):
        models |= set(models_present(store, suite=s, version=version))
    interp_models = sorted(
        (m for m in models if _alpha_of(m) is not None),
        key=lambda m: cast(float, _alpha_of(m)),
    )
    if not interp_models:
        return empty_view("no base/rl/interp:* runs found")

    pass1_vs_alpha: list[tuple[float, float, float, float]] = []
    for m in interp_models:
        items: list[ItemResult] = []
        for s in pass1_suites:
            run_id = latest_run_id(store, suite=s, version=version, model=m)
            if run_id is not None:
                items += list(items_by_pid(store, run_id).values())
        if not items:
            continue
        _k, mean, lo, hi = pass_curve(items, [1], B=B, seed=seed)[0]
        pass1_vs_alpha.append((cast(float, _alpha_of(m)), mean, lo, hi))

    pass_k_by_alpha: dict[str, list[tuple[int, float, float, float]]] = {}
    for m in interp_models:
        run_id = latest_run_id(store, suite=eval_suites.K128_SUITE, version=version, model=m)
        if run_id is None:
            continue
        items = list(items_by_pid(store, run_id).values())
        if not items:
            continue
        pass_k_by_alpha[str(_alpha_of(m))] = pass_curve(items, INTERP_KS, B=B, seed=seed)

    if not pass1_vs_alpha and not pass_k_by_alpha:
        return empty_view("base/rl/interp:* runs exist but have no items on the interp eval suites")
    return {
        "empty": False,
        "meta": {"version": version, "B": B},
        "pass1_vs_alpha": pass1_vs_alpha,
        "pass_k_by_alpha": pass_k_by_alpha,
    }


def view_probes(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    run_id = cast("str | None", cfg.get("run_id")) or latest_interp_run(store, "probes")
    if run_id is None:
        return empty_view("no `interp probes` run in the store yet")
    rows = load_interp_rows(store, "probes", run_id)
    if not rows:
        return empty_view(f"probes run {run_id!r} has no rows")

    by_model: dict[str, dict[str, dict[str, JSON]]] = {}
    for r in rows:
        by_model.setdefault(cast(str, r["model"]), {})[cast(str, r["layer"])] = r
    layers = [k for k in LAYER_KEYS if any(k in by_model[m] for m in by_model)]

    def series(metric_path: tuple[str, ...]) -> dict[str, list[tuple[int, JSON]]]:
        out: dict[str, list[tuple[int, JSON]]] = {}
        for model, per_layer in by_model.items():
            pts: list[tuple[int, JSON]] = []
            for i, layer in enumerate(layers):
                row = per_layer.get(layer)
                if row is None:
                    continue
                val: JSON = row
                for key in metric_path:
                    if val is None:
                        break
                    val = _d(val).get(key)
                pts.append((i, val))
            out[model] = pts
        return out

    return {
        "empty": False,
        "meta": {"run_id": run_id, "layers": layers},
        "next_rule_acc": series(("next_rule", "test_acc")),
        "lines_to_go_r2": series(("lines_to_go", "test_r2")),
        "lines_to_go_bin_acc": series(("lines_to_go", "test_bin_acc")),
        "cited_back_acc": series(("cited_back", "test_acc")),
        "length_generalisation_ood_r2": series(("length_generalisation", "ood_r2")),
        "mdl_next_rule_compression": series(("mdl_next_rule", "compression")),
        "mdl_lines_to_go_compression": series(("mdl_lines_to_go_binned", "compression")),
    }
