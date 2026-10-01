"""B7: excess description length -- dashboard page "edl".

**Row schema (fixed; WP8 integration fix list item 5).** WP6 originally wrote this view against a
guessed schema (`model`/`cls`/`test_nats_by_n`/`test_nats_theta0`/`pass8_theta0`/`pass8_thetan`)
before WP5's `edl.py` (`build_classes`/`run_edl`, plan §3's WP5 row) had landed. The rows
`Store.append_edl` actually receives, one per `(class, lr, order)` combination, come straight out
of `code.ndbench.edl.run_edl` and look like this instead::

    {"class": str, "model_identity": {"adapter": ..., "ckpt": ..., "ckpt_sha256": ..., ...},
     "lr": float, "order": int, "n": int, "m": int, "seed": int,
     "online_nats": [float, ...],                          # length n; ell(theta_{i-1}; x_i, y_i)
     "L_test_0": float, "L_test_n": float,                  # L_test(theta_0) / L_test(theta_n)
     "L_test_by_checkpoint": {"8": float, "16": float, ...},  # includes str(n) itself
     "EDL_by_checkpoint": {"8": float, ...}, "EDL": float, "EDL_per_example": float,
     "start_excluded_online_nats": [float, ...] | None,
     "L_test_0_start_excluded": float | None, "L_test_n_start_excluded": float | None,
     "EDL_start_excluded": float | None,
     "pass8_before": float | None, "pass8_after": float | None}

`Store` still has no run-registry entry for `append_edl` rows (`cli.py`'s `cmd_edl` writes
straight to `store.append_edl(run_id, rows)` without ever calling `store.new_run`), so EDL run ids
remain undiscoverable via `store.runs()`; this view keeps discovering them by listing
`store.edl_dir` directly (a public `Store` attribute).

There is no per-model *label* string in a row -- `model_identity` is the full adapter identity
dict `code.ndbench.adapters.base.ModelAdapter.identity()` returns. `_model_label` derives a
display/grouping key from it: `identity["label"]` when present (test doubles like
`test_wp5_fixtures.TinyGptAdapter`/`ConstantNllAdapter` set one), else `identity["ckpt"]` (the
production `AbsGptAdapter`, whose identity has no `"label"` key but always has a unique checkpoint
path), else the identity dict's own JSON as a last resort.

Since `run_edl` already computes `EDL`/`EDL_by_checkpoint` exactly (plan §B7's
`sum(online_nats[:n']) - n' * L_test(theta_n')`, via `metrics.edl`), this view reads those fields
directly rather than re-deriving them, falling back to `metrics.edl` on `online_nats`/
`L_test_by_checkpoint` only if a row is missing the precomputed field (`_edl_at`). Per `(class,
model)`, averaging over every `order` sharing an `lr`:

- `EDL` at each checkpoint in `cfg["checkpoints"]` (default 8/16/32/64) and at each row's own final
  `n`, per `lr`;
- `EDL* = min_lr(EDL) + ln(len(lrs))` (plan §B7's "hyperparameter cost"), with a percentile
  bootstrap CI on the *final* EDL at the best `lr` -- real rows carry no per-example held-out
  losses (only the mean `L_test_by_checkpoint`), so this CI is `(nan, nan)` in practice today;
  `_edl_ci` still honours an optional, forward-compatible `test_nats_per_example_by_n` field if a
  future `run_edl` variant adds one, resampling those per-example losses the same way the original
  WP6 guess did;
- `L_test(theta_0)` and `pass@8(theta_0)`, averaged over every row (the same starting checkpoint
  regardless of `lr`); `L_test(theta_n)` and `pass@8(theta_n)` averaged over only the best `lr`'s
  orders, since which `theta_n` those describe depends on which `lr` trained it -- pooling them
  across every `lr` would silently mix checkpoints from different training runs.

The view's *output* shape (`{"empty": ..., "meta": ..., "classes": {cls: {model: {...}}}}`, with
per-`(cls, model)` keys `n_rows`/`n_lrs`/`edl_by_lr`/`edl_curve_by_lr`/`best_lr`/`edl_star`/
`edl_star_ci`/`l_test_theta0`/`l_test_thetan`/`pass8_theta0`/`pass8_thetan`) is unchanged from the
original WP6 version -- only the *input* row parsing was wrong, and `dash/pages.py`'s `render_edl`
(a file this WP does not own) already reads that output shape.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from ..core.io import read_jsonl
from ..core.types import JSON
from ..metrics import edl as edl_stat
from ..store import Store
from ._common import cfg_get, empty_view

DEFAULT_CHECKPOINTS: tuple[int, ...] = (8, 16, 32, 64)


def _discover_edl_run_ids(store: Store) -> list[str]:
    if not store.edl_dir.exists():
        return []
    return sorted(p.stem for p in store.edl_dir.glob("*.jsonl"))


def _read_edl_rows(store: Store, run_id: str) -> list[dict[str, Any]]:
    path = store.edl_dir / f"{run_id}.jsonl"
    if not path.exists():
        return []
    return list(read_jsonl(path))


def _mean(vals: Sequence[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def _label_from_run_id(run_id: str) -> str | None:
    """Recover `--model`'s resolved store label (e.g. `"base"`/`"rl"`/`"armC"`) from `cli.py`'s
    `cmd_edl` run id convention `f"edl-{store_label}-{args.n}-{args.m}"`, or `None` if `run_id`
    doesn't look like that (e.g. a hand-written test file, or `cfg["run_ids"]` naming something
    else). Preferred over `model_identity` (WP8 integration fix list item: the EDL page showed raw
    checkpoint paths like `/scratch/.../pretrain.pt` instead of `"base"`/`"rl"`) because it is
    exactly the label the caller asked to run under, with no guessing from the checkpoint path --
    which has no portable mapping back to a friendly name and would need updating by hand every
    time a checkpoint moves."""
    parts = run_id.split("-")
    if len(parts) >= 4 and parts[0] == "edl" and parts[-1].isdigit() and parts[-2].isdigit():
        label = "-".join(parts[1:-2])
        if label:
            return label
    return None


def _model_label(identity: object, run_id: str | None = None) -> str:
    """A display/grouping string for an EDL row. Priority: (1) `identity["label"]` when the
    identity dict sets one explicitly (test doubles like `TinyGptAdapter`/`ConstantNllAdapter` --
    an explicit label is always the most trustworthy name available); (2) failing that, `run_id`'s
    own embedded label (see `_label_from_run_id`) -- this is what fixes the real bug, since the
    production `AbsGptAdapter`'s identity has no `"label"` key at all; (3) `identity["ckpt"]` as a
    last resort (`run_id` not in the `cmd_edl` naming convention, e.g. a hand-rolled `cfg
    ["run_ids"]`); (4) the identity dict's own JSON."""
    if isinstance(identity, Mapping):
        label = identity.get("label")
        if isinstance(label, str) and label:
            return label
    if run_id is not None:
        from_run_id = _label_from_run_id(run_id)
        if from_run_id is not None:
            return from_run_id
    if isinstance(identity, Mapping):
        ckpt = identity.get("ckpt")
        if isinstance(ckpt, str) and ckpt:
            return ckpt
        return json.dumps(identity, sort_keys=True)
    return str(identity)


def _edl_at(row: Mapping[str, JSON], cp: int) -> float | None:
    """`EDL` at checkpoint `cp`: the row's own precomputed `EDL_by_checkpoint[str(cp)]` if present,
    else derived from `online_nats[:cp]`/`L_test_by_checkpoint[str(cp)]` via `metrics.edl`."""
    by_cp = cast("Mapping[str, JSON] | None", row.get("EDL_by_checkpoint"))
    if isinstance(by_cp, Mapping) and str(cp) in by_cp:
        val = by_cp[str(cp)]
        return float(cast(float, val)) if val is not None else None
    online = cast("list[float] | None", row.get("online_nats"))
    test_by_cp = cast("Mapping[str, JSON] | None", row.get("L_test_by_checkpoint"))
    if online is not None and isinstance(test_by_cp, Mapping) and len(online) >= cp:
        tv = test_by_cp.get(str(cp))
        if tv is not None:
            return edl_stat(online[:cp], float(cast(float, tv)))
    return None


def _edl_ci(row: Mapping[str, JSON], n_final: int, seed: int, B: int = 2000) -> tuple[float, float]:
    """Bootstrap CI on this row's final-checkpoint EDL, resampling held-out per-example nats
    (`test_nats_per_example_by_n`) if present; `(nan, nan)` otherwise -- real `run_edl` rows carry
    no such field today (see the module docstring), so this is currently always `(nan, nan)` in
    production, kept for a future per-example-nats extension and exercised by tests that supply it.
    """
    per_example = cast("dict[str, list[float]] | None", row.get("test_nats_per_example_by_n"))
    online = cast("list[float] | None", row.get("online_nats"))
    if not per_example or online is None:
        return float("nan"), float("nan")
    examples = per_example.get(str(n_final))
    if not examples:
        return float("nan"), float("nan")
    online_sum = sum(online[:n_final])
    arr = np.asarray(examples, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(B, arr.size))
    l_test_boot = np.mean(arr[idx], axis=1)
    edl_boot = online_sum - n_final * l_test_boot
    lo, hi = (float(x) for x in np.percentile(edl_boot, [2.5, 97.5]))
    return lo, hi


def view(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    checkpoints = cast("list[int]", cfg_get(cfg, "checkpoints", list(DEFAULT_CHECKPOINTS)))
    seed = int(cast(int, cfg_get(cfg, "seed", 0)))
    run_ids = cast("list[str] | None", cfg.get("run_ids")) or _discover_edl_run_ids(store)
    if not run_ids:
        return empty_view("no EDL run files found under store/edl/")

    rows: list[tuple[str, dict[str, Any]]] = []
    for run_id in run_ids:
        rows.extend((run_id, row) for row in _read_edl_rows(store, run_id))
    if not rows:
        return empty_view(f"EDL run files found ({run_ids}) but contained no rows")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run_id, row in rows:
        model = _model_label(row.get("model_identity"), run_id)
        cls = str(row.get("class", row.get("cls", "unknown")))
        groups.setdefault((model, cls), []).append(row)

    out_classes: dict[str, dict[str, Any]] = {}
    for (model, cls), grp in groups.items():
        by_lr: dict[float, list[dict[str, Any]]] = {}
        for row in grp:
            by_lr.setdefault(float(cast(float, row.get("lr", 0.0))), []).append(row)

        edl_final_by_lr: dict[str, float] = {}
        edl_curve_by_lr: dict[str, list[tuple[int, float]]] = {}
        ci_by_lr: dict[str, tuple[float, float]] = {}
        # per-lr "after training" aggregates: theta_n depends on which lr trained it, so
        # l_test_thetan/pass8_thetan are reported at the *best* lr specifically (below), not
        # pooled across every lr the way theta_0 (the shared starting checkpoint) safely can be.
        pass8_before: list[float] = []
        l_test_before: list[float] = []
        pass8_after_by_lr: dict[str, list[float]] = {}
        l_test_after_by_lr: dict[str, list[float]] = {}
        for lr, lr_rows in by_lr.items():
            finals: list[float] = []
            cis: list[tuple[float, float]] = []
            curve_pts: dict[int, list[float]] = {cp: [] for cp in checkpoints}
            pass8_after: list[float] = []
            l_test_after: list[float] = []
            for row in lr_rows:
                n_final = int(cast(int, row.get("n", 0)))
                edl_final = row.get("EDL")
                final_val = (
                    float(cast(float, edl_final))
                    if edl_final is not None
                    else _edl_at(row, n_final)
                )
                if final_val is not None:
                    finals.append(final_val)
                    cis.append(_edl_ci(row, n_final, seed))
                for cp in checkpoints:
                    v = _edl_at(row, cp)
                    if v is not None:
                        curve_pts[cp].append(v)
                if row.get("pass8_before") is not None:
                    pass8_before.append(float(cast(float, row["pass8_before"])))
                if row.get("pass8_after") is not None:
                    pass8_after.append(float(cast(float, row["pass8_after"])))
                if row.get("L_test_0") is not None:
                    l_test_before.append(float(cast(float, row["L_test_0"])))
                l_test_n_val = row.get("L_test_n")
                if l_test_n_val is not None:
                    l_test_after.append(float(cast(float, l_test_n_val)))
            if finals:
                edl_final_by_lr[str(lr)] = _mean(finals)
                good_cis = [c for c in cis if not math.isnan(c[0])]
                ci_by_lr[str(lr)] = (
                    (_mean([c[0] for c in good_cis]), _mean([c[1] for c in good_cis]))
                    if good_cis
                    else (float("nan"), float("nan"))
                )
            edl_curve_by_lr[str(lr)] = [(cp, _mean(v)) for cp, v in sorted(curve_pts.items())]
            pass8_after_by_lr[str(lr)] = pass8_after
            l_test_after_by_lr[str(lr)] = l_test_after
        if not edl_final_by_lr:
            continue
        best_lr = min(edl_final_by_lr, key=lambda k: edl_final_by_lr[k])
        edl_star = edl_final_by_lr[best_lr] + math.log(max(len(edl_final_by_lr), 1))

        out_classes.setdefault(cls, {})[model] = {
            "n_rows": len(grp),
            "n_lrs": len(edl_final_by_lr),
            "edl_by_lr": edl_final_by_lr,
            "edl_curve_by_lr": edl_curve_by_lr,
            "best_lr": best_lr,
            "edl_star": edl_star,
            "edl_star_ci": ci_by_lr.get(best_lr, (float("nan"), float("nan"))),
            "l_test_theta0": _mean(l_test_before),
            "l_test_thetan": _mean(l_test_after_by_lr.get(best_lr, [])),
            "pass8_theta0": _mean(pass8_before),
            "pass8_thetan": _mean(pass8_after_by_lr.get(best_lr, [])),
        }

    if not out_classes:
        return empty_view("EDL rows found but none had a usable 'EDL'/'online_nats' field")
    return {
        "empty": False,
        "meta": {"checkpoints": list(checkpoints), "run_ids": run_ids},
        "classes": out_classes,
    }
