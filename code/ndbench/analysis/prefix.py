"""B6: prefix completion -- dashboard page "prefix".

Reads the `prefix` suite (WP4's `suites/prefix.py`; `Problem.tags` carry `j`/`L`/`src_pid`, plan
§3's WP4 row; `Problem.prefix.j` also gives `j` directly) and, per model:

- success (`ItemResult.c / .n`, "prefix + continuation passes `nd_verify`") grouped three ways:
  by `j` (pooled over every proof length), by lines left `L - j`, and by the proof's total length
  `L` (the plan's "by absolute position" -- read here as *where in the length spectrum the item
  sits*, since the prefix cut itself, `j`, already is "position from the start"; flagged as an
  interpretation in case the integration step meant something else by that phrase);
- a first-failure hazard by absolute line index (`metrics.hazard`), left-truncated at the forced
  prefix: an attempt enters the risk set at line `j + 1` (lines `1..j` were given, not generated,
  and cannot fail). Built from the run's samples file when present: a bad line is a failure at
  `first_bad_line`, everything else (verified, `early_qed`, `no_qed`, ...) is censored at its
  `written` length. Without a samples file it falls back to `ItemResult.ok_written` (censored) and
  `ItemResult.fail_lines` (failures), which drops attempts that stopped without a bad line;
- a fitted logit `p = a + b*(L - j) + c*j` (`_common.irls_binomial`, item-level `n`/`c`) -- **fixed
  effects only**: the plan calls for "proof random intercepts", a mixed-effects (GLMM) term; WP0's
  `metrics.py` has no GLMM/random-effects solver and adding one is out of scope for a numpy IRLS
  view module, so this reports the fixed-effects logit and leaves per-proof intercepts as an open
  issue (see the WP6 report).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from ..core.io import read_jsonl
from ..core.types import JSON
from ..metrics import hazard, pass_at_k
from ..store import Store
from ._common import (
    cfg_get,
    default_version,
    empty_view,
    irls_binomial,
    items_by_pid,
    latest_run_id,
    models_present,
    problems_by_pid,
)

DEFAULT_SUITE = "prefix"
J_TAG = "j"
L_TAG = "L"


def _bucket_success(rows: list[tuple[float, int, int]]) -> dict[str, Any]:
    """`rows`: `(bucket_value, n, c)`; pools `n`/`c` per bucket for pass@1."""
    by_bucket: dict[float, list[tuple[int, int]]] = {}
    for bucket, n, c in rows:
        by_bucket.setdefault(bucket, []).append((n, c))
    out: dict[str, Any] = {}
    for bucket, ncs in sorted(by_bucket.items()):
        n_sum = sum(n for n, _ in ncs)
        c_sum = sum(c for _, c in ncs)
        out[str(bucket)] = {
            "pass1": (c_sum / n_sum) if n_sum else float("nan"),
            "n_items": len(ncs),
            "n_samples": n_sum,
        }
    return out


def _hazard_rows_from_samples(
    store: Store, run_id: str, j_by_pid: Mapping[str, int]
) -> list[tuple[int, bool, int]] | None:
    """`(exit_line, failed, entry_line)` per sampled attempt (see the module docstring), or `None`
    when the run has no samples file."""
    path = store.samples_dir / f"{run_id}.jsonl.gz"
    if not path.exists():
        return None
    rows: list[tuple[int, bool, int]] = []
    for s in read_jsonl(path):
        j = j_by_pid.get(cast(str, s.get("pid")))
        if j is None:
            continue
        bad = s.get("first_bad_line")
        if not s.get("ok") and bad is not None:
            rows.append((int(cast(int, bad)), True, j + 1))
        elif s.get("written") is not None:
            rows.append((int(cast(int, s["written"])), False, j + 1))
    return rows


def view(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    suite = cast(str, cfg_get(cfg, "suite", DEFAULT_SUITE))
    override_models = cast("list[str] | None", cfg.get("models"))
    # Fix list item 1, applied uniformly: pick whichever version has runs rather than "v1".
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

    per_model: dict[str, Any] = {}
    for model in models:
        run_id = latest_run_id(store, suite=suite, model=model, version=version)
        if run_id is None:
            continue
        items = items_by_pid(store, run_id)
        rows: list[tuple[int, int, int, int]] = []  # (j, L, n, c)
        item_hazard_rows: list[tuple[int, bool, int]] = []
        j_by_pid: dict[str, int] = {}
        for pid, item in items.items():
            p = problems.get(pid)
            if p is None:
                continue
            j_val = p.tags.get(J_TAG)
            l_val = p.tags.get(L_TAG)
            if j_val is None and p.prefix is not None:
                j_val = p.prefix.j
            if j_val is None or l_val is None:
                continue
            j_int = int(cast(int, j_val))
            j_by_pid[pid] = j_int
            rows.append((j_int, int(cast(int, l_val)), item.n, item.c))
            for w in item.ok_written:
                item_hazard_rows.append((w, False, j_int + 1))
            for line in item.fail_lines:
                item_hazard_rows.append((line, True, j_int + 1))
        if not rows:
            continue
        hazard_rows = _hazard_rows_from_samples(store, run_id, j_by_pid)
        if hazard_rows is None:
            hazard_rows = item_hazard_rows

        success_by_j = _bucket_success([(float(j), n, c) for j, _l, n, c in rows])
        success_by_lines_left = _bucket_success([(float(l_v - j), n, c) for j, l_v, n, c in rows])
        success_by_l = _bucket_success([(float(l_v), n, c) for _j, l_v, n, c in rows])
        hazard_table = hazard(hazard_rows)

        design = np.array([[1.0, float(l_v - j), float(j)] for j, l_v, _n, _c in rows])
        n_arr = np.array([n for _j, _l, n, _c in rows], dtype=float)
        c_arr = np.array([c for _j, _l, _n, c in rows], dtype=float)
        beta, se = irls_binomial(design, n_arr, c_arr)
        fitted_logit = {
            name: {"coef": float(b), "se": float(s)}
            for name, b, s in zip(("intercept", "lines_left", "j"), beta, se, strict=True)
        }

        per_model[model] = {
            "n_items": len(rows),
            "success_by_j": success_by_j,
            "success_by_lines_left": success_by_lines_left,
            "success_by_L": success_by_l,
            "hazard": [
                {"line": line, "at_risk": at_risk, "failed": failed, "hazard": haz}
                for line, at_risk, failed, haz in hazard_table
            ],
            "fitted_logit": fitted_logit,
            "overall_pass1": (
                sum(pass_at_k(n, c, 1) for _j, _l, n, c in rows if n) / max(len(rows), 1)
            ),
        }

    if not per_model:
        return empty_view(f"suite {suite!r} has runs but no items had usable j/L tags")
    return {
        "empty": False,
        "meta": {"suite": suite, "version": version, "models": models},
        "models": per_model,
    }
