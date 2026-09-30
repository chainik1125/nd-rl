"""B3: the skill profile -- dashboard page "skills".

Reads the `skill_pool` suite (WP2's `suites/existing.py`; `Problem.tags` carry skill indicators --
either a per-skill tag, e.g. `tags["classical"] = True`, or a `tags["skills"]` list naming every
skill that applies, both handled by `_skill_active`) and, per model:

- a skill x length heat-map (pass@1 and pass@32, pooled over items with that skill active at that
  length, with item counts -- plan §2 B3's "heat-map of skill x L (pass@1 and pass@32, with cell
  counts)");
- a binomial GLM `pass ~ L + skill indicators` fit by IRLS (`_common.irls_binomial`, the
  multi-covariate generalisation of `metrics.logit_slope`), one row per item (`ItemResult.n`/`.c`),
  reporting each coefficient with its standard error.

`cfg["skills"]` overrides the default skill-tag list (plan §2 B3's table): `classical`,
`case_split`, `derived_case`, `nested_case`, `case_in_box`, `negi_ande_hyp`, `depth_ge_3`. The
`uses_*` rule-count tags are descriptive (plan §2 B3) and not part of the GLM design.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from ..core.types import JSON, Problem
from ..metrics import pass_at_k
from ..store import Store
from ._common import (
    cfg_get,
    empty_view,
    irls_binomial,
    item_length,
    items_by_pid,
    latest_run_id,
    models_present,
    problems_by_pid,
)

DEFAULT_SUITE = "skill_pool"
DEFAULT_SKILLS: tuple[str, ...] = (
    "classical",
    "case_split",
    "derived_case",
    "nested_case",
    "case_in_box",
    "negi_ande_hyp",
    "depth_ge_3",
)


def _skill_active(problem: Problem, skill: str) -> bool:
    v = problem.tags.get(skill)
    if v is not None:
        return bool(v)
    skills_list = problem.tags.get("skills")
    return isinstance(skills_list, list) and skill in skills_list


def _mean_or_nan(vals: Sequence[float]) -> float:
    clean = [v for v in vals if not math.isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


def view(store: Store, cfg: Mapping[str, JSON]) -> dict[str, Any]:
    suite = cast(str, cfg_get(cfg, "suite", DEFAULT_SUITE))
    version = cast(str, cfg_get(cfg, "version", "v1"))
    skills = cast("list[str]", cfg_get(cfg, "skills", list(DEFAULT_SKILLS)))
    override_models = cast("list[str] | None", cfg.get("models"))

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
        rows: list[tuple[float, dict[str, bool], int, int]] = []
        for pid, item in items.items():
            p = problems.get(pid)
            if p is None:
                continue
            length = item_length(p)
            if length is None:
                continue
            flags = {s: _skill_active(p, s) for s in skills}
            rows.append((length, flags, item.n, item.c))
        if not rows:
            continue

        heat1: dict[str, dict[str, Any]] = {}
        heat32: dict[str, dict[str, Any]] = {}
        for s in skills:
            by_l: dict[float, list[tuple[int, int]]] = {}
            for length, flags, n, c in rows:
                if not flags[s]:
                    continue
                by_l.setdefault(length, []).append((n, c))
            row1: dict[str, Any] = {}
            row32: dict[str, Any] = {}
            for length, ncs in sorted(by_l.items()):
                p1 = [pass_at_k(n, c, 1) for n, c in ncs]
                p32 = [pass_at_k(n, c, 32) for n, c in ncs if n >= 32]
                row1[str(length)] = {"pass1": _mean_or_nan(p1), "n_items": len(ncs)}
                row32[str(length)] = {"pass32": _mean_or_nan(p32), "n_items": len(p32)}
            heat1[s] = row1
            heat32[s] = row32

        design_cols = ["intercept", "L", *skills]
        design = np.zeros((len(rows), len(design_cols)))
        n_arr = np.zeros(len(rows))
        c_arr = np.zeros(len(rows))
        for i, (length, flags, n, c) in enumerate(rows):
            design[i, 0] = 1.0
            design[i, 1] = length
            for j, s in enumerate(skills):
                design[i, 2 + j] = 1.0 if flags[s] else 0.0
            n_arr[i] = n
            c_arr[i] = c
        beta, se = irls_binomial(design, n_arr, c_arr)
        glm = {
            name: {"coef": float(b), "se": float(s)}
            for name, b, s in zip(design_cols, beta, se, strict=True)
        }

        per_model[model] = {
            "n_items": len(rows),
            "heatmap_pass1": heat1,
            "heatmap_pass32": heat32,
            "glm": glm,
        }

    if not per_model:
        return empty_view(f"suite {suite!r} has runs but no items had a known length")
    return {
        "empty": False,
        "meta": {"suite": suite, "version": version, "skills": skills, "models": models},
        "models": per_model,
    }
