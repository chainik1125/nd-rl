"""Shared `runner.run` plumbing for the steering sweep and weight interpolation evaluations
(`INTERP_SPEC.md` §§2-3: "Steering sweeps and interpolation evaluations are ordinary Store runs via
`runner.run`"). Every steered/interpolated model is scored on the same eval suites
(`interp/eval_suites.py`) through the standard `RunSpec`/`run` path, so the existing dashboard pages
(built for ordinary benchmark runs) read these runs with no special-casing.
"""

from __future__ import annotations

from typing import Any

from ..adapters.base import ModelAdapter
from ..core.types import JSON
from ..judge.nd import NDJudge
from ..runner.runner import RunSpec, run
from ..store import Store
from .eval_suites import DEV_GEN_SUITE, K128_SUITE, SEALED_GEN_SUITE, EvalSuites


def run_eval_suites(
    store: Store,
    adapter: ModelAdapter,
    *,
    model_label: str,
    suites: EvalSuites,
    k_pass1: int = 8,
    k_pass_k: int = 128,
    temperature: float = 0.8,
    seed: int = 0,
    budget_s: float | None = None,
    version: str = "v1",
) -> dict[str, JSON]:
    """Run `model_label` (`adapter`) on `suites.dev`/`suites.sealed` at `k_pass1` (pass@1),
    `suites.k128` at `k_pass_k` (pass@k curve), and `suites.textbook_dev` at `k_pass1` if it was
    built. Returns `{suite_name: RunSummary}` for whichever suites actually have items."""
    judge = NDJudge()
    plan: list[tuple[str, int]] = [(DEV_GEN_SUITE, k_pass1), (SEALED_GEN_SUITE, k_pass1)]
    if suites.k128:
        plan.append((K128_SUITE, k_pass_k))
    if suites.textbook_dev:
        plan.append(("textbook_dev", k_pass1))

    out: dict[str, Any] = {}
    for suite_name, k in plan:
        spec = RunSpec(
            suite=suite_name,
            version=version,
            model=model_label,
            k=k,
            temperature=temperature,
            seed=seed,
            budget_s=budget_s,
        )
        out[suite_name] = run(spec, adapter, judge, store)
    return out
