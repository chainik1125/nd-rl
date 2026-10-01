"""Shared `runner.run` plumbing for the steering sweep and weight interpolation evaluations
(`INTERP_SPEC.md` §§2-3: "Steering sweeps and interpolation evaluations are ordinary Store runs via
`runner.run`"). Every steered/interpolated model is scored on the same eval suites
(`interp/eval_suites.py`) through the standard `RunSpec`/`run` path, so the existing dashboard pages
(built for ordinary benchmark runs) read these runs with no special-casing.

**Judge generalisation (`DANSTATE_PLAN.md` §7 WP-D6).** `run_eval_suites` used to hard-code
`NDJudge()` -- every caller got the `nd_v2` judge regardless of which model was being evaluated.
Dan's state-env models (`adapters.danstate.DanStateAdapter`) are judged with `judge.lean.LeanJudge`
instead (`judge_id="lean_v2"`; plan §2 WP-D2), so this now takes an optional `judge` parameter:
`None` (every pre-existing caller/config) keeps the exact old behaviour, a fresh `NDJudge()` built
right here -- not threaded in as a shared default-argument instance, so this matches the pre-fix
code's own per-call `NDJudge()` exactly. A caller wanting Dan's `LeanJudge()` (or any other `Judge`)
passes an already-built instance; `interp/__main__.py` builds it once per subcommand invocation via
`cli._build_judge` (shared with `cli.py cmd_run`'s own `--judge` resolution) and passes it through
`steering.steering_sweep`/`interpolate.run_interpolation`.
"""

from __future__ import annotations

from typing import Any

from ..adapters.base import ModelAdapter
from ..core.types import JSON
from ..judge.base import Judge
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
    judge: Judge | None = None,
) -> dict[str, JSON]:
    """Run `model_label` (`adapter`) on `suites.dev`/`suites.sealed` at `k_pass1` (pass@1),
    `suites.k128` at `k_pass_k` (pass@k curve), and `suites.textbook_dev` at `k_pass1` if it was
    built. Returns `{suite_name: RunSummary}` for whichever suites actually have items.

    `judge` defaults to a fresh `NDJudge()` (module docstring: exactly the pre-fix hard-coded
    behaviour) -- pass e.g. a `judge.lean.LeanJudge()` instance to judge a non-absgpt model
    (`adapters.danstate.DanStateAdapter`'s native-Lean-text samples) correctly instead.
    """
    judge = judge if judge is not None else NDJudge()
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
