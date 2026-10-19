"""RL-minus-base mean-difference steering vectors and the steering sweep (`INTERP_SPEC.md` §2).

`fit_diff_vectors` needs `Capability.ACTIVATIONS`; `steering_sweep`'s steered models need
`Capability.STEER`; both are already implemented for `AbsGptAdapter` (`adapters/absgpt.py`) -- this
module only selects positions, averages, and drives the ordinary runner/`Store` machinery
(`interp/runner_support.py`), it does not touch model internals itself.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..adapters.base import ModelAdapter, SteeringTerm
from ..core.io import sha256_str, utc_now_iso
from ..core.types import JSON, Sequent
from ..judge.base import Judge
from ..metrics import pass_at_k
from ..store import Store
from . import data, eval_suites
from .runner_support import eval_suites_from_cfg, run_eval_suites

DEFAULT_VECTOR_DIR = Path("/scratch/cowifornia/nd-rl/bench/out/interp/vectors")
#: `INTERP_SPEC.md` §2: "layer in {0,1,2,3}", "alpha in {0.25, 0.5, 1, 1.5, 2}".
LAYERS: tuple[int, ...] = (0, 1, 2, 3)
ALPHAS: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0)
POSITION_TYPES: tuple[str, ...] = ("proof", "prompt", "line_start")


def _positions_for(position_type: str, *, prompt_len: int, proof_len: int, body: str) -> list[int]:
    if position_type == "prompt":
        return list(range(prompt_len))
    if position_type == "proof":
        return list(range(prompt_len, prompt_len + proof_len))
    if position_type == "line_start":
        return [prompt_len + i for i in data.line_start_positions(body)]
    raise ValueError(f"unknown position_type {position_type!r}, expected one of {POSITION_TYPES}")


@dataclass(frozen=True)
class DiffVectorResult:
    layer: int
    position_type: str
    vector: np.ndarray  # shape (d,)
    norm: float
    mean_residual_norm: float
    n_examples: int
    path: Path


def fit_diff_vectors(
    base: ModelAdapter,
    rl: ModelAdapter,
    pairs: Sequence[tuple[Sequent, str]],
    layers: Sequence[int] = LAYERS,
    position_types: Sequence[str] = ("proof",),
    *,
    out_dir: Path = DEFAULT_VECTOR_DIR,
) -> list[DiffVectorResult]:
    """`delta_l = mean over selected positions of (h_rl - h_base)` after block `l`, for each
    `(layer, position_type)` -- `INTERP_SPEC.md` §2. `pairs` are the same teacher-forced
    `(sequent, proof)` examples scored under both models (2-5k reserve minimal proofs, L6-10, per
    the spec). Saves each vector as `out_dir/diff_l<layer>_<position_type>.npy` with a `.json`
    sidecar (`layers`->just this one, `position_type`, `n`, `source_sha`, `norm`,
    `mean_residual_norm`, `relative_norm = norm / mean_residual_norm`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_acts = base.activations(pairs, layers)
    rl_acts = rl.activations(pairs, layers)
    prompt_lens = [base.token_count(seq) for seq, _body in pairs]
    source_sha = sha256_str(json.dumps([[seq.thm(), body] for seq, body in pairs], sort_keys=True))[
        :16
    ]

    results: list[DiffVectorResult] = []
    for li, layer in enumerate(layers):
        for position_type in position_types:
            diffs: list[np.ndarray] = []
            base_norms: list[float] = []
            for (_seq, body), b_full, r_full, prompt_len in zip(
                pairs, base_acts, rl_acts, prompt_lens, strict=True
            ):
                b_layer, r_layer = b_full[li], r_full[li]  # (T, d) each
                proof_len = b_layer.shape[0] - prompt_len
                positions = _positions_for(
                    position_type, prompt_len=prompt_len, proof_len=proof_len, body=body
                )
                if not positions:
                    continue
                diffs.append(r_layer[positions] - b_layer[positions])
                base_norms.append(float(np.linalg.norm(b_layer[positions], axis=-1).mean()))
            if not diffs:
                continue
            all_diffs = np.concatenate(diffs, axis=0)
            delta = all_diffs.mean(axis=0)
            norm = float(np.linalg.norm(delta))
            mean_residual_norm = float(np.mean(base_norms)) if base_norms else float("nan")

            out_path = out_dir / f"diff_l{layer}_{position_type}.npy"
            np.save(out_path, delta.astype(np.float32))
            sidecar = {
                "layer": layer,
                "position_type": position_type,
                "n": len(pairs),
                "n_positions": int(all_diffs.shape[0]),
                "source_sha": source_sha,
                "norm": norm,
                "mean_residual_norm": mean_residual_norm,
                "relative_norm": (norm / mean_residual_norm) if mean_residual_norm else None,
                "created_utc": utc_now_iso(),
            }
            out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1, sort_keys=True))
            results.append(
                DiffVectorResult(
                    layer=layer,
                    position_type=position_type,
                    vector=delta,
                    norm=norm,
                    mean_residual_norm=mean_residual_norm,
                    n_examples=len(pairs),
                    path=out_path,
                )
            )
    return results


def _mean_pass1(store: Store, run_id: str) -> float:
    vals = [pass_at_k(it.n, it.c, 1) for it in store.items(run_id)]
    vals = [v for v in vals if v == v]  # drop nan
    return float(np.mean(vals)) if vals else float("-inf")


def steering_sweep(
    store: Store,
    cfg: Mapping[str, JSON],
    *,
    base_adapter: ModelAdapter,
    rl_adapter: ModelAdapter,
    vectors_dir: Path = DEFAULT_VECTOR_DIR,
    judge: Judge | None = None,
    base_label: str = "base",
    rl_label: str = "rl",
) -> dict[str, Any]:
    """`INTERP_SPEC.md` §2's sweep: for each `(layer, alpha)` in `cfg`'s (default `LAYERS`x`ALPHAS`)
    grid, `base_adapter.with_steering((SteeringTerm(layer, vector_path, alpha, "proof"),))`, run
    the standard eval suites (`k=8`, `T=0.8` by default) as an ordinary `Store` run labelled
    `steer:<layer>:<alpha>`; `alpha=0` (labelled plain `"base"`) is the unsteered run. Then steers
    `rl_adapter` with `-delta` (`alpha=-1.0` on the same vector) on whichever layer scored highest
    mean pass@1 on `dev`, as a control ("does removing delta hurt RL?"), labelled
    `steer:<layer>:-1`. Vectors missing on disk (not yet fit by `fit_diff_vectors`) are skipped
    with a note, not an error.

    **Generalisation (`DANSTATE_PLAN.md` §7 WP-D6).** `judge` (default `None`, i.e. a fresh
    `NDJudge()` -- `run_eval_suites`' own default, unchanged) lets a caller judge Dan's state-env
    models with `judge.lean.LeanJudge()` instead. `base_label`/`rl_label` (default `"base"`/`"rl"`,
    exactly the old hard-coded literals) name the two input models for the `Store`/dashboard and
    the steered-model label prefix; passing anything else only changes the *steered* labels'
    format (`f"steer:{base_label}:{layer}:{alpha:g}"` instead of the old bare `f"steer:{layer}:
    {alpha:g}"`) so two different base models' sweeps in one `Store` cannot collide -- the old
    default labels (`base_label == "base"`) are therefore byte-identical to every pre-existing
    caller/config/test, which is the whole point of keeping the two literals as the defaults."""

    def _int_cfg(key: str, default: int) -> int:
        v = cfg.get(key)
        return default if v is None else int(cast(float, v))

    def _float_cfg(key: str, default: float) -> float:
        v = cfg.get(key)
        return default if v is None else float(cast(float, v))

    def _steer_label(layer: int | str, alpha: float, label: str) -> str:
        tag = f"{layer}:{alpha:g}"
        return f"steer:{tag}" if label == "base" else f"steer:{label}:{tag}"

    position_type = str(cfg.get("position_type", "proof"))
    layers = tuple(cast("list[int]", cfg.get("layers", list(LAYERS))))
    alphas = tuple(cast("list[float]", cfg.get("alphas", list(ALPHAS))))
    k = _int_cfg("k", 8)
    temperature = _float_cfg("temperature", 0.8)
    seed = _int_cfg("seed", 0)
    budget_s = cast("float | None", cfg.get("budget_s"))

    excluder = None
    if cfg.get("exclude_train"):
        excluder = data.build_exclusion_index(cache_dir=cast("str | None", cfg.get("cache_dir")))
    suites = eval_suites.build_all(store, excluder=excluder, seed=seed)

    def _run(adapter: ModelAdapter, label: str) -> dict[str, Any]:
        return run_eval_suites(
            store,
            adapter,
            model_label=label,
            suites=suites,
            k_pass1=k,
            k_pass_k=k,
            temperature=temperature,
            seed=seed,
            budget_s=budget_s,
            judge=judge,
            only=eval_suites_from_cfg(cfg),
        )

    notes: list[str] = list(suites.notes)
    runs: dict[str, Any] = {base_label: _run(base_adapter, base_label)}
    best_layer: int | None = None
    best_metric = float("-inf")
    for layer in layers:
        vec_path = Path(vectors_dir) / f"diff_l{layer}_{position_type}.npy"
        if not vec_path.exists():
            notes.append(f"no fitted vector at layer {layer} ({vec_path}); skipping this layer")
            continue
        for alpha in alphas:
            label = _steer_label(layer, alpha, base_label)
            steered = base_adapter.with_steering(
                (SteeringTerm(layer, str(vec_path), alpha, "proof"),)
            )
            result = _run(steered, label)
            runs[label] = result
            metric = _mean_pass1(store, cast(str, result[eval_suites.DEV_GEN_SUITE]["run_id"]))
            if metric > best_metric:
                best_metric, best_layer = metric, layer

    if best_layer is not None:
        vec_path = Path(vectors_dir) / f"diff_l{best_layer}_{position_type}.npy"
        rl_control_label = _steer_label(best_layer, -1.0, rl_label if rl_label != "rl" else "base")
        rl_steered = rl_adapter.with_steering(
            (SteeringTerm(best_layer, str(vec_path), -1.0, "proof"),)
        )
        runs[rl_control_label] = _run(rl_steered, rl_control_label)
    else:
        notes.append("no fitted vectors found on disk; skipped the whole sweep + RL control")

    return {
        "runs": runs,
        "best_layer": best_layer,
        "best_layer_dev_pass1": best_metric,
        "notes": notes,
    }


def fit_early_stop_contrast(
    store: Store,
    base_adapter: ModelAdapter,
    *,
    suite: str,
    version: str,
    model: str = "base",
    layer: int,
    out_dir: Path = DEFAULT_VECTOR_DIR,
) -> DiffVectorResult | None:
    """Optional, off-by-default (`INTERP_SPEC.md` §2): from `base_adapter`'s own `Store` samples,
    the mean activation difference (at the last *written* token) between failed attempts that
    stopped short of the reference length and attempts that reached it. `None` if no matching run
    has samples (skip with a note, per the package's overall convention) or no examples of one of
    the two groups are found."""
    from ..analysis._common import latest_run_id  # WP6 module; read-only import

    run_id = latest_run_id(store, suite=suite, version=version, model=model)
    if run_id is None:
        return None
    problems = {p.pid: p for p in store.load_suite(suite, version)}
    path = store.samples_dir / f"{run_id}.jsonl.gz"
    if not path.exists():
        return None
    from ..core.io import read_jsonl
    from ..core.proof import line_count

    short_failed: list[tuple[Sequent, str]] = []
    reached: list[tuple[Sequent, str]] = []
    for row in read_jsonl(path):
        problem = problems.get(row["pid"])
        if problem is None or problem.ref is None or not row.get("body"):
            continue
        written = line_count(row["body"])
        if row.get("ok"):
            reached.append((problem.sequent, row["body"]))
        elif not row.get("ok") and written < problem.ref.lines:
            short_failed.append((problem.sequent, row["body"]))
    if not short_failed or not reached:
        return None

    def _last_token_activation(pairs: Sequence[tuple[Sequent, str]]) -> np.ndarray:
        acts = base_adapter.activations(pairs, (layer,))
        return np.stack([a[0, -1] for a in acts], axis=0)  # (n, d): last token of each sequence

    mean_reached = _last_token_activation(reached).mean(axis=0)
    mean_short = _last_token_activation(short_failed).mean(axis=0)
    delta = mean_reached - mean_short
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"earlystop_l{layer}.npy"
    np.save(out_path, delta.astype(np.float32))
    norm = float(np.linalg.norm(delta))
    sidecar = {
        "layer": layer,
        "n_reached": len(reached),
        "n_short_failed": len(short_failed),
        "norm": norm,
        "created_utc": utc_now_iso(),
    }
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1, sort_keys=True))
    return DiffVectorResult(
        layer=layer,
        position_type="last_written_line",
        vector=delta,
        norm=norm,
        mean_residual_norm=float("nan"),
        n_examples=len(reached) + len(short_failed),
        path=out_path,
    )
