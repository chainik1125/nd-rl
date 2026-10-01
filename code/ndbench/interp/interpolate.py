"""Weight interpolation between the base and RL checkpoints (`INTERP_SPEC.md` §3):
`theta = theta_base + alpha*(theta_rl - theta_base)` for every tensor, and the standard-runner
evaluation sweep over `alpha`.

Interpolated checkpoints are plain `absgpt` checkpoint files (same `{"state", "cfg", "tok_mode",
...}` shape `adapters.absgpt._build_model_and_tokenizer` expects) under
`/scratch/cowifornia/nd-rl/bench/out/interp/ckpts/`, with a `.sha256.json` sidecar recording both
parents' hashes and `alpha`; `AbsGptAdapter` loads one exactly like any other checkpoint, so its
`identity()` (checkpoint path + its own sha256) already makes an interpolated model "just another
model in the Store" with no interp-specific adapter code needed. `alpha in {0.0, 1.0}` are *not*
materialised here -- `INTERP_SPEC.md`'s storage section keeps the `base`/`rl` labels for the two
originals, so those two curve endpoints reuse the existing checkpoints/runs directly.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import torch

from ..adapters.base import ModelAdapter
from ..core.io import sha256_file
from ..core.types import JSON
from ..judge.base import Judge
from ..store import Store
from . import data, eval_suites
from .runner_support import eval_suites_from_cfg, run_eval_suites

#: `INTERP_SPEC.md` §3: "alpha in {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5}".
ALPHAS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
#: The values actually materialised as their own checkpoint file (0 and 1 reuse base/rl).
CKPT_ALPHAS: tuple[float, ...] = tuple(a for a in ALPHAS if a not in (0.0, 1.0))

DEFAULT_CKPT_DIR = Path("/scratch/cowifornia/nd-rl/bench/out/interp/ckpts")


def alpha_tag(alpha: float) -> str:
    return f"{alpha:g}"


def interp_model_label(
    alpha: float, *, base_label: str = "base", rl_label: str = "rl", prefix: str = "interp"
) -> str:
    """`DANSTATE_PLAN.md` §7 WP-D6: `base_label`/`rl_label`/`prefix` default to the old hard-coded
    literals (`"base"`/`"rl"`/`"interp"`), so every pre-existing call site's labels are unchanged.
    Dan's pair passes `base_label="dan12-fz-s0"`, `rl_label="dan12-t1-s0"`,
    `prefix="dan12-interp"` (plan §2's "labels `<family>-interp:<alpha>`")."""
    if alpha == 0.0:
        return base_label
    if alpha == 1.0:
        return rl_label
    return f"{prefix}:{alpha_tag(alpha)}"


def _strip_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    return {k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in state.items()}


def interpolate(base_ckpt: Path, rl_ckpt: Path, alpha: float, out_dir: Path) -> Path:
    """`theta_base + alpha*(theta_rl - theta_base)`, saved to `out_dir/interp_a<alpha>.pt` with a
    `.sha256.json` sidecar (`{alpha, base_ckpt, base_sha256, rl_ckpt, rl_sha256, out_sha256}`).
    Raises `ValueError` if the two checkpoints' state dicts do not have identical keys/shapes
    (after stripping a `_orig_mod.`/`module.` prefix, the same normalisation `adapters.absgpt`
    applies on load)."""
    base_ck = torch.load(str(base_ckpt), map_location="cpu", weights_only=True)
    rl_ck = torch.load(str(rl_ckpt), map_location="cpu", weights_only=True)
    base_state = _strip_prefix(base_ck["state"])
    rl_state = _strip_prefix(rl_ck["state"])
    if base_state.keys() != rl_state.keys():
        raise ValueError(
            "interpolate: base/rl checkpoints have different state-dict keys: "
            f"{sorted(base_state.keys() ^ rl_state.keys())}"
        )
    new_state: dict[str, torch.Tensor] = {}
    for key, v_base in base_state.items():
        v_rl = rl_state[key]
        if v_base.shape != v_rl.shape:
            raise ValueError(
                f"interpolate: shape mismatch at {key!r}: {v_base.shape} vs {v_rl.shape}"
            )
        blended = v_base.float() + alpha * (v_rl.float() - v_base.float())
        new_state[key] = blended.to(v_base.dtype)

    out_ck = dict(base_ck)
    out_ck["state"] = new_state
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"interp_a{alpha_tag(alpha)}.pt"
    torch.save(out_ck, out_path)

    sidecar = {
        "alpha": alpha,
        "base_ckpt": str(base_ckpt),
        "base_sha256": sha256_file(base_ckpt),
        "rl_ckpt": str(rl_ckpt),
        "rl_sha256": sha256_file(rl_ckpt),
        "out_sha256": sha256_file(out_path),
    }
    out_path.with_suffix(out_path.suffix + ".sha256.json").write_text(
        json.dumps(sidecar, indent=1, sort_keys=True)
    )
    return out_path


def build_checkpoints(
    out_dir: Path = DEFAULT_CKPT_DIR,
    *,
    alphas: tuple[float, ...] = CKPT_ALPHAS,
    base_ckpt: Path = data.BASE_CKPT,
    rl_ckpt: Path = data.RL_CKPT,
) -> dict[float, Path]:
    """`interpolate` every non-trivial `alpha`, skipping any whose checkpoint+sidecar already
    exists with a matching `alpha` recorded (idempotent re-runs)."""
    out: dict[float, Path] = {}
    for a in alphas:
        out_path = Path(out_dir) / f"interp_a{alpha_tag(a)}.pt"
        sidecar_path = out_path.with_suffix(out_path.suffix + ".sha256.json")
        if out_path.exists() and sidecar_path.exists():
            recorded = json.loads(sidecar_path.read_text())
            if recorded.get("alpha") == a:
                out[a] = out_path
                continue
        out[a] = interpolate(base_ckpt, rl_ckpt, a, out_dir)
    return out


def run_interpolation(
    store: Store,
    cfg: Mapping[str, JSON],
    *,
    adapter_for: Callable[[float, Path | None], ModelAdapter],
    base_adapter_label: str = "base",
    rl_adapter_label: str = "rl",
    label_prefix: str = "interp",
    judge: Judge | None = None,
) -> dict[str, Any]:
    """Evaluate every `alpha` in `cfg.get("alphas", ALPHAS)`: `adapter_for(alpha, ckpt_path)` builds
    the model for that alpha (`ckpt_path is None` for `alpha in {0, 1}`, since those reuse the
    base/rl checkpoints the caller already has an adapter for); `dev`/`sealed` `gen` suites at
    `k=8` (pass@1) and the fixed 150-theorem `k128` subset at `k=128` (pass@k curve), per
    `INTERP_SPEC.md` §3. Suites are (re)built into `store` first (`eval_suites.build_all`).

    **Generalisation (`DANSTATE_PLAN.md` §7 WP-D6).** `base_adapter_label`/`rl_adapter_label`/
    `label_prefix` (defaults `"base"`/`"rl"`/`"interp"`, the old hard-coded literals -- now
    actually threaded into `interp_model_label` below, where they were previously accepted but
    unused) and `judge` (default `None`, i.e. `run_eval_suites`' own fresh `NDJudge()`, unchanged)
    let a caller run this on Dan's pair with family-prefixed labels and `judge.lean.LeanJudge()`.
    """

    def _int_cfg(key: str, default: int) -> int:
        v = cfg.get(key)
        return default if v is None else int(cast(float, v))

    def _float_cfg(key: str, default: float) -> float:
        v = cfg.get(key)
        return default if v is None else float(cast(float, v))

    excluder = (
        data.build_exclusion_index(cache_dir=cast("str | None", cfg.get("cache_dir")))
        if cfg.get("exclude_train")
        else None
    )
    seed = _int_cfg("seed", 0)
    suites = eval_suites.build_all(store, excluder=excluder, seed=seed)
    alphas = tuple(cast("list[float]", cfg.get("alphas", list(ALPHAS))))
    # `data.BASE_CKPT`/`RL_CKPT` are looked up here (call time), not via `build_checkpoints`'s own
    # stale default arguments, so a caller (or a test) can override them through `data` itself.
    ckpts = build_checkpoints(
        Path(cast("str | Path", cfg.get("ckpt_dir", DEFAULT_CKPT_DIR))),
        alphas=CKPT_ALPHAS,
        base_ckpt=Path(cast("str | Path", cfg.get("base_ckpt", data.BASE_CKPT))),
        rl_ckpt=Path(cast("str | Path", cfg.get("rl_ckpt", data.RL_CKPT))),
    )

    budget_s = cfg.get("budget_s")
    per_alpha: dict[str, Any] = {}
    for alpha in alphas:
        label = interp_model_label(
            alpha, base_label=base_adapter_label, rl_label=rl_adapter_label, prefix=label_prefix
        )
        ckpt_path = None if alpha in (0.0, 1.0) else ckpts[alpha]
        adapter = adapter_for(alpha, ckpt_path)
        per_alpha[alpha_tag(alpha)] = run_eval_suites(
            store,
            adapter,
            model_label=label,
            suites=suites,
            k_pass1=_int_cfg("k_pass1", 8),
            k_pass_k=_int_cfg("k_pass_k", 128),
            temperature=_float_cfg("temperature", 0.8),
            seed=seed,
            budget_s=cast("float | None", budget_s),
            judge=judge,
            only=eval_suites_from_cfg(cfg),
        )
    return {"alphas": [alpha_tag(a) for a in alphas], "runs": per_alpha, "notes": suites.notes}
