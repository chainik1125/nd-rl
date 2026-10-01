"""`python -m code.ndbench.interp <likelihood|fit-vectors|steer-sweep|interpolate|probes> --out
<store root> --config <yaml>` (`INTERP_SPEC.md`'s CLI section).

A separate entry point from `code.ndbench.cli` (which owns `python -m code.ndbench <subcommand>`):
`--config` here resolves `base`/`rl` adapters from plain `base_ckpt`/`rl_ckpt`/`device` config keys
(defaulting to `interp.data.BASE_CKPT`/`RL_CKPT`, CPU) -- this is the *original*, absgpt-only
convention (`_adapters_from_cfg`/`_build_adapter`), kept byte-for-byte unchanged below so every
pre-existing config/test takes the exact same code path it always did.

**Model-label generalisation (`DANSTATE_PLAN.md` §7 WP-D6).** A config may instead give
`base_model`/`rl_model` (labels resolved through `configs/models.yaml`, exactly like
`cli._resolve_model` -- `_resolve_labeled_pair` below imports and reuses `cli.py`'s own
`_load_models_registry`/`_resolve_model`/`_build_judge` directly, read-only, rather than
re-deriving the same registry/sha256-check/judge-instantiation logic a second time) and a `judge`
key (`"nd"`/`"lean"`; default `"nd"`, matching every existing config's implicit behaviour). Each
`cmd_*` below checks for `base_model`/`rl_model` first and only falls back to the original
`base_ckpt`/`rl_ckpt` path when neither is given, so the original path's code is never reached by
a labelled config and vice versa -- the two are fully disjoint, not a merged/overridden one.
`likelihood`/`fit-vectors`/`probes` dispatch to this module's Dan-specific
`interp.danstate_support` functions when the resolved adapter is a `DanStateAdapter` (its
"neutral"/activation-shape assumptions do not apply to Dan's models, module docstring there);
`steer-sweep`/`interpolate` need no such dispatch -- `interp/steering.py`'s `steering_sweep` and
`interp/interpolate.py`'s `run_interpolation` are already adapter-agnostic, now generalised with
`judge`/label parameters instead.

Every subcommand prints its summary dict as JSON on stdout and otherwise only touches the `Store`
at `--out` (suite/run files for `interpolate`/`steer-sweep`, plus this package's own
`out/store/interp/<kind>/*` artefacts for `likelihood`/`fit-vectors`/`probes`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import yaml

from ..adapters.base import ModelAdapter
from ..store import Store
from . import data, interpolate, likelihood, probes, steering


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"config {path} must be a YAML mapping at the top level")
    return cfg


def _build_adapter(ckpt: str, device: str) -> ModelAdapter:
    from ..adapters.absgpt import AbsGptAdapter  # local import: keeps --help torch-free

    return AbsGptAdapter(ckpt, device=device)


def _adapters_from_cfg(
    cfg: dict[str, Any], *, need_rl: bool = True
) -> tuple[ModelAdapter | None, ModelAdapter | None]:
    """The original, absgpt-only resolution (module docstring): unchanged."""
    device = str(cfg.get("device", "cpu"))
    base = None
    if not cfg.get("skip_base"):
        base = _build_adapter(str(cfg.get("base_ckpt", data.BASE_CKPT)), device)
    rl = None
    if need_rl and not cfg.get("skip_rl"):
        rl = _build_adapter(str(cfg.get("rl_ckpt", data.RL_CKPT)), device)
    return base, rl


def _is_labeled_cfg(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("base_model") or cfg.get("rl_model"))


def _resolve_labeled_pair(
    cfg: dict[str, Any], *, need_rl: bool = True
) -> tuple[ModelAdapter | None, ModelAdapter | None, str, str, str | None, str | None, str | None]:
    """`cfg["base_model"]`/`cfg["rl_model"]` resolved through `configs/models.yaml` (module
    docstring): returns `(base_adapter_or_None, rl_adapter_or_None, base_label, rl_label,
    family_or_None, base_ckpt_or_None, rl_ckpt_or_None)`. `base_label`/`rl_label` default to the
    literal `"base"`/`"rl"` only if the corresponding `*_model` key is missing (so a config can
    give just one side, e.g. `fit-vectors`/`probes`' `need_rl=True` always needs both -- callers
    that can tolerate a one-sided pair pass `need_rl=False`, same convention as
    `_adapters_from_cfg`). `cfg["model_kwargs"]` (default `{}`) is layered on top of each
    registry entry's own kwargs, exactly as `--model-arg` does for `cli.py cmd_run`; `device`
    (default `"cpu"`) is injected into both unless `model_kwargs` already names one."""
    from ..cli import DEFAULT_MODELS_CONFIG, _load_models_registry, _resolve_model

    registry = _load_models_registry(str(cfg.get("models_config", DEFAULT_MODELS_CONFIG)))
    overrides = {str(k): str(v) for k, v in dict(cfg.get("model_kwargs") or {}).items()}
    overrides.setdefault("device", str(cfg.get("device", "cpu")))

    base_adapter: ModelAdapter | None = None
    base_label, base_ckpt, family = "base", None, None
    if cfg.get("base_model") and not cfg.get("skip_base"):
        base_label = str(cfg["base_model"])
        adapter_cls, kwargs, _store_label = _resolve_model(base_label, overrides, registry)
        base_ckpt = cast("str | None", registry.get(base_label, {}).get("ckpt"))
        family = cast("str | None", registry.get(base_label, {}).get("family"))
        base_adapter = adapter_cls(**kwargs)

    rl_adapter: ModelAdapter | None = None
    rl_label, rl_ckpt = "rl", None
    if need_rl and cfg.get("rl_model") and not cfg.get("skip_rl"):
        rl_label = str(cfg["rl_model"])
        adapter_cls, kwargs, _store_label = _resolve_model(rl_label, overrides, registry)
        rl_ckpt = cast("str | None", registry.get(rl_label, {}).get("ckpt"))
        family = family or cast("str | None", registry.get(rl_label, {}).get("family"))
        rl_adapter = adapter_cls(**kwargs)

    return base_adapter, rl_adapter, base_label, rl_label, family, base_ckpt, rl_ckpt


def _resolve_judge(cfg: dict[str, Any], out: str) -> Any:
    """`cfg["judge"]` (default `"nd"`, matching every pre-existing config's implicit `NDJudge()`)
    via `cli._build_judge` -- shared with `cli.py cmd_run`'s own `--judge` resolution, so
    `NDBENCH_LEAN_WORKERS`/the cache-dir convention stay consistent across both entry points."""
    from ..cli import _build_judge

    return _build_judge(str(cfg.get("judge", "nd")), out)


def _is_danstate(adapter: ModelAdapter | None) -> bool:
    from ..adapters.danstate import DanStateAdapter  # local import: keeps --help torch-free

    return isinstance(adapter, DanStateAdapter)


def _print(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, default=str))


def cmd_likelihood(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    store = Store(Path(args.out))
    if _is_labeled_cfg(cfg):
        base, rl, *_rest = _resolve_labeled_pair(cfg)
        if _is_danstate(base) or _is_danstate(rl):
            from .danstate_support import run_likelihood_dan

            _print(run_likelihood_dan(store, cfg, base_adapter=base, rl_adapter=rl))
            return
        _print(likelihood.run_likelihood(store, cfg, base_adapter=base, rl_adapter=rl))
        return
    base, rl = _adapters_from_cfg(cfg)
    _print(likelihood.run_likelihood(store, cfg, base_adapter=base, rl_adapter=rl))


def cmd_fit_vectors(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    if _is_labeled_cfg(cfg):
        base, rl, *_rest = _resolve_labeled_pair(cfg)
        if base is None or rl is None:
            raise SystemExit("fit-vectors needs both a base and an rl adapter")
        if _is_danstate(base) or _is_danstate(rl):
            from ..adapters.danstate import DanStateAdapter
            from .danstate_support import fit_diff_vectors_dan, reference_proofs_from_cfg

            store = Store(Path(args.out))
            n_pairs = int(cfg.get("n_pairs", 2000) or 0)
            proofs, _notes = reference_proofs_from_cfg(
                store, cfg, n=n_pairs, seed=int(cfg.get("seed", 0) or 0)
            )
            pairs = [(p.sequent, p.body) for p in proofs]
            layers = tuple(cast("list[int]", cfg.get("layers", list(range(6)))))
            out_dir = Path(cfg.get("out_dir", steering.DEFAULT_VECTOR_DIR))
            results = fit_diff_vectors_dan(
                cast(DanStateAdapter, base),
                cast(DanStateAdapter, rl),
                pairs,
                layers,
                out_dir=out_dir,
            )
            _print(
                {
                    "n_pairs": len(pairs),
                    "vectors": [
                        {
                            "layer": r.layer,
                            "position_type": "action",
                            "norm": r.norm,
                            "mean_residual_norm": r.mean_residual_norm,
                            "path": str(r.path),
                        }
                        for r in results
                    ],
                }
            )
            return
        _fit_vectors_absgpt(cfg, base, rl)
        return
    base, rl = _adapters_from_cfg(cfg)
    _fit_vectors_absgpt(cfg, base, rl)


def _fit_vectors_absgpt(
    cfg: dict[str, Any], base: ModelAdapter | None, rl: ModelAdapter | None
) -> None:
    if base is None or rl is None:
        raise SystemExit("fit-vectors needs both a base and an rl adapter")
    excluder = (
        data.build_exclusion_index(cache_dir=cfg.get("cache_dir"))
        if cfg.get("exclude_train", True)
        else None
    )
    n_pairs = int(cfg.get("n_pairs", 2000))
    proofs = data.load_neutral_proofs(
        n=n_pairs,
        excluder=excluder,
        seed=int(cfg.get("seed", 0) or 0),
        scan_limit=cfg.get("scan_limit"),
    )
    pairs = [(p.sequent, p.body) for p in proofs]
    layers = tuple(cfg.get("layers", steering.LAYERS))
    position_types = tuple(cfg.get("position_types", ("proof",)))
    out_dir = Path(cfg.get("out_dir", steering.DEFAULT_VECTOR_DIR))
    results = steering.fit_diff_vectors(base, rl, pairs, layers, position_types, out_dir=out_dir)
    _print(
        {
            "n_pairs": len(pairs),
            "vectors": [
                {
                    "layer": r.layer,
                    "position_type": r.position_type,
                    "norm": r.norm,
                    "mean_residual_norm": r.mean_residual_norm,
                    "path": str(r.path),
                }
                for r in results
            ],
        }
    )


def cmd_steer_sweep(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    store = Store(Path(args.out))
    judge = _resolve_judge(cfg, args.out)
    if _is_labeled_cfg(cfg):
        base, rl, base_label, rl_label, _family, _bc, _rc = _resolve_labeled_pair(cfg)
        if base is None or rl is None:
            raise SystemExit("steer-sweep needs both a base and an rl adapter")
        vectors_dir = Path(cfg.get("vectors_dir", steering.DEFAULT_VECTOR_DIR))
        _print(
            steering.steering_sweep(
                store,
                cfg,
                base_adapter=base,
                rl_adapter=rl,
                vectors_dir=vectors_dir,
                judge=judge,
                base_label=base_label,
                rl_label=rl_label,
            )
        )
        return
    base, rl = _adapters_from_cfg(cfg)
    if base is None or rl is None:
        raise SystemExit("steer-sweep needs both a base and an rl adapter")
    vectors_dir = Path(cfg.get("vectors_dir", steering.DEFAULT_VECTOR_DIR))
    _print(
        steering.steering_sweep(
            store, cfg, base_adapter=base, rl_adapter=rl, vectors_dir=vectors_dir, judge=judge
        )
    )


def cmd_interpolate(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    store = Store(Path(args.out))
    judge = _resolve_judge(cfg, args.out)
    device = str(cfg.get("device", "cpu"))

    if _is_labeled_cfg(cfg):
        from ..cli import DEFAULT_MODELS_CONFIG, _load_models_registry, _resolve_model

        registry = _load_models_registry(str(cfg.get("models_config", DEFAULT_MODELS_CONFIG)))
        base_label = str(cfg.get("base_model"))
        rl_label = str(cfg.get("rl_model"))
        overrides = {str(k): str(v) for k, v in dict(cfg.get("model_kwargs") or {}).items()}
        overrides.setdefault("device", device)
        adapter_cls, base_kwargs, _bl = _resolve_model(base_label, overrides, registry)
        _, rl_kwargs, _rl = _resolve_model(rl_label, overrides, registry)
        base_ckpt = str(cfg.get("base_ckpt", base_kwargs["ckpt"]))
        rl_ckpt = str(cfg.get("rl_ckpt", rl_kwargs["ckpt"]))
        family = registry.get(base_label, {}).get("family") or registry.get(rl_label, {}).get(
            "family"
        )
        prefix = f"{family}-interp" if family else "interp"
        default_ckpt_dir = store.store_dir / "cache" / "interp_ckpts" / str(family or "interp")

        def adapter_for(alpha: float, ckpt_path: Path | None) -> ModelAdapter:
            kwargs = dict(base_kwargs)
            if ckpt_path is not None:
                kwargs["ckpt"] = str(ckpt_path)
            else:
                kwargs["ckpt"] = base_ckpt if alpha == 0.0 else rl_ckpt
            return adapter_cls(**kwargs)

        labeled_cfg = {
            **cfg,
            "base_ckpt": base_ckpt,
            "rl_ckpt": rl_ckpt,
            "ckpt_dir": str(cfg.get("ckpt_dir", default_ckpt_dir)),
        }
        _print(
            interpolate.run_interpolation(
                store,
                labeled_cfg,
                adapter_for=adapter_for,
                base_adapter_label=base_label,
                rl_adapter_label=rl_label,
                label_prefix=prefix,
                judge=judge,
            )
        )
        return

    def adapter_for(alpha: float, ckpt_path: Path | None) -> ModelAdapter:
        ckpt = (
            ckpt_path
            if ckpt_path is not None
            else (data.BASE_CKPT if alpha == 0.0 else data.RL_CKPT)
        )
        return _build_adapter(str(ckpt), device)

    _print(interpolate.run_interpolation(store, cfg, adapter_for=adapter_for, judge=judge))


def cmd_probes(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    store = Store(Path(args.out))
    if _is_labeled_cfg(cfg):
        base, rl, *_rest = _resolve_labeled_pair(cfg)
        if _is_danstate(base) or _is_danstate(rl):
            from .danstate_support import run_probes_dan

            _print(run_probes_dan(store, cfg, base_adapter=base, rl_adapter=rl))
            return
        _print(probes.run_probes(store, cfg, base_adapter=base, rl_adapter=rl))
        return
    base, rl = _adapters_from_cfg(cfg)
    _print(probes.run_probes(store, cfg, base_adapter=base, rl_adapter=rl))


def cmd_probe_baselines(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    _print(probes.run_position_baselines(Store(Path(args.out)), cfg))


_COMMANDS: dict[str, Any] = {
    "likelihood": cmd_likelihood,
    "fit-vectors": cmd_fit_vectors,
    "steer-sweep": cmd_steer_sweep,
    "interpolate": cmd_interpolate,
    "probes": cmd_probes,
    "probe-baselines": cmd_probe_baselines,
}


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="out", help="Store root (default: ./out)")
    common.add_argument("--config", default=None, help="YAML config for this subcommand")
    parser = argparse.ArgumentParser(prog="python -m code.ndbench.interp")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in _COMMANDS.items():
        sp = sub.add_parser(name, parents=[common])
        sp.set_defaults(func=fn)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
