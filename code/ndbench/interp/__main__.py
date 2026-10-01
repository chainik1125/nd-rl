"""`python -m code.ndbench.interp <likelihood|fit-vectors|steer-sweep|interpolate|probes> --out
<store root> --config <yaml>` (`INTERP_SPEC.md`'s CLI section).

A separate entry point from `code.ndbench.cli` (which owns `python -m code.ndbench <subcommand>`;
this work package does not edit it, and this module neither imports from nor is imported by it):
`--config` here resolves `base`/`rl` adapters from plain `base_ckpt`/`rl_ckpt`/`device` config keys
(defaulting to `interp.data.BASE_CKPT`/`RL_CKPT`, CPU), not `cli.py`'s `configs/models.yaml`
label registry -- this package's runs always need *both* checkpoints (or explicitly skip one via
`skip_base`/`skip_rl`), so a single, simpler convention covers every subcommand.

Every subcommand prints its summary dict as JSON on stdout and otherwise only touches the `Store`
at `--out` (suite/run files for `interpolate`/`steer-sweep`, plus this package's own
`out/store/interp/<kind>/*` artefacts for `likelihood`/`fit-vectors`/`probes`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    device = str(cfg.get("device", "cpu"))
    base = None
    if not cfg.get("skip_base"):
        base = _build_adapter(str(cfg.get("base_ckpt", data.BASE_CKPT)), device)
    rl = None
    if need_rl and not cfg.get("skip_rl"):
        rl = _build_adapter(str(cfg.get("rl_ckpt", data.RL_CKPT)), device)
    return base, rl


def _print(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, default=str))


def cmd_likelihood(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    store = Store(Path(args.out))
    base, rl = _adapters_from_cfg(cfg)
    _print(likelihood.run_likelihood(store, cfg, base_adapter=base, rl_adapter=rl))


def cmd_fit_vectors(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    base, rl = _adapters_from_cfg(cfg)
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
    base, rl = _adapters_from_cfg(cfg)
    if base is None or rl is None:
        raise SystemExit("steer-sweep needs both a base and an rl adapter")
    vectors_dir = Path(cfg.get("vectors_dir", steering.DEFAULT_VECTOR_DIR))
    _print(
        steering.steering_sweep(
            store, cfg, base_adapter=base, rl_adapter=rl, vectors_dir=vectors_dir
        )
    )


def cmd_interpolate(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    store = Store(Path(args.out))
    device = str(cfg.get("device", "cpu"))

    def adapter_for(alpha: float, ckpt_path: Path | None) -> ModelAdapter:
        ckpt = (
            ckpt_path
            if ckpt_path is not None
            else (data.BASE_CKPT if alpha == 0.0 else data.RL_CKPT)
        )
        return _build_adapter(str(ckpt), device)

    _print(interpolate.run_interpolation(store, cfg, adapter_for=adapter_for))


def cmd_probes(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    store = Store(Path(args.out))
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
