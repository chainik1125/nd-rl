"""ndbench command-line interface: `python -m code.ndbench <subcommand> ...` (plan §1).

Subcommands: `build | run | edl | throughput | import-legacy | budget | dash`. Everything this
module imports at module level is cheap (argparse, `core.io`, `store.Store`, the plain-dict lazy
registries); every work-package-specific implementation (an adapter, a judge, a suite builder, an
analysis view, WP1's `tools/throughput`/`runner/gpu`, WP5's `edl.py`, WP6's `dash/build.py`) is
resolved with `_resolve` only once its subcommand actually runs, so `--help` (and importing this
module at all) never requires torch, a GPU, or a work package that has not landed yet.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from .adapters.base import ADAPTERS
from .analysis import VIEWS as ANALYSIS_VIEWS
from .judge import JUDGES
from .store import Store
from .suites import BUILDERS

#: Default `Store` root. Matches plan §1's `Outputs: .../bench/out/{suites,store,cache,dash,logs}`;
#: overridable with `--out` on every subcommand that touches a `Store`.
DEFAULT_OUT = "out"


def _resolve(path: str) -> Any:
    """Import `"module:attr"` and return the attribute (the lazy-registry convention used
    throughout ndbench: `adapters.base.ADAPTERS`, `judge.JUDGES`, `suites.BUILDERS`,
    `analysis.VIEWS`)."""
    module_name, sep, attr = path.partition(":")
    if not sep:
        raise ValueError(f"expected 'module:attr', got {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _resolve_registry(registry: dict[str, str], name: str, kind: str) -> Any:
    if name not in registry:
        raise SystemExit(f"unknown {kind} {name!r}; known: {sorted(registry)}")
    try:
        return _resolve(registry[name])
    except ModuleNotFoundError as e:
        raise SystemExit(f"{kind} {name!r} ({registry[name]}) is not implemented yet: {e}") from e


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    text = Path(path).read_text()
    cfg = yaml.safe_load(text) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"config {path} must be a YAML mapping at the top level")
    return cfg


def _parse_kv_list(items: Sequence[str] | None) -> dict[str, str]:
    """`["a=1", "b=2"]` -> `{"a": "1", "b": "2"}`, for simple `--model-arg`/`--judge-arg`
    overrides."""
    out: dict[str, str] = {}
    for item in items or ():
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {item!r}")
        out[key] = value
    return out


# --------------------------------------------------------------------------------------- build


def cmd_build(args: argparse.Namespace) -> None:
    from dataclasses import asdict

    from .core.io import git_commit_sha, sha256_str, utc_now_iso
    from .core.types import SuiteManifest

    cfg = _load_config(args.config)
    build_fn = _resolve_registry(BUILDERS, args.suite, "suite builder")
    certifier = _resolve(cfg["certifier"]) if "certifier" in cfg else None
    excluder = _resolve(cfg["excluder"]) if "excluder" in cfg else None
    problems, stats = build_fn(cfg, certifier=certifier, excluder=excluder, rng_seed=args.seed)

    items_json = [p.to_json() for p in problems]
    manifest = SuiteManifest(
        suite=args.suite,
        version=args.version,
        builder=BUILDERS[args.suite],
        config=cfg,
        seed=args.seed,
        created_utc=utc_now_iso(),
        git_commit=git_commit_sha(),
        n_items=len(problems),
        items_sha256=sha256_str(json.dumps(items_json, sort_keys=True)),
        excl_sources=dict(cfg.get("excl_sources", {})),
        stats=stats,
    )
    store = Store(Path(args.out))
    store.write_suite(problems, manifest)
    print(json.dumps({"suite": args.suite, "version": args.version, **asdict(manifest)}, indent=2))


# ----------------------------------------------------------------------------------------- run


def cmd_run(args: argparse.Namespace) -> None:
    from .runner.adaptive import AdaptiveRule
    from .runner.runner import RunSpec, run

    model_name, _, model_arg = args.model.partition(":")
    adapter_cls = _resolve_registry(ADAPTERS, model_name, "adapter")
    model_kwargs = _parse_kv_list(args.model_arg)
    if model_arg:
        model_kwargs.setdefault("ckpt", model_arg)
    adapter = adapter_cls(**model_kwargs)

    judge_cls = _resolve_registry(JUDGES, args.judge, "judge")
    judge = judge_cls()

    adaptive = None
    if args.adaptive:
        chunk, n_max, stop_c = (int(x) for x in args.adaptive.split(","))
        adaptive = AdaptiveRule(chunk=chunk, n_max=n_max, stop_c=stop_c)
    pids = tuple(args.pids.split(",")) if args.pids else None

    spec = RunSpec(
        suite=args.suite,
        version=args.version,
        model=args.model,
        k=args.k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        adaptive=adaptive,
        pids=pids,
        keep_samples=not args.no_keep_samples,
        budget_s=args.budget_s,
    )
    store = Store(Path(args.out))
    summary = run(spec, adapter, judge, store)
    adapter.close()
    print(json.dumps(summary, indent=2))


# ----------------------------------------------------------------------------------------- edl


def cmd_edl(args: argparse.Namespace) -> None:
    build_classes = _resolve_registry(
        {"edl": "code.ndbench.edl:build_classes"}, "edl", "edl module"
    )
    run_edl = _resolve("code.ndbench.edl:run_edl")
    adapter_cls = _resolve_registry(ADAPTERS, args.model, "adapter")

    cfg = _load_config(args.config)
    store = Store(Path(args.out))
    problems, stats = build_classes(cfg)
    adapter = adapter_cls(**_parse_kv_list(args.model_arg))
    rows = run_edl(
        adapter,
        problems,
        lrs=[float(x) for x in args.lrs.split(",")],
        orders=args.orders,
        n=args.n,
        m=args.m,
        ckpts=args.ckpts,
    )
    run_id = f"edl-{args.model}-{args.n}-{args.m}"
    store.append_edl(run_id, rows)
    adapter.close()
    print(json.dumps({"run_id": run_id, "n_rows": len(rows), "build_stats": stats}, indent=2))


# --------------------------------------------------------------------------------- throughput


def cmd_throughput(args: argparse.Namespace) -> None:
    throughput = _resolve("code.ndbench.tools.throughput:throughput")
    result = throughput(
        models=args.models.split(","),
        sets=args.sets.split(","),
        batches=[int(x) for x in args.batches.split(",")],
    )
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------- import-legacy


def cmd_import_legacy(args: argparse.Namespace) -> None:
    import_legacy = _resolve_registry(
        {"import-legacy": "code.ndbench.suites.existing:import_legacy"},
        "import-legacy",
        "importer",
    )
    store = Store(Path(args.out))
    result = import_legacy(store)
    print(json.dumps(result, indent=2))


# --------------------------------------------------------------------------------------- dash


def cmd_dash(args: argparse.Namespace) -> None:
    build = _resolve_registry(
        {"dash": "code.ndbench.dash.build:build"}, "dash", "dashboard builder"
    )
    store = Store(Path(args.out))
    build(store, Path(args.dash_dir))
    print(json.dumps({"dash_dir": args.dash_dir, "views": sorted(ANALYSIS_VIEWS)}, indent=2))


# ------------------------------------------------------------------------------------- budget


def cmd_budget(args: argparse.Namespace) -> None:
    """Report GPU-hours spent (per `Store.ledger_total_s`) against `--cap-h`, and (with
    `--est-h`) whether plan §4's stop rule (`spent + 1.5 * estimate <= cap`) allows the next job.
    """
    store = Store(Path(args.out))
    spent_h = store.ledger_total_s() / 3600.0
    result: dict[str, Any] = {
        "spent_h": spent_h,
        "cap_h": args.cap_h,
        "remaining_h": args.cap_h - spent_h,
    }
    if args.est_h is not None:
        result["est_h"] = args.est_h
        result["would_fit"] = spent_h + 1.5 * args.est_h <= args.cap_h
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ndbench", description="ndbench: capability benchmarks for ND provers"
    )
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build a suite and write it to the store")
    b.add_argument("--suite", required=True, choices=sorted(BUILDERS))
    b.add_argument("--version", required=True)
    b.add_argument("--config", default=None, help="YAML config path")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default=DEFAULT_OUT)
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("run", help="sample a suite against a model and judge every attempt")
    r.add_argument("--suite", required=True)
    r.add_argument("--version", required=True)
    r.add_argument("--model", required=True, help="adapter name, optionally 'name:ckpt_path'")
    r.add_argument(
        "--model-arg", action="append", help="key=value, forwarded to the adapter constructor"
    )
    r.add_argument("--judge", default="nd", choices=sorted(JUDGES))
    r.add_argument("--k", type=int, required=True)
    r.add_argument("--temperature", type=float, default=0.8)
    r.add_argument("--max-new-tokens", type=int, default=None, dest="max_new_tokens")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--adaptive", default=None, help="chunk,n_max,stop_c")
    r.add_argument("--pids", default=None, help="comma-separated pid subset")
    r.add_argument("--budget-s", type=float, default=None, dest="budget_s")
    r.add_argument("--no-keep-samples", action="store_true")
    r.add_argument("--out", default=DEFAULT_OUT)
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("edl", help="run the excess-description-length protocol (plan §B7)")
    e.add_argument("--model", required=True)
    e.add_argument("--model-arg", action="append")
    e.add_argument("--config", default=None)
    e.add_argument("--lrs", default="1e-4,3e-4,1e-3")
    e.add_argument("--orders", type=int, default=2)
    e.add_argument("--n", type=int, default=64)
    e.add_argument("--m", type=int, default=64)
    e.add_argument("--ckpts", default=None)
    e.add_argument("--out", default=DEFAULT_OUT)
    e.set_defaults(func=cmd_edl)

    t = sub.add_parser("throughput", help="measure samples/s for a grid of models/sets/batch sizes")
    t.add_argument("--models", required=True)
    t.add_argument("--sets", required=True)
    t.add_argument("--batches", required=True)
    t.set_defaults(func=cmd_throughput)

    il = sub.add_parser("import-legacy", help="import pre-ndbench run results into the store (WP2)")
    il.add_argument("--out", default=DEFAULT_OUT)
    il.set_defaults(func=cmd_import_legacy)

    bg = sub.add_parser("budget", help="report GPU-hour ledger spend against a cap (plan §4)")
    bg.add_argument("--out", default=DEFAULT_OUT)
    bg.add_argument("--cap-h", type=float, default=9.5, dest="cap_h")
    bg.add_argument("--est-h", type=float, default=None, dest="est_h")
    bg.set_defaults(func=cmd_budget)

    d = sub.add_parser("dash", help="render the analysis dashboard (WP6)")
    d.add_argument("--out", default=DEFAULT_OUT)
    d.add_argument("--dash-dir", required=True, dest="dash_dir")
    d.set_defaults(func=cmd_dash)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Parse `argv` and run the selected subcommand.

    Argument-parsing errors and `--help` raise `SystemExit` directly (the normal argparse
    behaviour, e.g. for `python -m code.ndbench --help`). Once a subcommand actually runs, a
    friendly `SystemExit(message)` (e.g. from `_resolve_registry` reporting an unimplemented work
    package) or any other exception is instead reported and turned into a return code, so calling
    `main(...)` from Python (as the test suite does) gets an int back rather than a raised
    `SystemExit`.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except SystemExit as e:
        if e.code is None:
            return 0
        if isinstance(e.code, int):
            return e.code
        print(e.code, file=sys.stderr)
        return 1
    except Exception as e:  # pragma: no cover - top-level CLI error reporting
        print(f"ndbench {args.command}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
