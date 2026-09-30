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
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from .adapters.base import ADAPTERS
from .analysis import VIEWS as ANALYSIS_VIEWS
from .judge import JUDGES
from .store import Store
from .suites import BUILDERS

#: Default `Store` root. Matches plan §1's `Outputs: .../bench/out/{suites,store,cache,dash,logs}`;
#: overridable with `--out` on every subcommand that touches a `Store`.
DEFAULT_OUT = "out"

#: Plan §4's hard GPU-hour cap.
DEFAULT_CAP_H = 9.5

#: `code/ndbench/cli.py` -> `code/ndbench` -> `code` -> repo/worktree root, so this resolves
#: correctly regardless of the process's cwd (plan §A1 always runs from the worktree root, but
#: tests may not).
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default model-label registry (fix list item 6): `python -m code.ndbench {run,edl,throughput}
#: --model <label>` resolves `<label>` (e.g. `base`/`rl`/`armC`) through this file before falling
#: back to the plain `"adapter_name"` / `"adapter_name:ckpt_path"` convention. Overridable with
#: `--models-config` (e.g. in a test, to avoid touching the real checkpoints this file points at).
DEFAULT_MODELS_CONFIG = str(
    _REPO_ROOT / "code/experiments/current/benchmarking_20260930/configs/models.yaml"
)


def _resolve(path: str) -> Any:
    """Import `"module:attr"` and return the attribute (the lazy-registry convention used
    throughout ndbench: `adapters.base.ADAPTERS`, `judge.JUDGES`, `suites.BUILDERS`,
    `analysis.VIEWS`)."""
    module_name, sep, attr = path.partition(":")
    if not sep:
        raise ValueError(f"expected 'module:attr', got {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _resolve_instance(spec: str) -> Any:
    """Resolve `cfg["certifier"]`/`cfg["excluder"]` (a `"module:attr"` string) via `_resolve`, then
    instantiate it with no arguments if it resolved to a *class* rather than an already-built
    instance. This lets a `build` config name a zero-arg-constructible class directly (e.g.
    `certifier: code.ndbench.logic.certify:MinlenCertifier`, used by `configs/textbook_dev.yaml`
    for WP8 integration fix list item 10) without needing a hand-written module-level singleton or
    factory function just to make it `"module:attr"`-resolvable -- `build_fn(cfg, certifier=...,
    ...)` receives the resolved value as-is either way, since `Certifier`/`Excluder` are structural
    Protocols this function has no other way to distinguish from a plain class object."""
    obj = _resolve(spec)
    return obj() if isinstance(obj, type) else obj


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


# ------------------------------------------------------------------------------- model registry


def _load_models_registry(path: str | None) -> dict[str, dict[str, Any]]:
    """`configs/models.yaml`'s labels (fix list item 6): `{label: {"adapter": ..., "ckpt": ...,
    "sha256_prefix": ...}}`. A missing file (e.g. `--models-config` left at its default outside the
    worktree, or a test pointing nowhere on purpose) is not an error -- it just means no label
    resolves through the registry, so every `--model` value falls back to the pre-existing bare
    `"adapter_name"` / `"adapter_name:ckpt_path"` convention."""
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"models config {path} must be a YAML mapping at the top level")
    return cast("dict[str, dict[str, Any]]", data)


def _check_model_sha(label: str, ckpt: str, sha_prefix: object) -> None:
    """Verify `ckpt`'s sha256 starts with `sha_prefix` (fix list item 6's "sha check"); a no-op if
    the registry entry has no `sha256_prefix`. Raises a friendly `SystemExit` on mismatch rather
    than silently loading the wrong checkpoint under a trusted label."""
    if not sha_prefix:
        return
    from .core.io import sha256_file

    actual = sha256_file(Path(ckpt))
    if not actual.startswith(str(sha_prefix)):
        raise SystemExit(
            f"model label {label!r}: checkpoint {ckpt!r} sha256 {actual[:16]}... does not start "
            f"with the configured prefix {sha_prefix!r} (configs/models.yaml)"
        )


def _resolve_model(
    value: str, model_kwargs_overrides: Mapping[str, str], registry: Mapping[str, dict[str, Any]]
) -> tuple[Any, dict[str, Any], str]:
    """`--model <value>` -> `(adapter_cls, adapter_kwargs, store_label)`.

    If `value` names a label in `registry` (`configs/models.yaml`), resolves through it: the
    adapter class its `adapter` field names (via `ADAPTERS`), its `ckpt` path (sha256-checked
    against `sha256_prefix` when given), and any `model_kwargs_overrides` (`--model-arg`) layered
    on top. `store_label` is then exactly `value` -- the YAML label itself, unchanged -- so a
    `Store` run's `model` field is literally `"base"`/`"rl"`/etc., which is what `analysis.yue.
    _pick_pair` looks for by name. Otherwise falls back to the pre-existing `"name"` /
    `"name:ckpt_path"` convention (`store_label = value`, unchanged from before this fix)."""
    if value in registry:
        entry = registry[value]
        adapter_name = str(entry.get("adapter", "absgpt"))
        adapter_cls = _resolve_registry(ADAPTERS, adapter_name, "adapter")
        ckpt = entry.get("ckpt")
        if not ckpt:
            raise SystemExit(f"model label {value!r} in models config has no 'ckpt'")
        _check_model_sha(value, str(ckpt), entry.get("sha256_prefix"))
        kwargs: dict[str, Any] = {"ckpt": ckpt, **model_kwargs_overrides}
        return adapter_cls, kwargs, value
    model_name, _, model_path = value.partition(":")
    adapter_cls = _resolve_registry(ADAPTERS, model_name, "adapter")
    kwargs = dict(model_kwargs_overrides)
    if model_path:
        kwargs.setdefault("ckpt", model_path)
    return adapter_cls, kwargs, value


def _guard_cuda_device(*devices: str | None) -> None:
    """Fix list item 8's CLI-level guard: before a subcommand resolves an adapter (which, for
    `absgpt`/`lean_seq`, imports torch at module level) or calls `tools.throughput.throughput`
    (same), refuse a `"cuda"`-ish `--device`/`--model-arg device=...` value unless
    `CUDA_VISIBLE_DEVICES=4` is already set in the environment. Launching the whole process with
    `CUDA_VISIBLE_DEVICES=4` set from the start is the only supported way to use the GPU (plan §A2:
    GPUs 0-3 belong to another user and must never be touched); `runner.gpu.lease` enforces the
    same invariant again at lease time, but that check runs only after torch has already been
    imported to build the adapter, which is too late to stop the import itself from touching a
    device other than 4 in this process (e.g. via `torch.cuda.is_available()` probing)."""
    if not any(d is not None and "cuda" in d for d in devices):
        return
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current != "4":
        raise SystemExit(
            f"a 'cuda' device was requested but CUDA_VISIBLE_DEVICES={current!r} (must be '4', "
            "already set before this process starts, e.g. "
            "`CUDA_VISIBLE_DEVICES=4 python -m code.ndbench ...`). GPUs 0-3 belong to another user "
            "and must never be touched (plan §A2)."
        )


# --------------------------------------------------------------------------------------- build


def cmd_build(args: argparse.Namespace) -> None:
    from dataclasses import asdict

    from .core.io import git_commit_sha, sha256_str, utc_now_iso
    from .core.types import SuiteManifest

    cfg = _load_config(args.config)
    build_fn = _resolve_registry(BUILDERS, args.suite, "suite builder")
    certifier = _resolve_instance(cfg["certifier"]) if "certifier" in cfg else None
    excluder = _resolve_instance(cfg["excluder"]) if "excluder" in cfg else None
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

    model_kwargs_overrides = _parse_kv_list(args.model_arg)
    _guard_cuda_device(args.device, model_kwargs_overrides.get("device"))
    registry = _load_models_registry(args.models_config)
    adapter_cls, model_kwargs, store_label = _resolve_model(
        args.model, model_kwargs_overrides, registry
    )
    if args.device is not None:
        model_kwargs["device"] = args.device
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
        # Exactly the resolved label (e.g. "base"/"rl" from configs/models.yaml), not
        # "absgpt:/path/...": `analysis.yue._pick_pair` looks for Store runs literally labelled
        # "base"/"rl" (fix list item 6).
        model=store_label,
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
    model_kwargs_overrides = _parse_kv_list(args.model_arg)
    _guard_cuda_device(args.device, model_kwargs_overrides.get("device"))
    build_classes = _resolve_registry(
        {"edl": "code.ndbench.edl:build_classes"}, "edl", "edl module"
    )
    run_edl = _resolve("code.ndbench.edl:run_edl")
    registry = _load_models_registry(args.models_config)
    adapter_cls, model_kwargs, store_label = _resolve_model(
        args.model, model_kwargs_overrides, registry
    )
    if args.device is not None:
        model_kwargs["device"] = args.device

    cfg = _load_config(args.config)
    store = Store(Path(args.out))
    # `--out` must reach `build_classes`, which reads suite-backed classes (families/laws/glue)
    # from the `Store` -- previously `build_classes(cfg)` was called without a `store`, so it built
    # its *own* `Store(Path(cfg.get("out", "out")))` pointed at the CLI's `--out` default ("out")
    # rather than wherever `--out` actually pointed, silently missing every suite the caller built.
    problems, stats = build_classes(cfg, store=store)
    adapter = adapter_cls(**model_kwargs)
    rows = run_edl(
        adapter,
        problems,
        lrs=[float(x) for x in args.lrs.split(",")],
        orders=args.orders,
        n=args.n,
        m=args.m,
        ckpts=args.ckpts,
    )
    run_id = f"edl-{store_label}-{args.n}-{args.m}"
    store.append_edl(run_id, rows)
    adapter.close()
    print(json.dumps({"run_id": run_id, "n_rows": len(rows), "build_stats": stats}, indent=2))


# --------------------------------------------------------------------------------- throughput


def _resolve_throughput_model_spec(spec: str, registry: Mapping[str, dict[str, Any]]) -> str:
    """`tools.throughput.throughput`'s `models` entries are `"label=ckpt_path"` (or a bare path);
    resolve a `configs/models.yaml` label (fix list item 6) into that form, sha256-checked, else
    pass `spec` through unchanged."""
    if spec in registry:
        entry = registry[spec]
        ckpt = entry.get("ckpt")
        if not ckpt:
            raise SystemExit(f"model label {spec!r} in models config has no 'ckpt'")
        _check_model_sha(spec, str(ckpt), entry.get("sha256_prefix"))
        return f"{spec}={ckpt}"
    return spec


def cmd_throughput(args: argparse.Namespace) -> None:
    _guard_cuda_device(args.device)
    throughput = _resolve("code.ndbench.tools.throughput:throughput")
    registry = _load_models_registry(args.models_config)
    model_specs = [_resolve_throughput_model_spec(m, registry) for m in args.models.split(",")]
    store = Store(Path(args.out))
    result = throughput(
        models=model_specs,
        sets=args.sets.split(","),
        batches=[int(x) for x in args.batches.split(",")],
        device=args.device,
        store=store,
        suite_version=args.suite_version,
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


def _budget_schedule_table(
    jobs: Sequence[Mapping[str, Any]], *, spent_h: float, cap_h: float
) -> dict[str, Any]:
    """Fix list item 10's schedule dry-run: walk `jobs` (as `configs/schedule.yaml`'s `jobs` list)
    in order, applying plan §4's stop rule at each one -- "continue only if spent + 1.5*estimate <=
    cap, else skip to the next job that fits (in order)" -- against a *projected* cumulative spend
    (each accepted job's own `est_min` estimate, not a real ledger measurement, since this never
    touches the GPU). A job with no `est_min` (e.g. the `R` reserve placeholder, whose real cost is
    "whatever reruns turn out to need") is reported but never counted for/against the cap.
    """
    rows: list[dict[str, Any]] = []
    cum_h = spent_h
    n_scheduled = 0
    n_skipped = 0
    for j in jobs:
        est_min = j.get("est_min")
        est_h = float(est_min) / 60.0 if est_min is not None else None
        row: dict[str, Any] = {
            "id": j.get("id"),
            "job": j.get("job"),
            "est_min": est_min,
            "cum_before_h": cum_h,
        }
        if est_h is None:
            row["status"] = "unestimated"  # e.g. the R reserve placeholder; never auto-scheduled
        elif cum_h + 1.5 * est_h <= cap_h:
            row["status"] = "scheduled"
            cum_h += est_h
            n_scheduled += 1
        else:
            row["status"] = "skipped"
            n_skipped += 1
        row["cum_after_h"] = cum_h
        rows.append(row)
    return {
        "spent_h": spent_h,
        "cap_h": cap_h,
        "n_jobs": len(rows),
        "n_scheduled": n_scheduled,
        "n_skipped": n_skipped,
        "projected_total_h": cum_h,
        "rows": rows,
    }


def cmd_budget(args: argparse.Namespace) -> None:
    """Report GPU-hours spent (per `Store.ledger_total_s`) against `--cap-h`.

    With `--est-h`: whether plan §4's stop rule (`spent + 1.5 * estimate <= cap`) allows one next
    job of that estimated size. With `--schedule <file>` (fix list item 10; mutually exclusive with
    `--est-h`): reads `configs/schedule.yaml`'s `jobs` list and prints the same stop rule applied to
    every job in order, as a table -- a CPU-only dry run (this never touches the GPU or `gpulock`,
    only `Store.ledger_total_s()` for the starting `spent_h`, which is 0 for a fresh/no `--out`).
    """
    store = Store(Path(args.out))
    spent_h = store.ledger_total_s() / 3600.0
    if args.schedule is not None:
        schedule = _load_config(args.schedule)
        # `--cap-h` (default `None` here, unlike every other numeric flag) wins if the caller
        # explicitly gave it; otherwise fall back to the schedule file's own `cap_h`, then the plan
        # default.
        cap_h = (
            args.cap_h
            if args.cap_h is not None
            else float(cast(float, schedule.get("cap_h", DEFAULT_CAP_H)))
        )
        jobs = cast("list[Mapping[str, Any]]", schedule.get("jobs", []))
        print(json.dumps(_budget_schedule_table(jobs, spent_h=spent_h, cap_h=cap_h), indent=2))
        return

    cap_h = args.cap_h if args.cap_h is not None else DEFAULT_CAP_H
    result: dict[str, Any] = {
        "spent_h": spent_h,
        "cap_h": cap_h,
        "remaining_h": cap_h - spent_h,
    }
    if args.est_h is not None:
        result["est_h"] = args.est_h
        result["would_fit"] = spent_h + 1.5 * args.est_h <= cap_h
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
    r.add_argument(
        "--model",
        required=True,
        help=(
            "a label from --models-config (e.g. 'base'/'rl'/'armC'), an adapter name, or "
            "'adapter_name:ckpt_path'"
        ),
    )
    r.add_argument(
        "--model-arg", action="append", help="key=value, forwarded to the adapter constructor"
    )
    r.add_argument(
        "--models-config",
        default=DEFAULT_MODELS_CONFIG,
        dest="models_config",
        help="YAML model-label registry (plan §3 WP8 fix list item 6); '' disables it",
    )
    r.add_argument(
        "--device",
        default=None,
        help="forwarded to the adapter; 'cuda' requires CUDA_VISIBLE_DEVICES=4 already set",
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
    e.add_argument(
        "--model",
        required=True,
        help="a label from --models-config, an adapter name, or 'adapter_name:ckpt_path'",
    )
    e.add_argument("--model-arg", action="append")
    e.add_argument(
        "--models-config", default=DEFAULT_MODELS_CONFIG, dest="models_config", help="see 'run'"
    )
    e.add_argument("--device", default=None, help="see 'run'")
    e.add_argument("--config", default=None)
    e.add_argument("--lrs", default="1e-4,3e-4,1e-3")
    e.add_argument("--orders", type=int, default=2)
    e.add_argument("--n", type=int, default=64)
    e.add_argument("--m", type=int, default=64)
    e.add_argument("--ckpts", default=None)
    e.add_argument("--out", default=DEFAULT_OUT)
    e.set_defaults(func=cmd_edl)

    t = sub.add_parser("throughput", help="measure samples/s for a grid of models/sets/batch sizes")
    t.add_argument(
        "--models",
        required=True,
        help=(
            "comma-separated 'label'/'label=ckpt_path'/'ckpt_path'; a --models-config label "
            "resolves automatically"
        ),
    )
    t.add_argument(
        "--sets",
        required=True,
        help=(
            "comma-separated probe sets: a JSONL file path, an integer target length, or a suite "
            "name already built in --out (e.g. textbook_dev, gen_ctrl, families)"
        ),
    )
    t.add_argument("--batches", required=True)
    t.add_argument(
        "--models-config", default=DEFAULT_MODELS_CONFIG, dest="models_config", help="see 'run'"
    )
    t.add_argument("--device", default=None, help="see 'run'; omit to auto-detect cuda/cpu")
    t.add_argument(
        "--out", default=DEFAULT_OUT, help="Store root to resolve suite-name --sets from"
    )
    t.add_argument(
        "--suite-version", default="v1", dest="suite_version", help="version for suite-name --sets"
    )
    t.set_defaults(func=cmd_throughput)

    il = sub.add_parser("import-legacy", help="import pre-ndbench run results into the store (WP2)")
    il.add_argument("--out", default=DEFAULT_OUT)
    il.set_defaults(func=cmd_import_legacy)

    bg = sub.add_parser("budget", help="report GPU-hour ledger spend against a cap (plan §4)")
    bg.add_argument("--out", default=DEFAULT_OUT)
    bg.add_argument(
        "--cap-h", type=float, default=None, dest="cap_h", help=f"default {DEFAULT_CAP_H}"
    )
    bg.add_argument("--est-h", type=float, default=None, dest="est_h")
    bg.add_argument(
        "--schedule",
        default=None,
        help=(
            "print the whole schedule.yaml stop-rule table (fix list item 10) instead of a "
            "single --est-h check; never touches the GPU"
        ),
    )
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
