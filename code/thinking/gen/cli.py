"""`python -m code.thinking.gen <command>` (run from the repository root).

    select       stratified source proofs, eval sets excluded      -> RUN/selection.jsonl
    distractors  base-model rejected actions per decision state    -> RUN/distractors.jsonl
    build        request JSONL per model + cost estimate (dry run) -> RUN/requests/<tag>.r1.jsonl
    estimate     cost estimate of request files (dry run)
    submit       dry run unless --submit --max-usd X               -> RUN/ledger.jsonl, RUN/batches/
    poll         retrieve batches; download results when ended     -> RUN/results/<batch id>.jsonl
    collect      validate, write accepted/rejected/stats + retries -> RUN/collect/, RUN/requests/
    report       print RUN/collect/report.md

`--config FILE.yaml` supplies defaults: top-level keys for every command, and a section per command
(`build: {models: [...]}`); command-line flags win.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from . import batch as bt
from . import collect as cl
from . import requests as rq
from . import select as sl
from .client import BatchClient, HttpBatchClient


def _csv(x: Any) -> list[str]:
    if isinstance(x, list | tuple):
        return [str(v) for v in x]
    return [v for v in str(x).split(",") if v and v != "none"]


def _assumptions(p: argparse.ArgumentParser) -> None:
    p.add_argument("--assume-thinking", type=int, default=2000, help="thinking tokens per attempt")
    p.add_argument("--assume-attempts", type=float, default=1.3, help="attempts per proof")
    p.add_argument(
        "--assume-cache-hit", type=float, default=0.7, help="batch cache hits are best effort"
    )
    p.add_argument("--input-safety", type=float, default=1.25, help="worst-case input factor")


def _est_kw(a: argparse.Namespace) -> dict[str, Any]:
    return {
        "assume_thinking": a.assume_thinking,
        "assume_attempts": a.assume_attempts,
        "assume_cache_hit": a.assume_cache_hit,
        "input_safety": a.input_safety,
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m code.thinking.gen", description=(__doc__ or "").split("\n\n")[0]
    )
    ap.add_argument("--config", type=Path)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path)
        p.add_argument("--config", type=Path)
        return p

    p = cmd("select")
    p.add_argument("--source", type=Path, default=sl.K12)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alloc", choices=["balanced", "proportional"], default="balanced")
    p.add_argument("--strata", default=",".join(sl.STRATA))
    p.add_argument("--line-bins", default="2-4,5-7,8-10,11-12")
    p.add_argument("--classical-rules", default="DN")
    p.add_argument("--exclude", action="append", default=[], help="LABEL=PATH (JSONL with prompt)")
    p.add_argument("--no-default-excludes", action="store_true")
    p.add_argument("--id-prefix", default="k12")
    p.add_argument("--source-label", default="K12")

    p = cmd("distractors")
    p.add_argument("--selection", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-per-state", type=int, default=3)
    p.add_argument("--max-action", type=int, default=256)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-states", type=int, default=0)
    p.add_argument(
        "--with-dead", action="store_true", help="also keep actions that leave an unprovable goal"
    )

    p = cmd("build")
    p.add_argument("--selection", type=Path)
    p.add_argument("--models", default="claude-opus-5-5")
    p.add_argument("--mode", choices=sorted(rq.MODES), default="reference")
    p.add_argument("--distractors", default=None, help="distractor JSONL, or 'none'")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="low")
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=4096,
        help="budget_tokens for Haiku 4.5; 0 = no thinking",
    )
    p.add_argument("--cache-ttl", choices=["5m", "1h"], default="1h")
    p.add_argument("--examples", default=",".join(rq.DEFAULT_EXAMPLES), help="pilot ids, or 'none'")
    p.add_argument("--deviations", choices=["allow", "forbid"], default="allow")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--assume-distractors", type=int, default=3, help="per decision state, without a file"
    )
    _assumptions(p)

    p = cmd("estimate")
    p.add_argument("--requests", action="append", type=Path, default=[])
    _assumptions(p)

    p = cmd("submit")
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--submit", action="store_true", help="actually send (default: dry run)")
    p.add_argument("--max-usd", type=float)
    p.add_argument("--force-resubmit", action="store_true")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--interval", type=float, default=60.0)
    _assumptions(p)

    p = cmd("poll")
    p.add_argument("--batch-id", action="append", default=[])
    p.add_argument("--wait", action="store_true")
    p.add_argument("--interval", type=float, default=60.0)
    p.add_argument("--timeout-h", type=float, default=25.0)

    p = cmd("collect")
    p.add_argument("--lean", action="store_true", help="also check every accepted trace in Lean 4")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--max-tokens-factor", type=float, default=1.5)

    cmd("report")
    return ap


def _apply_config(ap: argparse.ArgumentParser, argv: list[str]) -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, rest = pre.parse_known_args(argv)
    cfg_path = known.config
    if cfg_path is None or not rest:
        return
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    name = rest[0]
    vals = {k: v for k, v in cfg.items() if not isinstance(v, dict)}
    vals.update(cfg.get(name) or {})
    vals = {k.replace("-", "_"): v for k, v in vals.items()}
    subs = next(a for a in ap._actions if isinstance(a, argparse._SubParsersAction))
    sp = subs.choices.get(name)
    if sp is None:
        return
    dests = {a.dest: a for a in sp._actions}
    defaults = {}
    for k, v in vals.items():
        if k not in dests:
            continue
        a = dests[k]
        if a.type is Path and v is not None:
            v = Path(v)
        elif isinstance(v, list) and not isinstance(a.default, list):
            v = ",".join(str(x) for x in v)
        defaults[k] = v
    sp.set_defaults(**defaults)


def _run_dir(a: argparse.Namespace) -> Path:
    if a.run_dir is None:
        raise SystemExit("--run-dir (or run_dir in --config) is required")
    return Path(a.run_dir)


def _print_est(est: dict[str, Any]) -> None:
    for mid, d in est["by_model"].items():
        exp, worst = d["expected_usd"], d["worst_usd"]
        print(f"  {mid}: {d['n_requests']} requests, expected ${exp:.2f}, worst ${worst:.2f}")
        print(f"    {d['arithmetic']}")
    print(f"  total: expected ${est['expected_usd']:.2f}, worst case ${est['worst_usd']:.2f}")


def _request_files(run: Path) -> list[Path]:
    return sorted(
        p for p in (run / "requests").glob("*.jsonl") if not p.name.endswith(".manifest.jsonl")
    )


def main(argv: list[str] | None = None, client: BatchClient | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = parser()
    _apply_config(ap, argv)
    a = ap.parse_args(argv)
    run = _run_dir(a)
    if a.cmd == "select":
        ex: dict[str, Path] = {} if a.no_default_excludes else sl.default_excludes()
        for e in a.exclude:
            label, _, path = str(e).partition("=")
            ex[label] = Path(path)
        path, meta = sl.run(
            run,
            a.source,
            a.n,
            a.seed,
            ex,
            dims=tuple(_csv(a.strata)),
            bins=a.line_bins,
            classical=a.classical_rules,
            alloc=a.alloc,
            id_prefix=a.id_prefix,
            source_label=a.source_label,
        )
        print(f"{path}: {meta['n_selected']} / {meta['n_requested']} proofs")
        print(f"  {meta['rows_excluded']} source rows excluded: {meta['rows_excluded_by_set']}")
        return 0
    if a.cmd == "distractors":
        from . import distractors as ds

        meta = ds.run(
            a.selection or run / "selection.jsonl",
            a.out or run / "distractors.jsonl",
            ckpt=a.ckpt or ds.STAGE1,
            device=a.device,
            k=a.k,
            temperature=a.temperature,
            max_per_state=a.max_per_state,
            max_action=a.max_action,
            batch=a.batch,
            seed=a.seed,
            limit_states=a.limit_states,
            with_dead=a.with_dead,
        )
        print(json.dumps(meta, indent=1))
        return 0
    if a.cmd == "build":
        dist = None if a.distractors in (None, "none") else Path(a.distractors)
        paths = rq.build(
            a.selection or run / "selection.jsonl",
            run / "requests",
            _csv(a.models),
            mode=a.mode,
            distractors=dist,
            examples=_csv(a.examples),
            deviations=a.deviations,
            limit=a.limit,
            assume_distractors=a.assume_distractors,
            effort=a.effort,
            max_tokens=a.max_tokens,
            thinking_budget=a.thinking_budget,
            ttl=a.cache_ttl,
        )
        for p in paths:
            r = bt.submit(p, run, send=False, **_est_kw(a))
            print(f"{p} (dry run, nothing sent; estimate in {r['estimate_path']})")
            _print_est(r["estimate"])
        return 0
    if a.cmd == "estimate":
        for p in a.requests or _request_files(run):
            print(p)
            _print_est(rq.estimate_file(p, **_est_kw(a)))
        return 0
    if a.cmd == "submit":
        try:
            r = bt.submit(
                a.requests,
                run,
                client=client,
                send=a.submit,
                max_usd=a.max_usd,
                force=a.force_resubmit,
                **_est_kw(a),
            )
        except bt.Refused as e:
            print(f"refused: {e}", file=sys.stderr)
            return 2
        _print_est(r["estimate"])
        if not r["sent"]:
            print("dry run: nothing sent (add --submit --max-usd X to send)")
            return 0
        bid = r["batch"]["id"]
        print(f"submitted {bid}; ledger {run / 'ledger.jsonl'}")
        if a.wait:
            b = bt.poll(bid, run, client or HttpBatchClient(), wait=True, interval=a.interval)
            print(f"{bid}: {b.get('processing_status')} {b.get('request_counts')}")
        return 0
    if a.cmd == "poll":
        c = client or HttpBatchClient()
        ids = list(a.batch_id)
        if not ids:
            subs = [
                r["batch_id"]
                for r in bt.ledger_rows(run / "ledger.jsonl")
                if r.get("kind") == "submit"
            ]
            ids = [b for b in subs if not bt.results_path(run, b).exists()]
        for bid in ids:
            b = bt.poll(bid, run, c, wait=a.wait, interval=a.interval, timeout_h=a.timeout_h)
            print(f"{bid}: {b.get('processing_status')} {b.get('request_counts')}")
        return 0
    if a.cmd == "collect":
        st = cl.collect(
            run, lean=a.lean, max_attempts=a.max_attempts, max_tokens_factor=a.max_tokens_factor
        )
        print((run / "collect" / "report.md").read_text())
        for f in st["retry_files"]:
            print(f"retry requests: {f} (dry run until submitted)")
        return 0
    if a.cmd == "report":
        print((run / "collect" / "report.md").read_text())
        return 0
    raise AssertionError(a.cmd)
