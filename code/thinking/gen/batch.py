"""Submit a request file as one Message Batch, poll it, and download its results; every submission
and every downloaded batch's billed usage goes into the run's ledger (`ledger.jsonl`).

Safety: the default is a dry run (estimate only, nothing sent).  Sending needs `submit=True` and a
`max_usd` at or above the file's worst-case estimate, refuses a file whose sha256 the ledger
already shows as submitted (unless `force`), and never handles the API key itself (`client.py`).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from . import requests as rq
from .client import BatchClient, key_is_set
from .models import add, billed, cost, spec


class Refused(RuntimeError):
    pass


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ledger_rows(path: Path) -> list[dict[str, Any]]:
    return rq.read_jsonl(path) if path.exists() else []


def ledger_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"utc": utc(), **row}) + "\n")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_lines(lines: list[dict[str, Any]]) -> None:
    ids = [ln["custom_id"] for ln in lines]
    bad = [i for i in ids if not rq.CUSTOM_ID.match(i)]
    if bad:
        raise Refused(f"invalid custom_id(s): {bad[:3]}")
    dup = [i for i, c in Counter(ids).items() if c > 1]
    if dup:
        raise Refused(f"duplicate custom_id(s): {dup[:3]}")
    for ln in lines:
        p = ln["params"]
        if p.get("stream") or p.get("speed") or not p.get("max_tokens"):
            raise Refused(
                f"{ln['custom_id']}: stream, speed and max_tokens 0 are not allowed in a batch"
            )


def submit(
    requests_path: Path,
    run_dir: Path,
    client: BatchClient | None = None,
    send: bool = False,
    max_usd: float | None = None,
    force: bool = False,
    **est_kw: Any,
) -> dict[str, Any]:
    """Dry run unless `send`.  Returns {estimate, batch?}."""
    lines = rq.read_jsonl(requests_path)
    check_lines(lines)
    est = rq.estimate_file(requests_path, **est_kw)
    est_path = requests_path.with_name(requests_path.name.replace(".jsonl", ".estimate.json"))
    est_path.write_text(json.dumps(est, indent=1))
    out: dict[str, Any] = {"estimate": est, "estimate_path": str(est_path), "sent": False}
    if not send:
        return out
    if max_usd is None:
        raise Refused("--submit needs --max-usd")
    if est["worst_usd"] > max_usd:
        raise Refused(
            f"worst-case estimate ${est['worst_usd']:.2f} exceeds --max-usd {max_usd:.2f}"
        )
    sha = file_sha(requests_path)
    ledger = run_dir / "ledger.jsonl"
    prior = [r for r in ledger_rows(ledger) if r.get("kind") == "submit" and r.get("sha256") == sha]
    if prior and not force:
        raise Refused(f"{requests_path.name} was already submitted as {prior[-1]['batch_id']}")
    if client is None:
        if not key_is_set():
            raise Refused("ANTHROPIC_API_KEY is not set")
        from .client import HttpBatchClient

        client = HttpBatchClient()
    batch = client.create(lines)
    bdir = run_dir / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / f"{batch['id']}.json").write_text(json.dumps(batch, indent=1))
    models = Counter(ln["params"]["model"] for ln in lines)
    ledger_append(
        ledger,
        {
            "kind": "submit",
            "batch_id": batch["id"],
            "model": next(iter(models)) if len(models) == 1 else "mixed",
            "n_by_model": dict(models),
            "n_requests": len(lines),
            "requests_file": str(requests_path),
            "sha256": sha,
            "estimated_usd": round(est["expected_usd"], 4),
            "worst_usd": round(est["worst_usd"], 4),
            "max_usd": max_usd,
        },
    )
    out.update(sent=True, batch=batch)
    return out


def results_path(run_dir: Path, batch_id: str) -> Path:
    return run_dir / "results" / f"{batch_id}.jsonl"


def poll(
    batch_id: str,
    run_dir: Path,
    client: BatchClient,
    wait: bool = False,
    interval: float = 60.0,
    timeout_h: float = 25.0,
) -> dict[str, Any]:
    """Retrieve the batch (until it ends, with `wait`); download its results once it has ended."""
    t0 = time.time()
    while True:
        b = client.retrieve(batch_id)
        (run_dir / "batches").mkdir(parents=True, exist_ok=True)
        (run_dir / "batches" / f"{batch_id}.json").write_text(json.dumps(b, indent=1))
        if b.get("processing_status") == "ended" or not wait or time.time() - t0 > timeout_h * 3600:
            break
        time.sleep(interval)
    if b.get("processing_status") == "ended":
        rp = results_path(run_dir, batch_id)
        if not rp.exists():
            rp.parent.mkdir(parents=True, exist_ok=True)
            tmp = rp.with_suffix(".part")
            with open(tmp, "w") as f:
                for r in client.results(batch_id):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.rename(rp)
            ledger_append(
                run_dir / "ledger.jsonl",
                {"kind": "ended", "batch_id": batch_id, "request_counts": b.get("request_counts")},
            )
    return b


def usage_of_results(rows: list[dict[str, Any]], ttl: str = "1h") -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for r in rows:
        res = r.get("result", {})
        msg = res.get("message") or {}
        mid = msg.get("model")
        d = by_model.setdefault(mid or "?", {"tokens": {}, "results": Counter(), "cost_usd": 0.0})
        d["results"][res.get("type", "?")] += 1
        if res.get("type") == "succeeded" and mid:
            t = billed(msg.get("usage", {}), ttl)
            d["tokens"] = add(d["tokens"], t)
            d["cost_usd"] += cost(t, spec(mid))
    for d in by_model.values():
        d["results"] = dict(d["results"])
        d["cost_usd"] = round(d["cost_usd"], 6)
    return by_model


def record_usage(run_dir: Path, ttl: str = "1h") -> list[dict[str, Any]]:
    """Ledger rows with the billed usage of every downloaded batch not yet recorded."""
    ledger = run_dir / "ledger.jsonl"
    done = {r["batch_id"] for r in ledger_rows(ledger) if r.get("kind") == "usage"}
    rows = []
    for rp in sorted((run_dir / "results").glob("*.jsonl")):
        if rp.stem in done:
            continue
        u = usage_of_results(rq.read_jsonl(rp), ttl)
        row = {
            "kind": "usage",
            "batch_id": rp.stem,
            "by_model": u,
            "cost_usd": round(sum(d["cost_usd"] for d in u.values()), 6),
        }
        ledger_append(ledger, row)
        rows.append(row)
    return rows
