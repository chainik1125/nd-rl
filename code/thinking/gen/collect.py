"""Parse batch results, render them as pilot-format trace records, validate them with trace_dsl
(strict; Lean optional), and write accepted traces, rejections, retry requests and per-model stats.

Stateless and idempotent: every run re-reads all request manifests and all downloaded results under
the run directory.  A proof's attempts form a chain (attempt 1, 2, 3); the chain's last attempt
decides: accepted, pending (no result yet), retry (a new request file is written), or rejected.
Retries: a failed validation is sent back as a follow-up user turn after the unchanged assistant
turn; a truncated, expired, canceled or server-errored attempt is resubmitted as is.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from code.thinking import dan

from . import batch as bt
from . import dsl
from . import requests as rq
from .models import USAGE_KEYS, add, billed, cost, spec

CATEGORIES = (
    ("refusal", "refusal"),
    ("max_tokens", "max_tokens"),
    ("json:", "json"),
    ("schema:", "schema"),
    ("api:", "api_error"),
    ("expired", "expired"),
    ("canceled", "canceled"),
    ("(strict)", "strict_soft_finding"),
    ("tokens outside the vocabulary", "vocabulary"),
    ("false look", "false_look"),
    ("<no> on a candidate that works", "no_on_working_candidate"),
    ("unrejected candidate fails", "failing_candidate_unrejected"),
    ("<no> on a provable goal", "no_on_provable_goal"),
    ("<no> on a sound reduction", "no_on_sound_reduction"),
    ("<no> line without", "reject_without_candidate"),
    ("canonical names", "noncanonical_names"),
    ("action fails", "action_fails"),
    ("not the last unrejected", "commit_rule"),
    ("skeleton", "skeleton"),
    ("goal context", "goal_context"),
    ("formula", "formula_syntax"),
    ("unknown line", "unknown_line"),
    ("replay failed", "replay"),
    ("do not finish", "unfinished"),
    ("already finished", "extra_steps"),
    ("Lean rejects", "lean"),
    ("malformed line (checker", "validator_crash"),
)


def category(error: str | None) -> str:
    if not error:
        return "none"
    for needle, cat in CATEGORIES:
        if needle in error:
            return cat
    return "other"


def load_requests(req_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    lines: dict[str, dict[str, Any]] = {}
    metas: dict[str, dict[str, Any]] = {}
    for p in sorted(req_dir.glob("*.manifest.jsonl")):
        rp = p.with_name(p.name.replace(".manifest.jsonl", ".jsonl"))
        for m in rq.read_jsonl(p):
            metas[m["custom_id"]] = m
        for ln in rq.read_jsonl(rp):
            lines[ln["custom_id"]] = ln
    return lines, metas


def load_results(run_dir: Path) -> dict[str, tuple[str, dict[str, Any]]]:
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    for p in sorted((run_dir / "results").glob("*.jsonl")):
        for r in rq.read_jsonl(p):
            out[r["custom_id"]] = (p.stem, r)
    return out


def parse_message(msg: dict[str, Any]) -> tuple[list[dict[str, str]] | None, str | None]:
    sr = msg.get("stop_reason")
    if sr == "refusal":
        cat = (msg.get("stop_details") or {}).get("category")
        return None, f"refusal ({cat})"
    if sr == "max_tokens":
        return None, "max_tokens"
    text = "".join(b.get("text", "") for b in msg.get("content", []) if b.get("type") == "text")
    try:
        obj = json.loads(text)
    except ValueError as e:
        return None, f"json: {e}"
    steps = obj.get("steps") if isinstance(obj, dict) else None
    if not isinstance(steps, list) or not steps:
        return None, "schema: no steps"
    for s in steps:
        if not isinstance(s, dict) or set(s) != {"think", "action"}:
            return None, "schema: each step must be {think, action}"
        if not isinstance(s["think"], str) or not isinstance(s["action"], str):
            return None, "schema: think and action must be strings"
    return steps, None


def to_record(meta: dict[str, Any], steps: list[dict[str, str]]) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "id": meta["proof_id"],
        "source": meta.get("source", ""),
        "mode": "reference-guided" if meta["mode"] == "reference" else meta["mode"],
        "deviates": None,
        "prompt": meta["prompt"],
        "steps": steps,
    }
    ref = meta.get("ref_actions") or []
    if meta["mode"] == "reference" and ref:
        td = dsl.dsl()
        got = [td.tokens(s["action"]) for s in steps]
        want = [td.tokens(a) for a in ref]
        rec["deviates"] = got != want
        if rec["deviates"]:
            i = next(
                (k for k, (a, b) in enumerate(zip(got, want, strict=False)) if a != b),
                min(len(got), len(want)),
            )
            rec["deviation"] = (
                f"actions differ from the reference from step {i} ({len(got)} vs {len(want)} steps)"
            )
    return rec


def judge(meta: dict[str, Any], result: dict[str, Any], lean: bool) -> dict[str, Any]:
    """One attempt -> {status, error, record, report, tokens, content}."""
    res = result.get("result", {})
    typ = res.get("type")
    out: dict[str, Any] = {"status": "rejected", "error": None, "tokens": {}, "type": typ}
    if typ != "succeeded":
        if typ == "errored":
            err = (res.get("error") or {}).get("error") or res.get("error") or {}
            out["error"] = f"api: {err.get('type', '?')}: {err.get('message', '')}"[:300]
            out["status"] = (
                "rejected" if err.get("type") == "invalid_request_error" else "retry_fresh"
            )
        else:
            out["error"] = str(typ)
            out["status"] = "retry_fresh"
        return out
    msg = res["message"]
    out["tokens"] = billed(msg.get("usage", {}))
    out["stop_reason"] = msg.get("stop_reason")
    steps, err = parse_message(msg)
    if steps is None:
        out["error"] = err
        out["status"] = (
            "retry_fresh"
            if err == "max_tokens"
            else ("rejected" if (err or "").startswith("refusal") else "retry_feedback")
        )
        out["content"] = msg.get("content", [])
        return out
    rec = to_record(meta, steps)
    try:
        with dan.bare():
            report = dsl.dsl().validate(rec, lean=lean, strict=True)
    except Exception as e:  # a checker crash on malformed LLM output counts as a rejection
        report = {
            "ok": False,
            "error": f"malformed line (checker {type(e).__name__}): every have/exact line "
            "must be a complete action",
        }
    out.update(record=rec, report=report, content=msg.get("content", []))
    if report["ok"]:
        out["status"] = "accepted"
    else:
        out["error"] = report["error"]
        out["status"] = "retry_feedback"
    return out


def collect(
    run_dir: Path,
    lean: bool = False,
    max_attempts: int = 3,
    max_tokens_factor: float = 1.5,
) -> dict[str, Any]:
    req_dir = run_dir / "requests"
    lines, metas = load_requests(req_dir)
    results = load_results(run_dir)
    chains: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for m in metas.values():
        chains[(m["tag"], m["mode"], m["proof_id"])].append(m)
    accepted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retries: dict[tuple[str, str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(
        list
    )
    per_chain: list[dict[str, Any]] = []
    for (tag, mode, pid), chain in sorted(chains.items()):
        chain.sort(key=lambda m: int(m["attempt"]))
        tokens: dict[str, int] = {}
        errors = []
        last: dict[str, Any] | None = None
        batches = []
        for m in chain:
            hit = results.get(m["custom_id"])
            if hit is None:
                last = None
                break
            batches.append(hit[0])
            last = judge(m, hit[1], lean)
            tokens = add(tokens, last["tokens"])
            if last["error"]:
                errors.append(
                    {
                        "attempt": m["attempt"],
                        "error": last["error"],
                        "category": category(last["error"]),
                    }
                )
        m = chain[-1]
        model = spec(m["model"])
        info = {
            "tag": tag,
            "model": model.id,
            "mode": mode,
            "proof_id": pid,
            "attempts": len(batches),
            "tokens": tokens,
            "cost_usd": cost(tokens, model),
            "errors": errors,
            "batches": batches,
            "n_steps_ref": m.get("n_steps"),
        }
        if last is None:
            info["status"] = "pending"
        elif last["status"] == "accepted":
            info["status"] = "accepted"
            info["think_tokens"] = last["report"].get("think_tokens_total")
            info["deviates"] = last["record"].get("deviates")
            rec = dict(last["record"])
            rec["gen"] = {
                "model": model.id,
                "attempts": len(batches),
                "custom_ids": [x["custom_id"] for x in chain],
                "batch_ids": batches,
                "billed_tokens": tokens,
                "cost_usd": round(cost(tokens, model), 6),
                "lean_checked": lean,
                "validator": {
                    k: last["report"].get(k)
                    for k in ("think_tokens_total", "pair_tokens_max", "stats")
                },
            }
            accepted[tag].append(rec)
        elif last["status"].startswith("retry") and int(m["attempt"]) < max_attempts:
            kind = "feedback" if last["status"] == "retry_feedback" else "fresh"
            prev = lines[m["custom_id"]]
            mt = None
            if last.get("error") == "max_tokens":
                mt = int(prev["params"]["max_tokens"] * max_tokens_factor)
            line, meta = rq.retry_request(
                prev,
                m,
                kind,
                content=last.get("content"),
                feedback=rq.feedback_text(last["error"]) if kind == "feedback" else None,
                prev_output_tokens=last["tokens"].get("output", 0),
                max_tokens=mt,
            )
            retries[(tag, mode, int(meta["attempt"]))].append((line, meta))
            info["status"] = "retry_written"
        else:
            info["status"] = "rejected"
            rejected[tag].append(
                {
                    "id": pid,
                    "model": model.id,
                    "mode": mode,
                    "prompt": m["prompt"],
                    "attempts": len(batches),
                    "final_error": last["error"],
                    "category": category(last["error"]),
                    "errors": errors,
                }
            )
        per_chain.append(info)
    new_files = []
    for (tag, mode, att), pairs in sorted(retries.items()):
        stem, k = f"{tag}.{rq.MODES[mode]}{att}", 1
        p = req_dir / f"{stem}.jsonl"
        while p.exists():
            k += 1
            p = req_dir / f"{stem}_{k}.jsonl"
        rq.write_jsonl(p, (a for a, _ in pairs))
        rq.write_jsonl(rq.manifest_path(p), (b for _, b in pairs))
        new_files.append(str(p))
    out_dir = run_dir / "collect"
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = sorted({c["tag"] for c in per_chain})
    for tag in tags:
        rq.write_jsonl(out_dir / f"accepted.{tag}.jsonl", accepted.get(tag, []))
        rq.write_jsonl(out_dir / f"rejected.{tag}.jsonl", rejected.get(tag, []))
    rq.write_jsonl(out_dir / "chains.jsonl", per_chain)
    st = stats(per_chain)
    st["retry_files"] = new_files
    st["ledger_usage_rows"] = len(bt.record_usage(run_dir))
    (out_dir / "stats.json").write_text(json.dumps(st, indent=1))
    (out_dir / "report.md").write_text(report_md(st))
    return st


def stats(chains: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, dict[str, Any]] = {}
    for c in chains:
        d = by.setdefault(
            c["model"],
            {
                "n_proofs": 0,
                "status": Counter(),
                "accepted_by_attempt": Counter(),
                "tokens": {},
                "cost_usd": 0.0,
                "attempts": 0,
                "errors": Counter(),
                "final_errors": Counter(),
                "think_tokens": [],
                "deviates": 0,
            },
        )
        d["n_proofs"] += 1
        d["status"][c["status"]] += 1
        d["attempts"] += c["attempts"]
        d["tokens"] = add(d["tokens"], c["tokens"])
        d["cost_usd"] += c["cost_usd"]
        for e in c["errors"]:
            d["errors"][e["category"]] += 1
        if c["status"] == "accepted":
            d["accepted_by_attempt"][c["attempts"]] += 1
            if c.get("think_tokens") is not None:
                d["think_tokens"].append(c["think_tokens"])
            d["deviates"] += bool(c.get("deviates"))
        elif c["status"] == "rejected" and c["errors"]:
            d["final_errors"][c["errors"][-1]["category"]] += 1
    out: dict[str, Any] = {"by_model": {}}
    for mid, d in by.items():
        n, acc = d["n_proofs"], d["status"].get("accepted", 0)
        tok = {k: d["tokens"].get(k, 0) for k in USAGE_KEYS}
        tt = d["think_tokens"]
        out["by_model"][mid] = {
            "n_proofs": n,
            "accepted": acc,
            "accepted_pct": 100.0 * acc / n if n else 0.0,
            "first_pass_pct": 100.0 * d["accepted_by_attempt"].get(1, 0) / n if n else 0.0,
            "accepted_by_attempt": dict(sorted(d["accepted_by_attempt"].items())),
            "status": dict(d["status"]),
            "attempts": d["attempts"],
            "billed_tokens": tok,
            "billed_tokens_total": sum(tok.values()),
            "billed_tokens_per_accepted": {k: v / acc for k, v in tok.items()} if acc else None,
            "billed_tokens_total_per_accepted": sum(tok.values()) / acc if acc else None,
            "cost_usd": d["cost_usd"],
            "cost_per_accepted_usd": d["cost_usd"] / acc if acc else None,
            "error_categories_all_attempts": dict(d["errors"].most_common()),
            "final_error_categories": dict(d["final_errors"].most_common()),
            "think_tokens_per_trace_mean": sum(tt) / len(tt) if tt else None,
            "deviates_pct_of_accepted": 100.0 * d["deviates"] / acc if acc else None,
        }
    bm = out["by_model"]
    o, s = bm.get("claude-opus-5-5"), bm.get("claude-sonnet-5-5")
    if o and s and o["billed_tokens_total_per_accepted"] and s["billed_tokens_total_per_accepted"]:
        r = s["billed_tokens_total_per_accepted"] / o["billed_tokens_total_per_accepted"]
        out["sonnet_vs_opus"] = {
            "billed_tokens_per_accepted_ratio": r,
            "output_tokens_per_accepted_ratio": (
                s["billed_tokens_per_accepted"]["output"]
                / o["billed_tokens_per_accepted"]["output"]
            )
            if o["billed_tokens_per_accepted"]["output"]
            else None,
            "cost_per_accepted_ratio": s["cost_per_accepted_usd"] / o["cost_per_accepted_usd"],
            "sonnet_under_2x_tokens": r < 2.0,
        }
    return out


def _f(x: float | None, nd: int = 0) -> str:
    return "–" if x is None else f"{x:,.{nd}f}"


def report_md(st: dict[str, Any]) -> str:
    head = (
        "model",
        "proofs",
        "accepted",
        "first pass",
        "by attempt",
        "rejected",
        "pending",
        "billed tok / accepted (in / cache read / cache write / out)",
        "total tok / accepted",
        "$ / accepted",
        "$ total",
    )
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for mid, d in sorted(st["by_model"].items()):
        pa = d["billed_tokens_per_accepted"]
        tok = "–"
        if pa:
            w = pa["cache_write_5m"] + pa["cache_write_1h"]
            tok = " / ".join(_f(x) for x in (pa["input"], pa["cache_read"], w, pa["output"]))
        stt, n = d["status"], d["n_proofs"]
        cells = (
            mid,
            str(n),
            f"{d['accepted_pct']:.1f}% ({d['accepted']})",
            f"{d['first_pass_pct']:.1f}%",
            str(d["accepted_by_attempt"]),
            f"{100.0 * stt.get('rejected', 0) / n:.1f}%",
            str(stt.get("pending", 0) + stt.get("retry_written", 0)),
            tok,
            _f(d["billed_tokens_total_per_accepted"]),
            _f(d["cost_per_accepted_usd"], 4),
            f"{d['cost_usd']:.2f}",
        )
        rows.append("| " + " | ".join(cells) + " |")
    out = ["# Trace generation: collect report", "", *rows, ""]
    for mid, d in sorted(st["by_model"].items()):
        out.append(
            f"- {mid}: error categories over all attempts {d['error_categories_all_attempts']}; "
            f"final {d['final_error_categories']}; mean think tokens per accepted trace "
            f"{_f(d['think_tokens_per_trace_mean'])}; deviates "
            f"{_f(d['deviates_pct_of_accepted'], 1)}% of accepted"
        )
    sv = st.get("sonnet_vs_opus")
    if sv:
        verdict = "Sonnet qualifies" if sv["sonnet_under_2x_tokens"] else "use Opus"
        out += [
            "",
            "Sonnet 5.5 / Opus 5.5 billed tokens per accepted trace: "
            f"{sv['billed_tokens_per_accepted_ratio']:.2f}x (output only "
            f"{_f(sv['output_tokens_per_accepted_ratio'], 2)}x; cost "
            f"{sv['cost_per_accepted_ratio']:.2f}x). Rule (Sonnet only under 2x Opus's tokens): "
            f"{verdict}.",
        ]
    if st.get("retry_files"):
        out += [
            "",
            "Retry request files written (dry run until submitted): "
            + ", ".join(st["retry_files"]),
        ]
    return "\n".join(out) + "\n"
