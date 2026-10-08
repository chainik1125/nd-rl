"""Training pairs for `lean_thinkn`: `state <think> ... </think> action <eos>`, one pair per
environment step.

    python -m code.thinking.data --traces T.jsonl --plain train_k12.jsonl --no-trace-mode empty \
        --out /scratch/.../pairs_T2.pkl

Inputs
  --traces  trace records (`path` or `path:weight`, the repeat count) in the pilot format ({id,
            prompt, steps: [{think, action}]}, ASCII surface), validated by
            `trace_dsl.validate(strict=True)` (Lean too unless --no-lean); invalid records are
            dropped and counted by reason.  A plain record with the same prompt as a valid trace
            is replaced by the trace, and a trace whose prompt an earlier --traces file already
            covered is skipped (so `LLM:2 PROG` puts LLM traces on their proofs, programmatic ones
            on the rest).
  --empty-frac  the fraction of trace copies written with every think block emptied (`<think>
            </think>`, Dualformer-style; Leon 2026-10-05: 0.2), chosen per (id, copy) by a hash so
            the build is deterministic.  Repeated LLM copies are drawn independently.
  --plain   K12-style records ({prompt, proof}) decomposed by the fork's
            `state_env.decompose(canon=True)`; `--no-trace-mode empty` writes `<think> </think>`
            before the action, `none` writes the bare action (the T0 no-think control in the same
            vocabulary).
  --think-filler  the filler control (LIT_CRITIQUE C3, Pfau et al. / Gandhi et al.): every
            think-content token (`<nl>` included) becomes `<no>`, so each block keeps its length
            and position but carries no content.  Applied after validation and emptying, so the
            same records, copies and emptied copies as without the flag.

Eval exclusion: a record is dropped if its atom-renaming class key (`canon_key`, as the fork's
`gen.canon_key` and ndbench's `core.keys`) or its premise-order-invariant `pkey` (ndbench's dedup
rule, skipped above 5 premises) is in any --exclude file.  Defaults: the OOD tiers, textbook72 dev58
and the K12 held-out set (`DEFAULT_EXCLUDE`).

Output (pickle): {'format', 'mode', 'itos', 'no_trace_mode', 'records': [[(state bytes, target
bytes, think_len)]], 'meta': [{id, prompt, key, source, n_lines}], 'config', 'report'} -- per record
the list of its pairs, the layout of the fork's `state_train.load` with the target = think block +
action + <eos> (`to_fork` drops think_len).  The trained mask covers the whole target (state mask
0); `think_len` (the block's length, tags included) lets the trainer weight think tokens separately.
A JSON report (`<out>.report.json`) gives counts, the token-length distribution of the pairs and the
number of pairs longer than --max_len.

Graded effort (`lean_thinke`, `build_graded`):

    python -m code.thinking.data --graded 0:0.2 16:e16.jsonl:0.2 64:e64.jsonl:0.3 \
        256:e256.jsonl:0.3 --no-lean --out T1v1g4.pkl

Each proof of the level-256 file draws ONE effort level by `unit_hash(id/effort)` (probabilities
in ascending level order) and trains on that level's think blocks; every pair's state ends in the
level's effort token (`pair = state <eB> | <think> ... </think> action <eos>`, the layout of
`effort.relabel_records`).  SPEC is `0:PROB` (level 0 = the level-256 record with every think
emptied) or `LEVEL:PATH:PROB`.  The level-256 file defines the proof list and order; the other
files hold the same ids, prompts and actions.  A proof missing at its drawn level is skipped and
counted in `missing_at_level`; a record whose actions or prompt differ from the level-256 record,
or whose think block exceeds its level's budget (`ValueError`), is a broken input, not data.
Mutually exclusive with --traces / --plain.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import pickle
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from code.ndbench.core.keys import canon_key, pkey, prompt_to_thm, split_thm

from .dan import bare, load, trace_dsl
from .tok import END_THINK, LEVELS, THINK, EffortTokenizer, ThinkTokenizer

D = load()
FORMAT = "thinking-pairs/1"
MAX_PREM_FOR_PKEY = 5  # ndbench logic/exclusion.py
FORK = "/scratch/cowifornia/nd-rl/think/nd-takehome"
DEFAULT_EXCLUDE = sorted(glob.glob("/scratch/cowifornia/nd-rl/think/ood_check/sets/*.jsonl")) + [
    f"{FORK}/data/eval_only/textbook72/textbook_dev.jsonl",
    "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/heldout.jsonl",
]
Pair = tuple[bytes, bytes, int]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------------- eval exclusion
def class_keys(prompt: str) -> tuple[str, str | None]:
    """(atom-renaming class key, premise-order-invariant key or None above 5 premises) of a
    `THM ... PRF` prompt."""
    thm = prompt_to_thm(prompt)
    prem, _ = split_thm(thm)
    return canon_key(thm), (pkey(thm) if len(prem) <= MAX_PREM_FOR_PKEY else None)


class EvalIndex:
    """Renaming classes of the eval sets; `hit(prompt)` -> labels of the sets holding its class."""

    def __init__(self, files: Sequence[str | Path]) -> None:
        self.keys: dict[str, set[str]] = {}
        self.pkeys: dict[str, set[str]] = {}
        self.sizes: dict[str, int] = {}
        for p in files:
            label = Path(p).stem
            ks, pks = set(), set()
            rows = read_jsonl(p)
            for r in rows:
                k, pk = class_keys(r["prompt"])
                ks.add(k)
                if pk is not None:
                    pks.add(pk)
            self.keys[label], self.pkeys[label], self.sizes[label] = ks, pks, len(rows)

    def hit(self, prompt: str) -> list[str]:
        k, pk = class_keys(prompt)
        return [
            lab
            for lab in self.keys
            if k in self.keys[lab] or (pk is not None and pk in self.pkeys[lab])
        ]


# ------------------------------------------------------------------------------------------- pairs
def think_block(text: str, T: Any) -> list[str]:
    """ASCII think text -> `<think> line <nl> line ... </think>` (empty lines dropped, no trailing
    `<nl>`)."""
    toks = T.tokens(text)
    if toks and toks[0] == THINK:
        toks = toks[1:]
    if toks and toks[-1] == END_THINK:
        toks = toks[:-1]
    lines = T.split_lines([THINK, *toks, END_THINK])
    out = [THINK]
    for i, ln in enumerate(lines):
        out += (["<nl>"] if i else []) + ln
    return out + [END_THINK]


def encode_pair(
    tok: Any, state: Sequence[str], think: Sequence[str], action: Sequence[str]
) -> Pair:
    p = tok.encode_toks(list(state))
    q = tok.encode_toks(list(think) + list(action)) + [tok.eos]
    return bytes(p), bytes(q), len(think)


def filler_pair(tok: Any, pair: Pair) -> Pair:
    """The pair with every think-content token replaced by `<no>` (tags, action, length kept)."""
    p, q, tl = pair
    if tl <= 2:
        return pair
    assert q[0] == tok.think and q[tl - 1] == tok.end_think, "not a think block"
    return p, bytes([q[0], *([tok.no] * (tl - 2)), *q[tl - 1 :]]), tl


def plain_pairs(rec: dict[str, Any], tok: Any, mode: str) -> list[Pair]:
    """A K12-style record -> pairs (raises ParseFail, ValueError, ... as state_train.load does)."""
    steps, _, _ = D.state_env.decompose(rec["prompt"], rec["proof"], canon=True)
    think = [THINK, END_THINK] if mode == "empty" else []
    return [encode_pair(tok, s, think, a) for s, a, _ in steps]


def trace_pairs(rec: dict[str, Any], tok: Any) -> list[Pair]:
    """A validated trace record -> pairs, replayed in the environment as trace_dsl.validate does."""
    T = trace_dsl()
    env = D.state_env.Env(rec["prompt"], canon=True, base=0, assign=False)
    out = []
    for st in rec["steps"]:
        action = T.tokens(st["action"])
        out.append(encode_pair(tok, env.state_tokens(), think_block(st["think"], T), action))
        ok, why = env.apply(action)
        assert ok, why
    assert env.done
    return out


def validate_trace(rec: dict[str, Any], lean: bool) -> dict[str, Any]:
    """trace_dsl.validate, strict; never raises."""
    T = trace_dsl()
    with bare():
        try:
            return T.validate(rec, lean=lean, strict=True)
        except (
            Exception
        ) as e:  # a malformed record (missing fields, ...) is an invalid trace, not a crash
            return {"id": rec.get("id"), "ok": False, "error": f"{type(e).__name__}: {e}"}


def to_fork(records: Iterable[Sequence[Pair]]) -> list[list[tuple[bytes, bytes]]]:
    """-> the fork's training layout, per record [(state bytes, target bytes)]."""
    return [[(p, q) for p, q, _ in rec] for rec in records]


# ------------------------------------------------------------------------------------------ report
def _dist(xs: Sequence[int]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(xs)

    def q(f: float) -> int:
        return s[min(len(s) - 1, int(f * len(s)))]

    return {
        "n": len(s),
        "mean": round(sum(s) / len(s), 2),
        "min": s[0],
        "p50": q(0.5),
        "p90": q(0.9),
        "p99": q(0.99),
        "max": s[-1],
    }


def length_report(records: Sequence[Sequence[Pair]], max_len: int) -> dict[str, Any]:
    pairs = [x for r in records for x in r]
    total = [len(p) + len(q) for p, q, _ in pairs]
    think_body = [max(0, t - 2) for _, _, t in pairs]
    hist = collections.Counter(min(n // 64, 16) * 64 for n in total)
    return {
        "pairs": len(pairs),
        "records": len(records),
        "pair_tokens": _dist(total),
        "state_tokens": _dist([len(p) for p, _, _ in pairs]),
        "target_tokens": _dist([len(q) for _, q, _ in pairs]),
        "think_tokens_excl_tags": _dist(think_body),
        "think_tokens_nonempty": _dist([t for t in think_body if t > 0]),
        "steps_with_think": sum(t > 0 for t in think_body),
        "pair_tokens_hist_64": {str(k): v for k, v in sorted(hist.items())},
        "max_len": max_len,
        "pairs_over_max_len": sum(n > max_len for n in total),
        "tokens_total": sum(total),
        "loss_tokens_total": sum(len(q) for _, q, _ in pairs),
    }


# ------------------------------------------------------------------------------------------- build
def unit_hash(key: str) -> float:
    """A deterministic draw in [0, 1) from a string."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64


def split_weight(spec: str | Path) -> tuple[str, int]:
    """'traces.jsonl:4' -> ('traces.jsonl', 4); no ':<int>' suffix -> weight 0 (the default)."""
    s = str(spec)
    head, _, tail = s.rpartition(":")
    return (head, int(tail)) if head and tail.isdigit() else (s, 0)


def build(
    traces: Sequence[str | Path] = (),
    plain: Sequence[str | Path] = (),
    no_trace_mode: str = "empty",
    exclude: Sequence[str | Path] = (),
    lean: bool = True,
    trace_weight: int = 1,
    empty_frac: float = 0.0,
    max_len: int = 1024,
    cap: int = 0,
    limit_plain: int = 0,
    tok: Any = None,
    think_filler: bool = False,
) -> dict[str, Any]:
    assert no_trace_mode in ("empty", "none"), no_trace_mode
    tok = ThinkTokenizer() if tok is None else tok
    index = EvalIndex(exclude)
    rep: dict[str, Any] = {
        "excluded_by_set": collections.Counter(),
        "invalid_trace_reasons": collections.Counter(),
        "invalid_traces": [],
        "plain_skipped": collections.Counter(),
    }
    records: list[list[Pair]] = []
    meta: list[dict[str, Any]] = []

    def excluded(prompt: str) -> bool:
        hits = index.hit(prompt)
        for h in hits:
            rep["excluded_by_set"][h] += 1
        return bool(hits)

    n_trace_in = n_trace_ok = 0
    traced_prompts: set[str] = set()
    n_emptied = 0
    for spec in traces:
        path, w = split_weight(spec)
        earlier = set(traced_prompts)
        for rec in read_jsonl(path):
            n_trace_in += 1
            if rec["prompt"] in earlier:
                k_ = "traces_covered_by_earlier_file"
                rep[k_] = rep.get(k_, 0) + 1
                continue
            if excluded(rec["prompt"]):
                rep["excluded_traces"] = rep.get("excluded_traces", 0) + 1
                continue
            v = validate_trace(rec, lean)
            if not v["ok"]:
                reason = str(v["error"]).split(":")[0] if v["error"] else "unknown"
                rep["invalid_trace_reasons"][reason] += 1
                rep["invalid_traces"].append({"id": rec.get("id"), "error": v["error"]})
                continue
            pairs = trace_pairs(rec, tok)
            if think_filler:
                pairs = [filler_pair(tok, x) for x in pairs]
            n_trace_ok += 1
            traced_prompts.add(rec["prompt"])
            k, _ = class_keys(rec["prompt"])
            m = {
                "id": rec.get("id"),
                "prompt": rec["prompt"],
                "key": k,
                "source": "trace",
                "n_lines": None,
            }
            empty = None
            for c in range(w or trace_weight):
                if empty_frac and unit_hash(f"{rec.get('id')}/{c}") < empty_frac:
                    if empty is None:
                        steps = [{"think": "", "action": st["action"]} for st in rec["steps"]]
                        empty = trace_pairs({**rec, "steps": steps}, tok)
                    records.append(empty)
                    meta.append({**m, "source": "trace_emptied"})
                    n_emptied += 1
                else:
                    records.append(pairs)
                    meta.append(m)
    n_plain_in = n_plain_ok = 0
    for path in plain:
        for i, rec in enumerate(read_jsonl(path)):
            if limit_plain and i >= limit_plain:
                break
            n_plain_in += 1
            if cap and rec.get("n_lines", 0) > cap:
                rep["plain_skipped"]["over_cap"] += 1
                continue
            if rec["prompt"] in traced_prompts:
                rep["plain_skipped"]["replaced_by_trace"] += 1
                continue
            if excluded(rec["prompt"]):
                rep["plain_skipped"]["eval_class"] += 1
                continue
            try:
                pairs = plain_pairs(rec, tok, no_trace_mode)
            except (D.lean_tok.ParseFail, ValueError, AssertionError, IndexError, KeyError):
                rep["plain_skipped"]["decompose_failed"] += 1
                continue
            n_plain_ok += 1
            k, _ = class_keys(rec["prompt"])
            meta.append(
                {
                    "id": rec.get("name", f"{Path(path).stem}:{i}"),
                    "prompt": rec["prompt"],
                    "key": k,
                    "source": "plain",
                    "n_lines": rec.get("n_lines"),
                }
            )
            records.append(pairs)
    ids = {x for r in records for p, q, _ in r for x in p + q}
    assert not ids or max(ids) < tok.vocab_size <= 256
    report = {
        "traces_in": n_trace_in,
        "traces_valid": n_trace_ok,
        "trace_weight": trace_weight,
        "empty_frac": empty_frac,
        "trace_copies_emptied": n_emptied,
        "think_filler": think_filler,
        "plain_in": n_plain_in,
        "plain_used": n_plain_ok,
        "no_trace_mode": no_trace_mode,
        "exclude": {str(p): index.sizes[Path(p).stem] for p in exclude},
        **{k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in rep.items()},
        "lengths": length_report(records, max_len),
    }
    config = {
        "traces": [str(p) for p in traces],
        "plain": [str(p) for p in plain],
        "no_trace_mode": no_trace_mode,
        "exclude": [str(p) for p in exclude],
        "lean": lean,
        "trace_weight": trace_weight,
        "empty_frac": empty_frac,
        "max_len": max_len,
        "cap": cap,
        "limit_plain": limit_plain,
        "think_filler": think_filler,
    }
    return {
        "format": FORMAT,
        "mode": tok.mode,
        "itos": list(tok.itos),
        "no_trace_mode": no_trace_mode,
        "has_traces": n_trace_ok > 0,
        "records": records,
        "meta": meta,
        "config": config,
        "report": report,
    }


# ------------------------------------------------------------------------------- graded effort
def draw_level(rec_id: Any, specs: Sequence[tuple[int, Any, float]]) -> int:
    """The effort level of a record: `unit_hash(f"{id}/effort")` against the cumulative
    probabilities of `specs` ((level, path, prob)) in ascending level order."""
    u = unit_hash(f"{rec_id}/effort")
    acc = 0.0
    ordered = sorted(specs, key=lambda x: x[0])
    for level, _, prob in ordered:
        acc += prob
        if u < acc:
            return int(level)
    return int(ordered[-1][0])  # rounding: u is in [0, 1) and the probabilities sum to 1


def check_graded_specs(specs: Sequence[tuple[int, Any, float]], any_proof: bool = False) -> None:
    levels = [int(x[0]) for x in specs]
    if len(set(levels)) != len(levels) or not set(levels) <= set(LEVELS):
        raise ValueError(f"graded levels must be distinct and in {LEVELS}, got {levels}")
    if abs(sum(float(x[2]) for x in specs) - 1.0) > 1e-9 or any(x[2] < 0 for x in specs):
        raise ValueError(f"graded probabilities must be >= 0 and sum to 1: {[x[2] for x in specs]}")
    for level, path, _ in specs:
        if level == 0 and path is not None and not any_proof:
            raise ValueError("level 0 takes no path (the level-256 record, thinks emptied)")
        if level != 0 and path is None:
            raise ValueError(f"level {level} needs a path")
    if 256 not in levels:
        raise ValueError("level 256 is required: its file defines the proof list and its order")


def build_graded(
    specs: Sequence[tuple[int, Any, float]],
    exclude: Sequence[str | Path] = (),
    lean: bool = False,
    max_len: int = 1024,
    any_proof: bool = False,
) -> dict[str, Any]:
    """Mode `lean_thinke` pairs, one effort level per proof (see the module docstring).
    `any_proof` (search traces, `searchtrace.py`): a level's record may prove the theorem with
    other actions than the level-256 record (same prompt), and level 0 may take its own file
    (every think empty: the budget check enforces it)."""
    check_graded_specs(specs, any_proof)
    specs = sorted(((int(a), b, float(c)) for a, b, c in specs), key=lambda x: x[0])
    E = EffortTokenizer()
    index = EvalIndex(exclude)
    by_level: dict[int, dict[Any, dict[str, Any]]] = {
        lv: {r["id"]: r for r in read_jsonl(path)} for lv, path, _ in specs if path is not None
    }
    top = read_jsonl(next(p for lv, p, _ in specs if lv == 256))
    excluded_by_set: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    missing: collections.Counter[int] = collections.Counter()
    counts: dict[int, int] = dict.fromkeys(LEVELS, 0)
    think_by_level: dict[int, list[int]] = {lv: [] for lv in LEVELS}
    records: list[list[Pair]] = []
    meta: list[dict[str, Any]] = []
    n_valid = 0
    for rec in top:
        hits = index.hit(rec["prompt"])
        for h in hits:
            excluded_by_set[h] += 1
        if hits:
            continue
        level = draw_level(rec["id"], specs)
        if level == 0 and (not any_proof or 0 not in by_level):
            use = {**rec, "steps": [{**st, "think": ""} for st in rec["steps"]]}
        else:
            use = by_level[level].get(rec["id"])
            if use is None:
                missing[level] += 1
                continue
        if use["prompt"] != rec["prompt"] or (
            not any_proof
            and [s["action"] for s in use["steps"]] != [s["action"] for s in rec["steps"]]
        ):
            raise ValueError(f"level {level} record {rec['id']} differs from the level-256 one")
        v = validate_trace(use, lean)
        if not v["ok"]:
            reasons[str(v["error"]).split(":")[0] if v["error"] else "unknown"] += 1
            continue
        # the level's effort token (effort.effort_id; importing effort here closes a cycle
        # effort -> ckpt -> stateview -> data)
        e = bytes([E.stoi[f"<e{level}>"]])
        pairs = [(p + e, q, n) for p, q, n in trace_pairs(use, E)]
        body = [max(n - 2, 0) for _, _, n in pairs]
        if max(body) > level:
            raise ValueError(
                f"{rec['id']}: think block of {max(body)} tokens exceeds level {level}"
            )
        n_valid += 1
        counts[level] += 1
        think_by_level[level] += [b for b in body if b > 0]
        records.append(pairs)
        meta.append(
            {
                "id": rec["id"],
                "prompt": rec["prompt"],
                "key": class_keys(rec["prompt"])[0],
                "source": "trace_emptied" if level == 0 else "trace",
                "n_lines": None,
                "level": level,
            }
        )
    ids = {x for r in records for p, q, _ in r for x in p + q}
    assert not ids or max(ids) < E.vocab_size <= 256
    report = {
        "traces_in": len(top),
        "traces_valid": n_valid,
        "excluded_by_set": dict(excluded_by_set),
        "invalid_trace_reasons": dict(reasons),
        "missing_at_level": {str(k): v for k, v in sorted(missing.items())},
        "effort_levels": counts,
        "effort_probs": {str(lv): pr for lv, _, pr in specs},
        "think_tokens_by_level": {str(lv): _dist(xs) for lv, xs in think_by_level.items()},
        "lengths": length_report(records, max_len),
    }
    config = {
        "graded": [[lv, None if p is None else str(p), pr] for lv, p, pr in specs],
        "exclude": [str(p) for p in exclude],
        "lean": lean,
        "max_len": max_len,
        "draw": "unit_hash(id/effort)",
        "any_proof": any_proof,
    }
    return {
        "format": FORMAT,
        "mode": E.mode,
        "itos": list(E.itos),
        "no_trace_mode": "empty",
        "has_traces": True,
        "records": records,
        "meta": meta,
        "config": config,
        "report": report,
    }


def parse_graded(specs: Sequence[str]) -> list[tuple[int, str | None, float]]:
    """CLI specs `0:PROB` / `LEVEL:PATH:PROB` -> (level, path or None, prob)."""
    out: list[tuple[int, str | None, float]] = []
    for s in specs:
        parts = s.split(":")
        if len(parts) == 2:
            out.append((int(parts[0]), None, float(parts[1])))
        elif len(parts) >= 3:
            out.append((int(parts[0]), ":".join(parts[1:-1]), float(parts[-1])))
        else:
            raise ValueError(f"bad --graded spec {s!r}: want 0:PROB or LEVEL:PATH:PROB")
    return out


def save(data: dict[str, Any], out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(data, f, protocol=5)
    rp = out.with_suffix(out.suffix + ".report.json")
    rp.write_text(json.dumps({"config": data["config"], **data["report"]}, indent=1))
    return rp


def load_pairs(path: str | Path, tok: Any | None = None) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    assert data["format"] == FORMAT, data.get("format")
    if tok is not None:
        assert list(tok.itos) == data["itos"], "pairs were built for another vocabulary"
    return data


def heldout_records(
    path: str | Path, tok: Any, no_trace_mode: str, limit: int = 2000
) -> list[list[Pair]]:
    """Validation pairs as the fork's trainer takes them (first `limit` records, no trace)."""
    out = []
    for rec in read_jsonl(path):
        try:
            out.append(plain_pairs(rec, tok, no_trace_mode))
        except (D.lean_tok.ParseFail, ValueError, AssertionError, IndexError, KeyError):
            continue
        if len(out) >= limit:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--traces", nargs="*", default=[], help="trace files, optionally path:weight")
    ap.add_argument("--plain", nargs="*", default=[])
    ap.add_argument(
        "--no-trace-mode", dest="no_trace_mode", choices=["empty", "none"], default="empty"
    )
    ap.add_argument(
        "--exclude", nargs="*", default=None, help="eval sets (default: DEFAULT_EXCLUDE)"
    )
    ap.add_argument(
        "--extra-exclude", nargs="*", default=[], help="added to --exclude (e.g. textbook_train)"
    )
    ap.add_argument("--no-lean", action="store_true", help="validate traces without the Lean check")
    ap.add_argument(
        "--trace-weight", type=int, default=1, help="default repeat count per valid trace record"
    )
    ap.add_argument(
        "--empty-frac", type=float, default=0.0, help="fraction of trace copies with empty thinks"
    )
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument(
        "--cap", type=int, default=0, help="drop plain records with n_lines > cap (0: keep all)"
    )
    ap.add_argument("--limit-plain", type=int, default=0)
    ap.add_argument(
        "--think-filler",
        action="store_true",
        help="replace every think-content token with <no>, keeping each block's length",
    )
    ap.add_argument(
        "--graded",
        nargs="+",
        default=None,
        metavar="SPEC",
        help="graded effort (lean_thinke): 0:PROB and LEVEL:PATH:PROB; excludes --traces/--plain",
    )
    ap.add_argument(
        "--graded-any-proof",
        action="store_true",
        help="graded: levels may prove a theorem differently (search traces); 0:PATH:PROB allowed",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.graded and (a.traces or a.plain):
        ap.error("--graded is mutually exclusive with --traces / --plain")
    exclude = list(DEFAULT_EXCLUDE if a.exclude is None else a.exclude) + list(a.extra_exclude)
    missing = [p for p in exclude if not os.path.exists(p)]
    if missing:
        print(f"error: exclusion sets not found: {missing}", file=sys.stderr)
        return 1
    if a.graded:
        data = build_graded(
            parse_graded(a.graded),
            exclude,
            lean=not a.no_lean,
            max_len=a.max_len,
            any_proof=a.graded_any_proof,
        )
        rp = save(data, a.out)
        r = data["report"]
        print(
            f"graded {r['traces_valid']}/{r['traces_in']} valid, levels {r['effort_levels']}, "
            f"missing {r['missing_at_level']}, invalid {r['invalid_trace_reasons']}; "
            f"{r['lengths']['pairs']} pairs, {r['lengths']['pairs_over_max_len']} over "
            f"{a.max_len}; excluded {r['excluded_by_set']} -> {a.out}, {rp}"
        )
        return 0
    data = build(
        a.traces,
        a.plain,
        a.no_trace_mode,
        exclude,
        lean=not a.no_lean,
        trace_weight=a.trace_weight,
        empty_frac=a.empty_frac,
        max_len=a.max_len,
        cap=a.cap,
        limit_plain=a.limit_plain,
        think_filler=a.think_filler,
    )
    rp = save(data, a.out)
    r = data["report"]
    print(
        f"traces {r['traces_valid']}/{r['traces_in']} valid, "
        f"plain {r['plain_used']}/{r['plain_in']}; "
        f"{r['lengths']['pairs']} pairs, max {r['lengths']['pair_tokens'].get('max')} tokens, "
        f"{r['lengths']['pairs_over_max_len']} over {a.max_len}; excluded {r['excluded_by_set']}; "
        f"invalid {r['invalid_trace_reasons']} -> {a.out}, {rp}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
