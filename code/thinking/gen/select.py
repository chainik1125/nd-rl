"""Stratified, seeded selection of source proofs, excluding every evaluation set by renaming class.

Strata: line-count bin, case split (ORE), classical step (DN by default), premise count.  The
renaming class is `code.ndbench.core.keys.canon_key` (atoms relabelled by first appearance, the
fork's `gen.canon_key`), and also `pkey` (the same, minimised over premise orders): a source row is
excluded if either key matches any evaluation item.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from code.ndbench.core.keys import canon_key, pkey, prompt_to_thm

from . import dsl

K12 = Path(
    os.environ.get("ND_K12", "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl")
)
TAKEHOME_DATA = (
    Path(os.environ.get("ND_TAKEHOME", "/scratch/cowifornia/nd-rl/think/nd-takehome")) / "data"
)
OOD_SETS = Path("/scratch/cowifornia/nd-rl/think/ood_check/sets")
DAN_HELDOUT = Path("/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/heldout.jsonl")
STRATA = ("lines", "ore", "classical", "prem")


def default_excludes() -> dict[str, Path]:
    """Evaluation sets no trace may be generated for (any renaming of them)."""
    ex = {
        p.stem if p.stem.startswith("ood_") else f"ood_{p.stem}": p
        for p in sorted(OOD_SETS.glob("*.jsonl"))
    }
    tb = TAKEHOME_DATA / "eval_only" / "textbook72"
    ex["textbook72_dev"] = tb / "textbook_dev.jsonl"
    ex["textbook72_train14"] = tb / "textbook_train.jsonl"
    ex["fork_p2_heldout"] = DAN_HELDOUT
    for n in ("candidate_v0", "candidate_v1", "batch3", "hand_proved_14"):
        ex[f"charles_{n}"] = TAKEHOME_DATA / "gt" / f"{n}.jsonl"
    ex["bs_holdout250"] = TAKEHOME_DATA / "bs" / "holdout250.jsonl"
    ex["pilot_examples"] = dsl.PILOT
    return ex


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def keys_of(prompt: str) -> tuple[str, str]:
    th = prompt_to_thm(prompt)
    return canon_key(th), pkey(th)


class Exclusion:
    def __init__(self, sets: dict[str, Path]):
        self.keys: dict[str, set[str]] = {}
        self.pkeys: dict[str, set[str]] = {}
        self.meta: dict[str, dict[str, Any]] = {}
        for label, path in sets.items():
            rows = read_jsonl(path)
            ks = [keys_of(r["prompt"]) for r in rows]
            self.keys[label] = {k for k, _ in ks}
            self.pkeys[label] = {p for _, p in ks}
            self.meta[label] = {
                "path": str(path),
                "sha256": sha256(path),
                "n_items": len(rows),
                "n_classes": len(self.keys[label]),
            }

    def hits(self, key: str, pk: str) -> list[str]:
        return [lb for lb in self.keys if key in self.keys[lb] or pk in self.pkeys[lb]]


def parse_bins(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.append((int(lo), int(hi or lo)))
    return out


def stratum(
    rec: dict[str, Any], bins: list[tuple[int, int]], classical: set[str]
) -> dict[str, Any]:
    n = int(rec["n_lines"])
    b = next((f"{lo}-{hi}" for lo, hi in bins if lo <= n <= hi), "other")
    rules = set(rec.get("rules", []))
    return {
        "lines": b,
        "ore": "ORE" in rules,
        "classical": bool(rules & classical),
        "prem": int(rec.get("n_prem", 0)),
    }


def cell_of(st: dict[str, Any], dims: Iterable[str]) -> str:
    return "|".join(f"{d}={st[d]}" for d in dims)


def _git() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]

    def run(*a: str) -> str:
        try:
            r = subprocess.run(
                ["git", "-C", str(root), *a], capture_output=True, text=True, timeout=10
            )
            return r.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _ok_proof(rec: dict[str, Any]) -> tuple[int, int] | None:
    try:
        steps = dsl.reference_steps(rec["prompt"], rec["proof"])
    except Exception:
        return None
    return len(steps), sum(s.kind == "decision" for s in steps)


def select(
    rows: list[dict[str, Any]],
    n: int,
    seed: int,
    exclusion: Exclusion | None,
    dims: tuple[str, ...] = STRATA,
    bins: str = "2-4,5-7,8-10,11-12",
    classical: str = "DN",
    alloc: str = "balanced",
    id_prefix: str = "k12",
    source_label: str = "K12",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """`balanced`: round-robin over the non-empty cells (seeded order), so rare cells are
    over-represented; `proportional`: largest-remainder allocation by cell size."""
    b = parse_bins(bins)
    cl = set(classical.split(",")) if classical else set()
    cells: dict[str, list[int]] = defaultdict(list)
    excluded: Counter[str] = Counter()
    strata = {}
    for i, r in enumerate(rows):
        if exclusion is not None:
            hits = exclusion.hits(*keys_of(r["prompt"]))
            if hits:
                excluded.update(hits)
                excluded["_rows"] += 1
                continue
        st = stratum(r, b, cl)
        strata[i] = st
        cells[cell_of(st, dims)].append(i)
    rng = random.Random(seed)
    order = sorted(cells)
    rng.shuffle(order)
    for c in order:
        rng.shuffle(cells[c])
    quota: dict[str, int] = {}
    if alloc == "proportional":
        tot = sum(len(cells[c]) for c in order)
        raw = {c: n * len(cells[c]) / tot for c in order}
        quota = {c: int(raw[c]) for c in order}
        for c in sorted(order, key=lambda c: raw[c] - quota[c], reverse=True)[
            : n - sum(quota.values())
        ]:
            quota[c] += 1
    elif alloc != "balanced":
        raise ValueError(f"alloc must be balanced or proportional, not {alloc!r}")
    picked: list[dict[str, Any]] = []
    ptr = dict.fromkeys(order, 0)
    taken: Counter[str] = Counter()
    failures = 0
    live = list(order)
    while len(picked) < n and live:
        nxt = []
        for c in live:
            if len(picked) >= n:
                break
            if alloc == "proportional" and taken[c] >= quota[c]:
                continue
            while ptr[c] < len(cells[c]):
                i = cells[c][ptr[c]]
                ptr[c] += 1
                ok = _ok_proof(rows[i])
                if ok is None:
                    failures += 1
                    continue
                r = rows[i]
                picked.append(
                    {
                        "id": f"{id_prefix}_{i}",
                        "source": f"{source_label} row {i}",
                        "row": i,
                        "prompt": r["prompt"],
                        "proof": r["proof"],
                        "n_lines": r.get("n_lines"),
                        "rules": r.get("rules", []),
                        "n_prem": r.get("n_prem"),
                        "strata": strata[i],
                        "n_steps": ok[0],
                        "n_decision": ok[1],
                    }
                )
                taken[c] += 1
                break
            if ptr[c] < len(cells[c]):
                nxt.append(c)
        live = nxt
    meta = {
        "n_requested": n,
        "n_selected": len(picked),
        "seed": seed,
        "alloc": alloc,
        "strata": list(dims),
        "line_bins": bins,
        "classical_rules": sorted(cl),
        "rows_excluded": excluded.pop("_rows", 0),
        "rows_excluded_by_set": dict(excluded),
        "decompose_failures": failures,
        "cells": {c: {"available": len(cells[c]), "selected": taken[c]} for c in sorted(cells)},
    }
    return picked, meta


def run(
    out_dir: Path,
    source: Path = K12,
    n: int = 50,
    seed: int = 0,
    excludes: dict[str, Path] | None = None,
    **kw: Any,
) -> tuple[Path, dict[str, Any]]:
    ex_sets = default_excludes() if excludes is None else excludes
    exclusion = Exclusion(ex_sets) if ex_sets else None
    rows = read_jsonl(source)
    picked, meta = select(rows, n, seed, exclusion, **kw)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "selection.jsonl"
    with open(path, "w") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {"path": str(source), "sha256": sha256(source), "n_rows": len(rows)},
        **meta,
        "exclusions": exclusion.meta if exclusion else {},
        "renaming_class": "code.ndbench.core.keys canon_key or pkey (premise-order invariant)",
        "git": _git(),
        "argv": sys.argv,
        "output_sha256": sha256(path),
    }
    with open(out_dir / "selection.meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return path, meta
