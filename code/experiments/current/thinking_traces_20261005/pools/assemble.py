"""Select theorems for pass-rate sampling, and assemble the hard RL pool (`rl_hard.jsonl`).

    P="python -m code.experiments.current.thinking_traces_20261005.pools.assemble"
    $P select --cands cands_clean.jsonl --lens 13-20 --per_len 340 --out long_cands.jsonl
    $P select --cands cands_clean.jsonl --lens 7-12 --per_len 1500 --out short_cands.jsonl
    (passrate.py on each, k 16, under T0_s0)
    $P rl --rows long/rows.jsonl short/rows.jsonl --sets long_cands.jsonl short_cands.jsonl \
        --short_lens 7-12 --long_lens 13-20 --n_short 2500 --n_long 2500 \
        [--exclude_sets test_21_30.jsonl] --out rl_hard.jsonl

`select` draws --per_len candidates per generated length uniformly (seeded), within the shape limits
(--max_prompt prompt tokens, --max_state state tokens of the reference proof; K12's maxima are 156
and 386), skipping names in --exclude files.

`rl` builds the pool from the measured theorems (pass-rate rows of `passrate.py`) in two bands
(Leon's option B, 2026-10-06):
  * short (generated length --short_lens, K12's range): the OLMo 3 offline filter, pass@1 = n_ok/k
    <= --max_pass1 (0.625); unsolved theorems are kept unless --drop_unsolved;
  * long (--long_lens, at most 20 lines so the pool stays disjoint by length from the 21-30 test
    set): measured and reported, not filtered (unless --long_max_pass1).
Each band can be subsampled (--n_short / --n_long, seeded).  --exclude_sets drops any theorem whose
renaming class (`data.EvalIndex`: class key or premise-order key) is in those eval sets.  Rows keep
the ladder's `rl_targets.jsonl` fields (`n_lines` = generated reference length; `L_true` is not
computed: null) plus `band`, `t0_k`, `t0_n_ok`, `t0_pass1` and shape fields.  <out>.meta.json
(the .jsonl suffix replaced) gives the source, the filter and, per band, the pass-rate and n_lines
histograms of everything measured and of what was kept.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any

from code.thinking.data import EvalIndex, read_jsonl

T0 = "/scratch/cowifornia/nd-rl/think/runs/stage1/T0_s0/model.pt"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n attempts with c successes (as code.thinking.eval)."""
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


def lens(spec: str) -> list[int]:
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def names_in(paths: list[str]) -> set[str]:
    return {r["name"] for p in paths for r in read_jsonl(p)}


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def cmd_select(a: argparse.Namespace) -> None:
    rng = random.Random(a.seed)
    skip = names_in(a.exclude)
    by: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    shape_out = 0
    for r in read_jsonl(a.cands):
        if r["name"] in skip or r["n_lines"] not in lens(a.lens):
            continue
        if r["prompt_tokens"] > a.max_prompt or r["max_state"] > a.max_state:
            shape_out += 1
            continue
        by[r["n_lines"]].append(r)
    out = []
    for n in lens(a.lens):
        pool = by[n]
        rng.shuffle(pool)
        out += pool[: a.per_len]
    write_jsonl(a.out, out)
    print(
        json.dumps(
            {
                "out": a.out,
                "n": len(out),
                "per_len": {n: min(len(by[n]), a.per_len) for n in lens(a.lens)},
                "available": {n: len(by[n]) for n in lens(a.lens)},
                "shape_out": shape_out,
            }
        )
    )


def measured(rows_paths: list[str], sets: list[str]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows = {}
    for p in rows_paths:
        for r in read_jsonl(p):
            rows[r["name"]] = r
    recs = {r["name"]: r for p in sets for r in read_jsonl(p)}
    return [(recs[n], rows[n]) for n in recs if n in rows]


def pass_hist(rows: list[dict[str, Any]]) -> dict[str, int]:
    h = collections.Counter(f"{r['n_ok']}/{r['k']}" for r in rows)
    k = rows[0]["k"] if rows else 0
    return {f"{i}/{k}": h.get(f"{i}/{k}", 0) for i in range(k + 1)}


Pairs = list[tuple[dict[str, Any], dict[str, Any]]]


def by_len(pairs: Pairs, ks: tuple[int, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    g: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for rec, row in pairs:
        g[rec["n_lines"]].append(row)
    for n in sorted(g):
        rs = g[n]
        out[str(n)] = {
            "n": len(rs),
            **{
                f"pass@{k}_pct": round(
                    100.0 * sum(pass_at_k(r["k"], r["n_ok"], k) for r in rs) / len(rs), 2
                )
                for k in ks
            },
            "unsolved_pct": round(100.0 * sum(r["n_ok"] == 0 for r in rs) / len(rs), 2),
            "pass1_le_0.625_pct": round(
                100.0 * sum(r["n_ok"] / r["k"] <= 0.625 for r in rs) / len(rs), 2
            ),
        }
    return out


def pool_row(rec: dict[str, Any], row: dict[str, Any], band: str) -> dict[str, Any]:
    """The ladder's rl_targets fields (`n_lines` = generated reference length; no `L_true`), plus
    the T0 measurement and shape."""
    return {
        "name": rec["name"],
        "thm": rec["thm"],
        "key": rec["key"],
        "prompt": rec["prompt"],
        "n_lines": rec["n_lines"],
        "L_true": None,
        "minlen_bound": None,
        "source": "gen",
        "schema": None,
        "gen_lines": rec["n_lines"],
        "n_prem": rec["n_prem"],
        "rules": rec["rules"],
        "gen_proof": rec["proof"],
        "band": band,
        "t0_k": row["k"],
        "t0_n_ok": row["n_ok"],
        "t0_pass1": round(row["n_ok"] / row["k"], 4),
        "prompt_tokens": rec["prompt_tokens"],
        "formula_depth": rec["formula_depth"],
        "max_state": rec["max_state"],
    }


def cmd_rl(a: argparse.Namespace) -> None:
    rng = random.Random(a.seed)
    pairs = measured(a.rows, a.sets)
    hits: collections.Counter[str] = collections.Counter()
    if a.exclude_sets:
        idx = EvalIndex(a.exclude_sets)
        kept = []
        for rec, row in pairs:
            h = idx.hit(rec["prompt"])
            for x in h:
                hits[x] += 1
            if not h:
                kept.append((rec, row))
        pairs = kept
    short = [(rec, row) for rec, row in pairs if rec["n_lines"] in lens(a.short_lens)]
    long_ = [(rec, row) for rec, row in pairs if rec["n_lines"] in lens(a.long_lens)]
    s_keep = [
        (rec, row)
        for rec, row in short
        if row["n_ok"] / row["k"] <= a.max_pass1 and not (a.drop_unsolved and row["n_ok"] == 0)
    ]
    l_keep = [
        (rec, row)
        for rec, row in long_
        if a.long_max_pass1 is None or row["n_ok"] / row["k"] <= a.long_max_pass1
    ]
    for xs, n in ((s_keep, a.n_short), (l_keep, a.n_long)):
        xs.sort(key=lambda p: p[0]["name"])
        rng.shuffle(xs)
        if n:
            del xs[n:]
    out = [pool_row(rec, row, "short") for rec, row in s_keep]
    out += [pool_row(rec, row, "long") for rec, row in l_keep]
    rng.shuffle(out)
    write_jsonl(a.out, out)

    def band_meta(meas: Pairs, kept: Pairs, rule: str) -> dict[str, Any]:
        return {
            "rule": rule,
            "measured": len(meas),
            "measured_pass_hist": pass_hist([row for _, row in meas]),
            "measured_by_n_lines": by_len(meas, (1, 16)),
            "kept": len(kept),
            "kept_unsolved": sum(row["n_ok"] == 0 for _, row in kept),
            "kept_pass1_le_0.625": sum(row["n_ok"] / row["k"] <= 0.625 for _, row in kept),
            "kept_pass_hist": pass_hist([row for _, row in kept]),
            "kept_n_lines_hist": dict(
                sorted(collections.Counter(rec["n_lines"] for rec, _ in kept).items())
            ),
            "kept_mean_pass1": round(
                sum(row["n_ok"] / row["k"] for _, row in kept) / max(1, len(kept)), 4
            ),
        }

    k = pairs[0][1]["k"] if pairs else None
    meta = {
        "file": str(a.out),
        "n": len(out),
        "source": {
            "generator": "Dan's fork gen.sample_one(long=True) with Gen(max_prem=3, max_depth=3): "
            "the strict long generator of the ladder's rl_targets / transfer_long (fork "
            "4dcd0e4f), contradictory premises dropped (pools/gen_candidates.py, "
            "raw/cands.jsonl.summary.json)",
            "decontamination": "pools/prep.py (cands_clean.jsonl.report.json): renaming-class "
            "and premise-order keys against data.DEFAULT_EXCLUDE, textbook72 train, K12 train "
            "(all trace files' prompts are K12 prompts), and every jsonl under the fork's data/ "
            "(ladder + take-home rl_targets and transfer, transfer_long*, p2/r1-r3/bs/gt/lj)",
            "exclude_sets": {"files": a.exclude_sets, "hits": dict(hits)},
            "candidates": a.sets,
            "shape": "prompt <= 156 tokens, reference-proof states <= 386 tokens (K12's maxima)",
        },
        "filter": {
            "model": a.model,
            "k": k,
            "temperature": 0.8,
            "arm": "plain, no think (passrate.py: sample.think_generate + sample.judge, Lean)",
            "rows": a.rows,
        },
        "bands": {
            "short": band_meta(
                short,
                s_keep,
                f"generated length {a.short_lens}, pass@1 = n_ok/k <= {a.max_pass1}"
                + (", n_ok >= 1" if a.drop_unsolved else " (unsolved kept)")
                + (f", subsampled to {a.n_short}" if a.n_short else ""),
            ),
            "long": band_meta(
                long_,
                l_keep,
                f"generated length {a.long_lens}, "
                + (
                    "measured, not filtered"
                    if a.long_max_pass1 is None
                    else f"pass@1 <= {a.long_max_pass1}"
                )
                + (f", subsampled to {a.n_long}" if a.n_long else ""),
            ),
        },
        "n_lines_hist": dict(sorted(collections.Counter(r["n_lines"] for r in out).items())),
        "pass_hist": pass_hist([{"n_ok": r["t0_n_ok"], "k": r["t0_k"]} for r in out]),
    }
    Path(str(a.out).replace(".jsonl", ".meta.json")).write_text(json.dumps(meta, indent=1))
    print(json.dumps({"n": len(out), "short": len(s_keep), "long": len(l_keep), "hits": hits}))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--cands", required=True)
    s.add_argument("--lens", required=True)
    s.add_argument("--per_len", type=int, required=True)
    s.add_argument("--exclude", nargs="*", default=[])
    s.add_argument("--max_prompt", type=int, default=156)
    s.add_argument("--max_state", type=int, default=386)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    r = sub.add_parser("rl")
    r.add_argument("--rows", nargs="+", required=True)
    r.add_argument("--sets", nargs="+", required=True)
    r.add_argument("--short_lens", default="7-12")
    r.add_argument("--long_lens", default="13-20")
    r.add_argument("--max_pass1", type=float, default=0.625, help="short band (OLMo 3)")
    r.add_argument(
        "--long_max_pass1", type=float, default=None, help="long band (default: no filter)"
    )
    r.add_argument("--drop_unsolved", action="store_true", help="short band")
    r.add_argument("--n_short", type=int, default=0, help="subsample the short band (0: all)")
    r.add_argument("--n_long", type=int, default=0, help="subsample the long band (0: all)")
    r.add_argument(
        "--exclude_sets", nargs="*", default=[], help="eval sets: drop their renaming classes"
    )
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--model", default=T0)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"select": cmd_select, "rl": cmd_rl}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
