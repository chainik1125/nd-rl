"""Size ladder: Stage-1 pools of different sizes with K12's content distribution (ideas.md H8 / E1).

Nested pools, every one a stratified superset of the smaller:
  k12_15k ⊂ k12_50k ⊂ k12 (155,000) ⊂ k12_500k   (a 1.5M pool was built on 2026-10-06 and removed on 2026-10-07 at Charles's request)
Below K12: random stratified subsets of K12 (same share per pruned length). Above K12: K12 plus new records from the
same generator at the same knobs (cap-6 forward mode for lengths 2–6, long mode for 7–12; `gen.py` with fresh seeds),
deduplicated by renaming class against K12 and each other, audited against the evaluation sets and the sealed 72,
and added per length to keep the pool flat as far as the generator can supply (deficits are reported).

    cd code && uv run python pretrain_data/size_ladder.py --cap6 ../data/pretrain_data/gen/cap6_big.jsonl \
        --long ../data/pretrain_data/gen/long_big.jsonl

Writes data/pretrain_data/pools/size/<name>.jsonl.gz, stats, and appends a section to docs/pretrain-data/pools.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pool_common as PC  # noqa: E402

OUT_DIR = os.path.join(PC.DATA, "pools", "size")
SIZES = [15_000, 50_000, 500_000]
RECORDS_SEEN = {"9.56M (scale-d384-p1200)": 3_180_545, "38M (scale-d768-p1200)": 1_221_707}


def stratified_subset(rows, n, rng):
    by_len = collections.defaultdict(list)
    for r in rows:
        by_len[r["n_lines"]].append(r)
    out = []
    total = len(rows)
    for L, rs in sorted(by_len.items()):
        k = round(n * len(rs) / total)
        out.extend(rng.sample(rs, min(k, len(rs))))
    rng.shuffle(out)
    return out


HEAD = "## Size ladder (ideas.md H8): the same content at four sizes"


def write_doc(stats_out, rep, n_deep, max_depth):
    """append the size-ladder section to pools.md (replacing an earlier one)"""
    W = [f"\n{HEAD}\n",
         f"Built by `size_ladder.py`. Below K12: stratified random subsets. Above K12: K12 plus new generator records at the same knobs "
         f"(cap-6 forward mode for 2–6, long mode for 7–12), box depth capped at {max_depth} like K12 ({rep.get('depth_over', 0):,} deeper records "
         f"set aside as `gen_depth3.jsonl.gz`, {n_deep:,} after audit), deduplicated by renaming class against K12, audited "
         f"({rep.get('audit:removed_sealed_twin', 0)} sealed twins and {rep.get('audit:removed_eval_twin', 0)} evaluation twins removed). "
         f"Epochs = records seen by Robbie's 1,200 s pretrains ÷ pool size (9.56M model: 3,180,545 records seen; 38M: 1,221,707).\n",
         "| pool | n | epochs at 9.56M | epochs at 38M | mean lines | prem ≥4 | depth ≥3 | ∧E share | ∨E share | per-length counts | deficit (length: missing) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in stats_out:
        e = list(s["epochs"].values())
        W.append(f"| {s['label']} | {s['n']:,} | {e[0]} | {e[1]} | {s.get('mean_lines', '')} | {s.get('prem_ge4', '')} | {s.get('depth_ge3', '')} | "
                 f"{s.get('ande_share', '')} | {s.get('ore_share', '')} | {s['by_len_counts']} | {s['deficit'] or '—'} |")
    W.append("")
    W.append("Join rates per 1,000 citations:")
    W.append("")
    cols = list(PC.JOINS)
    W.append("| pool | " + " | ".join(cols) + " |")
    W.append("| --- | " + " | ".join("---" for _ in cols) + " |")
    for s in stats_out:
        if "joins" in s:
            W.append(f"| {s['label']} | " + " | ".join(str(s["joins"][c]) for c in cols) + " |")
    doc = PC.ROOT + "/docs/pretrain-data/pools.md"
    text = open(doc).read()
    if HEAD in text:
        i = text.index(HEAD)
        j = text.find("\n## ", i + len(HEAD))
        text = text[:i].rstrip("\n") + "\n" + (text[j:] if j > 0 else "")
    with open(doc, "w") as f:
        f.write(text.rstrip("\n") + "\n" + "\n".join(W) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap6", nargs="+")
    ap.add_argument("--long", nargs="+")
    ap.add_argument("--doc_only", action="store_true", help="rewrite the pools.md section from pools/size/stats.json without rebuilding")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_stats", action="store_true")
    ap.add_argument("--max_depth", type=int, default=2, help="K12 has no box depth above 2 (its cap-6 half was depth-3-filtered); match it")
    a = ap.parse_args()
    if a.doc_only:
        d = json.load(open(os.path.join(OUT_DIR, "stats.json")))
        import gzip
        with gzip.open(os.path.join(OUT_DIR, "gen_depth3.jsonl.gz"), "rt") as f:
            n_deep = sum(1 for _ in f)
        write_doc(d["pools"], d["new_records"], n_deep, a.max_depth)
        print("rewrote the size-ladder section of pools.md")
        return
    assert a.cap6 and a.long, "--cap6 and --long are required to build"
    rng = random.Random(a.seed)
    k12 = PC.load_k12()
    k12_keys = set()
    for r in k12:
        k12_keys |= PC.BR.keys(r["prompt"])
    print("k12", len(k12), flush=True)

    # new generator records: dedup against K12 and each other by curry-aware class, verify, audit
    new_by_len = collections.defaultdict(list)
    deep = []
    rep = collections.Counter()
    seen = set(k12_keys)
    for path, tag in [(p, "gen_cap6") for p in a.cap6] + [(p, "gen_long") for p in a.long]:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # a truncated last line from a stopped generator
                    rep["bad_line"] += 1
                    continue
                if "prompt" not in r:
                    continue
                ks = PC.BR.keys(r["prompt"])
                if ks & seen:
                    rep["dup_or_k12"] += 1
                    continue
                seen |= ks
                rec, why = PC.make_record(r["prompt"], r["proof"], tag, contra_prem=r.get("contra_prem", False))
                if rec is None:
                    rep["verify_fail"] += 1
                    continue
                depth = max(ln["depth"] for ln in PC.parse(rec["proof"]))
                if depth > a.max_depth:
                    rep["depth_over"] += 1
                    deep.append(dict(rec, pool="gen_depth3"))
                    continue
                new_by_len[rec["n_lines"]].append(rec)
                rep["new"] += 1
    idx = PC.EvalIndex()
    deep, drep = idx.audit(deep)
    os.makedirs(OUT_DIR, exist_ok=True)
    PC.write(deep, os.path.join(OUT_DIR, "gen_depth3.jsonl.gz"))
    print("depth-3+ side pool:", len(deep), drep, flush=True)
    audited = {}
    for L, rs in new_by_len.items():
        kept, arep = idx.audit(rs)
        audited[L] = kept
        for k, v in arep.items():
            if k != "kept":
                rep[f"audit:{k}"] += v
    print("new records by length:", {L: len(v) for L, v in sorted(audited.items())}, dict(rep), flush=True)

    pools = {}
    for n in [s for s in SIZES if s < len(k12)]:
        pools[f"k12_{n // 1000}k"] = stratified_subset(k12, n, rng)
    pools["k12"] = list(k12)
    k12_by_len = collections.Counter(r["n_lines"] for r in k12)
    deficits = {}
    for n in [s for s in SIZES if s > len(k12)]:
        target = n // 11
        out = list(k12)
        deficit = {}
        for L in range(2, 13):
            need = max(0, target - k12_by_len[L])
            have = audited.get(L, [])
            take = have[:need]
            out.extend(take)
            if len(take) < need:
                deficit[L] = need - len(take)
        # fill any deficit from the lengths that have spare records, so the pool reaches its nominal size
        short = n - len(out)
        if short > 0:
            spare = []
            for L in range(2, 13):
                used = max(0, target - k12_by_len[L])
                spare.extend(audited.get(L, [])[used:])
            rng.shuffle(spare)
            out.extend(spare[:short])
        rng.shuffle(out)
        pools[f"k12_{n // 1000}k"] = out
        deficits[f"k12_{n // 1000}k"] = deficit
    os.makedirs(OUT_DIR, exist_ok=True)
    rows_md = []
    stats_out = []
    for name, rows in pools.items():
        if name != "k12":
            PC.write(rows, os.path.join(OUT_DIR, f"{name}.jsonl.gz"))
        s = PC.stats(rows, name) if not a.no_stats else dict(label=name, n=len(rows))
        s["epochs"] = {k: round(v / len(rows), 2) for k, v in RECORDS_SEEN.items()}
        s["by_len_counts"] = dict(sorted(collections.Counter(r["n_lines"] for r in rows).items()))
        s["deficit"] = deficits.get(name, {})
        stats_out.append(s)
        print(name, len(rows), s["epochs"], flush=True)
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(dict(pools=stats_out, new_records=rep, new_by_len={L: len(v) for L, v in sorted(audited.items())}), f, indent=1, ensure_ascii=False)
    write_doc(stats_out, rep, len(deep), a.max_depth)
    print("wrote", OUT_DIR, "and appended to pools.md")


if __name__ == "__main__":
    main()
