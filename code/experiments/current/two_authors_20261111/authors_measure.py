"""Habit measurement of a checkpoint: sample, judge with Lean, profile, write counts (two authors, 2026-11-11).

    python authors_measure.py run --ckpt CKPT [--ckpt_id ID] [--tags A,B,none] [--sets marker_plain,route_unseen,marker_heldout_class]
                                  [--k 32] [--seed 0] [--batch 256] [--device cuda] [--limit N] [--out_dir DIR] [--raw_dir DIR]
                                  [--resume] [--dry-run]
    python authors_measure.py danread --read RUN/reads/it0010.json [--ckpt_id ID] [--out FILE]   # profile a Dan read
    python authors_measure.py estimate [--k 32] [--sets ...] [--tags A,B,none] [--accept 0.6]     # GPU time per checkpoint

`run`: for each tag condition (A, B, none) and each set, Dan's sampler (`state_sample.env_generate`, as
../author_tags_20261110/tag_sample.py calls it: T 0.8, 256 tokens per action, 48 steps, his name-base draw) on every
theorem x k, with the Lean gate (Lean alone decides). The tag goes in through the tag tokenizer (`tag_core`) and the
spy checks every decoder input carries it. The noise seed is the same for every tag (`seed * 1000 + set index`), so
the three tag conditions are paired draws. Every accepted proof is profiled (`authors_profile.profile`). Written:
  OUT_DIR/<set>__<tag>.jsonl          one row per theorem, counts only (below), never proof text;
  OUT_DIR/<set>__<tag>.summary.json   timing, sampler stats, spy counts, totals;
  RAW_DIR/<set>__<tag>.jsonl.gz       the accepted proofs (theorem, attempt, proof): raw, never committed.
Defaults: OUT_DIR = TA/measure/<ckpt_id>, RAW_DIR = TA/raw/<ckpt_id>. Nothing is overwritten (a (set, tag) whose
summary exists is refused; `--resume` skips it instead).

Row schema (one per checkpoint, tag, set, theorem):
  ckpt, tag, set, theorem, key, n_lines (the pool record's length), ref_lines (the set's reference proof, written
  lines), needs_route, has_F, k, noise_seed, attempts, accepted, unparsed (accepted but not parsable by habits.build),
  route (accepted proofs that are sigma+ with R lines followed: authors_profile), route_knock and sigma (Dan's knock
  and has_sigma, for comparison), lines (sum of written lines), dead (sum of dead lines), len_hist {lines: proofs},
  habits {order|reit|conj: {sites, a, b}} (conj: free sites), bundle {A, B, mixed, none}, joint {"oA.rB.c-": n},
  split {route|plain: {proofs, habits, bundle, joint}} (the same counts over route proofs and over the others),
  reit_fit {fits|nofit: {proofs, sites, a}} (reiteration over accepted proofs whose fully reiterated form fits in 12
  lines, and over the others: the corpus gave A's reiteration only where it fit), reit_by_lines {lines: [sites, a]}.

`danread`: the strict 106 of Dan's measure, read by tag_danread.py (k 64 on the 106 under one tag), profiled from the
read's `.proofs.jsonl`. Dan's judge keeps distinct accepted proofs only, so these rows count distinct proofs
(`unit: distinct`), with his `n_ok` as `accepted_attempts`. Set name `strict106`.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import authors_profile as AP  # noqa: E402
import authors_sets as AS  # noqa: E402

sys.path.insert(0, str(AP.TAGS_DIR))
import tag_core as T  # noqa: E402

TA = AS.TA
SETS = AS.SETS
SET_INDEX = {s: i for i, s in enumerate((*SETS, "strict106"))}
TEMPERATURE, MAX_ACTION, MAX_STEPS = 0.8, 256, 48
# GPU rates (see `estimate`): the tag experiment's base read sampled (106 + 240) x 32 = 11,072 attempts in about 120 s
# per tag on one serrano RTX 3090 (200 W cap), sampling and Lean together; Dan's best-recipe Stage-1 ran about 24,000
# steps in 1,200 s on an A40 against 1,489-1,701 s on serrano's 3090s, so an A40 is taken as 1.3 x a 3090.
RATE_3090 = 11072 / 120.0
A40_OVER_3090 = 1.3


def ckpt_id_of(path: str | Path) -> str:
    """`stage1_<name>.pt` -> <name>; `<line>_it<i>_snap.pt` -> <line>@it<i>; anything else: its stem."""
    p = Path(path)
    m = re.fullmatch(r"(.+)_it(\d+)_(?:snap|rolling|alarm)\.pt", p.name)
    if m:
        return f"{m.group(1)}@it{int(m.group(2))}"
    return p.name.removesuffix(".pt").removeprefix("stage1_")


def empty_counts() -> dict[str, Any]:
    return {"proofs": 0, "habits": {h: {"sites": 0, "a": 0, "b": 0} for h in AP.HABITS},
            "bundle": {"A": 0, "B": 0, "mixed": 0, "none": 0}, "joint": {}}  # fmt: skip


def add_profile(c: dict[str, Any], p: dict[str, Any]) -> None:
    c["proofs"] += 1
    for h in AP.HABITS:
        for f in ("sites", "a", "b"):
            c["habits"][h][f] += p[h][f]
    c["bundle"][p["bundle"] or "none"] += 1
    c["joint"][p["joint"]] = c["joint"].get(p["joint"], 0) + 1


def theorem_row(
    thm: dict, proofs: list[str], attempts: int, base: dict[str, Any]
) -> dict[str, Any]:
    """Counts for one theorem from its accepted proofs (module docstring)."""
    ref = thm.get("ref_proof")
    row = {**base, "theorem": thm["name"], "key": thm.get("key"), "n_lines": thm.get("n_lines"),
           "ref_lines": ref.split().count(";") if ref else None,
           "needs_route": thm.get("needs_route"), "has_F": thm.get("has_F"), "attempts": attempts,
           "accepted": len(proofs), "unparsed": 0, "route": 0, "route_knock": 0, "sigma": 0, "lines": 0,
           "dead": 0, "len_hist": {}}  # fmt: skip
    allc = empty_counts()
    split = {"route": empty_counts(), "plain": empty_counts()}
    fit = {k: {"proofs": 0, "sites": 0, "a": 0} for k in ("fits", "nofit")}
    by_lines: dict[str, list[int]] = {}
    for pr in proofs:
        p = AP.profile(thm["prompt"], pr)
        row["route_knock"] += bool(p.get("route_knock"))
        row["sigma"] += bool(p.get("sigma"))
        if not p["ok"]:
            row["unparsed"] += 1
            row["route"] += bool(p.get("route"))
            continue
        row["route"] += bool(p["route"])
        row["lines"] += p["lines"]
        row["dead"] += p["dead"]
        L = str(p["lines"])
        row["len_hist"][L] = row["len_hist"].get(L, 0) + 1
        add_profile(allc, p)
        add_profile(split["route" if p["route"] else "plain"], p)
        if p["reit"]["sites"]:
            f = fit["fits" if p["reit_fits"] else "nofit"]
            f["proofs"] += 1
            f["sites"] += p["reit"]["sites"]
            f["a"] += p["reit"]["a"]
            bl = by_lines.setdefault(L, [0, 0])
            bl[0] += p["reit"]["sites"]
            bl[1] += p["reit"]["a"]
    row["habits"], row["bundle"], row["joint"] = allc["habits"], allc["bundle"], allc["joint"]
    row["split"] = split
    row["reit_fit"], row["reit_by_lines"] = fit, by_lines
    return row


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def pair_paths(out_dir: Path, raw_dir: Path, s: str, tag: str) -> tuple[Path, Path, Path]:
    return (
        out_dir / f"{s}__{tag}.jsonl",
        out_dir / f"{s}__{tag}.summary.json",
        raw_dir / f"{s}__{tag}.jsonl.gz",
    )


# ------------------------------------------------------------------------------------------------ run
def run(a: argparse.Namespace) -> dict:
    tags = [T.check_tag(t) for t in a.tags.split(",")]
    sets = a.sets.split(",")
    for s in sets:
        assert s in SETS, f"unknown set {s}; sets: {SETS}"
    cid = a.ckpt_id or ckpt_id_of(a.ckpt)
    out_dir = Path(a.out_dir) if a.out_dir else TA / "measure" / cid
    raw_dir = Path(a.raw_dir) if a.raw_dir else TA / "raw" / cid
    sets_root = Path(a.sets_root)
    plan, skip = [], []
    for tag in tags:
        for s in sets:
            rows_p, sum_p, raw_p = pair_paths(out_dir, raw_dir, s, tag)
            if sum_p.exists():
                if not a.resume:
                    raise SystemExit(
                        f"{sum_p} exists: a measurement is never overwritten (--resume skips it)"
                    )
                skip.append((s, tag))
                continue
            plan.append((s, tag))
    n_thm = {s: len(AS.load_set(s, sets_root)[: a.limit or None]) for s in sets}
    info = {"ckpt": str(a.ckpt), "ckpt_id": cid, "out_dir": str(out_dir), "raw_dir": str(raw_dir), "k": a.k,
            "pairs": plan, "skipped": skip, "theorems": n_thm,
            "attempts": sum(n_thm[s] * a.k for s, _ in plan), "device": a.device, "batch": a.batch}  # fmt: skip
    if a.dry_run:
        print(json.dumps(info, indent=1))
        return info
    os.environ.setdefault("ND_OFFLINE", "1")
    os.environ.setdefault("ND_REGISTRY", "0")
    os.environ.setdefault("LEAN_GATE_WORKERS", "8")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LEAN_GATE_LOG"] = str(out_dir / "lean_gate.jsonl")  # never Dan's tree's default log
    T.install_dan()
    import torch
    from model import load_ckpt  # type: ignore[import-not-found]
    from state_sample import env_generate, env_stats_json  # type: ignore[import-not-found]

    if a.device == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    elif os.environ.get(
        "DANREAD_MEM_FRAC"
    ):  # beside a training process (an RL line's snapshot reads): a hard cap
        torch.cuda.set_per_process_memory_fraction(float(os.environ["DANREAD_MEM_FRAC"]))
    mode = T.ckpt_mode(a.ckpt)
    if mode != T.MODE and any(t != "none" for t in tags):
        raise SystemExit(f"{a.ckpt} is a {mode} checkpoint: tags A/B need {T.MODE}")
    model, tok, _ = load_ckpt(a.ckpt, a.device)
    t_all = time.time()
    for s, tag in plan:
        rows_p, sum_p, raw_p = pair_paths(out_dir, raw_dir, s, tag)
        thms = AS.load_set(s, sets_root)[: a.limit or None]
        prompts = [t["prompt"] for t in thms for _ in range(a.k)]
        seed = a.seed * 1000 + SET_INDEX[s]
        T.set_process_tag(tag)
        T.spy_dan(tag)
        s0, by0 = T.SPY["states"], dict(T.SPY["by_tag"])
        st: dict = {}
        t0 = time.time()
        nds = env_generate(model, tok, prompts, greedy=False, temperature=TEMPERATURE, max_action=MAX_ACTION,
                           max_steps=MAX_STEPS, batch=a.batch, seed=seed, stats=st, gate=True)  # fmt: skip
        t_gen = time.time() - t0
        base = {"ckpt": cid, "tag": tag, "set": s, "k": a.k, "noise_seed": seed}
        rows, raw = [], []
        for j, thm in enumerate(thms):
            outs = nds[j * a.k : (j + 1) * a.k]
            acc = [(i, x) for i, x in enumerate(outs) if not x.startswith(("LEANREJ", "LEANPARSE"))]
            rows.append(theorem_row(thm, [x for _, x in acc], a.k, base))
            raw += [{"theorem": thm["name"], "attempt": i, "proof": x} for i, x in acc]
        t_prof = time.time() - t0 - t_gen
        with gzip.open(raw_p.with_name(raw_p.name + ".tmp"), "wt", encoding="utf-8") as f:
            for r in raw:
                f.write(json.dumps(r) + "\n")
        raw_p.with_name(raw_p.name + ".tmp").replace(raw_p)
        _write_atomic(rows_p, "".join(json.dumps(r) + "\n" for r in rows))
        tot = {"attempts": len(nds), "accepted": sum(r["accepted"] for r in rows),
               "route": sum(r["route"] for r in rows), "unparsed": sum(r["unparsed"] for r in rows)}  # fmt: skip
        for h in AP.HABITS:
            si = sum(r["habits"][h]["sites"] for r in rows)
            tot[f"{h}_sites"], tot[f"{h}_a_share"] = (
                si,
                (sum(r["habits"][h]["a"] for r in rows) / si if si else None),
            )
        for k in ("fits", "nofit"):
            si = sum(r["reit_fit"][k]["sites"] for r in rows)
            tot[f"reit_{k}_sites"] = si
            tot[f"reit_{k}_a_share"] = sum(r["reit_fit"][k]["a"] for r in rows) / si if si else None
        tot["route_knock"] = sum(r["route_knock"] for r in rows)
        summ = {**info, "set": s, "tag": tag, "noise_seed": seed, "theorems": len(thms), "totals": tot,
                "secs_sample_and_lean": round(t_gen, 1), "secs_profile": round(t_prof, 1),
                "states_checked": T.SPY["states"] - s0,
                "spy_by_tag": {t: n - by0.get(t, 0) for t, n in T.SPY["by_tag"].items() if n - by0.get(t, 0)},
                "env": env_stats_json(st), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "host": os.uname().nodename, "temperature": TEMPERATURE, "max_action": MAX_ACTION,
                "max_steps": MAX_STEPS}  # fmt: skip
        summ.pop("pairs", None)
        _write_atomic(sum_p, json.dumps(summ, indent=1, default=str))
        sh = {
            h: (None if tot[f"{h}_a_share"] is None else round(100 * tot[f"{h}_a_share"], 1))
            for h in AP.HABITS
        }
        print(f"{cid} {s} tag {tag}: {tot['accepted']}/{tot['attempts']} accepted, route {tot['route']}, "
              f"A-form % {sh}; {t_gen:.0f} s sample+Lean, {t_prof:.0f} s profile", flush=True)  # fmt: skip
    print(f"{cid}: done in {time.time() - t_all:.0f} s", flush=True)
    return info


# ------------------------------------------------------------------------------------------------ danread
def strict106_names() -> list[str]:
    return (Path(AP.H.DAN) / "data" / "coe" / "sigma_heldout_strict.txt").read_text().split()


def danread(a: argparse.Namespace) -> Path:
    rp = Path(a.read)
    rd = json.loads(rp.read_text())
    pf = rp.with_suffix(".proofs.jsonl")
    proofs = {}
    if pf.exists():
        for line in pf.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                proofs[o["name"]] = o["proofs"]
    tag = rd.get("author_tag", "none")
    cid = a.ckpt_id or ckpt_id_of(rd["ckpt"])
    out = Path(a.out) if a.out else rp.with_name(rp.stem + ".authors.jsonl")
    if out.exists():
        raise SystemExit(f"{out} exists: never overwritten")
    prompts = {}
    tr = Path(AP.H.DAN) / "data" / "coe" / "transfer_track832.jsonl"
    names = strict106_names()
    want = set(names)
    for line in (
        tr.read_text().splitlines()
    ):  # only the strict 106 rows are read (the only rows a pod holds)
        o = json.loads(line)
        if o["name"] in want:
            prompts[o["name"]] = o
    base = {"ckpt": cid, "tag": tag, "set": "strict106", "k": rd.get("k"), "noise_seed": rd.get("noise_seed"),
            "unit": "distinct"}  # fmt: skip
    rows = []
    for n in names:
        thm = {"name": n, "prompt": prompts[n]["prompt"], "key": prompts[n].get("key"),
               "n_lines": prompts[n].get("n_lines"), "needs_route": True, "has_F": False}  # fmt: skip
        r = theorem_row(thm, proofs.get(n, []), rd.get("k"), base)
        r["accepted_attempts"] = (rd.get("sigma") or {}).get(n, {}).get("n_ok", 0)
        rows.append(r)
    _write_atomic(out, "".join(json.dumps(r) + "\n" for r in rows))
    return out


# ------------------------------------------------------------------------------------------------ estimate
def estimate(k: int, sets: list[str], tags: list[str], accept: float, prof_ms: float, sets_root: Path,
             share: float = 1.0) -> dict:  # fmt: skip
    """GPU and CPU time of one `run`, from the 3090 rate above (sampling + Lean) and a profile cost per accepted proof
    (measured in the CPU smoke). `share` < 1 when the GPU is shared with a GRPO process."""
    n = sum(len(AS.load_set(s, sets_root)) for s in sets)
    attempts = n * k * len(tags)
    g3090 = attempts / RATE_3090 / share
    ga40 = g3090 / A40_OVER_3090
    prof = attempts * accept * prof_ms / 1000
    return {"theorems": n, "attempts": attempts, "min_3090": round(g3090 / 60, 1), "min_A40": round(ga40 / 60, 1),
            "min_profile_cpu": round(prof / 60, 1), "min_total_A40": round((ga40 + prof) / 60, 1)}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ckpt_id", default=None)
    p.add_argument("--tags", default="A,B,none")
    p.add_argument("--sets", default=",".join(SETS))
    p.add_argument("--sets_root", default=str(AS.OUT))
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0, help="first N theorems of each set only (smoke)")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--raw_dir", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = sp.add_parser("danread")
    p.add_argument("--read", required=True)
    p.add_argument("--ckpt_id", default=None)
    p.add_argument("--out", default=None)
    p = sp.add_parser("estimate")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--sets", default=",".join(SETS))
    p.add_argument("--sets_root", default=str(AS.OUT))
    p.add_argument("--tags", default="A,B,none")
    p.add_argument("--accept", type=float, default=0.6)
    p.add_argument("--prof_ms", type=float, default=3.0)
    p.add_argument("--share", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a)
    elif a.cmd == "danread":
        print(danread(a))
    else:
        print(json.dumps(estimate(a.k, a.sets.split(","), a.tags.split(","), a.accept, a.prof_ms, Path(a.sets_root),
                                  a.share), indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
