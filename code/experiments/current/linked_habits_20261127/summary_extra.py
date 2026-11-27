"""Numbers for the experiment summary that the read-out (lh_read.py at 12fc233) does not print, computed with the
read-out's own estimators (lh_read.Reader, lh_stats; the same theorems, bootstrap weights and 50-site rule) and written
to LH/read/summary_extra.json for the figure scripts. CPU only; nothing else is written.

1. grpo_vs_base: each GRPO line (reward-model line `rl_<c>_rlhfreit_s<k>`, plain line `rl_<c>_anc_s<k>`, every RL seed
   found) against its base at every measured iteration: D vs base per habit (A-share in the line minus in the base,
   untagged, marker_plain, Lean-accepted proofs) and the pooled A-share of all notes, with 95 % bootstrap intervals
   over theorems; the shares themselves (points). The read-out prints only reward line minus plain line, which on the
   all-linked corpus is the sum of two opposite drifts (PREREG Amendment 10).
2. bases: untagged pass@1 / pass@8 on marker_plain of every base measured, with intervals, and second base minus first
   base where both exist (the base-to-base spread).
3. mixture: the GRPO training samples of the all-linked corpus (`runs/<line>/seg1/lean_dump.jsonl`), Lean-accepted and
   read by lh_profile, split into 40 chunks in file order (about one iteration each): the share of proofs with notes
   whose notes are all in author A's form, all in B's, or mixed (audit part 3, X3).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 5 PY summary_extra.py [--lh LH] [--code DIR] [--no-mixture]
"""

# ruff: noqa: E402, E501

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
LH = Path(os.environ.get("LH", "/scratch/cowifornia/nd-rl/linked_habits"))
GRPO_CORPORA = ("c24", "c4", "c0")
CHUNKS = 40


def _git(code: Path) -> str:
    try:
        rev = subprocess.run(["git", "-C", str(code), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()  # fmt: skip
        dirty = subprocess.run(["git", "-C", str(code), "status", "--porcelain", "--", "."], capture_output=True,
                               text=True, check=True).stdout.strip()  # fmt: skip
        return rev + (" (uncommitted changes in the folder)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def grpo_vs_base(R: Any, S: Any, lh: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    notes = list(R.notes)
    for c in GRPO_CORPORA:
        bdir = R.base_dir(c)
        bs = R.one(bdir)
        if bs is None:
            continue
        for run in sorted((lh / "runs").glob(f"rl_{c}_*_s*")):
            name = run.name
            line = "reward" if "_rlhf" in name else "plain" if "_anc_" in name else None
            if line is None:
                continue
            seed = int(name.rsplit("_s", 1)[1])
            for it in sorted((run / "seg1" / "authors").glob("it*")):
                ls = R.one(it)
                if ls is None or R.m(it) is None:
                    continue
                mm = {"b": R.m(bdir)["none"], "l": R.m(it)["none"]}
                entry = {
                    "corpus": c, "line": line, "seed": seed, "iteration": int(it.name[2:]), "run": name,
                    "vs_base": R.D_all([(1.0, it, bdir)], "reit"),
                    "notes_pooled": {
                        "base": float(S.pooled_share(bs, notes)), "line": float(S.pooled_share(ls, notes)),
                        "vs_base": S.est(lambda s: S.pooled_share(s["l"], notes) - S.pooled_share(s["b"], notes),
                                         mm, R.W["read"])},
                    "share": {h: _f(S.a_share(ls, h)) for h in R.habits},
                    "base_share": {h: _f(S.a_share(bs, h)) for h in R.habits},
                    "accepted_pct": _f(S.perf(ls)["accepted_pct"]),
                }  # fmt: skip
                out[f"{name}/{it.name}"] = entry
                print(
                    f"grpo {name} {it.name}: notes {S.fmt(entry['notes_pooled']['vs_base'])}",
                    flush=True,
                )
    return out


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else round(v, 3)


def bases(R: Any, S: Any, lh: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"levels": {}, "second_minus_first": {}}
    for d in sorted((lh / "measure").iterdir()):
        if d.name.endswith("_sanity") or R.m(d) is None:
            continue
        mm = {"x": R.m(d)["none"]}
        out["levels"][d.name] = {
            k: S.est(lambda s, k=k: S.perf(s["x"])[k], mm, R.W["read"]) for k in ("pass1", "pass8")
        }
    for c in ("c24", "c4", "c9", "c0"):
        d0, d1 = lh / "measure" / f"{c}_s0", lh / "measure" / f"{c}_s1"
        if R.m(d0) is None or R.m(d1) is None:
            continue
        mm = {"a": R.m(d1)["none"], "b": R.m(d0)["none"]}
        out["second_minus_first"][c] = {
            k: S.est(lambda s, k=k: S.perf(s["a"])[k] - S.perf(s["b"])[k], mm, R.W["read"])
            for k in ("pass1", "pass8")
        }
    return out


def mixture(P: Any, lh: Path) -> dict[str, Any]:
    tables = P.load_tables()
    notes = [h for h in P.HABITS if h.startswith("note_")]
    out: dict[str, Any] = {}
    for run in sorted((lh / "runs").glob("rl_c24_*_s*")):
        fn = run / "seg1" / "lean_dump.jsonl"
        if not fn.exists():
            continue
        with fn.open() as fh:
            n = sum(1 for _ in fh)
        counts = [[0, 0, 0] for _ in range(CHUNKS)]  # all A, all B, mixed
        with fn.open() as fh:
            for i, x in enumerate(fh):
                r = json.loads(x)
                if str(r.get("lean_ok")) != "True":
                    continue
                p = P.profile(r["prompt"], r["nd"], tables)
                if not p["ok"]:
                    continue
                a = sum(p["habits"][h]["a"] for h in notes)
                b = sum(p["habits"][h]["b"] for h in notes)
                if a or b:
                    counts[min(i * CHUNKS // n, CHUNKS - 1)][0 if not b else 1 if not a else 2] += 1
        chunks = []
        for a_only, b_only, mixed in counts:
            tot = a_only + b_only + mixed
            chunks.append({"proofs_with_notes": tot, "all_A": a_only / tot if tot else None,
                           "all_B": b_only / tot if tot else None, "mixed": mixed / tot if tot else None})  # fmt: skip
        out[run.name] = {"samples": n, "chunks": chunks}
        print(
            f"mixture {run.name}: all-A {chunks[0]['all_A']:.3f} -> {chunks[-1]['all_A']:.3f}",
            flush=True,
        )
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lh", default=str(LH))
    ap.add_argument(
        "--code", default=str(HERE), help="folder whose lh_read / lh_stats / lh_profile are used"
    )
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--no-mixture", action="store_true")
    ap.add_argument("--out", default=None, help="default LH/read/summary_extra.json")
    a = ap.parse_args(argv)
    lh, code = Path(a.lh), Path(a.code)
    sys.path.insert(0, str(code))
    import lh_profile as P
    import lh_read as LR
    import lh_stats as S

    t0 = time.time()
    R = LR.Reader(lh, ["c24", "c9", "c4", "c0"], a.B, a.seed, 50, 30.0)
    res: dict[str, Any] = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code": str(code), "code_rev": _git(code),
        "B": a.B, "seed": a.seed, "read_set": "marker_plain", "tag": "none",
        "intervals": "95 % bootstrap over theorems (the read-out's weights); base-to-base and run-to-run variance not in them",
        "grpo_vs_base": grpo_vs_base(R, S, lh), "bases": bases(R, S, lh),
    }  # fmt: skip
    old = Path(a.out) if a.out else lh / "read" / "summary_extra.json"
    if a.no_mixture and old.exists():
        res["mixture"] = json.loads(old.read_text()).get("mixture", {})
    else:
        res["mixture"] = mixture(P, lh)
    res["secs"] = round(time.time() - t0, 1)
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text(json.dumps(res, indent=1, default=str))
    print(f"wrote {old} ({res['secs']} s)")
    return res


if __name__ == "__main__":
    main()
