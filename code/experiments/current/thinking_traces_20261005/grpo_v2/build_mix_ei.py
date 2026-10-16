"""The targets of the EI knob pilot ARM_k3p: a fixed mix of 5,000 L3 targets, two thirds pool
rows and one third 3-premise rewrites, with its preflight.

    python -m code.experiments.current.thinking_traces_20261005.grpo_v2.build_mix_ei [--out MIX]

GRPO's merged recipe (line.sh ARM_mb) draws a third of each iteration's groups from the L3
3-premise rows (`--targets_extra L3_3p.jsonl --targets_extra_frac 0.33`). EI samples every target
k times in one round, so ARM_k3p gets the same share as a fixed file instead: `N_POOL` rows
sampled from the L3 pool plus `N_3P` from `curriculum_3p/L3_3p.jsonl`, without replacement,
shuffled together, with `SEED` (never the run seed: every seed of the arm trains on the same mix).
The rows are copied byte for byte, so each keeps its own `name`, which EI keys its found proofs
by; the names are checked unique.

Then the preflight the GRPO extra targets get in line.sh (`grpo_v2/preflight.py`, here on the L2
pool and the mix): prompt, renaming-key and renaming / curried-class overlap with the L2 pool and
with every evaluation set, textbook72 and the sealed 13-20 / 21-30 sets included. The 3-premise
rows are shorter than the L3 band by construction, so only the band line may fail, and all 22
evaluation-set lines must be `ok` (a set that is not built prints a skip line, which fails).
Writes, next to MIX:

    MIX                          the 5,000 rows
    MIX_STEM.json                counts, sources and their md5s, the seed, the mix's md5
    MIX_STEM.preflight.log       the preflight's output
    MIX_STEM.preflight_pass.txt  only if it passed: `PREFLIGHT PASSED <md5 of MIX>` first

line.sh refuses ARM_k3p unless the pass record names the md5 of the mix as it is. Exits 1 if the
preflight fails (the mix and the log stay, for a look; no pass record). Written 2026-10-16.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import random
import re
import sys
from pathlib import Path

from . import preflight

R = preflight.R
POOL_L2 = R / "pools/curriculum/L2/pool.jsonl"
POOL_L3 = R / "pools/curriculum/L3/pool.jsonl"
L3_3P = R / "pools/curriculum_3p/L3_3p.jsonl"
MIX = R / "pools/curriculum_3p/L3_mix_ei.jsonl"
N_POOL, N_3P = 3333, 1667  # 5,000 targets, a third of them 3-premise rows (XFRAC 0.33)
SEED = 20261016
N_OK = 22  # the preflight's evaluation-set lines: 11 sets x (L2, the mix)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def lines(p: Path) -> list[str]:
    return [x for x in p.read_text().split("\n") if x.strip()]


def build(pool: list[str], extra: list[str], n_pool: int, n_3p: int, seed: int) -> list[str]:
    """`n_pool` of `pool` and `n_3p` of `extra` (jsonl lines), without replacement, shuffled."""
    rng = random.Random(seed)
    out = rng.sample(pool, n_pool) + rng.sample(extra, n_3p)
    rng.shuffle(out)
    return out


def passed(log: str) -> bool:
    """line.sh's extra-targets rule: no BAD, MISSING or skip line but the (indented) band lines,
    and >= N_OK `ok` lines."""
    ls = log.split("\n")
    bad = [x for x in ls if re.match(r"^(BAD|MISSING|\(skip)", x)]
    return not bad and sum(x.startswith("ok ") for x in ls) >= N_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=MIX)
    a = ap.parse_args(argv)
    pool, extra = lines(POOL_L3), lines(L3_3P)
    rows = build(pool, extra, N_POOL, N_3P, SEED)
    recs = [json.loads(x) for x in rows]
    names = [r["name"] for r in recs]
    assert all(names) and len(set(names)) == len(names), "target names must be unique"
    assert all(r.get("prompt") for r in recs)
    stem = a.out.with_suffix("")
    pp = Path(f"{stem}.preflight_pass.txt")
    pp.unlink(missing_ok=True)  # first: a pass record is only ever of the mix written below
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(x + "\n" for x in rows))
    n3 = sum(r.get("source") == "uncurry3_L3" for r in recs)
    picked = {r["name"] for r in recs}
    both = sum(r.get("orig_name") in picked for r in recs if r.get("orig_name"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = preflight.main([str(POOL_L2), str(a.out)])
    log = buf.getvalue()
    Path(f"{stem}.preflight.log").write_text(log)
    ok = passed(log)
    srcs = ((POOL_L3, len(pool)), (L3_3P, len(extra)))
    stats = {
        "mix": str(a.out),
        "md5": md5(a.out),
        "rows": len(rows),
        "pool_rows": len(rows) - n3,
        "rows_3p": n3,
        "rows_3p_with_their_original_in_the_mix": both,
        "seed": SEED,
        "sources": {str(p): {"rows": n, "md5": md5(p)} for p, n in srcs},
        "preflight": {"l2": str(POOL_L2), "exit": rc, "passed": ok},
    }
    Path(f"{stem}.json").write_text(json.dumps(stats, indent=1) + "\n")
    if ok:
        pp.write_text(
            f"PREFLIGHT PASSED {stats['md5']}\n"
            f"{a.out.name}: preflight.py {POOL_L2} vs the mix; only the band line failed, "
            f"{N_OK} evaluation-set lines ok (log: {Path(f'{stem}.preflight.log').name})\n"
        )
    print(json.dumps(stats, indent=1))
    print("mix preflight", "PASSED" if ok else "FAILED", f"(see {stem}.preflight.log)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
