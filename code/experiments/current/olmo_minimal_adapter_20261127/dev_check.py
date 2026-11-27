# ruff: noqa: E501
"""Own-source recipe check on held-out non-reach dev prompts (PREREG Amendment 5(b) tuned cell, Amendment 6(c)).

For each candidate adapter, sampled on the own pool's dev prompts (`<pool>/test.jsonl` pids): strict pass rate, and the
paired difference to the checkpoint's plain rate on the same prompts (Dan's stored arm-A draws: `dpo` F_dpo_ifb, 16 per
prompt; later checkpoints his arm-A rows). Two-level bootstrap (prompts, then draws within prompt, both arms on the same
prompt resample), 10,000 reps. The rule fixed in advance: the candidate with the highest strict dev pass rate wins; an
exact tie goes to the earlier-listed candidate (list them in the pre-registered order). The reach prompts play no part.

  dev_check.py --ckpt dpo --pool .../v2/pools/dpo/own --cells name=sample_dir ... --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

OLD = Path(__file__).resolve().parent.parent / "olmo_elicitation_20261125"
sys.path.insert(0, str(OLD))
sys.dont_write_bytecode = True
import analyze_arms as AA  # noqa: E402
import dan_curve as dc  # noqa: E402


def plain_cells(ckpt: str, pids: list[str]) -> dict[str, list[int]]:
    if ckpt == "dpo":
        pairs = [
            (
                AA.D / "olmo_ifomega/samples/F_dpo_ifb.jsonl.gz",
                AA.D / "olmo_ifomega/graded/F_dpo_ifb.jsonl",
            )
        ]
        return {p: [s for s, _ in v][:16] for p, v in AA.h_cells(pairs, pids).items()}
    R = dc.Rows(AA.D / "olmo_ifcurve" / "rows")
    return {p: [int(bool(x["strict"])) for x in R.draws(ckpt, p, "A")] for p in pids}


def cell_draws(d: Path, pids: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {p: [] for p in pids}
    for ln in (d / "graded.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r["pid"] in out:
            out[r["pid"]].append(int(r["strict"] is True or str(r["strict"]) == "True"))
    return out


def boot(
    a: dict[str, list[int]], b: dict[str, list[int]] | None, pids: list[str], reps: int, seed: int
):
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        idx = [rng.choice(pids) for _ in pids]
        sa = sum(
            sum(rng.random() < (sum(a[p]) / len(a[p])) for _ in a[p]) / len(a[p]) for p in idx
        ) / len(idx)
        if b is None:
            vals.append(sa)
        else:
            sb = sum(
                sum(rng.random() < (sum(b[p]) / len(b[p])) for _ in b[p]) / len(b[p]) for p in idx
            ) / len(idx)
            vals.append(sa - sb)
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]


def rate(c: dict[str, list[int]], pids: list[str]) -> float:
    return sum(sum(c[p]) / len(c[p]) for p in pids) / len(pids)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument(
        "--cells", nargs="+", required=True, help="name=sample_dir, in the pre-registered order"
    )
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pids = [
        json.loads(ln)["pid"]
        for ln in (a.pool / "test.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    plain = plain_cells(a.ckpt, pids)
    miss = [p for p in pids if not plain.get(p)]
    if miss:
        raise SystemExit(f"no plain draws for {miss[:5]}")
    rows = []
    for spec in a.cells:
        name, d = spec.split("=", 1)
        c = cell_draws(Path(d), pids)
        empty = [p for p in pids if not c[p]]
        if empty:
            raise SystemExit(f"{name}: no draws for {empty[:5]}")
        lo, hi = boot(c, None, pids, a.reps, 1)
        dlo, dhi = boot(c, plain, pids, a.reps, 2)
        rows.append(
            {
                "cell": name,
                "draws": sum(len(v) for v in c.values()),
                "dev_rate": rate(c, pids),
                "lo": lo,
                "hi": hi,
                "minus_plain": rate(c, pids) - rate(plain, pids),
                "minus_plain_lo": dlo,
                "minus_plain_hi": dhi,
                "moves_dev": dlo > 0 or dhi < 0,
            }
        )
    best = max(range(len(rows)), key=lambda i: (rows[i]["dev_rate"], -i))
    out = {
        "ckpt": a.ckpt,
        "pool": str(a.pool),
        "n_prompts": len(pids),
        "plain_rate": rate(plain, pids),
        "plain_draws": sum(len(v) for v in plain.values()),
        "candidates": rows,
        "winner": rows[best]["cell"],
        "rule": "highest strict dev pass rate; exact tie to the earlier-listed candidate",
    }
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        print(
            f"{r['cell']:40s} dev {100 * r['dev_rate']:.1f} % [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]  minus plain {100 * r['minus_plain']:+.1f} [{100 * r['minus_plain_lo']:+.1f}, {100 * r['minus_plain_hi']:+.1f}]"
        )
    print(f"plain {100 * out['plain_rate']:.1f} %; winner: {out['winner']}")


if __name__ == "__main__":
    main()
