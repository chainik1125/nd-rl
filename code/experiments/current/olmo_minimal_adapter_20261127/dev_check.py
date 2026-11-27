# ruff: noqa: E501
"""Own-source recipe check on held-out non-reach dev prompts (PREREG Amendment 5(b) tuned cell, Amendment 6(c), Amendment 8).

For each candidate adapter, sampled on the own pool's dev prompts (`<pool>/test.jsonl` pids): strict pass rate, and the
paired difference to the checkpoint's plain rate on the same prompts (Dan's stored arm-A draws: `dpo` F_dpo_ifb, 16 per
prompt; later checkpoints his arm-A rows). Two-level bootstrap (prompts, then draws within prompt, both arms on the same
prompt resample), 10,000 reps.

Eligibility (audit T3): a candidate with no draws is "not sampled" (e.g. it failed the gate); one without exactly --draws
draws on every dev prompt is "incomplete". Neither can win. The rule fixed in advance, over eligible candidates: the
highest strict dev pass rate wins (compared as integer pass counts, so an exact tie is exact); a tie goes to the
earlier-listed candidate (list them in the pre-registered order). The reach prompts play no part.

Labels (audit T6): with --stage (the staged batch dir) and --gate (runs/<ckpt>/gate/<batch>.json), each candidate also
carries did_not_train, the dev-loss trajectory, max|B| and the gate verdict. `did_not_train` means held-out own-sample
loss never fell below step 0; it does not mean the weights stayed put.

Resolution (audit T7): at `dpo`, 32 prompts x 4 draws, a no-effect candidate's paired interval is about +-8 points; "no
candidate above 0" reads as "no lift larger than about 15 points".

Exit 0 with a winner; exit 3 (JSON still written, winner null) when no candidate is eligible.

  dev_check.py --ckpt dpo --pool .../v2/pools/dpo/own --cells name=sample_dir ... --out OUT.json [--stage DIR --gate FILE]
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


def plain_cells(ckpt: str, pids: list[str]) -> dict[str, list[int]]:
    import analyze_arms as AA
    import dan_curve as dc

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
    """Strict 0/1 draws per dev prompt; empty lists when the cell was not sampled (no graded.jsonl)."""
    out: dict[str, list[int]] = {p: [] for p in pids}
    g = d / "graded.jsonl"
    if not g.exists():
        return out
    for ln in g.read_text().splitlines():
        if not ln.strip():
            continue
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


def labels(stage: Path | None, gate: dict, name: str) -> dict:
    """did_not_train, dev-loss trajectory, max|B| (cell.json b_stats) and the gate verdict, for the report (T6)."""
    out: dict = {"gate": gate.get(name)}
    cj_path = stage / name / "cell.json" if stage else None
    if cj_path and cj_path.exists():
        cj = json.loads(cj_path.read_text())
        sel = cj.get("selected") or {}
        traj = ((cj.get("lr_summary") or {}).get(str(sel.get("lr"))) or {}).get("dev_loss_by_step")
        out |= {
            "did_not_train": cj.get("did_not_train"),
            "did_not_train_means": "held-out own-sample loss never fell below step 0; the weights did move (max_abs_B)",
            "dev_loss_by_step": traj,
            "max_abs_B": (cj.get("b_stats") or {}).get("max_abs_B"),
            "lr": sel.get("lr"),
            "steps": sel.get("step"),
        }
    return out


def evaluate(
    cands: list[tuple[str, dict[str, list[int]]]],
    plain: dict[str, list[int]],
    pids: list[str],
    draws: int,
    reps: int,
    extra: dict[str, dict] | None = None,
) -> dict:
    """Per candidate: status ok / not sampled / incomplete, rates and intervals for ok ones; the winner among ok ones."""
    rows = []
    for name, c in cands:
        counts = {p: len(c[p]) for p in pids}
        row: dict = {"cell": name, "draws": sum(counts.values()), **(extra or {}).get(name, {})}
        if row["draws"] == 0:
            row["status"] = "not sampled"
        elif any(n != draws for n in counts.values()):
            row["status"] = "incomplete"
            row["prompts_with_wrong_count"] = {p: n for p, n in counts.items() if n != draws}
        else:
            lo, hi = boot(c, None, pids, reps, 1)
            dlo, dhi = boot(c, plain, pids, reps, 2)
            row |= {
                "status": "ok",
                "passes": sum(sum(c[p]) for p in pids),
                "dev_rate": rate(c, pids),
                "lo": lo,
                "hi": hi,
                "minus_plain": rate(c, pids) - rate(plain, pids),
                "minus_plain_lo": dlo,
                "minus_plain_hi": dhi,
                "above_plain": dlo > 0,
                "moves_dev": dlo > 0 or dhi < 0,
            }
        rows.append(row)
    ok = [i for i, r in enumerate(rows) if r["status"] == "ok"]
    best = max(ok, key=lambda i: (rows[i]["passes"], -i)) if ok else None
    return {
        "n_prompts": len(pids),
        "draws_per_prompt": draws,
        "plain_rate": rate(plain, pids),
        "plain_draws": sum(len(v) for v in plain.values()),
        "candidates": rows,
        "winner": rows[best]["cell"] if best is not None else None,
        "rule": "eligible = exactly draws_per_prompt draws on every dev prompt; highest strict pass count wins; exact tie to the earlier-listed candidate",
        "resolution_note": "at 32 prompts x 4 draws a no-effect candidate's paired interval is about +-8 points: 'none above 0' reads 'no lift larger than about 15 points'",
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument(
        "--cells", nargs="+", required=True, help="name=sample_dir, in the pre-registered order"
    )
    ap.add_argument("--draws", type=int, default=4, help="required draws per dev prompt")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--stage", type=Path, default=None, help="staged batch dir (cell.json labels)")
    ap.add_argument("--gate", type=Path, default=None, help="gate result json of the batch")
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
    gate = json.loads(a.gate.read_text()).get("verdicts", {}) if a.gate and a.gate.exists() else {}
    cands, extra = [], {}
    for spec in a.cells:
        name, d = spec.split("=", 1)
        cands.append((name, cell_draws(Path(d), pids)))
        extra[name] = labels(a.stage, gate, name)
    out = {"ckpt": a.ckpt, "pool": str(a.pool)} | evaluate(
        cands, plain, pids, a.draws, a.reps, extra
    )
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    for r in out["candidates"]:
        if r["status"] != "ok":
            print(f"{r['cell']:40s} {r['status']} ({r['draws']} draws); not eligible")
            continue
        print(
            f"{r['cell']:40s} dev {100 * r['dev_rate']:.1f} % [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]  minus plain {100 * r['minus_plain']:+.1f} [{100 * r['minus_plain_lo']:+.1f}, {100 * r['minus_plain_hi']:+.1f}]"
        )
    print(f"plain {100 * out['plain_rate']:.1f} %; winner: {out['winner']}")
    if out["winner"] is None:
        sys.exit(3)


if __name__ == "__main__":
    main()
