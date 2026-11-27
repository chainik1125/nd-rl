# ruff: noqa: E501
#!/usr/bin/env python3
"""Capacity readout from training logs only (PREREG "Capacity and bits per trained weight").

For every final-source cell with n = 64 under <runs>/*/dpo/train/cells/: EDL_tok at t = 64 (SPEC: the lr with the lowest
dev loss at step 64 among the lrs that reached step 64, recomputed here from runs.jsonl, because the trainer's
`edl_fixed_step` also dropped lrs that were early-stopped AFTER step 64, e.g. M16k lr 0.3 at 512) and at the dev-selected step (`edl_selected`), raw and difficulty-matched
(`mini_train.edl_corrected`, using the step-0 nats of the same first m training examples from `step0-train`).
One-pass C(P) = EDL_64(P) / EDL_64(A16); extended C for the rungs trained to 512 = EDL_sel(P) / EDL_64(A16).
Bits per trained weight = EDL / P (A16-referenced, n = 64; not comparable with BTC thresholds).

  capacity.py --runs DIR [DIR ...] --out PREFIX
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from mini_train import edl_corrected

LN2 = math.log(2)


def t64_from_runs(runs_jsonl: Path, s0j: dict, P: int, t: int = 64):
    """EDL_tok at step t for the lr with the lowest dev loss at t among lrs whose run reached t (early stop None or >= t).
    Uses each lr's last record (an extended run carries the whole history). Returns (raw block, corrected block)."""
    last = {}
    for ln in runs_jsonl.read_text().splitlines():
        r = json.loads(ln)
        last[r["lr"]] = r
    cand = {}
    for lr, r in last.items():
        es = r.get("early_stopped_at")
        if str(t) in r["dev"] and (es is None or es >= t):
            on = r["online"]
            m = min(t, len(on["pids"]))
            nats = float(sum(on["think_nats"][:m]) + sum(on["answer_nats"][:m]))
            toks = int(sum(on["n_tok"][:m]))
            L = float(r["dev"][str(t)]["loss_total"])
            n0 = float(
                sum(
                    a + b
                    for a, b in zip(s0j["think_nats"][:m], s0j["answer_nats"][:m], strict=True)
                )
            )
            delta = n0 / sum(s0j["n_tok"][:m]) - s0j["L0_dev_total"]
            raw = nats - toks * L
            cand[lr] = (L, raw, raw - toks * delta, delta)
    if not cand:
        return {}, {}
    lr = min(cand, key=lambda k: cand[k][0])
    L, raw, cor, delta = cand[lr]
    return (
        {
            "lr": lr,
            "dev_nats_per_tok_total": L,
            "edl_tok_bits": raw / LN2,
            "per_lr_dev": {str(k): v[0] for k, v in cand.items()},
        },
        {"edl_tok_corr_bits": cor / LN2, "delta_nats_per_tok": delta},
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cells, s0 = {}, {}
    for r in a.runs:
        for cj in sorted((r / "dpo/train/cells").glob("*/cell.json")):
            c = json.loads(cj.read_text())
            if (cj.parent / "DONE").exists() and c.get("mdl_nats") is not None:
                cells[c["cell"]] = (cj, c)
        for f in (r / "dpo/train/step0_train").glob("*__n64.json"):
            s0[f.name] = f
    rows = []
    for cid, (cj, c) in sorted(cells.items()):
        if c.get("n") != 64 or "smoke" in cid:
            continue
        pool = c["pool"].replace("/", "_") if isinstance(c.get("pool"), str) else None
        src = c.get("source")
        key = f"dpo__dpo_{src}__n64.json"
        if key not in s0:
            continue
        s0j = json.loads(s0[key].read_text())
        corr = edl_corrected(json.loads(cj.read_text()), s0j)
        sel, csel = c.get("edl_selected") or {}, corr.get("edl_selected") or {}
        fx, cfx = t64_from_runs(cj.parent / "runs.jsonl", s0j, c["P"])
        rows.append(
            {
                "cell": cid,
                "source": src,
                "family": c["family"],
                "P": c["P"],
                "pool": pool,
                "t64_lr": fx.get("lr"),
                "t64_dev": fx.get("dev_nats_per_tok_total"),
                "t64_edl_bits_raw": fx.get("edl_tok_bits"),
                "t64_edl_bits_corr": cfx.get("edl_tok_corr_bits"),
                "sel_lr": c["selected"].get("lr"),
                "sel_step": c["selected"].get("step"),
                "sel_dev": sel.get("dev_nats_per_tok_total"),
                "sel_edl_bits_raw": sel.get("edl_tok_bits"),
                "sel_edl_bits_corr": csel.get("edl_tok_corr_bits"),
                "delta_nats_per_tok": cfx.get("delta_nats_per_tok"),
                "step0_dev": c.get("step0_dev_nats_per_tok_total"),
            }
        )
    ref = {r["source"]: r for r in rows if r["cell"] == f"dpo__{r['source']}__A16__n64"}
    for r in rows:
        a16 = ref.get(r["source"])
        for k in ("raw", "corr"):
            base = a16 and a16[f"t64_edl_bits_{k}"]
            r[f"C_onepass_{k}"] = (
                round(r[f"t64_edl_bits_{k}"] / base, 3)
                if base and r[f"t64_edl_bits_{k}"] is not None
                else None
            )
            r[f"C_ext_{k}"] = (
                round(r[f"sel_edl_bits_{k}"] / base, 3)
                if base and r[f"sel_edl_bits_{k}"] is not None
                else None
            )
            v = r[f"t64_edl_bits_{k}"]
            r[f"t64_bits_per_weight_{k}"] = v / r["P"] if v is not None else None
        print(
            f"{r['cell']:32s} P={r['P']:>11,} t64 lr={r['t64_lr']} dev={r['t64_dev'] and round(r['t64_dev'], 4)} "
            f"EDL64 raw={r['t64_edl_bits_raw'] and round(r['t64_edl_bits_raw'])} corr={r['t64_edl_bits_corr'] and round(r['t64_edl_bits_corr'])} "
            f"| sel lr={r['sel_lr']} step={r['sel_step']} corr={r['sel_edl_bits_corr'] and round(r['sel_edl_bits_corr'])} "
            f"| C1 raw={r['C_onepass_raw']} corr={r['C_onepass_corr']} Cext corr={r['C_ext_corr']}"
        )
    a.out.with_suffix(".json").write_text(json.dumps(rows, indent=1) + "\n")
    with open(a.out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
