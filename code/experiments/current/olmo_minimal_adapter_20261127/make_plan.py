# ruff: noqa: E501
"""Stage-1 plan for `dpo` (design discourse Turn 3 cell table), written to <out> with a projected H100 time.

    make_plan.py [--ckpt dpo] [--out /scratch/cowifornia/nd-rl/olmo_elicit/v2/plan_dpo.json]

Cell ids: <ckpt>__<source>__<fam>__n<n>; pool = "<ckpt>/<source>" under --root. Order: the cells needed first for sampling,
then the training-only capacity rungs, then conditional cells (flag "conditional": true; mini_train skips them unless --conditional or
--only) and the confirmation template ("template": true, family null: never run, fill the family and seed first).
Schedule: "steps" = cap; "extend": {"after", "to", "eval_steps"} = all lrs to `after`, the lr with the lowest dev loss at `after`
continues to `to`. Early stop and edge rule are in mini_train.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mini_train as mt  # noqa: E402

OUT = Path("/scratch/cowifornia/nd-rl/olmo_elicit/v2/plan_dpo.json")
A16 = ("lora:r=16:targets=all", [1e-4, 3e-4, 1e-3])
A1 = ("lora:r=1:targets=all", [3e-4, 1e-3, 3e-3])
O1 = ("lora:r=1:targets=o_proj", [1e-3, 3e-3, 1e-2])
M_LRS = [3e-2, 1e-1, 3e-1]
ATTN = "q_proj,k_proj,v_proj,o_proj"
EXT = {"after": 128, "to": 512, "eval_steps": [256, 512]}
EV128 = [16, 32, 64, 128]
RATES = {
    "own": (2.3, 20.0, 0.65),
    "unsel": (3.0, 28.0, 0.8),
    "final": (3.6, 34.0, 1.0),
}  # s/step, s/dev eval, s/new example
PRICE = 4.39


def cid(ckpt: str, src: str, fam: str, n: int) -> str:
    return f"{ckpt}__{src}__{fam}__n{n}"


def cell(ckpt, src, fam, spec, n, **kw) -> dict:
    return {"cell": cid(ckpt, src, fam, n), "pool": f"{ckpt}/{src}", "family": spec, "n": n, **kw}


def make_plan(ckpt: str = "dpo") -> list[dict]:
    c = []
    # sampling-critical cells first
    for src in ("own", "unsel", "final"):
        c.append(
            cell(
                ckpt,
                src,
                "A16",
                A16[0],
                64,
                lrs=A16[1],
                steps=128,
                eval_steps=EV128,
                edge_extend=True,
            )
        )
    for fam, (spec, lrs) in (("O1", O1), ("A1", A1)):
        c.append(
            cell(
                ckpt,
                "own",
                fam,
                spec,
                64,
                lrs=lrs,
                steps=512,
                eval_steps=EV128,
                extend=EXT,
                edge_extend=True,
            )
        )
    for fam, spec in (
        ("M4k_all", "mask:k=4096:seed=0"),
        ("M4k_attn", f"mask:k=4096:seed=0:targets={ATTN}"),
    ):
        c.append(
            cell(
                ckpt,
                "own",
                fam,
                spec,
                64,
                lrs=M_LRS,
                steps=512,
                eval_steps=EV128,
                extend=EXT,
                edge_extend=True,
            )
        )
    for n in (4, 1):
        c.append(cell(ckpt, "own", "A16", A16[0], n, lrs=A16[1], steps=64, edge_extend=True))
    # training-only capacity rungs, final source, t = 64
    for fam, k in (("M65k", 65536), ("M16k", 16384), ("M4k", 4096), ("M256", 256)):
        c.append(
            cell(
                ckpt,
                "final",
                fam,
                f"mask:k={k}:seed=0",
                64,
                lrs=M_LRS,
                steps=64,
                eval_steps=[16, 32, 64],
                edge_extend=True,
            )
        )
    # conditional cells, last
    c.append(
        cell(
            ckpt,
            "own",
            "M256_all",
            "mask:k=256:seed=0",
            64,
            lrs=M_LRS,
            steps=512,
            eval_steps=EV128,
            extend=EXT,
            edge_extend=True,
            conditional=True,
        )
    )
    c.append(
        cell(
            ckpt,
            "own",
            "A16",
            A16[0],
            16,
            lr_from=cid(ckpt, "own", "A16", 64),
            steps=64,
            edge_extend=False,
            conditional=True,
        )
    )
    c.append(
        {
            "cell": f"{ckpt}__own__CONFIRM__n64",
            "pool": f"{ckpt}/own",
            "family": None,
            "n": 64,
            "template": True,
            "conditional": True,
            "note": "confirmation at the claimed smallest rung: fill family (new mask seed in the spec), lrs, schedule; `seed` overrides the A-init seed",
            "seed": 20261128,
        }
    )
    return c


def project(cells: list[dict]) -> list[dict]:
    """Projected H100 seconds per cell: every lr to `after` (or steps), the winner to `to`; evals; online scores; one step-0 dev eval per pool. No early stops, no edge extensions."""
    out, pools = [], set()
    for c in cells:
        if c.get("template"):
            continue
        src = c["pool"].split("/")[1]
        s_step, s_dev, s_on = RATES[src]
        ext = c.get("extend")
        n = c["n"]
        steps = c.get("steps", 64)
        after = ext["after"] if ext else steps
        nl = len(c["lrs"]) if "lrs" in c else 1
        ev_a = {
            s for s in (c.get("eval_steps") or mt.default_eval_steps(n, after)) if s <= after
        } | {after}
        t = nl * (after * s_step + len(ev_a) * s_dev + min(n, after) * s_on)
        if ext:
            t += (ext["to"] - ext["after"]) * s_step + len(
                {s for s in ext["eval_steps"]} | {ext["to"]}
            ) * s_dev
        if c["pool"] not in pools:
            pools.add(c["pool"])
            t += s_dev
        out.append(
            {"cell": c["cell"], "conditional": bool(c.get("conditional")), "secs": t, "lrs": nl}
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", default="dpo")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    cells = make_plan(a.ckpt)
    a.out.write_text(json.dumps({"cells": cells}, indent=1) + "\n")
    print(f"{len(cells)} cells -> {a.out}")
    pr = project(cells)
    tot = {False: 0.0, True: 0.0}
    for p in pr:
        tot[p["conditional"]] += p["secs"]
        print(
            f"  {p['cell']:34s} {p['lrs']} lr  {p['secs'] / 60:6.1f} min  ${p['secs'] / 3600 * PRICE:5.2f}{'  (conditional)' if p['conditional'] else ''}"
        )
    print(
        f"projected: non-conditional {tot[False] / 3600:.2f} h (${tot[False] / 3600 * PRICE:.1f}); conditional {tot[True] / 3600:.2f} h (${tot[True] / 3600 * PRICE:.1f}) at ${PRICE}/h; no early stops or edge extensions"
    )


if __name__ == "__main__":
    main()
