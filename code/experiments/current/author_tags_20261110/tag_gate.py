"""The pre-registered manipulation check (PREREG.md), per base, from tag_check.py's reads.

    python tag_gate.py [--dir AT/reads/check] [--json]

Reads every `<base>.json` in the folder (tag_launch.py read / check write them). Per base:
  open A / B / none  mean natural log P of the route-opening actions (label `open`) of `sigma_unseen` under each tag
  d_open             open A - open B
  d_ctrl             A - B of the mean log P over all actions of `ctrl_unseen`
  d_open - d_ctrl    passes when >= log 2 (the opening at least twice as likely under A, beyond any general shift)
The three default bases give +0.309, +0.090 and +0.078 (GATE_RESULT.md).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_core as T  # noqa: E402

BAR = math.log(2)


def gate_row(res: dict[str, Any]) -> dict[str, Any]:
    """One base's gate numbers from a tag_check.py result."""
    sig, ctrl = res["sets"]["sigma_unseen"], res["sets"]["ctrl_unseen"]
    op = {t: sig[t]["open"]["mean_logp_action"] for t in sig}
    d_open = op["A"] - op["B"]
    d_ctrl = ctrl["A"]["all"]["mean_logp_action"] - ctrl["B"]["all"]["mean_logp_action"]
    diff = d_open - d_ctrl
    return {
        "open": op,
        "open_actions": sig["A"]["open"]["actions"],
        "d_open": d_open,
        "d_ctrl": d_ctrl,
        "d_open_minus_d_ctrl": diff,
        "ratio_A_over_B": math.exp(diff),
        "passes": diff >= BAR,
        "variant": res.get("variant"),
    }


def _order(p: Path) -> tuple[int, float, str, int]:
    """Default bases first, then the variants by share; seeds in order."""
    try:
        v, s = T.parse_base_name(p.stem)
    except ValueError:
        return (2, 0, p.stem, 0)
    return (0 if v is None else 1, T.VARIANTS.get(v or "", 0.0), p.stem if v is None else v, s)


def rows(folder: Path) -> dict[str, dict[str, Any]]:
    return {
        p.stem: gate_row(json.loads(p.read_text()))
        for p in sorted(folder.glob("*.json"), key=_order)
    }


def main(argv: list[str] | None = None) -> dict[str, dict[str, Any]]:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dir", default=str(T.AT / "reads" / "check"))
    ap.add_argument("--json", action="store_true", help="print the rows as JSON")
    a = ap.parse_args(argv)
    R = rows(Path(a.dir))
    if a.json:
        print(json.dumps(R, indent=1))
        return R
    print(
        f"manipulation check, {a.dir}: route opening on sigma_unseen; pass when d_open - d_ctrl >= log 2 = {BAR:.3f}"
    )
    print(
        f"{'base':16s} {'open A':>8s} {'open B':>8s} {'none':>8s} {'d_open':>7s} {'d_ctrl':>7s} {'diff':>7s} {'xA/B':>5s}  gate"
    )
    for name, r in R.items():
        o = r["open"]
        none = f"{o['none']:8.3f}" if "none" in o else f"{'-':>8s}"
        print(f"{name:16s} {o['A']:8.3f} {o['B']:8.3f} {none} {r['d_open']:+7.3f} {r['d_ctrl']:+7.3f} "
              f"{r['d_open_minus_d_ctrl']:+7.3f} {r['ratio_A_over_B']:5.2f}  {'PASS' if r['passes'] else 'fail'}")  # fmt: skip
    if not R:
        print("(no reads)")
    return R


if __name__ == "__main__":
    main()
