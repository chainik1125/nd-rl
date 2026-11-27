# ruff: noqa: E501
"""Release rule and gamma choice (Turn 5 B.2 steps 7-8). Scripts print the verdict and exit with a code; the orchestrator
starts the next job by hand.

  python3 rules.py release --a1 <A1 graded.jsonl> --refs refs.json --h pids_H.txt --r pids_R.txt --n-h 2 --n-r 4 [--out f.json]
      exit 0  RUN A2 then A1u  (A1 success, partial or collapse)
      exit 10 SKIP A1u and A2  (clean null: f < 0.25 on both H and R, and not collapsed; A1u is then inferred, not measured)
  python3 rules.py gamma --g46 <dev graded> --g35 <dev graded> --d pids_D.txt --n-draws 2 --out expo_choice.json
      choose 4.6 only if its D pass@1 leads 3.5 by more than one paired prompt-bootstrap SE and 4.6 is not collapsed; else 3.5.
      exit 0 chose 3.5, exit 1 chose 4.6 (so a shell can branch); both write the JSON with the timestamp.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import collapse_stats, prompt_rates, read_jsonl, read_pids  # noqa: E402
from stats import pass_diff, recovered_fraction  # noqa: E402

CLEAN_NULL_F = 0.25


def release(a1_rows, refs, h_pids, r_pids, n_h: int, n_r: int, reps: int = 2000) -> dict:
    planned = n_h * len(h_pids) + n_r * len(r_pids)
    out = {"planned": planned}
    cs = collapse_stats(a1_rows, planned)
    out["collapse"] = cs
    for tag, pids, n in (("H", h_pids, n_h), ("R", r_pids, n_r)):
        cell = prompt_rates(a1_rows, pids, n)
        fr = recovered_fraction(cell, refs["dpo"][tag], refs["s2300"][tag], reps, seed=1)
        out[f"f_{tag}"] = fr
    fh, fr_ = out["f_H"].get("f"), out["f_R"].get("f")
    clean_null = (
        (not cs["collapsed"])
        and fh is not None
        and fr_ is not None
        and fh < CLEAN_NULL_F
        and fr_ < CLEAN_NULL_F
    )
    out["clean_null"] = bool(clean_null)
    out["decision"] = (
        "SKIP A1u and A2 (clean null; A1u inferred, not measured)"
        if clean_null
        else "RUN A2 then A1u"
    )
    return out


def gamma_choice(rows46, rows35, d_pids, n_draws: int, reps: int = 2000) -> dict:
    planned46 = planned35 = n_draws * len(d_pids)
    r46, r35 = prompt_rates(rows46, d_pids, n_draws), prompt_rates(rows35, d_pids, n_draws)
    diff = pass_diff(r46, r35, reps, seed=2)
    c46, c35 = collapse_stats(rows46, planned46), collapse_stats(rows35, planned35)
    lead = diff.get("diff", float("nan"))
    se = diff.get("se", float("nan"))
    pick46 = bool(lead > se and not c46["collapsed"]) if lead == lead and se == se else False
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "p46": diff.get("mean_a"),
        "p35": diff.get("mean_b"),
        "lead_46_minus_35": lead,
        "paired_se": se,
        "ci": [diff.get("lo"), diff.get("hi")],
        "collapse46": c46,
        "collapse35": c35,
        "rule": "4.6 only if lead > 1 paired prompt-bootstrap SE and 4.6 not collapsed; else 3.5",
        "chosen_gamma": 4.6 if pick46 else 3.5,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("release")
    r.add_argument("--a1", required=True)
    r.add_argument("--refs", required=True)
    r.add_argument("--h", required=True)
    r.add_argument("--r", required=True)
    r.add_argument("--n-h", type=int, default=2)
    r.add_argument("--n-r", type=int, default=4)
    r.add_argument("--out")
    g = sub.add_parser("gamma")
    g.add_argument("--g46", required=True)
    g.add_argument("--g35", required=True)
    g.add_argument("--d", required=True)
    g.add_argument("--n-draws", type=int, default=2)
    g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "release":
        res = release(
            read_jsonl(a.a1), json.load(open(a.refs)), read_pids(a.h), read_pids(a.r), a.n_h, a.n_r
        )
        print(json.dumps(res, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"RELEASE: {res['decision']}")
        return 10 if res["clean_null"] else 0
    res = gamma_choice(read_jsonl(a.g46), read_jsonl(a.g35), read_pids(a.d), a.n_draws)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"GAMMA CHOICE: {res['chosen_gamma']}")
    return 1 if res["chosen_gamma"] == 4.6 else 0


if __name__ == "__main__":
    sys.exit(main())
