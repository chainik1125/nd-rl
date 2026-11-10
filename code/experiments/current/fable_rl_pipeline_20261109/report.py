#!/usr/bin/env python3
"""Tables from RL runs' `rounds.jsonl` files (training progress and dev results by length).

    python report.py RUN_DIR [RUN_DIR ...]            # progress per round, dev pass@1 per check
    python report.py --dev ROUND RUN_DIR [...]        # the dev table of one evaluated round
    python report.py --best RUN_DIR [...]             # the round the checkpoint rule picks
"""
import argparse
import json
import os


def load(d):
    fn = os.path.join(d, "rounds.jsonl")
    rows = [json.loads(l) for l in open(fn) if l.strip()]
    seen = {}
    for r in rows:
        seen[(r["round"], "final_dev" in r)] = r
    return list(seen.values())


def dev_of(r):
    return r.get("dev") or r.get("final_dev")


def pick(rows):
    """The checkpoint rule (PREREG amendment 2): highest dev pass@1 in plain mode, ties to the later round."""
    best = None
    for r in rows:
        d = dev_of(r)
        if not d or "plain" not in d:
            continue
        rnd = r["round"] + 1 if "dev" in r else r["round"]
        key = (d["plain"]["all"]["pass@1"], rnd)
        if best is None or key > best[0]:
            best = (key, rnd, d)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dev", type=int, default=None)
    ap.add_argument("--best", action="store_true")
    a = ap.parse_args()
    for d in a.runs:
        rows = load(d)
        print(f"## {d}")
        if a.best:
            b = pick(rows)
            if b:
                print(f"picked round {b[1]} (checkpoint r{b[1]}.pt): plain pass@1 {b[0][0]:.2f}")
                a.dev = b[1]
        if a.dev is not None:
            for r in rows:
                dv = dev_of(r)
                rnd = r["round"] + 1 if "dev" in r else r["round"]
                if dv and rnd == a.dev:
                    for mode, tb in dv.items():
                        print(f"\nround {rnd}, mode {mode}")
                        print("group | n | pass@1 | pass@5 | complete% | tok/attempt")
                        print("---|---|---|---|---|---")
                        for g, x in tb.items():
                            print(f"{g} | {x['n']} | {x['pass@1']:.1f} | {x.get('pass@5', float('nan')):.1f} "
                                  f"| {x['complete%']:.1f} | {x['tok/attempt']:.0f}")  # fmt: skip
            continue
        print("round | attempts | solved now | solved total | new | shorter | her | neg | mean len "
              "| roll s | fit s | dev plain p@1/p@5 | dev search p@1/p@5 | search tok")  # fmt: skip
        print("---|" * 14)
        for r in rows:
            if "attempts" not in r:
                continue
            dv = r.get("dev") or {}
            cell = []
            for mode in ("plain", "search"):
                if mode in dv:
                    x = dv[mode]["all"]
                    cell.append(f"{x['pass@1']:.1f}/{x.get('pass@5', float('nan')):.1f}")
                else:
                    cell.append("")
            tok = f"{dv['search']['all']['tok/attempt']:.0f}" if "search" in dv else ""
            print(f"{r['round'] + 1} | {r['attempts']} | {r['solved_this_round']} | {r['solved_total']} "
                  f"| {r['new']} | {r['shorter']} | {r['her']} | {r['neg']} | {r['mean_best_len']:.2f} "
                  f"| {r['secs']['roll']} | {r['secs']['fit']} | {cell[0]} | {cell[1]} | {tok}")  # fmt: skip
        for r in rows:
            if "attempts" in r:
                last = r
        print("solved by length:", last["solved_by_len"])


if __name__ == "__main__":
    main()
