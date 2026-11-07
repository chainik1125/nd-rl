"""Amendment 3 (post-hoc): the matched-compute checkpoints. Confirm each loads (CPU), record md5, x and the guard's state.

    python make_matched.py --ckpt_dir $FR/matched/ckpts --out ckpts_matched.json   # serrano, after copy_matched.sh

x comes from numbers.json's curve where the line has a read at that iteration. dd01_s1 it 35 and dd01_s2 it 25 have no read, so
their x is the same sum over the line's merged metrics rows that numbers.json uses (`merge.line_view`). Every x is recomputed that
way and checked against the curve, so the pilot's x is shown to come from the pilot's own metrics rows.
"""

# ruff: noqa: E501, SIM115, B905, B007, SIM300
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from factor_lib import KD

KW = Path(__file__).resolve().parent.parent / "knockout_danbase_20261104"
NUMBERS = Path(os.environ.get("NUMBERS", str(KW / "numbers.json")))
R = "/scratch/cowifornia/nd-rl/knockout_danbase/runs"
EARLY = "earliest available, x well below match"
# tag: line, base, the 1 % end it is set against, host and path of the original, group, label
MATCHED = {
    "m_dko_s0_it60": (
        "dko_s0",
        "ko_s0",
        "dd1_s0",
        "serrano",
        f"{R}/dko_s0/seg3/ckpts/dko_s0_it60_snap.pt",
        "matched",
        "",
    ),
    "m_dko_s1_it70": (
        "dko_s1",
        "ko_s1",
        "dd1_s1",
        "temescal",
        f"{R}/dko_s1/seg3/ckpts/dko_s1_it70_snap.pt",
        "matched",
        "",
    ),
    "m_dko_s2_it100": (
        "dko_s2",
        "ko_s2",
        "dd1_s2",
        "ynez",
        f"{R}/dko_s2/seg2/ckpts/dko_s2_it100_snap.pt",
        "matched",
        "",
    ),
    "m_dd01_s0_it60": (
        "dd01_s0",
        "dose01_s0",
        "dd1_s0",
        "serrano",
        f"{R}/dd01_s0/seg1/ckpts/dd01_s0_it60_snap.pt",
        "matched",
        "",
    ),
    "m_dd01g0_s2_it90": (
        "dd01g0_s2",
        "dose01_s2",
        "dd1_s2",
        "ynez",
        f"{R}/dd01g0_s2/seg1/ckpts/dd01g0_s2_it90_snap.pt",
        "pilot",
        "guard off, one run",
    ),
    "e_dd01_s1_it35": (
        "dd01_s1",
        "dose01_s1",
        "dd1_s1",
        "ynez",
        f"{R}/dd01_s1/seg3/ckpts/dd01_s1_it35_rolling.pt",
        "earliest",
        EARLY,
    ),
    "e_dd01_s2_it25": (
        "dd01_s2",
        "dose01_s2",
        "dd1_s2",
        "ynez",
        f"{R}/dd01_s2/seg3/ckpts/dd01_s2_it25_rolling.pt",
        "earliest",
        EARLY,
    ),
}
DOSE = {"ko": "0%", "dose01": "0.1%", "dose1": "1%"}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def x_by_iter(line: str, state: dict) -> dict[int, float]:
    """The cumulative x_dan per iteration from the line's own merged metrics rows."""
    sys.path.insert(0, str(KW))
    import common as C  # type: ignore[import-not-found]
    import guard as G  # type: ignore[import-not-found]

    D = C.dan_ladder(C.LINES[line]["ladder"])["D"]
    x, out = 0.0, {}
    for r in G.merged_rows(line, state[line]["segs"]):
        x += C.x_dan_row(r, D)
        out[r["iter"]] = x
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from model import load_ckpt  # type: ignore[import-not-found]

    num = json.load(open(NUMBERS))["lines"]
    state = json.load(open(KD / "sup" / "state.json"))
    out = []
    for tag, (line, base, target, host, src, group, label) in MATCHED.items():
        p = Path(a.ckpt_dir) / Path(src).name
        d, s = base.split("_s")
        it = int(p.name.split("_it")[1].split("_")[0])
        L = num[line]
        cur = {c["it"]: c for c in L["curve"]}
        xs = x_by_iter(line, state)
        x_rows = xs[it]
        if it in cur:
            assert abs(cur[it]["x_dan"] - x_rows) < 1e-9, (tag, cur[it]["x_dan"], x_rows)
        x_t = num[target]
        t_it = 90 if target == "dd1_s2" else 60
        x_target = {c["it"]: c for c in x_t["curve"]}[t_it]["x_dan"]
        rb = [r for r in L["guard"]["rollbacks"] if r["it"] <= it]
        scale = rb[-1]["lr_scale"] if rb else 1.0
        x_cut = rb[0]["x_dan"] if rb else None
        c = {"tag": tag, "role": "end", "line": line, "base": base, "dose": DOSE[d], "seed": int(s), "group": group, "label": label, "target": target,
             "x_target": x_target, "path": str(p), "source_host": host, "source_path": src, "file": p.name, "it": it,
             "kind": "alarm" if "_alarm" in p.name else ("snap" if "_snap" in p.name else "rolling"), "x_dan": cur[it]["x_dan"] if it in cur else x_rows,
             "x_in_curve": it in cur, "x_from_metrics_rows": x_rows, "x_exact": True, "x_read_it": it, "last_iter": L["last_iter"], "x_dan_last": L["x_dan"],
             "secondary": True, "pilot": group == "pilot", "guard_off": line == "dd01g0_s2", "lr_scale": scale, "rollbacks": rb,
             "share_of_x_at_reduced_lr": (x_rows - x_cut) / x_rows if x_cut is not None else 0.0,
             "own_read": {k: cur[it][k] for k in ("first32", "solved64", "e32")} if it in cur else None}  # fmt: skip
        c["md5"] = md5(p)
        model, tok, extra = load_ckpt(str(p), "cpu")
        c.update(n_params=model.n_params(), vocab=tok.vocab_size, tok_mode=tok.mode, canon=bool(getattr(tok, "canon", False)),
                 grpo_iter=extra.get("grpo_iter"), grpo_init=extra.get("grpo_init"), grpo_rollback=extra.get("grpo_rollback"))  # fmt: skip
        del model
        assert c["grpo_iter"] == it, (tag, c["grpo_iter"])
        assert Path(c["grpo_init"]).name == f"stage1_{base}.pt", (tag, c["grpo_init"])
        c["noise_seed"] = c["seed"] * 1000 + it + 500  # the frozen rule
        out.append(c)
        print(tag, c["kind"], "it", it, "x", round(c["x_dan"], 4), "target", round(x_target, 4), "curve" if it in cur else "rows", "lr x", scale,
              "reduced-lr share", round(c["share_of_x_at_reduced_lr"], 3), c["md5"], c["n_params"], c["tok_mode"], "init", Path(c["grpo_init"]).name, "rollback", c["grpo_rollback"], flush=True)  # fmt: skip
    json.dump(out, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
