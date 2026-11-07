"""The 19 checkpoints: confirm each loads (CPU), record md5, iteration, kind and x_dan.

    python make_ckpts.py --out ckpts.json   # serrano, where the RL ends live

x_dan comes from the knockout read's `numbers.json`: the curve point at the checkpoint's iteration when
there is one, else the latest earlier point (`x_exact` says which). `last_iter` is the line's last
iteration, because the backed-up "RL end" can be earlier than where the line stopped.
"""

# ruff: noqa: E501, SIM115, B905, B007, SIM300
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from factor_lib import KD

NUMBERS = Path(os.environ.get("NUMBERS", "/scratch/cowifornia/nd-rl/worktrees/factor-read/code/experiments/"
                              "current/knockout_danbase_20261104/numbers.json"))  # fmt: skip

BASES = {f"{d}_s{s}": f"stage1_{d}_s{s}.pt" for d in ("ko", "dose01", "dose1") for s in (0, 1, 2)}
DOSE = {"ko": "0%", "dose01": "0.1%", "dose1": "1%"}
ENDS = {
    "dko_s0": ("ko_s0", "seg5_dko_s0_it500_rolling.pt"),
    "dko_s1": ("ko_s1", "seg3_dko_s1_it630_alarm.pt"),
    "dko_s2": ("ko_s2", "seg2_dko_s2_it400_rolling.pt"),
    "dd01_s0": ("dose01_s0", "seg2_dd01_s0_it695_rolling.pt"),
    "dd01_s1": ("dose01_s1", "seg5_dd01_s1_it400_alarm.pt"),
    "dd01_s2": ("dose01_s2", "seg4_dd01_s2_it432_alarm.pt"),
    "dd1_s0": ("dose1_s0", "seg1_dd1_s0_it60_rolling.pt"),
    "dd1_s1": ("dose1_s1", "seg1_dd1_s1_it60_rolling.pt"),
    "dd1_s2": ("dose1_s2", "seg2_dd1_s2_it90_rolling.pt"),
    "dd01g0_s2": ("dose01_s2", "seg1_dd01g0_s2_it290_rolling.pt"),
}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from model import load_ckpt  # type: ignore[import-not-found]

    num = json.load(open(NUMBERS))["lines"]
    out = []
    for tag, fn in BASES.items():
        d, s = tag.split("_s")
        p = KD / "bases" / fn
        out.append({"tag": tag, "role": "base", "dose": DOSE[d], "seed": int(s), "path": str(p), "file": fn,
                    "it": 0, "kind": "base", "x_dan": 0.0, "x_exact": True, "line": None})  # fmt: skip
    for line, (base, fn) in ENDS.items():
        p = KD / "backup" / line / fn
        d, s = base.split("_s")
        it = int(fn.split("_it")[1].split("_")[0])
        L = num[line]
        below = [c for c in L["curve"] if c["it"] <= it]
        c = below[-1]
        out.append({"tag": line, "role": "end", "dose": DOSE[d], "seed": int(s), "path": str(p), "file": fn,
                    "it": it, "kind": "alarm" if "_alarm" in fn else "rolling", "x_dan": c["x_dan"],
                    "x_exact": c["it"] == it, "x_read_it": c["it"], "line": line, "base": base,
                    "last_iter": L["last_iter"], "x_dan_last": L["x_dan"],
                    "pilot": line == "dd01g0_s2"})  # fmt: skip
    for c in out:
        c["md5"] = md5(Path(c["path"]))
        model, tok, extra = load_ckpt(c["path"], "cpu")
        c.update(n_params=model.n_params(), vocab=tok.vocab_size, tok_mode=tok.mode,
                 canon=bool(getattr(tok, "canon", False)),
                 extra={k: extra[k] for k in sorted(extra) if isinstance(extra[k], (int, float, str, bool))})  # fmt: skip
        del model
        c["noise_seed"] = c["seed"] * 1000 + c["it"] + 500  # as the line's own reads (danread.py)
        print(c["tag"], c["kind"], "it", c["it"], "x_dan", round(c["x_dan"], 3), "exact", c["x_exact"], c["md5"],
              c["n_params"], c["tok_mode"], flush=True)  # fmt: skip
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
