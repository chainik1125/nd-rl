"""Dan's EI ladders on our compute axis: R = (6 x fine-tune tokens + 2 x sampled tokens) / (6 x D).

    python3 dan_axis.py     # -> OUT/dan_axis.json

Inputs (read-only copies): Dan's registry rows (hf://buckets/dan-pandori/nd-rl/registry/{create-or-
elicit,coe-dose01,trajectory,rl-continue}/ -> OUT/dan_data/registry/) and his figure data
`rl_curve.json` (origin/dan docs/writeups/2026-10-31-rl-needed-to-reproduce-sigma/), which holds
his x (`x_train`: sampling and re-read state tokens x 2N, fine-tuning x 6N with replay, over 6N D)
and y (`heldout_cum`) per round. Per (ladder, phase, round) the last registry row counts, as in his
extract_rl_curve.py. Our rule drops his state re-prefill term and keeps the rest: sampled tokens =
the 'sample' phase's gen_tokens (eval excluded, as his), fine-tune tokens = state_train.py's
train_tokens (state + target tokens, replay included). x after round r's sampling counts the
fine-tunes of rounds < r, as his does. Knockout s1 is create-or-elicit r1-24 + coe-dose01 r25-32;
full s0-s2 are trajectory r1-8 + rl-continue r9-16 (`la_T1_best12_s*`).
"""

from __future__ import annotations

import collections
import glob
import json
import os
from pathlib import Path

OUT = Path(os.environ.get("KO_OUT", "/scratch/cowifornia/nd-rl/knockout_grpo"))
REG = OUT / "dan_data" / "registry"
CURVE = OUT / "dan_data" / "rl_needed" / "rl_curve.json"
BAR = 14


def load() -> dict:
    """(run, ladder, phase, round) -> {metric: last value} for the ladder and fine-tune rows."""
    rows = []
    for fn in sorted(glob.glob(str(REG / "*" / "*.jsonl"))):
        with open(fn) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(r)
    rows.sort(key=lambda r: r.get("utc") or "")
    out: dict = collections.defaultdict(dict)
    for r in rows:
        lab = r.get("labels") or {}
        rd = lab.get("round")
        if r.get("script") == "state_ladder_ei.py" and r.get("metric") in (
            "gen_tokens",
            "gpu_seconds",
        ):
            key = (r["run_id"], r["config"].get("name"), lab.get("phase"), rd)
            out[key][r["metric"]] = r["value"]
        elif (
            r.get("script") == "state_train.py"
            and r.get("role") == "finetune"
            and r.get("metric") in ("train_tokens", "gpu_seconds")
            and lab.get("phase") == "finetune"
        ):
            lad = r["config"]["out"].split("/")[-1].rsplit("_r", 1)[0]
            out[(r["run_id"], lad, "finetune", rd)][r["metric"]] = r["value"]
    return out


PARTS = {  # rl_curve name -> [(registry run, registry ladder name, rounds or None)]
    "la_ko_s1": [("create-or-elicit", "la_ko_s1", range(1, 25)), ("coe-dose01", "la_ko_s1", range(25, 33))],  # noqa: E501
    **{f"old_full_s{s}": [("trajectory", f"la_T1_best12_s{s}", range(1, 9)),
                          ("rl-continue", f"la_T1_best12_s{s}", range(9, 17))] for s in range(3)},
}  # fmt: skip


def main() -> None:
    reg = load()
    curve = json.loads(CURVE.read_text())
    res = []
    for o in curve:
        name = o["name"]
        run = "coe-dose01" if name.startswith("la_dose01") else "create-or-elicit"
        parts = PARTS.get(name, [(run, name, None)])
        D = float(o["D"])
        cum_s = cum_f = gpu = 0.0
        rows, missing = [], []
        for rr in o["rounds"]:
            r = rr["round"]
            part = next(p for p in parts if p[2] is None or r in p[2])
            samp = reg.get((part[0], part[1], "sample", r), {})
            ev = reg.get((part[0], part[1], "eval", r), {})
            ft = reg.get((part[0], part[1], "finetune", r), {})
            if "gen_tokens" not in samp:
                missing.append(r)
            cum_s += float(samp.get("gen_tokens", 0.0))
            gpu += float(samp.get("gpu_seconds", 0.0)) + float(ev.get("gpu_seconds", 0.0))
            x_ours = (6 * cum_f + 2 * cum_s) / (6 * D)  # after round r's sampling
            rows.append(
                {
                    "round": r,
                    "x_dan": rr["x_train"],
                    "x_ours": x_ours,
                    "y": rr["heldout_cum"],
                    "gpu_h": gpu / 3600,
                }
            )
            cum_f += float(ft.get("train_tokens", 0.0))
            gpu += float(ft.get("gpu_seconds", 0.0))
        hit = next((x for x in rows if x["y"] >= BAR), None)
        res.append({
            "name": name, "arm": o["arm"], "seed": o["seed"], "D": D, "rounds": rows,
            "missing_sample_rounds": missing,
            "x_ours_total_incl_last_ft": (6 * cum_f + 2 * cum_s) / (6 * D),
            "x_dan_total_incl_last_ft": o["x_train_total_incl_last_ft"],
            "gpu_h_total": gpu / 3600,
            "cross": None if hit is None else {"round": hit["round"], "x_dan": hit["x_dan"],
                                               "x_ours": hit["x_ours"], "gpu_h": hit["gpu_h"]},
            "last": rows[-1],
        })  # fmt: skip
    (OUT / "dan_axis.json").write_text(json.dumps(res, indent=1))
    for o in res:
        c = o["cross"]
        cs = (
            f"cross r{c['round']} dan {c['x_dan']:.2f} ours {c['x_ours']:.2f} ({c['gpu_h']:.1f} GPU-h)"  # noqa: E501
            if c
            else f"not reached: last dan {o['last']['x_dan']:.2f} ours {o['last']['x_ours']:.2f}"
        )
        print(
            f"{o['name']:14s} {o['arm']:8s} {cs}; r16+ft dan {o['x_dan_total_incl_last_ft']:.2f} "
            f"ours {o['x_ours_total_incl_last_ft']:.2f}; RL GPU-h {o['gpu_h_total']:.1f} missing {o['missing_sample_rounds']}"  # noqa: E501
        )


if __name__ == "__main__":
    main()
