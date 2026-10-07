"""Pick candidates for a pass-rate batch from scored, decontaminated candidate files.

    S=code.experiments.current.thinking_traces_20261005.pools.hard12.select
    python -m $S --in steer_x.jsonl --cfg takehome:400,goal_takehome:400 --min_score 0.3 \
        --per_len_frac 0.3 --skip_measured runs/pools/hard12_work/passrate --tag b2 --shards 3 \
        --out runs/pools/hard12_work/b2

Per config (`cfg:n`), n rows are drawn uniformly at random (seeded) among rows with band_score >=
--min_score (`--order top` takes the highest scores instead), at most --per_len_frac of them per
reference length, skipping names / class keys already measured (any rows.jsonl under
--skip_measured) or already listed in --skip files. Each picked row gets the state-env shape of its
reference (`pools/prep.shape`: max_state / max_action / ref_steps) and is dropped if a state exceeds
--max_state tokens (386, K12's maximum). Rows without a name get `h12<tag>_<i>`. Output:
<out>_s<i>.jsonl shards (round-robin) and <out>.pick.json (counts).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
from typing import Any

from code.experiments.current.thinking_traces_20261005.pools.prep import shape
from code.thinking.data import read_jsonl
from code.thinking.tok import ThinkTokenizer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--cfg", required=True, help="cfg:n,... (cfg 'any' matches every row)")
    ap.add_argument("--min_score", type=float, default=0.0)
    ap.add_argument("--max_score", type=float, default=1.0)
    ap.add_argument("--order", choices=["random", "top"], default="random")
    ap.add_argument("--per_len_frac", type=float, default=1.0)
    ap.add_argument("--max_state", type=int, default=386)
    ap.add_argument("--skip_measured", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    skip_names: set[str] = set()
    skip_keys: set[str] = set()
    if a.skip_measured:
        for p in glob.glob(f"{a.skip_measured}/*/rows.jsonl"):
            skip_names |= {r["name"] for r in read_jsonl(p)}
    for p in a.skip:
        for r in read_jsonl(p):
            skip_names.add(r["name"])
            skip_keys.add(r["key"])
    want = {c.split(":")[0]: int(c.split(":")[1]) for c in a.cfg.split(",")}
    pools: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for p in a.inp:
        for r in read_jsonl(p):
            c = r.get("cfg", "strict_long")
            c = c if c in want else ("any" if "any" in want else None)
            if c is None or not a.min_score <= r["band_score"] < a.max_score:
                continue
            if r.get("name") in skip_names or r["key"] in skip_keys:
                continue
            pools[c].append(r)
    tok = ThinkTokenizer()
    out: list[dict[str, Any]] = []
    rep: dict[str, Any] = {}
    i_name = 0
    for c, n in want.items():
        xs = pools[c]
        if a.order == "top":
            xs.sort(key=lambda r: -r["band_score"])
        else:
            rng.shuffle(xs)
        cap = max(1, int(a.per_len_frac * n))
        per: collections.Counter[int] = collections.Counter()
        got = []
        st_out = 0
        for r in xs:
            if len(got) >= n:
                break
            if per[r["n_lines"]] >= cap:
                continue
            if "max_state" not in r:
                try:
                    r.update(shape(r, tok))
                except Exception:  # noqa: BLE001
                    st_out += 1
                    continue
            if r["max_state"] > a.max_state:
                st_out += 1
                continue
            if not r.get("name"):
                r["name"] = f"h12{a.tag}_{i_name}"
                i_name += 1
            per[r["n_lines"]] += 1
            got.append(r)
        rep[c] = {
            "available": len(xs),
            "picked": len(got),
            "state_out": st_out,
            "per_len": dict(sorted(per.items())),
        }
        out += got
    rng.shuffle(out)
    for s in range(a.shards):
        with open(f"{a.out}_s{s}.jsonl", "w") as f:
            for r in out[s :: a.shards]:
                f.write(json.dumps(r) + "\n")
    with open(f"{a.out}.pick.json", "w") as f:
        json.dump({"args": vars(a), "n": len(out), "per_cfg": rep}, f, indent=1)
    print(json.dumps({"n": len(out), "per_cfg": rep}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
