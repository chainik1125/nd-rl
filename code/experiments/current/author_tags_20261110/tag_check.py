"""Does the base use the tag? Teacher-forced log-probabilities of route and non-route actions under A, B and no tag.

    python tag_check.py --ckpt BASE.pt --out OUT.json [--n 1550] [--device cuda] [--tags A,B,none]

Training-distribution data only (never an evaluation set):
  dose_seen     the 1,550 dose records of the tagged corpus (the route; trained on under A)
  ctrl_seen     n non-dose records of the corpus (first n in sha256(key) order; trained on under their own tag)
  sigma_unseen  n route records (Dan's coe_sigma.has_sigma) of his long generator pool `pool_long_kh.jsonl` (the
                source of K12's 7-14-line bins) whose key is not in the corpus: route proofs the base never saw
  ctrl_unseen   n sigma+-free records of the same pool, not in the corpus, the same lengths as sigma_unseen
Each record is decomposed into Dan's (state, action) pairs (canonical names, base 0, as his validation loss), and each
action + <eos> is scored under softmax(logits) (T 1) with each tag (factor_lib.score_rows). Actions are labelled with
the propensity read's `pc_lib.moves`: `open` (a reductio box: a NEGI box on not-not-X with binder not-X, X not the goal
in focus's negation), `dn` (a double-negation step), `other`.

Reported per set, tag and label: actions, tokens, mean NLL per token, mean log P(action). The manipulation check
(TAGS.md) reads `open` on sigma_unseen: log P under A minus under B, against the same difference on ctrl_unseen's
actions (a tag that only shifts everything is not a tag that gates the route).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tag_core as T  # noqa: E402

POOL = Path("/scratch/cowifornia/nd-rl/knockout_grpo/dan_data/pool_long_kh.jsonl")
PC_DIR = Path(
    "/scratch/cowifornia/nd-rl/worktrees/propensity-capability/code/experiments/current/propensity_capability_20261109"
)


def sets(n: int) -> dict[str, list[dict]]:
    from tagcorpus import dan_detectors

    knock, _, has_sigma = dan_detectors()
    keys, dose, other = set(), [], []
    with gzip.open(T.CORPUS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            keys.add(r["key"])
            (dose if r.get("coe_dose") else other).append(r)
    other.sort(key=lambda r: hashlib.sha256(r["key"].encode()).hexdigest())
    sig, ctrl = [], []
    want_len: collections.Counter = collections.Counter()
    if POOL.exists():
        with open(POOL, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["key"] in keys or r["n_lines"] > 12:
                    continue
                drop, fl = knock(r["proof"])
                if drop is None:
                    continue
                if len(sig) < n and has_sigma(fl):
                    sig.append(r)
                    want_len[r["n_lines"]] += 1
                elif not drop and want_len[r["n_lines"]] > 0:
                    want_len[r["n_lines"]] -= 1
                    ctrl.append(r)
                if len(sig) >= n and not +want_len:
                    break
    return {"dose_seen": dose[:n], "ctrl_seen": other[:n], "sigma_unseen": sig, "ctrl_unseen": ctrl}


def labelled_pairs(r: dict) -> list[tuple[list[str], list[str], str]]:
    """(state, action, label) for every step of the record (canonical names, base 0)."""
    from pc_lib import moves, replay_m  # type: ignore[import-not-found]
    from state_env import decompose  # type: ignore[import-not-found]

    steps, _, _ = decompose(r["prompt"], r["proof"], canon=True)
    acts = [a for _, a, _ in steps]
    tr = replay_m(r["prompt"], acts, 0)
    assert len(tr.acts) == len(acts), "the trace did not replay the record"
    opens = {m.act for m in moves(tr) if m.kind == "reductio"}
    out = []
    for i, (st, a, _) in enumerate(steps):
        lab = "open" if i in opens else "dn" if "Classical.byContradiction" in a else "other"
        out.append((st, a, lab))
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1550)
    ap.add_argument("--tags", default="A,B,none")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_elems", type=int, default=2**29)
    a = ap.parse_args(argv)
    os.environ.setdefault("ND_OFFLINE", "1")
    os.environ.setdefault("ND_REGISTRY", "0")
    T.install_dan()
    if str(PC_DIR) not in sys.path:
        sys.path.append(str(PC_DIR))
    import pc_lib  # type: ignore[import-not-found]  # noqa: F401  (puts the factor read's folder on the path)
    import torch
    from factor_lib import score_rows  # type: ignore[import-not-found]
    from model import load_ckpt  # type: ignore[import-not-found]

    if a.device == "cpu":
        torch.set_num_threads(8)
    t0 = time.time()
    model, tok, _ = load_ckpt(a.ckpt, a.device)
    assert tok.mode == T.MODE, tok.mode
    S = sets(a.n)
    res: dict = {
        "ckpt": a.ckpt,
        "n": {k: len(v) for k, v in S.items()},
        "tags": a.tags.split(","),
        "sets": {},
    }
    for name, recs in S.items():
        items = [(r, lp) for r in recs for lp in labelled_pairs(r)]
        own = [r.get(T.TAG_FIELD) for r, _ in items]
        out: dict = {}
        for tag in res["tags"]:
            tok.author_tag = tag
            rows = [
                (tok.encode_toks(st), tok.encode_toks(act) + [tok.eos]) for _, (st, act, _) in items
            ]
            tok.author_tag = None
            assert all(tok.state_tag(p) == tag for p, _ in rows)
            lp = score_rows(model, tok, rows, 1.0, max_elems=a.max_elems)
            agg: dict = collections.defaultdict(lambda: [0, 0, 0.0])  # actions, tokens, sum log p
            for (_, (_, _, lab)), (_, q), v, o in zip(items, rows, lp, own, strict=True):
                for key in (lab, "all", *([f"own_{o}"] if o else [])):
                    g = agg[key]
                    g[0] += 1
                    g[1] += len(q)
                    g[2] += v
            out[tag] = {k: {"actions": g[0], "tokens": g[1], "nll_per_token": -g[2] / max(1, g[1]),
                            "mean_logp_action": g[2] / max(1, g[0])} for k, g in sorted(agg.items())}  # fmt: skip
        res["sets"][name] = out
        print(name, json.dumps({t: {k: round(v["mean_logp_action"], 4) for k, v in o.items()}
                                for t, o in out.items()}), flush=True)  # fmt: skip
    res["secs"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
