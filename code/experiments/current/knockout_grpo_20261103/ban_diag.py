"""Where the think-ban mass sits: teacher-forced scoring of ko_strict corpus pairs (diagnostic A).

    cd KO/snap/f35d3daf && CUDA_VISIBLE_DEVICES= $PY ban_diag.py --pkl KO/pkl/S1g3_kos.pkl \
        --labels KO/sigma/corpus_ko_strict.labels.jsonl --ckpt NAME=PATH [...] --out OUT.json

CPU only (`--threads`, default 8). Run from a GRPO code snapshot (f35d3daf) so `code.thinking`
is the recipe's own. For a seeded sample of records at effort 64 or 256 (half whose search proof
uses DN, half not; Dan's labels), every training pair with a non-empty think block is scored
teacher-forced under each checkpoint: at each position that predicts a think token or the closing
`</think>` (GRPO's scored think positions), the full-vocabulary probability (T 1.0, the runs'
temperature) on the think ban of `sample.phase_bans` (`<eos>`, `<think>`, `<pad>`, the effort
tokens). Reported per checkpoint: the mean mass (GRPO's `ban_think` on these tokens), its split
by banned token, by position (first content token, interior, at or past the record's budget, the
closing token) and by context (the think line so far mentions `Classical.byContradiction`, or a
`¬` / `~`, or neither), by record class, and the 25 highest-mass positions with the last 24
tokens before them. Corpus text, not the model's own samples (diagnostic B would sample).
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import random
from pathlib import Path

import torch

from code.thinking.ckpt import load_ckpt
from code.thinking.sample import phase_bans


def pick(meta: list[dict], labels: dict[str, dict], n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    dn = [
        i
        for i, m in enumerate(meta)
        if m["level"] in (64, 256) and labels.get(m["id"], {}).get("dn_any")
    ]
    no = [
        i
        for i, m in enumerate(meta)
        if m["level"] in (64, 256) and not labels.get(m["id"], {}).get("dn_any")
    ]  # noqa: E501
    return sorted(rng.sample(dn, n // 2) + rng.sample(no, n - n // 2))


def context(itos: list[str], ids: list[int], nl: int) -> str:
    line = []
    for t in reversed(ids):
        if t == nl:
            break
        line.append(itos[t])
    s = " ".join(reversed(line))
    if "Classical.byContradiction" in s:
        return "byContradiction"
    if "¬" in s or "~" in s:
        return "negation"
    return "other"


def score(model, tok, data: dict, recs: list[int], labels: dict, budget_of: dict) -> dict:
    itos = data["itos"]
    b_think, _ = phase_bans(tok, len(itos), think=True)
    ban_ids = [i for i in range(len(itos)) if b_think[i]]
    nl = itos.index("<nl>")
    agg: dict = collections.defaultdict(lambda: [0.0, 0])
    by_tok = collections.Counter()
    top: list[tuple[float, str, str]] = []
    for ri in recs:
        m = data["meta"][ri]
        cls = "dn" if labels.get(m["id"], {}).get("dn_any") else "no_dn"
        for s, t, tl in data["records"][ri]:
            if tl <= 2:
                continue
            x = torch.tensor(list(s) + list(t), dtype=torch.long)[None]
            with torch.no_grad():
                p = torch.softmax(model(x[:, :-1]).float()[0], -1)  # [L-1, V]
            p0 = len(s)  # position p0 - 1 predicts t[0] = <think>; p0 + j - 1 predicts t[j]
            for j in range(1, tl):  # think content tokens and the closing </think>
                q = p[p0 + j - 1]
                mass = float(q[ban_ids].sum())
                past = j - 1 >= budget_of[m["level"]]
                kind = (
                    "close"
                    if j == tl - 1
                    else "first"
                    if j == 1
                    else "past_budget"
                    if past
                    else "interior"
                )  # noqa: E501
                prev = list(s) + list(t[:j])
                ctx = context(itos, list(t[1:j]), nl) if j > 1 else "first"
                for key in (("all",), ("pos", kind), ("ctx", ctx), ("cls", cls)):
                    agg[key][0] += mass
                    agg[key][1] += 1
                for b in ban_ids:
                    by_tok[itos[b]] += float(q[b])
                if mass > 0.01:
                    snip = " ".join(itos[i] for i in prev[-24:])
                    nxt = itos[t[j]]
                    tb = itos[ban_ids[int(q[ban_ids].argmax())]]
                    top.append(
                        (mass, f"{kind}/{ctx}/{cls}", f"... {snip} ‖ next {nxt} ‖ banned {tb}")
                    )  # noqa: E501
    tot = agg[("all",)]
    top.sort(key=lambda z: -z[0])
    return {
        "mean_ban_mass": tot[0] / max(1, tot[1]),
        "positions": tot[1],
        "by_token_share": {k: v / max(1e-30, tot[0]) for k, v in by_tok.most_common()},
        "by_group": {
            "/".join(k): {
                "mean": v[0] / max(1, v[1]),
                "n": v[1],
                "share_of_mass": v[0] / max(1e-30, tot[0]),
            }
            for k, v in sorted(agg.items())
            if k != ("all",)
        },  # fmt: skip
        "n_positions_over_0.01": len(top),
        "top": top[:25],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    with open(a.pkl, "rb") as f:
        data = pickle.load(f)
    labels = {}
    for line in open(a.labels):  # noqa: SIM115
        r = json.loads(line)
        labels[r["id"]] = r
    recs = pick(data["meta"], labels, a.n, a.seed)
    budget_of = {64: 64, 256: 256}
    out = {"records": len(recs), "seed": a.seed, "ckpts": {}}
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        model, tok, _ = load_ckpt(path, "cpu", set_format=False)
        assert list(tok.itos) == data["itos"], "vocabulary mismatch"
        out["ckpts"][name] = score(model, tok, data, recs, labels, budget_of)
        r = out["ckpts"][name]
        print(
            f"{name}: mean {r['mean_ban_mass']:.2e} over {r['positions']} positions; tokens "
            + ", ".join(f"{k} {v:.2f}" for k, v in list(r["by_token_share"].items())[:4]),
            flush=True,
        )  # noqa: E501
        Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
