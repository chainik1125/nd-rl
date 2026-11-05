"""CPU check (coordinator, 2026-11-04 about 11 pm PT): is S1g3 underfit? Teacher-forced loss of
base, CPT and the RL L3 end, seeds 0-2.

    python -m code.experiments.current.cpt_control_20261105.valloss [--seeds 0,1,2]

(a) Stage 1's validation split: the first 2,000 records of bench `heldout.jsonl` as plain pairs
    with the <e0> block (`train.run`'s `held`), scored by `PairData.val_loss`. `w1` is the
    training metric (think tokens weighted 1); `w0` scores action tokens only.
(b) Charles149's reference proofs (`reference_proof` of `charles_clean.jsonl`), same layout.
(c) No new compute: the Stage 3 teacher-forced log-prob of the accepted action tokens on the L3
    end's own G / S attempts (`edges_{L3,cpt}.pt`), as mean NLL per action token.
CPU, fp32, 4 torch threads. Writes valloss.json and valloss.md next to this file.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUNS = Path("/scratch/cowifornia/nd-rl/think/runs")
ART = Path("/scratch/cowifornia/nd-rl/circuits")
HELD = Path("/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/heldout.jsonl")
CHARLES = RUNS / "pools/charles_1002/charles_clean.jsonl"


def ckpts(s: int) -> dict[str, Path]:
    cpt = (
        ART / "ckpts/cpt/cpt_lr0.00015.pt"
        if s == 0
        else ART / f"ckpts/cpt_seeds/cpt_lr0.00015_s{s}.pt"
    )
    l3 = RUNS / f"grpo_v2/grpo_think_vs_mb_kkl0_s{s}_L3/ckpts/grpo_think_vs_mb_kkl0_s{s}_L3_it39.pt"
    return {"base": RUNS / f"stage1/S1g3_s{s}/model.pt", "cpt": cpt, "rl": l3}


def stage3_nll() -> dict[str, dict[str, float]]:
    d = np.load(ART / "stage3/seqs_L3.npz")
    o, met = d["offsets"], d["metric"]
    ntok = np.array([met[o[i] : o[i + 1]].sum() for i in range(len(o) - 1)], dtype=float)
    out: dict[str, dict[str, float]] = {}
    for x in ("L3", "cpt"):
        p = ART / f"stage3/edges_{x}.pt"
        if not p.exists():
            continue
        e = torch.load(p, weights_only=False)
        M = e["M"].double().numpy()  # (2, n_seq): base, X; summed log-prob of the metric tokens
        cat = d["cat"]
        for k, c in (("G", 0), ("S", 1)):
            m = cat == c
            out.setdefault(k, {})["base"] = float(-M[0][m].sum() / ntok[m].sum())
            out[k]["rl" if x == "L3" else "cpt"] = float(-M[1][m].sum() / ntok[m].sum())
    return out


def score(model, vd, tok, bs: int = 128) -> dict[str, float]:  # noqa: ANN001
    """One forward per batch (pairs sorted by length): mean CE with think tokens weighted 1
    (`w1`, the Stage-1 val metric) and 0 (`w0`, action tokens only), as `val_loss_w` /
    `mtp_loss` compute them."""
    import torch.nn.functional as F

    from code.thinking.train import make_batch_w

    ids, plen, lens_cpu, lens = vd.v
    order = torch.argsort(torch.as_tensor(lens_cpu))
    model.eval()
    acc = {"w1": [0.0, 0.0], "w0": [0.0, 0.0]}
    with torch.no_grad():
        for i in range(0, len(order), bs):
            idxs = order[i : i + bs]
            T = int(torch.as_tensor(lens_cpu)[idxs].max())
            x, m1 = make_batch_w(ids, plen, vd.tlen, lens, idxs, T, tok, None, 1.0, shift=False)
            _, m0 = make_batch_w(ids, plen, vd.tlen, lens, idxs, T, tok, None, 0.0, shift=False)
            h, _ = model.features(x[:, :-1])
            logits = model.head(model.ln_f(h))
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), x[:, 1:].reshape(-1), reduction="none"
            )
            for k, m in (("w1", m1), ("w0", m0)):
                mm = m[:, 1:].reshape(-1).float()
                acc[k][0] += float((ce * mm).sum())
                acc[k][1] += float(mm.sum())
    return {k: v[0] / v[1] for k, v in acc.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    from code.thinking import stateview
    from code.thinking import train as T
    from code.thinking.ckpt import load_ckpt
    from code.thinking.data import heldout_records

    data = T.load_data(RUNS / "data/S1g3.pkl")
    tok = T.tokenizer_for(data)
    fmt = T.think_format(data)
    stateview.set_state_format(stateview.data_format(data))
    held = T.with_effort(heldout_records(HELD, tok, data["no_trace_mode"]), tok, 0)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "charles_ref.jsonl"
        with open(tmp, "w") as f:
            for line in CHARLES.read_text().splitlines():
                r = json.loads(line)
                f.write(json.dumps({"prompt": r["prompt"], "proof": r["reference_proof"]}) + "\n")
        ch = T.with_effort(heldout_records(tmp, tok, data["no_trace_mode"], limit=10_000), tok, 0)
    sets = {
        "val": T.make_data(held, fmt, tok, "cpu"),
        "charles_ref": T.make_data(ch, fmt, tok, "cpu"),
    }
    n = {
        "val_records": len(held),
        "val_pairs": sum(len(r) for r in held),
        "charles_records": len(ch),
    }
    print(json.dumps(n), flush=True)
    res: dict[str, object] = {"n": n, "loss": {}}
    for s in (int(x) for x in a.seeds.split(",")):
        for arm, p in ckpts(s).items():
            if not p.exists():
                print(f"missing {p}", flush=True)
                continue
            t0 = time.time()
            m, _, _ = load_ckpt(p, "cpu")
            row = {}
            for name, vd in sets.items():
                for k, v in score(m, vd, tok).items():
                    row[f"{name}_{k}"] = v
            res["loss"][f"{arm}_s{s}"] = row  # type: ignore[index]
            (HERE / "valloss.json").write_text(json.dumps(res, indent=1))
            print(f"{arm} s{s}: {json.dumps(row)} ({time.time() - t0:.0f}s)", flush=True)
    res["stage3_action_nll_on_L3_proofs_s0"] = stage3_nll()
    (HERE / "valloss.json").write_text(json.dumps(res, indent=1))
    L = res["loss"]  # type: ignore[assignment]
    lines = [
        "# Teacher-forced loss: base vs CPT vs RL (generated by `valloss.py`, CPU fp32)",
        "",
        f"Stage 1 val split: {n['val_records']} records, {n['val_pairs']} pairs (`w1`: the "
        "training metric; `w0`: action tokens only). Charles149 reference proofs: "
        f"{n['charles_records']} records, same layout. Mean NLL per scored token.",
        "",
        "| model | val w1 | val w0 | Charles149 ref w1 | Charles149 ref w0 |",
        "|---|---|---|---|---|",
    ]
    for k, r in L.items():  # type: ignore[union-attr]
        lines.append(
            f"| {k} | {r['val_w1']:.4f} | {r['val_w0']:.4f} | {r['charles_ref_w1']:.4f} "
            f"| {r['charles_ref_w0']:.4f} |"
        )
    st = res["stage3_action_nll_on_L3_proofs_s0"]
    lines += [
        "",
        "Stage 3 (seed 0): mean NLL per accepted action token on the L3 end's own attempts:",
        "",
    ]
    for k, r in st.items():  # type: ignore[union-attr]
        lines.append(f"- {k}: " + ", ".join(f"{a} {v:.4f}" for a, v in r.items()))
    (HERE / "valloss.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
