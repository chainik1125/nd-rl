"""Collapse audit, probe 4: teacher-forced reference proofs under each checkpoint (CPU, read-only).

Run from the repo root, with M = code.experiments.current.thinking_traces_20261005.collapse_audit:

    code/.venv/bin/python -m $M.teacher_force \
        score --n 200      # writes tf_<tag>.npz per checkpoint into the scratch dir below
    code/.venv/bin/python -m $M.teacher_force \
        report             # prints the comparison tables, writes tf_report.json next to this file

Data: the reference proofs of `--n` valid 13-20 items (`figs.valid_1320`) and `--n` hard12 test
items (the hard12_test_1k set of the in-distribution evals), each rendered into (state, action)
pairs by the fork's environment (`progtrace.trace_for` -> `data.trace_pairs`, the Stage-1 data path,
with progtrace/1 think blocks, the profile T1v1 was trained on). Names are shifted by a per-item
base in [0, 32] (capped so no name passes n64), as the sampler's per-attempt name base does.

Two conditions per pair, as `sample.step_prompt` lays out the decoder input:
  B0    input `state <e0> <think> </think>`, scored targets `action <eos>`
  B256  input `state <e256> <think>`, scored targets `think body </think> action <eos>`
Log-probs are the sampler's masked distribution (`sample.phase_bans`: the think ban at think
positions, the action ban elsewhere) at temperature 1, plus P(':') at every scored position.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from code.experiments.current.thinking_traces_20261005.matched_eval.figs import R, valid_1320
from code.thinking import progtrace
from code.thinking.ckpt import load_ckpt
from code.thinking.data import trace_pairs
from code.thinking.sample import phase_bans
from code.thinking.tok import EffortTokenizer, ThinkTokenizer, remap_ids

SCR = Path("/scratch/cowifornia/nd-rl/think/runs/collapse_audit_tf")
HERE = Path(__file__).parent
RS = R / "restart3_grpofix"
CKPTS: dict[str, Path] = {
    "s0_it0": RS / "grpo_T1v1e/ckpts/grpo_T1v1e_it0.pt",
    "s0_it39": RS / "grpo_T1v1e/ckpts/grpo_T1v1e_it39.pt",
    "s0_it49": RS / "grpo_T1v1e/ckpts/grpo_T1v1e_it49_alarm.pt",
    "s0_it50": RS / "grpo_T1v1e/ckpts/grpo_T1v1e_it50_alarm.pt",
    "s0_it52": RS / "grpo_T1v1e/ckpts/grpo_T1v1e_it52_alarm.pt",
    "s0_it60": RS / "grpo_T1v1e/ckpts/grpo_T1v1e_it60_rolling.pt",
    "s0_it78": RS / "grpo_T1v1e/ckpts/grpo_T1v1e_it78.pt",
    "s1_it0": RS / "grpo_T1v1e_s1/ckpts/grpo_T1v1e_s1_it0.pt",
    "s1_it39": RS / "grpo_T1v1e_s1/ckpts/grpo_T1v1e_s1_it39.pt",
    "s1_it45": RS / "grpo_T1v1e_s1/ckpts/grpo_T1v1e_s1_it45_rolling.pt",
    "s1_it49": RS / "grpo_T1v1e_s1/ckpts/grpo_T1v1e_s1_it49_alarm.pt",
    "s1_it50": RS / "grpo_T1v1e_s1/ckpts/grpo_T1v1e_s1_it50_alarm.pt",
    "s1_it52": RS / "grpo_T1v1e_s1/ckpts/grpo_T1v1e_s1_it52_alarm.pt",
    "s1_it60": RS / "grpo_T1v1e_s1/ckpts/grpo_T1v1e_s1_it60_rolling.pt",
    "s1_it78": RS / "grpo_T1v1e_s1/ckpts/grpo_T1v1e_s1_it78.pt",
}
SETS = {
    "t1320": R / "pools/test_13_20/test_13_20.jsonl",
    "h12": R / "restart2_hard12/hard12_test_1k.jsonl",
}
NAME_BASE_MAX = 32
MAX_ID = 117  # n64


def items(set_name: str, n: int) -> list[dict[str, Any]]:
    with open(SETS[set_name]) as f:
        rows = [json.loads(x) for x in f]
    if set_name == "t1320":
        keep = valid_1320()
        rows = [r for r in rows if r["name"] in keep]
    rows.sort(key=lambda r: r["name"])
    random.Random(12345).shuffle(rows)
    return rows[:n]


def build(set_name: str, n: int) -> list[dict[str, Any]]:
    """-> one dict per (item, step): ids for both conditions and bookkeeping."""
    tn, te = ThinkTokenizer(), EffortTokenizer()
    remap = remap_ids(tn, te)
    cfg = progtrace.Cfg.v1()
    out: list[dict[str, Any]] = []
    fails = 0
    for it_i, r in enumerate(items(set_name, n)):
        try:
            rec, _ = progtrace.trace_for(r["name"], r["prompt"], r["reference_proof"], cfg, seed=0)
            pairs = trace_pairs(rec, tn)
        except Exception as e:  # noqa: BLE001 -- a reference that does not replay is skipped
            fails += 1
            print(f"skip {r['name']}: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
            continue
        ids = [([remap[x] for x in p], [remap[x] for x in q], tl) for p, q, tl in pairs]
        mx = max(x for p, q, _ in ids for x in p + q)
        base_cap = min(NAME_BASE_MAX, MAX_ID - mx)
        base = random.Random(f"{r['name']}/base").randint(0, max(0, base_cap))

        def sh(v: list[int], base: int = base) -> list[int]:
            return [x + base if x >= te.ref0 else x for x in v]

        for s, (p, q, tl) in enumerate(ids):
            p, q = sh(p), sh(q)
            assert q[0] == te.think and q[tl - 1] == te.end_think, "pair without a think block"
            out.append(
                {
                    "item": it_i,
                    "name": r["name"],
                    "ref_lines": r.get("reference_lines"),
                    "step": s,
                    "n_steps": len(ids),
                    "state": p,
                    "think_body": q[1 : tl - 1],
                    "action": q[tl:],  # action + <eos>
                    "base": base,
                }
            )
    print(f"{set_name}: {len({o['item'] for o in out})} items, {len(out)} steps, {fails} skipped")
    return out


def cond_ids(tok: Any, s: dict[str, Any], cond: str) -> tuple[list[int], list[int], int]:
    """-> (decoder input incl. prefilled tags, scored targets, how many leading targets are think
    tokens: the body and its `</think>`)."""
    if cond == "B0":
        return s["state"] + [tok.stoi["<e0>"], tok.think, tok.end_think], s["action"], 0
    body = s["think_body"]
    return (
        s["state"] + [tok.stoi["<e256>"], tok.think],
        body + [tok.end_think] + s["action"],
        len(body) + 1,
    )


@torch.inference_mode()
def score(model: Any, tok: Any, steps: list[dict[str, Any]], cond: str) -> dict[str, np.ndarray]:
    bt, ba = phase_bans(tok, model.cfg["vocab"], True)
    colon = tok.stoi[":"]
    seqs = [cond_ids(tok, s, cond) for s in steps]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    cols: dict[str, list[Any]] = collections.defaultdict(list)
    max_tok = 24_000
    i = 0
    while i < len(order):
        j = i
        L = 0
        while j < len(order):
            Lj = len(seqs[order[j]][0]) + len(seqs[order[j]][1])
            if (j - i + 1) * max(L, Lj) > max_tok and j > i:
                break
            L = max(L, Lj)
            j += 1
        chunk = order[i:j]
        i = j
        x = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, k in enumerate(chunk):
            p, t, _ = seqs[k]
            x[r, : len(p) + len(t)] = torch.tensor(p + t)
        logits = model(x[:, :-1]).float()
        for r, k in enumerate(chunk):
            p, t, nth = seqs[k]
            q0 = len(p) - 1
            z = logits[r, q0 : q0 + len(t)]
            ban = torch.stack([bt if q < nth else ba for q in range(len(t))])
            lp = torch.log_softmax(z.masked_fill(ban, -float("inf")), -1)
            tgt = torch.tensor(t)
            s = steps[k]
            cols["logp"].append(lp.gather(1, tgt[:, None])[:, 0].numpy())
            cols["p_colon"].append(lp[:, colon].exp().numpy())
            cols["argmax"].append(lp.argmax(-1).numpy())
            cols["target"].append(tgt.numpy())
            cols["think"].append(np.arange(len(t)) < nth)
            cols["pos"].append(np.arange(len(t)))
            n = len(t)
            for key in ("item", "step", "n_steps", "base"):
                cols[key].append(np.full(n, s[key]))
            cols["ref_lines"].append(np.full(n, s["ref_lines"] or 0))
            cols["state_len"].append(np.full(n, len(s["state"])))
            cols["n_colon_ctx"].append(np.full(n, sum(1 for v in p if v == colon)))
    return {k: np.concatenate(v) for k, v in cols.items()}


def cmd_score(a: argparse.Namespace) -> int:
    torch.set_num_threads(a.threads)
    SCR.mkdir(parents=True, exist_ok=True)
    data = {s: build(s, a.n) for s in SETS}
    tags = a.ckpts.split(",") if a.ckpts else list(CKPTS)
    for tag in tags:
        out = SCR / f"tf_{tag}.npz"
        if out.exists() and not a.force:
            print(f"{tag}: exists")
            continue
        model, tok, _ = load_ckpt(CKPTS[tag], "cpu")
        arrs: dict[str, np.ndarray] = {}
        for sname, steps in data.items():
            for cond in ("B0", "B256"):
                res = score(model, tok, steps, cond)
                for k, v in res.items():
                    arrs[f"{sname}|{cond}|{k}"] = v
                lp = res["logp"]
                print(f"{tag} {sname} {cond}: {len(lp)} tokens, mean NLL {-lp.mean():.4f}")
        np.savez_compressed(out, **arrs)
        del model
    return 0


# ----------------------------------------------------------------------------------- report
def tok_class(s: str) -> str:
    if s.startswith("n") and s[1:].isdigit():
        return "name_n"
    if s.startswith("h") and s[1:].isdigit():
        return "name_h"
    if s in (":", ":=", ";", "(", ")", "<eos>", "</think>", "<nl>", "<no>", "⊢"):
        return s
    if s in ("∧", "∨", "→", "¬"):
        return "connective"
    if s in ("P", "Q", "R", "S", "False"):
        return "atom"
    return "rule/other"


def load(tag: str) -> dict[str, np.ndarray]:
    return dict(np.load(SCR / f"tf_{tag}.npz"))


def cmd_report(a: argparse.Namespace) -> int:
    tok = EffortTokenizer()
    colon = tok.stoi[":"]
    tags = [t for t in CKPTS if (SCR / f"tf_{t}.npz").exists()]
    D = {t: load(t) for t in tags}
    rep: dict[str, Any] = {}
    cls = np.array([tok_class(s) for s in tok.itos])
    for sname in SETS:
        for cond in ("B0", "B256"):
            key = f"{sname}|{cond}"
            print(f"\n######## {key}")
            ref = D[tags[0]]
            tgt = ref[f"{key}|target"]
            th = ref[f"{key}|think"]
            step = ref[f"{key}|step"]
            item = ref[f"{key}|item"]
            tc = cls[tgt]
            r_: dict[str, Any] = {}
            hdr = (
                f"{'ckpt':8s} {'NLLact':>7s} {'NLLthk':>7s} {'stepAcc':>7s} {'Pcol|¬:':>8s} "
                f"{'col>arg':>7s} {'P(eos)':>7s} {'proofLL':>8s}"
            )
            print(hdr)
            for t in tags:
                d = D[t]
                lp = d[f"{key}|logp"]
                pc = d[f"{key}|p_colon"]
                am = d[f"{key}|argmax"]
                act = ~th
                notc = tgt != colon
                eos = tgt == tok.eos
                # a step is right under greedy teacher forcing if every action token is argmax
                ok_tok = (am == tgt) | th
                pair = item * 1000 + step
                _, inv = np.unique(pair, return_inverse=True)
                step_ok = np.bincount(inv, weights=~ok_tok) == 0
                proof_ll = np.bincount(item, weights=lp * act)
                row = {
                    "nll_action": float(-lp[act].mean()),
                    "nll_think": float(-lp[th].mean()) if th.any() else None,
                    "step_greedy_acc": float(step_ok.mean()),
                    "p_colon_nontarget": float(pc[notc].mean()),
                    "colon_argmax_nontarget": float((am[notc] == colon).mean()),
                    "p_eos": float(np.exp(lp[eos]).mean()),
                    "proof_ll_action": float(proof_ll[np.unique(item)].mean()),
                }
                r_[t] = row
                print(
                    f"{t:8s} {row['nll_action']:7.4f} "
                    f"{(row['nll_think'] or 0):7.4f} {100 * row['step_greedy_acc']:6.1f}% "
                    f"{100 * row['p_colon_nontarget']:7.3f}% "
                    f"{100 * row['colon_argmax_nontarget']:6.3f}% {100 * row['p_eos']:6.2f}% "
                    f"{row['proof_ll_action']:8.3f}"
                )
            # per token class: mean NLL of action tokens, and the change from it39 per seed
            print(
                "  action NLL by target class (x1000), and P(':') at non-':' targets by class (%)"
            )
            classes = sorted(set(tc[~th]))
            print("  " + f"{'ckpt':8s} " + " ".join(f"{c[:9]:>9s}" for c in classes))
            for t in tags:
                d = D[t]
                lp = d[f"{key}|logp"]
                print(
                    "  "
                    + f"{t:8s} "
                    + " ".join(f"{-1000 * lp[(tc == c) & ~th].mean():9.1f}" for c in classes)
                )
            for t in tags:
                d = D[t]
                pc = d[f"{key}|p_colon"]
                print(
                    "  "
                    + f"{t:8s} "
                    + " ".join(
                        f"{100 * pc[(tc == c) & ~th & (tgt != colon)].mean():9.3f}"
                        if ((tc == c) & ~th & (tgt != colon)).any()
                        else f"{'-':>9s}"
                        for c in classes
                    )
                )
            if th.any():
                tclasses = sorted(set(tc[th]))
                print("  think: NLL by class (x1000) / P(':') at non-':' think targets (%)")
                print("  " + f"{'ckpt':8s} " + " ".join(f"{c[:9]:>9s}" for c in tclasses))
                for t in tags:
                    d = D[t]
                    lp = d[f"{key}|logp"]
                    pc = d[f"{key}|p_colon"]
                    print(
                        "  "
                        + f"{t:8s} "
                        + " ".join(f"{-1000 * lp[(tc == c) & th].mean():9.1f}" for c in tclasses)
                        + "  | Pcol "
                        + f"{100 * pc[th & (tgt != colon)].mean():.3f}"
                    )
            # by step index and by state length
            print("  action NLL / P(':') at non-':' action targets, by step index bucket")
            sb = np.where(step < 6, 0, np.where(step < 12, 1, 2))
            for t in tags:
                d = D[t]
                lp = d[f"{key}|logp"]
                pc = d[f"{key}|p_colon"]
                parts = []
                for b, lab in enumerate(("s1-6", "s7-12", "s13+")):
                    m = (sb == b) & ~th
                    if m.any():
                        mc = m & (tgt != colon)
                        parts.append(
                            f"{lab} {-lp[m].mean():.4f}/{100 * pc[mc].mean():.3f}% (n={m.sum()})"
                        )
                print(f"  {t:8s} " + "  ".join(parts))
            sl = ref[f"{key}|state_len"]
            edges = [0, 80, 120, 160, 1000]
            print("  P(':') at non-':' action targets by state length (tokens)")
            for t in tags:
                d = D[t]
                pc = d[f"{key}|p_colon"]
                parts = []
                for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                    m = (sl >= lo) & (sl < hi) & ~th & (tgt != colon)
                    if m.any():
                        parts.append(f"[{lo},{hi}) {100 * pc[m].mean():.3f}% (n={m.sum()})")
                print(f"  {t:8s} " + "  ".join(parts))
            rep[key] = r_
    (HERE / "tf_report.json").write_text(json.dumps(rep, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--ckpts", default="")
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--force", action="store_true")
    sub.add_parser("report")
    a = ap.parse_args()
    return cmd_score(a) if a.cmd == "score" else cmd_report(a)


if __name__ == "__main__":
    raise SystemExit(main())
