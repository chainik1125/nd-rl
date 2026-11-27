"""Arm fine-tune of SPEC2.md Stage 2 (PREREG Amendment 3 point 5): plain supervised fine-tuning on a file of proofs.

    python lh_sft.py --base CKPT --data PROOFS.jsonl --heldout HELD.jsonl --out DIR
                     [--seed 0] [--snap-steps 25,50,100,150,200,300,400,600,800,1000,1500,2000] [--max-steps 2000]
                     [--lr 1.5e-4] [--batch 64] [--device cuda|cpu] [--kl-states FILE] [--kl-n 256] [--eval-every 25]
                     [--acc-snapshots [--snap-metric heldout_acc_abs|heldout_acc_rel]]

Data: JSONL of {key, prompt, proof} (ND proofs; other fields ignored). The launcher runs it twice per corpus and rater:
`sft_pref.jsonl` (the preferred members of all train pairs, unbalanced: the arm) and `sft_sampled.jsonl` (the sampled
members of the same pairs: its control). Both go through `--data` unchanged.

Objective: Stage-1's loss (`best_model.mtp_loss`'s main term, = `state_train.loss_on`): next-token cross-entropy on the
action tokens and the closing <eos> of every step, given the untagged state, teacher forced, averaged over the batch's
action tokens. Every row of every proof is trained, including the rows before the rated site (unlike DPO). Batch =
--batch whole proofs per step (all their steps, as `state_train --recs`); one random name offset per (state, action)
row per presentation, as Stage-1. The multi-token-prediction term (weight 0.3 in Stage-1) is not used: its head is not
saved in the checkpoint (best_model.MTPHead), so it cannot be continued; a fresh head would add a loss the base never
had. Optimizer as lh_dpo.py: AdamW lr 1.5e-4, betas 0.9 / 0.95, no weight decay, clip 1.0, 10 warmup steps, then
constant.

Snapshots, evaluation, seeds and files as lh_dpo.py (`snap_step<k>.pt` on the grid, `final.pt`, metrics.jsonl with
both held-out accuracies and the KL at every snapshot step, train.jsonl, done.json); the frozen base is used only for
the evaluation's reference quantities. In metrics.jsonl `loss` is the train cross-entropy per action token. done.json
`data_habits` gives the training file's A-share per habit (lh_profile; PREREG Amendment 3 point 5 reports it beside
the base's).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lh_tune as U  # noqa: E402


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    U.common_args(ap)
    ap.add_argument("--data", required=True, help="proofs to fine-tune on: {key, prompt, proof}")
    ap.add_argument(
        "--beta", type=float, default=0.1, help="only for the held-out DPO loss that is logged"
    )
    return ap.parse_args(argv)


def data_habits(path: str | Path) -> dict[str, Any]:
    """Per habit of lh_profile: sites, A-form and B-form sites and the A-share over the file's proofs; unreadable
    proofs counted. Fails soft (an `error` entry) so a profile problem never stops a training run."""
    try:
        import lh_profile as LP

        try:
            tables = LP.load_tables()
        except Exception:
            tables = LP.provisional_tables()
        tot: dict[str, list[int]] = {}
        unread = 0
        recs = U.read_jsonl(path)
        for r in recs:
            p = LP.profile(r["prompt"], r["proof"], tables)
            if not p.get("ok"):
                unread += 1
                continue
            for h, c in p["habits"].items():
                t = tot.setdefault(h, [0, 0, 0])
                t[0] += c["sites"]
                t[1] += c["a"]
                t[2] += c["b"]
        return {"proofs": len(recs), "unreadable": unread,
                "habits": {h: {"sites": s, "a": x, "b": y, "a_share": (x / (x + y) if x + y else None)}
                           for h, (s, x, y) in tot.items()}}  # fmt: skip
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    a = parse(argv)
    t0 = time.time()
    out = Path(a.out)
    U.check_out(out)
    U.setup(a.device, a.threads)
    import torch

    torch.manual_seed(a.seed)
    base = Path(a.base).resolve()
    policy, tok, meta = U.load_base(base, a.device)
    ref, _, _ = U.load_base(base, a.device)
    ref.requires_grad_(False)
    ref.eval()
    ctx = U.make_ctx(a, tok)
    data, rep_data = U.load_proofs(a.data, tok)
    held, rep_held = U.load_pairs(a.heldout, tok)
    assert data and held, "no usable training proofs or held-out pairs"
    kl_proofs, rep_kl = U.load_proofs(a.kl_states, tok, a.kl_n) if a.kl_states else ([], None)
    rooms = [[max(0, U.MAXN - U.max_name([r], tok.ref0)) for r in rows] for rows in data]
    ntok_all = sum(len(r.ids) - r.plen for rows in data for r in rows)
    print(f"lh_sft: {len(data)} proofs ({rep_data['unreadable']} unreadable), "
          f"{sum(len(x) for x in data)} steps, {ntok_all} action tokens; {len(held)} held-out pairs; "
          f"{len(kl_proofs)} KL proofs", flush=True)  # fmt: skip
    ev = U.Evaluator(held, kl_proofs, ref, ctx, a.beta)
    del ref  # the reference quantities are computed; SFT never needs the base again
    print(f"lh_sft: base held-out accuracy (absolute) {ev.base_acc_abs:.4f}", flush=True)
    thresholds, steps, match_info = U.snap_plan(a)
    info = {"trainer": "sft", "base": str(base), "lr": a.lr, "batch": a.batch, "seed": a.seed,
            "data": str(Path(a.data).resolve()), "heldout": str(Path(a.heldout).resolve())}  # fmt: skip
    snaps = U.Snapshots(
        out, meta, base, info, thresholds=thresholds, steps=steps, metric=a.snap_metric
    )
    rng = random.Random(a.seed + 1)
    perm: list[int] = []
    fn = U.logp_fn(policy)

    def step_fn(step: int) -> dict[str, Any]:
        nonlocal perm
        if len(perm) < a.batch:
            perm = list(range(len(data)))
            rng.shuffle(perm)
        idx = [perm.pop() for _ in range(min(a.batch, len(perm)))]
        rows = [r for i in idx for r in data[i]]
        shifts = None if a.no_shift else [rng.randint(0, s) for i in idx for s in rooms[i]]
        ntok = sum(len(r.ids) - r.plen for r in rows)
        loss, _ = U.objective_backward(
            fn, rows, shifts, lambda o: -o.sum() / ntok, ctx, linear=-1 / ntok
        )
        return {"loss": loss, "rows": len(rows), "action_tokens": ntok,
                "tokens": sum(len(r.ids) for r in rows)}  # fmt: skip

    last = U.train_loop(a, policy, step_fn, ev, snaps, out)
    doc = U.done_doc(a, "sft", base, ev, snaps, last, t0, {
        "data": rep_data, "data_habits": data_habits(a.data), "pairs_heldout": rep_held, "kl_states": rep_kl,
        "action_tokens": ntok_all, "match": match_info,
        "loss": "cross-entropy per action token (Stage-1's main term; no MTP term)",
    })  # fmt: skip
    U.write_json(out / "done.json", doc)
    print(f"lh_sft: done in {doc['secs']:.0f} s; snapshots "
          f"{ {k: (v or {}).get('step') for k, v in snaps.done.items()} }", flush=True)  # fmt: skip
    return doc


if __name__ == "__main__":
    main()
