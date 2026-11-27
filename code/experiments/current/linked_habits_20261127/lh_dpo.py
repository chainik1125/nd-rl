"""Arm DPO of SPEC2.md Stage 2 (PREREG Amendment 3: the decisive arm): tune a base toward the rater's form of one
habit, against the frozen base.

    python lh_dpo.py --base CKPT --pairs TRAIN.jsonl --heldout HELD.jsonl --out DIR
                     [--swap-random] [--swap-seed 0] [--seed 0]
                     [--snap-steps 25,50,100,150,200,300,400,600,800,1000,1500,2000] [--max-steps 2000]
                     [--beta 0.1] [--lr 1.5e-4] [--batch 64] [--device cuda|cpu] [--kl-states FILE] [--kl-n 256]
                     [--eval-every 25] [--acc-snapshots [--snap-metric heldout_acc_abs|heldout_acc_rel]]

Objective (Rafailov et al. 2023), per pair (w = `preferred`, l = `other`), pi_ref = the base, frozen:
    loss = -log sigmoid(beta * [(log pi(w) - log pi_ref(w)) - (log pi(l) - log pi_ref(l))]),  mean over the batch,
log pi(proof) = sum over the proof's steps of log P(action tokens + <eos> | untagged state), teacher forced, T 1, the
positions Stage-1's loss trains on (lh_tune: encoding, scoring, the name offset). Rows the two members share (every
row before the first differing one: for reiteration twins the notes and step order) are dropped by
`lh_tune.split_common`; they cancel exactly in the loss and get no direct gradient. Batch = --batch pairs per step
(default 64), AdamW lr 1.5e-4 (the RL recipe's), betas 0.9 / 0.95, no weight decay, grad-norm clip 1.0, linear warmup
over 10 steps, then constant; --max-steps steps (default 2,000).

Snapshots (PREREG Amendment 3 point 3): `snap_step<k>.pt` at every step k of --snap-steps (default the grid above;
steps past --max-steps are dropped and listed in done.json `match`), and `final.pt` at --max-steps. `--match-steps
DIR` takes a companion run's snapshot steps instead (kept for old launchers; the grid is the same for every run).
`--acc-snapshots` (off) adds `snap_acc60/75/90.pt` at the first evaluation where --snap-metric reaches 60 / 75 / 90 %.

metrics.jsonl: one row at step 0, at every snapshot step, every --eval-every steps and at the end: the train loss,
relative margin and accuracy since the previous row, and the held-out preference accuracy in both definitions,
`heldout_acc_abs` (log pi(w) > log pi(l): biased against the longer member) and `heldout_acc_rel` (the margin relative
to the base > 0), `heldout_margin_rel`, `heldout_logratio_abs`, `heldout_loss`, and `kl` (KL(pi_ref || pi) per action
token on --kl-states; computed at every snapshot step, step 0 and every --eval-every steps). done.json `definitions`
spells each out.

Seeds. `--seed` sets the order of the pairs and the name offsets (lh_tune.SEED_AFFECTS). `--swap-random` (the
no-preference control): preferred and other trade places in a random half of the TRAIN pairs (exactly floor(n/2)),
drawn with `--swap-seed` (default 0), not with --seed, so runs of seeds 0 and 1 share one orientation (done.json
`swapped_sha256`). The held-out pairs are never swapped, so the control reads movement toward the real rater's form.

Written in DIR: the snapshots (ordinary checkpoints: cfg, state, tok_mode, extra = the base's + `tuning`; each with a
.tag.json copy of the base's), metrics.jsonl, train.jsonl (one row per step), done.json. A folder with done.json or
metrics.jsonl is refused.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import hashlib
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
    ap.add_argument("--pairs", required=True, help="train pair file")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--swap-random", action="store_true", help="the no-preference control")
    ap.add_argument(
        "--swap-seed", type=int, default=0, help="the swap orientation's seed (not --seed)"
    )
    return ap.parse_args(argv)


def swap_set(n: int, swap_seed: int) -> set[int]:
    """The seeded random half of n pairs (indices in file order) whose members trade places."""
    return set(random.Random(swap_seed).sample(range(n), n // 2))


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
    n_train = len(U.read_jsonl(a.pairs))
    swap = swap_set(n_train, a.swap_seed) if a.swap_random else set()
    train, rep_train = U.load_pairs(a.pairs, tok, swap)
    held, rep_held = U.load_pairs(a.heldout, tok)
    assert train and held, "no usable train or held-out pairs"
    kl_proofs, rep_kl = U.load_proofs(a.kl_states, tok, a.kl_n) if a.kl_states else ([], None)
    overlap = len({p.prompt for p in train} & {p.prompt for p in held})
    print(f"lh_dpo: {len(train)} train pairs ({rep_train['swapped']} swapped), {len(held)} held-out, "
          f"{len(kl_proofs)} KL proofs, theorems in both {overlap}; rows shared within a pair "
          f"{rep_train['rows_shared_share']:.1%}", flush=True)  # fmt: skip
    ev = U.Evaluator(held, kl_proofs, ref, ctx, a.beta)
    print(f"lh_dpo: base held-out accuracy (absolute) {ev.base_acc_abs:.4f}", flush=True)
    thresholds, steps, match_info = U.snap_plan(a)
    info = {"trainer": "dpo", "base": str(base), "beta": a.beta, "lr": a.lr, "batch": a.batch,
            "swap_random": a.swap_random, "swap_seed": a.swap_seed if a.swap_random else None, "seed": a.seed,
            "pairs": str(Path(a.pairs).resolve()), "heldout": str(Path(a.heldout).resolve())}  # fmt: skip
    snaps = U.Snapshots(
        out, meta, base, info, thresholds=thresholds, steps=steps, metric=a.snap_metric
    )
    rng = random.Random(a.seed + 1)
    perm: list[int] = []
    pol_fn = U.logp_fn(policy)
    ref_fn = U.logp_fn(ref)

    def step_fn(step: int) -> dict[str, Any]:
        nonlocal perm
        if len(perm) < a.batch:
            perm = list(range(len(train)))
            rng.shuffle(perm)
        batch = [train[perm.pop()] for _ in range(min(a.batch, len(perm)))]
        shifts = None if a.no_shift else [rng.randint(0, p.room) for p in batch]
        rows, sh, owner, sign = U.flatten_pairs(batch, shifts)
        d_ref = U.pair_sums(U.apply_rows(ref_fn, rows, sh, ctx), owner, sign, len(batch), a.device)

        def loss_fn(o: Any) -> Any:
            d = U.pair_sums(o, owner, sign, len(batch), a.device)
            return -torch.nn.functional.logsigmoid(a.beta * (d - d_ref)).mean()

        loss, o = U.objective_backward(pol_fn, rows, sh, loss_fn, ctx)
        m = U.pair_sums(o, owner, sign, len(batch), a.device) - d_ref
        return {"loss": loss, "train_margin_rel": float(m.mean()), "train_acc_rel": float((m > 0).float().mean()),
                "rows": len(rows), "tokens": sum(len(r.ids) for r in rows)}  # fmt: skip

    last = U.train_loop(a, policy, step_fn, ev, snaps, out)
    swapped = sorted(swap)
    doc = U.done_doc(a, "dpo", base, ev, snaps, last, t0, {
        "pairs_train": rep_train, "pairs_heldout": rep_held, "kl_states": rep_kl,
        "theorems_in_train_and_heldout": overlap, "swap_random": a.swap_random,
        "swap_seed": a.swap_seed if a.swap_random else None, "swapped": rep_train["swapped"],
        "swapped_sha256": hashlib.sha256(",".join(map(str, swapped)).encode()).hexdigest() if swap else None,
        "match": match_info, "ref_frozen": U.ref_frozen(ref, base),
    })  # fmt: skip
    U.write_json(out / "done.json", doc)
    print(f"lh_dpo: done in {doc['secs']:.0f} s; snapshots "
          f"{ {k: (v or {}).get('step') for k, v in snaps.done.items()} }", flush=True)  # fmt: skip
    return doc


if __name__ == "__main__":
    main()
