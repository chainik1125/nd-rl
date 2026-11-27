"""Arm DPO of SPEC2.md Stage 2 (PREREG Amendment 3: the decisive arm): tune a base toward the rater's form of one
habit, against the frozen base.

    python lh_dpo.py --base CKPT --pairs TRAIN.jsonl --heldout HELD.jsonl --out DIR
                     [--swap-random] [--swap-seed SEED (default --seed)] [--seed 0]
                     [--snap-steps 5,10,15,20,25,30,40,50,75,100,150,200,300,400] [--max-steps 400]
                     [--beta 0.1] [--lr 3e-5] [--batch 64] [--device cuda|cpu] [--kl-states FILE] [--kl-n 256]
                     [--eval-every 25] [--acc-snapshots [--snap-metric heldout_acc_abs|heldout_acc_rel]]

Objective (Rafailov et al. 2023), per pair (w = `preferred`, l = `other`), pi_ref = the base, frozen:
    loss = -log sigmoid(beta * [(log pi(w) - log pi_ref(w)) - (log pi(l) - log pi_ref(l))]),  mean over the batch,
log pi(proof) = sum over the proof's steps of log P(action tokens + <eos> | untagged state), teacher forced, T 1, the
positions Stage-1's loss trains on (lh_tune: encoding, scoring, the name offset). Rows the two members share (every
row before the first differing one: for reiteration twins the notes and step order) are dropped by
`lh_tune.split_common`; they cancel exactly in the loss and get no direct gradient. Batch = --batch pairs per step
(default 64), AdamW lr 3e-5 (PREREG Amendment 7 point 1; was the RL recipe's 1.5e-4), betas 0.9 / 0.95, no weight
decay, grad-norm clip 1.0, linear warmup over 10 steps, then constant; --max-steps steps (default 400).

Step-0 check (PREREG Amendment 7 point 7, audit T11). The first train.jsonl row carries `step0_loss`: the train loss
of step 1, computed before the optimizer moves anything, while the policy still equals the reference. It must be
ln 2 = 0.6931 within 0.001. Otherwise the run stops before the first update: the row is written with `stopped`,
`step0_failed.json` beside it, the message on stderr, exit status 4, no done.json. The policy's training forward and
the reference's no-grad forward pack the same rows into the same tensors (lh_tune.objective_backward), also under bf16
on cuda, so only a real mismatch (another model, dropout, a kernel that differs between grad and no-grad mode) moves it.

Snapshots (PREREG Amendment 7 point 1): `snap_step<k>.pt` at every step k of --snap-steps (default the grid above;
steps past --max-steps are dropped and listed in done.json `match`), and `final.pt` at --max-steps. `--match-steps
DIR` takes a companion run's snapshot steps instead (kept for old launchers; the grid is the same for every run).
`--acc-snapshots` (off) adds `snap_acc60/75/90.pt` at the first evaluation where --snap-metric reaches 60 / 75 / 90 %.

metrics.jsonl: one row at step 0, at every snapshot step, every --eval-every steps and at the end: the train loss,
relative margin and accuracy since the previous row, and the held-out preference accuracy in both definitions,
`heldout_acc_abs` (log pi(w) > log pi(l): biased against the longer member) and `heldout_acc_rel` (the margin relative
to the base > 0), `heldout_margin_rel`, `heldout_logratio_abs`, `heldout_loss`, and `kl` (KL(pi_ref || pi) per action
token on --kl-states; computed at every snapshot step, step 0 and every --eval-every steps). done.json `definitions`
spells each out.

Batches and the no-preference control (PREREG Amendment 7 point 3, audit T4 and T10). The swap set: within each
sampled side (the pair file's `sampled_side`), a random floor(n_side / 2) of the LOADED pairs (after unreadable and
identical pairs are dropped), drawn once from --swap-seed (default: --seed, so the two control seeds have different
orientations; the launcher passes it equal to --seed). Every run, rated or not, builds its batches from couples: each
epoch, within each side, the swap-set pairs and the other pairs are shuffled and zipped into couples (one of each);
the couples of all sides are shuffled together and batches take whole couples in order. So with an even --batch every
batch holds, within each sampled side, exactly as many swap-set pairs as others. `--swap-random` (the control) then
trades preferred and other in the swap-set pairs: exactly half of every batch is swapped within each sampled side, and
the same pairs stay swapped for the whole run (the labels are learnable). A rated run and its control of the same
seed see the same pairs in the same batches; only the orientation of half differs. Rounding:
  - a side with an odd number of pairs has one pair more outside the swap set; each epoch, one of them (a different
    one each epoch, by the shuffle) sits out, so every couple is whole;
  - an odd --batch splits one couple between consecutive batches: they alternate one extra swap-set pair and one extra
    other pair, in the same side, so every two consecutive batches are exactly balanced;
  - as before, the couples left over at the end of an epoch (fewer than --batch pairs) are not used that epoch.
train.jsonl rows of a --swap-random run carry `swapped_by_side` ({side: [swapped, pairs]} of the batch). The held-out
pairs are never swapped, so the control reads movement toward the real rater's form.

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
from collections.abc import Sequence
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
    ap.add_argument("--swap-seed", type=int, default=None,
                    help="seed of the swap set (and of the couples every run batches by); default --seed")  # fmt: skip
    a = ap.parse_args(argv)
    if a.swap_seed is None:
        a.swap_seed = a.seed
    return a


def swap_set(pairs: Sequence[U.Pair], swap_seed: int) -> set[int]:
    """Indices into the loaded pairs: within each sampled side, a seeded random floor(n_side / 2), drawn once."""
    rng = random.Random(swap_seed)
    out: set[int] = set()
    for side in sorted({p.side for p in pairs}):
        idx = [i for i, p in enumerate(pairs) if p.side == side]
        out.update(rng.sample(idx, len(idx) // 2))
    return out


class Couples:
    """The batch order (module docstring): whole couples (one swap-set pair, one other pair, same sampled side)."""

    def __init__(
        self, pairs: Sequence[U.Pair], swap: set[int], batch: int, rng: random.Random
    ) -> None:
        self.sides: dict[str, tuple[list[int], list[int]]] = {}
        for i, p in enumerate(pairs):
            self.sides.setdefault(p.side, ([], []))[0 if i in swap else 1].append(i)
        self.n_couples = sum(min(len(s), len(u)) for s, u in self.sides.values())
        if not self.n_couples:
            raise SystemExit(
                "lh_dpo: no couple to batch (every sampled side has fewer than 2 pairs)"
            )
        self.batch, self.rng = batch, rng
        self.queue: list[int] = []
        self.epochs = 0
        self.sat_out: list[int] = []  # the pairs that sit out the current epoch (odd sides)

    def _epoch(self) -> list[int]:
        units: list[tuple[int, int]] = []
        self.sat_out = []
        for side in sorted(self.sides):
            s, u = (list(x) for x in self.sides[side])
            self.rng.shuffle(s)
            self.rng.shuffle(u)
            units += list(zip(s, u, strict=False))
            self.sat_out += s[len(u) :] + u[len(s) :]
        self.rng.shuffle(units)
        self.epochs += 1
        return [i for c in units for i in c]

    def next(self) -> list[int]:
        if len(self.queue) < self.batch:
            self.queue = self._epoch()
        take = min(self.batch, len(self.queue))
        out, self.queue = self.queue[:take], self.queue[take:]
        return out


def step0_check(st: dict[str, Any]) -> str | None:
    """PREREG Amendment 7 point 7: the first step's loss, before any update, is ln 2 within 0.001. Adds the
    step-0 fields to the first train.jsonl row; returns the stop message or None."""
    loss = float(st["loss"])
    st["step0_loss"] = loss
    st["step0_ok"] = abs(loss - U.STEP0_LOSS) <= U.STEP0_TOL
    if st["step0_ok"]:
        return None
    return (f"lh_dpo: STOPPED before the first update (PREREG Amendment 7 point 7): the step-0 train loss is "
            f"{loss:.6f}, not ln 2 = {U.STEP0_LOSS:.4f} within {U.STEP0_TOL} (largest |relative margin| in the batch "
            f"{st.get('step0_margin_absmax', float('nan')):.4g} nats), so the policy's training forward does not "
            f"equal the reference's; no weight was changed")  # fmt: skip


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
    train, rep_train = U.load_pairs(a.pairs, tok)
    held, rep_held = U.load_pairs(a.heldout, tok)
    assert train and held, "no usable train or held-out pairs"
    swap = swap_set(train, a.swap_seed)  # over the loaded pairs (audit T10)
    if a.swap_random:
        train = U.swap_pairs(train, swap)
    by_side = {s: {"pairs": sum(p.side == s for p in train), "swap_set": sum(train[i].side == s for i in swap)}
               for s in sorted({p.side for p in train})}  # fmt: skip
    swapped = sum(p.swapped for p in train)
    kl_proofs, rep_kl = U.load_proofs(a.kl_states, tok, a.kl_n) if a.kl_states else ([], None)
    overlap = len({p.prompt for p in train} & {p.prompt for p in held})
    print(f"lh_dpo: {len(train)} train pairs ({swapped} swapped; by side {by_side}), {len(held)} held-out, "
          f"{len(kl_proofs)} KL proofs, theorems in both {overlap}; rows shared within a pair "
          f"{rep_train['rows_shared_share']:.1%}", flush=True)  # fmt: skip
    ev = U.Evaluator(held, kl_proofs, ref, ctx, a.beta)
    print(f"lh_dpo: base held-out accuracy (absolute) {ev.base_acc_abs:.4f}", flush=True)
    thresholds, steps, match_info = U.snap_plan(a)
    info = {"trainer": "dpo", "base": str(base), "beta": a.beta, "lr": a.lr, "batch": a.batch,
            "swap_random": a.swap_random, "swap_seed": a.swap_seed, "seed": a.seed,
            "pairs": str(Path(a.pairs).resolve()), "heldout": str(Path(a.heldout).resolve())}  # fmt: skip
    snaps = U.Snapshots(
        out, meta, base, info, thresholds=thresholds, steps=steps, metric=a.snap_metric
    )
    rng = random.Random(a.seed + 1)
    batches = Couples(train, swap, a.batch, rng)
    pol_fn = U.logp_fn(policy)
    ref_fn = U.logp_fn(ref)

    def step_fn(step: int) -> dict[str, Any]:
        batch = [train[i] for i in batches.next()]
        shifts = None if a.no_shift else [rng.randint(0, p.room) for p in batch]
        rows, sh, owner, sign = U.flatten_pairs(batch, shifts)
        d_ref = U.pair_sums(U.apply_rows(ref_fn, rows, sh, ctx), owner, sign, len(batch), a.device)

        def loss_fn(o: Any) -> Any:
            d = U.pair_sums(o, owner, sign, len(batch), a.device)
            return -torch.nn.functional.logsigmoid(a.beta * (d - d_ref)).mean()

        loss, o = U.objective_backward(pol_fn, rows, sh, loss_fn, ctx)
        m = U.pair_sums(o, owner, sign, len(batch), a.device) - d_ref
        st = {"loss": loss, "train_margin_rel": float(m.mean()), "train_acc_rel": float((m > 0).float().mean()),
              "rows": len(rows), "tokens": sum(len(r.ids) for r in rows), "pairs": len(batch)}  # fmt: skip
        if a.swap_random:
            st["swapped_by_side"] = {s: [sum(p.swapped for p in batch if p.side == s), sum(p.side == s for p in batch)]
                                     for s in sorted({p.side for p in batch})}  # fmt: skip
        if step == 1:
            st["step0_margin_absmax"] = float(m.abs().max())
        return st

    first: dict[str, Any] = {}

    def check(st: dict[str, Any]) -> str | None:
        msg = step0_check(st)
        first.update(
            {k: st[k] for k in ("step0_loss", "step0_ok", "step0_margin_absmax") if k in st}
        )
        return msg

    last = U.train_loop(a, policy, step_fn, ev, snaps, out, first_check=check)
    keys = ",".join(sorted(train[i].key for i in swap))
    sha = hashlib.sha256(keys.encode()).hexdigest()
    doc = U.done_doc(a, "dpo", base, ev, snaps, last, t0, {
        "pairs_train": rep_train, "pairs_heldout": rep_held, "kl_states": rep_kl,
        "theorems_in_train_and_heldout": overlap, "swap_random": a.swap_random, "swap_seed": a.swap_seed,
        "swapped": swapped, "swap_by_side": by_side, "swap_set_sha256": sha,
        "swapped_sha256": sha if a.swap_random else None,
        "batching": {"rule": "couples (one swap-set pair, one other, same sampled side), whole couples per batch",
                     "couples_per_epoch": batches.n_couples, "epochs_started": batches.epochs,
                     "sat_out_per_epoch": len(batches.sat_out)},
        "step0": {**first, "target": U.STEP0_LOSS, "tol": U.STEP0_TOL},
        "match": match_info, "ref_frozen": U.ref_frozen(ref, base),
    })  # fmt: skip
    U.write_json(out / "done.json", doc)
    print(f"lh_dpo: done in {doc['secs']:.0f} s; snapshots "
          f"{ {k: (v or {}).get('step') for k, v in snaps.done.items()} }", flush=True)  # fmt: skip
    return doc


if __name__ == "__main__":
    main()
