# Run brief: coe-dose01 — how much RL brings σ back when pretraining holds only 0.1 % of it?

Run id `coe-dose01`. Repository `~/work/coe-dose01` (branch `dan_coe-dose01` from the fork's `origin/dan_create-or-elicit`,
made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Authorised by Dan 2026-10-31 (proposal 33).

- **Budget:** **$45 / 90 pod-hours.** Dollars are binding. Priority 1 must fit first; priority 2 only if at least $10
  remains after it.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 72 h.

## The question (Dan, 2026-10-31)

For the midterm write-up, Dan wants a graph of how much RL is needed to reproduce a capability. That graph now exists,
built from create-or-elicit's ladders: nd-rl `docs/writeups/2026-10-31-rl-needed-to-reproduce-sigma/`.

**The capability:** solving the 106 strict σ-covered held-out members.
**The bar:** the full-exposure bases' median before any RL update, 14 of 106.

| pretraining | RL ÷ pretraining compute needed to reach the bar |
|---|---|
| full exposure (σ in 10.9 % of K12) | none |
| 1 % σ dose | 0.39–0.73× (3 of 3 seeds) |
| 0.1 % σ dose | **this run** |
| knockout (σ removed) | not reached by 5.6–9.5× (0 of 6) |

Two dose levels and a lower bound cannot tell a threshold from a smooth curve. A 0.1 % arm can. Read first: the figure's
README, the create-or-elicit summary (nd-rl `experiment-summaries/2026-10-12-create-or-elicit-…`), and its
`next_steps.md` items 3 and 6.

## Design

1. **Priority 1: the 0.1 % dose arm, seeds s0–s2.** Use exactly the 1 % arm's protocol, changing only the pretraining
   set (`data/coe/train_k12dose01.jsonl.gz`; also in the bucket under `create-or-elicit/data/coe/`; a read-only copy is
   in `~/work/create-or-elicit/data/coe/`).
   - Stage-1: the best-cap12 recipe, 1,200 s on an A40.
   - Then T1 EI for 16 rounds: k 32 per target, the same 4,495 targets, 20,000 replay records per round from the arm's
     own pretraining set, Lean alone, and the same per-round held-out reads on the 832 tracked theorems.
   - Diff your `args.json` against `artifacts/coe/ladder/la_dose1_s*/args.json` and report any difference.
   - Write the round files and `artifacts/coe/rounds/la_dose01_s{0,1,2}.json` with `coe/coe_rounds.py`, in the same
     format as `la_dose1_s*`.
   - If budget allows, also take the r16 plain read at k 256 that the 1 % arm has, so the family-acquisition verdict is
     comparable.
2. **Priority 2 (only if ≥ $10 remains): knockout s1, rounds 25–32.** Use create-or-elicit's Addendum 6 procedure,
   starting from the r24 checkpoint (look in the bucket). If the checkpoint is missing, skip this and say so.
3. **The figure.**
   - Copy `extract_rl_curve.py` and `make_figure.py` from the nd-rl folder above into `rl_needed/` on this branch.
   - Add the new ladders to the extraction (`arm: "dose01"`; the plotting script already has its colour and label), and
     extend knockout s1 if priority 2 ran.
   - Regenerate `rl_curve.json` and `rl_needed.{png,pdf}`.
   - Report each new ladder's cumulative RL / pretraining FLOPs at r16.
   - `extract_rl_curve.py` imports `review_rcs/ourruns.py` from `~/work/rl-compute-share` (fork branch
     `dan_rl-compute-share`). Check that its definitions give the same ratios on the existing ladders before relying
     on them for the new ones.

## Pre-registration

Commit `preregistration/coe-dose01.md` before the first pod. Write expectations for:
- each 0.1 % seed's held-out count at r1 and r16;
- the RL ÷ pretraining compute at which each seed first reaches 14 of 106, or "not reached by r16";
- for priority 2, whether knockout s1 reaches 14 by r32, and its count there.

Also say in advance what pattern would read as a threshold (0.1 % behaves like the knockout) and what would read as a
graded response (it crosses, but later than the 1 % arm).

## Deliverables

- `run_coe_dose01.md` (≤ 400 words + the updated figure); `numbers.md` § coe-dose01; `log.md`; compute rows.
- A plain-language summary of ≤ 600 words.
- `rl_needed/`: the scripts, `rl_curve.json` and the figure, which the librarian copies into nd-rl
  `docs/writeups/2026-10-31-rl-needed-to-reproduce-sigma/`, updating its README's table.
- A `STATUS.md` line `COE-DOSE01 DONE <UTC>`; then upload to `hf://buckets/dan-pandori/nd-rl/coe-dose01/`, delete
  pods, and `touch ~/runs/coe-dose01/executor.done`.

**For the reviewer:**
- Recompute each crossing from the round files.
- Recompute the FLOP ratios with `review_rcs/ourruns.py`'s definitions.
- Diff the ladders' `args.json` against the 1 % arm's.
- Lean-check a sample of the held-out proofs that count toward the bar.

**Rules:**
- Do not contact anyone.
- textbook72 is scoring only, and no textbook72 theorem is printed.
- At most 4 concurrent processes on the VPS.
- Never run `test_run_once.sh`.
- Keep the RunPod balance above $35.
- Leave `~/work/create-or-elicit` unchanged; copy from it.
