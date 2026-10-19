---
written_on: 2026-09-19
written_by: agent:claude
---

# Proposal 7: long-horizon continuation of the ladder's best arms

Requested by Dan after reading the ladder's per-round curves
(`experiment-summaries/2026-09-18-ladder-A-rl-technique-ladder/figures/ladder_saturation.png`).
Queued for after the pause; not necessarily the first run.

## Question

Where does expert iteration on the cap-6 model plateau, in count and in true length, if it is
simply run longer? Ladder Phase A stopped at 8 rounds with the transfer count still rising
(T1 s0: 452 → 526 → 569 → 612 over rounds 5–8), the count at `L_true` ≥ 10 rising almost
linearly (2, 12, 23, 27, 35), `L*` fixed at 10 since round 5, and ≥ 11 at 0–1. New accepted
target proofs per round decay geometrically (377, 813, 680, 417, 281, 213, 161, 172).

## Pre-registered predictions (falsifiable)

- Transfer solves for T1 plateau at **700–720 of 2,285** by round ~16 (ratio ≈ 0.7 per round on
  the increments); T6 (two siblings sharing finds) plateaus higher, 740–780.
- Solves at `L_true` ≥ 10 keep rising to **50–70 of 451** by round 16, then flatten.
- **`L*` stays at 10**; solves at `L_true` 11 stay ≤ 3 of 99 and 12–14 stay 0 through round 32.
  The first of these predictions is the one whose failure would matter: any arm reaching ≥ 5
  distinct theorems at 11 falsifies "the 10 → 11 step needs a pool or injection change".
- Held-out greedy stays ≥ 0.94; per-round fine-tune loss keeps drifting down without a floor.

## Design (suggestion)

Continue T1 (both seeds) and T6 (both sibling pairs) from their round-8 checkpoints with the
identical per-round protocol (k = 32 on the 4,495-target pool, 600 fine-tune steps, 20k retained
records, T = 0.8) to **round 32**, checkpointing every round; the frozen control continued to the
same cumulative attempt count. Every four rounds: transfer solves by `L_true` bin, `L*`, base
reachability of new ≥ 10 proofs, held-out greedy, new proofs per round; fit the increment ratio
and report the extrapolated plateau against the observed one. Same reviewer protocol.
One added measurement: at rounds 8, 16, 24, 32, the model's per-sample success rate on the
99 `L_true`-11 theorems at pass@2,000 — is 11 getting closer (rate rising from 0) or not?

## Budget and stop rule

Two 3090 pods, ~24 h, ≈ $30–40. Stop early if three consecutive rounds add < 5 transfer solves
for every arm (plateau reached) or if `L*` reaches 11 in any arm (report and stop; a new brief
decides what to do with it).

## Deliverables

`run_longhorizon.md` (≤ 400 words + the four-panel curve extended to 32 rounds), the usual
`numbers.md`/`log.md` sections, bucket upload, review, librarian merge into `dan`.
