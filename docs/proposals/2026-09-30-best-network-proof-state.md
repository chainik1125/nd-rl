---
written_on: 2026-09-30
written_by: agent:claude
status: approved by Dan 2026-09-30 ("I'd like to run this experiment today"); run `best-state`
---

# Proposal 17: Robbie's network and pretraining recipe with our proof-state interface, at cap 6 and cap 12

## Why

The group's two biggest gains come from different places, so they may add.
- **Robbie's combined model** (factorial of 2026-09-29, `robbie-experiments`, `factorial_20260929`): most of its
  gain is pretraining, +408 on his dev metric. That is a 6 × 384 network, about 10M parameters, with Peri-LN,
  ALiBi / NoPE heads, Muon, and multi-token prediction.
- **Ours:** the proof-state interface plus cap-12 training proofs.

On the 72 textbook problems (pass@256, `textbook72`, not yet reviewed), our SN-cap12 T1 solves 36–38 per seed
against his 30–32. Our cap-6 proof-state models solve only 14–22, so our edge needs the longer training proofs.
Neither pipeline has the other's strongest piece.

## Question

With the proof-state interface and our state RL held fixed, does Robbie's network and pretraining recipe beat our
3.2M recipe, at cap 6 and at cap 12? Does the cap-12 advantage survive, grow, or shrink on the better network?

## Design (2 × 2; two cells are inherited)

| | cap-6 data (control set) | cap-12 data (K12) |
|---|---|---|
| our 3.2M recipe | SN-v2 cap-6, s0 / s1 (inherited) | SN-cap12, s0–s3 (inherited) |
| **Robbie's recipe (new)** | **3 seeds** | **3 seeds** |

- **New cells:** Stage-1 in the proof-state format (`lean_staten`), using Robbie's architecture and optimizer
  ported into the state trainer, on the same data as the inherited cells. Then the same T1 ladder as
  `state-cap12`: 8 rounds, k 32, T 0.8, replay from the cell's own data.
- **Read-outs, frozen and T1:**
  - `textbook72` at pass@256, the head-to-head with Robbie;
  - Robbie's dev metric (dev half of the ladder transfer pool, 64 samples, solved at L_true ≥ 7), and his
    holdout250 at pass@k if the pool is on his branch;
  - the long pools (`rr600` 13–16 and `transfer_long2`);
  - held-out greedy.
- **Compute is recorded, not matched.** This compares recipes, and the bigger network costs more per sample and
  per step. Every arm reports GPU-seconds and tokens, as `compute-record` provides.

## Timing and cost

- **Today (a few hours):** port the recipe; 6 Stage-1 models (Robbie's budget is 300 s each; the run pilots
  whether the state format needs longer); frozen read-outs. That answers the pretrained half today.
- **Probably tomorrow:** 6 T1 ladders on parallel pods. Our 3.2M state ladders took about 7 h each, and the 3×
  larger network samples more slowly. The RL half lands roughly 12–20 h after the Stage-1 models.
- **Cost:** ≈ $45–55 of pods (budget $60), and the balance stays well above the $100 floor. Claude time is one
  executor session plus review.

## What would change our plans

- **Robbie's recipe wins at cap 12:** it becomes the Stage-1 standard for state models, and the group's two
  pipelines merge into one.
- **It wins at cap 6 but not at cap 12:** the bigger network helps where data is the limit and adds nothing once
  longer proofs are in the data.
- **It loses in the state format:** a recipe tuned on whole-proof sequences does not transfer to (state, action)
  training. Rather than pick a side, the next step would be his autoresearch loop run on the state format.

## Out of scope (possible follow-up)

Leon's RL recipe (EI with verifier-guided sampling, 15 small hardest-first rounds, GEM loss) ported to the state
environment. The state environment already rejects invalid steps, which is what his guided sampling adds, so this
run holds our state RL fixed and changes only the network and pretraining.
