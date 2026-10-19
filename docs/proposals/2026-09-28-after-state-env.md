---
written_on: 2026-09-28
written_by: agent:claude
status: proposed — held until `state-env` reports (Dan, 2026-09-28: "Let's wait until we see how the RL
  environment change turns out before kicking off more experiments")
---

# Proposal 14: what to run after `state-env`, by outcome

Running now: **`state-env`** (proposal 13; agent2; $30) — does giving the policy the Lean-style proof
state move the length wall? — and **`support-followups`** (VPS1; $15; approved before the hold) —
per-step surprisal of the 29 support-expansion survivors, a seed-1 column, 10⁷ base attempts on the
six longest survivors, and a ≈ 8× bigger base on the forward crux.

Nothing below launches until `state-env` has a reviewed result. Costs are estimates at this project's
measured rates (3.2 M model: one 6,000-step Stage-1 ≈ 21 min on an A40; ≈ 3.6 M samples per pod-hour
at the fast sampler settings); each brief re-measures.

## If the state moves the wall (arm S: T1 `L*` ≥ 13 with ≥ 5 solved at `L_true` ≥ 13, both seeds)

1. **Search: AlphaProof's other half.** The step-level environment is what tree search needs. Add a
   value head predicting steps-to-go (AlphaProof's target: −1 per step, the longest open branch) and a
   PUCT search with one AND node per `Or.elim`, against best-of-N sampling at equal model calls. Question:
   does search move `L*` further than sampling, and is the value head learnable at 3.2 M? Engineering
   heavy; ≈ $15–25 compute.
2. **Support curves on the state model.** `support-curves`' 383-theorem protocol with the state-policy
   base and its EI descendant. Are the 29 survivors base-reachable once the base sees the state — was
   the expansion about tracking state? Does the state model's EI expand support further? ≈ $10.
3. **GRPO in the state environment**, against EI at equal samples. Step-level credit is the natural home
   for policy gradient, and round 2 found GRPO a faster, bolder eliciter than EI (at the cost of the
   pretrained distribution without a KL term). ≈ $10.
4. **Textbook schemata with the state model** — the wall that survived every data manipulation (most De
   Morgan and distribution forms, excluded middle and Peirce stay at ≤ 2 of 40 in every arm). Free if it
   is read off 1–3.

## If it does not (arm S stays at `L*` ≤ 12 with 0 solved at `L_true` ≥ 13)

5. **Proposal 12's cuts to locate the wall.** Capacity: `support-followups` part C gives the first read;
   the full test is Stage-1 + ladder EI on the larger model (≈ $15–20). Search: EI continued to 24 rounds
   (proposal 7, queued since 2026-09-19; ≈ $5 per seed) and verifier-guided search at evaluation.
   Data difficulty: pretraining with theorems verified hard by `minlen.py` (`L_true` 10–13) at small
   frequencies (≈ $15).
6. **GRPO's support.** Does GRPO expand support where EI does not? The `support-curves` protocol on a GRPO
   descendant of the same base (≈ $10).

## Either way

7. **Review the unreviewed GRPO replication** (`dan_run4_grpo`, Sept 17–18; its session died in the auth
   failure and the results were salvaged, with an unfilled `XXX` in the write-up). It claims GRPO ignited
   depth-3 on 30 of 30 arms, including draws with 0 depth-3 samples in 600,000 base attempts, where paired
   EI ignited 6 of 10. If it holds it is the strongest "beyond the base" evidence before `support-curves`.
   No GPU: Claude usage only.
8. **Checkpoint averaging as the Stage-1 standard.** `stage1-dynamics` found the depth-3 "mode" is an
   oscillation within each run (all 12 long runs crossed the high/low cut 1–10 times), so the project's
   largest noise source is where a run happens to stop. Average the last N checkpoints (or pick by
   per-length validation loss, which ranks runs at ρ −0.905) across the eight saved `stage1-dynamics`
   trajectories and re-measure the seed-to-seed spread. No training — held-out evaluations of saved
   checkpoints only, ≈ $1–2. If the spread collapses, every later comparison needs fewer seeds.
9. **Engineering, no experiment.** Raise the loop scripts' `--batch` defaults once `lean-judge`'s edits
   have merged (the policy already asks for the largest batch that fits); and a short look at Stage-1
   throughput (`torch.compile`, larger batch — the sampler work found this model launch-bound).

## Order when the hold lifts

7 and 8 first in either branch — cheap and they change how everything else is read. Then 1–2 or 5–6 by
outcome.
