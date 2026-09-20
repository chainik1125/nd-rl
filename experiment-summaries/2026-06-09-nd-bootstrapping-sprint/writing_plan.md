# Writing plan (drafted 23:10, executed ~04:30-07:35)

## Narrative skeleton (provisional — update with tonight's numbers)

1. **The cliff is real and data does not move it.** From-scratch 3.3M transformer,
   87.6% greedy on held-out iid theorems; falls past L_train; 2x data and
   L_train sweep show boundary tracks the cutoff. [fig1]
2. **The cliff decomposes into two barriers.** Failure forensics: most OOD
   failures are invalid rule steps (structural), but a distinct mode — line-index
   tokens N9+ never seen in training — makes >8-line proofs *unwritable*
   (the novel-token wall). Offset ablation (random index shifts, same proofs)
   removes the wall; its round-0 OOD curve quantifies how much of the cliff was
   token-mechanical vs structural. [fig2 or table]
3. **Below the wall, verifier self-training sharpens but does not expand.**
   Chain-1: 313/1200 band theorems discovered round 1 — every single proof
   <=8 lines; transfer pass@1/8 flat; textbook flat-to-down; discovery
   accumulation ~matches the never-retrained static control. [fig3]
4. **With the wall removed, does RL cross the structural barrier?** Chain-2
   verdict: {PENDING}. [fig4]

## Checks to run before writing each claim

- C1: cliff curves all 4 conditions, CIs, eval_local vs eval_ood split.
- C2: fail-reason table round-0; L8off round-0 OOD vs L8 round-0 OOD.
- C3: chain-1 history + static accum overlay; discovered-length histograms;
  forgetting check (iid per round); pass@64 round5 vs round0.
- C4: chain-2 history; discovered-length histogram per round (>8-line count
  is THE number); example long discovered proof, pretty-printed.

## Red-team protocol (~05:30)

Spawn 2 agents on summary.md:
- RED: "skeptical senior mech-interp reviewer; attack unsupported claims,
  alternative explanations, unclear figures, overclaiming, inconsistent
  numbers; numbered findings with severity; do not rewrite."
- FIGURE-NAIVETE: show each figure PNG alone, ask "what does this show?" —
  if the answer misses the point, the figure fails the no-context test.
Then revise (blue team = me), repeat once if time.

## Honesty checklist (from sprint advice)

- Negative results stated as negative, mechanisms offered, no hype.
- Pre-registered predictions table: which held, which didn't.
- Cherry-picking flags: textbook proof examples are existence proofs (fine);
  all rates on full fixed eval sets.
- Limitations: single seed per condition; n_lines = generating-proof length
  (upper bound on difficulty); expert iteration not PPO; 4-atom propositional
  fragment; offset models have a soft start-index ceiling (start s limits
  proofs to 64-s+1 lines).
