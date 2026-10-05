---
written_on: 2026-10-02
written_by: agent:claude
---

# Week in review: 25 September – 1 October 2026

**Summary:** This week we built a model organism that largely solves our benchmarks: an AlphaProof-style
proof-state interface plus Robbie's pretraining recipe solves 51–52 of 72 textbook problems (up from 36–38) and
saturates our long-proof pools. First analysis suggests RL mostly sharpens proofs pretraining already brings
within reach.

The librarian's run-by-run digest for the week is [`digests/2026-40.md`](../digests/2026-40.md).

*Corrected 2026-10-02 after the [claim audit](2026-10-03-claim-audit.md). Every headline count reproduces; three
claims below are reworded to what the evidence supports (marked †).*

## What we did

**Is RL creating or eliciting?**
- Expert iteration solves 29 theorems its base never solved in 400k attempts:
  [support-curves](../../experiment-summaries/2026-09-27-support-curves-support-expansion-fires/README.md).
- An 8× larger base model does not reach them:
  [support-followups](../../experiment-summaries/2026-09-28-support-followups-expansion-not-removed-by-8x-params/README.md).
- A proof-state base reaches 28 of the 29 with no RL:
  [support-state](../../experiment-summaries/2026-09-28-support-state-state-base-reaches-survivors/README.md).
- † The gain survives without environment-assigned names. The audit notes that "the state" is confounded with
  the step-by-step interface, and that the naming comparison has 2 seeds per arm:
  [state-readouts](../../experiment-summaries/2026-09-30-state-readouts-state-not-naming-reaches-survivors/README.md).
- GRPO acquired depth-3 nesting from zero on old token-format runs:
  [run4-grpo-review](../../experiment-summaries/2026-09-28-run4-grpo-review-grpo-ignites-from-zero/README.md).
- New proofs hinge on one very improbable step:
  [lit-measures](../../experiment-summaries/2026-09-30-lit-measures-one-bad-step-and-rerun-noise/README.md).
- How proofs become likely through pretraining and RL:
  [trajectory](../../experiment-summaries/2026-10-01-trajectory-eventual-proof-worst-step-climbs-in-pretraining-and-rl/README.md).
  At cap 6: [trajectory-cap6](../../experiment-summaries/2026-10-02-trajectory-cap6-eventual-proof-worst-step-climbs-mainly-in-rl/README.md).

**The proof-state environment and stronger models**
([proposal 13](../proposals/2026-09-28-state-conditioned-environment.md),
[proposal 17](../proposals/2026-09-30-best-network-proof-state.md)):
- [state-env](../../experiment-summaries/2026-09-28-state-env-state-lifts-below-the-wall/README.md) introduced it.
- [state-frontier](../../experiment-summaries/2026-09-28-state-frontier-state-passes-12-rarely-no-low-mode-in-12-seeds/README.md)
  and [state-cap12](../../experiment-summaries/2026-09-29-state-cap12-state-and-cap12-proofs-compound-on-long-pool/README.md)
  pushed the length frontier.
- [textbook72](../../experiment-summaries/2026-09-30-textbook72-sn-cap12-t1-solves-37-of-72/README.md) compared us
  with Robbie on his 72 textbook problems.
- [best-state](../../experiment-summaries/2026-09-30-best-state-robbies-recipe-wins-after-rl-level-frozen-at-cap12/README.md)
  combined his recipe with our interface.
- New long pools: [long-pool](../../experiment-summaries/2026-09-28-long-pool-length-frontier-pool-wall-at-13-was-the-pool/README.md)
  and [long-pool-2](../../experiment-summaries/2026-09-30-long-pool-2-pool-past-17-no-fall-off-detected/README.md).

**Literature-driven** ([review](../literature/2026-09-29-lit-review/REVIEW.md),
[proposal 16](../proposals/2026-09-29-literature-shortlist.md)):
- [frontier-supply](../../experiment-summaries/2026-09-30-frontier-supply-filter-works-gain-inside-mdd/README.md): no gain.
- [search-expert](../../experiment-summaries/2026-09-30-search-expert-best-first-ties-sampling-at-quarter-spend/README.md):
  a tie with sampling.
- [grpo-state](../../experiment-summaries/2026-09-30-grpo-state-code-phase-grpo-in-the-state-env/README.md): code
  built; the experiment has not run.

**Method:**
- Seed variance: [noise-floor](../../experiment-summaries/2026-09-25-noise-floor-measurement-floor/README.md),
  [stage1-dynamics](../../experiment-summaries/2026-09-27-stage1-dynamics-saturation-and-seed-variance/README.md),
  [ckpt-avg](../../experiment-summaries/2026-09-28-ckpt-avg-averaging-does-not-remove-modes/README.md).
- Lean alone now judges proofs:
  [lean-judge](../../experiment-summaries/2026-09-27-lean-judge-lean-alone-decides/README.md).
- Every run now records per-arm compute.

**Infrastructure** ([proposal 15](../proposals/2026-09-28-repo-and-method-improvements.md)):
- Faster training and RL rounds:
  [fast-stage1](../../experiment-summaries/2026-09-28-fast-stage1-equivalent-2x6-faster/README.md) (≈ 2.6×) and
  [lean-prefilter](../../experiment-summaries/2026-09-28-lean-prefilter-sound-filter-4x-round/README.md) (≈ 4× per round).
- Bookkeeping:
  [results-registry](../../experiment-summaries/2026-09-29-results-registry-ckpt-upload-and-results-table/README.md) and
  [compute-record](../../experiment-summaries/2026-09-30-compute-record-scripts-record-their-own-compute/README.md).
- Repo cleanup: the fork's clone went from about 4.8 GB to 606 MB
  ([repo-hygiene-2](../../experiment-summaries/2026-09-29-repo-hygiene-2-small-files-back-manifest-size-guard/README.md)).
- Orchestration fixes, listed in the [tools README](../../code/tools/orchestration/README.md): `runmsg`, the pod
  watchdog fix, fewer notifications, automatic disk cleanup, and a 2-minute orchestrator tick.
- [PR #19](https://github.com/chainik1125/nd-rl/pull/19): the Lean tokenizer and judge for main. It is a draft
  awaiting Dan's review.

## What seems important

1. **"RL created it" depends on the interface.** The 29 theorems that looked like RL creating capability were
   mostly reachable once the base model could see the proof state. A create-vs-elicit claim has to say which
   interface it was measured on.
2. **The pieces compound, and evaluation is now the bottleneck.** The proof state, longer training proofs and
   Robbie's recipe stack up. The best model solves nearly all of the long pools, so its frontier can no longer be
   located.
3. † **At cap 12, RL mostly completes proofs that pretraining brought close.** After pretraining, the theorems only
   RL solves had a worst step of about 1-in-500 (IQM), and the rest of the proof still carried 7–9 nats, so this is
   not literally "one step". At cap 6 the worst step started near 1-in-25,000 and climbed mainly in RL. RL lifted
   its own proofs more than the reference proofs: about 1.7× on other seeds' proofs, less than first stated. The
   unsolved theorems have several hard steps. This rests on 3 seeds per cap.
4. † **GPU nondeterminism may be a large share of our noise.** In one cell on one pod, identical-seed re-runs gave
   about half the seed-to-seed variance (ratio 0.54, 95 % CI 0.23–2.05). That is suggestive, not established.
   Experiments are now planned against measured noise margins, and many earlier two-seed "findings" do not survive
   them.
5. **Two cheap no-gain results are informative.** Frontier supply and best-first search did not help at our scale.
   That argues against targets or search width being what limits RL here.

## Next steps for the research agenda

The agenda: how much does RL against a verifier create new capability, versus elicit rare behaviour the model
already has, and what limits it?

1. **Run RL from earlier pretraining checkpoints** of the cap-12 model organism (`trajectory`'s checkpoints). If
   RL only solves theorems whose hardest step starts above some likelihood, that is quantitative evidence of
   elicitation. If it solves far-out-of-reach theorems, that is creation. (Proposal 19.)
2. **A richer domain.** Propositional logic may be too shallow for the creation question. Dan, 2026-10-02:
   longer propositional proofs are "not that interesting", and other group members are working on harder
   benchmarks.
3. **A compute-matched control for best-state.** Robbie's recipe used about 1.6–1.9× our GPU time.
4. **Is it the RL algorithm?** GRPO in the proof-state environment on the model organism, with support-expanding
   advantages, against EI from the same starting checkpoints.
5. **trajectory-cap6.** The weaker cap-6 model leaves more theorems for RL to solve, so it tests the
   "within reach" picture harder.
