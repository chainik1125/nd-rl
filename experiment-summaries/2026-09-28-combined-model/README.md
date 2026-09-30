---
written_on: 2026-09-29
written_by: agent:claude
draft: true
---

```claude
For context, see [last week's post](https://robbiewmthompson.com/blog/pretraining-ablations/).
```

# The combined model, and where its gains come from

## Summary

```claude
Everyone in the group owns one stage of the pipeline: Dan the data and its format, me the
pretraining, Leon the RL. This week I plugged the best version of each into one model and asked
where its gains come from.

The combined model solves **904 of 1,108** validation theorems of 7+ lines (64 samples each), in
the same 2,400 s of one GPU as every other run here. The most naive version of the same pipeline
solves **142**. On theorems no run ever touched, it proves **79%** at 64 samples (naive: 13%),
and **31 of 72 textbook problems** at 256 samples (naive: 9).
```

![Dev theorems solved, by pipeline](charts/by-cell.png)

```claude
Where the 762 extra theorems come from, averaging each piece's effect over the other two:

| piece | naive → best | average effect |
| --- | --- | --- |
| pretraining | autoresearch run 012's baseline → the autoresearch incumbent (186) | **+408** |
| RL | the harness's plain expert iteration → Leon's recipe + verifier-guided sampling | **+268** |
| data format | token format → Dan's Lean format | **+92** |

- **Pretraining matters most**, which I did not expect: I predicted the ordering RL > format >
  pretraining. On its own, better pretraining triples the naive pipeline (142 → 457).
- **The Lean format only pays with good pretraining.** With the naive pretraining it adds +50
  (with plain RL) or −30 (with Leon's RL); with the 186 pretraining it adds +225 and +122. This
  is the largest interaction in the table (+82). The 186 recipe was found by searching in the Lean
  format, which is part of why.
- **Pretraining and RL add up**: their interaction is +6.
- **Leon's RL buys coverage, not accuracy.** It lowers pass@1 on held-out theorems (44% → 30%) and
  raises pass@64 (59% → 79%). Our metric is pass@64, which rewards that.
- **The textbook set finally moves.** Last week none of 184 pretraining runs got past ~22 of 72.
  The combined model gets 31; it is the RL step that moves it (best pretraining + plain RL: 20).
```

<details class="aside">
<summary>How the effects are computed</summary>

```claude
A factor's effect is the mean of the four cells at its best level minus the mean of the four at
its naive level. Interactions are the same contrast on products of ±1 codes, divided by 4:
format × pretraining +82, format × RL −46, pretraining × RL +6, all three −6.
```

</details>

## The combined model

```claude
| stage | piece | from |
| --- | --- | --- |
| data | 155k cap-6 proofs, the ladder RL and transfer pools | Dan |
| format | `lean_seq`: Lean 4 tactic proofs, one symbol per token; reward = Lean ∧ nd_verify | Dan |
| pretraining | autoresearch incumbent 186: 6 × 384 Peri-LN, 4 ALiBi + 4 NoPE heads, Muon, MTP, 300 s | autoresearch (last week's post) |
| RL | expert iteration in 15 small rounds (400 targets, ~32 samples each), hardest-first queue, training samples at T 1.0, fresh AdamW per round, GEM loss | Leon (evolved by his GEAR agents) |
| RL sampling | verifier-guided: every `have` the model writes is checked against the rules given the lines before it; a bad line is rolled back in the KV cache and resampled (up to 4 times) | Leon's run-2 sampler, ported to the Lean format this week |

It got there in four steps on the same pretrained weights (details in our private group repo):
Leon's RL recipe +117 (687 → 804), the full RL target pool +4 (noise), guided sampling +101 (905,
but over the time budget), then the same guided run sped up and cut off at 1,800 s so the whole
pipeline fits in 2,400 s (904). Charles's result that plain expert iteration matches GRPO agreed
with Leon's choice, so there was nothing of his to add.
```

<details class="aside">
<summary>Verifier-guided sampling in the Lean format</summary>

```claude
Leon's sampler checked `abs`-format proof lines, which are already natural-deduction lines. In
the Lean format a line is a `have n7 : F := term ;`, possibly inside nested
`( fun ( n3 : A ) => by ... )` boxes. At every `;`, the Lean prefix is parsed with Dan's own
Lean → ND parser (stopped at the cut) and each new ND line goes through Leon's incremental
checker. Checks: 0 false rejects on 23,936 line checks of real proofs, 0 false accepts on 9,000
mutated proofs, and the KV-cache rollback reproduces an un-rolled-back forward to 7e-7 in the
logits. On the pretrained model it accepts ~55% more samples per round. The final proof is still
checked by Lean and nd_verify; guidance only prunes.
```

</details>

## Where the gains come from

```claude
All 8 combinations of {naive, best} × {format, pretraining, RL}, three seeds each, fresh, under
one protocol: same data, same 1,500 RL targets, 300 s of pretraining, at most 2,400 s end to end
(the longest took 2,374 s), one A40 (three runs got an RTX A6000, the same chip).
```

![pass@k on 250 held-out theorems](charts/passk-holdout.png)

```claude
250 theorems from the held-out half of the transfer pool, all 7-14 lines, never used to pick
anything; 256 samples per theorem. pass@k per theorem is the unbiased estimator
1 − C(n − c, k) / C(n, k) from n = 256 samples with c verified, averaged over theorems and then
over the three seeds.

Without Leon's RL (orange on the right) the curve starts highest and flattens early: plain expert
iteration sharpens the model onto proofs it already finds. Leon's recipe samples its training
proofs at T 1.0 and trains with GEM, a loss built to keep the output distribution spread out, and
the curves show it: lower at k = 1, far higher by k = 8.
```

![pass@k on 72 textbook problems](charts/passk-textbook.png)

```claude
Dmitry's textbook set, 72 problems nobody trained on: 58 with reference proofs of 1 to 33 lines
plus 14 without a reference, scored like the held-out set. The best seed solves 32 of 72. On the
naive pretraining the data format adds nothing and RL about 3 problems; only better pretraining
lifts the left panel clearly. On the right, removing the pretraining collapses the model back to
naive.

| pipeline | dev (of 1,108) | held-out pass@1 | pass@64 | pass@256 | textbook pass@256 |
| --- | --- | --- | --- | --- | --- |
| naive | 142 | 7% | 13% | 14% | 13% |
| + Lean format | 193 | 8% | 14% | 17% | 11% |
| + best pretraining | 457 | 26% | 45% | 48% | 24% |
| + best RL | 444 | 12% | 40% | 43% | 17% |
| **best** | **904** | 30% | **79%** | **82%** | **44%** |
| − Lean format | 782 | 28% | 68% | 71% | 35% |
| − best pretraining | 414 | 7% | 34% | 41% | 14% |
| − best RL | 682 | **44%** | 59% | 62% | 28% |
```

## Test-time scaling

```claude
Everything above spends a fixed budget of 64 independent samples per theorem. Here the
combined model's weights are frozen, and the question is what smarter inference buys: no value
function, no extra training.

Setup:

- five inference methods;
- all 1,177 held-out theorems, of which 114 are longer than 10 lines;
- three seeds;
- compute measured as tokens sampled from the model per theorem, rejected attempts included.

Details are in the test-time scaling write-up in our private group repo.

![Solve rate vs sampled tokens per theorem, by inference method](charts/test-time-scaling.png)

- **Plain sampling** (what the metric uses): 31% at 1 sample, 81% at 64.
- **Resample bad lines.** The model writes a proof one Lean tactic (`have ... ;`) at a time, and
  the checker from Leon's RL sampler can judge each tactic the moment it is written. When one
  fails, throw it away and redraw it, up to 10 times per proof.
  - 57% at 1 sample and 86% at 64.
  - On the 114 long theorems: 54 solved vs plain's 41.
  - It needs fewer tokens than plain for the same result: 84.3% at 11k tokens per theorem, vs
    plain's 80.8% at 14k.
  - With the KV-cache implementation from RL it takes 1.5× plain's wall-clock.
- **Sequential Monte Carlo** is almost as good: 84.4% at 16k tokens, a little worse on long
  proofs. It keeps N partial proofs, drops those that write a bad line, and clones the survivors.
- **Beam search and best-first search** are no better than plain sampling at the same compute.
  Both rank partial proofs by the model's own probability. The model's confidence is a poor guide
  to which half-written proof will finish.

What this says about the model:

- **Most failures are local slips, not bad plans.**
  - 69% of samples write at least one invalid line.
  - 58% of all lines checked are invalid.
  - Nearly half of all successful proofs needed at least one redraw.
  - The usual error is a wrong citation or rule on one line (65% of rejections); another 31% are
    lines outside the Lean grammar.
- **It is confidently wrong.** Half of all redraws reproduce the exact line that was just
  rejected.
- **Banning the repeats helps a little.** A redraw can be made to exclude the lines already
  rejected at that point, which is sampling without replacement. That lifts pass@1 by 2 points
  and pass@64 by 1.
- **The curves still flatten, and the rest looks like missing plans rather than slips.** About
  142 theorems (12%) are solved by no method at 64 samples. At 1,024 resampled attempts only
  about 1 in 10 of them falls, and those that do are hit 1-8 times in 1,024. For the rest, the
  model puts almost no probability on any proof. Search over the same model will not reach them;
  that takes a better model (pretraining, RL or data). Training on these evaluation theorems
  themselves is off the table: they are only ever scored.
- **A learned value does not help either.** I trained a small head on the frozen model to predict,
  from a half-written proof, whether it will finish. It was trained on other theorems and tested
  on validation theorems only. It ranks two partial proofs of the same theorem correctly 69% of
  the time. Steering SMC, beam or best-first search with it changed nothing.
- **The metric scores without this sampler, although the RL trained with it.** Scoring the
  combined model with resampling would add about 26 points at 1 sample and 5 at 64. That was not
  measured for the other seven cells, so it could reorder them.
```

![pass@k on the 72 textbook problems by inference method, with a 30% reference line](charts/textbook-test-time.png)

```claude
**The textbook set.** Dmitry wanted to get to about 30% on his 72 textbook problems. The combined
model gets there, at 8 samples with plain sampling and at 2 with resampling. It does not get there
at 1 sample either way.

| k | plain | resample | resample, no repeats |
| --- | --- | --- | --- |
| 1 | 17.6% | 26.0% | 27.4% |
| 2 | 22.1% | 31.1% | 32.9% |
| 8 | 29.8% | 38.6% | 41.1% |
| 64 | 38.3% | 46.1% | 48.7% |
| 256 | 41.7% (30 of 72) | 50.5% (36) | 51.9% (37) |

- These are 3-seed means for c005. c005 is a separate training run of the same recipe as the
  table's "best" cell, which gets 44% at 256 samples.
- Resampling is worth about 8-9 points at every k here, much less than on the held-out theorems at
  k = 1 (+26). The textbook proofs are longer (reference proofs of up to 33 lines), and more of the
  misses are missing plans rather than slips.
- By split, at 256 samples: the 58 with reference proofs go 42.5% → 51.7% → 52.3%; the 14
  without go 38.1% → 45.2% → 50.0%. At 1 sample: 16.8% → 26.2% → 27.8%, and 21.1% → 24.9% →
  25.4%.
- The textbook problems are only ever scored. Nothing trains or tunes on them.
```

## Test-time RL: making up easier problems

```claude
Search over a frozen model tops out: at 1,024 attempts the model puts almost no probability on
any proof of about 90% of the theorems it misses. Search can't fix that, so the weights have to
change. AlphaProof (Hubert et al., 2025) does exactly this for its hardest problems, with what
it calls test-time RL. For each target it generates hundreds of thousands of related problems
("variants": simplifications, generalizations, lemmas, sub-steps) and runs its normal RL loop on
the target plus variants until the target falls. The logic: RL learns only from successes, so a
target the model never solves gives no gradient. A variant the model solves *sometimes* does,
and it sits right next to the target. On IMO and Putnam problems this added 15 percentage points
beyond a 12-TPU-hour search, at a cost of 50 to several hundred TPU-days per problem.

This is the simple version of that, on our model.
```

### Where the variants come from

```claude
AlphaProof uses Gemini, prompted with 791 hand-made (problem, variant) pairs, plus code that
perturbs hypotheses and goals. We don't need a language model. A theorem here is a sequent
`Γ ⊢ G` over four atoms, and every variant below is built from the statement alone: no
reference proof, no model. Each is kept only if it is valid. Validity is a 16-row truth table,
because the logic is classical and has four atoms, so valid means provable. AlphaProof has no
such check (Lean mathematics is undecidable) and instead lets its prover try to *disprove*
each variant.

| kind | move | example: `P>Q, Q>R, R>S ⊢ P>S` |
| --- | --- | --- |
| goal `A > B` | assume A: `Γ, A ⊢ B` | `P>Q, Q>R, R>S, P ⊢ S` |
| goal `~A` | assume A: `Γ, A ⊢ F` | |
| any goal G | reductio: `Γ, ~G ⊢ F` | `…, ~(P>S) ⊢ F` |
| goal `A & B` | each half: `Γ ⊢ A`, `Γ ⊢ B` | |
| goal `A v B` | one side, if still valid: `Γ ⊢ A` or `Γ ⊢ B` | |
| premise `A & B` | split it into A, B | |
| premise `A v B` | the two cases: replace it by A; by B | |
| premise `A > B` | strengthen it to B, or to `~A` | `Q, Q>R, R>S ⊢ P>S` |
| premise `~(A & B)` | strengthen it to `~A`, or to `~B` | |
| lemma (cut) | any subformula ψ of Γ or G with Γ ⊨ ψ gives two variants: `Γ ⊢ ψ` and `Γ, ψ ⊢ G` | none: every subformula that follows from Γ is a premise or the goal |

Every move is applied to the target, then once more to each result (depth 2). Duplicates are
dropped: same premises as a set, same goal. So is any variant whose goal is already a premise.
On the 64 targets below this gives **3,397 variants** (median 42 per target, 12 to 263). 78%
are lemma cuts.

A true lemma need not be a *useful* one. The reference proof would say which lemmas lie on the
path to the target, but it is off limits, so the solve rates have to sort useful from useless.
```

### The loop

```claude
Model: c005 seed 0, the combined model above. One copy is adapted on all 64 targets at once,
as AlphaProof does for several targets. Each round:

1. **Sample** 16 proofs of every active target and variant. The sampler is the one Leon's RL
   uses: guided resampling of bad lines (4 redraws per line, 24 per proof), T 1.0.
2. **A target with any verified proof is solved.** It leaves the pool with all its variants.
3. **Mastered variants leave.** A variant solved in more than 75% of its 16 samples (13+) goes
   out: the model already has it, and its proofs would only reinforce what it does anyway.
4. **Train on the frontier.** Variants solved in 1-12 of 16 samples contribute up to 4 distinct
   proofs each, repeated max(1, round(8 × (1 − rate))) times, so rarely solved variants count
   up to 8×. That is Leon's weighting. Also in the mix: 20,000 random cap-6 training proofs,
   against forgetting. Variants at 0 of 16 stay in the pool untrained. Learning on their
   neighbours may make them solvable next round, which is how a curriculum is meant to work.
5. **Update** with one round of Leon's RL step: GEM loss, fresh AdamW, 135 steps of 128.

8 rounds. The band (0, 0.75] is my call. It follows AlphaProof's matchmaker, which prioritises
problems with mixed recent results and lowers the priority of those that always or never
succeed. A variant at 0 gives no data. One at 16 of 16 gives data the model doesn't need.

**Targets.** 64 dev theorems that c005 seed 0 fails at 64 plain samples, chosen by hash:
lengths 7-14, mostly 9-10. These are dev theorems, not held-out ones: this is the development
run. If TTRL works here, the held-out theorems that every inference method misses are the real
test. That would reverse "training on these evaluation theorems is off the table" above:
TTRL trains on the *statement*, never on a proof of it.

**Control.** The same sampler on the 64 targets alone: no variants, no training, 256 samples per
target per round for 16 rounds (4,096 per target). TTRL counts as a win only if it solves more
targets than the control at the same number of samples. The control is plain sampling at
matched compute. TTRL also pays for fine-tuning, which is reported separately.
```

### Results

```claude
**The stepping stones exist.** Round 1 samples the unadapted model, so it tests the premise
directly: does the model solve variants of targets it can't solve?

| round 1, 16 samples each | items | any proof | in the band (1-12 of 16) | mean solve rate |
| --- | --- | --- | --- | --- |
| targets | 64 | 12% | 12% | 0.01 |
| all variants | 3,397 | 43% | 27% | 0.24 |
| cuts, depth 1 / 2 | 243 / 2,391 | 44% / 39% | 24% / 26% | 0.26 / 0.21 |
| decompositions, depth 1 / 2 | 166 / 425 | 40% / 61% | 33% / 35% | 0.17 / 0.35 |
| specializations, depth 1 / 2 | 26 / 146 | 73% / 58% | 19% / 29% | 0.60 / 0.36 |

- The model solves variants 24× more often than their targets (0.24 vs 0.01).
- 63 of the 64 targets have at least one variant in the training band, and every target has
  at least one variant with a proof.
- Specializations (stronger premises) are the easiest, as they should be: they remove a case
  split or an implication. Many are mastered at once and leave the pool.

**TTRL solves 48 of 64. Sampling more solves 31.**
```

![Targets solved vs samples per target: TTRL vs sampling more](charts/ttrl-dev64.png)

```claude
- **The budgets are matched.**
  - TTRL used 2,623 samples per target (target plus its variants) and 29 minutes on one A40,
    of which 3 minutes were fine-tuning.
  - The control used 2,412 samples per target and 25 minutes.
  - At 2,443 samples per target, just past the control's full budget, TTRL stood at 40.
- **TTRL solves everything the control does, plus 17 more.** The control solves nothing TTRL
  misses.
- **TTRL barely samples the targets themselves.** Of its 167,872 samples, only 4,832 were on
  targets: 75 per target, against the control's 2,412. The rest went to variants, and training
  on those made the targets solvable.
- **The gain is on the long theorems.** At 10 lines TTRL solves all 25 targets; the control
  solves 10. At 11 or more lines TTRL gets 8 of 16, the control 6.
- **It starts slower and doesn't flatten.** For the first ~2,000 samples per target TTRL is
  behind, because it spreads its samples over variants. Then rounds 5-8 add 7, 4, 6 and 2
  targets, while the control adds 2 over its last 1,100 samples per target. TTRL was still
  climbing at round 8, when it stopped.
```

### What's left

```claude
16 targets are still unsolved. For 9 of them every variant has been solved at least once, and
most are mastered. la_transfer_1167, for one, has all 19 variants mastered, and the target is
still unsolved. The model can prove
every piece but doesn't write the whole proof in one pass. That's the composition gap: AlphaProof
closes it with tree search, which assembles solved subgoals automatically. Our model writes
proofs left to right without search.

The cheap fix is mechanical. A cut gives proofs of `Γ ⊢ ψ` and `Γ, ψ ⊢ G`. Paste the first
into the second where ψ is cited as a premise, renumber, and verify: that's a proof of the
target, built from the model's own pieces. Training on it teaches the long proof. This wasn't
in this run.
```

### Caveats

```claude
- **One seed, one model, 64 dev targets.** The 17-target gap is large next to anything
  seed-level we've seen, but it is one run.
- **It could be generic RL rather than target-specific learning.** TTRL spends 8 rounds of
  RL on theorems like these; the control spends none. If variants of *unrelated* theorems help
  as much, the variants aren't what matters. That control is next.
- **Dev theorems.** The 64 were picked by failing c005 at 64 plain samples on this dev split.
  Every knob was set before the run and none was tuned, but the real test is the held-out
  theorems every inference method missed. At 1,024 attempts about 90% of those stay unsolved.
- **Easy targets included.** 23 of the 64 fall to 256 guided samples, so "unsolved at 64 plain
  samples" was a weak filter for hard. The comparison is still like for like.
- **Temperature.** Both arms sample at T 1.0, the RL temperature. The frozen eval and the
  test-time section above use T 0.8.

Code: `code/experiments/current/ttrl/` on branch `robbie-experiments`. Per-item results:
`artifacts/ttrl/dev64_ttrl/`, `artifacts/ttrl/dev64_control/` (in the checkout).
```

## Methodology

```claude
| factor | naive | best |
| --- | --- | --- |
| format | the token era's `abs` format; reward nd_verify | Dan's `lean_seq`; reward Lean ∧ nd_verify |
| pretraining | autoresearch 012's `pretrain.py`: fork GPT, 4 × 256, RoPE, AdamW | autoresearch 186's `pretrain.py` |
| RL | frozen harness: 4 rounds × 1,500 targets × 32 samples, 600 fine-tune steps each | Leon's recipe + guided sampling, stopped at 1,800 s |

Everything else is fixed: the same 155k training proofs (in the two formats), 300 s of
pretraining, the same seeded 1,500 RL targets, the same evaluation (64 samples at T 0.8 on 1,108
dev-transfer theorems; the charts' pass@k at T 0.8 too). The two pretraining recipes differ only
in a tokenizer switch from the files the autoresearch loop ran. Code:
`code/experiments/current/factorial_20260929/` and `combined_pipeline_20260928/`, branch
`robbie-experiments`.
```

## Caveats

```claude
- **"Best pretraining" was found in the Lean format**, by the loop that was selecting on this dev
  metric, and "best RL" was evolved by Leon's agents in the token format on his own dev set. Each
  "best" is best where it was searched, so the format interactions partly measure where the search
  happened. The held-out and textbook numbers are free of the dev selection.
- The token format is judged by nd_verify alone, the Lean format by Lean ∧ nd_verify; they agree
  on more than 99.9% of samples.
- Best RL runs until 1,800 s after start (~1,450 s of RL); plain RL stops after its fixed 192k
  samples, 330-1,590 s. Both fit the same total budget, so part of the RL effect is using time the
  plain recipe leaves idle.
- Three seeds per cell; seed spreads are 10-60 theorems, small next to the effects above.
```
