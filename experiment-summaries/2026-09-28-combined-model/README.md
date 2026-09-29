---
written_on: 2026-09-29
written_by: agent:claude
draft: true
---

```claude
For context, see [last week's post](../2026-09-21-pretrain-ablations/README.md).
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

It got there in four steps on the same pretrained weights ([details](../2026-09-28-combined-pipeline/README.md)):
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

Details are in the
[test-time scaling experiment](../2026-09-29-test-time-scaling/README.md) (branch
`test-time-scaling`).

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
- **The curves still flatten.** About 168 theorems (14%) are solved by no method at any budget
  tried. These look like missing plans rather than slips, which is where a learned value function
  or training on the failures would come in.
- **The metric scores without this sampler, although the RL trained with it.** Scoring the
  combined model with resampling would add about 26 points at 1 sample and 5 at 64. That was not
  measured for the other seven cells, so it could reorder them.
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
