---
written_on: 2026-11-27
written_by: agent:claude (opus, orchestrator)
---

# Olmo minimal adapter: readings fixed in advance

Written 2026-11-27 2:10 am PT, before any pod exists and before any training or sampling of this experiment.
Design: [SPEC.md](SPEC.md). It follows the design discourse ([discourse/edl_design.md](discourse/edl_design.md),
four turns), adjudicated by me; the points I settled are listed at the end. Amendments are appended below with a
time stamp, each before the data it changes.

## What is measured

- Primary, per sampled cell: **G**, the share of the gap between `dpo`'s plain rate (0 %) and the final model's plain
  rate (69.4 %, Dan's 32 new `B_s2300` draws per prompt) that the cell closes on the 48 reach prompts, with a 95 %
  two-level paired bootstrap interval (prompts, then draws for both the cell and the reference). Shown beside it: the
  strict pass rate (% of 384 draws), the paired difference to the reference in points, prompts with a pass in 8.
- All passes are checker passes. Nobody reads them for genuine against gamed before the result is reported; a
  hand-check of a sample of reach passes in the cells that decide a claim is planned (CPU) and reported separately.
- Secondary, from training logs only: dev loss (total and answer), EDL in bits, one-pass capacity ratio C(P) for
  source (ii), the checker bound for own-source training sets.

## Training rule per cell

As SPEC.md "Training rule", unchanged after this point: lr grids per family at scale alpha/r = 2, successive halving
to 512 steps for the small own-source rungs, early stop on two consecutive dev-loss rises, the (lr, step) with the
lowest total dev loss per token on the same source's 32 dev responses is "the trained adapter". Reach prompts are never
used to choose anything except through the search rule below.

## Prior (written in before the data)

TinyLoRA (Morris et al. 2026, §3) defines supervised fine-tuning data exactly as our source (i) and argues that SFT
needs many more trained weights than RL; Donoway et al. 2025 needed about 10^4 to 2×10^4 weights to close half the gap
in their only long chain-of-thought task. So a failure of the own-source cells at M4k or M256 is the expected result
and is not evidence against elicitation. The descending search below spends cells where a pass is plausible.

## Search rule (sampling decisions use G only, through these rules)

"Reaches" = the lower bound of G's interval is at least 0.80 (at the reference 69.4 %, at most 13.9 points below). It
steers the search; it is not a test of equality. Power: a cell needs a true G near 0.97 to clear it reliably, so a
miss with a point estimate above 0.80 is reported as a near miss, and the design cannot resolve differences below
about 6 points (0.09 in G).

1. **Batch 1:** the prompt control (`dpo` + answer-only suffix, no adapter), own A16 n64, unsel A16 n64, final A16
   n64.
2. **Pipeline check:** the final A16 n64 cell's G interval must overlap [0.77, 1.05] (the old adapter's interval,
   53.7-73.2 % / 69.4 %). If not, the pipeline is audited before any other cell is read.
3. **If own A16 n64 does not reach:** no further own size or example cells. Report batch 1 and the contrasts.
4. **If it reaches:** batch 2 = own O1 n64, own A16 n4, own A16 n1.
   - O1 reaches: train M4k-all and M4k-attn, sample the one with the lower dev loss; if it reaches, M256 (same module
     pool), if budget remains. O1 does not reach: A1 n64.
   - n4 reaches: n1 decides the example axis. n4 does not reach: own A16 n16.
   - n1: if its G lower bound is at least 0.75, sample prefix positions 2 and 3 of the same order as single-example
     cells (8 draws each). One example is claimed only if 2 of the 3 reach. A failure at n = 1 is a one-example
     result and says so.
5. **Confirmation:** the smallest rung that reaches is re-trained with a new mask and A-init seed (full procedure) and
   sampled with fresh draws. It is claimed only if both reach.
6. **Claims:** "the smallest reaching P is at most X" (X = the smallest confirmed rung). Between rungs that change
   construction (O1 dense o_proj → M4k sparse B entries), nothing is located. A non-reach on an `unconverged` rung is
   reported as "not shown" in the headline table, never as "insufficient".

## Question B: what counts as "elicitation without the RL model's help"

All three, on the same 48 prompts:
- an own-source (i) cell reaches at some (P, n), AND
- the paired difference (i) − (iv) at A16 n64 has a lower 95 % bound above 0 (then read as "checker selection,
  including its length correlate, matters"; a length-matched (iv) cell is a conditional extra), AND
- the prompt control does not reach.

Other outcomes and their readings:
- (i) reaches and (i) − (iv) includes 0: reached without any token from a later RL model, but not attributable to the
  checker's selection: training on any own response in the template does it.
- The prompt control reaches: one instruction elicits the behaviour; question A then reads "how small an adapter
  replaces one sentence of prompt".
- (ii) reaches and (i) does not at A16 n64 (paired (ii) − (i) lower bound above 0): the final model's tokens carry
  something a checker-selected `dpo` sample does not, at n ≤ 64. It does not show that RL is necessary: selection here
  is one round on prompts `dpo` already passes, not iterated expert iteration or exploration on hard prompts.
- Mild leak stated with every B claim: the 115-prompt matched set keeps only prompts the final model also passes
  (14 of `dpo`'s 129 checker-eligible prompts dropped; counts from the checker-only pool rebuild at about 2:20 am PT,
  before any pod).

## Capacity and bits per trained weight

- For source (ii) only: one-pass C(P) = EDL_64(P) / EDL_64(A16) at n = 64 for A16, M65k, M16k, M4k, M256; extended C
  at M16k and M4k (512 steps, dev-selected step). A rung is called "capacity-limited" only if its extended C stays
  below 0.95. EDL/P is reported in bits per trained weight, labelled "A16-referenced, n = 64, not comparable to BTC
  thresholds". No P* is claimed (BTC's thresholds need data that exceed capacity, about 10^4 examples).
- Descriptive comparison, not a test: with our measured EDL_64(A16), BTC's elicitation band (0.01-0.1 bits per
  parameter) puts the C < 0.95 crossing between EDL/0.1 and EDL/0.01 weights, the teaching band (≈ 1 bit) near EDL
  weights. We report where the observed crossing falls.
- Own sources: the checker bound Σ log2(1/a_i) over the training prompts (a_i estimated from 16 stored draws), the
  most information the filter can supply; divided by P only as a description. Own-source EDL has no BTC meaning (the
  loss includes the model's own sampling entropy).
- Conclusions say "reached with P weights and n examples", never "elicitation in BTC's sense": `dpo`'s IFBench
  capability is already demonstrated (pass@1 29 %), and BTC's criteria are not defined at n ≤ 64.

## Descriptive, no prediction

- Answer length and thinking length per cell; the rate split by instruction-type coverage of the training prompts.
- The labelled-gamed share in each training set.
- The token-level footprint side task (CPU, stored data), reported with its audit.

## Stopping rules

- Money: stage 1 launches no new cell once the pod ledger passes $50 (estimate $44-50); the whole experiment stops at
  $120, with stage 2 (other checkpoints) decided with the coordinator after stage 1.
- Time: stage 1 reported by 2026-11-27 1:00 pm PT; what is unfinished then is reported as unfinished.
- Drop order if money binds: the (ii) M256 capacity rung, the extra n = 1 examples, the (ii) M4k extension, own n16,
  M4k-attn.

## Points settled by the orchestrator after the discourse

1. No hand labels in example selection, for every source (auditor and proposer agree; labels are used only to report).
2. Headline per cell = G with its interval, the rate and the paired difference beside it; "reaches" (G lower bound ≥
   0.80) steers the search only.
3. No fresh `s2300` reference cell: `B_s2300` draws were new draws after the reach set was fixed (checked by both).
4. Scale alpha/r = 2 on every rung (the proposer's alpha = 16 at r1 dropped).
5. n = 1 at 8 draws (the brief's fixed 8 draws per prompt), extra examples only by the trigger in rule 4.
6. EDL at fixed t = 64 is "one-pass C"; capacity claims need the extended rungs (auditor N2 accepted).
7. Open points that no design change fixes are listed at the end of the discourse (Turn 4) and will be repeated in the
   summary.

## Amendment 1 (2026-11-27 2:12 am PT, before any pod exists)

Cost planning after the build (H100 at about 4.7 h and H200 at about 5.6 h put stage 1 near $50):
- The source-(ii) capacity rungs M65k and M256 are run only if stage-1 spend allows; they are first in the drop order.
  The one-pass C ladder is then A16, M16k, M4k (M16k and M4k also extended to 512 steps), which still covers
  7.6e3-7.6e4 weights, the only range where BTC's two bands differ for an EDL near the old run's.
- M4k-attn and A1 are trained before batch 2's result is known (so the H200 does not wait); M4k-all is trained only if
  O1 reaches, as rule 4 says. Nothing about which cell is sampled changes.
- The stage-1 launch stop stays at $50 on the pod ledger.

## Amendment 2 (2026-11-27 2:16 am PT, pods created, nothing run on them yet)

The coordinator approved stage 1 at about $50 with the capacity ladder kept, and set the stage-1 launch stop at $55 on
the pod ledger (replacing the $50 above); the whole-experiment cap stays $120. At the stop: finish and fetch what is
running, start nothing new, report. The search rule runs without checking back between batches.

## Amendment 3 (2:47 am PT, 2026-11-27; written from training logs only, before any reach-prompt result has been read. The prompt control has been sampled and fetched but not opened.)

**What I saw.** On rp-om1 the first own-source cell (dpo__own__A16__n64) gave a dev loss on dpo's own held-out passes
that rose with training at every learning rate. Step 0: 0.71402 nats per token. lr 1e-4: 0.71420, 0.71430 and 0.71434 at
steps 16, 32 and 64. lr 3e-4: 0.71527 to 0.71727. lr 1e-3: 0.73987 to 0.80737. The 3e-5 edge run: 0.71422 at step 16
and 0.71415 at step 32. The answer segment behaved the same (0.74123 at step 0; best 0.74121). A model's own temperature-1
samples are most likely under the model itself, so on own-sample (and unselected-own) sources the held-out likelihood
is maximised by not training. The pre-registered selection ("lowest dev loss per token") would therefore return an
essentially untrained adapter for sources (i) and (iv), whatever those examples could do to behaviour. This is the
self-sample problem (discourse Turn 4, open point 6) reaching the selection rule as well as EDL. I did not foresee it.

**Change, for sources (i) own and (iv) unsel only.** These cells use a fixed recipe and select nothing themselves: the
(lr, step) selected on the final source's dev loss at the same rung and n, trained for exactly that many steps (cells
`dpo__final__<rung>__n<n>`, trained for this purpose where they did not exist: A16 n4, A16 n1, O1 n64, and later A1
and M4k-attn if the search reaches them). So own, unsel and final are compared under the same training recipe and differ
only in the examples. The cost is a leak of hyperparameters, not tokens: the recipe is chosen on the final model's
responses. That is at most log2(number of candidate (lr, step) pairs), about 4 bits, and it is stated with every
question-B claim. The final-source cells keep the pre-registered dev-loss selection unchanged.

**Kept on file.** The dev-selected own A16 n64 cell's training logs are reported (EDL near zero or negative, a
"did_not_train" flag). It is not sampled: its selected adapter is the step-16 or step-32 adapter at the lowest lr, and
spending a cell on an untrained adapter does not answer the question.

**Search rule.** Unchanged, except that the own cells of every batch are the fixed-recipe cells.

**Schedule.** rp-om2 trains final A16 n64 and then own and unsel A16 n64 with its recipe; rp-om1 trains the final cells
that provide recipes (A16 n4, A16 n1, O1) and the capacity rungs (M16k, M4k, both extended).

## Amendment 4 (2026-11-27 5:01 am PT, appended at 5:01:44; batch 1 has been read; the cells below were started at 5:01 am, no result of them exists)

**Batch 1, read at 4:59 am (seed 660001, 384 draws per cell, checker passes).** final A16 n64 43.2 % of draws, G 0.62
[0.51, 0.74]; own A16 n64 (fixed recipe) 0.26 %, G 0.004 [0, 0.018]; unsel A16 n64 0.0 %; prompt control 6.25 %, G 0.09
[0.02, 0.19]. No cell reaches. **The pipeline check fails**: the final A16 n64 interval [0.51, 0.74] does not overlap
[0.77, 1.05]. By rule 2 the pipeline is audited before any other cell is read. Rule 3 (own A16 n64 does not reach, so
no further own cells) applies to the recipe used in batch 1.

**Audit, from training logs and the 25 Nov records.** The old adapter (63.5 %, G 0.92) was r16 on all modules at
alpha/r = 2, trained one pass over 64 of the final model's responses at lr 3e-4, the lr with the shortest online code
length at `dpo` on 25 Nov (a non-reach criterion). The dev-loss rule here picked lr 3e-5 for 128 steps. The candidates'
dev losses are near ties: 0.5373 for the selected cell, 0.5391 for 1e-4 at step 64, 0.5404 for 3e-4 at step 64 (answer
segment 0.3345, 0.3313, 0.3316). So the rule chose among near ties whose update sizes differ by 10×. Own and unsel
inherited that recipe (Amendment 3). Working hypothesis: the shortfall comes from the recipe, not from a fault in
training or sampling. The LoRA gate passed for all three cells (per-token correlation ≥ 0.976).

**Test (batch R, cells `dpo__<source>__A16__n64__r1p`).** final, own and unsel A16 n64 trained at the old adapter's
recipe: lr 3e-4, exactly 64 steps (one pass over the n = 64 pool prefix in pool order), no selection. They are sampled
together with seed 660003, 8 draws on the 48 reach prompts. Readings fixed now:
- Pipeline: if R-final's G interval overlaps [0.77, 1.05], the pipeline counts as validated, and batch 1's final
  shortfall is attributed to the dev-loss rule's choice of lr. If it does not overlap, the old adapter itself is
  sampled through this path before any other reading.
- Question B at A16 n64 is read on the R cells with the PREREG criteria: own reaches; (own − unsel) lower bound above 0;
  the prompt control (batch 1) does not reach. Batch 1's own and unsel cells are reported as the result at the
  dev-rule recipe. Neither recipe's result is dropped.
- Information note: (3e-4, one pass) was fixed on 25 Nov without any reach data from this run. The choice to test it
  now is informed by batch 1's reach result: one binary choice between two recipes, stated with every claim from R.
- Batch 2 (own O1, n4, n1) is not launched in stage 1. Under the old recipe, "one pass" at n < 64 means fewer than 64
  steps, which is a new design question. If R-own reaches, batch 2's recipe goes into the stage-1 report as a proposal.
- The capacity ladder (final source, dev-loss selection, training logs only) is unchanged.
