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

## Note after batch R (2026-11-27 6:38 am PT; a record of a pre-registered branch, not an amendment)

Batch R was read at 6:37 am. R-final (lr 3e-4, one pass of 64 steps) gave 40.6 %, G 0.585 [0.474, 0.695]. That again
misses [0.77, 1.05], and it equals the dev-rule final cell (paired difference -2.6 pts [-10.9, +5.5]). R-own and R-unsel
gave 0.0 %. Following Amendment 4, the 25 Nov adapter itself (cell `dpo__final__OLD_A16__n64`, loaded and not trained;
its gate file was built on rp-om1 before R was read) was started through this sampling path at 6:38 am with seed
660004. Readings fixed now, before it has any result:
- If its G interval overlaps [0.77, 1.05], this sampling path counts as validated. The gap between the 25 Nov adapter
  and our final cells is then in training (data or code), not in sampling.
  - Candidates: the 25 Nov pool kept only Dan's hand-labelled genuine passes on arm-A prompts [erratum: wrong, see
    Amendment 6], while our pool is checker-only on the matched prompts; the prompts differ; the trainer and the LoRA initialisation seed differ.
  - Separating these takes one more training cell: the 25 Nov pool in our trainer at the same recipe. That is proposed
    and not run in stage 1.
- If it does not overlap, this sampling path (several adapters in one vLLM engine) differs from Dan's single-adapter
  path, which measured 63.5 % for the same adapter. Every adapter result of this run is then held back as unvalidated.

## Decision, 2026-11-27 7:20 am PT (coordinator, after the stage-1 report)

Stage 2 as pre-registered is not run. Its readout was the smallest own-source adapter that reaches, per checkpoint, and
at `dpo` no own-source cell moves the reach rate at all (0.0-0.3 % at both recipes), so there is nothing to shrink. Two
follow-ups replace it (Amendment 5). The coordinator intends to run them when the H200 class is free, unless Leon
decides otherwise.

## Amendment 5 (2026-11-27 7:20 am PT): the gap test and one later checkpoint; nothing is launched by this text

Stage 1 ended with two open points. First, a 24-point gap between the 25 Nov adapter (64.8 % through this path) and
final-source adapters trained here at the same recipe (40.6 %): the pool or the trainer and initialisation. Second,
whether own-source training does anything at a checkpoint closer to the final model. Common to both parts:
- Strict checker on the 48 reach prompts, 8 draws per prompt (384 draws per cell), through the validated multi-adapter
  path, with the LoRA gate first. Analysis by `analyze_cells.py`: G with the 95 % two-level paired bootstrap and
  paired contrasts. All passes are checker passes. Each sampled cell gets a CPU hand check (`handcheck.py`: rule-1
  screen on every pass, 24 blinded single reads, 8 of Dan's labelled `B_s2300` items mixed in).
- Every cell is A16 (r16 on all projections, alpha/r = 2) at n = 64, with the 25 Nov recipe: lr 3e-4, exactly 64
  steps (one pass), batch 1, AdamW (0.9, 0.999, 1e-8, wd 0), clip 1, loss on the whole response. Fixed recipe, no
  selection, A-init seed 0 unless stated.
- The trainer is `mini_train.py` with the step-64 rule fixed and step-0 scoring of the training set in every run (see
  `log.md`). Neither change affects a fixed-recipe cell's adapter.

### (a) The gap test, at `dpo`

Cells (one sampling batch, batch seed 660005):
- `dpo__final25__A16__n64__r1p`, "trainer on the 25 Nov pool":
  - Training data: the 25 Nov adapter's own 64 training responses (`olmo_elicit/pool/train.jsonl`), in its order
    (order 0, the seed-0 permutation recorded as `perm` in `runs/dpo/edl/edl_rows.jsonl`).
  - Dev responses: our 32 final dev responses, used for logging only; 12 of them are in the 25 Nov training set.
  - Relative to the 25 Nov adapter, only the trainer code path and the LoRA initialisation differ. Relative to R-final
    (`dpo__final__A16__n64__r1p`), only the pool and its order differ.
- `dpo__final__A16__n64__r1p_s20261128`, "second seed": R-final's pool, order and recipe, with A-init seed 20261128.
  This gives the seed spread on our pool.

Readings, fixed now. Contrasts are paired against R-final (40.6 %). The "window" is the 25 Nov interval for G,
[0.77, 1.05].
1. **Pool.** The 25 Nov-pool cell overlaps the window, AND its contrast with R-final has a lower bound above 0, AND the
   second-seed contrast includes 0. Reading: the 24 points come from which 64 responses are trained on. The hand check
   of the 25 Nov-pool cell then says whether the extra passes are gamed (the 25 Nov adapter: about 38 % gamed against
   16 % for R-final).
2. **Trainer.** Both contrasts include 0. Reading: our trainer on the 25 Nov pool gives our rate, so the difference lies
   in the training code path or the initialisation scheme, not the data and not seed noise.
   - Next is a CPU diff of `mini_train.py` against `olmo_edl.py` on one example: initialisation, loss mask, template,
     dtype. The coordinator decides any GPU follow-up.
   - (b) is held until then, because own-source results from a faulty trainer would not answer question B.
3. **Seed spread.** The second-seed contrast has a lower bound above 0. Reading: one A-init seed moves the rate by
   itself, so single-seed comparisons in this run (including the 24-point gap) are within seed spread. Stated with every
   stage-1 contrast, and the pool reading is not made even if (1) also holds.
4. **Neither or partial.** Any other combination, for example the 25 Nov-pool cell above R-final but not overlapping
   the window. Reported as unresolved, with the two contrasts and the hand check.

Stop rule: one batch. No further cell is launched from (a)'s result under this amendment. A cell that fails the gate
is not sampled and is reported.

GPU plan and cost:
- Training can run on an H100 at any time: download, 2 cells and step-0 scoring take about 30 min, about $2.20.
- Sampling needs the H200 class: setup, download, gate, and two final-like cells take about 1.1 h, about $5.80.
- Alternatively both run on the H200: about 1.4 h, about $7.40.
- Total about $8, cap $12.

### (b) One later checkpoint: `s1100`

Runs only after (a) has been read, and not if (a) reads "trainer". The checkpoint is `s1100`, RL step 1100. Its plain
reach rate is 38.5 % (Dan's stored B rows, 32 draws per prompt). The final model's is 69.4 %, so
G = (cell − 38.5) / (69.4 − 38.5). Pools exist (`v2/pools/s1100/{own,unsel,final}`, 112 matched prompts, dev 32).
Checker bound for own at n = 64: 54.6 bits.

Cells (reach batch seed 660011):
- `s1100__own__A16__n64__r1p`: `s1100`'s own checker passes.
- `s1100__unsel__A16__n64__r1p`: `s1100`'s own draws, checker ignored.
- `s1100__final__A16__n64__r1p`: the final model's passes on the same prompts. This is the in-pipeline reference.
  If (a) read "pool", note that this pool is checker-only, like the one that under-performed at `dpo`.
- `plain:answeronly` at `s1100`: the prompt control, with Dan's answer-only suffix, no adapter, seed 650000 + 1100.
- An own-source recipe chosen on own-source data, `s1100__own__A16__n64__tuned` (recommended; see below).

The tuned own cell, and why it should be in. Amendment 3 gave own cells the final source's recipe, because
held-out loss on a model's own samples rises with any training and so cannot choose a recipe. A reader will ask
whether a recipe tuned for own samples would move anything. The only own-data criterion that targets behaviour is the
checker pass rate on held-out non-reach prompts. The rule, fixed now:
- Candidates: three own-source A16 n64 recipes, (lr 3e-4, 64 steps) (the fixed recipe), (lr 1e-3, 64 steps) and
  (lr 3e-4, 192 steps = 3 epochs).
- Each is sampled on the 32 own-source dev prompts, 4 draws each (128 draws, dev batch seed 660012).
- The candidate with the highest strict pass rate on the dev prompts wins (ties go to the weaker recipe, in the order
  listed). If the winner is not the fixed recipe, it joins the reach batch as the tuned cell. If it is, the fixed own
  cell already covers it, and that is reported.
- Dev prompts are never reach prompts. Nothing about the reach prompts informs the choice.
- The same rule could run at `dpo` alongside (a) for about $3 more. Not included by default.

Readings for (b), fixed now:
1. **Own-sample elicitation at `s1100`.** Own (fixed or tuned) reaches (G lower bound at least 0.80), AND own − unsel
   has a lower bound above 0, AND the prompt control does not reach.
2. **Partial.** Own − unsel has a lower bound above 0, or the own G lower bound is above 0, but rule 1 is not met.
   Own-source selection moves the rate without reaching.
3. **Null.** Own and unsel contrasts with `s1100` plain include 0, and own − unsel includes 0. The `dpo` null extends to
   step 1100. The curve is not run (`s0100` and `s0500` are skipped).
4. **The prompt control reaches.** Question A at `s1100` reads "how small an adapter replaces one sentence of prompt",
   as in PREREG.
5. **Extension.** `s0100` and `s0500` (about $7 each, the same three cells plus the control) only after reading 1 or 2,
   and by the coordinator's decision.

Stop rule: one dev batch and one reach batch. No confirmation cell, and no size sweep under this amendment.

GPU plan and cost:
- H100 (training, any time): download `s1100` (64.5 GB, about 4 min); three cells plus two extra tuned candidates
  plus step-0 scoring (about 45 min); about 50 min in all, about $3.70.
- H200 class (sampling): setup and download (about 10 min); the dev batch of three candidates (about 12 min); the
  prompt control (about 20 min; `s1100` answers are shorter); the reach batch of three or four adapters (about
  70-80 min); about 2 h in all, about $10.60.
- Total about $14, cap $18.

Combined: (a) + (b) about $22. The launch stop for this amendment is $70 on the experiment's pod ledger (stage 1
$43.54 + $26.50). The whole experiment stays under $120.

## Amendment 6 (2026-11-27 7:38 am PT): replaces 5(a), adjusts 5(b), adds an own-recipe check at `dpo`; nothing launched

Written after the independent audit of the stage-1 report (`discourse/stage1_report_audit.md`).

**Erratum to the note after batch R and to Amendment 5.** The 25 Nov pool was not "only genuine-labelled passes".
- It took one strict pass per arm-A prompt, excluded passes Dan's readers had labelled gamed and preferred
  labelled-genuine ones: 26 genuine, 38 unlabelled and 0 gamed of 64 (`olmo_elicit/pool/train.jsonl`).
- Ours took one checker pass uniformly: 11 genuine, 5 gamed and 48 unlabelled.

**What the files already say (audit S7).** A code fault is unlikely. The live candidates are which 64 responses are
trained on (and in what order) and the LoRA A-init seed, i.e. the run-to-run variance of a 64-example SFT.
- Code: the two trainers share `olmo_edl`'s `train_step`, `reset_lora`, `score` and `RECIPE`.
- Scores: identical responses get bit-identical step-0 nats.
- Updates: total update norms differ by 3 % (29.6 against 28.8), and per module type by at most 4 %.
- Pools: they match on length, difficulty and constraint families.
- Init seed: every stage-1 A16 cell used one A-init seed (1581104982). The 25 Nov adapter used 523373504.

### 6(a) Gap test at `dpo` (replaces 5(a))

Every cell is A16 at n = 64, lr 3e-4, exactly 64 steps (one pass), fixed recipe, `mini_train.py` as committed.

**Step 1, training-only replication (H100, no sampling).**
- Cell `dpo__final25__A16__n64__r1p_s523373504`: the 25 Nov pool's 64 responses, staged as a pool file in the 25 Nov
  order (the seed-0, order-0 permutation recorded as `perm` in `runs/dpo/edl/edl_rows.jsonl`), with A-init seed
  523373504. The cell field `seed` is passed unchanged to `oe.reset_lora`.
- Compare its weights on CPU with the 25 Nov adapter, per LoRA module: the cosine of ΔW = scale·B·A, and the ratio of
  Frobenius norms.
- Readings, fixed now:
  - **Match:** every module's cosine is at least 0.99, and the total norm is within 2 %. Our trainer reproduces the
    25 Nov adapter from its pool, order and seed, so the trainer-code branch is closed without a draw.
  - **No match:** any module's cosine is below 0.90, or the total norm is off by more than 10 %. The trainer branch is
    open. Step 2 and 6(b) are held, and a CPU diff of the two code paths on one example comes first.
  - **In between:** reported as "close, not identical" (bf16 order effects). Step 2 goes ahead, and the caveat is stated
    with every gap reading.

**Step 2, the crossed design, sampled only after "match" or "in between".** Pool (25 Nov or ours) × A-init seed (25 Nov
523373504 or ours 1581104982). Two corners exist already:

| corner | pool | A-init seed | status |
|---|---|---|---|
| A | 25 Nov | 25 Nov | the 25 Nov adapter: 64.8 %, batch_old |
| D | ours | ours | R-final, `dpo__final__A16__n64__r1p`: 40.6 %, batch R |
| B | 25 Nov, in its order | ours | new: `dpo__final25__A16__n64__r1p` |
| C | ours, in our order | 25 Nov | new: `dpo__final__A16__n64__r1p_s523373504` |

- B and C are sampled in one batch: seed 660005, 8 draws per prompt (384 each), with the gate first. That matches A
  and D.
- Paired contrasts use the two-level bootstrap.
  - Pool contrasts: B − D (at our seed) and A − C (at the 25 Nov seed).
  - Seed contrasts: C − D (on our pool) and A − B (on the 25 Nov pool).

Readings, fixed now ("above 0" means a 95 % lower bound above 0; "includes 0" means the interval does):
1. **Pool.** Both pool contrasts are above 0, and both seed contrasts include 0.
2. **Seed.** Both seed contrasts are above 0, and both pool contrasts include 0.
3. **Both.** At least one pool contrast and at least one seed contrast are above 0.
4. **Combination only.** A is above each of B, C and D, while B, C and D do not differ from each other. Only the
   specific 25 Nov combination gives the high rate, and the 24 points cannot be attributed to pool or seed.
5. **Unresolved.** Any other pattern: reported with all four contrasts.

What a seed effect does to stage 1. Let X be the larger of the two seed contrasts' point estimates, if either seed
contrast is above 0 (readings 2 or 3). Every stage-1 contrast trained on a single seed is then re-labelled "within seed
spread" if its 95 % lower bound is below X:
- final − own: +40.6 [+31.0, +50.3] and +43.0 [+32.8, +53.4];
- the 25 Nov adapter − R-final: +24.2 [+16.2, +32.0];
- the two final recipes: −2.6 [−10.9, +5.5].
The own-source null is then stated as "one seed". A second own seed at `dpo` is proposed (about $4), not run under
this amendment.

**Hand check: one blinded round for all four corners.**
- 24 fresh items from each of A, B, C and D (round-robin over prompts, a new seed), plus 8 of Dan's labelled
  `B_s2300` items, shuffled into batches. A and D are re-read in this round, so all corners share one calibration.
- Two independent Sonnet reads per item, and a third on a split. Dan's mechanical rule-1 screen runs on every pass.
- Readout per corner: the screened share, the read gamed share, and a genuine-pass rate with an interval. The
  interval comes from a bootstrap over prompts, then draws, then read items, so it includes the rate's own sampling
  error.
- Pre-set reading: the gap is "mostly gamed passes" only if the A − D genuine-pass contrast's 95 % interval includes 0
  and its point estimate is at most a third of the raw 24.2 points. Otherwise "not shown".

### 6(c) Own-source recipe check at `dpo`, alongside 6(a) (new)

A reader's first question about the own-source null is whether a recipe tuned on own data moves anything. The rule is
the same one Amendment 5 fixed for the tuned cell at `s1100`.
- **Candidates.** Three own-source A16 n64 recipes at `dpo`:
  - lr 3e-4 for 64 steps, which already exists as R-own;
  - lr 1e-3 for 64 steps;
  - lr 3e-4 for 192 steps (three epochs).
- **Dev batch.** Each candidate is sampled on the 32 own-source dev prompts, 4 draws each (128 draws per candidate,
  dev batch seed 660007). The reference is `dpo` plain on the same prompts, from Dan's stored arm-A draws
  (`F_dpo_ifb`, 16 per prompt).
- **Choice.** The highest strict dev pass rate wins; a tie goes to the earlier-listed recipe. If the winner is not R-own's
  recipe, it is added to step 2's reach batch as `dpo__own__A16__n64__tuned` (384 draws). Reach prompts play no part in
  the choice.
- **Readings, fixed now:**
  - Manipulation check: for each candidate, dev pass rate minus `dpo` plain, paired by prompt. If none is above 0, the
    reading is "own SFT at A16 n64 does not move held-out non-reach behaviour either".
  - Reach: a tuned cell whose G lower bound is above 0 qualifies the null as "moved by a recipe tuned on own data".
    Otherwise: "null also for an own-data-tuned recipe at A16 n64, one seed".

### 6(b) `s1100` (adjusts 5(b))

As 5(b), with three changes:
1. **16 draws per prompt** (768 per cell) for own, unsel, final and the tuned own cell. At 8 draws, a cell equal to the
   final model "reaches" about 70 % of the time; at 16 draws about 93 % (audit S14, simulated from Dan's rows). The
   prompt control stays at 8 draws. The reach batch seed is 660011; with 16 draws it runs as two chunks of 8 under the
   sampler's chunk index.
2. **Order.** 6(b) runs after 6(a) has been read, and not if step 1 reads "no match".
3. **Second seed if 6(a) finds a seed effect.** If 6(a) reads "seed" or "both", own and final at `s1100` get a second
   A-init seed each (523373504), at about $7 more.

The readings, the tuned-own rule and the stop rule are as in 5(b).

### Costs, caps and order

| part | where | time | cost | cap |
|---|---|---|---|---|
| 6(a) step 1 | H100 at any time: download, one cell, export, then the CPU weight comparison | about 25 min | about $1.80 | $3 |
| 6(a) step 2 + 6(c) | H100: B, C and two own candidates, plus step-0 scoring (about 40 min, about $3). H200 class: setup (about 10 min); dev batch of three own candidates (about 15 min); reach batch of B, C and the tuned own cell if any (about 70-85 min) | | about $13 | $18 |
| 6(b) | H100: about 50 min, $3.70. H200 class: about 3 h at 16 draws (about 12 M tokens at about 1,400 tok/s), about $16 | | about $20 | $26 |
| conditional second seed at `s1100` | | | about $7 | |

Order:
1. The README is corrected and the code and PREREG are committed. The coordinator commits.
2. 6(a) step 1. It needs only an H100, so it can run before the H200 frees up.
3. 6(a) step 2 with 6(c).
4. Read 6(a).
5. 6(b).

Total about $35, or about $42 with the conditional seed. The experiment's launch stop is $90 on the pod ledger (stage 1
$43.54 + $46), under the $120 cap. Each part runs once. Nothing else is launched under this amendment.
