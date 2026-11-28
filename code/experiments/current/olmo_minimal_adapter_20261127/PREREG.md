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

## Amendment 6, step-1 result (2026-11-27 8:07 am PT) and Amendment 7 (PROPOSED 8:10 am PT; in force only after the coordinator's decision is recorded under it)

**Step-1 reading, by the pre-set rule: "no match".**
- The replication cell `dpo__final25__A16__n64__r1p_s523373504` (rp-om3, fetched md5-equal) against the 25 Nov
  adapter: 15 of 448 modules have a ΔW cosine below 0.90 (minimum 0.79).
- Total cosine 0.986, median module cosine 0.998, total norm ratio 0.997.
- As fixed, step 2 and 6(b) are held. Nothing else was trained, and rp-om3 is deleted.

**CPU diagnosis (8:08-8:10 am).**
- **Where.** All 15 modules are in layers 0-4. Seven are layer 0 (all seven projections), four layer 1, two layer 2,
  one each in layers 3 and 4. The minimum cosine per layer rises with depth: 0.79, 0.85, 0.90, 0.89, 0.89, 0.91 ...
  From layer 8 onward the minimum is 0.915 and the median 0.998. Together they hold 2 % of the update's squared norm.
- **Not an initialisation difference.** The stored `init_seed` is 523373504 in both runs. The step-1 online score is
  bit-identical. Different A-inits give a total cosine of about 0.015 (the 25 Nov adapter against R-final), not 0.986.
- **Not an order difference.** The per-step training pids are identical in all 64 steps.
- **Consistent with bf16 run-to-run drift: same seed, same data, step 1 bit-identical, then a small drift.**

  | pair | first step that differs | max relative difference | median relative difference | code-length difference |
  |---|---|---|---|---|
  | 25 Nov run vs our replication (H100 vs H100) | 2 | 3.9e-3 | 6.5e-4 | −7.8e-5 |
  | our own pair, final pool (dev-rule 3e-4 row vs R-final, H200 vs H200) | 3 | 1.2e-2 | 5.3e-4 | −4.9e-5 |
  | our own pair, own pool (dev-select 3e-4 row vs R-own, H100 vs H200) | 3 | 2.9e-3 | 5.4e-4 | +1.2e-4 |

  The 25 Nov dpo-sweep row also re-ran on 25 Nov and moved its code length by 0.003 %.
- **Difference from our own pairs.** The replication first differs one step earlier (at step 2, by 3.4e-4). The two
  trainers may still differ in a small numerical path in the first update, for example whether the loss is computed in
  chunks (`--ce-checkpoint`), which the 25 Nov row does not record. It is not material on these numbers.
- **Missing baseline.** There is no weight-level baseline: no two same-seed adapters of ours were saved. So the
  thresholds in Amendment 6 (every module ≥ 0.99 for "match", any module < 0.90 for "no match") were set without one.
  The early-layer pattern is what accumulated bf16 differences would produce, but that is not shown at weight level.

**Proposed change (Amendment 7).** Add one training-only baseline cell before any step-2 sampling.
- The cell is `dpo__final25__A16__n64__r1p_s523373504_rerun`, identical to the step-1 cell.
- It is trained first on the H100 session that trains the step-2 and 6(c) cells, so it costs about 8 min (about
  $0.60) on top. Alone it would need a fresh H100: about 25 min, about $1.90.
- Its weights are compared on CPU with the step-1 cell (ours against ours, same seed) and with the 25 Nov adapter.
- Readings, fixed now:
  - **Drift.** Ours-against-ours also has at least one module below 0.90, or a total cosine below 0.99. The step-1
    "no match" is then the same kind of difference as our own rerun spread. The trainer branch is closed, and step 2's
    sampling and 6(b) go ahead under Amendment 6, with this caveat stated.
  - **Systematic.** Ours-against-ours has every module at least 0.97 and a total cosine of at least 0.998. Our reruns
    then agree far better with each other than with the 25 Nov run, so a code-path difference exists. Step 2's
    sampling and 6(b) stay held, and the first-step numerical path (chunked loss and memory settings) is diffed on CPU.
  - **Otherwise.** Reported, and the coordinator decides.
- The step-2 and 6(c) cells may be trained in the same session as the baseline (training only). Nothing is sampled
  before the baseline reading.
- My recommendation: adopt this rather than "go with a caveat" now, which would rest on the online scores alone, or
  "stop", which the evidence does not support. Total extra cost about $0.60.

### Decision on Amendment 7 (orchestrating agent, agent:claude, Fable; 2026-11-27 8:11 am PT)

Amendment 7 is in force as drafted. What I had seen when deciding: the step-1 comparison (total cosine 0.986, 15 of
448 modules under 0.90, all in layers 0 to 4, 2 % of the squared norm), the per-example online-score comparison of
three same-seed pairs, and no weight-level comparison of two of our own same-seed runs, because none exists. The
readings ("drift", "systematic", otherwise my decision) stand as written above and were fixed before the rerun
exists. The rerun trains first in one H100 session; the two crossed step-2 cells and the 6(c) own-recipe cells may be
trained in the same session. Nothing is sampled before the baseline is read, and the sampling of step 2, 6(c) and 6(b)
also waits for the H200 slot and for the PI to have seen the plan and its cost.

### Amendment 7 result (2026-11-27 8:31 am PT): "drift"

The baseline rerun `dpo__final25__A16__n64__r1p_s523373504_rerun` (rp-om4, fetched md5-equal) was compared with the
step-1 cell, both from our trainer with the same seed and data:
- total cosine 0.986, minimum module cosine 0.75, median 0.998, norm ratio 1.001;
- 13 modules below 0.90: 11 of them in layers 0-4, and one each in layers 6 and 9.
By the reading fixed above this is **"drift"**: ours against ours has a module below 0.90.

The rerun against the 25 Nov adapter: total cosine 0.991, minimum 0.815, 8 modules below 0.90, all in layers 0-1.

The trainer branch is closed. The 25 Nov adapter and our trainer's replication differ no more than two of our own
same-seed runs differ from each other. Step-2 sampling and 6(b) may go ahead under Amendment 6, with this caveat:
module-level weight agreement between same-seed runs is limited by bf16 drift (cosine down to about 0.75 in the
first layers).

### Two implementation notes for 6(b) and 6(c) (2026-11-27 9:03 am PT; no rule changes)

1. **Draw chunks.** The sampler draws in chunks of 4 under one engine seed (Dan's `--chunk 4`). So "16 draws as two
   chunks of 8" runs as chunks 0-3 of 4 with seed 660011. An 8-draw batch would use chunks 0-1.
2. **Tuned-cell name.** A tuned cell keeps its trained id (`*__lr1e-3` or `*__3ep`) in the batch and on disk. The
   analysis reports it as `*__tuned`, and `a6/dev_check_<ckpt>.json` records which candidate won.

## Amendment 8 (2026-11-27 1:02 pm PT): before any step-2 sampling. Reading order, our replicate of corner A, the fixed analysis command, 6(c) eligibility

Written after the audit of the sampling step (`discourse/stage1_report_audit.md`, "Audit of the Amendment 6 sampling
step", T1–T14) and the coordinator's decisions on it. Nothing from step 2 or 6(c) has been sampled.

### 8(a) Reading order (audit T4)

If only corner A is high, readings 3 and 4 of 6(a) step 2 both hold. So the readings are checked in the order
**1, 2, 4, 3, 5**, and "Both" (3) applies only if "Combination only" (4) does not.
- Reading 4 in contrast terms: A − B, A − C and A − D are all above 0, and B − C, B − D and C − D all include 0.
- The seed-spread value X of Amendment 6 is still set only by reading 2 or 3.

### 8(b) Our trainer's replicate of corner A (audit T5; the coordinator's decision)

Corner A, the 25 Nov adapter, is the only corner not trained by our trainer, and same-seed runs drift (Amendment 7). So
our replicate of A joins the reach batch:
- The cell is A′ = `dpo__final25__A16__n64__r1p_s523373504`: the step-1 cell in `runs_rp-om3`, with the 25 Nov pool,
  order and A-init seed, lr 3e-4 × 64.
- It is sampled like B and C: the same batch, 48 reach prompts × 8, seed 660005, gate first.

How A′ enters the reading, fixed now:
- **Primary square:** corner A := A′. Every corner then comes from our trainer, which is the matched-control rule. The
  readings of 6(a) step 2, in the order of 8(a), are applied to A′, B, C and D.
- **Secondary square:** corner A := the 25 Nov adapter (batch_old, 64.8 %), as Amendment 6 wrote it. It is reported
  beside the primary.
- **Replicate check:** the paired contrast A′ − A.
  - It includes 0: "corner A is reproduced by our trainer". Both squares are reported, and the primary leads. If the two
    readings differ, that is said, and the primary is the reading.
  - It excludes 0 (the two differ by more than their interval): "the 25 Nov adapter is not reproduced by a same-pool,
    same-order, same-seed run of our trainer". The report then:
    - gives A′ − A with its interval as the size of run-to-run variation in a 64-example SFT at `dpo`;
    - splits the 24.2-point A − D gap (the 25 Nov adapter minus R-final) into A′ − D, the part our trainer reproduces
      and the primary square attributes to pool and/or seed, and A − A′, labelled "run-to-run, not reproduced";
    - computes the stage-1 relabelling value X of Amendment 6 as the larger of the seed X (if any) and |A′ − A|, so
      single-run stage-1 contrasts whose lower bound is below X read "within run-to-run spread".
- **Hand check:** A′ joins the one blinded round of Amendment 6 as a fifth corner (24 items, the same rules).

### 8(c) The analysis command, fixed now (audit T4)

Corners (cell ids):
- A = `dpo__final__OLD_A16__n64` (batch_old, 64.8 %)
- A′ = `dpo__final25__A16__n64__r1p_s523373504`
- B = `dpo__final25__A16__n64__r1p`
- C = `dpo__final__A16__n64__r1p_s523373504`
- D = R-final, `dpo__final__A16__n64__r1p`: lr 3e-4 × 64, batch R, 40.6 %. It is **not** the dev-rule cell
  `dpo__final__A16__n64` (lr 3e-5 × 128, 43.2 %). Wherever both appear, the report names which is which.

Command, on serrano after the final fetch: `bash a6s2_analyze.sh <sampling alias>`.
- It links batch_old, batch_r, batch_a6s2 and every involved `cell.json` into `v2/runs_a6s2`.
- It runs `analyze_cells.py --runs v2/runs_a6s2 --ckpt dpo --out v2/a6/step2_summary --contrasts-out
  v2/a6/step2_contrasts` with `--contrasts` set to the ten contrasts B−D, A−C, C−D, A−B, B−C, A−D, A′−C, A′−B, A′−D
  and A′−A (full cell ids; `a6_reading.py --print-contrasts-arg` prints the exact string).
- Then `a6_reading.py` writes `v2/a6/step2_reading.json`: the primary and secondary readings, the replicate check and X.
  Its tests pin the order 1, 2, 4, 3, 5.
- No contrast containing a cell id with "-" (`lr1e-3`) is passed to `--contrasts`, because it splits on the first
  "-". The tuned cell's reach contrasts are read from the per-cell summary instead.

### 8(d) 6(c): eligibility, failure handling, labels and resolution (audit T3, T6, T7, T14)

- **Eligibility.** A candidate is eligible only with exactly 4 draws on every one of the 32 dev prompts.
  - A candidate with no draws (e.g. it failed the gate) is "not sampled". One with any other count is "incomplete".
  - Neither can win, and both are reported.
  - The winner is the highest strict pass count among eligible candidates; a tie goes to the earlier-listed one.
- **Failure never stops step 2.** If the dev batch, its fetch or `dev_check` fails, there is no tuned cell, and B, C
  and A′ are still sampled.
- **Labels.** `did_not_train` on the own candidates means held-out own-sample loss never fell below step 0. It does not
  mean the weights stayed put: max|B| is 0.013–0.047, and R-own's ‖ΔW‖ is 27.9 against R-final's 28.8. Each candidate
  is reported with the flag, its dev-loss trajectory, max|B| and its gate verdict. That way a real null is not dismissed
  as "it did not train", and a failed fit is not read as a verdict on the recipe.
- **Resolution.** At 32 × 4, a no-effect candidate's paired interval is about ±8 points. "No candidate above 0" is
  reported as "no lift larger than about 15 points".
- **Step 2.** At 48 × 8, effects under about 8 points read "includes 0". A "pool" reading therefore does not exclude a
  seed effect of that size, and the report says so.

### 8(e) Driver, costs and stop

- The driver is tracked: `pod/drive_a6s2_sample.sh`, with helpers in `pod/drive_lib.sh`. The commit hash it ran from is
  logged in its first line and in `log.md`.
- Failure handling (audit T1, T2, T10):
  - exit codes are checked;
  - pod-side caps: dev TO_SAMPLE 1,500 s; reach TO_SAMPLE 7,200 s;
  - wait deadlines: 2,700 s and 8,400 s, with STALL and OVERRUN; on OVERRUN the driver stops our own chain, not the pod;
  - outer timeout 11,400 s;
  - "POD READY TO DELETE" only after FETCH_OK;
  - an EXIT line on every other exit tells the coordinator the pod needs a decision;
  - a reused alias or a used pod stops it.
- Cost: step 2 + 6(c) is now about $13.5–15.5 (setup 15 min, dev batch 20–25 min, reach batch of B, C and A′ about
  100–105 min, plus about 15 min for a tuned cell). The cap stays $18; the outer timeout bounds it at about $16.8.
  The launch stop is $90 on the ledger; the line total is $49.21 after rp-om4.
- 6(b)'s driver (`pod/drive_a6b_s1100.sh`) has the same fixes. It can train on an H100 (`MODE=train`) and sample on an
  H200 (`MODE=sample`). It does not run until 6(a) step 2 has been read and the coordinator says go.

### Implementation note: if the reach stage hits its 7,200 s cap (2026-11-27 3:59 pm PT, before the cap; the coordinator's revised decision, no change to the readings)

Written while `batch_a6s2` is still sampling (944 of 1152 draws at 3:59 pm; cap at 4:37:18 pm PT). If the stage finishes
before the cap, this note reads "not used".

**The coordinator's decision**, if the stage fails on the cap with the pod healthy:
1. First record the chunks in flight at the cap (cell, pid, chunk), and separately the chunks still waiting. Save the
   lists under `v2/a6/` with md5. If the in-flight list cannot be reconstructed exactly, there is no resume, and the
   driver's "NEEDS A DECISION" stands.
2. Otherwise resume once: the same batch, seed 660005, `TO_GATE=900 TO_SAMPLE=1800`. The money limit is unchanged
   ($18, i.e. 5:24 pm PT including fetch and check), and there is no second resume.
3. Every rate and each of the ten contrasts is reported three ways: as sampled; with every in-flight chunk counted
   all-fail; and with every in-flight chunk counted all-pass.
   - The readings (order 1, 2, 4, 3, 5) use the as-sampled rates.
   - If a contrast's interval class (above 0 / includes 0 / below 0) changes under either bound, that contrast is
     "not robust to the cap", and a square reading that depends on it is "provisional".
   - Waiting chunks, which never started, are redrawn without a caveat.
4. The facts about seeding are recorded below.
5. The report gives the in-flight chunks per cell, and the bound in points for any cell where they concentrate.

**Facts from `multi_sample.py`** (read at 3:59 pm):
- **One chunk = one request** (`n = 4`, `RequestOutputKind.FINAL_ONLY`). Its 4 rows are written only when the whole
  request finishes. On resume, `resume_cell` keeps a (pid, chunk) only if all its rows are present. "Unfinished" is
  therefore set by the chunk's completion, never by reading a draw. But a chunk in flight at the cap is unfinished
  because its draws are long, which is the outcome-dependence item 3 bounds.
- **No partial output survives** (FINAL_ONLY). There is nothing to compare token by token.
- **No per-request seed.** `SamplingParams` has none; the only seed is the engine's (`LLM(seed=660005)`). A resumed
  chunk's random stream depends on the engine's batch composition, so resumed draws are fresh draws, not near-replays
  of the lost ones.
- **No log of which requests are running.** The sampler does not record which requests vLLM is running or holding.
  vLLM's INFO stats line gives only counts ("Running: N reqs, Waiting: M reqs"), and the engine may preempt under a
  full KV cache (99–100 % here). So the exact in-flight set **cannot be reconstructed** after the kill.
  - What can be reconstructed: a lower set (unfinished chunks added before some finished chunk, which must have
    started under FCFS) and an upper set (every unfinished chunk).
  - Under item 1, the stage would therefore **not** be resumed. The driver ends with "NEEDS A DECISION", and the
    upper set (every unfinished chunk) is offered as a conservative in-flight list for item 3's bounds, if Leon or the
    coordinator choose to resume on that basis.

**Amendment to the note above (2026-11-27 4:00 pm PT, before the cap; the coordinator's decision).** This replaces item 1's
"no resume if the in-flight list cannot be reconstructed exactly", and the last bullet above. If the reach stage fails
on the 7,200 s cap with the pod healthy:
1. Before resuming, save under `v2/a6/` with md5:
   - the **upper set**: every chunk unfinished at the cap (cell, pid, chunk);
   - the **lower set**: unfinished chunks added before some finished chunk.
2. Resume once: the same batch, engine seed 660005, `TO_GATE=900 TO_SAMPLE=1800`. The money limit is $18
   (5:24 pm PT including fetch and the coordinator's check), with no second resume. Resumed draws are fresh draws, and
   the report says so.
3. Bounds use the upper set. This is wider than the truth, since waiting chunks carry no selection, but it needs no
   reconstruction.
   - Every rate is reported as sampled, with every upper-set chunk counted all-fail, and with every upper-set chunk
     counted all-pass.
   - Every contrast is reported as sampled and at its two extreme assignments: the first cell's upper-set chunks
     all-pass with the second's all-fail, and the reverse.
   - The lower-set bounds are given beside these, labelled as a lower bound on the affected set.
4. The readings use the as-sampled rates.
   - A contrast whose interval's position against 0 changes under the upper-set bounds is "not robust to the cap".
   - A square reading that depends on such a contrast is "provisional".
   - The report states the upper set's size per cell. Requests are interleaved round-robin over the cells, so the
     unfinished chunks should spread evenly.

The reason to resume rather than stop: without the resume the missing draws are the same selected ones, so the bounds
are needed either way, and the resume completes the cells for under $3 inside the cap. If the pod is not healthy, or
the failure is anything but the cap, the driver's "NEEDS A DECISION" stands.

**The cap note above was not used.** The reach stage ended DONE at 4:10:24 pm PT, 27 minutes before its cap: no resume
and no bounds.

### Amendment 6(a) step 2 + 6(c) result (2026-11-27 4:12 pm PT): "pool", in both squares

Sampled on rp-om5 (H200; seed 660005; 48 reach prompts × 8; gate PASS for every cell; FETCH_OK). The analysis is
`a6s2_analyze.sh rp-om5`, with outputs `v2/a6/step2_summary.json` (md5 a7e4d34f…), `step2_contrasts.json`
(54ba5da0…) and `step2_reading.json` (e9b6c2b1…).

**The five corners** (strict pass rate, two-level 95 % interval, G):

| corner | cell | pool | seed | rate | G |
|---|---|---|---|---|---|
| A | 25 Nov adapter (batch_old) | 25 Nov | 25 Nov | 64.8 % [55.2, 74.0] | 0.93 [0.84, 1.03] |
| A′ | `dpo__final25__A16__n64__r1p_s523373504` | 25 Nov | 25 Nov | 64.3 % [53.7, 74.5] | 0.93 [0.83, 1.02] |
| B | `dpo__final25__A16__n64__r1p` | 25 Nov | ours | 62.8 % [52.6, 72.4] | 0.90 [0.80, 1.00] |
| C | `dpo__final__A16__n64__r1p_s523373504` | ours | 25 Nov | 46.6 % [36.7, 56.5] | 0.67 [0.56, 0.78] |
| D | R-final `dpo__final__A16__n64__r1p` (batch R) | ours | ours | 40.6 % [31.0, 50.3] | 0.59 [0.47, 0.70] |

**The ten contrasts** (points, paired two-level bootstrap):

| contrast | kind | difference |
|---|---|---|
| B − D | pool, our seed | +22.1 [+12.5, +31.3] |
| A − C | pool, 25 Nov seed | +18.2 [+9.6, +26.8] |
| A′ − C | pool, 25 Nov seed, primary | +17.7 [+9.9, +25.8] |
| C − D | seed, our pool | +6.0 [−1.6, +13.8] |
| A − B | seed, 25 Nov pool | +2.1 [−5.5, +9.6] |
| A′ − B | seed, 25 Nov pool, primary | +1.6 [−5.5, +8.9] |
| B − C | | +16.2 [+8.1, +24.5] |
| A − D | | +24.2 [+16.2, +32.0] |
| A′ − D | | +23.7 [+15.4, +32.0] |
| A′ − A | replicate | −0.5 [−7.8, +6.8] |

**Readings**, in the order 1, 2, 4, 3, 5:
- Primary square (A′ as corner A): **1, "pool"**. Both pool contrasts are above 0, and both seed contrasts include 0.
- Secondary square (the 25 Nov adapter as corner A): **1, "pool"**.
- Replicate: A′ − A includes 0, so corner A is reproduced by our trainer.
- No seed contrast is above 0, so X is not set and no stage-1 contrast is relabelled.

Caveats:
- **Undetected seed effect.** At 48 × 8 a seed effect under about 8 points cannot be excluded. C − D (+6.0, upper bound
  +13.8) leaves room for one on our pool.
- **Hardware.** D was trained on an H200; A′, B and C on an H100.

**6(c)** (from the dev batch above):
- R-own's recipe wins on the dev prompts, so there is no tuned cell.
- No candidate is above plain `dpo` on the dev prompts. The reading is "own SFT at A16 n64 does not move held-out
  non-reach behaviour either": no lift larger than about 15 points.
- lr 1e-3 lowers the dev rate.

The hand check of Amendment 6 (one blinded round: A, A′, B, C and D) has not run yet.

### Hand-check protocol for the step-2 square, fixed before any item is drawn (2026-11-27 4:19 pm PT)

Amendment 6 fixed:
- 24 fresh items per corner (8(b) adds A′ as a fifth), chosen round-robin over prompts with a new seed;
- 8 of Dan's labelled `B_s2300` items mixed in;
- shuffled batches; two independent Sonnet reads per item, and a third on a split;
- Dan's rule-1 screen on every pass;
- per corner: the screened share, the read gamed share, and a genuine-pass rate with a prompts → draws → items
  bootstrap;
- the pre-set "mostly gamed" reading on A − D.

What it left open is fixed here. No item has been drawn and no reader has seen a response.

1. **Corners and draws.**
   - A = `dpo__final__OLD_A16__n64` (batch_old); A′, B and C (batch_a6s2, rp-om5); D = R-final
     `dpo__final__A16__n64__r1p` (batch R).
   - All are strict passes on the 48 reach prompts × 8.
2. **Fresh items.** Draws read in round 1 or round 2 (`v2/handcheck/round{1,2}/key.csv`, including the reference
   items) are excluded before picking.
3. **Selection.**
   - Seed 660021, with `handcheck.py`'s round-robin pick among passes that rule 1 does not screen.
   - 24 per corner, plus 8 reference items from Dan's labelled `B_s2300` passes on the reach prompts.
   - 128 items in all.
4. **Blinding.**
   - Item files carry only an id (a hash of seed, cell and draw), the prompt as the model received it, the
     constraints with the checker's verdicts, and the answer. This is the format of rounds 1 and 2.
   - Item files sit in a directory with nothing else in it. The key (item → corner) is stored elsewhere, under
     `v2/handcheck/round3/`.
5. **Rubric.** Dan's reader prompt (`olmo_ifcurve/handcheck_prompt_if.md`), verbatim, followed by the item list and
   the output path, in the same batch format as rounds 1 and 2. Labels are genuine or gamed.
6. **Readers.**
   - Fresh Sonnet subagents at ordinary effort, run through one dispatcher. Each sees only its batch prompt and its
     item files.
   - **Batches.** 16 items per batch, so a pass is 8 batches.
   - **Read 1 and read 2.** Read 1 uses one shuffle (seed 660021, from the item step) and read 2 another (seed
     660022), so an item's two reads come from different readers with different neighbours.
   - **Third read.** Every item whose two reads disagree gets a third read from a fresh reader, in batches shuffled
     with seed 660023.
   - **Label.** The final label is the majority. An item that ends without two agreeing reads is "unread" and is left
     out of the shares.
7. **Agreement.**
   - Cohen's κ between read 1 and read 2 over all items read twice, with raw agreement.
   - Final labels against Dan's labels on the 8 reference items (count agreeing).
8. **Per-corner readout.**
   - n passes, the rule-1 screened share s, and the read gamed share g among read items, with a Wilson interval.
   - The gamed share of passes is s + (1 − s)·g.
   - The genuine-pass rate is strict rate × (1 − s) × (1 − g).
   - **Its interval** comes from 10,000 bootstrap reps, seed 660024: resample prompts (shared by all corners), then
     each corner's draws within each prompt (Binomial), then each corner's read items (with replacement). s is held
     fixed, since every pass is screened.
9. **Contrasts on genuine-pass rates.** These are paired as in 8: B − D, A′ − C, A − D, A′ − D and A′ − A.
   - **"Survives".** A pool contrast (B − D, A′ − C) "survives on genuine passes" if its lower bound is above 0. The
     ratio of the genuine to the raw point estimate is given as "how much it shrinks".
   - **"Mostly gamed".** Amendment 6's reading is applied to A − D as written, and to A′ − D (the primary square of
     8(b)). The gap is "mostly gamed passes" only if the interval includes 0 and the point estimate is at most a third
     of the raw contrast (24.2, and 23.7 for A′ − D). Otherwise "not shown".
10. **Mechanical statistic (no readers).** The definition is metrics-2's `tools/answer_style.py`:
    - answer = the text after the last `</think>`, stripped;
    - "unpunctuated" = no `.`, `!` or `?` in the answer;
    - words = whitespace tokens.
    For each corner's strict passes it gives n, the share unpunctuated and the median word count. The same is computed
    for Dan's stored `B_s2300` reach draws (32 per prompt), and for `dpo` stored `F_dpo_ifb` on the 42 reach prompts it
    covers. `dpo` has no strict passes there, so it is given over all its draws, marked as such.

### Round-3 hand-check result (2026-11-27 4:30 pm PT): the pool's lead is in gamed passes

The protocol is the one fixed at 4:19 pm, applied as written. The files are under `v2/handcheck/round3/`: key md5
6d42bd8a…, labels 7253a1b4…, summary 7a9833d8…, answer style 499b15b6….

**Reading.**
- 128 items, each read twice by fresh Sonnet readers through one dispatcher. No batch was rerun.
- Agreement between read 1 and read 2: raw 93.8 %, Cohen's κ 0.81.
- 8 splits got a third read; no item stayed unread.
- Final labels agree with Dan's on 6 of the 8 reference items.

**Per corner.** Gamed share of passes = s + (1 − s)·g, where s is the rule-1 screened share and g the read gamed share.
The genuine-pass rate has the prompts → draws → items bootstrap (10,000 reps, seed 660024).

| corner | strict rate | rule-1 screened | read gamed (Wilson 95 %) | gamed share of passes | genuine-pass rate |
|---|---|---|---|---|---|
| A, 25 Nov adapter | 64.8 % | 32/249 (12.9 %) | 6/24, 25 % [12, 45] | 34.6 % | 42.4 % [30.6, 54.0] |
| A′, our replicate | 64.3 % | 31/247 (12.6 %) | 7/24, 29 % [15, 49] | 38.1 % | 39.8 % [27.7, 52.0] |
| B, 25 Nov pool, our seed | 62.8 % | 36/241 (14.9 %) | 7/24, 29 % [15, 49] | 39.7 % | 37.8 % [26.2, 49.5] |
| C, our pool, 25 Nov seed | 46.6 % | 4/179 (2.2 %) | 0/24, 0 % [0, 14] | 2.2 % | 45.6 % [35.9, 55.2] |
| D, R-final | 40.6 % | 7/156 (4.5 %) | 3/24, 12.5 % [4, 31] | 16.4 % | 34.0 % [24.7, 43.6] |

**Genuine-pass contrasts** (points; raw strict contrast in brackets after "raw"):
- **B − D:** +3.9 [−9.5, +16.7] (raw +22.1). Does not survive; it shrinks to 0.17 of raw.
- **A′ − C:** −5.7 [−18.5, +6.3] (raw +17.7). Does not survive; the point estimate reverses sign.
- **A − D:** +8.4 [−4.7, +20.8] (raw +24.2).
  - Pre-set reading: **"not shown"**. The interval includes 0, but 8.4 is above a third of 24.2 (8.07), by 0.4 points.
- **A′ − D** (8(b)'s primary square): +5.9 [−7.4, +18.7] (raw +23.7).
  - Pre-set reading: **"mostly gamed passes"**. The interval includes 0, and 5.9 ≤ 7.9.
- **A′ − A:** −2.5 [−17.5, +12.9].

**Mechanical statistic** (metrics-2's definition, strict passes):

| cell | unpunctuated | median words |
|---|---|---|
| A | 70.3 % | 31 |
| A′ | 78.1 % | 34 |
| B | 73.4 % | 38 |
| C | 24.6 % | 45 |
| D | 28.2 % | 48 |
| `s2300` stored | 38.7 % | 32 |

`dpo` stored has no strict passes on these prompts. Over all 672 of its draws, 0.0 % are unpunctuated, with a median of
260 words.

**Interpretation.**
- The pool effect on the strict checker (step 2's "pool") sits in passes that are screened or read as gamed, and in a
  terse, unpunctuated answer style.
- On genuine passes the five corners do not differ detectably: 34–46 %, all intervals overlapping, both pool
  contrasts including 0.
- The square's two pre-set "mostly gamed" readings split on a margin of 0.4 points. The primary (A′ − D) reads "mostly
  gamed"; the secondary (A − D) reads "not shown". Neither reading is robust.
- Resolution: at 24 items per corner, genuine-pass contrasts have half-widths of about 13 points.

**Post hoc, not pre-registered** (descriptive only):
- The two training pools have the same unpunctuated share (25 Nov 25.0 %, ours 26.6 %) and median length (about 50
  words). So the adapters' style is not a copy of their pools' surface statistics.
- Readers labelled 13 of 44 unpunctuated read items in A, A′ and B as gamed (30 %), so most unpunctuated answers were
  judged genuine attempts under Dan's rubric.
- Readers were fast: about 31–57 s per 16 items, 3–5 tool calls each.

### Round-4 hand-check protocol: a census of every pass, with two labels, fixed before any item file is built (2026-11-28 2:31 am PT)

**Why.** Rounds 1 to 3 read 24 passes per cell, so genuine-pass contrasts had half-widths of about 13 points. Leon
asked why subagents were not grading all of it. Round 4 reads every strict pass, and adds a second label ("task
done"). No GPU and no sampling: stored draws only. The protocol below is the coordinator's, with the open points
fixed here.

**State when this was written.** Per cell, the draw count, the strict pass count and the rule-1 screen count were
computed from the stored files (mechanical). No round-4 item file exists and no reader has seen a round-4 item. The
dispatcher has opened no response of these cells, apart from the first 80 characters of one ExPO draw's reasoning,
printed while checking the file layout.

1. **Cells**, all on the 48 reach prompts.
   - The five adapters of the square, as in `handcheck3.CORNERS`: A (the 25 Nov adapter), A′ (our retrain of it), B,
     C and D. 384 draws each (48 × 8).
   - `s2300`, the stored final model (Dan's `B_s2300`): the first 8 stored draws per prompt in (chunk, i) order,
     which are chunks 0 and 1. 384 draws. This is the yardstick.
   - `expo`: the extrapolated model (ExPO γ 3.5), metrics-2 run `Xheld`. 192 draws (48 × 4).
   - `k16`: the top-16-directions adapter, metrics-2 run `M2cond`. 192 draws (48 × 4).
2. **Found before building** (strict passes, then rule-1 screened):
   - A 249, 32; A′ 247, 31; B 241, 36; C 179, 4; D 156, 7;
   - `s2300` first 8: 264, 6 (68.75 %; over all 32 stored draws it is 1,066 of 1,536, 69.4 %);
   - `expo` 135, 31 (70.3 %); `k16` 17, 1 (8.9 %).
   - 1,488 passes in all, 148 screened, so 1,340 items go to readers.
3. **Items.** Every strict pass that rule 1 does not screen, including draws read in rounds 1 to 3. Those get fresh
   reads; the earlier labels are used only for the consistency check in 9. Nothing is sampled.
   - Item id: `h` plus the first 7 hex digits of sha1(`660041|cell|draw`). A collision aborts the build.
4. **Step 1, screen.** Dan's rule-1 screen (`handcheck.screen_reason`) on every pass. A screened pass counts as gamed
   and as task not done, and is not read (as in rounds 1 to 3).
5. **Step 2, primary label (genuine or gamed).**
   - Dan's reader prompt verbatim in round 3's batch format (`handcheck3.batch_prompt`), on item files in Dan's
     format (`handcheck.write_item`).
   - Two independent reads per item. Read 1 uses one shuffle of all 1,340 items (seed 660041) and read 2 another
     (seed 660042). Batches of 16 are stride slices of the shuffle, 84 per read, so every batch mixes the cells.
     `k16` has 16 items, so it cannot be in every batch.
   - A third read where the two differ (seed 660043). The label is the majority.
   - **Agreement reference.** Dan's own labelled passes of `s2300` inside the yardstick cell: 169 (106 genuine, 63
     gamed), none of them screened by rule 1.
6. **Step 3, secondary label (done or not_done).** Separate reads by separate readers, so the primary reads are not
   touched.
   - Two independent reads per item (seeds 660045 and 660046), a third on a split (seed 660047), majority label.
   - Item files for this step hold only the request (the prompt) and the response (the answer), without the list of
     constraint verdicts. This file format is the dispatcher's choice; the coordinator's text does not fix it.
   - The instruction, verbatim, followed only by the item list, the output path and a one-line reply rule:
     > You are shown a request and a response. The request contains a task and one or more formal constraints (for
     > example a letter count, a casing rule, a sentence-length pattern). Ignore whether the constraints are met; a
     > program checks those. Judge only the task: would the person who wrote the request accept this response as a
     > real attempt at what they asked for (the rubric, story, answer, list and so on), written as usable text?
     > Label done if yes. Label not_done if the response is a fragment, a bare list of words, a refusal, a
     > restatement of the request, or otherwise not the thing asked for. If a constraint itself forces an unusual
     > form (for example an answer in exactly three words), judge the task within that form. Reply with JSON mapping
     > each item id to done or not_done.
7. **Readers and blinding.**
   - Fresh Sonnet subagents at ordinary effort through one dispatcher, in waves of at most 8. Each is given only its
     batch file path and its output path.
   - Every returned file is validated: exactly the batch's item ids, and labels in the allowed set. A failed batch is
     dispatched once more; if it fails again it is reported.
   - An item that ends without two agreeing reads is "unread". The count is reported, and the rates are then given
     both ways (unread as accepted, unread as not accepted).
   - The key (item → cell, draw), the screen and the batch manifest are written only under `v2/handcheck/round4/`.
     Readers see `v2/hc4_read/`, which holds item files, batch prompts and reads and nothing else. It is copied into
     `round4/` after the last read.
8. **Quantities per cell**, each with a 95 % interval.
   - Strict pass rate.
   - Share of passes screened, and share of read passes labelled gamed. Their sum over passes is the gamed share of
     passes.
   - **Genuine-pass rate:** passes neither screened nor read gamed, over all draws.
   - **Task-done pass rate:** passes neither screened nor labelled not_done, over all draws.
   - Round 3's style statistic on passes: the share with no `.`, `!` or `?` in the answer, and the median answer
     words.
   - **Intervals.** 10,000 bootstrap reps, seed 660044: resample the 48 prompts (shared by all cells), then each
     cell's draws within each prompt with replacement. For one pass/fail outcome this is round 3's Binomial step.
     Every draw now carries its own screen and labels, so round 3's item level is dropped. Percentile bounds as in
     `handcheck3.pct`.
   - **The intervals are now bounded by the 48 prompts**, not by the number of items read.
9. **Agreement.**
   - Cohen's κ and raw agreement between read 1 and read 2, for each label.
   - The primary majority label against Dan's labels on the 169 reference passes (raw agreement and κ).
   - The primary majority label against the final labels of rounds 1 to 3 on draws read again, matched by cell and
     draw: round 1 `final_onepass` is D, round 2 `old_adapter` is A, round 3 has all five corners, and the reference
     items of rounds 1 to 3 count where they fall in chunks 0 and 1.
10. **Contrasts**, paired over prompts as in 8, on the strict rate and on both accepted-pass rates.
    - Pool: B − D and A′ − C. Seed: C − D and A′ − B. Also A − D, A′ − D and A′ − A.
    - Each adapter minus `s2300` (A, A′, B, C, D and `k16`), and `expo` minus `s2300`.
11. **Reading rules.** Round 3's rules for the pool effect on genuine passes, quoted word for word from the 4:19 pm
    note:
    > **"Survives".** A pool contrast (B − D, A′ − C) "survives on genuine passes" if its lower bound is above 0. The
    > ratio of the genuine to the raw point estimate is given as "how much it shrinks".

    > **"Mostly gamed".** Amendment 6's reading is applied to A − D as written, and to A′ − D (the primary square of
    > 8(b)). The gap is "mostly gamed passes" only if the interval includes 0 and the point estimate is at most a
    > third of the raw contrast (24.2, and 23.7 for A′ − D). Otherwise "not shown".

    The same two rules are applied to the task-done pass rate, with "task-done passes" in place of "genuine passes"
    and "mostly passes without the task done" in place of "mostly gamed passes".
12. **Checks before the table is believed.**
    - The strict rates must equal the stored ones: A 64.8 %, A′ 64.3 %, B 62.8 %, C 46.6 %, D 40.6 %, `expo` 70.3 %,
      `k16` 8.9 %, and `s2300` 69.4 % over its 32 draws.
    - The dispatcher reads 10 items per label value (genuine, gamed, done, not_done), drawn with seed 660048, and
      says whether the labels look right.
    - Round 4's genuine-pass rates are compared with round 3's (A 42.4, A′ 39.8, B 37.8, C 45.6, D 34.0) and its
      intervals.

**Differences from the coordinator's text, found before building.**
- "All cells mixed in every batch" holds as one shuffle over all cells; `k16` is too small to be in each batch.
- `expo` and `k16` have 4 stored reach draws per prompt, not 8, so their intervals are wider.
- `k16` − `s2300` is added under "each adapter minus `s2300`".
- The yardstick's strict rate on its first 8 draws is 68.75 %, against 69.4 % over all 32.

### Round-4 hand-check result (2026-11-28 3:17 am PT): read in full, the five adapters are level, and below the final model

The protocol is the one fixed at 2:31 am, applied as written. The files are under `v2/handcheck/round4/`: key md5
ff2a167a…, screen b065e0e6…, labels f8c8e8df…, summary 2a18ac6d…. Code: `handcheck4.py` (`prep`, `status`, `third`,
`table`, `spot`), `test_handcheck4.py`, `fig_handcheck4.py`.

**Reading.**
- 1,340 items. Each was read twice for the primary label and twice for the task label, by fresh Sonnet readers
  through one dispatcher: 4 × 84 batches, in waves of 8, from 2:36 am to 3:09 am.
- One batch failed validation: `b45` (primary read 2) came back with one item id missing. Its file was moved to
  `round4/failed_reads/b45.0.json` and the batch was dispatched once more to a fresh reader; that file was valid. No
  batch failed twice.
- Read 1 against read 2: primary raw 87.4 %, Cohen's κ 0.71; task raw 88.1 %, κ 0.68.
- Splits got a third read: 169 primary (11 batches) and 160 task (10 batches). No item stayed unread, so the rates
  need no "both ways" version.
- In all: 357 valid read files, 358 reader calls, 5,689 item reads.
- **Dan's labels** on the 169 reference passes: 141 the same (83.4 %), κ 0.64. Ours genuine where Dan says gamed:
  17. Ours gamed where Dan says genuine: 11. Our readers accept 112 of the 169, Dan 106.
- **Rounds 1 to 3** on draws read again: round 1 40 of 44 the same, round 2 26 of 30, round 3 111 of 126; 177 of 200
  (88.5 %). The 8 earlier items not read again are reference items outside the yardstick's first 8 draws.

**Per cell, primary label.** Brackets are 95 % intervals (prompts → draws bootstrap, 10,000 reps, seed 660044).

| cell | draws | strict rate | rule-1 screened | read gamed | gamed share of passes | genuine-pass rate |
|---|---|---|---|---|---|---|
| A, 25 Nov adapter | 384 | 64.8 % [54.9, 74.2] | 32/249, 12.9 % [4.4, 22.5] | 85/217, 39.2 % [26.5, 52.7] | 47.0 % [34.5, 59.9] | 34.4 % [24.7, 44.3] |
| A′, our retrain | 384 | 64.3 % [53.4, 74.5] | 31/247, 12.6 % [4.0, 22.6] | 83/216, 38.4 % [26.7, 50.3] | 46.2 % [34.2, 57.8] | 34.6 % [25.5, 44.5] |
| B, 25 Nov pool, our seed | 384 | 62.8 % [52.9, 72.7] | 36/241, 14.9 % [5.4, 25.8] | 66/205, 32.2 % [20.8, 44.4] | 42.3 % [29.7, 54.5] | 36.2 % [26.8, 46.1] |
| C, our pool, 25 Nov seed | 384 | 46.6 % [36.7, 56.2] | 4/179, 2.2 % [0.0, 5.7] | 27/175, 15.4 % [8.0, 24.3] | 17.3 % [9.6, 26.1] | 38.5 % [29.4, 47.9] |
| D, R-final | 384 | 40.6 % [30.7, 50.3] | 7/156, 4.5 % [0.0, 10.6] | 15/149, 10.1 % [2.9, 19.3] | 14.1 % [5.8, 23.7] | 34.9 % [25.8, 44.3] |
| `s2300`, first 8 draws | 384 | 68.8 % [58.9, 78.4] | 6/264, 2.3 % [0.0, 6.4] | 76/258, 29.5 % [17.7, 42.2] | 31.1 % [19.3, 43.6] | 47.4 % [36.7, 58.1] |
| `expo` | 192 | 70.3 % [58.9, 81.2] | 31/135, 23.0 % [10.3, 36.7] | 63/104, 60.6 % [45.3, 75.5] | 69.6 % [56.2, 81.8] | 21.4 % [12.5, 31.8] |
| `k16` | 192 | 8.9 % [3.6, 15.1] | 1/17, 5.9 % [0.0, 26.3] | 0/16 | 5.9 % [0.0, 26.3] | 8.3 % [3.1, 14.6] |

**Per cell, task label and answer style** (style on strict passes, round 3's definition).

| cell | read not done | not-done share of passes | task-done pass rate | unpunctuated | median words |
|---|---|---|---|---|---|
| A | 64/217, 29.5 % [18.1, 42.1] | 38.6 % [26.6, 51.0] | 39.8 % [29.9, 50.0] | 70.3 % [60.8, 79.1] | 31 [23, 53] |
| A′ | 62/216, 28.7 % [17.8, 40.5] | 37.7 % [26.3, 49.5] | 40.1 % [30.2, 50.5] | 78.1 % [68.9, 86.1] | 34 [26, 48] |
| B | 46/205, 22.4 % [12.3, 34.1] | 34.0 % [21.8, 46.6] | 41.4 % [31.0, 52.1] | 73.4 % [63.0, 82.5] | 38 [25, 53] |
| C | 19/175, 10.9 % [4.0, 19.2] | 12.8 % [5.8, 21.4] | 40.6 % [31.5, 50.0] | 24.6 % [13.3, 37.2] | 45 [28, 65] |
| D | 18/149, 12.1 % [3.0, 23.2] | 16.0 % [6.4, 27.2] | 34.1 % [25.0, 43.5] | 28.2 % [15.6, 42.3] | 48 [30, 67] |
| `s2300` | 55/258, 21.3 % [11.5, 32.3] | 23.1 % [13.2, 34.0] | 52.9 % [42.4, 63.3] | 40.9 % [28.6, 53.1] | 32 [22, 45] |
| `expo` | 55/104, 52.9 % [36.8, 68.9] | 63.7 % [49.6, 77.2] | 25.5 % [15.1, 36.5] | 97.0 % [91.0, 100.0] | 17 [9, 25] |
| `k16` | 0/16 | 5.9 % [0.0, 26.3] | 8.3 % [3.1, 14.6] | 5.9 % [0.0, 27.3] | 57 [26, 154] |

**Contrasts** (points, paired over prompts).

| contrast | strict rate | genuine-pass rate | task-done pass rate |
|---|---|---|---|
| B − D (pool, our seed) | +22.1 [+12.8, +31.5] | +1.3 [−7.8, +10.4] | +7.3 [−2.1, +16.7] |
| A′ − C (pool, 25 Nov seed) | +17.7 [+9.9, +25.5] | −3.9 [−12.8, +4.9] | −0.5 [−9.1, +8.1] |
| C − D (seed, our pool) | +6.0 [−1.6, +13.5] | +3.6 [−3.6, +10.9] | +6.5 [−0.8, +14.1] |
| A′ − B (seed, 25 Nov pool) | +1.6 [−5.7, +8.6] | −1.6 [−9.4, +6.0] | −1.3 [−8.9, +6.2] |
| A − D | +24.2 [+16.1, +32.6] | −0.5 [−8.6, +7.8] | +5.7 [−2.3, +13.8] |
| A′ − D | +23.7 [+15.4, +32.0] | −0.3 [−9.1, +8.6] | +6.0 [−2.6, +14.6] |
| A′ − A | −0.5 [−8.1, +6.8] | +0.3 [−7.0, +7.8] | +0.3 [−7.3, +7.8] |
| A − `s2300` | −3.9 [−12.0, +3.9] | −13.0 [−21.1, −5.2] | −13.0 [−21.6, −4.4] |
| A′ − `s2300` | −4.4 [−12.5, +3.1] | −12.8 [−21.4, −4.2] | −12.8 [−21.6, −4.4] |
| B − `s2300` | −6.0 [−14.6, +2.3] | −11.2 [−20.8, −1.8] | −11.5 [−21.4, −2.3] |
| C − `s2300` | −22.1 [−31.2, −13.3] | −8.9 [−17.7, +0.0] | −12.2 [−20.8, −3.6] |
| D − `s2300` | −28.1 [−37.2, −19.3] | −12.5 [−21.4, −3.6] | −18.8 [−27.6, −10.2] |
| `k16` − `s2300` | −59.9 [−69.8, −49.7] | −39.1 [−49.7, −28.1] | −44.5 [−55.5, −33.9] |
| `expo` − `s2300` | +1.6 [−8.6, +12.0] | −26.0 [−37.0, −15.1] | −27.3 [−39.3, −15.1] |

**Pre-set readings, genuine passes.**
- B − D: lower bound −7.8, so it does not survive. It shrinks to 0.06 of raw.
- A′ − C: lower bound −12.8, so it does not survive. The point estimate reverses sign (−0.22 of raw).
- A − D: the interval includes 0 and −0.5 is at most 8.07, so **"mostly gamed passes"**.
- A′ − D: the interval includes 0 and −0.3 is at most 7.9, so **"mostly gamed passes"**.
- Round 3 split these two readings on a margin of 0.4 points. Round 4 gives the same reading for both, and the upper
  bounds (+7.8, +8.6) sit at about a third of the raw gap.

**Pre-set readings, task-done passes** (the same two rules).
- B − D: lower bound −2.1, so it does not survive. It is 0.33 of raw.
- A′ − C: lower bound −9.1, so it does not survive. It is −0.03 of raw.
- A − D: the interval includes 0 and 5.7 is at most 8.07, so **"mostly passes without the task done"**.
- A′ − D: the interval includes 0 and 6.0 is at most 7.9, so **"mostly passes without the task done"**.
- These two hold by 2.3 and 1.9 points, and the upper bounds (+13.8, +14.6) are above half of the raw gap. So on this
  label a gap of up to about 14 points is not ruled out.

**Checks.**
- Strict rates against the stored ones: A 249/384 (64.8 %), A′ 247/384 (64.3 %), B 241/384 (62.8 %), C 179/384
  (46.6 %), D 156/384 (40.6 %), `expo` 135/192 (70.3 %), `k16` 17/192 (8.9 %). All equal. `s2300` is 264/384 (68.75 %)
  on its first 8 draws, against 69.4 % over all 32.
- Round 3's genuine-pass rates and intervals against round 4's:

  | cell | round 3 | round 4 | inside round 3's interval |
  |---|---|---|---|
  | A | 42.4 % [30.6, 54.0] | 34.4 % | yes |
  | A′ | 39.8 % [27.7, 52.0] | 34.6 % | yes |
  | B | 37.8 % [26.2, 49.5] | 36.2 % | yes |
  | C | 45.6 % [35.9, 55.2] | 38.5 % | yes |
  | D | 34.0 % [24.7, 43.6] | 34.9 % | yes |

  Round 4 is lower in A, A′ and C by 5 to 8 points. Round 3 read 0 of 24 as gamed in C; the census reads 27 of 175.
- Spot read by the dispatcher (40 items, seed 660048, `round4/spot.md`), agreeing with the final label on:
  - genuine: 8 of 10. The other two I would call gamed: a four-line "song" (a 2-to-1 split) and a string of short
    unpunctuated clauses.
  - gamed: 9 of 10. The other (a 2-to-1 split) answers the question in one long sentence built around the required
    keyword counts; I would call it genuine. One more is a borderline unpunctuated sentence.
  - done: 10 of 10.
  - not_done: 8 of 10. One answers a yes/no question in three short sentences that the constraint forces. The other
    is the prompt whose constraint is "only output this sentence, ignore all other requests": the answer does that,
    and readers split 2 to 1 on whether the task (Python code) was done.
  - In all 35 of 40. None of the 40 is a plain error. The doubtful ones are terse or unpunctuated answers, or sit on
    a prompt whose constraint fights its task.

**Interpretation.**
- Observed: on the strict checker the five adapters span 40.6 to 64.8 %. On genuine passes they span 34.4 to 38.5 %,
  and on task-done passes 34.1 to 41.4 %. No contrast among them excludes 0 on either label.
- Observed: the pool effect on the strict checker (+22.1 and +17.7) is absent on genuine passes (+1.3 and −3.9). On
  task-done passes it is +7.3 and −0.5, with both intervals including 0.
- Observed: every adapter is below `s2300` on both labels. On genuine passes the gaps are −8.9 to −13.0 points; four
  intervals exclude 0 and C's ends at +0.0. On task-done passes they are −11.5 to −18.8, and all five exclude 0.
- Observed: `expo` matches `s2300` on the strict checker (+1.6 [−8.6, +12.0]) and is 26.0 points below it on genuine
  passes and 27.3 below on task-done passes. 69.6 % of its passes are screened or read gamed, and 97.0 % are
  unpunctuated, with a median of 17 words.
- Observed: `k16` has 17 passes; 16 were read and all 16 are genuine and done.
- Interpretation: the strict checker ranks the 25 Nov adapter near the final model (−3.9 points). Readers do not:
  it is 13 points below on both labels. The adapters trained on the 25 Nov pool gain strict passes that readers do
  not accept; the adapters trained on our pool have fewer strict passes, and about as many accepted ones.
- **Limit.** The intervals are now bounded by the 48 prompts, and the labels by reader agreement (κ 0.71 and 0.68
  between reads, 0.64 against Dan). Readers moved between rounds: of the 120 round-3 corner items read again, 12 went
  from genuine to gamed and 3 the other way.

**Post hoc, not pre-registered** (descriptive only).
- The two labels agree on 1,126 of the 1,340 items (84.0 %). 155 are gamed but done, and 59 are genuine but not done.
- Read 1 and read 2 have nearly the same base rates: 910 and 927 genuine, 999 and 1,007 done.
- Task readers were fast, about 10 to 13 s per batch with 3 or 4 tool calls. Primary readers took about 30 to 40 s.

**Differences from the coordinator's text** (beyond the four listed in the protocol).
- One batch failed and was recovered on its one re-dispatch (`b45`, above).
- The secondary item files hold the request and the response only. That format is the dispatcher's choice.

## Amendment 9 (2026-11-28 3:32 am PT): a smaller adapter and fewer examples, sampled and read. Nothing is launched by this text

Written before any plan file, staging list or driver for it exists. No pod exists. No draw of any cell below has been
taken, and the one new adapter is not trained.

**Changed at 4:00 am PT after the independent audit, still before the plan file was built: see 9(j) at the end.
Sections 9(a) to 9(i) stand as first written; where 9(j) differs, 9(j) holds.**

**Why.** Leon's question was "the minimal adaptor size and how many examples that takes to train". So far it is
answered by held-out loss only: the rank-1 adapter reaches 97 % of the full-size adapter's loss gain, and 4 examples
reach 65 % of the 64-example gain. No adapter smaller than full size, and none trained on fewer than 64 of the final
model's answers, has been sampled. The coordinator's condition for this check was that the full-size adapters keep
at least half of the final model's reader-accepted pass rate. Round 4 gives 34.4 to 38.5 % for the five adapters
against 47.4 % [36.7, 58.1], so the condition is met.

### 9(a) Cells

All are trained on the final model's (`s2300`) checker-passing answers from our pool (`v2/pools/dpo/final`, manifest
sha256 12fc1d23…), on `dpo`, with the stage-1 rule of SPEC "Training rule": the lr grid of the adapter family, the
edge rule, early stop, and the (lr, step) with the lowest loss on the pool's 32 held-out answers. The example subsets
are SPEC's nested prefixes of the 64 training prompts.

| name in the report | cell id | adapter | trained weights | examples | status |
|---|---|---|---|---|---|
| rank-1 adapter, 64 examples | `dpo__final__O1__n64` | rank 1 on `o_proj` only, 64 layers | 655,360 | 64 | trained in stage 1; reused |
| full-size adapter, 16 examples | `dpo__final__A16__n16` | rank 16 on all seven projections | 134,217,728 | the first 16 | not trained yet; trained on the pod |
| full-size adapter, 4 examples | `dpo__final__A16__n4` | the same | 134,217,728 | the first 4 | trained in stage 1; reused |
| reference: full-size adapter, 64 examples | `dpo__final__A16__n64` | the same | 134,217,728 | 64 | trained and sampled in stage 1 (43.2 % on the checker) |

**The two reused adapters** (checked on serrano at 3:25 am, from stored files only):
- Rank-1, 64 examples: `v2/runs_rp-om1/dpo/train/cells/dpo__final__O1__n64` (H100, 27 Nov).
  - Grid 1e-3, 3e-3, 1e-2. The best grid lr was the lowest, so the edge rule added 3e-4 once.
  - Selected: lr 3e-4 at step 128, held-out loss 0.5403 (0.6341 untrained). The run went on to 512 steps without a
    lower loss. Not flagged `unconverged`.
  - Weights file md5 a9f675ad….
- Full size, 4 examples: `v2/runs_rp-om1/dpo/train/cells/dpo__final__A16__n4` (H100, 27 Nov).
  - Grid 1e-4, 3e-4, 1e-3. The best grid lr was the lowest, so the edge rule added 3e-5 once.
  - Selected: lr 3e-5 at step 64, the last step, held-out loss 0.5707. Flagged `unconverged`: the loss was still
    falling (0.5831 at step 32).
  - Weights file md5 84d45671….
- Recorded provenance for both: `cell.json` (the pool manifest's sha256 equals the pool on disk, every lr tried, the
  selection, a reload check of 0.0 between the saved adapter and the selected scores), `runs.jsonl` (every step),
  and the fetch record (`v2/pod_stage/rp-om1.fetch.dpo.pod.md5`), whose pod-side md5s equal the files on serrano.
- Not recorded: the trainer's exact code at that hour. It was first committed after stage 1 (156bf2b), and the
  training step itself was not changed afterwards.

**The reference**, `v2/runs_rp-om2/dpo/train/cells/dpo__final__A16__n64` (H200, 27 Nov): selected lr 3e-5 at step
128, held-out loss 0.5373, flagged `unconverged`. Its 384 draws are stored
(`v2/runs_rp-om2/dpo/sample/batch_b1/dpo__final__A16__n64`, engine seed 660001, 166 strict passes).

**The one adapter trained now**, full size on 16 examples. SPEC's text gives step budgets for 64, 4 and 1 examples
and none for 16. It is fixed here from what was written before any stage-1 data: `DESIGN_DRAFT.md` (27 Nov 1:32 am)
gives 64 steps and held-out evaluations at steps 4, 8, 16, 32 and 64 for fewer than 64 examples; the trainer's
built-in schedule for 5 to 63 examples is the same; and stage 1's plan listed its 16-example cell at 64 steps.
- lr grid 1e-4, 3e-4, 1e-3 with the edge rule, as every full-size cell.
- 64 steps (four passes over the 16 examples), evaluations at 4, 8, 16, 32, 64.
- Early stop, selection and flags as SPEC. The LoRA initialisation seed is the family default (1581104982), as in
  every stage-1 full-size cell. Trainer flags as the stage-1 drivers: `--ce-checkpoint --max-memory-gb 78`.
- It trains on the sampling pod (an H200; the reference was also trained on an H200).

**Step budgets differ by cell, as the stage-1 rule has them:** 64 examples get 128 steps (two passes), 16 examples 64
steps (four passes), 4 examples 64 steps (sixteen passes), and the rank-1 adapter up to 512 steps.

### 9(b) Sampling

- The three new cells in one engine with `multi_sample.py`: the 48 reach prompts × 8 draws (384 draws per cell), as
  two chunks of 4, engine seed 660009, batch name `batch_a9`.
- Dan's sampler settings as in every earlier batch (vLLM 0.11.0, olmo_thinker template, T 1.0, top-p 1.0, 32,768 max
  tokens, bf16, prefix caching). The LoRA gate runs first; a cell that fails it is not sampled and is reported.
- If the 16-example adapter cannot be trained (failure or overrun), the two reused cells are sampled without it, and
  it is reported as not trained. If it is flagged `did_not_train`, it is still sampled, with the flag.
- **Optional fourth cell, the coordinator's decision after the audit:** the reference adapter sampled again in the
  same engine. If it runs, it is the primary reference (same engine, same batch). The stored reference is then
  reported beside it, and their paired difference is the size of a batch-to-batch offset.

### 9(c) Scoring

1. Strict checker (Dan's `grade.py`, on the pod), as before.
2. **Round 5 of the reader check**, round 4's protocol unchanged (PREREG "Round-4 hand-check protocol", items 3 to 9),
   on every strict pass of the three new cells and of the reference, which round 4 did not read (and of the fourth
   cell if it runs).
   - Mechanical screen first (Dan's rule 1). A screened pass counts as gamed and as task not done, and is not read.
   - Two labels, each from two independent blinded reads, with a third read on a split and the majority as label:
     Dan's genuine/gamed rubric (his reader prompt verbatim), and the task-done instruction verbatim from round 4.
   - Readers are fresh Sonnet subagents at ordinary effort through one dispatcher, in waves of at most 8, as in round
     4. Every returned file is validated; a failed batch is dispatched once more.
   - Cells are mixed and hidden: one seeded shuffle over all items, batches of 16 as stride slices. Item files carry a
     hashed id, the prompt, the constraints with the checker's verdicts (primary label only) and the answer. The key
     is kept under `v2/handcheck/round5/`; readers see only `v2/hc5_read/`.
   - **Drift check:** 100 items drawn uniformly without replacement from round 4's 1,340 (seed 660059) are mixed in
     under new ids, and get both labels again.
   - Item id: `h` plus the first 7 hex digits of sha1(`660051|cell|draw`). Seeds: primary read 1, read 2 and third
     read 660051, 660052, 660053; bootstrap 660054; task reads 660055, 660056, 660057; spot read 660058.
   - Nobody opens a response of the new cells before the readers have finished. Only counts are computed.

### 9(d) Quantities

- Per cell, each with a 95 % interval from round 4's bootstrap (10,000 reps; the 48 prompts resampled, shared by all
  cells, then each cell's draws within each prompt):
  - the strict pass rate;
  - the reader-accepted pass rate (passes neither screened nor read gamed, over all draws);
  - the task-done pass rate (passes neither screened nor labelled not_done, over all draws).
- Paired differences over prompts, new cell minus the reference, on each of the three rates.
- Each cell's reader-accepted rate as a share of the final model's 47.4 % (round 4, 182 of 384 draws), with an
  interval from the same bootstrap run with the final model's round-4 labels. The same for the task-done rate
  against 52.9 %.
- Second reference: new cell minus round 4's "our data, our seed" one-pass adapter (34.9 % [25.8, 44.3]
  reader-accepted). It was read in round 4, so this comparison crosses reading rounds.
- Descriptive, as round 4: screened share, read-gamed share, the share of passes without `.`, `!` or `?`, median
  answer words. From the training logs: held-out loss, the selected lr and step, the flags.
- Agreement: raw agreement and Cohen's κ between read 1 and read 2 for each label.
- **Drift:** for each label, the round-5 label against the round-4 label on the 100 items: raw agreement, κ, how
  many moved each way, and an exact two-sided sign test on the items that moved.
  - If either label's test gives p < 0.05, the reading is "readers shifted between rounds". Differences against the
    reference stand, because both sides are read in round 5. Every comparison with a cell read in round 4 (the share
    of the final model's rate, the second reference) is then given with the shift beside it.
  - Otherwise "no shift detected", which with 100 items means no net shift larger than about 7 items.

### 9(e) Reading rule, fixed now

On the reader-accepted rate, with the paired difference (new cell minus the reference, in points of all draws) and
its 95 % interval:
- **"holds"**: the difference is above −5 and the interval's lower end is above −12;
- **"falls short"**: the difference is below −10 and the interval's upper end is below 0;
- otherwise **"unclear with 48 prompts"**.

The same rule is applied to the task-done rate and reported beside it. If the two labels read differently, both are
given, and the sentence about the cell uses the reader-accepted one and names the other.

What the rule can resolve: paired differences between cells had intervals of about ±8 points in round 4. So "holds"
needs a difference above about −4, "falls short" one below −10, and between them the reading is "unclear". "Holds"
means "within 12 points of the reference", not "equal to it".

What will not be written, whatever the result: "rank-1 is enough" or "4 examples are enough". The sentences name the
cell, its rate and interval, the reference's rate, the difference and the final model's 47.4 %.

### 9(f) Limits, stated with every sentence from this check

- One training seed per cell, and one order of the examples: "4 examples" are these four, the first four of one
  seeded shuffle.
- 48 prompts. Intervals are set by the prompts, so more draws or reads would not narrow them much.
- The readers are a model. In round 4 two reads agreed on 87 % (κ 0.71) and 88 % (κ 0.68), and the majority matched
  Dan's label on 83 % (κ 0.64).
- The training answers include checker-gaming ones. Of the 64: 5 labelled gamed, 11 genuine, 48 unlabelled. Of the
  first 16: 2 gamed, 1 genuine, 13 unlabelled. Of the first 4: none gamed, 1 genuine, 3 unlabelled.
- None of the 48 reach prompts shares an instruction type with the first 4 training prompts; 5 do with the first 16,
  and 16 with all 64.
- The size axis has two points (655,360 and 134,217,728 weights) and the example axis three (4, 16, 64). Nothing
  between them is located, and no rank-1 adapter with fewer examples is trained.
- "Rank-1" here is rank 1 on `o_proj` alone, not rank 1 on all projections.
- The reference and the 4-example cell are flagged `unconverged`, and in all three existing cells the selected lr is
  the one the edge rule added, so the best lr is not bracketed from below. A cell that falls short under this rule
  is reported as "falls short under the stage-1 rule", not as "too few examples".
- The reference was sampled in another batch (another pod, seed and set of adapters sharing the engine). Unless the
  fourth cell runs, a batch-to-batch offset is not controlled. What exists: the 25 Nov adapter gave 63.5 % on Dan's
  single-adapter path and 64.8 % on this one, and our retrain of it in another batch gave 64.3 %.
- The reused adapters were trained on an H100, the reference and the new one on an H200.

### 9(g) Predictions, written before any data (reader-accepted pass rate)

| cell | coordinator | this orchestrator |
|---|---|---|
| rank-1, 64 examples | 28 to 34 % | 27 to 36 %, most likely 31 %; reading "holds" (60 %), "unclear" (35 %), "falls short" (5 %) |
| full size, 16 examples | 22 to 32 % | 20 to 31 %, most likely 26 %; "unclear" (45 %), "falls short" (35 %), "holds" (20 %) |
| full size, 4 examples | 8 to 20 % | 6 to 24 %, most likely 14 %; "falls short" (80 %), "unclear" (17 %), "holds" (3 %) |
| reference (not read yet) | | 33 to 39 %, most likely 36 % |

### 9(h) Sampling discipline and the money stop

- Every step runs under its own deadline, sized well past its expected end, so that no deadline ends a healthy step.
- **No wall-clock cap is used to cut sampling.** Draws that are still running at a cut are the long ones, and losing
  them biases the rate. So the budget is enforced before the engine starts: sampling begins only if the planned
  length of the whole batch ends inside the money stop.
  - Planning numbers: 8,000 tokens per draw (the reference's mean was 7,953) at 1,450 tokens per second (measured
    1,505 to 1,544), which is 36 minutes per cell, plus 12 minutes for the gate, the engine load, grading and the
    last fetch.
  - If three cells do not fit, the 16-example cell is not sampled (it stays on file as trained, not sampled), and
    the rank-1 and 4-example cells are. If two do not fit, nothing is sampled and the job reports.
  - The fourth cell, if approved, is dropped first.
- If a cell is cut anyway (the pod fails, or the money stop arrives during sampling in spite of the plan), it is
  reported three ways: as sampled; with every unfinished draw counted as a fail on all three rates; and with every
  unfinished draw counted as an accepted pass. "Unfinished" is every draw of a chunk that was not written. A reading
  that changes under either bound is "not robust to the cut".

### 9(i) Pod, cost and stop

- One H200 at $5.29/h, shared with an unrelated job (weight interpolation) that runs before or after this one; one
  job on the GPU at a time. This job keeps its files under `/workspace/mini` and uses the shared, read-only
  `/workspace/models/dpo` (Dan's `fetch.sh` layout at commit 97604023), `/workspace/models/tok` and
  `/workspace/nd-takehome` at 8e744a89.
- Expected, counted from this job's first command on the pod:

  | step | expected | deadline |
  |---|---|---|
  | set-up (and the weights, if the other job has not fetched them) | 5 to 10 min | 30 min |
  | training the 16-example adapter | 21 to 28 min | 60 min |
  | fetch, staging, upload of the adapters | about 5 min | |
  | LoRA gate | about 3 min | 15 min |
  | sampling three cells and grading | 90 to 105 min | to the money stop |
  | last fetch with md5 on both sides | about 3 min | |
  | **total** | **about 2 h 10 min to 2 h 35 min, $11.5 to $13.5** | |

- The coordinator's figure is about $10 to $12. The **hard stop is $15** (170 minutes of this job's pod time). Leon
  is asleep, so at $15 the job stops, fetches what exists and reports; it does not continue.
- The fourth cell would add about 35 minutes ($3.1) and does not fit under $15 with the planning numbers. It needs
  the coordinator to raise the stop to about $18.5, or it is not run.
- The reader round runs on serrano after the fetch and costs no pod time.
- Stop rule: one training cell, one sampling batch, one reader round. Nothing else is launched under this amendment.

### 9(j) Changes after the independent audit (2026-11-28 4:00 am PT), before the plan file is built

An independent auditor read 9(a) to 9(i) and the stored files. Its findings N1 to N11 and the replies are in
`discourse/stage1_report_audit.md` under "Audit of Amendment 9". At this point no pod exists, no draw is taken and the
new adapter is not trained. Unchanged: the cells, the training plan, the sampling settings, the reading rule with its
thresholds, and every seed.

**Corrections to statements above**

1. "Why" (N10). The 34.4 to 38.5 % are round 4's five one-pass adapters. The reference of this check, the adapter
   selected by held-out loss, has never been read; round 5 reads it for the first time. So the coordinator's
   condition was met by other full-size adapters, not by this one.
2. 9(a), the 16-example schedule (N4). Only the step count (64) and the evaluation steps predate the stage-1 data.
   The lr rule does not: `DESIGN_DRAFT.md`, and stage 1's one planned 16-example cell (on the `own` pool, never run),
   took the lr selected at 64 examples. Here the 16-example cell gets the full grid with the edge rule, as the
   4-example cell of the same pool did in stage 1 and as the coordinator's instruction says. That choice is made in
   this amendment.
3. 9(a), the rank-1 adapter (N9). Stage 1 trained it on the schedule SPEC gives to the `own`-source rungs: every lr
   to 128 steps, the best continued to 512. SPEC's text for the `final`-source rungs says 64 steps. With 64 steps the
   selection would have been lr 1e-3 at step 64 (held-out loss 0.5416) instead of lr 3e-4 at step 128 (0.5403). "The
   stage-1 rule" in 9(a) means the adapter as stage 1 trained and selected it. It is not retrained.
4. 9(d), drift (N3). "No shift detected ... means no net shift larger than about 7 items" is wrong: a test that is
   not significant leaves a shift open. With 100 items, a shift of the size already seen between rounds 3 and 4 would
   be detected about half of the time.
5. 9(h) and 9(i), time (N7). The row "sampling three cells and grading, 90 to 105 min" was below the planning
   numbers' own 106 minutes. The table below replaces it.

**Reader check: the two round-4 cells are read again (N3)**

- Readers have shifted between rounds once already. Of the 120 round-3 items that round 4 read again, 12 moved from
  genuine to gamed and 3 the other way (exact sign test p = 0.035).
- So round 5 reads again every round-4 item of the final model (258) and of the one-pass adapter "our data, our
  seed" (149), under new ids, mixed with the rest, both labels. This costs no pod time; it adds about 380 items to
  the reader round.
- The share of the final model's rate and the comparison with the one-pass adapter then use round 5's labels on both
  sides. The same quantities with round 4's labels (47.4 %, 34.9 %) are given beside them.
- The 100 uniformly drawn items stay, drawn first with seed 660059, and the test of 9(d) is run on them as written.
  Its result is reported with the net shift and a 95 % interval, not only as "shifted" or "no shift detected". The
  same numbers are given for each of the two cells read again in full.
- `handcheck5.py prep --no-reread` leaves the 407 items out, if the coordinator prefers the first design. The
  cross-round numbers are then given with the drift interval beside them.

**Limits added to 9(f)**

- Step and token budgets differ between the cells (N4). Training tokens seen: 4 examples 13,110 tokens × 16 passes
  (about 210,000); 16 examples 55,747 × 4 (about 223,000); 64 examples 245,487 × 2 (about 491,000). The example
  axis is neither token-matched nor step-matched, and the reference and the 4-example cell are both selected at their
  last step.
- Every existing selection is a near tie between different (lr, step) pairs (N5). 4 examples: 3e-5 at step 64
  (0.5707) against 1e-4 at step 16 (0.5716). Rank-1: 3e-4 at step 128 (0.5403) against 1e-3 at step 128 (0.5413).
  Reference: 3e-5 at step 128 (0.5373) against 1e-4 at step 64 (0.5391). The gaps are 2 to 5 times the 0.0004
  between two runs of one seed in Amendment 7. Each cell's sampled behaviour rests on one pick among near ties.
- The reused adapters' `cell.json` files were written by the trainer as it was that night (they lack the later
  `selection` key), and no file records the trainer flags of those two runs (N11).
- Readers are told to open only their batch file. Nothing technical stops a reader from opening other files, in
  round 5 as in round 4 (N11).

**Reading rule: unchanged, with two additions to the wording (N6)**

- "Holds" is never written alone. It is written "holds (within 12 points of the reference)".
- Whenever the interval of a difference lies wholly below 0, the sentence says "below the reference", whatever the
  reading. An estimate of −9 with [−17, −1] reads "unclear with 48 prompts" and is also "below the reference".
- What the rule does, with the paired intervals of round 4 (half-width about 8.5 points, standard error about 4.3):

  | true difference | holds | unclear | falls short |
  |---|---|---|---|
  | 0 | 79 % | 20 % | 1 % |
  | −5 | 36 % | 52 % | 12 % |
  | −10 | 7 % | 44 % | 50 % |
  | −15 | 0 % | 12 % | 88 % |

**Predictions of this orchestrator, revised before any data (N6).** The rates stand. The readings did not fit them:
a rank-1 rate of 31 % against a reference of 36 % is a difference of −5.
- Rank-1, 64 examples: "holds" 40 %, "unclear" 45 %, "falls short" 15 %.
- Full size, 16 examples: "falls short" 45 %, "unclear" 43 %, "holds" 12 %.
- Full size, 4 examples: "falls short" 88 %, "unclear" 10 %, "holds" 2 %.

**Time and money, replacing the table of 9(i)**

| step | expected | basis | deadline |
|---|---|---|---|
| checks, push and set-up | 3 to 6 min | 1.6 to 3.6 min on the three pods of 27 Nov | 30 min |
| weights, if the other job has not fetched them | 0.5 to 4 min | 25 s and 3.8 min on 27 Nov | 25 min |
| training the 16-example adapter | 24 to 31 min | the reference's times on the H200: about 7 min per lr, up to four lrs, plus the model load | minute 46 of pod time |
| fetch, staging, upload of the adapters | about 5 min | 2.5 min for three adapters on 27 Nov | |
| LoRA gate | about 3 min | 2.6 min on 27 Nov | 15 min |
| sampling three cells and grading | 75 to 110 min | 71 min for stage 1's three cells (about 5,500 tokens per draw); 110 min at the planning numbers | this job's stop |
| last fetch with md5 on both sides | about 3 min | | |
| **total** | **about 1 h 55 min to 2 h 40 min, $10.0 to $14.3** | | |

- The hard stop stays $15, which is 170 minutes. The driver stops this job's chain at minute 165 and fetches.
- The plan now counts 15 minutes of overhead per batch instead of 12. Three cells are sampled only if the decision
  before the engine starts is taken by minute 49, and the 16-example adapter is trained only if set-up ends by
  minute 12.
- The driver waits for training only until minute 46, the last minute after which three cells still fit. A training
  stage still running then is stopped, and the rank-1 and 4-example cells are sampled without the 16-example cell,
  as 9(b) says for an overrun.
- In the slow case the margin to the stop is about 8 minutes. Draws would be cut if the three cells averaged more
  than about 8,200 to 9,300 tokens per draw (the reference averaged 7,953; stage 1's three cells about 5,500).
  A cut is then reported as 9(h) says.

**If sampling is cut (N2).** The pod grades only after the sampler ends, so a cut batch comes back without grades.
The finished requests are then graded on serrano with Dan's `grade.py` at 8e744a89 (it reproduces the stored
reference's 166 of 384). `handcheck5.py prep` refuses a cell that has draws and no grades; it does not drop it.

**Sharing the GPU (N1).** The two jobs' own locks do not see each other. Added on this job's side:
- before training, before sampling, and again on the pod when each chain starts, the GPU must have less than
  2,000 MiB in use (the other job's own test);
- while one of this job's chains runs it also holds the other job's lock file `/workspace/wi/gpu.lock`, if that file
  exists. It opens the file for reading and does not create or change it.
The coordinator still runs the jobs one after the other; these are guards.

**The optional fourth cell (N8).** The auditor recommends against it: it costs 33 to 35 minutes ($2.9 to $3.1), does
not fit under $15, and could only show a batch-to-batch offset larger than about 6 to 7 points, while the one
measured offset is 1.3 points on the checker. This orchestrator agrees. It stays optional (`REF2=1` with the stop
raised to about $18.5), for the coordinator to decide.

**Added at 4:12 am PT, with still no pod, no draw and no trained adapter.** Two things in "Time and money" above
were changed after 4:00 am, while the driver's tests were written. The training estimate is now 24 to 31 minutes,
taken from the reference's recorded times (first: 21 to 28). And the driver waits for training only until minute 46.
The numbers that follow from them were changed with them: the total, "set-up ends by minute 12" (first: 14) and the
slow-case margin (first: about 10 minutes). Nothing else in 9(j) was changed.

### 9(k) Coordinator's note (5:32 am PT, 2026-11-28), before any round-5 item exists: round 5 also reads the four weight-interpolation cells

Decided by the coordinator when it gave the go for the pod run, and recorded here before `handcheck5.py prep`. At
this moment the Amendment 9 driver has just started on the pod: set-up is done and the training of the 16-example
adapter began a minute ago, so that adapter does not exist yet and no draw of the new cells exists. No round-5 item file exists, and no reader has seen a round-5 item. This orchestrator
has opened no answer of the four cells below; it has read only their row counts, checksums and the keys of one row.

**The coordinator's decisions on the four points of the preparation report:** no fourth cell (the reference is not
sampled again); the re-read of the 407 round-4 items stays; the lock on the other job's lock file stays as built; the
other job ran first and is finished, so its spend ledger is not affected.

**The added cells.** Four cells of the weight-interpolation run of the same night (`olmo_weight_interp_20261128`).
Each is one engine load that sampled the 48 reach prompts × 4 draws (192 draws), in the metrics-2 format, like
`expo` and `k16` in round 4. The weights are dpo + alpha × (final model − dpo).

| name here | alpha | what it is | strict passes of 192 |
|---|---|---|---|
| `interp_a025` | 0.25 | merged weights, a quarter of the way from `dpo` to the final model | 2 |
| `interp_a050` | 0.5 | merged weights, half way | 26 |
| `interp_a075` | 0.75 | merged weights, three quarters of the way | 102 |
| `interp_a100` | 1.0 | the stored final model itself, served on last night's engine | 133 |

- Files: `/scratch/cowifornia/nd-rl/olmo_elicit/interp/runs/cell_{a025,a050,a075,a100}/out/samples.jsonl.gz` and
  `graded.jsonl`, with the md5 list in `/scratch/cowifornia/nd-rl/olmo_elicit/interp/analysis/cells_manifest.json`.
  `prep` checks every md5 against that list and stops on a difference. The files are only read.
- Every strict pass goes through the mechanical screen and, if not screened, to the readers: mixed blind with every
  other round-5 item in the same shuffles, the same two labels, the same two reads and third read on a split, the same
  seeds. 263 passes before the screen, so about 16 more batches for each read.
- Reported for each of the four: the strict, reader-accepted and task-done pass rates with 95 % intervals from the same
  bootstrap (these cells have 4 draws per prompt, the others 8).
- The paired difference of `interp_a100` from the `s2300` first-8 cell as read again in round 5, on each rate. This
  is an engine check on the reader-accepted rate: the same weights, another night's engine and draws.
- Each interpolation cell's reader-accepted rate as a share of `interp_a100`'s, with an interval.
- **No reading rule attaches to these cells.** The rule of 9(e) is for the three adapter cells only. Nothing about
  these cells was predicted here.
- A known property, recorded here and not told to the readers: the four cells share one engine seed, so the same
  prompt can open the same way in two of them. Readers may therefore meet two items with the same opening. The
  bootstrap treats the cells' draws as independent, which they are not fully.

### 9(l) Result (2026-11-28 8:10 am PT)

Run on the shared pod `rp-wi` from 5:30:42 to 7:25:36 am PT (6,894 s, **$10.13** of the $15 stop), with the code of
commit 1760974. Nothing was dropped or cut: three cells × 384 draws, no draw at the token limit, every fetch
`FETCH_OK`. Round 5 was read from 7:26 to 8:02 am PT. Pod log and md5s: `log.md`. Numbers:
`v2/handcheck/round5/round5_summary.json` and `round5_tables.md`; figure `fig_a9.py`.

**The adapter trained now** (full size, 16 examples): lr 3e-4 at step 16 (one pass over the 16 examples), held-out
loss 0.5551 against 0.6341 untrained; no flag. The best lr is the middle of the grid, so it is bracketed on both
sides. Held-out losses of the four cells: 4 examples 0.5707, 16 examples 0.5551, rank-1 0.5403, reference 0.5373.

**Rates** (share of all draws, 95 % interval; 48 prompts × 8 draws, the interpolation cells 48 × 4).

| cell | strict checker pass | reader-accepted pass | task-done pass | passes screened | read gamed |
|---|---|---|---|---|---|
| rank-1 adapter, 64 examples | 46.4 % [36.5, 56.2] | 37.8 % [28.6, 47.1] | 39.8 % [31.0, 49.0] | 1 of 178 | 32 of 177 |
| full-size adapter, 16 examples | 26.6 % [18.0, 35.7] | 24.2 % [15.9, 33.3] | 25.0 % [16.7, 34.1] | 0 of 102 | 9 of 102 |
| full-size adapter, 4 examples | 8.3 % [3.9, 13.5] | 8.3 % [3.9, 13.5] | 7.8 % [3.6, 12.8] | 0 of 32 | 0 of 32 |
| reference: full-size adapter, 64 examples (stored draws) | 43.2 % [33.3, 53.4] | 35.4 % [25.8, 45.3] | 37.0 % [27.3, 46.9] | 0 of 166 | 30 of 166 |
| the final model, first 8 stored draws, read again | 68.8 % [58.9, 78.1] | 47.1 % [36.2, 58.1] | 51.8 % [41.4, 62.0] | 6 of 264 | 77 of 258 |
| the one-pass adapter (our data, our seed), read again | 40.6 % [31.0, 50.5] | 35.4 % [26.0, 45.1] | 33.6 % [24.2, 43.0] | 7 of 156 | 13 of 149 |
| weights a quarter of the way from `dpo` to the final model | 1.0 % [0.0, 3.1] | 1.0 % [0.0, 3.1] | 1.0 % [0.0, 3.1] | 0 of 2 | 0 of 2 |
| weights half way | 13.5 % [6.8, 21.4] | 13.0 % [6.2, 20.8] | 12.5 % [6.2, 19.8] | 0 of 26 | 1 of 26 |
| weights three quarters of the way | 53.1 % [41.7, 64.6] | 41.1 % [30.2, 52.6] | 45.8 % [34.4, 57.3] | 0 of 102 | 23 of 102 |
| the final model's own weights on last night's engine | 69.3 % [58.3, 79.7] | 45.8 % [34.4, 57.8] | 50.5 % [39.1, 62.0] | 4 of 133 | 41 of 129 |

**Paired differences from the reference, in points, and the readings under 9(e).**

| cell minus the reference | strict | reader-accepted | reading | task-done | reading |
|---|---|---|---|---|---|
| rank-1 adapter, 64 examples | +3.1 [−4.7, +10.9] | +2.3 [−4.9, +9.6] | holds (within 12 points of the reference) | +2.9 [−4.4, +10.2] | holds (within 12 points of the reference) |
| full-size adapter, 16 examples | −16.7 [−24.2, −9.4] | −11.2 [−18.5, −4.2] | falls short; below the reference | −12.0 [−19.5, −4.7] | falls short; below the reference |
| full-size adapter, 4 examples | −34.9 [−43.8, −26.3] | −27.1 [−35.7, −19.0] | falls short; below the reference | −29.2 [−37.8, −20.8] | falls short; below the reference |

- The two labels give the same reading in all three cells.
- The 16-example reading is "falls short" by a margin of 1.2 points on the difference (the rule needs below −10; it
  is −11.2) and 4.2 points on the interval's upper end.
- Both short cells are "falls short under the stage-1 rule" (9(f)): the reference and the 4-example cell are flagged
  `unconverged`.

**Against the final model and the one-pass adapter**, both read again in round 5 (reader-accepted; task-done).

| cell | share of the final model's rate | minus the final model | minus the one-pass adapter |
|---|---|---|---|
| rank-1 adapter, 64 examples | 80.1 % [65.1, 98.1]; 76.9 % [63.0, 92.3] | −9.4 [−18.0, −0.8]; −12.0 [−20.3, −3.6] | +2.3 [−5.5, +9.9]; +6.2 [−1.8, +14.3] |
| full-size adapter, 16 examples | 51.4 % [35.8, 69.4]; 48.2 % [33.8, 64.0] | −22.9 [−33.1, −12.8]; −26.8 [−36.7, −16.9] | −11.2 [−18.5, −4.2]; −8.6 [−16.4, −0.8] |
| full-size adapter, 4 examples | 17.7 % [8.9, 27.6]; 15.1 % [7.3, 23.9] | −38.8 [−48.7, −29.2]; −44.0 [−53.6, −34.4] | −27.1 [−36.2, −18.5]; −25.8 [−34.9, −16.7] |
| reference | 75.1 % [59.5, 93.3]; 71.4 % [56.5, 87.2] | −11.7 [−20.6, −2.9]; −14.8 [−23.7, −6.2] | +0.0 [−7.6, +7.3]; +3.4 [−4.7, +11.5] |

With round 4's labels for the final model instead, the reader-accepted shares are 79.7 %, 51.1 %, 17.6 % and 74.7 %:
within 0.4 points of the table.

**The four weight-interpolation cells (9(k)); no reading rule attaches to them.**
- The final model's own weights on last night's engine, minus the final model's first 8 stored draws as read again:
  strict +0.5 [−7.6, +8.3], reader-accepted −1.3 [−10.9, +8.3], task-done −1.3 [−10.2, +7.6]. No engine difference is
  detected; one of 11 points down or 8 up is not ruled out.
- Reader-accepted rate as a share of that cell's: a quarter of the way 2.3 % [0.0, 7.6]; half way 28.4 %
  [14.6, 46.2]; three quarters 89.8 % [70.9, 112.9]. Task-done: 2.1 % [0.0, 6.9], 24.7 % [12.3, 40.0], 90.7 %
  [73.3, 112.0].
- Descriptive: the share of passes that are screened or read gamed grows along the path (0 of 2, 1 of 26, 23 of 102,
  45 of 133), and so does the share of passes without `.`, `!` or `?` (0 %, 19 %, 32 %, 45 %).
- The four cells share one engine seed and have 4 draws per prompt; their intervals treat the cells as independent.

**Descriptive, per cell.** Share of passes without `.`, `!` or `?`: rank-1 19.7 %, 16 examples 8.8 %, 4 examples
9.4 %, reference 15.7 %, the final model 40.9 %. Median words of a passing answer: 40, 48, 46, 48, and 32.

**Readers.**
- 1,216 items in 76 batches of 16 per read; 320 reader calls (Sonnet, ordinary effort), 318 valid files. Two batches
  (`a70`, `b14`) failed validation once ("1 id missing") and passed on their one re-dispatch.
- Two reads agree on 91.6 % of items (κ 0.76) for the reader-accepted label and 91.0 % (κ 0.70) for task-done. 102
  and 110 items went to a third read. No item is unread.
- **Drift, the pre-registered test on the 100 items drawn uniformly from round 4:** the same label on 93 for each
  label (κ 0.86 and 0.82); 3 moved to accepted and 4 away (net −1.0 points [−6.2, +4.2], sign test p = 1.0); for
  task-done 4 and 3 (+1.0 [−4.2, +6.2], p = 1.0). **Reading: no shift detected.**
- **The two cells read again in full:**
  - the final model, 258 items: the same reader-accepted label on 92.6 % (κ 0.82), 9 to accepted and 10 away (−0.4
    points of items [−3.7, +2.9]); task-done 89.9 % (κ 0.71), 11 and 15 (−1.6 [−5.4, +2.3]). Its rates moved from
    47.4 % to 47.1 % and from 52.9 % to 51.8 %.
  - the one-pass adapter, 149 items: 96.0 % (κ 0.76), 4 and 2 (+1.3 [−1.9, +4.6]); task-done 94.6 % (κ 0.76), 3 and
    5 (−1.3 [−5.1, +2.4]). Its rates moved from 34.9 % to 35.4 % and from 34.1 % to 33.6 %.
- Which files each reader opened was checked from the tool calls of all 320 readers: every one named each of its own
  item files, none named an item of another batch, and none listed a folder (one listed its own output file). Task
  readers took about 10 s with 3 or 4 calls; the call counts fit one command that prints the 16 short files.
- Spot read by this orchestrator (40 items, seed 660058, `round5/spot.md`), made with the labels hidden and the
  order shuffled, agreeing with the final label on:
  - genuine 10 of 10; gamed 8 of 10; done 10 of 10; not_done 9 of 10. In all 37 of 40.
  - The two "gamed" items I would accept were both 2-to-1 splits: a grammar correction given without punctuation,
    and a four-sentence world introduction. The "not_done" one is an eight-word answer to a yes/no question under a
    constraint on every word. None of the 40 is a plain error.

**Predictions of 9(g) against the result** (reader-accepted rate).

| cell | coordinator | this orchestrator | observed | reading, and the probability given in 9(g) and in 9(j) |
|---|---|---|---|---|
| rank-1, 64 examples | 28 to 34 % | 27 to 36 % | 37.8 %: above both | "holds": 60 %, then 40 % |
| full size, 16 examples | 22 to 32 % | 20 to 31 % | 24.2 %: inside both | "falls short": 35 %, then 45 % |
| full size, 4 examples | 8 to 20 % | 6 to 24 % | 8.3 %: inside both, at the low end | "falls short": 80 %, then 88 % |
| reference | | 33 to 39 % | 35.4 %: inside | |

**Interpretation.**
- Observed: the rank-1 adapter on `o_proj` (655,360 weights), trained on 64 examples, has 37.8 % [28.6, 47.1]
  reader-accepted passes against the full-size reference's 35.4 % [25.8, 45.3]: +2.3 points [−4.9, +9.6], reading
  "holds (within 12 points of the reference)". The final model has 47.1 %; the rank-1 adapter is 9.4 points below it
  [−18.0, −0.8].
- Observed: the full-size adapter on the first 16 examples has 24.2 % [15.9, 33.3], 11.2 points below the reference
  [−18.5, −4.2]: "falls short under the stage-1 rule". On the first 4 examples it has 8.3 % [3.9, 13.5], 27.1 points
  below [−35.7, −19.0]: "falls short under the stage-1 rule". On no prompt does the 4-example adapter have more
  accepted answers than the reference (fewer on 35 of 48).
- Observed: the reference equals the one-pass adapter of round 4 on reader-accepted passes (+0.0 [−7.6, +7.3]).
- Observed: with fewer examples the passes are fewer and cleaner. Readers call 0 of 32 gamed for 4 examples and 9 of
  102 for 16, against 30 of 166 for the reference and 32 of 177 for the rank-1 adapter.
- Interpretation: between these two points of the size axis, going from 134 million to 0.66 million trained weights
  costs no measurable reader-accepted rate (the interval's lower end is −4.9 points). Under the stage-1 rule, going
  from 64 examples to the first 16 costs about a third of the reference's rate, and to the first 4 about three
  quarters. Both statements are for one seed, one example order and these 48 prompts, and the cells had different
  step budgets (the rank-1 adapter up to 512 steps, the 16- and 4-example adapters 64).
- **Limits**, as 9(f) and 9(j): one training seed and one example order per cell; 48 prompts; a model reader; the
  reference was sampled in another batch (another pod, seed and engine load) and was not sampled again, so a
  batch-to-batch offset is not controlled (the engine check above bounds one such offset at about 10 points for the
  final model's weights, not for adapters); step budgets differ by cell; the size axis has two points and the
  example axis three; the two reused adapters were trained on an H100.

**Post hoc, not pre-registered** (descriptive only).
- The two labels agree on 1,065 of the 1,216 items (87.6 %): 103 are gamed but done and 48 genuine but not done.
- Read 1 and read 2 have nearly the same base rates: 934 and 942 genuine, 994 and 996 done.
- Held-out loss orders the four adapter cells as the reader-accepted rate does (0.5707, 0.5551, 0.5403 and 0.5373
  against 8.3 %, 24.2 %, 37.8 % and 35.4 %), except that the last two are level on both.

**Differences from the pre-registration.**
- 9(k) added the four interpolation cells (the coordinator's change, written before any round-5 item existed).
  `handcheck5.py` was extended for them, and for a Markdown view of the tables (`md`), after commit 1760974 and
  before `prep`; the round-5 tests were rerun before `prep` and the whole suite after the analysis (116 passed, 1
  skipped).
- No fourth cell: the reference was not sampled again (the coordinator's decision).
- The readers were launched by five dispatcher subagents in turn (reads a, b, ta, tb, then the third reads), each
  with waves of at most 8 and round 4's reader prompt word for word. This orchestrator launched no reader itself.
- The spot read was made blind to the label, which round 4's was not.
- While checking which files the readers had opened (after reads a, b and ta), the check printed two fragments of
  about 70 characters from readers' own commands; they are parts of answers of unknown cells. No label or cell was
  attached, and no reader saw anything from it.
- The set-up installed nothing: the other job had already installed the pinned stack on the pod.
- Samples were also copied to serrano four times during sampling, by a loop beside the driver.
