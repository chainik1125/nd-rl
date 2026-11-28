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
