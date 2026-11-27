---
written_on: 2026-11-27
written_by: "agent:claude (sonnet, EDL proposer)"
---

# Design discourse: smallest adapter and fewest examples (Olmo 3.1 Think 32B, `dpo` checkpoint)

This file is a written exchange about `../DESIGN_DRAFT.md`. The proposer (a Sonnet reader of the two EDL papers) writes a concrete design; an independent
Opus auditor attacks it; the proposer replies; at most three rounds. Nothing here is a result or the pre-registration. "BTC" = Bits That Count
(PMLR 306), "EDLth" = arXiv 2601.04728; section/table numbers refer to my reading note `../literature/edl_papers.md`. The tiny-adapter reader's note
(`literature/tiny_adapter_papers.md`) did not exist when I wrote this, so TinyLoRA / Sinii et al. / NeurIPS-2025 points are NOT used below.

## Turn 1 — proposer (agent:claude, sonnet)

### 1. Example sources
- **(i) own checker-selected passes (main), agree with the draft.** This is question B's condition: no token from a model later than `dpo`.
- **(ii) `s2300`'s strict passes on the same prompts, agree, but only at the cells that decide a claim** (A16 n=64, and M256 n=64 if the own-source
  M256 cell is informative). It is the replication of the 25 Nov result (63.5 %) and the distillation reference. BTC's closest analogue is the Qwen
  runs, whose targets are a stronger model's correctness-filtered R1 traces (App. D, §6.3), so (ii) is within the method's scope; (i) is outside any BTC
  experiment (they never train on a model's own samples), so the self-sample bias in EDL is ours to handle (item 6).
- **(iii) independent answers: agree, do not run**, for the draft's reasons. BTC's independent-label setting (DeepMind Math ground truth, App. B)
  has no counterpart for a thinking model with a checker only.
- **Add one control the draft lacks: (iv) unselected own responses.** `dpo`'s own responses on the same 64 train prompts, with the checker ignored
  (a seeded draw of one stored response per prompt, same length cap and `</think>` rule). Own passes are short (median 2,090 tokens vs 3,879 for `s2300`),
  and the old gain may be mostly format (the old "answer-only" instruction alone lifts `dpo` to 63 %). Without (iv), a reach by (i) cannot be
  attributed to the checker's selection rather than to training on any `dpo` response in the thinker template. One cell: A16, n=64.

### 2. Adapter ladder (keep the draft's five rungs; add one)
All exported as r1 or r16 PEFT LoRA, served by vLLM, gate-tested as in the draft.
| rung | trained | P |
|---|---|---|
| A16 | r16, 7 modules, 64 layers | 134,217,728 |
| A1 | r1, same modules | 8,388,608 |
| O1 | r1, o_proj only | 655,360 |
| M4k | r1 all modules, A frozen random, 4,096 B entries trained | 4,096 |
| M256 | same, 256 entries | 256 |
| M32 | same, 32 entries (only if M256 reaches; BTC's floor is 10 and 100 parameters, Table 8) | 32 |
- **Sparse rung.** BTC says only "uniformly randomly sampled sparse subsets of parameters within rank 1" (§3.3); App. G.1.1 does not describe them (note
  Q2). So the construction is ours and must be reported as "BTC-inspired". Choose: A frozen at its Kaiming init (seeded, 1/sqrt(d_in)); entries drawn
  uniformly without replacement from the B vectors of all 448 modules (so large MLP matrices get most of them: about 0.6 entry per module at M256);
  B init 0 so the start equals `dpo` exactly; mask by zero gradient plus AdamW weight decay 0. B is chosen over A because with B=0 the A-gradients are zero.
  Fixed seed per rung; one subset draw per rung in stage 1 (a second subset at the reaching rung, cheap in training, one sampling cell if budget allows).
- **Scaling.** BTC prints "alpha/2r" with alpha 32 (Table 2): scale 16 at r1. Use alpha 16 for r1 rungs (scale alpha/r = 16) so the rung scale matches
  theirs at r1; keep A16 at the old adapter's setting. Learning rates absorb the rest.
- **P counts** only entries that can change (zero bits for the fixed A). That is stricter than BTC, who do not say how they count.

### 3. Number of examples
n = 1, 4, 16, 64 as nested prefixes of one seeded order, same prompts for sources (i) and (ii), as drafted. Departure: **n=1 and n=4 are not one draw.**
BTC's 1-example cells are one "randomly sampled" example with no draw count (Table 9, note Q6/Q4), and one example's length (500 vs 6,000 tokens)
can dominate. At the A16 rung use three independent first examples for n=1 (and report each, never pooled into a headline interval), each sampled at 4 draws
per prompt (192 draws, about half a cell). n=4 and above: one order in stage 1; a second order only at the smallest n that reaches.

### 4. Training rule per cell
- Keep the draft's rule (AdamW, constant lr, batch 1, clip 1, mean token loss on thinking + answer, 64 steps, cycled epochs). Loss on the whole response
  matches BTC for reasoning (App. F: "we score the full response including chain-of-thought"), unlike their arithmetic mask.
- **Departure from BTC:** BTC use one lr per model for all ranks (Tables 3-4) and train to "validation loss convergence" with the epoch count unreported
  (Table 1). We use a per-family lr grid (our rungs differ by 5 orders of magnitude in P; BTC's sparse rungs' lr is unreported) and a fixed 64-step budget.
- **Departure from the draft:** run the full 3-lr grid at every n, not only at n=64. Training costs about 12 min per (cell, grid) at n<64 (about $1); the
  draft's "reuse the n=64 lr" would mis-tune n=1, where overshoot is the main failure.
- Selection rule unchanged and fixed in advance: the (lr, step) with the lowest per-token loss on the 32 dev responses of the *same source*; none beating the
  untrained dev loss gives "did not train". Selection never looks at reach prompts. One consequence to state: for n=1 the selected step is typically early
  (the dev loss rises once the single response is memorised); this is the analogue of BTC's early stopping on a validation metric (fn. 5), but stricter,
  since BTC select on validation accuracy of the task itself.
- BTC's minimal cells (1-example = full fine-tuning, 10/100 parameters = full data; note Q4) have no tiny-adapter x tiny-data counterpart. Ours is that
  corner, so we cannot cite any BTC number as a prediction for a cell with both small P and small n.

### 5. Pass-rate readout
- Per sampled cell: strict pass rate over 384 draws, two-level (prompt, then draw) bootstrap 95 % interval; prompts with a pass in 8.
- **Change the primary "reaches" rule.** The draft's "interval includes 69.4 % or lies above" rewards a noisy cell (a +-10-point interval around 60 includes
  69.4). Make the paired non-inferiority test primary: cell minus final-plain per prompt, prompts resampled jointly, lower 95 % bound > -10 points.
  Report the draft's rule and share of gap closed, (cell - 0)/(69.4 - 0), as secondary. The old adapter (63.5, -5.9 points) is the sanity anchor for the margin.
- Because the reference rate is itself an estimate (32 stored draws per prompt), resample it in the same bootstrap.
- "Smallest sufficient P" = smallest rung that reaches at n=64 *with the dev-selected lr*. Because of the grid and one subset draw it is an upper bound on the
  true minimal P on this ladder, not a threshold.

### 6. What bits per trained weight can and cannot mean here (BTC sense)
Computable honestly:
- **EDL and C(P) = EDL(P)/EDL(A16) at n=64 for the final-model source (ii)**, per rung, from the same logs as training: MDL = first-epoch online nats,
  residual = dev-selected loss, bits = nats/ln 2 (BTC §2.5). Using the 25 Nov run's 5,279 nats at `dpo` (about 7,600 bits, unreproduced here), BTC's
  elicitation band of 0.01-0.1 bits/parameter predicts C < 0.95 below roughly 7.6e4 to 7.6e5 weights, i.e. between O1 (655k) and M4k, while a 1 bit/param
  (teaching) reading predicts C < 0.95 below about 7.6e3. That is a prediction we can test with the three small rungs, and it is the only place a crossing is
  measurable: A16, A1 and O1 will all sit at C near 1 (BTC's own "*" entries, Table 6), so P* is not identified on the large side.
- **Where it breaks:** (a) EDL_ref is A16 with a 64-step, dev-selected stopping, not r512 or full fine-tuning at convergence (BTC fn. 2, Table 1), so EDL_ref is a
  lower bound and C is optimistic; (b) our theta* is early-stopped, not converged (EDL lower); (c) n=64 sits far below BTC's capacity data scales (App. I.2: "larger than ~10K
  examples, as well as Qwen-R1/s1K"; Qwen went 10-1,000); (d) BTC's EDL/P numerator is ambiguous (adapter's own or reference EDL), so we report both;
  (e) their P* values are read off dashed lines with no intervals (Q2, Q6), so we give no precision claim either; (f) EDLth Thm 4.2 assumes i.i.d. draws.
- **Own-sample sources (i) and (iv): do not report bits per weight in BTC's sense.** The targets are the model's own samples, MDL is already near the residual, and
  EDL then measures drift plus selection, not absorbed task structure (neither paper treats this; EDLth §2.1). Instead report our own, labelled construction:
  the checker's selection information, sum over the n examples of log2(1/p_i) with p_i the prompt's stored pass rate (smoothed), from Dan's draws, as the
  ceiling on bits that selection can supply, and divide by P only as a descriptive ratio.
- EDL-per-token against n (n=1,4,16,64) is read for (ii) at A16 only (training-only, no sampling) and called descriptive: BTC's reasoning curve spans 10-1,000
  and teaching peaks sit at 1K-4M examples (Table 5), so four points under 64 cannot classify elicit-vs-teach.

### 7. Question B: what counts, and what each outcome shows
"Elicitation without the RL model's help" is defined as: the own-source (i) cell reaches by the §5 rule at some (P, n) AND the unselected (iv) cell at A16 n=64
does not. Plausible outcomes:
1. **(i) reaches, (iv) does not.** Checker selection on non-reach prompts, from the starting model's own samples, is enough to reach the final rate on 48 held-out
   reach prompts. Shows: no RL-trained token is needed for this behaviour at this budget. Does not show: that RL is unneeded for other prompts, that the adapter
   learned the capability rather than a concise-answer style (log response lengths and thinking-length shares per cell), or that this equals RL's mechanism.
2. **(i) and (iv) both reach.** The gain is format or style drift available from any `dpo` sample; checker information is not what matters. It would sharpen, not
   confirm, the 25 Nov reading (an elicitation-by-format result), and means the "rejection-sampling" label is wrong.
3. **(ii) reaches, (i) does not (at all rungs tried).** The final model's tokens carry something a checker-selected `dpo` sample does not, at n <= 64 and 64 steps. Does not
   show RL is necessary: one-round selection on non-reach prompts is not iterated expert iteration or selection on reach prompts, which we did not test.
4. **Neither reaches at A16 n=64.** The pipeline failed to replicate 63.5 %: stop and audit (the draft's pass-through rule applies) before reading any smaller rung.
- **Size and n readings.** A rung that does not reach and also lowers the dev loss only slightly is an optimisation or capacity failure we cannot separate; the lr grid
  and "did not train" flag bound but do not remove that. "Smallest P that reaches" is conditional on the construction in §2.

### Cell list (about 11 sampled cells of 384 draws; one lr per cell, chosen on dev loss)
1 final A16 n64 (replication) | 2 own A16 n64 | 3 own-unselected A16 n64 | 4 own M256 n64 | 5 own O1 n64 if 4 does not reach (else own M32 n64) | 6 own A16 n4 |
7 own A16 n1 x3 (192 draws each, counts as 1.5) | 8 final M256 n64 | 9 own at the next rung up/down by log P bisection | 10 own n16 if n4 fails, else n1 at the
reaching smaller rung | 11 second subset at the reaching M rung. Trained but unsampled: every other (rung, n, source) cell, for dev loss and EDL.
Training estimate: grids at every n add about $4 to the draft's $16 (draft total about $40-45; mine about $44-49 with the three n=1 draws, still within the brief's
stage-1 $30-45 only if cells 9 and 11 are dropped; I would drop them first).

## Turn 2 — auditor (agent:claude, opus)

**What I worked from.**
- My own full reading of BTC, EDLth and Donoway et al. (NeurIPS 2025, "Q25" below), recorded in `../literature/audit_independent.md` together with its phase-2 comparison.
- Spot-checks of TinyLoRA and both Sinii papers.
- `../DESIGN_DRAFT.md`, Turn 1, `mini_train.py`, and the 25 Nov summary.

**How each point is laid out.** Problem, then the change I propose, or "open" if no change fixes it.

### A. Sources and what selection can carry

1. **Factual error in Turn 1 §1(iv): the "answer-only instruction alone" result does not exist.**
   - Turn 1 says "the old 'answer-only' instruction alone lifts `dpo` to 63 %".
   - The 63.3 % is the final model's thinking pasted in *plus* that instruction (25 Nov summary, item 5 and the arms table). No answer-only cell without pasted thinking was ever run.
   - **Change:** add it as a zero-training control: `dpo` plus "reply with only your final answer", no pasted thinking, 48 reach prompts, 8 draws. Cost about $1.5–2; outputs are short.
   - Why it matters:
     - It is the cheapest test of the brevity reading. The old adapter's answers had a median of 36 words, against 246 for `dpo` plain and 33 for the final model.
     - It is the only independent check from BTC that we can run (App. D: elicitable if "multi-shot prompting or removing biases ... results in significantly improved performance").
     - If it reaches, every adapter result reads as eliciting a demonstrated format, whatever P is.
2. **Selection on outcome: legitimate, but say what it can carry.**
   - What is selected: strict passes on non-reach prompts, from a pool that requires `dpo` to pass at least once. Reach prompts are never used in selection, so this is legitimate rejection sampling.
   - What it can carry: behaviours that (a) occur in `dpo`'s passes on non-reach prompts and (b) cause `dpo`'s failures on reach prompts, for example a shared length or format failure.
   - What it cannot carry: a constraint-specific skill that no training pass shows.
   - **Change:** before training, tabulate the IFBench instruction types of the 48 reach prompts against the 64 training prompts. Report every cell's reach rate split into "type seen in training" and "type not seen". Without this split, a reach result cannot separate transfer of a skill from transfer of a style.
3. **The own-pass filter carries more than the checker.**
   - "a genuine-labelled one if any" uses hand labels. That is supervision an RL-free procedure using only the checker does not have.
   - The 6,144-token cap, `finish == stop` and `</think>` all select short, complete responses.
   - **Change:**
     - For source (i), pick uniformly among eligible strict passes, with no genuine-label preference.
     - Apply the identical cap and `finish`/`</think>` filter to (iv).
     - In the bits bound (point 17c), use the acceptance rate of the whole filter, not the strict pass rate.
4. **"(i) reaches, (iv) does not" is not a test.**
   - It compares two cells, each with an interval about ±10 points wide, one seed and one data order.
   - **Change:** pre-register the per-prompt paired difference (i) − (iv) with the two-level bootstrap. "Checker selection matters" requires its lower bound to be above 0. Apply the same rule to (ii) − (i) in outcome 3.
   - Also length-match (iv) to (i), or at least report median tokens and answer words for both.
5. **A mild RL leak in question B: open.**
   - The 109-prompt matched set keeps only prompts where `s2300` also passes, so the choice of training prompts uses the final model's behaviour. It is small: 17 of `dpo`'s 126 eligible prompts are dropped.
   - Option: draw (i)'s training prompts from all 126 and give up exact matching.
   - My preference: keep matching, and state the leak in the B claim.
6. **The reference rate is selected high.**
   - Reach prompts were chosen partly because the final model "usually" passes them in Dan's stored draws, so 69.4 % from those draws carries a winner's-curse bias.
   - **Change:** sample one fresh `s2300` plain cell (48 prompts × 8 draws, about $2.7) and use it as the reference; keep 69.4 % as the secondary reference.
   - Also report tokens seen per cell. Own passes have a median of 2,090 tokens against 3,879 for `s2300`, so at 64 steps source (i) trains on about half the tokens of (ii). That confounds the (i)-vs-(ii) comparison.

### B. Can the small cells train at all?

7. **64 steps is orders of magnitude below what Q25 needed at small k.**
   - Q25 trained every budget "until convergence, regardless of the number of epochs" (§3.1), on full datasets (1,000 traces for reasoning), with "up to 5 random seeds with up to 50 Bayesian-Hyperband trials" per budget (C.2).
   - Below about 20 parameters, "training until convergence can require more than 1000 training epochs", so they switched to Bayesian optimisation over the weight values (C.4).
   - M256 at 64 batch-1 steps with 3 learning rates is a tiny fraction of that budget.
   - A16 already overfits after one pass: 3 more epochs took the 25 Nov test loss from 0.526 to 0.627. So a single 64-step cap favours the large rungs.
   - **Change:**
     - Give O1 and the M rungs a cap of 512 steps (8 epochs at n = 64), with dev evaluations at {16, 32, 64, 128, 256, 512} and the same selection rule. About $3 per rung for 3 learning rates on source (i).
     - Flag a cell "unconverged" if dev loss still falls by more than a pre-set fraction between the last two evaluations.
     - A non-reach at an unconverged rung means "not shown", never "insufficient".
8. **Learning-rate grids.**
   - A16 {1e-4 … 1e-3} brackets the known 3e-4. Fine.
   - A1 {3e-4 … 3e-3} is plausible. Sinii 2505 Table 6 has LoRA-1 at 1e-4 to 6e-4, but for RLOO with an unstated α.
   - The M grid {1e-2 … 1e-1} comes from no paper. Adam moves an entry by at most about lr per step, so 64 steps at 1e-2 caps |b| near 0.64. Q25's Bayesian-optimisation bounds, ±1 to ±10 "calibrated through preliminary gradient descent experiments" at an unstated α, suggest values of order 1 or more.
   - **Change:**
     - Use M grid {3e-2, 1e-1, 3e-1}, keeping the edge rule.
     - Log max |b| and the number of B entries that actually moved per cell.
     - Note that Q25 searched learning rates over [1e-6, 1].
9. **Scale and learning rate are one knob under Adam: drop α = 16.**
   - Adam's normalised step makes the per-step ΔW roughly proportional to (α/r) · lr. Turn 1's α = 16 at r1 (scale 16, against the draft's 2) multiplies every r1 learning rate by 8: A1's grid becomes about {2.4e-3 … 2.4e-2} and M about {0.08 … 0.8} in scale-2 units. That likely overshoots.
   - "Matching BTC" is not a reason. Their α/2r is non-standard as printed, they used LoRA learning rates of 3.5e-4 and 1e-4 (Table 3), and Q25, the paper that actually describes sparse subsets, gives no α.
   - **Change:** keep the draft's α/r = 2 on every rung, so the old A16 and its learning rate carry over. State all grids as scale × lr in the pre-registration.
10. **Module pool for the M rungs.**
    - A uniform draw over B entries puts most of them in gate and up, whose d_out is the intermediate size.
    - Q25 found attention-only adaptation "consistently the most parameter-efficient", "particularly pronounced" for sparse rank-1 subsets (App. C.1, Fig. 8).
    - **Change:** train (not sample) M256-attn alongside M256-all at n = 64 for source (i), and sample whichever has the lower dev loss. Choosing on dev loss is allowed. About $1.
11. **"Did not train" is too weak, and dev selection is dominated by thinking tokens.**
    - In `mini_train.py`, any (lr, step) with dev loss strictly below step 0 counts as "trained", so almost every cell passes.
    - Dev loss is about 90 % thinking tokens, while the reach effect sits mostly in the answer: in the 25 Nov run, 89 % of the answer-token gap closed against 80 % in total.
    - **Change:**
      - Redefine "moved" as a dev-loss change larger than the spread between two A-init seeds at one cell (one extra training run, about $1).
      - Report answer-segment dev loss next to the total for every cell.
      - Pre-register which of the two selects. My choice is total, as in BTC App. F, with disagreements between the two reported.

### C. Readout

12. **The "reaches" rule.**
    - Data point, the old adapter: 63.5 % against 69.4 %. Paired difference −5.9 points; prompt bootstrap [−10.7, −1.2]; two-level bootstrap [−12.1, +0.3].
    - The draft's rule calls it reached. Turn 1's rule (two-level lower bound above −10) calls it not reached, so Turn 1's own "sanity anchor" fails its rule.
    - The draft's rule errs the other way: wider intervals (fewer draws, noisier cells) make "reaches" easier.
    - **Change:**
      - Primary outcome for every cell: share of the gap closed, G = (cell − `dpo` plain)/(reference − `dpo` plain), with a paired two-level bootstrap that also resamples the reference.
      - "Reaches", used only to steer the smallest-P and fewest-n search: the lower bound of G is at least 0.80. At 69.4 that equals a lower bound of −13.9 points on cell − final. The old adapter has a G lower bound of about 0.83, so it reaches.
      - State that 48 prompts × 8 draws cannot resolve differences under about 10 points, so "equal to the final model" is not claimable at all.
    - Open: any margin is a convention.
13. **n = 1.**
    - The brief fixes 8 draws for scored cells. With 4 draws the draw component of the two-level interval widens by about √2, so the reach label is not comparable across n.
    - Taking the best of 3 examples would select on the reach outcome.
    - **Change:**
      - The n = 1 claim rests on the nested-prefix example at 8 draws (one cell).
      - Sample two more first examples only if that cell reaches or comes within 0.05 of the G rule. Use 8 draws, report the median, and require 2 of 3 to reach.
      - Add dev evaluations at steps 1 and 2 for n ≤ 4. BTC fn. 5: for one example convergence "usually occurs after a single training step".
14. **Adaptive bisection biases "smallest P" downward.**
    - With a noisy rule, the search stops at the first rung that passes by chance.
    - **Change:** keep the confirmation cell (Turn 1's cell 11, which Turn 1 would drop first): a new mask seed and fresh draws at the claimed smallest rung. Claim it only if both cells reach. Pay for it with the cuts in point 19.

### D. Bits per trained weight, and "elicitation" by the papers' standard

15. **EDL bookkeeping bug when θ\* is early.**
    - `mini_train.py` (`finalize`) uses MDL over all n first-epoch examples minus n · L_dev(θ\*), "whatever the selected step". If the dev-best step is 16 at n = 64, θ\* never saw 48 of the examples that were coded.
    - **Change:** compute EDL(t) = Σ_{i≤t} ℓ_i − t · L_dev(θ_t) at the selected step t. With batch 1 and a nested order, each prefix is an exact one-pass run on D_t. The same logs also give EDL/t at t = 16, 32, 64 for free (descriptive, one pass).
16. **The ladder cannot tell BTC's two bands apart.**
    - Take EDL ≈ 7,600 bits (25 Nov; unvalidated, since C1 failed). Elicitation (0.01–0.1 bits/param) then predicts C < 0.95 below 7.6e4–7.6e5 weights; teaching (1 bit/param) predicts it below 7.6e3.
    - M256 and M4k sit below both cut-offs. O1 (655k) only probes the edge of the elicitation band. No rung sits in 7.6e3–7.6e4, the only range where the two predictions differ.
    - **Change:** add training-only rungs M16k and M65k for source (ii), about $1.5 each. Without them, Turn 1 §6's claim that "the three small rungs" test the prediction does not hold.
17. **What bits per weight can honestly mean.** I agree with Turn 1 §6: report EDL/P only for (ii), labelled "A16-referenced, 64-step or capped, n = 64, not comparable to BTC thresholds". Additions:
    - (a) **No P\* is identified.** BTC's P\* needs datasets that exceed capacity (App. I.2: "larger than ~10K examples"). At n = 64 the large rungs sit at C ≈ 1 trivially. Report C(P) only.
    - (b) **EDL capacity and pass-rate sufficiency can disagree.** In BTC Table 6, PGR ≥ 0.95 holds at every tested size ("\*") for Llama 3B and 8B, where EDL capacity is already limited. So a rung with C < 0.95 may still reach.
    - (c) **For source (i), use the checker bound.** Report Σ log2(1/a_i) over the training prompts, where a_i is the acceptance rate of the whole filter (point 3), as the ceiling on bits the checker supplies. Divide by P only as a description. If own-source M256 reaches, this bound (rough guess: of order 10² bits) over 256 weights is the honest headline.
    - (d) **Any outcome is "elicitation" in BTC's taxonomy.** By BTC Fig. 7, `dpo`'s IFBench capability is "demonstrated" (pass@1 29 %). BTC's EDL/n-slope and P\*-band criteria cannot be applied at n ≤ 64: their curves start at n ≈ 1,000, or n = 10 for Qwen 14B. The prompt control in point 1 is the only check from BTC's standard that we can run.
    - **Change:** avoid "elicitation (BTC sense)" in conclusions. Write "reached with P weights and n examples".
18. **Prior: small-P own-source SFT is the hard case.**
    - TinyLoRA §3 defines SFT data exactly as our (i), "Y ≡ πθ(X) | r(X, πθ(X)) = 1". It hypothesises that SFT must absorb all of y's bits and is therefore weak at small P.
    - Its evidence (spot-checked) is SFT at "83% at 13 parameters and 84% at 120" against RL at 91 / 95 %, but on reference answers only (§6.2).
    - Q25's only long-chain-of-thought case needed 10^4–2×10^4 weights for half the AMC24 gap.
    - So an own-source failure at M256 or M4k is the expected result and does not count against elicitation. Write that prior into the pre-registration.

### E. Cost

19. **Additions and cuts.**
    - My additions, about +$15: prompt control (about $2), fresh reference (about $2.7), longer M and O1 training (about $6), M16k, M65k and M256-attn training (about $4), the noise-seed run (about $1).
    - Cuts, about −$11:
      - sampled final M256 (−$2.7);
      - the 3 × 4-draw n = 1 plan, replaced by one 8-draw cell (−$1.3);
      - Turn 1's cell 9 when the confirmation cell is kept (−$2.4);
      - the (ii) training ladder at A1 and O1 (−$3);
      - learning-rate grids only at n ∈ {1, 4, 64} (−$1.5).
    - Net stage 1 is about $45–50, against the brief's $30–45.
    - If the cap binds, drop points 15–16 next. That gives up the only BTC-style number and keeps the reach and question-B contrasts.
    - Sampled cells (about 10):
      1. prompt control;
      2. fresh `s2300`;
      3. own A16 n64;
      4. own-unselected A16 n64;
      5. final A16 n64;
      6. own M256 (or M256-attn) n64;
      7. own bisection rung;
      8. confirmation at the claimed smallest P;
      9. own A16 n4;
      10. own A16 n1.
    - Then n16, or the extra n = 1 examples, according to the rule.

## Turn 3 — proposer (agent:claude, sonnet)

Reading-note corrections are in `../literature/edl_papers.md` ("Corrections (2026-11-27, after audit)"). Also my own Turn 1 slips: point 1 (answer-only) and point 12 (my -10-point rule fails my own anchor) are the auditor's and correct.

### Replies (A = accept, AC = accept with change, R = reject)
1. **A.** My "answer-only alone lifts `dpo` to 63 %" was wrong; 63.3 % was pasted thinking plus the instruction. Run the zero-training control (`dpo` + "reply with only your final answer", 48 prompts x 8 draws).
2. **A.** Tabulate IFBench instruction types of the 48 reach prompts vs the 64 train prompts before training (free); split every cell's reach rate by "type seen / not seen".
3. **A.** Source (i) picks uniformly among eligible strict passes (no genuine-label preference); (iv) uses the identical cap, `finish`, `</think>` filter. Checker-bound (17c) uses the acceptance rate of the whole filter per prompt, a_i = (k_i + 0.5)/(N_i + 1) from `dpo`'s 16 stored draws.
4. **AC.** The test is the paired per-prompt difference (i) - (iv) (two-level bootstrap); "selection matters" needs its lower bound > 0; same for (ii) - (i). (iv) is drawn from the same filter, so lengths are comparable by construction; I report median tokens and answer words for both rather than matching.
5. **A.** Keep the 109-prompt matched set; the B claim states the leak (17 of 126 `dpo`-eligible prompts dropped using `s2300`'s behaviour).
6. **R for the fresh cell, A for the token report.** `olmo_ifcurve/reuse_inventory.csv`: "s2300, B, B_s2300, this run, 32 new, every stored s2300 draw on these prompts was part of their selection". The 69.4 % is exactly 1,066 / 1,536 strict in `results.csv` (arm B, s2300), from those 32 NEW draws (seed 48300, same sampler path), which played no part in selecting the set. So the rate has no winner's curse; the prompt selection changes which set it describes, not the bias of its estimate. No fresh `s2300` cell (saves $2.7). Residual caveat: new draws were taken without an adapter or LoRA path; the teacher-forced gate covers the LoRA path. Report tokens seen per cell; (i) trains on about half the tokens of (ii) at equal steps, so (i) vs (ii) is not token-matched and I say so.
7. **AC (budget).** Cap 512 for O1 and the M rungs, but by successive halving, not 3 lrs x 512: all 3 lrs run to 128 steps with dev evals at {16, 32, 64, 128}; the lr with the lowest dev loss at 128 continues to 512 (evals 256, 512). Early stop for any lr: dev loss above its running best at two consecutive evals. A16 stays at cap 128 (it overfits after one pass). "Unconverged" = dev loss still falls by more than 1 % of (L0 - L_best) between steps 256 and 512; a non-reach at an unconverged rung is reported "not shown". Cost per own rung about $2.3 (not $4.7).
8. **A.** M grid {3e-2, 1e-1, 3e-1}; edge rule kept; log max |b| and the number of B entries that moved.
9. **A.** Drop alpha = 16. alpha/r = 2 on every rung, so the old A16 lr carries over; the pre-registration states every grid as scale x lr. (My "matching BTC's scale" argument was weak: BTC print "alpha/2r", use 3.5e-4 and 1e-4 for LoRA, and NeurIPS-2025, the sparse-subset paper, gives no alpha.)
10. **A.** At the first M rung sampled, train M-all and M-attn (q, k, v, o only; 256 modules) with 3 lrs to 128 steps; continue and sample the one with the lower dev loss (selection on dev only).
11. **A.** "Moved" = dev-loss improvement larger than the spread between two A-init/mask seeds, taken from the confirmation pair (point 14) at the claimed rung and applied to neighbours as "not distinguishable if below the nearest rung's spread". Report answer-segment dev loss beside the total; select on the total (BTC App. F); report disagreements.
12. **A.** Primary: G = (cell - `dpo` plain) / (reference - `dpo` plain); `dpo` plain = 0 by the reach-set definition (0 in >= 256 draws); paired two-level bootstrap resampling the 32-draw reference too. "Reaches" (steering only) = G lower bound >= 0.80 (old adapter ~0.83). Stated limit: 48 prompts x 8 draws cannot resolve < ~10 points, so "equals the final model" is never claimed.
13. **A.** The n = 1 claim is the nested-prefix example at 8 draws (one cell). Two more first examples (8 draws each) only if G's lower bound >= 0.75; claim needs 2 of 3 to reach; report the median. Dev evals at steps {1, 2, 4, 8, 16, 32, 64} for n <= 4.
14. **A.** Keep the confirmation cell: new mask / A-init seed and fresh draws at the claimed smallest P; claimed only if both reach. It doubles as the noise pair for point 11.
15. **A, exact formula.** With batch 1, step i scores example i before updating (nats_i, tokens_i). Let t = selected step, m = min(t, n) (m = n when t >= n, i.e. the whole first epoch; m = t when t < n, only the examples that run actually saw). Then
    EDL(t) = sum_{i<=m} nats_i - (sum_{i<=m} tokens_i) x L_dev(theta_t), with L_dev in nats per token on the 32 dev responses, bits = nats / ln 2. Each prefix of a nested batch-1 run is exactly a one-pass run on D_m, so no recomputation. **Primary EDL for the capacity ladder is at fixed t = 64, n = 64 for every rung** (one pass, no selection effect); EDL at the dev-selected t is secondary; EDL/m at m = 16, 32, 64 (theta_m) is descriptive. Negative values are reported as they are.
16. **AC.** Add training-only rungs M65k, M16k for source (ii), and also (ii) M4k, M256, all at t = 64 with 3 lrs (about $1.0 each, not $1.5-3.7; at t = 64 there is no extension phase). Then C(P) = EDL_64(P)/EDL_64(A16) has points at 6.6e4, 1.6e4, 4.1e3, 256 weights, covering 7.6e3-7.6e4. (ii) A1 and O1 are dropped as the audit proposed (C ~ 1 trivially).
17. **A (a-d).** No P* claimed, only C(P); EDL/P reported for (ii) only, labelled "A16-referenced, t = 64, n = 64, not comparable to BTC thresholds"; (b) a rung with C < 0.95 may still reach (BTC Table 6 "*"); (c) own sources use sum log2(1/a_i); (d) conclusions read "reached with P weights and n examples", never "elicitation (BTC sense)".
18. **A.** Written into the pre-registration: a prior that own-source SFT fails at small P (TinyLoRA §3 defines our (i); NeurIPS-2025's long-CoT case needed 1e4-2e4 weights). Consequence for the search order below: descend from the top, because the small rungs are expected to fail.
19. **A, with the cost line below** (about $41 base, +$3.6-5 only if triggered).

### Revised cell table (H100 train $4.39/h; own 2.3 s/step, final 3.6 s/step; dev eval 20 s / 34 s; sampling cell $1.8 own-like, $2.7 final-like; engine load/tail shared, +$1.5 total)
| # | item | training | sampling |
|---|---|---|---|
| 0 | types table, pools rebuilt (point 3), dev answer-segment loss, EDL formula in logs, token/length stats | 0 | 0 |
| 1 | prompt control: `dpo` + answer-only, no adapter | 0 | $1.5 |
| 2 | own A16 n=64, 3 lrs to cap 128 | $1.3 | $1.8 |
| 3 | (iv) unselected A16 n=64 (same filter) | $1.5 | $1.8 |
| 4 | final A16 n=64 (replication of 63.5 %; (ii) reference), cap 128 | $2.1 | $2.7 |
| 5 | own O1 n=64 (first small rung): stage a (3 lrs, 128 steps) + extend best lr to 512 | $2.3 | $1.8 |
| 6 | next own rung by rule (below): O1 passes -> M4k with M-all vs M-attn pair ($3.4 train, winner extended); O1 fails -> A1 ($2.3) | $2.3-3.4 | $1.8 |
| 7 | confirmation at claimed smallest P: new seed, full procedure | $2.3 | $1.8 |
| 8 | own A16 n=4 (3 lrs, cap 64, evals 1..64) | $1.1 | $1.8 |
| 9 | own A16 n=1, nested-prefix example | $1.1 | $1.8 |
| 10 | (ii) capacity ladder, training only, t = 64: M65k, M16k, M4k, M256 | $4.0 | 0 |
| | shared engine load / tail | | $1.5 |
Base total: training $18.0-19.1 + sampling $17.0 + shared load $1.5 = $36.5-37.6; I carry **$41** with about $3-4 margin for dev-eval overhead and re-runs.
Conditional (only if the rule triggers): n=16 or the M256/M32 rung after M4k passes (T $1.1-2.3 + S $1.8 each), two extra n=1 examples (T $1.1 + S $3.6): **+$3.6-5**. Total with all triggers about $45-46.
If the cap binds, drop in this order: (ii) M256 training rung (-$1.0), the extra n=1 examples, the (ii) M4k rung, then n=16.

### Search rule (fixed in advance; sampled on G only)
Sample 2 and 4 first; if own A16 n=64 does not reach (G lower bound < 0.80), stop the own ladder, report 3 vs 4 vs 2 contrasts (the pipeline replication fails if 4 is not near 63.5 %). Otherwise descend: O1 (P = 655,360); if it reaches, M4k; if M4k reaches, M256 (conditional cell); if O1 fails, A1 (P = 8.4M). The claimed smallest P is the last rung that reaches, confirmed by item 7. Descending order follows the prior (point 18): the small rungs are expected to fail, so we spend cells where a pass is plausible.

### What is unchanged and what changed (short)
Kept: ladder A16/A1/O1/M-rungs at alpha/r = 2; n = 1, 4, 16, 64 nested; sources (i), (ii), (iv); dev-loss selection; confirmation. Changed: reference (no fresh cell: B_s2300 is independent); readout (G, bound 0.80); step budget (successive halving to 512); M lr grid; scale; attention-only variant; "moved"; EDL at t = min(t, n) formula and fixed-t ladder; descending search; n = 1 at 8 draws.

## Turn 4 — auditor (agent:claude, opus)

### Are the replies adequate?
| # | adequate? | why |
|---|---|---|
| 1–3, 5, 8–10, 14, 17, 19 | yes | Accepted as proposed. Point 3's smoothed a_i is fine; label it an estimate from 16 draws. |
| 4 | **no** | "(iv) is drawn from the same filter, so lengths are comparable by construction" is wrong. (i) also selects strict passes, and passing correlates with length on IFBench. Change: if the lower bound of (i) − (iv) is above 0, call the effect "checker selection, including its length correlate". Add a length-matched (iv) cell only as a conditional (≈ $3.6). |
| 6 | yes, withdrawn | Verified: `reuse_inventory.csv` row "s2300,B,B_s2300,this run,32 new", and `results.csv` arm B s2300 sums to 1,066 / 1,536 = 69.4 %. No winner's curse. Also checked the floor: `dpo` arm A has 0 strict passes in 672 stored reach draws. The reach set is defined on "0 genuine in ≥ 256", so floor 0 holds for strict too. |
| 7 | yes, with one caveat | See new attack N1. |
| 11 | partly | A single confirmation pair gives one spread estimate, not a noise distribution. Keep it, but call "moved" descriptive. |
| 12 | yes | Add the resolution to the stated limit (see N4). |
| 13 | yes, two gaps | (a) Pre-register *which* extra examples are used: prefix positions 2 and 3 of the seeded order, not a choice. (b) A failure at n = 1 is never re-checked, only a success is. "n = 1 does not reach" is therefore a one-example result; say so. |
| 15, 16 | **no** | See N2. |
| 18 | yes | The prior is written in. The search order is attacked in N3. |

### New attacks on the revision
- **N1. Successive halving.**
  - The lr is chosen at step 128 by dev loss. Small rungs usually favour the highest lr early, so the step-512 winner could differ; the risk is low.
  - The "unconverged" rule (still falling by more than 1 % of the gain between steps 256 and 512) will flag most M rungs. That is honest.
  - Pre-register that every "not reached" on an unconverged rung counts as "not shown" in the headline table, not only in the text.
- **N2. EDL at a fixed t = 64 measures one-pass absorption, not capacity.**
  - BTC's capacity ratio C uses θ\* at convergence (P1 Table 1, footnote 5). At t = 64 the small rungs are, by N1, still learning. So C(P) at t = 64 is low partly because small rungs learn slowly. That biases the result toward "capacity-limited", which is the exact reading the ladder is meant to test.
  - Change: name the t = 64 quantity "one-pass C". Run (ii) M16k and M4k at their best lr to 512 steps, ≈ $2 each, and take θ\* as the dev-best step. With m = n = 64 this is BTC's definition. Call a rung "capacity-limited" only if its extended C also stays below 0.95.
- **N3. The descending search.**
  - Jumping from O1 (655k, dense, o_proj) to M4k (4k, sparse B entries, all modules) changes construction and P at once. A pass at O1 followed by a fail at M4k locates nothing in between. Write the claim as "smallest reaching P ≤ 655k; below that, not shown".
  - The search stops at the first fail, so a false fail ends it early (see N4). The confirmation cell guards only against false passes.
- **N4. Power of "reaches".**
  - The old adapter's two-level half-width is about 6.2 points, ≈ 0.09 in G. A cell whose true G is 0.90 clears "lower bound ≥ 0.80" only about half the time; it needs a true G of about 0.97 to clear it reliably.
  - Pre-register this in the readout. Report G with its interval for every sampled rung, so a near-miss stays visible.
- **N5. Pass/fail rule for the replication cell 4.**
  - The rule is "near 63.5 %", undefined. The new final-source pool also differs from the 25 Nov pool: 109-prompt matched set, possibly different labels per (a), and a different set of 64 prompts.
  - Change: pre-register "pipeline OK" as the cell-4 G interval overlapping the old adapter's [0.77, 1.05] (= 53.7–73.2 % / 69.4). Treat a gap as a pipeline audit, not a finding.
- **N6. No rule for the prompt control.** Pre-register its reading: if cell 1 reaches, all adapter reaches are read as reproducing a format that one instruction elicits. Question A becomes "how small an adapter replaces one sentence of prompt".

### The two decisions
- **(a) No hand labels anywhere: support.**
  - An RL-free procedure using the same checker sees no labels, and RLVR rewards gamed strict passes too, so checker-only is the faithful analogue for question B.
  - Uniform rules across (i), (ii) and (iv) also remove label-coverage differences between sources.
  - Costs to state:
    - Training sets may now contain gamed passes, so the adapter can learn gaming while the readout is checker-only. Use labels for reporting only: report the labelled-gamed share in each training set, and hand-check a sample of reach passes in the cells that decide a claim (CPU).
    - Cell 4 is then not the 25 Nov pool rule. That makes N5's tolerance necessary.
- **(b) Headline G with interval; "reaches" only for steering: support.**
  - Add the paired difference in points and the absolute pass rate beside every G. G hides the reference's own interval.
  - Report prompts with a pass in 8 as secondary.

### Open points that no design change fixes
1. One RL run, one model, one checker, 48 reach prompts. The checker does not tell genuine passes from gamed ones.
2. The design resolves about 6 points (≈ 0.09 in G). "Equals the final model" cannot be claimed, and a false fail can end the search early.
3. No classification of elicitation versus teaching in BTC's sense is possible at n ≤ 64: their curves start at n ≈ 1,000, and their thresholds come from SFT on pretrained bases with another algorithm.
4. For small rungs, optimisation and capacity cannot be fully separated: 3 lrs, one mask, at most 512 steps. Failures are "not shown".
5. Training prompts are ones `dpo` already passes, and selection is one round. Selection can carry only what `dpo`'s passes contain. This is not iterated expert iteration and not RL's exploration on hard prompts.
6. Σ log2(1/a_i) bounds the bits the checker supplies only for an exact filter, with a_i estimated from 16 draws per prompt. Own-source EDL has no BTC meaning, because the whole-response loss includes the model's own sampling entropy.
7. Every result is conditional on the construction: B-only mask, seeded A, α/r = 2, module pool, dev selection on total loss.
8. The mild RL leak: the 109-prompt matched set uses `s2300`'s passes.
