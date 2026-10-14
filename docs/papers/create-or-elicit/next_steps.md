---
written_on: 2026-10-13
written_by: agent:claude
---

> **Reading copy (librarian, 2026-10-14).** This is run `paper-clean`'s `paper/next_steps.md` (fork
> `dan-pandori/nd-takehome`, branch `dan_paper-clean`, `86b705ee`). The writing run `create-or-elicit-paper` wrote it
> on 2026-10-13. `paper-clean` corrected item 1b's cost and added the presentation proposals at the end. The
> [paper-clean review](https://github.com/dan-pandori/nd-takehome/blob/792db421/review_paper-clean.md) confirmed
> the cost correction (12.35 ms per attempt; 2.4 × 10⁷ attempts ≈ 82 A40-hours). Its correction to presentation
> proposal 6, and its view of proposal 4's blog cuts, are added below, marked *[review]*.

# next_steps — what would make "Did Expert Iteration Create It?" a stronger paper (create-or-elicit-paper, 2026-10-13)

Costs use the A40 price the experiments paid ($0.59 per hour) and measured times:
- a 16-round ladder with evaluations beside it: ≈ 15 A40-hours ≈ $9 per seed;
- with create-or-elicit's post-ladder work (three k = 256 evaluations, timing, a large-k certificate and the
  compute-matched reach measurement): ≈ 22–27 instance-hours ≈ $13–16 per seed;
- continuing a ladder from r16: ≈ $5 per seed per 8 rounds;
- one k = 256 evaluation of the 106 strict members: 10–40 min per checkpoint (this run).

The list is ranked by how much each item would change the paper's claims. "Decision value" says which claim it would
rewrite. The two critique rounds converged on items 1 and 2.

## Experiments

### 1. GRPO from the knockout bases, at matched samples — ≈ $40–60 (6 seeds × 8–16 rounds)
- **Why first.** The title is scoped to EI because this is the decisive missing experiment.
  - In the closest controlled study (Abdulsalam et al.), GRPO built behaviours that pretraining lacked, while RFT, the
    family EI belongs to, plateaued.
  - In this project's earlier 3.2M organism, GRPO acquired a depth-3 box-nesting habit from draws with no such proof
    in 600,000 samples (12/12 arms), where EI did so in 1/3.
  - Against that, Interplay's 0 % null was itself obtained with GRPO.
- **Design.** `grpo_state.py` (run `grpo-state`) from the six knockout bases. Use the same targets and the same
  attempts per round (k = 32), with group-normalised advantages. A lone success then gets an advantage of
  √31 ≈ 5.6 standard deviations, against roughly 4 of ≈ 90,000 records in EI's fine-tuning mix. Evaluate the strict
  106 at r8 and r16, plain and guided.
- **Write per-seed predictions first.** All-fail groups give no gradient, so by sampling alone GRPO at k = 32 would
  most likely first succeed where EI did (s1, s2 and s5). A rescue of s0, whose base had no strict success in 1.5
  million attempts, would point to drift driven by the negative-sample term. Include a pass@4 arm, the only GRPO variant level with EI in this
  organism (grpo-best), and a signed-update arm without a group baseline.
- **Decision value.** If GRPO acquires families from knockout bases, "needs exposure" becomes an EI-specific statement
  and the paper's discussion turns on update rules, not data. If it does not, the result generalises across the two
  standard algorithms.

### 1b. Measure the knockout ladders' σ rate along the run — ≈ $25–30 (≈ $14 at r8 and r16 only); ≈ $48 (≈ $24) at C4's measured rate, see the correction below
- A5's "the rate rose during RL" is post hoc and rests on plug-in base rates for held-out theorems. Sample the 123
  strict training theorems about 10⁶ times per checkpoint (≈ 8,000 attempts each) at the saved r4, r8, r12 and r16
  checkpoints of every knockout ladder. All 200 ladder checkpoints are in the bucket. That is 2.4 × 10⁷ attempts at
  6–8 ms each, ≈ 40–55 A40-hours. *(Correction, paper-clean 2026-10-14: C4 measured 12.35 ms per attempt on a knockout
  base at batch 1,024 over 19 strict members, `artifacts/coe_paper/c4/ko_s1.json` (5.5 ms on `916` alone). At that rate
  2.4 × 10⁷ attempts take ≈ 82 A40-hours ≈ $48, and r8 and r16 only ≈ $24; the review flagged this.)* It resolves pooled rates down to a few × 10⁻⁶. It turns A5 into an observed
  trajectory, and it separates "EI samples too little" from "EI amplifies too narrowly" (readings 2
  and 3).
- Pair it with a **replay-only control from the knockout bases** (the ladder's fine-tuning on replay alone). That
  tests whether any rate rise is RL's doing or general fine-tuning drift.

### 2. Exposure during RL vs in pretraining (replay swap) — ≈ $30–45 (3 + 3 seeds)
- Two arms: (a) knockout bases with the 1,550 dose records added to their replay; (b) full bases with K12⁻ replay.
- **Decision value.** It separates the two parts of "exposure", which the design cannot (paper §2, §8), and it
  doubles as a test of seeding at volume. In the earlier organism a single injected proof ignited the pattern within
  1–2 rounds (round-2 Run 3), so (a) is expected to ignite. If it does, the data-filtering point becomes "filtering
  pretraining is not enough if RL data or demonstrations carry the composition".

### 3. The 0.1 % dose arm, and 0.01 % — ≈ $33 per dose (3 seeds × 16 rounds; evaluations at r0 and r16)
- The 0.1 % set is built (`data/coe/train_k12dose01.jsonl.gz`); 0.01 % needs one more build.
- **Decision value.** A dose–response curve (0 / 0.01 / 0.1 / 1 / 10.9 %) tells a threshold from a graded response,
  which two doses cannot. It also lines up with Interplay's 0 / 0.1 / 1 % grid. With a large-k base-rate measurement per
  dose, acquisition could be plotted against base rate × EI budget, which is the quantity the waiting-time argument is
  about.

### 4. Per-theorem base rates at n ≥ K, and per-family certificates — ≈ $3–10
- C4 (this run) re-sampled knockout s1's 18 converse-contraposition members at n ≥ K, and three of the four members
  its early-stopped sample had put within reach came back at 0 in 18,432 attempts. Extending this to every strict
  member of every knockout base, at ≥ 60 K per family, is ≈ 60 × 20,000 × 6 families ≈ 7 M attempts per base. That
  would make a per-family certificate (and hence the pre-registered CREATED verdict) reachable. It is about $8–10 per
  base at the knockout bases' 6–8 ms per attempt *(≈ 24 A40-hours ≈ $14 per base at C4's measured 12.35 ms;
  correction, paper-clean 2026-10-14)*.
- Also compute the knockout certificates at all five budget allocations; this is CPU-only once the samples exist.

### 5. A matched-atom-frequency knockout — ≈ $90–100 (6 seeds)
- Remove σ while holding the DN and ¬-introduction shares at K12's levels, as CFQ holds atoms fixed while compounds
  diverge. Abdulsalam et al.'s matched-fraction control does the same.
- **Decision value.** It separates exposure to the composition from the frequency of its constituents, the knockout's
  main side effect (DN 13.7 → 3.3 %, NEGI 15.2 → 5.2 %).

### 5b. Two to four seeded rounds — ≈ $10–15
- The one-round seeded null (+2 on 1 of 5 seeds) contrasts with the earlier organism's next-round response to a single
  injected proof (6 and 17 targets). Run 2–4 rounds with the same seed proof, and also with an in-family proof, and
  report the training-side count of strict theorems with a free positive reductio. This tells "this organism resists
  one-proof seeding" from "the read-outs differ".

### 6. Longer and more exploratory knockout ladders — ≈ $5 per seed per 8 rounds
- Continue knockout s0 and s2–s5 to r24–r32, and add an exploration lever:
  - T = 1.0 rollouts;
  - k = 256 on the targets the ladder fails;
  - "assume ¬G" injection at positive goals with probability ε.
- In the earlier organism, one round at k = 128 or T = 1.0 changed almost nothing, so expect little from the first
  two. The waiting-time result (3 seeds first proved strict targets sooner than their bases' held-out rates predict) suggests that
  longer runs may help more than any single round's exploration.

### 7. A second composition knockout — ≈ $100 (6 + 6 seeds)
- One composition is one data point. A second one would show whether "needs exposure" is about σ or about compositions
  in general. Candidates: ∨-elimination whose branches each close by ¬-elimination, or a non-classical composition.

### 8. Scale — ≈ $60–120 per pair of arms at 30–40M parameters
- One knockout and one full arm at about 4× the parameters, 6 seeds each. The earlier organism found no rise in
  reductio rates from 3.2M to 85M. That result came from a different interface and an older checker, and needs
  repeating here.

### 9. Power sampling where it claims to work — ≈ $3–5
- Compare pass@1 / pass@4 of the power-sampled base with r8's pass@1 at matched tokens, on theorems the base solves
  sometimes. Use a sampler that passes the positive control: SMC, or more MH steps. This replaces the paper's
  uninformative null.

### 10. Interface pilot and a classical-rich target pool — ≈ $25 and ≈ $20
- Whole-proof vs proof-state from the same data. Separately, a target pool in which classical theorems are a large
  share of the RL signal. In the current pool, 123 of 4,495 training theorems require the composition or a variant.

## CPU-only analyses (free; can be done in this branch)

- **Proof-route anatomy of knockout s1** (`found` files in the bucket, ≈ 1 GB). Classify every proof of a strict
  member by route (σ / other σ⁺ / lemma route) per round. This would show whether the lemma route, absent from all
  pretraining, emerged during RL or was already in the base.
- **An unseen-mass (Good–Turing) estimate** for the knockout bases' σ rate, from the large-k samples.
- **Fit Tsilivis et al.'s STaR recursion** to the full and dose ladders' per-round σ shares, as a quantitative
  amplification model.
- **A table placing published creation claims on the evidence hierarchy** (ProRL, Burton, Abdulsalam, Shen et al.'s
  chess). This would make the measurement contribution concrete for LLM readers.

## Literature

- Read in full (abstract-level so far):
  - Ji et al. (2601.21590), the strongest "sharpening matches RL" claim;
  - Talwar, Nemeth & Moss (2610.13653), for a convergence diagnostic to apply to our sampler;
  - Ran-Milo et al. (2601.15158), the theory closest to the knockout;
  - Li & Liu (2605.08368), the closest conceptual frame (write a note).
- Check Tsilivis et al.'s appendix for a p_cot = 0 run.
- Pretraining-data filtering for safety (e.g. O'Brien et al. 2025, "Deep Ignorance"; not yet verified). This is the
  natural home for the data-filtering point.
- Physics of LMs Part 3.1 (fine-tuning extracts only what pretraining made extractable) and Ruis et al. on procedural
  knowledge in pretraining. Both are relevant to "exposure", and neither is yet in the verified notes.

## Presentation proposals (paper-clean, 2026-10-14)

Not applied. Each would either change more than presentation, or touch files that the review and `LEDGER.md` cite by
file and line.

1. **Reconcile the rest of the review's sub-audit (CPU only, ≈ 1–2 agent-hours).**
   - `review_coep/phase2/{front,results,back}.md` hold about 79 rows marked W or D. The review's P1–P17 and smaller
     items, and the librarian's 13 "possible errors", cover part of them, but no file maps each row to its resolution.
   - Rows still open in the clean text include:
     - I25: "almost all on converse contraposition" is 11 of 14 successes;
     - I43: §1's "the waiting time scales as 1/(kNr)" is listed as a finding that may transfer, but it is the
       definition of the post hoc Poisson null;
     - R47: van der Weij et al. are co-cited for a definition that is Hofstätter et al.'s.
   - A pass that maps every row and resolves the open ones, recorded as in `CLEANUP.md` §2, would close the audit.
2. **Name the section files in PDF order.** Today `03_setup.tex` is §2 and `02_related.tex` is §5, which slows every
   reader of the sources. Rename them once no further review will cite the current line numbers.
3. **A figure for "rare, not absent" (CPU only).** The evidence that the knockout bases could make the move is spread
   over a table and prose (§3.3–3.5): large-k successes per base and family, the certificate bound, and C4's
   re-sample. Proposed figure: one panel per knockout base, showing per-family successes per million attempts with
   95 % intervals, next to the full bases' r0 rates and the K_train(16) bound. Sources: `review_coe/out_cert.json` and
   `paper/data/c4_within_reach.json`.
4. **The blog's cuts.** The 2,500-word cap removed a few results (`CLEANUP.md` §4):
   - the likelihoods' movement during RL (Fig. 2 still shows it);
   - the one-round seeding null;
   - the best knockout run's 3 of 106.

   If the cap is relaxed, restore the likelihood sentence first: it is the blog's only statement that RL did not raise
   the knockout models' likelihood of the known σ proofs.

   *[review]* The review would restore two of these cuts:
   - the seeding null (F1, should fix). The paper reports it as a null and as a missed pre-registered expectation,
     and the earlier review's B2 held the blog to the paper's negative and null results;
   - the likelihood sentence (F2, recommended).

   The blog has 13 words of headroom: F1 alone (≈ 23 words) needs ≈ 10 words cut elsewhere, F1 and F2 (≈ 38) ≈ 25.
   Dan's call.
5. **Watch the abstract's length.** It is exactly 250 words (`wc -w sections/00_abstract.tex`); any later edit needs
   a matching cut.
6. **Two labels the blog lacks, in both the reviewed and the clean version** (the equivalence check's m8).
   - The "genuinely classical proofs" counts behind knockout s1's ladder dynamics are the free positive reductio, a
     post hoc measure (§7 of the paper).
   - The continuation to round 24 was exploratory.

   Labelling both costs about ten words. *[review]* The blog is at 2,487 words, 13 under the cap, so both labels fit.
