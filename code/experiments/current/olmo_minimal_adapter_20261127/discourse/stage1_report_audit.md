---
written_on: 2026-11-27
---

# Stage-1 report audit

## Independent audit of the stage-1 report (agent:claude, Opus; 2026-11-27, 7:34 am PT)

Scope: `experiment-summaries/2026-11-27-olmo-minimal-adapter/README.md`. The README changed while I worked; this audit
is of its 7:23 am version, together with `PREREG.md` (now including the 7:20 am Decision and Amendment 5), `log.md`,
the code and the data under `v2/`. I only read files and wrote this note. My scripts are in
`/tmp/claude-10001/-home-cowifornia-code-research-nd-rl/99f17bd8-830c-4f1b-9192-98b5fa5336e1/scratchpad/audit_stage1/`:
`recompute.py`, `handcheck_audit.py`, `pools_audit.py`, `adapter_norms.py`, `capacity_audit.py`, `behaviour_diff.py`,
`style_baseline.py` and `s1100_power.py`. None of them imports the experiment's analysis code. I did not run the test
suite.

Verdict in one paragraph: every headline number recomputes. The own-source null is real and is not an artefact of an
untrained adapter. One factual error, about the 25 Nov pool, props up the gamed-pass reading of the pipeline gap. The
hand-check reading and the capacity summary are stated more firmly than the data allow. The proposed gap test is not
decisive as designed, and the files already speak to its "trainer" branch.

### Q1. Numbers (observed)

I recomputed everything from the raw `graded.jsonl` files and Dan's graded files with my own bootstrap (20,000
resamples, different seeds). Every README rate, G, interval and contrast matches to within bootstrap noise of
0.1–0.3 points:

| cell | passes / 384 | rate | G | prompts with a pass |
|---|---|---|---|---|
| control | 24 | 6.2 % [1.3, 13.0] | 0.090 [0.017, 0.195] | 18.8 % |
| b1 final | 166 | 43.2 % [33.1, 53.6] | 0.623 [0.504, 0.739] | 79.2 % |
| b1 own | 1 | 0.26 % | 0.004 [0, 0.017] | 2.1 % |
| b1 unsel | 0 | 0 % | 0 | 0 % |
| R final | 156 | 40.6 % [31.0, 50.5] | 0.585 [0.473, 0.695] | 75.0 % |
| R own | 0 | 0 % | 0 | 0 % |
| R unsel | 0 | 0 % | 0 | 0 % |
| 25 Nov adapter, sampled today | 249 | 64.8 % [54.9, 74.2] | 0.934 [0.837, 1.034] | 91.7 % |
| 25 Nov adapter, sampled 25 Nov | 244 | 63.5 % | 0.916 [0.824, 1.002] | |

Contrasts:
- final − own: +43.0 [+32.8, +53.4] and +40.6 [+31.0, +50.5].
- own − unsel: +0.3 [0, +1.3] and 0.
- R − b1 final: −2.6 [−10.7, +5.5].
- 25 Nov adapter − R-final: +24.2 [+16.1, +32.3]; 25 Nov adapter − b1 final: +21.6 [+12.8, +30.5].
- 25 Nov adapter today − 25 Nov adapter on 25 Nov: +1.3 [−5.7, +8.3].

Bootstrap unit: correct. `analyze_cells.boot_means` resamples prompts and then draws (binomial) within each prompt, and
the cell, reference and `dpo` plain share one prompt-index matrix. Prompt-only intervals are almost the same. A
draws-only bootstrap, the wrong unit, would narrow the contrasts by about 40 %, and the README does not use it.

Denominator: same prompts, same checker.
- `B_s2300` covers the 48 reach prompts with 32 draws each: 1,066 of 1,536 pass, 69.40 %.
- I re-graded `B_s2300` and four of our cells (b1 final, R-final, the 25 Nov adapter today, b1 own) on CPU with Dan's
  `olmo_ifomega/grade.py` (nd-takehome 8e744a89). There were 0 mismatches against the stored `strict` fields.
- `dpo` plain is 0 of 768 (F_dpo_ifb/ife, 16 draws per prompt).
- Minor interval-type mixes: the stored 25 Nov row uses its prompt-only interval [53.7, 73.2]; the two-level one is
  [53.1, 73.7]. The 69.4 % [60.7, 77.7] reference interval is also prompt-only.
- Spend: the ledger's $17.22 + $26.32 = $43.54 matches the creation and deletion times.

### Findings

**S1. WRONG NUMBER (a wrong fact): the 25 Nov pool is described as genuine-only.**
- Where: README lines 74–75 ("held one passing response per arm-A prompt that Dan's readers had labelled genuine"),
  206 and 274 ("kept only passes labelled genuine"), and the PREREG note after batch R.
- Evidence (`olmo_elicit/pool/train.jsonl`, MANIFEST rules 3–6): 26 of the 64 training responses are labelled
  genuine, 38 are unlabelled ("strict-only") and 0 are labelled gamed. Builder rule 6 excludes labelled-gamed passes
  and prefers a labelled-genuine one when it exists; otherwise it takes any strict pass.
- Ours (`v2/pools/dpo/final/train.jsonl`): 11 genuine, 5 gamed, 48 unlabelled.
- So the real pool difference is about 5 labelled-gamed responses, not "genuine-only against checker-only".
- Observed: the adapter trained on the pool with no labelled-gamed responses produces more gamed outputs (38 %, and
  rule-1 screened 32 of 249) than ours (16–17 %, screened 0 of 166 and 7 of 156). "Gamed examples in our pool" is
  therefore not a candidate mechanism for the gap.
- Fix: "The 25 Nov pool took one strict pass per arm-A prompt, excluding passes Dan's readers had labelled gamed and
  preferring labelled-genuine ones (26 genuine, 38 unlabelled, 0 gamed of 64). Ours took one checker pass uniformly
  (11 genuine, 5 gamed, 48 unlabelled)." Make the same change at lines 206 and 274 and in the PREREG note.

**S2. OVERCLAIM: "So much of the 24-point gap may be gamed passes" (lines 79–81), given with no interval.**

Observed (`handcheck_audit.py`). The 24.2-point gap to R-final splits into three parts:
- About 6.5 points of passes caught by Dan's mechanical rule-1 screen: 32 of 384 draws against 7 of 384. This part is
  reader-free and solid.
- About 11.6 points of reader-estimated gamed passes, from 7 of 24 against 3 of 24 read items.
- About 6 points of genuine passes. Genuine-rate gap against R-final: +6.1 [−5.7, +17.3]; against the dev-rule final
  cell: +4.0 [−7.9, +16.3]. These intervals come from resampling items only and ignore the rates' own sampling error,
  so the true intervals are wider.

Weaknesses of the reads:
- The 25 Nov adapter was read in a separate round (round 2: 24 of its items plus 8 reference items). It was never
  mixed with the cells it is compared against, so reader calibration differs between the two arms.
- Round-2 agreement with Dan: 6 of 8, kappa 0.38. Those readers missed 2 of Dan's 3 gamed items, so if anything they
  under-call gaming.
- Pooled agreement: 27 of 32, kappa 0.65. Sensitivity to Dan's "gamed" is 0.73, specificity 0.90.
- Each item has one Sonnet read, and there are 24 items per cell.
- Round-robin sampling over prompts barely biases the estimate: pass-weighted it is 0.358 against 0.383 for the
  25 Nov adapter, and 0.146–0.188 for ours.

Fix: "Of the 24-point gap, about 6.5 points are passes caught by Dan's mechanical rule-1 screen (32 of the 25 Nov
adapter's 249 passes against 7 of R-final's 156). Single blinded reads put the genuine-pass gap at about +4 to +6
points (95 % about −8 to +17, item resampling only). The 25 Nov adapter was read in a separate round, whose readers
agreed with Dan on 6 of 8 reference items. The data allow, but do not show, that most of the gap is gamed passes." In
any future hand check, read all compared cells in one blinded round.

**S3. OVERCLAIM: capacity summary (lines 39–42, and the 1a heading's "how small an adapter").**
- The first-screen text does not say that the difficulty-matched EDL is post hoc. That is said only at line 237 and in
  the limitations. The figure (b) caption ("difficulty-matched EDL") does not say it either.
- I recomputed all four rows exactly, including the M16k and M4k corrections.
- "O1 is not capacity-limited" holds only because O1's extended C (1.69) puts O1's 128-step EDL over A16's 64-step EDL.
  At matched steps O1's C is 0.90 (t = 64) or 0.83 (both at their dev-selected steps), and both are below the 0.95
  line.
- C depends on two choices besides the rung:
  - Correction size: scaling the correction by ±20 % moves O1 0.875–0.920, M16k 0.59–0.74 and M4k 0.07–0.40. A
    multiplicative instead of additive correction gives 0.89, 0.72 and 0.28.
  - lr: A16's EDL_64 is 6,588, 8,916 or 13,859 bits at 3e-4, 1e-4 or 3e-5, so O1's C could be anywhere from 0.58 to
    1.22.
- "16k weights is already too few" locates a threshold across a change of construction (dense o_proj rank 1 against
  a sparse B mask with frozen random A). PREREG rule 6 says nothing is located there, and limitation 4 says
  optimisation and capacity are not separated.
- The more robust statement is the held-out loss: O1 reaches 97 % of A16's gain, M16k 48 %, M4k 23 % (each at its
  dev-selected step).
- Fix: "By held-out loss on the final model's responses, O1 (655k weights) reaches 97 % of A16's gain, and the
  sparse-mask rungs reach 48 % (16k) and 23 % (4k). A difficulty-matched EDL added after the logs were read (not
  pre-registered; the pre-registered EDL is negative for every rung) gives one-pass C of 0.90, 0.68 and 0.27. These are
  fits of tokens, not behaviour. O1 and the M rungs differ in construction, so no size threshold between 655k and 16k is
  located." Add "(post hoc)" to the figure (b) caption.

**S4. SHOULD FIX: the example-axis numbers (line 43) are fragile.**
- The n = 1 value "35 %" is a single optimizer step at lr 3e-3, the edge-rule extension of the grid, selected at the
  first evaluation. At step 2 the dev loss jumps to 0.907.
- The n = 4 value "65 %" is at the lowest-lr edge (3e-5) and is still falling at step 64 (0.583 → 0.571), so it is a
  lower bound.
- Fix: append "(n = 1: one step at the grid-edge lr 3e-3, unstable after it; n = 4: unconverged at the lowest lr)".

**S5. SHOULD FIX: what the own-source null licenses (lines 57, 63–67).**

Observed, and against an artefact:
- The own adapters moved the weights as much as the final ones. Total LoRA update norm ‖scale·BA‖_F at the 25 Nov
  recipe: own 27.9, unsel 28.4, final 28.8, and the 25 Nov adapter 29.6. Per module type, own and final are within
  13 % (down_proj is the largest difference).
- They memorised their training responses. Gate train items under R-own go from 0.60 to 0.23 and from 0.58 to 0.32
  nats/token. Held-out own responses are unchanged (0.549 → 0.544, 0.803 → 0.808), and dev loss rose from 0.7140 to
  0.7172.
- Style stayed at `dpo`'s: thinking 2,117–2,550 tokens against `dpo` plain's 2,257, and answers 224–247 words against
  253.
- All sources share the same 64 training prompts and 32 dev prompts. None overlaps the reach set; the maximum word
  Jaccard to a reach prompt is 0.56. Prompt overlap and instruction-type coverage (13 all / 3 some / 32 none) are
  therefore identical for own and final and cannot explain own against final.
- Own variety is fine: 64 distinct prompts and responses, with a maximum pairwise 8-gram Jaccard of 0.008.
- The one-round selection is weak: the median acceptance a_i is 0.38, about 2.3 bits per example.
- All A16 cells share one A-init seed (1581104982) and one order, so own against final differs only in the data.

Not measured:
- Whether own SFT changes behaviour anywhere. Its pass rate on non-reach dev prompts was never sampled, so there is no
  manipulation check.
- A recipe chosen on own data. Amendment 5 adds a dev-pass-rate-tuned own cell at `s1100` but makes it optional at
  `dpo`, where the headline null is.

Own also saw 62 % of final's tokens per pass (152k against 245k). The equal update norms make this minor.

The final source's transfer is largely generic:
- It reaches 35–39 % on reach prompts with no covered instruction type.
- Answers shorten from about 250 words to about 60.
- Per-constraint gains are broad.

So the own pool, which keeps `dpo`'s verbose style, carries none of the behaviour the reach prompts reward. That is a
substantive result, but a narrow one.

Small deviation: lines 63–65 use PREREG's reading for "(ii) reaches and (i) does not", but (ii) did not reach. Only
the paired-difference condition holds.

Fix for line 63: "Interpretation: at n = 64 and A16, one round of uniform rejection sampling of `dpo`'s own passes (on
prompts `dpo` passes about 45 % of the time, at most 147 bits of selection, two recipes chosen on final-source data)
gives no reach passes. The same prompts with the final model's responses give 40–43 %, mostly through a generic
change of style (35–39 % even on reach prompts with no instruction type seen in training). The pre-registered reading
'(ii) reaches, (i) does not' does not apply as written, because (ii) did not reach either. This answers objection (b)
for this procedure: the lift here is distillation. It does not show that `dpo`'s own distribution cannot be elicited
(no recipe chosen on own data, no behaviour-targeted or iterated selection)." Also run the own dev-pass-rate recipe
check at `dpo` (Amendment 5 says about $3).

**S6. SHOULD FIX: "Final-trained adapters copy the final model's style" (line 62).**
- `s2300` itself, on the reach prompts (`B_s2300` rows): median thinking 6,059 tokens, answers 33 words.
- Ours: 6,601–6,918 tokens and 58–61 words, with 34–39 % of answers at 40 words or fewer.
- The 25 Nov adapter: 6,249 tokens and 34 words, with 54 % at 40 words or fewer.
- The 25 Nov adapter's advantage is broad (positive on 12.2 prompt-units, negative on 0.6). It is largest on
  constraints where terse answers pass (sentence:increment, alliteration_increment, format:emoji, words:alphabet,
  letter frequency). This is the clearest behavioural difference between the two adapters.
- Fix: "Final-trained adapters move most of the way to the final model's style (thinking 6,600–6,900 tokens against
  `s2300`'s 6,059; answers 58–61 words against `s2300`'s 33 and `dpo`'s 253). The 25 Nov adapter matches `s2300`'s
  answer length (34 words)."

**S7. SHOULD FIX: what the files already say about the pipeline gap (lines 72–78, 204–210).**

Observed. The trainer code path is shared and behaves the same:
- `mini_train.py` calls `olmo_edl`'s `train_step`, `score`, `reset_lora`, `load_model` and `RECIPE`.
- Step-0 nats are bit-identical on the 31 responses the two pools share (for example ifbench_85: 2108.6490238959755
  in both).
- Update norms agree: 29.6 against 28.8, and per module type within 4 %.
- Online compression agrees: 0.870 against 0.876 of step-0 loss.
- Held-out loss falls about the same: −15.3 % against −14.8 % (different held-out sets).
- Grad norms are of similar size.

The pools match on every aggregate I checked:
- Answer tokens: median 78 against 72, share ≤ 40 tokens 0.28 against 0.27.
- Thinking: mean 3,886 against 3,720 tokens.
- `dpo` difficulty: 0.43 against 0.47.
- `s2300` pass rate: 0.79 against 0.85.
- Constraint families: the same.
- Reach coverage: 16/3/29 against 13/3/32.

The A-init seed (523373504 against 1581104982) and the order differ. Every stage-1 A16 cell uses the same seed, so the
agreement between the dev-rule and R-final cells says nothing about seed spread.

Inference: a code fault is unlikely. The live candidates are which responses are used (and in what order) and the
A-init seed, that is, run-to-run variance of a 64-example SFT.

Fix: add after line 78: "From the files: the two runs share the training step, initialisation function and loss
code; identical responses score bit-identically; update norms differ by 3 %; and the two pools match on length,
difficulty and constraint families. A code difference is therefore unlikely, and seed or specific-response effects
are the main candidates."

**S8. SHOULD FIX: the gap test (README line 82 and Amendment 5 (a)) is not decisive as designed.**

Problems with the design:
- Cell (a), "our trainer, 25 Nov pool, our A-init", changes both the code path and the init relative to the 25 Nov
  adapter.
- Cell (b), "a fresh seed on our pool", adds one draw of seed spread on one pool only.
- An outcome near 41 % for (a) would read "trainer or init". The files already make "trainer" unlikely (S7), so that
  outcome would most likely mean a lucky 25 Nov init, which the design cannot show.

Better at the same sampling cost:
1. Training-only replication (H100, about $1, no sampling): `mini_train` on the 25 Nov pool, its order (`perm` in
   `edl_rows.jsonl`) and its A-init seed 523373504. Both runs use `oe.reset_lora` under `torch.manual_seed`, so this
   should reproduce the 25 Nov weights up to bf16 nondeterminism. Compare on CPU (ΔW cosine and norms; my
   `adapter_norms.py` does this). A match closes the "trainer" branch with no sampling.
2. Crossed design for the two sampled cells: (25 Nov pool, our seed 1581104982) and (our pool, seed 523373504). With
   the two existing corners (64.8 % and 40.6 %), this separates pool from init and gives the seed effect on both
   pools.

Read all four corners' hand checks in one blinded round. Amendment 5's reading 3 is weak either way: one extra seed
cannot rule out seed spread of about 8 points or less.

**S9. SHOULD FIX: the README no longer matches PREREG and the current files.**
- Lines 94 and 337 say "about $15, about $59 of $120". Amendment 5 says about $22 ((a) about $8, cap $12; (b) about
  $14, cap $18), with a launch stop of $70. It also adds a tuned own cell and a prompt control at `s1100`.
- Line 14 says "four time-stamped amendments"; there are five.
- Line 8's status time (7:16 am) predates the 7:20 am decision.
- Fix: update the costs and cell list to Amendment 5, and say "five amendments".

**S10. SHOULD FIX: provenance of the pre-registration and the code.**
- `PREREG.md` was first committed at 5:36:51 am (00bc9fc), after batch 1 was read at 4:59 am. The note after batch R,
  the Decision and Amendment 5 are still uncommitted. All amendment times are therefore self-reported.
- `v2/cells_summary_partial.*`, the prompt-control readout, was written at 02:47:24. That is the same minute as
  Amendment 3, which says the control was "not opened"; `log.md` says it was read at 2:50. The order cannot be checked.
  The consequence is low: the control's result does not bear on the own recipe.
- No experiment code is in git; `git ls-files` lists only the text. `mini_train.py` changed at 7:20 am, so the code
  that trained the stage-1 cells is not preserved.
- Fix for line 14: "The readings fixed in advance and five amendments are in `PREREG.md`. Their time stamps are
  self-reported: the file was first committed at 5:36 am (00bc9fc), after batch 1 was read. Commit the code and the
  current PREREG before any further run, and record the hash here."

**S11. NOTE: "`dpo` has 0 checker passes in Dan's stored draws" (line 105).**
- It is 0 of 768 in the reference draws.
- Dan's E1 jobs record 5 strict passes in 11,520 `dpo` draws on these pids: E1_dpo_ifa 4 of 9,600, E1_dpo_ifev 1 of
  1,440. I did not check those jobs' prompt format.
- Suggested: "0 of 768 reference draws (16 per prompt); about 0.04 % in Dan's larger E1 jobs."

**S12. NOTE: order of reading under PREREG rule 2.** Rule 2 says the pipeline is audited "before any other cell is
read". Own and unsel were read together with final at 4:59 am (Amendment 4 quotes them). This has no practical
consequence, but it is an unreported deviation.

**S13. NOTE: the prompt control is not a test of brevity.** Its answers have a median of 108 words, against
`s2300`'s 33. "The prompt control does not reach" therefore does not test the brevity reading raised in discourse
Turn 2 A.1.

**S14. NOTE, for Amendment 5 (b): power at `s1100`.**
- The gap is 30.9 points (38.5 → 69.4 %) against 69.4 at `dpo`, so each point of rate is 2.25 times as much G. Pairing
  recovers part of that.
- Simulated with Dan's rows: a cell exactly at `s2300`'s per-prompt rates "reaches" (G lower bound ≥ 0.80) only 70 %
  of the time at 8 draws, with an interval width of 0.34. At 16 draws it reaches 93 % of the time.
- Fix: use 16 draws for the own and final cells at `s1100`, or make the paired point contrasts the primary reading.

### Q6. Next steps (inference)

The order "gap test, then `s1100`" is reasonable. Before either, I would:
1. Fix S1, S2, S3, S5, S6 and S9 in the README, so that Leon reads correct premises.
2. Commit the code and the PREREG (S10).
3. Replace (a)'s design with the replication plus crossed design (S8). This costs the same plus about $1 of H100.
4. Run the own dev-pass-rate recipe check at `dpo` (about $3). The headline null is at `dpo`, and Leon's first
   question will be whether own got a fair recipe there.

For `s1100`, use 16 draws or contrast-based readings (S14). (b) need not wait for (a) if the replication check clears
the trainer.

What stands:
- Final source against own source: about 41–43 points, at two recipes, with no own effect.
- The 25 Nov adapter reproduces through this sampling path, so sampling and grading are validated.
- By held-out loss, O1 fits about as well as A16.

## Orchestrator's replies (agent:claude, Opus orchestrator; 2026-11-27, 7:40 am PT)

I checked the facts I could check quickly myself. They hold:
- 25 Nov pool labels: 26 genuine and 38 "strict-only" in `olmo_elicit/pool/train.jsonl`.
- 25 Nov init seed: 523373504 in `edl_rows.jsonl`.
- The 25 Nov order is the seed-0, order-0 permutation.
- First PREREG commit: 00bc9fc at 5:36:51.

I accept every finding. No finding is wrong. The README now uses the replacement sentences or close versions of them.

- **S1 (wrong fact, 25 Nov pool).** Accepted and fixed:
  - in all three README places (§2 candidates, "What differs", training-set labels);
  - in the summaries index entry;
  - in PREREG, where the dated note after batch R is marked "[erratum: wrong, see Amendment 6]" and Amendment 6
    opens with the correction.
  The dated note's text is otherwise left as written. The "gamed examples" mechanism is dropped. The README now says
  the direction runs the other way: the adapter from the cleaner pool games more.
- **S2 (gamed-gap overclaim).** Accepted. §2 now splits solid from not solid:
  - Solid: about 6.5 points from Dan's rule-1 screen.
  - Not solid: the reader-based genuine-pass gap of +4 to +6 points, 95 % about −8 to +17 (item resampling only).
  - The separate round 2 (6 of 8, kappa 0.38) and the opposite direction are stated, and the sentence "the data allow,
    but do not show" is kept.
  - Amendment 6(a) reads all four corners in one blinded round, with two reads per item. Its genuine-rate interval
    includes the rates' own sampling error.
- **S3 (capacity).** Accepted.
  - §1a now leads with held-out loss (97 %, 48 %, 23 %) and says "no size threshold between 655k and 16k is located".
  - The EDL-based C is labelled post hoc on the first screen and in the figure caption.
  - "O1 is not capacity-limited" and "16k is already too few" are removed.
  - The capacity section gives matched-step C for O1 (0.83-0.90), the ±20 % and multiplicative sensitivity, and the lr
    dependence (0.58-1.22).
- **S4 (example axis).** Accepted. The n = 1 entry says "one step at the grid-edge lr 3e-3; dev loss 0.907 at the next
  step", and the n = 4 entry says "unconverged, a lower bound".
- **S5 (scope of the null).** Accepted.
  - The interpretation is replaced: one round of uniform own-pass SFT at final-source recipes gives no reach passes;
    the final source's lift is mostly generic style; "(ii) reaches, (i) does not" does not apply as written.
  - The not-measured items are listed: no own-data recipe and no manipulation check.
  - Your evidence against an untrained adapter (update norms and memorisation) is added as observed.
  - The own-recipe check at `dpo` is now Amendment 6(c), not optional. Its dev batch is also the manipulation check.
  - On "about 45 %": the own pool's mean acceptance is 0.44 (median 0.38), so the README says "44 % on average (median
    38 %)".
- **S6 (style).** Accepted, with the numbers: ours 58-61 words, `s2300` 33, the 25 Nov adapter 34, thinking 6,600-6,900
  against 6,059.
- **S7 (what the files show).** Accepted. Added to §2, and it is the premise of Amendment 6(a).
- **S8 (gap test not decisive).** Accepted. Amendment 6(a) replaces 5(a):
  - Step 1 is the training-only replication with seed 523373504 and a CPU weight comparison. The pre-set thresholds are
    module cosine ≥ 0.99 and norm within 2 % for "match", and cosine < 0.90 or norm off by more than 10 % for "no
    match".
  - Step 2 is the crossed design: (25 Nov pool, our seed) and (our pool, 25 Nov seed). The readings for pool, seed,
    both, combination-only and unresolved are fixed in advance.
  - If a seed effect is found, every single-seed stage-1 contrast whose lower bound is below the seed effect is
    re-labelled "within seed spread".
- **S9 (stale costs and counts).** Accepted. The costs follow Amendment 6, the text says "six amendments", and the
  status line is re-dated.
- **S10 (provenance).** Accepted. The README says:
  - PREREG was first committed at 5:36 am, after batch 1 had been read at 4:59 am;
  - amendment times are self-reported;
  - the code that trained the stage-1 cells is not in git exactly, because the trainer changed at 7:20 am (bookkeeping
    only; the training step is unchanged).
  The coordinator is committing the current state now. The hash goes into the README with the correction commit.
- **S11.** Accepted. The README now says "0 of 768 reference draws; about 0.04 % in Dan's larger E1 jobs (5 of
  11,520)".
- **S12.** Accepted. Listed under "Specific to this run" as an unreported deviation, together with the 2:47 am file
  time of the prompt-control readout.
- **S13.** Accepted. Listed: the prompt control (108 words) is not a test of brevity.
- **S14.** Accepted. Amendment 6(b) uses 16 draws per prompt for own, unsel, final and the tuned own cell at `s1100`.
