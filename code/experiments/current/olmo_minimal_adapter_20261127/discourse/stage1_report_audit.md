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

## Audit of the Amendment 6 sampling step (agent:claude, Opus; 2026-11-27, 12:48 pm PT)

**Verdict: GO WITH FIXES.** Fix T1–T4 before the run, and decide on T5. The science is correct; the driver's failure
handling is not:
- The composition, seeds, draw counts and decoding settings match Amendment 6, and the new cells are comparable with
  corners A and D.
- B and C really are the two missing corners.
- Nothing leaks.
- `dev_check.py` picks the winner by the fixed rule, and its test passes (2 of 2).
- The problem is failure handling: as written, most failures leave an H200 billing until the driver's 4-hour outer
  timeout, which is above the $18 cap.

What I read:
- `v2/logs/drive_a6s2_sample.sh`
- `pod/mini_pod.sh`, `pod/mini_chain.sh`, `pod/mini_setup.sh`
- `stage_cells.py`, `dev_check.py` and `test_dev_check.py`
- PREREG Amendments 6 and 7
- `log.md` and `ledger.md`
- the `cell.json` and `runs.jsonl` of every cell involved, the pools, the pid files and `a6/compare_*.json`

Scripts: `a6_audit.py` and `devsim/` in my scratchpad. I simulated `dev_check.py` on synthetic draws in four tie and
missing-draw cases.

### T1. FIX BEFORE RUN: unbounded waits and unchecked exit codes; the worst case exceeds the cap

The problem:
- Each `mini_pod.sh setup|sample|fetch` call is piped to `tail` without `pipefail`, so its exit code is lost.
- `wait_stage` has no deadline. It needs both `chain.DONE` and the stage's `DONE` or `FAILED` file.
- The driver therefore polls until its outer `timeout 14400` (4 h, about $21 at $5.29/h) in each of these cases:
  - setup fails or exceeds its 30 minutes;
  - a push md5 mismatch;
  - `chain_guard`'s "no setup.done";
  - LOCKED or GPU_BAD;
  - a chain killed without writing `chain.DONE`.
- On the pod, the chain runs under `TO_SAMPLE=21600` (31,200 s in total) and keeps going after the driver dies.

Exact changes:
- Line 2 of the script: `set -u` → `set -uo pipefail`.
- After each of the two `mini_pod.sh setup` and `sample` lines, add:
  `[ "${PIPESTATUS[0]}" = 0 ] || { log "STOP: <step> failed; delete $A"; exit 5; }`
- Cap the pod side:
  - dev call: prefix `TO_GATE=900 TO_SAMPLE=1500` to `timeout 900 bash pod/mini_pod.sh sample "$A" dpo "$DB" ...`.
    That is 25 min for about 15 min of normal work. An overfit candidate that degenerates fails the stage, and with T3
    B and C still run.
  - reach call: prefix `TO_GATE=900 TO_SAMPLE=6000`; with T5, use `TO_SAMPLE=7200`.
- Give `wait_stage` a deadline (second argument), a stall check and a stop:
  ```
  local end=$(( $(date +%s) + $2 ))
  [ "$(date +%s)" -gt "$end" ] && { log "OVERRUN $1"; bash pod/mini_pod.sh stop "$A" dpo; return 2; }
  ```
  Also log STALL when the age of `$ST/heartbeat` exceeds 600 s. Call it as `wait_stage sample_batch_dev_dpo 2700`
  and `wait_stage sample_batch_a6s2 7200` (8400 with T5).
- Outer timeout 11400 s (3 h 10 min) instead of 14400.
- Add `trap 'log "EXIT: delete pod $A once its fetch line is FETCH_OK or abandoned"' EXIT`, so every exit path tells
  the coordinator to delete the pod.

### T2. FIX BEFORE RUN: "POD READY TO DELETE" is printed unconditionally, and the dev fetch is unchecked

- The last line always contains "POD READY TO DELETE", even after `FETCH_MISMATCH`.
- After the dev batch, `dev_check` runs whatever the fetch left behind.
- Change both fetches to:
  ```
  f=$(timeout 900 bash pod/mini_pod.sh fetch "$A" dpo 2>&1 | tail -n 1); log "$f"
  case $f in FETCH_OK*) ;; *) log "STOP: fetch not OK; keep $A until fetched"; exit 6 ;; esac
  ```
- Print `POD READY TO DELETE: $A` only after the second `FETCH_OK`.

### T3. FIX BEFORE RUN: one failed own candidate stops the whole step, B and C included

- If a candidate FAILs the gate, the chain does not sample it. `dev_check.py` then exits with "no draws", `W` is
  empty, the staged count is wrong, and the driver STOPs before B and C, the main point of the step, are sampled.
- Amendment 5's stop rule says a gate-failed cell is "not sampled and reported", not that the batch stops.
- `dev_check.py` also does not check the draw count. In my simulation, a candidate missing one draw scored 50.5 %
  against 50.0 % and won what should have been an exact tie it loses.
- Changes in `dev_check.py`:
  - a candidate with no draws gets `"status": "not sampled"` and is excluded from the choice;
  - require exactly 4 draws on every prompt (128 in all), else `SystemExit`.
- Change in the driver: if `dev_check` fails, log it, set `W=dpo__own__A16__n64__r1p` (no tuned cell) and still
  stage and sample B and C.

### T4. FIX BEFORE RUN: the pre-set readings 3 and 4 overlap

- If only corner A is high (A above B, C and D; B, C and D alike), the pool contrast A − C and the seed contrast A − B
  are both above 0, so reading 3 ("Both") also holds.
- Add to Amendment 6 before sampling: "Readings are checked in the order 1, 2, 4, 3, 5; 'Both' applies only if
  reading 4 does not."
- Also fix the analysis command now:
  - contrasts B − D, A − C, C − D, A − B, plus B − C and A − D for reading 4;
  - D is `dpo__final__A16__n64__r1p` (batch R);
  - A is `dpo__final__OLD_A16__n64` (batch_old).
- A union runs tree must link batch_old, batch_r, batch_a6s2 and the cell.json files from `runs_rp-om4`, because
  `analyze_cells.py` reads cell metadata from the same tree.
- Never pass a contrast containing `lr1e-3` to `analyze_cells --contrasts`: it splits on the first "-".

### T5. FIX BEFORE RUN (recommended, the only change that adds cost; about +$2.5, total about $13.5–15.5 < $18): add our trainer's corner-A replicate

- Corner A is the only corner not trained by our trainer.
- Amendment 7 showed that two same-seed runs of our trainer differ at weight level (total cosine 0.986, a module
  minimum of 0.75) as much as ours differs from the 25 Nov run.
- D was trained on an H200; B and C on an H100.
- No one knows whether that drift moves the reach rate. If it does, the four-corner reading rests on one run of A.
- The step-1 cell is already trained and fetched: `dpo__final25__A16__n64__r1p_s523373504` in `runs_rp-om3`. Its
  dev loss of 0.502 equals B's 0.502 and the rerun's 0.5016.
- Driver changes:
  - add `python3 stage_cells.py --runs "$V2/runs_rp-om3" --ckpt dpo --cells dpo__final25__A16__n64__r1p_s523373504 --stage "$RB"`;
  - `want=3`, or 4 with a tuned cell.
- Reading to fix in PREREG: "A′ − A includes 0: corner A is reproduced by our trainer. Otherwise the step-2 reading is
  also computed with A′ as corner A, and both are reported."

### T6. NOTE: `did_not_train` labels, for the report

Sampling the forced candidates is what 6(c) prescribes.

All three own candidates carry the flag, R-own included. Dev loss:
- R-own: 0.714 → 0.717
- lr 1e-3: 0.714 → 0.811
- 3 epochs: 0.714 → 0.797

Their weights moved:
- max|B|: R-own 0.013 (the final cells' is also 0.013), lr 1e-3 0.047, 3 epochs 0.034
- R-own's ‖ΔW‖ is 27.9, against 28.8 for R-final.

So the flag means "held-out own-sample loss never fell", which Amendment 3 expects. It does not mean "weights
unchanged". The risk is the reverse of a fake success: a reader may dismiss a real null as "the adapter failed to
train".

Fix (can come after the run):
- `dev_check` and the summary table should carry, per candidate:
  - the flag;
  - the dev-loss trajectory;
  - max|B| or ‖ΔW‖;
  - the gate verdict.
- Add that sentence.

### T7. NOTE: resolution of the 6(c) manipulation check

- Simulated with Dan's F_dpo_ifb draws: the paired interval of a no-effect candidate is about ±8 points.
- A true lift of +10 points has a lower bound above 0 in 38 % of runs, +15 points in 85 % and +20 points in 95 %.
- So "none above 0" should be reported as "no lift larger than about 15 points".
- Two own candidates are heavily overfit and may degenerate into long generations (the cap is 32,768 tokens). T1's
  `TO_SAMPLE` cap bounds the dev batch, and with T3, B and C still run.

### T8. NOTE: corner D is 40.6 %, not 43.2 %

- D is R-final, `dpo__final__A16__n64__r1p`: lr 3e-4 × 64, batch R, sampling seed 660003, 40.6 %.
- The coordinator's message quotes 43.2 %. That is the dev-rule cell (lr 3e-5 × 128), a different recipe.
- Amendment 6's table is correct.

### T9. NOTE: checked and correct, no change needed

- **Seeds:** dev 660007, reach 660005.
- **Draw counts:**
  - `stage/pids_dev_dpo_own.txt` is the own pool's 32 test pids in order, each `:4`;
  - `stage2/pids_L.txt` is exactly the 48 reach pids, each `:8`.
- **Decoding:** the same path as stage 1 (`multi_sample.py`, `problems.jsonl`, `--n 0 --chunk 4`, T 1, top-p 1,
  32,768 max tokens, the same template).
- **F_dpo_ifb reference:** the same settings and template, vLLM 0.11.0. `dpo` plain on the dev prompts is 40.82 %
  (512 draws), as quoted.
- **Corner B** = the 25 Nov pool × our seed:
  - pool sha 3cb20b…, whose 64 responses equal the 25 Nov training set;
  - the per-step order is identical to the 25 Nov online order;
  - step-1 nats 3549.687811669217, identical to the 25 Nov run;
  - `init_seed` 1581104982, R-final's;
  - 256,647 tokens.
- **Corner C** = our pool × the 25 Nov seed:
  - pool sha 12fc1d…, R-final's;
  - the same order;
  - the same step-1 nats (410.8133374429914);
  - `init_seed` 523373504.
- **Recipe:** both are fixed-recipe lr 3e-4 × 64, as R-final.
- **Tie rule** (`max` over (rate, −index)): correct in my simulated ties.
- **Leakage:**
  - the dev prompts share nothing with the reach prompts or the own training prompts;
  - no pool's train or test set touches the reach prompts;
  - B's training prompts overlap the dev set in 14 prompts, but B is not in the dev batch.
- **What reaches the pod:**
  - the folder's top-level code and `.md` files, `olmo_edl.py` and `olmo_template.py`;
  - `v2/pools` (34 MB), `stage2/pids_L.txt` and `seeds_L.csv`, and the staged batch;
  - Dan's repository as a sparse checkout of the listed paths only.
  - Nothing from textbook72, test_13_20, test_21_30 or the 832 is in the push manifest. The only match for those names
    is a line in `log.md` saying no sealed set is pushed.
- **Nothing destructive on serrano with a fresh alias:**
  - rsync runs without `--delete`;
  - `stage_cells.py` removes only hard links in the stage directories;
  - `push` removes only `pod_stage/<alias>`.
- **Every command targets `$A` only.**
- **The test passes:** 2 passed. It does not cover the winner rule or `main`; my simulation does.

### T10. NOTE: alias reuse

A reused alias would make the fetch overwrite that pod's earlier logs and state under `v2/runs_<alias>`. Add at the
top: `[ -e "$V2/runs_$A" ] && { log "STOP: $V2/runs_$A exists"; exit 2; }`. Also make sure the alias is a fresh pod,
because `chain_guard` passes on a pod that has no `/workspace/mini`.

### T11. NOTE: the cost estimate is consistent

Stage-1 throughput:
- batch 1 (one final-like and two own-like cells, about 5.2 M tokens): 71 min;
- batch R: 73 min;
- batch_old (one final-like cell, 384 draws): 32 min.

Expected times:

| part | time |
|---|---|
| setup and download | about 15 min |
| dev batch, with gate and engine loads | 20–25 min |
| fetch and `dev_check` | about 5 min |
| reach batch, B and C | about 75–80 min |
| a tuned own cell, if it joins | +15 min |
| T5's replicate | +25 min |

That is 2.0–2.4 h, or $11–13 at $5.29/h. With T5 it is about $13.5–15.5, under the $18 cap. Confirm the new pod's
price and add its ledger line before launch.

### T12. NOTE: provenance

- The driver lives in `v2/logs`, outside git. Commit a copy, with the T1–T5 changes and the fixed analysis command
  (T4), before the run, and record the hash.
- The rest of the code is committed (3ab2b29), and the folder is clean.

### T13. NOTE: small bias in the plain reference

- 23 of the 32 dev prompts have their only strict `dpo` passes in F_dpo_ifb itself. The reference is therefore
  selected on the draws it is computed from.
- By my estimate the inflation of 40.8 % is under half a point, and it biases against "moves". No change.

### T14. NOTE: what 48 × 8 can resolve

- The stage-1 paired contrasts have half-widths of about 8 points.
- A 24-point effect carried by one factor is detected almost surely.
- A 12/12 split gives each contrast about 85 % power.
- Effects under about 8 points read as "includes 0". So "pool" cannot exclude a seed effect of that size, and the
  report should say so.

## Orchestrator's replies to T1–T14 (agent:claude, Opus orchestrator; 2026-11-27, 1:03 pm PT)

The driver now lives in git as `pod/drive_a6s2_sample.sh`, with shared helpers in `pod/drive_lib.sh`. The copies in
`v2/logs/` are replaced by "superseded" stubs that exit. The changes below are in that tracked copy. I tested its
control flow on serrano with stub `ssh`, `mini_pod.sh` and `stage_cells.py`, in five cases:
- normal run: a tuned cell joins, then POD READY;
- setup fails: exit 5 and the EXIT line;
- dev stage FAILED with one candidate unsampled: the winner is chosen among the others, and B, C and A′ still run;
- both fetches mismatch: exit 6 and no POD READY;
- a stalled stage past its deadline: STALL, then OVERRUN, then our chain is stopped.

- **T1, applied.**
  - `set -uo pipefail`; every `mini_pod.sh` step is checked through `PIPESTATUS` (helper `step`).
  - Pod-side caps: TO_GATE 900 s for both batches; TO_SAMPLE 1,500 s (dev) and 7,200 s (reach, the T5 value).
  - `wait_stage <ckpt> <stage> <deadline>`, with deadlines of 2,700 s and 8,400 s:
    - STALL when the heartbeat is older than 600 s;
    - OVERRUN at the deadline, which stops our chain with `mini_pod.sh stop` (never the pod);
    - it returns at once when `chain.DONE` exists without the stage file (NOTRUN), or when the chain pid is gone
      without `chain.DONE` (DEAD, two probes 60 s apart), instead of polling to the deadline.
  - Outer timeout 11,400 s.
  - An EXIT trap (plus `exit 143` on TERM/INT) prints `EXIT rc=<n>: pod <alias> NEEDS A DECISION from the
    coordinator ...` on every exit except a clean finish. The driver never stops or deletes a pod.
- **T2, applied.**
  - Each fetch must read FETCH_OK within two tries.
  - `dev_check` runs only on a fetch that read FETCH_OK. If the dev fetch fails, there is no tuned cell, and
    `dev_check` runs after the final fetch, for the report only.
  - `POD READY TO DELETE` is printed only when the final fetch is FETCH_OK and the reach stage ended DONE. Otherwise
    the EXIT line asks for a decision. The final fetch re-syncs the whole `runs/dpo` tree, so its FETCH_OK also covers
    the dev batch.
- **T3, applied.**
  - `dev_check.py` marks a candidate with no draws "not sampled", and one without exactly `--draws` (4) on every
    prompt "incomplete". Neither is eligible.
  - The winner is chosen on integer pass counts among eligible candidates, with a tie to the earlier-listed one. It
    exits 3 with `winner: null` when no candidate is eligible.
  - The driver then keeps R-own's recipe (no tuned cell) and still samples B, C and A′.
  - Both batches are staged on serrano before the pod is set up. A refused tuned cell is logged, and the batch goes
    ahead with B, C and A′.
  - Tests in `test_dev_check.py`:
    - your case, where a candidate missing one draw (50.5 % against 50.0 %) must not win the tie;
    - an unsampled candidate is skipped and the exact tie goes to the earlier-listed one;
    - no eligible candidate gives no winner;
    - a missing directory reads as empty.
- **T4, applied.** Amendment 8(a) fixes the order 1, 2, 4, 3, 5, and 8(c) fixes the analysis.
  - `a6s2_analyze.sh <alias>` builds `v2/runs_a6s2` (symlinks: batch_old, batch_r, batch_a6s2, gates, every
    `cell.json`, including A′ from `runs_rp-om3`).
  - It runs `analyze_cells.py` with the ten contrasts (your six plus A′−C, A′−B, A′−D, A′−A) and writes to
    `v2/a6/step2_*`, not over the stage-1 tables.
  - `a6_reading.py` applies the readings; its tests pin "only A high" to reading 4, not 3.
  - End-to-end check on stand-in cells in a scratch copy of v2: B as R-final's samples, C as R-own's, A′ as the 25 Nov
    adapter's. The ten contrasts resolve, A′−A = +0.0 [−5.2, +5.2], the reading is "3 both", and nothing was written to
    the real v2.
  - No contrast with `lr1e-3` goes to `--contrasts`.
- **T5, applied (coordinator's decision).**
  - A′ = the step-1 cell from `runs_rp-om3` (unforced) is staged into batch_a6s2; three reach cells, or four with a
    tuned cell.
  - Amendment 8(b) makes the all-ours square (A′ as corner A) the primary reading, and the 25 Nov adapter square the
    secondary.
  - If A′ − A excludes 0, the report:
    - gives that difference as run-to-run variation;
    - splits A − D into A′ − D and "run-to-run, not reproduced" A − A′;
    - uses max(seed X, |A′ − A|) for the stage-1 relabelling.
  - A′ joins the hand check as a fifth corner.
- **T6, applied now rather than after the run.** With `--stage` and `--gate`, `dev_check` carries for each candidate
  `did_not_train`, the dev-loss trajectory, max|B| and the gate verdict. The meaning of `did_not_train` is in
  Amendment 8(d) and in the JSON. Checked on R-own's real `cell.json`: gate PASS, flag true, max|B| 0.013, dev loss
  0.714 → 0.717.
- **T7, applied** in Amendment 8(d) and in `dev_check`'s output: "no lift larger than about 15 points". The
  degenerate-generation risk is bounded by the dev TO_SAMPLE cap, and with T3 it cannot block B and C.
- **T8, applied.** Amendment 8(c) names D as R-final at 40.6 % and the 43.2 % as the dev-rule cell. The report will
  name both wherever both appear.
- **T9:** noted, no change.
- **T10, applied.** The driver stops if `v2/runs_<alias>` exists, or if the pod already has `/workspace/mini`. An
  unreachable pod also stops it, with the probe's answer in the message.
- **T11:** agreed. The new estimate is in Amendment 8(e). I will add the ledger line once the coordinator gives me the
  pod's price.
- **T12, applied.** The tracked copies need the coordinator's commit before the run. The driver logs the folder's
  HEAD and its count of uncommitted paths in its first line.
- **T13:** left as is (it biases against "moves", under half a point).
- **T14, applied** in Amendment 8(d): a "pool" reading does not exclude a seed effect under about 8 points.
- **The 6(b) driver** (`pod/drive_a6b_s1100.sh`) has the same T1, T2, T3 and T10 handling. It can split training on an
  H100 (`MODE=train`) from sampling on an H200 (`MODE=sample`). It is syntax-checked only. It gets the same stub
  simulation before it runs, which is not before step 2 has been read.

## Audit of Amendment 9 (agent:claude, Opus auditor; 2026-11-28, 3:33-3:50 am PT)

A fresh auditor, read-only and CPU only, told to find flaws and not given the orchestrator's conclusions. It read
Amendment 9(a) to 9(i) as written at 3:32 am, the stage-1 files, and the code that existed while it worked (`a9.py`,
`handcheck5.py`, `pod/drive_a9.sh`, `test_a9.py`, all work in progress then). Its report, as returned:

### N1. FIX BEFORE RUN: nothing on the pod stops the two jobs from using the GPU at the same time

- Observed: our chain takes `/workspace/mini/gpu.lock` (`mini_chain.sh`). The weight-interpolation job takes
  `/workspace/wi/gpu.lock` (through `m2_job.sh`). Neither lock sees the other job.
- Observed: the other job checks used GPU memory (below 2,000 MiB) before each GPU job. `drive_a9.sh gpu_idle`
  instead counts `nvidia-smi --query-compute-apps` rows, and only twice.
- Inferred: inside a container that query often lists no processes, so the check can pass while vLLM holds about
  130 GB.
- Fix: check memory.used as `m2_job.sh` does, on the pod side at chain start. Also hold `/workspace/wi/gpu.lock` (or
  one shared lock) for the whole train and sample stages.

### N2. SHOULD FIX: a cut sample stage loses its grading, and round 5 then silently drops the cell

- Observed: `mini_chain.sh` runs `grade.py` only when the sampler exits with rc 0. An overrun stop gives rc 124 or
  143, so no `graded.jsonl` is written.
- Observed: `handcheck5.find_cells` needs `graded.jsonl`, so a cut cell becomes "not sampled". The three-way report
  of 9(h) would then never happen.
- Fix: on a cut, grade the fetched samples on serrano with Dan's `grade.py` at 8e744a89. Make `prep` refuse a staged
  cell that has samples but no graded file.

### N3. SHOULD FIX: the drift check has low power, and 9(d) misreads a non-significant test

- Observed: readers have already shifted once. Of 120 round-3 corner items read again in round 4, 12 moved to gamed
  and 3 the other way, and round 4 was 5 to 8 points lower for A, A′ and C.
- Computed: with 100 items, a shift of that size gives p < 0.05 only 47 % of the time. A significant result needs a
  net shift of 8 to 10 items.
- So "no shift detected" does not mean "no net shift above about 7 items". After a typical 7:5 split of 12 movers,
  the net shift could still be −5 to +8 items, which is −3.6 to +5.6 points on the final model's rate.
- A uniform draw of 100 also holds only about 19 of the final model's items and 11 of D's, the two cells the
  cross-round comparisons use.
- Fix (no pod cost): read all 258 of the final model's items and all 149 of D's again in round 5. The share of
  47.4 % and the second reference then become same-round comparisons. On 407 items the test has about 0.99 power for
  a shift of round 4's size. If this is not done, report the net shift with its interval and give the cross-round
  numbers as a range.

### N4. SHOULD FIX: the files only partly support "the 16-example schedule was fixed before stage-1 data", and the step budget is a confound

- Observed, the step count holds: `DESIGN_DRAFT.md` gives 64 steps for n < 64, and `default_eval_steps(16, 64)`
  returns [4, 8, 16, 32, 64].
- Observed, the lr grid does not: stage 1's only 16-example cell was `dpo__own__A16__n16` (own pool, conditional,
  `lr_from` own n64, `edge_extend: False`), and `DESIGN_DRAFT.md` also says n < 64 inherits the n = 64 lr.
- So the grid plus edge rule is new in Amendment 9. It is the right choice, because it matches the 4-example cell,
  but the text should say it is new.
- Computed: the 16-example cell gets 64 steps (about 223k training tokens). The reference got 128 steps (491k) and
  the 4-example cell 64 (210k). Both existing selections sit at their last step and are unconverged. Add this
  confound to 9(f).

### N5. SHOULD FIX: every existing selection is a near tie between very different (lr, step) pairs

- Observed: 4 examples, 3e-5 at step 64 (0.57069) against 1e-4 at step 16 (0.57157); rank-1, 3e-4 at step 128
  (0.54035) against 1e-3 at step 128 (0.54134); reference, 0.53725 against 0.53911.
- The gaps are 2 to 5 times the same-seed rerun gap of 0.0004 seen in Amendment 7.
- Add to 9(f): each cell's sampled behaviour depends on one pick among near ties.

### N6. SHOULD FIX: the reading rule fires as designed, but two kinds of outcome would be worded misleadingly

- Computed: on round 4's adapter pairs the paired half-widths are 7.3 to 9.5 points, and the auditor's bootstrap
  gives 7.5 to 8.5 for a low cell against the reference. With a half-width of 8.5, "holds" in practice needs an
  estimate above −3.5 (the 9(e) text says about −4), and "falls short" needs an estimate below −10.
- Operating characteristics, with a standard error of about 4.3:

  | true difference | holds | unclear | falls short |
  |---|---|---|---|
  | 0 | 79 % | 20 % | 1 % |
  | −5 | 36 % | 52 % | 12 % |
  | −10 | 7 % | 44 % | 50 % |
  | −15 | 0 % | 12 % | 88 % |

- No reading is unreachable. The wording problems:
  - A cell with an estimate of −9 and an interval of [−17, −1] reads "unclear", although it is clearly below the
    reference. Add "below the reference" whenever the upper end is below 0.
  - "Holds" can be given to a cell up to 12 points (a third of the rate) worse. Write it as "within 12 points".
- Observed: the orchestrator's 60 % "holds" for rank-1 does not fit its own point predictions (31 against 36, a
  difference of −5, which gives about 36 %).

### N7. SHOULD FIX: the time table is a few minutes optimistic, so the money stop may cut draws

- Observed: 3 cells × 384 draws × 8,000 tokens ÷ 1,450 tokens per second is 106 min of sampling alone, but the
  table's "sampling and grading" row says 90 to 105 min.
- Computed from the amendment's own numbers: 148 to 162 min in all ($13.0 to $14.3). That leaves 8 to 22 min before
  the $15 stop.
- Inferred: with `drive_a9.sh`'s deadline, a cut happens if the cells average more than about 8,600 tokens per draw
  at 1,450 tokens per second (about 9,300 at 1,505).

### N8. NOTE: sampling the reference again is not worth it under $15

- Computed: 3.054M tokens ÷ 1,464 to 1,544 tokens per second is 33 to 35 min, $2.9 to $3.1. The job would total 182
  to 197 min ($16.1 to $17.4).
- Computed: a same-engine copy only resolves offsets larger than about ±6 to 7 points (replicate against replicate).
- Observed: the only offset evidence is +1.3 points on the strict checker (64.8 % against 63.5 %).
- Recommendation: do not run it.

### N9. NOTE: the rank-1 cell was trained on a different schedule from SPEC's

- Observed: `plan_dpo_h100b` ran it on the own-source O1 schedule: 128 steps, extended to 512, with the edge lr
  added. SPEC gives "capacity rungs of source (ii)" 64 steps. Under that schedule it would have selected 1e-3 at
  step 64 (0.54159).
- Fix: say which rule 9(a) means.

### N10. NOTE: the coordinator's "at least half" condition was met by other adapters, not by this reference

- Observed: the 34.4 to 38.5 % are the one-pass adapters (A, A′, B, C, D). The dev-selected reference has never been
  read.

### N11. NOTE: provenance and sharing details

- Observed: the reused `cell.json` files lack the `selection` and `edl_*_corr` keys, so an older trainer wrote them.
- Observed: no file records the flags of the h100b and h200b trainings. The drivers that record
  `--ce-checkpoint --max-memory-gb 78` trained other cells.
- Observed: round-4 items and reads (`hc4_read/`, `round4/labels.csv`) are readable next to byte-identical drift
  items. Readers have shell access (a round-3 reader wrote `mk.py`). Low risk, but move `hc4_read` out of the
  readers' reach.
- Observed: the other job's ledger gate counts spend from `WI_POD_START_UTC` against its $21 cap. If this job runs
  first, set `WI_PRIOR_USD`, or the other job will skip its cells.

### The auditor's answers to the seven questions

1. **Is the training rule well defined for 4 and 16 examples?** Yes (observed). The edge rule adds one lr and never
   more; when that lr wins, the cell is unbracketed and the run goes on as normal, which is so in all three existing
   cells. Early stop is two consecutive evaluations above the run's own best. `did_not_train` fires only if no
   evaluation beats step 0; otherwise the lowest loss wins, however small the gain (N5). Only the step count and the
   evaluations predate the data (N4). The reused adapters do come from the rule (plans, `runs.jsonl`), and the saved
   weights are the selected (lr, step): reload check 0.0, and the md5s match the pod, the lab copy and the disk.
   H100 against H200 and the code version: inferred to cause drift of rerun size only.
2. **Is the reference matched?** On the rule, the pool (sha 12fc1d23 on disk), the init seed 1581104982, the H200,
   vLLM 0.11.0 with torch 2.8.0+cu128, and three adapters in the engine (observed). It differs in batch, engine seed
   and companion cells. Re-sampling costs 33 to 35 min and $2.9 to $3.1; recommended against (N8).
3. **Is the reader protocol blind?** Yes: item files hold the prompt, constraints and answer, and the round-5 ids are
   distinct. Style differences are what readers are meant to judge. The drift check is not adequate (N3).
4. **Can the reading rule fire sensibly?** See N6.
5. **What would make "rank-1 is enough" or "4 examples are enough" misleading?** Do not write either; also avoid
   "holds" alone. The reference is itself about 11 points below the final model, so name it. Missing from 9(f): N4,
   N5, N9, N10, cross-round drift (N3), and that the example axis is not token-matched (13,110, 55,747 and 245,487
   training tokens).
6. **Sampling discipline and sharing.** The planning numbers are consistent (7,953 tokens per draw; 1,505 to 1,544
   tokens per second). The timeline is low (N7). `a9.py decide` makes "decide before the engine starts" work; at
   $15, though, the driver's deadline can cut draws, and grading is then lost (N2). The shared `/workspace/models/dpo`
   is safe (Dan's `fetch.sh` uses a lock and the `.complete` marker; `DELETE_WEIGHTS` defaults to 0). `nd-takehome`
   and the pip packages are safe (both set-ups pin the same versions and paths). The GPU is not safe (N1).
7. **Anything else.** No reused seed: 660009 and 660051 to 660059 appear nowhere else. No file is overwritten: there
   are guards on the alias, `batch_a9` and the plan. The prediction table does not reach the readers. The one claim
   the files contradict is in N4.

**Checked and found correct by the auditor:** every number quoted for the reused cells (grids, edge additions,
selections, losses, flags); the reference's 166 of 384 (43.2 %), mean 7,953 tokens, no length cuts; the weight counts;
the label counts in the first 4, 16 and 64 examples and the instruction-type overlap (0, 5, 16); round 4's figures;
the trainer, sampler and gate unchanged since 156bf2b; `a9.decide`'s drop order; the item-id collision check.

## Orchestrator's replies to N1–N11 (agent:claude, Opus orchestrator; 2026-11-28, 4:00 am PT)

Each finding was checked against the files before it was accepted. The changes to the pre-registration are in PREREG
Amendment 9(j), written before the plan file was built.

- **N1, applied.** Checked: `m2_job.sh` takes `$M2_ROOT/gpu.lock` with `flock -n` and refuses a GPU with 2,000 MiB or
  more in use; our chain did neither for the other job.
  - `drive_a9.sh gpu_idle` now reads `memory.used` (limit 2,000 MiB, up to two minutes of waiting) before anything is
    pushed, before training and before sampling.
  - `mini_chain.sh` has two opt-in guards, off for every earlier driver: `GPU_FREE_MIB` (refuse to start on a GPU in
    use) and `ALSO_LOCK` (hold the other job's lock file too while the chain runs, if the file exists; opened for
    reading, never created or written). `mini_pod.sh` passes them and reports a refused chain as NOT STARTED.
    `drive_a9.sh` sets both (`A9_OTHER_LOCK=` turns the second off).
  - Tests: `test_chain_shared_pod_guards`, `test_chain_holds_the_other_jobs_lock_while_it_runs`, and three driver
    cases in `test_a9.py`.
  - Not done: one shared lock file for both jobs. That needs a change in the other job's folder, which this
    orchestrator may not touch. The coordinator still has to run the two jobs one after the other.
- **N2, applied.** Checked in `mini_chain.sh`: grading runs only after the sampler returns 0.
  - `handcheck5.py find_cells` stops with a message when a cell has draws and no grades.
  - `handcheck5.py salvage` writes the finished requests of a cut batch to a new directory; RUNBOOK_A9.md, "If
    sampling was cut", has the local grading command (it reproduces the stored reference's 384 rows and 166 passes).
- **N3, applied.** Checked: 120 common items, 12 moved to gamed and 3 to genuine, exact sign test p = 0.035; the
  power of the 100-item test for that shift is 0.48 by simulation, and 0.99 on 407 items.
  - Round 5 reads every round-4 item of the final model (258) and of the one-pass adapter (149) again. The share of
    the final model's rate and the second reference use round 5's labels on both sides; round 4's are given beside.
  - The 100 uniform items stay and are drawn first, so the pre-registered test is unchanged. Every drift result now
    carries the net shift with a 95 % interval.
  - The wrong sentence in 9(d) is corrected in 9(j).
  - Cost: about 380 more items, roughly 95 more reader calls. `prep --no-reread` gives the first design back.
- **N4, applied.** Checked in `plan_dpo.json`, `plan_dpo_h100.json` and `DESIGN_DRAFT.md`. 9(j) says that the grid
  with the edge rule for 16 examples is this amendment's choice, and adds the step and token budgets to the limits
  (13,110 × 16, 55,747 × 4 and 245,487 × 2 tokens; recomputed from the pool).
- **N5, applied.** Checked in the three `runs.jsonl` files; the numbers are as the auditor gives them. In 9(j).
- **N6, applied in the wording; the rule and its thresholds are not changed** (the coordinator fixed them).
  - "Holds" is written "holds (within 12 points of the reference)".
  - `handcheck5.py` reports `interval_below_zero` beside every reading, and the sentence then says "below the
    reference".
  - The table of operating characteristics is in 9(j) (recomputed: it follows from a normal estimate with standard
    error 4.3 and half-width 8.5).
  - My reading probabilities are revised in 9(j), before any data, with the first version left in 9(g).
- **N7, applied.** The time table is replaced in 9(j), with the 27 Nov timings as its basis: 1 h 55 min to 2 h 40 min,
  $10.0 to $14.3. The plan now counts 15 minutes of overhead per batch (was 12), so the decision before the engine
  starts leaves room for the last fetch. The stop stays $15. The driver also stops waiting for training at minute 46, so a slow training cannot use up the time of the two
  reused cells. In the slow case the margin is about 8 minutes, and a cut is possible if the three cells average more than about 8,200 to 9,300 tokens per draw.
- **N8, agreed; left to the coordinator as instructed.** The fourth cell stays optional (`REF2=1`, with the stop
  raised to about $18.5). Both the auditor and this orchestrator recommend not running it.
- **N9, applied** as a correction in 9(j): which schedule the rank-1 adapter had, and what SPEC's 64 steps would have
  selected. The adapter is reused as trained.
- **N10, applied** as a correction in 9(j).
- **N11.**
  - Older `cell.json` format and unrecorded trainer flags: added to the limits. `a9.py verify` accepts a `cell.json`
    without the `selection` key and refuses one that names another selection rule.
  - **Rejected: moving `hc4_read/` away.** It is part of an earlier round, which is not to be moved or changed, and
    moving it would not close the gap: `handcheck/round4/` holds the same reads and the key, and `handcheck/round5/`
    holds round 5's key. A reader with a shell could open any of them, in round 4 as now. The protection is the
    reader's instruction (its batch file and its output file, nothing else), as in round 4. Stated as a limit in 9(j).
  - The other job's spend ledger (`WI_PRIOR_USD`): not this experiment's code. Passed to the coordinator in the
    report.
- **Not taken up from the answers:** nothing; the answers repeat N1 to N11.
