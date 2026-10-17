# What the other mentees found about improving RL, compared with our results

Run `peer-rl-review` (executor, 2026-10-17). **Unreviewed.**

Every number carries its source. New numbers marked † are recomputed from stored per-attempt files of our runs (no new
sampling). They come from `ours_on_leon_passk.py` → `out/ours_on_leon_passk.txt`, plus `out/schema_overlap.txt`,
`out/lean_reject_share.txt` and `out/charles149_premises.txt`, all under `peer_rl_review/`. Reader notes with
line-level citations and an adversarial claim check (`notes/report_check.md`) are in `peer_rl_review/notes/`.

Source abbreviations (nd-rl branches):
- **L** = `origin/leon/main:code/experiments/current/thinking_traces_20261005/`
- **RE** = `origin/robbie-experiments`
- **DM** = `origin/dmitry/main`
- **CH** = `origin/charles/main`
- **ND** = `origin/dan`

Our models:
- **r8 / r16** = `la_T1_best12_s{0,1,2}_r{8,16}` (`state_env_october(_r16)`): 9,560,832-param ALiBiGPTs trained from
  scratch, proof state (`lean_staten`). Stage 1 is K12 (155,000 proofs, 2–12 lines). RL is the T1 expert-iteration
  ladder: 8 or 16 rounds × k 32 on 4,495 targets.
- **frozen** = `stage1_best12_s{0,1,2}_b1200`.

Leon's models have the same architecture (≈ 9.6M, from scratch) plus a think block before each step. **Lean alone**
judges unless marked.

## Summary for Dan

Leon's proof-state system is far ahead of ours. We scored our stored samples on his read-out protocol: the same
textbook72 files and Charles149 names, T 0.8, step caps, estimator and Lean judge code.
- **Charles149** (clean for both sides): pass@1 70.5 % for his recipe against 31.8 % for our r8.
- **textbook72**: 74.0 % against 42.9 % for our r16. His pass@5 (82.3 %) beats our plain coverage at 256 attempts
  (71–79 %).
- His no-think K12 base reads 17.2 % and ours 15.2 %, so the pipelines agree.

**Most of the gap is Stage-1 data, not the RL algorithm.** Stage 1 on proofs that a search finds in the environment,
instead of K12's generator proofs, adds 14 points before RL and about 21 after the same EI on the same pool (single
seeds). His RL pool adds up to 9 (partly designed on textbook72), thinking 4–12 per attempt at about 3× the tokens,
GRPO about 5 over EI.

None of Leon's or Robbie's RL numbers is reviewed. Robbie (whole proof) trails us for measured reasons. Dmitry's
selection leaves the prover unchanged. Charles's premise-count wall applies to our models too.

**Top lesson: try search-proof Stage-1.**

## 1. Per person

### 1.1 Leon (2026-10-05 → 10-16): thinking traces, search-proof Stage-1, GRPO v2

**What he tried:**
- **A think block before each action**, at budgets 0 / 64 / 256.
- **Stage-1 data.** First programmatic traces over K12's reference proofs. Then the proofs of an "honest search"
  (`searchtrace.py`), which builds proofs through the environment and never reads the reference proof: T0S (no
  thinking) and S1g3 (search-trace thinking).
- **RL.** EI (Dan's ladder) and GRPO on `rl_hard12` (5,000 targets, 7–12 lines) and on `hard712` (designed from a gap
  analysis of textbook72 dev58).
- **GRPO v2:**
  - a length curriculum, L2 (10–12 lines) → L3 (13–15);
  - verifier "stop" rollouts: the first step rejected by the environment or `step_check` ends the attempt, with a 0.1
    penalty;
  - canary and rollback;
  - an unlikeliness reward;
  - nine single-seed knob pilots.

**What worked:**
- **Search-proof Stage-1, before RL.** T0S − T0 on textbook72 pass@1 is +14.4 [+8.0, +21.4], and +26.0 on hard12
  (L/OVERNIGHT_20261008.md:L169–176). His mechanism: Lean rejections of environment-accepted textbook72 attempts fall
  from 55.4 % to 17.8 % (L179–181).
- **Search-proof Stage-1, after the same RL.** EI on `rl_hard12`, no thinking, reads 52.4 % (r1) and 53.6 % (r2) on
  textbook72 from T0S, against 30.8 % and 32.4 % from T0. These are single seeds (OVERNIGHT L234, L271;
  L/matched_eval/matched_tb72.md:L14, L21).
- **The RL pool.** hard712 against the old pool, from T0S: textbook72 +9.3 [+3.8, +15.2] without thinking, +7.2 with
  it. dev58 informed hard712's design (OVERNIGHT L265–277).
- **Search-trace thinking.** S1g3 − T0S is +10.3 [+3.7, +17.2] per attempt (L177). At matched sampled tokens on 13–20,
  the no-think model is ahead or tied at ≤ 2k tokens. At 16k the three seeds give −7.1 / +8.2 / +8.4
  (L/THINKING_UPLIFT.md:L309–315).
- **The primary recipe** `grpo_think_vs_mb_kkl0` (L/RECIPE.md:L35–39): pass@1 / pass@5 over 3 seeds, plain, T 0.8,
  budget 64.

  | set | k | pass@1 / pass@5 |
  |---|---|---|
  | 13–20 | 16 | 32.1 / 57.8 |
  | 21–30 | 8 | 7.2 / 19.4 |
  | textbook72 | 64 | 74.0 / 82.3 |
  | Charles149 | 64 | 70.5 / 78.9 |

  Charles149 is Charles's 173-problem release minus 24 that overlap his training sets.
- **GRPO vs EI at matched attempts**, 3 seeds (L/RECIPE.md:L277):

  | set | pass@1 | pass@5 |
  |---|---|---|
  | textbook72 | +5.6 [+0.8, +10.9] | +4.4 |
  | Charles149 | +8.4 [+3.6, +13.1] | +5.1 |
  | 13–20 | +0.6 | −0.1 |

  Where the lead shrinks:
  - At matched tokens, Charles149 falls from +7.3 (1k) to +1.7 [−3.2, +7.2] (8k) (L291–295).
  - Under guided reads it is +3.7 [−0.6, +8.3] (L/matched_eval/v2_results_ynez.md:L563).
  - EI had about 4 tuning pilots, against about 15 for GRPO (L/RECIPE.md:L317–322).

**What did not work:**
- **GRPO with verifier *redraws* during training.** 13–20 pass@1 is 14.9 % against 29.3 % for plain GRPO
  (L/matched_eval/v2_results.md:L12–14).
- **Pilots that lost** (v2_results_ynez.md §7–8, L1116–1274): in-context verifier feedback, an unseen-pair prior,
  64 × 64 groups, training at T 1.2, KL 0.1 and half the learning rate.
- **Phase-1 GRPO collapses.** There were three separate causes, each addressed in GRPO v2 (L/GRPO_COLLAPSE_AUDIT.md:
  L23–40, L87–107; L/collapse_audit/README.md:L74–88):
  - a mismatch between the sampler's mask and the loss;
  - overlong masking that taught looping;
  - a long-context attractor.

**Reviewed?** No: the documents are agent-written, and no review exists.

How solid the small effects are:
- A rerun with nothing changed moved Charles149 pass@5 by +3.7 [+1.2, +6.4] (v2_results_ynez.md:L883). So the
  single-seed knob gains (+2–4) are within noise on Charles149.
- On 13–20 three components beat that control: +2.1 to +3.9 against +1.3 (L827–877).
- Charles149 was a selection set for KL 0 (RECIPE L306–310).

### 1.2 Robbie (2026-09-28 → 10-14)

**What he tried.** His models are whole-proof `lean_seq`, 9.56M, pretrained 300 s on cap-6 data. The work covers:
- a combined pipeline: his pretraining, Leon's GEAR EI and verifier-guided RL sampling;
- test-time scaling: resampling invalid lines, SMC, beam, best-first and a value head;
- TTRL;
- retry-EI with Lean error tokens;
- 3-hour GRPO vs EI runs;
- model diffing of the "GRPO wall";
- a temperature sweep on our r16;
- agent-driven architecture loops.

The judge is Lean ∧ `nd_verify`, except Lean alone for his proof-state runs and the sweep.

**What worked:**
- **The combined pipeline.** dev1108 904.3 (915 / 893 / 905) for c005
  (RE:experiment-summaries/2026-09-28-combined-pipeline/README.md:L31).
- **Resampling invalid lines at test time.** textbook72 pass@1 17.6 → 26.0 %; 30 → 36 of 72 solved at k 256
  (RE:…/2026-09-28-combined-model/README.md:L191–197).
- **Proof state over whole proof.** His port wins on all four recipes, with dev medians +111 to +178 under Lean alone
  (RE:code/experiments/current/state_recipes_20260930/README.md:L103–109).

**What did not work:**
- **Retry-EI.** Every arm is below plain EI (iid 771; resume 762, retry 712, repair 706). Retries are accepted no more
  often than fresh samples (RE:…/2026-10-08-retry-ei/README.md:L63–72).
- **TTRL.** At 5× compute, sample-more scores 1,020, TTRL 1,079 and targets-only EI 1,088. TTRL and targets-only EI
  are transductive: they train on the dev statements (combined-model README L327–334).
- **GRPO.** In unpublished 3-hour runs, EI scores 852–929 on dev1108 at k 16 and GRPO 272–593. These numbers exist
  only in Hugging Face run summaries (`robbiethompson2018/nd-rl-checkpoints`, `long-rl-20261014-*`;
  `notes/robbie.md`).
- **Model diffing.** Pretraining on ≤ 6-line proofs teaches a box-closing shortcut that fails at depth 3. One GRPO seed
  broke through; another never did (RE:…/2026-10-14-grpo-wall-model-diff/README.md:L12, L27–35).

**Reviewed?** No.

### 1.3 Dmitry (2026-10-06)

Dmitry ran two agent sprints on our r16 s0. The first is test-time only. The second, a Codex sprint, also trained the
prover.

Picking one of 8 proofs without Lean, on textbook72 (DM:experiment-summaries/2026-10-06-proof-selection-sprint/
summary.md:L122–128):

| method | accuracy |
|---|---|
| random | 0.405 |
| own likelihood | 0.540 |
| sibling likelihood | 0.567 |
| trained judge | 0.582 |
| oracle | 0.594 |

With Lean, sampling to the first valid proof reaches the pool's ceiling of 0.653, using up to 64 attempts. The best
learned stopping rule reaches 0.630 at 9.1 attempts (L247–252, averaged over three generators).

The Codex sprint's REINFORCE raised in-distribution first-proof accuracy by 12.6 points over its format-only start. On
textbook72 the trained provers' first proof scores 41.44 %, against 41.67 % at the start
(DM:…/2026-10-06-learned-compute-allocation/summary.md:L105, L133).

**Reviewed?** Only by reviewers the sprints commissioned themselves.

### 1.4 Charles (data, 2026-09-30 → 10-12)

**Releases:**
- **Textbook release (2026-10-02).** 412 problems. candidate_v0 + v1 + batch3 form the 173 that Leon cleans to 149.
- **Candidate pretraining pools (2026-10-09).** None has been trained on.
- **The premise-count wall** (`textbook72_distract2`), measured on our `la_T1_best12_s0_r8`. Settings: k 32, T 1.0,
  48 steps, `nd_verify`, one seed.

**What the wall looks like:**
- **Distracted textbook72.** After two unused premises are added, 25 textbook72 problems cross to ≥ 4 premises. On
  those, solved problems go 17 → 0 (CH:data/pretrain_data/release_2026-10-12_textbook72_distract2/README.md:L28).
- **Generator problems.** 0 of 120 four-premise problems are solved (L266–268).
- **Hand-built problems.** 0 of 320 samples on 21 hand-built elementary problems, even `P , Q , R , S ⊢ P`
  (L200–210).

**Why it happens:**
- **Cause.** The premise-listing block. When the environment writes it, the same r8 solves 118 of 120 four-premise
  generator rows (L266–268).
- **Root cause.** K12 and every earlier pool cap premises at 3.
- **No RL attempted.** The teacher-forced probability of the correct block is about 4·10⁻²⁴
  (CH:experiments/pretrain-data/textbook72_distract2/results.md:L116).

**What it costs us:**
- textbook72 has one 4-premise problem.
- Charles149 has five. Our r8 accepts 0 of 256 attempts on each, in all three seeds, plain and guided
  (`out/charles149_premises.txt`).

## 2. Comparison with our results

Verdict codes:
- **AA** — apples to apples: same theorems, judge code, T, step caps, plain/guided and estimator; only the systems
  differ.
- **SP** — same set, different protocol.
- **NC** — not comparable.

| # | Their result (source) | Our closest number | Protocol difference | Verdict |
|---|---|---|---|---|
| 1 | Leon recipe, textbook72 pass@1 / @5: **74.0 / 82.3** (L/RECIPE.md:L38) | r16 **42.9 / 57.6**†; r8 39.2 / 53.1–53.6† (two seed sets) | Same read-out. His model also thinks ≤ 128 tokens per step, outside the caps. Set exposure on both sides (§3.1 item 6) | AA |
| 2 | Leon recipe, Charles149: **70.5 / 78.9** (L39) | r8 **31.8 / 48.5**† | Same. A KL-0 selection set for him. 5 four-premise problems we score 0 on | **AA (cleanest row)** |
| 3 | Leon recipe, guided pass@1: textbook72 80.3, Charles149 77.4 (derived means; v2_results_ynez.md:L212–220) | r16 guided 50.7†; r8 guided (Charles149) 43.4† | Same vendored guided sampler, ≤ 10 redraws | AA |
| 4 | Leon T0 (K12, no think, Dan's recipe), textbook72 pass@1: 17.2 (3 seeds; L/stage1_stats/tb72.md:L11) | frozen 15.2 [10.9, 19.7]† | — | AA; agrees |
| 5 | Leon T0 → EI on `rl_hard12`: r1 / r2 30.8 / 32.4 on textbook72; r2 24.8 on Charles149 (matched_tb72.md:L14, L21; confusion_charles149.md:L8) | r8 39.2 / 31.8† (8 rounds) | 1–2 rounds vs 8; 1 seed | AA; similar level |
| 6 | Leon S1g3 Stage-1 (no RL), textbook72 pass@1: 43.8 / 39.6 / 32.3 (L/RECIPE.md:L70–72) | r16 40.9 / 43.6 / 44.2† | Before RL vs after 16 rounds | AA |
| 7 | Leon T0S → EI, no think: r1 / r2 on `rl_hard12` 52.4 / 53.6; r2 on hard712 62.8 (OVERNIGHT L234, L271; tts_uplift_tables.md:L22) | r8 39.2† at 211–235 tokens per attempt (`out/charles149_premises.txt`); his 179 | One seed each. hard712 was designed from dev58, and its `pool_s5k_a` holds curried twins of 3 dev58 problems (L/curriculum/README.md:L218–219) | AA, with that caveat |
| 8 | Leon S1g3 → EI r2 s1, textbook72 pass@64: 90.3 % (65 of 72) (tts_uplift_tables.md:L24) | r16 expected solved at 64: 48.4 / 50.8 / 54.7 plain, 52.7 / 53.7 / 58.5 guided† | Single seed; thinking arm; same exposure caveat as row 7 | AA |
| 9 | Leon GRPO − EI (3 seeds), textbook72 pass@1: +5.6 [+0.8, +10.9] (L/RECIPE.md:L277) | Ours (`grpo-best`, matched attempts), pass@1: default +1.8 [−3.3, +6.8], unlikely +2.8 [−2.0, +7.3], pass@4 −1.8. At pass@5 all lower, none resolved† | Different bases and GRPO recipes | AA read-out |
| 10 | Robbie c005, dev1108: 904.3 (915 / 893 / 905) (RE combined-pipeline L31) | r8 1,058 / 1,050 / 1,055 (fork `numbers.md` L1542) | Judge Lean ∧ `nd_verify`, whole proof, cap 6. Same set, k 64, T 0.8 | SP |
| 11 | Robbie c005, textbook72 at k 256: 30 / 72 plain, 36 / 72 resampled (RE combined-model L191–197) | r8 52 / 51 / 52 plain (best-state seeds); 54 / 52 / 57 guided (trajectory seeds) | Judge differs | SP |
| 12 | Robbie, proof state − whole proof: +111 to +178 dev, Lean alone (RE state_recipes L103–109) | Atlas re-score: state ahead in 8 of 8 dev / holdout cells (ND evidence-atlas README) | Different models | Same direction |
| 13 | Robbie's sweep on our r16 s0, textbook72 pass@1: 0.407 at T 0.8 (RE temp_sweep_20261006/results_table.md:L23) | r16 s0 40.9† | Sample seed | AA; agrees |
| 14 | Dmitry, r16 s0, random pick of 8, textbook72: 0.405 (DM selection summary L123) | r16 s0 pass@1 40.9† | — | AA; agrees |
| 15 | Codex REINFORCE on r16, textbook72 first proof: 41.44 vs 41.67 % at start (DM allocation summary L133) | r8 → r16 (8 more EI rounds): +3.7 pass@1† | 256-token / 48-action caps | NC; same direction |
| 16 | Charles, our r8 on distracted textbook72, crossing problems: 17 → 0 of 25 (CH distract2 README L28) | — | `nd_verify`, k 32, T 1.0, 48 steps | Measured on our model |

Rows 4, 13 and 14 put the same model kind through both pipelines and agree within noise. Row 5 reaches a similar level
after fewer rounds. The large gaps in rows 1–3 are therefore differences between the models.

## 3. Why the numbers differ

### 3.1 Leon vs us (the one material difference)

textbook72 pass@1 (Charles149 in brackets), plain, T 0.8, Lean alone:

| Stage | Leon's pipeline | Ours |
|---|---|---|
| K12 Stage-1, no think | T0 17.2 | frozen 15.2 |
| + EI on generator targets (his `rl_hard12`: 7–12 lines; ours: 4,495, 7–14 lines, incl. 760 textbook-schema) | T0 → EI r1 / r2: 30.8 / 32.4 (24.8) | r8 39.2 (31.8); r16 42.9 |
| Search-proof Stage-1, no think | T0S 31.6 | — |
| Search-proof base + the same EI on `rl_hard12` | T0S → EI r1 / r2: 52.4 / 53.6 | — |
| + hard712 pool (designed from dev58) | T0S → EI r2: 62.8 | — |
| v2 curriculum, no think, s0 | EI 63.5 (54.4); GRPO 68.1 (59.6) (confusion files L16) | — |
| + thinking + GRPO recipe | 74.0 (70.5); EI with thinking 68.4 (62.2) (v2_results_ynez.md:L24, L30) | — |

The candidate causes, ranked:

1. **Stage-1 proof style.** This is the largest cause. It is *established* within Leon's pipeline, which is
   unreviewed.
   - Before RL it adds +14.4 on textbook72, from the same K12 theorems with only the proofs changed (T0 has 3 seeds,
     T0S 2).
   - After the same EI on the same pool it adds +21.6 at r1 and +21.2 at r2 (single seeds).
   - Mechanism. Lean rejects 48–56 % of our frozen model's textbook72 attempts after the environment accepted them;
     his T0 shows 55.4 %. For our r8 the figure is 11–17 % (`out/lean_reject_share.txt`), but after our RL 44–49 % of
     attempts end on an environment-rejected step instead. Whether search-proof data removes those is not measured.
2. **The RL pool.**
   - hard712 against the old pool, from the same base, adds +9.3 [+3.8, +15.2] on textbook72. That is *partly by
     design*: dev58 shaped hard712.
   - The later curriculum against older problems moves textbook72 by +0.1 (GRPO) and +2.0 (EI), and Charles149 by +2.7
     and +4.5 (L/matched_eval/v2_results.md:L548–558).
   - So the pool is worth up to about 9 points on textbook72 and a few on Charles149.
3. **Thinking.** This is *measured but weakly*: one seed per arm, intervals over problems only.
   - After RL: +4.2 [−3.2, +11.2] / +11.7 / +8.6 per attempt on textbook72, at 2.7–2.9× the tokens.
   - At 1k matched tokens: −0.9 / +5.4 / +1.5 (tts_uplift_tables.md:L48, L51, L52).
   - It is a minor share of the gap.
4. **GRPO over EI.**
   - Within his pipeline: +5.6 / +8.4, shrinking with tokens, against a less-tuned EI.
   - Ours: default and unlikely point the same way at pass@1 (+1.8 / +2.8, unresolved); pass@4 points the other way
     (−1.8); all three are lower at pass@5, unresolved (row 9).
   - Overall it is about 5 of the 31–39 points.
5. **Ruled out** (audit `notes/leon_eval_audit.md` and our recomputation):
   - **The judge.** 19 vendored files are byte-identical, and Lean checks the environment-assembled text against the
     original statement.
   - **The evaluation files.** They are byte-identical, and the 24 removed names match.
   - **Sampling settings and estimator.**
   - **Seed noise.** The gap is 31–39 points against interval half-widths of 5–10.
   - **Compute per attempt.** His no-think arms use 153–181 tokens per attempt against our 211–235.
   - **RL attempts.** He used 320k per seed; our r8 used 1.15M (8 × 32 × 4,495).
6. **Exposure and selection** (partial).
   - **His side.** His generator was designed from dev58, which he calls "no longer a clean held-out set"
     (L/gap_analysis/README.md:L8–12). His EI-lineage round-2 pool holds curried twins of 3 dev58 problems and of
     10 other release or textbook72 problems (curriculum/README.md:L218–219).
   - **Our side.** Our ladder's targets contain 400 substitution instances (40 each) of 10 textbook72 theorems.
     Without those, our textbook72 pass@1 drops by about 4 points (r16: 42.9 → 38.5†).
   - **Charles149.** Our targets hold no instance of a Charles173 theorem. His recipe's pools were checked against
     Charles173 by renaming and curried class (no hits), though not by substitution instance. Charles149 still shows
     the largest gap, and at most 3.4 points of it sit in its 5 four-premise problems.

### 3.2 Robbie vs us

The gap is dev 904 against 1,054, and textbook72 30 against 52. Its causes are already measured:
- **Interface.** *Established* by his own controlled comparison, +111 to +178. Our atlas re-score agrees.
- **Pretraining cap.** *Established* on our side: cap 6 gives a T1 textbook72 IQM of 37.3 of 72, against 51.7 at
  cap 12, in the state format (ND best-state README).
- **Judge.** Its effect is small, by *hypothesis*. The Lean-only class (accepted by Lean, rejected by `nd_verify`) was
  909 distinct proofs, beside 281,817 that both accepted (ND lean-judge README L87–100).
- **Compute.** It is not matched: 13 / 11 / 13 RL rounds inside a 2,192–2,298 s pipeline (RE combined-pipeline L31,
  L53–55), against our 8-round ladder.

### 3.3 GRPO vs EI disagrees across people

| Who | Interface | GRPO vs EI |
|---|---|---|
| Leon | proof state | GRPO ≥ EI at pass@1 |
| Robbie | whole proof | GRPO ≪ EI (unpublished runs) |
| Us | proof state | default and unlikely ≈ EI at pass@1; lower at pass@5 (unresolved) and on solved theorems (`grpo-best`: p 0.002 / 0.0002); pass@4 level overall |

The *hypothesis* that fits all three:
- GRPO sharpens and EI covers.
- GRPO needs Leon's safeguards. Robbie's model diff shows GRPO stuck behind the depth-3 shortcut.
- Step-level verifier signal in rollouts helps (stop), while redraws during training break it.

### 3.4 Dmitry and Charles

- **Dmitry.** His learned selectors approach the 8-sample oracle (0.582 against 0.594). Sampling to the first
  Lean-valid proof sets a higher ceiling (0.653, up to 64 attempts). His REINFORCE moved in-distribution accuracy but
  not textbook72.
- **Charles.** Our models share the premise-count wall (row 16; 0 of 5 four-premise Charles149 problems), and it is
  caused by the data:
  - the correlation (no ≥ 4-premise prompts anywhere in training);
  - the environment-written-block counterfactual (118 of 120).

  RL cannot fix it while reward is zero.

## 4. Lessons, ranked by expected value

**What we could adopt:**
1. **Search-proof Stage-1 (Leon's T0S).** Regenerate K12's proofs with `searchtrace.py` (deterministic, CPU), retrain
   our Stage-1, and run the T1 ladder unchanged. This is a two-arm test of cause 1, at 5 seeds against our MDDs. It is
   the largest effect in this review, and cheap.
2. **Read pass@1 / pass@5 and matched-token curves beside pass@256.** Our pass@256 view hid that r16 has pass@1 ≈ 43 %
   where a peer system has 74 %.
3. **Close the premise-count wall.** Either add 4–5-premise prompts to Stage-1 (Charles's pools exist), or let the
   environment write the premise block (0 → 118 of 120 on our r8).
4. **If we run GRPO, use Leon's working parts:** stop rollouts, canary and rollback, and `--overlong fail`. Never use
   redraws in training.
5. **Thinking is low priority.** It costs about 3× the tokens and is seed-dependent at matched tokens.

**What they could adopt from us:**
1. **Independent review and pre-registration of headline claims.** No RL number of Leon's or Robbie's is reviewed.
2. **Creation tests on Leon's recipe** (capability-defs / create-or-elicit protocols). Does the 74 % include theorems
   his base cannot solve at large k, or is it sharpening? He reports mostly pass@1 / pass@5.
3. **Seeds against a noise floor.** S1g3's textbook72 pass@1 spans 32–44 % over three seeds.
4. **A clean headline set.** dev58 is spent on design.
5. **Robbie: the proof state and cap-12 data for the main line.** His own result favours it.
6. **Per-arm compute records.** Leon matches attempts and tokens; Robbie budgets wall clock. Guided read-out is already
   adopted: Leon uses our sampler.

## 5. Common-protocol results and priors

**Ours on Leon's read-out protocol** (rows 1–9, §3.1). The full table is in `out/ours_on_leon_passk.txt`: frozen, both
r8 seed sets, r16, guided reads, `tb72_clean62`, and paired GRPO − EI.

**Theirs on ours** was impossible for Leon, whose checkpoints are on his machine (question in `QUESTIONS.md`).

**No pods were used:**
- Robbie's published checkpoints differ from ours in interface and cap, and both effects are measured.
- A Lean-alone re-judge would move his counts only by the small Lean-only class.
- Charles had already read our r8 on his set.

**Priors** (pre-registered 20:15 UTC, before reading):
- **P1 — most rows not apples to apples as first reported: supported.** 15 of 16 rows differ:
  - rows 1–9 and 13–15 in k (their pass@1 / pass@5 / k 8 against our pass@256 counts);
  - rows 10, 11 and 16 in judge;
  - row 12 does not differ.
- **P2 — GRPO − EI at large k under half its pass@1 gap: mixed.**
  - At pass@5 it holds 79 % (textbook72) and 61 % (Charles149) of the pass@1 gap.
  - At k 64 (his confusion tables, v2 pairs, s0) it falls to about 0 on Charles149 (+1 and −1 of 149, against +5.0)
    but not on textbook72 (+2 and +3 of 72, against +4.7 and +7.7) (confusion files L13, L15).
- **P3 — a Leon checkpoint within our MDD at k 256: not testable.** On reported numbers it would fail: his GRPO arms
  solve 62 and 66 of 72 at k 64 (confusion_tb72.md:L13, L15).
- **P4 — retry and selection help low k more than coverage: mixed.**
  - Retry-EI helped at no k.
  - Selection improves single picks, and leaves coverage unchanged by construction.
- **P5 — premise wall applies to us: supported.** Charles's r8 measurement gives 0 of 25; on Charles149 we score 0 of 5.
