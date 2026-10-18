---
written_on: 2026-10-17
written_by: agent:claude
---

# peer-rl-review: on Leon's own read-out protocol his proof-state recipe leads ours by 31–39 points of pass@1 (textbook72 74.0 vs 42.9, Charles149 70.5 vs 31.8); his EI baseline leads by about as much, so the gap is not GRPO against EI; in his unreviewed ablations the largest single step is Stage-1 proof style

Status: completed and reviewed 2026-10-17. **No hard-constraint violation, nothing quarantined.** This was a reading
run. The executor reviewed the other mentees' RL work (Leon, Robbie, Dmitry, Charles) against ours, and re-scored our
stored per-attempt reads on CPU: no pod, no sampling, no training, $0.

The reviewer first did a blind recount, with six read-only reader agents on the mentees' branches plus their own code
in `review_prr/`, and then checked 62 claims.
- **The central result stands.** The reviewer re-derived it exactly on the stored files.
- Every † point value in `numbers.md` and the report reproduces to one decimal. The bootstrap intervals beside them are
  too narrow.
- Lean accepted **6,078 / 6,078** counted proofs behind the re-scored reads, and the must-fail controls failed.
- **Not supported as written:** P1's "supported"; Dmitry's oracle "using up to 64 attempts"; row 15's "same direction".
- The other rewordings are applied in the nd-rl reading copy and listed below.

Question (Dan, 2026-10-17): "Some of the other mentees have recently done work on improving the RL performance. Can
you … review those results and summarize them for me? In particular, compare the performance to the results we've
already gotten ourselves, and give thoughts on how those differences came to be." Brief:
[`BRIEF_peer-rl-review.md`](../../docs/proposals/claude-heavy/BRIEF_peer-rl-review.md).

**Reading copy in nd-rl:** [`docs/project_strategy/2026-10-17-peer-rl-review.md`](../../docs/project_strategy/2026-10-17-peer-rl-review.md).
It is the executor's report with the review's corrections, 44 marks *[review]* in the body. `git diff 27b58dd` on
that file shows each change against the executor's text.

## Configuration

- **Repository.** Fork `dan-pandori/nd-takehome`, branch `dan_peer-rl-review`, from the fork's `origin/dan` at
  `4dcd0e4f`. Ran on the second VPS (agent2), every role at `--effort max`.
  - Pre-registration `preregistration/peer-rl-review.md`: committed at `cc078a49` (20:15:24Z; timestamp fix
    `2f7f2dd4`), before any reader note. Addendum 1 (21:36, "Part B: no pod") is in the DONE commit.
  - DONE `0f92c381` (21:53:24Z). Deliverables under `peer_rl_review/`: `REPORT.md`; `notes/` (eight reader notes, an
    audit of Leon's evaluation path, and the executor's claim check `report_check.md`); four scripts; `out/`. Also
    `run_peer_rl_review.md`, `numbers.md` § peer-rl-review and `log.md`.
  - Review [`review_peer-rl-review.md`](https://github.com/dan-pandori/nd-takehome/blob/6fb97e70/review_peer-rl-review.md),
    in three commits:
    - [`875e60d9`](https://github.com/dan-pandori/nd-takehome/commit/875e60d9): §Recount (phase 1, written before the
      executor's write-ups were opened);
    - [`d2f1ebe3`](https://github.com/dan-pandori/nd-takehome/commit/d2f1ebe3): phase 2 claim table and verdict;
    - [`6fb97e70`](https://github.com/dan-pandori/nd-takehome/commit/6fb97e70): corrections after an adversarial check
      of the review, which found one wrong number of the reviewer's own (the Lean total, now 6,078).

    The reviewer kit `review_prr/` holds a formula matcher, the exposure check, pass@k, a Lean harness and a
    bootstrap; the reader notes are in `review_prr/readers/`.
- **Timeline.** Executor Sat 2026-10-17 20:10–21:53 UTC; reviewer 21:54–23:42; librarian from 23:42.
- **Sources read** (nd-rl branches, read-only through `git show`):
  - Leon: `leon/main` and his GRPO v2 / thinking branches (2026-10-05 → 10-16);
  - Robbie: `robbie-experiments`, plus his public HF `summary.json` files for the unpublished `long-rl-20261014-*`
    runs;
  - Dmitry: `dmitry/main` / `selection-sprint`;
  - Charles: `charles/main` (the textbook and pretraining-data releases, including
    `release_2026-10-12_textbook72_distract2`).

  **None of Leon's or Robbie's RL numbers has been reviewed.** Dmitry's sprints were reviewed only by reviewers they
  commissioned themselves.
- **Our models** (every † number; Lean alone): `best_model.ALiBiGPT` 6 × 384, **9,560,832 params**, `lean_staten`
  proof state, from scratch on K12 (`data/kh/train_k12.jsonl`, 155,000 proofs, 2–12 lines, cap 12).
  - frozen = `stage1_best12_s{0,1,2}_b1200` (best-state; no RL).
  - r8 = `la_T1_best12_s{0,1,2}_r8`, after 8 T1 EI rounds (k 32 on the 4,495 `data/ladder/rl_targets.jsonl`, K12
    replay). Two trainings of one recipe share this name: the **best-state seeds** (r8-bs) and the **trajectory seeds**
    (r8-tj, A6000 ladders).
  - r16 = after 16 rounds: `rl-continue`'s continuation (rounds 9–16) of the trajectory seeds' r8 ladders.
  - gb-* = `grpo-best`'s GRPO arms (default, unlikely, pass@4), at EI r8's attempts, from the trajectory seeds'
    Stage-1.
- **Leon's models** (his numbers, unreviewed):
  - ≈ 9.6M per-step proof-state models on our vendored environment (`danstate`), from scratch, with a think block
    before each step.
  - Stage 1: T0 (K12, no think, our recipe), T0S (K12's theorems with proofs from an "honest search" through the
    environment, no think), and S1g3 (search-trace thinking).
  - RL: Dr. GRPO with an L2 (10–12-line) → L3 (13–15-line) curriculum, 39 iterations × 4,096 attempts per stage,
    KL 0 in L3, and verifier "stop" rollouts.
  - Final recipe `grpo_think_vs_mb_kkl0`, think budget 64, 3 seeds. His checkpoints are on his machine only.
- **Inputs.** 27 stored per-attempt read files from the bucket (plus their side files), sha256-identical to the
  reviewer's own downloads:
  - `best-state/artifacts/bs/eval/{T1,Fz}_best12_s{0,1,2}__tb72.jsonl`;
  - `guided-tts/artifacts/gt/eval/best12_s{0,1,2}_{plain,logical}.rows.jsonl.gz`;
  - `capability-defs/artifacts/cd/j3/s{0,1,2}_r16_logical.rows.jsonl.gz`;
  - `rl-continue/artifacts/rc/eval/s{0,1,2}_r16__tb72_x1.jsonl`;
  - `grpo-best/artifacts/gb/eval/gb_{default,passk,unlikely}_s{0,1,2}_r8__tb72_x1.jsonl`.
- **Read settings of those files** (Leon's read caps too): k 256 per theorem, T 0.8, `max_steps` 96, `max_action` 512.
  Guided = the `logical` arm, ≤ 10 redraws. Sample seed 0 (best-state) or 1 (the others).
- **Leon's read-out protocol** (the common protocol):
  - the unbiased pass@1 and pass@5 per problem, averaged over problems and seeds;
  - n 64 per problem in his reads; ours are re-scored from n 256, and the estimator is unbiased either way;
  - textbook72, and **Charles149** = Charles173 (`candidate_v0` 87 + `candidate_v1` 72 + `batch3` 14) minus the 24
    names in his `charles_clean.report.json`;
  - plain and guided reads, judged by Lean alone.

  The textbook72 and Charles files are byte-identical on both sides. 19 of the 21 vendored environment / judge files
  are blob-identical to our `origin/dan`; the other 2 are his provenance files.
- **Checker: Lean alone.** The stored reads were judged by `lean_gate` on the literal text. `nd_verify` is unmodified
  (tree `9437bb72`) and unused: the run's four scripts import no judge, no Lean and no torch. `artifacts/TEST_RUN_DONE`
  is unchanged.
- **Compute.** VPS CPU only, under a minute in total. The executor used no GPU-seconds, generated tokens, training
  steps or Lean checks. The reviewer's Lean re-check also ran on the VPS CPU.
- **Pods and cost.** None: **$0.00 and 0.00 pod-hours** of the brief's optional $30 / 60 h. The launcher registered the
  budget at 20:10:13Z. The pre-registration's addendum says it was not registered, a slip the review calls trivial.
- **Bucket.** `hf://buckets/dan-pandori/nd-rl/peer-rl-review/{out,data}` holds the five numeric outputs and
  `data/charles149_removed.json`; the librarian listed all 6. The report and notes are on the fork branch only, because
  they quote private nd-rl branches.

## Observations

All numbers are reviewer-checked unless labelled. Leon's numbers are his own unreviewed reads, quoted from his files.

**1. Ours on Leon's read-out protocol.** Pass@1 / pass@5 in % of the set, mean of 3 seeds (the IQM at n = 3), with
per-seed pass@1 in brackets.

| our model / read | textbook72 pass@1 | textbook72 pass@5 | Charles149 pass@1 | Charles149 pass@5 |
|---|---|---|---|---|
| frozen (Stage 1 only, best-state seeds) | 15.2 [14.8 / 14.0 / 16.7] | 23.0 | — | — |
| r8 plain, best-state seeds | 39.2 [39.6 / 39.0 / 39.0] | 53.6 | — | — |
| r8 plain, trajectory seeds | 39.2 [38.9 / 36.0 / 42.6] | 53.1 | 31.8 [29.5 / 29.8 / 36.1] | 48.5 |
| r8 guided, trajectory seeds | 47.0 [45.7 / 44.9 / 50.4] | 61.1 | 43.4 [41.7 / 40.9 / 47.6] | 60.9 |
| r16 plain | 42.9 [40.9 / 43.6 / 44.2] | 57.6 | not read | not read |
| r16 guided | 50.7 [48.7 / 50.5 / 52.9] | 64.7 | not read | not read |
| GRPO arms, r8 (default / passk / unlikely) | 41.0 / 37.3 / 42.0 | 49.8 / 51.4 / 51.2 | — | — |

Leon's numbers on the same protocol (pass@1 / pass@5, %):

| Leon model | textbook72 plain | Charles149 plain | textbook72 guided | Charles149 guided |
|---|---|---|---|---|
| final recipe (GRPO, 3 seeds) | **74.0 / 82.3** | **70.5 / 78.9** | 80.3 / 86.5 (pass@1 78.0 / 83.6 / 79.2 per seed) | 77.4 / 84.1 |
| his EI baseline | 68.4 / 77.9 | 62.2 / 73.8 | | |

- **Leon's model is far ahead under an identical read-out.** On textbook72 pass@1 it leads our best plain model by 31
  points (74.0 vs 42.9, r16); on Charles149 by 39 (70.5 vs 31.8, r8). Guided reads give 80.3 vs 50.7 and 77.4 vs
  43.4. His pass@5 (82.3 %) is above our plain coverage at 256 attempts (r16: 51 / 57 / 56 of 72, 70.8–79.2 %).
- **The gap is not GRPO against EI.** His EI baseline leads by about as much (68.4 / 62.2 against our r8's 39.2 /
  31.8).
- **The two pipelines agree where they should.**
  - His no-think K12 base (T0, 3 seeds, k 256) reads 17.2 % [10.4, 25.1], against our frozen 15.2 %.
  - Two other people's reads of our r16 s0 reproduce: Robbie's temperature sweep gives 0.407 and Dmitry's random pick
    of 8 gives 0.405, against 40.9 % in the re-score. Dmitry's pass@8 is 0.594 against 0.5985.
  - Those two test Robbie's and Dmitry's harnesses on our model, not Leon's. The protocol identity with Leon rests on
    his T0 and the code audit.

**2. Why the gap: what Leon's own ablations show** (unreviewed; 2–3 seeds before RL, single seeds after). In the
reviewer's wording, **in his own ablations the largest single step is Stage-1 proof style**. The order of the steps
below is one of several, so the decomposition is not unique.

| step (textbook72 pass@1; Charles149 in brackets) | Leon | ours |
|---|---|---|
| K12 Stage-1, no think | T0 17.2 | frozen 15.2 |
| + EI on generator targets | T0 → EI r1 / r2 30.8 / 32.4 (24.8) | r8 39.2 (31.8); r16 42.9 |
| search-proof Stage-1, no think | T0S 31.6 | — |
| search-proof base + the same EI on the same pool | T0S → EI r1 / r2 52.4 / 53.6 | — |
| + the `hard712` pool (designed from dev58 failures) | T0S → EI r2 62.8 | — |
| v2 curriculum, no think, s0 | EI 63.5 (54.4); GRPO 68.1 (59.6) | — |
| + thinking + GRPO recipe | 74.0 (70.5); EI with thinking 68.4 (62.2) | — |

- **Stage-1 proof style.** Changing only the proofs of the same K12 theorems adds +14.4 [+8.0, +21.4] on textbook72
  before RL (2 seeds each), and ≈ +21 after the same EI on the same pool (single seeds). His mechanism: Lean rejections
  of environment-accepted textbook72 attempts fall from 55.4 % to 17.8 %. Our frozen model's rejections are at his T0
  level (Observation 6).
- **The pool:** up to +9.3 [+3.8, +15.2] on textbook72. Part of that is by design, since dev58 shaped `hard712`.
- **Thinking:** +4.2 to +11.7 per attempt at ≈ 3× the tokens, one seed per arm; −0.9 / +5.4 / +1.5 at 1k matched
  tokens.
- **GRPO over EI:** ≈ +5, against a less-tuned EI (about 4–5 EI pilots vs about 15 GRPO pilots).
- **Ruled out as explanations:**
  - the judge, the files and the read settings, which are identical;
  - seed noise, also with shared-resample intervals;
  - compute per attempt for his no-think arms: they use fewer tokens per attempt than ours (≤ 181 against 211–235),
    and are also far ahead (63.5 / 68.1, s0);
  - RL attempts: his 320k per seed against our r8's 1.15M RL-target attempts.

  His **thinking** arms spend ≈ 434–488 tokens per attempt, ≈ 2× ours. The final recipe's tokens per attempt are not
  tabulated.

**3. Exposure on both sides** (the reviewer's own substitution matcher, `review_prr/overlap.py`; counts are evaluation
theorems with at least one substitution instance in the training file).

| training file | textbook72 (dev58 / train14) | Charles173 |
|---|---|---|
| `data/ladder/rl_targets.jsonl` (4,495; the RL targets of every ladder) | **10 theorems** (9 / 1), with 40 instance targets each (400), all textbook-schema rows; 0 renaming twins | 0 instances; 0 twins |
| K12 (155,000; Stage 1 of every model above, and the theorem source of Leon's Stage-1 sets) | 13 theorems (9 / 4); 1 renaming twin with premises swapped (`textbook_3ed45280e6c686e76ac6`, the known disjunctive-syllogism twin) | 22 theorems (16 / 3 / 3 in `candidate_v0` / `v1` / `batch3`), 14 of them in Charles149; 1 twin (`textbook_9d21fcccbcd6c3ebf207`, on Leon's removal list) |

- **What it means for our counts.**
  - Stage-1 solves 6 / 5 / 3 of the 10 `rl_targets`-exposed theorems, and r8 solves 9 / 9 / 10. Of RL's +20 / +24 /
    +25 textbook72 gain (best-state seeds), +3 / +4 / +7 is on schemata it trained on.
  - On the other 62 (`tb72_clean62`) pass@256 is 43 / 42 / 42 (r8-bs) and 42 / 47 / 46 (r16). Pass@1 is 35.4 (r8
    plain), 38.5 (r16 plain) and 46.6 (r16 guided), 3.8–4.4 points below the full set.
- **Leon's side.**
  - dev58 was the design set for the generator of his RL pools; he calls it "no longer a clean held-out set".
  - His EI-lineage round-2 pool holds curried twins of 3 dev58 problems and of 10 other release or textbook72 problems.
  - textbook72 was read in the single-seed pilots that chose the merged recipe's components, and Charles149 was his
    selection set for KL 0.
  - His pools were checked against Charles173 by renaming and curried class (no hits), not by substitution instance.
  - At most 3.4 points of the Charles149 gap sit in its 5 four-premise problems.

**4. GRPO against EI, across people.**
- **Leon** (3 seeds, matched attempts; pass@1 / pass@5):
  - textbook72 +5.6 [+0.8, +10.9] / +4.4;
  - Charles149 +8.4 [+3.6, +13.1] / +5.1;
  - 13–20, his primary v2 set, +0.6 / −0.1.

  The Charles149 lead shrinks at matched tokens (+7.3 at 1k → +1.7 [−3.2, +7.2] at 8k) and under guided reads
  (+3.7 [−0.6, +8.3]).
- **Ours** (`grpo-best`, same Stage-1 seed, textbook72; pass@1 / pass@5):
  - default +1.8 / −3.4; unlikely +2.8 / −1.9; pass@4 −1.8 / −1.8. None is resolved.
  - With one theorem resample shared across seeds and arms (Leon's method), default − EI is [−4.6, +8.2]; with seeds
    also resampled per arm, as in Leon's own differencing, [−5.9, +9.2].
  - Over the 322 theorems, the default and unlikely arms lose more theorems than they gain against EI (15 gained vs 38
    lost, and 15 vs 44; sign tests p 0.0022 / 0.0002).
- **Robbie** (whole proof; unpublished 3-hour runs, known only from HF `summary.json` files; dev1108 at k 16):
  - EI 852–929 against GRPO 272–593;
  - in the same-harness pair, v4, EI 925 / 859 against GRPO 276 / 272.
- **Redraws in training.** GRPO with verifier redraws during training collapsed in Leon's single-seed arms: on 13–20
  pass@1 the paired s0 difference is −14.2 [−15.7, −12.6].
- **A reconciling hypothesis, untested** (the report marks it as one): GRPO sharpens and EI covers; GRPO needs Leon's
  safeguards; step-level verifier "stop" signal in rollouts helps.

**5. Robbie, Dmitry and Charles report no RL gain over ours on textbook72.**
- **Robbie.** Whole-proof `lean_seq` models, 9.56M, pretrained on cap-6 data. His whole-proof numbers are
  **Lean ∧ `nd_verify`**; his proof-state comparison and the temperature sweep are Lean alone.
  - **Combined pipeline c005:** dev1108 904.3 (915 / 893 / 905), against our best-state r8 1,058 / 1,050 / 1,055 (Lean
    alone, k 64, one draw). textbook72 30 / 72 at k 256 (36 resampled), against our r8 52 / 51 / 52.
  - **Retry-EI** with Lean error tokens: every arm is below plain EI on his dev metric (iid 771; resume 762, blank 718,
    retry 712, repair 706, resume_rl 619).
  - **Resampling invalid lines at test time:** textbook72 pass@1 17.6 → 26.0 %, and 30 → 36 solved at k 256.
  - **TTRL** at 5× compute scores 1,079, against 1,020 for sample-more and 1,088 for targets-only EI. TTRL and
    targets-only EI are both transductive.
  - **His proof-state port** beats whole proof on all four recipes. The raw dev medians are +111 to +178; matched for
    compute they are **+46 to +156**, which he calls upper bounds.
  - **The gap to us has these causes:**
    - interface, measured (above);
    - pretraining cap, measured: our cap-6 T1 textbook72 IQM is 37.3 of 72 against 51.7 at cap 12;
    - judge, small by hypothesis: the Lean-only class is 909 distinct proofs beside 281,817 that both judges
      accepted;
    - compute, which is not matched (a confound, not a measured cause).
- **Dmitry.** All of his numbers are on our r16, mostly s0, under Lean alone.
  - **Picking 1 of 8 without Lean** (textbook72, s0): random 0.405, own likelihood 0.540, sibling likelihood 0.567,
    trained judge 0.582, oracle 0.594.
  - **With Lean,** stopping at the first valid proof reaches **0.653 within the first 16 attempts, at 7.2 attempts on
    average**. That is a three-generator mean; the 64-attempt ceiling is 0.694 for s0. The best learned stopping rule
    reaches 0.630 at 9.1 attempts.
  - **His Codex sprint's REINFORCE** raised in-distribution first-proof accuracy by 12.6 pp on a synthetic set, but not
    textbook72 (41.44 vs 41.67 %). Sprint B's per-item data are not in git.
- **Charles** works on data, and none of his pools has been trained on. See Observation 7.

**6. Failure shares on our textbook72 reads.** Shares are of all 18,432 attempts per read, s0 / s1 / s2. "env" means
that the strict grammar or the environment rejected a step; truncations are excluded from it.

| read | ok | Lean-rejected | env | cut-off | Lean-rejected / finished |
|---|---|---|---|---|---|
| frozen (best-state) | .148 / .140 / .167 | .484 / .491 / .557 | .368 / .367 / .275 | .000 / .002 / .000 | .77 / .78 / .77 |
| r8 plain (best-state) | .396 / .390 / .390 | .113 / .174 / .162 | .491 / .415 / .435 | .001 / .020 / .014 | .22 / .31 / .29 |
| r16 plain | .409 / .436 / .442 | .106 / .135 / .193 | .481 / .424 / .364 | .004 / .005 / .001 | .21 / .24 / .30 |

- Our frozen model's Lean-rejected share, 48–56 % of attempts, is at Leon's T0 level (55.4 %).
- RL brings it down to 11–17 %, but then 41.5–49.1 % of r8's attempts end on an environment-rejected step. Whether
  search-proof Stage-1 data remove those is not measured.
- With the pre-filter on, "Lean-rejected" is the pre-filter's decision. In the reviewer's control C2, Lean accepted
  0 of 600 such texts.

**7. The premise-count wall applies to our models (P5).**
- **Charles's measurement** (`release_2026-10-12_textbook72_distract2`; k 32, T 1.0, his CPU environment sampler,
  48 steps, one seed; **judge `nd_verify`**):
  - After two unused premises are added, 25 textbook72 problems cross to ≥ 4 premises. Of those, the ones our
    `la_T1_best12_s0_r8` solves undistracted go **17 → 0**, and `stage1_best12_s0_b1200`'s go 8 → 0.
  - It solves 0 of 120 four-premise generator problems.
  - On his hand-built elementary problems it verifies 0 of 320 samples on the 10 four-premise problems (of 21). At 2
    and 3 premises the same model verifies 154 / 160 and 111 / 128.
- **Our own reads** (pass@256, Lean alone; r8-tj): no seed solves a 4-premise problem, plain or guided.
  - textbook72 has one (0 / 0 / 0). Charles173 has five, all in Charles149, and each gets 0 of 256 in every seed.
  - At 3 premises, Charles173's 28 problems are solved 20 / 19 / 22 (plain) and 26 / 25 / 24 (guided).
  - Plain pass@1 by premise count 0 / 1 / 2 / 3 on Charles149 (s0) is 31.1 / 38.7 / 30.9 / 14.6.
- **The cause.**
  - When the environment writes the premise-listing block, the same r8 solves 118 of 120 four-premise generator
    problems with no training. This is a diagnostic in Charles's re-implementation, judged by `nd_verify`.
  - No ≥ 4-premise prompt exists in K12 or the round-8 mix.
  - That the data cause the wall is **a well-supported hypothesis, not tested by training**.
  - The failure is the environment's `PR position` rule on the first action, before any judge sees the proof. So it
    should be judge-independent for our environment; that is the reviewer's inference, not a Lean measurement.
  - The teacher-forced probability of the right block, ≈ 4·10⁻²⁴, depends on the naming convention. The robust
    evidence is 0 of 2,048 samples.

**8. Pre-registered priors** (the reviewer's scores; the executor's in brackets where they differ).
- **P1** (≥ half the comparison rows differ in k, set, plain/guided or judge): **not supported as worded** [executor:
  "supported, 15 of 16"].
  - In the report's table as built, 11 of 16 rows are apples to apples. Strictly, 3 of 16 differ in one of P1's four
    items. The reviewer's own 17-row table gives 6–7 of 17.
  - The executor's count compared their numbers with our previously published pass@256 counts ("as first reported"),
    a post-hoc reading.
- **P2** (Leon's GRPO − EI gap at his largest k is under half the pass@1 gap): **mixed**.
  - At pass@5 the gap keeps 79 % of itself on textbook72 and 61 % on Charles149.
  - At k 64 (serrano's v2 pairs, s0, not the final recipe) it falls to ≈ 0 on Charles149 (+1 / −1 of 149, against
    +5.0 pp), but not on textbook72 (+2 / +3 of 72, against +4.7 / +7.7).
  - On 13–20 the pass@1 gap is zero within its interval, so the test is vacuous there.
- **P3** (a Leon checkpoint within our MDD of our best plain textbook72 counts at k 256): **untestable**, because his
  checkpoints are unpublished. On his reported numbers it would fail: his GRPO arms solve 62 and 66 of 72 at k 64.
- **P4** (retry and selection help low k more than coverage): **mixed, half supported**. Dmitry's selection and
  stopping: yes, by construction. Retry-EI improved nothing at any k.
- **P5** (≤ 10 % of the crossing problems stay solved): **supported, 0 of 17** (one seed, `nd_verify` judge)
  [executor: "0 of 25"].
- **The pre-registration's disclosure understates what was seen.** Before writing the priors (20:15) the executor had
  read 80 lines of Leon's summary README (20:12). They include "Most 'wildly OOD' axes (≥4 premises, …) fail on
  format", so P5 was partly informed; the brief had also named the wall.
- **Two trivial slips.**
  - The addendum says `podbudget` was not registered; the launcher had registered it.
  - P3 attributes the MDD of 6.5 to `NOISE_FLOOR.md`, which has no textbook72 row. It comes from
    `preregistration/best-state.md:71` (3 vs 4 seeds, pass@256 counts).

**9. Checks.**
- **Lean re-check** (the reviewer's harness, Lean 4.34.0, `#print axioms` allowlist; the reviewer's own ND → Lean
  renderer where only ND strings are stored): **6,078 / 6,078** counted proofs accepted. They cover the frozen, r8
  (both seed sets, plain and guided), r16 (plain and guided) and GRPO (x0 and x1) arms.
  - The must-fail controls failed. Lean accepted 0 / 600 pre-filter-rejected gate-dump texts, 0 / 300 accepted texts
    paired with another theorem, 0 / 20 bare `sorry`, 0 / 300 with the final `exact` removed and 0 / 200 cut at a
    random token.
  - Single flips: 20 / 300 `Or.inl`↔`Or.inr` or `.1`↔`.2` flips passed. In 400 + 400 inspected flips, every acceptance
    was a no-op: a flip inside an `( X ∨ X )` step or on an `( X ∧ X )` hypothesis. Those 20, and the reviewer's own
    renderer's 17 / 300, were not inspected one by one.
- **Every † point value reproduces to one decimal; the bootstrap intervals do not.**
  - The executor's hierarchical bootstrap resamples problems independently within each resampled seed. Leon's method
    shares one theorem resample across seeds and arms.
  - So the executor's intervals are ≈ 21–36 % narrower than the reviewer's shared-resample ones: frozen [8.7, 22.5]
    instead of [10.9, 19.7]; r16 [33.7, 52.4]; default − EI [−4.6, +8.2] instead of [−3.3, +6.8]. Leon's own
    differencing also resamples seeds per arm, which widens default − EI further, to [−5.9, +9.2]. Every
    textbook72-level conclusion is unchanged.
  - `numbers.md`'s "unlikely +4.5 [+0.5, +9.0] on `tb72_clean62`" is not resolved: [−0.6, +10.5] with a shared
    resample, and [−2.4, +11.6] with seeds resampled per arm as in Leon's `stats.py`. It is also a post-hoc subset, so
    it is exploratory.
- **Cut-offs** (inherited, not caused by this run). The stored reads exceed the policy's ≈ 0.1 % per stratum in places,
  so several pass@256 counts are lower bounds.
  - best-state r8 s1: 2.0 % of attempts cut off overall, and 25 / 72 textbook72 theorems above 0.1 %. The worst,
    `textbook_3d2b05bca4cbb83ed73b`, is at 39.5 % and solved 0.
  - best-state r8 s2: 1.4 % (28 / 72).
  - r16 plain: 0.4 / 0.5 / 0.1 % (18 / 23 / 9 theorems).
  - Guided reads: ≤ 0.3 %, step cap only.
- **Inputs and files.**
  - All the input files are sha256-identical to the bucket (33 / 33 in the review).
  - The textbook72 files have the same sha256 as our `data/eval_only` copy and Charles's and Dmitry's sealed hashes.
    Our copies of Charles's files (`data/gt/candidate_*`, `batch3`, `hand_proved_14`) match his 2026-10-02 MANIFEST.
    The 24 removed names are identical.
  - Model labels and judges are complete at section level. Robbie's Lean ∧ `nd_verify` numbers and Charles's
    `nd_verify` numbers are labelled where they enter.

## Rewordings applied (from the review)

All are marked *[review]* in the nd-rl reading copy of the report. Some items sit in files not copied here
(`run_peer_rl_review.md`, `numbers.md`, the pre-registration); those are corrected in this summary only, and the fork
files are unchanged.

1. **P1:** "supported, 15 of 16 rows differ" → not supported as worded. **Not supported as written.**
2. **Dmitry's 0.653.** The report said "sampling to the first valid proof reaches the pool's ceiling of 0.653, using
   up to 64 attempts". It is a 16-attempt, three-generator first-valid oracle at 7.2 attempts; the 64-attempt ceiling
   is 0.694 (s0). The paragraph also set s0-only selection numbers against three-generator means. The executor's own
   claim check had endorsed the 64-attempt reading. **Not supported as written.**
3. **"Most of the gap is Stage-1 data, not the RL algorithm."** "Not the RL algorithm" is established by his EI
   baseline. The rest becomes "in his own ablations the largest single step is Stage-1 proof style", with 2–3 seeds
   before RL and single seeds after, and a decomposition that is not unique. Three table verdicts change:
   - row 6's "AA" → not established, because his Stage-1 read protocol is unstated;
   - row 5's "AA; similar level" → ours higher at the point estimate, after 4× the rounds (his single-seed interval
     [23.4, 41.9] contains our 39.2);
   - row 15's "same direction" is dropped (−0.2 against +3.7). **Not supported as written.**
4. **The inference under the comparison table.** Rows 13 and 14 test Robbie's and Dmitry's harnesses on our model, not
   Leon's. The conclusion that the gaps are differences between the models rests on row 4 and the code audit, which
   checks out.
5. **Charles's wall.**
   - "Caused by the data" → a well-supported hypothesis, untested by training.
   - "0 of 320 samples on 21 problems" → on its 10 four-premise problems.
   - P5's denominator is 17, not 25.
6. **The pre-registration** (Observation 8): the disclosure (P5 partly informed), the `podbudget` statement and the
   MDD's source.
7. **The bootstrap** (Observation 9), including the `tb72_clean62` GRPO − EI difference in `numbers.md` (not resolved;
   exploratory).
8. **"Charles149 (clean for both sides)."** It is clean of renaming and curried overlap, but it is a KL-0 selection set
   for Leon, and K12 holds substitution instances of 14 of its theorems.
9. **Minor:**
   - the redraw arm's 14.9 vs 29.3 compares s0 with s0 + s1 pooled; the paired s0 difference is −14.2 [−15.7, −12.6];
   - pilot citations: the unseen prior is in §3, and "half the learning rate" is serrano's result;
   - unlikeliness +2.1 [+1.0, +3.2] is not resolved from the rerun control's +1.3 [+0.2, +2.4];
   - Robbie's interface advantage is +46 to +156 when matched for compute;
   - "never use redraws in training" → they collapsed in his single-seed arms;
   - the run summary's "Robbie, Dmitry and Charles report no RL gain over ours" holds on textbook72; Dmitry's
     REINFORCE gained +12.6 pp in distribution on a synthetic set;
   - T0's 17.2 has 3 seeds, but the +14.4 comes from a 2-seed table.

**Outside this run's files.** The review found that an earlier nd-rl line looks wrong.
- The evidence atlas says Robbie's factorial cells "were RL-trained on 14 of the 72 textbook72 theorems" (`ATLAS.md`
  L61–62). That line is itself a *[review]* insertion.
- Robbie's code says otherwise. `factorial_20260929/passk.py` L5–7 says textbook72 was "never trained on", `split` is
  only the source file, and the RL pools of `leon_rl.py` are `rl_targets`.
- The librarian checked that docstring and added a marked note to the atlas.
- Robbie's RL does share our 10 schema-exposed textbook72 theorems, through `rl_targets` (Observation 3).

## Interpretation (not a finding)

This is the executor's reading, in the reworded form the review leaves standing.
- A peer pipeline on our own environment, judge and files reaches 74 % textbook72 pass@1, where our best plain model
  reaches 43 %. The advantage is already present before RL, and it survives the same EI.
- The most promising lever in Leon's ablations is the Stage-1 proof style: proofs found by search through the
  environment, instead of K12's generator proofs. In his data it cuts the Lean-rejected share of a frozen model's
  attempts from 55 % to 18 %, and our frozen model sits at his T0 level. For our ladder that is a hypothesis, not a
  result.
- Our pass@256 view hid the low-k gap.

Librarian's note: the project's creation tests so far (`capability-defs`, `create-or-elicit`) all use Stage-1 data of
generator proofs: K12, its knockout and dose variants, or the cap-6 set. The two of those models measured here (r8,
r16) are 31–39 pass@1 points below Leon's recipe. Whether the creation findings hold on a much stronger base is open.

## Limitations and next step

- **Leon's and Robbie's numbers are unreviewed,** from agent-written documents, and many of Leon's ablations have a
  single seed. His checkpoints are on his machine, so nothing of his was re-measured here, and the "theirs on ours"
  direction (P3) is untested.
- **Our side is re-scored from stored reads.** They have n 256 against his 64 (the estimator is unbiased either way),
  and the sampling seeds differ. Leon's attempts carry think tokens, so the match is per attempt, not per token. Our r16
  has no Charles149 read.
- Several inherited pass@256 counts are lower bounds (cut-offs above 0.1 %).
- P5 rests on one seed of Charles's `nd_verify`-judged measurement, plus our own 0-of-256 reads on 6 problems.
- **Next (reviewer), in order:**
  1. **The Stage-1 hypothesis on our side** (also the report's top lesson). Regenerate K12's proofs with Leon's honest
     search (`searchtrace.py`, deterministic, CPU), retrain Stage-1, and run the unchanged T1 ladder.
     - Use ≥ 5 seeds per arm.
     - Read pass@1 / pass@5 / pass@256, plain and guided, Lean alone.
     - Pre-register against an MDD computed for pass@1 at that seed count; best-state's 6.5 is for pass@256 counts.
     - It is not yet costed.
  2. **"Theirs on ours", and a creation test on Leon's recipe.** This needs his S1g3 / T0S and final L3 checkpoints:
     the run's `QUESTIONS.md` item 1, which the executor costs at ≈ $2–4 of pods once the checkpoints exist. Then read
     textbook72 and Charles149 at k 256, plain and guided, and the base's pass@k at large k on the theorems his recipe
     newly solves.
  3. **Cheap, no GPU:** re-score `trajectory`'s stored pretraining-only reads on Leon's protocol. That gives the
     Stage-1 pass@1 of the seeds behind r16.
  4. **P5 under our own judge:** run our r8 on the distract2 set through our Lean-alone pipeline at k 256, T 0.8. The
     `PR position` rule predicts 0 on every crossing problem.
- **Questions for Dan** (the run's `QUESTIONS.md`, 2026-10-17, with defaults):
  1. Ask Leon to publish his checkpoints? Default: no such run.
  2. The fork is public, and `peer_rl_review/` (report and notes) quotes private nd-rl branches. Default: leave it on
     `dan_peer-rl-review`.
- **The report's other lessons (§4), with the review's caveats.** For us:
  - read pass@1 / pass@5 and matched-token curves beside pass@256;
  - close the premise-count wall with 4–5-premise prompts in Stage-1 (Charles's pools exist) or with an
    environment-written premise block;
  - if we run GRPO, use Leon's working parts (stop rollouts, canary and rollback, `--overlong fail`) and avoid
    redraws in training;
  - treat thinking as low priority.

  For them: independent review and pre-registration, creation tests, seeds against a noise floor, a clean headline set
  and per-arm compute records.
