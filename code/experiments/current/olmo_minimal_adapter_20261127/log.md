---
written_on: 2026-11-27
written_by: agent:claude (opus, orchestrator)
---

# Olmo minimal adapter: log

## 2026-11-27 1:22-2:15 am PT (orchestrator): design, reading, discourse, build

- 1:32 am: `DESIGN_DRAFT.md` written (input to the discourse).
- Full-text readings (Sonnet readers; downloads under `/scratch/cowifornia/nd-rl/olmo_elicit/v2/papers/`):
  `literature/edl_papers.md` (Bits That Count, EDL theory paper), `literature/tiny_adapter_papers.md` (Donoway et al.
  2025, TinyLoRA, Sinii et al. ×2), `literature/footprint_papers.md` (Sparse but Critical, Akgül et al., Wang et al.).
- Independent audits (Opus, own downloads under `v2/papers_audit/`): `literature/audit_independent.md` (EDL papers and
  Donoway et al. 2025, then a comparison with the readers' notes), `literature/footprint_audit.md` (token footprint:
  numbers recomputed independently, sign-off with corrections C1-C10).
- Design discourse, four turns: `discourse/edl_design.md`. Adjudicated in `PREREG.md` ("Points settled").
- Build (Sonnet builders, checked by me: tests rerun, core code read): `build_pools.py`, `mini_train.py`,
  `make_plan.py`, `test_mini.py`; `multi_sample.py`, `lora_gate.py`, `stage_cells.py`, `analyze_cells.py`,
  `fig_cells.py`, `pod/mini_*.sh`, `pod/mini_status.py`, `test_sampling.py`; `footprint.py`, `test_footprint.py`.
  First check: 36 CPU tests pass, ruff clean, `bash -n` clean. The old dpo adapter cell is reproduced from its graded
  file by `analyze_cells.py` (63.54 % [53.65, 73.18]).
- 2:05 am `SPEC.md`, 2:10 am `PREREG.md`, both before any pod exists.

## 2026-11-27 2:00-2:15 am PT (orchestrator): revision after the discourse, pre-pod checks

- Pools rebuilt checker-only (no hand labels in any selection) with a third source `unsel`; matched prompts at `dpo`
  115 (train 64 / dev 32); median train tokens own 1,972, unsel 1,895, final 4,341. Checker bound (Σ log2 1/a_i) for
  own at n = 1, 4, 16, 64: 1.4, 4.9, 31.9, 146.7 bits. `types_table.py`: at n = 64, 32 of the 48 reach prompts share
  no instruction id with the 64 training prompts (13 all, 3 some).
- `mini_train.py`: successive halving to 512 steps, early stop, `unconverged`, EDL at m = min(t, n), fixed-step EDL
  at 64, B statistics, attention-only mask option. `analyze_cells.py`: G with a two-level paired bootstrap; on the old
  dpo adapter G = 0.916 [0.825, 1.0], "reaches" true. Prompt control built (`make_prompt_control.py`: Dan's A1 suffix,
  read from his stored A1 request rows, byte-identical).
- My edits to the pod scripts: `mini_pod.sh sample` takes `--seed` and `--pids` (smoke tests, distinct seeds per
  batch); a train stage is named per plan file (`train:<plan stem>`), so a second plan is not skipped as DONE.
- Plans: `v2/plan_dpo_h100.json`, `v2/plan_dpo_h200.json`, `v2/plan_smoke.json` (4 tiny cells, 2 steps, n = 1, for an
  end-to-end check of train → export → gate → multi-adapter sample on two dev prompts).
- Tests: 51 pass (test_mini 24, test_sampling 23, test_footprint 4); ruff clean; `bash -n` clean; dry runs of setup,
  train and sample print the expected commands; the push contains code, pools, plans and stage-2 L inputs only (no
  sealed set, no credential).
- Rule slip, reported: the footprint worker ran `pkill -V` once (prints the version only; nothing was signalled).
- 2:15 am: `mini_pod.sh fetch` writes to `$V2/runs_<alias>/<ckpt>/` (two pods both write `runs/dpo/logs` and `state`,
  so one local tree per pod); `analyze_cells.py` falls back to serrano's pool copy when cell.json names a pod path.
  51 tests pass. Waiting for pods.

## 2026-11-27 2:14-2:50 am PT (orchestrator): pods, smoke, first problem

- Pods (created by the coordinator 2:14 am PT): rp-om1 H100 ($4.39/h), rp-om2 H200 ($5.29/h). Setup 2 min each
  (versions as the 25 Nov run). dpo weights downloaded and verified on both (commit 97604023, 14 shards, 64.47 GB).
- Smoke: 4 two-step adapters (A16, O1, M256, M4k-attn) trained on rp-om1 (reload diff 0, base check after unload 0),
  gated on rp-om2 (all UNINFORMATIVE, as expected for 2 steps; the engine loads r16, r1, o_proj-only and masked
  adapters together), sampled on 2 dev prompts × 4 draws each through one engine, graded. Pipeline works end to end.
- Prompt control (dpo + Dan's answer-only suffix, no adapter), 384 draws: done 2:37 am.
- Problem (2:45 am): own-source dev loss rises with training at every lr (Amendment 3 in PREREG). The rp-om1 chain was
  stopped; `mini_pod.sh stop` killed the chain's process group but not the `timeout`-wrapped trainer (GNU timeout
  makes its own process group); I stopped that by pid (2640/2641, my own processes). Fix requested from the builder.
- 2:46 am: rp-om1 restarted on `plan_dpo_h100b.json` (final A16 n4, A16 n1, O1, M16k, M4k); rp-om2 trains final A16
  n64 (`plan_dpo_h200b.json`). 2:47 am Amendment 3 written before any reach result was opened.
- 2:50 am, prompt control read (after Amendment 3): 6.25 % of draws [1.3, 13.0], G = 0.09 [0.02, 0.19], does not
  reach; prompts with a pass in 8: 18.8 %; median answer 108 words, median thinking 2,628 tokens.
- 2:50 am: own A16 n64 dev-selected cell renamed on rp-om1 to `dpo__own__A16__n64__devsel` (kept, not sampled), so the
  fixed-recipe cell can take the canonical id. Builder added `fixed` / `recipe_from` cells to mini_train.py (tests).
  The builder's stop fix (`timeout --foreground`) would have stopped timeout from killing vLLM's child process group on
  expiry, so I reverted it: stages run under plain `timeout` again, and `mini_pod.sh stop` now signals every process
  group whose leader descends from the chain pid (tested locally; test_stop_reaches_timeout_child_groups). 54 tests
  pass, ruff clean.
- 2:54 am: two local drivers (logs/drive_om2_b1.sh, logs/drive_om1_c.sh; nice, timeout 5 h):
  rp-om2: after final A16 n64 -> plan_dpo_h200c (own + unsel A16 n64, recipe from final A16 n64) -> fetch -> stage
  batch_b1 (final unforced; own, unsel forced since `did_not_train` is descriptive for fixed cells) -> gate + sample,
  seed 660001. rp-om1: when final O1 n64 ends -> stop -> plan_dpo_h100c (own fixed A16 n4, n1, O1; final M16k, M4k).
  Pod ledger: $V2/ledger.md (launch stop 7:55 am PT with both pods up).

## 2026-11-27 4:13-5:05 am PT

- 4:12 am: final O1 n64 DONE (lr 3e-4 by the edge rule, step 128, dev 0.5403). rp-om1 chain stopped by the driver;
  the new stop logic signalled both process groups (chain 3386, timeout 3420); h100c started at 4:12 am. Own fixed
  cells (A16 n4 at 3e-5 × 64; A16 n1 at 3e-3 × 1; O1 at 3e-4 × 128) DONE by 4:26 am, fetched and staged as batch_b2.
  Batch 2 was not sampled because own A16 n64 did not reach (rule 3).
- 4:59 am: batch 1 sampled (11:47-12:58 UTC, about 1,400 tok/s) and fetched md5-equal. Read: see PREREG Amendment 4
  (pipeline check fails; nothing reaches). 5:01 am: batch R (old recipe, lr 3e-4 × 64, final, own and unsel) training on
  rp-om2 (plan_dpo_h200r.json). 5:01 am: Amendment 4.
- handcheck.py written (Dan's rule-1 screen verbatim, blinded items in his format, his B_s2300 labelled passes mixed in
  as a reference; tested on the prompt control).

## 2026-11-27 5:05-6:10 am PT

- 5:57 am: capacity ladder done on rp-om1. M16k: lr 0.3, dev-selected step 128 (0.5873); 256 and 512 were worse.
  M4k: lr 1.0 (edge rule), step 64 (0.6116), early stop at 256.
- 6:05 am: `step0-train` (new subcommand) scored the first 64 training examples of each pool at step 0. Train set
  against dev, nats per token: final 0.6068 vs 0.6341, own 0.7048 vs 0.7140, unsel 0.7138 vs 0.7524. The training
  responses are easier than the dev responses, which biases the raw EDL_tok negative; `edl_corrected` adjusts for it.
- 6:08 am: the 25 Nov adapter loaded unchanged as cell `dpo__final__OLD_A16__n64` (`init_adapter`; all 896 tensor
  names matched, re-export equal, reload diff 0). Gate file written; staged as `batch_old`. Its dev loss on our final
  dev set is 0.5015. That is not comparable with our cells: 12 of our 32 final dev responses are identical (same
  token ids) to responses in its 64-example training set, and 14 share a prompt with it. Our 64 final training
  examples share 32 prompts and 19 identical responses with it.
- 6:09 am: rp-om1 handed back to the coordinator (all fetched md5-equal).

## 2026-11-27 6:37-6:45 am PT

- 6:37 am: batch R sampled (13:24-14:37 UTC) and fetched md5-equal; read with all cells and contrasts
  (`cells_summary.*`, `cells_contrasts.*`, union tree `v2/runs_union/`). R-final 40.6 %, G 0.585 [0.474, 0.695];
  R-own 0.0 %; R-unsel 0.0 %. The pipeline window is missed again (PREREG note after batch R).
- 6:38 am: the 25 Nov adapter is sampling through this path on rp-om2 (batch_old, seed 660004). Coordinator told.
- 6:40 am: hand check round 1 (`v2/handcheck/round1`). Dan's rule-1 screen ran on every pass: final dev-rule 0 of 166
  screened, final one-pass 7 of 156, own dev-rule 0 of 1, prompt control 2 of 24, B_s2300 19 of 1,066. 95 blinded items
  (24 per final cell, 22 control, 1 own, and 24 of Dan's labelled B_s2300 passes) went to four Sonnet readers with
  Dan's reader prompt.
- Capacity (`capacity.py` -> `v2/capacity.*`): the raw EDL_tok is negative for every cell because the training
  responses are easier than the dev responses. The difficulty-matched values are in the summary.
- 6:42 am: the trainer's `edl_fixed_step` dropped every lr whose run was early-stopped at any step, even after step
  64 (M16k lr 0.3 and M4k lr 1.0 were dropped). `capacity.py` now recomputes the step-64 EDL from `runs.jsonl` with
  SPEC's rule (the lowest dev loss at 64 among the lrs that reached 64). One-pass C (difficulty-matched): A16 1, O1
  0.90, M16k 0.68, M4k 0.27. The trainer is not changed now; fix it before stage 2.
- Hand check round 1 read (four Sonnet readers, one read per item). Estimated gamed share of passes: final dev-rule
  17 %, final one-pass 16 %, prompt control 21 %. Readers agree with Dan's labels on 21 of 24 reference items.

## 2026-11-27 7:12 am PT

- The 25 Nov adapter sampled through this path (batch_old, seed 660004, 14:38-15:10 UTC): 64.8 % checker passes,
  G 0.934 [0.836, 1.034], gate PASS (corr 0.998). It reproduces the 25 Nov 63.5 %. The sampling path is validated,
  so the gap to our final cells is in training (old minus R-final +24.2 pts [+16.2, +32.0]).
- rp-om2 fetched md5-equal (127 files) and handed back at 7:12 am.
- 7:13 am: hand check round 2 (the 25 Nov adapter's passes; 32 of 249 screened by rule 1, 7 of 24 read as gamed;
  estimated 38 % gamed). rp-om2 deleted by the coordinator at 7:13 am ($26.32); stage 1 total $43.54.
- 7:16 am: summary written (`experiment-summaries/2026-11-27-olmo-minimal-adapter/README.md`, figures `stage1.*` and
  `footprint.*`); 59 tests pass, ruff check and format are clean. Stage-1 report handed to the coordinator.

## 2026-11-27 7:20 am PT onward (CPU only; no pod)

- Coordinator's decisions: stage 2 as pre-registered is not run (PREREG decision, 7:20 am). Amendment 5 (7:20 am)
  pre-registers the gap test (a) and `s1100` (b). Nothing is launched.
- Trainer fixes requested from the builder: the step-64 lr rule, and step-0 scoring of the training set in every
  run, with tests.
- Stage-1 numbers the bug touched: only `edl_fixed_step` in the `cell.json` of the final M16k and M4k cells. The
  summary's capacity table was already recomputed from `runs.jsonl` and is marked († rows).
- Entry added to `experiment-summaries/README.md`.
- 7:23 am: the builder's trainer fixes are in and verified.
  - `fixed_step_block` keeps an lr whose run reached step 64, even if it was early-stopped later.
  - Every cell now scores its first n training examples at step 0 (cached per pool) and writes
    `edl_*_corr`, `step0_train_nats_per_tok` and `delta_nats_per_tok` into `cell.json`.
  - `test_stage1_capacity_reproduced` rebuilds the stage-1 capacity table from the stored logs: lrs 1e-4, 1e-3, 0.3
    and 1.0, and 8,916, 8,044, 6,043 and 2,425 bits.
  - 62 tests pass; ruff check and format are clean.
- 7:37-7:41 am: independent audit of the stage-1 report (`discourse/stage1_report_audit.md`). All its findings are
  accepted. README corrected: the 25 Nov pool facts, the gamed-gap claim, capacity wording, the example-axis caveats,
  the scope of the null, style numbers, what the files show about the gap, provenance and deviations. PREREG: an
  erratum mark on the note after batch R, and Amendment 6 (7:38 am) replacing 5(a) and adjusting 5(b). Replies per
  finding are appended to the audit file. The index entry is corrected.

## 2026-11-27 7:43 am PT onward: Amendment 6(a) step 1 on rp-om3

- The coordinator committed the corrected state as 156bf2b, and the hash is now in the README.
- Pod rp-om3 (zoes0war84zae2): H100 SXM, $4.39/h, created 7:42 am, budget $6. Setup 7:45-7:52 am. The training chain
  (download, then `plan_dpo_a6_step1.json`) started at 7:53 am.
- Pool `v2/pools/dpo/final25`: the 25 Nov adapter's 64 training responses, in its order-0 permutation. The stored
  online pids of the 25 Nov lr-3e-4 order-0 row match this order exactly. Our final dev set is used as the test set.
- Plans: `plan_dpo_a6_step1.json` (the replication cell, seed 523373504) and `plan_dpo_a6_step2.json`:
  - corner B: 25 Nov pool, our seed 1581104982, which is R-final's recorded `init_seed`;
  - corner C: our pool, seed 523373504;
  - own lr 1e-3 × 64;
  - own lr 3e-4 × 192.
- `compare_adapters.py` (with a test): ΔW cosines and norms per module from low-rank traces. Sanity run on the 25 Nov
  adapter against R-final: norms 29.56 and 28.78 (the audit has 29.6 and 28.8), total cosine 0.015.
- 8:07 am: step 1 fetched (24 files, md5-equal). Comparison with the 25 Nov adapter: "no match" by the pre-set rule
  (15 of 448 modules below cosine 0.90, minimum 0.79), though the total cosine is 0.986, the median module cosine
  0.998 and the norm ratio 0.997. No step-2 or 6(c) cell was trained. rp-om3 was handed back at 8:08 am and deleted
  by the coordinator at 8:08 am ($1.86).
- 8:08-8:10 am, CPU diagnosis.
  - The low-cosine modules all sit in layers 0-4.
  - Init seed, step-1 score and training order are identical between the two runs.
  - Online-score drift matches our own same-seed pairs (stored runs.jsonl).
  - No weight-level baseline exists. Amendment 7 is proposed (a training-only same-seed rerun, about $0.60 in the
    step-2 session) and is not in force.

## 2026-11-27 8:11 am PT onward: Amendment 7 on rp-om4

- Pod rp-om4 (sulqgb3ig8eh9m): H100, created 8:11 am, budget $5. Setup 8:12-8:16 am; the chain started at 8:16 am
  with `plan_dpo_a7.json` (the rerun, then corners B and C, then own 1e-3 × 64 and own 3e-4 × 192).
- 8:30 am: the rerun was done and fetched (adapter md5 d09cae84 on both sides).
  - Ours against ours: total cosine 0.986, minimum 0.75, 13 modules below 0.90, mostly in layers 0-4.
  - Rerun against the 25 Nov adapter: total cosine 0.991, minimum 0.815, 8 modules below 0.90.
  - Reading: **"drift"**. The coordinator was told at 8:31 am.
- New: `dev_check.py` (the dev-prompt recipe rule, tested on synthetic draws) and `plan_s1100_a6b.json`.
  - Pid files: `stage/pids_dev_dpo_own.txt` and `stage/pids_dev_s1100_own.txt` (32 dev prompts × 4) and
    `stage/pids_L16.txt` (48 × 16).
  - `dpo` plain on its own dev prompts: 40.8 % (F_dpo_ifb, 16 per prompt). `s1100` plain on its dev prompts: 78.5 %
    (arm-A rows, 8 per prompt).
- 8:34 am: sent the coordinator a timing projection. The last cell was due before the 55-min mark, so it was not needed.
- Training finished at 9:02 am. FETCH_OK at 9:03 am: 56 files (2.6 GB) md5-equal. All 5 adapters were md5-compared
  on both sides. The coordinator was sent `POD READY TO DELETE: rp-om4` at 9:03 am. No sampling.
- Held-out loss on our final dev prompts at step 64 (descriptive only; sampling decides step 2):

  | cell | dev loss |
  |---|---|
  | B (25 Nov pool, our seed) | 0.5021 |
  | step-1 cell (25 Nov pool, 25 Nov seed) | 0.5020 |
  | rerun (25 Nov pool, 25 Nov seed) | 0.5016 |
  | C (our pool, 25 Nov seed) | 0.5403 |
  | R-final (our pool, our seed) | 0.5403 |

  On held-out loss the pool matters and the init seed does not.
- The own candidates' dev loss rose as before (step 0 0.714):
  - lr 1e-3 × 64: 0.811;
  - 3e-4 × 192: 0.797, after 0.717 at step 64 and 0.780 at step 128.
  - Both are `did_not_train` and are staged with `--force` (descriptive for fixed cells, as in batch_b1).
- Prepared, not run: `$V2/logs/drive_a6s2_sample.sh` (step-2 + 6(c) sampling) and `$V2/logs/drive_a6b_s1100.sh`
  (6(b): training plus sampling at `s1100`, with `SEED2=1` for the conditional second seed).
  - Plans: `plan_s1100_a6b.json` and `plan_s1100_a6b_seed2.json`.
  - Tests: `test_dev_check.py`; 67 tests pass, ruff clean.
- 9:04 am: the coordinator deleted rp-om4 at 17:03:57Z after his md5 check (56 of 56 files equal). Cost $3.81; line
  total $49.21. The pod meta is in `$V2/logs/rp-om4_meta/`. Ledger updated.
- Nothing is sampled until the coordinator says so. The H200 is with metrics-2 until about 12:30 pm PT, and Leon has
  not yet seen the plan.

## 2026-11-27 12:50 pm - 1:04 pm PT: audit of the sampling step (T1-T14), Amendment 8

- The audit's verdict was GO WITH FIXES. The coordinator decided T1, T2, T3 and T5 (yes) and the notes to carry.
  Replies are in `discourse/stage1_report_audit.md`.
- Tracked drivers: `pod/drive_a6s2_sample.sh` (step 2 + 6(c), with A′) and `pod/drive_a6b_s1100.sh` (6(b),
  `MODE=train|sample|all`), with helpers in `pod/drive_lib.sh`. The copies in `$V2/logs/` are now "superseded" stubs.
- `dev_check.py`:
  - eligibility: an unsampled or incomplete candidate cannot win;
  - integer pass counts;
  - exit 3 with no winner;
  - labels (`--stage`, `--gate`).
- New: `a6_reading.py` (readings in the order 1, 2, 4, 3, 5, the A′ replicate check and X) and `a6s2_analyze.sh`
  (union tree, fixed contrasts, reading). Tests: `test_dev_check.py` and `test_a6_reading.py`.
- Driver control flow was simulated with stubs in five cases (see the replies). The analysis was checked end to end on
  stand-in cells in a scratch copy of v2.
- PREREG Amendment 8 (1:02 pm), written before any step-2 sampling.
- Nothing sampled. No pod exists for this line.

## 2026-11-27 2:00 pm PT onward: step 2 + 6(c) sampling on rp-om5 (H200, US-CO-1, $5.29/h)

- The coordinator's GO came at 2:00 pm (Leon saw the plan at 1:30 pm; no objection). Pod created 22:00:17Z; ledger row
  added.
- The driver `pod/drive_a6s2_sample.sh rp-om5` started at 2:01 pm from 6db6c5c (0 uncommitted paths in the folder).
  Setup was done at 2:04 pm.
- Dev batch (seed 660007, 32 × 4): gate PASS ×3; DONE at 2:34:16 pm, just inside its 1,500 s cap. Every candidate is
  complete (128 draws). FETCH_OK.
- `dev_check` (`v2/a6/dev_check_dpo.json`), against plain `dpo` 40.8 %:

  | candidate | dev pass rate | minus plain |
  |---|---|---|
  | R-own 3e-4 × 64 | 46.9 % [32.0, 62.5] | +6.1 [−1.4, +14.5] |
  | lr 1e-3 × 64 | 25.0 % [12.5, 39.1] | −15.8 [−29.5, −2.5] |
  | 3e-4 × 192 | 44.5 % [29.7, 59.4] | +3.7 [−5.3, +13.5] |

  - All three are `did_not_train` (held-out loss rose). max|B|: 0.013, 0.047 and 0.034.
  - Winner: R-own's recipe, so there is no tuned cell.
  - Manipulation check: none is above plain, which reads "own SFT at A16 n64 does not move held-out non-reach
    behaviour either": no lift larger than about 15 points. lr 1e-3 lowers the dev rate.
- Reach batch: B, C and A′, seed 660005, 48 × 8. It started at 2:34 pm.
- 4:10 pm: the reach batch was DONE at 00:10:24Z, before its cap; the cap note was not used. FETCH_OK: 40 files, 26 MB.
  The driver printed POD READY at 4:10 pm. The coordinator deleted rp-om5 at 4:11 pm ($11.53; line total $60.74).
- Before the cap: PREREG notes on the cap contingency (3:59 and 4:00 pm), and `cap_bounds.py` (upper and lower sets,
  rate and contrast bounds). It was tested on stand-in data and not needed.
- 4:11 pm: `a6s2_analyze.sh rp-om5`:
  - rates: A 64.8 %, A′ 64.3 %, B 62.8 %, C 46.6 %, D 40.6 %;
  - pool contrasts: B − D +22.1 [+12.5, +31.3]; A′ − C +17.7 [+9.9, +25.8];
  - seed contrasts: C − D +6.0 [−1.6, +13.8]; A′ − B +1.6 [−5.5, +8.9];
  - A′ − A −0.5 [−7.8, +6.8];
  - reading: **"pool"** in both squares.
  The result is in PREREG (4:12 pm).
- 6(b) driver caps resized from measured lengths: about 7k tokens per draw, 1,464 tok/s. Ledger note added. Nothing
  started for 6(b).

## 2026-11-27 4:17 pm PT onward: round-3 hand check of the step-2 square (no GPU)

- The coordinator's go came at 4:17 pm. The protocol gaps (rubric, κ, seeds, batches, bootstrap, readout) were fixed
  in a dated PREREG note at 4:19 pm, before any item was drawn.
- `handcheck3.py` (with `test_handcheck3.py`):
  - `prep`, `third`, `table` and `style`;
  - it reuses handcheck.py's screen, item format, pick and reader prompt.
- Mechanical statistic (metrics-2's definition; `v2/handcheck/round3/answer_style.json`), strict passes:

  | cell | passes | unpunctuated | median words |
  |---|---|---|---|
  | A | 249 | 70.3 % | 31 |
  | A′ | 247 | 78.1 % | 34 |
  | B | 241 | 73.4 % | 38 |
  | C | 179 | 24.6 % | 45 |
  | D | 156 | 28.2 % | 48 |
  | `s2300` stored (B_s2300) | 1,066 | 38.7 % | 32 |
  | `dpo` stored (F_dpo_ifb, 42 reach prompts) | 0 | — | — |

  `dpo` has no passes; over all 672 of its draws, 0.0 % are unpunctuated, with a median of 260 words.
- Rule-1 screen (every pass): A 32 of 249, A′ 31 of 247, B 36 of 241, C 4 of 179, D 7 of 156.
- Items (4:21 pm):
  - 24 per corner plus 8 reference items, 128 in all. The 24 earlier-read draws were excluded for A and for D, and
    31 for the reference items.
  - Seed 660021. Key md5 6d42bd8a…, under `v2/handcheck/round3/`. The items are in a scratch directory with no key.
  - 8 read-1 batches and 8 read-2 batches of 16 items each.
- 4:21 pm: one dispatcher (Opus) started with the 16 Sonnet readers.
- 4:27 pm: 16 readers were done, all files valid, no reruns. 8 splits went to one third-read batch (c00), done at
  4:28 pm. A reader left a helper script `mk.py` in the reads folder; it was kept, and the analysis reads `*.json`
  only.
- 4:29 pm: `handcheck3.py table`. Read 1 against read 2: 93.8 %, κ 0.81. Dan's labels: 6 of 8.
  - Genuine-pass rates: A 42.4 %, A′ 39.8 %, B 37.8 %, C 45.6 %, D 34.0 %.
  - Pool contrasts on genuine passes: B − D +3.9 [−9.5, +16.7]; A′ − C −5.7 [−18.5, +6.3]. Neither survives.
  - Pre-set reading: A′ − D "mostly gamed passes"; A − D "not shown", by 0.4 points.
  - The result is in PREREG at 4:30 pm, and the items, batches and reads are copied to `v2/handcheck/round3/`.
