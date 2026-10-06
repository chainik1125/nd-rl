---
written_on: 2026-10-06
written_by: agent:claude (Worker B)
---

# Sources for the numbers in the sprint summary

This file traces every number in the executive summary and the findings of
`experiment-summaries/2026-10-06-proof-selection-sprint/summary.md`. Each row gives the summary value, the full-precision
value, and the file and key it comes from.

- Paths are relative to this `results/` folder unless they start with `~/runs/proof-selection/` (abbreviated `RUNS/`).
- "A" marks Worker A's files; "B" marks Worker B's.
- The summary itself was not edited.

## Flagged items

| # | summary text | issue | source |
|---|---|---|---|
| F1 | Exec. 4: "closes **18–23%** of that remaining distance **at 2–4 attempts**". Finding 4: "18–23% at the low-budget points" | The 23 % is textbook72 T-sib at **8.26** attempts. The holdout250 points at 2.3–3.9 attempts close 18–21 %, and the 3–4 attempt points of textbook72 do not exist (it spends 7.5–9.8 attempts). Suggested wording: "18–21 % at 2–4 attempts on holdout250; 18–23 % on textbook72 at 7–10 attempts". | `B5_matched.json`: keys `holdout250\|C_sib\|m3` gap_closed 0.2119, `holdout250\|C_sib\|m4` 0.1758, `holdout250\|T_sib\|m3` 0.1840, `holdout250\|T_sib\|m4` 0.2017, `textbook72\|T_sib\|m3` 0.2343 (att 8.26) |
| F2 | Finding 2: "Averaging all three judges is close to averaging the two siblings (within 0.01, except 0.856 vs 0.866 for s2's proofs on holdout250)" | A second exception: textbook72, s1's proofs, 0.852 vs 0.864 (difference 0.012). | `B3_results.json` → `textbook72.cells.auc.s1.siblings` 0.852304, `.xseed` 0.864210 |
| F3 | Limitations: "Worker B first read test stopping results at **02:38**" | 02:38 is when the readout was *logged*. The test stopping and AUROC runs started right after the 02:30 freeze. So the first read happened between 02:30 and 02:38. | `../logs/worker_B.md` headers "02:30 UTC dev results and FREEZE" and "02:38 UTC test readout" |
| F4 (minor) | Exec. 2: "by about **+0.04** nats per token when the proof is valid" | The range is +0.033 to +0.040, so "+0.03 to +0.04" would be exact. Finding 2 states the range correctly. | `B4_mechanism.json` → `<set>\|<gen>.self_minus_sib_shift.valid` |

Every other number below traces to its source within rounding.

## Executive summary

| summary value | full precision | source |
|---|---|---|
| 41 % of textbook72 on one attempt | 0.406901 | `B6_draw_compare.json` → `textbook72.validity.draw1` (B); also `RUNS/pools_v2/MANIFEST.json` → `textbook72__s0_r16__k64.validity_rate` |
| 69 % with 64 attempts | 0.694444 | `B6_draw_compare.json` → `textbook72.pass64.draw1`; `MANIFEST.json` → `pass_at_k` |
| 6,525 theorems | 4,495 + 1,108 + 600 + 250 + 72 | `splits.json` / `MANIFEST.json` `n_theorems` (A) |
| holdout250 84 % of attempts valid | 0.835938 | `B6_draw_compare.json` → `holdout250.validity.draw1`; `MANIFEST.json` |
| textbook72 41 % valid | 0.406901 | as above |
| 1. k=8: 0.836 → 0.918 (oracle 0.940), holdout250 | random expectation 0.8357375; completion_loglik 0.91752; oracle 0.93958 | `selection_holdout250.json` → `selection.all.8.completion_loglik.{random,acc,oracle}` (A) |
| 1. textbook72 0.405 → 0.540 (oracle 0.594) | 0.405451; 0.539792; 0.594236 | `selection_textbook72.json` → same keys (A) |
| "about three quarters" | gap_recovered 0.78756 / 0.71161 | `selection_*.json` → `selection.all.8.completion_loglik.gap_recovered` (A) |
| 2. siblings 0.928 / 0.567 | 0.92766 / 0.567083 | `selection_{holdout250,textbook72}_p3.json` → `selection.all.8.completion_loglik_siblings.acc` (A) |
| within 0.012 / 0.027 of oracle | 0.01192 / 0.02715 | oracle − siblings, same files |
| 0.01–0.04 across generators | 0.011, 0.022, 0.011 (holdout250); 0.025, 0.018, 0.038 (textbook72) | `B3_results.json` → `<set>.variance.<gen>.{oracle8, completion_loglik_siblings8}` (B) |
| lowest cell in 5 of 6 rows | — | `B3_results.json` → `<set>.cells.auc` (B); `phase3_matrix.json` → `matrix_judges` (A, same values) |
| self penalty −0.029 [−0.048, −0.010] | −0.028984 [−0.048054, −0.009862] | `B3_results.json` → `holdout250.twoway.auc` (B) |
| self penalty −0.068 [−0.115, −0.019] | −0.067660 [−0.115096, −0.019322] | `B3_results.json` → `textbook72.twoway.auc` |
| +0.04 nats valid / +0.07 to +0.11 invalid | valid 0.0326–0.0402; invalid 0.0707–0.1068 | `B4_mechanism.json` → `self_minus_sib_shift` (see F4) |
| 3. within-theorem AUROC 0.65 / 0.60 (head) vs 0.79 / 0.69 | verifier_ens 0.64604 / 0.59617; loglik 0.78794 / 0.68720 | `auroc_{holdout250,textbook72}__s0_r16__k64.json` → `scorers.{verifier_ens,loglik_mean}.within_finished` (B); `phase2_ladder_table.md` reference rows (A) |
| 3.3× more hard negatives | 10,676 / 3,195 = 3.34 | `phase2_ladder_table.md` rows `verifier2_rall_s0`, `verifier2_r100` (A) |
| reaches 0.68 / 0.64 | 0.677 / 0.638 | `phase2_ladder_table.md` row `verifier2_ens` (A) |
| 4. 0.924 at 3.9 attempts, holdout250 | 0.923707 at 3.9228 | `B4_headline_curves.json` → `holdout250.count_sib.per_generator`, mean over generators at c = 2 (B); `B4_headline_table.md` |
| oracle pass@4 0.927 | 0.926773 | `B4_headline_curves.json` → `holdout250.oracle_fixed`, k = 4 |
| 0.587 at 7.6, textbook72 | 0.58713 at 7.6032 | `B4_headline_curves.json` → `textbook72.count_sib`, c = 6 |
| oracle pass@8 0.607 | 0.606759 | `B4_headline_curves.json` → `textbook72.oracle_fixed`, k = 8 |
| oracle stop-at-first-valid 0.952 at 2.0 | 0.95192 at 1.9837 | `B4_headline_curves.json` → `holdout250.oracle_stop`, N_max nearest 2 |
| 18–23 % | see **F1** | `B5_matched.json` |
| second draw within 0.03 | max \|diff\| 0.028 (textbook72 pass@64) | `B6_draw_compare.json` |

## Setup

| summary value | full precision | source |
|---|---|---|
| 129,456 attempts, 0 mismatches | 4,608 + 16,000 + 17,728 + 19,200 + 71,920 | `RUNS/scores/aux/*__loglik_summary.json` (B); `MANIFEST.json` replay entries (A) |
| 113,760 | — | `worker_A_phase2.md` line "0 mismatches in 113,760 attempts" (A) |
| 41,216 | 2 × (16,000 + 4,608) | `MANIFEST.json` → `*__s{1,2}_r16__k64.replay_checked` (B, phase 3) |
| 20,608 | 16,000 + 4,608 | `MANIFEST.json` → `*__s0_r16__k64b.replay_checked` (B, phase 6) |
| rl_targets 94 % valid | 0.93960 | `MANIFEST.json` → `rl_targets__s0_r16__k16.validity_rate`; `worker_A_report.md` set table |
| pass@64 0.96 / 0.69 | 0.96 / 0.69444 | `MANIFEST.json` `pass_at_k`; `B6_draw_compare.json` → `pass64.draw1` |
| 116–135 / 34–42 theorems with both classes | 116, 135, 132 / 34, 41, 42 | `B3_results.json` → `<set>.cells.auc_n` |

## Finding 1

| summary value | full precision | source |
|---|---|---|
| within-theorem AUROC 0.788 / 0.687 | 0.787945 / 0.687197 | `auroc_*__s0_r16__k64.json` → `scorers.loglik_mean.within_finished`; `phase3_matrix.json` → `matrix_judges.<set>\|s0.s0` |
| increment over finished +0.022 [0.012, 0.031] | +0.021860 [0.012260, 0.031242] | `../reviews/02_midpoint_audit.md` item 6 (audit recomputation) |
| increment +0.020 [0.005, 0.040] | +0.020347 [0.004578, 0.040139] | same |
| table: random 0.836 / 0.405 | 0.8357375 / 0.405451 | `selection_*.json` → `selection.all.8.<any>.random` (A) |
| a finished attempt 0.896 / 0.519 | 0.89566 / 0.519444 | `selection_*.json` → `completion.acc` (A) |
| own likelihood 0.918 [0.886, 0.946] | 0.91752 [0.886020, 0.945593] | `selection_holdout250.json` → `completion_loglik.{acc,ci}` (A) |
| own likelihood 0.540 [0.441, 0.643] | 0.539792 [0.441108, 0.643266] | `selection_textbook72.json` (A) |
| sibling 0.928 / 0.567 | 0.92766 / 0.567083 | `selection_*_p3.json` → `completion_loglik_siblings.acc` (A) |
| oracle 0.940 / 0.594 | 0.93958 / 0.594236 | `selection_*.json` → `oracle` |
| seed sd at most 0.014 / 0.033 | holdout250 max sd 0.0145 (completion); textbook72 0.0330 (oracle) | `B3_results.json` → `variance`; `worker_B_phase3.md` §4 |

## Finding 2

| summary value | full precision | source |
|---|---|---|
| s2 judge 0.786 vs self 0.788 (holdout250, s0's proofs) | 0.786202 vs 0.787945 | `B3_results.json` → `holdout250.cells.auc.s0.{s2,s0}` |
| judge effects within about ±0.01 | +0.0039, −0.0092 (holdout250); +0.0103, +0.0002 (textbook72) | `B3_results.json` → `twoway.auc.coef.judge_s{1,2}` |
| six paired k=8 intervals exclude 0 | +0.01046, +0.01090, +0.01482 (holdout250); +0.03014, +0.02028, +0.02590 (textbook72) | `B3_results.json` → `<set>.penalty.sel8.<gen>\|siblings` |
| second draw +0.013 [0.002, 0.025] / +0.012 [−0.001, 0.027]; first draw +0.030 | +0.01252 [0.00184, 0.02466] / +0.01243 [−0.00118, 0.02688]; +0.03014 | `B6_draw_compare.json` → `sibling_minus_self.<set>\|draw{1,2}\|sel8_siblings-sel8_self` |
| xseed vs siblings 0.856 vs 0.866 | 0.856388 vs 0.865741 | `B3_results.json` → `holdout250.cells.auc.s2.{siblings,xseed}` (see **F2**) |
| T = 0.8 penalty unchanged | — | `B5_t08_check.txt`; `worker_B_phase5.md` last section |
| mechanism +0.033 to +0.040 / +0.071 to +0.107 | 0.0326–0.0402 / 0.0707–0.1068 | `B4_mechanism.json` → `self_minus_sib_shift` |
| centred separation 0.012–0.018 self / 0.040–0.063 siblings | 0.0121–0.0177 / 0.0395–0.0632 | `B4_mechanism.json` → `<judge>.sep_centred` |
| atomic `have` +0.15 to +0.21 (invalid) | 0.1468–0.2063 | `B4_mechanism.json` → `step_type_self_minus_sib.have_atomic.invalid` |
| r8 vs self +0.005 [−0.022, 0.030], −0.008 [−0.095, 0.073] | +0.00517 [−0.02153, 0.02988]; −0.00794 [−0.09505, 0.07349] | `B4_ancestors.json` → `<set>.r8.auc_minus_self` |
| r8 vs siblings −0.036 [−0.072, −0.002], −0.103 [−0.177, −0.036] | −0.03583 [−0.07189, −0.00197]; −0.10312 [−0.17705, −0.03647] | `B4_ancestors.json` → `<set>.r8_minus_siblings_auc` |
| stage-1 −0.097 [−0.140, −0.053]; −0.082 [−0.181, +0.008] | −0.09653 [−0.14047, −0.05316]; −0.08218 [−0.18069, +0.00794] | `B4_ancestors.json` → `<set>.st1.auc_minus_self` |

## Finding 3 (Worker A's numbers, except where marked B)

| summary value | full precision | source |
|---|---|---|
| 97,776 attempts, 88 % valid, 3,195 finished-invalid | 86,217 / 97,776 = 0.8818; 3,195 | `../reviews/02_midpoint_audit.md` item 1; `phase2_ladder_table.md` row `verifier2_r100` |
| pooled AUROC 0.92 / 0.95 | 0.92131 / 0.95188 | `auroc_*__s0_r16__k64.json` → `scorers.verifier_ens.pooled_all` (B); audit item 1 |
| within 0.646 / 0.596 | 0.64604 / 0.59617 | as above, `within_finished` |
| below own likelihood by 0.012 [0.001, 0.022] / 0.025 [0.010, 0.043] | +0.0116 [0.0011, 0.0220] / +0.0254 [0.0101, 0.0435] | `worker_A_report.md` rows "completion_loglik − verifier_ens"; audit item 0 table |
| input clipping 57–93 % | 77 / 80 / 93 / 81 / 57 % | `worker_A_report.md` (verifier input section) |
| 113,760 attempts; clipping 0.3–2 %; 10,676 finished-invalid | — | `worker_A_phase2.md`; `worker_A_report.md`; `phase2_ladder_table.md` |
| ladder 0.629 → 0.635 → 0.664 → 0.677 | as in table | `phase2_ladder_table.md` holdout250 column (`verifier2_r25`, `_r50`, `_r100`, `_ens`) |
| 0.8k → 1.6k → 3.2k → 10.7k | 799, 1,598, 3,195, 10,676 | `phase2_ladder_table.md` |
| seed spread ±0.02–0.04 | top-rung seeds 0.654–0.683 (holdout250), 0.596–0.662 (textbook72) | `phase2_ladder_table.md` rows `rall_s{0,1,2}`. Half-ranges are 0.015 / 0.033; consistent with the stated range after rounding. |
| paired −0.111 [−0.177, −0.040] | −0.110537 [−0.177231, −0.039889] | `phase2_within_ci.json` → `holdout250."verifier2_ens - loglik_mean_masked"` |
| textbook72 0.638 vs 0.687, −0.049 [−0.196, +0.089] | −0.049012 [−0.195932, 0.088713] | `phase2_within_ci.json` → `textbook72."verifier2_ens - loglik_mean_masked"` |
| flat from step 100 to 1,300 | — | `worker_A_phase2.md` ("plateaus at 0.63–0.68 from step 100 to step 1300") |
| likelihood as input 0.718 / 0.698 | 0.718 / 0.698 | `phase2_ladder_table.md` row `verifier2_ll` |
| −0.070 [−0.133, −0.010]; +0.011 | −0.070341 [−0.133217, −0.010247]; +0.011161 | `phase2_within_ci.json` → `"verifier2_ll - loglik_mean_masked"` |
| phase 3: +0.021 [−0.009, +0.051]; −0.029 [−0.094, +0.045] | +0.020983 [−0.009321, 0.050969]; −0.029484 [−0.093587, 0.044846] | `phase3_matrix.json` → `primary.<set>."P1 sibinit_ens - verifier2_ens (s0 init, 3 seeds)"` |
| vs sibling likelihood −0.131 [−0.184, −0.074]; −0.174 [−0.300, −0.041] | −0.130550 [−0.184016, −0.073680]; −0.173675 [−0.300104, −0.040889] | `phase3_matrix.json` → `primary.<set>."P2 sibinit_ens - loglik_mean_masked_siblings"` |
| heads 0.53–0.75 | min 0.5269, max over single-run cells 0.7513 (the sibinit ensemble reaches 0.7647) | `phase3_matrix.json` → `matrix_heads` |

## Finding 4

| summary value | full precision | source |
|---|---|---|
| first-finished 0.892 at 1.6; 0.527 at 4.0 | 0.891707 at 1.5914; 0.526667 at 3.9624 | `B4_headline_curves.json` → `<set>.ff` (generator means) |
| count + siblings 0.924 at 3.9, 0.934 at 8.1; 0.587 at 7.6 | 0.923707 at 3.9228; 0.934053 at 8.1052; 0.58713 at 7.6032 | `B4_headline_curves.json` → `<set>.count_sib` |
| oracle pass@k 0.927 at 4, 0.943 at 8, 0.607 at 8 | 0.926773; 0.943307; 0.606759 | `B4_headline_curves.json` → `oracle_fixed` |
| oracle stop 0.952 at 2.0; 0.652 at 7.2 | 0.95192 at 1.9837; 0.651667 at 7.185 | `B4_headline_curves.json` → `oracle_stop` |
| self threshold +0.012 [0.004, 0.021] at 4 attempts | +0.011610 [0.003661, 0.020855] at 3.925 attempts | `stopping_headline.json` → `test_results.holdout250__s0_r16__k64.completion_loglik.4.bootstrap.vs_same_at_realised_m` |
| 12 attempts on textbook72 | 12.0767 | `stopping_headline.json` → `test_results.textbook72__s0_r16__k64.completion_loglik.4.test_att` |
| secondary thresholds +0.023 [0.011, 0.038], +0.016 [0.005, 0.031] | +0.023041 [0.010712, 0.038443]; +0.016117 [0.005432, 0.031135] (both at the m = 2 dev point) | `stopping_headline.json` → `textbook72…{completion_minstep,completion_loglik_sum}.2.bootstrap.vs_same_at_realised_m` |
| difficulty-aware: five more attempts on textbook72 | Δatt +4.98 (da vs count, m = 4) | `B2_headline.json` → `test.textbook72__s0_r16__k64__phase2.npz.4.da.vs_count.att_diff` |
| phase 5: +0.006 to +0.010 at 2.3–3.9 attempts | +0.0062 to +0.0104 at 2.29–3.85 | `B5_matched.json` → `holdout250\|{T,C}_sib\|m{3,4}` |
| +0.003 [−0.000, +0.005] at 8.2 | +0.00268 [−0.00009, 0.00537] at 8.22 | `B5_matched.json` → `holdout250\|T_sib\|m8` |
| textbook72 +0.011 to +0.016 at 8–10 attempts | +0.0109 (T_sib m4, 9.83), +0.0116 (C_sib m4, 8.44), +0.0155 (T_sib m3, 8.26) | `B5_matched.json` |
| 18–23 % low-budget; 14 % at 8 | see **F1**; 14 % = 0.1401 (`holdout250\|T_sib\|m8`) | `B5_matched.json` |
| budget 2.1–2.5× vs 3× vs 1.5× | 8.438/4 = 2.11, 9.833/4 = 2.46; 12.077/4 = 3.02; 6.079/4 = 1.52 | `B5_results.json` → `textbook72\|{C,T}_sib\|m4.att`, `.count_att`; `stopping_headline.json` |

## Limitations

| summary value | full precision | source |
|---|---|---|
| draw-to-draw 0.001–0.004 / 0.001–0.028 | as stated | `B6_draw_compare.json` |
| half-widths about 0.03 / 0.10 | 0.026–0.034 / 0.093–0.106 (accuracy-type rows) | `B6_draw_compare.json` → `halfwidth_draw1` |
| AUROC moves up to 0.08 (0.687 → 0.764) | 0.687197 → 0.763957 | `B6_draw_compare.json` → `textbook72.auc_self` |
| T 0.8: +0.105 [0.027, 0.186] vs +0.095 | as stated | `B5_t08_check.txt` |
| 11 of 12 single-judge comparisons positive | — | `B3_results.json` → `<set>.penalty.auc` |
| Worker B read test at 02:38 | see **F3** | `../logs/worker_B.md` |
