# numbers.md — every number in writeup.md and where it comes from

## Data / generator
| number | value | source |
|---|---|---|
| cap-6 raw proofs generated / verifier rejects | 160,000 / 0 | `gen.py` stdout (log.md 03:30); `VERIFIER_REJECT` counter absent from stats |
| long raw proofs (strict) / rejects | 7,000 / 0 | same, `data/raw_long.jsonl` run |
| train / heldout / rl_targets / transfer sizes | 154,990 / 5,000 / 3,000 / 1,638 | `data/splits_stats.json` |
| per-length counts | train ~31k each 2–6; heldout 1,000 each; targets 300 each 7–16; transfer 125–200 each | `data/splits_stats.json` |
| renaming-only duplicates inside generator run | 78,931 (≈33% of distinct strings) | `gen.py` stats `dup_renaming_only` (log.md 03:30) |
| validation-class hits removed | 10 (cap-6), 6 (long) | `data/splits_stats.json` |
| trivial-pattern audit | 1,382/160k (0.9%) | `data/splits_stats.json` `cap6_trivial_pattern` |
| contradictory premise pairs | 9,803/160k cap-6 (6.1%); 1,896/7,000 long removed | `data/splits_stats.json` |
| parameters | 3,210,240 | `train.py` stdout (`artifacts/train_stage1_abs.log`) |
| Stage-1 training wall-clock | 722 s (rel), ~15 min (abs), sharing the GPU | `artifacts/train_stage1_*.log` |

## Stage 1 (greedy, held-out n=5000)
| number | value | source |
|---|---|---|
| rel by length 2..6 | 99.8 / 99.1 / 97.1 / 92.9 / 88.6 %, all 95.5% [94.9,96.0] | `artifacts/stage1_rel_heldout_greedy.json` |
| abs by length 2..6 | 99.8 / 98.6 / 96.2 / 92.0 / 87.3 %, all 94.8% [94.1,95.4] | `artifacts/stage1_abs_heldout_greedy.json` |
| abs failure reasons | ANDI 71, IMPE 45, ORI2 25, NEGE 24, IMPI 17, wrong conclusion 15 | same file, `reasons` |
| abs failure rate by rule / n_prem | ANDE2 24%, ORE 20%, ANDE1 16%; 3 premises 16% | computed from `artifacts/stage1_abs_heldout_greedy.jsonl` + `data/heldout.jsonl` (log.md) |

## Frozen Stage-1 beyond the cap (transfer v2, n=1638)
| number | value | source |
|---|---|---|
| greedy rel / abs | 26.4% [24.3,28.6] / 32.2% [30.0,34.5] | `artifacts/stage1_{rel,abs}_transfer2_greedy.json` |
| pass@16 rel / abs | 40.5% [38.2,42.9] / 44.7% [42.3,47.1] | `artifacts/stage1_{rel,abs}_transfer2_k16.json` |
| distinct len-7 / len-8 proofs rel / abs (start-index normalised; raw abs count was 501/1) | 79/0 vs 108/1 | `artifacts/stage1_{rel,abs}_transfer2_k16_norm.json` |
| transfer v1 (n=1600) pass@16 rel / abs | 33.2% / 37.8% | `artifacts/stage1_{rel,abs}_transfer_k16.json` |
| validation-36 greedy rel / abs | 10/36 / 7/36; >6 bin 0/24 both | `artifacts/stage1_{rel,abs}_val36_greedy.txt` |

## Stage 2, seed 0 (`artifacts/ei_abs_s0/round_<r>.json`, `artifacts/frozen_abs_s0/round_<r>.json`; tables in `artifacts/tables_s0.md`)
| number | value | source |
|---|---|---|
| transfer cumulative solved, round 8 (256 attempts) EI / frozen | 82.8% [80.9,84.6] / 55.3% [52.9,57.7] (n=1638) | round_8.json `transfer_cum` |
| transfer per-round pass@32, round 8 EI / frozen | 77.3% / 48.6% | round_8.json `transfer_round` |
| transfer greedy, round 8 EI / frozen | 64.2% [61.9,66.5] / 32.2% [30.0,34.5] | round_8.json `transfer_greedy` |
| RL targets cumulative, round 8 EI / frozen | 81.4% / 54.0% (n=3000) | round_8.json `targets_cum` |
| held-out greedy round 8 EI / frozen; EI minimum (round 4) | 94.9% / 94.8%; 93.1% | round_r.json `heldout_greedy` |
| robust frontier transfer written/pruned EI / frozen (start-index normalised) | 9/9 / 8/8 | `artifacts/ei_abs_s0_norm/round_8.json`, `artifacts/frozen_abs_s0_norm/round_8.json` `transfer_cum.frontier_*` |
| robust frontier targets written/pruned EI / frozen (normalised) | 9/9 at r8 (4 ten-line proofs), 10/10 from r15 / 8/8 | `artifacts/ei_abs_s0_norm/round_8.json`, `artifacts/ei_abs_s0_cont_norm/round_{15,16}.json` `targets_cum.frontier_*` |
| distinct transfer proofs written ≥8 / ≥9, EI / frozen (normalised; raw 2028 / 195 vs 54 / 0) | 193 / 20 vs 13 / 0 | `artifacts/*_norm/round_8.json` `transfer_cum.ge`; `artifacts/normalized_summary.md` |
| per-round table (rounds 1-8) | see writeup table | `artifacts/tables_s0.md` |
| padding: written>pruned fraction, mean excess (round 1 transfer, normalised; raw per-sample 41.7%, 1.07) | 51%, 0.58 lines | `artifacts/ei_abs_s0_norm/round_1.json` `transfer_cum.pad_frac/pad_gap` |
| transfer breakdown by depth / premises / ORE (EI r5 vs frozen r4) | see writeup table | `analyze_transfer.py` output, log.md 04:47 |
| greedy failure reasons round 8 EI / frozen | listed in writeup | round_8.json `transfer_greedy.reasons` |
| validation-36 per round greedy / pass@32 | r1 8/10, r2 9/11, r3 9/10, r4 7/11, r5 8/10, r6 10/11, r7 10/11, r8 9/11 | `artifacts/ei_abs_s0_r<r>_val36_greedy.txt`, `_val36_k32.json` |
| contraposition proof | 7 lines, round 7 greedy | `artifacts/ei_abs_s0_r7_val36_greedy.jsonl` |
| abs-fixed ablation held-out / transfer greedy / pass@16 / proofs ≥7 | 95.4% / 27.2% / 38.5% [36.2,40.9] / 0 of 26,208 samples | `artifacts/stage1_absfixed_{heldout,transfer2}_greedy.json`, `_transfer2_k16.json` |

## Stage 2, all arms (final rounds) — `artifacts/<arm>/round_<r>.json`; per-round tables `artifacts/tables_all.md`; merged 16-round dirs `artifacts/{ei,frozen}_abs_s0_all/`
| number | value | source |
|---|---|---|
| EI seed 1 r8: transfer cum / round / greedy / held-out | 82.6% [80.7,84.4] / 77.6% / 64.6% [62.2,66.9] / 94.9% | `artifacts/ei_abs_s1/round_8.json` |
| frozen seed 1 r8 | 55.5% [53.1,57.9] / 48.0% / 32.2% / 94.7% | `artifacts/frozen_abs_s1/round_8.json` |
| EI-long r8 | 82.1% [80.1,83.8] / 75.3% / 62.2% [59.8,64.5] / 95.1%; written ≥8 208, ≥9 21 (normalised; raw 2338 / 120) | `artifacts/ei_abs_long_s0_norm/round_8.json` |
| EI s0 r16 (512 attempts) | 86.1% [84.4,87.7] / 79.4% / 66.8% [64.5,69.0] / 95.3%; written ≥8 271, ≥9 29 (26 theorems), ≥10 0; targets ≥10: 13 (9 theorems), ≥11: 0 (normalised; raw 3628 / 397 / 0; 151) | `artifacts/ei_abs_s0_cont_norm/round_16.json` |
| frozen s0 r16 | 57.3% [54.9,59.7] / 48.4% / 32.2% / 94.7%; written ≥8 17, ≥9 0 (normalised; raw 92 / 0) | `artifacts/frozen_abs_s0_cont_norm/round_16.json` |
| seed-1 written ≥8 / ≥9 EI vs frozen (normalised; raw 2204 / 176 vs 54 / 0) | 205 / 22 vs 14 / 0 | `artifacts/ei_abs_s1_norm/round_8.json`, `artifacts/frozen_abs_s1_norm/round_8.json` |
| round time | 610–670 s with 2–3 jobs on the A40; 930–980 s with 5 jobs | `=== round` lines in `artifacts/*.log` |

## Test set (run once, 07:38 UTC) — `artifacts/test_scores.txt`, `artifacts/TEST_RUN_DONE`
| number | value | source |
|---|---|---|
| Stage 1 short / long | 73.0% (195/267; CI 67.4–78.0) / 9.8% (52/532; CI 7.5–12.6) | `artifacts/test_scores.txt` |
| final (EI s0 r16) short / long | 73.0% (195/267; CI 67.4–78.0) / 14.5% (77/532; CI 11.7–17.7) | `artifacts/test_scores.txt` |
| validation-36 final greedy / pass@32 | 9/36 / 11/36 | `artifacts/ei_abs_s0_cont_r16_val36_greedy.txt`, `_k32.json` |
| padding by round, cumulative distinct transfer proofs (padded fraction / mean written−pruned), normalised | r1 0.51/0.58; r8 0.61/0.85; r16 0.64/0.96; frozen r16 0.57/0.72 (superseded per-sample values: r1 0.42/0.45, r8 0.56/0.69, r16 0.61/0.84; frozen 0.47/0.53) | `artifacts/*_norm/round_<r>.json` `transfer_cum.pad_frac/pad_gap` |
| transfer breakdown by generating length / depth / premises (r8, r16) | table | `artifacts/transfer_breakdown.md` |
| pruned ≥9-line transfer proofs, final (normalised; raw 272 / 397) | 17 (written 29) | `artifacts/ei_abs_s0_cont_norm/round_16.json` `transfer_cum.ge_pruned` |
| validation >6 greedy attempt lengths, final | 2–7, median 5, 21/24 ≤6 | `artifacts/ei_abs_s0_cont_r16_val36_greedy.jsonl` + verifier |
| EI-long validation-36 r8 greedy / pass@32 | 8/36 / 9/36 | `artifacts/ei_abs_long_s0_r8_val36_*` |
| pass@128 one shot (seed 7), final / Stage 1 | 82.6% [80.7,84.4] / 52.9% [50.4,55.3]; written 8/9/10 (normalised): 188/33/2 vs 11/0/0 (raw 2215/335/2 vs 32/0/0); frontier 9 / 8 | `artifacts/final_transfer_k128_norm.json`, `artifacts/stage1_transfer_k128_norm.json` |

## Post-run review corrections (2026-09-15) — `normalize.py`, `audit_test_overlap.py`
| number | value | source |
|---|---|---|
| share of "distinct" proofs that were start-index variants of another proof | ~96% (e.g. EI s0 r16 transfer: 75,085 raw → 3,330 normalised; frozen: 47,393 → 1,724) | `artifacts/normalized_summary.md` |
| theorems with a written 8 / 9 / 10-line proof, EI s0 r16 transfer / targets | 180 / 26 / 0 ; 246 / 46 / 9 | `artifacts/ei_abs_s0_cont_norm/round_16.json` `*_cum.theorems_at_written` |
| theorems with a written 8 / 9-line proof, frozen s0 r16 transfer / targets | 14 / 0 ; 31 / 0 | `artifacts/frozen_abs_s0_cont_norm/round_16.json` |
| Stage-1 pass@16 length-7 / 8 proofs, abs / rel / abs-fixed (normalised; raw abs 501 / 1) | 108 / 1 ; 79 / 0 ; 0 / 0 | `artifacts/stage1_{abs,rel,absfixed}_transfer2_k16_norm.json` |
| test-short prompts whose renaming class is in Stage-1 train / exact string | 119 of 267 (45%) / 9 | `artifacts/test_overlap.json` |
| test-long prompts whose renaming class is in Stage-1 train / exact string | 6 of 532 (1%) / 1 | `artifacts/test_overlap.json` |
| validation-36 overlap with any pool | 0 | `artifacts/test_overlap.json` |

## Novelty campaign (branch dan_novelty, 2026-09-15/18) — Phase 1
| number | value | source |
|---|---|---|
| normalised-distinct proofs scored: final transfer / final targets / final val36 / EI-any-round val36 / frozen transfer / frozen targets | 3,330 (1,411 thms) / 4,506 (2,570) / 14 (11) / 26 (13) / 1,724 / 3,149 | `artifacts/novelty_phase1_proofs.jsonl` (`novelty.py`) |
| final transfer proofs with base p(T=0.8) < 1e-5, by written length 7 / 8 / 9 | 328/784, 212/242, 29/29 | `artifacts/phase1_tables.md` (reachability tables) |
| final transfer theorems (shortest found proof 8 / 9 lines) below the 1e-5 line by their found proofs | 83/87, 14/14 | same, per-theorem table |
| frozen transfer proofs (found in 512 attempts) with base p >= 1/512 | 1,463/1,724; < 1e-5: 7 | same, calibration table |
| contraposition: predicted base p (T=0.8) / measured at k=1e5 | 1.8e-4 (log -8.64) / 14 hits, first at sample 5,681 | `novelty_phase1_theorems.jsonl`, `artifacts/cov_base_val36_k1e5.s0.jsonl` |
| export: predicted base p / measured at k=1e5 | 6e-8 (log -16.6) / 0 hits | same |
| validation-36 base model k=1e5: solved / >6 bin solved | 14/36 / 1/24 (contraposition; explosion 57 hits, 5-line proofs) | `artifacts/cov_base_val36_k1e5.s0.jsonl` |
| minlen (bound 8): val36 labelled / agree with reference bound | 20/36 / 20/20 | `artifacts/minlen_val36.jsonl` |
| minlen: transfer labelled / with <=6-line proof; RL targets labelled / <=6 | 1,573/1,638 / 1,149 (70%); 2,860/3,000 / 2,071 (69%) | `artifacts/minlen_transfer.jsonl`, `artifacts/minlen_rl_targets.jsonl` |
| pattern rates in take-home cap-6 pool (n=160k): derived_ore / reductio / depth3 | 91 (0.057%) / 10,887 (6.8%) / 5,870 (3.7%); len-5/6 only | `patterns.py --stats data/raw_cap6.jsonl` (log.md 19:40) |
| pattern rates in take-home long pool (n=7,000) | 35.8% / 14.1% / 39.4% | `patterns.py --stats data/raw_long.jsonl` |

## Novelty campaign — Phase 2 pools (2026-09-15 20:05 UTC)
| number | value | source |
|---|---|---|
| cap-6 raw pool: distinct classes (len 2/3/4/5/6) | 820,125 (75,526 / 91,482 / 113,123 / 161,186 / 378,808); class duplicates dropped 848,626 | p1 `artifacts/…` merge stdout (log.md 20:05) |
| cap-6 pool pattern classes: derived_ore / reductio / depth3 | 3,531 / 158,590 / 193,577 | same |
| natural per-length pattern rates used as the held-fixed baseline (from the take-home's unfiltered pool) | reductio 17.34% (len 5) / 16.68% (len 6); depth3 18.34% (len 6); derived_ore 0.088% / 0.197% | `make_coverage_sets.py assemble` NATURAL |
| long strict pool (7-12 lines): classes / with a <=6-line proof (minlen) / without | 89,533 / 69,756 / 19,777 | `data/p2/pool_long_minlen.jsonl` |
| P1 f_max | 0.02 (3,100 proofs of 3,531 available) | assemble_report.json |
| achieved frequencies per set (derived_ore / reductio / depth3) | | |
| derived_ore_f0 | 0.00000 / 0.06805 / 0.03669 | `data/p2/assemble_report.json` |
| derived_ore_f0.0001 | 0.00010 / 0.06805 / 0.03669 | `data/p2/assemble_report.json` |
| derived_ore_f0.001 | 0.00100 / 0.06805 / 0.03669 | `data/p2/assemble_report.json` |
| derived_ore_f0.01 | 0.01000 / 0.06805 / 0.03669 | `data/p2/assemble_report.json` |
| derived_ore_f0.02 | 0.02000 / 0.06805 / 0.03669 | `data/p2/assemble_report.json` |
| targets_derived_ore: n / gen-length range / min_lines_ub 7,8,None / other patterns | 1000 / 10-9 / 654,219,127 / {'derived_ore': 1000, 'reductio': 43, 'depth3': 640} | `data/p2/targets_summary.json` |
| transfer_derived_ore: n / gen-length range / min_lines_ub 7,8,None / other patterns | 500 / 11-12 / 403,65,32 / {'derived_ore': 500, 'reductio': 31, 'depth3': 357} | `data/p2/targets_summary.json` |
| targets_reductio: n / gen-length range / min_lines_ub 7,8,None / other patterns | 972 / 10-9 / 452,286,234 / {'derived_ore': 9, 'reductio': 972, 'depth3': 402} | `data/p2/targets_summary.json` |
| transfer_reductio: n / gen-length range / min_lines_ub 7,8,None / other patterns | 332 / 10-9 / 174,86,72 / {'derived_ore': 1, 'reductio': 332, 'depth3': 137} | `data/p2/targets_summary.json` |
| targets_depth3: n / gen-length range / min_lines_ub 7,8,None / other patterns | 1000 / 10-9 / 575,240,185 / {'derived_ore': 0, 'reductio': 0, 'depth3': 1000} | `data/p2/targets_summary.json` |
| transfer_depth3: n / gen-length range / min_lines_ub 7,8,None / other patterns | 500 / 10-9 / 296,110,94 / {'derived_ore': 13, 'reductio': 0, 'depth3': 500} | `data/p2/targets_summary.json` |
| targets_none: n / gen-length range / min_lines_ub 7,8,None / other patterns | 1000 / 10-9 / 608,252,140 / {'derived_ore': 0, 'reductio': 0, 'depth3': 0} | `data/p2/targets_summary.json` |
| transfer_none: n / gen-length range / min_lines_ub 7,8,None / other patterns | 500 / 10-9 / 306,124,70 / {'derived_ore': 0, 'reductio': 0, 'depth3': 0} | `data/p2/targets_summary.json` |

## Novelty campaign — Phase 2 row 1 (arms as they finish; all normalised-distinct; `phase2_metrics.py`, `artifacts/p2/metrics_*.json`)
| arm (8 rounds, k=32) | targets solved / n | acquisition (thms with a pattern proof) / pattern proofs / first round | transfer solved / acq | held-out greedy | source |
|---|---|---|---|---|---|
| ei_reductio_f0_s0 | 609/972 | 0 / 0 / – | 222/332 / 0 | 0.886 | `artifacts/p2/ei_reductio_f0_s0/` |
| ei_reductio_f0_s1 | 170/972 | 0 / 0 / – | 68/332 / 0 | 0.885 | `artifacts/p2/ei_reductio_f0_s1/` |
| ei_reductio_f0.1_s0 | 193/972 | 13 (0.013) / 13 / 1 | 72/332 / 0 | 0.970 | `artifacts/p2/ei_reductio_f0.1_s0/` |
| ei_depth3_f0.1_s0 | 594/1000 | 355 (0.355) / 424 / 2 | 310/500 / 0.382 | 0.964 | `artifacts/p2/ei_depth3_f0.1_s0/` |
| ei_depth3_f0.1_s1 | 607/1000 | 316 (0.316) / 388 / 1 | 320/500 / 0.348 | 0.962 | `artifacts/p2/ei_depth3_f0.1_s1/` |
| frozen_reductio_f0_s0 (control, 256 samples/target) | 96/972 | 0 / 0 / – | 39/332 / 0 | 0.886 | `artifacts/p2/frozen_reductio_f0_s0/` |
| frozen_reductio_f0.1_s0 (control) | 74/972 | 11 (0.011) / 11 / 1 | 24/332 / 0 | 0.968 | `artifacts/p2/frozen_reductio_f0.1_s0/` |
| ei_derived_ore_f0_s0 | 364/1000 | 9 (0.009) / 9 / 3 | 328/500 / 0.002 | 0.973 | `artifacts/p2/ei_derived_ore_f0_s0/` |
| ei_derived_ore_f0_s1 | 330/1000 | 2 (0.002) / 2 / 7 | 311/500 / 0.004 | 0.970 | `artifacts/p2/ei_derived_ore_f0_s1/` |
| ei_derived_ore_f0.02_s0 | 572/1000 | 242 (0.242) / 294 / 1 | 329/500 / 0.022 | 0.965 | `artifacts/p2/ei_derived_ore_f0.02_s0/` |
| frozen_depth3_f0.1_s0 (control) | 127/1000 | 12 (0.012) / 17 / 1 | 66/500 / 0.012 | 0.966 | `artifacts/p2/frozen_depth3_f0.1_s0/` |
| ei_reductio_f0.001_s0 (row 2) | 212/972 | 13 (0.013) / 13 / 4 | 82/332 / 0 | 0.953 | `artifacts/p2/ei_reductio_f0.001_s0/` |
| ei_reductio_f0.001_s1 (row 2) | 374/972 | 0 / 0 / – | 156/332 / 0 | 0.948 | `artifacts/p2/ei_reductio_f0.001_s1/` |
| ei_derived_ore_f0.02_s1 | 660/1000 | 293 (0.293) / 305 / 1 | 359/500 / 0.042 | 0.974 | `artifacts/p2/ei_derived_ore_f0.02_s1/` |
| frozen_derived_ore_f0_s0 (control) | 180/1000 | 1 (0.001) / 1 / 1 | 193/500 / 0 | 0.971 | `artifacts/p2/frozen_derived_ore_f0_s0/` |
| frozen_derived_ore_f0.02_s0 (control) | 136/1000 | 10 (0.010) / 11 / 1 | 134/500 / 0.002 | 0.967 | `artifacts/p2/frozen_derived_ore_f0.02_s0/` |
| derived_ore f=0 arms: strict derived-ORE (rule-derived disjunction, distinct disjuncts) theorems | 0 (all 11 EI proofs and the 1 frozen proof are the ( X v X ) one-line-box form) | `phase2_metrics.py` acq_strict |
| ei_depth3_f0_s0 | 628/1000 | 341 (0.341) / 400 / 3 | 323/500 / 0.366 | 0.940 | `artifacts/p2/ei_depth3_f0_s0/` |
| ei_depth3_f0_s1 | 587/1000 | 271 (0.271) / 327 / 5 | 296/500 / 0.280 | 0.911 | `artifacts/p2/ei_depth3_f0_s1/` |
| frozen_depth3_f0_s0 (control) | 55/1000 | 0 / 0 / – | 30/500 / 0 | 0.874 | `artifacts/p2/frozen_depth3_f0_s0/` |
| ei_depth3_f0.001_s0 (row 2) | 631/1000 | 364 (0.364) / 443 / 1 | 324/500 / 0.384 | 0.964 | `artifacts/p2/ei_depth3_f0.001_s0/` |
| depth3 f=0 s0: depth-3 proofs found by EI / with base p(T=0.8, own Stage-1 model) < 1/256 / < 1e-5 / most probable | 610 / 610 / 610 / log p = −21.8 (round 4) | `artifacts/p2/novelty_depth3_f0_s0_proofs.jsonl` |
| depth3 f=0 s1: same | 491 / 491 / 485 / log p = −7.1 (round 6) | `artifacts/p2/novelty_depth3_f0_s1_proofs.jsonl` |
| depth3 f=0: max-surprisal token of the depth-3 proofs is the third `\|` | s0 608/610, s1 475/491 | same |
| ei_derived_ore_f0.001_s0 (row 2) | 583/1000 | 248 (0.248) / 281 / 1 | 327/500 / 0.028 | 0.968 | `artifacts/p2/ei_derived_ore_f0.001_s0/` |
| ei_depth3_f0.001_s1 (row 2) | 660/1000 | 339 (0.339) / 429 / 2 | 333/500 / 0.366 | 0.952 | `artifacts/p2/ei_depth3_f0.001_s1/` |
| ei_derived_ore_f0.001_s1 (row 2) | 583/1000 | 245 (0.245) / 268 / 2 | 330/500 / 0.022 | 0.972 | `artifacts/p2/ei_derived_ore_f0.001_s1/` |
| ei_reductio_f0.01_s0 (row 3) | 494/972 | 11 (0.011) / 11 / 4 | 190/332 / 0 | 0.966 | `artifacts/p2/ei_reductio_f0.01_s0/` |
| ei_reductio_f0.0001_s0 (row 3) | 203/972 | 3 (0.003) / 3 / – | – | – | `artifacts/p2/ei_reductio_f0.0001_s0/`, `artifacts/p2/metrics_reductio.json` |
| main figure (all arms finished so far) | figures/phase2_acquisition.png | `phase2_figure.py` |
| base derived_ore_f0_s0 pass@1e4 (284 P1 targets): solved / with derived-ORE proof / strict | 3 / 2 / 0 | `artifacts/p2/cov_derived_ore_f0_s0_targets.s0.jsonl` |
| Phase 3 frozen_textbook (256 samples/target): solved by minlen length 2/3/4/5/6/7/8/unlabelled | 47/47, 2/2, 48/48, 5/9, 55/82, 4/118, 0/32, 0/285 (161/623) | `artifacts/p3/frozen_textbook/round_8.json` |
| base reductio_f0_s0 pass@1e4 (300 P2 targets): solved / within 512 / with a reductio proof | 82 / 51 / 0 | `artifacts/p2/cov_reductio_f0_s0_targets.s0.jsonl` |
| base derived_ore_f0_s0 pass@1e4 (300 P1 targets, final): solved / with derived-ORE proof / strict | 3 / 2 / 0 | `artifacts/p2/cov_derived_ore_f0_s0_targets.s0.jsonl` |
| ei_depth3_f0.0001_s0 (row 3; resumed from r5 after an OOM) | 605/1000 | 354 (0.354) / 433 / 2 | 314/500 / 0.368 | 0.959 | `artifacts/p2/ei_depth3_f0.0001_s0/` |
| targets provably needing the pattern (bounded, <=8-line restricted search): depth3 / derived_ore / reductio | 142 of 815 labelled (max box depth 2) / 372 of 873 (ORE over premises only) / 56 of 972 (classical-only) | `data/p2/targets_depth3_maxdepth2.jsonl`, `data/p2/targets_derived_ore_noderived.jsonl`, `data/p2/targets_reductio_intuit.jsonl` |
| ei_depth3_f0.01_s0 (row 3) | 663/1000 | 363 (0.363) / 421 / 2 | 343/500 / 0.402 | 0.969 | `artifacts/p2/ei_depth3_f0.01_s0/` |
| ei_derived_ore_f0.0001_s0 (row 3) | 555/1000 | 222 (0.222) / 275 / 2 | 323/500 / 0.022 | 0.967 | `artifacts/p2/ei_derived_ore_f0.0001_s0/` |
| ei_derived_ore_f0.01_s0 (row 3) | 610/1000 | 272 (0.272) / 371 / 1 | 340/500 / 0.040 | 0.970 | `artifacts/p2/ei_derived_ore_f0.01_s0/` |
| Phase 3 EI arms, targets solved r8: ei_textbook / ei_textbook_precursor / frozen | see phase3.md | `artifacts/p3/*/round_8.json` |

## Novelty campaign — Phase 3 (textbook-shaped curriculum; `artifacts/p3/`)
| number | value | source |
|---|---|---|
| textbook pool: theorems / schemata / with a <=6-line proof (minlen) / min_lines_ub 7 / 8 / none <=8 | 623 / 26 / 188 / 118 / 32 / 285 | `data/p3/textbook_targets.jsonl` |
| precursor set (generator proofs matching a sub-step template of a schema): records / by length 2-6 / schemata covered | 9,455 / 1,326, 1,666, 2,100, 2,757, 1,606 / 23 | `data/p3/precursors.jsonl` (`precursors.py`) |
| targets solved per round, frozen / ei_textbook / ei_textbook_precursor | 153..161 / 153,180,193,200,202,203,207,213 / 153,186,198,207,216,222,222,222 | `artifacts/p3/*/round_r.json` |
| schemata moved by EI (r8, /24 each): contraposition / consequentia_mirabilis / export / hypothetical_syllogism | frozen 3, 9, 0, 22; ei 24, 24, 12, 24; ei+precursor 24, 24, 19, 24 | `artifacts/p3/*/found_8.jsonl` by schema |
| schemata at 0-2/24 for every arm | De Morgan x4, distribution x4, constructive_dilemma, disjunctive_syllogism, negated_conditional(_conv), peirce, peirce_sequent, contraposition_conv, import (2-4), excluded_middle (1) | same |
| validation-36 per round (greedy / pass@32, >6 bin): ei_textbook | >6 greedy 1/24 from r2 (contraposition), pass@32 1/24 every round; <=6 pass@32 10-11/12 | `artifacts/p3/ei_textbook_r*_val36_*` |
| validation-36 per round: ei_textbook_precursor | >6 pass@32 2/24 from r4 (contraposition, export); greedy >6 2/24 at r6; <=6 pass@32 10-11/12 | `artifacts/p3/ei_textbook_precursor_r*_val36_*` |
| validation theorems newly solved vs Stage-1 pass@32 (any round): ei_textbook / ei+precursor | consequentia_mirabilis, contraposition, explosion / consequentia_mirabilis, contraposition, export | same |
| transfer greedy r8 / held-out greedy r8: frozen, ei, ei+precursor | 0.322 / 0.947, 0.324 / 0.954, 0.338 / 0.952 | round_8.json |
| base depth3_f0_s0 pass@1e4 (first 180 of 300 P3 targets, running): solved / with a depth-3 proof | 1 / 0 | `artifacts/p2/cov_depth3_f0_s0_targets.s0.jsonl` |
| base depth3_f0_s0 pass@1e4 (300 P3 targets, final): solved / within 512 / with a depth-3 proof | 3 / 1 / 0 | `artifacts/p2/cov_depth3_f0_s0_targets.s0.jsonl` |

## Follow-up run (2026-09-16/17) — block A: depth-3 replication sets (`data/p2/assemble_report_{a1,a2,a3,b1,b2}.json`, `artifacts/fu/indep_check.log`)
- Pool: `data/p2/pool_cap6_recon.jsonl` = union of the 12 local raw shards + the 15 campaign-1 training sets + held-out, dedup by class, val-36 dropped: 723,534 classes (len 2/3/4/5/6: 75,365 / 91,375 / 112,399 / 149,025 / 295,370; depth-3 132,454; reductio 125,296; derived-ORE 3,744). Subset of the campaign-1 pool (820,125).
- New sets (155,000 each, 31,000 per length, campaign-1 held-out reused and excluded, depth-3 target/transfer classes excluded; assembler seeds 1 / 2 / 3 / 11 / 12): depth3_f0_a1, _a2, _a3: depth-3 **0 / 0 / 0** (pruned and written form; independent written-form counter 0 / 0 / 0), reductio 10,547 (0.0681), derived-ORE 88 (0.00057), strict derived-ORE 2 / 1 / 2. depth3_f0.1_b1, _b2: depth-3 15,500 (0.1000; independent count 15,500 / 15,500), reductio 10,547, derived-ORE 88. Class overlap with the campaign-1 f = 0 set: 47,352 / 47,580 / 47,269 (a1–a3), 42,968 / 42,595 (b1–b2) of 155,000.

## Follow-up — block B pool (`reductio_pool.py`, `data/p2/targets_reductio2.jsonl`, `transfer_reductio2.jsonl`, `reductio2_cands_minlen.jsonl`, `artifacts/fu_blockB_pool.log`, `fu_blockB_pool2.log`)
- Generator-native attempt: 20M tries (long mode, 7–12 lines, filter reductio ∧ no `( ~ ( ~`): 23,523 raw → 19,099 classes (len 7/8/9/10/11/12: 8,072 / 996 / 1,477 / 1,284 / 2,630 / 4,640); classical-only (`intuit.py`): **7** (0.04%); minlen (bound 8): 6 labelled {6: 1, 7: 2, 8: 3}, 1 unlabelled → 6 usable (min_lines_ub ≥ 7 or None).
- Schema pool: 21 classical-only schemata without double negation × 45 instances = 945 candidates (all truth-table valid and intuitionistically unprovable, asserted per instance; class-disjoint from the reconstructed cap-6 pool, held-out, old reductio pools, validation-36); minlen bound 8 / 10 s: labelled 365 {6: 45, 7: 90, 8: 230}, 580 unlabelled → 900 with min_lines_ub ≥ 7 or None.
- **targets_reductio2: 606** (600 schema = 30 per schema + 6 generator-native), min_lines_ub 7 / 8 / None = 62 / 158 / 386; **transfer_reductio2: 300** (15 per schema), 30 / 75 / 195. Hand check of ten printed targets (log.md 02:05): every one needs the negated goal assumed (excluded middle, Peirce, `~(X>Y) |- X`, case split on an atom via `A>B, ~A>B |- B`, …).

## Follow-up — block A arms as they finish (`followup_analysis.py` → `artifacts/fu/blockA_summary.json`; per-arm dirs `artifacts/p2/ei_depth3_*`; acquisition = target theorems solved with a written proof whose pruned form has box depth ≥ 3, normalised-distinct)
- depth3_f0_a1 s0: 645/1000 solved, **acq 0.335** (335 theorems / 409 depth-3 proofs / first round 2); transfer 0.372; held-out greedy 0.907.
- depth3_f0_a1 s1: 698/1000, **acq 0.364** (364 / 426 / r1); transfer 0.366; held-out greedy 0.930.

## Follow-up — block B arms (`phase2_metrics.py --pattern reductio`, `artifacts/fu/metrics_reductio2*.json`; targets_reductio2 n = 606, transfer 300; strict = `patterns.reductio`, loose = `patterns.derived_dn`, both on the model's normalised proof)
- reductio_f0 s0 (_t2): solved 58/606; **strict 58 (0.096), loose 58, any-DN 58**; 60 distinct proofs; first strict proof round 2 (1 theorem), then 11 / 24 / 36 / 55 / 58 / 58 at rounds 3–8; transfer 30/300 solved, strict 30 (0.100). Solved by schema: nand_neg 29/30, negimp_to_pos 27/30, generator-native 2/6, every other schema 0. Held-out greedy 0.893, transfer greedy 0.097.
- reductio_f0 s1 (_t2): solved **0/606** (8 rounds × 32 = 256 attempts per target), transfer 0/300; held-out greedy 0.871.
- depth3_f0_a2 s0: 589/1000 solved, **acq 0.350** (350 / 430 depth-3 proofs / first round 2); transfer acq 0.368; held-out greedy 0.945.
- depth3_f0_a2 s1: 652/1000 solved, **acq 0.341** (341 / 426 depth-3 proofs / first round 4); transfer acq 0.358; held-out greedy 0.903.
- depth3_f0_a3 s0: 605/1000 solved, **acq 0.361** (361 / 435 depth-3 proofs / first round 2); transfer acq 0.386; held-out greedy 0.945.
- depth3_f0.1_b2 s0: 604/1000 solved, **acq 0.352** (352 / 410 depth-3 proofs / first round 3); transfer acq 0.388; held-out greedy 0.965.
- depth3_f0.1_b2 s1: 596/1000 solved, **acq 0.347** (347 / 428 depth-3 proofs / first round 2); transfer acq 0.380; held-out greedy 0.971.
- depth3_f0_a3 s1: 583/1000 solved, **acq 0.352** (352 / 425 / first round 1); transfer acq 0.378; held-out greedy 0.919.
- **f = 0, all 8 arms** (4 sets × 2 seeds): 0.341, 0.271 (campaign-1 set a0), 0.335, 0.364 (a1), 0.350, 0.341 (a2), 0.361, 0.352 (a3): mean 0.339, SD 0.029, min 0.271, max 0.364.
- reductio_f0.1 s0 (_t2): solved 95/606, **strict 95 (0.157), loose 95, any-DN 95**, first round 1 (15 theorems), per round 15 / 46 / 61 / 64 / 71 / 77 / 81 / 95; transfer 40/300 solved, strict 40 (0.133); held-out greedy 0.970. Solved by schema: negimp_to_pos 30/30, nand_neg 30/30, neg_both 22/30, chain_neg 11/30, generator-native 2/6; the 16 longer schemata 0.
- reductio_f0.1 s1 (_t2): solved 63/606, **strict 63 (0.104), loose 63**, first round 1 (17), per round 17 / 48 / 61 / 62 / 62 / 63 / 63 / 63; transfer 30/300 (strict 30); held-out greedy 0.973. By schema: negimp_to_pos 30/30, nand_neg 30/30, chain_neg 2, generator-native 1.
- Block B frozen controls (256 attempts per target): reductio_f0 s0 **3/606 solved, all 3 strict reductio** (nand_neg), transfer 1/300; f0 s1 0/606; f0 s2 0/606; f0.1 s0 25/606 (strict 25: nand_neg 20, negimp_to_pos 5), transfer 15/300; f0.1 s1 29/606 (strict 29: nand_neg 18, negimp_to_pos 11), transfer 10/300. (`artifacts/fu/blockB_summary.json`)
- Block B base log-probs of the strict reductio proofs under each arm's own Stage-1 model (T = 0.8, start-index marginalised; `novelty_reductio_*_t2_proofs.jsonl`): f0 s0: 60 proofs, max log p −5.6 (round 4), median −21.2, 60 / 52 / 44 below 1/256 / 10⁻⁴ / 10⁻⁵, 0 theorems above 1/256; f0.1 s0: 96 proofs, max −1.4, median −13.1, 25 theorems above 1/256; f0.1 s1: 63 proofs, max −0.35, median −6.4, 26 theorems above 1/256.

## Follow-up — block C (cap 8; `artifacts/p2/ei_derived_ore_strict_*_c8_*`, `artifacts/fu/metrics_c8_*.json`; targets_c8 n = 500, transfer 250; acquisition = solved with a proof containing `patterns.derived_ore_strict`)
- derived_ore_strict_f0_c8 s1: 200/500 solved, **strict acq 0.008** (4 theorems / 5 proofs / first round 2); transfer 118/250, strict 0.004; held-out (cap 8) greedy 0.936; transfer greedy 0.208.
- derived_ore_strict_f0_c8 s0: 197/500 solved, **strict acq 0.008** (4 / 5 / first round 3); transfer 120/250, strict 0.004; held-out greedy 0.926; transfer greedy 0.176.
- Block B base pass@10⁴ (`coverage.py`, stage1_reductio_f0_s0, T = 0.8, first 300 targets of targets_reductio2, 3,000,000 samples; every distinct verified proof stored with its hit count and pattern labels, `hits_by_pattern` asserted = n_ok): **5/300 targets solved** (2 within 512, 3 within 1,000), 92 verified samples of 3·10⁶, all 92 with the strict reductio shape (5 distinct proofs), all 5 targets instances of `~(~A & B), B |- A` (nand_neg): per-sample rates 3.8·10⁻³, 3.6·10⁻³, 1.4·10⁻³, 2·10⁻⁴, 2·10⁻⁴ (first hits at samples 348 / 524 / 309 / 2,905 / 7,344). 0 depth-3, 0 derived-ORE samples. `artifacts/p2/cov_reductio_f0_s0_t2_targets.s0.jsonl`.
- depth3_f0.1_b1 s1: 484/1000 solved, **acq 0.122** (122 / 141 / first round 2); depth-3 theorems per round 1 / 2 / 2 / 2 / 2 / 5 / 27 / 122 (late ignition; sibling b1 s0: 3 / 86 / 198 / 262 / 298 / 332 / 340 / 349); transfer acq 0.124; held-out greedy 0.979.
- **f = 0.1, all 6 arms**: 0.355, 0.316 (b0), 0.349, 0.122 (b1), 0.352, 0.347 (b2): mean 0.307, SD 0.092 (without the late-igniting b1 s1: mean 0.344, SD 0.016).
- Frozen controls (256 attempts/target), depth-3 theorems: a0 s0 0, a1 s0 5, a1 s1 5, a2 s0 4, a2 s1 0, a3 s0 1, a3 s1 10; b0 s0 12, b1 s0 5, b1 s1 2, b2 s0 0, b2 s1 10 (frozen solve counts 55 / 175 / 175 / 166 / 144 / 98 / 127; 127 / 137 / 173).
- derived_ore_strict_f0.001_c8 s0: 202/500 solved, **strict acq 0.022** (11 / 17 / first round 4; per round 2 / 2 / 4 / 4 / 5 / 7 / 8 / 11); transfer 124/250, strict 0.004; held-out greedy 0.929.
- derived_ore_strict_f0.01_c8 s0: 218/500 solved, **strict acq 0.022** (11 / 14 / first round 3; per round 2 / 4 / 6 / 7 / 8 / 9 / 9 / 11); transfer 126/250, strict 0.008; held-out greedy 0.933. Written-length histogram of the solved targets' proofs: 9: 397, 10: 77, 11: 4 (targets have no ≤ 8-line proof); 150–162 of the ~200 solved targets are solved with depth-3 proofs.
- Block C base log-probs (novelty.py, each arm's own cap-8 Stage-1 model, T = 0.8, `novelty_derived_ore_strict_*_c8_*_proofs.jsonl`): median base log p of ALL target proofs found by the arms −4.0 to −4.3 (the cap-8 targets are largely reachable by the cap-8 base); strict proofs: f0 s0 5 proofs, max −1.2 (round 6), median −5.5, 2 below 1/256, 0 below 10⁻⁵, 2 theorems above 1/256; f0 s1 5 proofs, max −2.9, median −4.7, 2 below 1/256, 0 below 10⁻⁵; f10⁻³ 17 proofs, max −0.6, median −8.5, 13 below 1/256, 5 below 10⁻⁵; f10⁻² 14 proofs, max −1.9, median −5.3, 6 below 1/256, 1 below 10⁻⁵. Max-surprisal line: the `AS` opening an ORE box, or the `IMPE`/`ANDE` deriving the disjunction.
- Block C frozen controls (cap 8, 256 attempts/target): f0 s0 159/500 solved, strict 3 theorems (0.006, 4 proofs), transfer 94/250 (strict 1); f0 s1 161/500, strict 2 (0.004, 4 proofs), transfer 97/250 (strict 1). EI f = 0 arms: 197 / 200 solved, strict 4 / 4.
- First round with a depth-3 proof, recomputed with the min-round fix (log.md 05:10): a0 s0 (f=0): 2; a0 s1 (f=0): 4; a1 s0 (f=0): 1; a1 s1 (f=0): 1; a2 s0 (f=0): 1; a2 s1 (f=0): 3; a3 s0 (f=0): 1; a3 s1 (f=0): 1; b0 s0 (f=0.1): 1; b0 s1 (f=0.1): 1; b1 s0 (f=0.1): 1; b1 s1 (f=0.1): 1; b2 s0 (f=0.1): 2; b2 s1 (f=0.1): 1.
- Frozen depth-3 proofs (distinct, normalised): a0 s0 0, a1 s0 5, a1 s1 5, a2 s0 4, a2 s1 0, a3 s0 1, a3 s1 11; b0 s0 17, b1 s0 5, b1 s1 2, b2 s0 0, b2 s1 10. Variance decomposition f = 0 (4 sets x 2 seeds): sd_set 0.012, sd_seed 0.027, set share 17%, F = 1.4; f = 0.1 (3 x 2): sd_set 0, sd_seed 0.094. f = 0 vs f = 0.1 difference of means +0.033, permutation p = 0.46; 6/8 f = 0 arms inside the f = 0.1 range, 5/6 f = 0.1 arms inside the f = 0 range. `artifacts/fu/blockA_summary.json`.
- Block C base pass@10⁴ (`coverage.py`, stage1_derived_ore_strict_f0_c8_s0, first 300 targets_c8, 3·10⁶ samples): **118/300 solved** (87 within 512), 213,129 verified samples (582 distinct proofs); strict derived-ORE proofs on **7 targets** (3,780 hits: targets 8 [3,675 hits, p ≈ 0.37, first at sample 5], 49 [49], 7 [36], 33 [16], 0 [2], 2 [1], 194 [1]); depth-3 proofs 138,198 hits. EI f = 0 s0 acquired 4 strict targets (7, 8, 33, 50) in 256 attempts per target. `artifacts/p2/cov_derived_ore_strict_f0_c8_s0_targets.s0.jsonl`.

## Ignition study (2026-09-16, HANDOFF_BRIEF Task 2; `ignition_analysis.py` → `artifacts/ign/summary.json`, `summary_table.md`, `summary_examples.md`; job files `pod/ign/`; pod logs `artifacts/ign/podlogs_p{1,2,3}/`)
- Stage-1: `train.py --mode abs --steps 6000 --bs 128 --cap 6`, depth-3 `a1` seeds 2–9 (`ckpts/p2/stage1_depth3_f0_a1_s{2..9}.pt`), reductio f = 0 seeds 3–10 (`stage1_reductio_f0_s{3..10}.pt`); held-out greedy at EI round 1: 0.863–0.886 (a1 s0–s9), from `artifacts/p2/ei_depth3_f0_a1_s*/round_1.json`.
- Pre-RL samples (`coverage.py --k 2000 --temperature 0.8 --limit 300 --batch 2000 --seed 0 --procs 4`; first 300 targets of `targets_depth3.jsonl` / `targets_reductio2.jsonl`; `artifacts/ign/cov_<model>.s0.jsonl`, 300 records each, per-proof hit counts and pattern labels; rate = Σ pattern hits / 600,000). Depth-3 a1 s0–s9: **2.5·10⁻⁴ (152 hits, 6 targets), 6.2·10⁻⁴ (370, 7), 0, 0, 1.7·10⁻⁶ (1, 1), 0, 5.0·10⁻⁶ (3, 1), 0, 0, 1.9·10⁻³ (1,112, 29)**; other f = 0 draws a0 s0/s1, a2 s0/s1, a3 s0/s1: see `summary.json` `depth3.coverage` (a2 s0: 301 hits, 8 targets). Reductio s0–s10: **3.1·10⁻⁵ (92 / 3·10⁶, 5 targets; the follow-up's pass@10⁴ file `cov_reductio_f0_s0_t2_targets.s0.jsonl` copied to `cov_reductio_f0_s0.s0.jsonl`), 0, 0, 2.0·10⁻⁵ (12, 3), 0, 0, 0, 2.3·10⁻⁴ (137, 11), 0, 8.3·10⁻⁶ (5, 1), 0**. Every reductio hit is on a `nand_neg` target.
- Frozen control at matched attempts (first pattern-proof sample index ≤ 256, on the 300 sampled targets): depth-3 a1 s0–s9 **4, 3, 0, 0, 0, 0, 1, 0, 0, 16**; reductio s0–s10 **0, 0, 0, 2, 0, 0, 0, 8, 0, 1, 0**. EI round-8 pattern theorems on the same 300 targets (`artifacts/ign/ei_r8_first300.json`): depth-3 251, 263, 253, 238, 4, 0, 253, 0, 241, 260; reductio 34, 0, 0, 11, 0, 0, 0, 64, 0, 29, 0.
- EI arms (`expert_iter.py --rounds 8 --k 32 --temperature 0.8 --batch 768`, seed = model seed; `artifacts/p2/ei_depth3_f0_a1_s{2..9}`, `ei_reductio_f0_s{3..10}_t2`, all `found_<r>.jsonl` kept; counts normalised, min-round rule). Depth-3 cumulative pattern theorems per round, s0–s9: s0 [2, 8, 117, 259, 299, 317, 327, 335]; s1 [4, 41, 175, 279, 311, 340, 354, 364]; s2 [0, 6, 88, 239, 278, 309, 327, 336]; s3 [0, 0, 1, 3, 28, 166, 265, 295]; s4 [0, 0, 0, 0, 0, 0, 1, 4]; s5 [0 × 8]; s6 [0, 1, 13, 166, 279, 313, 325, 330]; s7 [0 × 8]; s8 [0, 0, 1, 15, 114, 254, 292, 312]; s9 [8, 132, 281, 329, 338, 346, 351, 354]. Round-8 theorems / proofs / solved: s2 336/410/677, s3 295/368/589, s4 4/4/323, s5 0/0/357, s6 330/401/601, s7 0/0/318, s8 312/375/582, s9 354/433/602. **Ignition round (≥ 20 theorems)**: s0 3, s1 2, s2 3, s3 5, s4 never, s5 never, s6 4, s7 never, s8 5, s9 2; first pattern proof: 1, 1, 2, 3, 7, never, 2, never, 3, 1.
- Reductio per round, s0–s10: s0 [0, 1, 11, 24, 36, 55, 58, 58]; s3 [0, 0, 0, 0, 0, 1, 4, 16]; s7 [4, 34, 47, 51, 59, 82, 99, 112]; s9 [0, 0, 0, 0, 0, 1, 19, 49]; s1, s2, s4, s5, s6, s8, s10 [0 × 8]. Round-8 theorems / proofs: s3 16/16, s7 112/112, s9 49/51 (all strict, `patterns.reductio`). **Ignition round (≥ 12)**: s7 2, s0 4, s9 7, s3 8, others never; first proof: s7 1, s0 2, s3 6, s9 6. Untrained rounds (no proof found → no training; a frozen sample of all 606 targets): s3 5 rounds = 96,960 samples, s9 5 = 96,960, seven never-igniters 8 = 155,136 each, 0 pattern hits.
- Predicted first-proof round 1/(r·k·N) vs observed: reductio s7 0.23 → 1, s0 1.7 → 2, s3 2.6 → 6, s9 6.2 → 6; depth-3 s9 0.02 → 1, s1 0.05 → 1, s0 0.12 → 1, s6 6.3 → 2, s4 18.8 → 7; zero-rate depth-3 draws: 2, 3, 3, never, never.
- Rule "ignites by round 8 iff ≥ 1 pattern hit in the pre-RL sample": reductio 11/11 correct; depth-3 6/10 (s4 has 1 hit and did not ignite; s2, s3, s8 have 0 hits and did).
- Interventions (from the round-4 state; `artifacts/p2/<arm>_iv{K,T,S}`; K: round 5 at k = 128 then k = 32; T: round 5 at T = 1.0 then 0.8; S: one fine-tuning step on own found_4 + sibling found_4 (`ignition_transfer_mix.py`, `<arm>_ivS_transfer.json`: sibling proofs 204 for reductio s7's found_4; depth-3 sibling s2), then 4 rounds). Cumulative pattern theorems rounds 5–8 — depth-3 s3: K [40, 177, 279, 311], T [34, 197, 290, 319], S [297, 320, 330, 336] (parent [28, 166, 265, 295]); s4: K [0, 0, 0, 0], T [0, 0, 0, 0], S [286, 310, 326, 330]; s5: K 0, T [0, 0, 0, 1], S [300, 322, 330, 335]; s7: K 0, T 0, S [349, 351, 351, 352]. Reductio s3: K [2, 14, 28, 49], T [0, 1, 5, 14], S [56, 60, 71, 79] (parent [0, 1, 4, 16]); s9: K [0, 1, 13, 37], T [0, 1, 10, 27], S [60, 66, 66, 67] (parent [0, 1, 19, 49]); s4, s5, s6, s8, s10: K 0, T 0, S [63, 73, 94, 107] / [60, 65, 72, 91] / [69, 91, 101, 111] / [65, 83, 93, 109] / [61, 72, 93, 102]. **S ignited 11/11 at round 5 (r8 acquisition depth-3 0.330–0.352, reductio 0.111–0.183); K 2/11 (s3 r6, s9 r7), T 2/11 (s3 r8, s9 r8); 0/9 zero-rate arms for K and T.** Attempts per target by round 8: parent 256, K 352, T/S 256 (+ outside data for S).
- Base generalisation (≥ 1 pattern sample in 600k): depth-3 **10 of 16** f = 0 draws (a0 s0/s1, a1 s0–s9, a2 s0/s1, a3 s0/s1; Clopper–Pearson 95 % 0.35–0.85); reductio **4 of 11** (0.11–0.69). Round-1 EI sample (32 × N, pre-training): depth-3 3/10, reductio 1/11.
- Self-check (`log.md` 22:10): every counted reductio pattern proof and 100 per depth-3 arm re-verified with `nd_verify` against its prompt; coverage pattern proofs of four files re-verified — failure counts in log.md.
- Pods: RTX 3090 (p1, $0.50/h, 3.3 h) + 2 × A100 80 GB (p2, p3, $1.59/h, 3.0 h + 3.05 h); no A40 in stock. ≈ $11.5.

## Round 2 — Run 5: target pools that REQUIRE a pattern (2026-09-17; `necessity.py`, `required_pool.py`, `run5_analysis.py` → `artifacts/r5/summary.json`; arms `artifacts/r5/<arm>/round_*.json`, `found_8.jsonl`; pools `data/p2/targets_{reductio,derived_ore}_req.jsonl`, `transfer_*_req.jsonl`; oracle labels `data/p2/run5_*_nec.jsonl`)
- Oracle: `minlen.py --forbid` (DN | ORE_DERIVED | ORE_DERIVED_LOOSE | ORE_IN_IMPI | ANDE_NEGI_HYP | ORE_IN_ORE), 37-case verifier-checked selftest. requires := unrestricted search finds a proof ≤ bound (10) AND restricted search fails within the bound without timeout. Consistency flags stored per record (`oracle_ok`, `restriction_ok`): 0 inconsistencies in 945 + 1,568 + 1,381 + 5,040 labellings; 0 timeouts.
- Reductio candidates (`reductio2_cands.jsonl`, 945 classical-only schema instances, no `( ~ ( ~`): `--forbid DN` proves **0 / 945**; unrestricted ≤ 10 for 559 (6/7/8/9/10: 45/90/230/140/54). Pool **targets_reductio_req 300** (7/8/9/10 lines: 52/133/82/33; 16 schemata, 26 each for 11 of them) + **transfer 150**; class overlap with train_reductio_f0, f0.1, heldout, val-36: 0/0/0/0.
- Cap-8 derived-ORE candidates (generator theorems, 12–16-line generating proofs, no ≤ 8-line proof): 1,568 existing + 1,381 (`raw_long_c8c`, 15M tries, 87,232 classes) + 5,040 (`raw_long_c8d`, 54M tries, 300,774 classes); reachable at ≤ 10: 1,441 / 1,283 / 4,660; **required 120 / 68 / 196** (7.7 / 4.9 / 3.9 %). Pool **targets_derived_ore_req 300** (9/10 lines: 126/174) + **transfer 65**; overlap with the three cap-8 sets, heldout_c8, val-36: 0.
- Reductio arms (k = 32, 8 rounds, T = 0.8; acquisition = targets solved with a normalised proof containing `patterns.reductio`; min-round rule; solved-without-pattern = oracle violations):
  - EI f = 0 s0: **51 / 300 solved, 51 pattern (0.170)**, 52 proofs, first round 1, per round 2/11/25/47/51/51/51/51, written 7-line 51 + 8-line 1, transfer 25/150 (all pattern), heldout greedy 0.891, violations 0. By schema: nand_neg 26/26, negimp_to_pos 25/26, all 14 others 0.
  - EI f = 0 s1: **0 / 300**; s2: **0 / 300** (transfer 0/150; heldout 0.871 / 0.894).
  - EI f = 0.1 s0: **53 (0.177)**, per round 14/39/51/53/53/53/53/53, transfer 27/150; by schema nand_neg 26, negimp_to_pos 26, chain_neg 1. s1: **52 (0.173)**, per round 17/40/51/52/…, transfer 27; nand_neg 26, negimp_to_pos 26.
  - Frozen (256 attempts): f = 0 s0 **4** (nand_neg), s1 0, s2 0; f = 0.1 s0 **27** (nand_neg 21, negimp_to_pos 6), s1 **21**.
  - Base pass@10⁴ on all 300 targets (`artifacts/r5/cov_reductio_f0_s{0,1,2}_req.s0.jsonl`, 3·10⁶ samples each): s0 **6 targets** (113 hits, all strict reductio, all nand_neg; first hits 139/257/388/392/3,428/4,899; 1 within 256), s1 **0**, s2 **1 target, 1 hit** (3.3·10⁻⁷ per sample). Violations 0.
- Cap-8 derived-ORE arms (acquisition = `patterns.derived_ore_strict` on the model's proof):
  - EI f = 0 s0: **26 / 300 solved, 26 pattern (0.087)**, 35 proofs, first round 1, per round 8/12/16/20/22/24/25/26, written 9-line 25 + 10-line 10, transfer 8/65, heldout 0.927, violations 0. s1: **24 (0.080)**, 34 proofs, 7/11/13/14/16/18/21/24, transfer 8/65, violations 0.
  - EI f = 10⁻² s0: **48 (0.160)**, 57 proofs, 14/26/33/34/37/42/44/48, transfer 13/65; s1: **63 (0.210)**, 96 proofs, 20/30/37/42/52/57/61/63, written 9/10/11-line 52/41/3, transfer 14/65. Violations 0.
  - Frozen: f = 0 s0 **13** (21 proofs), s1 **11**; f = 10⁻² s0 **21**, s1 **28**.
  - Base pass@10⁴ f = 0 s0 (`cov_derived_ore_strict_f0_c8_s0_req.s0.jsonl`): **27 / 300 targets** (17,043 hits of 3·10⁶ = 5.7·10⁻³ per sample, all strict; 13 within 256; hit counts per target 5,601 / 4,133 / 2,664 / 1,257 / 1,224 / 711 / …). EI f = 0 s0's 26 solved targets: 18 base-reachable at 10⁴, **8 EI-only**. s1 (`cov_derived_ore_strict_f0_c8_s1_req.s0.jsonl`): **27 / 300** (16,044 hits, 5.3·10⁻³ per sample, 11 within 256; per-target hits 6,746 / 4,261 / 2,584 / 1,600 / 365 / …); EI s1's 24 solved = 17 base-reachable + **7 EI-only**.
- Campaign numbers on the non-required pools, for the side-by-side: block B reductio2 (606) EI f = 0 s0/s1/s2 0.096/0/0, f = 0.1 0.157/0.104, frozen 0.005/0/0, 0.041/0.048; block C c8 (500) strict acquisition f = 0 0.008/0.008, f = 10⁻³ 0.022, f = 10⁻² 0.022, frozen 0.006/0.004, plain solve 0.39–0.44.
- Pods: p1, p2 RTX 3090 $0.50/h from 19:12 / 19:20 UTC (shared with run 2 from 21:34); run-5 share ≈ 2 × 2.8 h ≈ $3. Balance 19:00 → 22:05: $121.40 → see log.
- Bucket: `hf://buckets/dan-pandori/nd-rl/round2/run5/artifacts/r5` (arm dirs with `round_*.json`, `found_8.jsonl`, `found_transfer_8.jsonl`, coverage files, `summary.json`), `…/run5/ckpts/r5` (round-8 EI checkpoints of the 7 arms that trained), `…/run5/data/r5` (the two pools, transfer sets, oracle labels and candidate files).

## Round 2 — Run 3: minimum outside data for ignition (2026-09-17/18; `run3_inject.py`, `run3_analysis.py` → `artifacts/r3/summary.json`; arms `artifacts/r3/<arm>_<cond>/round_{5..8}.json`, `found_8.jsonl`; injected records in `artifacts/p2/<arm>_<cond>_manifest.json`; injection files `data/r3/inject_*.jsonl`)
- Conditions per arm (from the round-4 state; one 600-step fine-tune on own found_4 ≤ 4/theorem ×4 + injected ×4 + 20k retained; then rounds 5–8, k = 32): sib1 / sib4 / sib16 = sibling pattern proofs of distinct theorems (depth-3 from a1 s2's found_4, reductio from s7's found_4); gen4 = 4 verifier-valid 6-line generator proofs with the pattern (class-disjoint from every target/transfer pool); other4 = 4 generator proofs of the other pattern; inv4 = 4 strings with the pattern's tokens and one corrupted citation (nd_verify reasons stored in the file; `train.py --cap 0` does not verify). Ignition threshold 20 / 1,000 (depth-3), 12 / 606 (reductio). Counts: cumulative pattern theorems per round (patterns.py on the normalised proof, min-round rule), acquisition at round 8. Depth-3 s5 inv4's round 8 and reductio s2 inv4 were re-run after OOM kills (the killed s2 inv4 attempt had reached 6 / 47 at rounds 5–6; the clean rerun is what is reported).
- ei_depth3_f0_a1_s4 (depth3; parent per round [0, 0, 0, 0, 0, 0, 1, 4]):
  - sib1: rounds 5–8 pattern theorems 8 / 91 / 195 / 264, solved 256 / 355 / 473 / 555, ignition round 6, round-8 acquisition 0.264; injected 1 (valid 1, with pattern 1), own proofs in the mix 910
  - sib4: rounds 5–8 pattern theorems 105 / 250 / 308 / 322, solved 361 / 522 / 591 / 613, ignition round 5, round-8 acquisition 0.322; injected 4 (valid 4, with pattern 4), own proofs in the mix 910
  - sib16: rounds 5–8 pattern theorems 236 / 303 / 336 / 344, solved 484 / 564 / 612 / 627, ignition round 5, round-8 acquisition 0.344; injected 16 (valid 16, with pattern 16), own proofs in the mix 910
  - gen4: rounds 5–8 pattern theorems 1 / 27 / 98 / 167, solved 252 / 294 / 397 / 480, ignition round 6, round-8 acquisition 0.167; injected 4 (valid 4, with pattern 4), own proofs in the mix 910
  - other4: rounds 5–8 pattern theorems 0 / 0 / 0 / 0, solved 253 / 289 / 312 / 333, ignition round None, round-8 acquisition 0.0; injected 4 (valid 4, with pattern 4), own proofs in the mix 910
  - inv4: rounds 5–8 pattern theorems 4 / 56 / 173 / 252, solved 250 / 312 / 444 / 537, ignition round 6, round-8 acquisition 0.252; injected 4 (valid 0, with pattern 4), own proofs in the mix 910
- ei_depth3_f0_a1_s5 (depth3; parent per round [0, 0, 0, 0, 0, 0, 0, 0]):
  - sib1: rounds 5–8 pattern theorems 23 / 135 / 198 / 262, solved 296 / 439 / 528 / 607, ignition round 5, round-8 acquisition 0.262; injected 1 (valid 1, with pattern 1), own proofs in the mix 968
  - sib4: rounds 5–8 pattern theorems 91 / 241 / 291 / 316, solved 372 / 541 / 618 / 661, ignition round 5, round-8 acquisition 0.316; injected 4 (valid 4, with pattern 4), own proofs in the mix 968
  - sib16: rounds 5–8 pattern theorems 143 / 250 / 294 / 312, solved 413 / 533 / 587 / 625, ignition round 5, round-8 acquisition 0.312; injected 16 (valid 16, with pattern 16), own proofs in the mix 968
  - gen4: rounds 5–8 pattern theorems 2 / 54 / 170 / 235, solved 276 / 351 / 496 / 583, ignition round 6, round-8 acquisition 0.235; injected 4 (valid 4, with pattern 4), own proofs in the mix 968
  - other4: rounds 5–8 pattern theorems 0 / 0 / 0 / 0, solved 276 / 303 / 335 / 346, ignition round None, round-8 acquisition 0.0; injected 4 (valid 4, with pattern 4), own proofs in the mix 968
  - inv4: rounds 5–8 pattern theorems 0 / 0 / 0 / 0, solved 272 / 300 / 321 / 339, ignition round None, round-8 acquisition 0.0; injected 4 (valid 0, with pattern 4), own proofs in the mix 968
- ei_depth3_f0_a1_s3 (depth3; parent per round [0, 0, 1, 3, 28, 166, 265, 295]):
  - sib1: rounds 5–8 pattern theorems 69 / 228 / 286 / 322, solved 311 / 479 / 553 / 599, ignition round 5, round-8 acquisition 0.322; injected 1 (valid 1, with pattern 1), own proofs in the mix 916
  - sib4: rounds 5–8 pattern theorems 101 / 258 / 298 / 326, solved 348 / 516 / 565 / 595, ignition round 5, round-8 acquisition 0.326; injected 4 (valid 4, with pattern 4), own proofs in the mix 916
  - sib16: rounds 5–8 pattern theorems 201 / 284 / 327 / 344, solved 447 / 537 / 588 / 610, ignition round 5, round-8 acquisition 0.344; injected 16 (valid 16, with pattern 16), own proofs in the mix 916
  - gen4: rounds 5–8 pattern theorems 84 / 238 / 290 / 322, solved 325 / 490 / 551 / 589, ignition round 5, round-8 acquisition 0.322; injected 4 (valid 4, with pattern 4), own proofs in the mix 916
  - other4: rounds 5–8 pattern theorems 42 / 197 / 274 / 294, solved 291 / 473 / 570 / 602, ignition round 5, round-8 acquisition 0.294; injected 4 (valid 4, with pattern 4), own proofs in the mix 916
  - inv4: rounds 5–8 pattern theorems 45 / 183 / 267 / 305, solved 288 / 436 / 532 / 579, ignition round 5, round-8 acquisition 0.305; injected 4 (valid 0, with pattern 4), own proofs in the mix 916
- ei_reductio_f0_s1_t2 (reductio; parent per round [0, 0, 0, 0, 0, 0, 0, 0]):
  - sib1: rounds 5–8 pattern theorems 6 / 24 / 31 / 44, solved 6 / 24 / 31 / 44, ignition round 6, round-8 acquisition 0.073; injected 1 (valid 1, with pattern 1), own proofs in the mix 0
  - sib4: rounds 5–8 pattern theorems 21 / 36 / 41 / 43, solved 21 / 36 / 41 / 43, ignition round 5, round-8 acquisition 0.071; injected 4 (valid 4, with pattern 4), own proofs in the mix 0
  - sib16: rounds 5–8 pattern theorems 51 / 62 / 75 / 80, solved 51 / 62 / 75 / 80, ignition round 5, round-8 acquisition 0.132; injected 16 (valid 16, with pattern 16), own proofs in the mix 0
  - gen4: rounds 5–8 pattern theorems 0 / 0 / 0 / 0, solved 0 / 0 / 0 / 0, ignition round None, round-8 acquisition 0.0; injected 4 (valid 4, with pattern 4), own proofs in the mix 0
  - other4: rounds 5–8 pattern theorems 0 / 0 / 0 / 0, solved 0 / 0 / 0 / 0, ignition round None, round-8 acquisition 0.0; injected 4 (valid 4, with pattern 4), own proofs in the mix 0
  - inv4: rounds 5–8 pattern theorems 0 / 0 / 0 / 0, solved 0 / 0 / 0 / 0, ignition round None, round-8 acquisition 0.0; injected 4 (valid 0, with pattern 4), own proofs in the mix 0
- ei_reductio_f0_s2_t2 (reductio; parent per round [0, 0, 0, 0, 0, 0, 0, 0]):
  - sib1: rounds 5–8 pattern theorems 17 / 43 / 61 / 65, solved 17 / 43 / 61 / 65, ignition round 5, round-8 acquisition 0.107; injected 1 (valid 1, with pattern 1), own proofs in the mix 0
  - sib4: rounds 5–8 pattern theorems 32 / 57 / 65 / 65, solved 32 / 57 / 65 / 65, ignition round 5, round-8 acquisition 0.107; injected 4 (valid 4, with pattern 4), own proofs in the mix 0
  - sib16: rounds 5–8 pattern theorems 50 / 66 / 72 / 95, solved 50 / 66 / 72 / 95, ignition round 5, round-8 acquisition 0.157; injected 16 (valid 16, with pattern 16), own proofs in the mix 0
  - gen4: rounds 5–8 pattern theorems 5 / 29 / 53 / 61, solved 5 / 29 / 53 / 61, ignition round 6, round-8 acquisition 0.101; injected 4 (valid 4, with pattern 4), own proofs in the mix 0
  - other4: rounds 5–8 pattern theorems 0 / 0 / 0 / 0, solved 0 / 0 / 0 / 0, ignition round None, round-8 acquisition 0.0; injected 4 (valid 4, with pattern 4), own proofs in the mix 0
  - inv4: rounds 5–8 pattern theorems 0 / 1 / 4 / 14, solved 0 / 1 / 4 / 14, ignition round 8, round-8 acquisition 0.023; injected 4 (valid 0, with pattern 4), own proofs in the mix 0
- Pods: p4 (depth-3 jobs, from 22:05) and p5 (reductio, from 22:17), RTX 3090s, shared with run 4 from 23:09; run-3 share ≈ 2 × 1.6 h ≈ $1.6. Bucket: `hf://buckets/dan-pandori/nd-rl/round2/run3/{artifacts/r3,data/r3,ckpts/r3}` (ckpts: the injected-step checkpoints `*_r4t.pt` and the round-8 checkpoints).

## Round 2 — Run 1: ND → Lean and novelty by scale (2026-09-17/18; `nd2lean.py`, `lean_prompts.py`, `run_vllm.py`, `scale_prompts.py`, `run1_analysis.py` → `artifacts/r1/summary.json`)
- Step 1 agreement (`artifacts/r1/lean_<pool>.jsonl`, one record per proof with nd_verify and Lean verdicts; Lean 4.34.0 core via elan on the VPS; `lean_check` 40 theorems per file, re-split on Lean's ~100-error cap):
  - examples: n 21, both accept 21, nd-only 0, Lean-only 0, both reject 0
  - found_targets16: n 137828, both accept 137828, nd-only 0, Lean-only 0, both reject 0
  - found_transfer16: n 75085, both accept 75085, nd-only 0, Lean-only 0, both reject 0
  - heldout: n 5000, both accept 5000, nd-only 0, Lean-only 0, both reject 0
  - negatives: n 4012, both accept 0, nd-only 0, Lean-only 0, both reject 4012
  - r2_depth4_found: n 15327, both accept 15327, nd-only 0, Lean-only 0, both reject 0
  - r5_c8_found: n 3264, both accept 3264, nd-only 0, Lean-only 0, both reject 0
  - r5_reductio_found: n 2271, both accept 2271, nd-only 0, Lean-only 0, both reject 0
  - rl_targets: n 3000, both accept 3000, nd-only 0, Lean-only 0, both reject 0
  - train10k: n 10000, both accept 10000, nd-only 0, Lean-only 0, both reject 0
  - transfer: n 1638, both accept 1638, nd-only 0, Lean-only 0, both reject 0
  - val36_ref: n 36, both accept 36, nd-only 0, Lean-only 0, both reject 0
  - The found_transfer16 row above is the corrected re-run (the first run rejected 67,773 proofs that start above N1 because premise hypotheses were named by line index; fixed 23:20 UTC). Negatives: 4,012 corrupted proofs (cited line ±k 508, rule swapped 874, atom swapped 874, box bar added/removed 874, line dropped 874, plus the 8 run-3 invalid strings), all rejected by nd_verify, all rejected by Lean or by the translator's structural mirror.
- Step 2 (`data/r1/prompts.jsonl`: 236 theorems = validation-36 + 200 transfer stratified by generating length 7–16 (21 per length, 11 at 16) × 5 example-set draws × 3 forms; 20 worked examples per draw = 12 held-out generator proofs (lengths 2–6) + 8 RL-found transfer proofs of 7–9 written lines, class-disjoint from the test theorems, identical across forms; Qwen3-Coder-30B-A3B-Instruct, vLLM 0.29 on p6 (A100 80 GB), greedy + 8 samples at T = 0.7, top-p 0.95, max 1,200 tokens; `artifacts/r1/gens_qwen30b.jsonl`, `scored_qwen30b.jsonl` (31,860 rows; Lean outputs checked with Lean, token and English outputs with nd_verify after a deterministic parse back to tokens)):
  - english|val36_<=6: n 60 (theorem × draw), greedy 0.500 [0.377, 0.623], pass@8 0.550 [0.425, 0.669]
  - english|val36_>6: n 120 (theorem × draw), greedy 0.100 [0.058, 0.167], pass@8 0.117 [0.071, 0.186]
  - english|transfer_7: n 105 (theorem × draw), greedy 0.276 [0.200, 0.368], pass@8 0.371 [0.285, 0.467]
  - english|transfer_8: n 105 (theorem × draw), greedy 0.400 [0.311, 0.496], pass@8 0.476 [0.383, 0.571]
  - english|transfer_9: n 105 (theorem × draw), greedy 0.219 [0.151, 0.307], pass@8 0.286 [0.208, 0.378]
  - english|transfer_10: n 105 (theorem × draw), greedy 0.381 [0.294, 0.476], pass@8 0.457 [0.365, 0.552]
  - english|transfer_11: n 105 (theorem × draw), greedy 0.238 [0.167, 0.328], pass@8 0.381 [0.294, 0.476]
  - english|transfer_12: n 105 (theorem × draw), greedy 0.248 [0.175, 0.338], pass@8 0.352 [0.268, 0.447]
  - english|transfer_13: n 105 (theorem × draw), greedy 0.390 [0.303, 0.486], pass@8 0.543 [0.448, 0.635]
  - english|transfer_14: n 105 (theorem × draw), greedy 0.314 [0.233, 0.408], pass@8 0.457 [0.365, 0.552]
  - english|transfer_15: n 105 (theorem × draw), greedy 0.276 [0.200, 0.368], pass@8 0.390 [0.303, 0.486]
  - english|transfer_16: n 55 (theorem × draw), greedy 0.273 [0.173, 0.402], pass@8 0.327 [0.218, 0.459]
  - english|all: n 1180 (theorem × draw), greedy 0.292 [0.267, 0.319], pass@8 0.386 [0.358, 0.414]
  - lean|val36_<=6: n 60 (theorem × draw), greedy 0.867 [0.758, 0.931], pass@8 0.900 [0.799, 0.953]
  - lean|val36_>6: n 120 (theorem × draw), greedy 0.458 [0.372, 0.547], pass@8 0.542 [0.453, 0.628]
  - lean|transfer_7: n 105 (theorem × draw), greedy 0.648 [0.553, 0.732], pass@8 0.762 [0.672, 0.833]
  - lean|transfer_8: n 105 (theorem × draw), greedy 0.600 [0.504, 0.689], pass@8 0.705 [0.612, 0.784]
  - lean|transfer_9: n 105 (theorem × draw), greedy 0.552 [0.457, 0.644], pass@8 0.733 [0.642, 0.809]
  - lean|transfer_10: n 105 (theorem × draw), greedy 0.600 [0.504, 0.689], pass@8 0.667 [0.572, 0.750]
  - lean|transfer_11: n 105 (theorem × draw), greedy 0.505 [0.411, 0.599], pass@8 0.657 [0.562, 0.741]
  - lean|transfer_12: n 105 (theorem × draw), greedy 0.552 [0.457, 0.644], pass@8 0.667 [0.572, 0.750]
  - lean|transfer_13: n 105 (theorem × draw), greedy 0.581 [0.485, 0.671], pass@8 0.686 [0.592, 0.767]
  - lean|transfer_14: n 105 (theorem × draw), greedy 0.562 [0.466, 0.653], pass@8 0.657 [0.562, 0.741]
  - lean|transfer_15: n 105 (theorem × draw), greedy 0.505 [0.411, 0.599], pass@8 0.600 [0.504, 0.689]
  - lean|transfer_16: n 55 (theorem × draw), greedy 0.382 [0.265, 0.514], pass@8 0.600 [0.468, 0.719]
  - lean|all: n 1180 (theorem × draw), greedy 0.563 [0.534, 0.591], pass@8 0.675 [0.647, 0.701]
  - tokens|val36_<=6: n 60 (theorem × draw), greedy 0.433 [0.316, 0.559], pass@8 0.517 [0.393, 0.638]
  - tokens|val36_>6: n 120 (theorem × draw), greedy 0.158 [0.104, 0.234], pass@8 0.200 [0.138, 0.280]
  - tokens|transfer_7: n 105 (theorem × draw), greedy 0.305 [0.225, 0.398], pass@8 0.333 [0.250, 0.428]
  - tokens|transfer_8: n 105 (theorem × draw), greedy 0.267 [0.191, 0.358], pass@8 0.343 [0.259, 0.438]
  - tokens|transfer_9: n 105 (theorem × draw), greedy 0.238 [0.167, 0.328], pass@8 0.324 [0.242, 0.418]
  - tokens|transfer_10: n 105 (theorem × draw), greedy 0.257 [0.183, 0.348], pass@8 0.305 [0.225, 0.398]
  - tokens|transfer_11: n 105 (theorem × draw), greedy 0.162 [0.104, 0.244], pass@8 0.295 [0.216, 0.388]
  - tokens|transfer_12: n 105 (theorem × draw), greedy 0.210 [0.143, 0.297], pass@8 0.257 [0.183, 0.348]
  - tokens|transfer_13: n 105 (theorem × draw), greedy 0.314 [0.233, 0.408], pass@8 0.400 [0.311, 0.496]
  - tokens|transfer_14: n 105 (theorem × draw), greedy 0.314 [0.233, 0.408], pass@8 0.362 [0.276, 0.457]
  - tokens|transfer_15: n 105 (theorem × draw), greedy 0.276 [0.200, 0.368], pass@8 0.343 [0.259, 0.438]
  - tokens|transfer_16: n 55 (theorem × draw), greedy 0.236 [0.144, 0.363], pass@8 0.345 [0.234, 0.477]
  - tokens|all: n 1180 (theorem × draw), greedy 0.258 [0.233, 0.283], pass@8 0.326 [0.300, 0.354]
  - paired Lean − tokens (greedy): mean 0.305, bootstrap 95 % [0.257, 0.354] over 236 theorems (per theorem: mean over 5 draws of the 0/1 difference)
  - paired greedy|val36_<=6: 0.433 (n 12)
  - paired greedy|val36_>6: 0.300 (n 24)
  - paired greedy|transfer_7: 0.343 (n 21)
  - paired greedy|transfer_8: 0.333 (n 21)
  - paired greedy|transfer_9: 0.314 (n 21)
  - paired greedy|transfer_10: 0.343 (n 21)
  - paired greedy|transfer_11: 0.343 (n 21)
  - paired greedy|transfer_12: 0.343 (n 21)
  - paired greedy|transfer_13: 0.267 (n 21)
  - paired greedy|transfer_14: 0.248 (n 21)
  - paired greedy|transfer_15: 0.229 (n 21)
  - paired greedy|transfer_16: 0.145 (n 11)
  - paired Lean − tokens (pass8): mean 0.348, bootstrap 95 % [0.298, 0.397] over 236 theorems (per theorem: mean over 5 draws of the 0/1 difference)
  - paired pass8|val36_<=6: 0.383 (n 12)
  - paired pass8|val36_>6: 0.342 (n 24)
  - paired pass8|transfer_7: 0.429 (n 21)
  - paired pass8|transfer_8: 0.362 (n 21)
  - paired pass8|transfer_9: 0.410 (n 21)
  - paired pass8|transfer_10: 0.362 (n 21)
  - paired pass8|transfer_11: 0.362 (n 21)
  - paired pass8|transfer_12: 0.410 (n 21)
  - paired pass8|transfer_13: 0.286 (n 21)
  - paired pass8|transfer_14: 0.295 (n 21)
  - paired pass8|transfer_15: 0.257 (n 21)
  - paired pass8|transfer_16: 0.255 (n 11)

## Round 2 — Run 2: six new patterns (2026-09-17/18; `patterns2.py`, `run2_sets.py`, `necessity.py`, `required_pool.py`, `run2_analysis.py` → `artifacts/r2/summary.json`; arms `artifacts/r2/<kind>_<pattern>_<tag>/`, coverage `artifacts/r2/cov_<pattern>_<tag>.s0.jsonl`, drift `artifacts/r2/drift_negi_ande_hyp_s1_r4.s0.jsonl`; sets `data/r2/train_r2_*.jsonl`, reports `data/r2/assemble_report_r2.json` (cap 6) and `assemble_report_r2_c8.json` (cap 8); pools `data/r2/targets_<pattern>.jsonl`, `transfer_<pattern>.jsonl`, oracle labels `data/r2/nec_<pattern>.jsonl`)
- Pre-registered classes (log.md 19:47 UTC): structural = depth4, impe_chain4, nested_ore; rule sequence = impi_ore, negi_ande_hyp, ori_ore (decoration control). Predicates: `patterns2.py` (15 verifier-checked tests).
- Sets (155,000 = 31,000 per length 2–6, uniform per length from `pool_cap6_recon` with 3,151 run-2 pool classes and the held-out set excluded; written-form counts identical to pruned): struct: excluded none, counts {'derived_ore': 436, 'reductio': 18866, 'depth3': 13902, 'impi_ore': 130, 'negi_ande_hyp': 11, 'ori_ore': 36}; impi_ore_f0: excluded impi_ore, counts {'derived_ore': 356, 'reductio': 18879, 'depth3': 14048, 'negi_ande_hyp': 14, 'ori_ore': 62}; negi_ande_hyp_f0: excluded negi_ande_hyp, counts {'derived_ore': 406, 'reductio': 18932, 'depth3': 14038, 'impi_ore': 128, 'ori_ore': 43}; ori_ore_f0: excluded ori_ore, counts {'derived_ore': 367, 'reductio': 18667, 'depth3': 13970, 'impi_ore': 118, 'negi_ande_hyp': 19}. Cap-8 addendum set c8_depth4_f0 (154,994 = 22,142 per length 2–8 from `pool_cap8` with depth-4 excluded — the pool itself has 0 depth-4 / chain-4 / nested-ORE proofs, `sets_c8.log`): counts {'derived_ore': 3802, 'reductio': 8280, 'depth3': 10435, 'depth4': 0, 'impe_chain4': 0, 'nested_ore': 0, 'nested_ore3': 0, 'impi_ore': 5327, 'negi_ande_hyp': 35, 'ori_ore': 2882}.
- Pools (`necessity.py`, bound 10 / 13; `requires` = restricted search fails within the bound, `uses` = shortest found proof contains the pattern): depth4 500 targets (generator with box-depth cap 5; 8/9/10 lines 269/53/178; 176 required) + 200 transfer; impe_chain4 500 (schemata; 8/9/10: 110/246/144; no restriction oracle) + 200; nested_ore 300 (schemata; all required; 11/12/13: 113/172/15) + 41; impi_ore 500 (generator; 7–10: 144/171/132/53; 109 required) + 200; negi_ande_hyp 400 (generator; 213/86/83/18; 97 required) + 94; ori_ore 400 (generator; generating proof contains it, shortest never does; 310/81/8/1) + 160.
- Arms: EI k = 32, 8 rounds, T = 0.8, retain 20k; frozen at equal attempts; pre-RL pass@2,000 on the 300 shortest targets (nested_ore: 269 at k = 2,000 for s0, 100 at k = 500 for s1; impe_chain4 s0: 96 targets, the run was cut). Acquisition = targets solved with a normalised proof containing the pattern (min-round rule); `viol` = required targets solved without it.
- **depth4** (structural; 500 targets, 176 required):
  - ei_s0: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.957
  - ei_s1: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.959
  - frozen_s0: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.957
  - frozen_s1: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.959
  - ei_c8ctl_s0: solved 485, pattern theorems 485 (0.970), pattern proofs 517, required solved / with pattern / violations 162 / 162 / 0, ignition round 1, first pattern proof round 1, per round [123, 312, 344, 381, 453, 479, 484, 485], solved per round [123, 313, 344, 381, 453, 479, 484, 485], transfer solved 187, held-out greedy 0.924
  - ei_c8ctl_s1: solved 481, pattern theorems 481 (0.962), pattern proofs 506, required solved / with pattern / violations 162 / 162 / 0, ignition round 1, first pattern proof round 1, per round [147, 280, 321, 348, 455, 475, 479, 481], solved per round [148, 281, 322, 349, 456, 476, 479, 481], transfer solved 189, held-out greedy 0.934
  - frozen_c8ctl_s0: solved 150, pattern theorems 148 (0.296), pattern proofs 151, required solved / with pattern / violations 0 / 0 / 0, ignition round 1, first pattern proof round 1, per round [123, 129, 136, 139, 143, 146, 147, 148], solved per round [123, 129, 137, 140, 144, 147, 149, 150], transfer solved 72, held-out greedy 0.918
  - frozen_c8ctl_s1: solved 202, pattern theorems 201 (0.402), pattern proofs 201, required solved / with pattern / violations 0 / 0 / 0, ignition round 1, first pattern proof round 1, per round [147, 165, 180, 187, 189, 195, 199, 201], solved per round [148, 166, 180, 187, 189, 195, 200, 202], transfer solved 85, held-out greedy 0.931
  - ei_c8f0_s0: solved 481, pattern theorems 481 (0.962), pattern proofs 515, required solved / with pattern / violations 161 / 161 / 0, ignition round 1, first pattern proof round 1, per round [174, 297, 331, 417, 469, 474, 479, 481], solved per round [174, 297, 331, 417, 469, 474, 479, 481], transfer solved 187, held-out greedy 0.923
  - ei_c8f0_s1: solved 482, pattern theorems 482 (0.964), pattern proofs 515, required solved / with pattern / violations 161 / 161 / 0, ignition round 1, first pattern proof round 1, per round [131, 297, 332, 338, 348, 414, 474, 482], solved per round [131, 297, 332, 338, 348, 414, 474, 482], transfer solved 187, held-out greedy 0.932
  - frozen_c8f0_s1: solved 180, pattern theorems 180 (0.360), pattern proofs 180, required solved / with pattern / violations 0 / 0 / 0, ignition round 1, first pattern proof round 1, per round [131, 151, 160, 167, 172, 174, 178, 180], solved per round [131, 151, 160, 167, 172, 174, 178, 180], transfer solved 77, held-out greedy 0.923
  - pre-RL s0: 300 targets, 600000 samples, targets with a pattern proof 0, pattern hits 0 (rate 0.00e+00), within 256 attempts 0, targets solved at all 0
  - pre-RL s1: 300 targets, 600000 samples, targets with a pattern proof 0, pattern hits 0 (rate 0.00e+00), within 256 attempts 0, targets solved at all 0
  - pre-RL c8ctl_s0: 300 targets, 600000 samples, targets with a pattern proof 110, pattern hits 78987 (rate 1.32e-01), within 256 attempts 87, targets solved at all 113
  - pre-RL c8ctl_s1: 300 targets, 600000 samples, targets with a pattern proof 134, pattern hits 68779 (rate 1.15e-01), within 256 attempts 113, targets solved at all 137
  - pre-RL c8f0_s0: 300 targets, 600000 samples, targets with a pattern proof 131, pattern hits 117063 (rate 1.95e-01), within 256 attempts 113, targets solved at all 132
  - pre-RL c8f0_s1: 300 targets, 600000 samples, targets with a pattern proof 128, pattern hits 63849 (rate 1.06e-01), within 256 attempts 106, targets solved at all 128
- **impe_chain4** (structural; 500 targets, 0 required):
  - ei_s0: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.957
  - ei_s1: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.959
  - frozen_s0: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.957
  - frozen_s1: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.959
  - pre-RL s0: 96 targets, 192000 samples, targets with a pattern proof 0, pattern hits 0 (rate 0.00e+00), within 256 attempts 0, targets solved at all 0
  - pre-RL s1: 300 targets, 600000 samples, targets with a pattern proof 0, pattern hits 0 (rate 0.00e+00), within 256 attempts 0, targets solved at all 0
- **nested_ore** (structural; 300 targets, 300 required):
  - ei_s0: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.957
  - ei_s1: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.959
  - frozen_s0: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.957
  - frozen_s1: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.959
  - pre-RL s0: 269 targets, 538000 samples, targets with a pattern proof 0, pattern hits 0 (rate 0.00e+00), within 256 attempts 0, targets solved at all 0
  - pre-RL s1: 100 targets, 50000 samples, targets with a pattern proof 0, pattern hits 0 (rate 0.00e+00), within 256 attempts 0, targets solved at all 0
- **impi_ore** (rule_sequence; 500 targets, 109 required):
  - ei_s0: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.946
  - ei_s1: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.961
  - frozen_s0: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.946
  - frozen_s1: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.961
  - ei_natctl_s0: solved 95, pattern theorems 95 (0.190), pattern proofs 98, required solved / with pattern / violations 4 / 4 / 0, ignition round 2, first pattern proof round 1, per round [5, 30, 51, 78, 88, 92, 92, 95], solved per round [5, 30, 51, 78, 88, 92, 92, 95], transfer solved 34, held-out greedy 0.966
  - ei_natctl_s1: solved 111, pattern theorems 111 (0.222), pattern proofs 116, required solved / with pattern / violations 4 / 4 / 0, ignition round 2, first pattern proof round 1, per round [9, 37, 53, 64, 89, 100, 105, 111], solved per round [9, 37, 53, 64, 89, 100, 105, 111], transfer solved 45, held-out greedy 0.963
  - frozen_natctl_s0: solved 10, pattern theorems 10 (0.020), pattern proofs 10, required solved / with pattern / violations 0 / 0 / 0, ignition round 8, first pattern proof round 1, per round [5, 6, 6, 6, 8, 9, 9, 10], solved per round [5, 6, 6, 6, 8, 9, 9, 10], transfer solved 2, held-out greedy 0.957
  - frozen_natctl_s1: solved 17, pattern theorems 17 (0.034), pattern proofs 20, required solved / with pattern / violations 0 / 0 / 0, ignition round 3, first pattern proof round 1, per round [9, 9, 12, 14, 16, 16, 16, 17], solved per round [9, 9, 12, 14, 16, 16, 16, 17], transfer solved 11, held-out greedy 0.959
  - pre-RL s0: 300 targets, 600000 samples, targets with a pattern proof 0, pattern hits 0 (rate 0.00e+00), within 256 attempts 0, targets solved at all 0
  - pre-RL s1: 300 targets, 600000 samples, targets with a pattern proof 0, pattern hits 0 (rate 0.00e+00), within 256 attempts 0, targets solved at all 0
  - pre-RL natctl_s0: 300 targets, 600000 samples, targets with a pattern proof 12, pattern hits 374 (rate 6.23e-04), within 256 attempts 7, targets solved at all 12
  - pre-RL natctl_s1: 300 targets, 600000 samples, targets with a pattern proof 26, pattern hits 777 (rate 1.29e-03), within 256 attempts 14, targets solved at all 26
- **negi_ande_hyp** (rule_sequence; 400 targets, 97 required):
  - ei_s0: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.944
  - ei_s1: solved 59, pattern theorems 59 (0.147), pattern proofs 60, required solved / with pattern / violations 0 / 0 / 0, ignition round 5, first pattern proof round 4, per round [0, 0, 0, 1, 15, 40, 49, 59], solved per round [0, 0, 0, 1, 15, 40, 49, 59], transfer solved 12, held-out greedy 0.970
  - frozen_s0: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.944
  - frozen_s1: solved 2, pattern theorems 2 (0.005), pattern proofs 2, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round 4, per round [0, 0, 0, 1, 2, 2, 2, 2], solved per round [0, 0, 0, 1, 2, 2, 2, 2], transfer solved 1, held-out greedy 0.973
  - ei_natctl_s0: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.957
  - ei_natctl_s1: solved 11, pattern theorems 11 (0.028), pattern proofs 11, required solved / with pattern / violations 0 / 0 / 0, ignition round 7, first pattern proof round 4, per round [0, 0, 0, 1, 1, 3, 11, 11], solved per round [0, 0, 0, 1, 1, 3, 11, 11], transfer solved 3, held-out greedy 0.961
  - frozen_natctl_s0: solved 0, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [0, 0, 0, 0, 0, 0, 0, 0], transfer solved 0, held-out greedy 0.957
  - frozen_natctl_s1: solved 3, pattern theorems 3 (0.007), pattern proofs 3, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round 4, per round [0, 0, 0, 1, 1, 1, 2, 3], solved per round [0, 0, 0, 1, 1, 1, 2, 3], transfer solved 1, held-out greedy 0.959
  - pre-RL s0: 300 targets, 600000 samples, targets with a pattern proof 0, pattern hits 0 (rate 0.00e+00), within 256 attempts 0, targets solved at all 0
  - pre-RL s1: 300 targets, 600000 samples, targets with a pattern proof 2, pattern hits 3 (rate 5.00e-06), within 256 attempts 0, targets solved at all 2
  - pre-RL natctl_s0: 300 targets, 600000 samples, targets with a pattern proof 0, pattern hits 0 (rate 0.00e+00), within 256 attempts 0, targets solved at all 0
  - pre-RL natctl_s1: 300 targets, 600000 samples, targets with a pattern proof 4, pattern hits 7 (rate 1.17e-05), within 256 attempts 0, targets solved at all 4
  - drift s1_r4: {'targets': 300, 'samples': 600000, 'solved': 28, 'pattern_hits': 5138, 'targets_with_pattern': 28, 'rate': 0.008563333333333333, 'frozen256_pattern_targets': 22}
- **ori_ore** (rule_sequence; 400 targets, 0 required):
  - ei_s0: solved 217, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [118, 162, 180, 194, 202, 213, 216, 217], transfer solved 94, held-out greedy 0.967
  - ei_s1: solved 219, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [145, 183, 193, 207, 213, 217, 219, 219], transfer solved 97, held-out greedy 0.969
  - frozen_s0: solved 148, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [118, 126, 131, 136, 137, 140, 144, 148], transfer solved 66, held-out greedy 0.961
  - frozen_s1: solved 173, pattern theorems 0 (0.000), pattern proofs 0, required solved / with pattern / violations 0 / 0 / 0, ignition round None, first pattern proof round None, per round [0, 0, 0, 0, 0, 0, 0, 0], solved per round [145, 157, 162, 165, 165, 168, 171, 173], transfer solved 79, held-out greedy 0.968
  - pre-RL s0: 300 targets, 600000 samples, targets with a pattern proof 0, pattern hits 0 (rate 0.00e+00), within 256 attempts 0, targets solved at all 125
  - pre-RL s1: 300 targets, 600000 samples, targets with a pattern proof 0, pattern hits 0 (rate 0.00e+00), within 256 attempts 0, targets solved at all 148
- Control arms (not pre-registered, added 22:12 UTC): `c8ctl` = the follow-up's cap-8 f = 0 (strict derived-ORE) Stage-1 models on the depth-4 pool (their data has 0 depth-4 proofs; held-out `heldout_c8`, retained slice the cap-8 set); `c8f0` = the addendum set above; `natctl` = the `struct` models (impi_ore at 130 / 155k, negi_ande_hyp at 11 / 155k) on the impi_ore / negi_ande_hyp pools.
- Pods: p1–p3 (RTX 3090, $0.50/h) from 19:12 / 19:20 / 20:05 to 01:12 UTC (shared with run 5 until 22:05): ≈ 15 h ≈ $7.5 (run-2 share ≈ $6). Bucket: `hf://buckets/dan-pandori/nd-rl/round2/run2/{artifacts/r2,data/r2,ckpts/r2}`.
- Step 3 (`data/r1/scale_prompts.jsonl`: 106 theorems in the Lean form with draw-0's 20 examples — depth3_f0 40 (targets of `targets_depth3` solved with a depth-3 proof by the campaign's f = 0 arm a1 s0, drawn from 335), reductio_f0 40 (run 5's required targets solved by f = 0 s0, from 51), transfer9 26 (all transfer theorems whose take-home RL proof was written in 9 lines); Qwen3-{0.6B, 1.7B, 4B, 8B, 14B, 32B} instruct (thinking off), k = 16 at T = 0.7, Lean-checked; `artifacts/r1/gens_scale_<model>.jsonl`, `scored_scale_<model>.jsonl`, `scale_summary.json`):
  - depth3_f0 (n 40): smallest size that proves the theorem — {'Qwen3-14B': 12, 'Qwen3-4B': 12, 'Qwen3-8B': 14, 'Qwen3-32B': 2}; median Qwen3-8B; fraction proved by each size 0.6B 0.00, 1.7B 0.00, 4B 0.30, 8B 0.65, 14B 0.93, 32B 1.00; Spearman(first size, RL round of first proof) = 0.23
  - reductio_f0 (n 40): smallest size that proves the theorem — {'Qwen3-8B': 19, 'Qwen3-14B': 10, 'Qwen3-32B': 6, 'Qwen3-4B': 4, 'none': 1}; median Qwen3-8B; fraction proved by each size 0.6B 0.00, 1.7B 0.00, 4B 0.10, 8B 0.53, 14B 0.65, 32B 0.85; Spearman(first size, RL round of first proof) = 0.03
  - transfer9 (n 26): smallest size that proves the theorem — {'Qwen3-8B': 10, 'Qwen3-14B': 6, 'Qwen3-4B': 8, 'none': 1, 'Qwen3-0.6B': 1}; median Qwen3-8B; fraction proved by each size 0.6B 0.04, 1.7B 0.00, 4B 0.35, 8B 0.69, 14B 0.96, 32B 0.96; Spearman(first size, Phase-1 base log p at T = 0.8) = -0.24 over 26 theorems (negative = harder for the base ⇒ larger first size); Spearman(first size, RL round of first proof) = 0.38
- Pods: p6 (A100 80 GB PCIe, $1.59/h, 250 GB container disk) 23:22 → 01:30 UTC ≈ 2.1 h ≈ $3.4 (+ model downloads: 30B-A3B 57 GB, Qwen3 0.6B–32B ≈ 90 GB). Bucket: `hf://buckets/dan-pandori/nd-rl/round2/run1/{artifacts/r1,data/r1}` (generations, scored rows, agreement files, prompts).

## Round 2 — Run 4: GRPO vs expert iteration at f = 0 (2026-09-17/18; `grpo.py`, `run4_analysis.py` → `artifacts/r4/summary.json`; arms `artifacts/r4/grpo_g{8,32}_{a1,a2,a3}_s{0,1}/round_<r>.json`, `found_<r>.jsonl`; round-8 checkpoints `ckpts/r4/`)
- Algorithm: on-policy GRPO, binary verifier reward, group-mean baseline (no std normalisation), fixed loss divisor 400, no KL, AdamW lr 10⁻⁴ (β 0.9 / 0.95, clip 1.0), one update per batch of P × G samples at T = 0.8; G = 8 with P = 64, G = 32 with P = 16 (512 samples per step, 500 steps = 256,000 samples = EI's 8 × 32 × 1,000). Bookkeeping at 8 round-equivalents like `expert_iter.py` (every distinct verified proof sampled during training; transfer pass@32 at the boundary, never trained on; greedy on transfer and held-out). No retained pretraining data. Depth-3 acquisition = targets with a normalised proof of box depth ≥ 3 (min-round rule) — the same statistic as the follow-up's EI arms on the same six Stage-1 draws.
- Bookkeeping notes: grpo_g8_a1_s0's round-8 files were not pulled before p5 was deleted (its round-8 checkpoint was) — its numbers are at round-equivalent 7 (224,000 samples); grpo_g8_a2_s1's round_1.json is from a killed duplicate run (log.md 01:01) and is excluded from the per-round table; grpo_g32_a2_s0 was re-run from scratch after p4 was deleted before its pull (log.md 01:14).
- grpo_g32_a1_s0 (G = 32, draw a1_s0): round-equivalent 8: solved 728 / 1,000, **depth-3 acquisition 440 (0.440)** vs EI 0.335 (645 solved); pattern proofs 722; per round-equivalent [176, 321, 368, 389, 402, 415, 432, 440]; solved per round-equivalent [418, 600, 646, 670, 680, 695, 718, 728]; mean reward per round-equivalent [0.25, 0.44, 0.52, 0.55, 0.57, 0.59, 0.61, 0.63]; fraction of groups with reward variance [0.4, 0.47, 0.4, 0.37, 0.28, 0.25, 0.23, 0.22]; held-out greedy at the end 0.564 (Stage-1 ≈ 0.87–0.95); transfer pass@32 at the end 315 / 500, with depth-3 183; written-length histogram {'7': 615, '8': 432, '9': 161, '10': 8}
- grpo_g32_a1_s1 (G = 32, draw a1_s1): round-equivalent 8: solved 798 / 1,000, **depth-3 acquisition 467 (0.467)** vs EI 0.364 (698 solved); pattern proofs 746; per round-equivalent [251, 350, 387, 423, 440, 451, 458, 467]; solved per round-equivalent [521, 650, 706, 749, 770, 783, 790, 798]; mean reward per round-equivalent [0.34, 0.5, 0.55, 0.6, 0.63, 0.67, 0.67, 0.7]; fraction of groups with reward variance [0.49, 0.48, 0.44, 0.37, 0.35, 0.33, 0.28, 0.28]; held-out greedy at the end 0.643 (Stage-1 ≈ 0.87–0.95); transfer pass@32 at the end 353 / 500, with depth-3 202; written-length histogram {'7': 644, '8': 453, '9': 218, '10': 16, '11': 2}
- grpo_g32_a2_s0 (G = 32, draw a2_s0): round-equivalent 8: solved 737 / 1,000, **depth-3 acquisition 437 (0.437)** vs EI 0.350 (589 solved); pattern proofs 668; per round-equivalent [283, 342, 364, 372, 395, 415, 424, 437]; solved per round-equivalent [506, 605, 651, 665, 687, 711, 724, 737]; mean reward per round-equivalent [0.31, 0.45, 0.51, 0.55, 0.58, 0.6, 0.6, 0.63]; fraction of groups with reward variance [0.49, 0.47, 0.41, 0.33, 0.32, 0.29, 0.31, 0.26]; held-out greedy at the end 0.657 (Stage-1 ≈ 0.87–0.95); transfer pass@32 at the end 323 / 500, with depth-3 190; written-length histogram {'7': 617, '8': 378, '9': 162, '10': 7, '11': 2}
- grpo_g32_a2_s1 (G = 32, draw a2_s1): round-equivalent 8: solved 725 / 1,000, **depth-3 acquisition 404 (0.404)** vs EI 0.341 (652 solved); pattern proofs 584; per round-equivalent [124, 309, 351, 366, 376, 389, 399, 404]; solved per round-equivalent [390, 611, 667, 689, 697, 709, 719, 725]; mean reward per round-equivalent [0.24, 0.46, 0.5, 0.55, 0.57, 0.61, 0.6, 0.63]; fraction of groups with reward variance [0.38, 0.45, 0.36, 0.28, 0.26, 0.21, 0.22, 0.21]; held-out greedy at the end 0.610 (Stage-1 ≈ 0.87–0.95); transfer pass@32 at the end 338 / 500, with depth-3 188; written-length histogram {'7': 666, '8': 378, '9': 89, '10': 1}
- grpo_g32_a3_s0 (G = 32, draw a3_s0): round-equivalent 8: solved 710 / 1,000, **depth-3 acquisition 411 (0.411)** vs EI 0.361 (605 solved); pattern proofs 601; per round-equivalent [252, 337, 377, 384, 394, 402, 409, 411]; solved per round-equivalent [442, 581, 633, 651, 674, 694, 705, 710]; mean reward per round-equivalent [0.25, 0.42, 0.5, 0.53, 0.55, 0.59, 0.6, 0.61]; fraction of groups with reward variance [0.44, 0.49, 0.37, 0.27, 0.26, 0.23, 0.22, 0.22]; held-out greedy at the end 0.376 (Stage-1 ≈ 0.87–0.95); transfer pass@32 at the end 302 / 500, with depth-3 174; written-length histogram {'7': 629, '8': 373, '9': 104}
- grpo_g32_a3_s1 (G = 32, draw a3_s1): round-equivalent 8: solved 691 / 1,000, **depth-3 acquisition 395 (0.395)** vs EI 0.352 (583 solved); pattern proofs 554; per round-equivalent [201, 317, 357, 371, 377, 383, 389, 395]; solved per round-equivalent [449, 598, 644, 660, 668, 676, 685, 691]; mean reward per round-equivalent [0.27, 0.44, 0.51, 0.56, 0.58, 0.59, 0.6, 0.61]; fraction of groups with reward variance [0.45, 0.51, 0.46, 0.35, 0.26, 0.26, 0.23, 0.2]; held-out greedy at the end 0.521 (Stage-1 ≈ 0.87–0.95); transfer pass@32 at the end 313 / 500, with depth-3 183; written-length histogram {'7': 614, '8': 340, '9': 85, '10': 1}
- grpo_g8_a1_s0 (G = 8, draw a1_s0): round-equivalent 7: solved 778 / 1,000, **depth-3 acquisition 478 (0.478)** vs EI 0.335 (645 solved); pattern proofs 779; per round-equivalent [344, 375, 405, 436, 458, 471, 478]; solved per round-equivalent [593, 660, 697, 733, 755, 770, 778]; mean reward per round-equivalent [0.37, 0.56, 0.63, 0.68, 0.7, 0.72, 0.74]; fraction of groups with reward variance [0.31, 0.23, 0.12, 0.11, 0.1, 0.09, 0.08]; held-out greedy at the end 0.562 (Stage-1 ≈ 0.87–0.95); transfer pass@32 at the end 338 / 500, with depth-3 200; written-length histogram {'7': 604, '8': 403, '9': 217, '10': 36, '11': 2}
- grpo_g8_a1_s1 (G = 8, draw a1_s1): round-equivalent 8: solved 819 / 1,000, **depth-3 acquisition 505 (0.505)** vs EI 0.364 (698 solved); pattern proofs 816; per round-equivalent [357, 414, 453, 469, 483, 490, 501, 505]; solved per round-equivalent [648, 733, 767, 784, 798, 805, 814, 819]; mean reward per round-equivalent [0.42, 0.61, 0.7, 0.72, 0.74, 0.76, 0.77, 0.79]; fraction of groups with reward variance [0.33, 0.25, 0.14, 0.12, 0.09, 0.08, 0.07, 0.05]; held-out greedy at the end 0.644 (Stage-1 ≈ 0.87–0.95); transfer pass@32 at the end 381 / 500, with depth-3 229; written-length histogram {'7': 633, '8': 452, '9': 263, '10': 38, '11': 4}
- grpo_g8_a2_s0 (G = 8, draw a2_s0): round-equivalent 8: solved 777 / 1,000, **depth-3 acquisition 473 (0.473)** vs EI 0.350 (589 solved); pattern proofs 730; per round-equivalent [373, 400, 419, 433, 445, 454, 463, 473]; solved per round-equivalent [598, 640, 698, 724, 741, 755, 764, 777]; mean reward per round-equivalent [0.4, 0.55, 0.62, 0.66, 0.69, 0.71, 0.72, 0.73]; fraction of groups with reward variance [0.35, 0.24, 0.14, 0.12, 0.08, 0.08, 0.07, 0.06]; held-out greedy at the end 0.672 (Stage-1 ≈ 0.87–0.95); transfer pass@32 at the end 347 / 500, with depth-3 210; written-length histogram {'7': 604, '8': 406, '9': 203, '10': 18, '11': 3}
- grpo_g8_a2_s1 (G = 8, draw a2_s1): round-equivalent 8: solved 799 / 1,000, **depth-3 acquisition 472 (0.472)** vs EI 0.341 (652 solved); pattern proofs 790; per round-equivalent [329, 388, 402, 408, 433, 444, 467, 472]; solved per round-equivalent [605, 677, 710, 722, 751, 771, 793, 799]; mean reward per round-equivalent [0.38, 0.58, 0.64, 0.67, 0.69, 0.72, 0.74, 0.76]; fraction of groups with reward variance [0.31, 0.14, 0.12, 0.08, 0.1, 0.08, 0.09, 0.07]; held-out greedy at the end 0.621 (Stage-1 ≈ 0.87–0.95); transfer pass@32 at the end 351 / 500, with depth-3 207; written-length histogram {'7': 658, '8': 464, '9': 225, '10': 34}
- grpo_g8_a3_s0 (G = 8, draw a3_s0): round-equivalent 8: solved 747 / 1,000, **depth-3 acquisition 447 (0.447)** vs EI 0.361 (605 solved); pattern proofs 667; per round-equivalent [350, 381, 407, 421, 428, 435, 442, 447]; solved per round-equivalent [580, 647, 684, 705, 713, 726, 738, 747]; mean reward per round-equivalent [0.32, 0.53, 0.6, 0.65, 0.67, 0.68, 0.69, 0.71]; fraction of groups with reward variance [0.35, 0.23, 0.12, 0.08, 0.08, 0.08, 0.08, 0.07]; held-out greedy at the end 0.340 (Stage-1 ≈ 0.87–0.95); transfer pass@32 at the end 332 / 500, with depth-3 197; written-length histogram {'7': 628, '8': 365, '9': 151, '10': 10}
- grpo_g8_a3_s1 (G = 8, draw a3_s1): round-equivalent 8: solved 780 / 1,000, **depth-3 acquisition 472 (0.472)** vs EI 0.352 (583 solved); pattern proofs 749; per round-equivalent [325, 369, 395, 418, 433, 448, 465, 472]; solved per round-equivalent [572, 651, 678, 710, 731, 748, 768, 780]; mean reward per round-equivalent [0.35, 0.55, 0.61, 0.64, 0.67, 0.68, 0.71, 0.72]; fraction of groups with reward variance [0.33, 0.22, 0.12, 0.12, 0.1, 0.1, 0.09, 0.09]; held-out greedy at the end 0.531 (Stage-1 ≈ 0.87–0.95); transfer pass@32 at the end 330 / 500, with depth-3 192; written-length histogram {'7': 621, '8': 431, '9': 238, '10': 24}
- EI reference ei_depth3_f0_a1_s0: solved 645, acquisition 0.335, per round [2, 8, 117, 259, 299, 317, 327, 335], held-out greedy 0.907
- EI reference ei_depth3_f0_a1_s1: solved 698, acquisition 0.364, per round [4, 41, 175, 279, 311, 340, 354, 364], held-out greedy 0.930
- EI reference ei_depth3_f0_a2_s0: solved 589, acquisition 0.350, per round [3, 75, 199, 291, 308, 327, 341, 350], held-out greedy 0.945
- EI reference ei_depth3_f0_a2_s1: solved 652, acquisition 0.341, per round [0, 0, 8, 78, 242, 298, 317, 341], held-out greedy 0.903
- EI reference ei_depth3_f0_a3_s0: solved 605, acquisition 0.361, per round [1, 31, 185, 280, 319, 335, 348, 361], held-out greedy 0.945
- EI reference ei_depth3_f0_a3_s1: solved 583, acquisition 0.352, per round [7, 92, 227, 319, 338, 345, 346, 352], held-out greedy 0.919
- Summary: G = 8 acquisition mean 0.474 (range 0.447–0.505, n = 6), G = 32 mean 0.426 (0.395–0.467); EI mean 0.351 (0.335–0.364). Held-out greedy at the end: G = 8 [0.34, 0.53, 0.56, 0.62, 0.64, 0.67], G = 32 [0.38, 0.52, 0.56, 0.61, 0.64, 0.66].
- The 85M / relative-codec arm of the proposal was not run (no access to that code). Pods: p4 / p5 (RTX 3090) from 23:09 to 01:36 UTC ≈ 2 × 2.4 h ≈ $2.4 (shared with run 3). Bucket: `hf://buckets/dan-pandori/nd-rl/round2/run4/{artifacts/r4,ckpts/r4}`.

# lean-format (proposal 8) — Lean as the training format for the from-scratch model

Every number below is re-derived by `python3 lean_format_analysis.py` → `artifacts/lf/summary.json` (stdout copy `artifacts/lf/analysis_stdout.txt`) from pulled files under `artifacts/lf/`; token comparators from `artifacts/p2/{ei,frozen}_depth3_f0_a1_s{0,1}/`, `artifacts/stage1_*_transfer2_k16_norm.json` and `ladder.md` (branch `dan_ladder_a`). A Lean-format sample counts iff Lean accepts the literal sampled text **and** `nd_verify` accepts the ND proof it denotes; counts are distinct denoted proofs (numbered from N1). Formats: `lean_rand` (random labels, pre-registered primary), `lean_seq` (first-appearance order + random offset). Model 4 layers, d 256 in all arms; parameters differ only by vocabulary rows (107 vs 99).

## Sequence lengths (`artifacts/lf/seqlen.json`; one symbol per token)
| file | n | token `abs` prompt / proof / total mean (p95, max) | Lean prompt / proof / total mean (p95, max) | ratio total |
|---|---:|---|---|---:|
| `data/train.jsonl` | 154,990 | 36.9 / 72.1 / 109.1 (187, 516) | 51.8 / 82.8 / 134.5 (214, 537) | 1.233 |
| `data/p2/train_depth3_f0_a1.jsonl` | 155,000 | 39.1 / 73.6 / 112.6 (185, 444) | 54.2 / 84.3 / 138.4 (214, 464) | 1.229 |
| `data/heldout.jsonl` | 5,000 | 37.0 / 72.6 / 109.6 (191, 440) | 51.8 / 83.2 / 135.1 (215, 461) | 1.232 |
| `data/p2/heldout.jsonl` | 5,000 | 39.7 / 81.3 / 121.0 (212, 465) | 54.1 / 92.2 / 146.3 (238, 486) | 1.210 |

## P1 — Stage-1 held-out greedy (band: within 2 pp of token)
| Stage-1 set | token | `lean_seq` | `lean_rand` | source |
|---|---|---|---|---|
| a1 (depth-3 removed), seeds 0 / 1, `data/p2/heldout.jsonl` | 0.883 / 0.883 | **0.909 / 0.896** (in band) | **0.882 / 0.790** (s1 out of band, −9.3 pp) | `artifacts/lf/frozen_d3_<sch>_s<k>/round_1.json`; `artifacts/p2/ei_depth3_f0_a1_s<k>/round_1.json` |
| full cap-6 set, `data/heldout.jsonl` | 0.948 | **0.936** (−1.2 pp, in band) | **0.830** (−11.8 pp) | `artifacts/lf/stage1_full_<sch>_heldout_greedy.json` |
| full set, by proof length 2 / 3 / 4 / 5 / 6 | – | 1.000 / 0.988 / 0.950 / 0.888 / 0.852 | 0.999 / 0.979 / 0.864 / 0.645 / 0.662 | same |
- `lean_rand` failures are mostly wrong-name citations: of 851 failures on the full set, 813 parse and are rejected by both checkers (`nd_verify` reasons on the denoted proof: rule check ANDI 209, NEGE 174, IMPE 170, ORI 108, R 54, ANDE1 35, …), 38 are outside the grammar (36 `unbound`); `artifacts/lf/stage1_full_rand_heldout_greedy.jsonl`, log.md 03:17. `lean_seq --no_shift` full set: 0.939.

## P2 — depth-3 f = 0 dial, set a1, 8 rounds × 32 attempts (`phase2_metrics.arm_metrics`, pattern `depth3`, on `artifacts/lf/{ei,frozen}_d3_<sch>_s<k>/`)
| arm | targets solved / 1,000 | **acquisition** (theorems / depth-3 proofs / first round) | transfer solved / 500, acq | held-out greedy final |
|---|---|---|---|---|
| token EI s0 / s1 | 645 / 698 | **0.335 / 0.364** (335 / 409 / r1; 364 / 426 / r1) | 331, 0.372 / 337, 0.366 | 0.907 / 0.930 |
| token frozen s0 / s1 | 175 / 175 | 0.005 / 0.005 | 84, 0.004 / 93, 0.004 | 0.883 / 0.883 |
| `lean_seq` EI s0 / s1 | 715 / 736 | **0.476 / 0.479** (476 / 661 / r1; 479 / 710 / r1) | 369, 0.494 / 388, 0.512 | 0.965 / 0.969 |
| `lean_seq` frozen s0 / s1 | 409 / 326 | **0.206 / 0.134** (206 / 232; 134 / 168) | 206, 0.184 / 184, 0.156 | 0.909 / 0.896 |
| `lean_rand` EI s0 / s1 | 663 / 673 | **0.433 / 0.462** (433 / 619 / r1; 462 / 657 / r1) | 347, 0.472 / 344, 0.476 | 0.925 / 0.913 |
| `lean_rand` frozen s0 / s1 | 351 / 423 | **0.202 / 0.280** (202 / 241; 280 / 336) | 173, 0.198 / 233, 0.302 | 0.882 / 0.790 |
- Acquisition by round, `lean_seq` s0: 0.104 / 0.362 / 0.401 / 0.418 / 0.439 / 0.456 / 0.465 / 0.476; token s0: 0.002 / 0.008 / 0.117 / 0.259 / 0.299 / 0.317 / 0.327 / 0.335.
- **Base reachability**: the frozen Lean Stage-1 models (zero depth-3 proofs in pretraining) already solve 8.5–17.2 % of the targets with a depth-3 proof in the first 32 attempts and 13.4–28.0 % in 256; the token frozen controls: 0.5 % in 256. EI adds +0.27 / +0.35 (`lean_seq`) and +0.23 / +0.18 (`lean_rand`) over the frozen control at equal attempts (token: +0.33 / +0.36).

## P3 — ladder rung T1, 8 rounds × 32 attempts (`artifacts/lf/la_{T1,frozen}_<sch>_s<k>/found_transfer_8.jsonl`, `found_8.jsonl`; pools `data/ladder/`; `L*` = max L with ≥ 5 theorems solved at `L_true` ≥ L)
| arm | **`L*` transfer** | `L*` targets | transfer solved / 2,285 | by `L_true` 7 / 8 / 9 / 10 / 11 / 12 / 13 | targets solved / 4,495 | held-out greedy r1 → r8 | arm minutes (4 arms per GPU) |
|---|---:|---:|---:|---|---:|---|---:|
| token T1 s0 / s1 (on file) | 10 / 10 | 10 / 10 | 612 / 623 | 140, 137, 300, 34, 1, 0, 0 / 139, 138, 312, 34, 0, 0, 0 | 2,400 / 2,455 | 0.948 → 0.953 / 0.957 | 65 / 86 (2–3 per GPU) |
| token frozen s0 / s1 (on file) | 7 / 7 | 8 / 8 | 22 / 24 | 20, 2, 0 … / 21, 3, 0 … | 426 / 408 | 0.948 | – |
| `lean_seq` T1 s0 / s1 | **11 / 11** | 11 / 11 | **794 / 839** | 75, 161, 463, 83, 9, 3, 0 / 118, 162, 465, 81, 9, 3, 1 | 2,604 / 2,674 | 0.936 → 0.944 / 0.943 | 248 / 252 |
| `lean_seq` frozen s0 / s1 | **10 / 10** | 10 / 10 | 304 / 309 | 43, 113, 134, 14, 0 / 50, 111, 134, 14, 0 | 1,768 / 1,759 | 0.936 | 247 / 251 |
| `lean_rand` T1 s0 / s1 | **10 / 10** | 10 / 10 | 583 / 657 | 55, 131, 351, 44, 2 / 114, 139, 360, 40, 4 | 2,339 / 2,396 | 0.830 → 0.923 / 0.924 | 226 / 229 |
| `lean_rand` frozen s0 / s1 | **9 / 9** | 9 / 9 | 229 / 222 | 42, 89, 96, 2 / 41, 87, 93, 1 | 1,511 / 1,501 | 0.830 | 222 / 226 |
- `lean_seq` reaches transfer `L*` = 11 at round 5 in both seeds (`round_<r>.json` `transfer_cum.lstar`). Theorems solved at `L_true` ≥ 11: 12 / 13 (token 1 / 0); ≥ 12: 3 / 4. No pool label is contradicted by a shorter accepted proof (shortest written proof of the `L_true` ≥ 11 theorems: 11–13 lines), so `L*` is unchanged after relabelling.
- Unexplained and against the trend: the `L_true` = 7 bin is solved *less* often by Lean T1 arms (55–118 of 300) than by the token arms (139–140).

## P4 — mechanism test: Stage-1 base model on the full cap-6 set, pass@16 on `data/transfer.jsonl` (1,638; `artifacts/lf/stage1_full_<tag>_transfer2_k16.jsonl`)
| model | pass@16 | distinct verified proofs of written length 6 / 7 / 8 / ≥ 9 |
|---|---|---|
| token `abs` / `rel` / `abs-fixed` (on file) | 0.447 / 0.405 / 0.385 | 363 / 108 / 1 / 0 ; 332 / 79 / 0 / 0 ; 359 / 0 / 0 / 0 |
| `lean_seq` | **0.571** (935) | 333 / **266 / 130 / 18** |
| `lean_rand` | **0.524** (859) | 263 / **228 / 86 / 7** |
| `lean_seq --no_shift` (names n1… only) | 0.284 (466) | 229 / 37 / 0 / 0 |

## P5 — Lean vs `nd_verify` (`artifacts/lf/gate_*.jsonl`, `gate_*.disagree.jsonl`, `artifacts/lf/disagreement_kinds.json`, `artifacts/lf/record_*.json`)
- In the loop, every distinct grammar-valid sample of every job: 18,424,794 samples, 3,515,138 outside the strict grammar (19.1 %; `lean_rand` 20.7 %, `lean_seq` 17.4 %; mostly `unbound` = a name not in scope), **13,887,708 distinct checked by both: both accept 4,046,345, both reject 9,840,903, `nd_verify` accepts ∧ Lean rejects 0, Lean accepts ∧ `nd_verify` rejects 460** (33 per million; pre-registered expectation ≤ 10 per million: wrong).
- The 460, by cause: 206 `na.elim` on a non-`False` hypothesis (Lean resolves it to `Not.elim`; `nd2lean.py`'s BOTE rendering is looser than BOTE); 137 `¬A` ≡ `A → False` unfolding (NEGE 67, ORI 37, IMPI 30, ANDE / ANDI 3); 117 proofs that restate only some premises (legal Lean; ND requires every PR line — a gap in `lean_tok.inverse`'s strictness, not in Lean). None is counted anywhere.
- Checker of record (unmodified `nd2lean.py --check` + `nd_verify` on every counted proof, `found_8` and `found_transfer_8` of all 16 Lean arms): **41,840 / 41,840 both accept, 0 disagreements**.

## P6 — throughput and round time
- Lean 85.5 proofs per process-second in the loop (162,341 process-s for 13.89 M; 12 processes wide, 14,216 s wall over all jobs) vs `nd_verify` 14,497 per second on the same proofs: **≈ 170× slower per core**. Pod self-test on valid short proofs: 120–285 / s / core vs ≈ 20,000 / s.
- Solo like-for-like frozen round on the same pod, nothing else on the GPU (`artifacts/lf/timing_*/round_1.json`): depth-3 dial token 59 s / 64 s, `lean_rand` 105 s (**1.8×**), `lean_seq` 141 s (**2.2×**); ladder token 406 s / 402 s, `lean_rand` 594 s (**1.46×**), `lean_seq` 657 s (**1.64×**). Below the 3× stop rule. With four arms per 3090 a ladder round took 27–36 min.
- Cost: RunPod balance 127.78 → 121.56 (`rpbalance` at 03:03 and 08:04 UTC): **≈ $6.2** for ≈ 12.3 pod-hours on four RTX 3090 pods (`~/pods.log` lf-1 … lf-4, 03:04–08:03 UTC). Budget $30.

## Bucket
`hf://buckets/dan-pandori/nd-rl/lean-format/{ckpts,artifacts,data}` — `ckpts/lf` (Stage-1 models, depth-3 EI rounds), `ckpts/ladder` (ladder EI rounds), `artifacts/lf` (everything above incl. per-round found files), `data` (the a1 training set and the ladder pools used).


# ds-generator (proposal 9, run 2) — the generator's proof-shape distribution

**Every model here is a 3.3 M-parameter from-scratch GPT in the `lean_seq` Lean surface format**, Stage-1 =
`train.py --mode lean_seq --steps 6000 --bs 128 --cap 6 --seed <s>` on a 155,000-record set of **cap-6 generator
proofs, flat 31,000 per pruned length 2–6, depth-3 excluded**. The three sets differ only in the generator knobs
(`data/dsg/README.md`):

| arm | training set | Stage-1 checkpoints | generator |
|---|---|---|---|
| C0 | `data/p2/train_depth3_f0_a1.jsonl` | `ckpts/lf/stage1_a1_seq_s{0,1}.pt` (md5 `9bde44c0…`, `fc27e52d…`; trained 2026-09-21 on an RTX 3090, **not** retrained) | control knobs |
| G1 | `data/dsg/train_g1.jsonl` | `ckpts/dsg/stage1_g1_s{0,1}.pt` (retrained 2026-09-23/24 on A6000 / 4090) | `--ore_steps 3 --ore_boxes` |
| G2 | `data/dsg/train_g2.jsonl` | `ckpts/dsg/stage1_g2_s{0,1}.pt` (retrained 2026-09-23/24 on A6000 / 4090) | `--ladder --drop_contra --ore_boxes` (the ladder pools' strict long generator at cap 6) |

Sampler for every ladder number: this branch's decode path with the memoised RoPE table from run `efficiency`,
`ND_SAMPLE_COMPACT=0`, **batch 512, `--max_new` 512, T 0.8**, held fixed across all three arms and both seeds.
A proof counts only if `nd_verify` **and** Lean accept it. Source files are named per table; all are under
`artifacts/dsg/` unless said, and `dsg_analysis.py` regenerates every table below into `artifacts/dsg/summary.json`.

## G1 — held-out greedy (`heldout2_<arm>_s<k>.json`, `eval_set.py --k 1 --temperature 0 --batch 512`, `data/p2/heldout.jsonl`, 5,000)

| arm s | overall | 2 | 3 | 4 | 5 | 6 | none | pattern | depth-3 thms | 6-line none |
|---|---|---|---|---|---|---|---|---|---|---|
| c0 s0 | 0.909 | 0.994 | 0.989 | 0.951 | 0.924 | 0.686 | 0.952 | 0.770 | 0.488 | 0.818 |
| c0 s1 | 0.897 | 0.997 | 0.992 | 0.967 | 0.944 | 0.584 | 0.964 | 0.680 | 0.272 | 0.834 |
| g1 s0 | 0.861 | 0.998 | 0.984 | 0.947 | 0.923 | 0.451 | 0.951 | 0.569 | 0.014 | 0.830 |
| g1 s1 | 0.896 | 0.992 | 0.975 | 0.915 | 0.878 | 0.718 | 0.921 | 0.813 | 0.616 | 0.729 |
| g2 s0 | 0.608 | 0.996 | 0.963 | 0.525 | 0.417 | 0.138 | 0.737 | 0.193 | 0.076 | 0.332 |
| g2 s1 | 0.665 | 0.997 | 0.970 | 0.533 | 0.401 | 0.425 | 0.735 | 0.442 | 0.680 | 0.296 |

## G2 — coverage pass@2,000 (`cov_<pool>_<arm>_s<k>.s0.jsonl`, `coverage.py --k 2000 --temperature 0.8 --seed 0 --batch 1000`)

Measured 2026-09-22 **on the original G1 / G2 Stage-1 checkpoints**, which the host cleanup deleted; the ladder
rows below are on retrained replicas of the same (set, seed). See § G5 for how far a retrain moves a number.

| arm s | depth-3 1,000 | req8 300 | reductio-req 300 (7-line stratum; ≥ 8) | per-sample d3 / req8 / red | ≥ 8-line pattern proofs d3 / req8 / red | Lean rej |
|---|---|---|---|---|---|---|
| c0 s0 | 506 (280) / 1000 | 165 (164) / 300 | 31 (31) / 300 (31; 0) | 0.0926 / 0.0155 / 0.0016 | 179 / 177 / 0 | 0 |
| c0 s1 | 423 (197) / 1000 | 113 (113) / 300 | 27 (27) / 300 (27; 0) | 0.1003 / 0.0134 / 0.0091 | 96 / 129 / 0 | 0 |
| g1 s0 | 456 (247) / 1000 | 138 (138) / 300 | 38 (38) / 300 (38; 0) | 0.1025 / 0.0172 / 0.0110 | 153 / 175 / 0 | 0 |
| g1 s1 | 603 (391) / 1000 | 259 (259) / 300 | 31 (31) / 300 (31; 0) | 0.2579 / 0.3185 / 0.0028 | 384 / 356 / 0 | 0 |
| g2 s0 | 288 (65) / 1000 | 38 (38) / 300 | 0 (0) / 300 (0; 0) | 0.1022 / 0.0059 / 0.0000 | 42 / 45 / 0 | 0 |
| g2 s1 | 266 (69) / 1000 | 23 (23) / 300 | 0 (0) / 300 (0; 0) | 0.0812 / 0.0017 / 0.0000 | 24 / 25 / 0 | 0 |

## G3 — depth-3 dial (`{ei,frozen}_d3_<arm>_s<k>/`, `expert_iter.py --rounds 4 --k 32 --temperature 0.8 --batch 768`, targets `data/p2/targets_depth3.jsonl`)

C0's rows are `artifacts/lf_control/{ei,frozen}_d3_seq_s{0,1}/`, pulled from
`hf://buckets/dan-pandori/nd-rl/lean-format/artifacts/lf/`; their `args.json` confirms `--init
ckpts/lf/stage1_a1_seq_s<k>.pt --train data/p2/train_depth3_f0_a1.jsonl`, i.e. the same model and set as this
run's C0. G1 / G2 measured 2026-09-22 on the original checkpoints.

| arm s | EI | frozen | EI − frozen | EI acq by round | held-out r4 |
|---|---|---|---|---|---|
| c0 s0 | 0.418 | 0.163 | +0.255 | 0.104, 0.362, 0.401, 0.418 | 0.956 |
| c0 s1 | 0.408 | 0.116 | +0.292 | 0.085, 0.350, 0.392, 0.408 | 0.966 |
| g1 s0 | 0.403 | 0.165 | +0.238 | 0.120, 0.341, 0.379, 0.403 | 0.947 |
| g1 s1 | 0.436 | 0.322 | +0.114 | 0.297, 0.383, 0.421, 0.436 | 0.957 |
| g2 s0 | 0.415 | 0.032 | +0.383 | 0.020, 0.261, 0.378, 0.415 | 0.702 |
| g2 s1 | 0.396 | 0.034 | +0.362 | 0.022, 0.320, 0.382, 0.396 | 0.699 |

## G4 — ladder rung T1 and frozen, 8 rounds × 32 attempts (`la_{T1,frozen}_<arm>_s<k>/`, `ladder_ei.py`, pools `data/ladder/`, byte-identical to ladder-A's; `L*` = max L with ≥ 5 transfer theorems solved at `L_true` ≥ L)

Measured 2026-09-23/24, all arms on the same two pods, same sampler settings, same 2,285-theorem transfer pool.
`la_frozen_g1_s0` was **not run** — it was the pre-registered droppable tail and the $14 budget stopped at 11 of
12 jobs (`QUESTIONS.md`, 2026-09-24 08:35 UTC).

| arm s | rung | L* | solved | 7 / 8 / 9 / 10 / 11 / 12 / 13 / 14 | textbook | schemata ≥ 5 | held-out |
|---|---|---|---|---|---|---|---|
| c0 s0 | frozen | 9 | 158 | 42 / 69 / 44 / 3 / 0 / 0 / 0 / 0 | 6 |  | 0.909 → 0.909 |
| c0 s0 | T1 | 12 | 890 | 92 / 166 / 525 / 91 / 11 / 5 / 0 / 0 | 37 | contraposition 19, export 5 | 0.909 → 0.959 |
| c0 s1 | frozen | 9 | 114 | 44 / 47 / 22 / 1 / 0 / 0 / 0 / 0 | 13 | export 7 | 0.897 → 0.897 |
| c0 s1 | T1 | 11 | 965 | 145 / 165 / 531 / 106 / 15 / 3 / 0 / 0 | 89 | contraposition 30, disjunctive_syllogism 21, export 33 | 0.897 → 0.969 |
| g1 s0 | T1 | 12 | 916 | 133 / 158 / 522 / 87 / 11 / 5 / 0 / 0 | 80 | contraposition 39, disjunctive_syllogism 38 | 0.861 → 0.954 |
| g1 s1 | frozen | 9 | 170 | 35 / 72 / 61 / 2 / 0 / 0 / 0 / 0 | 2 |  | 0.896 → 0.896 |
| g1 s1 | T1 | 10 | 635 | 73 / 141 / 369 / 49 / 2 / 1 / 0 / 0 | 23 | disjunctive_syllogism 23 | 0.896 → 0.931 |
| g2 s0 | frozen | 9 | 44 | 18 / 18 / 8 / 0 / 0 / 0 / 0 / 0 | 1 |  | 0.608 → 0.608 |
| g2 s0 | T1 | 11 | 632 | 85 / 141 / 351 / 50 / 5 / 0 / 0 / 0 | 39 | disjunctive_syllogism 38 | 0.608 → 0.694 |
| g2 s1 | frozen | 9 | 125 | 30 / 64 / 31 / 0 / 0 / 0 / 0 / 0 | 0 |  | 0.665 → 0.665 |
| g2 s1 | T1 | 11 | 607 | 43 / 142 / 364 / 53 / 3 / 2 / 0 / 0 | 0 |  | 0.665 → 0.681 |

**Textbook schemata (`found_transfer_8.jsonl` joined to the transfer pool's `schema` field, 19 schemata × 40 = 760):**

| schema | n | C0 s0 | C0 s1 | G1 s0 | G1 s1 | G2 s0 | G2 s1 |
|---|---|---|---|---|---|---|---|
| constructive_dilemma | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| contraposition | 40 | **19** | **30** | **39** | 0 | 0 | 0 |
| contraposition_conv | 40 | 1 | 1 | 0 | 0 | 0 | 0 |
| demorgan_and_to_nor | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| demorgan_nand_to_or | 40 | 4 | 2 | 2 | 0 | 0 | 0 |
| demorgan_nor_to_and | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| demorgan_or_to_nand | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| disjunctive_syllogism | 40 | 4 | **21** | **38** | **23** | **38** | 0 |
| dist_and_over_or | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| dist_and_over_or_conv | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| dist_or_over_and | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| dist_or_over_and_conv | 40 | 1 | 0 | 0 | 0 | 0 | 0 |
| excluded_middle | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| export | 40 | **5** | **33** | 0 | 0 | 1 | 0 |
| import | 40 | 1 | 0 | 1 | 0 | 0 | 0 |
| negated_conditional | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| negated_conditional_conv | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| peirce | 40 | 1 | 1 | 0 | 0 | 0 | 0 |
| peirce_sequent | 40 | 1 | 1 | 0 | 0 | 0 | 0 |
| **total / 760** | 760 | 37 | 89 | 80 | 23 | 39 | 0 |
| **schemata at ≥ 5** | 19 | 2 | 3 | 2 | 1 | 1 | 0 |

## G5 — what a retrain and a hardware change are worth (`heldout_<arm>_s<k>.json` 2026-09-22 vs `heldout2_<arm>_s<k>.json` 2026-09-23/24)

C0's two rows are the **harness control**: byte-identical checkpoint (md5 verified), different GPU class, the new
decode path. G1 / G2's rows are a genuine retrain: same set, same seed, same command line.

| arm s | held-out greedy 2026-09-22 (RTX 3090) | held-out greedy 2026-09-23 (A6000 / 4090) | delta pp |
|---|---|---|---|
| c0 s0 | 0.909 | 0.909 | +0.00 |
| c0 s1 | 0.896 | 0.897 | +0.08 |
| g1 s0 | 0.875 | 0.861 | -1.44 |
| g1 s1 | 0.940 | 0.896 | -4.44 |
| g2 s0 | 0.626 | 0.608 | -1.78 |
| g2 s1 | 0.606 | 0.665 | +5.96 |

Every delta is concentrated in the 6-line bin: C0 s0 0.686 → 0.686, G1 s0 0.521 → 0.451, G1 s1 0.849 → 0.718,
G2 s0 0.205 → 0.138, G2 s1 0.101 → 0.425.

## G6 — checker of record (`record_<arm>_s<k>.json`, the unmodified `nd2lean.py --check` + `nd_verify` over every counted proof of the dial, the ladder and the three coverage runs)

- c0: 20426 counted proofs, both accept 20426, disagreements 0
- g1: 21162 counted proofs, both accept 21162, disagreements 0
- g2: 15425 counted proofs, both accept 15425, disagreements 0

**57,013 counted proofs, 57,013 accepted by both, 0 disagreements.** `coverage.py`'s in-line Lean check also
rejected 0 of the `nd_verify`-accepted distinct proofs in all six coverage runs. `nd_verify`, `nd2lean.py` and
`lean_gate.py` are unmodified in this run (`git diff origin/main HEAD -- nd_verify/ nd2lean.py` empty).

## G7 — sampler equivalence (`regress.json`, `dsg_regress.py`, 128 samples from `data/ladder/transfer.jsonl`, T 0.8)

`path='base'` with row-keyed rng, sample.py's default as configured here, and the fast path with compaction
explicitly off: **token streams and decoded proofs identical**, verdict PASS. With **compaction on**, 1 row of 128
differs, first at decode step 162 — which is why this run sets `ND_SAMPLE_COMPACT=0` and keeps batch 512.

## G8 — cost

`podbudget ds-generator`: **21.34 pod-hours, $13.02**, of a $14 / 28 pod-hour budget. `dsg-1` RTX A6000 48 GB,
11.56 h, $5.78 (billed $0.50/h); `dsg-2` RTX 4090 24 GB, 9.78 h, $7.24 ($0.74/h). Both deleted (`podls` empty of
`dsg-*`). RTX 3090 and A40 were out of stock throughout. The 2026-09-22 phase spent ≈ $2.5 of its own $20 ceiling.

## Bucket

`hf://buckets/dan-pandori/nd-rl/ds-generator/{artifacts/dsg,ckpts/dsg,ckpts/ladder,data/dsg}` — `data/dsg`
(the two 155,000-record sets and the three raw pools), `ckpts/dsg` (four Stage-1 checkpoints), `ckpts/ladder`
(per-round ladder checkpoints), `artifacts/dsg` (everything above, incl. every per-round `found_*.jsonl`).
The per-round training mixes `la_*/mix_<r>.jsonl` (263 MB) were not kept: no counted number reads them.

# noise-floor (proposal 11, run 2) — what difference can this project resolve?

Run id `noise-floor`, branch `dan_noise-floor`. **Interim numbers are published here as each stage
lands**, because the sibling run `cap-horizon` is running now and reads this section as its yardstick.

**Model behind every row below unless stated otherwise:** 3,214,336-parameter from-scratch GPT,
`lean_seq` Lean surface format, **cap 6**, `train.py --mode lean_seq --steps 6000 --bs 128 --cap 6
--seed {0,1}` — the same model, format and schedule as `ds-generator`'s C0/G1/G2 and
`ds-composition`'s C0/A1, which is what makes this floor quotable by those runs. Checkpoints
`ckpts/nf/stage1_p<i>_s<k>.pt`, training sets `data/nf/train_p<i>.jsonl`.

## N1. The four null pools are the same distribution (premise check)

Source: `artifacts/nf/premise.json` (built by `nf_premise.py` from `artifacts/nf/{shape,overlap,
assemble,render}_p<i>.json`; those were produced on the pods by `dsg_shape.py`, `dsg_overlap.py`,
`dsg_assemble.py`, `dsg_render_check.py`, all unmodified). Generation command in `log.md`
(2026-09-24 16:03) — `make_coverage_sets.py gen` with **no knob flags**, generator seeds
21000 / 22000 / 23000 / 24000, 6,000,000 tries each; the pods log `generator knobs: None`.

| quantity | P1 | P2 | P3 | P4 | max − min | C0 published (`data/dsg/README.md`) |
|---|---|---|---|---|---|---|
| records / length 2–6 | 155,000 / 31,000 each | same | same | same | 0 | 155,000 / 31,000 each |
| box depth 0 / 1 / 2 (%) | 53.11 / 35.52 / 11.37 | 53.26 / 35.33 / 11.41 | 53.28 / 35.46 / 11.26 | 53.30 / 35.44 / 11.26 | 0.19 / 0.19 / 0.15 | 53.2 / 35.4 / 11.4 |
| box depth 3 (%) | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 (excluded by design) |
| proofs with `ORE` (%) | 1.42 | 1.48 | 1.45 | 1.46 | 0.07 | 1.46 |
| proofs with `AS` (%) | 46.89 | 46.74 | 46.72 | 46.70 | 0.19 | 46.8 |
| proofs with `IMPI` / `IMPE` (%) | 36.57 / 39.76 | 36.31 / 39.83 | 36.50 / 39.85 | 36.36 / 39.95 | 0.26 / 0.19 | 36.4 / 40.2 |
| proofs with `NEGI` / `NEGE` / `DN` (%) | 9.77 / 11.99 / 7.97 | 9.81 / 12.06 / 7.93 | 9.65 / 12.06 / 7.91 | 9.73 / 11.96 / 8.05 | 0.17 / 0.10 / 0.14 | 9.9 / 12.0 / 8.3 |
| proofs with `ANDI` / `R` / `BOTE` (%) | 22.83 / 2.75 / 2.15 | 22.76 / 2.71 / 2.18 | 22.80 / 2.82 / 2.13 | 22.82 / 2.70 / 2.12 | 0.07 / 0.12 / 0.06 | 22.9 / 2.8 / — |
| mean premises | 1.41 | 1.41 | 1.42 | 1.42 | 0.00 | 1.41 |
| contradictory-premise theorems (%) | 6.41 | 6.35 | 6.37 | 6.28 | 0.13 | 6.06 |
| reductio proofs | 9,980 | 9,973 | 9,853 | 10,072 | 219 | — |
| derived-`ORE` proofs | 69 | 71 | 74 | 76 | 7 | — |
| mean ND / Lean (`lean_seq`) tokens | 112.98 / 138.77 | 112.81 / 138.60 | 112.75 / 138.55 | 112.71 / 138.52 | 0.26 / 0.25 | — |
| fill fraction (%) | 0 | 0 | 0 | 0 | 0 | 0 |

**Verdict: all four are replicates, not arms.** Every share agrees across the four pools to
≤ 0.26 pp and with the control's published table to ≤ 0.4 pp. Pre-registered expectation **E11 is
met** (`ORE` share predicted 1.2–1.8 %, measured 1.42–1.48 %; box depth predicted within ±2 pp of
53.2 / 35.4 / 11.4, measured within ±0.2 pp). The only quantity slightly off the control is the
contradictory-premise share, 6.28–6.41 % against 6.06 % — **consistent across all four pools**, so it
is the pre-registered uncapped-pool deviation (the control's pool used `--cap_np 1500 --cap_pat 3000`
per worker), not a between-pool difference; it cannot produce a difference between P1–P4.

**Disjointness.** Against **all nine** evaluation / ladder pools used in this run
(`data/p2/heldout.jsonl`, `targets_depth3`, `transfer_depth3`, `targets_reductio_req`,
`transfer_reductio_req`, `data/r3_1/depth3_req{,_transfer}`, `data/ladder/{rl_targets,transfer}`):
**0 exact-`thm` and 0 renaming-class collisions for all four sets.** Residual, disclosed: 15–26
theorems per set match a `data/p2/heldout.jsonl` theorem only after *reordering the premises*
(0.3–0.5 % of the 5,000), identically across the four pools. Against the take-home's *other*
held-out set `data/heldout.jsonl` (**not used anywhere in this run**) there are 477–510 class
collisions per set, which is expected: that set was never excluded at assembly and this run never
evaluates on it.

**Pairwise renaming-class overlap between the four null sets: 5.8–6.0 %** (|A ∩ B| / 155,000).
Pre-registered expectation **E14 (30–70 %) is a miss, in the harmless direction** — the ≤ 6-line
class universe is far larger than I assumed, so the four pools are nearly disjoint draws and the
between-pool component below is not diluted by shared records.

**Render check** (`dsg_render_check.py`, per set): 3,000 / 3,000 ND → Lean → ND round-trips
identical, 3,000 / 3,000 denoted proofs re-verified by `nd_verify` at cap 6, 1,000 / 1,000 Lean
positives accepted, 300 / 300 theorem-swapped Lean negatives rejected — on every one of the four sets.

## N2. Held-out greedy — the eight null cells (source: `artifacts/nf/heldout_p<i>_s<k>.{json,jsonl}`, `artifacts/nf/summary.json` § `floors`)

Measured on `ckpts/nf/stage1_p<i>_s<k>.pt` — 3,214,336-parameter from-scratch GPT, `lean_seq`, cap 6,
trained on `data/nf/train_p<i>.jsonl`. Command: `eval_set.py --in data/p2/heldout.jsonl --k 1
--temperature 0 --batch 512` (5,000 theorems), identical in every cell.

| quantity | P1 s0 | P1 s1 | P2 s0 | P2 s1 | P3 s0 | P3 s1 | P4 s0 | P4 s1 | mean | **sd** | max/min | **MDD at n = 2** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 0.8726 | 0.9362 | 0.9458 | 0.8770 | 0.8680 | 0.8858 | 0.8716 | 0.9630 | 0.9025 | **0.0390** | 1.11× | **±20.9 pp** |
| 2-line bin | 0.9970 | 0.9930 | 0.9990 | 1.0000 | 0.9970 | 0.9940 | 0.9980 | 0.9980 | 0.9970 | 0.0024 | 1.01× | ±1.3 pp |
| 3-line bin | 0.9900 | 0.9810 | 0.9920 | 0.9930 | 0.9910 | 0.9910 | 0.9800 | 0.9910 | 0.9886 | 0.0051 | 1.01× | ±2.7 pp |
| 4-line bin | 0.9560 | 0.9450 | 0.9700 | 0.9650 | 0.9530 | 0.9660 | 0.9600 | 0.9680 | 0.9604 | 0.0086 | 1.03× | ±4.6 pp |
| 5-line bin | 0.9330 | 0.9120 | 0.9560 | 0.9190 | 0.9260 | 0.9220 | 0.9260 | 0.9380 | 0.9290 | 0.0135 | 1.05× | ±7.3 pp |
| **6-line bin** | 0.4870 | 0.8500 | 0.8120 | 0.5080 | 0.4730 | 0.5560 | 0.4940 | 0.9200 | 0.6375 | **0.1887** | **1.95×** | **±101 pp (unresolvable)** |
| **depth-3 slice (n = 500)** | 0.0860 | 0.8180 | 0.7060 | 0.1060 | 0.0400 | 0.2040 | 0.0900 | 0.9180 | 0.3710 | **0.3740** | **22.9×** | **±2.01 absolute (unresolvable)** |
| 6-line, no pattern (n = 247) | 0.8300 | 0.7814 | 0.8623 | 0.8421 | 0.8340 | 0.8502 | 0.8300 | 0.8664 | 0.8370 | 0.0265 | 1.11× | ±14.2 pp |

**Pre-registered expectations scored.** **E1 missed** — predicted sd ≤ 0.02 and max − min ≤ 5 pp;
measured **sd 0.039, range 9.5 pp**, roughly twice the predicted floor. **E2 missed in the same
direction** — 6-line bin predicted sd 0.04–0.10, measured **0.189**. **E3 met** — 2-line bin sd
0.0024 ≤ 0.005. **E4 partly met** — the depth-3 slice is bimodal and its sd is 0.374 ≥ 0.20 as
predicted, but only **3 of 8** cells are in the high mode against a predicted 5–8; high-mode
proportion **0.375, Wilson 95 % [0.137, 0.694]**.

**The bimodality is one phenomenon, not two.** All 500 depth-3 held-out theorems are 6-line
theorems (`data/p2/heldout.jsonl`: depth-3 by length = {6: 500} of the 1,000 6-line theorems), and
across the eight null cells the depth-3 slice and the 6-line bin correlate at **r = 0.9994**.
`ds-rendering`'s "bimodal depth-3 cell" and `ds-generator`'s "retrain spread is almost all in the
6-line bin" are the same thing seen twice: **a 6-line-bin mode that a single Stage-1 training run
lands in or does not.** Cells split 3 high / 4 low / 1 between on `ds-rendering`'s own boundaries.

**It is neither the pool nor the seed — it is the individual training run.** Two-way decomposition
(4 pools × 2 seeds, one observation per cell; the interaction is confounded with error):

| quantity | MS_pool | MS_seed | MS_resid | var_pool | var_seed | var_resid |
|---|---|---|---|---|---|---|
| overall | 0.000638 | 0.001352 | **0.002458** | −0.00091 | −0.00028 | **0.00246** |
| 6-line bin | 0.01428 | 0.04033 | **0.05532** | −0.0205 | −0.0037 | **0.0553** |
| depth-3 slice | 0.05832 | 0.1579 | **0.2154** | −0.0786 | −0.0144 | **0.2154** |

Both between-pool and between-seed components come out **negative** (i.e. indistinguishable from
zero) on every held-out quantity: within a pool, seed 0 and seed 1 land in *different* modes
(P1 0.487 / 0.850, P2 0.812 / 0.508, P4 0.494 / 0.920), and P3 is low on both. **E12's premise is
wrong for held-out accuracy in a way that matters: re-drawing the data set buys you no more
independence than re-seeding Stage-1, and neither factor explains the spread — the residual does.**
The practical consequence is that an arm's two seeds are two coin flips on the 6-line mode, so a
two-seed held-out comparison is a comparison of two coin flips.

## N3. Frozen ladder — the eight null cells (source: `artifacts/nf/la_frozen_p<i>_s<k>/round_8.json`)

Measured on `ckpts/nf/stage1_p<i>_s<k>.pt` (3,214,336-parameter `lean_seq` GPT, cap 6, trained on
`data/nf/train_p<i>.jsonl`). Command, identical in every cell and identical to `ds-generator`'s and
`ds-composition`'s: `ladder_ei.py --no_train --rounds 8 --k 32 --temperature 0.8 --batch 512
--max_new 512` on `data/ladder/` (4,495 RL targets, 2,285 transfer theorems), `ND_SAMPLE_COMPACT=0`.
A proof counts only if `nd_verify` **and** Lean accept.

| cell | P1 s0 | P1 s1 | P2 s0 | P2 s1 | P3 s0 | P3 s1 | P4 s0 | P4 s1 | mean | sd | max/min | **MDD at n = 2** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **transfer solved / 2,285** | 117 | 262 | 200 | 96 | 62 | 174 | 177 | 265 | 169.1 | **74.1** | **4.27×** | **±397 (±235 %)** |
| transfer `L*` | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **10** | 9.125 | **0.354** | 1.11× | **±1.9 points** |
| RL targets solved / 4,495 | 1,036 | 1,715 | 1,512 | 823 | 714 | 1,298 | 1,187 | 1,765 | 1,256 | 391 | 2.47× | ±2,098 (±167 %) |

**This is the run's headline.** Eight models that differ only in a generator RNG seed and a Stage-1
seed span **62 to 265 solved transfer theorems, a 4.27× range**, sd = 43.8 % of the mean. The
control's own published frozen ladder — **158 (s0) and 114 (s1)**, `ckpts/lf/stage1_a1_seq_s{0,1}.pt`
— sits in the middle of that range, and so does **every** frozen-ladder value proposal 10 reported:
`ds-generator` G2 44 / 125 and G1 170, `ds-composition` A1 205 and A2 111. Pre-registered
**E8 missed** (max/min predicted 1.4–2.5×, measured **4.27×**; sd ≥ 25 % of the mean **met** at
43.8 %) and **E10 missed by ≈ 4×** (n = 2 resolvable difference predicted ±40–60 %, measured
**±235 %**).

**`L*` is the robust readout; the solve count is not.** Seven of the eight cells give `L*` = 9 and
one gives 10, sd 0.354, while the underlying solve count moves 4.27×. **E9 missed in the favourable
direction** (predicted 9–11, i.e. 2 points of pure noise; measured 9–10, **1 point**). The
resolvable `L*` difference at n = 2 is **±1.9 points**, so a 1-point `L*` difference is noise and a
2-point one is at the edge of resolution.

**Again the spread is the individual training run, not the pool or the seed** (4 × 2 unreplicated
decomposition): MS_pool 4,111, MS_seed 7,260, MS_resid **6,268** → var_pool **−1,079**, var_seed
**+248**, var_resid **6,268**. E12's ordering (σ²_seed ≥ σ²_pool) holds in sign but both components
are negligible beside the residual: **re-drawing the data set buys no more independence than
re-seeding Stage-1.** Bimodality coefficient 0.546 (< 5/9), so unlike the held-out 6-line bin the
frozen ladder is not bimodal — it is unimodal and very wide.

**What an n = 2 comparison of frozen-ladder solves can say.** Nothing below a **2.4× ratio**
(± 235 % of the mean). And non-parametrically, nothing at all: a two-sided exact permutation test at
2 vs 2 has minimum attainable p = **1/3**.

## N4. Coverage pass@2,000 — the eight null cells (source: `artifacts/nf/cov_{red,req8}_p<i>_s<k>.s0.jsonl`)

Measured on `ckpts/nf/stage1_p<i>_s<k>.pt` (3,214,336-parameter `lean_seq` GPT, cap 6). Command,
identical in every cell: `coverage.py --k 2000 --temperature 0.8 --seed 0 --batch 1000 --procs 2`.
A sample counts only if `nd_verify` **and** Lean accept. `targets_depth3` coverage was **dropped**
before it ran (the pre-registered first drop; see `log.md` 2026-09-24 17:45).

| pool (n targets) | P1 s0 | P1 s1 | P2 s0 | P2 s1 | P3 s0 | P3 s1 | P4 s0 | P4 s1 | mean | sd | max/min | **MDD at n = 2** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `targets_reductio_req` (300) | 31 | 19 | 39 | 43 | **6** | **46** | 34 | 27 | 30.6 | 13.2 | **7.67×** | **±71 (±232 %)** |
| `r3_1/depth3_req`, required@8 (300) | 89 | 242 | 205 | **61** | 71 | 172 | 144 | 238 | 152.8 | 73.3 | **3.97×** | **±393 (±257 %)** |

**E5 and E6 both missed, again with the floor far larger than predicted** (both predicted a max/min
ratio of 1.5–3×; measured **7.67×** and **3.97×**). The `redreq` pool is the worse of the two because
its counts are small: one null cell solves **6** of 300 and another **46**, and the control's own
published values (**31 / 27**) sit in the middle.

The published contrasts, against this floor:

| standing comparison | original | inside the null range? |
|---|---|---|
| `ds-generator` G1 s1 vs C0 s1, required@8 | 259 vs 113 (2.3×) | 259 is just outside the null range 61–242, but the ratio 2.29× is below the MDD's 3.57× — and G1 **is** a null arm, so this is the floor measuring itself |
| `ds-generator` G1 s0 vs C0 s0, required@8 | 138 vs 165 | **yes** |
| `ds-composition` A1 vs C0, `redreq` | 16 / 18 vs 28 / 26 | **yes** — the null range is 6–46 |
| `ds-composition` A2 s1 vs C0 s1, `redreq` | 4 vs 26 | **yes** |
| `ds-generator` G2 vs C0, `redreq` | **0 / 0** vs 31 / 27 | **zero on both seeds is below the null minimum of 6** — outside the null range; the ratio is infinite but the *difference* (30.6) is below the MDD of 71 |
| `ds-composition` A3 (cap 8) vs C0, `redreq` | 54 / 72 vs 28 / 26 | **both seeds are above the null maximum of 46**, but the ratio 2.33× is below the MDD's 3.32× — "outside the null range, but below the MDD" (see § N6 for the two criteria) |

**And again it is the individual training run.** `redreq`: var_pool **−84**, var_seed **−49**,
var_resid **276**. `required@8`: var_pool **−3,401**, var_seed **−899**, var_resid **8,796**. Both
between-factor components are indistinguishable from zero on both pools.

## N5. Gap-closer 1 — `ds-generator`'s missing `la_frozen_g1_s0` (source: `artifacts/nf/la_frozen_dsg_g1_s0/round_8.json`)

Model: `ckpts/dsg/stage1_g1_s0.pt` from `hf://buckets/dan-pandori/nd-rl/ds-generator/ckpts/dsg/` —
3,214,336-parameter `lean_seq` GPT, cap 6, trained on `ds-generator`'s **G1** set
(`gen.py --ore_steps 3 --ore_boxes`, a different generator flag set from P1–P4, shown by that run to
match the control's shape table to sampling noise). **Not retrained.** Same `ladder_ei` command,
same pools, same `--batch 512` as every cell above; run here on an A40.

| cell | transfer solved / 2,285 | transfer `L*` | RL targets solved / 4,495 |
|---|---|---|---|
| G1 seed 0 (**this run**) | **62** | 9 | 612 |
| G1 seed 1 (`ds-generator`) | 170 | 9 | — |

`ds-generator` reported the seed-1 value, 170, as *"larger than either C0 seed"* and flagged the
missing seed-0 cell as *"the run's most informative noise-floor cell is a one-seed estimate."* It is
now two seeds: **62 and 170, a 2.74× difference inside a single arm** whose data set is the
control's distribution by construction. G1's own two seeds **bracket C0's 158 / 114 and G2's
44 / 125 entirely**, which is the cleanest single demonstration in this project that the
frozen-ladder solve count cannot separate arms at two seeds. 62 is also exactly the minimum of my
own eight null cells.

Pooling G1's two seeds with the eight null cells gives **ten independent draws of the control's
distribution**: 62, 62, 96, 117, 170, 174, 177, 200, 262, 265 — mean 158.5, sd 73.6, **max/min
4.27×**. (Reported separately as well as pooled, because G1 is a different generator flag set and
its two cells were trained on different hardware; the pooled figures move the sd by under 2 %.)

## N6. Gap-closer 2 — `ds-composition`'s A1 ladder at Stage-1 seed 1 (source: `artifacts/nf/la_{T1,frozen}_dsc_a1_s1/round_8.json`)

Model: `ckpts/dsc/stage1_a1_s1.pt` from `hf://buckets/dan-pandori/nd-rl/ds-composition/ckpts/dsc/`,
training set `data/dsc/train_a1.jsonl` from the same bucket — 3,214,336-parameter `lean_seq` GPT,
cap 6, A1 = the pool's own length histogram. **Not retrained.** Same `ladder_ei` command, same pools,
same `--batch 512`. C0 seed 1 was **not** re-run: `ds-generator` measured the identical checkpoint
`ckpts/lf/stage1_a1_seq_s1.pt` with the identical command (T1 **965** / `L*` 11, frozen **114** /
`L*` 9).

| | C0 s0 | C0 s1 | A1 s0 | **A1 s1 (this run)** |
|---|---|---|---|---|
| T1 transfer solved / 2,285 | 856 *(`ds-composition`)* / 890 *(`ds-generator`)* | 965 | 965 | **847** |
| T1 transfer `L*` | 12 | 11 | 11 | **11** |
| frozen transfer solved | 158 | 114 | 205 | **194** |
| frozen transfer `L*` | 9 | 9 | 10 | **9** |
| frozen RL targets solved / 4,495 | — | — | — | **1,456** |

`ds-composition`'s **entire ladder was Stage-1 seed 0** and carried three headline claims. With seed 1
in hand:

1. **"A1's T1 ladder beats C0" (965 vs 856) — dead.** It reverses on seed 1: **A1 847 vs C0 965**.
   The sign is not stable across seeds, which is the strongest form of failure available.
2. **"A1's frozen ladder beats C0" (205 vs 158) — same sign on both seeds** (+30 % on s0, +70 % on
   s1) **but inside this run's floor**: the null cells span 62–265 and the MDD is 3.35×, against
   A1/C0 ratios of 1.30× and 1.70×. Reportable only as "consistent in sign at n = 2, not resolvable".
3. **"A1's frozen `L*` is 10 vs C0's 9" — dead.** On seed 1 both are 9. The `L*` floor is ±1.9 points.

**A free third measurement of harness noise.** C0 seed 0's frozen ladder was measured at **158** by
`ds-composition` and at **158** by `ds-generator`, on different pods and different GPU classes, from
the byte-identical checkpoint and the identical command — and its T1 ladder at **856** and **890**
(+4.0 %). So the *sampler and checker* contribute ≈ 0 to the frozen ladder and ≈ 4 % to a ladder
that trains; the 4.27× spread in § N3 is all data-draw + Stage-1-draw, i.e. all training.

## N7. Held-out greedy at 52 cells (4 pools × 13 Stage-1 seeds; pre-registration addendum 2)

Source: `artifacts/nf/heldout_p<i>_s<k>.{json,jsonl}` for `k` = 0…12, `artifacts/nf/summary.json`
§ `floors`. Models `ckpts/nf/stage1_p<i>_s<k>.pt`, all 52 the same 3,214,336-parameter `lean_seq`
GPT at cap 6, `train.py --steps 6000 --bs 128`, differing only in pool and Stage-1 seed. Held-out
command identical in every cell.

| quantity | cells | mean | sd | min | max | max/min | **MDD at n = 2** |
|---|---|---|---|---|---|---|---|
| overall | 52 | 0.9072 | **0.0304** | 0.8420 | 0.9630 | 1.14× | **±16.3 pp** |
| 2-line bin | 52 | 0.9963 | 0.0031 | 0.9870 | 1.0000 | 1.01× | ±1.7 pp |
| 3-line bin | 52 | 0.9883 | 0.0049 | 0.9760 | 0.9970 | 1.02× | ±2.6 pp |
| 4-line bin | 52 | 0.9567 | 0.0151 | 0.8920 | 0.9790 | 1.10× | ±8.1 pp |
| 5-line bin | 52 | 0.9272 | 0.0150 | 0.8770 | 0.9560 | 1.09× | ±8.1 pp |
| **6-line bin** | 52 | 0.6674 | **0.1523** | 0.4210 | 0.9200 | **2.19×** | **±81.7 pp (unresolvable)** |
| **depth-3 slice (500)** | 52 | 0.4402 | **0.3053** | **0.0080** | **0.9180** | **114.8×** | **±1.64 (unresolvable)** |
| 6-line, no pattern (247) | 52 | 0.8240 | 0.0367 | 0.6964 | 0.8785 | 1.26× | ±19.7 pp |

The eight pre-registered cells give sd 0.0390 on the overall rate; 52 cells give **0.0304**, so the
n = 8 figure already published in § N2 was ≈ 28 % high but the same size. **Nothing quoted from § N2
moves materially**; the ± 20.9 pp resolvable difference there becomes **± 16.3 pp** here.

**The depth-3 cell is bimodal, but less cleanly than `ds-rendering` at n = 26 suggested.** On that
run's own boundaries: **24 of 52 high (> 0.44), 13 low (< 0.11), 15 (28.8 %) in the gap**, bimodality
coefficient 0.697 (> 5/9). The right summary is the **probability a training run lands in the high
mode: 0.462, Wilson 95 % [0.333, 0.595]** — a coin flip, and now pinned to ±0.13 instead of ±0.28.

| pool | cells | high (> 0.44) | low (< 0.11) | in the gap | p(high) |
|---|---|---|---|---|---|
| P1 | 13 | 5 | 4 | 4 | 0.385 |
| P2 | 13 | 8 | 3 | 2 | 0.615 |
| P3 | 13 | 6 | 4 | 3 | 0.462 |
| P4 | 13 | 5 | 2 | 6 | 0.385 |

**Variance components at 52 cells** (balanced 4 × 13, one observation per cell):

| quantity | MS_pool | MS_seed | MS_resid | var_pool | var_seed | var_resid |
|---|---|---|---|---|---|---|
| overall | 2.74e−4 | 1.08e−3 | **9.23e−4** | **−5.0e−5** | +4.0e−5 | **9.23e−4** |
| 6-line bin | 6.65e−3 | 2.84e−2 | **2.28e−2** | **−1.2e−3** | +1.4e−3 | **2.28e−2** |
| depth-3 slice | 2.73e−2 | 1.19e−1 | **9.03e−2** | **−4.8e−3** | +7.1e−3 | **9.03e−2** |

With 3 and 12 degrees of freedom instead of 3 and 1, the answer does not change and is now firm:
**the pool contributes nothing measurable and the seed almost nothing; ≈ 99 % of the variance is the
individual training run.** Re-drawing the data set is not a stronger replicate than re-seeding
Stage-1 — a cheaper standard for future runs than the one proposal 11 feared.

## N8. Checker of record

Lean is the checker of record: a proof counts here only if the **unmodified** `nd2lean.py --check`
(official translation + Lean) **and** `nd_verify` accept it. `git diff origin/dan_ds-generator` is
empty for `nd_verify`, `nd2lean.py`, `lean_gate.py`, `train.py`, `eval_set.py`, `coverage.py`,
`ladder_ei.py`, `expert_iter.py`, `sample.py`, `model.py`, `prune.py`, `patterns.py`, `gen.py`,
`make_coverage_sets.py`, `dsg_assemble.py`.

| pass | proofs | both accept | `nd_verify`-only | **Lean-only** | source |
|---|---|---|---|---|---|
| ladder + coverage, every counted proof | 28,737 | **28,737** | 0 | 0 | `artifacts/nf/record_nf1.json`, `record_nf2.json` |
| held-out greedy, 200 sampled per cell × 52 | 10,400 | **10,400** | 0 | 0 | `artifacts/nf/record_heldout.json` (26 cells, Lean 4.34.0 on `nf-1`), `record_heldout_vps.json` (26 cells, Lean **4.34.1** on the VPS) |
| **total** | **39,137** | **39,137** | **0** | **0** | |

**0 disagreements on 39,137 counted proofs**, including 5,200 re-checked on a *different Lean patch
version* (4.34.1) from the one the pods ran (4.34.0) — disclosed, and it agrees.

**The in-loop gate's own rate, which `ds-generator`'s review asked every run to report** (source:
`artifacts/nf/summary.json` § `gate`, aggregated from all 408 `generate()`-call records in
`artifacts/nf/gate_*.jsonl` pulled back from both pods; the `_gate` blocks inside
`record_nf1.json` / `record_nf2.json` are the same count taken before the last few jobs finished):

| samples gated | parse failures | distinct checked by both | both accept | `nd_verify`-only | **Lean-only** |
|---|---|---|---|---|---|
| 20,010,560 | 6,500,782 | 12,033,100 | 1,977,301 | **0** | **808** |

**67.1 Lean-only acceptances per million distinct samples, 0 the other way** — one-directional, as in
every run that has measured it, and an order of magnitude below `ds-generator`'s 718 per million on
its heavier-premise workload (`lean-format` measured 33 per million). No counted proof is affected,
because acceptance is the conjunction. The individual disagreements are in
`artifacts/nf/gate_*.disagree.jsonl`.

## N9. Pre-registered expectations, scored

Every expectation from `preregistration/noise-floor.md` and its two addenda, scored against the
files above. The pattern is one-directional and is the run's second headline: **almost every miss is
the floor being larger than I predicted.**

| # | expectation | measured | verdict |
|---|---|---|---|
| E1 | held-out overall sd ≤ 0.02, max − min ≤ 5 pp | sd **0.0304** (0.039 at n = 8), range **12.1 pp** | **miss — floor ≈ 2× larger** |
| E2 | held-out 6-line bin sd 0.04–0.10 | **0.152** (0.189 at n = 8) | **miss — larger** |
| E3 | held-out 2-line bin sd ≤ 0.005 | 0.0031 | met |
| E4 | depth-3 slice bimodal; 5–8 of 8 high; sd ≥ 0.20 | bimodal (b = 0.697); **3 of 8** high; sd 0.305 | partly met — bimodal and wide, but fewer in the high mode |
| E5 | `redreq` max/min 1.5–3× | **7.67×** | **miss — larger** |
| E6 | `required@8` max/min 1.5–3× | **3.97×** | **miss — larger** |
| E7 | `targets_depth3` max/min 1.2–2× | **not run** (pre-registered first drop; see `log.md` 17:45) | not scored |
| E8 | frozen ladder max/min 1.4–2.5×; sd ≥ 25 % of mean | **4.27×**; sd 43.8 % | **ratio miss — larger**; sd met |
| E9 | frozen ladder `L*` spread 9–11 (2 points of noise) | **9–10 (1 point)** | **miss — smaller, the favourable direction** |
| E10 | n = 2 resolvable frozen-ladder difference ± 40–60 % | **± 235 %** | **miss — ≈ 4× larger** |
| E11 | pool shape tables agree; `ORE` 1.2–1.8 %; box depth within ±2 pp | all shares within 0.26 pp of each other; `ORE` 1.42–1.48 %; box depth within 0.2 pp | met |
| E12 | σ²_seed ≥ σ²_pool on the frozen ladder | var_seed **+248**, var_pool **−1,079** | met in sign, but both ≈ 0 against var_resid 6,268 |
| E13 | 0 Lean-vs-`nd_verify` disagreements on counted proofs; gate 0–1,000 per million, one-directional | **0 / 39,137**; gate **66.1 per million**, 789 Lean-only, 0 the other way | met |
| E14 | inter-pool renaming-class overlap 30–70 % | **5.8–6.0 %** | **miss — the pools are far more independent than assumed** |
| E15 | depth-3 high-mode proportion 0.30–0.55; Wilson half-width ≤ 0.15 | **0.462**, half-width **0.131** | met |
| E16 | ≤ 25 % of the 52 cells in the gap 0.11–0.44 | **28.8 %** (15 of 52) | narrow miss |
| E17 | held-out overall sd at 52 cells 0.030–0.050 | **0.0304** | met |
| E18 | var_pool ≤ 20 % of var_resid for held-out overall | var_pool **negative** (−5.0e−5 vs 9.23e−4) | met |
| E19 | the four pools' high-mode proportions within ±0.20 | 0.385 / 0.615 / 0.462 / 0.385 — spread **0.23** | narrow miss |

**The falsifier did not fire, decisively.** It required the frozen ladder's max/min to be **under
1.2×** across eight null cells, which would have meant `ds-generator`'s G1-vs-C0 gap was a real
effect of `--ore_steps 3 --ore_boxes` and would have reopened the shape account. Measured:
**4.27×** — more than three times the *upper* end of the pre-registered band, and 3.6× the
falsifier's threshold. The shape account stays dead.

## N10. The retrospective table (source: `nf_retro.py`, which reads `artifacts/nf/summary.json`)

Two criteria, both shown, because they disagree on four rows:

- **MDD** — the smallest difference a two-sample t-test at *n* seeds per arm has 80 % power to
  detect, `(t_{.975,2n-2} + t_{.80,2n-2}) · s · √(2/n)`, with `s` the pooled sd over the null cells.
  This is the **parametric best case**: at 2 vs 2 the exact two-sided permutation test cannot reach
  p < 0.05 at all (minimum attainable p = **1/3**).
- **null range** — does the arm's value fall outside the interval the null cells span? Weaker (one
  value outside the range of eight null draws is only about p = 0.22 one-sided) but distribution-free
  and it is the question a reader actually asks.

A finding is scored **survives** only if it clears the MDD.

| quantity | run | standing finding | original | null cells | MDD | verdict |
|---|---|---|---|---|---|---|
| `ladder_frozen_transfer_solved` | ds-composition | A3 (cap 8) frozen ladder beats C0 | 976 vs 158 (6.18×) | 62–265 | ±397 (3.35×) | **survives** |
| `ladder_frozen_transfer_solved` | ds-generator | G1 frozen ladder is above both C0 seeds | 170 vs 158/114 (1.25×) | 62–265 | ±397 (3.35×) | **inside the floor** |
| `ladder_frozen_transfer_solved` | ds-composition | A1 frozen ladder beats C0 (n = 1) | 205 vs 158 (1.30×) | 62–265 | ±397 (3.35×) | **inside the floor** |
| `ladder_frozen_transfer_solved` | ds-composition | A2 frozen ladder is below C0 (n = 1) | 111 vs 158 (1.42×) | 62–265 | ±397 (3.35×) | **inside the floor** |
| `ladder_frozen_transfer_solved` | ds-generator | G2 frozen ladder is above C0 on seed 1 | 125 vs 114 (1.10×) | 62–265 | ±397 (3.35×) | **inside the floor** |
| `ladder_frozen_transfer_solved` | ds-generator | G2 frozen ladder is far below C0 on seed 0 | 44 vs 158 (3.59×) | 62–265 | ±397 (3.35×) | outside the null range, but below the MDD |
| `ladder_frozen_transfer_lstar` | ds-composition | A3 (cap 8) frozen L* 11 vs C0 9 | 11 vs 9 (+2) | 9–10 | ±1.9 | **survives** |
| `ladder_frozen_transfer_lstar` | ds-composition | A1 frozen L* 10 vs C0 9 (n = 1) | 10 vs 9 (+1) | 9–10 | ±1.9 | **inside the floor** |
| `ladder_frozen_transfer_lstar` | lean-format | lean_seq ladder L* 11 vs token 10 | 11 vs 10 (+1) | 9–10 | ±1.9 | outside the null range, but below the MDD |
| `cov_red_solved` | ds-composition | A3 (cap 8) solves more `redreq` than C0 | 54/72 vs 28/26 (2.33×) | 6–46 | ±71 (3.32×) | outside the null range, but below the MDD |
| `cov_red_solved` | ds-composition | A1's reductio deficit vs C0 | 16/18 vs 28/26 (1.59×) | 6–46 | ±71 (3.32×) | **inside the floor** |
| `cov_red_solved` | ds-generator | G2 writes no `redreq` proof at all | 0/0 vs 31/27 (∞) | 6–46 | ±71 (3.32×) | outside the null range, but below the MDD |
| `cov_req8_solved` | ds-generator | G1 s1 required@8 is 2.3x C0 s1 | 259 vs 113 (2.29×) | 61–242 | ±393 (3.57×) | outside the null range, but below the MDD |
| `heldout_overall` | lean-format | lean_seq held-out beats the token format | 90.3 vs 88.3 (+2.0 pp) | 84.2–96.3 | ±16.3 pp | **inside the floor** |
| `heldout_overall` | ds-composition | A1 held-out beats C0 | 93.0 vs 90.3 (+2.7 pp) | 84.2–96.3 | ±16.3 pp | **inside the floor** |
| `heldout_overall` | ds-composition | A3 (cap 8) held-out beats C0 | 95.2 vs 90.3 (+5.0 pp) | 84.2–96.3 | ±16.3 pp | **inside the floor** |
| `heldout_len6_nopattern` | ds-composition | A1's 6-line no-pattern gain over C0 | 89.5 vs 82.6 (+6.9 pp) | 69.6–87.9 | ±19.7 pp | outside the null range, but below the MDD |
| `heldout_len6_nopattern` | ds-composition | A2's 6-line no-pattern deficit vs C0 | 77.5 vs 82.6 (-5.1 pp) | 69.6–87.9 | ±19.7 pp | **inside the floor** |
| `heldout_depth3_slice` | ds-rendering | R2 depth-3 held-out beats C0 (n = 6 each) | 66.1 vs 53.3 (+12.8 pp) | 0.8–91.8 | ±54.8 pp (n = 6) | **inside the floor** |
| `heldout_len6` | ds-generator | G2 6-line held-out is far below C0 | 28.1 vs 63.5 (-35.4 pp) | 42.1–92.0 | ±81.7 pp | **inside the floor** |

## N11. Cost, pods and artifacts

| pod | GPU | created | deleted | pod-hours | billed |
|---|---|---|---|---|---|
| `nf-1` (`kmol2dhu6iz9sw`) | NVIDIA A40 48 GB | 2026-09-24 16:05:45 UTC | 2026-09-25 07:05 UTC | 14.98 | $7.34 |
| `nf-2` (`a2vq56qgrp9e2z`) | NVIDIA A40 48 GB | 2026-09-24 16:06:22 UTC | 2026-09-25 05:26 UTC | 13.34 | $6.54 |
| **total** | | | | **28.32** | **$13.88** |

Real billed rate **$0.49/h**, read from `runpodctl pod list`'s `costPerHr` for both pods — which is
also what `podbudget` assumes for an A40, so for this run its dollar column is right (it is **not**
for A6000 or 4090). Ceiling as run: **42 h / $21** (raised from the brief's 36 h / $18 at 17:40 on
2026-09-24; `QUESTIONS.md`). RunPod balance after: **$151.21**, above proposal 11's $130 floor.

**Measured unit costs on an A40 at four concurrent jobs** (for future sizing — the brief's estimates
were low by 2–7×):

| job | cost |
|---|---|
| Stage-1, 6,000 steps, bs 128 | ≈ 5.3 pod-minutes |
| held-out greedy, 5,000 theorems, k = 1 | ≈ 1.2 pod-minutes |
| frozen ladder, 8 rounds × k 32 on `data/ladder/` | **≈ 1.94 pod-hours** |
| `pass@2,000` coverage, 300 targets | ≈ 0.7 pod-hours |
| `pass@2,000` coverage, 1,000 targets | ≈ 2.4 pod-hours |

Bucket: `hf://buckets/dan-pandori/nd-rl/noise-floor/{ckpts,artifacts,data}` —
`ckpts/nf/stage1_p<i>_s<k>.pt` (52), `artifacts/nf/` (per-cell held-out, coverage and ladder outputs,
`summary.json`, `premise.json`, the `record_*.json` checker passes, the `gate_*.jsonl` logs and their
`.disagree.jsonl`), `data/nf/train_p<i>.jsonl` (4 × 155,000) and `assemble_p<i>.json`.

---

# § lean-judge (2026-09-27) — Lean alone decides

Every count below is reproducible from files pulled back into `artifacts/lj/` (and the bucket) with the
commands named. The corpora are built by `python3 lj_regress.py collect` from the other runs' artifacts.

**Checker labels.** Stored verdicts in the source runs were produced under **Lean ∧ `nd_verify`**
(pre-2026-09-27). Every verdict computed in this run is under **Lean alone**. Lean versions: **4.34.1**
on the VPS, **4.34.0** on pod `lj1` (the version the stored data was produced under); both give the same
verdicts on every case measured here (`lean_check --selftest` 39/39 on both).

**Model labels.** No model was trained in this run. The numbers in L1–L4, L6, L8 are properties of
**stored samples** from runs `cap-horizon`, `noise-floor`, `ds-rendering`, `efficiency`, `lean-format` —
all **`lean_seq`**, **from-scratch**, ~19 M parameters, trained on this project's Stage-1 Lean-format
pools. L5, L7 are measured on **`ckpts/dsc/stage1_a3_s0.pt`** (run `ds-composition`, arm a3, seed 0:
`lean_seq`, from-scratch, ~19 M parameters, Stage-1 pool `train_a3`), pod `lj1` (NVIDIA A40, $0.49/h).

## L0 — the corpora

| quantity | value | source |
|---|---|---|
| `found*.jsonl` records read, Lean-gated arms (1,000 files) | 1,533,741 | `artifacts/lj/collect.json` |
| collected distinct `(prompt, proof)` pairs | 283,012 (282,393 from `found*`, 619 from 34 Lean-gated coverage files) | `artifacts/lj/collect.json`, `artifacts/lj/corpus_accepted.jsonl` (bucket only, 165 MB) |
| of those, **not ND proofs**: run `efficiency` (`artifacts/lo`) stores the **literal Lean text** in the `proof` field of its `found*.jsonl`, in both its free-form and its `lean_seq` arms | 1,195 (of 3,167 records from that directory; the other 1,972 are ND) | `artifacts/lj/t2_judge.rejected.jsonl` — all 1,195 rejected with `nd2lean: QED`, i.e. the string is not an ND proof |
| **distinct ND proofs the old gate counted** (the test-2 corpus) | **281,817** | `artifacts/lj/t2_judge.json` |
| `*.disagree.jsonl` records read (398 files) | 29,070, all `nd_rej & lean_ok` (0 `nd_ok & lean_rej`) | `artifacts/lj/collect.json` |
| distinct Lean-only records | **8,514** — 6,419 with an ND denotation, 2,095 without | `artifacts/lj/corpus_leanonly.jsonl` |

## L1 — acceptance test 1: no judging path calls `nd_verify`

26/26 checks pass, on the VPS and on pod `lj1`. Detection is by `ast`, not grep, over
`expert_iter.py`, `ladder_ei.py`, `coverage.py`, `eval_set.py`, `eval_targets.py`, `grpo.py`,
`lean_gate.py`, `lean_judge.py`: **0** imports of or attribute accesses on `nd_verify`.
Source: `python3 tests/test_lean_only_judge.py`; pod copy `artifacts/lj/setup.log`.

## L2 — acceptance test 2: zero regressions

| quantity | value | source |
|---|---|---|
| records judged | 283,012 | `artifacts/lj/t2_judge.json` |
| accepted by the new judge | **281,817** | same |
| rejected, all with reason `nd2lean: QED` (not ND proofs — see L0) | 1,195 | same, `artifacts/lj/t2_judge.rejected.jsonl` |
| Lean × `nd_verify` table | `nd=True,lean=True` **281,817**; `nd=False,lean=False` **1,195**; `nd=True,lean=False` **0**; `nd=False,lean=True` **0** | same, key `table_nd_lean` |
| **losses** (old gate counted, new judge did not) | **0** | the `nd=True, lean=False` cell |
| acceptance rate on the ND corpus | **281,817 / 281,817 = 100.000 %** | same |
| `n_lines` identical to `nd_verify`'s on proofs both accept (acceptance test 4) | **281,817 / 281,817 = 100 %**, 0 disagreements | same, keys `both_accept`, `n_lines_agree` |
| cost | 870 s Lean wall (3.075 s / 1,000), 19.9 s `nd_verify`, pod `lj1`, 6 workers | same |

The 1,195 are a **corpus-construction error of mine, not a regression**: `lj_regress.py collect` took every
`found*.jsonl` under a directory with a gate log, and run `efficiency`'s files store literal Lean text
(`( fun P => ( fun n1 => ( fun n2 => Or.inr h1 ) ) )`) where every other run stores an ND proof. `nd_verify`
rejects all 1,195, so the old `lean_seq` gate never counted them as ND proofs either. `collect` now skips any
`proof` that does not start with `N` and counts it under `not_nd_excluded`. Judging them properly is
`lean_check`'s job and needs run `efficiency`'s statement convention, which this run does not reimplement: a
spot check against a reconstructed `theorem t (P Q R S : Prop) (h1 : …)` statement accepts 567 and fails 628 on
elaboration (`artifacts/lj/t2_freeform.json`), which shows the statement form differs, not that the proofs are
bad. Not pursued further — out of this run's scope.

## L3 — acceptance test 3: the Lean-only class is now counted

| quantity | value | source |
|---|---|---|
| Lean-only records with an ND denotation | 6,419 | `artifacts/lj/t3_judge.json` |
| accepted by the new judge | **6,419 / 6,419 = 100 %** | same |
| excluded, no ND denotation (run `efficiency`'s `kind: no-denotation`) | 2,095 | same, key `skipped_no_nd` |

Classification of all 8,514 (`artifacts/lj/t3_classify.json`, `lj_regress.py classify`):

| class | n | fraction |
|---|---|---|
| omitted premise re-statement (`PR` lines fewer than declared premises) | 3,550 | 41.7 % |
| `BOTE` on a non-`F` line — i.e. `Not.elim` | 2,205 | 25.9 % |
| no ND denotation (excluded) | 2,095 | 24.6 % |
| other | 531 | 6.2 % |
| `NEGE` on `A` and `A > F` (Lean's `¬A` *is* `A → False`) | 133 | 1.6 % |

## L4 — pitfall 3 (coverage normalises before judging)

Lean's verdict on (i) the literal sampled text, (ii) `nd2lean(nd)`, (iii) `nd2lean(norm(nd))`, on all
6,419 Lean-only records that carry a literal text: **6,419 / 6,419 agree** (all three accept),
**0 disagreements**. So keying the verdict on the normalised string gives identical counts.
Source: `artifacts/lj/t3b_normform.json` (`lj_regress.py normform`, pod `lj1`).

## L5 — acceptance test 5: one expert-iteration round end to end

`expert_iter.py --init ckpts/stage1_a3_s0.pt --rounds 1 --k 16 --temperature 0.8 --relabel`, 200 depth-3
targets, 50 transfer, 200 held-out, retain 3,000, `ft_steps 50` (pod `lj1`).

| quantity | value | source |
|---|---|---|
| samples generated / grammar parse-failures / distinct texts Lean-checked | 4,250 / 476 / 3,506 | `artifacts/lj/t5_gate.jsonl` (4 gate calls) |
| Lean accepted (distinct) | 2,348 | same |
| registry conflicts (two texts, same ND proof, different verdicts) | **0** | same, key `registry_conflicts` |
| Lean cost for the round's gating | 11.5 s wall, 45.6 s process, 6 workers | same |
| **`LEAN*` markers in `found_1.jsonl`, `found_transfer_1.jsonl`, `mix_1.jsonl`** | **0, 0, 0** (228 / 56 / 3,912 records) | `artifacts/lj/t5_marker_grep.txt` |
| target / transfer sample acceptance | 0.5759 / 0.5975 | `artifacts/lj/t5_ei/round_1.json` |
| old gate's count on the same 3,506 samples (Lean ∧ `nd_verify`) | 2,348 | `artifacts/lj/t5_compare.json` |
| new judge's count (Lean alone) | 2,348 — **excess 0, losses 0** | same |

The excess is 0 on this round, which is consistent with the stored Lean-only rate: at 0.01–0.07 % of
distinct checked texts, 3,506 samples predict 0.4–2.5 such proofs, so 0 is the expected order. The
Lean-only class is evidenced at scale by L3, not by this round.

Hindsight relabelling on the round's real samples (`pod/lj/t5_relabel.py`, pitfall 2): of 3,506 dumped
samples, **246** yield a rewritten theorem that *differs* from the prompted one, reducing to 78 distinct
`(new theorem, proof)` pairs, none of which had a registered verdict (the gate never saw them) —
**166 accepted, 80 rejected** (1 untranslatable), 0.50 s of Lean.
Source: `artifacts/lj/t5_relabel.json`. One accepted relabel, checked by hand, in Lean:

```lean
theorem t (P Q R S : Prop) : ((¬(¬R)) → (((R ∨ Q) ∧ (S ∨ R)) → ((¬(¬R)) ∨ (¬(Q ∧ Q))))) := by
  have n5 : ((¬(¬R)) → (((R ∨ Q) ∧ (S ∨ R)) → ((¬(¬R)) ∨ (¬(Q ∧ Q))))) := (fun (n1 : (¬(¬R))) => by
    have n4 : (((R ∨ Q) ∧ (S ∨ R)) → ((¬(¬R)) ∨ (¬(Q ∧ Q)))) := (fun (n2 : ((R ∨ Q) ∧ (S ∨ R))) => by
      have n3 : ((¬(¬R)) ∨ (¬(Q ∧ Q))) := Or.inl n1
      exact n3)
    exact n4)
  exact n5
```

## L6 — acceptance test 6: throughput

20,000 stored accepted proofs, pod `lj1`, `LEAN_GATE_WORKERS=6` (`artifacts/lj/t6_throughput.json`):

| path | s / 1,000 strings |
|---|---|
| **old** `nd_verify` pass (dropped) | 0.0790 |
| **new** registry hit (what a loop sees after `generate()`) | **0.0123** |
| **new** marker (`LEANPARSE` / `LEANREJ`) | 0.0021 |
| **new** fallback: `nd2lean` + batched Lean (gate never saw the string) | **3.556** |

The Lean run on the literal texts is unchanged and is still done by the gate, so in the loop the new
judge is faster than the old one by exactly the dropped `nd_verify` pass: **6.4× faster on the judging
step**, 0.0667 s saved per 1,000 samples.

## L7 — acceptance test 7: `lean_check --selftest`

**39 / 39 pass**, VPS (Lean 4.34.1) and pod `lj1` (Lean 4.34.0).
Source: `artifacts/lj/lean_check_selftest.json`, `artifacts/lj/setup.log`.
Newly accepted: `h.elim a` on `h : ¬P` (term size 1), `Not.elim h a` (1), `Not.intro h` (1),
`And.elim` (5), `Iff.elim` (4), and the decision note's canonical example
`~P ⊢ P > Q` via `N1 (~P) : PR ; N2 (P>Q) : BOTE N1` → `have n2 : (P → Q) := n1.elim` (term size 1).
Still rejected: `sorry`, `sorry` inside a `have`, `simp`, `decide`, `Classical.em`, `Decidable.em`,
`Or.resolve_left`, `mt`, `not_not.mp`, `propext` as a term, `cases`, `match`, a truncated term.
The same canonical example is accepted by the `lean_seq` path: `lean_judge.verify_text` returns
`(True, '', 2)` where `nd_verify` returns
`(False, 'rule check failed: BOTE (line 2)', 2)` — asserted in `tests/test_lean_only_judge.py`.

## L8 — the grammar-as-allowlist claim

`lean_tok.LeanTokenizer('lean_seq').vocab_size` = **107** = 2 specials + 11 formula symbols + 6 prompt
symbols + 16 term/tactic symbols + `h1…h8` + `n1…n64`. The only tokens naming a Lean constant or
projection: `.1 .2 .elim Or.inl Or.inr Or.elim Classical.byContradiction` (plus the type formers
`P Q R S False Prop`). `sorry`, `simp`, `Classical.em` are not in the vocabulary.
Reproduce: the `print` in `LEAN_JUDGE.md` § "The grammar is the allowlist", or
`python3 -c "from lean_tok import LeanTokenizer; t=LeanTokenizer('lean_seq'); print(t.vocab_size, t.itos)"`.

## Bucket

`hf://buckets/dan-pandori/nd-rl/lean-judge/artifacts` — `artifacts/lj/` in full, including
`corpus_accepted.jsonl` (165 MB, gitignored) and `corpus_leanonly.jsonl`.


# state-env (proposal 13: an AlphaProof-style proof state for the policy)

Run `state-env`, 2026-09-28. Branch `dan_state-env`. Every number here is **under Lean alone** (`lean_gate` on the
literal `lean_seq` text, `lean_judge` elsewhere; Dan, 2026-09-27) unless the row says otherwise. `L_true` on the
ladder pools is `minlen.py`'s ND-derived label and is an **upper bound under Lean**.

**Models.** All arms are the same architecture as the control: 4 layers, d 256, 8 heads, **3,216,384 parameters**
(the control's 3,214,336 plus the 4 × 256 × 2 embedding and output rows of the four state tokens), trained **from
scratch** for 6,000 steps on **`data/p2/train_depth3_f0_a1.jsonl`** (155,000 proofs, cap 6, depth-3 f = 0), 128 whole
proofs per step. They differ only in what the policy reads and writes:

| arm | checkpoint | tokenizer mode | the policy's input | the policy's output |
|---|---|---|---|---|
| **S** | `ckpts/se/stage1_S_s{0,1}.pt` | `lean_state` | the focused goal's tactic state | one `lean_seq` step |
| **SH** | `ckpts/se/stage1_SH_s{0,1}.pt` | `lean_stateh` | the actions so far **and** the state | one `lean_seq` step |
| **SN-v2** (post-hoc diagnostic) | `ckpts/se/stage1_SN_s{0,1}.pt` | `lean_staten` | the state; the **environment** names what a step introduces (`max in scope + 1` from a per-attempt base `U[0,32]`) | one `lean_seq` step |
| **C0** | `ckpts/lf/stage1_a1_seq_s{0,1}.pt` (on file, **not re-run**) | `lean_seq` | the theorem | the whole proof |

C0's ladder and held-out numbers were measured by run `ds-generator` on 2026-09-23/24 **under Lean ∧ `nd_verify`**
and at whole-proof sampler batch 512 with `ND_SAMPLE_COMPACT=0`; per `AGENT_POLICY.md` a settings change is a
sampling re-draw, and the `ds-generator` spread on the same two checkpoints (T1 856 / 890 / 923 at s0) is what that
re-draw looks like. Lean-only versus Lean ∧ `nd_verify` is a smaller effect still: run `lean-format` found 460
Lean-only acceptances among 13.9 M samples.

**Sampler settings, identical for every arm and seed:** `ND_SAMPLE_PATH=fast`, `early='eos'`, compaction on, per-row
seeding; environment batch **2,048 attempts**, `--max_action 256`, `--max_steps 48`, temperature 0.8 on the ladder
and greedy (T 0) on held-out. Batch probe on Stage-1 S s0, one attempt per RL target
(`artifacts/se/probe_b*.json`): batch 1,024 / 2,048 / 4,096 / 8,192 → peak allocated **4.53 / 8.58 / 16.37 / 16.31 GB**,
wall 25.1 / 21.5 / 18.7 / 18.6 s, solved 986 / 943 / 974 / 971 (the spread is the re-draw). 2,048 was kept: 4,096
fits at Stage-1 state lengths but the environment's states grow with proof length and every card here is 24 GB.

**Gates** (all before the first measurement of the arm they cover):

| gate | arm S / SH (`lean_state`) | arm SN (`lean_staten`) |
|---|---|---|
| 1 round trip, byte for byte, on all 155,000 control proofs | 0 failures | 0 failures |
| 1b random reassembled texts accepted by Lean | 0 rejected / 5,000 | 0 rejected / 5,000 |
| 2 renderer's state == Lean's `trace_state` at a random cut | 0 mismatches / 2,800 | 0 mismatches / 1,300 |
| 3 environment replay of every control proof | 0 failures / 155,000 | 0 failures / 155,000 |
| `have` actions whose name is not `max index in scope + 1` | 15,821 / 526,784 (3.00 %) | **0** / 526,784 |
| 2b gate 2 on 7–15-line proofs the model found (T1 S s0, round 3) | 0 mismatches / 1,500 | — |

Sources: `artifacts/se/gate13.json`, `gate2.json`, `gate2_ore.json`, `gate2_neg.json`, `gate13_canon.json`,
`gate2_canon.json`, `gate2_canon_ore.json`; per-case Lean sources in `artifacts/se/gate2*_cases.jsonl`.


**The falsifiers (arm S, pre-registered):** neither fires. T1 `L*` = **12 / 12**; solved at `L_true` ≥ 13 = **1 / 2**
("is the wall" needed `L*` ≥ 13 and ≥ 5; "is not the wall" needed 0 on both seeds). Only **3 distinct** `L_true` ≥ 13
theorems were solved by any run: `la_transfer_1126` (S s0, S s1, SN s0, SN s1), `la_transfer_1198` (S s1, SN s0),
`la_transfer_978` (SN s0); source `artifacts/se/summary.json` → `ladders.runs.*.derived.solved_ge13_names`.

### Held-out greedy, depth-3 slice (`pat.depth3` of `data/p2/heldout.jsonl`, 500 records) — `artifacts/se/heldout_depth3.json` (`se_depth3.py`)

| model | depth-3 | rest (4,500) |
|---|---|---|
| SH_s0 | 430/500 = 0.860 | 0.962 |
| SH_s1 | 440/500 = 0.880 | 0.964 |
| SN_s0 | 478/500 = 0.956 | 0.972 |
| SN_s1 | 451/500 = 0.902 | 0.965 |
| SNv1_s0 | 479/500 = 0.958 | 0.973 |
| SNv1_s1 | 449/500 = 0.898 | 0.592 |
| S_s0 | 461/500 = 0.922 | 0.962 |
| S_s1 | 441/500 = 0.882 | 0.792 |
| C0_s0 (Lean AND nd_verify, on file) | 244/500 = 0.488 | 0.956 |
| C0_s1 (Lean AND nd_verify, on file) | 136/500 = 0.272 | 0.966 |

### Cost and bucket

| pod | GPU | $/h (billed) | hours | $ |
|---|---|---|---|---|
| se-1 | NVIDIA GeForce RTX 3090 | 0.50 | 9.83 | 4.91 |
| se-2 | NVIDIA RTX PRO 4000 Blackwell | 0.57 | 11.20 | 6.38 |
| se-3 | NVIDIA GeForce RTX 3090 | 0.50 | 4.62 | 2.31 |
| se-4 | NVIDIA GeForce RTX 4090 | 0.74 | 5.04 | 3.73 |
| se-5 | NVIDIA RTX PRO 4500 Blackwell Server Edition | 0.72 | 2.24 | 1.61 |
| se-6 | NVIDIA GeForce RTX 3090 | 0.50 | 2.51 | 1.26 |
| **total** | | | **35.44** | **20.20** |

`podbudget` reports 35.44 h / $19.42 (it bills `se-2` at its $0.50/h fallback). Budget 60 h / $30.
Bucket: `hf://buckets/dan-pandori/nd-rl/state-env/ckpts/se/` (six Stage-1 checkpoints S / SH / SN × 2 seeds, and
every EI round `ladder/la_T1_{S,SH,SN}_s{0,1}_r{1..8}.pt`), `…/artifacts/se/` (every ladder run's round jsons,
found files, mixes and logs; held-out per-record files; gates), `…/data/` (the control set and pools used).

### Held-out greedy (`data/p2/heldout.jsonl`, 5,000, k 1, T 0; per length)

| model | overall | 2 | 3 | 4 | 5 | 6 | source |
|---|---|---|---|---|---|---|---|
| SH_s0 `stage1_SH_s0.pt` | 0.9516 | 0.998 | 0.991 | 0.971 | 0.928 | 0.870 | `artifacts/se/heldout_SH_s0.json` |
| SH_s1 `stage1_SH_s1.pt` | 0.9554 | 1.000 | 0.985 | 0.961 | 0.936 | 0.895 | `artifacts/se/heldout_SH_s1.json` |
| SN_s0 `stage1_SN_s0.pt` | 0.9700 | 0.998 | 0.992 | 0.972 | 0.947 | 0.941 | `artifacts/se/heldout_SN_s0.json` |
| SN_s1 `stage1_SN_s1.pt` | 0.9584 | 0.999 | 0.993 | 0.970 | 0.926 | 0.904 | `artifacts/se/heldout_SN_s1.json` |
| SNv1_s0 `stage1_SN_s0.pt` | 0.9716 | 0.998 | 0.992 | 0.972 | 0.952 | 0.944 | `artifacts/se/heldout_SNv1_s0.json` |
| SNv1_s1 `stage1_SN_s1.pt` | 0.6230 | 0.608 | 0.527 | 0.674 | 0.643 | 0.663 | `artifacts/se/heldout_SNv1_s1.json` |
| S_s0 `stage1_S_s0.pt` | 0.9578 | 0.999 | 0.992 | 0.961 | 0.933 | 0.904 | `artifacts/se/heldout_S_s0.json` |
| S_s1 `stage1_S_s1.pt` | 0.8012 | 0.991 | 0.878 | 0.757 | 0.702 | 0.678 | `artifacts/se/heldout_S_s1.json` |
| C0 s0 `stage1_a1_seq_s0.pt` (on file, Lean ∧ nd_verify) | 0.9088 | 0.994 | 0.989 | 0.951 | 0.924 | 0.686 | `review_ds-generator.md` §3 |
| C0 s1 `stage1_a1_seq_s1.pt` (on file, Lean ∧ nd_verify) | 0.8968 | 0.997 | 0.992 | 0.967 | 0.944 | 0.584 | same |

### Ladder: transfer pool (2,285 theorems), cumulative at the last round

| run | last round | solved | `L*` | ≥13 | greedy | textbook | distinct proofs | source |
|---|---|---|---|---|---|---|---|---|
| `la_T1_SH_s0` | 8 | 1257 | 12 | 0 | 637 | 148/760 (11/19 schemata) | 2981 | `artifacts/se/la_T1_SH_s0/found_transfer_8.jsonl` |
| `la_T1_SH_s1` | 8 | 1390 | 12 | 0 | 912 | 201/760 (12/19 schemata) | 3623 | `artifacts/se/la_T1_SH_s1/found_transfer_8.jsonl` |
| `la_T1_SN_s0` | 8 | 1557 | 12 | 3 | 1100 | 236/760 (16/19 schemata) | 8377 | `artifacts/se/la_T1_SN_s0/found_transfer_8.jsonl` |
| `la_T1_SN_s1` | 8 | 1403 | 12 | 1 | 887 | 197/760 (14/19 schemata) | 4620 | `artifacts/se/la_T1_SN_s1/found_transfer_8.jsonl` |
| `la_T1_S_s0` | 8 | 1348 | 12 | 1 | 735 | 182/760 (12/19 schemata) | 3667 | `artifacts/se/la_T1_S_s0/found_transfer_8.jsonl` |
| `la_T1_S_s1` | 8 | 1389 | 12 | 2 | 840 | 216/760 (12/19 schemata) | 4187 | `artifacts/se/la_T1_S_s1/found_transfer_8.jsonl` |
| `la_frozen_SH_s0` | 8 | 516 | 10 | 0 | 158 | 79/760 (6/19 schemata) | 730 | `artifacts/se/la_frozen_SH_s0/found_transfer_8.jsonl` |
| `la_frozen_SH_s1` | 8 | 479 | 10 | 0 | 149 | 66/760 (7/19 schemata) | 698 | `artifacts/se/la_frozen_SH_s1/found_transfer_8.jsonl` |
| `la_frozen_SN_s0` | 8 | 975 | 11 | 0 | 429 | 111/760 (11/19 schemata) | 2075 | `artifacts/se/la_frozen_SN_s0/found_transfer_8.jsonl` |
| `la_frozen_SN_s1` | 8 | 801 | 11 | 0 | 261 | 110/760 (10/19 schemata) | 1552 | `artifacts/se/la_frozen_SN_s1/found_transfer_8.jsonl` |
| `la_frozen_S_s0` | 8 | 779 | 11 | 0 | 212 | 93/760 (7/19 schemata) | 1387 | `artifacts/se/la_frozen_S_s0/found_transfer_8.jsonl` |
| `la_frozen_S_s1` | 8 | 787 | 10 | 0 | 144 | 100/760 (7/19 schemata) | 1351 | `artifacts/se/la_frozen_S_s1/found_transfer_8.jsonl` |
| `la_T1_c0_s0` (C0, on file) | 8 | 890 | 12 | 0 | 553 | 37/760 (9/19 schemata) | 2108 | `artifacts/dsg/la_T1_c0_s0/found_transfer_8.jsonl` |
| `la_T1_c0_s1` (C0, on file) | 8 | 965 | 11 | 0 | 646 | 89/760 (7/19 schemata) | 2175 | `artifacts/dsg/la_T1_c0_s1/found_transfer_8.jsonl` |
| `la_frozen_c0_s0` (C0, on file) | 8 | 158 | 9 | 0 | 10 | 6/760 (4/19 schemata) | 203 | `artifacts/dsg/la_frozen_c0_s0/found_transfer_8.jsonl` |
| `la_frozen_c0_s1` (C0, on file) | 8 | 114 | 9 | 0 | 16 | 13/760 (4/19 schemata) | 136 | `artifacts/dsg/la_frozen_c0_s1/found_transfer_8.jsonl` |

### Transfer solved by `L_true` (ND-derived upper bound under Lean)

| run | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|
| `la_T1_SH_s0` | 183/300 | 196/300 | 703/1010 | 142/451 | 27/99 | 6/102 | 0/13 | 0/10 |
| `la_T1_SH_s1` | 195/300 | 233/300 | 743/1010 | 174/451 | 37/99 | 8/102 | 0/13 | 0/10 |
| `la_T1_SN_s0` | 196/300 | 235/300 | 834/1010 | 232/451 | 42/99 | 15/102 | 2/13 | 1/10 |
| `la_T1_SN_s1` | 191/300 | 210/300 | 752/1010 | 205/451 | 33/99 | 11/102 | 1/13 | 0/10 |
| `la_T1_S_s0` | 183/300 | 207/300 | 739/1010 | 187/451 | 24/99 | 7/102 | 1/13 | 0/10 |
| `la_T1_S_s1` | 187/300 | 234/300 | 745/1010 | 184/451 | 28/99 | 9/102 | 2/13 | 0/10 |
| `la_frozen_SH_s0` | 125/300 | 128/300 | 226/1010 | 36/451 | 1/99 | 0/102 | 0/13 | 0/10 |
| `la_frozen_SH_s1` | 111/300 | 119/300 | 225/1010 | 24/451 | 0/99 | 0/102 | 0/13 | 0/10 |
| `la_frozen_SN_s0` | 151/300 | 182/300 | 546/1010 | 87/451 | 9/99 | 0/102 | 0/13 | 0/10 |
| `la_frozen_SN_s1` | 145/300 | 152/300 | 432/1010 | 67/451 | 5/99 | 0/102 | 0/13 | 0/10 |
| `la_frozen_S_s0` | 137/300 | 159/300 | 419/1010 | 59/451 | 5/99 | 0/102 | 0/13 | 0/10 |
| `la_frozen_S_s1` | 142/300 | 154/300 | 427/1010 | 61/451 | 3/99 | 0/102 | 0/13 | 0/10 |
| `la_T1_c0_s0` | 92/300 | 166/300 | 525/1010 | 91/451 | 11/99 | 5/102 | 0/13 | 0/10 |
| `la_T1_c0_s1` | 145/300 | 165/300 | 531/1010 | 106/451 | 15/99 | 3/102 | 0/13 | 0/10 |
| `la_frozen_c0_s0` | 42/300 | 69/300 | 44/1010 | 3/451 | 0/99 | 0/102 | 0/13 | 0/10 |
| `la_frozen_c0_s1` | 44/300 | 47/300 | 22/1010 | 1/451 | 0/99 | 0/102 | 0/13 | 0/10 |

### Environment per-step diagnostics (last round of each run)

| run | attempts | finished | syntactic error | Lean rejected | truncated | step cap | mean steps/attempt | action tokens (mean) | peak alloc GB |
|---|---|---|---|---|---|---|---|---|---|
| `la_T1_SH_s0` | 224245 | 87.0 % | 13.0 % | 34.7 % | 0.027 % | 0.000 % | 8.30 | 19.2 | 18.98 |
| `la_T1_SH_s1` | 224245 | 89.6 % | 10.4 % | 32.3 % | 0.042 % | 0.000 % | 8.48 | 19.2 | 20.35 |
| `la_T1_SN_s0` | 224245 | 92.8 % | 7.2 % | 25.8 % | 0.003 % | 0.003 % | 9.41 | 18.5 | 12.51 |
| `la_T1_SN_s1` | 224245 | 93.9 % | 6.0 % | 34.2 % | 0.046 % | 0.001 % | 8.96 | 18.9 | 12.11 |
| `la_T1_S_s0` | 224245 | 88.8 % | 11.1 % | 33.1 % | 0.016 % | 0.000 % | 8.51 | 19.3 | 10.16 |
| `la_T1_S_s1` | 224245 | 88.9 % | 11.1 % | 32.9 % | 0.074 % | 0.000 % | 8.75 | 19.1 | 12.74 |
| `la_frozen_SH_s0` | 224245 | 78.5 % | 21.3 % | 63.0 % | 0.202 % | 0.000 % | 7.05 | 20.5 | 19.27 |
| `la_frozen_SH_s1` | 224245 | 81.6 % | 18.3 % | 65.1 % | 0.073 % | 0.000 % | 7.04 | 20.4 | 18.54 |
| `la_frozen_SN_s0` | 224245 | 89.1 % | 10.8 % | 57.0 % | 0.026 % | 0.000 % | 7.73 | 19.7 | 10.61 |
| `la_frozen_SN_s1` | 224245 | 86.6 % | 13.3 % | 64.1 % | 0.096 % | 0.000 % | 7.56 | 20.0 | 11.28 |
| `la_frozen_S_s0` | 224245 | 82.6 % | 17.4 % | 63.9 % | 0.054 % | 0.000 % | 7.41 | 20.3 | 11.41 |
| `la_frozen_S_s1` | 224245 | 82.6 % | 17.2 % | 62.2 % | 0.206 % | 0.000 % | 7.50 | 20.2 | 10.84 |

### Proof length: lines **and** elaborated Lean term size (`lean_check`), shortest accepted proof per theorem

| run | `L_true` | n | lines min/median/max | term size min/median/max |
|---|---|---|---|---|
| `la_T1_SH_s0` | 7 | 183 | 7/7/11 | 3/4/7 |
| `la_T1_SH_s0` | 8 | 196 | 6/8/11 | 3/4/8 |
| `la_T1_SH_s0` | 9 | 703 | 6/9/14 | 3/5/10 |
| `la_T1_SH_s0` | 10 | 142 | 10/10/12 | 4/5/9 |
| `la_T1_SH_s0` | 11 | 27 | 11/11/13 | 5/6/11 |
| `la_T1_SH_s0` | 12 | 6 | 12/12/12 | 5/7/8 |
| `la_T1_SH_s1` | 7 | 195 | 7/7/11 | 3/4/7 |
| `la_T1_SH_s1` | 8 | 233 | 6/8/12 | 3/4/9 |
| `la_T1_SH_s1` | 9 | 743 | 6/9/13 | 3/5/15 |
| `la_T1_SH_s1` | 10 | 174 | 7/10/13 | 4/5/11 |
| `la_T1_SH_s1` | 11 | 37 | 11/11/13 | 4/6/11 |
| `la_T1_SH_s1` | 12 | 8 | 12/12/13 | 5/8/9 |
| `la_T1_SN_s0` | 7 | 196 | 7/7/13 | 3/4/10 |
| `la_T1_SN_s0` | 8 | 235 | 8/8/14 | 3/4/9 |
| `la_T1_SN_s0` | 9 | 834 | 9/9/18 | 3/5/15 |
| `la_T1_SN_s0` | 10 | 232 | 10/10/27 | 4/6/25 |
| `la_T1_SN_s0` | 11 | 42 | 11/11/14 | 5/7/11 |
| `la_T1_SN_s0` | 12 | 15 | 12/12/22 | 5/8/14 |
| `la_T1_SN_s0` | 13 | 2 | 13/17/17 | 8/13/13 |
| `la_T1_SN_s0` | 14 | 1 | 14/14/14 | 11/11/11 |
| `la_T1_SN_s1` | 7 | 191 | 7/7/11 | 3/4/7 |
| `la_T1_SN_s1` | 8 | 210 | 6/8/13 | 3/4/9 |
| `la_T1_SN_s1` | 9 | 752 | 8/9/15 | 3/5/11 |
| `la_T1_SN_s1` | 10 | 205 | 9/10/14 | 4/6/14 |
| `la_T1_SN_s1` | 11 | 33 | 11/11/15 | 5/6/12 |
| `la_T1_SN_s1` | 12 | 11 | 12/12/15 | 5/8/14 |
| `la_T1_SN_s1` | 13 | 1 | 13/13/13 | 8/8/8 |
| `la_T1_S_s0` | 7 | 183 | 7/7/11 | 3/4/6 |
| `la_T1_S_s0` | 8 | 207 | 8/8/19 | 3/4/18 |
| `la_T1_S_s0` | 9 | 739 | 8/9/13 | 3/5/11 |
| `la_T1_S_s0` | 10 | 187 | 10/10/16 | 4/6/14 |
| `la_T1_S_s0` | 11 | 24 | 11/11/14 | 5/6/11 |
| `la_T1_S_s0` | 12 | 7 | 12/12/12 | 5/8/8 |
| `la_T1_S_s0` | 13 | 1 | 13/13/13 | 8/8/8 |
| `la_T1_S_s1` | 7 | 187 | 7/7/13 | 3/4/8 |
| `la_T1_S_s1` | 8 | 234 | 8/8/13 | 3/4/11 |
| `la_T1_S_s1` | 9 | 745 | 7/9/18 | 3/5/15 |
| `la_T1_S_s1` | 10 | 184 | 10/10/13 | 4/6/13 |
| `la_T1_S_s1` | 11 | 28 | 11/11/17 | 5/6/14 |
| `la_T1_S_s1` | 12 | 9 | 12/12/13 | 5/8/9 |
| `la_T1_S_s1` | 13 | 2 | 13/13/13 | 8/9/9 |
| `la_frozen_SH_s0` | 7 | 125 | 6/7/8 | 2/4/6 |
| `la_frozen_SH_s0` | 8 | 128 | 6/8/9 | 3/4/7 |
| `la_frozen_SH_s0` | 9 | 226 | 6/9/11 | 2/4/10 |
| `la_frozen_SH_s0` | 10 | 36 | 7/10/11 | 3/5/7 |
| `la_frozen_SH_s0` | 11 | 1 | 11/11/11 | 6/6/6 |
| `la_frozen_SH_s1` | 7 | 111 | 7/7/8 | 3/4/6 |
| `la_frozen_SH_s1` | 8 | 119 | 5/8/9 | 2/4/7 |
| `la_frozen_SH_s1` | 9 | 225 | 6/9/11 | 3/4/9 |
| `la_frozen_SH_s1` | 10 | 24 | 10/10/10 | 4/5/7 |
| `la_frozen_SN_s0` | 7 | 151 | 7/7/11 | 3/4/10 |
| `la_frozen_SN_s0` | 8 | 182 | 8/8/11 | 3/4/8 |
| `la_frozen_SN_s0` | 9 | 546 | 9/9/12 | 3/5/10 |
| `la_frozen_SN_s0` | 10 | 87 | 10/10/15 | 4/5/11 |
| `la_frozen_SN_s0` | 11 | 9 | 11/11/14 | 6/6/9 |
| `la_frozen_SN_s1` | 7 | 145 | 7/7/11 | 3/4/8 |
| `la_frozen_SN_s1` | 8 | 152 | 6/8/12 | 3/4/9 |
| `la_frozen_SN_s1` | 9 | 432 | 6/9/12 | 3/5/10 |
| `la_frozen_SN_s1` | 10 | 67 | 10/10/11 | 4/5/9 |
| `la_frozen_SN_s1` | 11 | 5 | 11/11/11 | 6/6/8 |
| `la_frozen_S_s0` | 7 | 137 | 7/7/8 | 3/4/6 |
| `la_frozen_S_s0` | 8 | 159 | 6/8/10 | 3/4/8 |
| `la_frozen_S_s0` | 9 | 419 | 7/9/11 | 3/5/9 |
| `la_frozen_S_s0` | 10 | 59 | 10/10/11 | 4/5/9 |
| `la_frozen_S_s0` | 11 | 5 | 11/11/11 | 5/6/6 |
| `la_frozen_S_s1` | 7 | 142 | 7/7/8 | 3/4/6 |
| `la_frozen_S_s1` | 8 | 154 | 5/8/9 | 3/4/8 |
| `la_frozen_S_s1` | 9 | 427 | 6/9/12 | 3/5/10 |
| `la_frozen_S_s1` | 10 | 61 | 9/10/11 | 4/5/9 |
| `la_frozen_S_s1` | 11 | 3 | 11/11/11 | 5/6/8 |
| `la_T1_c0_s0` | 7 | 92 | 7/7/11 | 3/4/6 |
| `la_T1_c0_s0` | 8 | 166 | 8/8/10 | 3/4/9 |
| `la_T1_c0_s0` | 9 | 525 | 9/9/11 | 3/5/10 |
| `la_T1_c0_s0` | 10 | 91 | 10/10/11 | 4/5/10 |
| `la_T1_c0_s0` | 11 | 11 | 11/11/11 | 5/6/8 |
| `la_T1_c0_s0` | 12 | 5 | 12/12/12 | 5/6/8 |
| `la_T1_c0_s1` | 7 | 145 | 7/7/11 | 3/4/6 |
| `la_T1_c0_s1` | 8 | 165 | 8/8/10 | 3/4/9 |
| `la_T1_c0_s1` | 9 | 531 | 9/9/11 | 3/5/10 |
| `la_T1_c0_s1` | 10 | 106 | 10/10/12 | 4/5/10 |
| `la_T1_c0_s1` | 11 | 15 | 11/11/11 | 4/6/8 |
| `la_T1_c0_s1` | 12 | 3 | 12/12/12 | 5/5/8 |
| `la_frozen_c0_s0` | 7 | 42 | 7/7/8 | 3/4/6 |
| `la_frozen_c0_s0` | 8 | 69 | 8/8/11 | 3/4/7 |
| `la_frozen_c0_s0` | 9 | 44 | 9/9/9 | 4/4/7 |
| `la_frozen_c0_s0` | 10 | 3 | 10/10/10 | 5/5/5 |
| `la_frozen_c0_s1` | 7 | 44 | 7/7/9 | 3/4/6 |
| `la_frozen_c0_s1` | 8 | 47 | 8/8/8 | 3/4/7 |
| `la_frozen_c0_s1` | 9 | 22 | 9/9/10 | 4/4/6 |
| `la_frozen_c0_s1` | 10 | 1 | 10/10/10 | 5/5/5 |

### Transfer solved by round (cumulative)

| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| `la_T1_SH_s0` | 378 | 665 | 837 | 970 | 1067 | 1146 | 1205 | 1257 |
| `la_T1_SH_s1` | 332 | 661 | 883 | 1023 | 1140 | 1237 | 1324 | 1390 |
| `la_T1_SN_s0` | 794 | 1090 | 1243 | 1341 | 1422 | 1474 | 1515 | 1557 |
| `la_T1_SN_s1` | 604 | 918 | 1092 | 1201 | 1278 | 1332 | 1372 | 1403 |
| `la_T1_S_s0` | 577 | 859 | 1019 | 1112 | 1208 | 1261 | 1311 | 1348 |
| `la_T1_S_s1` | 588 | 886 | 1069 | 1190 | 1251 | 1310 | 1346 | 1389 |
| `la_frozen_SH_s0` | 373 | 424 | 440 | 462 | 482 | 492 | 502 | 516 |
| `la_frozen_SH_s1` | 348 | 388 | 411 | 424 | 440 | 457 | 468 | 479 |
| `la_frozen_SN_s0` | 794 | 850 | 889 | 914 | 935 | 953 | 963 | 975 |
| `la_frozen_SN_s1` | 604 | 665 | 708 | 736 | 762 | 781 | 795 | 801 |
| `la_frozen_S_s0` | 577 | 645 | 671 | 697 | 716 | 742 | 767 | 779 |
| `la_frozen_S_s1` | 588 | 653 | 681 | 716 | 732 | 744 | 765 | 787 |
| `la_T1_c0_s0` | 76 | 395 | 546 | 665 | 736 | 790 | 847 | 890 |
| `la_T1_c0_s1` | 57 | 376 | 568 | 718 | 799 | 872 | 924 | 965 |
| `la_frozen_c0_s0` | 76 | 105 | 120 | 128 | 137 | 149 | 156 | 158 |
| `la_frozen_c0_s1` | 57 | 74 | 88 | 96 | 99 | 104 | 108 | 114 |

