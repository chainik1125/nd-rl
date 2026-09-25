# Reviewed leaderboard (arm B)

The incumbent is the row marked "ADOPT (incumbent)" (rank 1). Rows below it are sorted by dev score. A rejected run can out-score the incumbent by less than noise; such rows are listed below it. Only reviewed runs appear. Dev score = `score.py` k=32 seed=0 (n=1610);
seed-1 = reviewer's rescore of a copy of the same `final.pt` with `--seed 1`. Noise floor: ~15 theorems (SPEC rule 6); observed spread of four near-control runs is 219–249 (mean 237, SD ≈ 13), so treat single-run gains under ~30 as unproven.

| rank | experiment | checkpoint | dev score (s0) | seed-1 rescore | heldout_greedy | verdict | review |
|---:|---|---|---:|---:|---:|---|---|
| 1 | **exp13_textbook_s0** (textbook-source exploration stream at T 1.0 on exp10 code) | /work/exec/runs/exp13_textbook_s0/final.pt | **324** (109/86/129) | 328 (106/90/132) | 0.896 | **ADOPT (incumbent)** | review_exp13.md |
| 2 | exp23_cheapfocus_s0 (exp21 with focus_k 16; 5 rounds; 172 textbook solves) | /work/exec/runs/exp23_cheapfocus_s0/final.pt | 339 (112/95/132) | 338 (114/91/133) | 0.901 | REJECT (+12.5 on 2-sampling-seed mean, under the +15 bar; s1 replicate 323) | review_exp23.md |
| 3 | exp17_tbr1_s1 (exp13 with textbook stream from round 1; seed 1; 181 textbook solves, 3 schemas) | /work/exec/runs/exp17_tbr1_s1/final.pt | 331 (115/94/122) | 327 (111/93/123) | 0.891 | REJECT (+7, one seed; s0 replicate running) | review_exp17_s1.md |
| 4 | exp19_tbhot_s0 (textbook stream from r1, schema-aware, T 1.2 uncracked; 177 solves, 3 schemas) | /work/exec/runs/exp19_tbhot_s0/final.pt | 328 (115/91/122) | — | 0.897 | REJECT (+4, one seed) | review_exp19.md |
| 5 | exp16_schema_s0 (exp13 + schema-aware textbook allocation; still 2 schemas) | /work/exec/runs/exp16_schema_s0/final.pt | 324 (109/90/125) | — | 0.907 | REJECT (tie; heldout +0.011 not clearly better) | review_exp16.md |
| 6 | exp21_hot_focus_s0 (exp19 + gen focus from r4; 4 rounds; 165 textbook solves) | /work/exec/runs/exp21_hot_focus_s0/final.pt | 323 (107/93/123) | 319 (108/91/120) | 0.908 | REJECT (tie within noise) | review_exp21.md |
| 7 | exp23_cheapfocus_s1 (exp23 recipe, seed 1; 165 textbook solves) | /work/exec/runs/exp23_cheapfocus_s1/final.pt | 323 (105/95/123) | — | 0.902 | REJECT | review_exp23_s1.md |
| 8 | exp21_hot_focus_s1 (exp21 recipe, seed 1; 159 textbook solves) | /work/exec/runs/exp21_hot_focus_s1/final.pt | 318 (98/94/126) | — | 0.898 | REJECT | review_exp21_s1.md |
| 9 | exp19_tbhot_s2 (exp19 recipe, seed 2; late ignition, 66 solves) | /work/exec/runs/exp19_tbhot_s2/final.pt | 309 (98/90/121) | — | 0.889 | REJECT | review_exp19_s2.md |
| 10 | exp15_tb_pertarget_s0 (exp13 + per-target weighting; 150 textbook solves) | /work/exec/runs/exp15_tb_pertarget_s0/final.pt | 303 (104/89/110) | — | 0.897 | REJECT (< incumbent) | review_exp15.md |
| 11 | exp19_tbhot_s1 (exp19 recipe, seed 1; early ignition, 131 solves) | /work/exec/runs/exp19_tbhot_s1/final.pt | 301 (98/89/114) | — | 0.906 | REJECT | review_exp19_s1.md |
| 12 | exp18_tb_focus_s0 (exp13 + gen focus from r4; late ignition, 53 textbook solves) | /work/exec/runs/exp18_tb_focus_s0/final.pt | 300 (76/94/130) | — | 0.899 | REJECT (< incumbent) | review_exp18.md |
| 13 | exp21_hot_focus_s2 (exp21 recipe, seed 2; 120 textbook solves) | /work/exec/runs/exp21_hot_focus_s2/final.pt | 300 (96/91/113) | — | 0.883 | REJECT | review_exp21_s2.md |
| 14 | exp22_tbbig_s0 (exp19 with tb_k 64, gen k 12; 4 rounds) | /work/exec/runs/exp22_tbbig_s0/final.pt | 290 (107/86/97) | — | 0.902 | REJECT | review_exp22.md |
| 15 | exp13_textbook_s1 (exp13 recipe, training seed 1; replicate: 13 textbook solves vs 145) | /work/exec/runs/exp13_textbook_s1/final.pt | 269 (67/85/117) | — | 0.910 | REJECT as checkpoint; supports exp13 recipe (2 seeds) | review_exp13_s1.md |
| 16 | exp14_focus_s0 (frontier focus from r4: <2-proof targets, k48, T1.0; on exp10 code) | /work/exec/runs/exp14_focus_s0/final.pt | 269 (44/94/131) | — | 0.897 | REJECT (< incumbent) | review_exp14.md |
| 17 | exp12_pertarget_s0 (per-target weighting 12 records/target on exp10 code; 6 rounds) | /work/exec/runs/exp12_pertarget_s0/final.pt | 268 (71/87/110) | pending | 0.899 | REJECT (not better than exp13; +19 over exp01, one seed) | review_exp12.md |
| 18 | exp11_retain2k_s0 (retain 2k on exp10 code; 7 rounds) | /work/exec/runs/exp11_retain2k_s0/final.pt | 255 (62/85/108) | 250 (57/86/107) | 0.871 | REJECT (+6, one seed; heldout −0.05) | review_exp11.md |
| 19 | exp10_fastiter_fastgen_s0 (k16, ft300, retain 8k, fast sampler; 6 rounds) | /work/exec/runs/exp10_fastiter_fastgen_s0/final.pt | 252 (40/87/125) | 249 (38/89/122) | 0.910 | REJECT (+3, one seed) | review_exp10.md |
| 20 | exp08_fastiter2_s0 (k16, ft300, retain 8k, lib sampler; 5 rounds) | /work/exec/runs/exp08_fastiter2_s0/final.pt | 252 (39/87/126) | pending | 0.902 | REJECT (+3, one seed) | review_exp08.md |
| 21 | exp01_control_s1 (seed rl.py, seed 1) | /work/exec/runs/exp01_control_s1/final.pt | 249 (49/91/109) | 254 (53/91/110) | 0.920 | ADOPT (former incumbent, 01:35–02:45) | review_exp01.md |
| 22 | exp05_batch1024_s0 (sample_batch 1024; 3 rounds, effectively a control replicate) | /work/exec/runs/exp05_batch1024_s0/final.pt | 242 (41/89/112) | — | 0.915 | REJECT | review_exp05.md |
| 23 | exp03_fastiter_s0 (k16, ft300, retain 8k, ≤12 rounds) — confounded by reviewer rescore on its GPU | /work/exec/runs/exp03_fastiter_s0/final.pt | 238 (37/90/111) | — | 0.909 | REJECT (confounded) | review_exp03.md |
| 24 | exp17_tbr1_s0 (exp17 recipe, seed 0; no ignition, 1 textbook solve) | /work/exec/runs/exp17_tbr1_s0/final.pt | 236 (34/90/112) | — | 0.886 | REJECT | review_exp17_s0.md |
| 25 | exp09_fastgen_s0 (custom fast sampler, seed knobs; 3 rounds) | /work/exec/runs/exp09_fastgen_s0/final.pt | 235 (34/91/110) | — | 0.924 | REJECT | review_exp09.md |
| 26 | exp06_tailfill_s0 (shrunk last round + skip saturated; 3 rounds, r3 shrunk) | /work/exec/runs/exp06_tailfill_s0/final.pt | 234 (40/90/104) | — | 0.927 | REJECT | review_exp06.md |
| 27 | exp02_relabel_s0 (hindsight relabel ≥7 lines) | /work/exec/runs/exp02_relabel_s0/final.pt | 219 (38/89/92) | — | 0.916 | REJECT | review_exp02.md |
| 28 | exp07_curriculum_s0 (length curriculum 7→8→9, all targets, on exp06 code) | /work/exec/runs/exp07_curriculum_s0/final.pt | 217 (33/93/91) | — | 0.915 | REJECT | review_exp07.md |
| 29 | exp04_breadth_s0 (all 3,804 targets, k12, skip saturated; 2 rounds only) | /work/exec/runs/exp04_breadth_s0/final.pt | 193 (34/89/70) | — | 0.913 | REJECT | review_exp04.md |
| — | pretrain.pt (reference, brief) | /work/frozen/pretrain.pt | 7 | 5 | — | baseline | — |

**Seed-1 rescores** were blocked from 02:06 to 04:08 (both GPUs continuously in use). Done after the executor finished: exp13, exp23, exp21, exp11, exp17_s1, exp10 (sampling-seed deltas −5 to +4). Still pending at close: exp19, exp16, exp12, exp08 (queue left running; results land in rescore/*_seed1.txt).
