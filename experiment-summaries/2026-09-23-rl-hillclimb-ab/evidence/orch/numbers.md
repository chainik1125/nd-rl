# numbers (executor, arm B) — every row cites its source file

| run | seed | change | dev pass@32 | L7/L8/L9 | heldout_greedy | status | source |
|---|---|---|---|---|---|---|---|
| exp01_control_s1 | 1 | none (seed rl.py) | 249 | 49/91/109 | 0.920 | ok | runs/exp01_control_s1/summary.json |
| exp02_relabel_s0 | 0 | hindsight relabel (>=7 lines) | 219 | 38/89/92 | 0.916 | ok | runs/exp02_relabel_s0/summary.json |
| exp03_fastiter_s0 | 0 | k16, ft300, retain8k, rounds<=12 | 238 | 37/90/111 | 0.909 | ok | runs/exp03_fastiter_s0/summary.json |
| exp04_breadth_s0 | 0 | all 3804 targets, k12, skip saturated | 193 | 34/89/70 | 0.913 | ok | runs/exp04_breadth_s0/summary.json |
| exp05_batch1024_s0 | 0 | sample_batch 1024 | 242 | 41/89/112 | 0.915 | ok | runs/exp05_batch1024_s0/summary.json |
| exp06_tailfill_s0 | 0 | tail-fill shrunk last round + skip saturated | 234 | 40/90/104 | 0.927 | ok | runs/exp06_tailfill_s0/summary.json |
| exp07_curriculum_s0 | 0 | length curriculum over all 3804 (on exp06 code) | 217 | 33/93/91 | 0.915 | ok | runs/exp07_curriculum_s0/summary.json |
| exp08_fastiter2_s0 | 0 | k16, ft300, retain8k on exp06 code (5 rounds) | 252 | 39/87/126 | 0.902 | ok | runs/exp08_fastiter2_s0/summary.json |
| exp09_fastgen_s0 | 0 | fast sampler (compacting) | 235 | 34/91/110 | 0.924 | ok | runs/exp09_fastgen_s0/summary.json |
| exp10_fastiter_fastgen_s0 | 0 | exp08 + fast sampler (6 rounds) | 252 | 40/87/125 | 0.910 | ok | runs/exp10_fastiter_fastgen_s0/summary.json |
| exp11_retain2k_s0 | 0 | exp10 + retain 2000 (7 rounds, last shrunk) | 255 | 62/85/108 | 0.871 | ok | runs/exp11_retain2k_s0/summary.json |
| exp12_pertarget_s0 | 0 | exp10 + equal weight per solved target (12 records) | 268 | 71/87/110 | 0.899 | ok | runs/exp12_pertarget_s0/summary.json |
| exp13_textbook_s0 | 0 | exp10 + textbook exploration stream (830 textbook targets, k32, T1.0, from r2) | 324 | 109/86/129 | 0.896 | ok | runs/exp13_textbook_s0/summary.json |
| exp14_focus_s0 | 0 | exp10 + frontier focus from r4 (k48, T1.0, <2 proofs) | 269 | 44/94/131 | 0.897 | ok | runs/exp14_focus_s0/summary.json |
| exp13_textbook_s1 | 1 | exp13 replicate | 269 | 67/85/117 | 0.910 | ok | runs/exp13_textbook_s1/summary.json |
| exp15_tb_pertarget_s0 | 0 | exp13 + per-target weight 12 | 303 | 104/89/110 | 0.897 | ok | runs/exp15_tb_pertarget_s0/summary.json |
| exp16_schema_s0 | 0 | exp13 + schema-aware textbook allocation (cracked schemas k8) | 324 | 109/90/125 | 0.907 | ok | runs/exp16_schema_s0/summary.json |
| exp17_tbr1_s1 | 1 | exp13 + textbook stream from round 1 | 331 | 115/94/122 | 0.891 | ok | runs/exp17_tbr1_s1/summary.json |
| exp18_tb_focus_s0 | 0 | exp13 + gen frontier focus from r4 (k32, T1.0) | 300 | 76/94/130 | 0.899 | ok | runs/exp18_tb_focus_s0/summary.json |
| exp17_tbr1_s0 | 0 | exp17 replicate | 236 | 34/90/112 | 0.886 | ok | runs/exp17_tbr1_s0/summary.json |
| exp19_tbhot_s0 | 0 | exp16 alloc + tb from r1 + T1.2 on uncracked schemas | 328 | 115/91/122 | 0.897 | ok | runs/exp19_tbhot_s0/summary.json |
| exp21_hot_focus_s0 | 0 | exp19 + gen focus from r4 (k32, T1.0) | 323 | 107/93/123 | 0.908 | ok | runs/exp21_hot_focus_s0/summary.json |
| exp19_tbhot_s1 | 1 | exp19 replicate | 301 | 98/89/114 | 0.906 | ok | runs/exp19_tbhot_s1/summary.json |
| exp21_hot_focus_s1 | 1 | exp21 replicate | 318 | 98/94/126 | 0.898 | ok | runs/exp21_hot_focus_s1/summary.json |
| exp22_tbbig_s0 | 0 | exp19 with tb_k 64, gen k 12 | 290 | 107/86/97 | 0.902 | ok | runs/exp22_tbbig_s0/summary.json |
| exp19_tbhot_s2 | 2 | exp19 replicate | 309 | 98/90/121 | 0.889 | ok | runs/exp19_tbhot_s2/summary.json |
| exp21_hot_focus_s2 | 2 | exp21 replicate | 300 | 96/91/113 | 0.883 | ok | runs/exp21_hot_focus_s2/summary.json |
| exp23_cheapfocus_s0 | 0 | exp21 with focus_k 16 (5 rounds) | 339 | 112/95/132 | 0.901 | ok | runs/exp23_cheapfocus_s0/summary.json |
| exp23_cheapfocus_s1 | 1 | exp23 replicate | 323 | 105/95/123 | 0.902 | ok | runs/exp23_cheapfocus_s1/summary.json |
