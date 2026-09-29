The final CPU audit passed. All 196,608 saved attempts from 32 models on 768 final sources were checked, and the aggregate agrees exactly with the root report. The population contains 402 canonical contexts. No model weights were loaded and no GPU work was performed.

The four main RL rows below use the prespecified 4e15 proxy endpoint. Other rows are separately budgeted baselines and controls; their budget and exposure labels remain those in the frozen protocol.

| Model | Sources solved in eight attempts | Mean whole-reference bits |
|---|---:|---:|
| adapt | 351 / 768 | 107.4715 |
| selected_full_s1_f40 | 269 / 768 | 159.5810 |
| selected_reset_s1_f40 | 274 / 768 | 148.8818 |
| uniform_full_s1_f40 | 274 / 768 | 173.1674 |
| uniform_reset_s1_f40 | 279 / 768 | 161.0268 |
| retention_step8827 | 299 / 768 | 204.9189 |
| retention_step9054 | 300 / 768 | 205.3607 |
| retention_step9056 | 300 / 768 | 205.3664 |
| retention_step9096 | 304 / 768 | 205.3933 |
| retention_compute | 297 / 768 | 218.4631 |
| main_oracle_matched | 314 / 768 | 57.4844 |
| main_oracle_total | 307 / 768 | 58.7687 |
| transfer | 372 / 768 | 98.2249 |
| surprise | 362 / 768 | 99.3646 |
| uniform_matched | 369 / 768 | 99.0376 |
| uniform | 375 / 768 | 96.6118 |
| oracle_matched | 374 / 768 | 28.5116 |
| oracle_total | 329 / 768 | 51.9071 |
| prefix_mined | 360 / 768 | 98.2080 |
| prefix_reference | 354 / 768 | 99.0053 |

Primary final coverage contrasts, in percentage points:

- selection_main_effect_f40: -0.6510; paired context bootstrap range [-3.0939, +2.1334].
- reset_main_effect_f40: +0.6510; paired context bootstrap range [-1.2238, +2.6389].
- interaction_f40: +0.0000; paired context bootstrap range [-4.2289, +4.7514].

All four main final arms have lower coverage and higher whole-reference NLL than the adapted checkpoint. Relative selection/reset effects on reference NLL do not establish improved proof-solving performance. This audit verifies the declared computations; there is one online training seed, and context bootstrap intervals do not estimate variation across training seeds.

The exposure-matched retention checkpoints use the same ordered retention stream and update counts as their corresponding main arms. They are not compute-matched. The separate retention_compute endpoint uses the full declared 4e15 proxy budget.

All source caps, reference length bins, and certificate length counts use total written ND proof lines, including PR premise and AS assumption lines. Theorem prompts and PRF/FINAL/QED markers are excluded. Reference lengths are not minimum proof lengths.

The first aggregation copy sorted coefficient keys and therefore changed 78 bits/token contrast values only at floating roundoff (maximum 2.78e-17). Version 2 preserves the frozen coefficient order and matches root exactly. Both earlier exports remain intact; no tolerances or frozen analysis code were changed.

Audit artifacts: receipt.json, strict_replay.json, paired_analysis.json. The final gate and addition-order regression suite passed all four tests.
