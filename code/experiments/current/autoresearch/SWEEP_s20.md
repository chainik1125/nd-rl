# Sweep s20: five configs x three seeds at the 20-minute budget (2026-09-22)

Budget 1200 s total, 300 s pretrain, 4 EI rounds, 64 eval attempts. One A40 pod per run via sweep.py.
`dev>=7` = dev-transfer theorems solved at true length >= 7 (the loop metric); `RL solved` = cumulative
RL targets solved after round 4 (secondary).

| config | n | dev>=7 per seed | mean | sd | RL solved per seed | mean | sd | total s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 3 | [122, 133, 76] | 110 | 30 | [460, 499, 357] | 439 | 73 | [880, 885, 864] |
| length-buckets | 3 | [28, 125, 90] | 81 | 49 | [174, 454, 362] | 330 | 143 | [951, 884, 880] |
| lr-3e-3 | 3 | [58, 20, 11] | 30 | 25 | [259, 157, 141] | 186 | 64 | [893, 850, 800] |
| no-wd | 3 | [145, 14, 137] | 99 | 73 | [494, 166, 483] | 381 | 186 | [893, 745, 845] |
| rel-tokenizer | 3 | [9, 6, 26] | 14 | 11 | [127, 114, 144] | 128 | 15 | [828, 781, 817] |

pooled within-config sd: dev>=7 43.5, RL solved 113.8  (df=10)
