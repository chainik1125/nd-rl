# Scale-up memo — olmo-think32b (2026-11-04, executor)

Models: `allenai/Olmo-3-32B-Think-DPO` `main` (the RL start) and `allenai/Olmo-3.1-32B-Think` `step_0100` / `step_0750` /
`main` (= `step_2300`); 32 B, bf16, pretrained → Think SFT → DPO → RLVR. AIME 2025–26, 59 headline problems; Think RL
template and decoding (T 1, top-p 1, 32,768 tokens). Prices are this run's measurements on 1 × H200 141 GB,
RunPod **secure** ($5.29/h billed; no cheaper capacity was available on 2026-11-03/04).

## What remains after 256

One reach candidate survived the fixed phase: **aime2025_28** (answer 248). Its status at 256 samples:

| | `dpo` (256 draws) | `step_2300` (8 fixed + 32 fresh) |
|---|---|---|
| credited by RL's verifier | 7 (2.7 %) | 12 (30 %) |
| final answer stated and right | 1 (0.4 %) | 10 (25 %) |
| hand check (uniform rule / strict) | 2 / 0 sound of 7 checked (1 / 0 stated in the write-up) | confirmed / confirmed |
| stated answers (the 240 extension samples) | 208 × 70, none 163 (100 of them cut at 32,768), 248 × 1 | 248 in 10 of 40 |

`step_0100` stays close to `dpo` here (256 draws: 9 credited by RL's verifier, 5 with the answer stated; 208 is its
commonest stated answer) and `step_0750` sits between (256 draws: 21 and 15). So aime2025_28 is the one problem where the 21-day extension's checkpoint is far more reliable than its start.
Whether the start "can" solve it depends on the grade: under RL's verifier it does (7 times); with the answer stated
it does once in 256; under a strict proof standard it never does (every credited `dpo` solution leaves the crux —
that the reduced numerator is divisible by 3 at every step — unproved or argues it from residue representatives).

No other problem is a candidate: the 7 problems no model solved in the fixed phase (aime2025_13, 14, 15, 30;
aime2026_15, 29, 30) were not solved by any RL checkpoint either (8 samples each; `dpo` 16), and most of their samples
run to the 32,768 cap (dpo: 15, 11, 10, 6, 2, 8, 0 of 16 cut).

## What a larger extension would cost (this run's measured prices)

Per-sample cost on aime2025_28 (E1a_dpo: 240 samples, 6.93 M tokens, 28,871 tokens per sample, 820 tok/s): **$0.052
per `dpo` sample** on a secure H200 ($0.035 at the $3.59/h community rate when available).

| design | extra `dpo` samples | cost (secure / community H200) | what it decides |
|---|---|---|---|
| aime2025_28 to 1,024 | 768 | $40 / $27 | expected ≈ 3 more final-answer successes at 1 / 256: confirms or refutes "dpo solves it with a stated answer" |
| aime2025_28 to 2,048 | 1,792 | $93 / $63 | ≈ 7 more; enough to hand-check a rigorous `dpo` solution if one exists at ≥ 0.1 % |
| `step_2300` 64 samples on each of the 7 unsolved problems | 448 (RL side) | ≈ $25 / $17 | finds further RL-only solves (new candidates) before any `dpo` extension |
| both: the 7 problems first, then `dpo` to 1,024 on whatever RL solves | 448 + 768 per new candidate | ≈ $25 + $40 per candidate | the next screening step at the same k-budget logic |

Hand checks are not in these costs (sub-agent reading; ≈ 1 reader per 2–3 solutions of ≈ 30 k tokens).

## Recommendation

The screening found one problem, not several, so the brief's "several problems stay out of reach at 256" condition for
a larger extension is not met. The cheapest informative next step is **aime2025_28 to 1,024 under the final-answer
grade (≈ $27–40)**: it is the only place where the DPO model's capability is genuinely in question, and the answer
(rare-but-present vs absent under a strict proof standard) is the elicit-versus-create distinction in miniature. A
broader search for candidates (`step_2300` at 64 on the 7 problems nobody solved) would cost about as much and is
the better use of money if Dan wants more candidates rather than a sharper answer on this one.
