---
written_on: 2026-11-27
written_by: agent:claude (opus, orchestrator)
---

# Olmo minimal adapter: orchestrator's draft design (input to the design discourse, not the pre-registration)

Written 2026-11-27 1:32 am PT. Status: draft. The design discourse (`discourse/edl_design.md`) may change any of it;
the pre-registration (`PREREG.md`) is written only after the discourse.

## Questions (from Leon, 2026-11-27)

- **A.** How small an adapter, and how few examples, bring a checkpoint to the final model's plain pass rate on the
  48 reach prompts?
- **B.** Does that still work when no training token comes from an RL-trained model later than the checkpoint?

Background: the 2026-11-25 run (`../olmo_elicitation_20261125/`, summary
`experiment-summaries/2026-11-25-olmo-elicitation-curves/`) trained one LoRA (rank 16 on q, k, v, o, gate, up, down
of all 64 layers: 134,217,728 trained weights) for one pass on 64 of the final model's (`s2300`) strict-pass
responses, thinking included. It lifted the pre-RL model `dpo` from 0 % to 63.5 % [53.7, 73.2] checker passes on the
48 reach prompts (8 draws each); the final model's plain rate is 69.4 % [60.7, 77.7]. Two faults (Leon): no sweep
over adapter size or number of examples, and the examples were the final model's own responses (distillation).

## Fixed points (from the brief)

- Score: Dan's strict IFBench checker on the 48 reach prompts (never in training), 8 draws each, his sampler
  settings unchanged (vLLM 0.11.0, T 1.0, top-p 1.0, 32,768 max tokens, olmo_thinker template), vLLM's LoRA path, a
  teacher-forced gate that vLLM applies each adapter as PEFT does. Likelihoods are secondary.
- Stage 1 is `dpo` only; other checkpoints (`s0100`, `s0500`, `s1100`) only with the cells stage 1 shows are
  informative.

## Example sources

- **(i) own passes (main).** The checkpoint's own strict-pass responses (thinking + answer, stored token ids) to
  arm-A prompts that are not reach prompts, from Dan's stored draws of that checkpoint. Rules as the 2026-11-25 pool
  (`build_pool.py`): strict and `finish == stop`, has `</think>`, not hand-labelled gamed, at most 6,144 response
  tokens; one response per prompt, a genuine-labelled one if any else a seeded pick. No token comes from a later RL
  checkpoint. The information added is the checker's selection (rejection sampling), which is the point of the
  condition: it is what an RL-free procedure with the same verifier could do.
- **(ii) final model's responses.** `s2300`'s strict passes on the SAME prompts (same rules). The reference for how
  much of the old result was distillation.
- **(iii) independent answers: proposed NOT to run.** IFBench has checkers and no reference answers. A Think model's
  target must include a thinking segment; rule-written or empty thinking is far off the model's distribution, so an
  adapter would mostly learn to skip or shorten thinking, and rule-written answers that satisfy a format checker are
  checker gaming by construction. Claude-written reasoning traces for training another model are blocked in this
  project (an earlier attempt hit a safeguard). Recorded as not run, with these reasons.

Counts from Dan's stored draws (cap 6,144): `dpo` has an eligible own pass on 126 of the 176 arm-A non-reach prompts,
`s2300` on 125; both on 109. So at `dpo` the matched set is 64 train + 32 held-out ("dev") prompts from the 109.
Median response length: own passes 2,090 tokens, `s2300` 3,879 on the same prompts. Other checkpoints: `s0100` ∩
`s2300` = 85 prompts, `s0500` 103, `s1100` 108 (so `s0100` would need a smaller dev set).

Training subsets: one seeded order of the 64 train prompts; n = 1, 4, 16, 64 are its nested prefixes; the same
prompts for both sources.

## Adapter-size ladder (P = trained weights; all exported as ordinary PEFT LoRA files that vLLM serves)

| name | what is trained | P |
|---|---|---|
| A16 | LoRA r16 on q, k, v, o, gate, up, down, all 64 layers (the old adapter) | 134,217,728 |
| A1 | LoRA r1 on the same modules | 8,388,608 |
| O1 | LoRA r1 on o_proj only, all 64 layers ("rank 1 on a single module") | 655,360 |
| M4k | rank-1 all-module LoRA, A frozen at its seeded random init, a seeded uniform random 4,096 entries of B trained, all other B entries fixed at 0 | 4,096 |
| M256 | same, 256 entries of B | 256 |

M-cells follow "uniformly randomly sampled sparse subsets of parameters within rank 1" (Bits That Count §3.3, as
read on 2026-11-25; to be re-checked in full text). Masked B entries get zero gradient, so AdamW (weight decay 0)
leaves them exactly 0; the fixed A is a seeded random matrix (zero trained bits). Export: an r1 LoRA on all modules
(34 MB, mostly zeros).

## Training rule (one rule for every cell)

- AdamW (0.9, 0.999, eps 1e-8, weight decay 0), constant lr, batch 1, grad clip 1.0, loss = mean token
  cross-entropy over the response tokens (thinking + answer), bf16 base, fp32 adapter, dropout 0.
- 64 optimizer steps for every cell, cycling through the n examples in seeded epoch orders (n = 64: one pass;
  n = 16: 4 epochs; n = 4: 16; n = 1: 64 steps on one example).
- Held-out ("dev") loss on the source's 32 dev responses at steps {16, 32, 64} (n = 64) or {4, 8, 16, 32, 64}
  (n < 64).
- lr grid per family: A16 {1e-4, 3e-4, 1e-3}; A1 {3e-4, 1e-3, 3e-3}; O1 {1e-3, 3e-3, 1e-2}; M {1e-2, 3e-2, 1e-1}.
  The full grid at n = 64; for n < 64 the lr selected at n = 64 for that (family, source) (cost).
  Edge rule: a selected lr on a grid edge adds the next value beyond it once.
- **"Trained"** = the (lr, step) with the lowest dev loss per token. If no (lr, step) beats the untrained dev loss,
  the cell is reported as "did not train" and is not sampled.
- Online code (MDL) = the sum of each example's nats scored just before its first update (first epoch), as in the
  papers. EDL = MDL − (online tokens) × L_dev(θ*) per token. Bits per trained weight = EDL / ln 2 / P, and the
  capacity threshold in the Bits That Count sense: P* = the smallest P with EDL(P) / EDL(A16) ≥ 0.95, at n = 64.
  A16 is our reference (they used r512 or full fine-tuning): flagged wherever shown.

## Pass-rate readout

- Per sampled cell: strict pass rate over 384 draws, 95 % prompt-bootstrap interval; prompts with ≥ 1 pass in 8.
- Reference: the final model's plain rate on the 48 prompts, 69.4 % [60.7, 77.7] (Dan's 32 stored draws per
  prompt), and the checkpoint's plain rate (`dpo` 0.0 %).
- "Reaches" (primary): the cell's 95 % interval includes 69.4 % or lies above it. Secondary, stricter: the paired
  difference (cell − final plain, prompts resampled jointly) has a lower bound above −10 points. Also the share of
  the gap closed, (cell − plain) / (final − plain).
- "Smallest sufficient adapter" at n = 64 = the smallest P that reaches; "fewest examples" at a given size likewise.

## Sampling plan (sampling is the cost: about 384 draws × 4.5-7k tokens ≈ 20-30 min of H200 per cell)

Adaptive, with the rule fixed in advance and decided on reach results only through that rule:
- Round 1: own × A16 × 64, final × A16 × 64, own × M256 × 64, final × M256 × 64.
- Per source: if A16 × 64 does not reach, no further cells for that source. If M256 reaches, the smallest
  sufficient P is ≤ 256 (one more cell, M32, if budget). Otherwise bisect on the ladder in log P (O1 next, then M4k or
  A1).
- Examples: at A16 for each source that reaches at A16 × 64, n = 4 first, then 16 (if 4 fails) or 1 (if 4 reaches).
- Stage 1 stops at about 12 sampled cells.

## Cost (estimates)

Training (H100 80 GB, $4.39/h; step ≈ 3.6 s at 4.5k tokens, linear in length): own-source run ≈ 4.6 min, final ≈
7.2 min; n = 64 cells × 3 lrs: own ≈ 14 min, final ≈ 22 min per cell. Size ladder both sources ≈ 3 h; n-ladder
cells (one lr) ≈ 4-6 min each. Total ≈ 3.5 H100-hours ≈ $16.
Sampling (H200, $5.29/h, ≈ 1,450 tok/s, KV-bound): own cell ≈ 1.7 M tokens ≈ 20 min, final ≈ 2.6 M ≈ 30 min;
12 cells ≈ 5 h ≈ $27, less with several adapters in one vLLM engine (no per-job load and tail).
Stage 1 ≈ $40-45.
