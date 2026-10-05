---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - mukherjee2025reinforcement
---

# Reinforcement learning finetunes small subnetworks in LLMs

Paper: [@mukherjee2025reinforcement] (Mukherjee, Yuan, Hakkani-Tür, Peng, 2025)
Source: arXiv 2505.11711v2 (HTML rendering read: abstract, Sec. 1-7 incl. footnote 5; appendices not read)

## Learnings

- **Claim.** Large RL gains "result from updating only a small subnetwork comprising just 5%-30% of the
  parameters, with the rest effectively unchanged" (Abstract), across 7 RL algorithms (PPO, GRPO, DPO, ...)
  and 10 LLMs, with no sparsity regulariser. "the updates to almost all parameter matrices are nearly
  full-rank" (Abstract); mean update rank 99.2-99.8% of maximum, e.g. 99.4 for DeepSeek-Math-7B GRPO
  (Table 2). Sparsity is spread over all layers except layer norms (Fig. 3).
- **Measure.** Update sparsity = 1 − ‖θ1 − θ0‖_0 / n, where two "bfloat16 values as equal when their
  absolute difference does not exceed" 10^-5 (Sec. 2.2).
- **Numbers.** "68.5%–96.0% of parameters remain unchanged after RL"; "Deepseek-R1-Zero presents a update
  sparsity of 86.0%" after >8K GRPO steps from the base (Sec. 3, Table 1). By contrast "SFT induces dense
  updates (only 6%-15% sparsity)" (Sec. 3, Fig. 1). PRIME: "72% parameters are never updated, 8% have
  gradients canceling each other out, and 20% constitute the subnetwork" (Fig. 2 caption).
- **Subnetwork-only fine-tuning reproduces the model** (Conjecture 1; Sec. 4): masking gradients to the
  final RL subnetwork and re-training from the same start gives θ_sub with test scores matching or above
  θ_full (Table 3); "In DPO, 94.0% weights are same between" θ_full and θ_sub (90.5% for PRIME), and they
  "are 100% identical when using a tolerance of" 10^-4 (Sec. 4).
- **Consistency.** Subnetworks overlap well above a random-guess baseline: "varying the random seed yields
  overlaps of" o1 = 60.5%, o2 = 60.6% (random: 36.7%); with seed, data and algorithm all changed "we still
  observe notable overlaps of 59.1% and 33.2%" (Sec. 5, Table 4).
- **Cause: in-distribution data, not KL.** GRPO with KL: 69.8% sparsity, without KL: 68.8% (Sec. 6). "SFT on
  in-distribution data produces sparse updates, while DPO with out-of-distribution data produces dense
  ones"; rejection-sampling SFT on Qwen2.5-Math-7B "yields around 90.0% update sparsity" (91.2 in Table 5),
  RAFT++ (iterative rejection-sampling SFT) 69.4% (Sec. 6, Table 5). Rationale: "when gradients are
  computed on sequences that the policy already assigns high probabilities to, little update to the
  parameters would be needed" (Sec. 6). Sparsity declines slowly with training, levelling near 80% for
  PRIME (Fig. 5).
- **Important caveat in the paper itself (footnote 5).** "if one were to perform backpropagation manually on
  paper with unlimited numerical precision, the resulting parameter updates would be dense"; the observed
  sparsity comes from updates too small to register at finite precision.

## Evidence and limitations

- Evidence: public checkpoint pairs (Table 1), controlled DPO/PRIME runs (Tables 3-5, Fig. 4-6).
  Authors vary one factor at a time and partly rely on public checkpoints (Sec. 7).
- The measure is tied to bf16 storage and a 10^-5 tolerance (footnote 5; Table 6 for other tolerances,
  not read). With fp32 weights and Adam, nearly every parameter with any gradient would move.
- Independent check by another paper: Shenfeld et al. (2509.04259v1, Sec. 6) report that "the reason for the
  observed sparse updates was the use of bfloat16 for model training" and that the same training in float32
  gave identical performance "without any sparsity in their weight updates" (see `shenfeld2025razor.md`).

## Connections and questions

- **Definition offered:** none of capability; a size-of-update measure (L0 sparsity at a precision
  tolerance, plus update rank), and the empirical rule that update size tracks how *on-policy* the
  training data are.
- **New vs better access:** not claimed. The result is consistent with "RL mostly reweights what the policy
  already produces", but the paper's own explanation (in-distribution data ⇒ small updates) means sparsity
  reflects *how* the data were generated, not whether the behaviour is new: rejection-sampling SFT, i.e.
  our expert iteration, is sparse by construction. So sparsity alone cannot separate created from
  elicited. What can be borrowed: (1) the **subnetwork test** — if fine-tuning pend restricted to a small
  mask (from RL or random of equal size) on the excluded-middle demonstrations reproduces r16's ability,
  the change is small in parameter space; compare against the mask size needed for a control skill;
  (2) **per-round diffs** — whether the rounds where one seed acquired A ∨ ¬A (13-14) show a larger or
  differently located update than the same rounds in seeds that did not.
- **Null / floor:** random-mask overlap baseline (App. E) for subnetwork consistency; nothing for the
  random-weights objection.
- **Transfer to our setting:** read checkpoints pend, r8, r12, r16 (and per-round EI checkpoints if
  kept); compute relative-change sparsity |Δθ| > τ|θ| for a sweep of τ (our weights are probably fp32 —
  check), rank of per-matrix updates, overlap across seeds vs random. Cost: minutes on CPU. Failure modes:
  precision/optimizer artefacts dominate; Adam moves every parameter with a non-zero gradient by ≈ lr;
  sparsity confounds data on-policy-ness with novelty.
- Related: `aghajanyan2020intrinsic.md` (low-dimensional rather than sparse updates), `shenfeld2025razor.md`
  (KL to base as the size of change; same on-policy explanation), `lin2023urial.md` (output-side shift),
  `_screen_L3.md` row for Zhu et al. 2025 (The Path Not Taken).
