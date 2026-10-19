---
written_on: 2026-10-19
written_by: "agent:claude"
papers:
  - liuProRLProlongedReinforcement2025
---

# ProRL: RL compute and the scale-dependence of its expansion verdict

Paper: [@liuProRLProlongedReinforcement2025]
Source: https://arxiv.org/abs/2505.24864v1 (the latest version). Also the NVIDIA ProRL v2 blog (Aug 2025) and the HF model card. Extends nd-rl `docs/literature/2026-10-02-lit-review-2/notes/liu2025-prorl.md`; ids are in `ledger/C_threads.tsv`.

## Learnings

Compute, as the paper states it:
- The run used 32 H100s, and "the whole training runs for approximately 16k GPUs hours.".
- "We set batch size to 256 and mini-batch size to 64 (equating to 4 gradient updates per rollout step)". Each prompt got 16 rollouts (32 in runs 6–7).
- Responses were capped at 8k tokens. "In the final stage (~200 steps), we increase the context window to 16k tokens".
- More than 2k steps (Fig. 2 axis to 2500) over 136K prompts. Epochs, prompt length and mean response length are not reported.

The follow-up models:
- v2 trained for "3,000 RL steps (trained with 8K context)", at 512 prompts per step per BroRL's baseline row (C-BRORL-baseline_row). *[review]* Unclear: JustRL's Table 4, also secondary, gives ProRL-V2 a batch of 256. At 256, v2's high bound falls from 2.1 × 10⁻² to ≈ 1.6 × 10⁻²; "≤ ≈ 2 × 10⁻²" still holds.

Scale dependence: three checkpoints (base, an intermediate of unstated step, final), scored at pass@128 from 256 samples.
- On plateau tasks: "Comparing the intermediate and final checkpoints shows that ProRL offers negligible additional benefit".
- *[review]* On boxnet: "extended training sustains and amplifies performance gains consistently across all values of k". The sentence is about boxnet only; for code the paper reports continued improvements, with no k.
- On math, pass@128 often falls below the base (C-PRORL-diminish_math).
- The v2 blog says pass@k climbs "over thousands of RL steps" but gives no k and no numbers.

## Evidence and limitations

Interpretation:
- With no step-resolved large-k curve, any claim that expansion grows with steps rests on one unidentified checkpoint and the v1→v2 comparison.
- Invisible Leash (2507.14843v5) re-evaluates v1 and v2 at k = 4096/8192 on math and up to 16384 on Reasoning Gym. Both show net shrinkage (NSCR −0.05). Going from v1 to v2 (2,000 → 3,000 steps) adds 12 expansions and 12 shrinkages: "ProRL-1.5B-v2 loses 175 questions while gaining 48" (C-LEASH-prorlv1_row, C-LEASH-v2_ratio).
- The paper says that on boxnet "the base model exhibits no capability of solving the task." With format-tolerant extraction the base reaches 0.345 at k = 16384, against 0.492 for ProRL (C-LEASH-boxnet16384, C-LEASH-format_fix).
- RL Squeezes (2509.21128v2) finds fewer unique correct trajectories after RL from SFT checkpoints, ProRL included. *[review]* Unclear: after "applying RL from either Base model or SFT model", the paper says "we also observe a reduced number of correct trajectories", but its SFT+RL summary and the AceReason-14B appendix speak only of incorrect trajectories or minimal change.
- Revision trap: the HF main revision has served v2 since 2025-07-23 (v1 needs `revision="v1"`); RL Squeezes describes its checkpoint as "2500 steps".
- Contamination: the base is Qwen2.5-Math-1.5B (18T + >1T math tokens) fine-tuned on 800k samples from the RL-trained DeepSeek-R1 (C-BASE-r1d15_base, C-BASE-r1d_800k). The instruction-following prompts are synthetic.

## Connections and questions

- See [hu2025-brorl.md](hu2025-brorl.md), [wu2025-invisible-leash.md](wu2025-invisible-leash.md) and [matsutani2025-rl-squeezes.md](matsutani2025-rl-squeezes.md).
- *[review]* ProRL v1 (about 16k H100-hours) is not the largest open run in this set with a large-k test. Ring-Zero (2607.12395v2: a 1T MoE with 63B active, 320 H200 for 3,600 steps, pass@1024 tracked, though the panel does not name the model) is almost certainly larger in absolute compute; its GPU-hours and share are not stated. ProRL v2 is also larger than v1. Independent evaluation at k 1,024–16,384 still finds net narrowing; k ≥ 4,096 holds for maths only. The expansion that survives sits mostly in synthetic non-math tasks (NDR 0.036 non-math vs 0.007 math; C-LEASH-v2_expand). This covers 1.5B models at 2–3k steps, not frontier-scale RL.
