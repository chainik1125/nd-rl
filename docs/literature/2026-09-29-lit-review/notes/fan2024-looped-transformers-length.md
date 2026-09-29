---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# Looped transformers with adaptive depth for length generalisation
Paper: Looped Transformers for Length Generalization (Fan, Du, Ramchandran, Lee, 2024)
Source: https://arxiv.org/abs/2409.15647v5 (version reviewed)
## Learnings
- Defines n-RASP-L: tasks solvable as P_pre ∘ (P′)^{T(n)} ∘ P_post with each piece in RASP-L (Def 3.1, §3).
- Model (§4.2.1): one decoder block reused T times, "input injection", NoPE. Trained with full-output prediction (FOP, not next-token), with the loss applied after the known number of steps T_i per example; T(n) must be available at train time (§4.1).
- Inference stops by oracle T or "maximum confidence" (§4.3).
- Results (§6.2, Fig 4): Parity trained to 20 digits "generalize to more than 40 digits near perfectly"; addition and copy near-perfect at max training length +10 where NTP fails. NTP with weight-tied layers and pause tokens improve less.
- Uses a length curriculum during training "for all methods" (§6.1.3).
- Limitations (§7): "does not support tasks that require multiple loops followed by each other"; needs "the ground-truth number of steps in the training data"; limited compute.
## Evidence and limitations
Toy algorithmic tasks; lengths ≤ ~50; seed counts not checked; Figure data not read.
## Connections and questions
B1 only in principle. Our output is autoregressive multi-line text with a variable, search-dependent number of steps; FOP with a known T(n) does not fit whole-proof generation. Nearest fit is the state-env arm (one step per call), where recurrence over steps is already supplied by the environment loop — which is the external analogue of looping. Not recommended for Stage-1.
