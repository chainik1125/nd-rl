---
written_on: 2026-10-06
written_by: agent:claude
status: complete
---

# Coconut-style continuous thoughts on run 198: a pre-RL gain that RL erases

Question (Robbie): does a recurrent, Coconut-style model [@Hao_2024] give test-time scaling on
natural-deduction proofs, and does the thoughts' training signal ("the curriculum issue") matter?

Answer: no test-time scaling. Feedback thoughts raise what the pretrained model proves before RL;
after four rounds of expert iteration the best arms are level with a pause-token control, and no
arm solves more with more thoughts at test time.

## Setup

- Model: autoresearch run 198's `pretrain.py` (sha `6bdbe0047ac4`, untied 6 blocks) plus T
  continuous thoughts between the theorem statement and the proof (`pretrain_coconut.py`,
  `ThoughtGPT`). A thought's input is the model's previous output. T = 0 is exactly 198's network.
- Training draws T from 0..8 per batch; the frozen harness (`2aa52f23e7ec`) samples at T = 8.
- The plan used by the supervised arms is each training proof's sequence of inference rules,
  parsed from the proof text. It never enters the tokenizer; the handed-off model is fully latent.
- One seed (0) per arm, one RunPod A40 each. Seven pods, 58 to 102 minutes, $4.98 in total.

| `ND_ARM` | What trains the thoughts |
|---|---|
| `none` | proof loss only |
| `pause` | control: one learned vector per thought, no feedback [@Goyal_2023] |
| `probe` | a pretraining-only head reads rule t off thought t |
| `curriculum` | Coconut's schedule: plan steps start explicit and turn latent in stages |
| `codi` | self-distillation from an explicit-plan pass [@Shen_2025] |
| `future` | thought t predicts the model's state at the proof's t-th `;` |

The three citekeys above are not in `references.bib` yet.

## Results (observed)

| Arm | Pretrain steps | Round-1 sample accuracy | RL targets after round 4 | Harness score |
|---|---|---|---|---|
| Run 198 (no thoughts) | 5,363 | 8.1% | 1,164 | 680 |
| `none` | 2,432 | 12.9% | 1,077 | 617 |
| `pause` | 2,339 | 7.6% | 1,087 | 610 |
| `curriculum` | 2,619 | 19.9% | 1,052 | 601 (see limits) |
| `future` | 2,669 | 7.4% | 981 | 519 |
| `probe` | 2,765 | 18.3% | 989 | 516 |
| `codi` | 1,550 | 11.0% | 921 | 447 |

Problems solved after RL by test-time thought count (0, 1, 2, 4, 8, 16, 32):

| Arm | textbook72 (of 72, k=256) | long dev (of 329, k=64) | RL unsolved (k=128) |
|---|---|---|---|
| `none` | 16 16 15 14 15 13 12 | 85 86 84 86 84 85 80 | 64 61 65 61 64 62 60 of 423 |
| `pause` | 19 19 19 20 19 18 18 | 80 77 75 77 75 77 78 | 61 50 60 54 52 57 61 of 413 |
| `curriculum` | 14 15 15 14 15 16 12 | 73 72 73 74 74 76 70 | 67 73 68 69 66 67 63 of 448 |
| `probe` | 14 14 14 15 15 13 12 | 54 55 57 58 57 54 47 | 82 80 73 78 74 76 49 of 511 |
| `future` | 14 14 14 13 13 12 3 | 60 55 58 67 61 58 50 | 63 65 72 62 64 59 46 of 519 |
| `codi` | 13 13 13 12 12 12 11 | 37 43 43 41 36 42 32 | 78 80 85 90 82 72 59 of 579 |

Run 198's final checkpoint solves 18 of 72 and 110 of 329 on the first two pools.

- Before RL, feedback beats the control on sample accuracy (12.9% to 19.9% against 7.6%).
- After RL, `none`, `pause` and `curriculum` are level; `probe`, `future` and `codi` are lower.
- After RL, solved counts are flat from 0 to 8 thoughts in every arm and pool, and flat or lower
  at 16 and 32.

## Interpretation (not checked)

The harness's RL fine-tune cannot supervise the thoughts. Arms trained to put a plan in them lose
that signal during RL, which would explain why `probe` leads before RL and trails after it, and
why `curriculum`, whose supervision is already off by the end of pretraining, does not lose
ground. A test would keep the plan loss on during RL, which needs a forked harness.

## Limits

- Every arm trains at about half of 198's steps per second (per-row attention masks), so the
  arms are comparable with each other, not with 198. 198's code without its speed machinery
  scored 612 at 2,100 steps and 604 at 2,519 (autoresearch runs 200 and 201, three-seed means).
- `curriculum` hit the harness's 2,400 s limit during its dev eval; this arm samples at about
  150 samples/s against 250 to 280. Its 601 is the harness's own eval run afterwards on the saved
  round-4 weights (`finish_eval.py`), and its status stays `timeout`.
- `probe` and `curriculum` ran `pretrain_coconut.py` at sha `c7eeaa31ac17`, the other four at
  `a8822ed7ff89`. The two differ in one line: whether a thought step calls the block through
  `torch.compile` or eagerly. The arithmetic is the same; a 100 s benchmark gave 10.9 and 10.5
  steps per second. The committed file is `a8822ed7ff89`.
- `codi` runs two passes per batch and got 1,550 steps.
- One seed per arm; 198's three seeds span 670 to 694.

## Reproduce

`python launch.py go <arm>` per arm (rents a pod, runs `coconut_remote.sh`, pulls, terminates),
then `python report.py OUT.html`. `check_model.py` verifies that the sampling and training paths
agree and that T = 0 reproduces 198's forward pass. Outputs, checkpoints included (1.4 GB), are
in the local `artifacts/coconut_spike_20261006/` and have not been uploaded to Hugging Face.
