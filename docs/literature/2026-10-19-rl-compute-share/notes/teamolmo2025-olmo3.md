---
written_on: 2026-10-19
written_by: "agent:claude"
papers:
  - olmoOlmo32025
  - allenaiOLMo3RLZero2025
---

# Olmo 3: wall-clock for pretraining vs RL, and the RL-Zero recipe

Paper: [@olmoOlmo32025] (proposed key; first author Team Olmo) and [@allenaiOLMo3RLZero2025].
Source: arXiv 2512.13961v2 (14 Apr 2026). The open-instruct script `scripts/train/olmo3/7b_rlzero_math.sh` does not
exist at the requested commit d928a7c, so I read it at 3af17aacff (first version, 2025-11-20) and bb58c664fb (the
Olmo 3.1 RL-Zero version, 2026-01-23). Ledger: `B-OL3-*`, `B-OL31-*`, `B-OL3rz-*`.

I chose Olmo 3 over V3.2 and V4. V3.2 only states that RL "exceeds 10% of the pre-training cost"; V4 gives no RL
compute; Olmo 3 gives GPU counts and durations for both stages.

## Learnings

What the paper states:

- **32B overall.** "In total, approximately 56 days elapsed from the start of training to the evaluation of the Olmo 3
  Think 32B checkpoint, on a cluster with 1024 H100 GPUs dedicated to Olmo 3."
  - Pretraining (~47 days): "The initial pretraining phase on 5.5T tokens took about 9.5 days on 512 GPUs, followed by
    an additional 35 days on 1024 GPUs."
  - RL: "The final RL runs for the initial Olmo 3 Think 32B spanned approximately 5 days".
  - Olmo 3.1: "we continued our best RL run for another 21 days on 224 GPUs to produce Olmo 3.1 Think 32B."
- **Ratio.** Qualitative only: "While pretraining accounts for the majority of total GPU hours, a non-trivial share is
  consumed by post-training".
- **7B data.** 5.93T pretraining, 100B mid-training and 50B long-context tokens, on 512 H100 at 7.7K tokens/s/device.
- **RL (Table 49).** Columns are 7B Think, 32B Think, 7B Instruct, RL-Zero.
  - "Training steps 1,400 750 450 2,000".
  - Prompts per batch 64 / 128 / 64 / 32; group size 8.
  - Response length 32,768 / 32,768 / 8,192 / 16,384.
  - "Num learner GPUs 16 64 8 8 Num actor GPUs 56 160 56 64".
  - OlmoRL (GRPO-based), "No KL loss", no critic.
- **RL-Zero scripts.**
  - 3.0: 16 × 16, "--response_length 12000", "--total_episodes 512000".
  - 3.1: 32 × 8, 16384 tokens, "--total_episodes 768000", "--num_nodes 9".

My interpretation:

- GPU-days are recoverable for 32B pretraining and for the 3.1 Think RL (21 days × 224).
- RL-Zero wall-clock is not stated. Table 49's caption says "GPU hours assume NVIDIA H100 accelerator", but no
  GPU-hours row exists in either the HTML or the PDF.

## Evidence and limitations

- **Create-vs-elicit (RL-Zero math).** "We also see a decent improvement in pass@32 results, demonstrating that our run
  maintains diversity and RLVR pushes the model beyond its initial capabilities."
  - Measured on AIME 2024/25 with k max 32 (modest).
  - A random-reward control shows no gains.
- **Contamination.** The RL-Zero base's midtraining includes reasoning traces, chosen "to lay the foundation for Olmo 3
  Think and Olmo 3 RL-Zero". Example row: "Thinking (synth) | QWQ Reasoning Traces ˆ | 4.77B | 438K | 1.87B (1.87%) | 401K".
- **Secondary paraphrase.** Lambert's RLHF book says "4 weeks on 200 GPUs".

## Connections and questions

- Compare DeepSeek's GPU-hours ([guo2025-deepseek-r1-compute.md](guo2025-deepseek-r1-compute.md)).
- Open: RL-Zero GPU-days, and how much of the post-training share evaluation consumed.
