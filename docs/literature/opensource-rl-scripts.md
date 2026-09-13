---
written_on: 2026-09-13
written_by: "agent:codex"
papers: []
---

# Open-source RL training scripts

Sources: training scripts, configurations, and release documentation linked below, inspected on September 13, 2026. This is a review of software releases; no training run has been reproduced.

## Learnings

**OLMo is not the only project that releases a concrete RL training recipe.** Prime Intellect, NVIDIA, and the DAPO authors also publish scripts or configurations tied to model-training results. The useful unit to compare is the implemented algorithm together with its launch configuration, starting checkpoint, data, rewards, and evaluation procedure. A framework name alone does not specify these choices.

**Our recommended starting point is OLMo 3's 7B RL-Zero Math recipe.** It begins with a base model, making it useful for studying what RL adds before introducing a reasoning-specific SFT stage. For recent large-scale proof training, NVIDIA's IMO 2026 recipe is the strongest candidate from this review to study alongside it. These are recommendations for our research question, not evidence that either recipe achieves the best results under matched model, data, and compute budgets.

| Recipe | Concrete code to read | What it contributes |
| --- | --- | --- |
| **OLMo 3 RL-Zero Math** | [`7b_rlzero_math.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_rlzero_math.sh), calling `open_instruct/grpo_fast.py` | Specifies a base 7B model, Dolci-RLZero-Math data, eight samples per prompt, active sampling, learning rate `1e-6`, zero KL coefficient, and a 16K response limit. This is the first script we would adapt to proof verification. |
| **NVIDIA Nemotron IMO 2026** | [`run_proof_v1.sh`](https://github.com/NVIDIA-NeMo/RL/blob/imo-26-ultra-v3/run_proof_v1.sh) and the [proof-training guide](https://github.com/NVIDIA-NeMo/RL/blob/imo-26-ultra-v3/docs/guides/nemotron-3-ultra-imo.md) | A released proof-generation RL stage with a public dataset, asynchronous GRPO, asymmetric clipping, importance-sampling corrections, and proof-judge orchestration. Its natural-language proof rewards use model judges. |
| **Prime Intellect INTELLECT-3.1** | [`examples/advanced/intellect-3.1/rl.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/advanced/intellect-3.1/rl.toml) | Specifies the starting model, Muon optimizer, task mixture, filtering, evaluation, and training/rollout layout. It mixes math, logic, code, software engineering, and search tasks, providing a reference for training across environments. |
| **DAPO** | [`run_dapo_qwen2.5_32b.sh`](https://github.com/verl-project/verl-recipe/blob/main/dapo/run_dapo_qwen2.5_32b.sh) | An explicit verifiable-reward baseline with asymmetric clipping, token-level loss, dynamic sampling, and penalties for long responses. Older than the other recent releases, but useful for isolating algorithmic choices. |

## Evidence and limitations

**OLMo provides unusually useful provenance.** Its [release index](https://github.com/allenai/open-instruct/tree/main/scripts/train/olmo3) maps scripts to original commits and training logs, including RL-Zero and Think runs. However, the launchers target Ai2's Beaker infrastructure; the documentation says external users must adapt the launcher and configure their own Ray cluster. OLMo's visibility into the wider training history is valuable for studying prior exposure, but it does not by itself prove that a particular capability was absent before RL. Historical script paths and exact reproduction checkouts still need verification.

**NVIDIA's proof recipe is a large-cluster reference.** Its default allocation is 256 four-GPU GB200 nodes for training and rollout, plus eight two-node judge services. Reducing these numbers does not establish that the model fits a smaller allocation. The [guide](https://github.com/NVIDIA-NeMo/RL/blob/imo-26-ultra-v3/docs/guides/nemotron-3-ultra-imo.md) describes the proof-generation stage and links the released data; it is not a complete reproduction of every preceding training stage or the competition inference system. Replacing learned proof judges with our deterministic checker changes the learning problem.

**INTELLECT-3.1 has a concrete reproduction configuration.** The [configuration documentation](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/configuration.md#examples) identifies the file as reproducing that run. The [model release](https://huggingface.co/PrimeIntellect/INTELLECT-3.1) states that its training and evaluation environments are available. We have inspected the configuration, but have not validated every environment dependency, service, or checkpoint against the original run.

**DAPO distinguishes reproduction from ongoing maintenance.** Its [README](https://github.com/verl-project/verl-recipe/blob/main/dapo/README.md) provides reproduction results, hardware, container versions, commits, and logs, and distinguishes the historical reproduction branch from the maintained recipe. Changing the model, sampling, or reward can invalidate direct comparisons with those results.

The links above include mutable branches. Before running an experiment, record the chosen commit, resolved configuration, dataset revisions, starting checkpoint, and verifier version. Published scores across these releases use different models, data, budgets, and evaluation procedures; this review does not establish a universal winning script.

## Connections and questions

This complements the [reading guide on learning beyond training data](astra_suggestions/learning-beyond-training-data.md). The key question is whether a recipe produces transferable proof procedures rather than only improving success on the training distribution.

The next concrete step would be to adapt the OLMo RL-Zero recipe to binary proof verification and a fixed structural-transfer benchmark, then compare a DAPO-style variant while holding the starting model and total generation budget fixed. Track held-out success across sampling budgets, proof diversity, and training-task difficulty. Active sampling and rejection of uninformative rollout groups change which experience the model receives, so their generation costs and selection effects must be included in that comparison. This is a proposed experiment, not a result or a decision to replace the current pipeline.
