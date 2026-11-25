---
written_on: 2026-11-25
written_by: agent:claude
papers:
  - donoway2026bitsthatcount  # proposed key, not yet in references.bib (see to_add_to_zotero.md)
  - donowayQuantifyingElicitationLatent
  - donoway2026excessdescriptionlengthlearning
---

# Elicitation metrics we have not yet run on Olmo 3.1 Think 32B: a ranked shortlist

Leon's request (2026-11-25): "have another opus look for literature on any other elicitation metrics we can try on
olmo". The setting is a set of measurements, each computed at every checkpoint of the Olmo 3.1 Think 32B RL run
(`dpo` = step 0 … `s2300`) and plotted as a curve over RL steps. Each says how much of the final model's instruction-following
behaviour is already latent and cheaply recoverable at that checkpoint. The current plan is in
`code/experiments/current/olmo_elicitation_20261125/SPEC.md` and `PREREG.md`.

This review builds on the earlier reviews on `origin/dan` and does not repeat them:
- `docs/literature/2026-10-02-lit-review-2/` (EDL theory note, ProRL, Yue);
- `docs/literature/2026-10-04-capabilities/` (elicitation practice, MDL probing, intrinsic dimension, probes,
  steering, crosscoders; 185 papers);
- `docs/literature/2026-11-06-olmo-novelty/` (pass@k and coverage work up to 2026-11).

Papers those reviews read are cited as "earlier review" and were not re-read here.

**Reading levels used below.**
- **full text**: I read the paper's text myself.
- **full text, in part**: a fetch tool extracted the quoted passages and section numbers from the full text, and I did not
  read every section.
- **abstract only**.
- **from memory, unverified**.
- **earlier review**: claims taken from an earlier review's note, not re-checked here.

Only one paper was read in full text: Donoway et al., "Bits That Count" (ICML 2026, PMLR 306:26127–26151), the EDL companion paper.

## Cost assumptions (ours, not measured)

- **Loading:** one checkpoint is 64.5 GB of bf16 weights and takes about 10 min to load.
- **Generation:** vLLM generates about 1,700 tokens/s. A response is about 4–5k tokens, mostly thinking, so this is about 1,200–1,500 responses per H200-hour.
- **Teacher-forced scoring:** I assume about 5k tokens/s. This is a guess: 2 × 32 B FLOP per token at about 40 % utilisation of an H200.
- **LoRA training:** I assume about 1.5k tokens/s forward plus backward, with gradient checkpointing.
- **Price:** community H200 at $3.59/h.

All costs below are ±2×. They exclude pod overhead and assume the checkpoint is already loaded for stage 1, so loading is not
charged twice.

## The shortlist, ranked

The ranking weighs four things: how directly the metric bears on create-versus-elicit, its cost on the 32B model, how well the
literature validates it, and how little it overlaps with what we or Dan have already run.

| # | metric | what it needs | H200-h (5 ckpts unless said) | overlap with existing work |
|---|---|---|---|---|
| 1 | Parameter-budget (capacity) ladder on the EDL data | SFT with tiny adapters, multi-epoch | ≈ 6 (4 ckpts) | extends our stage 1 (LoRA r16) and Dan's 16-trajectory LoRA |
| 2 | In-context description length (k demos of the final model) | forward passes, long context; optional sampling | ≈ 2.5 scoring; ≈ 5 with sampling | none on Olmo; Dan's hint and ladder are instructions, not demos |
| 3 | Token-level footprint: divergence and rank of the final model's tokens | forward passes (adopted into stage 1 storage) | ≈ 0 extra; ≈ 0.5 for the reverse direction | none on Olmo; our forced-first-action is its one-position version |
| 4 | Rank-1 and extrapolated weight deltas | SVD of weight differences; sampling of new weight variants | ≈ 4–5 | Dan's `olmo-weights` (norms only); our nd weight interpolation |
| 5 | Prompt-optimisation elicitation gap (GEPA-style) | sampling, about 700 rollouts per checkpoint | ≈ 2 per ckpt | Dan's hand-written instruction ladder and brevity hint |
| 6 | Decoding-only elicitation: temperature sweep, then power sampling | sampling (power sampling about 10× cost) | ≈ 1.5 (T sweep, 3 ckpts) | Dan sampled at T 1.0 only |
| 7 | Compliance probe and steering along it | activations; steered sampling | ≈ 1.5 per ckpt | our nd probes and steering, not on Olmo |
| 8 | Cross-sampling: how many of the final model's tokens are needed | two models co-resident; custom decoding | ≈ 3 (dpo only) | Dan's transplant is the "all tokens" end of it |
| 9 | Tiny-budget RL (steering-vector or TinyLoRA RL) | GRPO with 5k-token rollouts | prohibitive at 32B; 7B only | none |

### 1. Parameter-budget (capacity) ladder: how few trainable parameters recover the final model's way of answering

**What it measures.** At each checkpoint, adapters of shrinking size are fitted to the same 64 strict-pass responses of
`s2300`, and we find the smallest size that still absorbs as much generalisable information as LoRA r16. If the latent
skill is there, a tiny adapter is enough. If it has to be taught, the adapter becomes the bottleneck at a much larger size.

**Literature.**
- Donoway et al. 2026, "Bits That Count" (full text). This is the empirical companion to the EDL theory paper, and it uses capacity as
  its second elicit-versus-teach criterion. §6.2: "elicitation (Llama 3 and pre-taught TinyStories models) is associated with
  adapter capacity limits around 0.01–0.1 bits/parameter, whereas teaching saturates around 1 bit/parameter or more."
  - **Threshold definition** (Table 6 caption): EDL/P* is "the threshold where EDL/EDLref drops below 0.95". EDL_ref is
    rank-512 LoRA, "which obtains the same performance as full fine-tuning in all of our experiments" (footnote 2).
  - **Causal check** (§5.1): pre-teaching moves TinyStories-1B multiplication from about 1 to 0.05 bits/parameter.
  - **Reasoning distillation** (Table 7, Qwen2.5 on R1 traces of s1K prompts): 0.04, 0.05 and 0.02 bits/parameter for the
    1.5B, 14B and 32B models.
  - **Minimal-information elicitation** (Table 8): for Qwen2.5-32B "Reasoning", PGR is 0.55 with one example, 0.27 with 10
    parameters and 0.41 with 100 parameters.
- Donoway et al. 2025, NeurIPS (abstract only; the abstract is in `references.bib` as `donowayQuantifyingElicitationLatent`):
  "training as few as 10–100 randomly chosen parameters … can recover up to 50% of the performance gap … and 1,000s to
  10,000s of parameters can recover 95%". They fit a logistic curve to parameter count against gap recovery.
- Morris et al. 2026, TinyLoRA (arXiv 2602.04118, full text in part). With GRPO, Qwen2.5-7B-Instruct reaches 91 % on GSM8K with
  13 trained parameters (base 76 %). SFT reaches 83 % at 13 parameters and 84 % at 120 (§6.2). The abstract: "models
  trained using SFT require 100-1000x larger updates to reach the same performance".
- Sinii et al. 2025 (arXiv 2505.18706, abstract only; and 2509.06608, full text in part). One RL-trained d-dimensional bias per
  layer, with base weights frozen, "can match the performance of fully fine-tuned models". The mean of six math benchmarks
  is 42.9 against 43.8 for full RL on Qwen2.5-Math-7B, and 25.8 against 26.4 on Llama-3.1-8B-It (2509.06608 Tables 2–3).

**What we need.** SFT on the stage-1 train set (64 responses), scored on the stage-1 test set (32). The protocol is identical
to stage 1, plus the multi-epoch residual already adopted. Forward and backward passes go through the frozen 32B model. The
design is under Recommendation R1.

**Curve and reading.** We plot P*(t), the smallest adapter with capacity ratio ≥ 0.95, and EDL/P*(t) in bits per parameter,
over RL steps.
- **Gradual elicitation.** If P* is small and roughly constant from `dpo` on, the distance to `s2300` is low-information at every
  checkpoint.
- **Teaching at the start.** If P*(`dpo`) is an order of magnitude or more above P*(`s1100`), the start needs more information per
  parameter. This is a teaching-like start, relative to later checkpoints.

**Known failure modes.**
- **The thresholds may not transfer.** They were calibrated on other models, tasks and data scales (see R1).
- **Our targets are SFT targets.** TinyLoRA's 100–1000× gap between SFT and RL means an SFT capacity figure overstates the information an RL update needs.
- **Learning rate.** A tiny adapter needs its own learning rate.
- **Our reference is not theirs.** LoRA r16 is the reference here, not full fine-tuning.

**Closest existing work.** Dan's `olmo-ifladder` LoRA (r16 on 16 trajectories, at `dpo` only) and our stage 1 (r16 at every
checkpoint). Neither varies the parameter budget.

### 2. In-context description length: how much of the distance a few demonstrations remove, with no weight change

**What it measures.** We put k of the final model's responses to other prompts in the context and measure the
negative log-likelihood (NLL) of the final model's responses to held-out prompts. This is how much of the gap a checkpoint closes by imitation alone. It needs no
optimizer, learning rate or adapter, so it checks EDL independently of those choices.

**Literature.**
- Elmoznino et al. 2024 (arXiv 2410.14086; abstract and the §2.4 heading only). The abstract says "the next-token prediction loss used to train in-context
  learners is directly equivalent to a data compression technique called prequential coding". §2.4 is titled "ICL
  Minimizes Prequential Code Length". Their claim is about the in-context learner's loss summed over a growing
  sequence of examples. A fixed-k score is therefore an ICL test loss; only the sum over a growing context is an ICL prequential
  code.
- Bits That Count (full text) uses multi-shot prompting as its independent check of elicitation. App. D: "We regard a model
  as capable of being elicited on a particular task if multi-shot prompting or removing biases in the model's logit
  distribution (for multiple choice tasks) results in significantly improved performance relative to the model's
  original zero-shot baseline". §5.1: few-shot gains appeared "only for models and tasks that EDL classifies as
  elicitation-dominated (Appendix I.9)".
- URIAL (Lin et al. 2023) is the base-versus-aligned version of the same idea. From the earlier review: "77.7% of the tokens are at
  such unshifted positions" between base and aligned models. That a few stylistic in-context examples close much of the gap is from memory, unverified.

**What we need.** Teacher-forced forward passes with up to about 57k tokens of context. Olmo-3.1-32B-Think's `config.json` has
`max_position_embeddings` 65,536 (YaRN factor 8 from 8,192). It also has `sliding_window` 4,096 on 48 of the 64 layers, in a
pattern of three sliding-attention layers to one full-attention layer. Demonstrations more than 4k tokens back are therefore visible only through the 16
full-attention layers. A sampling arm is optional. The design is under Recommendation R2.

**Curve and reading.** We plot ICL gap closure G_icl(t, k) = (L0(t) − L_icl(t, k)) / (L0(t) − L0(s2300)) and the raw nats
removed, over RL steps, beside LoRA's G_lora at the same number of examples.
- **Elicitable with no weight change.** At `dpo`, G_icl(4) ≥ 0.25 means a large part of the distance is imitable.
- **Needs weights.** G_icl near 0 while LoRA closes much of the gap means weights must change. This is still compatible with elicitation in EDL's sense.

**Known failure modes.**
- **Copying.** Demonstrations can lower NLL by copying n-grams, which is not capability. The control is demonstrations from an unrelated domain,
  plus an n-gram overlap split.
- **Off-distribution history.** Thinking text in the chat history may be off-distribution for the template.
- **Sliding-window attention.** It may weaken the use of distant demonstrations.
- **Unstable denominator.** The denominator → 0 near `s2300`, so the raw nats should be reported too.

**Closest existing work.** None on Olmo. Dan's brevity hint and instruction ladder are instructions, not demonstrations.
His transplant pastes the final model's thinking for the same prompt, which is the k = "the answer itself" end of this
scale.

### 3. Token-level footprint: where the distance sits, and whether the final model's tokens are already among a checkpoint's top candidates

**What it measures.** At each position of the final model's responses, it records three things: the divergence between checkpoint t and `s2300`; the rank
of `s2300`'s token under t; and whether the divergent positions are the high-entropy "forks". It says whether L0(t) is
spread over every token or concentrated in a few reranked choices.

**Literature.**
- "Sparse but Critical" (Qwen Pilot Team, Alibaba; arXiv 2603.22446, ICLR 2026; full text in part). They use per-position JS
  divergence on top-p-truncated distributions (§2.1). §2.2: "more than 83% of token positions exhibit near-zero
  divergence" under DAPO, and the proportion "exceeds 98% under SimpleRL" (Qwen2.5-32B, AIME24). Cross-sampling (§3, Table 1)
  swaps in RL tokens only at high-divergence positions: 3.86 % of tokens (SimpleRL) and 7.8 % (DAPO) move base accuracy from
  8.23 % to about the RL level. The reverse swap, 5 % and 10.1 % of tokens, drops the RL models to base level.
- Akgül et al. 2026, "Rethinking RL for LLM Reasoning: It's Sparse Policy Selection, Not Capability Learning" (arXiv
  2605.06241, full text in part). Greedy MATH-500 runs with identical prefixes, comparing argmax tokens (§3.1).
  - **Reranked positions:** 1.03–3.96 % of positions are "reranked", meaning the RL token is in the base top-5.
  - **Rank of the RL token:** "The teacher's preferred token is, on average, the second most likely token under the base model", with mean rank
    2.14–2.39.
  - **Entropy:** reranked positions have 7.6–12.6× higher base entropy (Table 2).
  - **Inconsistencies:** the paper contradicts itself in places. §3.1 says "zero shifted positions" but Table 2 shows 0.01–0.12 %, and §6.2's "on par
    with or better than the RL baseline" is contradicted by Table 3.
- Wang et al. 2025, "Beyond the 80/20 Rule" (arXiv 2506.01939, abstract only): RLVR "largely adheres to the base model's
  entropy patterns". Restricting the gradient to the top 20 % highest-entropy tokens keeps performance.

**What we need.** Per-position statistics from the stage-1 L0 pass: NLL, rank, entropy and top-20 log-probabilities, which the
coordinator adopted on 2026-11-25. Divergences between any two checkpoints are then computed offline. The reverse
direction needs one extra forward pass per checkpoint: score checkpoint t's own stored samples (Dan's draws, re-tokenised) under `s2300`
and under t.

**Curve and reading.** Over RL steps we plot:
- the share of positions with JS(t ‖ s2300) above ε, for ε = 0.01 and 0.1 nats;
- the share of `s2300` tokens at rank 1, ≤ 5 and > 20 under t;
- the share of the L0 gap carried by the 5 % most divergent positions.

Each is reported for the thinking and answer segments separately. The readings:
- **Reranking.** At least 99 % of `s2300`'s tokens are in `dpo`'s top 5, and the top 5 % of positions carry at least half the gap. This is the token-level form of "selection", as in Akgül et al.
- **New continuations.** The share outside the top 20 is non-trivial and does not shrink early.

**Known failure modes.**
- **Teacher forcing is off-policy.** Scoring on `s2300`'s samples is off-policy for t, which is why the reverse direction is needed.
- **Truncated distributions.** Top-20 truncation biases JS, as Sparse but Critical's truncated distributions do too.
- **Token-level support is not sequence-level reachability.** Reranking a few forks can still compound into whole responses that are
  vanishingly rare under `dpo`. This is the Yue/ProRL debate (earlier reviews): "in support per token" is not "reachable".

**Closest existing work.** None on Olmo. Our nd forced-first-action propensity-versus-capability factorisation is the
one-position version.

### 4. Rank-1 and extrapolated weight deltas: is RL's update one direction per matrix, set early?

**What it measures.** We rebuild checkpoint t from the `dpo` weights plus only the top singular component of each weight
difference, and check how much of t's behaviour survives. We also extrapolate the early update (s0100, s0500) to step 2300 and test
whether that reaches `s2300`'s pass rate.

**Literature.**
- Cai et al., "On Predictability of Reinforcement Learning Dynamics for Large Language Models" (arXiv 2510.00553v3, ICLR 2026; full
  text in part, first 100k characters).
  - **Rank-1 sufficiency** (§1): "the Rank-1 Subspace alone recovers an average of 99.17% of the reasoning capability".
  - **Construction** (§2.1, Eqs. 1–3): the rank-1 part is σ1·u1·v1ᵀ, rescaled to the norm of ΔW.
  - **Linear dynamics** (§3.1, §4.1): the rank-1 trajectory evolves almost linearly ("average R2 of 0.914" in one place, "exceeding 0.96" in another).
  - **AlphaRL** extrapolates from early training, "only 10% of the total training steps" (§4.2).
  - **SFT and distillation** "yielded neither property" (§1). But App. C says "Both On-Policy Distillation and DPO exhibit clear low-rank
    structures", which undercuts the "RL only" claim.
  - **No error bars** are visible in the parts read.
- ExPO, Zheng et al. 2024 (arXiv 2404.16792, abstract and search summary only). θ2 = θ1 + α(θ1 − θ0) improves DPO and RLHF models
  without training.
- Earlier review: RL updates are sparse or low-rank in some precisions (Mukherjee et al. 2025). The sparsity is a bf16 precision effect (Shenfeld et al. 2025; Zhu et al. 2025, "The Path Not Taken").

**What we need.**
- **Weights only:** the top singular triplet of each block matrix (64 layers × 7 = 448 matrices), by power iteration in fp32.
- **Variants:** θ_dpo + rank-1(Δ_t) for t ∈ {s0100, s0500, s2300}; an AlphaRL-style extrapolation from s0100 and s0500 to step 2300; and an ExPO-style full-delta extrapolation θ_dpo + (2300/500)·Δ_s0500.
- **Evaluation:** each variant is sampled on arm A (218 × 2) and the reach prompts (48 × 8) with the strict checker. That is about 820 responses, about 40 min plus a load.
- **Total:** 5 variants ≈ 4–5 H200-h.
- **Where the SVDs can run:** on a pod. On serrano they could run tensor by tensor on CPU (62 GB RAM; ≤ 8 cores).

**Curve and reading.** We plot the recovery R(t) = (pass(rank-1 variant) − pass(dpo)) / (pass(t) − pass(dpo)) over RL steps,
and the extrapolated model's pass rate beside `s2300`'s.
- **Few weight-space bits.** R ≥ 0.9 means the RL update is rank-1-dominated, which is consistent with selection but does not prove latent
  capability in `dpo`: a rank-1 update can still add computation.
- **Direction set early.** An extrapolation from s0500 that reaches ≥ 0.9 of `s2300`'s gain means later RL mostly scales a direction already set.

**Known failure modes.**
- **The paper's own inconsistencies.** These are listed in the literature bullets above.
- **Ambiguous construction choices.** Cai et al. do not say clearly whether rank-1 is applied to all matrices at once, or which norm they use. We would use Frobenius and say so.
- **bf16 noise in Δ** (earlier review).
- **Checkpoint lineage.** `olmo-weights`' review found that step_0150 and step_0200 lie on an abandoned branch, and that labels from 250 on
  may overstate the steps by about 100. Extrapolation should use s0100 and s0500 on the released lineage, with checkpoint
  index as a second x-axis.

**Closest existing work.**
- Dan's `olmo-weights` measured how far the weights move (norms per stage), not the rank or behaviour of the update.
- Our nd weight interpolation is the α ∈ [0, 1] version on small provers.

### 5. Prompt-optimisation elicitation gap

**What it measures.** How much of the RL gain an automatically optimised system prompt recovers at a fixed checkpoint.

**Literature.** GEPA, Agrawal et al. (arXiv 2507.19457v2, ICLR 2026; full text in part).
- **Setup:** IFBench with a train/validation/test split, prompts optimised and weights frozen.
- **Qwen3-8B** (Table 1): baseline 36.90, GRPO (24,000 rollouts, LoRA) 35.88, MIPROv2 36.22, GEPA 38.61. GEPA "found optimal prompts after just 678 rollouts" (Table 1 caption).
- **GPT-4.1-mini** (Table 2): GEPA plus Merge goes from 47.79 to 55.95.

So prompt search moved IFBench by about 2–8 points in that paper, and GRPO there did not help.

**What we need.**
- **Search:** about 700 search rollouts per checkpoint, with Olmo itself as the reflection model, so that no prompts go to outside APIs.
- **Data:** a train split disjoint from the 218 arm-A prompts.
- **Evaluation:** 218 × 2 draws.
- **Cost:** about 2 H200-h per checkpoint; dpo and s0500 first.

**Curve and reading.** The optimised-prompt pass@1 minus the default, as a share of the dpo → s2300 gap. Prompt elicitation substitutes for RL if that share is large at `dpo` and shrinks along the run.

**Failure modes.** The search can overfit the validation split. Reflection with a weak model limits it. GEPA's own
IFBench gains are small, so a small result is the expected one and says little.

**Overlap.** Dan's instruction ladder and 60-word hint are hand-written versions of this.

### 6. Decoding-only elicitation: temperature sweep, then power sampling

**What it measures.** How much of the RL gain is recovered at `dpo` by sharpening its own distribution, with no training at all.

**Literature.**
- Karan & Du 2025, "Reasoning with Sampling" (arXiv 2510.14901, abstract only). An MCMC sampler on the base model's own likelihoods
  gives single-shot gains "that nearly match and even outperform those from RL" on MATH500, HumanEval and GPQA.
- Ji et al. 2026, "Scalable Power Sampling" (arXiv 2601.21590, abstract only). They approximate the power distribution by a token-level low-temperature
  distribution scaled by a future-quality factor. This is ">10×" lower latency than MCMC, and it "matches or surpasses one-shot GRPO".
- Their framing is that RL gains are "distribution sharpening rather than the acquisition of new capabilities" (2601.21590 abstract).

**What we need.**
- **Temperature sweep first:** T ∈ {0.6, 0.8} beside Dan's T 1.0, on arm A, at dpo, s0100 and s0500. That is about 0.5 H200-h each. It also checks a confound: Dan's curve used T 1.0, and the model card's recommended decoding should be checked.
- **Power sampling only if the sweep moves pass@1.** It needs a custom sampler and costs about 5–10× ordinary sampling at 5k-token responses.

**Reading.** If lowering T at `dpo` recovers a large share of the gain, RL on IF acts at least partly as sharpening.

**Failure modes.** Low temperature and the power distribution are not the same thing: Karan & Du's point is the global p^α. Long thinking may also degenerate at low T.

### 7. Compliance probe and steering along it

**What it measures.** Two things: whether a checkpoint's activations before it answers already encode whether it will comply, and
whether pushing along that direction raises compliance.

**Literature.** Heo et al. 2025, "Do LLMs 'know' internally when they follow instructions?" (arXiv 2410.14516v5, ICLR 2025;
full text in part). The probe sits on the first-token representation, with IFEval checkers as labels, on 7–13B chat models.
- **Within instruction types:** AUROC 0.74–0.88 on unseen tasks.
- **Across instruction types:** 0.50–0.55, "close to chance".
- **Steering** along the direction raises success by 2–6 points, against 2–8 points lost for random directions (Table 4).

Steering as elicitation: Venhoff et al. and Ward et al. (earlier review).

**What we need.**
- **Data:** prompt-token activations for the 218 arm-A prompts, a few minutes per checkpoint. Labels are per-prompt pass rates from Dan's stored draws at that checkpoint.
- **Probe:** cross-validated over held-out instruction types.
- **Steering arm:** the mean activation difference between `s2300` and t on the prompts, added to t, at 3 strengths. Sampled on the 48 reach prompts × 8 draws, this is about 1.5 H200-h per checkpoint.

**Reading.** A direction present at `dpo` that predicts and, when added, raises compliance means latent knowledge of compliance.

**Failure modes.**
- Heo's own probe does not generalise across instruction types, and their steering effects are small.
- Probes need control tasks (Hewitt & Liang, earlier review).

### 8. Cross-sampling: the number of the final model's tokens a checkpoint needs

Sparse but Critical's procedure (§3) generates from t and takes `s2300`'s token wherever JS(t, s2300) > ε. The measurement is the number of interventions
needed for t to reach `s2300`'s strict pass rate on the reach prompts. The reverse swap is the control.

It interpolates between our forced first action (one position) and Dan's transplant (all thinking tokens).

**Costs and constraints.**
- **Memory:** both models must be resident, 2 × 64.5 GB. That means two H200s or an fp8 copy of one.
- **Engineering:** vLLM does not support this, so it needs custom decoding.
- **Cost:** about 3 H200-h at `dpo` alone.

It is worth doing only if item 3 shows a sparse footprint.

### 9. Tiny-budget RL (steering-vector RL, TinyLoRA RL)

This is the RL-native version of item 1: run GRPO from checkpoint t, training only a per-layer bias or about 10–1,000
parameters, and see how much of `s2300`'s gain it reaches. Sinii et al. and Morris et al. (above) show this works on 7–8B math models.

At 32B with 5k-token rollouts, one GRPO step of 128 prompts × 8 rollouts is about 5M generated tokens. That is about 50 min, which makes the run
prohibitive. A version on the 7B family is possible if Leon wants the RL-side measurement.

### Dropped, with reasons

- **Noise injection** (Tice et al., arXiv 2412.01784v3, abstract only). Standard models "show predictable performance degradation
  with increasing noise" and sandbagging models "exhibit anomalous performance improvements". The abstract also reports eliciting
  Mistral Large 120B's full performance where it underperformed without being instructed to.
  - **Why it does not fit:** the signal needs a suppression mechanism that noise disrupts. We have no reason to expect one in `dpo`'s IF behaviour.
  - **Cost:** each (noise level, seed) needs about 1 H200-h of sampling.
  - **Revisit** only if items 1–3 point to a "can but won't" (propensity) gap.
- **Crosscoder diffing between dpo and s2300.**
  - **Cost:** training a crosscoder on 32B activations is expensive.
  - **Artefacts:** "model-only" features are often artefacts without Latent Scaling (Minder et al. 2025, earlier review).
  - **Cheaper version:** the activation-difference lens (Minder et al. 2025b, earlier review) is folded into item 7's steering vector.
- **Relearning after unlearning.** Nothing was unlearned. As a measure of how fast a skill is learned, it is what EDL and item 1 already do.
- **On-policy distillation speed.** It measures what EDL measures, at the price of sampling.
- **KL / function distance between checkpoints.** This is Dan's proposal 48 and not ours to duplicate.
- **High-k pass@k and reach.** Dan's `olmo-ifcurve`.

## Recommendation: the three to run first on dpo, s0100, s0500, s1100, s2300

The coordinator has already folded item 3's storage into stage 1. Items 1 and 2 are candidates for a second phase on the same pods.

### R1. Parameter-budget ladder (item 1)

- **Checkpoints.** `dpo`, `s0100`, `s0500` and `s1100`. Skip `s2300`, where EDL ≈ 0 by design (the null control) makes EDL/P
  undefined. Add `s2300` back only if its EDL_ref is clearly positive.
- **Data.** The stage-1 sets unchanged: train n = 64 and test m = 32, response-token cap 6,144, data order 0. Loss on response tokens,
  with the thinking and answer segments reported separately, as in stage 1.
- **Arms** (trainable parameter count P). Olmo-3.1-32B has 64 layers, hidden size 5,120, intermediate size 27,648, and 40 query / 8 key-value
  heads of size 128 (`config.json`).
  - `R16`: LoRA r16 on q, k, v, o, gate, up and down. P = 134.2M. This is stage 1 and is the reference EDL_ref. Bits That Count used r512, which
    matched full fine-tuning in their runs, but r16 is safe here: EDL_ref/P will be far below any threshold.
  - `R1`: LoRA r1 on the same modules, P = 8.39M.
  - `B`: one trainable bias vector added to the residual stream at each layer's output, P = 64 × 5,120 = 327,680. This is
    Sinii et al.'s parameterisation, trained here by SFT.
  - `S_hi`, `S_lo`: uniformly random sparse subsets of the `R1` parameters, as Bits That Count does ("uniformly randomly
    sampled sparse subsets of parameters within rank 1", §3.3). Their sizes are set after stage 1 reports EDL_ref(`dpo`) in bits:
    P = EDL_ref(`dpo`) / 0.3 and EDL_ref(`dpo`) / 3. The same P is used at every checkpoint. With the arms above, the ladder then spans EDL/P
    from about 10^-4 to 3 bits/param.
- **Why the ladder must go below rank 1.** If EDL_ref is about 10^4 bits (my guess for 64 responses of about 5k tokens; unmeasured),
  EDL/P is about 10^-4 at r16 and about 10^-3 at r1. Neither can be capacity-bound under the paper's thresholds.
- **Training.**
  - The stage-1 protocol: score then step, batch 1, AdamW (0.9, 0.999), weight decay 0, clip 1.0, constant learning rate, bf16 base weights, fp32 adapters.
  - After the online pass, 3 more epochs give `L_test_conv`, as adopted for stage 1.
  - Learning-rate grids, piloted at `dpo` and then frozen per arm: `R1` {1e-4, 3e-4, 1e-3}; `B`, `S_hi` and `S_lo` {1e-3, 1e-2}. Bias and sparse adapters need larger rates; widen if the best rate is at an edge.
- **Readouts per (checkpoint, arm).**
  - MDL = online nats; EDL = MDL − 64 · L_test_conv, in bits.
  - The capacity ratio C = EDL(P) / EDL(R16).
  - PGR_loss = (L0 − L_test_conv(P)) / (L0 − L_test_conv(R16)).
  - EDL/P.
  - Intervals: bootstrap over test prompts.
- **Reading rule.**
  - P* = the smallest P with C ≥ 0.95 (the paper's definition, Tables 6–7), log-interpolated between arms.
  - **Primary, relative:** "the start needs more information per parameter" if P*(`dpo`) ≥ 10 × P*(`s1100`). A factor of 10 is the size of the
    paper's elicit/teach separation (§6.2: "two-order-of-magnitude difference").
  - **Secondary, absolute, flagged as transferred:** EDL/P* ≤ 0.1 bits/param reads "elicitation-like by Bits That Count's calibration",
    ≥ 1 reads "teaching-like", and values in between are not classified.
- **Calibration caveat, to print with every absolute reading.** The thresholds come from:
  - Llama 3.2 1B/3B and 3.1 8B on DeepMind arithmetic (0.004–0.03; one PGR-based 0.14) and Alpaca (0.02–0.03);
  - TinyStories-1B learning arithmetic (0.70–2.23);
  - Qwen2.5 1.5B/14B/32B distilled on R1 traces (0.04 / 0.05 / 0.02).

  They used LoRA ranks 1–512 plus random sparse subsets, never per-layer biases. Data ran up to millions of examples for arithmetic and about 10–1,000
  for reasoning. Training ran to validation-loss convergence, with 3 seeds (Tables 1, 6, 7; Fig. 5). All of it was SFT. §6.1 also finds
  "larger models have lower capacity thresholds", so thresholds for 32B may be lower still.
- **Optional behavioural check, at `dpo`.** Sample the 48 reach prompts × 8 draws through the `B` adapter. Can a 330k-parameter bias trained on
  64 responses reproduce part of Dan's r16 result (40–41 of 48 prompts from 16 trajectories)? About 0.4 H200-h.
- **Cost.** Each (arm, learning rate) is about 1.3M training tokens (one pass plus 3 epochs over about 320k tokens), plus test scoring: about 15–20 min. The pilot at `dpo` is 4 arms × 2–3 rates ≈ 10 runs ≈ 3 h. The other three checkpoints are 4 runs each ≈ 1.2 h each. That is about 6.5 H200-h in total ≈ $25 community, excluding loads.

### R2. In-context description length (item 2)

- **Checkpoints.** All five.
- **Demonstrations.** Drawn from the stage-1 train pool: `s2300`'s strict-pass responses to other prompts, preferring hand-labelled genuine ones, under the 6,144-token cap.
  - k ∈ {0, 1, 2, 4, 8}.
  - Three demo sets per k, drawn once and nested (the k = 2 set contains the k = 1 demo), identical at every checkpoint.
  - Format: earlier chat turns (user prompt, then the assistant's full response with its thinking). Check how the Olmo chat template renders
    thinking in history; if it strips it, insert the thinking text explicitly and record the choice. Keep it identical across
    checkpoints.
- **Targets.** The 32 stage-1 test responses and the 42 genuine reach responses: the target response tokens, teacher-forced after the demos and the
  target prompt. The k = 0 values are stage 1's L0.
- **Context.** k = 8 × ≤ 6,144 plus the target's prompt and up to 6,144 response tokens ≈ ≤ 57k, under 65,536. Responses average about 4k, so a typical k = 8 context is about 40k.
- **Controls.**
  - (a) An unrelated-domain demo set: `s2300` responses to non-IF prompts, for example the AIME solutions from Dan's `olmo-think32b`, if stored. This
    separates "general thinking style" from IF-specific information.
  - (b) An n-gram overlap split: the NLL drop on target tokens inside 8-grams that also occur in the demos, against all other tokens.
  - (c) At `s2300`, G_icl is undefined; report the raw nats.
- **In-context prequential code.** Put the first 8 train examples, in the stage-1 order, in one sequence. The code is Σ_{i ≤ 8} NLL(y_i | x_i,
  earlier examples). Compare it with the stage-1 LoRA online code over the same 8 examples. This is Elmoznino's quantity next to EDL's.
- **Readouts.**
  - L_icl(t, k) in nats per token, by segment.
  - G_icl(t, k) = (L0(t) − L_icl(t, k)) / (L0(t) − L0(s2300)), plus the raw nats.
  - The 8-example ICL code against the LoRA code.
  - Intervals: bootstrap over targets × demo sets.
- **Reading rule.**
  - "Elicitable in context at t" if G_icl(t, 4) ≥ 0.25 with the lower 95 % bound above 0, and the control (a) gap stays below half of it.
  - "Context does as much as weights" if G_icl(t, 8) ≥ G_lora(t, n′ = 8). This needs L_test at n′ = 8, which stage 1 records.
  - Curve: G_icl(4) and the raw nats over RL steps.
- **Optional behavioural arm**, at `dpo`, `s0100` and `s0500`. The 48 reach prompts with one fixed k = 2 demo set, 8 draws, strict checker. Gains are
  read against Dan's plain rates at the same checkpoint.
- **Cost.**
  - **Scoring:** 74 targets × 3 sets × 4 non-zero k = 888 scored targets per checkpoint. With prefix caching per demo set (vLLM
    `prompt_logprobs` with automatic prefix caching, or HF with cached `past_key_values`), that is about 5M prefill tokens, about 0.3–0.5 H200-h per checkpoint.
  - **Optional sampling:** 384 responses with a 10k prefix, about 0.5 h per checkpoint.
  - **Total:** about 2.5 H200-h for scoring at five checkpoints (≈ $9), about 4 h with sampling (≈ $15).

### R3. Token-level footprint (item 3)

Storage was adopted into stage 1. The remaining additions:
- **Reverse direction.** One forward pass per checkpoint over t's own stored strict passes and fails from Dan's draws, re-tokenised. Score them under t and under
  `s2300`, the latter on the `s2300` pod. That is about 1M tokens per checkpoint, about 5 min plus no extra load if run while the checkpoint is
  resident.
- **Fixed thresholds:** JS > 0.01 and > 0.1 nats; ranks 1 / ≤ 5 / > 20; the top 5 % of positions.
- **Reading rule:** "reranking" if at least 99 % of `s2300` tokens are in t's top 5 and the top 5 % of positions carry at least 50 % of
  L0(t) − L0(s2300).
- **Cost:** about 0.5 H200-h in total.

### Run next if budget remains

Item 4 (rank-1 and extrapolated deltas, about 4–5 H200-h). It is the only remaining metric that uses the checkpoint sequence itself and needs no
training.

## What the literature says about the weaknesses of our EDL design

Source: Bits That Count (full text) unless marked otherwise.

1. **The residual term is not the paper's.** The paper's EDL subtracts the test loss of a model trained to convergence. App. A, Fig. 6: "Multi-epoch
   training on the same data improves generalization … yielding larger EDL". Table 1: "Stopping criterion: Validation loss
   convergence". The MDL term uses the first epoch only (§2.5). Our L_test(θ_64) after one pass is therefore higher than their L*, and our EDL
   is smaller. The coordinator has adopted a 3-epoch `L_test_conv`. Check that it has converged, with a 4th epoch at `dpo`.
2. **One point on the scaling curve cannot show the signature.** The elicit/teach signature is the shape of EDL per token as n grows over orders of
   magnitude.
   - For reasoning, the paper plots about 10–1,000 examples (Fig. 5).
   - Teaching peaks appeared at about 1K (TinyStories language), about 150–300K (arithmetic) and over 4M examples (Table 5).
   - At n ≤ 64, "falling throughout" cannot rule out a teaching phase that peaks later. The PREREG's shape reading is descriptive only, and should stay so.
3. **Single pass, batch 1, no replay, lies in the most hyperparameter-sensitive regime.**
   - App. H.1.2: hyperparameter choices "primarily affect the low-data, low-parameter regime of single-epoch training".
   - Bornschein et al. 2022 (arXiv 2210.07931, abstract only): "online-learning with rehearsal has favorable performance compared to the previously widely used block-wise estimation". By implication, plain online learning without rehearsal gives longer codes.
   - A weak learner inflates the MDL term and leaves L_test high, which pushes EDL toward 0 at every checkpoint and compresses differences.
   - Our learning-rate-selection convention (shortest online code plus ln(number of rates)) controls one dimension of this. Rehearsal, or a block-wise
     check at `dpo` (train on blocks 1..j and score block j+1, retraining from scratch), would bound the rest.
   - Bits That Count used 3 seeds per configuration; we use 2 orders at 3 checkpoints.
4. **Thinking dominates the loss.** The paper computes loss "only on designated label tokens, typically the answer portion … This
   convention helps to isolate capability-relevant learning from format acquisition" (§2.1). The theory paper's Prop. 5.5 is
   about "format learning creates early transients" (earlier review note). Thinking is 96–98 % of our tokens, so whole-response EDL is
   mostly the final model's thinking style. The coordinator has adopted equal prominence for the answer segment.
5. **Targets from the final model.**
   - **Stronger-model targets are in the method's scope.** Bits That Count's reasoning runs fit Qwen2.5 bases to R1-generated, correctness-filtered traces (App. D).
   - **But L0(s2300) is low by construction.** These are its own samples, so part of every checkpoint's distance is `s2300`-specific style, not capability, as PREREG says.
   - **Two controls, both suggestions:**
     - (a) A shuffled-pairing control: the same 64 responses attached to the wrong prompts. Its EDL measures the style information that transfers without the
       prompt. EDL(matched) − EDL(shuffled) is the prompt-conditional part.
     - (b) A selection-only control at `dpo`: EDL on `dpo`'s own strict-pass responses, where they exist. By the KL-RL
       analysis (Korbak et al.; Shenfeld et al.'s split of KL into log 1/p and a reshaping term; earlier review), fitting the model's own
       success-conditioned outputs is the pure-selection reference. Comparing it with EDL on `s2300`'s outputs separates "select what dpo already produces" from
       "move toward s2300's way of answering".
6. **Fine-tuning can teach, so the null is not enough.**
   - §7.1: "fine-tuning-based evaluations may inadvertently teach capabilities rather than elicit them".
   - The UK AISI / FAR.AI / Anthropic auditing game (arXiv 2512.07810; read only through a search summary of the AISI blog, unverified) reports that training-based elicitation "sometimes also improved benign models".
   - Our null control (`s2300` on its own outputs) checks the zero point, not the teaching side. The capacity ladder (R1) and the in-context arm (R2) are the
     paper's own two cross-checks: capacity thresholds, and multi-shot prompting as independent validation.
7. **LoRA rather than full fine-tuning.** At n = 64, r16 is far from capacity-bound (see R1), so this is not a concern for the loss terms here.
   It matters only for the capacity reading, where r16 stands in for the r512 or full-fine-tuning reference.

## Open questions and what I could not verify

- **Bits That Count.** Read in full except some appendix tables, which were skimmed. The App. I.9 few-shot validation numbers were not checked.
- **Abstract only:**
  - Donoway et al. 2025, NeurIPS. Its abstract text is in `references.bib`; the PDF is in Dmitry's Zotero, not accessed.
  - Karan & Du; Ji et al.; Wang et al. (forking tokens); Tice et al.; Sinii et al. 2505.18706; Bornschein et al.; Elmoznino et al. (beyond the §2.4 heading).
- **Unverified (search summary only):** the auditing-games paper (2512.07810) and ExPO's numbers.
- **Read only in part** (the fetch tool stopped at 100k characters): Cai et al. and Sparse but Critical. Conclusions and later appendices were not seen.
- **Not found:** no paper measures EDL, capacity or in-context description length along an RL run's checkpoints, and none reports these metrics for instruction
  following specifically. If that holds up, it is a gap our curves would fill.
- **None of our cost figures are measured.** The throughput assumptions above are guesses.

## Files

- `screened.md`: every paper looked at.
- `notes/`: five notes:
  - `donoway2026-bits-that-count.md`
  - `tiny-adapter-elicitation.md` (TinyLoRA, Sinii et al., Donoway 2025)
  - `token-level-rl-footprint.md` (Sparse but Critical, Akgül et al., Wang et al.)
  - `cai2025-rank1-rl-dynamics.md`
  - `heo2025-if-dimension-and-gepa.md`
- `to_add_to_zotero.md`.
