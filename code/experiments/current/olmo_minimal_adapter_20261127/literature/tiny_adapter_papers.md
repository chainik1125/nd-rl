---
written_on: 2026-11-27
written_by: "agent:claude (sonnet, tiny-adapter reader)"
papers:
  - donowayQuantifyingElicitationLatent
  - morris2026tinylora  # proposed key; arXiv 2602.04118
  - sinii2025smallvectors  # proposed key; arXiv 2509.06608
  - sinii2025biasonly  # proposed key; arXiv 2505.18706
---

# Reading note: four papers on few-parameter elicitation, for the Olmo 3.1 Think 32B minimal-adapter design

Scope: a full-text reading of the four papers, written for an independent auditor. Section and figure
references are to the PDFs I extracted with PyMuPDF. Quotes are verbatim from that extraction. I mark my own
inferences as "Inference". Figures (curves) are not machine-readable from the text extraction, so any number
that appears only in a plot is not reported here.

Texts read (all under `/scratch/cowifornia/nd-rl/olmo_elicit/v2/papers/`):

| Key | Source | Pages | Extracted text |
|---|---|---|---|
| Donoway et al., NeurIPS 2025 | NeurIPS proceedings PDF (`proceedings.neurips.cc/paper_files/paper/2025/file/dd16e008534dc52fab72fdedad069d5c-Paper-Conference.pdf`); no arXiv version found | 32 | `donoway/donoway.txt` |
| Morris et al., arXiv 2602.04118v1 (4 Feb 2026) | arXiv PDF | 14 | `2602.04118/2602.04118.txt` |
| Sinii et al., arXiv 2509.06608v4 (2 Feb 2026) | arXiv PDF | 23 | `2509.06608/2509.06608.txt` |
| Sinii et al., arXiv 2505.18706v3 (1 Oct 2025) | arXiv PDF | 10 | `2505.18706/2505.18706.txt` |

The earlier note (`docs/literature/2026-11-25-elicitation-metrics/notes/tiny-adapter-elicitation.md`) was read
and not relied on. It matched the text on the headline numbers I spot-checked, but it had read the bias-only and
Donoway papers by abstract only, and several things below (what exactly is trained, learning rates, the
training targets, the Donoway reasoning setup) are not in it.

---

## 1. Donoway et al., "Quantifying Elicitation of Latent Capabilities in Language Models"

### Parametrisation

- Trained parameters are scalar entries of LoRA adapters, chosen uniformly at random from the pooled set of
  all LoRA weights. Quote (§3.1): "We select these trainable parameters using uniform random sampling across all
  LoRA modules in all layers, with each parameter having equal probability of selection regardless of its
  position in the network." Implemented "by applying a random binary mask over gradients during the backward
  pass". The mask is sampled once per run and held fixed (App. B.1).
- Which entries are in the pool: both factors. App. C.1: "Matrix A: R^{r×d_in} (Kaiming uniform
  initialization); Matrix B: R^{d_out×r} (zero initialization)". "learnable subsets of the LoRA adapter matrices
  are trained while the remaining adapter weights are held frozen at their initialized values" (§3.1). So
  unselected A entries stay at their Kaiming-uniform values and unselected B entries stay at 0.
- Parameter count: "When we refer to 'trainable parameters,' we exclusively mean scalar values within LoRA
  adapter matrices (except in full fine-tuning...)" (App. C.1). Frozen-at-init entries do not count. The cost
  of specifying which entries (the mask/seed) is not charged to the count in the headline numbers; it is
  charged as a "header" only in the MDL analysis (App. I, Table 3).
- Modules: the pool is "every projection matrix in every transformer block" for the Alpaca runs, ranks
  {1, 2, 4, 8, 16, 256, 512} (App. B.1). App. C.1 also tries Q,V; Q,V,K,O; G,U,D; K,Q,V,U,D and random module
  subsets, and reports "similar logistic scaling" for all. "attention-only tuning (adapting K, Q, V, O matrices)
  is consistently the most parameter-efficient across all datasets". Fig. 8 (Qwen R1-distill 1.5B, AMC23 loss
  improvement) shows this is "particularly pronounced when fine-tuning sparse random-parameter subsets of the
  rank-1 LoRA adapters".
- The LoRA rank used for the main logistic frontiers is not stated per experiment. Budgets span "a single
  parameter to the full set of LoRA adapter weights for ranks 1-1024" (§4.1). LoRA alpha/scaling is not stated.
- LM head: "In the very-few parameter regime (<10k trainable parameters) the original language-model head
  remains frozen...; at larger budgets the head is fine-tuned together with the adapters" (App. B.1, Alpaca
  only). Whether head parameters are counted at the larger budgets is not stated.
- For fewer than ~20-25 parameters they do not use gradient descent as the headline: "we also perform Bayesian
  optimization with a Gaussian process surrogate ... directly on the learnable adapter weights" (App. C.4),
  with weight bounds such as [-10, 10] for k <= 2, [-5, 5]/[-3, 3] for 3 <= k <= 10, [-3, 3]/[-1, 1] for 11 <= k <= 25,
  and 400 extra GP candidates scored on the full train+validation sets. "performance from models trained with
  Bayesian optimization matches or exceeds the performance of the best performing hyperparameter configurations".

### Training

- SFT only (no RL). The per-task targets:
  - GSM-8K-CoT-Choice, ARC, BoolQ: ground-truth class labels (a one-token answer, scored by comparing answer-token
    logits; App. C.3). GSM-8K-CoT-Choice is a binary "is this solution correct" task with Claude 3.5 Sonnet
    solution attempts, 4,000 examples = 1,000 questions x 4 (App. D.1); the fine-tuning set samples one correct
    and one incorrect attempt per question.
  - Alpaca: the Alpaca reference responses; loss on response tokens only ("all prompt tokens are replaced with
    the ignore-index", App. B.1); two epochs over the training split; sequences truncated to 256 tokens.
  - Reasoning (the case relevant to us, §4.2, Fig. 3, App. B.2, C.1, D.2): "Fine-tuning Qwen2.5-1.5B on 1,000
    unique reasoning traces generated by DeepSeek-R1-Distilled-Qwen-1.5B". The traces are the post-trained
    sibling's samples to the s1K prompts, up to 32,768 tokens, temperature 0.6; "we verify that the reasoning
    traces have the correct format ... but do not regenerate incorrect solutions" (D.2). So targets are a
    stronger/post-trained model's outputs, not verifier-filtered for correctness. Loss mask for this run is not
    stated (the "loss only on response" statement is in the Alpaca paragraph).
- "In all experiments, we train until convergence, regardless of the number of epochs required" (§3.1). Full
  fine-tune convergence: no validation improvement for 20 consecutive epochs. For multiple choice: at least 20
  epochs without improvement; generative: at least 2 epochs without validation-loss improvement (C.2). Training
  below ~20 parameters "can require more than 1000 training epochs" (C.4). Runtime 10 min to 36 h per run on one
  H100, about 3,500 runs (C.5).
- Hyperparameters: per budget and seed, up to 50 Bayesian-Hyperband trials over learning rate in [1e-6, 1]
  (log-uniform), batch size in {1,...,256}, weight decay in [0, 0.1] (C.2). The chosen learning rates are not
  reported anywhere in the paper. The only statement about them: "hyperparameters (learning rate, in particular)
  can differ significantly depending on the sparsity of trainable parameters" (App. H). Optimiser AdamW.
- Number of examples: fixed per task (1,000 reasoning traces; 4,000 GSM-CoT-Choice items; full Alpaca split,
  with an internal inconsistency: "raw 52k examples ... partitioned 90/10 into training (234,006 instructions)
  and held-out evaluation (26,006)"). Example count is never swept. The Limitations (§6.3) say: "we only train on
  datasets of finite (and sometimes relatively small) size", so performance at k is "a lower bound" on what is
  possible.
- Seeds: 3-5 per budget (App. E), "up to 5 random seeds with up to 50 ... trials each" (C.2).

### Measurement

- Metric: task accuracy on held-out validation/test data (multiple choice), Alpaca-Eval win rate against the
  best full-fine-tuned model judged by a preference model, loss improvement, and for reasoning "pass@1 and
  majority response (maj@16)" from 16 samples at temperature 0.6 on AMC24, AIME24, AIME25 (§3.2).
- Gap closure (App. E): GC(p) = (f(p) - Acc0) / (Acc_full - Acc0), where Acc0 is the zero-shot accuracy of the
  base model and Acc_full is full fine-tuning (§4.1 adds "or maximum of the logistic fit, whichever is higher").
  f(p) = max over seeds and hyperparameter trials of validation accuracy, and the Pareto frontier is the set of
  those maxima. So it is a best-of-N-configurations frontier selected on the validation set, not a mean over
  seeds. Inference: this is optimistic by construction, at every budget, and more so at small budgets where there
  is more run-to-run variance. App. H says near all runs lie within 2 percentage points of the frontier for ARC-Easy
  (Fig. 11 caption), which bounds but does not remove this.
- Curve: generalized logistic in log p, Eq. (4): f(p) ~ Acc_inf - (Acc_inf - Acc0) / (1 + exp(-a - b log p))^nu, fit by
  nonlinear least squares. AIC/BIC prefer it over saturating exponential, power law, broken power laws (App. F, Fig. 9;
  fits on only a handful of frontier points per curve). Error bars: bootstrap (drop one frontier point at random,
  refit; 10^4-10^5 resamples) 95% CIs on the fit, not on pass-rate sampling noise (App. E). Reasoning panels
  show "standard errors over 16 samples" (Fig. 3).

### Headline numbers

- Abstract: "training as few as 10-100 randomly chosen parameters ... can recover up to 50% of the performance
  gap between pretrained-only and full fine-tuned models, and 1,000s to 10,000s of parameters can recover 95%".
  Checked against the full text: these are for the multiple-choice/classification tasks (Table 1, Fig. 1,
  Fig. 10), not for the reasoning generation task.
- Table 1 (GSM-8K-CoT-Choice, Llama 3.2 1B / 3.2 3B / 3.1 8B): 50% gap closure at 39 / 33 / 77 parameters; 90% at
  1,036 / 2,752 / 2,403. Fig. 1's bottom table (read per model as 50/75/90/95%) gives 1B: 39/263/1,036/2,403;
  3B: 33/304/2,752/12,838; 8B: 77/487/3,048/9,589. So the paper disagrees with itself on the 8B 90% value (2,403 in Table 1 vs 3,048 in
  Fig. 1), and the 95% points are about 2.4k to 13k parameters. ARC-Easy 50% at 216 / 774 / 99 and 90% at 4,144 /
  159k / 318k. ARC-Challenge 50% at 3,023 / 236 / 98 and 90% at 154k / 3,077 / 415. BoolQ: about 100 parameters for
  50% (App. G.2, Fig. 10: 74 and 147 at 50%).
- Generation: Alpaca 50% win-rate gap at about 1,000 parameters (§4.2; App. B.1: 1B win rate "jumps from ~0.08 to
  ~0.71 as the budget grows from 10^3 to 10^5").
- Reasoning (the relevant case): "For both exact match accuracy (maj@16) and loss improvement on AMC24 ... ~10,000-20,000
  trainable parameters are sufficient to achieve 50% gap closure (Figure 3 & Figure 8)" for Qwen2.5-1.5B trained
  on the 1,000 R1-distill traces. The pass@1 gap-closure thresholds are not tabulated; Fig. 3 shows pass@1 and
  maj@16 "recover similar logistic scaling". The abstract's "extending to reasoning models and remaining robust to
  increases in inference compute" refers to this experiment. Note the model mix: §3.1 says Qwen2.5-1.5B, App. C.1
  says Qwen-2.5-Math-1.5B; the distilled comparison model is DeepSeek-R1-Distill-Qwen-1.5B.
- Pretrained vs randomly initialised comparison (§4.5, Fig. 2): TinyStories needs more than seven orders of magnitude
  more parameters from a random init for the same performance; for chess puzzles (a task requiring teaching) the gap
  is less than one order of magnitude and 50% needs more than 100,000 parameters.
- Curve shape: logistic in log p, "steep initial improvements that asymptotically plateau" (§1).
- Larger models: for 8B, "fewer than 100 trainable parameters" reaches 50% on all the multiple-choice sets (§4.1),
  while ARC-Challenge on 1B is the counter-example, attributed to "possible absence of the capability".

### Limitations stated (§6.3) and what they say about reasoning/instructions

- Task diversity ("none of them have the diversity and complexity of production settings"); MDL approximations;
  random selection may not be optimal; other methods may do vastly better, so these are only upper bounds on the
  parameters needed; finite dataset size.
- Reasoning: they do run a long-chain-of-thought SFT experiment (above) but only on a 1.5B Qwen model, only on the
  s1K prompt set, and report maj@16/loss rather than a tabulated pass@1 recovery. Instruction following: Alpaca
  (256-token sequences, preference-model win rate). They do not study thinking-trace length effects or a 30B+ model.
- Code is at `github.com/edonoway/quantifying-elicitation-neurips25`; the paper's own checklist (item 5) says code
  was to be released after anonymisation. I did not read the code.

### Does this match our parametrisation? What would it predict?

- Closest match of the four. Same object (entries of LoRA factors, zero-init B, frozen random A, uniform random
  entry mask over all projection matrices), and attention-only was found more efficient. Differences:
  1. Their pool includes A entries as well as B entries; we train B entries only. Inference: at B = 0 the gradient with respect to an
     A entry is exactly zero, so a selected A entry can only move after its module has a nonzero B; with rank 1 and a few
     hundred selected entries most selected A entries are inert early on. Their effective budget at small k is therefore
     probably smaller than the nominal k, and a B-only mask is, if anything, a cleaner parameterisation of the same idea.
  2. Their LoRA rank is not stated for the headline curves; ours is rank 1. At rank 1, a trained B entry b_i adds b_i * a^T to row i
     of the weight, i.e. it adds b_i * (a . x) to output channel i of that module. That is a one-channel, input-dependent
     (through a . x) nudge, not a token-independent bias.
  3. Their targets for reasoning are a post-trained sibling's traces (1,000 of them, trained to convergence over
     many epochs, learning rate tuned per budget); ours are 1-64 responses.
- Prediction for k = 256 and 4,096 (inference, extrapolating from a 1.5B model and one dataset): both fall below the
  10k-20k parameters at which a 1.5B model reached 50% of the AMC24 gap from 1,000 traces. The generic scaling remark
  (larger models need fewer parameters, §4.1; also TinyLoRA Fig. 3) pushes in our favour for a 32B model, but their
  largest model is 8B and none of those curves is a long-CoT generation task. A reasonable prior is "much less than
  half the gap at k = 256, plausibly a substantial part at k = 4,096 if the 1-64 examples were enough to find the direction",
  but the paper contains no evidence on the number-of-examples axis, so that second clause is untested.
- Learning rates they needed: not reported. Only that the optimum shifts strongly with sparsity (App. H), that the
  search range was 1e-6 to 1 (up to 1.0), and that below ~20 parameters gradient descent was unreliable enough that
  they switched to a derivative-free search with weight bounds of order 1-10 (C.4). Plan on a wide learning-rate sweep.

---

## 2. Morris, Mireshghallah, Ibrahim, Mahloujifar, "Learning to Reason in 13 Parameters" (TinyLoRA), arXiv 2602.04118

### Parametrisation

- §4.0.2: TinyLoRA builds on LoRA-XS (W' = W + U Sigma R V^T with U, Sigma, V from the truncated SVD of W; only the r x r matrix R is
  trained) and replaces R by "a low-dimensional trainable vector v in R^u projected through a fixed random tensor P in R^{u x r x r}":
  W' = W + U Sigma (sum_i v_i P_i) V^T. P_i are fixed random matrices. So the trained object is a u-dimensional vector per module
  (or per group of modules), living in the top-r singular subspace of W.
- Tying: "n_tie" modules share one v; total trainable = n m u / n_tie, down to u (even 1) when all modules share. Applied to all
  seven projections per block (q, k, v, o, gate/up/down). Tying strategies: "structured" (same module type) vs "tiled" (nearby
  depth, agnostic of type); tiled was better (§6.5, Fig. 4).
- Frozen rank r = 2 for all main experiments (ablation Fig. 7: 1 to 2 gives modest gains, larger values degrade).
- Parameter count: the number of trainable scalars (v entries after tying), also reported in bytes (bf16 = 2 bytes each, so 13
  parameters = 26 bytes). fp32 storage beats bf16/fp16 even per byte (Fig. 4).
- Not comparable to ours: the update lives in the top singular directions of each weight matrix, the random P_i are not
  a random mask of weight entries, and the same few numbers drive every layer.

### Training

- Both RL (GRPO, exact-match reward, in VERL with vLLM) and SFT are compared (§5.1, §6.2).
- RL setup, GSM8K: no KL penalty; full 7,500-problem training set, three epochs, 4 samples per problem, batch size 64, max generation
  length 4,096. MATH-style runs follow SimpleRL: hardest-difficulty subset of 8,523 problems, KL coefficient 0.001, temperature 1.0,
  batch size 256, 8 generations per prompt, response length 3,072.
- vLLM only supports LoRA rank >= 4 so they inference with merged weights each step and use truncated importance sampling to correct the
  train/inference mismatch (§5.1).
- SFT: "next-token prediction loss" (§3); details are thin. §6.2: "SFT training directly on answers is a version of off-policy
  learning, and dynamics may change when training directly on model outputs". So the SFT targets are the dataset's reference solutions,
  not the model's own samples; loss mask, epochs and example count for SFT are not stated.
- Learning rates (§5.1): swept over {1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 1e-4, 2e-4} and the best per update size taken, "averaged over three random
  seeds". The best learning rate at each update size is not reported in the text. Their remark: "changes in update size are known to
  alter effective learning rate". Inference: the grid tops out at 2e-4, so any need for much larger rates at tiny sizes would be invisible.
- Seeds: three. Backbones: Qwen2.5-{0.5, 1.5, 3, 7}B-Instruct, Qwen2.5-7B-Math (Table 2 row), Qwen3-8B (§6.4), Llama-3.2/3.1 Instruct.

### Measurement

- Accuracy (pass@1) on the GSM8K validation set, and on MATH500, Minerva, OlympiadBench, AIME24, AMC23 (Table 2; GAOKAO and CollegeMath are listed
  but not in Table 2). Not a PGR; they speak of "recover 95% of the net performance improvement" relative to base and full fine-tuning (§6.1).
  Error bars: none shown in the tables; three seeds averaged for the sweep. Held-out: yes (validation/test sets, not training problems), although the
  GSM8K runs train on the GSM8K train split of the same distribution.

### Headline numbers

- Qwen2.5-7B-Instruct, GSM8K, GRPO: "for a baseline of 76%, models trained with RL score 91% (15% absolute improvement) at 13 parameters and
  95% at 120 parameters" (§6.2); "We are able to recover 95% of the net performance improvement on GSM8K with only 120 parameters trained"
  (§6.1). Single parameter: "a 4% performance increase". SFT at the same sizes: "83% at 13 parameters and 84% at 120" (§6.2), and
  "SFT works best with larger update sizes of at least 1M parameters" (Fig. 2 caption).
  Caution: Table 2 lists the 7B-Instruct untrained GSM8K as 88.2, and 76.0 for Qwen2.5-3B-Instruct; the "76% baseline" in the
  text corresponds to the 3B row (or to a different eval setup) and does not match the 7B Table 2 row. I could not resolve this from the text.
- Table 2 (Qwen2.5-7B-Instruct, six-benchmark average, GRPO on MATH data): base 40.3; 13 params 50.1; 49 params 49.6; 196 params 53.2;
  392 params 54.4; 100,352 params 54.6; full FT 55.2. So 196 parameters retain (53.2 - 40.3)/(55.2 - 40.3) = 87% (my arithmetic; matches the text's
  "retains 87%"). The baseline rows for full fine-tuning and some others are "reported from (Zeng et al., 2025)" (Table 2 caption), i.e. not rerun.
  Hard-benchmark entries are noisy: AIME24 for 7B at 13 / 49 / 196 parameters is 16.0 / 12.6 / 16.7 (single-digit problem counts out of 30).
- Qwen2.5-7B-Math: base 37.2, 196 params 43.7, 392 params 52.8, full FT 59.5, so 196 parameters recover only about 29% of the gap; 392 recover 70% (Table 2; my arithmetic).
- Scaling: Fig. 3 and §6.4: the update size needed for a given fraction of peak performance falls with model size ("Larger models require smaller updates");
  Qwen is "around 10x" more efficient than Llama (§8, Discussion); "when we train fewer than five parameters, LLAMA barely improves performance".
  Qwen at one parameter reaches "near 82%" (§6.4), "a 5% improvement over the baseline". They flag possible contamination for Qwen (Wu et al., 2025).
- Curve shape: "relatively smooth interpolation from TinyLoRA (1-10^3) to LoRA-XS (10^3-10^6) to LoRA (> 10^6)" (§6.1); no logistic fit.

### Limitations stated and scope

- §8: "our findings are limited to math datasets". Discussion offers (as a theory, not a result): "the knowledge required to solve the task is
  already stored in the parameters of the model, and only the style has to change for task success. In particular, learning to generate longer outputs
  may be possible in few parameters".
- Thinking traces: responses are capped at 3-4k tokens; reasoning means chain-of-thought for math word problems and competition math on instruct or math
  models (not a thinking-mode model with multi-thousand-token traces). Instruction following is not studied.
- Their central claim is that the parameter-efficiency is specific to RL: SFT "require[s] 100-1000x larger updates". Their account (§3) is an information-content
  argument (RL reward is at most k bits per prompt, SFT absorbs all the demonstration's bits); it is a hypothesis, not tested by ablation (for instance they never
  compare SFT on the model's own verified samples).

### Does this match our parametrisation? What would it predict?

- No. The update here is a u-vector in a fixed SVD-aligned subspace with weight tying, not a mask over weight entries; the count of 13 reflects
  extreme tying across all 7 x n_layers modules. Our random B entries are untied, at fixed random positions, and independent per module.
- Predictions relevant to us (inference): (1) For SFT, the paper says tiny updates trained by SFT do poorly: 83% vs 91% at 13 parameters on a
  76%-to-95% task, and SFT needs hundreds to thousands of times more parameters than GRPO to reach the same level. Our SFT-on-1-64-examples design
  sits on the unfavourable side of this contrast. (2) Their SFT targets were reference solutions, so the off-policy caveat they raise in §6.2 applies:
  we will SFT on the RL model's own outputs, which they flag as possibly different and did not test. (3) Larger models need fewer parameters, which
  favours us. (4) Their reported learning-rate grid for the tiny runs is 1e-7 to 2e-4 and the optimum per size is not given, so no usable learning-rate
  rule can be taken from it.

---

## 3. Sinii et al., "Small Vectors, Big Effects: A Mechanistic Study of RL-Induced Reasoning via Steering Vectors", arXiv 2509.06608

### Parametrisation

- §2: "zero-initialized steering vectors s_l in R^d are added to the output residual stream of each layer l, while all other weights remain fixed. The vectors
  are trained with the RLOO objective". All-layer version: one d-dimensional vector per layer (L x d parameters); per-layer version (§3): a single vector at
  layer l with every other layer unchanged. The paper does not state the parameter count. My arithmetic from the architecture: Qwen2.5-Math-7B has hidden size 3584
  and 28 layers (so about 100k parameters all-layer, 3,584 for one layer); Llama-3.1-8B has 4,096 and 32 layers (about 131k all-layer). The companion paper's Table 5
  gives 100K and 131K for the all-layer versions, matching this arithmetic (§3, below).
- Token-independent: the same vector is added at every position. This is a different object from any weight-matrix adapter.

### Training

- RL, RLOO, binary Math-Verify reward on the DeepScaleR dataset, temperature 1.0, 128 prompts and 16 generations per gradient step, Adam without weight decay,
  4K context for Qwen2.5-Math-7B and 8K for Llama3.1-8B-It (§3). The number of steps and the learning rate are not stated in this paper (I searched the full text,
  including the appendices; for rates see the companion paper below). Seeds: "All metrics were averaged over three evaluation seeds"; the paper does not
  say how many training seeds. Open-s1 is used as a second training set for the last-layer experiments (App. H).

### Measurement

- Mean over six math benchmarks (AIME24/25, AMC23, MATH500, MinervaMath, OlympiadBench); PASS@1 for MATH500/Minerva/OlympiadBench and AVG@32 for
  AIME24/25 and AMC23, sampling at temperature 1.0, 4K eval context for the Qwen-Math model and 32K for Llama. Plus-minus values in App. B Tables 2-3 are across evaluation seeds.
  The benchmarks are held out from DeepScaleR training. The "Base Model" row in the tables uses greedy decoding, and a separate row "Base Model. tau = 1" gives
  the base sampled at temperature 1 (the actual starting point of RL).
- Important for reading gap numbers: Qwen2.5-Math-7B base is 24.8 greedy and 14.3 sampled at tau = 1; full RL is 43.8 and all-layer steering 42.9 (Table 2, App. B).
  Llama3.1-8B-It base 21.5 greedy, 11.7 at tau = 1; full-tune 26.4, steering 25.8 (Table 3). A large share of the gain relative to the tau = 1 starting point
  is entropy reduction. The paper says as much about single-layer vectors: "under greedy decoding, several do [match], suggesting that single-layer vectors target the right
  mechanisms yet cannot on their own sufficiently reduce the next-token distribution's entropy" (§3).

### Headline numbers

- All-layer steering matches full RL: 42.9 vs 43.8 on Qwen2.5-Math-7B and 25.8 vs 26.4 on Llama3.1-8B-It (mean accuracy; Tables 2-3).
- A single vector at a single layer (Fig. 1): "Many layers recover a substantial fraction of the gain from full fine-tuning". E.g. Qwen2.5-Math-7B mid layers
  reach about 38-40 (layer 14: 40.0, layer 16: 40.1 against full 43.8 and tau = 1 base 14.3); last layer (s27) 29.4, "closed over 50% of the gap between the base model and full
  training" (§5). Layers 23 and 24 are anomalously poor (20.9, 21.1) due to the next layer's input LayerNorm (§3, App. C).
  So even a single 3,584-dimensional vector recovers most of the RL gain on this model/benchmark mix (under sampling at tau = 1).
- Last-layer vector acts like first-token bias: logit-lens top token "To" (cosine 0.37); "appended 'To' to each prompt and evaluated the base model: accuracy increased by 10-11 points
  under both greedy decoding and sampling", "about 75% of the gain from s27" (§5, Fig. 4). Llama: "Step"/"final", weaker, and the prefix beats last-layer steering under greedy.
- Penultimate vector works largely via the MLP and the value-output path, not attention patterns (§6, Figs. 5-8).
- Transfer: vectors trained on one same-family checkpoint partially transfer (Table 1, normalised gain from -0.34 to 0.94; e.g. Qwen2.5-7B Base -> Math 0.74, Instruct -> Math -0.34).
- Curve shape: no parameter-count sweep, so no scaling curve.

### Limitations stated, and scope

- No dedicated limitations section. The conclusion says effects are "consistently stronger and cleaner in Qwen than in Llama" and that mid-layer vectors are "harder to localize".
  Models are 7-8B math/instruct models, short math answers, 4K-8K context. No thinking-mode model and no instruction-following target.
- Relevant caution for us: part of the "elicitation" gain is a first-token / opening-phrase change and entropy sharpening, which an RL gap measured at temperature 1 will contain.

### Does this match our parametrisation? What would it predict?

- No. Residual-stream additive vectors (RL-trained) are a different parameterisation from entries of a rank-1 LoRA B factor. A B-entry affects one output channel through a . x, so it can
  change per token; a steering vector cannot. The companion paper (below) reports LoRA closes the remaining gap that a vector leaves, which points the same way as the claim that token-specific
  adapters are at least as expressive.
- What it predicts: the RL gain on Qwen/Llama math is reachable with about 3.6k-4.1k parameters at one layer, and about 100-131k across all layers, when trained by RL on 128 x 16 samples per step for many steps. It says nothing about SFT on 1-64 examples.
  Its mechanistic finding (first-token bias, entropy reduction) does suggest checking whether our small-k gains concentrate on the opening tokens of the thinking trace.

---

## 4. Sinii et al., "Steering LLM Reasoning Through Bias-Only Adaptation", arXiv 2505.18706

### Parametrisation

- §3.2: "We insert a learnable steering vector s_l in R^d at the end of every transformer layer l ... added directly to the residual stream, so its dimensionality matches the
  model's hidden size d. All original weights remain frozen; only these L steering vectors are trained." Appendix A and B give the same (zero-initialised `nn.Parameter` added to the layer output).
  Inconsistency to note: §2 states "We implement steering vectors only by activating only layer-wise additive biases in MLPs while keeping all other model parameters frozen", whereas
  §3.2 and App. B add the vector to the layer output. I treat the residual-stream addition (code listing) as what was run.
- Parameter counts (Table 3, Table 5): Qwen2.5-14B 245K, Qwen2.5-7B 100K, Llama3.1-8B 131K; "0.0016%" for an 8B model (abstract). These match L x d (e.g. 48 x 5120 = 245,760).
- App. D/E add a LoRA variant: rank 4, alpha 4, no dropout, on the MLP down-projection in every layer (the main-text table is the vector).

### Training

- Online RL, RLOO, binary reward from a \boxed{} answer, DeepScaleR for the main table, temperature 1, 128 prompts per step, 4K context for the Math models and 8K for the others (§3.1, 3.3).
  GSM8K (8,790 problems) and MATH (12,000) runs in App. E, evaluated as mean@8.
- Hyperparameters (App. I, Table 6): this is the only place in these four papers where a learning rate per adapter size is given.
  - Qwen-2.5-1.5B: full fine-tune 2e-5, steering 5e-4, LoRA rank 1 and 4 at 5e-4.
  - Qwen-2.5-Math-1.5B: full 2e-5, steering 1e-3, LoRA 5e-4.
  - Llama-3.1-8B: full 5e-6, steering 5e-4, LoRA 1e-4.
  - Llama-3.1-8B-Instruct: full 1e-6, steering 2e-4, LoRA-1 6e-4 (GSM8K) / 3e-4 (MATH), LoRA-4 1e-4 / 3e-4.
  So the tiny-vector learning rates are 25x to 200x those used for full fine-tuning of the same model (e.g. 5e-4 vs 2e-5; 2e-4 vs 1e-6).
  Number of generations per prompt: 16-64. How rates were chosen is not stated. Number of steps: only "314 steps ~ 1 epoch" is mentioned for timing (Table 3).
  Compute: 16 H100s; 2.5 to 120 hours per model (App. H, garbled in the text: the 120 h entry is labelled for LLaMa3.1-8B-It twice).

### Measurement

- Same six benchmarks as the 2509 paper; Table 1 reports "Base" (greedy), "Full-Tune", "Steering" for seven models. Fine print: "Base models decode greedily, whereas trained models use sampling with tau = 1.0"
  (§3.3). So the comparison of base against trained mixes decoding settings; the Steering-vs-Full-Tune comparison (both sampled) is clean.
- Held out; error values are across three evaluation seeds. Training seed count is not stated.

### Headline numbers

- Table 1 (six-benchmark average, base / steering / full-tune): Qwen2.5-1.5B 0.3 / 23.2 / 22.7; Qwen2.5-7B 24.2 / 36.4 / 37.1; Qwen2.5-14B 26.1 / 42.3 / 41.3;
  Qwen2.5-Math-1.5B 18.1 / 33.9 / 33.6; Qwen2.5-Math-7B 24.8 / 43.3 / 43.5; Llama3.1-8B-It 21.7 / 26.1 / 26.1; Llama3.1-8B 3.9 / 9.1 / 11.5.
  Text: "The only exception is LLaMa3.1-8B, where steering recovers about 70% of the full-tuning gain" (§4.1; (9.1 - 3.9)/(11.5 - 3.9) = 68%, my check).
- Table 4 (GSM8K and MATH training, mean@8): steering is within a few points of full-tune in most cells and below it by 5-7 points in several (Qwen2.5-Math-1.5B 79.89 vs 86.49;
  Llama-3.1-8B 70.36 vs 76.49; MATH-trained Llama-3.1-8B 22.05 vs 27.39), and above it in others (Qwen2.5-1.5B on MATH test 48.69 vs 40.78). LoRA rank 4 is closer in nearly all cases.
  The abstract's "matches" is therefore the best-case summary of the DeepScaleR table.
- Resource cost: Qwen2.5-14B optimiser memory 13.8 GB vs 240 KB; 52 min vs 34 s per epoch (Table 3).
- Curve shape: none (no parameter sweep).

### Limitations stated

- "our experiments cover a narrow slice of online-training settings"; logit-lens does not capture downstream layers (Limitations). No discussion of thinking-mode models or instruction following.
- The logit-lens cluster analysis used an LLM (GPT-o3) to group tokens (App. F); the clusters in Table 2 are for Llama3.1-8B-It trained on GSM8K.

### Does this match our parametrisation? What would it predict?

- No (residual-stream vectors, RL). Value for our design: (a) the learning rate for a tiny-parameter adapter trained on the same loss was 25x-1000x the full-FT rate in the same codebase;
  (b) a token-specific low-rank adapter closed the gap that a global vector left (Table 4), so in the same setting more expressivity per parameter helps; (c) even a single d-dimensional vector
  per layer (about 0.0016% of an 8B model) matches RL, but under RL, not SFT.

---

## What transfers to our design and what does not

Our draft design (for the smallest cells): k = 256 or 4,096 randomly chosen entries of the B factor of a rank-1 LoRA on all attention and MLP projections; A frozen at a seeded random initialisation;
all other B entries 0; SFT on 1-64 responses (thinking plus answer, several thousand tokens each) of the RL-trained checkpoint, applied to the pre-RL Olmo 3.1 Think 32B.

### Transfers

1. Parametrisation. Only Donoway et al. use a comparable object (random entries of LoRA matrices, zero-init B, Kaiming-style A frozen, uniform mask over all projections, mask held fixed per run).
   Their reported finding that attention projections are the more parameter-efficient place to spend a sparse budget (App. C.1, Fig. 8) is a hint that a K/Q/V/O-only ablation is worthwhile.
2. Measurement. Donoway et al.'s gap-closure definition is the one most similar to what we want, but note its ingredients: floor = zero-shot base; ceiling = full fine-tune (or the logistic maximum); the point at each budget
   is the best over up to 5 seeds x 50 hyperparameter trials. If we copy it, we should say whether we report best-of-N or a seed mean, and use our existing RL checkpoint as the ceiling, which is a different (and cleaner) ceiling than their full SFT.
3. Scaling with model size. All of Donoway (Table 1, Fig. 1), TinyLoRA (Fig. 3, §6.4) report that bigger models need fewer parameters, which makes a 32B model a favourable case for small k.
4. Learning rates and optimiser behaviour at tiny sizes. Donoway: optimal rate depends strongly on sparsity; below about 20 parameters gradient descent was unstable, and they used a derivative-free search over weights bounded by +/-1 to +/-10.
   Sinii bias-only: the tiny-parameter rates were 25-1000x full-FT rates. TinyLoRA: grid 1e-7 to 2e-4 with optimum not reported. Plan on a learning-rate sweep over several decades for each k, not one rate.
5. Mechanistic caution (Sinii 2509): part of the RL gain is a change to the first tokens and a reduction of sampling entropy. Our cell results should check opening-token and temperature effects before attributing recovery to "reasoning".

### Does not transfer, or is untested

1. Number of examples. None of the four papers varies the number of training examples for the tiny-parameter SFT regime. Donoway fixes 1,000 traces for reasoning (and states finite-dataset-size as a limitation);
   TinyLoRA's SFT details are unspecified; the RL papers use thousands of prompts and many steps. Nothing here predicts what 1-64 examples buy.
2. Thinking traces at 32B and several thousand tokens. The closest experiment (Donoway's R1-distill SFT on 1,000 traces up to 32k tokens) is on a 1.5B Qwen model; the bar was about 10k-20k parameters for half the AMC24 gap (maj@16 and loss), above both of our k values.
   Neither TinyLoRA nor Sinii uses a thinking-mode model.
3. Starting point. Their base models are pretrained-only or instruct models whose gap to the target is largely format and style (Donoway's base-to-R1-distill, Sinii's base-to-RLVR). Ours is a pre-RL checkpoint of an already-thinking model;
   the gap we try to close may be smaller and of a different kind (sharpening, length, consistency). Inference, not something any paper tests.
4. SFT vs RL. TinyLoRA argues (as a hypothesis) that SFT needs 100-1000x more parameters than RL for the same recovery, and shows SFT at 13 parameters reaching 83% against 91% for RL on GSM8K. Sinii's all-layer vectors are RL-trained.
   Only Donoway's results are SFT-based and they trained to convergence on data from a stronger model. Their SFT gap numbers are therefore the relevant ones for us, but those are largely classification tasks.
5. Mask over B only vs over A and B. Donoway's mask includes A entries, which are inert while B is zero; the effective count at small k may be below nominal there (my inference). Our B-only count is the effective one.
6. Rank-1 entry semantics. A single B entry changes a single output channel of a single module (b_i * a . x). Donoway's results suggest this is enough at hundreds to thousands of entries for classification; no paper shows it for long-CoT generation at 32B.
7. Their "up to 50% at 10-100 parameters" headline is for multiple-choice classification with one-token answers, not generation. Quoting it for Olmo would overstate it.

### Predictions, stated cautiously (inference)

- k = 256: well below every tabulated 90% gap-closure budget in Donoway's Table 1 for non-trivial tasks, and below the 1.5B reasoning 50% point; expect a small fraction of the gap unless the size benefit is large.
- k = 4,096: in the range where Donoway's classification tasks reach about 90% for 1B-8B models; still below the 1.5B reasoning 50% point. Whether 32B and an already-thinking base shift this by an order of magnitude is the experiment.
- Across all four papers the curves, where they exist, rise smoothly with the log of the parameter count (logistic in Donoway; interpolating in TinyLoRA), so a ladder of k values across several decades is more informative than two points.

---

## What I did not read

- Donoway et al.: everything text-extractable was read (all 32 pages, including the NeurIPS checklist). Plots (Figs. 1-5, 7-12) were not read as images, only their captions and the text values embedded in the PDF, so curve values not stated in text or tables are unseen.
  The code repository (`edonoway/quantifying-elicitation-neurips25`) was not read; the paper's checklist says code release was planned after anonymisation. No arXiv version was found by title search, and OpenReview was not reachable (404/blocked HTML), so I used the proceedings PDF only.
- TinyLoRA: all 14 pages read, including the appendix (A.1). No code or supplementary material was found/read. Figure data (Figs. 1-9) were not read as images.
- Sinii et al. 2509.06608: main text and appendices A-J read; the long numeric tables in App. B (per-layer raw scores), D (raw cosine similarities) and J (unnormalised transfer) were only skimmed (headers and a sample of rows), not checked line by line. Figures not read as images.
- Sinii et al. 2505.18706: all 10 pages read. Figures not read as images.
- Cited-but-unread: "LoRA Without Regret" (Schulman and Thinking Machines Lab), LoRA-XS, SimpleRL-Zoo, Perez et al. (Rissanen data analysis). The earlier note's other items (Bits That Count) were not part of this task.
- I did not verify any claim against the authors' code or reproduce a number; contamination concerns for Qwen (TinyLoRA §6.4) were taken from the paper's text, not checked.
- Olmo 3.1 architecture figures (hidden size, layers, projection shapes) were not looked up for this note, so I have not computed how many modules or B entries our k = 256 / 4,096 spread over; I only used architecture numbers for Qwen and Llama where the papers' own tables confirm them.
