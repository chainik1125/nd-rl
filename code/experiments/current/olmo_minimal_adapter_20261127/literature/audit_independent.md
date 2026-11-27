---
written_on: 2026-11-27
written_by: "agent:claude (opus, auditor)"
---

# Independent audit reading: EDL and minimal elicitation (three Donoway et al. papers)

Phase 1 of the audit: my own reading, made before looking at the other readers' notes.
Statements marked **[inference]** are my own analysis and do not appear in the papers.
Everything else is taken from the papers, with section, equation, table or figure references.

## Sources and how I got them

| Tag | Paper | Copy I read |
|---|---|---|
| **P1** | Donoway, Joren, DeWeese, Perez, Schulman, Roger, Leike. "Bits That Count: Quantifying and Predicting Capabilities of Language Models", ICML 2026, PMLR 306 | `donoway26a.pdf` (25 pp) from the mlresearch v306 GitHub assets |
| **P2** | Donoway, Joren, Roger, Leike. "Excess Description Length of Learning Generalizable Predictors", arXiv 2601.04728v1 (8 Jan 2026) | arXiv PDF (26 pp) |
| **P3** | Donoway, Joren, Sleight, Somani, Schulman, Michael, DeWeese, Perez, Roger, Leike. "Quantifying Elicitation of Latent Capabilities in Language Models", NeurIPS 2025 | NeurIPS proceedings PDF `dd16e008534dc52fab72fdedad069d5c-Paper-Conference.pdf` (32 pp, with appendices A–J and the checklist). The OpenReview PDF returned HTTP 403 (a challenge page), so I could not compare against it. |

Local copies are in `/scratch/cowifornia/nd-rl/olmo_elicit/v2/papers_audit/` (`dl_p1/`, `dl_p2/`, `dl_p3/`), with text extracts in `text/`.
The P3 code repository (`github.com/edonoway/quantifying-elicitation-neurips25`) is public but empty: on 2026-11-27 the GitHub API answered "Git Repository is empty."

---

## Q1. EDL: definition and computation

**Definition.**
- P1 §2.2, Eq. (1) and P2 Def. 3.1, Eq. (6) define prequential MDL as MDL(D; θ0, A) = Σ_i ℓ(θ_{i−1}; x_i, y_i), with ℓ(θ; x, y) = −log p_θ(y | x).
- P1 Eq. (3) and P2 Def. 3.2, Eq. (7) define EDL(D; θ0, A) = MDL(D; θ0, A) − n · L_test(θ*).
- P2 Eq. (9) rewrites EDL as a per-step sum: EDL = Σ_i [ℓ(θ_{i−1}; x_i, y_i) − L_test(θ*)], "which makes explicit that EDL accumulates the per-example excess over the training trajectory".

**What is scored, and when.**
- **Batching.** P1 §2.2: "For each batch, we accumulate the total log-loss (summed over all labels in the batch) before the gradient update." P2 §3.1: all examples in a batch "are encoded using the pre-update parameters".
- **First epoch only.** P1 footnote 1: "Prequential MDL accumulates loss over the first epoch only … Accumulation over additional epochs would correspond to encoding redundant information". P1 §2.5: "for multi-epoch training, MDL is computed only on the first epoch (first exposure to the data), while test loss is computed using the final trained model (arbitrary epochs)."
- **Test loss.** P1 Eq. (2) gives L_test(θ*) = (1/n_test) Σ_j ℓ(θ*; x_j, y_j), computed on "held-out data from the same distribution". It is a mean per example, where each example's loss is summed over its label tokens.
- **Order.** P2 §3.1: "When comparing MDL across experiments, the same ordering (or averaging over orderings) should be used." P1 App. G.1 (data-controlled runs): "ensuring that the same examples are seen in the same order by each model during the first epoch".

**The residual term n · L_test(θ*).**
- The final model's per-example test codelength, multiplied by the number of training examples n (P1 Eq. 3).
- P2 §3.2 describes it as "the bits that would be required to encode n new samples from the same distribution using the final model".
- **[inference]** Because L_test is a mean per example, not per token, n · L_test is only a fair stand-in when train and test labels have the same length distribution. This matters for long, variable-length thinking traces, and for any selection step (such as verifier filtering) that changes lengths.

**Termination condition for θ\*.**
- P1 §2.3: "After training to some termination condition (e.g., convergence, some number of epochs, or a fixed compute budget), obtain final parameters θ∗."
- P1 Table 1 gives the stopping criterion as "Validation loss convergence".
- P1 App. E, footnote 5: "'Convergence' is determined by maximal performance on the chosen validation metric (either maximization of the validation accuracy or minimization of the validation loss); for a single example, this usually occurs after a single training step."
- P1 App. D: "we train to convergence in all experiments."
- P2 App. D.2: "maximum EDL can be obtained by training to best achievable performance on the validation set (with early stopping prior to overfitting) and evaluating test loss on the best model checkpoint."
- P2 Def. 4.1 and Thm 4.2 guarantee E[EDL] ≥ 0 only for a "population-monotonic" algorithm. The listed violations include "Training well past the optimal early stopping point on finite data" (P2 §4.1, Remark 4.3). Early stopping on a validation split is therefore part of the recipe.
- P1 never spells out its validation, test and train split sizes.

**Units and normalizations.**
- Losses are computed in nats; "we convert to bits by dividing by ln 2" (P1 App. F, §2.5; P2 §2.1).
- P1 §2.4 defines three normalizations:
  - EDL/D: D is "total label token count", used for scaling analysis.
  - EDL/P: P is the "trainable parameter count", used for capacity analysis.
  - Capacity (the EDL ratio): EDL(P, D)/EDL_ref(D), "The reference EDL is obtained from full fine-tuning or high-rank LoRA on the same data".
- P1 footnote 2: "we use rank 512 LoRA as a reference, EDLref = EDLr=512, which obtains the same performance as full fine-tuning in all of our experiments."

**Loss mask.**
- P1 §2.1: "We compute cross-entropy loss only on designated label tokens, typically the answer portion of the output (excluding the prompt and formatting)."
- P1 App. F, arithmetic: "we score only the numerical answer tokens. The prompt … and any formatting tokens are excluded from both MDL computation and test loss evaluation."
- P1 App. F, reasoning (Qwen): "we score the full response including chain-of-thought reasoning final answer". App. B: "we score the full reasoning trace plus answer."
- P1 §2.5: "token masking procedures … should be consistent between the train and test sets."
- P2 §5.5, Prop. 5.5 and App. B.5 recommend scoring only capability-relevant tokens, or teaching the format first, because format learning "creates early transients distinct from capability acquisition".

**Hyperparameter choice.**
- P1 Table 1, applied to all experiments: AdamW (β1 = 0.9, β2 = 0.999), weight decay 0.01, constant learning rate, gradient clipping 1.0, bfloat16, "Validation loss convergence", 3 seeds per configuration.
- P1 Table 2 (LoRA): all of Q, K, V, O, G, U, D in all layers; rank sweep 1 to 512; α = 32; scaling "α/2r"; dropout 0; A initialized Kaiming (scale 1/√d_in); B initialized to zero. App. G.1.1 adds: "the same learning rate for both A and B". (The PEFT default scaling is α/r. The paper says α/2r.)
- P1 App. H.1.2 gives the reasons for AdamW with a constant learning rate:
  - AdamW gives "EDL values closest to the supremal EDL".
  - Decaying schedules "make the effective learning rate dependent on dataset size".
  - RSLoRA makes the learning rate rank-dependent.
- P1 App. H.1.2 also says: "After appropriate renormalization … all tested configurations yield identical qualitative signatures and same-order-of-magnitude capacity limits … As renormalization is not straightforward, we omit those results".
- Learning rates and batch sizes are listed under Q2.

**Seeds, and how results are reported.**
- P1 Table 1 gives "Seeds per config 3". No figure or table in P1 shows error bars, confidence intervals or seed spread. P1 never says whether a reported number is a mean, a median or the best seed.
- P2 §6.1 recommends "reporting confidence intervals or variance estimates alongside EDL measurements, particularly for small datasets". P2 App. C.1 gives Var[EDL] = O(n).

**P3 uses a different quantity.**
- P3 reports a prequential first-epoch codelength (App. I, Algorithm 1: `L_k += ℓ_B × |B|`) and an "MDL compression" Δ_k = L_0 − L_k (§5, App. I).
- Here L_0 is the codelength under the frozen base model. P3 does not subtract a test loss, so this is not EDL.
- It is computed only "for the optimal (s, h) configuration", where hyperparameters are chosen to "maximize validation accuracy" (App. I, steps 1–3).
- It is not available for the Bayesian-optimization runs with fewer than 20 parameters: "we are unable to calculate prequential minimum description lengths for the resulting fine-tuned models" (App. C.4).

**[inference] Small-n consequences.**
- If the batch size is at least n, the whole first epoch is a single batch. MDL is then just the base model's zero-shot loss on the training labels; no online coding takes place.
- EDL then reduces to n · (mean zero-shot training loss − L_test(θ*)), which is a scaled loss improvement.
- For n = 1, EDL = ℓ(θ0; x1, y1) − L_test(θ*).
- P2 contradicts itself on this point. Prop. 5.2 and B.2 say a single example "can contribute at most min (log |Y|, log m) bits to EDL". Prop. B.3 and Eq. (69), however, write "EDL = n · (L(θ0) − L(θ∗)) = 1 · log m", which is not Def. 3.2.

---

## Q2. The capacity threshold

**EDL_ref.**
- EDL of rank-512 LoRA on the same data (P1 footnote 2).
- P1 Table 4 lists only LoRA (no full fine-tuning) for the Qwen reasoning runs. Fig. 5 shows ranks 1–256 for Qwen2.5 14B.
- P1 does not say what the Qwen reference was or what the PGR denominator for Qwen is ("Perf_FullFT").

**The rule for P\*.** P1 states it three ways that are close but not identical:
- §5.1: "The saturation point for each model, EDL/P∗, where PGR drops below 1, corresponds to the maximum learnable information density (per trainable parameter)."
- Table 6 caption: "EDL/P∗_EDL is the threshold where EDL/EDLref drops below 0.95; EDL/P∗_PGR is the threshold where PGR drops below 0.95."
- Table 7 caption: "EDL/P∗ at which EDL/EDLref = 0.95".

**What the rule pools, and what P1 does not specify.**
- Each plotted point is one (LoRA rank, dataset size) pair. §5.1: "Each point compares the fraction of the total performance gap recovered by the specified LoRA rank at a given dataset size". Ranks and dataset sizes are pooled onto one EDL/P axis ("Curves for individual LoRA ranks collapse onto each other", Fig. 3 caption).
- How the crossing is located is not stated: first point below 0.95, interpolation, or a fit.
- Which EDL sits in the x-axis numerator is not stated: the run's own EDL, or EDL_ref of that dataset.
- The practical guidance in §6.4 implies the reference EDL: "Estimate EDL from a pilot training run with high-rank LoRA at the target dataset size … Select an appropriate adapter size (based on P∗) to ensure EDL/P falls within the regime-appropriate threshold (<0.1 bit/parameter for elicitation, ~1+ bit/parameter for teaching)."
- Thresholds were computed only for "datasets that are large enough to exceed parameter capacity (larger than ~10K examples, as well as Qwen-R1/s1K Reasoning)" (P1 App. I.2).
- Formal regime definitions (P1 App. J.2): elicitation "P∗ ≈ 0.01–0.1 bits/parameter"; teaching "P∗ ≈ 1 bit/parameter".
- P1 Fig. 7 (conceptual) adds that a demonstrated capability (already elicited) should show "P∗ ≲ 0.01 bits" and "EDL very low".

**How P is varied.**
- P1 §3.3: "We train ranks 1 to 512, including uniformly randomly sampled sparse subsets of parameters within rank 1. Trainable parameter counts range from a single parameter to hundreds of millions".
- P1 gives no further detail on the sparse subsets: no sampling unit, no initialization or freezing of the unselected entries, no learning rate, and no count rule beyond "P is trainable parameter count".
- The mechanism is described only in P3:
  - §3.1: "learnable subsets of the LoRA adapter matrices are trained while the remaining adapter weights are held frozen at their initialized values. We select these trainable parameters using uniform random sampling across all LoRA modules in all layers, with each parameter having equal probability of selection regardless of its position in the network. This selection is implemented by applying a random binary mask over gradients during the backward pass".
  - App. B.1: "the mask is sampled once at the start of each run and held fixed for the duration of training".
  - App. C.1: "When we refer to 'trainable parameters,' we exclusively mean scalar values within LoRA adapter matrices … Matrix A: R^{r×d_in} (Kaiming uniform initialization); Matrix B: R^{d_out×r} (zero initialization); Total parameters per module: r × (d_in + d_out)."
  - App. B.1 (Alpaca): with fewer than 10k parameters, "the original language-model head remains frozen", while "at larger budgets the head is fine-tuned together with the adapters".
  - §3.1: they "compare the weights of the LoRA modules at initialization to the weights following fine-tuning and ensure that only the unmasked parameters have been updated."

**[inference] Dead A entries, and what a "random parameter" actually is.** This is not stated in the papers.
- The unselected B entries are frozen at zero. The gradient on A is proportional to Bᵀ · ∂L/∂W.
- So a selected A entry gets no gradient unless some B entry in the same module and rank component is also selected. In a uniform draw of 10–100 entries spread over 7 × (number of layers) modules, almost all selected A entries will be dead.
- Bayesian optimization has the same problem: setting an A value has no effect while that module's B is zero.
- Each live selected B entry b_i scales a fixed random Kaiming vector a into output row i: ΔW[i, :] = s · b_i · a.
- So the "effective" P is roughly the number of selected B entries, and "k random parameters" means about k/2 or fewer scalar gains on fixed random rank-1 directions.
- A second pitfall: masking gradients while using AdamW with weight decay still decays the "frozen" A entries. P3's weight-comparison check would catch this. P1 uses weight decay 0.01.

**Bits per parameter (all from P1).**

| Model / condition | Add/Sub EDL/P\*_EDL | Add/Sub EDL/P\*_PGR | Mult EDL/P\*_EDL | Mult EDL/P\*_PGR |
|---|---|---|---|---|
| Llama 3.2 1B | 0.03 | \* | 0.03 | 0.14 |
| Llama 3.2 3B | 0.01 | \* | 0.009 | \* |
| Llama 3.1 8B | 0.01 | \* | 0.004 | \* |
| Pre-elicit mult. (1B) | \* | \* | 0.07 | 0.06 |
| Pre-elicit add/sub (1B) | \* | \* | 0.07 | 0.06 |
| TinyStories-1B base | 2.21 | 2.23 | 0.70 | 1.02 |
| TinyStories format only | 1.04 | 0.96 | 0.82 | 0.73 |
| TinyStories pre-teach mult. | 0.93 | 1.07 | 0.06 | 0.05 |
| TinyStories pre-teach add/sub | 1.81 | 1.50 | 1.48 | 1.56 |

- Source: P1 Table 6. "\*" means "PGR ≥ 0.95 for all dataset sizes tested".
- P1 Table 7 (EDL/P\* at EDL/EDL_ref = 0.95):
  - TinyStories-v2: Llama 1B base 0.02; randomly initialized 1.2.
  - Alpaca: 1B 0.03, 3B 0.02, 8B 0.02.
  - Qwen-R1/s1K: Qwen2.5 1.5B 0.04, 14B 0.05, 32B 0.02.
- §6.3: Qwen 14B "around 0.05 bits/parameter".
- §6.2 and §7 summarize the regimes as "~0.01–0.1 bits/parameter for elicitation, ~1 bit/parameter for teaching".
- App. I.2 says "Larger models have lower elicitation thresholds". This holds for the Llama arithmetic runs. It does not hold for Qwen: 14B (0.05) is above 1.5B (0.04).

**Learning rates per size (P1 Tables 3–4, App. H.1.5).**
- DeepMind Mathematics: TinyStories-1B and Llama 3.2 1B use LoRA 3.53e-4 and full fine-tuning 2e-5. Llama 3.2 3B and Llama 3.1 8B use LoRA 1e-4 and full fine-tuning 2e-6. Batch 128, effective batch 1024, 8×H100.
- Qwen2.5 1.5B, 14B and 32B (reasoning): LoRA 1e-4, batch 1, effective batch 8.
- Other tasks: LoRA 1e-4, full fine-tuning 2e-5.
- Each size uses one learning rate across all ranks. No learning rate is given for the sparse subsets or the single-example runs.
- P3, by contrast, searches per budget: "Learning rate: [10^−6, 1] (log-uniform)", batch size {1, …, 256}, weight decay [0, 0.1], with "up to 5 random seeds with up to 50 Bayesian-Hyperband trials each" (App. C.2). P3 explains why (App. H): "hyperparameters (learning rate, in particular) can differ significantly depending on the sparsity of trainable parameters".

**Uncertainty.** None is reported for any P\* value. Values are given to 1–2 significant figures, from 3 seeds per configuration, with no aggregation rule stated.

---

## Q3. Whose outputs are the training targets

**P1**
- **Arithmetic** (DeepMind Mathematics, add/sub and multiplication): ground-truth labels (App. B). Pre-elicitation uses a single ground-truth example of a *different* operation and format (App. E.1.1). TinyStories format pre-teaching uses "randomly permuted labels" (App. E.1.2). Algorithm pre-teaching uses 4M ground-truth operator-notation examples (App. E.2).
- **BoolQ, ARC-Easy/Challenge:** ground-truth labels.
- **Alpaca:** the dataset's responses. P1 does not discuss their origin.
- **TinyStories-v2:** corpus text.
- **Reasoning (Qwen2.5 1.5B/14B/32B base):** a stronger model's traces. P1 App. D: "R1-generated reasoning traces filtered for accuracy using prompts from the s1K dataset". §6.3: "distillation of DeepSeek R1 into Qwen2.5 14B by fine-tuning on R1-generated reasoning traces".
- **Random-label control:** permuted labels (App. I.10).
- **No experiment uses the model's own samples or verifier-selected self-samples.**

**P3**
- **ARC-Easy/Challenge, BoolQ:** ground-truth single-token labels.
- **GSM-8K-CoT-Choice:** "Correct"/"Incorrect" labels on solutions written by Claude 3.5 Sonnet, validated by "exact string match", a human and Claude 3.5 Haiku (App. D.1).
- **Alpaca:** dataset responses.
- **TinyStories:** corpus text.
- **Lichess:** puzzle solutions.
- **s1K-Qwen-1.5B:** traces from the post-trained sibling of the same base. "completions generated by DeepSeek-R1-Qwen-1.5B" (footnote 3), "up to a maximum length of 32768 tokens (temperature=0.6) … we verify that the reasoning traces have the correct format … but do not regenerate incorrect solutions" (App. D.2). So the traces are **not** filtered for correctness, unlike P1's "filtered for accuracy".
- **No experiment uses own samples.**

**P2:** theory only.

---

## Q4. What "minimal" means

**P1 Table 8**, "Performance gap recovery with minimal information". Each cell is PGR (accuracy gain in percentage points).

| Model | Task | 1 Example | 10 Params | 100 Params |
|---|---|---|---|---|
| Llama 3.2 1B | Add/Sub | 0.66 (+66) | 0.65 (+65) | 0.71 (+71) |
| Llama 3.2 1B | Mult | 0.31 (+31) | 0.30 (+30) | 0.31 (+31) |
| Llama 3.2 3B | Add/Sub | 0.86 (+86) | 0.87 (+87) | 0.89 (+89) |
| Llama 3.2 3B | Mult | 0.54 (+54) | 0.65 (+65) | 0.68 (+68) |
| Llama 3.1 8B | Add/Sub | 0.96 (+96) | 0.91 (+91) | 0.96 (+96) |
| Llama 3.1 8B | Mult | 0.67 (+67) | 0.76 (+76) | 0.81 (+81) |
| TinyStories-1B | Add/Sub, Mult | 0.0 | 0.0 | 0.0 |
| Qwen2.5 1.5B | Reasoning | 0.53 (+7) | 0.14 (+2) | 0.61 (+8) |
| Qwen2.5 14B | Reasoning | 0.45 (+23) | 0.38 (+19) | 0.55 (+28) |
| Qwen2.5 32B | Reasoning | 0.55 (+16) | 0.27 (+8) | 0.41 (+12) |

**How the cells were built.**
- **"1 Example"** is "full fine-tuning on a single randomly sampled example" (P1 Table 9 caption), trained "to convergence" (App. D). Footnote 5 says that for a single example convergence "usually occurs after a single training step"; that remark is made in the context of pre-elicitation.
- **"10 Params" / "100 Params"**: "Training only 10 randomly sampled parameters on the full dataset yields comparable gains" (App. I.6). These are presumably the "sparse subsets … within rank 1" of §3.3.
- **Not stated:**
  - which example was drawn, and how many draws or seeds were averaged;
  - learning rates and step counts for these runs;
  - which test set "Reasoning" means (§3.2 lists MATH500, AIME-24, GPQA-Diamond, word games and crosswords as sources of R1 examples);
  - whether the Qwen single-example run used full fine-tuning or LoRA (Table 9 covers only the Llama and TinyStories tasks);
  - what the PGR denominator is for Qwen.

**The metric.**
- P1 Eq. (4): PGR = (Perf_LoRA − Perf_base)/(Perf_FullFT − Perf_base), "where performance (Perf) is measured by accuracy or loss improvement relative to the model's zero-shot performance. PGR = 1 indicates LoRA matches full fine-tuning; PGR = 0 indicates no improvement over the base rate (zero-shot)" (§3.4).
- Table 8 uses accuracy.
- The capacity analysis uses the EDL ratio (EDL_EDL) or PGR (EDL_PGR). Table 6 says "The close correspondence between these thresholds validates that EDL-based capacity limits predict performance-based capacity limits".

**What the zero-shot baseline means.**
- Llama zero-shot is 0% because of format, not capability. App. I.7: "this reflects the models' inability to follow instructions and produce correctly formatted responses, not an absence of the underlying capability."
- The "bits" claims are prior estimates, not measured EDL. App. I.6: "A single arithmetic example … contains approximately 3–7 bits of answer information (log2 of the answer space)". §6.1: "(~8 bits)". §7.2: "fewer than 10 bits".

**Comparison with instruction-tuned models (P1 Table 10).**
- On arithmetic, base models after 1 example beat their instruction-tuned versions zero-shot, at n\* = 1. Example: 8B add/sub IT 56.1 vs base 1-example 96.0.
- On BoolQ and ARC they do not always. Example: 1B BoolQ IT 71.5 vs 64.9, with n\* = 8.
- §4.1 still says "consistently outperform", which holds only for arithmetic.

**P3's version of "minimal".**
- The parameter budget runs from "a single parameter to the full set of LoRA adapter weights for ranks 1–1024", plus full fine-tuning (§4.1).
- Gap closure (P3 App. E): GC(p) = (f(p) − Acc0)/(Acc_full − Acc0). §4.1 defines the gap ceiling as "the full fine-tuned model's performance (or maximum of the logistic fit, whichever is higher)".
- P3 trains on full datasets; it does not vary dataset size.
- Numbers are under Q7(e).

**P2's version of "minimal".** The "Hypothesis collapse" toy model (§5.2, §6.2, App. B.2): a diagnostic example can resolve log m bits of hypothesis uncertainty while contributing only log|Y| bits of codelength.

---

## Q5. Criteria for elicitation versus teaching, and stated limits on transfer

**Criteria**
1. **EDL scaling with n.** P1 App. J.1: elicitation dominates when ∂/∂n [EDL(n)/n] < 0; teaching dominates when it is > 0. §4: "these signatures characterize which process dominates, not a strict dichotomy".
   - In P1, the EDL/D curves start at n ≈ 1,000 for arithmetic (Fig. 2, Fig. 4 insets) and at n = 10–1,000 for Qwen 14B (Fig. 5 inset).
   - **No scaling signature is measured for n ≤ 64 on any model other than Qwen 14B** (whose smallest n is 10).
2. **Capacity threshold.** P\* ≈ 0.01–0.1 bits/param means elicitation; P\* ≈ 1+ means teaching (J.2). Pre-teaching shifts the threshold by more than 10× (§5.1).
3. **Independent behavioural checks.**
   - P1 App. D: "We regard a model as capable of being elicited on a particular task if multi-shot prompting or removing biases in the model's logit distribution (for multiple choice tasks) results in significantly improved performance relative to the model's original zero-shot baseline."
   - P1 Table 11 gives 0- and 16-shot results. App. I.9 notes the "consistency between these two independent methods".
4. **Single-example gain.** P1 Table 9 caption: "Large ∆ with a single example indicates elicitation of latent capability; minimal ∆ indicates the capability must be taught."
5. **P3: compression and parameters.**
   - Table 2: "L_k ≪ L_0 (for k small)" means "Latent, present after training". "L_k ∼ L_0 (for k small)" means "**Absent or already elicited**", so the case is ambiguous.
   - §5.1 and Fig. 5: accuracy and MDL compression track each other when the capability is latent, and diverge (an order of magnitude apart) when it is absent.
   - App. I: "when accuracy improves without commensurate MDL compression, the model may be learning rather than eliciting."
   - §4.5: comparison with a randomly initialized model of the same architecture.
6. **P2 §7.1:** elicitation means "EDL per example monotonically decreases … total EDL is small relative to capability improvement … bits-per-parameter threshold … is low."

**Stated limits on transferring thresholds**
- **Tasks.** P1 §7.4: "Our experiments focus primarily on arithmetic and reasoning tasks for which high-quality training corpora are widely available and performance can be scored against verifiable ground truth solutions; other capabilities may behave differently." P3 §6.3: "none of them have the diversity and complexity of production settings."
- **Models and architecture.** P1 §7.4: "We study primarily Llama and Qwen families of dense transformer models; other architectures, such as Mixture-of-Experts (MoE), may have different capacity characteristics." All starting models are **pretrained-only base models**. P3 §3.1: "The models used in our experiments are all pretrained base models, unless otherwise noted." In P1 the instruction-tuned models appear only as zero-shot comparisons. **No threshold is measured from a post-trained or reasoning model as the starting point.** P1 Fig. 7 predicts "Elicitation (Demonstrated)" gives "EDL very low" and "P∗ ≲ 0.01 bits".
- **Model size.** Thresholds depend on size: "larger models have lower capacity thresholds (when architecture/pretraining is fixed)" (P1 §6.1). For Qwen the size trend is not monotonic (Table 7).
- **Algorithm and hyperparameters.**
  - P1 §7.4: "As EDL depends on the training algorithm, different training techniques, optimizers, learning rates, or other hyperparameters could yield different values".
  - P2 Prop. 4.9 and §4.5: "the algorithm should be held fixed or explicitly varied as an experimental factor."
  - P1 App. H.1.2: other configurations give "same-order-of-magnitude capacity limits" only "after appropriate renormalization", and those results are omitted.
- **Training method and RL.**
  - P2 §7.2: "our analysis focuses on supervised fine-tuning. Extensions to reinforcement learning, preference optimization, and other post-training methods require additional development."
  - P1 §7.3 speculates: "This suggests that RLHF, while computationally expensive, may be largely elicitation". This is an interpretation, not a measurement.
  - P1 §7.5 cites single-example RL and spurious rewards as consistent with the results, without testing them.
- **Data scale.** Thresholds come only from datasets of at least ~10K examples or from s1K (P1 App. I.2). P3 §6.3: "Finite dataset size … might result in our result overestimating gap closure for smaller k."
- **Finite compute.** P1 App. J.4: "with finite compute budgets, this relationship can reverse"; a low P\* can reflect "inefficient learning that stalls".
- **Meaning.** P1 §7.4: "EDL measures learning as data compression achieved through generalization, not semantic content or performance." P2 §6.4: "These thresholds are empirical observations; the theoretical basis for such thresholds, and their precise values, remain open questions".

---

## Q6. Which results have error bars or seeds

**P1**
- Table 1 gives 3 seeds per configuration.
- No error bars, confidence intervals or standard deviations appear in any table (Tables 6–11) or figure. I checked Figs. 2–5 visually: they are scatter points with no uncertainty.
- The aggregation over seeds is never stated.
- The few-shot baselines (Table 11) are single numbers.

**P2**
- Theory only. The toy figures show E[EDL].
- P2 recommends reporting confidence intervals (§6.1) and gives Var[EDL] = O(n) (App. C.1).

**P3**
- **Figs. 1, 4, 5:** "bootstrapped 95% confidence intervals" on the **logistic fits**. The bootstrap (App. E) is: "For each iteration, remove one frontier point uniformly at random; Fit logistic curve to remaining points", with 10^4–10^5 samples. This captures fit sensitivity, not seed or test-sampling variance.
- **Fig. 3 (reasoning):** "Error bars indicate standard errors over 16 samples", meaning sampling at temperature 0.6, not training seeds.
- **Fig. 10 (BoolQ):** "2σ standard errors are shown as shaded regions".
- **Figs. 4, 11, 12:** all individual seed and hyperparameter points are shown.
- **Seeds:** "S = 3 – 5 independent random seeds" per budget, each with "up to 50 hyperparameter trials" (App. E).
- **8-shot baseline:** 30 trials; the 97.5th percentile is reported as an upper bound (App. A).
- **Classifier head:** best of 5 seeds (App. A.1).
- **Table 1 (parameters needed for gap closure):** point values only.
- **How the frontier is defined is inconsistent.**
  - §4.1: "the average accuracy achieved across all seeds for the best set of hyperparameters".
  - App. E, Eq. (3): f(p) = max_{s,h} Acc(θ_{p,s,h}), on **validation** accuracy.
  - Fig. 1's axis is labelled "Test Accuracy".
- **Selection in Bayesian optimization (fewer than 20 parameters).** Candidates are evaluated "on the entire train and validation sets", and the final reported performance is the "candidate vector with highest validation accuracy" out of 3·k initial random vectors (k = number of trainable weights) plus up to 400 GP-proposed vectors (App. C.4).
- **[inference]** Both selection procedures give upward-biased frontier values at small k, where variance across configurations is largest.

---

## Q7. Points bearing on the design

**(a) Training on the model's own verifier-selected samples.**
- Not tested in any of the three papers.
- The closest cases are:
  - P1: a base model trained on a *stronger* model's (R1) correctness-filtered traces.
  - P3: a base model trained on its post-trained sibling's unfiltered traces.
- P2 §5.1 and Prop. 5.1: random labels give E[EDL] = 0. EDL measures generalizable structure, not training effort.
- **[inference]** With self-samples, the first-pass codelength is dominated by the model's own sampling entropy. In principle that entropy cancels in EDL, because it is irreducible and reappears in n · L_test. But it adds variance proportional to trace length (P2 App. C.1).
- **[inference]** The information a verifier filter adds per prompt is KL(filtered ‖ base) = −log2 P(accept), i.e. log2(1/pass rate) bits for an exact rejection filter. This is a cheap analytic yardstick to put next to any measured EDL. A measured EDL far above Σ log2(1/pass_i) would mean the extra bits come from sampling noise or from fitting the particular traces.
- For EDL to be valid, the test set must come from the same distribution as the training targets (P2 §3.2; P1 Eq. 2, "held-out data from the same distribution"). For verifier-selected self-samples, that means verifier-selected self-samples on held-out prompts. Ground-truth or other-model traces would not do.

**(b) Targets with long thinking segments.**
- P1 scores the full chain of thought plus the answer for Qwen (App. F). EDL/D for Qwen 14B is about 0.03–0.15 bits/token over n = 10–1,000 (Fig. 5 inset).
- P2 §5.5 and App. B.5 warn that format and style tokens create "early transients" and recommend scoring capability tokens or teaching the format first.
- P3: s1K traces run up to 32,768 tokens. Its "loss improvement" on AMC24 is computed "against reference solutions generated by DeepSeek-R1-Distill-Qwen-1.5B" (§4.2).
- **[inference]** For n = 1, EDL is bounded by the codelength of one long trace, which can be thousands of bits, mostly trace-specific. Because L_test is per example, a mismatch between train and test lengths can push EDL positive or negative by large amounts. Per-token normalization, matched length distributions, or answer-only masks need to be decided before running.

**(c) Single pass versus multiple epochs.**
- EDL is designed for multi-epoch training. MDL comes from epoch 1; θ\* comes after multi-epoch training with validation early stopping (P1 §2.5).
- P1 §7.5: EDL "accounts for multi-epoch training (standard in fine-tuning but unaddressed by single-epoch MDL)". The authors also report that "repeating our analyses with MDL instead of EDL failed to reliably distinguish elicitation from teaching".
- P1 App. H.1.2: "Larger datasets and multi-epoch training both smooth out these differences" (the hyperparameter differences); "If a small dataset is sufficient to elicit latent capability, multi-epoch training should find the minimum regardless of hyperparameters".
- Overfitting caveat: P2 §4.1 says training "well past the optimal early stopping point" breaks non-negativity.
- P3 training lengths:
  - multiple choice: until 20 epochs without validation-accuracy improvement (App. A, C.2);
  - generative: 2 epochs without validation-loss improvement;
  - Alpaca: "exactly two passes", which contradicts "regardless of the number of epochs required" (§3.1);
  - fewer than 20 parameters: "can require more than 1000 training epochs" (App. C.4).
- P3 §5: "in cases where the pointer to the underlying capability requires many epochs of training to find or generate, MDL compression will be low even if downstream task accuracy is high at the end of multi-epoch training".

**(d) n = 1–64 examples.**
- P1 single-example results: Tables 8, 9 and 10. Table 10 gives n\* (examples needed for the base model to beat the instruction-tuned model) of 1–8.
- Few-shot baselines: P1 Table 11 at 16 shots; P3 at 1–32 shots, where "8 shots resulted in the highest performance" (App. A). In P3 footnote 5 (Lichess), both models stay below 0.01% "with up to 64 in-context examples".
- The P1 EDL scaling curves do not reach n < 1,000, except Qwen 14B at n ≥ 10 (Fig. 5).
- P3 does not vary dataset size.
- With a batch size of 8 (P1 Qwen), n = 10 gives only about 2 online updates, so MDL is mostly zero-shot loss ([inference]; see Q1).

**(e) Random-parameter elicitation (P3).**
- **Which parameters.** Scalar entries of the LoRA A and B matrices, drawn uniformly over all LoRA modules in all layers. By default every projection matrix is adapted. Attention-only adaptation (K, Q, V, O) is "consistently the most parameter-efficient" (App. C.1, Fig. 8). Unselected entries are frozen at their initial values (A Kaiming uniform, B zero), and the mask is fixed for the run. See the dead-A-entry inference under Q2.
- **How they are trained.**
  - AdamW with per-budget Bayesian-Hyperband search over learning rate, batch size and weight decay, 5 seeds × 50 trials, trained to convergence.
  - Below about 20–25 parameters, also Bayesian optimization (Gaussian process with Matérn kernel and expected improvement) directly on the weight values, within bounds [−10, 10] down to [−1, 1] depending on k (App. C.4).
- **Fraction of the gap at about 10–100 parameters.**
  - P3 Table 1, 50% gap closure:
    - GSM-8K-CoT-Choice (binary; zero-shot 47.6–51.3%, i.e. chance): 39, 33 and 77 parameters (1B, 3B, 8B).
    - ARC-Easy: 216, 774 and 99.
    - ARC-Challenge: 3,023, 236 and 98.
  - BoolQ: 74 parameters (1B) and 147 (3B) (Fig. 10).
  - The abstract says: "10–100 randomly chosen parameters … can recover up to 50%".
- **At about 1,000 parameters.**
  - 90% closure on GSM-8K-CoT-Choice needs 1,036, 2,752 and 2,403 parameters (Table 1). Fig. 1 instead gives 3,048 for 8B at 90%; the 2,403 there is 1B at 95%. So Table 1 and Fig. 1 disagree; the tiny-adapter reader spotted this and I missed it in phase 1. On ARC-Challenge 8B it needs 415.
  - Alpaca reaches 50% "with ~1,000 learnable parameters (all models)" (§4.2). However, App. B.1 and Fig. 7 show 1B's win rate going from "~0.08 to ~0.71 as the budget grows from 10^3 to 10^5".
- **At thousands to tens of thousands of parameters.**
  - 95% closure on GSM-8K-CoT-Choice: 2,403, 12,838 and 9,589 (Fig. 1).
  - ARC-Easy 3B and 8B need 159k and 318k parameters for 90%.
- **Reasoning (the case closest to ours).** Qwen2.5-1.5B (App. C.1 says Qwen-2.5-Math-1.5B) trained on 1,000 s1K-Qwen-1.5B traces: "~10,000-20,000 trainable parameters are sufficient to achieve 50% gap closure" on AMC24 for maj@16 and for loss improvement (§4.2). In Fig. 3 (visual reading), AMC24, AIME24 and AIME25 accuracy is about 0 at 10–100 parameters. **The 10–100 parameter headline does not hold for reasoning.**
- **Teaching controls.** On TinyStories, a randomly initialized model needs "more than seven orders of magnitude more parameters". On Lichess both pretrained and random-init models are in the teaching regime: the gap is "less than an order of magnitude" and more than 100,000 parameters are needed (§4.5).
- **How it is measured.**
  - Multiple choice: accuracy on held-out examples *of the same task*, selected on validation as above. These are held-out items, not held-out tasks.
  - Reasoning: different problem sets from the s1K training prompts (AMC24, AIME24, AIME25; AMC24 and AIME25 were "released after the release of Qwen2.5"). pass@1 and maj@16 come from 16 samples at temperature 0.6.
  - Alpaca: win rate on the 805-prompt AlpacaEval set, judged by a preference model.
- **P1's random parameters.** 10 and 100 random parameters "on the full dataset" (Table 8), evaluated on held-out examples of the same task.

---

## Q8. What I did not read, and gaps

- **Text:** I read all three PDFs completely (P3 including its NeurIPS checklist). I extracted text with PyMuPDF.
- **Figures:** figure contents carried only as images were checked visually for P1 pp. 4, 6 and 7 (Figs. 2–5) and P3 pp. 3, 6, 7, 9, 15, 24 and 25 (Figs. 1–5, 7, 11, 12). Other figures were read from extracted axis text and captions only: P1 Figs. 1, 6–10, P2 Figs. 1–8, P3 Figs. 6, 8–10.
- **Not obtained or not read:**
  - the OpenReview version of P3 (HTTP 403 challenge);
  - any NeurIPS supplementary archive for P3 (I did not look for one beyond the main PDF, which already contains appendices A–J);
  - an arXiv version of P3 (none found);
  - the P1 PMLR HTML page (I read the PDF);
  - P3's code (the repository is empty);
  - Donoway (2026, ISIT), the companion EDL paper cited by P1;
  - cited works such as Perez et al. (Rissanen Data Analysis) and Whitney et al. (surplus description length);
  - the s1K, s1K-1.1 and R1 trace datasets.
- **Inconsistencies spotted (for phase 2):**
  - P3: the frontier is a mean over seeds (§4.1) but a max over seeds (App. E).
  - P3 Table 1: ARC-Easy 8B at 90% lists "318k" parameters but "3.96 × 10^−1" %Params.
  - P3 App. B.1: Alpaca split given as "234,006"/"26,006" from "52k examples".
  - P3: Alpaca judge and opponent given as "Anthropic's internal preference model" vs. the "best full fine-tuned model" (§3.2), but as an "off-the-shelf preference model" vs. the "un-adapted base model" (B.1).
  - P3: base model given as Qwen2.5-1.5B (Fig. 3) and as Qwen-2.5-Math-1.5B (App. C.1).
  - P3: the Fig. 12 caption says 8B ARC-Challenge, but its panels repeat Fig. 11.
  - P1 App. G.2: "Data-Controlled Training" is an empty heading.
  - P1: LoRA scaling given as "α/2r".
  - P2: single-example EDL given as min(log|Y|, log m) in Prop. B.2 but as log m in Eq. (69).
  - P2 Fig. 8: the caption's column order does not match the plotted panels.
  - P2 §7.1.1 says "Qwen2.5 1.5B–14B"; P1 uses 1.5B–32B.

---

## Comparison with the readers' notes (phase 2)

Written after phase 1, once I was allowed to open the readers' notes. I compared `edl_papers.md` (Sonnet reader of P1 and P2) and `tiny_adapter_papers.md` (Sonnet reader of P3, TinyLoRA and the two Sinii papers) against my own reading above and against the PDFs. Both notes are careful. The large majority of their factual statements match the papers and my reading:
- EDL definition and first-epoch MDL;
- the full chain-of-thought mask for the Qwen runs;
- the 0.95 rule and the Table 6–8 values;
- no error bars in P1;
- SFT only, with no own-sample targets;
- the P3 mechanism, the max-over-configurations frontier, and reasoning at 10^4–2×10^4 parameters.

Below I list only where they depart from the papers or from my reading.

### `edl_papers.md` (P1, P2)

1. **Overstated: "by eye".** Q2 says "The dashed lines in Figs. 3-5 are marked by eye", and the Qwen paragraph says "The values are read off figures by eye, so 14B vs 1.5B is within plausible reading error". P1 gives Table 7 values and never says how crossings were located. "By eye" is the reader's guess and should read "method not stated". The non-monotone Qwen order (0.04, 0.05, 0.02) is what the table prints.
2. **Wrong: "not monotone in parameter count".** Q4 says the Qwen Table 8 values are "not monotone in parameter count (e.g. 32B: 0.27 at 10 parameters, 0.41 at 100, 0.55 at 1 example)", and item 8 repeats this. From 10 to 100 parameters PGR rises for all three Qwen sizes (0.14→0.61, 0.38→0.55, 0.27→0.41). The "1 Example" column is a different intervention (full fine-tuning on one example, Table 9) on a different axis.
3. **Not stated in the paper: "one random draw".** Q4 says "The example is one random draw". P1 says only "a single randomly sampled example" (Table 9) and "Seeds per config 3" (Table 1). How many draws or seeds sit behind the cell is unknown.
4. **Missing points.**
   - P2's internal contradiction on single-example EDL (Prop. B.2's min(log|Y|, log m) vs Eq. 69's log m). The note quotes only the bound.
   - With batch size ≥ n, MDL is just the zero-shot loss.
   - L_test is per example, so it is sensitive to length differences between train and test.
   - P1 Fig. 7's "Elicitation (Demonstrated)" column, with P\* ≲ 0.01 and very low EDL.
   - Every starting model in P1 and P3 is a pretrained-only base (P3 §3.1). This scope point is more specific than the note's "nothing about OLMo, thinking-trained models".
5. **Incomplete: "whole-response EDL is in line with BTC".** The note's corrections item 1 says this, and that answer masking "would be a departure". That is right about the convention (P1 App. F). But P1's whole-response targets were another model's R1 traces. For the model's own samples, whole-response scoring puts the model's sampling entropy into both MDL and n·L_test. It cancels only in expectation, adds variance in proportion to length (P2 App. C.1), and P2 §5.5 recommends capability-token scoring. "In line" holds for the convention, not for this bias.
6. **Incomplete: the learning-rate protocol.** Corrections item 5 says "One LR per model across ranks" and that "a per-arm grid is a sensible departure but is not the paper's protocol". That is true of P1. P3, the only paper that describes the sparse subsets, searches learning rate per budget over [1e-6, 1] with up to 50 trials × 5 seeds (App. C.2). A per-arm grid follows P3; a 3-point grid is far thinner than P3's search.
7. **Partly right: "first-epoch loss already low".** Q7(a) says that with own-sample targets "the first-epoch loss is already low". It is lower than on another model's traces. In absolute terms it is the model's own per-token entropy on its samples, which is not small for thinking text. The real issue is that most of it is irreducible noise, not that it is low.
8. **Correct, and less complete than the tiny-adapter note.** Q2 says that with B = 0 and only A trainable, gradients vanish, "so the subsets must include B entries". Correct. The fuller point is in the tiny-adapter note and in my Q2: even then, selected A entries in modules with no selected B entry stay dead.

### `tiny_adapter_papers.md`: P3 section

9. **Caught something I missed.** Fig. 1 disagrees with Table 1 for GSM8K 8B at 90% (3,048 vs 2,403). I checked the rendered figure: the bars are grouped by threshold, 90% = 1,036 / 2,752 / 3,048. I corrected my Q7(e) above.
10. **Missing: §4.1's competing frontier definition.** §4.1 defines the frontier as "the average accuracy achieved across all seeds for the best set of hyperparameters". So the note's "not a mean over seeds" is true of App. E only; the paper contradicts itself.
11. **Overstated: the Fig. 11 caption as a bound.** The note says Fig. 11's caption ("nearly all … within 2 percentage points") "bounds but does not remove" the selection effect. The rendered Fig. 11 (and Fig. 12) shows many runs far below the frontier: for 1B ARC-Easy, points at 0.25–0.5 accuracy from 10^3 to 10^6 parameters, against a frontier near 0.7. The caption is contradicted by its own figure and bounds nothing.
12. **Missing.** The Alpaca opponent and judge inconsistency: §3.2 says the "best full fine-tuned model" judged by "Anthropic's internal preference model"; App. B.1 says the "un-adapted base model" judged by an "off-the-shelf preference model".
13. **Overstated: "bigger models need fewer parameters".** "Transfers" item 3 says "All of Donoway (Table 1, Fig. 1), TinyLoRA ... report that bigger models need fewer parameters". For P3 that holds for the ARC tasks, not for GSM-8K-CoT-Choice: 50% at 39 / 33 / 77 and 90% at 1,036 / 2,752 / 3,048 (Fig. 1) for 1B / 3B / 8B, so 8B needs more than 1B. P1's Qwen thresholds are also non-monotone. The size argument cannot be cited as favouring a 32B model in general.
14. **Contestable: the RL checkpoint as a "cleaner" ceiling.** "Transfers" item 2 calls `s2300` "a different (and cleaner) ceiling than their full SFT". In P1 and P3 the ceiling is full-capacity training on *the same data*, which separates a parameter limit from a data limit. `s2300` as the ceiling mixes both. We need both ceilings: A16 on the same n as the capacity ceiling, and `s2300` as the behaviour target.

### `tiny_adapter_papers.md`: TinyLoRA and Sinii sections

These are spot-checks against `v2/papers/*/…txt`, not a full reading.

15. **Spot-check, TinyLoRA: numbers confirmed.**
    - "83% at 13 parameters and 84% at 120" (§6.2).
    - Learning-rate grid {1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 1e-4, 2e-4}, best per size, averaged over 3 seeds (§5.1).
    - "SFT works best with larger update sizes of at least 1M" (Fig. 2 caption).
    - The "76%" baseline mismatch: Table 2 has Qwen2.5-3B-Instruct GSM8K at 76.0 and 7B-Instruct at 88.2, while the text attributes 76% to 7B.
16. **Spot-check, TinyLoRA: missed by the note.** TinyLoRA §3's information argument *defines* the SFT data as the model's own reward-filtered samples: "Y ≡ πθ(X) | r(X, πθ(X)) = 1". That is exactly our source (i). Their hypothesis, that SFT must absorb all of y's bits and so is weak at small P, is therefore stated for our condition, though their experiment used reference answers (§6.2: "SFT training directly on answers is a version of off-policy learning"). The note's "they never compare SFT on the model's own verified samples" is right about the experiments only.
17. **Spot-check, Sinii 2505.18706 Table 6: learning-rate ratios.**
    - Steering-vector rates are 25×–200× the full-fine-tune rate, as the note's §4 says.
    - "25x-1000x" (two later lines) is overstated. The largest ratio in the table is 600×: Llama-3.1-8B-Instruct LoRA-1 at 6e-4 vs full fine-tuning at 1e-6.
    - The LoRA-1 rates (1e-4 to 6e-4) are RLOO rates, with α not stated for rank 1, so they do not transfer directly to our SFT grids.
18. **Spot-check, Sinii 2509.06608: confirmed.** τ = 1 base 14.3 ± 1.7, full RL 43.8, all-layer steering 42.9. The "To" prefix gives "~10-11 points – about three quarters" of the s27 gain.

### Effect on the design

These points feed my Turn 2. Item 16 and item 13 weaken the case that small P will suffice for own-sample SFT at 32B. Item 14 argues for keeping an A16 capacity ceiling on the same n. Items 5 and 7 bear on whether bits per weight can be reported for source (i).
