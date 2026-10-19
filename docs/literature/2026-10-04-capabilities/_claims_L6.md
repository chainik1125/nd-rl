# Claim ledger — reader L6 (representational / mechanistic definitions of latent capability; novelty in games)

Written 2026-10-05 by agent:claude. Every V claim was found with `capability_defs/lit/quote.py` in the fetched
text (`~/cd_sources/<id or slug>.txt`); locations are section / equation / table / figure / page as read in the
rendering named in the note or the screen table (HTML renderings have no page numbers). Quotes are verbatim up
to whitespace, case and math-markup normalisation. UNVERIFIED rows say why.

| # | paper id | claim (short verbatim quote or exact number) | location | status V/UNVERIFIED |
|---|---|---|---|---|
| 1 | 1909.03368v1 | "control tasks, which associate word types with random outputs" | Abstract | V |
| 2 | 1909.03368v1 | "By construction, these tasks can only be learned by the probe itself." | Abstract | V |
| 3 | 1909.03368v1 | a good probe "should be selective, achieving high linguistic task accuracy and low control task accuracy" | Abstract; Sec. 6 | V |
| 4 | 1909.03368v1 | "as long as a representation is a lossless encoding, a sufficiently expressive probe with enough training data can learn any task on top of it" | Sec. 1 | V |
| 5 | 1909.03368v1 | "Selectivity is defined as the difference between linguistic task accuracy and control task accuracy" | Fig. 2 caption | V |
| 6 | 1909.03368v1 | MLP probe, PoS: "97.3 accuracy is achieved, compared to 92.8 control task accuracy, resulting in 4.5 selectivity" | Sec. 1 (finding 1); Table 1 | V |
| 7 | 1909.03368v1 | linear probe, PoS: "97.2 accuracy, and 71.2 control task accuracy, for 26.0 selectivity" | Sec. 1 (finding 2); Table 1 | V |
| 8 | 1909.03368v1 | control-task "ceiling on performance is the fraction of tokens in the evaluation set whose types occur in the training set" | Sec. 2.3 | V |
| 9 | 1909.03368v1 | control labels sampled from the empirical tag distribution "so the marginal probability of each label is similar" | Sec. 2.1, footnote 2 | V |
| 10 | 1909.03368v1 | restricting probe training data rests on "the intuition that general rules can be learned more sample-efficiently than memorization" | Sec. 3.2 | V |
| 11 | 1909.03368v1 | "learning each linguistic task requires fewer samples than our control task" (PoS; not dependency edges) | Sec. 3.5 | V |
| 12 | 1909.03368v1 | regularising to reduce the train-test gap "is insufficient if one is interested in selectivity" | Sec. 3.2 | V |
| 13 | 1909.03368v1 | default probes "are over-parameterized and needlessly high-capacity" | Sec. 3.6 | V |
| 14 | 1909.03368v1 | "the linear probe on ELMo2 achieves selectivity of 31.4, compared to selectivity of 26.0 for ELMo1" (accuracy 96.6 vs 97.2) | Sec. 4.2; Table 2 | V |
| 15 | 1909.03368v1 | without selectivity "it might be thought that ELMo2 encodes nothing about part-of-speech, since it doesn't beat the Proj0 random representation baseline" | Sec. 4.2 | V |
| 16 | 1909.03368v1 | Proj0 = "untrained BiLSTM run on the non-contextual character CNN word embeddings of ELMo"; linear PoS 96.3 accuracy / 20.6 selectivity | Sec. 4.1; Table 2 | V |
| 17 | 1909.03368v1 | ELMo2 probes "must rely on emergent properties of the representation" | Sec. 4.2 | V |
| 18 | 1909.03368v1 | "randomness is applied at the type-level rather than at the example-level" | Sec. 5.1 | V |
| 19 | 1909.03368v1 | "we suggest that probes be thought of as craftspeople" | Sec. 6 | V |
| 20 | 1909.03368v1 | text says "the bilinear probe achieves 16.7 selectivity", but Table 1 gives 6.6 for the default bilinear probe and 16.7 with 0.4 dropout (internal inconsistency) | Sec. 3.5 vs Table 1 | V |
| 21 | 1909.03368v1 | "part-of-speech probes designed with control tasks all use rank-10 weight matrices" | Table 1 caption | V |
| 22 | 2311.12786v2 | "does fine-tuning yield entirely novel capabilities or does it just modulate existing ones?" | Abstract | V |
| 23 | 2311.12786v2 | "(i) fine-tuning rarely alters the underlying model capabilities"; a 'wrapper' "is typically learned on top of the underlying model capabilities, creating the illusion that they have been modified" | Abstract | V |
| 24 | 2311.12786v2 | revival: "the model begins reusing these capabilities after only a few gradient steps" | Abstract | V |
| 25 | 2311.12786v2 | Def. 1: M "possesses a capability C" if for all x ∈ X_C there is a layer l ≤ L with Read_l(M(x)) = f_C(x_d) (Read_l = linear layer trained on layer-l outputs using D_PT) | Sec. 3, Definition 1 | V |
| 26 | 2311.12786v2 | "A linear readout at an intermediate layer is used in the definition above to emphasize that the notion of a capability need not correspond to only input-output behavior" | Sec. 3 | V |
| 27 | 2311.12786v2 | "Such structured failures imply claiming the existence of a capability should account for the input domain" | Sec. 3 | V |
| 28 | 2311.12786v2 | Def. 2: if g∘f_C is correct on all of P^FT the capability "is strongly relevant to the fine-tuning task; else, we call it weakly relevant" | Sec. 3, Definition 2 | V |
| 29 | 2311.12786v2 | "if the behavior corresponding to a pretraining capability is retrieved in a few steps of reFT, fine-tuning did not meaningfully alter said capability" | Sec. 5 | V |
| 30 | 2311.12786v2 | pruning: "the pretraining task's performance improves after just 5–15 neurons are pruned" | Fig. 7 caption | V |
| 31 | 2311.12786v2 | reFT revival in 0.1–1K (strong relevance) or 3K iterations (weak); "the Scr.+FT baseline only reaches perfect accuracy at 4.5K iterations and when using a larger learning rate" | Fig. 9 caption; Fig. 26 caption | V |
| 32 | 2311.12786v2 | TinyStories: "Fine-tuned models' loss goes down very quickly (30–300 iterations) compared to baselines (which never reach the same loss" | Fig. 11 caption | V |
| 33 | 2311.12786v2 | Twist proportion during reFT, iterations 0/30/300/3000: Filtering (η_M) 44/81/81/82 %; "Not in PT" control 12/31/44/81 %; "models relearn to generate stories with twist more sample-efficiently than the control model" | Table 1 (cells and caption) | V |
| 34 | 2311.12786v2 | "a wrapper, i.e., a localized transformation of the pretraining capability, is learned during fine-tuning" | Sec. 5.1 | V |
| 35 | 2311.12786v2 | jailbreak setup: with the non-jailbreak token "the model does encode the count of a in the outputs around the middle layers" | App. E.2 | V |
| 36 | 2311.12786v2 | "Probing is used to understand if a particular capability is present in the model" (linear probe on each block's residual output at the answer token) | App. D | V |
| 37 | 2311.12786v2 | reFT uses a smaller lr than FT: if revived "it is stronger evidence that the pretraining capability was never forgotten or removed" | App. D | V |
| 38 | 2311.12786v2 | reFT models "converge to a lower loss on this dataset than the control pre-trained model" | App. F.3 | V |
| 39 | 2311.12786v2 | TinyStories models: "We pretrain 91 million parameter autoregressive language models" | App. F.1 | V |
| 40 | 2311.12786v2 | probing: "a small amount of information about story classification has been removed from the activations of the fine-tuned models" | App. F.3 | V |
| 41 | 2311.12786v2 | fine-tuning studied = continued training at lr "one to three orders of magnitude lower than the average pretraining one" | Sec. 2 | V |
| 42 | 2311.12786v2 | "Note that the performance is not high to begin with, indicating the ability to count was learned during fine-tuning" | Sec. 5.1 | V |
| 43 | 2311.12786v2 | pruning picks "the weights/neurons with largest dot product between their gradient and weights" (single step, gradient of the loss of the capability to revive) | App. D | V |
| 44 | 2402.14811v1 | "in both the original model and its fine-tuned versions primarily the same circuit implements entity tracking" | Abstract | V |
| 45 | 2402.14811v1 | "the entity tracking circuit of the original model on the fine-tuned versions performs better than the full original model" (circuit 0.73 / 0.72 on Goat / FLoat vs full Llama-7B 0.66) | Abstract; Table 1 | V |
| 46 | 2402.14811v1 | "Performance boost in the fine-tuned models is primarily attributed to its improved ability to handle the augmented positional information" | Abstract | V |
| 47 | 2402.14811v1 | "fine-tuning enhances, rather than fundamentally alters, the mechanistic operation of the model" | Abstract | V |
| 48 | 2402.14811v1 | base circuit = "a sparse set of 72 attention heads in four groups"; in fine-tuned models it "alone can restore atleast 88% of the overall performance of the entire fine-tuned model" | Sec. 1 | V |
| 49 | 2402.14811v1 | Table 1: full-model / circuit / random-circuit accuracy, faithfulness: Llama-7B 0.66/0.66/0.00, 1.00; Vicuna-7B 0.67/0.65/0.00, 0.97; Goat-7B 0.82/0.73/0.01, 0.89; FLoat-7B 0.82/0.72/0.01, 0.88; "chance accuracy is 0.14" | Table 1 | V |
| 50 | 2402.14811v1 | null: "10 random circuits with the same total and per-position number of heads; random circuits have virtually zero accuracy" | Sec. 4.3 | V |
| 51 | 2402.14811v1 | faithfulness = "the percentage of model performance that can be recovered with the circuit, i.e. F(Cir)/F(M)" (non-circuit heads mean-ablated) | Sec. 4.2 | V |
| 52 | 2402.14811v1 | Goat / FLoat own circuits: "175 attenton heads and approximately forming a superset of the Llama-7B circuit"; "fine-tuning is inserting additional components to the circuitry that performs entity tracking" | Sec. 4.3 | V |
| 53 | 2402.14811v1 | CMAP: "patching activations of the same components of different models on the same input" | Sec. 6.1 | V |
| 54 | 2402.14811v1 | "the activations of fine-tuned models are compatible with base model, even though they could have been using completely different subspaces and/or norms to encode information" | Sec. 6.2 | V |
| 55 | 2402.14811v1 | patching Value Fetcher heads from the fine-tuned model into Llama-7B gives "the maximal increase in performance ... recovering the full fine-tuned models' performance" | Sec. 6.2; Fig. 4 | V |
| 56 | 2402.14811v1 | "neither additional functionality nor a shift in functionality is introduced in fine-tuned models" | Sec. 5.3 | V |
| 57 | 2402.14811v1 | "Goat-7B can achieve a performance improvement of 20% compared to Llama-7B" (Value Fetcher heads) | Sec. 5.3 | V |
| 58 | 2402.14811v1 | path patching "does not provide a clear threshold for the number of heads that should be included in the circuit" | Sec. 4.2 | V |
| 59 | 2402.14811v1 | "the Llama-7B minimal circuit is not perfectly complete" | App. C | V |
| 60 | 2402.14811v1 | "Understanding whether such mechanism invariance is typical will require experience with further tasks on more models" | Sec. 7 | V |
| 61 | 2402.14811v1 | "The mechanism invariance is observed in both low-rank adaptations (LoRA) ... and fully fine-tuned models" | Sec. 1 | V |
| 62 | 2507.12638v1 | backtracking in DeepSeek-R1-Distill-Llama-8B "is in part driven by a repurposed direction already present in base model activations" | Abstract | V |
| 63 | 2507.12638v1 | a base Llama-3.1-8B direction "which systematically induces backtracking when used to steer the distilled reasoning model" | Abstract | V |
| 64 | 2507.12638v1 | "this direction does not induce backtracking in the base model, suggesting that the reasoning finetuning process repurposes pre-existing representations to form new behavioral circuits" | Abstract | V |
| 65 | 2507.12638v1 | conclusion framed as repurposing "rather than learn new capabilities from scratch" | Abstract | V |
| 66 | 2507.12638v1 | "we derive steering vectors separately on residual stream activations from both the base and reasoning models on the same reasoning traces" | Sec. 3 | V |
| 67 | 2507.12638v1 | steering vector = difference of means: v = MeanAct(D+) − MeanAct(D) (D+ = GPT-4o-labelled backtracking positions, D = all) | App. A, Eq. 3 | V |
| 68 | 2507.12638v1 | best offset for the layer-10 residual stream "∼−13 to −8" tokens before the backtracking event | Sec. 3.1 | V |
| 69 | 2507.12638v1 | "base-derived steering vectors reliably induce backtracking when used to steer the reasoning model, and have comparable performance to their reasoning-derived counterparts" | Sec. 3.2; Fig. 3 | V |
| 70 | 2507.12638v1 | "neither base-derived nor reasoning-derived steering vectors invoke backtracking behavior in the base model"; "the base model never exhibits backtracking behavior, even when steered with the reasoning model-derived backtracking-inducing vector" | Sec. 3.2; Fig. 3 caption | V |
| 71 | 2507.12638v1 | base- and reasoning-derived vectors "have high cosine similarity of ∼0.74" | Sec. 3.2 | V |
| 72 | 2507.12638v1 | "the backtracking steering vector significantly outperforms all tested baselines" (mean, noise, self-amplification, deduction, initializing vectors) | Sec. 3.3; Fig. 4 | V |
| 73 | 2507.12638v1 | "adding Gaussian noise to activations has a nontrivial effect on the fraction of output words" that are "Wait" | Sec. 3.3 | V |
| 74 | 2507.12638v1 | "the base-derived steering vectors do not decode to backtracking keywords, yet are successful in eliciting backtracking in the fine-tuned model" | Sec. 4.1; Fig. 5 | V |
| 75 | 2507.12638v1 | "the base-derived steering direction is densely present in model activations" and does not cleanly correlate with backtracking as a probe | Sec. 4.2 | V |
| 76 | 2507.12638v1 | limitation: "instances both where backtracking occurs while the direction is not present, and instances where the direction is present but backtracking does not occur" | Sec. 5 | V |
| 77 | 2507.12638v1 | results are "an existence proof for latent reasoning-related representations in base models, rather than a comprehensive explanation of reasoning behavior"; "we examined a single reasoning model" | Sec. 5 | V |
| 78 | 2507.12638v1 | metric: fraction of words in B = {wait, hmm}; "∼83%" of keyword-flagged sentences are true backtracking by the authors' judgement | Sec. 2.2, Eq. 1; App. C | V |
| 79 | 2507.12638v1 | "base models may possess latent reasoning capabilities which are unexpressed until they are extracted by the finetuning process" | Sec. 1 | V |
| 80 | 2510.07364v4 | "What do thinking language models learn during training that their base models lack?" | Abstract | V |
| 81 | 2510.07364v4 | decomposition into "reasoning mechanisms (category vectors that can induce a reasoning behavior in the base model) and reasoning heuristics (a classifier determining when a mechanism should fire)" | Abstract | V |
| 82 | 2510.07364v4 | "hybrid models recover roughly 76% of the RL base-to-thinking gap but only 11% of the SFT gap" (nine pairs: four RL, four SFT-distilled, one mixed) | Abstract | V |
| 83 | 2510.07364v4 | "RL primarily teaches heuristics for orchestrating pre-existing base mechanisms, whereas SFT-distillation installs new ones" | Abstract | V |
| 84 | 2510.07364v4 | "Rec.% is the fraction of the base-to-thinking gap that the hybrid recovers"; "Base and hybrid models decode greedily; thinking models sample 3 rollouts at temperature 0.6" | Table 1 caption | V |
| 85 | 2510.07364v4 | Rec.% GSM8K / MATH500 / Hendrycks-MATH: ORZ-0.5B 83.0 / 95.3 / 107.4; R1-Distill-1.5B −3.0 / 5.7 / −8.5 | Table 1 | V |
| 86 | 2510.07364v4 | category-vector training: thinking-model rollouts teacher-forced "through the base model, identifying token positions where the two models disagree"; "Jointly optimize a per-category vector and a small MLP that predicts a steering coefficient, minimizing cross-entropy on the thinking model's next token" | Sec. 3.3 | V |
| 87 | 2510.07364v4 | heuristic: "First, we check whether the base- and the thinking model disagree in their next token prediction"; if they disagree the SAE's top category is steered | Sec. 3.5 | V |
| 88 | 2510.07364v4 | "with only 5-15 distinct category vectors applied sparsely, the hybrid cannot succeed by memorizing outputs" | Sec. 3.6 | V |
| 89 | 2510.07364v4 | RL pairs "steer only ∼5-12% of tokens per problem" | Sec. 3.7; Table 2 | V |
| 90 | 2510.07364v4 | ablations (ORZ-1.5B, ORZ-32B): full pipeline "∼77%"; random category "drops recovery to 20-28%"; norm-matched random vectors give −28 % and −13 % | Sec. 3.8; Fig. 4 | V |
| 91 | 2510.07364v4 | "the lower recovery for SFT-distilled models could reflect either genuine mechanism modification or limitations in our category vector optimization" | Sec. 5 | V |
| 92 | 2510.07364v4 | "RL training optimizes the model's own outputs against a reward signal, encouraging it to leverage what it already knows" | Sec. 3.7 | V |
| 93 | 2510.07364v4 | ORZ (RL) pairs "converge to substantially lower cross-entropy than the SFT-distilled R1 models" in category-vector training | Sec. 3.4; Fig. 3 | V |
| 94 | 2510.07364v4 | vectors at "approximately 37% of model depth"; V ∈ R^{K×d}, coefficient MLP "Linear (d,512)" → GELU → K heads (Softplus) on the base residual | Sec. 3.3; App. E.2 | V |
| 95 | 2510.07364v4 | QwQ-32B (SFT then RL) "recovers ∼20%" | Sec. 3.7 | V |
| 96 | 2510.07364v4 | crosscoder-style diffing "assume[s] the diff can be explained at the level of linear features" ("they assume the diff can be explained at the level of linear features") | Sec. 3.1 | V |
| 97 | 2510.07364v4 | training mix "9,794 questions" (MATH, Natural Reasoning, SciBench, TheoremQA) | App. E.1 | V |
| 98 | 2303.07462v2 | "analyzing more than 5.8 million move decisions made by professional Go players over the past 71 years (1950-2021)" | Abstract (p. 2) | V |
| 99 | 2303.07462v2 | "generating 58 billion counterfactual game patterns and comparing the win rates of actual human decisions with those of counterfactual AI decisions" (PDF watermark interleaved between "58" and "billion") | Abstract (p. 2) | V |
| 100 | 2303.07462v2 | "novel decisions (i.e., previously unobserved moves) occurred more frequently and became associated with higher decision quality after the advent of superhuman AI" | Abstract (p. 2) | V |
| 101 | 2303.07462v2 | KataGo simulations: "10,000 game patterns for each of the 5.8 million decisions" | Results (p. 5) | V |
| 102 | 2303.07462v2 | novelty: "the first move that makes the game's move sequence historically novel" (chess-community practice) | Results, Novelty in Decision-Making (p. 7) | V |
| 103 | 2303.07462v2 | Novelty Index = 602 − move number of the novel move: "subtracted its move number from the maximum move number observed in the dataset (i.e., 602) to form a Novelty Index" | p. 7 | V |
| 104 | 2303.07462v2 | "We examined only the first 60 moves in each game" | p. 7, footnote 2 | V |
| 105 | 2303.07462v2 | drift: "the observed set of unique move sequences grew over time and pushed novel moves later" | p. 7 | V |
| 106 | 2303.07462v2 | DQI "equals [100% - (win rate of the Counterfactual AI Decision - win rate of the actual human decision)]" (watermark interleaved) | Materials and Methods (p. 19) | V |
| 107 | 2303.07462v2 | DQI regression: "the interaction term was significant and positive, β3 = 0.515" (After AI × Novelty; Table 1: 0.51504, SE 0.02147; Novelty Dummy −0.60770) | p. 8; Table 1 (p. 9) | V |
| 108 | 2303.07462v2 | Table 1 "Observations 5,857,513"; SEs clustered by player | Table 1 (p. 9) | V |
| 109 | 2303.07462v2 | "about 40% of all human decisions matched the optimal AI decisions" | p. 10 | V |
| 110 | 2303.07462v2 | "more than half of all novel moves in our entire dataset occur before Move 10 and that 99% of opening move sequences become historically novel by Move 21" | p. 12 | V |
| 111 | 2303.07462v2 | control: "adding 600k move decisions made by a superhuman AI program into the dataset" placed before the advent of AI; "the time trends of the Novelty Index hardly change" | pp. 15-16 | V |
| 112 | 2303.07462v2 | "memorization of AI decisions cannot be the sole explanation for the increase in decision quality and novelty" | Discussion (p. 17) | V |
| 113 | 2303.07462v2 | open: "whether the advent of superhuman AI increased novelty and thereby increased decision quality (i.e., whether each link in the possible causal chain can be established)" | Discussion (p. 17) | V |
| 114 | 2303.07462v2 | "Calculating the Novelty Index for each game did not require any superhuman AI program" | Materials and Methods (p. 18) | V |
| 115 | 2303.07462v2 | pre-AI: "comparatively flat trend observed in the preceding 66 years" | p. 6 | V |
| 116 | lindsey2024crosscoders (transformer-circuits.pub/2024/crosscoders) | status: "preliminary work that we're excited about, but not at the level of quality or rigor we hold our full papers to" | Research-update box (top) | V |
| 117 | lindsey2024crosscoders | "Crosscoders can produce shared sets of features across models. This includes one model across training or finetuning" | Intro (Model Diffing bullet) | V |
| 118 | lindsey2024crosscoders | crosscoder = shared encoder f(x) = ReLU(Σ_l W_enc^l a^l(x) + b) with per-layer/per-model decoders; loss = Σ_l reconstruction + Σ_i f_i Σ_l ‖W_dec,i^l‖ (L1-of-norms) | Sec. 2 (Crosscoder Basics) | V |
| 119 | lindsey2024crosscoders | the L1-of-norms version "uncovers a mix of shared and model-specific features, while the L2-of-norms version results in uncovering only shared features" | Sec. 2 | V |
| 120 | lindsey2024crosscoders | "We trained a crosscoder with 1 million features on the residual stream activations from the middle layer of Claude 3 Sonnet and the base model from which it was finetuned" | Sec. 4.3 | V |
| 121 | lindsey2024crosscoders | "These model-specific features would indicate features learned, or forgotten, during finetuning" | Sec. 4.3 | V |
| 122 | lindsey2024crosscoders | by relative decoder norms "features cluster into three obvious groups"; "between four and five thousand model-specific features for each model, out of a total 1 million features" | Sec. 4.3 | V |
| 123 | lindsey2024crosscoders | example fine-tune-specific feature: "A refusal feature that activates on dangerous requests" | Sec. 4.3 | V |
| 124 | lindsey2024crosscoders | "the majority of the model-exclusive features are not immediately interpretable" | Sec. 4.3 | V |
| 125 | lindsey2024crosscoders | shared features with misaligned decoders: "for a few thousand features, the correlation was very low or even negative"; "we suspect that these indicate cases where the finetuned model uses a concept that was present in the base model, but in a new way" | Sec. 4.3 | V |
| 126 | lindsey2024crosscoders | "We suspect that the crosscoder approach is preferable when model finetuning involves a greater fraction of compute relative to the compute used for pretraining" | Sec. 4.3 | V |
| 127 | lindsey2024crosscoders | comparisons listed include "Training Snapshots" (evolution of features over training) and finetuning | Sec. 4.2 | V |
| 128 | lindsey2024crosscoders | "Crosscoder errors may be important and extremely difficult to interpret"; on diffing "our results there are a bit mixed" | Sec. 5.2; Sec. 5.1 | V |
| 129 | 2504.02922v4 | "two issues which stem from the crosscoders L1 training loss that can misattribute concepts as unique to the fine-tuned model, when they really exist in both models" | Abstract | V |
| 130 | 2504.02922v4 | Δnorm(j) = ½(1 + (‖d_chat‖ − ‖d_base‖)/max(‖d_chat‖, ‖d_base‖)) from Lindsey et al.; "we classify latents as base-only (0-0.1), chat-only (0.9-1.0), or shared (0.4-0.6)" | Sec. 2.2, Eq. 3 | V |
| 131 | 2504.02922v4 | "Complete Shrinkage: When the contribution of latent j is smaller in the base model than in the chat model, L1 regularization can force d^base_j to zero despite its presence in the base activation" | Sec. 2.2 | V |
| 132 | 2504.02922v4 | "Latent Decoupling: a chat-only latent j is also present in the base activations but is reconstructed by other base decoder latents" | Sec. 2.2 | V |
| 133 | 2504.02922v4 | Latent Scaling: β_j^base = argmin_β Σ‖β f_j(x) d_j^chat − h^base(x)‖²; ν_j = β^base/β^chat: "A value near zero indicates a chat-specific latent, while a value near one suggests the latent is equally present in both models" | Sec. 2.3, Eq. 4 | V |
| 134 | 2504.02922v4 | "most L1 crosscoder chat-only latents are not truly chat-specific (defined as ν^r<0.5 and ν^ε<0.2), while most BatchTopK chat-only latents are genuinely" chat-specific (3176 L1 chat-only latents compared) | Sec. 3.1; Fig. 2 | V |
| 135 | 2504.02922v4 | similar effects in Llama 3 chat models and "models fine-tuned with RL for reasoning and medical knowledge" | Sec. 3.1 (App. I) | V |
| 136 | 2504.02922v4 | limitation: "our inability to distinguish between truly novel latents learned during chat-tuning and existing latents that have merely shifted their activation patterns" | Sec. 5 | V |
| 137 | 2504.02922v4 | BatchTopK "error terms still contain a lot of information about the chat model behavior" | Sec. 5 | V |
| 138 | 2212.03827v2 | CCS finds "a direction in activation space that satisfies logical consistency properties, such as that a statement and its negation have opposite truth values" | Abstract | V |
| 139 | 2212.03827v2 | "it outperforms zero-shot accuracy by 4% on average" (6 models, 10 datasets) | Abstract | V |
| 140 | 2212.03827v2 | misleading prompts make "zero-shot accuracy to drop by up to 9.5%" without decreasing CCS accuracy | Sec. 1 (Sec. 3.2.2) | V |
| 141 | 2212.03827v2 | goal: "discovering what language models know, distinct from what they say" | Abstract | V |
| 142 | 2212.03827v2 | confidence loss read as "imposing a second consistency property on the probabilities: the law of excluded middle" | Sec. 2.2 | V |
| 143 | 2212.03827v2 | limitation: CCS needs a direction a supervised probe could find; "This requires that a model is both capable of evaluating the truth of a given input, and also that the model actively evaluates the truth of that input" | Sec. 5.1 | V |
| 144 | 2102.12452v4 | critiques include "the correlational nature of the method" | Sec. 1 | V |
| 145 | 2102.12452v4 | "Does model f use the information discovered by probe g?"; probing "does not tell us whether this property is involved in predictions of f" | Sec. 4.3 | V |
| 146 | 2102.12452v4 | probe performance "may tell us more about the probe g than about the model f" (on Hewitt & Liang); control tasks: "it is less clear how to apply this idea more broadly, such as in sentence-level properties" | Sec. 4.1 | V |
| 147 | 2102.12452v4 | control functions vs control tasks: "subsequent work showed that the two criteria are almost equivalent, both theoretically and empirically" | Sec. 4.1 | V |
| 148 | 1610.01644v4 | "We use linear classifiers, which we refer to as "probes", trained entirely independently of the model itself"; "the linear separability of features increase monotonically along the depth of the model" | Abstract | V |
| 149 | 2401.01967v1 | "capabilities learned from pre-training are not removed, but rather bypassed" | Abstract | V |
| 150 | 2401.01967v1 | "DPO does not remove the capability of generating toxic outputs, but learns an "offset", distributed amongst its layers, to "bypass" the regions that elicit toxicity" | Sec. 1 | V |
| 151 | 2401.01967v1 | every parameter "has a cosine similarity score greater than 0.99 and on average a norm difference less than 1e-5" vs GPT2 | Sec. 5.1 | V |
| 152 | 2401.01967v1 | un-aligning: select 7 toxic MLP vectors "and scale their key vectors by 10x. By doing so, the model reverts back to its pre-aligned toxic behavior" | Sec. 5.3 | V |
| 153 | 2308.10248v5 | prompting and fine-tuning "do not fully elicit a model's capabilities"; ActAdd "does not require any machine optimization and works with a single pair of data points" | Abstract | V |
| 154 | 2312.06681v4 | CAA vectors = average "difference in residual stream activations between pairs of positive and negative examples of a particular behavior" | Abstract | V |
| 155 | 2312.06681v4 | base-vs-chat vector cosine similarity "decays as we increase the layer from which they are extracted, except for a peak between layers 7 and 15" | Sec. 8.3; Fig. 9 | V |
| 156 | 2312.06681v4 | vectors from the Llama 2 base model applied to Llama 2 Chat: "the effect transfers significantly, especially between layers 10 and 15" | Sec. 8.3; Fig. 10 | V |
| 157 | 2406.00877v1 | Leela: "activations on certain squares of future moves are unusually important causally"; a probe predicts "the optimal move 2 turns ahead with 92% accuracy" (states with a single best line) | Abstract | V |
| 158 | 2006.00995v3 | "the utility of a property for a given task can be assessed by measuring the influence of a causal intervention that removes it from the representation"; "conventional probing performance is not correlated to task importance" | Abstract | V |
| 159 | 2005.00719v3 | "models can learn to encode linguistic properties even if they are not needed for the task on which the model was trained"; encoded "considerably above chance-level even when distributed in the data as random noise" | Abstract | V |
| 160 | 2004.03061v2 | "one should always select the highest performing probe one can, even if it is more complex, since it will result in a tighter estimate" (probing = mutual information estimation) | Abstract | V |
| 161 | 1711.11279v5 | TCAV "uses directional derivatives to quantify the degree to which a user-defined concept is important to a classification result" | Abstract | V |
| 162 | 1711.11279v5 | null: CAVs retrained against random examples, "typically 500" runs; "If we can reject the null hypothesis of a TCAV score of 0.5, we can consider the resulting concept as related to the class prediction in a significant way" (two-sided t-test, Bonferroni) | Sec. 3.5 | V |
| 163 | 2312.10029v2 | unsupervised methods "do not discover knowledge -- instead they seem to discover whatever feature of the activations is most prominent"; "arbitrary features (not just knowledge) satisfy the consistency structure" of CCS | Abstract | V |
| 164 | 2306.03341v6 | ITI "improves its truthfulness from 32.5% to 65.1%" (Alpaca, TruthfulQA); "locates truthful directions using only few hundred examples" | Abstract | V |
| 165 | 2306.03341v6 | "LLMs may have an internal representation of the likelihood of something being true, even as they produce falsehoods on the surface" | Abstract | V |
| 166 | 2510.13900v3 | "analyzing activation differences on the first few tokens of random text and steering by adding this difference to the model activations produces text similar to the format and general content of the finetuning data" | Abstract | V |
| 167 | 2510.13900v3 | "We suspect these biases reflect overfitting and find that mixing pretraining data into the finetuning corpus largely removes them" | Abstract | V |
| 168 | 2509.05291v2 | crosscoders across pretraining checkpoints; RelIE "to trace training stages at which individual features become causally important for task performance"; detects "feature emergence, maintenance, and discontinuation during pretraining" | Abstract | V |
| 169 | 2309.10105v2 | "language models implicitly infer the task of the prompt and that fine-tuning skews this inference towards tasks in the fine-tuning distribution"; conjugate prompting "recovers some of the pretraining capabilities in our synthetic setup" | Abstract | V |
| 170 | silver2016alphago (doi 10.1038/nature16961; DeepMind-hosted PDF) | SL policy network "57.0% using all input features" (expert-move prediction, held-out test set) | p. 2 | V |
| 171 | silver2016alphago | RL policy network "won more than 80% of games against the SL policy network"; with no search "won 85% of games against Pachi" | p. 2 | V |
| 172 | silver2016alphago | "It is worth noting that the SL policy network pσ performed better in AlphaGo than the stronger RL policy network pρ, presumably because humans select a diverse beam of promising moves, whereas RL optimizes for the single best move" (verified as single-line fragments; two-column PDF) | p. 3 | V |
| 173 | silver2016alphago | "Received 11 November 2015; accepted 5 January 2016" — i.e. before the March 2016 Lee Sedol match, so the paper cannot report Move 37 | p. 6 (end matter) | V |
| 174 | deepmind-alphago-page (https://deepmind.google/research/alphago/) | "In game two, it played Move 37 - a move that had a 1 in 10,000 chance of being used"; Lee Sedol's "move 78, which had a 1 in 10,000 chance of being played" | web page (first-party, undated) | V |
| 175 | wired2016-two-moves (Wired, C. Metz, Mar 2016) | "As Silver told me, AlphaGo had calculated that there was a one-in-ten-thousand chance that a human would make that move" | article, section "One in Ten Thousand" | V (secondary) |
| 176 | — | that 1/10,000 is the prior probability of Move 37 **under AlphaGo's SL policy network** (how it was computed, which network, which position encoding) | no peer-reviewed or technical source found; DeepMind page gives no method, Wired paraphrases Silver | UNVERIFIED (secondary sources only) |
| 177 | silver2017alphagozero (Nature 550:354; UCL author manuscript) | "Notably, although supervised learning achieved higher move prediction accuracy, the self-learned player performed much better overall"; "This suggests that AlphaGo Zero may be learning a strategy that is qualitatively different to human play" | Sec. 2 (p. 8) | V |
| 178 | silver2017alphagozero | professional-move prediction (GoKifu validation): supervised 20-block 54.3 %, RL 20-block 49.0 %, RL 40-block 51.3 % | Extended Data Table 1 (p. 35) | V |
| 179 | silver2017alphagozero | AlphaGo Zero found "non-standard strategies beyond the scope of traditional Go knowledge"; "ultimately AlphaGo Zero preferred new joseki variants that were previously unknown" | Sec. 3 (p. 10) | V |
| 180 | silver2017alphagozero | joseki dated by "the first time each sequence occured (taking account of rotation and reflection) during self-play training"; ladders "were only understood by AlphaGo Zero much later in training" | Fig. 5 caption (p. 11); Sec. 3 | V |
| 181 | romeraparedes2023funsearch (doi 10.1038/s41586-023-06924-6) | "we discover new constructions of large cap sets going beyond the best-known ones, both in finite dimensional and asymptotic cases" | Abstract | V |
| 182 | romeraparedes2023funsearch | "in dimension n = 8, FunSearch found a larger cap set than what was previously known" (512 vectors); "the largest improvement in 20 years to the asymptotic lower bound" | Results (cap set); Introduction | V |
| 183 | romeraparedes2023funsearch | "a scientific discovery—a new piece of verifiable knowledge about a notorious scientific problem—using an LLM"; the LLM is "a source of diverse (syntactically correct) programs with occasionally interesting ideas" | Introduction; Discussion | V |
| 184 | 2308.09175v3 | AZdb: "diverse chess playstyles and specialization in various openings contributed to a 50 Elo advantage over AZ" | Abstract | V |
| 185 | 2308.09175v3 | AZ (1M simulations) "only solved 11.76% of the Challenge set and 3.64% of the Penrose set" | Sec. 4.2 | V |
| 186 | 2308.09175v3 | "max-over-latents, is an oracle that selects the player who has successfully solved the puzzle"; it "is not a feasible method, because the solution to a puzzle is obviously unknown" | Sec. 3 | V |
| 187 | 2308.09175v3 | "AZ's ability to understand difficult chess positions (in terms of prior, raw value, and MCTS value) improves when it is allowed to train from those puzzle positions" | Sec. 4.2 | V |
| 188 | 2211.14673v1 | "we find that MCTS discovers concepts before the neural network learns to encode them" | Abstract; Sec. 4.3 | V |
| 189 | 2211.14673v1 | selectivity per Hewitt & Liang, control = boards whose cells are mapped "to a random cell in a consistent manner to form the transposed board" | Sec. 3.2 | V |
| 190 | 2211.14673v1 | "Success on these behavioral tests are necessary but are alone insufficient to establish that the model has the concept"; first improvement = "the epoch with a 5% increase over the baseline value"; "we do not run a counterfactual study" | Sec. 3.3; Sec. 4.3; Sec. 3.4 | V |
| 191 | 2510.07364v1 (abstract page, `2510.07364v1.abs.txt`) | v1 headline, superseded in v4: "Across three base and four thinking models", "our hybrid model recovers up to 91% of the performance gap to thinking models without any weight updates while steering only 12% of tokens"; "pre-training is when models acquire most of their reasoning mechanisms, and post-training teaches efficient deployment of these mechanisms at the right time" | v1 Abstract | V |
| 192 | 1909.03368v1 | Sec. 3.5: "We chose rank constraints of 10 and 45, respectively (with no other changes,) for linear and MLP part-of-speech tagging probes" — inconsistent with the Table 1 caption (all PoS probes rank 10) | Sec. 3.5 vs Table 1 caption | V |
