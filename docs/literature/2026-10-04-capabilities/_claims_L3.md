# Claim ledger — reader L3 (information and compression, learnability / sample complexity / size of the update, likelihood-based measures)

Written 2026-10-05 by agent:claude. Every V claim was found with `capability_defs/lit/quote.py` in the
fetched text (`~/cd_sources/<id or slug>.txt`); locations are section / equation / table / figure / page as
read in the rendering named in the note or screen row. Quotes are verbatim up to whitespace, case, quote
marks and math-markup normalisation. Numbers from tables were matched as table cells in the fetched text.

| # | paper id | claim (short verbatim quote or exact number) | location | status V/UNVERIFIED |
|---|---|---|---|---|
| 1 | 2003.12298v1 | "the measure of interest changes from probe accuracy to the description length of labels given representations" | Abstract | V |
| 2 | 2003.12298v1 | "the description length evaluates 'the amount of effort' needed to achieve the quality" | Abstract | V |
| 3 | 2003.12298v1 | probe accuracies "do not substantially favour pretrained representations over randomly initialized ones" | Abstract | V |
| 4 | 2003.12298v1 | L_p = −Σ log2 p(y_i\|x_i): "This is the Shannon-Huffman code" | Sec. 2.1, Eq. 1 | V |
| 5 | 2003.12298v1 | uniform encoding "yields codelength" n log2 K bits | Sec. 2.1 | V |
| 6 | 2003.12298v1 | "the compression is limited by the mutual information (MI) between inputs" and outputs | Sec. 2.1 | V |
| 7 | 2003.12298v1 | online code: "Alice starts by communicating" the first block "with a uniform code"; blocks at "0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.25, 12.5, 25, 50, 100 percent of the dataset" | Sec. 2.2.2, Eq. 4, footnote 4 | V |
| 8 | 2003.12298v1 | "The online code is related to the area under the learning curve" | Sec. 1 | V |
| 9 | 2003.12298v1 | model cost = "the difference between the cross-entropy of the model trained on all data and online codelength" | Sec. 2.3 | V |
| 10 | 2003.12298v1 | small data and small probe "reflect the same property: the strength of the regularity in the data" | Sec. 2.3 | V |
| 11 | 2003.12298v1 | layer 0: accuracy 93.7 / 96.3, online code 173 / 302 kbits (linguistic / control) | Table 2 | V |
| 12 | 2003.12298v1 | "for layer 0, accuracy for the control task is higher, but the code is twice longer than for the linguistic task" | Sec. 3.2 | V |
| 13 | 2003.12298v1 | control-task codelengths "at least twice larger" | Sec. 3.2 | V |
| 14 | 2003.12298v1 | "codelength shows large difference between trained and randomly initialized representations" | Sec. 4 | V |
| 15 | 2003.12298v1 | PoS layer 1: accuracy 97.8 / 95.7, online code 192 / 294 kbits (trained / random init) | Table 6 | V |
| 16 | 2003.12298v1 | "compression bounds for the randomly initialized model are closer to those of context-agnostic Layer 0 than representations from the trained model" | Sec. 4.2 | V |
| 17 | 2003.12298v1 | "In striking contrast to accuracy, MDL results are stable across settings" | Sec. 3.3 | V |
| 18 | 2003.12298v1 | "accuracy is wrong for 8 out of 10 settings, MDL is always correct" | Fig. 3 caption | V |
| 19 | 2003.12298v1 | "we do not consider practical implementations of transmission algorithms" | Sec. 2, footnote 2 | V |
| 20 | 1802.07044v5 | MDL: "a good model of data is a model that is good at losslessly compressing the data, including the cost of describing the model itself" | Abstract | V |
| 21 | 1802.07044v5 | "these variational methods provide surprisingly poor compression bounds"; "simple incremental encoding methods yield excellent compression values on deep networks" | Abstract | V |
| 22 | 1802.07044v5 | fake labels: "these models do not compress fake labels" ... "that no information is present in the model parameters, and that no learning has occurred" | Sec. 1, Fig. 1 | V |
| 23 | 1802.07044v5 | true labels: "half of the description length is information contained in the weights" | Fig. 1 caption | V |
| 24 | 1802.07044v5 | prequential code − final log-loss "can be interpreted as the amount of information that the trained parameters contain about the data" | Sec. 3.4 (after Eq. 3.6) | V |
| 25 | 1802.07044v5 | "The model parameters are never encoded explicitly in this method" | Sec. 3.4 | V |
| 26 | 1802.07044v5 | "Prequential codes depend on the performance of the underlying training algorithm" | Sec. 3.4 | V |
| 27 | 1802.07044v5 | MNIST uniform code 60000 × log2 10 = 199 kbits; prequential 4.10 kbits, "a compression ratio of 0.021", 99.5% test acc.; "6 times smaller than the variational codelength" | Sec. 2.3; Sec. 3.4; Table 1 | V |
| 28 | 1802.07044v5 | CIFAR10 prequential 45.3 kbits, "a compression ratio of 0.27" | Sec. 3.4 | V |
| 29 | 1802.07044v5 | float32 two-part code for 1 M params = 32 Mbits, "or 200 times the uniform encoding on CIFAR10" | Sec. 3.1 | V |
| 30 | 1802.07044v5 | intrinsic-dimension code on MNIST: > 9.28 kbits (weights only), 90% test accuracy | Table 1 | V |
| 31 | 1802.07044v5 | "A weakness of prequential codes is the catch-up phenomenon"; VGGb "needs 5,000 samples on CIFAR to reach a cumulative compression ratio" < 1 | Sec. 3.4 (Model Switching), Fig. 2 | V |
| 32 | 1802.07044v5 | classical codes ≈ nH(Y\|X) + (d/2) log2 n + O(1): "This corresponds to the BIC criterion for model selection" | Sec. 4 | V |
| 33 | 1802.07044v5 | random labels: code ≥ n log2 K − δ − 1 w.p. 1 − 2^−δ; "can just not be compressed by any algorithm" | App. A, Prop. 2 | V |
| 34 | 1802.07044v5 | "the gain of any codelength compared to the uniform code is limited by the amount of mutual information between input and output" | Sec. 2.4 | V |
| 35 | 2002.10689v1 | "consider a dataset of encrypted messages intercepted from an opponent" (high MI, not usable) | Sec. 1 | V |
| 36 | 2002.10689v1 | "A predictive family is a set of predictive models the agent is allowed to use, e.g., due to computational or statistical constraints"; condition named "optional ignorance" | Sec. 2, Def. 1, Eq. 1 | V |
| 37 | 2002.10689v1 | V-entropy "is the smallest expected negative log-likelihood that can be achieved predicting" Y given X with models from V | Sec. 2, Def. 2 | V |
| 38 | 2002.10689v1 | I_V(X→Y) = H_V(Y\|∅) − H_V(Y\|X), "to represent the change in predictability of an output variable" | Sec. 2, Def. 3, Eq. 3 | V |
| 39 | 2002.10689v1 | with V = Ω the quantity "is the Shannon mutual information" | Prop. 1.1 | V |
| 40 | 2002.10689v1 | "in violation of the data processing inequality", V-information "can be created through computation" | Abstract | V |
| 41 | 2002.10689v1 | decryption: "processing increases the "usable information"" | Sec. 3.2 | V |
| 42 | 2002.10689v1 | asymmetry: "it is easy to predict Y from X but not vice versa" | Sec. 3.3 | V |
| 43 | 2002.10689v1 | a family "with large Rademacher complexity) could lead to overfitting" (PAC bound Thm. 1) | Sec. 4, Thm. 1, Eq. 5 | V |
| 44 | 2002.10689v1 | deterministic Moving-MNIST: "every pair of frames has the same mutual information" | Sec. 6.3 | V |
| 45 | 2002.10689v1 | limitation: "Shannon information can be manipulated with certain additive algebra" but not general V-information | App. F | V |
| 46 | 2002.10689v1 | "exploitation of usable information (classification and reinforcement learning) could potentially be framed" similarly (future work) | App. F | V |
| 47 | 2110.08420v3 | difficulty = "the lack of V-usable information (Xu et al., 2019), where a lower value indicates a more difficult dataset" | Abstract | V |
| 48 | 2110.08420v3 | Shannon MI "is not an option—it would not change after X is encrypted" | Sec. 2.1 | V |
| 49 | 2110.08420v3 | "estimating V-information involves training or finetuning only two models" (with input / null input) | Sec. 2.2 | V |
| 50 | 2110.08420v3 | pvi(x→y) = −log2 g[∅](y) + log2 g′[x](y): "the difference in the log-probability these models place on the gold label" | Sec. 3, Def. 3.1, Eq. 4 | V |
| 51 | 2110.08420v3 | "pvi is to V-information what pmi is to Shannon information" | Sec. 3, Eq. 5 | V |
| 52 | 2110.08420v3 | "Although the V-information cannot be negative, the pvi can be"; "A negative pvi simply means that the model is better off predicting the majority class than considering" X | Sec. 3 | V |
| 53 | 2110.08420v3 | mean pvi gap correct vs incorrect (BERT-base) "is 3.03, 2.87, and 2.45 bits respectively" (SNLI, MultiNLI, CoLA) | Sec. 3.2 | V |
| 54 | 2110.08420v3 | "the point at which instances start being incorrectly predicted is similar across datasets" (pvi ≈ 0.5) | Sec. 3.2, Fig. 3 | V |
| 55 | 2110.08420v3 | "The cross-model Pearson correlation between pvi estimates of SNLI instances is very high" (r > 0.80); "the correlation across seeds is r>0.85" | Sec. 3.2 | V |
| 56 | 2110.08420v3 | "if a dataset contained no usable information, then we would expect the correlation between pvi estimates across different models and seeds to be close to zero" | Sec. 3.2 | V |
| 57 | 2110.08420v3 | "the models start becoming less certain about the correct label long before they start predicting the wrong label" | Sec. 2.5, Fig. 2 | V |
| 58 | 2110.08420v3 | "offensive words contain 0.482 bits of BERT-usable information about the label beyond that which is contained in text sentiment" | Sec. 4.4, Eq. 6 | V |
| 59 | 2110.08420v3 | RDA: "Since the framework depends on the order of instances (i.e., what data has been transmitted thus far), it is unsuitable for estimating dataset difficulty" | Sec. 5 | V |
| 60 | 2110.08420v3 | future work: "Extending V-information to open-ended text generation, which does not induce explicit distributions over the output space" | Sec. 6 | V |
| 61 | 2110.08420v3 | "Note that the average pvi of a slice of data is not its V-information" | Sec. 3.1 | V |
| 62 | 2009.07368v2 | "We propose to measure the quality of a representation by the complexity of learning a predictor on top of the representation that achieves low loss on a task of interest" | Abstract | V |
| 63 | 2009.07368v2 | "prior methods measure properties of an evaluation dataset of a specified size, whereas our methods measure properties of a predictor with a specified loss" | Abstract | V |
| 64 | 2009.07368v2 | "We take the position that the best representation is the one which allows for the most efficient learning of a predictor to solve the task" | Sec. 1 | V |
| 65 | 2009.07368v2 | "MDL grows without bound as the size of the evaluation dataset" grows (MDL ≥ n·H(Y\|φ(X))) | Sec. 4.1, Eq. 10 | V |
| 66 | 2009.07368v2 | parity example: ranking "favors the noisy label representation when" n ≪ d | Sec. 3.1 | V |
| 67 | 2009.07368v2 | "MI is insensitive to statistical complexity"; "MI is insensitive to computational complexity" | Sec. 3.2 | V |
| 68 | 2009.07368v2 | "All three prior methods lack a predefined notion of successfully solving a task" | Sec. 3.3 | V |
| 69 | 2009.07368v2 | SDL = Σ_i [L(A_φ,i) − ε]_+ : "it gives the cost (in terms of information) for re-creating an" ε-loss predictor | Sec. 4.1, Def. 2, Eq. 12 | V |
| 70 | 2009.07368v2 | εSC = min{n : L ≤ ε}: complexity of learning an ε-loss predictor "by the number of samples it takes" | Sec. 4.2, Def. 3, Eq. 13 | V |
| 71 | 2009.07368v2 | "an implementation is able to report that the given complexity estimate is only a lower bound" | Sec. 4.1 | V |
| 72 | 2009.07368v2 | ε "can be done by training a large model on the raw representation of the full evaluation dataset and using its validation loss as" ε | Sec. 4.3 | V |
| 73 | 2009.07368v2 | ELMo PoS: MDL at n = 461: 884.54 / 1009.26 / 1017.72; at n = 474,838: 92403.41 / 52648.50 / 65468.54; SDL(ε=0.1) > 40882.72 / 2765.11 / 7069.56; εSC > 474838 / 237967 / 474838 | Table 2 | V |
| 74 | 2009.07368v2 | "evaluating our measures to high precision took about an hour on one GPU" | Sec. 5.2 | V |
| 75 | 2009.07368v2 | "Each of these measures depends on a choice of algorithm" | Sec. 7 | V |
| 76 | 2009.07368v2 | capacity / data restrictions "are necessary to separate the performance of randomly- and linguistically-pretrained representations" (citing Zhang & Bowman; Hewitt & Liang) | Sec. 6 | V |
| 77 | 2012.13255v1 | "by optimizing only 200 trainable parameters randomly projected back into the full space, we can tune a RoBERTa model to achieve 90% of the full parameter performance levels on MRPC" | Abstract | V |
| 78 | 2012.13255v1 | "pre-training implicitly minimizes intrinsic dimension"; "larger models tend to have lower intrinsic dimension after a fixed number of pre-training updates" | Abstract | V |
| 79 | 2012.13255v1 | d90: satisfactory solution "defined ... as being 90% of the full training metric" | Sec. 3 | V |
| 80 | 2012.13255v1 | intrinsic dimensionality as "a continuous relaxation of the sparsification problem" | Sec. 2 | V |
| 81 | 2012.13255v1 | SAID d90: RoBERTa-Large 207 (MRPC) / 774 (QQP); BERT-Base 1608 / 8030 | Table 1 | V |
| 82 | 2012.13255v1 | "it is likely that the true intrinsic dimension is much lower" | Sec. 4.2 | V |
| 83 | 2012.13255v1 | d read "as the minimal description length of the task within the framework dictated by the pre-trained representations" | Sec. 5 | V |
| 84 | 2012.13255v1 | MRPC needs "less than a kilobyte of data to encode a complex natural language task within the framework provided by RoBERTa" | Sec. 5 | V |
| 85 | 2012.13255v1 | "the intrinsic dimensionality of RoBERTa-Base monotonically decreases as we continue pre-training"; search range d = 100 "and a maximum of 4 million" | Sec. 5.1, Fig. 2 | V |
| 86 | 2012.13255v1 | "Unable to compute means either we could not fine-tune the full checkpoint to accuracy above majority class or stabilize SAID training" | Fig. 2 caption | V |
| 87 | 2012.13255v1 | "the more parameters we have in the model, the less we need to represent a task" | Sec. 5.2 | V |
| 88 | 2012.13255v1 | bounds "only apply to pre-trained methods trained with the intrinsic dimension subspace method; research has yet to show that standard SGD optimizes in this low dimensional space" | Sec. 5.3.1 | V |
| 89 | 2312.01552v1 | "base LLMs and their alignment-tuned versions perform nearly identically in decoding on the majority of token positions (i.e., they share the top-ranked tokens)" | Abstract | V |
| 90 | 2312.01552v1 | "the knowledge required for answering user queries predominantly comes from the base LLMs themselves" | Abstract | V |
| 91 | 2312.01552v1 | URIAL "requiring as few as three constant stylistic examples and a system prompt" | Abstract | V |
| 92 | 2312.01552v1 | aligned output obtained "via greedy decoding"; shifted positions (η > 3): o_t "is rather unlikely to be sampled by" the base | Sec. 2.1 | V |
| 93 | 2312.01552v1 | "77.7% of the tokens are at such unshifted positions, which increases to 92.2% when including marginal positions" | Sec. 2.2 | V |
| 94 | 2312.01552v1 | "The shifted token ratios are all very low (5%-7%)" (three base/aligned pairs) | Sec. 2.2, Fig. 3 | V |
| 95 | 2312.01552v1 | "the KL-divergence goes down over time and the base-prob keeps increasing over time"; "the average base-rank of aligned tokens are lower than 5 soon after" t ≥ 5 | Sec. 2.2, Fig. 4 | V |
| 96 | 2312.01552v1 | "untuned LLMs can fluently generate the answer based solely on the context prefix" | Sec. 2.2 | V |
| 97 | 2312.01552v1 | "it is essential to accurately distinguish which knowledge and reasoning capabilities originate from pre-training as opposed to those that must be acquired through alignment tuning" | Sec. 1 | V |
| 98 | 2312.01552v1 | alignment affects "primarily stylistic elements and safety disclaimers in just 5-8% of cases" | Sec. 6 | V |
| 99 | 2312.01552v1 | "model tuning may still be necessary for tasks such as coding" ..., mathematics, interactive agents | Sec. 5.4 | V |
| 100 | 2312.01552v1 | cited forgetting: SuperNI SFT on Llama-13B, BBH "decreasing from 36.9 to 2.8" (from Wang et al. 2023) | Sec. 5.1 | V |
| 101 | 2505.11711v2 | gains "result from updating only a small subnetwork comprising just 5%-30% of the parameters, with the rest effectively unchanged" | Abstract | V |
| 102 | 2505.11711v2 | "the updates to almost all parameter matrices are nearly full-rank"; DeepSeek-Math-7B GRPO update rank 99.4% | Abstract; Table 2 | V |
| 103 | 2505.11711v2 | sparsity treats "two bfloat16 values as equal when their absolute difference does not exceed" 10^-5 | Sec. 2.2 | V |
| 104 | 2505.11711v2 | "68.5%–96.0% of parameters remain unchanged after RL"; "Deepseek-R1-Zero presents a update sparsity of 86.0%" | Sec. 3, Table 1 | V |
| 105 | 2505.11711v2 | "SFT induces dense updates (only 6%-15% sparsity)" | Sec. 3, Fig. 1 | V |
| 106 | 2505.11711v2 | "In PRIME, 72% parameters are never updated, 8% have gradients canceling each other out, and 20% constitute the subnetwork" | Fig. 2 caption | V |
| 107 | 2505.11711v2 | "In DPO, 94.0% weights are same between" θ_full and θ_sub (90.5% PRIME); "are 100% identical when using a tolerance of" 10^-4 | Sec. 4 | V |
| 108 | 2505.11711v2 | seed variation: "varying the random seed yields overlaps of" 60.5% / 60.6% (random 36.7%); seed+data+algorithm: "we still observe notable overlaps of 59.1% and 33.2%" | Sec. 5, Table 4 | V |
| 109 | 2505.11711v2 | GRPO "with KL regularization achieved a sparsity of 69.8%, while the variant trained without KL regularization reached 68.8%" | Sec. 6 | V |
| 110 | 2505.11711v2 | "SFT on in-distribution data produces sparse updates, while DPO with out-of-distribution data produces dense ones"; rejection-sampling SFT "yields around 90.0% update sparsity" | Sec. 6, Table 5 | V |
| 111 | 2505.11711v2 | "when gradients are computed on sequences that the policy already assigns high probabilities to, little update to the parameters would be needed" | Sec. 6 | V |
| 112 | 2505.11711v2 | footnote: "if one were to perform backpropagation manually on paper with unlimited numerical precision, the resulting parameter updates would be dense" | Sec. 6, footnote 5 | V |
| 113 | 2505.11711v2 | abstract conjecture: sparsity "can be primarily attributed to training on data that is near the policy distribution" | Abstract | V |
| 114 | 2509.04259v1 | "the degree of forgetting is determined by the distributional shift, measured as the KL-divergence between the fine-tuned and base policy evaluated on the new task" | Abstract | V |
| 115 | 2509.04259v1 | "on-policy RL is implicitly biased towards KL-minimal solutions among the many that solve the new task, whereas SFT can converge to distributions arbitrarily far from the base model" | Abstract | V |
| 116 | 2509.04259v1 | RL's Razor: "among all ways to solve a new task, RL prefers those closest in KL to the original model" | Abstract | V |
| 117 | 2509.04259v1 | "RL constrains learning to outputs already given non-negligible probability by the base model" | Sec. 1 | V |
| 118 | 2509.04259v1 | RL setup: "we used only a binary success indicator as the reward, without explicit KL regularization" | Sec. 3.1 | V |
| 119 | 2509.04259v1 | 1–0 Reinforce: "This is equivalent to sampling from the model and performing SFT on correct answers only"; "1–0 Reinforce behaves similarly to GRPO, while SimPO resembles SFT" | Sec. 5.1, Fig. 4 | V |
| 120 | 2509.04259v1 | "the critical factor is not the presence of negative gradients but the use of on-policy data" | Sec. 5.1 | V |
| 121 | 2509.04259v1 | rejection sampling = argmin_q KL(q‖p) s.t. E_q[R] = 1; proof: KL(q‖p) = KL(q‖p(·\|S)) − log p(S) for q on S | Lemma 5.1 / A.1 and its proof (App. A) | V |
| 122 | 2509.04259v1 | "policy gradient selects, among all optimal representable policies, the one closest in KL-divergence to the starting policy" | Sec. 5.2, Thm. 5.2 | V |
| 123 | 2509.04259v1 | caveat: policy set "induced by a neural network parametrization is not in general" e-flat | App. A (practical considerations) | V |
| 124 | 2509.04259v1 | ParityMNIST "A quadratic fit achieves R^2=0.96"; LLMs "with a quadratic fit achieving R^2=0.71" | Sec. 4 | V |
| 125 | 2509.04259v1 | "SFT trained on the oracle distribution retained more prior knowledge than RL"; "The distilled SFT matched RL's accuracy–forgetting trade-off" | Sec. 4 | V |
| 126 | 2509.04259v1 | predictors: forward KL R² 0.96 ± 0.01; weight change L1 0.34 ± 0.02; "none approached the predictive power of forward KL" | Table 1; Sec. 6 | V |
| 127 | 2509.04259v1 | "the reason for the observed sparse updates was the use of bfloat16 for model training"; "Performing the same training with float32 resulted in models with identical performance but without any sparsity in their weight updates" | Sec. 6 | V |
| 128 | 2509.04259v1 | "we found that all algorithms lead to full rank weight updates" | Sec. 6 | V |
| 129 | 2509.04259v1 | "we still lack a mechanistic account of why larger KL shifts on the new task disrupt prior knowledge" | Sec. 7 | V |
| 130 | 2511.08567v1 | "sparsity is a surface artifact of a model-conditioned optimization bias" | Abstract | V |
| 131 | 2511.08567v1 | "Gate III (Precision) hides micro-updates in non-preferred regions, making the off-principal bias appear as sparsity" | Abstract | V |
| 132 | 2511.08567v1 | "RLVR learns off-principal directions in weight space, achieving gains via minimal spectral drift" | Abstract | V |
| 133 | 2511.08567v1 | "SFT targets principal weights, distorts the spectrum" | Abstract | V |
| 134 | 2404.17546v1 | inference = "sampling from a target unnormalized density and estimating its intractable (log) normalization constant" | Sec. 1 | V |
| 135 | 2404.17546v1 | Z_σ = Σ p0(s)φ(s) "as the normalization constant or partition function, which is intractable due to the summation over" sequences | Sec. 1, Eq. 1 | V |
| 136 | 2404.17546v1 | φ may be "a verifier's prediction of correctness (for reasoning tasks)" | Sec. 1 | V |
| 137 | 2404.17546v1 | SMC gives lower bounds on log Z; "upper bounds may often be obtained when an exact target sample is available" | Sec. 1 | V |
| 138 | 2404.17546v1 | bound gap "in fact yields an upper bound on the symmetrized KL divergence between inference samples and the target distribution" | Sec. 1 | V |
| 139 | 2404.17546v1 | "twisted SMC can improve upon SIS and efficiently sample rare events" | Sec. 7.1 | V |
| 140 | 2404.17546v1 | promise of "the ability to estimate or bound probabilities of rare behaviors" | Sec. 8 | V |
| 141 | 2207.05221v4 | models "predict which questions they will be able to answer correctly" | Abstract | V |
| 142 | 2207.05221v4 | "larger models are well-calibrated on diverse multiple choice and true/false questions when they are provided in the right format" | Abstract | V |
| 143 | 2207.05221v4 | P(IK): "the probability that "I know" the answer to a question, without reference to any particular proposed answer" | Abstract | V |
| 144 | 2207.05221v4 | "they struggle with calibration of P(IK) on new tasks" | Abstract | V |
| 145 | 2010.02650v2 | "exact maximum a posteriori (MAP) decoding of neural language generators frequently leads to low-quality results" | Abstract | V |
| 146 | 2010.02650v2 | "why high probability under a model alone may not indicate adequacy" | Abstract | V |
| 147 | 2010.02650v2 | "beam search enforces uniform information density in text" | Abstract | V |
| 148 | 2005.10283v2 | "the most likely translations under the model accumulate so little probability mass that the mode can be considered essentially arbitrary" | Abstract | V |
| 149 | 2005.10283v2 | "We demonstrate that beam search outputs are rare events" | Sec. 1 | V |
| 150 | 2005.10283v2 | "The mode might only account for a tiny portion of the probability mass, and can actually be extremely unlikely under the learnt distribution" | Sec. 4 | V |
| 151 | 2005.10283v2 | 1,000 samples cover "between 16.4% and 57.8% of the probability mass" on held-out data; "only about half of the probability space has been explored" | Sec. 7.1, Fig. 2 | V |
| 152 | 2005.10283v2 | "an approximation to minimum Bayes risk decoding gives competitive results" | Abstract | V |
| 153 | 1904.09751v2 | "using likelihood as a decoding objective leads to text that is bland and strangely repetitive" | Abstract | V |
| 154 | 1904.09751v2 | "decoding strategies alone can dramatically effect the quality of machine text, even when generated from exactly the same neural language model" | Abstract | V |
| 155 | 1904.09751v2 | nucleus sampling: "effectively truncating the less reliable tail of the distribution" | Abstract | V |
| 156 | 2109.09234v1 | goal: "measuring information that is contained in the representation but not in the baseline" | Abstract | V |
| 157 | 2109.09234v1 | baseline comparisons "cannot detect when the representation is predictive of just the aspects of part-of-speech not explainable by the word identity" | Abstract | V |
| 158 | 2109.09234v1 | "propose conditional probing, which explicitly conditions on the information in the baseline" | Abstract | V |
| 159 | 2109.09234v1 | after conditioning, "properties like part-of-speech are accessible at deeper layers of a network than previously thought" | Abstract | V |
| 160 | 2103.03872v1 | "We introduce a method to determine if a certain capability helps to achieve an accurate model of given data" | Abstract | V |
| 161 | 2103.03872v1 | "a subroutine is useful if and only if the minimal program that invokes it is shorter than the one that does not" | Abstract | V |
| 162 | 2103.03872v1 | "Since minimum program length is uncomputable, we instead estimate the labels' minimum description length (MDL) as a proxy" | Abstract | V |
| 163 | 2103.03872v1 | applications include "evaluating the utility of generating subquestions before answering a question" | Abstract | V |
| 164 | 2309.10668v2 | "predictive models can be transformed into lossless compressors and vice versa" | Abstract | V |
| 165 | 2309.10668v2 | Chinchilla 70B "compresses ImageNet patches to 43.4% and LibriSpeech samples to 16.4% of their raw size" | Abstract | V |
| 166 | 2309.10668v2 | "the compression viewpoint provides novel insights into scaling laws, tokenization, and in-context learning" | Abstract | V |
| 167 | 1804.08838v1 | "note at which dimension solutions first appear, and define this to be the intrinsic dimension of the objective landscape" | Abstract | V |
| 168 | 1804.08838v1 | "the intrinsic dimension for a given dataset varies little across a family of models with vastly different sizes" | Abstract | V |
| 169 | 1804.08838v1 | "solving the inverted pendulum problem is 100 times easier than classifying digits from MNIST" | Abstract | V |
| 170 | 1804.08838v1 | "a simple technique for constructively obtaining an upper bound on the minimum description length of a solution" | Abstract | V |
| 171 | 2305.11206v1 | Superficial Alignment Hypothesis: "A model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users" | Sec. 2 | V |
| 172 | 2305.11206v1 | corollary: "one could sufficiently tune a pretrained language model with a rather small set of examples" | Sec. 2 | V |
| 173 | 2305.11206v1 | LIMA trained "on only 1,000 carefully curated prompts and responses" | Abstract | V |
| 174 | 2305.11206v1 | "almost all knowledge in large language models is learned during pretraining" | Abstract | V |
| 175 | 2305.11206v1 | LIMA "either equivalent or strictly preferred to GPT-4 in 43% of cases" | Abstract | V |
| 176 | 2305.11206v1 | term "Superficial Alignment Hypothesis" defined | Sec. 2 | V |
| 177 | 2502.03387v3 | LIMO Hypothesis: "In foundation models where domain knowledge has been comprehensively encoded during pre-training, sophisticated reasoning can emerge through minimal but strategically designed demonstrations of cognitive processes" | Abstract | V |
| 178 | 2502.03387v3 | "the threshold for eliciting complex reasoning is not dictated by task complexity" | Abstract | V |
| 179 | 2502.03387v3 | factor (1): "the completeness of the model's pre-trained knowledge base" | Abstract | V |
| 180 | 2502.03387v3 | LIMO "achieves 63.3% accuracy on AIME24" | Abstract | V |
| 181 | 2504.20571v3 | one example "elevates model performance on MATH500 from 36.0% to 73.6% (8.6% improvement beyond format correction)" | Abstract | V |
| 182 | 2504.20571v3 | "This result matches the performance obtained using the 1.2k DeepScaleR subset" | Abstract | V |
| 183 | 2504.20571v3 | "a phenomenon we term post-saturation generalization" | Abstract | V |
| 184 | 2504.20571v3 | "the effectiveness of 1-shot RLVR primarily arises from the policy gradient loss" | Abstract | V |
| 185 | 2309.06979v3 | "even simple models such as linear next-token predictors, trained on Chain-of-Thought (CoT) data, can approximate any function efficiently computed by a Turing machine" | Abstract | V |
| 186 | 2309.06979v3 | "length complexity -- which measures the number of intermediate tokens in a CoT sequence required to approximate some target function" | Abstract | V |
| 187 | 2503.07932v2 | learning studied "both when the chain-of-thought is observed and when training only on prompt-answer pairs, with the chain-of-thought latent" | Abstract | V |
| 188 | 2503.07932v2 | "time invariance allows for sample complexity that is independent of the length of the chain-of-thought" | Abstract | V |
| 189 | 2204.02892v4 | with intermediate supervision, "unlearnable composite problems can become learnable" | Abstract | V |
| 190 | 2204.02892v4 | condition: tasks that "can be decomposed into a polynomial number of simple sub-tasks, each of which depends only on O(1) previous sub-task results" | Abstract | V |
| 191 | 1901.11373v1 | general linguistic intelligence = "the ability to reuse previously acquired knowledge about a language's lexicon, syntax, semantics, and pragmatic conventions to adapt to new tasks quickly" | Abstract | V |
| 192 | 1901.11373v1 | "a new evaluation metric based on an online encoding of the test data that quantifies how quickly an existing agent (model) learns a new task" | Abstract | V |
| 193 | 1901.11373v1 | "these models still require a lot of in-domain training examples" | Abstract | V |
| 194 | 2205.11275v2 | "the standard RL approach is flawed as an objective for fine-tuning LMs because it leads to distribution collapse" | Abstract | V |
| 195 | 2205.11275v2 | "KL-regularised RL is equivalent to variational inference: approximating a Bayesian posterior which specifies how to update a prior LM to conform with evidence provided by the reward function" | Abstract | V |
| 196 | 2205.11275v2 | "These problems are best viewed as Bayesian inference: approximating a pre-defined target distribution" | Abstract | V |
| 197 | schulman2025lorawithoutregret | "LoRA fully matches the learning performance of FullFT when running policy gradient algorithms for reinforcement learning, even with ranks as low as 1" | section "Reinforcement learning" (fetched line 134) | V |
| 198 | schulman2025lorawithoutregret | "in policy gradient methods, learning is driven by the advantage function which provides only O(1) bits per episode" | section "Reinforcement learning" (line 145) | V |
| 199 | schulman2025lorawithoutregret | MATH run: ~10,000 problems × 32 samples, so "the whole training process only needs to absorb 320,000 bits" | section "Reinforcement learning" (line 147) | V |
| 200 | schulman2025lorawithoutregret | I(G;R\|history) ≤ H(Adv): "the number of bits of useful information gleaned per episode is O(1), independent of model size" | section "How much capacity is needed..." (line 279) | V |
| 201 | schulman2025lorawithoutregret | "Note that this estimate is an upper bound on the information absorbed by training"; with a zero-reward initial policy "the entropy of the advantage is zero" | same section (line 279) | V |
| 202 | schulman2025lorawithoutregret | "The claim of 1-bit-per-episode may only apply narrowly to policy gradient algorithms" | same section (line 271) | V |
| 203 | schulman2025lorawithoutregret | authorship: "John Schulman in collaboration with others at Thinking Machines Sep 29, 2025" | header | V |
| 204 | 2402.10193v3 | "it's intuitive to assume that fine-tuning adds less new information to the model, and is thus more compressible" | Abstract | V |
| 205 | 2402.10193v3 | BitDelta "successfully quantizes this delta down to 1 bit without compromising performance" | Abstract | V |
| 206 | 2402.10193v3 | "the potential redundancy of information added during fine-tuning" | Abstract | V |
| 207 | 2405.09673v2 | "in the standard low-rank settings, LoRA substantially underperforms full finetuning" (code, math) | Abstract | V |
| 208 | 2405.09673v2 | "LoRA better maintains the base model's performance on tasks outside the target domain" | Abstract | V |
| 209 | 2405.09673v2 | "full finetuning learns perturbations with a rank that is 10-100X greater than typical LoRA configurations" | Abstract | V |
| 210 | 1909.03368v1 | "does this mean that the representations encode linguistic structure or just that the probe has learned the linguistic task" | Abstract | V |
| 211 | 1909.03368v1 | control tasks: "By construction, these tasks can only be learned by the probe itself" | Abstract | V |
| 212 | 1909.03368v1 | a good probe "should be selective, achieving high linguistic task accuracy and low control task accuracy" | Abstract | V |
| 213 | 1908.10090v1 | "For more than 50% of the sentences, the model in fact assigns its global best score to the empty translation" | Abstract | V |
| 214 | 1908.10090v1 | "at the root of the problem of empty translations lies an inherent bias towards shorter translations" | Abstract | V |
| 215 | 2505.24832v3 | "We propose a new method for estimating how much a model knows about a datapoint" | Abstract | V |
| 216 | 2505.24832v3 | split into "unintended memorization, the information a model contains about a specific dataset, and generalization, the information a model contains about the true data-generation process" | Abstract | V |
| 217 | 2505.24832v3 | "GPT-style models have a capacity of approximately 3.6 bits per parameter" | Abstract | V |
| 218 | 2505.24832v3 | "a model is considered to have memorized an input if the input can be compressed into a shorter encoding when the model is available" | Sec. 1 | V |
| 219 | 2505.24832v3 | estimator: H^K(x\|θ̂) by −log p(x\|θ̂); with target and reference "We simply compute" −log max{p(x\|θ̂), p(x\|θ)} (Def. 3: mem_U = H^K(x\|θ) − H^K(x\|θ, θ̂)) | Sec. 2.2 Def. 3; Sec. 2.3 | V |
| 220 | 2505.24832v3 | "our choice of reference model is a larger model with the same architecture" | Sec. 2.3 | V |
| 221 | 2210.07931v1 | prequential MDL objective: "minimize the cumulative next-step log-loss when sequentially going through the data and using previous observations for parameter estimation" | Abstract | V |
| 222 | 2210.07931v1 | "online-learning with rehearsal has favorable performance compared to the previously widely used block-wise estimation" | Abstract | V |
| 223 | 2210.07931v1 | "We propose forward-calibration to better align the models predictions with the empirical observations" | Abstract | V |
| 224 | 2410.14086v4 | "the next-token prediction loss used to train in-context learners is directly equivalent to a data compression technique called prequential coding" | Abstract | V |
| 225 | 2410.14086v4 | minimising it "amounts to jointly minimizing both the training error and the complexity of the model that was implicitly learned from context" | Abstract | V |
| 226 | 1905.12213v5 | "Whatever information a deep neural network has gleaned from training data is encoded in its weights" | Abstract | V |
| 227 | 1905.12213v5 | "We measure information in a neural network via the optimal trade-off between accuracy of the response and complexity of the weights, measured by their coding length" | Abstract | V |
| 228 | 1905.12213v5 | "a trained network is a deterministic map, so standard information measures can be degenerate" | Abstract | V |
