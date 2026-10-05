# Claim ledger — reader L1 (sampling-based measures, compute-equivalence, emergence / metric choice)

Written 2026-10-05 by agent:claude. Every V claim was found with `capability_defs/lit/quote.py` in the
fetched text (`~/cd_sources/<id>.txt`); locations are section / equation / figure as read in the arXiv HTML
(or PDF) rendering named in the note. Quotes are verbatim up to whitespace and math-markup normalisation.

| # | paper id | claim (short verbatim quote or exact number) | location | status V/UNVERIFIED |
|---|---|---|---|---|
| 1 | 2107.03374v2 | "computing pass@ k in this way can have high variance" (Kulal-style k samples) | Sec. 2.1 | V |
| 2 | 2107.03374v2 | "we use n=200 and k≤100"; unbiased estimator pass@k := E_Problems[1 − C(n−c,k)/C(n,k)] | Sec. 2.1, Eq. 1 | V |
| 3 | 2107.03374v2 | "Calculating this estimator directly results in very large numbers and numerical instability" (stable product form in Fig. 3) | Sec. 2.1, Fig. 3 | V |
| 4 | 2107.03374v2 | plug-in 1−(1−p̂)^k "results in a consistent underestimate"; "The gap doesn't fully close even when n>5k" | App. A, Fig. 13 | V |
| 5 | 2107.03374v2 | Eq. 1 unbiased "because it estimates the fail probability (1−pass@1)^k as the probability of drawing k failed samples without replacement" | App. A | V |
| 6 | 2107.03374v2 | "The unbiased estimator may have a slightly higher variance initially but allows for a fair comparison across different numbers of samples" | App. A, Fig. 13 caption | V |
| 7 | 2107.03374v2 | "higher temperatures are optimal for larger k"; T*=0.2 (pass@1) vs 0.8 (pass@100) for 679M model | Sec. 3.3, Fig. 5 | V |
| 8 | 2107.03374v2 | "Performance appears to scale smoothly as a sigmoid in log-parameters" | Sec. 3.3, Fig. 6 caption | V |
| 9 | 2107.03374v2 | "GPT-Neo-2.7B roughly equivalent to Codex-85M (30× fewer parameters)" | Sec. 3.4 | V |
| 10 | 2407.21787v3 | coverage "the fraction of problems that are solved by any generated sample – scales with the number of samples over four orders of magnitude" | Abstract | V |
| 11 | 2407.21787v3 | "With unlimited samples, any model that assigns a non-zero probability to every sequence will achieve perfect coverage" | Sec. 1 | V |
| 12 | 2407.21787v3 | "repeated sampling is only practical if we can improve coverage with a feasible budget" | Sec. 1 | V |
| 13 | 2407.21787v3 | Gemma-2B CodeContests "from 0.02% with one sample to 7.1% with 10,000 samples" | Sec. 1 | V |
| 14 | 2407.21787v3 | Pythia-160M MATH "from a pass@1 of 0.27% to a pass@10k of 57%" | Sec. 2.2 | V |
| 15 | 2407.21787v3 | "All Pythia models achieve zero coverage on this dataset, even with a budget of 10,000 samples" (CodeContests) | Sec. 2.2 | V |
| 16 | 2407.21787v3 | coverage model log(c) ≈ a k^b (Eq. 2), c ≈ exp(a k^b) (Eq. 3) | Sec. 3.1 | V |
| 17 | 2407.21787v3 | "these laws are not as exact as training scaling laws (most strikingly on MiniF2F-MATH)" | Sec. 3.1 | V |
| 18 | 2407.21787v3 | "the traced S-curves have the same slope, but unique horizontal offsets" | Sec. 3.2 | V |
| 19 | 2407.21787v3 | "the increase in the log-sample-budget (or equivalently, the multiplicative increase in the sample budget) needed to improve coverage from c to c′ is approximately constant" | Sec. 3.2, Fig. 6 | V |
| 20 | 2407.21787v3 | "On MiniF2F, GSM8K and MATH, Llama-3-8B-Instruct always obtains higher coverage than the larger (and more expensive) 70B model when the FLOP budget is fixed" | Sec. 2.3, Fig. 4 | V |
| 21 | 2407.21787v3 | power-law fits use "40 points spaced evenly along a log scale" from 0 to 10,000 | App. C.1 | V |
| 22 | 2502.17578v1 | "a simple mathematical calculation predicts that on each problem, the failure rate should fall exponentially with the number of attempts" | Abstract; Sec. 2, Eq. 7 | V |
| 23 | 2502.17578v1 | aggregate power law arises "if the distribution of single-attempt success probabilities is heavy tailed such that a small fraction of tasks with extremely low success probabilities collectively warp the aggregate success trend into a power law" | Abstract | V |
| 24 | 2502.17578v1 | power law with exponent b "if and only if the distribution over problems of single-attempt success probabilities itself behaves like a power law near 0 with exponent b-1" | Sec. 3, Thm 3.1-3.2 | V |
| 25 | 2502.17578v1 | −log(pass_D@k) ∼ C Γ(b) k^(−b) | Sec. 3, Thm 3.1 | V |
| 26 | 2502.17578v1 | "the scale parameter was critical to obtain good fits" (scaled Kumaraswamy) | Sec. 3, Fig. 4 | V |
| 27 | 2502.17578v1 | Llama 3 8B IT "could be successfully jailbroken on every prompt within the permitted sampling budget and thus had no heavy left tail" | Sec. 4 | V |
| 28 | 2502.17578v1 | problems with pass_i@1 in "(0,1/Number of Samples) such that, due to finite sampling, we lack the resolution to measure" | Sec. 5 | V |
| 29 | 2502.17578v1 | fit "such that the distribution's probability mass in the interval (0,1/Number of Samples) matches the empirical fraction of problems in this tail bucket" | Sec. 5 | V |
| 30 | 2502.17578v1 | exponent forecast "with an order of magnitude lower relative error, or equivalently, ∼2−4 orders of magnitude less inference compute" | Abstract; Sec. 5, Fig. 7 | V |
| 31 | 2502.17578v1 | "The distributional estimator performs well even under distributional mismatch" | Sec. 5 | V |
| 32 | 2502.17578v1 | "the largest pass_i@1 values were typically 1-2 orders of magnitude less than 1.0" | App. F | V |
| 33 | 2502.17578v1 | "selection bias, in that more interesting patterns such as power law scaling are more likely to garner more interest" | Sec. 7 | V |
| 34 | 2502.17578v1 | prefer "average success rate" "as it avoids the binary implication that each problem either is or is not solved after k attempts" | Sec. 1 | V |
| 35 | 2510.05197v1 | "how can one accurately predict a model's behavior when scaled to a massive number of attempts, given a vastly smaller sampling budget?" | Abstract | V |
| 36 | 2510.05197v1 | Chen et al. estimators "are only defined when the number of samples taken for each problem is greater than or equal to the number of attempts k" | Sec. 3.1 | V |
| 37 | 2510.05197v1 | regression flaws: estimates "are not independent for different k"; "are not homoskedastic"; "Power laws typically apply only for large values of k" | Sec. 3.2 | V |
| 38 | 2510.05197v1 | discretized beta "consistently produces downward-biased estimates of the distribution U" | Sec. 3.3, Fig. 2 | V |
| 39 | 2510.05197v1 | "we find empirically that using the scale parameter does not improve predictions" | Sec. 3.3 | V |
| 40 | 2510.05197v1 | "we model U as a beta distribution" and fit beta-binomial likelihood to raw counts | Sec. 4.1, Eq. 15-17 | V |
| 41 | 2510.05197v1 | "Distinguishing between an easy problem (pass_i@1=0.25) and a very easy problem (pass_i@1=0.75) provides little to no information" | Sec. 4.2 | V |
| 42 | 2510.05197v1 | variance-optimal allocation b*_i ∝ sqrt(pass_i@1 (1−pass_i@1)^(2k−1)) | Sec. 4.2, Thm 1 | V |
| 43 | 2510.05197v1 | "it is difficult to empirically isolate the benefits of the sampling method alone" | Sec. 4.2 | V |
| 44 | 2510.05197v1 | log-log regression "often diverging to predict impossible pass rates greater than 1" | Sec. 5.2 | V |
| 45 | 2510.05197v1 | Schaeffer et al.'s method "consistently underestimates pass@k for large k" | Sec. 5.2 | V |
| 46 | 2510.05197v1 | "Ground truth estimates are computed for pass@k using all 10,000 available samples" | Sec. 5.1 | V |
| 47 | 2510.05197v1 | uniform sampling "prevents our estimator from determining whether these problems are impossible or just hard"; "This results in an upwards-biased estimate" | App. D.2, Fig. 6 | V |
| 48 | 2510.05197v1 | RLVR "training on difficult problems requires correctly sizing batches to ensure a non-zero success rate" | Sec. 1.1 | V |
| 49 | 2310.03262v3 | small models show improvements "not captured by conventional evaluation strategies due to insufficient measurement resolution" | Abstract | V |
| 50 | 2310.03262v3 | resolution "is the smallest probability difference that the evaluation strategy can detect" | Sec. 1, footnote 1 | V |
| 51 | 2310.03262v3 | "it is hard to define the budget K that is both acceptable in computation and has enough resolution" | Sec. 4.1 | V |
| 52 | 2310.03262v3 | "We stop sampling until r (a constant) samples have passed the evaluation and record the sampling number K"; PU = r/K | Sec. 4.1, Eq. 2 | V |
| 53 | 2310.03262v3 | "PU is a maximum likelihood estimate for P(s)"; "The failure time f=K-r follows the negative binomial distribution" | Sec. 4.1, Thm 1 | V |
| 54 | 2310.03262v3 | "we set r to as small as 1 or 2"; K capped "such as 10^5" | Sec. 4.1 | V |
| 55 | 2310.03262v3 | "deriving P(s) theoretically from the token probability on the ground truth solution is not feasible" ("there are likely to be multiple viable solutions") | Sec. 4.1 | V |
| 56 | 2310.03262v3 | "our evaluation strategy is designed to be applicable when a random baseline achieves P(s)=0" | Sec. 4.1, Limitations | V |
| 57 | 2310.03262v3 | small-model performances "are on the order of 10^-5" | Sec. 1 | V |
| 58 | 2310.03262v3 | 2.4B HumanEval (series 1): real 0.05990, instance-level fit 0.05987; "merely 0.05% deviation before training starts" | Table 1; Abstract | V |
| 59 | 2310.03262v3 | "0 PassUntil is obverseved even with 10^5 sampling time" (smallest models, some UICL tasks) | Sec. 6 | V |
| 60 | 2310.03262v3 | concave log(−log PU) vs log N = "super-scaling law growth, or 'accelerated emergence'" | Sec. 6, Def. 1 | V |
| 61 | 2310.03262v3 | "'multi-step reasoning' leads to sub-scaling law growth" | Sec. 6, Thm 2 | V |
| 62 | 2310.03262v3 | "one cannot deduce actual performance (accuracy) solely from loss values"; per-sample loss has no "one-to-one correlation with PassUntil results" | App. A.2 | V |
| 63 | 2310.03262v3 | "We suggest leveraging test loss on ground truth answers to assist the prediction for such instances" | Sec. 5.4 | V |
| 64 | 2304.15004v2 | emergent abilities "appear due the researcher's choice of metric rather than due to fundamental changes in model behavior with scale" | Abstract | V |
| 65 | 2304.15004v2 | "nonlinear or discontinuous metrics produce apparent emergent abilities, whereas linear or continuous metrics produce smooth, continuous, predictable changes" | Abstract | V |
| 66 | 2304.15004v2 | Wei et al. definition quoted: "abilities that are not present in smaller-scale models but are present in large-scale models; thus they cannot be predicted by simply extrapolating…" | Sec. 1 | V |
| 67 | 2304.15004v2 | secondary cause: "too few test data to accurately estimate the performance of smaller models" | Sec. 1 | V |
| 68 | 2304.15004v2 | "Accuracy(N) ≈ p_N(single token correct)^num. of tokens" | Sec. 2 | V |
| 69 | 2304.15004v2 | resolution "set by 1/test dataset size" | Sec. 2, footnote 2 | V |
| 70 | 2304.15004v2 | "all models in the InstructGPT/GPT-3 family achieve above-chance accuracy" with more test data | Sec. 3, Fig. 4 | V |
| 71 | 2304.15004v2 | "of the 39 preferred metrics in BIG-Bench, at most 5 display emergence"; "2 metrics account for >92% of claimed emergent abilities" | Sec. 4, Fig. 5 | V |
| 72 | 2304.15004v2 | "nothing in this paper should be interpreted as claiming that large language models cannot display emergent abilities" | Sec. 7 | V |
| 73 | 2304.15004v2 | Caballero et al. "explain emergence by assuming a piece-wise power law functional form" (emergence real under that view) | Sec. 6 | V |
| 74 | 2304.15004v2 | emergence claims "possibly infected by a failure to control for multiple comparisons" | Sec. 7 | V |
| 75 | 2312.07413v1 | CEG = "how much additional training compute would be needed to improve performance by the same amount as the enhancement" | Abstract | V |
| 76 | 2312.07413v1 | "The model should be trained compute-optimally, so that C is the minimal compute required to attain performance p"; "The CEG is given by C′/C" | Sec. 2 | V |
| 77 | 2312.07413v1 | "the exact relation between training compute and capabilities depends on the model family" | Sec. 2, footnote 9 | V |
| 78 | 2312.07413v1 | "The CEG can be seen as the ratio of the intrinsic performances of two models" | Sec. 2, footnote 10 | V |
| 79 | 2312.07413v1 | "most surveyed enhancements improve benchmark performance by more than a 5x increase in training compute, some by more than 20x"; fine-tuning "typically <1% of the original training cost" | Abstract | V |
| 80 | 2312.07413v1 | lower bound "if the evaluation data contains a model with the post-training enhancement that outperforms some bigger model without the enhancement"; "lower bounded by C(ML) / C(ME)" | Sec. 2; App. A | V |
| 81 | 2312.07413v1 | "a high CEG might not indicate that the post-training enhancement significantly improves performance, but instead indicate that additional training compute doesn't improve performance" | Sec. 4 | V |
| 82 | 2312.07413v1 | enhancements enabling tasks "impossible for any model without it. In these cases, even if the enhancement greatly improves performance, the CEG is not meaningful" | Sec. 4 | V |
| 83 | 2312.07413v1 | alternative definition: "the reduction in compute that can be achieved with an enhanced model without reducing performance" | Sec. 4 | V |
| 84 | 2312.07413v1 | Minerva "reaching a CEG of 30 in STEM benchmarks and 2400 in math benchmarks" | Sec. 5 | V |
| 85 | 2104.03113v2 | "for each additional 10× of train-time compute, about 15× of test-time compute can be eliminated, down to a floor of a single-node tree search" | Sec. IV-C, Fig. 9 | V |
| 86 | 2104.03113v2 | "the performance of a specific snapshot is sigmoid in the test-time compute budget" | Sec. IV-C, Fig. 8 | V |
| 87 | 2104.03113v2 | Elo anchored with perfect play: "we fix its play to zero for all Elo ratings reported herein" | Sec. II-D | V |
| 88 | 2104.03113v2 | "The minimum training compute needed to see any improvement over random play increases by 4× for each increment of board size" | Sec. IV-A3 | V |
| 89 | 2104.03113v2 | "the distance between random play and perfect play increases by 500 Elo for each increment of board size" | Sec. IV-A4 | V |
| 90 | 2104.03113v2 | "A spike with regards to compute might indicate the model had achieved some key insight"; observed performance "changes smoothly and predictably" | Sec. V | V |
| 91 | 2104.03113v2 | test-time optimisation "needs only optimise over one sample, while train-time compute meanwhile must optimise over the entire distribution of samples" | Sec. V | V |
| 92 | 2104.03113v2 | slope "500 Elo per order of magnitude increase in compute" | Sec. IV-A1 | V |
| 93 | 2301.13442v2 | "mean episode return, need not vary smoothly" | Abstract | V |
| 94 | 2301.13442v2 | intrinsic performance: "the minimum compute required to train a model of any size in the family to reach the same return (averaged over random seeds)" | Sec. 2.1, Definition | V |
| 95 | 2301.13442v2 | "the efficient frontier is mapped onto the line y=x by definition" | Sec. 2.1 | V |
| 96 | 2301.13442v2 | CoinRun natural metric "fail-to-success ratio" F = (10−R)/R; its log "can also be thought of as the logit function (inverse sigmoid) of the failure rate" | Sec. 4.5 | V |
| 97 | 2301.13442v2 | F ∝ I^-0.40 (easy) and I^-0.48 (hard) | Sec. 4.5 | V |
| 98 | 2301.13442v2 | "without a natural performance metric, we cannot extrapolate to unseen performance levels" | Sec. 5.1 | V |
| 99 | 2301.13442v2 | "we do not think conclusions that depend on the precise fitted values of our scaling constants can be drawn with confidence" | Sec. 5.3 | V |
| 100 | 2301.13442v2 | intrinsic performance fitted by "jointly fitting the power law constants and a monotonic function" | App. A | V |
| 101 | 2502.16797v1 | elicitation probability: "the probability that a sampled output from a query has a specific behavior" | Sec. 3.1, Eq. 1 | V |
| 102 | 2502.16797v1 | "the logarithm of the largest-quantile elicitation probabilities follows a power-law in the number of samples required to estimate them" | Sec. 1 | V |
| 103 | 2502.16797v1 | elicitation score ψ_i = −log(−log p_elicit(x_i)); "the tail of the log survival function is an approximately linear function of the elicitation score" | Sec. 3.3, Eq. 3-5 | V |
| 104 | 2502.16797v1 | OLS fit "for the ten highest elicitation scores during evaluation"; forecasts "sensitive to stochasticity in the specific evaluation set" | Sec. 3.3 | V |
| 105 | 2502.16797v1 | 900 → 90,000 samples: "within one order of magnitude of the true risk for 86% of misuse forecasts" | Sec. 1 | V |
| 106 | 2502.16797v1 | "The average absolute log error is 1.7 for the Gumbel-tail method, compared to 2.4 for the log-normal method"; underestimates 34% vs 72% | Sec. 4.2 | V |
| 107 | 2502.16797v1 | specific-output probability "can be done in a single forward pass—but may not reflect the actual likelihood of producing 'useful' instructions" | Sec. 4.1 | V |
| 108 | 2502.16797v1 | "We could also more efficiently compute probabilities via importance sampling"; adaptive stopping proposed | Sec. 4.5 | V |
| 109 | 2502.16797v1 | optimisation-based elicitation: "optimizing can find instances of a behavior that are too rare to ever come up in practice" | Sec. 7 | V |
| 110 | 2502.16797v1 | forecasts "across up to three orders of magnitude of query volume" | Abstract | V |
| 111 | 2410.13211v2 | low probability estimation: probability "too small to estimate by random sampling" | Abstract | V |
| 112 | 2410.13211v2 | naive sampling "uninformative at distinguishing between small probabilities like 10^-10 and 10^-20" | Sec. 2 | V |
| 113 | 2410.13211v2 | IS: "If we re-weight our observations properly, this gives an unbiased estimator for the true probability" | Sec. 3.1 | V |
| 114 | 2410.13211v2 | "importance sampling outperforms activation extrapolation, but both outperform naive sampling" | Abstract; Sec. 5 | V |
| 115 | 2410.13211v2 | ground truth probabilities "between 10^-9 and 10^-5" | Sec. 1 | V |
| 116 | 2410.13211v2 | "As long as the required importance sampling ratios can be computed, any method for red-teaming can be turned into an importance sampling method" | Sec. 6.2 | V |
| 117 | 2410.13211v2 | limitations: independent-token inputs; "single token sampled at temperature 0" | Sec. 6.4 | V |
| 118 | 2410.13211v2 | hash example: finding an input is "computationally infeasible" yet probability easy to model | Sec. 6.3 | V |
| 119 | 1906.04908v1 | "Our main evaluation metric is success rate at B: the fraction of test examples where the system generates an accepted program under the budget of B trials" | Sec. 6 | V |
| 120 | 1906.04908v1 | budget of 100 compilations: success rate "from 25.6% to 44.7%" (search vs top-one) | Abstract | V |
| 121 | 2505.22756v1 | GRPO "mainly enhances the execution skill—improving execution robustness on problems the model already knows how to solve—a phenomenon we call temperature distillation" | Abstract | V |
| 122 | 2505.22756v1 | "coverage wall, defined as the limit of Pass@K as K→∞" | Sec. 1 | V |
| 123 | 2505.22756v1 | Coverage(K,T*) = problems with ≥1 of K samples correct at optimal temperature; "k=64 is sufficiently large" | Sec. 3.3; App. C | V |
| 124 | 2505.22756v1 | "on the test set, GRPO fails to solve any new problems"; train subset "unlocks only two new problems" | Sec. 3.3, Fig. 3 | V |
| 125 | 2505.22756v1 | synthetic setting: conditions "under which RL can potentially overcome the coverage wall" | Abstract | V |
| 126 | 2509.24012v2 | "Our three scaling laws differ in the covariates used: (1) pretraining compute, (2) model parameters and pretraining tokens, (3) log likelihoods of gold reference solutions" | Abstract | V |
| 127 | 2509.24012v2 | "generative evaluations introduce new hyperparameters (in our setting, k) that act as a control lever for scaling behavior" | Abstract | V |
| 128 | 2509.24012v2 | compute law −log(pass_B@k) = E_0(k) + C_0(k)/C^α(k); "the irreducible error term E_0(k) falls roughly exponentially with k and is effectively 0 by k≈1×10^2" | Sec. 3.1, Eq. 3, Fig. 2 | V |
| 129 | 2509.24012v2 | compute exponent "rising moderately from 1.21×10^-1 to 3.75×10^-1" | Sec. 3.1 | V |
| 130 | 2509.24012v2 | "We calculated the average log-likelihood of these gold reference sequences to use to predict pass rates"; Eq. 6 −log(pass_B@k) = ξ_0(k) + K_0(k)[−log GoldProb_B]^κ(k) | Sec. 5, Eq. 5-6 | V |
| 131 | 2509.24012v2 | gold-law parameters "converge to their final values using models up to ∼5 orders of magnitude cheaper than the target" | Sec. 5.1, Fig. 7 | V |
| 132 | 2509.24012v2 | compute law: "reliable prediction requires checkpoints within ∼2 orders of magnitude of the target" | Fig. 3 caption | V |
| 133 | 2509.24012v2 | "the compute law predicts slightly worse for small k and the gold reference law predicts slightly worse for large k" | Abstract | V |
| 134 | 2509.24012v2 | "it is not immediately obvious why the likelihood of the specific benchmark-provided gold reference correlates so strongly with the pass rate" | Sec. 8 | V |
| 135 | 2509.24012v2 | "(ii) to what extent does this signal remain robust under heavy optimization pressure?" | Sec. 8 | V |
| 136 | 2509.24012v2 | "if we draw n samples per problem, then any pass rate on that problem below 1/n will likely appear to be 0" | App. B | V |
| 137 | 2509.24012v2 | "a minimum of 2^14 samples per model per problem, and then continued sampling until 10 successes were obtained or until a maximum of 2^15" | App. B | V |
| 138 | 2509.24012v2 | "The primary limitation of this work is its empirical focus on a single model family (Pythia)" | Sec. 8 | V |
| 139 | 2509.24012v2 | "pass-at-k is a continuous probability derived from the model's generative distribution" | Sec. 2 | V |
| 140 | 2509.24012v2 | "We used temperature-only sampling at τ=1.0" | Sec. 2 | V |
| 141 | 2206.07682v2 | "We consider an ability to be emergent if it is not present in smaller models but is present in larger models" | Abstract | V |
| 142 | 2206.07682v2 | "performance is near-random until a certain critical threshold of scale is reached, after which performance increases to substantially above random" | Sec. 2 | V |
| 143 | 2206.07682v2 | "cross-entropy loss improves even for small model scales where the downstream metrics … are close to random"; "improvements in the log-likelihood of the target sequence can be masked by such downstream metrics" | Sec. 5.1 | V |
| 144 | 2403.15796v3 | "models with the same pre-training loss, but different model and data sizes, generate the same performance on various downstream tasks" | Abstract | V |
| 145 | 2403.15796v3 | emergent "regardless of the continuity of metrics -- when its pre-training loss falls below a specific threshold" | Abstract | V |
| 146 | 2403.15796v3 | "redefine emergent abilities as those that manifest in models with lower pre-training losses" | Abstract | V |
| 147 | 2207.08799v3 | "learning abruptly occurs at approximately n^O(k) iterations" | Abstract | V |
| 148 | 2207.08799v3 | "SGD gradually amplifies the sparse solution via a Fourier gap in the population gradient, making continual progress that is invisible to loss and error metrics" | Abstract | V |
| 149 | 2207.08799v3 | "not explained by a Langevin-like mechanism" (SGD does not "stumble in the dark") | Abstract | V |
| 150 | 2408.03314v1 | compute-optimal allocation improves efficiency "by more than 4x compared to a best-of-N baseline" | Abstract | V |
| 151 | 2408.03314v1 | FLOPs-matched: "test-time compute can be used to outperform a 14x larger model" (problems with non-trivial base success) | Abstract | V |
| 152 | 2408.03314v1 | difficulty: "we bin the model's pass@1 rate – estimated from 2048 samples – on each question in the test set into five quantiles" | Sec. 3.2 | V |
| 153 | 2408.03314v1 | "we define the exchange rate between pretraining and inference FLOPs": X = 6ND_pretrain, Y = 2ND_inference | Sec. 7 | V |
| 154 | 2408.03314v1 | "on the harder questions or in settings with a higher inference load (e.g. R>>1), pretraining is a more effective way to improve performance" | Sec. 7, Fig. 9 | V |
| 155 | 2408.00724v3 | "scaling inference compute with inference strategies can be more computationally efficient than scaling model parameters" | Abstract | V |
| 156 | 2408.00724v3 | "the Llemma-7B model, when paired with our novel tree search algorithm, consistently outperforms the Llemma-34B model" | Abstract | V |
| 157 | 2411.17501v3 | resampling "fundamentally limited when verifiers are imperfect and have a non-zero probability of producing false positives" | Abstract | V |
| 158 | 2411.17501v3 | "no amount of inference scaling of weaker models can enable them to match the single-sample accuracy of a sufficiently strong model" | Abstract | V |
| 159 | 2411.17501v3 | "optimal sampling attempts are often fewer than 10" | Abstract | V |
| 160 | 2410.16377v2 | "a simple statistical ansatz based on memorization to study scaling laws in the context of inference" | Abstract | V |
| 161 | 2410.16377v2 | failure probabilities: "We think of p=p_i as a random variable, drawn from p∼Beta(α,β)" | Sec. 3.2 | V |
| 162 | 2410.16377v2 | "we also add an overall factor describing their maximal pass@k as A" | Sec. 3.2, Eq. 7 | V |
| 163 | 2410.16377v2 | "inference loss", "which exhibits a power law decay as the number of trials increases" | Abstract; Eq. 8 | V |
| 164 | 2410.16377v2 | fitted "β∼0.35, while the tail parameter is 2<α<20" | Sec. 3.2.1 | V |
| 165 | 2412.13147v5 | G-Pass@k "quantifying both the model's performance potential and its stability" | Abstract | V |
| 166 | 2412.13147v5 | G-Pass@k_τ = E[Σ_{j≥⌈τk⌉} C(c,j)C(n−c,k−j)/C(n,k)]; "Pass@k is a special case of G-Pass@k as τ approaches 0" | Sec. 2.2, Thm 2.1 | V |
| 167 | 2412.13147v5 | "at higher τ values, G-Pass@k_τ evaluates the model's stability, i.e., its level of mastery over the question" | Sec. 2.2 | V |
| 168 | 2510.13786v1 | "We fit sigmoidal compute-performance curves for RL training" | Abstract | V |
| 169 | 2510.13786v1 | R_C − R_0 = (A − R_0)/(1+(C_mid/C)^B); "0≤A≤1 represents the asymptotic pass rate, B>0 is a scaling exponent that determines the compute efficiency" | Sec. 1, Eq. 1 | V |
| 170 | 2510.13786v1 | design details "primarily modulate compute efficiency without materially shifting the asymptote" | Abstract | V |
| 171 | 2510.13786v1 | "we model pass rate versus log(compute) with a sigmoidal function" | Sec. 2.1 | V |
| 172 | 2510.02230v1 | "learning to solve certain training problems actively reduces the likelihood of correct solutions for others, leading to the decline of Pass@k performance" | Abstract | V |
| 173 | 2510.02230v1 | "RLVR disproportionately reinforces problems with high likelihood, correct solutions, under the base model, while suppressing other initially low-likelihood ones" | Abstract | V |
| 174 | 2510.02230v1 | "data curation algorithm that focuses RLVR learning on low-likelihood problems" | Abstract | V |
| 175 | 2410.05695v2 | RB = "the maximum of problem difficulty d at which the model's accuracy reaches a predefined threshold K1" | Sec. 2.1, Eq. 1 | V |
| 176 | 2410.05695v2 | "the part with an accuracy greater than 90% is a completely feasible reasoning boundary" (≤10% completely infeasible) | Sec. 2.3 | V |
| 177 | 2410.05695v2 | "Difficulty can be measured by factors like the number of reasoning steps or computational complexity" | Sec. 2.1 | V |
| 178 | villalobos2023tradingoff (epoch.ai) | "Increase the amount of compute per inference by 5-6 OOM in exchange for saving 3-4 OOM in training compute"; "only observed this in the case of solving coding problems and proving statements in formal mathematics" | summary list (top of page) | V |
| 179 | villalobos2023tradingoff (epoch.ai) | "as a rule of thumb … save around 1 OOM of compute in either training or inference, in exchange for increasing the other factor by somewhat more than 1 OOM" | main text after Fig. B | V |
| 180 | villalobos2023tradingoff (epoch.ai) | AlphaCode resampling: "spending 1.5 OOM of additional inference compute in order to save 1 OOM of training compute" | main text, Fig. B | V |
| 181 | 2411.16035v1 | "finetuning LLMs on a given task can shift the point in scaling at which emergence occurs towards less capable models" | Abstract | V |
| 182 | 2411.16035v1 | "accurately predict whether models trained with up to 4x more compute have emerged" | Abstract | V |
| 183 | 2411.16035v1 | task: "given access to current LLMs that have random few-shot accuracy on a task, can we predict whether future models (GPT-N+1) will have non-trivial accuracy" | Abstract | V |
| 184 | 2510.04265v4 | replace pass@k "with posterior estimates of a model's underlying success probability and credible intervals" | Abstract | V |
| 185 | 2510.04265v4 | "under a uniform prior, the Bayesian posterior mean is order-equivalent to average accuracy (Pass@1)" | Abstract | V |
| 186 | 2510.04265v4 | gaps "statistically meaningful (non-overlapping credible intervals) versus noise" | Abstract | V |
| 187 | 1812.01647v1 | "even matching the compute used for training is sometimes insufficient for evaluation" | Abstract | V |
| 188 | 1812.01647v1 | "focuses evaluation on adversarially chosen situations, while still providing unbiased estimates of failure probabilities" | Abstract | V |
| 189 | 1812.01647v1 | "a continuation approach that learns failure modes in related but less robust agents"; "reuse of data already collected for training the agent" | Abstract | V |
| 190 | 1812.01647v1 | "estimate failures rates of agents multiple orders of magnitude faster than standard evaluation schemes" | Abstract | V |
| 191 | 2403.05812v1 | "the compute required to reach a set performance threshold has halved approximately every 8 months" | Abstract | V |
| 192 | 2403.05812v1 | "augmented scaling laws, which enable us to quantify algorithmic progress" | Abstract | V |
| 193 | 2405.19550v1 | "a few high-quality demonstrations are often sufficient to fully elicit password-locked capabilities" | Abstract | V |
| 194 | 2405.19550v1 | "when only evaluations, and not demonstrations, are available, approaches like reinforcement learning are still often able to elicit capabilities" | Abstract | V |
| 195 | 2405.19550v1 | fine-tuning elicitation "may be unreliable when high-quality demonstrations are not available" | Abstract | V |
| 196 | 2405.10938v3 | "performance is a function of a low-dimensional capability space, and model families only vary in their efficiency in converting training compute to capabilities" | Abstract | V |
| 197 | 2405.10938v3 | "several emergent phenomena follow a smooth, sigmoidal behavior and are predictable from small models" | Abstract | V |
| 198 | 2405.10938v3 | "predict the impact of post-training interventions like Chain-of-Thought and Self-Consistency" | Abstract | V |
| 199 | 2407.21787v3 | SWE-bench: "we restrict the number of attempts per issue to 250" | Sec. 2.1 | V |
| 200 | 2502.17578v1 | Beta(α,β): −log(pass_D@k) ∝ k^−α; Kumaraswamy(α,β): ∝ k^−α; Uniform(0,β≤1): ∝ k^−1 | Sec. 3 | V |
| 201 | 2310.03262v3 | guessable items removed: "These common words make the exact string match susceptible to random guess's correctness" | Sec. 5.3 | V |
| 202 | 2510.05197v1 | Algorithm 1: s* ← min_i successes_i; among those, fewest attempts; uniform tie-break | Sec. 4.2, Alg. 1 | V |
| 203 | 2502.16797v1 | sampled elicitation quantiles "are frequently qualitatively linear for large enough n" (100-500 samples per query) | Sec. 4.5, Fig. 5 | V |
| 204 | 2502.16797v1 | "We do not study distribution shifts between evaluation and deployment queries" | Sec. 7 | V |
| 205 | 2510.05197v1 | data: "10,000 sampled successful or failed attempts for each of 100∼200 problems" (Brown 2024, Hughes 2024) | Sec. 5.1 | V |
