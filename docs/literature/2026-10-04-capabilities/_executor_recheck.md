# Executor's independent re-check of the readers' claims (run capability-defs)

Method: random rows drawn from each reader's `_claims_<TAG>.md` (`shuf --random-source=<(yes 42)`), each phrase
searched again with `capability_defs/lit/quote.py` in the reader's fetched text (`~/cd_sources/`). V = found verbatim
(up to whitespace / math markup) at the stated location; X = not found or wrong location.

| # | reader | id | claim (abridged) | stated location | found at | status |
|---|---|---|---|---|---|---|
| 1 | L1 | 2407.21787v3 | "these laws are not as exact as training scaling laws" | Sec. 3.1 | 3.1 Scaling Laws for Repeated Sampling | V |
| 2 | L1 | 2502.17578v1 | "selection bias … power law scaling are more likely to garner more interest" | Sec. 7 | 7 Discussion and Future Directions | V |
| 3 | L1 | 2310.03262v3 | "not captured by conventional evaluation strategies due to insufficient measurement resolution" | Abstract | abstract (line 70) | V |
| 4 | L1 | 2509.24012v2 | "the compute law predicts slightly worse for small k and the gold reference law … large k" | Abstract | abstract (line 57) | V |
| 5 | L1 | 2407.21787v3 | "With unlimited samples, any model that assigns a non-zero probability to every sequence will achieve perfect coverage" | Sec. 1 | 1 Introduction | V |
| 6 | L1 | 2510.05197v1 | RLVR "training on difficult problems requires correctly sizing batches …" | Sec. 1.1 | 1.1 Contributions | V |
| 7 | L1 | 2510.05197v1 | "Ground truth estimates are computed for pass@k using all 10,000 available samples" | Sec. 5.1 | 5.1 Experimental Setup (text has `10\,000`; found after normalising the LaTeX thin space) | V |
| 8 | L2 | 2305.15324v2 | "Researchers will need to bring latent capabilities to the surface (for example, by prompt engineering or f… | Sec. 4, Table 2 | 4 Building evaluations for extreme risk | V |
| 9 | L2 | 2405.19550v1 | "it is difficult to show that a model does not possess a particular capability" | Sec. 2 | 2 Password-locked models | V |
| 10 | L2 | 2502.02180v3 | "evaluators should at least have fine-tuning access (and use it)" | Sec. 5 | 5 Discussion | V |
| 11 | L2 | 2406.07358v4 | "actual capability to be the best performance on a certain task it can achieve, given the best currently av… | Sec. 2 | 2 Defining sandbagging | V |
| 12 | L2 | 2502.02180v3 | "Comparing anti-refusal training to fine-tuning on random noise might reveal why anti-refusal training works" | Sec. 6 | 6 Conclusion | V |
| 13 | L3 | 2509.04259v1 | "we found that all algorithms lead to full rank weight updates" | Sec. 6 | 6 Alternative Hypothesis | V |
| 14 | L3 | 1802.07044v5 | "Prequential codes depend on the performance of the underlying training algorithm" | Sec. 3.4 | 3.4 Prequential or Online Code | V |
| 15 | L3 | 2110.08420v3 | mean pvi gap correct vs incorrect (BERT-base) "is 3.03, 2.87, and 2.45 bits respectively" (SNLI, MultiNLI, … | Sec. 3.2 | 3.2 pvi in Practice (line 367; the tool's heading detector mislabels it 3.1) | V |
| 16 | L3 | 2110.08420v3 | Shannon MI "is not an option—it would not change after X is encrypted" | Sec. 2.1 | 2.1 Background | V |
| 17 | L3 | 2003.12298v1 | uniform encoding "yields codelength" n log2 K bits | Sec. 2.1 | 2.1 Transmission of Data Using a Model | V |
| 18 | L4 | 1806.02643v2 | "Elo is, essentially, a uniform average in logit space"; an Elo rating exists iff curl(logit P) = 0; "Elo r… | Sec. 3.1 (Prop. on Elo; "Multidimensional Elo") | 3.1 Agents vs agents (AvA) | V |
| 19 | L4 | 2402.14992v2 | dimension chosen by validation: "choose the dimension that maximizes the prediction power of the IRT model … | Sec. 4.4; Sec. 3.2 fn. 3 | 4.4 Fitting the IRT model | V |
| 20 | L4 | 2503.14499v4 | "AI agent success rate is imperfectly predicted by human time-to-complete, meaning that other factors also … | App. E.3 | E.3 Interpreting time horizon | V |
| 21 | L4 | 2405.10938v3 | limitation: extending to "other post-training setups, including scenarios involving fine-tuning or more int… | Sec. 7 | 7 Conclusion, Limitations, and Future Work | V |
| 22 | L4 | 2503.06378v2 | "introduces 18 open scales in the range (0,∞)" via DeLeAn rubrics; applied "to 16,108 instances from 63 tas… | Sec. 1; Sec. 2.2; Abstract | 2.2 Slicing the Demand-Ability Space | V |
| 23 | L5 | kakade2002cpi_tex | "Through a more uniform restart distribution, the agent can gain information about states that it wouldn't … | Sec. 1 | Sec. 1 (TeX source; no heading match) | V |
| 24 | L5 | 2205.11275v2 | posteriors are "generally non-parametric: they might lie outside the class of probability distributions rep… | Sec. 5 (Inference) | 5 Separation of modelling and inference | V |
| 25 | L5 | 1908.08351v2 | consistency on incorrect outputs: "Transformer is the most consistent, but with a low score of only 0.34" | Sec. 7.2, p. 32 | page 32 | V |
| 26 | L5 | 2503.21878v2 | r* "can represent the extent to which y agrees with human preference, passes a proof checker, or passes a u… | Sec. 2 | line 34 (whitespace-insensitive match; Sec. 2) | V |
| 27 | L5 | 1912.09713v2 | compositional generalisation = "the ability to systematically generalize to composed test examples of a cer… | Sec. 1 | 1 Introduction | V |
| 28 | L6 | lindsey2024crosscoders | "Crosscoder errors may be important and extremely difficult to interpret"; on diffing "our results there ar… | Sec. 5.2; Sec. 5.1 | (web text; no heading match) | V |
| 29 | L6 | 2311.12786v2 | "A linear readout at an intermediate layer is used in the definition above to emphasize that the notion of … | Sec. 3 | 3 Defining our notion of capabilities | V |
| 30 | L6 | 2402.14811v1 | CMAP: "patching activations of the same components of different models on the same input" | Sec. 6.1 | 6.1 Cross-model activation patching | V |
| 31 | L6 | 2402.14811v1 | base circuit = "a sparse set of 72 attention heads in four groups"; in fine-tuned models it "alone can rest… | Sec. 1 | 1 Introduction | V |
| 32 | L6 | 1909.03368v1 | "Selectivity is defined as the difference between linguistic task accuracy and control task accuracy" | Fig. 2 caption | Fig. 2 caption (placed before Sec. 2 in the HTML) | V |

**Result:** 32 claims re-checked (7 L1 at 05:41, 25 L2–L6 at 06:55; sample `shuf --random-source=<(yes 7)`), **32 / 32 verified** at the stated location. Pre-registered L1 expectation (≤ 2 errors in ≥ 30) met.
