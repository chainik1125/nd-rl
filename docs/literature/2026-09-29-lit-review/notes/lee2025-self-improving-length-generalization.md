---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# Self-improving transformers: length ladders with self-labels, filtering and error avalanches
Paper: Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges (Nayoung Lee, Ziyang Cai, Avi Schwarzschild, et al., 2025)
Source: https://arxiv.org/abs/2502.01612v2 (version reviewed)
## Learnings
- Setup (§3): 14M LLaMA-style model, 6 layers, d 384, **NoPE**; greedy decoding; exact match. Round r labels problems of difficulty in [d_{r-1}, d_r] with M_{r-1}'s greedy output; trains on D_0 ∪ … ∪ D_{r-1}, "up-sample [the newest] with a sampling probability of 50%" (§3).
- It is continued fine-tuning, not retraining: e.g. reverse addition "trained on the combined dataset … for 1,500 steps" per round after 10,000 initial steps (§4.1); copy "continue training M_r … for 500 steps" (§4.2).
- Schedule: +1 digit / +1 length / +1 hop per round, 50,000 new examples per round (§4.1, §4.2, §6.3.1). Reverse addition 1–16 → "up to 100 digits" (Fig 3); copy/reverse 1–10 → "over 120 after approximately 100 self-improvement rounds" (§4.2); forward addition 1–10 → "up to 75" over 60 rounds with length filtering (Fig 10); maze hops 9 → "up to 30 hops" with majority voting (§6.3.1).
- Filtering (§5): "OOD results are often Short" — wrong outputs drop steps (Fig 6). Relative length filter drops outputs shorter than the batch max by more than a threshold (§5.2; for multiplication "more than 10 tokens", App.). Majority vote over k independently seeded models: 5×6 multiplication data "accuracy increasing from an average of 31% to 93.3%" (Fig 7). Oracle move-validity verifier works; majority vote "performs comparably to verification-based filtering" (§6.3.3).
- Without filtering: maze "self-generated training data degrades … leading to a collapse" (§6.3.1). Error avalanche (§8.1). Structured noise (dropped/perturbed digits) is "more harmful than uniform noise" and later rounds tolerate more noise (§8.2). More low-quality self-data hurts: "5× more self-improvement data per round performs even worse" (App. B.2.2).
- Accelerated schedule (§7.2): sample all difficulties above 99% accuracy; multiplication 5×5→10×10 in 19 rounds instead of 41 — but "leverages information about test set accuracy" (§7.2).
- Seed variance: "Even when trained on identical training data, models exhibit substantial performance differences in extrapolation" (App. Fig 29–30); majority vote works better when each seed generates its own data ("model diversity … may be important", App. B.2.2).
## Evidence and limitations
Arithmetic/string/maze only; greedy single-answer tasks; the model never generates inputs (§9). Most curves 1–5 seeds. The 99%-accuracy "safe range" assumes near-perfect in-range accuracy; our rung solve rates are far lower. Appendix hyperparameter tables not read.
## Connections and questions
B1 and B3. Close to our EI+ladder but with differences: (a) Lean replaces majority vote, so label noise is ~0 for us — their main failure mode (avalanche from wrong labels) does not apply; ours is coverage, not correctness. (b) They cumulate all rounds' data with 50% on the newest; (c) +1 difficulty per round, advancing only while accuracy is ~99%; (d) many rounds (60–100). Transferable idea: the "short output" signature — check whether our failed long attempts are premature `exact`/closures (drop-step errors) — and a cross-seed ensemble (their majority vote) as a remedy for B3: pool EI data from several Stage-1 seeds. Smallest test: EI on T1 with data from 3 seeds pooled vs per-seed, same attempts.
