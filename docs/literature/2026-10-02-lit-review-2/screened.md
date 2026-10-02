# Screened papers (lit-review-2), full rows

| # | title | year | id | technique / design | rel. | depth | Q |
|---|---|---|---|---|---|---|---|
| 1 | Does RL Really Incentivize Reasoning Capacity Beyond the Base Model? (Yue et al.) | 2025 | 2504.13837v5 | equal-k pass@k up to 1024; per-problem solvable 2×2; base PPL of RL outputs; distillation contrast | 3 | full (Sec. 1, 3.1, 4.1–4.4, 7) | Q1 |
| 2 | ProRL: Prolonged RL Expands Reasoning Boundaries | 2025 | 2505.24864v1 | >2k steps, KL + reference reset; pass@128; Creativity Index | 3 | full (abs, 1, 2.3, 4, App. A) | Q1 |
| 3 | Excess Description Length of Learning Generalizable Predictors (Donoway et al.) | 2026 | 2601.04728v1 | prequential-coding EDL; elicit vs teach scaling signatures; pre-teaching intervention | 3 | full (abs, 1, 5, 7) | Q1 |
| 4 | The Coverage Principle (Chen, Foster et al.) | 2025 | 2510.15020v2 | coverage profile Cov_N; theory + graph-reasoning from scratch | 3 | full (abs, 1, 2–2.1, figs 1–2) | Q1 |
| 5 | Is a Good Foundation Necessary for Efficient RL? (Foster, Mhammedi, Rohatgi) | 2025 | 2503.07453v2 | theory: coverage lower-bounds RL runtime; token- vs sequence-level coverage | 3 | abstract + Sec. 1 | Q1 |
| 6 | RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs (Wen et al.) | 2025 | 2506.14245v2 | CoT-Pass@K (answer + reasoning checked) | 2 | abstract | Q1 |
| 7 | Self-Improvement in LMs: The Sharpening Mechanism (Huang et al.) | 2024 | 2412.01951v2 | statistical framework for sharpening via sample access | 2 | abstract | Q1 |
| 8 | Cognitive Behaviors that Enable Self-Improving Reasoners (Gandhi et al.) | 2025 | 2503.01307v2 | priming base with behaviours before RL; Qwen vs Llama | 2 | abstract | Q1 |
| 9 | Beyond Pass@k: Breadth-Depth Metrics (Dragoi et al.) | 2025 | 2510.08325v2 | Cover@τ (fraction solved at ≥ τ success rate) | 2 | abstract | Q1 |
| 10 | Spurious Rewards: Rethinking Training Signals in RLVR (Shao et al.) | 2025 | 2506.10947v2 | random/incorrect reward controls; GRPO clipping bias | 2 | abstract | Q1 |
| 11 | Reasoning or Memorization? Data Contamination (Wu et al.) | 2025 | 2507.10532v3 | clean generator (RandomCalculation); reward controls | 2 | abstract | Q1 |
| 12 | BroRL: Scaling RL via Broadened Exploration (Hu et al.) | 2025 | 2510.01180v1 | rollouts per prompt to hundreds; mass-balance analysis | 2 | abstract | Q1 |
| 13 | The Debate on RLVR Reasoning Capability Boundary: Two-Stage Dynamic View (Yao et al.) | 2025 | 2510.04028v1 | theory + experiments; exploitation then exploration stage | 2 | abstract | Q1 |
| 14 | Curriculum RL Can Incentivize Reasoning Beyond the Base Model (Cai et al.) | 2026 | 2606.22317v1 | pass@k-located boundary + teacher guidance + RL | 1 | abstract | Q1 |
| 15 | First return, then explore (Go-Explore) | 2021 | 2004.12919v6 | archive + return + random-action explore + robustify | 3 | full (note) | Q2 |
| 16 | Bridging the Human-AI Knowledge Gap (Schut et al.) | 2023 | 2310.16410v1 | concept discovery in AlphaZero, grandmaster transfer | 3 | full (note) | Q2 |
| 17 | Representation-Based Exploration for LMs (RepExp) | 2025 | 2510.11686v2 | elliptical bonus on LM hidden states | 3 | full (note) | Q2 |
| 18 | Learning to Reason in Large Theories without Imitation (DeepHOL-Zero) | 2019/20 | 1905.10501v3 | zero-RL prover + tf-idf premise injection | 3 | full (note) | Q2 |
| 19 | POPE: Privileged On-Policy Exploration | 2026 | 2601.18779v1 | oracle-solution prefixes guide on-policy RL on hard problems | 3 | abstract + 2 sections | Q2 |
| 20 | Beyond Pass@1: Self-play Variational Problem Synthesis (SvS) | 2025 | 2508.14029v4 | policy synthesises variants of under-performing problems | 2 | sections 1, 5.2, appendix | Q2 |
| 21 | Mastering Chess and Shogi by Self-Play (AlphaZero) | 2017 | 1712.01815v1 | tabula-rasa self-play MCTS, root Dirichlet noise | 2 | targeted grep | Q2 |
| 22 | MuZero | 2019 | 1911.08265v2 | learned-model MCTS | 1 | targeted grep | Q2 |
| 23 | Acquisition of Chess Knowledge in AlphaZero (McGrath et al.) | 2021/22 | 2111.09259v3 | concept probing over training | 2 | targeted grep | Q2 |
| 24 | Intelligent Go-Explore | 2024 | 2405.15143v4 | FM-judged interestingness in Go-Explore | 1 | abstract + intro | Q2 |
| 25 | Unifying Count-Based Exploration and Intrinsic Motivation | 2016 | 1606.01868v2 | density-model pseudo-counts | 2 | targeted grep | Q2 |
| 26 | Exploration by Random Network Distillation | 2018 | 1810.12894v1 | prediction error vs random net | 1 | abstract | Q2 |
| 27 | Agent57 | 2020 | 2003.13350v1 | family of explore/exploit policies + meta-controller | 1 | abstract | Q2 |
| 28 | Curiosity-driven Exploration by Self-supervised Prediction (ICM) | 2017 | 1705.05363v1 | inverse-dynamics feature prediction error | 1 | abstract | Q2 |
| 29 | CDE: Curiosity-Driven Exploration for LLM RL | 2025 | 2509.09675v1 | actor perplexity + critic variance bonus | 1 | abstract + table | Q2 |
| 30 | AlphaEvolve | 2025 | 2506.13131v1 | LLM + evolutionary program database | 2 | abstract + ablation | Q2 |
| 31 | TacticZero | 2021 | 2102.09756v2 | deep RL HOL4 prover from scratch, replay buffer | 2 | abstract + setup | Q2 |
| 32 | RL Post-Training Builds Compositional Reasoning Strategies (Abdulsalam, Patel, Saxe) | 2026 | 2607.07646v1 | rewrite grammar; 12-layer transformer pretrained from scratch on primitive rewrites (rho controls distribution); GRPO vs RFT, outcome reward | 3 | full (note) | Q3 |
| 33 | How RL After Next-Token Prediction Facilitates Learning (Tsilivis et al.) | 2025 | 2510.11495v2 | parity; small GPT-2/Mistral from scratch on short/long CoT mixture with exact p_cot; GRPO/REINFORCE/STaR | 3 | full (note) | Q3 |
| 34 | Benefits and Pitfalls of RL for LM Planning (Wang et al.) | 2025 | 2509.22613v2 | graph path-finding (ER 100 nodes, Blocksworld); 1-layer transformer; SFT vs PG vs Q-learning, theory + experiments | 3 | full (note) | Q3 |
| 35 | Outcome-Based RL Provably Leads Transformers to Reason, but Only With the Right Data | 2026 | 2601.15158v4 | graph traversal; single-layer transformer; PG with outcome reward; theory | 2 | abstract | Q3 |
| 36 | Atomic Skills are the Prerequisite | 2025 | 2512.01970v3 | synthetic biographies; SFT then RL on Qwen-2.5-1.5B (NOT from scratch) | 2 | abstract + grep | Q3 |
| 37 | On the Emergence of Implicit Curriculum in RLVR Learning Dynamics | 2026 | 2602.14872v3 | theory of RLVR on compositional tasks; synthetic + real runs | 2 | abstract | Q3 |
| 38 | Transcendence: Generative Models Can Outperform the Experts That Train Them | 2024 | 2406.11741v4 | chess; transformer from scratch on Lichess games capped at rating 1000/1300/1500; no RL | 2 | targeted full-text | Q3 |
| 39 | SFT Memorizes, RL Generalizes | 2025 | 2501.17161v2 | GeneralPoints card game, V-IRL; pretrained Llama-class VLM, not from scratch | 1 | abstract | Q3 |
| 40 | Physics of LMs Part 2.1: Grade-School Math | 2024 | 2407.20311v1 | synthetic GSM-like (iGSM); controlled pretraining; no RL | 2 | abstract | Q3 |
| 41 | Physics of LMs Part 1: Hierarchical Language Structures | 2023 | 2305.13673v4 | synthetic CFGs; GPT from scratch; probing | 1 | abstract | Q3 |
| 42 | Emergent World Representations (Othello-GPT) | 2022 | 2210.13382v5 | Othello legal-move prediction from scratch; probes/interventions | 1 | abstract | Q3 |
| 43 | Progress Measures for Grokking via Mech. Interp. | 2023 | 2301.05217v3 | modular addition, 1-layer transformer | 1 | abstract | Q3 |
| 44 | The Pitfalls of Next-Token Prediction | 2024 | 2403.06963v3 | path-star graph; teacher-forcing failure | 2 | abstract | Q3 |
| 45 | ALPINE (id check) | 2024 | 2405.09220v3 | graph path-finding theory (base of 2509.22613) | 1 | abstract | Q3 |
| 46 | DreamCoder (id check) | 2020 | 2006.08381v1 | program synthesis, wake-sleep library learning | 1 | abstract | Q3 |
| 47 | Grokking (id check) | 2022 | 2201.02177v1 | algorithmic datasets | 1 | abstract | Q3 |
| 48 | Scalable Power Sampling (Ji et al.) | 2026 | 2601.21590v1 | token-level approximation of the power distribution; training-free | 2 | abstract | Q1 |
| 49 | Power Distribution Bridges Sampling, Self-Reward RL, and Self-Distillation | 2026 | 2605.04542v1 | power distribution = optimum of KL-RL with self log-p reward | 2 | abstract | Q1 |
| 50 | Rethinking Sample Polarity in RLVR (A3PO) | 2025 | 2512.21625v1 | positive samples sharpen, negatives explore; token-level advantage shaping | 2 | abstract | Q1 |
| 51 | Mirage or Method? Model-Task Alignment | 2025 | 2508.21188v2 | counterintuitive RL results (1-shot, noisy reward, negative-only) hold only with high base pass@k | 3 | abstract | Q1 |
