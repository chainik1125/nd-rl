---
written_on: 2026-11-25
written_by: agent:claude
---

# Screened papers (elicitation metrics for Olmo, 2026-11-25)

**Read levels:**
- **full**: read by me in full text.
- **full, in part**: tool-extracted passages from the full text, sections as listed.
- **abstract**: abstract only.
- **snippet**: search-result summary only; the paper was not opened.
- **earlier review**: cited through a note in an earlier review on `origin/dan`, not re-read.

| # | id | title (first author) | read level | keep / drop, and why |
|---|---|---|---|---|
| 1 | PMLR 306:26127–26151 | Bits That Count: Quantifying and Predicting Capabilities of Language Models (Donoway) | full (main text, App. A, D, G, H, I.1–I.6) | keep: the EDL companion paper. Capacity thresholds, signatures, few-shot validation; basis for R1, R2 and the EDL-weakness section |
| 2 | NeurIPS 2025, OpenReview Dkgx2pS4Ww | Quantifying Elicitation of Latent Capabilities in Language Models (Donoway) | abstract (also in `references.bib`) | keep: 10–100 random parameters give up to 50 % gap recovery; the logistic fit |
| 3 | 2601.04728 | Excess Description Length of Learning Generalizable Predictors (Donoway) | earlier review | keep: theory of EDL (already in `references.bib`) |
| 4 | 2602.04118 | Learning to Reason in 13 Parameters (Morris) | full, in part (§3, 4, 6, 8, 9) | keep: RL needs 100–1000× smaller updates than SFT; parameter-budget elicitation |
| 5 | 2509.06608 | Small Vectors, Big Effects: RL-Induced Reasoning via Steering Vectors (Sinii) | full, in part (§2–3, 5, 7–9) | keep: per-layer RL-trained bias matches full RL on math |
| 6 | 2505.18706 | Steering LLM Reasoning Through Bias-Only Adaptation (Sinii) | abstract | keep: the original bias-only RL result |
| 7 | 2510.00553 | On Predictability of Reinforcement Learning Dynamics for LLMs (Cai) | full, in part (first 100k chars) | keep: rank-1 of ΔW recovers RL gains; AlphaRL extrapolation. Internal inconsistencies noted |
| 8 | 2404.16792 | Weak-to-Strong Extrapolation Expedites Alignment / Model Extrapolation (Zheng) | abstract + snippet | keep (secondary): weight extrapolation without training |
| 9 | 2603.22446 | Sparse but Critical: Token-Level Analysis of Distributional Shifts in RLVR (Qwen Pilot Team) | full, in part (§1–5, Table 1) | keep: per-position JS; cross-sampling recovers the RL gain with 4–8 % of tokens |
| 10 | 2605.06241 | Rethinking RL for LLM Reasoning: Sparse Policy Selection, Not Capability Learning (Akgül) | full, in part (§3, 6, Tables 1–3) | keep: RL tokens are in the base top-5; mean rank about 2; internal inconsistencies noted |
| 11 | 2506.01939 | Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL (Wang) | abstract | keep: forking tokens; RLVR follows the base entropy pattern |
| 12 | 2507.19457 | GEPA: Reflective Prompt Evolution Can Outperform RL (Agrawal) | full, in part (setup, Tables 1–2) | keep (lower rank): IFBench prompt optimisation; small gains |
| 13 | 2410.14516 | Do LLMs "know" internally when they follow instructions? (Heo) | full, in part (§2–4, 6.3, Tables 1, 4) | keep (lower rank): IF compliance probe and steering; small, does not generalise across types |
| 14 | 2510.14901 | Reasoning with Sampling: Your Base Model is Smarter Than You Think (Karan) | abstract | keep (lower rank): training-free sharpening nearly matches RL |
| 15 | 2601.21590 | Scalable Power Sampling (Ji) | abstract | keep (lower rank): about 10× cheaper than MCMC; matches one-shot GRPO |
| 16 | 2606.09926 | entropy-guided power sampling (Guo) | snippet | drop: not opened; a variant of #15 |
| 17 | 2410.14086 | In-context learning and Occam's razor (Elmoznino) | abstract + §2.4 heading | keep: the ICL loss is a prequential code; basis for R2 |
| 18 | 2210.07931 | Sequential Learning of Neural Networks for Prequential MDL (Bornschein) | abstract | keep: online learning with rehearsal beats block-wise; bears on our single-pass design |
| 19 | 2512.07810 | Auditing Games for Sandbagging (UK AISI, FAR.AI, Anthropic) | snippet (AISI blog summary) | keep as criticism, unverified: training-based elicitation also improved benign models |
| 20 | 2412.01784 | Noise Injection Reveals Hidden Capabilities of Sandbagging LMs (Tice), v3 | abstract | drop as a metric: needs a suppression mechanism; about 1 H200-h per (noise, seed) |
| 21 | 2506.10139 | Unsupervised Elicitation of Language Models | snippet (title) | drop: label-free elicitation, not a per-checkpoint measure |
| 22 | 2504.15777 | Tina: Tiny Reasoning Models via LoRA | snippet (title) | drop: subsumed by #4 |
| 23 | 2611.03627 | Not All Reasoning Generalizes: Learning Generalizable Reasoning Strategies in RLVR | snippet | drop: bits per token of strategies, not an elicitation measure |
| 24 | 2510.27044 | (title not opened) RLVR reinforces superficial heuristics | snippet | drop: not opened |
| 25 | 2602.09591 | length penalties and reasoning acquisition | snippet | drop: not opened, off-topic |
| 26 | arXiv id not confirmed (deeplearn.org listing) | Questioning Instrumental Convergence in RLVR: Checkpoint Evidence from OLMo-3 | snippet | drop: different property. Notable as another use of Olmo-3-7B-Think's 13 intermediate RLVR checkpoints |
| 27 | LessWrong post | SFT also drives safety eval results in Olmo 3.1 | snippet | drop: safety evals, not elicitation |
| 28 | 2405.19550 | Stress-Testing Capability Elicitation With Password-Locked Models (Greenblatt) | earlier review | context: fine-tuning as the strongest elicitation |
| 29 | 2502.02180 | The Elicitation Game (Hofstätter) | earlier review | context |
| 30 | 2410.08827 | Do Unlearning Methods Remove Information from LM Weights? (Deeb) | earlier review | context: control-model logic for relearning |
| 31 | 2003.12298 | Information-Theoretic Probing with MDL (Voita) | earlier review | context: block-wise online code |
| 32 | 1802.07044 | The Description Length of Deep Learning Models (Blier) | earlier review | context |
| 33 | 2012.13255 | Intrinsic Dimensionality Explains the Effectiveness of LM Fine-Tuning (Aghajanyan) | earlier review | context for item 1 |
| 34 | 2312.01552 | URIAL (Lin) | earlier review | context for items 2–3 |
| 35 | 2509.04259 | RL's Razor (Shenfeld) | earlier review | context: KL split into selection and reshaping; EDL control (b) |
| 36 | 2205.11275 | RL with KL penalties is better viewed as Bayesian inference (Korbak) | earlier review | context: success-conditioned optimum |
| 37 | 2505.11711 / 2511.08567 | RL finetunes small subnetworks (Mukherjee) / The Path Not Taken (Zhu) | earlier review | context for item 4 |
| 38 | 2510.07364 / 2507.12638 | Base Models Know How to Reason (Venhoff) / Reasoning-Finetuning Repurposes Latent Representations (Ward) | earlier review | context for item 7 |
| 39 | 2504.02922 / 2510.13900 | Crosscoder sparsity artefacts / Narrow finetuning leaves readable traces (Minder) | earlier review | reason for dropping crosscoders; activation-difference lens folded into item 7 |
| 40 | 2504.13837 | Does RL Really Incentivize Reasoning Capacity? (Yue) | earlier review | context: support versus reachability |
| 41 | 2405.09673 | LoRA Learns Less and Forgets Less (Biderman) | earlier review | context for EDL weakness 7 |

Counts: 27 rows screened in this review, plus 14 cited from earlier reviews. One was read in full by me (#1). Seven were read in
part from full text (#4, 5, 7, 9, 10, 12, 13). #17 was read as its abstract plus the §2.4 heading. The rest are abstracts or snippets.
