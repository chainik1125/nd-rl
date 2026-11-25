---
written_on: 2026-11-25
written_by: agent:claude
---

# Papers to add to Zotero (elicitation metrics, 2026-11-25)

`references.bib` is generated from Zotero; do not edit it. Add with
`uv run --script code/tools/literature.py add <id>`. The citekeys used in `notes/` are proposals; pin or replace them on import.

Already in `references.bib`: `donoway2026excessdescriptionlengthlearning` (2601.04728) and
`donowayQuantifyingElicitationLatent` (the NeurIPS 2025 paper; no URL or venue in the entry, so consider adding
https://neurips.cc/virtual/2025/poster/119197).

| priority | id | title | why | proposed key |
|---|---|---|---|---|
| 1 | PMLR 306:26127–26151, https://proceedings.mlr.press/v306/donoway26a.html (no arXiv) | Bits That Count: Quantifying and Predicting Capabilities of Language Models (Donoway, Joren, DeWeese, Perez, Schulman, Roger, Leike; ICML 2026) | the empirical EDL paper: capacity thresholds, signatures, the multi-epoch residual | donoway2026bitsthatcount |
| 1 | arXiv:2603.22446 | Sparse but Critical: A Token-Level Analysis of Distributional Shifts in RLVR Fine-Tuning of LLMs (Qwen Pilot Team; ICLR 2026) | token-level footprint and cross-sampling (items 3, 8) | qwenpilot2026sparsebutcritical |
| 1 | arXiv:2602.04118 | Learning to Reason in 13 Parameters (Morris et al. 2026) | parameter-budget elicitation; RL against SFT update size (items 1, 9) | morris2026tinylora |
| 1 | arXiv:2410.14086 | In-context learning and Occam's razor (Elmoznino et al. 2024) | ICL loss = prequential code (item 2) | elmoznino2024iclandoccam |
| 2 | arXiv:2510.00553 | On Predictability of Reinforcement Learning Dynamics for Large Language Models (Cai et al.; ICLR 2026) | rank-1 and extrapolated weight deltas (item 4) | cai2025predictabilityrldynamics |
| 2 | arXiv:2605.06241 | Rethinking RL for LLM Reasoning: It's Sparse Policy Selection, Not Capability Learning (Akgül et al. 2026) | the reranking reading of RL (item 3) | akgul2026sparsepolicyselection |
| 2 | arXiv:2509.06608 | Small Vectors, Big Effects: A Mechanistic Study of RL-Induced Reasoning via Steering Vectors (Sinii et al. 2025) | per-layer bias as a tiny adapter (items 1, 9) | sinii2025smallvectors |
| 2 | arXiv:2505.18706 | Steering LLM Reasoning Through Bias-Only Adaptation (Sinii et al. 2025) | original bias-only RL result | sinii2025biasonly |
| 2 | arXiv:2506.01939 | Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM Reasoning (Wang et al. 2025) | forking tokens (item 3) | wang2025highentropyminority |
| 2 | arXiv:2210.07931 | Sequential Learning of Neural Networks for Prequential MDL (Bornschein, Li, Hutter 2022) | estimator quality of online codes (EDL weakness 3) | bornschein2022prequentialmdl |
| 3 | arXiv:2507.19457 | GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning (Agrawal et al.; ICLR 2026) | prompt-optimisation gap on IFBench (item 5) | agrawal2025gepa |
| 3 | arXiv:2410.14516 | Do LLMs "know" internally when they follow instructions? (Heo et al.; ICLR 2025) | IF compliance probe and steering (item 7) | heo2025instructionfollowingdimension |
| 3 | arXiv:2510.14901 | Reasoning with Sampling: Your Base Model is Smarter Than You Think (Karan & Du 2025) | decoding-only elicitation (item 6) | karan2025reasoningwithsampling |
| 3 | arXiv:2601.21590 | Scalable Power Sampling (Ji et al. 2026) | cheaper power sampling (item 6) | ji2026scalablepowersampling |
| 3 | arXiv:2404.16792 | Weak-to-Strong Extrapolation Expedites Alignment (Zheng et al. 2024) | weight extrapolation (item 4) | zheng2024expo |
| 3 | arXiv:2412.01784 | Noise Injection Reveals Hidden Capabilities of Sandbagging Language Models (Tice et al.; v3) | dropped metric; cite if noise injection comes up | tice2024noiseinjection |
| 3 | arXiv:2512.07810 | Auditing Games for Sandbagging (UK AISI, FAR.AI, Anthropic, 2025) | criticism of training-based elicitation. Not read: verify before citing | aisi2025auditinggames |

Adding a paper here is not evidence that it was read; see the read levels in `screened.md`.
