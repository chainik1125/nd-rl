---
written_on: 2026-10-19
written_by: "agent:claude"
papers:
  - openaiLearningReasonLLMs2024
  - openaiIntroducingO3O4mini2025
  - xaiGrok42025
  - amodeiDeepSeekExportControls2025
  - epochHowFarCanReasoning2025
  - epochNotesGPT5Training2025
  - epochWhyGPT5Used2025
  - epochMostOpenAIsCompute2025
  - ordHowWellDoes2025
  - khatriArtScalingReinforcement2025
  - xieLongLiveShort2026
  - cursorIntroducingComposer152026
  - reflectionIntroducingBeam2026
---

# Frontier RL compute relative to pretraining, 2024–2026

Paper: [@epochHowFarCanReasoning2025], [@ordHowWellDoes2025], [@xaiGrok42025] and the other keys above (all proposed,
none in `references.bib`).
Source:
- lab posts: OpenAI o1 (2024-09-12) and o3 (2025-04-16), both via web.archive.org; xAI Grok 4 (2025-07-09); Cursor
  Composer 1.5; Reflection Beam; OpenAI's August 2026 pacing post;
- Epoch AI posts and database, fetched 2026-10-19;
- tobyord.com (2025-10-20);
- arXiv 2510.13786v1 and 2507.06261v6.

Every figure is in [the ledger](../ledger/D_frontier.tsv); the dated table and ranges are in
[the summary](../ledger/D_frontier_summary.md).

## Learnings

**What is known (grades A and B) is relative or qualitative.** No frontier lab has published an RL/pretraining FLOP
ratio.
- OpenAI: for o3 it "pushed an additional order of magnitude in both training compute and inference-time reasoning".
- xAI: Grok 4's RL ran "at pretraining scale", the closest statement of a ratio (about 1).
- Amodei (January 2025): "the amount being spent on the second, RL stage is small for all players".
- OpenAI in 2026: "Our largest planned frontier RL run remains on hold".
- Cursor, not a frontier lab: "The compute used in our post-training of Composer 1.5 even surpasses the amount used to
  pretrain the base model".
- Reflection: Beam's RL GPU-time was at least 1.7x its pretraining GPU-time (my arithmetic from stated GPU counts).

**What is only estimated (grades C and D).**
- Epoch: "In early 2025, RL compute was small—maybe 1-10% of pre-training." Its models put the 2024 o-series at 1 to
  30 %, Grok 4 at 10 to 100 %, and GPT-5 at 10 to 200 %.
- Ord: the first reasoning models cost about 1.01x the base model's training, and the big three were "not far off"
  parity by October 2025.
- SemiAnalysis (D, no method): Anthropic and OpenAI put 42 % of fleet compute into post-training/RL vs 23 % into
  pretraining in 1Q26, and 55 % vs 7 % by 4Q26E.

**Trajectory (my estimate).** *R* (RL FLOP / pretraining FLOP):
- 2024: about 0.01 to 0.1;
- H1 2025: about 0.1;
- H2 2025: 0.3 to 1 (xAI about 1);
- 2026: about 1 to 2 (range 0.3 to 5, grade D).

That is one to two orders of magnitude, mostly in 10x steps per generation (o1 to o3, Grok 3 to Grok 4).

## Evidence and limitations

- **Three currencies get conflated:** relative scale-ups, GPU-time, and fleet shares.
  - Fleet shares include experiments. Epoch puts $4.5B of OpenAI's $5B of 2024 R&D compute outside final runs.
  - RL GPU-time is mostly rollout inference. Beam ran 3.9 to 5.4 inference GPUs per trainer GPU, so its trainer-side
    GPU-time was at least ≈ 0.27-0.35x its pretraining GPU-time (pretraining took under four weeks).
- **The Grok 4 1:1 chart is seen only through Ord's reproduction.** No livestream transcripts were found.
- **Estimates disagree by about 5x.** Epoch put R1-Zero's RL at about 20 % of V3 pretraining (FLOPs, before DeepSeek's
  disclosure). DeepSeek's own GPU-hours give 3.6-3.8 % (ScaleRL quotes 3.75 %), and a token count gives 0.3-1.5 % in
  FLOPs (`../REVIEW.md`, Q3).
- **Pretraining also kept scaling.** Amodei said in 2026 that it is "continuing to give us gains", and fresh large
  pretrains lower *R*.

## Connections and questions

- **What would change the range:**
  - a system card giving RL and pretraining compute side by side;
  - Grok 4's chart with axis values;
  - Epoch estimates for 2026 frontier models (none found);
  - SemiAnalysis's Tokenomics method.
- **Academic RLVR sits far below frontier practice.** Sholto Douglas (Anthropic) said academic RL compute is not
  "anywhere comparable to the amount of compute that was used in the base models". Compare
  [our runs](../our_runs_table.md).
