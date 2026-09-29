---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# RL on from-scratch models collapses onto one pretraining "style"
Paper: Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining (Zhao et al., 2025)
Source: https://arxiv.org/abs/2504.07912v2 (v2 reviewed)
## Learnings
- §2.1: 150M and 1B OLMo models were pretrained from scratch on FineMath-3+ and Algebraic-Stack plus controlled repeats of TinyGSM (code), OpenMathInstruct1 (code) and OpenMathInstruct2 (natural language). Each dataset has a detectable format, so the share of outputs in each "style" can be tracked.
- §3.1 / Fig. 2: under PPO the model "rapidly converges to producing outputs that follow the format of a single data distribution", within the first epoch. That shift coincides with the largest pass@1 gain. Majority@64 rises by about 5%, while pass@64 "declines towards the end of training". Fig. 3: a higher KL coefficient (0.01 vs 0.001) keeps some minority-format outputs and keeps pass@64 stable.
- §3.2 / Fig. 4: the winning style is not always the most common one or the most accurate one at initialisation. With 4× OMI1, a style at 28% initial share wins. With 8× OMI1, the less accurate style wins and accuracy then drops, which they call "a failure mode of RL fine-tuning".
- §3.4: the winner depends on scale. The 150M models pick TinyGSM code; the 1B models pick natural language.
- §3.5 / App. F.2: GRPO shows the same collapse but is less stable. EI (k = 64, dedupe, retrain from the original base each round) "consistently underperforms PPO" and shows only "a mild shift". Example: with 8× TinyGSM, PPO reaches ~60% vs EI "below 45%" after 3 rounds. They attribute the gap to retraining from the fixed base.
## Evidence and limitations
- Evidence is per-mixture curves (figures), mostly single runs. I did not find seed replicates in the text I read, so style selection may be seed-dependent, as in our B3. The theory in §3.6 was not read.
## Connections and questions
- B2/B3: our Stage-1 data mixes generator styles and schemas, and our seeds are bimodal. Echo Chamber predicts that EI/GRPO will lock onto one mode early, and that the choice can be the wrong one.
- Measurement: tag every accepted proof by generator style/schema family and by proof-shape features. Plot per-round shares against the base shares, per seed. Test whether seed bimodality is "which style won".
- For the proposed ReST-EM retrain-from-base arm, their F.2 predicts slower collapse and lower pass@1 but better diversity. We should add style-share curves to that comparison.
- A KL/SFT anchor (already proposed for GRPO) is predicted to preserve pass@k (Fig. 3).
