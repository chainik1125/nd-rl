---
written_on: 2026-10-02
written_by: agent:claude
papers:
  - donoway2026excessdescriptionlengthlearning
---

# Excess Description Length (EDL): an information measure of "teach vs elicit"

Paper: [@donoway2026excessdescriptionlengthlearning]
Source: https://arxiv.org/abs/2601.04728v1 (v1 reviewed; html text)

## Learnings
Paper claims:
- Setting: "whether fine-tuning elicits latent capabilities or teaches new ones" (abstract). EDL is defined through prequential coding. It is the gap between the bits needed to encode the training labels online (each label scored by the model before it is updated on that label) and the residual code length under the final model (abstract).
- Properties: EDL is non-negative in expectation, converges to surplus description length, and bounds the expected generalisation gain (abstract). Random labels give EDL near zero. Rare-input structure contributes little to expected generalisation. "format learning creates early transients distinct from capability acquisition" (abstract, Prop. 5.5).
- The predicted signatures (Sec. 7.1.1, "validate empirically in a companion paper"): "Elicitation shows monotonically decreasing EDL/token with dataset size; teaching shows an initial increasing phase". Pre-teaching a skill converts teaching into elicitation, "reducing information thresholds by ~10–100x".
- Scope (Sec. 7.2): "our analysis focuses on supervised fine-tuning. Extensions to reinforcement learning … require additional development."

Our interpretation: EDL replaces our post-hoc log p threshold with a quantity that has a natural zero point and a shape test (decreasing EDL per token for elicitation, rising then falling for teaching). It also tells us to separate format bits from capability bits.

## Evidence and limitations
- Read the abstract, Sec. 1, 5 (format model) and 7 (implications, limitations). Did not check the proofs or the toy-model figures.
- The empirical claims are only a preview of a companion paper that I did not find or verify. They are UNVERIFIED as evidence.
- The theory is for SFT on fixed labels. In expert iteration the "labels" are the model's own Lean-accepted samples. By construction these are already likely under the sampling model, so prequential code length on self-generated data is biased low. On-policy EDL needs a correction, or we should measure it on fixed external targets (the reference proofs).
- EDL depends on the training algorithm (Sec. 7.2).

## Connections and questions
- Nearly free for us: the `trajectory` run already has teacher-forced log p of fixed proofs at every checkpoint and every RL round. Prequential code for round r is the sum over accepted proofs of −log p under the round r−1 model. Residual is −log p under the final model. Their difference per round, divided by tokens, gives an EDL curve for the ladder. Compute it separately for A and B targets, and for the reference proofs, which are off-policy and so avoid the self-sample bias.
- Causal intervention analogue: "Pre-teaching a skill (e.g., multiplication via operator notation) converts a teaching task to an elicitation task" (§7.1.1) *[reviewed: V6]*. Our cap-6 vs cap-12 pretraining is a natural version of this. Predict a teaching-like signature (an initial rise in EDL) on B theorems at cap 6 whose shortest proof is longer than 6 lines, and an elicitation-like signature at cap 12.
- Compare zhang2025-interplay-pre-mid-rl.md and fgx-rl-composition-2025.md, which also control pretraining exposure. EDL is the measurement and those papers supply the design.
