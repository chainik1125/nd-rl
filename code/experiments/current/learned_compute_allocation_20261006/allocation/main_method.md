---
written_on: 2026-10-06
written_by: agent:codex
status: method_frozen_before_main_training
---

# Primary joint allocation experiment

This experiment directly updates the proof generator and its stopping/selection heads
using final selected-proof outcomes. Cost and no-cost conditions start with the same
outcome-blind format initialization and fresh uniform heads. They inherit no selector
or allocation checkpoint. Three independent RL seeds, 101/102/103, are paired across
costs 0 and 0.03 per additional attempt. These are RL seeds, not different pretrained
base-checkpoint seeds.

Each arm receives 200 updates of eight complete episodes from the 1600 MAIN training
theorems. The theorem schedule is independent of variable rollout length and matched
across conditions for each seed. The optimizer is AdamW without weight decay, with
base learning rate 1e-5, head rate 1e-3, and global gradient norm clipping at 1. The loss
is the episode sum of every raw generated-token log probability and voluntary
decision log probability, averaged across episodes with a leave-one-out reward
baseline. There is no entropy bonus, KL reward, discount, correctness-to-stop target,
or proof-length normalization. Reward is selected-proof correctness minus the
condition's cost times the number of additional attempts.

The chosen initialization is the 1,000-step outcome-blind format adapter from the
same Dan round-16 ALiBiGPT checkpoint. Its learning rate was1e-5, clean-token weight 8,
and first-state sampling mixture 50%, using 96 safe MAIN training theorem pools.
Development initialization selection used first-proof retention and mean validity
over sampled eight-attempt paths. This is interface adaptation, not evidence of
learned reasoning from histories. Replay covered 0/1/4 earlier frozen-generator proofs;
deployment can contain up to seven adapted-model proofs. The resulting interface
confound and substantial initial quality decay with history must remain explicit.

Every attempt resets the original theorem's environment. Generator input consists
of earlier effective action traces followed by the current authoritative proof state.
Controller input contains the original theorem and full public environment text
transitions, including implicit ORE binders/closures and the attempted failing action.
EOS-missing actions retain their exact raw tokens, including sampled PAD. No private
correctness, parser verdict, error string, or termination label enters these inputs.

The sampling temperature is 0.8 and sampling/replay use FP32. Each attempt allows
at most 1,024 emitted tokens, 256 per action, and 48 actions. The declared context is
16,384 tokens, extending the base's nominal 1,024-token configuration; a fixed 1,024
token state reserve determines prospective continuation legality without future
sample lengths. There are at most eight attempts. Hard-forced stopping receives
selection likelihood only, with no stopping gradient. Voluntary and forced stops
are reported separately. Teacher likelihoods and first-proof correctness are logged
only as diagnostics after episodes; they do not alter rewards or later observations.

Checkpoint selection compares completed steps 100 and 200 on the frozen balanced 32
development panel, one full generated path per theorem at seed 52006. Primary policy
accuracy integrates categorical stop and index probabilities. No-cost selection
maximizes expected correctness; cost 0.03 selection maximizes correctness minus 0.03
per additional attempt. Ties prefer fewer attempts and then the earlier checkpoint.
No test outcome selects optimizer, checkpoint, readout, or utility coefficient.

Evaluation generates a forced eight-attempt contextual path once and integrates
learned, fixed counts 1/2/4/8, and development-fitted count-only hazard policies on that path.
Private proof outcomes are checked afterward. Controller work for fixed/count-only
controls is charged only at final selection among multiple candidates; singleton
selection is deterministic. Expected policy work is separate from full-path benchmark
expense. Prespecified secondary greedy-index and fully greedy stopping readouts are
reported alongside the primary categorical policy. An independently trained fixed-count 4
control is a prespecified supplementary run: seed 101, 200 updates of eight episodes,
the same format initialization and optimizer, zero cost, and four attempts unless
the declared hard context limit forces earlier termination. It receives generation
and terminal selection likelihood gradients, with no learned stop gradient. Its
completed-step 100/200 checkpoint is selected on the same development panel by
fixed-four expected correctness with categorical index selection, with the same tie
rule. Its unused accept head does not select the checkpoint.

A supplementary fixed-one training control uses seed 101, 200 updates of eight
episodes, the same initialization and optimizer, and zero cost. Its completed-step
100/200 checkpoint is selected by DEV32 fixed-one correctness, with ties preferring
the earlier checkpoint. The unused stop head does not select this control's
checkpoint. Both independently trained fixed-budget controls have one training
seed and therefore provide limited evidence about training variability.

Final test uses the untouched 400 MAIN test theorems after the selected checkpoints
and evaluation hashes enter the [frozen test registry](evaluation_protocol.md).
Two full paths per theorem are prespecified at test seed 62006 before any test
outcomes are inspected. Exact numerical settings, hashes, source snapshots, and commands are
stored in [main_method_freeze.json](main_method_freeze.json) and each raw run folder.
