---
written_on: 2026-10-19
written_by: agent:claude
status: approved by Dan 2026-10-19, $15 (queued as `kalmar-complexity` on agent1 after `kalmar-pretrain`)
---

# Proposal 31: how many bits is Kalmár's construction?

**Question (Dan):** what is the minimum description length of Kalmár's constructive proof algorithm, as a handle on
how many bits we would realistically have to instill into a model for it to learn the algorithm? Dan invites pushback:
another measure may describe this better.

**Why one program length is not enough.**
- It depends on the language. For a program of about 1 kB, the interpreter constant is as large as the quantity.
- What has to be instilled is conditional on what the learner already has.
- Describing the algorithm is not executing it: the construction is short, but its proofs average about 109 actions.

So the run measures several quantities and asks which of them predicts where kalmar-pretrain's models actually fail.

| step | what |
|---|---|
| upper bounds | shortest programs in minified Python (raw, compressed, in tokens), a fixed universal language with exact bits, and RASP-L (the per-step policy); the smallest exact decision procedure over the proof-state features; algorithm and rendering reported separately |
| calibration | the same measures for truth-table evaluation, K12 proof strategies, and algorithms with published transformer learnability results |
| absorbed bits | prequential code length from the first epoch of kalmar-pretrain's KAL-D and the K12 control (no pods); optional small block-wise prequential runs (pods) |
| prediction | which measure predicts kalmar-pretrain's per-step-type agreement and errors |
| literature | Kolmogorov and Levin complexity, MDL and prequential coding, RASP-L, simplicity bias, excess description length |

**Out of scope:** how many bits pretraining and RL can each supply. That is a separate, later investigation; this run
reports everything in bits so that it can use them.

**Budget** $15 / 40 pod-hours. **Run** `kalmar-complexity` on agent1, `--effort max`, 72 h hard stop, queued after
`kalmar-pretrain` is recorded. **Brief:** [state-env/BRIEF_kalmar-complexity.md](state-env/BRIEF_kalmar-complexity.md).
