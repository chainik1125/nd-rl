---
written_on: 2026-10-03
written_by: agent:claude
status: proposed for Dan (2026-10-03); not queued
---

# Proposal 24: verifier-guided sampling in the proof-state environment (cap 6), a replication of Robbie's test-time scaling

## Why

Robbie's test-time scaling study (`robbie-experiments`, `experiment-summaries/2026-09-29-test-time-scaling`) used
combined-pipeline **c005**: his 9.56M recipe, cap-6 `train.jsonl`, Leon's EI with guided RL sampling, and **whole-proof**
`lean_seq`. His biggest win was the simplest method:
- **Resample a line when it is invalid,** up to 10 rejections per proof, instead of letting the sample fail.
- On his 1,177-theorem holdout: pass@1 rose from 31.2 % to 57.4 %. At about 64 plain samples' worth of tokens, the
  solve rate rose from 80.8 % to 84.3–85.6 %, and proofs over 10 lines from 40.7 % to 51–54 %.
- SMC was close behind.
- Beam and best-first search were **worse** than plain sampling at matched tokens. His learned value head added
  nothing, and our `mcts-a` agrees: search does not beat sampling after RL.

**Our proof-state environment does not have this.** `state_env.py`: "There is **no symbolic type check** -- a wrong
term is caught by Lean on the finished proof." The environment checks grammar, names in scope, and that `exact`
matches the goal. A logically wrong step (`have n3 : Q := n1 n2` where `n1` does not have type P → Q) passes and
wastes the whole attempt when Lean rejects the finished proof. A structurally invalid step ends the attempt at once.
Either way, one bad draw kills an attempt that may have been one step from success. `trajectory` found that RL-only
proofs hinge on one or two improbable steps, which is where retrying should pay off.

## What changes

1. **A per-step logical check that is sound with respect to Lean.** A step is rejected only if Lean would reject it.
   This keeps "Lean alone decides": the check only prunes, and Lean still judges the finished proof.
   - **Reference:** Lean type-checks the step in its context (`example (hyps…) : F := term`), batched once per
     sampling wave across all live attempts.
   - **Fast path:** a small Python type checker for the `lean_seq` term fragment (application, `.1` / `.2`, `.elim`
     including `Not.elim`, `Or.inl` / `Or.inr`, `Or.elim`, `fun` boxes, `Classical.byContradiction`).
   - Use the fast path **only after** it agrees with Lean on ≥ 50,000 steps (accepted and rejected) with **0 false
     rejects**. False accepts are harmless: Lean catches them at the end. Never use `nd_verify` here; it rejects
     Lean-valid steps such as `Not.elim`.
2. **Guided sampling, without replacement:** when a step fails the check (structurally or logically), draw a new
   step from the same state, up to R rejections per attempt (R = 10, Robbie's setting).
   - **Redraws exclude every step already rejected at that state** (Dan, 2026-10-03: "re-draw w/o replacement to
     avoid wasted work").
   - Use exact sequence-level sampling without replacement (UniqueRandomizer, Shi et al. 2020). At each decoding
     position, add log(1 − R(prefix + token)) to the logits, where R is the probability of going on to write a
     rejected step exactly. The redraw then comes from the model's distribution with the rejected steps removed.
     **Reuse Robbie's implementation:** `_swor_shift` in `code/experiments/current/test_time_scaling/engine.py` on
     `robbie-experiments`.
   - **Why it matters more here than for Robbie.** His engine (without replacement) and Leon's original sampler,
     which appears to redraw with replacement, scored almost the same at k 64 (85.6 % vs 85.5 %). But after RL our
     policy is sharp: if its most likely step at a hard state is wrong, a with-replacement redraw mostly repeats it.
     Report how often a with-replacement redraw would have repeated a rejected step, from the same rejection logs.
   - **Scope of the forbidden set.** For pass@k **evaluation**, keep it per attempt, so attempts stay i.i.d. and the
     unbiased estimator holds. For **training rollouts** (Part B), share it across attempts at the same state. In
     the SN arm the environment assigns names canonically, so states recur across attempts at a theorem. Cache
     check verdicts per (state, step) in both cases. That saves Lean calls without changing the sampling.
   - **SMC:** N particles; a failed step gets weight 0; survivors are resampled back to N.
   - Count every sampled token, rejected ones included, as compute.

## Design

**Models:** `trajectory-cap6`'s three cap-6 seeds (Robbie's recipe, proof-state environment): the end-of-pretraining
checkpoints and the r8 checkpoints.

**Part A, test time (no training):** at matched sampled tokens per theorem, compare:
- plain;
- guided with the structural check only;
- guided with the logical check;
- SMC.

| pool | why |
|---|---|
| **Robbie's holdout:** the holdout half of `data/ladder/transfer.jsonl`, 1,177 theorems, 7–14 lines, 114 over 10 lines | Rebuild the split from his harness, so the comparison with c005 is direct |
| textbook72 | |
| `trajectory-cap6`'s group C | |

Report pass@1–64 (and 256), proofs over 10 lines separately, and tokens and wall-clock per method.

**Part B, guided EI against plain EI (training):** 8-round T1 ladders from the end-of-pretraining checkpoints (3 seeds)
with **guided** rollouts.
- The **plain-EI arm is `trajectory-cap6`'s ladders**, reused, at the same targets, attempts, replay and fine-tune
  steps.
- Training is on accepted proofs only, as now. Rejected steps are never trained on, positively or negatively.
- **Evaluate both policies with plain sampling and with guided sampling.** The question is whether guided EI teaches
  the policy more, or only helps the sampler. Robbie's blog found the guided RL recipe *lowered* unguided pass@1
  (43.8 % → 30.1 %) while raising pass@64.

## Pre-registration sketch

- **Part A:** guided with the logical check beats plain at matched tokens, most on proofs over 10 lines and on
  group C. The structural-only version gains little, because structural failures are rare for a trained state
  model; measure that rate first.
- **Part B:** guided EI's policy, sampled plainly, ends at least level with plain EI and solves more long theorems.
- **Falsifier:** at matched tokens, the logical-check sampler is within noise of plain on proofs over 10 lines.

## Cost

- **Code:** one executor session: the type checker and its Lean agreement test, the guided sampler, SMC.
- **Pods:** Part A ≈ $6; Part B ≈ $15–20 (3 cap-6 ladders, guided rollouts with per-wave checks, plus read-outs).
- **Total ≈ $25–30.** Part B could be gated on Part A, as with `mcts-a` / `mcts-b`.
