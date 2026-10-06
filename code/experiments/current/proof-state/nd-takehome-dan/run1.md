# Run 1: natural deduction in Lean, and novelty by model scale

Run 2026-09-17/18; numbers in `numbers.md` §Round 2 — Run 1. **Questions:** does an independent checker agree with `nd_verify`; does writing the same proofs as Lean 4 change what a pretrained code model can do in context; and can "the smallest model that proves it in Lean" serve as a novelty measure?

**Step 1 — `nd2lean.py`.** Deterministic translation of spec.md proofs into Lean 4 (core, no Mathlib): atoms as `Prop` variables, premises as hypotheses, each line a `have`, boxes as typed lambdas, `Or.elim`, `False.elim`, `Classical.byContradiction`; the verifier's structural rules mirrored where Lean would not enforce them. **0 disagreements** with `nd_verify` on 21 examples, 5,000 held-out, 3,000 RL targets, 1,638 transfer, 10,000 training proofs, the 36 validation references and 75,085 RL-found transfer proofs; **4,012 / 4,012 corrupted proofs rejected by both**. The project has a second checker.

**Step 2 — in-context v0 (Qwen3-Coder-30B-A3B-Instruct, vLLM on one A100, no hosted API).** 20 worked examples, one fresh theorem, greedy + 8 samples at T = 0.7; three surface forms of the same theorems and the same examples (Lean 4, the take-home tokens, plain-English rule names as control); validation-36 + 200 transfer theorems stratified by length; 5 example-set draws.

| form | greedy | pass@8 | val-36 `> 6` greedy |
|---|---|---|---|
| Lean 4 | **0.563** [0.53, 0.59] | **0.675** [0.65, 0.70] | 0.458 |
| tokens | 0.258 [0.23, 0.28] | 0.326 [0.30, 0.35] | 0.158 |
| English | 0.292 [0.27, 0.32] | 0.386 [0.36, 0.41] | 0.100 |

Paired Lean − tokens over 236 theorems: **+0.31 [0.26, 0.35] greedy, +0.35 [0.30, 0.40] pass@8**. The gain is largest on the shortest theorems (+0.34 at 7 lines, +0.15 at 16), the opposite of the proposal's prediction. The English control sits with the tokens: the cost is the calculus-as-string, not the rule names.

**Step 3 — novelty by scale.** Qwen3 0.6B–32B (instruct, thinking off), same Lean prompt, k = 16 at T = 0.7, on 40 depth-3 f = 0 theorems, 40 required-reductio theorems and the 26 transfer theorems with 9-line RL proofs. Nothing below 4B proves them; the median smallest size is **8B for all three classes**; the fraction proved at 4B / 8B / 14B / 32B is 0.30 / 0.65 / 0.93 / 1.00 (depth-3), 0.10 / 0.53 / 0.65 / 0.85 (reductio), 0.35 / 0.69 / 0.96 / 0.96 (9-line transfer). The classes separate only in the tail: the rule sequence keeps 15 % of its theorems beyond 14B, the structural class none. The ordering agrees weakly with Phase-1's base log-probabilities (Spearman −0.24 on the 26 transfer theorems) and ranks depth-3 proofs easier than their base probability (< 10⁻⁵) suggested, as the proposal predicted. A coarse measure (four usable levels), but model-independent.

![forms](figures/run1_forms.png)
![scale](figures/run1_scale.png)
