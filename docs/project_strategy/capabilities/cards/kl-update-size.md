---
written_on: 2026-10-05
written_by: agent:claude
---

> Reading copy of run `capability-defs`'s `capability_defs/cards/kl-update-size.md` (fork branch `dan_capability-defs`, DONE
> `645aa4f0`); reviewer's corrections marked *[review]*. Report: [`../REPORT.md`](../REPORT.md); review:
> [`review_capability-defs.md`](https://github.com/dan-pandori/nd-takehome/blob/da11be78/review_capability-defs.md).

# Card: how far did RL move the model? (KL to the base and the size of the update) — DROPPED as a decision rule; kept as description (critic, §9)

Family D. Slug `kl-update-size`. Notation: `_FRAME.md`.

## 1. Definition, formally

Two views of "how big a change RL made", each with a reference scale.

- **Output space.**
  - Per theorem: KL(π_R(· | t) ‖ π_B(· | t)) on RL's own samples. Estimate it as the mean over R's attempts of
    log π_R(y) − log π_B(y), which needs teacher-forced scores of R's samples, failures included.
  - On the success set it splits into selection + reshaping bits (`sharpen-expand`).
  - Token level (URIAL): the share of positions where R's chosen token is not among the base's top choices.
- **Parameter space.**
  - Relative update norm ‖θ_R − θ_B‖ / ‖θ_B‖ per matrix.
  - Effective rank of ΔW (the number of singular values carrying 90 % of ‖ΔW‖²).
  - Whether a rank-r (LoRA) or d-dimensional random-subspace fine-tune reproduces R's gain (intrinsic dimension,
    Aghajanyan et al.).

**The reference scales** that make sizes interpretable:
- the update pretraining makes over a comparable span (θ_pend − θ_p20000);
- the replay-only control's update (8 rounds of the same fine-tune on pretraining data, `rl-from-ckpt`);
- seed-to-seed differences.

## 2. Decision rule

| verdict | rule |
|---|---|
| **elicited** ("a small update that switches on a behaviour") | RL's gain is reproduced by an update much smaller than the reference scale (low rank, small KL per theorem relative to the selection bits), or the token-level shift touches few positions |
| **created** | the gain needs an update as large or as high-rank as pretraining-scale changes, and no small update reproduces it |
| **neither** | no gain |

**This card is weak by design.** "Small update" is not "old capability": a rank-1 change can add a new skill if the
skill is simple (`sharpen-expand` §6). The rule is offered because Dan's brief asks for it, and the critic is expected
to land.

## 3. Null or floor

- The replay-only control (training without RL proofs) is the placebo update.
- Random reinitialisation of one matrix is the upper reference: any update below it counts as "local".
- The random-weights objection does not arise (no k).

## 4. How to compute it here

- **Parameter space:** CPU, seconds, from local checkpoints (pend, step 1,600 and r8 for all seeds, md5-verified in
  `~/review/trajectory/rv/ck/`; r16 from the bucket) with `~/venv-cpu` torch.
- **Output KL on R's samples:** J3's plain arm stores R's accepted texts with counts, but not its failures. Scoring
  every R attempt is a new job (not run).
- **Low-rank sufficiency:** a LoRA sweep is a new job (≈ 1 GPU-hour); not run here (stated).
- **Result, parameter view** (seed 0, cap 12; `analysis/cd_update.py`, `out/update_size.txt`; 26 weight matrices):

  | update | relative Frobenius norm | effective rank (90 % energy), median share of full |
  |---|---|---|
  | RL, pend → r8 | 0.282 | 0.272 |
  | RL, pend → r16 | 0.384 | 0.336 |
  | replay-only control, pend → c_pend_r8 (no RL proofs) | 0.297 | 0.270 |
  | late pretraining, step 20,000 → pend | 0.809 | 0.533 |
  | seed gap, pend s0 vs pend s1 | 1.426 | 0.555 |

  Cosine between the RL r8 update and the replay-only update: **0.45**. Both are slightly anti-aligned with the
  late-pretraining update (−0.12). **RL's update is as large and as low-rank as the replay-only control's, and shares
  much of its direction.** By size or rank alone, RL cannot be told apart from training on pretraining data, and both
  are far smaller than pretraining's own late steps or a seed change.

## 5. Sensitivity

- **Parameterisation:** norms depend on layer and optimiser. Compare like with like (the same matrices, the same
  number of steps).
- **Precision:** Mukherjee et al.'s "RL updates small subnetworks" sparsity vanishes in float32 (Shenfeld et al.
  Sec. 6, via L3), so sparsity claims need fp32 checkpoints.
- **Temperature and decoding:** affect the output KL, not the parameter view.
- **Representation, renaming:** output KL is per prompt.
- **Seed:** the reference scale must include seed differences.

## 6. Failure modes

- **Small ≠ old.** The size of a change does not say whether the capability existed. A small update can unlock a new
  composition, a large one can only reformat.
- **Many updates achieve the same function:** parameter distance is not functional distance.
- **Output KL on R's samples is dominated by format / length changes,** not the hard step.

## 7. Relations

- Its success-set part is `sharpen-expand`.
- Low-rank sufficiency is a version of `elicit-finetune` (an elicitation budget counted in parameters: Donoway et al.,
  NeurIPS 2025, screened by L2).
- The token-level shift ties to `tf-proof-prob`'s step view.

## 8. Literature anchor

- **Lin et al. 2023 (2312.01552v1):** "77.7% of the tokens are at such unshifted positions" and shifted tokens are 5–7 %
  (Sec. 2.2).
- **Shenfeld et al. 2025 (2509.04259v1):** the forgetting "is determined by the distributional shift, measured as the
  KL-divergence between the fine-tuned and base policy evaluated on the new task" (abstract).
- **Mukherjee et al. 2025 (2505.11711):** RL changes 5–30 % of parameters (L3 note); disputed in fp32.
- **Aghajanyan et al. 2020 (2012.13255):** d90 intrinsic dimension falls over pretraining (L3 note).
- **LoRA Without Regret** (Thinking Machines blog): RL takes in "O(1) bits per episode" (L3 note; blog, quote verified
  by L3).
- Verified in `_claims_L3.md`.

## 9. Critic's verdict

**Strongest argument (critic): the size verdict is set by the optimizer, not by pend.**
- Stage-1 trains the block matrices and head with Muon (orthogonalised steps); EI and the placebo use AdamW. So "RL ≪
  late pretraining" compares two optimizers, and the comparison flips by matrix:
  - blocks: 0.27 vs 0.84;
  - embedding (AdamW in both): r8 0.170 vs 0.203;
  - head: 0.78 vs 0.38.
- The placebo matches RL on each.
- Counterexample: 30 J2 theorems get 0 pend successes in ≥ 17,152 attempts (s0; e.g. `la_transfer_2066`, r8
  252 / 256). RL's update reproduces those gains and is "small", so the rule says "elicited" for all 30, while the
  equally large placebo update solves only 3–4 of them.
- Secondary arguments:
  - The seed gap is the random-relabelling ceiling (matrices orthogonal), not a scale.
  - One model-level number cannot give per-theorem verdicts.
  - "KL small relative to the selection bits" fails both ways.

**My answer: accepted; dropped as a decision rule.** The update table stays as a description: RL's update is the size
and rank of the replay-only update and shares its direction (cosine 0.45). The useful model-level statement is "RL's
weight change is not larger than plain replay training's". The per-theorem questions belong to `sharpen-expand` and
`elicit-finetune`.

