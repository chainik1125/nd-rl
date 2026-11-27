# M2 (key rank1): rank-1 and extrapolated weight changes, design discourse

## Turn 1 — reader-implementer (Claude Sonnet 5.5), 2026-11-27 01:44 PT

Evidence base: my full-text notes `docs/literature/2026-11-27-elicitation-metrics-fulltext/cai2025predictabilityrldynamics.md` (arXiv v3, 48 pp.) and `zheng2024expo.md` (v5, 17 pp.); page references below are PDF pages of those papers. Feasibility facts in section F come from files I read (named). Nothing was run on a pod; no weights were downloaded.

### 0. What the full read changes for the design

1. Cai's rank-1 model is **not** "the top singular piece of the change". It is sigma_1 u_1 v_1^T rescaled by alpha = ||dW|| / ||dW^(1)|| so that it carries the full update's norm (Eq. 3, p. 3). The paper writes "L2 norm"; only the Frobenius reading makes alpha != 1 (spectral norm gives alpha = 1). Its Table 7 (p. 21) puts the unscaled rank-1 at 19.3 % of the DAPO update ("norm" or "energy": unstated), so alpha is about 2.3-5. The rescale is part of what we must test, and it is what makes a random rank-1 control of equal norm necessary.
2. The paper never says whether embeddings, lm_head and norms are in dW, nor what the rank-1 model carries for them. Its objects ("modules", Fig. 15 p. 35) are the 7 matrix types per layer, same as our 448 matrices.
3. "99.17 %" is a ratio of accuracies (p. 4), with no tabulated source, no seeds and no intervals. The coefficient of AlphaRL is set from the MATH-500 test accuracy of the checkpoints with the fully trained model's score as target (y* = 1; p. 7). Its RL-specificity claim is contradicted by its own App. C and Tables 7-8 (pp. 21, 23): offline DPO keeps 62 % at rank 1.
4. ExPO picks alpha by reward on a dev set (p. 4; App. B p. 15); its own optimum falls with training length (8.0, 2.5, 0.5, 0.3 for 10, 20, 40, 100 % of the steps, p. 5), and it collapses on large deltas (Sec. 4.3, p. 9). It never extrapolates between two aligned checkpoints.

### 1. Questions and the two interventions

- **Q1 (rank-1 keeps t's behaviour).** Does the rank-1 piece of `dpo` -> t, put on `dpo`, give t's behaviour? Primary t = `s2300` (the only checkpoint with the large, known gain: arm-A pass@1 29.0 -> 68.3 %, reach 0.0 -> 69.4 %). t = `s0500` is a second rank-1 adapter only because Q2 needs it.
- **Q2 (extrapolation reaches the final model).** Does a model built from early checkpoints only (`dpo`, `s0100`, `s0500`) reach `s2300`'s behaviour without the later 1,800 labelled steps? Two versions: (a) AlphaRL-style rank-1 extrapolation (cheap, LoRA-servable; only meaningful if Q1 is a yes), (b) ExPO full-rank extrapolation (needs a written checkpoint; meaningful whatever Q1 says).
- Reference: `dpo`, `s0100`, `s0500`, `s2300` draws already exist (Dan's, T 1.0, top_p 1.0; not re-sampled; the `dpo`, `s0500`, `s2300` rows are the references).

### 2. Prompt sets and the dev / held-out split

Rule (CONTEXT): no setting is chosen on the reach prompts.
- **Dev (D)**: the 64 `pool/train.jsonl` prompts (arm-A non-reach; they were the EDL training prompts, irrelevant here). Used only for (i) AlphaRL's regression targets (stored accuracies, no sampling), (ii) Stage-2 alpha choice.
- **Held-out arm A (H)**: the other arm-A non-reach prompts = 218 - 42 (reach inside arm A) - 64 = 112 prompts (this includes the 32 `pool/test` prompts).
- **Reach (R)**: the 48 reach prompts, plain, exactly as Dan's arm B sent them. Scored, never used to choose anything.
- All numbers: % of the prompt set, strict checker ("checker passes"), prompt bootstrap 95 % intervals, paired where cells share prompts; `dpo` first; pass@1 leads, pass@5 beside it where draws allow (H: only 2 draws, so pass@1 only; R: 4 draws, pass@1 and "any pass in 4").

### 3. Cells (all on the `dpo` weights through vLLM's LoRA path, Dan's sampler, T 1.0, top_p 1.0, 32,768 tokens)

| id | intervention on `dpo` | sets x draws | role |
|---|---|---|---|
| **A1** | rank-1 of dW = `s2300` - `dpo`, per matrix (448), rescaled to the Frobenius norm of dW (Cai Eq. 3, my Frobenius reading; unscaled and lambda scans dropped) | H x 2, R x 4 | Q1 |
| **A2** | random rank-1: same sigma_1 and same rescale as A1 per matrix, but u, v drawn as random unit vectors (seeded) | H x 2, R x 2 | Q1 control (norm-matched, direction-free) |
| **A4** | rank-1 of `s0500` - `dpo`, same construction as A1 | H x 2, R x 2 | baseline for A3 (separates "rank-1" from "extrapolation") |
| **A3** | AlphaRL-style: per matrix the scaled vector u^ = alpha sigma_1 u_1 is regressed on accuracy over the on-lineage checkpoints up to `s0500` (`dpo`, `s0050`, `s0100`, `s0500`; `s0200`/`s0150` excluded as abandoned branch), inverted at the `s2300` dev accuracy (y* from Dan's stored draws on D); v_1 from `s0500` | H x 2, R x 4 | Q2 (a) |
| *references* | `dpo`, `s0500`, `s2300` (stored) | existing | lower / partial / upper |
| **W** (no sampling) | weights only: sigma_1 energy fraction per matrix, alpha, cosine and sign-aligned rank-1 trajectories across `dpo`, `s0050`, `s0100`, `s0500`, `s2300`; held-out check "fit on checkpoints <= `s0100`, predict the `s0500` rank-1 piece" and "fit on <= `s0500`, predict `s2300`" (cosine, norm ratio); share of ||dW|| in the non-448 tensors (embeddings, lm_head, norms) | computed on the pod from the downloaded shards | free given the SVD pass |

Stage 2, conditional (see sec. 7): **B** = ExPO on `s0500` (theta = `s0500` + alpha (`s0500` - `dpo`)), alpha in {2.5, 3.6} chosen on D, scored on H and R. **C** = rank-k curve, **E** = hybrid checkpoint (see 4).

Why Stage 1 is only A1-A4 + W: A1 answers Q1; A2 is the one control without which A1 is unreadable (the rescaled rank-1 matrix is 2-5x larger than the true rank-1 piece); A4 and A3 are the Q2 pair and share the same dpo load; W costs no tokens and tests Property 2 in weight space before any sampling.

### 4. Controls: needed and dropped

- **Random rank-1 of equal norm (A2): needed.** Cai has no such control. Because of the rescale, A1 is a full-norm perturbation; it has to be shown that this norm along u_1 v_1^T, not along an arbitrary direction, matters. Cheap (about 1.9 Mtok). Both u, v random (not shuffled layers), one seed; if A2 moves pass@1 by more than noise I will add a second seed before reading A1 (OPEN: Auditor may ask for a layer-shuffled control instead, which keeps each matrix's spectral shape; I think it adds little).
- **Full change as upper reference: already have it (`s2300`).** But it differs from "`dpo` + dW on the 448 matrices" by the non-448 tensors (embeddings, lm_head, 4 norm families x 64 + final norm). W measures their share of ||dW||. Only if the share is not small (my guess: tiny, since Cai reports minimal embedding shift for RL, p. 5, but our run is not theirs) and A1 falls short of `s2300` do we need **E**, a hybrid checkpoint (`s2300` matrices + `dpo` non-matrices; 64.5 GB, one extra load, about $1-2). Dropped from Stage 1.
- **Rank-k curve: dropped from Stage 1, conditional.** It matters only if A1 recovers clearly less than the full change: the interpretable follow-up is k in {16, 64}: rank 16 is the shape of the 2026-11-25 adapters (load path already verified), and 64 is 4x that. Each is one adapter on the same load. Not run unless Q1 is "partly": it would otherwise cost about $3-5 per k.
- **Unscaled rank-1 and Cai's lambda scan (best at 0.7, p. 4): dropped.** The scan is chosen on the test benchmark; running lambda here would need a dev-selection run per lambda. We test the paper's stated construction (lambda = 1) once.
- **Random control for A3: dropped.** A3 is read against A4 (same construction, no extrapolation) and against `s2300`; a random control for the extrapolated amplitude adds nothing the A2 result does not already give.

### 5. How each coefficient is set without looking at R

- **Rank-1 rescale alpha:** computed from dW, nothing tuned. Pre-registered as Frobenius ratio.
- **AlphaRL target y\*:** the `s2300` accuracy on D from Dan's stored draws (as the paper uses the fully trained model's MATH-500 accuracy, p. 7). The regression accuracies for `dpo`, `s0050`, `s0100`, `s0500` come from stored draws on D too. No sampling on R is used. Caveat that stays open: the final model's score enters the construction, as it does in the paper. A version that does not (predict y* from step count) would need a model of accuracy against steps; not proposed.
- **AlphaRL's weak point on our data:** `dpo` and `s0100` differ by 2.5 points of pass@1 (29.0 vs 31.5 on all 218 prompts, with standard errors of several points on a 64-prompt dev set), so the slope is set mostly by `s0500` against the rest; a one-component PLS on 3-4 points is a line through those points and cannot test linearity. Reported as is; W reports the same fit's out-of-sample behaviour in weight space.
- **Sign alignment (not in the paper):** u_1 vectors of different checkpoints are aligned by sign against the `s2300` vector before regression; fixed in code before looking at any generation.
- **ExPO alpha (Stage 2 only):** candidate set {2.5, 3.6} fixed now: 2.5 is ExPO's own optimum at 20 % of the steps (p. 5); 3.6 = (2300 - 500)/500, the straight-line step ratio (4.5 if labels from step 250 overstate by about 100). Pick by pass@1 on D (64 prompts x 2 draws), tie to the smaller alpha; also look at response-length and cap-hit fractions on D (ExPO's collapse signature, p. 5, p. 9). R and H never see the choice.

### 6. The lineage problem for `s0100` -> `s0500`

- Both are on the released line, but the released line restarts the optimiser at step 100 (Dan's review) and labels from step 250 on probably overstate optimiser steps by about 100. So "400 steps after `s0100`" is about 300 actual steps with a fresh Adam state at the start; a restart transient (large first normalised updates) sits inside `s0100` -> `s0500`. Straightness over the whole run is 0.10 and consecutive 50-step changes have cosine about -0.2: the direction is not stationary, so the first-order premise (ExPO) and the linear-in-checkpoint premise (Cai) are both doubtful here, independent of the answer.
- Consequence for design: (i) the x-axis of AlphaRL's fit is accuracy, not steps, so the label problem does not enter A3 directly; it only changes which checkpoints exist on the line (hence `s0150`, `s0200` excluded). (ii) For ExPO the step-ratio rule 3.6 vs 4.5 is a lineage-dependent guess; it is one of the two candidate alphas, not a result. (iii) I use `dpo` as M0 for both (ExPO's definition: pre-alignment checkpoint), not `s0100`: `s0500` - `s0100` contains the restart and is not what Q1 measures. W reports cos(`s0500` - `dpo`, `s0500` - `s0100`) and the rank-1 cosines so the choice can be revisited.
- Not claimed: that restarts are harmless for Cai's property; Cai's runs are single clean trajectories (p. 22).

### 7. Cost (tokens, H200 minutes, $)

Assumptions (from CONTEXT and 2026-11-25): H200 $5.29/h; 1,400 tok/s on the LoRA path (measured 1,335-1,568); mean tokens per draw 6,900 for adapters that pull toward the final model (A1, A3), 5,000 for A4, 4,500 for A2 plus a 1.3x allowance for repetition loops at the cap; model load 5 min (2-10); checkpoint download 1.5 min each (20 s-3 min). Dan's sampler hard-wires max_loras = 1 and one LoRARequest per run (`sample.py` l. 259-260, 281), so I assume one run (one load of `dpo`) per adapter; a custom multi-adapter driver would save about $2 and is not worth new code.

| item | draws | tokens (M) | H200 min | $ |
|---|---|---|---|---|
| env setup (`oe_setup.sh`) | | | 10 | 0.9 |
| download `dpo`, `s0050`, `s0100`, `s0500`, `s2300` (5 x 64.5 GB) | | | 8-10 | 0.8 |
| SVD pass (448 matrices x 4 differences, randomized top-k, k up to 64, fp32 from bf16 diffs; plus 3 exact full SVDs for validation; norms of the 4 non-448 families; build 4 adapters) | | | 15 | 1.3 |
| 4 x load `dpo` + LoRA | | | 20 | 1.8 |
| A1 (H x 2 + R x 4 = 416) | 416 | 2.87 | 34 | 3.0 |
| A3 (same) | 416 | 2.87 | 34 | 3.0 |
| A4 (H x 2 + R x 2 = 320) | 320 | 1.60 | 19 | 1.7 |
| A2 (H x 2 + R x 2 = 320, x1.3) | 320 | 1.87 | 22 | 1.9 |
| **Stage 1 total** | 1,472 | 9.2 | about 162 | **about 14.4** |
| contingency (retries, loops, a second A2 seed) +20 % | | | | 2.9 |

Stage 1: about $14-17 (range $11-25). That fits the $15 M2 allowance only if nothing is repeated.

Stage 2 (only after the Stage-1 result and with Leon's go): B (ExPO) about $6-8: write 2 checkpoints from `dpo` + `s0500` (about 5 min each on the pod, plus 64.5 GB disk each), 2 loads (12 min), dev sampling 2 alphas x 128 draws x 5,500 tok = 1.4 Mtok (17 min), final eval H x 2 + R x 4 on the chosen alpha (about 2.9 Mtok, 35 min); about 70 min, $6.2; plus E (hybrid) $1.5 or C (rank-16 and 64 adapters, H x 2 + R x 4 each) $3 + $3.

Budget note: the M2 slice of $15 holds only Stage 1. If Leon wants Q2(b) without waiting, B must be added now (about $21-24 total).

### 8. What each outcome would and would not show

Let G = pass@1(`s2300`) - pass@1(`dpo`) on the same set (H: from stored draws; R: 69.4 - 0.0).
- **A1 recovers most of G on H and R; A2 does not.** Shows that, on this model, a per-matrix rank-1 matrix built from dW and rescaled to dW's norm, with all other parameters at `dpo`, carries the checker-pass behaviour of `s2300`, and that the effect needs the direction. Does not show that "RL's change is rank-1" (the rescale amplifies it 2-5x), nor that dW has low rank, nor that the non-448 parameters are irrelevant (W reports their share), nor anything about faithfulness of passes (checker only).
- **A1 recovers a fraction between 25 % and 75 % of G, A2 about 0.** Rank-1 is a partial carrier; run C (rank 16, 64) and E (hybrid) before saying more.
- **A1 about 0 (like A2).** In this setting the paper's property fails (or the construction as I read it is not the paper's: Frobenius vs spectral, scope of tensors, bf16 differences). Not evidence about RL in general: one run, one lineage with restarts, differences from bf16 weights (65 % of weights within 1 ulp of `dpo`), different task type.
- **A2 moves pass@1 up or down materially.** A norm-matched random direction changes behaviour: the rescale itself is destructive or constructive, and A1 is uninterpretable until a second control (layer-shuffled u, v) is run.
- **A3 about A4 and both below `s2300`:** extrapolation of the rank-1 piece does not bridge s0500 -> s2300; says nothing about full-rank extrapolation. **A3 clearly above A4 and near `s2300`:** consistent with a linear rank-1 trajectory over these checkpoints; with 3-4 fit points and a regression target set from the final model's dev accuracy this is weak evidence for "predictability" (the target score is an input). **A3 above `s2300`:** overshoot is possible (Cai's Kuhn Poker tables show it, p. 20); would then be read as the amplitude being large, not as a better model.
- **W alone (no sampling).** If cos of sign-aligned rank-1 u-vectors across `s0100`, `s0500`, `s2300` is near 0 or the in-sample linear fit does not predict the held-out checkpoint, then the weight-space Property 2 fails here, and A3 is not worth sampling (decision before spending: I propose running W first, then A1/A2/A4, then A3 only if the held-out prediction cosine is at least 0.3; threshold is a guess).
- **B (ExPO, Stage 2).** Success would show that an extrapolation along dpo -> s0500 reaches part of the later gains; it would not show the gains are "linear in steps" (alpha is tuned on D), nor transfer to other tasks. Failure (collapse, long outputs at the cap) is the expected failure mode of ExPO's own Sec. 4.3 and shows the first-order premise is broken by these deltas.

### 9. Order of work on the pod (when approved)

1. Download, SVD pass, W summary (report back; decide on A3). 2. A1 and A2 with a quick lora_check-style teacher-forced gate for the rank-1 adapters (the 2026-11-25 `lora_check.py` pattern, gate at `dpo`). 3. A4, then A3. 4. Stop and report; Stage 2 only on Leon's go.

### F. Feasibility facts (read-only; with sources)

**F1. Parameter families of `allenai/Olmo-3-32B-Think-DPO`** (index fetched from the commit in `olmo_ifcurve/checkpoints.csv`, 97604023f2602a7dfe6d31b1fc229985e0002022; saved at `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/dpo_index/model.safetensors.index.json`; its git blob hash 5e5e2fa9... equals the csv's `index_blob`; 707 tensors, metadata total_parameters 32,233,522,176):

| family (per layer, 64 layers) | tensors | shape [out, in] | notes |
|---|---|---|---|
| self_attn.q_proj | 64 | [5120, 5120] | 40 heads x 128 |
| self_attn.k_proj | 64 | [1024, 5120] | 8 KV heads x 128 |
| self_attn.v_proj | 64 | [1024, 5120] | |
| self_attn.o_proj | 64 | [5120, 5120] | |
| mlp.gate_proj / up_proj | 64 + 64 | [27648, 5120] | largest SVD inputs |
| mlp.down_proj | 64 | [5120, 27648] | |
| **sum: 448 matrices** | 448 | | what a rank-k change touches |
| self_attn.q_norm, k_norm | 64 + 64 | vector (full q / k width: 5120, 1024, from the HF Olmo3 definition; not read from the shards) | stay at `dpo` under the LoRA route |
| post_attention_layernorm, post_feedforward_layernorm | 64 + 64 | [5120] | |
| model.norm | 1 | [5120] | |
| model.embed_tokens, lm_head | 1 + 1 | [100278, 5120] (config.json: vocab 100,278, hidden 5,120; untied) | |

448 + 256 + 3 = 707 tensors, matching the index. Shapes in the table are derived from config.json (`tok/s2300/config.json`: hidden 5120, intermediate 27648, 40/8 heads), not read from the shard headers (the safetensors headers were not fetched).

**F2. Can the rank-k change be served on vLLM 0.11.0's LoRA path?** (v0.11.0 raw files in `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/vllm_v0110/`)
- `Olmo3ForCausalLM` maps to `Olmo2ForCausalLM` (registry.py), which is `SupportsLoRA` with `packed_modules_mapping` {qkv_proj: [q,k,v], gate_up_proj: [gate, up]} (olmo2.py l. 373-385): PEFT adapters naming q_proj, k_proj, v_proj, gate_proj, up_proj, o_proj, down_proj work; the 2026-11-25 adapters (r 16, alpha 32, 7 target modules, 896 tensors) already did, and its `lora_check.py` gate (teacher-forced vLLM vs PEFT nats; STAGE2.md section L) verified that path. Rank 1 is the same file layout with r = 1.
- `config/lora.py`: `max_lora_rank` must be one of (8, 16, 32, 64, 128, 256, 320, 512); an adapter of rank r is accepted when r <= max_lora_rank (`peft_helper.py` validate_legal). So rank-1 adapters run with max_lora_rank 8 (Dan's default is 16 via `--lora_rank`; both work), rank-16 with 16, rank-64 with 64. Scaling is lora_alpha / r (rsLoRA off): for rank k use lora_alpha = r = k and fold sqrt(sigma) into A and B, or any equivalent split. LoRA dtype defaults to the model dtype, so the factors are cast to bf16 (relative error about 0.4 % per factor; fine for rank-1, to be checked by the teacher-forced gate). Unmerged: the delta is not rounded into bf16 weights, which matters because a merge of sub-ulp deltas would erase most of them (Dan's merged model missed his 1 % merge check, 1.12 %).
- Slots: `max_loras` = adapters in one batch; `_apply_adapters` raises RuntimeError if a batch needs more adapters than slots (`worker_manager.py`); `max_cpu_loras` defaults to `max_loras`; LRU eviction beyond that. GPU memory per slot is about 2 x max_rank x sum(in+out) per layer: with max_rank 8 about 134 MB per slot over 64 layers (my estimate), with max_rank 64 about 1.1 GB. So one engine can hold several adapters. **But Dan's sampler hard-wires `max_loras=1` and a single `LoRARequest` for all requests** (`sample.py` l. 259-260, 281), so with the sampler unchanged each adapter is a separate run (a `dpo` reload each, about $0.5-0.9). A random control, the rank-1 and other variants therefore share `dpo` weights only through separate loads or through a modified driver. Not needed at this scale.
- Limits: the LoRA path covers only the 448 matrices (embeddings, lm_head, norms stay at `dpo`; vLLM 0.11.0's LoRA also does not support `modules_to_save`, peft_helper.py). A full-rank extrapolation (ExPO) is not a LoRA. Throughput in the 2026-11-25 jobs: 1,335-1,568 tok/s vs 1,400-1,700 plain (about 10 % less).
- Unverified (need the pod): that vLLM 0.11.0 packed-module handling of three separately scaled q/k/v rank-1 factors behaves as in the rank-16 case (expected, same code path), and the bf16 cast of tiny factors. The teacher-forced gate covers both.

**F3. SVD cost.** Randomized top-k SVD (e.g. `torch.svd_lowrank`, q = k + 10, 4 power iterations) on the largest matrices (27,648 x 5,120): about 12 matmul passes of 2 x m x n x q = 2 x 1.4e8 x 10 flops, about 30 GFLOP, well under a second per matrix on an H200 (a guess from flop counts, not measured); 448 matrices x 4 differences about 1,800 decompositions, about 5-10 min including reading 5 x 64.5 GB from pod disk (about 30 s each at NVMe speed, my guess). Three exact `torch.linalg.svd` validations of the largest shapes cost a few minutes each (fp32 full SVD of 27,648 x 5,120; guess 1-3 min). Differences are formed exactly in fp32 (a bf16 difference is representable). So the SVD pass is about $1-2, dominated by the downloads and disk, not compute. The pass must run on the pod because no weights go to serrano (about 50 GB free on /scratch).
**Writing a full checkpoint** (ExPO, hybrid E): elementwise fp32 combine of two checkpoints and a bf16 write of 64.5 GB: about 5 min of pod time, $0.5, plus 64.5 GB of pod disk and one extra load (5 min, $0.5). The dominant cost of ExPO is sampling, not writing.

### OPEN for the auditor

1. Frobenius vs spectral reading of Cai's rescale (a spectral reading makes alpha = 1; I chose Frobenius).
2. Whether non-448 tensors should be carried: LoRA route cannot; E is the answer if W says they matter.
3. Whether A2 should be layer-shuffled instead of Gaussian.
4. Whether AlphaRL's fit with 4 points deserves sampling at all (decision rule in sec. 8 is a guess).
5. 2 draws on H, 2-4 on R: enough to read recoveries of 50 % vs 80 % of G (interval half-width about 5-6 points on H by my estimate, not computed).
6. I did not read the AlphaRL code release; the construction questions in sec. 0 might be answerable from it.

## Turn 2 — auditor (Claude Opus 5.5), 2026-11-27 02:07 PT

**Basis.**
- My independent read is `discourse/rank1_audit_independent.md` (md5 `ba9bc183...`, written before I opened anything
  of yours).
- Since then I have read your two notes and Turn 1.
- New computations, all under `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/audit/`:
  - per-group shares of ‖Δ‖² from Dan's tracked per-tensor stats (`rank1_param_groups_output.txt`; it reproduces
    Dan's 0.3642 % total);
  - a simulation of draw-count resolution from Dan's stored per-prompt rates (`rank1_resolution.py` ->
    `rank1_resolution_output.txt`);
  - a read of vLLM 0.11.0 `olmo2.py` from your saved copy.
- Costs below use your Turn-1 assumptions (1,400 tok/s, $5.29/h, a 5-min load of about $0.45, 6,900 / 5,000 / 4,500
  tokens per draw) so the two tables compare.

### A. Disagreements between the readings

**Cai et al.** I agree with you on these, so one line:
- the Frobenius reading of Eq. 3 and α of about 2.3-5 in the paper;
- the unspecified parameter scope;
- all matrices at once, with a base-to-t pair;
- that sign alignment and precision are never mentioned;
- that "relative accuracy" is a ratio;
- that AlphaRL's target leaks the final model's MATH-500 score;
- the in-sample one-component PLS R² critique (we reached the p ≫ T argument independently);
- that RL-specificity is contradicted by App. C and Tables 7-8.

Disagreements:

1. **"The 99.17 % may be rank-1, rank-1 % or a mixture" (your note §2).**
   - The sentence names rank-1: "the Rank-1 Subspace alone recovers an average of 99.17% of the reasoning capability"
     (p. 2).
   - What is missing is a table behind it, not the variant. **The text supports me** (minor).
2. **Fig. 7 values (p. 24), which you did not read.**
   - Zoomed, the MATH-500 panel shows "Top Rank-100%" at about 26 % (DIST) and 57 % (SFT) relative accuracy.
   - Keeping every singular direction reconstructs ΔW exactly, so a faithful full-rank reconstruction must give
     100 %.
   - This is the best evidence in the paper on your "not settled: which parameters". It suggests changed tensors
     outside the decomposition (embeddings, which DIST and SFT move a lot, Fig. 5b) were left at base. That is my
     inference.
   - The same experiment is reported three inconsistent ways. Fig. 3(a)'s groups "70%" and "90%" carry Fig. 7's
     "80%" and "100%" values, and Table 8 gives different numbers again (SFT Rank-90% 46.75 against about 57 in both
     figures).
   - **The figures support my values; the inference is mine.**
3. **">96 %" for GRPO and RLOO (your note §3, "96 % for GRPO/RLOO, p. 8").**
   - Table 1's averages give 49.42/51.52 = 95.9 % and 48.52/50.82 = 95.5 %.
   - The text's "achieve 96%" is rounded up and the abstract's ">96%" holds only for DAPO. **The table supports me**
     (minor).
4. **"Table 1 DAPO block identical to Table 3, an anomaly" (your note §3).**
   - I read it as the same run shown twice, with Table 3 adding stages. Its caption omits the algorithm name.
   - Not an anomaly in the data, but it does mean there is no independent replication, which is your real point.
     Neither reading is contradicted.
5. **"Nothing in the paper tests a lineage with a prior preference-optimisation stage" (your note §5).**
   - App. B (p. 20) has DAPO after a cold-start SFT on Llama3.1-8B-Instruct.
   - Llama 3.1 Instruct was itself DPO-trained. That comes from the Llama 3 report, outside this paper.
   - So the paper has one such lineage, but only in Table 4's AlphaRL rows, with **no rank-1 reconstruction
     numbers**. "The rank-1 structure ... hold[s] across these diverse model initializations" (p. 18) has no table
     behind it.
   - Partly mine. The DPO-stage fact needs outside knowledge.
6. **Omissions on your side that matter for the prior** (not contradictions):
   - **Fig. 14 / p. 28**: "using only the first 20 tokens ... allows the base model to approach the performance of the
     full parameter update". Cai's gains are largely an opening-style switch in RL from base models; a mode switch
     is plausibly one direction. Our dpo → s2300 has no such switch.
   - **Fig. 7**: the only 32B model (DAPO*) is the weakest at rank-1 on hard sets, about 87 % (AIME24) and 91 %
     (AIME25) relative.
   - **Fig. 4(a)/11**: lower-ranked directions, each rescaled to full norm, also lift accuracy. So "the top direction"
     is specific only by degree, and the paper has no random-direction control.

**ExPO.** I agree with you on these, so one line:
- Eq. 6, and that M0 = dpo;
- α chosen by RM-Mistral-7B reward on the UltraFeedback dev set;
- optima 8.0, 2.5, 0.5 and 0.3;
- the failure cases in §4.3 and that large α gives length blow-ups;
- no seeds and no intervals;
- that ExPO never extrapolates between two aligned checkpoints.

Disagreements:

7. **You are right, and I missed two points.**
   - Tables 3, 4 and 6 report the selection criterion on the selection set itself (p. 4: "unless otherwise stated, we
     report the expected reward score on the UltraFeedback development set"). I had flagged only Table 2's reward
     column.
   - The §4.1 optima sit at the grid edge 0.5 for tulu2 and internlm2 (Table 7, p. 15), so they are censored.
     **The text supports you.**
8. **The lower γ candidate.**
   - You propose α = 2.5, i.e. γ = 3.5, ExPO's own 20 % optimum. I proposed γ = 3.17 = ‖Δ2300‖/‖Δ500‖, norm-matched,
     from Dan's stats.
   - Neither is derived in either paper. Both are borrowed (from another regime, or from the endpoint's norm).
   - **I accept 3.5.** 3.17 goes into W as a reported number only.
9. **A point neither of us made: ExPO's own deltas were probably deeper in the bf16 dead zone than ours.**
   - Table 1 (p. 2) gives a normalised Frobenius SFT → DPO change of 6.348e-6 per element. Against a typical 7B weight
     of about 1e-2 (bf16 ulp about 4e-5 to 1.2e-4), that is about 0.05-0.15 ulp RMS (my estimate).
   - The dtype of their checkpoints is unstated, and §4.1's released models are, I believe, bf16 (unverified).
   - So extrapolating heavily rounded deltas apparently worked there. That weakens the bf16 objection to ExPO from
     s0500, where the typical 448-matrix weight moved about a third of an ulp. It is not a reason to skip W.

**Turn 1's feasibility section.** One of my open items is now settled. vLLM 0.11.0's `Olmo2ForCausalLM` (`olmo2.py`
l. 373-386) defines `packed_modules_mapping` but no `embedding_modules`. LoRA on embeddings and lm_head goes through
`embedding_modules` (`vllm/lora/models.py` l. 127-146). So on this path lm_head stays at dpo, as you wrote. This is
read from the source, not run.

### B. Attack on the design

**B1. A1 alone does not answer Leon's question. Add the unscaled rank-1 (A1u) to Stage 1.**
- Leon's question is literally "keeping only the top singular direction of each weight matrix's change". That is
  σ1 u1 v1^⊤.
- A1 is that direction amplified α-fold to ‖ΔW‖_F. Your own §8 says an A1 success "does not show that RL's change is
  rank-1 (the rescale amplifies it 2-5x)".
- Your reason for dropping it ("the scan is chosen on the test benchmark") applies to a λ scan, not to A1u. A1u is a
  fixed construction (λ = 1/α) with nothing tuned.
- A1u is also Cai's Eq. 3 under its literal spectral-norm reading (α = 1), so it covers your OPEN 1 and your §8
  fallback "the construction as I read it is not the paper's".
- A1 against A1u is the only comparison that separates "this direction" from "this direction, amplified".
- Cost: H × 2 + R × 2 = 320 draws × 5,000 tokens = 1.6 Mtok, 19 min, $1.7, plus one load ($0.45). **+$2.1** ($1.7 if
  it shares an engine with A1).

**B2. A2 is the right control, specified exactly; trim it and let it serve both A1 and A1u.**
- Your A2 (u and v random unit vectors, with A1's σ1 and rescale) is my control, provided "random unit vector" means
  uniform on the sphere. Pin it down:
  - per matrix W (m × n), g ~ N(0, I_m) and h ~ N(0, I_n) in fp64;
  - seed = global seed plus a stable hash of the tensor name (e.g. `zlib.crc32`), independent per matrix;
  - R = ‖ΔW‖_F · (g/‖g‖)(h/‖h‖)^⊤, so the Frobenius norm equals the spectral norm equals α σ1;
  - PEFT factors B = √c · g/‖g‖ and A = √c · (h/‖h‖)^⊤ with c = ‖ΔW‖_F, lora_alpha = r = 1.
- One A2 at the larger (rescaled) norm is the bound on non-specific effects for both A1 and A1u. No second random
  cell at σ1 is needed unless A2 moves.
- Interpreting it: an isotropic v barely overlaps the activations' dominant directions, so A2 ≈ dpo is a weak null.
  It shows the effect is not "any perturbation of this size", not that u1 v1 is uniquely the right direction.
  Layer-shuffled stays OPEN (below).
- Draws: H × 1 + R × 2 = 208 (resolution in B6). **−$0.6** against Turn 1's H × 2 + R × 2.

**B3. A4 + A3 (AlphaRL on 3-4 checkpoints) should not be in Stage 1. ExPO from dpo should be.**

Why A3 is weak:
- Its fit points are not distinguishable. Dan's stored draws on the 64 dev prompts D give pass@1 (95 % prompt CI):
  - dpo 41.8 [32.0, 51.8]
  - s0050 43.4 [33.6, 53.1] (only 4 draws per prompt)
  - s0100 45.3 [35.7, 55.3]
  - s0500 60.2 [51.0, 69.1]
  - s2300 78.3 [70.3, 85.5]
- Three of four points lie within 3.5 points of each other, so the one-component PLS is a line through two clusters.
  Your §5 says this too.
- Its amplitude is set by D's saturation, not by the weight trajectory. Linearly in accuracy, y* = 78.3 corresponds to
  about 2.0× s0500's distance from dpo, against a weight-norm ratio of 3.17.
- The paper's "inversion" (a scalar target back to a d-dimensional vector, standardized over T = 4 checkpoints, with v1
  from an unnamed checkpoint) is unspecified. Any implementation is ours, not AlphaRL's.
- The target uses s2300's dev score. CONTEXT allows that, but it answers "knowing the final dev score, can the early
  rank-1 direction be scaled to it?", which is weaker than Leon's question.
- Your §1 says A3 is "only meaningful if Q1 is a yes". By your own logic it belongs after A1, not beside it.
- Even on its home ground (Table 3, at 20 % of training) AlphaRL closes only 33-43 % of the remaining gap.

Why ExPO (your Stage-2 B) belongs in Stage 1:
- It is the published method, and Turn 1 already wrote it in the faithful form (θ = dpo + γ(s0500 − dpo)).
- It does not depend on Q1.
- Its 20 % result (Table 2: 20 % + ExPO beats the fully trained model) is the sharpest falsifiable prediction in either
  paper, and it disagrees with AlphaRL's at s0500, so the cell discriminates between them.
- Its known risk (length blow-up, collapse) is caught cheaply on D.

Required form:
- γ ∈ {3.5, 4.6} (your set).
- Pilot: 16 D prompts × 1 per γ, aborting a γ if > 25 % of draws hit 32,768 tokens or > 10 % never close the thinking
  block.
- Then D 64 × 2 per γ. Choose by pass@1; ties go to the smaller γ. Record the choice in a timestamped file.
- Score once on H × 2 + R × 4.
- Delete the unchosen checkpoint.

Cost and trigger:
- B costs 2 writes $0.9, 2 loads $0.9, dev 256 draws × 6,900 = 1.77 Mtok $1.85, held-out 416 × 6,900 = 2.87 Mtok
  $3.0, so $6.7 (+$0.45 if the chosen γ needs a reload).
- Removing A3 + A4 from Stage 1 saves A3 $3.0 + A4 $1.7 + 2 loads $0.9 + the s0050 download $0.15 = $5.75.
- **Net +$1.0 to +$1.4.**
- A3 (or A4 + A3) then becomes conditional: run only if A1 recovers ≥ 50 % of G on both H and R **and** W's held-out
  prediction cosine (your rule, 0.3) passes, and only with budget left.

**B4. W must be a gate, and it lacks three things.**

Additions:
- **(a) Noise floor of σ1.** In bf16 a stored Δ is a dead-zone quantization of the true change, and at s2300 about one
  ulp is a typical move.
  - Per matrix, compare σ1(ΔW) with σ1(ΔW ⊙ S), where S is a random ±1 sign matrix. Sign randomization keeps every
    entry's magnitude, hence the |w|-dependent ulp pattern, and destroys low-rank structure.
  - Gate: if the ‖ΔW‖_F-weighted median ratio for Δ2300 is below about 1.5 (my guess at a threshold), A1 and A1u
    cannot be read as tests of the paper. Stop and report before sampling.
  - The same ratio for Δ500 and Δ100 qualifies your Property-2 trajectory test (s0100 has 79 % of matrix weights
    bitwise unchanged).
- **(b) For ExPO.**
  - Report cos(Δ500, Δ2300) per matrix and in total.
  - Report the weight-space oracle γ_proj = ⟨Δ500, Δ2300⟩/‖Δ500‖² and ‖dpo + γΔ500 − s2300‖/‖Δ2300‖ for γ ∈ {3.17,
    3.5, 4.6}.
  - These are reported, never used to choose. Dan's stats hold distances from dpo but not this inner product, so it
    needs the pod.
- **(c) The α distribution.** If the median α exceeds about 10 (a guess), run A1's first 32 draws as a pilot under the
  B3 abort rule before the full cell.
  - Cai's own SFT/DIST rank-1 rows (1.5 % and 2.5 % relative accuracy, far below base) show what over-amplified rank-1
    does.
- **(d) Drop the non-448 share from W; it is known** (Dan's stats, my aggregation):

  | | s0100 | s0500 | s2300 |
  |---|---|---|---|
  | 448 matrices | 98.97 % | 98.84 % | 98.43 % |
  | lm_head | 0.97 % | 1.09 % | 1.45 % |
  | embeddings | 0.06 % | 0.07 % | 0.12 % |
  | norms | 0.000 % | 0.000 % | 0.001 % |

  Per weight, lm_head moves 0.45 % at s2300, as much as the matrices (0.50 %), and it acts directly on the logits.
  "Tiny share" is right in norm but does not imply negligible behaviour.

Cost: one extra randomized SVD per matrix for the sign null, and the inner products. **+$0.4.**

**B5. Non-448 tensors: keep E conditional, but fix its trigger now and correct its cost.**
- Trigger: run E (s2300 with dpo's lm_head, embeddings and norms, i.e. dpo plus the full change on the 448 matrices)
  iff W's noise gate passed and A1 < 80 % of G on R.
- Sets: R × 4 + H × 1.
- Turn 1's "$1-2" (§4) and "$1.5" (§7) count the write and load but not the sampling. At s2300-like length:
  - R × 4 + H × 1 = 304 draws × 6,900 = 2.1 Mtok, 25 min, $2.2, plus write and load $0.9, so **about $3.1**;
  - H × 2 + R × 4 is about $3.9.
- Since lm_head cannot be adapted on vLLM 0.11.0 (A, last paragraph), E is the only way to bound the lm_head
  departure.

**B6. Draw counts: resolution is adequate for A1 and B, coarse for the controls (simulated, not guessed).**
- From Dan's stored per-prompt rates on the 112 H prompts: dpo 32.5 % and s2300 64.5 % pass@1, so G_H = 32.0 points;
  G_R = 69.4.
- Assume a cell whose per-prompt rate is dpo + f·(s2300 − dpo), paired prompt bootstrap. The 95 % half-width of the
  recovered fraction f is:

  | draws | f half-width |
  |---|---|
  | H × 1 | ±19-21 pts |
  | H × 2 | ±14 pts (about ±4.5 pass@1 points, close to your 5-6 guess) |
  | H × 4 | ±10 pts |
  | R × 2 | ±12-13 pts |
  | R × 4 | ±9 pts |
  | R × 8 | ±6 pts |

- So A1 at H × 2 + R × 4 separates f = 0.5 from 0.8 (about 3σ). A1u and A2 at 2 draws separate "about 0" from
  "≥ 0.4", which is their job.
- **I accept your counts** for A1 and the B held-out (H × 2 + R × 4), and **your H definition** (112 prompts, reach
  scored separately). It is cleaner than my 154.

**B7. Cost, M2's share, with setup and the dpo download shared with M1 and M3.**

| item | Turn 1 | revised |
|---|---|---|
| env setup | 0.9 | 0 (shared) |
| downloads | 0.8 (5 ckpts) | 0.5 (s0100, s0500, s2300) |
| SVD and W | 1.3 | 1.7 |
| adapter loads | 1.8 (4) | 1.35 (A1, A1u, A2), or 0.45 on one engine |
| A1 | 3.0 | 3.0 |
| A1u | – | 1.7 |
| A2 | 1.9 | 1.3 |
| A4 + A3 | 4.7 | 0 (conditional) |
| B (ExPO) | Stage 2 | 6.7 |
| **total** | **14.4** | **16.2** (15.3 with one engine for A1, A1u, A2; 14.4 with D at 32 × 2) |
| contingency +20 % | 2.9 | about 3 |

- This is about $15 ± 1 before contingency, $18-19 with it.
- The rank-1 half (W, A1, A1u, A2) is about $8.5. The extrapolation half (B) is about $6.7.
- If the orchestrator must cut further, my order is:
  1. one engine serving A1, A1u and A2 (a small wrapper with `max_loras = 3` and a LoRARequest per prompt; −$0.9);
  2. D at 32 × 2 (−$0.9);
  3. last, one pre-committed γ = 3.5 with only the collapse pilot (−$2.3). That keeps ExPO's construction but drops
     its tuning; say so if it is done.
- Conditional cells (E $3.1, rank-16 or rank-64 $3.5 each, A3 + A4 about $5) each need their trigger plus budget
  left, and Leon's go beyond $80 total.
- A note for sharing with M1: M1 compares against Dan's stored draws from a non-LoRA engine. Do not serve M1's base
  cells from M2's LoRA-enabled dpo engine unless a teacher-forced check shows identical base logits.

### Required changes

1. Add **A1u** = σ1 u1 v1^⊤ of Δ2300 per matrix, all 448, unscaled; H × 2 + R × 2; Stage 1. **+$2.1** ($1.7 on a
   shared engine).
2. Specify **A2** exactly as in B2 (normalized Gaussians in fp64, per-tensor seeds, magnitude ‖ΔW‖_F). Trim to
   H × 1 + R × 2. It is the single non-specific bound for A1 and A1u. **−$0.6.**
3. Move **A4 + A3** out of Stage 1. They become conditional on A1 ≥ 50 % of G on H and R plus your weight-space
   cosine rule. Drop the s0050 download. **−$5.75** (about +$5 if triggered later).
4. Move **ExPO B** (dpo + γ(s0500 − dpo), γ ∈ {3.5, 4.6}) into Stage 1, with the collapse pilot and abort rule, D
   64 × 2 per γ, a frozen choice, and held-out H × 2 + R × 4. **+$6.7** (+$0.45 on a reload).
5. Make **W a gate** with the additions in B4:
   - (a) a sign-randomized noise floor for σ1, gating A1 and A1u;
   - (b) cos(Δ500, Δ2300), γ_proj and the distance to s2300 for each γ, reported only;
   - (c) an α-distribution pilot rule;
   - (d) drop the non-448 share (known: lm_head 1.45 % of ‖Δ2300‖², 0.45 % per weight).
   **+$0.4.**
6. Fix **E's trigger** now (noise gate passed and A1 < 80 % of G on R) and its cost (about $3.1 at R × 4 + H × 1,
   not $1.5). Conditional.
7. Apply the **abort rule** of B3 (cap hits > 25 % or missing end of thinking > 10 % in the first 32 draws) to every
   new cell, A1 included. Cost-neutral; it protects the budget.
8. Report recovery as the **fraction of G**, with paired prompt-bootstrap intervals: G_H = 32.0 points (dpo 32.5,
   s2300 64.5 on the 112 H prompts, stored draws), G_R = 69.4. Give Cai's ratio only beside it. Cost-neutral.
9. Size the pod volume for dpo, s0100, s0500, s2300 and two ExPO writes (about 390 GB; 450 GB if E runs), and delete
   the unchosen ExPO checkpoint after the D choice. Cost-neutral apart from the volume.

### Points I would accept being left OPEN

1. Gaussian A2 versus a layer-shuffled or activation-matched random control. Gaussian in Stage 1; the others only if
   A1 succeeds and specificity becomes the question.
2. The noise-gate threshold (1.5) and the α-pilot threshold (10). Both are guesses; report the distributions whatever
   is decided.
3. Which conditional Q2(a) cell to run if triggered: your A3 (PLS on dpo, s0050, s0100, s0500 with the s2300 dev
   target) or an "AlphaRL-lite" adapter (κ × the rescaled rank-1 of Δ500, κ chosen on D by pass@1, with no endpoint
   information).
4. Whether the lm_head departure matters, until E runs or A1 ≥ 80 % of G makes it moot.
5. Norm versus energy reading of Cai's "Rank-1 fraction". It becomes moot once W measures α on our weights.
6. Whether to write a multi-adapter wrapper rather than reload dpo per adapter.
7. The AlphaRL code release (unread by both of us). If someone reads it later, the Q1 construction questions may close.

## Turn 3 — reader-implementer (Claude Sonnet 5.5), 2026-11-27 02:10 PT

I read Turn 2 and `rank1_audit_independent.md`. Corrections to my notes are in a dated "Correction after audit" section at the end of the Cai note (nothing edited above it). Page references are PDF pages.

### A. Reading disagreements

1. **99.17 % names rank-1 (p. 2).** Conceded: the sentence says "the Rank-1 Subspace alone". My "or rank-1 %, or a mixture" was too loose. What I keep: no table or figure lists its components, and the headline Figure 1 (p. 1) shows Rank-1 %.
2. **Fig. 7 "Top Rank-100%" (p. 24).** Conceded and verified on the rendered page: MATH-500 about 26 % (DIST) and 57 % (SFT); AIME24 about 44 / 47 %; GSM8K about 62 / 61 %. A full-rank reconstruction must be 100 %, so something changed was left outside the decomposition. Where I argue: this is evidence about *scope*, and the leading explanation (embeddings and other non-matrix tensors left at base) is an inference. A dtype effect in the reconstruction is not excluded. It does support the LoRA route (matrices only) as faithful to what the paper evaluated, which weakens the case for E as a fidelity fix and keeps E as a bound on lm_head only. Fig. 3a, Fig. 7 and Table 8 also report one experiment three ways; I now count that as another reason not to anchor on any one number.
3. **">96 %".** Conceded: GRPO 49.42 / 51.52 = 95.9 %, RLOO 48.52 / 50.82 = 95.5 %. The text's "96 %" is rounded up; the abstract's ">96 %" holds for DAPO only.
4. **Llama3.1-8B-Instruct lineage (p. 20, Table 4 p. 19).** Conceded: my "nothing tests a prior preference stage" was too strong. The paper has a cold-start-then-DAPO lineage on that model; Llama 3.1 Instruct being DPO-trained is outside knowledge that I accept. It has no rank-1 reconstruction numbers, and p. 18's "hold across these diverse model initializations" has no table. So the claim that nothing in the paper shows rank-1 working after a DPO stage stands in that narrower form.
5. **Tables 1 and 3 identical.** Accepted as the same run shown twice; my "anomaly" wording withdrawn, "no independent replication" kept.
6. **Fig. 14 / p. 28, Fig. 4a / 11, 32B row of Fig. 7.** Accepted as omissions. They matter for the prior (below, section C).
7. **ExPO.** Agreed on both of your additions (ablation tables scored on the selection set; censored grid edges). I accept gamma = 3.5 (ExPO alpha 2.5) as the lower candidate, 4.6 (alpha 3.6) as the upper; 3.17 (norm ratio) goes to W as a reported number. My Turn 1 "alpha in {2.5, 3.6}" and your "gamma in {3.5, 4.6}" are the same two models (theta = dpo + gamma (s0500 - dpo) = s0500 + alpha (s0500 - dpo)). Point 9 (ExPO's deltas probably sat deeper in the bf16 dead zone than ours) is accepted as a reason not to treat bf16 as disqualifying ExPO from s0500; it needs the unverified assumption that their released checkpoints are bf16, so I note it as such.

### B. The nine required changes

1. **A1u (unscaled sigma_1 u_1 v_1^T): accept, Stage 1.** It is Leon's literal question, it is Cai's Eq. 3 under the spectral reading, and it is nothing tuned. I expect it to sit near `dpo` (it is 1/alpha of A1's norm); that is a finding, not a waste: A1 vs A1u separates "the direction" from "the direction amplified". H x 2 + R x 2. Not a lambda scan.
2. **A2 as specified: accept** (Gaussian g, h in fp64, per-tensor seed from a stable hash, magnitude ||dW||_F, B = sqrt(c) g/||g||, A = sqrt(c) h^T/||h||, lora_alpha = r = 1), trimmed to H x 1 + R x 2. I add one caveat: an isotropic v overlaps the activation subspace weakly, so A2 is a weak null (it excludes "any perturbation of this norm", not "any rank-1 direction"). Layer-shuffled stays OPEN.
3. **A3 + A4 conditional: accept.** I agree they belong after A1: A3's target comes from the endpoint's dev score, its fit points are two clusters (dev pass@1 41.8 / 43.4 / 45.3 vs 60.2 vs 78.3 from your table), and the paper's inversion is unspecified. Trigger as you wrote (A1 >= 50 % of G on H and R, and the W prediction cosine >= 0.3). I drop `s0050` from the downloads.
4. **ExPO into Stage 1: accept.** Reasons I agree: it does not depend on Q1, it is the published method, and at 20 % of the steps its prediction (Table 2) is the sharpest falsifiable claim in either paper. I add two protections: (a) the pilot's "never closes the thinking block" criterion uses the stored stop reason and the presence of the end-of-thinking token in the saved row, counted per draw, not a re-parse; (b) a choice between the two gammas on dev is only made if the two dev pass@1 values differ by at least 2 points; below that the smaller gamma is taken (ties to smaller, as you said, made explicit). D is 64 prompts x 2 draws, so a 2-point difference is well inside noise; this rule is therefore a convention, and the held-out result is not read as "the best gamma".
5. **W as a gate: accept the sign-randomized null (a), with one change to the stop rule.** Your gate stops sampling if the weighted median sigma_1(dW) / sigma_1(dW o S) for Delta_2300 is below 1.5. The threshold is a guess, as you say. I argue the stop should be at **1.1**: between 1.1 and 1.5 sigma_1 is still distinguishable from the null, A1's result (a pass or a fail) is still informative about Leon's question ("does the top direction keep the behaviour?") even if it cannot be called a test of Cai's property, and A1 costs $3. Below 1.1 sigma_1 is the null's, A1 is a random rank-1 matrix by construction, and I skip A1 and A1u. Between 1.1 and 1.5 the cells run and are labelled "noise-dominated". Also accepted: (b) cos(Delta_500, Delta_2300), gamma_proj and ||dpo + gamma Delta_500 - s2300|| / ||Delta_2300|| for gamma in {3.17, 3.5, 4.6}, reported and never used to choose; (c) the alpha-distribution rule: if the median alpha exceeds 10, A1's first 32 draws run as a pilot under the abort rule first; (d) the non-448 share dropped from W, using your numbers from Dan's stats (lm_head 1.45 % of ||Delta_2300||^2, embeddings 0.12 %, norms about 0, 448 matrices 98.43 %).
6. **E's trigger and cost: accept.** Run E iff the noise gate passed (ratio >= 1.1 by my rule) and A1 < 80 % of G on R. Cost about $3.1 at R x 4 + H x 1, not $1.5.
7. **Abort rule on every new cell: accept** (cap hits > 25 % or missing end of thinking > 10 % in the first 32 draws; the cell is then reported as "collapsed" with those counts, which is itself a result). For A2 and A1u a collapse would be a finding about norm, not a bug.
8. **Recovery as a fraction of G: accept** (details in section D). Cai's ratio is printed beside it only.
9. **Volume: accept** about 390 GB (dpo, s0100, s0500, s2300, two ExPO writes), 450 GB if E runs; the unchosen ExPO checkpoint is deleted after the dev choice. Under the shared session dpo, s0500 and s2300 are already on the volume for M1 and M3.

**lm_head stays at dpo.** Verified: `olmo2.py` in v0.11.0 defines `packed_modules_mapping` and no `embedding_modules`; `lora/models.py` l. 127-146 and 555-556 handle embeddings and lm_head only through `embedding_modules`. So on the LoRA path lm_head, embeddings and norms stay at `dpo`. The faithful-to-Cai reading (matrices only, Fig. 7 inference) says this is what the paper did too; E bounds it for us. ExPO (full-rank) is applied to all tensors, as in the paper, since the written checkpoint can.

### C. What the full set of corrections does to the prior

Cai's gains come from RL runs starting at base models where a mode switch (opening style, Fig. 14) probably carries most of the gain; our dpo -> s2300 is a checkpoint that already has a thinking format, and the gain is IFBench strict passes. My prior for A1 being a clear yes is lower than the paper suggests, and A1u near `dpo`; both are guesses. I proceed because it is the test Leon asked for and costs about $8.5 with W.

### D. Revised design, self-contained

**Goal.** Q1: does the top singular direction of dW(dpo -> s2300), per matrix, keep s2300's behaviour? Q2: does extrapolating dpo -> s0500 reach s2300's behaviour?

**Prompt sets (unchanged).** D = 64 pool-train prompts (arm-A non-reach); H = 112 held-out arm-A non-reach prompts; R = 48 reach prompts, scored only. Draws are fresh, T 1.0, top_p 1.0, 32,768 tokens, Dan's sampler (`olmo_thinker` template), strict checker. Seeds: 710001 (A1), 710002 (A1u), 710003 (A2), 710010 and 710011 (ExPO dev, gamma 3.5 and 4.6), 710012 (ExPO held-out); 2026-11-25's seeds were 640000 + step, so none overlap. Reference draws: stored dpo, s0500, s2300 from Dan (not re-sampled).

**Order of operations on the pod (shared session; setup and the dpo, s0500, s2300 downloads are charged to M1/M3, not here).**
1. Download s0100 (M2's only extra download; for the noise-floor trajectory and cos checks).
2. **Weight-space pass W** (GPU, fp32, differences formed exactly from bf16): for each of the 448 matrices and Delta in {100, 500, 2300}: ||Delta||_F, randomized top-k SVD (k = 16, with a small subset by exact SVD for validation), sigma_1, energy fraction sigma_1^2 / ||Delta||_F^2, alpha = ||Delta||_F / sigma_1; the sign-randomized null sigma_1(Delta o S) (one S per matrix, seeded); sign-aligned u_1 and the cosines of u_1, v_1 across 100 / 500 / 2300; cos(Delta_500, Delta_2300) per matrix and in total; gamma_proj = <Delta_500, Delta_2300> / ||Delta_500||^2; distance ratios for gamma in {3.17, 3.5, 4.6}.
3. **Gate rules after W.** G1: weighted median sigma_1 / null for Delta_2300 >= 1.5 -> read A1 and A1u as tests of Cai's property; 1.1-1.5 -> run, label "noise-dominated"; < 1.1 -> skip A1 and A1u, report. G2: median alpha > 10 -> A1 pilot of 32 draws under the abort rule before the full cell. G3: nothing in (b) gates ExPO; it is reported. Print W's summary before any sampling; the orchestrator decides if G1 fails.
4. **Build.** Adapters (rank 1, r = lora_alpha = 1, the 7 target modules, fp32 files as in 2026-11-25): A1 (B = sqrt(c) u_1, A = sqrt(c) v_1^T with c = ||Delta||_F), A1u (c = sigma_1), A2 (random, c = ||Delta||_F). Checkpoints: ExPO gamma 3.5 and gamma 4.6, each theta = dpo + gamma (s0500 - dpo) combined in fp32 and rounded to bf16, all 707 tensors, same shard layout; verify three random tensors against an fp32 recompute and the index total_size.
5. **Teacher-forced LoRA gate** (the 2026-11-25 `lora_check.py` pattern, copy not edit): on the 2 check examples, vLLM base nats, vLLM with A1, vLLM base again, against a PEFT/HF unmerged reference for the same adapter. PASS if the change of nats agrees within 2 points, base nats are reproduced after the adapter ran (0.5 %), and per-token log-prob differences correlate > 0.9 with PEFT's. For A1u and A2 the check is "applied" (change beyond the base's bf16 repeatability). If it fails, stop and report; no sampling.
6. **ExPO collapse pilot:** 16 D prompts x 1 draw per gamma (about 0.22 Mtok). Abort a gamma if cap hits > 25 % or missing end of thinking > 10 %. If both abort, report ExPO as collapsed and stop B.
7. **ExPO dev:** D 64 x 2 per surviving gamma (seeds 710010 / 710011). Pick the higher pass@1 if it leads by >= 2 points, else the smaller gamma; freeze in a timestamped file; delete the unchosen checkpoint.
8. **Held-out cells:** A1 (H x 2 + R x 4), A1u (H x 2 + R x 2), A2 (H x 1 + R x 2), ExPO chosen gamma (H x 2 + R x 4). One adapter per run (separate dpo+LoRA load) with Dan's sampler unchanged; one engine serving A1, A1u, A2 via our own driver is the first cut (below).
9. **Conditional (own trigger and Leon's go beyond the cap):** E (A1 < 80 % of G on R, gate passed; R x 4 + H x 1); rank-16 / rank-64 adapters (A1 recovers 25-75 %); A3 + A4 (A1 >= 50 % of G on H and R and W's held-out prediction cosine >= 0.3).

**Analysis.** For each cell and set: pass@1 (% of prompts x draws), paired prompt-bootstrap 95 % interval against `dpo` and against `s2300`; recovered fraction f = (p_cell - p_dpo) / (p_s2300 - p_dpo) with a paired prompt bootstrap (2,000 resamples of prompts; dpo and s2300 per-prompt rates from stored draws), G_H = 32.0 points (dpo 32.5, s2300 64.5), G_R = 69.4. R also "any pass in 4". Cai's ratio p_cell / p_s2300 printed beside f. Expected f half-widths from your simulation: about +-14 points at H x 2, +-9 at R x 4, +-19-21 at H x 1, +-12-13 at R x 2: separates 0.5 from 0.8 for A1 and ExPO, and "about 0" from "at least 0.4" for A1u and A2. Controls are read against A1 on the same prompts.

**Cost (M2 only; shared setup and dpo / s0500 / s2300 downloads not charged; 1,400 tok/s, $5.29/h, 5-min load $0.45; 6,900 tokens per draw for the s2300-like cells, 5,000 for A1u, 4,500 x 1.3 for A2).**

| item | draws | Mtok | min | $ |
|---|---|---|---|---|
| download s0100 | | | 1.5 | 0.15 |
| W (448 matrices x 3 deltas, sign null, inner products, 3 exact SVDs, build adapters) | | | 19 | 1.7 |
| A1 + A1u + A2 loads (3) | | | 15 | 1.35 |
| A1 (H x 2 + R x 4) | 416 | 2.87 | 34 | 3.0 |
| A1u (H x 2 + R x 2) | 320 | 1.60 | 19 | 1.7 |
| A2 (H x 1 + R x 2) | 208 | 1.2 | 14 | 1.3 |
| ExPO: 2 writes + 2 loads | | | 20 | 1.8 |
| ExPO pilot (16 x 2) | 32 | 0.22 | 3 | 0.2 |
| ExPO dev (64 x 2 x 2) | 256 | 1.77 | 21 | 1.85 |
| ExPO held-out (H x 2 + R x 4) | 416 | 2.87 | 34 | 3.0 |
| **total** | 1,648 | 10.5 | about 180 | **about 16.1** |
| +20 % contingency | | | | 3.2 |

Range $13-25. Conditional extras: E about $3.1; rank-16 or rank-64 about $3.5 each; A3 + A4 about $5.

**Cut order if M2 is capped at $15** (cuts are labelled in the write-up):
1. One engine serving A1, A1u, A2 through a small driver of our own (max_loras = 3, a LoRARequest per prompt): -$0.9 ($15.2). This needs new code and the teacher-forced gate must cover multi-adapter batches; if that costs more than the saving in my time, skip it.
2. ExPO dev at D 32 x 2: -$0.9 ($14.3).
3. ExPO held-out H x 2 -> H x 1: -$0.8 ($13.5) before touching ExPO's tuning.
4. Last, one pre-committed gamma = 3.5 with only the pilot: -$2.3 more; ExPO is then untuned and the write-up says so.
Not to be served from M2's LoRA-enabled dpo engine: any of M1's base cells (a non-LoRA engine produced Dan's stored draws) unless the teacher-forced check shows identical base logits.

### E. OPEN points still disputed

1. **W's stop threshold:** 1.1 (mine: below it sigma_1 is the null's) vs 1.5 (yours: A1 should not be read as a test of the paper). Both are guesses; both rules print the distribution.
2. **Fig. 7 inference:** accepted as evidence on scope, but still an inference; I do not agree it settles that non-matrix tensors were left at base.
3. **A2 as a weak null:** Gaussian accepted for Stage 1; a layer-shuffled or activation-matched control only if A1 succeeds.
4. **Resolution of the ExPO dev choice:** at D 64 x 2 a 2-point difference is within noise; I treat the choice as a convention, and the cut order's step 4 shows its value is small.
5. **Which conditional Q2(a) cell** (A3 as specified, or the auditor's "AlphaRL-lite" with kappa chosen on D): undecided, only matters if triggered.
6. **Single multi-adapter engine vs reload per adapter** (about $0.9): depends on whether the driver is worth writing.
7. The AlphaRL code release is unread by both of us.

## Turn 4 — auditor (Claude Opus 5.5), 2026-11-27 02:17 PT

This turn covers Turn 3 and the dated correction at the end of the Cai note. I also checked four new things, saved
under `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/audit/`:

| file | what it shows |
|---|---|
| `rank1_baseline_tails_output.txt` | Cap-hit and missing-end-of-thinking rates in Dan's stored plain draws. |
| `rank1_bbp_gate_output.txt` | A spiked-matrix map from W's ratio to singular-vector overlap. |
| Dan's `olmo_ifladder/sample.py` l. 34-35 and 255-265 | How rows complete, and how LoRA is configured. |
| `dpo_config_fetch/config.json` | dpo's config at the pinned commit. Its blob `cb8feeab` matches `checkpoints.csv`. Against the RL config (`55a75497`) it differs only cosmetically: `beta_fast` 32 vs 32.0, `beta_slow` 1 vs 1.0, `use_cache` true vs false. Which config an ExPO checkpoint carries does not matter. |

### 1. Are the nine Turn-2 changes in the revised design as written?

Yes, all nine are there:

1. **A1u.** "A1u (unscaled sigma_1 u_1 v_1^T): accept, Stage 1 ... H x 2 + R x 2"; D.4 "A1u (c = sigma_1)".
2. **A2.** "A2 as specified: accept (Gaussian g, h in fp64, per-tensor seed from a stable hash, magnitude ||dW||_F,
   B = sqrt(c) g/||g||, A = sqrt(c) h^T/||h||, lora_alpha = r = 1), trimmed to H x 1 + R x 2".
3. **A3 + A4 conditional.** "Trigger as you wrote (A1 >= 50 % of G on H and R, and the W prediction cosine >= 0.3).
   I drop `s0050`".
4. **ExPO in Stage 1.** D.6-D.8, with gamma in {3.5, 4.6}, "freeze in a timestamped file; delete the unchosen
   checkpoint", and held-out "H x 2 + R x 4".
5. **W as a gate.** D.2-D.3 with (a)-(d), except the stop threshold. I rule on that in section 2.
6. **E.** "Run E iff the noise gate passed ... and A1 < 80 % of G on R. Cost about $3.1".
7. **Abort rule.** "cap hits > 25 % or missing end of thinking > 10 % in the first 32 draws". As written, but the
   rule is flawed, and the flaw was mine. See 3a.
8. **Fraction of G.** "G_H = 32.0 points (dpo 32.5, s2300 64.5), G_R = 69.4"; "Cai's ratio ... printed beside f".
9. **Volume.** "about 390 GB ... 450 GB if E runs; the unchosen ExPO checkpoint is deleted".

I also accept your corrections on the readings. Nothing in A remains disputed. On Fig. 7 we agree: it is evidence
about scope, not a settlement.

### 2. Ruling on W's stop threshold: your 1.1, with labels that carry a meaning

My 1.5 was a guess. Here is a way to read the ratio. In the spiked-matrix model (Benaych-Georges and Nadakuditi 2012,
iid noise), σ1 over the bulk edge maps to how well the estimated rank-1 matrix u1 v1^⊤ aligns with the planted one.
The sign-randomised σ1 stands in for the edge.

| ratio | cosine of the rank-1 matrix with the planted one (q/o 5120², k/v 1024×5120, MLP 27648×5120) |
|---|---|
| 1.02 | 0.32-0.33 |
| 1.05 | 0.46-0.47 |
| 1.10 | 0.58-0.59 |
| 1.20 | 0.70-0.71 |
| 1.50 | 0.84-0.85 |

- At 1.1 about a third of the estimated rank-1's energy lies along the signal. At 1.5 it is about 70 %.
- Below 1.1, A1 is mostly a random rank-1 at full norm, close to A2 by construction. Skipping A1 and A1u there is
  right, so **I accept 1.1 as the stop**.
- The labels should carry the meaning: below 1.1 "stop"; 1.1-1.5 "noisy top direction, implied cosine 0.58-0.84";
  1.5 and above "signal-dominated".
- Caveat: bf16 rounding is a dead-zone quantiser, neither iid nor independent of the signal. The map is a guide only.

Two additions belong to the gate (Required change 3):
- Print the fraction of ‖Δ2300‖² that sits in matrices below 1.1, next to the weighted median.
- Randomised SVD converges slowly on flat spectra, so it underestimates the null's σ1 and inflates the ratio.
  - Use at least 6 power iterations for both σ1 and the null.
  - Validate against an exact SVD on at least 3 matrices, including one sign-randomised null, to within 0.5 %.
  - The extra cost is negligible.

### 3. New points

**3a. The abort rule and the separate pilots do not work with Dan's sampler (my rule had the same flaw).**
- `sample.py` l. 34-35: "finished requests are appended to a gzip JSONL as they complete". All requests of a cell run
  concurrently (`max_num_seqs` 512).
- So the first 32 rows to appear are the shortest draws, and cap hits arrive last. An abort rule on "the first 32
  draws" almost never fires.
- A separate 16-prompt pilot is not 3 min ($0.2). It costs:
  - its own load, because each sampler call builds its own `LLM`;
  - a tail of one 32k-token sequence decoded at small batch, about 10-15 min of wall time (a guess at per-sequence
    decode speed).
  That is about $1-2 per γ.
- Stored baselines (`rank1_baseline_tails_output.txt`):
  - cap hits are ≤ 0.7 % and missing end-of-thinking ≤ 0.7 % at every stored checkpoint (dpo, s0100, s0500, s1100,
    s2300) on D, arm A and R;
  - D's mean length is 4,044 (s0500) and 4,837 (s2300) tokens.
  A collapse is therefore unmistakable once a cell finishes or is killed.
- **Replacement.**
  - Drop both the ExPO pilot and the A1 "first 32 draws" pilot.
  - Run every cell under a spend watchdog: kill it when its generated tokens (or wall time × measured throughput)
    reach 1.5 × its planned tokens before 95 % of its draws are done.
  - Count killed in-flight draws as cap hits.
  - Label the cell "collapsed" if cap hits plus missing end-of-thinking exceed 5 % (baseline ≤ 0.7 %, so the
    false-alarm risk is nil).
  - This bounds any overrun at +50 % of one cell. It also gives the collapse diagnosis ExPO's §4.3 predicts.

**3b. The teacher-forced gate.**
- On the 2026-11-25 gate's power: in `lora_check.py`, the drop-ratio test applies only "where PEFT's drop is at least
  2 % of the base nats". Otherwise the only test is |vLLM change − PEFT change| < 2 points.
  - An adapter that is not applied shows zero change, so it passes whenever the true change is under 2 %.
  - A1u and A2 will be under 2 %, and so will A1 if it is a null. **So at rank 1 the old gate has no power.**
- Your revision ("change of nats within 2 points, base reproduced 0.5 %, per-token log-prob differences correlate
  > 0.9 with PEFT's"; for A1u and A2 just "applied") has two problems.
  1. Its power depends on A1 working. A correctly applied A1 that is a behavioural null has per-token effects near
     bf16 cross-stack noise, so it fails the 0.9 correlation and blocks a legitimate null.
  2. The PEFT reference needs a 64.5 GB HF load on the pod with the GPU free of vLLM. That is about $0.9, missing
     from the cost table.
- **Replacement, with power that does not depend on A1's behaviour.**
  - (i) Offline writer tests on serrano (synthetic and small, one core):
    - B@A·(α/r) reproduces c·u1 v1^⊤ in fp64 to within 1e-6 relative from the fp32 files, and within 1e-3 after a
      bf16 cast;
    - for A1 and A1u, ⟨B@A, ΔW⟩ / (‖B@A‖ ‖ΔW‖) equals σ1/‖ΔW‖_F, which catches a transposed u v^⊤ in the square
      q and o matrices; for A2 it is about 0;
    - the 896 tensor names equal the 2026-11-25 adapter's.
  - (ii) A vLLM-only consistency gate in one eager engine. The same matrix m·c·u1 v1^⊤ is written two ways:
    - r = 1 (padded to `max_lora_rank`);
    - r = 16 with 16 equal columns of √(mc)/4 each and `lora_alpha` 16. Dividing by 4 is exact in bf16, so both
      files carry identical mantissas.
    - The r = 16 path is the one PEFT-verified on 2026-11-25 (r 16 there, α 32; same code).
    - Choose m from {1, 4, 16} so the change is at least 5 % of base nats on the two check examples (the
      2026-11-25 pattern: pool test prompts teacher-forced on the final model's responses).
    - PASS: summed-nats changes agree within 1 %, per-token change correlation ≥ 0.99, base reproduced within
      0.5 %.
    - This tests rank-1 padding, scaling and packing with full power, whatever A1 does behaviourally.
  - (iii) In the same engine, an "applied" check for A1, A1u and A2: a nonzero, deterministic change of nats, with
    the base identical before and after.
  - Cost: one eager load plus minutes, about $0.7. It replaces the HF/PEFT step, which would have cost about $0.9.

**3c. Adapter construction: accepted, with two notes.**
- The construction is right:
  - balanced split B = √c·u1, A = √c·v1^⊤ in the PyTorch [out, in] layout;
  - r = `lora_alpha` = 1, so the scale is 1;
  - fp32 files, cast by vLLM to bf16 (about 2^-9 relative per entry; entries are about 0.01, well inside bf16's
    normal range);
  - the 7 target modules, which vLLM 0.11.0 packs as qkv_proj (q, k, v slices, each with its own A and B) and
    gate_up_proj (`olmo2.py` l. 377-386). Different v1 vectors per slice are supported; the trained rank-16
    adapters used the same path.
- Dan's sampler passes `max_lora_rank = --lora_rank` (default 16, l. 259), so rank-1 adapters are padded. That is
  exactly what 3b(ii) checks.
- Each Δ is formed exactly in fp32 from bf16. SVD in fp32 is enough for σ1. Use fp64 for the inner products in W.
- ExPO checkpoints: the dpo and RL configs differ only cosmetically (see the top of this turn). Either one, with a
  self-consistent index, is fine. Verifying three tensors against an fp32 recompute is enough.

**3d. Order of operations on the pod.**
1. Download s0100, then run W on the GPU.
2. Apply the gate decision automatically by the pre-registered bands. Do not wait for the orchestrator with the pod
   idle ($5.29/h): print the summary and proceed. If the ratio is below 1.1, skip A1, A1u and A2, go to ExPO, and
   report.
3. Run the eager gate engine (3b ii and iii).
4. **A1 first. A1u and A2 depend on A1** (new):
   - If A1 is a clean null (f < 0.25 on H and R, no collapse signature), skip A1u and A2. A1u is A1 divided by α
     along the same direction, so it is predictably null when A1 is, and A2's specificity question is moot. The
     saved about $3.9 pays for one conditional cell within the cap.
   - If A1 succeeds or is partial, run A2, then A1u.
   - If A1 collapses, run A1u (now the informative cell) and A2 (whether the norm alone breaks the model).
5. **Write both ExPO checkpoints in the background while A1, A2 and A1u sample.** It is CPU and disk work. This saves
   about 10 min of idle GPU, about $0.9.
6. ExPO dev with γ = 4.6 first (the riskier cell, under the watchdog), then γ = 3.5. The tie default is then the
   engine already loaded for held-out, so no reload in the likely case.

**3e. The ExPO dev rule.**
- D 64 × 2 per γ. From the stored draws, the s0500 → s2300 gap on D is only 18 points (60.2 → 78.3). The paired SE
  of a γ-difference is about 5-6 points (my estimate), so your "≥ 2 points" picks on noise.
- Rule: choose 4.6 only if its D pass@1 leads 3.5 by more than one paired prompt-bootstrap SE and it has no collapse
  signature; otherwise 3.5.
- Write it up as a collapse screen with a strong-evidence override, not as tuning. This matches your own "the
  held-out result is not read as the best gamma". Cost-neutral.

**3f. The recovered fraction and its interval.**
- The analysis is right: paired prompt bootstrap, f unclipped, ratio beside it. The prompt bootstrap of per-prompt
  means already carries the within-prompt binomial noise of both the cell and the stored references. My simulation
  gave ±14 points (H × 2) and ±9 (R × 4).
- Two additions:
  - (i) For ExPO, report f′ = (p − p_s0500) / (p_s2300 − p_s0500), from stored s0500 draws: H 44.4 → 64.5, R 4.5 →
    69.4. That is where the papers disagree: AlphaRL's 20 % rows close 33-43 % of the gap, ExPO's Table 2 closes
    ≥ 100 %. R × 4 resolves f′ to about ±10 points; H × 2 only to about ±22.
  - (ii) Read A2 against stored dpo as the zero-point check of the LoRA-enabled engine. If A2 is off dpo beyond its
    interval, an engine offset cannot be told apart from a random-direction effect. Both are "non-specific", and the
    write-up says so.

**3g. Multi-adapter driver: ruling, reload per adapter with Dan's sampler unchanged.**
- With the dependent order of 3d.4, only one to three adapter runs happen. The saving is at most about $0.9, and
  less if reloads from a warm page cache take 2-3 min (a guess).
- The driver is new code on the measurement path, and its routing would need its own gate. Drop it from the cut
  order.

### 4. Cost and cut order under a $15 cap (M2 only; Turn 3's assumptions)

| item | $ |
|---|---|
| s0100 download | 0.15 |
| W (more power iterations; negligible extra) | 1.7 |
| eager gate engine (3b) | 0.7 |
| A1 (load + H × 2 + R × 4) | 3.45 |
| A2 (load + H × 1 + R × 2), not run in the clean-null branch | 1.75 |
| A1u (load + H × 2 + R × 2), not run in the clean-null branch | 2.15 |
| ExPO: background writes, 2 loads | 0.9 |
| ExPO dev D 64 × 2 × 2 | 1.85 |
| ExPO held-out H × 2 + R × 4 | 3.0 |
| **success, partial or collapse branch** | **15.65** |
| **clean-null branch** (no A1u or A2; one conditional cell instead, rank-16 at about $3.5 or E at about $2.65) | **11.75 + conditional ≈ 14.4-15.25** |

Against Turn 3's $16.05: I removed the separate pilot and the GPU-idle writes (−$1.1) and added the eager gate
(+$0.7). The HF/PEFT load (+$0.9) was never in your table.

Cut order:
1. ExPO dev at D 32 × 2: −$0.9 (success branch $14.75).
2. A1u at H × 1 + R × 2: −$0.6 ($14.15).
3. ExPO held-out H × 2 → H × 1: −$0.8 ($13.35). R × 4 carries ExPO's discrimination.
4. Last: γ = 3.5 pre-committed with no dev cell: about −$1.4 more. ExPO is then untuned, and the write-up says so.

Never cut A1's H × 2 + R × 4, A2's R × 2, W, or the gate. Hard stop at $15 of cumulative M2 spend: before each cell,
apply the next cut if the planned cost would cross. Beyond that, only on the orchestrator's go.

### Required changes

1. **Watchdog instead of pilots** (3a). Drop the ExPO pilot and the A1 first-32 pilot. Each cell runs under the
   1.5 × planned-tokens kill rule, with killed in-flight draws counted as cap hits and "collapsed" if cap hits plus
   missing end-of-thinking exceed 5 %. Saves about $0.2 on paper (about $1-2 in reality) and bounds overruns.
2. **Gate** (3b). Offline writer tests; the vLLM-only r = 1 vs r = 16 consistency gate on an amplified copy (≥ 5 %
   change, within 1 %, correlation ≥ 0.99, base within 0.5 %); "applied" checks for A1, A1u and A2; no HF/PEFT load.
   About $0.7, against about $0.9 for the unbudgeted PEFT route.
3. **W bands** (2). Stop below 1.1. Label 1.1-1.5 with the implied cosine (0.58-0.84, iid model). Print the share of
   ‖Δ2300‖² below 1.1. At least 6 power iterations, with exact-SVD validation of σ1 and the null within 0.5 %. The
   decision is automatic, with no idle wait. Cost-neutral.
4. **Dependent order** (3d.4). A1 first. A1u and A2 only if A1 is not a clean null. In the clean-null branch the
   saving funds one conditional cell. Saves up to $3.9.
5. **ExPO logistics and rule** (3d.5-6, 3e). Background checkpoint writes during adapter sampling; γ = 4.6 dev first;
   choose 4.6 only on a lead > 1 paired SE with no collapse, else 3.5; written up as a collapse screen. −$0.9.
6. **Analysis additions** (3f). f′ from s0500 for ExPO; A2 as the LoRA engine's zero-point check. Cost-neutral.
7. **Budget** (4). The cost table and cut order above, a hard stop at $15 cumulative, and the multi-adapter driver
   removed from the cut order.

**Once these are applied I have no remaining objection to running the design.**

### Points I would accept being left OPEN

1. A2 is a weak null (isotropic v). A layer-shuffled or activation-matched control comes only if A1 succeeds.
2. Which single conditional cell runs within the cap in the clean-null or partial branch: rank-16 (my preference; it
   answers "how many directions" and compares with the trained rank-16 LoRA's 63.5 %) or E (bounds lm_head, 1.45 % of
   ‖Δ2300‖²).
3. Which Q2(a) cell if triggered: A3 as specified, or AlphaRL-lite with κ chosen on D.
4. The Fig. 7 scope inference stays an inference.
5. The ratio-to-cosine map assumes iid noise; bf16 dead-zone rounding is not iid.
6. Reload times from a warm cache, and the per-sequence decode speed behind the tail estimates, are guesses.
7. The AlphaRL code release is unread by both of us.

## Turn 5 — reader-implementer (Claude Sonnet 5.5), 2026-11-27 02:20 PT

I accept all seven required changes of Turn 4. Section B is the FINAL design (self-contained, supersedes Turns 1 and 3 where they differ).

### A. Replies to the seven

1. **Watchdog instead of pilots: accept.** I had not seen that Dan's rows are appended on completion (`sample.py` l. 34-35, 512 concurrent sequences), so "first 32 draws" is the shortest 32. Implementation under the pod tooling: in-flight tokens are not observable, so the watchdog is the job timeout of `m2_pod.sh job`, set to 1.5 x planned tokens / 1,400 tok/s + 600 s for load (table in B.5). On timeout the rows written so far are kept; every planned draw without a row counts as a cap hit. "Collapsed" = (cap hits + draws without the end-of-thinking token) > 5 % of planned draws (stored baselines <= 0.7 %).
2. **Gate: accept.** The vLLM-only consistency gate (r = 1 vs r = 16 with 16 equal columns sqrt(m c)/4 and lora_alpha = 16, scale 1) plus offline writer tests plus "applied" checks. One note: both variants run through the same punica kernels, so the gate tests padding, scale and packing, not the kernel itself; the kernel is covered by the 2026-11-25 PEFT check at r = 16. No HF/PEFT load.
3. **W bands: accept.** Stop below 1.1; 1.1-1.5 "noisy top direction (implied cosine 0.58-0.84, iid model)"; >= 1.5 "signal-dominated". Share of ||Delta_2300||^2 in matrices below 1.1 printed. >= 6 power iterations for sigma_1 and the null; exact-SVD validation within 0.5 %. The script prints the verdict and exits with a code; you start the next job by hand.
4. **Dependent order: accept** (A1 first; A1u and A2 only if A1 is not a clean null; clean null = f < 0.25 on both H and R and no collapse). One wording requirement: in the clean-null branch A1u is *inferred* null, not measured, and the write-up says so, because A1u is Leon's literal question.
5. **ExPO logistics and rule: accept** (background writes, dev 4.6 first, choose 4.6 only on a lead of more than one paired prompt-bootstrap SE and no collapse, else 3.5; written up as a collapse screen). My 2-point convention is withdrawn.
6. **Analysis additions: accept** (f' for ExPO from stored s0500; A2 as the zero-point check of the LoRA engine).
7. **Budget: accept** (table, cut order, hard stop $15 cumulative; multi-adapter driver dropped). One adapter run = one separate `dpo` + LoRA load with Dan's sampler through `sample_wrap.py`, unchanged.

### B. FINAL design (M2 `rank1`)

**B.0 Questions.** Q1: does the per-matrix top singular direction of dW = s2300 - dpo, put on dpo, keep s2300's strict-checker behaviour? Q2: does extrapolating dpo -> s0500 reach s2300's behaviour? Checker passes only; no claim about genuine instruction following.

**B.1 Sets and files** (all in `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/inputs/m2/`, pushed by `m2_pod.sh push`; builder creates them from the existing lists, not by hand):
- `pids_D.txt`: the 64 prompts of `/scratch/cowifornia/nd-rl/olmo_elicit/pool/train.jsonl` (arm-A non-reach). Dev only.
- `pids_H.txt`: Dan's `olmo_ifcurve/pids_armA.txt` (218) minus `pids_reach.txt` members (42 of them are in arm A) minus D = 112 prompts (includes the 32 pool-test prompts). Held-out arm A.
- `pids_R.txt`: Dan's `pids_reach.txt`, 48 prompts, plain, as arm B sent them. Scored only; never used to choose anything.
- Problems file: the one the stored draws used for each set (arm A and reach: builder confirms with `analyze_arms.py`, writes the choice into the inputs MANIFEST). Grader: Dan's `grade.py` strict via `tools/grade_job.sh`.
- Sampling: `sample_wrap.py`, T 1.0, top_p 1.0, 32,768 tokens, `olmo_thinker` template, `--chunk` as Dan's, `--lora <dir> --lora_name <cell> --lora_rank 16` (rank-1 adapters are padded; the gate checks that).
- Reference draws: Dan's stored dpo, s0500, s2300 (not re-sampled): G_H = 32.0 points (dpo 32.5, s2300 64.5), G_R = 69.4 (0.0 to 69.4); s0500: H 44.4, R 4.5.
- Seeds: A1 710001, A1u 710002, A2 710003, ExPO dev gamma 4.6 710010, gamma 3.5 710011, ExPO held-out 710012. (2026-11-25 used 640000 + step.) A2's random vectors: seed = 710003 + crc32(tensor name), fp64 normals.

**B.2 Order of operations on the pod** (every step a separate job started by hand; the shared session, setup, and the dpo / s0500 / s2300 downloads belong to M1 and M3; M1 uses no LoRA-enabled engine):
1. `download s0100` (the only M2 download). Weights present then: dpo, s0100, s0500, s2300.
2. **Job W** (GPU, fp32, timeout 3,600 s). Per matrix of the 448 and per Delta in {100, 500, 2300}: Delta = bf16 difference formed exactly in fp32; ||Delta||_F; top-16 randomized SVD with >= 6 power iterations: sigma_1, energy fraction, alpha = ||Delta||_F / sigma_1; the sign-randomized null sigma_1(Delta o S) (seeded S, same iterations); sign-aligned u_1, v_1 and their cosines across 100 / 500 / 2300; cos(Delta_500, Delta_2300) per matrix and total, gamma_proj = <Delta_500, Delta_2300> / ||Delta_500||^2 (fp64 inner products), ||dpo + gamma Delta_500 - s2300|| / ||Delta_2300|| for gamma in {3.17, 3.5, 4.6}. **Exact-SVD check**: >= 3 matrices (including the largest 27,648 x 5,120 shape and one sign-randomized null) must agree with the randomized sigma_1 within 0.5 %, else the job exits non-zero. Also writes the 3 rank-1 factor sets (u_1, v_1, sigma_1, alpha per matrix, fp32) and the A2 random factors for B.3, and prints the verdict.
3. **Gate verdict (exit code, printed, no idle wait; I start the next job):** weighted (by ||Delta_2300||_F) median ratio r = sigma_1 / null for Delta_2300, and the share of ||Delta_2300||^2 in matrices with ratio < 1.1. r < 1.1: exit code 20 "STOP rank-1": skip A1, A1u, A2, go to step 7 (ExPO), report. 1.1 <= r < 1.5: exit 11 "noisy top direction, implied cosine 0.58-0.84 (iid map)", cells run and are labelled so. r >= 1.5: exit 0 "signal-dominated". Median alpha is printed (no pilot rule any more; the watchdog covers it).
4. **Job build** (CPU, on the pod; a copy of the offline writer tests runs first on serrano, B.3). Builds A1, A1u, A2 adapter dirs and an amplified gate copy.
5. **Job gate** (one eager engine, load about 5 min, minutes of teacher-forcing; `lora_check.py` pattern, copied not edited): the two check examples (pool test prompts teacher-forced on s2300's stored responses). (ii) consistency: the amplified rank-1 adapter m c u_1 v_1^T written as r = 1 and as r = 16 (16 equal columns sqrt(m c)/4, lora_alpha = 16, scale 1), m in {1, 4, 16} smallest giving >= 5 % change of base nats; PASS = summed nats changes agree within 1 %, per-token change correlation >= 0.99, base nats reproduced after the adapter within 0.5 %. (iii) applied: A1, A1u, A2 each give a nonzero deterministic change of nats, base identical before and after. FAIL = exit non-zero, no sampling.
6. **ExPO writes run in the background on CPU/disk** (job write_expo, `--nogpu`) while A1, A2, A1u sample: theta_gamma = dpo + gamma (s0500 - dpo) for gamma in {4.6, 3.5}, fp32 combine, bf16, all 707 tensors, dpo's shard layout and index (the dpo and RL configs differ only cosmetically); verify 3 random tensors against an fp32 recompute and the index total_size.
7. **A1 first** (B.4). **Release rule:** A1 clean null (f < 0.25 on both H and R and not collapsed) -> skip A1u and A2 (A1u reported as inferred); A1 success, partial, or collapse -> run A2, then A1u. If W stopped, none of the three runs.
8. **ExPO dev, gamma 4.6 first, then 3.5** (D 64 x 2 each, each under the watchdog); apply the rule: choose 4.6 only if its D pass@1 leads 3.5 by more than one paired prompt-bootstrap SE (2,000 prompt resamples) and 4.6 has no collapse; else 3.5. Write the choice with timestamp to `runs/expo_choice.json`; delete the unchosen checkpoint. Tie default 3.5 keeps the engine loaded last.
9. **ExPO held-out** on the chosen gamma: H x 2 + R x 4.
10. Fetch, grade, analyse on serrano (one core, `nice -n 5`, OMP/MKL/OPENBLAS threads = 1, one process at a time).

**B.3 Adapter construction.** Per matrix m of the 448 and per cell: PyTorch layout, B [out, 1] = sqrt(c) x p, A [1, in] = sqrt(c) x q^T, `r = lora_alpha = 1` (scale 1), files fp32 (vLLM casts to bf16), target modules q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, 896 tensor names identical to the 2026-11-25 adapter (checked offline), `adapter_config.json` as that run's with r = 1, alpha = 1, bias none, no DoRA, no rsLoRA. vLLM packs q/k/v into qkv_proj and gate/up into gate_up_proj with a separate A and B per slice (`olmo2.py` l. 377-386); distinct vectors per slice are supported.
- A1: p = u_1, q = v_1 of Delta_2300, c = ||Delta_2300||_F (Frobenius reading of Cai Eq. 3).
- A1u: same vectors, c = sigma_1.
- A2: p = g/||g||, q = h/||h|| (fp64 Gaussians, seed B.1), c = ||Delta_2300||_F.
- Offline writer tests (serrano, one core, synthetic and small, nice -n 5, single-thread BLAS): B A reproduces c p q^T in fp64 within 1e-6 relative from the fp32 files and within 1e-3 after a bf16 cast; for A1 and A1u <B A, Delta> / (||B A|| ||Delta||) = sigma_1 / ||Delta||_F (catches a transposed u v^T in the square q and o matrices), for A2 about 0; tensor names equal.

**B.4 Cells.**

| id | model | sets x draws | seed | planned tokens |
|---|---|---|---|---|
| A1 | dpo + rank-1 (Frobenius-rescaled) | H x 2, R x 4 (416) | 710001 | 2.87 M (6,900 / draw) |
| A2 | dpo + random rank-1, same norm | H x 1, R x 2 (208) | 710003 | 1.2 M (4,500 x 1.3) |
| A1u | dpo + unscaled rank-1 | H x 2, R x 2 (320) | 710002 | 1.6 M (5,000) |
| X-dev | ExPO gamma 4.6, then 3.5 | D x 2 each (128 each) | 710010 / 710011 | 0.88 M each |
| X-held | ExPO chosen gamma | H x 2, R x 4 (416) | 710012 | 2.87 M |

**B.5 Watchdog (timeout of `m2_pod.sh job`) and the label.** timeout = 1.5 x planned tokens / 1,400 tok/s + 600 s load: A1 3,700 s; A2 1,900 s; A1u 2,300 s; X-dev 1,550 s each; X-held 3,700 s; W 3,600 s. On kill, rows already written are kept and graded; draws without a row count as cap hits. Label a cell **"collapsed"** if (cap hits + draws missing the end-of-thinking token) > 5 % of planned draws. A collapsed cell is a result (reported with the counts), not a failure. Hard stop at $15 cumulative M2 spend.

**B.6 Analysis** (serrano). For each cell and set: pass@1 in % of prompts x draws, "any pass" on R, paired prompt-bootstrap 95 % intervals against stored dpo and stored s2300. Recovered fraction f = (p - p_dpo) / (p_s2300 - p_dpo), unclipped, paired prompt bootstrap (2,000 resamples; stored per-prompt rates), with Cai's ratio p / p_s2300 printed beside it. Expected interval half-widths (Turn 2 simulation): about +-14 points on H x 2, +-9 on R x 4, about +-19-21 on H x 1, +-12-13 on R x 2. For ExPO also f' = (p - p_s0500) / (p_s2300 - p_s0500) (stored s0500: H 44.4, R 4.5); R x 4 resolves f' to about +-10 points, H x 2 only about +-22. A2 is read against stored dpo as the zero-point check of the LoRA engine: if A2 differs from dpo beyond its interval, an engine offset cannot be separated from a non-specific direction effect, and the write-up says both are "non-specific". A1, A1u, ExPO read against A2 and against each other on the same prompts. Collapse counts for every cell. Never choose anything on R.

**B.7 Cost (M2 only; 1,400 tok/s, $5.29/h, 5-min load $0.45).**

| item | $ |
|---|---|
| s0100 download | 0.15 |
| W (6+ iterations, exact checks, factor files) | 1.7 |
| eager gate engine | 0.7 |
| A1 (load + H x 2 + R x 4) | 3.45 |
| A2 (load + H x 1 + R x 2) | 1.75 |
| A1u (load + H x 2 + R x 2) | 2.15 |
| ExPO writes (background) + 2 loads | 0.9 |
| ExPO dev (D 64 x 2 x 2) | 1.85 |
| ExPO held-out (H x 2 + R x 4) | 3.0 |
| **success / partial / collapse branch** | **15.65** |
| **clean-null branch** (no A1u, A2) | **11.75** + one conditional cell |

**Cut order** if M2 is capped at $15 (applied before a cell if its planned cost would cross the cap, and named in the write-up): (1) ExPO dev at D 32 x 2: -0.9 (14.75); (2) A1u at H x 1 + R x 2: -0.6 (14.15); (3) ExPO held-out H x 2 -> H x 1: -0.8 (13.35); (4) last, gamma 3.5 pre-committed with no dev cell: -1.4 (ExPO untuned, stated). Never cut A1's H x 2 + R x 4, A2's R x 2, W, or the gate. Contingency is not budgeted inside the cap; a +50 % overrun of one cell is the maximum the watchdog allows.

**B.8 Conditional cell (OPEN, your decision from the results).** Rank-16 adapter of the same construction (c = ||Delta_16||_F or the rank-16 truncation, decided then; about $3.5) versus E (s2300 matrices with dpo's lm_head, embeddings and norms; about $2.65 in the clean-null branch; run only if the noise gate passed and A1 < 80 % of G on R). A3 + A4 (AlphaRL) only if A1 >= 50 % of G on H and R and the W held-out prediction cosine >= 0.3 and budget remains; which A3 variant is OPEN.

**B.9 What each outcome shows.** As in Turns 1 and 3, with these updates: A1 is the test of Cai's construction as I read it (Frobenius rescale, matrices only, lm_head, embeddings, norms at dpo); A1u is Leon's literal "top direction only"; A2 is a weak null (isotropic) and the engine's zero-point check; ExPO's f' tells whether extrapolation closes the s0500 -> s2300 gap (ExPO's Table 2 says it can, AlphaRL's 20 % rows close 33-43 %). "Collapsed" for ExPO at 4.6 is the failure mode of ExPO's Sec. 4.3 and not a statement about gamma 3.5.

### C. OPEN (still undecided)
1. Rank-16 vs E as the single conditional cell (yours).
2. Which Q2(a) cell if triggered: A3 as specified, or AlphaRL-lite.
3. A2 is a weak null; a layer-shuffled control only if A1 succeeds.
4. The Fig. 7 scope inference stays an inference; the ratio-to-cosine map assumes iid noise.
5. Warm-cache reload times and per-sequence decode speed are guesses; the watchdog timeouts use 1,400 tok/s.
6. The AlphaRL code release is unread by both of us.

## Turn 6 — auditor (Claude Opus 5.5), 2026-11-27 02:22 PT (closing, round 3)

**My seven Turn-4 changes are all in section B as written.**
1. Watchdog: A.1 "the watchdog is the job timeout of `m2_pod.sh job`, set to 1.5 x planned tokens / 1,400 tok/s + 600 s for load", and B.5.
2. Gate: B.2.5 "(ii) consistency ... m in {1, 4, 16} ... PASS = summed nats changes agree within 1 %, per-token change correlation >= 0.99", plus the B.3 offline tests.
3. W bands: B.2.3 "r < 1.1: exit code 20 ... 1.1 <= r < 1.5: exit 11 ... r >= 1.5: exit 0"; B.2.2 ">= 6 power iterations ... within 0.5 %".
4. Dependent order: B.2.7 "A1 clean null (f < 0.25 on both H and R and not collapsed) -> skip A1u and A2 (A1u reported as inferred)".
5. ExPO rule: B.2.8 "choose 4.6 only if its D pass@1 leads 3.5 by more than one paired prompt-bootstrap SE ... and 4.6 has no collapse; else 3.5"; B.2.6 background writes.
6. Analysis: B.6 "For ExPO also f' ..." and "A2 is read against stored dpo as the zero-point check".
7. Budget: B.7 table and cut order; A.7 "multi-adapter driver dropped".

**Defects in section B.** Each has a mechanical fix, and none changes the design.

- **D1. Killed draws are miscounted.**
  - The problem: Dan's `sample.py` l. 277-280 submits one request per (prompt, chunk) with `SamplingParams(n=m, ..., output_kind=FINAL_ONLY)`. A row appears only when every draw of its chunk has finished. So on a timeout, short siblings of a long draw are lost too, and "every planned draw without a row counts as a cap hit" overcounts by up to 4×.
  - The fix: run M2 cells with `--chunk 1`, one draw per request. Prefix caching keeps the prompts cheap, and the draws are fresh, so nothing is lost for comparability.
  - In pass@1, score killed draws as fails, and print their count with the all-pass bound beside it.
  - Make "collapsed" count the union of draws that hit the cap, lack end-of-thinking, or have no row. Not the sum: a capped draw also lacks end-of-thinking.
- **D2. Timeouts are too tight for small cells.**
  - The problem: 1,400 tok/s is a whole-job average over about 1,000 draws. A 128-draw dev cell has a proportionally longer tail: one 32k-token draw at single-sequence speed takes about 10 min (a guess). With the stored cap-hit baseline up to 0.7 %, such a draw is plausible in any cell. X-dev's 1,550 s can therefore kill a normal cell.
  - The fix: timeout = 600 s + max(1.5 × planned/1,400, planned/1,400 + 900 s).
  - Resulting timeouts: X-dev 2,130 s, A2 2,360 s, A1u 2,640 s; A1 and X-held unchanged at about 3,700 s.
  - This costs nothing unless a cell actually runs slow.
- **D3. The kill path.**
  - The problem: vLLM 0.11 runs its engine core as a child process. A timeout that kills only the parent can leave the GPU held, and the next job then fails.
  - The fix: start each job in its own process group (`setsid`), kill the group by PGID on timeout (never `pkill`), and check with `nvidia-smi` that the GPU is free before starting the next job.
- **D4. Exit-code paths with no next step.**
  - (a) B.2.3's STOP branch says "go to step 7 (ExPO)", but ExPO dev is step 8.
  - (b) Exit 11 means "continue" but is non-zero. Either the runner records 11 as success, or W exits 0 and writes its band to a verdict file. Reserve non-zero codes for STOP (20) and for errors.
  - (c) If W's exact-SVD check fails: rerun W once with 12 power iterations. If it fails again, compute σ1 and the null for Δ2300 by exact SVD (about +20-40 min, a guess) and continue.
  - (d) If the gate fails: no adapter sampling. ExPO still runs, since it uses no LoRA. Report, and any fix waits for the orchestrator.
  - (e) If m = 16 still changes base nats by less than 5 %: try m = 64. If that is still below 5 %, report the gate as "underpowered" and follow (d).
  - (f) If both γ collapse on D: no held-out cell; report. Any other γ only on the orchestrator's go.
- **D5. The rank-16 construction is undefined.**
  - The problem: B.8 says "c = ||Delta_16||_F or the rank-16 truncation, decided then".
  - The fix: use the unscaled truncation Σ_{i≤16} σ_i u_i v_i^⊤, i.e. B = U16 S16^½, A = S16^½ V16^⊤, r = lora_alpha = 16.
  - W should save Δ2300's top-16 factors (about 0.6 GB in fp32), so the conditional cell needs no second SVD pass.
- **D6. Cost slips.**
  - (a) ExPO needs 3 loads, not 2. Each sampler job builds its own engine, so the γ = 3.5 dev engine cannot carry over into the held-out job. +$0.45.
  - (b) 19 min for W is optimistic. Reading 258 GB, about 2,700 randomized SVDs at ≥ 6 iterations, and the exact checks are more like 30-45 min (a guess), so +$1-2.3. Trim W to:
    - top-16 and the null for Δ2300;
    - top-1 and the null for Δ500;
    - top-1 only for Δ100.
  - (c) Starting about 10 jobs by hand idles the pod between them: at 5 min per gap, about $4. Either use one chain script that applies the D4 exit-code table, or budget the gaps.
  - With (a) and (b), the success branch is about $16.6-18 before cuts, so cut 4 (γ = 3.5 pre-committed) may be reached. In the clean-null branch, no conditional cell fits under $15.
- **D7. Disk.**
  - Peak weights: dpo, s0100, s0500, s2300 and both ExPO checkpoints are 6 × 64.47 GB = 387 GB (360 GiB). Add 64.5 GB if E runs, plus about 20-30 GB of environment and caches, plus whatever M3 writes.
  - s0100 is also one of M1's checkpoints (M1 sweeps dpo, s0100 and s0500), so its download is shared and M2 must not delete it.
  - The fix: `write_expo` checks `df` and refuses to start unless 2 × 64.5 GB + 20 GB is free. The pod volume should be ≥ 450 GB. Delete the unchosen ExPO checkpoint right after the D choice.
- **D8. Small specifications.**
  - The gate engine uses `max_lora_rank 16`, like the sampling jobs, so the padding is the same.
  - A2 draws g, then h, from one stream: `numpy.random.default_rng(710003 + crc32(name))`.
  - The clean-null and E rules use point estimates of f.

**Remaining objection: none.** With D1-D8 applied, I have no objection to running section B.

**Final OPEN list.**
1. Rank-16 or E as the single conditional cell.
2. A3 or AlphaRL-lite, if Q2(a) is triggered.
3. A2 is a weak null; a layer-shuffled control only if A1 succeeds.
4. The Fig. 7 scope reading and the iid ratio-to-cosine map are inferences.
5. Decode speed, tail length and reload times are guesses; D2 and D6 depend on them.
6. The AlphaRL code release is unread.
