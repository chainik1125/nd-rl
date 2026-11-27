---
written_on: 2026-11-27
written_by: agent:claude (orchestrator, Opus 5.5)
---

# Pre-registration, M2 (`rank1`): does the top singular direction of RL's weight change keep the final model's behaviour, and does extrapolating the early change reach it?

Written 2026-11-27 2:30 am PT (`date`), before any pod was requested, any weight was read and any sample existed. The
design of record is `discourse/rank1.md` Turn 5 section B (file md5 7a770cfe52d36f9184d5474a64ba67d7, before the
auditor's closing Turn 6) and `SPEC.md`; fixes from Turn 6 are added below as a dated amendment before any pod run.
This file freezes the readings; where it is silent, section B holds.

## Construction (fixed)

- Delta_t = theta_t - theta_dpo per matrix, formed exactly in fp32 from the bf16 checkpoints, for the 448 attention
  and MLP matrices (q, k, v, o, gate, up, down per layer). Embeddings, lm_head and norms stay at `dpo` in every
  adapter cell (vLLM 0.11.0's Olmo2 class has no LoRA on lm_head).
- A1 (Cai et al. Eq. 3, Frobenius reading): per matrix sigma_1 u_1 v_1^T of Delta_2300 rescaled to ||Delta_2300||_F,
  all 448 at once, served as a rank-1 LoRA (r = lora_alpha = 1, balanced sqrt(c) split) on `dpo`.
- A1u (the literal "top direction only"): the same direction with its own size sigma_1.
- A2 (control): random unit u, v (fp64 Gaussians, seed 710003 + crc32(tensor name)) at A1's size ||Delta_2300||_F.
- ExPO: theta_gamma = dpo + gamma (s0500 - dpo), all 707 tensors, gamma in {3.5, 4.6}, written as bf16 checkpoints.

## Gates (fixed)

- W pass: per matrix the top-16 randomized SVD (>= 6 power iterations) of Delta_2300 and a sign-randomized null;
  exact-SVD agreement within 0.5 % on >= 3 matrices or the job fails. Ratio r = norm-weighted median of
  sigma_1 / null: r < 1.1 stops A1, A1u, A2 ("top direction at the bf16 noise floor"); 1.1 <= r < 1.5 runs them,
  labelled "noisy top direction (implied cosine 0.58-0.84 under an iid model)"; r >= 1.5 "signal-dominated". The
  share of ||Delta_2300||^2 in matrices below 1.1 is printed.
- Adapter gate (vLLM only, eager): the same amplified rank-1 matrix written at r = 1 and at r = 16 must give summed
  nats changes within 1 % and per-token change correlation >= 0.99, with base nats reproduced within 0.5 %; A1, A1u
  and A2 must each change nats deterministically. A failure stops all adapter sampling.

## Cells (seeds as section B.1)

| cell | sets x draws |
|---|---|
| A1 | held-out arm A H (112 prompts) x 2, reach R (48) x 4 |
| A2 | H x 1, R x 2 (only if A1 is not a clean null) |
| A1u | H x 2, R x 2 (only if A1 is not a clean null) |
| ExPO dev | 64 dev prompts D x 2, gamma 4.6 first, then 3.5 |
| ExPO held-out | chosen gamma, H x 2, R x 4 |

Clean null for A1: recovered fraction f < 0.25 on both H and R, and not collapsed. Gamma choice: 4.6 only if its D
pass@1 leads 3.5 by more than one paired prompt-bootstrap SE and 4.6 is not collapsed; else 3.5 (a collapse screen,
not tuning). "Collapsed": cap hits plus draws without end-of-thinking above 5 % of planned draws. Watchdog: each cell
is killed at 1.5 x its planned tokens; rows already written are graded, missing draws count as cap hits. Nothing is
chosen on R.

## Primary readouts

- Recovered fraction f = (p - p_dpo) / (p_s2300 - p_dpo) on H and on R, unclipped, paired prompt bootstrap against
  Dan's stored draws (G_H = 32.0 points, G_R = 69.4 points); Cai's ratio p / p_s2300 beside it.
- ExPO also f' = (p - p_s0500) / (p_s2300 - p_s0500) (stored s0500: H 44.4 %, R 4.5 %).
- A2 against stored dpo: the zero-point check of the LoRA engine; if A2 differs from dpo beyond its interval, A1's
  and A1u's effects are reported as "non-specific" unless they also differ from A2.
- Expected interval half-widths: about +-14 points of f on H x 2, +-9 on R x 4.

## Priors (written before data; not results)

- Cai et al. predict A1 close to `s2300` (their "relative accuracy" of about 96-99 %; a ratio of accuracies, not a share
  of the gain). Their full-rank reconstruction itself falls short in their Fig. 7, and their offline SFT and
  distillation rows show that a norm-matched rank-1 matrix can destroy a model.
- On the W pass: Dan's statistics say 65 % of weights end within one bf16 step of `dpo`; the orchestrator expects r in
  the "noisy" band or above for the large MLP matrices and does not know for the small k and v matrices.
- Orchestrator's guess: A1 recovers less than half of G on reach (f_R < 0.5), A1u less than A1, A2 near dpo or below
  it. ExPO: AlphaRL's tables imply 33-43 % of the s0500 -> s2300 gap and ExPO's own results at least 100 %; given
  the path's straightness of 0.10, the orchestrator expects f' below 0.5 or a collapse at gamma 4.6.

## What would and would not follow

- A1 recovers most of G on H and R while A2 does not: on this model a per-matrix rank-1 change, rescaled to the size of
  the full change, carries the checker-pass behaviour of `s2300`. Not that RL's change is rank-1 (the rescale
  amplifies it), not that lm_head, embeddings and norms are irrelevant, not genuine passes.
- A1 is a clean null: Cai's property does not hold here as constructed (A1u is then inferred, not measured).
- ExPO reaches `s2300` on R: the dpo -> s0500 direction, extended, carries the later gain; a collapse is ExPO's own
  failure mode (its section 4.3), not a statement about gamma 3.5.

## Amendment 1 (2:35 am PT by `date`, before any pod request): the auditor's closing Turn 6 fixes D1-D8

Turn 6 (`discourse/rank1.md` at md5 1d2f46540d6aeee73a1eda5590094095) found mechanical defects; with them applied the
auditor has no remaining objection. Fixed now:
- Every M2 cell runs with `--chunk 1`. "Collapsed" counts the union of draws that hit the cap, lack end-of-thinking, or
  have no row (killed), over planned draws; killed draws are scored as fails and their count printed.
- Job timeouts = 600 s + max(1.5 x planned tokens / 1,400, planned / 1,400 + 900 s).
- W exits 0 for both run bands and writes the band (signal-dominated / noisy) to `w_band.txt`; 20 = STOP (go to ExPO);
  2 = exact-SVD check failed: rerun once with 12 power iterations, then exact SVD for Delta_2300 and proceed.
- Gate: if the amplified matrix at m = 16 still changes base nats by under 5 %, try m = 64, else gate failure (no adapter
  sampling; ExPO still runs). Gate engine uses max_lora_rank 16, as sampling.
- The conditional rank-16 cell is the unscaled top-16 truncation (B = U16 S16^1/2, A = S16^1/2 V16^T, r = alpha = 16),
  built from W's saved factors.
- A2's vectors: g then h from one numpy default_rng(710003 + crc32(tensor name)).
- The clean-null and E rules use point estimates of f.
- ExPO writes refuse to start without 2 x 64.5 GB + 20 GB free; the unchosen ExPO checkpoint is deleted after the gamma
  choice. In the clean-null branch no conditional cell fits under the $15 cap.

## Amendment 2 (2:40 am PT by `date`, before any pod job): the one conditional cell, fixed by the coordinator

Fixed before A1 is sampled (coordinator's decision, 2026-11-27): if A1 is a clean null by the pre-set rule (point
estimate f < 0.25 on both H and R, and not collapsed), the conditional cell is E, the lm_head / embeddings / norms
reference (a written checkpoint with s2300's 448 matrices and dpo's other tensors; it tests whether the null can be
blamed on the serving path, which leaves those tensors at dpo). Otherwise the conditional cell is the rank-16 adapter
(unscaled top-16 truncation, Amendment 1), the second point of the rank curve. Either cell is sampled on H x 2 and
R x 4 under the same watchdog and collapse label, within the $15 M2 cap.

## Amendment 3 (4:08 am PT by `date`, before any adapter cell is sampled): the gate's repeat criterion

Observed on rp-m2a (RTX PRO 6000, vLLM 0.11.0, eager, max_loras 1; `runs/gate2` and `runs/lora_diag`): consistency passed
(m = 4; r1 vs r16 summed-nats changes agree within 0.02-0.09 %, per-token correlation >= 0.9999), and A1 changed base nats by
11 % and 2.6 % on the two check examples, but every adapter failed "repeats exactly": base calls repeat bit for bit, while two
calls of the same adapter differ at 70-90 % of tokens (max 0.25 nats per token; summed 0.01-0.08 % of base nats; per-token
correlation 0.9996-0.9997), i.e. the LoRA kernels are not bit-deterministic on this stack. The criterion was meant to catch
an adapter applied in one call and not in another (a difference about the size of the change), not kernel rounding.

Amended (rank1/lora_gate.py `applied_verdict`): an adapter repeats when its two calls agree exactly, or with per-token
correlation >= 0.999 and summed difference <= 0.1 % of base nats; it changed the model when the summed change exceeds
max(2 x base repeatability, 1e-4 x base) or, with base exactly repeatable, it differs from base at >= 10 % of tokens (an
all-zero adapter adds exactly zero). The old flag stays in gate.json as `repeats_exact`. Consistency is unchanged.

Consequence for the cells: LoRA-path noise of this size is far below T = 1 sampling noise and is present in every adapter
cell; A2 (random rank-1, same norm, same path) is the control that carries it. A2's own summed change on the two check
examples (0.2 and 1.1 nats) is within that noise, which is expected for a random direction and is reported as such.

### Amendment 3a (4:10 am PT, still before any adapter cell): margin and a direct dropped-adapter check

gate3 (Amendment 3 criterion, rp-m2a, 12:09Z) passed: A1 / A1u / A2 changed 97 / 89-97 / 81-94 % of tokens from base, repeat
correlation 0.9996-0.9997, largest summed repeat difference 0.082 % of base (A1u), against a 0.1 % tolerance set from the same
diagnosis. To keep a different GPU (the H200 gate) from failing on rounding alone, the summed tolerance becomes 0.5 % of base,
and the check it stood in for is made direct: with base bit-exact, BOTH calls of each adapter must differ from base at >= 10 %
of tokens (a call that dropped the adapter equals base exactly). A1's change (2.6-11 % of base; first written "5-11 %", corrected at 5:25 am PT after the auditor's G4) is far outside either tolerance.

## Amendment 4 (5:25 am PT by `date`, before `gate_h`, the adapter decode check and any adapter draw): fail closed, a zero-B control, an A1 repeat rule, a decode check, and fallbacks written before the H200 data exist

What I had seen when writing this: only rp-m2a data (gate2, gate3, gate4, lora_diag, Wx; numbers in Amendments 3 / 3a and
`log.md`) and the independent audit `discourse/gate_amendments_audit.md` (G1-G9). No H200 measurement exists. Binding by the
coordinator's decision after the audit.

1. **Base exactness, fail closed (audit G2 a).** The gate's three base calls must be bit-identical per token
   (`base_exact`, written to gate.json). If not: exit **33**; the Amendment 3a token rule is never evaluated on a GPU
   where base is not bit-exact.
2. **Zero-B control (G2 b).** An adapter Z = A1's files with every `lora_B` set to 0 is called twice; with base bit-exact
   both Z calls must equal base bit for bit (premise "an adapter that should do nothing does nothing" on this stack).
   If not: exit **34**.
3. **A1 partial-application rule (G2, optional item, adopted).** |sum(A1 call 1) - sum(A1 call 2)| <= 10 % of A1's own
   change from base, per example. Source: rp-m2a repeat differences of A1 were 0.26-1.84 % of its change (lora_diag,
   gate3, gate4); 10 % is about 5 x the largest. Applies in every mode.
4. **Pre-stated fallback for exits 33 / 34 (noise-floor mode; G8 item 1).** Measured instead: per example the noise
   n = max mean |delta| per token of the two repeated base calls and the two Z calls against the first base call; each
   call of A1, A1u and A2 counts as applied if its mean |delta| per token from the first base call is >= 3 x n. The factor
   3 is the pre-registered tolerance factor of the M3 pre-flight (tolerance = 3 x the measured floor), not a value fitted
   to data; n itself is measured in the same gate run. Pass = every call of every adapter applied in both examples, the
   A1 rule of item 3, and consistency. The exit code stays 33 / 34 and the fallback verdict is written to gate.json; A1
   samples only if gate_h exits 0, or 33 / 34 with `fallback.pass` true. Real failure: any adapter call within 3 x n
   (that adapter cannot be shown applied on this GPU): its cell is not sampled and I stop and report.
5. **Decode path (audit G3).** Before A1, `tools/decode_check.py lora` runs in an engine built like Dan's sampler
   (compiled, CUDA graphs, prefix cache, LoRA rank 16): 128 tokens per check prompt sampled under A1 (T 1.0, seed 4242),
   re-scored by prefill under A1 and under base in the same engine; base decode re-scored likewise. Pass = (i) decode
   matches A1 prefill: mean <= 0.0334 and p99 <= 0.345 (the rp-m2a pre-flight tolerance, 3 x the vLLM-vs-HF floor
   measured there); (ii) power: A1 prefill differs from base prefill on those tokens by mean > 0.0334; (iii) base decode
   matches base prefill within the same tolerance. No power: rerun once with 512 tokens; still no power, or (i) / (iii)
   failing: real failure, A1 does not sample, I stop and report.
6. **Anything that breaks in a way items 1-5 do not cover: stop and report to the coordinator; no amendment on the spot.**

Order on the H200 (unchanged otherwise): C1 -> S5R -> gate_h -> decode_lora -> A1 -> release.
