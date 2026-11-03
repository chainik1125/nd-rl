---
written_on: 2026-11-03
written_by: agent:claude (executor, run olmo-rlzero-pilot; reviewed 2026-11-03, corrections marked by the librarian)
---

> **Reading copy (librarian, 2026-11-03).** This is the 32B decision memo of run `olmo-rlzero-pilot`, written by the
> executor. The source is fork `dan-pandori/nd-takehome`, branch `dan_olmo-rlzero-pilot`, file
> `olmo_pilot/decision_32b.md` at [`046d217a`](https://github.com/dan-pandori/nd-takehome/blob/046d217a/olmo_pilot/decision_32b.md).
> The corrections from its [review](https://github.com/dan-pandori/nd-takehome/blob/525e2e0e/review_olmo-rlzero-pilot.md)
> are applied, and each change is marked *[review]*:
> - "outputs unchanged" (Verdict item 4);
> - "none is a catalogue price" and "three verbatim" (item 6);
> - the "Found" paragraph's claims (items 2–3 and the "Not supported" note). That paragraph is split into its
>   clauses, so that each correction sits under the claim it qualifies (its "— and" became a full stop);
> - two notes on §3: what in it is projected, and that the review's additions come on top of its totals;
> - the review's additions to the 32B design (next measurement 2), added as change 8.
>
> The review checked the memo's inputs:
> - the §1 price table reproduces from the meta files and billed rates;
> - the 32B architecture matches `config.json`, and the RL settings match `32b_think_rl.sh` at bb58c66 and at `main`;
> - the review checked three of the four checkpoint identities (`step_0300`, `step_0750` and `main`; 14 / 14
>   identical LFS oids each), and all three hold. `step_0050` was not checked;
> - the RL-share numbers match rl-compute-share's tables;
> - M0's fixed phase ($55) reproduces from the memo's inputs. The rest of §3 is a model, a projection rather than a
>   measurement.
>
> See the [summary](README.md) for the reviewed numbers.

# Decision memo: the same test on Olmo 3.1 Think 32B — cost, design, and what to change

Written 2026-11-03 by the olmo-rlzero-pilot executor. Every price below is measured on this pilot or derived from
those measurements by the stated model; none is a catalogue price, *[review]* except the B200 ($5.98/h) and 2 × H100
($5.38/h) rates in §3, which were never billed in this pilot.

## 1. Measured price on this pilot (Olmo 3 7B, the RL settings)

RunPod, vLLM 0.11.0 (the version open-instruct pinned for RL), bf16 weights and KV cache, T 1, top-p 1, 16,384-token
responses. Throughput is the job average (`artifacts/olmo_pilot/meta/<job>.json`).

| job (model, GPU) | tokens / sample | cut at cap | tok/s | $/h billed | $ / M generated tokens |
|---|---|---|---|---|---|
| s0_base: base, A100-SXM4-80GB, plain decoding | 6,353 | 17.5 % | 557 | 1.39 | 0.69 |
| g_spec_base: same, n-gram speculation | 6,169 | 15.8 % | 759 | 1.39 | 0.51 |
| F_base: base, 90 problems × 16, speculation | 5,872 | 15.3 % | 777 | 1.39 | 0.50 |
| F_s0500: step_0500, × 8 | 11,790 | 25.1 % | 723 | 1.39 | 0.53 |
| F_s2800: step_2800, A100 80GB PCIe (secure) | 11,712 | 17.8 % | 659 | 1.79 | 0.75 |
| likelihood scoring (prefill only, prompt logprobs) | — | — | 13,800 | 1.39 | 0.03 |

The 7B is expensive to sample because its attention has 32 KV heads (no grouped-query attention): the cache costs
0.5 MB per token across 32 layers until a sequence passes the 4,096-token sliding window (≈ 2.1 GiB at 5k tokens), so
an 80 GB GPU holds ≈ 25 long sequences and every decode step re-reads ≈ 72 GB. Tokens per dollar then follow memory
bandwidth per dollar, which is similar across RunPod's NVIDIA classes; the community A100 SXM was the cheapest.
n-gram speculative decoding (exact speculative sampling) bought 1.36× *[review]* with no detectable change in pass@1 or
length (pre-registered gate; speculative sampling is exact only in exact arithmetic).

## 2. Olmo 3.1 Think 32B: the model and its checkpoints

- 64 layers (16 full attention, 48 sliding-window 4,096), 8 KV heads × 128 (GQA): **64 KB of KV per token** in the
  full-attention layers plus 0.75 GB per sequence in the windowed ones — about half the 7B's per sequence. Weights are
  64.5 GB in bf16, so it needs ≥ 141 GB on one GPU (H200, B200) or two 80 GB GPUs with tensor parallelism.
- RL settings (`scripts/train/olmo3/32b_think_rl.sh`): template `olmo_thinker` (with `<think>`), temperature 1.0,
  **32,768-token responses**, 64 prompts × 8 samples per step. Its RL data are the Think RL mix, not Dolci-RL-Zero-Math,
  so the contamination check must be redone against them.
- Checkpoints: the RL start `allenai/Olmo-3-32B-Think-DPO`, and `allenai/Olmo-3.1-32B-Think` `step_0050` … `step_2300`
  (46, every 50 steps), `main` = `step_2300`.
- **Step numbering.** `Olmo-3.1-32B-Think` `step_0050`, `step_0300` and `step_0750` are byte-identical (all 14
  safetensors shards) to `Olmo-3-32B-Think` `step_050`, `step_300`, `step_750`, and Olmo 3 Think 32B's `main` is its
  `step_750`. So 3.1 is one continuous RL run from the DPO checkpoint: steps 1–750 are the original Olmo 3 Think 32B
  run, steps 751–2,300 the extra 21 days on 224 GPUs that produced 3.1.
- RL compute share: ≈ 0.14 of pretraining in GPU-hours at step 2,300 (rl-compute-share's Olmo 3.1 row; GPU-hours
  overstate the FLOP share because rollouts run at low utilisation). In FLOPs, that table's step-750 bounds
  (1.8 × 10⁻³ – 7.4 × 10⁻³) scale to ≈ 6 × 10⁻³ – 2 × 10⁻² at step 2,300: 2–10× this pilot's 1.5–3.2 × 10⁻³, and at the
  top of the range the literature review found tested.

## 3. Cost of the 32B test

*[review]* Every throughput and dollar total in this section is a projection from the throughput model below. M0's
fixed phase ($55) reproduces from the memo's inputs; the rest is a model, not a measurement. The B200 and 2 × H100
rates were not billed in this pilot.

Throughput model, calibrated on §1: decode re-reads weights plus resident KV at 67 % of peak bandwidth; resident
sequences = KV pool ÷ KV per sequence; × 1.36 for speculation as measured here. At a mean response of 16k tokens
(Think models write > 10k; 8k–24k changes the totals by about −40 % / +50 %):

| GPU (RunPod, community $/h) | resident seqs | tok/s plain | $ / M plain | $ / M speculative |
|---|---|---|---|---|
| 1 × H200 141 GB ($3.59) | ≈ 34 | ≈ 880 | 1.13 | 0.83 |
| 1 × B200 180 GB ($5.98; vLLM 0.11 on Blackwell untested) | ≈ 55 | ≈ 1,830 | 0.91 | 0.67 |
| 2 × A100-SXM4-80GB, TP 2 ($2.78) | ≈ 44 | ≈ 735 | 1.05 | 0.77 |
| 2 × H100 SXM, TP 2 ($5.38) | ≈ 44 | ≈ 1,200 | 1.24 | 0.91 |
| 1 × 80 GB GPU | ≈ 4 | ≈ 125 | ≈ 6 | — |

Designs on 90 problems, at $0.83/M (H200 with speculation), 14k / 16k tokens per DPO / RL sample, extension modelled on
this pilot (resolved candidates cost ≈ 4× their fixed budget, ≈ 15 % run to K, failing samples 1.3× longer):

| design | fixed phase | confirm | extension | probe, likelihood, downloads | total |
|---|---|---|---|---|---|
| M0 — this pilot's design: DPO 16, final 8, 3 RL checkpoints × 8, extension to 256 | $55 | $4 | $32 | $18 | **≈ $110** |
| M1 — the brief's plan: DPO 32, final 16, 5 RL checkpoints (50 … 2,300) × 8, extension to 512 | $100 | $5 | $77 | $23 | **≈ $205** |
| M2 — decisive large k: as M1 with extension to 2,048 | $100 | $5 | $190 | $23 | **≈ $320** |
| brief scale: k = 512 on DPO and final, 5 × 64, extension to 2,048 | $1,530 | $5 | $770 | $76 | ≈ $2,400 |

Downloads: 64.5 GB per checkpoint (≈ 1 min on a fast pod); size the volume for two at a time.

*[review]* The review's additions (change 8 below: the symmetric confirmation of the base's resolutions, a
final-answer-only grade and a hand-check) come on top of these totals. On this pilot the review prices the symmetric
confirmation at ≈ $6–9.

## 4. What the pilot found, and what to change for the 32B

**Found (7B, 83 AIME 2024–26 problems; numbers.md § olmo-rlzero-pilot).** RL raised pass@1 from 0.094 to 0.419 and
pass@8 from 0.364 to 0.626, but every one of the 17 problems an RL checkpoint solved and the base had not (in 16) was
solved by the base when re-sampled — at draw 18 to 383, median 73.
- *[review]* "Solved" means credited by RL's verifier. Read by hand, the 17 resolving base solutions are 10 sound, 6
  flawed but arriving, and 1 a guess (aime2024_20, where all 5 RL successes are guesses too).

Every problem the base solved and step_2800 had not was solved by step_2800 the same way (draws 12–280).
- *[review]* The draw-280 success (aime2025_14) is a guess: it computes 34, then boxes 60. The base's only credit there
  is probably the verifier accepting a restated 60° angle, so on the evidence neither model solves aime2025_14.

Strong reach 0, reverse 0 at budgets up to 512.

RL's correct solutions on the late-solved problems are in the base's support (all inside its own-solution likelihood
range; per-token gap 0.06 nats at step 2,800, 0.006 at step 500).
- *[review]* This is the pre-registered per-token range. The whole-solution gap is a median 731 nats for step_2800's
  late-solved solutions.

At ≈ 10⁻³ of pretraining compute, this RL elicits.
- *[review]* This is shown at the level of RL's verifier's credit, up to 512 samples. At the level of sound
  derivations it is shown for 10–16 of the 17 problems; on the other two, the RL solutions are no better.

**Change for the 32B:**
1. **Spend on the extension, not the fixed phase.** The answer lived entirely in the base's re-sampling of the RL
   checkpoints' new solves; the fixed phase only finds candidates. Use DPO 16–32 and final 8–16 on all problems, and
   budget K ≥ 512 (better 2,048, design M2) for the DPO checkpoint on the candidates. Sequential stopping at the first
   correct chunk (chunks of 32) keeps resolved candidates cheap.
2. **Extend symmetrically, including problems the base solved only in its extension.** Here aime2025_22 looked like a
   loss for the final until the final got the same extra samples (6 correct in its first 16).
3. **Fresh confirmation samples** for every candidate (2 of step_2800's 16 candidates and 3 of step_0500's 8 had no
   fresh success).
4. **Drop AIME 2024** (6 of 30 contaminated here: three variants in the RL data, three *[review]* in midtraining, at
   8-gram containment 1.00, 0.82 and 0.52) and use AIME 2025 + 2026 + HMMT Feb 2026; rerun the n-gram and the embedding
   checks against the Think SFT, DPO and RL data (the embedding check found templated RL-data relatives the n-gram
   check missed).
5. **Gate speculative decoding** (it paid 1.36× here) and consider a second gate for an FP8 KV cache (≈ 2× more
   resident sequences), validated the same way before use.
6. **Operational:** pull outputs off pods as each job ends (a community host failure cost this pilot a job and the base's
   phase-F texts); test a real CUDA allocation on every new pod (one H200 had a broken stack); expect RunPod capacity
   only on CUDA 13.2 hosts and fall back to secure cloud; kill vLLM's `EngineCore` along with the sampler.
7. **Budget by measured lengths:** RL checkpoints wrote 2× the base's tokens; a Think model at 32k will cost ≈ 1.3–2×
   these per-sample figures even at similar $ per token.
8. *[review]* **Confirm and read the base's resolutions** (the review's next measurement 2). Without these additions,
   in the review's words, "a verifier that credits a restated angle, plus AIME's 0–999 integer answers at base budgets
   of hundreds of samples, can manufacture 'base solves it'":
   - a pre-registered secondary grade that reads only the response's final stated answer;
   - a blinded hand-check (or a judge) of every candidate-resolving solution;
   - the symmetric confirmation rule: re-sample the base on each candidate until a second correct success, as RL's
     candidates needed ≥ 2 including a fresh one.
