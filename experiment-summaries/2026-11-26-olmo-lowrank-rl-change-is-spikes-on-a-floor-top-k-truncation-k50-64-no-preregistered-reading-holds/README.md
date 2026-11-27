---
written_on: 2026-11-27
written_by: agent:claude
---

# olmo-lowrank: RL's change to Olmo 3 32B's 448 attention and MLP matrices (`Olmo-3-32B-Think-DPO` → `Olmo-3.1-32B-Think` step 2,300) is far more concentrated than noise, "spikes on a floor": its top 16 singular pieces per matrix hold 22 % of ‖ΔW‖² (noise baselines ≈ 0.75 %), and a median 227 directions per matrix stand above same-norm rounded noise, carrying 46 % of the energy; SFT's change is diffuse and DPO's the most concentrated. Served as LoRAs on DPO, RL's top-k pieces recover its IFBench / IFEval gain gradually (on the hand-gated reach rate, under the executor's / the review's labels: G 0.16 / 0.17 at k = 16, 0.61 / 0.64 at 64, 0.89 / 0.90 at 256; RL's full change in the 448 matrices, F448, 0.94 / 0.93; RL's change minus its top 16 pieces 0.50 / 0.54), so none of the three pre-registered readings holds and k₅₀ = 64

Status: completed 2026-11-27 (executor 11-26 19:44 – 11-27 05:57 UTC; pre-registration committed 11-26 20:03:04;
`OLMO-LOWRANK DONE` 05:37:11; the executor's claim-check fixes 05:57:14) and reviewed 2026-11-27 (§Recount, phase 1,
07:47:27 UTC; phase 2, errata to §Recount and §Verdict 08:31:22 UTC). **No hard-constraint violation** (the review's
policy checks); nothing is quarantined. The review's §Verdict gives no one-word verdict. It lists what stands, seven
items that must be reworded, and under "Not supported": "Nothing that a conclusion rests on. The only claims not
derivable from stored files are the second-pair zero-delta identity and its noise references (item 4)." What stands:
- **Part 1:** "Every number reproduces: S_k, baselines, stable / signal / spike medians, the LoRA overlaps and cosines,
  and the expectation scoring", from 42 seeded matrices recomputed from the raw weights with the review's own float64
  code, and aggregates over all 672 matrices from the stored spectra that this recompute validates. "RL's net change at
  2,300 is far more concentrated than noise: S₁₆ 29× Gaussian, and a median 227 directions per matrix stand above
  same-norm rounded noise, carrying 46 % of the energy." "SFT's change is diffuse and DPO's the most concentrated."
  "lora_main overlaps RL's top-16 subspaces 5× more than chance, but only 2.2× more than lora_ctrl";
- **Part 2:** "G on A reproduces exactly. G on R reproduces exactly from the executor's labels." The review's blind
  second team "gives G within 0.04 of it on every arm". "Under every label variant, none of the three pre-registered
  readings holds (each failing condition firm), and k₅₀ = 64 on both metrics." The truncations are what they claim to
  be, and F448 was built as specified;
- **Expectations:** "P1, P2, P3, E1, E2 and E5 missed; E4's high-rank call missed; E3 and E6 hit. All reported as
  such."

This summary uses the review's wording where it differs from the executor's. The plain-summary copy marks the items
that apply to it *[review]*. Items marked *[librarian]* are the librarian's own checks or arithmetic, not reviewed.

**Models.** Every number is measured on AI2's released checkpoints: bf16 safetensors, pretrained, none trained in this
run (the two LoRA adapters below were trained on `dpo` by this project's earlier olmo-ifladder run). Think chat format.
Commits as pre-registered (from `olmo-weights`' manifest; the review checked them against the Hub):
- **32B** (32.2 B parameters): base `allenai/Olmo-3-1125-32B` `main` (`c2b61dae`; same shard hashes as
  `stage3-step11921+11000+10000-nooptim`); SFT `allenai/Olmo-3-32B-Think-SFT` `main` (`a6d7f3cf`, the soup of two
  learning-rate runs, = `5e-5-step8000`); `dpo` = `allenai/Olmo-3-32B-Think-DPO` `main` (`97604023`), the RL start; RL
  `allenai/Olmo-3.1-32B-Think` `step_NNNN` for steps 50, 100, 500, 750, 1,100, 1,150, 1,600, 2,100, 2,250 and 2,300;
  `s2300` = `main` (`97ade57a`, = `step_2300` by shard hash). No change uses steps 150 or 200, which lie on an
  abandoned branch (`olmo-weights`' review).
- **7B** (7.3 B parameters; spectra only): `allenai/Olmo-3-1025-7B` `main` (`a81bae42`, = `stage3-step11921`);
  `Olmo-3-7B-Think-SFT` / `-DPO` `main` (`6ff85758` / `7b18bf92`); `Olmo-3-7B-Think` `step_NNNN` for 25, 50, 100, 500,
  525, 1,000, 1,350 and 1,375; `Olmo-3-7B-RL-Zero-IF` `main` (`3cae007c`), RL from the base.
- **LoRA adapters from `olmo-ifladder`:** `lora_main` (trained on 16 RL trajectories) and `lora_ctrl` (on 16 of `dpo`'s
  own answers); rank 16, α 32, on all 448 matrices of `dpo`; sha256 `9921a760…` / `4cf06004…`, as in olmo-ifladder's
  rows of the fork's `artifacts/MANIFEST.jsonl`.
- **Part-2 arms,** served by vLLM 0.11.0 in bf16 on one H200: T_k = `dpo` plus a rank-k LoRA holding the top k
  singular pieces of RL's change D = W_s2300 − W_dpo in each matrix (k = 1, 4, 16, 64, 256); F448 = `dpo` with the 448
  matrices replaced by `s2300`'s; C16 = F448 plus a negative rank-16 LoRA, that is `dpo` + D − D₁₆ (the review's
  erratum 2 corrects its own §1 here). The adapters were built in this run from an fp32 SVD; nothing was trained.

Natural-language output graded by the official IFBench / IFEval checkers plus a blinded hand gate; no Lean, no
`nd_verify`.

Review: [`review_olmo-lowrank.md`](https://github.com/dan-pandori/nd-takehome/blob/0aa3eff0/review_olmo-lowrank.md)
(fork `dan-pandori/nd-takehome`, branch `dan_olmo-lowrank`):
- §Recount, phase 1: [`74c8f64b`](https://github.com/dan-pandori/nd-takehome/commit/74c8f64b), 07:47:27 UTC;
- phase 2 (claims table, policy checks), "Errata to §Recount" and §Verdict:
  [`0aa3eff0`](https://github.com/dan-pandori/nd-takehome/commit/0aa3eff0), 08:31:22 UTC;
- kit: `review/olmo-lowrank-recount/` (its `README.md`; scripts `rvr_*.py`; the pod recompute `pod/rvr_pod.py`, outputs
  in `pod_out/`; outputs in `out/`; its blind hand gate in `hand/`, with `hand/POLICY.md`). `out/adversarial_check.md`
  is a sub-agent's adversarial check of the review (3 errors, 6 imprecise, 16 nits), corrected in the review's errata
  and phase 2. The answer texts `out/answers_scope.json.gz` are not committed; `rvr_screen.py` rebuilds them.

Run files at the claim-check commit [`2fbecc8c`](https://github.com/dan-pandori/nd-takehome/commit/2fbecc8c), which the
review's phase 2 read (unchanged at `0aa3eff0`). The DONE commit is
[`431e000c`](https://github.com/dan-pandori/nd-takehome/commit/431e000c); `2fbecc8c` added the executor's claim check
(`olmo_lowrank/claim_check.md`, by a sub-agent: 84 claims, 3 wrong, 6 flags, 11 rounding notes) and changed
`run_olmo_lowrank.md`, the plain summary, `log.md` and `STATUS.md` after it. The run files at `2fbecc8c`:
- `run_olmo_lowrank.md`, `numbers.md` § olmo-lowrank, `log.md` § olmo-lowrank, `STATUS.md`;
- `olmo_lowrank/`:
  - code: `plan_spectra.py`, `linalg.py`, `spectra.py`, `stage0_svd.py`, `analyze_spectra.py`, `adapters.py`,
    `zero_check.py`, `cells.py`, `hand.py`, `adjudicate.py`, `reader_task.sh`, `analyze.py`, `compute_rows.py`,
    `figures.py`, `numbers_section.py`, `mdd_sim.py`, `ckpt.py`, `shacheck.py`;
  - `README.md` (with recompute steps), `plain_summary.md`, `claim_check.md`, `handcheck_prompt_if.md`;
  - `manifest_checkpoints.csv`, `manifest_adapters.csv`, `plan/`;
  - `tables/` (per-matrix spectra `matrices_<job>.csv.gz`, `overlap_R32.csv.gz`, `changes.csv`, `by_type.csv`,
    `by_depth.csv`, `accuracy.csv`, `lora.csv`, `expectations_part1.json`);
  - `results.csv`, `summary.json`, `handcheck.csv`, `hand_cells.csv`, `hand_arm_summary.csv`, `compute.csv`;
- `artifacts/olmo_lowrank/`: `stage0/`, `meta/`, `logs/`, `queue/`, `rows/`, `hand/` (labels, batches, adjudication),
  the small files of `spectra/{R32,S32,R7}/`, `adapters_svd/norms.csv`;
- `figures/olmo_lowrank.png`, `pod/olmo_lowrank/` (pod scripts);
- the pre-registration `preregistration/olmo-lowrank.md`:
  [`02a6af37`](https://github.com/dan-pandori/nd-takehome/commit/02a6af37), 20:03:04 UTC. That is before the first pod
  request (20:09:59 in `~/pods.log`) and before any spectrum or draw. It was not amended.

Brief: [`BRIEF_olmo-lowrank.md`](../../docs/proposals/followups/BRIEF_olmo-lowrank.md). Proposal 49:
[`2026-11-26-olmo-lowrank.md`](../../docs/proposals/2026-11-26-olmo-lowrank.md), the low-rank add-on Dan paused from
Proposal 47, with a truncation test added; Dan approved it on 2026-11-26: "Approved for up to 80$." Prior work it cites:
Mukherjee et al. 2025 ([arXiv 2505.11711](https://arxiv.org/abs/2505.11711)), who report RL updates that are sparse but
nearly full-rank (not in the project's Zotero library). The run builds on:
- [olmo-weights](../2026-11-25-olmo-weights-rl-total-move-31-185x-below-sft-its-8-9x-lr-per-step-holds-only-at-25-50-step-spacing/README.md):
  the checkpoint manifest and lineage, and the cosine 0.0052 between `lora_main` and RL's change;
- [olmo-ifcurve](../2026-11-21-olmo-ifcurve-ifbench-pass1-0p29-to-0p68-sharpening-from-1100-no-abrupt-reach-switch-timing-no-order/README.md):
  the 48 reach and 218 arm-A prompts, the decoding, the stored `B_s2300` rows, and the hand-gate rules and addenda;
- [olmo-ifladder](../2026-11-09-olmo-ifladder-lora-on-16-rl-trajectories-elicits-40-41-of-48-if-reach-prompts-control-0/README.md):
  the LoRA serving code and the adapters `lora_main` and `lora_ctrl`;
- [olmo-ifomega](../2026-11-04-olmo-ifomega-reach-on-ifbench-49-strict-43-hand-and-ifeval-6-omega-test-out-inconclusive/README.md):
  the stored `F_s2300_ifb` and `F_dpo_ifb` rows on the 176 non-reach prompts.

Copies here:
- **the plain-language summary** (the brief's ≤ 500-word deliverable), with the review's corrections marked
  *[review]*: [`plain_summary.md`](plain_summary.md);
- **the run's figure**, unchanged: [`olmo_lowrank.png`](olmo_lowrank.png). Panel 1 is the top-k share against k for RL's
  net change, one RL interval, SFT's total change and the two noise baselines; Panel 2 is G against k for the T_k arms,
  with F448 and C16 to the right. The review read it in phase 2 and has no finding on it. *[librarian]* Three notes:
  - Panel 2's reach points use the executor's labels; the review's labels move each G by ≤ 0.04;
  - C16 is "RL's change minus its top 16", not "the floor" (the review's item 2);
  - Panel 1's single interval is the late one (2,250 → 2,300), so the figure does not show the concentrated mid-run
    interval.

Question (the brief): "**Part 1, spectra.** For the 448 attention and MLP matrices, how concentrated are these changes,
against a random baseline and a bf16-rounding baseline?" The changes: RL's net change at several steps, single RL
intervals, a mid-run chord, SFT's and DPO's total changes, and at 7B RL from the base as well. "How much of RL's top-16
subspace lies inside `lora_main`'s?" "**Part 2, truncation.** Does DPO plus only the top-k pieces of RL's change recover
RL's IFBench gain? And does the floor alone, C16, recover anything?"

## Configuration

- **Changes** (pre-registered; ΔW = float32(W_b) − float32(W_a), exact for bf16 values), for each of 448 matrices at
  32B (q, k, v, o, gate, up and down × 64 layers) and 224 at 7B (× 32 layers):

  | id | 32B | 7B |
  |---|---|---|
  | RL net `N<t>` | `dpo` → step t, t = 50, 100, 500, 750, 1,100, 1,600, 2,300 | `dpo` → step t, t = 25, 100, 500, 1,000, 1,375 |
  | single intervals `I_early` / `I_mid` / `I_late` | 50 → 100, 1,100 → 1,150, 2,250 → 2,300 | 25 → 50, 500 → 525, 1,350 → 1,375 |
  | chord `K` | 1,100 → 2,100 | 500 → 1,375 |
  | `SFT` total | base → SFT | base → SFT |
  | `DPO` total | SFT → DPO | SFT → DPO |
  | `ZIF` | – | base → RL-Zero-IF |

- **Spectra** (`spectra.py`, on pods): randomized SVD for the top 256 of every matrix (p 64, q 16, the cheapest setting
  that met the accuracy target at stage 0), the exact Frobenius norm, and the full spectrum of every matrix from the
  eigenvalues of the float64 Gram matrix. The pre-registration planned float64 full spectra only for a seeded subset
  (plus Part 2's fp32 SVD of N2300's 448 matrices); stage 0 found the Gram route at 78 s per 448-matrix change, so every
  matrix got one (declared in `log.md` as a deviation, "more than pre-registered"; it gives uncensored signal ranks and
  spike counts). Randomized against full, every matrix of every change (`tables/accuracy.csv`): maximum relative error
  σ₁ 5.5e-5, top-k shares 1.9e-4 (k ≤ 64), 3.8e-3 (k = 256), inside the pre-registered target (σ₁ 0.1 %; shares 2 % for
  k ≤ 64 and 5 % at 256).
- **Measures** per change and matrix: top-k shares (k = 1, 4, 16, 64, 256) and the energy-weighted whole-model share S_k
  (for RL's net change at 2,300, the share of ‖ΔW‖² that T_k keeps); stable rank ‖ΔW‖² ÷ σ₁²; **signal rank**, the
  number of singular values above σ₁ of rounding floor A; **spike count**, the number above σ₁ of floor B.
- **Baselines** (constructions pre-registered):
  - Gaussian: i.i.d. N(0, 1) of the same shape, scaled to ‖ΔW‖_F, 3 seeded draws per shape;
  - floor A, bf16 rounding at RL's single-interval size: R_A = float32(bf16_rne(W_dpo + s·Z)) − W_dpo per matrix, with
    s bisected so that ‖R_A‖² equals the mean of that matrix's three single-interval ‖ΔW‖² (32B 50-step, 7B 25-step);
  - floor B, matched-norm rounded noise: the same construction on each change's start weights, calibrated to that
    change's own norm. The pre-registration added it because floor A "cannot tell accumulated sampling noise from
    signal".
- **LoRA overlap** (32B, every `N<t>`): the overlap of RL's top-16 left and right singular subspaces with `lora_main`'s
  column and row spaces, ‖U₁₆ᵀ Q_B‖²_F ÷ 16 and ‖V₁₆ᵀ Q_A‖²_F ÷ 16, averaged (random ≈ 16 / n); the same for
  `lora_ctrl` and 4 seeded random rank-16 subspaces; and the cosines of the adapter's ΔW with D and with D_k.
- **Part 2 arms:**
  - adapters: fp32 PEFT files with B = U_k Σ_k, A = V_kᵀ and α = r, from an fp32 SVD of D on an H200. T1 and
    T4 were served at `max_lora_rank` 8 (zero padding), T256 at 256; C16 has B = −U₁₆ Σ₁₆, on F448;
  - F448: built by replacing tensors, then checked tensor by tensor: 448 tensors from `s2300` and 259 from `dpo`
    (embeddings, all norms and `lm_head`, all 259 of which differ between the two; the review);
  - the zero-delta check: `dpo` plain against `dpo` + an all-zero rank-8 LoRA (Z8), prompt log-probs of 1,451 tokens.
- **Draws:** every arm had the same 48 reach prompts (42 IFBench + 6 IFEval; olmo-ifcurve's set) × 16 and the 176 other
  arm-A prompts × 4: 1,472 draws per arm, 11,776 in all *[librarian]*. Decoding is olmo-ifcurve's: the olmo_thinker
  template, T 1, top-p 1, a 32,768-token response budget, vLLM 0.11.0, bf16, one H200, TP 1, prefix caching, chunk 8,
  engine seed 62026 for every arm, and `dpo`'s tokenizer. T256's draws are its stage-0 throughput run (240 draws on 24
  prompts) plus the continuation (1,232), as pre-registered.
- **Endpoints:**
  - `s2300` on the reach prompts: olmo-ifcurve's `B_s2300` (32 draws per prompt, engine seed 48300), re-gated by this
    run's readers;
  - `s2300` on the 176: olmo-ifomega's `F_s2300_ifb` (8 per prompt, seed 3004);
  - `dpo`: the fresh arm; on the 176 pooled with olmo-ifomega's `F_dpo_ifb` (16 per prompt, seed 3001). The stored
    `dpo` draws on the reach prompts are not used, because they selected the set.

  The stored rows use the same template, sampler path and vLLM version, but chunk 4 instead of 8: "a prefix-cache
  batching difference" (the review).
- **Grading:** strict = the official checkers (`olmo_ifomega/grade.py`). The hand gate covers the reach prompts with
  olmo-ifcurve's machinery and rules:
  - scope: per cell (arm × reach prompt), the first min(strict, 8) strict successes in draw order; for `s2300`, the
    same in `B_s2300`'s 32. A cell's estimated genuine rate = (genuine labelled + unlabelled strict × the labelled
    genuine share) ÷ draws;
  - a mechanical rule-1 screen; identity inheritance (same prompt, identical answer);
  - blinded reader sub-agents with olmo-ifcurve's reader prompt (run name changed). They see the prompt, the constraints
    with the checker's verdicts and the answer; no model, arm, step or thinking;
  - second reads of a seeded 25 % and of every `dpo`, T1 and T4 item; a split goes to a third reader;
  - the confirming rules 1–14 (olmo-ifomega 1–5, olmo-iffu 6, olmo-ifladder 7–12, olmo-ifcurve 13–14). The
    pre-registration also planned per-prompt consistency audits by sub-agents "after each round".
- **G and readings:** G = (arm − `dpo`) ÷ (`s2300` − `dpo`) on **R**, the 48 reach prompts' estimated genuine rate
  (primary), and on **A**, the 218 prompts' strict pass@1. 95 % intervals come from a paired prompt bootstrap (10,000
  replicates, percentile). Each reading is scored on point estimates, separately on R and A, and holds when it holds on
  both. A threshold comparison is firm when the 95 % interval lies on the same side. The readings:
  1. "low-rank carries it": G(T16) ≥ 0.5 and G(C16) ≤ 0.2;
  2. "high-rank": G(T256) < 0.5 while G(F448) ≥ 0.75;
  3. "outside the 448 matrices": G(F448) < 0.75;
  4. otherwise k₅₀, the smallest k ∈ {1, 4, 16, 64, 256} with G(T_k) ≥ 0.5, set beside Part 1's signal rank and spike
     count.
- **Detection limit** (pre-registration, `mdd_sim.py`, simulated from olmo-ifcurve's stored per-prompt rates): the 95 %
  half-width of G is 0.04–0.07 on R and 0.05–0.06 on A when an arm tracks `s2300` prompt by prompt, ≈ 0.10 with
  per-prompt deviations and hand-gate sampling. "A G within ≈ 0.10 of a threshold is not resolved, and two arms' G are
  distinguishable at ≈ 0.14." The draws are one engine seed per arm, matched across arms; no seed replicates.
- **Stage 0** (pod `lr-a`, 20:15–20:48 UTC; brief's gate ≤ $5):
  - downloads: 64.5 GB in 56–59 s per checkpoint, two in parallel; all 28 shards' sha256 passed;
  - SVD per 448-matrix change: randomized 66 s, exact fp32 935 s, float64 Gram 78 s;
  - vLLM 0.11.0 ran T256 at rank 256 (the log: ranks up to 512 allowed) at ≈ 900 tokens/s; the re-cost assumed
    ≈ 1,750 for plain models;
  - the zero-delta check.

  The re-cost was ≈ $56 ($61 with 10 %), under the $72 gate, so nothing was cut (`stage0/recost.json`).
- **Host and GPU events.**
  - The scheduler offered only community H200s on host 213.186.10.12, which olmo-ifcurve had blacklisted for one faulty
    GPU. The executor allowed the host after checking the GPU's remapped rows; `lr-a` and later `lr-b` ran on healthy
    GPUs there.
  - The host's faulty GPU (bus DB:00.0, 25 uncorrectable remapped rows) came up three times and was deleted each time.
    One of these pods sat unlaunched for 27 min ($1.67), because the stopped poller's command-substitution subshell
    finished creating it.
  - With no community capacity, Part 1 moved to a secure H100 80GB.
- **Pods and cost:**
  - `lr-a`: community H200 at $3.59/h, 20:15–02:43 UTC, 6.48 h, $23.27 (stage 0, adapters, T256, the F448 build, F448,
    C16);
  - `lr-b`: community H200 at $3.59/h, 21:04–04:16, 7.20 h, $25.85 (`dpo`, T1, T16, T4, T64);
  - `lr-e`: secure H100 80GB HBM3 at $4.39/h, 21:05–22:14, 1.16 h, $5.10 (Part 1: jobs R32 2,438 s, S32 527 s, R7
    962 s);
  - the three pods on the faulty GPU: $1.76; four `lr-a` creates undone at once (20:09–20:14, `UNDONE` in
    `~/pods.log`; *[librarian]* billed $0.13 together);
  - total **15.33 pod-hours, $55.98 of the $80 budget** (`podbudget`; the review: the same). The proposal estimated
    ≈ $50–65;
  - the reviewer's pod `rv-lr1`: A100-SXM4-80GB, 05:53–07:18 UTC, 1.43 h, $3.13, registered under `olmo-lowrank`; run
    and review $59.11 booked (`podbudget olmo-lowrank`: 16.76 h);
  - *[librarian]* RunPod billed **$59.79** with disk for the run and its review (unrounded $59.7944: the executor's 10
    pod ids $56.5880, the reviewer's $3.2063). Billing is complete: the balance dropped by the same amount since the
    last ledger update. See the [spend ledger](../../docs/infra/spend.md).
- **Bucket:** `hf://buckets/dan-pandori/nd-rl/olmo-lowrank/artifacts/olmo_lowrank/` (*[librarian]* 1,567 files,
  12.6 GB, listed at 08:49 UTC on 2026-11-27):
  - `samples/` and `graded/`: every draw's text, token ids and grade (8 arms);
  - `adapters/`: T1, T4, T16, T64, T256, C16 and Z8 (PEFT safetensors; sha256 in `manifest_adapters.csv`).
    *[librarian]* 12.2 GB; these are the only `.safetensors` under the run's folder, so no model weights were uploaded
    (the pre-registration: "Weights stay on pods");
  - `spectra/<job>/`: `full_*.npz` (float64 Gram spectra) and `sigma_*.npz` (randomized top 256);
  - `adapters_svd/`: the fp32 SVD spectra of D used for the adapters (the run calls them "exact"; see item 3 below);
  - `hand/items/`: the 1,400 hand-gate item texts.

  The registry rows are in `hf://buckets/dan-pandori/nd-rl/registry/olmo-lowrank/` (*[librarian]* one file, 67 rows,
  sha256 equal to the run worktree's copy). The zero-delta second pair's raw files `zero2_*.json` were never pulled
  from the pod; *[librarian]* they are in neither the bucket nor the VPS, and every pod of the run is deleted.
- **Recompute:** `olmo_lowrank/README.md` § "Recompute" (fetch the bulk files, rebuild the row caches, the screen, the
  cells and `analyze.py --reps 10000`, then `analyze_spectra.py` and `figures.py`). The review's scripts and their
  outputs are listed in its kit README.

## Observations

### What the review checked (its own code; phase 1 before reading the write-ups, phase 2 after)

- **Provenance (§1).**
  - Shard hashes of `main` against the pinned branches matched for 4 / 4 checkpoints.
  - The seeded raw-weight sample (`random.Random("olmo-lowrank review 2026-11-27 32B" / "… 7B")`) is 32B layers 6, 8,
    42 and 43 and 7B layers 19 and 31, all 7 matrix types: 28 + 14 matrices. 52 shard files (252.8 GB) passed sha256.
  - All 224 prompts equal the upstream IFBench / IFEval records. The prompt token ids match every meta of the 8 arms
    (224 / 224) and of `B_s2300` (48 / 48); the olmo-ifomega `F_*` rows store response ids but no prompt ids, and their
    meta has the same template sha256 (erratum 4). The template boundary was checked by hand on 8 rows, and the
    decoding settings are identical across arms.
  - Each arm's served adapter has the bucket file's sha256 (6 / 6); `lora_main` and `lora_ctrl` match `olmo-ifladder`'s
    MANIFEST.
  - No draw is duplicated.
- **Re-grade (§2).** Its own driver around the upstream checkers graded 20,512 rows (8 arms × 1,472, `B_s2300` 1,536,
  `F_s2300_ifb` 2,400, `F_dpo_ifb` 4,800): 0 strict and 0 loose disagreements with the run's grades.
- **Spectra from raw weights (§3).** On an A100, with code independent of the run's engine: float64 QR then the SVD of
  the triangular factor, its own bf16 round-to-nearest-even (0 mismatches against torch on 13 M values; truncation as a
  failing control), its own floor bisection, seeds and baselines. Over 532 (change, matrix) pairs:
  - ‖ΔW‖²_F agrees to 2.5e-13; the stored full spectra to 5.6e-8 σ₁; the randomized σ₁ to 6.5e-6 and share₁₆ to 3.2e-5;
  - share₂₅₆ to 3.4e-3;
  - the floors' σ₁ to ≈ 1 %, with independent random draws (floor A on the 42 matrices, floor B on 308 pairs);
  - the `lora_main` and `lora_ctrl` overlaps, left and right (N100, N1100 and N2300 on the 28 sampled 32B matrices, 84
    rows each), to 4.3e-5.

  The signal ranks and spike counts "move by a few units where the spectrum is dense near the threshold": they are equal
  in 257 / 532 and 76 / 308 pairs, with a median relative difference of 0.04 % for the signal rank.
- **Adapters (§5).** For the 28 sampled matrices of N2300, B·A from the bucket adapters against the review's float64
  U_k Σ_k V_kᵀ:

  | adapter | config (r, α) | ‖BA − D_k‖ / ‖D_k‖, max (median) | after a bf16 cast of B and A |
  |---|---|---|---|
  | T1 | 1, 1 | 8.6e-4 (2.6e-4) | 2.5e-3 (2.4e-3) |
  | T4 | 4, 4 | 7.3e-4 (3.9e-4) | 2.5e-3 (2.4e-3) |
  | T16 | 16, 16 | 1.2e-3 (4.7e-4) | 2.6e-3 (2.4e-3) |
  | T64 | 64, 64 | 7.4e-3 (5.8e-4) | 7.8e-3 (2.4e-3) |
  | T256 | 256, 256 | 3.0e-3 (8.5e-4) | 3.8e-3 (2.5e-3) |
  | C16 | 16, 16 (B = −U₁₆Σ₁₆) | 1.2e-3 (4.7e-4) | 2.6e-3 (2.4e-3) |

  "The adapters are the truncations": they agree with the exact truncation to about 1e-3, below the error of vLLM's
  bf16 cast of the factors (about 2.4e-3). The T64 maximum sits at a near-tie (σ₆₄ / σ₆₅ − 1 = 0.05 %, layer 43
  `up_proj`). The pre-registered 1e-6 check "is a self-check": it compares the factors with the executor's own fp32
  SVD. Z8's B·A is exactly 0.
- **F448** rests on the executor's tensor-by-tensor check (`meta/f448_check.json`); the served weights are gone with
  the pod.
- **Zero-delta check** (§1 with erratum 3). The stored pair (`stage0/zero_plain.json`, `zero_lora.json`) is not bitwise
  equal: over 1,451 tokens, 62 match and the largest log-prob difference is 0.25 nats (median 6e-5). The second pair,
  reported bitwise identical in `stage0/zero2_diff.txt`, and its run-to-run reference (max 0.23) rest on files that were
  not stored. Independent support that "the LoRA path itself shifts nothing": T1 matches `dpo` on the 176 other prompts
  (250 against 251 strict of 704) and on A (G −0.005).
- **Hand gate (§6), repeated blind** with its own readers, and **G and readings (§7)** with its own bootstrap (paired
  over prompts, 10,000 replicates, seed 20261127): below.
- **Compute (§8)** from the job logs and `~/podhours.log`: below.

### Part 1: how concentrated each change is (the review's §4, over every matrix)

S_k is the energy-weighted whole-model top-k share; "Gaussian" uses the review's own simulation (3 seeds per shape);
signal rank counts σ_i(D) above σ₁ of the run's floor A, uncensored; spike count counts σ_i(D) above σ₁ of the run's
floor B; "energy in spikes" is the energy-weighted share of ‖D‖² above floor B.

| model | change | S₁ | S₁₆ | S₆₄ | S₂₅₆ | S₁₆ ÷ Gaussian | S₁₆ ÷ floor A | median signal rank [IQR] | share ≥ 256 | median spike count [IQR] | energy in spikes | median stable rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32B | N50 | 0.0082 | 0.0329 | 0.0718 | 0.1690 | 4.4 | 4.4 | 1 [1, 3] | 0.00 | 124 [51, 191] | 0.13 | 77.9 |
| 32B | N100 | 0.0098 | 0.0406 | 0.0874 | 0.1946 | 5.4 | 5.4 | 12 [3, 25] | 0.00 | 166 [75, 262] | 0.18 | 64.1 |
| 32B | N500 | 0.0116 | 0.0477 | 0.1052 | 0.2296 | 6.4 | 6.4 | 392 [216, 788] | 0.70 | 222 [102, 359] | 0.26 | 51.8 |
| 32B | N750 | 0.0139 | 0.0572 | 0.1242 | 0.2628 | 7.7 | 7.6 | 1005 [515, 2064] | 0.93 | 241 [106, 416] | 0.32 | 45.2 |
| 32B | N1100 | 0.0185 | 0.0718 | 0.1458 | 0.2949 | 9.7 | 9.6 | 1552 [692, 3589] | 0.99 | 251 [109, 456] | 0.37 | 37.4 |
| 32B | N1600 | 0.1126 | 0.2650 | 0.3404 | 0.4686 | 35.4 | 35.3 | 1948 [824, 4770] | 1.00 | 212 [93, 332] | 0.49 | 9.3 |
| 32B | N2300 | 0.0895 | 0.2182 | 0.2940 | 0.4282 | 29.3 | 29.1 | 2305 [932, 5101] | 1.00 | 227 [101, 360] | 0.46 | 11.8 |
| 32B | I_early (50→100) | 0.0071 | 0.0285 | 0.0638 | 0.1569 | 3.8 | 3.8 | 2 [1, 5] | 0.00 | 114 [61, 171] | 0.11 | 95.8 |
| 32B | I_mid (1100→1150) | 0.1520 | 0.2214 | 0.2572 | 0.3343 | 29.5 | 29.5 | 580 [264, 1426] | 0.77 | 94 [42, 125] | 0.28 | 7.0 |
| 32B | I_late (2250→2300) | 0.0070 | 0.0255 | 0.0578 | 0.1476 | 3.5 | 3.4 | 38 [14, 78] | 0.00 | 144 [82, 185] | 0.11 | 104.0 |
| 32B | K (1100→2100) | 0.1291 | 0.2918 | 0.3524 | 0.4536 | 39.2 | 38.9 | 2102 [842, 4842] | 1.00 | 158 [65, 234] | 0.44 | 8.3 |
| 32B | SFT | 0.0047 | 0.0333 | 0.0917 | 0.2380 | 4.6 | 4.4 | 4925 [1024, 5120] | 1.00 | 344 [158, 519] | 0.36 | 149.1 |
| 32B | DPO | 0.1694 | 0.3501 | 0.4003 | 0.4934 | 37.4 | 46.7 | 13 [8, 19] | 0.00 | 74 [37, 192] | 0.43 | 6.4 |
| 7B | N25 | 0.0163 | 0.0401 | 0.0786 | 0.2047 | 3.4 | 3.3 | 0 [0, 1] | 0.00 | 11 [7, 20] | 0.04 | 120.9 |
| 7B | N100 | 0.0158 | 0.0467 | 0.0904 | 0.2181 | 4.0 | 3.8 | 7 [5, 12] | 0.00 | 44 [33, 58] | 0.08 | 89.0 |
| 7B | N500 | 0.0199 | 0.0661 | 0.1281 | 0.2689 | 5.7 | 5.4 | 1559 [1431, 2602] | 1.00 | 127 [113, 153] | 0.19 | 57.9 |
| 7B | N1000 | 0.0214 | 0.0757 | 0.1472 | 0.2980 | 6.5 | 6.2 | 2248 [2036, 3596] | 1.00 | 156 [134, 184] | 0.24 | 51.3 |
| 7B | N1375 | 0.0210 | 0.0779 | 0.1539 | 0.3103 | 6.7 | 6.3 | 2470 [2237, 3872] | 1.00 | 171 [145, 201] | 0.26 | 51.6 |
| 7B | I_early (25→50) | 0.0073 | 0.0254 | 0.0634 | 0.1907 | 2.2 | 2.1 | 0 [0, 1] | 0.00 | 9 [6, 18] | 0.03 | 205.2 |
| 7B | I_mid (500→525) | 0.0031 | 0.0169 | 0.0530 | 0.1783 | 1.5 | 1.4 | 43 [29, 76] | 0.01 | 18 [11, 24] | 0.02 | 390.5 |
| 7B | I_late (1350→1375) | 0.0022 | 0.0148 | 0.0503 | 0.1750 | 1.3 | 1.2 | 530 [454, 929] | 1.00 | 17 [12, 21] | 0.02 | 562.7 |
| 7B | K (500→1375) | 0.0133 | 0.0537 | 0.1147 | 0.2570 | 4.6 | 4.4 | 2416 [2178, 3787] | 1.00 | 141 [119, 164] | 0.19 | 81.8 |
| 7B | SFT | 0.0059 | 0.0358 | 0.1009 | 0.2804 | 3.2 | 2.9 | 4081 [3683, 4096] | 1.00 | 266 [240, 316] | 0.31 | 200.3 |
| 7B | DPO | 0.1451 | 0.3328 | 0.3932 | 0.5012 | 27.9 | 27.1 | 1034 [701, 1392] | 0.96 | 62 [46, 79] | 0.39 | 7.4 |
| 7B | ZIF (base→RL-Zero-IF) | 0.0845 | 0.2334 | 0.3559 | 0.5152 | 20.2 | 19.0 | 2898 [2612, 4095] | 1.00 | 142 [118, 197] | 0.46 | 11.5 |

- **RL's net change is far more concentrated than noise** (the §Verdict). At 2,300 the top 16 pieces hold S₁₆ 0.218,
  29× the Gaussian baseline (Gaussian 0.74 %, floor A 0.75 %; at k = 256, 10.9 % and 11.0 % against RL's 43 %), and a
  median 227 [101, 360] directions per matrix stand above same-norm rounded noise (floor B), carrying 46 % of the
  energy. Independent floor-B draws move individual spike counts by ≤ 14.
- **It grows over RL** (E3, Spearman ρ 0.96): S₁₆ 0.033 at step 50, 0.048 at 500, 0.218 at 2,300. It is not monotone at
  the end: N1600 has 0.265.
- **The mid-run interval** (1,100 → 1,150). The run doc's and the plain summary's "near-rank-1 jump" (the log:
  "rank-1-dominated") is reworded (item 1): it is "a jump concentrated in one direction per matrix (~15 % of the
  interval's ‖ΔW‖², about 20× the other intervals), strongest in late-layer v / o projections".
  - Across the whole model the interval's S₁ is 0.152, against 0.007 in the early and late intervals.
  - Per matrix, the top piece holds a median 14 % of the interval's change (max 61 %). 1.6 % of matrices are above
    50 %, and they carry 1.2 % of its energy.
  - The top piece's share is largest in the v / o projections of layers 56–63 (medians 0.46–0.53).
  - S₁ of the net change jumps after it: 0.0185 at 1,100, 0.113 at 1,600, 0.090 at 2,300. The review did not measure
    whether the interval's top directions align with the net change's. In energy, the interval's top directions are
    about a quarter of N1600's or N2300's (Σ_m σ₁² 8.2 against 31; erratum 7).
  - Size: "3× the late interval's size" (log) holds in squared norm (3.04×); it is 1.74× in ‖ΔW‖.
- **Floor A, sized to one interval** (erratum 1 replaces §4's "crossed early"). The median signal rank passes the 256
  censoring point from step 500: ≥ 256 in 70 % of matrices at 500, 93 % at 750, 99 % at 1,100 and 100 % at 2,300. But
  N2300 is above floor A in nearly every direction only in the MLP matrices (medians 5,120 / 5,041 / 5,116 of 5,120).
  Elsewhere it is not: q 2,230 / 5,120, o 1,926 / 5,120, k 711 / 1,024, v 825 / 1,024. At step 500 the type medians
  are 132–911.
- **Single intervals.** 32B's early and late 50-step intervals have S₁₆ 3.8× and 3.4× floor A's; the mid interval
  29.5×. 7B's 25-step intervals are 1.2–2.1×.
- **SFT and DPO** (reproduce). SFT's change is diffuse (S₁₆ 0.033, median stable rank 149) and DPO's the most
  concentrated (0.350). At 7B: SFT 0.036, DPO 0.333, RL N1375 0.078, RL-Zero-IF 0.233. The plain summary's inherited
  "31×" (SFT's move against RL's, from olmo-weights) is 30.6× in ‖ΔW‖ over these 448 matrices (the review).

### `lora_main` against RL's top-16 subspaces (32B)

The review's table (overlap = mean of left and right; random = mean of 4 seeded rank-16 bases; the inner products are
the run's, validated on the review's sample within 1.3e-4 (main) and 8.2e-3 (ctrl, whose inner products are near
zero); erratum 6):

| change | lora_main median overlap (left / right) | lora_ctrl | random | cos(ΔW_main, D) | cos(ΔW_main, D₁₆) | share of ⟨ΔW_main, D⟩ in D₁₆ |
|---|---|---|---|---|---|---|
| N50 | 0.0155 (0.0216 / 0.0081) | 0.0110 | 0.0030 | 0.0008 | 0.0036 | 0.84 |
| N100 | 0.0156 (0.0226 / 0.0080) | 0.0110 | 0.0030 | 0.0011 | 0.0044 | 0.82 |
| N500 | 0.0189 (0.0282 / 0.0084) | 0.0115 | 0.0030 | 0.0028 | 0.0105 | 0.82 |
| N750 | 0.0207 (0.0307 / 0.0085) | 0.0117 | 0.0030 | 0.0037 | 0.0128 | 0.83 |
| N1100 | 0.0213 (0.0313 / 0.0082) | 0.0118 | 0.0030 | 0.0045 | 0.0140 | 0.84 |
| N1600 | 0.0139 (0.0196 / 0.0068) | 0.0064 | 0.0030 | 0.0047 | 0.0068 | 0.74 |
| N2300 | 0.0156 (0.0237 / 0.0068) | 0.0070 | 0.0030 | 0.0052 | 0.0086 | 0.77 |

- At 2,300, `lora_main`'s median overlap is 0.0156: "5× more than chance, but only 2.2× more than lora_ctrl" (§Verdict).
  The plain summary's "five times more than chance" needs the control beside it (item 5): `lora_ctrl`, trained on 16 of
  `dpo`'s own answers, is already at 2.3× random.
- The cosine with RL's change is 0.00525, "this run's own measurement" (the review; the log notes that it reproduces
  olmo-weights, and *[librarian]* olmo-weights' summary gives 0.00525); with the top-16 truncation, 0.00865.
  77 % of ⟨ΔW_main, D⟩ lies in the top 16 pieces (`lora_ctrl`: 74 %, `numbers.md`).
- P3 (overlap ≥ 0.052, "far more than its 0.0052 cosine suggests") and E5 (it fails main ≥ 3× ctrl) miss.

### Part 2: how much of RL's gain each truncation recovers (32B)

**Re-grade and budget.** Strict passes per job, the same under the review's checker driver and the run's grades:

| job | reach draws | reach strict | other draws | other strict | cut at 32,768 tokens |
|---|---|---|---|---|---|
| dpo | 768 | 0 | 704 | 251 | 3 (0.20 %) |
| T1 | 768 | 2 | 704 | 250 | 0 |
| T4 | 768 | 5 | 704 | 268 | 10 (0.68 %) |
| T16 | 768 | 65 | 704 | 359 | 14 (0.95 %) |
| T64 | 768 | 264 | 704 | 452 | 2 (0.14 %) |
| T256 | 768 | 422 | 704 | 493 | 2 (0.14 %) |
| F448 | 768 | 511 | 704 | 517 | 0 |
| C16 | 768 | 216 | 704 | 427 | 1 (0.07 %) |
| B_s2300 (stored) | 1,536 | 1,066 | – | – | 0 (`numbers.md`) |

`dpo` had 0 strict passes in 768 reach draws (16 per prompt), so its reach rate is below 0.39 % at 95 %. T1's 2 strict
passes are both gamed, so T1 has 0 genuine in 768 and the same bound.

**G** (R: the reach prompts' estimated genuine rate; A: the 218 prompts' strict pass@1, with `dpo` pooled with
`F_dpo_ifb` on the 176). The review's §7 table, with the executor's R values beside it (`results.csv`):

| arm | R rate, review's labels | G on R, review's labels [95 %] | R rate, executor's labels | G on R, executor's labels [95 %] | R strict rate | G on R strict | A rate | G on A [95 %] (review) | G on A, fresh `dpo` only |
|---|---|---|---|---|---|---|---|---|---|
| dpo | 0.0000 | – | 0.0000 | – | 0.0000 | – | 0.2894 (pooled), 0.2878 (fresh) | – | – |
| T1 | 0.0000 | 0.000 [0, 0] | 0.0000 | 0.000 [0.000, 0.000] | 0.0026 | 0.004 | 0.2873 | −0.005 [−0.045, 0.034] | −0.001 |
| T4 | 0.0052 | 0.012 [0.003, 0.023] | 0.0052 | 0.011 [0.003, 0.021] | 0.0065 | 0.009 | 0.3085 | 0.047 [0.001, 0.093] | 0.051 |
| T16 | 0.0742 | 0.167 [0.109, 0.234] | 0.0755 | 0.158 [0.101, 0.221] | 0.0846 | 0.122 | 0.4292 | 0.348 [0.277, 0.419] | 0.351 |
| T64 | 0.2835 | 0.638 [0.524, 0.758] | 0.2925 | 0.613 [0.503, 0.730] | 0.3438 | 0.495 | 0.5834 | 0.733 [0.661, 0.806] | 0.734 |
| T256 | 0.4020 | 0.904 [0.799, 1.021] | 0.4271 | 0.895 [0.796, 1.008] | 0.5495 | 0.792 | 0.6717 | 0.953 [0.898, 1.011] | 0.953 |
| F448 | 0.4145 | 0.933 [0.857, 1.026] | 0.4469 | 0.936 [0.864, 1.019] | 0.6654 | 0.959 | 0.7173 | 1.067 [1.012, 1.128] | 1.067 |
| C16 | 0.2401 | 0.540 [0.411, 0.678] | 0.2397 | 0.502 [0.376, 0.634] | 0.2812 | 0.405 | 0.5378 | 0.619 [0.543, 0.699] | 0.621 |
| s2300 | 0.4445 | 1 | 0.4774 | 1 | 0.6940 | 1 | 0.6905 | 1 | |

- The executor's G on A has the same point values; its bootstrap intervals differ from the review's by ≤ 0.004
  *[librarian]*. Its T4 interval on A starts at −0.000.
- **The readings** (the review's §7, scored on point estimates, separately on R and A):

  | reading | R | A | final |
  |---|---|---|---|
  | 1. "Low-rank carries it" (G(T16) ≥ 0.5 and G(C16) ≤ 0.2) | no (0.167; 0.540), firm | no (0.348; 0.619), firm | does not hold |
  | 2. "High-rank" (G(T256) < 0.5 and G(F448) ≥ 0.75) | no (T256 0.904, firm; F448 0.933 holds) | no (0.953; 1.067) | does not hold |
  | 3. "Outside the 448 matrices" (G(F448) < 0.75) | no (0.933), firm | no (1.067), firm | does not hold |
  | 4. Otherwise: k₅₀ | **64** (T16 firmly below 0.5, T64 lower bound 0.524) | **64** (T64 lower bound 0.661) | k₅₀ = 64 |

- **Label sensitivity of G on R** (the review's walk throughout): "The readings are the same under every variant."

  | label variant | T16 | T64 | T256 | F448 | C16 | k₅₀ |
  |---|---|---|---|---|---|---|
  | main (the review's policy) | 0.167 | 0.638 | 0.904 | 0.933 | 0.540 | 64 |
  | first read only | 0.174 | 0.645 | 0.918 | 0.932 | 0.555 | 64 |
  | no mechanical screen | 0.164 | 0.631 | 0.898 | 0.938 | 0.536 | 64 |
  | rule-6 strict (1 ALL-CAPS item gamed) | 0.168 | 0.641 | 0.908 | 0.937 | 0.542 | 64 |
  | low-confidence proposals accepted | 0.166 | 0.634 | 0.899 | 0.931 | 0.537 | 64 |
  | the executor's labels (the review's bootstrap) | 0.158 | 0.613 [0.501, 0.732] | 0.895 | 0.936 | 0.502 [0.375, 0.636] | 64 |

  The review: "With the executor's labels the R rates are higher (s2300 0.477 vs my 0.445) but G hardly moves."
  *[librarian]* Not in every arm: C16's is 0.2397 against 0.2401, and `dpo`'s, T1's and T4's are equal.
- **k₅₀ against Part 1.** k₅₀ = 64 holds on the {1, 4, 16, 64, 256} grid; it lies in (16, 64]. That is below N2300's
  median spike count, 227 [101, 360], and its median signal rank, 2,305 (≥ 256 in every matrix). T64 keeps S₆₄ = 29 %
  of RL's ‖ΔW‖², and T256 keeps 43 %.
- **C16 is not "the floor alone"** (item 2). C16 is RL's change minus its top 16 pieces. "By the run's own spike counts
  it keeps ~210 above-floor directions per matrix and 24 % of the energy above floor B, so the 'low-rank carries it'
  test cannot say whether the floor itself carries anything." The label is the design's, "not the executor's
  arithmetic": the brief asks "does the floor alone, C16, recover anything?" and the pre-registration calls C16 "the
  floor only". The plain summary words it correctly ("the change minus its top 16 pieces").
- **The parts are not additive.** C16 holds 78 % of the energy and recovers G 0.54 on R; T16 holds 22 % and recovers
  0.17. Together, 0.71 is less than F448's 0.93.
- **F448 beats `s2300` on A**: G 1.067, with the interval above 1, but not on R (0.933). Hence the plain summary's
  "the other weights do not matter" becomes "the other weights' RL changes are not needed for the gain" (item 5). The
  endpoint `s2300` on A comes from stored rows at other seeds and chunk size; the review's next measurement 3 (a fresh
  `s2300` arm at engine seed 62026) would test this.
- **A is strict only.** It is not hand-gated, and the strict and genuine G differ on reach: T256 0.79 against 0.90.

### Hand gate (reach prompts)

- **The executor's gate:** 1,400 items and 1,802 reads; 367 items read twice (the first two reads agree on 332, 90 %)
  and 35 three times; 35 draws screened by rule 1 and 39 inherited. 119 confirmations: 33 checker disagreements
  adjudicated under rules 1–14 (7 reader labels overruled), and 86 high- or medium-confidence proposals of four blind
  consistency audits accepted, with 10 low-confidence ones rejected. On 231 `B_s2300` items that carry olmo-ifcurve
  final labels, it agrees on 206 (89 %; 12 gamed here and genuine there, 13 the reverse). The review recounted 1,400 /
  1,802 / 367 / 332 / 35 / 119 from the label files.
- **The review's blind gate (§6):**
  - scope: the same pre-registered scope, 1,474 draws in 1,434 groups of identical answers per prompt;
  - readers: the pre-registered reader prompt, item files only. Its label policy (`hand/POLICY.md`) was written after
    the first reads and before any second read or adjudication;
  - the rule-1 screen covered the same 35 draws as the executor's; readers had called 18 of them genuine, "so the
    screen is not redundant";
  - 405 second reads (all T1 / T4 / T16 groups, a seeded quarter of the rest, every checker disagreement) and 37 third
    reads. Sub-agents with the rule files adjudicated 214 groups; the review accepted the 205 high- or
    medium-confidence proposals and rejected 9;
  - 1,876 reads in all; first-to-second agreement 368 / 405 = 0.909.
- **Per arm:**

  | job | strict on reach | labelled | executor: genuine / gamed (gamed share) | review: genuine / gamed (gamed share) | prompts with ≥ 1 genuine (review) |
  |---|---|---|---|---|---|
  | dpo | 0 | 0 | – | – | 0 |
  | T1 | 2 | 2 | 0 / 2 (1.00) | 0 / 2 (1.00) | 0 |
  | T4 | 5 | 5 | 4 / 1 (0.20) | 4 / 1 (0.20) | 4 |
  | T16 | 65 | 65 | 58 / 7 (0.11) | 57 / 8 (0.12) | 24 |
  | T64 | 264 | 218 | 185 / 33 (0.15) | 179 / 39 (0.18) | 40 |
  | T256 | 422 | 298 | 229 / 69 (0.23) | 212 / 86 (0.29) | 44 |
  | F448 | 511 | 335 | 232 / 103 (0.31) | 214 / 121 (0.36) | 44 |
  | C16 | 216 | 183 | 156 / 27 (0.15) | 156 / 27 (0.15) | 32 |
  | s2300 (`B_s2300`) | 1,066 | 368 | 251 / 117 (0.32) | 233 / 135 (0.37) | 42 |

- **Agreement between the two teams,** draw by draw over the same 1,474 draws: 1,370 (0.929). By arm: T16 64 / 65, T64
  206 / 218, T256 277 / 298, F448 307 / 335, C16 169 / 183, `B_s2300` 340 / 368. "The disagreements lean one way":
  in 82 draws the review says gamed and the executor genuine, in 22 the reverse. The most affected prompts are
  ifbench_264 (10), ifbench_1 (9), ifbench_267 (8) and ifbench_209 (8); ifbench_1 leans the other way. The review's
  labels agree with olmo-ifcurve's on 200 of 233 `B_s2300` draws (0.858; 22 gamed by the review and genuine there,
  11 the reverse): "consistent with a stricter second team".
- **Gamed shares** depend on the reader standard (item 6): "0.11–0.32 (second team 0.12–0.37)" in the arms with at
  least 65 strict reach successes. *[librarian]* In both teams T16, T64, T256 and C16 have lower gamed shares than
  `s2300`, and F448 is just below it (executor 0.307 against 0.318; review 0.361 against 0.367).

### Expectations (pre-registered 20:03:04, before the first pod), scored by the review

P1–P3 are the proposal's guesses; E1–E6 are the executor's. 32B unless stated.

| id | expectation | value | the review's score |
|---|---|---|---|
| P1 | (a) N2300's median spike count ≥ 1 **and** (b) each 50-step interval's S₁₆ ≤ 1.5 × floor A's | (a) 227; (b) 3.79, 29.5 and 3.40 × floor A | **miss** ((a) hits, (b) misses) |
| P2 | 0.3 ≤ G(T16) ≤ 0.7, on R | 0.167 (executor's labels 0.158); on A 0.348 | **miss**. On A it is inside, but 0.348 is within the pre-registration's ≈ 0.10 resolution of 0.3: "unresolved" (item 6) |
| P3 | `lora_main` median overlap at 2,300 ≥ 0.052 | 0.0156 | **miss** |
| E1 | S₁₆(N2300) ≤ 3 × Gaussian **and** median signal rank ≥ 256 (censored) | 29.3 × Gaussian; ≥ 256 in 100 % of matrices | **miss** (the first part) |
| E2 | S₁₆(SFT) > S₁₆(N2300) | 0.033 against 0.218 | **miss** |
| E3 | S₁₆ of N_t rises with t (Spearman ρ > 0, 7 steps) | ρ = 0.96 | hit |
| E4 | G(T16) < 0.3, G(T256) < 0.5, G(F448) ≥ 0.75 and G(C16) ≥ 0.5, all on R, i.e. reading 2 | T16 0.167 hit; T256 0.904 miss; F448 0.933 hit; C16 0.540 [0.411, 0.678] (executor's labels 0.502) | **miss**: reading 2 does not hold. The C16 component is "unresolved" rather than "hit (tentative)" (item 6) |
| E5 | main ≥ 3 × ctrl **and** ≥ 3 × random **and** < 0.052 | main 0.0156 = 2.2 × ctrl (0.0070), 5.2 × random (0.0030) | **miss** (fails ≥ 3 × ctrl) |
| E6 | S₁₆(7B N1375) < S₁₆(32B N2300) | 0.078 against 0.218 | hit |

Two hits (E3, E6) and seven misses. The executor's own guess, reading 2 ("high-rank"), missed on T256, and the
proposal's guesses all missed. The review (claim 31) on the write-up: "every miss is listed as a miss".

### Compute and cost (the review's §8)

| arm | GPU | GPU-seconds (job wall, every attempt) | draws | generated tokens | tokens × `dpo` | *[librarian]* GPU-seconds × `dpo` | cut |
|---|---|---|---|---|---|---|---|
| dpo | H200 | 3,834 | 1,472 | 6,720,092 | 1.00 | 1.00 | 3 (0.20 %) |
| T1 | H200 | 4,513 | 1,472 | 6,856,209 | 1.02 | 1.18 | 0 |
| T4 | H200 | 4,590 | 1,472 | 6,920,863 | 1.03 | 1.20 | 10 (0.68 %) |
| T16 | H200 | 5,931 | 1,472 | 8,419,961 | **1.25** (flag) | **1.55** | 14 (0.95 %) |
| T64 | H200 | 6,714 | 1,472 | 9,160,345 | **1.36** (flag) | **1.75** | 2 (0.14 %) |
| T256 | H200 | 9,629 (stage 0 1,658 + 7,971) | 1,472 | 9,290,432 | **1.38** (flag) | **2.51** | 2 (0.14 %) |
| F448 | H200 | 5,439 | 1,472 | 9,390,637 | **1.40** (flag) | **1.42** | 0 |
| C16 | H200 | 6,706 | 1,472 | 9,577,410 | **1.43** (flag) | **1.75** | 1 (0.07 %) |

- **Matching:** all arms are matched on prompts and draws (attempts). Five arms used more than 1.25 × the `dpo` arm's
  generated tokens; the pre-registration says to flag this, and the run doc does ("Flagged (> 1.25× `dpo`'s tokens):
  T16–C16, 1.25–1.43×"; it adds "≤ 2.51× H200-seconds"). *[librarian]*
  By GPU-seconds the same five arms are above 1.25× (1.42–2.51×). Part of that is the serving path: generated tokens
  per job-second were 1,753 for `dpo` and 1,727 for F448 (plain weights), 1,420–1,519 for T1–T16 and C16, 1,364 for
  T64 and 965 for T256. Nothing was trained (0 training steps and tokens) and
  nothing went to Lean.
- *[librarian]* The eight arms: 11,776 draws, 66.3 M generated tokens and 47,356 H200-seconds (13.15 h) of sampling
  jobs, on `lr-a` and `lr-b` (13.68 H200 pod-hours). With the three faulty-GPU pods (0.49 h) the run used 14.17 H200
  pod-hours by `~/podhours.log`.
- `numbers.md`'s GPU-seconds come from the metas' `t_load + t_gen` (0.17–0.42 % below the job wall, e.g. `dpo` 3,820
  against 3,834); T256 ÷ `dpo` is 2.51× either way. `numbers.md`'s compute table has no token ratios; the flags appear
  only in the run doc.
- **Part 1:** three jobs on one H100 80GB HBM3: R32 2,438 s, S32 527 s, R7 962 s. *[librarian]* The adapters' SVD took
  704 H200-seconds summed over matrices (`numbers.md` and its registry row, "sum of per-matrix SVD seconds"; 727 s of
  wall time in `stage0/stage0.log`).
- **Registry:** rows in `hf://buckets/dan-pandori/nd-rl/registry/olmo-lowrank/` (*[librarian]* 67 rows; `gpu_seconds` 13
  rows: 8 arms, T256's stage-0 part, the three spectra jobs and `adapters_svd`; `gen_tokens`, `attempts`, `train_steps`,
  `train_tokens`, `lean_checks` and `verifier_checks` 9 rows each, with T256 split in two).
- **Pods:** 15.33 pod-hours, $55.98 (`podbudget`, the review's ledger); the reviewer's pod $3.13 more, $59.11. See
  Configuration for the billed figures.

### Policy checks (the review)

- Same chat template, tokenizer, decoding and token budget for every compared arm and endpoint; template boundary
  checked by hand.
- "The hand gate runs both ways": every arm and `s2300` get the same reach cells, the same 8-per-cell scope and the
  same rules; no arm gets extra tries.
- Strict and genuine are reported side by side, with the label sources. A rests on strict passes only, by design.
- Token budget and cut share are reported per model (the largest, T16, 0.95 %).
- "Never" states its tries in the run doc (0 / 768); the plain summary's "never solved" did not (item 5).
- **Not reported:** a scan of the evaluation prompts against Olmo 3's RL training data. The prompt sets are
  olmo-ifcurve's and every arm uses the same prompts, so the review calls this minor, "but the policy asks for it". The
  review did not repeat a scan either.
- **An unlabelled deviation** (item 7): the pre-registration says "Per-prompt consistency audits by sub-agents after
  each round"; the executor ran them once (u1–u4), after the last arm was read, with the policy fixed before them
  (`log.md` 00:27). Neither the write-up nor the log calls this a deviation.
- **Reproducibility:** everything recounted comes from git or the bucket, except the zero-delta second pair and the
  cost and pod-time figures, which come from the host's ledgers.
- Every number in the run doc, `numbers.md` and the plain summary names its model; the one inherited number (31×)
  names olmo-weights. The secret scan of 341 tracked files and 1,553 bucket objects found 0 hits. *[librarian]* The
  kit's scan skipped the whole `adapters/` prefix (15 of the run folder's 1,567 objects) and added the registry file.

### Blindness of the review's phase 1

- **The review's statement.**
  - It saw the executor's commit subjects at session start (05:38 UTC) and in its own `git log`. They include "DONE —
    spectra (RL's change is spikes on a floor), truncation arms (k50 = 64; no pre-registered reading holds)", "dpo arm
    sampled (reach strict 0/768)" and the hand-gate round sizes.
  - "I read no write-up, `README.md`, `results.csv`, `summary.json`, `hand_cells.csv`, `hand_arm_summary.csv`,
    `tables/` file or registry row before this section was committed."
  - It read the executor's per-item hand labels "only after my own labels were final (§6), to compute agreement".
- *[librarian]* **The transcript check.** The librarian parsed the reviewer's transcript, separating the main session
  from its sub-agents by `parent_tool_use_id`. Before 07:47:27 there were 186 main-session tool calls and 2,588
  sub-agent calls. They fit the statement:
  - Phase 1 read the run's files from `~/review/olmo-lowrank/`, a hard-linked copy of the worktree (the kit README).
    That copy lacks the run doc, `numbers.md`, `STATUS.md` and the plain summary (the claim check was written later),
    but still held `log.md` (with the run's results section), `olmo_lowrank/README.md`, `results.csv`, `summary.json`,
    `hand_cells.csv`, `hand_arm_summary.csv`, `tables/`, the figure and the registry rows. The session itself started in
    the fork worktree `~/work/olmo-lowrank`, which has every write-up; it ran its `git log`, used the run's pod scripts
    for its own pod and wrote its kit there. No
    tool call before the commit opens a write-up in either place, nor any of the files above. The one `summary.json`
    opened is `spectra/R32/summary.json`, the spectra job's timing record.
  - The sub-agents (blinded readers and adjudicators) touched only their items, prompts, tasks, label files and the
    hand-gate rule files.
  - The executor wrote its memory note at 05:57:38 and the note's index line in `MEMORY.md` ("k50 = 64, none of 3
    readings") at 05:57:41, after the reviewer's session started (05:38:03) and loaded its memory index. The note was
    never opened. The reviewer read the reviewer role's traps note
    and earlier runs' notes at 05:38:56–05:39:07.
  - The `git log` with the commit subjects ran at 05:38:25, before the executor's claim-check commit (05:57:14, whose
    subject has "T256 0.89"). No tool result before the §Recount commit contains that subject.
  - Executor values reached phase 1 through the commit subjects, the run's inputs and job records, and the executor's
    labels. The inputs and records: the stored per-matrix spectra, which §3 validates; metas, logs and stage-0 files;
    `adapters_svd/norms.csv`; the cost from `~/podhours.log`. The labels were read at 07:42:36, after the review's own
    readings (07:35:44). The walk with them printed the executor's G on R (0.158 … 0.502) at 07:42:56, 4.5 minutes
    before the commit.
- *[librarian]* The reviewer started while the executor was still finishing: `OLMO-LOWRANK DONE` was written at
  05:37:11, and the claim-check fixes followed at 05:57:14. The review's phase 2 (from 07:47) read the fixed files, as
  its claims table shows (T256 0.89).

### The review's rewordings (§Verdict "Must be reworded"), applied in this summary

1. "Near-rank-1 jump" at 1,100–1,150 (run doc, plain summary; "rank-1-dominated" in the log). Instead: "a jump
   concentrated in one direction per matrix (~15 % of the interval's ‖ΔW‖², about 20× the other intervals), strongest
   in late-layer v / o projections".
2. "The floor alone (C16)" (run doc). C16 is RL's change minus its top 16 pieces; see above.
3. "Exact SVD" / "exactly in fp32" for the adapters, and "exact spectra" for `adapters_svd/`. "The pre-registration
   specified an fp32 SVD, so the method is as planned; only the word 'exact' overstates." Say: fp32 SVD, ≈ 0.1 % from
   the exact truncation, ≈ 0.24 % after vLLM's bf16 cast. The stored `adapters_svd` spectra deviate from float64 by up
   to 4.2e-4 σ₁ (the Gram-route `full_*.npz`: 5.6e-8 σ₁).
4. Zero-delta "bitwise identical" and its run-to-run reference rest on files that were not stored. "Upload
   `zero2_*.json`, or cite what the stored pair shows: ≤ 0.25 nats, median 6e-5." *[librarian]* The upload is no longer
   possible (see Bucket); what stands is the stored pair and the T1 arm.
5. Plain summary: "never solved" (give the tries and the bound); "the other weights do not matter" (say "are not needed
   for the gain"); the LoRA's "five times more than chance" (add the control at 2.3×). Applied, marked, in the copy.
6. Gamed shares depend on the reader standard (0.12–0.37 against 0.11–0.32). By the pre-registration's own ≈ 0.10
   resolution, E4's C16 component (0.50 / 0.54 against 0.5) and P2 on A (0.348 against 0.3) are unresolved rather
   than hits or misses.
7. The unlabelled deviation: the consistency audits ran once at the end, not after each round.

Minor points from the claims table: the log's 22:16 accuracy line ("σ₁ ≤ 7e-6 and top-k shares ≤ 1.3e-5 (k ≤ 64)")
understates the run's own maxima (5.5e-5, 1.9e-4); `numbers.md`'s T4 G on A 0.048 is double rounding of 0.0475, and its
T16 R rate 0.075 against 0.076 in the hand table is double rounding of 0.0755; "more than half the change is spread over
thousands of directions" holds energy-weighted (54 % below the floor-B level; 48 % for the median matrix); the top 1 or
4 pieces "do nothing" is "fine as a plain-language summary" ("nearly nothing" is more exact).

**The fork's executor files are not corrected.** On branch `dan_olmo-lowrank`, `run_olmo_lowrank.md`, `numbers.md`,
`log.md`, `olmo_lowrank/README.md` and `olmo_lowrank/plain_summary.md` still carry the wording the review asks to
change. Only the plain-summary copy here is marked. *[librarian]* The design label for C16 also stands in nd-rl's
[Proposal 49](../../docs/proposals/2026-11-26-olmo-lowrank.md) ("the floor only"; "whether the floor alone does
anything") and the [brief](../../docs/proposals/followups/BRIEF_olmo-lowrank.md) ("does the floor alone, C16, recover
anything?"). These record what was proposed and approved, and are left unchanged.

## Limitations and next step

- **Scope** (the proposal): one RL run (Olmo 3.1 32B Think), one benchmark family for the behaviour test, and no
  truncation test at 7B. The same k is used for every matrix, and "matrices may differ in how many pieces they need".
  "A change that works when truncated is one sufficient change, not necessarily the one RL 'meant'."
- **What C16 tests.** C16 removes only the top 16 pieces and keeps ~210 above-floor directions per matrix, so the test
  "cannot say whether the floor itself carries anything" (the review): the second question of the brief's Part 2.
- **Labels and metrics.** R depends on the reader standard: the two teams' gamed shares differ (0.11–0.32 against
  0.12–0.37), though G moves by ≤ 0.04. A is strict only. Two near-threshold comparisons are unresolved at the
  pre-registered resolution.
- **Endpoints.** `s2300` comes from stored rows at other engine seeds and chunk 4. F448's excess over `s2300` on A
  (G 1.07, interval above 1) has not been checked at matched sampling.
- **Unstored evidence.** The zero-delta second pair (`zero2_*.json`) was never stored, and its run-to-run and
  batched-against-single references (max 0.23 and 0.15) rest on it. From stored files, the stored pair differs by
  ≤ 0.25 nats (median 6e-5; 62 of 1,451 tokens identical); the independent support is the T1 arm.
- **Not reported:** an RL-data overlap check for the evaluation prompts. *[librarian]* olmo-ifcurve's figures for these
  prompt sets are inherited from olmo-ifomega's scan against Dolci-Think-RL-32B: 0 of 218 and 0 of 48 at content
  containment ≥ 0.5.
- **Next measurements** (the review):
  1. **Does the floor carry anything?** Spike-matched arms: a variable-rank LoRA that keeps each matrix's own
     above-floor directions (median 227), and its complement, F448 minus those directions, "which is the floor
     proper". "If the complement recovers little (G ≲ 0.2) while the spike arm recovers most, the gain sits in the
     spikes. If the complement still recovers a lot, the gain needs the floor too."
  2. **Locate k₅₀:** T32 would place it inside (16, 64].
  3. **Check F448 > s2300 on A:** a fresh `s2300` arm at engine seed 62026 would show whether the firm excess (G 1.07)
     survives matched sampling.

  *[librarian]* No run is queued on this one.
