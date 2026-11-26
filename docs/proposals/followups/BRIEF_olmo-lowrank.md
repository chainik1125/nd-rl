# Run brief: olmo-lowrank — is RL's change to Olmo 3's weights low-rank, and does its low-rank part carry RL's IFBench gain?

Run id `olmo-lowrank`. Repository `~/work/olmo-lowrank` (branch `dan_olmo-lowrank` from the fork's
`origin/dan_olmo-weights`, made by `newrun`).

That branch carries:
- `olmo-weights`' checkpoint manifest, lineage findings and statistics code;
- `olmo-ifcurve`'s prompts, decoding, stored rows and hand-gate rules with dated addenda;
- `olmo-ifladder`'s LoRA serving code. Its adapters `lora_main` and `lora_ctrl` are in
  `hf://buckets/dan-pandori/nd-rl/olmo-ifladder/`.

Role: executor. `AGENT_POLICY.md` governs. Proposal 49 (nd-rl `docs/proposals/2026-11-26-olmo-lowrank.md`; read it
first). Authorised by Dan 2026-11-26: "Approved for up to 80$."

- **Budget:** **$80 / 60 pod-hours.** Dollars are binding.
- **Stage gate:** stage 0 (≤ $5) measures:
  - download throughput;
  - SVD time for one 32B MLP and one attention matrix, full and randomized top-256;
  - vLLM's maximum LoRA rank and draw throughput for one T_k arm.

  Re-cost, and pause and ask in `QUESTIONS.md` if the projection exceeds $72.
- **If over budget, cut in this order:**
  1. the 7B spectra;
  2. the 218-prompt aggregate from 4 draws to 2;
  3. arms T₁ and T₄;
  4. single-interval spectra from three to one.

  Keep T₁₆, T₂₅₆, F448 and C16.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 48 h.

## The question

**Part 1, spectra.** For the 448 attention and MLP matrices, how concentrated are these changes, against a random
baseline and a bf16-rounding baseline?
- RL's net change at several steps;
- single RL intervals;
- a mid-run chord;
- SFT's and DPO's total changes;
- at 7B, RL from the base as well.

How much of RL's top-16 subspace lies inside `lora_main`'s?

**Part 2, truncation.** Does DPO plus only the top-k pieces of RL's change recover RL's IFBench gain? And does the
floor alone, C16, recover anything?

## Design (pre-register your version before any spectrum or draw beyond stage 0)

1. **Checkpoints:** the proposal's table, on the lineage `olmo-weights` established. Avoid 32B RL steps 150–200,
   which are on an abandoned branch. Record each revision and hash from `olmo-weights`' manifest.
2. **Spectra:**
   - **Method:** fp32 from the bf16 values. Use randomized SVD for the top 256 of every matrix, with the exact
     Frobenius norm. Run a full SVD on a seeded subset (for example every 8th layer, all 7 types) to validate the
     randomized values and to draw full spectra.
   - **Measures:** top-k shares (k = 1, 4, 16, 64, 256), stable rank, and signal rank above the rounding baseline's
     largest singular value.
   - **Baselines:** random Gaussian at matched norm, and the rounding construction (pre-register it).
   - **LoRA overlap:** RL's top-16 left and right subspaces against `lora_main`, `lora_ctrl` and random rank-16. Check
     the adapters' sha256 against `olmo-ifladder`'s records.
3. **Truncation arms:**
   - **T_k** (k = 1, 4, 16, 64, 256) as rank-k LoRAs on the DPO base.
   - **F448:** DPO with the 448 matrices replaced by RL step 2,300's.
   - **C16:** F448 plus a negative rank-16 LoRA.

   Do not merge truncated changes into bf16 weights: RL's change sits at the rounding scale. Verify on a few matrices
   that each adapter reproduces U_k Σ_k V_kᵀ to fp32 precision. If vLLM's maximum LoRA rank is below 256, use the
   highest supported rank and record it.
4. **Draws:**
   - the 48 reach prompts × 16 per arm;
   - the 218 IFBench prompts × 4 per arm.

   Use `olmo-ifcurve`'s decoding, template and response cap. The end points `dpo` and `s2300` come from stored rows,
   only where the protocol matches; otherwise re-sample a matched control.
5. **Grading:** the strict checker, plus the blinded hand gate (`olmo-ifcurve`'s rules and addenda, readers confirmed by
   you) on every deciding reach success. Report the gamed share per arm.

## Pre-registered readings (from the proposal; keep the thresholds)

G = (arm − `dpo`) ÷ (`s2300` − `dpo`), on reach-prompt genuine rate and on 218-prompt strict pass@1, with
prompt-bootstrap intervals.

- **"Low-rank carries it":** G(T₁₆) ≥ 0.5 and G(C16) ≤ 0.2.
- **"High-rank":** G(T₂₅₆) < 0.5 while G(F448) ≥ 0.75.
- **"Outside the 448 matrices":** G(F448) < 0.75.
- **Otherwise:** report k₅₀, the smallest k with G ≥ 0.5, against Part 1's signal rank.

Record the proposal's guesses as expectations and mark each hit or miss.

## Deliverables

- `run_olmo_lowrank.md` (≤ 400 words + one figure);
  - Panel 1: top-k share against k for RL net, a single RL interval, SFT total and the two baselines.
  - Panel 2: G against k for the T_k arms, with F448 and C16 marked.
- `numbers.md` § olmo-lowrank; `log.md`; compute rows.
- A plain-language summary of ≤ 500 words, written for a reader who learned what a spectrum is last week.
- `olmo_lowrank/`: code, the manifest, the per-matrix spectrum tables, the overlap tables, `results.csv`, the
  hand-check table, and every sample's text and every adapter in the bucket.
- A `STATUS.md` line `OLMO-LOWRANK DONE <UTC>`; then upload to `hf://buckets/dan-pandori/nd-rl/olmo-lowrank/`, delete
  pods, and `touch ~/runs/olmo-lowrank/executor.done`.

**For the reviewer:**
- Recompute a seeded sample of matrices' spectra and overlaps from raw weights.
- Check that each adapter equals its truncation.
- Re-grade the results, and repeat the hand gate blind on the deciding successes.
- Recompute G and the readings.

**Rules:**
- Do not contact anyone.
- Weights stay on pods; adapters may go to the bucket.
- At most 4 concurrent processes on the VPS.
- Keep the RunPod balance above $35.
- Wait for jobs with one long poll, not many short check-ins.
