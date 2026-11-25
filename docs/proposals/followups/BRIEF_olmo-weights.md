# Run brief: olmo-weights — how much do Olmo 3's weights move at each stage of training, from pretraining to RL?

Run id `olmo-weights`. Repository `~/work/olmo-weights` (branch `dan_olmo-weights` from the fork's
`origin/dan_olmo-ifcurve`, made by `newrun`).

That branch carries:
- `olmo-ifcurve`'s reviewed results, which you need for the overlay;
- `olmo-ifladder`'s code and manifests. Its LoRA adapters are in `hf://buckets/dan-pandori/nd-rl/olmo-ifladder/`.

Role: executor. `AGENT_POLICY.md` governs. Proposal 47 (nd-rl `docs/proposals/2026-11-24-olmo-weight-trajectory.md`;
read it first: it holds the chart specification and the add-ons Dan chose). Authorised by Dan 2026-11-24: "If this
experiment is sub 50$ we don't need to think much harder about optimization."

- **Budget:** **$50 / 120 pod-hours.** Dollars are binding; the hours are advisory, for several cheap CPU pods.
- **Stage gate:** stage 0 (≤ $3) measures, on one CPU pod:
  - download throughput from Hugging Face;
  - statistics time, for one 7B adjacent pair and one 32B adjacent pair.

  Re-cost, and pause and ask in `QUESTIONS.md` if the projection exceeds $45.
- **If over budget, cut in this order:**
  1. 7B stage-1 locations, from 30 to 15;
  2. 32B stage-1 locations, from 25 to 12;
  3. 32B stage-2 ingredient 2;
  4. 7B RL-Zero-Code.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 48 h.

## The question

Plot how much the weights change between consecutive released checkpoints across the whole Olmo 3 pipeline:
pretraining stage 1, mid-training stage 2, long-context stage 3, SFT, DPO and RL. Do it for 32B (the pipeline our
runs use) and for 7B. Dan wants a deliberately simple picture that makes the switches between kinds of training
obvious. Add-ons, as specified in the proposal:
- distance from each stage's start, with straightness and consecutive cosines;
- per-module and per-layer breakdowns;
- the fraction of weights unchanged or within 1 bf16 ulp;
- the `olmo-ifcurve` overlay;
- the comparison with `olmo-ifladder`'s LoRA;
- 7B's RL-from-base runs against its RL from DPO.

Two things are **not** in this run:
- **The low-rank analysis.** Dan paused it.
- **Behavioural KL.** That is Proposal 48.

## Design (pre-register your version, with the exact checkpoint list, before any download beyond stage 0)

1. **Checkpoint manifest:** the proposal's table, with exact revisions and commit hashes, for:
   - 32B: `allenai/Olmo-3-1125-32B`, `Olmo-3-32B-Think-SFT`, `Olmo-3-32B-Think-DPO` and `Olmo-3.1-32B-Think`;
   - 7B: `allenai/Olmo-3-1025-7B`, `Olmo-3-7B-Think-SFT`, `Olmo-3-7B-Think-DPO`, `Olmo-3-7B-Think` and
     `Olmo-3-7B-RL-Zero-{IF,Math,Code}`.

   Fix the stage-1 sample locations before any download: log-spaced early, then even; each location is an adjacent pair
   s and s + 1,000.
2. **Hash checks.** Use the shard LFS hashes from the Hub API (no download needed) to establish:
   - which stage-3 checkpoint equals each base `main`;
   - which SFT checkpoint equals each SFT `main` (and so which 32B learning-rate run was released);
   - whether `allenai/Olmo-3-32B-Think`'s 15 checkpoints equal the 3.1 run's `step_0050`–`step_0750` (if so, skip them);
   - whether each RL-Zero run and the 7B SFT start from the same base `main`.
3. **Steps and learning rates.** Record the optimizer steps per interval, from branch names, and the learning-rate
   schedule where the released configs or the Olmo 3 report give it. Cite the source for each number. The SFT → DPO
   interval is one jump: record DPO's step count.
4. **Statistics:**
   - **Where:** on CPU pods, streaming shard by shard with `hf_transfer`, in float32 from the bf16 values. Keep only the
     previous checkpoint locally.
   - **Per interval and per tensor:**
     - ‖Δ‖, ‖θ‖;
     - the bitwise-unchanged fraction, and the fraction with |Δ| ≤ 1 ulp of the earlier bf16 value;
     - the dot product with the previous interval's change;
     - the dot product with the offset from the stage start.
   - **What leaves the pods:** only these statistics (CSV or Parquet). Weights stay on pods.
5. **The LoRA comparison:**
   - Get `lora_main` and `lora_ctrl` (rank 16) from the bucket, and check their sha256 against `olmo-ifladder`'s
     records.
   - Form scale · B·A for each targeted matrix.
   - Against θ_t − θ_dpo at every 32B RL checkpoint, compute norm ratios and cosines, per matrix and aggregated by
     module and depth.
   - Add a random rank-16 baseline at matched norm.
6. **Figures:**
   - **Figure 1, the headline** (one per pipeline): relative change per optimizer step since the previous checkpoint,
     on a log scale.
     - Colour the line by training type.
     - Add labelled vertical bands at the stage switches, and highlight each interval that crosses a boundary.
     - Draw the merge and the SFT-run choice as hollow markers.
     - Add a second panel of raw interval change, and the learning-rate overlay where known.
   - **Figure 2:** distance from stage start and straightness.
   - **Figure 3:** per-module and per-layer heatmaps, one per stage.
   - **Figure 4:** unchanged and ≤ 1-ulp fractions.
   - **Figure 5:** the 32B RL section with `olmo-ifcurve`'s IFBench pass@1 and pass@8 and a rug of first-genuine
     steps.
   - **Figure 6:** LoRA against RL.
   - **Figure 7:** 7B RL-from-base against RL-from-DPO.

## Expectations to record before the data (descriptive guesses from the proposal; mark each hit or miss)

- **Size:** RL's change per step is ≥ 10× smaller than SFT's, and RL's total change from DPO is smaller than SFT's
  total.
- **Unchanged weights:** most weights are within 1 ulp across RL.
- **Direction:** RL is straighter than SFT.
- **The LoRA:** `lora_main`'s cosine with RL's change exceeds `lora_ctrl`'s.

## Deliverables

- `run_olmo_weights.md` (≤ 400 words + Figure 1); `numbers.md` § olmo-weights; `log.md`; compute rows.
- A plain-language summary of ≤ 500 words, written for a reader new to weight-space measures.
- `olmo_weights/`: code, the checkpoint manifest with hashes, the step and learning-rate table with sources, the
  per-interval and per-tensor statistics, and Figures 1–7.
- A `STATUS.md` line `OLMO-WEIGHTS DONE <UTC>`; then upload to `hf://buckets/dan-pandori/nd-rl/olmo-weights/`, delete
  pods, and `touch ~/runs/olmo-weights/executor.done`.

**For the reviewer:**
- Recompute a seeded sample of intervals from the raw checkpoints on a pod. Check ‖Δ‖, the ulp fraction and one
  cosine, to 1e-4 relative.
- Check the manifest hashes and the step and learning-rate sources.
- Check that Figure 1's normalisation and stage marks match the table.

**Rules:**
- Do not contact anyone.
- Weights stay on pods.
- At most 4 concurrent processes on the VPS.
- Keep the RunPod balance above $35.
- Wait for jobs with one long poll, not many short check-ins.
