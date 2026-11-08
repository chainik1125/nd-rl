# Run brief: olmo-rlzero-if — does RL straight from the pretrained base create instruction-following reach?

Run id `olmo-rlzero-if`. Repository `~/work/olmo-rlzero-if` (branch `dan_olmo-rlzero-if` from the fork's
`origin/dan_olmo-iffu`, made by `newrun`; the pilot's RL-Zero pipeline and `olmo-ifomega`'s IF checkers, hand-gate rules
and extension code are all on it). Role: executor. `AGENT_POLICY.md` governs. Proposal 39 (nd-rl
`docs/proposals/2026-11-07-olmo-followups.md`); launch only on Dan's go-ahead.

- **Budget:** **$40 / 40 pod-hours.** Dollars are binding.
- **Stage gate:** stage 0 (≤ $3) measures response lengths and throughput. Re-cost, and pause and ask in
  `QUESTIONS.md` if the projection exceeds $36.
- **If over budget, cut in this order:** the optional math-RL arm, then the middle checkpoint.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 48 h.

## The question

On Olmo 3.1 Think 32B, RL created IF reach but not math reach (`olmo-ifomega` against `olmo-think32b`). There, math and
IF also differ in what SFT supplied before RL: math was saturated by DeepSeek R1 distillation. So is the IF/math split a
property of the domain, or of SFT?

The 7B RL-Zero models remove SFT: RL runs straight from the same pretrained base, one run per domain. Math from that
base showed no reach (`olmo-rlzero-pilot`). This run fills the missing cell, IF from the base.

## Design (pre-register your version)

1. **Models:**
   - base `allenai/Olmo-3-1025-7B`; confirm the revision RL-Zero-IF started from, and say if you cannot;
   - `allenai/Olmo-3-7B-RL-Zero-IF` at `step_100`, `step_500` and `main`. Check whether `main` equals `step_1900`, as for
     the math run.
   - **Optional:** `allenai/Olmo-3-7B-RL-Zero-Math` `main` on IFBench only. Does math RL from the base move IF reach?
     Note that the pilot used `Olmo-3.1-7B-RL-Zero-Math`; this is the Olmo 3 sibling of RL-Zero-IF.
2. **Template and decoding:** RL-Zero-IF's own recipe from open-instruct (prompt template, temperature, top-p,
   response cap), the same for every model. Record the commit.
3. **Tasks:**
   - IFBench, all 300 (`allenai/IFBench_test`). Report all 300, and the 218 that `olmo-ifomega` kept after its SFT
     exclusion, for comparability.
   - IFEval, the same seeded 150-prompt subset `olmo-ifomega` used.
   - **Contamination:** scan every prompt against `allenai/Dolci-RL-Zero-IF-7B` (13,179 prompts). Exclude
     near-duplicates and report them.
4. **Grading:** the official strict checkers (primary) and loose (secondary), plus `olmo-ifomega`'s hand gate (rules in
   `olmo_ifomega/handcheck_rule_if.md`; blinded reader sub-agents, confirmed by you) on every deciding success.
5. **Protocol (`olmo-ifomega`'s, symmetric):**
   - the base at k = 16 and each RL checkpoint at k = 8;
   - **reach candidates** (an RL checkpoint ≥ 1, the base 0): extend the base to its second genuine success, up to 256;
   - **reverse candidates:** the same rule for `main`;
   - confirm with ≥ 2 successes including a fresh one, for every model.
6. **Format check (required):** on every reach prompt, classify a seeded sample of the base's failures:
   - no usable answer (rambling continuation, no answer segment);
   - ignored the constraint;
   - attempted it, far off;
   - near miss;
   - broke another constraint.

   A pretrained base can fail IF prompts just by not answering cleanly; report how much of reach that explains.
7. **Compute axis:** cumulative RL ÷ pretraining compute per checkpoint, using rl-compute-share's definitions and the
   RL-Zero-IF script's settings.

## Verdict rule (per task, as `olmo-ifomega`)

- **"Reach":** ≥ 3 prompts where `main` is confirmed and the base has no genuine success within 256.
- **"No reach at 256":** every candidate is resolved for the base under the primary grade and by hand.
- **"Inconclusive":** anything else.

Report the gamed share among RL's first-read strict successes, against the 32B run's ≈ 36 %.

## Deliverables

- `run_olmo_rlzero_if.md` (≤ 400 words + one figure: the 2×2 grid filled with reach shares); `numbers.md` §
  olmo-rlzero-if; `log.md`; compute rows.
- A plain-language summary of ≤ 500 words.
- `olmo_rlzero_if/`: code, the manifest with contamination flags, `results.csv`, the hand-check table, the failure-mode
  table, and every sample's text in the bucket.
- A `STATUS.md` line `OLMO-RLZERO-IF DONE <UTC>`; then upload to `hf://buckets/dan-pandori/nd-rl/olmo-rlzero-if/`,
  delete pods, and `touch ~/runs/olmo-rlzero-if/executor.done`.

**For the reviewer:**
- Re-grade the stored texts, and repeat the hand gate blind on the deciding responses.
- Check the template and decoding parity.
- Check the contamination exclusions.
- Recompute the verdicts.

**Rules:**
- Do not contact anyone.
- Weights stay on pods.
- At most 4 concurrent processes on the VPS.
- Keep the RunPod balance above $35.
- Wait for jobs with one long poll, not many short check-ins.
