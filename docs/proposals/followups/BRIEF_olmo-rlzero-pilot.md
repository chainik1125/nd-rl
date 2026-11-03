# Run brief: olmo-rlzero-pilot — does RL from the base create reach along Olmo 3.1 7B RL-Zero-Math's trajectory, and what would the 32B test cost?

Run id `olmo-rlzero-pilot`. Repository `~/work/olmo-rlzero-pilot` (branch `dan_olmo-rlzero-pilot` from the fork's
`origin/dan`, made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Authorised by Dan 2026-11-02 (proposal 34),
up to $50. If more is needed, ask in `QUESTIONS.md` before spending it.

- **Budget:** **$50 / 90 pod-hours**, dollars binding, with a stage gate:
  - Stage 0 (≤ $3) measures throughput on one checkpoint.
  - Re-cost the whole design from that measurement, and stop and ask (`QUESTIONS.md`) if the projection exceeds $45.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 72 h.

## Why (Dan, 2026-11-02)

The project's controlled results say RL mostly amplifies what pretraining contains:
- **The figure:** nd-rl `docs/writeups/2026-10-31-rl-needed-to-reproduce-sigma/`.
- **The literature review:** `docs/literature/2026-10-19-rl-compute-share/`. It found no create-vs-elicit test on a
  pretrained LLM above ≈ 2 × 10⁻² RL ÷ pretraining compute.

Olmo 3.1 Think 32B spent ≈ 14 % of its pretraining GPU-hours on RL, and its 46 RL checkpoints are public. It is the
natural near-frontier test, but expensive: about $150–250 for a staged design, by an untested estimate.

**This pilot runs first, on Olmo 3.1 7B RL-Zero-Math.** RL there starts straight from the base, with no SFT or DPO in
between, which is the cleaner comparison. It has 28 public RL checkpoints, and its RL data are public. Its RL share is
small, about 10⁻³ (rl-compute-share's row for the 3.0 run: 4.6 × 10⁻⁴ – 2.3 × 10⁻³ over about 2,000 steps; the 3.1
checkpoints run to step 2,800). So the pilot is two things:
1. **A from-base trajectory test at large k.** Olmo 3's own report claims "RLVR pushes the model beyond its initial
   capabilities" from pass@32 alone.
2. **A pipeline and throughput test** that turns the 32B estimate into a measured cost.

## Models and data (verify each before use)

- **RL checkpoints:** `allenai/Olmo-3.1-7B-RL-Zero-Math`, revisions `step_0100` … `step_2800` (every 100), and `main`.
- **Base:** `allenai/Olmo-3-1025-7B`, the model card's `base_model`. Confirm which revision RL-Zero started from
  (probably `main`). If you cannot, say so.
- **RL data:** `allenai/Dolci-RL-Zero-Math-7B`.
- **Recipe:** AI2's open-instruct RL-Zero math script and the reward and verifier code. Use **its prompt template, its
  answer extraction and its verifier**, so that "correct" means what RL rewarded. Record the commit.
- **Data cutoff:** the model card gives Dec 2023.

## Design (a suggestion; pre-register your version)

1. **Evaluation set (about 90–120 problems):**
   - AIME 2024, AIME 2025, and at least one set from 2026 (for example AIME 2026), all after the cutoff;
   - optionally a harder olympiad set, if there is budget.
   - **Contamination:** run an n-gram overlap check against `Dolci-RL-Zero-Math-7B`, and against the midtraining math
     data if it is reachable at reasonable cost. Exclude near-duplicates and report how many.
2. **Checkpoints:** the base plus about 6–7 RL checkpoints, roughly log-spaced: steps 100, 200, 400, 800, 1,400, 2,000
   and 2,800 (= `main`, if it is step 2,800; check).
3. **Sampling:** the same template, temperature, top-p and maximum response length for every model, taken from the RL
   script.
   - Base and final checkpoint: k = 512 on every problem.
   - Intermediate checkpoints: k = 64.
   - **Adaptive base extension:** on problems any RL checkpoint solves but the base has not, sample the base further,
     up to k = 2,048 or the first success.
   - Report unbiased pass@k with problem-bootstrap intervals.
4. **Per-problem dynamics:** at each checkpoint, at matched k, the problems that enter and that leave the solved set,
   relative to the base and to the previous checkpoint. "Reach" means solved by the checkpoint and never by the base
   within its whole budget. Report reach both ways round.
5. **Likelihood:** for every correct proof (solution) an RL checkpoint writes on a reach problem, the base's
   log-likelihood per token against the checkpoint's own. This asks whether the solution sits in the base's support or
   is merely unlikely there.
6. **Overlap of the new solves:** for each reach problem, its nearest RL training prompt in `Dolci-RL-Zero-Math-7B`
   (n-gram or embedding). Report whether reach concentrates on problems close to RL training data.
7. **x-axis:** cumulative RL compute ÷ pretraining compute at each checkpoint.
   - Use rl-compute-share's definitions and its Olmo 3 inputs (pretraining 6·N·D with D = 6.08 T tokens), the 3.1
     script's episodes per step and response lengths, and any logged mean response length.
   - State the assumptions and give low and high bounds.
8. **The 32B decision memo:**
   - measured tokens per second and dollars per million generated tokens, on the GPU you used;
   - a costed design for Olmo 3.1 Think 32B: the DPO checkpoint, 5–6 RL checkpoints from step 50 to 2,300, the same
     adaptive sampling and the likelihood check;
   - what to change, given what the pilot found. Also say how 3.1's step numbering relates to Olmo 3 Think 32B's
     original run, if you can find out.

## Pre-registration

Commit `preregistration/olmo-rlzero-pilot.md` after stage 0's throughput measurement and before any other sampling.
Write expectations for:
- base and final pass@1, pass@64 and pass@512 on each set;
- the number of reach problems, and of problems the base solves that the final checkpoint does not;
- whether reach grows monotonically with RL compute;
- the base's likelihood of RL's reach solutions.

Say what would count as "RL extended reach beyond the base" and what would count as "elicitation only", at the k you
can afford.

## Deliverables

- `run_olmo_rlzero_pilot.md` (≤ 400 words + figures); `numbers.md` § olmo-rlzero-pilot; `log.md`; compute rows.
- A plain-language summary of ≤ 600 words.
- `olmo_pilot/`: the sampling and scoring code, the problem manifest with sources and contamination flags,
  `results.csv` (one row per checkpoint × problem: attempts, successes, first success), the likelihood table, and
  `decision_32b.md`.
- Figures:
  - pass@k curves for the base against the checkpoints;
  - reach, entries and exits against cumulative RL ÷ pretraining compute (the same x-axis as the σ figure);
  - the likelihood panel.
- A `STATUS.md` line `OLMO-RLZERO-PILOT DONE <UTC>`; then upload code and results (not model weights) to
  `hf://buckets/dan-pandori/nd-rl/olmo-rlzero-pilot/`, delete pods, and `touch ~/runs/olmo-rlzero-pilot/executor.done`.

**For the reviewer:**
- Recompute pass@k and reach from `results.csv`.
- Re-grade a sample of counted successes with the verifier, and hand-check 20.
- Check the contamination exclusions.
- Check that every model used the same template and decoding settings.
- Recompute the compute axis.

**Rules:**
- Do not contact anyone.
- Download weights only on pods, never on the VPS: it has ≈ 14 GB free. Use one checkpoint at a time and delete it
  after use, or size the pod's disk for the set.
- At most 4 concurrent processes on the VPS.
- Keep the RunPod balance above $35.
- Do not edit `references.bib` or run `code/tools/literature.py`.
- textbook72 is not involved; print none of it.
