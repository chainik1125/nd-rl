# Run brief: guided-tts — guided sampling against plain resampling at test time, on textbook problems, at matched compute

Run id `guided-tts`. Repository `~/work/guided-tts` (branch `dan_guided-tts` from the fork's `origin/dan`, made by `newrun`).
Role: executor. `AGENT_POLICY.md` governs, including per-arm compute rows. Budget **$10**, ceiling **20 pod-hours**.
Authorised by Dan 2026-10-03. His goal for the week is test-time scaling, so he wants a cheap comparison of Robbie's
"guided sampling at test time only" against plain full resampling, on our best existing models. Design: nd-rl
`docs/proposals/2026-10-03-guided-sampling.md` (proposal 24). **This run is Part A only, narrowed. No training.**

Read first:
- Proposal 24, including the without-replacement section.
- Robbie's `experiment-summaries/2026-09-29-test-time-scaling/README.md` and
  `code/experiments/current/test_time_scaling/engine.py` (`_swor_shift`), both on nd-rl branch `robbie-experiments`.
- `state_env.py`. Its docstring says the environment does **no symbolic type check**.
- `state_sample.py`, where an invalid action currently ends the attempt.
- The `trajectory`, `trajectory-cap6` and `textbook72` summaries.

## Models (existing checkpoints only)

- **cap 12:** `trajectory`'s `la_T1_best12_s{0,1,2}_r8.pt`, the best models to date.
- **cap 6:** `trajectory-cap6`'s `la_T1_best6_s{0,1,2}_r8.pt`, Robbie's setting.

Both are Robbie's 9.56M recipe in the proof-state environment (`lean_staten`).

## Problems (scoring only; never trained or tuned on)

- **textbook72:** the eval-only copy and manifest rule on `origin/dan_textbook72`.
- **Charles's release:** nd-rl `origin/charles/main`, `data/textbook_problem/release_2026-10-02/`. Use
  `candidate_v0` (87), `candidate_v1` (72), `batch3` (14) and `hand_proved_14` (14).
  - Check the sha256 values against its `MANIFEST.json`.
  - Skip the derived set. Its inputs are not plain theorems.
  - Report per file and by `min_lines` bin (≤ 10, 11–20, > 20).

## Arms

- **plain:** independent full attempts. This is the current sampler: an invalid step ends the attempt.
- **guided-logical:** a **per-step check that agrees with Lean.**
  - Never reject a step Lean would accept (`Not.elim` and the rest are allowed). Never use `nd_verify`.
  - Build a fast checker for the `lean_seq` term fragment, and use it only after it agrees with Lean on ≥ 50,000
    steps with **0 false rejects**. Otherwise check per sampling wave with batched Lean.
  - On a failure, **redraw the step from the same state without replacement:** exact, reusing Robbie's
    `_swor_shift`, with the forbidden set kept per attempt.
  - Up to 10 rejections per attempt, then the attempt fails.
  - Cache verdicts per (state, step).
- **guided-structural:** the same redraw, but only for steps the environment already rejects (grammar, names,
  `exact`). This shows how much of any gain needs the logical check.

## Compute matching (the point of the run)

- Measure **sampled tokens per problem, rejected draws included,** and **wall-clock** (GPU and Lean or checker time,
  separately).
- For each arm, build the **solve-rate-vs-budget curve** from k = 1 to k = 256 plain-equivalent. Compare at
  matched tokens and at matched wall-clock. Use the unbiased pass@k estimator; attempts stay i.i.d. because forbidden
  sets are per attempt.
- Use the same temperature (T 0.8), `max_steps` 96 and `max_action` 512 in every arm.

## Pre-registration (before the first pod)

- **Expectation:** guided-logical beats plain at matched tokens on problems over 10 lines and on the
  Roy / Pelletier / batch3 sets. Little or no gain on short ones. guided-structural is close to plain.
- **Falsifier:** guided-logical ≤ plain at every budget on ≥ 2 of 3 seeds at both caps.
- **Report:** the per-step rejection rate and causes, and how often a with-replacement redraw would have repeated a
  rejected step.

## Deliverables

- `preregistration/guided-tts.md`, `run_guided_tts.md` (≤ 300 words + the scaling-curve figure), `numbers.md`
  § guided-tts, `log.md`, and `STATUS.md` line `GUIDED-TTS DONE <UTC>`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/guided-tts/artifacts`.
- Pods deleted; `touch ~/runs/guided-tts/executor.done`.
- Merge the checker and the guided sampler into the fork's `dan` after review. Proposals 20 and 22 can reuse them.

Never train or tune on any problem scored here. Never run `test_run_once.sh`. At most 4 concurrent processes on the VPS.
Do not contact Charles or Robbie.
