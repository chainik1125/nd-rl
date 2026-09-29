# Run brief: search-expert — does a search expert teach the network more than its own samples?

Run id `search-expert`. Repository `~/work/search-expert` (branch `dan_search-expert` from the fork's `origin/dan`,
made by `newrun`; starts after `long-pool-2` is recorded). Role: executor. `AGENT_POLICY.md` governs. Budget
**$25**, ceiling **50 pod-hours**. Authorised by Dan 2026-09-29 (proposal 16 item 2).

Read first:
- nd-rl `docs/proposals/2026-09-29-literature-shortlist.md`: §2 and the plain-language appendix. The appendix
  is the question in Dan's terms.
- `docs/literature/2026-09-29-lit-review/REVIEW.md`: shortlist #1 and #5, and the notes on BFS-Prover,
  HTPS, InternLM2.5-StepProver, ExIt and DeepSeek-Prover-V1.5.
- The `state-cap12` and `long-pool-2` summaries.
- `state_ladder_ei.py`, `state_sample.py` (`env_generate`), `state_env.py` (`Env`, `decompose`), and
  `sample.py` (`generate_ids_fast`, which discards per-token log-probs).

**Code not on `dan`:** `lpool_reread.py` and `pod/sc12/` (including `sn_seed.sh`, the SN-cap12 Stage-1
recipe) are only on `origin/dan_state-cap12`. Take them with `git checkout origin/dan_state-cap12 --
lpool_reread.py pod/sc12/`.

## Question

Our expert iteration uses the network's own independent samples as the expert, so the expert is no stronger
than the apprentice. Replace it with best-first search over proof states, at **matched compute**. Does the
apprentice, **evaluated without search**, then prove more past the frontier?

## Design

- **Model:** SN-cap12, 6 Stage-1 seeds.
  - s0–s3 are in `hf://buckets/dan-pandori/nd-rl/state-cap12/ckpts/sc12/`.
  - s4–s5: reuse `frontier-supply`'s if they are in its bucket (same recipe); otherwise train them.
- **Arms** from each seed's Stage-1 checkpoint, under `state-cap12`'s ladder protocol (8 rounds, T 0.8,
  replay K12, the same targets):
  - **A. Sampling expert:** the current loop.
  - **B. Search expert:** best-first over (theorem, state) nodes.
    - Priority is length-normalised log-probability, Σ log p / L^α with α = 1 (BFS-Prover 2502.03438v3
      Eq. 1).
    - Expand the best node with 4 sampled actions (deduplicated). The environment rejects invalid steps.
    - Stop a theorem when it is proved or its action budget is spent.
    - Batch expansions across theorems so the GPU stays busy.
  - **C. Truncate-and-resume** (DeepSeek-Prover-V1.5 2408.08152v1 §3.1), on 2 seeds: after a failed
    attempt, resume from the state before the failing step.
- **The same selection rule in every arm:** train on the **shortest** found proof per theorem (HTPS
  2205.11491v1 Table 4). The arms then differ only in the expert. If you also want the current
  "all successes" rule, add it as a separate arm A′ on 2 seeds.
- **The same fine-tune** in every arm: steps, learning rate, replay.
- **Same-checkpoint variance:** reuse `frontier-supply`'s measurement if it exists (same protocol).
  Otherwise, run arm A twice with different sampling seeds on 2 seeds.

## Compute matching (Dan, 2026-09-29)

The comparison is only meaningful at equal compute.
- **Primary match: generated actions per round.** In the state environment, each action costs one state
  encoding plus the action's tokens in both arms: the policy sees the current state, not the history. So
  equal actions means roughly equal model FLOPs.
- **Setting B's budget:** use arm A's actions per theorem in the same round on the same seed (k 32 ×
  attempt lengths).
- **Report per arm and round:** generated actions and tokens, fine-tune tokens, environment and Lean checks,
  and **GPU-seconds** (measured per job).
- **If search is less efficient on the GPU:** if B's GPU-seconds exceed A's by more than 25 %, add **A+** on
  2 seeds, a sampling expert given B's GPU-seconds (more attempts). The question then becomes whether
  search beats simply sampling more at equal cost.
- **Search at test time** is secondary. Report it on one seed as an extra read-out, never as the result.

## Read-out

- Final apprentices, **without search**: `lpool_reread.py`, k 256, T 0.8, `max_steps` 96, on
  `transfer_long2` (per upper-bound bin) and `rr600` 13–16.
- Paired per seed, B vs A (and C, A+ where run): per-bin solve counts, `L*`, and the length of found
  proofs.
- Per round: proofs found by each expert on theorems past the other arm's frontier. This is the sanity
  check that the search does its job.

## Pre-registration (before the first pod)

- **Quantity:** per-seed solve count in the two bins nearest arm A's frontier, and `L*`.
- **Statistics:** the paired MDD, from the same-checkpoint spread. Per-seed values, IQM, stratified-bootstrap
  95 % CI.
- **Expectation:** B's expert finds ≥ 2× A's proofs per round on theorems past A's frontier at equal
  actions, and B's apprentice beats A's by more than the MDD.
- **Falsifier:** B − A ≤ 0 within the MDD, or search finds no more than sampling at equal actions.
- **Stated as a guess, not from a source:** B's found proofs are longer. The one length-specific source,
  InternLM2.5-StepProver, favours a critic over best-first.

## Deliverables

- The search code, merged into the fork's `dan` after review.
- `preregistration/search-expert.md`, `run_search_expert.md` (≤ 400 words + figure), `numbers.md`
  § search-expert, `log.md`, and `STATUS.md` line `SEARCH-EXPERT DONE <UTC>`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/search-expert/{ckpts,artifacts}`.
- Pods deleted; `touch ~/runs/search-expert/executor.done`.

Never run `test_run_once.sh`. A learned value head (proposal 16 item 5) is out of scope. It comes only if
B beats A.
