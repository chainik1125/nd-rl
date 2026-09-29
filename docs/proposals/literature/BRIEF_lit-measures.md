# Run brief: lit-measures — three cheap measurements from the literature review

Run id `lit-measures`. Repository `~/work/lit-measures` (branch `dan_lit-measures` from the fork's `origin/dan`,
made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Budget **$3**, ceiling **6 pod-hours**.
Authorised by Dan 2026-09-29 (proposal 16 item M).

Read first:
- nd-rl `docs/proposals/2026-09-29-literature-shortlist.md` §M.
- `docs/literature/2026-09-29-lit-review/REVIEW.md` §(c).
- The `support-followups`, `support-state`, `stage1-dynamics` and `NOISE_FLOOR` material in
  `~/nd-rl/experiment-summaries/` and the fork's `NOISE_FLOOR.md`.

**Code not on `dan`:** `lpool_reread.py` and `state-cap12`'s pod scripts (`pod/sc12/`, including `sn_seed.sh`, the SN-cap12 Stage-1 recipe) are only on `origin/dan_state-cap12` (`lpool_reread.py` also on `origin/dan_long-pool`). Take them with `git checkout origin/dan_state-cap12 -- lpool_reread.py pod/sc12/` and merge them with this run.

Three independent parts. Report each separately. One failing does not stop the others.

## M1. Where EI's new proofs are improbable, for the state models ($0; no sampling)

`support-followups` part D scored the 29 whole-proof survivors' EI proofs step by step under their base.
It found 28 concentrated, 1 mixed and 0 spread. Its classification rule also labels 42 / 53 control
proofs concentrated, so the survivors differed in degree.
- **Repeat for the state models.** Take the theorems SN EI s0 solves that SN base s0 does not reach at
  200,000 attempts per temperature (`support-state`: 7 theorems, files in
  `hf://buckets/dan-pandori/nd-rl/support-state/`). Score each EI proof's actions under the base,
  conditioned on the state.
- **Controls:** as in part D (EI proofs of theorems the base does reach, matched by length).
- **Report** total, worst, second-worst and rest log-probability, with the log base stated.
- **Compare degree, not the label:** worst-step distributions of the new-theorem proofs vs the controls,
  with a permutation test.
- **Second reading** (Invisible Leash 2507.14843v4 App. C.4): the fraction of proofs with a worst step
  below ln(3 / N), where N is the base's attempts. That is −11.1 at N = 200,000 and −11.8 at 400,000.
  Report it for these proofs and for part D's 29.

## M2. Initialisation seed vs data-order seed ($1–2)

`NOISE_FLOOR.md` attributes most variance to run-to-run training. Zhou et al. (2402.09371v1 §4.2) find
large variance from data order alone.
- **Code:** add `--data_seed` to the current Stage-1 trainer (default = `--seed`, so every existing command
  is unchanged; show that the default reproduces one existing run's loss curve).
- **Grid:** 3 init × 3 data-order seeds, on the cap-6 `lean_seq` control recipe `NOISE_FLOOR.md` used.
- **Read-outs:** held-out greedy overall, the depth-3 slice (the bimodal quantity), final validation loss.
- **Report** the share of variance from each factor with an interval. Say whether the depth-3 mode follows
  the init seed, the data order, or neither.

## M3. Mode shares round by round ($0; files only)

Echo Chamber (2504.07912v2) finds RL amplifying one pretraining mode and collapsing others in from-scratch
models.
- **Runs:** ≥ 3 run families with per-round EI found files in the bucket (e.g. the state ladders of
  `state-env`, `state-frontier` and `state-cap12`, the whole-proof C0 ladders, the p2 depth-3 dial arms).
- **Per seed and round,** compute the share of found proofs by style: maximum box depth, and the rule mix.
- **Question:** do seeds that end in the high and low depth-3 modes separate in rounds 1–2 or only later?

## Pre-registration (before the first pod)

An expectation and a falsifier for each part.
- M1: the state models' new proofs are concentrated like part D's.
- M2: data order carries ≥ 30 % of the depth-3 variance.
- M3: modes separate by round 2.

## Deliverables

- The `--data_seed` change, merged into the fork's `dan` after review. Nothing else merges.
- `preregistration/lit-measures.md`, `run_lit_measures.md` (≤ 350 words), `numbers.md` § lit-measures,
  `log.md`, and `STATUS.md` line `LIT-MEASURES DONE <UTC>`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/lit-measures/`.
- Pods deleted; `touch ~/runs/lit-measures/executor.done`.

Never run `test_run_once.sh`.
