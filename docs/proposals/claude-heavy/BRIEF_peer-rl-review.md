# Run brief: peer-rl-review — what the other mentees found about improving RL, compared with our results

Run id `peer-rl-review`. Repository `~/work/peer-rl-review` (branch `dan_peer-rl-review` from the fork's `origin/dan`,
made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Authorised by Dan 2026-10-17.

- **Budget:** **$30 / 60 pod-hours**, optional. Use it only to put numbers on a common protocol (below).
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 72 h; no hurry.

## The request

Dan: "Some of the other mentees have recently done work on improving the RL performance. Can you … review those
results and summarize them for me? In particular, compare the performance to the results we've already gotten
ourselves, and give thoughts on how those differences came to be."

## Whose work, and where (nd-rl branches; read-only)

**Start here, then follow links and look for anything newer:**
`git fetch --prune` and `git for-each-ref --sort=-committerdate refs/remotes/origin`, ignoring `dan` / `dan_*`.

- **Leon** (most active, 2026-10-05 → now). Under `code/experiments/current/thinking_traces_20261005/` on `leon/main`:
  - `RECIPE.md`, `grpo_v2/README.md`, `grpo_v2/launch/README.md`, `matched_eval/v2_results.md`;
  - on `leon/grpo-v2` / `leon/thinking`: `RL_RESULTS.md`, `gap_analysis/`, `collapse_audit/`, `hard712/`,
    `search_traces/`, `curriculum/`;
  - the GRPO v2 arm branches: `leon/grpo-v2-{merged,prior,feedback,pilots,sched,auto}`;
  - `leon/long-pool`, `leon/thinking-curriculum`, `leon/thinking-auto`.
- **Robbie:**
  - on `robbie-experiments`: `experiment-summaries/2026-10-08-retry-ei/`, `2026-10-14-grpo-wall-model-diff/`,
    `code/experiments/current/temp_sweep_20261006/`, and the earlier combined-model and bigger-model summaries;
  - `autoresearch-thoughts` and `autoresearch-loops` (test-time work);
  - his blog (robbiewmthompson.com).
- **Dmitry:** `dmitry/main` and `dmitry/selection-sprint`: `experiment-summaries/2026-10-06-proof-selection-sprint/`,
  `2026-10-06-learned-compute-allocation/`, `2026-10-06-learned-compute-transfer/`, and
  `proposals/learned-compute-allocation/`.
- **Charles:** `charles/main`. His work is mainly data: the textbook release and the `pretrain_data` releases,
  including `release_2026-10-12_textbook72_distract2` (the premise-count wall). Include it where it bears on RL
  performance.

## Our side (what to compare against)

- **Model cards:** `docs/models/state_env_october.md` and `state_env_october_r16.md`.
- **Summaries:** `trajectory`, `rl-continue`, `guided-tts`, `grpo-best`, `mcts-a`, `compute-match`, `capability-defs`,
  `create-or-elicit` and the paper draft (`docs/papers/create-or-elicit/`).
- **The evidence atlas** (`experiment-summaries/2026-10-02-evidence-atlas-…/ATLAS.md`): it already harmonises protocols
  across people up to 2026-10-02.

## What to produce

1. **Per person:** what they tried to improve RL, what worked, what didn't, and their headline numbers *as they report
   them*. Give the protocol behind each number:
   - model and size, interface (whole proof vs proof state), training data and length cap;
   - the RL algorithm and compute;
   - **the judge** (Lean alone, Lean ∧ `nd_verify`, or `nd_verify`);
   - k, temperature, and the evaluation sets: textbook72 dev58 / train14, Robbie's dev metric, holdout sets,
     Charles's release.

   Separate what is reviewed from what is not.
2. **A comparison with our results:**
   - a table that lines up the closest comparable numbers;
   - a protocol-difference column;
   - a verdict on whether each comparison is apples to apples.
3. **Explanations of the differences.** For each material one, give the most likely causes, with evidence. Candidates:
   interface, model size, data and premise-count coverage, RL algorithm, compute, the judge, sampling settings,
   evaluation-set composition, seed noise (our `NOISE_FLOOR.md`). Mark which are established and which are
   hypotheses.
4. **Lessons in both directions:** what we could adopt from them, and what they might adopt from us (guided read-out,
   the proof-state interface, budgeted creation tests). Rank by expected value.

## Common-protocol numbers (optional; the $30 budget)

Where a comparison hinges on protocol, put the two sides on one protocol, pre-registering each job in `log.md` first.
- **Ours on theirs:** score our checkpoints with their protocol.
- **Theirs on ours:** score their published checkpoints (e.g. Robbie's HF repo, the `ndrl` org) on ours: Lean alone,
  k 256, T 0.8, plain and guided where possible, textbook72 scoring only.

Prefer two or three decisive cross-checks to many shallow ones. Never train on anything.

## Deliverables

- `peer_rl_review/REPORT.md` (≤ 3,000 words plus tables). It opens with a plain summary of ≤ 200 words for Dan, then
  items 1–4, then any common-protocol results.
- `numbers.md` § peer-rl-review, `log.md`, `run_peer_rl_review.md` (≤ 200 words), `STATUS.md` line
  `PEER-RL-REVIEW DONE <UTC>`, and `touch ~/runs/peer-rl-review/executor.done`.
- **The librarian** copies the report to nd-rl `docs/project_strategy/2026-10-17-peer-rl-review.md` and writes the
  summary as usual.

**For the reviewer:** check each quoted number against its source, each protocol attribution, and each "apples to
apples" verdict.

**Rules:**
- Other people's branches and repositories are read-only.
- Do not contact anyone, and do not comment on their PRs or issues. The report is for Dan only.
- textbook72 is scoring only.
- At most 4 concurrent processes on the VPS; downloads one at a time.
