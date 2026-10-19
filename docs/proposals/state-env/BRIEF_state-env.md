# Run brief: state-env — an AlphaProof-style state for the policy

Run id `state-env`. Repository `~/work/state-env` (branch `dan_state-env` from the fork's `origin/dan`,
made by `newrun`: a sparse checkout, so earlier runs' `artifacts/<dir>/` are not on disk — `git show`
or `git sparse-checkout add` them). Role: executor. `AGENT_POLICY.md` (nd-rl canonical) governs; the
design below is a suggestion, the question is what matters. Budget **$30**, ceiling **60
pod-hours**, registered. Balance floor $100 (`rpbalance`); nothing else is running.

Read first: **proposal 13**, `~/nd-rl/docs/proposals/2026-09-28-state-conditioned-environment.md` (the
question, what AlphaProof's policy sees, the arms and what would count). Then `lean_tok.py`
(`lean_seq` rendering and its parser), `nd2lean.py`, `lean_judge.py`, `sample.py`, `train.py` (its
`batch()` already masks the loss to the part after the prompt), `ladder_ei.py`, `NOISE_FLOOR.md`.

## Question

Does giving the policy the proof state — every hypothesis in scope and the current goal, as a
pretty-printed Lean tactic state — move the length wall that every whole-proof model in this project
hits (T1 `L*` = 12; `L_true` ≥ 13 solved 0 times in 11 ladder runs)?

## What to build

1. **State renderer** for the `lean_seq` fragment: given a theorem and a proof prefix, the focused
   goal's tactic state — hypotheses in scope as `name : formula` (premises `h_i`, earlier `have`s at
   this or an enclosing level, box binders), then `⊢ goal`. Box semantics as in proposal 13
   (`A → B`: bind `n : A`, `⊢ B`; `¬A`: bind `n : A`, `⊢ False`; `Or.elim`: two subgoals, one at a time;
   `Classical.byContradiction`: bind `h : ¬A`, `⊢ False`; `exact n` closes the focused goal and adds the
   finished `have` to the parent). A new tokenizer mode (e.g. `lean_state`) adding the few tokens needed
   (`⊢`, a hypothesis separator, state delimiters); everything else is `lean_seq`'s vocabulary, with its
   random name offset.
2. **Step decomposition**: every `lean_seq` proof → a sequence of (state, action) pairs, where an
   action is one step — a `have … ;` line, a box opening up to its `by`, or `exact n ;` — and the
   actions concatenate back to the original text **byte for byte**.
3. **Environment loop** (batched): per attempt, render the state → the model generates one action
   (stop at the step boundary; set the action `max_new` from the data and report how often it is hit)
   → apply it syntactically (parse; cited names in scope; `exact n` only if `n`'s declared type is the
   goal; a step cap per attempt) → repeat until the top-level goal closes or the attempt fails. The
   assembled `lean_seq` text goes to **`lean_judge.py`** — Lean alone decides, as everywhere else.
   Scope, hypotheses and goal follow from the actions' syntax because every `have` declares its
   formula; a wrong term is caught by Lean at the end. A per-step symbolic type check is allowed as an
   early filter only if you measure its agreement with Lean and report it.
4. **Stage-1 on (state, action) pairs** from the control's set `data/p2/train_depth3_f0_a1.jsonl`,
   loss on the action tokens only (the state is an observation). Keep the control's model and schedule
   (4 layers, d 256, 6,000 steps); decide how many pairs per batch so the model sees the same proofs
   the same number of times as the control, and say what you chose.
5. **EI in the environment**: `ladder_ei.py`'s rung T1 (8 rounds × k 32 on `data/ladder/rl_targets.jsonl`,
   read out on `data/ladder/transfer.jsonl`) with the frozen control at equal attempts, the same
   retained-supervised mix and hyperparameters, found proofs converted to (state, action) pairs.

## Arms

- **S — state only**, AlphaProof-faithful (input: the tactic state). Primary.
- **SH — proof so far + state.** Separates "the state helps" from "dropping the history helps". Run if
  the budget allows; dropped first.
- **C0 — not re-run.** The whole-proof control, same model size and data, on file (label it so; it was
  measured under Lean ∧ `nd_verify` and at other sampler settings — a re-draw, not a bias):
  `ds-generator` (checkpoints `ckpts/lf/stage1_a1_seq_s{0,1}.pt`, batch 512): T1 transfer 890 / 965,
  `L*` 12 / 11; frozen 158 / 114, `L*` 9 / 9; held-out greedy 0.9088 / 0.8968. The same checkpoints gave
  T1 856 (`cap-horizon`, s0) and 923 / 1,003 (`ds-rendering`, batch 2,048) — that spread is the
  re-draw. Textbook at T1 (s0): 38 of 760 (`ds-composition` D4). `L_true` ≥ 13: 0 in all 11
  `ds-generator` ladder runs.

Two Stage-1 seeds per arm.

## Gates before any RL (CPU; commit the results with the pre-registration)

1. **Round trip:** every proof of the control set decomposes and reassembles byte for byte; a random
   5,000 of the reassembled texts are accepted by `lean_judge.py`.
2. **Your state is Lean's state:** for ≥ 1,000 random proof prefixes, cut the proof there, ask Lean for
   its goal state (e.g. a `trace_state` or `sorry` at the cut, parsed from Lean's output), and compare
   hypotheses (names and types) and goal with your renderer's. Report every mismatch; zero is the bar.
   This is what makes the state trustworthy, and the reviewer will re-run it.
3. **Environment replay:** feed each control proof's actions through the environment loop as if the
   model had produced them; all must complete and be accepted.

## Measurements (per arm × seed)

Held-out greedy (5,000; per length; depth-3 slice); ladder T1 and frozen: transfer solved, `L*`,
solved at `L_true` ≥ 13, textbook schemata (19 × 40); per-step diagnostics — attempts ended by a
syntactic error vs by Lean at the end, mean steps per attempt, action truncation rate. Sampler settings
per the policy (largest batch that fits; fixed across arms). Report term size beside line counts.

## Pre-registration (`preregistration/state-env.md`, before the first pod)

Your numbers for each headline quantity per arm, and the two outcomes from proposal 13 as falsifiers:
**the state is the wall** if S reaches T1 `L*` ≥ 13 with ≥ 5 solved at `L_true` ≥ 13 on both seeds;
**the state is not the wall (at this scale)** if S stays at T1 `L*` ≤ 12 with 0 solved at `L_true` ≥ 13
on both seeds. Also predict what happens to held-out and to the frozen ladder, and to SH relative to S.

## Budget and order

`podbudget state-env` is registered at 60 h / $30. Build and gate 1–3 on the VPS (2 vCPU; Lean at
`~/.elan/bin/lean`) or one cheap pod's cores. Then per seed: Stage-1 S → held-out → T1 + frozen. Then
SH if the projection allows. Drop order: SH, then seed-1 frozen, then seed-1 T1. Record GPU class and
billed rate; upload `ckpts/` (including every EI round) and `artifacts/` as you go.

## Deliverables

`preregistration/state-env.md`; the environment code (renderer, decomposition, env loop, `lean_state`
tokenizer mode, EI driver) with its tests; `STATE_ENV.md` at the repo root — one page: what the policy
sees, an example state and action sequence for one proof, the three gates' results, how to run it;
`run_state_env.md` (≤ 400 words + a figure: transfer solved by `L_true`, S vs C0, T1 and frozen);
`numbers.md` § state-env (every entry names its source file and model); `log.md` dated;
`artifacts/se/summary.json`; `STATUS.md` line `STATE-ENV DONE <UTC>`; bucket
`hf://buckets/dan-pandori/nd-rl/state-env/{ckpts,artifacts,data}`; pods deleted;
`touch ~/runs/state-env/executor.done`. Questions for Dan in `QUESTIONS.md` with your default.
