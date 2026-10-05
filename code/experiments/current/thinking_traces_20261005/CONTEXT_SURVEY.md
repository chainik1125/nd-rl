---
written_on: 2026-10-05
written_by: agent:claude
---

# Context survey for the thinking-traces experiment

A read-only survey of team work and Dan's pipeline, made 2026-10-05 against origin/dan 816e553 and
Dan's fork (`dan-pandori/nd-takehome`, branch `dan`, 4dcd0e4f). Fork paths are relative to the fork; nd-rl
paths are relative to this repository.

## What has worked for test-time scaling so far

- **Dan, guided-tts** (2026-10-03): redrawing a step that `step_check.py` rejects (without replacement, up to
  10 redraws) beats plain resampling at matched sampled tokens on the 90 long (>10-line) release theorems:
  +9.1 pp (cap 12) / +8.3 pp (cap 6) IQM at k 64, 6/6 models. Redrawing structural rejections only: |Δ| ≤ 1.2 pp.
- **Dan, mcts-a**: PUCT + learned value on r8 solves 2/2/3 group-C theorems vs 4/2/4 for k-256 sampling at
  matched wall clock (gate failed; mcts-b not run).
- **Robbie, test-time scaling** (frozen whole-proof c005, 1,177 held-out): resampling invalid lines raises
  pass@1 31.2% → 57.4%. Compute-matched beam (76.6%) and best-first (75.8%) are below plain; SMC 84.4%. "50-55% of
  rejections redraw a segment already rejected at the same prefix" (`experiment-summaries/2026-09-29-test-time-scaling/README.md`
  on robbie-experiments). TTRL: plain EI on the eval statements wins (1,088 vs 1,079 of 1,108).
- Reading: only verifier-in-the-loop has helped. Learned search over a frozen model has not.

## Prior thinking experiment: r1_cot (2026-09-09, old Fitch stack, not Dan's state env)

- Added `<THINK> </THINK> <FINAL>` (and `<DRAFT> <ERROR> <REVISE>`); the trace was the proof's own lines in
  reverse (`backward_plan`, `code/experiments/current/r1_cot/run.py:234-247`). **The trace was a deterministic function
  of the answer**: no search, no failed steps, no verifier feedback.
- SFT cold start on a direct-proof parent (not pretraining), then 256 RL updates.
- Result (`experiment-summaries/2026-09-08-r1-cot/pair1_review.md`): CoT 10/58 vs direct 11/58 at k 16 (CoT's
  solved set ⊂ direct's), greedy 4/58 vs 10/58, about 2× the tokens; direct dominates at every matched token budget.
  The s1-style budget-forcing plan (`inference_scaling_plan.md`) was never implemented.
- Lesson: a trace that re-renders the answer costs tokens and buys nothing. It has to carry computation the
  answer does not (rejected candidate steps, check outcomes, backtracks, subgoals).

## state_env_october pipeline (fork)

- Stage 1: `python3 state_train.py --recipe best --data data/kh/train_k12.jsonl --heldout data/p2/heldout.jsonl
  --mode lean_staten --cap 12 --seed S --budget_secs 1200 --out …` (`pod/bs/stage1.sh`). `--recipe best` →
  `state_train_best.train`: Muon + AdamW, MTP weight 0.3, token-budget batches 128×144, cosine over wall clock.
  s0: 23,951 steps, 3.67 epochs of 1.24M (state, action) pairs, about 21.5 A40-min. On a 3090 pass `--best_steps 24000`.
- T1 EI: `python3 state_ladder_ei.py --init CK --name … --seed S --batch 2048 --train data/kh/train_k12.jsonl …`;
  8 rounds, k 32, T 0.8, `max_action` 256, `max_steps` 48, 4,495 rl_targets; ≤4 proofs/theorem × weight 4 + 20,000
  retained K12; fine-tune = `state_train.py --steps 600 --lr 3e-4` (plain AdamW). 7.2-8.5 A40-h per seed.
- Read-outs: `state_eval.py` (plain) and `guided_eval.py --arm {plain,structural,logical}`; Lean 4 core judge
  (`lean_gate.py`, Lean path hardcoded at :26; local Lean is 4.34.0 vs 4.34.1 on pods).
- Data: K12 155,000 records, 14,091 per length 2-12, 0-3 premises. Local copy
  `/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl` (md5 matches). Record: `prompt`
  (`THM … SEQ … PRF`), `proof` (old-style numbered lines with rules), `n_lines`, `rules`, `n_prem`, ….
- Serialisation: `state_env.decompose(prompt, proof, canon=True)` → (state, action) pairs. The state is Markov
  (hypotheses in scope + goal, no history), e.g.
  `<st> h1 : ( ( R ∨ P ) ∨ ( S → Q ) ) <nl> h2 : … <nl> ⊢ ( R → … ) <act>` → action
  `have n1 : ( ( R ∨ P ) ∨ ( S → Q ) ) := h1 ; <eos>`. K12: mean 8.0 actions/proof, longest state 386 tokens,
  longest action 220, mean pair 96.9.
- Tokenizer (`lean_tok.py`, mode `lean_staten`): 111 tokens, pad 0, eos 1, ref0 = 47:
  `<pad> <eos> ( ) ¬ ∧ ∨ → P Q R S False theorem t : Prop := by have exact ; fun => ⟨ ⟩ , .1 .2 .elim Or.inl Or.inr
  Or.elim Classical.byContradiction hh <st> <nl> ⊢ <act> h1–h8 n1–n64`. 4 atoms, MAXH 8, MAXN 64 (random name offset in
  training; env base drawn U[0,32] at sampling). No hard length limit (ALiBi), but `max_len` 1024 in config and
  `max_action` 256/512, `max_steps` 48/96.
- Model: `best_model.ALiBiGPT`, 6 layers, d 384, 8 heads (4 ALiBi + 4 NoPE), d_ff 1280, Peri-LN, 9,560,832 params
  (+1.88M MTP head, not saved).
- Local checkpoints: `/scratch/cowifornia/nd-rl/bench/dan_ckpts/best-state/{stage1,ladder}/` (best12 s0-2 r8 =
  state_env_october). r16 only in Dan's bucket.

## Integration constraints for a thinking variant

- Token ids must stay < 256 (stored as uint8 / `bytes()`, `state_train_best.py:36-39`, `state_train.py:39-40`).
- Any id ≥ ref0 is treated as a name and shifted (`lean_tok.py:160-181`, `state_train_best.py:66-71`,
  `fast_train.py`, `sample.py`). New tokens must go before `h1` (changes ref0; fine from scratch) or the shift
  must be bounded to [ref0, ref0+64).
- Add a new mode to `STATE_MODES`; update the `canon = mode == 'lean_staten'` test (`lean_tok.py:33-34,122-126`).
- Loss mask: state 0, action + `<eos>` 1 (`state_train.py:47-66`; `state_train_best.py:65`; MTP uses the same
  mask). A think segment placed in the action is trained.
- A think segment must not contain `<eos>`, must be stripped before `Env.apply` (`state_sample.py:75-78`,
  `guided_sample.py`), and must fit `max_action`. EI replay stores only the ND `proof`, so traces are dropped in
  replay unless the sampled tokens are stored.
- Running locally: set `ND_OFFLINE=1` (otherwise `save_ckpt` uploads to Dan's bucket, `model.py:119-125`);
  run from our own copy of the code (registry rows are written under the code directory); venv first on PATH.
  The venv has torch 2.14 + CUDA, no huggingface_hub or scipy.

## Saturation, as stated by the team

- Week in review (`docs/project_strategy/2026-10-01-week-in-review.md:8-10`): "we built a model organism that largely
  solves our benchmarks… saturates our long-proof pools".
- Model cards: RL targets 97.1-98.0% solved by r8; 62-92 of 4,495 unsolved at r16; long pools nearly saturated.
- Headroom that remains: textbook72 about 21-33% unsolved; Charles's release (best-state r8 at k 32: candidate_v0
  70-73%, v1 61-62.5%, batch3 48-57%, hand_proved_14 0%; by reference length 14-20 lines 33%, 21+ lines 3%; all 7
  four-premise problems unsolved); trajectory's group C (theorems unsolved by base and r8).
