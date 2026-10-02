# eval_dan_best_state: Dan's newest RL models (best-state cap-12 T1 r8, three seeds) on the textbook-problem datasets

Pre-registered 2026-10-02 (UTC), before the run. Charles: "Dan should've released some new RL models — check there."

## Checkpoints

Dan's public HF bucket `dan-pandori/nd-rl` (47,143 files; listed through
`https://huggingface.co/api/buckets/dan-pandori/nd-rl/tree/<prefix>`). RL runs newer than Robbie's
`scale-*` pretrains: best-state (done 2026-10-01 05:00 UTC), trajectory (2026-10-01), rl-from-ckpt
(2026-10-02 10:54) and grpo-best (GRPO arms × 3 seeds, checkpoints still uploading at 15:56 UTC today,
run not DONE). This experiment scores the strongest completed family:

`best-state/ckpts/bs/ladder/la_T1_best12_s{0,1,2}_r8.pt` — Robbie's recipe (ALiBiGPT 6 × 384, Peri-LN,
9.56M params) trained from scratch in Dan's proof-state format (`lean_staten`, cap 12), then eight
rounds of T1 expert iteration. Dan's own read-outs (unreviewed, `run_best_state.md`): textbook72
pass@256 IQM 51.7 / 72; dev metric 1,054 / 1,108. Three seeds, so seed-to-seed spread can be reported;
still no comparison claim against Robbie's pretrains, which were one seed each and a different protocol.

Local copies (sha256 recorded in `run_*/log.txt`): scratchpad `dan_ckpts/`, 38,278,981 bytes each.

## Code

Dan's own evaluator at the `dan` branch tip (0c492902, worktree `scratchpad/fork_dan`):
`state_eval.py --ckpt … --in <set>.jsonl --k 32 --temperature 1.0 --batch 256 --seed 0`, which samples
in the step environment (`state_sample.env_generate`, max 48 steps, 256 tokens per action) and gates
through Lean 4.34.1 (`~/.elan/bin/lean`). His rows carry the distinct Lean-accepted ND proofs and the
per-sample counts (`n_ok` / `n_tried`). Post-processing (`code/textbook_problem/dan_metrics.py`, to be
written): every accepted proof is re-judged by the pinned nd_verify through strict_verify; rows where
Lean and nd_verify disagree are counted and listed; avg solve rate = mean over problems of
`n_ok / n_tried`, reported alongside any-of-32, with the disagreement count as a caveat.

Prefix nuances of the derived set cannot be fed without replaying the given lines as environment
actions, which is not built; derived rows are run as plain sequents for the 13 non-prefix nuances
(132 rows). Structural checks (line budget, no DN, ∨E required) are applied to the distinct accepted
proofs; those three nuances report any-of-32 only.

Smoke test before this proposal (s0, 8 v0 problems, k = 4, temperature 1.0): 4 of 8 solved, n_ok
4 / 2 / 3 / 3 of 4; 32 attempts in 14 s wall on CPU.

## Protocol

k = 32, temperature 1.0, seed 0 (sampling), batch 256, environment limits as Dan's defaults (48 steps,
256 tokens per action). Sets: v0 (104), v1 (80), hand14 (14), batch3 (21), lean_v2 (50), derived_plain
(132), textbook72 (72, aggregate only; inputs and outputs in the gitignored
`docs/textbook-problem/textbook72/`). Order: seed 0 over all sets, then seeds 1 and 2.

## Expected result (written before the run)

From the smoke test and Dan's textbook72 pass@256 of 51.7:

| set | expected avg solve rate | expected any-of-32 |
|---|---|---|
| v0 | 0.30–0.50 | 50–70 % |
| v1 | 0.15–0.35 | 35–55 % |
| hand14 | 0.00–0.05 | 0–3 of 14 |
| batch3 | 0.05–0.20 | 20–45 % |
| lean_v2 | 0.25–0.45 | 45–65 % |
| derived_plain | 0.15–0.35 | 35–55 %; forward_control and permutation highest, hand-derived and long nuances lowest |
| textbook72 | 0.25–0.45 | 30–45 of 72 (below the pass@256 figure of 51.7) |
| seed spread | any-of-32 within ±5 problems across seeds on v0 | |

We conclude either way: within the bands, the textbook sets are learnable by RL in the state format
and remain unsaturated (hand14, batch3 and the long tail stay mostly unsolved); above the bands,
check for a leak first, in particular whether Dan's K12 training pool overlaps our bases by renaming
class, before any claim; below the bands, the state-format gains on textbook72 do not carry to the
new sources, which would itself be the finding.

## Budget and stop rule

No money: laptop CPU, ≈ 90 min per seed. Stop if a set exceeds 3 h or Lean fails to launch.

## Launch command

```
cd <scratchpad>/fork_dan && for s in 0 1 2; do for set in v0 v1 hand14 batch3 lean_v2 derived_plain textbook72; do
  LEAN_GATE_LOG=<run>/lean_gate_s$s.jsonl .venv python state_eval.py --ckpt ../dan_ckpts/la_T1_best12_s${s}_r8.pt \
    --in <inputs>/$set.jsonl --out <run>/s$s/$set.jsonl --summary <run>/s$s/$set.summary.json \
    --k 32 --temperature 1.0 --batch 256 --seed 0; done; done
```
(`<run>` = `experiments/textbook-problem/eval_dan_best_state/run_best12_r8`; textbook72 inputs and outputs
redirected to the gitignored textbook72 folder.)
