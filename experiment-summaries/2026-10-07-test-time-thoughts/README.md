---
written_on: 2026-10-08
written_by: agent:claude
---

# Test-time thoughts: recurrence that matters is too slow to win, so far

Status: paused on 2026-10-07 at 20:02 PT after 9 experiments (700 to 708), when the RunPod
balance fell below the loop's $5 floor. About 11.5 hours of a planned 24.

Question (Robbie): does a recurrent, Coconut-style model [@Hao_2024], thinking in continuous
vectors before it writes, solve more natural-deduction proofs when given more thinking at test
time? Sibling of the test-time-loops loop, which asks the same of a weight-shared looped model.
(`Hao_2024` is not in `references.bib` yet.)

Working folders: [the loop](../../code/experiments/current/autoresearch_thoughts/README.md)
(rules in `program.md`, every experiment's registration and result under `results/`, table in
`RESULTS.md`) and [the spike that preceded it](../../code/experiments/current/coconut_spike_20261006/README.md).
Pages: [the six-arm spike](https://claude.ai/artifact/AhXVH4SPaz2mc1CznnNEmm) and
[problems solved by thinking budget, 16 models](https://claude.ai/artifact/1Fc4UtojuEht7UtbtAyuGG)
(private to Robbie).

## Configuration

- Branch `autoresearch-thoughts`. Harness `code/experiments/current/autoresearch_thoughts/harness.py`,
  sha256 prefix `3149b2713a6d`: the loops loop's harness at `bee8868` with `KNOB = "n_thoughts"`
  and `DIAG_LEVEL = 0`. 300 s of pretraining, 1,800 s of recipe-owned RL through a sampling oracle
  (Lean-gate time not charged), dev eval of 1,108 theorems with 64 samples at 8 thoughts per proof
  line (`metric`) and at 0 (`metric_diag`), 10M parameter cap, seeds 0, 1, 2, one A40 each.
- Data and verifier: `dan-pandori/nd-takehome` at `51604b3`.
- Starting recipe (701): run 198's six untied blocks; after every `;` and `by`, n thought positions
  whose input is the model's own output at that token, one parallel pass per line; n drawn from
  0 to 8 per training batch; a training-only plan probe (the token after the line's `:=` or
  `exact`, read off the last thought) at weight 0.3 in pretraining and RL.
- Agent: `claude-fable-5-1` from `~/repos/nd-rl-14` for 702 to 708; 700 and 701 run by the operator
  session from `~/repos/nd-rl-12`.
- Checkpoints: Hugging Face `robbiethompson2018/nd-rl-checkpoints`, `runs/<id>_s<seed>`.

## Observations

Three-seed means, dev theorems solved of 1,108.

| id | recipe | at 8 thoughts | at 0 | outcome |
|---|---|---|---|---|
| 700 | starting recipe | | | crash: out of GPU memory in the RL fine-tune |
| 701 | 700 with the fine-tune batch run in four length-sorted parts | 531 | 479 | keep |
| 702 | RL sampling and fine-tune at 8 thoughts only | 548 | 329 | keep |
| 703 | thoughts chained in two sequential passes | 448 | 253 | discard |
| 704 | pause slots: thought input is a learned vector, one training pass | 676 | 670, 656, 121 by seed | keep |
| 705 | thought input from a detached plain pass | 5 | | crash: no gradient to the blocks (autocast cache, torch 2.9.1) |
| 706 | the loops loop's RL recipe (507) on 704 | 574 (726, 699, 297) | | discard |
| 707 | 705 with `torch.clear_autocast_cache()` | 638 | 589 | discard |
| 708 | 704's sampling with a token-budget fine-tune | 711 | 693 | keep, incumbent |

- Before RL, thinking helps on every pretraining run of the feedback recipe (six runs across 700
  and 701): share of valid samples on the 7 to 14 line RL targets, 8 samples per target, about
  4% at 0 thoughts per line, 8 to 9% at 2, 10 to 11% at 4, 10.5 to 12% at 8.
- After RL, with both paths trained alike (701), thinking is worth 44, 53 and 57 dev theorems on
  the three seeds. In RL sampling the gain is the step from 0 to 2 thoughts (30.0% to 36.7% valid
  after round 1) and nothing from 2 to 8.
- Without gradient through the fed-back state the thoughts go unused: pause slots (704) leave 0
  thoughts within 12 to 24 of 8 on two seeds; detached feedback (707) learned a weight of about
  0.04 on the fed-back state.
- Speed decides the score. Feedback recipes get about 1,550 pretraining steps and 3 RL rounds;
  pause slots get about 3,500 and 6 to 7. 703's extra pass cost its third RL round.
- 702's large gap (548 against 329) comes from never training the 0-thought path, not from the
  thoughts doing more; per-sample accuracy at 8 thoughts was unchanged from 701.
- Control from the loops loop (its 590): run 198, untied, no thoughts, with that loop's RL recipe
  scores 792 (790, 785, 800) in this harness. Its best looped model: 748.
- In the earlier spike (one block of thoughts after the statement, old harness), solved counts
  after RL were flat in the thought count from 0 to 32 in all six arms.

Interpretation, not established: thoughts mattered after RL in 701 either because they sit before
every line or because the plan probe stayed on in RL; no experiment separated the two. 706's
failed seed suggests expert iteration has a take-off the weaker pretrained model does not always
reach in budget; that was observed on one seed.

## Limitations and next step

- One architecture family and one seed set; a keep needs only a higher 3-seed mean, so 702's +17
  and 708's +35 are within or near seed noise (about 10 to 25 per seed).
- The per-line models were scored at 0 and 8 thoughts only. No per-theorem counts were kept at 0,
  and no model was sampled at 1, 2, 4 or 16 thoughts on the dev pool.
- `metric_diag` stops measuring what thoughts buy once RL trains at 8 only (702 onward).
- Next: feedback with gradient at a cost that keeps three or more RL rounds (703's chained thoughts
  with a shorter fine-tune was the agent's queued follow-up), judged against the 792 control with
  the same `rl()`. To resume: top up RunPod, then from `~/repos/nd-rl-14` run
  `code/.venv/bin/python code/experiments/current/autoresearch_thoughts/loop.py --agent fable`.
