# Autoresearch: pretrain workstream

You are an autonomous researcher improving the pretraining stage of a small proof model. One
long session, one experiment at a time, forever, until a human stops you. Adapted from
Karpathy's autoresearch (2026) `program.md`; the differences are the metric and the harness split.

## Setup

- `harness.py` is FROZEN. It owns the wall-clock (10 min total from process start), the parameter
  cap (10M), the RL stage (2 rounds of expert iteration, fixed in samples) and the eval. You never
  edit it. Its sha256 is written into every result; a changed harness invalidates the run.
- `pretrain.py` is the ONLY file you edit. Contract in its docstring. Your budget is 240 s of
  wall-clock, compile and model construction included, then the harness takes the model.
- Frozen data and verifier come from Dan's fork at the commit pinned in `remote.sh`.
- The metric is `metric` in `summary.json`: dev-transfer theorems solved at true length >= 7
  with 32 attempts each. Higher is better. The RL stage is what turns pretraining into that
  number, so think about what makes a model a good starting point for expert iteration, not
  only about pretrain loss.

## Loop

1. Read `results/*.json` (all previous experiments, kept and discarded) and `git log`.
2. Pick one idea. Write `results/<NNN>_<slug>.json` BEFORE running with the prediction fields
   filled in (schema below). Committing to a prediction first is the rule, not a suggestion.
3. Edit `pretrain.py`.
4. Run: `source ~/repos/nd-takehome/.envrc.local && python3 pod.py run <NNN>_<slug>`.
   Read `artifacts/autoresearch/<NNN>_<slug>/summary.json` (repo `artifacts/` is gitignored).
5. Fill in the outcome fields in the JSON.
6. Keep if `metric` > current baseline's metric: `git commit` pretrain.py + the JSON. Otherwise
   `git checkout pretrain.py` and commit only the JSON. Ties: keep only if simpler.
7. Go to 1.

## Rules

- **Keep rule.** Strictly greater than the baseline metric. No noise threshold (the metric is
  noisy at roughly +-10%; a keep on noise raises the bar for the next idea, that is accepted).
- **Simplicity.** All else equal, simpler is better. An improvement of ~0 with less code: keep.
  A small improvement with 20 lines of hacks: discard.
- **Crash / timeout counts as a failed experiment.** If the cause is dumb (typo, import), fix
  and re-run under the same NNN. If the idea is what crashed (OOM, overran the budget), log it
  as `crash` or `timeout` and move on. Never edit the harness to make something fit.
- **No cheating.** No eval or RL code in pretrain.py. No touching the pools. No external
  weights, no hand-written proofs, no calling the verifier from the decoder.
- **Serial.** One pod, one experiment at a time.
- **Tags** (multi-select): systems, optimizer, schedule, architecture, tokenizer, data-order,
  augmentation, loss, init, regularization, batch, simplification.
- **NEVER STOP** to ask whether to continue. If out of ideas, re-read the fork's code, combine
  near-misses, try something radical.

## Result JSON schema

```json
{
  "id": "003", "slug": "wider-ffn", "commit": "<7 chars, filled after the run>",
  "title": "one line",
  "description": "two sentences: what changes and why it should help expert iteration",
  "tags": ["architecture"],
  "hypothesis": "what mechanism would make this work",
  "p_keep": 0.35,
  "expected_delta": "+3 to +8 theorems at >=7",
  "baseline_id": "000", "baseline_metric": 38,

  "metric": null, "status": "keep|discard|crash|timeout",
  "pretrain_secs": null, "n_params": null, "heldout_greedy": null, "lstar": null,
  "surprise": "one line: what you did not expect"
}
```
