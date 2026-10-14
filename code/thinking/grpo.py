"""GRPO for the proof-state thinking model: proof-level reward, failures credited too.

    python -m code.thinking.grpo --init DIR/model.pt --name G_s0 --out GDIR \
        --targets rl_targets.jsonl --budgets 0,16,64,256 --effort --group 16 --prompts 128

Expert iteration (`ei.py`, Dan's ladder) fine-tunes on the Lean-verified attempts only. Here every
attempt counts.

* **Curriculum.** Per iteration, `--prompts` (target, budget) pairs, `--group` attempts each, so a
  group compares attempts made under the same effort. Pairs are drawn by learnability, with weight
  p(1 - p) + `--floor`. Here p is the pair's pass rate, a Beta(1, 1) posterior over its past groups
  whose counts decay by `--decay` each time a new group of the pair is added (a pair that is not
  drawn keeps its counts). Unseen pairs have p = 0.5, so the first pass
  covers everything. Afterwards groups that are always or never solved, which give no gradient,
  are drawn rarely, but the floor brings them back as the policy changes. Plain arm, temperature
  T = `--temperature` (1.0 by default, the usual RLVR choice; evals sample at 0.8).
* **Reward.** 1 if Lean accepts the whole proof, else 0. A success pays `--len_penalty` per 1,000
  sampled tokens (think + action, all steps): the shortness pressure.
* **Length target** (LCPO, L1: Aggarwal & Welleck 2025, arXiv 2503.04697; off by default). With
  `--len_target_coef` alpha > 0 every attempt at a budget B > 0, failures included, also pays
  alpha x d, d = the mean over its think blocks of min(1, d_b), d_b = |L_b - T| / T
  (`--len_target_mode exact`) or max(0, L_b - T) / T (`max`, overshoot only), T =
  `--len_target_frac` x B (`length_deviation`). Each block is clipped before the mean, so one block
  run to the line cap (B + 64: d_b up to 4 at B = 16) cannot saturate the attempt. L_b counts a
  block's think tokens, tags excluded: what the budget forcing compares with B. Budget 0 has no
  term. L1's LCPO-Max is multiplicative (correct x clip(alpha (n_target - n) + 0.5, 0, 1)); `max`
  here is the additive overshoot penalty. A group that is all solved (or all failed) now gives a
  gradient whenever its lengths differ.
* **Overlong masking** (DAPO's overlong filtering, `--overlong mask`). An attempt that a cap ended
  rather than a verdict (`OVERLONG`: max_steps, an action at max_action, the context) gets no
  gradient and is left out of its group's baseline. A think block force-closed at its cap is not
  overlong: the attempt goes on to a verdict. `--overlong fail` scores capped attempts as failures.
* **Advantage.** r minus the mean over the group's unmasked attempts (no division by the group
  std: Dr. GRPO, Liu et al. 2025). Groups with fewer than 2 unmasked attempts, or whose unmasked
  rewards are all equal, give no gradient and are skipped.
* **Loss.** The attempt's advantage times the log-probability of every sampled token of every step,
  at temperature T. That covers the think block and the action, and for a failed attempt the draw
  that ended it. A `</think>` the budget forced was not sampled and is excluded. The loss is a
  token-level mean. Each sample batch is split into `--updates` minibatches, one AdamW step each,
  with the PPO clipped ratio r against the log-probs of the sampling policy. That policy is
  computed once before the first step, so the first minibatch is exactly on-policy. The clip is
  DAPO's clip-higher (Yu et al. 2025, arXiv 2503.14476): min(r A, clip(r, 1 - `--clip_eps_low`,
  1 + `--clip_eps_high`) A), 0.2 / 0.28 by default, so a positive advantage can raise an unlikely
  token further than a symmetric clip allows, against entropy collapse; `--clip_eps` sets both.
  For A < 0 the dual clip (Ye et al. 2020, arXiv 1912.09729) bounds the surrogate below by
  `--dual_clip` x A (3 by default, 0: off), so a token whose r has blown up gives no gradient.
  There is no KL term.
* **Scoring** (2026-10-07). pi_T is the sampler's distribution: inside the think block (its
  `</think>` included) and in the action, the tokens `sample.decode_wave` bans in that phase
  (`sample.phase_bans`: `<eos>`, `<think>`, `<pad>` in the block, effort tokens everywhere) are
  masked and the rest renormalised. Scoring over the full vocabulary instead let a negative
  advantage move think mass onto `<eos>`, which is never sampled and so never penalised (the
  collapsed restart2 run: up to all of the mass at think positions, word-salad think text).
  GRPO samples with the plain arm and no min budget, so those two masks are the whole sampling
  distribution and the old log-probs are recomputed (`old_logps`), not taken from the sampler.
* **Monitoring.** Per iteration: the entropy of the masked pi_T (`entropy`, `entropy_think` /
  `entropy_action`), the full-vocabulary pi_T mass on the think ban at the scored think tokens
  (`ban_think`, `ban_think_max`), the largest r (`ratio_max`), the share of the A < 0 tokens with
  r > 3 (`neg_r3`), the clip fractions below / above, the grad norm (mean and max over the
  minibatch steps), the think tokens per attempt by budget, and per budget the think tokens per
  block, the share of blocks force-closed and the mean d (`length_metrics`, also with the length
  target off), and the mean penalty alpha x d. An `ALARM` line (`alarms`) flags a think ban mass
  above 1e-3, a grad-norm max above 5x the median of the earlier iterations',
  clip-low above 1%, an overlong share above 5%, or a scored token banned in its phase, and the
  iteration's checkpoint is saved. A positive loss is not an alarm: the token-level mean is
  positive whenever failed attempts run longer than successes.

Safeguards (collapse audit 2026-10-08, `experiments/.../collapse_audit/FIXES.md`; the seed-0 run
drifted into a long-context `:` attractor the RL pool never reaches). Every one is opt-in except
the two alarms, which only add alarm lines and alarm checkpoints; the updates are unchanged.
* **Capped attempts** (always logged). `capped_frac` / `capped_by_budget` / `capped_kinds`: the
  share of attempts a cap ended (`OVERLONG`), whatever `--overlong` does with them
  (`overlong_frac` counts the masked ones, so it is 0 under `--overlong fail`). Alarm when a
  budget's share exceeds `--alarm_capped` (0.05; 0: off).
* **Relative clip-low.** Alarm when clip-low exceeds `--clip_low_rel` (2.0; 0: off) x its median
  over the last `ALARM_REL_WINDOW` stepping iterations (`clip_low_median`), once `ALARM_GN_AFTER`
  are in, and `ALARM_CLIP_LOW_FLOOR`. The absolute 1% alarm stays.
* **Long-context canary** (`--canary FILE`, `canary.py`): every `--canary_every` iterations, after
  the update, `--canary_n` teacher-forced reference steps of long proofs (states with >= 10
  hypothesis lines, B0 layout, T = 1): the action NLL, P(':') at non-':' targets and the largest
  mean probability any single wrong token gets (`row['canary']`). Its value before the first
  iteration is the base (`canary_base.json`); alarm when one of the three exceeds `--canary_mult`
  (5) x its base.
* **Rollback** (`--rollback_on_alarm`): on any alarm, the alarm checkpoint is saved, then the
  newest good checkpoint (`LoopState.last_good`: the last rolling or permanent one saved in an
  iteration without alarms, or after a rollback, or one the restore check let through; a start
  checkpoint `<name>_it<s>_rolling.pt` is written first) is restored -- weights, optimizer,
  curricula, RNGs, step count, alarm histories -- the lr is halved (`lr_scale`) and the run
  continues with the next iteration number; at most `--max_rollbacks` times (`row['rollback']`).
* **Restore check** (`--ban_restore_check`, with `--rollback_on_alarm`; Leon 2026-10-11): when an
  iteration's only alarm is the think ban mass and a rollback is due, its training items are
  first scored under the restore point's weights as `policy_loss` scores them
  (`restore_ban_mass`, on a scratch copy of the model). If that mass is over `ALARM_BAN` too and
  the iteration's own is under `BAN_SPIKE` (2) times it, restoring would only halve the lr (seed-1
  L3 runs sat at the limit from their start: three rollbacks to it 0 left them at lr x0.125). Then
  there is no rollback; the alarm is still logged, and the iteration is checkpointed as one without
  alarms (no alarm checkpoint; its rolling or permanent checkpoint is good). A spike (2 times or
  more) or a restore point under the limit rolls back as before. `row['ban_check']`: the restore
  point, both masses and the choice.
* **Think ban stop** (`--ban_stop` X; 0: off): an iteration whose think ban mass exceeds X writes
  its row and its alarm checkpoint and ends the run with exit code `STOP_EXIT` (3), before any
  rollback or restore check (`row['stop']`). Both options are in `row['alarm_cfg']`.
* **Training state in checkpoints.** Every checkpoint this script writes carries `train_state`
  (`ckpt.save_ckpt`, a top-level key that model loaders ignore): AdamW's state, the iteration, the
  curricula, the RNGs and `LoopState`. `--resume CKPT` restores all of it and continues to
  iteration `--start_iter` + `--iters` (re-run the same command with `--resume` added); rows of
  later iterations already in metrics.jsonl move to metrics.pre_resume.jsonl. `--init` stays a
  weight-only start (fresh optimizer, warmup and curriculum); `--start_iter S` numbers its
  iterations S + 1 .. S + `--iters` (sampling seeds and checkpoint names follow).
* **KL anchor** (`--kl_anchor_coef` beta, `--kl_anchor_ref PATH`): at every AdamW step, beta x
  KL(pi_ref || pi_theta) on `--kl_anchor_n` teacher-forced long-state steps (`canary.anchor_kl`;
  from `--kl_anchor_data`, default the canary file minus the canary's own proofs), over the action
  tokens, on the sampler's masked distribution at `--temperature`; pi_ref frozen from PATH.
  Logged: `kl_anchor` (mean over the steps) and `grad_norm_policy` (before the anchor's gradient).
* **Batch dumps** (`--dump_batches DIR`, `grpo_io.py`): per iteration the full training batch and
  the AdamW state before its first step, for update attribution.
* **Long proofs in the pool** (`--targets_extra FILE --targets_extra_frac f`): round(f x
  `--prompts`) groups per iteration come from a second target file, with its own curriculum;
  logged as `extra_*`.

Learning-rate schedule (opt-in, `--lr_schedule`; Leon 2026-10-13). The default, constant, is the
earlier code's lr: `--lr` x the rollback scale `lr_scale`, after a linear ramp over the first
`--warmup` (10) AdamW steps of the run (each stage's `--init` starts a fresh AdamW and step count:
2.5 iterations at `--updates` 4). `cosine` and `wsd` multiply it by a factor per iteration: the
run's iteration t = 1 .. T (T = `--iters`, t = the iteration's number minus `--start_iter`) trains
at `--lr` x m(t) x `lr_scale` x the step ramp, m = `lr_mult`. With `--warmup_iters` W > 0 the
warmup is m's, t / W while t < W, and the step ramp is off; with W = 0 (the default) the step ramp
stays, as in the constant schedule. After the warmup,
* **cosine**: m from 1 at t = W to f = `--min_lr_frac` at t = T (EI's fine-tune schedule,
  `train.finetune`, over iterations instead of steps);
* **wsd** (warmup-stable-decay): m = 1 until the last D = round(`--decay_frac` x T) iterations (at
  least 1), then down to f at t = T, linearly, along a half cosine, or as 1 - sqrt
  (`--decay_shape`; the 1-sqrt cooldown of Hägele et al. 2024, arXiv 2405.18392). Until the decay
  its lr is the constant schedule's bit for bit.
A rollback halves `lr_scale` from the next iteration, as before, on top of the schedule. It restores
the training state (the step count too) but not the iteration number, so the schedule goes on from
the alarming iteration's number and the run still ends at m(T) (f, unless the warmup covers it). A
resume continues the same schedule and refuses a checkpoint whose run had another one. Logged:
`lr` (the iteration's last step's; its next step's when it takes none), `lr_mult`, `lr_ramp`,
`alarm_cfg['lr_schedule']` and the guard line's `lr L = base x kind m x scale [x ramp r]`. The
options are left out of args.json unless given.

Verifier-guided sampling and the step penalty (opt-in; Leon 2026-10-09, "GRPO + verifier"):
* **`--arm`** is the sampler's arm (`sample.think_generate`): `plain` (the default) ends an
  attempt at its first rejected draw; `structural` redraws a draw the environment rejects (or a
  truncated action); `logical` also redraws a step `step_check` rejects. `step_check` is sound with
  respect to Lean (it rejects only steps Lean is certain to reject: Dan's guided-tts validation,
  0 false rejects in 314,541 (state, step) pairs). A redraw is the whole think + action from the
  same state, without replacement over that decoded segment, at most `--max_rej` (10) per attempt
  (then `fail:max_rej`). Every decoded token counts as sampled compute (`tokens`), rejected draws
  included. Scoring: the old log-probs are pi_T under the phase masks, as for the plain arm; the
  without-replacement exclusion of a redraw is not modelled, so a redrawn step is scored as if pi_T
  had drawn it (its first-minibatch ratio is 1, a small bias toward accepted redraws).
* **`--rej_penalty` P** (0: off): every rejected draw -- each redrawn one (`keep_rejected`; by
  default its tokens from the first one where it departs from the draw accepted at the same
  state, since the shared prefix led to a valid step too; `--rej_penalty_scope draft` the whole
  think + action, `action` the action only) and the draw that
  ended a failed attempt -- gets advantage -P whatever its group's outcome, so failures give a
  gradient even where every attempt failed. Accepted steps keep the group's advantage. A group is
  skipped only if its outcomes give no signal and it has no rejected draw (`group_items`). The
  penalised draws' ratios fall below 1 - eps on purpose, so with the penalty the clip-low alarms
  (absolute and relative) read the accepted items' clip-low (`clip_low_accepted`, `clip_low_basis`).
  The loss is then normalised by the accepted items' tokens, not every scored token, so an accepted
  token gets plain GRPO's per-token weight and the penalty comes on top (`tokens_accepted_share`
  logs the normaliser's share of the scored tokens).
* **`--monitor`**: per iteration, where the gradient comes from (`row['monitor']`,
  `monitor_metrics`): per target length and per group pass-rate band, the groups, the share with
  a gradient, the pass rate and the share of the |A|-weighted token mass, its think and
  rejected-draw parts; overall the think / action and accepted / rejected split, and rejections per
  draw with their top causes. `matched_eval/rl_dashboard.py` plots these rows.

Pilot options (opt-in; Leon 2026-10-13: late in stage L3 about half the groups were all solved and
skipped). Without them the rollouts, rows, log lines, weights and args.json are c4b8548's
(digests in test_grpo_pilots.py); the options are left out of args.json unless given.
* **Refill** (`--refill`; dynamic sampling, DAPO: Yu et al. 2025, arXiv 2503.14476; Olmo 3's active
  sampling). A group has an outcome signal when its unmasked rewards are not all equal (not all
  solved, not all failed). While fewer than `--prompts` of the iteration's groups have one and
  fewer than `--refill_max` (2) extra rounds have run, another round of pairs is drawn with the same
  curricula and weights (updated with the groups so far; pairs already drawn this iteration
  excluded; `--targets_extra_frac` of them from the extra file) and sampled: ceil(D / q) pairs, at
  most `--prompts`, for a deficit D and the iteration's share q of groups with a signal so far.
  Groups are taken in draw order until `--prompts` with a signal are in; later ones (the surplus)
  update the curriculum and the sampling metrics but are not trained on. Every sampled group
  updates its curriculum and counts in the rows' sampling metrics (pass@1, tokens, capped, the
  monitor; `groups_skipped` is over every sampled group), and every sampled attempt counts as
  compute. `row['refill']`: per round the pairs drawn, those with a signal and those never drawn
  before; `skip_before` (round 0's share without a signal) and `skip_after` (the share of the
  `--prompts` target left without one); the surplus; this iteration's sampled groups, attempts
  and tokens (`tokens` as the sampler counts them, rejected draws included) and the run's totals
  (`*_cum`: rolled-back iterations included, carried through a resume in `train_state`).
* **Pass-rate filter** (`--max_pass` x, 0.5 <= x <= 1): pairs whose estimate p is above x are not
  drawn, in any round. Olmo 3 drops prompts its SFT model solves more than 62.5% of the time,
  offline; here it is the running estimate, so a pair once over x is not drawn again (no floor).
  Unseen pairs (p = 0.5) are never filtered. `row['max_pass']`: the pairs over x per curriculum.
* **Unlikeliness reward** (`--unlikely_beta` beta, 0 <= beta <= 1; He, Fried & Welleck 2025,
  arXiv 2506.02355, Sec. 4.1, beta_rank = 0.25 there). In a group with an outcome signal the n
  unmasked attempts are ranked by their likelihood under the sampling policy pi_old (the weights
  before this iteration's update): rho_j = the number of attempts less likely than j, ties
  counting half, so 0 is the least likely and n - 1 the most likely. A solved attempt's reward 1
  becomes 1 - beta rho_j / n; failures keep 0. This is the paper's r_i = R(x, y_i) (1 - beta_rank
  (G - rank(y_i)) / G) with rank 1 the most likely of the G samples (its text also says "rank 0";
  the released code, `ray_lean_trainer._compute_advantages`, uses rank = argsort(argsort(log p))
  ascending / G, which is rho / n). Only groups with an outcome signal change: the paper skips
  groups with zero advantage before the perturbation, so an all-solved group stays skipped and
  the outcome still sets each attempt's sign. The advantage stays r - the group mean (Dr. GRPO;
  the paper divides by the std). The likelihood is the mean log pi_old per scored token of the
  attempt's steps (`old_logps`, the PPO ratio's log-probs, so no extra forward pass): every
  sampled think and action token, under the phase masks at T, the draw that ended a failed attempt
  included, a forced `</think>` excluded. The released code also takes the per-token mean; the sum
  would rank mostly by length (a longer proof or think block is less likely) and so reward long
  proofs. Redrawn draws (guided arms) are not part of the attempt and do not count; non-finite
  log-probs (a token banned in its phase: the `nonfinite` alarm) are left out. With `--rej_penalty`
  the rejected draws keep -P on top of their attempt's advantage; with `--overlong fail` a capped
  attempt is a failure (reward 0) and is ranked like the others; with `--overlong mask` the masked
  attempts are out of the ranking, as they are out of the baseline. `--len_penalty` and
  `--len_target_coef` stay additive (a success's 1 - beta rho / n pays them as its 1 did).
  `row['unlikely']`: the groups and solved attempts ranked, the mean and largest deduction, and
  the mean within-group correlation of rho with the attempt's scored tokens (a length bias shows
  there).
* **Unseen-pair prior** (`--unseen_prior`; Leon 2026-10-13). The pass-rate estimate is already a
  shrunk one: the Beta(1, 1) posterior mean (s + 1) / (n + 2) pulls a pair's decayed rate s / n
  toward 1/2 with weight 2 / (n + 2), and a pair never drawn sits at 1/2, the largest weight.
  L3's 15,000 pairs outnumber a stage's 9,984 draws, so in the v2 L3 runs unseen pairs were
  81-86% of the draws while the seen pairs' mean p was about 0.87 (0.75 at B0). `half` (the
  default) keeps that. `mean` makes the prior Beta(2 m_b, 2 (1 - m_b)), the same strength 2 with
  mean m_b: p = (s + 2 m_b) / (n + 2), so an unseen pair is at m_b and a seen one is pulled toward
  m_b instead of 1/2 (the shrinkage stays as consistent as before). m_b is the mean over budget
  b's seen pairs of their decayed rate s / n, unshrunk, per curriculum (the extra targets have
  their own), recomputed at every draw; until PRIOR_MIN_SEEN (32) of the budget's pairs are seen
  it is 1/2 (in practice only before the first update). Budget, not target, is the grouping: the
  budget is the curriculum's summary grouping and its means differ (B0 lower); a pair's
  same-target pairs at other budgets correlate with its rate (r 0.36-0.41 at the end of v2 L3)
  but predict it worse than the budget mean (squared error 0.043-0.053 vs 0.035-0.043). The
  `--max_pass` filter applies to seen pairs only (the same under `half`, where an unseen pair is
  at 1/2 <= x). `row['prior']` (whenever the option is given, `half` too): the pairs drawn this
  iteration and how many were never drawn before (every --refill round), and at its first draw
  the prior per budget (`fallback`: still at 1/2), the seen / unseen pairs, their mean weight and
  the unseen pairs' share of the total weight; `extra` for the extra curriculum.

Budget tokens (`--effort`): a `lean_thinkn` init is expanded to `lean_thinke` (effort.py), and the
sampler prefills `<eB>`. Without `--effort` the budget is applied but never shown to the model (as
in ei.py). `--force_mode` is the sampler's budget forcing: 'line' (default) lets the block finish
its line past B, up to B + 64; 'exact' closes it at B. Either way the forced `</think>` is masked.

Files in GDIR: metrics.jsonl (one row per iteration), ckpts/<name>_it<i>.pt every --save_every
iterations and at the end, <name>_it<i>_alarm.pt at each alarming iteration and
<name>_it<i>_rolling.pt every --save_rolling iterations (the last --keep_rolling kept), args.json,
registry/, lean_gate.jsonl; with --canary, canary_base.json; with --dump_batches DIR,
DIR/<name>_it<i>_batch.pt.
"""

from __future__ import annotations

import argparse
import collections
import copy
import dataclasses
import json
import math
import random
import re
import statistics
import sys
import time
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch

from . import stateview
from .canary import anchor_kl, canary_metrics, load_canary
from .ckpt import TRAIN_STATE, load_ckpt, load_train_state, run_env, save_ckpt
from .dan import bare, load
from .data import read_jsonl
from .effort import LEVELS, expand_to_effort, has_effort
from .grpo_io import compact_adam, dump_batch
from .sample import ARMS, FORCE_MODES, judge, phase_bans, sample_format, think_generate
from .tok import has_think

D = load()
Item = tuple[list[int], list[int], list[bool], float]  # prompt, generated, loss mask, advantage
Bans = tuple[torch.Tensor, torch.Tensor]  # sample.phase_bans: [V] bool, think block / action
R_WATCH = 3.0  # neg_r3: the share of the scored A < 0 tokens with r above this
# per-iteration alarms (`alarms`)
ALARM_BAN = 1e-3  # mean full-vocabulary mass on the think ban at the scored think tokens
# --ban_restore_check: a think ban alarm still rolls back when the iteration's mass is at least this
# x the restore point's on the same tokens, a spike (restoring does help then)
BAN_SPIKE = 2.0
STOP_EXIT = 3  # --ban_stop: the exit code of a run it ends
ALARM_GN = 5.0  # grad-norm max vs the median of the earlier iterations' grad-norm max ...
ALARM_GN_AFTER = 5  # ... once this many iterations are in
ALARM_CLIP_LOW = 0.01
ALARM_OVERLONG = 0.05
ALARM_CAPPED = 0.05  # --alarm_capped: a budget's share of attempts a cap ended
ALARM_CLIP_LOW_REL = 2.0  # --clip_low_rel: clip-low vs its running median ...
ALARM_REL_WINDOW = 20  # ... over the last this many stepping iterations (ALARM_GN_AFTER first)
ALARM_CLIP_LOW_FLOOR = 1e-3  # ... and above this, so a near-zero median does not fire on noise
CANARY_MULT = 5.0  # --canary_mult: a canary value vs its base
CANARY_KEYS = (("nll", "action NLL"), ("wrong_max", "top wrong-token P"), ("colon", "P(':')"))
STATE_VERSION = 1  # `train_state`
LEN_TARGET_MODES = ("exact", "max")  # --len_target_mode: |L_b - B| / B, or the overshoot only


def think_blocks(x: dict[str, Any], think_ids: tuple[int, int] | None) -> list[tuple[int, bool]]:
    """An attempt's think blocks, in trace order -> (L_b, force-closed?) each. A block is a step
    whose decoder input ends with an open `<think>` (`think_len`'s test), so budget 0 (`<think>
    </think>` prefilled) and R's action-only steps have none; the failing draw's block counts.
    L_b = the generated tokens before its `</think>`, tags excluded: the count
    `sample.decode_wave` compares with B to force the close (under line forcing the tokens past B
    are think tokens too). [] without `think_ids` or a trace."""
    if think_ids is None:
        return []
    out: list[tuple[int, bool]] = []
    for s in x.get("trace") or []:
        p, g = s.get("prompt") or b"", s.get("gen") or b""
        if p and p[-1] == think_ids[0]:
            n = g.index(think_ids[1]) if think_ids[1] in g else len(g)
            out.append((n, bool(s.get("forced"))))
    return out


def length_deviation(
    blocks: Sequence[int], budget: int, mode: str = "exact", frac: float = 1.0
) -> float | None:
    """LCPO's length term: the mean over the attempt's blocks of min(1, d_b), d_b = |L_b - T| / T
    ('exact') or max(0, L_b - T) / T ('max'), target T = frac x B. Each d_b is clipped before the
    mean. None (no term) at B = 0 or without a block."""
    assert mode in LEN_TARGET_MODES, mode
    assert frac > 0, frac
    if budget <= 0 or not blocks:
        return None
    t = frac * budget
    dev = (abs(n - t) if mode == "exact" else max(0.0, n - t) for n in blocks)
    return sum(min(1.0, x / t) for x in dev) / len(blocks)


def rewards(
    att: Sequence[dict[str, Any]],
    len_penalty: float,
    len_target_coef: float = 0.0,
    len_target_mode: str = "exact",
    think_ids: tuple[int, int] | None = None,
    len_target_frac: float = 1.0,
) -> list[float]:
    """1 per Lean-accepted attempt minus len_penalty per 1,000 sampled tokens; 0 per failure. With
    `len_target_coef` alpha > 0, every attempt (failures too, as in L1) also pays alpha x
    `length_deviation` of its think blocks (`think_blocks`) from its target, `len_target_frac` x
    its budget; none at B = 0."""
    r = [(1.0 - len_penalty * x["tokens"] / 1000.0) if x.get("ok") else 0.0 for x in att]
    if not len_target_coef:
        return r
    for i, x in enumerate(att):
        blocks = [n for n, _ in think_blocks(x, think_ids)]
        d = length_deviation(blocks, int(x.get("budget") or 0), len_target_mode, len_target_frac)
        if d is not None:
            r[i] -= len_target_coef * d
    return r


def length_metrics(
    groups: Sequence[Sequence[dict[str, Any]]],
    budgets: Sequence[int],
    think_ids: tuple[int, int] | None,
    mode: str = "exact",
    coef: float = 0.0,
    frac: float = 1.0,
) -> dict[str, Any]:
    """One iteration's think-length rows, per budget B > 0 with blocks (keys str(B)): the mean
    think tokens per block (`think_block_by_budget`) against the target T = frac x B
    (`len_target_by_budget`), the share of blocks
    force-closed (`forced_by_budget`; `forced_frac` over every block) and the mean
    `length_deviation` d per attempt (`len_dev_by_budget`; `len_dev` over the B > 0 attempts), in
    `mode` whatever `coef` is; `len_target_penalty` = coef x len_dev, the mean reward deduction of
    an attempt that pays it."""
    blk: dict[int, list[int]] = collections.defaultdict(list)
    frc: dict[int, list[bool]] = collections.defaultdict(list)
    dev: dict[int, list[float]] = collections.defaultdict(list)
    for att, b in zip(groups, budgets, strict=True):
        for x in att:
            bl = think_blocks(x, think_ids)
            blk[b] += [n for n, _ in bl]
            frc[b] += [f for _, f in bl]
            d = length_deviation([n for n, _ in bl], b, mode, frac)
            if d is not None:
                dev[b].append(d)
    n_blk = sum(len(v) for v in frc.values())
    n_dev = sum(len(v) for v in dev.values())
    len_dev = sum(map(sum, dev.values())) / n_dev if n_dev else 0.0
    return {
        "think_block_by_budget": {str(b): sum(v) / len(v) for b, v in sorted(blk.items()) if v},
        "len_target_by_budget": {str(b): frac * b for b, v in sorted(blk.items()) if v},
        "forced_by_budget": {str(b): sum(v) / len(v) for b, v in sorted(frc.items()) if v},
        "forced_frac": sum(map(sum, frc.values())) / n_blk if n_blk else 0.0,
        "len_dev_by_budget": {str(b): sum(v) / len(v) for b, v in sorted(dev.items())},
        "len_dev": len_dev,
        "len_target_penalty": coef * len_dev,
    }


# statuses (sample.think_generate) of attempts that a cap ended, not a verdict
OVERLONG = ("fail:step_cap", "fail:truncated", "fail:context")


def is_overlong(x: dict[str, Any]) -> bool:
    return x.get("status") in OVERLONG


def capped_metrics(
    groups: Sequence[Sequence[dict[str, Any]]], budgets: Sequence[int]
) -> dict[str, Any]:
    """The share of attempts a cap ended (`is_overlong`), whatever `--overlong` does with them:
    over all attempts, per budget (keys str(B)), and the count per cap (`OVERLONG` without
    'fail:')."""
    by_b: dict[int, list[float]] = collections.defaultdict(list)
    kinds = dict.fromkeys((s.split(":", 1)[1] for s in OVERLONG), 0)
    for att, b in zip(groups, budgets, strict=True):
        for x in att:
            c = is_overlong(x)
            by_b[b].append(float(c))
            if c:
                kinds[x["status"].split(":", 1)[1]] += 1
    n = sum(len(v) for v in by_b.values())
    return {
        "capped_frac": sum(map(sum, by_b.values())) / max(1, n),
        "capped_by_budget": {str(b): sum(v) / len(v) for b, v in sorted(by_b.items())},
        "capped_kinds": kinds,
    }


def advantages(r: Sequence[float], keep: Sequence[bool] | None = None) -> list[float] | None:
    """r - the mean over the kept attempts, 0 for the others (no gradient); None when fewer than 2
    are kept or every kept reward is equal (no signal)."""
    keep = [True] * len(r) if keep is None else list(keep)
    kr = [x for x, k in zip(r, keep, strict=True) if k]
    if len(kr) < 2 or max(kr) - min(kr) < 1e-9:
        return None
    m = sum(kr) / len(kr)
    return [x - m if k else 0.0 for x, k in zip(r, keep, strict=True)]


def _step_mask(s: dict[str, Any], tok: Any) -> list[bool]:
    """A draw's loss mask: every sampled token but a forced `</think>`."""
    gen = s["gen"]
    mask = [True] * len(gen)
    if s.get("forced") and tok.end_think in gen:
        mask[list(gen).index(tok.end_think)] = False
    return mask


def attempt_items(x: dict[str, Any], adv: float, tok: Any, rej_extra: float = 0.0) -> list[Item]:
    """Every step of an attempt -> (prompt, generated ids, loss mask, advantage). A forced
    `</think>` is masked out. `rej_extra` is added to the advantage of the draw that ended a
    failed attempt (`rejected` in the trace): the step penalty, -P."""
    out: list[Item] = []
    for s in x.get("trace") or []:
        if "gen" not in s:
            continue
        gen = list(s["gen"])
        if gen:
            a = adv + (rej_extra if s.get("rejected") else 0.0)
            out.append((list(s["prompt"]), gen, _step_mask(s, tok), a))
    return out


# --rej_penalty_scope: what a redrawn draw's -P scores. 'diverge': its tokens from the first one
# where it differs from the draw accepted at the same state (the shared prefix led to a valid step
# too; with no accepted draw there, the whole draw); 'draft': the whole think + action; 'action':
# the action only.
REJ_SCOPES = ("diverge", "draft", "action")


def _lcp(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for u, v in zip(a, b, strict=False):
        if u != v:
            break
        n += 1
    return n


def rejected_items(
    x: dict[str, Any], penalty: float, tok: Any, scope: str = "diverge"
) -> list[Item]:
    """The draws a guided arm rejected and redrew (`sample.think_generate(keep_rejected=True)`'s
    `redrawn`) -> items with advantage -penalty, whatever the attempt's outcome, scored as
    `scope` says (`REJ_SCOPES`). A redrawn draw's `step` is the number of steps accepted before
    it, so the draw accepted at its state is the attempt's accepted trace entry of that index."""
    assert scope in REJ_SCOPES, scope
    if penalty <= 0:
        return []
    think_ids = (tok.think, tok.end_think) if has_think(tok) else None
    acc = [s for s in x.get("trace") or [] if "gen" in s and not s.get("rejected")]
    out: list[Item] = []
    for s in x.get("redrawn") or []:
        p, gen = list(s["prompt"]), list(s["gen"])
        if not gen:
            continue
        mask = _step_mask(s, tok)
        if scope == "action":
            n = think_len(p, gen, think_ids)
            mask = [m and k >= n for k, m in enumerate(mask)]
        elif scope == "diverge":
            t = s.get("step")
            if isinstance(t, int) and 0 <= t < len(acc) and list(acc[t]["prompt"]) == p:
                n = _lcp(gen, acc[t]["gen"])
                mask = [m and k >= n for k, m in enumerate(mask)]
        if any(mask):
            out.append((p, gen, mask, -penalty))
    return out


def n_rejected(x: dict[str, Any]) -> int:
    """Rejected draws of an attempt: the redrawn ones and the one that ended it."""
    end = sum(bool(s.get("rejected")) for s in x.get("trace") or [] if "gen" in s)
    return len(x.get("redrawn") or []) + end


def group_items(
    att: Sequence[dict[str, Any]],
    tok: Any,
    len_penalty: float,
    overlong: str = "mask",
    len_target_coef: float = 0.0,
    len_target_mode: str = "exact",
    len_target_frac: float = 1.0,
    rej_penalty: float = 0.0,
    rej_scope: str = "diverge",
    src: list[tuple[int, str]] | None = None,
    signal: list[bool] | None = None,
    unlikely: Sequence[float] | None = None,
) -> tuple[list[Item] | None, list[bool]]:
    """One group's training items -> (items, None when the group is skipped; per attempt, whether
    it is masked as overlong). Masked attempts give no items, so no gradient. `len_target_*`: the
    length target (`rewards`). `signal`, when given, gets whether the outcomes give a signal (the
    unmasked rewards are not all equal). `unlikely`: per attempt a deduction from its reward (the
    unlikeliness reward, `unlikely_penalties`), applied only when the outcomes give a signal.

    Step penalty (`rej_penalty` P > 0, a guided arm): every rejected draw gets -P on top of what
    its attempt gives it -- the redrawn draws (`rejected_items`, of every attempt, masked ones
    included: a rejected draw is a known-invalid step whatever the outcome) and the draw that
    ended a failed attempt. A group is then skipped only if its outcomes give no signal AND it
    has no rejected draw; a no-signal group with rejected draws gives only the penalised draws
    (and the attempt-ending ones). `src`, when given, gets one (attempt index, kind) per item:
    'acc' an attempt's steps, 'rej' a redrawn draw."""
    assert overlong in ("mask", "fail"), overlong
    masked = [overlong == "mask" and is_overlong(x) for x in att]
    ids = (tok.think, tok.end_think) if len_target_coef and has_think(tok) else None
    r = rewards(att, len_penalty, len_target_coef, len_target_mode, ids, len_target_frac)
    keep = [not m for m in masked]
    adv = advantages(r, keep)
    if signal is not None:
        signal.append(adv is not None)
    if adv is not None and unlikely is not None:  # the skip rule reads the rewards before it
        adv2 = advantages([x - u for x, u in zip(r, unlikely, strict=True)], keep)
        adv = adv if adv2 is None else adv2
    pen = rej_penalty > 0 and any(n_rejected(x) for x in att)
    if adv is None and not pen:
        return None, masked
    items: list[Item] = []
    for j, (x, mk) in enumerate(zip(att, masked, strict=True)):
        if not mk:
            if adv is not None:
                its = attempt_items(x, adv[j], tok, -rej_penalty if pen else 0.0)
            else:  # no outcome signal: only the draw that ended a failed attempt, penalised
                its = [it for it in attempt_items(x, 0.0, tok, -rej_penalty) if it[3] != 0.0]
            items += its
            if src is not None:
                src += [(j, "acc")] * len(its)
        if pen:
            rj = rejected_items(x, rej_penalty, tok, rej_scope)
            items += rj
            if src is not None:
                src += [(j, "rej")] * len(rj)
    return items, masked


def frac_ranks(v: Sequence[float]) -> list[float]:
    """Ascending ranks from 0, ties averaged: rho_i = #{k: v_k < v_i} + #{k != i: v_k = v_i} / 2."""
    return [sum(y < x for y in v) + (sum(y == x for y in v) - 1) / 2 for x in v]


def attempt_logp(
    n_att: int, src: Sequence[tuple[int, str]], old: Sequence[torch.Tensor]
) -> list[float | None]:
    """A group's attempts -> the mean log pi_old per scored token of each (the unlikeliness
    reward's likelihood), from its items' `old_logps` (`old`, aligned with `src`, `group_items`'s
    (attempt, kind) per item): the attempt's own steps ('acc'), not its redrawn draws; non-finite
    log-probs left out. None for an attempt without a scored token."""
    tot, cnt = [0.0] * n_att, [0] * n_att
    for (j, kind), o in zip(src, old, strict=True):
        if kind == "acc":
            f = o[torch.isfinite(o)]
            tot[j] += float(f.sum())
            cnt[j] += int(f.numel())
    return [t / c if c else None for t, c in zip(tot, cnt, strict=True)]


def unlikely_penalties(
    att: Sequence[dict[str, Any]],
    scores: Sequence[float | None],
    masked: Sequence[bool],
    beta: float,
) -> list[float]:
    """The unlikeliness reward (module docstring; He, Fried & Welleck 2025, Sec. 4.1) -> per
    attempt the deduction from its reward: beta x rho / n for a solved attempt, rho its rank among
    the n unmasked attempts with a score (`attempt_logp`), ascending in likelihood (`frac_ranks`: 0
    the least likely); 0 for the others."""
    ranked = [
        (j, s)
        for j, (s, m) in enumerate(zip(scores, masked, strict=True))
        if s is not None and not m
    ]
    out = [0.0] * len(att)
    rho = frac_ranks([s for _, s in ranked])
    for (j, _), r in zip(ranked, rho, strict=True):
        if att[j].get("ok"):
            out[j] = beta * r / len(ranked)
    return out


GroupFn = Callable[..., tuple[list[Item] | None, list[bool]]]


def unlikely_rewrite(
    per: Sequence[Sequence[dict[str, Any]]],
    sig: Sequence[bool],
    spans: Sequence[tuple[int, int, int, int, list[bool]]],
    items: list[Item],
    item_src: Sequence[tuple[int, int, str]],
    old: Sequence[torch.Tensor],
    beta: float,
    grp: GroupFn,
) -> dict[str, Any]:
    """--unlikely_beta: the items of every trained group with an outcome signal get the
    unlikeliness reward's advantages, in place (`grp`: `group_items` with the run's settings,
    called with the deductions). The items' tokens, masks and order stay, so `old` (their
    `old_logps`, which the ranks read) still aligns. `spans`: per trained group (group, first item,
    one past its last, dump row, masked). -> row['unlikely']: the groups and solved attempts
    ranked, the mean and largest deduction, the mean within-group Pearson correlation of rho with
    the attempt's scored tokens (`len_corr`, over `len_corr_groups` groups with 3 or more ranked
    attempts); plus 'spans' and 'pens', the rewritten groups' spans and per-attempt deductions."""
    done: list[tuple[int, int, int, int, list[bool]]] = []
    pens: list[list[float]] = []
    solved: list[float] = []
    corrs: list[float] = []
    for sp in spans:
        g, i0, i1, _, masked = sp
        if not sig[g]:
            continue
        att = per[g]
        sg = [(j, kind) for _, j, kind in item_src[i0:i1]]
        sc = attempt_logp(len(att), sg, old[i0:i1])
        pen = unlikely_penalties(att, sc, masked, beta)
        src2: list[tuple[int, str]] = []
        its, _ = grp(att, src2, None, pen)
        assert (
            its is not None
            and src2 == sg
            and all(x[:3] == y[:3] for x, y in zip(its, items[i0:i1], strict=True))
        ), "the unlikeliness reward changed a group's items, not only their advantages"
        items[i0:i1] = its
        done.append(sp)
        pens.append(pen)
        ranked = [(j, s) for j, s in enumerate(sc) if s is not None and not masked[j]]
        solved += [pen[j] for j, _ in ranked if att[j].get("ok")]
        n_tok = [0] * len(att)
        for (j, kind), x in zip(sg, its, strict=True):
            if kind == "acc":
                n_tok[j] += sum(x[2])
        c = _corr(frac_ranks([s for _, s in ranked]), [float(n_tok[j]) for j, _ in ranked])
        if c is not None:
            corrs.append(c)
    return {
        "beta": beta,
        "groups": len(done),
        "solved": len(solved),
        "pen_mean": sum(solved) / len(solved) if solved else 0.0,
        "pen_max": max(solved, default=0.0),
        "len_corr": sum(corrs) / len(corrs) if corrs else None,
        "len_corr_groups": len(corrs),
        "spans": done,
        "pens": pens,
    }


def _corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson's r; None for fewer than 3 points or a constant side."""
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((u - mx) * (v - my) for u, v in zip(x, y, strict=True))
    sxx, syy = sum((u - mx) ** 2 for u in x), sum((v - my) ** 2 for v in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


def _chunks(items: Sequence[Item], max_tokens: int) -> list[list[int]]:
    """Length-sorted micro-batches of at most max_tokens padded tokens."""
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]))
    chunks: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for idx in order:
        L = len(items[idx][0]) + len(items[idx][1])
        if cur and (len(cur) + 1) * max(cur_max, L) > max_tokens:
            chunks.append(cur)
            cur, cur_max = [], 0
        cur.append(idx)
        cur_max = max(cur_max, L)
    if cur:
        chunks.append(cur)
    return chunks


def think_len(p: Sequence[int], g: Sequence[int], think_ids: tuple[int, int] | None) -> int:
    """How many leading tokens of g are think-block tokens (its `</think>` included): those after
    a prompt that ends with an open `<think>`. 0 without `think_ids` (no think vocabulary)."""
    if think_ids is None or not p or p[-1] != think_ids[0]:
        return 0
    return g.index(think_ids[1]) + 1 if think_ids[1] in g else len(g)


class Scored(NamedTuple):
    """`_token_logp` -> [B, T-1] each; logits at position q predict token q + 1."""

    logp: torch.Tensor  # log pi_T of the next token (-inf: banned in its phase); 0 off the mask
    w: torch.Tensor  # advantage weights, 0 off the mask
    m: torch.Tensor  # loss mask
    th: torch.Tensor  # think-block positions (`think_len`)
    ent: torch.Tensor | None  # with `entropy`: the entropy of pi_T, detached
    ban_mass: torch.Tensor | None  # with `entropy`: full-vocabulary pi_T mass on the think ban


def _token_logp(
    model: Any,
    chunk: Sequence[Item],
    temperature: float,
    dev: str,
    entropy: bool = False,
    think_ids: tuple[int, int] | None = None,
    bans: Bans | None = None,
) -> Scored:
    """pi_T of every next token, the sampler's distribution: with `bans` (`sample.phase_bans`, as
    `decode_wave` got them) the think ban is masked at the think-block positions and the action
    ban at the others, then renormalised; without, the full vocabulary (a tokenizer with neither
    think nor effort tokens bans nothing).  Rows are right-padded with <pad> (id 0). The model is
    causal, so padding never reaches a scored position."""
    T = max(len(p) + len(g) for p, g, _, _ in chunk)
    x = torch.zeros((len(chunk), T), dtype=torch.long)
    w = torch.zeros((len(chunk), T - 1))
    m = torch.zeros((len(chunk), T - 1), dtype=torch.bool)
    th = torch.zeros((len(chunk), T - 1), dtype=torch.bool)
    for r, (p, g, mk, a) in enumerate(chunk):
        assert p and len(mk) == len(g), "an item needs a prompt and one mask entry per token"
        x[r, : len(p) + len(g)] = torch.tensor(p + g)
        s = slice(len(p) - 1, len(p) - 1 + len(g))  # the positions that predict g
        m[r, s] = torch.tensor(mk, dtype=torch.bool)
        w[r, s] = m[r, s].float() * a
        th[r, len(p) - 1 : len(p) - 1 + think_len(p, g, think_ids)] = True
    x, w, m, th = x.to(dev), w.to(dev), m.to(dev), th.to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        logits = model(x[:, :-1])
    z = logits.float() / temperature
    ban = b_think = None
    if bans is not None:
        b_think, b_act = (b.to(dev) for b in bans)
        assert b_think.shape[-1] == z.shape[-1], "bans need the model's vocabulary size"
        ban = torch.where(th[..., None], b_think, b_act)  # [B, T-1, V]: each position's phase
        logp = torch.log_softmax(z.masked_fill(ban, -math.inf), dim=-1)
    else:
        logp = torch.log_softmax(z, dim=-1)
    ent = ban_mass = None
    if entropy:
        with torch.no_grad():
            lp = logp.detach()
            ent = -(lp.exp() * (lp if ban is None else lp.masked_fill(ban, 0.0))).sum(-1)
            ban_mass = torch.zeros_like(ent)
            if b_think is not None:
                ban_mass = torch.softmax(z.detach(), -1).masked_fill(~b_think, 0.0).sum(-1)
    # a prompt or forced token can be banned in its position's phase (-inf): where, not * 0
    tgt = torch.where(m, logp.gather(-1, x[:, 1:, None])[..., 0], 0.0)
    return Scored(tgt, w, m, th, ent, ban_mass)


@torch.no_grad()
def old_logps(
    model: Any,
    items: Sequence[Item],
    temperature: float,
    dev: str,
    max_tokens: int = 50_000,
    think_ids: tuple[int, int] | None = None,
    bans: Bans | None = None,
) -> list[torch.Tensor]:
    """The sampling policy's log-prob of every scored token, one tensor per item (mask order)."""
    out: list[torch.Tensor] = [torch.empty(0)] * len(items)
    for ids in _chunks(items, max_tokens):
        chunk = [items[k] for k in ids]
        s = _token_logp(model, chunk, temperature, dev, think_ids=think_ids, bans=bans)
        tgt, m = s.logp.cpu(), s.m.cpu()
        for r, k in enumerate(ids):
            out[k] = tgt[r][m[r]]
    return out


class LossStats(NamedTuple):
    loss: float = 0.0
    tokens: int = 0  # scored tokens
    clip_frac: float = 0.0  # share of the scored tokens with r outside [1 - eps_low, 1 + eps_high]
    clip_low: float = 0.0  # ... with r < 1 - eps_low
    clip_high: float = 0.0  # ... with r > 1 + eps_high
    entropy: float = 0.0  # mean entropy of pi_T over the scored tokens, nats
    tokens_think: int = 0  # scored think-block tokens (`think_len`)
    entropy_think: float = 0.0  # mean entropy over them (0 when there are none)
    entropy_action: float = 0.0  # ... over the other scored tokens
    ban_think: float = 0.0  # mean full-vocabulary pi_T mass on the think ban, scored think tokens
    ban_think_max: float = 0.0  # ... the largest
    ratio_max: float = 0.0  # the largest r over the scored tokens (1 without `old`)
    tokens_neg: int = 0  # scored tokens with A < 0
    neg_r3: float = 0.0  # share of them with r > R_WATCH
    nonfinite: int = 0  # scored tokens banned in their phase (log-prob -inf): left out of the loss
    tokens_acc: int = 0  # with `rejected`: scored tokens of the accepted items (not redrawn draws)
    clip_low_acc: float = 0.0  # ... the share of them with r < 1 - eps_low


def policy_loss(
    model: Any,
    items: Sequence[Item],
    temperature: float,
    dev: str,
    max_tokens: int = 50_000,
    old: Sequence[torch.Tensor] | None = None,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
    think_ids: tuple[int, int] | None = None,
    bans: Bans | None = None,
    dual_clip: float = 3.0,
    rejected: Sequence[bool] | None = None,
) -> LossStats:
    """Backpropagate the policy-gradient loss over `items`, as a token-level mean.
    Without `old`: -A * log pi_T (REINFORCE). With `old` (`old_logps` of the sampling policy, one
    tensor per item of `items`, in the same order): the PPO clipped surrogate with DAPO's
    clip-higher, -min(r A, clip(r, 1 - eps_low, 1 + eps_high) A), r = pi_T / pi_old, and for A < 0
    with `dual_clip` c > 0 the dual clip, -max(that, c A). At r = 1 its gradient is REINFORCE's.
    `think_ids` = (`<think>`, `</think>`) splits the entropy into think and action tokens, `bans`
    masks pi_T as the sampler did (`_token_logp`). `rejected` (one flag per item: a redrawn draw
    the step penalty scores) adds clip-low over the other items' tokens (`clip_low_acc`), the
    basis of the clip-low alarms when the penalty is on, and makes the loss a sum over every
    scored token divided by the accepted items' token count, so the penalised draws do not dilute
    the accepted tokens' per-token weight (plain GRPO's)."""
    assert old is None or len(old) == len(items), "old log-probs must align with the items"
    n_tok = sum(sum(m) for _, _, m, _ in items)
    if n_tok == 0:
        return LossStats()
    # the loss is a mean over the accepted items' tokens when `rejected` flags the penalised draws:
    # an accepted token weighs what it weighs in plain GRPO, the penalised draws come on top
    n_norm = n_tok
    if rejected is not None:
        n_norm = sum(sum(m) for (_, _, m, _), r in zip(items, rejected, strict=True) if not r)
        n_norm = n_norm or n_tok  # a minibatch of penalised draws only: their own mean
    total = ent_sum = ent_think = ban_sum = ban_max = 0.0
    r_max = 1.0 if old is None else 0.0
    n_low = n_high = n_think = n_neg = n_r3 = n_bad = 0
    n_acc = n_low_acc = 0
    for ids in _chunks(items, max_tokens):
        chunk = [items[k] for k in ids]
        s = _token_logp(
            model, chunk, temperature, dev, entropy=True, think_ids=think_ids, bans=bans
        )
        assert s.ent is not None and s.ban_mass is not None
        tgt, w, m, mt = s.logp, s.w, s.m, s.m & s.th
        ent_sum += float(s.ent[m].sum())
        ent_think += float(s.ent[mt].sum())
        n_think += int(mt.sum())
        if mt.any():
            ban_sum += float(s.ban_mass[mt].sum())
            ban_max = max(ban_max, float(s.ban_mass[mt].max()))
        if old is None:
            ok = m & torch.isfinite(tgt)
            loss = -(torch.where(ok, tgt, 0.0) * w).sum() / n_norm
        else:
            o = torch.zeros_like(tgt)
            for r, k in enumerate(ids):
                o[r][m[r]] = old[k].to(tgt.device)
            ok = m & torch.isfinite(tgt) & torch.isfinite(o)
            ratio = torch.exp(torch.where(ok, tgt - o, 0.0))  # 1 off the scored tokens
            surr = torch.minimum(ratio * w, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * w)
            if dual_clip > 0:  # A < 0: no lower than c A however large r gets
                surr = torch.where(w < 0, torch.maximum(surr, dual_clip * w), surr)
            loss = -torch.where(ok, surr, 0.0).sum() / n_norm
            rd = ratio.detach()
            n_low += int((rd < 1 - eps_low)[ok].sum())
            if rejected is not None:
                acc = torch.tensor([not rejected[k] for k in ids], device=ok.device)[:, None] & ok
                n_acc += int(acc.sum())
                n_low_acc += int((rd < 1 - eps_low)[acc].sum())
            n_high += int((rd > 1 + eps_high)[ok].sum())
            neg = ok & (w < 0)
            n_neg += int(neg.sum())
            n_r3 += int((rd > R_WATCH)[neg].sum())
            if ok.any():
                r_max = max(r_max, float(rd[ok].max()))
        n_bad += int((m & ~ok).sum())
        loss.backward()
        total += float(loss.detach())
    n_act = n_tok - n_think
    return LossStats(
        loss=total,
        tokens=n_tok,
        clip_frac=(n_low + n_high) / n_tok,
        clip_low=n_low / n_tok,
        clip_high=n_high / n_tok,
        entropy=ent_sum / n_tok,
        tokens_think=n_think,
        entropy_think=ent_think / n_think if n_think else 0.0,
        entropy_action=(ent_sum - ent_think) / n_act if n_act else 0.0,
        ban_think=ban_sum / n_think if n_think else 0.0,
        ban_think_max=ban_max,
        ratio_max=r_max,
        tokens_neg=n_neg,
        neg_r3=n_r3 / n_neg if n_neg else 0.0,
        nonfinite=n_bad,
        tokens_acc=n_acc,
        clip_low_acc=n_low_acc / n_acc if n_acc else 0.0,
    )


def clip_range(a: argparse.Namespace) -> tuple[float, float]:
    """(eps_low, eps_high): `--clip_eps_low` / `--clip_eps_high` when given, else `--clip_eps` when
    given, else 0.2 / 0.28 (DAPO's clip-higher)."""
    lo = a.clip_eps_low if a.clip_eps_low is not None else a.clip_eps
    hi = a.clip_eps_high if a.clip_eps_high is not None else a.clip_eps
    return (0.2 if lo is None else lo), (0.28 if hi is None else hi)


# --lr_schedule (module docstring, "Learning-rate schedule"): its options and their defaults. The
# parser leaves them out of the namespace unless given (argparse.SUPPRESS), so a constant run's
# args.json and checkpoint args are the earlier code's; `lr_schedule` resolves the ones a kind uses
LR_SCHEDULES = ("constant", "cosine", "wsd")
DECAY_SHAPES = ("linear", "cosine", "sqrt")  # wsd's decay; sqrt: 1 - sqrt(p), Hägele et al. 2024
SCHED_DEFAULTS: dict[str, Any] = {
    "warmup_iters": 0,
    "min_lr_frac": 0.1,
    "decay_frac": 0.2,
    "decay_shape": "linear",
}
SCHED_USES = {
    "constant": (),
    "cosine": ("warmup_iters", "min_lr_frac"),
    "wsd": ("warmup_iters", "min_lr_frac", "decay_frac", "decay_shape"),
}


def lr_schedule(a: argparse.Namespace) -> dict[str, Any] | None:
    """`--lr_schedule` and its options -> the run's schedule (kind, `iters` T, the options the kind
    uses, `warmup_steps`: the --warmup ramp over AdamW steps that applies on top, --warmup without
    --warmup_iters and 0 with it; wsd also `decay_iters`), None for constant (the earlier code's lr:
    --lr after the --warmup ramp). Refuses an option the kind does not use (it would do nothing).
    Sets the used options on `a`, defaults resolved, so args.json records them."""
    kind = getattr(a, "lr_schedule", "constant")
    assert kind in LR_SCHEDULES, kind
    bad = [k for k in SCHED_DEFAULTS if hasattr(a, k) and k not in SCHED_USES[kind]]
    assert not bad, f"--lr_schedule {kind} does not use --{', --'.join(bad)}"
    if kind == "constant":
        return None
    s: dict[str, Any] = {"kind": kind, "iters": int(a.iters)}
    for k in SCHED_USES[kind]:
        s[k] = getattr(a, k, SCHED_DEFAULTS[k])
        setattr(a, k, s[k])
    a.lr_schedule = kind
    assert s["iters"] >= 1, "--lr_schedule needs --iters >= 1"
    assert s["warmup_iters"] >= 0, "--warmup_iters must be >= 0"
    # one warmup: over iterations (in m(t)) when --warmup_iters is set, else --warmup's step ramp
    s["warmup_steps"] = 0 if s["warmup_iters"] else int(a.warmup)
    assert 0.0 <= s["min_lr_frac"] <= 1.0, "--min_lr_frac must be in [0, 1]"
    if kind == "wsd":
        assert 0.0 < s["decay_frac"] <= 1.0, "--decay_frac must be in (0, 1]"
        assert s["decay_shape"] in DECAY_SHAPES, s["decay_shape"]
        # round half up, at least one: the last iteration is always at min_lr_frac
        s["decay_iters"] = min(s["iters"], max(1, math.floor(s["decay_frac"] * s["iters"] + 0.5)))
    return s


def lr_mult(t: int, s: dict[str, Any]) -> float:
    """The schedule's lr factor m(t) at iteration t = 1 .. T of the run (`lr_schedule`'s `iters`;
    the iteration's number minus --start_iter). A linear warmup t / W while t < W (W =
    `warmup_iters`, none at 0); then cosine: f + (1 - f) (1 + cos(pi (t - W) / (T - W))) / 2, from 1
    at t = W to f = `min_lr_frac` at t = T; wsd: 1 until the last D = `decay_iters` iterations, then
    f + (1 - f) g(p), p = (t - (T - D)) / D, g = 1 - p (linear), (1 + cos(pi p)) / 2 (cosine) or
    1 - sqrt(p) (sqrt), f at t = T. The warmup comes first: a run no longer than its warmup (as a
    2-iteration smoke run at W = 3) is all warmup and does not reach f. EI's fine-tune
    (`train.finetune`) has the same warmup and cosine, over AdamW steps."""
    T, W, f = s["iters"], s["warmup_iters"], s["min_lr_frac"]
    assert 1 <= t <= T, f"iteration {t} outside the schedule's 1 .. {T}"
    if t < W:
        return t / W
    if s["kind"] == "cosine":
        return f + (1.0 - f) * 0.5 * (1.0 + math.cos(math.pi * (t - W) / max(1, T - W)))
    assert s["kind"] == "wsd", s["kind"]
    D = s["decay_iters"]
    if t <= T - D:
        return 1.0
    p = (t - (T - D)) / D
    g = {
        "linear": 1.0 - p,
        "cosine": 0.5 * (1.0 + math.cos(math.pi * p)),
        "sqrt": 1.0 - math.sqrt(p),
    }[s["decay_shape"]]
    return f + (1.0 - f) * g


def sched_lr(base: float, mult: float, scale: float, n_step: int, warmup_steps: int) -> float:
    """A scheduled run's lr at its AdamW step `n_step`: --lr x m(t) x the rollback scale x the
    step ramp (`warmup_steps`; 1 at 0). The constant code's product with m(t) inserted, so m = 1
    gives its lr bit for bit."""
    return base * mult * scale * min(1.0, n_step / max(1, warmup_steps))


def _sched_line(s: dict[str, Any], lr: float) -> str:
    """The run's opening line under a schedule."""
    head = f"lr schedule: {s['kind']} over {s['iters']} iterations, --lr {lr:g}"
    if s["warmup_iters"]:
        warm = f", warmup over {s['warmup_iters']} iterations"
    else:
        warm = f", warmup over {s['warmup_steps']} AdamW steps (--warmup)"
    tail = (
        f", decay over the last {s['decay_iters']} ({s['decay_shape']})"
        if s["kind"] == "wsd"
        else ""
    )
    return f"{head}{warm}{tail}, x{s['min_lr_frac']:g} at the last; a rollback halves on top"


def check_resume_schedule(path: str, sched: dict[str, Any] | None, start_iter: int) -> None:
    """`--resume`: the checkpoint's run (its saved args) must have this command's schedule, T and
    `--start_iter` included, else the resumed iterations would get another run's lrs. Two
    constant runs are not compared (a resume may change --iters, as before). The checkpoint is
    memory-mapped: only its args are read."""
    ck = torch.load(str(path), map_location="cpu", mmap=True)
    prev = (ck.get("extra") or {}).get("args") or {}
    if sched is None and prev.get("lr_schedule", "constant") == "constant":
        return
    keys = ("lr_schedule", "iters", "warmup", *SCHED_DEFAULTS)
    was = lr_schedule(argparse.Namespace(**{k: prev[k] for k in keys if k in prev}))
    p0 = prev.get("start_iter", 0)
    assert (was, p0) == (sched, start_iter), (
        f"--resume {path}: its run's lr schedule {was} (start_iter {p0}) is not this "
        f"command's {sched} (start_iter {start_iter})"
    )


UNSEEN_PRIORS = ("half", "mean")  # --unseen_prior (module docstring, "Unseen-pair prior")
PRIOR_MIN_SEEN = 32  # --unseen_prior mean: a budget's seen pairs before its mean replaces 1/2


class Curriculum:
    """Learnability sampling over (target, budget) pairs; see the module docstring. `prior`: the
    prior of a pair's pass rate, 'half' (Beta(1, 1)) or 'mean' (--unseen_prior)."""

    def __init__(
        self,
        n: int,
        budgets: Sequence[int],
        floor: float,
        decay: float,
        seed: int,
        prior: str = "half",
    ) -> None:
        assert prior in UNSEEN_PRIORS, prior
        self.keys = [(t, b) for t in range(n) for b in budgets]
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.s = [0.0] * len(self.keys)
        self.n = [0.0] * len(self.keys)
        self.floor, self.decay = floor, decay
        self.rng = random.Random(seed)
        self.prior = prior
        self._mu: dict[int, tuple[float, int]] | None = None  # `prior_means`, until s / n change

    def prior_means(self) -> dict[int, tuple[float, int]]:
        """Per budget -> (m_b, the budget's seen pairs): the prior mean of a pair's pass rate.
        'half': 1/2. 'mean': the mean over the budget's seen pairs (n > 0) of their decayed pass
        rate s / n (unshrunk), once PRIOR_MIN_SEEN are seen; 1/2 before. Cached until an update."""
        if self._mu is None:
            tot: dict[int, float] = collections.defaultdict(float)
            cnt: dict[int, int] = collections.defaultdict(int)
            for i, (_, b) in enumerate(self.keys):
                cnt[b] += 0
                if self.n[i] > 0:
                    tot[b] += self.s[i] / self.n[i]
                    cnt[b] += 1
            ok = self.prior == "mean"
            self._mu = {
                b: (tot[b] / c if ok and c >= PRIOR_MIN_SEEN else 0.5, c) for b, c in cnt.items()
            }
        return self._mu

    def p(self, i: int) -> float:
        """The pair's pass-rate estimate: the posterior mean under Beta(1, 1), (s + 1) / (n + 2),
        or with --unseen_prior mean under Beta(2 m_b, 2 (1 - m_b)) (the same strength 2, mean
        m_b), (s + 2 m_b) / (n + 2): an unseen pair is at m_b."""
        if self.prior == "half":
            return (self.s[i] + 1.0) / (self.n[i] + 2.0)
        m = self.prior_means()[self.keys[i][1]][0]
        return (self.s[i] + 2.0 * m) / (self.n[i] + 2.0)

    def weight(self, i: int) -> float:
        p = self.p(i)
        return p * (1.0 - p) + self.floor

    def draw(
        self, m: int, exclude: Collection[int] = (), max_pass: float | None = None
    ) -> list[tuple[int, int]]:
        """m distinct pairs, weighted, without replacement (Efraimidis-Spirakis keys u^(1/w), as
        log(u) / w: u^(1/w) underflows to 0 for small weights, which would tie and fall back to
        index order). `exclude` (pair indices) and seen pairs with p > `max_pass` are not drawn
        (fewer than m when too few are left); the RNG draws one number per pair either way."""
        keyed = [
            (math.log(1.0 - self.rng.random()) / self.weight(i), i) for i in range(len(self.keys))
        ]
        if exclude or max_pass is not None:
            keyed = [
                (k, i)
                for k, i in keyed
                if i not in exclude
                and (max_pass is None or self.n[i] == 0 or self.p(i) <= max_pass)
            ]
        keyed.sort(reverse=True)
        return [self.keys[i] for _, i in keyed[: min(m, len(keyed))]]

    def over(self, max_pass: float) -> int:
        """How many pairs `draw(max_pass=...)` leaves out (seen ones only: under the 1/2 prior an
        unseen pair, at 1/2, is never over a --max_pass, which is at least 1/2)."""
        return sum(self.n[i] > 0 and self.p(i) > max_pass for i in range(len(self.keys)))

    def prior_stats(self) -> dict[str, Any]:
        """--unseen_prior's row: the prior per budget (`prior_mean`; `fallback`: the budgets still
        at 1/2 under 'mean'), the seen and unseen pairs, their mean draw weight, and the unseen
        pairs' share of the total weight (the expected unseen share of a draw's first pick)."""
        mu = self.prior_means()
        w = [self.weight(i) for i in range(len(self.keys))]
        un = [self.n[i] == 0 for i in range(len(self.keys))]
        n_un = sum(un)
        w_un = sum(x for x, u in zip(w, un, strict=True) if u)
        w_all = sum(w)
        return {
            "prior_mean": {str(b): m for b, (m, _) in sorted(mu.items())},
            "fallback": [
                b for b, (_, c) in sorted(mu.items()) if self.prior == "mean" and c < PRIOR_MIN_SEEN
            ],
            "seen": len(w) - n_un,
            "unseen": n_un,
            "w_seen": (w_all - w_un) / max(1, len(w) - n_un),
            "w_unseen": w_un / max(1, n_un),
            "unseen_weight_share": w_un / w_all if w_all else 0.0,
        }

    def update(self, key: tuple[int, int], successes: int, trials: int) -> None:
        i = self.index[key]
        self.s[i] = self.s[i] * self.decay + successes
        self.n[i] = self.n[i] * self.decay + trials
        self._mu = None

    def state(self) -> dict[str, Any]:
        return {"s": list(self.s), "n": list(self.n), "rng": self.rng.getstate()}

    def load_state(self, st: dict[str, Any]) -> None:
        assert len(st["s"]) == len(self.keys), "the curriculum's pairs differ from the checkpoint's"
        self.s, self.n = list(st["s"]), list(st["n"])
        self.rng.setstate(st["rng"])
        self._mu = None

    def summary(self) -> dict[str, Any]:
        seen = [i for i in range(len(self.keys)) if self.n[i] > 0]
        ps = [self.p(i) for i in seen]
        by_b: dict[str, list[float]] = collections.defaultdict(list)
        for i in seen:
            by_b[str(self.keys[i][1])].append(self.p(i))
        return {
            "seen": len(seen),
            "pairs": len(self.keys),
            "frontier": sum(0.1 < p < 0.9 for p in ps) / max(1, len(ps)),
            "mean_p_by_budget": {b: sum(v) / len(v) for b, v in sorted(by_b.items())},
        }


def sample_groups(
    model: Any,
    tok: Any,
    targets: Sequence[dict[str, Any]],
    budgets: Sequence[int],
    a: argparse.Namespace,
    seed: int,
    think: bool,
    stats: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    """`a.group` attempts per target, all at the target's budget (one think_generate call per
    budget, so each call has one budget). `a.arm` (default plain) is the sampler's arm: a guided
    arm redraws a rejected draw, at most `a.max_rej` times per attempt, and with a step penalty
    the redrawn draws are kept (`keep_rejected`). `stats` gets the sampler's counters, summed
    over the calls (`rej_cause`, `end`, `redraws`, ...)."""
    arm = getattr(a, "arm", "plain")
    keep_rej = arm != "plain" and getattr(a, "rej_penalty", 0.0) > 0
    by: dict[int, list[int]] = collections.defaultdict(list)
    for i, b in enumerate(budgets):
        by[b].append(i)
    per: list[list[dict[str, Any]]] = [[] for _ in targets]
    for b, idx in sorted(by.items()):
        prompts = [targets[i]["prompt"] for i in idx for _ in range(a.group)]
        st: dict[str, Any] = {}
        res = think_generate(
            model,
            tok,
            prompts,
            arm=arm,
            budgets=[b] if think else 0,
            think=think,
            temperature=a.temperature,
            max_action=a.max_action,
            max_steps=a.max_steps,
            max_rej=getattr(a, "max_rej", 10),
            batch=a.batch,
            seed=seed * 10_007 + b,  # distinct per (iteration, budget) for any budget < 10,007
            stats=st,
            keep_trace=True,
            force_mode=a.force_mode,
            keep_rejected=keep_rej,
        )
        res = judge(tok, prompts, res, a.lean)
        for n, i in enumerate(idx):
            per[i] = res[n * a.group : (n + 1) * a.group]
        if stats is not None:
            for k, v in st.items():
                if isinstance(v, collections.Counter):
                    stats.setdefault(k, collections.Counter()).update(v)
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    stats[k] = stats.get(k, 0) + v
    return per


PASS_BANDS = ("0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "1")  # a group's pass rate


def pass_band(p: float) -> str:
    if p <= 0:
        return PASS_BANDS[0]
    if p >= 1:
        return PASS_BANDS[-1]
    return PASS_BANDS[1 + min(3, int((p - 1e-12) // 0.25))]


def target_length(t: dict[str, Any]) -> int | None:
    """A target's proof length: `min_lines` (the minimal length, hard712 and later pools), else
    `n_lines` (the generator's), else `reference_lines`."""
    for k in ("min_lines", "n_lines", "reference_lines"):
        v = t.get(k)
        if isinstance(v, int):
            return v
    return None


def monitor_metrics(
    per: Sequence[Sequence[dict[str, Any]]],
    lengths: Sequence[int | None],
    skipped: Sequence[bool],
    items: Sequence[Item],
    item_src: Sequence[tuple[int, int, str]],
    think_ids: tuple[int, int] | None,
    sampler: dict[str, Any] | None = None,
    top: int = 6,
) -> dict[str, Any]:
    """`--monitor`: where an iteration's gradient comes from. Per target length (str(L)) and per
    group pass-rate band (`PASS_BANDS`): groups, the share with a gradient, the mean pass rate,
    the band's share of the |advantage|-weighted scored-token mass (sum over items of |A| x the
    scored tokens: the pre-clip size of each item's push), and within the band the share of that
    mass on think-block tokens and on rejected draws. Overall: the mass split think / action and
    accepted (an attempt's steps) / rejected (redrawn draws), and the sampler's rejections: per
    draw, per attempt, and the most common causes (share of all rejections). `item_src`: per
    item (group, attempt, kind) as `group_items` fills it, the group index first."""
    n_g = len(per)
    g_mass = [0.0] * n_g
    g_think = [0.0] * n_g
    g_rej = [0.0] * n_g
    tot = th = rj = 0.0
    for (p, g, m, adv), (gi, _, kind) in zip(items, item_src, strict=True):
        n = sum(m)
        nt = sum(m[: think_len(p, g, think_ids)])
        w = abs(adv)
        g_mass[gi] += w * n
        g_think[gi] += w * nt
        tot += w * n
        th += w * nt
        if kind == "rej":
            g_rej[gi] += w * n
            rj += w * n
    rates = [sum(bool(x.get("ok")) for x in att) / max(1, len(att)) for att in per]

    def bands(keys: Sequence[str]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for k in sorted(set(keys), key=lambda s: (len(s), s)):
            ix = [i for i in range(n_g) if keys[i] == k]
            m = sum(g_mass[i] for i in ix)
            out[k] = {
                "groups": len(ix),
                "with_grad": sum(not skipped[i] for i in ix) / len(ix),
                "pass": sum(rates[i] for i in ix) / len(ix),
                "grad_share": m / tot if tot else 0.0,
                "think_share": sum(g_think[i] for i in ix) / m if m else 0.0,
                "rej_share": sum(g_rej[i] for i in ix) / m if m else 0.0,
            }
        return out

    flat = [x for att in per for x in att]
    draws = sum(int(x.get("draws") or 0) for x in flat)
    n_rej = sum(len(x.get("causes") or []) for x in flat)
    causes: collections.Counter[str] = collections.Counter(
        c for x in flat for c in x.get("causes") or []
    )
    toks = sum(int(x.get("tokens") or 0) for x in flat)
    rtoks = sum(int(x.get("rejected") or 0) for x in flat)
    # the step penalty's 'diverge' scope: how many redrawn draws found the draw accepted at their
    # state (a systematic offset between `step` and the trace would show here), and how much of a
    # matched redrawn draw is the prefix it shares with that draw
    n_red = n_match = 0
    shared: list[float] = []
    for x in flat:
        acc = [s for s in x.get("trace") or [] if "gen" in s and not s.get("rejected")]
        for r in x.get("redrawn") or []:
            n_red += 1
            st_ = r.get("step")
            if isinstance(st_, int) and 0 <= st_ < len(acc) and acc[st_]["prompt"] == r["prompt"]:
                n_match += 1
                shared.append(_lcp(r["gen"], acc[st_]["gen"]) / max(1, len(r["gen"])))
    out: dict[str, Any] = {
        "by_len": bands(["?" if L is None else str(L) for L in lengths]),
        "by_pass": bands([pass_band(r) for r in rates]),
        "mass_total": tot,
        "mass_think": th / tot if tot else 0.0,
        "mass_action": (tot - th) / tot if tot else 0.0,
        "mass_rejected": rj / tot if tot else 0.0,
        "mass_accepted": (tot - rj) / tot if tot else 0.0,
        "rej_per_draw": n_rej / draws if draws else 0.0,
        "rej_per_attempt": n_rej / max(1, len(flat)),
        "rejected_token_share": rtoks / toks if toks else 0.0,
        "rej_causes": {c: v / n_rej for c, v in causes.most_common(top)} if n_rej else {},
        "draws": draws,
        "redrawn_drafts": n_red,
        "diverge_match": n_match / n_red if n_red else None,
        "shared_prefix": sum(shared) / len(shared) if shared else None,
    }
    if sampler:
        out["redraws"] = int(sampler.get("redraws", 0))
        out["repeat_action_redraws"] = int(sampler.get("repeat_action_redraws", 0))
    return out


def _length_line(row: dict[str, Any]) -> str:
    """The per-iteration line's length part: per budget, think tokens per block / the target and
    the share force-closed; then every block's forced share, the mean d and alpha x d."""
    per = " ".join(
        f"{b}:{n:.1f}/{row['len_target_by_budget'][b]:g} f{100 * row['forced_by_budget'][b]:.0f}%"
        for b, n in row["think_block_by_budget"].items()
    )
    return (
        f"think/blk {per or '-'} forced {100 * row['forced_frac']:.1f}% "
        f"dev {row['len_dev']:.3f} lpen {row['len_target_penalty']:.3f}"
    )


def clip_low_basis(row: dict[str, Any]) -> tuple[float, str]:
    """The clip-low the alarms read -> (value, label). With the step penalty the redrawn draws'
    tokens are pushed down on purpose, so their ratios fall below 1 - eps_low within an
    iteration; the alarms then read the accepted items' clip-low (`clip_low_accepted`), which
    keeps the collapse signal they were built for. Otherwise `clip_low`."""
    if "clip_low_accepted" in row:
        return float(row["clip_low_accepted"]), " (accepted tokens)"
    return float(row["clip_low"]), ""


def alarms(row: dict[str, Any], gn_prev: Sequence[float]) -> list[str]:
    """An iteration's metrics row -> its alarms, one message each (the ALARM_* thresholds).
    `gn_prev`: the grad-norm max of the earlier iterations that took a step. The safeguard alarms
    read their thresholds from the row (`alarm_cfg`: `capped`, `clip_low_rel`, `canary_mult`; 0
    or absent: off) with the values they compare (`capped_by_budget`, `clip_low_median`,
    `canary` and its `base`), so a row says why it alarmed."""
    out = _base_alarms(row, gn_prev)
    cfg = row.get("alarm_cfg") or {}
    cap = cfg.get("capped") or 0.0
    over = [(b, v) for b, v in (row.get("capped_by_budget") or {}).items() if v > cap]
    if cap > 0 and over:
        worst = ", ".join(f"B{b} {v:.4f}" for b, v in over)
        out.append(f"capped attempts {worst} > {cap:g}")
    rel, med = cfg.get("clip_low_rel") or 0.0, row.get("clip_low_median")
    cl, what = clip_low_basis(row)
    if rel > 0 and med is not None and cl > max(rel * med, ALARM_CLIP_LOW_FLOOR):
        out.append(f"clip-low{what} {cl:.4f} > {rel:g} x running median {med:.4f}")
    mult, can = cfg.get("canary_mult") or 0.0, row.get("canary")
    if mult > 0 and can and can.get("base"):
        for k, label in CANARY_KEYS:
            v, b = can[k], can["base"][k]
            if b > 0 and v > mult * b:
                out.append(f"canary {label} {v:.4g} > {mult:g} x base {b:.4g}")
    return out


def ban_alarm(row: dict[str, Any]) -> str | None:
    """The think ban alarm's message, None when the row's mass is not over ALARM_BAN."""
    if row["ban_think"] > ALARM_BAN:
        return (
            f"think banned mass {row['ban_think']:.3g} > {ALARM_BAN:g} "
            f"(max {row['ban_think_max']:.3g})"
        )
    return None


def _base_alarms(row: dict[str, Any], gn_prev: Sequence[float]) -> list[str]:
    out = []
    ban = ban_alarm(row)
    if ban is not None:
        out.append(ban)
    if len(gn_prev) >= ALARM_GN_AFTER:
        med = statistics.median(gn_prev)
        if row["grad_norm_max"] > ALARM_GN * med:
            out.append(
                f"grad-norm max {row['grad_norm_max']:.3g} > {ALARM_GN:g} x median {med:.3g}"
            )
    cl, what = clip_low_basis(row)
    if cl > ALARM_CLIP_LOW:
        out.append(f"clip-low{what} {cl:.4f} > {ALARM_CLIP_LOW:g}")
    if row["overlong_frac"] > ALARM_OVERLONG:
        out.append(f"overlong {row['overlong_frac']:.4f} > {ALARM_OVERLONG:g}")
    if row["nonfinite"]:
        out.append(f"{row['nonfinite']} scored tokens banned in their phase (sampler / scorer)")
    return out


def ban_rollback(restore_mass: float, mass: float) -> bool:
    """`--ban_restore_check`'s choice for an iteration whose only alarm is the think ban mass ->
    roll back? `restore_mass`: the mass of the checkpoint the rollback would restore, on the
    iteration's think tokens (`restore_ban_mass`); `mass`: the iteration's own (`ban_think`). No
    when the restore point is over ALARM_BAN too and `mass` is under BAN_SPIKE x it: restoring
    would halve the lr without bringing the mass under the limit."""
    return not (restore_mass > ALARM_BAN and mass < BAN_SPIKE * restore_mass)


@torch.no_grad()
def ban_mass(
    model: Any,
    items: Sequence[Item],
    temperature: float,
    dev: str,
    max_tokens: int = 50_000,
    think_ids: tuple[int, int] | None = None,
    bans: Bans | None = None,
) -> float:
    """`policy_loss`'s `ban_think` of `items` without a gradient: the mean full-vocabulary pi_T
    mass on the think ban at their scored think tokens (0 without any)."""
    tot, n = 0.0, 0
    for ids in _chunks(items, max_tokens):
        chunk = [items[k] for k in ids]
        s = _token_logp(
            model, chunk, temperature, dev, entropy=True, think_ids=think_ids, bans=bans
        )
        assert s.ban_mass is not None
        mt = s.m & s.th
        tot += float(s.ban_mass[mt].sum())
        n += int(mt.sum())
    return tot / n if n else 0.0


def restore_ban_mass(
    model: Any,
    path: str,
    items: Sequence[Item],
    temperature: float,
    dev: str,
    max_tokens: int = 50_000,
    think_ids: tuple[int, int] | None = None,
    bans: Bans | None = None,
) -> float:
    """`ban_mass` of `items` under the weights of the checkpoint at `path` (a rollback's restore
    point), on a scratch copy of `model`, so the run's model and optimizer are untouched. The
    model has no dropout and no other train / eval difference, so this is what `policy_loss`
    computes at those weights."""
    scratch = copy.deepcopy(model).requires_grad_(False)
    scratch.load_state_dict(torch.load(str(path), map_location="cpu")["state"])
    scratch.eval()
    return ban_mass(scratch, items, temperature, dev, max_tokens, think_ids, bans)


def _ban_check_line(it: int, bc: dict[str, Any]) -> str:
    """`--ban_restore_check`'s line: the restore point's think ban mass and the choice."""
    x, y = bc["restore_mass"], bc["mass"]
    head = f"ALARM it {it}: {'' if bc['rollback'] else 'no '}rollback: the restore point "
    head += f"(it {bc['to_iter']}) has think banned mass {x:.3g}"
    if bc["rollback"]:
        return f"{head} on this iteration's think tokens (this iteration {y:.3g})"
    return (
        f"{head} > {ALARM_BAN:g} on this iteration's think tokens "
        f"(this iteration {y:.3g} < {BAN_SPIKE:g} x {x:.3g})"
    )


def prune_rolling(ckdir: Path, name: str, keep: int) -> list[Path]:
    """Delete all but the newest `keep` rolling checkpoints `<name>_it<i>_rolling.pt` (and their
    upload sidecars) -> the deleted. Permanent and alarm checkpoints are never matched."""
    pat = re.compile(re.escape(name) + r"_it(\d+)_rolling\.pt")
    found = sorted((int(mt[1]), p) for p in ckdir.iterdir() if (mt := pat.fullmatch(p.name)))
    gone = [p for _, p in found[: max(0, len(found) - keep)]]
    for p in gone:
        p.unlink()
        p.with_name(p.name + ".upload.json").unlink(missing_ok=True)
    return gone


@dataclasses.dataclass
class LoopState:
    """The trainer's state besides the weights, optimizer, curricula and RNGs. A resume restores
    all of it; a rollback restores the step count and the alarm histories, and keeps the rest."""

    n_step: int = 0  # AdamW steps taken (the warmup reads it)
    gn_prev: list[float] = dataclasses.field(default_factory=list)  # grad-norm max, per iteration
    clip_prev: list[float] = dataclasses.field(default_factory=list)  # clip-low, per iteration
    lr_scale: float = 1.0  # halved at each rollback
    rollbacks: int = 0
    canary_base: dict[str, Any] | None = None  # the canary before the first iteration
    last_good: str | None = None  # the checkpoint a rollback restores ...
    last_good_iter: int | None = None  # ... and its iteration


def train_state(
    it: int,
    ls: LoopState,
    opt: torch.optim.Optimizer,
    rngs: dict[str, random.Random],
    curs: dict[str, Curriculum],
    compute: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Everything a resume needs besides the weights (plain types and tensors, so
    `torch.load(weights_only=True)` reads it). `compute`: --refill's run totals of sampled
    groups, attempts and tokens (a rollback does not restore them; a resume does)."""
    st = {
        "version": STATE_VERSION,
        "iter": it,
        "loop": dataclasses.asdict(ls),
        "opt": opt.state_dict(),
        "rng": {k: r.getstate() for k, r in rngs.items()},
        "curriculum": {k: c.state() for k, c in curs.items()},
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if compute is not None:
        st["compute"] = dict(compute)
    return st


def restore_state(
    st: dict[str, Any],
    opt: torch.optim.Optimizer,
    ls: LoopState,
    rngs: dict[str, random.Random],
    curs: dict[str, Curriculum],
    full: bool,
) -> int:
    """`train_state` back into the run's objects -> its iteration. `full` (a resume) also restores
    the lr scale, the rollback count, the canary base, the last good checkpoint and the torch
    RNGs; a rollback keeps its own."""
    assert st.get("version") == STATE_VERSION, f"train_state version {st.get('version')}"
    assert set(st["curriculum"]) == set(curs), (
        f"the checkpoint's curricula {sorted(st['curriculum'])} are not this run's {sorted(curs)} "
        "(--targets_extra)"
    )
    opt.load_state_dict(st["opt"])
    for k, r in rngs.items():
        r.setstate(st["rng"][k])
    for k, c in curs.items():
        c.load_state(st["curriculum"][k])
    loop = st["loop"]
    ls.n_step = int(loop["n_step"])
    ls.gn_prev, ls.clip_prev = list(loop["gn_prev"]), list(loop["clip_prev"])
    if full:
        ls.lr_scale, ls.rollbacks = float(loop["lr_scale"]), int(loop["rollbacks"])
        ls.canary_base = loop["canary_base"]
        ls.last_good, ls.last_good_iter = loop["last_good"], loop["last_good_iter"]
        torch.set_rng_state(st["torch_rng"].cpu())
        cuda = st.get("cuda_rng") or []
        if cuda and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda])
    return int(st["iter"])


def _trim_metrics(out: Path, start: int) -> int:
    """On a resume from iteration `start`: rows of later iterations move from metrics.jsonl to
    metrics.pre_resume.jsonl (appended) -> how many moved."""
    path = out / "metrics.jsonl"
    if not path.exists():
        return 0
    lines = [x for x in path.read_text().splitlines() if x.strip()]
    keep = [x for x in lines if json.loads(x)["iter"] <= start]
    moved = [x for x in lines if json.loads(x)["iter"] > start]
    if moved:
        with open(out / "metrics.pre_resume.jsonl", "a") as f:
            f.write("".join(x + "\n" for x in moved))
        path.write_text("".join(x + "\n" for x in keep))
    return len(moved)


def _canary_line(it: int, c: dict[str, Any]) -> str:
    b = c.get("base") or {}
    base = (
        f" | base nll {b['nll']:.4f} P(:) {100 * b['colon']:.4f}% "
        f"wrong {100 * b['wrong_max']:.3f}% ({b['wrong_tok']})"
        if b
        else ""
    )
    return (
        f"canary it {it}: nll {c['nll']:.4f} P(:) {100 * c['colon']:.4f}% "
        f"(log {c['colon_logp']:.2f}) wrong {100 * c['wrong_max']:.3f}% ({c['wrong_tok']}) "
        f"[{c['steps']} steps]{base}"
    )


def _pilot_lines(it: int, row: dict[str, Any]) -> list[str]:
    """The pilot options' lines (module docstring): --refill's rounds, skip shares and sampled
    compute; --max_pass's filtered pairs; the unlikeliness reward's deductions."""
    out = []
    if "refill" in row:
        f = row["refill"]
        out.append(
            f"refill it {it}: {f['rounds']} rounds, drawn {f['drawn']} signal {f['signal']} "
            f"unseen {f['unseen']}; skip {100 * f['skip_before']:.0f}% -> "
            f"{100 * f['skip_after']:.0f}% ({f['signal_groups']}/{f['target']} with a signal), "
            f"surplus {f['surplus']}; sampled "
            f"{f['attempts']} attempts {f['tokens']} tokens (run {f['attempts_cum']} / "
            f"{f['tokens_cum']})"
        )
    if "max_pass" in row:
        mp = row["max_pass"]
        extra = f", extra {mp['over_extra']}" if "over_extra" in mp else ""
        out.append(f"max_pass it {it}: {mp['over']} pairs over {mp['threshold']:g}{extra}")
    if "unlikely" in row:
        u = row["unlikely"]
        lc = "-" if u["len_corr"] is None else f"{u['len_corr']:+.2f}"
        out.append(
            f"unlikely it {it}: beta {u['beta']:g}, {u['groups']} groups, {u['solved']} solved "
            f"attempts ranked, deduction mean {u['pen_mean']:.3f} max {u['pen_max']:.3f}, "
            f"rank/length corr {lc} ({u['len_corr_groups']} groups)"
        )
    if "prior" in row:
        q = row["prior"]
        mu = " ".join(f"{b}:{m:.3f}" for b, m in q["prior_mean"].items())
        fb = f" (fallback 1/2 at B{','.join(map(str, q['fallback']))})" if q["fallback"] else ""
        out.append(
            f"prior it {it}: {q['mode']}, prior mean {mu}{fb}; unseen drawn "
            f"{q['unseen_drawn']}/{q['drawn']} ({100 * q['unseen_drawn_share']:.0f}%); unseen "
            f"{q['unseen']} pairs hold {100 * q['unseen_weight_share']:.0f}% of the weight "
            f"(mean weight unseen {q['w_unseen']:.3f}, seen {q['w_seen']:.3f})"
        )
    return out


def _monitor_line(it: int, mo: dict[str, Any]) -> str:
    """`--monitor`'s line: the gradient share and pass rate per length, the rejected-draw share of
    the gradient, rejections per draw and the top causes."""
    by = " ".join(
        f"{k}:{100 * v['grad_share']:.0f}%/p{100 * v['pass']:.0f}" for k, v in mo["by_len"].items()
    )
    cz = ", ".join(f"{c[:28]} {100 * v:.0f}%" for c, v in list(mo["rej_causes"].items())[:3])
    return (
        f"mon it {it}: grad/pass by len {by} | think {100 * mo['mass_think']:.0f}% "
        f"rejected-draw {100 * mo['mass_rejected']:.0f}% of grad | rej/draw "
        f"{100 * mo['rej_per_draw']:.1f}% ({mo['rej_per_attempt']:.2f}/att) {cz}"
        + (
            f" | diverge match {100 * mo['diverge_match']:.0f}%"
            if mo.get("diverge_match") is not None
            else ""
        )
    )


REFILL_MAX = 2  # --refill_max: extra sampling rounds per iteration


def pilot_options(a: argparse.Namespace) -> tuple[bool, float | None, float]:
    """The pilot options (module docstring; left out of the namespace unless given) -> (--refill,
    --max_pass or None, --unlikely_beta or 0). With --refill, --refill_max is set on `a` (default
    REFILL_MAX), so args.json records it."""
    refill = bool(getattr(a, "refill", False))
    assert refill or not hasattr(a, "refill_max"), "--refill_max needs --refill"
    if refill:
        a.refill_max = int(getattr(a, "refill_max", REFILL_MAX))
        assert a.refill_max >= 1, "--refill_max must be >= 1"
    max_pass = getattr(a, "max_pass", None)
    assert max_pass is None or 0.5 <= max_pass <= 1.0, (
        "--max_pass must be in [0.5, 1] (unseen pairs are at p = 0.5)"
    )
    beta = float(getattr(a, "unlikely_beta", 0.0))
    assert 0.0 <= beta <= 1.0, "--unlikely_beta must be in [0, 1]"
    return refill, max_pass, beta


def run(a: argparse.Namespace) -> list[dict[str, Any]]:
    assert not a.save_rolling or a.keep_rolling >= 1, "--keep_rolling must be >= 1"
    assert a.len_target_coef >= 0, "--len_target_coef must be >= 0"
    assert a.len_target_frac > 0, "--len_target_frac must be > 0"
    assert 0 <= a.targets_extra_frac < 1, "--targets_extra_frac must be in [0, 1)"
    assert not a.targets_extra_frac or a.targets_extra, "--targets_extra_frac needs --targets_extra"
    assert a.kl_anchor_coef >= 0, "--kl_anchor_coef must be >= 0"
    assert not a.kl_anchor_coef or a.kl_anchor_ref, "--kl_anchor_coef needs --kl_anchor_ref"
    assert not a.kl_anchor_coef or a.kl_anchor_data or a.canary, (
        "--kl_anchor_coef needs --kl_anchor_data or --canary (the anchor's steps)"
    )
    assert a.canary_every >= 1 and a.canary_n >= 1 and a.kl_anchor_n >= 1
    assert a.rej_penalty >= 0, "--rej_penalty must be >= 0"
    assert a.ban_stop >= 0, "--ban_stop must be >= 0"
    assert not a.ban_restore_check or a.rollback_on_alarm, (
        "--ban_restore_check needs --rollback_on_alarm"
    )
    assert a.arm in ARMS, a.arm
    if a.rej_penalty and a.arm == "plain":
        print("NOTE: --rej_penalty with the plain arm: only an attempt's ending draw is penalised")
    if a.len_target_coef and not a.effort:
        print("WARNING: --len_target_coef without --effort: the target is not in the prompt")
    sched = lr_schedule(a)  # None: constant; else resolved onto `a` (args.json)
    refill, max_pass, beta = pilot_options(a)  # resolved onto `a` when given (args.json)
    prior = getattr(a, "unseen_prior", None)  # --unseen_prior; None: not given (half, no row)
    if a.resume:  # before anything is written: a refused resume leaves the run's files alone
        check_resume_schedule(a.resume, sched, a.start_iter)
    out = Path(a.out)
    run_env(out)
    a.clip_eps_low, a.clip_eps_high = clip_range(a)  # resolved, so args.json records them
    if sched is not None:
        print(_sched_line(sched, a.lr), flush=True)
        # m(T) = min_lr_frac unless the warmup covers the last iteration (`lr_mult`)
        if sched["warmup_iters"] >= sched["iters"] + (sched["kind"] == "wsd"):
            print(
                f"NOTE: --warmup_iters {sched['warmup_iters']} covers the last of the run's "
                f"{sched['iters']} iterations: it ends above --min_lr_frac"
            )
    if a.resume and (out / "args.json").exists():  # keep the args of the run being resumed
        k = 0
        while (out / f"args.pre_resume{k or ''}.json").exists():
            k += 1
        (out / "args.json").rename(out / f"args.pre_resume{k or ''}.json")
    with bare():
        D.record.save_config(vars(a), str(out) + "/", arm=a.name, role="rl")
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    budgets = [int(b) for b in a.budgets.split(",")]
    init = a.init
    if a.effort:
        bad = [b for b in budgets if b not in LEVELS]
        assert not bad, f"--effort: budgets {bad} are not effort levels {LEVELS}"
    resume_state = None
    if a.resume:  # the weights and the training state (`train_state`) of a checkpoint
        model, tok, extra = load_ckpt(a.resume, dev)
        assert not a.effort or has_effort(tok), "--resume --effort: no effort tokens in the ckpt"
        resume_state = load_train_state(a.resume, dev)
        assert resume_state is not None, f"--resume {a.resume}: saved without a training state"
        for k in ("grpo_alarms", "grpo_rollback"):
            extra.pop(k, None)
    else:
        if a.effort:
            _, t0_, _ = load_ckpt(init)
            if not has_effort(t0_):
                init = str(expand_to_effort(init, out / "ckpts" / f"{a.name}_it0.pt", seed=a.seed))
        model, tok, extra = load_ckpt(init, dev)
    think = has_think(tok) and extra.get("think_format", "think") != "none"
    think_ids = (tok.think, tok.end_think) if think else None
    # the sampler's masks: think_generate bans think tokens in R's actions (one block per proof)
    bans = phase_bans(tok, model.cfg["vocab"], think, think and sample_format(model) == "root")
    targets = read_jsonl(a.targets)
    rng = random.Random(a.seed * 7919)
    cur = Curriculum(
        len(targets), budgets if think else [0], a.floor, a.decay, a.seed, prior or "half"
    )
    curs = {"main": cur}
    tx = read_jsonl(a.targets_extra) if a.targets_extra else []
    n_x = round(a.targets_extra_frac * a.prompts) if tx else 0
    assert n_x < a.prompts, "--targets_extra_frac leaves no --targets group"
    if tx:
        curs["extra"] = Curriculum(
            len(tx),
            budgets if think else [0],
            a.floor,
            a.decay,
            a.seed + 7_000_003,
            prior or "half",
        )
    kl_rng = random.Random(a.seed * 7919 + 1)  # the anchor's draws; the batch shuffle keeps `rng`
    rngs = {"batch": rng, "kl": kl_rng}
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    history = []
    ls = LoopState()
    can: list[Any] | None = None  # the canary's steps
    pool: list[Any] = []  # the anchor's steps
    if a.canary or a.kl_anchor_coef:
        assert sample_format(model) == "think", "--canary / --kl_anchor: per-step formats only"
        assert stateview.current() == "flat", "--canary / --kl_anchor: the file holds flat states"
    if a.canary:
        cd = load_canary(a.canary, tok, think, a.canary_n)
        can, pool = cd.canary, cd.anchor
        print(f"canary: {len(can)} steps of {cd.n_proofs} proofs ({a.canary})", flush=True)
    ref = None
    if a.kl_anchor_coef:
        if a.kl_anchor_data and a.kl_anchor_data != a.canary:
            pool = load_canary(a.kl_anchor_data, tok, think, 0).anchor
        assert pool, "--kl_anchor_coef: no steps for the anchor"
        ref, rtok, _ = load_ckpt(a.kl_anchor_ref, dev)
        assert rtok.mode == tok.mode and ref.cfg["vocab"] == model.cfg["vocab"], (
            f"--kl_anchor_ref {a.kl_anchor_ref}: tokenizer {rtok.mode}, the policy's {tok.mode}"
        )
        ref.eval()
        ref.requires_grad_(False)
        print(
            f"kl anchor: beta {a.kl_anchor_coef:g}, pi_ref {a.kl_anchor_ref}, {len(pool)} steps, "
            f"{a.kl_anchor_n} per AdamW step",
            flush=True,
        )
    start = a.start_iter
    if resume_state is not None:
        start = restore_state(resume_state, opt, ls, rngs, curs, full=True)
        if a.rollback_on_alarm:  # the newest good state is the one resumed
            ls.last_good, ls.last_good_iter = str(a.resume), start
        moved = _trim_metrics(out, start)
        print(
            f"resumed {a.resume} at iteration {start}"
            + (f"; {moved} later rows moved to metrics.pre_resume.jsonl" if moved else ""),
            flush=True,
        )
    last = a.start_iter + a.iters
    ckdir = out / "ckpts"
    # --refill: the run's sampled compute (rolled-back iterations included; a resume carries it)
    compute: dict[str, int] | None = None
    if refill:
        compute = {"groups": 0, "attempts": 0, "tokens": 0}
        if resume_state is not None:
            compute.update(resume_state.get("compute") or {})

    def snapshot(i: int) -> dict[str, Any]:
        return train_state(i, ls, opt, rngs, curs, compute)

    if can is not None and ls.canary_base is None:
        model.eval()
        ls.canary_base = canary_metrics(model, can, tok, bans[1], dev, a.max_tokens)
        base_row = {"iter": start, "ckpt": a.resume or init, **ls.canary_base}
        (out / "canary_base.json").write_text(json.dumps(base_row, indent=1) + "\n")
        print(_canary_line(start, ls.canary_base) + " (base)", flush=True)
    if a.rollback_on_alarm and ls.last_good is None:  # something to roll back to from the start
        p = ckdir / f"{a.name}_it{start}_rolling.pt"
        meta0 = {**extra, "grpo_iter": start, "grpo_init": a.init, "args": vars(a)}
        save_ckpt(str(p), model, tok.mode, extra=meta0, train_state=snapshot(start))
        ls.last_good, ls.last_good_iter = str(p), start

    def grp(
        att: Sequence[dict[str, Any]],
        src_g: list[tuple[int, str]],
        signal: list[bool] | None = None,
        unlikely: Sequence[float] | None = None,
    ) -> tuple[list[Item] | None, list[bool]]:
        return group_items(
            att,
            tok,
            a.len_penalty,
            a.overlong,
            a.len_target_coef,
            a.len_target_mode,
            a.len_target_frac,
            a.rej_penalty,
            a.rej_penalty_scope,
            src_g,
            signal,
            unlikely,
        )

    def n_unseen(ks: Sequence[tuple[int, int]], ss: Sequence[int]) -> int:
        """--refill's rows: how many of the pairs no group has been drawn for yet."""
        cs = [curs["extra" if s else "main"] for s in ss]
        return sum(c.n[c.index[k]] == 0 for c, k in zip(cs, ks, strict=True))

    for it in range(start + 1, last + 1):
        t0 = time.time()
        # --unseen_prior: the weights this iteration's first draw uses
        pstats = {k: c.prior_stats() for k, c in curs.items()} if prior else {}
        pairs = cur.draw(a.prompts - n_x, max_pass=max_pass)
        pairs_x = curs["extra"].draw(n_x, max_pass=max_pass) if n_x else []
        assert pairs or pairs_x, f"--max_pass {max_pass}: no pair left to draw"
        keys = pairs + pairs_x
        src = [0] * len(pairs) + [1] * len(pairs_x)  # 1: --targets_extra
        batch = [targets[t] for t, _ in pairs] + [tx[t] for t, _ in pairs_x]
        tb = [b for _, b in keys]
        # --refill: per round the pairs drawn, those with an outcome signal, those never drawn
        rinfo: dict[str, list[int]] = {"drawn": [len(keys)], "signal": [], "unseen": []}
        if refill or prior:
            rinfo["unseen"].append(n_unseen(keys, src))
        model.eval()
        D.record.phase("sample", round=it)
        samp_st: dict[str, Any] | None = {} if a.monitor else None
        per = sample_groups(model, tok, batch, tb, a, a.seed * 100003 + it, think, samp_st)
        items: list[Item] = []
        item_src: list[tuple[int, int, str]] = []  # per item: (group, attempt, kind)
        skipped: list[bool] = []  # per group
        n_skip = 0
        by_b: dict[int, list[float]] = collections.defaultdict(list)  # solved, per attempt
        over_b: dict[int, list[float]] = collections.defaultdict(list)  # masked as overlong
        think_b: dict[int, list[float]] = collections.defaultdict(list)  # think tokens
        dumping = bool(a.dump_batches)
        att_meta: list[dict[str, Any]] = []  # --dump_batches: per attempt, per group, per item
        grp_meta: list[dict[str, Any]] = []
        item_att: list[int] = []
        r_ids = (tok.think, tok.end_think) if a.len_target_coef and has_think(tok) else None
        sig: list[bool] = []  # per group: its outcomes give a signal
        # per trained group: (group, its first item, one past its last, its first dump attempt
        # row or -1, its attempts masked as overlong): the unlikeliness reward rewrites its items
        spans: list[tuple[int, int, int, int, list[bool]]] = []
        n_sig = n_surplus = 0  # trained groups with a signal; --refill: groups past the target
        rnd = r_at = 0  # the sampling round (0, then --refill's), and its first group
        while True:
            for g in range(r_at, len(per)):
                att, b, key, s = per[g], tb[g], keys[g], src[g]
                by_b[b] += [float(bool(x.get("ok"))) for x in att]
                think_b[b] += [float(x["think"]) for x in att]
                # the curriculum counts every attempt: a capped one is a failure at eval
                curs["extra" if s else "main"].update(
                    key, sum(bool(x.get("ok")) for x in att), len(att)
                )
                src_g: list[tuple[int, str]] = []
                its, masked = grp(att, src_g, sig)
                surplus = rnd > 0 and n_sig >= a.prompts  # --refill: the target is reached
                skipped.append(its is None or surplus)
                over_b[b] += [float(mk) for mk in masked]
                a0 = -1
                if dumping:
                    rw = rewards(
                        att,
                        a.len_penalty,
                        a.len_target_coef,
                        a.len_target_mode,
                        r_ids,
                        a.len_target_frac,
                    )
                    a0 = len(att_meta)
                    att_meta += [
                        {
                            "group": g,
                            "status": x.get("status"),
                            "ok": x.get("ok"),
                            "reward": r,
                            "masked": m,
                        }
                        for x, r, m in zip(att, rw, masked, strict=True)
                    ]
                    grp_meta.append(
                        {
                            "budget": b,
                            "target": key[0],
                            "source": s,
                            "skipped": its is None or surplus,
                            **({"round": rnd, "surplus": surplus} if refill else {}),
                        }
                    )
                    if its is not None and not surplus:
                        item_att += [a0 + j for j, _ in src_g]
                if its is None:
                    n_skip += 1
                    continue
                if surplus:
                    n_surplus += 1
                    continue
                n_sig += sig[g]
                spans.append((g, len(items), len(items) + len(its), a0, masked))
                items += its
                item_src += [(g, j, kind) for j, kind in src_g]
            if not refill:
                break
            rinfo["signal"].append(sum(sig[r_at:]))
            if rnd >= a.refill_max or n_sig >= a.prompts:
                break
            # the next round: enough pairs for the deficit at this iteration's signal share so far
            q = sum(sig) / len(per)
            m = a.prompts if q <= 0 else min(a.prompts, math.ceil((a.prompts - n_sig) / q))
            m_x = round(a.targets_extra_frac * m) if tx else 0
            new = cur.draw(m - m_x, {cur.index[k] for k in pairs}, max_pass)
            new_x = (
                curs["extra"].draw(m_x, {curs["extra"].index[k] for k in pairs_x}, max_pass)
                if m_x
                else []
            )
            if not new and not new_x:  # every pair is drawn or filtered
                break
            rnd, r_at = rnd + 1, len(per)
            k_r, s_r = new + new_x, [0] * len(new) + [1] * len(new_x)
            b_r = [targets[t] for t, _ in new] + [tx[t] for t, _ in new_x]
            tb_r = [b for _, b in k_r]
            rinfo["drawn"].append(len(k_r))
            rinfo["unseen"].append(n_unseen(k_r, s_r))
            seed_r = a.seed * 100003 + it + rnd * 1_000_003  # apart from every round 0's
            per = per + sample_groups(model, tok, b_r, tb_r, a, seed_r, think, samp_st)
            pairs, pairs_x = pairs + new, pairs_x + new_x
            keys, src, batch, tb = keys + k_r, src + s_r, batch + b_r, tb + tb_r
        t_sample = time.time() - t0
        D.record.phase("update", round=it)
        old = old_logps(model, items, a.temperature, dev, a.max_tokens, think_ids, bans)
        ul: dict[str, Any] | None = None
        if beta > 0:  # the unlikeliness reward: the advantages from pi_old's ranks
            ul = unlikely_rewrite(per, sig, spans, items, item_src, old, beta, grp)
            spans_u, pens_u = ul.pop("spans"), ul.pop("pens")
            if dumping:  # the attempts' rewards as trained
                for (_, _, _, a0, _), pen in zip(spans_u, pens_u, strict=True):
                    for j, u in enumerate(pen):
                        att_meta[a0 + j]["reward"] -= u
                        att_meta[a0 + j]["unlikely"] = u
        model.train()
        order = list(range(len(items)))
        rng.shuffle(order)
        adam0 = compact_adam(opt, model) if dumping else {}
        stats: list[LossStats] = []
        gns: list[float] = []
        kls: list[float] = []  # the anchor's KL, per AdamW step
        gns_pol: list[float] = []  # the grad norm before the anchor's gradient
        steps_meta: list[dict[str, float]] = []
        # --lr_schedule: every step of iteration t at base x m(t) x the rollback scale x the step
        # ramp (`sched_lr`). t follows the loop's iteration number, which a rollback does not move
        # back, so the last iteration is at m(T). Set here too, so the row's lr is this iteration's
        # (its next step's) when it takes no step.
        mult = 1.0 if sched is None else lr_mult(it - a.start_iter, sched)
        n_lr = ls.n_step + 1  # the step the optimizer's lr is set for
        if sched is not None:
            lr_it = sched_lr(a.lr, mult, ls.lr_scale, n_lr, sched["warmup_steps"])
            for g in opt.param_groups:
                g["lr"] = lr_it
        for u in range(a.updates):
            sel = order[u :: a.updates]
            if not sel:
                continue
            ls.n_step += 1
            if sched is None:  # constant: --lr after the --warmup ramp over AdamW steps
                for g in opt.param_groups:
                    g["lr"] = a.lr * ls.lr_scale * min(1.0, ls.n_step / max(1, a.warmup))
            else:
                n_lr = ls.n_step
                lr_it = sched_lr(a.lr, mult, ls.lr_scale, n_lr, sched["warmup_steps"])
                for g in opt.param_groups:
                    g["lr"] = lr_it
            opt.zero_grad(set_to_none=True)
            st = policy_loss(
                model,
                [items[k] for k in sel],
                a.temperature,
                dev,
                a.max_tokens,
                old=[old[k] for k in sel],
                eps_low=a.clip_eps_low,
                eps_high=a.clip_eps_high,
                think_ids=think_ids,
                bans=bans,
                dual_clip=a.dual_clip,
                rejected=[item_src[k][2] == "rej" for k in sel] if a.rej_penalty else None,
            )
            if st.tokens:
                kl = 0.0
                if ref is not None:
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    gns_pol.append(float(torch.nn.utils.get_total_norm(grads)))
                    pick = kl_rng.sample(range(len(pool)), min(a.kl_anchor_n, len(pool)))
                    kl = anchor_kl(
                        model,
                        ref,
                        [pool[i] for i in pick],
                        a.temperature,
                        bans[1],
                        dev,
                        a.max_tokens,
                        a.kl_anchor_coef,
                    )
                    kls.append(kl)
                gns.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)))
                opt.step()
                stats.append(st)
                steps_meta.append(
                    {"lr": opt.param_groups[0]["lr"], "grad_norm": gns[-1], "kl_anchor": kl}
                )
        n_tok = sum(s.tokens for s in stats)
        n_st = max(1, len(stats))  # means over the minibatch steps
        loss = sum(s.loss for s in stats) / n_st
        clip_frac = sum(s.clip_frac for s in stats) / n_st
        clip_low = sum(s.clip_low for s in stats) / n_st
        clip_high = sum(s.clip_high for s in stats) / n_st
        gn = sum(gns) / max(1, len(gns))
        entropy = sum(s.entropy * s.tokens for s in stats) / max(1, n_tok)  # per scored token
        n_th = sum(s.tokens_think for s in stats)
        ent_think = sum(s.entropy_think * s.tokens_think for s in stats) / max(1, n_th)
        ent_action = sum(s.entropy_action * (s.tokens - s.tokens_think) for s in stats) / max(
            1, n_tok - n_th
        )
        ban_think = sum(s.ban_think * s.tokens_think for s in stats) / max(1, n_th)
        n_neg = sum(s.tokens_neg for s in stats)
        gn_max = max(gns, default=0.0)
        flat = [x for att in per for x in att]
        row: dict[str, Any] = {
            "iter": it,
            "reward_pass1": sum(bool(x.get("ok")) for x in flat) / max(1, len(flat)),
            "pass1_by_budget": {str(b): sum(v) / len(v) for b, v in sorted(by_b.items())},
            "solved_frac": sum(any(x.get("ok") for x in att) for att in per) / len(per),
            "groups_skipped": n_skip / len(per),
            "tokens_per_attempt": sum(x["tokens"] for x in flat) / max(1, len(flat)),
            "think_per_attempt": sum(x["think"] for x in flat) / max(1, len(flat)),
            "think_by_budget": {str(b): sum(v) / len(v) for b, v in sorted(think_b.items())},
            **length_metrics(
                per, tb, think_ids, a.len_target_mode, a.len_target_coef, a.len_target_frac
            ),
            "overlong_frac": sum(map(sum, over_b.values())) / max(1, len(flat)),  # masked
            "overlong_by_budget": {str(b): sum(v) / len(v) for b, v in sorted(over_b.items())},
            "items": len(items),
            "trained_tokens": n_tok,
            "loss": loss,
            "clip_frac": clip_frac,  # mean over the minibatch steps
            "clip_frac_max": max((s.clip_frac for s in stats), default=0.0),
            "clip_low": clip_low,
            **(
                {
                    "clip_low_accepted": sum(s.clip_low_acc for s in stats) / n_st,
                    # the loss's normaliser (accepted items' tokens) over every scored token
                    "tokens_accepted_share": sum(s.tokens_acc for s in stats) / max(1, n_tok),
                }
                if a.rej_penalty
                else {}
            ),
            "clip_high": clip_high,
            "entropy": entropy,  # nats per scored token, masked pi_T, during the update
            "entropy_think": ent_think,  # ... over the think-block tokens
            "entropy_action": ent_action,  # ... over the action tokens
            "ban_think": ban_think,  # full-vocabulary pi_T mass on the think ban, think tokens
            "ban_think_max": max((s.ban_think_max for s in stats), default=0.0),
            "ratio_max": max((s.ratio_max for s in stats), default=0.0),
            "neg_r3": sum(s.neg_r3 * s.tokens_neg for s in stats) / max(1, n_neg),
            "nonfinite": sum(s.nonfinite for s in stats),
            "grad_norm": gn,  # mean pre-clip norm over the minibatch steps
            "grad_norm_max": gn_max,
            "steps": len(gns),
            "curriculum": cur.summary(),
            "lr": opt.param_groups[0]["lr"],
            "secs": time.time() - t0,
            "secs_sample": t_sample,  # sampling and the Lean gate
            "attempts_per_s": len(flat) / max(1e-9, t_sample),
            **capped_metrics(per, tb),  # whatever --overlong does with them
            "clip_low_median": (
                statistics.median(ls.clip_prev[-ALARM_REL_WINDOW:])
                if len(ls.clip_prev) >= ALARM_GN_AFTER
                else None
            ),
            "alarm_cfg": {
                "capped": a.alarm_capped,
                "clip_low_rel": a.clip_low_rel,
                "canary_mult": a.canary_mult,
                # what a think ban alarm leads to (not alarm thresholds): `ban_check`, `stop`
                "ban_restore_check": a.ban_restore_check,
                "ban_stop": a.ban_stop,
            },
        }
        if pairs_x:
            xa = [x for att, s in zip(per, src, strict=True) if s for x in att]
            xb: dict[int, list[float]] = collections.defaultdict(list)
            for att, b, s in zip(per, tb, src, strict=True):
                if s:
                    xb[b] += [float(bool(x.get("ok"))) for x in att]
            main = [x for att, s in zip(per, src, strict=True) if not s for x in att]
            row["extra_groups"] = len(pairs_x)
            row["extra_pass1"] = sum(bool(x.get("ok")) for x in xa) / max(1, len(xa))
            row["extra_pass1_by_budget"] = {str(b): sum(v) / len(v) for b, v in sorted(xb.items())}
            row["main_pass1"] = sum(bool(x.get("ok")) for x in main) / max(1, len(main))
            row["curriculum_extra"] = curs["extra"].summary()
        if ref is not None:
            row["kl_anchor"] = sum(kls) / max(1, len(kls))  # nats per anchor position
            row["kl_anchor_coef"] = a.kl_anchor_coef
            row["grad_norm_policy"] = sum(gns_pol) / max(1, len(gns_pol))
        if a.rollback_on_alarm or ls.lr_scale != 1.0:
            row["lr_scale"], row["rollbacks"] = ls.lr_scale, ls.rollbacks
        if sched is not None:  # row['lr'] = --lr x lr_mult x lr_scale x lr_ramp
            row["lr_mult"] = mult
            row["lr_ramp"] = min(1.0, n_lr / max(1, sched["warmup_steps"]))  # the step ramp
            # a rollback's lr halving multiplies the schedule (`row['rollback']['lr_scale']`)
            row["alarm_cfg"]["lr_schedule"] = dict(sched)
        if a.monitor:
            row["monitor"] = monitor_metrics(
                per, [target_length(t) for t in batch], skipped, items, item_src, think_ids, samp_st
            )
            row["monitor"]["arm"], row["monitor"]["rej_penalty"] = a.arm, a.rej_penalty
        if refill:
            assert compute is not None
            n_att = sum(len(att) for att in per)
            n_tok_s = sum(int(x["tokens"]) for x in flat)
            compute["groups"] += len(per)
            compute["attempts"] += n_att
            compute["tokens"] += n_tok_s
            row["refill"] = {
                "rounds": rnd,
                "refill_max": a.refill_max,
                "target": a.prompts,
                **rinfo,
                "skip_before": 1.0 - rinfo["signal"][0] / max(1, rinfo["drawn"][0]),
                "skip_after": 1.0 - n_sig / a.prompts,
                "signal_groups": n_sig,  # trained, with a signal
                "trained_groups": len(spans),
                "surplus": n_surplus,
                "groups": len(per),
                "attempts": n_att,
                "tokens": n_tok_s,
                **{f"{k}_cum": v for k, v in compute.items()},
            }
        if max_pass is not None:  # the pairs the next draw leaves out
            row["max_pass"] = {"threshold": max_pass, "over": cur.over(max_pass)}
            if tx:
                row["max_pass"]["over_extra"] = curs["extra"].over(max_pass)
        if ul is not None:
            row["unlikely"] = ul
        if prior:  # the draws' unseen share (every round), the weights at the first draw
            row["prior"] = {
                "mode": prior,
                "drawn": sum(rinfo["drawn"]),
                "unseen_drawn": sum(rinfo["unseen"]),
                "unseen_drawn_share": sum(rinfo["unseen"]) / max(1, sum(rinfo["drawn"])),
                **pstats["main"],
            }
            if "extra" in pstats:
                row["prior"]["extra"] = pstats["extra"]
        if can is not None and it % a.canary_every == 0:  # the policy this iteration leaves
            t1 = time.time()
            model.eval()
            c = canary_metrics(model, can, tok, bans[1], dev, a.max_tokens)
            row["canary"] = {**c, "base": ls.canary_base, "secs": time.time() - t1}
        row["alarms"] = alarms(row, ls.gn_prev)
        if gns:
            ls.gn_prev.append(gn_max)
            ls.clip_prev.append(clip_low_basis(row)[0])  # the running median the alarm reads
        stop = a.ban_stop > 0 and row["ban_think"] > a.ban_stop  # before any rollback or check
        if stop:
            row["stop"] = {"ban_think": row["ban_think"], "ban_stop": a.ban_stop}
        rb: dict[str, Any] | None = None
        if row["alarms"] and a.rollback_on_alarm:
            if not stop and ls.rollbacks < a.max_rollbacks and ls.last_good:
                rb = {
                    "to": ls.last_good,
                    "to_iter": ls.last_good_iter,
                    "n": ls.rollbacks + 1,
                    "lr_scale": ls.lr_scale * 0.5,
                }
                if a.ban_restore_check and row["alarms"] == [ban_alarm(row)]:
                    x = restore_ban_mass(
                        model, rb["to"], items, a.temperature, dev, a.max_tokens, think_ids, bans
                    )
                    back = ban_rollback(x, row["ban_think"])
                    row["ban_check"] = {
                        "to": rb["to"],
                        "to_iter": rb["to_iter"],
                        "restore_mass": x,
                        "mass": row["ban_think"],
                        "rollback": back,
                    }
                    if not back:
                        rb = None
            row["rollback"] = rb  # None: --max_rollbacks reached, the restore check, or the stop
        # an alarm the restore check let through: checkpointed as no alarm (below)
        passed = "ban_check" in row and not row["ban_check"]["rollback"]
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        history.append(row)
        print(
            f"it {it}: pass@1 {100 * row['reward_pass1']:.1f}% "
            f"{ {k: round(100 * v, 1) for k, v in row['pass1_by_budget'].items()} } "
            f"skip {100 * row['groups_skipped']:.0f}% tok/att {row['tokens_per_attempt']:.0f} "
            f"over {100 * row['overlong_frac']:.1f}% cap {100 * row['capped_frac']:.1f}% "
            f"think/att {row['think_per_attempt']:.0f} "
            f"{_length_line(row)} "
            f"loss {loss:+.4f} clip lo/hi {clip_low:.4f}/{clip_high:.4f} "
            f"rmax {row['ratio_max']:.2f} neg_r3 {row['neg_r3']:.4f} "
            f"ent {ent_think:.2f}/{ent_action:.2f} "
            f"ban {ban_think:.1e}/{row['ban_think_max']:.1e} gn {gn:.3f}/{gn_max:.3f} "
            f"frontier {row['curriculum']['frontier']:.2f} {row['secs']:.0f}s "
            f"({row['attempts_per_s']:.0f} att/s sampling)",
            flush=True,
        )
        if "canary" in row:
            print(_canary_line(it, row["canary"]), flush=True)
        if "monitor" in row:
            print(_monitor_line(it, row["monitor"]), flush=True)
        for line in _pilot_lines(it, row):
            print(line, flush=True)
        guard = []
        if "kl_anchor" in row:
            guard.append(
                f"kl anchor {row['kl_anchor']:.4f} x {a.kl_anchor_coef:g} "
                f"(gn policy {row['grad_norm_policy']:.3f})"
            )
        if "extra_groups" in row:
            guard.append(f"extra {row['extra_groups']} groups {100 * row['extra_pass1']:.1f}%")
        if "lr_scale" in row:
            guard.append(f"lr x{row['lr_scale']:g} ({row['rollbacks']} rollbacks)")
        if sched is not None:  # the effective lr; the entry above keeps its text (line.sh's _b)
            ramp = f" x ramp {row['lr_ramp']:.2f}" if row["lr_ramp"] < 1 else ""
            guard.append(
                f"lr {row['lr']:.3g} = {a.lr:g} x {sched['kind']} {row['lr_mult']:.4f} "
                f"x {ls.lr_scale:g}{ramp}"
            )
        if guard:
            cap_b = " ".join(f"B{b} {100 * v:.1f}%" for b, v in row["capped_by_budget"].items())
            print(f"guard it {it}: " + "; ".join(guard) + f"; capped {cap_b}", flush=True)
        for msg in row["alarms"]:
            print(f"ALARM it {it}: {msg}", flush=True)
        if "ban_check" in row:
            print(_ban_check_line(it, row["ban_check"]), flush=True)
        elif row["alarms"] and a.rollback_on_alarm and rb is None and not stop:
            print(f"ALARM it {it}: no rollback (--max_rollbacks {a.max_rollbacks} reached)")
        if dumping:  # before the checkpoints: a checkpoint of iteration i implies its dump
            assert len(item_att) == len(items), "dump: items vs attempts"
            dump_batch(
                Path(a.dump_batches) / f"{a.name}_it{it}_batch.pt",
                it=it,
                items=items,
                old=old,
                item_attempt=item_att,
                order=order,
                updates=a.updates,
                attempts=att_meta,
                groups=grp_meta,
                steps=steps_meta,
                adam=adam0,
                settings={
                    k: getattr(a, k)
                    for k in (
                        "temperature",
                        "clip_eps_low",
                        "clip_eps_high",
                        "dual_clip",
                        "clip",
                        "max_tokens",
                        "len_penalty",
                        "overlong",
                        "len_target_coef",
                        "len_target_mode",
                        "len_target_frac",
                        "kl_anchor_coef",
                    )
                },
                vocab=model.cfg["vocab"],
            )
        model.eval()
        meta = {**extra, "grpo_iter": it, "grpo_init": a.init, "args": vars(a)}
        saved = False
        # An alarm the restore check let through is checkpointed like an iteration without alarms:
        # no alarm checkpoint, and its rolling or permanent checkpoint is good (`ls.last_good`).
        # Its restore point was over the limit too, so the next check compares with these weights
        # and a later rollback restores them rather than older ones. A stop saves the alarm
        # checkpoint (its weights and training state) and nothing else.
        alarmed = (bool(row["alarms"]) and not passed) or stop
        if alarmed:
            save_ckpt(
                str(ckdir / f"{a.name}_it{it}_alarm.pt"),
                model,
                tok.mode,
                extra={**meta, "grpo_alarms": row["alarms"]},
                train_state=snapshot(it),
            )
            saved = True
        if stop:
            print(
                f"STOP it {it}: think banned mass {row['ban_think']:.3g} > {a.ban_stop:g}: "
                "ending the run",
                flush=True,
            )
            break
        if rb is not None:  # the alarm checkpoint holds this iteration's weights; restore
            ck = torch.load(str(rb["to"]), map_location=dev)
            model.load_state_dict(ck["state"])
            restore_state(ck[TRAIN_STATE], opt, ls, rngs, curs, full=False)
            ls.lr_scale, ls.rollbacks = rb["lr_scale"], rb["n"]
            meta = {**meta, "grpo_rollback": rb}
            print(
                f"ROLLBACK it {it}: restored {rb['to']} (it {rb['to_iter']}), "
                f"lr x{ls.lr_scale:g} from the next iteration ({ls.rollbacks}/{a.max_rollbacks})",
                flush=True,
            )
        good = not alarmed or rb is not None
        if it % a.save_every == 0 or it == last:
            p = ckdir / f"{a.name}_it{it}.pt"
            save_ckpt(str(p), model, tok.mode, extra=meta, train_state=snapshot(it))
            saved = True
            if good:
                ls.last_good, ls.last_good_iter = str(p), it
        if a.save_rolling and it % a.save_rolling == 0 and not saved:
            p = ckdir / f"{a.name}_it{it}_rolling.pt"
            save_ckpt(str(p), model, tok.mode, extra=meta, train_state=snapshot(it))
            prune_rolling(ckdir, a.name, a.keep_rolling)
            ls.last_good, ls.last_good_iter = str(p), it
        if dev == "cuda":
            torch.cuda.empty_cache()
    D.record.phase(None)
    return history


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--init", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--budgets", default="16,64,256", help="drawn per target (one per group)")
    ap.add_argument("--effort", action="store_true", help="show the budget (effort.py)")
    ap.add_argument("--iters", type=int, default=280)  # 280 x 256 x 16 ~ EI's 8 x 4,495 x 32
    ap.add_argument("--prompts", type=int, default=256, help="(target, budget) pairs / iter")
    ap.add_argument("--group", type=int, default=16, help="attempts per target")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--updates", type=int, default=4, help="minibatch steps per sample batch")
    ap.add_argument("--clip_eps", type=float, default=None, help="sets both (shortcut)")
    ap.add_argument("--clip_eps_low", type=float, default=None, help="default 0.2")
    ap.add_argument("--clip_eps_high", type=float, default=None, help="default 0.28 (DAPO)")
    ap.add_argument("--dual_clip", type=float, default=3.0, help="A < 0: surrogate >= c A; 0: off")
    ap.add_argument("--floor", type=float, default=0.03, help="curriculum weight floor")
    ap.add_argument("--decay", type=float, default=0.7, help="curriculum count decay")
    ap.add_argument(
        "--warmup", type=int, default=10, help="lr ramp over AdamW steps (off with --warmup_iters)"
    )
    ap.add_argument("--clip", type=float, default=1.0)
    s = ap.add_argument_group("learning-rate schedule (opt-in; module docstring)")
    s.add_argument(
        "--lr_schedule",
        choices=LR_SCHEDULES,
        default=argparse.SUPPRESS,
        help="per iteration of the run: constant (default; --lr after the --warmup step ramp), "
        "cosine, or wsd (warmup-stable-decay); a rollback's halving multiplies it",
    )
    s.add_argument(
        "--warmup_iters",
        type=int,
        default=argparse.SUPPRESS,
        help="cosine / wsd: linear warmup over this many iterations instead of --warmup's "
        "step ramp (default 0: the step ramp)",
    )
    s.add_argument(
        "--min_lr_frac",
        type=float,
        default=argparse.SUPPRESS,
        help="cosine / wsd: the lr at the run's last iteration, x --lr (default 0.1)",
    )
    s.add_argument(
        "--decay_frac",
        type=float,
        default=argparse.SUPPRESS,
        help="wsd: the share of the run's iterations spent decaying (default 0.2)",
    )
    s.add_argument(
        "--decay_shape",
        choices=DECAY_SHAPES,
        default=argparse.SUPPRESS,
        help="wsd: the decay, linear (default), cosine or sqrt (1 - sqrt; Hägele et al. 2024)",
    )
    ap.add_argument("--len_penalty", type=float, default=0.0, help="reward per 1k tokens")
    ap.add_argument(
        "--len_target_coef",
        type=float,
        default=0.0,
        help="LCPO length target: every attempt at B > 0 pays alpha x the mean think-block "
        "deviation from its target (relative, each block clipped at 1); 0: off",
    )
    ap.add_argument(
        "--len_target_mode",
        choices=LEN_TARGET_MODES,
        default="exact",
        help="'exact': |L_b - T| / T; 'max': max(0, L_b - T) / T (overshoot only)",
    )
    ap.add_argument(
        "--len_target_frac",
        type=float,
        default=1.0,
        help="the length target T = frac x B per effort level",
    )
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling and log-probs")
    ap.add_argument(
        "--overlong",
        choices=("mask", "fail"),
        default="mask",
        help="attempts a cap ended (OVERLONG): no gradient, out of the baseline; or failures",
    )
    ap.add_argument(
        "--force_mode",
        choices=FORCE_MODES,
        default="line",
        help="budget forcing (sample.py): 'line' finishes the think line past B (cap B + 64); "
        "'exact' closes at B",
    )
    ap.add_argument("--max_action", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=48)
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=50_000,
        help="padded tokens per micro-batch (memory only; 200k peaked near 8 GB, 50k was faster)",
    )
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--save_every", type=int, default=35)
    ap.add_argument("--save_rolling", type=int, default=5, help="rolling checkpoints; 0: none")
    ap.add_argument("--keep_rolling", type=int, default=4, help="rolling checkpoints kept")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lean", default=None)
    ap.add_argument("--device", default=None)
    g = ap.add_argument_group("safeguards (collapse audit 2026-10-08; module docstring)")
    g.add_argument(
        "--resume",
        default=None,
        help="a checkpoint of this script with its training state: continue the run exactly "
        "(to iteration --start_iter + --iters); --init stays a weight-only start",
    )
    g.add_argument(
        "--start_iter", type=int, default=0, help="iterations run start_iter + 1 .. + --iters"
    )
    g.add_argument(
        "--alarm_capped",
        type=float,
        default=ALARM_CAPPED,
        help="alarm when a budget's share of capped attempts exceeds this (0: off)",
    )
    g.add_argument(
        "--clip_low_rel",
        type=float,
        default=ALARM_CLIP_LOW_REL,
        help="alarm when clip-low exceeds this x its running median (0: off)",
    )
    g.add_argument("--canary", default=None, help="canary/1 file (canary.py): long-state steps")
    g.add_argument("--canary_n", type=int, default=1000, help="canary steps")
    g.add_argument("--canary_every", type=int, default=1, help="iterations between canaries")
    g.add_argument(
        "--canary_mult",
        type=float,
        default=CANARY_MULT,
        help="alarm when a canary value exceeds this x its base (0: off)",
    )
    g.add_argument(
        "--rollback_on_alarm",
        action="store_true",
        help="on an alarm restore the last good checkpoint and halve the lr",
    )
    g.add_argument("--max_rollbacks", type=int, default=3)
    g.add_argument(
        "--ban_restore_check",
        action="store_true",
        help="a think ban alarm alone does not roll back when the restore point's mass on the "
        "iteration's think tokens is over the limit too and the iteration's is under "
        "BAN_SPIKE (2) x it",
    )
    g.add_argument(
        "--ban_stop",
        type=float,
        default=0.0,
        help="end the run (exit code 3) when the think ban mass exceeds this (0: off)",
    )
    g.add_argument(
        "--kl_anchor_coef",
        type=float,
        default=0.0,
        help="beta: add beta x KL(pi_ref || pi) on long-state reference steps (0: off)",
    )
    g.add_argument("--kl_anchor_ref", default=None, help="pi_ref: a frozen checkpoint")
    g.add_argument(
        "--kl_anchor_data",
        default=None,
        help="canary/1 file of the anchor's steps (default: --canary minus the canary's proofs)",
    )
    g.add_argument("--kl_anchor_n", type=int, default=512, help="anchor steps per AdamW step")
    g.add_argument("--dump_batches", default=None, help="DIR: per-iteration batch + AdamW dump")
    v = ap.add_argument_group("verifier-guided sampling and the step penalty (module docstring)")
    v.add_argument(
        "--arm",
        choices=ARMS,
        default="plain",
        help="the sampler's arm: plain ends an attempt at a rejected draw; structural redraws "
        "what the environment rejects; logical also what step_check rejects (sound w.r.t. Lean)",
    )
    v.add_argument("--max_rej", type=int, default=10, help="guided arms: redraws per attempt")
    v.add_argument(
        "--rej_penalty",
        type=float,
        default=0.0,
        help="P: every rejected draw gets advantage -P whatever the outcome (0: off)",
    )
    v.add_argument(
        "--rej_penalty_scope",
        choices=REJ_SCOPES,
        default="diverge",
        help="what P scores in a redrawn draw: from where it departs from the draw accepted at its "
        "state (diverge), its think block and action (draft), or its action only",
    )
    v.add_argument(
        "--monitor",
        action="store_true",
        help="log where the gradient comes from (row['monitor'], `monitor_metrics`)",
    )
    g.add_argument("--targets_extra", default=None, help="a second target file (long proofs)")
    g.add_argument(
        "--targets_extra_frac",
        type=float,
        default=0.0,
        help="share of each iteration's groups drawn from --targets_extra",
    )
    # left out of the namespace (and args.json) unless given: `pilot_options`
    pg = ap.add_argument_group("pilot options (opt-in; module docstring)")
    pg.add_argument(
        "--refill",
        action="store_true",
        default=argparse.SUPPRESS,
        help="draw and sample more pairs until --prompts groups have an outcome signal "
        "(dynamic sampling), at most --refill_max extra rounds",
    )
    pg.add_argument(
        "--refill_max",
        type=int,
        default=argparse.SUPPRESS,
        help=f"--refill: extra sampling rounds per iteration (default {REFILL_MAX})",
    )
    pg.add_argument(
        "--max_pass",
        type=float,
        default=argparse.SUPPRESS,
        help="do not draw pairs whose pass-rate estimate is above this (0.5 - 1)",
    )
    pg.add_argument(
        "--unlikely_beta",
        type=float,
        default=argparse.SUPPRESS,
        help="unlikeliness reward (He et al. 2025): a solved attempt's reward is 1 - beta x its "
        "likelihood rank / n in its group (default 0: off; 0.25 in the paper)",
    )
    pg.add_argument(
        "--unseen_prior",
        choices=UNSEEN_PRIORS,
        default=argparse.SUPPRESS,
        help="the pass-rate prior: half (default: Beta(1, 1), an unseen pair at 1/2) or mean "
        "(the budget's mean pass rate over its seen pairs, as Beta(2 m, 2 (1 - m)))",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    hist = run(parser().parse_args(argv))
    return STOP_EXIT if hist and "stop" in hist[-1] else 0  # --ban_stop ended the run


if __name__ == "__main__":
    sys.exit(main())
