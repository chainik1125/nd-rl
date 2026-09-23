"""Callbacks that write the plan section 4 metric set to metrics.jsonl, one JSON line per step.

TRL's GRPO/RLOO trainers already log: reward, reward_std, kl, entropy, frac_reward_zero_std,
completions/{mean,max,min}_length, completions/clipped_ratio (fraction hitting the cap),
clip_ratio/*, num_tokens, plus the Trainer's loss, grad_norm, learning_rate. This module adds a
`utc` timestamp, `step`, rollout throughput (completions per second from wall-clock between
rollout logs), the reward function's unparseable / invalid fractions when TRL did not log them
itself, and a periodic greedy pass-rate-by-length eval. Everything is persisted locally; no
external logging service.
"""

from __future__ import annotations

import json
import os
import time

import ndpath  # noqa: F401
from transformers import TrainerCallback

# isort: split
from eval.evaluator import utc_now

# The plan section 4 set, as the key each metric has in metrics.jsonl.
REQUIRED_METRICS = {
    "mean reward": "reward",
    "reward std within groups": "reward_std",
    "pass rate by proof length (periodic eval)": "eval/pass_rate_len*",
    "KL to reference": "kl",
    "policy entropy": "entropy",
    "unparseable fraction": "reward/unparseable_frac",
    "invalid fraction": "reward/invalid_frac",
    "completion length mean": "completions/mean_length",
    "completion length max": "completions/max_length",
    "fraction at length cap": "completions/clipped_ratio",
    "gradient norm": "grad_norm",
    "learning rate": "learning_rate",
    "rollout throughput": "rollout/completions_per_sec",
    "utc timestamp": "utc",
    "step": "step",
}


class MetricsJsonlCallback(TrainerCallback):
    def __init__(self, path: str, completions_per_step: int, reward_fn=None):
        self.path = path
        self.cps = completions_per_step
        self.reward_fn = reward_fn
        self._t_last = None
        self._step_last = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def on_train_begin(self, args, state, control, **kwargs):
        self._t_last = time.time()
        self._step_last = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = dict(logs or {})
        now = time.time()
        is_rollout_log = "reward" in logs  # the end-of-training summary log has no rollouts
        if is_rollout_log and self._t_last is not None:
            dsteps = max(state.global_step - self._step_last, 1)
            secs = now - self._t_last
            logs["rollout/completions_per_sec"] = self.cps * dsteps / max(secs, 1e-9)
            logs["rollout/secs_per_step"] = secs / dsteps
            self._t_last, self._step_last = now, state.global_step
        if is_rollout_log and self.reward_fn is not None:
            for k, v in getattr(self.reward_fn, "last_stats", {}).items():
                logs.setdefault(k, v)
        logs["step"] = state.global_step
        logs["epoch"] = state.epoch
        logs["utc"] = utc_now()
        logs["time"] = now
        with open(self.path, "a") as f:
            f.write(json.dumps(logs) + "\n")


class PeriodicEvalCallback(TrainerCallback):
    """Every `every` steps: greedy pass rate by length on a fixed small set, into metrics.jsonl."""

    def __init__(
        self,
        path: str,
        tok,
        records,
        every: int,
        max_new: int,
        batch_size: int = 64,
        dump: str | None = None,
        label: str = "periodic_eval",
    ):
        self.path, self.tok, self.records, self.every = path, tok, records, every
        self.max_new, self.batch_size, self.dump, self.label = max_new, batch_size, dump, label
        self._done_steps = set()

    def _run(self, model, step):
        from eval.evaluator import evaluate_pool

        if step in self._done_steps:
            return None
        self._done_steps.add(step)
        res = evaluate_pool(
            model,
            self.tok,
            self.records,
            k=1,
            temperature=0.0,
            max_new=self.max_new,
            batch_size=self.batch_size,
            dump=self.dump,
            label=f"{self.label}@{step}",
        )
        row = {
            "step": step,
            "utc": utc_now(),
            "eval/pass_rate": res["overall"]["rate"],
            "eval/max_length_solved": res["max_length_solved"],
            "eval/unparseable_frac": res["unparseable_frac"],
            "eval/samples_per_sec": res["samples_per_sec"],
            "time": time.time(),
        }
        for L, v in res["by_length"].items():
            row[f"eval/pass_rate_len{L}"] = v["rate"]
            row[f"eval/solved_len{L}"] = v["solved"]
            row[f"eval/n_len{L}"] = v["n"]
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._run(model, state.global_step)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every and state.global_step % self.every == 0 and model is not None:
            self._run(model, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._run(model, state.global_step)
