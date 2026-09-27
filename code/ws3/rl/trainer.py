"""`WS3GRPOTrainer`: TRL's GRPOTrainer plus three hooks, none of which change TRL's loss.

1. Advantage transform (`advantage: {name, ...params}`, see `rl/advantages.py`). TRL computes
   advantages inline in `_generate_and_score_completions`; we record the raw rewards in
   `_calculate_rewards` and replace `advantages` in its output. `name: trl` leaves TRL's own.
2. Prompt statistics (`rl/curriculum.py`). Every scored group updates a `PromptStats`, which the
   adaptive sampler reads to pick the next prompts. With `curriculum` unset the stock
   `RepeatSampler` is used and the stats are only logged.
3. Extra metrics, into TRL's logs and so into metrics.jsonl: all-fail / all-success / mixed
   group fractions, cumulative completions (for matched-budget comparisons with expert
   iteration), the advantage mean and spread, and the curriculum's state.
"""

from __future__ import annotations

import torch
from accelerate.utils import gather_object
from trl import GRPOTrainer

# isort: split
from rl import advantages as A
from rl.curriculum import AdaptiveRepeatSampler, PromptStats


class WS3GRPOTrainer(GRPOTrainer):
    def __init__(self, *args, advantage=None, curriculum=None, prompt_stats=None, **kwargs):
        adv = dict(advantage or {"name": "trl"})
        self.ws3_adv_name = adv.pop("name", "trl")
        if self.ws3_adv_name not in A.NAMES:
            raise ValueError(f"unknown advantage {self.ws3_adv_name!r}; choose from {A.NAMES}")
        self.ws3_adv_params = adv
        self.ws3_curriculum = curriculum
        self.prompt_stats = prompt_stats if prompt_stats is not None else PromptStats()
        self.ws3_total_completions = 0
        self._ws3_last = None
        super().__init__(*args, **kwargs)
        if self.ws3_adv_name == "unlikely" and self.accelerator.num_processes > 1:
            raise NotImplementedError("the unlikely transform is single-process only")

    # ------------------------------------------------------------------ prompt selection

    def _get_train_sampler(self, dataset=None):
        if not self.ws3_curriculum:
            return super()._get_train_sampler(dataset)
        dataset = dataset if dataset is not None else self.train_dataset
        return AdaptiveRepeatSampler(
            prompts=list(dataset["prompt"]),
            lengths=list(dataset["length"]),
            stats=self.prompt_stats,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            seed=self.args.seed,
            **self.ws3_curriculum,
        )

    # ------------------------------------------------------------------ rewards and advantages

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = super()._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )
        # rewards_per_func is already gathered across processes; gather the prompt texts to match.
        local = [x["prompt"] for x in inputs]
        all_prompts = gather_object(local) if self.accelerator.num_processes > 1 else local
        self._ws3_last = (rewards_per_func, all_prompts, len(local))
        return rewards_per_func

    def _sequence_logps(self, out):
        ids = torch.cat([out["prompt_ids"], out["completion_ids"]], dim=1)
        mask = torch.cat([out["prompt_mask"], out["completion_mask"]], dim=1)
        with torch.no_grad():
            logps, _, _ = self._get_per_token_logps_and_entropies(
                self.model,
                ids,
                mask,
                out["completion_ids"].size(1),
                self.args.per_device_train_batch_size,
            )
        return (logps * out["completion_mask"]).sum(dim=1)

    def _generate_and_score_completions(self, inputs):
        mode = "train" if self.model.training else "eval"
        out = super()._generate_and_score_completions(inputs)
        if mode != "train" or self._ws3_last is None:
            return out
        rewards_per_func, all_prompts, n_local = self._ws3_last
        self._ws3_last = None
        device = rewards_per_func.device
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        rewards[torch.isnan(rewards_per_func).all(dim=1)] = torch.nan
        g = self.num_generations
        metrics = self._metrics[mode]

        self.ws3_total_completions += rewards.numel()
        metrics["rollout/total_completions"].append(float(self.ws3_total_completions))
        for k, v in A.group_outcomes(rewards, g).items():
            metrics[k].append(v)
        succ = A.success(torch.nan_to_num(rewards, nan=0.0)).view(-1, g)
        self.prompt_stats.update([all_prompts[i * g] for i in range(succ.shape[0])], succ.tolist())
        for k, v in self.prompt_stats.summary().items():
            metrics[k].append(v)

        if self.ws3_adv_name != "trl":
            seq_logps = self._sequence_logps(out) if self.ws3_adv_name == "unlikely" else None
            adv = A.compute(
                self.ws3_adv_name,
                rewards.cpu(),
                g,
                self.ws3_adv_params,
                seq_logps=seq_logps.cpu() if seq_logps is not None else None,
            )
            i = self.accelerator.process_index
            local = adv[i * n_local : (i + 1) * n_local]
            old = out["advantages"]
            out["advantages"] = local.to(device=old.device, dtype=old.dtype)
            metrics["adv/mean"].append(adv.mean().item())
            metrics["adv/abs_mean"].append(adv.abs().mean().item())
            metrics["adv/frac_nonzero"].append((adv != 0).float().mean().item())
        return out
