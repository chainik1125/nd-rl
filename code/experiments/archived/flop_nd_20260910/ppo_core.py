"""Auditable response-token PPO and relative-outcome updates.

The policy is the unchanged vendor ProofModel. A temporary final_norm hook
exposes h_t, the state BEFORE target token a_t, to a separate learned value
head. The primary task is an entire finite-budget generation episode: both QED
and budget exhaustion terminate it and have zero bootstrap. GAE also accepts
nonterminal rollout segments, but the training CLI never silently treats a
budget-exhausted proof attempt as a segment.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

import common
import learning as L


class ValueHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.projection = nn.Linear(d_model, 1)
        # Initial V=0 makes the initial critic interpretation explicit.
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, hidden):
        # Keep the value calculation and all return arithmetic in float32.
        with torch.autocast(hidden.device.type, enabled=False):
            return self.projection(hidden.float()).squeeze(-1)


def policy_and_values(model, critic, inputs, *, precision="fp32"):
    """Return aligned policy logits and pre-action values without source edits.

    The hook is installed only for this call, removed even on exceptions, and
    captures rather than changes final_norm's output. Gradients from the value
    head update the shared transformer when this call is not under no_grad.
    """
    hidden = []
    handle = model.final_norm.register_forward_hook(lambda module, args, output: hidden.append(output))
    try:
        with L.amp(model, precision):
            logits = model(inputs)
        if len(hidden) != 1:
            raise RuntimeError("Expected exactly one final_norm call")
        values = critic(hidden[0]) if critic is not None else None
    finally:
        handle.remove()
    return logits.float(), values


def generalized_advantage_estimate(rewards: Sequence[float] | torch.Tensor,
                                   old_values: Sequence[float] | torch.Tensor,
                                   *, gamma=1.0, gae_lambda=0.95,
                                   terminated=True, bootstrap_value=0.0):
    """Detached GAE and lambda returns for one contiguous response sequence.

    rewards[t] follows action t; old_values[t] is V(state before action t).
    A nonterminal rollout segment requires the value of its final successor.
    An actual terminal MUST bootstrap zero, including finite-budget failures.
    """
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    rewards = torch.as_tensor(rewards, dtype=torch.float32).detach()
    values = torch.as_tensor(old_values, dtype=torch.float32, device=rewards.device).detach()
    if rewards.ndim != 1 or rewards.shape != values.shape or not len(rewards):
        raise ValueError("Rewards and pre-action values need the same nonempty one-dimensional shape")
    bootstrap = float(bootstrap_value)
    if terminated and bootstrap != 0:
        raise ValueError("A terminal episode cannot have a nonzero bootstrap")
    if not torch.isfinite(rewards).all() or not torch.isfinite(values).all() or not math.isfinite(bootstrap):
        raise FloatingPointError("Nonfinite reward, value or bootstrap")
    advantages = torch.empty_like(values)
    following_value = values.new_tensor(bootstrap)
    following_advantage = values.new_zeros(())
    for t in range(len(values) - 1, -1, -1):
        delta = rewards[t] + gamma * following_value - values[t]
        following_advantage = delta + gamma * gae_lambda * following_advantage
        advantages[t] = following_advantage
        following_value = values[t]
    return advantages.detach(), (advantages + values).detach()


def relative_outcome_advantages(rewards: Sequence[float]):
    """Leave-one-out group baseline, with no per-group variance division."""
    if len(rewards) < 2:
        raise ValueError("Relative outcome needs at least two independent attempts per group")
    total = sum(rewards)
    return [float(r - (total - r) / (len(rewards) - 1)) for r in rewards]


@torch.no_grad()
def audit_and_attach_old_values(model, critic, tok, rows, *, temperature=0.8,
                                microbatch=8, precision="fp32", tolerance=0.001,
                                ledger=None):
    """Audit actual sampled behavior probabilities before any weight update."""
    if temperature <= 0 or not rows:
        raise ValueError("A nonempty stochastic rollout is required")
    model.eval()
    if critic is not None:
        critic.eval()
    largest = total = 0.0
    count = 0
    for start in range(0, len(rows), microbatch):
        chunk = rows[start:start + microbatch]
        batch = L.collate(chunk, tok.pad_id, next(model.parameters()).device)
        logits, values = policy_and_values(model, critic, batch["inputs"], precision=precision)
        log_probs = (logits / temperature).log_softmax(-1)
        selected = log_probs.gather(-1, batch["targets"][..., None]).squeeze(-1)
        error = (selected - batch["old_logprobs"]).abs()[batch["mask"]]
        largest = max(largest, float(error.max()))
        total += float(error.sum())
        count += error.numel()
        for i, row in enumerate(chunk):
            if values is not None:
                row["old_values"] = values[i][batch["mask"][i]].float().cpu().tolist()
            else:
                row["old_values"] = [0.0] * len(row["response_ids"])
        if ledger is not None:
            L.account(ledger, model, batch, "behavior_audit")
            if critic is not None:
                ledger["behavior_audit"]["flop_proxy"] += 2 * (model.config.d_model + 1) * batch["inputs"].numel()
    result = {"max_abs_error": largest, "mean_abs_error": total / count,
              "n_action_tokens": count, "tolerance": tolerance}
    if largest > tolerance:
        raise RuntimeError(f"Behavior/teacher likelihood error {largest:.6g} > {tolerance}; no update applied")
    return result


def attach_advantages(rows, *, objective, group_size, gamma=1.0, gae_lambda=0.95,
                      normalization="none"):
    """Store inspectable pre-update reward, bootstrap, advantage and return arrays."""
    if objective not in ("relative", "ppo") or len(rows) % group_size:
        raise ValueError("Invalid objective or incomplete rollout group")
    if normalization not in ("none", "std", "center_scale"):
        raise ValueError("Unknown advantage normalization")
    for start in range(0, len(rows), group_size):
        group = rows[start:start + group_size]
        relative = relative_outcome_advantages([r["reward"] for r in group]) if objective == "relative" else None
        for j, row in enumerate(group):
            n = len(row["response_ids"])
            if not n or row["finish_reason"] not in ("qed", "token_limit"):
                raise ValueError("Training requires completed finite-budget episodes")
            if row["finish_reason"] == "token_limit" and row["reward"] != 0:
                raise ValueError("Budget exhaustion must be terminal failure")
            token_rewards = [0.0] * (n - 1) + [float(row["reward"])]
            old = row["old_values"]
            if len(old) != n:
                raise ValueError("Every action needs one aligned old value")
            if objective == "ppo":
                advantages, returns = generalized_advantage_estimate(
                    token_rewards, old, gamma=gamma, gae_lambda=gae_lambda,
                    terminated=True, bootstrap_value=0.0)
                raw = advantages.tolist()
                returns = returns.tolist()
            else:
                raw = [relative[j]] * n
                # Audit-only undiscounted terminal outcomes; no relative critic loss.
                returns = [float(row["reward"])] * n
            row.update(token_rewards=token_rewards, advantages_raw=raw,
                       returns=returns, terminated=True, truncated=False,
                       bootstrap_value=0.0, terminal_reason=row["finish_reason"])
    flattened = torch.tensor([x for r in rows for x in r["advantages_raw"]], dtype=torch.float64)
    mean = float(flattened.mean())
    std = float(flattened.std(unbiased=False))
    # Zero/tiny variance should never divide signal by epsilon and amplify it.
    denominator = std if std > 1e-8 else 1.0
    for row in rows:
        row["advantages"] = [float((a - (mean if normalization == "center_scale" else 0.0)) /
                                   (denominator if normalization != "none" else 1.0))
                             for a in row["advantages_raw"]]
    return {"normalization": normalization, "raw_token_mean": mean, "raw_token_std": std,
            "normalization_scale": denominator if normalization != "none" else 1.0,
            "active_advantage_tokens": sum(a != 0 for r in rows for a in r["advantages"]),
            "action_tokens": flattened.numel()}


def collate_training(rows, tok, device):
    batch = L.collate(rows, tok.pad_id, device)
    for key in ("old_values", "advantages", "returns"):
        values = torch.zeros(batch["mask"].shape, device=device, dtype=torch.float32)
        for i, row in enumerate(rows):
            expected = len(row["response_ids"])
            if len(row[key]) != expected:
                raise ValueError(f"Unaligned {key}")
            values[i, batch["mask"][i]] = torch.tensor(row[key], device=device, dtype=torch.float32)
        batch[key] = values.detach()
    return batch


def ppo_terms(chosen_logprobs, old_logprobs, values, old_values, advantages,
              returns, mask, *, clip=0.2, value_clip=None):
    """Unreduced clipped actor and critic terms; supplied/padded positions zero."""
    if not 0 < clip < 1:
        raise ValueError("Policy clipping must lie in (0, 1)")
    # Replacing off-mask inputs before exp also excludes malformed padded old
    # probabilities from numerical effects. Clipping does not alter denominators.
    delta = torch.where(mask, chosen_logprobs - old_logprobs.detach(), 0.0)
    ratios = delta.exp()
    if not torch.isfinite(ratios).all():
        raise FloatingPointError("Nonfinite importance ratios")
    advantages = advantages.detach()
    surrogate = torch.minimum(ratios * advantages,
                              ratios.clamp(1.0 - clip, 1.0 + clip) * advantages)
    actor = torch.where(mask, -surrogate, 0.0)
    if values is None:
        critic = torch.zeros_like(actor)
    else:
        squared = (values - returns.detach()).square()
        if value_clip is not None:
            if value_clip <= 0:
                raise ValueError("value_clip must be positive or None")
            clipped = old_values.detach() + (values - old_values.detach()).clamp(-value_clip, value_clip)
            squared = torch.maximum(squared, (clipped - returns.detach()).square())
        critic = torch.where(mask, 0.5 * squared, 0.0)
    return actor, critic, ratios


def update_policy(model, critic, tok, rows, optimizer, retention, *, objective,
                  temperature=0.8, max_new_tokens=256, microbatch=8,
                  retention_weight=0.1, precision="fp32", ledger=None,
                  clip=0.2, value_clip=None, value_coef=0.5, entropy_coef=0.0,
                  epochs=1, loss_normalization="fixed_budget", grad_clip=1.0):
    """Update only from frozen fresh behavior rows; retain the same rows per epoch.

    `fixed_budget` divides token sums by N_episodes * configured response budget.
    It avoids making variable proof length an implicit arm-specific rescaling.
    `action_tokens` is an explicit alternative, recorded in the run protocol.
    Retention is mean-example token NLL at temperature 1, for both objectives.
    """
    if objective not in ("relative", "ppo") or (objective == "ppo" and critic is None):
        raise ValueError("PPO requires its learned critic")
    if epochs < 1 or temperature <= 0 or not rows or microbatch < 1:
        raise ValueError("Invalid optimizer batch")
    if min(retention_weight, value_coef, entropy_coef) < 0:
        raise ValueError("Loss coefficients must be nonnegative")
    if loss_normalization == "fixed_budget":
        denominator = len(rows) * max_new_tokens
    elif loss_normalization == "action_tokens":
        denominator = sum(len(r["response_ids"]) for r in rows)
    else:
        raise ValueError("Unknown loss normalization")
    if any(len(r["response_ids"]) > max_new_tokens for r in rows):
        raise ValueError("Response exceeds declared fixed token budget")
    parameters = list(model.parameters()) + (list(critic.parameters()) if objective == "ppo" else [])
    model.eval()  # Deliberately disable dropout in behavior and update alike.
    if critic is not None:
        critic.eval()
    actor_signal = any(a != 0 for r in rows for a in r["advantages"])
    critic_signal = objective == "ppo" and value_coef > 0 and any(
        target != old for r in rows for target, old in zip(r["returns"], r["old_values"]))
    if not (actor_signal or critic_signal or entropy_coef or (retention and retention_weight)):
        # Adam momentum would otherwise move the model on a zero-signal batch.
        optimizer.zero_grad(set_to_none=True)
        return {"optimizer_steps": 0, "skipped_zero_signal": True, "epochs": [],
                "loss_denominator": denominator, "loss_normalization": loss_normalization}
    records = []
    device = next(model.parameters()).device
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        actor_loss = critic_loss = entropy = kl_sum = clipped_sum = action_tokens = 0.0
        ratio_min, ratio_max = math.inf, -math.inf
        for start in range(0, len(rows), microbatch):
            chunk = rows[start:start + microbatch]
            batch = collate_training(chunk, tok, device)
            logits, values = policy_and_values(model, critic if objective == "ppo" else None,
                                               batch["inputs"], precision=precision)
            all_logprobs = (logits / temperature).log_softmax(-1)
            selected = all_logprobs.gather(-1, batch["targets"][..., None]).squeeze(-1)
            actor, value, ratios = ppo_terms(selected, batch["old_logprobs"], values,
                                            batch["old_values"], batch["advantages"],
                                            batch["returns"], batch["mask"],
                                            clip=clip, value_clip=value_clip)
            entropies = -(all_logprobs.exp() * all_logprobs).sum(-1)
            entropy_sum = entropies[batch["mask"]].sum()
            loss = (actor.sum() + (value_coef * value.sum() if objective == "ppo" else 0.0)
                    - entropy_coef * entropy_sum) / denominator
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss")
            loss.backward()
            actor_loss += float(actor.detach().sum()) / denominator
            critic_loss += float(value.detach().sum()) / denominator
            entropy += float(entropy_sum.detach()) / denominator
            with torch.no_grad():
                rs = ratios[batch["mask"]]
                log_ratio = (selected - batch["old_logprobs"])[batch["mask"]]
                # Nonnegative k3 estimator, averaged per actual response token.
                kl_sum += float((rs - 1.0 - log_ratio).sum())
                clipped_sum += int(((rs - 1.0).abs() > clip).sum())
                action_tokens += rs.numel()
                ratio_min = min(ratio_min, float(rs.min()))
                ratio_max = max(ratio_max, float(rs.max()))
            if ledger is not None:
                L.account(ledger, model, batch, "policy_training", True)
                if objective == "ppo":
                    ledger["policy_training"]["flop_proxy"] += 6 * (model.config.d_model + 1) * batch["inputs"].numel()
        retention_loss = 0.0
        if retention and retention_weight:
            for start in range(0, len(retention), microbatch):
                chunk = retention[start:start + microbatch]
                batch = L.collate(chunk, tok.pad_id, device)
                chosen = L.logprobs(model, batch, precision=precision)
                per_example = -(chosen * batch["mask"]).sum(-1) / batch["mask"].sum(-1)
                loss = per_example.sum() / len(retention)
                (retention_weight * loss).backward()
                retention_loss += float(loss.detach())
                if ledger is not None:
                    L.account(ledger, model, batch, "retention_training", True)
        norm = nn.utils.clip_grad_norm_(parameters, grad_clip)
        if not torch.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Nonfinite gradient; optimizer not advanced")
        optimizer.step()
        records.append({"epoch": epoch, "policy_loss": actor_loss, "value_loss": critic_loss,
                        "entropy_fixed_normalization": entropy, "retention_loss": retention_loss,
                        "grad_norm": float(norm), "approx_kl_per_action": kl_sum / action_tokens,
                        "clip_fraction": clipped_sum / action_tokens,
                        "ratio_min": ratio_min, "ratio_max": ratio_max})
    return {"optimizer_steps": epochs, "skipped_zero_signal": False, "epochs": records,
            "loss_denominator": denominator, "loss_normalization": loss_normalization}
