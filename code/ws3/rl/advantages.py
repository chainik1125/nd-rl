"""Advantage transforms for binary-reward GRPO (brainstorm sections K2, K4, K5).

Each transform maps the rewards of one generation batch, shape (N,) with N = groups x G and the
G samples of a prompt contiguous (TRL's layout), to per-sample advantages of shape (N,). They are
pure functions of tensors so they can be unit-tested without a model; `WS3GRPOTrainer`
(`rl/trainer.py`) swaps TRL's advantages for one of these.

    trl               keep TRL's own advantage (loss_type / scale_rewards)
    maxrl             (r - mean) / mean, the gradient of log pass rate
                      (MaxRL, ICML 2026, arXiv 2602.02710)
    w_reinforce       +lam for a success, -1 for a failure, no baseline
                      (NSR / W-REINFORCE, NeurIPS 2025, arXiv 2506.01347)
    virtual_positive  the baseline counts one imagined success, so all-fail groups get a negative
                      advantage (NGRPO, arXiv 2509.18851; AVSPO, ICML 2026)
    pass_at_k         exact per-sample advantage for the group's pass@k
                      (Pass@k Training, arXiv 2508.10751; PKPO, NeurIPS 2025)
    unlikely          success reward shrinks for likelier correct proofs, then a mean-centred
                      group advantage (Rewarding the Unlikely, EMNLP 2025)

All but `trl` read success as reward > 0.5, so they need the binary reward; `run.py` refuses a
shaped reward with them. None of them divide by the group std unless `std_scale: true`.
"""

from __future__ import annotations

import math

import torch

NAMES = ("trl", "maxrl", "w_reinforce", "virtual_positive", "pass_at_k", "unlikely")


def _groups(x: torch.Tensor, g: int) -> torch.Tensor:
    if x.numel() % g:
        raise ValueError(f"{x.numel()} rewards do not split into groups of {g}")
    return x.view(-1, g)


def success(rewards: torch.Tensor) -> torch.Tensor:
    return (rewards > 0.5).to(torch.float32)


def _std_scale(adv: torch.Tensor, ref: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Divide each group's advantages by the std of `ref` (the rewards) in that group."""
    return adv / (ref.std(dim=1, unbiased=False, keepdim=True) + eps)


def maxrl(rewards, g, std_scale=False, eps=1e-6):
    """A = (r - mean) / mean per group; 0 for an all-fail group.

    For a binary reward this is the gradient of log(pass rate) rather than of pass rate, so a
    prompt at 1/64 gets 64x the weight of one at 1. That puts the update on the hard prompts.
    """
    s = _groups(success(rewards), g)
    mean = s.mean(dim=1, keepdim=True)
    adv = torch.where(mean > 0, (s - mean) / (mean + eps), torch.zeros_like(s))
    if std_scale:
        adv = _std_scale(adv, s)
    return adv.reshape(-1)


def w_reinforce(rewards, g, lam=0.1):
    """REINFORCE with +lam on successes and -1 on failures, no baseline.

    lam = 0 is pure negative-sample reinforcement (NSR). Every failed sample gets gradient, so
    all-fail groups are not wasted. The paper's default is lam = 0.1.
    """
    s = success(rewards)
    _groups(s, g)  # shape check
    return lam * s - (1.0 - s)


def virtual_positive(rewards, g, std_scale=False):
    """Mean-centre against a baseline that includes one imagined success.

    With c successes out of G the baseline is (c + 1) / (G + 1), so an all-fail group gets
    A = -1 / (G + 1) on every sample instead of 0. With `std_scale` the std also includes
    the imagined sample.
    """
    s = _groups(success(rewards), g)
    base = (s.sum(dim=1, keepdim=True) + 1.0) / (g + 1)
    adv = s - base
    if std_scale:
        full = torch.cat([s, torch.ones_like(s[:, :1])], dim=1)
        adv = adv / (full.std(dim=1, unbiased=False, keepdim=True) + 1e-4)
    return adv.reshape(-1)


def pass_at_k(rewards, g, k=4, std_scale=True):
    """Per-sample advantage for maximising pass@k of the group (binary reward).

    Split the G samples into all k-subsets; a subset scores 1 if any member succeeds. The
    group's mean score is rho = 1 - C(G - c, k) / C(G, k). A success is in only passing subsets,
    so its expected score is 1; a failure's is 1 - C(G - c - 1, k - 1) / C(G - 1, k - 1). The
    advantage is expected score minus rho, divided by sqrt(rho (1 - rho)) when `std_scale`.
    k = 1 gives plain mean-centred GRPO.
    """
    if not 1 <= k <= g:
        raise ValueError(f"pass_at_k needs 1 <= k <= G, got k={k}, G={g}")
    s = _groups(success(rewards), g)
    out = torch.zeros_like(s)
    for i, row in enumerate(s):
        c = int(row.sum().item())
        neg = g - c
        rho = 1.0 - (math.comb(neg, k) / math.comb(g, k))
        a_pos = 1.0 - rho
        a_neg = (
            (1.0 - math.comb(neg - 1, k - 1) / math.comb(g - 1, k - 1) - rho) if neg > 0 else 0.0
        )
        if std_scale:
            sd = math.sqrt(rho * (1.0 - rho))
            if sd < 1e-8:  # all pass or all fail: no signal
                a_pos = a_neg = 0.0
            else:
                a_pos, a_neg = a_pos / sd, a_neg / sd
        out[i] = torch.where(row > 0.5, torch.tensor(a_pos), torch.tensor(a_neg))
    return out.reshape(-1)


def unlikely_rewards(rewards, g, seq_logps, beta=0.25):
    """Reshaped rewards: successes ranked by sequence log-prob within their group.

    The least likely success keeps reward 1; the most likely of c successes gets
    1 - beta * (c - 1) / c. Failures stay 0. This counters GRPO's tendency to pile probability
    onto one proof per prompt (Rewarding the Unlikely, tested on formal theorem proving).
    """
    s = _groups(success(rewards), g)
    lp = _groups(seq_logps.to(torch.float32), g)
    out = s.clone()
    for i in range(s.shape[0]):
        idx = torch.nonzero(s[i] > 0.5).flatten()
        c = idx.numel()
        if c < 2:
            continue
        order = torch.argsort(lp[i, idx])  # ascending: least likely first
        for rank, j in enumerate(order.tolist()):  # rank 0 = least likely
            out[i, idx[j]] = 1.0 - beta * rank / c
    return out.reshape(-1)


def unlikely(rewards, g, seq_logps, beta=0.25, std_scale=False):
    r = _groups(unlikely_rewards(rewards, g, seq_logps, beta), g)
    adv = r - r.mean(dim=1, keepdim=True)
    if std_scale:
        adv = _std_scale(adv, r)
    return adv.reshape(-1)


def compute(name, rewards, g, params=None, seq_logps=None):
    """Dispatch by name. `params` holds the transform's keyword arguments from the config."""
    params = dict(params or {})
    rewards = torch.nan_to_num(rewards.to(torch.float32), nan=0.0)
    if name == "maxrl":
        return maxrl(rewards, g, **params)
    if name == "w_reinforce":
        return w_reinforce(rewards, g, **params)
    if name == "virtual_positive":
        return virtual_positive(rewards, g, **params)
    if name == "pass_at_k":
        return pass_at_k(rewards, g, **params)
    if name == "unlikely":
        if seq_logps is None:
            raise ValueError("unlikely needs sequence log-probs")
        return unlikely(rewards, g, seq_logps, **params)
    raise ValueError(f"unknown advantage transform {name!r}; choose from {NAMES}")


def group_outcomes(rewards, g) -> dict:
    """Fractions of groups that are all-fail, all-success, and mixed (TRL's
    frac_reward_zero_std lumps the first two together)."""
    s = _groups(success(torch.nan_to_num(rewards.to(torch.float32), nan=0.0)), g)
    c = s.sum(dim=1)
    n = max(s.shape[0], 1)
    return {
        "groups/frac_all_fail": float((c == 0).sum()) / n,
        "groups/frac_all_success": float((c == g).sum()) / n,
        "groups/frac_mixed": float(((c > 0) & (c < g)).sum()) / n,
    }
