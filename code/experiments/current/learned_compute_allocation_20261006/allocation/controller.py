"""Base-copy controller for raw proof histories, with an explicit factorized policy.

The caller loads Dan's actual ALiBiGPT checkpoint independently for each run.
No verifier metadata or engineered correctness features enter this module.
The new heads are initialized identically for cost and correctness-only runs.
"""

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical


class Controller(nn.Module):
    def __init__(self, base: nn.Module, max_attempts: int = 8):
        super().__init__()
        self.base = base
        self.max_attempts = max_attempts
        self.accept_head = nn.Linear(base.cfg["d"], 1)
        self.select_head = nn.Linear(base.cfg["d"], max_attempts)
        nn.init.zeros_(self.accept_head.weight)
        nn.init.zeros_(self.accept_head.bias)
        nn.init.zeros_(self.select_head.weight)
        nn.init.zeros_(self.select_head.bias)

    def distributions(self, ids, lengths, counts):
        if torch.any(lengths < 1) or torch.any(lengths > ids.shape[1]):
            raise ValueError("invalid prompt lengths")
        if torch.any(counts < 1) or torch.any(counts > self.max_attempts):
            raise ValueError("invalid candidate counts")
        positions = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
        valid = positions < lengths[:, None]
        mask = valid[:, None, None, :] & (
            positions[:, None, :, None] >= positions[:, None, None, :]
        )
        hidden, _ = self.base.features(ids, pos=positions, mask=mask)
        last = self.base.ln_f(hidden[torch.arange(len(ids), device=ids.device), lengths - 1])
        accept = Bernoulli(logits=self.accept_head(last).squeeze(-1))
        indices = torch.arange(self.max_attempts, device=ids.device)[None]
        logits = self.select_head(last).masked_fill(indices >= counts[:, None], float("-inf"))
        return accept, Categorical(logits=logits)

    @staticmethod
    def action_logprob(accept_dist, index_dist, accepted, index, forced):
        if torch.any(forced & ~accepted.bool()):
            raise ValueError("forced CONTINUE is impossible")
        # At a hard limit ACCEPT has probability one under the masked behavior policy.
        accept_lp = torch.where(forced, torch.zeros_like(accepted), accept_dist.log_prob(accepted))
        return accept_lp + torch.where(
            accepted.bool(), index_dist.log_prob(index), torch.zeros_like(accepted)
        )


def encode_history(tok, theorem_ids, raw_attempts, context_limit=1024):
    """Uses existing token alphabet; n1..n8 label proof blocks, <act> asks for a decision.

    Raw attempt tokens include attempted malformed actions and do not contain
    outcome-dependent prefixes. Names are opaque data within separated blocks.
    There is no clipping, proof filtering, verdict substitution, or history dropping.
    """
    ids = list(theorem_ids)
    for i, candidate in enumerate(raw_attempts):
        ids.extend((tok.stoi["<st>"], tok.stoi[f"n{i + 1}"]))
        ids.extend(candidate)
        ids.append(tok.eos)
    ids.append(tok.stoi["<act>"])
    if len(ids) > context_limit:
        raise ValueError("raw attempt history exceeds controller context capacity")
    return ids
