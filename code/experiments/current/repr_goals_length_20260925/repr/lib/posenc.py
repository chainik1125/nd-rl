"""Training-time positions for the representation variants (ab3/repr). Pure helpers; no model compute.

  record_positions(model, ids, rng) -> list    one training sequence (prompt + proof ids, from its start):
        pos_train='rand': sorted(rng.sample(range(pos_range), len(ids)))   (Ruoss et al. 2023, per sequence)
        otherwise:        model.seq_positions(ids)   (0, 1, 2, ...; [line, off] pairs for pe='line')
  batch_positions(model, x, lens, gen) -> tensor or None   a right-padded (B, T) batch on the GPU:
        pos_train='rand': per row a sorted uniform random subset of range(pos_range) of size lens[i] in the
                          first lens[i] columns (padding columns get positions >= pos_range; they are masked
                          out of the loss and, being after the sequence, never attended to by it)
        otherwise:        None (the model's default positions: arange, or line positions from the tokens)
  needs_positions(model) -> bool    True when training must pass explicit positions (pos_train='rand')
"""
import torch


def needs_positions(model):
    return getattr(model, 'pos_train', 'arange') == 'rand'


def record_positions(model, ids, rng):
    if needs_positions(model):
        R = model.pos_range
        if len(ids) > R:
            raise ValueError(f'sequence of {len(ids)} tokens is longer than pos_range {R}')
        return sorted(rng.sample(range(R), len(ids)))
    return model.seq_positions(ids)


def batch_positions(model, x, lens, gen=None):
    if not needs_positions(model):
        return None
    B, T = x.shape
    R = model.pos_range
    if T > R:
        raise ValueError(f'batch width {T} is longer than pos_range {R}')
    keys = torch.rand(B, R, device=x.device, generator=gen)
    perm = keys.argsort(1)[:, :T]                                  # a uniform random T-subset per row, unsorted
    j = torch.arange(T, device=x.device)[None, :]
    lens = torch.as_tensor(lens, device=x.device)[:, None]
    perm = torch.where(j < lens, perm, R + j)                      # keep the first lens[i] draws; pads after them
    return perm.sort(1).values
