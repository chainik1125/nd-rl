"""GRPO training-batch dumps for update attribution (`grpo.py --dump_batches DIR`; audit E1).

One file per iteration, `DIR/<name>_it<i>_batch.pt` (`torch.save` of plain types and tensors, so
`torch.load(weights_only=True)` reads it), holding everything needed to replay the iteration's
AdamW steps from the previous iteration's checkpoint and split each step by token class:

* items (one per attempt step that was trained on): the decoder input `prompt`, the sampled ids
  `gen` and the loss `mask` (a forced `</think>` is False), concatenated with offsets
  (`prompt_off`, `gen_off`; uint8 ids when the vocabulary fits, else int16); the advantage `adv`
  (float64, exact); the sampling policy's log-probs `old_logp` of the masked tokens (float32,
  exact; `old_off`); the item's `budget`, `attempt` and `minibatch` (`order[u::updates]` is
  minibatch u, as the trainer split it);
* attempts (every sampled attempt, skipped groups included): `group`, `status`, `ok`, `reward`
  (float64), `masked` (overlong-masked);
* groups: `budget`, `target` (row index into its file), `source` (0 `--targets`, 1
  `--targets_extra`), `skipped`;
* per AdamW step: `step_lr`, `step_grad_norm` (pre-clip), `step_kl_anchor`; and the loss settings;
* `adam`: the optimizer state before the iteration's first step, compact: the step count and, per
  parameter (in `model.named_parameters()` order, `names`), `exp_avg` and `exp_avg_sq` in bfloat16
  (8 exponent bits: the second moments' range survives; about 3 significant digits). The
  full-precision state is in every checkpoint (`ckpt.save_ckpt(train_state=...)`).

`load_batch` reads a file back; `unpack_items` rebuilds the trainer's items and old log-probs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

FORMAT = "grpo-batch/1"
Item = tuple[list[int], list[int], list[bool], float]


def _cat(rows: Sequence[Sequence[int]], dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    off = torch.zeros(len(rows) + 1, dtype=torch.int64)
    if rows:
        off[1:] = torch.cumsum(torch.tensor([len(r) for r in rows], dtype=torch.int64), 0)
    flat = [x for r in rows for x in r]
    return torch.tensor(flat, dtype=dtype), off


def compact_adam(opt: torch.optim.Optimizer, model: torch.nn.Module) -> dict[str, Any]:
    """The optimizer's AdamW moments in bfloat16, by parameter name (none before a first step)."""
    names, m1, m2 = [], [], []
    step = 0
    for n, p in model.named_parameters():
        st = opt.state.get(p)
        if not st:
            continue
        names.append(n)
        m1.append(st["exp_avg"].detach().to("cpu", torch.bfloat16))
        m2.append(st["exp_avg_sq"].detach().to("cpu", torch.bfloat16))
        step = int(st["step"])
    return {"step": step, "names": names, "exp_avg": m1, "exp_avg_sq": m2}


def dump_batch(
    path: str | Path,
    *,
    it: int,
    items: Sequence[Item],
    old: Sequence[torch.Tensor],
    item_attempt: Sequence[int],
    order: Sequence[int],
    updates: int,
    attempts: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, float]],
    adam: Mapping[str, Any],
    settings: Mapping[str, Any],
    vocab: int,
) -> Path:
    """Write one iteration's batch (module docstring). `attempts`: per attempt `group`,
    `status`, `ok`, `reward`, `masked`; `groups`: per group `budget`, `target`, `source`,
    `skipped`; `steps`: per AdamW step `lr`, `grad_norm`, `kl_anchor`."""
    assert len(old) == len(items) == len(item_attempt), "items, old log-probs, attempts"
    ids = torch.uint8 if vocab <= 256 else torch.int16
    mb = [0] * len(items)
    for u in range(updates):
        for k in order[u::updates]:
            mb[k] = u
    prompt, prompt_off = _cat([p for p, _, _, _ in items], ids)
    gen, gen_off = _cat([g for _, g, _, _ in items], ids)
    old_off = torch.zeros(len(items) + 1, dtype=torch.int64)
    if items:
        old_off[1:] = torch.cumsum(torch.tensor([len(o) for o in old], dtype=torch.int64), 0)
    a_group = [int(x["group"]) for x in attempts]
    d = {
        "format": FORMAT,
        "iter": it,
        "prompt": prompt,
        "prompt_off": prompt_off,
        "gen": gen,
        "gen_off": gen_off,
        "mask": torch.tensor([m for _, _, mk, _ in items for m in mk], dtype=torch.bool),
        "adv": torch.tensor([a for *_, a in items], dtype=torch.float64),
        "old_logp": (torch.cat([o.float().cpu() for o in old]) if old else torch.zeros(0)),
        "old_off": old_off,
        "item_attempt": torch.tensor(list(item_attempt), dtype=torch.int32),
        "budget": torch.tensor(
            [int(groups[a_group[k]]["budget"]) for k in item_attempt], dtype=torch.int32
        ),
        "minibatch": torch.tensor(mb, dtype=torch.int16),
        "order": torch.tensor(list(order), dtype=torch.int64),
        "updates": updates,
        "attempt_group": torch.tensor(a_group, dtype=torch.int32),
        "attempt_status": [str(x["status"]) for x in attempts],
        "attempt_ok": torch.tensor([bool(x["ok"]) for x in attempts], dtype=torch.bool),
        "attempt_reward": torch.tensor([float(x["reward"]) for x in attempts], dtype=torch.float64),
        "attempt_masked": torch.tensor([bool(x["masked"]) for x in attempts], dtype=torch.bool),
        "group_budget": torch.tensor([int(g["budget"]) for g in groups], dtype=torch.int32),
        "group_target": torch.tensor([int(g["target"]) for g in groups], dtype=torch.int32),
        "group_source": torch.tensor([int(g["source"]) for g in groups], dtype=torch.int8),
        "group_skipped": torch.tensor([bool(g["skipped"]) for g in groups], dtype=torch.bool),
        "step_lr": [float(s["lr"]) for s in steps],
        "step_grad_norm": [float(s["grad_norm"]) for s in steps],
        "step_kl_anchor": [float(s.get("kl_anchor", 0.0)) for s in steps],
        "settings": dict(settings),
        "adam": dict(adam),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(d, str(tmp))
    tmp.replace(path)
    return path


def load_batch(path: str | Path) -> dict[str, Any]:
    d = torch.load(str(path), map_location="cpu")
    assert d.get("format") == FORMAT, f"{path}: not a {FORMAT} file"
    return d


def unpack_items(d: Mapping[str, Any]) -> tuple[list[Item], list[torch.Tensor]]:
    """A loaded batch -> (the trainer's items: prompt, gen, mask, advantage; old log-probs)."""
    po, go, oo = d["prompt_off"].tolist(), d["gen_off"].tolist(), d["old_off"].tolist()
    prompt, gen, mask = d["prompt"].tolist(), d["gen"].tolist(), d["mask"].tolist()
    adv = d["adv"].tolist()
    items: list[Item] = [
        (prompt[po[k] : po[k + 1]], gen[go[k] : go[k + 1]], mask[go[k] : go[k + 1]], adv[k])
        for k in range(len(adv))
    ]
    old = [d["old_logp"][oo[k] : oo[k + 1]] for k in range(len(adv))]
    return items, old
