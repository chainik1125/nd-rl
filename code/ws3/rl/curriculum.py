"""Per-prompt success tracking and an adaptive prompt sampler (brainstorm sections E and K3).

`PromptStats` keeps, for every RL prompt, how its recent groups went. `AdaptiveRepeatSampler`
replaces TRL's `RepeatSampler`. It keeps the same index layout: each prompt repeated
`mini_repeat_count` times back to back, each batch repeated `repeat_count` times. But it picks
each batch's prompts from weights computed from the stats when that batch is drawn. The
DataLoader draws about one generation batch ahead, so the weights lag the stats by at most one
batch.

Sampler options (the `train.curriculum` block of the YAML; all optional):

    weighting: uniform | p1mp    p1mp: weight p(1-p) + floor, from a Beta(a, b) estimate over the
                                 last `window` groups. Unseen prompts get p = a / (a + b), 0.5 by
                                 default, so they are tried early (Online Difficulty Filtering)
    floor: 0.01                  added to every p1mp weight, so no allowed prompt reaches 0
    window: 4                    groups per prompt that the estimate uses
    prior: [0.5, 0.5]            Beta prior (a, b)
    skip_after: null | m         GRESO: after m same-outcome groups in a row (all fail or all
    skip_keep: 0.1               succeed), multiply the weight by skip_keep. Not 0, so skipped
                                 prompts are still retried now and then
    drop_solved_after: null | m  weight 0 after m all-success groups in a row
    ladder: {start, threshold, min_groups}
                                 length ladder: only lengths <= the current max are allowed. The
                                 max goes up by one when the recent pass rate at the current max,
                                 pooled over prompts seen at least once there, reaches `threshold`
                                 with at least `min_groups` recent groups. Lengths that are not
                                 in the pool are skipped
    easy_frac: 0.0, easy_max: 6  this fraction of each batch comes uniformly from lengths
                                 <= easy_max, to guard against forgetting

Only prompts already in the training dataset are ever sampled. The pool rules
(`data.build_prompt_set`, which runs the leak check) are unchanged.
"""

from __future__ import annotations

import collections
import random

from torch.utils.data import Sampler


class PromptStats:
    def __init__(self, window: int = 4):
        self.window = window
        self.recent = collections.defaultdict(lambda: collections.deque(maxlen=self.window))
        self.total = collections.defaultdict(lambda: [0, 0])  # prompt -> [samples, successes]
        self.same_streak = collections.Counter()  # consecutive groups with one outcome
        self.success_streak = collections.Counter()  # consecutive all-success groups
        self.n_groups = 0
        self.extra: dict[str, float] = {}  # the sampler writes its state here for logging

    def update(self, prompts, success_rows):
        """One entry per group: the prompt text and its G success indicators (0/1)."""
        for p, row in zip(prompts, success_rows, strict=True):
            g, c = len(row), int(sum(row))
            self.recent[p].append((g, c))
            self.total[p][0] += g
            self.total[p][1] += c
            self.same_streak[p] = self.same_streak[p] + 1 if c in (0, g) else 0
            self.success_streak[p] = self.success_streak[p] + 1 if c == g else 0
            self.n_groups += 1

    def recent_counts(self, p):
        r = self.recent.get(p)
        if not r:
            return 0, 0
        return sum(g for g, _ in r), sum(c for _, c in r)

    def p_hat(self, p, prior=(0.5, 0.5)):
        n, c = self.recent_counts(p)
        a, b = prior
        return (c + a) / (n + a + b)

    def summary(self) -> dict:
        seen = [p for p in self.total if self.total[p][0]]
        out = {
            "prompts/seen": float(len(seen)),
            "prompts/ever_solved": float(sum(self.total[p][1] > 0 for p in seen)),
        }
        out.update(self.extra)
        return out


class AdaptiveRepeatSampler(Sampler):
    def __init__(
        self,
        prompts,
        lengths,
        stats: PromptStats,
        mini_repeat_count: int,
        batch_size: int,
        repeat_count: int = 1,
        seed: int = 0,
        weighting: str = "p1mp",
        floor: float = 0.01,
        window: int | None = None,
        prior=(0.5, 0.5),
        skip_after: int | None = None,
        skip_keep: float = 0.1,
        drop_solved_after: int | None = None,
        ladder: dict | None = None,
        easy_frac: float = 0.0,
        easy_max: int = 6,
    ):
        if weighting not in ("uniform", "p1mp"):
            raise ValueError(f"weighting must be uniform or p1mp, got {weighting!r}")
        if len(prompts) < batch_size:
            raise ValueError(f"{len(prompts)} prompts < {batch_size} prompts per batch")
        self.prompts, self.lengths, self.stats = list(prompts), [int(L) for L in lengths], stats
        if window is not None:
            stats.window = window
        self.m, self.bs, self.rc = mini_repeat_count, batch_size, repeat_count
        self.rng = random.Random(seed)
        self.weighting, self.floor, self.prior = weighting, floor, tuple(prior)
        self.skip_after, self.skip_keep = skip_after, skip_keep
        self.drop_solved_after = drop_solved_after
        self.ladder = dict(ladder) if ladder else None
        self.pool_lengths = sorted(set(self.lengths))
        if self.ladder:
            self.max_len = int(self.ladder["start"])
        else:
            self.max_len = max(self.pool_lengths)
        self.easy_frac, self.easy_max = easy_frac, easy_max
        self.n_batches = len(self.prompts) // batch_size

    def __len__(self):
        return self.n_batches * self.bs * self.m * self.rc

    # ---------------------------------------------------------------- weights

    def _advance_ladder(self):
        if not self.ladder:
            return
        higher = [L for L in self.pool_lengths if self.max_len < L]
        if not higher:
            return
        n = c = 0
        for p, L in zip(self.prompts, self.lengths, strict=True):
            if self.max_len == L:
                pn, pc = self.stats.recent_counts(p)
                n, c = n + pn, c + pc
        groups = n / max(self.m, 1)
        if (
            groups >= self.ladder.get("min_groups", 32)
            and c / max(n, 1) >= self.ladder["threshold"]
        ):
            self.max_len = higher[0]

    def weight(self, i: int) -> float:
        p, L = self.prompts[i], self.lengths[i]
        if self.max_len < L:
            return 0.0
        st = self.stats
        if self.drop_solved_after and st.success_streak[p] >= self.drop_solved_after:
            return 0.0
        if self.weighting == "uniform":
            w = 1.0
        else:
            ph = st.p_hat(p, self.prior)
            w = ph * (1.0 - ph) + self.floor
        if self.skip_after and st.same_streak[p] >= self.skip_after:
            w *= self.skip_keep
        return w

    def _draw(self, idx, weights, k, taken):
        """Up to k distinct indices from idx by weight, skipping `taken` and zero weights."""
        cand = [(i, w) for i, w in zip(idx, weights, strict=True) if i not in taken and w > 0]
        out = []
        while cand and len(out) < k:
            tot = sum(w for _, w in cand)
            r, acc = self.rng.random() * tot, 0.0
            for j, (i, w) in enumerate(cand):
                acc += w
                if acc >= r:
                    out.append(i)
                    cand.pop(j)
                    break
        return out

    def next_batch(self):
        self._advance_ladder()
        all_idx = list(range(len(self.prompts)))
        taken: set = set()
        n_easy = round(self.easy_frac * self.bs)
        batch = []
        if n_easy:
            easy = [i for i in all_idx if self.lengths[i] <= self.easy_max]
            batch += self._draw(easy, [1.0] * len(easy), n_easy, taken)
            taken |= set(batch)
        weights = [self.weight(i) for i in all_idx]
        main = self._draw(all_idx, weights, self.bs - len(batch), taken)
        batch += main
        live = [w for w in weights if w > 0]
        if len(batch) < self.bs:
            # Fewer allowed prompts than the batch needs: repeat allowed ones (two groups of one
            # prompt are fine for GRPO) rather than break the ladder or drop rule. If every weight
            # is 0 (all dropped as solved), fall back to the lengths the ladder has unlocked.
            allowed = [i for i, w in zip(all_idx, weights, strict=True) if w > 0] or [
                i for i in all_idx if self.lengths[i] <= self.max_len
            ]
            batch += self.rng.choices(allowed, k=self.bs - len(batch))
        self.stats.extra = {
            "curriculum/max_length": float(self.max_len),
            "curriculum/allowed_prompts": float(len(live)),
            "curriculum/mean_len": sum(self.lengths[i] for i in batch) / len(batch),
        }
        return batch

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = self.next_batch()
            for _ in range(self.rc):
                for i in batch:
                    for _ in range(self.m):
                        yield i
