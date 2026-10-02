# exp05 — sampling batch 384 -> 1024 (prereg 2026-09-23 ~01:34 UTC)

**Question.** Sampling is ~65 % of each round (exp01: 110 s of ~175 s) at 441 samples/s with batch 384. The model is
3.2M params; decoding is launch-bound, so a larger batch should raise throughput. rl.py samples and trains
sequentially in one process (generate() empties the cache), so the "1024 does not fit next to training" note
should not bite. Does batch 1024 raise throughput enough to fit a 4th round, and does that raise the score?

**Change vs incumbent (seed rl.py).** `variants/exp05.py`: sample_batch 384 -> 1024; rounds cap 4 -> 8 (still
clock-limited). Nothing else.

**Expected.** samples_per_sec >= 700 in round 1; 4 rounds complete; dev = 280 (range 245–330) vs control 249/238.
**Falsified** if throughput < 550/s (then batching is not the bottleneck) or dev <= 250. OOM -> crash, logged.

**Seeds.** 0. **Stop rule.** One run; reruns only on crash (then with 768).
