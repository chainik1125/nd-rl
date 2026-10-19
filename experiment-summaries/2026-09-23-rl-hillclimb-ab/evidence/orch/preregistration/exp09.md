# exp09 — faster sampler inside rl.py (drop finished rows from the batch) (prereg 2026-09-23 ~02:05 UTC)

**Question.** Sampling is ~65 % of each round. lib.sample.generate keeps decoding every row of a 384-row batch until
the longest finishes and syncs with the host every step. `fast_generate` in rl.py (same model, same temperature
sampling, EOS rows compacted out of the batch and KV cache every 8 steps once < 80 % remain; no per-step sync)
measured 1.68x faster on pretrain.pt (bench_gen.py, 1,536 samples, run during exp07's scoring: 195 -> 327
samples/s; verified 39 vs 45, mean length 128.2 vs 128.3 words: same distribution). Does ~1.6x more sampling per
second turn into more rounds and a higher score?

**Change vs incumbent (seed rl.py).** `variants/exp09.py`: sample_round uses fast_generate instead of
lib.sample.generate; rounds cap 4 -> 8 (clock-limited). Nothing else.

**Expected.** round-1 sampling <= 75 s (exp01: 109 s); 4 full rounds; dev = 275 (range 245–320).
**Falsified** if round-1 sampling > 90 s or dev <= 250.

**Seeds.** 0, then seed 1 on the next free GPU if dev >= 265. **Stop rule.** One run per seed; reruns only on crash.
