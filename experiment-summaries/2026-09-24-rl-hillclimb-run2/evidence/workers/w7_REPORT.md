# w7 report (GEAR run 2, ab2), 2026-09-24

**Frontier at 13:55Z:** BEST n078 (two seeds, score 1175 = dev 383 + 4 x frontier 198/1000). It is w2's index-cap mutation of my n071, so my temperature ramp is in the BEST line.

**My best checkpoints:** n094 score 1074 (dev 374, frontier 175, heldout 0.929) `/scratch/cowifornia/nd-rl/ab2/w7/runs/ramp_n086_s0/final.pt`; n071 score 1026 (dev 370, frontier 164, heldout 0.901) `/scratch/cowifornia/nd-rl/ab2/w7/runs/ramp_n053_s0/final.pt` (parent of BEST n078).

**What moved the score:** a POLARIS rising training-sample temperature (https://hkunlp.github.io/blog/2025/Polaris/): base T ramps 1.0 to 1.3 (0.1 steps, cuda-graph state rebuilt per step). Vs parent: n058 +40.5, n071 +167 (+22% cumulative train solves), n064 -241 (16 vs 21 rounds), n079 -72 (code-identical to n078 = 1198; 20 vs 30 rounds), n094 -99. It helps the length-weighted line, not the schema-coverage lines.

**What did not:** NSR on the first invalid line (arXiv 2506.01347, n012: dev 1, syntax collapse); GEM crossover (arXiv 2408.16673, n037, neutral); low-overhead lv sampler (n043, neutral); e2h order (arXiv 2506.06632, n051, -187); line-head temperature on the cuda-graph sampler (arXiv 2510.08892, n087, -146 at 19 vs 26 rounds on slow GPU0, unresolved); head-scaling temperature calibration (n028, ruled out by the monitor).

**Controller steps:** v15 (pick_p2 skips a partner already transplanted into p1's line; release restores the fresh-elite index) and v16 (a fresh elite already paired in a crossover no longer forces another). Both 27/27 tests and installed. I also reported GPU-lock starvation; the monitor made the lock FIFO.

**Largest noise source:** GPU speed. Fine-tune ms/step runs from about 108 (GPU4) to about 177 (GPU0). Identical code and seed gave 1198 (30 rounds) and 1031 (20 rounds).

**Next:** re-run n087 on a fast GPU; put the ramp into the weight-averaged n092 line for the champion run; compare by rounds or pin seeds to GPUs of similar speed; L14 is still about 0 and needs a targeted idea.
