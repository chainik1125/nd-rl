# Autoresearch loop for test-time loops

Started 2026-10-07 (agent:claude, at Robbie's request) after the
[looped-transformer spike](../looped_spike_20261006/README.md) found that extra test-time loops
stop helping at about 4. Rules in [program.md](program.md); how to run it in [LOOP.md](LOOP.md).

Forked from [../autoresearch](../autoresearch/README.md), whose harness stays frozen for the
pretraining loop. What differs here:

| | pretraining loop | this loop |
|---|---|---|
| agent edits | `pretrain.py`: `pretrain()` | `pretrain.py`: `pretrain()` and `rl()` |
| RL | harness-owned, 4 rounds, samples at the model's default loops | agent-owned, 1,800 s, through a sampling oracle |
| metric | dev solved (length 7+), 64 samples | the same at `model.n_loops = 8`; also recorded at 2 |
| total budget | 2,400 s | 5,400 s |
| bar | incumbent's 3-seed mean | incumbent's 3-seed mean; target: best spike model (program.md) |

`harness.py` hands `rl()` the target statements and an oracle that returns the model's own
attempts that Lean and nd_verify accept. Lean-gate time is not charged to the RL budget. The other
scripts are the pretraining loop's with paths, pod names and the schema adapted; `pools/` is a
copy. Decisions behind this design (Robbie, 2026-10-07): RL may diverge from the old recipe, the
bar is the best of the spike's looped models and not run 198, three seeds per idea in parallel.
