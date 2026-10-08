# Autoresearch loop for test-time thoughts

Started 2026-10-07 (agent:claude, at Robbie's request) after the
[Coconut-style spike](../coconut_spike_20261006/README.md) found that continuous thoughts help
the pretrained model and stop mattering after RL. Rules in [program.md](program.md); how to run it
in [LOOP.md](LOOP.md).

This is the test-time-loops loop (`autoresearch_loops/` on branch `autoresearch-loops`, commit
`bee8868`) with a different recipe. The two share one harness: `harness.py` here differs from
that commit in two constants and nothing else.

| | loops loop | this loop |
|---|---|---|
| `KNOB` | `n_loops`: passes through the shared core | `n_thoughts`: continuous thoughts per proof line |
| `EVAL_LEVEL` / `DIAG_LEVEL` | 8 / 2 | 8 / 0 |
| starting recipe | looped trunk, 6.4M parameters | run 198's six untied blocks plus a thinking round before each line, 9.56M |
| ids, branch | from 500, `autoresearch-loops` | from 700, `autoresearch-thoughts` |

Everything else is the same file by file: 300 s of pretraining, 1,800 s of agent-owned RL through
the sampling oracle with Lean-gate time not charged, the pools, 64 samples per dev theorem, the
10M parameter cap, three seeds per idea, and the keep rule. The other scripts are the loops
loop's with folder, pod and artifact names changed and the schema's `loops` block replaced by a
`thoughts` block. `check_model.py` verifies that the recipe's sampling and training paths agree.

## State: paused 2026-10-07 20:02 PT

Stopped by the launcher's balance check (RunPod $4.08, floor $5) after 700 to 708. Incumbent: 708
at 711. Summary and what was learned:
[experiment-summaries/2026-10-07-test-time-thoughts](../../../../experiment-summaries/2026-10-07-test-time-thoughts/README.md).
Checkouts: `~/repos/nd-rl-12` (operator) and `~/repos/nd-rl-14` (the Fable agent's). `pod.py` here
also replaces pods whose GPU PyTorch cannot use, which the pretraining loop's copy does not.
